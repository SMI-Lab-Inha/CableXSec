# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Compiled per-wire friction-stress history for the fatigue path.

Reproduces the force recursion of
:func:`cablexsec.mechanics.papailiou.wire_slip_history` (``stress_Pa`` only)
for a constant stick stiffness: the signed Papailiou bounds at the current
bending direction and clamping, the scalar elastic--Coulomb return map at
every wire, and the held/blended bending direction below
``f m(lam) kappa_on`` (see :func:`~cablexsec.mechanics.papailiou.wire_slip_history`).
Work terms
are not formed here; callers needing the moment or the energy balance use the
reference routine. Regression tests compare the two.

A non-neutral :class:`~cablexsec.mechanics.friction_law.FrictionLaw` runs a
second kernel with the per-wire static/dynamic state, stick timer and
rate-dependent return of the reference
(:func:`~cablexsec.mechanics.papailiou.wire_slip_history`), and the pressure
factor of the law at every wire's local pressure; ``None`` or a neutral law
keeps the rate-independent kernel, bit for bit.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from cablexsec._numba import HAVE_NUMBA, njit
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.friction_law import (
    FrictionLaw,
    _pressure_factor,
    _return_stress,
    _slip_magnitude,
    active_friction_law,
    friction_law_time_steps,
)
from cablexsec.mechanics.papailiou import (
    DIRECTION_HOLD_FRACTION,
    _law_threshold_layer,
    _threshold_coefficient,
)

if TYPE_CHECKING:
    from cablexsec.mechanics.mkappa import HelicalLayerSpec

__all__ = ["HAVE_NUMBA", "wire_friction_stress"]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@njit(cache=True)
def _cap(
    phi: FloatArray,
    angle: float,
    pressure: float,
    r_helix: float,
    n_wires: int,
    area: float,
    lay_angle: float,
    mu: float,
    g: FloatArray,
) -> FloatArray:
    # Row 0: tension bound sigma0 g expm1(lam psi); row 1: compression bound
    # sigma0 g (1 - exp(-lam psi)) (capstan on the wire's own force).
    out = np.zeros((2, phi.size))
    if pressure <= 0.0:
        return out
    sin_a = abs(math.sin(lay_angle))
    if sin_a < 1.0e-12:
        return out
    width = 2.0 * math.pi * r_helix / n_wires
    sigma0 = r_helix * pressure * width * math.cos(lay_angle) / (area * sin_a**2)
    for w in range(phi.size):
        c = abs(math.cos(phi[w] - angle))
        if c > 1.0:
            c = 1.0
        psi = math.asin(c)
        out[0, w] = sigma0 * g[w] * math.expm1(mu * sin_a * psi)
        out[1, w] = -sigma0 * g[w] * math.expm1(-mu * sin_a * psi)
    return out


@njit(cache=True)
def _stress_kernel(
    kx: FloatArray,
    ky: FloatArray,
    pressure: FloatArray,
    k: float,
    phi: FloatArray,
    r_helix: float,
    area: float,
    lay_angle: float,
    mu: float,
    g: FloatArray,
    angle0: float,
    threshold_coefficient: float,
    n_total: int,
) -> FloatArray:
    # phi and g hold the evaluated wires; n_total is the layer's wire count,
    # which sets the tributary width of each wire.
    n_w = phi.size
    n_t = kx.size
    stress = np.empty((n_w, n_t))
    dm0 = r_helix * np.cos(phi)
    dm1 = r_helix * np.sin(phi)
    force = np.zeros(n_w)
    old_drive = np.zeros(n_w)
    held = angle0
    current = angle0
    held_angle = math.nan
    held_pressure = math.nan
    held_cap = np.zeros((2, n_w))
    cur_angle = math.nan
    cur_pressure = math.nan
    cur_cap = np.zeros((2, n_w))
    cap = np.zeros((2, n_w))
    for i in range(n_t):
        x = kx[i]
        y = ky[i]
        p = pressure[i]
        if x != 0.0 or y != 0.0:
            current = math.atan2(y, x)
        threshold = threshold_coefficient * p / k
        magnitude = math.sqrt(x * x + y * y)
        blend = 0.0
        if magnitude >= threshold:
            held = current
        else:
            blend = magnitude / threshold
        if not (held == held_angle and p == held_pressure):
            held_cap = _cap(phi, held, p, r_helix, n_total, area, lay_angle, mu, g)
            held_angle = held
            held_pressure = p
        if blend > 0.0:
            if not (current == cur_angle and p == cur_pressure):
                cur_cap = _cap(
                    phi, current, p, r_helix, n_total, area, lay_angle, mu, g
                )
                cur_angle = current
                cur_pressure = p
            for s in range(2):
                for w in range(n_w):
                    cap[s, w] = (1.0 - blend) * held_cap[s, w] + blend * cur_cap[s, w]
        else:
            for s in range(2):
                for w in range(n_w):
                    cap[s, w] = held_cap[s, w]
        for w in range(n_w):
            drive = dm0[w] * x + dm1[w] * y
            inc = drive - old_drive[w]
            trial = force[w] + k * inc
            if trial > cap[0, w]:
                force[w] = cap[0, w]
            elif trial < -cap[1, w]:
                force[w] = -cap[1, w]
            else:
                force[w] = trial
            old_drive[w] = drive
            stress[w, i] = force[w]
    return stress


@njit(cache=True)
def _cap_parts(
    phi: FloatArray,
    angle: float,
    pressure: float,
    r_helix: float,
    n_wires: int,
    area: float,
    lay_angle: float,
    mu: float,
    g: FloatArray,
    scale: FloatArray,
    x: FloatArray,
    table: FloatArray,
) -> None:
    # Bounds of factor m: scale expm1(m x) (rising), -scale expm1(-m x);
    # x carries the pressure factor f(P g) of the wire (table rows [P, f]).
    sin_a = abs(math.sin(lay_angle))
    if pressure <= 0.0 or sin_a < 1.0e-12:
        for w in range(phi.size):
            scale[w] = 0.0
            x[w] = 0.0
        return
    width = 2.0 * math.pi * r_helix / n_wires
    sigma0 = r_helix * pressure * width * math.cos(lay_angle) / (area * sin_a**2)
    for w in range(phi.size):
        c = abs(math.cos(phi[w] - angle))
        if c > 1.0:
            c = 1.0
        scale[w] = sigma0 * g[w]
        x[w] = mu * sin_a * math.asin(c)
        if table.shape[1] > 0:
            x[w] = x[w] * _pressure_factor(pressure * g[w], table)


@njit(cache=True)
def _stress_kernel_law(
    kx: FloatArray,
    ky: FloatArray,
    pressure: FloatArray,
    k: float,
    phi: FloatArray,
    r_helix: float,
    area: float,
    lay_angle: float,
    mu: float,
    g: FloatArray,
    angle0: float,
    threshold_coefficient: float,
    dt: FloatArray,
    ms: float,
    md: float,
    t_stick: float,
    rc: float,
    v0: float,
    inv_veta: float,
    table: FloatArray,
    n_total: int,
) -> FloatArray:
    n_w = phi.size
    n_t = kx.size
    stress = np.empty((n_w, n_t))
    dm0 = r_helix * np.cos(phi)
    dm1 = r_helix * np.sin(phi)
    # Slip rate per plastic strain rate; a zero lay angle has no capstan
    # friction (zero bounds), so its slip rate does not enter.
    sin_a = abs(math.sin(lay_angle))
    slip_length = r_helix / sin_a if sin_a >= 1.0e-12 else 0.0
    force = np.zeros(n_w)
    old_drive = np.zeros(n_w)
    healed = np.ones(n_w, dtype=np.bool_)
    timer = np.zeros(n_w)
    held = angle0
    current = angle0
    held_angle = math.nan
    held_pressure = math.nan
    scale = np.zeros(n_w)
    x_held = np.zeros(n_w)
    cur_angle = math.nan
    cur_pressure = math.nan
    cur_scale = np.zeros(n_w)
    x_cur = np.zeros(n_w)
    for i in range(n_t):
        x = kx[i]
        y = ky[i]
        p = pressure[i]
        if x != 0.0 or y != 0.0:
            current = math.atan2(y, x)
        threshold = threshold_coefficient * p / k
        magnitude = math.sqrt(x * x + y * y)
        blend = 0.0
        if magnitude >= threshold:
            held = current
        else:
            blend = magnitude / threshold
        if not (held == held_angle and p == held_pressure):
            _cap_parts(
                phi,
                held,
                p,
                r_helix,
                n_total,
                area,
                lay_angle,
                mu,
                g,
                scale,
                x_held,
                table,
            )
            held_angle = held
            held_pressure = p
        if blend > 0.0 and not (current == cur_angle and p == cur_pressure):
            _cap_parts(
                phi,
                current,
                p,
                r_helix,
                n_total,
                area,
                lay_angle,
                mu,
                g,
                cur_scale,
                x_cur,
                table,
            )
            cur_angle = current
            cur_pressure = p
        speed = slip_length / dt[i]
        for w in range(n_w):
            c0 = (1.0 - blend) * scale[w]
            c1 = blend * scale[w]
            xh = x_held[w]
            xc = x_cur[w] if blend > 0.0 else xh
            drive = dm0[w] * x + dm1[w] * y
            trial = force[w] + k * (drive - old_drive[w])
            m = ms if healed[w] else md
            upper = c0 * math.expm1(m * xh) + c1 * math.expm1(m * xc)
            lower = -(c0 * math.expm1(-m * xh) + c1 * math.expm1(-m * xc))
            if trial > upper or trial < -lower:
                side = 1.0 if trial > 0.0 else -1.0
                size = abs(trial)
                dp = _slip_magnitude(
                    size, k, md, side, c0, xh, c1, xc, 0.0, speed, rc, v0, inv_veta
                )
                if not dp >= 0.0:
                    force[w] = math.nan  # unresolved return: refused below
                elif dp > 0.0:
                    # On the end-of-step bound of the slip rate.
                    force[w] = side * _return_stress(
                        size, md, side, c0, xh, c1, xc, 0.0, speed, rc, v0, inv_veta, dp
                    )
                else:
                    force[w] = trial
                healed[w] = False
                timer[w] = 0.0
            else:
                force[w] = trial
                if not healed[w]:
                    timer[w] += dt[i]
                    if timer[w] >= t_stick:
                        healed[w] = True
            old_drive[w] = drive
            stress[w, i] = force[w]
    return stress


def wire_friction_stress(
    spec: HelicalLayerSpec,
    kappa_x: FloatArray,
    kappa_y: FloatArray,
    *,
    contact_pressure_Pa: FloatArray | None = None,
    initial_bending_angle: float = 0.0,
    direction_hold_fraction: float = DIRECTION_HOLD_FRACTION,
    friction_law: FrictionLaw | None = None,
    time_s: FloatArray | None = None,
    wires: IntArray | None = None,
) -> FloatArray:
    """Friction stress [Pa], shape (wires, samples), for a constant stick stiffness.

    ``direction_hold_fraction`` is the fraction ``f`` of the smallest slip-onset
    curvature below which the cap direction is held; ``0`` gives the earlier
    behaviour. ``friction_law`` and ``time_s`` as for
    :func:`~cablexsec.mechanics.papailiou.wire_slip_history`. ``wires``
    (indices into the layer's wires) evaluates those wires only, rows in that
    order; each wire's recursion is independent, so the rows equal the same
    rows of the full evaluation bit for bit.
    """
    kx = np.ascontiguousarray(kappa_x, dtype=np.float64)
    ky = np.ascontiguousarray(kappa_y, dtype=np.float64)
    pressure = (
        np.full(kx.size, float(spec.contact_pressure_Pa))
        if contact_pressure_Pa is None
        else np.ascontiguousarray(contact_pressure_Pa, dtype=np.float64)
    )
    g = (
        np.ones(spec.n_wires)
        if spec.shape_g is None
        else np.ascontiguousarray(spec.shape_g, dtype=np.float64)
    )
    phi = np.ascontiguousarray(spec.phi_wires, dtype=np.float64)
    if wires is not None:
        phi = np.ascontiguousarray(phi[wires])
        g = np.ascontiguousarray(g[wires])
    law = active_friction_law(friction_law)
    if law is not None:
        steps = friction_law_time_steps(law, time_s, kx.size)
        if not float(spec.k_f) > 0.0:
            raise CableInputError("a friction law needs a positive stick stiffness")
        stress = _stress_kernel_law(
            kx,
            ky,
            pressure,
            float(spec.k_f),
            phi,
            float(spec.r_helix_m),
            float(spec.A_wire_m2),
            float(spec.lay_angle_rad),
            float(spec.mu),
            g,
            float(initial_bending_angle),
            _threshold_coefficient(
                _law_threshold_layer(spec, law), float(direction_hold_fraction)
            ),
            np.ones(kx.size) if steps is None else steps,
            law.static_factor,
            law.dynamic_factor,
            law.stick_time_s,
            law.rate_coefficient,
            law.rate_reference_m_per_s,
            law.inverse_viscous_velocity,
            law.pressure_table,
            int(spec.n_wires),
        )
        if not np.isfinite(stress).all():
            raise CableInputError(
                "the friction-law return did not converge (slip rate out of range)"
            )
        return stress
    friction_law_time_steps(None, time_s, kx.size)
    return _stress_kernel(
        kx,
        ky,
        pressure,
        float(spec.k_f),
        phi,
        float(spec.r_helix_m),
        float(spec.A_wire_m2),
        float(spec.lay_angle_rad),
        float(spec.mu),
        g,
        float(initial_bending_angle),
        _threshold_coefficient(spec, float(direction_hold_fraction)),
        int(spec.n_wires),
    )
