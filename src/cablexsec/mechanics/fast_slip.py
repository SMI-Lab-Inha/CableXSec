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
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from cablexsec._numba import HAVE_NUMBA, njit
from cablexsec.mechanics.papailiou import (
    DIRECTION_HOLD_FRACTION,
    _threshold_coefficient,
)

if TYPE_CHECKING:
    from cablexsec.mechanics.mkappa import HelicalLayerSpec

__all__ = ["HAVE_NUMBA", "wire_friction_stress"]

FloatArray = NDArray[np.float64]


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
) -> FloatArray:
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
            held_cap = _cap(phi, held, p, r_helix, n_w, area, lay_angle, mu, g)
            held_angle = held
            held_pressure = p
        if blend > 0.0:
            if not (current == cur_angle and p == cur_pressure):
                cur_cap = _cap(phi, current, p, r_helix, n_w, area, lay_angle, mu, g)
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


def wire_friction_stress(
    spec: HelicalLayerSpec,
    kappa_x: FloatArray,
    kappa_y: FloatArray,
    *,
    contact_pressure_Pa: FloatArray | None = None,
    initial_bending_angle: float = 0.0,
    direction_hold_fraction: float = DIRECTION_HOLD_FRACTION,
) -> FloatArray:
    """Friction stress [Pa], shape (wires, samples), for a constant stick stiffness.

    ``direction_hold_fraction`` is the fraction ``f`` of the smallest slip-onset
    curvature below which the cap direction is held; ``0`` gives the earlier
    behaviour.
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
    return _stress_kernel(
        kx,
        ky,
        pressure,
        float(spec.k_f),
        np.ascontiguousarray(spec.phi_wires, dtype=np.float64),
        float(spec.r_helix_m),
        float(spec.A_wire_m2),
        float(spec.lay_angle_rad),
        float(spec.mu),
        g,
        float(initial_bending_angle),
        _threshold_coefficient(spec, float(direction_hold_fraction)),
    )
