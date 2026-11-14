# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Progressive (Iwan) stick-to-slip transition on the direction-tied cap.

Optional alternative to the single spring-slider per wire of
:func:`cablexsec.mechanics.papailiou.wire_slip_history`
(``friction_model="direction_tied"``, the default). Each wire is split into
``N`` parallel elastic--Coulomb sub-elements (an Iwan 1967 parallel-Jenkins
population). Every sub-element has the wire's stick stiffness ``k`` and a
share ``1/N`` of the stress; its cap is a fraction of the package's
Papailiou cap ``sigma_0 g expm1(lam psi)`` of the wire (``lam = mu |sin a|``,
``sin psi = |cos(phi - phi_bend)|``), with the fractions averaging to one, so
that the stick stiffness and the full-slip stress are those of the default
law. The wire stress is the mean of the sub-element forces. The caps follow
the bending direction exactly as the default cap does (held direction and
blend below ``f kappa_on``), so the law applies to biaxial histories and a
rotation of the history rotates the response. Every sub-element dissipates
non-negatively; reversal follows Masing's rule and nested loops close, as for
any parallel-Jenkins population (Iwan 1967, Eqs. 9-15).

Densities
---------
``"continuity"`` (no parameter): the friction-continuity slip law of Saevik
(1992), Dong et al. (2015), Kim (2020, thesis Eq. 3.44) and Lukassen (2019,
Eqs. 32-35). The slip zone grows from the neutral axis to ``psi*`` with
``k r kappa cos psi* = d cap / d psi (psi*)``; a stuck wire carries the
stress of the slip boundary plus its elastic increment beyond it, so its
tangent is ``k (1 - sin psi*/sin psi)`` and full slip is reached only
asymptotically, instead of at ``pi/2`` times the onset curvature for the
pointwise cap. Cell-averaging the yield density over equal shares
``q in [j/N, (j+1)/N]`` gives the closed form

.. math::

    c_j = \sigma_0 g N \left[\mathrm{expm1}(\lambda \arcsin(q_{j+1} S))
          - \mathrm{expm1}(\lambda \arcsin(q_j S))\right],
    \quad S = \sin\psi ,

whose mean over ``j`` is the package cap exactly. ``N = 1`` is the default
law, bit for bit.

``"uniform"`` (spread ``h`` in ``[0, 1]``): caps ``cap (1 + h (2 (j + 1/2)/N
- 1))``, an Iwan density of friction or contact pressure around its mean
(sensitivity studies; no published density for a cable interface is held).
``h = 0`` is the default law, bit for bit.

The rigid-interface limit of the distributed shear-lag model
(:mod:`cablexsec.mechanics.distributed_friction`) is the continuity law; the
two differ through the interface compliance and in biaxial paths (no held
direction in the distributed model).

References
----------
[Iwan1967]; [Saevik1992]; [Kim2020] thesis Eqs. 3.43-3.44; [Lukassen2019]
Eqs. 32-35; [Papailiou1997] Eqs. 14-15.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from cablexsec._numba import njit
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.papailiou import (
    DIRECTION_HOLD_FRACTION,
    BendingWork,
    WireSlipHistory,
    _threshold_coefficient,
)

if TYPE_CHECKING:
    from cablexsec.mechanics.mkappa import HelicalLayerSpec

__all__ = [
    "DEFAULT_PROGRESSIVE_ELEMENTS",
    "PROGRESSIVE_DENSITIES",
    "ProgressiveFriction",
    "ProgressiveWireHistory",
    "progressive_caps",
    "progressive_wire_friction_stress",
    "progressive_wire_history",
]

FloatArray = NDArray[np.float64]
ProgressiveDensity = Literal["continuity", "uniform"]

#: Accepted sub-element densities.
PROGRESSIVE_DENSITIES: tuple[str, ...] = ("continuity", "uniform")
#: Default number of sub-elements per wire.
DEFAULT_PROGRESSIVE_ELEMENTS = 16
_MAX_ELEMENTS = 4096


@dataclass(frozen=True)
class ProgressiveFriction:
    """Settings of the progressive (Iwan) friction law.

    Parameters
    ----------
    density : {"continuity", "uniform"}
        Sub-element cap density (module docstring). ``"continuity"`` has no
        parameter; ``"uniform"`` uses ``spread``.
    n_elements : int
        Sub-elements per wire (``N``); ``1`` reproduces the default law.
    spread : float
        Half-width ``h`` of the uniform density, in ``[0, 1]``; must be zero
        for ``"continuity"``.
    """

    density: ProgressiveDensity = "continuity"
    n_elements: int = DEFAULT_PROGRESSIVE_ELEMENTS
    spread: float = 0.0

    def __post_init__(self) -> None:
        if self.density not in PROGRESSIVE_DENSITIES:
            raise CableInputError(
                f"progressive density must be one of {PROGRESSIVE_DENSITIES}"
            )
        if (
            isinstance(self.n_elements, bool)
            or not isinstance(self.n_elements, (int, np.integer))
            or not 1 <= self.n_elements <= _MAX_ELEMENTS
        ):
            raise CableInputError(
                f"n_elements must be an integer in [1, {_MAX_ELEMENTS}]"
            )
        spread = float(self.spread)
        if not math.isfinite(spread) or not 0.0 <= spread <= 1.0:
            raise CableInputError("spread must lie in [0, 1]")
        if self.density == "continuity" and spread != 0.0:
            raise CableInputError("the continuity density has no spread parameter")
        object.__setattr__(self, "n_elements", int(self.n_elements))
        object.__setattr__(self, "spread", spread)

    @property
    def density_code(self) -> int:
        """0 = continuity, 1 = uniform (compiled kernel argument)."""
        return PROGRESSIVE_DENSITIES.index(self.density)

    def describe(self) -> str:
        """Short provenance text."""
        extra = f", spread {self.spread:g}" if self.density == "uniform" else ""
        return f"{self.density} density, {self.n_elements} sub-elements per wire{extra}"


def _resolve(progressive: ProgressiveFriction | None) -> ProgressiveFriction:
    if progressive is None:
        return ProgressiveFriction()
    if not isinstance(progressive, ProgressiveFriction):
        raise CableInputError("progressive must be a ProgressiveFriction or None")
    return progressive


# --------------------------------------------------------------------------- #
# Caps                                                                         #
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _caps_kernel(
    phi: FloatArray,
    angle: float,
    pressure: float,
    r_helix: float,
    n_wires: int,
    area: float,
    lay_angle: float,
    mu: float,
    g: FloatArray,
    density: int,
    n_el: int,
    spread: float,
) -> FloatArray:
    out = np.zeros((phi.size, n_el))
    if pressure <= 0.0:
        return out
    sin_a = abs(math.sin(lay_angle))
    if sin_a < 1.0e-12:
        return out
    width = 2.0 * math.pi * r_helix / n_wires
    sigma0 = r_helix * pressure * width * math.cos(lay_angle) / (area * sin_a**2)
    lam = mu * sin_a
    scale = float(n_el)
    for w in range(phi.size):
        c = abs(math.cos(phi[w] - angle))
        if c > 1.0:
            c = 1.0
        base = sigma0 * g[w]
        if density == 0:
            lower = 0.0
            for j in range(n_el):
                q = (j + 1.0) / n_el
                if j + 1 == n_el:
                    q = 1.0
                upper = math.expm1(lam * math.asin(q * c))
                out[w, j] = base * scale * (upper - lower)
                lower = upper
        else:
            cap = base * math.expm1(lam * math.asin(c))
            for j in range(n_el):
                out[w, j] = cap * (1.0 + spread * (2.0 * (j + 0.5) / n_el - 1.0))
    return out


def progressive_caps(
    layer: HelicalLayerSpec,
    phi_bend: float,
    progressive: ProgressiveFriction | None = None,
    *,
    contact_pressure_Pa: float | None = None,
) -> FloatArray:
    """Sub-element caps [Pa], shape ``(n_wires, n_elements)``.

    The mean over sub-elements equals
    :func:`cablexsec.mechanics.papailiou.wire_slip_stress_cap` (exactly for
    the uniform density; to rounding for the continuity density).
    """
    spec = _resolve(progressive)
    pressure = (
        layer.contact_pressure_Pa
        if contact_pressure_Pa is None
        else float(contact_pressure_Pa)
    )
    g = (
        np.ones(layer.n_wires)
        if layer.shape_g is None
        else np.ascontiguousarray(layer.shape_g, dtype=np.float64)
    )
    return np.asarray(
        _caps_kernel(
            np.ascontiguousarray(layer.phi_wires, dtype=np.float64),
            float(phi_bend),
            float(pressure),
            float(layer.r_helix_m),
            int(layer.n_wires),
            float(layer.A_wire_m2),
            float(layer.lay_angle_rad),
            float(layer.mu),
            g,
            spec.density_code,
            spec.n_elements,
            spec.spread,
        )
    )


# --------------------------------------------------------------------------- #
# Reference history (stress, moment and work)                                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProgressiveWireHistory(WireSlipHistory):
    """:class:`WireSlipHistory` of the progressive law.

    ``slipping_count`` counts wires with at least one slipping sub-element;
    ``slipping_share`` is the summed share of slipping sub-elements (wires).
    ``final_plastic_drive`` is the sub-element mean.
    """

    slipping_share: FloatArray = field(default_factory=lambda: np.zeros(0))
    settings: ProgressiveFriction = field(default_factory=ProgressiveFriction)


def progressive_wire_history(
    layer: HelicalLayerSpec,
    kappa_x: FloatArray,
    kappa_y: FloatArray,
    *,
    progressive: ProgressiveFriction | None = None,
    contact_pressure_Pa: FloatArray | None = None,
    stick_modulus_Pa: FloatArray | None = None,
    initial_bending_angle: float = 0.0,
    direction_hold_fraction: float = DIRECTION_HOLD_FRACTION,
) -> ProgressiveWireHistory:
    """Stress, moment and work of the progressive law (reference path).

    Same inputs and bending-direction rule as
    :func:`cablexsec.mechanics.papailiou.wire_slip_history`; a changing stick
    modulus rescales every sub-element force at fixed elastic strain and
    records the material work, as the default law does.
    """
    spec = _resolve(progressive)
    kx, ky = np.asarray(kappa_x, dtype=float), np.asarray(kappa_y, dtype=float)
    if (
        kx.ndim != 1
        or ky.shape != kx.shape
        or not kx.size
        or not np.isfinite([kx, ky]).all()
    ):
        raise CableInputError(
            "finite matching nonempty curvature histories are required"
        )
    if not math.isfinite(initial_bending_angle):
        raise CableInputError("initial bending angle must be finite")
    threshold_coefficient = _threshold_coefficient(
        layer, float(direction_hold_fraction)
    )
    count = kx.size
    pressure = (
        np.full(count, layer.contact_pressure_Pa)
        if contact_pressure_Pa is None
        else np.asarray(contact_pressure_Pa, dtype=float)
    )
    stiffness = (
        np.full(count, layer.k_f)
        if stick_modulus_Pa is None
        else np.asarray(stick_modulus_Pa, dtype=float)
    )
    for label, values in (("pressure", pressure), ("stiffness", stiffness)):
        if values.shape != kx.shape or not np.isfinite(values).all():
            raise CableInputError(
                f"{label} must be finite and match curvature histories"
            )
    if np.any(pressure < 0) or np.any(stiffness <= 0):
        raise CableInputError(
            "nonnegative pressure and positive stiffness are required"
        )
    n_el = spec.n_elements
    share = 1.0 / n_el
    n_w = layer.n_wires
    directions = np.column_stack((np.cos(layer.phi_wires), np.sin(layer.phi_wires)))
    drive_map = layer.r_helix_m * directions
    weight = layer.A_wire_m2 * math.cos(layer.lay_angle_rad)
    stress = np.empty((n_w, count))
    moment = np.empty((count, 2))
    slipping_count = np.empty(count, dtype=np.int64)
    slipping_share = np.empty(count)
    stored, friction, endpoint, remainder, material = (
        np.zeros(count) for _ in range(5)
    )
    totals = np.zeros(4)
    force = np.zeros((n_w, n_el))
    plastic = np.zeros((n_w, n_el))
    old_drive = np.zeros(n_w)
    k_now = float(stiffness[0])
    held = current = initial_bending_angle
    caches: dict[str, tuple[tuple[float, float] | None, FloatArray]] = {
        "held": (None, np.zeros((n_w, n_el))),
        "current": (None, np.zeros((n_w, n_el))),
    }

    def oriented(slot: str, angle: float, load: float) -> FloatArray:
        key, value = caches[slot]
        if key != (angle, load):
            value = progressive_caps(layer, angle, spec, contact_pressure_Pa=load)
            if not np.isfinite(value).all() or np.any(value < 0):
                raise CableInputError("sub-element caps must be finite and nonnegative")
            caches[slot] = ((angle, load), value)
        return value

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            for index, (x, y) in enumerate(zip(kx, ky, strict=True)):
                k_new = float(stiffness[index])
                if k_new != k_now:
                    totals[3] += (
                        0.5
                        * weight
                        * (k_new - k_now)
                        * share
                        * ((force / k_now) ** 2).sum()
                    )
                    force = force * (k_new / k_now)
                    k_now = k_new
                old_force, old_plastic = force, plastic
                load = float(pressure[index])
                if x != 0 or y != 0:
                    current = math.atan2(y, x)
                threshold = threshold_coefficient * load / k_now
                magnitude = math.sqrt(x * x + y * y)
                if magnitude >= threshold:
                    held = current
                    cap = oriented("held", held, load)
                else:
                    blend = magnitude / threshold
                    cap = oriented("held", held, load)
                    if blend > 0.0:
                        cap = (1.0 - blend) * cap + blend * oriented(
                            "current", current, load
                        )
                drive = drive_map @ np.array([x, y])
                increment = drive - old_drive
                if not np.isfinite(increment).all():
                    raise CableInputError("derived wire increment must be finite")
                trial = force + k_now * increment[:, None]
                slipping = np.abs(trial) > cap
                force = np.where(slipping, np.copysign(cap, trial), trial)
                slip = np.where(
                    slipping, increment[:, None] - (force - old_force) / k_now, 0.0
                )
                plastic = old_plastic + slip
                wire = force.sum(axis=1) * share if n_el > 1 else force[:, 0]
                stress[:, index] = wire
                moment[index] = weight * (wire @ drive_map)
                slipping_count[index] = np.count_nonzero(slipping.any(axis=1))
                slipping_share[index] = slipping.sum() * share
                stored[index] = weight * share * (force**2).sum() / (2 * k_now)
                totals[0] += weight * share * (force * (plastic - old_plastic)).sum()
                totals[1] += weight * (wire * increment).sum()
                totals[2] += (
                    weight * share * ((force - old_force) ** 2).sum() / (2 * k_now)
                )
                friction[index], endpoint[index], remainder[index], material[index] = (
                    totals
                )
                old_drive = drive
            balance = endpoint + material - stored - friction - remainder
        except FloatingPointError as error:
            raise CableInputError(
                "wire history exceeds floating-point range"
            ) from error
    if not all(
        np.isfinite(v).all()
        for v in (
            stress,
            moment,
            plastic,
            stored,
            friction,
            endpoint,
            remainder,
            balance,
        )
    ):
        raise CableInputError("wire history produced nonfinite results")
    return ProgressiveWireHistory(
        stress,
        moment,
        slipping_count,
        plastic.mean(axis=1),
        material,
        BendingWork(stored, friction, endpoint, remainder, balance),
        slipping_share=slipping_share,
        settings=spec,
    )


# --------------------------------------------------------------------------- #
# Compiled stress kernel (fatigue path)                                        #
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _progressive_stress_kernel(
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
    density: int,
    n_el: int,
    spread: float,
) -> FloatArray:
    n_w = phi.size
    n_t = kx.size
    stress = np.empty((n_w, n_t))
    dm0 = r_helix * np.cos(phi)
    dm1 = r_helix * np.sin(phi)
    force = np.zeros((n_w, n_el))
    old_drive = np.zeros(n_w)
    held = angle0
    current = angle0
    held_angle = math.nan
    held_pressure = math.nan
    held_cap = np.zeros((n_w, n_el))
    cur_angle = math.nan
    cur_pressure = math.nan
    cur_cap = np.zeros((n_w, n_el))
    share = 1.0 / n_el
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
            held_cap = _caps_kernel(
                phi,
                held,
                p,
                r_helix,
                n_w,
                area,
                lay_angle,
                mu,
                g,
                density,
                n_el,
                spread,
            )
            held_angle = held
            held_pressure = p
        if blend > 0.0 and not (current == cur_angle and p == cur_pressure):
            cur_cap = _caps_kernel(
                phi,
                current,
                p,
                r_helix,
                n_w,
                area,
                lay_angle,
                mu,
                g,
                density,
                n_el,
                spread,
            )
            cur_angle = current
            cur_pressure = p
        for w in range(n_w):
            drive = dm0[w] * x + dm1[w] * y
            inc = drive - old_drive[w]
            total = 0.0
            for j in range(n_el):
                cap = held_cap[w, j]
                if blend > 0.0:
                    cap = (1.0 - blend) * held_cap[w, j] + blend * cur_cap[w, j]
                trial = force[w, j] + k * inc
                if abs(trial) > cap:
                    force[w, j] = math.copysign(cap, trial)
                else:
                    force[w, j] = trial
                total += force[w, j]
            old_drive[w] = drive
            stress[w, i] = total * share if n_el > 1 else total
    return stress


def progressive_wire_friction_stress(
    spec: HelicalLayerSpec,
    kappa_x: FloatArray,
    kappa_y: FloatArray,
    *,
    progressive: ProgressiveFriction | None = None,
    contact_pressure_Pa: FloatArray | None = None,
    initial_bending_angle: float = 0.0,
    direction_hold_fraction: float = DIRECTION_HOLD_FRACTION,
) -> FloatArray:
    """Friction stress [Pa], shape (wires, samples), constant stick stiffness.

    Compiled when ``numba`` is active; regression tests compare it with
    :func:`progressive_wire_history`.
    """
    settings = _resolve(progressive)
    kx = np.ascontiguousarray(kappa_x, dtype=np.float64)
    ky = np.ascontiguousarray(kappa_y, dtype=np.float64)
    if kx.ndim != 1 or ky.shape != kx.shape or not kx.size:
        raise CableInputError(
            "finite matching nonempty curvature histories are required"
        )
    if not (np.isfinite(kx).all() and np.isfinite(ky).all()):
        raise CableInputError(
            "finite matching nonempty curvature histories are required"
        )
    pressure = (
        np.full(kx.size, float(spec.contact_pressure_Pa))
        if contact_pressure_Pa is None
        else np.ascontiguousarray(contact_pressure_Pa, dtype=np.float64)
    )
    if (
        pressure.shape != kx.shape
        or not np.isfinite(pressure).all()
        or np.any(pressure < 0)
    ):
        raise CableInputError(
            "pressure must be finite, nonnegative and match the history"
        )
    g = (
        np.ones(spec.n_wires)
        if spec.shape_g is None
        else np.ascontiguousarray(spec.shape_g, dtype=np.float64)
    )
    return np.asarray(
        _progressive_stress_kernel(
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
            settings.density_code,
            settings.n_elements,
            settings.spread,
        )
    )
