# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Simulation of a combined tension--bending rig test for gauge planning.

A specimen of free length ``L`` is clamped at both ends, tensioned and bent in
one plane (optionally at a raised temperature). :func:`simulate_rig_test`
predicts what strain gauges on the armour wires would read and where they
should go:

* moment--curvature loops at stations along the specimen;
* armour-wire axial strain histories at candidate gauge positions (layer,
  angle round the section, distance from the clamp face), split into the
  axial (tension and temperature), local wire-bending and friction parts;
* strain ranges and their sensitivity to the friction coefficient and to the
  clamp (anchored armour against the sectional, unanchored result);
* recommended gauge positions with expected amplitudes against an assumed
  gauge resolution.

Mechanics
---------
Normal state: the solved radial state (:func:`solve_cable_radial_state`,
``contact_model="solved"``) at a grid of tension and temperature levels,
interpolated over the protocol, with the twist held at zero as at a rig clamp.
It gives the friction-setting pressure ``P1+P2`` and the uniform wire stress
of each armour layer. Cables the solved model rejects fall back to the bounded
pressure chain (``contact_model="auto"``).

Sectional friction and local bending: the package laws at the local curvature
``kappa(x, t) = kappa_ref(t) k_hat(x)`` (direction-tied friction,
:func:`~cablexsec.mechanics.fast_slip.wire_friction_stress`; outward-fibre
local bending, :func:`~cablexsec.mechanics.wire_stress.local_bending_fibre_stress`).

Clamp: each wire is followed along the whole specimen with the anchored-wire
solver of :mod:`cablexsec.mechanics.clamp_layer` (anchors at both clamps,
rigid stick, Coulomb bound scaled with ``P(t)``). The friction stress is the
sectional value plus the clamp increment ``sigma_u (n - n_ref)``, where
``n_ref`` is the pointwise sectional field of the same solver. Far from the
clamps the increment vanishes, so the constant-curvature sectional result is
the far-field limit.

Curvature profiles ``k_hat(x)``: ``"uniform"`` (bending over a template or a
four-point rig; the armour is anchored where the cable is curved),
``"end_rotation"`` (tensioned beam, one clamp rotated, the other fixed;
decay length ``sqrt(EI / T)``) and ``"table"`` (user profile, e.g. from a
rig finite-element model).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray

from cablexsec.analysis.thermal_coupling import (
    ThermalHistory,
    ThermalState,
    _record_blocks,
    thermal_material_state,
)
from cablexsec.api import CoreFrictionSpec, compute_mkappa, compute_stiffness
from cablexsec.cable_adapters import (
    cable_to_mkappa_specs,
    require_fully_specified,
    structural_armour_layers,
)
from cablexsec.cables.schema import CableData
from cablexsec.exceptions import CableInputError, CableXSecError
from cablexsec.materials import library
from cablexsec.mechanics.bend_restrictor import CurvatureField
from cablexsec.mechanics.clamp_layer import (
    anchored_wire_field_response,
    anchored_wire_response,
    clamp_amplification,
    clamp_boundary_layer_length,
    clamp_mean_force,
    clamp_setback_length,
)
from cablexsec.mechanics.fast_slip import wire_friction_stress
from cablexsec.mechanics.mkappa import HelicalLayerSpec
from cablexsec.mechanics.papailiou import slip_onset_curvature
from cablexsec.mechanics.radial_cable import (
    CableRadialState,
    solve_cable_radial_state,
    solve_cable_radial_states,
)
from cablexsec.mechanics.stiffness import xi_for_layer
from cablexsec.mechanics.wire_stress import local_bending_fibre_stress
from cablexsec.provenance import (
    Provenance,
    real_array,
    relayed_warnings,
    require_finite,
)

__all__ = [
    "DEFAULT_GAUGE_RESOLUTION_MICROSTRAIN",
    "GaugeLocation",
    "GaugeRecommendation",
    "GaugeResult",
    "LayerScales",
    "RigProtocol",
    "RigSetup",
    "RigTestResult",
    "StationLoop",
    "default_gauge_grid",
    "end_rotation_profile",
    "simulate_rig_test",
]

FloatArray = NDArray[np.float64]

#: Assumed resolution of a bonded foil gauge on a quarter or half bridge with a
#: 24-bit bridge amplifier in a laboratory [microstrain]. Drift and
#: installation scatter are larger; set the value for the actual system.
DEFAULT_GAUGE_RESOLUTION_MICROSTRAIN = 2.0
#: Detectability: a difference is called measurable above this many resolutions.
DETECTION_FACTOR = 3.0

_PROFILES = ("uniform", "end_rotation", "table", "field")
_CLAMPS = ("anchored", "none")
_CONTACT_MODELS = ("auto", "solved", "bounded")


# --------------------------------------------------------------------------- #
# Inputs                                                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RigProtocol:
    """Load history of the rig.

    Attributes
    ----------
    time_s : numpy.ndarray
        Sample times [s], strictly increasing.
    tension_N : numpy.ndarray
        True wall tension [N] per sample (non-negative).
    curvature_1_m : numpy.ndarray or None
        Signed curvature at the reference station of the setup [1/m]
        (strain gradient towards the tensile extreme fibre at 0 deg).
    end_rotation_rad : numpy.ndarray or None
        Rotation of the clamp at ``x = 0`` [rad] (``"end_rotation"`` profile
        only); give this or ``curvature_1_m``.
    temperature_C : numpy.ndarray or None
        Uniform specimen temperature [deg C]; ``None`` is the reference
        temperature (20 deg C).
    retain_from_s : float
        Ranges are taken over samples with ``time_s >= retain_from_s``.
    """

    time_s: FloatArray
    tension_N: FloatArray
    curvature_1_m: FloatArray | None = None
    end_rotation_rad: FloatArray | None = None
    temperature_C: FloatArray | None = None
    retain_from_s: float = 0.0

    def __post_init__(self) -> None:
        t = real_array(self.time_s, "time_s")
        if t.ndim != 1 or t.size < 3 or not np.isfinite(t).all():
            raise CableInputError(
                "time_s must be a finite vector of at least 3 samples"
            )
        if np.any(np.diff(t) <= 0):
            raise CableInputError("time_s must be strictly increasing")
        object.__setattr__(self, "time_s", _frozen(t))
        for name in ("tension_N", "curvature_1_m", "end_rotation_rad", "temperature_C"):
            value = getattr(self, name)
            if value is None:
                continue
            arr = real_array(value, name)
            if arr.ndim == 0:
                arr = np.full(t.size, float(arr))
            if arr.shape != t.shape or not np.isfinite(arr).all():
                raise CableInputError(f"{name} must be finite and match time_s")
            object.__setattr__(self, name, _frozen(arr))
        if np.any(self.tension_N < 0):
            raise CableInputError("tension_N must be non-negative")
        if (self.curvature_1_m is None) == (self.end_rotation_rad is None):
            raise CableInputError(
                "give exactly one of curvature_1_m or end_rotation_rad"
            )
        require_finite(self.retain_from_s, "retain_from_s")
        if self.retain_from_s > t[-1]:
            raise CableInputError("retain_from_s lies after the last sample")

    @classmethod
    def cyclic(
        cls,
        *,
        tension_N: float,
        curvature_amplitude_1_m: float | None = None,
        end_rotation_amplitude_rad: float | None = None,
        mean: float = 0.0,
        tension_amplitude_N: float = 0.0,
        temperature_C: float | None = None,
        n_cycles: int = 3,
        samples_per_cycle: int = 80,
        period_s: float = 10.0,
    ) -> RigProtocol:
        """Tension ramp from zero (one period), then sinusoidal bending.

        Tension may cycle in phase with bending (``tension_amplitude_N``).
        Ranges are retained over the last cycle.
        """
        if (curvature_amplitude_1_m is None) == (end_rotation_amplitude_rad is None):
            raise CableInputError(
                "give one of curvature_amplitude_1_m or end_rotation_amplitude_rad"
            )
        if n_cycles < 1 or samples_per_cycle < 8 or period_s <= 0:
            raise CableInputError(
                "need n_cycles >= 1, samples_per_cycle >= 8, period > 0"
            )
        amp = float(
            curvature_amplitude_1_m
            if curvature_amplitude_1_m is not None
            else end_rotation_amplitude_rad  # type: ignore[arg-type]
        )
        ramp_t = np.linspace(0.0, period_s, samples_per_cycle // 2 + 1)
        cyc_t = (
            period_s
            + np.linspace(0.0, n_cycles * period_s, n_cycles * samples_per_cycle + 1)[
                1:
            ]
        )
        time = np.concatenate([ramp_t, cyc_t])
        phase = np.zeros(time.size)
        phase[ramp_t.size :] = 2 * math.pi * (cyc_t - period_s) / period_s
        bend_ramp = np.zeros(ramp_t.size)
        bend_ramp[:] = mean * ramp_t / period_s
        bend = np.concatenate([bend_ramp, mean + amp * np.sin(phase[ramp_t.size :])])
        tension = np.concatenate(
            [
                tension_N * ramp_t / period_s,
                tension_N + tension_amplitude_N * np.sin(phase[ramp_t.size :]),
            ]
        )
        temperature = (
            None if temperature_C is None else np.full(time.size, temperature_C)
        )
        retain = period_s * n_cycles
        return cls(
            time_s=time,
            tension_N=tension,
            curvature_1_m=bend if curvature_amplitude_1_m is not None else None,
            end_rotation_rad=bend if end_rotation_amplitude_rad is not None else None,
            temperature_C=temperature,
            retain_from_s=retain,
        )

    @classmethod
    def from_curvature_field(
        cls,
        field: CurvatureField,
        *,
        time_s: FloatArray,
        tension_N: FloatArray | float,
        reference_x_m: float,
        temperature_C: FloatArray | None = None,
        retain_from_s: float = 0.0,
    ) -> RigProtocol:
        """Protocol whose reference curvature is ``field`` at ``reference_x_m``.

        Use with ``RigSetup(profile="field", curvature_field=field,
        reference_x_m=reference_x_m)`` (gap P5).
        """
        return cls(
            time_s=time_s,
            tension_N=np.broadcast_to(
                np.asarray(tension_N, dtype=np.float64), np.shape(time_s)
            ).copy(),
            curvature_1_m=field.at(reference_x_m),
            temperature_C=temperature_C,
            retain_from_s=retain_from_s,
        )


@dataclass(frozen=True)
class RigSetup:
    """Specimen and clamp description.

    Attributes
    ----------
    free_length_m : float
        Length between the two clamp faces [m].
    profile : {"uniform", "end_rotation", "table"}
        Curvature shape along the specimen.
    profile_x_m, profile_shape : numpy.ndarray or None
        Tabulated shape for ``"table"`` (``x`` from the clamp face at 0).
    clamp : {"anchored", "none"}
        ``"anchored"``: the armour is anchored at both clamps (end fitting,
        epoxy or a gripping collar). ``"none"``: sectional result only.
    clamp_setback_m : float
        Straight, frictional length between the armour anchor and the clamp
        face (zero curvature) at each end [m]; zero for an anchor at the face.
    reference_x_m : float or None
        Station at which ``RigProtocol.curvature_1_m`` is defined. Default:
        mid-span (``"uniform"``), the rotated clamp face (``"end_rotation"``)
        or the largest tabulated value (``"table"``).
    bending_stiffness_Nm2 : float or None
        Secant bending stiffness for ``"end_rotation"``; default iterates the
        secant stiffness of the package bending loop at the clamp amplitude.
    curvature_field : CurvatureField or None
        Non-proportional field ``kappa(x, t)`` for ``profile="field"`` (gap
        P5: :func:`~cablexsec.mechanics.bend_restrictor.restrictor_curvature_field`
        or :func:`~cablexsec.mechanics.bend_restrictor.stiffener_curvature_field`),
        one column per protocol sample, ``x`` from the clamp face. The
        protocol's ``curvature_1_m`` must be the field at the reference
        station (:meth:`RigProtocol.from_curvature_field`).
    """

    free_length_m: float
    profile: str = "uniform"
    profile_x_m: FloatArray | None = None
    profile_shape: FloatArray | None = None
    clamp: str = "anchored"
    clamp_setback_m: float = 0.0
    reference_x_m: float | None = None
    bending_stiffness_Nm2: float | None = None
    curvature_field: CurvatureField | None = None

    def __post_init__(self) -> None:
        if (self.profile == "field") != (self.curvature_field is not None):
            raise CableInputError(
                "profile='field' and curvature_field must be given together"
            )
        require_finite(self.free_length_m, "free_length_m")
        require_finite(self.clamp_setback_m, "clamp_setback_m")
        if self.free_length_m <= 0 or self.clamp_setback_m < 0:
            raise CableInputError(
                "free length must be positive and set-back non-negative"
            )
        if self.profile not in _PROFILES:
            raise CableInputError(f"profile must be one of {_PROFILES}")
        if self.clamp not in _CLAMPS:
            raise CableInputError(f"clamp must be one of {_CLAMPS}")
        if self.profile == "table":
            if self.profile_x_m is None or self.profile_shape is None:
                raise CableInputError(
                    "a table profile needs profile_x_m and profile_shape"
                )
            x = real_array(self.profile_x_m, "profile_x_m")
            s = real_array(self.profile_shape, "profile_shape")
            if (
                x.ndim != 1
                or x.shape != s.shape
                or x.size < 2
                or not (np.isfinite(x).all() and np.isfinite(s).all())
                or np.any(np.diff(x) <= 0)
                or np.max(np.abs(s)) == 0
            ):
                raise CableInputError(
                    "profile table must be finite, increasing in x and not all zero"
                )
            object.__setattr__(self, "profile_x_m", _frozen(x))
            object.__setattr__(self, "profile_shape", _frozen(s))
        if (
            self.reference_x_m is not None
            and not 0 <= self.reference_x_m <= self.free_length_m
        ):
            raise CableInputError("reference_x_m must lie on the free length")
        if self.bending_stiffness_Nm2 is not None and not (
            math.isfinite(self.bending_stiffness_Nm2) and self.bending_stiffness_Nm2 > 0
        ):
            raise CableInputError("bending_stiffness_Nm2 must be positive")


@dataclass(frozen=True)
class GaugeLocation:
    """Candidate gauge: armour layer, angle round the section, distance from clamp.

    ``angle_deg`` is measured from the tensile extreme fibre (positive
    curvature) towards ``+y``; the cable axis points from the clamp face at
    ``x = 0`` into the specimen. The gauge sits on the outward face of the wire
    nearest to this angle at ``x_m``.
    """

    layer: str
    angle_deg: float
    x_m: float


# --------------------------------------------------------------------------- #
# Results                                                                      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LayerScales:
    """Friction scales of one armour layer at the reference state.

    ``kappa_c_1_m`` is the slip-onset curvature, ``friction_unit_microstrain``
    the wire strain per unit normalised force (``sigma_u / E``), ``a_peak`` the
    largest ``|kappa| / kappa_c`` on the specimen, ``slip_length_m`` the
    corresponding ``L_s = a P / (2 pi)``, ``boundary_layer_m`` the clamp layer
    ``sqrt(L_s P) / 2``, ``clamp_law_amplification`` ``1 + 0.54 sqrt(a)`` at the
    clamp-face curvature, ``setback_length_m`` the map-fit set-back
    ``0.4 sqrt(L_s P)`` and ``clamp_law_peak_microstrain`` the averaged-law
    peak friction strain of the neutral-axis wire at the clamp
    (``(min(a0, pi/2) + |H0|) sigma_u / E``, monotonic, ``a0`` at the face).
    """

    layer: str
    pitch_m: float
    lay_direction: str
    n_wires: int
    contact_pressure_Pa: float
    kappa_c_1_m: float
    friction_unit_microstrain: float
    sectional_cap_microstrain: float
    axial_microstrain_at_reference: float
    a_peak: float
    a_clamp: float
    slip_length_m: float
    boundary_layer_m: float
    setback_length_m: float
    clamp_law_amplification: float
    clamp_law_peak_microstrain: float


@dataclass(frozen=True)
class StationLoop:
    """Moment--curvature loop at one station [1/m, N m].

    ``moment_sectional_Nm`` is :func:`compute_mkappa` at the local curvature
    (mean protocol tension); ``moment_clamp_Nm`` the moment of the clamp
    increment of the wire forces; ``moment_Nm`` their sum.
    """

    x_m: float
    kappa_1_m: FloatArray
    moment_sectional_Nm: FloatArray
    moment_clamp_Nm: FloatArray
    moment_Nm: FloatArray


@dataclass(frozen=True)
class GaugeResult:
    """Strain histories at one gauge [microstrain] (nominal friction coefficient).

    ``friction`` is the sectional friction strain plus the clamp increment
    (``friction_sectional`` alone is the unanchored, constant-curvature
    result). ``total = axial + local_bending + friction``. ``ranges`` holds
    max minus min over the retained samples per part; ``range_by_mu`` the total
    range per friction coefficient; ``range_sectional`` the total range
    without the clamp increment.
    """

    location: GaugeLocation
    wire: int
    angle_deg: float
    axial: FloatArray
    local_bending: FloatArray
    friction: FloatArray
    friction_sectional: FloatArray
    total: FloatArray
    ranges: Mapping[str, float]
    range_by_mu: Mapping[float, float]
    range_sectional: float
    range_by_setback: Mapping[float, float] = field(default_factory=dict)

    @property
    def clamp_contrast(self) -> float:
        """Total range with the clamp minus without it [microstrain]."""
        return self.ranges["total"] - self.range_sectional

    @property
    def mu_sensitivity(self) -> float:
        """Total-range change between the highest and lowest mu [microstrain]."""
        keys = sorted(self.range_by_mu)
        return self.range_by_mu[keys[-1]] - self.range_by_mu[keys[0]]


@dataclass(frozen=True)
class GaugeRecommendation:
    """A recommended gauge with its purpose and expected signal."""

    purpose: str
    gauge: GaugeResult
    amplitude_microstrain: float
    signal_to_resolution: float
    reason: str


@dataclass(frozen=True)
class RigTestResult:
    """Output of :func:`simulate_rig_test`."""

    time_s: FloatArray
    retained: NDArray[np.bool_]
    x_m: FloatArray
    curvature_shape: FloatArray
    reference_curvature_1_m: FloatArray
    reference_x_m: float
    bending_stiffness_Nm2: float | None
    mu_nominal: Mapping[str, float]
    mu_values: tuple[float, ...]
    contact_model: str
    layers: tuple[LayerScales, ...]
    stations: tuple[StationLoop, ...]
    gauges: tuple[GaugeResult, ...]
    recommendations: tuple[GaugeRecommendation, ...]
    gauge_resolution_microstrain: float
    provenance: Provenance
    notes: tuple[str, ...] = field(default=())

    def gauge(self, layer: str, angle_deg: float, x_m: float) -> GaugeResult:
        """Result of the candidate gauge nearest to the given location."""
        best = min(
            (g for g in self.gauges if g.location.layer == layer),
            key=lambda g: (
                abs(g.location.x_m - x_m),
                abs(_wrap_deg(g.location.angle_deg - angle_deg)),
            ),
        )
        return best

    def summary(self) -> str:
        """Plain-text summary of scales and recommendations."""
        lines = [
            f"contact model: {self.contact_model}; gauge resolution "
            f"{self.gauge_resolution_microstrain:g} microstrain (assumed)",
        ]
        for s in self.layers:
            lines.append(
                f"{s.layer}: P={s.pitch_m:.3f} m, "
                f"P1+P2={s.contact_pressure_Pa / 1e6:.2f} MPa, "
                f"kappa_c={s.kappa_c_1_m:.4f} 1/m, a_peak={s.a_peak:.1f}, "
                f"L_s={s.slip_length_m:.2f} m, l_x={s.boundary_layer_m:.2f} m, "
                f"clamp law A={s.clamp_law_amplification:.2f}, "
                f"sectional cap {s.sectional_cap_microstrain:.0f} ue, "
                f"axial {s.axial_microstrain_at_reference:.0f} ue at mean tension"
            )
        for r in self.recommendations:
            g = r.gauge
            lines.append(
                f"[{r.purpose}] {g.location.layer} wire {g.wire} at "
                f"{g.angle_deg:.0f} deg, x={g.location.x_m:.3f} m: total range "
                f"{g.ranges['total']:.0f} ue (axial {g.ranges['axial']:.0f}, "
                f"bending {g.ranges['local_bending']:.0f}, "
                f"friction {g.ranges['friction']:.0f}); "
                f"amplitude/resolution {r.signal_to_resolution:.0f}; "
                f"clamp contrast {g.clamp_contrast:.0f} ue; mu sensitivity "
                f"{g.mu_sensitivity:.0f} ue. {r.reason}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _frozen(values: FloatArray) -> FloatArray:
    out = np.array(values, dtype=np.float64, copy=True)
    out.flags.writeable = False
    return out


def _wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def end_rotation_profile(
    x_m: FloatArray, free_length_m: float, decay_length_m: float
) -> FloatArray:
    r"""Curvature per unit end rotation of a tensioned beam [1/m per rad].

    ``EI w'''' - T w'' = 0`` on ``[0, L]`` with ``w(0) = 0``, ``w'(0) = 1``,
    ``w(L) = w'(L) = 0`` and ``decay_length_m = sqrt(EI / T)``; for
    ``decay_length_m > 20 L`` (or infinite) the untensioned beam
    ``w'' = -4/L + 6x/L^2`` is used. For ``sqrt(EI / T) << L`` the clamp
    curvature is ``-1 / sqrt(EI / T)``.
    """
    x = np.asarray(x_m, dtype=np.float64)
    length = float(free_length_m)
    lam = float(decay_length_m)
    if not math.isfinite(lam) or lam > 20 * length:
        return np.asarray(-4.0 / length + 6.0 * x / length**2)
    if lam <= 0:
        raise CableInputError("decay length must be positive")
    e = math.exp(-length / lam)
    matrix = np.array(
        [
            [1.0, 0.0, lam**2, lam**2 * e],
            [0.0, 1.0, -lam, lam * e],
            [1.0, length, lam**2 * e, lam**2],
            [0.0, 1.0, -lam * e, lam],
        ]
    )
    c = np.linalg.solve(matrix, np.array([0.0, 1.0, 0.0, 0.0]))
    return np.asarray(c[2] * np.exp(-x / lam) + c[3] * np.exp(-(length - x) / lam))


def default_gauge_grid(
    cable: CableData,
    setup: RigSetup,
    *,
    angles_deg: Sequence[float] = (0, 45, 90, 135, 180, 225, 270, 315),
    min_distance_m: float = 0.05,
) -> list[GaugeLocation]:
    """Candidate gauges for every structural armour layer.

    Stations: ``min_distance_m`` from the clamp face, a quarter, half and one
    pitch from it, and mid-span. Angles: ``angles_deg`` plus the two angles at
    which the wires that lie on the neutral axis at the anchor pass the
    station (the worst wires of the clamp law).
    """
    out: list[GaugeLocation] = []
    half = setup.free_length_m / 2
    for lyr in structural_armour_layers(cable):
        assert lyr.r_helix_m is not None and lyr.lay_angle_rad is not None
        pitch = 2 * math.pi * lyr.r_helix_m / math.tan(lyr.lay_angle_rad)
        hand = 1.0 if lyr.lay_direction == "right_hand" else -1.0
        xs = {min_distance_m, 0.25 * pitch, 0.5 * pitch, pitch, half}
        for x in sorted(v for v in xs if min_distance_m <= v <= half):
            turn = hand * 360.0 * (x + setup.clamp_setback_m) / pitch
            angles = [float(a) % 360.0 for a in angles_deg]
            for base in (90.0, 270.0):
                extra = round((base + turn) % 360.0, 1)
                if all(abs(_wrap_deg(extra - a)) > 1.0 for a in angles):
                    angles.append(extra)
            out.extend(GaugeLocation(lyr.name, a, round(float(x), 4)) for a in angles)
    return out


def _levels(values: FloatArray, n: int) -> FloatArray:
    lo, hi = float(values.min()), float(values.max())
    if hi - lo <= 1e-9 * max(1.0, abs(hi)):
        return np.array([lo])
    return np.linspace(lo, hi, n)


def _interp2(
    grid: FloatArray,
    t_levels: FloatArray,
    c_levels: FloatArray,
    t: FloatArray,
    c: FloatArray,
) -> FloatArray:
    """Bilinear interpolation of ``grid[it, ic]`` (single levels are constant)."""
    if t_levels.size == 1:
        rows = grid[0][None, :] * np.ones((t.size, 1))
    else:
        rows = np.stack(
            [np.interp(t, t_levels, grid[:, j]) for j in range(c_levels.size)], axis=1
        )
    if c_levels.size == 1:
        return np.asarray(rows[:, 0])
    return np.asarray(
        [np.interp(ci, c_levels, row) for ci, row in zip(c, rows, strict=True)]
    )


@dataclass(frozen=True)
class _NormalField:
    contact_model: str
    pressure: dict[str, FloatArray]  # per sample
    wire_stress_change: dict[str, FloatArray]  # per sample, from the gauge zero
    reference_state: CableRadialState | None
    reference_tension_N: float
    notes: tuple[str, ...]
    xi_moduli: Mapping[str, float] | None = None


def _thermal_normal_field(
    cable: CableData,
    protocol: RigProtocol,
    reference_tension_N: float,
    thermal: ThermalHistory | ThermalState,
    zero_tension_N: float,
    levels: int,
) -> _NormalField:
    """Solved normal state per slow-time block of a layer-temperature history.

    Every block state is solved on the tension levels and interpolated over
    the samples of that block; the reference (specs, loops) is the block that
    holds most retained samples.
    """
    tension = protocol.tension_N
    names = [lyr.name for lyr in structural_armour_layers(cable)]
    multi = isinstance(thermal, ThermalHistory) and thermal.time_s.size > 1
    states, index = _record_blocks(
        thermal, tension.size, protocol.time_s if multi else None
    )
    t_levels = _levels(tension, levels)
    pressure = {n: np.zeros(tension.size) for n in names}
    stress = {n: np.zeros(tension.size) for n in names}
    materials = [thermal_material_state(cable, s) for s in states]
    for b, material in enumerate(materials):
        mask = index == b
        if not np.any(mask):
            continue
        if material.radial_inputs:
            solved = solve_cable_radial_states(
                cable, t_levels, **material.radial_inputs
            )
        else:
            solved = [solve_cable_radial_state(cable, float(v)) for v in t_levels]
        for n in names:
            p_lv = np.array([s.armour_contact_pressure()[n] for s in solved])
            s_lv = np.array([s.result.wire_stress_Pa[n] for s in solved])
            if t_levels.size == 1:
                pressure[n][mask] = p_lv[0]
                stress[n][mask] = s_lv[0]
            else:
                pressure[n][mask] = np.interp(tension[mask], t_levels, p_lv)
                stress[n][mask] = np.interp(tension[mask], t_levels, s_lv)
    # Gauge zero: the unloaded specimen at T_ref (relaxed polymer moduli when
    # the history requests relaxation; otherwise the unenergised state).
    zero_inputs = materials[0].reference_radial_inputs
    zero = (
        solve_cable_radial_states(cable, (zero_tension_N,), **zero_inputs)[0]
        if zero_inputs
        else solve_cable_radial_state(cable, zero_tension_N)
    )
    retained = protocol.time_s >= protocol.retain_from_s
    counts = np.bincount(index[retained], minlength=len(states))
    ref_block = int(np.argmax(counts))
    ref_material = materials[ref_block]
    ref = (
        solve_cable_radial_states(
            cable, (reference_tension_N,), **ref_material.radial_inputs
        )[0]
        if ref_material.radial_inputs
        else solve_cable_radial_state(cable, reference_tension_N)
    )
    lo = min(min(s.layer_temperature_C.values()) for s in states)
    hi = max(max(s.layer_temperature_C.values()) for s in states)
    return _NormalField(
        "solved",
        pressure,
        {n: stress[n] - zero.result.wire_stress_Pa[n] for n in names},
        ref,
        reference_tension_N,
        (
            f"thermal history: {len(states)} slow-time block(s), layer "
            f"temperatures {lo:.1f}-{hi:.1f} deg C (one-way coupling; free "
            "thermal strain and registered E(T) in the solved state)",
        ),
        ref_material.xi_moduli_Pa,
    )


def _normal_field(
    cable: CableData,
    protocol: RigProtocol,
    reference_tension_N: float,
    reference_temperature_C: float | None,
    contact_model: str,
    zero_tension_N: float,
    levels: int,
    thermal: ThermalHistory | ThermalState | None = None,
) -> _NormalField:
    tension = protocol.tension_N
    temperature = protocol.temperature_C
    names = [lyr.name for lyr in structural_armour_layers(cable)]
    notes: list[str] = []
    if thermal is not None:
        if temperature is not None:
            raise CableInputError(
                "give a uniform protocol temperature or a thermal history, not both"
            )
        if contact_model == "bounded":
            raise CableInputError(
                "a thermal history needs the solved radial state (contact_model "
                "'auto' or 'solved')"
            )
        return _thermal_normal_field(
            cable, protocol, reference_tension_N, thermal, zero_tension_N, levels
        )
    if contact_model in ("auto", "solved"):
        try:
            t_levels = _levels(tension, levels)
            c_levels = (
                np.array([20.0]) if temperature is None else _levels(temperature, 3)
            )
            pressure = {n: np.zeros((t_levels.size, c_levels.size)) for n in names}
            stress = {n: np.zeros((t_levels.size, c_levels.size)) for n in names}
            for i, t_val in enumerate(t_levels):
                for j, c_val in enumerate(c_levels):
                    state = solve_cable_radial_state(
                        cable,
                        float(t_val),
                        temperature_C=None if temperature is None else float(c_val),
                    )
                    p = state.armour_contact_pressure()
                    for n in names:
                        pressure[n][i, j] = p[n]
                        stress[n][i, j] = state.result.wire_stress_Pa[n]
            zero = solve_cable_radial_state(cable, zero_tension_N)
            ref = solve_cable_radial_state(
                cable, reference_tension_N, temperature_C=reference_temperature_C
            )
            c_hist = np.full(tension.size, 20.0) if temperature is None else temperature
            return _NormalField(
                "solved",
                {
                    n: _interp2(pressure[n], t_levels, c_levels, tension, c_hist)
                    for n in names
                },
                {
                    n: _interp2(stress[n], t_levels, c_levels, tension, c_hist)
                    - zero.result.wire_stress_Pa[n]
                    for n in names
                },
                ref,
                reference_tension_N,
                tuple(notes),
            )
        except CableXSecError as exc:
            if contact_model == "solved":
                raise
            notes.append(
                f"solved radial state unavailable for this cable ({exc}); "
                "bounded pressure chain used"
            )
    if temperature is not None:
        raise CableInputError(
            "a temperature history needs the solved radial state, which this "
            "cable does not support"
        )
    t_levels = _levels(tension, levels)
    pressure_b = {n: np.zeros(t_levels.size) for n in names}
    for i, t_val in enumerate(t_levels):
        for spec, n in zip(
            cable_to_mkappa_specs(cable, float(t_val), contact_model="bounded"),
            names,
            strict=True,
        ):
            pressure_b[n][i] = spec.contact_pressure_Pa
    ea = compute_stiffness(cable).ea_fixed_N
    stress_change = {}
    for spec, n in zip(
        cable_to_mkappa_specs(cable, reference_tension_N, contact_model="bounded"),
        names,
        strict=True,
    ):
        stress_change[n] = (
            spec.E_wire_Pa
            * math.cos(spec.lay_angle_rad) ** 2
            * (tension - zero_tension_N)
            / ea
        )
    return _NormalField(
        "bounded",
        {
            n: (
                np.full(tension.size, pressure_b[n][0])
                if t_levels.size == 1
                else np.interp(tension, t_levels, pressure_b[n])
            )
            for n in names
        },
        stress_change,
        None,
        reference_tension_N,
        tuple(notes),
    )


def _specs(
    cable: CableData, normal: _NormalField, mu: float | None
) -> list[HelicalLayerSpec]:
    if normal.contact_model == "solved":
        specs = cable_to_mkappa_specs(
            cable,
            normal.reference_tension_N,
            mu=mu,
            contact_model="solved",
            radial_state=normal.reference_state,
        )
        if normal.xi_moduli is None:
            return specs
        # Thermal history: stick coefficient with the polymer moduli at the
        # reference block temperatures ([Yun2020] through xi_for_layer).
        return [
            replace(
                spec,
                xi=xi_for_layer(cable, lyr.name, youngs_modulus_Pa=normal.xi_moduli),
            )
            for spec, lyr in zip(specs, structural_armour_layers(cable), strict=True)
        ]
    return cable_to_mkappa_specs(
        cable, normal.reference_tension_N, mu=mu, contact_model="bounded"
    )


def _substeps(
    values: FloatArray, scale: float, max_increment: float
) -> tuple[FloatArray, NDArray[np.bool_], NDArray[np.int64]]:
    """Subdivide a sample path (prefixed by zero) so increments stay small.

    Returns the fine parameter in sample units, the record flags and, per fine
    step, the sample interval index (for linear interpolation of other data).
    """
    path = np.concatenate([[0.0], values])
    counts = np.maximum(
        1, np.ceil(np.abs(np.diff(path)) * scale / max_increment)
    ).astype(int)
    frac = np.concatenate([np.arange(1, c + 1) / c + i for i, c in enumerate(counts)])
    record = np.concatenate([np.r_[np.zeros(c - 1, bool), True] for c in counts])
    return frac, record, np.floor(frac - 1e-12).astype(np.int64)


def _secant_stiffness(
    cable: CableData,
    normal: _NormalField,
    mu: float | None,
    kappa_amp: float,
    core_friction: bool | CoreFrictionSpec | None = None,
) -> float:
    history = kappa_amp * np.sin(np.linspace(0, 4 * math.pi, 161))
    with relayed_warnings():
        if normal.contact_model == "solved":
            loop = compute_mkappa(
                cable,
                history,
                mu=mu,
                radial_state=normal.reference_state,
                core_friction=core_friction,
            )
        else:
            loop = compute_mkappa(
                cable,
                history,
                mu=mu,
                tension_N=normal.reference_tension_N,
                contact_model="bounded",
                core_friction=core_friction,
            )
    return float(loop.EI_apparent_Nm2)


@dataclass(frozen=True)
class _ClampRun:
    wire: dict[int, int]
    theta: dict[int, float]
    friction: dict[int, FloatArray]  # microstrain: sectional + clamp increment
    sectional: dict[int, FloatArray]  # microstrain
    station_moment: list[FloatArray]  # N m, clamp increment
    kappa_c: float
    sigma_u: float
    sigma0: float


def _clamp_run(
    spec: HelicalLayerSpec,
    hand: float,
    setback: float,
    *,
    length: float,
    shape: Callable[[FloatArray], FloatArray],
    kappa_ref: FloatArray,
    pressure: FloatArray,
    gauges: list[tuple[int, GaugeLocation]],
    stations: list[float],
    anchored: bool,
    segments_per_pitch: int,
    max_increment: float,
    field: CurvatureField | None = None,
) -> _ClampRun:
    """Sectional friction and anchored-wire increment for one layer and variant.

    With ``field`` the curvature is the non-proportional field
    ``kappa(x, t)`` (gap P5) instead of ``kappa_ref(t) shape(x)``.
    """
    pitch = 2 * math.pi * spec.r_helix_m / math.tan(spec.lay_angle_rad)
    domain = length + 2 * setback
    n_seg = max(8, int(math.ceil(domain / pitch * segments_per_pitch)))
    h_x = domain / n_seg
    h = 2 * math.pi * h_x / pitch
    x_seg = (np.arange(n_seg) + 0.5) * h_x - setback
    shape_seg = shape(np.asarray(x_seg, dtype=np.float64))
    g_seg = np.array(
        [min(n_seg - 1, int((g.x_m + setback) / h_x)) for _, g in gauges],
        dtype=np.int64,
    )
    s_seg = np.array(
        [min(n_seg - 1, int((s + setback) / h_x)) for s in stations], dtype=np.int64
    )
    index = np.concatenate([g_seg, s_seg])
    theta0 = 2 * math.pi * np.arange(spec.n_wires) / spec.n_wires
    theta_seg = (
        theta0[:, None] + hand * 2 * math.pi * (x_seg[None, :] + setback) / pitch
    )

    kappa_c = slip_onset_curvature(spec)
    sin_a = abs(math.sin(spec.lay_angle_rad))
    width = 2 * math.pi * spec.r_helix_m / spec.n_wires
    p_ref = spec.contact_pressure_Pa
    sigma0 = (
        spec.r_helix_m
        * p_ref
        * width
        * math.cos(spec.lay_angle_rad)
        / (spec.A_wire_m2 * sin_a**2)
    )
    sigma_u = sigma0 * spec.mu * sin_a
    scale_strain = 1e6 / spec.E_wire_Pa

    increment = np.zeros((spec.n_wires, kappa_ref.size, index.size))
    if field is not None and anchored and kappa_c > 0 and p_ref > 0:
        inside = (x_seg >= 0.0) & (x_seg <= length)
        drive = np.zeros((kappa_ref.size, n_seg))
        drive[:, inside] = field.sample(x_seg[inside]).T / kappa_c
        cap_rows = np.arcsin(np.clip(np.abs(np.cos(theta_seg)), 0.0, 1.0))
        for wire in range(spec.n_wires):
            n_rec, ref_rec, _ = anchored_wire_field_response(
                drive,
                np.cos(theta_seg[wire]),
                h,
                max_increment=max_increment,
                rho=pressure / p_ref,
                cap_hat=cap_rows[wire],
                index=index,
            )
            increment[wire] = n_rec - ref_rec
    elif anchored and kappa_c > 0 and p_ref > 0:
        e_rows = shape_seg[None, :] * np.cos(theta_seg)
        cap_rows = np.arcsin(np.clip(np.abs(np.cos(theta_seg)), 0.0, 1.0))
        scale = float(np.max(np.abs(shape_seg))) / kappa_c
        frac, record, interval = _substeps(kappa_ref, scale, max_increment)
        kap_path = np.concatenate([[0.0], kappa_ref])
        p_path = np.concatenate([[pressure[0]], pressure])
        w = frac - interval
        kap_fine = kap_path[interval] * (1 - w) + kap_path[interval + 1] * w
        rho_fine = (p_path[interval] * (1 - w) + p_path[interval + 1] * w) / p_ref
        for wire in range(spec.n_wires):
            n_rec, ref_rec, _ = anchored_wire_response(
                e_rows[wire],
                kap_fine / kappa_c,
                h,
                rho=rho_fine,
                cap_hat=cap_rows[wire],
                record=record,
                index=index,
            )
            increment[wire] = n_rec - ref_rec

    wires: dict[int, int] = {}
    thetas: dict[int, float] = {}
    friction: dict[int, FloatArray] = {}
    sectional: dict[int, FloatArray] = {}
    for k, (i, g) in enumerate(gauges):
        target = math.radians(g.angle_deg)
        diff = np.angle(np.exp(1j * (theta_seg[:, g_seg[k]] - target)))
        wire = int(np.argmin(np.abs(diff)))
        theta_g = float(np.angle(np.exp(1j * theta_seg[wire, g_seg[k]])))
        kx_local = (
            kappa_ref * float(shape(np.array([g.x_m]))[0])
            if field is None
            else field.at(g.x_m)
        )
        first = np.flatnonzero(kx_local)
        start = -theta_g + (math.pi if first.size and kx_local[first[0]] < 0 else 0.0)
        # rotated frame: wire 0 of the sectional law sits at the gauge angle
        stress = wire_friction_stress(
            spec,
            kx_local * math.cos(theta_g),
            -kx_local * math.sin(theta_g),
            contact_pressure_Pa=pressure,
            initial_bending_angle=start,
        )[0]
        wires[i] = wire
        thetas[i] = theta_g
        sectional[i] = np.asarray(stress * scale_strain)
        friction[i] = np.asarray(
            sectional[i] + sigma_u * increment[wire, :, k] * scale_strain
        )
    lever = spec.A_wire_m2 * spec.r_helix_m * math.cos(spec.lay_angle_rad)
    moments = [
        np.asarray(
            sigma_u
            * lever
            * (
                np.cos(theta_seg[:, seg])[:, None] * increment[:, :, g_seg.size + j]
            ).sum(axis=0)
        )
        for j, seg in enumerate(s_seg)
    ]
    return _ClampRun(
        wires, thetas, friction, sectional, moments, kappa_c, sigma_u, sigma0
    )


# --------------------------------------------------------------------------- #
# Main entry point                                                             #
# --------------------------------------------------------------------------- #
def simulate_rig_test(
    cable: CableData,
    protocol: RigProtocol,
    setup: RigSetup,
    *,
    gauges: Sequence[GaugeLocation] | None = None,
    stations_m: Sequence[float] | None = None,
    mu: float | None = None,
    mu_factors: Sequence[float] = (0.75, 1.25),
    contact_model: str = "auto",
    gauge_resolution_microstrain: float = DEFAULT_GAUGE_RESOLUTION_MICROSTRAIN,
    gauge_zero_tension_N: float = 0.0,
    segments_per_pitch: int = 64,
    max_increment: float = 0.5,
    tension_levels: int = 5,
    setback_variants_m: Sequence[float] = (),
    core_friction: bool | CoreFrictionSpec | None = None,
    thermal: ThermalHistory | ThermalState | None = None,
) -> RigTestResult:
    r"""Simulate a tension--bending rig test and plan armour strain gauges.

    Parameters
    ----------
    cable : CableData
        Fully specified cable (public definitions only for shared results).
    protocol : RigProtocol
        Tension, curvature or end-rotation and optional temperature histories.
    setup : RigSetup
        Free length, curvature profile and clamp condition.
    gauges : sequence of GaugeLocation, optional
        Candidate gauges (default :func:`default_gauge_grid`).
    stations_m : sequence of float, optional
        Stations for moment--curvature loops (default clamp face and mid-span).
    mu : float, optional
        Nominal armour friction coefficient (default: material ``mu_on_hdpe``).
    mu_factors : sequence of float
        Multipliers of the nominal ``mu`` for the sensitivity runs.
    contact_model : {"auto", "solved", "bounded"}
        Normal state; ``"auto"`` uses the solved radial state where the cable
        supports it, else the bounded chain.
    gauge_resolution_microstrain : float
        Assumed gauge resolution used to judge signals
        (default :data:`DEFAULT_GAUGE_RESOLUTION_MICROSTRAIN`).
    gauge_zero_tension_N : float
        Tension at which the gauges are zeroed (straight specimen, 20 deg C).
    segments_per_pitch : int
        Clamp-solver segments per pitch (64 converges the clamp peak to < 1 %).
    max_increment : float
        Largest clamp-solver load increment in units of ``kappa_c``.
    tension_levels : int
        Solved normal states across the tension range (linear interpolation).
    setback_variants_m : sequence of float
        Additional anchor set-backs [m] for the clamp sensitivity
        (``GaugeResult.range_by_setback``; nominal mu).
    core_friction : bool or CoreFrictionSpec, optional
        Core stick-slip in the sectional moment--curvature law
        (:func:`~cablexsec.api.compute_mkappa`): the station loops and the
        secant stiffness of the end-rotation decay length. Armour gauge
        strains are curvature driven and do not change. Default: armour only.
    thermal : ThermalHistory or ThermalState, optional
        Layer-temperature profile of an energised specimen (for example
        ``ThermalHistory.steady(cable, current_A)`` or a transient history
        from a current programme). Replaces the uniform
        ``protocol.temperature_C``: each slow-time block is solved with its
        free thermal strains and registered polymer ``E(T)``, giving the
        clamping, friction caps and the thermal part of the axial gauge
        strain; the stick coefficient uses the polymer moduli of the
        reference block. See :mod:`cablexsec.analysis.thermal_coupling`.

    Notes
    -----
    The friction laws depend on ``mu`` and ``P1+P2`` almost only through their
    product (exactly for the linearised clamp increment), so the ``mu``
    sensitivity also stands for the sensitivity to the clamping pressure.

    Returns
    -------
    RigTestResult
    """
    require_fully_specified(cable, "simulate_rig_test")
    if contact_model not in _CONTACT_MODELS:
        raise CableInputError(f"contact_model must be one of {_CONTACT_MODELS}")
    for name, value in (
        ("gauge_resolution_microstrain", gauge_resolution_microstrain),
        ("max_increment", max_increment),
    ):
        require_finite(value, name)
        if value <= 0:
            raise CableInputError(f"{name} must be positive")
    if segments_per_pitch < 8 or tension_levels < 2:
        raise CableInputError("need segments_per_pitch >= 8 and tension_levels >= 2")
    factors = tuple(float(f) for f in mu_factors)
    if any(not math.isfinite(f) or f <= 0 for f in factors):
        raise CableInputError("mu_factors must be positive")
    setback_list = tuple(float(s) for s in setback_variants_m)
    if any(not math.isfinite(s) or s < 0 for s in setback_list):
        raise CableInputError("setback_variants_m must be non-negative")
    if protocol.end_rotation_rad is not None and setup.profile != "end_rotation":
        raise CableInputError("an end-rotation history needs the end_rotation profile")

    time = protocol.time_s
    retained = time >= protocol.retain_from_s
    length = setup.free_length_m
    field = setup.curvature_field
    if field is not None and field.n_samples != time.size:
        raise CableInputError(
            "the curvature field needs one column per protocol sample"
        )
    t_ref = float(np.mean(protocol.tension_N[retained]))
    temp_ref = (
        None
        if protocol.temperature_C is None
        else float(np.mean(protocol.temperature_C[retained]))
    )
    normal = _normal_field(
        cable,
        protocol,
        t_ref,
        temp_ref,
        contact_model,
        gauge_zero_tension_N,
        tension_levels,
        thermal,
    )
    notes = list(normal.notes)
    armours = structural_armour_layers(cable)
    nominal_specs = _specs(cable, normal, mu)
    mu_nominal = {lyr.name: s.mu for lyr, s in zip(armours, nominal_specs, strict=True)}

    # ----------------------------------------------------------- curvature shape
    def shape_at(x: FloatArray, decay: float | None) -> FloatArray:
        x = np.asarray(x, dtype=np.float64)
        inside = (x >= 0) & (x <= length)
        if setup.profile == "uniform":
            raw = np.ones_like(x)
        elif setup.profile == "end_rotation":
            assert decay is not None
            raw = end_rotation_profile(np.clip(x, 0, length), length, decay)
        elif field is not None:
            # envelope |kappa(x)| max over time, signed as at the reference
            raw = np.max(np.abs(field.sample(np.clip(x, 0, length))), axis=1)
        else:
            assert setup.profile_x_m is not None and setup.profile_shape is not None
            raw = np.interp(x, setup.profile_x_m, setup.profile_shape)
        return np.asarray(np.where(inside, raw, 0.0))

    if setup.reference_x_m is not None:
        x_ref = setup.reference_x_m
    elif setup.profile == "uniform":
        x_ref = length / 2
    elif setup.profile == "end_rotation":
        x_ref = 0.0
    elif field is not None:
        inside = field.x_m[field.x_m <= length]
        envelope = np.max(np.abs(field.sample(inside)), axis=1)
        x_ref = float(inside[int(np.argmax(envelope))])
    else:
        assert setup.profile_x_m is not None and setup.profile_shape is not None
        x_ref = float(setup.profile_x_m[int(np.argmax(np.abs(setup.profile_shape)))])

    ei_used: float | None = None
    decay: float | None = None
    rotation_gain = 1.0
    if setup.profile == "end_rotation":
        if t_ref <= 0:
            decay = math.inf
            rotation_gain = float(
                end_rotation_profile(np.array([x_ref]), length, decay)[0]
            )
        else:
            ei = setup.bending_stiffness_Nm2
            fixed = ei is not None
            if ei is None:
                ei = float(compute_stiffness(cable).ei_max_Nm2)
            drive = (
                protocol.end_rotation_rad
                if protocol.end_rotation_rad is not None
                else protocol.curvature_1_m
            )
            assert drive is not None
            amp = float(np.max(np.abs(drive[retained])))
            for _ in range(0 if fixed else 8):
                decay = math.sqrt(ei / t_ref)
                gain0 = abs(
                    float(end_rotation_profile(np.array([0.0]), length, decay)[0])
                )
                gain_ref = abs(
                    float(end_rotation_profile(np.array([x_ref]), length, decay)[0])
                )
                peak = (
                    amp * gain0
                    if protocol.end_rotation_rad is not None
                    else (amp * gain0 / max(gain_ref, 1e-12 * gain0))
                )
                new = _secant_stiffness(
                    cable, normal, mu, max(peak, 1e-6), core_friction
                )
                converged = abs(new - ei) <= 1e-3 * ei
                ei = new
                if converged:
                    break
            decay = math.sqrt(ei / t_ref)
            rotation_gain = float(
                end_rotation_profile(np.array([x_ref]), length, decay)[0]
            )
            ei_used = ei
    norm = float(shape_at(np.array([x_ref]), decay)[0])
    if norm == 0:
        raise CableInputError("the curvature shape vanishes at the reference station")

    def shape(x: FloatArray) -> FloatArray:
        return np.asarray(shape_at(x, decay) / norm)

    if protocol.end_rotation_rad is not None:
        kappa_ref = np.asarray(protocol.end_rotation_rad * rotation_gain)
    else:
        assert protocol.curvature_1_m is not None
        kappa_ref = np.asarray(protocol.curvature_1_m)
    if field is not None:
        at_ref = field.at(x_ref)
        scale_ref = float(np.max(np.abs(at_ref))) or 1.0
        if float(np.max(np.abs(at_ref - kappa_ref))) > 1e-9 * scale_ref:
            raise CableInputError(
                "protocol curvature_1_m must be the curvature field at the "
                f"reference station x = {x_ref:.6g} m "
                "(RigProtocol.from_curvature_field)"
            )
    x_plot = np.linspace(0.0, length, 201)
    shape_plot = shape(x_plot)

    # ------------------------------------------------------------- gauges
    candidates = (
        list(gauges) if gauges is not None else default_gauge_grid(cable, setup)
    )
    names = {lyr.name for lyr in armours}
    for g in candidates:
        if g.layer not in names:
            raise CableInputError(
                f"gauge layer {g.layer!r} is not a structural armour layer"
            )
        if not (0 <= g.x_m <= length and math.isfinite(g.angle_deg)):
            raise CableInputError("gauge positions must lie on the free length")
    station_list = (
        [0.0, length / 2] if stations_m is None else [float(s) for s in stations_m]
    )
    if any(not 0 <= s <= length for s in station_list):
        raise CableInputError("stations must lie on the free length")

    mu_values = tuple(sorted({1.0, *factors}))
    nominal_setback = setup.clamp_setback_m
    setbacks = tuple(sorted({nominal_setback, *setback_list}))
    nominal_key = (1.0, nominal_setback)
    runs: dict[tuple[str, float, float], _ClampRun] = {}
    scales: list[LayerScales] = []
    station_moment = [np.zeros(time.size) for _ in station_list]
    a_max = float(np.max(np.abs(kappa_ref)))

    for lyr, spec_nom in zip(armours, nominal_specs, strict=True):
        layer_gauges = [(i, g) for i, g in enumerate(candidates) if g.layer == lyr.name]
        pressure = normal.pressure[lyr.name]
        for factor in mu_values:
            spec = (
                spec_nom
                if factor == 1.0
                else replace(spec_nom, mu=spec_nom.mu * factor)
            )
            for sb in setbacks:
                if factor != 1.0 and sb != nominal_setback:
                    continue
                run = _clamp_run(
                    spec,
                    1.0 if lyr.lay_direction == "right_hand" else -1.0,
                    sb,
                    length=length,
                    shape=shape,
                    kappa_ref=kappa_ref,
                    pressure=pressure,
                    gauges=layer_gauges,
                    stations=station_list,
                    anchored=setup.clamp == "anchored",
                    segments_per_pitch=segments_per_pitch,
                    max_increment=max_increment,
                    field=field,
                )
                runs[(lyr.name, factor, sb)] = run
                if setup.clamp == "anchored" and run.kappa_c <= 0:
                    notes.append(
                        f"{lyr.name}: no friction-setting pressure; no clamp effect"
                    )
        nominal = runs[(lyr.name, *nominal_key)]
        for j in range(len(station_list)):
            station_moment[j] += nominal.station_moment[j]
        pitch = 2 * math.pi * spec_nom.r_helix_m / math.tan(spec_nom.lay_angle_rad)
        kappa_c = nominal.kappa_c
        a_peak = (
            a_max * float(np.max(np.abs(shape(np.linspace(0, length, 401))))) / kappa_c
            if kappa_c > 0
            else math.inf
        )
        a_clamp = (
            a_max * abs(float(shape(np.array([0.0]))[0])) / kappa_c
            if kappa_c > 0
            else math.inf
        )
        unit = nominal.sigma_u / spec_nom.E_wire_Pa * 1e6
        sin_a = abs(math.sin(spec_nom.lay_angle_rad))
        scales.append(
            LayerScales(
                layer=lyr.name,
                pitch_m=pitch,
                lay_direction=lyr.lay_direction,
                n_wires=spec_nom.n_wires,
                contact_pressure_Pa=spec_nom.contact_pressure_Pa,
                kappa_c_1_m=kappa_c,
                friction_unit_microstrain=unit,
                sectional_cap_microstrain=nominal.sigma0
                * math.expm1(spec_nom.mu * sin_a * math.pi / 2)
                / spec_nom.E_wire_Pa
                * 1e6,
                axial_microstrain_at_reference=float(
                    np.interp(
                        t_ref,
                        np.sort(protocol.tension_N),
                        normal.wire_stress_change[lyr.name][
                            np.argsort(protocol.tension_N)
                        ],
                    )
                )
                / spec_nom.E_wire_Pa
                * 1e6,
                a_peak=a_peak,
                a_clamp=a_clamp,
                slip_length_m=a_peak * pitch / (2 * math.pi),
                boundary_layer_m=float(clamp_boundary_layer_length(a_peak, pitch)),
                setback_length_m=float(clamp_setback_length(a_peak, pitch)),
                clamp_law_amplification=float(clamp_amplification(a_clamp)),
                clamp_law_peak_microstrain=(
                    min(a_clamp, math.pi / 2) + float(clamp_mean_force(a_clamp))
                )
                * unit,
            )
        )

    # ------------------------------------------------------------- assemble gauges
    def rng(values: FloatArray) -> float:
        v = values[retained]
        return float(v.max() - v.min())

    results: list[GaugeResult] = []
    spec_by_name = {lyr.name: s for lyr, s in zip(armours, nominal_specs, strict=True)}
    for i, g in enumerate(candidates):
        spec = spec_by_name[g.layer]
        kx_local = (
            kappa_ref * float(shape(np.array([g.x_m]))[0])
            if field is None
            else field.at(g.x_m)
        )
        diameter = 2 * math.sqrt(spec.A_wire_m2 / math.pi)
        axial = normal.wire_stress_change[g.layer] / spec.E_wire_Pa * 1e6

        def bending(
            theta: float,
            kx: FloatArray = kx_local,
            sp: HelicalLayerSpec = spec,
            d: float = diameter,
        ) -> FloatArray:
            stress = local_bending_fibre_stress(
                kx, 0.0, theta, sp.E_wire_Pa, d, sp.lay_angle_rad, 0.0
            )
            return np.asarray(stress / sp.E_wire_Pa * 1e6)

        nominal = runs[(g.layer, *nominal_key)]
        theta_g = nominal.theta[i]
        bend = bending(theta_g)
        friction = nominal.friction[i]
        total = axial + bend + friction
        by_mu = {
            mu_nominal[g.layer] * f: rng(
                axial + bend + runs[(g.layer, f, nominal_setback)].friction[i]
            )
            for f in mu_values
        }
        by_setback = {
            sb: rng(
                axial
                + bending(runs[(g.layer, 1.0, sb)].theta[i])
                + runs[(g.layer, 1.0, sb)].friction[i]
            )
            for sb in setbacks
        }
        results.append(
            GaugeResult(
                location=g,
                wire=nominal.wire[i],
                angle_deg=float(np.degrees(np.angle(np.exp(1j * theta_g)))),
                axial=_frozen(axial),
                local_bending=_frozen(bend),
                friction=_frozen(friction),
                friction_sectional=_frozen(nominal.sectional[i]),
                total=_frozen(total),
                ranges={
                    "axial": rng(axial),
                    "local_bending": rng(bend),
                    "friction": rng(friction),
                    "total": rng(total),
                },
                range_by_mu=by_mu,
                range_sectional=rng(axial + bend + nominal.sectional[i]),
                range_by_setback=by_setback,
            )
        )

    for lyr in armours:
        try:
            yield_pa = library.get_property(lyr.material, "sigma_y_Pa")
        except (KeyError, CableXSecError):
            continue
        peak = max(
            (
                float(np.max(np.abs(r.total)))
                for r in results
                if r.location.layer == lyr.name
            ),
            default=0.0,
        )
        peak_mpa = peak * 1e-12 * spec_by_name[lyr.name].E_wire_Pa
        if peak_mpa * 1e6 > yield_pa:
            notes.append(
                f"{lyr.name}: largest gauge stress change {peak_mpa:.0f} MPa exceeds "
                f"the wire yield strength {yield_pa / 1e6:.0f} MPa; the elastic "
                "model is outside its range there"
            )

    # ------------------------------------------------------------- stations
    loops: list[StationLoop] = []
    for j, x_s in enumerate(station_list):
        kap = (
            kappa_ref * float(shape(np.array([x_s]))[0])
            if field is None
            else field.at(x_s)
        )
        if np.ptp(kap) == 0:
            m_sec = np.zeros(time.size)
        else:
            with relayed_warnings():
                if normal.contact_model == "solved":
                    loop = compute_mkappa(
                        cable,
                        kap,
                        mu=mu,
                        radial_state=normal.reference_state,
                        core_friction=core_friction,
                    )
                else:
                    loop = compute_mkappa(
                        cable,
                        kap,
                        mu=mu,
                        tension_N=t_ref,
                        contact_model="bounded",
                        core_friction=core_friction,
                    )
            m_sec = np.asarray(loop.moment_Nm)
        loops.append(
            StationLoop(
                x_m=x_s,
                kappa_1_m=_frozen(kap),
                moment_sectional_Nm=_frozen(m_sec),
                moment_clamp_Nm=_frozen(station_moment[j]),
                moment_Nm=_frozen(m_sec + station_moment[j]),
            )
        )

    recommendations = _recommend(results, scales, length, gauge_resolution_microstrain)
    if setup.clamp == "none":
        notes.append("clamp='none': sectional (unanchored) result only")
    if field is not None:
        notes.append(
            f"non-proportional curvature field ({field.kind}); the clamp "
            "solver follows kappa(x, t) segment by segment"
        )
        notes.extend(field.notes)
    provenance = Provenance(
        method=(
            "rig-test simulation: solved (or bounded) normal state interpolated over "
            "the protocol; sectional direction-tied friction and outward-fibre local "
            "bending at the local curvature; anchored-wire clamp increment "
            "(rigid stick, Coulomb, exact chain projection)"
        ),
        citations=(
            "[Papailiou1997]",
            "[KraincanicKebadze2001] (far-field slip front)",
            "[Lutchansky1969] (elastic clamp limit)",
            "[RaoofHobbs1984] (clamp slip at the neutral axis)",
            "[ZhuTan2014] (shifted friction field at an end fitting)",
            "[MenardCartraud2023] Eq. 14 (local bending)",
        ),
        gates=(
            "tests/test_clamp_layer.py: projection vs QP, Kraincanic-Kebadze far "
            "field, stick regime, energy identity, clamp law at a = 100",
            "tests/test_rig_test.py: far-field and clamp='none' limits equal the "
            "sectional package result",
        ),
        limitations=(
            "clamp model: axial slip only, rigid stick, constant bound along the "
            "wire, one wire per layer without layer interaction, planar proportional "
            "curvature; not verified against helix FE or a test",
            "clamp increment uses the linearised Papailiou cap; the sectional part "
            "uses the package's signed Papailiou bounds",
            "moment-curvature loops use the mean protocol tension; the end-rotation "
            "profile is frozen at the mean tension and secant stiffness",
            "gauge strain is mechanical wire strain on the outward fibre, zeroed at "
            "gauge_zero_tension_N, 20 deg C and zero curvature (temperature-"
            "compensated gauge assumed)",
            "the clamp idealisation (anchor plane, set-back) controls the near-clamp "
            "result; a bonded length inside a fitting is not frictional",
            "transverse slip, wire flattening and bending-induced pressure "
            "redistribution are not modelled",
        ),
    )
    return RigTestResult(
        time_s=time,
        retained=retained,
        x_m=_frozen(x_plot),
        curvature_shape=_frozen(shape_plot),
        reference_curvature_1_m=_frozen(kappa_ref),
        reference_x_m=float(x_ref),
        bending_stiffness_Nm2=ei_used,
        mu_nominal=mu_nominal,
        mu_values=mu_values,
        contact_model=normal.contact_model,
        layers=tuple(scales),
        stations=tuple(loops),
        gauges=tuple(results),
        recommendations=recommendations,
        gauge_resolution_microstrain=gauge_resolution_microstrain,
        provenance=provenance,
        notes=tuple(notes),
    )


def _recommend(
    gauges: list[GaugeResult],
    scales: list[LayerScales],
    length: float,
    resolution: float,
) -> tuple[GaugeRecommendation, ...]:
    """Per layer: clamp boundary layer, far field and friction calibration."""
    layer_scale = {s.layer: s for s in scales}
    out: list[GaugeRecommendation] = []
    used: set[int] = set()

    def add(purpose: str, pool: list[GaugeResult], key: str, reason: str) -> None:
        pool = [g for g in pool if id(g) not in used]
        if not pool:
            return
        metric = {
            "contrast": lambda g: abs(g.clamp_contrast),
            "mu": lambda g: abs(g.mu_sensitivity),
            "total": lambda g: g.ranges["total"],
        }[key]
        best = max(pool, key=metric)
        used.add(id(best))
        amp = best.ranges["total"] / 2
        if best.ranges["total"] < DETECTION_FACTOR * resolution:
            reason += " Signal below three gauge resolutions."
        out.append(
            GaugeRecommendation(
                purpose=purpose,
                gauge=best,
                amplitude_microstrain=amp,
                signal_to_resolution=amp / resolution,
                reason=reason,
            )
        )

    def near_clamp(g: GaugeResult) -> bool:
        s = layer_scale[g.location.layer]
        return g.location.x_m <= max(0.5 * s.pitch_m, s.boundary_layer_m)

    def neutral(g: GaugeResult) -> bool:
        return abs(abs(_wrap_deg(g.angle_deg)) - 90.0) <= 30.0

    def extreme(g: GaugeResult) -> bool:
        return (
            abs(_wrap_deg(g.angle_deg)) <= 30.0 or abs(_wrap_deg(g.angle_deg)) >= 150.0
        )

    for layer in layer_scale:
        own = [g for g in gauges if g.location.layer == layer]
        if not own:
            continue
        mid = [g for g in own if abs(g.location.x_m - length / 2) <= 0.05 * length]
        add(
            f"{layer}: clamp boundary layer",
            [g for g in own if near_clamp(g) and neutral(g)],
            "contrast",
            "Wire near the neutral axis close to the clamp: outward-fibre local "
            "bending and sectional friction strain are small, so the reading is "
            "dominated by the friction-fed mean force of the anchored wire "
            "(clamp law); compare with the same gauge without the clamp.",
        )
        add(
            f"{layer}: far field, extreme fibre",
            [g for g in mid if extreme(g)] or mid,
            "total",
            "Extreme-fibre wire at mid-span: largest local-bending and friction "
            "ranges at the sectional (constant-curvature) state; check the clamp "
            "contrast to confirm the station is outside the clamp layer.",
        )
        add(
            f"{layer}: friction calibration",
            [g for g in own if not near_clamp(g)] or own,
            "mu",
            "Largest change of strain range with the friction coefficient "
            "(equivalently with the clamping pressure) outside the clamp layer, "
            "where the sectional law alone applies.",
        )
    return tuple(out)
