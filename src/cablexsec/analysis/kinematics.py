# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Kinematics of the cable under a curvature and tension history.

:func:`kinematic_history` follows a cable through a history of the
axial-strain-gradient curvature ``(kx, ky)`` (fibre strain
``eps0 + kx x + ky y``) and the force on the armour, with the friction law of
the stress route, and returns per sample the curvature, the axial strain and
the twist of the section, and per helical population (each armour layer; the
laid cores when their stage is on) the axial slip of the wire along its helix
relative to the stuck state, its state (stick or slip), its stress and its
utilisation. Every value comes from the same calculation as the wire stress
of :func:`cablexsec.analysis.fatigue.wire_stress_field`.

Slip along the helix
--------------------
Under a curvature uniform along the cable the state of a wire depends on its
azimuth ``phi`` only: a wire passes through every azimuth along one pitch, and
the population of a section samples that field. Each field is therefore held
at stations round the section (the wires, or the helix nodes of the
distributed law) and read anywhere by periodic linear interpolation in
``phi``.

The direction-tied and progressive laws are spring-sliders on the reduced
axial-strain drive ``d = r (kx cos phi + ky sin phi)``: the friction stress is
``sigma_f = k_f (d - p)`` with ``k_f = xi E cos^2 a`` and ``p`` the plastic
(Coulomb) part of the drive (the mean over the sub-elements of the
progressive law). In wire-strain units the Coulomb slip strain is
``gamma = cos^2 a p`` (``sigma_f (1 - xi) / (xi E)`` is the elastic shear of
the interface, not slip), so ``p = d - sigma_f / k_f`` is recovered from the
stress the law gives. Along the wire ``ds = r dphi / |sin a|`` and the slip
displacement of the wire from its stuck position, positive along the wire
towards increasing ``z``, is

.. math::

    u(\phi) = -s \frac{r}{|\sin a|} \int_0^{\phi} \gamma \, d\phi'
              + C, \qquad s = \pm 1 \ \text{(right- or left-hand lay)},

integrated exactly over the piecewise-linear field between the stations
(trapezoids). A uniform curvature cannot slide a wire as a whole over a
pitch, so the pitch mean of ``gamma`` (zero in planar bending by symmetry;
round-off or a turning curvature otherwise) is taken out before the
integration, which keeps ``u`` periodic; the largest share taken out is kept
as ``mean_slip_share``. The pointwise law gives the slip strain, not the
constant ``C``; it is fixed step by step by the stick zones, where the wire
does not move on its bed: over each step ``C`` changes so that the stations
in stick (a station whose slip strain did not change, with both neighbours)
keep their displacement on average, and so that the pitch mean of ``u`` keeps
its value when no station sticks. In planar bending the stick zones round the
two extreme fibres then stay exactly in place. From the stress-free start
``p = 0`` and ``u = 0`` until a wire slips. In frictionless slip
``sigma_f = 0``, ``p = d`` and
``u = -s (r^2 cos^2 a / |sin a|) (kx sin phi - ky cos phi)``: the full-slip
kinematic slip of [KraincanicKebadze2001], ``r^2 cos^2 a / sin a`` times the
curvature.

The distributed law solves the displacement ``u`` and the slip ``p`` along the
helix itself (:mod:`cablexsec.mechanics.distributed_friction`); its nodal slip
``p`` (the Coulomb part of ``u``) is taken as it is, in metres
(``parameters["length_unit_m"]``), with no reconstruction.

A station is slipping at a sample when its slip changed over the step to that
sample (the plastic drive, or the distributed nodal slip).

Section
-------
The axial strain is the force on the armour over the axial stiffness with the
twist held, ``eps0 = T / EA_fixed`` (as the stress route), or a prescribed
strain; the ends are held against twist, so the twist is zero. The centre-line
wire stress is ``E cos^2 a eps0 + sigma_f``; the utilisation is the largest
stress round the wire (with its local bending,
:func:`~cablexsec.mechanics.wire_stress.local_bending_fibre_stress`) over the
yield stress of the armour material.

The transverse slip of the wire from its loxodrome
(:class:`~cablexsec.mechanics.tendon_bending.TransverseSlipModel`) is added
when asked for; its local bending then follows that model, as in the stress
route.

References
----------
[KraincanicKebadze2001] full-slip axial slip; [Papailiou1997] slip cap;
[Saevik1992] transverse slip.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from cablexsec._numba import HAVE_NUMBA
from cablexsec.cables.schema import CableData, HelicalLayer
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.materials import library
from cablexsec.mechanics.core_friction import (
    CoreFrictionSpec,
    build_core_friction,
    resolve_core_friction,
)
from cablexsec.mechanics.distributed_friction import distributed_friction_history
from cablexsec.mechanics.fast_slip import wire_friction_stress
from cablexsec.mechanics.mkappa import HelicalLayerSpec
from cablexsec.mechanics.papailiou import wire_slip_history
from cablexsec.mechanics.progressive_friction import (
    ProgressiveFriction,
    progressive_wire_friction_stress,
)
from cablexsec.mechanics.tendon_bending import TransverseSlipModel
from cablexsec.mechanics.tendon_section import WireSection, round_section
from cablexsec.mechanics.wire_stress import local_bending_fibre_stress
from cablexsec.provenance import Provenance, real_array

if TYPE_CHECKING:
    from cablexsec.analysis.stress import FrictionField

__all__ = [
    "FRICTION_LAWS",
    "HelixField",
    "KinematicHistory",
    "armour_transverse_model",
    "friction_field_of",
    "kinematic_history",
    "kinematic_slip_amplitude",
    "periodic_interpolate",
    "slip_along_helix",
]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

FrictionLaw = Literal["direction_tied", "distributed", "progressive"]
FRICTION_LAWS: tuple[str, ...] = ("direction_tied", "distributed", "progressive")

SLIP_TOLERANCE = 1.0e-9
"""A station slips when its plastic drive changes by more than this share of
the largest drive ``r |kappa|`` of the history over the step."""

_TWO_PI = 2.0 * math.pi
_BLOCK = 4096
"""Samples followed at a time by the spring-slider slip (memory)."""


def kinematic_slip_amplitude(
    r_helix_m: float, lay_angle_rad: float, curvature_1_m: float
) -> float:
    """Full-slip axial slip amplitude of a frictionless wire [m].

    ``r^2 cos^2 a / |sin a|`` times the curvature ([KraincanicKebadze2001]):
    the largest displacement along the wire, at the neutral axis, from its
    stuck (plane-section) position.
    """
    sin_a = abs(math.sin(lay_angle_rad))
    if sin_a < 1.0e-12:
        raise CableInputError("the slip needs a lay angle above zero")
    return r_helix_m**2 * math.cos(lay_angle_rad) ** 2 / sin_a * abs(curvature_1_m)


def _sorted_stations(azimuth: FloatArray) -> tuple[IntArray, FloatArray, FloatArray]:
    """Order of the stations round the section, their azimuths and gaps."""
    wrapped = np.mod(np.asarray(azimuth, dtype=np.float64), _TWO_PI)
    order = np.argsort(wrapped, kind="stable").astype(np.int64)
    phi = wrapped[order]
    gaps = np.diff(np.append(phi, phi[0] + _TWO_PI))
    if np.any(gaps <= 0.0):
        raise CableInputError("the stations must have distinct azimuths")
    return order, phi, gaps


def slip_along_helix(
    azimuth_rad: FloatArray,
    slip_strain: FloatArray,
    r_helix_m: float,
    lay_angle_rad: float,
    lay_sign: int = 1,
) -> tuple[FloatArray, FloatArray]:
    """Slip displacement of a wire along its helix from its slip strain.

    ``slip_strain`` (stations, samples) is the Coulomb slip strain ``gamma``
    at the station azimuths. Returns the displacement ``u`` [m] along the
    wire (positive towards increasing ``z``), periodic with zero pitch mean,
    and per sample the share of the largest ``|gamma|`` taken out as the pitch
    mean (module notes).
    """
    gamma = np.asarray(slip_strain, dtype=np.float64)
    if gamma.ndim == 1:
        gamma = gamma[:, None]
    order, _phi, gaps = _sorted_stations(azimuth_rad)
    u, share = _helix_integral(gamma[order], gaps, r_helix_m, lay_angle_rad, lay_sign)
    u = u - _pitch_mean(u, gaps)[None, :]
    out = np.empty_like(u)
    out[order] = u
    return out, share


def _pitch_mean(values: FloatArray, gaps: FloatArray) -> FloatArray:
    """Mean over the pitch of a field (sorted stations, samples)."""
    pieces = 0.5 * (values + np.roll(values, -1, axis=0)) * gaps[:, None]
    return np.asarray(pieces.sum(axis=0) / _TWO_PI)


def _helix_integral(
    gamma: FloatArray, gaps: FloatArray, r: float, alpha: float, sign: int
) -> tuple[FloatArray, FloatArray]:
    """``-s r / |sin a|`` times the integral of ``gamma`` from the first station.

    ``gamma`` (sorted stations, samples) less its pitch mean, integrated over
    the piecewise-linear field; with the share of the mean taken out.
    """
    sin_a = abs(math.sin(alpha))
    if sin_a < 1.0e-12:
        raise CableInputError("the slip needs a lay angle above zero")
    if sign not in (1, -1):
        raise CableInputError("the lay sign is +1 or -1")
    mean = _pitch_mean(gamma, gaps)
    largest = np.max(np.abs(gamma), axis=0)
    share = np.divide(
        np.abs(mean), largest, out=np.zeros_like(mean), where=largest > 0.0
    )
    centred = gamma - mean[None, :]
    pieces = 0.5 * (centred + np.roll(centred, -1, axis=0)) * gaps[:, None]
    cumulative = np.zeros_like(centred)
    cumulative[1:] = np.cumsum(pieces[:-1], axis=0)
    return np.asarray(-sign * r / sin_a * cumulative), np.asarray(share)


def periodic_interpolate(
    azimuth_rad: FloatArray, values: FloatArray, phi: FloatArray | float
) -> FloatArray:
    """Values (stations, ...) at the azimuths ``phi``, linear and periodic."""
    order, stations, gaps = _sorted_stations(azimuth_rad)
    data = np.asarray(values)[order]
    angle = np.mod(np.asarray(phi, dtype=np.float64), _TWO_PI)
    # Station k covers [stations[k], stations[k] + gaps[k]); the first station
    # also covers the arc before it, reached from the last one.
    shifted = np.mod(angle - stations[0], _TWO_PI)
    offsets = stations - stations[0]
    k = np.searchsorted(offsets, shifted, side="right") - 1
    k = np.clip(k, 0, stations.size - 1)
    f = (shifted - offsets[k]) / gaps[k]
    f = f.reshape(f.shape + (1,) * (data.ndim - 1))
    following = (k + 1) % stations.size
    return np.asarray((1.0 - f) * data[k] + f * data[following])


@dataclass(frozen=True)
class HelixField:
    """The state of one helical population round the section, per sample.

    Arrays have shape (stations, samples) with the stations at
    ``azimuth_rad`` (the physics frame: the first wire on the x axis). A wire
    at azimuth ``phi`` anywhere along the cable has the values there
    (:meth:`along`).
    """

    name: str
    kind: str
    """``"armour"`` or ``"cores"``."""
    bodies: int
    """Wires of the layer, or laid cores."""
    r_helix_m: float
    lay_angle_rad: float
    """Lay angle magnitude."""
    lay_sign: int
    """``+1`` right-hand lay, ``-1`` left-hand lay."""
    azimuth_rad: FloatArray
    slip_m: FloatArray
    """Axial slip along the wire from its stuck position [m], positive
    towards increasing ``z``."""
    slipping: BoolArray
    stress_Pa: FloatArray | None
    """Centre-line axial stress (tension and friction) [Pa]; ``None`` for
    the cores."""
    bending_Pa: FloatArray | None
    """Largest local bending stress round the wire [Pa]."""
    utilisation: FloatArray | None
    """Largest stress round the wire over the yield stress; for the cores, the
    largest utilisation of the components of the core against their limits
    (:func:`~cablexsec.analysis.components.core_component_utilisation`),
    ``None`` when none has a limit."""
    transverse_m: FloatArray | None
    """Transverse slip from the loxodrome [m], when that model is on."""
    friction_model: str
    mean_slip_share: float = 0.0
    """Largest share of the slip strain taken out as its pitch mean."""
    moving: bool = True
    """Whether the bodies slip along their helix in this history; the laid
    cores without the core lay stage carry their utilisation only (their
    slip is not followed, and ``slip_m`` is zero)."""
    friction_Pa: FloatArray | None = None
    """Friction stress the law gives at the stations [Pa]: of a wire, its
    friction stress; of a laid core, that of the core population. ``None``
    for cores held to the cable axis (no friction law)."""
    friction_modulus_Pa: float = math.nan
    """Modulus that turns the friction stress into the strain of the body's
    axis (the wire's, or the core population's)."""

    @property
    def stations(self) -> int:
        return int(self.azimuth_rad.size)

    def along(self, values: FloatArray, phi: FloatArray | float) -> FloatArray:
        """``values`` (stations, samples) at the azimuths ``phi``."""
        return periodic_interpolate(self.azimuth_rad, values, phi)


@dataclass(frozen=True)
class KinematicHistory:
    """The section and its helical populations at the kept samples."""

    samples: IntArray
    """Indices of the kept samples in the history."""
    time_s: FloatArray
    kappa_x_1_m: FloatArray
    kappa_y_1_m: FloatArray
    axial_strain: FloatArray
    twist_1_m: FloatArray
    """Twist of the section [rad/m]; zero, the ends held against twist."""
    tension_N: FloatArray
    layers: tuple[HelixField, ...]
    cores: HelixField | None
    friction_model: str
    provenance: Provenance

    @property
    def curvature_1_m(self) -> FloatArray:
        return np.asarray(np.hypot(self.kappa_x_1_m, self.kappa_y_1_m))

    @property
    def fields(self) -> tuple[HelixField, ...]:
        """The armour layers, then the cores."""
        return self.layers + ((self.cores,) if self.cores is not None else ())

    def friction_field(self, index: int) -> FrictionField:
        """The friction the history left at kept sample ``index``, for the
        stress routes (:class:`~cablexsec.analysis.stress.FrictionField`)."""
        return friction_field_of(self.fields, index)


def friction_field_of(fields: Sequence[HelixField], index: int) -> FrictionField:
    """The friction of ``fields`` at kept sample ``index`` (physics frame).

    The armour layers give their wires' friction stress; laid cores with a
    friction law give the strain of their axes (friction over modulus).
    """
    from cablexsec.analysis.stress import FrictionField  # noqa: PLC0415

    wires: dict[str, tuple[FloatArray, FloatArray]] = {}
    cores = None
    for item in fields:
        if item.friction_Pa is None:
            continue
        column = np.asarray(item.friction_Pa[:, index], dtype=np.float64)
        azimuth = np.asarray(item.azimuth_rad, dtype=np.float64)
        if item.kind == "armour":
            wires[item.name] = (azimuth, column)
        elif math.isfinite(item.friction_modulus_Pa) and item.friction_modulus_Pa > 0:
            cores = (azimuth, column / item.friction_modulus_Pa)
    return FrictionField(wires, cores)


# ----------------------------------------------------------------------------- #
# Inputs                                                                        #
# ----------------------------------------------------------------------------- #


def _history(values: Any, count: int, name: str) -> FloatArray:
    array = real_array(values, name).reshape(-1)
    if array.shape != (count,) or not np.isfinite(array).all():
        raise CableInputError(f"{name} must be finite and match the curvature")
    return array


def _kept(samples: Sequence[int] | IntArray | None, count: int) -> IntArray:
    if samples is None:
        return np.arange(count, dtype=np.int64)
    kept = np.asarray(samples, dtype=np.int64).reshape(-1)
    if not kept.size or np.any(np.diff(kept) <= 0) or kept[0] < 0 or kept[-1] >= count:
        raise CableInputError("the kept samples must be increasing indices")
    return kept


def _lay_sign(direction: str) -> int:
    return 1 if direction == "right_hand" else -1


def _section(layer: HelicalLayer, spec: HelicalLayerSpec) -> WireSection:
    if spec.section is not None:
        return spec.section
    return round_section(2.0 * math.sqrt(spec.A_wire_m2 / math.pi))


def _yield_stress(material: str) -> float | None:
    try:
        value = float(library.get_property(material, "sigma_y_Pa"))
    except KeyError:
        return None
    return value if value > 0.0 else None


def armour_transverse_model(
    cable: CableData,
    layer: HelicalLayer,
    spec: HelicalLayerSpec,
    wire_tension_N: float,
) -> TransverseSlipModel:
    """The transverse equilibrium of a wire of ``layer``.

    The shear modulus is ``E / (2 (1 + nu))`` with Poisson's ratio of the
    armour material; the friction capacity per unit wire length is the axial
    one, ``mu P (2 pi r / n) cos a`` (the clamping of the slip cap).
    """
    try:
        nu = float(library.get_property(layer.material, "nu"))
    except KeyError:
        raise CableInputError(
            f"the transverse slip of {layer.name} needs Poisson's ratio of its material"
        ) from None
    width = _TWO_PI * spec.r_helix_m / spec.n_wires
    capacity = spec.mu * spec.contact_pressure_Pa * width * math.cos(spec.lay_angle_rad)
    return TransverseSlipModel(
        section=_section(layer, spec),
        E_Pa=spec.E_wire_Pa,
        G_Pa=spec.E_wire_Pa / (2.0 * (1.0 + nu)),
        r_helix_m=spec.r_helix_m,
        lay_angle_rad=abs(spec.lay_angle_rad),
        tension_N=max(float(wire_tension_N), 0.0),
        capacity_N_per_m=max(capacity, 0.0),
        sense=_lay_sign(layer.lay_direction),
    )


# ----------------------------------------------------------------------------- #
# Fields                                                                        #
# ----------------------------------------------------------------------------- #


@dataclass
class _Drive:
    """The curvature history and the kept samples."""

    kx: FloatArray
    ky: FloatArray
    kept: IntArray
    before: IntArray
    """The sample before each kept one (``-1``: the stress-free start)."""

    def at(self, phi: FloatArray, r: float, index: IntArray) -> FloatArray:
        """Reduced drive ``r (kx cos phi + ky sin phi)`` (stations, kept)."""
        kx = np.where(index >= 0, self.kx[np.maximum(index, 0)], 0.0)
        ky = np.where(index >= 0, self.ky[np.maximum(index, 0)], 0.0)
        return np.asarray(
            r
            * (np.cos(phi)[:, None] * kx[None, :] + np.sin(phi)[:, None] * ky[None, :])
        )

    def tolerance(self, r: float) -> float:
        largest = float(np.max(np.hypot(self.kx, self.ky), initial=0.0))
        return SLIP_TOLERANCE * r * largest + 1e-300


def _friction_stress(
    spec: HelicalLayerSpec,
    drive: _Drive,
    pressure: FloatArray | None,
    law: str,
    progressive: ProgressiveFriction | None,
) -> FloatArray:
    if law == "progressive":
        return progressive_wire_friction_stress(
            spec,
            drive.kx,
            drive.ky,
            progressive=progressive,
            contact_pressure_Pa=pressure,
        )
    if HAVE_NUMBA:
        return wire_friction_stress(
            spec, drive.kx, drive.ky, contact_pressure_Pa=pressure
        )
    return wire_slip_history(
        spec, drive.kx, drive.ky, contact_pressure_Pa=pressure
    ).stress_Pa


def _spring_slider_slip(
    spec: HelicalLayerSpec,
    phi: FloatArray,
    stress: FloatArray,
    drive: _Drive,
    lay_sign: int,
) -> tuple[FloatArray, BoolArray, float]:
    """Slip, slipping flags and mean share of a spring-slider law.

    The whole history is followed in blocks of samples, because the constant
    of the integration is fixed by the stick zones step by step (module
    notes); the kept samples are returned.
    """
    r = spec.r_helix_m
    k_f = spec.k_f
    alpha = spec.lay_angle_rad
    cos2 = math.cos(alpha) ** 2
    tolerance = drive.tolerance(r)
    order, _sorted_phi, gaps = _sorted_stations(phi)
    station_phi = np.asarray(phi, dtype=np.float64)[order]
    weights = gaps / _TWO_PI
    count = drive.kx.size
    kept_slip = np.empty((phi.size, drive.kept.size))
    kept_slipping = np.empty((phi.size, drive.kept.size), dtype=bool)
    last_plastic = np.zeros(phi.size)
    last_integral = np.zeros(phi.size)
    constant = 0.0
    share = 0.0
    position = 0
    for start in range(0, count, _BLOCK):
        stop = min(start + _BLOCK, count)
        index = np.arange(start, stop, dtype=np.int64)
        plastic = drive.at(station_phi, r, index) - stress[order, start:stop] / k_f
        before = np.column_stack((last_plastic, plastic[:, :-1]))
        changed = np.abs(plastic - before) > tolerance
        integral, shares = _helix_integral(cos2 * plastic, gaps, r, alpha, lay_sign)
        share = max(share, float(np.max(shares, initial=0.0)))
        step = integral - np.column_stack((last_integral, integral[:, :-1]))
        # Stick zones: a station whose slip strain did not change, with both
        # of its neighbours (a lone station there is a sign change of the
        # drive, not a zone where the wire holds to its bed).
        held = ~changed & ~np.roll(changed, 1, axis=0) & ~np.roll(changed, -1, axis=0)
        held_weight = weights[:, None] * held
        total = held_weight.sum(axis=0)
        anchored = -(held_weight * step).sum(axis=0) / np.where(total > 0, total, 1.0)
        free = -(weights[:, None] * 0.5 * (step + np.roll(step, -1, axis=0))).sum(
            axis=0
        )
        constants = constant + np.cumsum(np.where(total > 0, anchored, free))
        constant = float(constants[-1])
        last_plastic = plastic[:, -1]
        last_integral = integral[:, -1]
        wanted = drive.kept[(drive.kept >= start) & (drive.kept < stop)]
        columns = wanted - start
        number = wanted.size
        kept_slip[order, position : position + number] = (
            integral[:, columns] + constants[columns][None, :]
        )
        kept_slipping[order, position : position + number] = changed[:, columns]
        position += number
    return kept_slip, kept_slipping, share


def _bending(
    spec: HelicalLayerSpec,
    section: WireSection,
    phi: FloatArray,
    kx: FloatArray,
    ky: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Local bending stress at the outer face and the edge (stations, kept)."""
    depth = 2.0 * section.half_thickness_m
    width = 2.0 * section.half_width_m
    radial = local_bending_fibre_stress(
        kx[None, :],
        ky[None, :],
        phi[:, None],
        spec.E_wire_Pa,
        depth,
        spec.lay_angle_rad,
    )
    tangential = local_bending_fibre_stress(
        kx[None, :],
        ky[None, :],
        phi[:, None],
        spec.E_wire_Pa,
        width,
        spec.lay_angle_rad,
        math.pi / 2,
    )
    return np.asarray(radial), np.asarray(tangential)


def _transverse(
    model: TransverseSlipModel,
    spec: HelicalLayerSpec,
    section: WireSection,
    phi: FloatArray,
    drive: _Drive,
    radial: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Transverse slip and the local bending it gives (stations, kept).

    As :func:`~cablexsec.analysis.fatigue.wire_stress_field` with a
    transverse model: the lateral curvature ``cos a (1 + sin^2 a) p_l - k^2 v``
    and the tangent rotation ``-(sin 2a / r) k dv/dphi`` of the normal one.
    """
    p_l = (
        -drive.kx[None, :] * np.sin(phi)[:, None]
        + drive.ky[None, :] * np.cos(phi)[:, None]
    )
    v, _q, k_l = model.slip_history(p_l)
    width = 2.0 * section.half_width_m
    depth = 2.0 * section.half_thickness_m
    tangential = spec.E_wire_Pa * k_l * (width / 2)
    step = _TWO_PI / phi.size
    dv = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * step)
    k_wave = abs(math.sin(spec.lay_angle_rad)) / spec.r_helix_m
    dk_n = -(math.sin(2 * spec.lay_angle_rad) / spec.r_helix_m) * k_wave * dv
    kept = drive.kept
    radial = radial + spec.E_wire_Pa * dk_n[:, kept] * (depth / 2)
    return v[:, kept], radial, tangential[:, kept]


def _armour_field(
    layer: HelicalLayer,
    spec: HelicalLayerSpec,
    drive: _Drive,
    strain: FloatArray,
    pressure: FloatArray | None,
    law: str,
    progressive: ProgressiveFriction | None,
    transverse: TransverseSlipModel | None,
) -> HelixField:
    sign = _lay_sign(layer.lay_direction)
    alpha = abs(spec.lay_angle_rad)
    section = _section(layer, spec)
    tension_stress = spec.E_wire_Pa * math.cos(alpha) ** 2 * strain[drive.kept]
    if law == "distributed":
        phi, slip, slipping, friction, share = _distributed(spec, drive, pressure, sign)
    else:
        phi = np.asarray(spec.phi_wires, dtype=np.float64)
        stress = _friction_stress(spec, drive, pressure, law, progressive)
        slip, slipping, share = _spring_slider_slip(spec, phi, stress, drive, sign)
        friction = stress[:, drive.kept]
    kx, ky = drive.kx[drive.kept], drive.ky[drive.kept]
    radial, tangential = _bending(spec, section, phi, kx, ky)
    transverse_slip = None
    if transverse is not None:
        transverse_slip, radial, tangential = _transverse(
            transverse, spec, section, phi, drive, radial
        )
    centre = friction + tension_stress[None, :]
    bending = section.max_bending_stress(radial, tangential)
    limit = _yield_stress(layer.material)
    utilisation = None if limit is None else (np.abs(centre) + bending) / limit
    return HelixField(
        name=layer.name,
        kind="armour",
        bodies=int(spec.n_wires),
        r_helix_m=float(spec.r_helix_m),
        lay_angle_rad=alpha,
        lay_sign=sign,
        azimuth_rad=phi,
        slip_m=slip,
        slipping=slipping,
        stress_Pa=np.asarray(centre, dtype=np.float64),
        bending_Pa=np.asarray(bending, dtype=np.float64),
        utilisation=None if utilisation is None else np.asarray(utilisation),
        transverse_m=transverse_slip,
        friction_model=law,
        mean_slip_share=share,
        friction_Pa=np.asarray(friction, dtype=np.float64),
        friction_modulus_Pa=float(spec.E_wire_Pa),
    )


def _distributed(
    spec: HelicalLayerSpec,
    drive: _Drive,
    pressure: FloatArray | None,
    sign: int,
) -> tuple[FloatArray, FloatArray, BoolArray, FloatArray, float]:
    """Stations, slip, slipping, friction stress and mean share of the
    distributed law, from its own helix field."""
    history = distributed_friction_history(
        spec,
        drive.kx,
        drive.ky,
        contact_pressure_Pa=pressure,
        field_samples=drive.kept,
    )
    n_nodes = int(history.parameters["nodes_per_pitch"])
    h = _TWO_PI / n_nodes
    phi = np.mod((np.arange(n_nodes) - 0.5) * h, _TWO_PI)
    field = history.kinematic_field
    if field is None:
        # No friction bound: the wire slides freely and carries no friction
        # stress; its slip is the full kinematic slip (p equal to the drive).
        stress = np.zeros((n_nodes, drive.kx.size))
        slip, slipping, share = _spring_slider_slip(spec, phi, stress, drive, sign)
        return phi, slip, slipping, stress[:, drive.kept], share
    unit = float(history.parameters["length_unit_m"])
    stress_unit = float(history.parameters["stress_unit_Pa"])
    nodal = np.transpose(field, (1, 2, 0))  # (4, nodes, kept)
    slip = sign * unit * nodal[1]
    increment = np.abs(nodal[2]) * unit
    scale = (
        drive.tolerance(spec.r_helix_m)
        * spec.r_helix_m
        / abs(math.sin(spec.lay_angle_rad))
    )
    slipping = increment > scale
    force = nodal[3]
    # Segment j spans nodes j and j + 1: a node takes the mean of the two
    # segments it joins.
    at_nodes = 0.5 * (force + np.roll(force, 1, axis=0))
    return phi, np.asarray(slip), slipping, stress_unit * at_nodes, 0.0


def _held_core_field(cable: CableData, drive: _Drive, strain: FloatArray) -> HelixField:
    """The utilisation of cores that are not laid: held to the cable axis.

    A core that is not laid follows the bending strain of the section at its
    axis, ``r_c (kx cos phi + ky sin phi)`` (the stuck state of the component
    response, which is its frictional state too); one station per core.
    """
    from cablexsec.analysis.components import (  # noqa: PLC0415
        core_component_utilisation,
    )
    from cablexsec.cables.schema import CylindricalOffsetLayer  # noqa: PLC0415

    layer = next(
        item
        for item in cable.layers
        if isinstance(item, CylindricalOffsetLayer) and item.n_cores > 0
    )
    n = int(layer.n_cores)
    phi = 2.0 * math.pi * np.arange(n) / n
    kept = drive.kept
    kx, ky = drive.kx[kept], drive.ky[kept]
    r_c = float(layer.r_offset_m or 0.0)
    axis = r_c * (
        np.cos(phi)[:, None] * kx[None, :] + np.sin(phi)[:, None] * ky[None, :]
    )
    utilisation = core_component_utilisation(cable, phi, kx, ky, strain[kept], axis)
    zeros = np.zeros((n, kept.size))
    return HelixField(
        name="cores",
        kind="cores",
        bodies=n,
        r_helix_m=r_c,
        lay_angle_rad=0.0,
        lay_sign=1,
        azimuth_rad=phi,
        slip_m=zeros,
        slipping=np.zeros((n, kept.size), dtype=bool),
        stress_Pa=None,
        bending_Pa=None,
        utilisation=utilisation,
        transverse_m=None,
        friction_model="held",
        moving=False,
    )


def _has_cores(cable: CableData) -> bool:
    from cablexsec.cables.schema import CylindricalOffsetLayer  # noqa: PLC0415

    return any(
        isinstance(layer, CylindricalOffsetLayer) and layer.n_cores > 1
        for layer in cable.layers
    )


def _core_field(
    cable: CableData,
    spec: CoreFrictionSpec,
    tension_N: float,
    contact_model: str,
    drive: _Drive,
    strain: FloatArray,
    moving: bool = True,
) -> HelixField:
    from cablexsec.analysis.components import (  # noqa: PLC0415
        core_component_utilisation,
    )

    model = build_core_friction(cable, spec, tension_N, contact_model=contact_model)
    population = model.population
    history = wire_slip_history(
        population,
        drive.kx,
        drive.ky,
        direction_hold_fraction=spec.direction_hold_fraction,
    )
    phi = np.asarray(population.phi_wires, dtype=np.float64)
    lay = cable.core_lay
    sign = _lay_sign(lay.lay_direction) if lay is not None else 1
    slip, slipping, share = _spring_slider_slip(
        population, phi, history.stress_Pa, drive, sign
    )
    kept = drive.kept
    # The components of each core against their limits: the core-axis strain
    # of the friction law with the section strain and the core's own bending.
    utilisation = core_component_utilisation(
        cable,
        phi,
        drive.kx[kept],
        drive.ky[kept],
        strain[kept],
        np.asarray(history.stress_Pa)[:, kept] / population.E_wire_Pa,
    )
    field = HelixField(
        name="cores",
        kind="cores",
        bodies=int(model.n_cores),
        r_helix_m=float(model.r_offset_m),
        lay_angle_rad=abs(float(model.lay_angle_rad)),
        lay_sign=sign,
        azimuth_rad=phi,
        slip_m=slip,
        slipping=slipping,
        stress_Pa=None,
        bending_Pa=None,
        utilisation=utilisation,
        transverse_m=None,
        friction_model="direction_tied",
        mean_slip_share=share,
        moving=moving,
        friction_Pa=np.asarray(
            np.asarray(history.stress_Pa)[:, kept], dtype=np.float64
        ),
        friction_modulus_Pa=float(population.E_wire_Pa),
    )
    if not moving:
        # The utilisation from the core friction law (the frictional state of
        # the component response); the slip is not shown without the stage.
        field = dataclasses.replace(
            field,
            slip_m=np.zeros_like(field.slip_m),
            slipping=np.zeros_like(field.slipping),
            mean_slip_share=0.0,
        )
    return field


# ----------------------------------------------------------------------------- #
# The history                                                                   #
# ----------------------------------------------------------------------------- #


def kinematic_history(
    cable: CableData,
    kappa_x: FloatArray,
    kappa_y: FloatArray,
    *,
    time_s: FloatArray | None = None,
    tension_N: FloatArray | float | None = None,
    axial_strain: FloatArray | None = None,
    contact_tension_N: float | None = None,
    contact_pressure_Pa: Mapping[str, FloatArray] | None = None,
    mu: float | None = None,
    contact_model: str = "bounded",
    friction_model: FrictionLaw = "direction_tied",
    progressive: ProgressiveFriction | None = None,
    core_friction: bool | CoreFrictionSpec | None = None,
    transverse: bool | Mapping[str, TransverseSlipModel] = False,
    samples: Sequence[int] | IntArray | None = None,
    core_utilisation: bool = True,
) -> KinematicHistory:
    """The kinematics of ``cable`` over a curvature history (module notes).

    Parameters
    ----------
    kappa_x, kappa_y : numpy.ndarray
        Axial-strain-gradient curvature [1/m] at every sample.
    time_s : numpy.ndarray, optional
        Sample times [s] (default: the sample number).
    tension_N : numpy.ndarray or float, optional
        Force on the armour [N]; the axial strain is ``T / EA_fixed``.
    axial_strain : numpy.ndarray, optional
        A prescribed axial strain instead of the tension.
    contact_tension_N : float, optional
        Tension of the armour contact pressure (default: the first sample).
    contact_pressure_Pa : mapping, optional
        Contact pressure history ``P1 + P2`` [Pa] per armour layer name (the
        clamping following the force, :func:`cablexsec.armour_contact_histories`).
    mu, contact_model : optional
        Armour friction coefficient and contact model, as
        :func:`cablexsec.compute_mkappa`.
    friction_model : {"direction_tied", "distributed", "progressive"}
        The armour friction law of the stress route.
    core_friction : bool or CoreFrictionSpec, optional
        Add the slip of the laid cores (the core lay stage).
    core_utilisation : bool
        Without the core lay stage, still give the laid cores the utilisation
        of their components (default), from the core friction law of the
        frictional state (cores held to the cable axis when not laid); they
        do not move (``HelixField.moving`` is false).
    transverse : bool or mapping, optional
        Add the transverse slip of every armour layer (``True``: the model of
        :func:`armour_transverse_model`), or of the named layers.
    samples : sequence of int, optional
        Samples kept (increasing; default all). The friction laws always run
        over the whole history.
    """
    from cablexsec.cable_adapters import (  # noqa: PLC0415
        cable_to_mkappa_specs,
        structural_armour_layers,
    )
    from cablexsec.mechanics.stiffness import ea_fixed  # noqa: PLC0415

    if friction_model not in FRICTION_LAWS:
        raise CableInputError(
            "friction_model must be 'direction_tied', 'distributed' or 'progressive'"
        )
    kx = real_array(kappa_x, "kappa_x").reshape(-1)
    count = kx.size
    if not count or not np.isfinite(kx).all():
        raise CableInputError("the curvature history must be finite and not empty")
    ky = _history(kappa_y, count, "kappa_y")
    times = (
        np.arange(count, dtype=np.float64)
        if time_s is None
        else _history(time_s, count, "time")
    )
    if tension_N is not None and axial_strain is not None:
        raise CableInputError("give the tension or the axial strain, not both")
    tension = np.zeros(count)
    if tension_N is not None:
        tension = (
            np.full(count, float(tension_N))
            if np.ndim(tension_N) == 0
            else _history(tension_N, count, "tension")
        )
        if np.any(tension < 0.0):
            raise CableInputError("the force on the armour must not be negative")
        strain = tension / ea_fixed(cable)
    elif axial_strain is not None:
        strain = _history(axial_strain, count, "axial strain")
    else:
        strain = np.zeros(count)
    kept = _kept(samples, count)
    before = kept - 1
    drive = _Drive(kx, ky, kept, before)
    reference = float(tension[0]) if contact_tension_N is None else contact_tension_N
    specs = cable_to_mkappa_specs(cable, reference, mu=mu, contact_model=contact_model)
    armours = structural_armour_layers(cable)
    models: dict[str, TransverseSlipModel] = {}
    if transverse is True:
        for layer, spec in zip(armours, specs, strict=True):
            sigma = spec.E_wire_Pa * math.cos(spec.lay_angle_rad) ** 2
            static = sigma * (reference / ea_fixed(cable)) * spec.A_wire_m2
            models[layer.name] = armour_transverse_model(cable, layer, spec, static)
    elif isinstance(transverse, Mapping):
        models = dict(transverse)
    pressures = dict(contact_pressure_Pa or {})
    fields = []
    for layer, spec in zip(armours, specs, strict=True):
        pressure = pressures.get(layer.name)
        if pressure is not None:
            pressure = _history(pressure, count, f"contact pressure of {layer.name}")
        fields.append(
            _armour_field(
                layer,
                spec,
                drive,
                strain,
                pressure,
                friction_model,
                progressive,
                models.get(layer.name),
            )
        )
    core_spec = resolve_core_friction(core_friction)
    cores = (
        None
        if core_spec is None
        else _core_field(cable, core_spec, reference, contact_model, drive, strain)
    )
    if cores is None and core_utilisation and _has_cores(cable):
        # Without the core lay stage the cores do not move, but they carry
        # the utilisation of their components, as in the component response:
        # from the core friction law of the frictional state, or held to the
        # cable axis when the cores are not laid.
        try:
            cores = _core_field(
                cable,
                CoreFrictionSpec(),
                reference,
                contact_model,
                drive,
                strain,
                moving=False,
            )
        except (CableInputError, CableSchemaError):
            cores = _held_core_field(cable, drive, strain)
    limitations = [
        "uniform curvature along the cable: the state of a wire depends on its "
        "azimuth only, and the stations round the section are read along the helix",
        "the pitch mean of the slip strain is taken out (no net slide of a wire "
        "over a pitch); its largest share is kept with each layer",
        "ends held against twist: the twist is zero and the axial strain is the "
        "force over the axial stiffness with the twist held",
        "mean clamping prescribed per sample, as in the stress route",
    ]
    return KinematicHistory(
        samples=kept,
        time_s=times[kept],
        kappa_x_1_m=kx[kept],
        kappa_y_1_m=ky[kept],
        axial_strain=strain[kept],
        twist_1_m=np.zeros(kept.size),
        tension_N=tension[kept],
        layers=tuple(fields),
        cores=cores,
        friction_model=friction_model,
        provenance=Provenance(
            method="wire slip along the helix from the friction law of the stress "
            f"route ({friction_model.replace('_', ' ')}): the Coulomb slip strain "
            "integrated along the wire, or the distributed law's own slip field",
            citations=(
                "[KraincanicKebadze2001] (full-slip slip)",
                "[Papailiou1997] (slip cap)",
                *(("[Saevik1992] (transverse slip)",) if models else ()),
            ),
            gates=(
                "zero slip in stick",
                "full-slip kinematic slip in frictionless slip",
                "stress equal to the wire stress field of the fatigue route",
            ),
            limitations=tuple(limitations),
        ),
    )
