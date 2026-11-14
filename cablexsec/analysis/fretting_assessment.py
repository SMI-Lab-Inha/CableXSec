# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Fretting regime maps, wear and knock-down for helical layers and conductors.

This module applies :mod:`cablexsec.analysis.fretting_regime` to the contacts
of a cable section under a curvature history:

* **helical wire on a polymer face** (armour on bedding or sheath, screen wire
  on tape): one Hertz line contact per face and wire
  (:func:`~cablexsec.analysis.fretting_regime.line_contact`), sharing the
  support stiffness ``k_G = EA (sin a / r_h)^2 xi/(1 - xi)`` of the section
  law; the wire's relative displacement is the kinematic slip of
  :func:`~cablexsec.mechanics.mindlin_contact.slip_increment` (loxodromic,
  or with the geodesic or legacy transverse component);
* **armour crossing** (opposite-laid armour layers in direct contact): the
  Mindlin point contact of the existing crossing model (Hertz ``K_t``,
  crossing force from the shared-interface line load);
* **conductor trellis** (:func:`conductor_trellis_fretting`): the
  tangential-load ratio of each crossing follows the friction mobilisation
  of the stick--slip operator ([KarlsenHusoy2026]); cycles without slip are
  partial slip, slipping cycles gross slip with the kinematic relative slip
  of the two rings as an upper estimate of the sliding amplitude.

For every wire and face the result holds the counted cycles per regime, the
dissipated energy, the mean wear depth and, for armour, the Miner damage at
the contact fibre with and without the regime knock-down
(:func:`~cablexsec.analysis.fretting_regime.fretting_fatigue_damage`). Values
are annualised with the record duration and an occurrence fraction. The
angular position of each wire is also expressed as its distance from the
extreme fibre of the principal bending direction (0 deg extreme fibre,
90 deg neutral axis), so that wear and cracking locations can be compared.

The normal loads are those of the contact model at one reference tension
(default: the mean of the tension history); their variation within the
record is not followed. Face pressures split the friction-setting pressure
``P1 + P2`` of :func:`cablexsec.cable_adapters.armour_contact_pressure` in
the ratio of the bounded radial chain.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cablexsec.analysis.fast_rainflow import rainflow_table
from cablexsec.analysis.fatigue import FatigueLayer, wire_stress_field
from cablexsec.analysis.fretting_regime import (
    GROSS,
    PARTIAL,
    REGIMES,
    FrettingContact,
    _window_half_range,
    assess_contact,
    fretting_fatigue_damage,
    line_contact,
    mindlin_cycle_energy,
    point_contact,
    worn_section_fraction,
)
from cablexsec.exceptions import CableInputError
from cablexsec.materials.wear import (
    FrettingKnockdown,
    WearCoefficient,
)
from cablexsec.mechanics.hertz_contact import (
    crossed_cylinder_curvatures,
    layer_crossing_angle_rad,
    layer_crossing_density_per_m,
)
from cablexsec.mechanics.mindlin_contact import (
    TransverseMap,
    geometric_slip_factor,
    slip_increment,
    transverse_map_or_warn,
)
from cablexsec.mechanics.mkappa import HelicalLayerSpec
from cablexsec.mechanics.strand_slip import (
    ConductorRing,
    PressureTransmission,
    conductor_axial_stiffness,
    conductor_ring_parameters,
    interface_line_loads,
    strand_slip_strain,
)
from cablexsec.provenance import Provenance

__all__ = [
    "SECONDS_PER_YEAR",
    "CableFrettingResult",
    "FaceSpec",
    "FrettingFace",
    "TrellisFrettingResult",
    "conductor_trellis_fretting",
    "extreme_fibre_offset_deg",
    "helical_layer_fretting",
    "principal_bending_angle",
    "wire_displacement_histories",
]

FloatArray = NDArray[np.float64]

#: Julian year [s] used to annualise one record.
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0


def principal_bending_angle(kappa_x: ArrayLike, kappa_y: ArrayLike) -> float:
    """Azimuth [rad] of the extreme fibre of the dynamic curvature.

    The principal axis of the curvature fluctuation about its mean; the
    bending strain at azimuth ``theta`` is ``r (kx cos theta + ky sin
    theta)``, so the extreme fibre lies along that axis.
    """
    kx = np.asarray(kappa_x, dtype=np.float64)
    ky = np.asarray(kappa_y, dtype=np.float64)
    dx, dy = kx - kx.mean(), ky - ky.mean()
    sxx, syy, sxy = float(dx @ dx), float(dy @ dy), float(dx @ dy)
    return 0.5 * math.atan2(2.0 * sxy, sxx - syy)


def extreme_fibre_offset_deg(phi_rad: ArrayLike, beta_rad: float) -> FloatArray:
    """Angular distance from the nearest extreme fibre, folded to [0, 90] deg."""
    phi = np.asarray(phi_rad, dtype=np.float64)
    offset = np.mod(phi - beta_rad + 0.5 * math.pi, math.pi) - 0.5 * math.pi
    return np.asarray(np.degrees(np.abs(offset)))


def wire_displacement_histories(
    spec: HelicalLayerSpec,
    kappa_x: ArrayLike,
    kappa_y: ArrayLike,
    *,
    transverse_ratio: float = 0.0,
    transverse_map: TransverseMap | None = None,
) -> FloatArray:
    """Kinematic contact displacement of every wire, shape ``(wire, time, 2)``.

    :func:`~cablexsec.mechanics.mindlin_contact.slip_increment` is linear in
    the curvature increment, so the history is ``Delta kx e_x + Delta ky e_y``
    with the unit responses ``e_x``, ``e_y`` of each wire, measured from the
    first sample.
    """
    resolved = transverse_map_or_warn(transverse_ratio, transverse_map, stacklevel=3)
    kx = np.asarray(kappa_x, dtype=np.float64)
    ky = np.asarray(kappa_y, dtype=np.float64)
    c_geo = geometric_slip_factor(spec.r_helix_m, spec.lay_angle_rad)
    dkx, dky = kx - kx[0], ky - ky[0]
    out = np.empty((spec.n_wires, kx.size, 2))
    for k, phi in enumerate(spec.phi_wires):
        args = {
            "transverse_ratio": transverse_ratio,
            "transverse_map": resolved,
            "lay_angle_rad": spec.lay_angle_rad,
        }
        e_x = slip_increment(1.0, 0.0, float(phi), c_geo, **args)  # type: ignore[arg-type]
        e_y = slip_increment(0.0, 1.0, float(phi), c_geo, **args)  # type: ignore[arg-type]
        out[k] = dkx[:, None] * e_x[None, :] + dky[:, None] * e_y[None, :]
    return out


@dataclass(frozen=True)
class FaceSpec:
    """One contact face of a helical wire layer.

    ``kind="line"``: the wire bears on a polymer or tape face of modulus
    ``partner_E_Pa`` with face pressure ``pressure_Pa`` (referred to the helix
    radius). ``kind="point"``: crossings with an opposite-laid wire layer of
    wire diameter ``partner_d_wire_m`` and crossing angle
    ``crossing_angle_rad``, with ``crossing_force_N`` per crossing and
    ``crossing_density_per_m`` crossings per metre of wire.
    """

    name: str
    angle_rad: float
    kind: str
    partner: str
    partner_E_Pa: float
    partner_nu: float
    mu: float
    wear: WearCoefficient | None
    pressure_Pa: float = 0.0
    partner_d_wire_m: float = 0.0
    crossing_angle_rad: float = 0.0
    crossing_force_N: float = 0.0
    crossing_density_per_m: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("line", "point"):
            raise CableInputError("face kind must be 'line' or 'point'")
        values = (
            self.pressure_Pa,
            self.crossing_force_N,
            self.crossing_density_per_m,
            self.partner_E_Pa,
        )
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise CableInputError("face loads and moduli must be finite, >= 0")


@dataclass(frozen=True)
class FrettingFace:
    """Per-wire fretting results of one face of one layer (annualised).

    Arrays are indexed by wire; ``*_by_regime`` arrays have a second axis
    over :data:`~cablexsec.analysis.fretting_regime.REGIMES`. Energies are in
    J per metre of wire per year (line faces) or J per crossing per year
    (point faces); ``wear_depth_per_year_m`` is the mean depth of the worn
    body per year. ``D_*`` are Miner damages per year at the contact fibre
    (``nan`` when no S-N assessment was requested).
    """

    layer: str
    face: str
    partner: str
    contact: FrettingContact
    contacts: tuple[FrettingContact, ...]
    wire_azimuth_deg: FloatArray
    extreme_fibre_offset_deg: FloatArray
    cycles_by_regime: FloatArray
    energy_by_regime: FloatArray
    wear_depth_per_year_m: FloatArray
    max_amplitude_m: FloatArray
    slip_to_width: FloatArray
    dominant_regime: tuple[str, ...]
    mixed: NDArray[np.bool_]
    D_plain_per_year: FloatArray
    D_fretting_per_year: FloatArray
    D_fretting_by_regime: FloatArray
    section_loss_factor: FloatArray
    knockdown: FrettingKnockdown | None

    @property
    def regime_fraction(self) -> FloatArray:
        """Share of counted cycles per regime, per wire."""
        total = self.cycles_by_regime.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.asarray(np.where(total > 0, self.cycles_by_regime / total, 0.0))

    def location(self) -> dict[str, float]:
        """Extreme-fibre offsets [deg] of the largest wear and fretting damage."""
        wear = self.wear_depth_per_year_m
        out = {
            "max_wear_offset_deg": float(
                self.extreme_fibre_offset_deg[int(np.argmax(wear))]
            )
            if np.max(wear) > 0
            else math.nan,
            "max_wear_per_year_m": float(np.max(wear)),
        }
        if (
            np.isfinite(self.D_fretting_per_year).all()
            and np.max(self.D_fretting_per_year) > 0
        ):
            k = int(np.argmax(self.D_fretting_per_year))
            out["max_damage_offset_deg"] = float(self.extreme_fibre_offset_deg[k])
            out["max_D_fretting_per_year"] = float(self.D_fretting_per_year[k])
            out["max_D_plain_per_year"] = float(np.max(self.D_plain_per_year))
        return out


def _line_contacts(
    layer_name: str,
    spec: HelicalLayerSpec,
    faces: Sequence[FaceSpec],
    nu_wire: float,
) -> dict[str, list[FrettingContact]]:
    """Per-wire line contacts of every line face (shared support stiffness)."""
    n = spec.n_wires
    g = np.ones(n) if spec.shape_g is None else np.asarray(spec.shape_g, float)
    alpha = abs(spec.lay_angle_rad)
    tributary = 2.0 * math.pi * spec.r_helix_m / n * math.cos(alpha)
    ea = spec.E_wire_Pa * spec.A_wire_m2
    q = math.sin(alpha) / spec.r_helix_m
    k_support = math.inf if spec.xi >= 1.0 else ea * q * q * spec.xi / (1.0 - spec.xi)
    radius = math.sqrt(spec.A_wire_m2 / math.pi)
    line_faces = [f for f in faces if f.kind == "line"]
    out: dict[str, list[FrettingContact]] = {}
    for face in line_faces:
        out[face.name] = []
    for k in range(n):
        loads = {f.name: f.pressure_Pa * tributary * float(g[k]) for f in line_faces}
        limit = sum(f.mu * loads[f.name] for f in line_faces)
        for face in line_faces:
            out[face.name].append(
                line_contact(
                    f"{layer_name}/{face.name}",
                    loads[face.name],
                    radius,
                    spec.E_wire_Pa,
                    nu_wire,
                    face.partner_E_Pa,
                    face.partner_nu,
                    face.mu,
                    k_support,
                    friction_limit_N_per_m=limit,
                    wear=face.wear,
                )
            )
    return out


def helical_layer_fretting(
    layer: FatigueLayer,
    faces: Sequence[FaceSpec],
    kappa_x: ArrayLike,
    kappa_y: ArrayLike,
    *,
    duration_s: float,
    nu_wire: float = 0.3,
    tension_N: ArrayLike | None = None,
    EA_full_N: float | None = None,
    axial_strain: ArrayLike | None = None,
    time_fraction: float = 1.0,
    design_life_years: float | None = None,
    knockdown: FrettingKnockdown | None = None,
    section_loss: bool = False,
    partial_slip_wear: bool = False,
) -> dict[str, FrettingFace]:
    """Fretting regime, energy, wear and knock-down of one helical layer.

    Parameters
    ----------
    layer : FatigueLayer
        The layer (geometry, Yun ``xi``, bulk S-N curve, transverse map).
    faces : sequence of FaceSpec
        Its contact faces (``angle_rad`` is the contact fibre: ``pi`` inner,
        ``0`` outer).
    kappa_x, kappa_y : array_like
        Curvature history [1/m].
    duration_s : float
        Duration of the record [s]; results are per year of this loading
        times ``time_fraction``.
    tension_N, EA_full_N, axial_strain : optional
        Axial loading for the contact-fibre stress (as
        :func:`~cablexsec.analysis.fatigue.wire_stress_field`).
    design_life_years : float or None
        Life for the section-loss factor (required with ``section_loss``).
    knockdown : FrettingKnockdown or None
        Regime knock-down; ``None`` skips the S-N assessment.
    section_loss : bool
        Raise the contact-fibre stress by ``1/(1 - f)``, ``f`` the wire
        section worn on all faces over the design life (only faces whose worn
        body is the wire itself).
    """
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise CableInputError("duration_s must be finite and positive")
    if not math.isfinite(time_fraction) or time_fraction < 0:
        raise CableInputError("time_fraction must be finite and nonnegative")
    if section_loss and (design_life_years is None or design_life_years <= 0):
        raise CableInputError("section_loss needs a positive design life")
    spec = layer.spec
    kx = np.asarray(kappa_x, dtype=np.float64)
    ky = np.asarray(kappa_y, dtype=np.float64)
    per_year = SECONDS_PER_YEAR / duration_s * time_fraction
    u = wire_displacement_histories(
        spec,
        kx,
        ky,
        transverse_ratio=layer.transverse_ratio,
        transverse_map=layer.transverse_map
        if layer.transverse_ratio > 0 or layer.transverse_map is not None
        else "geodesic",
    )
    beta = principal_bending_angle(kx, ky)
    offset = extreme_fibre_offset_deg(spec.phi_wires, beta)
    line = _line_contacts(layer.name, spec, faces, nu_wire)
    contacts: dict[str, list[FrettingContact]] = dict(line)
    radius = math.sqrt(spec.A_wire_m2 / math.pi)
    g = (
        np.ones(spec.n_wires)
        if layer.contact_shape_g is None
        else layer.contact_shape_g
    )
    for face in faces:
        if face.kind != "point":
            continue
        e_wire = spec.E_wire_Pa
        a_curv, b_curv = crossed_cylinder_curvatures(
            2.0 * radius,
            face.crossing_angle_rad,
            partner_diameter_m=face.partner_d_wire_m,
        )
        contacts[face.name] = [
            point_contact(
                f"{layer.name}/{face.name}",
                face.crossing_force_N * float(g[k]),
                a_curv,
                b_curv,
                e_wire,
                nu_wire,
                face.partner_E_Pa,
                face.partner_nu,
                face.mu,
                wear=face.wear,
                wire_radius_m=radius,
            )
            if face.crossing_force_N * float(g[k]) > 0
            else line_contact(
                f"{layer.name}/{face.name}",
                0.0,
                radius,
                e_wire,
                nu_wire,
                face.partner_E_Pa,
                face.partner_nu,
                face.mu,
                math.inf,
            )
            for k in range(spec.n_wires)
        ]
    # Pass 1: regime, energy and wear.
    assessed = {
        face.name: [
            assess_contact(
                u[k], contacts[face.name][k], partial_slip_wear=partial_slip_wear
            )
            for k in range(spec.n_wires)
        ]
        for face in faces
    }
    # Section loss over the design life (wire-worn faces only).
    loss_fraction = np.zeros(spec.n_wires)
    if section_loss:
        assert design_life_years is not None
        for face in faces:
            if face.wear is None or "wire" not in face.wear.worn_body:
                continue
            depth = np.array([r.wear_depth_m for r in assessed[face.name]])
            loss_fraction += worn_section_fraction(
                depth * per_year * design_life_years, radius
            )
    loss_fraction = np.minimum(loss_fraction, 0.99)
    loss_factor = 1.0 / (1.0 - loss_fraction)
    field = None
    if knockdown is not None:
        field = wire_stress_field(
            layer,
            kx,
            ky,
            None if tension_N is None else np.asarray(tension_N, dtype=np.float64),
            EA_full_N,
            axial_strain=None
            if axial_strain is None
            else np.asarray(axial_strain, dtype=np.float64),
        )
    results: dict[str, FrettingFace] = {}
    n = spec.n_wires
    for face in faces:
        rows = assessed[face.name]
        d_plain = np.full(n, np.nan)
        d_fret = np.full(n, np.nan)
        d_by = np.full((n, 3), np.nan)
        if field is not None and knockdown is not None:
            stress = field.at_fibre(face.angle_rad)
            for k in range(n):
                kd = fretting_fatigue_damage(
                    stress[k],
                    u[k],
                    contacts[face.name][k],
                    layer.sn_bulk,
                    knockdown,
                    section_loss_factor=float(loss_factor[k]),
                )
                d_plain[k] = kd.damage_plain * per_year
                d_fret[k] = kd.damage_fretting * per_year
                d_by[k] = kd.damage_by_regime * per_year
        results[face.name] = FrettingFace(
            layer=layer.name,
            face=face.name,
            partner=face.partner,
            contact=contacts[face.name][0],
            contacts=tuple(contacts[face.name]),
            wire_azimuth_deg=np.degrees(spec.phi_wires),
            extreme_fibre_offset_deg=offset,
            cycles_by_regime=np.array([r.count_by_regime for r in rows]) * per_year,
            energy_by_regime=np.array([r.energy_by_regime for r in rows]) * per_year,
            wear_depth_per_year_m=np.array([r.wear_depth_m for r in rows]) * per_year,
            max_amplitude_m=np.array([r.max_amplitude_m for r in rows]),
            slip_to_width=np.array([r.slip_to_width for r in rows]),
            dominant_regime=tuple(r.dominant_regime for r in rows),
            mixed=np.array([r.mixed for r in rows], dtype=np.bool_),
            D_plain_per_year=d_plain,
            D_fretting_per_year=d_fret,
            D_fretting_by_regime=d_by,
            section_loss_factor=loss_factor.copy(),
            knockdown=knockdown,
        )
    return results


# --------------------------------------------------------------------------- #
# Conductor trellis contacts                                                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrellisFrettingResult:
    """Trellis fretting of a stranded conductor, per ring (annualised).

    Ring ``i`` is assessed at its crossings with ring ``i + 1`` (oppositely
    laid rings only; others have zero entries). Arrays ``(ring, azimuth)``
    hold per-year cycles per regime (last axis), energy per crossing and mean
    wear depth; ``contacts`` holds the Hertz--Mindlin crossing (``None`` where
    no trellis exists).
    """

    orientations_deg: FloatArray
    extreme_fibre_offset_deg: FloatArray
    cycles_by_regime: FloatArray
    energy_per_year_J: FloatArray
    wear_depth_per_year_m: FloatArray
    relative_slip_factor_m2: FloatArray
    contacts: tuple[FrettingContact | None, ...]
    provenance: Provenance

    @property
    def regime_fraction(self) -> FloatArray:
        total = self.cycles_by_regime.sum(axis=-1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.asarray(np.where(total > 0, self.cycles_by_regime / total, 0.0))


def _relative_slip_factor(ring: ConductorRing, partner: ConductorRing) -> float:
    """``|c_i t_i - c_j t_j|`` of two rings' loxodromic slips [m^2]."""
    c_i = geometric_slip_factor(ring.r_helix_m, ring.lay_angle_rad)
    c_j = geometric_slip_factor(partner.r_helix_m, partner.lay_angle_rad)
    t_i = (math.cos(ring.lay_angle_rad), ring.lay_sign * math.sin(ring.lay_angle_rad))
    t_j = (
        math.cos(partner.lay_angle_rad),
        partner.lay_sign * math.sin(partner.lay_angle_rad),
    )
    return math.hypot(c_i * t_i[0] - c_j * t_j[0], c_i * t_i[1] - c_j * t_j[1])


def conductor_trellis_fretting(
    rings: Sequence[ConductorRing],
    E_Pa: float,
    nu: float,
    kappa_x: ArrayLike,
    kappa_y: ArrayLike,
    tension_N: ArrayLike | None = None,
    *,
    mu_wire: float,
    duration_s: float,
    wear: WearCoefficient | None,
    time_fraction: float = 1.0,
    reference_tension_N: float | None = None,
    external_pressure_Pa: float = 0.0,
    r_outer_m: float | None = None,
    pressure_transmission: PressureTransmission = "outer_interface",
    n_orientations: int = 8,
) -> TrellisFrettingResult:
    """Regime map and wear of the trellis crossings of a stranded conductor.

    The tangential-load ratio of a crossing at azimuth ``theta`` is the
    signed friction mobilisation of the ring's first stick--slip order
    (:func:`~cablexsec.mechanics.strand_slip.strand_slip_strain`). This is the
    gap-1.3 reading; since gap 1.3c the SWT trellis branch of
    :func:`~cablexsec.analysis.conductor_fatigue.assess_conductor_fatigue`
    uses the local shear-flow ratio
    (:func:`~cablexsec.analysis.conductor_fatigue.trellis_force_ratio`)
    instead, and this regime map has not been moved to it.
    Each rainflow cycle of that ratio without slip is partial slip with the
    Mindlin energy at ``x = Q_a/(mu P)``; a cycle with slip is gross slip with
    ``W = 4 mu P delta_a - 3.2 mu P delta_t``, ``delta_a`` the larger of
    ``delta_t`` and the kinematic relative slip amplitude of the two rings
    in the window (full-slip kinematics: an upper estimate of the sliding).
    """
    kx = np.asarray(kappa_x, dtype=np.float64)
    ky = np.asarray(kappa_y, dtype=np.float64)
    if kx.ndim != 1 or kx.shape != ky.shape or kx.size < 2:
        raise CableInputError("curvature histories must be equal 1-D arrays")
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise CableInputError("duration_s must be finite and positive")
    tension = None if tension_N is None else np.asarray(tension_N, dtype=np.float64)
    if reference_tension_N is None:
        peak = 0.0 if tension is None else float(np.max(np.abs(tension)))
        reference_tension_N = peak if peak > 0 else 1.0
    per_year = SECONDS_PER_YEAR / duration_s * time_fraction
    params = conductor_ring_parameters(
        rings,
        E_Pa,
        mu_wire=mu_wire,
        reference_tension_N=reference_tension_N,
        external_pressure_Pa=external_pressure_Pa,
        r_outer_m=r_outer_m,
        pressure_transmission=pressure_transmission,
        fibre="outer",
    )
    outer_load, _inner = interface_line_loads(
        rings,
        E_Pa,
        reference_tension_N / conductor_axial_stiffness(rings, E_Pa),
        external_pressure_Pa=external_pressure_Pa,
        r_outer_m=r_outer_m,
        pressure_transmission=pressure_transmission,
    )
    theta = 2.0 * math.pi * np.arange(n_orientations) / n_orientations
    beta = principal_bending_angle(kx, ky)
    n_r = len(rings)
    cycles = np.zeros((n_r, n_orientations, 3))
    energy = np.zeros((n_r, n_orientations))
    depth = np.zeros((n_r, n_orientations))
    factors = np.zeros(n_r)
    contacts: list[FrettingContact | None] = []
    drive = (
        np.sin(theta)[:, None] * (kx - kx[0])[None, :]
        - np.cos(theta)[:, None] * (ky - ky[0])[None, :]
    )
    for i, (ring, par) in enumerate(zip(rings, params, strict=True)):
        if (
            i + 1 >= n_r
            or ring.r_helix_m == 0
            or rings[i + 1].lay_sign == ring.lay_sign
            or outer_load[i] <= 0
            or par.n_orders == 0
        ):
            contacts.append(None)
            continue
        partner = rings[i + 1]
        density = layer_crossing_density_per_m(
            partner.n_wires,
            ring.r_helix_m,
            ring.lay_angle_rad,
            ring.lay_sign,
            partner.r_helix_m,
            partner.lay_angle_rad,
            partner.lay_sign,
        )
        if density <= 0:
            contacts.append(None)
            continue
        force = float(outer_load[i]) / ring.n_wires / density
        a_c, b_c = crossed_cylinder_curvatures(
            ring.d_wire_m,
            layer_crossing_angle_rad(ring.lay_angle_rad, partner.lay_angle_rad),
            partner_diameter_m=partner.d_wire_m,
        )
        contact = point_contact(
            f"ring{i}/ring{i + 1}",
            force,
            a_c,
            b_c,
            E_Pa,
            nu,
            E_Pa,
            nu,
            mu_wire,
            wear=wear,
            wire_radius_m=0.5 * ring.d_wire_m,
        )
        contacts.append(contact)
        factors[i] = _relative_slip_factor(ring, partner)
        result = strand_slip_strain(par, kx, ky, tension, theta_rad=theta)
        dt = contact.transition_amplitude_m
        for e in range(n_orientations):
            ratio = np.ascontiguousarray(result.mobilisation[0, e])
            slip = result.slipping[0, e]
            rng, _m, count, start, end = rainflow_table(ratio)
            slipped = np.array(
                [
                    bool(np.any(slip[min(a, b) + 1 : max(a, b) + 1]))
                    for a, b in zip(start, end, strict=True)
                ],
                dtype=np.bool_,
            )
            x_a = np.minimum(0.5 * rng, 1.0)
            delta_partial = dt * -np.expm1(
                (2.0 / 3.0) * np.log1p(-np.minimum(x_a, 1.0 - 1e-15))
            )
            kin = factors[i] * _window_half_range(
                np.ascontiguousarray(drive[e]), start, end
            )
            delta = np.where(
                slipped, np.maximum(kin, dt), np.minimum(delta_partial, dt)
            )
            w = mindlin_cycle_energy(delta, mu_wire, force, dt)
            regime = np.where(slipped, GROSS, PARTIAL)
            for c in (PARTIAL, GROSS):
                cycles[i, e, c] = float(np.sum(count[regime == c])) * per_year
            energy[i, e] = float(np.sum(count * w)) * per_year
            if wear is not None:
                g = regime == GROSS
                depth[i, e] = (
                    float(
                        wear.alpha_m3_per_J
                        * np.sum(count[g] * w[g] / contact.swept_area(delta[g]))
                    )
                    * per_year
                )
    provenance = Provenance(
        method=(
            "trellis fretting regime: stick-slip mobilisation as Q/(mu P) "
            "([KarlsenHusoy2026]), Mindlin partial-slip energy ([Fouvry1995] "
            "eq. 10), gross slip from kinematic relative ring slip, energy wear"
        ),
        citations=("[KarlsenHusoy2026]", "[Fouvry1995]", "[Fouvry1996]", "[Poon2023]"),
        limitations=(
            "gross-slip sliding amplitude from full-slip kinematics: upper estimate",
            "constant crossing force at the reference tension",
            "circular Mindlin forms applied to the area-equivalent elliptical contact",
        ),
    )
    return TrellisFrettingResult(
        orientations_deg=np.degrees(theta),
        extreme_fibre_offset_deg=extreme_fibre_offset_deg(theta, beta),
        cycles_by_regime=cycles,
        energy_per_year_J=energy,
        wear_depth_per_year_m=depth,
        relative_slip_factor_m2=factors,
        contacts=tuple(contacts),
        provenance=provenance,
    )


@dataclass(frozen=True)
class CableFrettingResult:
    """Fretting of every assessed face of a cable's helical layers.

    ``faces`` is keyed ``"<layer>/<face>"``. ``extreme_fibre_deg`` is the
    azimuth of the extreme fibre of the principal bending direction of the
    record; ``reference_tension_N`` the tension of the contact loads.
    """

    faces: dict[str, FrettingFace]
    extreme_fibre_deg: float
    reference_tension_N: float
    duration_s: float
    time_fraction: float
    design_life_years: float | None
    provenance: Provenance

    def summary(self) -> dict[str, dict[str, float | str]]:
        """Headline numbers per face (regime shares, wear, damage, location)."""
        out: dict[str, dict[str, float | str]] = {}
        for key, face in self.faces.items():
            share = face.cycles_by_regime.sum(axis=0)
            total = float(share.sum())
            row: dict[str, float | str] = {
                "partner": face.partner,
                "mu": face.contact.mu,
                "line_or_point_load": face.contact.normal_load,
                "half_width_um": 1e6 * face.contact.half_width_m,
                "p0_MPa": face.contact.p0_Pa / 1e6,
                "transition_amplitude_um": 1e6 * face.contact.transition_amplitude_m,
                "max_amplitude_um": 1e6 * float(np.max(face.max_amplitude_m)),
                "max_slip_to_width": float(np.nanmax(face.slip_to_width))
                if np.isfinite(face.slip_to_width).any()
                else math.nan,
                "min_slip_to_width": float(np.nanmin(face.slip_to_width))
                if np.isfinite(face.slip_to_width).any()
                else math.nan,
            }
            for c, name in enumerate(REGIMES):
                row[f"share_{name}"] = float(share[c]) / total if total > 0 else 0.0
            row.update(face.location())
            out[key] = row
        return out
