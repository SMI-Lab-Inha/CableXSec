# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Section properties of a cable per load case, for global analysis.

A load case is an effective tension with the end condition for twist (held or
free) and, optionally, a uniform temperature and an external pressure. For
each case :func:`compute_section_properties` reports what a global analysis
needs and what a designer reads from a cross-section tool:

* axial stiffness for the stated end condition and the torsional stiffness for
  a positive and a negative torque step, from the coupled radial-axial-torsional
  section with unilateral contact (:func:`~cablexsec.mechanics.radial_cable.
  solve_cable_radial_state`) where that model accepts the cable; otherwise the
  fixed-radius values of :func:`~cablexsec.api.compute_stiffness`, labelled as
  such (the torsional stiffness is then the lay contribution only, the same in
  both directions);
* the share of the applied tension carried by each layer and the contact
  pressure at each interface; on the solved route every layer's axial force
  is recovered from its own strain and material law, so no layer is found by
  difference;
* the friction-setting contact pressure ``P1+P2`` of each armour layer, its
  slip-onset curvature (:func:`~cablexsec.mechanics.papailiou.
  slip_onset_curvature`, the neutral-axis limit) and its full-slip curvature
  (:func:`~cablexsec.mechanics.papailiou.full_slip_curvature`, the last wire to
  slip), both with the stick stiffness of the bending law;
* the monotonic moment-curvature backbone of :func:`~cablexsec.api.
  compute_mkappa` and its tangent against curvature (the staircase);
* optionally the core lay stage (``core_friction``): the laid power cores
  stick to the section and carry their parallel-axis term until the friction
  of their contacts lets them slip, as the armour wires do. The cores are the
  helical elements of the core lay (the treatment of every helix element of a
  layer in [Skeie2012] Section 2.4), with the Papailiou cap of
  :mod:`cablexsec.mechanics.core_friction` (gap R1b) and the core contact
  line load of the same contact model as the armour pressure. Their slip
  onset is ``mu q / (xi EA_core sin(beta) cos(beta))`` ([Skeie2012]
  Eq. 2.31 with ``f = mu q cos(beta)`` per core length) and their full slip
  ``q expm1(mu sin(beta) pi/2) / (xi EA_core sin(beta)^2 cos(beta))``
  ([Papailiou1997] Eq. 15 at the extreme fibre; the factor tends to the
  ``pi/2`` of [Skeie2012] Eq. 2.35 as the friction tends to zero).

Case-independent values are reported once: the three bending stiffness
bounds and each layer's share of them (:func:`~cablexsec.mechanics.stiffness.
bending_stiffness_by_layer`) and the mass, weight in water and outer
diameter (:func:`compute_mass_properties`).

:func:`compute_load_sharing` traces the axial force and share of every layer
at tensions from zero to a chosen maximum, for one end condition,
temperature and pressure (the load sharing graph of cross-section tools).

Tension and pressure. The tension of a case is the effective tension. With
an external pressure ``p`` a sealed (dry) annulus is loaded by the true wall
tension ``T_e - p A_o`` (``A_o`` the area of the outer diameter), as in
:func:`~cablexsec.analysis.compression.true_wall_tension`; a flooded annulus
is loaded by the effective tension, the pressure acting on every solid (the
solved section's flooded model), and in the bounded contact chain the
pressure then gives no clamping, as in the capacity and compression routes.

References
----------
[WitzTan1992a], [WitzTan1992b] (fixed-radius bounds); [Papailiou1997] (slip
front); [Yun2020] (shear coefficient); [FeretBournazel1987] (manufacturing
pressure); [Skeie2012] (helix elements of a layer, core lay stage); the
solved section of doc/radial_model.md.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from cablexsec.analysis.compression import _outer_radius, true_wall_tension
from cablexsec.cable_adapters import (
    cable_to_mkappa_specs,
    require_fully_specified,
    structural_armour_layers,
)
from cablexsec.cables.schema import CableData
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.materials import library
from cablexsec.mechanics.core_friction import (
    CoreFrictionSpec,
    build_core_friction,
    resolve_core_friction,
)
from cablexsec.mechanics.papailiou import full_slip_curvature, slip_onset_curvature
from cablexsec.mechanics.radial_cable import (
    CableRadialState,
    solve_cable_radial_states,
)
from cablexsec.mechanics.section_geometry import material_volume_per_length
from cablexsec.mechanics.section_state import SectionState, solve_section_state
from cablexsec.mechanics.stiffness import bending_stiffness_by_layer
from cablexsec.provenance import Provenance

__all__ = [
    "GRAVITY_M_S2",
    "SEAWATER_DENSITY_KG_M3",
    "ArmourSlip",
    "BendingShare",
    "CoreSlip",
    "LoadSharingResult",
    "MassProperties",
    "PUBLISHED_TOLERANCE",
    "PublishedComparison",
    "SectionCaseResult",
    "SectionLoadCase",
    "SectionPropertiesResult",
    "compute_load_sharing",
    "compare_published",
    "compute_mass_properties",
    "compute_section_properties",
    "tangent_bending_stiffness",
]

FloatArray = NDArray[np.float64]
TwistEnd = Literal["held", "free"]

GRAVITY_M_S2 = 9.80665
"""Standard gravity [m/s²]."""
SEAWATER_DENSITY_KG_M3 = 1025.0
"""Default sea-water density [kg/m³] (the CableDyn default)."""

_LOOSE_LAYERS = "neighbour"
"""Representation of loose core screens in the solved section, as in
:func:`~cablexsec.analysis.torque_balance.torque_balance`: the fixed-radius
stiffness gives them no axial load, so they take the material of their softer
structural neighbour instead of a solid annulus of their own material."""


# --------------------------------------------------------------------------- #
# Inputs                                                                      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SectionLoadCase:
    """One load case of the section.

    Attributes
    ----------
    name : str
        Name shown with the results.
    tension_N : float
        Effective tension [N], not negative (the contact models need tension).
    twist : {"held", "free"}
        End condition for twist: ``"held"`` restrains the twist (fixed ends),
        ``"free"`` lets the section twist with zero end torque.
    temperature_C : float or None
        Uniform temperature of every layer [°C]; ``None`` is the reference
        temperature of the materials (no thermal strain).
    external_pressure_Pa : float or None
        External pressure [Pa]; ``None`` keeps the cable's documented
        initial-state pressure, as every other contact route does.
    """

    name: str
    tension_N: float
    twist: TwistEnd = "held"
    temperature_C: float | None = None
    external_pressure_Pa: float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise CableInputError("a load case needs a name")
        if not math.isfinite(self.tension_N) or self.tension_N < 0.0:
            raise CableInputError(
                f"load case {self.name!r}: the tension must be finite and not negative"
            )
        if self.twist not in ("held", "free"):
            raise CableInputError(
                f"load case {self.name!r}: twist must be 'held' or 'free'"
            )
        if self.temperature_C is not None and not math.isfinite(self.temperature_C):
            raise CableInputError(f"load case {self.name!r}: temperature not finite")
        pressure = self.external_pressure_Pa
        if pressure is not None and (not math.isfinite(pressure) or pressure < 0.0):
            raise CableInputError(
                f"load case {self.name!r}: the external pressure must be finite "
                "and not negative"
            )

    def describe(self) -> str:
        """The case in words: tension, end condition and optional loads."""
        parts = [f"{self.tension_N / 1e3:g} kN", f"twist {self.twist}"]
        if self.temperature_C is not None:
            parts.append(f"{self.temperature_C:g} °C")
        if self.external_pressure_Pa is not None:
            parts.append(f"external pressure {self.external_pressure_Pa / 1e6:g} MPa")
        return ", ".join(parts)


# --------------------------------------------------------------------------- #
# Mass, weight and outer diameter                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MassProperties:
    """Mass per length, weight in water and outer diameter of a cable.

    The material volume per cable length of each layer is
    :func:`~cablexsec.mechanics.section_geometry.material_volume_per_length`
    (helical wires along their lay, laid cores along their lay, fillers
    outside the core envelopes); the mass is that volume times the material's
    density. A layer whose material has no density makes the total ``NaN``
    and is listed in ``missing_density``.
    """

    cable_name: str
    layer_area_m2: Mapping[str, float]
    """Material volume per cable length [m²] of each layer."""
    layer_mass_kg_per_m: Mapping[str, float]
    """Mass per length [kg/m] of each layer (``NaN`` without a density)."""
    density_source: Mapping[str, str]
    """Citation of each layer's density, as the material states it."""
    missing_density: tuple[str, ...]
    outer_diameter_m: float
    water_density_kg_m3: float
    gravity_m_s2: float
    notes: tuple[str, ...] = ()

    @property
    def mass_kg_per_m(self) -> float:
        """Dry mass per length [kg/m]: solids only, voids empty."""
        if self.missing_density:
            return math.nan
        return float(sum(self.layer_mass_kg_per_m.values()))

    @property
    def solid_area_m2(self) -> float:
        """Material volume per cable length of all layers [m²]."""
        return float(sum(self.layer_area_m2.values()))

    @property
    def outer_area_m2(self) -> float:
        """Area of the outer diameter [m²] (the displaced volume per length)."""
        return math.pi * self.outer_diameter_m**2 / 4.0

    @property
    def void_area_m2(self) -> float:
        """Area inside the outer diameter not filled by a declared material [m²]."""
        return self.outer_area_m2 - self.solid_area_m2

    @property
    def weight_in_air_N_per_m(self) -> float:
        """Dry weight per length in air [N/m]."""
        return self.mass_kg_per_m * self.gravity_m_s2

    @property
    def submerged_weight_dry_voids_N_per_m(self) -> float:
        """Weight in water with the voids dry (sealed) [N/m]: ``(m - rho A_o) g``."""
        return (
            self.mass_kg_per_m - self.water_density_kg_m3 * self.outer_area_m2
        ) * self.gravity_m_s2

    @property
    def submerged_weight_flooded_voids_N_per_m(self) -> float:
        """Weight in water with every void flooded [N/m]: ``(m - rho A_s) g``."""
        return (
            self.mass_kg_per_m - self.water_density_kg_m3 * self.solid_area_m2
        ) * self.gravity_m_s2


def compute_mass_properties(
    cable: CableData,
    *,
    water_density_kg_m3: float = SEAWATER_DENSITY_KG_M3,
    gravity_m_s2: float = GRAVITY_M_S2,
) -> MassProperties:
    """Mass per length, weight in water and outer diameter of ``cable``.

    Parameters
    ----------
    cable : CableData
        A fully specified cable.
    water_density_kg_m3, gravity_m_s2 : float, optional
        Sea-water density and gravity used for the weight in water.

    Returns
    -------
    MassProperties
        The outer diameter is the largest outer radius of the declared
        layers, doubled; the voids are the rest of the outer area (conductor
        interstices, gaps between wires and cores).
    """
    require_fully_specified(cable, "compute_mass_properties")
    for value, what in (
        (water_density_kg_m3, "water density"),
        (gravity_m_s2, "gravity"),
    ):
        if not (math.isfinite(value) and value > 0.0):
            raise CableInputError(f"the {what} must be finite and positive")
    areas: dict[str, float] = {}
    masses: dict[str, float] = {}
    sources: dict[str, str] = {}
    missing: list[str] = []
    notes: list[str] = []
    for layer in cable.layers:
        area = material_volume_per_length(layer, cable)
        areas[layer.name] = area
        try:
            density = library.get_property(layer.material, "rho_kg_m3")
            sources[layer.name] = library.get_citation(layer.material, "rho_kg_m3")
        except KeyError:
            masses[layer.name] = math.nan
            missing.append(layer.name)
            continue
        masses[layer.name] = density * area
        if getattr(layer, "f_corrugation", 1.0) != 1.0:
            notes.append(
                f"{layer.name}: mass from the smooth-wall volume at the mean "
                "radius; the corrugation profile is not resolved"
            )
    return MassProperties(
        cable_name=cable.cable.name,
        layer_area_m2=areas,
        layer_mass_kg_per_m=masses,
        density_source=sources,
        missing_density=tuple(missing),
        outer_diameter_m=2.0 * _outer_radius(cable),
        water_density_kg_m3=float(water_density_kg_m3),
        gravity_m_s2=float(gravity_m_s2),
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------- #
# Results                                                                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ArmourSlip:
    """Stick-slip data of one armour layer in one load case."""

    layer: str
    contact_pressure_Pa: float
    """Friction-setting pressure ``P1+P2`` the bending law uses [Pa]."""
    mu: float
    xi: float
    """Shear coefficient in the stick stiffness ``xi E cos² a`` of the law."""
    onset_curvature_1_m: float
    """Neutral-axis limit of the slip onset (a lower bound for the wires)."""
    full_slip_curvature_1_m: float
    """Curvature from which every wire of the layer slips."""
    stick_bending_stiffness_Nm2: float
    """The layer's stick term ``1/2 xi n E A r² cos³ a`` in the law [N m²]."""


@dataclass(frozen=True)
class CoreSlip:
    """Stick-slip data of the laid power cores in one load case (core lay stage).

    From the core population of
    :func:`~cablexsec.mechanics.core_friction.build_core_friction` that the
    bending law uses in the case.
    """

    contact_line_load_N_per_m: float
    """Sum of the normal contact line loads on one core [N per m of cable]."""
    contact_source: str
    """``"bounded"`` (estimate from the armour pressure chain) or ``"solved"``."""
    mu: float
    """Load-weighted friction coefficient of the core contacts."""
    xi: float
    """Stick coefficient of a core (insulation shear and contact penalty)."""
    onset_curvature_1_m: float
    """Neutral-axis limit of the core slip onset."""
    full_slip_curvature_1_m: float
    """Curvature from which every core slips."""
    stick_bending_stiffness_Nm2: float
    """Stick term of the cores ``xi 1/2 sum EA r_c² cos³ beta`` (and of
    resolved internal wire rings) in the law [N m²]."""


@dataclass(frozen=True)
class BendingShare:
    """One layer's bending stiffness under the three kinematic definitions [N m²]."""

    layer: str
    full_stick_Nm2: float
    no_slip_shear_compliant_Nm2: float
    full_slip_Nm2: float


@dataclass(frozen=True)
class SectionCaseResult:
    """Section properties of one load case.

    ``route`` is ``"solved"`` when the axial and torsional values come from
    the coupled radial section, ``"fixed_radius"`` otherwise. Values a route
    cannot give are ``NaN``.
    """

    case: SectionLoadCase
    route: str
    route_reason: str
    wall_tension_N: float
    """The tension given to the section models after the pressure conversion."""
    axial_stiffness_N: float
    """Tangent axial stiffness for the case's end condition [N]."""
    axial_stiffness_secant_N: float
    """Secant from zero tension, same end condition and loads [N]."""
    torque_change_Nm: float
    """Torque change from zero tension with the twist held [N m]."""
    twist_change_rad_per_m: float
    """Twist change from zero tension with the twist free [rad/m]."""
    torsional_stiffness_positive_Nm2: float
    torsional_stiffness_negative_Nm2: float
    torsional_stiffness_tangent_Nm2: float
    torque_step_Nm: float
    load_share_N: Mapping[str, float]
    load_share_basis: str
    interface_pressure_Pa: Mapping[str, float]
    interface_radius_m: Mapping[str, float]
    armour: tuple[ArmourSlip, ...]
    bending_contact: str
    kappa_1_m: FloatArray
    moment_Nm: FloatArray
    ei_stick_Nm2: float
    """Initial tangent of the backbone: every armour wire sticks, and with
    the core lay stage every laid core too [N m²]."""
    ei_slip_Nm2: float
    """Final tangent: every armour wire slips (individual bending) [N m²]."""
    notes: tuple[str, ...] = ()
    cores: CoreSlip | None = None
    """Core lay stage data; ``None`` when the stage is off."""
    wall_tension_basis: str = ""
    """How :attr:`wall_tension_N` follows from the effective tension: the
    pressure conversion of the annulus setting and the area it uses."""
    mechanical_force_N: float = math.nan
    """The axial force the layers carry mechanically [N]: in a flooded
    annulus the wall tension plus the water pressure on the solid area (the
    effective tension), otherwise the wall tension. Contact pressures,
    torque, slip curvatures and the coupling follow it."""

    @property
    def tangent(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        """``(kappa_low, kappa_high, EI)`` of every backbone interval."""
        return tangent_bending_stiffness(self.kappa_1_m, self.moment_Nm)

    def backbone_table_kNm(self) -> list[tuple[float, float]]:
        """Curvature [rad/m] and moment [kN m] of the backbone."""
        return [
            (float(k), float(m) / 1e3)
            for k, m in zip(self.kappa_1_m, self.moment_Nm, strict=True)
        ]


PUBLISHED_TOLERANCE = 0.10
"""Default relative difference above which a computed value is flagged
against the cable's published value."""


@dataclass(frozen=True)
class PublishedComparison:
    """One published property of the cable beside its computed counterpart.

    Values are in SI base units (``unit`` names them). ``condition`` is the
    published value's stated condition (a tension) or empty; ``counterpart``
    says which computed value stands beside it and why it is the one to
    compare. The published value is a comparison target, never an input.
    """

    key: str
    """``mass``, ``outer_diameter``, ``axial_stiffness``,
    ``axial_stiffness_exported``, ``bending_full_slip``, ``bending_no_slip``
    or ``torsional_stiffness``."""
    quantity: str
    published: float
    computed: float
    unit: str
    condition: str
    counterpart: str
    tolerance: float

    @property
    def difference(self) -> float:
        """``computed / published - 1`` (``NaN`` without a computed value)."""
        if not (math.isfinite(self.computed) and self.published > 0.0):
            return math.nan
        return self.computed / self.published - 1.0

    @property
    def flagged(self) -> bool:
        """Whether the difference exceeds the tolerance."""
        return math.isfinite(self.difference) and abs(self.difference) > (
            self.tolerance + 1e-12
        )

    def describe(self) -> str:
        """One line: quantity, both values, the counterpart and the difference."""
        condition = f" {self.condition}" if self.condition else ""
        return (
            f"{self.quantity}: published {self.published:.6g} {self.unit}"
            f"{condition}; computed {self.computed:.6g} {self.unit}, "
            f"{self.counterpart}; difference {100.0 * self.difference:+.1f} %"
        )


@dataclass(frozen=True)
class SectionPropertiesResult:
    """Section properties of a cable at a list of load cases."""

    cable_name: str
    cases: tuple[SectionCaseResult, ...]
    ei_full_stick_Nm2: float
    ei_no_slip_shear_compliant_Nm2: float
    ei_full_slip_Nm2: float
    ea_fixed_radius_held_N: float
    ea_fixed_radius_free_N: float
    gj_lay_Nm2: float
    xi_per_layer: Mapping[str, float]
    bending_share: tuple[BendingShare, ...]
    mass: MassProperties
    settings: Mapping[str, Any] = field(default_factory=dict)
    provenance: Provenance | None = None
    notes: tuple[str, ...] = ()
    published: tuple[PublishedComparison, ...] = ()
    """The cable's published properties beside the computed values
    (:func:`compare_published`); empty when the cable publishes none."""
    min_bend_radius_dynamic_m: float | None = None
    """The cable's declared dynamic minimum bend radius [m] (``None``: not
    declared)."""
    min_bend_radius_installation_m: float | None = None
    """The cable's declared installation minimum bend radius [m]."""

    @property
    def published_flagged(self) -> tuple[PublishedComparison, ...]:
        """The published properties the computed values differ from by more
        than the tolerance."""
        return tuple(item for item in self.published if item.flagged)


def _held_case_at(
    result: SectionPropertiesResult, tension_N: float
) -> SectionCaseResult | None:
    """A case of ``result`` at ``tension_N``, twist held, no temperature or pressure."""
    for case in result.cases:
        load = case.case
        if (
            load.twist == "held"
            and abs(load.tension_N - tension_N) <= 1e-6 * max(1.0, tension_N)
            and load.temperature_C is None
            and load.external_pressure_Pa is None
        ):
            return case
    return None


def compare_published(
    cable: CableData,
    result: SectionPropertiesResult,
    *,
    tolerance: float = PUBLISHED_TOLERANCE,
    case_at: Callable[[float], SectionCaseResult | None] | None = None,
) -> tuple[PublishedComparison, ...]:
    """The cable's published properties beside their computed counterparts.

    Each published value of ``cable.aggregate_properties`` is set beside the
    computed value of the same definition:

    * mass per length: the dry mass of the declared layers (solids only);
    * outer diameter: twice the largest outer radius of the declared layers;
    * axial stiffness (ends fixed): the fixed-radius assembly with the twist
      held, independent of the tension (no tension is stated with it), and
      beside it the value a global model receives, the tangent of the first
      twist-held load case without temperature or pressure;
    * full-slip bending stiffness: at a stated tension the full-slip tangent
      of the backbone at that tension (every armour wire slips), otherwise
      the full-slip bound (every wire and core about its own axis);
    * no-slip bending stiffness: the full-stick bound (plane sections), the
      manufacturers' definition;
    * torsional stiffness: at a stated tension the tangent at constant
      tension of the twist-held state at that tension (solved section where
      it accepts the cable), otherwise the fixed-radius lay term.

    ``case_at`` returns the twist-held case at a tension; by default a case
    of ``result`` with that tension, the twist held and no temperature or
    pressure. Without one the tension-independent counterpart is used and
    the row says so.
    """
    if not (math.isfinite(tolerance) and tolerance > 0.0):
        raise CableInputError("the tolerance must be finite and positive")
    published = cable.aggregate_properties
    mass = result.mass
    rows: list[PublishedComparison] = []

    def add(
        key: str,
        quantity: str,
        value: float | None,
        scale: float,
        computed: float,
        unit: str,
        counterpart: str,
        condition: str = "",
    ) -> None:
        if value is None:
            return
        rows.append(
            PublishedComparison(
                key,
                quantity,
                float(value) * scale,
                float(computed),
                unit,
                condition,
                counterpart,
                float(tolerance),
            )
        )

    def at(tension_kN: float | None) -> SectionCaseResult | None:
        if tension_kN is None:
            return None
        tension = float(tension_kN) * 1e3
        found = _held_case_at(result, tension)
        if found is None and case_at is not None:
            found = case_at(tension)
        return found

    add(
        "mass",
        "Mass per length",
        published.mass_kg_m,
        1.0,
        mass.mass_kg_per_m,
        "kg/m",
        "dry mass of the declared layers (solids only, voids empty)",
    )
    add(
        "outer_diameter",
        "Outer diameter",
        published.OD_mm,
        1e-3,
        mass.outer_diameter_m,
        "m",
        "twice the largest outer radius of the declared layers",
    )
    add(
        "axial_stiffness",
        "Axial stiffness, ends fixed",
        published.EA_fixed_MN,
        1e6,
        result.ea_fixed_radius_held_N,
        "N",
        "twist held, fixed-radius assembly (no tension is stated with the "
        "published value; this value is independent of the tension)",
    )
    exported = next(
        (
            case
            for case in result.cases
            if case.case.twist == "held"
            and case.case.temperature_C is None
            and case.case.external_pressure_Pa is None
        ),
        None,
    )
    if exported is not None:
        # The value a global model receives: the tangent of the load case
        # (solved section where it accepts the cable), beside the
        # tension-independent fixed-radius value.
        add(
            "axial_stiffness_exported",
            "Axial stiffness, ends fixed, as exported",
            published.EA_fixed_MN,
            1e6,
            exported.axial_stiffness_N,
            "N",
            f"tangent of load case {exported.case.name} "
            f"({exported.case.tension_N / 1e3:g} kN, twist held, "
            + ("solved section" if exported.route == "solved" else "fixed radius")
            + "): the axial stiffness a global model receives",
        )
    tension_ei = published.EI_min_tension_kN
    case = at(tension_ei)
    if published.EI_min_kNm2 is not None:
        if case is not None:
            counterpart = (
                f"full-slip tangent of the backbone at {tension_ei:g} kN "
                "(every armour wire slips)"
            )
            value = case.ei_slip_Nm2
        else:
            counterpart = (
                "full-slip bound: every wire and core bends about its own axis"
                + (f" (no case at {tension_ei:g} kN)" if tension_ei is not None else "")
            )
            value = result.ei_full_slip_Nm2
        add(
            "bending_full_slip",
            "Bending stiffness, full slip",
            published.EI_min_kNm2,
            1e3,
            value,
            "N m2",
            counterpart,
            f"at {tension_ei:g} kN" if tension_ei is not None else "",
        )
    add(
        "bending_no_slip",
        "Bending stiffness, no slip",
        published.EI_max_kNm2,
        1e3,
        result.ei_full_stick_Nm2,
        "N m2",
        "full-stick bound (plane sections), the manufacturers' definition; "
        "independent of the tension",
    )
    tension_gj = published.GJ_tension_kN
    case = at(tension_gj)
    if published.GJ_kNm2 is not None:
        if case is not None and case.route == "solved":
            counterpart = (
                f"tangent at constant tension at {tension_gj:g} kN, twist "
                "held, solved section"
            )
            value = case.torsional_stiffness_tangent_Nm2
        else:
            counterpart = "lay contribution only, fixed radius" + (
                f" (the solved section does not accept the cable at {tension_gj:g} kN)"
                if case is not None
                else f" (no case at {tension_gj:g} kN)"
                if tension_gj is not None
                else ""
            )
            value = result.gj_lay_Nm2
        add(
            "torsional_stiffness",
            "Torsional stiffness",
            published.GJ_kNm2,
            1e3,
            value,
            "N m2",
            counterpart,
            f"at {tension_gj:g} kN" if tension_gj is not None else "",
        )
    return tuple(rows)


def tangent_bending_stiffness(
    kappa_1_m: FloatArray, moment_Nm: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Tangent ``dM/dkappa`` of each interval of a backbone [N m²].

    Returns the lower and upper curvature of each interval and the slope
    over it, so that a step plot draws the staircase without interpolation.
    """
    kappa = np.asarray(kappa_1_m, dtype=np.float64)
    moment = np.asarray(moment_Nm, dtype=np.float64)
    if kappa.ndim != 1 or kappa.shape != moment.shape or kappa.size < 2:
        raise CableInputError("a backbone needs matching curvature and moment arrays")
    step = np.diff(kappa)
    if np.any(step <= 0.0):
        raise CableInputError("the backbone curvature must rise strictly")
    return kappa[:-1], kappa[1:], np.diff(moment) / step


# --------------------------------------------------------------------------- #
# Computation                                                                 #
# --------------------------------------------------------------------------- #
def _radial_options(case: SectionLoadCase, annulus: str) -> dict[str, Any]:
    options: dict[str, Any] = {
        "loose_layers": _LOOSE_LAYERS,
        "temperature_C": case.temperature_C,
    }
    pressure = case.external_pressure_Pa
    if pressure is not None:
        if annulus == "flooded":
            options.update(flooded_pressure_Pa=pressure, external_pressure_Pa=0.0)
        else:
            options.update(external_pressure_Pa=pressure)
    return options


def _twist_options(twist: str, torque_Nm: float = 0.0) -> dict[str, Any]:
    if twist == "held":
        return {"prescribed_twist_per_m": 0.0}
    return {"prescribed_twist_per_m": None, "torque_Nm": torque_Nm}


def _wall_tensions(
    cable: CableData,
    case: SectionLoadCase,
    annulus: str,
    effective: Sequence[float],
    options: Mapping[str, Any],
) -> list[float]:
    """Tensions to give the solved section so that it carries ``effective``."""
    pressure = case.external_pressure_Pa
    if pressure is None or pressure == 0.0:
        return [float(t) for t in effective]
    if annulus == "sealed":
        area = math.pi * _outer_radius(cable) ** 2
        return [true_wall_tension(float(t), pressure, area) for t in effective]
    # Flooded: the mechanical axial force is the wall tension plus p times
    # the solid area, a constant of the section; one solve finds it.
    probe = solve_cable_radial_states(
        cable, [0.0], **options, **_twist_options("held")
    )[0]
    assert probe.mechanical_axial_force_N is not None
    offset = probe.mechanical_axial_force_N - probe.tension_N
    return [float(t) - offset for t in effective]


@dataclass(frozen=True)
class _Solved:
    zero: CableRadialState
    state: CableRadialState
    axial_stiffness: float
    secant: float
    torque_change: float
    twist_change: float
    gj_positive: float
    gj_negative: float
    gj_tangent: float
    load_share: dict[str, float]
    interface_pressure: dict[str, float]
    interface_radius: dict[str, float]


def _layer_order(cable: CableData, names: Sequence[str]) -> list[str]:
    """``names`` in the order of the cable's layers (inner to outer), others last."""
    order = [layer.name for layer in cable.layers if layer.name in names]
    return order + [name for name in names if name not in order]


def _force_change(
    cable: CableData, zero: CableRadialState, state: CableRadialState
) -> dict[str, float]:
    """Change of each layer's axial force from ``zero`` to ``state`` [N].

    Every layer's force is recovered from its own strain and constitutive
    law (:meth:`~cablexsec.mechanics.radial_cable.CableRadialState.
    layer_axial_force`), so the changes sum to the change of the section's
    axial force; none is found by difference.
    """
    before, after = zero.layer_axial_force(), state.layer_axial_force()
    return {
        name: after[name] - before.get(name, 0.0)
        for name in _layer_order(cable, list(after))
    }


def _solve_case(
    cable: CableData, case: SectionLoadCase, annulus: str, torque_step: float
) -> _Solved:
    options = _radial_options(case, annulus)
    zero_t, case_t = _wall_tensions(
        cable, case, annulus, [0.0, case.tension_N], options
    )
    zero, state = solve_cable_radial_states(
        cable, [zero_t, case_t], **options, **_twist_options(case.twist)
    )
    result, reference = state.result, zero.result
    held = case.twist == "held"
    axial = result.ea_fixed_twist_N if held else result.ea_free_twist_N
    strain = result.axial_strain - reference.axial_strain
    secant = case.tension_N / strain if case.tension_N > 0.0 and strain else math.nan
    base = result.torque_Nm if held else 0.0
    twists = []
    for torque in (base, base + torque_step, base - torque_step):
        twisted = solve_cable_radial_states(
            cable, [case_t], **options, **_twist_options("free", torque)
        )[0]
        twists.append(twisted.result.twist_per_m)
    gj_positive = torque_step / (twists[1] - twists[0])
    gj_negative = torque_step / (twists[0] - twists[2])
    share = _force_change(cable, zero, state)
    pressure: dict[str, float] = {}
    radius: dict[str, float] = {}
    for (inner, outer), r, force in zip(
        result.interface_names,
        result.interface_radius_m,
        result.interface_line_force_N_per_m,
        strict=True,
    ):
        name = f"{inner} / {outer}"
        pressure[name] = float(force) / (2.0 * math.pi * float(r))
        radius[name] = float(r)
    return _Solved(
        zero,
        state,
        float(axial),
        float(secant),
        float(result.torque_Nm - reference.torque_Nm) if held else math.nan,
        float(result.twist_per_m - reference.twist_per_m) if not held else math.nan,
        float(gj_positive),
        float(gj_negative),
        float(result.gj_fixed_tension_Nm2),
        share,
        pressure,
        radius,
    )


def _bounded_state(
    cable: CableData, case: SectionLoadCase, annulus: str, tension_N: float
) -> SectionState | None:
    """Bounded-chain section state when the case has loads beyond tension."""
    pressure = case.external_pressure_Pa
    sealed_pressure = pressure is not None and annulus == "sealed"
    if case.temperature_C is None and not sealed_pressure:
        return None
    return solve_section_state(
        cable,
        tension_N=tension_N,
        p_ext_Pa=pressure if sealed_pressure else None,
        temperature_C=case.temperature_C,
        contact_model="bounded",
    )


def _bounded_wall_tension(
    cable: CableData, case: SectionLoadCase, annulus: str
) -> float:
    pressure = case.external_pressure_Pa
    if pressure is not None and annulus == "sealed":
        area = math.pi * _outer_radius(cable) ** 2
        return true_wall_tension(case.tension_N, pressure, area)
    return case.tension_N


def _wall_tension_basis(
    case: SectionLoadCase, annulus: str, route: str, wall: float, solid_area: float
) -> str:
    """How the tension given to the section follows from the effective tension.

    ``solid_area`` is the solids' area of the mass [m²], named beside the
    area of the solved section that a flooded case uses.
    """
    pressure = case.external_pressure_Pa
    if pressure is None or pressure == 0.0:
        return "the effective tension (no external pressure)"
    if annulus == "sealed":
        area = (case.tension_N - wall) / pressure
        return (
            f"true wall tension Te − p Ao with Ao {area:.6g} m2, the area of the "
            "outer diameter (sealed annulus: the pressure acts on the outer "
            "sheath)"
        )
    if route != "solved":
        return (
            "the effective tension (flooded annulus: the pressure acts on every "
            "solid and gives no clamping; the fixed-radius assembly takes the "
            "effective tension)"
        )
    area = (case.tension_N - wall) / pressure
    return (
        f"true wall tension Te − p As with As {area:.6g} m2, the solid area of "
        "the solved section (flooded annulus: the water pressure acts on every "
        "solid, which then carry the effective tension as their mechanical "
        "force, and gives no clamping of the armour; the solids of the mass, "
        f"from the declared layers, have {solid_area:.6g} m2)"
    )


def _secant_note(case: SectionLoadCase, solved: _Solved) -> str:
    """Why the secant from zero tension differs from the tangent, if it does."""
    secant, tangent = solved.secant, solved.axial_stiffness
    if not (math.isfinite(secant) and tangent > 0.0):
        return ""
    if abs(secant - tangent) <= 0.02 * tangent:
        return ""
    before = solved.zero.result.active_contacts
    after = solved.state.result.active_contacts
    loads = (
        "case temperature and pressure"
        if case.temperature_C is not None or case.external_pressure_Pa
        else "same loads"
    )
    change = (
        f"the active contacts of the solved section go from {before} at zero "
        f"tension to {after} at the case tension"
        if before != after
        else "the solved section's contact state changes"
    )
    return (
        f"axial stiffness: the secant from zero tension ({secant / 1e6:.4g} MN) "
        f"is the case tension over the strain change from zero tension at the "
        f"{loads}, so a free thermal strain cancels; it differs from the "
        f"tangent at the case ({tangent / 1e6:.4g} MN) because the section's "
        f"stiffness changes between them: {change}"
    )


_SOLVED_BASIS = (
    "solved radial section: each layer's axial force from its own strain and "
    "material law (axial strain, twist and radial displacements), the three "
    "cores of a layer together"
)
_FIXED_RADIUS_BASIS = (
    "fixed radius, equal axial strain of every layer (the share the bounded "
    "contact chain uses)"
)


def _fixed_radius_forces(
    cable: CableData, case: SectionLoadCase, annulus: str, tensions: Sequence[float]
) -> list[dict[str, float]]:
    """Axial force of each layer at each effective tension, equal axial strain.

    The bounded section state at the wall tension of each value: the case's
    pressure conversion applies, and the twist is held whatever the case says
    (the fixed-radius assembly has no twist freedom).
    """
    pressure = case.external_pressure_Pa
    sealed = pressure is not None and annulus == "sealed"
    states = [
        solve_section_state(
            cable,
            tension_N=_bounded_wall_tension(
                cable, replace(case, tension_N=float(value)), annulus
            ),
            p_ext_Pa=pressure if sealed else None,
            temperature_C=case.temperature_C,
            contact_model="bounded",
        )
        for value in tensions
    ]
    return [
        {name: float(layer.axial_force_N) for name, layer in state.layers.items()}
        for state in states
    ]


def _fixed_radius_share(
    cable: CableData, case: SectionLoadCase, annulus: str
) -> dict[str, float]:
    """Change of each layer's force from zero tension, equal axial strain."""
    zero, loaded = _fixed_radius_forces(cable, case, annulus, [0.0, case.tension_N])
    return {
        name: loaded[name] - zero[name] for name in loaded if loaded[name] or zero[name]
    }


def _curvature_grid(
    kappa_max: float, n_points: int, nodes: Sequence[float]
) -> FloatArray:
    """Zero, a geometric grid up to ``kappa_max`` and the given nodes."""
    grid = np.geomspace(kappa_max * 1e-4, kappa_max, n_points - 1)
    extra = [k for k in nodes if 0.0 < k < kappa_max and math.isfinite(k)]
    return np.unique(np.concatenate(([0.0], grid, extra)))


def _spec_settings(spec: CoreFrictionSpec) -> dict[str, Any]:
    """The core lay stage inputs as JSON-ready settings (rigid contact: None)."""
    stiffness = spec.interface_stiffness_Pa_per_m
    return {
        "mu_core_core": spec.mu_core_core,
        "mu_core_filler": spec.mu_core_filler,
        "mu_core_sheath": spec.mu_core_sheath,
        "interface_stiffness_Pa_per_m": stiffness if math.isfinite(stiffness) else None,
        "core_shear": spec.core_shear,
        "strand_slip": spec.strand_slip,
    }


def _core_slip(
    cable: CableData, spec: CoreFrictionSpec, mkappa_args: Mapping[str, Any]
) -> CoreSlip:
    """Core lay stage data of one case, from the population the law uses.

    The contact state is the one :func:`~cablexsec.api.compute_mkappa`
    receives in ``mkappa_args``, so the core contact line load comes from the
    same contact model as the armour pressure ``P1+P2``.
    """
    radial: CableRadialState | None = mkappa_args.get("radial_state")
    state: SectionState | None = mkappa_args.get("section_state")
    if radial is not None:
        tension, contact = radial.tension_N, "solved"
    elif state is not None:
        tension, contact = state.tension_N, state.contact_model
    else:
        tension, contact = mkappa_args["tension_N"], mkappa_args["contact_model"]
    try:
        model = build_core_friction(
            cable,
            spec,
            tension,
            contact_model=contact,
            radial_state=radial,
            section_state=state,
        )
    except CableSchemaError as exc:
        raise CableSchemaError(
            f"the core lay stage needs one family of laid cores ({exc})"
        ) from exc
    population = model.population
    return CoreSlip(
        model.loads.total_N_per_m,
        model.loads.source,
        model.mu_effective,
        model.xi,
        slip_onset_curvature(population),
        full_slip_curvature(population),
        model.stick_EI_Nm2,
    )


def compute_section_properties(
    cable: CableData,
    load_cases: Sequence[SectionLoadCase],
    *,
    kappa_max_1_m: float = 0.2,
    n_points: int = 120,
    mu: float | None = None,
    bending_contact: str = "bounded",
    annulus: str = "flooded",
    torque_step_Nm: float = 1.0e3,
    water_density_kg_m3: float = SEAWATER_DENSITY_KG_M3,
    progress: Callable[[float, str], None] | None = None,
    core_friction: bool | CoreFrictionSpec | None = None,
    published_tolerance: float | None = PUBLISHED_TOLERANCE,
) -> SectionPropertiesResult:
    """Section properties of ``cable`` at each load case.

    Parameters
    ----------
    cable : CableData
        A fully specified cable with at least one structural armour layer.
    load_cases : sequence of SectionLoadCase
        The cases, with unique names.
    kappa_max_1_m : float, optional
        Largest curvature of the backbones [1/m].
    n_points : int, optional
        Points of the geometric curvature grid (the slip curvatures of every
        armour layer below ``kappa_max_1_m`` are added as nodes).
    mu : float, optional
        Armour friction coefficient; ``None`` reads each armour material's.
    bending_contact : {"bounded", "solved"}, optional
        Contact model of the bending law: the benchmarked bounded radial chain
        (default, as :func:`~cablexsec.analysis.bending_law.export_bending_law`)
        or the solved radial section of the case.
    annulus : {"flooded", "sealed"}, optional
        Whether an external pressure acts on every solid (flooded) or on the
        outer sheath only (sealed); see the module notes.
    torque_step_Nm : float, optional
        Torque step for the torsional stiffness in each direction [N m].
    water_density_kg_m3 : float, optional
        Sea-water density for the weight in water.
    progress : callable, optional
        Called with the fraction done and a message (it may raise to cancel).
    core_friction : bool or CoreFrictionSpec, optional
        The core lay stage of the bending law (module notes): passed to
        :func:`~cablexsec.api.compute_mkappa`; ``True`` uses the default
        :class:`~cablexsec.mechanics.core_friction.CoreFrictionSpec`. The core
        slip curvatures are added as nodes of the curvature grid and reported
        per case. ``None`` or ``False`` (default) keeps the armour-only law,
        in which the laid cores bend about their own axes from the start.
        Needs one family of laid cores (a declared core lay).
    published_tolerance : float or None, optional
        Relative difference above which a computed value is flagged against
        the cable's published value (:func:`compare_published`, reported in
        ``published``); ``None`` leaves the comparison out. A published
        value stated at a tension that no case has, twist held, is compared
        with a further case at that tension, solved with the same settings.

    Returns
    -------
    SectionPropertiesResult
    """
    from cablexsec.api import compute_mkappa, compute_stiffness  # noqa: PLC0415

    if published_tolerance is not None and not (
        math.isfinite(published_tolerance) and published_tolerance > 0.0
    ):
        raise CableInputError("the tolerance against published values must be > 0")

    require_fully_specified(cable, "compute_section_properties")
    cases = tuple(load_cases)
    if not cases:
        raise CableInputError("at least one load case is required")
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        raise CableInputError("load case names must be unique")
    if bending_contact not in ("bounded", "solved"):
        raise CableInputError("bending contact must be 'bounded' or 'solved'")
    if annulus not in ("flooded", "sealed"):
        raise CableInputError("annulus must be 'flooded' or 'sealed'")
    if not (math.isfinite(kappa_max_1_m) and kappa_max_1_m > 0.0):
        raise CableInputError("the largest curvature must be finite and positive")
    if n_points < 8:
        raise CableInputError("the backbone needs 8 points or more")
    if not (math.isfinite(torque_step_Nm) and torque_step_Nm > 0.0):
        raise CableInputError("the torque step must be finite and positive")
    if mu is not None and not (math.isfinite(mu) and mu >= 0.0):
        raise CableInputError("the friction coefficient must be finite, not negative")
    armour_layers = structural_armour_layers(cable)
    if not armour_layers:
        raise CableSchemaError("the cable has no structural armour layer")
    core_spec = resolve_core_friction(core_friction)

    def report(fraction: float, message: str) -> None:
        if progress is not None:
            progress(fraction, message)

    stiffness = compute_stiffness(cable)
    plane = bending_stiffness_by_layer(cable)
    compliant = bending_stiffness_by_layer(cable, stiffness.xi_per_layer)
    shares = tuple(
        BendingShare(
            name,
            plane[name].plane_section_Nm2,
            compliant[name].shear_compliant_Nm2,
            plane[name].individual_Nm2,
        )
        for name in plane
    )
    mass = compute_mass_properties(cable, water_density_kg_m3=water_density_kg_m3)

    # Contact state and stick-slip data of every case first, so that one
    # curvature grid carries every slip curvature as a node.
    prepared: list[dict[str, Any]] = []
    total = len(cases)
    for index, case in enumerate(cases):
        report(0.05 + 0.6 * index / total, f"Solving the section: {case.name}")
        entry: dict[str, Any] = {"case": case, "notes": []}
        solved: _Solved | None = None
        reason = ""
        try:
            solved = _solve_case(cable, case, annulus, torque_step_Nm)
        except CableSchemaError as exc:
            if bending_contact == "solved":
                raise
            reason = str(exc)
        entry["solved"], entry["reason"] = solved, reason
        if bending_contact == "solved":
            assert solved is not None
            mkappa_args: dict[str, Any] = {"radial_state": solved.state}
            specs = cable_to_mkappa_specs(
                cable,
                solved.state.tension_N,
                mu=mu,
                contact_model="solved",
                radial_state=solved.state,
            )
        else:
            wall = _bounded_wall_tension(cable, case, annulus)
            state = _bounded_state(cable, case, annulus, wall)
            if state is None:
                mkappa_args = {"tension_N": case.tension_N, "contact_model": "bounded"}
                specs = cable_to_mkappa_specs(
                    cable, case.tension_N, mu=mu, contact_model="bounded"
                )
            else:
                mkappa_args = {"section_state": state}
                specs = cable_to_mkappa_specs(
                    cable,
                    state.tension_N,
                    mu=mu,
                    contact_model="bounded",
                    section_state=state,
                )
            if (
                case.external_pressure_Pa is not None
                and annulus == "flooded"
                and case.external_pressure_Pa > 0.0
            ):
                entry["notes"].append(
                    "flooded annulus in the bounded chain: the external pressure "
                    "gives no clamping of the armour"
                )
            if case.twist == "free":
                entry["notes"].append(
                    "the bounded contact chain has no twist freedom: its armour "
                    "contact is that of the twist-held state"
                )
        entry["mkappa_args"] = mkappa_args
        entry["armour"] = tuple(
            ArmourSlip(
                layer.name,
                spec.contact_pressure_Pa,
                spec.mu,
                spec.xi,
                slip_onset_curvature(spec),
                full_slip_curvature(spec),
                spec.EI_stick,
            )
            for layer, spec in zip(armour_layers, specs, strict=True)
        )
        entry["cores"] = (
            None if core_spec is None else _core_slip(cable, core_spec, mkappa_args)
        )
        prepared.append(entry)
    nodes = [
        value
        for entry in prepared
        for slip in (*entry["armour"], *[c for c in (entry["cores"],) if c])
        for value in (slip.onset_curvature_1_m, slip.full_slip_curvature_1_m)
    ]
    kappa = _curvature_grid(kappa_max_1_m, n_points, nodes)

    results: list[SectionCaseResult] = []
    for index, entry in enumerate(prepared):
        case = entry["case"]
        report(0.65 + 0.33 * index / total, f"Tracing the backbone: {case.name}")
        law = compute_mkappa(
            cable, kappa, mu=mu, core_friction=core_spec, **entry["mkappa_args"]
        )
        moment = np.asarray(law.moment_Nm, dtype=np.float64).copy()
        moment[0] = 0.0
        solved = entry["solved"]
        notes = list(entry["notes"])
        if solved is not None:
            route = "solved"
            wall = solved.state.tension_N
            mechanical = solved.state.mechanical_axial_force_N
            force = float(wall if mechanical is None else mechanical)
            axial = solved.axial_stiffness
            values = (
                solved.secant,
                solved.torque_change,
                solved.twist_change,
                solved.gj_positive,
                solved.gj_negative,
                solved.gj_tangent,
            )
            share, basis = solved.load_share, _SOLVED_BASIS
            interface, radii = solved.interface_pressure, solved.interface_radius
            notes += list(solved.state.notes)
            secant_note = _secant_note(case, solved)
            if secant_note:
                notes.append(secant_note)
        else:
            route = "fixed_radius"
            wall = _bounded_wall_tension(cable, case, annulus)
            force = float(wall)
            held = case.twist == "held"
            axial = stiffness.ea_fixed_N if held else stiffness.ea_free_N
            values = (
                math.nan,
                math.nan,
                math.nan,
                stiffness.gj_Nm2,
                stiffness.gj_Nm2,
                stiffness.gj_Nm2,
            )
            share = _fixed_radius_share(cable, case, annulus)
            basis = _FIXED_RADIUS_BASIS
            interface, radii = {}, {}
        results.append(
            SectionCaseResult(
                case=case,
                route=route,
                route_reason=entry["reason"],
                wall_tension_N=float(wall),
                mechanical_force_N=force,
                axial_stiffness_N=float(axial),
                axial_stiffness_secant_N=values[0],
                torque_change_Nm=values[1],
                twist_change_rad_per_m=values[2],
                torsional_stiffness_positive_Nm2=values[3],
                torsional_stiffness_negative_Nm2=values[4],
                torsional_stiffness_tangent_Nm2=values[5],
                torque_step_Nm=float(torque_step_Nm),
                load_share_N=share,
                load_share_basis=basis,
                interface_pressure_Pa=interface,
                interface_radius_m=radii,
                armour=entry["armour"],
                bending_contact=bending_contact,
                kappa_1_m=kappa.copy(),
                moment_Nm=moment,
                ei_stick_Nm2=float(law.EI_max_Nm2),
                ei_slip_Nm2=float(law.EI_min_Nm2),
                notes=tuple(dict.fromkeys(notes)),
                cores=entry["cores"],
                wall_tension_basis=_wall_tension_basis(
                    case, annulus, route, float(wall), mass.solid_area_m2
                ),
            )
        )
    report(1.0, "")
    core_law = (
        "the laid cores bend about their own axes from the start (core lay stage off)"
        if core_spec is None
        else "the laid cores stick and slip on the core lay (core contact "
        "loads of the same contact model, held fixed in bending; the layers "
        "of a core move together)"
    )
    result = SectionPropertiesResult(
        cable_name=cable.cable.name,
        cases=tuple(results),
        ei_full_stick_Nm2=stiffness.ei_plane_section_Nm2,
        ei_no_slip_shear_compliant_Nm2=stiffness.ei_max_Nm2,
        ei_full_slip_Nm2=stiffness.ei_min_Nm2,
        ea_fixed_radius_held_N=stiffness.ea_fixed_N,
        ea_fixed_radius_free_N=stiffness.ea_free_N,
        gj_lay_Nm2=stiffness.gj_Nm2,
        xi_per_layer=dict(stiffness.xi_per_layer),
        bending_share=shares,
        mass=mass,
        min_bend_radius_dynamic_m=cable.cable.MBR_dynamic_m,
        min_bend_radius_installation_m=cable.cable.MBR_installation_m,
        settings={
            "kappa_max_1_m": float(kappa_max_1_m),
            "n_points": int(n_points),
            "mu": mu,
            "bending_contact": bending_contact,
            "annulus": annulus,
            "torque_step_Nm": float(torque_step_Nm),
            "water_density_kg_m3": float(water_density_kg_m3),
            "loose_layers": _LOOSE_LAYERS,
            "core_lay_stage": None if core_spec is None else _spec_settings(core_spec),
        },
        provenance=Provenance(
            method=(
                "per load case: coupled radial-axial-torsional section with "
                "unilateral contact for the axial and torsional stiffness, load "
                "share and interface pressure (fixed-radius assembly where that "
                "model does not accept the cable); Papailiou slip-front bending "
                f"law with the {bending_contact} contact model for the backbone "
                "and the slip curvatures"
                + ("" if core_spec is None else ", with the core lay stage")
                + "; fixed-radius bending bounds"
            ),
            citations=(
                "[WitzTan1992a]",
                "[WitzTan1992b]",
                "[Papailiou1997]",
                "[Yun2020]",
                "[FeretBournazel1987]",
                *(() if core_spec is None else ("[Skeie2012] Eqs. 2.31-2.35",)),
            ),
            gates=(
                "A7 EA-Schur",
                "A8/A8b/A8c bending bounds",
                "TQ1-TQ6 solved-section torque checks",
                "slip curvatures checked against the backbone of compute_mkappa",
            ),
            limitations=(
                "torsional stiffness by direction is the secant over the torque "
                "step; it differs between directions only where a contact opens "
                "or closes within the step",
                "the solved section has frictionless contact; stick-slip enters "
                "the bending law only",
                "the bending bounds are fixed-radius assemblies at the reference "
                "moduli, independent of the load case",
                f"the backbone is planar; {core_law}",
                "slip onset is the neutral-axis limit of the Papailiou law, a "
                "lower bound for a finite wire population",
            ),
        ),
    )
    if published_tolerance is None:
        return result

    def case_at(tension_N: float) -> SectionCaseResult | None:
        # A published value stated at a tension no case has: one more case,
        # twist held, with the same settings (its backbone is not reported).
        reference = compute_section_properties(
            cable,
            (SectionLoadCase("published condition", tension_N, "held"),),
            kappa_max_1_m=kappa_max_1_m,
            n_points=8,
            mu=mu,
            bending_contact=bending_contact,
            annulus=annulus,
            torque_step_Nm=torque_step_Nm,
            water_density_kg_m3=water_density_kg_m3,
            core_friction=core_spec,
            published_tolerance=None,
        )
        return reference.cases[0]

    return replace(
        result,
        settings={**result.settings, "published_tolerance": published_tolerance},
        published=compare_published(
            cable, result, tolerance=published_tolerance, case_at=case_at
        ),
    )


# --------------------------------------------------------------------------- #
# Load sharing against tension                                                 #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LoadSharingResult:
    """Axial force of every layer against the tension, at one end condition.

    ``force_N`` holds, per layer (inner to outer), the change of its axial
    force from the zero-tension state of the same temperature and pressure,
    at each effective tension of ``tension_N``; the changes sum to the
    tension. ``force_at_zero_N`` is each layer's true axial force at zero
    tension: self-balancing forces locked in by the manufacturing clamping,
    the temperature or, in a sealed annulus, the pressure (zero for the
    fixed-radius route without those loads).
    """

    cable_name: str
    tension_N: FloatArray
    """Effective tensions, rising from zero [N]."""
    wall_tension_N: FloatArray
    """Tension given to the section at each value, after the pressure conversion."""
    twist: TwistEnd
    temperature_C: float | None
    external_pressure_Pa: float | None
    annulus: str
    route: str
    """``"solved"`` (coupled radial section) or ``"fixed_radius"``."""
    route_reason: str
    basis: str
    force_N: Mapping[str, FloatArray]
    force_at_zero_N: Mapping[str, float]
    notes: tuple[str, ...] = ()
    provenance: Provenance | None = None

    @property
    def layers(self) -> tuple[str, ...]:
        """The layers, inner to outer."""
        return tuple(self.force_N)

    @property
    def total_N(self) -> FloatArray:
        """Sum of the layers' force changes at each tension [N]."""
        total = np.zeros_like(self.tension_N, dtype=np.float64)
        for name in self.layers:
            total = total + self.force_N[name]
        return total

    def share(self) -> dict[str, FloatArray]:
        """Each layer's force change as a fraction of the tension (``NaN`` at zero)."""
        tension = self.tension_N
        with np.errstate(divide="ignore", invalid="ignore"):
            return {
                name: np.where(tension > 0.0, force / tension, np.nan)
                for name, force in self.force_N.items()
            }


def compute_load_sharing(
    cable: CableData,
    tension_max_N: float,
    n_tensions: int = 11,
    *,
    twist: TwistEnd = "held",
    temperature_C: float | None = None,
    external_pressure_Pa: float | None = None,
    annulus: str = "flooded",
) -> LoadSharingResult:
    """Axial force and share of every layer at tensions from zero to a maximum.

    Parameters
    ----------
    cable : CableData
        A fully specified cable.
    tension_max_N : float
        Largest effective tension [N], positive.
    n_tensions : int, optional
        Tensions, evenly spaced from zero to ``tension_max_N`` (two or more).
    twist : {"held", "free"}, optional
        End condition for twist.
    temperature_C, external_pressure_Pa : float or None, optional
        Uniform temperature and external pressure, as in
        :class:`SectionLoadCase`.
    annulus : {"flooded", "sealed"}, optional
        How an external pressure acts; see the module notes.

    Returns
    -------
    LoadSharingResult
        On the solved route (the coupled radial section with unilateral
        contact, where it accepts the cable) every layer's force is recovered
        from its own strain and material law: the solved axial strain, twist
        and radial displacements give each annulus its axial stress, each
        helical layer its wire stress and each trefoil element its axial
        stress, and in a flooded section the uniform water pressure on the
        layer's area is included. The forces sum to the section's axial force,
        which equals the tension given to it to the solver's equilibrium
        residual. Elsewhere the fixed-radius route is used: equal axial strain
        of every layer with the twist held, labelled as such.
    """
    require_fully_specified(cable, "compute_load_sharing")
    if not (math.isfinite(tension_max_N) and tension_max_N > 0.0):
        raise CableInputError("the largest tension must be finite and positive")
    if isinstance(n_tensions, bool) or not isinstance(n_tensions, int):
        raise CableInputError("the number of tensions must be an integer")
    if n_tensions < 2:
        raise CableInputError("load sharing needs two tensions or more")
    if annulus not in ("flooded", "sealed"):
        raise CableInputError("annulus must be 'flooded' or 'sealed'")
    case = SectionLoadCase(
        "load sharing", float(tension_max_N), twist, temperature_C, external_pressure_Pa
    )
    tensions = np.linspace(0.0, float(tension_max_N), n_tensions)
    options = _radial_options(case, annulus)
    notes: list[str] = []
    reason = ""
    try:
        walls = _wall_tensions(cable, case, annulus, tensions.tolist(), options)
        states = solve_cable_radial_states(
            cable, walls, **options, **_twist_options(twist)
        )
    except CableSchemaError as exc:
        reason = str(exc)
        states = []
    if states:
        route, basis = "solved", _SOLVED_BASIS
        forces = [state.layer_axial_force() for state in states]
        wall = np.array([state.tension_N for state in states], dtype=np.float64)
        notes += list(states[0].notes)
    else:
        route, basis = "fixed_radius", _FIXED_RADIUS_BASIS
        forces = _fixed_radius_forces(cable, case, annulus, tensions.tolist())
        wall = np.array(
            [
                _bounded_wall_tension(cable, replace(case, tension_N=float(t)), annulus)
                for t in tensions
            ],
            dtype=np.float64,
        )
        if twist == "free":
            notes.append(
                "the fixed-radius share has no twist freedom: it is that of "
                "the twist-held state"
            )
    names = _layer_order(
        cable, [n for n in forces[-1] if any(f.get(n, 0.0) for f in forces)]
    )
    zero = forces[0]
    force = {
        name: np.array(
            [f.get(name, 0.0) - zero.get(name, 0.0) for f in forces], dtype=np.float64
        )
        for name in names
    }
    return LoadSharingResult(
        cable_name=cable.cable.name,
        tension_N=tensions,
        wall_tension_N=wall,
        twist=twist,
        temperature_C=temperature_C,
        external_pressure_Pa=external_pressure_Pa,
        annulus=annulus,
        route=route,
        route_reason=reason,
        basis=basis,
        force_N=force,
        force_at_zero_N={name: float(zero.get(name, 0.0)) for name in names},
        notes=tuple(dict.fromkeys(notes)),
        provenance=Provenance(
            method=(
                "each layer's axial force from the solved displacements and its "
                "material law in the coupled radial-axial-torsional section with "
                "unilateral contact; equal axial strain at fixed radius where "
                "that model does not accept the cable"
            ),
            citations=("[WitzTan1992a]", "[FeretBournazel1987]"),
            gates=("per-layer forces sum to the section's axial force",),
            limitations=(
                "frictionless contact: the share of the armour layers is that "
                "of a section without stick-slip",
                "loose core screens take the material of their softer "
                "structural neighbour, and carry the force of that material",
                "change from the zero-tension state: forces locked in by the "
                "manufacturing clamping or the temperature are reported apart",
            ),
        ),
    )
