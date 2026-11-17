# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Solved radial contact state of a cable definition.

Translates a :class:`~cablexsec.cables.schema.CableData` into the coupled
section of :mod:`cablexsec.mechanics.radial_solution` and solves one load
state. Concentric cables use exact annulus and helical-ring layers. Three-core
cables use the plane trefoil model for the cores, the interstitial filler and
the contiguous polymer layers that enclose them, and concentric layers
outside. Manufacturing pressure (``pretension_Pa``) and winding stress
(``winding_stress_Pa``) become residual wire stresses that are identified at
the reference state and then carried through every load case, so contact
can open or close; they are not superposed on the solution.

Load semantics
--------------
``tension_N`` is the true wall tension. With ``end_cap=True`` the closed-end
pressure thrust pi (p_i r_i² - p_e r_o²) is added (test-rig convention).
``external_pressure_Pa`` acts on the outermost surface only (a dry,
watertight outer sheath); by default it is the cable's documented
initial-state pressure, as in the existing contact paths. A flooded (wet)
section uses ``flooded_pressure_Pa``: every solid then carries the uniform
water pressure, contact reactions are mechanical (net of that pressure), and
the mechanical axial force is the true wall tension plus p times the solid
area. Layer temperatures give free thermal strain alpha (T - T_ref) and, for
registered materials, E(T).

Representation choices that are stated, not hidden
---------------------------------------------------
* Unlisted radial gaps inside a power core (semiconducting screens) are
  filled with the next outer layer's material; the result lists them.
* Loose (non-structural) core layers, such as copper tape and wire screens,
  carry no axial load in the fixed-radius stiffness. By default they are
  meshed with their declared material. ``loose_layers="neighbour"`` gives
  each the material of its softer structural neighbour instead, so that the
  solved axial and torsional rigidities follow that convention.
* A three-core ``cylindrical`` layer that overlaps the cores is the
  interstitial filler. ``filler="void"`` removes it (flooded or loose yarn
  interstices); ``"solid"`` meshes it in the outer interstices with
  unilateral contact; ``"auto"`` uses it when the cable declares one.
* Geometric overlaps below 0.5 % of the contact radius are treated as a snug
  fit (zero gap) and reported; larger overlaps are rejected.
* Stranded conductors whose metal area differs from the envelope, nested
  strands and corrugated shells are rejected, as in the concentric adapter.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cablexsec.cables.schema import (
    CableData,
    CylindricalLayer,
    CylindricalOffsetLayer,
    CylindricalShellLayer,
    HelicalLayer,
    is_serving,
)
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.materials import library
from cablexsec.mechanics.concentric_section import ElasticAnnulus, ElasticHelicalRing
from cablexsec.mechanics.core_lay import core_angle
from cablexsec.mechanics.radial_solution import (
    RadialSection,
    RadialSectionResult,
    capstan_residual_stress,
    flooded_free_strain,
)
from cablexsec.mechanics.trefoil_section import (
    SYMMETRY_FACTOR,
    ElasticFiller,
    TrefoilBundle,
)

__all__ = [
    "CableRadialState",
    "build_radial_section",
    "loose_layer_sources",
    "residual_wire_stress",
    "solve_cable_radial_state",
    "solve_cable_radial_states",
]

_OVERLAP_WORDS = {
    "core_ring": "the cores overlap the ring around them",
    "core_core": "neighbouring cores overlap",
}
"""The small manufacturing overlaps the trefoil model closes, in words."""


def _modulus_key(material: str) -> str:
    # Same convention as the concentric adapter: the stranded-copper entry's
    # axial key holds the copper wire modulus, used here as isotropic.
    return library.axial_modulus_key(material)


def _annulus_bounds(layer: object) -> tuple[float, float]:
    if isinstance(layer, CylindricalShellLayer):
        if layer.f_corrugation != 1:
            raise CableSchemaError(
                f"{layer.name}: corrugated shells require a radial constitutive model"
            )
        if layer.r_mid_m is None or layer.t_m is None:
            raise CableSchemaError(f"{layer.name}: incomplete shell geometry")
        return layer.r_mid_m - layer.t_m / 2, layer.r_mid_m + layer.t_m / 2
    a = getattr(layer, "r_inner_m", None)
    b = getattr(layer, "r_outer_m", None)
    if a is None or b is None:
        raise CableSchemaError(f"{getattr(layer, 'name', '?')}: incomplete geometry")
    return float(a), float(b)


def _check_core(layer: CylindricalOffsetLayer) -> None:
    if layer.strand_layers or layer.strand_lay_angle_deg not in (None, 0.0):
        raise CableSchemaError(
            f"{layer.name}: a resolved strand bundle needs a strand radial law"
        )
    if layer.conductor_area_mm2 is not None and not math.isclose(
        layer.conductor_area_mm2 * 1e-6,
        math.pi * ((layer.r_outer_m or 0) ** 2 - (layer.r_inner_m or 0) ** 2),
        rel_tol=1e-12,
    ):
        raise CableSchemaError(
            f"{layer.name}: conductor metal area differs from its envelope; a "
            "radial bundle law is required instead of a solid annulus"
        )


@dataclass(frozen=True)
class _Material:
    modulus: float
    poisson: float
    free_strain: float
    # In-plane free strain of a frozen extrusion shrinkage (-hoop_strain).
    transverse_strain: float = 0.0


def _layer_state(
    cable: CableData,
    *,
    temperature_C: float | Mapping[str, float] | None,
    T_ref_C: float,
    temperature_dependent_modulus: bool,
    youngs_modulus_Pa: Mapping[str, float] | None,
    flooded_pressure_Pa: float,
    material_source: Mapping[str, str] | None = None,
    include_shrinkage: bool = True,
) -> dict[str, _Material]:
    names = {layer.name for layer in cable.layers}
    sources = dict(material_source or {})
    for values in (
        youngs_modulus_Pa,
        temperature_C if isinstance(temperature_C, Mapping) else None,
    ):
        if values is not None and set(values) - names:
            raise CableSchemaError("radial-state input names an unknown layer")
    out: dict[str, _Material] = {}
    for layer in cable.layers:
        if isinstance(temperature_C, Mapping):
            t = float(temperature_C.get(layer.name, T_ref_C))
        else:
            t = float(T_ref_C if temperature_C is None else temperature_C)
        if not math.isfinite(t):
            raise CableSchemaError("layer temperatures must be finite")
        material = (
            cable.get_layer(sources[layer.name]).material
            if layer.name in sources
            else layer.material
        )
        modulus = library.get_property(material, _modulus_key(material))
        if (
            temperature_dependent_modulus
            and temperature_C is not None
            and library.has_temperature_modulus_model(material)
        ):
            modulus = library.youngs_modulus_at_temperature(material, t)
        if youngs_modulus_Pa is not None and layer.name in youngs_modulus_Pa:
            modulus = float(youngs_modulus_Pa[layer.name])
        if not (math.isfinite(modulus) and modulus > 0):
            raise CableSchemaError("layer moduli must be finite and positive")
        nu = library.get_property(material, "nu")
        free = 0.0
        if t != T_ref_C:
            free += library.get_property(material, "alpha_T_K") * (t - T_ref_C)
        if flooded_pressure_Pa:
            free += flooded_free_strain(modulus, nu, flooded_pressure_Pa)
        shrink = getattr(layer, "extrusion_shrinkage", None)
        transverse = -float(shrink.hoop_strain) if include_shrinkage and shrink else 0.0
        out[layer.name] = _Material(modulus, nu, free, transverse)
    return out


_RADIUS_MATCH_M = 1e-9


def loose_layer_sources(cable: CableData) -> dict[str, str]:
    """Softer structural neighbour of every loose (non-structural) core layer.

    A loose core layer is a non-structural ``cylindrical_offset`` layer (a
    tape or wire screen), to which the fixed-radius stiffness gives no axial
    load. Its neighbours are the structural layers of the same core (or of
    the concentric stack, for a centred core) whose outer radius meets its
    inner radius or whose inner radius meets its outer radius. The one with
    the lower reference modulus is returned, the outer one on a tie. A loose
    layer without a structural neighbour is omitted.
    """

    def modulus(layer: object) -> float:
        material = str(getattr(layer, "material", ""))
        return library.get_property(material, _modulus_key(material))

    out: dict[str, str] = {}
    for loose in cable.layers:
        if not isinstance(loose, CylindricalOffsetLayer) or loose.structural:
            continue
        a, b = float(loose.r_inner_m or 0.0), float(loose.r_outer_m or 0.0)
        offset = float(loose.r_offset_m or 0.0)
        candidates: list[tuple[float, int, str]] = []
        for other in cable.layers:
            if other is loose or not other.structural:
                continue
            if isinstance(other, CylindricalOffsetLayer):
                if float(other.r_offset_m or 0.0) != offset:
                    continue
            elif not (
                isinstance(other, CylindricalLayer | CylindricalShellLayer)
                and offset == 0.0
            ):
                continue
            lo, hi = _annulus_bounds(other)
            if abs(hi - a) <= _RADIUS_MATCH_M:
                candidates.append((modulus(other), 1, other.name))
            elif abs(lo - b) <= _RADIUS_MATCH_M:
                candidates.append((modulus(other), 0, other.name))
        if candidates:
            out[loose.name] = min(candidates)[2]
    return out


def residual_wire_stress(cable: CableData) -> dict[str, float]:
    """Residual wire stress per helical layer from the manufacturing inputs.

    pretension_Pa (a capstan pressure jump P0) is converted by the inverse of
    Féret-Bournazel Eq. 5; winding_stress_Pa is a wire stress already. Both
    are added. Layers with neither input are omitted.
    """
    out: dict[str, float] = {}
    for layer in cable.helical_layers:
        stress = float(layer.winding_stress_Pa or 0.0)
        if layer.pretension_Pa:
            assert layer.n_wires and layer.d_wire_m and layer.r_helix_m
            assert layer.lay_angle_rad is not None
            stress += capstan_residual_stress(
                layer.pretension_Pa,
                wire_count=layer.n_wires,
                wire_diameter_m=layer.d_wire_m,
                helix_radius_m=layer.r_helix_m,
                lay_angle_rad=layer.lay_angle_rad,
                wire_area_m2=None
                if layer.wire_shape == "round"
                else layer.wire_section.area,
            )
        if stress:
            out[layer.name] = stress
    return out


def build_radial_section(
    cable: CableData,
    *,
    temperature_C: float | Mapping[str, float] | None = None,
    T_ref_C: float = 20.0,
    temperature_dependent_modulus: bool = True,
    youngs_modulus_Pa: Mapping[str, float] | None = None,
    flooded_pressure_Pa: float = 0.0,
    filler: str = "auto",
    angular_cells: int = 48,
    loose_layers: str = "declared",
    include_shrinkage: bool = True,
) -> tuple[RadialSection, tuple[str, ...]]:
    """Assemble the coupled section of a cable; returns it with notes.

    ``include_shrinkage`` carries the declared frozen extrusion shrinkage of
    concentric layers as their in-plane free strain.
    """
    if cable.missing_fields():
        raise CableSchemaError("radial solution requires complete geometry")
    if filler not in {"auto", "solid", "void"}:
        raise CableSchemaError("filler must be 'auto', 'solid' or 'void'")
    if loose_layers not in {"declared", "neighbour"}:
        raise CableSchemaError("loose_layers must be 'declared' or 'neighbour'")
    sources = loose_layer_sources(cable) if loose_layers == "neighbour" else {}
    if not math.isfinite(flooded_pressure_Pa) or flooded_pressure_Pa < 0:
        raise CableSchemaError("flooded pressure must be finite and non-negative")
    mats = _layer_state(
        cable,
        temperature_C=temperature_C,
        T_ref_C=T_ref_C,
        temperature_dependent_modulus=temperature_dependent_modulus,
        youngs_modulus_Pa=youngs_modulus_Pa,
        flooded_pressure_Pa=flooded_pressure_Pa,
        material_source=sources,
        include_shrinkage=include_shrinkage,
    )
    notes: list[str] = [
        f"loose layer {name!r} meshed with the material of {source!r}"
        for name, source in sources.items()
    ]
    cores = [
        x
        for x in cable.layers
        if isinstance(x, CylindricalOffsetLayer)
        and not (x.n_cores == 1 and (x.r_offset_m or 0.0) == 0.0)
    ]
    if any(x.n_cores != 3 for x in cores):
        raise CableSchemaError("only single concentric or three-core trefoil cores")
    bundle = None
    concentric_source = [x for x in cable.layers if x not in cores]
    if cores:
        offsets = {x.r_offset_m for x in cores}
        if len(offsets) != 1:
            raise CableSchemaError("all trefoil core layers need one pitch radius")
        rc = float(offsets.pop() or 0.0)
        for core_def in cores:
            _check_core(core_def)
        cores.sort(key=lambda x: x.r_inner_m or 0.0)
        core_layers = tuple(
            ElasticAnnulus(
                x.name,
                float(x.r_inner_m or 0.0),
                float(x.r_outer_m or 0.0),
                mats[x.name].modulus,
                mats[x.name].poisson,
                mats[x.name].free_strain,
            )
            for x in cores
        )
        envelope = rc + core_layers[-1].outer_radius_m
        tol = 5e-3 * envelope
        fillers = [
            x
            for x in concentric_source
            if isinstance(x, CylindricalLayer)
            and x.r_inner_m is not None
            and x.r_inner_m < envelope - tol
        ]
        if len(fillers) > 1:
            raise CableSchemaError("more than one interstitial filler layer")
        if any(is_serving(x) for x in fillers):
            raise CableSchemaError(
                "a serving is wound on a layer; it cannot fill the core interstices"
            )
        filler_body = None
        if fillers:
            concentric_source.remove(fillers[0])
            f = fillers[0]
            if mats[f.name].transverse_strain:
                raise CableSchemaError(
                    f"{f.name}: an interstitial filler cannot declare an "
                    "extrusion shrinkage"
                )
            if filler in {"auto", "solid"}:
                filler_body = ElasticFiller(
                    f.name,
                    mats[f.name].modulus,
                    mats[f.name].poisson,
                    mats[f.name].free_strain,
                )
                notes.append(
                    f"filler {f.name!r} meshed in the three outer interstices; "
                    "central interstice void"
                )
            else:
                notes.append(f"filler {f.name!r} omitted (void interstices)")
        elif filler == "solid":
            raise CableSchemaError("filler='solid' but the cable declares no filler")
        ordered = sorted(
            concentric_source,
            key=lambda x: (
                (x.r_helix_m or 0.0)
                if isinstance(x, HelicalLayer)
                else _annulus_bounds(x)[0]
            ),
        )
        ring: list[ElasticAnnulus] = []
        for item in ordered:
            if isinstance(item, HelicalLayer):
                break
            a, b = _annulus_bounds(item)
            if ring and abs(a - ring[-1].outer_radius_m) > 1e-12:
                break
            if not ring and a < rc:
                raise CableSchemaError(
                    f"{item.name}: annulus intersects the core centres"
                )
            ring.append(
                ElasticAnnulus(
                    item.name,
                    a,
                    b,
                    mats[item.name].modulus,
                    mats[item.name].poisson,
                    mats[item.name].free_strain,
                    free_transverse_strain=mats[item.name].transverse_strain,
                    carries_axial_load=not is_serving(item),
                )
            )
        if not ring:
            raise CableSchemaError("a trefoil bundle needs an enclosing polymer ring")
        ring_names = {r.name for r in ring}
        concentric_source = [item for item in ordered if item.name not in ring_names]
        beta = core_angle(cable, cores[0])
        bundle = TrefoilBundle(
            core_layers,
            rc,
            tuple(ring),
            core_lay_angle_rad=beta,
            filler=filler_body,
            angular_cells=angular_cells,
        )
        notes.append(
            "trefoil plane model: cores, "
            + ("filler, " if filler_body is not None else "")
            + "ring layers "
            + ", ".join(r.name for r in ring)
            + " bonded to each other; layers outside apply uniform pressure"
        )
    layers: list[ElasticAnnulus | ElasticHelicalRing] = []
    for item in concentric_source:
        x = item
        m = mats[x.name]
        if isinstance(x, HelicalLayer):
            assert x.r_helix_m and x.d_wire_m and x.n_wires
            assert x.lay_angle_rad is not None
            angle = abs(x.lay_angle_rad) * (-1 if x.lay_direction == "left_hand" else 1)
            layers.append(
                ElasticHelicalRing(
                    x.name,
                    x.r_helix_m,
                    x.d_wire_m,
                    x.n_wires,
                    angle,
                    m.modulus,
                    m.free_strain,
                    radial_thickness_m=None
                    if x.wire_shape == "round"
                    else x.wire_radial_thickness_m,
                    wire_area_m2=None
                    if x.wire_shape == "round"
                    else x.wire_section.area,
                )
            )
        elif isinstance(x, CylindricalOffsetLayer):
            _check_core(x)
            a, b = _annulus_bounds(x)
            layers.append(
                ElasticAnnulus(x.name, a, b, m.modulus, m.poisson, m.free_strain)
            )
        elif isinstance(x, (CylindricalLayer, CylindricalShellLayer)):
            a, b = _annulus_bounds(x)
            layers.append(
                ElasticAnnulus(
                    x.name,
                    a,
                    b,
                    m.modulus,
                    m.poisson,
                    m.free_strain,
                    free_transverse_strain=m.transverse_strain,
                    carries_axial_load=not is_serving(x),
                )
            )
        else:
            raise CableSchemaError(
                f"{x.name}: unsupported layer type for radial solution"
            )
    try:
        section = RadialSection(layers, bundle=bundle)
    except CableInputError as exc:
        raise CableSchemaError(str(exc)) from exc
    if section.model is not None:
        for label, value in section.model.adjusted_overlaps_m.items():
            which = _OVERLAP_WORDS.get(label, label.replace("_", " ") + " overlap")
            notes.append(f"{which} by {value * 1e3:.3g} mm, treated as a snug fit")
        for text in section.model.filled_core_gaps:
            notes.append(f"core gap {text}")
    return section, tuple(notes)


def _solid_area(section: RadialSection) -> float:
    area = 0.0
    for layer in section.layers:
        if isinstance(layer, ElasticHelicalRing):
            area += layer.total_area_m2 / math.cos(layer.signed_lay_angle_rad)
        else:
            area += math.pi * (layer.outer_radius_m**2 - layer.inner_radius_m**2)
    model = section.model
    if model is not None:
        p = model.nodes[model.element_nodes[:, :3]]
        tri = 0.5 * np.abs(
            (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
            - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
        )
        area += SYMMETRY_FACTOR * float(np.sum(tri))
    return area


@dataclass(frozen=True)
class CableRadialState:
    """Solved radial state of one cable at one load state."""

    cable: CableData
    result: RadialSectionResult
    tension_N: float
    mechanical_axial_force_N: float | None
    internal_pressure_Pa: float
    external_pressure_Pa: float
    flooded_pressure_Pa: float
    end_cap: bool
    initial_wire_strain: Mapping[str, float]
    residual_wire_stress_Pa: Mapping[str, float]
    notes: tuple[str, ...] = field(default=())
    pressure_barrier: str | None = None
    layer_solid_area_m2: Mapping[str, float] = field(default_factory=dict)
    """Area of each layer's material in the cable cross-section [m²]."""

    def layer_axial_force(self) -> dict[str, float]:
        """True axial force of each layer [N], the three cores of a layer together.

        The mechanical force of the solved section
        (:attr:`RadialSectionResult.layer_axial_force_N`) less, in a flooded
        section, the uniform water pressure on the layer's area in the cable
        cross-section. The forces sum to the true wall tension (plus the
        closed-end thrust with ``end_cap``), as the section's axial force is
        the mechanical tension of the module notes.
        """
        pressure = self.flooded_pressure_Pa
        return {
            name: float(force) - pressure * self.layer_solid_area_m2.get(name, 0.0)
            for name, force in self.result.layer_axial_force_N.items()
        }

    def require_cable(self, cable: CableData) -> None:
        if cable != self.cable:
            raise CableSchemaError("radial state belongs to a different cable")

    def armour_contact_pressure(self) -> dict[str, float]:
        """P1 + P2 [Pa] for every helical layer of the solved state."""
        return {
            name: inner + outer
            for name, (inner, outer) in self.result.helical_face_pressure_Pa.items()
        }

    def normal_line_load_per_wire(self) -> dict[str, float]:
        """Both-face normal line load per wire [N/m] (historical F_n field)."""
        out = {}
        counts = {x.name: x.n_wires for x in self.cable.helical_layers}
        for name, (inner, outer) in self.result.helical_face_line_force_N_per_m.items():
            out[name] = (inner + outer) / float(counts[name] or 1)
        return out

    def interface_line_force(self, inner: str, outer: str) -> float:
        """Shared reaction [N/m] between two adjacent concentric layers."""
        try:
            index = self.result.interface_names.index((inner, outer))
        except ValueError as exc:
            raise CableSchemaError(
                f"{inner!r} and {outer!r} are not adjacent in the solved section"
            ) from exc
        return float(self.result.interface_line_force_N_per_m[index])


_SECTION_CACHE: dict[tuple[object, ...], tuple[RadialSection, tuple[str, ...]]] = {}


def _cached_section(
    cable: CableData, key: tuple[object, ...], **kwargs: Any
) -> tuple[RadialSection, tuple[str, ...]]:
    try:
        cache_key = (hash(cable), cable.model_dump_json(), key)
    except TypeError:
        return build_radial_section(cable, **kwargs)
    if cache_key not in _SECTION_CACHE:
        if len(_SECTION_CACHE) > 16:
            _SECTION_CACHE.clear()
        _SECTION_CACHE[cache_key] = build_radial_section(cable, **kwargs)
    return _SECTION_CACHE[cache_key]


def _cached_identification(
    section: RadialSection, residual: Mapping[str, float]
) -> dict[str, float]:
    if not residual:
        return {}
    cache = section.__dict__.setdefault("_identified", {})
    key = tuple(sorted(residual.items()))
    if key not in cache:
        cache[key] = section.identify_initial_wire_strain(residual)
    return dict(cache[key])


def solve_cable_radial_state(
    cable: CableData,
    tension_N: float = 0.0,
    *,
    prescribed_axial_strain: float | None = None,
    torque_Nm: float = 0.0,
    prescribed_twist_per_m: float | None = 0.0,
    internal_pressure_Pa: float = 0.0,
    external_pressure_Pa: float | None = None,
    flooded_pressure_Pa: float = 0.0,
    end_cap: bool = False,
    temperature_C: float | Mapping[str, float] | None = None,
    T_ref_C: float = 20.0,
    temperature_dependent_modulus: bool = True,
    youngs_modulus_Pa: Mapping[str, float] | None = None,
    filler: str = "auto",
    angular_cells: int = 48,
    include_manufacturing: bool = True,
    pressure_barrier: str | None = None,
    loose_layers: str = "declared",
) -> CableRadialState:
    """Solve the cable's radial contact state; see the module notes.

    Twist is fixed (zero) by default, as in the concentric contact path; pass
    prescribed_twist_per_m=None to prescribe torque. Pass
    prescribed_axial_strain to fix the ends axially instead of tension.

    ``pressure_barrier`` (opt-in) names the innermost fluid-tight layer of a
    concentric section, for example the internal pressure sheath of a
    flexible pipe whose interlocked carcass is not fluid-tight. The internal
    pressure then acts on that layer's bore, the layers inside it carry the
    fluid pressure on both faces (radially; their axial pressure term is not
    applied) and the closed-end thrust uses the barrier bore radius. ``None``
    (default) keeps the bore of the innermost layer, unchanged.
    """
    return solve_cable_radial_states(
        cable,
        (tension_N,),
        prescribed_axial_strain=prescribed_axial_strain,
        torque_Nm=torque_Nm,
        prescribed_twist_per_m=prescribed_twist_per_m,
        internal_pressure_Pa=internal_pressure_Pa,
        external_pressure_Pa=external_pressure_Pa,
        flooded_pressure_Pa=flooded_pressure_Pa,
        end_cap=end_cap,
        temperature_C=temperature_C,
        T_ref_C=T_ref_C,
        temperature_dependent_modulus=temperature_dependent_modulus,
        youngs_modulus_Pa=youngs_modulus_Pa,
        filler=filler,
        angular_cells=angular_cells,
        include_manufacturing=include_manufacturing,
        pressure_barrier=pressure_barrier,
        loose_layers=loose_layers,
    )[0]


def solve_cable_radial_states(
    cable: CableData,
    tensions_N: Sequence[float] | np.ndarray,
    *,
    prescribed_axial_strain: float | None = None,
    torque_Nm: float = 0.0,
    prescribed_twist_per_m: float | None = 0.0,
    internal_pressure_Pa: float = 0.0,
    external_pressure_Pa: float | None = None,
    flooded_pressure_Pa: float = 0.0,
    end_cap: bool = False,
    temperature_C: float | Mapping[str, float] | None = None,
    T_ref_C: float = 20.0,
    temperature_dependent_modulus: bool = True,
    youngs_modulus_Pa: Mapping[str, float] | None = None,
    filler: str = "auto",
    angular_cells: int = 48,
    include_manufacturing: bool = True,
    pressure_barrier: str | None = None,
    loose_layers: str = "declared",
) -> list[CableRadialState]:
    """Solved radial states at several tensions with one section assembly.

    Each state equals :func:`solve_cable_radial_state` at that tension bit
    for bit; the section (and, with temperatures or moduli, its material
    state) is assembled once. Used for the slow-time thermal blocks, where
    one temperature state is solved over a tension grid.
    """
    tensions = [float(value) for value in tensions_N]
    if not tensions:
        raise CableSchemaError("at least one tension is required")
    p_ext = (
        cable.initial_state.external_pressure_Pa
        if external_pressure_Pa is None
        else float(external_pressure_Pa)
    )
    values = [*tensions, torque_Nm, internal_pressure_Pa, p_ext, flooded_pressure_Pa]
    if (
        not np.isfinite(values).all()
        or min(internal_pressure_Pa, p_ext, flooded_pressure_Pa) < 0
    ):
        raise CableSchemaError("radial loads must be finite and pressures non-negative")
    if flooded_pressure_Pa and (p_ext or internal_pressure_Pa):
        raise CableSchemaError(
            "a flooded section has no separate dry surface pressures; "
            "use flooded_pressure_Pa alone"
        )
    residual = residual_wire_stress(cable) if include_manufacturing else {}
    reference, notes = _cached_section(
        cable,
        (
            "reference",
            filler,
            angular_cells,
            T_ref_C,
            loose_layers,
            include_manufacturing,
        ),
        filler=filler,
        angular_cells=angular_cells,
        T_ref_C=T_ref_C,
        loose_layers=loose_layers,
        include_shrinkage=include_manufacturing,
    )
    initial = _cached_identification(reference, residual)
    loaded_inputs = (
        temperature_C is not None
        or youngs_modulus_Pa is not None
        or flooded_pressure_Pa
    )
    if loaded_inputs:
        section, notes = build_radial_section(
            cable,
            temperature_C=temperature_C,
            T_ref_C=T_ref_C,
            temperature_dependent_modulus=temperature_dependent_modulus,
            youngs_modulus_Pa=youngs_modulus_Pa,
            flooded_pressure_Pa=flooded_pressure_Pa,
            filler=filler,
            angular_cells=angular_cells,
            loose_layers=loose_layers,
            include_shrinkage=include_manufacturing,
        )
    else:
        section = reference
    barrier = 0
    if pressure_barrier is not None:
        names = [layer.name for layer in section.layers]
        if section.model is not None or pressure_barrier not in names:
            raise CableSchemaError(
                f"pressure_barrier {pressure_barrier!r} must name a layer of a "
                "concentric section"
            )
        if flooded_pressure_Pa:
            raise CableSchemaError(
                "a flooded section has no fluid-tight barrier; omit pressure_barrier"
            )
        barrier = names.index(pressure_barrier)
    areas = section.layer_solid_areas()
    states: list[CableRadialState] = []
    for tension_N in tensions:
        axial = None
        if prescribed_axial_strain is None:
            axial = float(tension_N)
            if flooded_pressure_Pa:
                axial += flooded_pressure_Pa * _solid_area(section)
            if end_cap:
                bore = (
                    section.layers[barrier].inner_radius_m
                    if (section.model is None and section.layers)
                    else 0.0
                )
                outer = (
                    section.layers[-1].outer_radius_m
                    if section.layers
                    else section.model.ring_outer_radius_m  # type: ignore[union-attr]
                )
                axial += math.pi * (internal_pressure_Pa * bore**2 - p_ext * outer**2)
        elif tension_N or end_cap:
            raise CableSchemaError(
                "prescribed axial strain excludes tension and end cap"
            )
        result = section.solve(
            tension_N=axial or 0.0,
            prescribed_axial_strain=prescribed_axial_strain,
            torque_Nm=torque_Nm,
            prescribed_twist_per_m=prescribed_twist_per_m,
            internal_pressure_Pa=internal_pressure_Pa,
            external_pressure_Pa=p_ext,
            initial_wire_strain=initial,
            internal_pressure_layer=barrier,
        )
        states.append(
            CableRadialState(
                cable,
                result,
                float(tension_N),
                axial,
                float(internal_pressure_Pa),
                p_ext,
                float(flooded_pressure_Pa),
                end_cap,
                initial,
                residual,
                notes,
                pressure_barrier,
                areas,
            )
        )
    return states
