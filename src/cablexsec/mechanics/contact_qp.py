# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Cable-data adapter for coupled concentric radial contact.

The historical solve_contact_qp name is retained. The former pressure-based
compliance chain is replaced by elastic-energy minimisation with common
extension and twist. Multipliers are normal forces per cable-axis length;
pressure is obtained by dividing by the interface circumference.

All layers must be genuinely concentric. Offset cores, nested strands and
corrugated shells require explicit radial constitutive models and are
rejected, not silently omitted. Each accepted annulus uses its isotropic
material modulus. Manufacturing contact pressures are not assembly prestrains
and are not included in this incremental mechanical equilibrium. For assembly
strain or layer temperatures, use solve_concentric_section directly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from cablexsec.cables.schema import (
    CableData,
    CylindricalLayer,
    CylindricalOffsetLayer,
    CylindricalShellLayer,
    HelicalLayer,
    is_serving,
    layer_stiffness,
)
from cablexsec.cables.schema import scope_entry_points as _scope_entry_points
from cablexsec.exceptions import CableSchemaError
from cablexsec.materials import library
from cablexsec.mechanics.axisymmetric import lame_thick_wall
from cablexsec.mechanics.concentric_section import (
    ConcentricSectionResult,
    ElasticAnnulus,
    ElasticHelicalRing,
    solve_concentric_section,
)

__all__ = [
    "ContactQPResult",
    "radial_foundation_modulus",
    "solve_contact_qp",
    "surface_compliances",
]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ContactQPResult:
    """Concentric equilibrium, retaining historical result field names.

    Fn_per_wire_N is a normal line load [N/m of cable axis], not a point force.
    Both radial-face reactions are counted; capstan is not added a second time.
    radial_displacement_m is the outer-surface motion. Both surface motions,
    common extension and twist, and resultants are available in section.
    complementarity_residual is max(abs(line_force * gap)) [N].
    """

    interface_radii_m: FloatArray
    contact_pressure_Pa: FloatArray
    radial_displacement_m: dict[str, float]
    Fn_per_wire_N: dict[str, float]
    complementarity_residual: float
    n_active: int
    section: ConcentricSectionResult


def radial_foundation_modulus(layer: object) -> float:
    """Isolated radial stiffness [Pa/m], not an assembled section tangent.

    Helical extension and twist are fixed, with pressure referred to the
    wire-centre circumference. The annulus expression is thin-wall hoop.
    """
    if isinstance(layer, HelicalLayer):
        if None in (
            layer.n_wires,
            layer.r_helix_m,
            layer.lay_angle_rad,
            layer.d_wire_m,
        ):
            raise CableSchemaError("radial stiffness requires complete helix geometry")
        assert layer.n_wires and layer.r_helix_m and layer.d_wire_m
        assert layer.lay_angle_rad is not None
        angle = layer.lay_angle_rad
        if not 0 < abs(angle) < math.pi / 2:
            raise CableSchemaError(
                "radial helix stiffness requires a non-zero lay angle"
            )
        area = math.pi * layer.d_wire_m**2 / 4
        return (
            layer.n_wires
            * library.get_property(layer.material, "E_Pa")
            * area
            * math.sin(angle) ** 4
            / (2 * math.pi * layer.r_helix_m**3 * math.cos(angle))
        )
    if isinstance(layer, CylindricalLayer):
        if layer.r_inner_m is None or layer.r_outer_m is None:
            raise CableSchemaError(
                "radial stiffness requires complete annulus geometry"
            )
        a, b = layer.r_inner_m, layer.r_outer_m
        return (
            library.get_property(layer.material, "E_Pa") * (b - a) / ((a + b) / 2) ** 2
        )
    raise CableSchemaError("unsupported layer for radial stiffness")


def surface_compliances(layer: object) -> tuple[float, float, float]:
    """Plane-stress Lame (inner, outer, inner-from-outer) values [m/Pa].

    The reverse cross-compliance is (a/b) times the returned cross value.
    These two pressure compliances are NOT interchangeable. The assembled
    solver uses force-weighted reciprocal elasticity directly. Solid cores
    have no inner surface and return zero inner/cross values.
    """
    if isinstance(layer, CylindricalLayer):
        if layer.r_inner_m is None or layer.r_outer_m is None:
            raise CableSchemaError("surface compliance requires complete geometry")
        a, b = layer.r_inner_m, layer.r_outer_m
        e = library.get_property(layer.material, "E_Pa")
        nu = library.get_property(layer.material, "nu")
        if a == 0:
            return 0.0, (1 - nu) * b / e, 0.0
        return (
            float(lame_thick_wall(a, a, b, 1, 0, e, nu).u_r_m),
            float(-lame_thick_wall(b, a, b, 0, 1, e, nu).u_r_m),
            float(-lame_thick_wall(a, a, b, 0, 1, e, nu).u_r_m),
        )
    c = 1 / radial_foundation_modulus(layer)
    return c, c, c


def _concentric_layers(
    cable: CableData,
    *,
    youngs_modulus_Pa: Mapping[str, float] | None = None,
    free_thermal_strain: Mapping[str, float] | None = None,
    initial_wire_strain: Mapping[str, float] | None = None,
) -> list[ElasticAnnulus | ElasticHelicalRing]:
    """Translate every layer or raise; no silent geometrical homogenisation."""
    stiff = [x.name for x in cable.layers if layer_stiffness(x) is not None]
    if stiff:
        raise CableSchemaError(
            f"{stiff[0]}: a stiffness-defined layer has no radial constitutive law; "
            "the coupled radial models ('qp', 'solved') need a material; use the "
            "bounded contact model"
        )
    result: list[ElasticAnnulus | ElasticHelicalRing] = []
    for layer in cable.layers:
        if isinstance(layer, CylindricalOffsetLayer) and (
            layer.n_cores != 1 or layer.r_offset_m != 0
        ):
            raise CableSchemaError(
                "coupled radial contact requires concentric layers; "
                "trefoil/offset cores need a separate radial constitutive model"
            )
        if isinstance(layer, CylindricalOffsetLayer):
            if layer.strand_layers or layer.strand_lay_angle_deg not in (None, 0.0):
                raise CableSchemaError(
                    "coupled radial contact requires explicit concentric wire layers; "
                    "a strand bundle cannot be replaced by a solid annulus"
                )
            if layer.conductor_area_mm2 is not None and not math.isclose(
                layer.conductor_area_mm2 * 1e-6,
                math.pi * ((layer.r_outer_m or 0) ** 2 - (layer.r_inner_m or 0) ** 2),
                rel_tol=1e-12,
            ):
                raise CableSchemaError(
                    "conductor metal area differs from its envelope; a coupled "
                    "radial bundle law is required instead of a solid annulus"
                )
        # This material's historical axial-key value is the copper wire
        # modulus, not a stranded-pack homogenisation (see the library).
        # Its use here models a solid isotropic annulus, not strand compaction.
        key = library.axial_modulus_key(layer.material)
        e = library.get_property(layer.material, key)
        if isinstance(layer, HelicalLayer):
            if None in (
                layer.n_wires,
                layer.r_helix_m,
                layer.d_wire_m,
                layer.lay_angle_rad,
            ):
                raise CableSchemaError(f"{layer.name}: incomplete helix geometry")
            assert layer.r_helix_m and layer.d_wire_m and layer.n_wires
            assert layer.lay_angle_rad is not None
            angle = abs(layer.lay_angle_rad) * (
                -1 if layer.lay_direction == "left_hand" else 1
            )
            result.append(
                ElasticHelicalRing(
                    layer.name,
                    layer.r_helix_m,
                    layer.d_wire_m,
                    layer.n_wires,
                    angle,
                    e,
                    radial_thickness_m=None
                    if layer.wire_shape == "round"
                    else layer.wire_radial_thickness_m,
                    wire_area_m2=None
                    if layer.wire_shape == "round"
                    else layer.wire_section.area,
                )
            )
            continue
        if isinstance(layer, CylindricalOffsetLayer):
            if layer.n_cores != 1 or layer.r_offset_m != 0:
                raise CableSchemaError(
                    "coupled radial contact requires concentric layers; "
                    "trefoil/offset cores need a separate radial constitutive model"
                )
            a, b = layer.r_inner_m, layer.r_outer_m
        elif isinstance(layer, CylindricalLayer):
            a, b = layer.r_inner_m, layer.r_outer_m
        elif isinstance(layer, CylindricalShellLayer):
            if layer.f_corrugation != 1:
                raise CableSchemaError(
                    "corrugated shells require a radial constitutive model"
                )
            if layer.r_mid_m is None or layer.t_m is None:
                raise CableSchemaError(f"{layer.name}: incomplete shell geometry")
            a, b = layer.r_mid_m - layer.t_m / 2, layer.r_mid_m + layer.t_m / 2
        else:
            raise CableSchemaError(f"{layer.name}: unsupported concentric layer type")
        if a is None or b is None:
            raise CableSchemaError(f"{layer.name}: incomplete annulus geometry")
        shrink = getattr(layer, "extrusion_shrinkage", None)
        result.append(
            ElasticAnnulus(
                layer.name,
                a,
                b,
                e,
                library.get_property(layer.material, "nu"),
                free_transverse_strain=-float(shrink.hoop_strain) if shrink else 0.0,
                carries_axial_load=not is_serving(layer),
            )
        )
    result.sort(key=lambda layer: layer.inner_radius_m)
    names = {layer.name for layer in result}
    helices = {layer.name for layer in result if isinstance(layer, ElasticHelicalRing)}
    for values, allowed, label in (
        (youngs_modulus_Pa, names, "modulus"),
        (free_thermal_strain, names, "thermal strain"),
        (initial_wire_strain, helices, "initial wire strain"),
    ):
        if values is not None and set(values) - allowed:
            raise CableSchemaError(f"unknown or incompatible layer in {label} inputs")
    updated = []
    for elastic in result:
        current = replace(
            elastic,
            youngs_modulus_Pa=(youngs_modulus_Pa or {}).get(
                elastic.name, elastic.youngs_modulus_Pa
            ),
            free_thermal_strain=(free_thermal_strain or {}).get(
                elastic.name, elastic.free_thermal_strain
            ),
        )
        if isinstance(current, ElasticHelicalRing) and initial_wire_strain:
            current = replace(
                current, initial_wire_strain=initial_wire_strain.get(current.name, 0.0)
            )
        updated.append(current)
    return updated


def solve_contact_qp(
    cable: CableData,
    T_N: float,
    *,
    p_ext_Pa: float = 0.0,
    p_int_Pa: float = 0.0,
    torque_Nm: float = 0.0,
    prescribed_twist_per_m: float | None = 0.0,
    youngs_modulus_Pa: Mapping[str, float] | None = None,
    free_thermal_strain: Mapping[str, float] | None = None,
    initial_wire_strain: Mapping[str, float] | None = None,
) -> ContactQPResult:
    """Concentric equilibrium with true axial force T_N [N].

    Include pressure end thrust explicitly in T_N. All layers share extension;
    twist is fixed by default. Pass prescribed_twist_per_m=None for applied
    torque. Stated geometric gaps are retained. Direct manufacturing contact
    pressures are not interpreted as assembly strains. This calculation does
    not supply the azimuthal pressure variation caused by bending.
    """
    layers = _concentric_layers(
        cable,
        youngs_modulus_Pa=youngs_modulus_Pa,
        free_thermal_strain=free_thermal_strain,
        initial_wire_strain=initial_wire_strain,
    )
    section = solve_concentric_section(
        layers,
        tension_N=T_N,
        torque_Nm=torque_Nm,
        prescribed_twist_per_m=prescribed_twist_per_m,
        external_pressure_Pa=p_ext_Pa,
        internal_pressure_Pa=p_int_Pa,
    )
    radii = np.array(
        [
            (inner.outer_radius_m + outer.inner_radius_m) / 2
            for inner, outer in zip(layers[:-1], layers[1:], strict=True)
        ]
    )
    line_load: dict[str, float] = {}
    reactions = section.contact_line_force_N_per_m
    for i, layer in enumerate(layers):
        if isinstance(layer, ElasticHelicalRing):
            below = (
                reactions[i - 1] if i else 2 * math.pi * layer.inner_radius_m * p_int_Pa
            )
            above = (
                reactions[i]
                if i < len(reactions)
                else 2 * math.pi * layer.outer_radius_m * p_ext_Pa
            )
            line_load[layer.name] = float((below + above) / layer.wire_count)
    return ContactQPResult(
        radii,
        section.contact_pressure_Pa,
        section.outer_displacement_m,
        line_load,
        section.complementarity_residual_N,
        int(np.count_nonzero(reactions > 0)),
        section,
    )


# Public entry points read the cable's own materials when called directly.
_scope_entry_points(globals(), __all__)
