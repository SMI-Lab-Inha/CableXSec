# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""A solved normal section state shared by bending and local stress recovery.

True axial force includes pressure end thrust. Surface pressure is imposed
separately, without adding that thrust again. Concentric sections retain
extension, twist, radial displacement and thermal strain in one equilibrium.
The fixed-radius approximation retains its separate, fixed-twist definition.
Manufacturing clamping remains an explicit input to the bending adapters;
it is not inferred from an arbitrary prestrain or added twice to equilibrium.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from cablexsec.cables.schema import (
    CableData,
    CylindricalLayer,
    CylindricalOffsetLayer,
    CylindricalShellLayer,
    HelicalLayer,
    HelicalStrandedLayer,
)
from cablexsec.exceptions import CableSchemaError, CableSolverError
from cablexsec.materials import library
from cablexsec.mechanics.axisymmetric import AxisymmetricResult, compute_contact_forces
from cablexsec.mechanics.concentric_section import ElasticHelicalRing
from cablexsec.mechanics.contact_qp import (
    ContactQPResult,
    _concentric_layers,
    solve_contact_qp,
)
from cablexsec.mechanics.core_lay import core_angle
from cablexsec.mechanics.core_strands import axial_area, thermal_force_area
from cablexsec.mechanics.section_geometry import axial_cosine
from cablexsec.mechanics.stiffness import axial_stiffness_by_layer
from cablexsec.mechanics.thermal import ThermalResult


@dataclass(frozen=True)
class LayerNormalState:
    """Normal response; wire stress is along its helix, annulus stress is axial.

    For an annulus, sigma_r=A-B/r² and sigma_theta=A+B/r². Its torsional
    shear is G*r*twist. Axial force and torque are section-axis resultants.
    """

    longitudinal_stress_Pa: float
    axial_force_N: float
    torque_Nm: float
    radial_stress_constant_Pa: float = 0.0
    radial_stress_inverse_Pa_m2: float = 0.0
    shear_modulus_Pa: float = 0.0
    # In resolved cores, one longitudinal wire stress per declared ring.
    # longitudinal_stress_Pa is then the projected-area-weighted mean, not
    # a stress shared by every ring.
    strand_stress_Pa: tuple[float, ...] = ()


@dataclass(frozen=True)
class SectionState:
    cable: CableData
    contact_model: str
    tension_N: float
    axial_strain: float
    twist_per_m: float
    internal_pressure_Pa: float
    external_pressure_Pa: float
    temperature_C: Mapping[str, float]
    youngs_modulus_Pa: Mapping[str, float]
    modulus_scale: Mapping[str, float]
    free_thermal_strain: Mapping[str, float]
    layers: Mapping[str, LayerNormalState]
    radial: ContactQPResult | None
    bounded: AxisymmetricResult | None
    temperature_models: Mapping[str, str]
    torque_Nm: float
    prescribed_twist_per_m: float | None
    initial_wire_strain: Mapping[str, float]

    def require_cable(self, cable: CableData) -> None:
        if cable != self.cable:
            raise CableSchemaError(
                "section state belongs to a different cable definition"
            )


def _reference_modulus(layer: object) -> float:
    material = layer.material  # type: ignore[attr-defined]
    if isinstance(layer, (CylindricalLayer, CylindricalOffsetLayer)):
        try:
            return library.get_property(material, "E_eff_axial_Pa")
        except KeyError:
            pass
    return library.get_property(material, "E_Pa")


def _radius(layer: object) -> float:
    if isinstance(layer, HelicalLayer):
        return float(layer.r_helix_m or 0.0)
    if isinstance(layer, CylindricalShellLayer):
        return float(layer.r_mid_m or 0.0)
    if isinstance(layer, CylindricalOffsetLayer):
        return float((layer.r_offset_m or 0.0) + (layer.r_outer_m or 0.0))
    if isinstance(layer, CylindricalLayer):
        return float(layer.r_outer_m or 0.0)
    raise CableSchemaError(
        "nested-strand normal section mechanics require a strand model"
    )


def solve_section_state(
    cable: CableData,
    *,
    tension_N: float = 0.0,
    torque_Nm: float = 0.0,
    prescribed_twist_per_m: float | None = 0.0,
    p_int_Pa: float = 0.0,
    p_ext_Pa: float | None = None,
    thermal: ThermalResult | None = None,
    temperature_C: float | Mapping[str, float] | None = None,
    T_ref_C: float = 20.0,
    temperature_dependent_modulus: bool = True,
    youngs_modulus_Pa: Mapping[str, float] | None = None,
    initial_wire_strain: Mapping[str, float] | None = None,
    contact_model: str = "qp",
) -> SectionState:
    """Solve the section and recover layer stresses from that same solution.

    A temperature input activates registered E(T) laws within their source
    ranges. Materials without a registered law retain their stated constant
    modulus; the selected laws are returned. Explicit layer moduli override
    E(T). Each annulus has one temperature in this reduced radial model.
    An explicit external pressure replaces the initial-state boundary value.
    """
    if cable.missing_fields():
        raise CableSchemaError("section-state calculation requires complete geometry")
    if contact_model not in {"bounded", "qp"}:
        raise CableSchemaError("contact_model must be 'bounded' or 'qp'")
    if contact_model == "bounded":
        from cablexsec.clamping import require_radial_shrinkage_route  # noqa: PLC0415

        require_radial_shrinkage_route(cable, "bounded pressure chain")
    if any(isinstance(layer, HelicalStrandedLayer) for layer in cable.layers):
        raise CableSchemaError(
            "nested-strand normal section mechanics require a strand model"
        )
    external = (
        cable.initial_state.external_pressure_Pa if p_ext_Pa is None else p_ext_Pa
    )
    inputs = [tension_N, torque_Nm, p_int_Pa, external, T_ref_C]
    if prescribed_twist_per_m is not None:
        inputs.append(prescribed_twist_per_m)
    if not np.isfinite(inputs).all() or min(p_int_Pa, external) < 0:
        raise CableSchemaError(
            "section loads must be finite and pressures non-negative"
        )
    if thermal is not None and temperature_C is not None:
        raise CableSchemaError(
            "supply a thermal profile or layer temperatures, not both"
        )
    names = {layer.name for layer in cable.layers}
    for values in (
        youngs_modulus_Pa,
        temperature_C if isinstance(temperature_C, Mapping) else None,
    ):
        if values is not None and set(values) - names:
            raise CableSchemaError("section-state input names an unknown layer")
    if isinstance(temperature_C, Mapping) and set(temperature_C) != names:
        raise CableSchemaError("layer temperatures must cover every section component")
    if thermal is not None and (
        thermal.r_m.ndim != 1
        or len(thermal.r_m) < 2
        or thermal.T_C.shape != thermal.r_m.shape
        or not np.isfinite(thermal.r_m).all()
        or not np.isfinite(thermal.T_C).all()
        or np.any(np.diff(thermal.r_m) <= 0)
    ):
        raise CableSchemaError(
            "thermal profile must have finite ordered radial coordinates"
        )
    temperatures, moduli, scales, free, models = {}, {}, {}, {}, {}
    for layer in cable.layers:
        value = (
            float(np.interp(_radius(layer), thermal.r_m, thermal.T_C))
            if thermal is not None
            else float(temperature_C[layer.name])
            if isinstance(temperature_C, Mapping)
            else float(T_ref_C if temperature_C is None else temperature_C)
        )
        if not math.isfinite(value):
            raise CableSchemaError("layer temperatures must be finite")
        reference = _reference_modulus(layer)
        modulus = reference
        if (
            temperature_dependent_modulus
            and (temperature_C is not None or thermal is not None)
            and library.has_temperature_modulus_model(layer.material)
            and (youngs_modulus_Pa is None or layer.name not in youngs_modulus_Pa)
        ):
            modulus = library.youngs_modulus_at_temperature(layer.material, value)
            models[layer.name] = library.get_temperature_modulus_citation(
                layer.material
            )
        if youngs_modulus_Pa is not None and layer.name in youngs_modulus_Pa:
            modulus = youngs_modulus_Pa[layer.name]
            models[layer.name] = "explicit supplied layer modulus"
        if not math.isfinite(modulus) or modulus <= 0:
            raise CableSchemaError("layer moduli must be finite and positive")
        temperatures[layer.name], moduli[layer.name] = value, modulus
        scales[layer.name] = modulus / reference
        delta = value - T_ref_C
        free[layer.name] = (
            library.get_property(layer.material, "alpha_T_K") * delta if delta else 0.0
        )
    radial = None
    bounded = None
    layer_states = {}
    if contact_model == "qp":
        geometry = _concentric_layers(
            cable,
            youngs_modulus_Pa=moduli,
            free_thermal_strain=free,
            initial_wire_strain=initial_wire_strain,
        )
        radial = solve_contact_qp(
            cable,
            tension_N,
            p_ext_Pa=external,
            p_int_Pa=p_int_Pa,
            torque_Nm=torque_Nm,
            prescribed_twist_per_m=prescribed_twist_per_m,
            youngs_modulus_Pa=moduli,
            free_thermal_strain=free,
            initial_wire_strain=initial_wire_strain,
        )
        solution = radial.section
        strain, twist = solution.axial_strain, solution.twist_per_m
        for elastic in geometry:
            e = elastic.youngs_modulus_Pa
            if isinstance(elastic, ElasticHelicalRing):
                a, r = elastic.signed_lay_angle_rad, elastic.helix_radius_m
                c, s = math.cos(a), math.sin(a)
                wire_strain = (
                    c * c * strain
                    + r * c * s * twist
                    + s * s * solution.inner_displacement_m[elastic.name] / r
                    - elastic.free_thermal_strain
                    + elastic.initial_wire_strain
                )
                stress = e * wire_strain
                wire_area = elastic.total_area_m2
                layer_states[elastic.name] = LayerNormalState(
                    stress, stress * wire_area * c, stress * wire_area * r * s
                )
            else:
                a, b, nu = (
                    elastic.inner_radius_m,
                    elastic.outer_radius_m,
                    elastic.poisson_ratio,
                )
                u_a, u_b = (
                    solution.inner_displacement_m[elastic.name],
                    solution.outer_displacement_m[elastic.name],
                )
                coeff_a = (b * u_b - a * u_a) / (b * b - a * a)
                coeff_b = a * b * (b * u_a - a * u_b) / (b * b - a * a)
                shear = e / (2 * (1 + nu))
                lame = e * nu / ((1 + nu) * (1 - 2 * nu))
                # Free strain (radial, hoop, axial) = (e + t, e + t, e): the
                # isotropic thermal part e and the in-plane extrusion shrinkage t.
                free_e = elastic.free_thermal_strain
                free_t = elastic.free_transverse_strain
                common = lame * (2 * coeff_a + strain - 3 * free_e - 2 * free_t)
                axial = common + 2 * shear * (strain - free_e)
                hoop_const = common + 2 * shear * (coeff_a - free_e - free_t)
                torque = shear * math.pi * (b**4 - a**4) / 2 * twist
                layer_states[elastic.name] = LayerNormalState(
                    axial,
                    axial * math.pi * (b * b - a * a),
                    torque,
                    hoop_const,
                    2 * shear * coeff_b,
                    shear,
                )
        recovered = sum(layer.axial_force_N for layer in layer_states.values())
        if not math.isclose(recovered, tension_N, rel_tol=1e-9, abs_tol=1e-5):
            raise CableSolverError(
                "recovered layer forces do not balance the applied section force"
            )
        recovered_torque = sum(layer.torque_Nm for layer in layer_states.values())
        if not math.isclose(
            recovered_torque, solution.torque_resultant_Nm, rel_tol=1e-9, abs_tol=1e-7
        ):
            raise CableSolverError(
                "recovered layer torques do not balance the section torque"
            )
    else:
        if torque_Nm != 0 or prescribed_twist_per_m != 0 or initial_wire_strain:
            raise CableSchemaError(
                "torque, free twist and assembly strain require the "
                "coupled section model"
            )
        stiffness = {
            name: value * scales[name]
            for name, value in axial_stiffness_by_layer(cable).items()
        }
        total = sum(stiffness.values())
        if total <= 0:
            raise CableSchemaError("section has no axial rigidity")
        free_axis = {}
        for layer in cable.layers:
            assembly_cosine = (
                math.cos(core_angle(cable, layer))
                if isinstance(layer, CylindricalOffsetLayer)
                else 1.0
            )
            if isinstance(layer, CylindricalOffsetLayer) and layer.strand_layers:
                free_axis[layer.name] = (
                    free[layer.name]
                    * thermal_force_area(layer)
                    / axial_area(layer)
                    / assembly_cosine**2
                )
                continue
            cosine = axial_cosine(layer)
            free_axis[layer.name] = free[layer.name] / (cosine * assembly_cosine) ** 2
        strain = (
            tension_N
            + sum(value * free_axis[name] for name, value in stiffness.items())
        ) / total
        twist = 0.0
        forces = {}
        for layer in cable.layers:
            force = stiffness.get(layer.name, 0.0) * (strain - free_axis[layer.name])
            forces[layer.name] = force
            assembly_angle = (
                core_angle(cable, layer)
                if isinstance(layer, CylindricalOffsetLayer)
                else 0.0
            )
            assembly_cosine = math.cos(assembly_angle)
            if isinstance(layer, CylindricalOffsetLayer) and layer.strand_layers:
                wire_stress = tuple(
                    moduli[layer.name]
                    * (
                        (math.cos(r.lay_angle_rad) * assembly_cosine) ** 2 * strain
                        - free[layer.name]
                    )
                    if layer.structural
                    else 0.0
                    for r in layer.strand_layers
                )
                recovered_force = (
                    layer.n_cores
                    * assembly_cosine
                    * sum(
                        stress * ring.area_m2 * math.cos(ring.lay_angle_rad)
                        for stress, ring in zip(
                            wire_stress, layer.strand_layers, strict=True
                        )
                    )
                )
                if not math.isclose(
                    recovered_force, force, rel_tol=1e-10, abs_tol=1e-7
                ):
                    raise CableSolverError("resolved core wire forces do not balance")
                layer_states[layer.name] = LayerNormalState(
                    force
                    / (layer.n_cores * assembly_cosine * thermal_force_area(layer)),
                    force,
                    force * (layer.r_offset_m or 0) * math.tan(assembly_angle),
                    strand_stress_Pa=wire_stress,
                )
                continue
            cosine = axial_cosine(layer)
            modulus = moduli[layer.name] * (
                layer.f_corrugation if isinstance(layer, CylindricalShellLayer) else 1.0
            )
            stress = (
                modulus * ((cosine * assembly_cosine) ** 2 * strain - free[layer.name])
                if stiffness.get(layer.name, 0)
                else 0.0
            )
            if isinstance(layer, HelicalLayer):
                sign = 1 if layer.lay_direction == "right_hand" else -1
                torque = (
                    force
                    * (layer.r_helix_m or 0)
                    * math.tan(layer.lay_angle_rad or 0)
                    * sign
                )
            elif isinstance(layer, CylindricalOffsetLayer):
                torque = force * (layer.r_offset_m or 0) * math.tan(assembly_angle)
            else:
                torque = 0.0
            layer_states[layer.name] = LayerNormalState(stress, force, torque)
        if cable.helical_layers:
            bounded = compute_contact_forces(
                cable,
                tension_N,
                p_ext_Pa=external,
                p_int_Pa=p_int_Pa,
                layer_tension_N=forces,
            )
    return SectionState(
        cable,
        contact_model,
        tension_N,
        strain,
        twist,
        p_int_Pa,
        external,
        temperatures,
        moduli,
        scales,
        free,
        layer_states,
        radial,
        bounded,
        models,
        torque_Nm,
        prescribed_twist_per_m,
        dict(initial_wire_strain or {}),
    )
