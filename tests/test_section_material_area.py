# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Independent metal-area, helical-volume and interstitial-filler checks."""

import math

import pytest
from fowt_construction import _global_properties
from pydantic import ValidationError
from scipy.integrate import quad

import cablexsec as cx
from cablexsec.cables.schema import (
    CableData,
    CableMeta,
    CoreStrandRing,
    CylindricalLayer,
    CylindricalOffsetLayer,
)
from cablexsec.exceptions import CableSchemaError
from cablexsec.materials import library
from cablexsec.mechanics.section_geometry import filler_area, material_volume_per_length
from cablexsec.mechanics.stiffness import axial_stiffness_by_layer

pytestmark = pytest.mark.analytical_sanity


def _bundle(angle=None):
    layer = CylindricalOffsetLayer(
        name="conductor",
        material="copper_stranded",
        n_cores=1,
        r_offset_m=0.0,
        r_inner_m=0,
        r_outer_m=0.01,
        conductor_area_mm2=240,
        strand_lay_angle_deg=angle,
    )
    return CableData(cable=CableMeta(name="synthetic porous bundle"), layers=(layer,))


@pytest.mark.parametrize("angle", [None, 0.0, 12.0, 25.0])
@pytest.mark.parametrize("temperature", [20, 70])
def test_declared_metal_area_controls_mass_force_and_recovered_wire_stress(
    angle, temperature
):
    cable = _bundle(angle)
    layer = cable.layers[0]
    area, cosine = 240e-6, math.cos(math.radians(angle or 0))
    e = library.get_property(layer.material, "E_eff_axial_Pa")
    thermal = library.get_property(layer.material, "alpha_T_K") * (temperature - 20)
    expected = e * area * cosine**3
    assert axial_stiffness_by_layer(cable)[layer.name] == pytest.approx(expected)
    assert material_volume_per_length(layer, cable) == pytest.approx(area / cosine)
    state = cx.solve_section_state(
        cable, tension_N=350, temperature_C=temperature, contact_model="bounded"
    )
    assert state.axial_strain == pytest.approx(350 / expected + thermal / cosine**2)
    component = cx.compute_component_response(cable, 0, section_state=state).states[0]
    assert component.sigma_local_max_Pa * area * cosine == pytest.approx(350, rel=1e-12)
    assert component.sigma_stick_min_Pa == pytest.approx(
        state.layers[layer.name].longitudinal_stress_Pa
    )


def _filler_case(inner=0.0, outer=0.10, count=3, offset=0.04):
    core = CylindricalOffsetLayer(
        name="core",
        material="copper_stranded",
        n_cores=count,
        r_offset_m=offset,
        r_inner_m=0,
        r_outer_m=0.022,
    )
    filler = CylindricalLayer(
        name="filler",
        material="pp_filler",
        structural=False,
        r_inner_m=inner,
        r_outer_m=outer,
    )
    return CableData(
        cable=CableMeta(name="synthetic interstice"), layers=(core, filler)
    )


def test_full_core_envelopes_are_excluded_once_from_both_axial_rigidity_and_mass():
    cable = _filler_case()
    core, filler = cable.layers
    area = math.pi * (0.10**2 - 3 * 0.022**2)
    assert filler_area(filler, cable) == pytest.approx(area)
    assert axial_stiffness_by_layer(cable)["filler"] == pytest.approx(
        area * library.get_property("pp_filler", "E_Pa")
    )
    expected_mass = area * library.get_property(
        "pp_filler", "rho_kg_m3"
    ) + 3 * math.pi * 0.022**2 * library.get_property("copper_stranded", "rho_kg_m3")
    assert _global_properties(cable)["dry_mass_kg_m"] == pytest.approx(expected_mass)


@pytest.mark.parametrize(
    "inner,outer", [(0.0, 0.035), (0.027, 0.048), (0.043, 0.07), (0.07, 0.1)]
)
def test_partial_envelope_intersection_against_independent_polar_quadrature(
    inner, outer
):
    cable = _filler_case(inner, outer)
    core, filler = cable.layers
    # Integrate the available angle on each radius, rather than using the
    # circle-circle area formula in the implementation. Three disjoint cores.
    d, a = core.r_offset_m, core.r_outer_m

    def available_ring(radius):
        if radius <= d - a or radius >= d + a:
            covered = 0.0
        else:
            covered = 2 * math.acos(
                (radius * radius + d * d - a * a) / (2 * radius * d)
            )
        return radius * (2 * math.pi - 3 * covered)

    points = [x for x in (d - a, d + a) if inner < x < outer]
    independent = quad(
        available_ring, inner, outer, points=points, epsabs=1e-13, epsrel=1e-11
    )[0]
    assert filler_area(filler, cable) == pytest.approx(independent, rel=1e-10)


def test_concentric_single_core_is_not_counted_as_three_cores():
    cable = _filler_case(count=1, offset=0, inner=0.022)
    assert filler_area(cable.layers[1], cable) == pytest.approx(
        math.pi * (0.1**2 - 0.022**2)
    )


def test_filler_through_a_concentric_core_is_refused():
    with pytest.raises(ValidationError, match="overlaps or lies inside"):
        _filler_case(count=1, offset=0)


def test_nested_offset_annuli_form_one_core_envelope():
    cable = _filler_case()
    core, filler = cable.layers
    sheath = core.model_copy(
        update={
            "name": "sheath",
            "material": "hdpe",
            "r_inner_m": 0.022,
            "r_outer_m": 0.026,
        }
    )
    cable = cable.model_copy(update={"layers": (core, sheath, filler)})
    assert filler_area(filler, cable) == pytest.approx(
        math.pi * (0.1**2 - 3 * 0.026**2)
    )


def test_overlapping_cores_are_refused_before_any_filler_area():
    with pytest.raises(ValidationError, match="cores overlap"):
        _filler_case(offset=0.01)


def test_resolved_ring_mass_is_sum_of_actual_wire_lengths():
    layer = (
        _bundle()
        .layers[0]
        .model_copy(
            update={
                "conductor_area_mm2": None,
                "strand_layers": (
                    CoreStrandRing(n_wires=1, d_wire_m=0.002, r_helix_m=0),
                    CoreStrandRing(
                        n_wires=8, d_wire_m=0.001, r_helix_m=0.004, pitch_m=0.1
                    ),
                ),
            }
        )
    )
    cable = _bundle().model_copy(update={"layers": (layer,)})
    exact_volume = math.pi * 0.002**2 / 4 + 8 * math.pi * 0.001**2 / 4 * math.sqrt(
        1 + (2 * math.pi * 0.004 / 0.1) ** 2
    )
    assert material_volume_per_length(layer, cable) == pytest.approx(exact_volume)


@pytest.mark.parametrize("angle", [None, 12])
def test_radial_solver_rejects_porous_or_laid_bundle_as_solid_annulus(angle):
    with pytest.raises(CableSchemaError, match="(metal area|strand bundle)"):
        cx.solve_section_state(_bundle(angle), contact_model="qp")


def test_radial_solver_rejects_resolved_rings_even_at_zero_offset():
    layer = (
        _bundle()
        .layers[0]
        .model_copy(
            update={
                "conductor_area_mm2": None,
                "strand_layers": (
                    CoreStrandRing(n_wires=1, d_wire_m=0.002, r_helix_m=0),
                ),
            }
        )
    )
    with pytest.raises(CableSchemaError, match="explicit concentric wire"):
        cx.solve_section_state(
            _bundle().model_copy(update={"layers": (layer,)}), contact_model="qp"
        )


def test_public_solid_equivalent_has_same_material_area_in_both_normal_models():
    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    core = cable.get_layer("cu_conductor")
    assert math.pi * core.r_outer_m**2 == pytest.approx(
        core.conductor_area_mm2 * 1e-6, rel=1e-12
    )
    single = cable.model_copy(update={"layers": (core,)})
    bounded = cx.solve_section_state(single, tension_N=500, contact_model="bounded")
    coupled = cx.solve_section_state(single, tension_N=500, contact_model="qp")
    assert coupled.axial_strain == pytest.approx(bounded.axial_strain, rel=1e-12)
