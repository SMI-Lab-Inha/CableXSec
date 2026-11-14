# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Independent stress/resultant limits and transfer of the solved section state."""

import math

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.capacity import wire_yield_curvature
from cablexsec.cables.schema import CableData, CableMeta, CylindricalLayer
from cablexsec.exceptions import CableSchemaError
from cablexsec.materials import library

pytestmark = pytest.mark.analytical_sanity


def _annulus(bore=0.0):
    return CableData(
        cable=CableMeta(name="isotropic reference"),
        layers=(
            CylindricalLayer(
                name="metal",
                material="galvanised_steel_armour",
                r_inner_m=bore,
                r_outer_m=0.05,
            ),
        ),
    )


def _cable():
    return cx.load_cable(cx.inter_array_single_core_yaml_path(), geometry_override=None)


@pytest.mark.parametrize("bore", [0.0, 0.04])
def test_annulus_recovers_applied_axial_force_and_saint_venant_torque(bore):
    cable = _annulus(bore)
    state = cx.solve_section_state(
        cable, tension_N=1e5, torque_Nm=150, prescribed_twist_per_m=None
    )
    e = library.get_property("galvanised_steel_armour", "E_Pa")
    nu = library.get_property("galvanised_steel_armour", "nu")
    area = math.pi * (0.05**2 - bore**2)
    polar = math.pi * (0.05**4 - bore**4) / 2
    assert state.axial_strain == pytest.approx(1e5 / (e * area), rel=1e-11)
    assert state.twist_per_m == pytest.approx(
        150 / (e / (2 * (1 + nu)) * polar), rel=1e-11
    )
    assert state.layers["metal"].longitudinal_stress_Pa == pytest.approx(
        1e5 / area, rel=1e-11
    )
    assert state.layers["metal"].torque_Nm == pytest.approx(150, rel=1e-11)


@pytest.mark.parametrize("closed_ends", [False, True])
def test_solid_hydrostatic_limit_has_correct_axial_and_equivalent_stress(closed_ends):
    cable = _annulus()
    pressure, area = 2e6, math.pi * 0.05**2
    tension = -pressure * area if closed_ends else 0.0
    state = cx.solve_section_state(cable, tension_N=tension, p_ext_Pa=pressure)
    component = cx.compute_component_response(cable, 0, section_state=state).states[0]
    assert component.sigma_stick_max_Pa == pytest.approx(
        -pressure if closed_ends else 0, abs=1e-6
    )
    assert component.sigma_equivalent_max_Pa == pytest.approx(
        0 if closed_ends else pressure, abs=1e-6
    )


def test_thick_tube_pressure_end_thrust_is_not_added_twice():
    a, b, pressure = 0.04, 0.05, 3e6
    cable = _annulus(a)
    state = cx.solve_section_state(
        cable, tension_N=pressure * math.pi * a * a, p_int_Pa=pressure
    )
    axial = pressure * a * a / (b * b - a * a)
    normal = state.layers["metal"]
    assert normal.longitudinal_stress_Pa == pytest.approx(axial, rel=1e-11)
    assert (
        normal.radial_stress_constant_Pa - normal.radial_stress_inverse_Pa_m2 / a** 2
        == pytest.approx(-pressure, rel=1e-11)
    )
    assert (
        normal.radial_stress_constant_Pa + normal.radial_stress_inverse_Pa_m2 / a** 2
        == pytest.approx(pressure * (a * a + b * b) / (b * b - a * a), rel=1e-11)
    )


def test_wire_stress_and_layer_resultants_retain_radial_twist_strain():
    cable = _cable()
    state = cx.solve_section_state(
        cable, tension_N=1e5, torque_Nm=100, prescribed_twist_per_m=None, p_ext_Pa=2e6
    )
    wires = cx.compute_stress(cable, 0, section_state=state)
    assert state.radial is not None
    assert sum(v.axial_force_N for v in state.layers.values()) == pytest.approx(
        1e5, rel=1e-10
    )
    assert sum(v.torque_Nm for v in state.layers.values()) == pytest.approx(
        100, rel=1e-10
    )
    for layer in cable.helical_layers:
        angle = abs(layer.lay_angle_rad) * (
            -1 if layer.lay_direction == "left_hand" else 1
        )
        c, s, r = math.cos(angle), math.sin(angle), layer.r_helix_m
        exact_strain = (
            c * c * state.axial_strain
            + r * c * s * state.twist_per_m
            + s * s * state.radial.section.inner_displacement_m[layer.name] / r
        )
        exact_stress = library.get_property(layer.material, "E_Pa") * exact_strain
        np.testing.assert_allclose(
            wires.sigma_total_Pa[layer.name], exact_stress, rtol=1e-11
        )
        np.testing.assert_array_equal(wires.sigma_T_Pa[layer.name], 0)
    assert wires.sigma_A_Pa["inner_armour"][0] != pytest.approx(
        wires.sigma_A_Pa["outer_armour"][0]
    )


def test_component_pressure_is_forwarded_to_the_same_normal_state():
    cable = _cable()
    a = cx.compute_component_response(
        cable, 0.05, tension_N=1e5, contact_model="qp", p_ext_Pa=0
    )
    b = cx.compute_component_response(
        cable, 0.05, tension_N=1e5, contact_model="qp", p_ext_Pa=2e6
    )
    assert b.section_state.external_pressure_Pa == 2e6
    armour_a = next(s for s in a.states if s.name == "inner_armour")
    armour_b = next(s for s in b.states if s.name == "inner_armour")
    assert armour_a.sigma_local_max_Pa != pytest.approx(armour_b.sigma_local_max_Pa)
    wires = cx.compute_stress(cable, 0.05, section_state=b.section_state)
    assert armour_b.sigma_local_max_Pa == pytest.approx(
        np.max(wires.sigma_perimeter_max_Pa["inner_armour"]), rel=1e-12
    )


def test_free_thermal_expansion_does_not_receive_a_second_thermal_stress():
    cable = _annulus()
    alpha = library.get_property("galvanised_steel_armour", "alpha_T_K")
    state = cx.solve_section_state(cable, temperature_C=60, T_ref_C=20)
    response = cx.compute_component_response(cable, 0, section_state=state)
    assert state.axial_strain == pytest.approx(alpha * 40, rel=1e-11)
    assert response.states[0].sigma_equivalent_max_Pa < 1e-5


def test_temperature_changes_contact_shear_support_and_shared_bending_stress():
    cable = _cable()
    state = cx.solve_section_state(cable, tension_N=1e5, p_ext_Pa=1e6, temperature_C=60)
    assert state.temperature_models
    for layer in cable.layers:
        if layer.material in library.TEMPERATURE_MODULUS_MODELS:
            assert state.youngs_modulus_Pa[
                layer.name
            ] == library.youngs_modulus_at_temperature(layer.material, 60)
    kappa = np.linspace(0, 0.04, 200)
    loop = cx.compute_mkappa(cable, kappa, section_state=state)
    stress = cx.compute_stress(cable, 0.04, section_state=state)
    assert loop.moment_Nm[-1] == pytest.approx(stress.moment_recovered_Nm, rel=1e-11)
    assert sum(v.axial_force_N for v in state.layers.values()) == pytest.approx(
        1e5, rel=1e-10
    )


def test_state_rejects_conflicting_loads_and_different_geometry():
    cable = _cable()
    state = cx.solve_section_state(cable, tension_N=1e5)
    with pytest.raises(CableSchemaError, match="not both"):
        cx.compute_stress(cable, 0, tension_N=1e5, section_state=state)
    with pytest.raises(CableSchemaError, match="different cable"):
        cx.compute_stress(_annulus(), 0, section_state=state)


def test_temperature_law_is_not_extrapolated():
    with pytest.raises(ValueError, match="outside the sourced"):
        cx.solve_section_state(_cable(), temperature_C=100)


def test_explicit_modulus_replaces_a_temperature_law_without_extrapolating_it():
    cable = _cable()
    # A supplied modulus is a separately declared input, not extrapolated E(T).
    supplied = {
        layer.name: library.get_property(layer.material, "E_Pa")
        for layer in cable.layers
        if layer.material in library.TEMPERATURE_MODULUS_MODELS
    }
    state = cx.solve_section_state(cable, temperature_C=100, youngs_modulus_Pa=supplied)
    for name, value in supplied.items():
        assert state.youngs_modulus_Pa[name] == value
        assert state.temperature_models[name] == "explicit supplied layer modulus"


def test_solved_mean_stress_and_moduli_enter_wire_yield_screening():
    cable = _cable()
    moduli = {layer.name: 1.5e11 for layer in cable.helical_layers}
    state = cx.solve_section_state(
        cable, tension_N=1e5, p_ext_Pa=1e6, youngs_modulus_Pa=moduli
    )
    expected = math.inf
    for layer in cable.helical_layers:
        angle = layer.lay_angle_rad
        # With mu=0 only local wire bending remains. Resolve the actual wire
        # azimuths: a wire need not coincide with the continuous maximum.
        slope = (
            moduli[layer.name]
            * layer.d_wire_m
            / 2
            * max(
                math.hypot(
                    math.cos(angle) ** 2 * math.cos(2 * angle) * math.cos(phi),
                    math.cos(angle) * (1 + math.sin(angle) ** 2) * math.sin(phi),
                )
                for phi in np.arange(layer.n_wires) * 2 * math.pi / layer.n_wires
            )
        )
        allowable = library.get_property(layer.material, "sigma_y_Pa")
        mean = state.layers[layer.name].longitudinal_stress_Pa
        expected = min(expected, (allowable - abs(mean)) / slope)
    result = wire_yield_curvature(cable, 0, section_state=state, mu=0)
    assert result == pytest.approx(expected, rel=1e-11)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"p_ext_Pa": -1},
        {"p_int_Pa": np.inf},
        {"tension_N": np.nan},
        {"temperature_C": {"missing": 20}},
        {"youngs_modulus_Pa": {"inner_armour": -1}},
        {"initial_wire_strain": {"missing": 0.001}},
    ],
)
def test_invalid_normal_state_inputs_are_rejected(kwargs):
    with pytest.raises(CableSchemaError):
        cx.solve_section_state(_cable(), **kwargs)
