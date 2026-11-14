# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Analytical pressure-chain checks with selectively structural wire layers."""

import math

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.cable_adapters import (
    armour_contact_pressure,
    armour_crossing_line_loads,
    cable_to_mkappa_specs,
)
from cablexsec.cables.schema import (
    CableData,
    CableMeta,
    HelicalLayer,
    InitialStateProperties,
)
from cablexsec.mechanics.axisymmetric import compute_winding_contact_pressure


def cable_with_wires(inactive=-1, winding=(0.0, 0.0, 0.0), external=0.0):
    layers = tuple(
        HelicalLayer(
            name=f"wire-{i}",
            material="galvanised_steel_armour",
            n_wires=12 * (i + 1),
            d_wire_m=0.002,
            r_helix_m=0.02 + 0.005 * i,
            lay_angle_deg=20 + 10 * i,
            structural=i != inactive,
            winding_stress_Pa=winding[i],
        )
        for i in range(3)
    )
    return CableData(
        cable=CableMeta(name="synthetic pressure chain"),
        layers=layers,
        initial_state=InitialStateProperties(
            external_pressure_Pa=external,
            identification="measured" if external else "none",
            source="synthetic verification boundary" if external else "",
        ),
    )


def expected_pressures(cable, tension=0.0, external=0.0):
    # All wires carry axial load in the declared fixed-radius model. The
    # structural flag selects bending/fatigue sites, not the pressure-chain rows.
    wires = cable.helical_layers
    weights = np.array([x.n_wires * math.cos(x.lay_angle_rad) ** 3 for x in wires])
    forces = tension * weights / weights.sum()
    jumps = np.array(
        [
            force * math.tan(x.lay_angle_rad) ** 2 / (2 * math.pi * x.r_helix_m**2)
            + x.n_wires
            * (math.pi * x.d_wire_m**2 / 4)
            * x.winding_stress_Pa
            * math.sin(x.lay_angle_rad)
            * math.tan(x.lay_angle_rad)
            / (2 * math.pi * x.r_helix_m**2)
            for x, force in zip(wires, forces, strict=True)
        ]
    )
    return {
        x.name: 2 * external + sum(jumps[i:]) + sum(jumps[i + 1 :])
        for i, x in enumerate(wires)
        if x.structural
    }


@pytest.mark.parametrize("inactive", [0, 1, 2])
@pytest.mark.parametrize("saved_state", [False, True])
def test_tension_pressure_faces_follow_full_radial_order(inactive, saved_state):
    cable = cable_with_wires(inactive=inactive, external=2e5)
    kwargs = {}
    if saved_state:
        kwargs["section_state"] = cx.solve_section_state(
            cable, tension_N=1e4, contact_model="bounded"
        )
    result = armour_contact_pressure(cable, 1e4, **kwargs)
    expected = expected_pressures(cable, tension=1e4, external=2e5)
    assert result.keys() == expected.keys()
    for name in expected:
        assert result[name] == pytest.approx(expected[name], rel=1e-12)


def test_winding_stress_applies_only_to_declared_layers():
    cable = cable_with_wires(winding=(0, 0, 50e6))
    result = armour_contact_pressure(cable)
    for name, expected in expected_pressures(cable).items():
        assert result[name] == pytest.approx(expected, rel=1e-12)


def test_different_winding_stresses_have_separate_pressure_jumps():
    cable = cable_with_wires(winding=(10e6, 30e6, 50e6))
    result = armour_contact_pressure(cable)
    for name, expected in expected_pressures(cable).items():
        assert result[name] == pytest.approx(expected, rel=1e-12)


def test_nonstructural_outer_wire_winding_still_clamps_inner_armour():
    cable = cable_with_wires(inactive=2, winding=(0, 0, 50e6))
    result = armour_contact_pressure(cable)
    for name, expected in expected_pressures(cable).items():
        assert result[name] == pytest.approx(expected, rel=1e-12)


def test_uniform_winding_preserves_existing_scalar_helper_result():
    cable = cable_with_wires(winding=(50e6, 50e6, 50e6))
    assert armour_contact_pressure(cable) == compute_winding_contact_pressure(
        cable, 50e6
    )


def touching_wires(inactive=-1):
    source = cable_with_wires(inactive=inactive)
    layers = tuple(
        x.model_copy(
            update={
                "r_helix_m": 0.02 + 0.002 * i,
                "lay_direction": "left_hand" if i == 1 else "right_hand",
            }
        )
        for i, x in enumerate(source.layers)
    )
    return source.model_copy(update={"layers": layers})


def test_nonstructural_touching_wire_remains_a_contact_partner():
    both = touching_wires()
    both = both.model_copy(update={"layers": both.layers[:2]})
    reference = armour_crossing_line_loads(both, 1e4)
    changed = both.model_copy(
        update={
            "layers": (
                both.layers[0].model_copy(update={"structural": False}),
                both.layers[1],
            )
        }
    )
    result = armour_crossing_line_loads(changed, 1e4)
    assert set(result) == {"wire-1"}
    partner, force = result["wire-1"]
    assert partner.name == "wire-0" and not partner.structural
    assert force == pytest.approx(reference["wire-1"][1], rel=1e-12)


def test_nonstructural_middle_wire_can_support_two_separate_active_partners():
    cable = touching_wires(inactive=1)
    result = armour_crossing_line_loads(cable, 1e4)
    assert set(result) == {"wire-0", "wire-2"}
    assert all(partner.name == "wire-1" for partner, force in result.values())
    assert all(force > 0 for partner, force in result.values())


def test_active_wire_with_two_crossing_interfaces_requires_separate_histories():
    with pytest.raises(
        cx.CableSchemaError, match="multiple direct crossing interfaces"
    ):
        armour_crossing_line_loads(touching_wires(), 1e4)


def test_outer_inactive_winding_reaches_the_active_crossing_interface():
    cable = touching_wires(inactive=2)
    outer = cable.layers[2].model_copy(
        update={"r_helix_m": 0.028, "winding_stress_Pa": 50e6}
    )
    cable = cable.model_copy(update={"layers": (*cable.layers[:2], outer)})
    result = armour_crossing_line_loads(cable)
    jump = (
        outer.n_wires
        * (math.pi * outer.d_wire_m**2 / 4)
        * 50e6
        * math.sin(outer.lay_angle_rad)
        * math.tan(outer.lay_angle_rad)
        / (2 * math.pi * outer.r_helix_m**2)
    )
    reaction = 2 * math.pi * 0.021 * jump
    for layer in cable.layers[:2]:
        assert result[layer.name][1] == pytest.approx(
            reaction / layer.n_wires, rel=1e-12
        )


def test_winding_mapping_omitted_layers_are_unwound():
    cable = cable_with_wires(winding=(0, 0, 50e6))
    actual = compute_winding_contact_pressure(cable, {"wire-2": 50e6}, p_ext_Pa=2e5)
    for name, expected in expected_pressures(cable, external=2e5).items():
        assert actual[name] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    "winding",
    [-1, math.nan, math.inf, {"wire-1": -1}, {"wire-1": math.nan}, {"typo": 10e6}],
)
def test_invalid_winding_inputs_are_rejected(winding):
    with pytest.raises(ValueError, match="winding"):
        compute_winding_contact_pressure(cable_with_wires(), winding)


@pytest.mark.parametrize("external", [-1, math.nan, math.inf])
def test_invalid_winding_boundary_pressure_is_rejected(external):
    with pytest.raises(ValueError, match="external pressure"):
        compute_winding_contact_pressure(cable_with_wires(), 10e6, p_ext_Pa=external)


@pytest.mark.parametrize("model", ["bounded", "qp"])
@pytest.mark.parametrize("inactive", ["inner_armour", "outer_armour"])
@pytest.mark.parametrize("saved_state", [False, True])
def test_partner_selection_preserves_solved_interface_reaction(
    model, inactive, saved_state
):
    cable = cx.load_cable(cx.inter_array_single_core_yaml_path())
    reference = armour_crossing_line_loads(cable, 1e4, contact_model=model)
    modified = cable.model_copy(
        update={
            "layers": tuple(
                layer.model_copy(update={"structural": False})
                if layer.name == inactive
                else layer
                for layer in cable.layers
            )
        }
    )
    kwargs = {"contact_model": model}
    if saved_state:
        kwargs["section_state"] = cx.solve_section_state(
            modified, tension_N=1e4, contact_model=model
        )
    recovered = armour_crossing_line_loads(modified, 1e4, **kwargs)
    assert set(recovered) == set(reference) - {inactive}
    for name, (partner, load) in recovered.items():
        assert partner.name == inactive
        assert load == pytest.approx(reference[name][1], rel=1e-11)


@pytest.mark.parametrize(
    "adapter",
    [armour_contact_pressure, armour_crossing_line_loads, cable_to_mkappa_specs],
)
def test_contact_recovery_rejects_mixed_solver_models(adapter):
    cable = cx.load_cable(cx.inter_array_single_core_yaml_path())
    state = cx.solve_section_state(cable, tension_N=1e4, contact_model="bounded")
    with pytest.raises(cx.CableSchemaError, match="contact model conflicts"):
        adapter(cable, 1e4, section_state=state, contact_model="qp")


def test_per_layer_winding_is_added_to_qp_face_pressures_without_resolving_prestrain():
    cable = cx.load_cable(cx.inter_array_single_core_yaml_path())
    # This remains the declared additive manufacturing-pressure approximation.
    # Changing winding inputs must not silently change the normal-state solution.
    modified = cable.model_copy(
        update={
            "layers": tuple(
                layer.model_copy(
                    update={
                        "winding_stress_Pa": 10e6
                        if layer.name == "inner_armour"
                        else 30e6
                    }
                )
                if isinstance(layer, HelicalLayer)
                else layer
                for layer in cable.layers
            )
        }
    )
    state = cx.solve_section_state(cable, tension_N=1e4, p_ext_Pa=2e5)
    other = cx.solve_section_state(modified, tension_N=1e4, p_ext_Pa=2e5)
    np.testing.assert_array_equal(
        state.radial.contact_pressure_Pa, other.radial.contact_pressure_Pa
    )
    pressure = armour_contact_pressure(
        modified, contact_model="qp", section_state=other
    )
    winding = compute_winding_contact_pressure(
        modified, {"inner_armour": 10e6, "outer_armour": 30e6}
    )
    names = list(other.radial.section.outer_displacement_m)
    for layer in modified.helical_layers:
        index = names.index(layer.name)
        below = other.radial.contact_pressure_Pa[index - 1]
        above = (
            other.radial.contact_pressure_Pa[index] if index < len(names) - 1 else 2e5
        )
        assert pressure[layer.name] == pytest.approx(
            winding[layer.name] + below + above
        )


def test_public_stress_retains_contact_shear_from_an_inactive_partner():
    cable = cx.load_cable(cx.inter_array_single_core_yaml_path())
    modified = cable.model_copy(
        update={
            "layers": tuple(
                layer.model_copy(update={"structural": False})
                if layer.name == "inner_armour"
                else layer
                for layer in cable.layers
            )
        }
    )
    expected = cx.compute_stress(cable, 0.0, tension_N=1e4)
    actual = cx.compute_stress(modified, 0.0, tension_N=1e4)
    name = "outer_armour"
    np.testing.assert_array_equal(actual.sigma_vm_Pa[name], expected.sigma_vm_Pa[name])
    assert np.all(actual.sigma_vm_Pa[name] > np.abs(actual.sigma_total_Pa[name]))


def test_requested_fretting_law_receives_the_inactive_partners_hertz_contact(
    monkeypatch,
):
    from cablexsec.materials.sn_curves import SN_CURVES, SNcurve

    monkeypatch.setitem(
        SN_CURVES, "synthetic-slip-curve", SNcurve("synthetic-slip-curve", "um", 3, 8)
    )
    original = cx.load_cable(cx.inter_array_single_core_yaml_path())
    cable = original.model_copy(
        update={
            "layers": tuple(
                layer.model_copy(update={"fretting_sn": "synthetic-slip-curve"})
                if layer.name == "outer_armour"
                else layer
                for layer in original.layers
            )
        }
    )
    modified = cable.model_copy(
        update={
            "layers": tuple(
                layer.model_copy(update={"structural": False})
                if layer.name == "inner_armour"
                else layer
                for layer in cable.layers
            )
        }
    )
    reference = {
        x.name: x for x in cx.cable_to_fatigue_layers(cable, contact_tension_N=1e4)
    }
    recovered = cx.cable_to_fatigue_layers(modified, contact_tension_N=1e4)
    assert len(recovered) == 1
    actual = recovered[0]
    expected = reference[actual.name]
    assert actual.K_t_contact_N_per_m == expected.K_t_contact_N_per_m > 0
    assert actual.contact_p0_Pa == expected.contact_p0_Pa > 0
    assert actual.crossing_density_per_m == expected.crossing_density_per_m > 0
