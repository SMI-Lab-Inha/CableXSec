# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Adapter compatibility, pressure limits and unsupported radial geometries."""

import math

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.api import _contact_histories, _papailiou_slip_fraction
from cablexsec.cable_adapters import (
    armour_contact_pressure,
    armour_normal_force,
    cable_to_papailiou_layers,
    mechanical_equivalent,
    structural_armour_layers,
)
from cablexsec.cables.schema import (
    CableData,
    CableMeta,
    CylindricalLayer,
    CylindricalOffsetLayer,
    CylindricalShellLayer,
    HelicalLayer,
    HelicalStrandedLayer,
    InitialStateProperties,
)
from cablexsec.mechanics.contact_qp import (
    radial_foundation_modulus,
    surface_compliances,
)


def cable_with(*layers):
    return CableData(cable=CableMeta(name="public synthetic fixture"), layers=layers)


@pytest.mark.parametrize("mu", [None, 0.0, 0.3])
def test_slip_front_adapter_preserves_pressure_and_zero_history(mu):
    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    layers = cable_to_papailiou_layers(cable, mu=mu)
    assert len(layers) == 2
    pressure = armour_contact_pressure(cable)
    for layer, value in zip(layers, pressure.values(), strict=True):
        assert layer.contact_pressure_Pa == value
    np.testing.assert_array_equal(
        _papailiou_slip_fraction(layers, np.zeros(3)), np.zeros(3)
    )


def test_unclamped_adapter_has_no_friction_layers():
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    layers = tuple(
        layer.model_copy(update={"pretension_Pa": 0, "winding_stress_Pa": 0})
        if isinstance(layer, HelicalLayer)
        else layer
        for layer in cable.layers
    )
    assert (
        cable_to_papailiou_layers(
            cable.model_copy(
                update={"layers": layers, "initial_state": InitialStateProperties()}
            )
        )
        == []
    )
    np.testing.assert_array_equal(
        _papailiou_slip_fraction([], np.zeros(2)), np.zeros(2)
    )


def test_qp_history_matches_individual_states_and_handles_empty_history():
    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    history = np.array([0, 1e4, 0])
    pressure, force = _contact_histories(cable, history, contact_model="qp")
    for name in pressure:
        np.testing.assert_allclose(
            pressure[name],
            [
                armour_contact_pressure(cable, t, contact_model="qp")[name]
                for t in history
            ],
        )
        assert force[name][0] == force[name][2]
    assert _contact_histories(cable, np.array([]), contact_model="qp") == ({}, {})


@pytest.mark.parametrize("function", [armour_contact_pressure, armour_normal_force])
def test_unknown_contact_law_rejected(function):
    with pytest.raises(cx.CableSchemaError, match="contact_model"):
        function(cable_with(), contact_model="typo")


def test_structural_armour_filters_incomplete_nonstructural_and_unknown_material():
    complete = HelicalLayer(
        name="valid",
        material="galvanised_steel_armour",
        n_wires=8,
        d_wire_m=0.001,
        r_helix_m=0.01,
        lay_angle_deg=20,
    )
    # Distinct radii: layers at one radius would overlap and are refused.
    layers = [
        complete,
        complete.model_copy(
            update={"name": "inactive", "structural": False, "r_helix_m": 0.012}
        ),
        complete.model_copy(
            update={"name": "incomplete", "n_wires": None, "r_helix_m": 0.014}
        ),
        complete.model_copy(
            update={"name": "unknown", "material": "missing", "r_helix_m": 0.016}
        ),
    ]
    assert structural_armour_layers(cable_with(*layers)) == [complete]
    cable = cable_with(complete)
    assert mechanical_equivalent(cable) is cable


@pytest.mark.parametrize(
    "layer",
    [
        HelicalLayer(name="wire", material="galvanised_steel_armour"),
        HelicalLayer(
            name="straight",
            material="galvanised_steel_armour",
            n_wires=8,
            d_wire_m=0.001,
            r_helix_m=0.01,
            lay_angle_deg=0,
        ),
        CylindricalLayer(name="missing", material="hdpe"),
        CylindricalShellLayer(name="shell", material="hdpe"),
    ],
)
def test_radial_stiffness_rejects_incomplete_or_unsupported_geometry(layer):
    with pytest.raises(cx.CableSchemaError):
        radial_foundation_modulus(layer)


def test_solid_surface_compliance_has_no_bore():
    from cablexsec.materials.library import get_property

    solid = CylindricalLayer(
        name="solid", material="galvanised_steel_armour", r_inner_m=0, r_outer_m=0.01
    )
    inner, outer, cross = surface_compliances(solid)
    assert inner == cross == 0
    assert outer == pytest.approx(
        (1 - get_property(solid.material, "nu"))
        * 0.01
        / get_property(solid.material, "E_Pa")
    )
    with pytest.raises(cx.CableSchemaError, match="complete"):
        surface_compliances(solid.model_copy(update={"r_outer_m": None}))


@pytest.mark.parametrize(
    "layer,match",
    [
        (
            CylindricalOffsetLayer(
                name="porous",
                material="copper_stranded",
                n_cores=1,
                r_offset_m=0,
                r_inner_m=0,
                r_outer_m=0.01,
                conductor_area_mm2=10,
            ),
            "metal area",
        ),
        (
            CylindricalOffsetLayer(
                name="strands",
                material="copper_stranded",
                n_cores=1,
                r_offset_m=0,
                r_inner_m=0,
                r_outer_m=0.01,
                strand_lay_angle_deg=15,
            ),
            "strand bundle",
        ),
        (
            CylindricalShellLayer(
                name="shell",
                material="galvanised_steel_armour",
                r_mid_m=0.01,
                t_m=0.001,
                f_corrugation=0.8,
            ),
            "corrugated",
        ),
        (
            CylindricalShellLayer(name="shell", material="galvanised_steel_armour"),
            "incomplete shell",
        ),
        (
            CylindricalLayer(name="annulus", material="galvanised_steel_armour"),
            "incomplete annulus",
        ),
        (
            HelicalStrandedLayer(name="nested", material="galvanised_steel_armour"),
            "unsupported",
        ),
    ],
)
def test_coupled_contact_rejects_unsupported_layers(layer, match):
    with pytest.raises(cx.CableSchemaError, match=match):
        cx.solve_contact_qp(cable_with(layer), 0)


def test_smooth_shell_concentric_adapter_matches_annulus():
    shell = CylindricalShellLayer(
        name="shell", material="galvanised_steel_armour", r_mid_m=0.01, t_m=0.001
    )
    cable = cable_with(shell)
    equivalent = mechanical_equivalent(cable)
    a = cx.solve_contact_qp(cable, 1000).section
    b = cx.solve_contact_qp(equivalent, 1000).section
    assert a.axial_strain == pytest.approx(b.axial_strain)
    assert a.axial_resultant_N == pytest.approx(1000)
    assert math.isfinite(a.axial_strain)
