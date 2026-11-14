# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Supplied fatigue loads must be used, or rejected with an explicit reason."""

import math
from dataclasses import replace

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.fatigue import FatigueLayer, compute_fatigue, wire_stress_field
from cablexsec.io.series import LoadCaseSeries, fatigue_from_series
from cablexsec.materials.library import MATERIALS
from cablexsec.materials.sn_curves import SN_CURVES, SNcurve
from cablexsec.mechanics.mkappa import HelicalLayerSpec


def layer(name="wire"):
    return FatigueLayer(
        spec=HelicalLayerSpec(
            n_wires=4,
            r_helix_m=0.05,
            lay_angle_rad=0.3,
            E_wire_Pa=200e9,
            A_wire_m2=1e-5,
            xi=1,
            mu=0,
            F_n_axi_N=0,
        ),
        name=name,
        sn_bulk=SNcurve("analytical-test-only", "MPa", 3, 12),
        sigma_UTS_Pa=1e9,
    )


@pytest.mark.parametrize("ea", [None, 0, -1, np.nan, np.inf])
@pytest.mark.parametrize("entry", ["fatigue", "stress_field"])
def test_tension_cannot_be_ignored_without_valid_stiffness(ea, entry):
    sample = layer()
    function = compute_fatigue if entry == "fatigue" else wire_stress_field
    first = [sample] if entry == "fatigue" else sample
    with pytest.raises(ValueError, match="EA_full_N"):
        function(
            first,
            np.zeros(3),
            np.zeros(3),
            tension_N=np.array([0, 1e5, 0]),
            EA_full_N=ea,
        )


def test_axial_and_thermal_cycles_match_independent_linear_stress_and_miner():
    sample = layer()
    tension = np.array([0, 1e5, 0, 1e5, 0])
    thermal = np.array([0, -10e6, 0, -10e6, 0])
    ea = 2e8
    expected_range = 200e9 * math.cos(0.3) ** 2 * 1e5 / ea - 10e6
    expected_damage = 2 * (expected_range / 1e6) ** 3 / 1e12
    result = compute_fatigue(
        [sample],
        np.zeros(5),
        np.zeros(5),
        tension_N=tension,
        EA_full_N=ea,
        thermal_stress_Pa={"wire": thermal},
    )
    np.testing.assert_allclose(result.D_bulk_per_fibre["wire"], expected_damage)
    field = wire_stress_field(
        sample,
        np.zeros(5),
        np.zeros(5),
        tension_N=tension,
        EA_full_N=ea,
        thermal_stress_Pa=thermal,
    )
    expected_stress = expected_range * np.array([0, 1, 0, 1, 0])
    np.testing.assert_allclose(field.at_fibre(0), np.tile(expected_stress, (4, 1)))


@pytest.mark.parametrize(
    "name", ["thermal_stress_Pa", "contact_pressure_Pa", "contact_force_N"]
)
def test_misspelled_history_layer_is_not_silently_ignored(name):
    with pytest.raises(ValueError, match="unknown.*wirre"):
        compute_fatigue(
            [layer()], np.zeros(3), np.zeros(3), **{name: {"wirre": np.ones(3)}}
        )


def test_partial_thermal_mapping_applies_only_to_named_layers():
    result = compute_fatigue(
        [layer("hot"), layer("cold")],
        np.zeros(3),
        np.zeros(3),
        thermal_stress_Pa={"hot": np.array([0, 20e6, 0])},
    )
    np.testing.assert_allclose(result.D_bulk_per_wire["hot"], 20**3 / 1e12)
    np.testing.assert_array_equal(result.D_bulk_per_wire["cold"], 0)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("entry", ["fatigue", "stress_field"])
def test_nonfinite_tension_is_rejected_before_stress_recovery(bad, entry):
    function = compute_fatigue if entry == "fatigue" else wire_stress_field
    first = [layer()] if entry == "fatigue" else layer()
    with pytest.raises(ValueError, match="tension_N must be finite"):
        function(
            first,
            np.zeros(3),
            np.zeros(3),
            tension_N=np.array([0, bad, 0]),
            EA_full_N=2e8,
        )


@pytest.mark.parametrize("history", [np.zeros(2), np.array([0, np.nan, 0])])
def test_stress_field_rejects_invalid_thermal_history(history):
    with pytest.raises(ValueError, match="thermal stress"):
        wire_stress_field(layer(), np.zeros(3), np.zeros(3), thermal_stress_Pa=history)


def test_constant_contact_history_must_match_its_hertz_reference():
    sample = replace(
        layer(),
        sn_fretting=SNcurve("synthetic", "um", 3, 12),
        K_t_contact_N_per_m=1e7,
        crossing_density_per_m=20,
        contact_line_load_N_per_m=100,
    )
    with pytest.raises(ValueError, match="matching the supplied Hertz"):
        compute_fatigue(
            [sample],
            np.zeros(3),
            np.zeros(3),
            contact_force_N={"wire": np.full(3, 400)},
        )


def test_duplicate_layer_names_cannot_overwrite_damage():
    with pytest.raises(ValueError, match="unique"):
        compute_fatigue([layer(), layer()], np.zeros(3), np.zeros(3))


@pytest.mark.parametrize("name", ["contact_pressure_Pa", "contact_force_N"])
def test_negative_contact_history_is_rejected_even_in_bulk_only_assessment(name):
    with pytest.raises(ValueError, match="nonnegative"):
        compute_fatigue(
            [layer()],
            np.zeros(3),
            np.zeros(3),
            **{name: {"wire": np.array([0, -1, 0])}},
        )


@pytest.mark.parametrize("law", ["synthetic-slip", "copper_poon2022"])
@pytest.mark.parametrize("starts_open", [False, True])
def test_public_variable_tension_cannot_freeze_initial_hertz_properties(
    monkeypatch, law, starts_open
):
    monkeypatch.setitem(SN_CURVES, "synthetic-slip", SNcurve("synthetic", "um", 3, 12))
    # A fully specified test-only material exercises the SWT routing. This is
    # not a qualified copper construction or a new registered engineering material.
    monkeypatch.setitem(
        MATERIALS,
        "test_contact_material",
        {**MATERIALS["galvanised_steel_armour"], "mu_on_cu": (0.25, "test only")},
    )
    original = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    cable = original.model_copy(
        update={
            "layers": tuple(
                item.model_copy(
                    update={
                        "winding_stress_Pa": 0,
                        "pretension_Pa": 0,
                        "fretting_sn": law,
                        "material": "test_contact_material"
                        if law == "copper_poon2022"
                        else item.material,
                    }
                )
                if item.name in {"inner_armour", "outer_armour"}
                else item
                for item in original.layers
            )
        }
    )
    tension = np.array([0 if starts_open else 1e4, 1e5, 1e4])
    with pytest.raises(ValueError, match="constant.*contact|constant.*normal"):
        cx.compute_fatigue(
            cable, np.array([0, 0.01, 0]), np.zeros(3), tension_N=tension
        )


def test_kernel_variable_contact_life_cannot_freeze_hertz_stiffness():
    sample = replace(
        layer(),
        sn_fretting=SNcurve("synthetic", "um", 3, 12),
        K_t_contact_N_per_m=1e7,
        crossing_density_per_m=20,
        contact_line_load_N_per_m=100,
        fretting_mu=0.2,
    )
    with pytest.raises(ValueError, match="constant.*contact|constant.*normal"):
        compute_fatigue(
            [sample],
            np.array([0, 0.01, 0]),
            np.zeros(3),
            contact_force_N={"wire": np.array([100, 400, 100])},
        )


@pytest.mark.parametrize("frozen", [None, 0.0, 8e4])
def test_series_and_direct_api_use_the_same_tension_contact_history(frozen):
    original = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    cable = original.model_copy(
        update={
            "layers": tuple(
                item.model_copy(update={"winding_stress_Pa": 0, "pretension_Pa": 0})
                if item.name in {"inner_armour", "outer_armour"}
                else item
                for item in original.layers
            )
        }
    )
    kappa = np.array([0, 0.02, -0.02, 0.02, 0])
    tension = np.array([1e4, 2e5, 1e5, 3e5, 1e4])
    series = LoadCaseSeries(
        time_s=np.arange(5, dtype=float),
        curvature_x_1_m=kappa,
        curvature_y_1_m=kappa * 0.3,
        tension_N=tension,
        tension_kind="true",
    )
    kwargs = {} if frozen is None else {"contact_tension_N": frozen}
    actual = fatigue_from_series(cable, series, **kwargs)
    expected = cx.compute_fatigue(
        cable, kappa, kappa * 0.3, tension_N=tension, **kwargs
    )
    for name in actual.D_bulk_per_fibre:
        np.testing.assert_array_equal(
            actual.D_bulk_per_fibre[name], expected.D_bulk_per_fibre[name]
        )
    if frozen is None:
        zero_reference = cx.compute_fatigue(
            cable, kappa, kappa * 0.3, tension_N=tension, contact_tension_N=0
        )
        assert any(
            not np.array_equal(
                expected.D_bulk_per_fibre[name], zero_reference.D_bulk_per_fibre[name]
            )
            for name in expected.D_bulk_per_fibre
        )
