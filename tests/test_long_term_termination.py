# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Termination-aware lifetime projection must retain a coincident material point."""

import math
from dataclasses import replace

import numpy as np
import pytest

import cablexsec as cx
from cablexsec import FatigueLoadCase, FatigueResult, compute_long_term_fatigue
from cablexsec.analysis.fatigue import FatigueLayer, compute_fatigue
from cablexsec.materials.sn_curves import SNcurve
from cablexsec.mechanics.mkappa import HelicalLayerSpec


def case(name="a", bulk=((1, 0.2),), adjusted=((3, 2),), probability=1, duration=10):
    b = np.asarray(bulk, dtype=float)
    a = np.asarray(adjusted, dtype=float)
    result = FatigueResult(
        D_bulk_per_wire={"wire": b.max(axis=1)},
        D_fretting_per_wire={"wire": np.array([1e10])},
        D_term_per_wire={"wire": np.array([1e10])},
        D_total_per_wire={"wire": np.array([1e10])},
        phi_critical_deg=0,
        critical_layer="wire",
        above_core=None,
        D_fretting_fraction=0,
        fibre_angles_deg=np.array([0, 180]),
        D_bulk_per_fibre={"wire": b},
        D_bulk_with_termination_per_fibre={"wire": a},
    )
    return FatigueLoadCase(name, result, duration, probability, "same-model-and-scf")


def project(cases, **kwargs):
    return compute_long_term_fatigue(
        cases,
        **{
            "exposure_years": 2,
            "seconds_per_year": 100,
            "damage_basis": "bulk_with_termination",
            **kwargs,
        },
    )


def test_adjusted_lifetime_hotspot_is_selected_after_material_point_accumulation():
    a = case(probability=0.4)
    b = case("b", bulk=((0.1, 1),), adjusted=((0.2, 4),), probability=0.6, duration=20)
    out = project([a, b], design_fatigue_factor=3)
    # Scales are 4 and 3. The sum of record maxima is 24; coincident maximum is 20.
    np.testing.assert_allclose(out.annual_damage_per_fibre["wire"], [[12.6, 20]])
    assert out.annual_damage == 20
    assert out.critical_fibre_deg == 180
    assert out.design_utilization == 120
    assert out.allowable_exposure_years == pytest.approx(1 / 60)
    assert out.case_contributions_at_critical_per_year == {"a": 8, "b": 12}
    assert out.damage_basis == "bulk_with_termination"
    assert "bulk_with_termination" in out.provenance.method
    assert any(
        "SCFs require independent qualification" in x
        for x in out.provenance.limitations
    )
    bulk = project([a, b], damage_basis="bulk")
    np.testing.assert_allclose(bulk.annual_damage_per_fibre["wire"], [[4.3, 3.8]])
    assert bulk.critical_fibre_deg == 0
    assert bulk.damage_basis == "bulk"


def test_default_is_unchanged_bulk_even_when_adjusted_data_are_available():
    c = case()
    out = compute_long_term_fatigue([c], exposure_years=2, seconds_per_year=100)
    assert out.annual_damage == 10
    assert out.damage_basis == "bulk"
    assert project([c]).annual_damage == 30


def test_legacy_results_require_explicit_fixed_fibre_termination_data():
    c = case()
    legacy = replace(c, result=replace(c.result, D_bulk_with_termination_per_fibre={}))
    assert project([legacy], damage_basis="bulk").annual_damage == 10
    with pytest.raises(ValueError, match="fixed-fibre.*every analysed layer"):
        project([legacy])


@pytest.mark.parametrize("basis", ["total", "fretting", "termination", "", None, True])
def test_unavailable_damage_basis_is_rejected(basis):
    with pytest.raises(ValueError, match="damage_basis"):
        project([case()], damage_basis=basis)


@pytest.mark.parametrize(
    "values",
    [
        np.zeros(2),
        np.zeros((0, 2)),
        np.zeros((2, 2)),
        np.zeros((1, 3)),
        np.array([[1, -1]]),
        np.array([[1, np.nan]]),
        np.array([[1, np.inf]]),
    ],
)
def test_invalid_adjusted_damage_cannot_fall_back_to_bulk(values):
    c = case()
    c.result.D_bulk_with_termination_per_fibre["wire"] = values
    with pytest.raises(ValueError, match="invalid fibre damage"):
        project([c])


def test_adjusted_field_requires_same_layers_in_every_record():
    a, b = case(probability=0.5), case("b", probability=0.5)
    b.result.D_bulk_with_termination_per_fibre["extra"] = np.zeros((1, 2))
    with pytest.raises(ValueError, match="every analysed layer"):
        project([a, b])


def test_each_adjusted_field_must_match_its_own_bulk_wire_layout():
    a, b = case(probability=0.5), case("b", probability=0.5)
    b.result.D_bulk_per_fibre["wire"] = np.ones((2, 2))
    with pytest.raises(ValueError, match="invalid fibre damage"):
        project([a, b])


@pytest.mark.parametrize(
    "path",
    [
        cx.inter_array_single_core_yaml_path,
        cx.inter_array_three_core_yaml_path,
        cx.export_three_core_yaml_path,
    ],
)
def test_public_constructions_retain_identity_without_termination_factors(path):
    cable = cx.load_cable(path(), geometry_override=None)
    kappa = np.array([0, 0.01, -0.01, 0])
    short = cx.compute_fatigue(cable, kappa, kappa * 0.3)
    record = FatigueLoadCase("record", short, 30, 1, "unaltered-public-input")
    bulk = project([record], damage_basis="bulk")
    adjusted = project([record])
    for name in bulk.annual_damage_per_fibre:
        np.testing.assert_array_equal(
            bulk.annual_damage_per_fibre[name], adjusted.annual_damage_per_fibre[name]
        )
    assert adjusted.design_utilization == bulk.design_utilization
    assert adjusted.critical_layer == bulk.critical_layer
    assert adjusted.critical_wire_index == bulk.critical_wire_index
    assert adjusted.critical_fibre_index == bulk.critical_fibre_index


def test_zero_probability_does_not_hide_a_missing_adjusted_field():
    a, b = case(), case("unused", probability=0)
    b.result.D_bulk_with_termination_per_fibre.clear()
    with pytest.raises(ValueError, match="fixed-fibre"):
        project([a, b])


def test_outputs_are_detached_from_adjusted_source_arrays():
    c = case()
    out = project([c])
    c.result.D_bulk_with_termination_per_fibre["wire"][:] = 0
    assert out.annual_damage == 30
    assert out.annual_damage_per_fibre["wire"][0, 0] == 30
    with pytest.raises(ValueError):
        out.annual_damage_per_fibre["wire"][:] = 0


def test_explicit_range_reduction_can_reduce_projected_damage_to_zero():
    out = project([case(adjusted=((0, 0),))])
    assert out.annual_damage == 0
    assert out.design_utilization == 0
    assert math.isinf(out.allowable_exposure_years)


@pytest.mark.parametrize("bilinear", [False, True])
@pytest.mark.parametrize("scf", [0.5, 1, 2])
def test_actual_short_term_cycles_match_independent_lifetime_arithmetic(bilinear, scf):
    curve = (
        SNcurve("synthetic-bilinear", "MPa", 3, 12, m2=5, log_C2=16)
        if bilinear
        else SNcurve("synthetic-cutoff", "MPa", 3, 12, threshold=50)
    )
    layer = FatigueLayer(
        spec=HelicalLayerSpec(
            n_wires=1,
            r_helix_m=0.05,
            lay_angle_rad=0,
            E_wire_Pa=200e9,
            A_wire_m2=math.pi * 0.002**2,
            xi=1,
            mu=0,
            F_n_axi_N=0,
        ),
        name="wire",
        sn_bulk=curve,
        sigma_UTS_Pa=1e9,
        SCF=scf,
        termination=True,
    )
    cases = []
    expected = 0
    for name, span, p, duration in (("a", 40, 0.25, 10), ("b", 70, 0.75, 20)):
        short = compute_fatigue(
            [layer],
            np.zeros(3),
            np.zeros(3),
            normal_stress_Pa={"wire": np.array([0, span * 1e6, 0])},
            n_fibres=4,
        )
        cases.append(FatigueLoadCase(name, short, duration, p, "same-model-and-scf"))
        amplified = span * scf
        if bilinear:
            damage = amplified**5 / 1e16 if amplified < 100 else amplified**3 / 1e12
        else:
            damage = 0 if amplified < 50 else amplified**3 / 1e12
        expected += p * 100 / duration * damage
    out = project(cases)
    np.testing.assert_allclose(
        out.annual_damage_per_fibre["wire"], expected, rtol=1e-12
    )
    assert out.design_utilization == pytest.approx(expected * 2)
