# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Independent exposure arithmetic and material-point alignment checks."""

import math
from dataclasses import replace

import numpy as np
import pytest

from cablexsec import (
    FatigueLoadCase,
    FatigueResult,
    Provenance,
    compute_fatigue,
    compute_long_term_fatigue,
    inter_array_single_core_yaml_path,
    load_cable,
)


def result(values=((1.0, 0.0),), angles=(0.0, 180.0)):
    array = np.array(values, dtype=float)
    return FatigueResult(
        D_bulk_per_wire={"armour": np.max(array, axis=1)},
        D_fretting_per_wire={},
        D_term_per_wire={},
        D_total_per_wire={},
        phi_critical_deg=0,
        critical_layer="armour",
        above_core=None,
        D_fretting_fraction=0,
        fibre_angles_deg=np.array(angles),
        D_bulk_per_fibre={"armour": array},
    )


def case(name="a", values=((1.0, 0.0),), **kwargs):
    return FatigueLoadCase(
        name=name,
        result=result(values),
        **(
            {"duration_s": 10.0, "probability": 1.0, "assessment_id": "model-1"}
            | kwargs
        ),
    )


def compute(cases, **kwargs):
    return compute_long_term_fatigue(
        cases, **({"exposure_years": 2.0, "seconds_per_year": 100.0} | kwargs)
    )


def test_accumulates_material_points_before_selecting_maximum():
    # Each condition peaks on a different fibre. Summing maxima gives 10,
    # but the correct accumulated maximum is 8 at fibre 180 degrees.
    a = case(values=((1.0, 0.2),), probability=0.25)
    b = case("b", ((0.0, 2.0),), probability=0.75, duration_s=20.0)
    out = compute([a, b], design_fatigue_factor=3)
    np.testing.assert_allclose(out.annual_damage_per_fibre["armour"], [[2.5, 8.0]])
    assert out.annual_damage == 8
    assert out.critical_fibre_deg == 180
    assert out.critical_wire_index == 0
    assert out.design_utilization == 48
    assert out.allowable_exposure_years == pytest.approx(1 / 24)
    assert out.case_contributions_at_critical_per_year == {"a": 0.5, "b": 7.5}


def test_seed_probabilities_and_record_durations_control_rates():
    # Identical stationary rates sampled for different durations have identical
    # aggregate damage to one representative record.
    single = compute([case()])
    ensemble = compute(
        [
            case("seed-1", probability=0.5),
            case("seed-2", ((2.0, 0.0),), duration_s=20, probability=0.5),
        ]
    )
    assert ensemble.annual_damage == single.annual_damage
    assert ensemble.design_utilization == single.design_utilization


def test_governing_layer_wire_and_contributions_are_coincident():
    a = case(values=((1.0, 0.0), (3.0, 0.0)), probability=0.5)
    b = case("b", ((0.0, 5.0), (0.0, 1.0)), probability=0.5)
    for c in (a, b):
        c.result.D_bulk_per_fibre["conductor"] = np.array([[0.1, 0.2]])
    out = compute([a, b])
    assert (out.critical_layer, out.critical_wire_index, out.critical_fibre_index) == (
        "armour",
        0,
        1,
    )
    assert out.annual_damage == 25
    assert sum(out.case_contributions_at_critical_per_year.values()) == 25
    np.testing.assert_allclose(out.annual_damage_per_wire["armour"], [25, 15])


def test_zero_damage_reports_projection_infinity():
    out = compute([case(values=((0, 0),))])
    assert out.design_utilization == 0
    assert math.isinf(out.allowable_exposure_years)


def test_results_are_detached_and_array_readonly():
    c = case()
    out = compute([c])
    c.result.D_bulk_per_fibre["armour"][:] = 0
    c.result.fibre_angles_deg[:] = 90
    assert out.annual_damage == 10
    assert out.annual_damage_per_fibre["armour"][0, 0] == 10
    assert out.fibre_angles_deg[0] == 0
    for array in (
        out.fibre_angles_deg,
        out.annual_damage_per_fibre["armour"],
        out.annual_damage_per_wire["armour"],
    ):
        with pytest.raises(ValueError):
            array[:] = 0


def test_source_limitations_propagate_and_wire_level_indicators_are_excluded():
    c = case()
    r = replace(
        c.result,
        provenance=Provenance("test", limitations=("experimental", "experimental")),
    )
    r.D_fretting_per_wire["armour"] = np.array([1000.0])
    r.D_term_per_wire["armour"] = np.array([1000.0])
    out = compute([replace(c, result=r)])
    assert out.annual_damage == 10
    assert out.provenance.limitations.count("experimental") == 1
    assert "bulk" in out.provenance.method


@pytest.mark.parametrize(
    "field", ["exposure_years", "seconds_per_year", "design_fatigue_factor"]
)
@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf])
def test_rejects_invalid_exposure_scalars(field, value):
    with pytest.raises(ValueError, match=field):
        compute([case()], **{field: value})


def test_rejects_design_factor_below_one():
    with pytest.raises(ValueError, match="at least one"):
        compute([case()], design_fatigue_factor=0.5)


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf])
def test_rejects_invalid_record_duration(value):
    with pytest.raises(ValueError, match="duration_s"):
        compute([case(duration_s=value)])


@pytest.mark.parametrize("value", [-1, 2, math.nan, math.inf])
def test_rejects_invalid_probability(value):
    with pytest.raises(ValueError, match="probability"):
        compute([case(probability=value)])


@pytest.mark.parametrize(
    "cases",
    [
        [],
        [case(probability=0.5)],
        [case(), case("b")],
        [case(assessment_id=" ")],
        [case(name=" ")],
        [case(probability=0.5), case(probability=0.5)],
    ],
)
def test_rejects_incomplete_or_ambiguous_assessments(cases):
    with pytest.raises(ValueError):
        compute(cases)


def test_rejects_different_assessment_identity():
    with pytest.raises(ValueError, match="assessment_id"):
        compute(
            [case(probability=0.5), case("b", probability=0.5, assessment_id="other")]
        )


@pytest.mark.parametrize(
    "angles",
    [[], [[0, 180]], [0, math.nan], [0, math.inf], [0, 0], [-1, 180], [0, 360]],
)
def test_rejects_invalid_fibre_locations(angles):
    c = case()
    with pytest.raises(ValueError, match="fibre angles"):
        compute(
            [replace(c, result=replace(c.result, fibre_angles_deg=np.array(angles)))]
        )


def test_rejects_permuted_material_frame():
    b = case("b", probability=0.5)
    b = replace(b, result=replace(b.result, fibre_angles_deg=np.array([180, 0])))
    with pytest.raises(ValueError, match="ordered material"):
        compute([case(probability=0.5), b])


def test_rejects_wire_maxima_without_fibres():
    c = case()
    with pytest.raises(ValueError, match="wire maxima are insufficient"):
        compute([replace(c, result=replace(c.result, D_bulk_per_fibre={}))])


def test_rejects_different_layers():
    b = case("b", probability=0.5)
    b.result.D_bulk_per_fibre["extra"] = np.array([[1, 2]])
    with pytest.raises(ValueError, match="same layers"):
        compute([case(probability=0.5), b])


@pytest.mark.parametrize(
    "values",
    [
        np.zeros(2),
        np.zeros((0, 2)),
        np.zeros((1, 3)),
        np.array([[1, -1]]),
        np.array([[1, math.nan]]),
        np.array([[1, math.inf]]),
    ],
)
def test_rejects_invalid_damage(values):
    c = case()
    c.result.D_bulk_per_fibre["armour"] = values
    with pytest.raises(ValueError, match="invalid fibre damage"):
        compute([c])


def test_rejects_changed_wire_count():
    with pytest.raises(ValueError, match="invalid fibre damage"):
        compute([case(probability=0.5), case("b", ((1, 0), (1, 0)), probability=0.5)])


def test_zero_probability_retains_no_damage():
    out = compute([case(), case("unused", ((1000, 1000),), probability=0)])
    assert out.annual_damage == 10
    assert out.case_contributions_at_critical_per_year["unused"] == 0


def test_overflow_is_explicit():
    with pytest.raises(ValueError, match="accumulation"):
        compute([case(values=((1e308, 0),))])
    with pytest.raises(ValueError, match="design utilization"):
        compute([case()], exposure_years=1e308)


def test_real_fatigue_result_roundtrip_and_default_year():
    cable = load_cable(inter_array_single_core_yaml_path())
    kappa = 0.005 * np.sin(np.linspace(0, 4 * math.pi, 81))
    fatigue = compute_fatigue(cable, kappa, np.zeros_like(kappa))
    load = FatigueLoadCase("regular", fatigue, 20, 1, "public-single-core-defaults")
    out = compute_long_term_fatigue([load], exposure_years=25)
    scale = 365.25 * 24 * 3600 / 20
    for name, array in fatigue.D_bulk_per_fibre.items():
        np.testing.assert_allclose(out.annual_damage_per_fibre[name], array * scale)
    assert out.annual_damage == pytest.approx(
        max(np.max(a) for a in fatigue.D_bulk_per_fibre.values()) * scale
    )
    assert out.design_utilization == pytest.approx(25 * out.annual_damage)
