# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Invalid curve definitions must not become plausible fatigue lives or damage."""

import math
from dataclasses import replace
from decimal import Decimal, localcontext

import numpy as np
import pytest

from cablexsec.analysis.fatigue import FatigueLayer, compute_fatigue, miner_damage
from cablexsec.materials.sn_curves import SNcurve, get_sn_curve
from cablexsec.mechanics.mkappa import HelicalLayerSpec


def curve(**changes):
    parameters = {"name": "synthetic", "stress_unit": "MPa", "m1": 3.0, "log_C1": 12.0}
    return SNcurve(**(parameters | changes))


def layer(**changes):
    parameters = {
        "spec": HelicalLayerSpec(2, 0.03, 0.3, 200e9, 1e-5, 1.0, 0.2, 10.0),
        "sn_bulk": curve(),
        "sigma_UTS_Pa": 1e9,
        "name": "wire",
    }
    return FatigueLayer(**(parameters | changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"m2": 5},
        {"log_C2": 16},
        {"knee_stress": 100},
        {"m2": 3, "log_C2": 16},
        {"m2": 3, "log_C2": 12},
        {"m1": 0},
        {"m1": -2},
        {"m2": 0, "log_C2": 16},
        {"m2": -1, "log_C2": 16},
        {"threshold": -1},
        {"m2": 5, "log_C2": 16, "knee_stress": 0},
        {"m2": 5, "log_C2": 16, "knee_stress": -1},
        {"stress_unit": "Pa"},
        {"stress_unit": ""},
        {"name": "  "},
        {"m1": True},
        {"m1": "3"},
        {"m1": 3 + 2j},
        {"m1": None},
        {"log_C1": None},
        {"R_ratio": None},
        {"m1": 10**1000},
        {"m2": 4, "log_C2": 1000},
        {"m2": 4, "log_C2": -1000},
    ],
)
def test_invalid_curve_definition_is_rejected_at_construction(changes):
    with pytest.raises(ValueError):
        curve(**changes)


@pytest.mark.parametrize(
    "field", ["m1", "log_C1", "m2", "log_C2", "threshold", "knee_stress", "R_ratio"]
)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_curve_parameters_rejected(field, value):
    with pytest.raises(ValueError):
        curve(**({"m2": 5.0, "log_C2": 16.0} | {field: value}))


@pytest.mark.parametrize(
    "stress", [np.nan, np.inf, -np.inf, 1 + 2j, np.complex64(1 + 2j), True, "100"]
)
def test_invalid_stress_cannot_be_runout_or_nan(stress):
    with pytest.raises(ValueError):
        curve(threshold=50).cycles_to_failure(stress)


@pytest.mark.parametrize("log_c", [-400.0, 400.0])
def test_unrepresentable_positive_life_is_an_error(log_c):
    with pytest.raises(ValueError, match="floating-point"):
        curve(log_C1=log_c).cycles_to_failure(1.0)


@pytest.mark.parametrize("stress", [1e-20, 1e20])
def test_log_life_arithmetic_overflow_cannot_bypass_the_guard(stress):
    with pytest.raises(ValueError, match="floating-point"):
        curve(m1=1e308).cycles_to_failure(stress)


@pytest.mark.parametrize("log_c", [-320.0, -300.0, -1.0, 0.0, 300.0])
def test_representable_life_including_fractional_and_subnormal(log_c):
    with localcontext() as context:
        context.prec = 80
        expected = float(Decimal(10) ** Decimal(str(log_c)))
    assert curve(log_C1=log_c).cycles_to_failure(1) == expected


def test_numpy_real_parameters_and_zero_cutoff_are_valid():
    sample = curve(m1=np.float32(3), log_C1=np.int64(12), threshold=0)
    assert sample.cycles_to_failure(np.float64(100)) == 1e6
    assert sample.cycles_to_failure(0) == math.inf
    assert sample.cycles_to_failure(-1) == math.inf


def test_explicit_knee_can_define_distinct_parallel_branches():
    # An explicit switch is meaningful even if continuity cannot define it.
    sample = curve(m2=3, log_C2=13, knee_stress=100)
    assert sample.transition_stress() == 100
    assert sample.cycles_to_failure(100) == 1e6
    assert sample.cycles_to_failure(10) == 1e10


def test_explicit_knee_and_cutoff_select_correct_branches():
    sample = curve(m2=5, log_C2=16, knee_stress=100, threshold=40)
    assert sample.cycles_to_failure(40) == math.inf
    assert sample.cycles_to_failure(50) == pytest.approx(1e16 / 50**5)
    assert sample.cycles_to_failure(100) == pytest.approx(1e12 / 100**3)
    assert sample.cycles_to_failure(200) == pytest.approx(1e12 / 200**3)


@pytest.mark.parametrize("history", [[], [0, 0], [0, 1e6, 0]])
def test_miner_requires_stress_curve_even_for_zero_damage(history):
    with pytest.raises(ValueError, match="MPa"):
        miner_damage(np.array(history), curve(stress_unit="um"), 1e9)


@pytest.mark.parametrize(
    "changes,unit",
    [
        ({"sn_bulk": curve(stress_unit="um")}, "MPa"),
        ({"sn_fretting": curve()}, "um"),
    ],
)
def test_direct_fatigue_layer_rejects_wrong_curve_units(changes, unit):
    with pytest.raises(ValueError, match=unit):
        layer(**changes)


def test_accumulated_miner_damage_cannot_overflow_silently():
    # Three complete unit-range cycles, each contributing 1e308 damage.
    values = np.array([0, 1e6, 0, 1e6, 0, 1e6, 0])
    with pytest.raises(ValueError, match="damage.*floating-point"):
        miner_damage(values, curve(log_C1=-308), 1e9)


def test_underflowed_stress_range_cannot_become_zero_damage():
    with pytest.raises(ValueError, match="stress range.*floating-point"):
        miner_damage(np.array([0, np.nextafter(0.0, 1.0)]), curve(), 1e9)


@pytest.mark.parametrize("log_c", [-400, -310])
def test_unrepresentable_slip_life_or_damage_cannot_be_returned(log_c):
    specimen = layer(
        sn_fretting=curve(stress_unit="um", log_C1=log_c),
        K_t_contact_N_per_m=1e7,
        fretting_mu=0.2,
        crossing_density_per_m=20,
        contact_line_load_N_per_m=400,
        transverse_ratio=1,
        transverse_map="legacy",
    )
    curvature = np.array([0, 0.05, -0.05, 0.05, -0.05])
    with pytest.raises(ValueError, match="floating-point"):
        compute_fatigue([specimen], curvature, np.zeros_like(curvature), n_fibres=4)


def test_contact_fraction_stays_finite_when_sum_over_wires_overflows(monkeypatch):
    from cablexsec.analysis import fatigue

    specimen = layer(
        sn_bulk=curve(log_C1=-307),
        sn_fretting=curve(stress_unit="um", log_C1=-307),
        K_t_contact_N_per_m=1e7,
        fretting_mu=0.2,
        crossing_density_per_m=20,
        contact_line_load_N_per_m=400,
    )
    specimen = replace(specimen, spec=replace(specimen.spec, n_wires=10))
    monkeypatch.setattr(fatigue, "_fretting_amplitudes_um", lambda *args: np.ones(10))
    result = compute_fatigue(
        [specimen],
        np.zeros(3),
        np.zeros(3),
        n_fibres=4,
        normal_stress_Pa={"wire": np.array([0, 1e6, 0])},
    )
    np.testing.assert_allclose(result.D_total_per_wire["wire"], 2e307)
    assert result.D_fretting_fraction == pytest.approx(0.5)


def test_registered_curve_values_keep_existing_arithmetic():
    for name in (
        "dnv_rpc203_t_curve",
        "saevik_2011_flat_steel",
        "nasution_2014_copper",
        "en1999_1_3_aluminium",
    ):
        sample = get_sn_curve(name)
        for stress in (41.0, 50.0, 100.0, 160.0, 300.0):
            knee = sample.transition_stress()
            m, log_c = (
                (sample.m2, sample.log_C2)
                if knee is not None and stress < knee
                else (sample.m1, sample.log_C1)
            )
            expected = 10 ** (log_c - m * math.log10(stress))
            assert sample.cycles_to_failure(stress) == expected


def test_replacing_a_valid_curve_revalidates_changed_definition():
    with pytest.raises(ValueError):
        replace(curve(), m2=5)
