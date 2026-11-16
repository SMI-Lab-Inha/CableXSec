# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Static/dynamic friction law against the measured wire-on-tape loops of
Dai, Sævik and Ye (2018), Fig. 4 (rows DAI18-* of the validation report).

Scope: a small-scale friction rig (one armour wire between two anti-wear
tapes), not a cable section. Gated: the digitisation against the printed
static coefficients, and re-sticking at every reversal. The model errors
(peak, dynamic level, work, decay, frequency) are reported, not gated
(doc/benchmarks.md, DAI18).
"""

from __future__ import annotations

import math

import check_dai2018_friction as dai
import numpy as np
import pytest

full_scale = pytest.mark.full_scale


@pytest.fixture(scope="module")
def result() -> dict:
    return dai.compare()


def test_digitised_data_carry_their_provenance():
    for name, series in (
        (dai.TEST1_CSV, {"8 MPa", "4 MPa", "1 MPa"}),
        (dai.TEST3_CSV, {"0.01 Hz", "0.05 Hz", "0.1 Hz", "0.2 Hz"}),
    ):
        text = (dai.DATA / name).read_text(encoding="utf-8")
        assert "Marine Structures 59:423-443" in text
        assert "Source PDF SHA-256: b5f8d199" in text
        assert "616a05a2db2d9d91" in text  # embedded figure raster
        assert "NOT experimental uncertainty" in text
        assert "end-of-stroke rises" in text
        loops = dai.read_loops(name)
        assert set(loops) == series
        for loop in loops.values():
            for stroke in loop.strokes:
                assert stroke.x.size > 100 and stroke.branch_x.size > 100
                assert stroke.x.min() >= -2.0
                assert stroke.x.max() <= 2.5
            assert min(loop.stick_slope_per_mm) > 0.0
        assert 0.0 < dai.reading_resolution(name) < 0.005


@full_scale
def test_digitised_maximum_matches_the_printed_static_coefficient(result):
    """The printed mu_s is the loop maximum (p. 428): the digitised maxima
    agree within the printed precision plus the reading resolution."""
    assert result["digitisation_error"] <= result["digitisation_tolerance"]
    for label, value in result["digitised_maximum"].items():
        assert value == pytest.approx(dai.TEST1_PRINTED[label][0], abs=0.0071)


@full_scale
def test_every_reversal_re_sticks(result):
    """Every measured stroke peaks above its dynamic level; the slider does so
    on every stroke with stick_time_s = 0, and never after the first
    breakaway with the default stick_time_s = inf."""
    n = result["strokes"]
    assert n == 14
    assert result["measured_peaks"] == n
    assert result["model_peaks"] == n
    assert result["never_heals_peaks"] == 0


def test_slider_reproduces_its_inputs(result):
    """Harness check: the model's dynamic level is mu_d and its peak is the
    static bound m_s mu_d less at most one elastic sample step."""
    for case in result["cases"]:
        for m in case["model"]:
            assert m.level == pytest.approx(case["mu_d"], rel=1e-12)
            assert case["mu_s"] * (1 - 5e-3) < m.peak <= case["mu_s"] * (1 + 1e-12)
            assert m.decay_mm == 0.0
            assert m.work > 0.0
        for m in case["never_heals"]:
            assert m.peak == pytest.approx(case["mu_d"], rel=1e-12)
        # The measured measures are finite and ordered.
        for m in case["measured"]:
            assert m.peak > m.level > 0.0
            assert math.isfinite(m.decay_mm) and m.decay_mm > 0.0
            assert 0.0 < m.peak_travel_mm <= dai.PEAK_ZONE_MM


def test_reported_values_are_stated(result):
    """The reported quantities exist and are finite (no gate on their size)."""
    for case in result["cases"]:
        assert np.isfinite(case["peak_error_pct"]).all()
        assert np.isfinite(case["level_error_pct"]).all()
        assert math.isfinite(case["work_error_pct"])
    assert math.isfinite(result["test3_peak_rise_pct"])
    assert len(result["test3_level_change_pct"]) == 3
