# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Ensure a numerical disagreement cannot be labelled successful in the report."""

from dataclasses import replace

import pytest
import run_validation

pytestmark = pytest.mark.analytical_sanity


def test_changed_wire_bending_fails_reported_comparison(monkeypatch):
    original = run_validation.cx.compute_stress

    def incorrect_bending(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(
            result, sigma_B_Pa={k: 2 * v for k, v in result.sigma_B_Pa.items()}
        )

    monkeypatch.setattr(run_validation.cx, "compute_stress", incorrect_bending)
    row = next(r for r in run_validation._rows() if r.ident == "σ_B")
    assert row.status == "FAIL"
    assert float(row.value.rstrip("%")) == pytest.approx(100.0)


def test_bending_work_report_compares_identical_measured_paths_and_units():
    row = next(r for r in run_validation._rows() if r.ident == "BM1-loop")
    assert "matched experimental" in row.target
    assert "70.4 J/m vs measured 62.3 J/m" in row.value
    assert "12.9%" in row.value
    assert row.status == "FAIL"  # The existing 10% threshold is not relaxed.


_XSEC = run_validation.Row(
    "1",
    "xsec-K",
    "Coupling tangent bounds (κ→0=EI_max, κ→∞=EI_min); slip-front local slope",
    "< 0.01%",
    "κ0:0.00e+00% / κ∞:7.39e-12%",
    "PASS",
)


def _recorded(row, value):
    return replace(row, value=value).line()


def test_machine_round_off_is_not_reported_stale():
    # The 18 October 2026 false report: the recorded value differed only in
    # round-off digits (6.33e-13% against 7.39e-12%, tolerance < 0.01%).
    line = _recorded(_XSEC, "κ0:0.00e+00% / κ∞:6.33e-13%")
    assert line != _XSEC.line()
    assert run_validation._row_recorded(_XSEC, line)
    text = "header\n" + _XSEC.line() + "\nfooter"
    recorded = "header\n" + line + "\nfooter"
    assert not run_validation.report_is_stale(
        [_XSEC], text, recorded, public_only=False
    )
    assert not run_validation.report_is_stale([_XSEC], text, recorded, public_only=True)


@pytest.mark.parametrize(
    "value",
    [
        "κ0:0.00e+00% / κ∞:2.00e-06%",  # a real change, still inside the target
        "κ0:1.00e-03% / κ∞:6.33e-13%",
        "κ0:0.00e+00% / κ∞:6.33e-13% (changed)",
    ],
)
def test_real_value_changes_remain_stale(value):
    line = _recorded(_XSEC, value)
    assert not run_validation._row_recorded(_XSEC, line)
    text = "header\n" + _XSEC.line()
    assert run_validation.report_is_stale(
        [_XSEC], text, "header\n" + line, public_only=False
    )
    assert run_validation.report_is_stale([_XSEC], text, line, public_only=True)


def test_status_target_and_other_lines_are_compared_exactly():
    failed = replace(_XSEC, status="FAIL").line()
    assert not run_validation._row_recorded(_XSEC, failed)
    tighter = replace(_XSEC, target="< 0.001%").line()
    assert not run_validation._row_recorded(_XSEC, tighter)
    text = "header\n" + _XSEC.line()
    assert run_validation.report_is_stale(
        [_XSEC], text, "other header\n" + _XSEC.line(), public_only=False
    )
    assert run_validation.report_is_stale([_XSEC], text, None, public_only=False)
    assert run_validation.report_is_stale([_XSEC], text, text + "\n", public_only=False)


def test_round_off_class_follows_the_paired_tolerance():
    # STATE-T style row: one tolerance per value segment.
    target = "< 1e-10 relative strain; < 1e-5 Pa stress"
    a = run_validation._value_classes("strain 3.11e-15; stress 1.30e-06 Pa", target)
    b = run_validation._value_classes("strain 9.20e-16; stress 1.30e-06 Pa", target)
    c = run_validation._value_classes("strain 3.11e-15; stress 1.31e-06 Pa", target)
    assert a == b  # strain is 1e-5 of its tolerance: round-off
    assert a != c  # stress is 0.13 of its tolerance: compared exactly
    # Without a positive tolerance nothing is classed as round-off.
    assert run_validation._value_classes("4.8e-06", ">= 0") != (
        run_validation._value_classes("4.9e-06", ">= 0")
    )
    # A 1e-12 target: a few machine epsilons are round-off on every platform
    # (Linux and Windows printed 1.07e-15 and 9.48e-16), a change is not.
    tight = "< 1e-12 relative; ratio <= 1"
    linux = run_validation._value_classes("identity 9.48e-16; ratio 0.333", tight)
    windows = run_validation._value_classes("identity 1.07e-15; ratio 0.333", tight)
    changed = run_validation._value_classes("identity 3.00e-13; ratio 0.333", tight)
    assert linux == windows
    assert linux != changed
    # Inside-word digits (κ0, BM1) are text, not numbers.
    assert run_validation._value_classes("κ0 BM1", "< 0.01%") == ("κ0 BM1",)
