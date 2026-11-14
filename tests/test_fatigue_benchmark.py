# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""A failed numerical comparison must not produce completed speed evidence."""

import json
from dataclasses import replace

import benchmark_fatigue_counter as benchmark
import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis import fatigue


def test_benchmark_completes_only_after_comparing_every_case(tmp_path, monkeypatch):
    output = tmp_path / "benchmark"
    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["benchmark", "--output", str(output), "--samples", "13", "--repeat", "1"],
    )
    counter = fatigue.rainflow_cycles
    assert benchmark.main() == 0
    assert fatigue.rainflow_cycles is counter
    report = json.loads((output / "comparison.json").read_text())
    assert report["complete"]
    assert len(report["cases"]) == 3
    assert len(report["benchmark_sha256"]) == 64
    for row in report["cases"]:
        assert row["all_damage_fields_exactly_equal"]
        assert len(row["seconds"]["unreduced"]) == 1
        assert len(row["seconds"]["reduced"]) == 1


def test_comparison_failure_restores_counter_and_never_reports_completion(
    tmp_path, monkeypatch
):
    output = tmp_path / "benchmark"
    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["benchmark", "--output", str(output), "--samples", "3", "--repeat", "1"],
    )
    counter = fatigue.rainflow_cycles

    def fail(first, second):
        raise AssertionError("deliberate numerical disagreement")

    monkeypatch.setattr(benchmark, "compare", fail)
    with pytest.raises(AssertionError, match="numerical disagreement"):
        benchmark.main()
    assert fatigue.rainflow_cycles is counter
    assert not (output / "comparison.json").exists()


def test_comparison_detects_wrong_location_or_damage():
    cable = cx.load_cable(cx.inter_array_single_core_yaml_path())
    k = np.array([0, 0.01, 0])
    result = cx.compute_fatigue(cable, k, k * 0, n_fibres=4)
    with pytest.raises(AssertionError, match="critical_layer"):
        benchmark.compare(result, replace(result, critical_layer="wrong"))
    with pytest.raises(AssertionError, match="layer sets"):
        benchmark.compare(result, replace(result, D_total_per_wire={}))
    wrong = {name: values + 1 for name, values in result.D_bulk_per_fibre.items()}
    with pytest.raises(AssertionError):
        benchmark.compare(result, replace(result, D_bulk_per_fibre=wrong))
