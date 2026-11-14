# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Turning-point reduction preserves cycle ranges, means, counts and indices."""

from itertools import product

import numpy as np
import pytest
import rainflow

from cablexsec.analysis.fatigue import rainflow_cycles


def upstream(values):
    return [cycle for cycle in rainflow.extract_cycles(list(values)) if cycle[0] > 0]


def test_exhaustive_short_quantized_records_match_unreduced_counter():
    # Includes repeated turning plateaus, equal ranges, nested cycles, open
    # excursions and constants. Compare indices as well as fatigue quantities.
    for count in range(3, 9):
        for record in product((-1.0, 0.0, 1.0), repeat=count):
            assert list(rainflow_cycles(np.array(record))) == upstream(record), record


@pytest.mark.parametrize("seed", range(12))
def test_irregular_drifting_and_quantized_records_match(seed):
    rng = np.random.default_rng(seed)
    series = rng.normal(size=2000).cumsum()
    for values in (series, np.round(series), np.repeat(series[::20], 20)):
        assert list(rainflow_cycles(values)) == upstream(values)


def test_repeated_plateau_uses_original_last_plateau_index():
    values = np.array([0, 0, 3, 3, 3, -1, -1, 2, 2], dtype=float)
    assert list(rainflow_cycles(values)) == [
        (3.0, 1.5, 0.5, 0, 4),
        (4.0, 1.0, 0.5, 4, 6),
        (3.0, 0.5, 0.5, 6, 8),
    ]


@pytest.mark.parametrize("sign", [1, -1])
def test_long_monotonic_record_preserves_endpoints(sign):
    values = sign * np.linspace(-25, 100, 10001)
    assert list(rainflow_cycles(values)) == [(125.0, sign * 37.5, 0.5, 0, 10000)]


def test_nested_cycle_counts_have_independent_expected_values():
    values = np.array([0, 4, 1, 3, -2, 5, 0], dtype=float)
    assert list(rainflow_cycles(values)) == [
        (2.0, 2.0, 1.0, 2, 3),
        (4.0, 2.0, 0.5, 0, 1),
        (6.0, 1.0, 0.5, 1, 4),
        (7.0, 1.5, 0.5, 4, 5),
        (5.0, 2.5, 0.5, 5, 6),
    ]


def test_affine_interpolation_preserves_cycles_and_maps_event_indices():
    turns = np.array([0.0, 4.0, 1.0, 3.0, -2.0, 5.0, 0.0])
    time = np.arange(turns.size) * 100
    dense = np.interp(np.arange(time[-1] + 1), time, turns)
    expected = [(r, m, c, i * 100, j * 100) for r, m, c, i, j in upstream(turns)]
    assert list(rainflow_cycles(dense)) == expected
