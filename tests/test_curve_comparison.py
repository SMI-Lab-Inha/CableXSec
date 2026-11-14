# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Independent cyclic-work identities, including small abscissa reversals."""

import numpy as np
import pytest

from cablexsec.analysis.curve_comparison import (
    cycle_work,
    insert_cycle_crossings,
    major_branches,
)


def test_ellipse_work_and_elastic_addition():
    phase = np.linspace(-0.25, 8 * np.pi + 0.25, 10001)
    x = np.sin(phase)
    y = 0.3 * np.cos(phase)
    work = cycle_work(x, y)
    assert len(work) == 4
    np.testing.assert_allclose(work, 0.3 * np.pi, rtol=2.0e-6)
    np.testing.assert_allclose(cycle_work(x, y + 7 * x), work, atol=2.0e-13)


def test_jitter_does_not_create_spurious_cycles():
    phase = np.linspace(-0.25, 8 * np.pi + 0.25, 10001)
    x = np.sin(phase) + 0.002 * np.sin(500 * phase)
    y = 0.3 * np.cos(phase)
    work = cycle_work(x, y)
    assert len(work) == 4
    np.testing.assert_allclose(work, 0.3 * np.pi, rtol=2.0e-4)


def test_no_complete_cycle_and_constant_abscissa():
    assert cycle_work(np.arange(10.0), np.arange(10.0)).size == 0
    assert cycle_work(np.zeros(10), np.arange(10.0)).size == 0


def test_branch_labels_depend_only_on_abscissa():
    direction, _ = major_branches(np.array([0.0, 1.0, 0.9, -1.0, -0.9, 1.0]))
    np.testing.assert_array_equal(direction, [1.0, -1.0, -1.0, 1.0, 1.0, 1.0])


@pytest.mark.parametrize(
    "x,y",
    [
        ([0.0, float("nan"), 1.0], [0.0, 1.0, 2.0]),
        ([0.0, 1.0], [0.0, 1.0]),
        ([0.0, 1.0, 2.0], [0.0, 1.0]),
    ],
)
def test_invalid_curves_rejected(x, y):
    with pytest.raises(ValueError):
        cycle_work(np.array(x), np.array(y))


@pytest.mark.parametrize("crossing", [None, 0.0, 0.37, -0.6, 2.0])
def test_inserted_crossings_preserve_observations_and_polygon_work(crossing):
    x = np.array([-0.8, 0.9, 0.7, -1.0, -0.7, 1.0, 0.0, -0.9, 0.8])
    y = np.arange(len(x), dtype=float) ** 2 / 10
    augmented, original, boundaries = insert_cycle_crossings(x, crossing=crossing)
    np.testing.assert_array_equal(augmented[original], x)
    centre = 0.5 * (min(x) + max(x)) if crossing is None else crossing
    np.testing.assert_array_equal(
        augmented[boundaries], np.full(len(boundaries), centre)
    )
    # Inserted ordinates lie on the same measured line segments.
    augmented_y = np.interp(np.arange(len(augmented)), original, y)
    for lo, hi in zip(original[:-1], original[1:], strict=True):
        if hi > lo + 1:
            fraction = (augmented[lo + 1] - augmented[lo]) / (
                augmented[hi] - augmented[lo]
            )
            augmented_y[lo + 1] = augmented_y[lo] + fraction * (
                augmented_y[hi] - augmented_y[lo]
            )
    np.testing.assert_allclose(
        cycle_work(augmented, augmented_y, crossing=crossing),
        cycle_work(x, y, crossing=crossing),
        atol=2e-15,
    )


def test_endpoint_crossing_is_not_duplicated():
    values = np.array([-1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0])
    augmented, original, boundaries = insert_cycle_crossings(values, crossing=0.0)
    np.testing.assert_array_equal(augmented, values)
    np.testing.assert_array_equal(original, np.arange(len(values)))
    np.testing.assert_array_equal(boundaries, [1, 5])


def test_truncated_cycle_retains_the_last_physical_reversal():
    x = np.array([0.02, 0.6, 1.0, 0.4, -0.5, -0.97, -0.8, -0.06])
    directions, indices = major_branches(x)
    np.testing.assert_array_equal(indices, [0, 2, 5, 7])
    assert directions[-1] == 1
    # A global half-range threshold loses both end reversals because less
    # than half a cycle is recorded outside them.
    _, earlier = major_branches(x, prominence_fraction=0.5)
    np.testing.assert_array_equal(earlier, [0, 7])


def test_partial_end_branches_do_not_remove_complete_interior_cycles():
    phase = np.linspace(0.01, 8 * np.pi + 0.02, 4001)
    x = np.sin(phase)
    y = 0.4 * np.cos(phase) + 3 * x
    assert cycle_work(x, y).size == 3
    np.testing.assert_allclose(cycle_work(x, y), 0.4 * np.pi, rtol=1e-5)


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1, np.nan, np.inf])
def test_invalid_prominence_rejected(fraction):
    with pytest.raises(ValueError, match="prominence"):
        major_branches(np.array([0, 1, 0]), prominence_fraction=fraction)
