# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Independent work limits and exhaustive checks of initial-state bounds."""

import numpy as np
import pytest
from slider_initial_work import initial_work_bounds, path_work


def test_elastic_work_and_unloading_can_be_negative():
    assert path_work([0, 0.2], 0.3) == pytest.approx(0.08)
    assert path_work([0, -0.2], 0.3) == pytest.approx(-0.04)


def test_monotone_yield_integrates_the_elastic_triangle_and_plastic_rectangle():
    assert path_work([0, 3], 0) == pytest.approx(2.5)
    bounds = initial_work_bounds([0, 3])
    assert bounds.minimum == pytest.approx(1)
    assert bounds.maximum == pytest.approx(3)
    assert bounds.minimising_initial_force == -1
    assert bounds.maximising_initial_force == 1


def test_reversal_breakpoint_is_needed_for_the_minimum():
    bounds = initial_work_bounds([0, 1, -1, 0])
    # From zero initial force the path only touches yield: no plastic work.
    # At either saturated initial force it has a half-unit net work input.
    assert bounds.minimum == pytest.approx(0)
    assert bounds.maximum == pytest.approx(0.5)
    assert bounds.minimising_initial_force == pytest.approx(0)


def test_shifted_displacement_and_reversed_force_have_same_work_bounds():
    path = np.array([0, 0.6, -0.8, 0.1, 1.3, -0.4])
    a = initial_work_bounds(path)
    b = initial_work_bounds(-path + 2.4)
    assert b.minimum == pytest.approx(a.minimum, abs=1e-13)
    assert b.maximum == pytest.approx(a.maximum, abs=1e-13)


@pytest.mark.parametrize("seed", range(10))
def test_piecewise_bounds_enclose_a_dense_independent_initial_state_sample(seed):
    rng = np.random.default_rng(seed)
    path = np.r_[0, np.cumsum(rng.uniform(-1.7, 1.7, 14))]
    initial = np.linspace(-1, 1, 10001)
    force = initial.copy()
    # Independent energy accounting, using stored-energy change plus slip work.
    loss = np.zeros_like(initial)
    for increment in np.diff(path):
        trial = force + increment
        slip = np.sign(trial) * np.maximum(np.abs(trial) - 1, 0)
        force = trial - slip
        loss += force * slip
    work = 0.5 * (force**2 - initial**2) + loss
    bounds = initial_work_bounds(path)
    assert bounds.minimum <= np.min(work) + 1e-12
    assert bounds.maximum >= np.max(work) - 1e-12
    assert bounds.minimum == pytest.approx(np.min(work), abs=2e-4)
    assert bounds.maximum == pytest.approx(np.max(work), abs=2e-4)
    assert path_work(path, bounds.minimising_initial_force) == pytest.approx(
        bounds.minimum, abs=1e-13
    )


def test_constant_and_zero_length_paths_have_zero_work():
    assert initial_work_bounds([3]).minimum == 0
    assert initial_work_bounds([3, 3]).maximum == 0


@pytest.mark.parametrize("path", [[], [np.nan], [[0, 1]]])
def test_invalid_paths_are_rejected(path):
    with pytest.raises(ValueError, match="path"):
        initial_work_bounds(path)


@pytest.mark.parametrize("force", [-1.01, 1.01, np.nan])
def test_invalid_initial_force_is_rejected(force):
    with pytest.raises(ValueError, match="initial force"):
        path_work([0, 1], force)
