# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Frictionless and unloaded contacts have no tangential force or stuck area."""

import numpy as np
import pytest

from cablexsec.mechanics.mindlin_contact import mindlin_monotonic, stuck_zone_ratio


@pytest.mark.parametrize("mu,normal", [(0, 1), (0.3, 0), (0, 0)])
def test_zero_capacity_contact(mu, normal):
    with np.errstate(all="raise"):
        np.testing.assert_array_equal(
            mindlin_monotonic(np.array([0, 0.1]), mu, normal, 100), [0, 0]
        )
        assert stuck_zone_ratio(0, normal, mu) == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mu": -1},
        {"F_n": -1},
        {"K_t": 0},
        {"mu": np.nan},
        {"delta": np.array([np.inf])},
    ],
)
def test_invalid_monotonic_inputs(kwargs):
    with pytest.raises(ValueError):
        mindlin_monotonic(**({"delta": 0, "mu": 0.3, "F_n": 1, "K_t": 100} | kwargs))


@pytest.mark.parametrize("mu,normal", [(-1, 1), (0.3, -1), (np.nan, 1)])
def test_invalid_stuck_zone_inputs(mu, normal):
    with pytest.raises(ValueError):
        stuck_zone_ratio(0, normal, mu)
