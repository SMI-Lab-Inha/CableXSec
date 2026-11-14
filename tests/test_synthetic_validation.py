# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""The public synthetic generator rejects invalid physical inputs explicitly."""

import numpy as np
import pytest

from cablexsec.io.synthetic import jonswap_series


@pytest.mark.parametrize("name", ["dt_s", "duration_s", "tp_s", "gamma"])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_positive_parameters(name, value):
    with pytest.raises(ValueError, match=name):
        jonswap_series(**{name: value})


@pytest.mark.parametrize("name", ["hs_m", "rms_curvature_1_m"])
@pytest.mark.parametrize("value", [-1, np.nan, np.inf])
def test_nonnegative_parameters(name, value):
    with pytest.raises(ValueError, match=name):
        jonswap_series(**{name: value})


@pytest.mark.parametrize(
    "name", ["mean_tension_N", "tension_curvature_coupling_N_per_1_m"]
)
def test_finite_force_parameters(name):
    with pytest.raises(ValueError, match="tension"):
        jonswap_series(**{name: np.inf})


def test_short_record_is_rejected_before_synthesis():
    with pytest.raises(ValueError, match="at least two samples"):
        jonswap_series(duration_s=0.1, dt_s=1)


@pytest.mark.parametrize("name", ["hs_m", "rms_curvature_1_m"])
def test_zero_excitation_is_valid(name):
    result = jonswap_series(duration_s=4, dt_s=1, **{name: 0})
    np.testing.assert_array_equal(result.curvature_x_1_m, np.zeros(4))
