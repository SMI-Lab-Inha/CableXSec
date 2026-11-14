# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Finiteness checks support NumPy inputs without bypassing shape validation."""

import numpy as np
import pytest

from cablexsec.exceptions import PhysicalRangeError
from cablexsec.provenance import require_finite


@pytest.mark.parametrize(
    "value", [np.int64(2), np.float64(3), np.array(1), np.ones((2, 3))]
)
def test_finite_numpy_values(value):
    require_finite(value, "load")


@pytest.mark.parametrize(
    "value", [np.array(np.nan), np.array([[0, np.inf]]), np.float64(-np.inf)]
)
def test_nonfinite_numpy_values_raise_domain_error(value):
    with pytest.raises(PhysicalRangeError, match="load"):
        require_finite(value, "load")


def test_iterators_remain_supported():
    require_finite(iter([1.0, 2.0]), "load")
    with pytest.raises(PhysicalRangeError, match="load"):
        require_finite(iter([1.0, np.nan]), "load")
