# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Independent high-precision force checks across contact parameter scales."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from cablexsec.mechanics.mindlin_contact import mindlin_monotonic, stuck_zone_ratio


@pytest.mark.parametrize(
    "delta,mu,normal,stiffness",
    [
        (1e-20, 0.3, 100, 1e6),
        (1e-30, 0.3, 100, 1e6),
        (1e-8, 0.3, 100, 1e6),
        (1e-5, 0.3, 100, 1e6),
        (1, 0.3, 100, 1e6),
        (1e-200, 1e200, 1e200, 1e200),
        (1e200, 1e-200, 1e-200, 1e-200),
        (1e-200, 0.3, 1e200, 1e-100),
        (1e200, 0.3, 1e-200, 1e100),
    ],
)
def test_monotonic_force_matches_decimal_reference(delta, mu, normal, stiffness):
    with localcontext() as context:
        context.prec = 1100
        d, m, n, k = (Decimal(str(v)) for v in (delta, mu, normal, stiffness))
        cap = m * n
        ratio = min(Decimal(2) * k * d / (3 * cap), Decimal(1))
        expected = float(cap * (1 - (1 - ratio) ** Decimal("1.5")))
    if expected == 0:
        with pytest.raises(ValueError, match="range"):
            mindlin_monotonic(delta, mu, normal, stiffness)
    else:
        actual = float(mindlin_monotonic(delta, mu, normal, stiffness))
        assert actual / expected == pytest.approx(1, rel=3e-13)


@pytest.mark.parametrize(
    "delta,mu,normal,stiffness",
    [
        (1e308, 1e308, 1e308, 1e308),
        (1e-300, 1, 1, 1e-300),
    ],
)
def test_unrepresentable_monotonic_force_is_rejected(delta, mu, normal, stiffness):
    with pytest.raises(ValueError, match="range"):
        mindlin_monotonic(delta, mu, normal, stiffness)


def test_monotonic_array_preserves_zero_and_negative_displacement_convention():
    result = mindlin_monotonic(
        np.array([[-1, 0, 1e-30], [1e-8, 1e-5, 1]]), 0.3, 100, 1e6
    )
    assert result.shape == (2, 3)
    np.testing.assert_array_equal(result[0, :2], 0)
    assert result[0, 2] > 0
    assert np.all(np.diff(result.ravel()) >= 0)


def test_complex_displacement_is_not_silently_discarded():
    with pytest.raises(ValueError):
        mindlin_monotonic(np.array([1j]), 0.3, 100, 1e6)


def test_no_positive_displacement_returns_zero_without_logarithms():
    np.testing.assert_array_equal(mindlin_monotonic([-1, 0], 0.3, 100, 1e6), 0)


@pytest.mark.parametrize(
    "force,normal,mu,expected",
    [
        (1e300, 1e200, 1e200, 1.0),
        (1e-300, 1e-200, 1e-200, 0.0),
        (0, 1e-200, 1e-200, 1.0),
    ],
)
def test_stick_diagnostic_handles_product_range_without_division_failure(
    force, normal, mu, expected
):
    assert stuck_zone_ratio(force, normal, mu) == expected
