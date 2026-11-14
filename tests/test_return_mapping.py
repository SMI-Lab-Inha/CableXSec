# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for the universal Coulomb return-mapping element (R1.1)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cablexsec.mechanics.return_mapping import CoulombElement, coulomb_return_map

# Test-taxonomy category.
pytestmark = pytest.mark.analytical_sanity


def test_1d_elastic_below_yield() -> None:
    f, slip, slipping = coulomb_return_map(0.0, 0.3, k=2.0, f_y=1.0)
    assert float(f) == pytest.approx(0.6)  # elastic: k*du
    assert float(slip) == 0.0
    assert not bool(slipping)


def test_1d_returns_to_surface_on_slip() -> None:
    # f_trial = 0 + 2*0.6 = 1.2 > f_y=1.0 -> saturate.
    f, slip, slipping = coulomb_return_map(0.0, 0.6, k=2.0, f_y=1.0)
    assert float(f) == pytest.approx(1.0)  # exactly on the yield surface
    assert float(slip) == pytest.approx(0.6 - (1.0 - 0.0) / 2.0)  # 0.1
    assert bool(slipping)


def test_1d_gross_slip_magnitude_capped() -> None:
    el = CoulombElement(k=3.0)
    for _ in range(20):
        el.update(0.5, f_y=1.0)
    assert abs(float(el.f)) == pytest.approx(1.0)  # |f| == f_y in gross slip


def test_masing_factor_two_on_reversal() -> None:
    """On reversal the element stays elastic over a force change of 2*f_y."""
    k, f_y = 2.0, 1.0
    el = CoulombElement(k=k)
    el.update(2.0, f_y)  # drive well past yield -> f = +f_y
    assert float(el.f) == pytest.approx(f_y)
    # Reverse in small steps; it should re-yield only after du_total = 2*f_y/k.
    reyield_at = None
    travelled = 0.0
    for _ in range(40):
        _, slipping = el.update(-0.05, f_y)
        travelled += 0.05
        if bool(slipping):
            reyield_at = travelled
            break
    assert reyield_at is not None
    # 2*f_y/k = 1.0 exactly; discrete 0.05 steps detect it within one step.
    assert reyield_at == pytest.approx(2.0 * f_y / k, abs=0.06)
    assert float(el.f) == pytest.approx(-f_y, abs=1e-9)


def test_slip_increment_decomposition() -> None:
    """du = elastic part + plastic slip when slipping."""
    f_prev, du, k, f_y = 0.5, 0.4, 2.0, 1.0
    f, slip, _ = coulomb_return_map(f_prev, du, k=k, f_y=f_y)
    elastic = (float(f) - f_prev) / k
    assert elastic + float(slip) == pytest.approx(du)


def test_vectorised_population_1d() -> None:
    el = CoulombElement(k=1.0, shape=(4,))
    du = np.array([0.5, 1.5, -2.0, 0.1])
    f, slipping = el.update(du, f_y=1.0)
    assert np.allclose(f, [0.5, 1.0, -1.0, 0.1])
    assert slipping.tolist() == [False, True, True, False]


def test_2d_returns_along_trial_direction() -> None:
    f, slip, slipping = coulomb_return_map(
        np.array([0.0, 0.0]), np.array([2.0, 0.0]), k=1.0, f_y=1.0, vector=True
    )
    assert np.linalg.norm(f) == pytest.approx(1.0)
    assert f[0] == pytest.approx(1.0) and f[1] == pytest.approx(0.0)
    assert bool(slipping)


def test_2d_nonproportional_direction_change() -> None:
    el = CoulombElement(k=1.0, vector=True)
    el.update(np.array([2.0, 0.0]), f_y=1.0)  # -> [1, 0]
    f, _ = el.update(np.array([0.0, 2.0]), f_y=1.0)
    assert np.linalg.norm(f) == pytest.approx(1.0)
    # direction follows the new trial f_prev+du = [1, 2]
    assert f[0] == pytest.approx(1.0 / math.sqrt(5.0))
    assert f[1] == pytest.approx(2.0 / math.sqrt(5.0))


def test_step_varying_yield_limit() -> None:
    """f_y may shrink between steps (non-proportional contact load)."""
    el = CoulombElement(k=1.0)
    el.update(5.0, f_y=2.0)
    assert float(el.f) == pytest.approx(2.0)
    _, slipping = el.update(0.0, f_y=1.0)  # yield surface shrinks below |f|
    assert float(el.f) == pytest.approx(1.0)
    assert bool(slipping)


def test_invalid_stiffness_raises() -> None:
    with pytest.raises(ValueError, match="stiffness"):
        CoulombElement(k=0.0)
