# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for per-wire stress components and the Yun shear coefficient."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cablexsec.mechanics.wire_stress import (
    friction_stiffness,
    friction_yield_stress,
    local_bending_stress,
    von_mises_stress,
)
from cablexsec.mechanics.yun_coefficient import (
    series_shear_stiffness,
    shear_modulus,
    yun_xi,
)

E_W, NU = 210.0e9, 0.30
A_W = math.pi * 0.005**2 / 4.0
ALPHA = math.radians(16.0)
R_H = 0.082


# Test-taxonomy category.
pytestmark = pytest.mark.analytical_sanity


def test_friction_stiffness_formula() -> None:
    xi = 0.75
    assert friction_stiffness(xi, E_W, ALPHA) == pytest.approx(
        xi * E_W * math.cos(ALPHA) ** 2
    )


def test_friction_yield_stress() -> None:
    fy = friction_yield_stress(0.2, 1700.0, A_W)
    assert float(fy) == pytest.approx(0.2 * 1700.0 / A_W)


def test_local_bending_stress_peaks_in_plane() -> None:
    sb0 = float(local_bending_stress(0.05, 0.0, E_W, 0.005, ALPHA))
    expected = E_W * 0.05 * math.cos(ALPHA) ** 2 * math.cos(2 * ALPHA) * 0.0025
    assert sb0 == pytest.approx(expected)
    assert abs(float(local_bending_stress(0.05, math.pi / 2, E_W, 0.005, ALPHA))) < (
        1e-9 * abs(sb0)
    )


def test_von_mises_combines_axial_and_shear() -> None:
    vm = float(von_mises_stress(100.0e6, 50.0e6))
    assert vm == pytest.approx(math.sqrt((100e6) ** 2 + 3 * (0.31 * 50e6) ** 2))


# --- Yun xi --------------------------------------------------------------- #
def test_yun_xi_no_polymer_is_one() -> None:
    """No intervening polymer (rigid stack) => xi = 1 (Witz-Tan plane sections)."""
    K_G = series_shear_stiffness([])
    assert math.isinf(K_G)
    EA_w = E_W * A_W
    assert yun_xi(K_G, EA_w, ALPHA, R_H) == pytest.approx(1.0)


def test_yun_xi_soft_polymer_tends_to_zero() -> None:
    """Very compliant polymer => xi -> 0 (full shear compliance)."""
    G_soft = shear_modulus(1.0e6, 0.45)  # very soft
    K_G = series_shear_stiffness([(0.02, G_soft, 0.005)])
    EA_w = E_W * A_W
    assert yun_xi(K_G, EA_w, ALPHA, R_H) < 0.2


def test_yun_xi_monotonic_in_stiffness() -> None:
    EA_w = E_W * A_W
    stiff = yun_xi(
        series_shear_stiffness([(0.01, shear_modulus(1.0e9, 0.4), 0.01)]),
        EA_w,
        ALPHA,
        R_H,
    )
    soft = yun_xi(
        series_shear_stiffness([(0.01, shear_modulus(1.0e8, 0.4), 0.01)]),
        EA_w,
        ALPHA,
        R_H,
    )
    assert stiff > soft
    assert 0.0 < soft < stiff < 1.0


def test_series_compliances_add() -> None:
    """Two identical layers in series halve the stiffness."""
    one = series_shear_stiffness([(0.01, 1.0e9, 0.01)])
    two = series_shear_stiffness([(0.01, 1.0e9, 0.01), (0.01, 1.0e9, 0.01)])
    assert two == pytest.approx(one / 2.0)


def test_von_mises_array() -> None:
    out = von_mises_stress(np.array([0.0, 100e6]), np.array([0.0, 0.0]))
    assert np.allclose(out, [0.0, 100e6])
