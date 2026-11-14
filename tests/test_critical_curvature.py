# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for per-wire critical curvature (R3.4)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cablexsec.mechanics.critical_curvature import distribution_from_Fn, kappa_c

E_W, NU = 210.0e9, 0.30
A_W = math.pi * 0.005**2 / 4.0
ALPHA = math.radians(16.0)
MU = 0.20


# Test-taxonomy category.
pytestmark = pytest.mark.analytical_sanity


def test_kappa_c_formula() -> None:
    fn = 50.0
    expected = MU * fn / (E_W * A_W * math.cos(ALPHA) ** 2 * math.sin(ALPHA))
    assert float(kappa_c(fn, E_W, A_W, ALPHA, MU)) == pytest.approx(expected)


def test_linear_scaling_with_Fn() -> None:
    k1 = float(kappa_c(10.0, E_W, A_W, ALPHA, MU))
    k2 = float(kappa_c(30.0, E_W, A_W, ALPHA, MU))
    assert k2 == pytest.approx(3.0 * k1)


def test_uniform_contact_gives_uniform_kappa_c() -> None:
    """Axisymmetric (uniform F_n) -> all kappa_c equal (spread ratio 1)."""
    fn = np.full(93, 42.0)
    dist = distribution_from_Fn(fn, E_W, A_W, ALPHA, MU)
    assert dist.spread_ratio == pytest.approx(1.0)
    assert dist.min == pytest.approx(dist.max)


def test_nonuniform_Fn_widens_distribution() -> None:
    """Three-core F_n(phi) -> a spread of kappa_c (wires slip at different kappa)."""
    phi = np.linspace(0, 2 * np.pi, 93, endpoint=False)
    fn = 40.0 + 15.0 * np.cos(3 * phi)  # trefoil 3-fold variation
    dist = distribution_from_Fn(fn, E_W, A_W, ALPHA, MU, phi_rad=phi)
    assert dist.spread_ratio > 1.5
    assert dist.min < dist.mean < dist.max


def test_zero_lay_angle_raises() -> None:
    with pytest.raises(ValueError, match="lay_angle"):
        kappa_c(10.0, E_W, A_W, 0.0, MU)


def test_summary_string() -> None:
    dist = distribution_from_Fn(np.full(10, 30.0), E_W, A_W, ALPHA, MU)
    assert "kappa_c" in dist.summary() and "max/min" in dist.summary()


def test_pretension_only_zero_tension_limit() -> None:
    """With tension removed, kappa_c is driven by the pretension F_n = P0 only."""
    P0_force = 3.0  # N per wire from pretension
    expected = MU * P0_force / (E_W * A_W * math.cos(ALPHA) ** 2 * math.sin(ALPHA))
    assert float(kappa_c(P0_force, E_W, A_W, ALPHA, MU)) == pytest.approx(expected)
