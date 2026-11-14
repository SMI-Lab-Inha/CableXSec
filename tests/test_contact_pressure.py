# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for the three-core contact-pressure shape g(phi) (R3.2, corrected)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cablexsec.mechanics.contact_pressure import (
    foundation_stiffness,
    solve_three_core,
)

# Representative three-core parameters (round, generic values).
PARAMS = {
    "k_core": 1.0e11,
    "k_filler": 2.0e10,
    "theta_core_rad": math.radians(80.0),
    "EI_ring": 1.0e2,
    "r_bedding_m": 0.0755,
    "n_wires": 90,
}
PHI_CORES = (math.pi / 6, math.pi / 6 + 2 * math.pi / 3, math.pi / 6 + 4 * math.pi / 3)


# Test-taxonomy category.
pytestmark = pytest.mark.analytical_sanity


def test_foundation_is_binary() -> None:
    over_core = foundation_stiffness(math.pi / 6, 9.0, 1.0, math.radians(80), PHI_CORES)
    over_void = foundation_stiffness(math.pi / 2, 9.0, 1.0, math.radians(40), PHI_CORES)
    assert over_core == 9.0
    assert over_void == 1.0  # midway between cores is a filler void


def test_shape_g_has_unit_mean() -> None:
    """Correction A: g(phi) is normalised to mean 1 over the wire set, <1e-12."""
    res = solve_three_core(5.0e6, **PARAMS)
    assert float(np.mean(res.shape_g)) == pytest.approx(1.0, abs=1e-12)


def test_shape_g_is_uniform_when_no_trefoil() -> None:
    """g reduces to 1.0 everywhere when k3 = k6 = 0 (single-core / export)."""
    p = dict(PARAMS)
    p["k_filler"] = p["k_core"]  # uniform foundation -> no harmonics
    res = solve_three_core(5.0e6, **p)
    assert res.u3 == pytest.approx(0.0, abs=1e-18)
    assert res.u6 == pytest.approx(0.0, abs=1e-18)
    assert np.allclose(res.shape_g, 1.0, atol=1e-12)


def test_R3p3_single_core_export_reduction_k3_k6_vanish() -> None:
    """R3.3: a single-core export cable has a uniform annular bedding (no trefoil
    voids), so the trefoil harmonics k3 = k6 = 0 and the contact shape g ≡ 1 —
    the three-core Fourier correction reduces to the axisymmetric F_n exactly.
    """
    p = dict(PARAMS)
    p["k_filler"] = p["k_core"]  # single-core: no core/void contrast
    res = solve_three_core(5.0e6, **p)
    assert res.k0 == pytest.approx(p["k_core"], rel=1e-9)  # uniform foundation
    assert res.k3 == pytest.approx(0.0, abs=1e-3)  # no trefoil 3-fold harmonic
    assert res.k6 == pytest.approx(0.0, abs=1e-3)  # no 6-fold harmonic
    assert np.allclose(res.shape_g, 1.0, atol=1e-12)  # F_n correction = 1×


def test_shape_g_independent_of_applied_pressure() -> None:
    """The shape is dimensionless: it does not scale with q0."""
    lo = solve_three_core(2.0e6, **PARAMS)
    hi = solve_three_core(8.0e6, **PARAMS)
    assert np.allclose(lo.shape_g, hi.shape_g, rtol=1e-9)


def test_mean_ring_and_foundation_force_balance() -> None:
    """Mean equilibrium includes both ring and foundation reactions."""
    res = solve_three_core(5.0e6, **PARAMS)
    assert res.balance_error < 1e-10


def test_three_fold_periodicity() -> None:
    """Gate: g(phi) is 3-fold periodic and genuinely varies."""
    res = solve_three_core(5.0e6, **PARAMS)
    g = res.shape_g
    m = 99  # divisible by 3
    grid = 2 * np.pi * np.arange(m) / m
    u = res.radial_displacement(grid)
    kk = np.array(
        [
            foundation_stiffness(
                p,
                PARAMS["k_core"],
                PARAMS["k_filler"],
                PARAMS["theta_core_rad"],
                PHI_CORES,
            )
            for p in grid
        ]
    )
    g_grid = kk * u
    third = m // 3
    assert np.allclose(g_grid, np.roll(g_grid, third), rtol=1e-6, atol=1e-9)
    assert g.max() - g.min() > 1e-6  # genuine variation


def test_fourier_coefficients_reported() -> None:
    res = solve_three_core(5.0e6, **PARAMS)
    # k0 is the average stiffness, between filler and core values.
    assert PARAMS["k_filler"] < res.k0 < PARAMS["k_core"]
