# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Gap G1: tendon curvature, transverse slip, relief and the end restraint."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.mindlin_contact import transverse_geodesic_factor
from cablexsec.mechanics.tendon_bending import (
    LateralRelief,
    TransverseSlipModel,
    curvature_coefficients,
    dong_anchor_bending,
    dong_anchor_tension,
    geodesic_slip_ratio,
    saevik_transverse_fraction,
)
from cablexsec.mechanics.tendon_section import WireSection
from cablexsec.mechanics.wire_stress import local_bending_fibre_stress

STRIP = WireSection(shape="rectangular", width_m=0.006, thickness_m=0.003)
E, G = 210e9, 210e9 / 2.6
J_ROARK = 0.006 * 0.003**3 * (1 / 3 - 0.21 / 2 * (1 - 1 / (12 * 2**4)))


@pytest.mark.parametrize("deg", [20.0, 24.8, 35.0, 55.0])
def test_loxodrome_coefficients_match_the_fatigue_path(deg):
    a = math.radians(deg)
    c = curvature_coefficients(a, 0.0)
    radial = float(local_bending_fibre_stress(1.0, 0.0, 0.0, 2.0, 1.0, a))
    tangential = float(
        local_bending_fibre_stress(1.0, 0.0, -0.5 * math.pi, 2.0, 1.0, a, 0.5 * math.pi)
    )
    assert c.normal == pytest.approx(radial, abs=1e-15)
    assert c.lateral == pytest.approx(tangential, abs=1e-15)


@pytest.mark.parametrize("deg", [20.0, 24.8, 35.0, 55.0])
def test_geodesic_coefficients_saevik_1992(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    geo = curvature_coefficients(a, 1.0)
    assert geo.lateral == 0.0
    assert geo.normal == pytest.approx(3 * c * c, rel=1e-14)  # Eq. 2.32
    assert geo.torsion == pytest.approx(
        -s * c * (1 / (s * s) - 3), rel=1e-14
    )  # Eq. 2.31
    lox = curvature_coefficients(a, 0.0)
    assert lox.torsion == pytest.approx(s * c**3, rel=1e-14)  # Eq. 2.64
    half = curvature_coefficients(a, 0.5)
    assert half.lateral == pytest.approx(0.5 * lox.lateral, rel=1e-14)
    assert half.normal == pytest.approx(0.5 * (lox.normal + geo.normal), rel=1e-14)
    with pytest.raises(CableInputError):
        curvature_coefficients(a, 1.2)


def _clamped_model(mu: float, tension: float, a: float, r: float, section=STRIP):
    """Tension-clamped outer wire: capacity mu N sin^2 a / r, no torsion terms."""
    return TransverseSlipModel(
        section, E, 1e-9, r, a, tension, mu * tension * math.sin(a) ** 2 / r
    )


def test_jenkin_amplitude_is_saevik_eq_4_1():
    a, r, tension, mu = math.radians(35.0), 0.1, 150e6 * 18e-6, 0.05
    thin = WireSection(shape="rectangular", width_m=0.006, thickness_m=1e-9)
    model = _clamped_model(mu, tension, a, r, thin)
    assert model.frictionless_fraction == pytest.approx(1.0, abs=1e-9)
    for kappa in (0.25, 0.5, 1.0):  # above the Eq. 4.2 threshold
        ramp = np.linspace(0.0, kappa, 400)
        v, _q, _kl = model.slip_history(ramp[None, :])
        eta = v[0, -1] / (model.geodesic_offset_m2 * kappa)
        # Sævik's single-sine Galerkin estimate carries 4/pi on the friction
        # term; the point-wise Jenkin amplitude does not.
        expected = saevik_transverse_fraction(
            mu * math.pi / 4, tension, E, thin.I_lateral, r, a, kappa
        )
        assert eta == pytest.approx(expected, abs=1e-9)
        assert 0.0 < eta < 1.0


def test_frictionless_slip_reaches_the_geodesic_and_cancels_lateral_curvature():
    a, r = math.radians(30.0), 0.1
    model = TransverseSlipModel(STRIP, E, G, r, a, 2.0e4, 0.0)
    p = np.linspace(0.0, 0.2, 50)
    v, q, kl = model.slip_history(p[None, :])
    eta0 = model.frictionless_fraction
    assert 0.9 < eta0 < 1.0
    assert np.allclose(v[0], eta0 * model.geodesic_offset_m2 * p, rtol=1e-12, atol=0)
    assert np.allclose(q, 0.0, atol=1e-9)
    lox = curvature_coefficients(a, 0.0).lateral
    assert np.allclose(kl[0], (1 - eta0) * lox * p, rtol=1e-10, atol=1e-15)
    assert model.geodesic_offset_m2 == pytest.approx(
        abs(transverse_geodesic_factor(r, a)), rel=1e-15
    )


def test_stuck_wire_keeps_the_loxodrome():
    a, r = math.radians(25.0), 0.132
    model = TransverseSlipModel(STRIP, E, G, r, a, 1.7e4, 1.0e9)
    p = 0.15 * np.sin(np.linspace(0, 4 * math.pi, 200))
    v, q, kl = model.slip_history(p[None, :])
    assert np.all(v == 0.0)
    assert np.allclose(kl[0], curvature_coefficients(a, 0.0).lateral * p, rtol=1e-14)


def test_relief_factor_bounds_and_identity():
    a, r = math.radians(24.8), 0.132
    model = TransverseSlipModel(STRIP, E, G, r, a, 1.7e4, 1.3e4)
    relief = LateralRelief(model)
    theta = np.linspace(0, 2 * math.pi, 73)
    g = relief.factor(theta, 0.146)
    assert np.all(g <= 1.0) and np.all(g >= model.relief_floor)
    assert g[0] == 1.0  # bending plane, no gradient: no demand
    huge = LateralRelief(TransverseSlipModel(STRIP, E, G, r, a, 1.7e4, 1e30))
    assert np.all(huge.factor(theta, 0.146, 0.5) == 1.0)
    weak = LateralRelief(TransverseSlipModel(STRIP, E, G, r, a, 1.7e4, 1.0))
    assert np.allclose(weak.factor(theta[1:18], 0.146), model.relief_floor)
    floor = 1 / math.sqrt(1 + geodesic_slip_ratio(a) ** 2)
    assert model.relief_floor == pytest.approx(floor, rel=1e-15)
    anchored = LateralRelief(model, anchor_layer_m=0.05)
    assert np.all(anchored.factor(theta, 0.01, 0.0, 0.01) == model.relief_floor)
    assert np.array_equal(
        anchored.factor(theta, 0.01, 0.0, 1.0), relief.factor(theta, 0.01)
    )


@pytest.mark.parametrize(("mu", "published"), [(0.025, 21.509e6), (0.1, 33.573e6)])
def test_dong_2018_table_1(mu, published):
    out = dong_anchor_tension(
        STRIP, 210e9, 210e9 / 2.6, 0.10, math.radians(35.0), 0.001, mu, mu, J_m4=J_ROARK
    )
    assert out["edge_stress_Pa"] == pytest.approx(published, rel=1e-3)
    assert out["decay_length_m"] == pytest.approx(0.0634, rel=2e-3)


@pytest.mark.parametrize(("mu", "published"), [(0.025, 35.463e6), (0.1, 55.624e6)])
def test_dong_2025_table_1(mu, published):
    out = dong_anchor_bending(
        STRIP,
        210e9,
        210e9 / 2.6,
        0.10,
        math.radians(35.0),
        0.001,
        0.005,
        math.pi,
        mu,
        mu,
        2.55e6,
        J_m4=J_ROARK,
    )
    assert out["anchor_edge_stress_Pa"] == pytest.approx(published, rel=1e-3)
    s, stress = out["s_m"], out["edge_stress_Pa"]
    # far from the anchor: the loxodromic lateral stress, held by friction
    lox = 210e9 * 0.003 * 0.005 * curvature_coefficients(math.radians(35.0)).lateral
    assert float(np.max(np.abs(stress[s > 0.5 * s[-1]]))) == pytest.approx(
        lox, rel=0.02
    )
