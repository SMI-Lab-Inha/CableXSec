# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for the CMR fretting model (R1.3): gates A4, A5, A10."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cablexsec.mechanics.mindlin_contact import (
    MindlinContact,
    _trajectory_diameter,
    geometric_slip_factor,
    loxodromic_slip_increment,
    mindlin_monotonic,
    slip_increment,
    stuck_zone_ratio,
)

# --------------------------------------------------------------------------- #
# A4 — Mindlin monotonic                                                      #
# --------------------------------------------------------------------------- #
# Test-taxonomy category.
pytestmark = pytest.mark.analytical_sanity


def test_A4_mindlin_initial_slope_is_Kt() -> None:
    """A4: dF_t/d(delta) at delta=0 equals K_t (<0.1%)."""
    mu, F_n, K_t = 0.3, 100.0, 5.0e6
    h = 1e-9
    slope = float(mindlin_monotonic(h, mu, F_n, K_t)) / h
    assert slope == pytest.approx(K_t, rel=1e-3)


def test_A4_reaches_gross_slip_at_delta_star() -> None:
    mu, F_n, K_t = 0.3, 100.0, 5.0e6
    delta_star = 3.0 * mu * F_n / (2.0 * K_t)
    assert float(mindlin_monotonic(delta_star, mu, F_n, K_t)) == pytest.approx(mu * F_n)
    # beyond delta* -> saturates at mu F_n (gross slip)
    assert float(mindlin_monotonic(2 * delta_star, mu, F_n, K_t)) == pytest.approx(
        mu * F_n
    )


def test_A4_monotonic_increasing() -> None:
    mu, F_n, K_t = 0.25, 200.0, 1.0e7
    d = np.linspace(0, 3.0 * mu * F_n / (2.0 * K_t), 50)
    f = mindlin_monotonic(d, mu, F_n, K_t)
    assert np.all(np.diff(f) >= -1e-12)


def test_stuck_zone_ratio_limits() -> None:
    assert stuck_zone_ratio(0.0, 100.0, 0.3) == pytest.approx(1.0)  # full stick
    assert stuck_zone_ratio(30.0, 100.0, 0.3) == pytest.approx(0.0)  # gross slip


# --------------------------------------------------------------------------- #
# A10 — rotation invariance theorem                                           #
# --------------------------------------------------------------------------- #
def test_A10_rotation_invariance() -> None:
    """A10: |d delta_t|^2 = c_geo^2 (d_kx^2 + d_ky^2), independent of phi, <1e-12."""
    c_geo = geometric_slip_factor(0.08, math.radians(16.0))
    dkx, dky = 0.013, -0.027
    target = c_geo**2 * (dkx**2 + dky**2)
    for phi in np.linspace(0, 2 * np.pi, 37):
        d = slip_increment(dkx, dky, phi, c_geo, transverse_map="legacy")
        assert float(d @ d) == pytest.approx(target, rel=1e-12)


def test_geometric_slip_factor_matches_loxodromic_scale() -> None:
    radius = 0.08
    alpha = math.radians(16.0)
    expected = radius**2 * math.cos(alpha) ** 2 / math.sin(alpha)
    assert geometric_slip_factor(radius, alpha) == pytest.approx(expected)
    with pytest.raises(ValueError, match="non-zero helical lay angle"):
        geometric_slip_factor(radius, 0.0)


def test_A10_uniform_around_cross_section() -> None:
    """The proposed vector map has azimuth-independent increment magnitude."""
    c_geo = geometric_slip_factor(0.08, math.radians(16.0))
    phi = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    d = slip_increment(0.02, 0.01, phi, c_geo, transverse_map="legacy")
    mag = np.linalg.norm(d, axis=-1)
    assert mag.std() < 1e-12 * mag.mean()


def test_A10_matched_fixed_plane_vector_loxodromic_contrast() -> None:
    """One load history isolates the transverse-component constraint."""
    c_geo = geometric_slip_factor(0.08, math.radians(16.0))
    phi = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])
    vector = slip_increment(0.02, 0.0, phi, c_geo, transverse_map="legacy")
    loxodromic = loxodromic_slip_increment(0.02, 0.0, phi, c_geo)
    vector_mag = np.linalg.norm(vector, axis=-1)
    loxodromic_mag = np.linalg.norm(loxodromic, axis=-1)
    assert np.ptp(vector_mag) < 1e-12 * vector_mag.mean()
    # Loxodromic displacement is zero at the extreme fibres and maximal at NA.
    assert loxodromic_mag[[0, 2]] == pytest.approx(np.zeros(2), abs=1e-18)
    assert loxodromic_mag[[1, 3]] == pytest.approx(vector_mag[[1, 3]])


def test_eta_zero_is_exact_loxodromic_limit() -> None:
    c_geo = geometric_slip_factor(0.08, math.radians(16.0))
    phi = np.linspace(0.0, 2.0 * np.pi, 37)
    family = slip_increment(0.02, -0.01, phi, c_geo, transverse_ratio=0.0)
    lox = loxodromic_slip_increment(0.02, -0.01, phi, c_geo)
    assert np.allclose(family, lox)


def test_A10_full_isotropic_trajectory_is_rotation_equivariant() -> None:
    """Equal contact states give equal nonlinear Mindlin amplitudes at any azimuth."""
    c_geo = geometric_slip_factor(0.08, math.radians(16.0))
    theta = np.linspace(0.0, 12.0 * np.pi, 1200)
    kx = 0.02 * np.sin(theta) + 0.006 * np.sin(0.37 * theta)
    ky = 0.013 * np.cos(0.83 * theta)
    contacts = [MindlinContact(K_t=5.0e6, mu=0.3) for _ in range(2)]
    phis = (0.13, 1.71)
    previous = np.zeros(2)
    for current in np.column_stack((kx, ky)):
        delta = current - previous
        previous = current
        for contact, phi in zip(contacts, phis, strict=True):
            contact.update(
                slip_increment(delta[0], delta[1], phi, c_geo, transverse_map="legacy"),
                80.0,
            )
    assert contacts[0].fretting_amplitude == pytest.approx(
        contacts[1].fretting_amplitude, rel=1e-12
    )


def test_A10_uniform_damage_requires_uniform_contact_state() -> None:
    """Different normal-force histories break amplitude uniformity, as scoped."""
    contacts = [MindlinContact(K_t=5.0e6, mu=0.3) for _ in range(2)]
    for step in np.sin(np.linspace(0.0, 8.0 * np.pi, 800)) * 2.0e-6:
        contacts[0].update(np.array([step, 0.5 * step]), 20.0)
        contacts[1].update(np.array([step, 0.5 * step]), 200.0)
    assert contacts[0].fretting_amplitude != pytest.approx(
        contacts[1].fretting_amplitude
    )


# --------------------------------------------------------------------------- #
# A5 — Masing symmetry from the incremental element                           #
# --------------------------------------------------------------------------- #
def test_A5_masing_symmetric_cycle() -> None:
    """A5: a symmetric 1-D tangential cycle gives an antisymmetric F_t loop, <1e-10."""
    mu, F_n, K_t = 0.3, 100.0, 5.0e6
    c = MindlinContact(K_t=K_t, mu=mu)
    amp = 3.0 * mu * F_n / K_t  # drive well into gross slip
    n = 400
    # ramp 0 -> +amp -> -amp -> +amp (one full symmetric cycle after the first ramp)
    forces, deltas = [], []
    pos = 0.0
    segs = [(+amp, n), (-2 * amp, 2 * n), (+2 * amp, 2 * n)]
    for total, steps in segs:
        for _ in range(steps):
            pos += total / steps
            f_t, _, _ = c.update(np.array([total / steps, 0.0]), F_n)
            forces.append(float(f_t[0]))
            deltas.append(pos)
    forces_a = np.array(forces)
    deltas_a = np.array(deltas)
    # Antisymmetry of the steady loop: F_t(+delta) = -F_t(-delta) at the extremes.
    assert forces_a.max() == pytest.approx(mu * F_n, rel=1e-10)
    assert forces_a.min() == pytest.approx(-mu * F_n, rel=1e-10)
    # symmetric drive range
    assert deltas_a.max() == pytest.approx(amp, abs=1e-9)
    assert deltas_a.min() == pytest.approx(-amp, abs=1e-9)


# --------------------------------------------------------------------------- #
# Fretting amplitude + shear-slip product                                     #
# --------------------------------------------------------------------------- #
def test_fretting_amplitude_full_stick_is_zero() -> None:
    """Below the slip threshold the slip trajectory is ~0 => delta* ~ 0."""
    c = MindlinContact(K_t=1.0e7, mu=0.3)
    # tiny oscillation, |F_t| never reaches mu F_n
    for _ in range(20):
        c.update(np.array([1e-9, 0.0]), 100.0)
        c.update(np.array([-1e-9, 0.0]), 100.0)
    assert c.fretting_amplitude < 1e-12


def test_fretting_amplitude_gross_slip() -> None:
    """In gross slip, delta* approaches half the total slip stroke."""
    mu, F_n, K_t = 0.3, 100.0, 5.0e6
    c = MindlinContact(K_t=K_t, mu=mu)
    stroke = 5.0 * mu * F_n / K_t
    step = stroke / 100
    for _ in range(100):
        c.update(np.array([step, 0.0]), F_n)
    for _ in range(100):
        c.update(np.array([-step, 0.0]), F_n)
    # slip trajectory spans roughly +/- (stroke - elastic part); delta* > 0
    assert c.fretting_amplitude > 0.4 * stroke
    c_p0 = MindlinContact(K_t=K_t, mu=mu)
    p0 = 50.0e6
    for i in range(100):
        # The peak pressure varies; Phi uses the largest value supplied.
        c_p0.update(np.array([step, 0.0]), F_n, p0_current=p0 if i == 40 else 2.0e7)
    phi = c_p0.shear_slip_product()
    assert phi > 0.0
    # Phi = mu p0 delta*, with p0 the largest peak pressure supplied.
    assert phi == pytest.approx(mu * p0 * c_p0.fretting_amplitude, rel=1e-12)
    # The legacy name is an alias of the same quantity, not a Ruiz parameter.
    assert c_p0.ruiz_parameter() == phi


def test_planar_diameter_matches_brute_force() -> None:
    """The scalable hull/calipers algorithm remains exact on irregular paths."""
    rng = np.random.default_rng(20260728)
    points = rng.normal(size=(200, 2))
    brute = np.max(np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1))
    assert _trajectory_diameter(points) == pytest.approx(brute, rel=1e-12)


def test_planar_diameter_scales_to_long_history() -> None:
    """A 100k-point storm trajectory runs without quadratic allocation."""
    theta = np.linspace(0.0, 30.0 * np.pi, 100_001)
    points = np.column_stack((2.0 * np.cos(theta), np.sin(theta)))
    assert _trajectory_diameter(points) == pytest.approx(4.0, rel=2e-8)
