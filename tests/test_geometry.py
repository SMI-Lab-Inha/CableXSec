# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for helical / nested-helical kinematics.

Includes Sprint 1 gates:
  * A1 — straight-cable axial strain, error < 1e-12.
  * R2 G1 — nested helix with alpha_s = r_s = 0 reduces to single helix, <1e-12.
  * R2 G2 — sum of strand axial forces equals conductor macro axial force, <0.5%.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cablexsec.mechanics.geometry import HelixGeometry, NestedHelixGeometry

# Test-taxonomy category.
pytestmark = pytest.mark.analytical_sanity


def make_helix() -> HelixGeometry:
    return HelixGeometry(
        n_wires=56,
        r_helix_m=0.080,
        lay_angle_rad=math.radians(22.0),
        d_wire_m=0.005,
        lay_direction="right_hand",
    )


# --------------------------------------------------------------------------- #
# Single-helix basics                                                         #
# --------------------------------------------------------------------------- #
def test_wire_azimuths_uniform() -> None:
    h = make_helix()
    phi = h.wire_azimuths()
    assert phi.shape == (56,)
    assert phi[0] == 0.0
    assert np.allclose(np.diff(phi), 2.0 * np.pi / 56)


def test_geometry_helpers() -> None:
    h = make_helix()
    assert h.wire_area_m2 == pytest.approx(math.pi * 0.005**2 / 4)
    assert h.pitch_length_m == pytest.approx(
        2 * math.pi * 0.080 / math.tan(math.radians(22.0))
    )
    assert h.lay_sign == 1.0
    assert HelixGeometry(1, 0.08, 0.3, 0.005, "left_hand").lay_sign == -1.0


def test_invalid_construction_raises() -> None:
    with pytest.raises(ValueError, match="n_wires"):
        HelixGeometry(0, 0.08, 0.3, 0.005)
    with pytest.raises(ValueError, match="lay_direction"):
        HelixGeometry(7, 0.08, 0.3, 0.005, "diagonal")


# --------------------------------------------------------------------------- #
# Tier-1 analytical cases A1, B, C                                            #
# --------------------------------------------------------------------------- #
def test_A1_straight_cable_axial_strain() -> None:
    """A1: straight cable (kappa=0, tau=0) -> eps_w = eps_s cos^2(a), all phi."""
    h = make_helix()
    eps_s = 1.5e-3
    phi = h.wire_azimuths()
    eps = h.axial_strain(eps_s, 0.0, 0.0, 0.0, phi)
    expected = eps_s * math.cos(h.lay_angle_rad) ** 2
    assert np.max(np.abs(eps - expected)) < 1e-12


def test_B_pure_bending_phi_zero() -> None:
    """B: pure bending, phi=0 -> eps_w = eps_s cos^2(a) + kappa r cos^2(a)."""
    h = make_helix()
    eps_s, kappa = 1.0e-3, 0.05
    eps = float(h.axial_strain(eps_s, kappa, 0.0, 0.0, 0.0))
    a = h.lay_angle_rad
    expected = eps_s * math.cos(a) ** 2 + kappa * h.r_helix_m * math.cos(a) ** 2
    assert eps == pytest.approx(expected, abs=1e-15)


def test_C_pure_bending_phi_half_pi() -> None:
    """C: pure bending about x, phi=pi/2 -> curvature term vanishes."""
    h = make_helix()
    eps_s, kappa = 1.0e-3, 0.05
    eps = float(h.axial_strain(eps_s, kappa, 0.0, 0.0, math.pi / 2))
    expected = eps_s * math.cos(h.lay_angle_rad) ** 2
    assert abs(eps - expected) < 1e-12


def test_biaxial_equals_scalar_form() -> None:
    """kx cos(phi) + ky sin(phi) == |kappa| cos(phi - phi_bend)."""
    h = make_helix()
    kx, ky = 0.03, 0.04
    kappa = math.hypot(kx, ky)
    phi_bend = math.atan2(ky, kx)
    phi = h.wire_azimuths()
    biaxial = h.axial_strain(1e-3, kx, ky, 0.0, phi)
    scalar = h.axial_strain(
        1e-3, kappa * math.cos(phi_bend), kappa * math.sin(phi_bend), 0.0, phi
    )
    assert np.allclose(biaxial, scalar, atol=1e-15)


def test_torsion_sign_follows_lay_direction() -> None:
    rh = HelixGeometry(7, 0.08, math.radians(20), 0.005, "right_hand")
    lh = HelixGeometry(7, 0.08, math.radians(20), 0.005, "left_hand")
    tau = 0.02
    e_rh = float(rh.axial_strain(0.0, 0.0, 0.0, tau, 0.0))
    e_lh = float(lh.axial_strain(0.0, 0.0, 0.0, tau, 0.0))
    assert e_rh == pytest.approx(-e_lh)
    assert e_rh != 0.0


# --------------------------------------------------------------------------- #
# Menard bending stress kernel                                                #
# --------------------------------------------------------------------------- #
def test_bending_stress_peaks_in_bending_plane() -> None:
    h = make_helix()
    E = 210.0e9
    kappa = 0.05
    a = h.lay_angle_rad
    peak = float(h.bending_stress(kappa, 0.0, E, phi_bend=0.0))
    expected = E * kappa * math.cos(a) * (1 + math.sin(a) ** 2) * (h.d_wire_m / 2)
    assert peak == pytest.approx(expected)
    # 90 degrees out of the bending plane -> zero bending stress
    # (residual is cos(pi/2) rounding, ~1e-16 relative to the peak).
    assert abs(float(h.bending_stress(kappa, math.pi / 2, E, 0.0))) < 1e-9 * peak


# --------------------------------------------------------------------------- #
# R2 nested-helix gates G1, G2                                                #
# --------------------------------------------------------------------------- #
def test_G1_nested_reduces_to_single_helix() -> None:
    """G1: alpha_s = 0 and r_s = 0 -> strand strain == conductor strain, <1e-12."""
    alpha_c = math.radians(12.0)
    nested = NestedHelixGeometry(
        r_cable_m=0.050,
        lay_angle_cable_rad=alpha_c,
        n_strands=7,
        r_strand_m=0.0,
        lay_angle_strand_rad=0.0,
        d_strand_m=0.002,
    )
    single = HelixGeometry(
        n_wires=1, r_helix_m=0.050, lay_angle_rad=alpha_c, d_wire_m=0.0
    )
    eps_s, kx, ky, tau, phi_c = 1.2e-3, 0.04, 0.01, 0.005, 0.7
    single_eps = float(single.axial_strain(eps_s, kx, ky, tau, phi_c))
    for phi_s in (0.0, 1.0, 2.5, math.pi):
        strand_eps = float(nested.strand_axial_strain(eps_s, kx, ky, tau, phi_c, phi_s))
        assert abs(strand_eps - single_eps) < 1e-12


def test_G2_strand_axial_force_sums_to_macro() -> None:
    """G2: sum of strand axial-force components == macro axial force, <0.5%."""
    alpha_s = math.radians(9.0)
    nested = NestedHelixGeometry(
        r_cable_m=0.050,
        lay_angle_cable_rad=math.radians(12.0),
        n_strands=19,
        r_strand_m=0.004,
        lay_angle_strand_rad=alpha_s,
        d_strand_m=0.002,
    )
    E = 95.0e9
    A_s = nested.strand_area_m2
    eps_s, kappa, phi_c = 1.0e-3, 0.08, 0.4

    eps_cond = nested.conductor_axial_strain(eps_s, kappa, 0.0, 0.0, phi_c)
    phi_strands = nested.strand_azimuths()
    eps_strands = nested.strand_axial_strain(
        eps_s, kappa, 0.0, 0.0, phi_c, phi_strands, kappa=kappa, phi_bend=0.0
    )
    # Strand axial force projected onto the conductor axis (factor cos a_s).
    n_sum = float(np.sum(E * A_s * eps_strands * math.cos(alpha_s)))
    # Consistent macro stiffness from the same strand population (Costello).
    ea_macro = nested.n_strands * E * A_s * math.cos(alpha_s) ** 3
    macro_force = ea_macro * eps_cond
    rel_err = abs(n_sum - macro_force) / abs(macro_force)
    assert rel_err < 0.5e-2


def test_invalid_nested_construction_raises() -> None:
    with pytest.raises(ValueError, match="n_strands"):
        NestedHelixGeometry(0.05, 0.2, 0, 0.004, 0.15, 0.002)
