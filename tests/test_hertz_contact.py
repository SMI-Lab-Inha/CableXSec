# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for elliptical Hertz contact (Johnson §4.2), incl. gate A3."""

from __future__ import annotations

import math

import pytest

from cablexsec.mechanics.hertz_contact import (
    crossed_cylinder_curvatures,
    layer_crossing_angle_rad,
    layer_crossing_density_per_m,
    reduced_modulus,
    reduced_shear_modulus,
    solve_contact,
)

# Steel-on-steel reference.
E_STEEL, NU_STEEL = 210.0e9, 0.30


# Test-taxonomy category.
pytestmark = pytest.mark.analytical_sanity


def test_A3_perpendicular_crossing_recovers_circular_hertz() -> None:
    """A3: perpendicular crossing (A=B) -> circular contact = sphere Hertz, <0.1%."""
    d = 0.005
    estar = reduced_modulus(E_STEEL, NU_STEEL, E_STEEL, NU_STEEL)
    A, B = crossed_cylinder_curvatures(d, math.pi / 2)
    assert pytest.approx(B) == A  # perpendicular -> equal curvatures
    F = 500.0
    res = solve_contact(F, A, B, estar)
    # Circular: a == b, and a == (3 F R / 4E*)^(1/3) with R = 1/(A+B).
    assert res.aspect_ratio == pytest.approx(1.0, abs=1e-6)
    R = 1.0 / (A + B)
    a_sphere = (3.0 * F * R / (4.0 * estar)) ** (1.0 / 3.0)
    assert res.a_m == pytest.approx(a_sphere, rel=1e-3)
    # p0 = 3F/(2 pi a^2) for the circle.
    assert res.p0_Pa == pytest.approx(3.0 * F / (2.0 * math.pi * a_sphere**2), rel=1e-3)
    assert res.eccentricity == pytest.approx(0.0, abs=1e-4)


def test_johnson_worked_example_45deg_crossed_cylinders() -> None:
    """Johnson p.97: cylinders crossing at 45 deg -> a/b ~ 3.2, R_e = sqrt(2) r."""
    r = 0.01  # cylinder radius
    d = 2.0 * r
    A, B = crossed_cylinder_curvatures(d, math.radians(45.0))
    # (R''/R')^(1/2) = 2.41 ; R_e = sqrt(2) r
    Rp = 1.0 / (2.0 * B)
    Rpp = 1.0 / (2.0 * A)
    assert math.sqrt(Rpp / Rp) == pytest.approx(2.414, rel=2e-3)
    assert math.sqrt(Rp * Rpp) == pytest.approx(math.sqrt(2.0) * r, rel=1e-3)
    estar = reduced_modulus(E_STEEL, NU_STEEL, E_STEEL, NU_STEEL)
    res = solve_contact(300.0, A, B, estar)
    assert 3.0 <= res.aspect_ratio <= 3.4  # Johnson Fig 4.4 ~3.18-3.25


def test_load_balance_pressure_identity() -> None:
    """P = (2/3) p0 pi a b  must hold exactly (Johnson 4.27)."""
    estar = reduced_modulus(E_STEEL, NU_STEEL, E_STEEL, NU_STEEL)
    A, B = crossed_cylinder_curvatures(0.005, math.radians(40.0))
    F = 250.0
    res = solve_contact(F, A, B, estar)
    assert (2.0 / 3.0) * res.p0_Pa * math.pi * res.a_m * res.b_m == pytest.approx(F)


def test_tau_max_is_031_p0() -> None:
    estar = reduced_modulus(E_STEEL, NU_STEEL, E_STEEL, NU_STEEL)
    A, B = crossed_cylinder_curvatures(0.005, math.pi / 2)
    res = solve_contact(400.0, A, B, estar)
    assert res.tau_max_Pa == pytest.approx(0.31 * res.p0_Pa)


def test_tangential_stiffness_circular_limit() -> None:
    """K_t = 8 G* a; equal materials => K_t = 4 G a /(2-nu) (Johnson 7.44)."""
    estar = reduced_modulus(E_STEEL, NU_STEEL, E_STEEL, NU_STEEL)
    gstar = reduced_shear_modulus(E_STEEL, NU_STEEL, E_STEEL, NU_STEEL)
    A, B = crossed_cylinder_curvatures(0.005, math.pi / 2)
    res = solve_contact(400.0, A, B, estar, G_star=gstar)
    G = E_STEEL / (2.0 * (1.0 + NU_STEEL))
    expected = 4.0 * G * res.a_m / (2.0 - NU_STEEL)
    assert res.K_t_N_per_m == pytest.approx(expected, rel=1e-9)


def test_higher_load_gives_larger_contact_and_pressure() -> None:
    estar = reduced_modulus(E_STEEL, NU_STEEL, E_STEEL, NU_STEEL)
    A, B = crossed_cylinder_curvatures(0.005, math.radians(35.0))
    lo = solve_contact(100.0, A, B, estar)
    hi = solve_contact(800.0, A, B, estar)
    assert hi.a_m > lo.a_m and hi.b_m > lo.b_m and hi.p0_Pa > lo.p0_Pa


def test_layer_crossing_angle_helper() -> None:
    th = layer_crossing_angle_rad(math.radians(16.0), math.radians(16.0))
    assert th == pytest.approx(math.radians(32.0))


def test_opposite_lay_crossing_density() -> None:
    density = layer_crossing_density_per_m(
        60,
        0.08,
        math.radians(16.0),
        1.0,
        0.085,
        math.radians(16.0),
        -1.0,
    )
    expected = (
        60.0
        / (2.0 * math.pi)
        * (math.tan(math.radians(16.0)) / 0.08 + math.tan(math.radians(16.0)) / 0.085)
    )
    assert density == pytest.approx(expected)


def test_invalid_inputs_raise() -> None:
    estar = reduced_modulus(E_STEEL, NU_STEEL, E_STEEL, NU_STEEL)
    with pytest.raises(ValueError, match="force"):
        solve_contact(-1.0, 1.0, 2.0, estar)
