# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Hertz geometry must fit the elastic displacement, not just balance force."""

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ellipe, ellipkm1

from cablexsec.mechanics.hertz_contact import (
    crossed_cylinder_curvatures,
    reduced_modulus,
    reduced_shear_modulus,
    solve_contact,
)


@pytest.mark.parametrize("ratio", [1, 1 + 1e-8, 1 + 1e-6, 1.01, 2, 3, 5, 10, 100])
def test_independent_displacement_integrals_recover_both_gap_curvatures(ratio):
    """Greenwood (2018), eqs 8-10: w = delta - A*x*x - B*y*y.

    Direct quadrature after sqrt(s)=a*t; no Carlson routines, eccentricity
    root or ellipse shape factor are used in the reference recovery.
    """
    force, elastic, a_curv = 17.0, 115e9, 230.0
    out = solve_contact(force, a_curv, ratio * a_curv, elastic)
    k = out.b_m / out.a_m
    ix = quad(
        lambda t: 1 / ((1 + t * t) ** 1.5 * math.sqrt(k * k + t * t)),
        0,
        math.inf,
        epsabs=1e-12,
        epsrel=1e-12,
    )[0]
    iy = quad(
        lambda t: 1 / ((k * k + t * t) ** 1.5 * math.sqrt(1 + t * t)),
        0,
        math.inf,
        epsabs=1e-12,
        epsrel=1e-12,
    )[0]
    factor = out.p0_Pa * k / (elastic * out.a_m)
    np.testing.assert_allclose(
        [factor * ix, factor * iy], [a_curv, ratio * a_curv], rtol=2e-11
    )
    assert 2 * math.pi * out.a_m * out.b_m * out.p0_Pa / 3 == pytest.approx(force)


def test_published_ellipticity_iteration_example():
    # Greenwood (2018), immediately after eq 18: R1/R2=5 -> a/b=2.890181...
    out = solve_contact(1, 1, 5, 1)
    assert out.aspect_ratio == pytest.approx(2.890181, abs=5e-7)


@pytest.mark.parametrize("ratio", [1e8, 1e12, 1e14, 1e20])
def test_shallow_crossings_preserve_axis_ratio_without_eccentricity_roundtrip(ratio):
    out = solve_contact(10, 100 / ratio, 100, 100e9)
    q = (out.b_m / out.a_m) ** 2
    # Independent Legendre form, retaining the complementary parameter q.
    kk, ee = ellipkm1(q), ellipe(1 - q)
    recovered_ratio = (ee / q - kk) / (kk - ee)
    assert recovered_ratio == pytest.approx(ratio, rel=2e-12)
    assert out.a_m > out.b_m > 0 and math.isfinite(out.p0_Pa)


@pytest.mark.parametrize("scale", [1e-150, 1e-50, 1, 1e50, 1e150])
def test_similar_geometry_force_scaling_preserves_pressure(scale):
    # Length scales by s, force by s^2, curvature by 1/s; stress stays fixed.
    base = solve_contact(3, 1, 3, 17, 7)
    out = solve_contact(3 * scale**2, 1 / scale, 3 / scale, 17, 7)
    np.testing.assert_allclose(
        [out.a_m / scale, out.b_m / scale, out.p0_Pa, out.K_t_N_per_m / scale],
        [base.a_m, base.b_m, base.p0_Pa, base.K_t_N_per_m],
        rtol=2e-12,
    )


@pytest.mark.parametrize("parameter", range(4))
@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf])
def test_invalid_normal_contact_inputs_are_rejected(parameter, value):
    values = [10, 100, 300, 100e9]
    values[parameter] = value
    with pytest.raises(ValueError):
        solve_contact(*values)


@pytest.mark.parametrize("value", [-1, math.nan, math.inf])
def test_invalid_shear_modulus_is_rejected(value):
    with pytest.raises(ValueError):
        solve_contact(10, 100, 300, 100e9, value)


def test_zero_or_omitted_tangential_modulus_has_zero_stiffness():
    assert solve_contact(10, 100, 300, 100e9).K_t_N_per_m == 0
    assert solve_contact(10, 100, 300, 100e9, 0).K_t_N_per_m == 0


@pytest.mark.parametrize("function", [reduced_modulus, reduced_shear_modulus])
@pytest.mark.parametrize(
    "values",
    [
        (0, 0.3, 2, 0.3),
        (-1, 0.3, 2, 0.3),
        (math.nan, 0.3, 2, 0.3),
        (1, 0.3, math.inf, 0.3),
        (1, -1, 2, 0.3),
        (1, 0.3, 2, 0.51),
        (1, math.nan, 2, 0.3),
    ],
)
def test_invalid_isotropic_materials_are_rejected(function, values):
    with pytest.raises(ValueError):
        function(*values)


@pytest.mark.parametrize("nu", [-0.99, 0, 0.3, 0.5])
def test_material_compliance_symmetry_and_elastic_limits(nu):
    elastic = 200e9
    shear = elastic / (2 * (1 + nu))
    assert reduced_modulus(elastic, nu, elastic, nu) == pytest.approx(
        elastic / (2 * (1 - nu * nu))
    )
    assert reduced_shear_modulus(elastic, nu, elastic, nu) == pytest.approx(
        shear / (2 * (2 - nu))
    )
    assert reduced_modulus(elastic, nu, 1e9, 0.2) == reduced_modulus(
        1e9, 0.2, elastic, nu
    )


@pytest.mark.parametrize("diameter", [1e-300, 0.003, 1e300])
def test_cylinder_geometry_stays_scale_invariant(diameter):
    a, b = crossed_cylinder_curvatures(diameter, math.pi / 3)
    np.testing.assert_allclose([a * diameter, b * diameter], [0.5, 1.5], rtol=2e-15)


def test_parallel_cylinders_have_zero_minor_curvature_and_no_point_contact():
    for angle in [0, math.pi, -math.pi, 2 * math.pi]:
        a, b = crossed_cylinder_curvatures(0.003, angle)
        assert a == 0
        with pytest.raises(ValueError):
            solve_contact(1, a, b, 100e9)


def test_unrepresentable_outputs_fail_instead_of_returning_infinity_or_zero():
    with pytest.raises(ValueError):
        solve_contact(1e300, 1e-320, 1e-320, 1e-320)
    with pytest.raises(ValueError):
        crossed_cylinder_curvatures(5e-324, math.pi / 2)


def test_large_intermediates_do_not_discard_representable_circular_solution():
    out = solve_contact(1e300, 1e-300, 1e-300, 1e-300)
    assert out.a_m / 1e300 == pytest.approx((3 / 8) ** (1 / 3), rel=2e-13)
    assert out.p0_Pa / 1e-300 == pytest.approx(0.9181645931463445, rel=2e-13)


def test_swapping_curvatures_preserves_the_sorted_contact_solution():
    assert solve_contact(10, 100, 300, 100e9, 40e9) == solve_contact(
        10, 300, 100, 100e9, 40e9
    )


def test_curvature_contrast_outside_carlson_range_is_explicitly_rejected():
    with pytest.raises(ValueError, match="aspect ratio"):
        solve_contact(1, 1e-300, 1e300, 1)


def test_extreme_diameter_contrast_retains_representable_eigenvalues():
    a, b = crossed_cylinder_curvatures(1e-300, math.pi / 3, partner_diameter_m=1e300)
    assert a * b == pytest.approx(0.75)
    assert b == pytest.approx(1e300)
