# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Independent normal-contact stress checks, not cable fatigue qualification."""

import math
from dataclasses import replace

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize_scalar

from cablexsec.mechanics.hertz_contact import HertzContact, solve_contact
from cablexsec.mechanics.hertz_stress import compute_hertz_axis_stress


def contact(a=1.0, b=1.0, pressure=1.0):
    return HertzContact(a, b, pressure, 0, 0.31 * pressure, 0)


def boussinesq_integral(a, b, z, nu, order=128):
    """Direct pressure integration of point-force radial/hoop/vertical stresses.

    The classical Boussinesq radial stress convention is compressive positive;
    negate the result for the library convention. This does not use elliptic
    integrals or the production potential-function formulas. Integration uses
    x=a*sin(t)*cos(theta), y=b*sin(t)*sin(theta), with Hertz p=cos(t).
    Reference for radial point-load stress: Cambridge P2B2005QD005, data sheet.
    """
    nodes, weights = leggauss(order)
    t = (nodes + 1)[:, None] * math.pi / 4
    theta = (nodes + 1)[None, :] * math.pi
    x = a * np.sin(t) * np.cos(theta)
    y = b * np.sin(t) * np.sin(theta)
    rho2 = x * x + y * y
    radius = np.sqrt(rho2 + z * z)
    radial = (3 * rho2 * z / radius**5 - (1 - 2 * nu) / (radius * (radius + z))) / (
        2 * math.pi
    )
    hoop = (1 - 2 * nu) * (1 / (radius * (radius + z)) - z / radius**3) / (2 * math.pi)
    vertical = 3 * z**3 / (2 * math.pi * radius**5)
    pressure_area = a * b * np.sin(t) * np.cos(t) ** 2
    weighted = pressure_area * (weights[:, None] * weights[None, :]) * math.pi**2 / 4
    sigma_x = radial * x * x / rho2 + hoop * y * y / rho2
    sigma_y = radial * y * y / rho2 + hoop * x * x / rho2
    return -np.array(
        [np.sum(weighted * value) for value in (sigma_x, sigma_y, vertical)]
    )


@pytest.mark.parametrize("nu", [-0.3, 0, 0.3, 0.49, 0.5])
def test_circular_contact_matches_independent_closed_form(nu):
    depth = np.r_[0, np.geomspace(1e-5, 30, 60)]
    out = compute_hertz_axis_stress(contact(), depth, poisson_ratio=nu)
    radial = -((1 + nu) * (1 - depth * np.arctan2(1, depth)) - 0.5 / (1 + depth**2))
    vertical = -1 / (1 + depth**2)
    np.testing.assert_allclose(
        out.contact_stress_Pa[:, 0, 0], radial, rtol=1e-8, atol=2e-15
    )
    np.testing.assert_allclose(
        out.contact_stress_Pa[:, 1, 1], radial, rtol=1e-8, atol=2e-15
    )
    np.testing.assert_allclose(out.contact_stress_Pa[:, 2, 2], vertical, rtol=2e-15)


@pytest.mark.parametrize("a,b", [(1, 1), (2, 1), (1, 3), (5, 1)])
@pytest.mark.parametrize("nu", [0, 0.3, 0.5])
def test_ellipse_matches_independent_point_load_integration(a, b, nu):
    depths = min(a, b) * np.array([0.2, 0.8, 2.0])
    result = compute_hertz_axis_stress(contact(a, b), depths, poisson_ratio=nu)
    expected = np.array([boussinesq_integral(a, b, float(z), nu) for z in depths])
    actual = np.diagonal(result.contact_stress_Pa, axis1=1, axis2=2)
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-9)


def test_independent_quadrature_converges_for_elongated_contact():
    coarse = boussinesq_integral(5, 1, 0.2, 0.3, order=64)
    refined = boussinesq_integral(5, 1, 0.2, 0.3, order=192)
    fine = boussinesq_integral(5, 1, 0.2, 0.3, order=256)
    assert np.max(np.abs(refined - fine)) < np.max(np.abs(coarse - fine))
    np.testing.assert_allclose(refined, fine, rtol=1e-10, atol=1e-12)


def test_surface_axis_permutation_and_line_contact_limit():
    nu = 0.3
    result = compute_hertz_axis_stress(contact(3, 1), [0])
    np.testing.assert_allclose(
        np.diag(result.contact_stress_Pa[0]), [-(1 + 6 * nu) / 4, -(3 + 2 * nu) / 4, -1]
    )
    # Along the long contact direction the limit is plane strain: sigma_x=-2*nu*p0.
    line = compute_hertz_axis_stress(contact(1e8, 1), 0)
    np.testing.assert_allclose(
        np.diag(line.contact_stress_Pa[0]), [-0.6, -1, -1], atol=1e-8
    )


def test_axis_swap_preserves_invariants_and_swaps_normal_stresses():
    a = compute_hertz_axis_stress(contact(3, 1), [0, 0.2, 1, 3])
    b = compute_hertz_axis_stress(contact(1, 3), [0, 0.2, 1, 3])
    np.testing.assert_array_equal(
        a.contact_stress_Pa[:, 0, 0], b.contact_stress_Pa[:, 1, 1]
    )
    np.testing.assert_array_equal(a.von_mises_Pa, b.von_mises_Pa)
    np.testing.assert_array_equal(a.maximum_shear_Pa, b.maximum_shear_Pa)


def test_circular_peak_shear_and_von_mises_are_distinct_invariants():
    optimum = minimize_scalar(
        lambda z: -compute_hertz_axis_stress(contact(), z).von_mises_Pa[0],
        bounds=(0, 2),
        method="bounded",
        options={"xatol": 1e-12},
    )
    result = compute_hertz_axis_stress(contact(), optimum.x)
    assert optimum.x == pytest.approx(0.48086, abs=2e-5)
    assert result.maximum_shear_Pa[0] == pytest.approx(0.31002, abs=1e-5)
    assert result.von_mises_Pa[0] == pytest.approx(0.62004, abs=1e-5)
    assert result.sampled_peak_depth_m == optimum.x
    assert result.sampled_peak_von_mises_Pa == result.von_mises_Pa[0]


@pytest.mark.parametrize("scale", [1e-150, 1e-6, 1, 1e6, 1e150])
def test_stress_is_invariant_under_common_length_scaling(scale):
    depths = np.array([0, 0.1, 0.5, 1, 20])
    reference = compute_hertz_axis_stress(contact(2, 1, 1e9), depths)
    scaled = compute_hertz_axis_stress(contact(2 * scale, scale, 1e9), depths * scale)
    np.testing.assert_allclose(
        scaled.contact_stress_Pa, reference.contact_stress_Pa, rtol=2e-13, atol=1e-8
    )


def test_point_load_far_field_and_actual_hertz_contact_integration():
    solved = solve_contact(100, 50, 100, 100e9)
    depth = 1000 * solved.a_m
    result = compute_hertz_axis_stress(solved, depth)
    # At large distance the integrated pressure acts as the same total point force.
    expected = -3 * 100 / (2 * math.pi * depth**2)
    assert result.contact_stress_Pa[0, 2, 2] == pytest.approx(expected, rel=1e-6)


def test_zero_contact_pressure_recovers_supplied_pure_shear_tensor():
    shear = 123.0
    background = np.array([[0, shear, 0], [shear, 0, 0], [0, 0, 0]])
    out = compute_hertz_axis_stress(
        contact(pressure=0), [0, 1], background_stress_Pa=background
    )
    np.testing.assert_array_equal(out.contact_stress_Pa, 0)
    np.testing.assert_allclose(out.von_mises_Pa, math.sqrt(3) * shear)
    np.testing.assert_allclose(out.maximum_shear_Pa, shear)
    np.testing.assert_allclose(out.principal_stress_Pa, [[-shear, 0, shear]] * 2)


def test_uniform_hydrostatic_background_does_not_change_invariants():
    base = compute_hertz_axis_stress(contact(), [0, 0.4, 1])
    loaded = compute_hertz_axis_stress(
        contact(), [0, 0.4, 1], background_stress_Pa=17 * np.eye(3)
    )
    np.testing.assert_allclose(loaded.von_mises_Pa, base.von_mises_Pa, atol=2e-14)
    np.testing.assert_allclose(
        loaded.maximum_shear_Pa, base.maximum_shear_Pa, atol=2e-14
    )


def test_varying_background_is_added_at_the_same_depth_before_invariants():
    background = np.array([np.diag([1, 2, 3]), np.diag([3, 2, 1])])
    result = compute_hertz_axis_stress(
        contact(3, 1), [0, 1], background_stress_Pa=background
    )
    np.testing.assert_allclose(
        result.total_stress_Pa, result.contact_stress_Pa + background
    )
    expected = np.linalg.eigvalsh(result.contact_stress_Pa + background)
    np.testing.assert_array_equal(result.principal_stress_Pa, expected)


def test_outputs_are_readonly_and_do_not_alias_inputs():
    depth = np.array([0.0, 1.0])
    background = np.eye(3)
    result = compute_hertz_axis_stress(
        contact(), depth, background_stress_Pa=background
    )
    depth[:] = 2
    background[:] = 0
    np.testing.assert_array_equal(result.depth_m, [0, 1])
    assert result.total_stress_Pa[0, 2, 2] == 0
    for field in (
        result.depth_m,
        result.contact_stress_Pa,
        result.total_stress_Pa,
        result.principal_stress_Pa,
        result.von_mises_Pa,
        result.maximum_shear_Pa,
    ):
        with pytest.raises(ValueError):
            field[:] = 0


@pytest.mark.parametrize("depth", [[], [-1], [np.nan], [np.inf], [[0, 1]]])
def test_invalid_depths_fail(depth):
    with pytest.raises(ValueError, match="depth_m"):
        compute_hertz_axis_stress(contact(), depth)


@pytest.mark.parametrize("nu", [-1, -2, 0.5001, np.nan, np.inf])
def test_invalid_poisson_ratios_fail(nu):
    with pytest.raises(ValueError):
        compute_hertz_axis_stress(contact(), [0, 1], poisson_ratio=nu)


@pytest.mark.parametrize(
    "field,value",
    [("a_m", 0), ("b_m", -1), ("a_m", np.inf), ("p0_Pa", -1), ("p0_Pa", np.nan)],
)
def test_invalid_contact_inputs_fail(field, value):
    with pytest.raises(ValueError):
        compute_hertz_axis_stress(replace(contact(), **{field: value}), [0, 1])


@pytest.mark.parametrize(
    "background",
    [
        np.zeros(3),
        np.zeros((3, 3, 3)),
        np.full((3, 3), np.nan),
        np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
    ],
)
def test_invalid_background_tensors_fail(background):
    with pytest.raises(ValueError, match="background stress"):
        compute_hertz_axis_stress(contact(), [0, 1], background_stress_Pa=background)


def test_unrepresentable_geometry_and_stress_fail_explicitly():
    with pytest.raises(ValueError, match="floating-point"):
        compute_hertz_axis_stress(contact(1e-200, 1), 0)
    with pytest.raises(ValueError, match="floating-point"):
        compute_hertz_axis_stress(
            contact(pressure=1e308), 0, background_stress_Pa=-1e308 * np.eye(3)
        )
