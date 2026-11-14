# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Independent half-space mechanics and prescribed-traction field contracts."""

import math

import numpy as np
import pytest
from scipy.special import roots_jacobi, roots_legendre

from cablexsec.mechanics.hertz_contact import HertzContact
from cablexsec.mechanics.hertz_field import (
    _point_stress,
    compute_hertz_subsurface_stress,
)
from cablexsec.mechanics.hertz_stress import compute_hertz_axis_stress


def contact(a=1.0, b=1.0, pressure=1.0):
    return HertzContact(a, b, pressure, 0, 0.31 * pressure, 0)


def displacement(point, nu, force):
    """Johnson 3.18/3.75 displacements; G=1, independently differentiated."""
    x, y, z = point
    radius = np.sqrt(x * x + y * y + z * z)
    normal = np.array(
        [
            x * z / radius**3 - (1 - 2 * nu) * x / (radius * (radius + z)),
            y * z / radius**3 - (1 - 2 * nu) * y / (radius * (radius + z)),
            z * z / radius**3 + 2 * (1 - nu) / radius,
        ]
    )

    def shear(xx, yy):
        return np.array(
            [
                1 / radius
                + xx**2 / radius**3
                + (1 - 2 * nu)
                * (1 / (radius + z) - xx**2 / (radius * (radius + z) ** 2)),
                xx * yy / radius**3
                - (1 - 2 * nu) * xx * yy / (radius * (radius + z) ** 2),
                xx * z / radius**3 + (1 - 2 * nu) * xx / (radius * (radius + z)),
            ]
        )

    basis = (normal, shear(x, y), shear(y, x)[[1, 0, 2]])
    return basis[force] / (4 * np.pi)


def matrix(components):
    xx, yy, zz, xy, xz, yz = components
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


@pytest.mark.parametrize("nu", [-0.4, 0, 0.3, 0.49])
@pytest.mark.parametrize("force", [0, 1, 2])
def test_point_stress_matches_independent_displacement_derivative(nu, force):
    for point in np.array([[0.3, -0.4, 0.8], [0, 0, 0.7], [-1.2, 0.6, 0.1]]):
        gradient = np.column_stack(
            [
                displacement(point.astype(complex) + 1e-25j * direction, nu, force).imag
                / 1e-25
                for direction in np.eye(3)
            ]
        )
        expected = (
            gradient
            + gradient.T
            + np.eye(3) * (2 * nu / (1 - 2 * nu)) * np.trace(gradient)
        )
        actual = matrix(_point_stress(point[None, :], nu)[force, 0])
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-14)


@pytest.mark.parametrize("force", [0, 1, 2])
def test_point_kernel_equilibrium_and_surface_traction_away_from_load(force):
    point = np.array([0.8, -0.5, 0.3])
    step = 1e-5
    divergence = np.zeros(3)
    for j, direction in enumerate(np.eye(3)):
        plus = matrix(_point_stress((point + step * direction)[None], 0.3)[force, 0])
        minus = matrix(_point_stress((point - step * direction)[None], 0.3)[force, 0])
        divergence += (plus[:, j] - minus[:, j]) / (2 * step)
    np.testing.assert_allclose(divergence, 0, atol=3e-10)
    surface = matrix(_point_stress(np.array([[0.8, -0.5, 0]]), 0.3)[force, 0])
    np.testing.assert_array_equal(surface[:, 2], 0)


@pytest.mark.parametrize("axes", [(1, 1), (3, 1), (1, 2)])
@pytest.mark.parametrize("nu", [-0.2, 0.3, 0.5])
def test_normal_axis_matches_independent_elliptic_solution(axes, nu):
    footprint = contact(*axes, pressure=2e8)
    depth = min(axes) * np.array([0.2, 0.8, 2])
    points = np.column_stack((np.zeros_like(depth), np.zeros_like(depth), depth))
    result = compute_hertz_subsurface_stress(footprint, points, poisson_ratio=nu)
    reference = compute_hertz_axis_stress(footprint, depth, poisson_ratio=nu)
    np.testing.assert_allclose(
        result.normal_stress_Pa, reference.contact_stress_Pa, rtol=3e-8, atol=1e-5
    )
    np.testing.assert_array_equal(result.tangential_stress_Pa, 0)


@pytest.mark.parametrize("nu", [-0.3, 0.0, 0.3, 0.5])
def test_circular_shear_axis_matches_closed_radial_integral(nu):
    z = np.array([0.1, 0.5, 1, 3])
    ratios = np.array([0.4, -0.2])
    result = compute_hertz_subsurface_stress(
        contact(pressure=1e8),
        np.column_stack((0 * z, 0 * z, z)),
        poisson_ratio=nu,
        tangential_traction_ratio=ratios,
    )
    # Integrating Cerruti over azimuth leaves an elementary radial integral.
    shear = -1e8 * (1.5 * (1 - z * np.arctan(1 / z)) - 0.5 / (1 + z * z))
    expected = np.zeros((len(z), 3, 3))
    expected[:, 0, 2] = expected[:, 2, 0] = ratios[0] * shear
    expected[:, 1, 2] = expected[:, 2, 1] = ratios[1] * shear
    np.testing.assert_allclose(
        result.tangential_stress_Pa, expected, rtol=1e-9, atol=2e-7
    )


def test_axis_permutation_rotates_normal_and_shear_tensors():
    points = np.array([[0.4, -0.2, 0.3], [2.2, 1.3, 0.6]])
    swap = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    baseline = compute_hertz_subsurface_stress(
        contact(3, 1), points, tangential_traction_ratio=[0.3, -0.2], atol_Pa=1e-10
    )
    rotated = compute_hertz_subsurface_stress(
        contact(1, 3),
        points @ swap,
        tangential_traction_ratio=[-0.2, 0.3],
        atol_Pa=1e-10,
    )
    np.testing.assert_allclose(
        rotated.total_stress_Pa,
        swap @ baseline.total_stress_Pa @ swap,
        rtol=2e-7,
        atol=1e-9,
    )
    np.testing.assert_allclose(rotated.von_mises_Pa, baseline.von_mises_Pa, rtol=2e-7)


def test_shear_reversal_is_linear_and_normal_response_is_unchanged():
    points = [[0.5, 0.3, 0.5], [-0.7, 0.2, 0.3]]
    plus = compute_hertz_subsurface_stress(
        contact(2, 1), points, tangential_traction_ratio=[0.4, 0.2]
    )
    minus = compute_hertz_subsurface_stress(
        contact(2, 1), points, tangential_traction_ratio=[-0.4, -0.2]
    )
    np.testing.assert_array_equal(plus.normal_stress_Pa, minus.normal_stress_Pa)
    np.testing.assert_array_equal(
        plus.tangential_stress_Pa, -minus.tangential_stress_Pa
    )


@pytest.mark.parametrize("scale", [1e-150, 1e150])
def test_physical_length_scaling_preserves_stress(scale):
    points = np.array([[0.7, -0.3, 0.5], [-1, 0.6, 0.2]])
    first = compute_hertz_subsurface_stress(
        contact(2, 1), points, tangential_traction_ratio=[0.3, -0.2]
    )
    scaled = compute_hertz_subsurface_stress(
        contact(2 * scale, scale), points * scale, tangential_traction_ratio=[0.3, -0.2]
    )
    np.testing.assert_allclose(
        scaled.total_stress_Pa, first.total_stress_Pa, rtol=1e-13, atol=1e-14
    )


def test_coincident_background_added_before_invariants_and_output_is_detached():
    points = np.array([[0.2, 0.4, 0.5]])
    background = np.array([[3, 2, 0], [2, 0, -1], [0, -1, -3.0]])
    result = compute_hertz_subsurface_stress(
        contact(), points, background_stress_Pa=background
    )
    expected = result.normal_stress_Pa[0] + background
    np.testing.assert_allclose(result.total_stress_Pa[0], expected)
    dev = expected - np.trace(expected) * np.eye(3) / 3
    assert result.von_mises_Pa[0] == pytest.approx(np.sqrt(1.5 * np.sum(dev * dev)))
    principal = np.linalg.eigvalsh(expected)
    assert result.maximum_shear_Pa[0] == pytest.approx(
        (principal[-1] - principal[0]) / 2
    )
    assert result.sampled_peak_von_mises_Pa == result.von_mises_Pa[0]
    np.testing.assert_array_equal(result.sampled_peak_point_m, points[0])
    points[:] = 9
    background[:] = 9
    assert result.points_m[0, 0] == 0.2
    for array in (
        result.points_m,
        result.normal_stress_Pa,
        result.tangential_stress_Pa,
        result.total_stress_Pa,
        result.principal_stress_Pa,
        result.von_mises_Pa,
        result.maximum_shear_Pa,
        result.quadrature_order,
        result.refinement_difference_Pa,
        result.tangential_traction_ratio,
        result.sampled_peak_point_m,
    ):
        assert not array.flags.writeable


def test_zero_pressure_retains_background_without_quadrature():
    result = compute_hertz_subsurface_stress(
        contact(pressure=0), [0, 0, 0.1], background_stress_Pa=np.diag([3.0, 0, 0])
    )
    np.testing.assert_array_equal(result.normal_stress_Pa, 0)
    np.testing.assert_array_equal(result.tangential_stress_Pa, 0)
    np.testing.assert_array_equal(result.quadrature_order, 0)
    np.testing.assert_array_equal(result.refinement_difference_Pa, 0)
    assert result.von_mises_Pa[0] == 3


@pytest.mark.parametrize("point", [[0.2, 0.3, 1e-5], [0, 0, 0.2]])
def test_exhausted_refinement_never_returns_unresolved_stress(point):
    with pytest.raises(RuntimeError, match="did not converge at point 0"):
        compute_hertz_subsurface_stress(contact(), point, initial_order=4, max_order=8)


@pytest.mark.parametrize(
    "points",
    [
        [],
        [0, 1],
        [0, 0, 0],
        [0, 0, -1],
        [0, 0, np.nan],
        [np.inf, 0, 1],
        [0, 0, 1j],
        np.ones((1, 1, 3)),
    ],
)
def test_invalid_points_rejected(points):
    with pytest.raises(ValueError):
        compute_hertz_subsurface_stress(contact(), points)


@pytest.mark.parametrize(
    "ratio", [[], [1], [1, 2, 3], [np.nan, 0], [0, np.inf], [1j, 0]]
)
def test_invalid_traction_ratio_rejected(ratio):
    with pytest.raises(ValueError):
        compute_hertz_subsurface_stress(
            contact(), [0, 0, 1], tangential_traction_ratio=ratio
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"poisson_ratio": -0.999 - 0.001},
        {"poisson_ratio": 0.51},
        {"poisson_ratio": np.nan},
        {"rtol": -1},
        {"rtol": 1},
        {"atol_Pa": -1},
        {"rtol": 0, "atol_Pa": 0},
        {"initial_order": 3},
        {"initial_order": True},
        {"max_order": 1025},
        {"max_order": 10.5},
        {"initial_order": 32, "max_order": 16},
    ],
)
def test_invalid_controls_rejected(kwargs):
    with pytest.raises(ValueError):
        compute_hertz_subsurface_stress(contact(), [0, 0, 1], **kwargs)


@pytest.mark.parametrize(
    "footprint", [contact(a=0), contact(b=-1), contact(pressure=-1), contact(a=np.inf)]
)
def test_invalid_contact_rejected(footprint):
    with pytest.raises(ValueError):
        compute_hertz_subsurface_stress(footprint, [0, 0, 1])


@pytest.mark.parametrize(
    "background",
    [
        np.zeros((2, 2)),
        np.ones((3, 3), dtype=complex),
        np.full((3, 3), np.inf),
        np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]]),
    ],
)
def test_invalid_background_rejected(background):
    with pytest.raises(ValueError):
        compute_hertz_subsurface_stress(
            contact(), [0, 0, 1], background_stress_Pa=background
        )


def cartesian_displacement_stress(point, axes, nu, ratios, order):
    """Independent Cartesian area quadrature and displacement differentiation.

    Map x=a*u, y=b*sqrt(1-u²)*v. Pressure times Jacobian is
    a*b*(1-u²)*sqrt(1-v²), integrated with Gauss-Jacobi weights.
    No production stress kernel, polar mapping or refinement rule is used.
    """
    u, wu = roots_jacobi(order, 1, 1)
    v, wv = roots_jacobi(order, 0.5, 0.5)
    a, b = axes
    sources = np.zeros((3, order, order))
    sources[0] = a * u[:, None]
    sources[1] = b * np.sqrt(1 - u[:, None] ** 2) * v
    weights = a * b * wu[:, None] * wv
    gradient = np.zeros((3, 3))
    for j in range(3):
        observer = np.asarray(point, dtype=complex).copy()
        observer[j] += 1e-25j
        relative = observer[:, None, None] - sources
        field = sum(
            coefficient * displacement(relative, nu, force)
            for force, coefficient in enumerate([1, *ratios])
        )
        gradient[:, j] = np.sum(field.imag * weights, axis=(1, 2)) / 1e-25
    return (
        gradient + gradient.T + (2 * nu / (1 - 2 * nu)) * np.trace(gradient) * np.eye(3)
    )


@pytest.mark.parametrize("nu", [-0.2, 0.3])
@pytest.mark.parametrize("point", [[0.7, 0.4, 0.3], [-1.4, 0.2, 0.6], [2.4, 1.2, 0.4]])
def test_off_axis_matches_cartesian_displacement_integration(nu, point):
    ratios = [0.4, -0.25]
    reference = cartesian_displacement_stress(point, (2, 1), nu, ratios, 80)
    refined = cartesian_displacement_stress(point, (2, 1), nu, ratios, 160)
    np.testing.assert_allclose(reference, refined, rtol=2e-8, atol=2e-10)
    result = compute_hertz_subsurface_stress(
        contact(2, 1),
        point,
        poisson_ratio=nu,
        tangential_traction_ratio=ratios,
        rtol=1e-8,
        atol_Pa=1e-11,
    )
    np.testing.assert_allclose(
        result.total_stress_Pa[0], refined, rtol=2e-8, atol=2e-10
    )


@pytest.mark.parametrize("nu", [-0.2, 0.3, 0.5])
def test_point_kernel_hemisphere_resultant_balances_applied_force(nu):
    nodes, weights = roots_legendre(64)
    theta = (nodes + 1) * math.pi / 4
    phi = np.arange(128) * 2 * math.pi / 128
    normals = np.stack(
        np.broadcast_arrays(
            np.sin(theta)[:, None] * np.cos(phi),
            np.sin(theta)[:, None] * np.sin(phi),
            np.cos(theta)[:, None],
        ),
        axis=-1,
    ).reshape(-1, 3)
    area = np.broadcast_to(
        (weights * math.pi / 4 * np.sin(theta))[:, None] * (2 * math.pi / 128),
        (64, 128),
    ).ravel()
    kernels = _point_stress(normals, nu)
    for force, direction in enumerate(np.eye(3)[[2, 0, 1]]):
        tensors = np.array([matrix(value) for value in kernels[force]])
        resultant = np.einsum("n,nij,nj->i", area, tensors, normals)
        np.testing.assert_allclose(resultant, -direction, atol=2e-14)


def test_far_field_matches_resultant_point_forces():
    point = np.array([200, -160, 240.0])
    ratios = [0.3, -0.2]
    result = compute_hertz_subsurface_stress(
        contact(2, 1),
        point,
        tangential_traction_ratio=ratios,
        rtol=1e-9,
        atol_Pa=1e-16,
    )
    force = 2 * math.pi / 3 * 2 * 1
    expected = force * matrix(
        np.array([1, *ratios]) @ _point_stress(point[None], 0.3)[:, 0]
    )
    np.testing.assert_allclose(result.total_stress_Pa[0], expected, rtol=3e-4, atol=0)


def test_shallow_interior_tractions_approach_prescribed_boundary():
    depths = np.array([0.2, 0.1, 0.05, 0.025])
    points = np.column_stack((np.full(4, 0.2), np.full(4, 0.3), depths))
    result = compute_hertz_subsurface_stress(
        contact(),
        points,
        tangential_traction_ratio=[0.4, -0.2],
        atol_Pa=1e-10,
        max_order=1024,
    )
    boundary = -np.sqrt(1 - 0.2**2 - 0.3**2) * np.array([0.4, -0.2, 1])
    errors = np.linalg.norm(result.total_stress_Pa[:, :, 2] - boundary, axis=1)
    assert np.all(np.diff(errors) < 0)
    assert errors[-1] < 0.04


def test_circle_arbitrary_in_plane_rotation_preserves_invariants():
    angle = 0.731
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    points = np.array([[0.4, -0.2, 0.3], [1.2, 0.5, 0.6]])
    ratios = np.array([0.35, -0.1])
    background = np.array([[[2, 0.3, 0], [0.3, -1, 0], [0, 0, 0]], np.eye(3)])
    original = compute_hertz_subsurface_stress(
        contact(),
        points,
        tangential_traction_ratio=ratios,
        background_stress_Pa=background,
        atol_Pa=1e-10,
    )
    rotated = compute_hertz_subsurface_stress(
        contact(),
        points @ rotation.T,
        tangential_traction_ratio=rotation[:2, :2] @ ratios,
        background_stress_Pa=rotation @ background @ rotation.T,
        atol_Pa=1e-10,
    )
    np.testing.assert_allclose(
        rotated.total_stress_Pa,
        rotation @ original.total_stress_Pa @ rotation.T,
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(rotated.von_mises_Pa, original.von_mises_Pa, rtol=1e-8)


def test_pressure_linearity_and_absolute_tolerance_refinement():
    loose = compute_hertz_subsurface_stress(
        contact(2, 1, 1e8),
        [0.6, 0.2, 0.1],
        tangential_traction_ratio=[0.3, 0.1],
    )
    strict = compute_hertz_subsurface_stress(
        contact(2, 1, 2e8),
        [0.6, 0.2, 0.1],
        tangential_traction_ratio=[0.3, 0.1],
        rtol=0,
        atol_Pa=1e-4,
        max_order=1024,
    )
    np.testing.assert_allclose(
        strict.total_stress_Pa, 2 * loose.total_stress_Pa, rtol=2e-7, atol=1e-4
    )
    assert strict.quadrature_order[0] >= loose.quadrature_order[0]
    assert strict.refinement_difference_Pa[0] <= 1e-4
    assert strict.semi_axes_m == (2, 1)
    assert strict.peak_pressure_Pa == 2e8


@pytest.mark.parametrize(
    "footprint,ratios",
    [
        (contact(1e-300, 1e-300), [0, 0]),
        (contact(pressure=1e308), [1e308, 0]),
    ],
)
def test_unrepresentable_stress_is_rejected(footprint, ratios):
    with pytest.raises(ValueError, match="floating-point range"):
        compute_hertz_subsurface_stress(
            footprint,
            [0.1, 0.2, 1],
            tangential_traction_ratio=ratios,
        )
