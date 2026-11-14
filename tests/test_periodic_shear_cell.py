# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Independent displacement equilibrium versus the analytical wire solution."""

import math

import numpy as np
import pytest
from periodic_shear_cell import solve_periodic_shear_cell

pytestmark = pytest.mark.independent_method

EA, RADIUS, ANGLE, COUNT = 1.5e6, 0.05, 0.35, 24


def _solve(support, elements=128, angle=ANGLE):
    return solve_periodic_shear_cell(
        axial_rigidity_N=EA,
        helix_radius_m=RADIUS,
        lay_angle_rad=angle,
        wire_count=COUNT,
        support_N_per_m2=support,
        elements=elements,
    )


@pytest.mark.parametrize("xi", [0.02, 0.2, 0.8, 0.98])
def test_periodic_displacement_recovers_yun_limit_with_mesh_convergence(xi):
    q = math.sin(ANGLE) / RADIUS
    support = EA * q * q * xi / (1 - xi)
    exact = 0.5 * COUNT * EA * RADIUS**2 * math.cos(ANGLE) ** 3 * xi
    errors = []
    for elements in [32, 64, 128]:
        result = _solve(support, elements)
        errors.append(abs(result.moment_stiffness_Nm2[0, 0] / exact - 1))
        np.testing.assert_allclose(
            result.moment_stiffness_Nm2,
            result.wire_energy_stiffness_Nm2 + result.support_energy_stiffness_Nm2,
            rtol=2e-10,
            atol=1e-7,
        )
        assert result.relative_equilibrium_residual < 1e-10
    assert errors[1] < 0.27 * errors[0]
    assert errors[2] < 0.27 * errors[1]
    assert errors[-1] < 0.011  # worst case is the almost unsupported wire
    exact_displacement = (
        -(1 - xi) * RADIUS * math.cos(ANGLE) ** 2 / q * np.sin(result.phi_nodes_rad)
    )
    np.testing.assert_allclose(
        result.warping_per_curvature_m2[:, 0], exact_displacement, atol=2e-6
    )


def test_variable_support_preserves_reciprocity_and_work_conjugacy():
    # A prescribed threefold foundation is not a solved three-core solid.
    stiffness = EA * (math.sin(ANGLE) / RADIUS) ** 2
    result = _solve(lambda phi: stiffness * (1 + 0.7 * np.cos(3 * phi)), elements=192)
    matrix = result.moment_stiffness_Nm2
    np.testing.assert_allclose(matrix, matrix.T, rtol=1e-12, atol=1e-7)
    np.testing.assert_allclose(matrix, np.eye(2) * matrix[0, 0], rtol=1e-12, atol=1e-7)
    np.testing.assert_allclose(
        matrix,
        result.wire_energy_stiffness_Nm2 + result.support_energy_stiffness_Nm2,
        rtol=1e-12,
        atol=1e-7,
    )
    average_support_result = _solve(stiffness, elements=192)
    assert matrix[0, 0] < average_support_result.moment_stiffness_Nm2[0, 0]


def test_zero_support_converges_to_zero_parallel_axis_stiffness():
    # Linear element strain cannot reproduce the sinusoid exactly at finite h.
    coarse, fine = _solve(0, elements=64), _solve(0, elements=128)
    assert fine.moment_stiffness_Nm2[0, 0] < 0.26 * coarse.moment_stiffness_Nm2[0, 0]
    assert np.max(np.abs(fine.support_energy_stiffness_Nm2)) == 0
    assert np.max(np.abs(np.mean(fine.warping_per_curvature_m2, axis=0))) < 1e-15


def test_helix_hand_changes_displacement_phase_but_not_isotropic_stiffness():
    positive, negative = _solve(1e8), _solve(1e8, angle=-ANGLE)
    np.testing.assert_allclose(
        positive.moment_stiffness_Nm2,
        negative.moment_stiffness_Nm2,
        rtol=1e-12,
        atol=1e-7,
    )


@pytest.mark.parametrize("support", [-1, np.inf, np.nan])
def test_invalid_support_is_rejected(support):
    with pytest.raises(ValueError, match="support"):
        _solve(support)
