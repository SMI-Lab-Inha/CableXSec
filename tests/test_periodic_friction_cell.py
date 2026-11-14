# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Analytical limits and work checks for independently solved wire slip."""

import math

import numpy as np
import pytest
from periodic_friction_cell import solve_friction_cell


def _inputs():
    return {
        "axial_rigidity_N": 2e6,
        "helix_radius_m": 0.04,
        "lay_angle_rad": 0.3,
        "wire_count": 40,
        "support_N_m2": 4e8,
        "friction_limit_N_m": 1000.0,
    }


def test_elastic_solution_matches_continuous_displacement_and_stiffness():
    inputs = _inputs()
    ea, radius, alpha, stiffness = (
        inputs[key]
        for key in (
            "axial_rigidity_N",
            "helix_radius_m",
            "lay_angle_rad",
            "support_N_m2",
        )
    )
    q = math.sin(alpha) / radius
    xi = stiffness / (stiffness + ea * q * q)
    kappa = 1e-5
    result = solve_friction_cell(
        **inputs, curvature_1_m=np.array([0.0, kappa]), elements=256
    )
    expected_u = (
        -(1 - xi) * kappa * radius * math.cos(alpha) ** 2 / q * np.sin(result.phi_rad)
    )
    np.testing.assert_allclose(
        result.final_warping_m, expected_u, rtol=3e-4, atol=1e-12
    )
    expected_ei = (
        0.5 * inputs["wire_count"] * ea * radius**2 * math.cos(alpha) ** 3 * xi
    )
    assert result.moment_Nm[-1] / kappa == pytest.approx(expected_ei, rel=3e-5)
    assert result.friction_J_m[-1] < 1e-13


def test_slip_onset_matches_force_equilibrium():
    inputs = _inputs()
    alpha, ea, radius = (
        inputs["lay_angle_rad"],
        inputs["axial_rigidity_N"],
        inputs["helix_radius_m"],
    )
    q = math.sin(alpha) / radius
    xi = inputs["support_N_m2"] / (inputs["support_N_m2"] + ea * q * q)
    onset = inputs["friction_limit_N_m"] / (
        xi * ea * math.cos(alpha) ** 2 * math.sin(alpha)
    )
    below = solve_friction_cell(**inputs, curvature_1_m=np.array([0.99 * onset]))
    above = solve_friction_cell(**inputs, curvature_1_m=np.array([1.01 * onset]))
    assert below.friction_J_m[-1] < 1e-12
    assert above.friction_J_m[-1] > 1e-8


def test_cyclic_equilibrium_and_incremental_work_balance():
    kappa = 0.03 * np.sin(np.linspace(0, 6 * np.pi, 361))
    result = solve_friction_cell(**_inputs(), curvature_1_m=kappa)
    residual = (
        result.endpoint_work_J_m
        - result.rod_energy_J_m
        - result.support_energy_J_m
        - result.friction_J_m
        - result.incremental_remainder_J_m
    )
    assert np.max(np.abs(residual)) < 1e-8
    assert np.max(result.equilibrium_relative_residual) < 1e-10
    assert np.min(np.diff(result.friction_J_m)) > -1e-12
    np.testing.assert_allclose(
        result.moment_Nm[-121:], result.moment_Nm[-241:-120], atol=1e-8
    )


def test_zero_friction_converges_to_free_warping():
    inputs = _inputs() | {"friction_limit_N_m": 0.0}
    errors = []
    for elements in (16, 32, 64, 128):
        result = solve_friction_cell(
            **inputs, curvature_1_m=np.array([0.01]), elements=elements
        )
        errors.append(abs(result.moment_Nm[-1]))
        assert result.friction_J_m[-1] == 0
        assert result.support_energy_J_m[-1] == 0
    assert all(3.9 < a / b < 4.1 for a, b in zip(errors[:-1], errors[1:], strict=True))


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("support_N_m2", 0),
        ("friction_limit_N_m", -1),
        ("elements", True),
        ("lay_angle_rad", 0),
        ("tolerance", float("nan")),
    ],
)
def test_invalid_reference_inputs_rejected(parameter, value):
    with pytest.raises(ValueError):
        solve_friction_cell(
            **(_inputs() | {parameter: value}), curvature_1_m=np.array([0.0])
        )
