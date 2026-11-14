# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Contracts retained by the optimized physical-wire history path."""

from dataclasses import replace

import numpy as np
import pytest

from cablexsec.mechanics.mkappa import HelicalLayerSpec
from cablexsec.mechanics.papailiou import wire_slip_history, wire_slip_stress_cap


def layer():
    return HelicalLayerSpec(
        8, 0.05, 0.4, 2e11, 7e-6, 0.8, 0.2, 10, contact_pressure_Pa=3e5
    )


@pytest.mark.parametrize("shape", [np.ones((2, 8)), np.ones((1, 8)), np.ones(3)])
def test_derived_cap_cannot_change_the_wire_population(shape):
    with pytest.raises(ValueError):
        wire_slip_history(
            replace(layer(), shape_g=shape), np.array([0, 0.1]), np.zeros(2)
        )


@pytest.mark.parametrize(
    "values", [np.full(8, np.nan), np.full(8, np.inf), -np.ones(8)]
)
def test_invalid_derived_caps_are_rejected(values):
    with pytest.raises(ValueError):
        wire_slip_history(
            replace(layer(), shape_g=values), np.array([0.1]), np.zeros(1)
        )


def test_finite_input_with_overflowing_trial_does_not_return_a_clipped_history():
    with pytest.raises(ValueError, match="range"):
        wire_slip_history(layer(), np.array([1e300]), np.zeros(1))


def test_finite_input_with_overflowing_drive_is_rejected():
    with pytest.raises(ValueError):
        wire_slip_history(
            replace(layer(), r_helix_m=1e300), np.array([1e300]), np.zeros(1)
        )


def test_pressure_changes_release_stress_at_fixed_curvature_and_zero_hold_keeps_direction():
    specimen = layer()
    kx = np.array([0.0, 0.1, 0.1, 0.0, 0.0])
    ky = np.array([0.0, 0.2, 0.2, 0.0, 0.0])
    pressure = np.array([3e5, 3e5, 0, 2e5, 1e5])
    result = wire_slip_history(specimen, kx, ky, contact_pressure_Pa=pressure)
    np.testing.assert_array_equal(result.stress_Pa[:, 2], 0)
    angle = np.arctan2(0.2, 0.1)
    cap = wire_slip_stress_cap(specimen, angle, contact_pressure_Pa=1e5)
    assert np.all(np.abs(result.stress_Pa[:, -1]) <= cap)
    np.testing.assert_allclose(result.work.balance_residual_J_per_m, 0, atol=2e-13)


def test_constant_direction_cache_does_not_cross_calls_or_alias_shape_input():
    shape = np.ones(8)
    specimen = replace(layer(), shape_g=shape)
    curvature = np.array([0.1, 0.2, 0.3])
    first = wire_slip_history(specimen, curvature, curvature * 0)
    saved = first.stress_Pa.copy()
    shape *= 0.5
    second = wire_slip_history(specimen, curvature, curvature * 0)
    np.testing.assert_array_equal(first.stress_Pa, saved)
    assert np.max(np.abs(second.stress_Pa)) < np.max(np.abs(first.stress_Pa))


def test_variable_material_work_balance_and_caller_inputs_remain_intact():
    specimen = layer()
    rng = np.random.default_rng(714)
    kx, ky = rng.normal(0, 0.04, (2, 81))
    pressure = rng.uniform(0, 4e5, 81)
    stiffness = specimen.k_f * rng.uniform(0.5, 1.3, 81)
    inputs = tuple(v.copy() for v in (kx, ky, pressure, stiffness))
    result = wire_slip_history(
        specimen, kx, ky, contact_pressure_Pa=pressure, stick_modulus_Pa=stiffness
    )
    # Independent accounting identity for the reported material, stored,
    # friction and backward-Euler remainder terms over a changing-stiffness path.
    supplied = result.work.endpoint_work_J_per_m + result.material_work_J_per_m
    accounted = (
        result.work.stored_J_per_m
        + result.work.friction_J_per_m
        + result.work.integration_remainder_J_per_m
    )
    np.testing.assert_allclose(supplied, accounted, rtol=2e-14, atol=1e-12)
    for actual, saved in zip((kx, ky, pressure, stiffness), inputs, strict=True):
        np.testing.assert_array_equal(actual, saved)
