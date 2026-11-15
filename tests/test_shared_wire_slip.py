# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Independent scalar plastic-strain and work checks for the shared wire update."""

import math

import numpy as np
import pytest

from cablexsec.mechanics.mkappa import HelicalLayerSpec
from cablexsec.mechanics.papailiou import (
    compute_mkappa_wire_slip_front,
    wire_slip_history,
)

pytestmark = pytest.mark.analytical_sanity


@pytest.fixture
def layer():
    return HelicalLayerSpec(
        n_wires=16,
        r_helix_m=0.05,
        lay_angle_rad=0.3,
        E_wire_Pa=2e11,
        A_wire_m2=math.pi * 0.003**2 / 4,
        xi=0.7,
        mu=0.2,
        F_n_axi_N=1,
        contact_pressure_Pa=2e5,
    )


@pytest.mark.parametrize("angle", [0.0, 0.31, math.pi / 4, 1.2])
def test_fixed_plane_reversal_and_zero_hold_match_independent_plastic_strain(
    layer, angle
):
    kappa = np.array([0, 0.02, 0.08, 0.02, 0, 0, -0.04, -0.1, 0, 0.06])
    result = wire_slip_history(
        layer,
        kappa * math.cos(angle),
        kappa * math.sin(angle),
        initial_bending_angle=angle,
    )
    phi = 2 * np.pi * np.arange(layer.n_wires) / layer.n_wires
    c, s = math.cos(layer.lay_angle_rad), math.sin(layer.lay_angle_rad)
    stiffness = layer.xi * layer.E_wire_Pa * c**2
    width = 2 * math.pi * layer.r_helix_m / layer.n_wires
    sigma0 = (
        layer.contact_pressure_Pa
        * width
        * c
        * layer.r_helix_m
        / (layer.A_wire_m2 * s**2)
    )
    # Signed capstan ([Papailiou1997] Eqs. 11, 14 with the signed helix
    # angle): a rising stress is bounded by sigma0 (exp(lam psi) - 1), a
    # falling one by -sigma0 (1 - exp(-lam psi)), whichever side the wire is
    # on; psi depends on the bending plane only, not on the sign of kappa.
    lam_psi = layer.mu * s * np.arcsin(np.abs(np.cos(phi - angle)))
    upper = sigma0 * (np.exp(lam_psi) - 1)
    lower = sigma0 * (1 - np.exp(-lam_psi))
    plastic = np.zeros(layer.n_wires)
    expected = []
    for curvature in kappa:
        strain = curvature * layer.r_helix_m * np.cos(phi - angle)
        stress = np.clip(stiffness * (strain - plastic), -lower, upper)
        plastic = strain - stress / stiffness
        expected.append(stress)
    np.testing.assert_allclose(result.stress_Pa.T, expected, rtol=2e-12, atol=2e-7)
    np.testing.assert_array_equal(result.stress_Pa[:, 4], result.stress_Pa[:, 5])
    np.testing.assert_allclose(
        result.final_plastic_drive, plastic, rtol=2e-12, atol=1e-17
    )


def test_biaxial_moment_is_work_conjugate_to_the_same_wire_stress(layer):
    phase = np.linspace(0, 4 * math.pi, 129)
    kx, ky = 0.09 * np.sin(phase), 0.06 * (np.cos(phase) - 1)
    result = wire_slip_history(layer, kx, ky)
    phi = layer.phi_wires
    lever = layer.A_wire_m2 * layer.r_helix_m * math.cos(layer.lay_angle_rad)
    independent_moment = lever * np.column_stack(
        (np.cos(phi) @ result.stress_Pa, np.sin(phi) @ result.stress_Pa)
    )
    np.testing.assert_allclose(
        result.moment_Nm, independent_moment, rtol=2e-12, atol=1e-11
    )
    work = np.cumsum(
        np.sum(
            independent_moment
            * np.column_stack((np.diff(kx, prepend=0), np.diff(ky, prepend=0))),
            axis=1,
        )
    )
    np.testing.assert_allclose(
        result.work.endpoint_work_J_per_m, work, rtol=2e-12, atol=1e-11
    )
    assert np.max(np.abs(result.work.balance_residual_J_per_m)) < 1e-10
    assert np.min(np.diff(result.work.friction_J_per_m, prepend=0)) >= -1e-12


def test_changing_stiffness_work_is_not_counted_as_friction(layer):
    kappa = np.array([0, 1e-5, 1e-5, 1e-5])
    stiffness = layer.k_f * np.array([1, 1, 0.5, 0.8])
    result = wire_slip_history(layer, kappa, kappa * 0, stick_modulus_Pa=stiffness)
    weight = layer.A_wire_m2 * math.cos(layer.lay_angle_rad)
    elastic = 1e-5 * layer.r_helix_m * np.cos(layer.phi_wires)
    expected = 0.5 * weight * (stiffness[-1] - stiffness[1]) * np.sum(elastic**2)
    assert result.material_work_J_per_m[-1] == pytest.approx(expected, rel=1e-12)
    assert np.max(np.abs(result.work.balance_residual_J_per_m)) < 1e-14
    assert result.work.friction_J_per_m[-1] < 1e-14
    assert result.work.endpoint_work_J_per_m[-1] == result.work.endpoint_work_J_per_m[1]


def test_pressure_release_and_reclosure_at_held_curvature_retain_slip(layer):
    kx = np.array([0, 0.08, 0.08, 0.08, 0])
    pressure = np.array([1, 1, 0, 1, 1]) * layer.contact_pressure_Pa
    result = wire_slip_history(layer, kx, kx * 0, contact_pressure_Pa=pressure)
    np.testing.assert_array_equal(result.stress_Pa[:, 2:4], 0)
    assert result.stress_Pa[0, -1] < 0
    assert np.max(np.abs(result.work.balance_residual_J_per_m)) < 1e-10


def test_section_loop_integrates_the_shared_wire_work(layer):
    kappa = 0.1 * np.sin(np.linspace(0, 6 * math.pi, 181))
    angle = 0.37
    wire = wire_slip_history(
        layer,
        kappa * math.cos(angle),
        kappa * math.sin(angle),
        initial_bending_angle=angle,
    )
    section = compute_mkappa_wire_slip_front(
        [layer], kappa, phi_bend=angle, EI_min_Nm2=15
    )
    np.testing.assert_allclose(
        section.moment_Nm,
        wire.moment_Nm @ [math.cos(angle), math.sin(angle)] + 15 * kappa,
    )
    np.testing.assert_array_equal(
        section.work.friction_J_per_m, wire.work.friction_J_per_m
    )
    assert np.max(np.abs(section.work.balance_residual_J_per_m)) < 1e-10


@pytest.mark.parametrize("argument", ["contact_pressure_Pa", "stick_modulus_Pa"])
def test_bad_normal_history_is_not_silently_broadcast(layer, argument):
    with pytest.raises(ValueError, match="match curvature"):
        wire_slip_history(layer, np.zeros(3), np.zeros(3), **{argument: np.ones(2)})
