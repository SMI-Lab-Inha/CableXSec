# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Multilinear strand-slip operator ([KarlsenHusoy2026] Eqs. 2-10)."""

from __future__ import annotations

import math

import check_conductor_strand_slip as check
import numpy as np
import pytest

from cablexsec.mechanics.strand_slip import (
    ConductorRing,
    StrandSlipParameters,
    conductor_axial_stiffness,
    conductor_ring_parameters,
    interface_line_loads,
    strand_slip_strain,
)

TABLE3 = check.KH_TABLE3


def _sine(amplitude: float, cycles: int = 3, points: int = 3001) -> np.ndarray:
    return amplitude * np.sin(np.linspace(0, 2 * math.pi * cycles, points))


def test_trilinear_breakpoints_match_closed_form():
    values = check.karlsen_husoy_breakpoints()
    # One sample of 0.05 * 4 pi / 8000 resolves the curvature.
    tolerance = 0.05 * 4 * math.pi / 8000
    assert values["first_operator"] == pytest.approx(
        values["first_closed_form"], abs=tolerance
    )
    assert values["second_operator"] == pytest.approx(
        values["second_closed_form"], abs=tolerance
    )
    assert values["full_slip_range"] == pytest.approx(
        values["full_slip_closed_form"], rel=1e-12
    )


def test_limits_reduce_to_linear_methods():
    kappa = _sine(0.05)
    zeros = np.zeros_like(kappa)
    tension = np.full_like(kappa, 1e3)
    stick = StrandSlipParameters(0.0, (0.04, 0.01), (1.0,), 1e3)
    slip = StrandSlipParameters(0.0, (0.04, 0.01), (0.0,), 1e3)
    np.testing.assert_allclose(
        strand_slip_strain(stick, kappa, zeros, tension, theta_rad=[0]).strain[0],
        0.04 * kappa,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        strand_slip_strain(slip, kappa, zeros, tension, theta_rad=[0]).strain[0],
        0.01 * kappa,
        atol=1e-15,
    )


def test_friction_amplitude_scales_with_tension_and_orientation():
    """Eq. 4: eps_f proportional to T/T0 and |cos phi|."""
    params = StrandSlipParameters(0.0, (0.04, 0.01), (2e-4,), 1e3)
    kappa = _sine(0.05)
    zeros = np.zeros_like(kappa)
    for scale in (1.0, 0.5):
        tension = np.full_like(kappa, scale * 1e3)
        result = strand_slip_strain(
            params, kappa, zeros, tension, theta_rad=[0.0, math.pi / 3, math.pi / 2]
        )
        steady = result.strain[:, 1000:]
        span = steady.max(axis=1) - steady.min(axis=1)
        cosine = np.array([1.0, 0.5, 0.0])
        expected = 0.01 * 0.1 * cosine + 2 * scale * 2e-4 * cosine
        np.testing.assert_allclose(span, expected, rtol=1e-9, atol=1e-15)
    # Compression uses |T| (reversed inequality set of the source).
    compressed = strand_slip_strain(
        params, kappa, zeros, np.full_like(kappa, -1e3), theta_rad=[0.0]
    )
    tensioned = strand_slip_strain(
        params, kappa, zeros, np.full_like(kappa, 1e3), theta_rad=[0.0]
    )
    np.testing.assert_allclose(compressed.bending_strain, tensioned.bending_strain)


def test_axial_strain_and_mobilisation():
    params = StrandSlipParameters.from_karlsen_husoy(1.4e-6, (0.04, 0.01), (2e-4,), 10)
    kappa = _sine(0.05)
    tension = np.full_like(kappa, 10e3)
    result = strand_slip_strain(
        params, kappa, np.zeros_like(kappa), tension, theta_rad=[0]
    )
    np.testing.assert_allclose(result.axial_strain, 1.4e-6 * 10)
    assert np.all(np.abs(result.mobilisation) <= 1 + 1e-12)
    assert np.any(result.slipping) and not np.all(result.slipping)
    slip = result.slipping[0, 0]
    np.testing.assert_allclose(np.abs(result.mobilisation[0, 0, slip]), 1.0)


def test_rotating_curvature_holds_direction_through_zero():
    params = StrandSlipParameters(0.0, (0.04, 0.01), (2e-4,), 1e3)
    kx = np.array([0.0, 0.01, 0.0, 0.0, -0.01])
    ky = np.zeros_like(kx)
    result = strand_slip_strain(
        params, kx, ky, np.full_like(kx, 1e3), theta_rad=[0.0, math.pi / 2]
    )
    assert np.all(np.isfinite(result.strain))
    # The element on the neutral axis has zero friction amplitude: pure slip.
    np.testing.assert_allclose(result.bending_strain[1], 0.0, atol=1e-18)


def test_table4_strain_ranges_follow_the_table5_damage_order():
    """Fully separated methods keep the source's damage ranking."""
    ranges = check.karlsen_husoy_methods(0.05)
    for higher, lower in (("FS", "SOE"), ("SOE", "FFS"), ("FFS", "AO"), ("AO", "FSS")):
        assert ranges[higher] > ranges[lower]
    small = check.karlsen_husoy_methods(0.002)
    assert small["MOE"] >= small["AO"] > small["FSS"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"beta_m": (), "eps_f_ref": ()},
        {"beta_m": (0.1, 0.01), "eps_f_ref": ()},
        {"beta_m": (0.1, 0.01), "eps_f_ref": (-1e-4,)},
        {"beta_m": (0.1, 0.01), "eps_f_ref": (1e-4,), "T0_N": 0.0},
        {"beta_m": (math.nan,), "eps_f_ref": ()},
    ],
)
def test_parameters_are_validated(kwargs):
    arguments = {"alpha_per_N": 0.0, "T0_N": 1.0, **kwargs}
    with pytest.raises(ValueError):
        StrandSlipParameters(**arguments)


def test_ring_geometry_is_validated():
    with pytest.raises(ValueError):
        ConductorRing(0, 1e-3, 0.0, 0.0)
    with pytest.raises(ValueError):
        ConductorRing(6, 1e-3, 2e-3, 0.0)
    with pytest.raises(ValueError):
        ConductorRing(6, 1e-3, 2e-3, 0.1, lay_sign=0.5)


def test_interface_loads_follow_helix_tension_and_pressure():
    rings = check.nasution_rings()
    outer, inner = interface_line_loads(rings, 115e9, 1e-3)
    # The outer face of the outermost ring is free; loads accumulate inwards.
    assert outer[-1] == 0.0
    assert np.all(np.diff(inner[1:]) < 0)
    # The outer face of ring i is the inner face of ring i + 1.
    np.testing.assert_allclose(outer[:-1], inner[1:])
    pressed, _ = interface_line_loads(
        rings, 115e9, 0.0, external_pressure_Pa=1e6, r_outer_m=0.011
    )
    np.testing.assert_allclose(pressed[-1], 2 * math.pi * 0.011 * 1e6)
    assert np.all(pressed[:-1] == 0)
    everywhere, _ = interface_line_loads(
        rings,
        115e9,
        0.0,
        external_pressure_Pa=1e6,
        r_outer_m=0.011,
        pressure_transmission="all_interfaces",
    )
    np.testing.assert_allclose(everywhere, 2 * math.pi * 0.011 * 1e6)


def test_nasution_friction_stress_range_of_the_second_layer():
    """[Nasution2014b] Table 5: analytical friction range 117 MPa (2nd layer)."""
    values = check.nasution()
    # This model reads Eq. 9 from the line loads of the helix tensions; it
    # gives 106.6 MPa, 9% below the source's 117 MPa.
    assert values["friction_range_L2_MPa"] == pytest.approx(106.6, abs=0.2)
    assert 0.85 < values["friction_range_L2_MPa"] / 117.0 < 1.0
    # Full slip at both radii: friction range plus local bending (Eq. 6).
    assert values["r3_mu0.2_range_L2_MPa"] == pytest.approx(227.5, abs=0.5)
    assert values["r3_mu0.2_critical_layer"] == 2  # failures in layers 2-3
    ratio = values["r3_mu0.2_life_model"] / values["r3_mu0.2_life_tests_mean"]
    assert 1.0 < ratio < 1.6  # unlubricated 3 m tests: model 1.34x longer


def test_utting_jones_wire_strain_and_torque():
    values = check.utting_jones()
    assert values["strain_per_kN_V_model"] == pytest.approx(65.6, abs=0.1)
    assert values["strain_per_kN_V_measured"] == pytest.approx(62.85, abs=0.01)
    torque = [values[f"torque_{name}"] for name in check.UJ_TORQUE]
    # Fixed-end torque per tension is 6-12% above the measured values.
    assert all(1.05 < ratio < 1.13 for ratio in torque)


def test_hu_conductor_loop_against_the_test():
    outer = check.hu_loop("outer_interface")
    assert outer["EI_full_slip"] == pytest.approx(12.9, abs=0.1)  # test 14.4
    assert outer["M_max"] == pytest.approx(7.58, abs=0.05)  # test 8.18
    assert outer["loop_area"] == pytest.approx(4.08, abs=0.05)  # test 3.185
    assert outer["EI_stick"] == pytest.approx(793, abs=1)
    everywhere = check.hu_loop("all_interfaces")
    assert everywhere["M_max"] > 1.5 * check.NAM_M_MAX


def test_conductor_parameters_reproduce_plane_section_stiffness():
    rings = check.hu_rings()
    params = conductor_ring_parameters(
        rings, 118e9, mu_wire=0.2, reference_tension_N=100.0, fibre="centroid"
    )
    assert params[0].n_orders == 0
    for ring, value in zip(rings[1:], params[1:], strict=True):
        assert value.beta_m[0] == pytest.approx(
            ring.r_helix_m * math.cos(ring.lay_angle_rad) ** 2
        )
        assert value.beta_m[1] == 0.0
    ea = conductor_axial_stiffness(rings, 118e9)
    assert params[2].alpha_per_N == pytest.approx(
        math.cos(rings[2].lay_angle_rad) ** 2 / ea
    )


def test_compiled_cascade_matches_the_interpreted_kernel():
    from cablexsec._numba import kernels_compiled
    from cablexsec.mechanics import strand_slip

    if not kernels_compiled():
        pytest.skip("kernels not compiled; the interpreted kernel is the only path")
    rng = np.random.default_rng(3)
    kappa = np.cumsum(rng.normal(0, 2e-3, size=(5, 400)), axis=1)
    eps_f = np.abs(rng.normal(1e-4, 3e-5, size=(2, 5, 400)))
    beta = np.array([0.04, 0.008, 0.002])
    compiled = strand_slip._cascade(kappa, eps_f, beta)
    interpreted = strand_slip._cascade.py_func(kappa, eps_f, beta)
    for left, right in zip(compiled, interpreted, strict=True):
        np.testing.assert_array_equal(left, right)
