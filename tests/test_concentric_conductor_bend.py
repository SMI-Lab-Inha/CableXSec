# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Mechanical checks for the bounded concentric conductor reference."""

import math

import numpy as np
import pytest

pytest.importorskip("clarabel")

from concentric_conductor_bend import (
    AnnularSheath,
    ConcentricConductorBend,
    ConductorRing,
)


def model(*, elements=8, friction=0.2, handedness=1, sheath=None):
    return ConcentricConductorBend(
        rings=[
            ConductorRing(6, 0.002, handedness * 0.1, 0.001, 1e11),
            ConductorRing(12, 0.004, -handedness * 0.2, 0.001, 1e11),
        ],
        interface_radii_m=[0.001, 0.003, 0.005],
        interface_friction=[friction] * 3,
        interface_pressure_Pa=[1e5] * 3,
        individual_bending_Nm2=2.0,
        span_m=0.4,
        elements=elements,
        sheath=sheath,
    )


def elastic_sheath():
    return AnnularSheath(0.005, 0.006, 420e6, 0.35)


def uniform_curvature_state(m, curvature):
    x = np.linspace(0, m.span, m.elements + 1)
    state = np.zeros(m.ndof)
    state[: 2 * len(x) : 2] = 0.5 * curvature * x * (x - m.span)
    state[1 : 2 * len(x) : 2] = curvature * (x - m.span / 2)
    return x, state


def test_locked_uniform_curvature_recovers_independent_wire_energy():
    m = model()
    x = np.linspace(0, m.span, m.elements + 1)
    curvature = 0.2
    d = np.zeros(m.ndof)
    d[: 2 * len(x) : 2] = 0.5 * curvature * x * (x - m.span)
    d[1 : 2 * len(x) : 2] = curvature * (x - m.span / 2)
    expected = m.individual_bending + sum(
        ring.count
        * ring.modulus_Pa
        * math.pi
        * ring.wire_diameter_m**2
        / 4
        * math.cos(ring.angle_rad) ** 3
        * ring.radius_m**2
        / 2
        for ring in m.rings
    )
    assert d @ m.stiffness @ d == pytest.approx(
        expected * curvature**2 * m.span, rel=1e-12
    )


def test_shared_interface_has_equal_and_opposite_action():
    m = model()
    nodes = m.elements + 1
    d = np.zeros(m.ndof)
    # The same physical axial harmonic on both rings has no relative
    # displacement at their common interface, regardless of handedness.
    for ring in range(2):
        start = 2 * nodes * (ring + 1)
        d[start : start + 2 * nodes : 2] = 0.001
        d[start + 1 : start + 2 * nodes : 2] = -0.002
    relative = m.contact @ d
    assert np.max(np.abs(relative[1::3])) == pytest.approx(0, abs=1e-15)
    for component in (0, 1):
        row = m.contact[1, component]
        assert np.sum(row[2 * nodes :]) == pytest.approx(0, abs=1e-15)


def test_quadratic_force_is_energy_derivative():
    m = model()
    rng = np.random.default_rng(34)
    state = rng.normal(size=m.ndof) * 1e-4
    direction = rng.normal(size=m.ndof)
    h = 1e-7
    plus, minus = state + h * direction, state - h * direction
    finite = (plus @ m.stiffness @ plus - minus @ m.stiffness @ minus) / (4 * h)
    assert finite == pytest.approx(direction @ m.stiffness @ state, rel=1e-10)


def test_free_slip_approaches_individual_wire_bending_under_refinement():
    errors = []
    for n in (8, 16, 32):
        m = model(elements=n, friction=0)
        result = m.run([0, 0.001])
        apparent = result.force_N[-1] * m.span**3 / (48 * 0.001)
        errors.append(abs(apparent - m.individual_bending))
        assert np.max(np.abs(result.friction_work_J)) == 0
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] / 2 < 0.01


def test_reversal_work_and_mirrored_lays():
    history = np.r_[
        np.linspace(0, 0.003, 13),
        np.linspace(0.003, -0.003, 25)[1:],
        np.linspace(-0.003, 0.003, 25)[1:],
    ]
    left, right = model(handedness=-1).run(history), model().run(history)
    np.testing.assert_allclose(left.force_N, right.force_N, rtol=1e-7, atol=1e-6)
    assert np.all(np.diff(right.friction_work_J) >= -1e-12)
    assert right.friction_work_J[-1] > 0
    assert np.max(np.abs(right.work_residual_J)) < 1e-7
    assert np.max(right.equilibrium_relative_residual) < 1e-7
    assert np.max(np.abs(right.contact_work_gap_J)) < 1e-9
    np.testing.assert_allclose(right.apparent_moment_Nm, right.force_N * 0.4 / 4)
    # Apparent loop work, not physical dissipation divided by span.
    work = np.trapezoid(right.apparent_moment_Nm, right.apparent_curvature_1_m)
    assert work == pytest.approx(3 / 0.4 * np.trapezoid(right.force_N, history))


@pytest.mark.parametrize("elements", [3, 5, True, 4.0])
def test_invalid_mesh_is_rejected(elements):
    with pytest.raises(ValueError):
        model(elements=elements)


@pytest.mark.parametrize("history", [[], [np.nan], [[0, 1]]])
def test_invalid_history_is_rejected(history):
    with pytest.raises(ValueError):
        model().run(history)


def test_locked_sheath_adds_annular_bending_once():
    bare, covered = model(), model(sheath=elastic_sheath())
    kappa = 0.2
    _, bare_state = uniform_curvature_state(bare, kappa)
    _, covered_state = uniform_curvature_state(covered, kappa)
    increment = (
        covered_state @ covered.stiffness @ covered_state
        - bare_state @ bare.stiffness @ bare_state
    )
    exact_ei = 420e6 * math.pi * (0.006**4 - 0.005**4) / 4
    assert increment == pytest.approx(exact_ei * kappa**2 * bare.span, rel=1e-11)
    assert covered.individual_bending == pytest.approx(2.0 + exact_ei)


def test_uniform_sheath_warp_has_exact_annular_shear_energy():
    m = model(sheath=elastic_sheath())
    d = np.zeros(m.ndof)
    start = 2 * (m.elements + 1) * (1 + len(m.rings))
    a, b = 1e-4, -2e-4
    d[start::2], d[start + 1 :: 2] = a, b
    expected = (
        420e6
        / (2 * (1 + 0.35))
        * math.pi
        * (0.006**2 - 0.005**2)
        / 0.005**2
        * (a**2 + b**2)
        * m.span
    )
    assert d @ m.stiffness @ d == pytest.approx(expected, rel=1e-12)


def test_sheath_axial_relaxation_retains_explicit_shear_energy():
    bare, covered = model(), model(sheath=elastic_sheath())
    kappa = 0.2
    _, bare_state = uniform_curvature_state(bare, kappa)
    x, covered_state = uniform_curvature_state(covered, kappa)
    start = 2 * len(x) * (1 + len(covered.rings))
    covered_state[start::2] = -0.005 * kappa * (x - covered.span / 2)
    increment = (
        covered_state @ covered.stiffness @ covered_state
        - bare_state @ bare.stiffness @ bare_state
    )
    exact_ga = 420e6 / (2 * (1 + 0.35)) * math.pi * (0.006**2 - 0.005**2)
    assert increment == pytest.approx(
        exact_ga * kappa**2 * covered.span**3 / 12, rel=1e-11
    )


def test_sheath_contact_uses_relative_motion_and_equal_opposite_force():
    m = model(sheath=elastic_sheath())
    nodes = m.elements + 1
    d = np.zeros(m.ndof)
    for layer in (len(m.rings), len(m.rings) + 1):
        start = 2 * nodes * layer
        d[start : start + 2 * nodes : 2] = 0.001
        d[start + 1 : start + 2 * nodes : 2] = -0.002
    outer = m.contact[len(m.rings) :: len(m.rings) + 1]
    np.testing.assert_allclose(outer @ d, 0, atol=1e-15)
    np.testing.assert_allclose(np.sum(outer, axis=2), 0, atol=1e-15)


def test_elastic_sheath_cycle_has_equilibrium_and_nonnegative_dissipation():
    history = np.r_[
        np.linspace(0, 0.003, 13),
        np.linspace(0.003, -0.003, 25)[1:],
        np.linspace(-0.003, 0.003, 25)[1:],
    ]
    result = model(sheath=elastic_sheath()).run(history)
    assert np.all(np.diff(result.friction_work_J) >= -1e-12)
    assert result.friction_work_J[-1] > 0
    assert np.max(np.abs(result.work_residual_J)) < 1e-7
    assert np.max(result.equilibrium_relative_residual) < 1e-7
    assert np.max(np.abs(result.contact_work_gap_J)) < 1e-9


@pytest.mark.parametrize(
    "values",
    [
        (0, 0.006, 420e6, 0.35),
        (0.006, 0.005, 420e6, 0.35),
        (0.005, 0.006, -420e6, 0.35),
        (0.005, 0.006, 420e6, -1),
        (0.005, 0.006, 420e6, 0.5),
        (0.005, 0.006, 420e6, np.nan),
    ],
)
def test_invalid_sheath_material_or_geometry_is_rejected(values):
    with pytest.raises(ValueError):
        AnnularSheath(*values)


def test_sheath_must_meet_outer_interface():
    with pytest.raises(ValueError, match="outer contact interface"):
        model(sheath=AnnularSheath(0.0051, 0.006, 420e6, 0.35))
