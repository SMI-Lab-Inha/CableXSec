# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Return-map invariants, numerical scaling and transactional state updates."""

import numpy as np
import pytest

from cablexsec.mechanics.mindlin_contact import MindlinContact
from cablexsec.mechanics.return_mapping import CoulombElement, coulomb_return_map


@pytest.mark.parametrize("scale", [1e-250, 1e-170, 1.0, 1e170, 1e250])
def test_vector_radial_projection_is_force_scale_invariant(scale):
    force, slip, slipping = coulomb_return_map(
        np.zeros(2), np.array([3.0, 4.0]), scale, 2 * scale, vector=True
    )
    np.testing.assert_allclose(force / scale, [1.2, 1.6], rtol=5e-15)
    np.testing.assert_allclose(slip, [1.8, 2.4], rtol=5e-15)
    assert slipping


def test_large_vector_with_unrepresentable_norm_has_finite_projection():
    force, _, slipping = coulomb_return_map(
        np.zeros(2), np.ones(2), 1.5e308, 1.0, vector=True
    )
    np.testing.assert_allclose(force, np.full(2, 1 / np.sqrt(2)), rtol=5e-15)
    assert slipping


def test_tiny_nonzero_vector_at_zero_friction_is_fully_released():
    force, slip, slipping = coulomb_return_map(
        np.zeros(2), np.array([1e-250, -1e-250]), 1.0, 0.0, vector=True
    )
    np.testing.assert_array_equal(force, 0)
    np.testing.assert_array_equal(slip, [1e-250, -1e-250])
    assert slipping


def test_extreme_ratio_does_not_underflow_the_projected_force():
    force, _, slipping = coulomb_return_map(
        np.zeros(2), np.array([3e300, 4e300]), 1.0, 5e-300, vector=True
    )
    np.testing.assert_allclose(force / 1e-300, [3, 4], rtol=5e-15)
    assert slipping


@pytest.mark.parametrize("population", [2, 3, 7])
def test_vector_stiffness_is_per_element_not_per_component(population):
    stiffness = np.arange(1, population + 1, dtype=float)
    increment = np.tile([0.3, 0.4], (population, 1))
    force, slip, slipping = coulomb_return_map(
        np.zeros_like(increment), increment, stiffness, 1e6, vector=True
    )
    np.testing.assert_allclose(force, stiffness[:, None] * increment)
    np.testing.assert_array_equal(slip, 0)
    np.testing.assert_array_equal(slipping, False)


def test_batched_projection_matches_independent_scalar_contact_solutions():
    increments = np.array([[3, 4], [-4, 3], [-3, -4.0]])
    stiffness = np.array([2, 3, 4.0])
    caps = np.array([4, 6, 8.0])
    force, slip, flags = coulomb_return_map(
        np.zeros_like(increments), increments, stiffness, caps, vector=True
    )
    np.testing.assert_allclose(force, increments * (caps / 5)[:, None])
    np.testing.assert_allclose(slip, increments * 0.6)
    np.testing.assert_array_equal(flags, True)


@pytest.mark.parametrize(
    "name,value",
    [
        ("f_prev", np.nan),
        ("du", np.inf),
        ("k", 0),
        ("k", -1),
        ("k", np.inf),
        ("f_y", -1),
        ("f_y", np.inf),
        ("f_y", np.nan),
        ("f_prev", 1j),
        ("du", 1j),
        ("k", 1j),
        ("f_y", 1j),
    ],
)
def test_invalid_pure_function_input_rejected(name, value):
    kwargs = {"f_prev": 0.0, "du": 0.1, "k": 1.0, "f_y": 1.0}
    kwargs[name] = value
    with pytest.raises(ValueError):
        coulomb_return_map(**kwargs)


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 1)])
def test_vector_component_shape_is_explicit(shape):
    with pytest.raises(ValueError):
        coulomb_return_map(np.zeros(shape), np.zeros(shape), 1.0, 1.0, vector=True)


@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize(
    "bad", ["negative_cap", "nan_increment", "extra_population", "overflow"]
)
def test_failed_element_update_preserves_entire_state(vector, bad):
    element = CoulombElement(2.0, vector=vector)
    delta = np.array([1.0, -0.5]) if vector else 1.0
    element.update(delta, 0.5)
    before = (element.f.copy(), element.slip_total.copy(), element.k)
    increment, cap = delta, 0.5
    if bad == "negative_cap":
        cap = -1
    elif bad == "nan_increment":
        increment = np.full_like(delta, np.nan)
    elif bad == "extra_population":
        increment = np.ones((3, 2)) if vector else np.ones(3)
    else:
        increment = np.full_like(delta, 1.7e308)
    with pytest.raises(ValueError):
        element.update(increment, cap)
    np.testing.assert_array_equal(element.f, before[0])
    np.testing.assert_array_equal(element.slip_total, before[1])
    assert element.k == before[2]


@pytest.mark.parametrize("vector", [False, True])
def test_slip_accumulation_overflow_does_not_commit_force(vector):
    element = CoulombElement(1.0, vector=vector)
    element.slip_total[...] = 1.7e308
    before_force, before_slip = element.f.copy(), element.slip_total.copy()
    with pytest.raises(ValueError):
        element.update(np.full_like(element.f, 1.7e308), 0.5)
    np.testing.assert_array_equal(element.f, before_force)
    np.testing.assert_array_equal(element.slip_total, before_slip)


def test_stiffness_rescaling_does_not_overflow_an_intermediate_ratio():
    element = CoulombElement(1e-200)
    element.update(1.0, 1.0)
    element.set_stiffness(1e200)
    assert float(element.f) == pytest.approx(1e200, rel=1e-14)
    assert element.k == 1e200


def test_stiffness_rescaling_does_not_underflow_an_intermediate_ratio():
    element = CoulombElement(1e200)
    element.update(1.0, 1e300)
    element.set_stiffness(1e-200)
    assert float(element.f) / 1e-200 == pytest.approx(1.0, rel=1e-14)


def test_failed_stiffness_rescaling_is_atomic():
    element = CoulombElement(1.0)
    element.update(2.0, 3.0)
    with pytest.raises(ValueError):
        element.set_stiffness(1.7e308)
    assert float(element.f) == 2.0
    assert element.k == 1.0


def test_unrepresentable_rescaled_force_is_not_silently_zeroed():
    element = CoulombElement(1.0)
    element.update(1e-200, 1.0)
    with pytest.raises(ValueError, match="range"):
        element.set_stiffness(1e-200)
    assert float(element.f) == 1e-200
    assert element.k == 1.0


@pytest.mark.parametrize("value", [1j, np.array([1, 2]), np.nan, np.inf])
def test_element_stiffness_must_be_a_real_finite_scalar(value):
    with pytest.raises(ValueError):
        CoulombElement(value)
    element = CoulombElement(1.0)
    with pytest.raises(ValueError):
        element.set_stiffness(value)
    assert element.k == 1.0


@pytest.mark.parametrize("mu", [-0.1, np.nan, np.inf])
def test_invalid_mindlin_friction_rejected_at_construction(mu):
    with pytest.raises(ValueError):
        MindlinContact(K_t=10.0, mu=mu)


@pytest.mark.parametrize(
    "increment,normal,pressure",
    [
        ([np.nan, 0], 2, 3),
        ([1, 0], -1, 3),
        ([1, 0], np.inf, 3),
        ([1, 0], 2, -1),
        ([1, 0], 2, np.nan),
        ([1, 0], 2, np.inf),
        ([1, 0, 0], 2, 3),
        ([1j, 0], 2, 3),
        ([1e308, 0], 2, 3),
    ],
)
def test_failed_contact_update_preserves_force_slip_trajectory_and_pressure(
    increment, normal, pressure
):
    contact = MindlinContact(K_t=10.0, mu=0.3)
    contact.update(np.array([0.2, -0.1]), 2.0, 3.0)
    before = (
        contact._element.f.copy(),
        contact._element.slip_total.copy(),
        contact._delta_total.copy(),
        np.array(contact._slip_traj),
        contact._p0,
    )
    with pytest.raises(ValueError):
        contact.update(increment, normal, pressure)
    for actual, expected in zip(
        (
            contact._element.f,
            contact._element.slip_total,
            contact._delta_total,
            np.array(contact._slip_traj),
        ),
        before[:4],
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    assert contact._p0 == before[4]


def test_random_vector_projections_satisfy_cap_flow_and_hold_invariants():
    rng = np.random.default_rng(931)
    increments = rng.normal(size=(2000, 2))
    caps = rng.uniform(0.01, 2, 2000)
    force, slip, flags = coulomb_return_map(
        np.zeros_like(increments), increments, 1.0, caps, vector=True
    )
    assert np.all(np.hypot(force[:, 0], force[:, 1]) <= caps)
    np.testing.assert_allclose(force + slip, increments, atol=2e-16)
    assert np.all(np.einsum("ij,ij->i", force, slip) >= -1e-15)
    np.testing.assert_allclose(
        (force[:, 0] * slip[:, 1] - force[:, 1] * slip[:, 0])[flags], 0, atol=1e-15
    )
    held, extra_slip, held_flags = coulomb_return_map(
        force, np.zeros_like(force), 1.0, caps, vector=True
    )
    np.testing.assert_array_equal(held, force)
    np.testing.assert_array_equal(extra_slip, 0)
    assert not held_flags.any()


def test_vector_population_broadcasts_over_two_leading_axes():
    increments = np.array([[[3.0, 4.0]], [[-4.0, 3.0]]])
    force, slip, flags = coulomb_return_map(
        0.0, increments, np.array([1, 2, 3.0]), np.array([2, 4, 6.0]), vector=True
    )
    np.testing.assert_allclose(
        force, increments * np.array([0.4, 0.8, 1.2])[None, :, None]
    )
    np.testing.assert_allclose(slip, np.broadcast_to(increments * 0.6, (2, 3, 2)))
    assert flags.shape == (2, 3)
    assert flags.all()


def test_scalar_increment_broadcasts_to_vector_components():
    force, slip, _ = coulomb_return_map(0.0, 0.1, 2.0, 1.0, vector=True)
    np.testing.assert_array_equal(force, [0.2, 0.2])
    np.testing.assert_array_equal(slip, 0)


def test_vector_return_is_covariant_under_arbitrary_frame_rotation():
    angle = 0.719
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    previous = np.array([[0.3, -0.7], [0.1, 0.2]])
    increment = np.array([[0.9, 0.4], [-0.03, 0.04]])
    original = coulomb_return_map(previous, increment, [3, 4], [1, 2], vector=True)
    rotated = coulomb_return_map(
        previous @ rotation.T, increment @ rotation.T, [3, 4], [1, 2], vector=True
    )
    for actual, expected in zip(rotated[:2], original[:2], strict=True):
        np.testing.assert_allclose(
            actual, expected @ rotation.T, rtol=2e-14, atol=1e-15
        )
    np.testing.assert_array_equal(rotated[2], original[2])


def test_element_cap_population_cannot_expand_during_update():
    element = CoulombElement(2.0, shape=(3,))
    element.update(0.1, 2.0)
    before = element.f.copy()
    with pytest.raises(ValueError):
        element.update(0.1, np.ones((2, 3)))
    np.testing.assert_array_equal(element.f, before)


def test_element_compatible_broadcast_preserves_population():
    element = CoulombElement(2.0, shape=(2, 3))
    force, flags = element.update(np.full((1, 3), 0.1), np.ones(3))
    np.testing.assert_array_equal(force, np.full((2, 3), 0.2))
    assert flags.shape == (2, 3)


def test_mindlin_changed_stiffness_rejected_without_state_mutation():
    contact = MindlinContact(10.0, 0.3)
    contact.K_t = 20.0
    with pytest.raises(ValueError, match="fixed"):
        contact.update(np.array([0.1, 0]), 1.0)
    np.testing.assert_array_equal(contact._element.f, 0)
    np.testing.assert_array_equal(contact._delta_total, 0)


def test_mindlin_accumulated_displacement_overflow_preserves_accepted_step():
    contact = MindlinContact(1e-308, 0.0)
    contact.update(np.array([1.7e308, 0]), 1.0)
    before = np.array(contact._slip_traj)
    with pytest.raises(ValueError, match="range"):
        contact.update(np.array([1.7e308, 0]), 1.0)
    np.testing.assert_array_equal(contact._slip_traj, before)
    np.testing.assert_array_equal(contact._delta_total, [1.7e308, 0])


def test_mindlin_unrepresentable_friction_limit_is_rejected():
    contact = MindlinContact(1.0, 1e200)
    with pytest.raises(ValueError, match="finite"):
        contact.update(np.array([1.0, 0]), 1e200)
    np.testing.assert_array_equal(contact._element.f, 0)
