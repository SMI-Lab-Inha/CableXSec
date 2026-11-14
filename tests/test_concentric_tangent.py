# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Analytical and load-perturbation checks of condensed section stiffness."""

import math
from dataclasses import replace

import numpy as np
import pytest

from cablexsec.mechanics.concentric_section import (
    ElasticAnnulus,
    ElasticHelicalRing,
    solve_concentric_section,
)

pytestmark = pytest.mark.analytical_sanity


@pytest.mark.parametrize("pressure", [0.0, 2e6])
@pytest.mark.parametrize("free_strain", [0.0, 1e-3])
def test_annulus_tangent_is_not_force_over_total_strain(pressure, free_strain):
    a, b, e, nu = 0.04, 0.05, 70e9, 0.3
    ring = ElasticAnnulus("tube", a, b, e, nu, free_strain)
    result = solve_concentric_section(
        [ring],
        tension_N=1e4,
        torque_Nm=100,
        prescribed_twist_per_m=None,
        internal_pressure_Pa=pressure,
    )
    ea = e * math.pi * (b * b - a * a)
    gj = e / (2 * (1 + nu)) * math.pi / 2 * (b**4 - a**4)
    tangent = result.tangent
    np.testing.assert_allclose(
        tangent.matrix, [[ea, 0], [0, gj]], rtol=1e-12, atol=1e-8
    )
    assert tangent.ea_free_N == pytest.approx(ea, rel=1e-12)
    assert tangent.gj_at_fixed_tension_Nm2 == pytest.approx(gj, rel=1e-12)
    assert not tangent.branch_junction_interfaces
    if pressure or free_strain:
        assert 1e4 / result.axial_strain != pytest.approx(ea, rel=0.001)
    with pytest.raises(ValueError):
        tangent.matrix[0, 0] = 0


def _layer_pair():
    return [
        ElasticAnnulus("core", 0, 0.03, 70e9, 0.3),
        ElasticHelicalRing("armour", 0.032, 0.004, 40, 0.4, 210e9),
    ]


@pytest.mark.parametrize("torque", [-1000.0, 0.0, 1000.0])
def test_contact_branch_tangent_matches_two_independent_load_perturbations(torque):
    layers = _layer_pair()
    tension = 1e5
    base = solve_concentric_section(
        layers,
        tension_N=tension,
        torque_Nm=torque,
        prescribed_twist_per_m=None,
    )
    inverse = np.linalg.inv(base.tangent.matrix)
    for column, increment in enumerate([10.0, 0.1]):
        states = [
            solve_concentric_section(
                layers,
                tension_N=tension + (sign * increment if column == 0 else 0),
                torque_Nm=torque + (sign * increment if column == 1 else 0),
                prescribed_twist_per_m=None,
            )
            for sign in (-1, 1)
        ]
        assert all(
            s.tangent.active_interfaces == base.tangent.active_interfaces
            for s in states
        )
        response = np.array(
            [
                states[1].axial_strain - states[0].axial_strain,
                states[1].twist_per_m - states[0].twist_per_m,
            ]
        ) / (2 * increment)
        np.testing.assert_allclose(response, inverse[:, column], rtol=2e-8, atol=1e-15)
    np.testing.assert_allclose(
        base.tangent.matrix, base.tangent.matrix.T, rtol=1e-12, atol=1e-7
    )


def test_fixed_twist_load_increment_and_reaction_torque_share_the_tangent():
    layers = _layer_pair()
    base = solve_concentric_section(layers, tension_N=1e5)
    loaded = solve_concentric_section(layers, tension_N=1e5 + 10)
    extension = loaded.axial_strain - base.axial_strain
    assert 10 / extension == pytest.approx(base.tangent.ea_fixed_N, rel=1e-8)
    assert (
        loaded.torque_resultant_Nm - base.torque_resultant_Nm
    ) / extension == pytest.approx(
        base.tangent.matrix[1, 0],
        rel=1e-8,
    )


def test_touching_unloaded_interface_is_reported_as_a_branch_junction():
    result = solve_concentric_section(_layer_pair())
    assert result.tangent.branch_junction_interfaces == (0,)
    # Resolving a nonzero load chooses its own open/closed branch. No penalty
    # spring is introduced to invent a unique tangent at first contact.
    loaded = solve_concentric_section(_layer_pair(), tension_N=1e5)
    assert not loaded.tangent.branch_junction_interfaces
    assert loaded.tangent.active_interfaces == (True,)


def test_tangent_does_not_depend_on_thermal_prestrain_with_unchanged_contact():
    layers = _layer_pair()
    warm = [replace(layer, free_thermal_strain=1e-4) for layer in layers]
    cold_state = solve_concentric_section(layers, tension_N=1e5)
    warm_state = solve_concentric_section(warm, tension_N=1e5)
    assert warm_state.tangent.active_interfaces == cold_state.tangent.active_interfaces
    np.testing.assert_allclose(
        warm_state.tangent.matrix, cold_state.tangent.matrix, rtol=1e-12
    )
