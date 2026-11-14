# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Closed-form independent limits for coupled concentric section mechanics."""

import math

import numpy as np
import pytest

from cablexsec.mechanics.axisymmetric import lame_thick_wall
from cablexsec.mechanics.concentric_section import (
    ElasticAnnulus,
    ElasticHelicalRing,
    solve_concentric_section,
)


@pytest.mark.parametrize("bore", [0.0, 0.04])
def test_axial_stiffness_includes_poisson_relaxation(bore):
    layer = ElasticAnnulus("metal", bore, 0.05, 210e9, 0.3)
    result = solve_concentric_section([layer], tension_N=1e5)
    expected = 1e5 / (210e9 * math.pi * (0.05**2 - bore**2))
    assert result.axial_strain == pytest.approx(expected, rel=1e-12)
    assert result.outer_displacement_m["metal"] == pytest.approx(
        -0.3 * expected * 0.05, rel=1e-12
    )
    assert result.axial_resultant_N == pytest.approx(1e5, rel=1e-12)


def test_free_uniform_heating_gives_zero_resultants():
    thermal_strain = 0.001
    layers = [
        ElasticAnnulus("inner", 0.0, 0.04, 110e9, 0.34, thermal_strain),
        ElasticAnnulus("outer", 0.04, 0.05, 2e9, 0.4, thermal_strain),
    ]
    result = solve_concentric_section(layers)
    assert result.axial_strain == pytest.approx(thermal_strain, rel=1e-12)
    assert result.outer_displacement_m["outer"] == pytest.approx(
        0.05 * thermal_strain, rel=1e-12
    )
    assert abs(result.axial_resultant_N) < 1e-7
    assert np.max(np.abs(result.contact_pressure_Pa)) < 1e-5


def test_pressure_surface_displacement_matches_lame():
    ring = ElasticAnnulus("ring", 0.04, 0.05, 70e9, 0.33)
    result = solve_concentric_section(
        [ring], internal_pressure_Pa=1e6, external_pressure_Pa=2e6
    )
    for radius, value in [
        (0.04, result.inner_displacement_m["ring"]),
        (0.05, result.outer_displacement_m["ring"]),
    ]:
        exact = lame_thick_wall(radius, 0.04, 0.05, 1e6, 2e6, 70e9, 0.33).u_r_m
        assert value == pytest.approx(exact, rel=1e-12)


def test_two_ring_thermal_interference_has_closed_form():
    inner = ElasticAnnulus("inner", 0.02, 0.03, 2e9, 0.0, 0.001)
    outer = ElasticAnnulus("outer", 0.03, 0.04, 210e9, 0.0)
    result = solve_concentric_section([inner, outer])
    ci = -lame_thick_wall(0.03, 0.02, 0.03, 0.0, 1.0, 2e9, 0.0).u_r_m
    co = lame_thick_wall(0.03, 0.03, 0.04, 1.0, 0.0, 210e9, 0.0).u_r_m
    expected = 0.03 * 0.001 / (ci + co)
    assert result.contact_pressure_Pa[0] == pytest.approx(expected, rel=1e-12)
    assert abs(result.gap_m[0]) < 1e-15


def test_open_contact_does_not_carry_tension():
    result = solve_concentric_section(
        [
            ElasticAnnulus("inner", 0.0, 0.03, 70e9, 0.3),
            ElasticAnnulus("outer", 0.0301, 0.04, 2e9, 0.4, 0.002),
        ]
    )
    assert result.contact_pressure_Pa[0] == 0.0
    assert result.gap_m[0] > 0.0


def test_helix_force_and_pressure_follow_same_strain_energy():
    core = ElasticAnnulus("core", 0.0, 0.03, 210e9, 0.3)
    helix = ElasticHelicalRing("armour", 0.032, 0.004, 40, math.radians(25), 210e9)
    result = solve_concentric_section([core, helix], tension_N=1e5)
    sine, cosine = (
        math.sin(helix.signed_lay_angle_rad),
        math.cos(helix.signed_lay_angle_rad),
    )
    strain = (
        cosine**2 * result.axial_strain
        + sine**2 * result.inner_displacement_m["armour"] / 0.032
    )
    tension_wire = 210e9 * math.pi * 0.004**2 / 4 * strain
    expected = 40 * tension_wire * sine**2 / (0.032 * cosine)
    assert result.contact_line_force_N_per_m[0] == pytest.approx(expected, rel=1e-10)
    assert result.axial_resultant_N == pytest.approx(1e5, rel=1e-11)
    assert result.relative_equilibrium_residual < 1e-10


def test_axial_torsional_tangent_is_reciprocal_with_contact():
    layers = [
        ElasticAnnulus("core", 0.0, 0.03, 70e9, 0.3),
        ElasticHelicalRing(
            "armour", 0.032, 0.004, 40, 0.4, 210e9, initial_wire_strain=0.0001
        ),
    ]
    results = [
        solve_concentric_section(
            layers, tension_N=t, torque_Nm=q, prescribed_twist_per_m=None
        )
        for t, q in [(1e5, 0.0), (1e5 + 1.0, 0.0), (1e5, 1.0)]
    ]
    assert results[0].contact_pressure_Pa[0] > 0.0
    cross1 = results[1].twist_per_m - results[0].twist_per_m
    cross2 = results[2].axial_strain - results[0].axial_strain
    assert cross1 == pytest.approx(cross2, rel=1e-8)


def test_refinement_for_thick_annulus():
    layer = ElasticAnnulus("ring", 0.01, 0.05, 70e9, 0.3)
    a = solve_concentric_section([layer], internal_pressure_Pa=1e6, quadrature_order=24)
    b = solve_concentric_section([layer], internal_pressure_Pa=1e6, quadrature_order=48)
    assert a.inner_displacement_m["ring"] == pytest.approx(
        b.inner_displacement_m["ring"], rel=1e-10
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tension_N": float("nan")},
        {"torque_Nm": float("inf")},
        {"prescribed_twist_per_m": float("nan")},
        {"internal_pressure_Pa": -1},
        {"external_pressure_Pa": -1},
        {"quadrature_order": True},
        {"quadrature_order": 3},
        {"quadrature_order": 4.5},
        {"torque_Nm": 1, "prescribed_twist_per_m": 0},
        {"internal_pressure_Pa": 1},
    ],
)
def test_invalid_load_or_quadrature_is_rejected(kwargs):
    core = ElasticAnnulus("core", 0, 0.03, 70e9, 0.3)
    with pytest.raises(ValueError):
        solve_concentric_section([core], **kwargs)


@pytest.mark.parametrize(
    "modification",
    [
        {"youngs_modulus_Pa": 0},
        {"youngs_modulus_Pa": float("nan")},
        {"poisson_ratio": 0.5},
        {"poisson_ratio": float("nan")},
        {"outer_radius_m": -0.01},
        {"free_thermal_strain": float("inf")},
    ],
)
def test_invalid_annulus_is_rejected(modification):
    from dataclasses import replace

    core = ElasticAnnulus("core", 0, 0.03, 70e9, 0.3)
    with pytest.raises(ValueError):
        solve_concentric_section([replace(core, **modification)])


def test_duplicate_layers_and_initial_overlap_are_rejected():
    core = ElasticAnnulus("core", 0, 0.03, 70e9, 0.3)
    with pytest.raises(ValueError, match="unique"):
        solve_concentric_section([core, core])
    with pytest.raises(ValueError, match="overlap"):
        solve_concentric_section([core, ElasticAnnulus("shell", 0.02, 0.04, 70e9, 0.3)])


def test_isolated_helix_extension_radial_mechanism_is_rejected():
    layer = ElasticHelicalRing("wire_ring", 0.03, 0.003, 50, 0.4, 210e9)
    with pytest.raises(ValueError, match="mechanism"):
        solve_concentric_section([layer], tension_N=1e4)
