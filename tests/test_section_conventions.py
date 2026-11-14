# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Section sign conventions from rigid-section kinematics, and force balance.

The expected values are built from physics, not from the code under test:
a right-handed frame (x, y, s), a small rotation vector theta = (theta_x,
theta_y, 0) of the section at s displaces the fibre at r = (x, y, 0) along s
by (theta x r)_z = theta_x*y - theta_y*x, so the rotation curvature
k = d theta / ds gives axial strain k_x*y - k_y*x. Virtual work of fibre forces
f then defines the conjugate moment M = (sum f*y, -sum f*x).
"""

import numpy as np
import pytest

from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.section_conventions import (
    CURVATURE_MAPS,
    rotation_to_strain_curvature,
    rotation_to_strain_moment,
    true_tension_from_effective,
)

POINTS = np.array([[0.1, 0.0], [0.0, 0.1], [-0.02, 0.03], [0.04, -0.07]])
FORCES = np.array([100.0, -60.0, 35.0, 12.0])


def _axial_strain_from_rotation(curvature, points):
    """Axial strain of each fibre from the cross product theta x r."""
    k = np.array([curvature[0], curvature[1], 0.0])
    r = np.column_stack((points, np.zeros(len(points))))
    return np.cross(k, r)[:, 2]


@pytest.mark.analytical_sanity
@pytest.mark.parametrize("curvature", [(0.03, -0.02), (0.0, 0.05), (-0.04, 0.0)])
def test_strain_gradient_reproduces_rigid_section_rotation(curvature):
    axial = _axial_strain_from_rotation(curvature, POINTS)
    kx, ky = curvature
    assert np.allclose(axial, kx * POINTS[:, 1] - ky * POINTS[:, 0], atol=1e-18)
    gradient = np.array(rotation_to_strain_curvature(np.array(kx), np.array(ky)))
    assert np.allclose(POINTS @ gradient, axial, atol=1e-18)


@pytest.mark.analytical_sanity
def test_moment_mapping_from_fibre_forces_and_virtual_work():
    curvature = np.array([0.03, -0.02])
    axial = _axial_strain_from_rotation(curvature, POINTS)
    moment = np.array([FORCES @ POINTS[:, 1], -(FORCES @ POINTS[:, 0])])
    assert FORCES @ axial == pytest.approx(moment @ curvature)
    gradient = np.array(rotation_to_strain_curvature(*curvature))
    gradient_moment = np.array(rotation_to_strain_moment(*moment))
    assert np.allclose(gradient_moment, FORCES @ POINTS, rtol=1e-15)
    assert FORCES @ axial == pytest.approx(gradient_moment @ gradient)


@pytest.mark.analytical_sanity
def test_hogging_and_sagging_tensile_fibres():
    """Bending about +y puts the -x fibre in tension, about -y the +x fibre."""
    phi = np.radians(np.arange(0, 360, 15))
    ring = np.column_stack((np.cos(phi), np.sin(phi)))
    for ky, expected_deg in ((0.1, 180.0), (-0.1, 0.0)):
        gradient = np.array(rotation_to_strain_curvature(np.array(0.0), np.array(ky)))
        assert np.degrees(phi[np.argmax(ring @ gradient)]) == expected_deg
    for kx, expected_deg in ((0.1, 90.0), (-0.1, 270.0)):
        gradient = np.array(rotation_to_strain_curvature(np.array(kx), np.array(0.0)))
        assert np.degrees(phi[np.argmax(ring @ gradient)]) == expected_deg


def test_paper_map_is_the_reversed_legacy_mapping():
    kx, ky = np.array([0.03, -0.01]), np.array([-0.02, 0.04])
    legacy = rotation_to_strain_curvature(kx, ky, curvature_map="paper_r2")
    assert np.array_equal(legacy[0], ky) and np.array_equal(legacy[1], -kx)
    right = rotation_to_strain_curvature(kx, ky)
    assert np.array_equal(right[0], -legacy[0])
    assert np.array_equal(right[1], -legacy[1])
    mx, my = np.array([5.0]), np.array([-3.0])
    assert np.array_equal(
        np.array(rotation_to_strain_moment(mx, my, curvature_map="paper_r2")),
        [[-3.0], [-5.0]],
    )
    assert np.array_equal(np.array(rotation_to_strain_moment(mx, my)), [[3.0], [5.0]])
    assert CURVATURE_MAPS == ("right_handed", "paper_r2")


@pytest.mark.parametrize(
    "function", [rotation_to_strain_curvature, rotation_to_strain_moment]
)
def test_unknown_map_and_non_finite_input_are_rejected(function):
    with pytest.raises(CableInputError, match="curvature_map"):
        function(np.zeros(1), np.zeros(1), curvature_map="left_handed")
    with pytest.raises(CableInputError, match="finite"):
        function(np.array([np.nan]), np.zeros(1))


def test_pressure_thrust_of_sealed_solid_cable():
    result = true_tension_from_effective(
        np.array([20000.0, 22000.0]), np.array([1e6, 1.1e6]), 0.05
    )
    assert np.array_equal(result, [-30000.0, -33000.0])


def test_internal_and_external_pressure_force_balance():
    result = true_tension_from_effective(
        np.array([1e5]), 2e6, 0.04, internal_pressure_Pa=5e6, internal_area_m2=0.02
    )
    assert result[0] == 120000.0


@pytest.mark.parametrize("areas", [(-1.0, 0.0), (0.1, 0.2), (float("nan"), 0.0)])
def test_invalid_pressure_areas(areas):
    with pytest.raises(ValueError, match="areas"):
        true_tension_from_effective(
            np.array([1.0]), 1.0, areas[0], internal_area_m2=areas[1]
        )


@pytest.mark.parametrize("kind", ["effective", "true", "unspecified"])
def test_force_convention_survives_exchange_file(tmp_path, kind):
    from cablexsec.io.series import LoadCaseSeries, read_series_csv, write_series_csv

    x = np.array([0.0, 1.0])
    series = LoadCaseSeries(x, x * 0.01, x * 0, x + 100.0, tension_kind=kind)
    restored = read_series_csv(write_series_csv(series, tmp_path / "series.csv"))
    assert restored.tension_kind == kind
    assert np.array_equal(restored.tension_N, series.tension_N)


def test_prescribed_strain_survives_csv_and_public_fatigue_bridge(tmp_path):
    import cablexsec as cx
    from cablexsec.io.series import (
        LoadCaseSeries,
        fatigue_from_series,
        read_series_csv,
        write_series_csv,
    )

    t = np.linspace(0, 10, 101)
    k = 0.01 * np.sin(t)
    strain = 0.0001 + 0.00001 * np.sin(t)
    # A supplied stress-driving strain retains its meaning through exchange.
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    from cablexsec.mechanics.stiffness import ea_fixed

    series = LoadCaseSeries(
        t,
        k,
        np.zeros_like(t),
        ea_fixed(cable) * strain,
        tension_kind="true",
        axial_strain=strain,
        axial_strain_kind="direct",
    )
    restored = read_series_csv(write_series_csv(series, tmp_path / "strain.csv"))
    assert np.array_equal(restored.axial_strain, strain)
    assert restored.axial_strain_kind == "direct"
    actual = fatigue_from_series(cable, restored)
    expected = cx.compute_fatigue(cable, k, np.zeros_like(k), axial_strain=strain)
    for name in expected.D_total_per_wire:
        assert np.allclose(
            actual.D_total_per_wire[name],
            expected.D_total_per_wire[name],
            rtol=1.0e-12,
            atol=1.0e-25,
        )


def test_effective_force_requires_explicit_conversion_before_fatigue():
    import cablexsec as cx
    from cablexsec.exceptions import CableSchemaError
    from cablexsec.io.series import LoadCaseSeries, fatigue_from_series

    x = np.array([0.0, 1.0])
    series = LoadCaseSeries(x, x * 0.01, x * 0, x + 100.0, tension_kind="effective")
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    with pytest.raises(CableSchemaError, match="effective tension is not"):
        fatigue_from_series(cable, series)
    converted = series.with_true_tension(1e6, 0.05)
    assert np.array_equal(series.tension_N, x + 100.0)
    assert np.array_equal(converted.tension_N, x + 100.0 - 50000.0)
    assert converted.tension_kind == "true"
    with pytest.raises(CableSchemaError, match="requires effective"):
        converted.with_true_tension(1e6, 0.05)


@pytest.mark.parametrize("kind", ["total", "unspecified"])
def test_poisson_or_unqualified_strain_is_not_a_direct_stress_input(tmp_path, kind):
    import cablexsec as cx
    from cablexsec.exceptions import CableSchemaError
    from cablexsec.io.series import (
        LoadCaseSeries,
        fatigue_from_series,
        read_series_csv,
        write_series_csv,
    )

    time = np.array([0.0, 1.0, 2.0])
    series = LoadCaseSeries(
        time,
        time * 0,
        time * 0,
        time * 0,
        tension_kind="true",
        axial_strain=np.full(3, 1e-4),
        axial_strain_kind=kind,
    )
    restored = read_series_csv(write_series_csv(series, tmp_path / "strain.csv"))
    assert restored.axial_strain_kind == kind
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    with pytest.raises(CableSchemaError, match="requires direct"):
        fatigue_from_series(cable, restored)


def test_axially_unloaded_pressurised_line_has_no_false_longitudinal_stress():
    import cablexsec as cx
    from cablexsec.analysis.fatigue import wire_stress_field
    from cablexsec.cable_adapters import cable_to_fatigue_layers
    from cablexsec.mechanics.stiffness import ea_fixed

    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    layer = cable_to_fatigue_layers(cable)[0]
    pressure_thrust = 1e4
    total_strain = np.full(3, 2 * 0.3 * pressure_thrust / ea_fixed(cable))
    zero = np.zeros(3)
    direct = wire_stress_field(layer, zero, zero, axial_strain=zero)
    incorrect = wire_stress_field(layer, zero, zero, axial_strain=total_strain)
    np.testing.assert_array_equal(direct.mean_Pa, 0)
    assert np.all(incorrect.mean_Pa > 0)


def test_prescribed_axial_strain_matches_pressure_free_force_input():
    import cablexsec as cx
    from cablexsec.analysis.fatigue import _wire_stress_series
    from cablexsec.cable_adapters import cable_to_fatigue_layers
    from cablexsec.mechanics.stiffness import ea_fixed

    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    layer = cable_to_fatigue_layers(cable)[0]
    kx = 0.01 * np.sin(np.linspace(0, 4 * np.pi, 201))
    ky = 0.002 * np.cos(np.linspace(0, 4 * np.pi, 201))
    strain = 0.0001 + 0.00001 * np.sin(np.linspace(0, 4 * np.pi, 201))
    ea = ea_fixed(cable)
    force = _wire_stress_series(layer, kx, ky, ea * strain, ea, None, None)
    displacement = _wire_stress_series(
        layer, kx, ky, None, None, None, None, axial_strain=strain
    )
    assert np.allclose(force, displacement, rtol=1.0e-13, atol=1.0e-7)
    with pytest.raises(ValueError, match="not both"):
        _wire_stress_series(
            layer, kx, ky, ea * strain, ea, None, None, axial_strain=strain
        )
