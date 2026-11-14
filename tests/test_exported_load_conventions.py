# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Exported histories retain material force and signed bending conventions."""

from dataclasses import replace

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.exceptions import CableSchemaError
from cablexsec.io.series import LoadCaseSeries
from cablexsec.materials import library
from cablexsec.materials.sn_curves import get_sn_curve


def write_csv(tmp_path, extra_header="", extra_values=None):
    path = tmp_path / "line.csv"
    header = "Time (s),x curvature (1/m),y curvature (1/m),Wall tension (kN)"
    rows = ["0,0.01,-0.02,20", "1,-0.03,0.04,60", "2,0.01,-0.02,20"]
    if extra_header:
        header += "," + extra_header
        rows = [
            row + "," + str(value)
            for row, value in zip(rows, extra_values, strict=True)
        ]
    path.write_text("\n".join([header, *rows]), encoding="utf-8")
    return path


def cable():
    return cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)


def test_native_csv_retains_force_and_converts_signed_rotation_components(tmp_path):
    series = cx.read_time_history_csv(write_csv(tmp_path))
    assert series.tension_kind == "true"
    np.testing.assert_array_equal(series.tension_N, [20000, 60000, 20000])
    # Right-handed rotation components (k_x, k_y) -> gradient (-k_y, k_x).
    np.testing.assert_array_equal(series.curvature_x_1_m, [0.02, -0.04, 0.02])
    np.testing.assert_array_equal(series.curvature_y_1_m, [0.01, -0.03, 0.01])
    # Strain at a fixed point equals (rotation x position)'s axial component,
    # k_x*y - k_y*x.
    x, y = 0.025, -0.04
    kx, ky = np.array([0.01, -0.03, 0.01]), np.array([-0.02, 0.04, -0.02])
    expected = np.cross(np.column_stack((kx, ky, 0 * kx)), np.array([x, y, 0.0]))[:, 2]
    np.testing.assert_allclose(expected, kx * y - ky * x)
    np.testing.assert_allclose(
        series.curvature_x_1_m * x + series.curvature_y_1_m * y, expected
    )


@pytest.mark.parametrize(
    "kind,column",
    [("direct", "Direct tensile strain (%)"), ("total", "Total mean axial strain (%)")],
)
def test_exported_strain_kind_and_percent_scale_survive_exchange(
    tmp_path, kind, column
):
    path = write_csv(tmp_path, column, [0.01, 0.02, 0.01])
    series = cx.read_time_history_csv(path, axial_strain_col=column)
    assert series.axial_strain_kind == kind
    np.testing.assert_array_equal(series.axial_strain, [1e-4, 2e-4, 1e-4])
    back = cx.read_series_csv(cx.write_series_csv(series, tmp_path / "exchange.csv"))
    assert back.axial_strain_kind == kind
    assert back.tension_kind == "true"
    np.testing.assert_array_equal(back.axial_strain, series.axial_strain)
    if kind == "total":
        with pytest.raises(CableSchemaError, match="requires direct"):
            cx.fatigue_from_series(cable(), back)


def test_effective_force_csv_needs_pressure_thrust_conversion(tmp_path):
    path = write_csv(tmp_path)
    path.write_text(path.read_text().replace("Wall tension", "Effective tension"))
    series = cx.read_time_history_csv(path, tension_col="Effective tension")
    assert series.tension_kind == "effective"
    with pytest.raises(CableSchemaError, match="effective tension is not"):
        cx.fatigue_from_series(cable(), series)
    converted = series.with_true_tension(np.array([1e6, 2e6, 1e6]), 0.01)
    np.testing.assert_array_equal(converted.tension_N, [10000, 40000, 10000])
    direct = cx.compute_fatigue(
        cable(),
        converted.curvature_x_1_m,
        converted.curvature_y_1_m,
        tension_N=converted.tension_N,
    )
    actual = cx.fatigue_from_series(cable(), converted)
    for name in direct.D_bulk_per_fibre:
        np.testing.assert_array_equal(
            actual.D_bulk_per_fibre[name], direct.D_bulk_per_fibre[name]
        )


def test_custom_force_column_needs_an_explicit_kind(tmp_path):
    path = write_csv(tmp_path)
    path.write_text(path.read_text().replace("Wall tension", "Load"))
    series = cx.read_time_history_csv(path, tension_col="Load")
    assert series.tension_kind == "unspecified"
    with pytest.raises(CableSchemaError, match="unqualified"):
        cx.fatigue_from_series(cable(), series)
    declared = cx.read_time_history_csv(path, tension_col="Load", tension_kind="true")
    assert declared.tension_kind == "true"


@pytest.mark.parametrize(
    "changes",
    [
        {"tension_kind": "effective"},
        {"tension_kind": "bad"},
        {"axial_strain_col": "Direct tensile strain", "axial_strain_kind": "total"},
    ],
)
def test_native_quantity_cannot_be_relabelled(tmp_path, changes):
    path = write_csv(tmp_path, "Direct tensile strain (%)", [0, 0, 0])
    with pytest.raises(CableSchemaError, match="kind"):
        cx.read_time_history_csv(path, **changes)


@pytest.mark.parametrize(
    "column_arg", ["curvature_y_col", "temperature_col", "axial_strain_col"]
)
def test_requested_column_is_not_silently_omitted(tmp_path, column_arg):
    with pytest.raises(CableSchemaError, match="not found"):
        cx.read_time_history_csv(write_csv(tmp_path), **{column_arg: "Absent"})


def test_ambiguous_exported_line_column_is_rejected(tmp_path):
    path = write_csv(tmp_path, "Another line Wall tension (kN)", [30, 70, 30])
    with pytest.raises(CableSchemaError, match="ambiguous"):
        cx.read_time_history_csv(path)


def test_magnitude_and_repeated_component_selectors_are_rejected(tmp_path):
    path = write_csv(tmp_path)
    with pytest.raises(CableSchemaError, match="magnitude"):
        cx.read_time_history_csv(
            path, curvature_x_col="Curvature", curvature_y_col=None
        )
    with pytest.raises(CableSchemaError, match="ambiguous"):
        cx.read_time_history_csv(path, curvature_y_col="x curvature")


def test_declared_pretransformed_components_are_not_rotated_again(tmp_path):
    series = cx.read_time_history_csv(
        write_csv(tmp_path),
        curvature_convention="strain_gradient",
        curvature_scale_to_1_m=1000,
    )
    np.testing.assert_array_equal(series.curvature_x_1_m, [10, -30, 10])
    np.testing.assert_array_equal(series.curvature_y_1_m, [-20, 40, -20])


@pytest.mark.parametrize(
    "name", ["curvature_scale_to_1_m", "tension_scale_to_N", "axial_strain_scale"]
)
@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan])
def test_invalid_unit_multiplier_is_rejected(tmp_path, name, value):
    with pytest.raises(CableSchemaError, match="multipliers"):
        cx.read_time_history_csv(write_csv(tmp_path), **{name: value})


def test_positional_reader_preserves_declared_force_and_curvature_convention(tmp_path):
    path = tmp_path / "riflex.asc"
    path.write_text("0 0.01 -0.02 20000\n1 -0.03 0.04 60000\n")
    result = cx.read_riflex_ascii(
        path,
        curvature_y_col=3,
        tension_col=4,
        tension_kind="effective",
        curvature_convention="rotation",
    )
    assert result.tension_kind == "effective"
    np.testing.assert_array_equal(result.curvature_x_1_m, [0.02, -0.04])
    np.testing.assert_array_equal(result.curvature_y_1_m, [0.01, -0.03])


def test_force_controlled_heating_is_not_silently_treated_as_restrained():
    zero = np.zeros(5)
    series = LoadCaseSeries(
        np.arange(5, dtype=float),
        zero,
        zero,
        zero,
        temperature_C=np.array([20, 60, 20, 60, 20]),
        tension_kind="true",
    )
    with pytest.raises(CableSchemaError, match="solved section_states"):
        cx.fatigue_from_series(cable(), series, T_ref_C=20)


def test_synthetic_force_retains_its_declared_meaning():
    kwargs = {"duration_s": 20, "dt_s": 0.5, "mean_tension_N": 1e4}
    effective = cx.jonswap_series(**kwargs)
    material = cx.jonswap_series(**kwargs, tension_kind="true")
    assert effective.tension_kind == "effective"
    assert material.tension_kind == "true"
    np.testing.assert_array_equal(effective.tension_N, material.tension_N)
    np.testing.assert_array_equal(effective.curvature_x_1_m, material.curvature_x_1_m)
    with pytest.raises(CableSchemaError, match="effective tension"):
        cx.fatigue_from_series(cable(), effective)


def test_prescribed_strain_heating_matches_analytical_stress_range():
    zero = np.zeros(5)
    series = LoadCaseSeries(
        np.arange(5, dtype=float),
        zero,
        zero,
        zero,
        temperature_C=np.array([20, 60, 20, 60, 20]),
        axial_strain=zero,
        axial_strain_kind="direct",
    )
    section = cable()
    result = cx.fatigue_from_series(section, series, T_ref_C=20)
    for layer in section.helical_layers:
        span_MPa = (
            library.get_property(layer.material, "E_Pa")
            * library.get_property(layer.material, "alpha_T_K")
            * 40
            / 1e6
        )
        expected = 2 / get_sn_curve(layer.bulk_sn).cycles_to_failure(span_MPa)
        np.testing.assert_allclose(
            result.D_bulk_per_fibre[layer.name], expected, rtol=1e-12
        )
    # A supplied direct strain, not the auxiliary force column, controls this path.
    alternate = cx.fatigue_from_series(
        section, replace(series, tension_N=zero + 1e6), T_ref_C=20
    )
    for name in result.D_bulk_per_fibre:
        np.testing.assert_array_equal(
            result.D_bulk_per_fibre[name], alternate.D_bulk_per_fibre[name]
        )
