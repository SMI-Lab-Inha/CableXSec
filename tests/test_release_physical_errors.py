# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Domain errors and degenerate physical limits across supported entry points."""

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.capacity import wire_yield_curvature
from cablexsec.analysis.fatigue import compute_fatigue
from cablexsec.io.series import LoadCaseSeries, read_series_csv
from cablexsec.materials import library, sn_curves
from cablexsec.mechanics.axisymmetric import capstan_force_per_wire
from cablexsec.mechanics.hertz_contact import (
    crossed_cylinder_curvatures,
    layer_crossing_density_per_m,
)
from cablexsec.mechanics.section_conventions import (
    rotation_to_strain_curvature,
    true_tension_from_effective,
)


@pytest.fixture
def cable():
    return cx.load_cable(cx.inter_array_single_core_yaml_path(), geometry_override=None)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"tension_kind": "bad"}, "tension_kind"),
        ({"axial_strain_kind": "bad"}, "axial_strain_kind"),
        ({"axial_strain": np.zeros(3)}, "axial_strain"),
        ({"tension_N": np.zeros(3)}, "length"),
        ({"temperature_C": np.zeros(3)}, "temperature_C"),
    ],
)
def test_exchange_shape_and_kind_guards(kwargs, match):
    values = {
        "time_s": np.arange(2.0),
        "curvature_x_1_m": np.zeros(2),
        "curvature_y_1_m": np.zeros(2),
        "tension_N": np.zeros(2),
    }
    with pytest.raises(cx.CableSchemaError, match=match):
        LoadCaseSeries(**(values | kwargs))


def test_single_sample_time_step():
    series = LoadCaseSeries(*[np.zeros(1) for _ in range(4)])
    assert series.dt_s == 0
    assert series.n_steps == 1


@pytest.mark.parametrize(
    "header,match",
    [
        (
            "time_s,curvature_x_1_m,curvature_y_1_m,true_tension_N,direct_axial_strain,total_axial_strain",
            "at most one",
        ),
        ("time_s,true_tension_N", "missing columns"),
    ],
)
def test_ambiguous_or_incomplete_exchange_header(tmp_path, header, match):
    path = tmp_path / "exchange.csv"
    path.write_text(header + "\n")
    with pytest.raises(cx.CableSchemaError, match=match):
        read_series_csv(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"polymer_crush_allowable_Pa": 0, "polymer_crush_source": "test"},
        {"polymer_crush_allowable_Pa": 1, "polymer_crush_source": " "},
    ],
)
def test_capacity_requires_positive_documented_allowable(cable, kwargs):
    with pytest.raises(ValueError, match="polymer_crush"):
        cx.compute_capacity(cable, np.array([0]), **kwargs)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"phi_bend": np.nan}, "azimuth"),
        ({"contact_model": "qp", "sigma_T_Pa": 1}, "thermal"),
    ],
)
def test_yield_curvature_rejects_inconsistent_inputs(cable, kwargs, match):
    with pytest.raises(cx.CableSchemaError, match=match):
        wire_yield_curvature(cable, 0, **kwargs)


def test_yield_curvature_rejects_state_and_separate_load(cable):
    state = cx.solve_section_state(cable, contact_model="bounded")
    with pytest.raises(cx.CableSchemaError, match="not both"):
        wire_yield_curvature(cable, 1, section_state=state)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_wires": 0},
        {"r_helix_m": 0},
        {"T_layer_N": np.inf},
        {"lay_angle_rad": np.pi / 2},
    ],
)
def test_capstan_invalid_geometry_and_load(kwargs):
    with pytest.raises(ValueError):
        capstan_force_per_wire(
            **(
                {"T_layer_N": 1, "n_wires": 8, "r_helix_m": 0.01, "lay_angle_rad": 0.2}
                | kwargs
            )
        )


@pytest.mark.parametrize("diameter,angle", [(0, 0.3), (np.inf, 0.3), (0.01, np.nan)])
def test_crossed_cylinder_inputs(diameter, angle):
    with pytest.raises(ValueError):
        crossed_cylinder_curvatures(diameter, angle)


def test_crossing_count_requires_positive_population():
    with pytest.raises(ValueError, match="positive wire count"):
        layer_crossing_density_per_m(0, 0.1, 0.2, 1, 0.2, 0.3, -1)


@pytest.mark.parametrize("pressure", [-1, np.inf])
def test_pressure_thrust_conversion_rejects_invalid_pressure(pressure):
    with pytest.raises(ValueError):
        true_tension_from_effective(np.ones(2), pressure, 0.01)


def test_curvature_conversion_rejects_nonfinite():
    with pytest.raises(ValueError, match="finite"):
        rotation_to_strain_curvature(np.array([np.nan]), np.zeros(1))


@pytest.mark.parametrize(
    "function,args",
    [
        (library.get_citation, ("galvanised_steel_armour", "missing")),
        (library.get_temperature_modulus_citation, ("missing",)),
        (sn_curves.get_sn_curve, ("missing",)),
    ],
)
def test_unknown_material_evidence_is_not_substituted(function, args):
    with pytest.raises(KeyError):
        function(*args)


@pytest.mark.parametrize(
    "function", [cx.compute_shell_stress, cx.compute_shell_fatigue]
)
def test_shell_requires_existing_layer(cable, function):
    args = (0, 0) if function is cx.compute_shell_stress else (np.zeros(2),)
    with pytest.raises(cx.CableSchemaError):
        function(cable, "missing", *args)


def test_shell_fatigue_requires_a_shell(cable):
    with pytest.raises(cx.CableSchemaError, match="cylindrical_shell"):
        cx.compute_shell_fatigue(cable, "inner_armour", np.zeros(2))


def test_shell_fatigue_requires_a_curve():
    cable = cx.load_cable(cx.cigre_tb880_yaml_path(), geometry_override=None)
    with pytest.raises(cx.CableSchemaError, match="no bulk_sn"):
        cx.compute_shell_fatigue(cable, "cu_metallic_sheath", np.zeros(2))


@pytest.mark.parametrize(
    "name", ["normal_stress_Pa", "wire_modulus_Pa", "stick_modulus_Pa"]
)
def test_engine_rejects_unknown_material_history(name):
    with pytest.raises(ValueError, match="exactly"):
        compute_fatigue([], np.zeros(2), np.zeros(2), **{name: {"unknown": np.ones(2)}})
