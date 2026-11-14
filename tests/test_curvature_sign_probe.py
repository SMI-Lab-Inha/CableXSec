# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tensile fibre against a global line model: the curvature sign regression.

``tests/data/curvature_sign_probe.json`` holds four static results of a global
line model (a sagging pinned span and a hogging clamped cantilever, each with
End A gamma 0 and 90 deg): x/y curvature, x/y bend moment and the outer ZZ
stress around the circumference. No global solver is needed to run these
tests. The physically tensile fibre is taken from the global model's own
stress, and the geometry (concave up or down) confirms it independently.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.fatigue import wire_stress_field
from cablexsec.cable_adapters import cable_to_fatigue_layers
from cablexsec.io.time_history_csv import read_time_history_csv
from cablexsec.mechanics.section_conventions import (
    rotation_to_strain_curvature,
    rotation_to_strain_moment,
)

PROBE = json.loads(
    (Path(__file__).parent / "data" / "curvature_sign_probe.json").read_text(
        encoding="utf-8"
    )
)
CASES = {f"{c['case']}_g{int(c['gamma_deg'])}": c for c in PROBE["cases"]}


def _angle_difference_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


@pytest.mark.parametrize("name", sorted(CASES))
def test_model_stress_follows_right_handed_strain(name):
    """The global model's own ZZ stress varies as k_x*y - k_y*x around the section."""
    case = CASES[name]
    theta = np.radians(case["theta_deg"])
    stress = np.asarray(case["zz_stress_outer_kPa"])
    shape = case["x_curvature_1_m"] * np.sin(theta) - case["y_curvature_1_m"] * np.cos(
        theta
    )
    bending = stress - stress.mean()
    assert np.corrcoef(bending, shape)[0, 1] > 0.999
    assert case["theta_deg"][int(np.argmax(stress))] == case["theta_max_tension_deg"]


def test_geometry_confirms_the_tensile_side():
    """Sag is concave up, the cantilever root concave down (hogging)."""
    for name, case in CASES.items():
        if name.startswith("sag"):
            assert case["concavity_d2z_m"] > 0
        else:
            assert case["concavity_d2z_m"] < 0
    # Same physical bend, local axes turned by gamma: the tensile fibre turns
    # with the axes, from 180 deg (cantilever, gamma 0) to 90 deg (gamma 90).
    assert CASES["cantilever_g0"]["theta_max_tension_deg"] == 180.0
    assert CASES["cantilever_g90"]["theta_max_tension_deg"] == 90.0
    assert CASES["sag_g0"]["theta_max_tension_deg"] == 0.0
    assert CASES["sag_g90"]["theta_max_tension_deg"] == 270.0


@pytest.mark.parametrize("name", sorted(CASES))
def test_gradient_and_moment_point_to_the_model_tensile_fibre(name):
    case = CASES[name]
    gx, gy = rotation_to_strain_curvature(
        np.array(case["x_curvature_1_m"]), np.array(case["y_curvature_1_m"])
    )
    direction = math.degrees(math.atan2(float(gy), float(gx))) % 360.0
    assert _angle_difference_deg(direction, case["theta_max_tension_deg"]) < 1e-9
    mx, my = rotation_to_strain_moment(
        np.array(case["x_bend_moment_kNm"]), np.array(case["y_bend_moment_kNm"])
    )
    moment_direction = math.degrees(math.atan2(float(my), float(mx))) % 360.0
    assert _angle_difference_deg(moment_direction, case["theta_max_tension_deg"]) < 1e-9


@pytest.mark.parametrize("name", sorted(CASES))
def test_csv_reader_and_wire_stress_put_tension_at_the_physical_fibre(tmp_path, name):
    """CSV export -> reader -> armour wire stress: largest tension where the
    global model has it (within one wire pitch), for 55 and 63 wire layers."""
    case = CASES[name]
    ramp = np.linspace(0.0, 1.0, 41)
    rows = ["Time (s),L x curvature (rad/m),L y curvature (rad/m),L Wall tension (kN)"]
    rows += [
        f"{0.1 * i:.3f},{r * case['x_curvature_1_m']:.12e},"
        f"{r * case['y_curvature_1_m']:.12e},{r * case['wall_tension_kN']:.12e}"
        for i, r in enumerate(ramp)
    ]
    path = tmp_path / "export.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    series = read_time_history_csv(path)
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    for layer in cable_to_fatigue_layers(cable):
        field = wire_stress_field(layer, series.curvature_x_1_m, series.curvature_y_1_m)
        phi = np.degrees(layer.spec.phi_wires) % 360.0
        final = field.mean_Pa[:, -1]
        pitch = 360.0 / layer.spec.n_wires
        tensile = float(phi[int(np.argmax(final))])
        compressive = float(phi[int(np.argmin(final))])
        assert _angle_difference_deg(tensile, case["theta_max_tension_deg"]) <= pitch
        assert (
            _angle_difference_deg(compressive, case["theta_max_tension_deg"] + 180.0)
            <= pitch
        )


def test_legacy_paper_map_is_half_a_turn_off():
    """The submitted (paper_r2) mapping puts the cantilever tension at 0 deg."""
    case = CASES["cantilever_g0"]
    gx, gy = rotation_to_strain_curvature(
        np.array(case["x_curvature_1_m"]),
        np.array(case["y_curvature_1_m"]),
        curvature_map="paper_r2",
    )
    direction = math.degrees(math.atan2(float(gy), float(gx))) % 360.0
    assert _angle_difference_deg(direction, case["theta_max_tension_deg"]) == 180.0
