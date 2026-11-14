# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Project runs read the cable's ``materials`` block and refuse what they ignore."""

from __future__ import annotations

import copy
import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

import cablexsec as cx
from cablexsec.cables.schema import load_cable
from cablexsec.exceptions import CableInputError
from cablexsec.io.series import LoadCaseSeries
from cablexsec.materials import library
from cablexsec.materials.properties import MATERIAL_PROPERTIES
from cablexsec.preflight import preflight_request
from cablexsec.project import (
    AnalysisRequest,
    AnalysisRunner,
    BendingSettings,
    CableModel,
    ConductorFatigueSettings,
    FatigueSettings,
    ThermalSettings,
)

N = 40
T = np.arange(N) * 0.1


def _series(**update) -> LoadCaseSeries:
    data = {
        "time_s": T,
        "curvature_x_1_m": 0.01 * np.sin(2 * np.pi * T),
        "curvature_y_1_m": np.zeros(N),
        "tension_N": np.full(N, 50e3),
        "tension_kind": "true",
    }
    data.update(update)
    return LoadCaseSeries(**data)


def _override(tmp_path: Path, key: str, scale: dict[str, float], polymer: bool):
    raw = yaml.safe_load(
        Path(cx.iea_task49_66kv_yaml_path()).read_text(encoding="utf-8")
    )
    body = {
        prop: {"value": value[0] * scale.get(prop, 1.0), "source": "test input"}
        for prop, value in library.get_material(key).items()
        if prop in MATERIAL_PROPERTIES
    }
    if polymer:
        body = {"polymer": True, **body}
    over = copy.deepcopy(raw)
    over["materials"] = {key: body}
    path = tmp_path / "override.yaml"
    path.write_text(yaml.safe_dump(over), encoding="utf-8")
    return load_cable(path)


def test_project_thermal_reads_the_cable_materials(tmp_path) -> None:
    cable = _override(tmp_path, "hdpe_janocha2024", {"lambda_W_mK": 10.0}, True)
    settings = ThermalSettings(current_A=800.0, conductor_area_m2=630e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = cx.compute_thermal(cable, 800.0, 630e-6).T_conductor_C
        library_value = cx.compute_thermal(
            load_cable(cx.iea_task49_66kv_yaml_path()), 800.0, 630e-6
        ).T_conductor_C
        model = CableModel(cable)
        direct = model.thermal(settings).T_conductor_C
        record = AnalysisRunner().run(
            model, AnalysisRequest(name="t", model="m", settings=settings)
        )
    assert expected < library_value - 0.5
    assert direct == expected
    assert record.result.T_conductor_C == expected


def test_project_fatigue_reads_the_cable_materials(tmp_path) -> None:
    cable = _override(tmp_path, "steel_janocha2024", {"alpha_T_K": 3.0}, False)
    series = _series(
        curvature_x_1_m=0.02 * np.sin(2 * np.pi * T),
        axial_strain=5e-4 * np.sin(2 * np.pi * T),
        axial_strain_kind="direct",
        temperature_C=20 + 60 * np.abs(np.sin(2 * np.pi * T)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = cx.fatigue_from_series(cable, series).D_total_per_wire
        model = CableModel(cable)
        direct = model.fatigue(series, FatigueSettings()).D_total_per_wire
        record = AnalysisRunner().run(
            model,
            AnalysisRequest(
                name="f", model="m", settings=FatigueSettings(), load_case="lc"
            ),
            series,
        )
    for name, value in expected.items():
        np.testing.assert_array_equal(direct[name], value)
        np.testing.assert_array_equal(record.result.D_total_per_wire[name], value)


def test_thermal_settings_take_the_conductor_material_from_the_cable() -> None:
    assert (
        ThermalSettings(current_A=1.0, conductor_area_m2=1e-4).conductor_material
        is None
    )


@pytest.mark.parametrize(
    "update",
    [
        {"temperature_C": np.full(N, 20.0)},
        {"axial_strain": 1e-3 * np.sin(2 * np.pi * T), "axial_strain_kind": "direct"},
    ],
)
def test_conductor_fatigue_refuses_histories_it_does_not_use(update) -> None:
    model = CableModel(cx.iea_task49_66kv_stranded())
    settings = ConductorFatigueSettings(n_orientations=1)
    with pytest.raises(CableInputError, match="conductor fatigue does not consume"):
        model.conductor_fatigue(_series(**update), settings)
    request = AnalysisRequest(name="c", model="m", settings=settings, load_case="lc")
    issues = preflight_request(model.definition, request, _series(**update))
    assert "unused-history" in [x.code for x in issues if x.severity == "error"]


@pytest.mark.parametrize(
    "settings,update,code",
    [
        (BendingSettings(), {"temperature_C": np.full(N, 20.0)}, "unused-history"),
        (
            BendingSettings(),
            {"axial_strain": np.zeros(N), "axial_strain_kind": "direct"},
            "unused-history",
        ),
        (FatigueSettings(), {"tension_kind": "unspecified"}, "tension-kind"),
        (FatigueSettings(), {"tension_kind": "effective"}, "tension-kind"),
        (FatigueSettings(), {"temperature_C": np.full(N, 20.0)}, "temperature-force"),
        (
            FatigueSettings(),
            {"axial_strain": np.zeros(N), "axial_strain_kind": "total"},
            "strain-kind",
        ),
    ],
)
def test_preflight_reports_runs_that_fail(settings, update, code) -> None:
    cable = load_cable(cx.iea_task49_66kv_yaml_path())
    series = _series(**update)
    request = AnalysisRequest(name="a", model="m", settings=settings, load_case="lc")
    errors = [
        x.code
        for x in preflight_request(cable, request, series)
        if x.severity == "error"
    ]
    assert code in errors
    model = CableModel(cable)
    with warnings.catch_warnings(), pytest.raises(cx.CableXSecError):
        warnings.simplefilter("ignore")
        if settings.kind == "bending":
            model.bending(series, settings)
        else:
            model.fatigue(series, settings)


def test_preflight_passes_a_valid_fatigue_history() -> None:
    cable = load_cable(cx.iea_task49_66kv_yaml_path())
    series = _series(
        axial_strain=5e-4 * np.sin(2 * np.pi * T),
        axial_strain_kind="direct",
        temperature_C=np.full(N, 20.0),
        tension_kind="unspecified",
    )
    request = AnalysisRequest(
        name="a", model="m", settings=FatigueSettings(), load_case="lc"
    )
    errors = [
        x for x in preflight_request(cable, request, series) if x.severity == "error"
    ]
    assert errors == []
