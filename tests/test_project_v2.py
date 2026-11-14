# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Project format version 2 (five new families) and the preflight report."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.exceptions import CableInputError
from cablexsec.io.series import LoadCaseSeries
from cablexsec.preflight import preflight, preflight_request
from cablexsec.project import (
    AnalysisRequest,
    BendingLawSettings,
    CableModel,
    CapacitySettings,
    CoilingSettings,
    ConductorFatigueSettings,
    Project,
    StiffnessSettings,
    ThermalSettings,
)


def _series(kind: str = "true") -> LoadCaseSeries:
    t = np.linspace(0.0, 30.0, 301)
    return LoadCaseSeries(
        time_s=t,
        curvature_x_1_m=0.05 * np.sin(2 * math.pi * t / 10.0),
        curvature_y_1_m=0.01 * np.cos(2 * math.pi * t / 13.0),
        tension_N=np.full_like(t, 40e3),
        tension_kind=kind,
    )


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    menard = CableModel(
        cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    )
    stranded = CableModel(cx.iea_task49_66kv_stranded())
    chae = CableModel(cx.load_cable(cx.chae2023_yaml_path()))
    proj = Project("v2 families")
    proj.set_model("menard", menard)
    proj.set_model("iea66_stranded", stranded)
    proj.set_model("chae", chae)
    proj.set_load_case("hour", _series())
    proj.set_analysis(
        AnalysisRequest(
            name="law",
            model="menard",
            settings=BendingLawSettings(
                tensions_N=(0.0, 20e3), kappa_max_1_m=0.1, n_points=40
            ),
        )
    )
    proj.set_analysis(
        AnalysisRequest(
            name="capacity",
            model="menard",
            settings=CapacitySettings(tension_N=(0.0, 50e3, 100e3)),
        )
    )
    proj.set_analysis(
        AnalysisRequest(
            name="coil",
            model="chae",
            settings=CoilingSettings(tank_diameter_m=12.0, n_ramp=10, n_per_turn=36),
        )
    )
    proj.set_analysis(
        AnalysisRequest(
            name="conductor",
            model="iea66_stranded",
            load_case="hour",
            settings=ConductorFatigueSettings(),
        )
    )
    proj.set_analysis(
        AnalysisRequest(
            name="rating",
            model="iea66_stranded",
            settings=ThermalSettings(current_A=800.0, conductor_area_m2=630e-6),
        )
    )
    return proj


def test_new_families_run_and_round_trip(project, tmp_path):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        records = {
            name: project.run(name)
            for name in ("law", "capacity", "coil", "conductor", "rating")
        }
    assert isinstance(records["law"].result, cx.BendingLawExport)
    assert records["conductor"].result.critical_ring >= 0
    path = project.save(tmp_path / "v2.cxproj")
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 2
    loaded = Project.load(path)
    for original, restored in zip(project.records, loaded.records, strict=True):
        assert restored.fingerprint == original.fingerprint
        assert type(restored.result) is type(original.result)
    law = loaded.records[0].result
    np.testing.assert_array_equal(law.moment_Nm, records["law"].result.moment_Nm)


def test_version_1_file_loads_and_rejects_new_kinds(tmp_path):
    proj = Project("v1")
    proj.set_model(
        "m",
        CableModel(cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)),
    )
    proj.set_analysis(
        AnalysisRequest(name="k", model="m", settings=StiffnessSettings())
    )
    path = proj.save(tmp_path / "p.cxproj")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["version"] = 1
    path.write_text(json.dumps(document), encoding="utf-8")
    assert Project.load(path).analyses["k"].settings.kind == "stiffness"
    document["payload"]["analyses"].append(
        AnalysisRequest(
            name="c", model="m", settings=CapacitySettings(tension_N=(0.0,))
        ).model_dump(mode="json")
    )
    from cablexsec._project_codec import canonical
    from cablexsec.project import _digest

    document["sha256"] = _digest(canonical(document["payload"]))
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CableInputError, match="version 2"):
        Project.load(path)


def test_preflight_finds_known_unsupported_combinations(project):
    report = preflight(project)
    codes = {(x.analysis, x.code) for x in report.issues}
    assert ("law", "planar-law") in codes
    assert report.ok
    stranded = project.models["iea66_stranded"].definition
    bad = preflight_request(
        stranded,
        AnalysisRequest(
            name="x",
            model="m",
            settings=CapacitySettings(tension_N=(0.0,), contact_model="qp"),
        ),
    )
    assert {x.code for x in bad} >= {"qp-offset-cores", "strand-bundle-radial"}
    effective = preflight_request(
        stranded,
        AnalysisRequest(
            name="y", model="m", load_case="h", settings=ConductorFatigueSettings()
        ),
        _series("effective"),
    )
    assert "tension-kind" in {x.code for x in effective}
    coarse = preflight_request(
        project.models["menard"].definition,
        AnalysisRequest(
            name="z",
            model="m",
            settings=BendingLawSettings(tensions_N=(0.0, 100e3), kappa_max_1_m=0.1),
        ),
    )
    assert "coarse-tension-grid" in {x.code for x in coarse}
