# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Preflight capability report: every finding code and the report itself."""

from __future__ import annotations

import numpy as np
import pytest
from _bundled import incomplete_cable

import cablexsec as cx
from cablexsec.io.series import LoadCaseSeries
from cablexsec.preflight import (
    PreflightIssue,
    PreflightReport,
    preflight,
    preflight_request,
)
from cablexsec.project import (
    AnalysisRequest,
    BendingLawSettings,
    BendingSettings,
    CableModel,
    ConductorFatigueSettings,
    FatigueSettings,
    Project,
)


def _series(n=50, kind="true", planar=True):
    t = np.linspace(0.0, 10.0, n)
    return LoadCaseSeries(
        time_s=t,
        curvature_x_1_m=0.05 * np.sin(t),
        curvature_y_1_m=np.zeros(n) if planar else 0.01 * np.cos(t),
        tension_N=np.full(n, 30e3),
        tension_kind=kind,
    )


def _layer(cable, name, **update):
    layers = [
        layer.model_copy(update=update) if layer.name == name else layer
        for layer in cable.layers
    ]
    return cable.model_copy(update={"layers": layers})


def _codes(cable, settings, series=None):
    # History analyses name a load case; ``series`` is what the caller has
    # loaded for it (None: checked before the data are available).
    history = settings.kind in ("bending", "fatigue", "conductor_fatigue")
    request = AnalysisRequest(
        name="a", model="m", load_case="lc" if history else None, settings=settings
    )
    return [x.code for x in preflight_request(cable, request, series)]


@pytest.fixture(scope="module")
def menard():
    return cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)


@pytest.fixture(scope="module")
def stranded():
    return cx.iea_task49_66kv_stranded()


def test_report_text_and_severities():
    empty = PreflightReport(())
    assert empty.ok and empty.warnings == () and empty.errors == ()
    assert empty.to_text() == "preflight: no findings"
    report = PreflightReport(
        (
            PreflightIssue("warning", "a", "w-code", "look"),
            PreflightIssue("error", "b", "e-code", "stop"),
        )
    )
    assert not report.ok
    assert [x.code for x in report.warnings] == ["w-code"]
    assert [x.code for x in report.errors] == ["e-code"]
    assert report.to_text().splitlines() == [
        "WARNING a: [w-code] look",
        "ERROR   b: [e-code] stop",
    ]


def test_fatigue_curves_blocked_unknown_and_missing(menard):
    bad = _layer(
        _layer(menard, "inner_armour", bulk_sn="perier_2009_steel"),
        "outer_armour",
        bulk_sn="no_such_curve",
    )
    assert _codes(bad, FatigueSettings(), _series()) == [
        "blocked-sn-curve",
        "unknown-sn-curve",
    ]
    # A non-structural layer is not assessed; a structural one without a curve warns.
    none = _layer(
        _layer(menard, "inner_armour", bulk_sn=""),
        "outer_armour",
        structural=False,
        bulk_sn="no_such_curve",
    )
    assert _codes(none, FatigueSettings(), _series()) == ["no-sn-curve"]
    assert _codes(menard, FatigueSettings(), _series()) == []


def test_conductor_fatigue_needs_a_resolved_conductor_and_its_curves(menard, stranded):
    assert _codes(menard, ConductorFatigueSettings(), _series()) == [
        "no-resolved-conductor"
    ]
    assert _codes(stranded, ConductorFatigueSettings(), _series()) == []
    bare = _layer(stranded, "cu_conductor", bulk_sn="")
    assert _codes(bare, ConductorFatigueSettings(), _series()) == ["no-sn-curve"]
    assert _codes(
        bare, ConductorFatigueSettings(sn_curve="no_such_curve"), _series()
    ) == ["unknown-sn-curve"]
    surface = ConductorFatigueSettings(
        trellis_fretting=True, trellis_model="surface_swt"
    )
    assert _codes(stranded, surface, _series()) == [
        "no-fretting-law",
        "unvalidated-model",
    ]
    with_law = _layer(stranded, "cu_conductor", fretting_sn="copper_poon2022")
    assert _codes(with_law, surface, _series()) == ["unvalidated-model"]
    critical = ConductorFatigueSettings(trellis_fretting=True)
    assert _codes(stranded, critical, _series()) == []


def test_bending_law_tension_grid(menard):
    single = BendingLawSettings(tensions_N=(0.0,), kappa_max_1_m=0.1)
    assert _codes(menard, single) == ["planar-law"]
    falling = BendingLawSettings(tensions_N=(20e3, 0.0), kappa_max_1_m=0.1)
    assert _codes(menard, falling) == ["tension-order", "planar-law"]


def test_load_case_checks(menard):
    assert _codes(menard, FatigueSettings()) == ["missing-load-case"]
    assert _codes(menard, FatigueSettings(), _series(n=1)) == ["short-history"]
    assert _codes(menard, BendingSettings(), _series()) == []
    assert _codes(menard, BendingSettings(), _series(planar=False)) == ["not-planar"]
    assert _codes(menard, BendingSettings(), _series(kind="effective")) == [
        "tension-kind"
    ]
    varying = _series()
    varying = LoadCaseSeries(
        time_s=varying.time_s,
        curvature_x_1_m=varying.curvature_x_1_m,
        curvature_y_1_m=varying.curvature_y_1_m,
        tension_N=np.linspace(20e3, 40e3, varying.n_steps),
        tension_kind="true",
    )
    assert _codes(menard, BendingSettings(), varying) == ["not-planar"]


def test_project_preflight_resolves_models_and_load_cases(menard):
    project = Project("preflight")
    project.set_model("m", CableModel(menard))
    project.set_load_case("hour", _series())
    project.set_analysis(
        AnalysisRequest(
            name="fat", model="m", load_case="hour", settings=FatigueSettings()
        )
    )
    project.set_analysis(
        AnalysisRequest(
            name="law",
            model="m",
            settings=BendingLawSettings(tensions_N=(20e3, 0.0), kappa_max_1_m=0.1),
        )
    )
    report = preflight(project)
    assert [(x.analysis, x.code) for x in report.issues] == [
        ("law", "tension-order"),
        ("law", "planar-law"),
    ]
    assert not report.ok


def test_incomplete_cable_stops_the_other_checks():
    assert _codes(incomplete_cable(), FatigueSettings()) == ["incomplete-cable"]
