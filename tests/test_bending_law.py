# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Gap MX: Iwan bending law exported from the section law."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.bending_law import IwanBendingLaw, export_bending_law
from cablexsec.exceptions import CableInputError


@pytest.fixture(scope="module")
def menard():
    return cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)


def test_bilinear_backbone_gives_one_element_and_masing_loops():
    # Elastic-perfectly-plastic element (k=900, yield 0.01) plus EI_slip=100.
    kappa = np.array([0.0, 0.01, 0.05])
    moment = np.array([0.0, 10.0, 14.0])
    law = IwanBendingLaw.from_backbone(kappa, moment, 100.0)
    assert law.stiffness_Nm2.tolist() == pytest.approx([900.0, 0.0])
    assert law.ei_stick_Nm2 == pytest.approx(1000.0)
    history = np.array([0.0, 0.05, -0.05, 0.05])
    m = law.moment(np.interp(np.linspace(0, 3, 301), [0, 1, 2, 3], history))
    # Masing: the reversal from (0.05, 14) to -0.05 ends at -14.
    assert m[200] == pytest.approx(-14.0)
    assert m[-1] == pytest.approx(14.0)
    # Dissipation of one full cycle: 4 * k * k_y * (amplitude - k_y).
    cycle = np.interp(np.linspace(1, 3, 201), [1, 2, 3], [0.05, -0.05, 0.05])
    full = np.concatenate([np.linspace(0, 0.05, 50), cycle])
    w = law.dissipated_energy(full) - law.dissipated_energy(np.linspace(0, 0.05, 50))
    assert w == pytest.approx(4 * 900 * 0.01 * (0.05 - 0.01), rel=1e-9)


def test_rising_tangent_is_rejected():
    with pytest.raises(CableInputError, match="Masing"):
        IwanBendingLaw.from_backbone([0.0, 0.01, 0.02], [0.0, 1.0, 3.0], 0.0)


def test_export_reproduces_compute_mkappa(menard, tmp_path):
    export = export_bending_law(
        menard, tensions_N=[0.0, 20e3], kappa_max_1_m=0.12, n_points=120
    )
    t = np.linspace(0.0, 4 * math.pi, 801)
    kappa = 0.1 * np.sin(t) * np.exp(-0.1 * t)
    ref = np.asarray(cx.compute_mkappa(menard, kappa, tension_N=20e3).moment_Nm)
    mom = export.laws[1].moment(kappa)
    assert np.max(np.abs(mom - ref)) / np.ptp(ref) < 2e-3
    record = json.loads(
        export.to_json(tmp_path / "law.json").read_text(encoding="utf-8")
    )
    assert record["hysteresis_rule"] == "masing_iwan" and record["planar_only"]
    table = export.backbone_table_kNm(10e3)
    assert table[0] == (0.0, 0.0) and len(table) == 120
    with pytest.raises(CableInputError, match="outside"):
        export.law_at(30e3)
    assert export.to_csv(tmp_path / "law.csv").read_text().startswith("kappa_1_m,")


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((1e3, [1.0, 2.0], [0.1]), "equal 1-D arrays"),
        ((1e3, [1.0, np.nan], [0.1, 0.2]), "finite"),
        ((1e3, [-1.0, 2.0], [0.1, 0.2]), "ascending yield"),
        ((1e3, [1.0, 2.0], [0.2, 0.1]), "ascending yield"),
        ((-1.0, [1.0], [0.1]), "ei_slip_Nm2"),
    ],
)
def test_iwan_law_checks(args, match):
    with pytest.raises(CableInputError, match=match):
        IwanBendingLaw(*args)


@pytest.mark.parametrize(
    ("kappa", "moment", "ei", "match"),
    [
        ([0.0, 0.1], [0.0], 1e3, "equal"),
        ([0.0, np.inf], [0.0, 1.0], 1e3, "finite"),
        ([0.1, 0.2], [0.0, 1.0], 1e3, "start"),
        ([0.0, 0.1], [0.0, 1.0], np.nan, "ei_slip_Nm2"),
    ],
)
def test_backbone_checks(kappa, moment, ei, match):
    with pytest.raises(CableInputError, match=match):
        IwanBendingLaw.from_backbone(kappa, moment, ei)


def test_moment_history_check():
    law = IwanBendingLaw.from_backbone([0.0, 0.1, 0.2], [0.0, 100.0, 150.0], 250.0)
    with pytest.raises(CableInputError, match="finite 1-D"):
        law.moment(np.array([[0.0, 0.1]]))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tensions_N": [-1.0]}, "non-negative"),
        ({"tensions_N": [1e4, 0.0]}, "rise strictly"),
        ({"kappa_max_1_m": 0.0}, "kappa_max_1_m"),
        ({"n_points": 2}, "n_points"),
        ({"phi_bend": 0.3}, "set by the export"),
        ({"spacing": "log"}, "spacing"),
    ],
)
def test_export_checks(kwargs, match):
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    args = {"tensions_N": [0.0], "kappa_max_1_m": 0.1, "n_points": 20}
    args.update(kwargs)
    with pytest.raises(CableInputError, match=match):
        export_bending_law(cable, **args)


def test_single_tension_and_linear_spacing():
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    export = export_bending_law(
        cable, tensions_N=[2e4], kappa_max_1_m=0.1, n_points=11, spacing="linear"
    )
    np.testing.assert_allclose(np.diff(export.kappa_1_m), 0.01, rtol=1e-12)
    assert export.law_at(2e4) is export.laws[0]
