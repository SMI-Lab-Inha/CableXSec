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
    assert table[0] == (0.0, 0.0) and len(table) == export.kappa_1_m.size
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
    # The 0.01 1/m step is wider than the stick-to-slip transition at 20 kN:
    # the export adds a stick point and the slip points and warns.
    with pytest.warns(UserWarning, match="linear step"):
        export = export_bending_law(
            cable, tensions_N=[2e4], kappa_max_1_m=0.1, n_points=11, spacing="linear"
        )
    assert np.isin(np.linspace(0.0, 0.1, 11), export.kappa_1_m).all()
    extra = np.setdiff1d(export.kappa_1_m, np.linspace(0.0, 0.1, 11))
    assert extra.size >= 3 and np.all(extra < 0.01)
    stick = cx.compute_mkappa(cable, np.zeros(2), tension_N=2e4).EI_max_Nm2
    assert export.laws[0].ei_stick_Nm2 == pytest.approx(stick, rel=1e-12)
    assert export.law_at(2e4) is export.laws[0]


@pytest.fixture(scope="module")
def iea66_stick():
    """66 kV template at 10-200 kN: its slip onsets at 10 kN (7.3e-6 and
    2.3e-5 1/m) lie below the base grid floor 1e-4 kappa_max = 4e-5 1/m."""
    cable = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
    tensions = [10e3, 25e3, 50e3, 100e3, 200e3]
    props = cx.compute_section_properties(
        cable, [cx.SectionLoadCase(f"T{t:g}", tension_N=t) for t in tensions]
    )
    export = export_bending_law(
        cable,
        tensions_N=tensions,
        kappa_max_1_m=0.4,
        n_points=200,
        slip_scale=True,
    )
    return cable, tensions, props, export


@pytest.mark.filterwarnings("error:the exported law:UserWarning")
def test_export_starts_at_the_section_stick_stiffness(iea66_stick):
    cable, tensions, props, export = iea66_stick
    table = export.slip_scale
    assert table is not None and table.reference_tension_N == tensions[0]
    onsets = [a.onset_curvature_1_m for c in props.cases for a in c.armour]
    assert min(onsets) < 0.4 * 1e-4  # below the base grid floor
    assert export.kappa_1_m[1] < 0.5 * min(onsets)
    for i, case in enumerate(props.cases):
        stick = case.ei_stick_Nm2
        assert export.laws[i].ei_stick_Nm2 == pytest.approx(stick, rel=1e-12)
        assert table.law_at(tensions[i]).ei_stick_Nm2 == pytest.approx(stick, rel=1e-12)
        for slip in case.armour:
            for value in (slip.onset_curvature_1_m, slip.full_slip_curvature_1_m):
                assert np.min(np.abs(export.kappa_1_m - value)) <= 1e-9 * value


def test_exported_and_scaled_laws_match_compute_mkappa_off_the_grid(iea66_stick):
    # Before the grid carried the slip curvatures the 10 kN law started at
    # 0.46 of the stick stiffness and every scaled tension inherited it.
    cable, tensions, _, export = iea66_stick
    table = export.slip_scale
    assert table is not None
    grid = export.kappa_1_m[1:]
    between = np.sqrt(grid[:-1] * grid[1:])
    between = between[(between > 1e-6) & (between < 0.05)]
    for i, tension in enumerate(tensions):
        direct = np.asarray(
            cx.compute_mkappa(
                cable, np.concatenate([[0.0], between]), tension_N=tension
            ).moment_Nm
        )[1:]
        for law in (export.laws[i], table.law_at(tension)):
            error = np.max(np.abs(law.backbone(between) - direct) / direct)
            assert error < 2e-3, (tension, error)
