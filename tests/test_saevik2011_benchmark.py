# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Sævik (2011) 8-inch FBG riser: packaged definition, digitised data, gates.

The riser of Sævik (2011, Table 1) is mapped to CableXSec layers in
``saevik2011.yaml``; the measured inner-armour stress ranges (Figs 17-23) and
the absolute LC 14 loops of Gjøsteen et al. (2027, Figs 27-35) are digitised
in ``validation/paper_data``. The SV11 targets were registered before any model
output (doc/benchmarks.md); the pins below catch a later relaxation of the
targets or a silent change of the result (VALIDATION_POLICY Rules 2 and 9).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import replace

import check_saevik2011 as sv
import numpy as np
import pytest
import run_validation

import cablexsec as cx
from cablexsec._numba import HAVE_NUMBA

#: Long friction histories run pure Python without numba (minutes per case).
needs_numba = pytest.mark.skipif(
    not HAVE_NUMBA, reason="long friction histories need the compiled kernels"
)

pytestmark = pytest.mark.full_scale

# Sævik (2011) Table 3, calculated/measured maximum stress range, MM column.
TABLE3_MM = {
    4: 0.98,
    5: 1.02,
    6: 1.01,
    7: 0.76,
    8: 0.94,
    9: 1.09,
    12: 0.98,
    13: 0.94,
    14: 1.05,
    15: 1.05,
    16: 0.98,
    17: 0.94,
    18: 0.99,
    19: 1.04,
}


@pytest.fixture(scope="module")
def rows():
    return {r.ident: r for r in run_validation._saevik_rows()}


def test_packaged_riser_reproduces_table_1():
    cable = cx.load_cable(cx.saevik2011_yaml_path(), geometry_override=None)
    layers = {layer.name: layer for layer in cable.layers}
    inner, outer = layers["inner_tensile_armour"], layers["outer_tensile_armour"]
    assert (inner.n_wires, outer.n_wires) == (55, 57)
    for armour in (inner, outer):
        area = math.pi * armour.d_wire_m**2 / 4
        assert area == pytest.approx(5e-3 * 12.5e-3, rel=1e-4)
    pitch = 2 * math.pi * inner.r_helix_m / math.tan(inner.lay_angle_rad)
    assert pitch == pytest.approx(1.795, abs=2e-3)
    assert 2 * inner.r_helix_m == pytest.approx(0.264)
    sheath = layers["pressure_sheath"]
    assert 2 * sheath.r_inner_m == pytest.approx(0.2152)
    assert sheath.r_outer_m - sheath.r_inner_m == pytest.approx(0.0087)
    assert 2 * layers["outer_sheath"].r_outer_m == pytest.approx(0.300)
    pa = layers["pressure_armour"]
    area_per_length = (
        math.pi
        * pa.d_wire_m**2
        / 4
        * math.tan(pa.lay_angle_rad)
        / (2 * math.pi * pa.r_helix_m)
    )
    assert area_per_length == pytest.approx(10.5e-3, rel=5e-3)
    assert not pa.structural


def test_polymer_modulus_reproduces_the_published_bending_stiffness():
    cable = cx.load_cable(cx.saevik2011_yaml_path(), geometry_override=None)
    assert cx.compute_stiffness(cable).ei_min_Nm2 == pytest.approx(72.8e3, rel=1e-3)


def test_digitised_figures_reproduce_table_3_and_are_complete():
    rows = sv.measured_saevik()
    assert len(rows) == 14 * 20
    by_case: dict[int, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_case[int(r["load_case"])].append(r)
    agree = 0
    for lc, case_rows in by_case.items():
        meas = max(float(r["measured"]) for r in case_rows)
        model = max(float(r["mm_model"]) for r in case_rows)
        agree += abs(model / meas - TABLE3_MM[lc]) <= 0.05
    # MM maximum over measured maximum within 0.05 of the printed ratio for
    # 12 of 14 cases (LC 5 and LC 18 differ; doc/benchmarks.md).
    assert agree >= 12
    gj = sv.measured_gjosteen()
    assert set(gj) == {"2.1", "2.2", "2.3", "2.5", "3.1", "3.2", "3.3", "3.4", "3.5"}
    assert gj["2.1"] == pytest.approx(186.2, abs=0.1)


def test_load_case_table_and_rig_constants():
    cases = {c.lc: c for c in sv.load_cases()}
    assert len(cases) == 14
    assert cases[4].pressure_Pa == pytest.approx(17e6)  # Table 3 / Fig. 19(b)
    assert cases[19].pressure_Pa == pytest.approx(17e6)
    assert cases[16].tension_N == pytest.approx(750e3)
    assert pytest.approx(-2.305) == sv.X_ANCHOR_M
    assert len(sv.plane_stations()) == 8


@needs_numba
def test_registered_targets_and_results_are_pinned(rows):
    plane = rows["SV11-PLANE"]
    assert plane.target == (
        "median R in [0.85, 1.18] and >= 12 of 14 R in [0.70, 1.43] "
        "(R = mean predicted / mean measured, bending-plane stations)"
    )
    # Signed capstan bound; the former symmetric cap gave median R 1.65
    # (0.45-3.65) and sectional median 1.03 (0.45-2.28). The stick load
    # cases are unchanged, the slipping ones fall by 3-5 %.
    assert plane.value == (
        "median R 1.62 (0.45-3.59), 3 of 14 in band; sectional median 1.00 "
        "(0.45-2.17), 11 of 14"
    )
    assert plane.status == "FAIL"
    gj = rows["SV11-GJ"]
    assert gj.target == "mean ratio in [0.72, 1.38]"
    # Former symmetric cap: 1.90 (sectional 1.32).
    assert gj.value == "1.80 (sectional 1.24); static stress 271 MPa"
    assert gj.status == "FAIL"


def test_rig_defaults_are_the_registered_rig():
    rig = sv.sv11_rig()
    assert (rig.x_anchor_m, rig.free_length_m, rig.rigid_right_m) == (
        sv.X_ANCHOR_M,
        sv.FREE_LENGTH_M,
        sv.RIGID_RIGHT_M,
    )
    assert (rig.gap_m, rig.bell_end_m, rig.kappa_b0, rig.kappa_b1) == (
        sv.GAP_M,
        sv.BELL_END_M,
        sv.KAPPA_B0,
        sv.KAPPA_B1,
    )
    assert rig.inlet_bounded and rig.x_rotation_m == 0.0
    moment, _ei_max, ei_min = sv.backbone(sv.load_cases()[0], sv.MU_DECLARED)
    default = sv.Beam(moment, ei_min, 500e3)
    explicit = sv.Beam(moment, ei_min, 500e3, rig=rig)
    assert (default.lower == explicit.lower).all()
    assert (default.upper == explicit.upper).all()
    a = default.solve(math.radians(2.0))[1]
    b = explicit.solve(math.radians(2.0))[1]
    assert (a == b).all()


def test_rotation_point_only_changes_the_lever_arm():
    # Turning the rocker about the clamp instead of x = 0 moves the far pin
    # by the lever-arm ratio, nothing else.
    moment, _ei_max, ei_min = sv.backbone(sv.load_cases()[0], sv.MU_DECLARED)
    rig = sv.sv11_rig()
    turned = sv.Beam(
        moment, ei_min, 500e3, rig=replace(rig, x_rotation_m=rig.x_anchor_m)
    )
    w = turned.solve(math.radians(1.0))[0]
    lever = turned.x[-1] - rig.x_anchor_m
    assert w[-1] == pytest.approx(-lever * math.sin(math.radians(1.0)), rel=1e-15)


def test_eq88_consistency_favours_the_1p72_m_end():
    eq = sv.eq88_consistency()
    # Saevik (2011): 3.9 m at s = 1.72 m and a 16 deg slope; Gjosteen et al.
    # (2027) the same radius at 1.79 m.
    assert eq["1.72"]["radius_m"] == pytest.approx(3.90, abs=0.005)
    assert eq["1.72"]["slope_deg"] == pytest.approx(15.9, abs=0.05)
    assert eq["1.79"]["radius_m"] == pytest.approx(3.79, abs=0.005)
    assert eq["1.79"]["slope_deg"] == pytest.approx(17.0, abs=0.05)
    assert eq["radius_3.9"]["s_m"] == pytest.approx(1.722, abs=5e-4)


@needs_numba
def test_split_rows_are_pinned():
    rows = {r.ident: r for r in run_validation._saevik_split_rows()}
    split = rows["SV11-SPLIT"]
    assert split.target == "stated, not gated"
    assert split.value == (
        "(a) median R 1.01 (11 of 14 in band), GJ 1.27; (b) LC 14 R 1.70, GJ "
        "1.80, largest change against the registered correction below 1e-12; "
        "registered 1.62 and 1.80, sectional 1.00 and 1.24"
    )
    reach = rows["SV11-REACH"]
    assert reach.value == (
        "a_max 0.26 and 0.36; R corrected 0.45 and 0.54 (sectional 0.45 and "
        "0.54); sectional outside the band: LC 4 2.17, LC 7 0.45, LC 8 0.54"
    )
    eq = rows["SV11-EQ88"]
    assert eq.value == (
        "1.72 m: 3.90 m, 15.9 deg; 1.79 m: 3.79 m, 17.0 deg; 3.9 m at s = 1.722 m"
    )
    assert {r.status for r in rows.values()} == {"INFO"}


def test_free_inlet_needs_a_node_at_the_entrance():
    moment, _ei_max, ei_min = sv.backbone(sv.load_cases()[0], sv.MU_DECLARED)
    rig = replace(sv.sv11_rig(), inlet_bounded=False)  # x_anchor -2.305 m
    with pytest.raises(ValueError, match="no node at the bellmouth entrance"):
        sv.Beam(moment, ei_min, 500e3, rig=rig)
    on_grid = replace(rig, x_anchor_m=-2.3)
    beam = sv.Beam(moment, ei_min, 500e3, rig=on_grid)
    inlet = beam.x < -1e-9
    assert np.isinf(beam.upper[inlet]).all()


def test_rig_anchor_sets_the_termination_coordinates():
    # A rig with another end-fitting position moves the profile coordinates
    # and the station distances together (the anchor is at the rig's face).
    case = next(c for c in sv.load_cases() if c.lc == 14)
    rig = replace(sv.sv11_rig(), x_anchor_m=-3.0, free_length_m=12.875)
    res = sv.predict_case(case, sv.MU_DECLARED, only=("2.1",), rig=rig)
    term = res["stations"]["2.1"]["termination"]
    assert term["termination_applied"] == 1.0
    assert term["axial_p"] > 0.0


def test_setting_changes_clear_the_split_cache(monkeypatch):
    # split_values is cached by mu only: changing the tendon definition or the
    # anchor stiffness must drop it together with gate_values.
    cleared = []

    class Spy:
        def __init__(self, name):
            self.name = name

        def cache_clear(self):
            cleared.append(self.name)

    monkeypatch.setattr(sv, "gate_values", Spy("gate"))
    monkeypatch.setattr(sv, "split_values", Spy("split"))
    monkeypatch.setattr(sv, "cable", Spy("cable"))
    monkeypatch.setattr(sv, "TENDONS", sv.TENDONS)
    monkeypatch.setattr(sv, "K_ANCHOR_N_M", sv.K_ANCHOR_N_M)
    sv.use_anchor(None)
    assert cleared == ["gate", "split"]
    cleared.clear()
    sv.use_tendons("round")
    assert cleared == ["cable", "gate", "split"]
