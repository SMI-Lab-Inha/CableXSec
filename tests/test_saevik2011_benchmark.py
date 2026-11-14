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

import check_saevik2011 as sv
import pytest
import run_validation

import cablexsec as cx

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


def test_registered_targets_and_results_are_pinned(rows):
    plane = rows["SV11-PLANE"]
    assert plane.target == (
        "median R in [0.85, 1.18] and >= 12 of 14 R in [0.70, 1.43] "
        "(R = mean predicted / mean measured, bending-plane stations)"
    )
    assert plane.value == (
        "median R 1.65 (0.45-3.65), 3 of 14 in band; sectional median 1.03 "
        "(0.45-2.28), 11 of 14"
    )
    assert plane.status == "FAIL"
    gj = rows["SV11-GJ"]
    assert gj.target == "mean ratio in [0.72, 1.38]"
    assert gj.value == "1.90 (sectional 1.32); static stress 271 MPa"
    assert gj.status == "FAIL"
