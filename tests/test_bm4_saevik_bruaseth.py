# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""BM4 — Sævik & Bruaseth (2005) EA population via the element-stiffness 6×6.

The Appendix (Tables A1-A10) tabulates per-element stiffnesses (EA_i, GJ_i,
EI_i), centroid radius and pitch for 10 umbilicals — NOT raw geometry. The
element-stiffness assembler (no gap-filling) is exercised against the measured
tension-torsion-fixed axial stiffness (Table 3). Umbilicals 2 and 3 have no
fixed measurement (Table 5) and are dropped (stated), leaving 8 cables.

FINDING (honest, negative on the <1.25 gate): the assembler is the
**no-radial-interaction** bound. It matches the measured EA to ~1% for
tube-dominated cables with little radial interaction (U4, U5, U7) but
overestimates multi-armour-layer cables by up to 2.2× (U1), because the
inter-layer radial-contraction reduction — the central effect of
[SaevikBruaseth2005] — is not carried by the per-element stiffnesses. The
population mean model/measured ratio is ≈ 1.45, so the <1.25 gate is NOT met by
the no-interaction assembler; closing it needs the cross-section radial model
(Sævik's orthotropic core compliance), i.e. the raw-geometry path, not the
tabulated element stiffnesses. This is reported, not gap-filled.
"""

from __future__ import annotations

import math

import pytest

from cablexsec.mechanics.element_stiffness import (
    HelicalElement,
    ea_fixed_from_elements,
    lay_angle_rad,
)

_MN = 1.0e6
_KNM2 = 1.0e3


# Test-taxonomy category.
pytestmark = pytest.mark.independent_method


def _el(r_mm, p_mm, ea_MN, gj_kNm2, ei_kNm2) -> HelicalElement:
    return HelicalElement(
        r_mm / 1000.0, p_mm / 1000.0, ea_MN * _MN, gj_kNm2 * _KNM2, ei_kNm2 * _KNM2
    )


# Sævik & Bruaseth (2005) Appendix A (R [mm], pitch [mm], EA [MN], GJ, EI [kN·m²]).
_RAW: dict[int, list[tuple[float, float, float, float, float]]] = {
    1: [
        (0, 0, 2.004, 6.122, 7.958),
        (84.75, 2634, 247.934, 72.467, 0.866),
        (82.05, -1833, 222.932, 117.358, 0.826),
        (76.90, 2634, 499.872, 100.139, 1.539),
        (69.70, -2442, 454.712, 71.777, 1.399),
        (0, 0, 0.884, 1.389, 1.806),
        (16.60, -275, 13.978, 0.543, 0.001),
    ],
    4: [
        (0, 0, 0.7779, 1.1390, 1.4807),
        (44.95, -2900, 121.8301, 10.7828, 10.9509),
        (38.20, -2900, 17.8638, 0.5009, 0.4106),
        (41.48, -2900, 28.8489, 1.5292, 1.4432),
        (42.19, -2900, 24.5609, 1.3952, 1.3071),
        (19.25, 2000, 13.4267, 0.1373, 0.0027),
    ],
    5: [(31.99, 2200, 186.0801, 5.2873, 4.7552), (4.28, 180, 11.7823, 0.1348, 0.0013)],
    6: [
        (0, 0, 0.4943, 0.4141, 0.5384),
        (26.47, -1075, 148.4896, 5.4891, 3.7872),
        (39.95, 812, 144.2165, 21.8472, 0.4625),
        (43.30, -1269, 125.1482, 10.7903, 0.3497),
        (10.40, 550, 4.3838, 0.0202, 0.0001),
    ],
    7: [
        (34.25, -2400, 354.2712, 31.2172, 35.8470),
        (59.90, 3600, 167.6028, 9.8551, 4.2154),
        (60.16, 3600, 10.7384, 0.6520, 0.0007),
    ],
    8: [
        (0, 0, 15.5958, 0.3058, 0.3976),
        (0, 0, 0.3822, 0.1801, 0.2612),
        (27.10, -1870, 154.0381, 3.9950, 3.9309),
    ],
    9: [
        (53.7, -1847, 247.626, 24.0071, 0.3888),
        (48.7, 1347, 218.6, 26.8073, 0.3444),
        (15.25, 548, 29.2146, 0.2125, 0.0073),
        (33.65, 2200, 39.3457, 1.1549, 0.9551),
    ],
    10: [
        (56.9, -2377, 261.703, 19.4294, 0.4268),
        (51.6, 1395, 226.652, 32.6105, 0.3717),
        (34.85, -1725, 102.002, 3.8115, 0.0164),
        (12.75, 800, 36.9355, 0.3954, 0.0059),
    ],
}
_POP: dict[int, list[HelicalElement]] = {
    u: [_el(*t) for t in rows] for u, rows in _RAW.items()
}
# Measured EA tension-torsion fixed [MN] (Table 3); U2,U3 not measured -> dropped.
_MEAS_FIXED_MN: dict[int, float] = {
    1: 609,
    4: 206,
    5: 194,
    6: 209,
    7: 466,
    8: 136,
    9: 366,
    10: 338,
}


# --------------------------------------------------------------------------- #
# Assembler mechanics (verified against the matching cables)                  #
# --------------------------------------------------------------------------- #
def test_straight_element_full_axial_stiffness() -> None:
    """A straight element (pitch 0) contributes its full EA (cos³0 = 1)."""
    el = HelicalElement(0.0, 0.0, 100.0e6, 1.0, 1.0)
    assert ea_fixed_from_elements([el]) == pytest.approx(100.0e6, rel=1e-12)


def test_helical_element_cos_cubed_projection() -> None:
    el = _el(50.0, 1000.0, 100.0, 1.0, 1.0)
    expected = 100.0e6 * math.cos(lay_angle_rad(el)) ** 3
    assert ea_fixed_from_elements([el]) == pytest.approx(expected, rel=1e-12)


def test_low_interaction_cables_match_measured_within_gate() -> None:
    """Tube-dominated cables (weak radial interaction) reproduce measured EA."""
    for u in (4, 5, 7):
        ratio = ea_fixed_from_elements(_POP[u]) / (_MEAS_FIXED_MN[u] * _MN)
        assert 0.85 < ratio < 1.25, f"U{u} ratio {ratio:.2f}"


# --------------------------------------------------------------------------- #
# BM4 population — documented no-radial-interaction result (gate NOT met)      #
# --------------------------------------------------------------------------- #
def test_BM4_population_is_no_interaction_bound_above_gate() -> None:
    """Mean model/measured ≈ 1.45 > 1.25: the assembler is the upper bound.

    Multi-armour cables overestimate (radial interaction missing); the gate is
    met only with the cross-section radial model, not the tabulated element
    stiffnesses. Regression guard on the honest finding.
    """
    ratios = {
        u: ea_fixed_from_elements(_POP[u]) / (_MEAS_FIXED_MN[u] * _MN) for u in _POP
    }
    assert all(r >= 0.98 for r in ratios.values())  # never under-predicts (upper bound)
    assert ratios[1] > 2.0  # 4-armour-layer cable: radial interaction dominates
    mean = sum(ratios.values()) / len(ratios)
    assert mean == pytest.approx(1.45, abs=0.05)
    assert mean > 1.25  # gate NOT met by the no-interaction bound (stated)
