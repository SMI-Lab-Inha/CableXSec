# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""BM1, Ménard and Cartraud (2023) calibration-transfer benchmark.

The full geometry is specified by the source (Table 1 + Section 6.1 pitches), so
the cable is built from cited values with no inference. The full-slip bending
stiffness EI_min is **independent of the (unknown, paper-calibrated) initial
contact state** — the paper itself reports it "almost insensitive to the
friction coefficient and pressure" — so it is validated here against the
analytical value (3593 N·m²) and the measured slip-zone stiffness range, with no
calibration.

The source prescribes ``mu=0.2`` and calibrates the applied external pressure
to 0.3 MPa on the [-0.05, 0.05] 1/m cycle, then holds both values fixed for
five other cycles. The same split is retained here. It is a published
calibration transferred to a held-out loading range, not a parameter-free
prediction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cablexsec import compute_mkappa
from cablexsec.analysis.curve_comparison import cycle_work, insert_cycle_crossings
from cablexsec.cables import load_cable, menard2023_yaml_path
from cablexsec.mechanics.stiffness import ei_min_slip

# Measured slip-zone (minimum) bending stiffness across the six ranges,
# [MenardCartraud2023] Table 3 (loading & unloading), N·m².
_TEST_EI_MIN = [3955, 3854, 3529, 3410, 3517, 3990, 3943, 3615, 3296, 3554, 3677, 3083]
_ANALYTICAL_EI_MIN = 3593.0  # [MenardCartraud2023] Table 3 (Witz-Tan full slip)


# Test-taxonomy category.
pytestmark = pytest.mark.full_scale

MEASURED_B = (
    Path(__file__).resolve().parents[1]
    / "validation"
    / "paper_data"
    / "menard_fig16b_ordered.csv"
)


def _cable():
    return load_cable(menard2023_yaml_path(), geometry_override=None)


def test_geometry_fully_specified_from_source() -> None:
    """The cable is fully specified from Table 1 — no null/inferred geometry."""
    cable = _cable()
    assert cable.is_fully_specified()
    assert cable.missing_fields() == []


def test_extracted_geometry_matches_table1() -> None:
    """Wire counts, lay angles and derived helix radii match the source."""
    cable = _cable()
    inner = cable.get_layer("inner_armour")
    outer = cable.get_layer("outer_armour")
    assert (inner.n_wires, outer.n_wires) == (55, 63)  # Table 1
    assert inner.lay_angle_deg == pytest.approx(20.1)  # |−20.1| Table 1
    assert outer.lay_angle_deg == pytest.approx(19.6)  # Table 1
    # Contacting surfaces from Table 1. Rounded pitches from Section 6.1
    # must not override the radii and cause millimetres of initial overlap.
    assert inner.r_helix_m == pytest.approx(0.0374 + 0.004 / 2)
    assert outer.r_helix_m == pytest.approx(0.042 + 0.004 / 2)


def test_BM1_ei_min_within_15pct_of_analytical_no_calibration() -> None:
    """BM1 gate: EI_min within 15% of the analytical 3593 N·m² (calibration-free)."""
    ei = ei_min_slip(_cable())
    rel = abs(ei - _ANALYTICAL_EI_MIN) / _ANALYTICAL_EI_MIN
    assert rel < 0.15, f"EI_min {ei:.0f} N.m2 deviates {rel * 100:.1f}% (> 15%)"
    # Retain a tighter analytical comparison alongside the measured-range check.
    assert rel < 0.05


def test_BM1_ei_min_inside_measured_slip_range() -> None:
    """EI_min falls inside the measured slip-zone stiffness band (Table 3)."""
    ei = ei_min_slip(_cable())
    assert min(_TEST_EI_MIN) <= ei <= max(_TEST_EI_MIN)


def test_BM1_initial_state_is_explicitly_labelled_published_calibration() -> None:
    cable = _cable()
    state = cable.initial_state
    assert state.external_pressure_Pa == pytest.approx(0.3e6)
    assert state.identification == "published_calibration"
    assert "Sections 5.1-5.3" in state.source


def test_BM1_held_out_pm01_loop_matches_digitised_test_branches() -> None:
    """Ordered held-out +/-0.1 moment comparison, without selecting by residual."""
    cable = _cable()
    measured = np.genfromtxt(
        MEASURED_B,
        delimiter=",",
        names=True,
    )
    curvature = measured["curvature_1_per_m"]
    result = compute_mkappa(cable, np.tile(curvature, 2), tension_N=0.0, mu=0.2)
    model_moment = result.moment_Nm[-len(curvature) :]
    measured_moment = measured["moment_Nm"]
    rmse = float(np.sqrt(np.mean((model_moment - measured_moment) ** 2)))
    normalized_rmse = rmse / float(np.ptp(measured_moment))
    assert normalized_rmse < 0.05


def test_BM1_matched_experimental_cycle_work_criterion() -> None:
    """Registered 10 % criterion on the matched cycle work (J/m, same paths).

    It failed with the former symmetric capstan cap (+12.9 %); with the
    signed bound of [Papailiou1997] the full-slip loop work fell by 3.7 %
    and the difference was +8.8 %. Since the armour shear stack stops at its
    support (xi 0.686 / 0.888 instead of 0.208 / 0.248) the stiffer stick
    raises the work to 69.7 J/m, +11.8 %, as registered before the run
    (doc/benchmarks.md, "Yun shear stack cut at the support"): the criterion
    fails. The threshold is unchanged; the test pins the documented
    difference, which lies wholly outside it.
    """
    data = np.genfromtxt(MEASURED_B, delimiter=",", names=True)
    curvature, moment = data["curvature_1_per_m"], data["moment_Nm"]
    augmented, _, crossings = insert_cycle_crossings(curvature, crossing=0.0)
    result = compute_mkappa(_cable(), np.tile(augmented, 2), tension_N=0.0, mu=0.2)
    mechanical = (result.work.stored_J_per_m + result.work.friction_J_per_m)[
        -len(augmented) :
    ]
    measured_work = float(np.mean(cycle_work(curvature, moment, crossing=0.0)))
    difference = float(np.mean(np.diff(mechanical[crossings]))) / measured_work - 1
    assert difference == pytest.approx(0.118, abs=1e-3)
    assert abs(difference) > 0.10  # the registered criterion is not met
