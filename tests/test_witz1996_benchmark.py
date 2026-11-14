# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Witz (1996) 2.5-inch flexible pipe: packaged definition and registered gates.

WITZ-EI is parameter-free. WITZ-ID identifies the initial external pressure
on the 0-bar loop width with mu = 0.10 declared and predicts the 300-bar
loop. Its registered targets fail; the pins below catch any later
relaxation of the targets (VALIDATION_POLICY Rules 6 and 9).
"""

from __future__ import annotations

import warnings

import pytest
import run_validation

import cablexsec as cx
from cablexsec.cables.schema import InitialStateProperties

pytestmark = pytest.mark.full_scale


@pytest.fixture(scope="module")
def rows():
    return {r.ident: r for r in run_validation._witz_rows()}


def test_packaged_pipe_declares_an_identified_initial_state():
    cable = cx.load_cable(cx.witz1996_yaml_path(), geometry_override=None)
    state = cx.clamping_state(cable)
    assert state.initial_identification == "identified"
    assert state.is_declared
    with warnings.catch_warnings():
        warnings.simplefilter("error", cx.MissingClampingStateWarning)
        cx.compute_mkappa(cable, [0.0, 0.1, 0.0], mu=0.1)


def test_identified_class_needs_a_source_and_a_value():
    InitialStateProperties(
        external_pressure_Pa=1e6, identification="identified", source="fit to X"
    )
    with pytest.raises(ValueError):
        InitialStateProperties(external_pressure_Pa=1e6, identification="identified")
    with pytest.raises(ValueError):
        InitialStateProperties(identification="identified", source="fit to X")


def test_full_slip_stiffness_gate(rows):
    row = rows["WITZ-EI"]
    assert row.target == "within [913, 1309] N·m² (gradients ±10%)"
    assert row.value.startswith("1007 N·m²")
    assert row.status == "PASS"


def test_identify_on_0_bar_predict_300_bar_stays_a_registered_failure(rows):
    row = rows["WITZ-ID"]
    # Registered targets (doc/benchmarks.md), not to be relaxed.
    assert row.target == (
        "width within [1692, 2489] N·m and energy within 15% of 3125 J/m"
    )
    # The live identification reproduces the declared (identified) value.
    assert "p0 1.037 MPa (YAML 1.037)" in row.value
    assert "300-bar width 1383 vs [1990, 2164] N·m (-30.5% / -36.1%)" in row.value
    assert "energy 2732 J/m (-12.6%)" in row.value
    assert row.status == "FAIL"
