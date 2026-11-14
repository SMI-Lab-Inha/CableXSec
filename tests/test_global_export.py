# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Files of section properties for global analysis: content, units and headers."""

from __future__ import annotations

import csv
import io
import json
import math
import warnings

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.section_properties import SectionLoadCase
from cablexsec.exceptions import CableInputError
from cablexsec.io.global_export import (
    bending_tables_csv,
    cabledyn_bending_law,
    section_properties_csv,
)

FINGERPRINT = "0123abcd"


@pytest.fixture(scope="module")
def result() -> cx.SectionPropertiesResult:
    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    cases = [
        SectionLoadCase("Low tension", 10e3),
        SectionLoadCase("Design tension", 50e3),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cx.compute_section_properties(cable, cases, n_points=40)


def _comments(text: str) -> list[str]:
    return [line[2:] for line in text.splitlines() if line.startswith("# ")]


def _data(text: str) -> list[list[str]]:
    rows = [line for line in text.splitlines() if not line.startswith("#")]
    return list(csv.reader(io.StringIO("\n".join(rows))))


def test_headers_state_the_cable_fingerprint_cases_and_units(
    result: cx.SectionPropertiesResult,
) -> None:
    for text in (
        section_properties_csv(result, FINGERPRINT),
        bending_tables_csv(result, FINGERPRINT),
    ):
        header = _comments(text)
        joined = "\n".join(header)
        assert f"Cable: {result.cable_name}" in header
        assert f"Fingerprint (cable and settings): {FINGERPRINT}" in header
        assert any(line.startswith("Units: ") for line in header)
        assert "Load case Low tension: 10 kN, twist held" in joined
        assert "solved radial section" in joined
        assert "_" not in joined and "—" not in joined


def test_neutral_table_round_trip(result: cx.SectionPropertiesResult) -> None:
    rows = _data(section_properties_csv(result, FINGERPRINT))
    header, body = rows[0], rows[1:]
    assert len(body) == len(result.cases)
    for case, row in zip(result.cases, body, strict=True):
        values = dict(zip(header, row, strict=True))
        assert values["load case"] == case.case.name
        assert values["axial and torsional values"] == "solved section"
        assert float(values["axial stiffness tangent [N]"]) == pytest.approx(
            case.axial_stiffness_N, rel=1e-8
        )
        assert float(
            values["torsional stiffness positive twist [N m2]"]
        ) == pytest.approx(case.torsional_stiffness_positive_Nm2, rel=1e-8)
        assert values["twist from the tension twist free [rad/m]"] == ""
        for slip in case.armour:
            layer = slip.layer.replace("_", " ")
            assert float(values[f"{layer} full slip curvature [1/m]"]) == pytest.approx(
                slip.full_slip_curvature_1_m, rel=1e-8
            )


def test_backbone_table_round_trip(result: cx.SectionPropertiesResult) -> None:
    rows = _data(bending_tables_csv(result, FINGERPRINT))
    data = np.array(rows[1:], dtype=float)
    np.testing.assert_allclose(data[:, 0], result.cases[0].kappa_1_m, rtol=1e-8)
    for index, case in enumerate(result.cases, start=1):
        np.testing.assert_allclose(data[:, index], case.moment_Nm, rtol=1e-8)


def test_cabledyn_law_round_trip(result: cx.SectionPropertiesResult) -> None:
    law = cabledyn_bending_law(result, FINGERPRINT)
    record = json.loads(law.json_text())
    assert record["format"] == "cablexsec.bending_law/1"
    assert record["tension_N"] == [10e3, 50e3]
    assert record["settings"]["fingerprint"] == FINGERPRINT
    assert any("Load case" in line for line in record["settings"]["header"])
    for case in result.cases:
        backbone = law.law_at(case.case.tension_N).backbone(case.kappa_1_m)
        np.testing.assert_allclose(backbone, case.moment_Nm, rtol=1e-9, atol=1e-9)
    mixed = result.cases[0].case.__class__("Free", 30e3, "free")
    from dataclasses import replace

    other = replace(result.cases[1], case=mixed)
    with pytest.raises(CableInputError, match="tension only"):
        cabledyn_bending_law(replace(result, cases=(result.cases[0], other)))
    same = replace(result.cases[1], case=replace(result.cases[1].case, tension_N=10e3))
    with pytest.raises(CableInputError, match="different tension"):
        cabledyn_bending_law(replace(result, cases=(result.cases[0], same)))
    assert math.isfinite(record["ei_slip_Nm2"][0])
