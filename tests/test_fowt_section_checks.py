# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Material-range checks use prescribed stresses, not a commercial solver."""

import math

import pytest
from fowt_section_checks import (
    assess_shell_bending,
    require_shell_bending_range,
)

import cablexsec as cx
from cablexsec.cables.schema import CylindricalShellLayer


@pytest.fixture
def export_cable():
    return cx.load_cable(cx.export_three_core_yaml_path(), geometry_override=None)


def test_template_sheath_stress_and_stiffness_match_hand_calculation(export_cable):
    report = assess_shell_bending(export_cable, 0.09573379141686437)
    shell = report["layers"][0]
    assert shell["nominal_bending_stress_Pa"] == pytest.approx(376951803.7039035)
    assert shell["root_bending_stress_Pa"] == pytest.approx(753903607.407807)
    assert shell["EI_Nm2"] == pytest.approx(35e9 * math.pi * 0.1125**3 * 0.003)
    assert shell["root_bending_utilization"] == pytest.approx(7.935827446397968)
    with pytest.raises(ValueError, match="elastic bending range"):
        require_shell_bending_range(report)


def test_small_curvature_does_not_claim_combined_loading_acceptance(export_cable):
    report = assess_shell_bending(export_cable, 0.001)
    require_shell_bending_range(report)
    assert report["status"] == "within_elastic_bending_range"
    assert "not combined loading" in report["scope"]


def test_root_scf_can_exceed_limit_with_nominal_stress_below_it(export_cable):
    report = assess_shell_bending(export_cable, 0.02)
    layer = report["layers"][0]
    assert layer["nominal_bending_stress_Pa"] < layer["yield_stress_Pa"]
    assert layer["root_bending_stress_Pa"] > layer["yield_stress_Pa"]
    with pytest.raises(ValueError):
        require_shell_bending_range(report)


@pytest.mark.parametrize("curvature", [-0.1, math.nan, math.inf])
def test_invalid_curvature_is_not_an_assessment(export_cable, curvature):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        assess_shell_bending(export_cable, curvature)


def test_missing_yield_limit_is_not_zero_utilisation(export_cable, monkeypatch):
    import fowt_section_checks as checks

    original = checks.library.get_property

    def missing_limit(material, property_name):
        if property_name == "sigma_y_Pa":
            raise KeyError(property_name)
        return original(material, property_name)

    monkeypatch.setattr(checks.library, "get_property", missing_limit)
    report = assess_shell_bending(export_cable, 0)
    assert report["status"] == "missing_material_limit"
    assert report["layers"][0]["root_bending_utilization"] is None
    with pytest.raises(ValueError):
        require_shell_bending_range(report)


def test_no_shell_is_explicitly_not_applicable(export_cable):
    cable = export_cable.model_copy(
        update={
            "layers": [
                layer
                for layer in export_cable.layers
                if not isinstance(layer, CylindricalShellLayer)
            ]
        }
    )
    report = assess_shell_bending(cable, 0.1)
    assert report["status"] == "no_structural_shells"
    assert report["layers"] == []
    require_shell_bending_range(report)


@pytest.mark.parametrize("assessment", [None, {}, {"status": "unknown"}])
def test_old_or_unknown_assessment_cannot_enter_statistics(assessment):
    with pytest.raises(ValueError):
        require_shell_bending_range(assessment)
