# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Regression cases from the public distribution and input-boundary audit."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

import cablexsec as cx
from cablexsec._project_codec import canonical, pack, unpack
from cablexsec.analysis.fatigue import rainflow_cycles
from cablexsec.provenance import real_array, require_finite


@pytest.fixture
def cable():
    return cx.load_cable(cx.inter_array_single_core_yaml_path(), geometry_override=None)


@pytest.mark.parametrize(
    "argument", ["kappa_x", "kappa_y", "tension_N", "axial_strain", "thermal_stress_Pa"]
)
def test_fatigue_rejects_complex_inputs_without_cast_warning(cable, argument):
    args = {
        "kappa_x": np.array([0.0, 0.01, 0.0]),
        "kappa_y": np.zeros(3),
        "n_fibres": 4,
    }
    bad = np.array([0.0, 0.01 + 0.02j, 0.0])
    args[argument] = (
        {cable.helical_layers[0].name: bad} if argument == "thermal_stress_Pa" else bad
    )
    with pytest.raises(cx.CableXSecError, match="real"):
        cx.compute_fatigue(cable, **args)


@pytest.mark.parametrize("imag", [0.0, 0.02])
def test_bending_and_cycle_counting_reject_complex_histories(cable, imag):
    values = np.array([0, 0.01 + imag * 1j, 0])
    with pytest.raises(cx.CableXSecError, match="real"):
        cx.compute_mkappa(cable, values)
    with pytest.raises(cx.CableXSecError, match="real"):
        list(rainflow_cycles(values))
    with pytest.raises(cx.CableXSecError, match="real"):
        cx.compute_shell_fatigue(cable, "unused", values)


@pytest.mark.parametrize(
    "value", [[0, "invalid"], [0, 10**400], np.array([1j], dtype=object)]
)
def test_real_array_rejects_unrepresentable_inputs(value):
    with pytest.raises(cx.PhysicalRangeError, match="input"):
        real_array(value, "input")


def test_real_array_keeps_valid_values():
    source = np.array([0.0, -0.2, 0.5])
    assert real_array(source, "input") is source
    require_finite((value for value in source), "input")


@pytest.mark.parametrize(
    "field,value",
    [
        ("ea_fixed_N", "not numeric"),
        ("ea_fixed_N", True),
        ("xi_per_layer", {"mapping": {"layer": "bad"}}),
        ("cable_name", 12),
        ("provenance", "missing provenance"),
    ],
)
def test_result_field_types_are_checked(cable, field, value):
    encoded = pack(cx.compute_stiffness(cable))
    encoded["fields"][field] = value
    with pytest.raises(cx.CableInputError, match=field):
        unpack(encoded)


def test_result_array_dtype_is_checked(cable):
    encoded = pack(cx.compute_mkappa(cable, np.array([0.0, 0.001, 0.0])))
    encoded["fields"]["moment_Nm"]["dtype"] = "i8"
    with pytest.raises(cx.CableInputError, match="moment_Nm"):
        unpack(encoded)


@pytest.mark.parametrize("contact", ["bounded", "qp"])
def test_pressure_stress_projects_preserve_nested_section_states(
    tmp_path, cable, contact
):
    project = cx.Project("pressure stress")
    project.set_model("cable", cx.CableModel(cable))
    settings = cx.StressSettings(
        contact_model=contact, kappa_x_1_m=0.003, tension_N=8e4, p_ext_Pa=1e5
    )
    project.set_analysis(
        cx.AnalysisRequest(name="stress", model="cable", settings=settings)
    )
    record = project.run("stress")
    original = cx.compute_stress(
        cable, 0.003, tension_N=8e4, p_ext_Pa=1e5, contact_model=contact
    )
    assert canonical(pack(record.result)) == canonical(pack(original))
    restored = cx.Project.load(project.save(tmp_path / "pressure.cxproj"))
    assert canonical(pack(restored.records[0].result)) == canonical(pack(original))
    state = restored.records[0].result.section_state
    assert state.cable == cable
    assert not restored.is_stale(restored.records[0])


def document(tmp_path, cable):
    p = cx.Project("sanity")
    p.set_model("cable", cx.CableModel(cable))
    p.set_analysis(
        cx.AnalysisRequest(
            name="stiffness", model="cable", settings=cx.StiffnessSettings()
        )
    )
    p.run("stiffness")
    path = p.save(tmp_path / "project.cxproj")
    return path, json.loads(path.read_text())


def write_modified(path, data):
    data["sha256"] = hashlib.sha256(canonical(data["payload"]).encode()).hexdigest()
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize(
    "edit",
    [
        lambda e: e.clear(),
        lambda e: e.update(python=1),
        lambda e: e.update(source_sha256="bad"),
        lambda e: e.update(dependencies={}),
        lambda e: e.update(clarabel=1),
        lambda e: e.update(materials=12),
    ],
)
def test_incomplete_or_mistyped_environment_is_rejected(tmp_path, cable, edit):
    path, data = document(tmp_path, cable)
    edit(data["payload"]["engine"])
    write_modified(path, data)
    with pytest.raises(cx.CableInputError, match="engine"):
        cx.Project.load(path)


@pytest.mark.parametrize("history", [12, "not a history", {"mapping": {}}])
def test_non_history_analysis_cannot_hide_arbitrary_history(tmp_path, cable, history):
    path, data = document(tmp_path, cable)
    record = data["payload"]["records"][0]
    inputs = json.loads(record["_inputs_json"])
    inputs["series"] = history
    record["_inputs_json"] = canonical(inputs)
    write_modified(path, data)
    with pytest.raises(cx.CableInputError, match="history"):
        cx.Project.load(path)


def test_extra_record_inputs_are_rejected(tmp_path, cable):
    path, data = document(tmp_path, cable)
    record = data["payload"]["records"][0]
    inputs = json.loads(record["_inputs_json"])
    inputs["ignored"] = "data"
    record["_inputs_json"] = canonical(inputs)
    write_modified(path, data)
    with pytest.raises(cx.CableInputError, match="input fields"):
        cx.Project.load(path)


def test_long_term_damage_cannot_discard_imaginary_component(cable):
    result = cx.compute_fatigue(
        cable, np.array([0.0, 0.003, 0.0]), np.zeros(3), n_fibres=4
    )
    damaged = {
        name: values.astype(complex) + 1j
        for name, values in result.D_bulk_per_fibre.items()
    }
    case = cx.FatigueLoadCase(
        "case", replace(result, D_bulk_per_fibre=damaged), 10.0, 1.0, "assessment"
    )
    with pytest.raises(cx.CableXSecError, match="real"):
        cx.compute_long_term_fatigue([case], exposure_years=20.0)
