# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Application-level ownership, persistence and numerical-equivalence contracts."""

import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, is_dataclass, replace

import numpy as np
import pytest
from pydantic import ValidationError

import cablexsec as cx
from cablexsec._project_codec import canonical, pack, parse, unpack
from cablexsec.materials import library


def series():
    t = np.linspace(0, 20, 41)
    return cx.LoadCaseSeries(
        t,
        0.006 * np.sin(t),
        np.zeros_like(t),
        np.full_like(t, 8e4),
        tension_kind="true",
        source="test load",
    )


def model(path=None):
    return cx.CableModel.from_yaml(path or cx.inter_array_single_core_yaml_path())


def request(kind="stiffness"):
    settings = {
        "stiffness": cx.StiffnessSettings(),
        "bending": cx.BendingSettings(),
        "stress": cx.StressSettings(
            kappa_x_1_m=0.006, kappa_y_1_m=0.003, tension_N=8e4
        ),
        "fatigue": cx.FatigueSettings(n_fibres=8),
    }[kind]
    return cx.AnalysisRequest(
        name=kind,
        model="cable",
        settings=settings,
        load_case="sea" if kind in {"bending", "fatigue"} else None,
    )


def project(kind="stiffness"):
    p = cx.Project("Engineering assessment")
    p.set_model("cable", model())
    p.set_load_case("sea", series())
    p.set_analysis(request(kind))
    return p


def equal(actual, expected):
    if isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    elif is_dataclass(expected):
        assert type(actual) is type(expected)
        for field in fields(expected):
            equal(getattr(actual, field.name), getattr(expected, field.name))
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            equal(actual[key], expected[key])
    elif isinstance(expected, float) and np.isnan(expected):
        assert np.isnan(actual)
    else:
        assert actual == expected


@pytest.mark.parametrize(
    "path",
    [
        cx.inter_array_single_core_yaml_path,
        cx.inter_array_three_core_yaml_path,
        cx.export_three_core_yaml_path,
    ],
)
@pytest.mark.parametrize("kind", ["stiffness", "stress", "bending", "fatigue"])
def test_object_api_is_numerically_identical_to_functions(path, kind):
    m, s, r = model(path()), series(), request(kind)
    actual = cx.AnalysisRunner().run(m, r, s if r.load_case else None)
    if kind == "stiffness":
        expected = cx.compute_stiffness(m.definition)
    elif kind == "stress":
        expected = cx.compute_stress(
            m.definition, 0.006, 0.003, tension_N=8e4, contact_model="bounded"
        )
    elif kind == "bending":
        expected = cx.compute_mkappa(
            m.definition, s.curvature_x_1_m, tension_N=8e4, contact_model="bounded"
        )
    else:
        expected = cx.fatigue_from_series(m.definition, s, n_fibres=8)
    equal(actual.result, expected)
    assert actual.request == r
    assert actual.engine["cablexsec"] == cx.__version__


@pytest.mark.parametrize("kind", ["stiffness", "stress", "bending", "fatigue"])
def test_project_roundtrip_preserves_inputs_results_and_staleness(tmp_path, kind):
    p = project(kind)
    record = p.run(kind)
    assert not p.is_stale(record)
    path = p.save(tmp_path / "assessment.cxproj")
    saved = path.read_bytes()
    restored = cx.Project.load(path)
    assert restored.name == p.name
    assert len(restored.records) == 1
    equal(restored.records[0].result, record.result)
    equal(restored.load_cases["sea"], p.load_cases["sea"])
    assert restored.records[0].fingerprint == record.fingerprint
    assert not restored.is_stale(restored.records[0])
    restored.save(path)
    assert path.read_bytes() == saved
    second = restored.run(kind)
    assert second.run_id != record.run_id
    equal(second.result, record.result)
    assert len(restored.records) == 2


@pytest.mark.parametrize("edit", ["model", "load", "settings", "materials"])
def test_relevant_edits_mark_results_stale_without_altering_record(edit, monkeypatch):
    p = project("fatigue")
    record = p.run("fatigue")
    before = copy.deepcopy(record.result)
    if edit == "model":
        cable = p.models["cable"].definition
        raw = cable.model_dump(mode="json")
        raw["cable"]["name"] = "changed construction"
        p.set_model("cable", cx.CableModel(cx.CableData.model_validate(raw)))
    elif edit == "load":
        p.set_load_case("sea", replace(p.load_cases["sea"], tension_N=np.full(41, 9e4)))
    elif edit == "settings":
        p.set_analysis(
            request("fatigue").model_copy(
                update={"settings": cx.FatigueSettings(n_fibres=12)}
            )
        )
    else:
        material = copy.deepcopy(library.MATERIALS["galvanised_steel_armour"])
        value, source = material["E_Pa"]
        material["E_Pa"] = (value * 1.01, source)
        monkeypatch.setitem(library.MATERIALS, "galvanised_steel_armour", material)
    assert p.is_stale(record)
    equal(record.result, before)


def test_unrelated_edit_and_result_mutation_cannot_change_record():
    p = project("bending")
    record = p.run("bending")
    expected = record.result
    result = record.result
    result.moment_Nm[:] = -99
    p.set_model("other", model())
    p.set_load_case("other", series())
    assert not p.is_stale(record)
    equal(record.result, expected)
    with pytest.raises(TypeError):
        p.models["other"] = model()
    with pytest.raises(TypeError):
        p.analyses["x"] = request()


def test_failed_analysis_leaves_completed_records_intact():
    p = project("bending")
    first = p.run("bending")
    p.set_load_case("sea", replace(series(), curvature_y_1_m=np.ones(41)))
    with pytest.raises(cx.CableInputError, match="zero y-curvature"):
        p.run("bending")
    assert p.records == (first,)
    assert p.is_stale(first)


def test_runner_has_no_shared_solver_state():
    m, r, s = model(), request("bending"), series()
    runner = cx.AnalysisRunner()
    with ThreadPoolExecutor(max_workers=2) as executor:
        outputs = list(executor.map(lambda _: runner.run(m, r, s), range(2)))
    equal(outputs[0].result, outputs[1].result)
    assert outputs[0].run_id != outputs[1].run_id
    assert outputs[0].fingerprint == outputs[1].fingerprint


@pytest.mark.parametrize(
    "changes",
    [
        {"name": " "},
        {"model": " cable"},
        {"load_case": "sea"},
        {"settings": {"kind": "unknown"}},
        {"extra": True},
    ],
)
def test_request_validation(changes):
    raw = request().model_dump()
    raw.update(changes)
    with pytest.raises((ValidationError, cx.CableInputError)):
        cx.AnalysisRequest.model_validate(raw)


@pytest.mark.parametrize(
    "options",
    [
        {"n_fibres": 3},
        {"n_fibres": True},
        {"mu": -1.0},
        {"T_ref_C": np.nan},
        {"n_fibres": 4.5},
        {"unknown": 0},
    ],
)
def test_settings_validation(options):
    with pytest.raises(ValidationError):
        cx.FatigueSettings(**options)


def test_revalidation_catches_model_copy_bypass_and_preserves_previous_request():
    p = project("fatigue")
    invalid = request("fatigue").model_copy(
        update={"settings": cx.FatigueSettings().model_copy(update={"n_fibres": -1})}
    )
    with pytest.raises(ValidationError):
        p.set_analysis(invalid)
    assert p.analyses["fatigue"] == request("fatigue")
    with pytest.raises(ValidationError):
        p.models["cable"].fatigue(series(), invalid.settings)


@pytest.mark.parametrize("name", ["", " ", " padded "])
def test_project_names_are_validated(name):
    with pytest.raises(cx.CableInputError):
        cx.Project(name)


def test_missing_references_and_runner_history_contract():
    p = cx.Project("empty")
    with pytest.raises(cx.CableLookupError):
        p.set_analysis(request())
    with pytest.raises(cx.CableLookupError):
        p.run("absent")
    p.set_model("cable", model())
    with pytest.raises(cx.CableLookupError):
        p.set_analysis(request("fatigue"))
    with pytest.raises(cx.CableInputError):
        cx.AnalysisRunner().run(model(), request(), series())
    with pytest.raises(cx.CableInputError):
        cx.AnalysisRunner().run(model(), request("fatigue"))
    record = cx.AnalysisRunner().run(model(), request())
    assert p.is_stale(record)


@pytest.mark.parametrize(
    "changes",
    [
        {"tension_kind": "effective"},
        {"temperature_C": np.zeros(41)},
        {"axial_strain": np.zeros(41)},
        {"curvature_y_1_m": np.ones(41)},
        {"tension_N": np.arange(41.0)},
    ],
)
def test_bending_rejects_unconsumed_or_incompatible_history(changes):
    with pytest.raises(cx.CableInputError):
        model().bending(replace(series(), **changes))


def test_atomic_save_preserves_previous_file_on_failure(tmp_path, monkeypatch):
    import cablexsec.project as implementation

    p = project()
    path = p.save(tmp_path / "project.json")
    before = path.read_bytes()

    def fail(*args):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(implementation.os, "replace", fail)
    with pytest.raises(OSError, match="replacement"):
        p.save(path)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".cablexsec-*.tmp"))


def rewrite(path, mutation):
    document = json.loads(path.read_text())
    mutation(document)
    document["sha256"] = hashlib.sha256(
        canonical(document["payload"]).encode()
    ).hexdigest()
    path.write_text(json.dumps(document), encoding="utf-8")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(version=3),
        lambda d: d.update(version=True),
        lambda d: d.update(format="another format"),
        lambda d: d.update(extra=1),
        lambda d: d["payload"].update(extra=1),
        lambda d: d["payload"].update(models=[]),
        lambda d: d["payload"]["load_cases"].update(sea=12),
        lambda d: d["payload"]["analyses"].append(d["payload"]["analyses"][0]),
        lambda d: d["payload"]["analyses"][0].update(model="absent"),
    ],
)
def test_load_rejects_invalid_envelopes_and_references(tmp_path, mutation):
    path = project().save(tmp_path / "bad.cxproj")
    rewrite(path, mutation)
    with pytest.raises(cx.CableInputError):
        cx.Project.load(path)


def test_load_rejects_binary_and_truncated_files(tmp_path):
    binary = tmp_path / "binary.cxproj"
    binary.write_bytes(bytes(range(256)))
    with pytest.raises(cx.CableInputError, match="not UTF-8"):
        cx.Project.load(binary)
    path = project().save(tmp_path / "truncated.cxproj")
    path.write_bytes(path.read_bytes()[:-40])
    with pytest.raises(cx.CableInputError, match="invalid project JSON"):
        cx.Project.load(path)


def test_load_rejects_corruption_and_native_missing_file(tmp_path):
    path = project().save(tmp_path / "bad.cxproj")
    document = json.loads(path.read_text())
    document["payload"]["name"] = "corrupted"
    path.write_text(json.dumps(document))
    with pytest.raises(cx.CableInputError, match="checksum"):
        cx.Project.load(path)
    with pytest.raises(FileNotFoundError):
        cx.Project.load(tmp_path / "missing")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["records"].append(p["records"][0]),
        lambda p: p["records"][0].update(_result_json=canonical(pack(12))),
        lambda p: p["records"][0].update(_engine_json="[]"),
        lambda p: p["records"][0].update(created_utc="invalid"),
        lambda p: p["records"][0].update(run_id=""),
        lambda p: p["records"][0].update(
            _inputs_json=canonical(
                {
                    "request": request().model_dump(),
                    "model": model().definition.model_dump(mode="json"),
                    "series": pack(series()),
                }
            )
        ),
    ],
)
def test_load_rejects_inconsistent_records(tmp_path, mutation):
    p = project()
    p.run("stiffness")
    path = p.save(tmp_path / "bad.cxproj")
    rewrite(path, lambda d: mutation(d["payload"]))
    with pytest.raises(cx.CableInputError):
        cx.Project.load(path)


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        4,
        0.2,
        "name",
        float("inf"),
        float("-inf"),
        float("nan"),
        (1, "x"),
        [1, "x"],
        {"array": [1, 2]},
        np.empty((0, 3)),
        np.array([[True, False]]),
        np.array([1, 2], dtype=np.int32),
    ],
)
def test_codec_preserves_values_and_nonfinite_result_markers(value):
    decoded = unpack(parse(canonical(pack(value))))
    equal(decoded, value)


@pytest.mark.parametrize(
    "value", [object(), {1: "bad key"}, np.array([object()]), np.array([1j])]
)
def test_codec_rejects_unsupported_values(value):
    with pytest.raises(cx.CableInputError):
        pack(value)


@pytest.mark.parametrize(
    "value",
    [
        [],
        {"unknown": 1},
        {"float": "bad"},
        {"list": 1},
        {"type": "NotAllowed", "fields": {}},
        {"type": "Provenance", "fields": {}},
        {"array": {"list": [1]}, "dtype": "O", "shape": [1]},
        {"array": {"list": [1]}, "dtype": "f8", "shape": [2]},
        {"array": {"list": [1]}, "dtype": "f8", "shape": [True]},
        {"array": {"list": [1]}, "dtype": "invalid", "shape": [1]},
    ],
)
def test_codec_rejects_malformed_tags(value):
    with pytest.raises(cx.CableInputError):
        unpack(value)


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', "NaN", "Infinity", "{"])
def test_project_json_is_strict(text):
    with pytest.raises(cx.CableInputError):
        parse(text)


def test_saved_environment_and_historical_inputs_survive_edits(tmp_path, monkeypatch):
    import cablexsec.project as implementation

    p = project("fatigue")
    record = p.run("fatigue")
    assert not p.environment_changed
    original_engine = p.engine
    p.set_load_case("sea", replace(series(), tension_N=np.full(41, 9e4)))
    path = p.save(tmp_path / "history.cxproj")
    changed = p.engine
    changed["cablexsec"] = "future-version"
    monkeypatch.setattr(implementation, "_engine", lambda: canonical(changed))
    restored = cx.Project.load(path)
    assert restored.environment_changed
    assert restored.engine == original_engine
    assert restored.is_stale(restored.records[0])
    equal(restored.records[0].series, series())
    assert restored.records[0].model.definition == record.model.definition
    np.testing.assert_array_equal(restored.load_cases["sea"].tension_N, 9e4)


def test_empty_project_roundtrip_and_failed_edits(tmp_path):
    p = cx.Project("empty")
    with pytest.raises(cx.CableInputError):
        p.name = " "
    assert p.name == "empty"
    with pytest.raises(cx.CableInputError):
        p.set_model("model", None)
    with pytest.raises(cx.CableInputError):
        p.set_load_case("history", None)
    restored = cx.Project.load(p.save(tmp_path / "empty.cxproj"))
    assert not restored.records and not restored.models and not restored.load_cases
    assert restored.engine == p.engine
    with pytest.raises(FileNotFoundError):
        p.save(tmp_path / "missing" / "file.cxproj")


def test_invalid_saved_project_engine_is_rejected(tmp_path):
    path = project().save(tmp_path / "bad.cxproj")
    rewrite(path, lambda d: d["payload"].update(engine=[]))
    with pytest.raises(cx.CableInputError, match="engine"):
        cx.Project.load(path)


def test_biaxial_direct_strain_thermal_fatigue_matches_existing_bridge():
    m = model()
    source = series()
    history = replace(
        source,
        curvature_y_1_m=0.002 * np.cos(source.time_s),
        axial_strain=np.full(41, 1e-4),
        axial_strain_kind="direct",
        temperature_C=np.full(41, 30.0),
    )
    settings = cx.FatigueSettings(
        n_fibres=12, mu=0.15, contact_tension_N=8e4, T_ref_C=20.0
    )
    equal(
        m.fatigue(history, settings),
        cx.fatigue_from_series(
            m.definition,
            history,
            n_fibres=12,
            mu=0.15,
            contact_tension_N=8e4,
            T_ref_C=20.0,
        ),
    )


def test_object_stress_preserves_pressure_options():
    m = model()
    options = cx.StressSettings(
        kappa_x_1_m=0.005,
        kappa_y_1_m=-0.002,
        tension_N=8e4,
        p_int_Pa=1e5,
        p_ext_Pa=2e5,
        torque_Nm=0.0,
        prescribed_twist_per_m=0.0,
        mu=0.15,
    )
    equal(
        m.stress(options),
        cx.compute_stress(
            m.definition,
            0.005,
            -0.002,
            tension_N=8e4,
            p_int_Pa=1e5,
            p_ext_Pa=2e5,
            torque_Nm=0.0,
            prescribed_twist_per_m=0.0,
            mu=0.15,
            contact_model="bounded",
        ),
    )


def test_object_stress_preserves_unsupported_torque_rejection():
    m = model()
    with pytest.raises(cx.CableSchemaError, match="coupled section"):
        m.stress(cx.StressSettings(torque_Nm=10.0, prescribed_twist_per_m=None))
    with pytest.raises(cx.CableSchemaError, match="coupled section"):
        cx.compute_stress(
            m.definition, 0.0, torque_Nm=10.0, prescribed_twist_per_m=None
        )


def test_optional_solver_availability_is_recorded(monkeypatch):
    from importlib.metadata import PackageNotFoundError

    import cablexsec.project as implementation

    original = implementation.version

    def without_contact(name):
        if name == "clarabel":
            raise PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(implementation, "version", without_contact)
    p = cx.Project("no optional solver")
    assert p.engine["clarabel"] is None
    assert not p.environment_changed
