# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Sea-state batch: input checks, specification entries and records."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from test_capacity_check import _linear_envelope
from test_sea_state_batch import _runs

import cablexsec as cx
from cablexsec.analysis.sea_state_batch import (
    CapacityBatchCheck,
    SeaStateBatchOptions,
    SeaStateRun,
    SectionRequest,
    _json_default,
    _limit_threads,
    _portable,
    _restore,
    _termination_record,
    _thermal_record,
    load_batch_spec,
    run_sea_state_batch,
)
from cablexsec.analysis.termination import TerminationSpec
from cablexsec.analysis.thermal_coupling import ThermalHistory, ThermalState
from cablexsec.exceptions import CableInputError


@pytest.fixture(scope="module")
def cable():
    return cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)


@pytest.fixture(scope="module")
def stiffness(cable):
    return cx.compute_stiffness(cable)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": " "}, "run names"),
        ({"name": "a/b"}, "run names"),
        ({"probability": 1.5}, "probability"),
        ({"probability": math.nan}, "probability"),
        ({"start_time_s": -1.0}, "start_time_s"),
        ({"start_time_s": 5.0, "end_time_s": 5.0}, "end_time_s"),
    ],
)
def test_run_checks(tmp_path, kwargs, match):
    base = {"name": "r", "out_root": tmp_path / "r", "deck": tmp_path / "r.dat"}
    base.setdefault("probability", 0.5)
    with pytest.raises(CableInputError, match=match):
        SeaStateRun(**{**base, **kwargs})


@pytest.mark.parametrize(
    ("args", "match"),
    [
        (("", 1.0), "section names"),
        (("a\\b", 1.0), "section names"),
        (("a", -1.0), "arc_length_m"),
        (("a", math.inf), "arc_length_m"),
    ],
)
def test_section_checks(args, match):
    with pytest.raises(CableInputError, match=match):
        SectionRequest(*args)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tension_basis": "wall"}, "tension_basis"),
        ({"external_diameter_m": 0.0}, "external_diameter_m"),
        ({"property_rtol": 0.0}, "property_rtol"),
    ],
)
def test_option_checks(kwargs, match):
    with pytest.raises(CableInputError, match=match):
        SeaStateBatchOptions(**kwargs)


def test_capacity_check_options():
    with pytest.raises(CableInputError, match="CapacityEnvelope"):
        CapacityBatchCheck("envelope")  # type: ignore[arg-type]
    with pytest.raises(CableInputError, match="fractile"):
        CapacityBatchCheck(_linear_envelope(), fractile=1.0)


def test_batch_level_checks(tmp_path, cable, stiffness):
    runs = _runs(tmp_path, stiffness)
    section = SectionRequest("a", 2.0)
    with pytest.raises(CableInputError, match="at least one run"):
        run_sea_state_batch(cable, [], [section])
    with pytest.raises(CableInputError, match="run names must be unique"):
        run_sea_state_batch(cable, [runs[0], runs[0]], [section])
    with pytest.raises(CableInputError, match="section names must be unique"):
        run_sea_state_batch(cable, runs, [section, section])
    check = CapacityBatchCheck(_linear_envelope(), groups={"g": ["nope"]})
    with pytest.raises(CableInputError, match="unknown runs"):
        run_sea_state_batch(cable, runs, [section], capacity=check)


def test_in_process_fretting_and_single_seed_capacity_groups(
    tmp_path, cable, stiffness
):
    runs = _runs(tmp_path, stiffness)
    envelope = cx.build_capacity_envelope(cable, np.linspace(1.5e4, 4.5e4, 7))
    check = CapacityBatchCheck(envelope, groups={r.name: [r.name] for r in runs})
    result = run_sea_state_batch(
        cable,
        runs,
        [
            SectionRequest(
                "worn",
                6.0,
                fretting=True,
                sn_curves={"inner_armour": "dnv_rpc203_t_curve"},
            )
        ],
        options=SeaStateBatchOptions(tension_basis="true", workers=1),
        capacity=check,
    )
    section = result.sections["worn"]
    assert section.fretting and all(
        v["max_wear_depth_per_year_m"] >= 0 for v in section.fretting.values()
    )
    for group in section.capacity["extremes"].values():
        assert group["fit"] is None and "fewer than two" in group["note"]
    assert result.manifest["sections"][0]["sn_curves"] == {
        "inner_armour": "dnv_rpc203_t_curve"
    }


def test_records_and_worker_transfer(cable):
    profile = (np.array([0.0, 1.0]), np.array([1.0, 0.5]))
    record = _termination_record(
        TerminationSpec(distance_m=0.5, curvature_profile=profile)
    )
    assert record["curvature_profile"] == [[0.0, 1.0], [1.0, 0.5]]
    history = ThermalHistory(time_s=[0.0, 1.0], layer_temperature_C={"a": [20.0, 30.0]})
    thermal = _thermal_record(history)
    assert thermal["kind"] == "ThermalHistory" and thermal["layer_temperature_C"] == {
        "a": [20.0, 30.0]
    }
    hot = SectionRequest(
        "hot", 4.0, thermal=ThermalState({layer.name: 35.0 for layer in cable.layers})
    )
    portable = _portable(hot)
    assert isinstance(portable, tuple)
    restored = _restore(portable)
    assert restored.name == "hot" and restored.thermal == hot.thermal


def test_thread_limit_of_a_worker(monkeypatch):
    for name in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
        monkeypatch.delenv(name, raising=False)
    from cablexsec._numba import HAVE_NUMBA

    numba = __import__("numba") if HAVE_NUMBA else None
    before = numba.get_num_threads() if numba else None
    try:
        _limit_threads()
        assert os.environ["OMP_NUM_THREADS"] == "1"
        assert os.environ["NUMBA_NUM_THREADS"] == "1"
        if numba:
            assert numba.get_num_threads() == 1
    finally:
        if numba:
            numba.set_num_threads(before)


def test_manifest_json_encoder():
    assert _json_default(np.float64(2.5)) == 2.5
    assert _json_default(np.array([1, 2])) == [1, 2]
    assert _json_default(Path("a") / "b") == str(Path("a") / "b")
    with pytest.raises(TypeError, match="not JSON serialisable"):
        _json_default(object())


def _spec(tmp_path, runs, **changes):
    spec = {
        "cable": "bundled:iea_task49_66kv",
        "runs": [
            {
                "name": r.name,
                "out_root": r.out_root.name,
                "deck": r.deck.name,
                "probability": r.probability,
            }
            for r in runs
        ],
        "sections": [{"name": "a", "arc_length_m": 2.0}],
    }
    spec.update(changes)
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def test_specification_entries(tmp_path, stiffness):
    runs = _runs(tmp_path, stiffness)
    yaml = Path(cx.iea_task49_66kv_yaml_path())
    (tmp_path / "cable.yaml").write_bytes(yaml.read_bytes())
    sections = [
        {
            "name": "a",
            "arc_length_m": 2.0,
            "mean_stress": {
                "inner_armour": {"method": "swt"},
                "outer_armour": "goodman",
            },
            "scf": {"inner_armour": {"axial_dynamic": 1.1, "source": "assumed"}},
            "termination": {
                "distance_m": 0.5,
                "curvature_profile": [[0.0, 1.0], [1.0, 0.5]],
            },
        },
        {
            "name": "b",
            "arc_length_m": 3.0,
            "mean_stress": {"method": "swt"},
            "scf": {"source": "unit", "axial_mean": 1.0},
        },
    ]
    _, _, loaded, _, cable_path = load_batch_spec(
        _spec(tmp_path, runs, cable="cable.yaml", sections=sections)
    )
    assert cable_path == tmp_path / "cable.yaml"
    assert loaded[0].mean_stress["inner_armour"].method == "swt"
    assert loaded[0].mean_stress["outer_armour"] == "goodman"
    assert loaded[0].scf["inner_armour"].axial_dynamic == 1.1
    np.testing.assert_array_equal(
        loaded[0].termination.curvature_profile[1], [1.0, 0.5]
    )
    assert loaded[1].mean_stress.method == "swt" and loaded[1].scf.source == "unit"


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"cable": ""}, "'cable' must name a YAML file"),
        (
            {"sections": [{"name": "a", "arc_length_m": 2.0, "mean_stress": 3}]},
            "mean_stress must be",
        ),
        (
            {"sections": [{"name": "a", "arc_length_m": 2.0, "scf": 1.2}]},
            "scf must be an object",
        ),
    ],
)
def test_specification_errors(tmp_path, stiffness, change, match):
    runs = _runs(tmp_path, stiffness)
    with pytest.raises(CableInputError, match=match):
        load_batch_spec(_spec(tmp_path, runs, **change))


def test_specification_must_be_an_object(tmp_path):
    path = tmp_path / "batch.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(CableInputError, match="must be an object"):
        load_batch_spec(path)


def test_cable_source_is_checked_before_any_run_is_read(tmp_path, cable):
    # The runs point nowhere: a late check would fail reading them first.
    runs = [SeaStateRun("r", tmp_path / "none", tmp_path / "none.dat", 1.0)]
    with pytest.raises(FileNotFoundError, match=r"missing.yaml is not a file"):
        run_sea_state_batch(
            cable,
            runs,
            [SectionRequest("a", 2.0)],
            cable_source=tmp_path / "missing.yaml",
        )
