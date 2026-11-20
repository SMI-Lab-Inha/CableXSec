# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Sea-state batch over CableDyn runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import warnings
from dataclasses import replace

import numpy as np
import pytest
from test_cabledyn_reader import _bending_case, _write_case

import cablexsec as cx
from cablexsec._numba import HAVE_NUMBA
from cablexsec.analysis.sea_state_batch import (
    SeaStateBatchOptions,
    SeaStateRun,
    SectionRequest,
    load_batch_spec,
    run_sea_state_batch,
)
from cablexsec.analysis.termination import TerminationSpec
from cablexsec.analysis.thermal_coupling import ThermalState
from cablexsec.cli import batch_main
from cablexsec.exceptions import CableInputError, CableSchemaError

#: Long friction histories run pure Python without numba (minutes per case).
needs_numba = pytest.mark.skipif(
    not HAVE_NUMBA, reason="long friction histories need the compiled kernels"
)

YEAR = 365.25 * 24 * 3600
# Coarse clamp solver settings: the test checks the plumbing, not the clamp law.
CHEAP_CLAMP = {
    "distance_m": 0.5,
    "segments_per_pitch": 8,
    "max_increment": 2.0,
    "reach_lengths": 1.0,
}


@pytest.fixture(scope="module")
def cable():
    return cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)


@pytest.fixture(scope="module")
def stiffness(cable):
    return cx.compute_stiffness(cable)


def _runs(tmp_path, stiffness, amplitudes=(0.01, 0.02), probabilities=(0.6, 0.4)):
    runs = []
    nt = 61
    t = np.arange(nt) * 0.1
    for i, (amplitude, probability) in enumerate(
        zip(amplitudes, probabilities, strict=True)
    ):
        kappa = 0.03 + amplitude * np.sin(2 * math.pi * t / 1.7 + i)
        phi = 0.3 * np.sin(2 * math.pi * t / 2.3)
        positions = _bending_case(kappa, phi, nt)
        tension = np.tile(
            (30000.0 + 4000.0 * np.sin(2 * math.pi * t / 2.0))[:, None], (1, 100)
        )
        root, deck = _write_case(
            tmp_path,
            positions,
            tension=tension,
            root=f"ss{i}",
            deck_kwargs={
                "ea": stiffness.ea_fixed_N,
                "ei": stiffness.ei_min_Nm2,
                "t_max": 6.0,
            },
        )
        runs.append(SeaStateRun(f"ss{i}", root, deck, probability, start_time_s=1.0))
    return runs


@pytest.mark.analytical_sanity
def test_weighted_annual_damage_equals_direct_sum(tmp_path, cable, stiffness):
    runs = _runs(tmp_path, stiffness)
    sections = [SectionRequest("a", 2.0), SectionRequest("b", 7.5)]
    result = run_sea_state_batch(
        cable,
        runs,
        sections,
        output_directory=tmp_path / "out",
        options=SeaStateBatchOptions(tension_basis="true"),
    )
    area = math.pi * result.manifest["external_diameter_m"] ** 2 / 4
    for section in sections:
        total = None
        for run in runs:
            history = result.histories[section.name][run.name]
            assert history.time_s[0] == pytest.approx(1.0)
            kx, ky = history.strain_gradient()
            direct = cx.compute_fatigue(
                cable, kx, ky, tension_N=history.true_tension_N(area)
            )
            rate = {
                layer: values * run.probability * YEAR / history.duration_s
                for layer, values in direct.D_bulk_per_fibre.items()
            }
            total = rate if total is None else {k: total[k] + rate[k] for k in total}
        long_term = result.sections[section.name].long_term
        for layer, values in total.items():
            assert np.allclose(
                long_term.annual_damage_per_fibre[layer], values, rtol=1e-12, atol=0
            )
        expected = max(float(np.max(v)) for v in total.values())
        assert long_term.annual_damage == pytest.approx(expected, rel=1e-12)
        assert result.sections[section.name].life_years == pytest.approx(1 / expected)
    manifest = json.loads(
        (tmp_path / "out" / "manifest.json").read_text(encoding="utf-8")
    )
    for run in manifest["runs"]:
        for record in run["files"]:
            with open(record["path"], "rb") as handle:
                assert hashlib.sha256(handle.read()).hexdigest() == record["sha256"]
        assert {f["role"] for f in run["files"]} >= {
            "main",
            "positions",
            "tensions",
            "deck",
        }
    assert all(
        r["EA_match"] and r["EI_within_bounds"] for r in manifest["property_check"]
    )
    data = np.load(tmp_path / "out" / "a_ss0_history.npz")
    assert "strain_gradient_x_1_m" in list(data["columns"])
    with pytest.raises(FileExistsError):
        run_sea_state_batch(
            cable,
            runs,
            sections,
            output_directory=tmp_path / "out",
            options=SeaStateBatchOptions(tension_basis="true"),
        )


def test_worker_count_does_not_change_results(tmp_path, cable, stiffness):
    runs = _runs(tmp_path, stiffness)
    sections = [SectionRequest("a", 2.0)]
    one = run_sea_state_batch(
        cable, runs, sections, options=SeaStateBatchOptions(workers=1)
    )
    two = run_sea_state_batch(
        cable, runs, sections, options=SeaStateBatchOptions(workers=2)
    )
    a, b = one.sections["a"].long_term, two.sections["a"].long_term
    # Equal to round-off: spawned workers run BLAS on one thread, the calling
    # process on its own setting (2e-12 to 3e-12 relative measured).
    for layer in a.annual_damage_per_fibre:
        np.testing.assert_allclose(
            a.annual_damage_per_fibre[layer],
            b.annual_damage_per_fibre[layer],
            rtol=1e-10,
            atol=0.0,
        )
    assert one.manifest["inputs_sha256"] == two.manifest["inputs_sha256"]


def test_batch_friction_model_follows_the_library_default(tmp_path, cable, stiffness):
    # The default is the library's: its notice on a non-planar path reaches
    # the caller (from the worker processes too), the results are those of
    # the model it resolves to and the manifest records that model.
    assert SeaStateBatchOptions().friction_model is None
    runs = _runs(tmp_path, stiffness)
    sections = [SectionRequest("a", 2.0)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", cx.DirectionTiedFrictionWarning)
        explicit = run_sea_state_batch(
            cable,
            runs,
            sections,
            options=SeaStateBatchOptions(friction_model="direction_tied"),
        )
    for workers in (1, 2):
        with pytest.warns(cx.DirectionTiedFrictionWarning, match="'auto'"):
            default = run_sea_state_batch(
                cable, runs, sections, options=SeaStateBatchOptions(workers=workers)
            )
        a, b = default.sections["a"].long_term, explicit.sections["a"].long_term
        for layer in a.annual_damage_per_fibre:
            np.testing.assert_allclose(
                a.annual_damage_per_fibre[layer],
                b.annual_damage_per_fibre[layer],
                rtol=0.0 if workers == 1 else 1e-10,
                atol=0.0,
            )
        assert default.manifest["options"]["friction_model"] == "direction_tied"
        assert default.manifest["inputs_sha256"] == explicit.manifest["inputs_sha256"]
        assert a.assessment_id == b.assessment_id


def test_in_process_batch_leaves_thread_settings_alone(
    tmp_path, cable, stiffness, monkeypatch
):
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
    run_sea_state_batch(
        cable,
        _runs(tmp_path, stiffness),
        [SectionRequest("a", 2.0)],
        options=SeaStateBatchOptions(workers=1),
    )
    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert "NUMBA_NUM_THREADS" not in os.environ


def test_flooded_cable_takes_effective_tension(tmp_path, cable, stiffness):
    runs = _runs(tmp_path, stiffness)
    result = run_sea_state_batch(cable, runs, [SectionRequest("a", 2.0)])
    assert cable.water_ingress.armour_annulus == "flooded"
    assert result.manifest["tension_basis"] == "effective"
    history = result.histories["a"]["ss0"]
    assert history.external_pressure_Pa is None
    assert np.array_equal(history.true_tension_N(1.0), history.effective_tension_N)


@needs_numba
def test_local_options_termination_thermal_fretting(tmp_path, cable, stiffness):
    runs = _runs(tmp_path, stiffness)
    temperatures = {layer.name: 35.0 for layer in cable.layers}
    sections = [
        SectionRequest("clamped", 0.5, termination=TerminationSpec(**CHEAP_CLAMP)),
        SectionRequest("hot", 4.0, thermal=ThermalState(temperatures)),
        SectionRequest("worn", 6.0, fretting=True),
    ]
    result = run_sea_state_batch(
        cable,
        runs,
        sections,
        options=SeaStateBatchOptions(workers=2),
        output_directory=tmp_path / "o",
    )
    plain = run_sea_state_batch(cable, runs, [SectionRequest("clamped", 0.5)])
    assert (
        result.sections["clamped"].long_term.annual_damage
        != plain.sections["clamped"].long_term.annual_damage
    )
    assert result.sections["hot"].long_term.annual_damage > 0
    fretting = result.sections["worn"].fretting
    assert fretting and all(
        v["max_wear_depth_per_year_m"] >= 0 for v in fretting.values()
    )
    manifest = json.loads(
        (tmp_path / "o" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["sections"][1]["thermal"]["layer_temperature_C"] == temperatures
    assert manifest["sections"][0]["termination"]["distance_m"] == 0.5


@pytest.mark.parametrize(
    "change, match",
    [
        ({"probabilities": (0.6, 0.3)}, "sum to one"),
        ({"stiff": False}, "global line properties"),
    ],
)
def test_inconsistent_batches_fail(tmp_path, cable, stiffness, change, match):
    if change.get("stiff") is False:
        runs = _runs(tmp_path, replace(stiffness, ea_fixed_N=2 * stiffness.ea_fixed_N))
    else:
        runs = _runs(tmp_path, stiffness, probabilities=change["probabilities"])
    with pytest.raises(CableInputError, match=match):
        run_sea_state_batch(cable, runs, [SectionRequest("a", 2.0)])


def test_undeclared_annulus_needs_an_explicit_tension_basis(tmp_path, stiffness):
    menard = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    if menard.water_ingress is not None:
        pytest.skip("cable declares its annulus")
    runs = _runs(tmp_path, stiffness)
    with pytest.raises(CableInputError, match="tension_basis"):
        run_sea_state_batch(menard, runs, [SectionRequest("a", 2.0)])


def test_compressive_material_force_is_named(tmp_path, cable, stiffness):
    runs = _runs(tmp_path, stiffness)
    options = SeaStateBatchOptions(tension_basis="true", external_diameter_m=2.5)
    with pytest.raises(CableSchemaError, match=r"run 'ss0', section 'a'.*< 0"):
        run_sea_state_batch(cable, runs, [SectionRequest("a", 2.0)], options=options)


@pytest.mark.parametrize("workers", [0, -1, True, 2.0])
def test_worker_count_must_be_a_positive_integer(workers):
    with pytest.raises(CableInputError, match="workers"):
        SeaStateBatchOptions(workers=workers)


def test_worker_count_has_no_library_cap():
    # The machine sets the useful count; the pool starts one worker per run
    # at most.
    assert SeaStateBatchOptions(workers=99).workers == 99


def test_tangent_placement_reaches_the_reader(tmp_path, cable):
    """CableDyn 0.4.2 output read as chord tangents fails the curvature check."""
    data = os.path.join(os.path.dirname(__file__), "data", "cabledyn")
    run = SeaStateRun(
        "lw128", os.path.join(data, "lw128"), os.path.join(data, "lw128.dat"), 1.0
    )
    options = SeaStateBatchOptions(
        frame="section", tension_basis="true", allow_property_mismatch=True
    )
    sections = [SectionRequest("hangoff", 0.25)]
    result = run_sea_state_batch(cable, [run], sections, options=options)
    assert result.manifest["options"]["tangent_placement"] == "auto"
    chord = replace(options, tangent_placement="chord")
    with pytest.raises(CableSchemaError, match="needs the default tangent_placement"):
        run_sea_state_batch(cable, [run], sections, options=chord)


@needs_numba
def test_json_specification_and_command_line(tmp_path, cable, stiffness):
    runs = _runs(tmp_path, stiffness)
    spec = {
        "cable": "bundled:iea_task49_66kv",
        "runs": [
            {
                "name": r.name,
                "out_root": r.out_root.name,
                "deck": r.deck.name,
                "probability": r.probability,
                "start_time_s": 1.0,
            }
            for r in runs
        ],
        "sections": [
            {"name": "a", "arc_length_m": 2.0, "termination": CHEAP_CLAMP},
            {"name": "b", "arc_length_m": 7.5},
        ],
        "options": {"n_fibres": 16, "reference_vector": [0, 0, 1]},
    }
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    loaded_cable, loaded_runs, sections, options, cable_path = load_batch_spec(path)
    assert loaded_runs[1].out_root == runs[1].out_root
    assert sections[0].termination.distance_m == 0.5
    assert options.n_fibres == 16
    assert cable_path.name == "iea_task49_66kv.yaml"
    assert batch_main([str(path), "--output-directory", str(tmp_path / "cli")]) == 0
    manifest = json.loads(
        (tmp_path / "cli" / "manifest.json").read_text(encoding="utf-8")
    )
    assert (
        manifest["cable"]["file"]["sha256"]
        == hashlib.sha256(cable_path.read_bytes()).hexdigest()
    )
    assert manifest["options"]["n_fibres"] == 16
    spec["options"]["unknown"] = 1
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(CableInputError, match="options"):
        load_batch_spec(path)
    spec["options"].pop("unknown")
    spec["runs"][0]["seed"] = 3
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(CableInputError, match="unknown keys"):
        load_batch_spec(path)


def test_hang_off_rotation_enters_the_input_digest(tmp_path, cable, stiffness):
    """The hang-off rotation turns the material frame, so two assessments of
    the same files with different rotations have different input digests;
    the same rotation gives the same digest."""
    nt = 31
    t = np.arange(nt) * 0.1
    positions = _bending_case(0.03 + 0.01 * np.sin(t), 0.2 * np.sin(t), nt)
    channels = {f"Twist1N{j}": np.zeros(nt) for j in (21, 22)}
    root, deck = _write_case(
        tmp_path,
        positions,
        tension=np.full((nt, 100), 30000.0),
        channels=channels,
        deck_kwargs={
            "ea": stiffness.ea_fixed_N,
            "ei": stiffness.ei_min_Nm2,
            "torsion": True,
        },
    )
    options = SeaStateBatchOptions(tension_basis="true", allow_property_mismatch=True)
    digests = []
    for rotation in ("none", np.linspace(0.0, 30.0, nt), "none"):
        run = SeaStateRun("r", root, deck, 1.0, hang_off_rotation=rotation)
        result = run_sea_state_batch(
            cable, [run], [SectionRequest("a", 2.0)], options=options
        )
        digests.append(result.manifest["inputs_sha256"])
    assert digests[0] == digests[2] != digests[1]
