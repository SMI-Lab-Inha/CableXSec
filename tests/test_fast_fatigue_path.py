# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""The fused fatigue path equals the reference arithmetic it replaces.

``fibre_miner_damage`` counts every fixed fibre of a stress field without
forming the fibre histories; ``assemble_wire_field`` forms the field in one
pass. Both must reproduce the expressions they replace (the compiled counter
on ``WireStressField.at_fibre`` histories, cycle for cycle, and the NumPy
field expressions), so that ``compute_fatigue`` is unchanged. Timings are
recorded by ``validation/scripts/benchmark_throughput.py``; no wall-clock
assertion is made here.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import replace

import benchmark_throughput as benchmark
import numpy as np
import pytest

import cablexsec as cx
import cablexsec.analysis.fatigue as fatigue
from cablexsec.analysis.fast_field import assemble_wire_field
from cablexsec.analysis.fast_rainflow import fibre_miner_damage, miner_rows
from cablexsec.analysis.fatigue import FatigueLayer, compute_fatigue, wire_stress_field
from cablexsec.analysis.termination import TerminationSpec
from cablexsec.exceptions import CableInputError
from cablexsec.materials.sn_curves import SN_CURVES, SNcurve
from cablexsec.mechanics.mkappa import HelicalLayerSpec
from cablexsec.mechanics.wire_stress import local_bending_fibre_stress

CURVES = [
    SN_CURVES["saevik_2011_flat_steel"],
    SN_CURVES["dnv_rpc203_t_curve"],
    SNcurve(
        name="test_threshold",
        stress_unit="MPa",
        m1=3.0,
        log_C1=12.0,
        m2=5.0,
        log_C2=15.5,
        threshold=5.0,
    ),
]


def _field(n_t=240, seed=11):
    """Field components with plateaus, constant and monotonic rows."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_t)
    mean = np.vstack(
        [
            40e6 * np.sin(9 * t) + rng.normal(0, 5e6, n_t),
            np.round(rng.normal(0, 30e6, n_t) / 10e6) * 10e6,  # plateaus, ties
            np.full(n_t, 7e6),  # constant
            np.linspace(-20e6, 50e6, n_t),  # monotonic
            np.zeros(n_t),
        ]
    )
    radial = np.vstack(
        [
            25e6 * np.cos(17 * t),
            np.zeros(n_t),
            np.zeros(n_t),
            np.zeros(n_t),
            30e6 * np.sin(23 * t),
        ]
    )
    tangential = np.vstack(
        [
            rng.normal(0, 8e6, n_t),
            np.round(rng.normal(0, 20e6, n_t) / 10e6) * 10e6,
            np.zeros(n_t),
            np.zeros(n_t),
            10e6 * np.cos(5 * t),
        ]
    )
    return mean, radial, tangential


def _offsets(n_fibres):
    angles = np.arange(n_fibres) * (2 * math.pi / n_fibres)
    return (
        np.array([math.cos(a) for a in angles]),
        np.array([math.sin(a) for a in angles]),
    )


@pytest.mark.parametrize("curve", CURVES, ids=lambda c: c.name)
@pytest.mark.parametrize("start", [0, 1, 37, 238])
@pytest.mark.parametrize("factors", [[1.0], [1.0, 1.7], [1.0, 0.3]])
def test_fibre_damage_equals_the_compiled_counter_on_fibre_histories(
    curve, start, factors
):
    mean, radial, tangential = _field()
    normal, lateral = _offsets(6)
    damage, rerun = fibre_miner_damage(
        mean, radial, tangential, normal, lateral, curve, factors, start=start
    )
    assert damage.shape == rerun.shape == (len(factors), 5, 6)
    for j in range(6):
        # WireStressField.at_fibre arithmetic for the same fibre.
        history = (mean + radial * normal[j] + tangential * lateral[j])[:, start:]
        for q, factor in enumerate(factors):
            expected, flags = miner_rows(history, curve, factor)
            assert np.array_equal(damage[q, :, j], expected)  # bitwise
            assert np.array_equal(rerun[q, :, j], flags)


def test_fibre_damage_leaves_non_finite_fibres_to_the_reference():
    mean, radial, tangential = _field()
    normal, lateral = _offsets(4)
    radial[1, 50] = math.nan
    _, rerun = fibre_miner_damage(
        mean, radial, tangential, normal, lateral, CURVES[0], [1.0, 2.0]
    )
    assert rerun[:, 1, :].all()  # NaN times any offset
    assert not rerun[:, [0, 2, 3, 4], :].any()
    radial[1, 50] = 0.0
    tangential[3, 10] = math.inf  # times the zero lateral offset of fibre 0: NaN
    _, rerun = fibre_miner_damage(
        mean, radial, tangential, normal, lateral, CURVES[0], [1.0]
    )
    # An infinite range has a zero life, which stops the kernel: every entry
    # is then left to the reference, as for miner_rows.
    assert rerun[:, 3, :].all()


def test_fibre_damage_rejects_malformed_input():
    mean, radial, tangential = _field()
    normal, lateral = _offsets(4)
    with pytest.raises(ValueError, match="two-dimensional"):
        fibre_miner_damage(
            mean, radial[:2], tangential, normal, lateral, CURVES[0], [1]
        )
    with pytest.raises(ValueError, match="offset"):
        fibre_miner_damage(
            mean, radial, tangential, normal, lateral[:2], CURVES[0], [1]
        )
    with pytest.raises(ValueError, match="factors"):
        fibre_miner_damage(mean, radial, tangential, normal, lateral, CURVES[0], [])
    with pytest.raises(ValueError, match="two samples"):
        fibre_miner_damage(
            mean, radial, tangential, normal, lateral, CURVES[0], [1], start=239
        )


def _spec(n_wires=8, **extra):
    return HelicalLayerSpec(
        n_wires=n_wires,
        r_helix_m=0.05,
        lay_angle_rad=0.3,
        E_wire_Pa=200e9,
        A_wire_m2=math.pi * 0.002**2,
        xi=0.8,
        mu=0.15,
        F_n_axi_N=2e3,
        **extra,
    )


def _history(n=260, seed=4):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 120, n)
    kx = 0.02 * np.sin(2 * np.pi * 0.07 * t) + 0.003 * rng.standard_normal(n)
    ky = 0.01 * np.cos(2 * np.pi * 0.043 * t) + 0.002 * rng.standard_normal(n)
    return kx, ky, 2e5 + 4e4 * np.sin(np.linspace(0, 9, n))


def _assert_same(a, b):
    for name in (
        "D_bulk_per_wire",
        "D_term_per_wire",
        "D_total_per_wire",
        "D_bulk_per_fibre",
        "fibre_critical_deg_per_wire",
        "D_bulk_with_termination_per_fibre",
        "termination_fibre_critical_deg_per_wire",
    ):
        left, right = getattr(a, name), getattr(b, name)
        assert left.keys() == right.keys()
        for key in left:
            assert np.array_equal(left[key], right[key]), (name, key)
    assert a.critical_layer == b.critical_layer
    assert a.fibre_critical_deg == b.fibre_critical_deg


@pytest.mark.parametrize(
    "options",
    [
        {"count_from_sample": 40},
        {"wire_positions": 3},
        {"count_from_sample": 7, "wire_positions": [0.0, 200.0]},
    ],
    ids=["lead_in", "positions", "both"],
)
def test_compute_fatigue_fused_counter_equals_the_reference_loop(options):
    kx, ky, tension = _history()
    layer = FatigueLayer(
        spec=_spec(),
        sn_bulk=CURVES[1],
        sigma_UTS_Pa=1e9,
        name="armour",
        SCF=1.4,
        termination=True,
    )
    layers = [layer, replace(layer, name="plain", termination=False)]
    runs = [
        compute_fatigue(
            layers,
            kx,
            ky,
            tension_N=tension,
            EA_full_N=4e8,
            n_fibres=7,
            rainflow_counter=counter,
            **options,
        )
        for counter in ("reference", "compiled")
    ]
    _assert_same(*runs)
    assert runs[0].D_term_per_wire["armour"].max() > 0


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
def test_fused_counter_raises_the_reference_error():
    kx, ky, _ = _history(60)
    layer = FatigueLayer(
        spec=_spec(4),
        sn_bulk=SNcurve(name="steep", stress_unit="MPa", m1=3.0, log_C1=12.0),
        sigma_UTS_Pa=1e9,
        SCF=1e300,
        termination=True,
    )
    for counter in ("reference", "compiled"):
        with pytest.raises(CableInputError, match="floating-point range"):
            compute_fatigue([layer], kx, ky, n_fibres=4, rainflow_counter=counter)


def _field_pair(monkeypatch, layer, kx, ky, **kwargs):
    monkeypatch.setattr(fatigue, "_FAST_FIELD", True)
    fast = wire_stress_field(layer, kx, ky, **kwargs)
    monkeypatch.setattr(fatigue, "_FAST_FIELD", False)
    plain = wire_stress_field(layer, kx, ky, **kwargs)
    return fast, plain


def _assert_close_fields(fast, plain):
    # Same IEEE operations in the same order (bitwise equal where checked);
    # compared to 1e-12 of the field scale across platforms.
    for name in ("mean_Pa", "radial_Pa", "tangential_Pa"):
        a, b = getattr(fast, name), getattr(plain, name)
        scale = max(float(np.abs(b).max()), 1.0)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12 * scale, err_msg=name)


def test_assembled_field_equals_the_numpy_expressions(monkeypatch):
    kx, ky, tension = _history()
    layer = FatigueLayer(spec=_spec(), sn_bulk=CURVES[0], sigma_UTS_Pa=1e9)
    modulus = 200e9 * (1 + 0.01 * np.sin(np.linspace(0, 5, kx.size)))
    for kwargs in (
        {"tension_N": tension, "EA_full_N": 4e8},
        {
            "tension_N": tension,
            "EA_full_N": 4e8,
            "thermal_stress_Pa": 5e6 * np.cos(np.linspace(0, 3, kx.size)),
            "wire_modulus_Pa": modulus,
            "wires": np.array([1, 4, 6]),
        },
        {
            "normal_stress_Pa": 9e7 + 1e7 * np.sin(np.linspace(0, 7, kx.size)),
            "keep_friction": True,
        },
        {
            "termination": TerminationSpec(
                distance_m=0.5,
                segments_per_pitch=8,
                max_increment=2.0,
                reach_lengths=1.0,
            )
        },
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fast, plain = _field_pair(monkeypatch, layer, kx, ky, **kwargs)
        _assert_close_fields(fast, plain)


def test_assembly_reproduces_local_bending_at_both_fibres():
    kx, ky, _ = _history(50)
    phi = np.linspace(0, 2 * math.pi, 5, endpoint=False)
    friction = np.zeros((5, kx.size))
    zeros = np.zeros(kx.size)
    mean, radial, tangential = assemble_wire_field(
        kx, ky, phi, 200e9, 0.004, 0.006, 0.3, zeros, friction, zeros
    )
    radial_ref = local_bending_fibre_stress(kx, ky, phi[:, None], 200e9, 0.004, 0.3)
    edge_ref = local_bending_fibre_stress(
        kx, ky, phi[:, None], 200e9, 0.006, 0.3, math.pi / 2
    )
    scale = float(np.abs(radial_ref).max())
    np.testing.assert_allclose(radial, radial_ref, rtol=0, atol=1e-12 * scale)
    np.testing.assert_allclose(tangential, edge_ref, rtol=0, atol=1e-12 * scale)
    np.testing.assert_allclose(mean, 0.0, rtol=0, atol=1e-12 * scale)


def test_benchmark_script_reports_cases_and_differences(tmp_path, monkeypatch):
    first, second = tmp_path / "a", tmp_path / "b"
    for output, extra in ((first, []), (second, ["--reference", str(first)])):
        monkeypatch.setattr(
            benchmark.sys,
            "argv",
            [
                "benchmark",
                "--output",
                str(output),
                "--steps",
                "500",
                "--repeat",
                "1",
                "--cases",
                "direction_tied_12pos_f4",
                *extra,
            ],
        )
        assert benchmark.main() == 0
    report = json.loads((second / "throughput.json").read_text(encoding="utf-8"))
    (row,) = report["cases"]
    assert row["case"] == "direction_tied_12pos_f4"
    assert row["series"] == 2 * 12 * 4  # two armour layers, 12 wires, 4 fibres
    assert len(row["seconds"]) == 1 and row["ns_per_series_step"] > 0
    differences = report["differences_from_reference"]
    assert differences and all(d["bitwise_equal"] for d in differences.values())


def test_benchmark_batch_writes_readable_runs():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cable = cx.load_cable(cx.iea_task49_66kv_yaml_path())
        batch = benchmark.time_batch(cable, [1], runs=2, steps=41, dt=0.18)
    assert batch["annual_damage_independent_of_workers"]
    assert batch["rows"][0]["workers"] == 1 and batch["rows"][0]["efficiency"] == 1.0
