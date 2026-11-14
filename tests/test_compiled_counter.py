# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""compute_fatigue's compiled rainflow counter must equal the reference exactly.

Also checks the optional-numba mechanism: the reference paths can be forced
(module attributes or ``CABLEXSEC_DISABLE_NUMBA``) and give the same damage,
and project engine records store the numba version.
"""

import importlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import cablexsec as cx
import cablexsec.analysis.fatigue as fatigue
from cablexsec._numba import DISABLE_ENV, kernels_compiled
from cablexsec.analysis.fast_rainflow import miner_rows
from cablexsec.analysis.fatigue import (
    FatigueLayer,
    compute_fatigue,
    miner_damage,
)
from cablexsec.exceptions import CableInputError
from cablexsec.materials.sn_curves import SN_CURVES, SNcurve
from cablexsec.mechanics.mkappa import HelicalLayerSpec

ROOT = Path(__file__).resolve().parents[1]

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


def histories():
    rng = np.random.default_rng(17)
    return [
        rng.normal(0, 30e6, 3000),
        np.round(rng.normal(0, 30e6, 3000) / 5e6) * 5e6,  # plateaus and ties
        np.cumsum(rng.normal(0, 1e6, 3000)),
        np.linspace(0, 50e6, 100),  # monotonic
        np.linspace(50e6, -5e6, 7),  # monotonic falling
        np.full(50, 7e6),  # constant
        np.array([0.0, 40e6]),  # two samples
        np.array([3e6, 3e6]),  # two equal samples
        np.array([1e6, 1e6, 3e6, 3e6, -2e6, -2e6, 4e6, 4e6]),  # plateau reversals
        np.r_[
            np.zeros(10), 20e6 * np.sin(np.linspace(0, 40 * np.pi, 2000)), np.zeros(10)
        ],
        np.array([0.0, 2e6, 0.0]),  # below the threshold of the test curve
    ]


@pytest.mark.parametrize("curve", CURVES, ids=lambda c: c.name)
@pytest.mark.parametrize("factor", [1.0, 1.7, 0.3])
def test_miner_rows_is_bitwise_equal_to_miner_damage(curve, factor):
    for history in histories():
        damage, rerun = miner_rows(history[None, :], curve, factor)
        assert not rerun[0]
        reference = miner_damage(history, curve, 1e9, stress_range_factor=factor)
        assert damage[0] == reference  # bitwise, not approximately


def test_constant_and_two_point_histories():
    curve = CURVES[0]
    damage, _ = miner_rows(np.full((2, 40), 5e6), curve)
    assert np.array_equal(damage, [0.0, 0.0])
    two = np.array([[0.0, 40e6], [40e6, 0.0]])
    damage, _ = miner_rows(two, curve)
    expected = 0.5 / curve.cycles_to_failure(40.0)
    assert damage[0] == damage[1] == miner_damage(two[0], curve, 1e9) == expected


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_nonfinite_history_is_rejected_on_both_paths(bad):
    history = np.array([[0.0, 10e6, bad, 0.0]])
    for compiled in (False, True):
        with pytest.raises(CableInputError, match="finite"):
            fatigue._miner_damage_rows(history, CURVES[0], 1e9, 1.0, compiled)


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
def test_reference_errors_are_reproduced_for_flagged_rows():
    # A tiny range gives an S-N life beyond floating-point range: the
    # reference raises, and the compiled counter must defer to it.
    curve = SNcurve(name="steep", stress_unit="MPa", m1=3.0, log_C1=12.0)
    tiny = np.array([[0.0, 1e-300], [0.0, 10e6]])
    _, rerun = miner_rows(tiny, curve)
    # Compiled arithmetic flags the row; interpreted kernels flag every row.
    assert rerun[0] and (not rerun[1] or not kernels_compiled())
    for compiled in (False, True):
        with pytest.raises(CableInputError, match="floating-point range"):
            fatigue._miner_damage_rows(tiny, curve, 1e9, 1.0, compiled)
    huge = np.array([[0.0, 1e300, 0.0]])
    for compiled in (False, True):
        with pytest.raises(CableInputError, match="floating-point range"):
            fatigue._miner_damage_rows(huge, curve, 1e9, 1e10, compiled)


def test_unknown_counter_is_rejected():
    with pytest.raises(CableInputError, match="rainflow_counter"):
        fatigue._use_compiled_counter("fast")


def _assert_results_identical(a, b):
    for name in (
        "D_bulk_per_wire",
        "D_fretting_per_wire",
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
    assert a.D_fretting_fraction == b.D_fretting_fraction
    assert a.governing == b.governing


def _curvature(n=400, seed=3):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 200, n)
    kx = 0.03 * np.sin(2 * np.pi * 0.05 * t) + 0.004 * rng.standard_normal(n)
    ky = 0.02 * np.cos(2 * np.pi * 0.031 * t) + 0.002 * rng.standard_normal(n)
    return kx, ky


@pytest.mark.parametrize(
    "path",
    [cx.menard2023_yaml_path, cx.export_three_core_yaml_path],
    ids=["menard2023", "export_three_core"],
)
def test_public_compute_fatigue_counters_agree_bitwise(path):
    cable = cx.load_cable(path(), geometry_override=None)
    kx, ky = _curvature()
    tension = 2e5 + 5e4 * np.sin(np.linspace(0, 12, kx.size))
    runs = {
        counter: cx.compute_fatigue(
            cable, kx, ky, tension_N=tension, n_fibres=8, rainflow_counter=counter
        )
        for counter in ("reference", "compiled", "auto")
    }
    _assert_results_identical(runs["reference"], runs["compiled"])
    _assert_results_identical(runs["reference"], runs["auto"])


def test_termination_amplification_counters_agree_bitwise():
    spec = HelicalLayerSpec(
        n_wires=6,
        r_helix_m=0.05,
        lay_angle_rad=0.2,
        E_wire_Pa=200e9,
        A_wire_m2=math.pi * 0.002**2,
        xi=1,
        mu=0.1,
        F_n_axi_N=1e3,
    )
    layer = FatigueLayer(
        spec=spec,
        sn_bulk=CURVES[2],
        sigma_UTS_Pa=1e9,
        name="armour",
        SCF=1.6,
        termination=True,
    )
    kx, ky = _curvature(300, 9)
    normal = 80e6 + 30e6 * np.sin(np.linspace(0, 30, kx.size))
    runs = [
        compute_fatigue(
            [layer, replace(layer, name="plain", termination=False)],
            kx,
            ky,
            normal_stress_Pa={"armour": normal, "plain": normal},
            n_fibres=8,
            rainflow_counter=counter,
        )
        for counter in ("reference", "compiled")
    ]
    _assert_results_identical(*runs)
    assert runs[0].D_term_per_wire["armour"].max() > 0


def test_forced_numpy_fallback_matches_compiled_path(monkeypatch):
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    kx, ky = _curvature(300, 5)
    compiled = cx.compute_fatigue(cable, kx, ky, n_fibres=8)
    monkeypatch.setattr(fatigue, "_FAST_SLIP", False)
    monkeypatch.setattr(fatigue, "_FAST_RAINFLOW", False)
    fallback = cx.compute_fatigue(cable, kx, ky, n_fibres=8)
    # Rainflow counting is bitwise equal; the friction-stress kernel agrees
    # with wire_slip_history to round-off (tests/test_fast_slip.py), which
    # Miner sums amplify by at most the S-N slope.
    for name, value in compiled.D_total_per_wire.items():
        np.testing.assert_allclose(
            fallback.D_total_per_wire[name], value, rtol=1e-8, atol=0
        )
    assert compiled.critical_layer == fallback.critical_layer


_SUBPROCESS = """
import json, sys
import numpy as np
import cablexsec as cx
from cablexsec import _numba
from cablexsec.analysis import fatigue
k = 0.02 * np.sin(np.linspace(0, 20 * np.pi, 300))
cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
res = cx.compute_fatigue(cable, k, 0.5 * k, n_fibres=8)
print(json.dumps({
    "have": _numba.HAVE_NUMBA,
    "slip": fatigue._FAST_SLIP,
    "rainflow": fatigue._FAST_RAINFLOW,
    "prange_is_range": _numba.prange is range,
    "damage": {n: v.tolist() for n, v in res.D_total_per_wire.items()},
}))
"""


def _run(disable):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), *filter(None, [env.get("PYTHONPATH")])]
    )
    env.pop(DISABLE_ENV, None)
    if disable:
        env[DISABLE_ENV] = "1"
    out = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_environment_variable_disables_compiled_kernels():
    disabled = _run(True)
    assert disabled["have"] is False
    assert disabled["slip"] is False and disabled["rainflow"] is False
    assert disabled["prange_is_range"] is True
    enabled = _run(False)
    # The parent may itself run with the variable set; compare installation.
    assert enabled["have"] is (importlib.util.find_spec("numba") is not None)
    for name, value in enabled["damage"].items():
        np.testing.assert_allclose(disabled["damage"][name], value, rtol=1e-8, atol=0)


def test_project_engine_records_numba_version(monkeypatch):
    from importlib.metadata import PackageNotFoundError

    import cablexsec.project as implementation

    original = implementation.version
    p = cx.Project("with kernels")
    try:
        expected = original("numba")
    except PackageNotFoundError:
        expected = None
    assert p.engine["numba"] == expected

    def without_numba(name):
        if name == "numba":
            raise PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(implementation, "version", without_numba)
    p = cx.Project("no kernels")
    assert p.engine["numba"] is None
    assert not p.environment_changed


def test_engine_records_without_numba_field_still_load():
    from cablexsec.project import _engine, _validate_engine

    engine = json.loads(_engine())
    _validate_engine(engine)
    del engine["numba"]
    _validate_engine(engine)
    engine["numba"] = 1
    with pytest.raises(CableInputError, match="numba"):
        _validate_engine(engine)
