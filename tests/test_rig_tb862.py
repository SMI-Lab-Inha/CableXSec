# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Rig protocols with a tension phase lag and constant-tension blocks; TB 862 lengths."""

from __future__ import annotations

import dataclasses
import math
import warnings

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.rig_test import (
    GaugeLocation,
    RigBlock,
    RigProtocol,
    RigSetup,
    component_pitches,
    simulate_rig_test,
    tb862_specimen_checks,
)
from cablexsec.exceptions import CableInputError

pytestmark = pytest.mark.analytical_sanity


@pytest.fixture(scope="module")
def menard():
    return cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)


@pytest.fixture(scope="module")
def t66():
    return cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)


def test_tension_lags_the_bending_by_the_given_phase() -> None:
    base = dict(  # noqa: C408
        tension_N=5e4,
        curvature_amplitude_1_m=0.05,
        tension_amplitude_N=1e4,
        n_cycles=2,
        samples_per_cycle=40,
    )
    in_phase = RigProtocol.cyclic(**base)
    default = RigProtocol.cyclic(**base, tension_phase_lag_deg=0.0)
    assert np.array_equal(in_phase.tension_N, default.tension_N)
    lagged = RigProtocol.cyclic(**base, tension_phase_lag_deg=90.0)
    assert np.array_equal(lagged.time_s, in_phase.time_s)
    assert np.array_equal(lagged.curvature_1_m, in_phase.curvature_1_m)
    cyc = lagged.time_s > 10.0
    phase = 2.0 * math.pi * (lagged.time_s[cyc] - 10.0) / 10.0
    assert lagged.tension_N[cyc] == pytest.approx(5e4 - 1e4 * np.cos(phase), abs=1e-6)
    # At the bending peak the tension is at its mean (quadrature).
    peak = int(np.argmax(lagged.curvature_1_m))
    assert lagged.tension_N[peak] == pytest.approx(5e4, abs=1e-6)
    # The ramp ends at the tension of the first cycle (no jump).
    n_ramp = int(np.sum(lagged.time_s <= 10.0))
    assert lagged.tension_N[n_ramp - 1] == pytest.approx(5e4 - 1e4, abs=1e-6)
    antiphase = RigProtocol.cyclic(**base, tension_phase_lag_deg=180.0)
    assert antiphase.tension_N[cyc] == pytest.approx(
        5e4 - 1e4 * np.sin(phase), abs=1e-6
    )
    with pytest.raises(ValueError):
        RigProtocol.cyclic(**base, tension_phase_lag_deg=math.nan)


def test_constant_tension_blocks_hold_their_tension() -> None:
    blocks = [RigBlock(80e3, 0.029, 2), RigBlock(110e3, 0.085, 1, mean=0.01)]
    p = RigProtocol.blocks(blocks, samples_per_cycle=20, period_s=10.0)
    assert len(p.block_times_s) == 2
    (s1, e1), (s2, e2) = p.block_times_s
    assert (s1, e1) == pytest.approx((10.0, 30.0))
    assert (s2, e2) == pytest.approx((40.0, 50.0))
    for (start, end), block in zip(p.block_times_s, blocks, strict=True):
        inside = (p.time_s > start) & (p.time_s <= end)
        assert np.all(p.tension_N[inside] == block.tension_N)
        bend = p.curvature_1_m[inside]
        assert bend.max() == pytest.approx(block.mean + block.amplitude, rel=1e-12)
        assert bend.min() == pytest.approx(block.mean - block.amplitude, rel=1e-12)
    # Linear transitions between the blocks, from zero before the first.
    between = (p.time_s > e1) & (p.time_s < s2)
    tau = (p.time_s[between] - e1) / 10.0
    assert p.tension_N[between] == pytest.approx(80e3 + 30e3 * tau, rel=1e-12)
    assert p.curvature_1_m[between] == pytest.approx(0.01 * tau, abs=1e-15)
    assert p.tension_N[0] == 0.0 and p.curvature_1_m[0] == 0.0
    # Ranges over the last cycle of the last block.
    assert p.retain_from_s == pytest.approx(e2 - 10.0)
    rotation = RigProtocol.blocks(blocks, quantity="end_rotation")
    assert rotation.curvature_1_m is None and rotation.end_rotation_rad is not None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"blocks": []}, "at least one"),
        ({"blocks": [RigBlock(1e4, 0.01)], "quantity": "strain"}, "quantity"),
        ({"blocks": [RigBlock(1e4, 0.01)], "transition_s": 0.0}, "transition"),
        ({"blocks": [(1e4, 0.01, 1)]}, "RigBlock instances"),
    ],
)
def test_block_programme_inputs_are_checked(kwargs, match) -> None:
    with pytest.raises(CableInputError, match=match):
        RigProtocol.blocks(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tension_N": -1.0, "amplitude": 0.01}, "non-negative"),
        ({"tension_N": 1.0, "amplitude": 0.01, "n_cycles": 0}, "at least 1"),
        ({"tension_N": 1.0, "amplitude": 0.01, "n_cycles": 1.5}, "integer"),
        ({"tension_N": 1.0, "amplitude": math.inf}, "finite"),
    ],
)
def test_rig_block_inputs_are_checked(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        RigBlock(**kwargs)


def test_block_times_must_lie_in_the_record() -> None:
    p = RigProtocol.blocks([RigBlock(1e4, 0.01)])
    with pytest.raises(CableInputError, match="block_times_s"):
        dataclasses.replace(p, block_times_s=((0.0, 1.0e6),))


def test_block_programme_runs_through_the_rig_simulation(menard) -> None:
    p = RigProtocol.blocks(
        [RigBlock(4e4, 0.02), RigBlock(6e4, 0.03)], samples_per_cycle=16
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = simulate_rig_test(
            menard,
            p,
            RigSetup(free_length_m=1.5, clamp="none"),
            gauges=[GaugeLocation("outer_armour", 90.0, 0.02)],
            mu_factors=(),
        )
    assert result.gauges[0].ranges["total"] > 0.0


def test_component_pitches_of_the_66kv_template(t66) -> None:
    pitches = component_pitches(t66)
    outer = t66.get_layer("outer_armour")
    expected = 2.0 * math.pi * outer.r_helix_m / math.tan(math.radians(10.0))
    assert pitches["outer_armour"] == pytest.approx(expected, rel=1e-12)
    assert pitches["core lay"] == pytest.approx(1.4542, rel=1e-12)
    assert set(pitches) >= {"inner_armour", "outer_armour", "core lay"}


def test_tb862_specimen_lengths(t66) -> None:
    pitches = component_pitches(t66)
    p_a = pitches["outer_armour"]
    p_max = max(pitches.values())
    setup = RigSetup(free_length_m=p_max + 0.1, clamp_setback_m=p_a - 0.01)
    loaded, ends = tb862_specimen_checks(t66, "fatigue_template", setup=setup)
    assert loaded.clause == ends.clause == "6.3.2"
    assert loaded.required_m == pytest.approx(p_max) and loaded.passed
    assert ends.required_m == pytest.approx(p_a) and not ends.passed
    assert not loaded.mandatory and not ends.mandatory
    top, bottom = tb862_specimen_checks(
        t66, "fatigue_bend_stiffener", top_distance_m=p_a, bottom_distance_m=0.5 * p_a
    )
    assert top.mandatory and top.passed and not bottom.passed
    (stiff,) = tb862_specimen_checks(t66, "bending_stiffness", sample_length_m=p_a)
    assert stiff.clause == "6.4.2.2" and stiff.passed
    (comp_bend,) = tb862_specimen_checks(
        t66, "compression_bending", sample_length_m=1.9 * p_max
    )
    assert comp_bend.required_m == pytest.approx(2.0 * p_max) and not comp_bend.passed
    (comp,) = tb862_specimen_checks(t66, "compression", sample_length_m=p_max)
    assert comp.clause == "6.6.1.2" and comp.mandatory and comp.passed


@pytest.mark.parametrize(
    ("test", "kwargs", "match"),
    [
        ("fatigue", {}, "test must be one of"),
        ("fatigue_template", {}, "loaded_length_m"),
        ("fatigue_bend_stiffener", {"top_distance_m": 1.0}, "bottom_distance_m"),
        ("compression", {"sample_length_m": -1.0}, "non-negative"),
    ],
)
def test_tb862_inputs_are_checked(t66, test, kwargs, match) -> None:
    with pytest.raises(CableInputError, match=match):
        tb862_specimen_checks(t66, test, **kwargs)
