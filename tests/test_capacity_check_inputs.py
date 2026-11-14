# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Capacity check of histories: envelope inputs, options and report items."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from test_cabledyn_inputs import _case

import cablexsec as cx
from cablexsec.analysis.capacity_check import (
    CapacityEnvelope,
    build_capacity_envelope,
    capacity_extremes,
    check_capacity,
    check_capacity_series,
    gumbel_fit,
)
from cablexsec.exceptions import CableInputError
from cablexsec.io.cabledyn import read_cabledyn_sections

T = np.linspace(0.0, 10.0, 101)


def _envelope(**kw):
    return CapacityEnvelope.from_points(
        [0.0, 5.0e4, 1.0e5], [0.5, 0.4, 0.3], source="synthetic test curve", **kw
    )


@pytest.fixture(scope="module")
def menard():
    return cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"tension_N": [0.0, 0.0], "kappa_capacity_1_m": [0.5, 0.4]},
            "increase strictly",
        ),
        (
            {"tension_N": [0.0, 1.0], "kappa_capacity_1_m": [0.5]},
            "nonnegative per tension",
        ),
        (
            {"tension_N": [0.0, 1.0], "kappa_capacity_1_m": [0.5, -0.1]},
            "nonnegative per tension",
        ),
        (
            {"tension_N": [0.0], "kappa_capacity_1_m": [0.5], "tension_kind": "wall"},
            "tension_kind",
        ),
        (
            {
                "tension_N": [0.0],
                "kappa_capacity_1_m": [0.5],
                "utilisation_factor": 1.2,
            },
            "utilisation_factor",
        ),
        ({"tension_N": [0.0], "kappa_capacity_1_m": [0.5], "mbr_m": 0.0}, "mbr_m"),
    ],
)
def test_envelope_checks(kwargs, match):
    base = {"utilisation_factor": 1.0, "tension_kind": "effective"}
    base["tension_utilisation"] = np.zeros(len(kwargs["tension_N"]))
    with pytest.raises(CableInputError, match=match):
        CapacityEnvelope(**{**base, **kwargs})


def test_utilisation_shapes_must_match():
    with pytest.raises(CableInputError, match="must match"):
        _envelope().utilisation(np.zeros(3), np.zeros(4))


def test_single_level_curve_extrapolates_through_the_origin(menard):
    # One level: beyond it the utilisation rises along the secant from zero.
    env = build_capacity_envelope(menard, [0.0, 1e5], levels=[1.0])
    k1 = float(env.kappa_levels_1_m[0, 0])
    u, _, counts = env.utilisation(np.zeros(2), np.array([k1, 2.0 * k1]))
    np.testing.assert_allclose(u, [1.0, 2.0], rtol=1e-12)
    assert counts["extrapolated_beyond_levels"] == 1


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tension_N": [0.0, 0.0]}, "increase strictly"),
        ({"levels": [0.5, 0.2]}, "positive and ascending"),
        ({"levels": [0.0, 0.5]}, "positive and ascending"),
        ({"levels": [0.5, 1.5]}, "beyond first yield"),
        ({"mbr_m": 3.0}, "mbr_source"),
    ],
)
def test_build_checks(menard, kwargs, match):
    kwargs.setdefault("tension_N", [0.0, 1e5])
    with pytest.raises(CableInputError, match=match):
        build_capacity_envelope(menard, **kwargs)


def test_compression_governs_by_radial_and_lateral_buckling(menard):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = build_capacity_envelope(
            menard, [-6e5, -2e5, 0.0], levels=[0.5, 1.0], mbr_m=3.0, mbr_source="test"
        )
    assert env.governing == ("radial_buckling", "lateral_buckling", "wire_yield")
    assert env.kappa_capacity_1_m[0] == 0.0
    assert 0.0 < env.kappa_capacity_1_m[1] < env.kappa_capacity_1_m[2]
    assert (env.mbr_m, env.mbr_source) == (3.0, "test")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"time_s": np.r_[T[:-1], T[-2]]}, "increase strictly"),
        ({"tension_N": np.zeros(5)}, "must match"),
        ({"mbr_m": 3.0}, "mbr_source"),
        ({"torque_Nm": np.zeros(5)}, "torque_Nm must match"),
        ({"blocks": 1}, "blocks"),
        ({"blocks": True}, "blocks"),
    ],
)
def test_history_checks(kwargs, match):
    args = {
        "time_s": T,
        "tension_N": np.full(T.size, 2e4),
        "kappa_x_1_m": 0.1 * np.sin(T),
    }
    args.update(kwargs)
    with pytest.raises(CableInputError, match=match):
        check_capacity(_envelope(), **args)


def test_torque_and_extrapolation_are_reported(menard):
    env = build_capacity_envelope(menard, [0.0, 1e5])
    kappa = 3.0 * float(np.max(env.kappa_levels_1_m[-1]))
    result = check_capacity(
        env,
        T,
        np.full(T.size, 2e4),
        kappa * np.sin(T),
        torque_Nm=50.0 * np.cos(T),
    )
    assert result.torque_max_Nm == pytest.approx(50.0)
    limitations = " ".join(result.provenance.limitations)
    assert "torque reported only" in limitations
    assert "beyond the highest level" in limitations


def test_gumbel_and_extremes_checks():
    with pytest.raises(CableInputError, match="at least two maxima"):
        gumbel_fit([1.0])
    with pytest.raises(CableInputError, match="fractile"):
        gumbel_fit([1.0, 2.0], fractile=1.0)
    with pytest.raises(CableInputError, match="duration_ratio"):
        gumbel_fit([1.0, 2.0], duration_ratio=0.5)
    one = check_capacity(_envelope(), T, np.full(T.size, 2e4), 0.1 * np.sin(T))
    with pytest.raises(CableInputError, match="at least two seeds"):
        capacity_extremes([one])


def test_series_of_a_cabledyn_section(tmp_path):
    root, deck = _case(tmp_path, nt=6)
    history = read_cabledyn_sections(root, deck, {"mid": 5.0}).histories["mid"]
    effective = check_capacity_series(_envelope(), history)
    assert effective.name == "mid"
    true_env = CapacityEnvelope.from_points(
        [0.0, 1e5], [0.5, 0.3], source="test", tension_kind="true"
    )
    with pytest.raises(CableInputError, match="external_area_m2"):
        check_capacity_series(true_env, history)
    true = check_capacity_series(true_env, history, external_area_m2=0.03)
    assert true.max_utilisation > 0
    with pytest.raises(
        CableInputError, match="LoadCaseSeries or a CableDynSectionHistory"
    ):
        check_capacity_series(_envelope(), {"time_s": T})  # type: ignore[arg-type]


def test_one_tension_envelope_and_non_finite_input():
    env = CapacityEnvelope.from_points([0.0], [0.5], source="one point")
    u, _, counts = env.utilisation(
        np.array([0.0, 3e4]), np.array([0.25, 0.25]), out_of_range="clamp"
    )
    np.testing.assert_allclose(u, [0.5, 0.5])
    assert counts["outside_tension_grid"] == 1
    with pytest.raises(CableInputError, match="finite one-dimensional"):
        env.utilisation(np.array([np.nan]), np.array([0.1]))


def test_polymer_crush_allowable_needs_its_source(menard):
    with pytest.raises(CableInputError, match="go together"):
        build_capacity_envelope(menard, [0.0], polymer_crush_allowable_Pa=5e6)
