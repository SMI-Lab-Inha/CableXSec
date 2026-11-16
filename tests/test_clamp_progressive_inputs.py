# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Clamp layer and progressive friction: NumPy fallback, checks and closed forms."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics import clamp_layer as cl
from cablexsec.mechanics.progressive_friction import (
    progressive_wire_friction_stress,
    progressive_wire_history,
)


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _drive(samples=12, segments=48):
    phi = np.linspace(0.0, 4 * 2 * math.pi, segments)
    path = np.concatenate(
        [np.linspace(0, 3, samples // 2 + 1)[1:], np.linspace(3, -3, samples // 2)]
    )
    return np.outer(path, np.cos(phi)), np.cos(phi), float(phi[1] - phi[0])


def test_field_numpy_fallback_matches_the_kernel():
    drive, c_hat, h = _drive()
    m = c_hat.size
    cap = np.full(m, 1.2)
    counts = np.full(drive.shape[0], 2, dtype=np.int64)
    rho = np.linspace(1.0, 0.9, drive.shape[0])
    index = np.array([0, m // 2, m - 1], dtype=np.int64)
    outs = [np.zeros((drive.shape[0], index.size)) for _ in range(4)]
    d1 = cl._field_kernel(
        drive, c_hat, cap, counts, rho, h, index, outs[0], outs[1], 0.0
    )
    d2 = cl._field_numpy(
        drive, c_hat, cap, counts, rho, h, index, outs[2], outs[3], 0.0
    )
    np.testing.assert_allclose(outs[0], outs[2], atol=1e-10)
    np.testing.assert_allclose(outs[1], outs[3], atol=1e-12)
    assert d1 == pytest.approx(d2, rel=1e-9)


def test_chain_projection_without_compiled_kernels(monkeypatch):
    y = np.array([0.4, -1.5, 2.0, 0.1])
    expected = cl.chain_projection_reference(y, 1.0)
    monkeypatch.setattr(cl, "HAVE_NUMBA", False)
    np.testing.assert_allclose(cl.chain_projection(y, 1.0), expected, atol=1e-12)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"drive": np.zeros(5)}, "finite \\(samples, segments\\)"),
        ({"c_hat": np.zeros(3)}, "c_hat must match"),
        ({"h": 0.0}, "segment length"),
        ({"max_increment": 0.0}, "max_increment"),
        ({"rho": -np.ones(12)}, "rho must be finite"),
        ({"cap_hat": np.ones(3)}, "cap_hat must match"),
        ({"index": np.array([100])}, "index out of range"),
        ({"substeps": np.zeros(12, dtype=np.int64)}, "substeps"),
    ],
)
def test_field_response_checks(changes, match):
    drive, c_hat, h = _drive()
    args = {"drive": drive, "c_hat": c_hat, "h": h}
    args.update(changes)
    with pytest.raises(CableInputError, match=match):
        cl.anchored_wire_field_response(**args)


def test_boundary_layer_length_in_radians_and_metres():
    radians = cl.clamp_boundary_layer_length(2.0)
    assert float(radians) == pytest.approx(math.sqrt(math.pi))
    metres = cl.clamp_boundary_layer_length(2.0, pitch_m=2 * math.pi)
    assert float(metres) == pytest.approx(float(radians))


@pytest.fixture(scope="module")
def spec():
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    return cx.cable_to_fatigue_layers(cable, contact_tension_N=2e4)[0].spec


K = 0.05 * np.sin(np.linspace(0.0, 4.0 * math.pi, 41))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"progressive": "slow"}, "ProgressiveFriction"),
        ({"kappa_y": K[:-1]}, "matching nonempty"),
        ({"initial_bending_angle": math.inf}, "initial bending angle"),
        ({"contact_pressure_Pa": np.ones(3)}, "match"),
        (
            {"contact_pressure_Pa": -np.ones(K.size)},
            "non-negative|nonnegative|positive",
        ),
    ],
)
def test_progressive_history_checks(spec, kwargs, match):
    args = {"kappa_x": K, "kappa_y": 0 * K}
    args.update(kwargs)
    with pytest.raises(CableInputError, match=match):
        progressive_wire_history(spec, **args)


def test_progressive_stress_checks(spec):
    with pytest.raises(CableInputError):
        progressive_wire_friction_stress(spec, K, K[:-1])
    with pytest.raises(CableInputError):
        progressive_wire_friction_stress(spec, K * np.inf, 0 * K)
