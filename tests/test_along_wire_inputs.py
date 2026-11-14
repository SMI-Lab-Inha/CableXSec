# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Along-wire chains: layer, builder and history checks, contact shape."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics import along_wire as aw
from cablexsec.mechanics.strand_slip import ConductorRing, StrandSlipParameters


def _layer(**kw):
    base = {
        "name": "unit",
        "n_wires": 2,
        "r_helix_m": 0.05,
        "lay_angle_rad": math.radians(30.0),
        "kappa_c_1_m": 1.0,
        "k_hat": 0.26,
        "lam": 0.1,
        "force_unit_N": 1.0,
        "wire_area_m2": 1.0,
        "ea_N": 1.0,
    }
    base.update(kw)
    return aw.WireChainLayer(**base)


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"n_wires": 0}, "n_wires"),
        ({"n_wires": True}, "n_wires"),
        ({"r_helix_m": 0.0}, "r_helix_m must be finite and positive"),
        ({"lay_angle_rad": 0.0}, "lay_angle_rad"),
        ({"lam": -0.1}, "lam"),
        ({"sense": 0}, "sense"),
        ({"stick": "glued"}, "stick"),
        ({"phase0_rad": math.inf}, "phase0_rad"),
        ({"shape_g": (1.0,)}, "shape_g"),
    ],
)
def test_layer_checks(kw, match):
    with pytest.raises(CableInputError, match=match):
        _layer(**kw)


def test_contact_shape_interpolation():
    uniform = _layer()
    np.testing.assert_array_equal(uniform.shape_at(np.array([0.0, 1.0])), 1.0)
    shaped = _layer(shape_g=(1.0, 3.0))
    # Wires at 0 and pi; halfway between, the shape is the mean.
    np.testing.assert_allclose(
        shaped.shape_at(np.array([0.0, math.pi / 2, math.pi, 2 * math.pi])),
        [1.0, 2.0, 3.0, 1.0],
    )


def test_builders_refuse_unsupported_inputs():
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    spec = cx.cable_to_fatigue_layers(cable, contact_tension_N=2e4)[0].spec
    with pytest.raises(CableInputError):
        aw.WireChainLayer.from_layer_spec(replace(spec, mu=0.0))
    params = StrandSlipParameters(0.0, (0.04, 0.01), (2e-4,), 1e3)
    with pytest.raises(CableInputError, match="centre wire"):
        aw.WireChainLayer.from_conductor_ring(
            ConductorRing(1, 0.003, 0.0, 0.0), 1e11, params
        )
    ring = ConductorRing.from_pitch(6, 0.003, 0.003, 0.2, 1)
    two_orders = StrandSlipParameters(0.0, (0.04, 0.02, 0.01), (2e-4, 3e-4), 1e3)
    with pytest.raises(CableInputError, match="one slipping order"):
        aw.WireChainLayer.from_conductor_ring(ring, 1e11, two_orders)
    no_capacity = StrandSlipParameters(0.0, (0.01, 0.04), (2e-4,), 1e3)
    with pytest.raises(CableInputError, match="no friction capacity"):
        aw.WireChainLayer.from_conductor_ring(ring, 1e11, no_capacity)


@pytest.mark.parametrize(
    ("args", "kwargs", "match"),
    [
        ((1.0, 0.5), {}, "x_end_m > x_start_m"),
        ((0.0, 1.0), {"ends": ("anchored", "glued")}, "ends"),
        ((0.0, 1.0), {"segments_per_pitch": 4}, "segments_per_pitch"),
        ((0.0, 1.0), {"wires": [5]}, "wire indices"),
    ],
)
def test_chain_build_checks(args, kwargs, match):
    with pytest.raises(CableInputError, match=match):
        aw.WireChain.build(_layer(), *args, **kwargs)


@pytest.mark.parametrize(
    ("x", "kx", "kwargs", "match"),
    [
        (np.array([0.5, 0.2]), np.zeros((2, 3)), {}, "strictly increasing"),
        (np.array([0.0, 0.5]), np.zeros((3, 3)), {}, "shape \\(positions, samples\\)"),
        (
            np.array([0.0, 0.5]),
            np.zeros((2, 3)),
            {"kappa_y": np.zeros((2, 2))},
            "match kappa_x",
        ),
        (
            np.array([0.0, 0.5]),
            np.zeros((2, 3)),
            {"contact_ratio": -np.ones(3)},
            "contact",
        ),
        (
            np.array([0.0, 0.5]),
            np.zeros((2, 3)),
            {"max_increment": 0.0},
            "max_increment",
        ),
        (
            np.array([0.0, 0.5]),
            np.zeros((2, 3)),
            {"relief": "lateral"},
            "LateralRelief",
        ),
        (
            np.array([0.0, 0.5]),
            np.zeros((2, 3)),
            {"record_x_m": np.array([])},
            "record_x_m",
        ),
    ],
)
def test_history_checks(x, kx, kwargs, match):
    with pytest.raises(CableInputError, match=match):
        aw.along_wire_history(_layer(), x, kx, segments_per_pitch=16, **kwargs)
