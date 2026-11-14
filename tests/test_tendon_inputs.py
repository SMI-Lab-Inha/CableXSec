# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Rectangular tendons: section and transverse-slip checks and round branches."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.tendon_bending import (
    LateralRelief,
    TransverseSlipModel,
    dong_anchor_tension,
    geodesic_slip_ratio,
    saevik_transverse_fraction,
)
from cablexsec.mechanics.tendon_section import (
    WireSection,
    rectangle_torsion_constant,
    round_section,
    rounded_rectangle_properties,
)

STRIP = WireSection(shape="rectangular", width_m=0.0125, thickness_m=0.005)
A = math.radians(35.0)


def test_rectangle_helpers():
    with pytest.raises(CableInputError, match="finite and positive"):
        rectangle_torsion_constant(0.0, 1.0)
    with pytest.raises(CableInputError, match="positive"):
        rounded_rectangle_properties(0.0, 0.005)
    with pytest.raises(CableInputError, match="half the smaller side"):
        rounded_rectangle_properties(0.0125, 0.005, 0.003)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"shape": "hexagonal"}, "wire shape"),
        ({"shape": "round"}, "positive diameter"),
        ({"shape": "rectangular", "width_m": 0.01}, "positive thickness"),
        (
            {
                "shape": "rectangular",
                "width_m": 0.01,
                "thickness_m": 0.005,
                "corner_radius_m": -1e-4,
            },
            "corner_radius_m",
        ),
        ({"shape": "round", "diameter_m": 0.004, "width_m": 0.01}, "round"),
        (
            {
                "shape": "rectangular",
                "width_m": 0.01,
                "thickness_m": 0.005,
                "area_m2": 1e-4,
            },
            "computes its properties",
        ),
        (
            {"shape": "profiled", "width_m": 0.01, "thickness_m": 0.005},
            "profiled wire needs",
        ),
        (
            {
                "shape": "profiled",
                "width_m": 0.01,
                "thickness_m": 0.005,
                "area_m2": 1e-4,
                "I_normal_m4": 1e-10,
                "I_lateral_m4": 4e-10,
                "J_m4": 3e-10,
            },
            "exceeds its width x thickness",
        ),
        (
            {
                "shape": "rectangular",
                "width_m": 0.01,
                "thickness_m": 0.005,
                "corner_radius_m": 0.003,
            },
            "half the smaller side",
        ),
    ],
)
def test_section_checks(kwargs, match):
    with pytest.raises(CableInputError, match=match):
        WireSection(**kwargs)


def test_round_section_branches():
    wire = round_section(0.004)
    assert wire.equivalent_diameter_m == 0.004
    assert wire.radial_thickness_m == 0.004 and wire.surface_width_m == 0.004
    normal, lateral = wire.corner_offsets()
    assert np.all(np.isfinite(normal)) and np.all(np.isfinite(lateral))
    np.testing.assert_allclose(wire.max_bending_stress(3.0, 4.0), 5.0)


def test_transverse_slip_checks():
    with pytest.raises(CableInputError, match="0 < lay angle < 90"):
        geodesic_slip_ratio(0.0)
    assert saevik_transverse_fraction(0.1, 1e4, 2e11, 1e-10, 0.1, A, 0.0) == 0.0
    base = (STRIP, 2.1e11, 8e10, 0.1, A, 2e4, 100.0)
    for index, value, match in (
        (1, 0.0, "E_Pa must be finite and positive"),
        (4, 0.0, "lay angle"),
        (6, -1.0, "non-negative"),
    ):
        args = list(base)
        args[index] = value
        with pytest.raises(CableInputError, match=match):
            TransverseSlipModel(*args)
    with pytest.raises(CableInputError, match="sense"):
        TransverseSlipModel(*base, sense=0)
    model = TransverseSlipModel(*base)
    with pytest.raises(CableInputError, match="anchor_layer_m"):
        LateralRelief(model, anchor_layer_m=-1.0)


def test_holding_load_gradient_and_zero_capacity():
    model = TransverseSlipModel(STRIP, 2.1e11, 8e10, 0.1, A, 2e4, 100.0)
    kappa = np.array([0.0, 0.1, 0.2])
    with_gradient = model.holding_load(kappa, gradient_normal=np.array([0.0, 0.5, 1.0]))
    assert with_gradient.shape == kappa.shape
    free = TransverseSlipModel(STRIP, 2.1e11, 8e10, 0.1, A, 2e4, 0.0)
    theta = np.linspace(0.0, 2 * math.pi, 5)
    np.testing.assert_array_equal(free.demand_ratio(0.1, 0.0, theta), 1.0)
    factor = LateralRelief(free).factor(theta, 0.1)
    assert np.all(np.isfinite(factor))


def test_dong_anchor_inputs():
    with pytest.raises(CableInputError, match="positive tension"):
        dong_anchor_tension(STRIP, 2.1e11, 8e10, 0.1, A, -1e-3, 0.1)
    with pytest.raises(CableInputError, match="friction"):
        dong_anchor_tension(STRIP, 2.1e11, 8e10, 0.1, A, 1e-3, 0.0)
