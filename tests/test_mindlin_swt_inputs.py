# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Mindlin contact slip maps and SWT fretting life: input checks and edges."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from cablexsec.analysis.fretting_swt import (
    partial_slip_surface_stress,
    swt_life,
    swt_slip_cycle_damage,
)
from cablexsec.exceptions import CableInputError
from cablexsec.materials.sn_curves import get_swt_life
from cablexsec.mechanics.mindlin_contact import (
    slip_increment,
    transverse_geodesic_factor,
    transverse_map_or_warn,
)

COPPER = get_swt_life("copper_poon2022")


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((math.nan, "geodesic"), "transverse_ratio"),
        ((0.5, "spiral"), "transverse"),
        ((1.5, "geodesic"), "geodesic"),
    ],
)
def test_transverse_map_checks(args, match):
    with pytest.raises(CableInputError, match=match):
        transverse_map_or_warn(*args)


def test_geodesic_slip_needs_a_lay_angle():
    with pytest.raises(CableInputError, match="non-zero helical lay angle"):
        transverse_geodesic_factor(0.05, 0.0)
    with pytest.raises(CableInputError, match="needs lay_angle_rad"):
        slip_increment(
            0.01, 0.0, 0.3, 1e-3, transverse_ratio=0.5, transverse_map="geodesic"
        )
    with pytest.raises(CableInputError):
        slip_increment(
            0.01,
            0.0,
            0.3,
            1e-3,
            transverse_ratio=0.5,
            transverse_map="geodesic",
            lay_angle_rad=math.pi / 2,
        )


def test_swt_life_and_surface_stress_inputs():
    assert swt_life(5e8, COPPER) > 0
    with pytest.raises(CableInputError, match="floating-point range"):
        swt_life(1e-300, COPPER)
    with pytest.raises(CableInputError, match="finite"):
        partial_slip_surface_stress(math.nan, 0.5, 0.3)


def test_slip_cycle_damage_inputs():
    q = 50.0 * np.sin(np.linspace(0.0, 6.0 * math.pi, 61))
    slipping = np.abs(q) > 45.0
    args = (100.0, 4e8, 0.5, 1.2e11, 0.34, COPPER)  # load, p0, mu, E, nu, law
    with pytest.raises(CableInputError, match="real"):
        swt_slip_cycle_damage(q.astype(complex), slipping, *args)
    with pytest.raises(CableInputError, match="finite 1-D"):
        swt_slip_cycle_damage(q[None, :], slipping, *args)
    with pytest.raises(CableInputError, match="contact inputs must be finite"):
        swt_slip_cycle_damage(q, slipping, math.nan, *args[1:])
    result = swt_slip_cycle_damage(q, slipping, *args, partial_slip="cattaneo_mindlin")
    assert np.isfinite(result.damage) and result.damage >= 0.0
