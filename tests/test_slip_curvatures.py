# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Slip onset and full-slip curvature of each armour layer against tension."""

from __future__ import annotations

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.cable_adapters import cable_to_mkappa_specs, structural_armour_layers
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.papailiou import full_slip_curvature, slip_onset_curvature


@pytest.fixture(scope="module")
def menard():
    return cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)


def test_one_entry_per_structural_layer_inside_out(menard):
    result = cx.armour_slip_curvatures(menard, [0.0, 50e3])
    names = [layer.name for layer in structural_armour_layers(menard)]
    assert [r.layer for r in result] == names
    for r in result:
        assert r.onset_1_m.shape == r.full_slip_1_m.shape == (2,)


def test_values_are_the_papailiou_limits_of_the_bending_law_chain(menard):
    tension = 40e3
    result = cx.armour_slip_curvatures(menard, [tension], mu=0.2)
    specs = cable_to_mkappa_specs(menard, tension, mu=0.2, contact_model="bounded")
    for r, spec in zip(result, specs, strict=True):
        assert r.onset_1_m[0] == pytest.approx(slip_onset_curvature(spec), rel=1e-12)
        assert r.full_slip_1_m[0] == pytest.approx(full_slip_curvature(spec), rel=1e-12)


def test_onset_precedes_full_slip_and_both_rise_with_tension(menard):
    tensions = np.array([0.0, 20e3, 60e3, 120e3])
    for r in cx.armour_slip_curvatures(menard, tensions):
        assert np.all(r.onset_1_m <= r.full_slip_1_m + 1e-15)
        assert np.all(np.diff(r.onset_1_m) >= 0.0)
        assert np.all(np.diff(r.full_slip_1_m) >= 0.0)
        assert r.full_slip_1_m[-1] > r.full_slip_1_m[0]


def test_friction_scales_both_curvatures(menard):
    low = cx.armour_slip_curvatures(menard, [50e3], mu=0.1)
    high = cx.armour_slip_curvatures(menard, [50e3], mu=0.2)
    for a, b in zip(low, high, strict=True):
        assert b.onset_1_m[0] > a.onset_1_m[0]
        assert b.full_slip_1_m[0] > a.full_slip_1_m[0]


@pytest.mark.parametrize("tensions", [[], [np.nan], [np.inf, 1.0]])
def test_bad_tensions_are_refused(menard, tensions):
    with pytest.raises(CableInputError):
        cx.armour_slip_curvatures(menard, tensions)
