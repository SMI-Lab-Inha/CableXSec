# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Sheath and laminate strain life, and capacity details: checks and branches."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.capacity import (
    birdcage_buckling_detail,
    compute_capacity,
    wire_yield_curvature_levels,
)
from cablexsec.analysis.sheath_strain import (
    compute_core_layer_strain,
    strain_life_sheath_fatigue,
)
from cablexsec.exceptions import CableInputError, CableSchemaError

K = 0.2 * np.sin(np.linspace(0.0, 4.0 * math.pi, 81))


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(scope="module")
def export():
    return cx.load_cable(cx.export_three_core_yaml_path(), geometry_override=None)


@pytest.fixture(scope="module")
def thies():
    return cx.load_cable(cx.thies2016_yaml_path(), geometry_override=None)


SHEATH = {"strain_life": "aluminium_foil_mus_assumed"}


@pytest.mark.parametrize(
    ("layer", "kwargs", "match"),
    [
        ("absent", {}, "absent"),
        ("al_metallic_sheath", {"correction": "goodman"}, "mean-stress correction"),
        ("al_metallic_sheath", {"n_theta": 0}, "n_theta"),
        ("al_metallic_sheath", {"kappa_y": K[:-1]}, "equal 1-D curvature"),
        ("al_metallic_sheath", {"core_state": "free"}, "core_state"),
        ("inner_armour", {}, "cylindrical|shell|layer"),
    ],
)
def test_sheath_strain_life_checks(export, layer, kwargs, match):
    with pytest.raises((CableInputError, CableSchemaError), match=match):
        strain_life_sheath_fatigue(export, layer, K, **{**SHEATH, **kwargs})


def test_sheath_axial_strain_and_hoop_ratio(export):
    plain = strain_life_sheath_fatigue(export, "al_metallic_sheath", K, **SHEATH)
    pulled = strain_life_sheath_fatigue(
        export, "al_metallic_sheath", K, axial_strain=1e-3 * (1 + K / 0.2), **SHEATH
    )
    assert pulled.D_miner > plain.D_miner
    hoop = strain_life_sheath_fatigue(
        export, "al_metallic_sheath", K, hoop_ratio=-0.3, **SHEATH
    )
    assert np.isfinite(hoop.D_miner)


def test_laminate_needs_a_core_state(thies):
    with pytest.raises(CableInputError, match="core_state"):
        strain_life_sheath_fatigue(
            thies, "al_laminate", K, strain_life="aluminium_foil_mus_assumed"
        )


def test_core_layer_strain_inputs(thies):
    with pytest.raises(CableInputError, match="n_theta"):
        compute_core_layer_strain(thies, "al_laminate", K, core_state="free", n_theta=0)
    screen = compute_core_layer_strain(thies, "cu_screen_wires", K, core_state="free")
    ring = thies.get_layer("cu_screen_wires").strand_layers[0]
    assert screen.radius_m == pytest.approx(ring.r_helix_m)


def test_yield_curvature_levels_checks():
    menard = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    for levels, match in (
        ([0.0, 0.5], "finite and positive"),
        ([math.nan], "finite and positive"),
        ([0.5, 1.5], "lie in \\(0, 1\\]"),
    ):
        with pytest.raises(CableInputError, match=match):
            wire_yield_curvature_levels(menard, 1e4, levels)


def test_rectangular_wire_branches():
    rect = cx.load_cable(
        cx.saevik2011_yaml_path(tendons="rectangular"), geometry_override=None
    )
    layer = rect.get_layer("inner_tensile_armour")
    detail = birdcage_buckling_detail(rect, layer)
    assert detail.P_crit_N > 0
    # The tendon material has no yield strength: no capacity envelope.
    with pytest.raises(
        CableInputError, match="inner_tensile_armour lacks a published yield stress"
    ):
        compute_capacity(rect, np.array([0.0, 1e5]))
