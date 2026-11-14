# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Radial section of a cable: material inputs, loose layers and trefoil checks."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.exceptions import CableSchemaError
from cablexsec.mechanics.radial_cable import (
    _annulus_bounds,
    build_radial_section,
    loose_layer_sources,
    solve_cable_radial_state,
)


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(scope="module")
def iea66():
    return cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)


def _with(cable, name, **update):
    layers = [
        layer.model_copy(update=update) if layer.name == name else layer
        for layer in cable.layers
    ]
    return cable.model_copy(update={"layers": layers})


def _without(cable, name):
    return cable.model_copy(
        update={"layers": [x for x in cable.layers if x.name != name]}
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"youngs_modulus_Pa": {"nowhere": 1e9}}, "unknown layer"),
        ({"temperature_C": {"nowhere": 30.0}}, "unknown layer"),
        (
            {"temperature_C": {"xlpe_insulation": math.nan}},
            "temperatures must be finite",
        ),
        ({"youngs_modulus_Pa": {"xlpe_insulation": 0.0}}, "moduli must be finite"),
        ({"loose_layers": "glued"}, "loose_layers"),
    ],
)
def test_material_inputs(iea66, kwargs, match):
    with pytest.raises(CableSchemaError, match=match):
        build_radial_section(iea66, **kwargs)


def test_layer_temperatures_and_modulus_override(iea66):
    hot = build_radial_section(iea66, temperature_C={"xlpe_insulation": 60.0})
    plain = build_radial_section(iea66)
    # The heated layer's free thermal strain loads the section.
    assert not np.allclose(hot[0].free_load, plain[0].free_load)
    stiff = solve_cable_radial_state(iea66, 2e4, youngs_modulus_Pa={"mdpe_filler": 5e8})
    assert stiff.tension_N == 2e4


def test_loose_layers_take_a_neighbour_material(iea66):
    sources = loose_layer_sources(iea66)
    assert sources and all(iea66.get_layer(s).structural for s in sources.values())
    _, notes = build_radial_section(iea66, loose_layers="neighbour")
    assert any("meshed with the material of" in note for note in notes)


def test_trefoil_checks(iea66):
    with pytest.raises(CableSchemaError, match="strand radial law"):
        build_radial_section(cx.iea_task49_66kv_stranded())
    with pytest.raises(CableSchemaError, match="three-core trefoil"):
        build_radial_section(_with(iea66, "xlpe_insulation", n_cores=2))
    offset = iea66.get_layer("xlpe_insulation").r_offset_m
    with pytest.raises(CableSchemaError, match="one pitch radius"):
        build_radial_section(_with(iea66, "xlpe_insulation", r_offset_m=offset * 1.01))
    filler = iea66.get_layer("mdpe_filler")
    twin = filler.model_copy(update={"name": "mdpe_filler_2"})
    doubled = iea66.model_copy(update={"layers": [*iea66.layers, twin]})
    with pytest.raises(CableSchemaError, match="more than one interstitial filler"):
        build_radial_section(doubled)
    bare = _without(iea66, "mdpe_filler")
    with pytest.raises(CableSchemaError, match="declares no filler"):
        build_radial_section(bare, filler="solid")
    _, notes = build_radial_section(iea66, filler="void")
    assert any("omitted (void interstices)" in note for note in notes)


def test_interface_lookup(iea66):
    state = solve_cable_radial_state(iea66, 2e4)
    with pytest.raises(CableSchemaError, match="not adjacent"):
        state.interface_line_force("inner_armour", "hdpe_outer_sheath")


def test_shell_bounds():
    export = cx.load_cable(cx.export_three_core_yaml_path(), geometry_override=None)
    shell = export.get_layer("al_metallic_sheath")
    flat = shell.model_copy(update={"f_corrugation": 1.0})
    lo, hi = _annulus_bounds(flat)
    assert hi - lo == pytest.approx(shell.t_m) and (lo + hi) / 2 == pytest.approx(
        shell.r_mid_m
    )
    with pytest.raises(CableSchemaError, match="corrugated shells"):
        _annulus_bounds(flat.model_copy(update={"f_corrugation": 1.2}))
    with pytest.raises(CableSchemaError, match="incomplete shell geometry"):
        _annulus_bounds(flat.model_copy(update={"t_m": None}))
    with pytest.raises(CableSchemaError, match="incomplete geometry"):
        _annulus_bounds(
            export.get_layer("hdpe_outer_sheath").model_copy(update={"r_inner_m": None})
        )
