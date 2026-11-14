# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Cable adapters: contact-state combinations, crossing loads and curve checks."""

from __future__ import annotations

import math
import warnings

import pytest

import cablexsec as cx
from cablexsec.cable_adapters import (
    armour_contact_pressure,
    armour_crossing_line_loads,
    cable_to_fatigue_layers,
    cable_to_mkappa_specs,
)
from cablexsec.exceptions import CableSchemaError
from cablexsec.mechanics.radial_cable import solve_cable_radial_state


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(scope="module")
def menard():
    return cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)


def _with(cable, name, **update):
    layers = [
        layer.model_copy(update=update) if layer.name == name else layer
        for layer in cable.layers
    ]
    return cable.model_copy(update={"layers": layers})


def test_contact_state_combinations(menard):
    section = cx.solve_section_state(menard, tension_N=1e4, contact_model="bounded")
    radial = solve_cable_radial_state(menard, 1e4)
    with pytest.raises(CableSchemaError, match="not both"):
        armour_contact_pressure(
            menard,
            1e4,
            contact_model="solved",
            section_state=section,
            radial_state=radial,
        )
    with pytest.raises(CableSchemaError, match="does not use a section state"):
        armour_contact_pressure(
            menard, 1e4, contact_model="solved", section_state=section
        )


def test_crossing_load_checks():
    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    with pytest.raises(CableSchemaError, match="contact_model"):
        armour_crossing_line_loads(cable, 1e4, contact_model="exact")
    with pytest.raises(CableSchemaError, match="finite"):
        armour_crossing_line_loads(cable, math.nan)
    # One structural armour: only that layer receives an entry.
    single = _with(cable, "outer_armour", structural=False)
    loads = armour_crossing_line_loads(single, 1e4)
    assert set(loads) == {"inner_armour"}
    solved = armour_crossing_line_loads(cable, 1e4, contact_model="solved")
    assert set(solved) == {"inner_armour", "outer_armour"} and all(
        load > 0 for _, load in solved.values()
    )


def test_no_armour_specs(menard):
    bare = menard
    for layer in menard.helical_layers:
        bare = _with(bare, layer.name, structural=False)
    with pytest.raises(CableSchemaError):
        cable_to_mkappa_specs(bare)


def test_blocked_and_unknown_armour_curves(menard):
    with pytest.raises(CableSchemaError, match="BLOCKED"):
        cable_to_fatigue_layers(
            _with(menard, "inner_armour", bulk_sn="perier_2009_steel")
        )
    with pytest.raises(CableSchemaError, match="unknown bulk S-N"):
        cable_to_fatigue_layers(_with(menard, "inner_armour", bulk_sn="no_such_curve"))
