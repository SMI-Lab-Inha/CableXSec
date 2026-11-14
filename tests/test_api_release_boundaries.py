# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Public calls reject conflicting loads and malformed histories before solving."""

import numpy as np
import pytest

import cablexsec as cx


@pytest.fixture
def cable():
    return cx.load_cable(cx.inter_array_single_core_yaml_path(), geometry_override=None)


@pytest.mark.parametrize("history", [[], [0.0], [[0.0, 0.1]]])
def test_bending_requires_a_history(cable, history):
    with pytest.raises(cx.CableSchemaError, match="history"):
        cx.compute_mkappa(cable, np.asarray(history))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"tension_N": -1}, "tension"),
        ({"mu": -1}, "mu"),
        ({"contact_model": "unknown"}, "contact_model"),
    ],
)
def test_bending_rejects_invalid_contact_inputs(cable, kwargs, match):
    with pytest.raises(cx.CableSchemaError, match=match):
        cx.compute_mkappa(cable, np.zeros(2), **kwargs)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"mu": -1}, "mu"),
        ({"contact_model": "unknown"}, "contact_model"),
        ({"axial_strain": np.zeros(2), "tension_N": np.zeros(2)}, "not both"),
        ({"axial_strain": np.zeros(3)}, "shape"),
        ({"tension_N": np.zeros(3)}, "shape"),
        ({"tension_N": -np.ones(2)}, "tension"),
        ({"contact_tension_N": -1}, "contact_tension"),
        ({"section_states": []}, "section states"),
        ({"section_states": [], "tension_N": np.zeros(2)}, "not both"),
        ({"contact_model": "qp", "contact_tension_N": 1}, "consistent section states"),
        ({"contact_model": "qp", "axial_strain": np.zeros(2)}, "strain alone"),
    ],
)
def test_fatigue_rejects_inconsistent_loads(cable, kwargs, match):
    with pytest.raises(cx.CableSchemaError, match=match):
        cx.compute_fatigue(cable, np.zeros(2), np.zeros(2), **kwargs)


@pytest.mark.parametrize(
    "x,y", [([], []), ([0], [0]), ([0, 1], [0]), ([[0, 1]], [[0, 1]])]
)
def test_fatigue_requires_matching_histories(cable, x, y):
    with pytest.raises(cx.CableSchemaError, match="histories"):
        cx.compute_fatigue(cable, np.asarray(x), np.asarray(y))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"contact_model": "unknown"}, "contact_model"),
        ({"p_int_Pa": -1}, "pressures"),
        ({"p_ext_Pa": -1}, "pressures"),
        ({"tension_N": np.nan}, "finite"),
        ({"youngs_modulus_Pa": {"missing": 1e9}}, "unknown layer"),
        ({"temperature_C": {}}, "every section component"),
        ({"temperature_C": np.nan}, "finite"),
        ({"youngs_modulus_Pa": {"inner_armour": -1}}, "modul"),
    ],
)
def test_section_state_rejects_invalid_loads(cable, kwargs, match):
    with pytest.raises(cx.CableSchemaError, match=match):
        cx.solve_section_state(cable, **kwargs)


def test_bending_rejects_conflicting_state_and_load(cable):
    state = cx.solve_section_state(cable, contact_model="bounded")
    with pytest.raises(cx.CableSchemaError, match="not both"):
        cx.compute_mkappa(cable, np.zeros(2), section_state=state, tension_N=1)


def test_fatigue_accepts_explicit_friction(cable):
    result = cx.compute_fatigue(cable, np.zeros(2), np.zeros(2), mu=0)
    for damage in result.D_total_per_wire.values():
        np.testing.assert_array_equal(damage, np.zeros_like(damage))
