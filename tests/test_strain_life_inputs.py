# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Strain-life constants and damage: checks, cyclic curve and edge histories."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cablexsec.exceptions import CableInputError, CableLookupError
from cablexsec.materials.strain_life import (
    StrainLifeConstants,
    _cyclic_stress_scalar,
    get_strain_life,
    list_strain_life,
    masing_stresses,
    register_strain_life,
    strain_life_damage,
    strain_life_damage_map,
)

COPPER = get_strain_life("copper_poon2022")


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"E_Pa": 0.0}, "finite and positive"),
        ({"eps_f_prime": float("nan")}, "finite and positive"),
        ({"b": 0.1}, "-1 < b < 0"),
        ({"c": -2.5}, "-2 < c < 0"),
        ({"K_prime_Pa": 5e8, "n_prime": None}, "both K' and n'"),
        ({"K_prime_Pa": -5e8, "n_prime": 0.1}, "K' must be finite"),
        ({"K_prime_Pa": 5e8, "n_prime": 1.5}, "n' must lie"),
        ({"reference": " "}, "need a reference"),
        ({"basis": "fitted"}, "basis"),
    ],
)
def test_constant_checks(changes, match):
    with pytest.raises(CableInputError, match=match):
        replace(COPPER, **changes)


def test_universal_slopes_need_positive_inputs():
    with pytest.raises(CableInputError, match="ductility"):
        StrainLifeConstants.from_modified_universal_slopes(
            "x", E_Pa=1e11, sigma_UTS_Pa=3e8, ductility=0.0, reference="test"
        )


def test_given_cyclic_curve_is_used_as_given():
    given = replace(COPPER, K_prime_Pa=4.0e8, n_prime=0.15)
    assert given.cyclic_curve == (4.0e8, 0.15)
    assert not given.cyclic_curve_is_estimate
    with pytest.raises(CableInputError, match="mean-stress correction"):
        given.cycles_to_failure(1e-3, correction="goodman")  # type: ignore[arg-type]


@pytest.mark.parametrize("strain", [1e-6, 1e-3, 2e-2, 0.2])
def test_cyclic_stress_solves_ramberg_osgood(strain):
    e, k, n = 1.2e11, 4.0e8, 0.15
    sigma = _cyclic_stress_scalar(strain, e, k, n)
    assert sigma / e + (sigma / k) ** (1.0 / n) == pytest.approx(strain, rel=1e-10)
    assert _cyclic_stress_scalar(0.0, e, k, n) == 0.0


def test_masing_points():
    with pytest.raises(CableInputError, match="finite 1-D"):
        masing_stresses(np.zeros((2, 2)), COPPER)
    stresses = masing_stresses([1e-3, 1e-3, -1e-3, -1e-3], COPPER)
    assert stresses[0] == stresses[1] and stresses[2] == stresses[3]
    assert stresses[2] == pytest.approx(-stresses[0], rel=1e-9)


def test_damage_edge_histories():
    with pytest.raises(CableInputError, match="mean-stress correction"):
        strain_life_damage([0.0, 1e-3], COPPER, correction="goodman")  # type: ignore[arg-type]
    with pytest.raises(CableInputError, match="one-dimensional"):
        strain_life_damage(np.zeros((2, 2)), COPPER)
    for history in ([1e-3], [1e-3, 1e-3, 1e-3]):
        result = strain_life_damage(history, "copper_poon2022")
        assert result.damage == 0.0 and result.cycles.shape == (0, 6)


def test_damage_map_edge_histories():
    with pytest.raises(CableInputError, match="mean-stress correction"):
        strain_life_damage_map(np.zeros((2, 3)), COPPER, correction="goodman")  # type: ignore[arg-type]
    with pytest.raises(CableInputError, match="points, time"):
        strain_life_damage_map(np.zeros(3), COPPER)
    single = strain_life_damage_map(np.full((2, 1), 1e-3), COPPER)
    assert np.all(single.damage == 0.0)
    flat = strain_life_damage_map(np.full((2, 4), 1e-3), COPPER)
    assert np.all(flat.damage == 0.0)


def test_registry_lookups():
    assert "copper_poon2022" in list_strain_life()
    with pytest.raises(CableLookupError):
        get_strain_life("no_such_constants")
    with pytest.raises(CableInputError, match="StrainLifeConstants instance"):
        register_strain_life("copper")  # type: ignore[arg-type]
