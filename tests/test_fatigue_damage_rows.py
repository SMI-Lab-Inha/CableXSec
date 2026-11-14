# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Fatigue route: compiled and reference error rules, corrected lead-in count."""

from __future__ import annotations

import dataclasses
import math
import warnings

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.fatigue import _miner_damage_rows, wire_stress_field
from cablexsec.analysis.fretting_swt import swt_life, swt_slip_cycle_damage
from cablexsec.analysis.sn_assessment import corrected_damage_rows
from cablexsec.cable_adapters import cable_to_fatigue_layers
from cablexsec.exceptions import CableInputError
from cablexsec.materials.mean_stress import MeanStressSpec
from cablexsec.materials.sn_curves import SNcurve, get_sn_curve, get_swt_life

MPA = 1.0e6


@pytest.fixture(scope="module")
def menard_layer():
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cable_to_fatigue_layers(cable)[0]


@pytest.mark.parametrize("compiled", [False, True])
def test_compiled_and_reference_rows_raise_the_same_errors(compiled):
    curve = get_sn_curve("nasution_2014_copper")
    rows = np.tile([-50.0, 50.0], (2, 4)) * MPA
    slip_curve = SNcurve(name="slip", stress_unit="um", m1=2.0, log_C1=10.0)
    with pytest.raises(CableInputError, match="requires an S-N curve in MPa"):
        _miner_damage_rows(rows, slip_curve, 4e8, 1.0, compiled)
    with pytest.raises(CableInputError, match="must be finite and positive"):
        _miner_damage_rows(rows, curve, 4e8, 0.0, compiled)
    bad = rows.copy()
    bad[1, 2] = math.nan
    with pytest.raises(CableInputError, match="one-dimensional and finite"):
        _miner_damage_rows(bad, curve, 4e8, 1.0, compiled)


def test_compiled_rows_equal_the_reference_rows():
    curve = get_sn_curve("nasution_2014_copper")
    t = np.linspace(0.0, 8 * math.pi, 401)
    rows = np.vstack([60.0 * np.sin(t), 90.0 * np.sin(2 * t) + 20.0]) * MPA
    reference = _miner_damage_rows(rows, curve, 4e8, 1.3, False)
    compiled = _miner_damage_rows(rows, curve, 4e8, 1.3, True)
    np.testing.assert_allclose(compiled, reference, rtol=1e-12)


def test_corrected_lead_in_counts_the_slice_of_the_full_field(menard_layer):
    """SWT route with a lead-in: the retained slice of the whole-record field."""
    lead = np.r_[np.linspace(0.0, 0.08, 40), np.linspace(0.08, 0.04, 20)]
    t = np.linspace(0.0, 12 * np.pi, 300)
    kx = np.r_[lead, 0.04 + 0.006 * np.sin(t)]
    ky = 0.002 * np.sin(np.linspace(0.0, 7.0, kx.size))
    n = lead.size
    layer = dataclasses.replace(menard_layer, mean_stress=MeanStressSpec(method="swt"))
    field = wire_stress_field(layer, kx, ky)
    reference = corrected_damage_rows(
        field.at_fibre(0.0)[:, n:],
        layer.sn_bulk,
        mean_stress=layer.mean_stress,
        thickness_m=layer.thickness_m or 2 * math.sqrt(layer.spec.A_wire_m2 / math.pi),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lead_in = cx.analysis.fatigue.compute_fatigue(
            [layer], kx, ky, count_from_sample=n, n_fibres=4
        )
        whole = cx.analysis.fatigue.compute_fatigue([layer], kx, ky, n_fibres=4)
    damage = lead_in.D_bulk_per_wire[layer.name]
    # The per-wire damage is the worst of the fibres, fibre 0 among them.
    assert np.all(damage >= reference * (1 - 1e-12))
    assert np.max(damage) > 0.0
    # Dropping the start-up excursion never adds damage.
    assert np.all(damage <= whole.D_bulk_per_wire[layer.name] * (1 + 1e-12))


# --------------------------------------------------------------------------- #
# SWT fretting on a contact-force history                                     #
# --------------------------------------------------------------------------- #
def test_frictionless_contact_has_no_fretting_damage():
    t = np.linspace(0.0, 6 * math.pi, 601)
    force = np.sin(t)
    slipping = np.ones(t.size, dtype=bool)
    result = swt_slip_cycle_damage(
        force, slipping, 2.0, 26.0e6, 0.0, 97.0e9, 0.34, get_swt_life("copper_poon2022")
    )
    # Every cycle slips and is evaluated, but without friction the surface
    # driver vanishes: zero SWT, infinite lives, no damage.
    assert result.evaluated_cycles == result.counted_cycles > 0
    np.testing.assert_array_equal(result.amplitude_ratio, 1.0)
    np.testing.assert_array_equal(result.swt_Pa, 0.0)
    assert np.all(np.isinf(result.life_cycles))
    assert result.damage == 0.0


def test_swt_life_rejects_exponents_beyond_floating_point_range():
    copper = get_swt_life("copper_poon2022")
    with pytest.raises(CableInputError, match="exponents exceed floating-point"):
        swt_life(1.0e6, dataclasses.replace(copper, b=-1.0e308))
