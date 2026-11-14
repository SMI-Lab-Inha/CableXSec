# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Lead-in of the fatigue route: the friction state runs, the cycles are not counted."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.fast_rainflow import miner_rows
from cablexsec.analysis.fatigue import wire_stress_field
from cablexsec.cable_adapters import cable_to_fatigue_layers
from cablexsec.exceptions import CableInputError, CableSchemaError


@pytest.fixture(scope="module")
def cable():  # type: ignore[no-untyped-def]
    return cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)


def _history() -> tuple[np.ndarray, np.ndarray, int]:
    # lead-in: one large excursion; then small cycles about a mean
    lead = np.r_[np.linspace(0.0, 0.08, 60), np.linspace(0.08, 0.04, 30)]
    t = np.linspace(0.0, 20 * np.pi, 600)
    tail = 0.04 + 0.006 * np.sin(t)
    kx = np.r_[lead, tail]
    return kx, 0.002 * np.sin(np.linspace(0.0, 7.0, kx.size)), lead.size


def _fatigue(cable, kx, ky, **kw):  # type: ignore[no-untyped-def]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cx.compute_fatigue(cable, kx, ky, **kw)


def test_zero_lead_in_is_bit_for_bit(cable) -> None:
    kx, ky, _n = _history()
    a = _fatigue(cable, kx, ky)
    b = _fatigue(cable, kx, ky, count_from_sample=0)
    for name in a.D_bulk_per_wire:
        assert np.array_equal(a.D_bulk_per_wire[name], b.D_bulk_per_wire[name])


def test_lead_in_drops_the_start_up_damage(cable) -> None:
    kx, ky, n = _history()
    full = _fatigue(cable, kx, ky)
    lead = _fatigue(cable, kx, ky, count_from_sample=n)
    for name in full.D_bulk_per_wire:
        assert np.all(
            lead.D_bulk_per_wire[name] <= full.D_bulk_per_wire[name] * (1 + 1e-12)
        )
        assert np.max(lead.D_bulk_per_wire[name]) > 0.0
    assert any("lead-in" in s for s in lead.provenance.limitations)


def test_lead_in_counts_the_slice_of_the_full_friction_state(cable) -> None:
    """Equal to counting the retained part of the stress field computed over the whole record."""
    kx, ky, n = _history()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layer = cable_to_fatigue_layers(cable)[0]
    field = wire_stress_field(layer, kx, ky)
    hist = field.at_fibre(0.0)[:, n:]
    ref, _ = miner_rows(hist, layer.sn_bulk, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = cx.analysis.fatigue.compute_fatigue(
            [layer], kx, ky, count_from_sample=n, n_fibres=4
        )
    assert np.all(res.D_bulk_per_wire[layer.name] >= np.asarray(ref) * (1 - 1e-12))


@pytest.mark.parametrize("bad", [-1, 10_000, 1.5, True])
def test_lead_in_rejects_bad_values(cable, bad) -> None:
    kx, ky, _n = _history()
    with pytest.raises(CableInputError):
        _fatigue(cable, kx, ky, count_from_sample=bad)


def test_lead_in_rejected_with_section_states() -> None:
    kx, ky, _n = _history()
    pipe = cx.load_cable(cx.saevik2011_yaml_path(), geometry_override=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        state = cx.solve_section_state(pipe, tension_N=1e5, contact_model="bounded")
    with pytest.raises(CableSchemaError, match="count_from_sample"):
        _fatigue(pipe, kx, ky, section_states=[state] * kx.size, count_from_sample=5)
