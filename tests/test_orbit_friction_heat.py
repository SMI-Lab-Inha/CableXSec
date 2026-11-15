# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Frictional heat of armour under rotating (biaxial) curvature.

Gates:

* large-amplitude whirl, closed form: one revolution of a circular orbit of
  radius ``R`` against the two planar cycles of amplitude ``R``, ratio
  ``pi / 4`` for the distributed helix model (every helix node slips through
  ``4 R`` at its bound) and ``pi (pi/2 - 1) / 4`` for the direction-tied cap
  (:func:`cablexsec.mechanics.papailiou.wire_slip_history` notes);
* the heat of ``compute_thermal_fatigue`` is the distributed dissipation of
  the fatigue-path discretisation within 0.5 %, on a planar record (where it
  also matches the direction-tied work within 3 %) and on a whirl record
  (where the direction-tied work is below half of it);
* a rigid interface (``xi = 1``) falls back to the direction-tied work and
  says so in the provenance.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace

import numpy as np
import pytest

import cablexsec as cx
from cablexsec._numba import HAVE_NUMBA
from cablexsec.analysis import thermal_coupling
from cablexsec.analysis.thermal_coupling import (
    HEAT_MAX_INCREMENT,
    HEAT_NODES_PER_PITCH,
    ThermalHistory,
    compute_thermal_fatigue,
    record_friction_work,
    thermal_section,
)
from cablexsec.cable_adapters import cable_to_mkappa_specs
from cablexsec.cables.schema import CableData
from cablexsec.mechanics.distributed_friction import distributed_friction_history
from cablexsec.mechanics.mkappa import HelicalLayerSpec
from cablexsec.mechanics.papailiou import slip_onset_curvature, wire_slip_history

#: Long friction histories run pure Python without numba (minutes per case).
needs_numba = pytest.mark.skipif(
    not HAVE_NUMBA, reason="long friction histories need the compiled kernels"
)

TENSION_N = 50e3
POINTS = 200  # samples per revolution or cycle
CYCLES = 3


@pytest.fixture(scope="module")
def t66() -> CableData:
    return cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)


def _orbit(kind: str, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """A ramp from straight to ``(radius, 0)``, then ``CYCLES`` cycles."""
    theta = np.linspace(0.0, 2.0 * math.pi * CYCLES, CYCLES * POINTS + 1)
    ramp = np.linspace(0.0, radius, POINTS // 4 + 1)[:-1]
    first = np.r_[ramp, radius * np.cos(theta)]
    if kind == "circle":
        return first, np.r_[0.0 * ramp, radius * np.sin(theta)]
    return (first, 0.0 * first) if kind == "x" else (0.0 * first, first)


def _last_cycle(cumulative: np.ndarray) -> float:
    return float(cumulative[-1] - cumulative[-1 - POINTS])


@pytest.fixture(scope="module")
def spec(t66: CableData) -> HelicalLayerSpec:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # zero manufacturing clamping
        return cable_to_mkappa_specs(t66, 100e3)[0]


@needs_numba
@pytest.mark.analytical_sanity
def test_large_whirl_dissipation_matches_closed_form(spec):
    radius = 1024.0 * slip_onset_curvature(spec)
    distributed, direction_tied = {}, {}
    for kind in ("circle", "x", "y"):
        kx, ky = _orbit(kind, radius)
        distributed[kind] = _last_cycle(
            distributed_friction_history(
                spec,
                kx,
                ky,
                nodes_per_pitch=HEAT_NODES_PER_PITCH,
                max_increment=HEAT_MAX_INCREMENT,
            ).dissipation_J_per_m
        )
        direction_tied[kind] = _last_cycle(
            wire_slip_history(spec, kx, ky).work.friction_J_per_m
        )
    ratio_d = distributed["circle"] / (distributed["x"] + distributed["y"])
    ratio_p = direction_tied["circle"] / (direction_tied["x"] + direction_tied["y"])
    # Corrections are of order kappa_on / R and of the capstan term.
    assert ratio_d == pytest.approx(math.pi / 4.0, rel=0.01)
    assert ratio_p == pytest.approx(math.pi * (math.pi / 2.0 - 1.0) / 4.0, rel=0.02)
    # In planar bending the two models dissipate the same work.
    assert direction_tied["x"] == pytest.approx(distributed["x"], rel=0.01)


def _block_heat(t66: CableData, kx: np.ndarray, ky: np.ndarray):
    duration = float(kx.size)
    tension = np.full(kx.size, TENSION_N)
    history = ThermalHistory.isothermal(t66)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compute_thermal_fatigue(
            t66, kx, ky, history, tension_N=tension, record_duration_s=duration
        )
        specs = thermal_section(
            t66, history.blocks()[0].state, tension_N=TENSION_N
        ).specs
    fine = sum(
        float(distributed_friction_history(s, kx, ky).dissipation_J_per_m[-1])
        for s in specs
    )
    tied = sum(
        float(wire_slip_history(s, kx, ky).work.friction_J_per_m[-1]) for s in specs
    )
    return result, result.blocks[0].frictional_heat_W_m * duration, fine, tied


def _onset(t66: CableData) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        specs = thermal_section(
            t66, ThermalHistory.isothermal(t66).blocks()[0].state, tension_N=TENSION_N
        ).specs
    return max(slip_onset_curvature(s) for s in specs)


@needs_numba
@pytest.mark.independent_method
def test_planar_record_heat_keeps_the_planar_work(t66):
    kx, ky = _orbit("x", 4.0 * _onset(t66))
    result, heat, fine, tied = _block_heat(t66, kx, ky)
    assert heat == pytest.approx(fine, rel=5e-3)
    assert heat == pytest.approx(tied, rel=0.03)
    assert any("distributed helix" in x for x in result.provenance.limitations)


@needs_numba
@pytest.mark.independent_method
def test_whirl_record_heat_follows_the_distributed_model(t66):
    kx, ky = _orbit("circle", 4.0 * _onset(t66))
    _result, heat, fine, tied = _block_heat(t66, kx, ky)
    assert heat == pytest.approx(fine, rel=5e-3)
    # The direction-tied cap (the earlier heat) dissipates less than half.
    assert tied < 0.5 * heat


def test_rigid_interface_falls_back_to_direction_tied_work(t66, spec, monkeypatch):
    kx, ky = _orbit("circle", 3.0 * slip_onset_curvature(spec))
    rigid = replace(spec, xi=1.0)
    work, model = record_friction_work(rigid, kx, ky)
    assert model == "direction_tied"
    assert work == float(wire_slip_history(rigid, kx, ky).work.friction_J_per_m[-1])
    work, model = record_friction_work(spec, kx, ky)
    assert model == "distributed" and work > 0.0

    original = thermal_coupling.record_friction_work
    monkeypatch.setattr(
        thermal_coupling,
        "record_friction_work",
        lambda s, x, y: original(replace(s, xi=1.0), x, y),
    )
    short_x, short_y = kx[:60], ky[:60]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compute_thermal_fatigue(
            t66,
            short_x,
            short_y,
            ThermalHistory.isothermal(t66),
            tension_N=np.full(short_x.size, TENSION_N),
            record_duration_s=60.0,
        )
    notes = [x for x in result.provenance.limitations if "xi = 1" in x]
    assert len(notes) == 2 and "inner_armour" in notes[0]
