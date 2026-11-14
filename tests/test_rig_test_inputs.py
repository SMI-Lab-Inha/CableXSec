# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Full-scale rig test: protocol and set-up checks, notes of a short run."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.rig_test import (
    GaugeLocation,
    RigProtocol,
    RigSetup,
    end_rotation_profile,
    simulate_rig_test,
)
from cablexsec.analysis.thermal_coupling import ThermalState
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.bend_restrictor import (
    ArticulatedRestrictor,
    restrictor_curvature_field,
)


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(scope="module")
def menard():
    return cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)


def _protocol(amplitude=0.03, cycles=1):
    return RigProtocol.cyclic(
        tension_N=5e4,
        curvature_amplitude_1_m=amplitude,
        n_cycles=cycles,
        samples_per_cycle=16,
    )


def _short(menard, protocol, **kwargs):
    return simulate_rig_test(
        menard,
        protocol,
        kwargs.pop("setup", RigSetup(free_length_m=1.5, clamp="none")),
        gauges=[GaugeLocation("outer_armour", kwargs.pop("angle", 90.0), 0.02)],
        mu_factors=(),
        **kwargs,
    )


def test_protocol_checks():
    with pytest.raises(CableInputError, match="time_s"):
        RigProtocol(
            time_s=np.array([0.0, 1.0]),
            tension_N=np.zeros(2),
            curvature_1_m=np.zeros(2),
        )
    with pytest.raises(CableInputError, match="n_cycles|period"):
        RigProtocol.cyclic(tension_N=5e4, curvature_amplitude_1_m=0.03, n_cycles=0)
    with pytest.raises(CableInputError, match="decay length"):
        end_rotation_profile(np.array([0.5]), 2.0, 0.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_increment": 0.0}, "max_increment must be positive"),
        ({"segments_per_pitch": 4}, "segments_per_pitch"),
        ({"tension_levels": 1}, "tension_levels"),
        ({"setback_variants_m": (-0.1,)}, "setback_variants_m"),
    ],
)
def test_simulation_checks(menard, kwargs, match):
    with pytest.raises(CableInputError, match=match):
        _short(menard, _protocol(), **kwargs)


def test_curvature_field_needs_one_column_per_sample(menard):
    restrictor = ArticulatedRestrictor(
        pitch_m=0.2, lock_angle_rad=math.radians(5.0), n_elements=5, bore_diameter_m=0.3
    )
    field = restrictor_curvature_field(
        restrictor,
        [0.1, 0.2],
        cable_od_m=0.12,
        bending_stiffness_Nm2=5e3,
        tension_N=5e4,
    )
    setup = RigSetup(
        free_length_m=1.5, clamp="none", profile="field", curvature_field=field
    )
    with pytest.raises(CableInputError, match="one column per protocol sample"):
        _short(menard, _protocol(), setup=setup)


def test_thermal_history_needs_the_solved_state(menard):
    state = ThermalState({layer.name: 40.0 for layer in menard.layers})
    with pytest.raises(CableInputError, match="solved radial state"):
        _short(menard, _protocol(), thermal=state, contact_model="bounded")


def test_large_amplitude_notes_wire_yield(menard):
    # Gauge in the bending plane; 5 1/m drives the wire well past 690 MPa.
    result = _short(menard, _protocol(amplitude=5.0), angle=0.0)
    assert any("exceeds the wire yield strength" in note for note in result.notes)
