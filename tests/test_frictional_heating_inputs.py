# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Frictional heating: separable sources, the heater set point and input checks.

Complements tests/test_frictional_heating.py: the armour heat does not depend
on whether the core bundle is included; a heater that cannot cool lets the
conductor float above a falling set point and takes it again when the set
point catches up; explicit ambient node temperatures equal the ``"ambient"``
start; and every invalid input is rejected with its reason.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from cablexsec.analysis.frictional_heating import (
    CORE_BUNDLE,
    CyclicBending,
    anchored_heat_profile,
    friction_heat,
    simulate_frictional_heating,
)
from cablexsec.cables import iea_task49_66kv_yaml_path, load_cable
from cablexsec.cables.schema import CableData, InitialStateProperties
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.thermal_network import ThermalNetwork, thermal_network

RIG = CyclicBending(
    tension_N=80e3,
    amplitude_1_m=0.5 / 3.5,
    mean_1_m=0.5 / 3.5,
    frequency_Hz=0.17,
    points_per_cycle=48,
)


def _iea(pressure_Pa: float = 0.0) -> CableData:
    cable = load_cable(iea_task49_66kv_yaml_path())
    if pressure_Pa == 0.0:
        return cable
    state = InitialStateProperties(
        external_pressure_Pa=pressure_Pa,
        identification="published_calibration",
        source="Menard and Cartraud (2023), four-point bending calibration",
    )
    return cable.model_copy(update={"initial_state": state})


def _network(cable: CableData) -> ThermalNetwork:
    return thermal_network(
        cable, 630e-6, T4_external_Km_W=1.0 / (math.pi * 0.184 * 8.0), ambient_C=20.0
    )


# --------------------------------------------------------------------------- #
# Friction heat of one cycle                                                   #
# --------------------------------------------------------------------------- #
def test_armour_heat_does_not_depend_on_the_core_bundle() -> None:
    cable = _iea(0.30e6)
    armour_only = CyclicBending(
        tension_N=RIG.tension_N,
        amplitude_1_m=RIG.amplitude_1_m,
        mean_1_m=RIG.mean_1_m,
        frequency_Hz=RIG.frequency_Hz,
        points_per_cycle=RIG.points_per_cycle,
        core_friction=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        both = friction_heat(cable, RIG)
        armour = friction_heat(cable, armour_only)
    assert CORE_BUNDLE in both.heat_W_m and CORE_BUNDLE not in armour.heat_W_m
    for name, value in armour.heat_W_m.items():
        assert value == pytest.approx(both.heat_W_m[name], rel=1e-12)
    assert both.total_heat_W_m == pytest.approx(sum(both.heat_W_m.values()))
    assert both.total_heat_W_m == pytest.approx(
        armour.total_heat_W_m + both.heat_W_m[CORE_BUNDLE], rel=1e-12
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"amplitude_1_m": math.nan}, "amplitude_1_m must be finite"),
        ({"mean_1_m": math.inf}, "mean_1_m must be finite"),
        ({"tension_N": -1.0}, "non-negative"),
    ],
)
def test_cyclic_bending_rejects_bad_values(kwargs: dict, match: str) -> None:
    arguments = {"tension_N": 1e3, "amplitude_1_m": 0.1, "frequency_Hz": 0.1} | kwargs
    with pytest.raises(CableInputError, match=match):
        CyclicBending(**arguments)


# --------------------------------------------------------------------------- #
# Anchored heat profile                                                        #
# --------------------------------------------------------------------------- #
def _field(kappa: float, points: int = 24) -> object:
    def field(x: np.ndarray) -> np.ndarray:
        phase = np.linspace(0.0, 2.0, 2 * points + 1)
        history = 0.5 * kappa * (1.0 - np.cos(2 * math.pi * phase))
        return history[:, None] * np.ones(x.size)[None, :]

    return field


_SMALL = {"phases": 2, "segments_per_pitch": 8, "far_field_pitches": 4}


def test_anchored_profile_needs_slip_at_the_reference_station() -> None:
    # A specimen that is not bent releases no sectional heat to refer to.
    with pytest.raises(CableInputError, match="does not slip"):
        anchored_heat_profile(1.0, 0.01, 6.0, _field(0.0), 24, **_SMALL)
    profile = anchored_heat_profile(1.0, 0.01, 6.0, _field(0.2), 24, **_SMALL)
    # The ratio over a range is the mean of the nodes inside it.
    inside = (profile.x_m >= 2.0) & (profile.x_m <= 4.0)
    assert profile.mean(2.0, 4.0) == pytest.approx(profile.ratio[inside].mean())
    with pytest.raises(CableInputError, match="no node"):
        profile.mean(10.0, 11.0)


@pytest.mark.parametrize(
    ("args", "kwargs", "match"),
    [
        ((0.0, 0.01, 6.0), {}, "pitch_m must be positive"),
        ((1.0, math.nan, 6.0), {}, "slip_onset_curvature_1_m must be positive"),
        ((1.0, 0.01, 6.0), {"phases": 0}, "phases >= 1"),
        ((1.0, 0.01, 6.0), {"segments_per_pitch": 4}, "8 segments per pitch"),
    ],
)
def test_anchored_profile_rejects_bad_geometry(
    args: tuple, kwargs: dict, match: str
) -> None:
    options = _SMALL | kwargs
    with pytest.raises(CableInputError, match=match):
        anchored_heat_profile(*args, _field(0.2), 24, **options)


def test_anchored_profile_rejects_a_mismatched_field() -> None:
    # 6 pitches of 8 segments: 48 columns.
    with pytest.raises(CableInputError, match="48 columns"):
        anchored_heat_profile(1.0, 0.01, 6.0, np.zeros((10, 47)), 0, **_SMALL)
    with pytest.raises(CableInputError, match="last_cycle_start"):
        anchored_heat_profile(1.0, 0.01, 6.0, np.zeros((10, 48)), 10, **_SMALL)


# --------------------------------------------------------------------------- #
# Coupled heating                                                              #
# --------------------------------------------------------------------------- #
def test_heater_lets_the_conductor_float_and_takes_it_again() -> None:
    cable = _iea()
    net = _network(cable)
    hour = 3600.0
    t = np.arange(0.0, 6 * hour + 1.0, 600.0)
    # Up to 70 C, a fast drop to 30 C, then a slow climb back to 70 C.
    set_point = np.interp(
        t, [0.0, 2 * hour, 2 * hour + 600.0, 6 * hour], [20.0, 70.0, 30.0, 70.0]
    )
    run = {"conductor_temperature_C": set_point, "ambient_C": 20.0, "coupling": "none"}
    floating = simulate_frictional_heating(cable, net, RIG, t, **run)
    exact = simulate_frictional_heating(cable, net, RIG, t, heater_only=False, **run)
    conductor = floating.node_temperature_C[:, 0]
    # The exact control follows the set point everywhere, cooling included.
    np.testing.assert_allclose(exact.node_temperature_C[:, 0], set_point, atol=1e-9)
    assert exact.conductor_heat_W_m.min() < 0.0
    # The heater that cannot cool lets the conductor float above the set
    # point without power, until the climbing set point catches it.
    above = conductor > set_point + 1e-6
    assert above.any() and not above[-1]
    assert np.all(floating.conductor_heat_W_m >= 0.0)
    assert np.all(floating.conductor_heat_W_m[:-1][above[1:]] == 0.0)
    assert conductor[-1] == pytest.approx(set_point[-1], abs=1e-9)


def test_explicit_ambient_nodes_equal_the_ambient_start() -> None:
    cable = _iea()
    net = _network(cable)
    t = np.linspace(0.0, 2 * 3600.0, 5)
    run = {"current_A": 500.0, "ambient_C": 20.0, "coupling": "none"}
    default = simulate_frictional_heating(cable, net, RIG, t, **run)
    explicit = simulate_frictional_heating(
        cable, net, RIG, t, initial=np.full(net.n_nodes, 20.0), **run
    )
    np.testing.assert_array_equal(
        explicit.node_temperature_C, default.node_temperature_C
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"update_interval_s": 0.0}, "update_interval_s must be positive"),
        ({"max_step_s": math.nan}, "max_step_s must be positive"),
        ({"time_s": np.array([0.0, 0.0])}, "strictly increasing"),
        ({"time_s": np.array([0.0])}, "strictly increasing"),
        ({"current_A": [0.0, math.nan]}, "conductor history must be finite"),
        ({"initial": "steady"}, "initial must be"),
        ({"initial": np.zeros(2)}, "do not match the network"),
    ],
)
def test_simulation_rejects_bad_values(kwargs: dict, match: str) -> None:
    cable = _iea()
    net = _network(cable)
    arguments = {"time_s": np.array([0.0, 60.0]), "current_A": 0.0} | kwargs
    time_s = arguments.pop("time_s")
    with pytest.raises(CableInputError, match=match):
        simulate_frictional_heating(cable, net, RIG, time_s, **arguments)
