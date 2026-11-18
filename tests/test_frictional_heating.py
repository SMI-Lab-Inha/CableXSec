# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Frictional heating coupled with temperature (cablexsec.analysis.frictional_heating)."""

from __future__ import annotations

import math
import warnings
from functools import cache

import numpy as np
import pytest

from cablexsec.analysis.frictional_heating import (
    CORE_BUNDLE,
    CyclicBending,
    anchored_heat_profile,
    friction_heat,
    simulate_frictional_heating,
)
from cablexsec.analysis.thermal_coupling import ThermalState, thermal_section
from cablexsec.cables import load_cable
from cablexsec.cables.schema import CableData, InitialStateProperties
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.clamp_layer import (
    anchored_wire_field_response,
    anchored_wire_node_dissipation,
)
from cablexsec.mechanics.papailiou import compute_mkappa_wire_slip_front
from cablexsec.mechanics.thermal_network import ThermalNetwork, thermal_network

# The heated rig of Stamelos et al. (2026): 80 kN, one-sided onto a 3.5 m former.
RIG = CyclicBending(
    tension_N=80e3,
    amplitude_1_m=0.5 / 3.5,
    mean_1_m=0.5 / 3.5,
    frequency_Hz=0.17,
    points_per_cycle=48,
)


@cache
def _iea(pressure_Pa: float = 0.0) -> CableData:
    from cablexsec.cables import iea_task49_66kv_yaml_path

    cable = load_cable(iea_task49_66kv_yaml_path())
    if pressure_Pa == 0.0:
        return cable
    state = InitialStateProperties(
        external_pressure_Pa=pressure_Pa,
        identification="published_calibration",
        source="Menard and Cartraud (2023), four-point bending calibration",
    )
    return cable.model_copy(update={"initial_state": state})


def _network(cable: CableData, h_surface: float = 8.0) -> ThermalNetwork:
    return thermal_network(
        cable,
        630e-6,
        T4_external_Km_W=1.0 / (math.pi * 0.184 * h_surface),
        ambient_C=20.0,
    )


# --------------------------------------------------------------------------- #
# Anchored wires                                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.analytical_sanity
def test_node_dissipation_sums_to_the_field_total() -> None:
    rng = np.random.default_rng(7)
    m = 48
    x = (np.arange(m) + 0.5) / m
    shape = np.exp(-(((x - 0.5) / 0.25) ** 2))
    load = np.concatenate([np.linspace(0, 6, 25), np.linspace(6, -4, 30)[1:]])
    drive = load[:, None] * shape[None, :]
    c_hat = np.cos(rng.uniform(0, 2 * math.pi) + 2 * math.pi * 1.5 * x)
    h = 2 * math.pi * 1.5 / m
    nodes = anchored_wire_node_dissipation(drive, c_hat, h)
    total = anchored_wire_field_response(drive, c_hat, h)[2]
    assert nodes.shape == (m - 1,)
    assert np.all(nodes >= 0.0)
    assert nodes.sum() == pytest.approx(total, rel=1e-12, abs=1e-14)


def test_node_dissipation_window_counts_only_later_samples() -> None:
    m = 32
    x = (np.arange(m) + 0.5) / m
    drive = np.linspace(0, 8, 40)[:, None] * np.ones(m)[None, :]
    c_hat = np.cos(2 * math.pi * x)
    h = 2 * math.pi / m
    every = anchored_wire_node_dissipation(drive, c_hat, h)
    late = anchored_wire_node_dissipation(drive, c_hat, h, from_sample=20)
    assert 0.0 < late.sum() < every.sum()
    with pytest.raises(CableInputError):
        anchored_wire_node_dissipation(drive, c_hat, h, from_sample=40)


def _one_sided_field(
    x: np.ndarray, kappa: float, cycles: int, points: int
) -> np.ndarray:
    phase = np.linspace(0.0, cycles, cycles * points + 1)
    history = 0.5 * kappa * (1.0 - np.cos(2 * math.pi * phase))
    return history[:, None] * np.ones(x.size)[None, :]


@pytest.mark.analytical_sanity
def test_anchored_heat_is_sectional_far_from_the_anchors() -> None:
    pitch, kappa_c, kappa = 1.0, 0.01, 0.2  # a = 20: boundary layer below a pitch
    points, cycles = 60, 2
    profile = anchored_heat_profile(
        pitch,
        kappa_c,
        8 * pitch,
        lambda x: _one_sided_field(x, kappa, cycles, points),
        (cycles - 1) * points,
        phases=8,
        segments_per_pitch=32,
        far_field_pitches=8,
    )
    assert profile.mean(3.0, 5.0) == pytest.approx(1.0, abs=0.03)
    # The wire cannot slip at the clamp, and the anchor raises the heat just
    # inside its boundary layer.
    assert profile.ratio[0] < 0.5
    assert profile.ratio[profile.x_m <= 1.0].max() > 1.05
    assert profile.share_outside(0.0, 8.0) == 0.0


# --------------------------------------------------------------------------- #
# Friction heat of one cycle                                                   #
# --------------------------------------------------------------------------- #
def test_friction_heat_is_frequency_times_work_and_the_layers_add() -> None:
    cable = _iea(0.30e6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        heat = friction_heat(cable, RIG)
        doubled = friction_heat(
            cable,
            CyclicBending(
                tension_N=RIG.tension_N,
                amplitude_1_m=RIG.amplitude_1_m,
                mean_1_m=RIG.mean_1_m,
                frequency_Hz=RIG.frequency_Hz,
                points_per_cycle=RIG.points_per_cycle,
                anchoring_factor={"inner_armour": 2.0},
            ),
        )
        state = ThermalState({layer.name: 20.0 for layer in cable.layers})
        section = thermal_section(cable, state, tension_N=RIG.tension_N)
    assert set(heat.work_J_m) == {"inner_armour", "outer_armour", CORE_BUNDLE}
    for name, work in heat.work_J_m.items():
        assert work > 0.0
        assert heat.heat_W_m[name] == pytest.approx(RIG.frequency_Hz * work)
    assert doubled.heat_W_m["inner_armour"] == pytest.approx(
        2.0 * heat.heat_W_m["inner_armour"]
    )
    # The wires of the slip-front law are independent: layers add up.
    kappa, last = RIG.history()
    together = compute_mkappa_wire_slip_front(list(section.specs), kappa)
    cumulative = together.work.friction_J_per_m
    armour = heat.work_J_m["inner_armour"] + heat.work_J_m["outer_armour"]
    assert cumulative[-1] - cumulative[last] == pytest.approx(armour, rel=1e-9)


def test_clamping_raises_the_friction_work() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loose = friction_heat(_iea(), RIG)
        clamped = friction_heat(_iea(0.30e6), RIG)
    for name in ("inner_armour", "outer_armour"):
        assert clamped.contact_pressure_Pa[name] > loose.contact_pressure_Pa[name]
        assert clamped.work_J_m[name] > 5.0 * loose.work_J_m[name]


def test_cyclic_bending_rejects_bad_input() -> None:
    with pytest.raises(CableInputError):
        CyclicBending(tension_N=1e3, amplitude_1_m=0.0, frequency_Hz=0.1)
    with pytest.raises(CableInputError):
        CyclicBending(tension_N=1e3, amplitude_1_m=0.1, frequency_Hz=0.1, cycles=1)
    with pytest.raises(CableInputError):
        CyclicBending(
            tension_N=1e3,
            amplitude_1_m=0.1,
            frequency_Hz=0.1,
            anchoring_factor={"inner_armour": -1.0},
        )


def test_history_starts_straight_and_ends_with_a_full_cycle() -> None:
    kappa, last = RIG.history()
    assert kappa[0] == 0.0
    assert kappa.size - 1 - last == RIG.points_per_cycle
    assert kappa[last:].max() == pytest.approx(2 * RIG.amplitude_1_m)
    reversed_bending = CyclicBending(tension_N=0.0, amplitude_1_m=0.1, frequency_Hz=1.0)
    kappa, last = reversed_bending.history()
    assert kappa[0] == 0.0 and kappa[last] == pytest.approx(-0.1)


# --------------------------------------------------------------------------- #
# Coupled heating                                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.analytical_sanity
def test_without_friction_the_network_transient_is_reproduced() -> None:
    cable = _iea()
    net = _network(cable)
    t = np.linspace(0.0, 6 * 3600.0, 13)
    current = np.where(t < 3 * 3600.0, 600.0, 300.0)
    reports: list[tuple[float, str]] = []
    result = simulate_frictional_heating(
        cable,
        net,
        RIG,
        t,
        current_A=current,
        ambient_C=20.0,
        coupling="none",
        progress=lambda share, text: reports.append((share, text)),
    )
    assert [share for share, _ in reports] == pytest.approx(np.arange(1, 13) / 12)
    assert reports[-1][1] == "Interval 12 of 12"
    reference = net.transient(t, current, ambient_C=20.0, initial="ambient")
    assert np.max(np.abs(result.node_temperature_C - reference)) < 1e-9
    assert np.all(result.total_friction_heat_W_m == 0.0)
    joule = net.n_conductors * net.conductor_loss_W_m(600.0, 60.0)
    assert result.conductor_heat_W_m[0] == pytest.approx(joule, rel=0.5)


@pytest.mark.analytical_sanity
def test_passive_cores_keep_their_heat_capacity_and_share_the_core_mean() -> None:
    # The rig heated two of the three cores (Hoyle et al. 2023).
    cable = _iea()
    full = _network(cable)
    two = thermal_network(
        cable,
        630e-6,
        n_cores=2,
        T4_external_Km_W=1.0 / (math.pi * 0.184 * 8.0),
        ambient_C=20.0,
    )
    assert (full.passive_cores, two.passive_cores) == (0, 1)
    assert two.capacitance_J_mK.sum() == pytest.approx(
        full.capacitance_J_mK.sum(), rel=1e-12
    )
    nodes = two.steady_state(600.0, ambient_C=20.0)
    plain = two.layer_temperatures(nodes)
    mean = two.core_mean_temperatures(nodes)
    bundle = nodes[int(np.flatnonzero(~two.per_core)[0])]
    for name in plain:
        column = two.layer_weights[:, two.layer_names.index(name)]
        if np.any(column[~two.per_core] != 0.0):
            assert mean[name] == plain[name]
        else:
            assert mean[name] == pytest.approx((2 * plain[name] + bundle) / 3)
            assert mean[name] < plain[name]
    assert full.core_mean_temperatures(nodes[: full.n_nodes]).keys() == plain.keys()


@pytest.mark.analytical_sanity
def test_steady_friction_heat_leaves_through_the_surface() -> None:
    cable = _iea(0.30e6)
    net = _network(cable)
    t = np.linspace(0.0, 30 * 86400.0, 31)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = simulate_frictional_heating(
            cable,
            net,
            RIG,
            t,
            current_A=0.0,
            ambient_C=20.0,
            coupling="one_way",
            max_step_s=3600.0,
        )
    heat = {name: values[0] for name, values in result.friction_heat_W_m.items()}
    q = np.zeros(net.n_nodes)
    sources = {"inner_armour": "inner_armour", "outer_armour": "outer_armour"}
    sources[CORE_BUNDLE] = "mdpe_core_sheath"
    for name, layer in sources.items():
        q += heat[name] * net.source_weights(layer)
    k, s = net.system_matrices(0.0, ambient_C=20.0)
    steady = np.linalg.solve(k, s + q)
    assert np.max(np.abs(result.node_temperature_C[-1] - steady)) < 1e-3
    surface = (steady[-1] - 20.0) / net.external_resistance_Km_W
    # To the conditioning of the 0.10 mm copper tape screens.
    assert surface == pytest.approx(sum(heat.values()), rel=1e-8)


def test_heater_holds_the_set_point_until_friction_takes_over() -> None:
    cable = _iea(0.30e6)
    net = _network(cable)
    t = np.arange(0.0, 8 * 3600.0 + 1.0, 600.0)
    set_point = 20.0 + 50.0 * np.minimum(t / 7200.0, 1.0)
    active = t >= 7200.0
    plain = simulate_frictional_heating(
        cable,
        net,
        RIG,
        t,
        conductor_temperature_C=set_point,
        bending_active=active,
        ambient_C=20.0,
        coupling="none",
    )
    assert np.allclose(plain.node_temperature_C[:, 0], set_point)
    assert np.all(plain.conductor_heat_W_m[1:] > 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        heated = simulate_frictional_heating(
            cable,
            net,
            RIG,
            t,
            conductor_temperature_C=set_point,
            bending_active=active,
            ambient_C=20.0,
            coupling="one_way",
        )
    # The friction heat alone keeps the conductor above its set point.
    assert heated.node_temperature_C[-1, 0] > set_point[-1] + 1.0
    assert heated.conductor_heat_W_m[-2] == 0.0
    sheath = "mdpe_inner_sheath"
    assert (
        heated.layer_temperature_C[sheath][-1]
        > plain.layer_temperature_C[sheath][-1] + 10.0
    )


def test_two_way_heating_loosens_the_armour_and_converges() -> None:
    cable = _iea(0.30e6)
    net = _network(cable)
    t = np.arange(0.0, 3 * 3600.0 + 1.0, 600.0)
    finals = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        one_way = simulate_frictional_heating(
            cable, net, RIG, t, current_A=0.0, ambient_C=20.0, coupling="one_way"
        )
        for interval in (1800.0, 900.0, 450.0):
            result = simulate_frictional_heating(
                cable,
                net,
                RIG,
                t,
                current_A=0.0,
                ambient_C=20.0,
                coupling="two_way",
                update_interval_s=interval,
            )
            finals.append(result.layer_temperature_C["inner_armour"][-1])
    # Self-heating of the armour lowers its clamping and so its own heat.
    pressure = result.contact_pressure_Pa["inner_armour"]
    assert pressure[-2] < pressure[0]
    assert result.total_friction_heat_W_m[-2] < one_way.total_friction_heat_W_m[-2]
    assert finals[-1] < one_way.layer_temperature_C["inner_armour"][-1]
    # The staggered loop converges as the update interval shrinks.
    assert abs(finals[2] - finals[1]) < abs(finals[1] - finals[0])
    assert abs(finals[2] - finals[1]) < 0.5


def test_simulation_rejects_bad_input() -> None:
    cable = _iea()
    net = _network(cable)
    t = np.array([0.0, 60.0])
    with pytest.raises(CableInputError):
        simulate_frictional_heating(cable, net, RIG, t, coupling="none")
    with pytest.raises(CableInputError):
        simulate_frictional_heating(
            cable, net, RIG, t, current_A=0.0, conductor_temperature_C=20.0
        )
    with pytest.raises(CableInputError):
        simulate_frictional_heating(
            cable, net, RIG, t, current_A=0.0, coupling="three_way"
        )
    with pytest.raises(CableInputError):
        simulate_frictional_heating(
            cable, net, RIG, t, current_A=0.0, bending_active=[True]
        )
