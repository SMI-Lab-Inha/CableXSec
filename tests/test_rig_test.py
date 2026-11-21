# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Rig-test simulation: limits, decomposition and gauge planning."""

import math

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.rig_test import (
    GaugeLocation,
    RigProtocol,
    RigSetup,
    default_gauge_grid,
    end_rotation_profile,
    simulate_rig_test,
)
from cablexsec.cable_adapters import cable_to_mkappa_specs
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.clamp_layer import HAVE_NUMBA
from cablexsec.mechanics.fast_slip import wire_friction_stress
from cablexsec.mechanics.radial_cable import solve_cable_radial_state
from cablexsec.mechanics.wire_stress import local_bending_fibre_stress

#: The anchored-wire solver runs pure Python without numba (minutes per case).
needs_numba = pytest.mark.skipif(
    not HAVE_NUMBA, reason="anchored-wire solver without numba is slow"
)


@pytest.fixture(scope="module")
def menard():
    return cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)


@pytest.fixture(scope="module")
def three_core():
    return cx.load_cable(cx.inter_array_three_core_yaml_path(), geometry_override=None)


def _protocol(amplitude=0.1, tension=50e3, cycles=2, **kw):
    return RigProtocol.cyclic(
        tension_N=tension,
        curvature_amplitude_1_m=amplitude,
        n_cycles=cycles,
        samples_per_cycle=40,
        **kw,
    )


def _pitch(cable, name):
    layer = cable.get_layer(name)
    return 2 * math.pi * layer.r_helix_m / math.tan(layer.lay_angle_rad)


@pytest.fixture(scope="module")
def menard_run(menard):
    pitch = _pitch(menard, "outer_armour")
    length = 8 * pitch
    gauges = [
        GaugeLocation(layer, angle, x)
        for layer in ("inner_armour", "outer_armour")
        for angle in (0.0, 45.0, 90.0, 180.0)
        for x in (0.02, length / 2)
    ]
    return simulate_rig_test(
        menard,
        _protocol(amplitude=0.03),
        RigSetup(free_length_m=length),
        gauges=gauges,
        mu_factors=(1.25,),
        setback_variants_m=(2.0 * pitch,),
    )


@needs_numba
def test_parts_add_up_and_far_field_is_sectional(menard, menard_run):
    res = menard_run
    for g in res.gauges:
        assert np.allclose(g.total, g.axial + g.local_bending + g.friction)
    # The far field needs many clamp boundary layers on both sides. With the
    # armour on a stiff support (xi 0.69 / 0.89) the outer armour's layer is
    # 1.34 m (inner 0.63 m), so the 8-pitch specimen of ``menard_run``
    # (6.2 m) leaves mid-span only 2.3 layers from each clamp; 16 pitches
    # leave 4.6.
    length = 16 * _pitch(menard, "outer_armour")
    far = simulate_rig_test(
        menard,
        _protocol(amplitude=0.03),
        RigSetup(free_length_m=length),
        gauges=[
            GaugeLocation(layer, angle, length / 2)
            for layer in ("inner_armour", "outer_armour")
            for angle in (0.0, 45.0, 90.0, 180.0)
        ],
    )
    caps = {s.layer: s.sectional_cap_microstrain for s in far.layers}
    assert max(s.boundary_layer_m for s in far.layers) < length / 2 / 4
    for g in far.gauges:
        diff = np.abs(g.friction - g.friction_sectional).max()
        assert diff < 0.05 * caps[g.location.layer]
        assert abs(g.clamp_contrast) < 0.05 * 2 * caps[g.location.layer]


@needs_numba
def test_clamp_increment_at_neutral_axis_near_clamp(menard_run):
    res = menard_run
    for scale in res.layers:
        if scale.a_clamp < 3:
            continue
        near = [
            g
            for g in res.gauges
            if g.location.layer == scale.layer
            and g.location.x_m < 0.1
            and abs(abs(g.angle_deg) - 90) < 30
        ]
        assert near
        best = max(near, key=lambda g: g.clamp_contrast)
        assert best.ranges["friction"] > 3 * (best.range_sectional + 1.0)
        # the cyclic half range stays below the averaged clamp-law peak plus margin
        assert best.ranges["friction"] / 2 < 1.5 * scale.clamp_law_peak_microstrain


@needs_numba
def test_local_bending_and_axial_parts(menard, menard_run):
    res = menard_run
    specs = {s.layer: s for s in res.layers}
    state0 = solve_cable_radial_state(menard, 0.0)
    for g in res.gauges[:4]:
        layer = menard.get_layer(g.location.layer)
        E = cx.cable_to_mkappa_specs(menard)[0].E_wire_Pa
        expected = (
            local_bending_fibre_stress(
                res.reference_curvature_1_m,
                0.0,
                math.radians(g.angle_deg),
                E,
                layer.d_wire_m,
                layer.lay_angle_rad,
                0.0,
            )
            / E
            * 1e6
        )
        # The normal curvature follows the slip state: the friction strain the
        # wire carries adds (d/2)(sin^2 a / r) times it at the outer face.
        expected = expected + (
            layer.d_wire_m
            / 2
            * math.sin(layer.lay_angle_rad) ** 2
            / layer.r_helix_m
            * np.asarray(g.friction)
        )
        assert np.allclose(g.local_bending, expected, atol=1e-9)
        tension = 50e3
        state = solve_cable_radial_state(menard, tension)
        change = (
            (
                state.result.wire_stress_Pa[layer.name]
                - state0.result.wire_stress_Pa[layer.name]
            )
            / E
            * 1e6
        )
        assert g.axial[-1] == pytest.approx(change, rel=1e-6)
        assert specs[layer.name].axial_microstrain_at_reference == pytest.approx(
            change, rel=1e-6
        )


def test_rotated_sectional_friction_matches_package_wire(menard):
    spec = cable_to_mkappa_specs(menard, 5e4)[0]
    kappa = 0.08 * np.sin(np.linspace(0, 4 * np.pi, 120))
    direct = wire_friction_stress(spec, kappa, np.zeros_like(kappa))
    k = 7
    theta = spec.phi_wires[k]
    rotated = wire_friction_stress(
        spec,
        kappa * math.cos(theta),
        -kappa * math.sin(theta),
        initial_bending_angle=-theta,
    )[0]
    assert np.allclose(rotated, direct[k], rtol=1e-9, atol=1e-3)


@needs_numba
def test_mu_and_setback_sensitivity(menard_run):
    res = menard_run
    assert len(res.mu_values) == 2
    g = res.gauges[0]
    assert len(g.range_by_mu) == 2
    assert len(g.range_by_setback) == 2
    near = [
        x
        for x in res.gauges
        if x.location.x_m < 0.1 and abs(abs(x.angle_deg) - 90) < 30
    ]
    contrast = max(near, key=lambda x: x.clamp_contrast)
    setbacks = sorted(contrast.range_by_setback)
    # an anchor set back over two pitches of straight cable removes most of the clamp effect
    assert (
        contrast.range_by_setback[setbacks[1]] < contrast.range_by_setback[setbacks[0]]
    )


def test_clamp_none_is_sectional(menard):
    res = simulate_rig_test(
        menard,
        _protocol(cycles=1),
        RigSetup(free_length_m=1.5, clamp="none"),
        gauges=[GaugeLocation("outer_armour", 90.0, 0.02)],
        mu_factors=(),
    )
    g = res.gauges[0]
    assert np.array_equal(g.friction, g.friction_sectional)
    assert all(np.all(s.moment_clamp_Nm == 0) for s in res.stations)
    assert np.allclose(res.stations[0].moment_Nm, res.stations[1].moment_Nm)
    assert any("sectional" in n for n in res.notes)


@needs_numba
def test_station_loop_is_sum(menard_run):
    for s in menard_run.stations:
        assert np.allclose(s.moment_Nm, s.moment_sectional_Nm + s.moment_clamp_Nm)
    assert np.ptp(menard_run.stations[0].moment_sectional_Nm) > 0


@needs_numba
def test_recommendations_and_summary(menard_run):
    res = menard_run
    purposes = [r.purpose for r in res.recommendations]
    assert any("clamp boundary layer" in p for p in purposes)
    assert any("far field" in p for p in purposes)
    for r in res.recommendations:
        assert r.signal_to_resolution == pytest.approx(
            r.amplitude_microstrain / res.gauge_resolution_microstrain
        )
    text = res.summary()
    assert "outer_armour" in text and "resolution" in text
    assert res.gauge("outer_armour", 90.0, 0.0).location.x_m == pytest.approx(0.02)
    assert res.provenance.limitations


def test_end_rotation_profile_closed_forms():
    length = 4.0
    x = np.linspace(0, length, 9)
    assert np.allclose(
        end_rotation_profile(x, length, math.inf), -4 / length + 6 * x / length**2
    )
    stiff = end_rotation_profile(x, length, 19.9 * length)
    assert np.allclose(
        stiff, end_rotation_profile(x, length, math.inf), atol=0.02 * 4 / length
    )
    lam = length / 50
    assert end_rotation_profile(np.array([0.0]), length, lam)[0] * lam == pytest.approx(
        -1, rel=0.03
    )
    # boundary conditions by integration: w'(L) - w'(0) = int kappa = -1 and w(L) = 0
    xs = np.linspace(0, length, 20001)
    k = end_rotation_profile(xs, length, 0.7)
    slope = 1 + np.concatenate([[0], np.cumsum(0.5 * (k[1:] + k[:-1]) * np.diff(xs))])
    assert slope[-1] == pytest.approx(0.0, abs=1e-6)
    w = np.concatenate([[0], np.cumsum(0.5 * (slope[1:] + slope[:-1]) * np.diff(xs))])
    assert w[-1] == pytest.approx(0.0, abs=1e-6)


@needs_numba
def test_end_rotation_run_and_temperature(three_core):
    proto = RigProtocol.cyclic(
        tension_N=100e3,
        end_rotation_amplitude_rad=0.05,
        n_cycles=1,
        samples_per_cycle=24,
        temperature_C=40.0,
    )
    gauges = [
        GaugeLocation("outer_armour", 90.0, 0.05),
        GaugeLocation("outer_armour", 0.0, 2.0),
    ]
    res = simulate_rig_test(
        three_core,
        proto,
        RigSetup(free_length_m=4.0, profile="end_rotation"),
        gauges=gauges,
        mu_factors=(),
        tension_levels=2,
    )
    assert res.contact_model == "solved"
    assert res.bending_stiffness_Nm2 is not None
    lam = math.sqrt(res.bending_stiffness_Nm2 / 100e3)
    peak = np.abs(res.reference_curvature_1_m).max()
    assert peak == pytest.approx(
        0.05 * abs(end_rotation_profile(np.array([0.0]), 4.0, lam)[0])
    )
    # the far clamp adds a chord rotation of order lam / L to the free-end value theta / lam
    assert 0.05 / lam < peak < 0.05 / lam * (1 + 3 * lam / 4.0)
    assert res.curvature_shape[0] == pytest.approx(1.0)
    cold = solve_cable_radial_state(three_core, 100e3).armour_contact_pressure()[
        "outer_armour"
    ]
    hot = next(s for s in res.layers if s.layer == "outer_armour").contact_pressure_Pa
    assert hot > cold


@needs_numba
def test_bounded_fallback_and_errors(menard):
    export = cx.load_cable(cx.export_three_core_yaml_path(), geometry_override=None)
    gauges = [GaugeLocation("outer_armour", 90.0, 0.05)]
    res = simulate_rig_test(
        export,
        _protocol(cycles=1),
        RigSetup(free_length_m=2.0),
        gauges=gauges,
        mu_factors=(),
    )
    assert res.contact_model == "bounded" and res.notes
    with pytest.raises(CableInputError):
        simulate_rig_test(
            export,
            _protocol(cycles=1, temperature_C=40.0),
            RigSetup(free_length_m=2.0),
            gauges=gauges,
            mu_factors=(),
        )
    setup = RigSetup(free_length_m=2.0)
    with pytest.raises(CableInputError):
        simulate_rig_test(
            menard, _protocol(), setup, gauges=[GaugeLocation("nope", 0, 0.1)]
        )
    with pytest.raises(CableInputError):
        simulate_rig_test(
            menard, _protocol(), setup, gauges=[GaugeLocation("outer_armour", 0, 3.0)]
        )
    with pytest.raises(CableInputError):
        simulate_rig_test(menard, _protocol(), setup, contact_model="qp")
    with pytest.raises(CableInputError):
        simulate_rig_test(menard, _protocol(), setup, mu_factors=(-1,))
    rot = RigProtocol.cyclic(tension_N=1e4, end_rotation_amplitude_rad=0.1)
    with pytest.raises(CableInputError):
        simulate_rig_test(menard, rot, setup)
    with pytest.raises(CableInputError):
        simulate_rig_test(menard, _protocol(), setup, stations_m=[5.0])


def test_protocol_and_setup_validation():
    t = np.linspace(0, 1, 5)
    with pytest.raises(CableInputError):
        RigProtocol(time_s=t, tension_N=np.ones(5))
    with pytest.raises(CableInputError):
        RigProtocol(time_s=t[::-1], tension_N=np.ones(5), curvature_1_m=np.zeros(5))
    with pytest.raises(CableInputError):
        RigProtocol(time_s=t, tension_N=-np.ones(5), curvature_1_m=np.zeros(5))
    with pytest.raises(CableInputError):
        RigProtocol(time_s=t, tension_N=np.ones(4), curvature_1_m=np.zeros(5))
    with pytest.raises(CableInputError):
        RigProtocol(time_s=t, tension_N=1.0, curvature_1_m=0.0, retain_from_s=2.0)
    p = RigProtocol(time_s=t, tension_N=1.0, curvature_1_m=0.0)
    assert p.tension_N.shape == (5,) and not p.tension_N.flags.writeable
    with pytest.raises(CableInputError):
        RigProtocol.cyclic(tension_N=1.0)
    c = RigProtocol.cyclic(
        tension_N=1e3, curvature_amplitude_1_m=0.1, n_cycles=3, period_s=2.0
    )
    assert c.retain_from_s == pytest.approx(6.0)
    assert np.ptp(c.curvature_1_m[c.time_s >= c.retain_from_s]) == pytest.approx(
        0.2, rel=1e-3
    )
    for kw in (
        {"free_length_m": 0},
        {"free_length_m": 1, "profile": "x"},
        {"free_length_m": 1, "clamp": "x"},
        {"free_length_m": 1, "profile": "table"},
        {"free_length_m": 1, "reference_x_m": 2.0},
        {"free_length_m": 1, "bending_stiffness_Nm2": -1.0},
        {
            "free_length_m": 1,
            "profile": "table",
            "profile_x_m": [0, 1],
            "profile_shape": [0, 0],
        },
    ):
        with pytest.raises(CableInputError):
            RigSetup(**kw)


@needs_numba
def test_table_profile_and_default_grid(menard):
    setup = RigSetup(
        free_length_m=2.0,
        profile="table",
        profile_x_m=np.array([0.0, 1.0, 2.0]),
        profile_shape=np.array([0.0, 2.0, 0.0]),
    )
    grid = default_gauge_grid(menard, setup)
    assert {g.layer for g in grid} == {"inner_armour", "outer_armour"}
    assert all(0.05 <= g.x_m <= 1.0 for g in grid)
    res = simulate_rig_test(
        menard,
        _protocol(cycles=1),
        setup,
        gauges=[GaugeLocation("outer_armour", 0.0, 1.0)],
        mu_factors=(),
    )
    assert res.reference_x_m == pytest.approx(1.0)
    assert res.curvature_shape.max() == pytest.approx(1.0)


def test_public_exports():
    for name in (
        "simulate_rig_test",
        "RigProtocol",
        "RigSetup",
        "GaugeLocation",
        "RigTestResult",
    ):
        assert name in cx.__all__ and hasattr(cx, name)
