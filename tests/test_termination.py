# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Termination (clamp) correction of the fatigue friction stress."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.fatigue import compute_fatigue, wire_stress_field
from cablexsec.analysis.termination import (
    TerminationSpec,
    principal_direction,
    termination_friction_increment,
)
from cablexsec.exceptions import CableInputError, PhysicalRangeError
from cablexsec.mechanics.clamp_layer import (
    HAVE_NUMBA,
    clamp_amplification,
    clamp_mean_force,
)
from cablexsec.mechanics.papailiou import slip_onset_curvature

#: The anchored-wire solver runs pure Python without numba (minutes per case).
needs_numba = pytest.mark.skipif(
    not HAVE_NUMBA, reason="anchored-wire solver without numba is slow"
)


@pytest.fixture(scope="module")
def menard():
    return cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)


@pytest.fixture(scope="module")
def layer(menard):
    return cx.cable_to_fatigue_layers(menard, contact_tension_N=50e3)[0]


@pytest.fixture(scope="module")
def kappa_c(layer):
    return slip_onset_curvature(layer.spec)


def _cyclic(amplitude, cycles=2, samples=60):
    t = np.linspace(0.0, 2 * math.pi * cycles, cycles * samples + 1)
    return amplitude * np.sin(t)


def test_kappa_c_and_unit_match_package(layer, kappa_c):
    k = 5 * kappa_c * np.array([0.0, 1.0])
    inc = termination_friction_increment(layer.spec, k, 0 * k, TerminationSpec(0.0))
    assert inc.kappa_c_1_m == pytest.approx(kappa_c, rel=1e-12)
    spec = layer.spec
    sin_a = abs(math.sin(spec.lay_angle_rad))
    width = 2 * math.pi * spec.r_helix_m / spec.n_wires
    sigma0 = (
        spec.r_helix_m
        * spec.contact_pressure_Pa
        * width
        * math.cos(spec.lay_angle_rad)
        / (spec.A_wire_m2 * sin_a**2)
    )
    assert inc.sigma_u_Pa == pytest.approx(sigma0 * spec.mu * sin_a, rel=1e-12)


def test_none_and_far_field_are_bitwise_sectional(layer, kappa_c):
    kx = _cyclic(20 * kappa_c)
    ky = 8 * kappa_c * np.sin(1.7 * np.linspace(0.0, 4 * math.pi, kx.size))
    base = wire_stress_field(layer, kx, ky)
    for spec in (TerminationSpec(0.1, anchor="none"), TerminationSpec(50.0)):
        field = wire_stress_field(layer, kx, ky, termination=spec)
        assert not field.termination.applied
        assert np.array_equal(field.mean_Pa, base.mean_Pa)
        assert np.array_equal(field.radial_Pa, base.radial_Pa)
        assert np.array_equal(field.tangential_Pa, base.tangential_Pa)
        assert not np.any(field.termination.stress_Pa)
    assert base.termination is None


@needs_numba
def test_increment_decays_to_round_off_within_reach(layer, kappa_c):
    """Increment vanishes a few boundary layers from the anchor (uniform curvature)."""
    kx = _cyclic(20 * kappa_c)
    probe = termination_friction_increment(layer.spec, kx, 0 * kx, TerminationSpec(0.0))
    ell = probe.boundary_layer_m
    unit = probe.sigma_u_Pa * math.pi / 2
    peaks = []
    for d in (0.0, 1.0, 2.0, 4.0, 7.0):
        inc = termination_friction_increment(
            layer.spec, kx, 0 * kx, TerminationSpec(d * ell)
        )
        assert inc.applied
        peaks.append(float(np.abs(inc.stress_Pa).max()) / unit)
    assert peaks[0] > 1.0  # the anchored wire carries more than the sectional cap
    assert peaks[0] > peaks[1] > peaks[2] > peaks[3]
    assert peaks[3] < 0.15
    assert peaks[4] < 1e-2
    # The discrete layer has finite support (about 6-9 l_x, growing slowly
    # with mesh refinement); just inside the solved reach only round-off is
    # left because the far-field run uses the same solver and phases.
    edge = termination_friction_increment(
        layer.spec, kx, 0 * kx, TerminationSpec(0.97 * probe.reach_m)
    )
    assert edge.applied
    assert float(np.abs(edge.stress_Pa).max()) / unit < 1e-9


@needs_numba
def test_far_field_damage_equals_sectional(layer, kappa_c):
    kx = _cyclic(20 * kappa_c)
    ky = 0.3 * kx
    base = compute_fatigue([layer], kx, ky)
    probe = termination_friction_increment(layer.spec, kx, ky, TerminationSpec(0.0))
    near_reach = 0.95 * probe.reach_m
    res = compute_fatigue([layer], kx, ky, termination=TerminationSpec(near_reach))
    assert res.clamp_correction[layer.name]["applied"]
    np.testing.assert_allclose(
        res.D_bulk_per_fibre[layer.name],
        base.D_bulk_per_fibre[layer.name],
        rtol=1e-9,
        atol=0,
    )
    clamped = compute_fatigue([layer], kx, ky, termination=TerminationSpec(0.0))
    assert clamped.D_total_per_wire[layer.name].max() > (
        2 * base.D_total_per_wire[layer.name].max()
    )
    assert "anchored-wire clamp increment" in clamped.provenance.method


@needs_numba
def test_clamp_law_at_the_anchor(layer, kappa_c):
    """Monotonic uniform bending, a = 100: clamp law ``A pi / 2`` at the anchor.

    The peak friction force ``pi / 2 + |H0|`` sits on the wire at the neutral
    axis at the anchor, the worst wire of the clamp law.
    """
    a = 100.0
    kx = np.linspace(0.0, a * kappa_c, 401)
    inc = termination_friction_increment(layer.spec, kx, 0 * kx, TerminationSpec(0.0))
    n_w = layer.spec.n_wires
    theta = 2 * math.pi * np.arange(n_w) / n_w
    far = np.sign(np.cos(theta)) * np.arcsin(np.abs(np.cos(theta)))  # full slip
    n_anchor = inc.stress_Pa[:, -1] / inc.sigma_u_Pa + far
    peak = float(np.max(np.abs(n_anchor)))
    assert peak == pytest.approx(float(clamp_amplification(a)) * math.pi / 2, rel=0.05)
    assert peak == pytest.approx(math.pi / 2 + float(clamp_mean_force(a)), rel=0.05)
    neutral = int(np.argmin(np.abs(np.cos(theta))))
    assert abs(n_anchor[neutral]) > 0.95 * peak


def test_stick_regime_has_no_increment(layer, kappa_c):
    kx = _cyclic(0.8 * kappa_c)
    inc = termination_friction_increment(layer.spec, kx, 0 * kx, TerminationSpec(0.0))
    assert inc.applied
    assert np.abs(inc.stress_Pa).max() < 1e-9 * inc.sigma_u_Pa


@needs_numba
def test_rotation_shifts_wires(layer, kappa_c):
    """A planar history rotated by one wire spacing moves the increment one wire."""
    n_w = layer.spec.n_wires
    beta = 0.3
    k = _cyclic(15 * kappa_c)
    first = termination_friction_increment(
        layer.spec, k * math.cos(beta), k * math.sin(beta), TerminationSpec(0.05)
    )
    rot = beta + 2 * math.pi / n_w
    second = termination_friction_increment(
        layer.spec, k * math.cos(rot), k * math.sin(rot), TerminationSpec(0.05)
    )
    assert first.principal_fraction == pytest.approx(1.0)
    np.testing.assert_allclose(
        second.stress_Pa,
        np.roll(first.stress_Pa, 1, axis=0),
        atol=1e-9 * first.sigma_u_Pa,
    )


@needs_numba
def test_biaxial_modes(layer, kappa_c):
    k = _cyclic(15 * kappa_c)
    planar = [
        termination_friction_increment(
            layer.spec, k, 0 * k, TerminationSpec(0.05, biaxial=mode)
        )
        for mode in ("principal", "components")
    ]
    np.testing.assert_allclose(
        planar[0].stress_Pa, planar[1].stress_Pa, atol=1e-9 * planar[0].sigma_u_Pa
    )
    ky = 7 * kappa_c * np.sin(1.7 * np.linspace(0.0, 4 * math.pi, k.size))
    principal, components = (
        termination_friction_increment(
            layer.spec, k, ky, TerminationSpec(0.05, biaxial=mode)
        )
        for mode in ("principal", "components")
    )
    beta, fraction = principal_direction(k, ky)
    assert principal.directions_rad == (beta,)
    assert components.directions_rad == (beta, beta + math.pi / 2)
    assert principal.principal_fraction == pytest.approx(fraction)
    assert 0.5 < fraction < 1.0
    assert not np.allclose(principal.stress_Pa, components.stress_Pa)
    fixed = termination_friction_increment(
        layer.spec, k, ky, TerminationSpec(0.05, direction_rad=0.0)
    )
    assert fixed.directions_rad == (0.0,)
    assert fixed.principal_fraction == pytest.approx(
        np.var(k) / (np.var(k) + np.var(ky))
    )


@needs_numba
def test_profile_pressure_and_hand_options(layer, kappa_c):
    k = _cyclic(15 * kappa_c)
    base = termination_friction_increment(layer.spec, k, 0 * k, TerminationSpec(0.1))
    flat = termination_friction_increment(
        layer.spec,
        k,
        0 * k,
        TerminationSpec(0.1, curvature_profile=(np.array([0.0, 5.0]), np.full(2, 3.0))),
    )
    assert np.array_equal(base.stress_Pa, flat.stress_Pa)
    same_pressure = termination_friction_increment(
        layer.spec,
        k,
        0 * k,
        TerminationSpec(0.1),
        contact_pressure_Pa=np.full(k.size, layer.spec.contact_pressure_Pa),
    )
    assert np.array_equal(base.stress_Pa, same_pressure.stress_Pa)
    # Curvature rising from zero at the clamp: smaller increment than uniform.
    ramp = termination_friction_increment(
        layer.spec,
        k,
        0 * k,
        TerminationSpec(
            0.5, curvature_profile=(np.array([0.0, 0.5]), np.array([0.0, 1.0]))
        ),
    )
    uniform = termination_friction_increment(layer.spec, k, 0 * k, TerminationSpec(0.5))
    assert np.abs(ramp.stress_Pa).max() < np.abs(uniform.stress_Pa).max()
    reversed_hand = termination_friction_increment(
        layer.spec, k, 0 * k, TerminationSpec(0.1, axis_sign=-1)
    )
    assert np.abs(reversed_hand.stress_Pa).max() == pytest.approx(
        np.abs(base.stress_Pa).max(), rel=0.1
    )
    # Set-back moves the anchor away from the curved section.
    setback = termination_friction_increment(
        layer.spec, k, 0 * k, TerminationSpec(0.1, setback_m=0.5)
    )
    assert np.abs(setback.stress_Pa).max() < np.abs(base.stress_Pa).max()
    frictionless = termination_friction_increment(
        replace(layer.spec, mu=0.0), k, 0 * k, TerminationSpec(0.0)
    )
    assert not frictionless.applied


@needs_numba
def test_cable_api_and_mapping(menard, kappa_c):
    k = _cyclic(12 * kappa_c)
    base = cx.compute_fatigue(menard, k, 0.2 * k)
    res = cx.compute_fatigue(
        menard, k, 0.2 * k, termination={"inner_armour": TerminationSpec(0.0)}
    )
    assert set(res.clamp_correction) == {"inner_armour"}
    assert np.array_equal(
        res.D_bulk_per_fibre["outer_armour"], base.D_bulk_per_fibre["outer_armour"]
    )
    assert not np.array_equal(
        res.D_bulk_per_fibre["inner_armour"], base.D_bulk_per_fibre["inner_armour"]
    )
    layers = cx.cable_to_fatigue_layers(menard)
    hands = {lyr.name: lyr.lay_hand for lyr in layers}
    expected = {
        lyr.name: 1 if lyr.lay_direction == "right_hand" else -1
        for lyr in menard.layers
        if lyr.name in hands
    }
    assert hands == expected
    with pytest.raises(CableInputError):
        compute_fatigue(layers, k, 0 * k, termination={"nope": TerminationSpec(0.0)})
    with pytest.raises(CableInputError):
        compute_fatigue(layers, k, 0 * k, termination={"inner_armour": 1.0})
    assert "TerminationSpec" in cx.__all__ and cx.TerminationSpec is TerminationSpec


def test_spec_validation():
    for kwargs in (
        {"distance_m": -1.0},
        {"distance_m": 0.0, "setback_m": -0.1},
        {"distance_m": 0.0, "anchor": "bonded"},
        {"distance_m": 0.0, "biaxial": "vector"},
        {"distance_m": 0.0, "axis_sign": 0},
        {"distance_m": 0.0, "segments_per_pitch": 4},
        {"distance_m": 0.0, "max_increment": 0.0},
        {"distance_m": 0.0, "reach_lengths": -1.0},
        {"distance_m": math.nan},
        {"distance_m": 0.0, "curvature_profile": (np.array([0.0]), np.array([1.0]))},
        {"distance_m": 0.0, "curvature_profile": (np.array([1.0, 0.5]), np.ones(2))},
        {
            "distance_m": 1.0,
            "curvature_profile": (np.array([0.0, 2.0]), np.array([1.0, 1.0]) * [1, -1]),
        },
    ):
        with pytest.raises((CableInputError, PhysicalRangeError)):
            TerminationSpec(**kwargs)
    with pytest.raises(CableInputError):
        TerminationSpec(0.0, curvature_profile=(np.array([0.0, 1.0]), np.zeros(2)))
