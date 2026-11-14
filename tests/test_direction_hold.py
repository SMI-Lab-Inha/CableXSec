# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Slip-cap bending direction is held below a fraction of slip-onset curvature.

Near zero curvature atan2(kappa_y, kappa_x) is not a physical direction. A
perturbation of 1e-12 1/m must not re-orient the Papailiou cap and clip the
locked friction stress (defect found in the October 2026 audit).
"""

import math
from dataclasses import replace

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.cable_adapters import cable_to_fatigue_layers
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.fast_slip import wire_friction_stress
from cablexsec.mechanics.papailiou import (
    DIRECTION_HOLD_FRACTION,
    direction_threshold_curvature,
    slip_onset_curvature,
    wire_slip_history,
)


def _specs():
    cable = cx.load_cable(cx.export_three_core_yaml_path(), geometry_override=None)
    return [layer.spec for layer in cable_to_fatigue_layers(cable, contact_tension_N=0)]


def _paths(spec, kx, ky, f, pressure=None):
    ref = wire_slip_history(
        spec, kx, ky, contact_pressure_Pa=pressure, direction_hold_fraction=f
    ).stress_Pa
    fast = wire_friction_stress(
        spec, kx, ky, contact_pressure_Pa=pressure, direction_hold_fraction=f
    )
    return ref, fast


def _load_unload_reload(kappa_max):
    up = np.linspace(0.0, kappa_max, 50)
    down = np.linspace(kappa_max, 0.0, 50)[1:]
    kx = np.concatenate([up, down, up[1:]])
    return kx, len(up) + len(down) - 1


def test_onset_curvature_formula_and_virgin_onset():
    for spec in _specs():
        sin_a = abs(math.sin(spec.lay_angle_rad))
        q = spec.contact_pressure_Pa * 2 * math.pi * spec.r_helix_m / spec.n_wires
        q *= math.cos(spec.lay_angle_rad)
        g_min = 1.0 if spec.shape_g is None else float(np.min(spec.shape_g))
        expected = spec.mu * q * g_min / (spec.A_wire_m2 * sin_a * spec.k_f)
        onset = slip_onset_curvature(spec)
        assert onset == pytest.approx(expected, rel=1e-12)
        assert direction_threshold_curvature(spec) == pytest.approx(
            DIRECTION_HOLD_FRACTION * onset, rel=1e-12
        )
        # Dense uniform population: no wire slips below the bound, and the
        # first slip occurs just above it (neutral-axis wires).
        dense = replace(spec, n_wires=720, shape_g=None)
        onset = slip_onset_curvature(dense)
        kappa = np.linspace(0.0, 1.2 * onset, 1201)
        count = wire_slip_history(dense, kappa, np.zeros_like(kappa)).slipping_count
        first = kappa[np.argmax(count > 0)]
        # A grid point can fall one ulp below the onset (the outer export
        # armour does at index 1000); the neutral-axis wire then slips by
        # rounding. Exclude points within a relative 1e-12 of the bound.
        assert count[kappa < onset * (1.0 - 1.0e-12)].max() == 0
        assert onset * (1.0 - 1.0e-12) <= first <= 1.02 * onset


@pytest.mark.parametrize("f", [0.01, DIRECTION_HOLD_FRACTION, 0.5])
def test_perpendicular_perturbation_at_zero_changes_nothing(f):
    for spec in _specs():
        kx, i0 = _load_unload_reload(20 * slip_onset_curvature(spec))
        ky0 = np.zeros_like(kx)
        ky1 = ky0.copy()
        ky1[i0] = 1e-12
        a_ref, a_fast = _paths(spec, kx, ky0, f)
        b_ref, b_fast = _paths(spec, kx, ky1, f)
        scale = np.abs(a_ref).max()
        assert np.abs(a_ref[:, i0]).max() > 0.1 * scale  # locked residual
        # Only the elastic response to the 1e-12 1/m sample and the cap blend
        # weight 1e-12/kappa_th remain: the change is proportional to the input.
        threshold = direction_threshold_curvature(spec, f)
        tol = 10 * spec.k_f * spec.r_helix_m * 1e-12 + 2e-12 / threshold * scale
        assert np.abs(a_ref - b_ref).max() <= tol
        assert np.abs(a_ref - b_ref).max() <= 1e-6 * scale
        assert np.abs(b_ref - b_fast).max() <= 1e-9 * scale
        assert np.abs(a_ref - a_fast).max() <= 1e-9 * scale


@pytest.mark.parametrize("f", [0.01, DIRECTION_HOLD_FRACTION, 0.5])
def test_response_is_continuous_in_small_perpendicular_curvature(f):
    spec = _specs()[0]
    kx, i0 = _load_unload_reload(20 * slip_onset_curvature(spec))
    base = wire_friction_stress(spec, kx, np.zeros_like(kx), direction_hold_fraction=f)
    threshold = direction_threshold_curvature(spec, f)
    changes = []
    for eps in (1e-6, 1e-4, 1e-2):
        ky = np.zeros_like(kx)
        ky[i0] = eps * threshold
        changes.append(
            np.abs(
                wire_friction_stress(spec, kx, ky, direction_hold_fraction=f) - base
            ).max()
        )
    scale = np.abs(base).max()
    assert changes[-1] <= 0.05 * scale
    assert changes[0] <= 1e-5 * scale
    assert changes[0] <= changes[1] <= changes[2]


@pytest.mark.parametrize("f", [0.01, DIRECTION_HOLD_FRACTION, 0.5, 1.0])
def test_rotating_sub_threshold_path_stays_in_stick(f):
    # A plain direction hold would let wires on the held neutral axis slip
    # under rotating curvature below kappa_th; the blended cap must not.
    for spec in _specs():
        phase = np.linspace(0.0, 10.0 * math.pi, 401)
        amplitude = 0.8 * direction_threshold_curvature(spec, f)
        for kx, ky in (
            (amplitude * np.cos(phase), amplitude * np.sin(phase)),
            (amplitude * np.cos(phase), 0.5 * amplitude * np.sin(phase)),
        ):
            stick = (
                spec.k_f
                * spec.r_helix_m
                * (
                    np.cos(spec.phi_wires)[:, None] * kx
                    + np.sin(spec.phi_wires)[:, None] * ky
                )
            )
            for path in _paths(spec, kx, ky, f):
                np.testing.assert_allclose(
                    path, stick, rtol=0, atol=1e-9 * np.abs(stick).max()
                )
            old = _paths(spec, kx, ky, 0.0)[0]
            np.testing.assert_allclose(
                _paths(spec, kx, ky, f)[0], old, rtol=0, atol=1e-9 * np.abs(old).max()
            )


def _same(a, b):
    """Equal to rounding: two floating-point paths may differ in the last digit across platforms."""
    a, b = np.asarray(a), np.asarray(b)
    scale = float(np.abs(a).max()) if a.size else 0.0
    np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-12 * scale)


def test_legacy_fraction_zero_keeps_old_sensitivity():
    spec = _specs()[0]
    kx, i0 = _load_unload_reload(20 * slip_onset_curvature(spec))
    ky1 = np.zeros_like(kx)
    ky1[i0] = 1e-12
    a, _ = _paths(spec, kx, np.zeros_like(kx), 0.0)
    b, b_fast = _paths(spec, kx, ky1, 0.0)
    assert np.abs(a - b).max() > 0.05 * np.abs(a).max()
    _same(b, b_fast)


def _biaxial(rng, n=4000, scale=0.03):
    t = np.linspace(0, 400, n)
    kx = scale * np.sin(2 * np.pi * 0.05 * t) + 0.3 * scale * rng.standard_normal(n)
    ky = 0.7 * scale * np.cos(
        2 * np.pi * 0.031 * t
    ) + 0.2 * scale * rng.standard_normal(n)
    return kx, ky


@pytest.mark.parametrize("f", [0.0, 0.01, DIRECTION_HOLD_FRACTION, 0.5])
@pytest.mark.parametrize("varying_pressure", [False, True])
def test_compiled_equals_reference_through_small_curvature(f, varying_pressure):
    rng = np.random.default_rng(5)
    for spec in _specs():
        kx, ky = _biaxial(rng, scale=5 * slip_onset_curvature(spec))
        # Slow passages through the origin in changing directions.
        ramp = np.linspace(1.0, 0.0, 40)
        kx[1000:1040] *= ramp
        ky[1000:1040] *= ramp
        kx[1040:1080] = 1e-3 * slip_onset_curvature(spec) * rng.standard_normal(40)
        ky[1040:1080] = 1e-3 * slip_onset_curvature(spec) * rng.standard_normal(40)
        kx[1080:1100] = ky[1080:1100] = 0.0
        pressure = None
        if varying_pressure:
            pressure = spec.contact_pressure_Pa * (
                1 + 0.3 * np.sin(np.linspace(0, 20, kx.size))
            )
        ref, fast = _paths(spec, kx, ky, f, pressure)
        assert np.abs(fast - ref).max() <= 1e-9 * np.abs(ref).max()


def test_histories_away_from_zero_are_unchanged():
    rng = np.random.default_rng(7)
    for spec in _specs():
        threshold = direction_threshold_curvature(spec, 0.5)
        kx, ky = _biaxial(rng, scale=5 * slip_onset_curvature(spec))
        small = np.hypot(kx, ky) < 1.01 * threshold
        kx[small] = ky[small] = 0.0  # exact zeros were already held before
        for f in (0.01, DIRECTION_HOLD_FRACTION, 0.5):
            for path in (0, 1):
                _same(_paths(spec, kx, ky, f)[path], _paths(spec, kx, ky, 0.0)[path])


def test_planar_reversal_through_zero_unchanged():
    # A fixed plane passing through zero changes the angle by pi only, which
    # leaves the cap |cos(phi - phi_bend)| unchanged up to rounding.
    for spec in _specs():
        onset = slip_onset_curvature(spec)
        s = 10 * onset * np.sin(np.linspace(0, 40 * np.pi, 6001))
        s[np.abs(s) < 0.3 * onset] *= 0.01  # dwell near zero
        for angle in (0.0, 0.4, 1.3):
            kx, ky = s * math.cos(angle), s * math.sin(angle)
            old = wire_slip_history(
                spec, kx, ky, initial_bending_angle=angle, direction_hold_fraction=0
            ).stress_Pa
            for f in (0.01, DIRECTION_HOLD_FRACTION, 0.5):
                new = wire_slip_history(
                    spec,
                    kx,
                    ky,
                    initial_bending_angle=angle,
                    direction_hold_fraction=f,
                ).stress_Pa
                assert np.abs(new - old).max() <= 1e-9 * np.abs(old).max()


@pytest.mark.parametrize("bad", [-0.1, 1.5, math.nan, math.inf])
def test_invalid_fraction_rejected(bad):
    spec = _specs()[0]
    with pytest.raises(CableInputError):
        wire_slip_history(spec, np.zeros(2), np.zeros(2), direction_hold_fraction=bad)
    with pytest.raises(CableInputError):
        wire_friction_stress(
            spec, np.zeros(2), np.zeros(2), direction_hold_fraction=bad
        )
