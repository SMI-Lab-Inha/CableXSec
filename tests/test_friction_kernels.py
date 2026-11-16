# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Friction kernels: closed forms, limits and failure reports on small inputs.

The release gate runs this file a second time with ``NUMBA_DISABLE_JIT=1`` so
that coverage traces the kernel bodies. Each case asserts a property of its
kernel: the discrete stick solution of the distributed helix model, the
Lipschitz bound of full slip, the uniform force without friction, the cyclic
solvers against dense algebra, the first sample that cannot be advanced, the
NumPy and compiled anchored-wire paths against each other, and the
sub-threshold direction blend of the progressive law against its reference.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.cable_adapters import cable_to_mkappa_specs
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics import clamp_layer as cl
from cablexsec.mechanics import distributed_friction as dfm
from cablexsec.mechanics import fast_slip
from cablexsec.mechanics.papailiou import slip_onset_curvature
from cablexsec.mechanics.progressive_friction import (
    ProgressiveFriction,
    progressive_caps,
    progressive_wire_friction_stress,
    progressive_wire_history,
)

PI = math.pi


@pytest.fixture(scope="module")
def menard():
    return cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)


@pytest.fixture(scope="module")
def armour(menard):
    """Armour layer of the distributed model, with clamping."""
    return cx.cable_to_fatigue_layers(menard, contact_tension_N=2e4)[0].spec


@pytest.fixture(scope="module")
def mkappa_layer(menard):
    return cable_to_mkappa_specs(menard, 0.0, mu=0.2, contact_model="bounded")[0]


def _run(
    ax,
    ay,
    *,
    khat=0.33,
    lam=0.0,
    nodes=32,
    rho=None,
    max_increment=math.inf,
    newton=True,
    wires=(0,),
):
    """Normalised distributed kernel, segment ``j`` at the azimuth ``j h``."""
    ax = np.ascontiguousarray(ax, dtype=float)
    ay = np.ascontiguousarray(ay, dtype=float)
    return dfm._distributed_kernel(
        ax,
        ay,
        np.ones(ax.size) if rho is None else np.ascontiguousarray(rho, dtype=float),
        float(khat),
        float(lam),
        np.ones(nodes),
        0.0,
        1.0,
        float(max_increment),
        np.asarray(wires, dtype=np.int64),
        np.zeros(len(wires)),
        False,
        newton,
    )


# --------------------------------------------------------------------------- #
# Distributed helix model: cyclic solvers                                      #
# --------------------------------------------------------------------------- #
def _dense_cyclic(lower, diag, upper):
    n = diag.size
    a = np.diag(diag)
    for i in range(n):
        a[i, (i - 1) % n] += lower[i]
        a[i, (i + 1) % n] += upper[i]
    return a


def _scratch(n):
    return np.empty(n), np.empty(n), np.empty(n), np.empty(n)


def test_cyclic_solvers_match_dense_algebra():
    rng = np.random.default_rng(4)
    n = 7
    rhs = rng.normal(size=n)
    diag = 4.0 + rng.random(n)
    c = -1.3
    x, cp, wy, wz = _scratch(n)
    assert dfm._cyclic_solve(diag, c, rhs, x, cp, wy, wz)
    dense = _dense_cyclic(np.full(n, c), diag, np.full(n, c))
    np.testing.assert_allclose(x, np.linalg.solve(dense, rhs), rtol=1e-12, atol=1e-14)
    lower = rng.uniform(-1.5, -0.5, n)
    upper = rng.uniform(-1.5, -0.5, n)
    assert dfm._cyclic_solve_general(lower, diag, upper, rhs, x, cp, wy, wz)
    dense = _dense_cyclic(lower, diag, upper)
    np.testing.assert_allclose(x, np.linalg.solve(dense, rhs), rtol=1e-12, atol=1e-14)


def test_cyclic_solvers_report_zero_pivots():
    # A zero first pivot (d_0 = 0), a zero interior pivot (d_1 = c^2 / (2 d_0))
    # and a nonfinite right-hand side are failures, not divisions by zero.
    rhs = np.ones(3)
    x, cp, wy, wz = _scratch(3)
    ones = np.ones(3)
    assert not dfm._cyclic_solve(np.array([0.0, 2.0, 2.0]), 1.0, rhs, x, cp, wy, wz)
    assert not dfm._cyclic_solve(np.array([1.0, 0.5, 2.0]), 1.0, rhs, x, cp, wy, wz)
    bad = np.array([1.0, math.nan, 1.0])
    assert not dfm._cyclic_solve(np.full(3, 4.0), 1.0, bad, x, cp, wy, wz)
    zero = np.array([0.0, 2.0, 2.0])
    assert not dfm._cyclic_solve_general(ones, zero, ones, rhs, x, cp, wy, wz)
    pivot = np.array([1.0, 0.5, 2.0])
    assert not dfm._cyclic_solve_general(ones, pivot, ones, rhs, x, cp, wy, wz)


def test_same_active_set_tracks_stick_and_slip_sign():
    z0 = np.zeros(3)
    b = np.ones(3)
    v = np.array([0.5, 2.0, -2.0])  # stick, forward slip, backward slip
    assert dfm._same_active_set(z0, v, v.copy(), b, 1.0)
    assert dfm._same_active_set(z0, v, np.array([0.1, 3.0, -5.0]), b, 1.0)
    assert not dfm._same_active_set(z0, v, np.array([2.0, 2.0, -2.0]), b, 1.0)
    assert not dfm._same_active_set(z0, v, np.array([0.5, -2.0, -2.0]), b, 1.0)


def test_fixed_bound_without_friction_leaves_the_mean_force():
    # With a zero bound the wire slides freely: the force is uniform along the
    # pitch and equal to the pitch mean of the imposed force.
    n = 12
    h = 2 * PI / n
    y = np.random.default_rng(1).normal(size=n)
    v = np.zeros(n)
    work = [np.zeros(n) for _ in range(8)]
    iters = dfm._solve_fixed_bound(
        y, np.zeros(n), np.zeros(n), 0.33, h, 1e-12, v, *work
    )
    assert iters == 0
    force = y + (np.roll(v, -1) - v) / h
    np.testing.assert_allclose(force, y.mean(), rtol=0, atol=1e-12)


# --------------------------------------------------------------------------- #
# Distributed helix model: closed forms and limits                             #
# --------------------------------------------------------------------------- #
def test_stick_solution_is_the_discrete_shear_lag_closed_form():
    # High clamping keeps every node in stick: the force is the exact discrete
    # solution A cos(j h), A = a k / (k + s^2), s = 2 sin(h / 2) / h, which
    # tends to xi a (k / (1 + k) = xi) as h -> 0.
    nodes, khat, a = 32, 0.33, 0.5
    h = 2 * PI / nodes
    s2 = (2 * math.sin(h / 2) / h) ** 2
    ax = np.linspace(0.0, a, 6)[1:]
    out, hist, slip, stats, _ = _run(
        ax, np.zeros(ax.size), khat=khat, rho=np.full(ax.size, 5.0), wires=(0, 8, 16)
    )
    assert stats[0] == 0
    amplitude = ax * khat / (khat + s2)
    np.testing.assert_allclose(out[0], amplitude, rtol=1e-9)
    np.testing.assert_allclose(out[1], 0.0, atol=1e-10)  # quarter pitch
    np.testing.assert_allclose(out[2], -amplitude, rtol=1e-9)
    assert abs(amplitude[-1] / a - khat / (1 + khat)) < 2 * h * h
    assert np.all(slip == 0.0)
    np.testing.assert_allclose(hist[:, 1], 0.0, atol=1e-14)  # no dissipation


def test_zero_clamping_carries_no_friction_force():
    ax = np.linspace(0.0, 2.0, 5)[1:]
    out, hist, _, stats, _ = _run(ax, 0.5 * ax, rho=np.zeros(ax.size), wires=(0, 5))
    assert stats[0] == 0
    np.testing.assert_allclose(out, 0.0, atol=1e-12)
    np.testing.assert_allclose(hist[:, 1], 0.0, atol=1e-14)


def test_full_slip_force_obeys_the_lipschitz_bound():
    # In full slip |dn/dphi| <= b = 1 and the pitch mean of n is zero, so
    # |n| <= pi / 2 (a triangle wave) up to one segment.
    nodes = 32
    h = 2 * PI / nodes
    ax = np.linspace(0.0, 60.0, 13)[1:]
    out, hist, slip, stats, _ = _run(
        ax, np.zeros(ax.size), nodes=nodes, wires=range(nodes), max_increment=2.0
    )
    assert stats[0] == 0
    assert np.abs(out).max() <= PI / 2 + h
    np.testing.assert_allclose(out[:, -1].mean(), 0.0, atol=1e-9)
    assert slip[-1] >= 0.9
    assert np.all(np.diff(hist[:, 1]) >= -1e-12)


def test_capstan_newton_fallback_reproduces_the_fixed_point():
    # Large unsubdivided reversals with a strong capstan term defeat the
    # coupled Newton iteration; the fixed point it falls back on solves the
    # same equations, so the result equals the fixed-point-only solver.
    ax = np.array([100.0, -100.0, 100.0])
    ay = np.zeros(3)
    newton = _run(ax, ay, khat=1e4, lam=1.0, wires=range(32))
    fixed = _run(ax, ay, khat=1e4, lam=1.0, wires=range(32), newton=False)
    assert newton[3][0] == 0 and fixed[3][0] == 0
    assert newton[3][6] > 0  # Newton fall-backs counted
    scale = np.abs(fixed[0]).max()
    np.testing.assert_allclose(newton[0], fixed[0], rtol=0, atol=1e-9 * scale)


def test_kernel_reports_the_first_sample_it_cannot_advance():
    # A near-rigid interface (k_hat = 1e8) taking an unsubdivided reversal of
    # 2e10 onset curvatures is beyond the solver even after twelve halvings:
    # the status names that sample and the samples before it are kept.
    ax = np.array([1e10, -1e10, 1e10])
    out, _, _, stats, _ = _run(ax, np.zeros(3), khat=1e8)
    first = -int(stats[0]) - 1
    assert 0 < first < ax.size
    clean, _, _, ok, _ = _run(ax[:first], np.zeros(first), khat=1e8)
    assert ok[0] == 0
    np.testing.assert_array_equal(out[:, :first], clean)


def test_kept_slip_increments_accumulate_to_the_slip_field(armour):
    kx = 0.05 * np.sin(np.linspace(0.0, 2 * PI, 9)[1:])
    ky = 0.2 * kx
    samples = np.arange(kx.size)
    kept = dfm.distributed_friction_history(
        armour, kx, ky, nodes_per_pitch=16, field_samples=samples
    )
    both = dfm.distributed_friction_history(
        armour, kx, ky, nodes_per_pitch=16, field_samples=samples, record_field=True
    )
    field = kept.kinematic_field
    assert field is not None and both.field_normalised is not None
    np.testing.assert_allclose(
        np.cumsum(field[:, 2], axis=0), field[:, 1], rtol=0, atol=1e-12
    )
    # The kept force equals the recorded segment field of the same samples.
    np.testing.assert_array_equal(field[:, 3], both.field_normalised)
    np.testing.assert_array_equal(kept.stress_Pa, both.stress_Pa)


# --------------------------------------------------------------------------- #
# Distributed helix model: inputs                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kappa_y": np.zeros(3)}, "finite matching nonempty"),
        ({"contact_pressure_Pa": np.ones(3)}, "pressure must be finite"),
        ({"contact_pressure_Pa": -np.ones(4)}, "nonnegative pressure"),
        ({"stick_modulus_Pa": np.ones(3)}, "stiffness must be finite"),
        ({"field_samples": np.array([2, 1])}, "field samples must be increasing"),
        ({"field_samples": np.array([0, 4])}, "field samples must be increasing"),
    ],
)
def test_distributed_inputs_are_checked(armour, kwargs, message):
    args = {"kappa_x": np.full(4, 0.01), "kappa_y": np.zeros(4)}
    args.update(kwargs)
    with pytest.raises(CableInputError, match=message):
        dfm.distributed_friction_history(armour, nodes_per_pitch=16, **args)


def test_distributed_contact_shape_is_checked(armour):
    spec = replace(armour, shape_g=np.ones(armour.n_wires + 1))
    with pytest.raises(CableInputError, match="shape_g must be finite"):
        dfm.distributed_friction_history(spec, np.full(3, 0.01), np.zeros(3))


def test_unconverged_kernel_names_the_sample(armour, monkeypatch):
    name = "_distributed_kernel_fast" if dfm._FAST_KERNEL else "_distributed_kernel"
    kernel = getattr(dfm, name)

    def failing_at_sample_2(*args):
        out, hist, slip, stats, fld = kernel(*args)
        stats = stats.copy()
        stats[0] = -3  # the kernel's status for sample index 2
        return out, hist, slip, stats, fld

    monkeypatch.setattr(dfm, name, failing_at_sample_2)
    with pytest.raises(CableInputError, match="did not converge at sample 2"):
        dfm.distributed_friction_history(
            armour, np.full(4, 0.01), np.zeros(4), nodes_per_pitch=16
        )


# --------------------------------------------------------------------------- #
# Anchored wire (clamp layer)                                                 #
# --------------------------------------------------------------------------- #
def _anchored_drive(samples=24, segments=40):
    h = 6 * 2 * PI / segments
    phi = (np.arange(segments) + 0.5) * h
    c_hat = np.cos(phi)
    amp = np.sin(np.linspace(0.0, 4 * PI, samples)) * np.linspace(0.5, 2.0, samples)
    drive = np.outer(amp, np.linspace(1.0, 0.6, segments))
    return drive, c_hat, h


def test_node_dissipation_numpy_path_matches_the_kernel():
    drive, c_hat, h = _anchored_drive()
    for start in (0, 12):
        fast = cl.anchored_wire_node_dissipation(
            drive, c_hat, h, from_sample=start, max_increment=0.25
        )
        previous = np.vstack((np.zeros((1, c_hat.size)), drive[:-1]))
        step = np.max(np.abs((drive - previous) * c_hat[None, :]), axis=1)
        counts = np.maximum(1, np.ceil(step / 0.25)).astype(np.int64)
        reference = np.zeros(c_hat.size - 1)
        cl._node_dissipation_numpy(drive, c_hat, counts, h, start, reference)
        np.testing.assert_allclose(fast, reference, rtol=1e-10, atol=1e-14)
        assert np.all(fast >= 0.0)
    whole = cl.anchored_wire_node_dissipation(drive, c_hat, h, max_increment=0.25)
    late = cl.anchored_wire_node_dissipation(
        drive, c_hat, h, from_sample=12, max_increment=0.25
    )
    assert np.all(late <= whole + 1e-14) and late.sum() < whole.sum()


@pytest.mark.parametrize(
    ("args", "kwargs", "message"),
    [
        ((np.ones(5), np.ones(5), 0.1), {}, "drive must be a finite"),
        ((np.ones((3, 4)), np.ones(3), 0.1), {}, "c_hat must match"),
        ((np.ones((3, 4)), np.ones(4), 0.0), {}, "segment length must be positive"),
        ((np.ones((3, 4)), np.ones(4), 0.1), {"max_increment": 0.0}, "max_increment"),
    ],
)
def test_node_dissipation_inputs_are_checked(args, kwargs, message):
    with pytest.raises(CableInputError, match=message):
        cl.anchored_wire_node_dissipation(*args, **kwargs)


def test_recording_a_subset_of_steps_does_not_change_the_response():
    segments = 48
    h = 6 * 2 * PI / segments
    e = np.cos((np.arange(segments) + 0.5) * h)
    path = 2.0 * np.sin(np.linspace(0.0, 3 * PI, 40))
    full, full_ref, full_d = cl.anchored_wire_response(e, path, h)
    mask = np.zeros(path.size, dtype=bool)
    mask[::3] = True
    part, part_ref, part_d = cl.anchored_wire_response(e, path, h, record=mask)
    np.testing.assert_array_equal(part, full[mask])
    np.testing.assert_array_equal(part_ref, full_ref[mask])
    assert part_d == full_d > 0.0


# --------------------------------------------------------------------------- #
# Progressive (Iwan) law and the direction-tied cap                            #
# --------------------------------------------------------------------------- #
def test_compiled_progressive_law_matches_reference_below_the_hold_threshold(
    mkappa_layer,
):
    # Small curvatures (below the direction-hold threshold) blend the cap of
    # the held direction with that of the current one; zero samples keep the
    # last direction.
    onset = slip_onset_curvature(mkappa_layer)
    t = np.linspace(0.0, 6 * PI, 240)
    radius = onset * (0.002 + 0.6 * np.abs(np.sin(t / 3)))
    kx = radius * np.cos(t)
    ky = radius * np.sin(t)
    kx[[0, 60, 61]] = 0.0
    ky[[0, 60, 61]] = 0.0
    for settings in (ProgressiveFriction(), ProgressiveFriction("uniform", 5, 0.3)):
        reference = progressive_wire_history(
            mkappa_layer, kx, ky, progressive=settings, direction_hold_fraction=0.5
        ).stress_Pa
        fast = progressive_wire_friction_stress(
            mkappa_layer, kx, ky, progressive=settings, direction_hold_fraction=0.5
        )
        assert np.abs(reference).max() > 0.0
        np.testing.assert_allclose(
            fast, reference, rtol=0, atol=1e-12 * np.abs(reference).max()
        )


def test_caps_vanish_without_clamping_or_lay_angle(mkappa_layer):
    for settings in (ProgressiveFriction(), ProgressiveFriction("uniform", 4, 0.2)):
        unclamped = progressive_caps(
            mkappa_layer, 0.3, settings, contact_pressure_Pa=0.0
        )
        assert unclamped.shape == (mkappa_layer.n_wires, settings.n_elements)
        assert np.all(unclamped == 0.0)
        straight = progressive_caps(
            replace(mkappa_layer, lay_angle_rad=0.0), 0.3, settings
        )
        assert np.all(straight == 0.0)
    layer = mkappa_layer
    args = (
        np.ascontiguousarray(layer.phi_wires),
        0.3,
        1.0e5,
        layer.r_helix_m,
        layer.n_wires,
        layer.A_wire_m2,
    )
    g = np.ones(layer.n_wires)
    assert np.all(fast_slip._cap(*args, 0.0, layer.mu, g) == 0.0)
    assert np.all(fast_slip._cap(*args, layer.lay_angle_rad, layer.mu, g) > 0.0)


def test_progressive_history_beyond_floating_point_range_is_rejected(mkappa_layer):
    kx = np.array([1.0e300, -1.0e300, 1.0e300])
    with pytest.raises(CableInputError, match="floating-point range"):
        progressive_wire_history(mkappa_layer, kx, np.zeros(3))


def test_compiled_progressive_pressure_is_checked(mkappa_layer):
    with pytest.raises(CableInputError, match="pressure must be finite, nonnegative"):
        progressive_wire_friction_stress(
            mkappa_layer, np.ones(3), np.zeros(3), contact_pressure_Pa=-np.ones(3)
        )
