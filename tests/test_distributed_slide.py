# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""The slide of a full-slip increment of the distributed friction model.

With every helix node on its bound the equations leave a slide common to the
displacement ``u`` and the slip ``p`` of all nodes free; the increment takes
the one that gives the step slip a zero pitch mean, or the admissible one
nearest to it (module notes of :mod:`cablexsec.mechanics.distributed_friction`).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import cablexsec as cx
from cablexsec._numba import kernels_compiled
from cablexsec.mechanics import distributed_friction as dfm

PI = math.pi
NODES = 32


def _run(ax, ay, lam, *, khat=4.4, max_increment=math.inf):
    ax = np.ascontiguousarray(ax, dtype=float)
    ay = np.ascontiguousarray(ay, dtype=float)
    steps = ax.size
    kinematics = np.zeros((steps, 4, NODES))
    out = dfm._distributed_kernel(
        ax,
        ay,
        np.ones(steps),
        khat,
        lam,
        np.ones(NODES),
        0.0,
        1.0,
        max_increment,
        np.zeros(1, dtype=np.int64),
        np.zeros(1),
        True,
        True,
        np.arange(steps, dtype=np.int64),
        kinematics,
    )
    assert out[3][0] == 0
    return out, kinematics


def _cycles(steps=200):
    s = np.linspace(0.0, 1.0, steps)
    ax = 30.0 * np.sin(4 * PI * s)
    return ax, 0.5 * ax


@pytest.mark.parametrize("lam", [0.0, 0.127], ids=["uniform", "signed"])
def test_full_slip_step_slip_has_zero_pitch_mean(lam):
    (_o, hist, slip, _s, _f), kin = _run(*_cycles(), lam)
    full = slip >= 1.0
    assert full.sum() > 50  # the cycles are mostly in full slip
    step = kin[:, 2]
    scale = np.abs(step).max()
    for i in np.flatnonzero(full)[1:]:
        mean = step[i].mean()
        # Zero mean, or the admissible slide nearest to it: then one node
        # does not slip over the step.
        assert abs(mean) <= 1e-12 * scale or np.abs(step[i]).min() <= 1e-12 * scale
    # Over consecutive full-slip steps the pitch mean of the slip is kept.
    both = full[1:] & full[:-1]
    mean_p = kin[:, 1].mean(axis=1)
    assert np.abs(np.diff(mean_p)[both]).max() <= 1e-12 * np.abs(kin[:, 1]).max()
    # The energy identity is unchanged by the slide.
    identity = hist[:, 0] - hist[:, 1] - hist[:, 2] - hist[:, 3]
    assert np.abs(identity).max() <= 1e-12 * np.abs(hist[:, 0]).max()


@pytest.mark.parametrize("lam", [0.0, 0.127], ids=["uniform", "signed"])
def test_full_slip_slide_keeps_every_node_slipping_with_its_traction(lam):
    (_o, _h, slip, _s, force), kin = _run(*_cycles(), lam)
    u, p = kin[:, 0], kin[:, 1]
    tau = 4.4 * (u - p)  # interface traction
    step = kin[:, 2]
    full = np.flatnonzero(slip >= 1.0)
    # Friction work node by node: t dp >= 0 (round-off aside).
    work = tau[full] * step[full]
    assert work.min() >= -1e-12 * np.abs(work).max()
    # The forces do not depend on the slide: n_j - n_{j-1} = h t_j.
    h = 2 * PI / NODES
    balance = force - np.roll(force, 1, axis=1) - h * tau
    assert np.abs(balance).max() <= 1e-10 * np.abs(force).max()


@pytest.mark.parametrize("lam", [0.0, 0.127], ids=["uniform", "signed"])
def test_full_slip_fields_are_unique(lam):
    """Before the slide was fixed, a relative change of 1e-15 in the
    curvature moved the full-slip slip field by a sizeable fraction of a step
    slip (the solver left one node on its bound, chosen at round-off)."""
    ax, ay = _cycles()
    _, kin = _run(ax, ay, lam)
    _, nudged = _run(ax * (1.0 + 1e-15), ay, lam)
    assert np.abs(nudged - kin).max() <= 1e-9 * np.abs(kin).max()


def test_partial_slip_stick_nodes_do_not_slip():
    # Below full slip the stick nodes fix the slide: their step slip is zero.
    # Steady rotation at 1.3 times the rigid-stick onset (xi = 0.248).
    a = 1.3 * (1.0 + 0.33) / 0.33
    ramp = np.linspace(0.0, a, 101)[1:]
    th = np.linspace(0.0, 4 * PI, 721)[1:]
    ax = np.concatenate([ramp, a * np.cos(th)])
    ay = np.concatenate([np.zeros(ramp.size), a * np.sin(th)])
    (_o, _h, slip, _s, force), kin = _run(ax, ay, 0.127, khat=0.33)
    assert 0.0 < slip.max() < 1.0
    tau = 0.33 * (kin[:, 0] - kin[:, 1])
    # Node j joins segments j - 1 and j.
    bound = 1.0 + 0.127 * 0.5 * (force + np.roll(force, 1, axis=1))
    stick = np.abs(tau) < bound * (1.0 - 1e-6)
    assert stick.any() and (~stick).any()
    assert np.abs(kin[:, 2][stick]).max() <= 1e-12 * np.abs(kin[:, 2]).max()


def _slide(z0, v, nseg, rho=1.0, lam=0.0, khat=2.0):
    v = np.array(v, dtype=float)
    n = v.size
    dfm._full_slip_slide(
        np.array(z0, dtype=float),
        v,
        np.array(nseg, dtype=float),
        np.ones(n),
        rho,
        np.ones(n),
        khat,
        lam,
    )
    return v


def test_slide_rule_cases():
    """The rule on four nodes, k_hat = 2, unit bound (r = z0 + v,
    ``t = clip(k_hat r, -b, b)``, step slip ``dp = r - t / k_hat``)."""
    zero = np.zeros(4)
    # Full slip, step slip (0.5, -0.5, 0.25, -0.25), mean 0: no change.
    v = _slide([1.0, -1.0, 0.75, -0.75], zero, zero)
    np.testing.assert_allclose(v, zero, atol=1e-15)
    # Step slip (1, 0, 0.75, -0.25), mean 0.375, admissible slides
    # [-0.75, 0]: shifted by -0.375.
    v = _slide([1.5, -0.5, 1.25, -0.75], zero, zero)
    np.testing.assert_allclose(v, -0.375, atol=1e-15)
    # Step slip (0.05, -0.25, 1, 1), mean 0.45, admissible [-0.05, 0.25]:
    # the slide stops at -0.05, where the first node stops slipping.
    v = _slide([0.55, -0.75, 1.5, 1.5], zero, zero)
    np.testing.assert_allclose(v, -0.05, atol=1e-15)
    # ... and from the other side.
    v = _slide([-0.55, 0.75, -1.5, -1.5], zero, zero)
    np.testing.assert_allclose(v, 0.05, atol=1e-15)
    # A node that sticks clear of its bound fixes the slide.
    v = _slide([0.1, -1.0, 1.5, 1.5], zero, zero)
    np.testing.assert_array_equal(v, zero)
    # A floored (zero) bound constrains nothing: segment 3 at -6 with
    # lam = 0.5 floors the bound of nodes 3 and 0 (1 - 1.5 < 0); step slip
    # (0.3, -0.5, 0.5, 0.2), mean 0.125, admissible [-0.5, 0.5].
    v = _slide([0.3, -1.0, 1.0, 0.2], zero, [0.0, 0.0, 0.0, -6.0], lam=0.5)
    np.testing.assert_allclose(v, -0.125, atol=1e-15)


@pytest.mark.skipif(
    not kernels_compiled(), reason="a 556-node helix over 121 samples needs numba"
)
def test_kinematic_history_slip_keeps_its_mean_in_full_slip():
    """The armour slip of :func:`cablexsec.kinematic_history` with the
    distributed law: planar bending at about 40 times the slip onset of the
    66 kV template, most samples in full slip."""
    cable = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
    t = np.linspace(0.0, 30.0, 121)
    kx = 0.01 * np.sin(2 * PI * 0.1 * t)
    with pytest.warns(cx.MissingClampingStateWarning):
        history = cx.kinematic_history(
            cable,
            kx,
            np.zeros_like(t),
            time_s=t,
            tension_N=1e5,
            friction_model="distributed",
        )
    for layer in history.layers:
        mean = layer.slip_m.mean(axis=0)
        amplitude = 0.5 * np.ptp(layer.slip_m, axis=1).max()
        # Planar bending, symmetric about the bending plane: no net slide.
        assert np.abs(mean).max() <= 1e-9 * amplitude
