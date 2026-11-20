# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""The fast distributed-friction kernel must reproduce the reference kernel.

``_distributed_kernel_fast`` (used when numba is active) solves the same
increments as ``_distributed_kernel`` with a predicted start and a kept
factorisation; the two agree to round-off on every output: wire forces,
energies, slip fraction, segment field and kept kinematics.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import cablexsec as cx
from cablexsec._numba import kernels_compiled
from cablexsec.mechanics import distributed_friction as dfm
from cablexsec.mechanics.friction_law import FrictionLaw

PI = math.pi
# Relative to the largest value of each output; the kernels differ by the
# order of floating-point operations only.
RTOL = 1e-10
# Interpreted kernels (no numba, or the coverage pass) take shorter histories.
COMPILED = kernels_compiled()
STEPS = 240 if COMPILED else 48
INCREMENT = 0.1 if COMPILED else 0.5


def _paths(kind, steps=STEPS):
    t = np.linspace(0.0, 1.0, steps)
    if kind == "planar":
        return 6.0 * np.sin(6 * PI * t) * (0.4 + t), np.zeros(steps)
    if kind == "rotating":
        a = np.minimum(1.0, 4 * t) * 2.5
        return a * np.cos(8 * PI * t), a * np.sin(8 * PI * t)
    if kind == "one_sided":
        # Bending that never reverses its sign: partial unloading only.
        return 3.0 + 2.0 * np.sin(5 * PI * t), 0.5 + 0.3 * np.cos(3 * PI * t)
    if kind == "full_slip":
        # A planar ramp far beyond the slip onset: every node at the bound.
        return np.linspace(0.0, 60.0, steps), np.zeros(steps)
    raise ValueError(kind)


def _run(kernel, ax, ay, *, khat=0.33, lam=0.127, nodes=48, **options):
    rho = options.pop("rho", None)
    g = options.pop("g", None)
    samples = options.pop("record_index", None)
    kinematics = None
    if samples is not None:
        kinematics = np.zeros((int(samples.max()) + 1, 4, nodes))
    out = kernel(
        np.ascontiguousarray(ax, dtype=float),
        np.ascontiguousarray(ay, dtype=float),
        np.ones(len(ax)) if rho is None else np.ascontiguousarray(rho, dtype=float),
        float(khat),
        float(lam),
        np.ones(nodes) if g is None else np.ascontiguousarray(g, dtype=float),
        options.pop("phase0", 0.0),
        options.pop("xi", 1.0),
        options.pop("max_increment", INCREMENT),
        np.arange(0, nodes, 4, dtype=np.int64),
        np.zeros(len(range(0, nodes, 4))),
        True,
        options.pop("newton", True),
        samples,
        kinematics,
        *options.pop("law", ()),
    )
    assert not options
    return out, kinematics


def _assert_kinematics_close(fast, ref):
    """Kept helix fields ``(samples, [u, p, step slip, n], N)``, also in full
    slip, where both kernels fix the slide of ``u`` and ``p`` by the same
    rule (``_full_slip_slide``)."""
    scale = max(float(np.abs(ref[:, :3]).max()), 1.0)
    assert np.abs(fast[:, 3] - ref[:, 3]).max() <= RTOL * np.abs(ref[:, 3]).max()
    assert np.abs(fast[:, :3] - ref[:, :3]).max() <= RTOL * scale


def _assert_close(fast, ref):
    (f_out, f_hist, f_slip, f_stats, f_fld), f_kin = fast
    (r_out, r_hist, r_slip, r_stats, r_fld), r_kin = ref
    assert f_stats[0] == r_stats[0] == 0
    assert f_stats[3] == r_stats[3]  # the same substeps
    pairs = [(f_out, r_out), (f_fld, r_fld)]
    if r_kin is not None:
        _assert_kinematics_close(f_kin, r_kin)
    for a, b in pairs:
        scale = max(float(np.abs(b).max()), 1e-300)
        assert np.abs(a - b).max() <= RTOL * scale
    # Energies on the scale of the largest one: the dissipation of an
    # elastic path and the remainder are themselves round-off there.
    scale = float(np.abs(r_hist).max())
    assert np.abs(f_hist - r_hist).max() <= RTOL * scale
    np.testing.assert_allclose(f_slip, r_slip, rtol=0, atol=2.0 / f_fld.shape[1])


@pytest.mark.parametrize("kind", ["planar", "rotating", "one_sided", "full_slip"])
@pytest.mark.parametrize("lam", [0.127, 0.0], ids=["signed", "uniform"])
def test_fast_kernel_matches_reference(kind, lam):
    ax, ay = _paths(kind)
    for khat in (0.33, 4.4):
        args = {"khat": khat, "lam": lam}
        ref = _run(dfm._distributed_kernel, ax, ay, **args)
        fast = _run(dfm._distributed_kernel_fast, ax, ay, **args)
        _assert_close(fast, ref)


def test_fast_kernel_matches_reference_with_shape_pressure_and_kinematics():
    ax, ay = _paths("rotating")
    nodes = 64
    rho = 1.0 + 0.3 * np.sin(np.linspace(0.0, 9.0, ax.size))
    g = 1.0 + 0.25 * np.cos(2 * PI * np.arange(nodes) / nodes)
    index = np.full(ax.size, -1, dtype=np.int64)
    index[::17] = np.arange(index[::17].size)
    common = {"nodes": nodes, "rho": rho, "g": g, "phase0": PI / nodes}
    for newton in (True, False):
        options = dict(common, newton=newton)
        ref = _run(dfm._distributed_kernel, ax, ay, record_index=index, **options)
        fast = _run(dfm._distributed_kernel_fast, ax, ay, record_index=index, **options)
        _assert_close(fast, ref)


def test_fast_kernel_matches_reference_with_friction_law():
    ax, ay = _paths("planar")
    nodes = 48
    rho = 1.0 + 0.4 * np.sin(np.linspace(0.0, 7.0, ax.size))
    table = np.array([[0.5, 1.5], [1.2, 0.8]])  # [P, f]: falls with pressure
    for law in (
        (np.array([1.3, 1.0, 0.0]), np.full(ax.size, 0.01), None, 0.0),
        (np.array([1.3, 1.0, 0.05]), np.full(ax.size, 0.01), table, 1.0),
        (np.array([1.0, 1.0, math.inf]), np.zeros(ax.size), table, 1.0),
    ):
        options = {"nodes": nodes, "rho": rho, "law": law}
        ref = _run(dfm._distributed_kernel, ax, ay, **options)
        fast = _run(dfm._distributed_kernel_fast, ax, ay, **options)
        _assert_close(fast, ref)


def _held_path(steps=STEPS):
    """A ramp past the slip onset, a hold, a reversal and a second hold."""
    q = steps // 4
    ax = np.concatenate(
        (
            np.linspace(0.0, 4.0, q),
            np.full(q, 4.0),
            np.linspace(4.0, -2.0, q),
            np.full(steps - 3 * q, -2.0),
        )
    )
    return ax, 0.3 * ax, q


@pytest.mark.parametrize("lam", [0.127, 0.0], ids=["signed", "uniform"])
def test_a_repeated_sample_keeps_the_state(lam):
    """Under the Coulomb bound a sample that repeats the last one leaves the
    state as it is, in both kernels: its outputs equal the last sample's bit
    for bit and it takes no substep. Solved, it moved the forces by round-off
    (the fast kernel by the Newton tolerance), which counted as cycles."""
    ax, ay, q = _held_path()
    held = np.r_[q : 2 * q, 3 * q : ax.size]
    runs = {}
    for kernel in (dfm._distributed_kernel, dfm._distributed_kernel_fast):
        run = _run(kernel, ax, ay, lam=lam)
        out, hist, slip, stats, fld = run[0]
        for i in held:
            assert np.array_equal(out[:, i], out[:, i - 1])
            assert np.array_equal(fld[i], fld[i - 1])
            assert np.array_equal(hist[i], hist[i - 1])
            assert slip[i] == slip[i - 1]
        runs[kernel] = run
        assert stats[0] == 0
    fast, ref = runs[dfm._distributed_kernel_fast], runs[dfm._distributed_kernel]
    _assert_close(fast, ref)
    moving = _run(dfm._distributed_kernel, ax + 1e-3 * np.arange(ax.size), ay)
    assert moving[0][3][3] > ref[0][3][3]  # the held samples took no substep


def test_a_repeated_sample_under_a_friction_law_is_solved_from_zero():
    """With a friction law a repeated sample can change the law state (a node
    heals while it sticks), so it is solved; the fast kernel starts it from a
    zero increment, as the reference does."""
    ax, ay, _q = _held_path()
    rho = np.ones(ax.size)
    law = (np.array([1.3, 1.0, 0.05]), np.full(ax.size, 0.01), None, 0.0)
    options = {"nodes": 48, "rho": rho, "law": law}
    ref = _run(dfm._distributed_kernel, ax, ay, **options)
    fast = _run(dfm._distributed_kernel_fast, ax, ay, **options)
    _assert_close(fast, ref)


def test_fast_kernel_on_a_small_helix():
    # Fewer than three nodes go to the reference kernel unchanged.
    ax, ay = _paths("rotating", steps=40)
    for nodes in (2, 3, 4):
        ref = _run(dfm._distributed_kernel, ax, ay, nodes=nodes)
        fast = _run(dfm._distributed_kernel_fast, ax, ay, nodes=nodes)
        _assert_close(fast, ref)


def _cyclic(rng, n):
    lower = -1.0 - 0.3 * rng.random(n)
    upper = -1.0 + 0.3 * rng.random(n)
    diag = 2.6 + rng.random(n)
    dense = np.diag(diag)
    for i in range(n):
        dense[i, (i - 1) % n] += lower[i]
        dense[i, (i + 1) % n] += upper[i]
    return lower, diag, upper, dense


@pytest.mark.parametrize("n", [3, 4, 7, 16, 33])
def test_kept_twisted_factorisation_solves_after_row_changes(n):
    """Pivots kept outside the changed rows give the dense solution."""
    rng = np.random.default_rng(n)
    mat = np.full((dfm._M_ROWS, n), np.nan)
    mstate = np.array([0.0, n - 1.0, 0.0, 0.0, 0.0, 0.0])
    lower, diag, upper, dense = _cyclic(rng, n)
    for trial in range(40):
        if trial:
            # Change one to three rows, marking them stale as the Newton pass does.
            for j in rng.choice(n, size=min(n, 1 + trial % 3), replace=False):
                lower[j] = -1.0 - 0.3 * rng.random()
                upper[j] = -1.0 + 0.3 * rng.random()
                diag[j] = 2.6 + rng.random()
                mstate[0] = min(mstate[0], j)
                mstate[1] = max(mstate[1], j)
            dense = np.diag(diag)
            for i in range(n):
                dense[i, (i - 1) % n] += lower[i]
                dense[i, (i + 1) % n] += upper[i]
        mat[dfm._M_LOWER] = lower
        mat[dfm._M_DIAG] = diag
        mat[dfm._M_UPPER] = upper
        rhs = rng.normal(size=n)
        v = np.full(n, 0.5)
        assert dfm._twisted_solve(rhs, v, np.empty(n), mat, mstate)
        assert mstate[0] >= n and mstate[1] < 0
        np.testing.assert_allclose(v - 0.5, np.linalg.solve(dense, rhs), atol=1e-12)


def test_twisted_factorisation_reports_singular_pivots():
    n = 8
    lower, diag, upper, _ = _cyclic(np.random.default_rng(1), n)
    # Zero pivots: row 0, a top row, a bottom row, the corner row and the
    # middle row (the entries named are set to zero).
    for row, zeros in (
        (0, (dfm._M_DIAG,)),
        (2, (dfm._M_DIAG, dfm._M_LOWER)),
        (6, (dfm._M_DIAG, dfm._M_UPPER)),
        (n - 1, (dfm._M_DIAG, dfm._M_UPPER)),
        (n // 2, (dfm._M_DIAG, dfm._M_LOWER, dfm._M_UPPER)),
    ):
        mat = np.full((dfm._M_ROWS, n), np.nan)
        mat[dfm._M_LOWER] = lower
        mat[dfm._M_DIAG] = diag
        mat[dfm._M_UPPER] = upper
        for band in zeros:
            mat[band, row] = 0.0
        mstate = np.array([0.0, n - 1.0, 0.0, 0.0, 0.0, 0.0])
        assert not dfm._twisted_solve(np.ones(n), np.zeros(n), np.empty(n), mat, mstate)
        assert mstate[0] == 0.0 and mstate[1] == n - 1


python_kernels = pytest.mark.skipif(
    COMPILED, reason="replaces kernel functions, which needs them as Python"
)


@python_kernels
@pytest.mark.parametrize("lam", [0.127, 0.0])
def test_fast_kernel_fall_backs(monkeypatch, lam):
    """Without its Newton the fast kernel falls back to the fixed point of
    the reference; an increment that never converges names its sample."""
    ax, ay = _paths("rotating", steps=30)
    ref = _run(dfm._distributed_kernel, ax, ay, lam=lam, nodes=16)
    monkeypatch.setattr(dfm, "_capstan_newton_fast", lambda *args: -1)
    fast = _run(dfm._distributed_kernel_fast, ax, ay, lam=lam, nodes=16)
    _assert_close(fast, ref)
    if lam > 0.0:
        assert fast[0][3][6] == fast[0][3][3]  # every substep fell back
    calls = []

    def fixed_point_failing_after_three(*args):
        calls.append(1)
        return len(calls) <= 3 and real_fixed_point(*args)

    real_fixed_point = dfm._fixed_point
    monkeypatch.setattr(dfm, "_fixed_point", fixed_point_failing_after_three)
    (out, _h, _s, stats, _f), _ = _run(
        dfm._distributed_kernel_fast, ax, ay, lam=lam, nodes=16, max_increment=math.inf
    )
    assert stats[0] == -4  # samples 0-2 advanced, sample 3 not, after 12 halvings
    np.testing.assert_allclose(out[:, :3], ref[0][0][:, :3], rtol=0, atol=1e-12)


@python_kernels
def test_fast_newton_restarts_from_zero_before_the_fixed_point(monkeypatch):
    ax, ay = _paths("rotating", steps=30)
    ref = _run(dfm._distributed_kernel, ax, ay, nodes=16)
    real = dfm._capstan_newton_fast

    def failing_from_a_prediction(*args):
        v = args[8]
        return -1 if np.any(v != 0.0) else real(*args)

    monkeypatch.setattr(dfm, "_capstan_newton_fast", failing_from_a_prediction)
    fast = _run(dfm._distributed_kernel_fast, ax, ay, nodes=16)
    _assert_close(fast, ref)
    assert fast[0][3][6] == 0  # no fixed-point fall-back


def _armour():
    cable = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
    return cx.cable_to_fatigue_layers(cable, contact_tension_N=1e5)[0].spec


def _histories(monkeypatch, law, bound):
    spec = _armour()
    t = np.linspace(0.0, 30.0, 301 if COMPILED else 61)
    kx = 0.004 * np.sin(2 * PI * 0.1 * t)
    ky = 0.002 * np.sin(2 * PI * 0.1 * t + 1.0)
    pressure = spec.contact_pressure_Pa * (1.0 + 0.2 * np.sin(2 * PI * 0.07 * t))
    healing = law is not None and math.isfinite(law.stick_time_s)
    runs = []
    for fast in (False, True):
        monkeypatch.setattr(dfm, "_FAST_KERNEL", fast)
        runs.append(
            dfm.distributed_friction_history(
                spec,
                kx,
                ky,
                contact_pressure_Pa=pressure,
                friction_law=law,
                time_s=t if healing else None,
                bound=bound,
                field_samples=np.arange(0, t.size, 10),
                nodes_per_pitch=None if COMPILED else 32,
            )
        )
    return runs


_ENERGIES = (
    "dissipation_J_per_m",
    "stored_J_per_m",
    "endpoint_work_J_per_m",
    "integration_remainder_J_per_m",
)


@pytest.mark.parametrize("bound", ["signed", "uniform"])
@pytest.mark.parametrize(
    "law",
    [None, FrictionLaw(pressure_factor=((2e4, 1.1), (1.5e5, 0.9)))],
    ids=["no_law", "pressure_factor"],
)
def test_public_history_is_the_reference_to_round_off(monkeypatch, bound, law):
    ref, fast = _histories(monkeypatch, law, bound)
    assert (
        np.abs(fast.stress_Pa - ref.stress_Pa).max()
        <= RTOL * np.abs(ref.stress_Pa).max()
    )
    _assert_kinematics_close(fast.kinematic_field, ref.kinematic_field)
    energy = np.abs(ref.endpoint_work_J_per_m).max()
    for name in _ENERGIES:
        a, b = getattr(fast, name), getattr(ref, name)
        assert np.abs(a - b).max() <= RTOL * energy, name
    assert fast.parameters["substeps"] == ref.parameters["substeps"]


@pytest.mark.parametrize("bound", ["signed", "uniform"])
def test_public_history_with_healing_matches_the_reference(monkeypatch, bound):
    """With a finite healing time a node's factor follows binary events (it
    heals after sticking long enough, breaks away when its trial traction
    leaves the static bound) that a round-off change can move by a substep,
    in the reference as much as between the kernels. On the 32-node history
    of the interpreted run with the uniform bound a relative change of 1e-15
    in the curvature moves the largest reference stress by 0.24 % of its
    peak, as much as the kernels differ there; with the default nodes
    (compiled run) they agree to round-off.
    """
    law = FrictionLaw(static_factor=1.2, dynamic_factor=1.0, stick_time_s=0.5)
    ref, fast = _histories(monkeypatch, law, bound)
    tol = RTOL if COMPILED else 0.01
    scale = np.abs(ref.stress_Pa).max()
    assert np.abs(fast.stress_Pa - ref.stress_Pa).max() <= tol * scale
    energy = np.abs(ref.endpoint_work_J_per_m).max()
    for name in _ENERGIES:
        a, b = getattr(fast, name), getattr(ref, name)
        assert np.abs(a - b).max() <= tol * energy, name
