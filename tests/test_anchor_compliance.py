# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Elastic (compliant) wire anchors: closed forms, limits, energy, routes.

A wire anchored at an end fitting is held by the potting with a finite
stiffness ``K`` [N/m per wire]. The anchored chains of the clamp solver, the
along-wire solver, the termination correction and the along-wire bending
test take it as a spring at the anchor node in parallel with the node's half
segment of friction. Closed form (uniform imposed strain ``A`` from the
virgin state, uniform bound ``rho``, normalised anchor stiffness ``k_a``,
interface stiffness ``k_hat = beta^2``; ``beta = inf`` for rigid stick): the
slip zone of length ``L`` next to the anchor solves

    k_a rho L^2 / 2 + rho (1 + k_a / beta) L + k_a rho / beta^2 + rho / beta - A = 0,

the anchor force is ``S = A - rho / beta - rho L`` and the dissipation
``rho^2 (L^2 / (2 beta) + L^3 / 6)``; without slip ``S = k_a A / (k_a +
beta)``. ``k_a -> 0`` is the free end (``L = A / rho - 1 / beta``),
``k_a -> inf`` the rigid anchor.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import minimize

import cablexsec as cx
from cablexsec.analysis.beam_test import (
    AlongWireSection,
    AlongWireSpec,
    BendingProtocol,
    ThreePoint,
    simulate_bending_test,
)
from cablexsec.analysis.sea_state_batch import _termination_record
from cablexsec.analysis.termination import (
    TerminationSpec,
    termination_friction_increment,
)
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics import along_wire as aw
from cablexsec.mechanics import clamp_layer as cl
from cablexsec.mechanics.bend_restrictor import CurvatureField
from cablexsec.mechanics.papailiou import slip_onset_curvature
from cablexsec.provenance import Provenance

pytestmark = pytest.mark.analytical_sanity
compiled = pytest.mark.skipif(not aw.HAVE_NUMBA, reason="needs the compiled kernels")

A_LOAD = 3.0


def _layer(k_hat: float = aw.RIGID_K_HAT) -> aw.WireChainLayer:
    return aw.WireChainLayer(
        name="unit",
        n_wires=1,
        r_helix_m=0.05,
        lay_angle_rad=math.radians(30.0),
        kappa_c_1_m=1.0,
        k_hat=k_hat,
        lam=0.0,
        force_unit_N=1.0,
        wire_area_m2=1.0,
        ea_N=1.0,
    )


def _closed_form(k_a: float, beta: float, a: float, rho: float = 1.0):
    """Anchor force, slip length and dissipation (module notes)."""
    inv_b = 0.0 if math.isinf(beta) else 1.0 / beta
    if k_a == 0.0:
        length = max(a / rho - inv_b, 0.0)
        if length == 0.0:
            return 0.0, 0.0, 0.0
        force = 0.0
    else:
        c2 = 0.5 * k_a * rho
        c1 = rho * (1.0 + k_a * inv_b)
        c0 = k_a * rho * inv_b**2 + rho * inv_b - a
        if c0 >= 0.0:
            return k_a * a / (k_a + beta), 0.0, 0.0
        length = (-c1 + math.sqrt(c1 * c1 - 4.0 * c2 * c0)) / (2.0 * c2)
        force = a - rho * inv_b - rho * length
    return force, length, rho**2 * (0.5 * length**2 * inv_b + length**3 / 6.0)


def _ramp(a: float = A_LOAD, steps: int = 120) -> np.ndarray:
    return np.linspace(0.0, a, steps + 1)[1:]


def _cycle(amp: float, steps: int = 24) -> np.ndarray:
    return np.r_[
        np.linspace(0, amp, steps + 1)[1:],
        np.linspace(amp, -amp, 2 * steps + 1)[1:],
        np.linspace(-amp, amp, 2 * steps + 1)[1:],
    ]


def _soft(r: np.ndarray | float, c: float) -> np.ndarray:
    return np.sign(r) * np.maximum(np.abs(r) - c, 0.0)


# --------------------------------------------------------------------------- #
# Projections                                                                  #
# --------------------------------------------------------------------------- #
def _generic_projection(y, cap, left, right):
    """SLSQP of the projection with elastic end anchors ``(weight, centre)``."""
    m = y.size

    def cost(z):
        f = 0.5 * np.sum((z - y) ** 2)
        if left is not None:
            f += 0.5 * left[0] * _soft(z[0] - left[1], cap[0]) ** 2
        if right is not None:
            f += 0.5 * right[0] * _soft(z[-1] - right[1], cap[-1]) ** 2
        return f

    def grad(z):
        g = z - y
        if left is not None:
            g[0] += left[0] * _soft(z[0] - left[1], cap[0])
        if right is not None:
            g[-1] += right[0] * _soft(z[-1] - right[1], cap[-1])
        return g

    rows, rhs = [], []
    for i in range(1, m):
        r = np.zeros(m)
        r[i], r[i - 1] = -1.0, 1.0
        rows += [r, -r]
        rhs += [cap[i], cap[i]]
    a_mat, b_vec = np.array(rows), np.array(rhs)
    return minimize(
        cost,
        np.zeros(m),
        jac=grad,
        constraints=[
            {"type": "ineq", "fun": lambda z: b_vec + a_mat @ z, "jac": lambda z: a_mat}
        ],
        method="SLSQP",
        options={"ftol": 1e-15, "maxiter": 2000},
    ).x


@pytest.mark.parametrize("seed", [3, 11])
def test_spring_projections_match_a_generic_quadratic_programme(seed):
    rng = np.random.default_rng(seed)
    m = 7
    y = rng.normal(size=m) * 2.0
    cap = np.abs(rng.normal(size=m + 1)) * 0.4 + 0.05
    cap[0] = 0.0 if seed == 11 else cap[0]
    left = (2.5, 0.3)
    right = (0.7, -0.4)
    work = [np.zeros(4 * m + 8) for _ in range(5)]
    for lw, rw in ((left, None), (None, right), (left, right)):
        out = np.empty(m)
        cl._project_ends(
            y,
            cap,
            False,
            False,
            out,
            *work,
            0.0 if lw is None else lw[0],
            0.0 if lw is None else lw[1],
            0.0 if rw is None else rw[0],
            0.0 if rw is None else rw[1],
        )
        ref = _generic_projection(y, cap, lw, rw)
        assert np.abs(out - ref).max() < 1e-5
    # the clamp-layer spring step: uniform interior bound, half cap at node 0
    bound = 0.3
    caps = np.full(m + 1, bound)
    caps[0] = 0.5 * bound
    ref = _generic_projection(y, caps, (left[0] / 1.0, left[1]), None)
    work = cl._spring_workspace(m)
    _d, force = cl._spring_step(y, bound, 1.0, left[0], left[1], *work)
    assert np.abs(work[1] - ref).max() < 1e-5
    assert force == pytest.approx(
        work[1][0] - np.clip(work[1][0] - left[1], -0.15, 0.15)
    )


def test_right_end_root_handles_walls_and_coincident_breakpoints():
    # g with a jump at 0 (a free-end wall upstream) and the spring centre there
    xs = np.array([-1.0, 0.0, 0.0, 2.0])
    gs = np.array([-3.0, -1.0, 0.5, 2.5])
    assert cl._end_root(xs, gs, 4, 1.0, 1.0, 2.0, 0.0, 0.0) == 0.0
    # roots beyond the last and before the first breakpoint
    for shift, centre in ((-10.0, 5.0), (10.0, -5.0)):
        g = gs + shift
        root = cl._end_root(xs, g, 4, 1.0, 1.0, 1.0, centre, 0.5)
        value = cl._g_at(xs, g, 4, 1.0, 1.0, root) + float(_soft(root - centre, 0.5))
        assert abs(value) < 1e-12
        assert (root > xs[-1]) if shift < 0 else (root < xs[0])


# --------------------------------------------------------------------------- #
# Closed forms                                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k_a", [10.0, 1.0, 0.1, 0.0])
def test_clamp_solver_spring_anchor_closed_form(k_a):
    """Rigid stick: anchor force, slip length and dissipation."""
    h = 2.0 * math.pi / 256
    m = int(14.0 / h)
    n, _ref, dissipation = cl.anchored_wire_response(
        np.ones(m), _ramp(), h, anchor_stiffness=k_a
    )
    force, length, d_exact = _closed_form(k_a, math.inf, A_LOAD)
    last = n[-1]
    # segment 0 carries the anchor force plus the first half segment's friction
    assert last[0] - 0.5 * h == pytest.approx(force, abs=1e-4 * A_LOAD)
    slipping = np.count_nonzero(np.abs(np.diff(last)) >= h * (1.0 - 1e-9))
    assert abs(slipping * h - length) <= 1.5 * h
    assert dissipation == pytest.approx(d_exact, rel=2e-3, abs=1e-4)


@compiled
@pytest.mark.parametrize(("k_stick", "beta"), [(aw.RIGID_K_HAT, math.inf), (4.0, 2.0)])
@pytest.mark.parametrize("k_a", [10.0, 1.0, 0.1, 0.0])
def test_chain_spring_anchor_closed_form(k_stick, beta, k_a):
    """Rigid and elastic (shear-lag) stick: the along-wire chain."""
    lay = _layer(k_stick)
    chain = aw.WireChain.build(
        lay,
        0.0,
        14.0 / (2.0 * math.pi) * lay.pitch_m,
        segments_per_pitch=512,
        anchor_stiffness_N_m=(k_a / lay.anchor_stiffness_hat(1.0), None),
    )
    m = chain.segments
    state = (np.zeros((1, m + 1)), np.zeros((1, m + 1)), np.zeros((1, m)))
    state = (*state, np.zeros((1, m + 1)))
    e_old = np.zeros((1, m))
    dissipation = 0.0
    for a in _ramp():
        e_new = np.full((1, m), a)
        state, energy = aw.chain_layer_step(
            chain, e_new, e_old, 1.0, state, np.zeros((1, m + 1))
        )
        e_old = e_new
        dissipation += float(energy[0, 1])
    force, length, d_exact = _closed_form(k_a, beta, A_LOAD)
    anchor = float(chain.spring[0] * state[0][0, 0])
    assert anchor == pytest.approx(force, abs=1e-4 * A_LOAD)
    slipping = np.count_nonzero(np.abs(np.abs(state[3][0]) - 1.0) < 1e-9)
    assert abs(slipping * chain.h - length) <= 1.5 * chain.h
    assert dissipation == pytest.approx(d_exact, rel=5e-4, abs=1e-4)


@compiled
@pytest.mark.parametrize("k_a", [0.0, 0.3, 3.0])
def test_clamp_and_chain_spring_anchors_agree(k_a):
    """Two implementations of the same projection: round-off."""
    lay = _layer()
    path = _cycle(6.0)
    x = np.array([0.0, 6.0 * lay.pitch_m])
    hist = aw.along_wire_history(
        lay,
        x,
        np.tile(path, (2, 1)),
        segments_per_pitch=63,
        max_increment=math.inf,
        anchor_stiffness_N_m=(k_a / lay.anchor_stiffness_hat(1.0), None),
    )
    n_cl, _ref, d_cl = cl.anchored_wire_response(
        hist.chain.cos_seg[0].copy(), path, hist.chain.h, anchor_stiffness=k_a
    )
    scale = np.abs(n_cl).max()
    assert np.abs(hist.force[0].T - n_cl).max() < 1e-12 * scale
    d_chain = hist.energy_J[0, -1, 1] / lay.energy_unit_J
    assert d_chain == pytest.approx(d_cl, rel=1e-9)


def test_numpy_and_compiled_spring_kernels_agree():
    h = 2.0 * math.pi / 31
    m = 60
    phi = h * (np.arange(m) + 0.5)
    e = np.cos(phi)
    path = _cycle(5.0, 12)
    rho = 1.0 + 0.2 * np.cos(np.arange(path.size))
    cap = np.arcsin(np.abs(np.cos(phi + 0.3)))
    record = (np.arange(path.size) % 3 == 0).astype(np.int64)
    index = np.array([0, 5, m - 1], dtype=np.int64)
    rows = int(record.sum())
    drive = np.outer(path, 1.0 + 0.3 * np.sin(phi / 7.0))
    counts = np.full(path.size, 2, dtype=np.int64)
    for compliance in (math.inf, 0.7, 1e-3):
        outs = [np.zeros((rows, 3)) for _ in range(4)]
        args = (e, cap, path, rho, h, record, index)
        d1 = cl._anchored_kernel(*args, outs[0], outs[1], compliance)
        d2 = cl._anchored_numpy(*args, outs[2], outs[3], compliance)
        assert np.abs(outs[0] - outs[2]).max() < 1e-12 * np.abs(outs[0]).max()
        assert d1 == pytest.approx(d2, rel=1e-12)
        fouts = [np.zeros((path.size, 3)) for _ in range(4)]
        fargs = (drive, np.cos(phi), cap, counts, rho, h, index)
        f1 = cl._field_kernel(*fargs, fouts[0], fouts[1], compliance)
        f2 = cl._field_numpy(*fargs, fouts[2], fouts[3], compliance)
        assert np.abs(fouts[0] - fouts[2]).max() < 1e-12 * np.abs(fouts[0]).max()
        assert f1 == pytest.approx(f2, rel=1e-12)


# --------------------------------------------------------------------------- #
# Limits                                                                       #
# --------------------------------------------------------------------------- #
def test_rigid_anchor_is_the_unchanged_code_path():
    h = 2.0 * math.pi / 31
    m = 60
    e = np.cos(h * (np.arange(m) + 0.5))
    path = _cycle(5.0, 12)
    base = cl.anchored_wire_response(e, path, h)
    drive = np.outer(path, np.ones(m))
    field_base = cl.anchored_wire_field_response(drive, e, h)
    for k_a in (None, math.inf):
        same = cl.anchored_wire_response(e, path, h, anchor_stiffness=k_a)
        assert np.array_equal(same[0], base[0]) and same[2] == base[2]
        field = cl.anchored_wire_field_response(drive, e, h, anchor_stiffness=k_a)
        assert np.array_equal(field[0], field_base[0])
        assert field[2] == field_base[2]
    assert cl.anchor_compliance(None) == cl.anchor_compliance(math.inf) == 0.0
    assert cl.anchor_compliance(0.0) == math.inf
    assert cl.anchor_compliance(4.0) == 0.25
    lay = _layer()
    chain = aw.WireChain.build(lay, 0.0, lay.pitch_m, segments_per_pitch=16)
    rigid = aw.WireChain.build(
        lay,
        0.0,
        lay.pitch_m,
        segments_per_pitch=16,
        anchor_stiffness_N_m=(math.inf, None),
    )
    assert np.array_equal(chain.fixed, rigid.fixed)
    assert not chain.spring.any() and not rigid.spring.any()
    assert rigid.anchor_stiffness_N_m == (math.inf, None)


@compiled
def test_rigid_anchor_reproduces_the_chain_and_bending_test_bit_for_bit():
    lay = _layer(0.26)
    x = np.array([0.0, 3.0 * lay.pitch_m])
    kx = np.tile(_cycle(4.0, 10), (2, 1))
    base = aw.along_wire_history(lay, x, kx, segments_per_pitch=16)
    same = aw.along_wire_history(
        lay, x, kx, segments_per_pitch=16, anchor_stiffness_N_m=(math.inf, None)
    )
    assert np.array_equal(base.force, same.force)
    assert np.array_equal(base.energy_J, same.energy_J)
    sec = AlongWireSection((lay,), 2.0)
    prot = BendingProtocol.cyclic(
        "displacement", 0.02 * lay.pitch_m, n_cycles=1, samples_per_cycle=12
    )
    spec = AlongWireSpec(ends=("anchored", "anchored"), segments_per_pitch=16)
    r0 = simulate_bending_test(
        sec, ThreePoint(2 * lay.pitch_m), prot, stations_per_segment=6, along_wire=spec
    )
    r1 = simulate_bending_test(
        sec,
        ThreePoint(2 * lay.pitch_m),
        prot,
        stations_per_segment=6,
        along_wire=AlongWireSpec(
            ends=("anchored", "anchored"),
            segments_per_pitch=16,
            k_anchor_N_m=(None, math.inf),
        ),
    )
    assert np.array_equal(r0.moment_Nm, r1.moment_Nm)


@compiled
def test_zero_stiffness_is_the_free_end():
    lay = _layer(0.26)
    x = np.array([0.0, 3.0 * lay.pitch_m])
    kx = np.tile(_cycle(4.0, 10), (2, 1))
    free = aw.along_wire_history(lay, x, kx, ends=("free", "anchored"))
    zero = aw.along_wire_history(lay, x, kx, anchor_stiffness_N_m=(0.0, None))
    assert np.array_equal(free.force, zero.force)
    assert np.array_equal(free.energy_J, zero.energy_J)
    # the clamp solver's zero-stiffness anchor against the chain's free end
    rigid = _layer()
    path = _cycle(6.0)
    hist = aw.along_wire_history(
        rigid,
        np.array([0.0, 6.0 * rigid.pitch_m]),
        np.tile(path, (2, 1)),
        ends=("free", "anchored"),
        segments_per_pitch=63,
        max_increment=math.inf,
    )
    n_cl, _ref, _d = cl.anchored_wire_response(
        hist.chain.cos_seg[0].copy(), path, hist.chain.h, anchor_stiffness=0.0
    )
    assert np.abs(hist.force[0].T - n_cl).max() < 1e-12 * np.abs(n_cl).max()


@pytest.fixture(scope="module")
def menard_layer():
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    return cx.cable_to_fatigue_layers(cable, contact_tension_N=50e3)[0]


@compiled
def test_termination_solvers_agree_and_bracket_the_anchor_stiffness(menard_layer):
    """Clamp and along-wire solvers (same convention) for every stiffness,
    the rigid limit approached from below and the free end below every
    finite anchor."""
    spec = menard_layer.spec
    kc = slip_onset_curvature(spec)
    kx = 20 * kc * np.sin(np.linspace(0, 4 * math.pi, 81))
    ea = spec.E_wire_Pa * spec.A_wire_m2 * spec.xi
    unit = spec.r_helix_m / (ea * abs(math.sin(spec.lay_angle_rad)))
    rigid = termination_friction_increment(spec, kx, 0 * kx, TerminationSpec(0.2))
    same = termination_friction_increment(
        spec, kx, 0 * kx, TerminationSpec(0.2, k_anchor_N_m=math.inf)
    )
    assert np.array_equal(rigid.stress_Pa, same.stress_Pa)
    scale = np.abs(rigid.stress_Pa).max()
    ranges = []
    errors = []
    for k_hat in (0.0, 0.1, 1.0, 10.0, 1e3, 1e6):
        term = TerminationSpec(0.2, k_anchor_N_m=k_hat / unit)
        a = termination_friction_increment(spec, kx, 0 * kx, term)
        b = termination_friction_increment(
            spec,
            kx,
            0 * kx,
            TerminationSpec(0.2, k_anchor_N_m=k_hat / unit, solver="along_wire"),
        )
        assert np.abs(a.stress_Pa - b.stress_Pa).max() < 1e-12 * scale
        assert "anchor" in a.note
        ranges.append(float(np.ptp(a.stress_Pa, axis=1).max()))
        errors.append(float(np.abs(a.stress_Pa - rigid.stress_Pa).max()) / scale)
    assert all(
        r1 <= r2 * (1 + 1e-9) for r1, r2 in zip(ranges, ranges[1:], strict=False)
    )
    assert ranges[-1] <= float(np.ptp(rigid.stress_Pa, axis=1).max()) * (1 + 1e-9)
    assert errors[-1] < 1e-4
    assert all(e2 <= e1 for e1, e2 in zip(errors, errors[1:], strict=False))


@compiled
def test_termination_field_and_capstan_routes_take_the_anchor(menard_layer):
    """A softer anchor lowers the clamp increment under a non-proportional
    field (both solvers) and with the capstan bound."""
    spec = menard_layer.spec
    kc = slip_onset_curvature(spec)
    t = np.linspace(0, 4 * math.pi, 61)
    kx = 15 * kc * np.sin(t)
    x = np.linspace(0.0, 2.0, 41)
    growth = 0.5 + 0.3 * np.sin(0.5 * t)
    shape = 1.0 + growth[None, :] * (np.exp(-x[:, None]) - math.exp(-0.2))
    n = t.size
    fld = CurvatureField(
        x,
        shape * kx[None, :],
        None,
        "test",
        np.zeros(n),
        np.zeros(n, dtype=np.int64),
        np.zeros(n, dtype=bool),
        math.inf,
        np.zeros(n),
        Provenance(method="test"),
    )
    soft = 1e5
    options = (
        {"curvature_field": fld},
        {"curvature_field": fld, "solver": "along_wire"},
        {"solver": "along_wire", "stick": "spring", "capstan": True},
    )
    for option in options:
        base = termination_friction_increment(
            spec, kx, 0 * kx, TerminationSpec(0.2, **option)
        )
        sprung = termination_friction_increment(
            spec, kx, 0 * kx, TerminationSpec(0.2, k_anchor_N_m=soft, **option)
        )
        assert sprung.applied and "anchor" in sprung.note
        assert np.isfinite(sprung.stress_Pa).all()
        assert (
            np.ptp(sprung.stress_Pa, axis=1).max()
            < np.ptp(base.stress_Pa, axis=1).max()
        )


# --------------------------------------------------------------------------- #
# Energy and the two-way bending test                                          #
# --------------------------------------------------------------------------- #
@compiled
@pytest.mark.parametrize("k_stick", [aw.RIGID_K_HAT, 0.26])
@pytest.mark.parametrize("lam", [0.0, 0.3])
def test_energy_identity_with_elastic_anchors(k_stick, lam):
    lay = aw.WireChainLayer(
        name="unit",
        n_wires=3,
        r_helix_m=0.05,
        lay_angle_rad=math.radians(30.0),
        kappa_c_1_m=1.0,
        k_hat=k_stick,
        lam=lam,
        force_unit_N=1.0,
        wire_area_m2=1.0,
        ea_N=1.0,
    )
    x = np.linspace(0.0, 4.0 * lay.pitch_m, 5)
    shape = np.exp(-x / lay.pitch_m)[:, None]
    kx = shape * _cycle(8.0, 16)[None, :]
    unit = lay.anchor_stiffness_hat(1.0)
    for stiffness in ((0.7 / unit, 2.0 / unit), (0.05 / unit, None)):
        hist = aw.along_wire_history(
            lay, x, kx, segments_per_pitch=24, anchor_stiffness_N_m=stiffness
        )
        work = np.abs(hist.energy_J[..., 0]).max()
        assert np.abs(hist.balance_residual_J).max() < 1e-12 * work
        assert (np.diff(hist.energy_J[..., 1], axis=1) >= -1e-12 * work).all()


@compiled
def test_spring_anchor_jacobian_equals_the_finite_difference_in_stick():
    lay = _layer(0.26)
    unit = lay.anchor_stiffness_hat(1.0)
    chain = aw.WireChain.build(
        lay,
        0.0,
        lay.pitch_m,
        segments_per_pitch=16,
        anchor_stiffness_N_m=(0.8 / unit, 3.0 / unit),
    )
    stations = 9
    x_all = np.linspace(0.0, chain.x_end_m, stations)
    pos = np.clip(np.searchsorted(x_all, chain.x_seg_m, side="right") - 1, 0, 7)
    frac = np.clip((chain.x_seg_m - x_all[pos]) / (x_all[pos + 1] - x_all[pos]), 0, 1)
    j0, fj = chain.record_weights(x_all)
    cos_st = np.cos(lay.azimuth(x_all[None, :], np.zeros((1, 1))))

    def strain(kappa):
        k_seg = kappa[pos] * (1.0 - frac) + kappa[pos + 1] * frac
        return np.ascontiguousarray(chain.cos_seg * k_seg)

    def moment(n):
        vals = n[:, j0] * (1.0 - fj) + n[:, j0 + 1] * fj
        return lay.lever_Nm * np.sum(cos_st * vals, axis=0)

    def step(state, k_old, k_new):
        guess = np.zeros_like(state[0])
        new, _ = aw.chain_layer_step(
            chain, strain(k_new), strain(k_old), 20.0, state, guess
        )
        return new, guess

    m = chain.segments
    zero = (np.zeros((1, m + 1)), np.zeros((1, m + 1)), np.zeros((1, m)))
    zero = (*zero, np.zeros((1, m + 1)))
    kappa0 = 0.3 * np.cos(np.linspace(0.0, 1.0, stations))
    state1, _ = step(zero, np.zeros(stations), kappa0)
    kappa1 = kappa0 + 0.1
    new, solution = step(state1, kappa0, kappa1)
    jac = aw.chain_layer_jacobian(
        chain, state1, solution, new[2], 20.0, pos, frac, j0, fj, cos_st
    )
    base = moment(new[2])
    fd = np.empty_like(jac)
    for j in range(stations):
        kp = kappa1.copy()
        kp[j] += 1e-3
        moved, _ = step(state1, kappa0, kp)
        fd[:, j] = (moment(moved[2]) - base) / 1e-3
    np.testing.assert_allclose(jac, fd, rtol=0, atol=1e-7 * np.abs(fd).max())
    # rigid interface: the stick mask must not count the spring as friction
    rigid = aw.WireChain.build(
        _layer(),
        0.0,
        lay.pitch_m,
        segments_per_pitch=16,
        anchor_stiffness_N_m=(0.8 / unit, None),
    )
    zr = (np.zeros((1, m + 1)), np.zeros((1, m + 1)), np.zeros((1, m)))
    zr = (*zr, np.zeros((1, m + 1)))
    guess = np.zeros((1, m + 1))
    e1 = np.ascontiguousarray(rigid.cos_seg * 0.3)
    new_r, _ = aw.chain_layer_step(rigid, e1, 0 * e1, 20.0, zr, guess)
    jr = aw.chain_layer_jacobian(
        rigid, zr, guess, new_r[2], 20.0, pos, frac, j0, fj, cos_st
    )
    assert np.isfinite(jr).all()


@pytest.fixture(scope="module")
def cable33():
    return cx.load_cable(cx.inter_array_single_core_yaml_path())


@compiled
def test_two_way_bending_test_with_elastic_end_fittings(cable33):
    """Anchored short specimen: equilibrium, closed-cycle work against the
    friction work, the free and rigid limits."""
    sec = AlongWireSection.from_cable(cable33)
    lay = sec.layers[0]
    p = max(layer.pitch_m for layer in sec.layers)
    prot = BendingProtocol.cyclic(
        "displacement", 0.02 * p, n_cycles=2, samples_per_cycle=40
    )

    def run(ends, k_anchor):
        return simulate_bending_test(
            sec,
            ThreePoint(2 * p),
            prot,
            stations_per_segment=8,
            along_wire=AlongWireSpec(
                ends=ends,
                overhang_m=(0.2 * p, 0.2 * p),
                segments_per_pitch=32,
                k_anchor_N_m=k_anchor,
            ),
        )

    anchored = ("anchored", "anchored")
    k_mid = 1.0 / lay.anchor_stiffness_hat(1.0)  # one radian of wire
    free = run(("free", "free"), (None, None))
    zero = run(anchored, (0.0, 0.0))
    assert np.array_equal(free.moment_Nm, zero.moment_Nm)
    rigid = run(anchored, (None, None))
    mid = run(anchored, (k_mid, k_mid))
    assert mid.equilibrium_residual < 1e-6
    cyc, fr = mid.cycle_external_work_J(), mid.cycle_friction_work_J()
    assert fr is not None and abs(cyc - fr) / fr < 0.01
    peak = [float(np.abs(r.load_N).max()) for r in (free, mid, rigid)]
    assert peak[0] <= peak[1] <= peak[2]


# --------------------------------------------------------------------------- #
# Inputs and records                                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [-1.0, math.nan, True, "1e8"])
def test_termination_anchor_stiffness_validation(bad):
    with pytest.raises(CableInputError):
        TerminationSpec(0.1, k_anchor_N_m=bad)


def test_anchor_stiffness_input_errors():
    with pytest.raises(CableInputError):
        TerminationSpec(0.1, anchor="none", k_anchor_N_m=1e8)
    assert TerminationSpec(0.1, k_anchor_N_m=2).k_anchor_N_m == 2.0
    for bad in ((-1.0, None), (math.nan, None), (True, None)):
        with pytest.raises(CableInputError):
            AlongWireSpec(ends=("anchored", "anchored"), k_anchor_N_m=bad)
    with pytest.raises(CableInputError):
        AlongWireSpec(k_anchor_N_m=(1e8, None))  # free ends
    with pytest.raises(CableInputError):
        AlongWireSpec(ends=("anchored", "anchored"), k_anchor_N_m=(1e8,))  # type: ignore[arg-type]
    spec = AlongWireSpec(ends=("anchored", "free"), k_anchor_N_m=(5, None))
    assert spec.k_anchor_N_m == (5.0, None)
    lay = _layer()
    for ends, k in (
        (("free", "anchored"), (1e8, None)),
        (("anchored", "anchored"), (-1.0, None)),
        (("anchored", "anchored"), (1e8,)),
    ):
        with pytest.raises(CableInputError):
            aw.WireChain.build(lay, 0.0, lay.pitch_m, ends=ends, anchor_stiffness_N_m=k)
    with pytest.raises(CableInputError):
        cl.anchor_compliance(-1.0)
    with pytest.raises(CableInputError):
        cl.anchored_wire_response(
            np.ones(4), np.ones(2), 0.1, anchor_stiffness=math.nan
        )


def test_anchor_stiffness_survives_the_section_record():
    spec = TerminationSpec(0.3, k_anchor_N_m=2.5e8)
    record = _termination_record(spec)
    assert record["k_anchor_N_m"] == 2.5e8
    assert TerminationSpec(**record) == spec
    record.pop("k_anchor_N_m")
    assert TerminationSpec(**record).k_anchor_N_m is None  # older records
