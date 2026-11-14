# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Slip redistribution along the wires (gap T2): verification.

T2-V1 infinite-length limit (periodic distributed kernel, distributed
sectional law in a beam test), T2-V2 one anchor at constant curvature
(clamp-layer solver, termination clamp solver), T2-V3 energy, T2-V4 mesh,
T2-V5 free ends, plus the Lutchansky closed form near an anchor and the
station Jacobian of the beam coupling.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import minimize

import cablexsec as cx
from cablexsec.analysis.beam_test import (
    AlongWireSection,
    AlongWireSpec,
    BendingProtocol,
    FourPoint,
    SectionLaw,
    ThreePoint,
    simulate_bending_test,
)
from cablexsec.analysis.termination import (
    TerminationSpec,
    termination_friction_increment,
)
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics import along_wire as aw
from cablexsec.mechanics import clamp_layer as cl
from cablexsec.mechanics import distributed_friction as dfm

pytestmark = pytest.mark.analytical_sanity
compiled = pytest.mark.skipif(not aw.HAVE_NUMBA, reason="needs the compiled kernels")


def _layer(
    k_hat: float = aw.RIGID_K_HAT, lam: float = 0.0, n_wires: int = 1
) -> aw.WireChainLayer:
    return aw.WireChainLayer(
        name="unit",
        n_wires=n_wires,
        r_helix_m=0.05,
        lay_angle_rad=math.radians(30.0),
        kappa_c_1_m=1.0,
        k_hat=k_hat,
        lam=lam,
        force_unit_N=1.0,
        wire_area_m2=1.0,
        ea_N=1.0,
    )


def _cycle(amp: float, steps: int = 30) -> np.ndarray:
    return np.r_[
        np.linspace(0, amp, steps + 1)[1:],
        np.linspace(amp, -amp, 2 * steps + 1)[1:],
        np.linspace(-amp, amp, 2 * steps + 1)[1:],
    ]


# --------------------------------------------------------------------------- #
# Chain                                                                        #
# --------------------------------------------------------------------------- #
def test_projection_with_free_ends_matches_generic_qp():
    rng = np.random.default_rng(7)
    for left, right in ((True, False), (False, True), (True, True), (False, False)):
        m = 7
        y = rng.normal(size=m) * 2.0
        cap = np.abs(rng.normal(size=m + 1)) * 0.4 + 0.05
        out = np.empty(m)
        work = [np.zeros(4 * m + 8) for _ in range(5)]
        aw._project_var(y, cap, left, right, out, *work)
        rows, rhs = [], []
        for i in range(1, m):
            r = np.zeros(m)
            r[i], r[i - 1] = -1.0, 1.0
            rows += [r, -r]
            rhs += [cap[i], cap[i]]
        for flag, j, c in ((left, 0, cap[0]), (right, m - 1, cap[m])):
            if flag:
                r = np.zeros(m)
                r[j] = 1.0
                rows += [r, -r]
                rhs += [c, c]
        a_mat, b_vec = np.array(rows), np.array(rhs)
        ref = minimize(
            lambda z, y=y: 0.5 * np.sum((z - y) ** 2),
            np.zeros(m),
            jac=lambda z, y=y: z - y,
            constraints=[
                {
                    "type": "ineq",
                    "fun": lambda z, a=a_mat, b=b_vec: b + a @ z,
                    "jac": lambda z, a=a_mat: a,
                }
            ],
            method="SLSQP",
            options={"ftol": 1e-15, "maxiter": 1000},
        )
        assert np.abs(out - ref.x).max() < 1e-5


@compiled
def test_rigid_chain_equals_clamp_layer_solver():
    """T2-V2a: one anchor, constant curvature (and a far anchor)."""
    lay = _layer()
    x = np.array([0.0, 8 * lay.pitch_m])
    path = _cycle(6.0, 24)
    hist = aw.along_wire_history(
        lay, x, np.tile(path, (2, 1)), segments_per_pitch=63, max_increment=math.inf
    )
    n_cl, _ref, _d = cl.anchored_wire_response(
        hist.chain.cos_seg[0].copy(), path, hist.chain.h
    )
    assert np.abs(hist.force[0].T - n_cl).max() < 1e-12 * np.abs(n_cl).max()


@compiled
@pytest.mark.parametrize(
    ("k_hat", "lam"), [(0.26, 0.0), (0.26, 0.1), (aw.RIGID_K_HAT, 0.1)]
)
def test_long_chain_middle_equals_periodic_distributed_kernel(k_hat, lam):
    """T2-V1a: the middle pitch of a long anchored chain is the sectional field."""
    lay = _layer(k_hat, lam)
    spp, n_p = 64, 16
    path = _cycle(30.0, 40)
    hist = aw.along_wire_history(
        lay,
        np.array([0.0, n_p * lay.pitch_m]),
        np.tile(path, (2, 1)),
        segments_per_pitch=spp,
        max_increment=math.inf,
    )
    mid = (n_p // 2) * spp
    n_mid = hist.force[0, mid : mid + spp, :].T
    xi = k_hat / (1 + k_hat)
    _out, _en, _s, stats, fld = dfm._distributed_kernel(
        path.copy(),
        np.zeros_like(path),
        np.ones_like(path),
        k_hat,
        lam,
        np.ones(spp),
        hist.chain.h / 2,
        xi,
        math.inf,
        np.zeros(1, dtype=np.int64),
        np.zeros(1),
        True,
    )
    assert stats[0] == 0
    # the rigid chain is exact; the periodic kernel stands in for it with
    # k_hat = 1e4 (stick factor 0.9999)
    tol = 1e-9 if k_hat < aw.RIGID_K_HAT else 1e-3
    assert np.abs(n_mid - fld).max() < tol * np.abs(fld).max()


@compiled
@pytest.mark.parametrize(
    "ends", [("anchored", "anchored"), ("free", "free"), ("anchored", "free")]
)
@pytest.mark.parametrize(
    ("k_hat", "lam"), [(0.5, 0.0), (0.5, 0.08), (aw.RIGID_K_HAT, 0.08)]
)
def test_energy_identity(ends, k_hat, lam):
    """T2-V3a: W_be = dU + D + R to round-off, D >= 0."""
    lay = _layer(k_hat, lam, n_wires=3)
    p = lay.pitch_m
    x = np.array([0.0, 0.5 * p, 1.5 * p, 2.0 * p])
    shape = np.array([0.0, 1.0, 1.0, 0.2])
    path = _cycle(25.0, 20)
    hist = aw.along_wire_history(
        lay,
        x,
        shape[:, None] * path[None, :],
        ends=ends,
        segments_per_pitch=32,
        max_increment=0.5,
    )
    e = hist.energy_J
    scale = max(float(np.max(e[..., 1])), 1.0)
    assert np.abs(hist.balance_residual_J).max() < 1e-10 * scale
    assert np.all(np.diff(e[..., 1], axis=1) >= -1e-12 * scale)


@compiled
def test_free_end_force_vanishes_and_tends_to_periodic():
    """T2-V5: n = 0 beyond a free end; the middle tends to the periodic field."""
    lay = _layer(0.26, 0.0)
    spp = 64
    path = _cycle(30.0, 40)
    errors = []
    for n_p in (8, 16):
        hist = aw.along_wire_history(
            lay,
            np.array([0.0, n_p * lay.pitch_m]),
            np.tile(path, (2, 1)),
            ends=("free", "free"),
            segments_per_pitch=spp,
            max_increment=math.inf,
            record_x_m=np.array([0.0, n_p * lay.pitch_m]),
        )
        h = hist.chain.h
        # the end segment carries at most the bound of half a node
        assert np.abs(hist.force[0]).max() <= 0.75 * h * (1 + 1e-9) + 1e-12
        full = aw.along_wire_history(
            lay,
            np.array([0.0, n_p * lay.pitch_m]),
            np.tile(path, (2, 1)),
            ends=("free", "free"),
            segments_per_pitch=spp,
            max_increment=math.inf,
        )
        mid = (n_p // 2) * spp
        _o, _e, _s, _st, fld = dfm._distributed_kernel(
            path.copy(),
            np.zeros_like(path),
            np.ones_like(path),
            0.26,
            0.0,
            np.ones(spp),
            h / 2,
            0.26 / 1.26,
            math.inf,
            np.zeros(1, dtype=np.int64),
            np.zeros(1),
            True,
        )
        errors.append(np.abs(full.force[0, mid : mid + spp, :].T - fld).max())
    assert errors[1] < errors[0] or errors[1] < 1e-9


@compiled
def test_spring_anchor_matches_lutchansky_closed_form():
    """Stick regime, uniform curvature from an anchor: Lutchansky (1969).

    n = a k/(1+k) [cos(phi + phi0) - sin(phi0) exp(-sqrt(k) phi)/sqrt(k)]
    (DERIVATION section 2 of the curvature-gradient study, beta = i).
    """
    k = 0.5
    lay = _layer(k, 0.0)
    spp = 256
    a = 0.4  # far field a xi = 0.13 < 1: stick
    hist = aw.along_wire_history(
        lay,
        np.array([0.0, 10 * lay.pitch_m]),
        np.full((2, 1), a),
        segments_per_pitch=spp,
        max_increment=math.inf,
    )
    h = hist.chain.h
    phi = (np.arange(hist.chain.segments) + 0.5) * h
    sk = math.sqrt(k)
    # phase phi0 = 0 at the anchor (sin phi0 = 0): no offset term at the clamp
    exact = a * k / (1 + k) * np.cos(phi)
    num = hist.force[0, :, 0]
    near = phi < 4 * math.pi
    assert np.abs(num[near] - exact[near]).max() < 2e-3 * a
    lay2 = replace(lay, phase0_rad=math.pi / 2)
    hist2 = aw.along_wire_history(
        lay2,
        np.array([0.0, 10 * lay.pitch_m]),
        np.full((2, 1), a),
        segments_per_pitch=spp,
        max_increment=math.inf,
    )
    exact2 = a * k / (1 + k) * (np.cos(phi + math.pi / 2) - np.exp(-sk * phi) / sk)
    assert np.abs(hist2.force[0, near, 0] - exact2[near]).max() < 5e-3 * a


def test_layer_conventions_from_package_spec():
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    spec = cx.cable_to_fatigue_layers(cable, contact_tension_N=50e3)[0].spec
    spring = aw.WireChainLayer.from_layer_spec(spec, stick="spring")
    rigid = aw.WireChainLayer.from_layer_spec(spec, stick="rigid")
    scaled = aw.WireChainLayer.from_layer_spec(spec, stick="scaled")
    xi = spec.xi
    assert spring.kappa_c_1_m == pytest.approx(rigid.kappa_c_1_m, rel=1e-14)
    assert scaled.kappa_c_1_m == pytest.approx(rigid.kappa_c_1_m / xi, rel=1e-14)
    assert spring.k_hat == pytest.approx(xi / (1 - xi), rel=1e-14)
    assert rigid.rigid and scaled.rigid and not spring.rigid
    assert scaled.ea_N == pytest.approx(xi * rigid.ea_N, rel=1e-14)
    # far-field stick stiffness of the spring convention is the package's
    far = spring.k_hat / (1 + spring.k_hat) / spring.kappa_c_1_m
    assert far == pytest.approx(1 / scaled.kappa_c_1_m, rel=1e-12)
    with pytest.raises(CableInputError):
        aw.WireChainLayer.from_layer_spec(spec, stick="soft")


# --------------------------------------------------------------------------- #
# Beam coupling                                                                #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cable33():
    return cx.load_cable(cx.inter_array_single_core_yaml_path())


@compiled
def test_station_jacobian_matches_finite_difference(cable33):
    from cablexsec.analysis.along_wire_test import _LayerState

    sec = AlongWireSection.from_cable(cable33, stick="spring", capstan=False)
    lay = sec.layers[0]
    length = 2 * lay.pitch_m
    x_all = np.linspace(0.0, length, 17)
    st = _LayerState(aw.WireChain.build(lay, 0.0, length, segments_per_pitch=16), x_all)
    kap = 0.2 * lay.kappa_c_1_m * np.sin(np.pi * x_all / length)  # stick
    _new, _en, mom, jac, _e = st.trial(kap, True)
    assert jac is not None
    eps = 1e-3 * lay.kappa_c_1_m
    for j in (3, 8, 12):
        dk = np.zeros_like(kap)
        dk[j] = eps
        _n2, _e2, mom2, _j2, _ee = st.trial(kap + dk, False)
        assert np.allclose(
            (mom2 - mom) / eps, jac[:, j], rtol=1e-6, atol=1e-9 * np.abs(jac).max()
        )


@compiled
def test_long_specimen_zone_tends_to_distributed_law(cable33):
    """T2-V1b: a long four-point specimen reproduces the sectional law."""
    sec = AlongWireSection.from_cable(cable33)
    p = max(lay.pitch_m for lay in sec.layers)
    amp = 0.9
    prot = BendingProtocol.cyclic("curvature", amp, n_cycles=2, samples_per_cycle=40)
    law = SectionLaw.from_cable(cable33, friction_model="distributed")
    errors = []
    for n_p in (4, 12):
        arr = FourPoint(n_p * p, n_p * p / 2)
        r = simulate_bending_test(
            cable33,
            arr,
            prot,
            stations_per_segment=8,
            along_wire=AlongWireSpec(ends=("anchored", "anchored")),
        )
        r1 = simulate_bending_test(law, arr, prot, stations_per_segment=8)
        errors.append(
            np.abs(r.moment_Nm - r1.moment_Nm).max() / np.abs(r1.moment_Nm).max()
        )
        assert r.equilibrium_residual < 1e-6
        cyc, fr = r.cycle_external_work_J(), r.cycle_friction_work_J()
        assert fr is not None and abs(cyc - fr) / fr < 0.01  # T2-V3b
    assert errors[1] < errors[0]
    assert errors[1] < 5e-3


@compiled
def test_short_specimen_mesh_convergence(cable33):
    """T2-V4: loop work of a two-pitch three-point specimen."""
    sec = AlongWireSection.from_cable(cable33)
    p = max(lay.pitch_m for lay in sec.layers)
    prot = BendingProtocol.cyclic(
        "displacement", 0.02 * p, n_cycles=2, samples_per_cycle=40
    )
    works = []
    for spp, n_st in ((32, 8), (64, 16)):
        r = simulate_bending_test(
            sec,
            ThreePoint(2 * p),
            prot,
            stations_per_segment=n_st,
            along_wire=AlongWireSpec(
                ends=("free", "free"),
                overhang_m=(0.2 * p, 0.2 * p),
                segments_per_pitch=spp,
            ),
        )
        works.append(r.loop_work_J_per_m())
    assert abs(works[1] - works[0]) / works[1] < 0.01


def test_along_wire_spec_and_dispatch_validation(cable33):
    with pytest.raises(CableInputError):
        AlongWireSpec(ends=("anchored", "glued"))
    with pytest.raises(CableInputError):
        AlongWireSpec(overhang_m=(-1.0, 0.0))
    with pytest.raises(CableInputError):
        AlongWireSpec(stick="soft")
    prot = BendingProtocol.cyclic("curvature", 0.1, n_cycles=1, samples_per_cycle=8)
    with pytest.raises(CableInputError, match="replaces the sectional armour law"):
        simulate_bending_test(
            cable33,
            FourPoint(2.0, 1.0),
            prot,
            friction_model="progressive",
            along_wire=AlongWireSpec(),
        )
    law = SectionLaw.elastic(1.0e3)
    with pytest.raises(CableInputError):
        simulate_bending_test(
            law, FourPoint(2.0, 1.0), prot, along_wire=AlongWireSpec()
        )


# --------------------------------------------------------------------------- #
# Termination                                                                  #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def menard_layer():
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    return cx.cable_to_fatigue_layers(cable, contact_tension_N=50e3)[0]


@compiled
def test_termination_along_wire_reproduces_clamp_solver(menard_layer):
    """T2-V2b: same convention, same mesh and phases -> round-off."""
    from cablexsec.mechanics.papailiou import slip_onset_curvature

    spec = menard_layer.spec
    kc = slip_onset_curvature(spec)
    kx = 20 * kc * np.sin(np.linspace(0, 4 * math.pi, 81))
    for stick in ("scaled", "rigid"):
        a = termination_friction_increment(
            spec, kx, 0 * kx, TerminationSpec(0.2, stick=stick)
        )
        b = termination_friction_increment(
            spec, kx, 0 * kx, TerminationSpec(0.2, stick=stick, solver="along_wire")
        )
        assert (
            np.abs(a.stress_Pa - b.stress_Pa).max() < 1e-12 * np.abs(a.stress_Pa).max()
        )
    rigid = termination_friction_increment(
        spec, kx, 0 * kx, TerminationSpec(0.2, stick="rigid")
    )
    assert rigid.kappa_c_1_m == pytest.approx(spec.xi * kc, rel=1e-12)


def test_termination_option_validation():
    with pytest.raises(CableInputError):
        TerminationSpec(0.1, stick="spring")
    with pytest.raises(CableInputError):
        TerminationSpec(0.1, capstan=True)
    with pytest.raises(CableInputError):
        TerminationSpec(0.1, solver="fem")
    TerminationSpec(0.1, stick="spring", solver="along_wire", capstan=True)
