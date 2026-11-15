# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Distributed elastic--Coulomb armour friction (``friction_model="distributed"``).

Reference values marked "stick_check" are those recorded by a separate,
independently verified solver of the same discrete problem: 256 nodes per
pitch, segment midpoints at ``(j + 1/2) h``, backward Euler, the same paths.
The capstan bound is that of the wire's signed force, ``b = rho g + lam n``
([Papailiou1997] Eqs. 9-14 with the signed helix angle); the values with
``lam > 0`` are recomputed by the NumPy port of that solver
(:class:`_Reference`), which reproduces the external solver's recorded value
for the former ``lam |n|`` bound (26.579392168691143) to 1e-15 when given
that bound.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import solve_banded
from scipy.optimize import fsolve

import cablexsec as cx
from cablexsec.analysis.fatigue import compute_fatigue, wire_stress_field
from cablexsec.cable_adapters import cable_to_fatigue_layers
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics import distributed_friction as dfm
from cablexsec.mechanics.fast_slip import wire_friction_stress

PI = math.pi
compiled = pytest.mark.skipif(
    not dfm.HAVE_NUMBA, reason="long histories need the compiled kernel"
)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _ramp(a_to, steps):
    s = np.linspace(0.0, 1.0, steps + 1)[1:]
    return s * a_to[0], s * a_to[1]


def _rotation(a, revolutions, steps_per_rev, ramp_steps=200):
    rx, ry = _ramp((a, 0.0), ramp_steps)
    th = 2 * PI * np.arange(1, revolutions * steps_per_rev + 1) / steps_per_rev
    return np.concatenate([rx, a * np.cos(th)]), np.concatenate([ry, a * np.sin(th)])


def _kernel(ax, ay, khat, lam, nodes, *, rho=None, g=None, phase0=None, newton=True):
    """Normalised kernel on the stick_check grid (segments at (j + 1/2) h)."""
    ax = np.ascontiguousarray(ax, dtype=float)
    ay = np.ascontiguousarray(ay, dtype=float)
    h = 2 * PI / nodes
    out = dfm._distributed_kernel(
        ax,
        ay,
        np.ones(ax.size) if rho is None else np.ascontiguousarray(rho, dtype=float),
        float(khat),
        float(lam),
        np.ones(nodes) if g is None else np.ascontiguousarray(g, dtype=float),
        h / 2 if phase0 is None else phase0,
        1.0,
        math.inf,
        np.zeros(1, dtype=np.int64),
        np.zeros(1),
        True,
        newton,
    )
    assert out[3][0] == 0
    return out


def _layers(path=None):
    cable = cx.load_cable(path or cx.menard2023_yaml_path(), geometry_override=None)
    return cable_to_fatigue_layers(cable, contact_tension_N=0)


def _cap(spec):
    sa = abs(math.sin(spec.lay_angle_rad))
    width = 2 * PI * spec.r_helix_m / spec.n_wires
    sigma0 = (
        spec.r_helix_m
        * spec.contact_pressure_Pa
        * width
        * math.cos(spec.lay_angle_rad)
        / (spec.A_wire_m2 * sa**2)
    )
    psi = np.arcsin(np.clip(np.abs(np.cos(spec.phi_wires)), 0, 1))
    # Signed capstan: sigma0 expm1(lam psi) on the tension side,
    # -sigma0 (1 - exp(-lam psi)) = sigma0 expm1(-lam psi) on the other.
    return sigma0 * np.expm1(spec.mu * sa * psi * np.sign(np.cos(spec.phi_wires)))


def _sigma0(spec):
    sa = abs(math.sin(spec.lay_angle_rad))
    width = 2 * PI * spec.r_helix_m / spec.n_wires
    return (
        spec.r_helix_m
        * spec.contact_pressure_Pa
        * width
        * math.cos(spec.lay_angle_rad)
        / (spec.A_wire_m2 * sa**2)
    )


def _kappa_on(spec):
    return dfm.distributed_friction_history(spec, np.zeros(2), np.zeros(2)).parameters[
        "kappa_on"
    ]


def _rot_closed(a):
    """Rigid stick, constant bound: dissipation per revolution (DERIVATION eq. 13)."""
    if a >= math.hypot(1.0, PI / 2):
        return 8 * PI * math.sqrt(a * a - PI * PI / 4)
    pa = math.asin(1 / a)
    c0 = math.sqrt(a * a - 1) + pa
    lo, hi = PI - pa, PI + pa
    f = lambda x: a * math.cos(x) - (c0 - x)  # noqa: E731
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if f(lo) * f(mid) <= 0 else (mid, hi)
    pe = 0.5 * (lo + hi)
    s = c0 * (pe - pa) - (pe * pe - pa * pa) / 2 - a * (math.sin(pe) - math.sin(pa))
    return 4 * PI * s


class _Reference:
    """NumPy port of the independent ``stick_check`` solver (semismooth Newton
    on the active set with a banded Sherman--Morrison solve, fixed point on the
    capstan bound), extended by the pressure ratio and the contact shape."""

    def __init__(self, nodes, khat, lam, g=None):
        self.N, self.k, self.lam = nodes, khat, lam
        self.h = 2 * PI / nodes
        phi = self.h * np.arange(nodes) + self.h / 2
        self.cs, self.sn = np.cos(phi), np.sin(phi)
        self.g = np.ones(nodes) if g is None else np.asarray(g, float)
        self.u, self.p, self.n = (np.zeros(nodes) for _ in range(3))
        self.ax = self.ay = 0.0

    def _solve(self, d, c, rhs):
        n = self.N
        gam = -d[0]
        ab = np.zeros((3, n))
        ab[0, 1:] = c
        ab[2, :-1] = c
        ab[1] = d
        ab[1, 0] = d[0] - gam
        ab[1, -1] = d[-1] - c * c / gam
        u = np.zeros(n)
        u[0], u[-1] = gam, c
        yq = solve_banded((1, 1), ab, np.column_stack([rhs, u]))
        y, q = yq[:, 0], yq[:, 1]
        return y - ((y[0] + c / gam * y[-1]) / (1.0 + q[0] + c / gam * q[-1])) * q

    def _newton(self, y, z0, b, v):
        h, k = self.h, self.k
        ftr = y - np.roll(y, 1)
        prev = None
        for _ in range(60):
            r = z0 + v
            el = np.abs(k * r) < b
            s = np.sign(r)
            key = (el.tobytes(), (s * ~el).tobytes())
            if key == prev:
                return v
            prev = key
            assert el.any(), "reference port needs a stick node"
            rhs = ftr - h * k * el * z0 - h * b * s * (~el)
            v = self._solve(2.0 / h + h * k * el, -1.0 / h, rhs)
        raise AssertionError("active set did not settle")

    def advance(self, ax, ay, rho):
        h, k = self.h, self.k
        y = ax * self.cs + ay * self.sn + (np.roll(self.u, -1) - self.u) / h
        z0 = self.u - self.p
        n = self.n.copy()
        v = np.zeros(self.N)
        for _ in range(100):
            b = np.maximum(rho * self.g + self.lam * 0.5 * (n + np.roll(n, 1)), 0.0)
            v = self._newton(y, z0, b, v)
            n_new = y + (np.roll(v, -1) - v) / h
            done = np.abs(n_new - n).max() <= 1e-12 * max(1.0, np.abs(n_new).max())
            n = n_new
            if self.lam == 0.0 or done:
                break
        b = np.maximum(rho * self.g + self.lam * 0.5 * (n + np.roll(n, 1)), 0.0)
        tau = np.clip(k * (z0 + v), -b, b)
        self.p = self.u + v - tau / k
        self.u = self.u + v
        self.n = n
        return n


# --------------------------------------------------------------------------- #
# (1) agreement with the reference results                                     #
# --------------------------------------------------------------------------- #
@compiled
@pytest.mark.parametrize(
    ("khat", "lutchansky", "d_rev"),
    [
        (0.33, 3.0193389859767272e-05, 28.609458055098784),
        (3.0, 1.0039049199273009e-05, 7.622586096093627),
    ],
)
def test_stick_check_verification_cases(khat, lutchansky, d_rev):
    xi = khat / (1 + khat)
    # Below slip the field is the Lutchansky shear-lag solution a xi cos(phi).
    a = 0.8 / xi
    ax, ay = _ramp((a, 0.0), 50)
    fld = _kernel(ax, ay, khat, 0.0, 256)[4]
    phi = (np.arange(256) + 0.5) * 2 * PI / 256
    err = np.abs(fld[-1] - a * xi * np.cos(phi)).max()
    assert err == pytest.approx(lutchansky, rel=1e-8)
    # Steady rotation at a_bar = 1.3: dissipation of the 4th revolution.
    ax, ay = _rotation(1.3 / xi, 4, 720)
    hist = _kernel(ax, ay, khat, 0.0, 256)[1]
    d = hist[:, 1]
    assert d[-1] - d[-721] == pytest.approx(d_rev, rel=1e-10)
    identity = hist[:, 0] - hist[:, 1] - hist[:, 2] - hist[:, 3]
    assert np.abs(identity).max() <= 1e-13 * np.abs(hist[:, 0]).max()


@compiled
def test_stick_check_rigid_limit_and_capstan_cases():
    ax, ay = _rotation(2.0 * 1.001, 3, 720)
    hist = _kernel(ax, ay, 1000.0, 0.0, 256)[1]
    assert hist[-1, 1] - hist[-721, 1] == pytest.approx(30.622844477479035, rel=1e-10)
    # Capstan case, signed bound: the NumPy port of stick_check gives
    # 29.42502785927062 (the former lam |n| bound: 26.579392168691143). The
    # falling force of the compression side lowers its bound, so this partial
    # slip rotation dissipates more than with the former bound.
    ax, ay = _rotation(1.3 / (0.33 / 1.33), 3, 720)
    reference = _Reference(256, 0.33, 0.127)
    dissipated = []
    for x, y in zip(ax, ay, strict=True):
        p_old = reference.p.copy()
        reference.advance(x, y, 1.0)
        tau = reference.k * (reference.u - reference.p)
        dissipated.append(reference.h * float(np.sum(tau * (reference.p - p_old))))
    d_ref = float(np.sum(dissipated[-720:]))
    assert d_ref == pytest.approx(29.42502785927062, rel=1e-10)
    for newton in (True, False):
        hist = _kernel(ax, ay, 0.33, 0.127, 256, newton=newton)[1]
        d_rev = hist[-1, 1] - hist[-721, 1]
        assert d_rev == pytest.approx(d_ref, rel=1e-10)


@compiled
@pytest.mark.parametrize("a", [1.3, 3.0])
def test_steady_rotation_tends_to_rigid_closed_form(a):
    """Dissipation per revolution -> DERIVATION eq. (13) as k_hat -> infinity."""
    closed = _rot_closed(a)
    errors = []
    for khat in (1e2, 1e3, 1e4):
        xi = khat / (1 + khat)
        ax, ay = _rotation(a / xi, 3, 720)
        hist = _kernel(ax, ay, khat, 0.0, 256)[1]
        errors.append(abs((hist[-1, 1] - hist[-721, 1]) / closed - 1))
    assert errors[2] < errors[1] < errors[0]
    if a < math.hypot(1.0, PI / 2):  # partial slip: first order in 1/k_hat
        assert errors[0] / errors[1] > 5 and errors[1] / errors[2] > 5
        assert errors[2] < 1e-3
    else:  # full slip
        assert errors[2] < 5e-3


def _rigid_partial_slip(lam, a):
    """Rigid-stick limit of monotonic planar loading, signed capstan bound.

    With ``x = pi/2 - |phi|`` from the neutral axis toward the tension side
    the stick zones carry ``n = a sin x``; the slip zone ``-x_c < x < x_t``
    round the neutral axis has ``dn/dx = b = 1 + lam n``, so
    ``n = C exp(lam x) - 1/lam``. ``n`` is continuous at both slip fronts,
    and ``u`` vanishes in both stick zones, so the slip zone adds no net
    displacement: ``int (n - a sin x) dx = 0`` over it. These three equations
    fix ``C``, ``x_t`` and ``x_c``; the zone is asymmetric (it reaches
    further into the compression side, whose bound falls with its force).
    With the former ``lam |n|`` bound the solution was the pointwise
    minimum of ``a |cos phi|`` and ``expm1(lam psi) / lam``.
    """

    def equations(v):
        c, xt, xc = v
        return [
            c * math.exp(lam * xt) - 1 / lam - a * math.sin(xt),
            c * math.exp(-lam * xc) - 1 / lam + a * math.sin(xc),
            c * (math.exp(lam * xt) - math.exp(-lam * xc)) / lam
            - (xt + xc) / lam
            + a * (math.cos(xt) - math.cos(xc)),
        ]

    c, xt, xc = fsolve(equations, [1 / lam, 0.8, 0.8], xtol=1e-12)
    assert np.abs(equations([c, xt, xc])).max() < 1e-12
    # The stick zones are admissible: |dn/dx| = a cos x within the bound.
    for side, edge in ((1.0, xt), (-1.0, xc)):
        x = np.linspace(edge, PI / 2, 200)
        assert np.all(a * np.cos(x) <= 1 + side * lam * a * np.sin(x))
    return c, xt, xc


@compiled
def test_planar_capstan_limit_partial_slip():
    """Monotonic planar loading -> the rigid-stick partial-slip field."""
    lam, a = 0.127, 1.3
    nodes = 256
    phi = (np.arange(nodes) + 0.5) * 2 * PI / nodes
    c, xt, xc = _rigid_partial_slip(lam, a)
    assert xc > xt
    x = PI / 2 - np.abs(np.angle(np.exp(1j * phi)))
    ref = np.where((x > -xc) & (x < xt), c * np.exp(lam * x) - 1 / lam, a * np.sin(x))
    errors = []
    for khat in (1e2, 1e3, 1e4):
        ax = np.linspace(0, a * (1 + khat) / khat, 401)[1:]
        fld = _kernel(ax, np.zeros_like(ax), khat, lam, nodes)[4]
        errors.append(np.abs(fld[-1] - ref).max())
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 2e-3


@compiled
def test_planar_full_slip_closed_form():
    """Physical units, full slip. With the signed bound ``dn/dphi = +-(g +
    lam n)`` between the extreme fibres and the zero pitch mean of ``n``
    (periodic ``u``) give ``sigma = sigma0 (C exp(lam x) - 1)``,
    ``C = lam pi / (2 sinh(lam pi / 2))``, ``x = pi/2 - |phi|`` from the
    geometric neutral axis toward tension; the direction-tied law holds every
    wire at its own bound, ``sigma0 expm1(+-lam psi)``."""
    spec = _layers()[0].spec
    assert spec.shape_g is None
    lam = spec.mu * abs(math.sin(spec.lay_angle_rad))
    x = PI / 2 - np.abs(np.angle(np.exp(1j * spec.phi_wires)))
    c = lam * PI / (2 * math.sinh(lam * PI / 2))
    closed = _sigma0(spec) * (c * np.exp(lam * x) - 1)
    kappa = np.linspace(0, 1000 * _kappa_on(spec), 301)[1:]
    errors = []
    for per_wire in (4, 8):
        stress = dfm.distributed_wire_friction_stress(
            spec, kappa, np.zeros_like(kappa), nodes_per_pitch=per_wire * spec.n_wires
        )
        errors.append(np.abs(stress[:, -1] - closed).max() / np.abs(closed).max())
    assert errors[1] < 1e-6
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.05)  # second order
    cap = _cap(spec)
    direction_tied = wire_friction_stress(spec, kappa, np.zeros_like(kappa))
    np.testing.assert_allclose(
        direction_tied[:, -1], cap, rtol=0, atol=1e-9 * np.abs(cap).max()
    )


# --------------------------------------------------------------------------- #
# (2) planar histories: distributed versus the direction-tied cap              #
# --------------------------------------------------------------------------- #
def _planar(spec, abar, cycles=3, per_cycle=200):
    t = np.linspace(0, 2 * PI * cycles, cycles * per_cycle + 1)[1:]
    kx = abar * _kappa_on(spec) * np.sin(t)
    ky = np.zeros_like(kx)
    dist = dfm.distributed_wire_friction_stress(spec, kx, ky)
    tied = wire_friction_stress(spec, kx, ky)
    last = slice(-2 * per_cycle, None)
    return dist[:, last], tied[:, last]


@compiled
@pytest.mark.parametrize("abar", [0.5, 0.9])
def test_planar_stick_agrees_with_direction_tied(abar):
    dist, tied = _planar(_layers()[0].spec, abar)
    assert np.abs(dist - tied).max() <= 1e-3 * np.abs(tied).max()


@compiled
@pytest.mark.parametrize(
    ("abar", "range_ratio"),
    [(1.2, 0.9425), (1.5, 0.8266), (3.0, 0.9042), (10.0, 0.9719), (30.0, 0.9896)],
)
def test_planar_partial_slip_stick_representation_effect(abar, range_ratio):
    """Documented difference (not forced to agree): with the shear-lag interface
    the slip front and the elastic zone at the extreme fibre differ from the
    pointwise xi-reduced spring slider. Ratio = largest per-wire stress range,
    distributed / direction-tied, Menard inner armour, default N. Recorded with
    the signed capstan bound (the former bound: 0.9513, 0.8430, 0.8917,
    0.9696, 0.9900); the full-slip limit is the closed form of the next test."""
    dist, tied = _planar(_layers()[0].spec, abar)
    ratio = np.ptp(dist, axis=1).max() / np.ptp(tied, axis=1).max()
    assert ratio == pytest.approx(range_ratio, abs=2e-3)


@compiled
def test_full_slip_range_ratio_is_x_over_sinh_x():
    """Full-slip limit of the ratio above. The extreme-fibre wire reverses
    between its own bounds in the direction-tied law, ``2 sigma0 sinh(x)``,
    ``x = lam pi / 2``, and over ``2 sigma0 C sinh(x) = sigma0 lam pi`` in the
    distributed one (the closed form of the full-slip field): the ratio is
    ``x / sinh(x)``. With the former ``lam |n|`` bound both ranges were
    ``2 sigma0 expm1(x)`` and the ratio tended to one."""
    spec = _layers()[0].spec
    x = spec.mu * abs(math.sin(spec.lay_angle_rad)) * PI / 2
    dist, tied = _planar(spec, 100.0)
    ratio = np.ptp(dist, axis=1).max() / np.ptp(tied, axis=1).max()
    assert ratio == pytest.approx(x / math.sinh(x), rel=1e-9)
    assert ratio < 1.0 - 1e-3


# --------------------------------------------------------------------------- #
# (3) energy balance                                                           #
# --------------------------------------------------------------------------- #
def _random_history(spec, n=600, seed=4, amp=4.0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 40, n)
    k0 = amp * _kappa_on(spec)
    kx = k0 * (np.sin(0.8 * t) + 0.3 * rng.standard_normal(n))
    ky = k0 * (np.cos(0.55 * t + 1.0) + 0.3 * rng.standard_normal(n))
    pressure = spec.contact_pressure_Pa * (1 + 0.4 * np.sin(0.3 * t))
    return kx, ky, pressure


@compiled
def test_energy_balance_biaxial_varying_clamping():
    spec = _layers()[1].spec
    spec = replace(spec, shape_g=1 + 0.3 * np.cos(3 * spec.phi_wires))
    kx, ky, pressure = _random_history(spec)
    hist = dfm.distributed_friction_history(spec, kx, ky, contact_pressure_Pa=pressure)
    work = hist.endpoint_work_J_per_m
    assert np.abs(hist.balance_residual_J_per_m).max() <= 1e-10 * np.abs(work).max()
    assert np.all(np.diff(hist.dissipation_J_per_m) >= -1e-12 * work.max())
    assert hist.dissipation_J_per_m[-1] > 0
    assert hist.slip_fraction.max() > 0.5


@compiled
def test_steady_rotation_cycle_work_equals_dissipation():
    remainder = []
    for steps in (720, 1440):
        ax, ay = _rotation(2.0 / (0.33 / 1.33), 4, steps)
        hist = _kernel(ax, ay, 0.33, 0.127, 128)[1]
        w, d, u, r = (hist[-1, k] - hist[-1 - steps, k] for k in range(4))
        assert abs(u) <= 1e-8 * d  # periodic state
        assert (w - d) / d == pytest.approx(r / d, abs=1e-12)
        remainder.append(r / d)
    # Work over a closed cycle = dissipation + the backward-Euler remainder,
    # which is first order in the step.
    assert remainder[0] < 1e-2
    assert remainder[0] / remainder[1] == pytest.approx(2.0, rel=0.05)


# --------------------------------------------------------------------------- #
# (4) compiled kernel = reference                                              #
# --------------------------------------------------------------------------- #
def test_kernel_matches_numpy_reference():
    nodes, khat, lam, steps = 48, 0.33, 0.127, 120
    t = np.linspace(0, 4, steps)
    ax = 2.5 * np.sin(1.3 * t) + 0.8 * t
    ay = 2.0 * np.cos(0.9 * t) - 2.0
    rho = 1 + 0.25 * np.sin(2.0 * t)
    g = 1 + 0.2 * np.cos(2 * PI * np.arange(nodes) / nodes)
    fld = _kernel(ax, ay, khat, lam, nodes, rho=rho, g=g)[4]
    fld_fp = _kernel(ax, ay, khat, lam, nodes, rho=rho, g=g, newton=False)[4]
    ref = _Reference(nodes, khat, lam, g)
    expected = np.array(
        [ref.advance(x, y, r) for x, y, r in zip(ax, ay, rho, strict=True)]
    )
    scale = np.abs(expected).max()
    assert np.abs(fld - expected).max() <= 1e-10 * scale
    assert np.abs(fld_fp - expected).max() <= 1e-10 * scale


# --------------------------------------------------------------------------- #
# (6) default behaviour, wiring and validation                                 #
# --------------------------------------------------------------------------- #
def _short(n=400):
    t = np.linspace(0, 40, n)
    return 0.02 * np.sin(0.7 * t), 0.012 * np.cos(0.4 * t)


def test_default_is_direction_tied_bitwise():
    layer = _layers()[0]
    kx, ky = _short()
    default = wire_stress_field(layer, kx, ky)
    explicit = wire_stress_field(layer, kx, ky, friction_model="direction_tied")
    for name in ("mean_Pa", "radial_Pa", "tangential_Pa"):
        assert np.array_equal(getattr(default, name), getattr(explicit, name))
    r0 = compute_fatigue(_layers(), kx, ky)
    r1 = compute_fatigue(_layers(), kx, ky, friction_model="direction_tied")
    for name in r0.D_total_per_wire:
        assert np.array_equal(r0.D_total_per_wire[name], r1.D_total_per_wire[name])
    assert r0.provenance == r1.provenance
    assert "distributed" not in str(r0.provenance)


@compiled
def test_public_api_option_and_provenance():
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    kx, ky = _short()
    nodes = {"inner_armour": 110, "outer_armour": 126}
    res = cx.compute_fatigue(
        cable,
        kx,
        ky,
        friction_model="distributed",
        distributed_nodes_per_pitch=nodes,
    )
    text = str(res.provenance)
    assert "friction_model='distributed'" in text
    assert "inner_armour: N = 110" in text and "outer_armour: N = 126" in text
    assert "k_hat" in text and "lam" in text
    assert all(np.isfinite(v).all() for v in res.D_total_per_wire.values())
    with pytest.raises(CableInputError):
        cx.compute_fatigue(cable, kx, ky, friction_model="pointwise")
    with pytest.raises(CableInputError):
        cx.compute_fatigue(
            cable,
            kx,
            ky,
            friction_model="distributed",
            distributed_nodes_per_pitch={"inner_armour": 110},
        )


@compiled
def test_section_state_path_accepts_constant_stick_modulus():
    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    kx, ky = _short(200)
    tension = np.full(kx.size, 1.0e5)
    res = cx.compute_fatigue(
        cable,
        kx,
        ky,
        tension_N=tension,
        contact_model="qp",
        friction_model="distributed",
    )
    assert all(np.isfinite(v).all() for v in res.D_total_per_wire.values())


def test_validation():
    spec = _layers()[0].spec
    kx, ky = _short(20)
    with pytest.raises(CableInputError):
        dfm.distributed_friction_history(replace(spec, xi=1.0), kx, ky)
    with pytest.raises(CableInputError):
        dfm.distributed_friction_history(spec, kx, ky, nodes_per_pitch=4)
    with pytest.raises(CableInputError):
        dfm.distributed_friction_history(spec, kx, ky, max_increment=0.0)
    with pytest.raises(CableInputError):
        dfm.distributed_friction_history(
            spec, kx, ky, stick_modulus_Pa=np.linspace(1, 2, kx.size) * spec.k_f
        )
    same = dfm.distributed_friction_history(
        spec, kx, ky, stick_modulus_Pa=np.full(kx.size, spec.k_f)
    )
    assert np.array_equal(
        same.stress_Pa, dfm.distributed_wire_friction_stress(spec, kx, ky)
    )
    zero = dfm.distributed_friction_history(replace(spec, mu=0.0), kx, ky)
    assert not zero.stress_Pa.any()


@compiled
def test_contact_shape_and_pressure_scaling():
    spec = _layers()[0].spec
    kx, ky = _short()
    base = dfm.distributed_wire_friction_stress(spec, kx, ky)
    ones = dfm.distributed_wire_friction_stress(
        replace(spec, shape_g=np.ones(spec.n_wires)), kx, ky
    )
    assert np.array_equal(base, ones)
    doubled_shape = dfm.distributed_wire_friction_stress(
        replace(spec, shape_g=np.full(spec.n_wires, 2.0)), kx, ky
    )
    doubled_pressure = dfm.distributed_wire_friction_stress(
        spec, kx, ky, contact_pressure_Pa=np.full(kx.size, 2 * spec.contact_pressure_Pa)
    )
    np.testing.assert_allclose(
        doubled_shape, doubled_pressure, rtol=0, atol=1e-9 * np.abs(base).max()
    )


@compiled
def test_released_clamping_releases_friction_stress():
    spec = _layers()[0].spec
    kx, ky = _short()
    pressure = np.full(kx.size, spec.contact_pressure_Pa)
    pressure[250:] = 0.0
    stress = dfm.distributed_wire_friction_stress(
        spec, kx, ky, contact_pressure_Pa=pressure
    )
    assert np.abs(stress[:, 200]).max() > 0
    assert np.abs(stress[:, 260:]).max() <= 1e-9 * np.abs(stress[:, :250]).max()


@compiled
def test_nodes_per_pitch_convergence_and_interpolation():
    spec = _layers()[1].spec
    kx, ky, pressure = _random_history(spec, n=400, amp=2.0)
    ranges = {}
    for nodes in (
        2 * spec.n_wires,
        4 * spec.n_wires,
        8 * spec.n_wires,
        4 * spec.n_wires + 7,
    ):
        stress = dfm.distributed_wire_friction_stress(
            spec, kx, ky, contact_pressure_Pa=pressure, nodes_per_pitch=nodes
        )
        ranges[nodes] = np.ptp(stress[:, 100:], axis=1).max()
    n2, n4, n8, odd = (ranges[k] for k in ranges)
    assert abs(n4 / n8 - 1) < abs(n2 / n8 - 1) + 1e-12
    assert abs(n4 / n8 - 1) < 5e-3
    assert abs(odd / n8 - 1) < 1e-2  # wires sample the field by interpolation
