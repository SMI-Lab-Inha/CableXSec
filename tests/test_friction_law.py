# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Static/dynamic and rate-dependent friction law of the armour slip kernels.

Closed forms on one spring-slider (breakaway peak, velocity strengthening,
healing, steady viscous loop area ``pi eta omega x0^2``, steady logarithmic
rate), the energy identity with the viscous dissipation, neutral parameters
reproducing the rate-independent kernels bit for bit within each code path,
and the compiled direction-tied kernel against the NumPy reference.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.bending_law import export_bending_law
from cablexsec.analysis.fatigue import compute_fatigue, wire_stress_field
from cablexsec.analysis.termination import TerminationSpec
from cablexsec.cable_adapters import cable_to_fatigue_layers
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.mechanics import friction_law as flm
from cablexsec.mechanics.distributed_friction import (
    distributed_friction_history,
    distributed_wire_friction_stress,
)
from cablexsec.mechanics.fast_slip import wire_friction_stress
from cablexsec.mechanics.friction_law import (
    FrictionLaw,
    active_friction_law,
    friction_law_time_steps,
    slider_history,
)
from cablexsec.mechanics.mkappa import HelicalLayerSpec
from cablexsec.mechanics.papailiou import (
    compute_mkappa_wire_slip_front,
    full_slip_curvature,
    wire_slip_history,
    wire_slip_stress_bounds,
)

analytical = pytest.mark.analytical_sanity

BREAKAWAY = FrictionLaw(static_factor=1.4, dynamic_factor=1.0)
STRENGTHENING = FrictionLaw(static_factor=0.8, dynamic_factor=1.15, stick_time_s=0.3)
RATE = FrictionLaw(
    rate_coefficient=0.08,
    rate_reference_m_per_s=2.0e-5,
    viscous_velocity_m_per_s=4.0e-4,
)
COMBINED = FrictionLaw(
    static_factor=1.25,
    dynamic_factor=0.9,
    stick_time_s=0.2,
    rate_coefficient=0.05,
    rate_reference_m_per_s=1.0e-5,
    viscous_velocity_m_per_s=1.0e-3,
)
LAWS = [BREAKAWAY, STRENGTHENING, RATE, COMBINED]


def _spec(n_wires: int = 12, shape: bool = True) -> HelicalLayerSpec:
    g = None
    if shape:
        phi = 2 * np.pi * np.arange(n_wires) / n_wires
        g = 1.0 + 0.3 * np.cos(phi)
    return HelicalLayerSpec(
        n_wires=n_wires,
        r_helix_m=0.05,
        lay_angle_rad=0.4,
        E_wire_Pa=2.0e11,
        A_wire_m2=2.0e-5,
        xi=0.3,
        mu=0.2,
        F_n_axi_N=0.0,
        shape_g=g,
        contact_pressure_Pa=2.0e6,
    )


def _history(spec: HelicalLayerSpec, n: int = 1200, seed: int = 4):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 12.0, n)
    big = 3.0 * full_slip_curvature(spec)
    kx = big * np.sin(2 * np.pi * 0.25 * t) + 0.01 * big * rng.standard_normal(n)
    ky = 0.4 * big * np.sin(2 * np.pi * 0.13 * t)
    kx[200:240] = 0.0
    ky[200:240] = 0.0  # held zero curvature: the direction is held
    return t, kx, ky


def _time(law: FrictionLaw, t: np.ndarray) -> np.ndarray | None:
    return t if law.needs_time else None


# --------------------------------------------------------------------------- #
# Settings                                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [
        {"static_factor": 0.0},
        {"dynamic_factor": -1.0},
        {"static_factor": math.inf},
        {"stick_time_s": -1.0},
        {"stick_time_s": math.nan},
        {"rate_coefficient": -0.1},
        {"rate_coefficient": math.inf},
        {"rate_reference_m_per_s": 0.0},
        {"viscous_velocity_m_per_s": 0.0},
        {"viscous_velocity_m_per_s": math.nan},
        {"static_factor": True},
        {"dynamic_factor": "1"},
    ],
)
def test_settings_are_checked(kwargs):
    with pytest.raises(CableInputError):
        FrictionLaw(**kwargs)


def test_settings_properties():
    neutral = FrictionLaw()
    assert neutral.is_neutral and not neutral.needs_time
    assert "neutral" in neutral.describe()
    assert active_friction_law(None) is None
    assert active_friction_law(neutral) is None
    assert active_friction_law(BREAKAWAY) is BREAKAWAY
    with pytest.raises(CableInputError):
        active_friction_law("coulomb")  # type: ignore[arg-type]
    assert not BREAKAWAY.needs_time  # never heals: no time base
    assert not FrictionLaw(static_factor=1.2, stick_time_s=0.0).needs_time
    assert FrictionLaw(static_factor=1.2, stick_time_s=1.0).needs_time
    assert not FrictionLaw(stick_time_s=1.0).needs_time  # equal factors
    assert RATE.needs_time and RATE.rate_dependent and not RATE.state_dependent
    scaled = FrictionLaw(static_factor=1.2, dynamic_factor=1.2)
    assert not scaled.is_neutral and "static factor 1.2" in scaled.describe()
    assert "viscous" in RATE.describe() and "logarithmic" in RATE.describe()
    assert COMBINED.minimum_factor == 0.9
    assert FrictionLaw(static_factor=np.float64(1.1)).static_factor == 1.1
    np.testing.assert_allclose(
        RATE.rate_factor(np.array([0.0, 1.0e-4])),
        [1.0, 1.0 + 0.08 * math.log1p(5.0) + 0.25],
        rtol=1e-15,
    )


def test_time_base_is_checked():
    t = np.linspace(0.0, 1.0, 5)
    steps = friction_law_time_steps(RATE, t, 5)
    assert steps is not None
    np.testing.assert_array_equal(steps, np.full(5, 0.25))
    assert friction_law_time_steps(None, None, 5) is None
    assert friction_law_time_steps(BREAKAWAY, None, 5) is None
    with pytest.raises(CableInputError, match="give the sample times"):
        friction_law_time_steps(RATE, None, 5)
    with pytest.raises(CableInputError, match="used only"):
        friction_law_time_steps(BREAKAWAY, t, 5)
    with pytest.raises(CableInputError, match="used only"):
        friction_law_time_steps(None, t, 5)
    for bad in (t[:4], t[::-1], np.array([0.0, 0.1, 0.1, 0.2, 0.3])):
        with pytest.raises(CableInputError):
            friction_law_time_steps(RATE, bad, 5)
    with pytest.raises(CableInputError, match="real"):
        friction_law_time_steps(RATE, t + 0j, 5)
    with pytest.raises(CableInputError):
        friction_law_time_steps(RATE, np.array([0.0]), 1)


# --------------------------------------------------------------------------- #
# One spring-slider: closed forms                                              #
# --------------------------------------------------------------------------- #
@analytical
def test_slider_breakaway_peak():
    """Monotonic loading: elastic to the static bound, then the dynamic bound."""
    k, b, step = 2.0e6, 10.0, 1.0e-7
    drive = step * np.arange(1, 201)
    s = slider_history(drive, b, k, friction_law=BREAKAWAY)
    first = int(np.argmax(np.abs(s.slip_m) > 0.0))
    np.testing.assert_allclose(s.force_N[:first], k * drive[:first], rtol=1e-14)
    peak = s.force_N[first - 1]
    assert 1.4 * b - k * step < peak <= 1.4 * b
    np.testing.assert_allclose(s.force_N[first:], b, rtol=1e-15)
    # The breakaway step releases the overshoot as friction work.
    assert abs(s.balance_J).max() <= 1e-14 * s.input_work_J[-1]
    assert np.all(s.viscous_J == 0.0)


@analytical
def test_slider_velocity_strengthening_has_no_drop():
    k, b, step = 2.0e6, 10.0, 1.0e-7
    drive = step * np.arange(1, 201)
    s = slider_history(drive, b, k, friction_law=STRENGTHENING, time_s=drive * 10.0)
    assert np.all(np.diff(s.force_N) >= -1e-12 * b)
    # Elastic through the static bound, slip only at the dynamic bound.
    first = int(np.argmax(np.abs(s.slip_m) > 0.0))
    np.testing.assert_allclose(s.force_N[:first], k * drive[:first], rtol=1e-14)
    assert s.force_N[first - 1] > 0.8 * b
    np.testing.assert_allclose(s.force_N[first:], 1.15 * b, rtol=1e-15)


@analytical
@pytest.mark.parametrize(("hold_s", "heals"), [(0.5, True), (0.1, False)])
def test_slider_healing_time(hold_s, heals):
    """After ``t_stick`` of stick the static bound returns (a second peak)."""
    k, b, step = 2.0e6, 10.0, 1.0e-7
    law = FrictionLaw(static_factor=1.4, dynamic_factor=1.0, stick_time_s=0.3)
    up = step * np.arange(1, 101)
    hold = np.full(int(hold_s / 0.01), up[-1])
    drive = np.concatenate([up, hold, up[-1] + up])
    time = 0.01 * np.arange(drive.size)
    s = slider_history(drive, b, k, friction_law=law, time_s=time)
    second = s.force_N[up.size + hold.size :]
    if heals:
        assert second.max() > 1.3 * b
    else:
        assert second.max() <= b * (1 + 1e-15)
    np.testing.assert_allclose(second[-1], b, rtol=1e-15)


@analytical
def test_slider_steady_viscous_loop_area():
    """Sinusoidal full slip: viscous work per cycle ``pi eta omega x0^2``."""
    b, k, x0, omega = 1.0, 1.0e10, 1.0e-3, 2 * math.pi
    v_eta = 1.0e-3
    time = np.linspace(0.0, 3.0, 30_001)
    s = slider_history(
        x0 * np.sin(omega * time),
        b,
        k,
        friction_law=FrictionLaw(viscous_velocity_m_per_s=v_eta),
        time_s=time,
    )
    i0 = int(np.searchsorted(time, 2.0))
    eta = b / v_eta
    viscous = s.viscous_J[-1] - s.viscous_J[i0]
    friction = s.friction_J[-1] - s.friction_J[i0]
    assert viscous == pytest.approx(math.pi * eta * omega * x0**2, rel=1e-6)
    assert friction == pytest.approx(4.0 * b * x0, rel=1e-5)
    assert abs(s.balance_J).max() <= 1e-13 * s.input_work_J[-1]


@analytical
def test_slider_steady_logarithmic_rate():
    """Steady slip at ``v``: the force is ``m_d b R(v)`` (backward Euler exact)."""
    b, k, v = 3.0, 1.0e7, 2.0e-4
    law = FrictionLaw(
        dynamic_factor=1.1,
        static_factor=1.1,
        rate_coefficient=0.07,
        rate_reference_m_per_s=1.0e-5,
        viscous_velocity_m_per_s=5.0e-3,
    )
    time = np.linspace(0.0, 1.0, 2001)
    s = slider_history(v * time, b, k, friction_law=law, time_s=time)
    expected = 1.1 * b * float(law.rate_factor(v))
    np.testing.assert_allclose(s.force_N[-500:], expected, rtol=1e-12)
    assert abs(s.balance_J).max() <= 1e-13 * s.input_work_J[-1]


@analytical
@pytest.mark.parametrize("law", LAWS)
def test_slider_energy_identity(law):
    rng = np.random.default_rng(7)
    time = np.cumsum(rng.uniform(0.001, 0.02, 3000))
    drive = 2.0e-5 * np.cumsum(rng.standard_normal(3000))
    bound = 5.0 * (1.0 + 0.3 * np.sin(time))
    s = slider_history(drive, bound, 1.0e6, friction_law=law, time_s=_time(law, time))
    assert np.all(np.diff(s.friction_J) >= 0.0)
    assert np.all(np.diff(s.viscous_J) >= -1e-18)
    assert abs(s.balance_J).max() <= 1e-12 * max(s.input_work_J.max(), 1e-30)


def test_slider_inputs_and_neutral_coulomb():
    drive = 1.0e-5 * np.sin(np.linspace(0.0, 20.0, 400))
    s = slider_history(drive, 2.0, 1.0e6)
    # Rate-independent Coulomb return.
    f = 0.0
    previous = 0.0
    expected = []
    for d in drive:
        f = min(max(f + 1.0e6 * (d - previous), -2.0), 2.0)
        previous = d
        expected.append(f)
    np.testing.assert_allclose(s.force_N, expected, rtol=0, atol=1e-15 * 2.0)
    for args in (
        (np.array([np.nan]), 1.0, 1.0),
        (np.array([]), 1.0, 1.0),
        (drive + 0j, 1.0, 1.0),
        (drive, -1.0, 1.0),
        (drive, 1.0, 0.0),
    ):
        with pytest.raises(CableInputError):
            slider_history(*args)


def test_slip_return_array_matches_scalar():
    """The NumPy reference return and the compiled scalar return agree."""
    rng = np.random.default_rng(2)
    n = 400
    side = np.where(rng.random(n) > 0.5, 1.0, -1.0)
    c0 = rng.uniform(1e6, 5e7, n)
    c1 = rng.uniform(0.0, 1e7, n)
    x0 = rng.uniform(0.0, 0.3, n)
    x1 = rng.uniform(0.0, 0.3, n)
    a = 2.0 * (c0 + c1) * rng.uniform(0.0, 1.0, n)
    k, md, speed = 4.0e10, 0.9, 50.0
    vec = flm._slip_magnitude_array(
        a, k, md, side, c0, x0, c1, x1, speed, 0.06, 1e-5, 800.0
    )
    sca = np.array(
        [
            flm._slip_magnitude(
                a[i],
                k,
                md,
                side[i],
                c0[i],
                x0[i],
                c1[i],
                x1[i],
                0.0,
                speed,
                0.06,
                1e-5,
                800.0,
            )
            for i in range(n)
        ]
    )
    np.testing.assert_allclose(vec, sca, rtol=0, atol=1e-13 * np.max(a) / k)
    # Each root satisfies the backward-Euler equation.
    v = speed * vec
    m = md * (1 + 0.06 * np.log1p(v / 1e-5) + v * 800.0)
    up = c0 * np.expm1(m * x0) + c1 * np.expm1(m * x1)
    down = -(c0 * np.expm1(-m * x0) + c1 * np.expm1(-m * x1))
    bound = np.where(side > 0, up, down)
    slipping = vec > 0
    np.testing.assert_allclose(
        (a - k * vec)[slipping], bound[slipping], rtol=1e-12, atol=0
    )
    # Overflowing rates fall back to bisection and stay inside the bracket.
    huge = flm._slip_magnitude(
        1e9, 1e9, 1.0, 1.0, 1e8, 0.3, 0.0, 0.0, 0.0, 1e12, 0.0, 1.0, 1e9
    )
    assert 0.0 < huge < (1e9 - 1e8 * math.expm1(0.3)) / 1e9


# --------------------------------------------------------------------------- #
# Direction-tied kernel                                                        #
# --------------------------------------------------------------------------- #
def test_neutral_law_is_bitwise():
    spec = _spec()
    t, kx, ky = _history(spec)
    ref = wire_slip_history(spec, kx, ky)
    neutral = wire_slip_history(spec, kx, ky, friction_law=FrictionLaw())
    np.testing.assert_array_equal(neutral.stress_Pa, ref.stress_Pa)
    np.testing.assert_array_equal(neutral.moment_Nm, ref.moment_Nm)
    np.testing.assert_array_equal(
        neutral.work.friction_J_per_m, ref.work.friction_J_per_m
    )
    assert neutral.work.viscous_J_per_m is None
    np.testing.assert_array_equal(
        wire_friction_stress(spec, kx, ky, friction_law=FrictionLaw()),
        wire_friction_stress(spec, kx, ky),
    )
    with pytest.raises(CableInputError, match="used only"):
        wire_friction_stress(spec, kx, ky, time_s=t)


@pytest.mark.parametrize("factor", [0.7, 1.3])
def test_equal_factors_scale_the_coefficient(factor):
    """``m_s = m_d = m`` is the rate-independent law with ``m mu``."""
    spec = _spec()
    _t, kx, ky = _history(spec)
    law = FrictionLaw(static_factor=factor, dynamic_factor=factor)
    scaled = dataclasses.replace(spec, mu=spec.mu * factor)
    ref = wire_slip_history(scaled, kx, ky)
    got = wire_slip_history(spec, kx, ky, friction_law=law)
    scale = np.abs(ref.stress_Pa).max()
    np.testing.assert_allclose(got.stress_Pa, ref.stress_Pa, rtol=0, atol=1e-12 * scale)
    fast = wire_friction_stress(spec, kx, ky, friction_law=law)
    np.testing.assert_allclose(fast, ref.stress_Pa, rtol=0, atol=1e-12 * scale)


@pytest.mark.parametrize("law", LAWS)
@pytest.mark.parametrize("varying_pressure", [False, True])
def test_compiled_kernel_matches_reference(law, varying_pressure):
    spec = _spec()
    t, kx, ky = _history(spec)
    pressure = None
    if varying_pressure:
        pressure = spec.contact_pressure_Pa * (
            1 + 0.4 * np.sin(np.linspace(0, 9, t.size))
        )
    ref = wire_slip_history(
        spec,
        kx,
        ky,
        contact_pressure_Pa=pressure,
        friction_law=law,
        time_s=_time(law, t),
    )
    fast = wire_friction_stress(
        spec,
        kx,
        ky,
        contact_pressure_Pa=pressure,
        friction_law=law,
        time_s=_time(law, t),
    )
    scale = np.abs(ref.stress_Pa).max()
    np.testing.assert_allclose(fast, ref.stress_Pa, rtol=0, atol=1e-12 * scale)
    # The law changes the result (the history slips).
    plain = wire_slip_history(spec, kx, ky, contact_pressure_Pa=pressure)
    assert np.abs(ref.stress_Pa - plain.stress_Pa).max() > 1e-3 * scale


@analytical
@pytest.mark.parametrize("law", LAWS)
def test_kernel_energy_identity(law):
    spec = _spec()
    t, kx, ky = _history(spec)
    h = wire_slip_history(spec, kx, ky, friction_law=law, time_s=_time(law, t))
    work = h.work
    assert np.all(np.diff(work.friction_J_per_m) >= 0.0)
    if law.rate_dependent:
        assert work.viscous_J_per_m is not None
        assert work.viscous_J_per_m[-1] > 0.0
        assert np.all(
            np.diff(work.viscous_J_per_m) >= -1e-12 * work.viscous_J_per_m[-1]
        )
    else:
        assert work.viscous_J_per_m is None
    scale = work.dissipation_J_per_m[-1]
    assert np.abs(work.balance_residual_J_per_m).max() <= 1e-12 * scale


@analytical
def test_kernel_energy_identity_with_stiffness_change():
    spec = _spec()
    t, kx, ky = _history(spec, n=600)
    stiffness = spec.k_f * (1 + 0.2 * (t > 6.0))
    h = wire_slip_history(
        spec, kx, ky, stick_modulus_Pa=stiffness, friction_law=COMBINED, time_s=t
    )
    assert h.material_work_J_per_m[-1] != 0.0
    scale = h.work.dissipation_J_per_m[-1]
    assert np.abs(h.work.balance_residual_J_per_m).max() <= 1e-12 * scale


@analytical
def test_kernel_breakaway_peak_at_the_extreme_fibre():
    """Planar monotonic bending: the extreme-fibre wire peaks at its static
    capstan bound ``sigma0 g expm1(m_s lam pi/2)`` and drops to the dynamic one."""
    spec = _spec(shape=False)
    kappa = np.linspace(0.0, 2.0 * full_slip_curvature(spec), 4001)[1:]
    zero = np.zeros_like(kappa)
    h = wire_slip_history(spec, kappa, zero, friction_law=BREAKAWAY, keep_cap=True)
    wire = h.stress_Pa[0]
    static = wire_slip_stress_bounds(dataclasses.replace(spec, mu=1.4 * spec.mu), 0.0)[
        0
    ][0]
    dynamic = wire_slip_stress_bounds(spec, 0.0)[0][0]
    step = spec.k_f * spec.r_helix_m * (kappa[1] - kappa[0])
    assert static - step < wire.max() <= static
    np.testing.assert_allclose(wire[-1], dynamic, rtol=1e-14)
    # The kept cap is the bound of the end of each step.
    assert h.cap_Pa is not None
    assert np.all(np.abs(h.stress_Pa) <= h.cap_Pa * (1 + 1e-14))
    np.testing.assert_allclose(h.cap_Pa[0, -1], dynamic, rtol=1e-14)
    # The strengthening order rises without a drop.
    time = np.linspace(0.0, 4.0, kappa.size)
    s = wire_slip_history(spec, kappa, zero, friction_law=STRENGTHENING, time_s=time)
    assert np.all(np.diff(s.stress_Pa[0]) >= -1e-9 * dynamic)


def test_kept_cap_under_a_rate_law():
    spec = _spec()
    t, kx, ky = _history(spec, n=400)
    h = wire_slip_history(spec, kx, ky, friction_law=RATE, time_s=t, keep_cap=True)
    assert h.cap_Pa is not None
    slipping = h.cap_Pa * (1 - 1e-12) <= np.abs(h.stress_Pa)
    assert slipping.any()
    np.testing.assert_allclose(
        np.abs(h.stress_Pa)[slipping], h.cap_Pa[slipping], rtol=1e-12
    )
    assert np.all(np.abs(h.stress_Pa) <= h.cap_Pa * (1 + 1e-12) + 1e-9)


def test_zero_contact_and_never_slipping_wires():
    spec = dataclasses.replace(_spec(), contact_pressure_Pa=0.0)
    t, kx, ky = _history(_spec(), n=200)
    h = wire_slip_history(spec, kx, ky, friction_law=COMBINED, time_s=t)
    np.testing.assert_array_equal(h.stress_Pa, 0.0)
    fast = wire_friction_stress(spec, kx, ky, friction_law=COMBINED, time_s=t)
    np.testing.assert_array_equal(fast, 0.0)
    assert h.work.viscous_J_per_m is not None
    assert np.all(h.work.viscous_J_per_m == 0.0)
    # Small curvature: every wire sticks; the law changes nothing.
    small = 1e-3 * kx
    a = wire_slip_history(_spec(), small, 0 * ky, friction_law=COMBINED, time_s=t)
    b = wire_slip_history(_spec(), small, 0 * ky)
    np.testing.assert_array_equal(a.stress_Pa, b.stress_Pa)
    np.testing.assert_array_equal(a.work.friction_J_per_m, 0.0)


def test_rate_law_converges_with_the_time_step():
    """Halving the steps of the same path changes the loop at O(dt)."""
    spec = _spec(shape=False)
    big = 3.0 * full_slip_curvature(spec)
    results = []
    for n in (201, 401, 801):
        t = np.linspace(0.0, 4.0, n)
        kx = big * np.sin(2 * np.pi * 0.5 * t)
        h = wire_slip_history(spec, kx, 0 * kx, friction_law=RATE, time_s=t)
        results.append(h.work.dissipation_J_per_m[-1])
    e1, e2 = abs(results[0] - results[1]), abs(results[1] - results[2])
    assert e2 < 0.6 * e1
    assert e2 < 1e-2 * results[2]


# --------------------------------------------------------------------------- #
# Distributed kernel (static/dynamic option)                                   #
# --------------------------------------------------------------------------- #
def _short(spec, n=160):
    t = np.linspace(0.0, 8.0, n)
    big = 3.0 * full_slip_curvature(spec)
    return (
        t,
        big * np.sin(2 * np.pi * 0.25 * t),
        0.4 * big * np.cos(2 * np.pi * 0.13 * t),
    )


def test_distributed_neutral_and_scaled_law():
    spec = _spec()
    t, kx, ky = _short(spec)
    ref = distributed_friction_history(spec, kx, ky)
    neutral = distributed_friction_history(spec, kx, ky, friction_law=FrictionLaw())
    np.testing.assert_array_equal(neutral.stress_Pa, ref.stress_Pa)
    np.testing.assert_array_equal(neutral.dissipation_J_per_m, ref.dissipation_J_per_m)
    m = 1.25
    law = FrictionLaw(static_factor=m, dynamic_factor=m)
    got = distributed_friction_history(
        spec, kx, ky, friction_law=law, max_increment=math.inf
    )
    exp = distributed_friction_history(
        dataclasses.replace(spec, mu=spec.mu * m), kx, ky, max_increment=math.inf
    )
    scale = np.abs(exp.stress_Pa).max()
    np.testing.assert_allclose(got.stress_Pa, exp.stress_Pa, rtol=0, atol=1e-10 * scale)
    np.testing.assert_allclose(
        got.dissipation_J_per_m, exp.dissipation_J_per_m, rtol=1e-10
    )
    assert "friction_law" in got.parameters


@analytical
@pytest.mark.parametrize(
    "law",
    [
        BREAKAWAY,
        STRENGTHENING,
        FrictionLaw(static_factor=1.3, dynamic_factor=0.9, stick_time_s=0.0),
    ],
)
def test_distributed_static_dynamic_energy(law):
    spec = _spec()
    t, kx, ky = _short(spec)
    h = distributed_friction_history(
        spec, kx, ky, friction_law=law, time_s=_time(law, t), field_samples=[5, 50]
    )
    ref = distributed_friction_history(spec, kx, ky)
    scale = h.dissipation_J_per_m[-1]
    assert np.abs(h.balance_residual_J_per_m).max() <= 1e-12 * scale
    assert np.all(np.diff(h.dissipation_J_per_m) >= -1e-14 * scale)
    assert (
        np.abs(h.stress_Pa - ref.stress_Pa).max() > 1e-3 * np.abs(ref.stress_Pa).max()
    )
    assert h.kinematic_field is not None and h.kinematic_field.shape[0] == 2
    assert np.all(h.slip_fraction >= 0.0) and np.all(h.slip_fraction <= 1.0)
    stress = distributed_wire_friction_stress(
        spec, kx, ky, friction_law=law, time_s=_time(law, t)
    )
    np.testing.assert_array_equal(stress, h.stress_Pa)


@analytical
def test_distributed_breakaway_peak():
    """Planar monotonic bending: up to the first slip the static law is the
    model with ``m_s mu``; breakaway then lowers the bound below it."""
    spec = _spec(shape=False)
    kappa = np.linspace(0.0, 3.0 * full_slip_curvature(spec), 300)[1:]
    zero = np.zeros_like(kappa)
    static = distributed_friction_history(
        dataclasses.replace(spec, mu=1.4 * spec.mu), kappa, zero, max_increment=math.inf
    )
    h = distributed_friction_history(
        spec, kappa, zero, friction_law=BREAKAWAY, max_increment=math.inf
    )
    first = int(np.argmax(static.slip_fraction > 0.0))
    assert first > 1
    scale = np.abs(static.stress_Pa).max()
    np.testing.assert_allclose(
        h.stress_Pa[:, :first], static.stress_Pa[:, :first], rtol=0, atol=1e-10 * scale
    )
    assert h.stress_Pa[0, -1] < static.stress_Pa[0, -1] * (1 - 1e-3)
    assert np.abs(h.balance_residual_J_per_m).max() <= 1e-12 * h.dissipation_J_per_m[-1]


def test_distributed_refuses_rate_terms():
    spec = _spec()
    t, kx, ky = _short(spec, 20)
    with pytest.raises(CableInputError, match="static/dynamic option"):
        distributed_friction_history(spec, kx, ky, friction_law=RATE, time_s=t)
    with pytest.raises(CableInputError, match="give the sample times"):
        distributed_friction_history(spec, kx, ky, friction_law=STRENGTHENING)
    # No friction: zero stress whatever the law.
    free = dataclasses.replace(spec, mu=0.0)
    h = distributed_friction_history(free, kx, ky, friction_law=BREAKAWAY)
    np.testing.assert_array_equal(h.stress_Pa, 0.0)


# --------------------------------------------------------------------------- #
# Entry points                                                                 #
# --------------------------------------------------------------------------- #
def _menard():
    return cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)


def test_compute_mkappa_with_a_law():
    cable = _menard()
    time = np.linspace(0.0, 10.0, 2001)
    kappa = 4.0e-3 * np.sin(2 * np.pi * time / 5.0)
    ref = cx.compute_mkappa(cable, kappa)
    neutral = cx.compute_mkappa(cable, kappa, friction_law=FrictionLaw())
    np.testing.assert_array_equal(neutral.moment_Nm, ref.moment_Nm)
    assert neutral.W_diss_J_per_m == ref.W_diss_J_per_m
    law = FrictionLaw(static_factor=1.3, dynamic_factor=1.0, stick_time_s=0.5)
    r = cx.compute_mkappa(cable, kappa, friction_law=law, time_s=time)
    assert (
        np.abs(r.moment_Nm - ref.moment_Nm).max() > 1e-3 * np.abs(ref.moment_Nm).max()
    )
    assert r.work is not None
    assert np.abs(r.work.balance_residual_J_per_m).max() <= 1e-12 * r.W_diss_J_per_m
    assert r.provenance is not None and "friction law" in r.provenance.method
    rate = cx.compute_mkappa(cable, kappa, friction_law=RATE, time_s=time)
    assert rate.work is not None and rate.work.viscous_J_per_m is not None
    assert rate.W_diss_J_per_m == pytest.approx(
        rate.work.friction_J_per_m[-1] + rate.work.viscous_J_per_m[-1], rel=1e-15
    )
    assert rate.W_diss_J_per_m > ref.W_diss_J_per_m
    assert (
        np.abs(rate.work.balance_residual_J_per_m).max() <= 1e-12 * rate.W_diss_J_per_m
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"friction_law": RATE}, "give the sample times"),
        ({"time_s": "t"}, "used only"),
        ({"friction_law": BREAKAWAY, "time_s": "t"}, "used only"),
        ({"friction_law": BREAKAWAY, "core_friction": True}, "direction-tied"),
        (
            {"friction_law": BREAKAWAY, "friction_model": "progressive"},
            "direction-tied",
        ),
        ({"friction_law": "coulomb"}, "FrictionLaw"),
    ],
)
def test_compute_mkappa_refusals(kwargs, match):
    cable = _menard()
    time = np.linspace(0.0, 1.0, 50)
    kappa = 1.0e-3 * np.sin(time)
    kwargs = {k: time if v == "t" else v for k, v in kwargs.items()}
    with pytest.raises(CableSchemaError, match=match):
        cx.compute_mkappa(cable, kappa, **kwargs)


def test_mkappa_law_refuses_the_progressive_route():
    spec = _spec()
    kappa = np.linspace(0.0, 1e-3, 20)
    with pytest.raises(CableInputError, match="progressive"):
        compute_mkappa_wire_slip_front(
            [spec],
            kappa,
            progressive=cx.ProgressiveFriction(),
            friction_law=BREAKAWAY,
        )


def test_compute_fatigue_with_a_law():
    cable = _menard()
    time = np.linspace(0.0, 100.0, 400)
    kx = 0.02 * np.sin(2 * np.pi * time / 10.0)
    ky = np.zeros_like(kx)
    ref = cx.compute_fatigue(cable, kx, ky)
    neutral = cx.compute_fatigue(cable, kx, ky, friction_law=FrictionLaw())
    for name, damage in ref.D_total_per_wire.items():
        np.testing.assert_array_equal(neutral.D_total_per_wire[name], damage)
    got = cx.compute_fatigue(cable, kx, ky, friction_law=RATE, time_s=time)
    assert got.provenance is not None and "friction law" in got.provenance.method
    changed = any(
        not np.array_equal(got.D_total_per_wire[name], damage)
        for name, damage in ref.D_total_per_wire.items()
    )
    assert changed
    with pytest.raises(CableSchemaError, match="give the sample times"):
        cx.compute_fatigue(cable, kx, ky, friction_law=RATE)
    with pytest.raises(CableSchemaError, match="thermal history or a time-dependent"):
        cx.compute_fatigue(cable, kx, ky, friction_law=BREAKAWAY, time_s=time)
    with pytest.raises(CableSchemaError, match="FrictionLaw"):
        cx.compute_fatigue(cable, kx, ky, friction_law=1.2)  # type: ignore[arg-type]
    with pytest.raises(CableInputError, match="progressive"):
        cx.compute_fatigue(
            cable, kx, ky, friction_law=BREAKAWAY, friction_model="progressive"
        )
    with pytest.raises(CableInputError, match="static/dynamic option"):
        cx.compute_fatigue(
            cable, kx, ky, friction_law=RATE, time_s=time, friction_model="distributed"
        )


def test_compute_fatigue_section_states_with_a_law():
    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    time = np.linspace(0.0, 20.0, 12)
    kx = 0.04 * np.sin(2 * np.pi * time / 10.0)
    ky = np.zeros_like(kx)
    tension = np.where(np.arange(time.size) % 2 == 0, 1.0e5, 2.0e5)
    ref = cx.compute_fatigue(cable, kx, ky, tension_N=tension, contact_model="qp")
    got = cx.compute_fatigue(
        cable,
        kx,
        ky,
        tension_N=tension,
        contact_model="qp",
        friction_law=COMBINED,
        time_s=time,
    )
    neutral = cx.compute_fatigue(
        cable, kx, ky, tension_N=tension, contact_model="qp", friction_law=FrictionLaw()
    )
    for name, damage in ref.D_total_per_wire.items():
        np.testing.assert_array_equal(neutral.D_total_per_wire[name], damage)
        assert np.all(np.isfinite(got.D_total_per_wire[name]))


def test_fatigue_pipeline_refusals():
    cable = _menard()
    layers = cable_to_fatigue_layers(cable, contact_tension_N=0)
    time = np.linspace(0.0, 10.0, 50)
    kx = 0.02 * np.sin(time)
    ky = np.zeros_like(kx)
    with pytest.raises(CableInputError, match="termination"):
        wire_stress_field(
            layers[0],
            kx,
            ky,
            termination=TerminationSpec(distance_m=0.1),
            friction_law=BREAKAWAY,
        )
    with pytest.raises(CableInputError, match="used only"):
        wire_stress_field(layers[0], kx, ky, time_s=time)
    a = compute_fatigue(layers, kx, ky, friction_law=COMBINED, time_s=time)
    b = compute_fatigue(layers, kx, ky)
    assert set(a.D_total_per_wire) == set(b.D_total_per_wire)


def test_bending_law_export_refuses_a_law():
    cable = _menard()
    args = {"tensions_N": [0.0], "kappa_max_1_m": 0.05, "n_points": 20}
    with pytest.raises(CableInputError, match="Masing"):
        export_bending_law(cable, friction_law=BREAKAWAY, **args)
    with pytest.raises(CableInputError, match="Masing"):
        export_bending_law(cable, friction_law="coulomb", **args)
    with pytest.raises(CableInputError, match="time_s"):
        export_bending_law(cable, time_s=np.zeros(20), **args)
    neutral = export_bending_law(cable, friction_law=FrictionLaw(), **args)
    plain = export_bending_law(cable, **args)
    np.testing.assert_array_equal(neutral.moment_Nm, plain.moment_Nm)


# --------------------------------------------------------------------------- #
# Robustness of the return (steep rate terms, zero lay angle)                 #
# --------------------------------------------------------------------------- #
def _return_check(a, k, md, side, c0, x0, speed, rc, v0, inv_veta):
    """Slip of the scalar and NumPy returns and the bound at that slip."""
    dp = flm._slip_magnitude(
        a, k, md, side, c0, x0, 0.0, 0.0, 0.0, speed, rc, v0, inv_veta
    )
    v = speed * dp
    bound = flm._bound(
        md * (1.0 + rc * math.log1p(v / v0) + v * inv_veta),
        side,
        c0,
        x0,
        0.0,
        0.0,
        0.0,
    )
    vec = flm._slip_magnitude_array(
        np.array([a]),
        k,
        md,
        np.array([side]),
        np.array([c0]),
        np.array([x0]),
        np.zeros(1),
        np.zeros(1),
        speed,
        rc,
        v0,
        inv_veta,
    )[0]
    return dp, bound, vec


@analytical
def test_return_converges_with_a_steep_viscous_term():
    """A strong viscous term makes the bound rise steeply with the slip; the
    return still meets its equation, and the stress stays below the trial."""
    a, k = 1.0e8, 6.0e10
    dp, stress, vec = _return_check(a, k, 1.0, 1.0, 1.0e6, 0.1, 100.0, 0.0, 1e-3, 1e5)
    assert math.isfinite(dp) and 0.0 < dp < (a - 1.0e6 * math.expm1(0.1)) / k
    assert stress <= a * (1 + 1e-14)
    assert abs(a - k * dp - stress) <= 1e-14 * a
    assert vec == pytest.approx(dp, rel=1e-12)
    # The kernels' end-of-step stress: the bound, never above the trial.
    end = flm._return_stress(
        a, 1.0, 1.0, 1.0e6, 0.1, 0.0, 0.0, 0.0, 100.0, 0.0, 1e-3, 1e5, dp
    )
    assert end == pytest.approx(stress, rel=1e-15) and end <= a


@analytical
@pytest.mark.parametrize("side", [1.0, -1.0])
def test_return_sweep_of_extreme_rate_terms(side):
    k, md, v0 = 6.0e10, 0.9, 1.0e-4
    cases = []
    for inv_veta in (0.0, 1.0, 1.0e3, 1.0e6, 1.0e9):
        for rc in (0.0, 0.2):
            if inv_veta == 0.0 and rc == 0.0:
                continue
            for speed in (1.0, 1.0e2, 1.0e4, 1.0e6):
                for c0 in (1.0e5, 1.0e7):
                    for x0 in (0.01, 0.1, 0.3):
                        for ratio in (1.5, 1.0e3):
                            b0 = flm._bound(md, side, c0, x0, 0.0, 0.0, 0.0)
                            cases.append((ratio * b0, c0, x0, speed, rc, inv_veta))
    for a, c0, x0, speed, rc, inv_veta in cases:
        dp, stress, vec = _return_check(a, k, md, side, c0, x0, speed, rc, v0, inv_veta)
        b0 = flm._bound(md, side, c0, x0, 0.0, 0.0, 0.0)
        assert math.isfinite(dp) and 0.0 <= dp <= (a - b0) / k
        assert b0 <= stress <= a * (1 + 1e-14)
        assert abs(a - k * dp - stress) <= 1e-14 * a
        assert vec == pytest.approx(dp, rel=1e-12, abs=1e-300)


def test_kernels_with_a_steep_viscous_term_and_short_steps():
    spec = _spec()
    law = FrictionLaw(
        static_factor=1.2,
        rate_coefficient=0.5,
        rate_reference_m_per_s=1.0e-9,
        viscous_velocity_m_per_s=1.0e-9,
    )
    n = 400
    t = 1.0e-6 * np.arange(n)
    big = 3.0 * full_slip_curvature(spec)
    kx = big * np.sin(np.linspace(0.0, 4 * np.pi, n))
    ky = 0.3 * big * np.sin(np.linspace(0.0, 3 * np.pi, n))
    ref = wire_slip_history(spec, kx, ky, friction_law=law, time_s=t)
    fast = wire_friction_stress(spec, kx, ky, friction_law=law, time_s=t)
    # A spring-slider from rest cannot exceed the elastic stress of its path.
    path = np.cumsum(np.hypot(np.diff(kx, prepend=0.0), np.diff(ky, prepend=0.0)))
    envelope = spec.k_f * spec.r_helix_m * path
    assert np.all(np.abs(ref.stress_Pa) <= envelope * (1 + 1e-12))
    scale = np.abs(ref.stress_Pa).max()
    np.testing.assert_allclose(fast, ref.stress_Pa, rtol=0, atol=1e-12 * scale)
    assert ref.work.viscous_J_per_m is not None and ref.work.viscous_J_per_m[-1] > 0
    # The steep rate term almost locks the wires: scale by the input work.
    total = np.abs(ref.work.endpoint_work_J_per_m).max()
    assert np.abs(ref.work.balance_residual_J_per_m).max() <= 1e-12 * total
    s = slider_history(1e-3 * np.sin(1e3 * t), 1.0, 1e9, friction_law=law, time_s=t)
    assert np.all(
        np.abs(s.force_N)
        <= 1e9
        * np.cumsum(np.abs(np.diff(1e-3 * np.sin(1e3 * t), prepend=0.0)))
        * (1 + 1e-12)
    )
    assert abs(s.balance_J).max() <= 1e-12 * s.input_work_J.max()


def test_unresolved_returns_are_refused(monkeypatch):
    spec = _spec()
    t, kx, ky = _history(spec, n=200)
    monkeypatch.setattr(flm, "_MAX_NEWTON", 1)
    a = np.array([1.0e8])
    out = flm._slip_magnitude_array(
        a,
        6e10,
        1.0,
        np.ones(1),
        np.array([1e6]),
        np.array([0.1]),
        np.zeros(1),
        np.zeros(1),
        100.0,
        0.0,
        1e-3,
        1e5,
    )
    assert np.isnan(out).all()
    with pytest.raises(CableInputError, match="did not converge"):
        wire_slip_history(spec, kx, ky, friction_law=RATE, time_s=t)
    monkeypatch.undo()

    import cablexsec.mechanics.fast_slip as fsm

    def unresolved(*args):
        return np.full((spec.n_wires, kx.size), np.nan)

    monkeypatch.setattr(fsm, "_stress_kernel_law", unresolved)
    with pytest.raises(CableInputError, match="did not converge"):
        wire_friction_stress(spec, kx, ky, friction_law=RATE, time_s=t)

    def unresolved_slider(drive, *args):
        nan = np.full(drive.size, np.nan)
        return nan, nan, nan, nan, nan

    monkeypatch.setattr(flm, "_slider_kernel", unresolved_slider)
    with pytest.raises(CableInputError, match="did not converge"):
        slider_history(np.ones(3), 1.0, 1.0)
    with pytest.raises(CableInputError, match="positive stick stiffness"):
        wire_friction_stress(
            dataclasses.replace(spec, xi=0.0), kx, ky, friction_law=BREAKAWAY
        )


@pytest.mark.parametrize("law", LAWS)
def test_zero_lay_angle_has_no_capstan_friction(law):
    """A zero lay angle has zero bounds; a friction law keeps it that way."""
    spec = dataclasses.replace(_spec(), lay_angle_rad=0.0)
    t, kx, ky = _history(_spec(), n=300)
    plain = wire_slip_history(spec, kx, ky)
    np.testing.assert_array_equal(plain.stress_Pa, 0.0)
    ref = wire_slip_history(spec, kx, ky, friction_law=law, time_s=_time(law, t))
    np.testing.assert_array_equal(ref.stress_Pa, 0.0)
    np.testing.assert_array_equal(ref.work.friction_J_per_m, 0.0)
    fast = wire_friction_stress(spec, kx, ky, friction_law=law, time_s=_time(law, t))
    np.testing.assert_array_equal(fast, 0.0)
