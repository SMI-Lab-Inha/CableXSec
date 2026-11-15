# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Progressive (Iwan) friction law on the direction-tied cap (gap G8)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.cable_adapters import cable_to_mkappa_specs
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.mechanics import fast_slip
from cablexsec.mechanics.papailiou import (
    _compression_onset_factor,
    slip_onset_curvature,
    wire_slip_history,
    wire_slip_stress_cap,
)
from cablexsec.mechanics.progressive_friction import (
    ProgressiveFriction,
    progressive_caps,
    progressive_wire_friction_stress,
    progressive_wire_history,
)
from cablexsec.mechanics.strand_slip import (
    StrandSlipParameters,
    conductor_bending_moment,
    strand_slip_strain,
)


@pytest.fixture(scope="module")
def menard():
    return cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)


@pytest.fixture(scope="module")
def spec(menard):
    return cable_to_mkappa_specs(menard, 0.0, mu=0.2, contact_model="bounded")[0]


def _same(a, b):
    """Equal to rounding: two floating-point paths may differ in the last digit across platforms."""
    a, b = np.asarray(a), np.asarray(b)
    scale = float(np.abs(a).max()) if a.size else 0.0
    np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-12 * scale)


def _random_path(n: int = 1500, seed: int = 3, step: float = 0.002):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0, step, n)), np.cumsum(rng.normal(0, step, n))


def _rotate(x, y, angle):
    c, s = math.cos(angle), math.sin(angle)
    return x * c - y * s, x * s + y * c


# --------------------------------------------------------------------------- #
# Limits                                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "settings",
    [ProgressiveFriction(n_elements=1), ProgressiveFriction("uniform", 8, 0.0)],
)
def test_zero_spread_is_the_default_law(spec, settings):
    kx, ky = _random_path()
    pressure = spec.contact_pressure_Pa * (1.0 + 0.3 * np.sin(np.arange(kx.size) / 50))
    ref = wire_slip_history(spec, kx, ky, contact_pressure_Pa=pressure)
    new = progressive_wire_history(
        spec, kx, ky, progressive=settings, contact_pressure_Pa=pressure
    )
    _same(ref.stress_Pa, new.stress_Pa)
    _same(ref.moment_Nm, new.moment_Nm)
    assert np.array_equal(ref.slipping_count, new.slipping_count)
    if settings.n_elements == 1:
        _same(ref.work.friction_J_per_m, new.work.friction_J_per_m)
        _same(ref.work.stored_J_per_m, new.work.stored_J_per_m)
    else:  # identical sub-elements: work sums differ by rounding only
        for a, b in (
            (ref.work.friction_J_per_m, new.work.friction_J_per_m),
            (ref.work.stored_J_per_m, new.work.stored_J_per_m),
        ):
            assert np.allclose(a, b, rtol=1e-12, atol=1e-12 * np.abs(a).max())


def test_zero_spread_with_changing_stick_modulus(spec):
    kx, ky = _random_path(800)
    stick = spec.k_f * (1.0 + 0.2 * (np.arange(kx.size) > 400))
    ref = wire_slip_history(spec, kx, ky, stick_modulus_Pa=stick)
    new = progressive_wire_history(
        spec,
        kx,
        ky,
        progressive=ProgressiveFriction(n_elements=1),
        stick_modulus_Pa=stick,
    )
    scale = float(np.abs(ref.stress_Pa).max())
    assert np.abs(ref.stress_Pa - new.stress_Pa).max() <= 1e-12 * scale
    assert np.allclose(
        ref.material_work_J_per_m, new.material_work_J_per_m, rtol=1e-12, atol=0
    )


@pytest.mark.parametrize("angle", [0.0, 0.3, 1.2])
@pytest.mark.parametrize(
    "settings",
    [ProgressiveFriction(), ProgressiveFriction("uniform", 7, 0.6)],
)
def test_sub_caps_average_to_the_papailiou_cap(spec, angle, settings):
    caps = progressive_caps(spec, angle, settings)
    assert caps.shape == (spec.n_wires, settings.n_elements)
    assert np.all(caps >= 0)
    cap = wire_slip_stress_cap(spec, angle)
    assert np.abs(caps.mean(axis=1) - cap).max() <= 1e-13 * cap.max()


def test_full_slip_moment_equals_default(spec):
    kappa = np.r_[np.linspace(0, 0.5, 400), np.linspace(0.5, -0.5, 800)]
    zeros = np.zeros_like(kappa)
    ref = wire_slip_history(spec, kappa, zeros)
    new = progressive_wire_history(
        spec, kappa, zeros, progressive=ProgressiveFriction()
    )
    k_on = slip_onset_curvature(spec)
    # Far beyond the transition (and the reversal) every sub-element slips.
    loading = np.arange(kappa.size) < 400
    far = (loading & (kappa > 20 * k_on)) | (~loading & (0.5 - kappa > 40 * k_on))
    scale = float(np.abs(ref.moment_Nm).max())
    assert far.any()
    assert np.abs(ref.moment_Nm[far] - new.moment_Nm[far]).max() < 1e-9 * scale


# --------------------------------------------------------------------------- #
# Physics                                                                      #
# --------------------------------------------------------------------------- #
def _continuity_backbone(spec, kappa, s):
    """Closed-form friction-continuity stress of a wire with sin(psi) = s."""
    sin_a = abs(math.sin(spec.lay_angle_rad))
    width = 2 * math.pi * spec.r_helix_m / spec.n_wires
    sigma0 = (
        spec.r_helix_m
        * spec.contact_pressure_Pa
        * width
        * math.cos(spec.lay_angle_rad)
        / (spec.A_wire_m2 * sin_a**2)
    )
    lam = spec.mu * sin_a
    k, r = spec.k_f, spec.r_helix_m
    psi = math.asin(s)
    if k * r * kappa <= sigma0 * lam:
        return k * r * kappa * s
    from scipy.optimize import brentq

    def front(p):
        return k * r * kappa * math.cos(p) - sigma0 * lam * math.exp(lam * p)

    p_star = (
        brentq(front, 0.0, math.pi / 2 - 1e-15)
        if front(math.pi / 2 - 1e-15) < 0
        else math.pi / 2
    )
    if p_star >= psi:
        return sigma0 * math.expm1(lam * psi)
    return sigma0 * math.expm1(lam * p_star) + k * r * kappa * (s - math.sin(p_star))


def test_continuity_density_reproduces_the_closed_form_backbone(spec):
    k_on = slip_onset_curvature(spec)
    kappa = np.linspace(0.0, 4.0 * k_on, 801)
    zeros = np.zeros_like(kappa)
    hist = progressive_wire_history(
        spec, kappa, zeros, progressive=ProgressiveFriction(n_elements=512)
    )
    c = np.abs(np.cos(spec.phi_wires))
    cap = wire_slip_stress_cap(spec, 0.0).max()
    for w in (0, 3, 7, 11):
        expected = np.array([_continuity_backbone(spec, k, c[w]) for k in kappa])
        got = np.abs(hist.stress_Pa[w])
        assert np.abs(got - expected).max() < 5e-3 * cap


def test_transition_is_more_gradual_than_the_default(spec):
    k_on = slip_onset_curvature(spec)
    kappa = np.linspace(0.0, 3.0 * k_on, 301)
    zeros = np.zeros_like(kappa)
    ref = wire_slip_history(spec, kappa, zeros).moment_Nm[:, 0]
    new = progressive_wire_history(
        spec, kappa, zeros, progressive=ProgressiveFriction()
    ).moment_Nm[:, 0]
    # Same stick branch, never above the pointwise law, strictly below it in
    # the transition (friction continuity removes the KK overshoot). The stick
    # branch ends at the compression-side first slip m(lam) k_on, below k_on.
    lam = spec.mu * abs(math.sin(spec.lay_angle_rad))
    stick = kappa < _compression_onset_factor(lam) * k_on
    assert np.count_nonzero(stick) > 3
    assert np.allclose(ref[stick], new[stick], rtol=1e-12, atol=0)
    assert np.all(new <= ref + 1e-9 * np.abs(ref).max())
    assert (ref - new)[kappa > 1.2 * k_on].min() > 0


def _reversal(spec, multiple):
    k_on = slip_onset_curvature(spec)
    amp = multiple * k_on
    up = np.linspace(0.0, amp, 1201)
    down = np.linspace(amp, -amp, 1201)[1:]
    kappa = np.r_[up, down]
    m = progressive_wire_history(
        spec, kappa, np.zeros_like(kappa), progressive=ProgressiveFriction()
    ).moment_Nm[:, 0]
    return amp, up, down, m


def _sub_elements(spec, amp):
    """Bounds of every sub-element, ``(wires, elements)``: ``b1`` on the side
    of the first excursion (upper on the tension side, lower on the
    compression side), ``b2`` the opposite one, and the peak force."""
    settings = ProgressiveFriction()
    upper = progressive_caps(spec, 0.0, settings)
    lower = progressive_caps(spec, 0.0, settings, side=-1.0)
    c = np.cos(spec.phi_wires)[:, None]
    b1 = np.where(c >= 0, upper, lower)
    b2 = np.where(c >= 0, lower, upper)
    stick = spec.k_f * spec.r_helix_m * np.abs(c)
    peak = np.minimum(stick * amp, b1)
    return b1, b2, peak, stick


def _moment(spec, force):
    """Moment of the sub-element forces (signed by the wire side)."""
    c = np.cos(spec.phi_wires)
    weight = spec.A_wire_m2 * math.cos(spec.lay_angle_rad) * spec.r_helix_m
    return (
        weight
        * np.einsum("wjt,w->t", np.sign(c)[:, None, None] * force, c)
        / (force.shape[1])
    )


def test_reversal_is_the_elementwise_closed_form(spec):
    """Each sub-element is elastic--perfectly plastic with the interval
    ``[-lower, upper]``: from the peak ``f_p = min(k x_p, b1)`` of its first
    excursion it reverses as ``f_p - min(k d, f_p + b2)`` at the reversal
    distance ``d`` (written out here, no time stepping)."""
    amp, up, down, m = _reversal(spec, 3.0)
    b1, b2, peak, stick = _sub_elements(spec, amp)
    d = (amp - down)[None, None, :]
    force = peak[..., None] - np.minimum(stick[..., None] * d, (peak + b2)[..., None])
    expected = _moment(spec, force)
    assert np.abs(m[up.size :] - expected).max() < 1e-12 * np.abs(m).max()


def test_masing_reversal_in_its_generalised_form(spec):
    """Masing's rule for the signed interval: a sub-element that slipped at
    ``b1`` reverses over ``b1 + b2``, ``b1 - min(k d, b1 + b2) = b1 - 2
    min(k d / 2, (b1 + b2) / 2)``, and one that stuck with ``f_p <= b2``
    reverses elastically over the whole return; both give
    ``M_down(k) = M(amp) - 2 M_s((amp - k) / 2)`` with ``M_s`` the virgin
    branch of the population with the mean bounds ``(b1 + b2) / 2`` (the
    ``sinh`` shares). Only a sub-element stuck at ``b2 < f_p < b1`` breaks
    it; none does at ten times the onset. The plain doubling of the virgin
    branch, exact for the former symmetric cap, no longer holds."""
    amp, up, down, m = _reversal(spec, 10.0)
    b1, b2, peak, stick = _sub_elements(spec, amp)
    assert not np.any((peak < b1) & (peak > b2))
    half = ((amp - down) / 2)[None, None, :]
    mean_virgin = _moment(
        spec, np.minimum(stick[..., None] * half, (0.5 * (b1 + b2))[..., None])
    )
    expected = m[up.size - 1] - 2.0 * mean_virgin
    scale = np.abs(m).max()
    assert np.abs(m[up.size :] - expected).max() < 1e-12 * scale
    plain = m[up.size - 1] - 2.0 * np.interp((amp - down) / 2.0, up, m[: up.size])
    assert np.abs(m[up.size :] - plain).max() > 1e-3 * scale


def test_dissipation_nonnegative_and_balance(spec):
    kx, ky = _random_path(2000, seed=11)
    hist = progressive_wire_history(
        spec, kx, ky, progressive=ProgressiveFriction("uniform", 6, 0.8)
    )
    assert np.diff(hist.work.friction_J_per_m).min() >= 0.0
    scale = float(np.abs(hist.work.endpoint_work_J_per_m).max())
    assert np.abs(hist.work.balance_residual_J_per_m).max() < 1e-10 * max(scale, 1.0)


@pytest.mark.parametrize(
    "settings", [ProgressiveFriction(), ProgressiveFriction("uniform", 5, 1.0)]
)
def test_rotation_equivariance(spec, settings):
    kx, ky = _random_path(1200, seed=5)
    base = progressive_wire_history(spec, kx, ky, progressive=settings)
    angle = 3 * 2 * math.pi / spec.n_wires  # a symmetry of the wire population
    rx, ry = _rotate(kx, ky, angle)
    rot = progressive_wire_history(
        spec, rx, ry, progressive=settings, initial_bending_angle=angle
    )
    mx, my = _rotate(base.moment_Nm[:, 0], base.moment_Nm[:, 1], angle)
    scale = float(np.abs(base.moment_Nm).max())
    assert np.abs(rot.moment_Nm[:, 0] - mx).max() < 1e-12 * scale
    assert np.abs(rot.moment_Nm[:, 1] - my).max() < 1e-12 * scale
    assert np.array_equal(
        rot.work.friction_J_per_m.round(12), base.work.friction_J_per_m.round(12)
    ) or np.allclose(rot.work.friction_J_per_m, base.work.friction_J_per_m, rtol=1e-12)


def test_compiled_kernel_matches_reference(spec):
    kx, ky = _random_path(1500, seed=8)
    pressure = spec.contact_pressure_Pa * (1.0 + 0.4 * np.cos(np.arange(kx.size) / 70))
    for settings in (ProgressiveFriction(), ProgressiveFriction("uniform", 9, 0.4)):
        ref = progressive_wire_history(
            spec, kx, ky, progressive=settings, contact_pressure_Pa=pressure
        ).stress_Pa
        fast = progressive_wire_friction_stress(
            spec, kx, ky, progressive=settings, contact_pressure_Pa=pressure
        )
        assert np.abs(fast - ref).max() <= 1e-12 * np.abs(ref).max()
    one = progressive_wire_friction_stress(
        spec, kx, ky, progressive=ProgressiveFriction(n_elements=1)
    )
    _same(one, fast_slip.wire_friction_stress(spec, kx, ky))


def test_settings_validation():
    with pytest.raises(CableInputError):
        ProgressiveFriction("continuity", 8, 0.2)
    with pytest.raises(CableInputError):
        ProgressiveFriction("uniform", 8, 1.5)
    with pytest.raises(CableInputError):
        ProgressiveFriction("iwan", 8, 0.0)  # type: ignore[arg-type]
    with pytest.raises(CableInputError):
        ProgressiveFriction(n_elements=0)
    with pytest.raises(CableInputError):
        ProgressiveFriction(n_elements=True)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Public paths                                                                 #
# --------------------------------------------------------------------------- #
def test_compute_mkappa_progressive(menard):
    kappa = 0.1 * np.sin(np.linspace(0, 4 * np.pi, 400))
    base = cx.compute_mkappa(menard, kappa)
    one = cx.compute_mkappa(
        menard,
        kappa,
        friction_model="progressive",
        progressive=cx.ProgressiveFriction(n_elements=1),
    )
    _same(base.moment_Nm, one.moment_Nm)
    _same(base.work.friction_J_per_m, one.work.friction_J_per_m)
    prog = cx.compute_mkappa(menard, kappa, friction_model="progressive")
    assert prog.W_diss_J_per_m < base.W_diss_J_per_m
    assert "progressive" in prog.provenance.method
    with pytest.raises(CableSchemaError):
        cx.compute_mkappa(menard, kappa, progressive=cx.ProgressiveFriction())
    with pytest.raises(CableSchemaError):
        cx.compute_mkappa(menard, kappa, friction_model="distributed")  # type: ignore[arg-type]


def test_compute_fatigue_progressive(menard):
    t = np.linspace(0, 20 * np.pi, 600)
    kx, ky = 0.05 * np.sin(t), 0.03 * np.cos(1.3 * t)
    base = cx.compute_fatigue(menard, kx, ky, n_fibres=8)
    one = cx.compute_fatigue(
        menard,
        kx,
        ky,
        n_fibres=8,
        friction_model="progressive",
        progressive=cx.ProgressiveFriction(n_elements=1),
    )
    for name in base.D_bulk_per_wire:
        _same(base.D_bulk_per_wire[name], one.D_bulk_per_wire[name])
    prog = cx.compute_fatigue(menard, kx, ky, n_fibres=8, friction_model="progressive")
    assert "progressive" in prog.provenance.method
    with pytest.raises(CableInputError):
        cx.compute_fatigue(
            menard, kx, ky, n_fibres=8, progressive=cx.ProgressiveFriction()
        )


# --------------------------------------------------------------------------- #
# Conductor operator: Papailiou cap shape                                      #
# --------------------------------------------------------------------------- #
def test_strand_cap_shape_spreads_onset_by_pi_over_two():
    params = StrandSlipParameters(1e-9, (0.01, 0.0), (1e-4,), 1.0, (1e-4,))
    kappa = np.linspace(0.0, 0.03, 3001)
    theta = np.linspace(0.0, math.pi / 2, 46)[:-1]  # extreme fibre towards the axis
    onset = {}
    for shape in ("karlsen_husoy", "papailiou"):
        res = strand_slip_strain(
            params, kappa, np.zeros_like(kappa), theta_rad=theta, cap_shape=shape
        )
        onset[shape] = kappa[np.argmax(res.slipping[0], axis=1)]
    k_c = 1e-4 / 0.01
    assert np.allclose(onset["karlsen_husoy"], k_c, atol=2e-5)
    assert abs(onset["papailiou"][0] - k_c) < 2e-5  # extreme fibre unchanged
    assert onset["papailiou"].min() < 0.66 * k_c  # near the axis: (2/pi) k_c
    assert np.all(np.diff(onset["papailiou"]) <= 1e-12)  # spreads from the axis


def test_strand_cap_shape_rotation_and_validation():
    params = StrandSlipParameters(1e-9, (0.01, 0.002), (1e-4,), 1.0, (1e-4,))
    kx, ky = _random_path(600, seed=2, step=0.003)
    theta = 2 * math.pi * np.arange(12) / 12
    angle = 0.4
    base = strand_slip_strain(params, kx, ky, theta_rad=theta, cap_shape="papailiou")
    rx, ry = _rotate(kx, ky, angle)
    rot = strand_slip_strain(
        params, rx, ry, theta_rad=theta + angle, cap_shape="papailiou"
    )
    assert np.abs(rot.strain - base.strain).max() < 1e-12 * np.abs(base.strain).max()
    with pytest.raises(CableInputError):
        strand_slip_strain(params, kx, ky, cap_shape="iwan")  # type: ignore[arg-type]


def test_hu_loop_area_with_the_papailiou_cap():
    from check_conductor_strand_slip import HU_E, HU_KAPPA

    g8 = pytest.importorskip("check_g8_gradual_reversal")
    CYCLE, _hu_metrics, _hu_setup = g8.CYCLE, g8._hu_metrics, g8._hu_setup

    rings, params, _ = _hu_setup()
    kappa = HU_KAPPA * np.sin(CYCLE)
    areas = {}
    for shape in ("karlsen_husoy", "papailiou"):
        moment, _ = conductor_bending_moment(
            rings,
            HU_E,
            params,
            kappa,
            np.zeros_like(kappa),
            np.full_like(kappa, 100.0),
            cap_shape=shape,
        )
        areas[shape] = _hu_metrics(kappa, moment)["area"]  # elastic terms add no area
    assert areas["karlsen_husoy"] == pytest.approx(4.084, rel=5e-3)
    assert areas["papailiou"] == pytest.approx(3.340, rel=5e-3)
