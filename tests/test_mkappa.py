# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for the M-κ engine (R1.2): gate A11 (energy closure) + smooth knee."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cablexsec.mechanics.mkappa import (
    HelicalLayerSpec,
    compute_mkappa_loop,
    energy_closure,
)

# Test-taxonomy category.
pytestmark = pytest.mark.analytical_sanity


def _layer(n_wires: int = 90, shape_g: np.ndarray | None = None) -> HelicalLayerSpec:
    return HelicalLayerSpec(
        n_wires=n_wires,
        r_helix_m=0.082,
        lay_angle_rad=math.radians(16.0),
        E_wire_Pa=210.0e9,
        A_wire_m2=math.pi * 0.005**2 / 4.0,
        xi=0.75,
        mu=0.20,
        F_n_axi_N=1700.0,
        shape_g=shape_g,
    )


def _steady_cycle_kappa(kappa_a: float, n_per_cycle: int = 2000, cycles: int = 3):
    """kappa(t) over several cycles; return (history, last-cycle slice)."""
    t = np.arange(n_per_cycle * cycles)
    kappa = kappa_a * np.sin(2.0 * np.pi * t / n_per_cycle)
    last = slice(n_per_cycle * (cycles - 1), n_per_cycle * cycles + 1)
    return kappa, last


# --------------------------------------------------------------------------- #
# A11 — energy closure (decisive analytical gate)                             #
# --------------------------------------------------------------------------- #
def test_A11_energy_closure_uniform_g() -> None:
    """A11: integrated steady-loop area == closed-form sum over slipped wires.

    Run with UNIFORM g first (per PI): proves the projection factor 1/|cos|
    alone gives the correct dissipation, with no F_n spread. Target < 0.1%.
    """
    layer = _layer(n_wires=180)  # dense population for a clean integral
    kappa_a = 0.05
    kappa, last = _steady_cycle_kappa(kappa_a)
    res = compute_mkappa_loop([layer], kappa)
    # Steady-state loop area over the last full cycle.
    m, k = res.moment_Nm[last], res.kappa[last]
    w_num = abs(float(np.trapezoid(m, k)))
    w_closed = energy_closure([layer], kappa_a)
    assert w_closed > 0.0
    assert abs(w_num - w_closed) / w_closed < 1e-3


def test_A11_energy_closure_with_three_core_shape() -> None:
    """A11 also holds with a non-uniform three-core shape g(phi)."""
    n = 180
    phi = 2.0 * np.pi * np.arange(n) / n
    g = 1.0 + 0.15 * np.cos(3 * phi)  # mean 1
    g = g / g.mean()
    layer = _layer(n_wires=n, shape_g=g)
    kappa_a = 0.05
    kappa, last = _steady_cycle_kappa(kappa_a)
    res = compute_mkappa_loop([layer], kappa)
    w_num = abs(float(np.trapezoid(res.moment_Nm[last], res.kappa[last])))
    w_closed = energy_closure([layer], kappa_a)
    assert abs(w_num - w_closed) / w_closed < 1e-3


# --------------------------------------------------------------------------- #
# Correction B — smooth knee from the projection factor alone                 #
# --------------------------------------------------------------------------- #
def test_smooth_knee_with_uniform_g() -> None:
    """The M-κ knee is smooth even with UNIFORM g: slip spreads over kappa."""
    layer = _layer(n_wires=360)  # uniform g
    kappa = np.linspace(0.0, 0.15, 1500)  # monotonic loading
    res = compute_mkappa_loop([layer], kappa)
    sf = res.slip_fraction
    # Slip fraction rises gradually 0 -> ~1, not as a single step:
    assert sf[0] == 0.0
    assert sf[-1] > 0.95
    # the curvature window over which 5%..95% of wires slip is wide (smooth)
    k5 = res.kappa[np.argmax(sf > 0.05)]
    k95 = res.kappa[np.argmax(sf > 0.95)]
    assert k95 > 3.0 * k5  # transition spans a wide kappa band -> smooth knee


def test_backbone_softens_monotonically_from_EImax_to_EImin() -> None:
    layer = _layer(n_wires=360)
    EI_min = 1.0e3
    kappa = np.linspace(1e-6, 0.15, 1500)
    res = compute_mkappa_loop([layer], kappa, EI_min_Nm2=EI_min)
    dM = np.gradient(res.moment_Nm, res.kappa)
    # tangent stiffness starts near EI_max and decays toward EI_min
    assert dM[1] == pytest.approx(res.EI_max_Nm2, rel=0.05)
    assert dM[-1] < 0.2 * res.EI_max_Nm2
    assert dM[-1] >= EI_min * 0.5


def test_A8b_stuck_EImax_equals_witztan_cos3() -> None:
    """A8b: stuck-limit EI per helical layer = 1/2 n E A r^2 cos^3(a), <1e-10."""
    layer = _layer(n_wires=90)
    witztan = (
        0.5
        * layer.n_wires
        * layer.E_wire_Pa
        * layer.A_wire_m2
        * layer.r_helix_m**2
        * math.cos(layer.lay_angle_rad) ** 3
    )
    # with xi = 1 the friction stick EI must equal the Witz-Tan plane-section bound
    import dataclasses

    layer1 = dataclasses.replace(layer, xi=1.0)
    assert layer1.EI_stick == pytest.approx(witztan, rel=1e-10)
    # xi scales it linearly
    assert layer.EI_stick == pytest.approx(layer.xi * witztan, rel=1e-10)


def test_uniform_g_default_is_unit_shape() -> None:
    layer = _layer(n_wires=10, shape_g=None)
    assert np.allclose(layer.f_y, layer.mu * layer.F_n_axi_N / layer.A_wire_m2)


def test_no_slip_below_threshold_gives_zero_dissipation() -> None:
    """Tiny kappa amplitude: no wire slips, W_diss ~ 0."""
    layer = _layer(n_wires=90)
    kappa_a = 1e-6  # well below kappa_c
    kappa, _ = _steady_cycle_kappa(kappa_a, n_per_cycle=400, cycles=2)
    res = compute_mkappa_loop([layer], kappa)
    # Dissipation is negligible vs the elastic energy (closed elastic loop).
    assert res.W_diss_J_per_m < 1e-3 * res.EI_max_Nm2 * kappa_a**2
    assert energy_closure([layer], kappa_a) == 0.0  # no wire slips
