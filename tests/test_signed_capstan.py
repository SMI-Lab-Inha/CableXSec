# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Signed capstan bound of the armour friction stress ([Papailiou1997] Eqs. 9-15).

The capstan growth acts on the wire's own force: a wire whose force rises
presses harder on its support, a wire whose force falls loses normal load.
With ``lam = mu |sin a|`` and ``psi`` the distance from the neutral axis, the
admissible friction stress is ``-sigma0 (1 - exp(-lam psi)) <= sigma <=
sigma0 expm1(lam psi)``, and a full reversal spans ``2 sigma0 sinh(lam psi)``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cablexsec.mechanics.distributed_friction import (
    HAVE_NUMBA,
    distributed_friction_history,
)
from cablexsec.mechanics.fast_slip import wire_friction_stress
from cablexsec.mechanics.mkappa import HelicalLayerSpec
from cablexsec.mechanics.papailiou import (
    _compression_onset_factor,
    direction_threshold_curvature,
    slip_onset_curvature,
    wire_slip_history,
    wire_slip_stress_bounds,
)

pytestmark = pytest.mark.analytical_sanity
compiled = pytest.mark.skipif(
    not HAVE_NUMBA, reason="long histories need the compiled kernel"
)


def _spec(alpha_deg: float = 35.0, mu: float = 0.3, n: int = 72) -> HelicalLayerSpec:
    return HelicalLayerSpec(
        n_wires=n,
        r_helix_m=0.05,
        lay_angle_rad=math.radians(alpha_deg),
        E_wire_Pa=207e9,
        A_wire_m2=math.pi * 0.005**2 / 4,
        xi=0.6,
        mu=mu,
        F_n_axi_N=0.0,
        contact_pressure_Pa=1.0e6,
        normal_load_is_line=True,
    )


def _sigma0(spec: HelicalLayerSpec) -> float:
    a = spec.lay_angle_rad
    width = 2 * math.pi * spec.r_helix_m / spec.n_wires
    return (
        spec.r_helix_m
        * spec.contact_pressure_Pa
        * width
        * math.cos(a)
        / (spec.A_wire_m2 * math.sin(a) ** 2)
    )


def _reversal(spec: HelicalLayerSpec, multiple: float = 50.0, steps: int = 400):
    amp = multiple * slip_onset_curvature(spec)
    up = np.linspace(0.0, amp, steps + 1)[1:]
    down = np.linspace(amp, -amp, 2 * steps + 1)[1:]
    kx = np.concatenate([up, down])
    return kx, np.zeros_like(kx), steps - 1


def test_bounds_are_papailiou_signed_closed_form():
    spec = _spec()
    lam = spec.mu * math.sin(spec.lay_angle_rad)
    psi = np.arcsin(np.abs(np.cos(spec.phi_wires)))
    upper, lower = wire_slip_stress_bounds(spec, 0.0)
    s0 = _sigma0(spec)
    assert upper == pytest.approx(s0 * np.expm1(lam * psi), rel=1e-12, abs=1e-6)
    assert lower == pytest.approx(s0 * -np.expm1(-lam * psi), rel=1e-12, abs=1e-6)
    assert np.all(lower <= upper)


def test_full_slip_reversal_range_is_two_sinh():
    spec = _spec()
    lam = spec.mu * math.sin(spec.lay_angle_rad)
    s0 = _sigma0(spec)
    kx, ky, peak = _reversal(spec)
    stress = wire_slip_history(spec, kx, ky).stress_Pa
    psi = np.arcsin(np.abs(np.cos(spec.phi_wires)))
    tension = np.cos(spec.phi_wires) > 1e-9
    compression = np.cos(spec.phi_wires) < -1e-9
    # At the first peak: tension side at the tension bound, compression side
    # at the (smaller) compression bound.
    assert stress[tension, peak] == pytest.approx(
        s0 * np.expm1(lam * psi[tension]), rel=1e-9
    )
    assert stress[compression, peak] == pytest.approx(
        s0 * np.expm1(-lam * psi[compression]), rel=1e-9
    )
    loaded = tension | compression
    span = stress[loaded, peak] - stress[loaded, -1]
    expected = (
        2 * s0 * np.sinh(lam * psi[loaded]) * np.sign(np.cos(spec.phi_wires[loaded]))
    )
    assert span == pytest.approx(expected, rel=1e-9)


def test_compiled_kernel_is_the_reference():
    spec = _spec(n=61)
    t = np.linspace(0, 6, 900)
    kon = slip_onset_curvature(spec)
    kx = 12 * kon * np.sin(1.3 * t) * np.cos(0.2 * t)
    ky = 7 * kon * np.sin(0.7 * t)
    reference = wire_slip_history(spec, kx, ky).stress_Pa
    # Round-off only: the reference's vectorised NumPy arithmetic and the
    # scalar kernel differ by a few ulp on some builds (2e-15 of the peak).
    fast = wire_friction_stress(spec, kx, ky)
    assert np.abs(fast - reference).max() <= 1e-12 * np.abs(reference).max()


def test_small_friction_limit_is_linear_coulomb():
    spec = _spec(mu=1e-6)
    upper, lower = wire_slip_stress_bounds(spec, 0.3)
    assert lower == pytest.approx(upper, rel=1e-5, abs=1e-12)


@compiled
def test_distributed_full_slip_field_closed_form():
    # Rigid-stick-independent full-slip field of the distributed model with
    # the signed bound: dn/dphi = +-(g + lam n) from the extreme fibres and a
    # zero pitch mean give sigma = sigma0 (C exp(lam phi) - 1),
    # C = lam pi / (2 sinh(lam pi / 2)), phi from the neutral axis.
    spec = _spec(alpha_deg=45.0, mu=0.3, n=60)
    lam = spec.mu * math.sin(spec.lay_angle_rad)
    kon = slip_onset_curvature(spec)
    kx = np.linspace(0.0, 300 * kon, 3001)[1:]
    d = distributed_friction_history(spec, kx, np.zeros_like(kx), nodes_per_pitch=600)
    phi = np.pi / 2 - np.abs(np.angle(np.exp(1j * spec.phi_wires)))
    c = lam * math.pi / (2 * math.sinh(lam * math.pi / 2))
    expected = _sigma0(spec) * (c * np.exp(lam * phi) - 1)
    assert np.abs(d.stress_Pa[:, -1] - expected).max() <= 1e-5 * np.abs(expected).max()
    assert abs(d.stress_Pa[:, -1].mean()) <= 1e-4 * _sigma0(spec)


@pytest.mark.parametrize("lam", [0.02, 0.1, 0.2])
def test_compression_onset_factor(lam):
    m = _compression_onset_factor(lam)
    assert m == pytest.approx(1 - 3 * lam**2 / 8, abs=2 * lam**3)
    psi = np.linspace(1e-4, math.pi / 2, 20001)
    assert m <= np.min(-np.expm1(-lam * psi) / (lam * np.sin(psi))) + 1e-12


def test_no_wire_slips_below_the_hold_threshold():
    spec = _spec(n=720)
    kth = direction_threshold_curvature(spec, 1.0)
    kappa = np.linspace(0.0, kth * (1 - 1e-9), 400)
    count = wire_slip_history(spec, kappa, np.zeros_like(kappa)).slipping_count
    assert count.max() == 0
