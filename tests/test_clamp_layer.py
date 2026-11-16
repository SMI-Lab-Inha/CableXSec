# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Anchored-wire friction boundary layer against its closed forms.

Port of the verified reference of the curvature-gradient study: exact
projection, Kraincanic--Kebadze far field, stick regime, energy identity,
convergence and the clamp law A = 1 + 0.5427 sqrt(a).
"""

import math

import numpy as np
import pytest
from scipy.optimize import minimize

from cablexsec.exceptions import CableInputError
from cablexsec.mechanics import clamp_layer as cl

pytestmark = pytest.mark.analytical_sanity


def _mesh(n_pitch, spp):
    m = int(round(n_pitch * spp))
    h = 2 * math.pi / spp
    phi = (np.arange(m) + 0.5) * h
    return m, h, phi


def _uniform(phi, theta0, xi_g=None):
    shape = (
        np.ones_like(phi) if xi_g is None else np.clip(phi / (2 * math.pi) / xi_g, 0, 1)
    )
    return shape * np.cos(phi + theta0)


def test_projection_matches_reference_and_generic_qp():
    rng = np.random.default_rng(3)
    for _ in range(12):
        y = rng.normal(size=9) * 2.0
        h = 0.3
        n_c = cl.chain_projection(y, h)
        n_np = cl.chain_projection_reference(y, h)
        cons = [
            {"type": "ineq", "fun": (lambda z, i=i, h=h: h - (z[i + 1] - z[i]))}
            for i in range(8)
        ]
        cons += [
            {"type": "ineq", "fun": (lambda z, i=i, h=h: h + (z[i + 1] - z[i]))}
            for i in range(8)
        ]
        ref = minimize(
            lambda z, y=y: 0.5 * np.sum((z - y) ** 2),
            np.zeros(9),
            jac=lambda z, y=y: z - y,
            constraints=cons,
            method="SLSQP",
            options={"ftol": 1e-14, "maxiter": 500},
        )
        assert np.abs(n_np - ref.x).max() < 1e-6
        assert np.abs(n_c - n_np).max() < 1e-12
    y = rng.normal(size=3000) * 5
    out = cl.chain_projection(y, 0.05)
    assert np.abs(out - cl.chain_projection_reference(y, 0.05)).max() < 1e-10
    assert np.abs(np.diff(out)).max() <= 0.05 * (1 + 1e-12)
    assert abs(out.sum() - y.sum()) < 1e-8  # translation invariance: far anchor fixed


@pytest.mark.parametrize("a", [0.8, 1.3, 3.0])
def test_far_field_is_kraincanic_kebadze(a):
    m, h, phi = _mesh(14, 128)
    n, _, _ = cl.anchored_wire_response(
        _uniform(phi, 0.4),
        np.linspace(0, a, 61)[1:],
        h,
        record=np.r_[np.zeros(59, bool), True],
    )
    mid = (phi / (2 * math.pi) > 6) & (phi / (2 * math.pi) < 8)
    ref = cl.uniform_curvature_force(phi[mid] + 0.4, a)
    assert np.abs(n[0, mid] - ref).max() < 2 * h


def test_far_field_equals_pointwise_reference_in_cycles():
    m, h, phi = _mesh(12, 64)
    path = np.concatenate(
        [
            np.linspace(0, 6, 31)[1:],
            np.linspace(6, -6, 61)[1:],
            np.linspace(-6, 6, 61)[1:],
        ]
    )
    cap = np.arcsin(np.abs(np.cos(phi + 1.0)))
    n, ref, _ = cl.anchored_wire_response(_uniform(phi, 1.0), path, h, cap_hat=cap)
    mid = (phi / (2 * math.pi) > 5) & (phi / (2 * math.pi) < 7)
    assert np.abs(n[:, mid] - ref[:, mid]).max() < 2 * h


def test_stick_regime_has_no_clamp_effect():
    m, h, phi = _mesh(6, 64)
    for xi_g in (None, 0.3):
        e = _uniform(phi, 1.3, xi_g)
        n, _, d = cl.anchored_wire_response(e, np.linspace(0, 0.9, 11)[1:], h)
        assert np.abs(n[-1] - 0.9 * e).max() < 1e-12
        assert d < 1e-12


def test_energy_identity():
    m, h, phi = _mesh(6, 64)
    e = np.exp(-phi / (2 * math.pi)) * np.cos(phi + 1.0)
    path = np.concatenate(
        [
            np.linspace(0, 8, 101)[1:],
            np.linspace(8, -8, 201)[1:],
            np.linspace(-8, 8, 201)[1:],
        ]
    )
    rho = 1.0 + 0.3 * np.sin(np.arange(path.size) / 17.0)
    n, _, dissipation = cl.anchored_wire_response(e, path, h, rho=rho)
    previous = np.zeros(m)
    work = remainder = 0.0
    old = 0.0
    for i, lam in enumerate(path):
        work += h * np.dot(n[i], (lam - old) * e)
        remainder += 0.5 * h * np.dot(n[i] - previous, n[i] - previous)
        previous, old = n[i], lam
    stored = 0.5 * h * np.dot(n[-1], n[-1])
    assert dissipation > 0
    assert abs(work - dissipation - stored - remainder) < 1e-9 * dissipation


def test_mesh_and_step_convergence_of_clamp_peak():
    peaks = []
    for spp in (32, 64, 128):
        m, h, phi = _mesh(10, spp)
        n, _, _ = cl.anchored_wire_response(
            _uniform(phi, math.pi / 2),
            np.linspace(0, 20, 101)[1:],
            h,
            record=np.r_[np.zeros(99, bool), True],
        )
        peaks.append(np.abs(n[0, : 2 * spp]).max())
    assert abs(peaks[2] - peaks[1]) < abs(peaks[1] - peaks[0]) + 1e-3
    assert abs(peaks[2] - peaks[1]) / peaks[2] < 0.01
    m, h, phi = _mesh(10, 64)
    values = []
    for steps in (50, 400):
        n, _, _ = cl.anchored_wire_response(
            _uniform(phi, math.pi / 2), np.linspace(0, 20, steps + 1)[1:], h
        )
        values.append(np.abs(n[-1, :128]).max())
    assert abs(values[1] - values[0]) / values[1] < 2e-3


def test_clamp_law_at_large_curvature():
    """Exact clamp peak tends to pi/2 + sqrt(a (2 - 4/pi)) for a >> 1."""
    a = 100.0
    m, h, phi = _mesh(14, 64)
    best = 0.0
    for theta0 in np.linspace(0, 2 * math.pi, 16, endpoint=False):
        n, _, _ = cl.anchored_wire_response(
            _uniform(phi, theta0),
            np.linspace(0, a, 81)[1:],
            h,
            record=np.r_[np.zeros(79, bool), True],
        )
        best = max(best, np.abs(n[0, : 4 * 64]).max())
    predicted = math.pi / 2 + float(cl.clamp_mean_force(a))
    assert abs(best / predicted - 1) < 0.03
    assert float(cl.clamp_amplification(a)) == pytest.approx(
        1 + cl.CLAMP_LAW_COEFFICIENT * 10
    )


def test_cyclic_clamp_amplification_matches_reference_map():
    """Uniform curvature from the anchor, a = 20 (L_s/P = 3.18): stabilised cyclic
    A* = 3.35 in the reference regime map (64 segments/pitch, 50 steps per
    half cycle, 16 phases)."""
    a = 20.0
    m, h, phi = _mesh(10, 64)
    path = np.concatenate(
        [np.linspace(0, a, 51)[1:]]
        + [np.linspace(a, -a, 101)[1:], np.linspace(-a, a, 101)[1:]] * 3
    )
    best = 0.0
    for theta0 in np.linspace(0, 2 * math.pi, 16, endpoint=False):
        n, _, _ = cl.anchored_wire_response(_uniform(phi, theta0), path, h)
        down, up = n[-101], n[-1]
        best = max(best, 0.5 * np.abs(up - down)[: m // 2].max())
    assert best / (math.pi / 2) == pytest.approx(3.35, rel=0.03)


def test_closed_form_values():
    assert pytest.approx(0.5427, abs=1e-4) == cl.CLAMP_LAW_COEFFICIENT
    assert float(cl.clamp_mean_force(4.0)) == pytest.approx(
        2 * math.sqrt(2 - 4 / math.pi)
    )
    assert float(cl.clamp_amplification(1.0)) == 1.0  # below full slip
    assert float(cl.clamp_mean_force(10.0, 0.0)) == 0.0  # extreme-fibre wire
    pitch = 0.8
    a = 12.0
    slip = a * pitch / (2 * math.pi)
    assert float(cl.clamp_boundary_layer_length(a, pitch)) == pytest.approx(
        0.5 * math.sqrt(slip * pitch)
    )
    assert float(cl.clamp_setback_length(a, pitch)) == pytest.approx(
        0.4 * math.sqrt(slip * pitch)
    )
    assert float(cl.clamp_offset_potential(3.0, 0.0)) == 3.0
    assert float(cl.clamp_offset_potential(5.0, 2.0)) == pytest.approx(
        4 / math.pi * (math.pi / 2 - 1) + 3
    )
    assert cl.gradient_slip_onset_factor(1.0, 1.0) == pytest.approx(
        1 / math.sqrt(1 + 1 / (4 * math.pi**2))
    )
    assert np.allclose(
        cl.uniform_curvature_force(np.array([0.0, math.pi]), 0.5), [0.5, -0.5]
    )


def test_one_shot_projection_is_first_increment():
    m, h, phi = _mesh(5, 64)
    e = np.exp(-phi / (2 * math.pi * 0.8)) * np.cos(phi + 2.0)
    n, _, _ = cl.anchored_wire_response(e, np.array([7.0]), h)
    assert np.abs(n[0] - cl.chain_projection(7.0 * e, h)).max() < 1e-12


def test_numpy_path_matches_kernel():
    m, h, phi = _mesh(3, 32)
    e = np.exp(-phi / 4) * np.cos(phi + 0.3)
    path = np.concatenate([np.linspace(0, 6, 13)[1:], np.linspace(6, -6, 25)[1:]])
    rho = 1.0 + 0.2 * np.cos(np.arange(path.size))
    cap = np.arcsin(np.abs(np.cos(phi + 0.3)))
    record = (np.arange(path.size) % 3 == 0).astype(np.int64)
    index = np.array([0, 5, m - 1], dtype=np.int64)
    rows = int(record.sum())
    outs = [np.zeros((rows, 3)) for _ in range(4)]
    d1 = cl._anchored_kernel(e, cap, path, rho, h, record, index, outs[0], outs[1], 0.0)
    d2 = cl._anchored_numpy(e, cap, path, rho, h, record, index, outs[2], outs[3], 0.0)
    assert np.allclose(outs[0], outs[2], atol=1e-10)
    assert np.allclose(outs[1], outs[3], atol=1e-12)
    assert d1 == pytest.approx(d2, rel=1e-9)


def test_input_validation():
    with pytest.raises(CableInputError):
        cl.chain_projection(np.array([np.nan]), 0.1)
    with pytest.raises(CableInputError):
        cl.chain_projection(np.ones(3), -1.0)
    with pytest.raises(CableInputError):
        cl.anchored_wire_response(np.ones(1), np.ones(2), 0.1)
    with pytest.raises(CableInputError):
        cl.anchored_wire_response(np.ones(4), np.ones(2), 0.1, rho=np.ones(3))
    with pytest.raises(CableInputError):
        cl.anchored_wire_response(np.ones(4), np.ones(2), 0.1, index=np.array([9]))
    with pytest.raises(CableInputError):
        cl.anchored_wire_response(np.ones(4), np.ones(2), 0.0)
    with pytest.raises(CableInputError):
        cl.anchored_wire_response(np.ones(4), np.ones(2), 0.1, cap_hat=np.ones(3))
    with pytest.raises(CableInputError):
        cl.anchored_wire_response(np.ones(4), np.ones(2), 0.1, record=np.ones(3, bool))
    with pytest.raises(CableInputError):
        cl.anchored_wire_response(np.ones(4), np.array([np.inf]), 0.1)
    with pytest.raises(CableInputError):
        cl.gradient_slip_onset_factor(0.0, 1.0)
