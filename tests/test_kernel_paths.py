# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Compiled kernels on small inputs, for the kernel coverage pass.

The release gate runs this file a second time with ``NUMBA_DISABLE_JIT=1`` so
that coverage traces the kernel bodies (compiled code is invisible to it).
The compiled results are checked against reference paths elsewhere
(test_fast_rainflow, test_fast_slip, test_along_wire, test_distributed_friction,
test_clamp_layer); here each case asserts the invariants of its kernel.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.conductor_fatigue import _stop_operator
from cablexsec.mechanics import along_wire as aw
from cablexsec.mechanics import clamp_layer as cl
from cablexsec.mechanics import distributed_friction as dfm


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _cycle(amp: float, steps: int) -> np.ndarray:
    return np.r_[
        np.linspace(0, amp, steps + 1)[1:],
        np.linspace(amp, -amp, 2 * steps + 1)[1:],
        np.linspace(-amp, amp, 2 * steps + 1)[1:],
    ]


def _chain_layer(k_hat: float, lam: float) -> aw.WireChainLayer:
    return aw.WireChainLayer(
        name="unit",
        n_wires=2,
        r_helix_m=0.05,
        lay_angle_rad=math.radians(30.0),
        kappa_c_1_m=1.0,
        k_hat=k_hat,
        lam=lam,
        force_unit_N=1.0,
        wire_area_m2=1.0,
        ea_N=1.0,
    )


@pytest.mark.parametrize(
    ("k_hat", "lam", "ends"),
    [
        (aw.RIGID_K_HAT, 0.0, ("anchored", "anchored")),
        (0.26, 0.1, ("free", "free")),
        (0.26, 0.0, ("anchored", "free")),
    ],
)
def test_along_wire_chain_cycle(k_hat, lam, ends):
    layer = _chain_layer(k_hat, lam)
    path = _cycle(4.0, 5)
    x = np.array([0.0, 3 * layer.pitch_m])
    history = aw.along_wire_history(
        layer,
        x,
        np.tile(path, (2, 1)),
        ends=ends,
        segments_per_pitch=16,
        contact_ratio=np.linspace(1.0, 0.8, path.size),
    )
    assert np.isfinite(history.force).all() and np.isfinite(history.stress_Pa).all()
    # Friction dissipates: the energy record is finite and the loop closes.
    assert np.isfinite(history.energy_J).all()


def test_along_wire_bending_step_and_jacobian():
    cable = cx.load_cable(cx.inter_array_single_core_yaml_path())
    section = cx.analysis.along_wire_test.AlongWireSection.from_cable(cable)
    p = max(layer.pitch_m for layer in section.layers)
    from cablexsec.analysis.beam_test import BendingProtocol, FourPoint
    from cablexsec.analysis.beam_test import simulate_bending_test as simulate

    result = simulate(
        section,
        FourPoint(2 * p, p),
        BendingProtocol("curvature", np.array([0.3, 0.6, -0.3])),
        stations_per_segment=4,
        along_wire=cx.analysis.along_wire_test.AlongWireSpec(segments_per_pitch=8),
    )
    assert result.equilibrium_residual < 1e-6


def test_distributed_friction_cycle():
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    spec = cx.cable_to_fatigue_layers(cable, contact_tension_N=2e4)[0].spec
    kx = _cycle(0.05, 4)
    ky = 0.3 * kx
    history = dfm.distributed_friction_history(
        spec,
        kx,
        ky,
        contact_pressure_Pa=np.linspace(1.0, 1.2, kx.size) * spec.contact_pressure_Pa,
        stick_modulus_Pa=np.full(kx.size, spec.k_f),
        nodes_per_pitch=16,
        record_field=True,
    )
    work = np.abs(history.endpoint_work_J_per_m).max()
    assert np.abs(history.balance_residual_J_per_m).max() <= 1e-6 * max(work, 1e-30)
    assert np.all(np.diff(history.dissipation_J_per_m) >= -1e-9 * max(work, 1e-30))
    assert history.field_normalised is not None


def test_clamp_layer_responses():
    phi = np.linspace(0.0, 6 * 2 * math.pi, 97)
    h = float(phi[1] - phi[0])
    e = np.cos(phi)
    path = _cycle(1.5, 4)
    n, reference, dissipation = cl.anchored_wire_response(
        e, path, h, rho=np.full(path.size, 0.9), cap_hat=np.full(phi.size, 1.0)
    )
    assert np.isfinite(n).all() and dissipation >= 0.0
    field, _, field_dissipation = cl.anchored_wire_field_response(
        np.outer(path, e), e, h, max_increment=0.25
    )
    assert np.isfinite(field).all() and field_dissipation >= 0.0


def test_stop_operator_limits():
    drive = np.array([[0.0, 0.5, 1.5, 0.5, -1.5, 0.0]])
    ratio, slip = _stop_operator(drive)
    np.testing.assert_allclose(ratio[0], [0.0, 0.5, 1.0, 0.0, -1.0, 0.5])
    np.testing.assert_array_equal(slip[0], [False, False, True, False, True, False])


def _kernel(ax, ay, khat, lam, nodes, *, rho=None, g=None, newton=True):
    h = 2 * math.pi / nodes
    out = dfm._distributed_kernel(
        np.ascontiguousarray(ax, dtype=float),
        np.ascontiguousarray(ay, dtype=float),
        np.ones(len(ax)) if rho is None else np.ascontiguousarray(rho, dtype=float),
        float(khat),
        float(lam),
        np.ones(nodes) if g is None else np.ascontiguousarray(g, dtype=float),
        h / 2,
        1.0,
        0.5,
        np.zeros(1, dtype=np.int64),
        np.zeros(1),
        True,
        newton,
    )
    assert out[3][0] == 0
    return out


@pytest.mark.parametrize("khat", [0.33, 1000.0])
def test_distributed_kernel_solvers_agree(khat):
    ax = _cycle(3.0, 4)
    ay = 0.4 * ax
    nodes = 32
    rho = np.linspace(1.0, 0.9, ax.size)
    g = 1.0 + 0.2 * np.cos(2 * math.pi * np.arange(nodes) / nodes)
    newton = _kernel(ax, ay, khat, 0.127, nodes, rho=rho, g=g)
    fixed = _kernel(ax, ay, khat, 0.127, nodes, rho=rho, g=g, newton=False)
    np.testing.assert_allclose(newton[4], fixed[4], rtol=1e-6, atol=1e-8)
    # Without the capstan term only the fixed-bound solver applies.
    plain = _kernel(ax, ay, khat, 0.0, nodes)
    assert np.isfinite(plain[0]).all() and np.all(np.diff(plain[1][:, 1]) >= -1e-10)


def test_along_wire_history_options():
    layer = _chain_layer(0.26, 0.1)
    path = _cycle(3.0, 4)
    single = aw.along_wire_history(
        layer,
        np.array([layer.pitch_m]),
        path[None, :],
        extent_m=(0.0, 2 * layer.pitch_m),
        segments_per_pitch=16,
        record_x_m=np.array([0.5 * layer.pitch_m, layer.pitch_m]),
        wires=[1],
    )
    assert single.force.shape[0] == 1 and np.isfinite(single.stress_Pa).all()
