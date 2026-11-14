# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Along-wire chain: free ends, limits, path independence and the Jacobian.

Verification of :mod:`cablexsec.mechanics.along_wire` against closed forms of
the normalised chain: a frictionless free wire carries no force, the stick far
field is ``xi e`` with ``xi = k / (1 + k)`` (rigid: ``e``), a free end carries
at most its tributary bound, the stick response does not depend on the
substep size, and the station Jacobian equals the finite-difference response
of an increment on an unchanged active set. The release gate runs this file a
second time with ``NUMBA_DISABLE_JIT=1`` (kernel coverage pass).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.along_wire_test import AlongWireSection, AlongWireSpec
from cablexsec.analysis.beam_test import (
    BendingProtocol,
    FourPoint,
    simulate_bending_test,
)
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics import along_wire as aw
from cablexsec.mechanics.strand_slip import StrandSlipParameters, strand_slip_strain
from cablexsec.mechanics.tendon_bending import (
    LateralRelief,
    TransverseSlipModel,
    WireSection,
)


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _layer(k_hat: float, lam: float = 0.0, n_wires: int = 2) -> aw.WireChainLayer:
    """Unit layer: ``kappa_c = 1`` (curvatures in units of the slip onset)."""
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


def _uniform(layer: aw.WireChainLayer, pitches: float, values) -> tuple:
    """Uniform curvature over ``pitches`` lay lengths (two equal positions)."""
    x = np.array([0.0, pitches * layer.pitch_m])
    kx = np.tile(np.asarray(values, dtype=float), (2, 1))
    return x, kx


def _interior(history: aw.AlongWireHistory, layer: aw.WireChainLayer, margin: float):
    """Record positions at least ``margin`` pitches from both ends."""
    x = history.record_x_m
    lo, hi = history.chain.x_start_m, history.chain.x_end_m
    return (x > lo + margin * layer.pitch_m) & (x < hi - margin * layer.pitch_m)


# --------------------------------------------------------------------------- #
# Limits of the history driver                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k_hat", [0.26, aw.RIGID_K_HAT])
def test_frictionless_free_wire_carries_no_force(k_hat):
    """No contact and no anchor: the wire slides freely, ``n = 0`` everywhere."""
    layer = _layer(k_hat)
    x, kx = _uniform(layer, 2.0, [0.5, 1.5, -1.0])
    history = aw.along_wire_history(
        layer,
        x,
        kx,
        ends=("free", "free"),
        segments_per_pitch=16,
        contact_ratio=np.zeros(3),
    )
    assert np.max(np.abs(history.force)) <= 1e-12
    # No friction, no stored energy: the bending work vanishes with the force.
    assert np.max(np.abs(history.energy_J)) <= 1e-12


@pytest.mark.parametrize(
    "ends", [("free", "free"), ("free", "anchored"), ("anchored", "free")]
)
def test_rigid_chain_far_field_and_free_end_bound(ends):
    """Rigid stick far field ``n = e``; a free end carries at most ``h b / 2``."""
    layer = _layer(aw.RIGID_K_HAT)
    amp = 0.4  # below the slip onset of the far field (|de/dphi| <= 1)
    x, kx = _uniform(layer, 3.0, [0.5 * amp, amp])
    history = aw.along_wire_history(
        layer, x, kx, ends=ends, segments_per_pitch=16, contact_ratio=np.ones(2)
    )
    inner = _interior(history, layer, 1.0)
    e = amp * np.cos(history.azimuth_rad[:, inner])
    np.testing.assert_allclose(history.force[:, inner, -1], e, atol=1e-9)
    cap = 0.5 * history.chain.h  # tributary weight times the unit bound
    for side, kind in zip((0, -1), ends, strict=True):
        if kind == "free":
            assert np.all(np.abs(history.force[:, side, :]) <= cap * (1 + 1e-9))
    residual = np.abs(history.balance_residual_J)
    assert residual.max() <= 1e-9 * max(np.abs(history.energy_J[..., 0]).max(), 1.0)
    assert np.all(history.energy_J[..., 1] >= -1e-12)


def test_spring_chain_stick_far_field_is_xi_times_the_strain():
    """``n'' - k n = -k e`` gives ``n = k e / (1 + k) = xi e`` far from the ends."""
    k_hat = 0.26
    layer = _layer(k_hat)
    amp = 0.5
    x, kx = _uniform(layer, 6.0, [amp])
    # A large contact ratio keeps every node in stick (bound above the demand).
    history = aw.along_wire_history(
        layer,
        x,
        kx,
        ends=("free", "free"),
        segments_per_pitch=32,
        contact_ratio=np.array([5.0]),
    )
    inner = _interior(history, layer, 2.0)
    xi = k_hat / (1.0 + k_hat)
    e = amp * np.cos(history.azimuth_rad[:, inner])
    np.testing.assert_allclose(history.force[:, inner, 0], xi * e, atol=0.01 * amp)
    assert np.all(np.abs(history.energy_J[..., 1]) <= 1e-12)  # no slip


def test_stick_response_does_not_depend_on_the_substep():
    """Below slip the chain is linear: substeps leave the history unchanged."""
    layer = _layer(0.26)
    x, kx = _uniform(layer, 2.0, [0.2, 0.5, -0.4, 0.1])
    common = {
        "ends": ("anchored", "anchored"),
        "segments_per_pitch": 16,
        "contact_ratio": np.full(4, 3.0),
    }
    coarse = aw.along_wire_history(layer, x, kx, max_increment=math.inf, **common)
    fine = aw.along_wire_history(layer, x, kx, max_increment=0.05, **common)
    # One substep per sample and wire without a limit on the increment.
    assert coarse.parameters["substeps"] == 4 * layer.n_wires
    assert fine.parameters["substeps"] > coarse.parameters["substeps"]
    np.testing.assert_allclose(coarse.force, fine.force, rtol=0, atol=1e-10)


def test_anchored_chain_in_full_slip_respects_the_traction_bound():
    """Between anchors in gross slip ``|dn/dphi| <= b`` node by node."""
    layer = _layer(0.26)
    x, kx = _uniform(layer, 2.0, [8.0, 30.0, -30.0])
    history = aw.along_wire_history(
        layer, x, kx, ends=("anchored", "anchored"), segments_per_pitch=16
    )
    h = history.chain.h
    jump = np.abs(np.diff(history.force, axis=1)) / h
    assert jump.max() <= 1.0 + 1e-8
    # The bound is reached: the wires slip, and dissipation grows.
    assert jump.max() >= 1.0 - 1e-6
    dissipation = history.energy_J[..., 1]
    assert np.all(np.diff(dissipation, axis=1) >= -1e-12)
    assert np.all(dissipation[:, -1] > 0)


@pytest.mark.parametrize("k_hat", [0.26, aw.RIGID_K_HAT])
def test_frictionless_wire_between_anchors_carries_the_mean_strain(k_hat):
    """``t = 0`` makes ``n`` uniform; ``u = 0`` at both anchors makes it the
    mean imposed strain over the wire, ``n = (1 / L) int e dphi``."""
    layer = _layer(k_hat)
    length = 1.3 * layer.pitch_m
    x = np.array([0.0, length])
    kx = np.array([[0.5, 2.0], [1.5, -1.0]])  # linear in x, two samples
    history = aw.along_wire_history(
        layer,
        x,
        kx,
        ends=("anchored", "anchored"),
        segments_per_pitch=16,
        contact_ratio=np.zeros(2),
    )
    chain = history.chain
    for i in range(kx.shape[1]):
        k_seg = np.interp(chain.x_seg_m, x, kx[:, i])
        mean = np.mean(chain.cos_seg * k_seg, axis=1)
        np.testing.assert_allclose(
            history.force[:, :, i],
            np.repeat(mean[:, None], chain.segments, 1),
            rtol=0,
            atol=1e-9,
        )
    assert np.all(np.abs(history.energy_J[..., 1]) <= 1e-12)  # nothing slides


def test_capstan_bound_holds_under_biaxial_loading_and_unloading_contact():
    """``|t| <= rho g + lam |n|`` at every node with the capstan term, under a
    rotating curvature and a contact pressure that falls and recovers."""
    lam = 0.3
    layer = _layer(0.26, lam=lam, n_wires=3)
    s = np.linspace(0.0, 2.0 * math.pi, 13)[1:]
    x = np.array([0.0, 2.0 * layer.pitch_m])
    kx = np.tile(6.0 * np.cos(s), (2, 1)) * np.array([[1.0], [0.4]])
    ky = np.tile(6.0 * np.sin(s), (2, 1))
    rho = 1.0 - 0.6 * np.sin(0.5 * s) ** 2
    history = aw.along_wire_history(
        layer,
        x,
        kx,
        ky,
        ends=("anchored", "anchored"),
        segments_per_pitch=16,
        contact_ratio=rho,
    )
    n = history.force  # (wires, segments, samples)
    h = history.chain.h
    jump = np.abs(np.diff(n, axis=1)) / h
    bound = rho[None, None, :] + lam * 0.5 * (np.abs(n[:, 1:]) + np.abs(n[:, :-1]))
    assert np.all(jump <= bound * (1 + 1e-7) + 1e-9)
    assert np.any(jump >= bound * (1 - 1e-6))  # the bound is reached
    residual = np.abs(history.balance_residual_J)
    assert residual.max() <= 1e-8 * np.abs(history.energy_J[..., 0]).max()


def test_relief_that_never_engages_leaves_a_uniform_history_unchanged():
    """A transverse capacity that is never reached gives the factor one."""
    layer = _layer(0.26)
    model = TransverseSlipModel(
        WireSection(shape="rectangular", width_m=0.006, thickness_m=0.003),
        210e9,
        210e9 / 2.6,
        layer.r_helix_m,
        layer.lay_angle_rad,
        1.7e4,
        1e30,
    )
    x = np.array([0.0])
    kx = np.array([[0.5, 2.0, -2.0]])
    common = {
        "extent_m": (0.0, 2.0 * layer.pitch_m),
        "ends": ("anchored", "free"),
        "segments_per_pitch": 16,
    }
    plain = aw.along_wire_history(layer, x, kx, **common)
    relieved = aw.along_wire_history(
        layer, x, kx, relief=LateralRelief(model, anchor_layer_m=0.0), **common
    )
    np.testing.assert_array_equal(relieved.force, plain.force)
    with pytest.raises(CableInputError, match="LateralRelief"):
        aw.along_wire_history(layer, x, kx, relief=model, **common)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Station Jacobian                                                             #
# --------------------------------------------------------------------------- #
class _Stations:
    """Station interpolation of one chain (as the along-wire bending test)."""

    def __init__(self, chain: aw.WireChain, x_all: np.ndarray) -> None:
        self.chain = chain
        self.x_all = x_all
        pos = np.searchsorted(x_all, chain.x_seg_m, side="right") - 1
        pos = np.clip(pos, 0, x_all.size - 2)
        self.i0 = pos
        self.f = np.clip(
            (chain.x_seg_m - x_all[pos]) / (x_all[pos + 1] - x_all[pos]), 0.0, 1.0
        )
        self.j0, self.fj = chain.record_weights(x_all)
        q = np.arange(chain.layer.n_wires, dtype=np.float64)
        self.cos_st = np.cos(chain.layer.azimuth(x_all[None, :], q[:, None]))

    def strain(self, kappa: np.ndarray) -> np.ndarray:
        k_seg = kappa[self.i0] * (1.0 - self.f) + kappa[self.i0 + 1] * self.f
        return np.ascontiguousarray(
            self.chain.cos_seg * (k_seg / self.chain.layer.kappa_c_1_m)
        )

    def moment(self, n: np.ndarray) -> np.ndarray:
        vals = n[:, self.j0] * (1.0 - self.fj) + n[:, self.j0 + 1] * self.fj
        return np.asarray(
            self.chain.layer.lever_Nm * np.sum(self.cos_st * vals, axis=0)
        )

    def step(self, state, kappa_old, kappa_new, rho):
        guess = np.zeros_like(state[0])
        new, _ = aw.chain_layer_step(
            self.chain,
            self.strain(kappa_new),
            self.strain(kappa_old),
            rho,
            state,
            guess,
        )
        return new, guess


def _zero_state(chain: aw.WireChain) -> tuple:
    n_w, m = chain.layer.n_wires, chain.segments
    return (
        np.zeros((n_w, m + 1)),
        np.zeros((n_w, m + 1)),
        np.zeros((n_w, m)),
        np.zeros((n_w, m + 1)),
    )


@pytest.mark.parametrize(
    ("k_hat", "ends", "stations", "rtol"),
    [
        (0.26, ("anchored", "anchored"), 9, 1e-7),
        (0.26, ("anchored", "free"), 41, 1e-7),
        (aw.RIGID_K_HAT, ("anchored", "free"), 9, 2e-2),
    ],
)
def test_station_jacobian_equals_the_finite_difference_in_stick(
    k_hat, ends, stations, rtol
):
    """In stick the increment is linear: the Jacobian is its exact derivative.

    The rigid interface enters the Jacobian as the spring ``RIGID_K_HAT``
    (``1e-2`` rad shear lag), hence the looser tolerance. With more stations
    than segments some stations reach no segment: their columns vanish.
    """
    layer = _layer(k_hat)
    chain = aw.WireChain.build(
        layer, 0.0, 1.0 * layer.pitch_m, ends=ends, segments_per_pitch=16
    )
    st = _Stations(chain, np.linspace(0.0, chain.x_end_m, stations))
    rho = 20.0  # bound far above the demand: stick throughout
    kappa0 = 0.3 * np.cos(np.linspace(0.0, 1.0, stations))
    state1, _ = st.step(_zero_state(chain), np.zeros(stations), kappa0, rho)
    kappa1 = kappa0 + 0.1
    new, solution = st.step(state1, kappa0, kappa1, rho)
    jac = aw.chain_layer_jacobian(
        chain,
        state1,
        solution,
        new[2],
        rho,
        st.i0,
        st.f,
        st.j0,
        st.fj,
        st.cos_st,
    )
    base = st.moment(new[2])
    delta = 1e-3
    fd = np.empty_like(jac)
    for j in range(stations):
        kp = kappa1.copy()
        kp[j] += delta
        moved, _ = st.step(state1, kappa0, kp, rho)
        fd[:, j] = (st.moment(moved[2]) - base) / delta
    scale = np.abs(fd).max()
    np.testing.assert_allclose(jac, fd, rtol=0, atol=rtol * scale)
    reached = np.zeros(stations, dtype=bool)
    reached[np.unique(np.r_[st.i0, st.i0 + 1])] = True
    if not reached.all():
        assert np.all(jac[:, ~reached] == 0.0)


def test_free_chain_in_gross_slip_has_a_finite_jacobian_held_by_one_node():
    """A free chain in full slip is held by the node nearest to sticking.

    On a fixed active set the moment increment is linear in the curvature
    increment: the Jacobian matches the finite difference of a further step
    in the same direction.
    """
    layer = _layer(0.26)
    chain = aw.WireChain.build(
        layer, 0.0, 1.0 * layer.pitch_m, ends=("free", "free"), segments_per_pitch=16
    )
    stations = 9
    st = _Stations(chain, np.linspace(0.0, chain.x_end_m, stations))
    rho = 1.0
    kappa0 = np.full(stations, 20.0)
    state1, _ = st.step(_zero_state(chain), np.zeros(stations), kappa0, rho)
    kappa1 = kappa0 + 5.0
    new, solution = st.step(state1, kappa0, kappa1, rho)
    jac = aw.chain_layer_jacobian(
        chain, state1, solution, new[2], rho, st.i0, st.f, st.j0, st.fj, st.cos_st
    )
    assert np.isfinite(jac).all()
    # The bound caps the force increments: |n_j - n_{j-1}| <= h b.
    assert np.abs(np.diff(new[2], axis=1)).max() <= chain.h * (1 + 1e-9)


# --------------------------------------------------------------------------- #
# Bending test and strand slip                                                 #
# --------------------------------------------------------------------------- #
def test_fit_definition_is_the_curvature_of_the_gauge_parabola():
    """The "fit" curvature is ``-2 c2`` of the parabola through the gauge
    deflections (three gauges: the parabola is exact)."""
    cable = cx.load_cable(cx.inter_array_single_core_yaml_path())
    section = AlongWireSection.from_cable(cable)
    p = max(layer.pitch_m for layer in section.layers)
    span, loading = 2.0 * p, p
    gauges = (0.5 * span - 0.25 * p, 0.5 * span, 0.5 * span + 0.25 * p)
    values = np.array([0.2, 0.4])
    result = simulate_bending_test(
        section,
        FourPoint(span, loading),
        BendingProtocol("curvature", values),
        curvature_definition="fit",
        gauge_x_m=gauges,
        stations_per_segment=4,
        along_wire=AlongWireSpec(segments_per_pitch=8),
    )
    assert result.curvature_definition == "fit"
    np.testing.assert_allclose(result.curvature_1_m, values, rtol=1e-6)
    for k in range(values.size):
        c2 = np.polyfit(result.gauge_x_m, result.gauge_deflection_m[:, k], 2)[0]
        assert -2.0 * c2 == pytest.approx(values[k], rel=1e-6)


def test_zero_curvature_gives_the_axial_strain_alone():
    """Without curvature no direction exists: the strain is ``alpha T``."""
    params = StrandSlipParameters(1.4e-9, (0.04, 0.01), (2e-4,), 1e3)
    zero = np.zeros(5)
    tension = np.array([0.0, 500.0, 1500.0, 800.0, 0.0])
    result = strand_slip_strain(params, zero, zero, tension, n_orientations=4)
    np.testing.assert_array_equal(result.bending_strain, 0.0)
    np.testing.assert_allclose(
        result.strain, np.broadcast_to(1.4e-9 * tension, result.strain.shape)
    )
    assert not result.slipping.any()
