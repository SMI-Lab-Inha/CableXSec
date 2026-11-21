# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""The normal wire curvature follows the wire's axial slip state.

On the deformed helix the normal (weak-axis) curvature change is
``cos^2 a cos 2a p_n`` per unit current length; the wire's bending strain is
its cross-section rotation per unit material length, which adds
``(sin^2 a / r) eps_w`` of the axial strain ``eps_w = sigma_f / E`` the wire
carries (``mechanics.wire_stress`` module notes). The analytical limits:

* stuck wire with the plane-section strain (``xi = 1``): ``cos^4 a``;
* stuck with the shear-lag coefficient ``xi``:
  ``cos^2 a cos 2a + xi sin^2 a cos^2 a``;
* no friction stress (``mu = 0``): ``cos^2 a cos 2a``
  ([MenardCartraud2023] Eq. 14);
* Coulomb slip: slope ``cos^2 a cos 2a``, offset of the friction stress.

The lateral (in-surface) term ``cos a (1 + sin^2 a)`` is unchanged.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import cablexsec as cx
import cablexsec.analysis.fatigue as fatigue
from cablexsec.analysis.capacity import _local_bending_max
from cablexsec.analysis.fast_field import assemble_wire_field
from cablexsec.analysis.fatigue import FatigueLayer, wire_stress_field
from cablexsec.exceptions import CableInputError
from cablexsec.materials import library
from cablexsec.materials.sn_curves import SN_CURVES
from cablexsec.mechanics.core_kinematics import core_kinematics
from cablexsec.mechanics.mkappa import HelicalLayerSpec
from cablexsec.mechanics.strand_slip import (
    ConductorRing,
    conductor_ring_parameters,
    strand_slip_strain,
)
from cablexsec.mechanics.wire_stress import (
    local_bending_amplitude,
    local_bending_fibre_stress,
    local_bending_stress,
    slip_state_normal_curvature,
)

pytestmark = pytest.mark.analytical_sanity

ANGLES_DEG = (10.0, 20.1, 35.0)
E_W = 200e9
R_H = 0.05
D_W = 0.004


def _spec(alpha_deg: float, xi: float = 1.0, mu: float = 50.0) -> HelicalLayerSpec:
    return HelicalLayerSpec(
        n_wires=16,
        r_helix_m=R_H,
        lay_angle_rad=math.radians(alpha_deg),
        E_wire_Pa=E_W,
        A_wire_m2=math.pi * (D_W / 2) ** 2,
        xi=xi,
        mu=mu,
        F_n_axi_N=2e3,
        contact_pressure_Pa=2e5,
    )


def _layer(spec: HelicalLayerSpec) -> FatigueLayer:
    return FatigueLayer(
        spec=spec, sn_bulk=SN_CURVES["dnv_rpc203_t_curve"], sigma_UTS_Pa=1e9
    )


def _ratio(field, spec: HelicalLayerSpec, kx: np.ndarray) -> np.ndarray:
    """Outer-face bending over ``E (d/2) p_n`` where ``|p_n|`` is not small."""
    p_n = np.cos(spec.phi_wires)[:, None] * kx[None, :]
    mask = np.abs(p_n) > 0.05 * np.abs(p_n).max()
    return np.asarray(field.radial_Pa[mask] / (E_W * D_W / 2 * p_n[mask]))


def _coefficients(alpha_deg: float) -> tuple[float, float]:
    a = math.radians(alpha_deg)
    return math.cos(a), math.sin(a)


def test_slip_state_normal_curvature_is_the_initial_curvature_times_the_strain():
    a = math.radians(20.1)
    value = slip_state_normal_curvature(np.array([3e7, -2e7]), E_W, a, R_H)
    expected = math.sin(a) ** 2 / R_H * np.array([3e7, -2e7]) / E_W
    np.testing.assert_allclose(value, expected, rtol=1e-15, atol=0)
    for radius in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(CableInputError, match="helix radius"):
            slip_state_normal_curvature(1e7, E_W, a, radius)
    with pytest.raises(CableInputError, match="r_helix_m"):
        local_bending_fibre_stress(0.01, 0.0, 0.0, E_W, D_W, a, friction_stress_Pa=1e7)


@pytest.mark.parametrize("alpha_deg", ANGLES_DEG)
@pytest.mark.parametrize("fast", [True, False])
def test_stuck_wire_takes_cos4(monkeypatch, alpha_deg, fast):
    # Monotonic ramp far below the slip onset: every wire sticks with the
    # plane-section strain (xi = 1).
    monkeypatch.setattr(fatigue, "_FAST_FIELD", fast)
    spec = _spec(alpha_deg)
    kx = np.linspace(0.0, 1e-3, 11)
    field = wire_stress_field(_layer(spec), kx, np.zeros_like(kx), keep_friction=True)
    assert field.friction_Pa is not None
    stick = spec.k_f * R_H * np.cos(spec.phi_wires)[:, None] * kx[None, :]
    np.testing.assert_allclose(field.friction_Pa, stick, rtol=1e-12, atol=1e-6)
    c, _s = _coefficients(alpha_deg)
    np.testing.assert_allclose(_ratio(field, spec, kx), c**4, rtol=1e-12, atol=0)


@pytest.mark.parametrize("alpha_deg", ANGLES_DEG)
@pytest.mark.parametrize("xi", [0.35, 0.8])
def test_stuck_wire_with_shear_lag_carries_xi_of_the_correction(alpha_deg, xi):
    spec = _spec(alpha_deg, xi=xi)
    kx = np.linspace(0.0, 1e-3, 11)
    field = wire_stress_field(_layer(spec), kx, np.zeros_like(kx))
    c, s = _coefficients(alpha_deg)
    expected = c * c * math.cos(2 * math.radians(alpha_deg)) + xi * s * s * c * c
    np.testing.assert_allclose(_ratio(field, spec, kx), expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("alpha_deg", ANGLES_DEG)
@pytest.mark.parametrize("model", ["direction_tied", "distributed", "progressive"])
def test_frictionless_wire_keeps_the_deformed_helix_value(alpha_deg, model):
    spec = _spec(alpha_deg, mu=0.0)
    kx = 0.01 * np.sin(np.linspace(0.0, 6.0, 40))
    ky = 0.004 * np.cos(np.linspace(0.0, 5.0, 40))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        field = wire_stress_field(_layer(spec), kx, ky, friction_model=model)
    legacy = local_bending_fibre_stress(
        kx, ky, spec.phi_wires[:, None], E_W, D_W, spec.lay_angle_rad
    )
    scale = float(np.abs(legacy).max())
    np.testing.assert_allclose(field.radial_Pa, legacy, rtol=0, atol=1e-12 * scale)
    c, _s = _coefficients(alpha_deg)
    ratio = _ratio(wire_stress_field(_layer(spec), kx, 0 * kx), spec, kx)
    np.testing.assert_allclose(
        ratio, c * c * math.cos(2 * math.radians(alpha_deg)), rtol=1e-12, atol=0
    )


def _cyclic(n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    # Amplitude well above the onset, off the sample grid at every transition.
    t = np.linspace(0.0, 3.0, n)
    return 0.031 * np.sin(2 * np.pi * t + 0.17), 0.0071 * np.cos(2 * np.pi * t)


@pytest.mark.parametrize("model", ["direction_tied", "distributed", "progressive"])
@pytest.mark.parametrize("fast", [True, False])
def test_partial_slip_adds_the_term_of_the_carried_friction_stress(
    monkeypatch, model, fast
):
    monkeypatch.setattr(fatigue, "_FAST_FIELD", fast)
    spec = _spec(20.1, xi=0.8, mu=0.15)
    kx, ky = _cyclic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        field = wire_stress_field(
            _layer(spec), kx, ky, friction_model=model, keep_friction=True
        )
    assert field.friction_Pa is not None
    legacy = local_bending_fibre_stress(
        kx, ky, spec.phi_wires[:, None], E_W, D_W, spec.lay_angle_rad
    )
    s2 = math.sin(spec.lay_angle_rad) ** 2
    expected = legacy + D_W / 2 * s2 / R_H * field.friction_Pa
    scale = float(np.abs(field.radial_Pa).max())
    np.testing.assert_allclose(field.radial_Pa, expected, rtol=0, atol=1e-12 * scale)
    # The term is not negligible in this history.
    assert np.abs(field.radial_Pa - legacy).max() > 1e-3 * scale


def test_stick_and_slip_segments_take_their_own_slopes():
    # Direction-tied spring-slider on a planar history: a step is stick when
    # the friction increment is the elastic increment k_f r dp_n, slip when
    # the friction stress holds its cap.
    xi = 0.8
    spec = _spec(20.1, xi=xi, mu=0.15)
    t = np.linspace(0.0, 3.0, 601)
    kx = 0.031 * np.sin(2 * np.pi * t + 0.17)
    field = wire_stress_field(_layer(spec), kx, 0 * kx, keep_friction=True)
    assert field.friction_Pa is not None
    p_n = np.cos(spec.phi_wires)[:, None] * kx[None, :]
    dp = np.diff(p_n, axis=1)
    df = np.diff(field.friction_Pa, axis=1)
    dr = np.diff(field.radial_Pa, axis=1) / (E_W * D_W / 2)
    big = np.abs(dp) > 1e-6
    stick = big & (np.abs(df - spec.k_f * R_H * dp) <= 1e-9 * spec.k_f * R_H * 0.03)
    slip = big & (np.abs(df) <= 1e-9 * spec.k_f * R_H * 0.03)
    assert stick.sum() > 50 and slip.sum() > 50
    c, s = _coefficients(20.1)
    lox = c * c * math.cos(2 * math.radians(20.1))
    np.testing.assert_allclose(
        dr[stick] / dp[stick], lox + xi * s * s * c * c, rtol=1e-9
    )
    np.testing.assert_allclose(dr[slip] / dp[slip], lox, rtol=1e-9)


def test_tangential_term_is_unchanged():
    spec = _spec(35.0, xi=0.8, mu=0.15)
    kx, ky = _cyclic(200)
    field = wire_stress_field(_layer(spec), kx, ky)
    legacy = local_bending_fibre_stress(
        kx, ky, spec.phi_wires[:, None], E_W, D_W, spec.lay_angle_rad, math.pi / 2
    )
    scale = float(np.abs(legacy).max())
    np.testing.assert_allclose(field.tangential_Pa, legacy, rtol=0, atol=1e-12 * scale)


def test_assembly_with_friction_and_increment_equals_the_numpy_expressions():
    rng = np.random.default_rng(3)
    n_w, n_t = 6, 70
    kx, ky = rng.normal(0, 0.01, n_t), rng.normal(0, 0.005, n_t)
    phi = np.linspace(0, 2 * math.pi, n_w, endpoint=False)
    friction = rng.normal(0, 4e7, (n_w, n_t))
    extra = rng.normal(0, 1e7, (n_w, n_t))
    modulus = 200e9 * (1 + 0.02 * np.sin(np.linspace(0, 4, n_t)))
    axial, thermal = rng.normal(0, 1e7, n_t), rng.normal(0, 1e6, n_t)
    mean, radial, tangential = assemble_wire_field(
        kx, ky, phi, modulus, D_W, 0.006, 0.3, axial, friction, thermal, extra, R_H
    )
    held = friction + extra
    radial_ref = local_bending_fibre_stress(
        kx, ky, phi[:, None], modulus, D_W, 0.3, friction_stress_Pa=held, r_helix_m=R_H
    )
    edge_ref = local_bending_fibre_stress(
        kx,
        ky,
        phi[:, None],
        modulus,
        0.006,
        0.3,
        math.pi / 2,
        friction_stress_Pa=held,
        r_helix_m=R_H,
    )
    mean_ref = axial[None, :] + friction + thermal[None, :] + extra
    scale = float(np.abs(mean_ref).max())
    np.testing.assert_allclose(radial, radial_ref, rtol=0, atol=1e-12 * scale)
    np.testing.assert_allclose(tangential, edge_ref, rtol=0, atol=1e-12 * scale)
    np.testing.assert_allclose(mean, mean_ref, rtol=0, atol=1e-12 * scale)
    with pytest.raises(CableInputError, match="helix radius"):
        assemble_wire_field(
            kx, ky, phi, modulus, D_W, 0.006, 0.3, axial, friction, thermal, None, 0.0
        )


def test_termination_increment_enters_the_carried_stress(monkeypatch):
    from cablexsec.analysis.termination import TerminationSpec  # noqa: PLC0415

    spec = _spec(20.1, xi=0.8, mu=0.15)
    kx, ky = _cyclic(120)
    term = TerminationSpec(
        distance_m=0.5, segments_per_pitch=8, max_increment=2.0, reach_lengths=1.0
    )
    for fast in (True, False):
        monkeypatch.setattr(fatigue, "_FAST_FIELD", fast)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            field = wire_stress_field(
                _layer(spec), kx, ky, termination=term, keep_friction=True
            )
        assert field.termination is not None and field.termination.applied
        assert field.friction_Pa is not None  # sectional plus increment
        legacy = local_bending_fibre_stress(
            kx, ky, spec.phi_wires[:, None], E_W, D_W, spec.lay_angle_rad
        )
        s2 = math.sin(spec.lay_angle_rad) ** 2
        expected = legacy + D_W / 2 * s2 / R_H * field.friction_Pa
        scale = float(np.abs(field.radial_Pa).max())
        np.testing.assert_allclose(
            field.radial_Pa, expected, rtol=0, atol=1e-12 * scale
        )


def test_scalar_helpers_add_the_slip_state_term():
    a = math.radians(20.1)
    c, s = math.cos(a), math.sin(a)
    phi = np.linspace(0, 2 * math.pi, 9, endpoint=False)
    kappa = 0.02
    sigma = 4e7 * np.cos(phi)
    slip = s * s / R_H * sigma / E_W
    normal = kappa * c * c * math.cos(2 * a) * np.cos(phi) + slip
    stress = local_bending_stress(
        kappa, phi, E_W, D_W, a, friction_stress_Pa=sigma, r_helix_m=R_H
    )
    np.testing.assert_allclose(stress, E_W * D_W / 2 * normal, rtol=1e-13, atol=0)
    amplitude = local_bending_amplitude(
        kappa, phi, E_W, D_W, a, friction_stress_Pa=sigma, r_helix_m=R_H
    )
    lateral = kappa * c * (1 + s * s) * np.sin(phi)
    np.testing.assert_allclose(
        amplitude, E_W * D_W / 2 * np.hypot(normal, lateral), rtol=1e-13, atol=0
    )
    # Without a friction stress: the unchanged Eq. 14 helpers.
    assert np.array_equal(
        local_bending_amplitude(-kappa, phi, E_W, D_W, a),
        local_bending_amplitude(kappa, phi, E_W, D_W, a),
    )


def test_stress_route_follows_the_slip_front():
    cable = cx.load_cable(cx.menard2023_yaml_path())
    for mu in (0.0, None):
        result = cx.compute_stress(cable, 0.03, 0.01, tension_N=1e4, mu=mu)
        for name, sigma_b in result.sigma_B_Pa.items():
            layer = cable.get_layer(name)
            a = layer.lay_angle_rad
            assert a is not None and layer.r_helix_m is not None
            e = library.get_property(layer.material, "E_Pa")
            phi = np.radians(result.phi_deg[name])
            theta = phi - math.atan2(0.01, 0.03)
            kappa = math.hypot(0.03, 0.01)
            expected = (
                e
                * layer.d_wire_m
                / 2
                * (
                    kappa * math.cos(a) ** 2 * math.cos(2 * a) * np.cos(theta)
                    + math.sin(a) ** 2 / layer.r_helix_m * result.sigma_f_Pa[name] / e
                )
            )
            scale = float(np.abs(expected).max())
            np.testing.assert_allclose(sigma_b, expected, rtol=0, atol=1e-12 * scale)
            if mu == 0.0:
                assert np.all(result.sigma_f_Pa[name] == 0.0)


def test_capacity_bending_adds_the_slip_state_term():
    cable = cx.load_cable(cx.menard2023_yaml_path())
    layer = next(
        x for x in cable.helical_layers if getattr(x, "wire_shape", None) == "round"
    )
    a = layer.lay_angle_rad
    assert a is not None and layer.r_helix_m is not None
    phi = np.linspace(0, 2 * math.pi, 12, endpoint=False)
    sigma = 3e7 * np.sign(np.cos(phi))
    got = _local_bending_max(
        layer, 0.05, phi, E_W, a, 0.0, friction_stress_Pa=sigma, r_helix_m=0.06
    )
    ref = local_bending_amplitude(
        0.05, phi, E_W, layer.d_wire_m, a, friction_stress_Pa=sigma, r_helix_m=0.06
    )
    np.testing.assert_allclose(got, ref, rtol=1e-15, atol=0)
    plain = _local_bending_max(layer, 0.05, phi, E_W, a, 0.0)
    assert np.all(got[np.cos(phi) > 0.1] > plain[np.cos(phi) > 0.1])


def test_capacity_rectangular_face_adds_the_slip_state_term():
    rect = cx.load_cable(cx.saevik2011_yaml_path("rectangular"))
    layer = next(x for x in rect.helical_layers if x.wire_shape != "round")
    a = layer.lay_angle_rad
    assert a is not None
    section = layer.wire_section
    phi = np.linspace(0, 2 * math.pi, 12, endpoint=False)
    sigma = 3e7 * np.cos(phi)
    got = _local_bending_max(
        layer, 0.05, phi, E_W, a, 0.0, friction_stress_Pa=sigma, r_helix_m=0.1
    )
    c, s = math.cos(a), math.sin(a)
    face = E_W * (
        0.05 * c * c * math.cos(2 * a) * np.cos(phi) + s * s / 0.1 * sigma / E_W
    )
    edge = E_W * 0.05 * c * (1 + s * s) * np.sin(phi)
    ref = section.max_bending_stress(
        face * section.half_thickness_m, edge * section.half_width_m
    )
    np.testing.assert_allclose(got, ref, rtol=1e-14, atol=0)


def test_laid_core_normal_gradient_follows_the_core_state():
    cable = cx.load_cable(cx.iea_task49_66kv_yaml_path())
    t = np.linspace(0.0, 2.0, 30)
    kx, ky = 0.02 * np.sin(2 * np.pi * t + 0.1), 0.01 * np.cos(2 * np.pi * t)
    eps = 1e-4 * np.ones_like(t)
    for state in ("stuck", "free", "frictional"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kin = core_kinematics(cable, kx, ky, state=state, axial_strain=eps)
        beta, r_c = kin.lay_angle_rad, kin.r_offset_m
        c, s = math.cos(beta), math.sin(beta)
        kr = (
            np.cos(kin.phase_rad)[:, None] * kx[None, :]
            + np.sin(kin.phase_rad)[:, None] * ky[None, :]
        )
        bending = kin.axis_strain - c * c * eps[None, :]
        expected = c * c * math.cos(2 * beta) * kr + s * s / r_c * bending
        scale = float(np.abs(kr).max())
        np.testing.assert_allclose(
            kin.gradient_radial_per_m, expected, rtol=0, atol=1e-12 * scale
        )
        if state == "stuck":
            np.testing.assert_allclose(
                kin.gradient_radial_per_m, c**4 * kr, rtol=0, atol=1e-12 * scale
            )
        if state == "free":
            np.testing.assert_allclose(
                kin.gradient_radial_per_m,
                c * c * math.cos(2 * beta) * kr,
                rtol=0,
                atol=1e-12 * scale,
            )


def test_strand_slip_outer_fibre_stick_slope_is_cos4():
    rings = [
        ConductorRing(n_wires=1, d_wire_m=0.003, r_helix_m=0.0, lay_angle_rad=0.0),
        ConductorRing(
            n_wires=6, d_wire_m=0.003, r_helix_m=0.003, lay_angle_rad=math.radians(12.0)
        ),
        ConductorRing(
            n_wires=12,
            d_wire_m=0.003,
            r_helix_m=0.006,
            lay_angle_rad=math.radians(14.0),
        ),
    ]
    kwargs = {"mu_wire": 0.3, "reference_tension_N": 2e4}
    outer = conductor_ring_parameters(rings, 110e9, fibre="outer", **kwargs)
    centroid = conductor_ring_parameters(rings, 110e9, fibre="centroid", **kwargs)
    assert outer[0].slip_state_factor == centroid[0].slip_state_factor == 1.0
    # Planar ramp to well above the onset, then the element at theta = 0.
    kappa = np.linspace(0.0, 1.0, 2001)
    for ring, o, m in zip(rings[1:], outer[1:], centroid[1:], strict=True):
        c, s = math.cos(ring.lay_angle_rad), math.sin(ring.lay_angle_rad)
        r, half = ring.r_helix_m, ring.d_wire_m / 2
        lox = half * c * c * math.cos(2 * ring.lay_angle_rad)
        # The friction amplitude is the friction strain; the outer fibre
        # carries the wire-axis strain with the factor of its slip state.
        assert o.eps_f_ref == m.eps_f_ref
        assert o.slip_state_factor == pytest.approx(1.0 + half * s * s / r, rel=1e-15)
        assert m.slip_state_factor == 1.0
        assert m.beta_m == pytest.approx((r * c * c, 0.0), rel=1e-15)
        strain = strand_slip_strain(
            o, kappa, np.zeros_like(kappa), np.full_like(kappa, 2e4), theta_rad=[0.0]
        ).strain[0]
        onset = o.eps_f_ref[0] / (o.beta_m[0] - o.beta_m[1])
        stick = kappa < 0.9 * onset
        slip = kappa > 1.1 * onset
        assert stick.sum() > 10 and slip.sum() > 10
        axial = o.alpha_per_N * 2e4
        np.testing.assert_allclose(
            strain[stick] - axial, (r * c * c + half * c**4) * kappa[stick], rtol=1e-12
        )
        np.testing.assert_allclose(
            strain[slip] - axial,
            lox * kappa[slip] + o.slip_state_factor * o.eps_f_ref[0],
            rtol=1e-12,
        )
