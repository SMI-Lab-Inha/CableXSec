# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Geodesic transverse slip map (gap 1.4b).

``transverse_map="geodesic"`` puts the transverse contact slip in phase with
the frictionless geodesic slip of Out and von Morgen (1997, Eqs. 26-27);
``transverse_map="legacy"`` keeps the former map, which is in quadrature with
it. See ``scripts/check_out_von_morgen.py`` for the sources and conventions.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace

import numpy as np
import pytest
from check_out_von_morgen import (
    INSIDE,
    cablexsec_cycle_slip,
    compare_geodesic_map,
    geodesic_map_exact_error,
    geodesic_transverse_ratio,
    ovm_slip_magnitude,
    transverse_geodesic_scale,
)

from cablexsec.analysis.fatigue import (
    FatigueLayer,
    _contact_force_histories,
    _fretting_amplitudes_um,
    compute_fatigue,
)
from cablexsec.exceptions import CableInputError
from cablexsec.materials.sn_curves import SNcurve, get_sn_curve
from cablexsec.mechanics.mindlin_contact import (
    MindlinContact,
    TransverseMapWarning,
    _trajectory_diameter,
    geometric_slip_factor,
    loxodromic_slip_increment,
    slip_increment,
    transverse_geodesic_factor,
)
from cablexsec.mechanics.mkappa import HelicalLayerSpec

pytestmark = pytest.mark.analytical_sanity

R, ALPHA = 0.05, math.radians(35.0)
C_GEO = geometric_slip_factor(R, ALPHA)


def _layer(**changes) -> FatigueLayer:
    spec = HelicalLayerSpec(
        n_wires=12,
        r_helix_m=R,
        lay_angle_rad=ALPHA,
        E_wire_Pa=200e9,
        A_wire_m2=2e-5,
        xi=0.8,
        mu=0.3,
        F_n_axi_N=100.0,
        contact_pressure_Pa=3e5,
        normal_load_is_line=True,
    )
    base = FatigueLayer(
        spec=spec,
        sn_bulk=get_sn_curve("dnv_rpc203_t_curve"),
        sigma_UTS_Pa=1.4e9,
        sn_fretting=SNcurve(
            name="test-only", stress_unit="um", m1=2.0, log_C1=10.0, threshold=0.0
        ),
        K_t_contact_N_per_m=1e7,
        fretting_mu=0.2,
        crossing_density_per_m=20.0,
        contact_line_load_N_per_m=400.0,
    )
    return replace(base, **changes)


# --------------------------------------------------------------------------- #
# Scale and closed form                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("deg", [15.0, 25.0, 35.0, 45.0, 55.0])
def test_scale_matches_saevik_eq_2_30_and_the_check_script(deg):
    alpha = math.radians(deg)
    assert transverse_geodesic_factor(R, alpha) == pytest.approx(
        transverse_geodesic_scale(R, alpha), rel=1e-14
    )
    assert transverse_geodesic_factor(R, alpha) / geometric_slip_factor(
        R, alpha
    ) == pytest.approx(geodesic_transverse_ratio(alpha), rel=1e-14)
    assert transverse_geodesic_factor(R, -alpha) == -transverse_geodesic_factor(
        R, alpha
    )
    with pytest.raises(CableInputError, match="non-zero"):
        transverse_geodesic_factor(R, 0.0)


def test_eta_one_reproduces_out_von_morgen_closed_form():
    worst = compare_geodesic_map(azimuths=24)
    assert worst["closed_form_residual"] < 1e-13
    # Frictionless cycle: delta* equals |Delta| of Eq. 28 at every azimuth.
    assert worst["cycle_amplitude_over_eq28_min"] == pytest.approx(1.0, abs=1e-12)
    assert worst["cycle_amplitude_over_eq28_max"] == pytest.approx(1.0, abs=1e-12)
    # In phase: zero at the extreme fibres, geodesic at the neutral axis.
    assert worst["transverse_at_extreme_fibre_over_geodesic"] < 1e-14
    assert worst["transverse_at_neutral_axis_over_geodesic_min"] == pytest.approx(
        1.0, abs=1e-12
    )
    assert worst["transverse_at_neutral_axis_over_geodesic_max"] == pytest.approx(
        1.0, abs=1e-12
    )


def test_eta_one_converges_to_the_exact_geodesic_at_first_order():
    errors = [
        geodesic_map_exact_error(k, ALPHA, points=7, nodes=801) for k in (1e-3, 1e-4)
    ]
    assert errors[1] < 1e-3
    assert errors[0] / errors[1] == pytest.approx(10.0, rel=0.05)


def test_eta_scales_the_transverse_component_only():
    phi = np.linspace(0.0, 2 * math.pi, 19)
    full = slip_increment(
        0.02, -0.01, phi, C_GEO, 1.0, transverse_map="geodesic", lay_angle_rad=ALPHA
    )
    for eta in (0.25, 0.5, 0.75):
        part = slip_increment(
            0.02, -0.01, phi, C_GEO, eta, transverse_map="geodesic", lay_angle_rad=ALPHA
        )
        np.testing.assert_array_equal(part[:, 0], full[:, 0])
        np.testing.assert_allclose(part[:, 1], eta * full[:, 1], rtol=1e-15, atol=0)


# --------------------------------------------------------------------------- #
# Loxodromic limit and defaults                                               #
# --------------------------------------------------------------------------- #
def test_eta_zero_is_bitwise_loxodromic():
    phi = np.linspace(0.0, 2 * math.pi, 37)
    legacy = slip_increment(0.02, -0.01, phi, C_GEO, transverse_ratio=0.0)
    for transverse_map in ("geodesic", "legacy"):
        family = slip_increment(
            0.02,
            -0.01,
            phi,
            C_GEO,
            0.0,
            transverse_map=transverse_map,
            lay_angle_rad=ALPHA,
        )
        assert family.tobytes() == legacy.tobytes()
    np.testing.assert_array_equal(
        legacy, loxodromic_slip_increment(0.02, -0.01, phi, C_GEO)
    )


def test_default_fatigue_layer_is_unchanged_by_the_map_option():
    kappa = np.array([0.0, 0.01, -0.012, 0.008, -0.01, 0.011])
    reference = _layer()
    assert reference.transverse_ratio == 0.0 and reference.transverse_map is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        geodesic_zero = replace(reference, transverse_map="geodesic")
        a = _fretting_amplitudes_um(reference, kappa, 0.3 * kappa, None)
        b = _fretting_amplitudes_um(geodesic_zero, kappa, 0.3 * kappa, None)
        ra = compute_fatigue([reference], kappa, 0.3 * kappa, n_fibres=4)
        rb = compute_fatigue([geodesic_zero], kappa, 0.3 * kappa, n_fibres=4)
    assert a.tobytes() == b.tobytes()
    for name in ra.D_total_per_wire:
        assert (
            ra.D_total_per_wire[name].tobytes() == rb.D_total_per_wire[name].tobytes()
        )


# --------------------------------------------------------------------------- #
# Legacy option and warnings                                                  #
# --------------------------------------------------------------------------- #
def test_implicit_legacy_map_warns_and_explicit_choice_does_not():
    with pytest.warns(TransverseMapWarning, match="quadrature"):
        implicit = slip_increment(0.02, 0.0, 0.4, C_GEO, 0.5)
    with pytest.warns(TransverseMapWarning):
        _layer(transverse_ratio=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        explicit = slip_increment(0.02, 0.0, 0.4, C_GEO, 0.5, transverse_map="legacy")
        slip_increment(0.02, 0.0, 0.4, C_GEO, 0.0)
        _layer(transverse_ratio=1.0, transverse_map="legacy")
        _layer(transverse_ratio=1.0, transverse_map="geodesic")
    assert implicit.tobytes() == explicit.tobytes()


def test_legacy_map_is_in_quadrature_with_the_geodesic_map():
    phi = np.array([0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi])
    legacy = slip_increment(0.02, 0.0, phi, C_GEO, 1.0, transverse_map="legacy")
    geodesic = slip_increment(
        0.02, 0.0, phi, C_GEO, 1.0, transverse_map="geodesic", lay_angle_rad=ALPHA
    )
    # Extreme fibres (0, pi): legacy transverse is maximal, geodesic is zero.
    assert np.abs(legacy[[0, 2], 1]) == pytest.approx([0.02 * C_GEO] * 2)
    assert np.max(np.abs(geodesic[[0, 2]])) < 1e-18
    # Neutral axis: legacy transverse is zero, geodesic is s_geo dk.
    assert np.max(np.abs(legacy[[1, 3], 1])) < 1e-18
    assert np.abs(geodesic[[1, 3], 1]) == pytest.approx(
        [0.02 * transverse_geodesic_factor(R, ALPHA)] * 2, rel=1e-14
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"transverse_ratio": 1.5, "transverse_map": "geodesic"}, "0 <= transverse"),
        ({"transverse_ratio": 0.5, "transverse_map": "geodesic"}, "lay_angle_rad"),
        ({"transverse_ratio": 0.5, "transverse_map": "vector"}, "one of"),
        ({"transverse_ratio": -0.1, "transverse_map": "legacy"}, "non-negative"),
        ({"transverse_ratio": math.nan, "transverse_map": "geodesic"}, "finite"),
    ],
)
def test_invalid_map_inputs(kwargs, match):
    with pytest.raises(CableInputError, match=match):
        slip_increment(0.02, 0.0, 0.4, C_GEO, **kwargs)


@pytest.mark.parametrize("angle", [0.0, 0.5 * math.pi])
def test_geodesic_map_needs_a_helical_lay_angle(angle):
    with pytest.raises(CableInputError, match="lay angle"):
        slip_increment(
            0.02, 0.0, 0.4, C_GEO, 0.5, transverse_map="geodesic", lay_angle_rad=angle
        )


def test_fatigue_layer_rejects_geodesic_eta_above_one():
    with pytest.raises(CableInputError, match="0 <= transverse"):
        _layer(transverse_ratio=2.0, transverse_map="geodesic")


# --------------------------------------------------------------------------- #
# Frictionless limit                                                          #
# --------------------------------------------------------------------------- #
def test_contact_slip_approaches_the_geodesic_as_friction_vanishes():
    kappa, phi = 0.1, 0.5 * math.pi
    bound = float(ovm_slip_magnitude(kappa, R, ALPHA, phi + 1.5 * math.pi, INSIDE))
    ratios = [
        cablexsec_cycle_slip(
            kappa, R, ALPHA, phi, mu=mu, eta=1.0, transverse_map="geodesic"
        )[2]
        / bound
        for mu in (0.3, 0.1, 0.03, 0.01, 0.003, 0.0)
    ]
    assert all(b >= a for a, b in zip(ratios, ratios[1:], strict=False))
    assert ratios[-1] == pytest.approx(1.0, abs=1e-12)


def test_frictionless_fatigue_amplitude_scales_by_the_geodesic_norm():
    kappa = np.array([0.0, 0.01, -0.01, 0.01, -0.01])
    lox = _fretting_amplitudes_um(_layer(fretting_mu=0.0), kappa, 0 * kappa, None)
    geo = _fretting_amplitudes_um(
        _layer(fretting_mu=0.0, transverse_ratio=1.0, transverse_map="geodesic"),
        kappa,
        0 * kappa,
        None,
    )
    moving = lox > 1e-9 * lox.max()
    norm = math.sqrt(1.0 + geodesic_transverse_ratio(ALPHA) ** 2)
    np.testing.assert_allclose(geo[moving] / lox[moving], norm, rtol=1e-12)
    assert np.all(geo[~moving] < 1e-9 * geo.max())


def test_force_history_of_the_geodesic_map_is_collinear():
    kappa = np.array([0.0, 0.01, -0.01, 0.01, -0.01, 0.005])
    layer = _layer(transverse_ratio=1.0, transverse_map="geodesic")
    force, _slip, normal = _contact_force_histories(layer, kappa, 0 * kappa, None)
    assert force.shape == (12, kappa.size)
    # The signed principal-axis force reaches the Coulomb cap at the neutral axis.
    assert np.max(np.abs(force)) == pytest.approx(
        layer.fretting_mu * float(np.max(normal)), rel=1e-12
    )


# --------------------------------------------------------------------------- #
# Biaxial rotation covariance                                                 #
# --------------------------------------------------------------------------- #
def _history(beta: float = 0.0) -> np.ndarray:
    theta = np.linspace(0.0, 12.0 * np.pi, 1200)
    kx = 0.02 * np.sin(theta) + 0.006 * np.sin(0.37 * theta)
    ky = 0.013 * np.cos(0.83 * theta)
    c, s = math.cos(beta), math.sin(beta)
    return np.column_stack((c * kx - s * ky, s * kx + c * ky))


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_biaxial_rotation_moves_the_pattern_with_the_bending_direction(eta):
    beta = 0.73
    phi = np.linspace(0.0, 2 * math.pi, 16, endpoint=False)
    base, turned = _history(), _history(beta)
    kwargs = {"transverse_map": "geodesic", "lay_angle_rad": ALPHA}
    for row in range(1, 40):
        d0 = base[row] - base[row - 1]
        d1 = turned[row] - turned[row - 1]
        a = slip_increment(d0[0], d0[1], phi, C_GEO, eta, **kwargs)
        b = slip_increment(d1[0], d1[1], phi + beta, C_GEO, eta, **kwargs)
        np.testing.assert_allclose(b, a, rtol=0, atol=1e-14 * C_GEO)
    contacts = {key: MindlinContact(K_t=5.0e6, mu=0.3) for key in ("base", "turned")}
    for path, key, azimuth in ((base, "base", 0.4), (turned, "turned", 0.4 + beta)):
        previous = np.zeros(2)
        for current in path:
            delta = current - previous
            previous = current
            contacts[key].update(
                slip_increment(delta[0], delta[1], azimuth, C_GEO, eta, **kwargs), 80.0
            )
    assert contacts["base"].fretting_amplitude == pytest.approx(
        contacts["turned"].fretting_amplitude, rel=1e-10
    )


def test_geodesic_increments_keep_one_direction_for_any_bending_direction():
    rng = np.random.default_rng(1)
    phi = np.linspace(0.0, 2 * math.pi, 13)
    ratio = geodesic_transverse_ratio(ALPHA)
    for dkx, dky in rng.normal(size=(20, 2)) * 0.01:
        d = slip_increment(
            dkx, dky, phi, C_GEO, 1.0, transverse_map="geodesic", lay_angle_rad=ALPHA
        )
        # Out and von Morgen Eqs. 23/29: slip direction independent of theta.
        np.testing.assert_allclose(d[:, 1], -ratio * d[:, 0], rtol=1e-14, atol=0)


# --------------------------------------------------------------------------- #
# Amplitude of almost collinear trajectories                                  #
# --------------------------------------------------------------------------- #
def test_diameter_of_an_almost_collinear_slip_path_is_exact():
    phi = math.radians(15.0)
    path = np.concatenate(
        (
            np.linspace(0, 0.1, 21),
            np.linspace(0.1, -0.1, 41)[1:],
            np.linspace(-0.1, 0.1, 41)[1:],
        )
    )
    total = np.zeros(2)
    points = [total.copy()]
    for dk in np.diff(path):
        total = total + slip_increment(
            float(dk),
            0.0,
            phi,
            C_GEO,
            1.0,
            transverse_map="geodesic",
            lay_angle_rad=ALPHA,
        )
        points.append(total.copy())
    trajectory = np.asarray(points)
    brute = np.max(
        np.linalg.norm(trajectory[:, None, :] - trajectory[None, :, :], axis=-1)
    )
    assert _trajectory_diameter(trajectory) == pytest.approx(brute, rel=1e-12)
