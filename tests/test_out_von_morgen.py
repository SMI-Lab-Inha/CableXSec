# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Frictionless slip bounds of Out and von Morgen (1997) and Out (1989).

Analytical verification of CableXSec's armour slip kinematics; see
``scripts/check_out_von_morgen.py`` for the sources and conventions.
"""

import math

import numpy as np
import pytest
from check_out_von_morgen import (
    INSIDE,
    SIDE,
    cablexsec_cycle_slip,
    compare_frictionless_contact,
    contact_approach,
    deformed_helix_curvature_coefficients,
    exact_geodesic_slip,
    geodesic_attractor_residual,
    geodesic_transverse_ratio,
    numerical_curvature_coefficients,
    out1989_contact_checks,
    out1989_damage_ratio,
    out1989_effective_change,
    ovm_relative_slip_max,
    ovm_slip,
    ovm_slip_angle,
    ovm_slip_magnitude,
    saevik_limit_friction,
    transverse_geodesic_scale,
    wire_frame,
)

from cablexsec.mechanics.mindlin_contact import geometric_slip_factor
from cablexsec.mechanics.wire_stress import local_bending_fibre_stress

pytestmark = pytest.mark.analytical_sanity

ALPHA = math.radians(35.0)


@pytest.mark.parametrize("boundary", [INSIDE, SIDE])
def test_closed_forms_are_first_order_in_curvature(boundary):
    theta0 = 0.5 * math.pi if boundary == INSIDE else 0.0
    theta = theta0 + np.linspace(0.3, 2 * math.pi - 0.3, 7)
    errors = []
    for k in (1e-3, 1e-4):
        exact = exact_geodesic_slip(k, ALPHA, boundary, theta, nodes=801)
        closed = ovm_slip(k, 1.0, ALPHA, theta, boundary)
        scale = float(np.max(ovm_slip_magnitude(k, 1.0, ALPHA, theta, boundary)))
        errors.append(float(np.max(np.linalg.norm(exact - closed, axis=1))) / scale)
    assert errors[1] < 1e-3
    assert errors[0] / errors[1] == pytest.approx(10.0, rel=0.05)


@pytest.mark.parametrize("boundary", [INSIDE, SIDE])
def test_magnitude_angle_and_relative_slip_are_consistent(boundary):
    theta = np.linspace(0.0, 2 * math.pi, 33)[1:-1]
    vector = ovm_slip(0.08, 0.06, ALPHA, theta, boundary)
    magnitude = ovm_slip_magnitude(0.08, 0.06, ALPHA, theta, boundary)
    np.testing.assert_allclose(np.linalg.norm(vector, axis=1), magnitude, rtol=1e-14)
    # Eqs. 23/29: the direction is independent of theta and curvature.
    moving = np.linalg.norm(vector, axis=1) > 1e-12
    angle = np.arctan(vector[moving, 1] / vector[moving, 0])
    np.testing.assert_allclose(angle, ovm_slip_angle(ALPHA), rtol=1e-12)
    # Eq. 30: contra-wound layers (-alpha) give the vector difference.
    if boundary == INSIDE:
        relative = vector - ovm_slip(0.08, 0.06, -ALPHA, theta, boundary)
        assert np.max(np.linalg.norm(relative, axis=1)) == pytest.approx(
            ovm_relative_slip_max(0.08, 0.06, ALPHA), rel=1e-3
        )


def test_wire_frame_components_are_saevik_eqs_2_29_and_2_30():
    kappa, r = 0.25, 0.1  # Saevik (1992) Section 4.3: 25 deg, R = 0.1 m
    alpha = math.radians(25.0)
    theta = np.linspace(0.0, 2 * math.pi, 13)
    lon, tra = wire_frame(ovm_slip(kappa, r, alpha, theta, INSIDE), alpha)
    np.testing.assert_allclose(
        lon, -geometric_slip_factor(r, alpha) * kappa * np.cos(theta), atol=1e-15
    )
    np.testing.assert_allclose(
        tra, transverse_geodesic_scale(r, alpha) * kappa * np.cos(theta), atol=1e-15
    )
    # Figure 4.2 amplitudes of the thesis: about 5 mm and 15 mm.
    assert geometric_slip_factor(r, alpha) * kappa == pytest.approx(4.86e-3, rel=1e-3)
    assert transverse_geodesic_scale(r, alpha) * kappa == pytest.approx(
        1.495e-2, rel=1e-3
    )
    assert geodesic_transverse_ratio(alpha) == pytest.approx(
        transverse_geodesic_scale(r, alpha) / geometric_slip_factor(r, alpha)
    )
    # Eq. 4.2, Figure 4.12 of the thesis: about 0.6 at 25 deg and R/rho = 0.1.
    assert saevik_limit_friction(0.1, 1.0, alpha) == pytest.approx(0.6, abs=0.02)


def test_cablexsec_slip_at_zero_friction_stays_within_the_geodesic_bound():
    worst = compare_frictionless_contact(
        alphas_deg=(20.0, 35.0, 50.0), etas=(0.0, 1.0, 2.0), azimuths=16
    )
    # Default loxodromic map: the wire-direction component of the geodesic slip.
    assert worst["longitudinal_identity"] < 1e-12
    assert worst["transverse_eta0_m"] == 0.0
    assert worst["magnitude_ratio_inside"] < 0.34
    # The prescribed transverse family stays inside the end-fitting envelope
    # for eta <= 2, but it is in quadrature with the geodesic transverse slip.
    assert worst["envelope_ratio"] <= 1.0
    assert worst["transverse_at_extreme_fibre_over_c_geo_kappa"] == pytest.approx(2.0)
    assert worst["transverse_at_neutral_axis_over_geodesic"] < 1e-12


def test_longitudinal_slip_approaches_the_geodesic_component_as_mu_vanishes():
    rows = contact_approach()
    ratios = [ratio for _, ratio in rows]
    assert all(b >= a for a, b in zip(ratios, ratios[1:], strict=False))
    assert max(ratios) <= 1.0 + 1e-12
    assert ratios[-1] == pytest.approx(1.0, abs=1e-12)


def test_frictionless_contact_slip_magnitude_never_exceeds_out_von_morgen():
    kappa, r = 0.1, 0.05
    for deg in (15.0, 30.0, 60.0):
        alpha = math.radians(deg)
        phi = 0.5 * math.pi  # neutral axis
        cx = cablexsec_cycle_slip(kappa, r, alpha, phi, mu=0.0)
        bound = float(ovm_slip_magnitude(kappa, r, alpha, phi + 1.5 * math.pi, INSIDE))
        assert cx[2] / bound == pytest.approx(
            math.sin(alpha) / math.sqrt(1 + 4 * math.tan(alpha) ** 2), rel=1e-12
        )


def test_cell_transverse_slip_is_below_the_geodesic_and_targets_it():
    assert geodesic_attractor_residual(ALPHA, 0.05, 0.1) < 1e-12
    from check_out_von_morgen import cell_friction_sweep  # noqa: PLC0415

    rows = [
        row
        for row in cell_friction_sweep(factors=(1.0, 0.1), increments=80, elements=32)
        if row["status"] == "ok"
    ]
    assert len(rows) == 4
    for row in rows:
        assert 0.0 < row["pointwise_transverse_over_geodesic"] < 1.0
        assert 0.9 < row["longitudinal_over_loxodromic"] < 1.0
    for low, high in ((rows[1], rows[0]), (rows[3], rows[2])):
        assert low["transverse_over_geodesic"] > high["transverse_over_geodesic"]


def test_local_bending_coefficients_against_deformed_helix_curvature():
    for deg in (20.0, 35.0, 50.0):
        alpha = math.radians(deg)
        numeric = numerical_curvature_coefficients(alpha)
        closed = deformed_helix_curvature_coefficients(alpha)
        radial = float(local_bending_fibre_stress(1.0, 0.0, 0.0, 2.0, 1.0, alpha, 0.0))
        tangential = float(
            local_bending_fibre_stress(
                1.0, 0.0, -0.5 * math.pi, 2.0, 1.0, alpha, 0.5 * math.pi
            )
        )
        assert radial == pytest.approx(closed["eq41"], rel=1e-12)
        assert numeric["normal"] == pytest.approx(radial, abs=2e-4)
        assert numeric["binormal"] == pytest.approx(tangential, rel=1e-4)
        # Eq. 42 as printed in the source differs from the exact curvature.
        assert closed["eq42_printed"] / numeric["binormal"] > 1.03


def test_out_1989_effective_change_and_damage_ratio():
    assert out1989_effective_change(1.0, 0.2) == pytest.approx(0.6)
    assert out1989_effective_change(0.3, 0.2) == 0.0
    for c in (1.0, 2.0, 3.0):
        assert out1989_damage_ratio(1.0, c) == pytest.approx(1 / (c + 1))
        assert out1989_damage_ratio(1e6, c) == pytest.approx(1.0, rel=1e-5)
    checks = out1989_contact_checks(a_values=(0.5, 1.0, 2.5), points=400)
    assert checks["eq1_residual"] < 1e-12
    assert checks["damage_ratio_abs_error"] < 1e-4
