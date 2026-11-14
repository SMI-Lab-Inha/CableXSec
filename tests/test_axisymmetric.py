# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for the axisymmetric contact chain (R3.1): gates A2 (capstan) & A6 (Lamé)."""

from __future__ import annotations

import math

import pytest

from cablexsec.cables.schema import CableData, CableMeta, HelicalLayer
from cablexsec.mechanics.axisymmetric import (
    capstan_force_per_wire,
    compute_contact_forces,
    compute_winding_contact_pressure,
    lame_thick_wall,
)

# Test-taxonomy category.
pytestmark = pytest.mark.analytical_sanity


def _single_layer_cable() -> CableData:
    armour = HelicalLayer(
        name="armour",
        material="galvanised_steel_armour",
        n_wires=40,
        d_wire_m=0.005,
        r_helix_m=0.080,
        lay_angle_deg=20.0,
    )
    return CableData(cable=CableMeta(name="one-layer"), layers=(armour,))


# --------------------------------------------------------------------------- #
# A2 — capstan limit                                                          #
# --------------------------------------------------------------------------- #
def test_A2_capstan_single_layer_rigid_foundation() -> None:
    """Radial load per axial metre from tension-vector differentiation."""
    cable = _single_layer_cable()
    T = 50.0e3
    res = compute_contact_forces(cable, T, layer_tension_N={"armour": T})
    expected = T * math.tan(math.radians(20.0)) ** 2 / (40 * 0.080)
    assert res.Fn_per_wire_N["armour"] == pytest.approx(expected, rel=1e-10)


def test_capstan_formula_and_pretension() -> None:
    """Pretension P0 [Pa] times tributary width gives line load [N/m]."""
    d_w, n, r = 0.005, 40, 0.08
    P0 = 2.0e5  # Pa
    f = capstan_force_per_wire(
        10.0e3, math.radians(20.0), n, r, d_wire_m=d_w, pretension_Pa=P0
    )
    spacing = 2.0 * math.pi * r / n
    expected = 10.0e3 * math.tan(math.radians(20.0)) ** 2 / (n * r) + P0 * spacing
    assert f == pytest.approx(expected)


def test_capstan_scales_linearly_with_tension() -> None:
    cable = _single_layer_cable()
    f1 = compute_contact_forces(cable, 0.0, layer_tension_N={"armour": 10e3})
    f2 = compute_contact_forces(cable, 0.0, layer_tension_N={"armour": 20e3})
    assert f2.Fn_per_wire_N["armour"] == pytest.approx(2.0 * f1.Fn_per_wire_N["armour"])


# --------------------------------------------------------------------------- #
# A6 — Lamé thick-wall cylinder                                               #
# --------------------------------------------------------------------------- #
def test_A6_lame_internal_pressure_boundary_stresses() -> None:
    """A6: internal pressure only -> classic Lamé boundary values, < 0.1%."""
    ri, ro, p = 0.05, 0.10, 10.0e6
    E, nu = 210.0e9, 0.30
    inner = lame_thick_wall(ri, ri, ro, p, 0.0, E, nu)
    outer = lame_thick_wall(ro, ri, ro, p, 0.0, E, nu)
    # sigma_r(ri) = -p ; sigma_r(ro) = 0.
    assert inner.sigma_r_Pa == pytest.approx(-p, rel=1e-3)
    assert outer.sigma_r_Pa == pytest.approx(0.0, abs=1e-3 * p)
    # Hoop at bore: p (ro^2 + ri^2)/(ro^2 - ri^2).
    hoop_in = p * (ro**2 + ri**2) / (ro**2 - ri**2)
    assert inner.sigma_theta_Pa == pytest.approx(hoop_in, rel=1e-3)
    # Hoop at outer: 2 p ri^2/(ro^2 - ri^2).
    hoop_out = 2.0 * p * ri**2 / (ro**2 - ri**2)
    assert outer.sigma_theta_Pa == pytest.approx(hoop_out, rel=1e-3)


def test_A6_lame_external_pressure() -> None:
    ri, ro, p = 0.04, 0.09, 8.0e6
    E, nu = 70.0e9, 0.33
    inner = lame_thick_wall(ri, ri, ro, 0.0, p, E, nu)
    outer = lame_thick_wall(ro, ri, ro, 0.0, p, E, nu)
    assert outer.sigma_r_Pa == pytest.approx(-p, rel=1e-3)
    assert inner.sigma_r_Pa == pytest.approx(0.0, abs=1e-3 * p)
    # Bore hoop under external pressure: -2 p ro^2/(ro^2 - ri^2).
    assert inner.sigma_theta_Pa == pytest.approx(
        -2.0 * p * ro**2 / (ro**2 - ri**2), rel=1e-3
    )


def test_A6_lame_equilibrium_sum() -> None:
    """sigma_r + sigma_theta = 2A is constant through the wall (Lamé)."""
    ri, ro, p = 0.05, 0.10, 10.0e6
    s1 = lame_thick_wall(0.06, ri, ro, p, 0.0, 210e9, 0.3)
    s2 = lame_thick_wall(0.09, ri, ro, p, 0.0, 210e9, 0.3)
    assert (s1.sigma_r_Pa + s1.sigma_theta_Pa) == pytest.approx(
        s2.sigma_r_Pa + s2.sigma_theta_Pa, rel=1e-9
    )


# --------------------------------------------------------------------------- #
# Multi-layer                                                                 #
# --------------------------------------------------------------------------- #
def test_external_pressure_increases_inner_Fn() -> None:
    cable = _single_layer_cable()
    base = compute_contact_forces(cable, 0.0, layer_tension_N={"armour": 10e3})
    withp = compute_contact_forces(
        cable, 0.0, p_ext_Pa=5.0e6, layer_tension_N={"armour": 10e3}
    )
    assert withp.Fn_per_wire_N["armour"] > base.Fn_per_wire_N["armour"]


# --------------------------------------------------------------------------- #
# Winding-stress -> contact pressure (Feret-Bournazel Eq. 5 bridge)            #
# --------------------------------------------------------------------------- #
def test_winding_pressure_matches_feret_bournazel_eq5() -> None:
    """Pressure jump includes sin(a) tan(a), Féret--Bournazel (1987) Eq. 5."""
    cable = _single_layer_cable()
    sigma = 80.0e6
    p = compute_winding_contact_pressure(cable, sigma)["armour"]
    n, d_w, r, a = 40, 0.005, 0.080, math.radians(20.0)
    area = math.pi * d_w**2 / 4.0
    eq5 = n * sigma * area * math.sin(a) * math.tan(a) / (2.0 * math.pi * r**2)
    # outermost single layer has no layer above (p_above = 0), so P1+P2 = Eq. 5.
    assert p == pytest.approx(eq5, rel=1e-9)


@pytest.mark.parametrize("angle_deg", [5.0, 20.0, 45.0, 70.0])
def test_helix_line_load_matches_differentiated_tension_vector(angle_deg):
    """Independent centred difference of N_w t(z), with axial-length units."""
    import numpy as np

    angle = math.radians(angle_deg)
    radius, wire_tension = 0.08, 1000.0
    omega = math.tan(angle) / radius
    dz = 1.0e-7 / omega

    def tangent(z):
        return np.array(
            [
                -math.sin(angle) * math.sin(omega * z),
                math.sin(angle) * math.cos(omega * z),
                math.cos(angle),
            ]
        )

    radial_load = -wire_tension * (tangent(dz) - tangent(-dz))[0] / (2 * dz)
    calculated = capstan_force_per_wire(
        wire_tension * math.cos(angle), angle, 1, radius
    )
    assert calculated == pytest.approx(radial_load, rel=1.0e-12)
    assert (
        capstan_force_per_wire(wire_tension * math.cos(angle), -angle, 1, radius)
        == calculated
    )


def test_winding_pressure_linear_and_zero() -> None:
    cable = _single_layer_cable()
    assert compute_winding_contact_pressure(cable, 0.0)["armour"] == pytest.approx(0.0)
    p1 = compute_winding_contact_pressure(cable, 50.0e6)["armour"]
    p2 = compute_winding_contact_pressure(cable, 100.0e6)["armour"]
    assert p2 == pytest.approx(2.0 * p1, rel=1e-9)


def test_contact_load_sharing_matches_section_axial_stiffness():
    from cablexsec import load_cable, menard2023_yaml_path
    from cablexsec.mechanics.stiffness import load_sharing

    cable = load_cable(menard2023_yaml_path(), geometry_override=None)
    result = compute_contact_forces(cable, 1.0e5)
    expected = load_sharing(cable, 1.0e5)
    assert result.layer_tension_N == pytest.approx(expected)
    assert sum(result.layer_tension_N.values()) == pytest.approx(1.0e5)


def test_pressure_increment_retains_both_armour_faces_and_has_pressure_units():
    from cablexsec import load_cable, menard2023_yaml_path
    from cablexsec.cable_adapters import armour_contact_pressure

    cable = load_cable(menard2023_yaml_path(), geometry_override=None)
    baseline = armour_contact_pressure(cable, 0.0)
    updated = armour_contact_pressure(cable, 1.0e5)
    force0 = compute_contact_forces(cable, 0.0)
    force1 = compute_contact_forces(cable, 1.0e5)
    jump = force1.interface_pressure_Pa - force0.interface_pressure_Pa
    layers = sorted(cable.helical_layers, key=lambda layer: layer.r_helix_m)
    for i, layer in enumerate(layers):
        expected = jump[i] + (jump[i + 1] if i + 1 < len(jump) else 0.0)
        assert updated[layer.name] - baseline[layer.name] == pytest.approx(
            expected, rel=1e-12
        )
