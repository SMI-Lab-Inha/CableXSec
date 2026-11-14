# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Gap G1: rectangular and profiled wires through the section paths.

Schema, contact geometry, stiffness, the solved radial route, stress
recovery at the corners and edges, the transverse model in the fatigue
field, and the identity of the lateral relief at zero demand (V2, V6).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from pydantic import ValidationError

import cablexsec as cx
from cablexsec.analysis.fatigue import wire_stress_field
from cablexsec.analysis.termination import TerminationSpec
from cablexsec.cable_adapters import cable_to_fatigue_layers
from cablexsec.cables.schema import HelicalLayer
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.along_wire import WireChainLayer, along_wire_history
from cablexsec.mechanics.concentric_section import ElasticHelicalRing
from cablexsec.mechanics.core_lay import bending_factor
from cablexsec.mechanics.radial_cable import solve_cable_radial_state
from cablexsec.mechanics.stiffness import _individual_bending_by_layer
from cablexsec.mechanics.tendon_bending import (
    LateralRelief,
    TransverseSlipModel,
    curvature_coefficients,
)
from cablexsec.mechanics.tendon_section import WireSection


def _strip(**extra):
    base = {
        "name": "a",
        "material": "steel_saevik2011",
        "n_wires": 55,
        "r_helix_m": 0.132,
        "lay_angle_deg": 24.8,
        "wire_shape": "rectangular",
        "wire_width_m": 0.0125,
        "wire_thickness_m": 0.005,
    }
    base.update(extra)
    return HelicalLayer(**base)


def test_schema_fills_the_equivalent_diameter_and_checks_it():
    layer = _strip()
    assert layer.d_wire_m == pytest.approx(math.sqrt(4 * 62.5e-6 / math.pi), rel=1e-15)
    assert layer.wire_radial_thickness_m == 0.005
    assert layer.wire_surface_width_m == 0.0125
    assert layer.wire_section.area == pytest.approx(62.5e-6, rel=1e-15)
    with pytest.raises(ValidationError):
        _strip(d_wire_m=0.009)
    with pytest.raises(ValidationError):
        HelicalLayer(name="a", material="steel", d_wire_m=0.004, wire_width_m=0.004)
    with pytest.raises(ValidationError):
        _strip(n_wires=70)  # fill > 1
    with pytest.raises(ValidationError):
        _strip(transverse_slip_fraction=1.5)
    round_layer = HelicalLayer(name="b", material="steel", d_wire_m=0.004)
    assert round_layer.wire_section == WireSection(shape="round", diameter_m=0.004)
    assert round_layer.wire_radial_thickness_m == 0.004


def test_packaged_strip_definitions_fit_their_layers():
    for path in (
        cx.saevik2011_yaml_path("rectangular"),
        cx.witz1996_yaml_path("rectangular"),
    ):
        cable = cx.load_cable(path, geometry_override=None)
        for layer in cable.helical_layers:
            if layer.wire_shape == "round":
                continue  # nominal near-hoop round helix (A3)
            section = layer.wire_section
            assert (
                section.fill_factor(layer.n_wires, layer.r_helix_m, layer.lay_angle_rad)
                <= 1.0
            )
    sv = cx.load_cable(cx.saevik2011_yaml_path("rectangular"), geometry_override=None)
    inner = sv.get_layer("inner_tensile_armour")
    assert (inner.wire_shape, inner.wire_width_m, inner.wire_thickness_m) == (
        "rectangular",
        0.0125,
        0.005,
    )
    # the strips fill the space between the tapes exactly
    assert inner.r_helix_m - 0.0025 == pytest.approx(
        sv.get_layer("antiwear_tape_inner").r_outer_m
    )
    assert inner.r_helix_m + 0.0025 == pytest.approx(
        sv.get_layer("antiwear_tape_middle").r_inner_m
    )


def test_individual_bending_uses_both_inertias():
    cable = cx.load_cable(
        cx.saevik2011_yaml_path("rectangular"), geometry_override=None
    )
    layer = cable.get_layer("inner_tensile_armour")
    a = layer.lay_angle_rad
    s = layer.wire_section
    c0 = curvature_coefficients(a, 0.0)
    expected = (
        55
        * 207e9
        * (s.I_lateral * c0.lateral**2 + s.I_normal * c0.normal**2)
        / (2 * math.cos(a))
    )
    assert _individual_bending_by_layer(cable)["inner_tensile_armour"] == pytest.approx(
        expected, rel=1e-13
    )
    # a square wire with equal inertias reduces to the round-wire factor
    i = s.I_normal
    assert (i * c0.lateral**2 + i * c0.normal**2) / (2 * math.cos(a)) == pytest.approx(
        i * bending_factor(a), rel=1e-14
    )
    # full-slip EI_min reproduces the published 72.8 kN m2 after identification
    assert cx.compute_stiffness(cable).ei_min_Nm2 == pytest.approx(72.8e3, rel=2e-4)


def test_ring_thickness_and_area():
    ring = ElasticHelicalRing(
        "r",
        0.132,
        0.0089206,
        55,
        0.43,
        207e9,
        radial_thickness_m=0.005,
        wire_area_m2=62.5e-6,
    )
    assert (ring.inner_radius_m, ring.outer_radius_m) == pytest.approx((0.1295, 0.1345))
    assert ring.total_area_m2 == pytest.approx(55 * 62.5e-6)
    plain = ElasticHelicalRing("r", 0.132, 0.004, 10, 0.43, 207e9)
    assert plain.total_area_m2 == 10 * math.pi * 0.004**2 / 4
    assert plain.outer_radius_m == 0.132 + 0.004 / 2


def test_solved_route_runs_on_the_true_strips():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rect = cx.load_cable(
            cx.witz1996_yaml_path("rectangular"), geometry_override=None
        )
        state = solve_cable_radial_state(
            rect,
            116.1e3,
            internal_pressure_Pa=30e6,
            external_pressure_Pa=1.0e6,
            pressure_barrier="pressure_sheath",
        )
        pressures = state.armour_contact_pressure()
        assert pressures["inner_tensile_armour"] > pressures["outer_tensile_armour"] > 0
        round_ = cx.load_cable(cx.witz1996_yaml_path(), geometry_override=None)
        with pytest.raises(Exception, match="overlap"):
            solve_cable_radial_state(round_, 0.0, external_pressure_Pa=1.0e6)


@pytest.fixture(scope="module")
def strip_layer():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cable = cx.load_cable(
            cx.saevik2011_yaml_path("rectangular"), geometry_override=None
        )
        state = cx.solve_section_state(
            cable, tension_N=1.737e6, p_int_Pa=34e6, contact_model="bounded"
        )
        layers = cable_to_fatigue_layers(
            cable, mu=0.2, contact_model="bounded", section_state=state
        )
    return next(x for x in layers if x.name == "inner_tensile_armour")


def test_strip_stress_at_faces_edges_and_corners(strip_layer):
    spec = strip_layer.spec
    assert spec.section is not None and spec.section.shape == "rectangular"
    kx = np.array([0.0, 0.05, 0.1])
    ky = np.zeros(3)
    field = wire_stress_field(strip_layer, kx, ky)
    c = curvature_coefficients(spec.lay_angle_rad)
    phi = spec.phi_wires[:, None]
    e = spec.E_wire_Pa
    assert np.allclose(
        field.radial_Pa, e * c.normal * kx * np.cos(phi) * 0.0025, rtol=1e-12, atol=1e-6
    )
    assert np.allclose(
        field.tangential_Pa,
        e * c.lateral * kx * -np.sin(phi) * 0.00625,
        rtol=1e-12,
        atol=1e-6,
    )
    corner = math.atan2(0.0125, 0.005)
    assert np.allclose(
        field.at_fibre(corner), field.mean_Pa + field.radial_Pa + field.tangential_Pa
    )
    assert np.allclose(
        field.at_point(0.0, 0.00525),
        field.mean_Pa + field.tangential_Pa * 0.00525 / 0.00625,
    )


def test_transverse_model_in_the_field_reaches_the_geodesic(strip_layer):
    spec = strip_layer.spec
    section = WireSection(shape="rectangular", width_m=0.0125, thickness_m=1e-9)
    model = TransverseSlipModel(
        section, spec.E_wire_Pa, 1e-9, spec.r_helix_m, spec.lay_angle_rad, 1.7e4, 0.0
    )
    assert model.frictionless_fraction == pytest.approx(1.0, abs=1e-12)
    kx = np.linspace(0.0, 0.1, 21)
    field = wire_stress_field(strip_layer, kx, np.zeros_like(kx), transverse=model)
    plain = wire_stress_field(strip_layer, kx, np.zeros_like(kx))
    assert np.allclose(field.tangential_Pa, 0.0, atol=1e-3)
    geo = curvature_coefficients(spec.lay_angle_rad, 1.0).normal
    lox = curvature_coefficients(spec.lay_angle_rad, 0.0).normal
    # the normal curvature gains the tangent-rotation term (finite difference
    # over the 55 wire azimuths)
    assert np.allclose(
        field.radial_Pa, plain.radial_Pa * geo / lox, rtol=3e-3, atol=1e-3
    )
    assert np.array_equal(field.mean_Pa, plain.mean_Pa)


def test_zero_demand_relief_is_bitwise_identity(strip_layer):
    spec = strip_layer.spec
    model = TransverseSlipModel(
        spec.section,
        spec.E_wire_Pa,
        spec.E_wire_Pa / 2.6,
        spec.r_helix_m,
        spec.lay_angle_rad,
        1.7e4,
        1e40,
    )
    relief = LateralRelief(model)
    kx = 0.1 * np.sin(np.linspace(0, 3 * math.pi, 61))
    ky = np.zeros_like(kx)
    a = wire_stress_field(strip_layer, kx, ky, friction_model="distributed")
    b = wire_stress_field(
        strip_layer, kx, ky, friction_model="distributed", relief=relief
    )
    assert np.array_equal(a.mean_Pa, b.mean_Pa)
    with pytest.raises(CableInputError):
        wire_stress_field(strip_layer, kx, ky, relief=relief)
    term = TerminationSpec(distance_m=1.0, solver="along_wire", stick="scaled")
    t_a = wire_stress_field(strip_layer, kx, ky, termination=term)
    t_b = wire_stress_field(
        strip_layer,
        kx,
        ky,
        termination=TerminationSpec(
            distance_m=1.0, solver="along_wire", stick="scaled", lateral_relief=relief
        ),
    )
    assert np.array_equal(t_a.mean_Pa, t_b.mean_Pa)
    with pytest.raises(CableInputError):
        TerminationSpec(distance_m=1.0, lateral_relief=relief)
    lay = WireChainLayer.from_layer_spec(spec, stick="spring", capstan=True)
    x = np.linspace(0.0, 4.0, 9)
    field = np.outer(np.linspace(0.5, 1.0, 9), kx)
    h1 = along_wire_history(lay, x, field, record_x_m=np.array([2.0]))
    h2 = along_wire_history(lay, x, field, record_x_m=np.array([2.0]), relief=relief)
    assert np.array_equal(h1.stress_Pa, h2.stress_Pa)


def test_relief_lowers_the_distributed_friction(strip_layer):
    spec = strip_layer.spec
    model = TransverseSlipModel(
        spec.section,
        spec.E_wire_Pa,
        spec.E_wire_Pa / 2.6,
        spec.r_helix_m,
        spec.lay_angle_rad,
        1.7e4,
        2.0e3,
    )
    kx = 0.15 * np.sin(np.linspace(0, 3 * math.pi, 61))
    ky = np.zeros_like(kx)
    a = wire_stress_field(strip_layer, kx, ky, friction_model="distributed")
    b = wire_stress_field(
        strip_layer, kx, ky, friction_model="distributed", relief=LateralRelief(model)
    )
    assert np.ptp(b.mean_Pa[0]) < np.ptp(a.mean_Pa[0])


def test_capacity_and_stress_paths_accept_strips():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cable = cx.load_cable(
            cx.witz1996_yaml_path("rectangular"), geometry_override=None
        )
        result = cx.compute_stress(cable, 0.2, mu=0.1)
    hi = result.sigma_perimeter_max_Pa["inner_tensile_armour"]
    lo = result.sigma_perimeter_min_Pa["inner_tensile_armour"]
    assert np.all(np.isfinite(hi)) and np.all(hi >= lo)
