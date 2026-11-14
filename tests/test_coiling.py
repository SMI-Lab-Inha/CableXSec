# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Coiling into a static tank.

Verification of :func:`cablexsec.compute_coiling` on the packaged cable of
Chae, Nam and Shim (2023): tank-to-load mapping, twist sense, closed-form
limits (pure twist, frictionless rotating bending), torque control, the
capacity formulas and the input checks (doc/benchmarks.md, CH-CR).
"""

from __future__ import annotations

import math

import pytest

import cablexsec as cx
from cablexsec.analysis.capacity import birdcage_buckling_detail
from cablexsec.exceptions import CableInputError
from cablexsec.materials import library


@pytest.fixture(scope="module")
def cable() -> cx.CableData:
    return cx.load_cable(cx.chae2023_yaml_path(), geometry_override=None)


def _armour(cable: cx.CableData):  # type: ignore[no-untyped-def]
    (layer,) = [x for x in cable.helical_layers if x.name == "armour"]
    return layer


def test_packaged_cable_geometry(cable: cx.CableData) -> None:
    layer = _armour(cable)
    assert layer.n_wires == 94
    assert layer.lay_direction == "left_hand"
    pitch = 2.0 * math.pi * layer.r_helix_m / math.tan(layer.lay_angle_rad)
    assert pitch == pytest.approx(1.8, abs=1e-4)


def test_tank_diameter_sets_curvature_and_twist(cable: cx.CableData) -> None:
    res = cx.compute_coiling(cable, tank_diameter_m=14.4, n_per_turn=36)
    assert res.curvature_per_m == pytest.approx(2.0 / 14.4)
    # Left-hand armour: the unwinding twist is positive (package convention).
    assert res.twist_per_m == pytest.approx(2.0 / 14.4)
    assert res.tank_diameter_m == 14.4
    assert res.provenance.citations[0] == "[ChaeNamShim2023]"


def test_unwinding_compresses_and_tightening_stretches(cable: cx.CableData) -> None:
    kw = {"curvature_per_m": 0.0, "twist_per_m": 0.05, "n_per_turn": 8}
    unwind = cx.compute_coiling(cable, **kw).layers["armour"]
    tighten = cx.compute_coiling(cable, twist_sense="tighten", **kw).layers["armour"]
    assert unwind.axisymmetric_Pa < 0.0 < tighten.axisymmetric_Pa
    signed = cx.compute_coiling(
        cable,
        curvature_per_m=0.0,
        twist_per_m=-0.05,
        twist_sense="signed",
        n_per_turn=8,
    )
    assert signed.layers["armour"].axisymmetric_Pa == pytest.approx(
        tighten.axisymmetric_Pa, rel=1e-12
    )


def test_pure_twist_has_no_bending_or_friction(cable: cx.CableData) -> None:
    res = cx.compute_coiling(cable, curvature_per_m=0.0, twist_per_m=0.1, n_per_turn=8)
    layer = res.layers["armour"]
    assert layer.friction_max_Pa == 0.0
    for value in (
        layer.centre_min_Pa,
        layer.centre_max_Pa,
        layer.fibre_min_Pa,
        layer.fibre_max_Pa,
    ):
        assert value == pytest.approx(layer.axisymmetric_Pa, rel=1e-12)
    # The compressed armour lifts off the bundle and bears on the outer yarn
    # (whose round-off gap of 1e-17 m once read as open); the cable lengthens.
    assert res.open_interfaces == ("bundle/armour",)
    assert layer.contact_pressure_Pa > 0.0
    assert res.axial_strain > 0.0


def test_frictionless_rotating_bending_is_local_bending(cable: cx.CableData) -> None:
    kappa = 0.14
    res = cx.compute_coiling(cable, curvature_per_m=kappa, twist_per_m=0.0, mu=0.0)
    layer = res.layers["armour"]
    armour = _armour(cable)
    a = armour.lay_angle_rad
    e = library.get_property(armour.material, "E_Pa")
    amplitude = (
        e * kappa * armour.d_wire_m / 2.0 * math.cos(a) * (1.0 + math.sin(a) ** 2)
    )
    assert layer.axisymmetric_Pa == pytest.approx(0.0, abs=1e-6)
    assert layer.friction_max_Pa == 0.0
    assert layer.fibre_min_Pa == pytest.approx(-amplitude, rel=1e-3)
    assert layer.fibre_max_Pa == pytest.approx(amplitude, rel=1e-3)


def test_torque_control_round_trip(cable: cx.CableData) -> None:
    by_twist = cx.compute_coiling(
        cable, curvature_per_m=0.0, twist_per_m=0.08, n_per_turn=8
    )
    by_torque = cx.compute_coiling(
        cable, curvature_per_m=0.0, torque_Nm=by_twist.torque_Nm, n_per_turn=8
    )
    assert by_torque.twist_per_m == pytest.approx(by_twist.twist_per_m, rel=1e-6)
    assert by_torque.layers["armour"].axisymmetric_Pa == pytest.approx(
        by_twist.layers["armour"].axisymmetric_Pa, rel=1e-6
    )


def test_capacities_and_utilisation(cable: cx.CableData) -> None:
    res = cx.compute_coiling(cable, tank_diameter_m=14.4, n_per_turn=36)
    layer = res.layers["armour"]
    armour = _armour(cable)
    # Saevik-Thorsen capacities are cable axial forces P = n cos(a) Q (Eqs. 32,
    # 34): the wire stress at a capacity is P/(n A cos a).
    force = armour.n_wires * math.pi * armour.d_wire_m**2 / 4.0
    force *= math.cos(armour.lay_angle_rad)
    detail = birdcage_buckling_detail(cable, armour)
    assert layer.sigma_c1_Pa == pytest.approx(detail.P2_radial_N / force, rel=1e-12)
    assert layer.sigma_lateral_Pa == pytest.approx(
        detail.P1_lateral_N / force, rel=1e-12
    )
    assert layer.sigma_c2_Pa == 400.0e6  # P4/(n A cos a) = sigma_y (Eq. 34)
    assert 1.0 / layer.sigma_cr_Pa == pytest.approx(
        1.0 / layer.sigma_c1_Pa + 1.0 / layer.sigma_c2_Pa, rel=1e-12
    )
    # Radial buckling with the yield interaction is the capacity; the
    # friction-free lateral bound is reported, far below it, and not used.
    assert layer.capacity_Pa == layer.sigma_cr_Pa
    assert layer.sigma_lateral_Pa < 0.1 * layer.sigma_cr_Pa
    assert layer.lateral_utilisation > 10.0
    # Demand: wire-centre stress for buckling, surface stress for yield.
    assert layer.buckling_utilisation == pytest.approx(
        -layer.centre_min_Pa / layer.sigma_cr_Pa, rel=1e-12
    )
    assert layer.yield_utilisation == pytest.approx(
        max(-layer.fibre_min_Pa, layer.fibre_max_Pa) / 400.0e6, rel=1e-12
    )
    assert layer.utilisation == max(layer.buckling_utilisation, layer.yield_utilisation)
    assert layer.governing_mode == "radial"
    assert res.utilisation == layer.utilisation


def test_radial_capacity_is_chae_table6_in_wire_stress() -> None:
    # Chae, Nam and Shim (2023) Table 6 at pitch 1.8 m: alpha 0.267, c 4.84 MPa,
    # sigma_c1 535.4 MPa, sigma_c2 385.8 MPa, from their Eqs. 17-18, which
    # divide P by n A. In wire stress (P/(n A cos a)): 2 sqrt(EI c)/A and
    # sigma_y. Hand values: EI = 200 GPa pi (5 mm)^4/64 = 6.136 N m2,
    # 2 sqrt(6.136 x 4.84e6)/19.635 mm2 = 555.1 MPa.
    from cablexsec.analysis.compression import (
        radial_buckling_load_straight,
        round_wire_section,
        wire_yield_load,
    )

    alpha, n, area = 0.267, 94, math.pi * 0.005**2 / 4.0
    section = round_wire_section(200.0e9, 0.30, 0.005)
    p2 = radial_buckling_load_straight(n, alpha, section, 4.84e6)
    p4 = wire_yield_load(n, alpha, 400.0e6, area)
    wire = n * area * math.cos(alpha)
    assert p2 / wire == pytest.approx(555.1e6, rel=2e-4)
    assert p2 / wire == pytest.approx(535.4e6 / math.cos(alpha), rel=1e-3)
    assert p4 / wire == pytest.approx(400.0e6, rel=1e-12)
    assert p4 / (n * area) == pytest.approx(385.8e6, rel=1e-3)  # source form


def test_twisted_section_matches_helix_kinematics(cable: cx.CableData) -> None:
    # Wire strain eps cos^2 a + r tau sin a cos a + (u/r) sin^2 a, from the
    # solved axial strain, twist and helix-radius displacement.
    from cablexsec.mechanics.radial_cable import solve_cable_radial_state

    armour = _armour(cable)
    tau = 2.0 / 10.0
    res = cx.compute_coiling(
        cable, curvature_per_m=0.0, twist_per_m=tau, n_per_turn=8
    ).layers["armour"]
    state = solve_cable_radial_state(
        cable, 0.0, prescribed_twist_per_m=tau, loose_layers="neighbour"
    ).result
    # Left-hand lay: the signed lay angle is negative, so the positive
    # (unwinding) twist compresses the wire.
    a, r = -armour.lay_angle_rad, armour.r_helix_m
    u = state.inner_displacement_m["armour"]
    assert u > 0.0  # the compressed armour lifts off the bundle
    strain = (
        state.axial_strain * math.cos(a) ** 2
        + r * tau * math.sin(a) * math.cos(a)
        + u / r * math.sin(a) ** 2
    )
    e = library.get_property(armour.material, "E_Pa")
    assert res.axisymmetric_Pa == pytest.approx(e * strain, rel=1e-9)
    # The twist term alone is -E r tau sin a cos a, with tan a = 2 pi 0.081/1.8:
    # 200 GPa x 0.081 x 0.2 x 0.26181 = -848.3 MPa; the free extension and the
    # radial lift-off relieve it to about -325 MPa.
    twist_only = e * r * tau * math.sin(-a) * math.cos(a)
    assert twist_only == pytest.approx(848.3e6, rel=2e-4)
    assert -0.45 * twist_only < res.axisymmetric_Pa < -0.35 * twist_only


def test_yield_validity_is_reported(cable: cx.CableData) -> None:
    small = cx.compute_coiling(cable, tank_diameter_m=10.0, n_per_turn=36)
    layer = small.layers["armour"]
    assert layer.yield_utilisation > 1.0 and not layer.elastic_valid
    assert not small.elastic_valid
    assert any("beyond first yield" in x for x in small.provenance.limitations)
    large = cx.compute_coiling(cable, tank_diameter_m=18.0, n_per_turn=36)
    assert large.elastic_valid and large.layers["armour"].yield_utilisation < 1.0
    assert not any("yield" in x.split(":")[0] for x in large.provenance.limitations)
    assert not any("counter-helical" in x for x in large.provenance.limitations)


def test_flexible_pipe_is_refused() -> None:
    from cablexsec.analysis.coiling import flexible_pipe_layer

    pipe = cx.load_cable(cx.saevik2011_yaml_path(), geometry_override=None)
    layer = flexible_pipe_layer(pipe)
    assert layer is not None and layer.name == "pressure_armour"
    with pytest.raises(CableInputError, match="unbonded flexible pipe"):
        cx.compute_coiling(pipe, tank_diameter_m=10.0)


def test_counter_helical_armour_carries_the_caveat() -> None:
    from cablexsec.analysis.coiling import counter_helical_armour, flexible_pipe_layer

    iea = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
    assert counter_helical_armour(iea) and flexible_pipe_layer(iea) is None
    res = cx.compute_coiling(iea, tank_diameter_m=10.0, n_per_turn=36)
    assert any("non-coilable" in x for x in res.provenance.limitations)
    # Unwinding the right-hand outer armour tightens the left-hand inner one:
    # r tau sin a cos a E = 718 MPa of tension in the inner wires (13 deg),
    # beyond the 375 MPa yield stress.
    inner = res.layers["inner_armour"]
    assert inner.axisymmetric_Pa == pytest.approx(
        200e9
        * 0.081925
        * 0.2
        * math.sin(math.radians(13))
        * math.cos(math.radians(13)),
        rel=0.1,
    )
    assert inner.governing_mode == "yield" and not inner.elastic_valid
    assert res.layers["outer_armour"].buckling_utilisation > 1.0


def test_smaller_tank_is_more_severe(cable: cx.CableData) -> None:
    stresses = [
        cx.compute_coiling(cable, tank_diameter_m=d, n_per_turn=36)
        .layers["armour"]
        .fibre_min_Pa
        for d in (10.0, 14.4, 18.0)
    ]
    assert stresses[0] < stresses[1] < stresses[2] < 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"tank_diameter_m": -1.0},
        {"tank_diameter_m": 0.0},
        {"curvature_per_m": 0.1},
        {"curvature_per_m": -0.1, "twist_per_m": 0.1},
        {"tank_diameter_m": 10.0, "twist_per_m": 0.1, "torque_Nm": 1.0},
        {"tank_diameter_m": 10.0, "twist_per_m": -0.1},
        {"tank_diameter_m": 10.0, "twist_sense": "sideways"},
        {"tank_diameter_m": 10.0, "tension_N": -1.0},
        {"tank_diameter_m": 10.0, "turns": 0.0},
        {"tank_diameter_m": 10.0, "n_fibres": 0},
    ],
)
def test_input_errors(cable: cx.CableData, kwargs: dict) -> None:  # type: ignore[type-arg]
    with pytest.raises(CableInputError):
        cx.compute_coiling(cable, **kwargs)
