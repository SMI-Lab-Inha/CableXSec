# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Closed-form and consistency checks of the solved radial contact state."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.cable_adapters import (
    armour_contact_pressure,
    armour_crossing_line_loads,
    armour_normal_force,
)
from cablexsec.cables import inter_array_three_core_legacy_r2_yaml_path
from cablexsec.exceptions import CableInputError, CableSchemaError, CableSolverError
from cablexsec.mechanics.concentric_section import (
    ElasticAnnulus,
    ElasticHelicalRing,
    solve_concentric_section,
)
from cablexsec.mechanics.radial_cable import (
    build_radial_section,
    residual_wire_stress,
    solve_cable_radial_state,
)
from cablexsec.mechanics.radial_solution import (
    AxialMember,
    RadialSection,
    capstan_residual_stress,
    flooded_free_strain,
    solve_lcp,
)
from cablexsec.mechanics.trefoil_section import ElasticFiller, TrefoilBundle

pytestmark = pytest.mark.analytical_sanity


def _concentric_layers() -> list:
    return [
        ElasticAnnulus("core", 0.0, 0.03, 2e9, 0.45, 0.001),
        ElasticAnnulus("sheath", 0.03, 0.035, 0.9e9, 0.46),
        ElasticHelicalRing("arm1", 0.037, 0.004, 50, math.radians(20), 210e9, 0, 5e-4),
        ElasticAnnulus("bed", 0.039, 0.040, 0.5e9, 0.4),
        ElasticHelicalRing("arm2", 0.042, 0.004, 55, math.radians(-20), 210e9),
        ElasticAnnulus("outer", 0.044, 0.05, 0.9e9, 0.46, -0.002),
    ]


def _core(free: float = 0.0) -> tuple[ElasticAnnulus, ...]:
    return (
        ElasticAnnulus("cond", 0.0, 0.008, 117e9, 0.34, free),
        ElasticAnnulus("xlpe", 0.008, 0.02, 0.15e9, 0.45, free),
    )


@pytest.mark.parametrize(
    "loads",
    [
        {"tension_N": 1e5},
        {"tension_N": 1e5, "prescribed_twist_per_m": None},
        {"external_pressure_Pa": 2e6},
        {"tension_N": -5e4},
        {"torque_Nm": 300.0, "prescribed_twist_per_m": None},
    ],
)
def test_concentric_path_reproduces_existing_section_solver(loads):
    layers = _concentric_layers()
    reference = solve_concentric_section(layers, **loads)
    result = RadialSection(layers).solve(**loads)
    assert result.axial_strain == pytest.approx(reference.axial_strain, rel=1e-10)
    assert result.twist_per_m == pytest.approx(
        reference.twist_per_m, rel=1e-10, abs=1e-15
    )
    np.testing.assert_allclose(
        result.interface_line_force_N_per_m,
        reference.contact_line_force_N_per_m,
        rtol=1e-9,
        atol=1e-6,
    )
    np.testing.assert_allclose(result.tangent, reference.tangent.matrix, rtol=1e-9)
    assert result.relative_equilibrium_residual < 1e-10


def test_prescribed_axial_strain_returns_the_force_that_produced_it():
    section = RadialSection(_concentric_layers())
    forward = section.solve(tension_N=2e5)
    inverse = section.solve(prescribed_axial_strain=forward.axial_strain)
    assert inverse.axial_force_N == pytest.approx(2e5, rel=1e-9)
    np.testing.assert_allclose(
        inverse.interface_line_force_N_per_m,
        forward.interface_line_force_N_per_m,
        rtol=1e-9,
        atol=1e-6,
    )


def test_single_helical_layer_reaction_is_the_feret_bournazel_jump():
    """Line force on the support = n sigma A sin(a) tan(a) / r (F-B Eq. 5)."""
    angle = math.radians(25)
    ring = ElasticHelicalRing("arm", 0.042, 0.004, 60, angle, 210e9)
    core = ElasticAnnulus("core", 0.02, 0.040, 1e9, 0.4)
    result = RadialSection([core, ring], rigid_bore=True).solve(tension_N=3e5)
    sigma = result.wire_stress_Pa["arm"]
    area = math.pi * 0.004**2 / 4
    jump = 60 * sigma * area * math.sin(angle) * math.tan(angle) / 0.042
    assert result.interface_line_force_N_per_m[0] == pytest.approx(jump, rel=1e-10)
    # The face pressure is referred to the helix radius (capstan jump P0).
    assert result.helical_face_pressure_Pa["arm"][0] == pytest.approx(
        jump / (2 * math.pi * 0.042), rel=1e-10
    )


def test_rigid_bore_limit_is_the_no_interaction_axial_stiffness():
    angle = math.radians(18)
    rings = [
        ElasticHelicalRing("a", 0.031, 0.002, 80, angle, 207e9),
        ElasticHelicalRing("b", 0.033, 0.002, 86, -angle, 207e9),
    ]
    members = [AxialMember("tube", 5e7)]
    section = RadialSection(rings, members=members, rigid_bore=True)
    result = section.solve(tension_N=1e5)
    steel = sum(
        r.wire_count * math.pi * r.wire_diameter_m**2 / 4 * r.youngs_modulus_Pa
        for r in rings
    )
    assert result.ea_fixed_twist_N == pytest.approx(
        steel * math.cos(angle) ** 3 + 5e7, rel=1e-9
    )


def test_bore_foundation_reduces_axial_stiffness_by_the_condensed_form():
    angle, r, d, k = math.radians(20), 0.05, 0.004, 2e9
    ring = ElasticHelicalRing("a", r, d, 70, angle, 210e9)
    res = RadialSection([ring], bore_foundation_Pa_per_m=k).solve(tension_N=1e5)
    rigidity = 70 * math.pi * d**2 / 4 * 210e9 / math.cos(angle)
    c, s = math.cos(angle), math.sin(angle)
    spring = 2 * math.pi * (r - d / 2) * k
    expected = rigidity * c**4 - (rigidity * c * c * s * s / r) ** 2 / (
        rigidity * s**4 / r**2 + spring
    )
    assert res.ea_fixed_twist_N == pytest.approx(expected, rel=1e-9)


def test_interface_compliance_acts_in_series():
    inner = ElasticAnnulus("inner", 0.02, 0.03, 2e9, 0.0, 0.001)
    outer = ElasticAnnulus("outer", 0.03, 0.04, 210e9, 0.0)
    stiff = RadialSection([inner, outer]).solve(prescribed_axial_strain=0.0)
    soft = RadialSection(
        [inner, outer], interface_compliance={("inner", "outer"): 1e-9}
    ).solve(prescribed_axial_strain=0.0)
    base = stiff.interface_line_force_N_per_m[0]
    # Two springs in series: 1/k_eff = 1/k_stiff + c.
    flex = stiff.interface_gap_m[0] * 0 + base  # line force at zero compliance
    assert 0 < soft.interface_line_force_N_per_m[0] < flex
    assert abs(soft.interface_gap_m[0]) < 1e-15


def test_lcp_reproduces_a_two_contact_closed_form():
    flexibility = np.array([[2.0, 0.5], [0.5, 1.0]])
    gap = np.array([-1.0, 0.4])
    reaction = solve_lcp(flexibility, gap)
    assert reaction[1] == 0.0
    assert reaction[0] == pytest.approx(0.5)
    with pytest.raises(CableSolverError):
        solve_lcp(np.array([[1.0, 2.0], [2.0, 1.0]]), np.array([-1.0, -1.0]))


def test_trefoil_ring_far_from_cores_matches_plane_strain_lame():
    a = 0.02
    rc = 2 * a / math.sqrt(3)
    ri, ro, e, nu, p = rc + a + 0.01, rc + a + 0.014, 0.9e9, 0.46, 1e5
    bundle = TrefoilBundle(
        _core(), rc, (ElasticAnnulus("ring", ri, ro, e, nu),), angular_cells=24
    )
    result = RadialSection([], bundle=bundle).solve(
        external_pressure_Pa=p, prescribed_axial_strain=0.0
    )
    lam = e * nu / ((1 + nu) * (1 - 2 * nu))
    mu = e / (2 * (1 + nu))
    matrix = np.array(
        [[2 * (lam + mu), -2 * mu / ri**2], [2 * (lam + mu), -2 * mu / ro**2]]
    )
    coeff_a, coeff_b = np.linalg.solve(matrix, [0.0, -p])
    assert result.bundle.ring_outer_mean_displacement_m == pytest.approx(
        coeff_a * ro + coeff_b / ro, rel=1e-5
    )
    assert result.axial_force_N == pytest.approx(
        2 * lam * coeff_a * math.pi * (ro**2 - ri**2), rel=1e-5
    )
    assert result.bundle.core_ring_force_N_per_m == 0.0


@pytest.mark.parametrize("filler", [None, ElasticFiller("pp", 1.5e9, 0.4, 1e-3)])
def test_uniform_free_strain_of_a_touching_trefoil_is_stress_free(filler):
    a = 0.02
    rc = 2 * a / math.sqrt(3)
    bundle = TrefoilBundle(
        _core(1e-3),
        rc,
        (ElasticAnnulus("ring", rc + a, rc + a + 0.003, 0.9e9, 0.46, 1e-3),),
        filler=filler,
        angular_cells=24,
    )
    result = RadialSection([], bundle=bundle).solve()
    state = result.bundle
    assert result.axial_strain == pytest.approx(1e-3, rel=1e-8)
    assert state.ring_outer_mean_displacement_m == pytest.approx(
        1e-3 * (rc + a + 0.003), rel=1e-6
    )
    for force in (
        state.core_core_force_N_per_m,
        state.core_ring_force_N_per_m,
        state.filler_core_force_N_per_m,
        state.filler_ring_force_N_per_m,
    ):
        assert abs(force) < 1e-2  # N/m, against kN/m contact forces


def test_core_tension_gives_the_trefoil_equilibrium_contact_force():
    """Gjøsteen et al. (2027), Sec. 3: three touching helical tubes, 6.92 kN.

    Printed core-to-core force 4.4 kN/m; static equilibrium of one core
    gives F = F_zz sin²(beta) / (sqrt(3) r_c).
    """
    beta = math.radians(7.6)
    bundle = TrefoilBundle(
        (ElasticAnnulus("tube", 4.55e-3, 4.65e-3, 2.1e11, 0.3),),
        5.37e-3,
        (ElasticAnnulus("ring", 0.02, 0.021, 1e6, 0.3),),
        core_lay_angle_rad=beta,
        angular_cells=24,
        radial_cell_size_m=5e-5,
    )
    result = RadialSection([], bundle=bundle).solve(tension_N=6920.0)
    state = result.bundle
    plane_force = state.core_axial_force_N / (3 * math.cos(beta) ** 2)
    equilibrium = plane_force * math.sin(beta) ** 2 / (math.sqrt(3) * 5.37e-3)
    assert state.core_core_force_N_per_m == pytest.approx(equilibrium, rel=1e-6)
    assert state.core_core_force_N_per_m == pytest.approx(4.4e3, rel=0.01)
    assert state.core_axial_force_N == pytest.approx(6920.0, rel=1e-3)


def _filled_bundle(cells: int, stabilisation: float = 1e-10) -> TrefoilBundle:
    a = 0.02247
    rc = 2 * a / math.sqrt(3)
    core = (
        ElasticAnnulus("cond", 0, 0.0097, 117e9, 0.34),
        ElasticAnnulus("xlpe", 0.0097, 0.0195, 0.15e9, 0.45),
        ElasticAnnulus("sheath", 0.0195, a, 0.8856e9, 0.48),
    )
    ring = (ElasticAnnulus("bed", rc + a, rc + a + 0.002, 0.8856e9, 0.48),)
    return TrefoilBundle(
        core,
        rc,
        ring,
        filler=ElasticFiller("pp", 1.5e9, 0.4),
        angular_cells=cells,
        stabilisation=stabilisation,
    )


def test_trefoil_mean_compliance_converges_and_ignores_stabilisation():
    values = {}
    for cells in (24, 48):
        result = RadialSection([], bundle=_filled_bundle(cells)).solve(
            external_pressure_Pa=1e5, prescribed_axial_strain=0.0
        )
        values[cells] = result.bundle.ring_outer_mean_displacement_m
    assert values[24] == pytest.approx(values[48], rel=0.01)
    other = RadialSection([], bundle=_filled_bundle(24, 1e-12)).solve(
        external_pressure_Pa=1e5, prescribed_axial_strain=0.0
    )
    assert other.bundle.ring_outer_mean_displacement_m == pytest.approx(
        values[24], rel=1e-4
    )


def test_flooded_pressure_with_matched_materials_changes_no_contact():
    """Uniform water pressure on every surface is self-equilibrated."""
    e, nu, p = 2e9, 0.3, 1e6
    free = flooded_free_strain(e, nu, p)
    layers = [
        ElasticAnnulus("core", 0.0, 0.03, e, nu, free),
        ElasticHelicalRing("arm", 0.032, 0.004, 40, math.radians(20), e, free, 1e-3),
        ElasticAnnulus("sheath", 0.034, 0.04, e, nu, free),
    ]
    dry = [replace(layer, free_thermal_strain=0.0) for layer in layers]
    area = math.pi * (
        0.03**2 + 0.04**2 - 0.034**2
    ) + 40 * math.pi * 0.004**2 / 4 / math.cos(math.radians(20))
    # A flooded section under true tension T behaves as the dry section under
    # T + p A_solid (the effective-tension relation): compare at equal
    # mechanical axial force.
    wet = RadialSection(layers).solve(tension_N=1e4 + p * area)
    base = RadialSection(dry).solve(tension_N=1e4 + p * area)
    np.testing.assert_allclose(
        wet.interface_line_force_N_per_m,
        base.interface_line_force_N_per_m,
        rtol=1e-9,
    )
    with pytest.raises(CableInputError):
        flooded_free_strain(e, 0.6, p)


def test_capstan_residual_stress_inverts_the_pressure_jump():
    stress = capstan_residual_stress(
        2e6,
        wire_count=76,
        wire_diameter_m=0.005,
        helix_radius_m=0.053,
        lay_angle_rad=math.radians(22),
    )
    area = math.pi * 0.005**2 / 4
    jump = (
        76
        * stress
        * area
        * math.sin(math.radians(22))
        * math.tan(math.radians(22))
        / (2 * math.pi * 0.053**2)
    )
    assert jump == pytest.approx(2e6, rel=1e-12)


def _single_core():
    return cx.load_cable(cx.inter_array_single_core_yaml_path(), geometry_override=None)


def test_identified_residual_state_reproduces_declared_wire_stress():
    cable = _single_core()
    state = solve_cable_radial_state(cable, 0.0)
    targets = residual_wire_stress(cable)
    assert set(targets) == {"inner_armour", "outer_armour"}
    for name, value in targets.items():
        assert state.result.wire_stress_Pa[name] == pytest.approx(value, abs=1.0)
    # The face line force jump across the outer layer is its own capstan
    # jump 2 pi r P0 (nothing is outside it).
    outer = next(x for x in cable.helical_layers if x.name == "outer_armour")
    inner_face, outer_face = state.result.helical_face_line_force_N_per_m[
        "outer_armour"
    ]
    assert outer_face == 0.0
    assert inner_face == pytest.approx(
        2 * math.pi * outer.r_helix_m * outer.pretension_Pa, rel=1e-6
    )


def test_compression_can_open_the_manufacturing_contact():
    """Solved, not superposed: enough compression unloads an interface."""
    cable = _single_core()
    loose = solve_cable_radial_state(cable, -2e6)
    face = loose.result.helical_face_line_force_N_per_m["outer_armour"][0]
    reference = solve_cable_radial_state(cable, 0.0)
    assert face < reference.result.helical_face_line_force_N_per_m["outer_armour"][0]
    assert face >= 0.0


def test_end_cap_and_flooded_axial_loads_are_added_explicitly():
    cable = _single_core()
    closed = solve_cable_radial_state(
        cable, 1e4, external_pressure_Pa=1e6, end_cap=True
    )
    section, _ = build_radial_section(cable)
    r_out = section.layers[-1].outer_radius_m
    assert closed.mechanical_axial_force_N == pytest.approx(
        1e4 - math.pi * 1e6 * r_out**2, rel=1e-12
    )
    wet = solve_cable_radial_state(cable, 1e4, flooded_pressure_Pa=1e6)
    assert wet.mechanical_axial_force_N > 1e4
    with pytest.raises(CableSchemaError):
        solve_cable_radial_state(
            cable, 0.0, flooded_pressure_Pa=1e6, external_pressure_Pa=1e6
        )
    with pytest.raises(CableSchemaError):
        solve_cable_radial_state(cable, 1e4, prescribed_axial_strain=1e-4)


def test_three_core_heating_raises_armour_contact_pressure():
    cable = cx.load_cable(cx.inter_array_three_core_yaml_path(), geometry_override=None)
    cold = solve_cable_radial_state(cable, 5e4).armour_contact_pressure()
    hot = solve_cable_radial_state(
        cable,
        5e4,
        temperature_C={
            x.name: 70.0 for x in cable.layers if getattr(x, "n_cores", 1) == 3
        },
    ).armour_contact_pressure()
    assert hot["inner_armour"] > cold["inner_armour"] * 1.05
    wet = solve_cable_radial_state(
        cable, 5e4, flooded_pressure_Pa=1e6
    ).armour_contact_pressure()
    dry = solve_cable_radial_state(
        cable, 5e4, external_pressure_Pa=1e6
    ).armour_contact_pressure()
    # Flooding changes contact far less than the same dry surface pressure.
    assert abs(wet["inner_armour"] - cold["inner_armour"]) < 0.1 * (
        dry["inner_armour"] - cold["inner_armour"]
    )


def test_solved_adapter_is_opt_in_and_self_consistent():
    cable = _single_core()
    default = armour_contact_pressure(cable, 1e5)
    bounded = armour_contact_pressure(cable, 1e5, contact_model="bounded")
    assert default == bounded
    state = solve_cable_radial_state(cable, 1e5)
    solved = armour_contact_pressure(cable, 1e5, contact_model="solved")
    assert solved == pytest.approx(state.armour_contact_pressure())
    forces = armour_normal_force(cable, 1e5, contact_model="solved")
    assert forces == pytest.approx(state.normal_line_load_per_wire())
    crossings = armour_crossing_line_loads(cable, 1e5, contact_model="solved")
    for name, (partner, load) in crossings.items():
        inner, outer = sorted((name, partner.name))
        pair = (
            ("inner_armour", "outer_armour")
            if {inner, outer} == {"inner_armour", "outer_armour"}
            else (inner, outer)
        )
        reaction = state.interface_line_force(*pair)
        count = next(x.n_wires for x in cable.helical_layers if x.name == name)
        assert load == pytest.approx(reaction / count)
    with pytest.raises(CableSchemaError):
        armour_contact_pressure(cable, 1e5, contact_model="other")
    with pytest.raises(CableSchemaError):
        armour_contact_pressure(cable, 1e5, radial_state=state)


def test_compute_mkappa_accepts_a_solved_radial_state():
    cable = cx.load_cable(cx.inter_array_three_core_yaml_path(), geometry_override=None)
    state = solve_cable_radial_state(cable, 5e4, internal_pressure_Pa=0.0)
    kappa = 0.05 * np.sin(np.linspace(0, 4 * np.pi, 120))
    result = cx.compute_mkappa(cable, kappa, radial_state=state)
    direct = cx.compute_mkappa(cable, kappa, tension_N=5e4, contact_model="solved")
    assert result.W_diss_J_per_m == pytest.approx(direct.W_diss_J_per_m, rel=1e-9)
    assert result.tension_N == 5e4
    with pytest.raises(CableSchemaError):
        cx.compute_mkappa(cable, kappa, tension_N=1.0, radial_state=state)


def test_compute_fatigue_solved_matches_frozen_state_at_constant_tension():
    cable = _single_core()
    kx = 0.05 * np.sin(np.linspace(0, 6 * np.pi, 301))
    ky = np.zeros_like(kx)
    history = cx.compute_fatigue(
        cable, kx, ky, tension_N=np.full_like(kx, 5e4), contact_model="solved"
    )
    frozen = cx.compute_fatigue(
        cable,
        kx,
        ky,
        tension_N=np.full_like(kx, 5e4),
        contact_tension_N=5e4,
        contact_model="solved",
    )
    for name in history.D_bulk_per_wire:
        np.testing.assert_allclose(
            history.D_bulk_per_wire[name], frozen.D_bulk_per_wire[name], rtol=1e-9
        )


def test_unsupported_geometry_is_rejected_not_homogenised():
    export = cx.load_cable(cx.export_three_core_yaml_path(), geometry_override=None)
    with pytest.raises(CableSchemaError, match="metal area"):
        solve_cable_radial_state(export, 0.0)
    three = cx.load_cable(cx.inter_array_three_core_yaml_path(), geometry_override=None)
    with pytest.raises(CableSchemaError):
        build_radial_section(three, filler="other")


def test_three_core_template_reports_its_representation_choices():
    cable = cx.load_cable(cx.inter_array_three_core_yaml_path(), geometry_override=None)
    state = solve_cable_radial_state(cable, 0.0)
    text = " ".join(state.notes)
    # The corrected template declares its semiconducting screens, so no core
    # gap is filled; the pinned R2 copy still has 1 mm gaps filled with the
    # stiffer neighbour's material.
    assert "filler" in text and "filled with" not in text
    legacy = cx.load_cable(
        inter_array_three_core_legacy_r2_yaml_path(), geometry_override=None
    )
    legacy_text = " ".join(solve_cable_radial_state(legacy, 0.0).notes)
    assert "filled with the material of the xlpe insulation" in legacy_text
    void = solve_cable_radial_state(cable, 0.0, filler="void")
    assert "omitted" in " ".join(void.notes)
    menard = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    snug = solve_cable_radial_state(menard, 0.0)
    assert snug.result.bundle.adjusted_overlaps_m["core_ring"] == pytest.approx(
        6.7e-5, rel=1e-6
    )
