# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Section properties per load case: engine checks.

Public cables only. Every value is checked against a hand calculation or
against the engine function it comes from.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.section_properties import (
    SectionLoadCase,
    compute_mass_properties,
    compute_section_properties,
    tangent_bending_stiffness,
)
from cablexsec.cable_adapters import armour_contact_pressure
from cablexsec.cables.schema import (
    CylindricalLayer,
    CylindricalOffsetLayer,
    HelicalLayer,
)
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.materials import library
from cablexsec.mechanics.mkappa import HelicalLayerSpec
from cablexsec.mechanics.papailiou import full_slip_curvature, slip_onset_curvature
from cablexsec.mechanics.radial_cable import solve_cable_radial_states
from cablexsec.mechanics.stiffness import load_sharing

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def single() -> cx.CableData:
    return cx.load_cable(cx.inter_array_single_core_yaml_path(), geometry_override=None)


@pytest.fixture(scope="module")
def cigre() -> cx.CableData:
    return cx.load_cable(cx.cigre_tb880_yaml_path(), geometry_override=None)


def _spec(n: int = 36, mu: float = 0.2, pressure: float = 2.0e6) -> HelicalLayerSpec:
    return HelicalLayerSpec(
        n_wires=n,
        r_helix_m=0.05,
        lay_angle_rad=math.radians(15.0),
        E_wire_Pa=2.0e11,
        A_wire_m2=math.pi * 0.005**2 / 4,
        xi=0.4,
        mu=mu,
        F_n_axi_N=0.0,
        contact_pressure_Pa=pressure,
        normal_load_is_line=True,
    )


# --------------------------------------------------------------------------- #
# Slip curvatures (engine addition: full_slip_curvature)                      #
# --------------------------------------------------------------------------- #
def test_full_slip_curvature_matches_the_closed_form() -> None:
    spec = _spec()
    a, r = spec.lay_angle_rad, spec.r_helix_m
    sin_a = math.sin(a)
    width = 2 * math.pi * r / spec.n_wires
    sigma0 = (
        r * spec.contact_pressure_Pa * width * math.cos(a) / (spec.A_wire_m2 * sin_a**2)
    )
    k = spec.xi * spec.E_wire_Pa * math.cos(a) ** 2
    # Wire 0 lies on the extreme fibre for bending at phi = 0 (c = 1).
    expected_full = sigma0 * math.expm1(spec.mu * sin_a * math.pi / 2) / (k * r)
    expected_onset = spec.mu * sin_a * sigma0 / (k * r)
    assert full_slip_curvature(spec) == pytest.approx(expected_full, rel=1e-12)
    assert slip_onset_curvature(spec) == pytest.approx(expected_onset, rel=1e-12)
    ratio = full_slip_curvature(spec) / slip_onset_curvature(spec)
    assert ratio == pytest.approx(
        math.expm1(spec.mu * sin_a * math.pi / 2) / (spec.mu * sin_a), rel=1e-12
    )
    # Small friction: the ratio tends to pi/2 (the uniform-pressure relation).
    tiny = _spec(mu=1e-6)
    assert full_slip_curvature(tiny) / slip_onset_curvature(tiny) == pytest.approx(
        math.pi / 2, rel=1e-6
    )


def test_full_slip_curvature_is_the_last_wire_for_any_bending_direction() -> None:
    spec = _spec(n=3)
    phi = 0.3
    a, r = spec.lay_angle_rad, spec.r_helix_m
    sin_a = math.sin(a)
    sigma0 = (
        r
        * spec.contact_pressure_Pa
        * (2 * math.pi * r / 3)
        * math.cos(a)
        / (spec.A_wire_m2 * sin_a**2)
    )
    k = spec.xi * spec.E_wire_Pa * math.cos(a) ** 2
    by_hand = []
    for j in range(3):
        c = abs(math.cos(2 * math.pi * j / 3 - phi))
        cap = sigma0 * math.expm1(spec.mu * sin_a * math.asin(c))
        by_hand.append(cap / (k * r * c))
    assert full_slip_curvature(spec, phi_bend=phi) == pytest.approx(
        max(by_hand), rel=1e-12
    )
    assert full_slip_curvature(_spec(pressure=0.0)) == 0.0
    with pytest.raises(CableInputError):
        full_slip_curvature(spec, phi_bend=math.nan)


# --------------------------------------------------------------------------- #
# Load cases                                                                   #
# --------------------------------------------------------------------------- #
def test_load_case_validation() -> None:
    for bad in (
        {"name": " ", "tension_N": 1.0},
        {"name": "a", "tension_N": -1.0},
        {"name": "a", "tension_N": math.inf},
        {"name": "a", "tension_N": 1.0, "twist": "loose"},
        {"name": "a", "tension_N": 1.0, "external_pressure_Pa": -1.0},
        {"name": "a", "tension_N": 1.0, "temperature_C": math.nan},
    ):
        with pytest.raises(CableInputError):
            SectionLoadCase(**bad)  # type: ignore[arg-type]
    case = SectionLoadCase("Hot", 5e4, "free", 60.0, 1e6)
    assert case.describe() == "50 kN, twist free, 60 °C, external pressure 1 MPa"


def test_bad_settings_are_refused(single: cx.CableData) -> None:
    case = SectionLoadCase("A", 1e4)
    with pytest.raises(CableInputError):
        compute_section_properties(single, [])
    with pytest.raises(CableInputError):
        compute_section_properties(single, [case, case])
    for option in (
        {"bending_contact": "exact"},
        {"annulus": "wet"},
        {"kappa_max_1_m": 0.0},
        {"n_points": 4},
        {"torque_step_Nm": 0.0},
        {"mu": -0.1},
    ):
        with pytest.raises(CableInputError):
            compute_section_properties(single, [case], **option)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Solved route                                                                 #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def solved(single: cx.CableData) -> cx.SectionPropertiesResult:
    cases = [
        SectionLoadCase("Held", 50e3, "held"),
        SectionLoadCase("Free", 50e3, "free"),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute_section_properties(single, cases, kappa_max_1_m=2.0)


def test_axial_and_torsional_values_come_from_the_solved_section(
    single: cx.CableData, solved: cx.SectionPropertiesResult
) -> None:
    held, free = solved.cases
    assert held.route == free.route == "solved"
    zero, state = solve_cable_radial_states(
        single, [0.0, 50e3], loose_layers="neighbour", prescribed_twist_per_m=0.0
    )
    assert held.axial_stiffness_N == state.result.ea_fixed_twist_N
    strain = state.result.axial_strain - zero.result.axial_strain
    assert held.axial_stiffness_secant_N == pytest.approx(50e3 / strain, rel=1e-12)
    assert held.torque_change_Nm == pytest.approx(
        state.result.torque_Nm - zero.result.torque_Nm, rel=1e-12
    )
    assert math.isnan(held.twist_change_rad_per_m)
    zero_f, state_f = solve_cable_radial_states(
        single,
        [0.0, 50e3],
        loose_layers="neighbour",
        prescribed_twist_per_m=None,
        torque_Nm=0.0,
    )
    assert free.axial_stiffness_N == state_f.result.ea_free_twist_N
    assert math.isnan(free.torque_change_Nm)
    # The torque balance solves the same section (its default loose-layer
    # convention is the same): fixed-end torque and free-end twist agree.
    balance = cx.torque_balance(single, tensions_N=[50e3])
    assert held.torque_change_Nm == pytest.approx(
        balance.torque_fixed_Nm[-1] - balance.torque_fixed_Nm[0], rel=1e-9
    )
    assert free.twist_change_rad_per_m == pytest.approx(
        balance.twist_free_rad_per_m[-1], rel=1e-9
    )
    # Torsional stiffness by direction: secant over the step, from separate
    # solves; with no contact opening both equal the tangent at constant tension.
    step = held.torque_step_Nm
    twists = [
        solve_cable_radial_states(
            single,
            [50e3],
            loose_layers="neighbour",
            prescribed_twist_per_m=None,
            torque_Nm=state.result.torque_Nm + dm,
        )[0].result.twist_per_m
        for dm in (0.0, step, -step)
    ]
    assert held.torsional_stiffness_positive_Nm2 == pytest.approx(
        step / (twists[1] - twists[0]), rel=1e-12
    )
    assert held.torsional_stiffness_negative_Nm2 == pytest.approx(
        step / (twists[0] - twists[2]), rel=1e-12
    )
    assert held.torsional_stiffness_positive_Nm2 == pytest.approx(
        held.torsional_stiffness_tangent_Nm2, rel=1e-6
    )
    assert held.torsional_stiffness_tangent_Nm2 == state.result.gj_fixed_tension_Nm2


def test_load_share_and_interface_pressure(
    single: cx.CableData, solved: cx.SectionPropertiesResult
) -> None:
    held = solved.cases[0]
    assert sum(held.load_share_N.values()) == pytest.approx(50e3, rel=1e-9)
    zero, state = solve_cable_radial_states(
        single, [0.0, 50e3], loose_layers="neighbour", prescribed_twist_per_m=0.0
    )
    layer = single.get_layer("inner_armour")
    assert isinstance(layer, HelicalLayer)
    assert layer.d_wire_m and layer.n_wires and layer.lay_angle_rad is not None
    area = layer.n_wires * math.pi * layer.d_wire_m**2 / 4
    change = (
        state.result.wire_stress_Pa["inner_armour"]
        - zero.result.wire_stress_Pa["inner_armour"]
    )
    assert held.load_share_N["inner_armour"] == pytest.approx(
        change * area * math.cos(layer.lay_angle_rad), rel=1e-12
    )
    names = state.result.interface_names
    for (inner, outer), radius, force in zip(
        names,
        state.result.interface_radius_m,
        state.result.interface_line_force_N_per_m,
        strict=True,
    ):
        key = f"{inner} / {outer}"
        assert held.interface_pressure_Pa[key] == pytest.approx(
            force / (2 * math.pi * radius), rel=1e-12
        )
        assert held.interface_radius_m[key] == radius


# --------------------------------------------------------------------------- #
# Bending: backbone, staircase and slip curvatures                             #
# --------------------------------------------------------------------------- #
def test_backbone_is_compute_mkappa_and_the_staircase_has_the_bounds(
    single: cx.CableData, solved: cx.SectionPropertiesResult
) -> None:
    held = solved.cases[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        law = cx.compute_mkappa(single, held.kappa_1_m, tension_N=50e3)
    moment = np.asarray(law.moment_Nm).copy()
    moment[0] = 0.0
    np.testing.assert_array_equal(held.moment_Nm, moment)
    stiffness = cx.compute_stiffness(single)
    assert held.ei_stick_Nm2 == pytest.approx(
        stiffness.ei_armour_no_slip_Nm2, rel=1e-12
    )
    assert held.ei_slip_Nm2 == pytest.approx(stiffness.ei_min_Nm2, rel=1e-12)
    low, high, tangent = held.tangent
    onset = min(s.onset_curvature_1_m for s in held.armour)
    full = max(s.full_slip_curvature_1_m for s in held.armour)
    assert full < held.kappa_1_m[-1]
    # The reported onset is the neutral-axis limit. With the signed capstan
    # bound the compression side slips first, at m = min_psi (1 - exp(-lam
    # psi)) / (lam sin psi) = 1 - 3 lam^2 / 8 + O(lam^3) times it (brute force
    # here), so the pure stick steps end there.
    m = 1.0
    psi = np.linspace(1e-7, np.pi / 2, 200_001)
    specs = cx.cable_to_mkappa_specs(single, 50e3)
    for s, spec in zip(held.armour, specs, strict=True):
        lam = s.mu * abs(np.sin(spec.lay_angle_rad))
        m = min(m, float(np.min(-np.expm1(-lam * psi) / (lam * np.sin(psi)))))
    assert 0.99 < m < 1.0
    stick = high <= m * onset
    slip = low >= full
    assert stick.any() and slip.any()
    np.testing.assert_allclose(tangent[stick], held.ei_stick_Nm2, rtol=1e-9)
    np.testing.assert_allclose(tangent[slip], held.ei_slip_Nm2, rtol=1e-9)
    # The wires slip between the two curvatures and nowhere else.
    fraction = np.asarray(law.slip_fraction)
    kappa = held.kappa_1_m
    assert np.all(fraction[(kappa > 0.0) & (kappa < onset)] <= 1.0 / 60.0)
    # At the full-slip curvature itself the last wire just reaches its cap.
    np.testing.assert_allclose(fraction[kappa > full], 1.0)
    assert fraction[kappa == full][0] < 1.0
    # The slip curvatures are grid nodes, so the kinks are not smeared.
    for item in held.armour:
        assert np.any(np.isclose(kappa, item.onset_curvature_1_m, rtol=0, atol=0))


def test_armour_pressure_is_the_bending_law_pressure(
    single: cx.CableData, solved: cx.SectionPropertiesResult
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pressure = armour_contact_pressure(single, 50e3, contact_model="bounded")
    for item in solved.cases[0].armour:
        assert item.contact_pressure_Pa == pressure[item.layer]
    # The bounded chain has no twist freedom: the free case has the same contact.
    assert [a.contact_pressure_Pa for a in solved.cases[1].armour] == [
        a.contact_pressure_Pa for a in solved.cases[0].armour
    ]
    assert any("no twist freedom" in note for note in solved.cases[1].notes)


def test_solved_contact_bending_uses_the_case_state(single: cx.CableData) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compute_section_properties(
            single, [SectionLoadCase("Free", 50e3, "free")], bending_contact="solved"
        )
        state = solve_cable_radial_states(
            single,
            [50e3],
            loose_layers="neighbour",
            prescribed_twist_per_m=None,
            torque_Nm=0.0,
        )[0]
        law = cx.compute_mkappa(single, result.cases[0].kappa_1_m, radial_state=state)
    np.testing.assert_array_equal(result.cases[0].moment_Nm[1:], law.moment_Nm[1:])
    pressures = state.armour_contact_pressure()
    for item in result.cases[0].armour:
        assert item.contact_pressure_Pa == pytest.approx(pressures[item.layer])


def test_bending_shares_sum_to_the_bounds(solved: cx.SectionPropertiesResult) -> None:
    shares = solved.bending_share
    assert sum(s.full_stick_Nm2 for s in shares) == pytest.approx(
        solved.ei_full_stick_Nm2, rel=1e-12
    )
    assert sum(s.no_slip_shear_compliant_Nm2 for s in shares) == pytest.approx(
        solved.ei_no_slip_shear_compliant_Nm2, rel=1e-12
    )
    assert sum(s.full_slip_Nm2 for s in shares) == pytest.approx(
        solved.ei_full_slip_Nm2, rel=1e-12
    )


def test_tangent_of_a_backbone() -> None:
    low, high, slope = tangent_bending_stiffness(
        np.array([0.0, 1.0, 3.0]), np.array([0.0, 2.0, 3.0])
    )
    assert low.tolist() == [0.0, 1.0] and high.tolist() == [1.0, 3.0]
    assert slope.tolist() == [2.0, 0.5]
    with pytest.raises(CableInputError):
        tangent_bending_stiffness(np.array([0.0, 0.0]), np.array([0.0, 1.0]))


# --------------------------------------------------------------------------- #
# Fixed-radius route, pressure and temperature                                 #
# --------------------------------------------------------------------------- #
def test_fixed_radius_route_when_the_solved_section_refuses(
    cigre: cx.CableData,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compute_section_properties(cigre, [SectionLoadCase("A", 40e3)])
        case = result.cases[0]
        stiffness = cx.compute_stiffness(cigre)
    assert case.route == "fixed_radius" and "radial bundle law" in case.route_reason
    assert case.axial_stiffness_N == stiffness.ea_fixed_N
    assert (
        case.torsional_stiffness_positive_Nm2
        == case.torsional_stiffness_negative_Nm2
        == stiffness.gj_Nm2
    )
    assert math.isnan(case.axial_stiffness_secant_N) and not case.interface_pressure_Pa
    expected = load_sharing(cigre, 40e3)
    for name, force in case.load_share_N.items():
        assert force == pytest.approx(expected[name], rel=1e-9)
    with pytest.raises(CableSchemaError):
        compute_section_properties(
            cigre, [SectionLoadCase("A", 40e3)], bending_contact="solved"
        )


def test_pressure_conversion_of_the_annulus(single: cx.CableData) -> None:
    case = SectionLoadCase("Deep", 50e3, "held", None, 1.0e6)
    radius = compute_mass_properties(single).outer_diameter_m / 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sealed = compute_section_properties(single, [case], annulus="sealed").cases[0]
        flooded = compute_section_properties(single, [case], annulus="flooded").cases[0]
    assert sealed.wall_tension_N == pytest.approx(50e3 - 1.0e6 * math.pi * radius**2)
    # Flooded: the mechanical axial force of the solved section is the
    # effective tension, so the share of the layers sums to it.
    assert sum(flooded.load_share_N.values()) == pytest.approx(50e3, rel=1e-9)
    state = solve_cable_radial_states(
        single,
        [flooded.wall_tension_N],
        loose_layers="neighbour",
        flooded_pressure_Pa=1.0e6,
        external_pressure_Pa=0.0,
    )[0]
    assert state.mechanical_axial_force_N == pytest.approx(50e3, rel=1e-12)
    assert any("no clamping" in note for note in flooded.notes)
    # The row note states the area each conversion used.
    outer = math.pi * radius**2
    assert f"Ao {outer:.6g} m2, the area of the outer diameter" in (
        sealed.wall_tension_basis
    )
    solid = (50e3 - flooded.wall_tension_N) / 1.0e6
    assert f"As {solid:.6g} m2, the solid area of the solved section" in (
        flooded.wall_tension_basis
    )
    mass_solids = compute_mass_properties(single).solid_area_m2
    assert f"have {mass_solids:.6g} m2" in flooded.wall_tension_basis
    assert 0.9 * mass_solids < solid < outer


def test_secant_from_zero_tension_cancels_the_thermal_strain(
    single: cx.CableData,
) -> None:
    """A hot case: both states of the secant are at the case temperature."""
    hot = SectionLoadCase("Hot", 50e3, "held", 65.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        case = compute_section_properties(single, [hot], n_points=8).cases[0]
        zero, loaded = solve_cable_radial_states(
            single,
            [0.0, 50e3],
            loose_layers="neighbour",
            temperature_C=65.0,
            prescribed_twist_per_m=0.0,
        )
    strain = loaded.result.axial_strain - zero.result.axial_strain
    assert case.axial_stiffness_secant_N == pytest.approx(50e3 / strain, rel=1e-9)
    assert case.wall_tension_basis == "the effective tension (no external pressure)"
    differs = abs(case.axial_stiffness_secant_N / case.axial_stiffness_N - 1.0) > 0.02
    noted = any("so a free thermal strain cancels" in note for note in case.notes)
    assert noted == differs


def test_secant_note_names_the_contacts_that_close() -> None:
    from types import SimpleNamespace

    from cablexsec.analysis.section_properties import _secant_note

    def state(contacts: int) -> SimpleNamespace:
        return SimpleNamespace(result=SimpleNamespace(active_contacts=contacts))

    hot = SectionLoadCase("Hot", 80e3, "held", 65.0)
    solved = SimpleNamespace(
        secant=434.7e6, axial_stiffness=564.1e6, zero=state(111), state=state(139)
    )
    note = _secant_note(hot, solved)  # type: ignore[arg-type]
    assert "434.7 MN" in note and "564.1 MN" in note
    assert "at the case temperature and pressure" in note
    assert "from 111 at zero tension to 139 at the case tension" in note
    same = SimpleNamespace(
        secant=600e6, axial_stiffness=601e6, zero=state(5), state=state(5)
    )
    assert _secant_note(hot, same) == ""  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Mass                                                                         #
# --------------------------------------------------------------------------- #
def test_mass_by_hand(single: cx.CableData) -> None:
    mass = compute_mass_properties(single, water_density_kg_m3=1025.0)
    total = 0.0
    radii = []
    for layer in single.layers:
        rho = library.get_property(layer.material, "rho_kg_m3")
        if isinstance(layer, HelicalLayer):
            assert layer.n_wires and layer.d_wire_m and layer.r_helix_m
            assert layer.lay_angle_rad is not None
            area = layer.n_wires * math.pi * layer.d_wire_m**2 / 4
            area /= math.cos(layer.lay_angle_rad)
            radii.append(layer.r_helix_m + layer.d_wire_m / 2)
        elif isinstance(layer, CylindricalOffsetLayer):
            assert layer.r_outer_m is not None and layer.r_inner_m is not None
            assert not layer.strand_lay_angle_deg and single.core_lay is None
            area = (
                layer.conductor_area_mm2 * 1e-6
                if layer.conductor_area_mm2 is not None
                else math.pi * (layer.r_outer_m**2 - layer.r_inner_m**2)
            ) * layer.n_cores
            radii.append((layer.r_offset_m or 0.0) + layer.r_outer_m)
        else:
            assert isinstance(layer, CylindricalLayer)
            assert layer.r_outer_m is not None and layer.r_inner_m is not None
            area = math.pi * (layer.r_outer_m**2 - layer.r_inner_m**2)
            radii.append(layer.r_outer_m)
        assert mass.layer_area_m2[layer.name] == pytest.approx(area, rel=1e-12)
        total += rho * area
    assert mass.mass_kg_per_m == pytest.approx(total, rel=1e-12)
    assert mass.outer_diameter_m == pytest.approx(2 * max(radii), rel=1e-12)
    g, outer = 9.80665, math.pi * mass.outer_diameter_m**2 / 4
    assert mass.submerged_weight_dry_voids_N_per_m == pytest.approx(
        (total - 1025.0 * outer) * g, rel=1e-12
    )
    assert mass.submerged_weight_flooded_voids_N_per_m == pytest.approx(
        (total - 1025.0 * sum(mass.layer_area_m2.values())) * g, rel=1e-12
    )
    assert mass.void_area_m2 > 0.0 and not mass.missing_density


def test_mass_without_a_density_is_not_given() -> None:
    cable = cx.load_cable(cx.witz1996_yaml_path(), geometry_override=None)
    mass = compute_mass_properties(cable)
    assert mass.missing_density and math.isnan(mass.mass_kg_per_m)
    assert math.isnan(mass.submerged_weight_dry_voids_N_per_m)
    with pytest.raises(CableInputError):
        compute_mass_properties(cable, water_density_kg_m3=0.0)
