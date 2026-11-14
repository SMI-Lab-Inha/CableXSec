# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Independent section integrals and agreement between public calculations."""

from __future__ import annotations

import math

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.capacity import _contact_forces
from cablexsec.cable_adapters import armour_contact_pressure, armour_normal_force
from cablexsec.cables.schema import CableData, CableMeta, CylindricalShellLayer
from cablexsec.materials import library
from cablexsec.materials.sn_curves import get_sn_curve
from cablexsec.mechanics.axisymmetric import compute_contact_forces
from cablexsec.mechanics.stiffness import ea_fixed, load_sharing

pytestmark = pytest.mark.analytical_sanity


def _shell(factor: float = 1.0) -> CylindricalShellLayer:
    return CylindricalShellLayer(
        name="sheath",
        material="aluminium_corrugated",
        r_mid_m=0.05,
        t_m=0.002,
        f_corrugation=factor,
        scf=1.7,
    )


def _export() -> CableData:
    return cx.load_cable(cx.export_three_core_yaml_path(), geometry_override=None)


@pytest.mark.parametrize("factor", [0.2, 0.5, 1.0])
def test_shell_axial_force_matches_annular_integral(factor):
    shell = _shell(factor)
    cable = CableData(cable=CableMeta(name="annular axial reference"), layers=(shell,))
    modulus = library.get_property(shell.material, "E_Pa") * factor
    area = math.pi * (0.051**2 - 0.049**2)
    stiffness = cx.compute_stiffness(cable)
    assert stiffness.ea_fixed_N == pytest.approx(modulus * area, rel=1e-12)
    assert load_sharing(cable, 1e4) == pytest.approx({shell.name: 1e4})


@pytest.mark.parametrize("factor", [0.2, 0.5, 1.0])
def test_shell_bending_matches_circumferential_work(factor):
    shell = _shell(factor)
    cable = CableData(cable=CableMeta(name="shell bending reference"), layers=(shell,))
    # Direct thin-wall integration of stress times lever arm over circumference.
    phi = np.linspace(0, 2 * math.pi, 4096, endpoint=False)
    radius, thickness, curvature = 0.05, 0.002, 0.01
    modulus = library.get_property(shell.material, "E_Pa") * factor
    lever = radius * np.cos(phi)
    moment = np.sum(modulus * curvature * lever**2 * thickness * radius)
    moment *= 2 * math.pi / len(phi)
    stiffness = cx.compute_stiffness(cable)
    assert stiffness.ei_min_Nm2 == pytest.approx(moment / curvature, rel=1e-12)
    assert stiffness.ei_max_Nm2 == pytest.approx(stiffness.ei_min_Nm2, rel=1e-12)


def test_export_component_and_wire_axial_strains_match_stiffness():
    cable = _export()
    tension = 1e5
    expected_strain = tension / ea_fixed(cable)
    components = cx.compute_component_response(cable, 0.0, tension_N=tension)
    wire = cx.compute_stress(cable, 0.0, tension_N=tension)
    assert components.mechanical_axial_strain == pytest.approx(
        expected_strain, rel=1e-12
    )
    for layer in cable.helical_layers:
        modulus = library.get_property(layer.material, "E_Pa")
        expected_stress = modulus * math.cos(layer.lay_angle_rad) ** 2 * expected_strain
        np.testing.assert_allclose(
            wire.sigma_A_Pa[layer.name], expected_stress, rtol=1e-12
        )
        state = next(s for s in components.states if s.name == layer.name)
        assert state.sigma_local_max_Pa == pytest.approx(expected_stress, rel=1e-12)


def test_export_contact_uses_the_original_section_load_sharing():
    cable = _export()
    tension = 1e5
    direct = compute_contact_forces(cable, tension)
    capacity = _contact_forces(cable, tension)
    assert capacity.layer_tension_N == pytest.approx(direct.layer_tension_N)
    assert capacity.Fn_per_wire_N == pytest.approx(direct.Fn_per_wire_N)
    assert armour_normal_force(cable, tension) == pytest.approx(
        direct.Fn_per_wire_N, rel=1e-12
    )
    base = armour_contact_pressure(cable, 0.0)
    loaded = armour_contact_pressure(cable, tension)
    zero = compute_contact_forces(cable, 0.0)
    increments = direct.interface_pressure_Pa - zero.interface_pressure_Pa
    ordered = sorted(cable.helical_layers, key=lambda layer: layer.r_helix_m)
    for i, layer in enumerate(ordered):
        above = increments[i + 1] if i + 1 < len(ordered) else 0.0
        assert loaded[layer.name] - base[layer.name] == pytest.approx(
            increments[i] + above, rel=1e-12
        )


@pytest.mark.parametrize("pressure", [-2e6, 0.0, 2e6])
def test_component_shell_keeps_hoop_and_longitudinal_stress_separate(pressure):
    shell = _shell(0.5)
    cable = CableData(cable=CableMeta(name="biaxial shell reference"), layers=(shell,))
    tension, curvature = 1e4, 0.02
    modulus = library.get_property(shell.material, "E_Pa") * 0.5
    area = 2 * math.pi * 0.05 * 0.002
    longitudinal = tension / area
    bending = shell.scf * modulus * curvature * 0.05
    hoop = pressure * 0.05 / 0.002
    expected_vm = max(
        math.sqrt(z * z + hoop * hoop - z * hoop)
        for z in (longitudinal - bending, longitudinal + bending)
    )
    result = cx.compute_component_response(
        cable,
        curvature,
        tension_N=tension,
        p_int_Pa=max(pressure, 0.0),
        p_ext_Pa=max(-pressure, 0.0),
    ).states[0]
    assert result.sigma_local_min_Pa == pytest.approx(longitudinal - bending)
    assert result.sigma_local_max_Pa == pytest.approx(longitudinal + bending)
    assert result.sigma_equivalent_max_Pa == pytest.approx(expected_vm)


def test_export_axial_fatigue_matches_one_half_cycle_per_wire():
    cable = _export()
    tension = np.array([1e5, 2e6])
    zero = np.zeros(2)
    result = cx.compute_fatigue(cable, zero, zero, tension_N=tension)
    for layer in cable.helical_layers:
        modulus = library.get_property(layer.material, "E_Pa")
        stress_range = (
            modulus
            * math.cos(layer.lay_angle_rad) ** 2
            * np.ptp(tension)
            / ea_fixed(cable)
        )
        expected = 0.5 / get_sn_curve(layer.bulk_sn).cycles_to_failure(
            stress_range / 1e6
        )
        assert expected > 0.0
        np.testing.assert_allclose(
            result.D_bulk_per_wire[layer.name], expected, rtol=1e-12
        )


def test_multiple_shells_are_counted_once_in_component_load_sharing():
    first = _shell(0.5)
    second = first.model_copy(update={"name": "second", "r_mid_m": 0.06, "t_m": 0.001})
    cable = CableData(cable=CableMeta(name="two shells"), layers=(first, second))
    modulus = library.get_property(first.material, "E_Pa") * 0.5
    expected_ea = modulus * 2 * math.pi * (0.05 * 0.002 + 0.06 * 0.001)
    assert ea_fixed(cable) == pytest.approx(expected_ea, rel=1e-12)
    result = cx.compute_component_response(cable, 0.0, tension_N=1e4)
    assert result.axial_strain == pytest.approx(1e4 / expected_ea, rel=1e-12)


def test_local_stress_concentration_does_not_amplify_section_stiffness():
    shell = _shell(0.5)
    first = CableData(cable=CableMeta(name="reference"), layers=(shell,))
    second = CableData(
        cable=first.cable, layers=(shell.model_copy(update={"scf": 4.0}),)
    )
    a, b = cx.compute_stiffness(first), cx.compute_stiffness(second)
    assert a.ea_fixed_N == b.ea_fixed_N
    assert a.ei_min_Nm2 == b.ei_min_Nm2


def test_shell_material_override_is_used_in_equilibrium_and_stress():
    shell = _shell(0.5)
    cable = CableData(cable=CableMeta(name="shell material state"), layers=(shell,))
    supplied = library.get_property(shell.material, "E_Pa") / 2
    area = 2 * math.pi * shell.r_mid_m * shell.t_m
    state = cx.solve_section_state(
        cable,
        tension_N=1e4,
        contact_model="bounded",
        youngs_modulus_Pa={shell.name: supplied},
    )
    result = cx.compute_component_response(cable, 0.02, section_state=state).states[0]
    bending = shell.scf * supplied * shell.f_corrugation * 0.02 * shell.r_mid_m
    assert state.axial_strain == pytest.approx(
        1e4 / (supplied * shell.f_corrugation * area)
    )
    assert result.sigma_local_max_Pa == pytest.approx(1e4 / area + bending)
    assert result.sigma_local_min_Pa == pytest.approx(1e4 / area - bending)
