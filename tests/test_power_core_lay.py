# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Independent helical-path energy and core-axis transfer checks."""

import math

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.cable_adapters import mechanical_equivalent
from cablexsec.cables.schema import (
    CableData,
    CableMeta,
    CoreLay,
    CoreStrandRing,
    CylindricalLayer,
    CylindricalOffsetLayer,
    CylindricalShellLayer,
)
from cablexsec.materials import library
from cablexsec.mechanics.core_lay import bending_factor, core_angle
from cablexsec.mechanics.section_geometry import filler_area, material_volume_per_length
from cablexsec.mechanics.stiffness import (
    axial_torsional_matrix,
    bending_stiffness_by_layer,
)

pytestmark = pytest.mark.analytical_sanity


def _core(*, pitch=0.8, resolved=False, material="hdpe", direction="right_hand"):
    core = CylindricalOffsetLayer(
        name="core",
        material=material,
        n_cores=3,
        r_offset_m=0.05,
        r_inner_m=0,
        r_outer_m=0.022,
        strand_layers=(
            CoreStrandRing(n_wires=1, d_wire_m=0.002, r_helix_m=0),
            CoreStrandRing(n_wires=12, d_wire_m=0.001, r_helix_m=0.005, pitch_m=0.13),
        )
        if resolved
        else (),
    )
    return CableData(
        cable=CableMeta(name="synthetic helical cores"),
        layers=(core,),
        core_lay=CoreLay(pitch_m=pitch, lay_direction=direction),
    )


def _path_energy(cable, extension, twist):
    core = cable.layers[0]
    pitch = cable.core_lay.pitch_m
    sign = 1 if cable.core_lay.lay_direction == "right_hand" else -1
    original = math.hypot(pitch, 2 * math.pi * core.r_offset_m)
    deformed = math.hypot(
        pitch * (1 + extension), core.r_offset_m * (sign * 2 * math.pi + twist * pitch)
    )
    strain = deformed / original - 1
    area = math.pi * core.r_outer_m**2
    e = library.get_property(core.material, "E_Pa")
    return core.n_cores * e * area * strain**2 / 2 * original / pitch


@pytest.mark.parametrize("direction", ["right_hand", "left_hand"])
def test_axial_torsional_lay_terms_from_actual_helical_path_length(direction):
    cable = _core(direction=direction)
    stiffness = axial_torsional_matrix(cable)
    step = 2e-5
    axial = (_path_energy(cable, step, 0) + _path_energy(cable, -step, 0)) / step**2
    torsion = (_path_energy(cable, 0, step) + _path_energy(cable, 0, -step)) / step**2
    mixed = (
        _path_energy(cable, step, step)
        - _path_energy(cable, step, -step)
        - _path_energy(cable, -step, step)
        + _path_energy(cable, -step, -step)
    ) / (4 * step**2)
    assert stiffness.K_ee == pytest.approx(axial, rel=2e-8)
    assert stiffness.K_tt == pytest.approx(torsion, rel=2e-8)
    assert stiffness.K_et == pytest.approx(mixed, rel=2e-8)


def test_core_mass_uses_actual_helix_length_and_not_its_axial_projection():
    cable = _core()
    core = cable.layers[0]
    ratio = (
        math.hypot(cable.core_lay.pitch_m, 2 * math.pi * core.r_offset_m)
        / cable.core_lay.pitch_m
    )
    assert material_volume_per_length(core, cable) == pytest.approx(
        3 * math.pi * 0.022**2 * ratio
    )


@pytest.mark.parametrize("resolved", [False, True])
@pytest.mark.parametrize("temperature", [20, 70])
def test_solved_mean_wire_stress_recovers_force_after_both_lay_projections(
    resolved, temperature
):
    cable = _core(resolved=resolved, material="copper_stranded")
    core = cable.layers[0]
    state = cx.solve_section_state(
        cable, tension_N=1500, temperature_C=temperature, contact_model="bounded"
    )
    cosine = math.cos(math.atan2(2 * math.pi * core.r_offset_m, cable.core_lay.pitch_m))
    if resolved:
        force = (
            core.n_cores
            * cosine
            * sum(
                stress * ring.area_m2 * math.cos(ring.lay_angle_rad)
                for stress, ring in zip(
                    state.layers[core.name].strand_stress_Pa,
                    core.strand_layers,
                    strict=True,
                )
            )
        )
    else:
        force = (
            core.n_cores
            * cosine
            * math.pi
            * core.r_outer_m**2
            * state.layers[core.name].longitudinal_stress_Pa
        )
    assert force == pytest.approx(1500, abs=1e-7)
    component = cx.compute_component_response(cable, 0, section_state=state).states[0]
    if resolved:
        assert component.sigma_local_max_Pa == pytest.approx(
            max(state.layers[core.name].strand_stress_Pa)
        )
    else:
        assert component.sigma_local_max_Pa == pytest.approx(
            state.layers[core.name].longitudinal_stress_Pa
        )


def test_recovered_fixed_twist_lay_moment_matches_axial_twist_energy_derivative():
    cable = _core()
    state = cx.solve_section_state(cable, tension_N=1500, contact_model="bounded")
    k = axial_torsional_matrix(cable)
    assert sum(v.torque_Nm for v in state.layers.values()) == pytest.approx(
        k.K_et * state.axial_strain
    )


def test_laid_annulus_local_stresses_integrate_to_the_bending_energy():
    cable = _core()
    e = library.get_property("hdpe", "E_Pa")
    angle = math.atan2(2 * math.pi * 0.05, 0.8)
    radius, curvature = 0.022, 0.01
    # Independent local-curvature coefficients and circular-area integral.
    total = 0.0
    for phase in np.arange(3) * 2 * math.pi / 3:
        a = math.cos(angle) ** 2 * math.cos(2 * angle) * curvature * math.cos(phase)
        b = -math.cos(angle) * (1 + math.sin(angle) ** 2) * curvature * math.sin(phase)
        total += (
            e
            * math.pi
            * radius**4
            / 4
            * (a * a + b * b)
            / math.cos(angle)
            / curvature**2
        )
    result = bending_stiffness_by_layer(cable)["core"]
    assert result.individual_Nm2 == pytest.approx(total, rel=1e-12)
    component = cx.compute_component_response(cable, curvature, n_points=720).states[0]
    expected_max = (
        e
        * curvature
        * radius
        * max(
            math.hypot(
                math.cos(angle) ** 2 * math.cos(2 * angle) * math.cos(phase),
                math.cos(angle) * (1 + math.sin(angle) ** 2) * math.sin(phase),
            )
            for phase in np.arange(3) * 2 * math.pi / 3
        )
    )
    # The circumference is sampled, so a fibre need not lie at the exact maximum.
    assert component.sigma_local_max_Pa == pytest.approx(expected_max, rel=1e-5)


def test_internal_ring_and_core_parallel_axes_remain_distinct_with_assembly_lay():
    cable = _core(resolved=True, material="copper_stranded")
    no_lay = cable.model_copy(update={"core_lay": None})
    current, straight = (bending_stiffness_by_layer(x)["core"] for x in (cable, no_lay))
    angle = core_angle(cable, cable.layers[0])
    assert current.internal_strand_parallel_axis_Nm2 == pytest.approx(
        straight.internal_strand_parallel_axis_Nm2 * bending_factor(angle)
    )
    assert current.core_parallel_axis_Nm2 == pytest.approx(
        straight.core_parallel_axis_Nm2 * math.cos(angle) ** 3
    )


def test_laid_core_filler_area_excludes_ellipse_volume_not_circle_volume():
    cable = _core()
    filler = CylindricalLayer(
        name="filler",
        material="pp_filler",
        r_inner_m=0,
        r_outer_m=0.1,
        structural=False,
    )
    cable = cable.model_copy(update={"layers": (*cable.layers, filler)})
    c = math.cos(math.atan2(2 * math.pi * 0.05, 0.8))
    assert filler_area(filler, cable) == pytest.approx(
        math.pi * (0.1**2 - 3 * 0.022**2 / c), rel=1e-11
    )


def test_shell_adapter_does_not_drop_assembly_lay():
    cable = _core()
    shell = CylindricalShellLayer(
        name="shell", material="copper_tape_screen", r_mid_m=0.09, t_m=0.001
    )
    cable = cable.model_copy(update={"layers": (*cable.layers, shell)})
    assert mechanical_equivalent(cable).core_lay == cable.core_lay


def test_missing_assembly_pitch_is_not_a_straight_core():
    cable = _core(pitch=None)
    assert "core_lay.pitch_m" in cable.missing_fields()
