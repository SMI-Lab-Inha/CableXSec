# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Independent energy and force checks for mixed-diameter core wire rings.

All dimensions here are synthetic.
"""

import math

import numpy as np
import pytest
from pydantic import ValidationError

import cablexsec as cx
from cablexsec.cables.schema import (
    CableData,
    CableMeta,
    CoreStrandRing,
    CylindricalOffsetLayer,
)
from cablexsec.materials import library
from cablexsec.mechanics.stiffness import (
    bending_stiffness_by_layer,
    ea_fixed,
    ei_min_slip,
)

pytestmark = pytest.mark.analytical_sanity


def _cable(offset=0.025, n_cores=3):
    layer = CylindricalOffsetLayer(
        name="conductor",
        material="copper_stranded",
        n_cores=n_cores,
        r_offset_m=offset,
        r_inner_m=0.0,
        r_outer_m=0.010,
        strand_layers=(
            CoreStrandRing(n_wires=1, d_wire_m=0.002, r_helix_m=0.0),
            CoreStrandRing(n_wires=8, d_wire_m=0.0015, r_helix_m=0.004, pitch_m=0.12),
            CoreStrandRing(n_wires=12, d_wire_m=0.001, r_helix_m=0.008, pitch_m=0.14),
        ),
    )
    return CableData(cable=CableMeta(name="synthetic mixed rings"), layers=(layer,))


def _energy(cable, eps, thermal=0.0):
    e = library.get_property("copper_stranded", "E_eff_axial_Pa")
    layer = cable.layers[0]
    total = 0.0
    for ring in layer.strand_layers:
        a = math.atan2(2 * math.pi * ring.r_helix_m, ring.pitch_m or 1)
        c = math.cos(a)
        wire_strain = c * c * eps - thermal
        total += (
            layer.n_cores
            * ring.n_wires
            * e
            * math.pi
            * ring.d_wire_m**2
            / 4
            * wire_strain**2
            / (2 * c)
        )
    return total


def test_mixed_ring_axial_rigidity_from_independent_wire_energy():
    cable = _cable()
    eps = 0.001
    assert 2 * _energy(cable, eps) / eps**2 == pytest.approx(ea_fixed(cable), rel=1e-13)


@pytest.mark.parametrize("temperature", [20, 75])
@pytest.mark.parametrize("tension", [0, 1000, -1000])
def test_ring_stresses_recover_force_and_thermal_energy_derivative(
    temperature, tension
):
    cable = _cable()
    state = cx.solve_section_state(
        cable, tension_N=tension, temperature_C=temperature, contact_model="bounded"
    )
    layer = cable.layers[0]
    normal = state.layers[layer.name]
    recovered = 0.0
    for stress, ring in zip(normal.strand_stress_Pa, layer.strand_layers, strict=True):
        a = math.atan2(2 * math.pi * ring.r_helix_m, ring.pitch_m or 1)
        recovered += (
            layer.n_cores
            * ring.n_wires
            * stress
            * math.pi
            * ring.d_wire_m**2
            / 4
            * math.cos(a)
        )
    assert recovered == pytest.approx(tension, abs=1e-8)
    step = 1e-7
    force = (
        _energy(cable, state.axial_strain + step, state.free_thermal_strain[layer.name])
        - _energy(
            cable, state.axial_strain - step, state.free_thermal_strain[layer.name]
        )
    ) / (2 * step)
    assert force == pytest.approx(tension, abs=1e-6)
    component = cx.compute_component_response(cable, 0, section_state=state).states[0]
    assert component.sigma_local_min_Pa == pytest.approx(min(normal.strand_stress_Pa))
    assert component.sigma_stick_max_Pa == pytest.approx(max(normal.strand_stress_Pa))


def test_uniform_single_ring_free_heating_is_stress_free():
    cable = _cable()
    layer = cable.layers[0].model_copy(
        update={"strand_layers": (cable.layers[0].strand_layers[-1],)}
    )
    cable = cable.model_copy(update={"layers": (layer,)})
    state = cx.solve_section_state(cable, temperature_C=70, contact_model="bounded")
    assert state.layers[layer.name].strand_stress_Pa[0] == pytest.approx(0, abs=1e-6)


def test_individual_bending_from_perimeter_and_helix_energy():
    cable = _cable()
    layer = cable.layers[0]
    e = library.get_property(layer.material, "E_eff_axial_Pa")
    phi = np.linspace(0, 2 * math.pi, 720, endpoint=False)
    beta = np.linspace(0, 2 * math.pi, 180, endpoint=False)
    # Exact radial integration of epsilon² in a circular wire: R^4/4.
    # Remaining perimeter/lay quadratures are independent of core_strands.
    rigidity = 0.0
    for ring in layer.strand_layers:
        a = math.atan2(2 * math.pi * ring.r_helix_m, ring.pitch_m or 1)
        c, s = math.cos(a), math.sin(a)
        strain_per_radius = c * c * math.cos(2 * a) * np.cos(phi[:, None]) * np.cos(
            beta[None, :]
        ) - c * (1 + s * s) * np.sin(phi[:, None]) * np.sin(beta[None, :])
        rigidity += (
            layer.n_cores
            * ring.n_wires
            * e
            / c
            * 2
            * math.pi
            * (ring.d_wire_m / 2) ** 4
            / 4
            * np.mean(strain_per_radius**2)
        )
    assert ei_min_slip(cable) == pytest.approx(rigidity, rel=1e-13)


def test_plane_section_reference_keeps_internal_and_core_axes_distinct():
    cable = _cable()
    layer = cable.layers[0]
    result = bending_stiffness_by_layer(cable)[layer.name]
    phi = np.linspace(0, 2 * math.pi, 720, endpoint=False)
    e = library.get_property(layer.material, "E_eff_axial_Pa")
    rigidity = 0.0
    for i in range(layer.n_cores):
        core_x = layer.r_offset_m * math.cos(2 * math.pi * i / layer.n_cores)
        for ring in layer.strand_layers:
            c = math.cos(math.atan2(2 * math.pi * ring.r_helix_m, ring.pitch_m or 1))
            strain = c * c * (core_x + ring.r_helix_m * np.cos(phi))
            rigidity += (
                ring.n_wires
                * e
                * math.pi
                * ring.d_wire_m**2
                / 4
                / c
                * np.mean(strain**2)
            )
    assert result.plane_section_Nm2 - result.individual_Nm2 == pytest.approx(
        rigidity, rel=1e-13
    )
    assert result.plane_section_Nm2 - result.shear_compliant_Nm2 == pytest.approx(
        result.internal_strand_parallel_axis_Nm2
    )


def test_bending_limits_do_not_use_equivalent_solid_conductor_radius():
    cable = _cable()
    state = cx.compute_component_response(cable, 0.1).states[0]
    e = library.get_property("copper_stranded", "E_eff_axial_Pa")
    assert state.sigma_local_max_Pa < e * 0.1 * 0.002
    assert state.sigma_stick_max_Pa > e * 0.1 * 0.025
    assert "internal-slip" in state.basis


@pytest.mark.parametrize(
    "update",
    [
        {"d_wire_m": float("nan")},
        {"pitch_m": float("inf")},
        {"n_wires": True},
        {"n_wires": 2},
        {"d_wire_m": 0.1},
        {"pitch_m": None},
    ],
)
def test_invalid_ring_is_rejected(update):
    data = _cable().layers[0].strand_layers[1].model_dump()
    data.update(update)
    with pytest.raises(ValidationError):
        CoreStrandRing.model_validate(data)


@pytest.mark.parametrize(
    "update",
    [
        {"n_strands": 999},
        {"strand_diameter_mm": 2},
        {"strand_lay_angle_deg": 10},
        {"conductor_area_mm2": 999},
        {"r_outer_m": 0.001},
    ],
)
def test_conflicting_summary_geometry_is_rejected(update):
    data = _cable().layers[0].model_dump()
    data.update(update)
    with pytest.raises(ValidationError):
        CylindricalOffsetLayer.model_validate(data)


def test_missing_resolved_geometry_is_not_a_fully_specified_solid_annulus():
    layer = (
        _cable()
        .layers[0]
        .model_copy(
            update={
                "strand_layers": (),
                "construction": "resolved_wire_screen",
            }
        )
    )
    cable = _cable().model_copy(update={"layers": (layer,)})
    assert "conductor.strand_layers" in cable.missing_fields()
