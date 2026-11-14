# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Independent component-axis energy integrals and named stiffness limits."""

import math

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.cables.schema import CableData, CableMeta, HelicalLayer
from cablexsec.materials import library
from cablexsec.mechanics.stiffness import (
    bending_stiffness_by_layer,
    ei_max_stuck,
    ei_min_slip,
)

pytestmark = pytest.mark.analytical_sanity


def _wire_layer():
    layer = HelicalLayer(
        name="armour",
        material="galvanised_steel_armour",
        n_wires=24,
        d_wire_m=0.003,
        r_helix_m=0.05,
        lay_angle_deg=20,
    )
    return CableData(cable=CableMeta(name="helical energy reference"), layers=(layer,))


@pytest.mark.parametrize("xi", [0.01, 0.2, 0.8, 1.0])
def test_strain_and_support_energy_recover_reduced_parallel_axis_stiffness(xi):
    cable = _wire_layer()
    layer = cable.helical_layers[0]
    e = library.get_property(layer.material, "E_Pa")
    angle = layer.lay_angle_rad
    area = math.pi * layer.d_wire_m**2 / 4
    q = math.sin(angle) / layer.r_helix_m
    phi = np.linspace(0, 2 * math.pi, 4096, endpoint=False)
    curvature = 0.01
    e0 = curvature * layer.r_helix_m * math.cos(angle) ** 2
    strain = xi * e0 * np.cos(phi)
    wire_work = 0.5 * e * area * np.mean(strain**2)
    support_work = 0.0
    if xi < 1:
        stiffness = e * area * q * q * xi / (1 - xi)
        displacement = (1 - xi) * e0 / q * np.sin(phi)
        support_work = 0.5 * stiffness * np.mean(displacement**2)
    # Wire-axis energy per length -> cable-axis energy per length.
    rigidity = (
        2 * layer.n_wires * (wire_work + support_work) / math.cos(angle) / curvature**2
    )
    reported = bending_stiffness_by_layer(cable, {layer.name: xi})[layer.name]
    assert reported.armour_shear_compliant_Nm2 == pytest.approx(rigidity, rel=1e-12)
    assert reported.armour_shear_compliant_Nm2 == pytest.approx(
        xi * reported.armour_parallel_axis_Nm2, rel=1e-12
    )


@pytest.mark.parametrize(
    "path",
    [
        cx.menard2023_yaml_path,
        cx.inter_array_three_core_yaml_path,
        cx.export_three_core_yaml_path,
    ],
)
def test_public_stiffness_retains_distinct_no_slip_and_plane_section_limits(path):
    cable = cx.load_cable(path(), geometry_override=None)
    result = cx.compute_stiffness(cable)
    layers = bending_stiffness_by_layer(cable, result.xi_per_layer)
    assert sum(v.individual_Nm2 for v in layers.values()) == pytest.approx(
        ei_min_slip(cable)
    )
    assert sum(v.shear_compliant_Nm2 for v in layers.values()) == pytest.approx(
        result.ei_max_Nm2
    )
    assert sum(v.plane_section_Nm2 for v in layers.values()) == pytest.approx(
        result.ei_plane_section_Nm2
    )
    assert result.ei_shear_compliant_Nm2 == result.ei_max_Nm2
    loop = cx.compute_mkappa(cable, np.array([0.0, 1e-9]))
    assert result.ei_armour_no_slip_Nm2 == pytest.approx(loop.EI_max_Nm2)
    assert (
        result.ei_shear_compliant_Nm2 - result.ei_armour_no_slip_Nm2
        == pytest.approx(sum(layer.core_parallel_axis_Nm2 for layer in layers.values()))
    )
    assert result.ei_plane_section_Nm2 == pytest.approx(ei_max_stuck(cable))
    assert (
        result.ei_min_Nm2 <= result.ei_shear_compliant_Nm2 < result.ei_plane_section_Nm2
    )


@pytest.mark.parametrize("xi", [-0.1, 1.1, np.nan, np.inf])
def test_nonphysical_shear_reductions_are_rejected(xi):
    with pytest.raises(ValueError, match="between zero and one"):
        bending_stiffness_by_layer(_wire_layer(), {"armour": xi})


def test_unknown_reduction_layer_is_rejected():
    with pytest.raises(ValueError, match="unknown helical"):
        bending_stiffness_by_layer(_wire_layer(), {"missing": 0.5})


def test_nonstructural_helix_has_no_bending_parallel_axis_term():
    cable = _wire_layer()
    loose = cable.helical_layers[0].model_copy(update={"structural": False})
    cable = cable.model_copy(update={"layers": (loose,)})
    result = bending_stiffness_by_layer(cable, {loose.name: 0.25})[loose.name]
    assert result.individual_Nm2 == 0
    assert result.plane_section_Nm2 == 0
    assert result.shear_compliant_Nm2 == 0
    assert ei_max_stuck(cable) == ei_min_slip(cable) == 0
