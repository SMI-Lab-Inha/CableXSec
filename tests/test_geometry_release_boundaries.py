# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Independent area/volume limits and explicitly unsupported constructions."""

import math

import pytest
from pydantic import ValidationError

from cablexsec.cables.schema import (
    CableData,
    CableMeta,
    CoreStrandRing,
    CylindricalLayer,
    CylindricalOffsetLayer,
    CylindricalShellLayer,
    HelicalLayer,
    HelicalStrandedLayer,
)
from cablexsec.exceptions import CableSchemaError
from cablexsec.mechanics.section_geometry import (
    axial_cosine,
    filler_area,
    material_volume_per_length,
    offset_metal_area,
)


def cable(*layers):
    return CableData(cable=CableMeta(name="synthetic area fixture"), layers=layers)


def core(**kwargs):
    return CylindricalOffsetLayer(
        **(
            {
                "name": "core",
                "material": "copper_stranded",
                "n_cores": 3,
                "r_offset_m": 0.03,
                "r_inner_m": 0,
                "r_outer_m": 0.01,
            }
            | kwargs
        )
    )


def filler(**kwargs):
    return CylindricalLayer(
        **(
            {
                "name": "filler",
                "material": "pp_filler",
                "r_inner_m": 0,
                "r_outer_m": 0.06,
            }
            | kwargs
        )
    )


@pytest.mark.parametrize("offset", [0.03, 0.04])
def test_filler_subtracts_core_envelopes_once(offset):
    conductor, bedding = core(r_offset_m=offset), filler()
    expected = math.pi * (0.06**2 - 3 * 0.01**2)
    assert filler_area(bedding, cable(conductor, bedding)) == pytest.approx(expected)
    assert material_volume_per_length(
        bedding, cable(conductor, bedding)
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "layers,match",
    [
        ([core(r_outer_m=None), filler()], "complete core"),
        ([core(), filler(r_outer_m=None)], "complete radii"),
    ],
)
def test_ambiguous_filler_geometry_rejected(layers, match):
    with pytest.raises(CableSchemaError, match=match):
        filler_area(layers[-1], cable(*layers))


@pytest.mark.parametrize(
    "layers,match",
    [
        ([core(), core(name="other", r_offset_m=0.04), filler()], "one core family"),
        ([core(r_offset_m=0.001), filler()], "cores overlap"),
    ],
)
def test_impossible_core_geometry_is_refused_when_the_cable_is_built(layers, match):
    with pytest.raises(ValidationError, match=match):
        cable(*layers)


def test_resolved_ring_volume_includes_helix_length():
    ring = CoreStrandRing(n_wires=6, d_wire_m=0.001, r_helix_m=0.002, pitch_m=0.1)
    layer = core(strand_layers=(ring,))
    assert offset_metal_area(layer) == pytest.approx(6 * math.pi * 0.001**2 / 4)
    expected = 3 * ring.area_m2 * math.sqrt(1 + (2 * math.pi * 0.002 / 0.1) ** 2)
    assert material_volume_per_length(layer, cable(layer)) == pytest.approx(expected)


def test_volume_for_shell_wire_and_solid():
    shell = CylindricalShellLayer(
        name="shell", material="aluminium", r_mid_m=0.03, t_m=0.001
    )
    wire = HelicalLayer(
        name="wire",
        material="galvanised_steel_armour",
        n_wires=8,
        d_wire_m=0.001,
        r_helix_m=0.0105,
        lay_angle_deg=30,
    )
    solid = CylindricalLayer(name="solid", material="hdpe", r_inner_m=0, r_outer_m=0.01)
    section = cable(solid, wire, shell)  # from the axis outwards, no overlap
    assert material_volume_per_length(shell, section) == pytest.approx(
        2 * math.pi * 0.03 * 0.001
    )
    assert material_volume_per_length(wire, section) == pytest.approx(
        8 * math.pi * 0.001**2 / (4 * math.cos(math.pi / 6))
    )
    assert material_volume_per_length(solid, section) == pytest.approx(
        math.pi * 0.01**2
    )
    assert axial_cosine(wire) == pytest.approx(math.cos(math.pi / 6))
    assert axial_cosine(solid) == 1
    assert axial_cosine(core(strand_lay_angle_deg=30)) == pytest.approx(
        math.cos(math.pi / 6)
    )


def test_incomplete_and_nested_area_require_resolved_geometry():
    with pytest.raises(CableSchemaError, match="complete radii"):
        offset_metal_area(core(r_outer_m=None))
    nested = HelicalStrandedLayer(name="nested", material="copper_stranded")
    with pytest.raises(CableSchemaError, match="resolved geometry"):
        material_volume_per_length(nested, cable(nested))
