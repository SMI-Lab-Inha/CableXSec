# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Cable schema: validators of the declarations and the geometry override merge."""

from __future__ import annotations

import pytest
import yaml
from _bundled import incomplete_cable
from pydantic import ValidationError

import cablexsec as cx
from cablexsec.cables.schema import (
    CoreStrandRing,
    CylindricalShellLayer,
    FacePressure,
    ManufacturingClamping,
    TapeWinding,
)
from cablexsec.exceptions import CableLookupError


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"pressure_Pa": -1.0}, "pressure_Pa"),
        ({"identification": "none"}, "identification class"),
        ({"source": " "}, "requires a source"),
        (
            {"identification": "assumed_bound", "bounds_Pa": (5e5, 1e5)},
            "lower <= upper",
        ),
    ],
)
def test_face_pressure(kwargs, match):
    base = {"pressure_Pa": 2e5, "identification": "identified", "source": "test"}
    with pytest.raises(ValidationError, match=match):
        FacePressure(**{**base, **kwargs})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"width_m": 0.0}, "width_m"),
        ({"n_tapes": 0}, "n_tapes"),
        ({"source": ""}, "source"),
    ],
)
def test_tape_winding(kwargs, match):
    base = {"width_m": 0.03, "lay_angle_deg": 60.0, "source": "test"}
    with pytest.raises(ValidationError, match=match):
        TapeWinding(**{**base, **kwargs})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"armour_winding_stress_Pa": 5e7, "initial_contact_pressure_Pa": 1e6},
            "exactly one|one of",
        ),
        ({}, "exactly one|one of"),
        ({"armour_winding_stress_Pa": -1.0}, ">= 0"),
        ({"armour_winding_stress_Pa": 5e7, "source": ""}, "requires a source"),
        ({"armour_winding_stress_Pa": 5e7, "basis": "not_published"}, "not_published"),
    ],
)
def test_manufacturing_clamping(kwargs, match):
    base = {"basis": "assumed", "source": "test"}
    with pytest.raises(ValidationError, match=match):
        ManufacturingClamping(**{**base, **kwargs})


def test_clamping_declaration_and_legacy_field_in_copies():
    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    layer = next(
        x for x in cable.helical_layers if x.manufacturing_clamping is not None
    )
    declared = {
        "armour_winding_stress_Pa": 5e7,
        "basis": "assumed",
        "source": "copy test",
    }
    with pytest.raises(ValueError, match="not both"):
        layer.model_copy(
            update={"manufacturing_clamping": declared, "winding_stress_Pa": 1e7}
        )
    redeclared = layer.model_copy(update={"manufacturing_clamping": declared})
    assert redeclared.manufacturing_clamping.armour_winding_stress_Pa == 5e7
    # A legacy value alone replaces the declaration (a sensitivity study).
    swept = layer.model_copy(update={"winding_stress_Pa": 1e7})
    assert swept.manufacturing_clamping is None and swept.winding_stress_Pa == 1e7


def test_strand_ring_and_layer_geometry():
    with pytest.raises(ValidationError, match="centre wire"):
        CoreStrandRing(n_wires=2, d_wire_m=0.003, r_helix_m=0.0)
    cable = cx.iea_task49_66kv_stranded()
    conductor = cable.get_layer("cu_conductor")
    data = conductor.model_dump()
    with pytest.raises(ValidationError, match="greater than r_inner_m"):
        type(conductor).model_validate({**data, "r_inner_m": data["r_outer_m"]})
    rings = list(data["strand_layers"])
    rings[2] = {**rings[2], "r_helix_m": rings[1]["r_helix_m"]}
    with pytest.raises(ValidationError):
        type(conductor).model_validate({**data, "strand_layers": rings})


def test_bonded_laminate_is_smooth():
    export = cx.load_cable(cx.export_three_core_yaml_path(), geometry_override=None)
    shell = export.get_layer("al_metallic_sheath")
    data = {
        **shell.model_dump(),
        "construction": "bonded_laminate",
        "f_corrugation": 1.3,
    }
    with pytest.raises(ValidationError, match="bonded laminate is smooth"):
        CylindricalShellLayer.model_validate(data)


def test_geometry_override_merge(tmp_path):
    override = {
        "cable": {"name": "Ménard 2023 reference cable"},
        "layers": [
            "not a layer",
            {"name": "absent_layer", "d_wire_m": 1.0},
            {"name": "inner_armour", "d_wire_m": 0.0041},
            {
                "name": "outer_armour",
                "manufacturing_clamping": {
                    "armour_winding_stress_Pa": 4e7,
                    "basis": "assumed",
                    "source": "override test",
                },
            },
        ],
        "aggregate_properties": {"OD_mm": 120.0},
        "initial_state": {"external_pressure_Pa": 2e5},
        "core_lay": {
            "pitch_m": 0.8,
            "lay_direction": "right_hand",
            "source": "override",
        },
    }
    path = tmp_path / "override.yaml"
    path.write_text(yaml.safe_dump(override, allow_unicode=True), encoding="utf-8")
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=path)
    assert cable.get_layer("inner_armour").d_wire_m == 0.0041
    clamping = cable.get_layer("outer_armour").manufacturing_clamping
    assert clamping is not None and clamping.armour_winding_stress_Pa == 4e7
    assert cable.aggregate_properties.OD_mm == 120.0
    assert cable.initial_state.external_pressure_Pa == 2e5
    assert cable.core_lay.pitch_m == 0.8
    other = {"cable": {"name": "another cable"}}
    path.write_text(yaml.safe_dump(other), encoding="utf-8")
    with pytest.raises(ValueError, match="identity does not match"):
        cx.load_cable(cx.menard2023_yaml_path(), geometry_override=path)


@pytest.mark.parametrize(
    "content,match",
    [
        (b"cable: [unclosed\n", "invalid YAML"),
        (bytes(range(256)), "not a UTF-8 text file"),
        (b"- a\n- b\n", "root must be a mapping, not list"),
        (b"42\n", "root must be a mapping, not int"),
        (b"cable: {name: x}\nlayers: 5\n", "'layers' must be a list"),
    ],
)
def test_load_cable_refuses_malformed_files(tmp_path, content, match) -> None:
    path = tmp_path / "cable.yaml"
    path.write_bytes(content)
    with pytest.raises(cx.CableSchemaError, match=match):
        cx.load_cable(path, geometry_override=None)


def test_load_cable_refuses_malformed_override(tmp_path) -> None:
    override = tmp_path / "override.yaml"
    override.write_bytes(b"- not a mapping\n")
    with pytest.raises(cx.CableSchemaError, match="root must be a mapping"):
        cx.load_cable(cx.menard2023_yaml_path(), geometry_override=override)


def test_unspecified_wire_geometry():
    layer = incomplete_cable().get_layer("inner_armour")
    for item in ("wire_section", "wire_radial_thickness_m", "wire_surface_width_m"):
        with pytest.raises(CableLookupError, match="d_wire_m is not specified"):
            getattr(layer, item)
