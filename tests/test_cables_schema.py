# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for the cable YAML schema and loader."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from cablexsec.cables import load_cable, menard2023_yaml_path
from cablexsec.cables.schema import (
    CylindricalLayer,
    HelicalLayer,
)
from cablexsec.cables.schema import (
    load_cable as load_cable_direct,
)
from cablexsec.exceptions import CableSchemaError

MINIMAL_CABLE = textwrap.dedent(
    """
    cable:
      name: "Test cable"
      voltage_kV: 70
    layers:
      - name: filler
        layer_type: cylindrical
        material: pp_filler
        r_inner_m: 0.01
        r_outer_m: 0.05
      - name: armour
        layer_type: helical
        material: galvanised_steel_armour
        n_wires: null
        d_wire_m: null
        r_helix_m: null
        lay_angle_deg: null
        lay_direction: right_hand
    aggregate_properties:
      EA_fixed_MN: 100.0
    """
)


# Test-taxonomy category.
pytestmark = pytest.mark.analytical_sanity


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_lay_angle_radians_helper() -> None:
    layer = HelicalLayer(
        name="a", material="galvanised_steel_armour", lay_angle_deg=None
    )
    assert layer.lay_angle_rad is None
    layer45 = HelicalLayer(
        name="b", material="galvanised_steel_armour", lay_angle_deg=45.0
    )
    assert layer45.lay_angle_rad == pytest.approx(0.7853981633974483)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    bad = textwrap.dedent(
        """
        cable:
          name: "Bad cable"
        layers:
          - name: filler
            layer_type: cylindrical
            material: pp_filler
            r_inner_m: 0.01
            r_outer_m: 0.05
            bogus: 1
        """
    )
    path = _write(tmp_path, "bad.yaml", bad)
    with pytest.raises(ValidationError):
        load_cable_direct(path, geometry_override=None)


def test_unknown_layer_type_is_rejected(tmp_path: Path) -> None:
    bad = MINIMAL_CABLE.replace(
        "layer_type: cylindrical\n", "layer_type: wormhole\n", 1
    )
    path = _write(tmp_path, "bad.yaml", bad)
    with pytest.raises(ValidationError):
        load_cable_direct(path, geometry_override=None)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_cable_direct(tmp_path / "nope.yaml", geometry_override=None)


def test_geometry_override_fills_withheld_values(tmp_path: Path) -> None:
    cable_path = _write(tmp_path, "cable.yaml", MINIMAL_CABLE)
    override = textwrap.dedent(
        """
        layers:
          - name: armour
            n_wires: 56
            d_wire_m: 0.005
            r_helix_m: 0.080
            lay_angle_deg: 22.0
        aggregate_properties:
          EA_fixed_MN: 900.0
        """
    )
    ov_path = _write(tmp_path, "override.yaml", override)

    # Without override: armour geometry is null.
    base = load_cable_direct(cable_path, geometry_override=None)
    assert not base.is_fully_specified()

    # With override: armour geometry is supplied and the cable is complete.
    merged = load_cable_direct(cable_path, geometry_override=ov_path)
    assert merged.is_fully_specified()
    armour = merged.get_layer("armour")
    assert isinstance(armour, HelicalLayer)
    assert armour.n_wires == 56
    assert armour.lay_angle_rad == pytest.approx(0.3839724354)
    assert merged.aggregate_properties.EA_fixed_MN == 900.0


def test_override_does_not_introduce_new_layers(tmp_path: Path) -> None:
    # A layer name that matches no layer is refused, not dropped: it is
    # usually a misspelt name whose values would otherwise never apply.
    cable_path = _write(tmp_path, "cable.yaml", MINIMAL_CABLE)
    override = textwrap.dedent(
        """
        layers:
          - name: phantom
            n_wires: 99
        """
    )
    ov_path = _write(tmp_path, "override.yaml", override)
    with pytest.raises(CableSchemaError, match="'phantom' is not a layer"):
        load_cable_direct(cable_path, geometry_override=ov_path)


def test_round_trip_dump_and_reload(tmp_path: Path) -> None:
    """Dumping a loaded cable back to YAML reloads to an equal object."""
    cable = load_cable_direct(
        _write(tmp_path, "c.yaml", MINIMAL_CABLE), geometry_override=None
    )
    dumped = {
        "cable": cable.cable.model_dump(),
        "layers": [layer.model_dump() for layer in cable.layers],
        "aggregate_properties": cable.aggregate_properties.model_dump(),
    }
    round_path = _write(tmp_path, "round.yaml", yaml.safe_dump(dumped))
    reloaded = load_cable_direct(round_path, geometry_override=None)
    assert reloaded == cable


@pytest.mark.parametrize(
    "r_inner_m, r_outer_m, message",
    [
        (-0.01, 0.05, "r_inner_m"),
        (0.05, 0.01, "r_outer_m"),
    ],
)
def test_loader_rejects_physically_invalid_annuli(
    tmp_path: Path, r_inner_m: float, r_outer_m: float, message: str
) -> None:
    raw = textwrap.dedent(
        f"""
        cable:
          name: Invalid cable
        layers:
          - name: invalid
            layer_type: cylindrical
            material: pp_filler
            r_inner_m: {r_inner_m}
            r_outer_m: {r_outer_m}
        """
    )
    with pytest.raises(ValidationError, match=message):
        load_cable_direct(_write(tmp_path, "invalid.yaml", raw), geometry_override=None)


def test_loader_rejects_duplicate_layer_names(tmp_path: Path) -> None:
    raw = MINIMAL_CABLE.replace("name: armour", "name: filler")
    with pytest.raises(ValidationError, match="unique"):
        load_cable_direct(
            _write(tmp_path, "duplicate.yaml", raw), geometry_override=None
        )


def test_override_identity_must_match_target(tmp_path: Path) -> None:
    cable_path = _write(tmp_path, "cable.yaml", MINIMAL_CABLE)
    override = textwrap.dedent(
        """
        cable:
          name: Different cable
        layers: []
        """
    )
    with pytest.raises(CableSchemaError, match="identity"):
        load_cable_direct(
            cable_path,
            geometry_override=_write(tmp_path, "override.yaml", override),
        )


def test_get_layer_and_helical_layers() -> None:
    cable = load_cable(menard2023_yaml_path())
    assert cable.get_layer("inner_armour").layer_type == "helical"
    assert {layer.name for layer in cable.helical_layers} == {
        "inner_armour",
        "outer_armour",
    }
    with pytest.raises(KeyError, match="No layer named"):
        cable.get_layer("nonexistent")


def test_models_are_frozen() -> None:
    cable = load_cable(menard2023_yaml_path())
    with pytest.raises(ValidationError):
        cable.layers[0].name = "mutated"  # type: ignore[misc]


def test_cylindrical_layer_type_round_trips() -> None:
    cable = load_cable(menard2023_yaml_path())
    sheath = cable.get_layer("inner_sheath")
    assert isinstance(sheath, CylindricalLayer)


def test_full_construction_variant_beside_the_benchmark_input(tmp_path: Path) -> None:
    """The filled Menard cable keeps the benchmark file intact."""
    from cablexsec.cables import schema

    folder = Path(schema.__file__).parent
    filled = load_cable(folder / "menard2023_filled.yaml", geometry_override=None)
    plain = load_cable(folder / "menard2023.yaml", geometry_override=None)
    assert len(filled.layers) == len(plain.layers) + 1
    # A geometry override is merged by layer name.
    override = tmp_path / "geometry.yaml"
    override.write_text(
        "layers:\n  - name: inner_armour\n    n_wires: 54\n", encoding="utf-8"
    )
    merged = load_cable(folder / "menard2023.yaml", geometry_override=override)
    assert merged.get_layer("inner_armour").n_wires == 54
    assert plain.get_layer("inner_armour").n_wires == 55


@pytest.mark.parametrize("key", ["enviroment", "aggregate_property", "notes"])
def test_unknown_top_level_key_is_refused(tmp_path: Path, key: str) -> None:
    from cablexsec.exceptions import CableSchemaError

    text = MINIMAL_CABLE + f"{key}:\n  value: 1\n"
    with pytest.raises(CableSchemaError, match="unknown top-level key"):
        load_cable(_write(tmp_path, "typo.yaml", text))


def test_every_bundled_cable_has_only_known_top_level_keys() -> None:
    folder = Path(menard2023_yaml_path()).parent
    for path in sorted(folder.glob("*.yaml")):
        load_cable(path, geometry_override=None)
