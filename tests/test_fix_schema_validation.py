# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Inputs the cable schema refuses instead of accepting silently.

Geometry overrides whose values would never reach the cable, a lay
handedness given by the sign of the angle, and geometry that cannot be
built (overfilled, overlapping or out-of-order layers, a whole number given
as a flag or text, temperatures below absolute zero, a termination factor
below one).
"""

from __future__ import annotations

import copy
import math
import warnings
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import cablexsec as cx
from cablexsec.cables import schema as s
from cablexsec.cables.schema import WireFillWarning

pytestmark = pytest.mark.analytical_sanity

CABLES = Path(cx.__file__).parent / "cables"
IEA = CABLES / "iea_task49_66kv.yaml"
IEA_NAME = "IEA Task 49 / Janocha 2024 66 kV three-core (public template)"


def _raw(path: Path = IEA) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _layer(data: dict, name: str) -> dict:
    return next(x for x in data["layers"] if x["name"] == name)


def _load(tmp_path: Path, data: dict) -> cx.CableData:
    path = tmp_path / "cable.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return cx.load_cable(path, geometry_override=None)


def _override(tmp_path: Path, data: object, base: Path = IEA) -> cx.CableData:
    path = tmp_path / "override.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return cx.load_cable(base, geometry_override=path)


# --------------------------------------------------------------------------- #
# 1. Geometry override                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "override, match",
    [
        (
            {"layers": [{"name": "outer_armor", "n_wires": 100}]},
            "'outer_armor' is not a layer of the cable",
        ),
        (
            {"layer": [{"name": "outer_armour", "n_wires": 100}]},
            r"unknown top-level key\(s\) layer",
        ),
        ({"environment": {"T4_external_Km_W": 9.9}}, "environment"),
        ({"materials": {"x": {"E_Pa": {"value": 1e9, "source": "u"}}}}, "materials"),
        ({"layers": [["outer_armour", 100]]}, "layer entry 0 must be a mapping"),
        ({"layers": [{"n_wires": 100}]}, "layer entry 0 has no 'name'"),
        (
            {
                "layers": [
                    {"name": "outer_armour", "n_wires": 140},
                    {"name": "outer_armour", "n_wires": 100},
                ]
            },
            "'outer_armour' is given twice",
        ),
        (
            {"layers": [{"name": "outer_armour", "layer_type": "cylindrical"}]},
            "layer_type 'helical' in the definition, not 'cylindrical'",
        ),
        (
            {"cable": {"name": IEA_NAME, "voltage_kV": 33}, "layers": []},
            "may hold only the 'name'",
        ),
        ({"aggregate_properties": [1, 2]}, "'aggregate_properties' must be a mapping"),
        ({"cable": {"name": IEA_NAME}}, "supplies no values"),
        ({}, "supplies no values"),
    ],
)
def test_override_items_that_would_be_dropped_are_refused(tmp_path, override, match):
    with pytest.raises(cx.CableSchemaError, match=match):
        _override(tmp_path, override)


def test_empty_override_file_is_refused(tmp_path):
    path = tmp_path / "override.yaml"
    path.write_bytes(b"")
    with pytest.raises(cx.CableSchemaError, match="supplies no values"):
        cx.load_cable(IEA, geometry_override=path)


def test_override_path_that_is_not_a_file_is_refused(tmp_path):
    with pytest.raises(cx.CableSchemaError, match="not a file"):
        cx.load_cable(IEA, geometry_override=tmp_path)


def test_missing_override_path_is_still_ignored(tmp_path):
    # Documented for withheld geometry: the null placeholders remain.
    cable = cx.load_cable(IEA, geometry_override=tmp_path / "absent.yaml")
    assert cable == cx.load_cable(IEA, geometry_override=None)


def test_override_may_restate_the_layer_type(tmp_path):
    cable = _override(
        tmp_path,
        {
            "cable": {"name": IEA_NAME},
            "layers": [
                {"name": "outer_armour", "layer_type": "helical", "n_wires": 140}
            ],
        },
    )
    assert cable.get_layer("outer_armour").n_wires == 140


# --------------------------------------------------------------------------- #
# 2. Lay handedness only through lay_direction                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("angle", [-10.0, -1e-9, 90.0, -90.0, 95.0])
def test_helical_lay_angle_lies_in_0_to_90(angle):
    with pytest.raises(ValidationError, match="lay_direction"):
        s.HelicalLayer(
            name="a", material="galvanised_steel_armour", lay_angle_deg=angle
        )


@pytest.mark.parametrize("field", ["lay_angle_cable_deg", "lay_angle_strand_deg"])
@pytest.mark.parametrize("angle", [-10.0, 90.0])
def test_stranded_lay_angles_lie_in_0_to_90(field, angle):
    with pytest.raises(ValidationError, match="lay_direction"):
        s.HelicalStrandedLayer(name="a", material="copper_stranded", **{field: angle})


def test_straight_wire_layer_is_accepted():
    layer = s.HelicalLayer(
        name="a", material="galvanised_steel_armour", lay_angle_deg=0.0
    )
    assert layer.lay_angle_deg == 0.0


def test_negative_lay_angle_in_a_cable_file_is_refused(tmp_path):
    data = _raw()
    _layer(data, "outer_armour").update(lay_angle_deg=-10.0, lay_direction="right_hand")
    with pytest.raises(ValidationError, match="not with the sign of the angle"):
        _load(tmp_path, data)


def test_handedness_from_lay_direction_reverses_the_coupling(tmp_path):
    data = _raw()
    right = cx.compute_stiffness(_load(tmp_path, data))
    flip = {"left_hand": "right_hand", "right_hand": "left_hand"}
    for item in (_layer(data, "inner_armour"), _layer(data, "outer_armour")):
        item["lay_direction"] = flip[item["lay_direction"]]
    data["core_lay"]["lay_direction"] = flip[data["core_lay"]["lay_direction"]]
    mirrored = cx.compute_stiffness(_load(tmp_path, data))
    assert mirrored.axial_torsion_coupling_Nm == pytest.approx(
        -right.axial_torsion_coupling_Nm, rel=1e-9
    )
    assert mirrored.ea_free_N == pytest.approx(right.ea_free_N, rel=1e-9)


# --------------------------------------------------------------------------- #
# 3. Geometry that cannot be built                                             #
# --------------------------------------------------------------------------- #
def test_round_wires_that_overfill_their_layer_warn(tmp_path):
    data = _raw()
    _layer(data, "outer_armour")["n_wires"] = 400  # about 165 fit
    with pytest.warns(WireFillWarning, match="outer_armour: 400 round wires"):
        _load(tmp_path, data)


def test_overfilled_strand_ring_warns():
    with pytest.warns(WireFillWarning, match="20 round wires"):
        s.HelicalStrandedLayer(
            name="hs",
            material="copper_stranded",
            n_strands=20,
            r_strand_m=0.003,
            d_strand_m=0.002,
            lay_angle_strand_deg=10.0,
        )


def test_bundled_cables_load_and_only_the_known_ones_overfill():
    # The pinned legacy R2 copies keep their infeasible armour on purpose and
    # the Saevik round variant stands for an interlocked pressure-armour
    # profile by its thickness; no other bundled cable may overfill a layer.
    expected = {
        "inter_array_single_core_33kv_legacy_r2.yaml",
        "inter_array_three_core_33kv_legacy_r2.yaml",
        "saevik2011.yaml",
        "saevik2011_rect.yaml",
    }
    warned = set()
    for path in sorted(CABLES.glob("*.yaml")):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cx.load_cable(path, geometry_override=None)
        if any(issubclass(w.category, WireFillWarning) for w in caught):
            warned.add(path.name)
    assert warned == expected


def test_wire_larger_than_twice_its_helix_radius_is_refused():
    with pytest.raises(ValidationError, match="crosses the cable axis"):
        s.HelicalLayer(
            name="a",
            material="galvanised_steel_armour",
            d_wire_m=0.01,
            r_helix_m=0.004,
        )


def test_round_wire_larger_than_its_radial_room_is_refused(tmp_path):
    data = _raw()
    _layer(data, "outer_armour")["d_wire_m"] = 0.03
    with pytest.raises(ValidationError, match="larger than its radial room"):
        _load(tmp_path, data)


def test_strip_thicker_than_its_radial_room_is_refused(tmp_path):
    data = _raw(CABLES / "witz1996_rect.yaml")
    _layer(data, "inner_tensile_armour")["wire_thickness_m"] = 0.004
    with pytest.raises(
        ValidationError,
        match="'inner_tensile_armour' .* overlaps or lies inside 'antifriction_tape_inner'",
    ):
        _load(tmp_path, data)


def test_stranded_strand_diameter_beyond_its_ring_is_refused():
    with pytest.raises(ValidationError, match="d_strand_m"):
        s.HelicalStrandedLayer(
            name="hs", material="copper_stranded", r_strand_m=0.001, d_strand_m=0.01
        )


def test_annulus_with_equal_radii_is_refused():
    with pytest.raises(ValidationError, match="greater than r_inner_m"):
        s.CylindricalLayer(name="a", material="hdpe", r_inner_m=0.05, r_outer_m=0.05)


def _mutate(data: dict, change: str) -> None:
    if change == "sheath_overlaps":
        _layer(data, "hdpe_outer_sheath")["r_inner_m"] = 0.01
    elif change == "armour_inside_sheath":
        _layer(data, "outer_armour")["r_helix_m"] = 0.079
    elif change == "reversed":
        data["layers"].reverse()
    elif change == "core_after_armour":
        core = _layer(data, "mdpe_core_sheath")
        data["layers"].remove(core)
        data["layers"].append(core)
    elif change == "cores_overlap":
        for layer in data["layers"]:
            if layer.get("n_cores") == 3:
                layer["r_offset_m"] = 0.03
    elif change == "two_core_families":
        _layer(data, "mdpe_core_sheath")["r_offset_m"] = 0.05
    elif change == "core_layers_overlap":
        _layer(data, "xlpe_insulation")["r_inner_m"] = 0.010
    elif change == "two_fillers":
        filler = copy.deepcopy(_layer(data, "mdpe_filler"))
        filler.update(name="second_filler", r_inner_m=0.0759 - 0.001)
        filler["r_outer_m"] = 0.0759
        _layer(data, "mdpe_filler")["r_outer_m"] = 0.0759 - 0.001
        data["layers"].insert(
            data["layers"].index(_layer(data, "mdpe_filler")) + 1, filler
        )
    elif change == "shell_inside_cores":
        data["layers"].insert(
            5,
            {
                "name": "shell",
                "layer_type": "cylindrical_shell",
                "material": "lead_sheath",
                "r_mid_m": 0.05,
                "t_m": 0.002,
            },
        )
    else:  # pragma: no cover - test table typo
        raise AssertionError(change)


@pytest.mark.parametrize(
    "change, match",
    [
        ("sheath_overlaps", "'hdpe_outer_sheath' .* overlaps or lies inside"),
        ("armour_inside_sheath", "'outer_armour' .* overlaps or lies inside"),
        ("reversed", "listed after|overlaps or lies inside|core layer"),
        ("core_after_armour", "'mdpe_core_sheath' is listed after"),
        ("cores_overlap", "neighbouring power cores overlap"),
        ("two_core_families", "one core family"),
        ("core_layers_overlap", "core layer 'xlpe_insulation'"),
        ("two_fillers", "only one interstitial filler"),
        ("shell_inside_cores", "'shell' .* inside the power-core envelope"),
    ],
)
def test_layers_out_of_radial_order_or_overlapping_are_refused(tmp_path, change, match):
    data = _raw()
    _mutate(data, change)
    with pytest.raises(ValidationError, match=match):
        _load(tmp_path, data)


def test_touching_layers_within_rounding_are_accepted():
    inner = s.CylindricalLayer(
        name="a", material="hdpe", r_inner_m=0.01, r_outer_m=0.05
    )
    outer = s.CylindricalLayer(
        name="b", material="hdpe", r_inner_m=0.05 - 1e-12, r_outer_m=0.06
    )
    cable = s.CableData(cable=s.CableMeta(name="touching"), layers=(inner, outer))
    assert [x.name for x in cable.layers] == ["a", "b"]
    with pytest.raises(ValidationError, match="overlaps"):
        s.CableData(
            cable=s.CableMeta(name="overlapping"),
            layers=(inner, outer.model_copy(update={"r_inner_m": 0.049})),
        )


@pytest.mark.parametrize("value", [True, "145", 145.0])
@pytest.mark.parametrize(
    "model, field",
    [
        (s.HelicalLayer, "n_wires"),
        (s.HelicalStrandedLayer, "n_conductors"),
        (s.HelicalStrandedLayer, "n_strands"),
        (s.CylindricalOffsetLayer, "n_cores"),
        (s.CylindricalOffsetLayer, "n_strands"),
    ],
)
def test_counts_must_be_integers(model, field, value):
    with pytest.raises(ValidationError, match=field):
        model(name="a", material="galvanised_steel_armour", **{field: value})


def test_tape_count_must_be_an_integer():
    with pytest.raises(ValidationError, match="n_tapes"):
        s.TapeWinding(width_m=0.01, lay_angle_deg=30.0, n_tapes=True, source="x")


@pytest.mark.parametrize("ambient", [-273.15, -400.0])
def test_ambient_temperature_below_absolute_zero_is_refused(ambient):
    with pytest.raises(ValidationError, match="absolute zero"):
        s.EnvironmentProperties(ambient_C=ambient)
    assert s.EnvironmentProperties(ambient_C=-40.0).ambient_C == -40.0


@pytest.mark.parametrize("scf", [0.5, 0.999])
def test_termination_scf_below_one_is_refused(scf):
    with pytest.raises(ValidationError, match="termination_scf"):
        s.CylindricalLayer(name="a", material="hdpe", termination_scf=scf)
    assert s.CylindricalLayer(name="a", material="hdpe", termination_scf=1.0)


def test_conductor_strand_angle_of_90_degrees_is_refused():
    with pytest.raises(ValidationError, match="strand_lay_angle_deg"):
        s.CylindricalOffsetLayer(
            name="c", material="copper_stranded", strand_lay_angle_deg=90.0
        )
    assert math.isclose(
        s.CylindricalOffsetLayer(
            name="c", material="copper_stranded", strand_lay_angle_deg=0.0
        ).strand_lay_angle_deg
        or 0.0,
        0.0,
    )
