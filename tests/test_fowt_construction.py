# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Mass inputs remain separate from the measured-reference section mechanics."""

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest
from fowt_construction import (
    _global_properties,
    construction_properties,
    load_case_section,
)

import cablexsec as cx
from cablexsec.mechanics.core_lay import core_angle
from cablexsec.mechanics.stiffness import ea_fixed, ei_min_slip


@pytest.fixture
def construction():
    root = Path(__file__).resolve().parents[1]
    return (
        cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None),
        json.loads((root / "examples/fowt/menard2023_global_mass.json").read_text()),
    )


def test_mass_model_preserves_local_stiffness(construction):
    cable, recipe = construction
    before = cable.model_dump()
    props = construction_properties(cable, recipe)
    assert props["EA_N"] == ea_fixed(cable)
    assert props["EI_full_slip_Nm2"] == ei_min_slip(cable)
    assert props["dry_mass_kg_m"] == pytest.approx(19.27890480286816)
    assert cable.model_dump() == before


def test_filler_mass_matches_direct_bulk_volume_difference(construction):
    cable, recipe = construction
    zero = copy.deepcopy(recipe)
    zero["unmodelled_filler"]["filled_fraction"] = 0
    with_filler = construction_properties(cable, recipe)
    without_filler = construction_properties(cable, zero)
    beta = core_angle(cable, cable.get_layer("cu_conductor"))
    filler_mass = 905 * (math.pi * 0.0329**2 - 3 * math.pi * 0.0153**2 / math.cos(beta))
    assert with_filler["dry_mass_kg_m"] - without_filler[
        "dry_mass_kg_m"
    ] == pytest.approx(filler_mass)
    assert filler_mass == pytest.approx(1.05584253634375)
    for key in ("EA_N", "EI_full_slip_Nm2", "GJ_Nm2", "outer_diameter_m"):
        assert with_filler[key] == without_filler[key]


def test_default_export_properties_do_not_change():
    cable = cx.load_cable(cx.export_three_core_yaml_path(), geometry_override=None)
    assert construction_properties(cable) == _global_properties(cable)


def test_missing_mass_does_not_fall_back_to_another_material(construction):
    cable, _ = construction
    with pytest.raises(KeyError, match="rho_kg_m3"):
        construction_properties(cable)


@pytest.mark.parametrize(
    "property_name,value",
    [
        ("filled_fraction", -1),
        ("filled_fraction", 1.1),
        ("rho_kg_m3", math.nan),
        ("rho_kg_m3", 0),
    ],
)
def test_invalid_filler_recipe_is_rejected(construction, property_name, value):
    cable, recipe = construction
    recipe["unmodelled_filler"][property_name] = value
    with pytest.raises(ValueError):
        construction_properties(cable, recipe)


def test_undocumented_density_is_rejected(construction):
    cable, recipe = construction
    recipe["densities"]["copper_menard2023"]["source"] = ""
    with pytest.raises(ValueError, match="source"):
        construction_properties(cable, recipe)


def test_nondefault_section_is_loaded_from_its_record():
    path = cx.menard2023_yaml_path().resolve()
    record = {
        "local_section_file": str(path),
        "local_section_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    cable = load_case_section(record)
    assert cable.cable.name == "Ménard 2023 reference cable"
    record["local_section_sha256"] = "wrong"
    with pytest.raises(ValueError, match="changed"):
        load_case_section(record)
