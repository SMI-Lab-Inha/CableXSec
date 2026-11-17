# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""A cylindrical layer defined by its datasheet stiffness instead of a material.

A layer whose ``stiffness`` block repeats the ``E A``, ``E I``, mass and
Poisson ratio of a material layer must reproduce that layer's results to
round-off on the fixed-radius and bounded routes; its ``G J`` adds to the
torsional stiffness; the coupled radial models refuse it.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

import cablexsec as cx
from cablexsec.cables.schema import (
    STIFFNESS_LAYER_POISSON,
    CylindricalLayer,
    layer_stiffness,
    stiffness_material_name,
)
from cablexsec.exceptions import CableSchemaError
from cablexsec.materials import library

NAME = "hdpe_bedding"


@pytest.fixture(scope="module")
def base() -> cx.CableData:
    return cx.load_cable(cx.inter_array_single_core_yaml_path())


def _block(base: cx.CableData, *, poisson: bool = True, gj: float = 0.0) -> dict:
    layer = base.get_layer(NAME)
    assert isinstance(layer, CylindricalLayer)
    ri, ro = float(layer.r_inner_m or 0), float(layer.r_outer_m or 0)
    area = math.pi * (ro**2 - ri**2)
    with base.material_scope():
        e = library.get_property(layer.material, "E_Pa")
        rho = library.get_property(layer.material, "rho_kg_m3")
        nu = library.get_property(layer.material, "nu")
    block = {
        "EA_N": e * area,
        "EI_Nm2": e * math.pi * (ro**4 - ri**4) / 4,
        "GJ_Nm2": gj,
        "mass_kg_per_m": rho * area,
        "source": "test datasheet",
    }
    if poisson:
        block["poisson_ratio"] = nu
    return block


def _with_block(base: cx.CableData, block: dict) -> cx.CableData:
    data = base.model_dump(mode="json")
    for layer in data["layers"]:
        if layer["name"] == NAME:
            layer.pop("material")
            layer["stiffness"] = block
    return cx.CableData.model_validate(data)


def test_matches_the_material_layer(base) -> None:
    cable = _with_block(base, _block(base))
    layer = cable.get_layer(NAME)
    assert layer.material == stiffness_material_name(NAME)
    assert layer_stiffness(layer) is not None
    for a, b in ((base, base), (base, cable)):
        sa, sb = cx.compute_stiffness(a), cx.compute_stiffness(b)
        for name in ("ea_fixed_N", "ea_free_N", "ei_min_Nm2", "ei_max_Nm2", "gj_Nm2"):
            assert getattr(sa, name) == pytest.approx(getattr(sb, name), rel=1e-13)
    assert cx.compute_mass_properties(cable).mass_kg_per_m == pytest.approx(
        cx.compute_mass_properties(base).mass_kg_per_m, rel=1e-14
    )
    t = np.linspace(0.0, 20.0, 400)
    kx = 0.01 * np.sin(t)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        damage = [
            cx.compute_fatigue(
                c,
                kx,
                0.3 * kx,
                tension_N=np.full_like(kx, 5.0e4),
                friction_model="direction_tied",
                n_fibres=4,
            )
            for c in (base, cable)
        ]
    for name, values in damage[0].D_total_per_wire.items():
        np.testing.assert_allclose(values, damage[1].D_total_per_wire[name], rtol=1e-10)


def test_gj_adds_to_the_torsional_stiffness(base) -> None:
    plain = cx.compute_stiffness(_with_block(base, _block(base)))
    torsion = cx.compute_stiffness(_with_block(base, _block(base, gj=150.0)))
    assert torsion.gj_Nm2 - plain.gj_Nm2 == pytest.approx(150.0, rel=1e-12)
    assert torsion.ea_fixed_N == plain.ea_fixed_N
    assert torsion.ei_min_Nm2 == plain.ei_min_Nm2


def test_stand_in_material(base) -> None:
    cable = _with_block(base, _block(base, poisson=False))
    material = cable.get_material(stiffness_material_name(NAME))
    layer = cable.get_layer(NAME)
    assert isinstance(layer, CylindricalLayer)
    ri, ro = float(layer.r_inner_m or 0), float(layer.r_outer_m or 0)
    block = _block(base)
    assert material.E_Pa is not None and material.E_eff_bending_Pa is not None
    assert material.E_Pa.value == pytest.approx(
        block["EA_N"] / (math.pi * (ro**2 - ri**2)), rel=1e-14
    )
    assert material.nu is not None and material.nu.value == STIFFNESS_LAYER_POISSON
    assert "model assumption" in material.nu.source
    assert material.alpha_T_K is not None and material.alpha_T_K.value == 0.0
    assert cable.materials == base.materials  # nothing is added to the YAML block


def test_yaml_round_trip(base, tmp_path: Path) -> None:
    cable = _with_block(base, _block(base, gj=10.0))
    path = tmp_path / "cable.yaml"
    path.write_text(yaml.safe_dump(cable.model_dump(mode="json")), encoding="utf-8")
    again = cx.load_cable(str(path))
    assert again == cable


def test_inputs_are_checked(base) -> None:
    good = {
        "name": "unit",
        "layer_type": "cylindrical",
        "r_inner_m": 0.01,
        "r_outer_m": 0.02,
        "stiffness": _block(base),
    }
    CylindricalLayer.model_validate(good)
    for change, words in (
        ({"material": "hdpe"}, "takes no material"),
        ({"structural": False}, "cannot be a serving or non-structural"),
        ({"construction": "serving"}, "cannot be a serving"),
        ({"strain_life": "x"}, "no material for a tape"),
        (
            {
                "extrusion_shrinkage": {
                    "hoop_strain": 0.0,
                    "basis": "assumed",
                    "source": "t",
                }
            },
            "extrusion shrinkage",
        ),
        ({"stiffness": {**_block(base), "EA_N": 0.0}}, "EA_N must be > 0"),
        ({"stiffness": {**_block(base), "GJ_Nm2": -1.0}}, "GJ_Nm2 must be >= 0"),
        ({"stiffness": {**_block(base), "source": " "}}, "needs a source"),
        ({"stiffness": {**_block(base), "poisson_ratio": 0.5}}, "poisson_ratio"),
    ):
        with pytest.raises(ValidationError, match=words):
            CylindricalLayer.model_validate({**good, **change})


def test_coupled_radial_models_refuse_it(base) -> None:
    cable = _with_block(base, _block(base))
    with pytest.raises(CableSchemaError, match="stiffness-defined"):
        cx.solve_section_state(cable, tension_N=1.0e5, contact_model="qp")
    with pytest.raises(CableSchemaError, match="stiffness-defined"):
        cx.solve_cable_radial_state(cable, 1.0e5)
