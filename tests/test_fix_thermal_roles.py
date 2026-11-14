# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""compute_thermal: cable materials, conductor material and layer roles.

The thermal ladder reads the cable's own ``materials`` block whichever way it
is called, takes the conductor material from the conductor layer, and finds
the insulation, the metallic sheath and the armour from the schema (layer
type, material conductivity and position), never from layer names.
"""

from __future__ import annotations

import copy
import math
import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

import cablexsec as cx
from cablexsec.cables.schema import load_cable
from cablexsec.exceptions import CableInputError
from cablexsec.materials import library
from cablexsec.materials.properties import MATERIAL_PROPERTIES
from cablexsec.mechanics import thermal

pytestmark = pytest.mark.analytical_sanity

AREA = 630e-6
CURRENT = 800.0


def _raw(path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _load(raw: dict, tmp_path: Path, name: str = "cable.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_cable(path)


def _quiet(func, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return func(*args, **kwargs)


def _library_body(key: str, scale: dict[str, float] | None = None) -> dict:
    scale = scale or {}
    return {
        prop: {"value": value[0] * scale.get(prop, 1.0), "source": "test input"}
        for prop, value in library.get_material(key).items()
        if prop in MATERIAL_PROPERTIES
    }


def test_module_function_reads_the_cable_materials(tmp_path) -> None:
    """mechanics.thermal.compute_thermal applies the cable's material scope."""
    raw = _raw(cx.iea_task49_66kv_yaml_path())
    base = _load(raw, tmp_path, "base.yaml")
    over = copy.deepcopy(raw)
    over["materials"] = {
        "hdpe_janocha2024": {
            "polymer": True,
            **_library_body("hdpe_janocha2024", {"lambda_W_mK": 10.0}),
        }
    }
    cable = _load(over, tmp_path, "over.yaml")
    library_value = _quiet(thermal.compute_thermal, base, CURRENT, AREA)
    module_value = _quiet(thermal.compute_thermal, cable, CURRENT, AREA)
    top_value = _quiet(cx.compute_thermal, cable, CURRENT, AREA)
    assert module_value.T_conductor_C == top_value.T_conductor_C
    assert module_value.T_conductor_C < library_value.T_conductor_C - 0.5
    # The top-level wrapper does not wrap the scoped function a second time.
    assert cx.compute_thermal is thermal.compute_thermal


def _aluminium_cable(tmp_path):
    raw = _raw(cx.iea_task49_66kv_yaml_path())
    raw["materials"] = {
        "al_conductor": {
            key: {"value": value, "source": "test input"}
            for key, value in {
                "E_Pa": 6.9e10,
                "nu": 0.33,
                "rho_kg_m3": 2700.0,
                "rho_ohm_m": 2.8264e-8,
                "alpha_R_per_K": 4.03e-3,
                "lambda_W_mK": 237.0,
                "alpha_T_K": 23e-6,
            }.items()
        }
    }
    conductor = next(x for x in raw["layers"] if x["name"] == "cu_conductor")
    conductor["material"] = "al_conductor"
    return _load(raw, tmp_path, "al.yaml")


def test_conductor_material_comes_from_the_conductor_layer(tmp_path) -> None:
    """An aluminium conductor layer is not computed as copper."""
    cable = _aluminium_cable(tmp_path)
    default = _quiet(cx.compute_thermal, cable, CURRENT, AREA)
    explicit = _quiet(
        cx.compute_thermal, cable, CURRENT, AREA, conductor_material="al_conductor"
    )
    copper = _quiet(
        cx.compute_thermal, cable, CURRENT, AREA, conductor_material="copper_stranded"
    )
    assert default.T_conductor_C == explicit.T_conductor_C
    assert default.T_conductor_C > copper.T_conductor_C + 10.0
    assert default.provenance is not None
    assert any(
        "'al_conductor' from the conductor layer 'cu_conductor'" in x
        for x in default.provenance.limitations
    )
    assert copper.provenance is not None
    assert any("given explicitly" in x for x in copper.provenance.limitations)


def _concentric(voltage_kV: float = 70.0):
    from cablexsec.cables.schema import (  # noqa: PLC0415
        _LAYER_ADAPTER,
        CableData,
        CableMeta,
        EnvironmentProperties,
    )

    layers = [
        {
            "name": "layer_a",
            "layer_type": "cylindrical",
            "material": "xlpe_qiao2025",
            "r_inner_m": 0.010,
            "r_outer_m": 0.020,
        },
        {
            "name": "layer_b",
            "layer_type": "cylindrical",
            "material": "hdpe",
            "r_inner_m": 0.020,
            "r_outer_m": 0.024,
        },
    ]
    return CableData(
        cable=CableMeta(name="no conductor layer", voltage_kV=voltage_kV),
        layers=tuple(_LAYER_ADAPTER.validate_python(x) for x in layers),
        environment=EnvironmentProperties(ambient_C=15.0, T4_external_Km_W=0.30),
    )


def test_undeterminable_conductor_material_raises() -> None:
    cable = _concentric()
    with pytest.raises(CableInputError, match="conductor material cannot be"):
        cx.compute_thermal(cable, 600.0, 300e-6)
    result = cx.compute_thermal(
        cable, 600.0, 300e-6, conductor_material="copper_stranded"
    )
    assert math.isfinite(result.T_conductor_C)


def test_conductor_material_without_resistivity_is_refused(tmp_path) -> None:
    """No silent copper resistivity for a conductor material without one."""
    raw = _raw(cx.iea_task49_66kv_yaml_path())
    body = _library_body("copper_janocha2024")
    del body["rho_ohm_m"]
    raw["materials"] = {"cu_no_resistivity": body}
    conductor = next(x for x in raw["layers"] if x["name"] == "cu_conductor")
    conductor["material"] = "cu_no_resistivity"
    cable = _load(raw, tmp_path)
    with pytest.raises(CableInputError, match="'cu_no_resistivity' needs rho_ohm_m"):
        cx.compute_thermal(cable, CURRENT, AREA)


def test_insulation_is_not_found_by_its_name(tmp_path) -> None:
    """Renaming the insulation leaves the T1/T2 split, hence the result, alone."""
    raw = _raw(cx.iea_task49_66kv_yaml_path())
    base = _load(raw, tmp_path, "base.yaml")
    renamed = copy.deepcopy(raw)
    layer = next(x for x in renamed["layers"] if x["name"] == "xlpe_insulation")
    layer["name"] = "main_dielectric"
    other = _load(renamed, tmp_path, "renamed.yaml")
    a = _quiet(cx.compute_thermal, base, CURRENT, AREA, lambda1=0.3)
    b = _quiet(cx.compute_thermal, other, CURRENT, AREA, lambda1=0.3)
    assert b.T_conductor_C == pytest.approx(a.T_conductor_C, rel=1e-14)


def test_no_sheath_refuses_losses_that_need_the_split() -> None:
    cable = _concentric()
    kw = {"conductor_material": "copper_stranded"}
    with pytest.raises(CableInputError, match="no metallic sheath or screen"):
        cx.compute_thermal(cable, 600.0, 300e-6, lambda1=0.2, **kw)
    with pytest.raises(CableInputError, match="no metallic sheath or screen"):
        _raise_dielectric_without_sheath()


def _raise_dielectric_without_sheath() -> None:
    from cablexsec.cables.schema import (  # noqa: PLC0415
        _LAYER_ADAPTER,
        CableData,
        CableMeta,
        EnvironmentProperties,
    )

    layers = [
        {
            "name": "core",
            "layer_type": "cylindrical_offset",
            "material": "copper_stranded",
            "n_cores": 1,
            "r_offset_m": 0.0,
            "r_inner_m": 0.0,
            "r_outer_m": 0.020,
        },
        {
            "name": "dielectric",
            "layer_type": "cylindrical_offset",
            "material": "xlpe_qiao2025",
            "n_cores": 1,
            "r_offset_m": 0.0,
            "r_inner_m": 0.020,
            "r_outer_m": 0.040,
            "electrical": {"permittivity_relative": 2.5, "loss_factor_tan_delta": 1e-3},
        },
        {
            "name": "jacket",
            "layer_type": "cylindrical",
            "material": "hdpe",
            "r_inner_m": 0.040,
            "r_outer_m": 0.046,
        },
    ]
    cable = CableData(
        cable=CableMeta(name="unscreened 220 kV", voltage_kV=220.0),
        layers=tuple(_LAYER_ADAPTER.validate_python(x) for x in layers),
        environment=EnvironmentProperties(ambient_C=15.0, T4_external_Km_W=0.30),
    )
    cx.compute_thermal(cable, 600.0, 1000e-6)


@pytest.mark.parametrize(
    "path_fn",
    [
        "inter_array_single_core_yaml_path",
        "inter_array_three_core_yaml_path",
        "cigre_tb880_yaml_path",
        "iea_task49_66kv_yaml_path",
    ],
)
def test_armour_temperature_at_the_innermost_armour_helix(path_fn) -> None:
    cable = load_cable(getattr(cx, path_fn)())
    result = _quiet(cx.compute_thermal, cable, CURRENT, AREA)
    radius = min(
        float(layer.r_helix_m) for layer in cable.helical_layers if layer.structural
    )
    expected = float(np.interp(radius, result.r_m, result.T_C))
    assert result.T_armour_C == pytest.approx(expected, rel=1e-14)
    # Never the surface temperature where the armour lies inside the cable.
    assert result.T_armour_C > result.T_C[-1] + 1e-6


def test_armour_found_without_its_name(tmp_path) -> None:
    raw = _raw(cx.inter_array_single_core_yaml_path())
    base = _load(raw, tmp_path, "base.yaml")
    renamed = copy.deepcopy(raw)
    for index, layer in enumerate(renamed["layers"]):
        layer["name"] = f"layer_{index}"
    other = _load(renamed, tmp_path, "renamed.yaml")
    a = _quiet(cx.compute_thermal, base, CURRENT, AREA)
    b = _quiet(cx.compute_thermal, other, CURRENT, AREA)
    assert b.T_armour_C == a.T_armour_C
    assert b.T_conductor_C == a.T_conductor_C


def test_cable_without_armour_has_no_armour_temperature() -> None:
    result = cx.compute_thermal(
        _concentric(), 600.0, 300e-6, conductor_material="copper_stranded"
    )
    assert math.isnan(result.T_armour_C)
    assert result.provenance is not None
    assert any("T_armour_C is NaN" in x for x in result.provenance.limitations)


def test_docstring_states_lambda2_is_used() -> None:
    doc = thermal.compute_thermal.__doc__ or ""
    assert "**not** implemented" not in doc
    assert "(``λ2 = 0``)" not in doc
    assert "lambda2 : float" in doc


def test_component_conductor_role_comes_from_the_schema() -> None:
    """Names neither make nor unmake the conductor (net-section limit)."""
    from cablexsec.analysis.components import _component_type  # noqa: PLC0415

    cable = load_cable(cx.iea_task49_66kv_yaml_path())
    layers = {layer.name: layer for layer in cable.layers}
    conductor = layers["cu_conductor"].model_copy(update={"name": "core_centre"})
    tape = layers["cu_conductor_screen"].model_copy(update={"name": "conductor_tape"})
    assert _component_type(conductor) == "conductor"
    assert _component_type(tape) == "screen"
    assert _component_type(layers["cu_conductor_screen"]) == "screen"
    insulation = layers["xlpe_insulation"].model_copy(update={"name": "dielectric"})
    assert _component_type(insulation) == "insulation"
