# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Materials defined in the cable file: schema, writer and physics resolution.

A cable's ``materials:`` block never enters the library. The physics reads it
in the cable's material scope, which is local to a thread, so two cables with
conflicting definitions of one key can be analysed in turn and concurrently.
"""

from __future__ import annotations

import copy
import textwrap
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import cablexsec as cx
import cablexsec.api as api
from cablexsec.cables.schema import (
    CableData,
    CableMaterial,
    load_cable,
    material_overlay,
    scope_entry_points,
    takes_cable,
    uses_cable_materials,
)
from cablexsec.exceptions import CableInputError, CableLookupError
from cablexsec.materials import library
from cablexsec.materials.properties import MATERIAL_PROPERTIES

pytestmark = pytest.mark.analytical_sanity

ARMOUR = "galvanised_steel_armour"

CABLE_WITH_MATERIALS = textwrap.dedent(
    """
    cable:
      name: "Cable with its own materials"
    materials:
      supplier_hdpe:
        description: Sheath grade of the supplier
        polymer: true
        temperature_law: hdpe
        E_Pa: {value: 9.0e8, source: "supplier data sheet", unit: Pa}
        nu: {value: 0.46, source: "model assumption"}
      galvanised_steel_armour:
        E_Pa: {value: 2.0e11, source: "user input"}
        nu: {value: 0.3, source: "user input"}
    layers:
      - name: filler
        layer_type: cylindrical
        material: supplier_hdpe
        r_inner_m: 0.01
        r_outer_m: 0.05
    """
)


def _write(tmp_path: Path, text: str, name: str = "cable.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _menard() -> CableData:
    return cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)


def _with_materials(cable: CableData, materials: dict[str, Any]) -> CableData:
    raw = cable.model_dump(mode="json")
    raw["materials"] = materials
    return CableData.model_validate(raw)


def _armour(modulus: float) -> dict[str, Any]:
    return {
        ARMOUR: {
            "E_Pa": {"value": modulus, "source": "user input"},
            "nu": {"value": 0.3, "source": "user input"},
            "sigma_y_Pa": {"value": 690e6, "source": "user input"},
            "sigma_UTS_Pa": {"value": 1400e6, "source": "user input"},
        }
    }


# --------------------------------------------------------------------------- #
# Schema                                                                      #
# --------------------------------------------------------------------------- #


def test_cable_file_materials_load_as_immutable_hashable_data(tmp_path: Path) -> None:
    cable = load_cable(_write(tmp_path, CABLE_WITH_MATERIALS), geometry_override=None)
    assert [m.name for m in cable.materials] == ["supplier_hdpe", ARMOUR]
    hdpe = cable.get_material("supplier_hdpe")
    assert hdpe.polymer and hdpe.temperature_law == "hdpe"
    assert hdpe.entries() == {
        "E_Pa": (9.0e8, "supplier data sheet"),
        "nu": (0.46, "model assumption"),
    }
    assert isinstance(hash(cable), int)
    with pytest.raises(ValidationError):
        hdpe.polymer = False  # type: ignore[misc]
    with pytest.raises(CableLookupError, match="No cable material named 'steel'"):
        cable.get_material("steel")


def test_property_names_are_the_library_names_and_library_values_admissible() -> None:
    fields = set(CableMaterial.model_fields) - {
        "name",
        "description",
        "polymer",
        "temperature_law",
    }
    assert fields == set(MATERIAL_PROPERTIES)
    for key, entry in library.MATERIALS.items():
        for name, (value, _) in entry.items():
            assert name in MATERIAL_PROPERTIES, (key, name)
            assert MATERIAL_PROPERTIES[name].admits(value), (key, name, value)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"E_Pa": {"value": -1.0, "source": "x"}}, "E_Pa = -1 is outside"),
        ({"nu": {"value": 0.5, "source": "x"}}, "nu = 0.5 is outside"),
        ({"E_Pa": {"value": 1e9, "source": "  "}}, "needs a source"),
        ({"E_Pa": {"value": 1e9}}, "source"),
        ({"E_Pa": {"value": 1e9, "source": "x", "unit": "MPa"}}, "stored in Pa"),
        ({"nu": {"value": 0.3, "source": "x", "unit": "-"}}, "no unit"),
        ({"E_GPa": {"value": 1.0, "source": "x"}}, "Extra inputs"),
        ({"temperature_law": "steel"}, "not a registered"),
        ({"temperature_law": "hdpe"}, "scales E_Pa"),
    ],
)
def test_material_values_are_checked(body: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _with_materials(_menard(), {"m": body})


def test_material_keys_and_bodies_are_checked() -> None:
    base = _menard()
    with pytest.raises(ValidationError, match="must be a mapping"):
        _with_materials(base, {"m": 3})
    with pytest.raises(ValidationError, match="remove 'name'"):
        _with_materials(base, {"m": {"name": "m"}})
    with pytest.raises(ValidationError, match="non-empty"):
        _with_materials(base, {" m": {}})
    raw = base.model_dump(mode="json")
    raw["materials"] = [{"name": "m"}, {"name": "m"}]
    with pytest.raises(ValidationError, match="material names must be unique"):
        CableData.model_validate(raw)
    raw["materials"] = None
    assert CableData.model_validate(raw).materials == ()
    # A material without properties is valid; the model check reports what
    # the layers that use it need.
    assert _with_materials(base, {"m": {}}).get_material("m").entries() == {}


def test_dump_without_materials_is_unchanged_and_round_trips() -> None:
    base = _menard()
    assert "materials" not in base.model_dump(mode="json")
    cable = _with_materials(base, _armour(1.0e11))
    dumped = cable.model_dump(mode="json")
    assert list(dumped["materials"]) == [ARMOUR]
    assert dumped["materials"][ARMOUR]["E_Pa"] == {
        "value": 1.0e11,
        "source": "user input",
    }
    assert CableData.model_validate(dumped) == cable
    assert cable != base and hash(cable) != hash(base)


# --------------------------------------------------------------------------- #
# Library overlay                                                             #
# --------------------------------------------------------------------------- #


def test_loading_never_changes_the_library(tmp_path: Path) -> None:
    before = copy.deepcopy(library.MATERIALS)
    polymers = library.POLYMER_MATERIALS
    cable = load_cable(_write(tmp_path, CABLE_WITH_MATERIALS), geometry_override=None)
    with cable.material_scope():
        assert library.get_property(ARMOUR, "E_Pa") == 2.0e11
    assert before == library.MATERIALS
    assert library.POLYMER_MATERIALS is polymers
    assert "supplier_hdpe" not in library.list_materials()
    assert library.get_property(ARMOUR, "E_Pa") == 210.0e9


def test_overlay_resolves_first_and_scopes_nest(tmp_path: Path) -> None:
    cable = load_cable(_write(tmp_path, CABLE_WITH_MATERIALS), geometry_override=None)
    assert library.active_overlay() is None
    with cable.material_scope():
        assert library.active_overlay() is material_overlay(cable.materials)
        assert library.get_property("supplier_hdpe", "nu") == 0.46
        assert library.get_citation("supplier_hdpe", "E_Pa") == "supplier data sheet"
        assert library.get_property("hdpe", "nu") == 0.48  # the library still answers
        assert library.is_polymer("supplier_hdpe")
        assert not library.is_polymer(ARMOUR)
        with pytest.raises(CableLookupError, match="Materials of the cable"):
            library.get_property("missing_key", "E_Pa")
        with pytest.raises(CableLookupError, match="has no property 'sigma_y_Pa'"):
            library.get_property(ARMOUR, "sigma_y_Pa")
        with _menard().material_scope():  # a cable without materials
            assert library.active_overlay() is None
            assert library.get_property(ARMOUR, "E_Pa") == 210.0e9
        assert library.get_property(ARMOUR, "E_Pa") == 2.0e11
    assert library.active_overlay() is None
    with pytest.raises(CableLookupError, match="Unknown material 'supplier_hdpe'"):
        library.get_property("supplier_hdpe", "E_Pa")


def test_temperature_law_of_a_cable_material_scales_its_modulus(
    tmp_path: Path,
) -> None:
    cable = load_cable(_write(tmp_path, CABLE_WITH_MATERIALS), geometry_override=None)
    law = library.law_modulus_at_temperature
    with cable.material_scope():
        assert library.has_temperature_modulus_model("supplier_hdpe")
        assert not library.has_temperature_modulus_model(ARMOUR)
        assert library.youngs_modulus_at_temperature("supplier_hdpe", 60.0) == (
            pytest.approx(9.0e8 * law("hdpe", 60.0) / law("hdpe", 20.0), rel=1e-15)
        )
        assert library.youngs_modulus_at_temperature("hdpe", 60.0) == law("hdpe", 60.0)
        citation = library.get_temperature_modulus_citation("supplier_hdpe")
        assert citation.startswith("hdpe law scaled to the cable material's")
        assert "supplier data sheet" in citation
        for function in (
            library.youngs_modulus_at_temperature,
            lambda m, *_: library.get_temperature_modulus_citation(m),
        ):
            with pytest.raises(CableLookupError, match="temperature-dependent"):
                function(ARMOUR, 20.0)
    assert library.get_temperature_modulus_citation("hdpe") == (
        library.get_temperature_law_citation("hdpe")
    )
    with pytest.raises(CableLookupError, match="temperature-dependent"):
        library.law_modulus_at_temperature("supplier_hdpe", 20.0)
    with pytest.raises(CableLookupError, match="temperature-dependent"):
        library.get_temperature_law_citation("supplier_hdpe")


def test_axial_modulus_key_follows_the_properties_given() -> None:
    assert library.axial_modulus_key("copper_stranded") == "E_eff_axial_Pa"
    assert library.axial_modulus_key(ARMOUR) == "E_Pa"
    overlay = library.MaterialOverlay(
        {"my_copper": {"E_eff_axial_Pa": (1.1e11, "user input")}}
    )
    with library.material_scope(overlay):
        assert library.axial_modulus_key("my_copper") == "E_eff_axial_Pa"


def test_overlay_rejects_names_it_does_not_define_and_unknown_laws() -> None:
    with pytest.raises(CableInputError, match="does not define"):
        library.MaterialOverlay({}, frozenset({"x"}))
    with pytest.raises(CableInputError, match="not a registered"):
        library.MaterialOverlay({"x": {}}, temperature_laws={"x": "steel"})
    overlay = library.MaterialOverlay({"x": {"nu": (0.3, "user input")}})
    with pytest.raises(TypeError):
        overlay.materials["x"]["nu"] = (0.2, "y")  # type: ignore[index]


# --------------------------------------------------------------------------- #
# Physics: two cables with conflicting definitions of one key                 #
# --------------------------------------------------------------------------- #


def test_conflicting_cables_analysed_in_turn() -> None:
    base = _menard()
    soft = _with_materials(base, _armour(1.0e11))
    stiff = _with_materials(base, _armour(3.0e11))
    reference = cx.compute_stiffness(base).ea_fixed_N
    first = cx.compute_stiffness(soft).ea_fixed_N
    second = cx.compute_stiffness(stiff).ea_fixed_N
    assert first < reference < second
    assert cx.compute_stiffness(soft).ea_fixed_N == first
    assert cx.compute_stiffness(base).ea_fixed_N == reference
    assert api.compute_stiffness(stiff).ea_fixed_N == second


def test_copy_of_a_library_material_changes_no_result() -> None:
    base = _menard()
    entry = library.MATERIALS[ARMOUR]
    same = _with_materials(
        base,
        {ARMOUR: {k: {"value": v, "source": s} for k, (v, s) in entry.items()}},
    )
    assert cx.compute_stiffness(same) == cx.compute_stiffness(base)


def test_conflicting_cables_analysed_concurrently() -> None:
    base = _menard()
    cables = {
        "soft": _with_materials(base, _armour(1.0e11)),
        "stiff": _with_materials(base, _armour(3.0e11)),
        "library": base,
    }
    expected = {k: cx.compute_stiffness(c).ea_fixed_N for k, c in cables.items()}
    barrier = threading.Barrier(len(cables))

    def inside_scope(key: str) -> tuple[float, float]:
        cable = cables[key]
        with cable.material_scope():
            barrier.wait(timeout=30)  # every thread is inside its own scope now
            modulus = library.get_property(ARMOUR, "E_Pa")
            barrier.wait(timeout=30)
        return modulus, cx.compute_stiffness(cable).ea_fixed_N

    for _ in range(3):
        with ThreadPoolExecutor(max_workers=len(cables)) as pool:
            results = dict(zip(cables, pool.map(inside_scope, cables), strict=True))
        assert results["soft"][0] == 1.0e11
        assert results["stiff"][0] == 3.0e11
        assert results["library"][0] == 210.0e9
        assert {k: r[1] for k, r in results.items()} == expected
    assert library.active_overlay() is None


# --------------------------------------------------------------------------- #
# Entry points                                                                #
# --------------------------------------------------------------------------- #


def test_every_public_function_that_takes_a_cable_uses_its_materials() -> None:
    import inspect

    for module in (cx, api):
        for name in module.__all__:
            value = getattr(module, name)
            if inspect.isfunction(value) and takes_cable(value):
                assert getattr(value, "__uses_cable_materials__", False), name
    assert cx.compute_stiffness is api.compute_stiffness


def test_cable_scope_decorator() -> None:
    seen: list[object] = []

    def probe(cable: CableData | None = None, *, other: object = None) -> object:
        seen.append(library.active_overlay())
        return other

    wrapped = uses_cable_materials(probe)
    assert uses_cable_materials(wrapped) is wrapped
    soft = _with_materials(_menard(), _armour(1.0e11))
    assert wrapped(other=3) == 3
    wrapped(soft)
    wrapped(cable=soft)
    overlay = material_overlay(soft.materials)
    assert overlay is not None and seen == [None, overlay, overlay]
    assert not takes_cable(object())  # type: ignore[arg-type]
    namespace: dict[str, Callable[..., object] | int] = {"probe": probe, "n": 1}
    scope_entry_points(namespace, ["probe", "n", "absent"])
    assert getattr(namespace["probe"], "__uses_cable_materials__", False)
    assert namespace["n"] == 1
