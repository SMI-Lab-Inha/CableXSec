# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Laid-core friction: material fall-backs and the structural-core requirement.

Generic made-up conductor stranding on the public 66 kV template; each case
asserts the rule the module notes state for a missing material property.
"""

from __future__ import annotations

import pytest

import cablexsec as cx
from cablexsec.api import CoreFrictionSpec
from cablexsec.cables.schema import CoreStrandRing
from cablexsec.exceptions import CableSchemaError
from cablexsec.materials import library
from cablexsec.mechanics.core_friction import _core_layers, build_core_friction

T = 50.0e3


@pytest.fixture(scope="module")
def cable():
    return cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)


@pytest.fixture
def scratch_material():
    """Register a copy of a library material without some properties."""
    names: list[str] = []

    def register(name: str, source: str, drop: set[str]) -> str:
        props = {k: v for k, v in library.get_material(source).items() if k not in drop}
        cx.register_material(name, props)
        names.append(name)
        return name

    yield register
    for name in names:
        cx.unregister_material(name)


def _with_material(cable, layer_name: str, material: str):
    layers = tuple(
        x.model_copy(update={"material": material}) if x.name == layer_name else x
        for x in cable.layers
    )
    return cable.model_copy(update={"layers": layers})


def _stranded(cable):
    conductor = cable.layers[0]
    d = conductor.r_outer_m / 4.5
    rings = [CoreStrandRing(n_wires=1, d_wire_m=d, r_helix_m=0.0)]
    for i, n in enumerate((6, 12, 18, 24), start=1):
        r = i * d
        rings.append(
            CoreStrandRing(n_wires=n, d_wire_m=d, r_helix_m=r, pitch_m=12 * 2 * r)
        )
    layer = conductor.model_copy(
        update={
            "strand_layers": tuple(rings),
            "conductor_area_mm2": None,
            "construction": "resolved_concentric_stranded",
        }
    )
    return cable.model_copy(update={"layers": (layer, *cable.layers[1:])})


def test_conductor_without_copper_friction_takes_the_screen_value(
    cable, scratch_material
):
    stranded = _stranded(cable)
    conductor = stranded.layers[0]
    name = scratch_material("test_conductor_no_mu", conductor.material, {"mu_on_cu"})
    bare = _with_material(stranded, conductor.name, name)
    screens = [
        x for x in _core_layers(bare) if "mu_on_cu" in library.get_material(x.material)
    ]
    assert screens  # the wire screen keeps its own coefficient
    mu_screen = library.get_property(screens[0].material, "mu_on_cu")
    implied = build_core_friction(bare, CoreFrictionSpec(), T)
    explicit = build_core_friction(bare, CoreFrictionSpec(mu_strand=mu_screen), T)
    assert len(implied.strand_rings) == 4
    for a, b in zip(implied.strand_rings, explicit.strand_rings, strict=True):
        assert a.parameters == b.parameters


def test_layer_without_isotropic_modulus_adds_no_shear_compliance(
    cable, scratch_material
):
    # The series shear stack loses the layer, so the interface is stiffer and
    # the stick coefficient rises (xi grows with the interface stiffness).
    screen = next(x for x in _core_layers(cable) if x.material == "copper_wire_screen")
    name = scratch_material("test_screen_no_nu", screen.material, {"nu"})
    stiffer = build_core_friction(
        _with_material(cable, screen.name, name), CoreFrictionSpec(), T
    )
    base = build_core_friction(cable, CoreFrictionSpec(), T)
    rigid = build_core_friction(cable, CoreFrictionSpec(core_shear=False), T)
    assert base.xi < stiffer.xi < rigid.xi


def test_cores_without_structural_layers_are_rejected(cable):
    cores = {x.name for x in _core_layers(cable)}
    layers = tuple(
        x.model_copy(update={"structural": False}) if x.name in cores else x
        for x in cable.layers
    )
    with pytest.raises(CableSchemaError, match="no structural axial stiffness"):
        build_core_friction(
            cable.model_copy(update={"layers": layers}), CoreFrictionSpec(), T
        )
