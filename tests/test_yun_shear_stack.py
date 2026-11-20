# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""The Yun shear stack of a helical layer stops at its support ([Yun2020]).

Walking inward from the helical layer, the concentric cylindrical layers shear
in series; the walk stops at the first helical layer inside and at the laid
power cores, whose layers are never in the stack. A single core on the cable
axis keeps its concentric insulation.
"""

from __future__ import annotations

import math

import pytest

import cablexsec as cx
from cablexsec.cables.schema import (
    CableData,
    CableMeta,
    CylindricalLayer,
    CylindricalOffsetLayer,
    HelicalLayer,
)
from cablexsec.exceptions import CableInputError
from cablexsec.materials import library
from cablexsec.mechanics.stiffness import xi_for_layer, yun_shear_stack
from cablexsec.mechanics.yun_coefficient import (
    series_shear_stiffness,
    shear_modulus,
    yun_xi,
)

pytestmark = pytest.mark.analytical_sanity


def _names(cable: CableData, layer: str) -> list[str]:
    return [x.name for x in yun_shear_stack(cable, layer)]


def _closed_form(cable: CableData, layer_name: str, shear: list[str]) -> float:
    """Yun's formula with the named layers as the series stack."""
    wire = cable.get_layer(layer_name)
    assert isinstance(wire, HelicalLayer)
    assert wire.d_wire_m is not None and wire.r_helix_m is not None
    assert wire.lay_angle_rad is not None
    stack = []
    for name in shear:
        lyr = cable.get_layer(name)
        assert isinstance(lyr, CylindricalLayer | CylindricalOffsetLayer)
        assert lyr.r_inner_m is not None and lyr.r_outer_m is not None
        g = shear_modulus(
            library.get_property(lyr.material, "E_Pa"),
            library.get_property(lyr.material, "nu"),
        )
        stack.append((lyr.r_outer_m - lyr.r_inner_m, g, wire.wire_surface_width_m))
    ea = library.get_property(wire.material, "E_Pa") * math.pi * wire.d_wire_m**2 / 4
    return yun_xi(series_shear_stiffness(stack), ea, wire.lay_angle_rad, wire.r_helix_m)


def test_laid_cores_are_the_support_not_a_shear_layer() -> None:
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    assert _names(cable, "inner_armour") == ["inner_sheath"]
    xi = xi_for_layer(cable, "inner_armour")
    assert xi == pytest.approx(
        _closed_form(cable, "inner_armour", ["inner_sheath"]), rel=1e-12
    )


def test_stack_stops_at_the_helical_layer_inside() -> None:
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    assert _names(cable, "outer_armour") == ["bedding"]
    assert xi_for_layer(cable, "outer_armour") == pytest.approx(
        _closed_form(cable, "outer_armour", ["bedding"]), rel=1e-12
    )


def test_armour_on_armour_has_no_shear_layer() -> None:
    """Two armour layers with nothing between them: plane-section limit."""
    path = cx.inter_array_three_core_yaml_path()
    cable = cx.load_cable(path, geometry_override=None)
    assert _names(cable, "outer_armour") == []
    assert xi_for_layer(cable, "outer_armour") == 1.0
    # The inner armour sees the bedding only; filler and cores are not in it.
    assert _names(cable, "inner_armour") == ["hdpe_bedding"]


def test_flexible_pipe_tensile_armour_stops_at_the_pressure_armour() -> None:
    cable = cx.load_cable(cx.saevik2011_yaml_path(), geometry_override=None)
    assert _names(cable, "pressure_armour") == ["pressure_sheath"]
    assert _names(cable, "inner_tensile_armour") == ["antiwear_tape_inner"]
    assert _names(cable, "outer_tensile_armour") == ["antiwear_tape_middle"]


def test_single_core_keeps_its_concentric_insulation() -> None:
    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    names = _names(cable, "inner_armour")
    assert "xlpe_insulation" in names and names[0] == "hdpe_bedding"
    for layer in names:
        lyr = cable.get_layer(layer)
        if isinstance(lyr, CylindricalOffsetLayer):
            assert lyr.n_cores == 1 and lyr.r_offset_m == 0.0


def test_filler_is_skipped_and_cores_end_the_walk() -> None:
    """A synthetic three-core section, listed from the axis outwards."""
    core = CylindricalOffsetLayer(
        name="insulation",
        material="xlpe_qiao2025",
        n_cores=3,
        r_offset_m=0.03,
        r_inner_m=0.0,
        r_outer_m=0.025,
    )
    filler = CylindricalLayer(
        name="filler", material="pp_filler", r_inner_m=0.0, r_outer_m=0.06
    )
    bedding = CylindricalLayer(
        name="bedding", material="hdpe", r_inner_m=0.06, r_outer_m=0.063
    )
    tape = CylindricalLayer(
        name="tape", material="hdpe", r_inner_m=0.063, r_outer_m=0.064
    )
    armour = HelicalLayer(
        name="armour",
        material="galvanised_steel_armour",
        n_wires=60,
        d_wire_m=0.004,
        r_helix_m=0.066,
        lay_angle_deg=12.0,
    )
    cable = CableData(
        cable=CableMeta(name="three cores"),
        layers=(core, filler, bedding, tape, armour),
    )
    assert _names(cable, "armour") == ["tape", "bedding"]
    assert xi_for_layer(cable, "armour") == pytest.approx(
        _closed_form(cable, "armour", ["tape", "bedding"]), rel=1e-12
    )


def test_modulus_map_needs_only_the_wire_and_the_stack() -> None:
    cable = cx.load_cable(cx.saevik2011_yaml_path(), geometry_override=None)
    with cable.material_scope():
        layer = cable.get_layer("inner_tensile_armour")
        tape = cable.get_layer("antiwear_tape_inner")
        moduli = {
            layer.name: library.get_property(layer.material, "E_Pa"),
            tape.name: library.get_property(tape.material, "E_Pa"),
        }
        xi = xi_for_layer(cable, layer.name, youngs_modulus_Pa=moduli)
        assert xi == xi_for_layer(cable, layer.name)
        moduli[tape.name] *= 0.5
        assert xi_for_layer(cable, layer.name, youngs_modulus_Pa=moduli) < xi


def test_unknown_layer_is_refused() -> None:
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    with pytest.raises(CableInputError):
        yun_shear_stack(cable, "no_such_layer")
    with pytest.raises(CableInputError):
        yun_shear_stack(cable, "bedding")  # not a helical layer


def test_screen_ring_stack_stops_at_an_inner_wire_screen() -> None:
    """The ring of a core's wire screen shears through the core layers inside
    it, down to another resolved wire screen if one lies inside."""
    from cablexsec.analysis.core_screen import _ring_xi

    cable = cx.load_cable(cx.thies2016_yaml_path(), geometry_override=None)
    screen = cable.get_layer("cu_screen_wires")
    assert isinstance(screen, CylindricalOffsetLayer)
    (ring,) = screen.strand_layers
    assert ring.lay_angle_rad is not None
    args = (ring.r_helix_m, ring.d_wire_m, 1.0e3, ring.lay_angle_rad)
    free = _ring_xi(cable, screen, *args)
    assert 0.0 < free < 1.0
    inner = cable.get_layer("swelling_tape_inner")
    layers = tuple(
        x.model_copy(update={"construction": "resolved_wire_screen"})
        if x.name == inner.name
        else x
        for x in cable.layers
    )
    supported = cable.model_copy(update={"layers": layers})
    # The tape directly under the ring is now the support: no layer between.
    assert _ring_xi(supported, screen, *args) == 1.0
