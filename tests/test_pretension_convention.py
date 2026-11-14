# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""One definition of ``pretension_Pa`` for friction, normal load and Hertz contact.

``pretension_Pa`` is the per-layer radial pressure jump ``P0``. It enters the
bounded radial chain (:func:`compute_contact_forces`) and accumulates from the
outside in. The friction-setting ``P1+P2`` sums the two face pressures of that
chain, exactly as for winding stress.
"""

import math

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.cable_adapters import (
    armour_contact_pressure,
    armour_crossing_line_loads,
)
from cablexsec.cables.schema import HelicalLayer
from cablexsec.mechanics.axisymmetric import (
    compute_contact_forces,
    compute_winding_contact_pressure,
)


def _load(path):
    return cx.load_cable(path, geometry_override=None)


def _two_face(cable):
    """``P1+P2`` from the bounded chain at zero tension and zero pressure."""
    helices = sorted(cable.helical_layers, key=lambda layer: layer.r_helix_m)
    chain = compute_contact_forces(cable, 0.0).interface_pressure_Pa
    above = np.r_[chain[1:], 0.0]
    return {
        layer.name: float(below + upper)
        for layer, below, upper in zip(helices, chain, above, strict=True)
    }


@pytest.mark.parametrize(
    ("path", "p0"),
    [
        (cx.export_three_core_yaml_path(), 3.0e6),
        (cx.inter_array_three_core_yaml_path(), 2.0e6),
        (cx.inter_array_single_core_yaml_path(), 2.0e6),
    ],
)
def test_pretension_pressure_follows_the_accumulating_chain(path, p0):
    cable = _load(path)
    chain = compute_contact_forces(cable, 0.0).interface_pressure_Pa
    # Two equal armour layers of P0: interface pressures (2 P0, P0).
    np.testing.assert_allclose(chain, [2 * p0, p0], rtol=1e-12)
    pressure = armour_contact_pressure(cable)
    expected = _two_face(cable)
    assert pressure == pytest.approx(expected, rel=1e-12)
    # P1+P2 = (3 P0, P0), not 2 P0 on each layer.
    assert pressure["inner_armour"] == pytest.approx(3 * p0, rel=1e-12)
    assert pressure["outer_armour"] == pytest.approx(p0, rel=1e-12)


def test_pretension_and_winding_share_one_chain():
    cable = _load(cx.inter_array_three_core_yaml_path())
    # With zero winding stress the winding helper is the pretension-only chain.
    assert armour_contact_pressure(cable) == pytest.approx(
        compute_winding_contact_pressure(cable, {}), rel=0, abs=0
    )
    # Adding winding stress to one layer adds its own jump to the same chain.
    sigma = 50.0e6
    wound = cable.model_copy(
        update={
            "layers": tuple(
                layer.model_copy(update={"winding_stress_Pa": sigma})
                if isinstance(layer, HelicalLayer) and layer.name == "outer_armour"
                else layer
                for layer in cable.layers
            )
        }
    )
    outer = next(x for x in wound.helical_layers if x.name == "outer_armour")
    angle = abs(outer.lay_angle_rad)
    jump = (
        outer.n_wires
        * math.pi
        * outer.d_wire_m**2
        / 4
        * sigma
        * math.sin(angle)
        * math.tan(angle)
        / (2 * math.pi * outer.r_helix_m**2)
    )
    base = armour_contact_pressure(cable)
    with_winding = armour_contact_pressure(wound)
    # Outer layer: jump below it only; inner layer: jump on both faces.
    assert with_winding["outer_armour"] - base["outer_armour"] == pytest.approx(
        jump, rel=1e-9
    )
    assert with_winding["inner_armour"] - base["inner_armour"] == pytest.approx(
        2 * jump, rel=1e-9
    )


def test_tension_increment_is_added_to_the_same_chain_base():
    cable = _load(cx.export_three_core_yaml_path())
    tension = 200.0e3
    now = compute_contact_forces(cable, tension).interface_pressure_Pa
    # The bounded chain at tension already includes the pretension jumps; the
    # adapter must equal its two-face sum (no mixed convention).
    expected = {"inner_armour": now[0] + now[1], "outer_armour": now[1]}
    assert armour_contact_pressure(cable, tension) == pytest.approx(expected, rel=1e-12)


def test_hertz_crossing_load_uses_the_same_interface_pressure():
    cable = _load(cx.inter_array_single_core_yaml_path())
    inner = next(x for x in cable.helical_layers if x.name == "inner_armour")
    radius = inner.r_helix_m + inner.d_wire_m / 2
    shared = compute_contact_forces(cable, 0.0).interface_pressure_Pa[1]
    friction_outer = armour_contact_pressure(cable)["outer_armour"]
    # The outer layer's inner face carries the shared crossing pressure.
    assert friction_outer == pytest.approx(shared, rel=1e-12)
    for model in ("bounded", "qp"):
        loads = armour_crossing_line_loads(cable, contact_model=model)
        partner, line_load = loads["inner_armour"]
        assert partner.name == "outer_armour"
        assert line_load * inner.n_wires == pytest.approx(
            2 * math.pi * radius * shared, rel=1e-9
        )
