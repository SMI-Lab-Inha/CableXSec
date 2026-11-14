# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Exact pressure projection in the radial-chain approximation, not gap closure."""

import math
import warnings

import numpy as np
import pytest

from cablexsec.cables.schema import CableData, CableMeta, HelicalLayer
from cablexsec.mechanics.axisymmetric import compute_contact_forces


@pytest.mark.parametrize("tension", [-1e5, -1e-12, 0.0, 1e-12, 1e5])
def test_identity_pressure_projection_is_exact_at_zero_and_nonnegative(tension):
    layer = HelicalLayer(
        name="armour",
        material="galvanised_steel_armour",
        n_wires=40,
        d_wire_m=0.005,
        r_helix_m=0.08,
        lay_angle_deg=20.0,
    )
    cable = CableData(cable=CableMeta(name="radial projection"), layers=(layer,))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = compute_contact_forces(
            cable, tension, layer_tension_N={"armour": tension}
        )
    rhs = tension * math.tan(math.radians(20)) ** 2 / (2 * math.pi * 0.08**2)
    if tension <= 0:
        np.testing.assert_array_equal(result.interface_pressure_Pa, [0.0])
    else:
        assert result.interface_pressure_Pa[0] == pytest.approx(rhs, rel=1e-14, abs=0)


def test_two_interface_projection_preserves_independent_pressure_jumps():
    layers = tuple(
        HelicalLayer(
            name=name,
            material="galvanised_steel_armour",
            n_wires=40,
            d_wire_m=0.005,
            r_helix_m=radius,
            lay_angle_deg=20.0,
        )
        for name, radius in (("inner", 0.08), ("outer", 0.10))
    )
    cable = CableData(cable=CableMeta(name="two pressure jumps"), layers=layers)
    # Prescribed signed helical forces isolate the existing radial-chain
    # projection; they do not establish a compatible compressed-cable state.
    jumps = [-7000.0, 4000.0]
    tensions = {
        layer.name: jump
        * 2
        * math.pi
        * layer.r_helix_m**2
        / math.tan(layer.lay_angle_rad) ** 2
        for layer, jump in zip(layers, jumps, strict=True)
    }
    result = compute_contact_forces(
        cable, 0.0, layer_tension_N=tensions, p_ext_Pa=1000.0
    )
    np.testing.assert_allclose(
        result.interface_pressure_Pa, [0.0, 5000.0], rtol=1e-14, atol=0
    )
