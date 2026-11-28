# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Transient thermal network: environment overrides, source shares and limits."""

from __future__ import annotations

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.cables.schema import CableData
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.thermal_network import thermal_network

MATERIAL = "copper_janocha2024"
RATED_A = 1093.0  # [IEATask49] Table 13, 630 mm2


@pytest.fixture(scope="module")
def t66() -> CableData:
    return cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)


def test_environment_overrides_equal_the_cable_environment(t66):
    env = t66.environment
    assert env.T4_external_Km_W is not None and env.ambient_C is not None
    bare = t66.model_copy(
        update={"environment": env.model_copy(update={"T4_external_Km_W": None})}
    )
    with pytest.raises(CableInputError, match="needs T4_external_Km_W and ambient_C"):
        thermal_network(bare, 630e-6, conductor_material=MATERIAL)
    reference = thermal_network(t66, 630e-6, conductor_material=MATERIAL)
    override = thermal_network(
        bare,
        630e-6,
        conductor_material=MATERIAL,
        T4_external_Km_W=env.T4_external_Km_W,
        ambient_C=env.ambient_C,
    )
    np.testing.assert_allclose(
        override.steady_state(RATED_A), reference.steady_state(RATED_A), rtol=1e-12
    )


def test_source_weights_sum_to_one_and_name_unknown_layers(t66):
    net = thermal_network(t66, 630e-6, conductor_material=MATERIAL)
    for name in ("inner_armour", "outer_armour", "xlpe_insulation"):
        weights = net.source_weights(name)
        assert weights.sum() == pytest.approx(1.0, rel=1e-14)
        assert np.all(weights >= 0.0)
    with pytest.raises(CableInputError, match="unknown layer 'no_such_layer'"):
        net.source_weights("no_such_layer")


def test_periodic_regime_needs_enough_periods(t66):
    net = thermal_network(t66, 630e-6, conductor_material=MATERIAL)
    t = np.linspace(0.0, 6 * 3600.0, 7)
    current = RATED_A * np.array([0.2, 0.6, 1.0, 1.0, 0.6, 0.4, 0.2])
    # Starting cold, one six-hour period cannot be periodic to a micro-kelvin.
    with pytest.raises(CableInputError, match="periodic thermal regime not reached"):
        net.transient(
            t,
            current,
            initial="ambient",
            periodic=True,
            periodic_tolerance_K=1e-6,
            max_periods=1,
            max_step_s=1800.0,
        )
    bad = np.zeros(net.n_nodes + 1)
    with pytest.raises(CableInputError, match="do not match the network"):
        net.transient(t, current, initial=bad)


def test_armour_loss_needs_a_structural_armour(t66):
    loose = t66.model_copy(
        update={
            "layers": tuple(
                layer.model_copy(update={"structural": False})
                if layer.layer_type == "helical"
                else layer
                for layer in t66.layers
            )
        }
    )
    with pytest.raises(CableInputError, match="lambda2 > 0 needs a structural"):
        thermal_network(loose, 630e-6, conductor_material=MATERIAL, lambda2=0.1)
    # Without structural armour the SL-type filler factor and the magnetic
    # armour of the sheath loss ([IEC60287-1-1] 5.3.11) do not apply, and the
    # network says so.
    net = thermal_network(loose, 630e-6, conductor_material=MATERIAL)
    notes = " ".join(net.provenance.limitations)
    assert "given for an armoured three-core cable" in notes
    assert "5.3.2" in notes
