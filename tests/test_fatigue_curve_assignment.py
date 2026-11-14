# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""A requested fatigue mechanism cannot disappear through a misspelled law."""

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.cable_adapters import cable_to_fatigue_layers
from cablexsec.cables.schema import HelicalLayer
from cablexsec.exceptions import CableSchemaError
from cablexsec.materials import sn_curves


def cable_with_curve(field, name):
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    return cable.model_copy(
        update={
            "layers": [
                layer.model_copy(update={field: name})
                if isinstance(layer, HelicalLayer)
                else layer
                for layer in cable.layers
            ]
        }
    )


@pytest.mark.parametrize("name", ["perier_2009_steel", "perier_typo", "none", " "])
def test_requested_unavailable_fretting_law_is_not_silently_ignored(name):
    with pytest.raises(CableSchemaError, match="unavailable contact-fatigue"):
        cable_to_fatigue_layers(cable_with_curve("fretting_sn", name))


def test_public_fatigue_preserves_the_error():
    with pytest.raises(CableSchemaError, match="unavailable contact-fatigue"):
        cx.compute_fatigue(
            cable_with_curve("fretting_sn", "perier_2009_steel"),
            np.array([0, 0.01]),
            np.zeros(2),
        )


def test_stress_curve_cannot_be_used_as_a_slip_amplitude_curve():
    with pytest.raises(CableSchemaError, match="requires a curve in um"):
        cable_to_fatigue_layers(
            cable_with_curve("fretting_sn", "saevik_2011_flat_steel")
        )


def test_slip_curve_cannot_be_used_as_a_bulk_stress_curve(monkeypatch):
    curve = sn_curves.SNcurve("test_slip", "um", 3.0, 9.0)
    monkeypatch.setitem(sn_curves.SN_CURVES, "test_slip", curve)
    with pytest.raises(CableSchemaError, match="bulk fatigue requires"):
        cable_to_fatigue_layers(cable_with_curve("bulk_sn", "test_slip"))


def test_explicit_custom_slip_curve_remains_available(monkeypatch):
    curve = sn_curves.SNcurve("test_slip", "um", 3.0, 9.0)
    monkeypatch.setitem(sn_curves.SN_CURVES, "test_slip", curve)
    layers = cable_to_fatigue_layers(cable_with_curve("fretting_sn", "test_slip"))
    assert all(layer.sn_fretting is curve for layer in layers)


@pytest.mark.parametrize(
    "path",
    [cx.inter_array_single_core_yaml_path(), cx.inter_array_three_core_yaml_path()],
)
def test_shipped_bulk_only_examples_are_explicit_and_operational(path):
    cable = cx.load_cable(path, geometry_override=None)
    layers = cable_to_fatigue_layers(cable)
    assert all(
        layer.sn_fretting is None and layer.swt_const is None for layer in layers
    )
    result = cx.compute_fatigue(cable, np.array([0, 0.01, -0.01]), np.zeros(3))
    assert any(np.max(value) > 0 for value in result.D_bulk_per_wire.values())
    assert all(np.all(value == 0) for value in result.D_fretting_per_wire.values())
