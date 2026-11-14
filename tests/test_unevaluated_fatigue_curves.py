# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Declared S-N curves that compute_fatigue does not evaluate are reported.

compute_fatigue assesses only structural helical armour. A ``bulk_sn`` or
``fretting_sn`` on a conductor or sheath, or an armour ``fretting_sn`` with no
direct crossed-wire contact, contributes no damage. The calculation must say
so (warning and provenance) without changing any damage value.
"""

import warnings

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.cables.schema import HelicalLayer
from cablexsec.materials import sn_curves
from cablexsec.provenance import UnevaluatedFatigueCurveWarning

KAPPA = 0.01 * np.sin(np.linspace(0.0, 8.0 * np.pi, 161))


def _with_layer_update(cable, name, **update):
    layers = tuple(
        layer.model_copy(update=update) if layer.name == name else layer
        for layer in cable.layers
    )
    return cable.model_copy(update={"layers": layers})


def _unevaluated(record):
    return [
        str(item.message)
        for item in record
        if issubclass(item.category, UnevaluatedFatigueCurveWarning)
    ]


def _assert_same_damage(a, b):
    for field in ("D_bulk_per_wire", "D_fretting_per_wire", "D_total_per_wire"):
        for name, values in getattr(a, field).items():
            np.testing.assert_array_equal(values, getattr(b, field)[name])


def test_warning_class_is_public():
    assert cx.UnevaluatedFatigueCurveWarning is UnevaluatedFatigueCurveWarning
    assert issubclass(UnevaluatedFatigueCurveWarning, UserWarning)
    assert "UnevaluatedFatigueCurveWarning" in cx.__all__


def test_no_warning_when_every_declared_curve_is_evaluated():
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = cx.compute_fatigue(cable, KAPPA, np.zeros_like(KAPPA))
    assert _unevaluated(record) == []
    assert not any("not evaluated" in note for note in result.provenance.limitations)


def test_conductor_curves_warn_and_are_recorded_without_changing_damage():
    base = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    cable = _with_layer_update(
        base,
        "cu_conductor",
        bulk_sn="nasution_2014_copper",
        fretting_sn="poon_2022_swt",
    )
    with pytest.warns(UnevaluatedFatigueCurveWarning) as record:
        result = cx.compute_fatigue(cable, KAPPA, np.zeros_like(KAPPA))
    messages = _unevaluated(record)
    assert len(messages) == 1
    assert "'cu_conductor'" in messages[0]
    assert "nasution_2014_copper" in messages[0]
    assert "poon_2022_swt" in messages[0]
    assert messages[0] in result.provenance.limitations
    reference = cx.compute_fatigue(base, KAPPA, np.zeros_like(KAPPA))
    _assert_same_damage(result, reference)


def test_sheath_curve_warns_and_points_to_shell_fatigue():
    cable = cx.load_cable(cx.export_three_core_yaml_path(), geometry_override=None)
    with pytest.warns(UnevaluatedFatigueCurveWarning) as record:
        result = cx.compute_fatigue(cable, KAPPA, np.zeros_like(KAPPA))
    messages = _unevaluated(record)
    assert len(messages) == 1
    assert "'al_metallic_sheath'" in messages[0]
    assert "compute_shell_fatigue" in messages[0]
    assert messages[0] in result.provenance.limitations


def test_armour_contact_law_without_crossing_warns(monkeypatch):
    curve = sn_curves.SNcurve("test_slip", "um", 3.0, 9.0)
    monkeypatch.setitem(sn_curves.SN_CURVES, "test_slip", curve)
    base = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    # Menard's armour layers are separated by bedding: no direct crossing.
    cable = _with_layer_update(base, "inner_armour", fretting_sn="test_slip")
    with pytest.warns(UnevaluatedFatigueCurveWarning) as record:
        result = cx.compute_fatigue(cable, KAPPA, np.zeros_like(KAPPA))
    messages = _unevaluated(record)
    assert len(messages) == 1
    assert "'inner_armour'" in messages[0]
    assert "no direct crossed-wire contact" in messages[0]
    assert np.all(result.D_fretting_per_wire["inner_armour"] == 0.0)


def test_section_state_path_also_warns():
    base = cx.load_cable(cx.inter_array_single_core_yaml_path(), geometry_override=None)
    cable = _with_layer_update(base, "cu_conductor", bulk_sn="nasution_2014_copper")
    state = cx.solve_section_state(cable, tension_N=1.0e4)
    kappa = KAPPA[:21]
    with pytest.warns(UnevaluatedFatigueCurveWarning) as record:
        result = cx.compute_fatigue(
            cable, kappa, np.zeros_like(kappa), section_states=[state] * kappa.size
        )
    assert any("'cu_conductor'" in message for message in _unevaluated(record))
    assert any("'cu_conductor'" in note for note in result.provenance.limitations)


def test_nonstructural_helix_curves_warn():
    base = cx.load_cable(cx.inter_array_single_core_yaml_path(), geometry_override=None)
    cable = _with_layer_update(base, "outer_armour", structural=False)
    assert isinstance(
        next(x for x in cable.layers if x.name == "outer_armour"), HelicalLayer
    )
    with pytest.warns(UnevaluatedFatigueCurveWarning) as record:
        result = cx.compute_fatigue(cable, KAPPA, np.zeros_like(KAPPA))
    assert any("'outer_armour'" in message for message in _unevaluated(record))
    assert "outer_armour" not in result.D_total_per_wire
