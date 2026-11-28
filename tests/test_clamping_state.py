# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Manufacturing clamping state as an explicit input (gap 4.1b).

Gates:

* schema: ``manufacturing_clamping`` (one value, basis, source) sets the
  legacy field every contact route reads; conflicts, copies, round trips and
  the geometry override keep the two in step;
* bitwise: a declared block and the equivalent legacy key give identical
  results in the bounded, ``qp`` and solved contact routes, ``compute_mkappa``,
  fatigue, fretting, the rig test and the thermal coupling;
* consistency: the declared clamping reaches every one of those routes;
* warning: ``MissingClampingStateWarning`` for an undeclared state or a
  not-published zero, never for a declared state;
* the packaged cables carry the documented sources;
* the FOWT postprocessor's contact state follows the tension history or the
  declared state, and the declared state reproduces the previous recovery.
"""

from __future__ import annotations

import warnings

import numpy as np
import pydantic
import pytest
import yaml

import cablexsec as cx
from cablexsec.analysis.thermal_coupling import ThermalHistory, armour_thermal_state
from cablexsec.cable_adapters import (
    armour_contact_pressure,
    armour_crossing_line_loads,
    winding_interface_pressure,
)
from cablexsec.cables.schema import (
    CableData,
    HelicalLayer,
    ManufacturingClamping,
    load_cable,
)
from cablexsec.clamping import clamping_state, warn_if_unclamped
from cablexsec.exceptions import CableSchemaError
from cablexsec.mechanics.axisymmetric import compute_winding_contact_pressure
from cablexsec.provenance import MissingClampingStateWarning

WIND = 50e6  # an illustrative winding stress for the equivalence tests


@pytest.fixture(scope="module")
def single() -> CableData:
    return load_cable(cx.inter_array_single_core_yaml_path(), geometry_override=None)


@pytest.fixture(scope="module")
def t66() -> CableData:
    return load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)


def _with_layers(cable: CableData, update) -> CableData:
    layers = tuple(
        update(layer) if isinstance(layer, HelicalLayer) else layer
        for layer in cable.layers
    )
    return cable.model_copy(update={"layers": layers})


def _block(**kw) -> dict:
    return {"basis": "assumed", "source": "test value", **kw}


# --------------------------------------------------------------------------- #
# Schema                                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "data, message",
    [
        ({"basis": "assumed", "source": "x"}, "exactly one"),
        (
            _block(armour_winding_stress_Pa=1e6, initial_contact_pressure_Pa=1e5),
            "exactly one",
        ),
        (_block(armour_winding_stress_Pa=-1.0), ">= 0"),
        (
            {"armour_winding_stress_Pa": 1e6, "basis": "assumed", "source": " "},
            "source",
        ),
        (
            {"armour_winding_stress_Pa": 1e6, "basis": "not_published", "source": "x"},
            "must be zero",
        ),
        ({"armour_winding_stress_Pa": 1e6, "basis": "guess", "source": "x"}, "basis"),
    ],
)
def test_declaration_is_validated(data, message):
    with pytest.raises(pydantic.ValidationError, match=message):
        ManufacturingClamping.model_validate(data)


def test_declaration_sets_the_legacy_field(single):
    layer = single.get_layer("inner_armour")
    assert isinstance(layer, HelicalLayer)
    declared = layer.manufacturing_clamping
    assert declared is not None
    assert declared.kind == "contact_pressure"
    assert declared.basis == "representative"
    assert layer.pretension_Pa == declared.value_Pa == 2.0e6
    assert layer.winding_stress_Pa is None
    data = layer.model_dump()
    data["manufacturing_clamping"] = _block(armour_winding_stress_Pa=WIND)
    data["pretension_Pa"] = None
    wound = HelicalLayer.model_validate(data)
    assert wound.winding_stress_Pa == WIND and wound.pretension_Pa is None
    # A legacy value that agrees is accepted, a different one is rejected.
    assert HelicalLayer.model_validate({**data, "winding_stress_Pa": WIND})
    with pytest.raises(pydantic.ValidationError, match="conflicts"):
        HelicalLayer.model_validate({**data, "winding_stress_Pa": 2 * WIND})
    with pytest.raises(pydantic.ValidationError, match="conflicts"):
        HelicalLayer.model_validate({**data, "pretension_Pa": 1e5})


def test_model_copy_keeps_declaration_and_legacy_field_in_step(single):
    layer = single.get_layer("inner_armour")
    # Updating the legacy field (sensitivity studies) drops the declaration.
    legacy = layer.model_copy(update={"pretension_Pa": 3e6})
    assert legacy.manufacturing_clamping is None and legacy.pretension_Pa == 3e6
    # Updating the declaration sets the legacy field.
    wound = layer.model_copy(
        update={"manufacturing_clamping": _block(armour_winding_stress_Pa=WIND)}
    )
    assert wound.winding_stress_Pa == WIND and wound.pretension_Pa is None
    assert wound.manufacturing_clamping.kind == "winding_stress"
    with pytest.raises(ValueError, match="not both"):
        layer.model_copy(
            update={
                "manufacturing_clamping": _block(armour_winding_stress_Pa=WIND),
                "pretension_Pa": 1.0,
            }
        )
    with pytest.raises(pydantic.ValidationError):
        layer.model_copy(
            update={"manufacturing_clamping": _block(armour_winding_stress_Pa=-1.0)}
        )
    # Other updates leave both untouched.
    moved = layer.model_copy(update={"r_helix_m": 0.025})
    assert moved.manufacturing_clamping == layer.manufacturing_clamping


def test_round_trip_through_a_dump(single):
    again = CableData.model_validate(single.model_dump(mode="json"))
    assert again == single
    assert cx.CableModel(single).definition == single


def test_override_legacy_value_replaces_the_template_declaration(tmp_path):
    with open(cx.inter_array_single_core_yaml_path(), encoding="utf-8") as fh:
        template = yaml.safe_load(fh)
    path = tmp_path / "cable.yaml"
    path.write_text(yaml.safe_dump(template), encoding="utf-8")
    override = tmp_path / "override.yaml"
    override.write_text(
        yaml.safe_dump(
            {"layers": [{"name": "inner_armour", "winding_stress_Pa": WIND}]}
        ),
        encoding="utf-8",
    )
    cable = load_cable(path, geometry_override=override)
    layer = cable.get_layer("inner_armour")
    assert layer.winding_stress_Pa == WIND
    assert layer.pretension_Pa is None and layer.manufacturing_clamping is None
    override.write_text(
        yaml.safe_dump(
            {
                "layers": [
                    {
                        "name": "inner_armour",
                        "manufacturing_clamping": _block(armour_winding_stress_Pa=WIND),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    layer = load_cable(path, geometry_override=override).get_layer("inner_armour")
    assert layer.winding_stress_Pa == WIND and layer.pretension_Pa is None
    assert layer.manufacturing_clamping.basis == "assumed"


# --------------------------------------------------------------------------- #
# Packaged cables and the warning                                             #
# --------------------------------------------------------------------------- #


def test_legacy_values_are_undocumented_and_legacy_zeros_missing(single):
    legacy = _with_layers(single, lambda x: x.model_copy(update={"pretension_Pa": 1e6}))
    state = clamping_state(legacy)
    assert {x.basis for x in state.layers} == {"undocumented"}
    assert state.warning_message() is None
    zero = _with_layers(
        single,
        lambda x: x.model_copy(update={"pretension_Pa": 0, "winding_stress_Pa": 0}),
    )
    assert clamping_state(zero).missing_layers == ("inner_armour", "outer_armour")
    with pytest.warns(MissingClampingStateWarning, match="declares no"):
        warn_if_unclamped(zero)
    assert (
        "winding stress"
        in clamping_state(
            _with_layers(
                single, lambda x: x.model_copy(update={"winding_stress_Pa": 1e6})
            )
        ).describe()
    )


def _caught(fn) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [w for w in caught if issubclass(w.category, MissingClampingStateWarning)]


def test_friction_dependent_analyses_warn_only_without_a_declaration(t66, single):
    kappa = 0.01 * np.sin(np.linspace(0.0, 4 * np.pi, 41))
    t = np.arange(40.0)
    kx, ky = 0.004 * np.sin(t), 0.001 * np.cos(t)
    tension = np.full(t.size, 50e3)
    proto = cx.RigProtocol.cyclic(
        tension_N=50e3, curvature_amplitude_1_m=0.01, n_cycles=1, samples_per_cycle=8
    )
    setup = cx.RigSetup(free_length_m=4.0, clamp="none")
    calls = {
        "mkappa": lambda c: cx.compute_mkappa(c, kappa, tension_N=50e3),
        "fatigue": lambda c: cx.compute_fatigue(c, kx, ky, tension_N=tension),
        "fretting": lambda c: cx.compute_fretting(
            c, kx, ky, tension_N=tension, duration_s=40.0
        ),
        "rig": lambda c: cx.simulate_rig_test(
            c, proto, setup, gauges=[], mu_factors=(), contact_model="solved"
        ),
        "thermal": lambda c: armour_thermal_state(
            c, ThermalHistory.uniform(c, [0.0], 30.0).blocks()[0].state
        ),
    }
    for name, call in calls.items():
        assert _caught(lambda call=call: call(t66)), name
        assert not _caught(lambda call=call: call(single)), name
    menard = load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    assert not _caught(lambda: cx.compute_mkappa(menard, kappa, tension_N=1e4))


# --------------------------------------------------------------------------- #
# Bitwise equivalence and consistency                                         #
# --------------------------------------------------------------------------- #
def _results(cable: CableData) -> dict[str, np.ndarray]:
    t = np.arange(60.0)
    kx = 0.004 * np.sin(2 * np.pi * t / 10)
    ky = 0.001 * np.sin(2 * np.pi * t / 13)
    tension = 30e3 + 5e3 * np.sin(2 * np.pi * t / 10)
    kappa = 0.02 * np.sin(np.linspace(0.0, 4 * np.pi, 81))
    out: dict[str, np.ndarray] = {}
    for model in ("bounded", "qp", "solved"):
        for ten in (0.0, 30e3):
            p = armour_contact_pressure(cable, ten, contact_model=model)
            f = armour_crossing_line_loads(cable, ten, contact_model=model)
            out[f"p.{model}.{ten}"] = np.array(sorted(p.items()), dtype=object)
            out[f"f.{model}.{ten}"] = np.array([v[1] for _, v in sorted(f.items())])
        m = cx.compute_mkappa(cable, kappa, tension_N=20e3, contact_model=model)
        out[f"m.{model}"] = np.asarray(m.moment_Nm)
        fat = cx.compute_fatigue(cable, kx, ky, tension_N=tension, contact_model=model)
        for name, d in fat.D_total_per_wire.items():
            out[f"d.{model}.{name}"] = np.asarray(d)
        # Zero tension: friction from the manufacturing clamping alone.
        fat = cx.compute_fatigue(cable, 3 * kx, 3 * ky, contact_model=model)
        for name, d in fat.D_total_per_wire.items():
            out[f"d0.{model}.{name}"] = np.asarray(d)
    fret = cx.compute_fretting(cable, kx, ky, tension_N=tension, duration_s=60.0)
    for name, face in fret.faces.items():
        out[f"fret.{name}"] = np.asarray(face.max_amplitude_m)
        out[f"fretload.{name}"] = np.array([face.contact.normal_load])
    rig = cx.simulate_rig_test(
        cable,
        cx.RigProtocol.cyclic(
            tension_N=30e3,
            curvature_amplitude_1_m=0.01,
            n_cycles=1,
            samples_per_cycle=8,
        ),
        cx.RigSetup(free_length_m=4.0, clamp="none"),
        gauges=[cx.GaugeLocation("inner_armour", 0.0, 2.0)],
        mu_factors=(),
        contact_model="solved",
    )
    out["rig"] = np.asarray(rig.gauges[0].total)
    for name, a in armour_thermal_state(
        cable,
        ThermalHistory.uniform(cable, [0.0], 45.0).blocks()[0].state,
        tension_N=30e3,
    ).items():
        out[f"thermal.{name}"] = np.array(
            [a.contact_pressure_Pa, a.thermal_wire_stress_Pa, a.friction_cap_Pa]
        )
    return out


@pytest.mark.parametrize(
    "block, legacy",
    [
        ({"initial_contact_pressure_Pa": 2.0e6}, {"pretension_Pa": 2.0e6}),
        ({"armour_winding_stress_Pa": WIND}, {"winding_stress_Pa": WIND}),
    ],
)
def test_declaration_and_legacy_key_are_bitwise_identical(single, block, legacy):
    declared = _with_layers(
        single,
        lambda x: x.model_copy(update={"manufacturing_clamping": _block(**block)}),
    )
    old = _with_layers(
        single,
        lambda x: x.model_copy(
            update={"pretension_Pa": None, "winding_stress_Pa": None, **legacy}
        ),
    )
    assert old.helical_layers[0].manufacturing_clamping is None
    a, b = _results(declared), _results(old)
    assert a.keys() == b.keys()
    for key in a:
        assert np.array_equal(a[key], b[key]), key


def test_declared_clamping_reaches_every_route(single):
    zero = _with_layers(
        single,
        lambda x: x.model_copy(
            update={
                "manufacturing_clamping": {
                    "armour_winding_stress_Pa": 0.0,
                    "basis": "not_published",
                    "source": "test",
                }
            }
        ),
    )
    wound = _with_layers(
        single,
        lambda x: x.model_copy(
            update={"manufacturing_clamping": _block(armour_winding_stress_Pa=WIND)}
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingClampingStateWarning)
        a, b = _results(zero), _results(wound)
    for model in ("bounded", "solved"):
        p0 = dict(a[f"p.{model}.0.0"])
        p1 = dict(b[f"p.{model}.0.0"])
        # Zero clamping at zero tension: no pressure (solver round-off only).
        assert all(abs(p0[k]) < 1e-3 and p1[k] > 1e5 for k in p1), model
    # The bounded face split gives the outermost armour no outer-face (sheath)
    # pressure whatever the clamping, so only the inner-armour faces count.
    routes = ("m.", "d0.", "rig", "thermal", "fretload.inner_armour", "f.")
    same = [
        key for key in a if key.startswith(routes) and np.array_equal(a[key], b[key])
    ]
    assert not same


def test_winding_interface_pressure_matches_the_two_face_chain(single):
    wound = _with_layers(
        single,
        lambda x: x.model_copy(
            update={"pretension_Pa": None, "winding_stress_Pa": WIND}
        ),
    )
    iface = winding_interface_pressure(wound)
    assert iface[0] > iface[1] > 0.0
    p1p2 = compute_winding_contact_pressure(wound, WIND)
    assert p1p2["inner_armour"] == pytest.approx(iface[0] + iface[1], rel=1e-12)
    assert p1p2["outer_armour"] == pytest.approx(iface[1], rel=1e-12)
    assert not np.any(winding_interface_pressure(single))


# --------------------------------------------------------------------------- #
# Contact histories and the FOWT postprocessor                                #
# --------------------------------------------------------------------------- #
def test_armour_contact_histories_follow_the_tension(t66):
    tension = np.linspace(0.0, 80e3, 9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingClampingStateWarning)
        pressure, force = cx.armour_contact_histories(t66, tension)
        assert pressure["inner_armour"][0] == 0.0
        assert np.all(np.diff(pressure["inner_armour"]) > 0)
        for i, value in enumerate(tension):
            ref = armour_contact_pressure(t66, float(value))
            assert pressure["inner_armour"][i] == pytest.approx(
                ref["inner_armour"], rel=1e-12, abs=1e-9
            )
    with pytest.raises(CableSchemaError, match="non-negative"):
        cx.armour_contact_histories(t66, -tension)
    with pytest.raises(CableSchemaError, match="contact_model"):
        cx.armour_contact_histories(t66, tension, contact_model="x")


def test_packaged_cables_carry_their_documented_clamping():
    for path, value in (
        (cx.inter_array_single_core_yaml_path(), 2.0e6),
        (cx.inter_array_three_core_yaml_path(), 2.0e6),
        (cx.export_three_core_yaml_path(), 3.0e6),
    ):
        state = clamping_state(load_cable(path, geometry_override=None))
        assert {(x.kind, x.contact_pressure_Pa, x.basis) for x in state.layers} == {
            ("contact_pressure", value, "representative")
        }
        assert state.warning_message() is None
    menard = clamping_state(
        load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    )
    assert menard.missing_layers == ("inner_armour", "outer_armour")
    assert menard.has_initial_state and menard.is_declared
    assert menard.initial_identification == "published_calibration"
    assert menard.warning_message() is None
    t66 = clamping_state(
        load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
    )
    assert t66.is_zero and t66.is_declared
    assert {x.basis for x in t66.layers} == {"not_published"}
    assert "not published" in t66.warning_message()
    tb = clamping_state(
        load_cable(cx.poradowski2025_220kv_yaml_path(), geometry_override=None)
    )
    assert not tb.is_declared and "declares no" in tb.warning_message()
