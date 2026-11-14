# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Public API: input checks and option routes of the fatigue and fretting calls."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.thermal_coupling import ThermalState
from cablexsec.api import armour_contact_histories
from cablexsec.cables.schema import ManufacturingClamping
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.materials.sn_curves import SNcurve

K = 0.05 * np.sin(np.linspace(0.0, 4.0 * math.pi, 81))
Z = np.zeros_like(K)


def _layer(cable, name, **update):
    layers = [
        layer.model_copy(update=update) if layer.name == name else layer
        for layer in cable.layers
    ]
    return cable.model_copy(update={"layers": layers})


@pytest.fixture(scope="module")
def export():
    return cx.load_cable(cx.export_three_core_yaml_path(), geometry_override=None)


@pytest.fixture(scope="module")
def menard():
    return cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)


@pytest.fixture(scope="module")
def stranded():
    return cx.iea_task49_66kv_stranded()


@pytest.fixture(scope="module")
def iea66():
    return cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# --------------------------------------------------------------------------- #
# Shell fatigue                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"method": "fracture"}, "method must be"),
        ({"sn_curve": 5}, "stress-range S-N curve in MPa"),
        ({"static_mean_stress_Pa": math.inf}, "static_mean_stress_Pa must be finite"),
    ],
)
def test_shell_fatigue_checks(export, kwargs, match):
    k = np.array([-0.01, 0.01])
    with pytest.raises(CableInputError, match=match):
        cx.compute_shell_fatigue(export, "al_metallic_sheath", k, **kwargs)


def test_shell_strain_life_with_out_of_plane_and_axial_histories(export):
    k = 0.4 * np.sin(np.linspace(0.0, 4.0 * math.pi, 81))
    options = {"method": "strain_life", "strain_life": "aluminium_foil_mus_assumed"}
    for item in ("kappa_y", "axial_strain"):
        with pytest.raises(CableSchemaError, match=f"{item} must match"):
            cx.compute_shell_fatigue(
                export, "al_metallic_sheath", k, **options, **{item: k[:-1]}
            )
    planar = cx.compute_shell_fatigue(export, "al_metallic_sheath", k, **options)
    both = cx.compute_shell_fatigue(
        export,
        "al_metallic_sheath",
        k,
        kappa_y=0.5 * k,
        axial_strain=np.full_like(k, 1e-4),
        **options,
    )
    assert both.D_miner > planar.D_miner
    assert "constants are assumed" in " ".join(both.provenance.limitations)


# --------------------------------------------------------------------------- #
# Conductor fatigue                                                           #
# --------------------------------------------------------------------------- #
def test_conductor_fatigue_history_checks(stranded):
    with pytest.raises(CableSchemaError, match="equal 1-D histories"):
        cx.compute_conductor_fatigue(stranded, K, K[:-1])
    with pytest.raises(CableSchemaError, match="tension_N must match"):
        cx.compute_conductor_fatigue(stranded, K, Z, tension_N=np.ones(3))


def test_conductor_fatigue_curves(stranded):
    bare = _layer(stranded, "cu_conductor", bulk_sn="")
    with pytest.raises(CableSchemaError, match="has no bulk_sn curve"):
        cx.compute_conductor_fatigue(bare, K, Z, layer_name="cu_conductor")
    for curve in ("perier_2009_steel", "no_such_curve"):
        with pytest.raises(CableSchemaError, match="bulk S-N unavailable"):
            cx.compute_conductor_fatigue(stranded, K, Z, sn_curve=curve)
    strain_curve = SNcurve(name="in_pa", stress_unit="um", m1=3.0, log_C1=12.0)
    with pytest.raises(CableSchemaError, match="stress-range curve"):
        cx.compute_conductor_fatigue(stranded, K, Z, sn_curve=strain_curve)
    named = cx.compute_conductor_fatigue(
        bare, K, Z, layer_name="cu_conductor", sn_curve="nasution_2013_copper_corrected"
    )
    assert named.critical_ring >= 0


def test_conductor_fatigue_laws(stranded):
    wrong = _layer(stranded, "cu_conductor", fretting_sn="steel_unknown")
    with pytest.raises(CableSchemaError, match="registered SWT strain-life key"):
        cx.compute_conductor_fatigue(wrong, K, Z)
    with pytest.raises(CableSchemaError, match="trellis_model"):
        cx.compute_conductor_fatigue(stranded, K, Z, trellis_model="contact")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Armour fatigue options                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sn_curves": ["dnv_rpc203_t_curve"]}, "sn_curves must map"),
        ({"sn_curves": {"inner_armour": 5}}, "names or SNcurve"),
        ({"mean_stress": 5}, "MeanStressSpec or a name"),
        ({"scf": 5}, "scf must be a ComponentSCF"),
        ({"contact_model": "solved", "section_states": []}, "section states belong"),
        (
            {
                "contact_model": "solved",
                "thermal": "thermal",
                "axial_strain": Z,
            },
            "thermal coupling takes tension_N only",
        ),
    ],
)
def test_fatigue_option_checks(menard, kwargs, match):
    if kwargs.get("thermal") == "thermal":
        kwargs["thermal"] = ThermalState({layer.name: 30.0 for layer in menard.layers})
    with pytest.raises(CableSchemaError, match=match):
        cx.compute_fatigue(menard, K, Z, **kwargs)


def test_per_layer_curve_leaves_the_other_layer_unchanged(menard):
    base = cx.compute_fatigue(menard, K, Z)
    one = cx.compute_fatigue(
        menard, K, Z, sn_curves={"inner_armour": "dnv_rpc203_t_curve"}
    )
    np.testing.assert_array_equal(
        one.D_total_per_wire["outer_armour"], base.D_total_per_wire["outer_armour"]
    )
    assert not np.array_equal(
        one.D_total_per_wire["inner_armour"], base.D_total_per_wire["inner_armour"]
    )


def test_armour_contact_history_checks(menard):
    with pytest.raises(CableSchemaError, match="non-negative 1-D"):
        armour_contact_histories(menard, np.array([1e4, -1.0]))
    with pytest.raises(CableSchemaError, match="contact_model"):
        armour_contact_histories(menard, np.array([1e4]), contact_model="exact")


# --------------------------------------------------------------------------- #
# Fretting                                                                    #
# --------------------------------------------------------------------------- #
def test_fretting_is_for_round_wires():
    rect = cx.load_cable(
        cx.saevik2011_yaml_path(tendons="rectangular"), geometry_override=None
    )
    with pytest.raises(CableSchemaError, match="round wires"):
        cx.compute_fretting(rect, K, Z, duration_s=10.0)


@pytest.mark.parametrize(
    ("args", "kwargs", "match"),
    [
        ((K, K[:-1]), {"duration_s": 10.0}, "equal 1-D histories"),
        ((K, Z), {"duration_s": 0.0}, "duration_s"),
        ((K, Z), {"duration_s": 10.0, "tension_N": Z, "axial_strain": Z}, "not both"),
        ((K, Z), {"duration_s": 10.0, "tension_N": np.ones(3)}, "tension_N must match"),
    ],
)
def test_fretting_input_checks(menard, args, kwargs, match):
    with pytest.raises(CableSchemaError, match=match):
        cx.compute_fretting(menard, *args, **kwargs)


def test_fretting_between_touching_armours_and_options():
    # Opposite-laid armours in direct contact: one crossed-cylinder point
    # force, the same seen from either layer (bulk-only cable: no fretting_sn).
    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    assert not any(x.fretting_sn for x in cable.helical_layers)
    result = cx.compute_fretting(
        cable,
        K,
        Z,
        duration_s=10.0,
        knockdown="steel_nishioka_hirakawa",
        transverse_ratio=0.1,
    )
    inner = result.faces["inner_armour/outer_face"].contact
    outer = result.faces["outer_armour/inner_face"].contact
    assert inner.kind == outer.kind == "point"
    assert inner.normal_load > 0
    assert inner.normal_load == pytest.approx(outer.normal_load, rel=1e-12)
    assert not any("no crossing load" in x for x in result.provenance.limitations)


def test_fretting_reports_faces_without_partners():
    cable = cx.load_cable(cx.saevik2011_yaml_path(), geometry_override=None)
    result = cx.compute_fretting(cable, K, Z, duration_s=10.0)
    assert any("no touching layer" in x for x in result.provenance.limitations)


def test_fretting_includes_the_winding_clamping(iea66):
    clamping = ManufacturingClamping(
        armour_winding_stress_Pa=60e6, basis="assumed", source="test value"
    )
    wound = _layer(iea66, "inner_armour", manufacturing_clamping=clamping)
    plain = cx.compute_fretting(iea66, K, Z, duration_s=10.0)
    clamped = cx.compute_fretting(wound, K, Z, duration_s=10.0)
    key = "inner_armour/inner_face"
    assert clamped.faces[key].contact.normal_load > plain.faces[key].contact.normal_load
