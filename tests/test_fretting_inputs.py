# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Fretting regime and assessment: contact-law limits, checks and summaries."""

from __future__ import annotations

import math
import warnings
from dataclasses import replace

import numpy as np
import pytest

import cablexsec as cx
import cablexsec.analysis.fretting_regime as fr
from cablexsec.analysis.fretting_assessment import (
    FaceSpec,
    conductor_trellis_fretting,
    helical_layer_fretting,
)
from cablexsec.cable_adapters import cable_to_fatigue_layers
from cablexsec.exceptions import CableInputError
from cablexsec.materials.sn_curves import SNcurve, get_sn_curve
from cablexsec.materials.wear import get_knockdown, get_wear_coefficient
from cablexsec.mechanics.strand_slip import ConductorRing

D = np.array([0.0, 1e-6, 1e-5])


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _contact():
    return fr.point_contact("p", 10.0, 50.0, 80.0, 210e9, 0.3, 210e9, 0.3, 0.5)


# --------------------------------------------------------------------------- #
# Contact laws                                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((math.nan, 10.0, 1e6), "finite"),
        ((0.5, math.inf, 1e6), "finite"),
        ((0.5, -1.0, 1e6), "nonnegative"),
        ((0.5, 10.0, 0.0), "positive stiffness"),
    ],
)
def test_transition_amplitude_checks(args, match):
    with pytest.raises(CableInputError, match=match):
        fr.mindlin_transition_amplitude(*args)


def test_zero_transition_amplitude_is_gross_slip():
    np.testing.assert_array_equal(fr.mindlin_load_ratio(D, 0.0), [0.0, 1.0, 1.0])
    np.testing.assert_allclose(
        fr.mindlin_cycle_energy(D, 0.5, 10.0, 0.0), 4.0 * 0.5 * 10.0 * D
    )
    a, d, b = fr.fouvry_ratios(D, 0.0)
    np.testing.assert_array_equal(a, [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(d, [0.0, 1.0, 1.0])
    stick = fr.fouvry_ratios(D, math.inf)
    assert all(np.all(x == 0.0) for x in stick)
    assert np.all(fr.cattaneo_line_energy(D, math.inf, 1e3, 1e11) == 0.0)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"kind": "area"}, "unknown contact kind"),
        ({"law": "coulomb"}, "unknown contact kind"),
        ({"mu": -0.1}, "nonnegative"),
        ({"normal_load": -1.0}, "finite, >= 0"),
        ({"transition_amplitude_m": math.nan}, "transition amplitude"),
    ],
)
def test_contact_checks(changes, match):
    with pytest.raises(CableInputError, match=match):
        replace(_contact(), **changes)


def test_line_contact_limits():
    args = ("l", 1e3, 1e-3, 210e9, 0.3, 1e9, 0.4, 0.3, 1e8)
    with pytest.raises(CableInputError, match="line load"):
        fr.line_contact("l", -1.0, *args[2:])
    with pytest.raises(CableInputError, match="support stiffness"):
        fr.line_contact(*args[:-1], 0.0)
    empty = fr.line_contact("l", 0.0, *args[2:])
    assert empty.normal_load == 0.0 and empty.transition_amplitude_m == math.inf
    stuck = fr.line_contact(*args[:7], math.inf, args[8])
    assert stuck.transition_amplitude_m == math.inf
    np.testing.assert_array_equal(stuck.load_ratio(D), 0.0)
    sliding = replace(fr.line_contact(*args), transition_amplitude_m=0.0)
    np.testing.assert_array_equal(sliding.load_ratio(D), [0.0, 1.0, 1.0])


def test_assess_contact_paths():
    contact = _contact()
    with pytest.raises(CableInputError, match="real and finite"):
        fr.assess_contact(np.array([0.0, 1j]), contact)
    with pytest.raises(CableInputError, match="shape"):
        fr.assess_contact(np.zeros((4, 3)), contact)
    with pytest.raises(CableInputError, match="mixed_fraction"):
        fr.assess_contact(np.zeros(4), contact, mixed_fraction=0.6)
    single = fr.assess_contact(np.array([1e-6]), contact)
    assert single.count.size == 0 and np.sum(single.count_by_regime) == 0.0


def test_knockdown_inputs():
    contact = _contact()
    curve = get_sn_curve("dnv_rpc203_t_curve")
    law = get_knockdown("steel_nishioka_hirakawa")
    stress = 1e8 * np.sin(np.linspace(0.0, 8.0 * math.pi, 81))
    slip = 1e-5 * np.sin(np.linspace(0.0, 8.0 * math.pi, 81))
    with pytest.raises(CableInputError, match="MPa"):
        fr.range_at_cycles(
            SNcurve(name="u", stress_unit="um", m1=3.0, log_C1=12.0), 1e6
        )
    with pytest.raises(CableInputError, match="one-dimensional"):
        fr.fretting_fatigue_damage(stress[None, :], slip, contact, curve, law)
    with pytest.raises(CableInputError, match="section_loss_factor"):
        fr.fretting_fatigue_damage(
            stress, slip, contact, curve, law, section_loss_factor=0.5
        )
    with pytest.raises(CableInputError, match="must match"):
        fr.fretting_fatigue_damage(stress, slip[:-1], contact, curve, law)
    planar = np.column_stack((slip, 0.5 * slip))
    result = fr.fretting_fatigue_damage(stress, planar, contact, curve, law)
    line = fr.fretting_fatigue_damage(
        stress, slip * math.hypot(1.0, 0.5), contact, curve, law
    )
    assert result.damage_fretting == pytest.approx(line.damage_fretting, rel=1e-9)


# --------------------------------------------------------------------------- #
# Assessment                                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("changes", "match"),
    [({"kind": "area"}, "line' or 'point"), ({"pressure_Pa": -1.0}, "finite, >= 0")],
)
def test_face_checks(changes, match):
    base = {
        "name": "f",
        "angle_rad": 0.0,
        "kind": "line",
        "partner": "p",
        "partner_E_Pa": 1e9,
        "partner_nu": 0.4,
        "mu": 0.2,
        "wear": None,
        "pressure_Pa": 1e5,
    }
    with pytest.raises(CableInputError, match=match):
        FaceSpec(**{**base, **changes})


def test_layer_fretting_checks_and_section_loss():
    cable = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
    layer = cable_to_fatigue_layers(cable)[0]
    k = 0.05 * np.sin(np.linspace(0.0, 4.0 * math.pi, 81))
    face = FaceSpec("inner_face", math.pi, "line", "p", 1e9, 0.4, 0.2, None, 1e5)
    for kwargs, match in (
        ({"duration_s": 0.0}, "duration_s"),
        ({"duration_s": 1.0, "time_fraction": -1.0}, "time_fraction"),
        ({"duration_s": 1.0, "section_loss": True}, "design life"),
    ):
        with pytest.raises(CableInputError, match=match):
            helical_layer_fretting(layer, [face], k, 0 * k, nu_wire=0.3, **kwargs)
    worn = cx.compute_fretting(cable, k, 0 * k, duration_s=10.0, section_loss=True)
    plain = cx.compute_fretting(cable, k, 0 * k, duration_s=10.0)
    key = "inner_armour/inner_face"
    assert np.max(worn.faces[key].D_fretting_per_year) >= np.max(
        plain.faces[key].D_fretting_per_year
    )
    summary = plain.summary()[key]
    assert summary["partner"] == plain.faces[key].partner
    assert sum(summary[f"share_{name}"] for name in fr.REGIMES) == pytest.approx(1.0)


def _rings(signs=(1, -1), pitches=(0.265, 0.285)):
    d = 3.21e-3
    rings = [ConductorRing(1, d, 0.0, 0.0)]
    for i, (n, pitch, sign) in enumerate(zip((6, 12), pitches, signs, strict=True), 1):
        rings.append(ConductorRing.from_pitch(n, d, i * d, pitch, sign))
    return rings


def test_trellis_inputs_and_regime_fractions():
    k = 0.05 * np.sin(np.linspace(0.0, 4.0 * math.pi, 161))
    args = {"mu_wire": 0.2, "duration_s": 100.0, "wear": None}
    with pytest.raises(CableInputError, match="equal 1-D"):
        conductor_trellis_fretting(_rings(), 115e9, 0.34, k, k[:-1], **args)
    with pytest.raises(CableInputError, match="duration_s"):
        conductor_trellis_fretting(
            _rings(), 115e9, 0.34, k, 0 * k, **{**args, "duration_s": 0.0}
        )
    result = conductor_trellis_fretting(
        _rings(), 115e9, 0.34, k, 0 * k, reference_tension_N=3e4, **args
    )
    assert np.all(result.wear_depth_per_year_m == 0.0)  # no wear coefficient
    fractions = result.regime_fraction
    totals = fractions.sum(axis=-1)
    assert np.all((np.isclose(totals, 1.0)) | (totals == 0.0))
    # Same lay: no trellis crossing between the rings (not assessed).
    parallel = conductor_trellis_fretting(
        _rings(signs=(1, 1), pitches=(0.265, 0.265)),
        115e9,
        0.34,
        k,
        0 * k,
        **{**args, "wear": get_wear_coefficient("copper_copper")},
    )
    assert parallel.contacts[1] is None
