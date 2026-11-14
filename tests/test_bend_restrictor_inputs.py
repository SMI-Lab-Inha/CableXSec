# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Bend restrictors and stiffeners: checks, field helpers and edge histories."""

from __future__ import annotations

import math
import warnings
from dataclasses import replace

import numpy as np
import pytest

from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.bend_restrictor import (
    ArticulatedRestrictor,
    BendStiffener,
    restrictor_curvature_field,
    stiffener_curvature_field,
)

THETA_L = math.radians(4.0)
FIELD = {"cable_od_m": 0.123, "bending_stiffness_Nm2": 5e3, "tension_N": 2e4}


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _restrictor(**kw):
    base = {"pitch_m": 0.36, "lock_angle_rad": THETA_L, "n_elements": 15}
    base["bore_diameter_m"] = 0.1375
    base.update(kw)
    return ArticulatedRestrictor(**base)


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"pitch_m": 0.0}, "pitch_m must be finite and positive"),
        ({"lock_angle_rad": 0.0}, "lock_angle_rad"),
        ({"lock_angle_rad": math.pi / 2}, "lock_angle_rad"),
        ({"n_elements": 0}, "n_elements"),
        ({"n_elements": True}, "n_elements"),
    ],
)
def test_restrictor_checks(kw, match):
    with pytest.raises(CableInputError, match=match):
        _restrictor(**kw)


def test_from_mbr_and_summary():
    with pytest.raises(CableInputError, match="shorter than the MBR diameter"):
        ArticulatedRestrictor.from_mbr(
            mbr_m=0.5, pitch_m=1.0, n_elements=5, bore_diameter_m=0.2
        )
    stated = _restrictor(stated_mbr_m=2.6, source="catalogue")
    summary = stated.summary(cable_od_m=0.123)
    assert summary["stated_mbr_m"] == 2.6
    assert summary["cable_locking_radius_m"] == pytest.approx(
        stated.cable_locking_radius_m(0.123)
    )


def test_stiffener_tip_must_clear_the_cable():
    cone = BendStiffener(length_m=2.0, root_od_m=0.5, tip_od_m=0.15, E_Pa=45e6)
    with pytest.raises(CableInputError, match="wider than the cable"):
        cone.bending_stiffness(np.array([0.0, 1.0]), cable_od_m=0.2, cable_EI_Nm2=5e3)


def test_curvature_field_contract_and_helpers():
    field = restrictor_curvature_field(_restrictor(), [0.0, 0.3, -0.3], **FIELD)
    with pytest.raises(CableInputError, match="start at x >= 0 and increase"):
        replace(field, x_m=field.x_m[::-1])
    with pytest.raises(CableInputError, match="finite \\(x, time\\) array"):
        replace(field, kappa_1_m=field.kappa_1_m[:, 0])
    np.testing.assert_array_equal(field.at(field.x_m[-1] + 1.0), np.zeros(3))
    assert 0.0 < field.forward_peak() <= field.peak()
    with pytest.raises(CableInputError, match="fraction"):
        field.bent_length_m(0.0)
    lengths = field.bent_length_m(1.0)
    assert lengths[0] == 0.0 and lengths[1] > 0.0


@pytest.mark.parametrize(
    ("args", "kwargs", "match"),
    [
        (([0.3],), {"rule": "elements"}, "rule"),
        (([[0.3]],), {}, "finite 1-D"),
        (([0.3],), {"tension_N": -1.0}, "non-negative"),
        (([0.3],), {"span_m": 2.0, "x_m": [0.0, 3.0]}, "within \\[0, span\\]"),
    ],
)
def test_restrictor_field_checks(args, kwargs, match):
    with pytest.raises(CableInputError, match=match):
        restrictor_curvature_field(_restrictor(), *args, **{**FIELD, **kwargs})


@pytest.mark.parametrize(
    ("angles", "kwargs", "match"),
    [
        ([[0.1]], {}, "finite 1-D"),
        ([math.pi / 2], {}, "below 90 degrees"),
        ([0.1], {"levels": 1}, "levels"),
        ([0.1], {"levels": True}, "levels"),
        ([0.1], {"x_m": [0.0, 1e3]}, "within the solved length"),
    ],
)
def test_stiffener_field_checks(angles, kwargs, match):
    cone = BendStiffener(length_m=2.0, root_od_m=0.5, tip_od_m=0.15, E_Pa=45e6)
    with pytest.raises(CableInputError, match=match):
        stiffener_curvature_field(
            cone,
            angles,
            cable_od_m=0.1,
            bending_stiffness_Nm2=5e3,
            tension_N=1e5,
            **kwargs,
        )


def test_zero_angle_history_gives_a_straight_cable():
    cone = BendStiffener(length_m=2.0, root_od_m=0.5, tip_od_m=0.15, E_Pa=45e6)
    field = stiffener_curvature_field(
        cone, [0.0, 0.0], cable_od_m=0.1, bending_stiffness_Nm2=5e3, tension_N=1e5
    )
    assert field.peak() == 0.0
    np.testing.assert_array_equal(field.bent_length_m(0.5), [0.0, 0.0])


def test_summary_without_optional_inputs():
    summary = _restrictor().summary()
    assert "stated_mbr_m" not in summary and "cable_locking_radius_m" not in summary
    assert summary["locking_radius_m"] == pytest.approx(_restrictor().locking_radius_m)
