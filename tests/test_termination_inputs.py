# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Terminations: specification checks, principal direction and increment inputs."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.termination import (
    TerminationSpec,
    principal_direction,
    termination_friction_increment,
)
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.bend_restrictor import (
    ArticulatedRestrictor,
    restrictor_curvature_field,
)

K = 0.05 * np.sin(np.linspace(0.0, 4.0 * math.pi, 41))


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(scope="module")
def spec():
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    return cx.cable_to_fatigue_layers(cable, contact_tension_N=2e4)[0].spec


def _field(n):
    restrictor = ArticulatedRestrictor(
        pitch_m=0.2, lock_angle_rad=math.radians(5.0), n_elements=5, bore_diameter_m=0.3
    )
    return restrictor_curvature_field(
        restrictor,
        np.linspace(0.05, 0.1, n),
        cable_od_m=0.12,
        bending_stiffness_Nm2=5e3,
        tension_N=5e4,
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"solver": "along_wire", "lateral_relief": "x"}, "LateralRelief"),
        ({"curvature_field": "x"}, "CurvatureField"),
        ({"solver": "along_wire", "stick": "glued"}, "stick"),
        ({"curvature_profile": (1.0, 2.0, 3.0)}, "a pair"),
    ],
)
def test_spec_checks(kwargs, match):
    with pytest.raises(CableInputError, match=match):
        TerminationSpec(distance_m=0.5, **kwargs)


def test_principal_direction_without_variance():
    assert principal_direction(np.full(5, 0.1), np.full(5, 0.1)) == pytest.approx(
        (math.pi / 4, 1.0)
    )
    assert principal_direction(np.zeros(4), np.zeros(4)) == (0.0, 1.0)


def test_increment_inputs(spec):
    clamp = TerminationSpec(distance_m=0.5)
    with pytest.raises(CableInputError, match="finite matching"):
        termination_friction_increment(spec, K, K[:-1], clamp)
    with pytest.raises(CableInputError, match="lay_hand"):
        termination_friction_increment(spec, K, 0 * K, clamp, lay_hand=0)
    with pytest.raises(CableInputError, match="contact pressure"):
        termination_friction_increment(
            spec, K, 0 * K, clamp, contact_pressure_Pa=np.ones(3)
        )
    field_spec = TerminationSpec(distance_m=0.1, curvature_field=_field(5))
    with pytest.raises(CableInputError, match="field"):
        termination_friction_increment(spec, K, 0 * K, field_spec)
    zero = termination_friction_increment(spec, 0 * K, 0 * K, clamp)
    assert np.all(zero.stress_Pa == 0.0)
