# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Armour fatigue layers and stress fields: input checks and accessors."""

from __future__ import annotations

import math
import warnings
from dataclasses import replace

import numpy as np
import pytest

import cablexsec as cx
import cablexsec.analysis.fatigue as fatigue
from cablexsec.analysis.fatigue import compute_fatigue, wire_stress_field
from cablexsec.exceptions import CableInputError
from cablexsec.materials.sn_curves import SNcurve, get_sn_curve

K = 0.05 * np.sin(np.linspace(0.0, 4.0 * math.pi, 81))
Z = np.zeros_like(K)


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(scope="module")
def layer():
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    return cx.cable_to_fatigue_layers(cable, contact_tension_N=2e4)[0]


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"mean_stress": "goodman"}, "MeanStressSpec"),
        ({"component_scf": 1.2}, "ComponentSCF"),
        ({"thickness_m": 0.0}, "thickness_m"),
        ({"lay_hand": 0}, "lay_hand"),
        ({"above_core_mask": np.ones(3, dtype=bool)}, "above_core_mask|mask"),
    ],
)
def test_layer_checks(layer, changes, match):
    with pytest.raises(CableInputError, match=match):
        replace(layer, **changes)


def test_miner_row_checks(layer):
    rows = np.array([[0.0, 1e8, 0.0]])
    um = SNcurve(name="slip", stress_unit="um", m1=3.0, log_C1=12.0)
    with pytest.raises(CableInputError, match="in MPa"):
        fatigue._miner_damage_rows(rows, um, 1e9, 1.0, False)
    with pytest.raises(CableInputError, match="stress_range_factor"):
        fatigue._miner_damage_rows(
            rows, get_sn_curve("dnv_rpc203_t_curve"), 1e9, 0.0, False
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"normal_stress_Pa": Z, "tension_N": Z}, "solved normal stress"),
        ({"wire_modulus_Pa": np.full(K.size, -1.0)}, "wire modulus must be positive"),
        ({"stick_modulus_Pa": np.ones(3)}, "stick modulus must be finite"),
        ({"axial_strain": Z, "tension_N": Z, "EA_full_N": 1e9}, "not both"),
        ({"axial_strain": np.ones(3)}, "axial strain must be finite"),
        ({"curvature_gradient": (np.ones(3), np.ones(3))}, "curvature gradient"),
        ({"relief": "lateral", "friction_model": "distributed"}, "LateralRelief"),
        ({"termination": "clamp"}, "TerminationSpec"),
        ({"transverse": 0.1}, "TransverseSlipModel"),
    ],
)
def test_stress_field_checks(layer, kwargs, match):
    with pytest.raises(CableInputError, match=match):
        wire_stress_field(layer, K, Z, **kwargs)


def test_stress_field_accessors(layer):
    field = wire_stress_field(layer, K, Z)
    with pytest.raises(CableInputError, match="friction"):
        field.scf_fibre(0.0, cx.ComponentSCF(axial_dynamic=1.1, source="test"))
    with pytest.raises(CableInputError, match="fibre angle must be finite"):
        field.at_fibre(math.nan)
    with pytest.raises(CableInputError, match="section point must be finite"):
        field.at_point(math.inf, 0.0)
    with pytest.raises(CableInputError):
        field.at_point(0.0, 0.0)  # round wire: no section points
    kept = wire_stress_field(layer, K, Z, keep_friction=True)
    fibre = kept.scf_fibre(0.0, cx.ComponentSCF(source="unit"))
    np.testing.assert_allclose(fibre, kept.at_fibre(0.0), rtol=1e-12, atol=1e-6)


def test_compute_fatigue_load_conflict(layer):
    with pytest.raises(CableInputError, match="solved normal stress"):
        compute_fatigue(
            [layer],
            K,
            Z,
            tension_N=Z,
            EA_full_N=1e9,
            normal_stress_Pa={layer.name: Z},
        )
