# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Concentric solvers reject ambiguous constraints and singular constructions."""

from dataclasses import replace

import numpy as np
import pytest
from _bundled import incomplete_cable

import cablexsec as cx
from cablexsec.mechanics.concentric_section import (
    ConcentricTangent,
    ElasticAnnulus,
    ElasticHelicalRing,
    solve_concentric_section,
)


def test_duplicate_section_names_are_rejected():
    inner = ElasticAnnulus("same", 0, 0.01, 1e9, 0.3)
    outer = ElasticAnnulus("same", 0.01, 0.02, 1e9, 0.3)
    with pytest.raises(ValueError, match="unique"):
        solve_concentric_section([inner, outer])


@pytest.mark.parametrize(
    "updates",
    [
        {"signed_lay_angle_rad": 0},
        {"signed_lay_angle_rad": np.nan},
        {"wire_count": 0},
        {"wire_count": True},
        {"initial_wire_strain": np.inf},
    ],
)
def test_invalid_helical_ring(updates):
    ring = ElasticHelicalRing("ring", 0.02, 0.001, 8, 0.3, 210e9)
    with pytest.raises(ValueError):
        solve_concentric_section([replace(ring, **updates)])


def test_free_twist_requires_torsional_support():
    tangent = ConcentricTangent(np.diag([1.0, 0.0]), (), ())
    with pytest.raises(ValueError, match="torsionally supported"):
        _ = tangent.ea_free_N


@pytest.mark.parametrize(
    "kwargs",
    [
        {"torque_Nm": 1},
        {"prescribed_twist_per_m": None},
        {"initial_wire_strain": {"inner_armour": 0.001}},
    ],
)
def test_bounded_section_cannot_claim_coupled_torsion(kwargs):
    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    with pytest.raises(cx.CableSchemaError, match="coupled section"):
        cx.solve_section_state(cable, contact_model="bounded", **kwargs)


def test_section_requires_complete_geometry():
    with pytest.raises(cx.CableSchemaError, match="complete geometry"):
        cx.solve_section_state(incomplete_cable())
