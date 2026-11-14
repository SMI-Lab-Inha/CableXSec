# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Radial contact solution: section assembly, load and identification checks."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from _bundled import incomplete_cable

import cablexsec as cx
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.mechanics.concentric_section import ElasticAnnulus, ElasticHelicalRing
from cablexsec.mechanics.radial_cable import (
    build_radial_section,
    solve_cable_radial_state,
    solve_cable_radial_states,
)
from cablexsec.mechanics.radial_solution import (
    AxialMember,
    RadialSection,
    capstan_residual_stress,
    solve_lcp,
)


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _rings():
    return [
        ElasticAnnulus("sheath", 0.0351, 0.0400, 0.3e9, 0.4),
        ElasticHelicalRing("armour", 0.04125, 0.0025, 90, math.radians(35.0), 207e9),
        ElasticAnnulus("jacket", 0.0425, 0.0465, 0.3e9, 0.4),
    ]


# --------------------------------------------------------------------------- #
# Section                                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("layers", "kwargs", "match"),
    [
        (_rings() + [ElasticAnnulus("sheath", 0.05, 0.06, 1e9, 0.4)], {}, "unique"),
        ([ElasticAnnulus("bundle", 0.03, 0.04, 1e9, 0.4)], {}, "not 'bundle'"),
        ([], {}, "at least one layer"),
        (
            _rings(),
            {"rigid_bore": True, "bore_foundation_Pa_per_m": 1e9},
            "bore foundation",
        ),
        (_rings(), {"bore_foundation_Pa_per_m": -1.0}, "bore foundation"),
        (_rings(), {"interface_compliance": {("a", "b"): 1e-9}}, "unknown interface"),
        (
            _rings(),
            {"interface_compliance": {("sheath", "armour"): -1.0}},
            "unknown interface",
        ),
        ([ElasticAnnulus("x", 0.04, 0.03, 1e9, 0.4)], {}, "finite and ordered"),
        ([ElasticAnnulus("x", 0.03, 0.04, 0.0, 0.4)], {}, "modulus must be positive"),
        ([ElasticAnnulus("x", 0.03, 0.04, 1e9, 0.6)], {}, "Poisson ratio"),
        ([ElasticHelicalRing("x", 0.04, 0.002, 60, 0.0, 2e11)], {}, "lay angle"),
        (
            [
                ElasticAnnulus("a", 0.0, 0.01, 1e9, 0.3),
                ElasticAnnulus("b", 0.0, 0.02, 1e9, 0.3),
            ],
            {},
            "overlap",  # a second solid layer overlaps the first
        ),
        (
            [ElasticAnnulus("core", 0.0, 0.02, 1e9, 0.3)],
            {"rigid_bore": True},
            "no bore",
        ),
        (
            [ElasticAnnulus("core", 0.0, 0.02, 1e9, 0.3)],
            {"bore_foundation_Pa_per_m": 1e9},
            "no bore",
        ),
        (_rings(), {"members": [AxialMember("tube", -1.0)]}, "non-negative"),
    ],
)
def test_section_checks(layers, kwargs, match):
    with pytest.raises(CableInputError, match=match):
        RadialSection(layers, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tension_N": math.nan}, "finite"),
        ({"internal_pressure_Pa": -1.0}, "non-negative"),
        ({"tension_N": 1e4, "prescribed_axial_strain": 1e-3}, "not both"),
        ({"torque_Nm": 10.0}, "torque or twist"),
        ({"initial_wire_strain": {"jacket": 1e-4}}, "unknown helical layer"),
    ],
)
def test_solve_checks(kwargs, match):
    with pytest.raises(CableInputError, match=match):
        RadialSection(_rings()).solve(**kwargs)


def test_result_properties_and_solid_core():
    result = RadialSection(_rings()).solve(tension_N=5e4, prescribed_twist_per_m=None)
    assert result.ea_free_twist_N <= result.ea_fixed_twist_N
    assert result.gj_fixed_tension_Nm2 > 0
    inner, outer = result.helical_face_pressure_Pa["armour"]
    assert result.p1_plus_p2("armour") == pytest.approx(inner + outer)
    solid = RadialSection([ElasticAnnulus("core", 0.0, 0.0351, 1e9, 0.3), *_rings()])
    with pytest.raises(CableInputError, match="no internal pressure surface"):
        solid.solve(internal_pressure_Pa=1e5)


def test_initial_strain_identification_inputs():
    section = RadialSection(_rings())
    with pytest.raises(CableInputError, match="unknown helical layer"):
        section.identify_initial_wire_strain({"sheath": 1e6})
    assert section.identify_initial_wire_strain({}) == {}


def test_capstan_and_lcp_edges():
    with pytest.raises(CableInputError, match="lay angle"):
        capstan_residual_stress(
            1e6,
            wire_count=60,
            wire_diameter_m=0.002,
            helix_radius_m=0.04,
            lay_angle_rad=0.0,
        )
    assert solve_lcp(np.zeros((0, 0)), np.zeros(0)).size == 0


# --------------------------------------------------------------------------- #
# Cable radial section                                                        #
# --------------------------------------------------------------------------- #


def test_cable_section_checks():
    menard = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    with pytest.raises(CableSchemaError, match="flooded pressure"):
        build_radial_section(menard, flooded_pressure_Pa=-1.0)
    with pytest.raises(CableSchemaError, match="complete geometry"):
        build_radial_section(incomplete_cable())
    with pytest.raises(CableSchemaError, match="at least one tension"):
        solve_cable_radial_states(menard, [])
    with pytest.raises(CableSchemaError, match="finite"):
        solve_cable_radial_states(menard, [1e4], internal_pressure_Pa=-1.0)
    state = solve_cable_radial_state(menard, 1e4)
    other = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
    with pytest.raises(CableSchemaError, match="different cable"):
        state.require_cable(other)
