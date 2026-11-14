# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Gap SH: cable over a sheave."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.sheave import compute_sheave, groove_pressure, sheave_contact
from cablexsec.exceptions import CableInputError


def test_feyrer_example_3_2_tape_on_sheave():
    contact = sheave_contact(10e3, 0.8, 200e9 * 0.030 * 0.002**3 / 12.0)
    assert math.degrees(contact.run_on_angle_rad) == pytest.approx(2.86, abs=0.005)
    assert contact.run_on_force_N == pytest.approx(500.0, abs=1.0)
    assert contact.line_pressure_N_per_m == pytest.approx(25e3)
    assert contact.lever_arm_m == pytest.approx(1.0e-3, rel=1e-3)


def test_groove_factors_of_feyrer():
    assert groove_pressure() == pytest.approx((1.351, 1.170), abs=5e-4)
    undercut = groove_pressure(math.radians(47.5), math.radians(80.0))
    assert undercut == pytest.approx((5.627, 2.062), abs=5e-3)
    with pytest.raises(CableInputError):
        groove_pressure(math.radians(60.0), math.radians(10.0))


def test_sheave_on_the_66kv_template():
    cable = cx.load_cable(cx.iea_task49_66kv_yaml_path())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        installation = cx.ARMOUR_UTILISATION_FACTORS["installation"]
        r = compute_sheave(
            cable,
            sheave_diameter_m=6.4,
            tension_N=50e3,
            n_ramp=10,
            utilisation_factor=installation,
        )
        root = compute_sheave(
            cable,
            sheave_diameter_m=6.4 - r.cable_diameter_m,
            tension_N=50e3,
            diameter_basis="groove_root",
            n_ramp=10,
            utilisation_factor=installation,
        )
    assert r.curvature_1_m == pytest.approx(1.0 / 3.2)
    assert root.curvature_1_m == pytest.approx(r.curvature_1_m)
    assert r.global_pressure_Pa == pytest.approx(2 * 50e3 / (6.4 * r.cable_diameter_m))
    assert 0 < r.characteristic_utilisation < 1 and r.passes_allowed > 1
    assert 0 < r.design_utilisation < 1
    # The conductor's 0.2 % proof stress governs the 3.2 m bend radius (a
    # 3 m radius takes it above the limit in the worst bending direction).
    assert r.governing_mechanism == "component_limits"
    assert r.governing_component == "cu_conductor: yielding"
    assert r.run_on_small_angle == (False, False)  # stiff cable: flagged
    with pytest.raises(CableInputError):
        compute_sheave(cable, sheave_diameter_m=0.1, tension_N=50e3)
    assert np.isfinite(max(r.wire_stress_max_Pa.values()))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tension_N": 0.0}, "tension_N"),
        ({"sheave_diameter_m": np.nan}, "sheave_diameter_m"),
        ({"diameter_basis": "rim"}, "diameter_basis"),
        ({"n_ramp": 1}, "n_ramp"),
        ({"cable_diameter_m": -0.1}, "cable_diameter_m"),
    ],
)
def test_sheave_input_checks(kwargs, match):
    cable = cx.load_cable(cx.iea_task49_66kv_yaml_path())
    args = {"sheave_diameter_m": 6.0, "tension_N": 50e3}
    args.update(kwargs)
    with pytest.raises(CableInputError, match=match):
        compute_sheave(cable, **args)


def test_tape_contact_and_outer_diameter_checks():
    with pytest.raises(CableInputError, match="tension_N"):
        sheave_contact(0.0, 0.8, 1.0)
    from cablexsec.analysis.sheave import _outer_diameter

    cable = cx.load_cable(cx.iea_task49_66kv_yaml_path())
    empty = cable.model_copy(update={"layers": []})
    with pytest.raises(CableInputError, match="outer diameter"):
        _outer_diameter(empty)


def test_sheave_capacity_matches_the_capacity_envelope():
    """The design utilisation is that of the capacity envelope (same factor,
    contact chain and mechanisms); the characteristic ratio stays available."""
    cable = cx.load_cable(cx.iea_task49_66kv_yaml_path())
    tension = 50e3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = compute_sheave(cable, sheave_diameter_m=4.0, tension_N=tension, n_ramp=4)
        envelope = cx.build_capacity_envelope(cable, np.array([0.0, tension, 1e5]))
        shared = compute_sheave(
            cable,
            sheave_diameter_m=4.0,
            tension_N=tension,
            n_ramp=4,
            capacity_envelope=envelope,
        )
        installation = compute_sheave(
            cable,
            sheave_diameter_m=4.0,
            tension_N=tension,
            n_ramp=4,
            utilisation_factor=cx.ARMOUR_UTILISATION_FACTORS["installation"],
        )
    u, governing, _ = envelope.utilisation(np.array([tension]), np.array([0.5]))
    assert r.utilisation_factor == cx.ARMOUR_UTILISATION_FACTORS["normal"]
    assert r.design_utilisation == pytest.approx(float(u[0]), rel=1e-9)
    assert shared.design_utilisation == float(u[0])
    # The conductor's 0.2 % proof stress governs, not the armour.
    assert r.governing_mechanism == "component_limits" and int(governing[0]) == 3
    assert r.governing_component == "cu_conductor: yielding"
    assert r.design_curvature_capacity_1_m == pytest.approx(
        float(envelope.curvature_capacity(np.array([tension]))[0]), rel=1e-9
    )
    # Characteristic (first-yield) curvature lies at 1/eta on the design scale.
    assert r.design_curvature_capacity_1_m < r.characteristic_curvature_capacity_1_m
    assert installation.design_utilisation < r.design_utilisation
    with pytest.deprecated_call():
        assert r.utilisation == r.characteristic_utilisation
    with pytest.deprecated_call():
        assert r.curvature_capacity_1_m == r.characteristic_curvature_capacity_1_m
    with pytest.raises(CableInputError, match="not both"):
        compute_sheave(
            cable,
            sheave_diameter_m=4.0,
            tension_N=tension,
            utilisation_factor=0.67,
            capacity_envelope=envelope,
        )
