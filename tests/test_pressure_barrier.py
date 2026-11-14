# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Opt-in fluid-tight barrier for the bore pressure of the solved radial state.

A flexible pipe's interlocked carcass is not fluid-tight: the bore pressure
acts on the internal pressure sheath (Kraincanic and Kebadze 2001, p. 266:
the plastic sheaths are the barriers to the transported fluid). With
``pressure_barrier`` the layers inside the barrier carry the fluid on both
faces, so the section outside the barrier must respond as if those layers
were absent, and the closed-end thrust uses the barrier bore.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from cablexsec import load_cable
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.mechanics.concentric_section import ElasticAnnulus, ElasticHelicalRing
from cablexsec.mechanics.radial_cable import solve_cable_radial_state
from cablexsec.mechanics.radial_solution import RadialSection

pytestmark = pytest.mark.analytical_sanity

P_INT = 30.0e6

_PIPE = """
cable:
  name: "carcass pipe for the pressure-barrier tests"
layers:
  - {name: carcass, layer_type: helical, material: galvanised_steel_armour,
     n_wires: 2, d_wire_m: 0.0035, r_helix_m: 0.03335, lay_angle_deg: 87.5,
     lay_direction: left_hand, source: test}
  - {name: sheath, layer_type: cylindrical, material: hdpe,
     r_inner_m: 0.0351, r_outer_m: 0.0400, source: test}
  - {name: hoop, layer_type: helical, material: galvanised_steel_armour,
     n_wires: 3, d_wire_m: 0.0062, r_helix_m: 0.04310, lay_angle_deg: 85.5,
     lay_direction: left_hand, source: test}
  - {name: tape, layer_type: cylindrical, material: hdpe,
     r_inner_m: 0.04620, r_outer_m: 0.04775, source: test}
  - {name: armour, layer_type: helical, material: galvanised_steel_armour,
     n_wires: 102, d_wire_m: 0.0030, r_helix_m: 0.04925, lay_angle_deg: 35.0,
     lay_direction: left_hand, source: test}
  - {name: jacket, layer_type: cylindrical, material: hdpe,
     r_inner_m: 0.05075, r_outer_m: 0.05375, source: test}
"""


@pytest.fixture(scope="module")
def pipes(tmp_path_factory):
    folder = tmp_path_factory.mktemp("barrier")
    full = folder / "pipe.yaml"
    full.write_text(_PIPE, encoding="utf-8")
    lines = _PIPE.splitlines()
    start = next(i for i, x in enumerate(lines) if "name: carcass" in x)
    reduced = folder / "pipe_without_carcass.yaml"
    reduced.write_text("\n".join(lines[:start] + lines[start + 3 :]), encoding="utf-8")
    return (
        load_cable(full, geometry_override=None),
        load_cable(reduced, geometry_override=None),
    )


def _rings() -> list:
    return [
        ElasticHelicalRing("carcass", 0.03335, 0.0035, 2, math.radians(87.5), 190e9),
        ElasticAnnulus("sheath", 0.0351, 0.0400, 0.3e9, 0.4),
        ElasticHelicalRing("hoop", 0.0431, 0.0062, 3, math.radians(-85.5), 207e9),
        ElasticAnnulus("tape", 0.0462, 0.04775, 0.3e9, 0.4),
        ElasticHelicalRing("armour", 0.04925, 0.003, 102, math.radians(35.0), 207e9),
    ]


def test_default_bore_is_bitwise_unchanged():
    section = RadialSection(_rings())
    a = section.solve(tension_N=5e4, internal_pressure_Pa=P_INT)
    b = section.solve(
        tension_N=5e4, internal_pressure_Pa=P_INT, internal_pressure_layer=0
    )
    np.testing.assert_array_equal(
        a.interface_line_force_N_per_m, b.interface_line_force_N_per_m
    )
    assert a.helical_face_pressure_Pa == b.helical_face_pressure_Pa


def test_section_outside_the_barrier_responds_as_without_the_inner_layers():
    """The fluid-filled carcass separates; the rest equals the reduced section.

    With extension and twist prescribed the separated carcass is coupled to
    nothing else, so the agreement is to solver precision. (Under prescribed
    tension its own small axial stiffness, 0.4% of the section, remains.)
    """
    with_carcass = RadialSection(_rings()).solve(
        prescribed_axial_strain=1.0e-3,
        internal_pressure_Pa=P_INT,
        internal_pressure_layer=1,
    )
    without = RadialSection(_rings()[1:]).solve(
        prescribed_axial_strain=1.0e-3, internal_pressure_Pa=P_INT
    )
    assert with_carcass.interface_line_force_N_per_m[0] == 0.0  # separated
    np.testing.assert_allclose(
        with_carcass.interface_line_force_N_per_m[1:],
        without.interface_line_force_N_per_m,
        rtol=1e-9,
    )
    for name in ("hoop", "armour"):
        np.testing.assert_allclose(
            with_carcass.helical_face_pressure_Pa[name],
            without.helical_face_pressure_Pa[name],
            rtol=1e-9,
        )
    # The bore fluid inside the barrier is not a friction-setting pressure.
    assert with_carcass.helical_face_pressure_Pa["carcass"] == (0.0, 0.0)


def test_cable_route_end_cap_uses_the_barrier_bore(pipes):
    full, reduced = pipes
    p_ext = 0.5e6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        barrier = solve_cable_radial_state(
            full,
            2e4,
            internal_pressure_Pa=P_INT,
            external_pressure_Pa=p_ext,
            end_cap=True,
            pressure_barrier="sheath",
        )
        reference = solve_cable_radial_state(
            reduced,
            2e4,
            internal_pressure_Pa=P_INT,
            external_pressure_Pa=p_ext,
            end_cap=True,
        )
        bore = solve_cable_radial_state(
            full,
            2e4,
            internal_pressure_Pa=P_INT,
            external_pressure_Pa=p_ext,
            end_cap=True,
        )
    r_out = 0.05375
    assert barrier.mechanical_axial_force_N == pytest.approx(
        2e4 + math.pi * (P_INT * 0.0351**2 - p_ext * r_out**2), rel=1e-12
    )
    assert barrier.mechanical_axial_force_N == pytest.approx(
        reference.mechanical_axial_force_N, rel=1e-12
    )
    assert barrier.pressure_barrier == "sheath" and bore.pressure_barrier is None
    pa, pr = barrier.armour_contact_pressure(), reference.armour_contact_pressure()
    for name in ("hoop", "armour"):
        assert pa[name] == pytest.approx(pr[name], rel=1e-3)
    # The default (pressure on the carcass bore) loads the carcass instead.
    assert bore.armour_contact_pressure()["carcass"] > 0.5 * P_INT
    assert pa["carcass"] == 0.0
    assert bore.armour_contact_pressure()["armour"] < pa["armour"]


def test_barrier_naming_the_innermost_layer_is_the_default(pipes):
    full, _ = pipes
    a = solve_cable_radial_state(full, 1e4, internal_pressure_Pa=P_INT, end_cap=True)
    b = solve_cable_radial_state(
        full, 1e4, internal_pressure_Pa=P_INT, end_cap=True, pressure_barrier="carcass"
    )
    assert a.mechanical_axial_force_N == b.mechanical_axial_force_N
    assert a.armour_contact_pressure() == b.armour_contact_pressure()


def test_invalid_barriers_are_rejected(pipes):
    full, _ = pipes
    with pytest.raises(CableSchemaError):
        solve_cable_radial_state(
            full, 0.0, internal_pressure_Pa=P_INT, pressure_barrier="no_such_layer"
        )
    with pytest.raises(CableSchemaError):
        solve_cable_radial_state(
            full, 0.0, flooded_pressure_Pa=1e6, pressure_barrier="sheath"
        )
    with pytest.raises(CableInputError):
        RadialSection(_rings()).solve(
            internal_pressure_Pa=P_INT, internal_pressure_layer=7
        )
