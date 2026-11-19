# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Open and closed contacts of the radial section, as the solver decides them.

The how-to recipe counted gaps above 1e-9 m as open. A closed contact reports
a gap that is the round-off of the linear solve: on the 66 kV template at
100 kN and a sealed 2 MPa it is about 1.5e-10 m on two interfaces that carry
1.1 MN/m, within the solver's own complementarity tolerance in metres
(1e-8 of the scaled problem times the square root of the contact
flexibility: 2e-10 to 4e-10 m here) and close enough to 1e-9 m that another
platform's linear algebra crossed it (one macOS run printed "open gaps 1").
"""

from __future__ import annotations

import numpy as np
import pytest

import cablexsec as cx


@pytest.fixture(scope="module")
def states():
    cable = cx.load_cable(cx.iea_task49_66kv_yaml_path())
    return {
        p: cx.solve_cable_radial_state(cable, 100e3, external_pressure_Pa=p).result
        for p in (0.0, 2e6)
    }


def test_closed_contacts_are_closed_whatever_their_round_off_gap(states):
    closed = states[2e6]
    assert closed.open_interfaces == ()
    assert not closed.interface_open.any()
    # Their reported gaps are round-off, of either sign, not zero.
    assert 1e-12 < float(np.max(np.abs(closed.interface_gap_m))) < 1e-9
    assert float(np.min(closed.interface_line_force_N_per_m)) > 1e5


def test_open_contact_carries_no_force(states):
    result = states[0.0]
    assert result.open_interfaces == (("outer_armour", "hdpe_outer_sheath"),)
    gap = result.interface_gap_m[result.interface_open]
    assert gap.size == 1 and gap[0] > 1e-6  # a solved gap, far above round-off
    assert result.interface_line_force_N_per_m[result.interface_open][0] == 0.0


def test_classification_ignores_the_gap_round_off(states):
    """Either side of a gap threshold: a closed contact whose round-off gap is
    raised past 1e-9 m stays closed; an open one whose gap is lowered stays
    open. Only the line force decides."""
    import dataclasses

    closed = states[2e6]
    noisy = dataclasses.replace(closed, interface_gap_m=closed.interface_gap_m + 5e-9)
    assert noisy.open_interfaces == ()
    opened = states[0.0]
    tight = dataclasses.replace(opened, interface_gap_m=opened.interface_gap_m * 0.0)
    assert tight.open_interfaces == opened.open_interfaces
