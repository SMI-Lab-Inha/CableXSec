# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Bending-test simulation: law construction, protocol and set-up checks."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.along_wire_test import AlongWireSection, AlongWireSpec
from cablexsec.analysis.beam_test import (
    BendingProtocol,
    Cantilever,
    FourPoint,
    SectionLaw,
    ThreePoint,
    simulate_bending_test,
)
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.bend_restrictor import ArticulatedRestrictor
from cablexsec.mechanics.progressive_friction import ProgressiveFriction
from cablexsec.mechanics.radial_cable import solve_cable_radial_state
from cablexsec.mechanics.strand_slip import ConductorRing, conductor_ring_parameters

EI = 2.0e4
LOADS = BendingProtocol("load", np.array([100.0, 300.0, 0.0, -300.0, 0.0]))


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(scope="module")
def menard():
    return cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)


# --------------------------------------------------------------------------- #
# Section laws                                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tension_N": -1.0}, ">= 0"),
        ({"progressive": ProgressiveFriction()}, "friction_model='progressive'"),
        (
            {"progressive": "slow", "friction_model": "progressive"},
            "must be a ProgressiveFriction",
        ),
        ({"core_friction": "yes"}, "bool or CoreFrictionSpec"),
        ({"section_state": "state"}, "must be a SectionState"),
        ({"radial_state": "state"}, "must be a CableRadialState"),
        ({"contact_model": "exact"}, "contact_model"),
    ],
)
def test_cable_law_checks(menard, kwargs, match):
    with pytest.raises(CableInputError, match=match):
        SectionLaw.from_cable(menard, **kwargs)


def test_cable_law_from_a_section_state(menard):
    section = cx.solve_section_state(menard, tension_N=1e4, contact_model="bounded")
    law = SectionLaw.from_cable(menard, section_state=section)
    assert "bounded contact, tension 10000 N" in law.description
    for kwargs in ({"tension_N": 5.0}, {"contact_model": "qp"}):
        with pytest.raises(CableInputError, match="not both"):
            SectionLaw.from_cable(menard, section_state=section, **kwargs)


def test_cable_law_from_a_radial_state(menard):
    state = solve_cable_radial_state(menard, 1e4)
    law = SectionLaw.from_cable(menard, radial_state=state)
    assert "solved contact" in law.description
    with pytest.raises(CableInputError, match="radial state alone"):
        SectionLaw.from_cable(menard, radial_state=state, tension_N=1.0)


def test_distributed_law_with_laid_core_friction():
    cable = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
    kappa = np.linspace(0.0, 0.02, 6)
    tied = SectionLaw.from_cable(cable, core_friction=True).evaluate(kappa)
    law = SectionLaw.from_cable(cable, core_friction=True, friction_model="distributed")
    assert "laid-core friction" in law.description
    out = law.evaluate(kappa)
    assert np.all(np.diff(out.moment_Nm) > 0)
    assert out.friction_J_per_m is not None and np.all(out.friction_J_per_m >= 0)
    # Same stick and slip limits: the two armour laws agree at the ends.
    assert out.moment_Nm[1] == pytest.approx(tied.moment_Nm[1], rel=0.05)


def _rings():
    d = 3.21e-3
    rings = [ConductorRing(1, d, 0.0, 0.0)]
    for i, (n, pitch, sign) in enumerate(((6, 0.265, 1), (12, 0.285, -1)), start=1):
        rings.append(ConductorRing.from_pitch(n, d, i * d, pitch, sign))
    return rings


def test_conductor_law_and_a_test_without_friction_accounting():
    rings = _rings()
    params = conductor_ring_parameters(
        rings, 115e9, mu_wire=0.3, reference_tension_N=3e4, fibre="centroid"
    )
    with pytest.raises(CableInputError, match="cap_shape"):
        SectionLaw.from_conductor(rings, 115e9, params, cap_shape="round")  # type: ignore[arg-type]
    with pytest.raises(CableInputError, match="one parameter set per ring"):
        SectionLaw.from_conductor(rings, 115e9, params[:-1])
    law = SectionLaw.from_conductor(rings, 115e9, params, tension_N=3e4)
    assert law.ei_stick_Nm2 > law.ei_slip_Nm2 > 0
    out = law.evaluate(np.array([0.0, 0.05, 0.1]))
    assert out.friction_J_per_m is None and np.all(np.isfinite(out.moment_Nm))
    protocol = BendingProtocol.cyclic("load", 5.0, n_cycles=1, samples_per_cycle=16)
    result = simulate_bending_test(law, ThreePoint(0.5), protocol)
    assert result.friction_work_J is None and result.cycle_friction_work_J() is None
    assert result.cycle_external_work_J() > 0


# --------------------------------------------------------------------------- #
# Protocol and set-up                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("args", "match"),
    [
        (("load", np.zeros((2, 2))), "finite 1-D history"),
        (("load", np.array([1.0])), "finite 1-D history"),
        (("load", np.array([0.0, 1.0, 0.0]), 3), "retain_from"),
        (("torque", np.array([0.0, 1.0])), "control"),
    ],
)
def test_protocol_checks(args, match):
    with pytest.raises(CableInputError, match=match):
        BendingProtocol(*args)


def test_cyclic_protocol_checks():
    with pytest.raises(CableInputError, match="n_cycles"):
        BendingProtocol.cyclic("load", 1.0, n_cycles=0)
    with pytest.raises(CableInputError, match="samples_per_cycle"):
        BendingProtocol.cyclic("load", 1.0, samples_per_cycle=4)


@pytest.mark.parametrize(
    ("arrangement", "protocol", "kwargs", "match"),
    [
        ("three", LOADS, {}, "arrangement"),
        (ThreePoint(1.0), "load", {}, "protocol"),
        (ThreePoint(1.0), LOADS, {"stations_per_segment": 1}, "stations_per_segment"),
        (
            ThreePoint(1.0),
            LOADS,
            {"stations_per_segment": True},
            "stations_per_segment",
        ),
        (ThreePoint(1.0), LOADS, {"rtol": 0.1}, "rtol"),
        (
            ThreePoint(1.0),
            LOADS,
            {"curvature_definition": "best"},
            "curvature_definition",
        ),
        (ThreePoint(1.0), LOADS, {"gauge_x_m": [-0.1]}, "gauge positions"),
        (
            ThreePoint(1.0),
            LOADS,
            {"curvature_definition": "fit", "gauge_x_m": [0.4, 0.6]},
            "three gauges",
        ),
    ],
)
def test_setup_checks(arrangement, protocol, kwargs, match):
    with pytest.raises(CableInputError, match=match):
        simulate_bending_test(SectionLaw.elastic(EI), arrangement, protocol, **kwargs)


def test_section_argument(menard):
    with pytest.raises(CableInputError, match="SectionLaw or a CableData"):
        simulate_bending_test(5, ThreePoint(1.0), LOADS)  # type: ignore[arg-type]
    with pytest.raises(CableInputError, match="law options need a cable"):
        simulate_bending_test(SectionLaw.elastic(EI), ThreePoint(1.0), LOADS, mu=0.1)


def test_restrictor_options(menard):
    restrictor = ArticulatedRestrictor(
        pitch_m=0.2, lock_angle_rad=math.radians(5.0), n_elements=5, bore_diameter_m=0.3
    )
    with pytest.raises(CableInputError, match="not available with along_wire"):
        simulate_bending_test(
            menard,
            Cantilever(2.0),
            LOADS,
            restrictor=restrictor,
            along_wire=AlongWireSpec(),
        )
    with pytest.raises(CableInputError, match="give cable_od_m"):
        simulate_bending_test(
            SectionLaw.elastic(EI), Cantilever(2.0), LOADS, restrictor=restrictor
        )
    # The cable's own outer diameter sets the bore clearance.
    result = simulate_bending_test(
        menard, Cantilever(2.0), LOADS, restrictor=restrictor, stations_per_segment=4
    )
    assert result.restrictor_moment_field_Nm is not None


def test_cantilever_deflection_and_zero_histories():
    law = SectionLaw.elastic(EI)
    length, force = 2.0, 50.0
    result = simulate_bending_test(
        law, Cantilever(length), BendingProtocol("load", np.array([0.0, force]))
    )
    for x in (0.5, 1.0, 2.0):
        expected = force * x**2 * (3 * length - x) / (6 * EI)
        assert abs(result.deflection_m(x)[1]) == pytest.approx(expected, rel=1e-6)
    for control in ("load", "displacement"):
        zero = simulate_bending_test(
            law, FourPoint(1.0, 0.4), BendingProtocol(control, np.zeros(3))
        )
        assert np.all(zero.load_N == 0.0) and np.all(zero.curvature_field_1_m == 0.0)


# --------------------------------------------------------------------------- #
# Along-wire route                                                            #
# --------------------------------------------------------------------------- #
def test_along_wire_arguments():
    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    arrangement = FourPoint(1.0, 0.5)
    with pytest.raises(CableInputError, match="must be an AlongWireSpec"):
        simulate_bending_test(cable, arrangement, LOADS, along_wire="anchored")  # type: ignore[arg-type]
    section = AlongWireSection.from_cable(cable)
    with pytest.raises(CableInputError, match="law options need a cable"):
        simulate_bending_test(
            section, arrangement, LOADS, along_wire=AlongWireSpec(), mu=0.1
        )
    with pytest.raises(CableInputError, match="core_friction"):
        simulate_bending_test(
            cable, arrangement, LOADS, along_wire=AlongWireSpec(), core_friction=True
        )
    with pytest.raises(CableInputError, match="three gauges"):
        simulate_bending_test(
            cable,
            arrangement,
            LOADS,
            along_wire=AlongWireSpec(),
            curvature_definition="fit",
            gauge_x_m=[0.3],
        )


def test_along_wire_section_needs_a_cable():
    with pytest.raises(CableInputError, match="needs a CableData"):
        AlongWireSection.from_cable(SectionLaw.elastic(EI))  # type: ignore[arg-type]
