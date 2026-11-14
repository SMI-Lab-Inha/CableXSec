# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Along-wire bending test: specification and section checks, other routes."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from test_beam_test_inputs import _rings

import cablexsec as cx
from cablexsec.analysis.along_wire_test import AlongWireSection, AlongWireSpec
from cablexsec.analysis.beam_test import (
    BendingProtocol,
    FourPoint,
    SectionLaw,
    ThreePoint,
    simulate_bending_test,
)
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.along_wire import HAVE_NUMBA
from cablexsec.mechanics.strand_slip import conductor_ring_parameters

compiled = pytest.mark.skipif(not HAVE_NUMBA, reason="needs the compiled kernels")


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(scope="module")
def cable33():
    return cx.load_cable(cx.inter_array_single_core_yaml_path())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sense": 0}, "sense"),
        ({"segments_per_pitch": 4}, "segments_per_pitch"),
        ({"substeps": True}, "substeps"),
        ({"max_iterations": 2}, "max_iterations"),
    ],
)
def test_spec_checks(kwargs, match):
    with pytest.raises(CableInputError, match=match):
        AlongWireSpec(**kwargs)


def test_section_checks_and_stick_stiffness(cable33):
    with pytest.raises(CableInputError, match="ei_elastic_Nm2"):
        AlongWireSection((), 0.0)
    with pytest.raises(CableInputError, match="WireChainLayer"):
        AlongWireSection(("layer",), 1e3)  # type: ignore[arg-type]
    section = AlongWireSection.from_cable(cable33)
    law = SectionLaw.from_cable(cable33)
    assert section.ei_stick_Nm2 > section.ei_elastic_Nm2
    assert section.ei_stick_Nm2 == pytest.approx(law.ei_stick_Nm2, rel=0.05)
    with pytest.raises(CableInputError, match="without friction"):
        AlongWireSection.from_cable(cable33, mu=0.0)


@compiled
def test_conductor_rings_along_the_wires():
    rings = _rings()
    params = conductor_ring_parameters(
        rings, 115e9, mu_wire=0.3, reference_tension_N=3e4, fibre="centroid"
    )
    section = AlongWireSection.from_conductor(rings, 115e9, params, tension_N=3e4)
    assert len(section.layers) == len(rings) - 1  # the centre wire is elastic only
    law = SectionLaw.from_conductor(rings, 115e9, params, tension_N=3e4)
    assert section.ei_elastic_Nm2 == pytest.approx(law.ei_slip_Nm2, rel=1e-12)


@compiled
def test_displacement_control_and_elastic_definition(cable33):
    section = AlongWireSection.from_cable(cable33)
    p = max(lay.pitch_m for lay in section.layers)
    protocol = BendingProtocol.cyclic(
        "displacement", 2e-3, n_cycles=1, samples_per_cycle=16
    )
    result = simulate_bending_test(
        section,
        ThreePoint(4 * p),
        protocol,
        stations_per_segment=8,
        along_wire=AlongWireSpec(),
    )
    assert result.curvature_definition == "elastic"
    np.testing.assert_allclose(
        result.displacement_m, protocol.values, rtol=1e-6, atol=1e-12
    )
    assert result.equilibrium_residual < 1e-6
    four = simulate_bending_test(
        section,
        FourPoint(4 * p, 2 * p),
        BendingProtocol("load", np.array([50.0, 100.0, 0.0])),
        stations_per_segment=8,
        along_wire=AlongWireSpec(substeps=2),
    )
    assert np.all(np.isfinite(four.curvature_1_m))
