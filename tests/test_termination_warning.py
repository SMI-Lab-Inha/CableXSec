# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""A clamped termination correction is flagged as unsupported by measurement.

On the only published near-termination measurements (Saevik 2011, Gjosteen
et al. 2027; BENCHMARKS SV11 and T2) the anchored-wire Coulomb correction gives
1.6-3.4 times the measured ranges, the sectional result 1.0 times; the
excess comes from the bellmouth curvature profile, not from the end-fitting
anchor. The warning changes no number.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.fatigue import compute_fatigue
from cablexsec.analysis.termination import (
    CLAMPED_TERMINATION_WARNING,
    TerminationSpec,
)
from cablexsec.mechanics.clamp_layer import HAVE_NUMBA
from cablexsec.mechanics.papailiou import slip_onset_curvature
from cablexsec.provenance import UnvalidatedModelWarning

pytestmark = pytest.mark.skipif(
    not HAVE_NUMBA, reason="anchored-wire solver without numba is slow"
)


@pytest.fixture(scope="module")
def menard():
    return cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)


@pytest.fixture(scope="module")
def history(menard):
    layer = cx.cable_to_fatigue_layers(menard, contact_tension_N=50e3)[0]
    t = np.linspace(0.0, 4 * math.pi, 121)
    return layer, 6 * slip_onset_curvature(layer.spec) * np.sin(t)


def _unvalidated(caught):
    return [w for w in caught if issubclass(w.category, UnvalidatedModelWarning)]


def _run(function, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = function(*args, **kwargs)
    return result, _unvalidated(caught)


@pytest.mark.parametrize("solver", ["clamp", "along_wire"])
def test_clamped_termination_warns_for_either_solver(history, solver):
    layer, k = history
    spec = TerminationSpec(0.0, solver=solver)
    result, caught = _run(compute_fatigue, [layer], k, 0 * k, termination=spec)
    assert len(caught) == 1
    message = str(caught[0].message)
    assert message == CLAMPED_TERMINATION_WARNING
    for part in (
        "upper estimate",
        "1.6-3.4",
        "1.0 times",
        "not from the end-fitting anchor",
        "SV11",
        "T2",
        "Saevik 2011",
    ):
        assert part in message
    assert result.clamp_correction[layer.name]["applied"]
    # Bitwise the same result as without the warning machinery.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quiet = compute_fatigue([layer], k, 0 * k, termination=spec)
    assert np.array_equal(
        result.D_bulk_per_fibre[layer.name], quiet.D_bulk_per_fibre[layer.name]
    )


def test_unanchored_or_absent_termination_does_not_warn(history):
    layer, k = history
    base, caught = _run(compute_fatigue, [layer], k, 0 * k)
    assert not caught
    none, caught = _run(
        compute_fatigue, [layer], k, 0 * k, termination=TerminationSpec(0.0, "none")
    )
    assert not caught
    assert np.array_equal(
        none.D_bulk_per_fibre[layer.name], base.D_bulk_per_fibre[layer.name]
    )


def test_cable_api_warns_for_any_clamped_layer(menard, history):
    _, k = history
    _, caught = _run(
        cx.compute_fatigue,
        menard,
        k,
        0.2 * k,
        termination={
            "inner_armour": TerminationSpec(0.0),
            "outer_armour": TerminationSpec(0.0, "none"),
        },
    )
    assert len(caught) == 1
    _, caught = _run(
        cx.compute_fatigue,
        menard,
        k,
        0.2 * k,
        termination={"inner_armour": TerminationSpec(0.0, "none")},
    )
    assert not caught


def test_spec_docstring_states_the_limitation():
    doc = TerminationSpec.__doc__ or ""
    assert "upper estimate" in doc and "UnvalidatedModelWarning" in doc
