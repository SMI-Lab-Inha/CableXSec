# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Fallbacks, clamps and skipped checks are explicit: an error or a warning.

Each case is one that used to return a value silently: a bisection that hit
its bracket, or a guard that was skipped on any error.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from _bundled import bundled

import cablexsec as cx
from cablexsec.analysis import fretting_regime as fr
from cablexsec.exceptions import CableInputError
from cablexsec.materials.sn_curves import get_sn_curve
from cablexsec.provenance import ExtrapolationWarning


@pytest.mark.parametrize(
    ("cycles", "match"),
    [
        (math.nan, "cycles must be finite and positive"),
        (0.0, "cycles must be finite and positive"),
        (-5.0, "cycles must be finite and positive"),
        (math.inf, "cycles must be finite and positive"),
        (1e-30, "even at 1e6 MPa"),
    ],
)
def test_range_at_cycles_refuses_a_life_outside_its_bracket(cycles, match):
    """The bisection on [1e-6, 1e6] MPa returned its bound (1e12 Pa for a
    life below the curve's at 1e6 MPa, 1 Pa for NaN) instead of a range."""
    curve = get_sn_curve("dnv_rpc203_t_curve")
    with pytest.raises(CableInputError, match=match):
        fr.range_at_cycles(curve, cycles)


def test_range_at_cycles_inside_the_bracket_is_unchanged():
    curve = get_sn_curve("dnv_rpc203_t_curve")
    value = fr.range_at_cycles(curve, 2e6)
    assert curve.cycles_to_failure(value / 1e6) == pytest.approx(2e6, rel=1e-9)


def test_range_at_cycles_of_a_steep_curve():
    """A steep curve's life overflows at the bracket's low end (1e-6 MPa)
    while its root is representable: compared as log10 lives."""
    from cablexsec.materials.sn_curves import SNcurve

    steep = SNcurve(name="steep", stress_unit="MPa", m1=40.0, log_C1=80.0)
    value = fr.range_at_cycles(steep, 1e6)
    assert value == pytest.approx(1e6 * 10.0 ** ((80.0 - 6.0) / 40.0), rel=1e-9)


def test_range_at_cycles_on_the_lower_bracket_end():
    """A root exactly at 1e-6 MPa (1 Pa) lies on the bracket, not outside."""
    from cablexsec.materials.sn_curves import SNcurve

    edge = SNcurve(name="edge", stress_unit="MPa", m1=1.0, log_C1=0.0)
    assert fr.range_at_cycles(edge, 1e6) == pytest.approx(1.0, rel=1e-12)
    with pytest.raises(CableInputError, match="even at 1e-6 MPa"):
        fr.range_at_cycles(edge, 2e6)


def test_mkappa_yield_guard_is_evaluated_or_said_to_be_skipped(monkeypatch):
    """The wire-yield guard of compute_mkappa swallowed every ValueError, so
    a bound that could not be evaluated left no trace. It now warns."""
    cable = bundled("inter_array_single_core_33kv")
    kappa = np.linspace(0.0, 0.02, 20)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExtrapolationWarning)
        cx.compute_mkappa(cable, kappa, tension_N=1e5)  # inside the bound

    def refuse(*args, **kwargs):
        raise CableInputError("no bound for this state")

    monkeypatch.setattr("cablexsec.api.wire_yield_curvature", refuse)
    with pytest.warns(ExtrapolationWarning, match="could not be evaluated .no bound"):
        cx.compute_mkappa(cable, kappa, tension_N=1e5)


def test_stalled_bending_test_inversion_is_refused():
    """With no new point to try, the inversion returned the closest cached
    point (here u = 0 for a target of 1) as if it had converged."""
    from cablexsec.analysis.beam_test import _MonotoneInverse
    from cablexsec.exceptions import CableSolverError

    flat = _MonotoneInverse(
        lambda v: np.zeros_like(v), slope_min=math.inf, slope_max=math.inf, tol=1e-9
    )
    with pytest.raises(CableSolverError, match="inversion stalled"):
        flat.solve(np.array([1.0]))
    linear = _MonotoneInverse(
        lambda v: 2.0 * v, slope_min=1.0, slope_max=3.0, tol=1e-12
    )
    v, u = linear.solve(np.array([0.5, 1.0]))
    np.testing.assert_allclose(u, [0.5, 1.0], atol=1e-12)
    np.testing.assert_allclose(v, [0.25, 0.5], atol=1e-12)


def test_laid_cores_without_a_friction_law_are_said_to_be_held(monkeypatch):
    """The kinematic history fell back to cores held to the cable axis on any
    error of the core friction law, without a trace."""
    from cablexsec.analysis import kinematics

    cable = bundled("inter_array_three_core_33kv")

    def refuse(*args, **kwargs):
        raise CableInputError("no core friction for this cable")

    monkeypatch.setattr(kinematics, "_core_field", refuse)
    kx = 0.01 * np.sin(np.linspace(0.0, 6.0, 12))
    with pytest.warns(ExtrapolationWarning, match="core friction law unavailable"):
        history = kinematics.kinematic_history(
            cable, kx, 0.0 * kx, tension_N=1e5, core_friction=False
        )
    assert any(
        "core friction law unavailable" in note
        for note in history.provenance.limitations
    )


def test_sheave_relays_the_warnings_of_its_internal_calls(monkeypatch):
    """compute_sheave evaluated its M-kappa loop, stress and capacity under
    ``simplefilter("ignore")``: a wire-yield or clamping warning about the
    caller's own sheave curvature was dropped. Each distinct one is relayed."""
    from cablexsec import api
    from cablexsec.analysis import sheave
    from cablexsec.provenance import MissingClampingStateWarning

    real = api.compute_mkappa

    def noisy(*args, **kwargs):
        warnings.warn("inner notice", MissingClampingStateWarning, stacklevel=2)
        warnings.warn("inner notice", MissingClampingStateWarning, stacklevel=2)
        return real(*args, **kwargs)

    monkeypatch.setattr(api, "compute_mkappa", noisy)
    cable = cx.load_cable(cx.iea_task49_66kv_yaml_path())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sheave.compute_sheave(cable, sheave_diameter_m=6.4, tension_N=50e3, n_ramp=10)
    relayed = [w for w in caught if str(w.message) == "inner notice"]
    assert len(relayed) == 1 and relayed[0].category is MissingClampingStateWarning
