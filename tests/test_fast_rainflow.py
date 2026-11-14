# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""The compiled rainflow kernel must reproduce the reference counting exactly."""

import math

import numpy as np
import pytest

from cablexsec.analysis.fast_rainflow import fatigue_rows
from cablexsec.analysis.fatigue import rainflow_cycles
from cablexsec.materials.sn_curves import SN_CURVES, SNcurve


def fatigue_pair(stress, sn_curve, duration):
    """Reference: Miner damage and 1 Hz equivalent range [MPa] from the
    package rainflow count."""
    damage = power_sum = 0.0
    for span, _mean, count, _first, _last in rainflow_cycles(np.asarray(stress)):
        damage += count / sn_curve.cycles_to_failure(span / 1e6)
        power_sum += count * span**sn_curve.m1
    return damage, (power_sum / duration) ** (1 / sn_curve.m1) / 1e6


CURVES = [
    SN_CURVES["saevik_2011_flat_steel"],
    SN_CURVES["dnv_rpc203_t_curve"],
    SNcurve(
        name="test_threshold",
        stress_unit="MPa",
        m1=3.0,
        log_C1=12.0,
        m2=5.0,
        log_C2=15.5,
        threshold=5.0,
    ),
]


def histories(rng):
    out = [
        rng.normal(0, 30e6, 4000),
        np.round(rng.normal(0, 30e6, 4000) / 5e6) * 5e6,  # plateaus and ties
        np.cumsum(rng.normal(0, 1e6, 4000)),
        np.linspace(0, 50e6, 100),  # monotonic
        np.full(50, 7e6),  # constant
        np.array([0.0, 40e6]),  # two samples
        np.array([1e6, 1e6, 3e6, 3e6, -2e6, -2e6, 4e6, 4e6]),  # plateau reversals
        np.r_[
            np.zeros(10), 20e6 * np.sin(np.linspace(0, 40 * np.pi, 3000)), np.zeros(10)
        ],
    ]
    return out


@pytest.mark.parametrize("curve", CURVES, ids=lambda c: c.name)
@pytest.mark.parametrize("parallel", [False, True])
def test_matches_reference(curve, parallel):
    rng = np.random.default_rng(7)
    for h in histories(rng):
        d_ref, e_ref = fatigue_pair(h, curve, 3600.0)
        d, e = fatigue_rows(h[None, :], curve, 3600.0, parallel=parallel)
        assert d[0] == pytest.approx(d_ref, rel=1e-13, abs=0.0)
        assert e[0] == pytest.approx(e_ref, rel=1e-13, abs=0.0)


def test_rows_independent():
    rng = np.random.default_rng(3)
    block = rng.normal(0, 20e6, (16, 2000))
    curve = SN_CURVES["saevik_2011_flat_steel"]
    d, e = fatigue_rows(block, curve, 1000.0)
    for i in range(block.shape[0]):
        d_ref, e_ref = fatigue_pair(block[i], curve, 1000.0)
        assert d[i] == pytest.approx(d_ref, rel=1e-13)
        assert e[i] == pytest.approx(e_ref, rel=1e-13)
    assert all(math.isfinite(x) for x in d)


@pytest.mark.parametrize("parallel", [False, True])
def test_fibre_fatigue_matches_at_fibre(parallel):
    from cablexsec.analysis.fast_rainflow import fibre_fatigue

    rng = np.random.default_rng(5)
    shape = (12, 3000)
    mean = rng.normal(100e6, 20e6, shape)
    radial = rng.normal(0, 15e6, shape)
    tangential = rng.normal(0, 10e6, shape)
    angles = np.arange(8) * (2 * np.pi / 8)
    curve = SN_CURVES["saevik_2011_flat_steel"]
    d, e, lo, hi = fibre_fatigue(
        mean, radial, tangential, angles, curve, 900.0, parallel=parallel
    )
    for f, a in enumerate(angles):
        stress = mean + radial * math.cos(a) + tangential * math.sin(a)
        for w in range(shape[0]):
            d_ref, e_ref = fatigue_pair(stress[w], curve, 900.0)
            assert d[w, f] == pytest.approx(d_ref, rel=1e-13)
            assert e[w, f] == pytest.approx(e_ref, rel=1e-13)
            assert lo[w, f] == stress[w].min() / 1e6
            assert hi[w, f] == stress[w].max() / 1e6
