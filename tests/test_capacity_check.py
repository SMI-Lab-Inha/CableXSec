# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
from __future__ import annotations

import math

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis import capacity as cap
from cablexsec.analysis.capacity_check import (
    EULER_GAMMA,
    CapacityEnvelope,
    capacity_extremes,
    check_capacity,
    check_capacity_series,
    gumbel_fit,
)
from cablexsec.exceptions import CableInputError


def _linear_envelope(**kw):
    tension = np.array([0.0, 5.0e4, 1.0e5])
    return CapacityEnvelope.from_points(
        tension, 0.5 - 2.0e-6 * tension, source="synthetic test curve", **kw
    )


# --------------------------------------------------------------------------- #
# Synthetic histories with known extremes                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.analytical_sanity
def test_known_maximum_and_exceedances():
    env = _linear_envelope(mbr_m=2.5, mbr_source="test MBR")
    time = np.arange(1000) * 0.1
    tension = np.full(time.size, 5.0e4)  # capacity 0.4 1/m
    kappa = np.full(time.size, 0.2)  # utilisation 0.5
    bursts = [(100, 110), (400, 405), (700, 730)]
    for a, b in bursts:
        kappa[a:b] = 0.48  # utilisation 1.2
    kappa[715] = 0.6  # utilisation 1.5, the maximum
    res = check_capacity(
        env, time, tension, kappa, np.zeros_like(kappa), thresholds=(1.0, 1.4)
    )
    assert res.max_utilisation == pytest.approx(1.5, rel=1e-12)
    assert res.time_of_max_s == pytest.approx(71.5)
    assert res.capacity_curvature_at_max_1_m == pytest.approx(0.4)
    above = res.exceedances[0]
    assert above.samples == sum(b - a for a, b in bursts)
    assert above.upcrossings == 3
    assert above.time_above_s == pytest.approx(0.1 * above.samples)
    assert res.exceedances[1].samples == 1
    assert np.allclose(res.utilisation, kappa / 0.4, rtol=1e-12)
    # Minimum bend radius: limit 0.4 1/m.
    assert res.mbr is not None
    assert res.mbr.curvature_max_1_m == pytest.approx(0.6)
    assert res.mbr.utilisation_max == pytest.approx(1.5)
    assert res.mbr.minimum_radius_m == pytest.approx(1 / 0.6)
    assert res.mbr.exceedance.upcrossings == 3


def test_biaxial_magnitude_and_tension_interpolation():
    env = _linear_envelope()
    time = np.arange(5.0)
    tension = np.array([0.0, 2.5e4, 5.0e4, 7.5e4, 1.0e5])
    kx = np.array([0.3, 0.27, 0.24, 0.21, 0.18])
    ky = np.array([0.4, 0.36, 0.32, 0.28, 0.24])
    res = check_capacity(env, time, tension, kx, ky)
    capacity = 0.5 - 2.0e-6 * tension
    assert np.allclose(res.utilisation, np.hypot(kx, ky) / capacity, rtol=1e-12)
    with pytest.raises(CableInputError, match="outside the envelope grid"):
        check_capacity(env, time, tension + 1.0e3, kx, ky)
    clamped = check_capacity(env, time, tension + 1.0e3, kx, ky, out_of_range="clamp")
    assert clamped.counts["outside_tension_grid"] == 1
    with pytest.raises(CableInputError, match="effective"):
        check_capacity(env, time, tension, kx, ky, tension_kind="true")
    with pytest.raises(CableInputError, match="source"):
        CapacityEnvelope.from_points([0.0, 1.0], [1.0, 1.0], source=" ")


# --------------------------------------------------------------------------- #
# Gumbel extremes                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.analytical_sanity
def test_gumbel_method_of_moments():
    maxima = np.array([1.0, 1.2, 0.9, 1.4, 1.1])
    fit = gumbel_fit(maxima, fractile=0.9)
    s = np.std(maxima, ddof=1)
    beta = math.sqrt(6) * s / math.pi
    assert fit.scale == pytest.approx(beta, rel=1e-14)
    assert fit.location == pytest.approx(np.mean(maxima) - EULER_GAMMA * beta)
    assert fit.expected == pytest.approx(np.mean(maxima))
    assert fit.fractile_value == pytest.approx(
        fit.location - beta * math.log(-math.log(0.9))
    )
    # Recovery of known parameters from a large sample.
    rng = np.random.default_rng(7)
    sample = rng.gumbel(2.0, 0.3, 40000)
    big = gumbel_fit(sample)
    assert big.location == pytest.approx(2.0, abs=0.01)
    assert big.scale == pytest.approx(0.3, rel=0.03)
    # Max of n independent Gumbel variables: location + beta ln n.
    shifted = gumbel_fit(sample, duration_ratio=10.0)
    assert shifted.location - big.location == pytest.approx(big.scale * math.log(10))
    with pytest.raises(CableInputError):
        gumbel_fit([1.0])


def test_seed_extremes_and_block_maxima():
    env = _linear_envelope()
    rng = np.random.default_rng(3)
    results = []
    for seed in range(6):
        time = np.arange(2000) * 0.5
        kappa = 0.1 + 0.05 * np.abs(rng.standard_normal(time.size))
        results.append(
            check_capacity(
                env, time, np.full(time.size, 5.0e4), kappa, blocks=8, name=f"s{seed}"
            )
        )
    fit = capacity_extremes(results)
    assert fit.n == 6
    assert fit.mean == pytest.approx(np.mean([r.max_utilisation for r in results]))
    block = results[0].block_extremes
    assert block is not None and block.n == 8
    assert block.duration_shift == pytest.approx(block.scale * math.log(8))


# --------------------------------------------------------------------------- #
# Built envelope                                                              #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def menard():
    return cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)


@pytest.fixture(scope="module")
def envelope(menard):
    grid = np.linspace(-2.0e4, 1.5e5, 18)
    with pytest.warns(cx.ExtrapolationWarning):
        return cx.build_capacity_envelope(
            menard, grid, mbr_m=1.2, mbr_source="test MBR"
        )


@pytest.mark.analytical_sanity
def test_level_curve_at_eta_is_the_wire_yield_curvature(menard, envelope):
    eta = envelope.utilisation_factor
    assert eta == cx.ARMOUR_UTILISATION_FACTORS["normal"]
    for i in (3, 9, 15):
        t = float(envelope.tension_N[i])
        direct = cap.wire_yield_curvature(menard, t, usage_limit=eta)
        assert envelope.kappa_capacity_1_m[i] == direct
        u, governing, _ = envelope.utilisation(np.array([t]), np.array([direct]))
        assert u[0] == pytest.approx(1.0, rel=1e-12)
        assert governing[0] == 0
        levels = cap.wire_yield_curvature_levels(menard, t, [0.3, eta])
        assert levels[1] == direct


def test_built_envelope_utilisation_is_monotone_and_flags_extrapolation(envelope):
    t = np.full(50, 6.0e4)
    kappa = np.linspace(0.0, 2.0, 50)
    u, _, counts = envelope.utilisation(t, kappa)
    assert np.all(np.diff(u) >= 0)
    top = envelope.levels[-1]
    assert counts["extrapolated_beyond_levels"] == int(np.count_nonzero(u > top))
    # Lateral buckling in compression above first slip.
    u_comp, governing, _ = envelope.utilisation(np.array([-2.0e4]), np.array([0.01]))
    assert governing[0] == 1
    assert u_comp[0] == pytest.approx(envelope.lateral_utilisation[0])


def test_reader_adapter_load_case_series(envelope):
    time = np.arange(200) * 0.1
    series = cx.LoadCaseSeries(
        time_s=time,
        curvature_x_1_m=0.3 * np.sin(time),
        curvature_y_1_m=np.zeros_like(time),
        tension_N=5.0e4 + 1.0e3 * np.cos(time),
        tension_kind="effective",
    )
    res = check_capacity_series(envelope, series)
    direct = check_capacity(
        envelope,
        time,
        series.tension_N,
        series.curvature_x_1_m,
        series.curvature_y_1_m,
    )
    assert np.array_equal(res.utilisation, direct.utilisation)
    unspecified = cx.LoadCaseSeries(
        time_s=time,
        curvature_x_1_m=series.curvature_x_1_m,
        curvature_y_1_m=series.curvature_y_1_m,
        tension_N=series.tension_N,
    )
    with pytest.raises(CableInputError, match="tension kind"):
        check_capacity_series(envelope, unspecified)


def test_sea_state_batch_capacity(tmp_path):
    from test_sea_state_batch import _runs

    from cablexsec.analysis.sea_state_batch import (
        CapacityBatchCheck,
        SeaStateBatchOptions,
        SectionRequest,
        run_sea_state_batch,
    )

    cable = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
    runs = _runs(tmp_path, cx.compute_stiffness(cable))
    envelope = cx.build_capacity_envelope(cable, np.linspace(1.5e4, 4.5e4, 7))
    sections = [SectionRequest("a", 2.0)]
    options = SeaStateBatchOptions(tension_basis="true")
    plain = run_sea_state_batch(cable, runs, sections, options=options)
    check = CapacityBatchCheck(envelope, thresholds=(0.05, 1.0))
    result = run_sea_state_batch(
        cable,
        runs,
        sections,
        options=options,
        capacity=check,
        output_directory=tmp_path / "out",
    )
    section = result.sections["a"]
    # Fatigue unchanged by the capacity check.
    assert (
        section.long_term.annual_damage == plain.sections["a"].long_term.annual_damage
    )
    assert "capacity" not in plain.manifest["results"]["a"]
    for run in runs:
        history = result.histories["a"][run.name]
        kx, ky = history.strain_gradient()
        direct = check_capacity(
            envelope, history.time_s, history.effective_tension_N, kx, ky
        )
        assert section.capacity_runs[run.name].max_utilisation == direct.max_utilisation
    summary = section.capacity
    assert summary is not None
    assert summary["max_utilisation"] == max(
        r.max_utilisation for r in section.capacity_runs.values()
    )
    assert summary["extremes"]["all_runs"]["fit"]["n"] == 2
    assert (tmp_path / "out" / "a_ss0_utilisation.npz").exists()
    assert result.manifest["results"]["a"]["capacity"]["max_run"] in {"ss0", "ss1"}
    assert result.manifest["capacity_check"]["tension_kind"] == "effective"


def test_usage_at_zero_curvature_is_never_understated() -> None:
    """Levels passed at zero curvature must not cap the utilisation.

    Beyond the tension capacity every design level up to one sits at zero
    curvature; the utilisation there is the usage at zero curvature (above
    one), and bending only adds to it.
    """
    import cablexsec as cx
    from cablexsec.cables.schema import load_cable

    cable = load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
    grid = np.linspace(0.0, 1.3e6, 27)
    envelope = cx.build_capacity_envelope(cable, grid, utilisation_factor=0.67)
    zero = np.asarray(envelope.zero_curvature_utilisation)
    at_zero, _, _ = envelope.utilisation(grid, np.zeros(grid.size))
    components = np.asarray(envelope.component_zero_utilisation)
    np.testing.assert_allclose(
        at_zero,
        np.maximum(np.maximum(zero, envelope.tension_utilisation), components),
    )
    beyond = zero > 1.0
    assert beyond.any() and np.all(at_zero[beyond] > 1.0)
    bent, _, _ = envelope.utilisation(grid, np.full(grid.size, 0.01))
    assert np.all(bent >= at_zero - 1e-12)
