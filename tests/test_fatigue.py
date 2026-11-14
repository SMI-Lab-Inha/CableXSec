# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for the fatigue pipeline (§6.12): gate A12, BM3 (Nasution), unit checks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cablexsec.analysis.fatigue import (
    FatigueLayer,
    compute_fatigue,
    miner_damage,
    rainflow_cycles,
)
from cablexsec.materials.sn_curves import SNcurve, get_sn_curve
from cablexsec.mechanics.mkappa import HelicalLayerSpec

# Test-taxonomy category.
pytestmark = pytest.mark.analytical_sanity


def _const_amplitude_series(amp_Pa: float, n_cycles: int) -> np.ndarray:
    """n closed, fully-reversed cycles of amplitude amp_Pa (mean 0).

    The series ``[A, -A] * n + [A]`` rainflow-decomposes into exactly n cycles
    of range ``2A`` with zero mean (verified: total count == n).
    """
    return np.array([amp_Pa, -amp_Pa] * n_cycles + [amp_Pa], dtype=np.float64)


# --------------------------------------------------------------------------- #
# A12 — Miner closure (the A11 pattern applied to fatigue; analytical, exact)  #
# --------------------------------------------------------------------------- #
def test_A12_miner_closure_exact() -> None:
    """Constant-amplitude history: D_bulk == n / N_f(2*sigma_a), error < 1e-10."""
    sn = get_sn_curve("nasution_2014_copper")
    uts = 220.0e6
    amp = 80.0e6  # sigma_a = 80 MPa -> range 160 MPa
    for n in (1, 5, 37):
        series = _const_amplitude_series(amp, n)
        d = miner_damage(series, sn, uts)
        n_f = sn.cycles_to_failure(2.0 * amp / 1.0e6)  # range basis, MPa
        assert d == pytest.approx(n / n_f, rel=1e-10)


def test_A12_zero_amplitude_no_damage() -> None:
    sn = get_sn_curve("nasution_2014_copper")
    flat = np.zeros(50)
    assert miner_damage(flat, sn, 220.0e6) == 0.0


def test_two_sample_open_excursion_is_half_cycle() -> None:
    sn = get_sn_curve("nasution_2014_copper")
    series = np.array([-80.0e6, 80.0e6])
    assert list(rainflow_cycles(series)) == [(160.0e6, 0.0, 0.5, 0, 1)]
    assert miner_damage(series, sn, 220.0e6) == pytest.approx(
        0.5 / sn.cycles_to_failure(160.0)
    )


@pytest.mark.parametrize("values", [[], [3.0], [3.0, 3.0], [3.0, 3.0, 3.0]])
def test_constant_record_has_no_physical_cycles(values) -> None:
    assert list(rainflow_cycles(np.asarray(values))) == []


@pytest.mark.parametrize("values", [[1.0, np.nan], [np.inf], [[1.0, 2.0]]])
def test_invalid_stress_record_is_rejected(values) -> None:
    with pytest.raises(ValueError, match="one-dimensional and finite"):
        miner_damage(np.asarray(values), get_sn_curve("nasution_2014_copper"), 220.0e6)


# --------------------------------------------------------------------------- #
# Stress-range and threshold behaviour                                        #
# --------------------------------------------------------------------------- #
def test_constant_shift_does_not_change_range_damage() -> None:
    """A stress-range curve is invariant to a constant mean-stress shift."""
    sn = get_sn_curve("nasution_2014_copper")
    uts = 220.0e6
    amp = 50.0e6
    zero_mean = _const_amplitude_series(amp, 10)
    with_mean = zero_mean + 60.0e6  # shift mean to +60 MPa
    assert miner_damage(with_mean, sn, uts) == pytest.approx(
        miner_damage(zero_mean, sn, uts)
    )


def test_runout_below_threshold_is_zero() -> None:
    """Amplitudes below a curve's fatigue limit give infinite life -> D = 0."""
    sn = SNcurve(
        name="threshold_demo", stress_unit="MPa", m1=3.0, log_C1=12.0, threshold=50.0
    )
    # range 40 MPa < 50 MPa threshold -> run-out.
    series = _const_amplitude_series(20.0e6, 100)
    assert miner_damage(series, sn, 220.0e6) == 0.0


# --------------------------------------------------------------------------- #
# BM3 — Nasution (2014b) copper conductor individual-wire S-N                  #
# --------------------------------------------------------------------------- #
@pytest.mark.full_scale
def test_BM3_nasution_curve_reproduces_table1_within_scatter() -> None:
    """Registered curve vs the three [Nasution2014b] Table 1 stress levels."""
    sn = get_sn_curve("nasution_2014_copper")
    # Table 1 geometric-mean lives (nine outer wires).
    levels = {
        130.0: math.exp(np.mean(np.log([6_799_803, 8_212_235, 8_878_805]))),
        160.0: math.exp(np.mean(np.log([2_482_476, 2_586_506, 3_176_399]))),
        190.0: math.exp(np.mean(np.log([706_421, 728_768, 831_666]))),
    }
    for d_sigma, n_mean in levels.items():
        n_curve = sn.cycles_to_failure(d_sigma)
        # Single power-law fit to scattered data: each level within ~15%
        # (the data's own +/-2 SD band spans a factor of ~2).
        assert abs(n_curve - n_mean) / n_mean < 0.15


@pytest.mark.full_scale
def test_BM3_pipeline_reproduces_curve_exactly() -> None:
    """One closed cycle at range 160 MPa -> D = 1 / N_f(160), error < 1e-10."""
    sn = get_sn_curve("nasution_2014_copper")
    series = _const_amplitude_series(80.0e6, 1)  # range 160 MPa, mean 0
    d = miner_damage(series, sn, 220.0e6)
    assert d == pytest.approx(1.0 / sn.cycles_to_failure(160.0), rel=1e-10)


# --------------------------------------------------------------------------- #
# Full pipeline (compute_fatigue) — consumes the frozen M-κ core              #
# --------------------------------------------------------------------------- #
def _armour_spec() -> HelicalLayerSpec:
    return HelicalLayerSpec(
        n_wires=24,
        r_helix_m=0.082,
        lay_angle_rad=math.radians(16.0),
        E_wire_Pa=210.0e9,
        A_wire_m2=math.pi * 0.005**2 / 4.0,
        xi=0.98,
        mu=0.2,
        F_n_axi_N=80.0,
    )


def _history(amp: float, n_cycles: int, pts: int = 200) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi * n_cycles, pts)
    return amp * np.sin(t)


def test_compute_fatigue_populates_result_and_critical_wire() -> None:
    layer = FatigueLayer(
        spec=_armour_spec(),
        sn_bulk=get_sn_curve("dnv_rpc203_t_curve"),
        sigma_UTS_Pa=1400.0e6,
        name="inner_armour",
    )
    kx = _history(0.05, 5)
    ky = np.zeros_like(kx)
    res = compute_fatigue([layer], kx, ky)
    assert "inner_armour" in res.D_total_per_wire
    assert res.D_total_per_wire["inner_armour"].shape == (24,)
    assert np.all(res.D_total_per_wire["inner_armour"] >= 0.0)
    assert res.critical_layer == "inner_armour"
    assert math.isfinite(res.phi_critical_deg)
    assert res.governing == "bulk"  # no fretting/termination configured


def test_termination_branch_amplifies_bulk() -> None:
    base = FatigueLayer(
        spec=_armour_spec(),
        sn_bulk=get_sn_curve("dnv_rpc203_t_curve"),
        sigma_UTS_Pa=1400.0e6,
        name="a",
    )
    term = FatigueLayer(
        spec=_armour_spec(),
        sn_bulk=get_sn_curve("dnv_rpc203_t_curve"),
        sigma_UTS_Pa=1400.0e6,
        name="a",
        SCF=1.5,
        termination=True,
    )
    kx = _history(0.05, 4)
    ky = np.zeros_like(kx)
    d0 = compute_fatigue([base], kx, ky).D_total_per_wire["a"].max()
    d1 = compute_fatigue([term], kx, ky).D_total_per_wire["a"].max()
    # D_total = D_bulk + D_bulk * (SCF^m - 1), so SCF>1 raises damage.
    assert d1 > d0


def test_unsupported_steel_fretting_curve_is_blocked() -> None:
    from cablexsec.exceptions import BlockedSNCurveError  # noqa: PLC0415

    with pytest.raises(BlockedSNCurveError):
        get_sn_curve("perier_2009_steel")


def test_explicit_custom_slip_curve_uses_crossing_force() -> None:
    """An explicitly supplied slip-life curve exercises the documented hook."""
    custom = SNcurve(
        name="test_only_slip_curve",
        stress_unit="um",
        m1=2.0,
        log_C1=10.0,
        threshold=0.0,
    )
    layer = FatigueLayer(
        spec=_armour_spec(),
        sn_bulk=get_sn_curve("dnv_rpc203_t_curve"),
        sigma_UTS_Pa=1400.0e6,
        name="custom",
        sn_fretting=custom,
        K_t_contact_N_per_m=1.0e8,
        fretting_mu=0.2,
        crossing_density_per_m=20.0,
        transverse_ratio=1.0,
        transverse_map="legacy",
    )
    kx = _history(0.01, 4)
    force = np.full(kx.shape, 80.0)
    res = compute_fatigue(
        [layer],
        kx,
        np.zeros_like(kx),
        contact_force_N={"custom": force},
    )
    assert float(np.sum(res.D_fretting_per_wire["custom"])) > 0.0
    assert res.D_fretting_fraction > 0.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"kappa_y": np.zeros(3)}, "same shape"),
        ({"tension_N": np.zeros(3)}, "tension_N"),
        ({"thermal_stress_Pa": {"a": np.zeros(3)}}, "thermal_stress_Pa"),
    ],
)
def test_compute_fatigue_rejects_mismatched_histories(
    kwargs: dict[str, object], match: str
) -> None:
    kx = np.zeros(4)
    ky = np.zeros(4)
    if "kappa_y" in kwargs:
        ky = kwargs.pop("kappa_y")  # type: ignore[assignment]
    with pytest.raises(ValueError, match=match):
        compute_fatigue([], kx, ky, **kwargs)  # type: ignore[arg-type]


def test_compute_fatigue_empty_layer_set_is_well_defined() -> None:
    kappa = np.zeros(4)
    res = compute_fatigue([], kappa, kappa)
    assert math.isnan(res.phi_critical_deg)
    assert res.critical_layer == ""
    assert res.D_fretting_fraction == 0.0


def test_copper_swt_fretting_branch_positive() -> None:
    """Copper conductor: the SWT branch yields positive fretting damage."""
    from cablexsec.materials.sn_curves import get_swt_life

    layer = FatigueLayer(
        spec=_armour_spec(),  # geometry stand-in; copper SWT branch
        sn_bulk=get_sn_curve("nasution_2014_copper"),
        sigma_UTS_Pa=220.0e6,
        name="cu",
        swt_const=get_swt_life("copper_poon2022"),
        contact_p0_Pa=26.0e6,
        contact_nu=0.34,
        fretting_mu=0.8,
        E_eff_axial_Pa=97.0e9,
        K_t_contact_N_per_m=1.0e6,
        crossing_density_per_m=50.0,
        contact_line_load_N_per_m=100.0,
    )
    kx = _history(0.01, 6)
    res = compute_fatigue([layer], kx, np.zeros_like(kx))
    assert float(np.sum(res.D_fretting_per_wire["cu"])) > 0.0
    assert res.D_fretting_fraction > 0.0
    legacy = compute_fatigue(
        [layer], kx, np.zeros_like(kx), copper_contact_model="legacy_gross_sliding"
    )
    assert float(np.sum(legacy.D_fretting_per_wire["cu"])) > 0.0
    assert "legacy" in legacy.provenance.method
