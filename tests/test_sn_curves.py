# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for the S-N curve library."""

from __future__ import annotations

import math

import pytest

from cablexsec.materials import sn_curves

FULLY_SPECIFIED = [
    "dnv_rpc203_t_curve",
    "saevik_2011_flat_steel",
    "nasution_2014_copper",
]


# Test-taxonomy category.
pytestmark = pytest.mark.analytical_sanity


def test_fully_specified_curves_present() -> None:
    for name in FULLY_SPECIFIED:
        assert name in sn_curves.list_sn_curves()


def test_every_curve_has_a_reference() -> None:
    for name in sn_curves.list_sn_curves():
        curve = sn_curves.get_sn_curve(name)
        assert curve.reference.strip()


def test_unknown_curve_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="Unknown S-N curve"):
        sn_curves.get_sn_curve("not_a_curve")


def test_saevik_basquin_single_slope() -> None:
    """Sævik steel curve: log10 N = 15.01 - 4 log10 Delta sigma."""
    c = sn_curves.get_sn_curve("saevik_2011_flat_steel")
    s = 100.0
    expected = 10.0 ** (c.log_C1 - c.m1 * math.log10(s))
    assert c.cycles_to_failure(s) == pytest.approx(expected, rel=1e-12)


def test_perier_slip_curve_is_blocked() -> None:
    assert "perier_2009_steel" in sn_curves.BLOCKED_SN_CURVES
    assert "perier_2009_steel" not in sn_curves.list_sn_curves()
    with pytest.raises(sn_curves.BlockedSNCurveError):
        sn_curves.get_sn_curve("perier_2009_steel")


def test_swt_life_constants_registered_and_cited() -> None:
    swt = sn_curves.get_swt_life("copper_poon2022")
    assert swt.sigma_f_prime_Pa == pytest.approx(1035.0e6)
    assert swt.b == pytest.approx(-0.175)
    assert swt.eps_f_prime == pytest.approx(0.8989)
    assert swt.c == pytest.approx(-0.601)
    assert "Poon2022" in swt.reference


def test_removed_poon_power_law_is_gone() -> None:
    """The mis-cited poon_2022_copper power-law curve must not be present."""
    assert "poon_2022_copper" not in sn_curves.list_sn_curves()


def test_lead_sheath_sn_is_blocked_not_invented() -> None:
    """cigre_tb623_lead stays a BLOCKED placeholder (Phase-2), not a fake curve."""
    name = "cigre_tb623_lead"
    assert name in sn_curves.BLOCKED_SN_CURVES
    assert name not in sn_curves.list_sn_curves()
    with pytest.raises(sn_curves.BlockedSNCurveError, match="BLOCKED"):
        sn_curves.get_sn_curve(name)


def test_old_bs7910_placeholder_is_renamed_away() -> None:
    """The mis-cited bs7910_class_d_aluminium name is gone (renamed to EN 1999)."""
    assert "bs7910_class_d_aluminium" not in sn_curves.BLOCKED_SN_CURVES
    assert "bs7910_class_d_aluminium" not in sn_curves.list_sn_curves()


def test_en1999_aluminium_curve_known_answer() -> None:
    """The EN 1999-1-3 71-7 curve registers correctly: N_f = C·Δσ^-7 closed form,
    and the constant-amplitude cut-off at Δσ_L = 40.6 MPa gives infinite life."""
    import math  # noqa: PLC0415

    c = sn_curves.get_sn_curve("en1999_1_3_aluminium")
    assert c.m1 == 7.0 and c.m2 is None  # single slope
    # anchor: Δσ_c = 71 MPa (range) at N_C = 2e6.
    assert c.cycles_to_failure(71.0) == pytest.approx(2.0e6, rel=2e-3)
    # closed form at an arbitrary range above the cut-off.
    ds = 100.0
    expected = 10.0 ** (c.log_C1 - 7.0 * math.log10(ds))
    assert c.cycles_to_failure(ds) == pytest.approx(expected, rel=1e-12)
    # cut-off: just below 40.6 MPa range -> run-out (infinite life).
    assert math.isinf(c.cycles_to_failure(40.5))
    assert math.isfinite(c.cycles_to_failure(41.0))


def test_unknown_curve_is_plain_keyerror_not_blocked() -> None:
    """An unknown name is a KeyError but not the BLOCKED subtype."""
    with pytest.raises(KeyError) as exc:
        sn_curves.get_sn_curve("not_a_curve")
    assert not isinstance(exc.value, sn_curves.BlockedSNCurveError)


def test_nonpositive_stress_is_runout() -> None:
    c = sn_curves.get_sn_curve("dnv_rpc203_t_curve")
    assert c.cycles_to_failure(0.0) == math.inf
    assert c.cycles_to_failure(-10.0) == math.inf


def test_dnv_t_curve_is_bilinear_with_continuous_knee() -> None:
    """Bilinear branches must meet at the knee (N = 1.8e6 for the T curve in seawater)."""
    c = sn_curves.get_sn_curve("dnv_rpc203_t_curve")
    assert c.is_bilinear
    knee = c.transition_stress()
    assert knee is not None
    # Both branch formulae give the same N at the knee stress (continuity).
    n_branch1 = 10.0 ** (c.log_C1 - c.m1 * math.log10(knee))
    n_branch2 = 10.0 ** (c.log_C2 - c.m2 * math.log10(knee))
    assert n_branch1 == pytest.approx(n_branch2, rel=1e-9)
    # [DNV-RP-C203] (2024) Table 2-3, seawater with cathodic protection: the
    # knee is at N = 1.8e6; the rounded (m, log a) constants place it within
    # 0.1 % of that.
    assert n_branch1 == pytest.approx(1.8e6, rel=5e-3)


def test_bilinear_branch_selection_monotonic() -> None:
    """Cycles to failure must increase as stress decreases."""
    c = sn_curves.get_sn_curve("dnv_rpc203_t_curve")
    stresses = [200.0, 100.0, 52.6, 30.0, 10.0]
    lives = [c.cycles_to_failure(s) for s in stresses]
    assert all(a < b for a, b in zip(lives, lives[1:], strict=False))
