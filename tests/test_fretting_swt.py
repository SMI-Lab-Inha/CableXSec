# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""SWT numerical closure and literature-scale screening, not life validation."""

from __future__ import annotations

import dataclasses
import math

import pytest

from cablexsec.analysis.fretting_swt import swt_driver_from_contact, swt_life
from cablexsec.materials.sn_curves import get_swt_life

# These checks do not reproduce measured fatigue lives or the FE plane search.
pytestmark = pytest.mark.analytical_sanity


def _swt_rhs(two_n: float, const) -> float:
    """SWT equation right-hand side ([Poon2022] Eq. 30), in Pa."""
    elastic = const.sigma_f_prime_Pa**2 / const.E_Pa
    plastic = const.sigma_f_prime_Pa * const.eps_f_prime
    return elastic * two_n ** (2.0 * const.b) + plastic * two_n ** (const.b + const.c)


# --------------------------------------------------------------------------- #
# A13 — SWT closure (analytical; the A11/A12 pattern applied to strain-life)   #
# --------------------------------------------------------------------------- #
def test_A13_swt_closure_residual() -> None:
    """The returned N satisfies the SWT equation to < 1e-10 (relative)."""
    const = get_swt_life("copper_poon2022")
    for swt_Pa in (5.0e3, 2.0e4, 1.0e5, 5.0e5):
        n = swt_life(swt_Pa, const)
        residual = _swt_rhs(2.0 * n, const) - swt_Pa
        assert abs(residual) / swt_Pa < 1e-10


def test_A13_basquin_limit_closed_form() -> None:
    """With the plastic (Coffin-Manson) term off, N reduces to the Basquin line."""
    const = dataclasses.replace(get_swt_life("copper_poon2022"), eps_f_prime=0.0)
    elastic = const.sigma_f_prime_Pa**2 / const.E_Pa
    swt_Pa = 2.0e4
    # Closed form: SWT = elastic * (2N)^{2b}  =>  2N = (SWT/elastic)^{1/(2b)}.
    two_n_exact = (swt_Pa / elastic) ** (1.0 / (2.0 * const.b))
    n = swt_life(swt_Pa, const)
    assert 2.0 * n == pytest.approx(two_n_exact, rel=1e-10)


def test_swt_life_monotonic_and_runout() -> None:
    const = get_swt_life("copper_poon2022")
    assert swt_life(1.0e5, const) < swt_life(1.0e4, const)  # higher SWT -> shorter
    assert swt_life(0.0, const) == math.inf  # no driving stress -> infinite life


def test_swt_driver_2mu_p0() -> None:
    """sigma_max = 2 mu p0 ([Johnson1985] §7); eps_a = sigma_max (1-nu^2)/E."""
    sigma_max, eps_a, swt = swt_driver_from_contact(26.0e6, 0.8, 97.0e9, 0.34)
    assert sigma_max == pytest.approx(2.0 * 0.8 * 26.0e6)
    assert eps_a == pytest.approx(sigma_max * (1.0 - 0.34**2) / 97.0e9)
    assert swt == pytest.approx(sigma_max * eps_a)


# --------------------------------------------------------------------------- #
# BM6 — retained literature-scale sanity check, not reproduction/validation #
# --------------------------------------------------------------------------- #
def test_BM6_poon_representative_life() -> None:
    """Simplified contact -> SWT -> N_f remains in a broad literature-scale band.

    [Poon2022]: as-received pure copper, peak contact pressure p0 ~ 26 MPa (Fig.
    7A, ~4 deg crossing angle), CoF mu = 0.8, gross slip (Delta = +-0.5 um).
    Reported crack-initiation lives: 9.7e7 cycles (3D crossed cylinder) and
    1.2e8 (2D cylinder-on-flat), within a representative-SPC band of 1e6-1e8
    (Fig. 8A). Our driver is a different model: a matching order of magnitude
    does not reproduce those FE predictions or validate physical fatigue life.
    """
    const = get_swt_life("copper_poon2022")  # E = 97 GPa, Karlsen constants
    _sig, _eps, swt = swt_driver_from_contact(
        p0_Pa=26.0e6, mu=0.8, E_Pa=const.E_Pa, nu=0.34
    )
    n_f = swt_life(swt, const)
    # Inside Poon's representative SPC band, and within a factor of ~3 of the
    # reported 3D prediction (9.7e7). This loose bound is a historical numerical
    # sanity check, not an experimental scatter interval or acceptance criterion.
    assert 1.0e6 <= n_f <= 2.0e8
    assert (1.0 / 3.0) <= n_f / 9.7e7 <= 3.0


def test_BM6_higher_pressure_shortens_life() -> None:
    """Poon: doubling peak pressure cuts life by ~2 orders -> strong reduction."""
    const = get_swt_life("copper_poon2022")
    _s1, _e1, swt_lo = swt_driver_from_contact(26.0e6, 0.8, const.E_Pa, 0.34)
    _s2, _e2, swt_hi = swt_driver_from_contact(52.0e6, 0.8, const.E_Pa, 0.34)
    assert swt_life(swt_hi, const) < swt_life(swt_lo, const)
