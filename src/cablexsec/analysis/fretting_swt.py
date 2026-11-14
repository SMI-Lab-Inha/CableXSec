# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Smith-Watson-Topper (SWT) equation and a simplified contact-driver estimate.

The registered copper branch uses the SWT strain-life equation. [Poon2022]
uses it within a critical-plane methodology: the
crack-initiation life ``N_f`` solves the SWT fatigue-indicator equation
([Poon2022] Eq. 30)

.. math::

    \sigma_{max}\,\frac{\Delta\varepsilon_a}{2}
    = \frac{(\sigma_f')^2}{E}(2N_f)^{2b}
    + \sigma_f' \varepsilon_f' (2N_f)^{b+c},

combining the Basquin elastic line and the Coffin-Manson plastic line, with the
copper strain-life constants of :data:`cablexsec.materials.sn_curves.SWT_LIFE`
(Karlsen 2010 SPC Class-2 data, [Poon2022] Table 4). This module solves that
equation using a bracketed logarithmic root, with checked relative residual.
It does not implement Poon's FE stress-strain critical-plane search.

The helper SWT *driver* ``(sigma_max, Delta_eps_a/2)`` estimates trailing-edge
surface tensile stress using an assumed gross-sliding line contact,

.. math::  \sigma_{max} = 2 \mu p_0,

([Johnson1985] §7, sliding line contact), with the plane-strain free-surface
strain amplitude ``eps_a = sigma_max (1 - nu^2) / E``. This consumes the Hertz
peak pressure ``p_0`` and the contact friction ``mu`` (from the frozen
:mod:`cablexsec.mechanics.hertz_contact` / :mod:`mindlin_contact`); it is the
simplified sliding-line-contact estimate. Gross slip is assumed, not verified
from these inputs. It is not a resolved elliptical-contact or critical-plane
driver, and numerical solution of the life equation is not life validation.

Slip-cycle assessment
---------------------
:func:`swt_slip_cycle_damage` replaces the former one-driver-per-reversal
estimate. It rainflow-counts the tangential contact force history, skips every
counted cycle in which the contact does not slip (unless the Cattaneo-Mindlin
partial-slip option is selected), and evaluates one SWT driver per remaining
cycle from that cycle's tangential-load amplitude and the contact pressure.
Miner summation of ``count / N_f(SWT)`` follows the incremental fretting
damage sum of [Poon2023] Eqs. 7-9. For a cylindrical (line) contact under a
tangential load ``Q <= mu P`` the Cattaneo-Mindlin traction
([Johnson1985] §7.2) gives, through the half-plane surface-stress integral,
the trailing-edge surface stress ``2 mu p_0 sqrt(Q / (mu P))``. This closed
form is derived here from that traction; it is not quoted from a table. A
reversal of ``Delta Q`` superposes twice the monotonic traction for
``Delta Q / 2`` ([Johnson1985] §7.2, unloading), so the per-cycle stress
amplitude is ``2 mu p_0 sqrt(Q_a / (mu P))`` with ``Q_a = Delta Q / 2``. At
gross slip ``Q_a = mu P`` this recovers the sliding-line value ``2 mu p_0``.
The use of the trailing-edge tensile stress at interlayer (trellis) contacts
as a fretting indicator follows [HobbsRaoof1994].

References
----------
[Poon2022] Eqs. 30, 33, Table 4; [Poon2023] Eqs. 6-9; [Johnson1985] §7
(sliding-contact surface tensile stress ``2 mu p_0``; Cattaneo-Mindlin partial
slip of cylinders); [HobbsRaoof1994].
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from cablexsec.exceptions import CableInputError, CableSolverError
from cablexsec.materials.sn_curves import SWTLifeConstants

__all__ = [
    "PartialSlipCycles",
    "SlipCycleSWTResult",
    "partial_slip_surface_stress",
    "swt_driver_from_contact",
    "swt_life",
    "swt_slip_cycle_damage",
]

FloatArray = NDArray[np.float64]
PartialSlipCycles = Literal["skip", "cattaneo_mindlin"]


def swt_driver_from_contact(
    p0_Pa: float,
    mu: float,
    E_Pa: float,
    nu: float,
) -> tuple[float, float, float]:
    """Simplified ``(sigma_max, eps_a, SWT)`` for assumed gross sliding.

    ``sigma_max = 2 mu p_0`` ([Johnson1985] §7, trailing-edge surface tensile
    stress); ``eps_a = sigma_max (1 - nu^2) / E`` (plane-strain free-surface
    strain amplitude); ``SWT = sigma_max * eps_a`` [Pa].

    Parameters
    ----------
    p0_Pa : float
        Peak Hertz contact pressure at the wire-wire contact [Pa].
    mu : float
        Friction coefficient at the contact.
    E_Pa, nu : float
        Young's modulus [Pa] and Poisson's ratio of the wire material.

    This uses a sliding-line-contact stress and an assumed elastic strain
    amplitude. It does not search planes or infer slip from a loading history.
    Results are not an independently validated fretting-life driver.
    """
    if not all(math.isfinite(x) for x in (p0_Pa, mu, E_Pa, nu)):
        raise CableInputError("contact driver inputs must be finite")
    if min(p0_Pa, mu) < 0 or E_Pa <= 0 or not -1 < nu <= 0.5:
        raise CableInputError(
            "nonnegative pressure/friction, positive modulus "
            "and -1 < nu <= 0.5 required"
        )
    if p0_Pa == 0 or mu == 0:
        return 0.0, 0.0, 0.0
    log_sigma = math.log(2) + math.log(mu) + math.log(p0_Pa)
    log_eps = log_sigma + math.log1p(-nu) + math.log1p(nu) - math.log(E_Pa)
    try:
        result = tuple(math.exp(x) for x in (log_sigma, log_eps, log_sigma + log_eps))
    except OverflowError as error:
        raise CableInputError("contact driver exceeds floating-point range") from error
    if any(not math.isfinite(x) or x <= 0 for x in result):
        raise CableInputError("contact driver exceeds positive floating-point range")
    return result[0], result[1], result[2]


def swt_life(
    swt_Pa: float,
    const: SWTLifeConstants,
    *,
    n_init: float = 1.0e6,
    max_iter: int = 200,
    tol: float = 1.0e-12,
) -> float:
    """Positive formal life ``N_f`` solving the SWT equation in logarithmic form.

    Solves ``SWT = (sf'^2/E)(2N)^{2b} + sf' ef' (2N)^{b+c}`` for ``N`` via the
    bracketed scalar root. Returns ``inf`` only for finite non-positive SWT
    (no tensile driving indicator). Positive indicators with unrepresentable
    life raise ValueError; iteration exhaustion or failed residual checks raise
    RuntimeError. No unconverged iterate or arbitrary minimum life is returned.

    Fractional cycles are mathematical extrapolations of the fitted equation,
    not a demonstrated sub-cycle failure model. Material calibration range and
    applicability of the supplied driver must be assessed separately.

    Parameters
    ----------
    swt_Pa : float
        SWT fatigue-indicator value ``sigma_max * Delta_eps_a/2`` [Pa].
    const : SWTLifeConstants
        Material strain-life constants.
    n_init : float, optional
        Initial guess for N; narrows the analytical log-life bracket if inside it.
    max_iter : int, optional
        Maximum bracketed root iterations (positive integer).
    tol : float, optional
        Relative residual tolerance on the SWT equation, strictly between 0 and 1.

    Returns
    -------
    float
        Cycles to crack initiation ``N_f``.
    """
    sf, b, ef, c, e = (
        const.sigma_f_prime_Pa,
        const.b,
        const.eps_f_prime,
        const.c,
        const.E_Pa,
    )
    if not all(math.isfinite(x) for x in (swt_Pa, sf, b, ef, c, e, n_init, tol)):
        raise CableInputError(
            "SWT, material constants and solver controls must be finite"
        )
    if min(sf, e, n_init) <= 0 or ef < 0 or b >= 0 or c >= 0:
        raise CableInputError(
            "positive strength/modulus/initial life, nonnegative ductility "
            "and negative strain-life exponents are required"
        )
    if (
        not 0 < tol < 1
        or isinstance(max_iter, bool)
        or not isinstance(max_iter, Integral)
        or max_iter < 1
    ):
        raise CableInputError(
            "0 < tol < 1 and a positive integer max_iter are required"
        )
    if not math.isfinite(2 * b) or not math.isfinite(b + c):
        raise CableInputError("strain-life exponents exceed floating-point range")
    if swt_Pa <= 0:
        return math.inf

    log_target = math.log(swt_Pa)
    log_two = math.log(2)
    powers = [-2 * b]
    coefficients = [2 * math.log(sf) - math.log(e)]
    if ef > 0:
        powers.append(-(b + c))
        coefficients.append(math.log(sf) + math.log(ef))

    def residual(log_two_n: float) -> float:
        terms = [
            a - power * log_two_n for a, power in zip(coefficients, powers, strict=True)
        ]
        maximum = max(terms)
        log_rhs = maximum + math.log(sum(math.exp(x - maximum) for x in terms))
        return log_rhs - log_target

    # A term alone equals SWT at its individual root. At their largest root,
    # the sum is >= SWT. At the largest SWT/2 root, each term is <= SWT/2.
    lower = max(
        (a - log_target) / power for a, power in zip(coefficients, powers, strict=True)
    )
    upper = max(
        (a - log_target + math.log(len(powers))) / power
        for a, power in zip(coefficients, powers, strict=True)
    )
    smallest = math.log(math.ulp(0.0)) + log_two
    largest = math.log(sys.float_info.max) + log_two
    if lower > largest or upper < smallest:
        raise CableInputError("positive SWT life lies outside floating-point range")
    lower, upper = max(lower, smallest), min(upper, largest)
    tolerance = math.log1p(tol)
    f_lower, f_upper = residual(lower), residual(upper)
    if f_lower < -tolerance or f_upper > tolerance:
        raise CableInputError("positive SWT life lies outside floating-point range")
    if abs(f_lower) <= tolerance:
        root = lower
    elif abs(f_upper) <= tolerance:
        root = upper
    else:
        initial = math.log(n_init) + log_two
        if lower < initial < upper:
            if residual(initial) > 0:
                lower = initial
            else:
                upper = initial
        try:
            root = float(
                brentq(
                    residual,
                    lower,
                    upper,
                    xtol=5e-324,
                    rtol=4 * sys.float_info.epsilon,
                    maxiter=int(max_iter),
                )
            )
        except RuntimeError as error:
            raise CableSolverError(
                "SWT life did not converge within the iteration limit"
            ) from error
    try:
        life = math.exp(root - log_two)
    except OverflowError as error:
        raise CableInputError(
            "positive SWT life exceeds floating-point range"
        ) from error
    if not math.isfinite(life) or life <= 0:
        raise CableInputError("positive SWT life exceeds floating-point range")
    if abs(residual(math.log(life) + log_two)) > tolerance:
        raise CableSolverError(
            "SWT life did not converge to the requested relative residual"
        )
    return life


def partial_slip_surface_stress(p0_Pa: float, mu: float, load_ratio: float) -> float:
    """Trailing-edge surface stress ``2 mu p_0 sqrt(Q/(mu P))`` of a line contact [Pa].

    ``load_ratio`` is ``Q / (mu P)``; values above one are gross slip and give
    the sliding-line value ``2 mu p_0``. The expression follows from the
    Cattaneo-Mindlin traction of a cylindrical contact ([Johnson1985] §7.2)
    and is derived in this module (see the module notes).
    """
    if not all(math.isfinite(x) for x in (p0_Pa, mu, load_ratio)):
        raise CableInputError("surface-stress inputs must be finite")
    if p0_Pa < 0 or mu < 0 or load_ratio < 0:
        raise CableInputError("pressure, friction and load ratio must be nonnegative")
    return 2.0 * mu * p0_Pa * math.sqrt(min(load_ratio, 1.0))


@dataclass(frozen=True)
class SlipCycleSWTResult:
    """Per-cycle SWT fretting damage of one contact.

    Attributes
    ----------
    damage : float
        Miner sum ``sum(count / N_f(SWT))`` over the evaluated cycles.
    counted_cycles : float
        Total rainflow count of the tangential-load history (half cycles as
        one half).
    evaluated_cycles : float
        Count of the cycles that received an SWT driver.
    skipped_stick_cycles : float
        Count of the cycles in which the contact did not slip and which were
        skipped (``partial_slip="skip"``).
    amplitude_ratio, swt_Pa, life_cycles, count : numpy.ndarray
        Per evaluated cycle: ``Q_a / (mu P)``, SWT indicator [Pa], formal life
        and rainflow count.
    """

    damage: float
    counted_cycles: float
    evaluated_cycles: float
    skipped_stick_cycles: float
    amplitude_ratio: FloatArray
    swt_Pa: FloatArray
    life_cycles: FloatArray
    count: FloatArray


def swt_slip_cycle_damage(
    tangential_force: ArrayLike,
    slipping: ArrayLike,
    normal_force: float,
    p0_Pa: float,
    mu: float,
    E_Pa: float,
    nu: float,
    const: SWTLifeConstants,
    *,
    partial_slip: PartialSlipCycles = "skip",
    bulk_stress_Pa: ArrayLike | None = None,
) -> SlipCycleSWTResult:
    """Miner sum of per-cycle SWT fretting damage from a contact-force history.

    Parameters
    ----------
    tangential_force : array_like, shape (n,)
        Signed tangential contact load along the slip direction, in the unit
        of ``normal_force`` (force per crossing, or traction with a pressure).
    slipping : array_like of bool, shape (n,)
        ``True`` where the contact slipped during the increment ending at
        that sample (the first entry describes loading from the unloaded
        state).
    normal_force : float
        Constant positive normal load ``P`` of the contact (same unit).
    p0_Pa : float
        Peak Hertz pressure of the contact [Pa].
    mu : float
        Contact friction coefficient.
    E_Pa, nu : float
        Wire modulus and Poisson ratio for the elastic surface strain.
    const : SWTLifeConstants
        Strain-life constants of the wire material.
    partial_slip : {"skip", "cattaneo_mindlin"}, optional
        Treatment of counted cycles in which the contact does not slip.
        ``"skip"`` (default) assigns them no fretting damage. With
        ``"cattaneo_mindlin"`` they receive the partial-slip driver
        ``2 mu p_0 sqrt(Q_a/(mu P))``. [Poon2023] find partial slip the more
        damaging regime for unworn copper, so ``"skip"`` is not a bound.
    bulk_stress_Pa : array_like or None, optional
        Longitudinal wire stress at the contact. When supplied, its range and
        maximum over each cycle window are superposed in phase on the contact
        stress (an upper estimate of the combined driver).

    Returns
    -------
    SlipCycleSWTResult

    Notes
    -----
    Cycles are counted on ``tangential_force`` with
    :func:`cablexsec.analysis.fatigue.rainflow_cycles`. A cycle is a slip
    cycle if any increment between its two turning points slipped. Its
    amplitude ratio is ``Q_a / (mu P)`` with ``Q_a`` half the counted range,
    and its peak ratio ``(|Q_m| + Q_a) / (mu P)``, both capped at one. The
    driver is ``sigma_a = 2 mu p_0 sqrt(ratio_a)``,
    ``sigma_max = 2 mu p_0 sqrt(ratio_peak)``, the plane-strain surface strain
    amplitude ``sigma_a (1 - nu^2) / E`` and ``SWT = sigma_max eps_a``
    ([Poon2022] Eq. 30 solved by :func:`swt_life`). The peak value uses the
    monotonic Cattaneo-Mindlin state at the most loaded turning point, an
    approximation for cycles with a nonzero mean. The driver is a closed-form
    surface estimate for a line contact; it is neither a critical-plane search
    nor a wear-corrected life ([Poon2023] report that wear changes gross-slip
    lives by about one order of magnitude).
    """
    from cablexsec.analysis.fatigue import rainflow_cycles  # avoids import cycle

    if partial_slip not in ("skip", "cattaneo_mindlin"):
        raise CableInputError("partial_slip must be 'skip' or 'cattaneo_mindlin'")
    if np.iscomplexobj(tangential_force):
        raise CableInputError("tangential force must be real")
    q = np.asarray(tangential_force, dtype=np.float64)
    if q.ndim != 1 or not np.isfinite(q).all():
        raise CableInputError("tangential force must be a finite 1-D history")
    slip = np.asarray(slipping)
    if slip.shape != q.shape or slip.dtype != np.bool_:
        raise CableInputError(
            "slipping must be one Boolean per tangential-force sample"
        )
    bulk = None
    if bulk_stress_Pa is not None:
        bulk = np.asarray(bulk_stress_Pa, dtype=np.float64)
        if bulk.shape != q.shape or not np.isfinite(bulk).all():
            raise CableInputError(
                "bulk stress must be finite and match the force history"
            )
    if not all(math.isfinite(x) for x in (normal_force, p0_Pa, mu, E_Pa, nu)):
        raise CableInputError("contact inputs must be finite")
    if normal_force <= 0 or p0_Pa < 0 or mu < 0 or E_Pa <= 0 or not -1 < nu <= 0.5:
        raise CableInputError(
            "positive normal load and modulus, nonnegative pressure and friction, "
            "and -1 < nu <= 0.5 are required"
        )
    cap = mu * normal_force
    sigma_scale = 2.0 * mu * p0_Pa
    strain_factor = (1.0 - nu * nu) / E_Pa
    counted = evaluated = skipped = 0.0
    ratios: list[float] = []
    swts: list[float] = []
    lives: list[float] = []
    counts: list[float] = []
    cache: dict[float, float] = {}
    damage = 0.0
    for span, mean, count, i0, i1 in rainflow_cycles(q):
        counted += count
        slipped = bool(np.any(slip[i0 + 1 : i1 + 1]))
        if not slipped and partial_slip == "skip":
            skipped += count
            continue
        if cap > 0:
            ratio_a = min(0.5 * span / cap, 1.0)
            ratio_p = min((abs(mean) + 0.5 * span) / cap, 1.0)
        else:
            ratio_a = ratio_p = 1.0
        sigma_a = sigma_scale * math.sqrt(ratio_a)
        sigma_max = sigma_scale * math.sqrt(ratio_p)
        if bulk is not None:
            window = bulk[i0 : i1 + 1]
            sigma_a += 0.5 * float(window.max() - window.min())
            sigma_max += float(window.max())
        swt = sigma_max * sigma_a * strain_factor if sigma_max > 0 else 0.0
        life = cache.get(swt)
        if life is None:
            life = swt_life(swt, const)
            cache[swt] = life
        evaluated += count
        ratios.append(ratio_a)
        swts.append(swt)
        lives.append(life)
        counts.append(count)
        if math.isfinite(life):
            damage += count / life
    if not math.isfinite(damage):
        raise CableInputError(
            "accumulated fretting damage exceeds floating-point range"
        )
    return SlipCycleSWTResult(
        damage=damage,
        counted_cycles=counted,
        evaluated_cycles=evaluated,
        skipped_stick_cycles=skipped,
        amplitude_ratio=np.asarray(ratios, dtype=np.float64),
        swt_Pa=np.asarray(swts, dtype=np.float64),
        life_cycles=np.asarray(lives, dtype=np.float64),
        count=np.asarray(counts, dtype=np.float64),
    )
