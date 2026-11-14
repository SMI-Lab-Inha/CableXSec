# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Strain-life (Coffin--Manson--Basquin) fatigue of metallic sheaths and foils.

Gap P6. A metallic sheath, foil or tape that is bonded to, or carried by, a
polymer body is strain controlled: its strain follows the body
(:mod:`cablexsec.mechanics.core_kinematics`), and at the strain amplitudes of
dynamic bending (0.1--2 %) it yields every cycle. Its life is then a
strain-life problem, not an elastic stress-range S-N problem.

Strain-life curve ([Basquin1910], [Coffin1954], [Manson1954]; notation of
[Dowling2013] Ch. 14):

.. math::

    \varepsilon_a = \frac{\sigma_f'}{E}(2N_f)^b + \varepsilon_f'(2N_f)^c .

Mean-stress corrections:

* ``"morrow"`` ([Morrow1968]):
  ``epsilon_a = (sigma_f' - sigma_m)/E (2N)^b + epsilon_f' (2N)^c``;
* ``"swt"`` ([SmithWatsonTopper1970]):
  ``sigma_max epsilon_a = sigma_f'^2/E (2N)^(2b) + sigma_f' epsilon_f' (2N)^(b+c)``,
  no damage for ``sigma_max <= 0``;
* ``"none"``: the curve at the counted amplitude.

The stresses of a strain history follow the cyclic stress--strain curve
``epsilon = sigma/E + (sigma/K')^(1/n')`` with Masing branches
``Delta epsilon = Delta sigma/E + 2 (Delta sigma / 2K')^(1/n')`` and the
material memory of closed loops ([Dowling2013] Sec. 14.4). Without published
``K'``, ``n'`` the compatibility estimate ``n' = b/c``,
``K' = sigma_f' / epsilon_f'^(b/c)`` is used: it makes the cyclic curve
consistent with the strain-life curve, so that SWT and the uncorrected curve
coincide for fully reversed loading. Cyclic mean-stress relaxation is not
followed (a conservative assumption for a tensile mean).

Registry
--------
:data:`STRAIN_LIFE` holds the constants with their source; every entry
carries a ``basis``: ``"published"`` (constants of a cited test series),
``"derived"`` (a cited estimation method applied to cited monotonic
properties) or ``"assumed"`` (method or inputs assumed; a bound, not a
material property). :func:`register_strain_life` adds user constants.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cablexsec.exceptions import CableInputError, CableLookupError

__all__ = [
    "MEAN_STRESS_CORRECTIONS",
    "STRAIN_LIFE",
    "MeanStressCorrection",
    "StrainLifeConstants",
    "StrainLifeDamage",
    "StrainLifeMap",
    "get_strain_life",
    "list_strain_life",
    "masing_stresses",
    "register_strain_life",
    "strain_life_damage",
    "strain_life_damage_map",
]

FloatArray = NDArray[np.float64]

MeanStressCorrection = Literal["swt", "morrow", "none"]
MEAN_STRESS_CORRECTIONS: tuple[str, ...] = ("swt", "morrow", "none")
StrainLifeBasis = Literal["published", "derived", "assumed"]

#: Upper limit of the life search, reversals ``2N``: beyond it the cycle is
#: treated as causing no damage (the curves have no endurance limit).
MAX_REVERSALS = 1.0e20


@dataclass(frozen=True)
class StrainLifeConstants:
    """Coffin--Manson--Basquin constants and the cyclic stress--strain curve.

    Attributes
    ----------
    name : str
    E_Pa : float
        Modulus of the elastic line [Pa].
    sigma_f_prime_Pa, b : float
        Fatigue strength coefficient [Pa] and exponent (Basquin).
    eps_f_prime, c : float
        Fatigue ductility coefficient and exponent (Coffin--Manson).
    K_prime_Pa, n_prime : float or None
        Cyclic strength coefficient [Pa] and hardening exponent; ``None``
        selects the compatibility estimate (module notes).
    reference : str
        Source of the constants (``[AuthorYear] location``).
    basis : {"published", "derived", "assumed"}
    applicability : str
        Material, condition and test range of the constants.
    check_points : tuple of (float, float)
        Published ``(strain amplitude, cycles to failure)`` pairs that the
        constants reproduce, used by the registry gate.
    """

    name: str
    E_Pa: float
    sigma_f_prime_Pa: float
    b: float
    eps_f_prime: float
    c: float
    reference: str
    basis: StrainLifeBasis
    applicability: str = ""
    K_prime_Pa: float | None = None
    n_prime: float | None = None
    check_points: tuple[tuple[float, float], ...] = field(default=())

    def __post_init__(self) -> None:
        positive = (self.E_Pa, self.sigma_f_prime_Pa, self.eps_f_prime)
        if not all(math.isfinite(v) and v > 0 for v in positive):
            raise CableInputError(
                "E, sigma_f' and epsilon_f' must be finite and positive"
            )
        if not (-1.0 < self.b < 0.0 and -2.0 < self.c < 0.0):
            raise CableInputError(
                "strain-life exponents need -1 < b < 0 and -2 < c < 0"
            )
        if (self.K_prime_Pa is None) != (self.n_prime is None):
            raise CableInputError("give both K' and n' or neither")
        if self.K_prime_Pa is not None and self.n_prime is not None:
            if not (math.isfinite(self.K_prime_Pa) and self.K_prime_Pa > 0):
                raise CableInputError("K' must be finite and positive")
            if not 0.0 < self.n_prime < 1.0:
                raise CableInputError("n' must lie in (0, 1)")
        if not self.reference.strip():
            raise CableInputError("strain-life constants need a reference")
        if self.basis not in ("published", "derived", "assumed"):
            raise CableInputError("basis must be 'published', 'derived' or 'assumed'")

    @classmethod
    def from_modified_universal_slopes(
        cls,
        name: str,
        *,
        E_Pa: float,
        sigma_UTS_Pa: float,
        ductility: float,
        reference: str,
        basis: StrainLifeBasis = "derived",
        applicability: str = "",
    ) -> StrainLifeConstants:
        r"""Constants of the modified universal slopes ([MuralidharanManson1988]).

        ``Delta epsilon = 0.0266 D^0.155 (S_u/E)^-0.53 N^-0.56
        + 1.17 (S_u/E)^0.832 N^-0.09`` with ``D = ln(1/(1 - RA))`` the true
        fracture ductility, rewritten exactly in reversals ``2N``:
        ``b = -0.09``, ``c = -0.56``,
        ``sigma_f' = 0.585 * 2^0.09 E (S_u/E)^0.832``,
        ``epsilon_f' = 0.0133 * 2^0.56 D^0.155 (S_u/E)^-0.53``.
        """
        if not all(math.isfinite(v) and v > 0 for v in (E_Pa, sigma_UTS_Pa, ductility)):
            raise CableInputError(
                "E, S_u and the ductility must be finite and positive"
            )
        ratio = sigma_UTS_Pa / E_Pa
        return cls(
            name=name,
            E_Pa=E_Pa,
            sigma_f_prime_Pa=0.585 * 2.0**0.09 * E_Pa * ratio**0.832,
            b=-0.09,
            eps_f_prime=0.0133 * 2.0**0.56 * ductility**0.155 * ratio**-0.53,
            c=-0.56,
            reference=reference,
            basis=basis,
            applicability=applicability,
        )

    @property
    def cyclic_curve(self) -> tuple[float, float]:
        """``(K', n')`` of the cyclic stress--strain curve."""
        if self.K_prime_Pa is not None and self.n_prime is not None:
            return self.K_prime_Pa, self.n_prime
        n = self.b / self.c
        return self.sigma_f_prime_Pa / self.eps_f_prime**n, n

    @property
    def cyclic_curve_is_estimate(self) -> bool:
        return self.K_prime_Pa is None

    def strain_amplitude(self, reversals: ArrayLike) -> FloatArray:
        """``epsilon_a(2N)`` of the uncorrected curve."""
        two_n = np.asarray(reversals, dtype=np.float64)
        return np.asarray(
            self.sigma_f_prime_Pa / self.E_Pa * two_n**self.b
            + self.eps_f_prime * two_n**self.c
        )

    def stress_amplitude(self, strain_amplitude: ArrayLike) -> FloatArray:
        """Stress amplitude on the cyclic curve for a strain amplitude [Pa]."""
        return _cyclic_stress(np.abs(np.asarray(strain_amplitude, float)), self)

    def cycles_to_failure(
        self,
        strain_amplitude: ArrayLike,
        *,
        sigma_mean_Pa: ArrayLike = 0.0,
        sigma_max_Pa: ArrayLike | None = None,
        correction: MeanStressCorrection = "none",
    ) -> FloatArray:
        """Cycles ``N_f`` for strain amplitudes (``inf``: no damage)."""
        eps_a = np.abs(np.asarray(strain_amplitude, dtype=np.float64))
        if correction not in MEAN_STRESS_CORRECTIONS:
            raise CableInputError(
                f"mean-stress correction must be one of {MEAN_STRESS_CORRECTIONS}"
            )
        e, sf, b, ef, c = (
            self.E_Pa,
            self.sigma_f_prime_Pa,
            self.b,
            self.eps_f_prime,
            self.c,
        )
        if correction == "swt":
            smax = (
                self.stress_amplitude(eps_a)
                if sigma_max_Pa is None
                else np.broadcast_to(np.asarray(sigma_max_Pa, float), eps_a.shape)
            )
            target = np.where(smax > 0.0, smax * eps_a, 0.0)

            def rhs(x: FloatArray) -> FloatArray:
                return np.asarray(sf * sf / e * x ** (2 * b) + sf * ef * x ** (b + c))

            return _solve_reversals(target, rhs) / 2.0
        sm = (
            np.zeros_like(eps_a)
            if correction == "none"
            else np.broadcast_to(np.asarray(sigma_mean_Pa, float), eps_a.shape)
        )
        elastic = np.maximum(sf - sm, 0.0) / e

        def rhs_m(x: FloatArray) -> FloatArray:
            return np.asarray(elastic * x**b + ef * x**c)

        return _solve_reversals(eps_a, rhs_m) / 2.0


def _solve_reversals(
    target: FloatArray, rhs: Callable[[FloatArray], FloatArray]
) -> FloatArray:
    """Reversals ``2N`` with ``rhs(2N) = target`` (decreasing ``rhs``)."""
    t = np.asarray(target, dtype=np.float64)
    out = np.full(t.shape, math.inf)
    active = t > 0.0
    if not np.any(active):
        return out
    f = rhs
    tt = t[active]
    lo = np.zeros(tt.shape)  # log10 2N
    hi = np.full(tt.shape, math.log10(MAX_REVERSALS))
    first = np.asarray(f(np.ones(tt.shape))) <= tt
    last = np.asarray(f(np.full(tt.shape, MAX_REVERSALS))) > tt
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        above = np.asarray(f(10.0**mid)) > tt
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    result = 10.0 ** (0.5 * (lo + hi))
    result = np.where(first, 1.0, result)
    result = np.where(last, math.inf, result)
    out[active] = result
    return out


def _cyclic_stress(strain: FloatArray, constants: StrainLifeConstants) -> FloatArray:
    """Inverse of ``epsilon = sigma/E + (sigma/K')^(1/n')`` for ``epsilon >= 0``."""
    k, n = constants.cyclic_curve
    e = constants.E_Pa
    eps = np.asarray(strain, dtype=np.float64)
    lo = np.zeros(eps.shape)
    hi = np.minimum(e * eps, k * eps**n) + 0.0
    hi = np.maximum(hi, 0.0)
    # sigma <= min(E eps, K' eps^n') brackets the root; bisect on sigma.
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        strain_mid = mid / e + (mid / k) ** (1.0 / n)
        over = strain_mid > eps
        hi = np.where(over, mid, hi)
        lo = np.where(over, lo, mid)
    return np.asarray(0.5 * (lo + hi))


def _cyclic_stress_scalar(strain: float, e: float, k: float, n: float) -> float:
    """Scalar inverse of the cyclic curve (safeguarded Newton, pure Python)."""
    if strain <= 0.0:
        return 0.0
    lo = 0.0
    hi = float(min(e * strain, k * strain**n))
    sigma = hi
    inv_n = 1.0 / n
    for _ in range(100):
        residual = float(sigma / e + (sigma / k) ** inv_n - strain)
        if abs(residual) <= 1e-14 * strain:
            return sigma
        if residual > 0.0:
            hi = sigma
        else:
            lo = sigma
        if hi - lo <= 1e-13 * hi:
            return 0.5 * (lo + hi)
        slope = float(
            1.0 / e + inv_n / k * (sigma / k) ** (inv_n - 1.0) if sigma > 0 else 1.0 / e
        )
        step = sigma - residual / slope
        sigma = step if lo < step < hi else 0.5 * (lo + hi)
    return sigma


def masing_stresses(
    strain_points: ArrayLike, constants: StrainLifeConstants
) -> FloatArray:
    """Stresses at a sequence of strain reversal points [Pa].

    The path starts from the virgin state at zero strain: it follows the
    cyclic curve to the first point and Masing branches with closed-loop
    memory after every reversal ([Dowling2013] Sec. 14.4).
    """
    points = np.asarray(strain_points, dtype=np.float64)
    if points.ndim != 1 or not np.isfinite(points).all():
        raise CableInputError("strain points must be a finite 1-D sequence")
    out = np.empty(points.size)
    k_c, n_c = constants.cyclic_curve
    e_c = constants.E_Pa
    # Stack of branch origins; None marks the virgin (skeleton) origin.
    stack: list[tuple[float, float] | None] = [None]
    current = (0.0, 0.0)
    last_direction = 0.0
    for index, target in enumerate(points.tolist()):
        if target == current[0]:
            out[index] = current[1]
            continue
        move = 1.0 if target > current[0] else -1.0
        if last_direction != 0.0 and move != last_direction:
            stack.append(current)  # a reversal: new branch origin
        last_direction = move
        while True:
            origin = stack[-1]
            if origin is None:
                stress = math.copysign(
                    _cyclic_stress_scalar(abs(target), e_c, k_c, n_c), target
                )
                break
            previous = stack[-2]
            limit = -origin[0] if previous is None else previous[0]
            direction = 1.0 if target > origin[0] else -1.0
            if direction * (target - limit) > 0.0:
                stack.pop()
                if previous is not None:
                    stack.pop()
                continue
            half = _cyclic_stress_scalar(abs(target - origin[0]) / 2.0, e_c, k_c, n_c)
            stress = origin[1] + direction * 2.0 * half
            break
        current = (target, stress)
        out[index] = stress
    return out


def _turning_indices(series: FloatArray) -> NDArray[np.int64]:
    """Indices of the first sample, the turning points and the last sample.

    Same reduction as :func:`cablexsec.analysis.fatigue.rainflow_cycles`.
    """
    s = series
    if s.size < 3:
        return np.arange(s.size, dtype=np.int64)
    changes = np.flatnonzero(s[1:] != s[:-1])
    if changes.size == 0:
        return np.array([0], dtype=np.int64)
    rising = s[changes + 1] > s[changes]
    turning = changes[1:][rising[1:] != rising[:-1]]
    return np.concatenate(([0], turning, [s.size - 1])).astype(np.int64)


@dataclass(frozen=True)
class StrainLifeDamage:
    """Miner damage of one strain history.

    ``cycles`` rows are ``(strain range, strain mean, count, sigma_max [Pa],
    sigma_mean [Pa], N_f)``.
    """

    damage: float
    n_cycles: float
    strain_amplitude_max: float
    stress_max_Pa: float
    cycles: FloatArray
    correction: str
    constants: str


def _cycles(series: FloatArray) -> Iterator[tuple[float, float, float, int, int]]:
    from cablexsec.analysis.fatigue import rainflow_cycles  # noqa: PLC0415

    yield from rainflow_cycles(series)


def _cycle_rows(
    eps: FloatArray, const: StrainLifeConstants
) -> tuple[list[tuple[float, float, float, float, float]], float]:
    """Rainflow rows ``(range, mean, count, sigma_max, sigma_mean)`` and max |sigma|."""
    indices = _turning_indices(eps)
    stresses = masing_stresses(eps[indices], const)
    by_index = dict(zip(indices.tolist(), stresses.tolist(), strict=True))
    rows = []
    for span, mean, count, i0, i1 in _cycles(eps):
        s0, s1 = by_index[i0], by_index[i1]
        rows.append((span, mean, count, max(s0, s1), 0.5 * (s0 + s1)))
    peak = float(np.max(np.abs(stresses))) if stresses.size else 0.0
    return rows, peak


@dataclass(frozen=True)
class StrainLifeMap:
    """Damage of many strain histories (one per point), shape ``(points,)``."""

    damage: FloatArray
    n_cycles: FloatArray
    strain_amplitude_max: FloatArray
    stress_max_Pa: FloatArray


def strain_life_damage_map(
    strain_histories: ArrayLike,
    constants: StrainLifeConstants | str,
    *,
    correction: MeanStressCorrection = "swt",
) -> StrainLifeMap:
    """:func:`strain_life_damage` of every row of ``(points, time)`` at once."""
    const = get_strain_life(constants) if isinstance(constants, str) else constants
    if correction not in MEAN_STRESS_CORRECTIONS:
        raise CableInputError(
            f"mean-stress correction must be one of {MEAN_STRESS_CORRECTIONS}"
        )
    eps = np.asarray(strain_histories, dtype=np.float64)
    if eps.ndim != 2 or not np.isfinite(eps).all():
        raise CableInputError("strain histories must be a finite (points, time) array")
    points = eps.shape[0]
    peak = np.zeros(points)
    owner: list[int] = []
    rows: list[tuple[float, float, float, float, float]] = []
    if eps.shape[1] >= 2:
        for i in range(points):
            found, peak[i] = _cycle_rows(eps[i], const)
            rows.extend(found)
            owner.extend([i] * len(found))
    damage = np.zeros(points)
    cycles = np.zeros(points)
    amplitude = np.zeros(points)
    if rows:
        table = np.asarray(rows, dtype=np.float64)
        index = np.asarray(owner, dtype=np.int64)
        life = const.cycles_to_failure(
            table[:, 0] / 2.0,
            sigma_mean_Pa=table[:, 4],
            sigma_max_Pa=table[:, 3],
            correction=correction,
        )
        damage = np.asarray(
            np.bincount(index, weights=table[:, 2] / life, minlength=points),
            dtype=np.float64,
        )
        cycles = np.asarray(
            np.bincount(index, weights=table[:, 2], minlength=points),
            dtype=np.float64,
        )
        np.maximum.at(amplitude, index, table[:, 0] / 2.0)
    return StrainLifeMap(damage, cycles, amplitude, peak)


def strain_life_damage(
    strain_history: ArrayLike,
    constants: StrainLifeConstants | str,
    *,
    correction: MeanStressCorrection = "swt",
) -> StrainLifeDamage:
    """Rainflow strain cycles, Masing stresses and Miner damage.

    Cycles are counted on the strain history ([ASTM E1049]); each cycle's
    stresses are the Masing stresses at its two reversal points, giving
    ``sigma_max`` (SWT) and ``sigma_m`` (Morrow).
    """
    const = get_strain_life(constants) if isinstance(constants, str) else constants
    if correction not in MEAN_STRESS_CORRECTIONS:
        raise CableInputError(
            f"mean-stress correction must be one of {MEAN_STRESS_CORRECTIONS}"
        )
    eps = np.asarray(strain_history, dtype=np.float64)
    if eps.ndim != 1 or not np.isfinite(eps).all():
        raise CableInputError("strain history must be one-dimensional and finite")
    if eps.size < 2:
        return StrainLifeDamage(
            0.0, 0.0, 0.0, 0.0, np.zeros((0, 6)), correction, const.name
        )
    rows, peak = _cycle_rows(eps, const)
    if not rows:
        return StrainLifeDamage(
            0.0, 0.0, 0.0, peak, np.zeros((0, 6)), correction, const.name
        )
    table = np.asarray(rows, dtype=np.float64)
    life = const.cycles_to_failure(
        table[:, 0] / 2.0,
        sigma_mean_Pa=table[:, 4],
        sigma_max_Pa=table[:, 3],
        correction=correction,
    )
    damage = float(np.sum(table[:, 2] / life))
    return StrainLifeDamage(
        damage=damage,
        n_cycles=float(np.sum(table[:, 2])),
        strain_amplitude_max=float(np.max(table[:, 0]) / 2.0),
        stress_max_Pa=peak,
        cycles=np.column_stack((table, life)),
        correction=correction,
        constants=const.name,
    )


# --------------------------------------------------------------------------- #
# Registry. Values are data with their source in ``reference``.                #
# --------------------------------------------------------------------------- #
STRAIN_LIFE: dict[str, StrainLifeConstants] = {}

_BUILT_IN: set[str] = set()


def _builtin(constants: StrainLifeConstants) -> None:
    STRAIN_LIFE[constants.name] = constants
    _BUILT_IN.add(constants.name)


def list_strain_life() -> list[str]:
    """Names of the registered strain-life constants, sorted."""
    return sorted(STRAIN_LIFE)


def get_strain_life(name: str) -> StrainLifeConstants:
    """Return the registered :class:`StrainLifeConstants` ``name``."""
    try:
        return STRAIN_LIFE[name]
    except KeyError:
        raise CableLookupError(
            f"Unknown strain-life constants {name!r}. "
            f"Registered: {', '.join(list_strain_life())}."
        ) from None


def register_strain_life(
    constants: StrainLifeConstants, *, replace_existing: bool = False
) -> None:
    """Register user strain-life constants (their reference is required).

    A built-in entry cannot be replaced; a user entry only with
    ``replace_existing=True``.
    """
    if not isinstance(constants, StrainLifeConstants):
        raise CableInputError("register a StrainLifeConstants instance")
    if constants.name in _BUILT_IN:
        raise CableInputError(f"{constants.name!r} is a built-in entry")
    if constants.name in STRAIN_LIFE and not replace_existing:
        raise CableInputError(
            f"{constants.name!r} is already registered; pass replace_existing=True"
        )
    STRAIN_LIFE[constants.name] = replace(constants)


def _register_built_ins() -> None:
    from cablexsec.materials.library import get_property  # noqa: PLC0415
    from cablexsec.materials.sn_curves import SWT_LIFE  # noqa: PLC0415

    copper = SWT_LIFE["copper_poon2022"]
    _builtin(
        StrainLifeConstants(
            name="copper_poon2022",
            E_Pa=copper.E_Pa,
            sigma_f_prime_Pa=copper.sigma_f_prime_Pa,
            b=copper.b,
            eps_f_prime=copper.eps_f_prime,
            c=copper.c,
            reference=(
                "[Poon2022] Table 4 and Eq. 30 (fit to the Karlsen 2010 SPC "
                "Class-2 conductor wire data, Fig. 11), E from Table 1; the "
                "same set in [Poon2023] Table 2 and [Poon2024] Table 2"
            ),
            basis="published",
            applicability=(
                "Copper conductor wire (Class 2). Applied to a copper tape, foil "
                "or screen wire it is a transfer assumption (no copper-foil "
                "strain-life data in the reference library). K', n' by the "
                "compatibility estimate."
            ),
        )
    )
    # Gap P7. Annealed copper screen or conductor wire from cited monotonic
    # properties by the modified universal slopes: a second, independent
    # copper estimate beside the Poon (Karlsen-data) fit. No cyclic
    # stress-strain constants (K', n') for copper are in the reference
    # library (search of 18 Oct 2026), so both sets use the compatibility
    # estimate of the Masing curve.
    _builtin(
        StrainLifeConstants.from_modified_universal_slopes(
            "copper_annealed_mus_derived",
            E_Pa=115.0e9,
            sigma_UTS_Pa=235.0e6,
            ductility=math.log(1.0 / (1.0 - 0.725)),
            reference=(
                "[MuralidharanManson1988] modified universal slopes (secondary "
                "form in [Cruzado2013] Table 4) with E 115 GPa ([Karlsen2010] "
                "p. 2), S_u 235 MPa ([Wan2022] Table 2, annealed ETP Cu wire, "
                "1e-4 1/s) and reduction of area Z 0.725 ([Jiang2023] Sec. 4.2, "
                "TU2 Cu wire), D = ln(1/(1 - Z))"
            ),
            basis="derived",
            applicability=(
                "Annealed copper wire (screen or conductor), room temperature, "
                "smooth. An estimation method on cited monotonic properties of "
                "two different coppers (ETP, TU2), not a fit to copper fatigue "
                "data: use beside copper_poon2022 as a check. K', n' by the "
                "compatibility estimate."
            ),
        )
    )
    # Modified universal slopes with ASSUMED monotonic inputs for an O-temper
    # laminate foil (no aluminium-foil or laminate strain-life constants in
    # the reference library; EN 1999-1-3 Sec. 3(1) excludes the foil alloys
    # EN AW-8011A, -3005, -3103, -5005 for lack of reliable fatigue data).
    _builtin(
        StrainLifeConstants.from_modified_universal_slopes(
            "aluminium_foil_mus_assumed",
            E_Pa=get_property("aluminium_laminate_foil", "E_Pa"),
            sigma_UTS_Pa=get_property("aluminium_laminate_foil", "sigma_UTS_Pa"),
            ductility=0.6,  # ASSUMED true fracture ductility, bound 0.3-1.0
            reference=(
                "[MuralidharanManson1988] modified universal slopes (not in the "
                "reference library; secondary form in [Cruzado2013] Table 4) "
                "with ASSUMED inputs E 70 GPa, S_u 90 MPa (bound 70-110), "
                "D 0.6 (bound 0.3-1.0)"
            ),
            basis="assumed",
            applicability=(
                "Smooth aluminium foil of a bonded Al-PE laminate sheath, O "
                "temper, room temperature. A bound estimate, not a material "
                "property: run the bounds with "
                "StrainLifeConstants.from_modified_universal_slopes."
            ),
        )
    )


_register_built_ins()
