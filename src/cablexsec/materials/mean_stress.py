# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Mean-stress corrections for stress-range S-N curves.

A rainflow cycle has a range :math:`\Delta\sigma = 2\sigma_a` and a mean
:math:`\sigma_m`. A correction converts it to the range of an equivalent
cycle at the test condition of the S-N curve, so that curves measured at a
stress ratio other than :math:`R = -1` (or at a constant mean stress) are
entered consistently (with the curve's ``R`` ratio, or for a curve given at
constant mean stress), with the textbook mean-stress equations
([Dowling2013] Ch. 9):

=============== ========================================================
Method          Equivalent fully reversed amplitude :math:`\sigma_{ar}`
=============== ========================================================
``"goodman"``   :math:`\sigma_a / (1 - \sigma_m/\sigma_u)`
``"gerber"``    :math:`\sigma_a / (1 - (\sigma_m/\sigma_u)^2)`
``"soderberg"`` :math:`\sigma_a / (1 - \sigma_m/\sigma_y)`
``"swt"``       :math:`\sqrt{\sigma_{max}\sigma_a}` ([SmithWatsonTopper1970])
``"walker"``    :math:`\sigma_{max}^{1-\gamma}\sigma_a^{\gamma}` ([Walker1970])
=============== ========================================================

The equivalent amplitude is then mapped back to the curve's test condition:
for a curve at ratio :math:`R` the amplitude :math:`\sigma_{a,c}` with mean
:math:`r\,\sigma_{a,c}`, :math:`r = (1 + R)/(1 - R)`, that has the same
:math:`\sigma_{ar}`; for a curve at constant mean stress
(``reference_mean_stress_Pa``) the amplitude at that mean. For ``R = -1`` the
range is :math:`2\sigma_{ar}`. SWT and Walker cycles with
:math:`\sigma_{max} \le 0` do no damage.

Two range-reduction factors of [DNV-RP-C203] Sec. 2.5 are also provided;
they multiply the range directly and do not use the curve's ratio:

* ``"dnv_base"``, Eq. (2.15), base material without significant residual
  stress: :math:`f_m = (\sigma_t + 0.6|\sigma_c|)/(\sigma_t + |\sigma_c|)`
  for :math:`\sigma_m > -\Delta\sigma/2`, else 0;
* ``"dnv_pwht"``, Eq. (2.16), post-weld heat treated connections:
  :math:`f_m = (\sigma_t + 0.8|\sigma_c|)/(\sigma_t + |\sigma_c|)`,

with :math:`\sigma_t` the maximum tension and :math:`\sigma_c` the maximum
compression of the cycle.

Compressive means. With ``compressive_mean="zero"`` (default) a compressive
mean is taken as zero in the Goodman, Gerber and Soderberg equations: no
credit is taken, and Gerber does not penalise it. ``"signed"`` uses the
signed mean. A tensile mean at or above the strength (Goodman, Gerber) or
the yield stress (Soderberg) is a static failure and raises an input error.

References
----------
[Dowling2013] Ch. 9; [SmithWatsonTopper1970]; [Walker1970]; [DNV-RP-C203]
Sec. 2.5, Eqs. (2.15)-(2.16).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cablexsec.exceptions import CableInputError

__all__ = [
    "MEAN_STRESS_METHODS",
    "MeanStressMethod",
    "MeanStressSpec",
    "equivalent_range",
]

FloatArray = NDArray[np.float64]
MeanStressMethod = Literal[
    "none", "goodman", "gerber", "soderberg", "swt", "walker", "dnv_base", "dnv_pwht"
]
MEAN_STRESS_METHODS: tuple[str, ...] = (
    "none",
    "goodman",
    "gerber",
    "soderberg",
    "swt",
    "walker",
    "dnv_base",
    "dnv_pwht",
)
_DNV_FACTOR = {"dnv_base": 0.6, "dnv_pwht": 0.8}


def _positive(value: float | None, name: str) -> float:
    if value is None or isinstance(value, bool):
        raise CableInputError(f"{name} is required for this mean-stress method")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise CableInputError(f"{name} must be finite and positive")
    return number


@dataclass(frozen=True)
class MeanStressSpec:
    """A mean-stress correction with its strength data and their source.

    Parameters
    ----------
    method : str
        One of :data:`MEAN_STRESS_METHODS` (module notes).
    uts_Pa, yield_Pa : float or None
        Ultimate tensile strength (Goodman, Gerber) or yield stress
        (Soderberg) [Pa]; required by those methods.
    strength_source : str
        Source of the strength value; required with ``uts_Pa``/``yield_Pa``.
    walker_gamma : float or None
        Walker exponent ``gamma`` in (0, 1]; ``gamma = 0.5`` is SWT and
        ``gamma = 1`` no correction. Required for ``"walker"``.
    walker_source : str
        Source of ``walker_gamma``; required for ``"walker"``.
    compressive_mean : {"zero", "signed"}
        Treatment of a compressive mean in Goodman, Gerber and Soderberg.
    reference_mean_stress_Pa : float or None
        Mean stress of a curve measured at constant mean stress; ``None``
        uses the curve's ``R_ratio``.
    """

    method: MeanStressMethod = "none"
    uts_Pa: float | None = None
    yield_Pa: float | None = None
    strength_source: str = ""
    walker_gamma: float | None = None
    walker_source: str = ""
    compressive_mean: Literal["zero", "signed"] = "zero"
    reference_mean_stress_Pa: float | None = None

    def __post_init__(self) -> None:
        if self.method not in MEAN_STRESS_METHODS:
            raise CableInputError(
                f"mean-stress method must be one of {MEAN_STRESS_METHODS}, "
                f"got {self.method!r}"
            )
        if self.compressive_mean not in ("zero", "signed"):
            raise CableInputError("compressive_mean must be 'zero' or 'signed'")
        if self.method in ("goodman", "gerber"):
            _positive(self.uts_Pa, "uts_Pa")
        if self.method == "soderberg":
            _positive(self.yield_Pa, "yield_Pa")
        for value, name in ((self.uts_Pa, "uts_Pa"), (self.yield_Pa, "yield_Pa")):
            if value is not None:
                _positive(value, name)
                if not self.strength_source.strip():
                    raise CableInputError(f"{name} needs its source in strength_source")
        if self.method == "walker":
            gamma = _positive(self.walker_gamma, "walker_gamma")
            if gamma > 1.0:
                raise CableInputError("walker_gamma must lie in (0, 1]")
            if not self.walker_source.strip():
                raise CableInputError("walker_gamma needs its source in walker_source")
        if self.reference_mean_stress_Pa is not None:
            if self.method.startswith("dnv"):
                raise CableInputError(
                    "the DNV-RP-C203 range factors do not use a reference mean"
                )
            if not math.isfinite(self.reference_mean_stress_Pa):
                raise CableInputError("reference_mean_stress_Pa must be finite")

    @classmethod
    def from_material(
        cls,
        method: MeanStressMethod,
        material: str,
        *,
        walker_gamma: float | None = None,
        walker_source: str = "",
        compressive_mean: Literal["zero", "signed"] = "zero",
        reference_mean_stress_Pa: float | None = None,
    ) -> MeanStressSpec:
        """Correction with the UTS or yield stress of a library material.

        The strength and its citation are read from
        :mod:`cablexsec.materials.library` (``sigma_UTS_Pa`` for Goodman and
        Gerber, ``sigma_y_Pa`` for Soderberg); other fields as keywords.
        """
        from cablexsec.materials import library  # noqa: PLC0415

        uts = yield_stress = None
        source = ""
        if method in ("goodman", "gerber"):
            uts = library.get_property(material, "sigma_UTS_Pa")
            source = f"{material} sigma_UTS_Pa: " + library.get_citation(
                material, "sigma_UTS_Pa"
            )
        elif method == "soderberg":
            yield_stress = library.get_property(material, "sigma_y_Pa")
            source = f"{material} sigma_y_Pa: " + library.get_citation(
                material, "sigma_y_Pa"
            )
        return cls(
            method=method,
            uts_Pa=uts,
            yield_Pa=yield_stress,
            strength_source=source,
            walker_gamma=walker_gamma,
            walker_source=walker_source,
            compressive_mean=compressive_mean,
            reference_mean_stress_Pa=reference_mean_stress_Pa,
        )

    @property
    def active(self) -> bool:
        """``False`` for ``"none"`` (ranges are used unchanged)."""
        return self.method != "none"

    def describe(self) -> str:
        """One-line record of the correction for provenance."""
        if self.method == "none":
            return "no mean-stress correction"
        if self.method.startswith("dnv"):
            eq = "2.15" if self.method == "dnv_base" else "2.16"
            return f"DNV-RP-C203 Eq. ({eq}) range reduction factor f_m"
        parts: list[str] = [self.method]
        if self.uts_Pa is not None:
            parts.append(f"UTS {self.uts_Pa / 1e6:.6g} MPa ({self.strength_source})")
        if self.yield_Pa is not None:
            parts.append(
                f"yield {self.yield_Pa / 1e6:.6g} MPa ({self.strength_source})"
            )
        if self.walker_gamma is not None:
            parts.append(f"gamma {self.walker_gamma:g} ({self.walker_source})")
        parts.append(f"compressive mean {self.compressive_mean}")
        if self.reference_mean_stress_Pa is not None:
            parts.append(f"curve at mean {self.reference_mean_stress_Pa / 1e6:.6g} MPa")
        return ", ".join(parts)


def _strength(spec: MeanStressSpec) -> float:
    value = spec.yield_Pa if spec.method == "soderberg" else spec.uts_Pa
    assert value is not None  # validated by MeanStressSpec
    return float(value)


def _amplitude_equivalent(
    amplitude: FloatArray, mean: FloatArray, spec: MeanStressSpec
) -> FloatArray:
    """Fully reversed equivalent amplitude ``sigma_ar`` [Pa]."""
    method = spec.method
    if method in ("swt", "walker"):
        peak = mean + amplitude
        out = np.zeros_like(amplitude)
        live = (peak > 0.0) & (amplitude > 0.0)
        if method == "swt":
            out[live] = np.sqrt(peak[live] * amplitude[live])
        else:
            gamma = float(spec.walker_gamma or 1.0)
            out[live] = peak[live] ** (1.0 - gamma) * amplitude[live] ** gamma
        return out
    m = np.maximum(mean, 0.0) if spec.compressive_mean == "zero" else mean
    strength = _strength(spec)
    ratio = m / strength
    denominator = 1.0 - ratio**2 if method == "gerber" else 1.0 - ratio
    if np.any(denominator <= 0.0):
        k = int(np.argmin(denominator))
        raise CableInputError(
            f"mean stress {mean[k] / 1e6:.6g} MPa reaches the "
            f"{'yield stress' if method == 'soderberg' else 'tensile strength'} "
            f"({strength / 1e6:.6g} MPa): static failure, not fatigue"
        )
    return np.asarray(amplitude / denominator, dtype=np.float64)


def _amplitude_at_reference(
    sigma_ar: FloatArray, spec: MeanStressSpec, curve_R: float
) -> FloatArray:
    """Amplitude at the curve's test condition with the same ``sigma_ar``."""
    method = spec.method
    zero_rule = spec.compressive_mean == "zero" and method in (
        "goodman",
        "gerber",
        "soderberg",
    )
    if spec.reference_mean_stress_Pa is not None:
        mean_c = float(spec.reference_mean_stress_Pa)
        if method in ("goodman", "gerber", "soderberg"):
            m = max(mean_c, 0.0) if zero_rule else mean_c
            strength = _strength(spec)
            factor = (
                1.0 - (m / strength) ** 2 if method == "gerber" else 1.0 - m / strength
            )
            if factor <= 0.0:
                raise CableInputError("the curve's mean stress reaches the strength")
            return sigma_ar * factor
        if method == "swt":
            return np.asarray(
                0.5 * (-mean_c + np.sqrt(mean_c**2 + 4.0 * sigma_ar**2)),
                dtype=np.float64,
            )
        return _walker_at_mean(sigma_ar, mean_c, float(spec.walker_gamma or 1.0))
    if curve_R == -1.0:
        return sigma_ar
    r = (1.0 + curve_R) / (1.0 - curve_R)
    if method in ("goodman", "soderberg"):
        if r <= 0.0 and zero_rule:
            return sigma_ar
        strength = _strength(spec)
        return np.asarray(sigma_ar / (1.0 + r * sigma_ar / strength), dtype=np.float64)
    if method == "gerber":
        if r == 0.0 or (r < 0.0 and zero_rule):
            return sigma_ar
        q = (r / _strength(spec)) ** 2
        out = np.zeros_like(sigma_ar)
        live = sigma_ar > 0.0
        s = sigma_ar[live]
        out[live] = (np.sqrt(1.0 + 4.0 * q * s * s) - 1.0) / (2.0 * q * s)
        return out
    if method == "swt":
        return np.asarray(sigma_ar / math.sqrt(1.0 + r), dtype=np.float64)
    gamma = float(spec.walker_gamma or 1.0)
    return np.asarray(sigma_ar / (1.0 + r) ** (1.0 - gamma), dtype=np.float64)


def _walker_at_mean(sigma_ar: FloatArray, mean: float, gamma: float) -> FloatArray:
    """Solve ``(mean + a)^(1-gamma) a^gamma = sigma_ar`` for ``a`` (bisection)."""
    out = np.zeros_like(sigma_ar)
    live = sigma_ar > 0.0
    if not np.any(live):
        return out
    target = sigma_ar[live]
    lo = np.full(target.shape, max(0.0, -mean))
    hi = lo + 2.0 * target + abs(mean) + 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        value = (mean + mid) ** (1.0 - gamma) * mid**gamma
        high = value >= target
        hi = np.where(high, mid, hi)
        lo = np.where(high, lo, mid)
    out[live] = hi
    return out


def equivalent_range(
    range_Pa: ArrayLike,
    mean_Pa: ArrayLike,
    spec: MeanStressSpec,
    *,
    curve_R: float = -1.0,
) -> FloatArray:
    """Equivalent stress ranges at the S-N curve's test condition [Pa].

    Parameters
    ----------
    range_Pa, mean_Pa : array_like
        Cycle ranges (nonnegative) and means [Pa].
    spec : MeanStressSpec
        The correction.
    curve_R : float
        Stress ratio of the curve's tests (``SNcurve.R_ratio``); ignored with
        ``spec.reference_mean_stress_Pa`` and by the DNV range factors.

    Returns
    -------
    numpy.ndarray
        Ranges to enter the curve with. ``"none"`` returns the ranges
        unchanged.
    """
    rng = np.asarray(range_Pa, dtype=np.float64)
    mean = np.asarray(mean_Pa, dtype=np.float64)
    if rng.shape != mean.shape:
        raise CableInputError("cycle ranges and means must have the same shape")
    if not (np.isfinite(rng).all() and np.isfinite(mean).all()) or np.any(rng < 0):
        raise CableInputError("cycle ranges must be finite and nonnegative")
    if not math.isfinite(curve_R) or curve_R >= 1.0:
        raise CableInputError("curve_R must be finite and below 1")
    if spec.method == "none":
        return rng.copy()
    amplitude = 0.5 * rng
    if spec.method in _DNV_FACTOR:
        c = _DNV_FACTOR[spec.method]
        tension = np.maximum(mean + amplitude, 0.0)
        compression = np.abs(np.minimum(mean - amplitude, 0.0))
        total = tension + compression
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(total > 0.0, (tension + c * compression) / total, 1.0)
        if spec.method == "dnv_base":
            factor = np.where(mean <= -amplitude, 0.0, factor)
        return np.asarray(factor * rng, dtype=np.float64)
    sigma_ar = _amplitude_equivalent(amplitude, mean, spec)
    return np.asarray(2.0 * _amplitude_at_reference(sigma_ar, spec, curve_R))
