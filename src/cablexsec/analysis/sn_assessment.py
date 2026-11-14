# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Component stress concentration factors and corrected Miner sums.

Two user options of the stress-range fatigue routes live here.

Component SCFs (:class:`ComponentSCF`). The fixed-fibre wire stress is the
sum of an axial part (tension or solved normal stress, thermal stress), the
friction stress of the slip law (with any termination increment) and the
local bending at the outer face (normal curvature) and at the edge (lateral
curvature). Each part may take its own factor, separately for its mean over
the record and for its fluctuation about that mean:

.. math::

    \sigma(t) = \sum_c \bigl[k_{c,m}\,\bar\sigma_c
        + k_{c,d}\,(\sigma_c(t) - \bar\sigma_c)\bigr].

Separate factors act on the mean and dynamic parts of the axial,
normal-curvature and transverse-curvature stress, with the friction stress of
the sectional slip law as a fourth component. The mean is the time average
of the record. The conductor route has an axial part and the stick-slip
(friction) strain of its operator; the sheath route has the normal-bending
part only; the tape route has one membrane part, which takes the axial
factors.

Corrected Miner sum (:func:`corrected_damage_rows`). Rainflow cycles
([ASTM E1049]) of each history are scaled by a stress factor (termination
SCF), converted by the mean-stress correction of
:mod:`cablexsec.materials.mean_stress` to the curve's test condition,
multiplied by the curve's thickness factor ([DNV-RP-C203] Eq. 2.10) and
summed against the curve ([Miner1945]). Without a correction, a unit
factor and no thickness exponent, the uncorrected routes are used instead,
so that their results are unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import NDArray

from cablexsec.exceptions import CableInputError
from cablexsec.materials.mean_stress import MeanStressSpec, equivalent_range
from cablexsec.materials.sn_curves import SNcurve

__all__ = [
    "ComponentSCF",
    "corrected_damage",
    "corrected_damage_rows",
    "needs_correction",
    "scaled_component",
]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ComponentSCF:
    """Stress concentration factors per stress component (mean and dynamic).

    Every factor is finite and positive (default 1). ``source`` names the
    origin of the factors (a test, an FE study or an assumption) and is
    required when any factor differs from one.
    """

    axial_mean: float = 1.0
    axial_dynamic: float = 1.0
    bending_normal_mean: float = 1.0
    bending_normal_dynamic: float = 1.0
    bending_lateral_mean: float = 1.0
    bending_lateral_dynamic: float = 1.0
    friction_mean: float = 1.0
    friction_dynamic: float = 1.0
    source: str = ""

    def __post_init__(self) -> None:
        for item in fields(self):
            if item.name == "source":
                continue
            value = getattr(self, item.name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise CableInputError(f"{item.name} must be finite and positive")
            object.__setattr__(self, item.name, float(value))
        if not self.is_identity and not self.source.strip():
            raise CableInputError("component SCFs need their source in 'source'")

    @classmethod
    def uniform(
        cls,
        *,
        axial: float = 1.0,
        bending: float = 1.0,
        friction: float = 1.0,
        source: str = "",
    ) -> ComponentSCF:
        """Same factor for mean and dynamic parts (and both bending axes)."""
        return cls(
            axial_mean=axial,
            axial_dynamic=axial,
            bending_normal_mean=bending,
            bending_normal_dynamic=bending,
            bending_lateral_mean=bending,
            bending_lateral_dynamic=bending,
            friction_mean=friction,
            friction_dynamic=friction,
            source=source,
        )

    @property
    def is_identity(self) -> bool:
        """``True`` when every factor is one."""
        return all(
            getattr(self, item.name) == 1.0
            for item in fields(self)
            if item.name != "source"
        )

    def describe(self) -> str:
        """Compact record of the non-unit factors and their source."""
        parts = [
            f"{item.name} {getattr(self, item.name):g}"
            for item in fields(self)
            if item.name != "source" and getattr(self, item.name) != 1.0
        ]
        if not parts:
            return "no component SCF"
        return "component SCF " + ", ".join(parts) + f" ({self.source})"


def scaled_component(
    history: FloatArray, mean_factor: float, dynamic_factor: float
) -> FloatArray:
    """``k_m mean + k_d (x - mean)`` along the last (time) axis."""
    values = np.asarray(history, dtype=np.float64)
    if mean_factor == 1.0 and dynamic_factor == 1.0:
        return values
    mean = values.mean(axis=-1, keepdims=True)
    return np.asarray(mean_factor * mean + dynamic_factor * (values - mean))


def _cycles(
    history: FloatArray, compiled: bool
) -> tuple[FloatArray, FloatArray, FloatArray]:
    if compiled:
        from cablexsec.analysis.fast_rainflow import rainflow_table  # noqa: PLC0415

        rng, mean, count, _s, _e = rainflow_table(history)
        return rng, mean, count
    from cablexsec.analysis.fatigue import rainflow_cycles  # noqa: PLC0415

    rows = list(rainflow_cycles(history))
    if not rows:
        empty = np.zeros(0)
        return empty, empty, empty
    table = np.asarray([row[:3] for row in rows], dtype=np.float64)
    return table[:, 0], table[:, 1], table[:, 2]


def corrected_damage(
    history_Pa: FloatArray,
    curve: SNcurve,
    *,
    mean_stress: MeanStressSpec | None = None,
    stress_factor: float = 1.0,
    thickness_m: float | None = None,
    compiled: bool = False,
) -> float:
    """Miner damage of one stress history with the corrections of the module.

    ``stress_factor`` scales the cycle ranges and means (e.g. a termination
    SCF); the mean-stress correction then maps each cycle to the curve's test
    condition and the thickness factor multiplies the range.
    """
    if curve.stress_unit != "MPa":
        raise CableInputError("Miner stress damage requires an S-N curve in MPa")
    if not math.isfinite(stress_factor) or stress_factor <= 0:
        raise CableInputError("stress_factor must be finite and positive")
    s = np.asarray(history_Pa, dtype=np.float64)
    if s.ndim != 1 or not np.isfinite(s).all():
        raise CableInputError("stress history must be one-dimensional and finite")
    rng, mean, count = _cycles(s, compiled)
    if rng.size == 0:
        return 0.0
    rng = rng * stress_factor
    mean = mean * stress_factor
    spec = mean_stress if mean_stress is not None else MeanStressSpec()
    equivalent = equivalent_range(rng, mean, spec, curve_R=curve.R_ratio)
    stress_MPa = equivalent / 1.0e6 * curve.thickness_factor(thickness_m)
    log_life = curve.log10_life(stress_MPa)
    live = np.isfinite(log_life)
    if np.any(log_life[live] < -300.0) or np.any(log_life[live] > 300.0):
        raise CableInputError("S-N life exceeds floating-point range")
    damage = float(np.sum(count[live] * 10.0 ** (-log_life[live])))
    if not math.isfinite(damage):
        raise CableInputError("accumulated damage exceeds floating-point range")
    return damage


def corrected_damage_rows(
    histories_Pa: FloatArray,
    curve: SNcurve,
    *,
    mean_stress: MeanStressSpec | None = None,
    stress_factor: float = 1.0,
    thickness_m: float | None = None,
    compiled: bool = False,
) -> FloatArray:
    """:func:`corrected_damage` of every row of a (row, time) array."""
    rows = np.asarray(histories_Pa, dtype=np.float64)
    if rows.ndim != 2:
        raise CableInputError("stress histories must be two-dimensional")
    return np.array(
        [
            corrected_damage(
                row,
                curve,
                mean_stress=mean_stress,
                stress_factor=stress_factor,
                thickness_m=thickness_m,
                compiled=compiled,
            )
            for row in rows
        ],
        dtype=np.float64,
    )


def needs_correction(
    curve: SNcurve,
    mean_stress: MeanStressSpec | None,
    scf: ComponentSCF | None,
) -> bool:
    """Whether the corrected route is required (otherwise results are unchanged)."""
    return (
        (mean_stress is not None and mean_stress.active)
        or (scf is not None and not scf.is_identity)
        or curve.thickness_exponent != 0.0
    )
