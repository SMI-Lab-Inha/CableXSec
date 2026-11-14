# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Per-wire critical (slip-onset) curvature.

The cable curvature at which wire ``k`` begins to slip is

    kappa_c(phi_k) = mu * F_n(phi_k) / (E_w * A_w * cos^2(alpha) * sin(alpha))

([Saevik2011] — the slip-onset/critical curvature; the survey locates this as
Eq. 55 of the paper, with the Coulomb friction capacity Eq. 54). The normal
force ``F_n(phi_k)`` comes strictly from the axisymmetric chain plus the
three-core correction. This function performs no inverse calibration.
The supplied normal force may originate from a measured state, a manufacturing
specification, or an explicitly labelled published calibration.

The distribution of ``kappa_c`` across the wire population is the physical
origin of the smooth M-κ knee: wires over filler voids carry lower ``F_n`` and
slip first; wires over cores carry higher ``F_n`` and slip later. A wide
distribution gives a gradual transition without an empirical knee function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cablexsec.exceptions import CableInputError

__all__ = ["KappaCDistribution", "kappa_c", "distribution_from_Fn"]

FloatArray = NDArray[np.float64]


def kappa_c(
    F_n_N: float | FloatArray,
    E_wire_Pa: float,
    A_wire_m2: float,
    lay_angle_rad: float,
    mu: float,
) -> FloatArray:
    """Critical (slip-onset) curvature [1/m] from the normal force.

    ``kappa_c = mu F_n / (E_w A_w cos^2(alpha) sin(alpha))`` ([Saevik2011]).

    Parameters
    ----------
    F_n_N : float or numpy.ndarray
        Per-wire normal contact force [N] (scalar or per-azimuth array).
    E_wire_Pa : float
        Wire Young's modulus [Pa].
    A_wire_m2 : float
        Wire cross-sectional area [m^2].
    lay_angle_rad : float
        Lay angle ``alpha`` [rad] (must be > 0).
    mu : float
        Coulomb friction coefficient.

    Returns
    -------
    numpy.ndarray
        Critical curvature at each ``F_n`` [1/m].
    """
    a = lay_angle_rad
    if a <= 0.0:
        raise CableInputError("lay_angle_rad must be > 0 (straight wires never slip)")
    denom = E_wire_Pa * A_wire_m2 * math.cos(a) ** 2 * math.sin(a)
    return np.asarray(mu * np.asarray(F_n_N, dtype=np.float64) / denom)


@dataclass(frozen=True)
class KappaCDistribution:
    """Per-wire critical-curvature distribution for one helical layer."""

    phi_rad: FloatArray
    kappa_c_per_m: FloatArray

    @property
    def min(self) -> float:
        return float(np.min(self.kappa_c_per_m))

    @property
    def mean(self) -> float:
        return float(np.mean(self.kappa_c_per_m))

    @property
    def max(self) -> float:
        return float(np.max(self.kappa_c_per_m))

    @property
    def spread_ratio(self) -> float:
        """``max/min`` — width of the distribution (1.0 = uniform contact)."""
        lo = self.min
        return self.max / lo if lo > 0 else math.inf

    def summary(self) -> str:
        """One-line ``min / mean / max`` summary in 1/m."""
        return (
            f"kappa_c [1/m]: min={self.min:.4g} mean={self.mean:.4g} "
            f"max={self.max:.4g} (max/min={self.spread_ratio:.3g})"
        )


def distribution_from_Fn(
    F_n_phi_N: FloatArray,
    E_wire_Pa: float,
    A_wire_m2: float,
    lay_angle_rad: float,
    mu: float,
    phi_rad: FloatArray | None = None,
) -> KappaCDistribution:
    """Build the per-wire ``kappa_c`` distribution from ``F_n(phi)``.

    Parameters
    ----------
    F_n_phi_N : numpy.ndarray
        Per-wire normal force at each azimuth [N].
    E_wire_Pa, A_wire_m2, lay_angle_rad, mu :
        As in :func:`kappa_c`.
    phi_rad : numpy.ndarray or None, optional
        Wire azimuths; defaults to a uniform ``2 pi k / n`` grid.

    Returns
    -------
    KappaCDistribution
    """
    fn = np.asarray(F_n_phi_N, dtype=np.float64)
    if phi_rad is None:
        phi_rad = 2.0 * np.pi * np.arange(fn.size) / fn.size
    kc = kappa_c(fn, E_wire_Pa, A_wire_m2, lay_angle_rad, mu)
    return KappaCDistribution(
        phi_rad=np.asarray(phi_rad, dtype=np.float64), kappa_c_per_m=kc
    )
