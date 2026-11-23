# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Coupled spring--slider stations of laid cores (relative intralayer slip).

The relative slip of neighbouring cores, its onset and the coalescence of the
slip zones at ``pi / 3`` follow Skeie, Sødahl, Pettersen and Verde (2023,
OMAE2023-104817, doi:10.1115/OMAE2023-104817); the interlayer share, the
direction-tied cap and the closest-point return map of a group are this
module's.

Used by :func:`cablexsec.mechanics.papailiou.wire_slip_history` when the
laid-core law takes ``CoreFrictionSpec(core_core_coupling="relative")``
(:mod:`cablexsec.mechanics.core_friction`). Neighbouring cores of one layer
touch at the same axial position, and the friction between them acts on their
relative slip, so the stations of the core population that sit at one axial
position are not independent.

Groups
------
A population of ``K = n G`` stations at azimuths ``2 pi k / K`` stands for
``n`` cores; the ``n`` stations ``g, g + G, ..., g + (n - 1) G`` (azimuths
``2 pi / n`` apart) are the cores at one axial position, group ``g``.

Bounds
------
The friction stress of a station is the sum ``sigma = a + c`` of the friction
accumulated from the neutral axis over its interlayer contacts (filler,
bedding, sheath; the absolute slip of the core) and over its intralayer
contacts (the neighbouring cores; their relative slip):

* ``-L <= a <= U``, the direction-tied bounds of the interlayer share (the
  capstan form of :func:`~cablexsec.mechanics.papailiou.wire_slip_stress_bounds`
  with the interlayer coefficient);
* ``-C <= c <= C``, ``C = sigma_0 mu_cc |sin b| g min(psi, pi / n)`` (constant
  Coulomb bound of the core--core load, ``psi`` the angle of the station from
  the neutral axis): the planar full-slip profile of the coupled law, whose
  intralayer share grows at the full rate only over ``psi < pi / n``, where a
  core slips against both neighbours in the same sense, and stays flat beyond,
  where the two neighbours pull in opposite senses;
* ``c_g + c_{g+G} + ... = 0`` in every group: the friction between two
  neighbours is an action and a reaction, so the intralayer shares of the
  cores at one axial position cancel.

The admissible set of a group is the Minkowski sum ``A + (B ∩ H)`` (``A`` the
interlayer box, ``B`` the intralayer box, ``H`` the zero-sum plane). Without
intralayer friction it is the box ``A`` and the return map is the clip of the
uncoupled law.

Return map
----------
The stations have equal stick stiffness, so the return map of a group is the
Euclidean closest point of the trial stress ``t`` on the admissible set. With
a multiplier ``nu`` of the zero-sum constraint, every station has the closed
form ``c = clip(t - U - nu, -C, C)`` for ``nu > 0`` and
``c = clip(t + L - nu, -C, C)`` for ``nu < 0``, then ``a = clip(t - c, -L,
U)``; ``nu = 0`` (the box clip ``clip(t, -(L + C), U + C)``) holds when
``sum clip(t - U, -C, C) <= 0 <= sum clip(t + L, -C, C)`` (to a round-off
margin of 1e-13 of the group's stress magnitudes, so that a group in stick
stays in stick exactly). Otherwise ``nu``
is the root of the piecewise-linear, monotone ``sum c(nu) = 0``, found exactly
from its sorted breakpoints. The stress ``a + c`` is unique. The plastic work
of every update is non-negative (the set contains zero).

Two implementations solve the same update: a NumPy path vectorised over the
groups and a compiled kernel (numba); they agree to round-off
(tests/test_core_coupled_slip.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cablexsec._numba import HAVE_NUMBA, njit
from cablexsec.exceptions import CableInputError

__all__ = [
    "StationCoupling",
    "group_view",
    "project_station_groups",
]

FloatArray = NDArray[np.float64]

#: Relative size of a zero-sum residual taken as round-off: a group whose box
#: clip leaves its intralayer shares out of balance by less than this times
#: the sum of its stress magnitudes keeps the box clip (stick stays stick).
_ROUND_OFF = 1.0e-13


@dataclass(frozen=True)
class StationCoupling:
    """Coupled groups of a station population (module notes).

    ``group_size`` is the number of cores ``n`` (stations per group),
    ``interlayer_mu`` the load-weighted coefficient of the interlayer contacts
    over the total core load (the capstan bound of the interlayer share takes
    it with the total contact pressure), ``intralayer_mu`` the core--core
    coefficient and ``intralayer_pressure_Pa`` the pressure equivalent
    ``n q_cc / (2 pi r_c)`` of the core--core line load ``q_cc``.
    """

    group_size: int
    interlayer_mu: float
    intralayer_mu: float
    intralayer_pressure_Pa: float

    def __post_init__(self) -> None:
        n = self.group_size
        if isinstance(n, bool) or not isinstance(n, int) or n < 2:
            raise CableInputError("the coupled group size must be an integer >= 2")
        for name in ("interlayer_mu", "intralayer_mu", "intralayer_pressure_Pa"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise CableInputError(f"{name} must be finite and non-negative")

    @property
    def plateau_rad(self) -> float:
        """Angle from the neutral axis beyond which ``C`` is flat, ``pi / n``."""
        return math.pi / self.group_size


def group_view(values: FloatArray, group_size: int) -> FloatArray:
    """Station values ``(K,)`` as groups ``(G, n)``: row ``g`` holds stations
    ``g, g + G, ..., g + (n - 1) G`` (a view, no copy)."""
    count = values.shape[0]
    if count % group_size:
        raise CableInputError(
            f"{count} stations do not form groups of {group_size} cores"
        )
    return values.reshape(group_size, count // group_size).T


def _root_numpy(d: FloatArray, cap: FloatArray) -> FloatArray:
    """``nu > 0`` with ``sum clip(d - nu, -cap, cap) = 0`` row by row, for rows
    whose sum at ``nu = 0`` is positive (so ``sum cap > 0``)."""
    points = np.sort(np.concatenate((d - cap, d + cap), axis=1), axis=1)
    values = np.sum(
        np.clip(d[:, None, :] - points[:, :, None], -cap[:, None, :], cap[:, None, :]),
        axis=2,
    )
    # Non-increasing along the breakpoints, sum(cap) > 0 at the first and
    # -sum(cap) < 0 at the last: the root lies after the last positive value.
    upper = np.argmax(values <= 0.0, axis=1)
    rows = np.arange(d.shape[0])
    x0, x1 = points[rows, upper - 1], points[rows, upper]
    f0, f1 = values[rows, upper - 1], values[rows, upper]
    return np.asarray(x0 + f0 * (x1 - x0) / (f0 - f1), dtype=np.float64)


def _project_numpy(
    trial: FloatArray, upper: FloatArray, lower: FloatArray, cap: FloatArray
) -> FloatArray:
    """NumPy path of :func:`project_station_groups` on ``(G, n)`` arrays."""
    out = np.clip(trial, -(lower + cap), upper + cap)
    margin = _ROUND_OFF * np.sum(np.abs(trial) + upper + lower + cap, axis=1)
    rising = np.sum(np.clip(trial - upper, -cap, cap), axis=1) > margin
    falling = np.sum(np.clip(trial + lower, -cap, cap), axis=1) < -margin
    for rows, sign in ((rising, 1.0), (falling, -1.0)):
        if not np.any(rows):
            continue
        t, u, lo, c = trial[rows], upper[rows], lower[rows], cap[rows]
        # Falling: the mirrored problem on -(t + L) has the same root form.
        d = t - u if sign > 0.0 else -(t + lo)
        nu = _root_numpy(d, c)
        share = sign * np.clip(d - nu[:, None], -c, c)
        out[rows] = np.clip(t - share, -lo, u) + share
    return np.asarray(out, dtype=np.float64)


@njit(cache=True)
def _root_kernel(d: FloatArray, cap: FloatArray, points: FloatArray) -> float:
    n = d.size
    for i in range(n):
        points[i] = d[i] - cap[i]
        points[n + i] = d[i] + cap[i]
    m = 2 * n
    for i in range(1, m):  # insertion sort (n is the core count)
        key = points[i]
        j = i - 1
        while j >= 0 and points[j] > key:
            points[j + 1] = points[j]
            j -= 1
        points[j + 1] = key
    previous_x = points[0]
    previous_f = 0.0
    for i in range(n):
        previous_f += cap[i]
    for k in range(1, m):
        x = points[k]
        f = 0.0
        for i in range(n):
            v = d[i] - x
            if v > cap[i]:
                v = cap[i]
            elif v < -cap[i]:
                v = -cap[i]
            f += v
        if f <= 0.0:
            return float(previous_x + previous_f * (x - previous_x) / (previous_f - f))
        previous_x = x
        previous_f = f
    return float(points[m - 1])  # pragma: no cover - unreachable (sum(cap) > 0)


@njit(cache=True)
def _project_kernel(
    trial: FloatArray,
    upper: FloatArray,
    lower: FloatArray,
    cap: FloatArray,
    out: FloatArray,
) -> None:
    """Compiled path of :func:`project_station_groups` on ``(G, n)`` arrays."""
    groups, n = trial.shape
    d = np.empty(n)
    c = np.empty(n)
    points = np.empty(2 * n)
    for g in range(groups):
        rising = 0.0
        falling = 0.0
        margin = 0.0
        for i in range(n):
            ci = cap[g, i]
            v = trial[g, i] - upper[g, i]
            rising += min(max(v, -ci), ci)
            w = trial[g, i] + lower[g, i]
            falling += min(max(w, -ci), ci)
            margin += abs(trial[g, i]) + upper[g, i] + lower[g, i] + ci
        margin *= _ROUND_OFF
        if rising > margin:
            sign = 1.0
        elif falling < -margin:
            sign = -1.0
        else:
            for i in range(n):
                hi = upper[g, i] + cap[g, i]
                lo = -(lower[g, i] + cap[g, i])
                out[g, i] = min(max(trial[g, i], lo), hi)
            continue
        for i in range(n):
            c[i] = cap[g, i]
            if sign > 0.0:
                d[i] = trial[g, i] - upper[g, i]
            else:
                d[i] = -(trial[g, i] + lower[g, i])
        nu = _root_kernel(d, c, points)
        for i in range(n):
            share = sign * min(max(d[i] - nu, -c[i]), c[i])
            a = min(max(trial[g, i] - share, -lower[g, i]), upper[g, i])
            out[g, i] = a + share


def project_station_groups(
    trial: FloatArray,
    upper: FloatArray,
    lower: FloatArray,
    cap: FloatArray,
    group_size: int,
    *,
    compiled: bool | None = None,
) -> FloatArray:
    """Return map of the coupled station groups (module notes).

    ``trial``, ``upper``, ``lower`` (interlayer bounds, magnitudes) and ``cap``
    (intralayer bound ``C``) are per station, shape ``(K,)``; the result is
    the stress of every station. ``compiled`` selects the numba kernel
    (default: when numba is active) or the NumPy path.
    """
    arrays = [
        group_view(np.ascontiguousarray(x, dtype=np.float64), group_size)
        for x in (trial, upper, lower, cap)
    ]
    use_kernel = HAVE_NUMBA if compiled is None else compiled
    if use_kernel:
        t, u, lo, c = (np.ascontiguousarray(x) for x in arrays)
        out = np.empty_like(t)
        _project_kernel(t, u, lo, c, out)
    else:
        out = _project_numpy(*arrays)
    # Back to station order (row g, column j is station j G + g).
    return np.ascontiguousarray(out.T).reshape(-1)
