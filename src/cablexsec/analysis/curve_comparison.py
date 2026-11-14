# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Sequence-preserving comparison of cyclic, experimentally measured curves.

Branches are determined from the imposed/measured abscissa alone. Predicted
ordinates never choose the experimental branch. Work is integrated between
equal-abscissa crossings, with one crossing per major increasing branch, so
small instrument reversals cannot be mistaken for complete loading cycles.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from cablexsec.exceptions import CableInputError

FloatArray = NDArray[np.float64]


def major_branches(
    abscissa: FloatArray, *, prominence_fraction: float = 0.25
) -> tuple[FloatArray, NDArray[np.int64]]:
    """Return branch directions and major reversals in acquisition order.

    Prominence is relative to the recorded abscissa range. A half-range
    threshold can miss actual reversals near the ends of an incomplete cycle;
    the default quarter-range retains them while rejecting small instrument
    fluctuations. This is a comparison setting, not a constitutive parameter.
    """
    if not np.isfinite(prominence_fraction) or not 0 < prominence_fraction <= 1:
        raise CableInputError(
            "prominence fraction must be finite and between zero and one"
        )
    values = np.asarray(abscissa, dtype=float)
    if values.ndim != 1 or values.size < 3 or not np.isfinite(values).all():
        raise CableInputError(
            "a finite one-dimensional record with at least three points is required"
        )
    extent = float(np.ptp(values))
    if extent == 0:
        return np.zeros_like(values), np.array([0, len(values) - 1], dtype=np.int64)
    maxima = find_peaks(values, prominence=prominence_fraction * extent)[0]
    minima = find_peaks(-values, prominence=prominence_fraction * extent)[0]
    reversals = np.unique(np.r_[0, maxima, minima, len(values) - 1]).astype(np.int64)
    direction = np.zeros_like(values)
    for lo, hi in zip(reversals[:-1], reversals[1:], strict=True):
        direction[lo : hi + 1] = np.sign(values[hi] - values[lo])
    return direction, reversals


def _cycle_crossings(
    abscissa: FloatArray,
    crossing: float | None,
) -> tuple[float, list[tuple[int, float]]]:
    """Segments and fractions for one crossing per major increasing branch."""
    _, reversals = major_branches(abscissa)
    centre = (
        float(0.5 * (np.min(abscissa) + np.max(abscissa)))
        if crossing is None
        else float(crossing)
    )
    if not np.isfinite(centre):
        raise CableInputError("crossing must be finite")
    crossings = []
    for lo, hi in zip(reversals[:-1], reversals[1:], strict=True):
        if abscissa[hi] <= abscissa[lo]:
            continue
        indices = np.flatnonzero(
            (abscissa[lo:hi] < centre) & (abscissa[lo + 1 : hi + 1] >= centre)
        )
        if indices.size:
            i = int(lo + indices[0])
            fraction = (centre - abscissa[i]) / (abscissa[i + 1] - abscissa[i])
            crossings.append((i, float(fraction)))
    return centre, crossings


def insert_cycle_crossings(
    abscissa: FloatArray,
    *,
    crossing: float | None = None,
) -> tuple[FloatArray, NDArray[np.int64], NDArray[np.int64]]:
    """Insert cycle boundaries without changing the recorded loading path.

    Returns the augmented abscissa, indices of the original observations and
    indices of cycle crossings. The additional points let a constitutive
    calculation evaluate stored energy and friction exactly at those boundaries,
    instead of interpolating a coarse moment polygon through a yield transition.
    Added points are calculation stations, not experimental observations.
    """
    values = np.asarray(abscissa, dtype=float)
    centre, crossings = _cycle_crossings(values, crossing)
    segments = dict(crossings)
    augmented, original_indices, crossing_indices = [float(values[0])], [0], []
    for i in range(len(values) - 1):
        fraction = segments.get(i)
        if fraction is not None and fraction < 1.0:
            crossing_indices.append(len(augmented))
            augmented.append(centre)
        original_indices.append(len(augmented))
        augmented.append(float(values[i + 1]))
        if fraction == 1.0:
            crossing_indices.append(len(augmented) - 1)
    return (
        np.asarray(augmented, dtype=float),
        np.asarray(original_indices, dtype=np.int64),
        np.asarray(crossing_indices, dtype=np.int64),
    )


def cycle_work(
    abscissa: FloatArray,
    ordinate: FloatArray,
    *,
    crossing: float | None = None,
) -> FloatArray:
    """Signed loop areas between complete, equal-abscissa increasing crossings.

    The sign follows the acquisition order; no absolute value hides a negative
    dissipation. An empty array means that the record contains no complete cycle.
    The first crossing on each major increasing branch is used independently of
    the ordinate values. Linear interpolation includes the two partial segments.
    """
    values, forces = (
        np.asarray(abscissa, dtype=float),
        np.asarray(ordinate, dtype=float),
    )
    centre, crossings = _cycle_crossings(values, crossing)
    if forces.shape != values.shape or not np.isfinite(forces).all():
        raise CableInputError("ordinate must be finite and match the abscissa")
    integral = np.r_[0.0, np.cumsum(0.5 * (forces[1:] + forces[:-1]) * np.diff(values))]
    crossing_work = []
    for i, fraction in crossings:
        force_at_crossing = forces[i] + fraction * (forces[i + 1] - forces[i])
        crossing_work.append(
            integral[i] + 0.5 * (forces[i] + force_at_crossing) * (centre - values[i])
        )
    return np.diff(np.asarray(crossing_work, dtype=float))
