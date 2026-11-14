# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Exact path-work bounds for an elastic/Coulomb slider's unknown initial force.

Displacement is measured in elastic yield displacement and force in yield
force. Initial force may be anywhere in [-1, 1]. The input is an ordered,
piecewise-linear displacement path; neither a closing segment nor a zero
initial state is silently added. No CableXSec mechanics are imported.
"""

from dataclasses import dataclass

import numpy as np


def _path(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("a finite nonempty displacement path is required")
    return values


def path_work(displacement, initial_force):
    """Integrate force along each elastic and plastic segment exactly.

    Work includes the change of stored energy and can be negative on an
    open unloading path. It is not necessarily frictional dissipation.
    """
    displacement = _path(displacement)
    force = np.asarray(initial_force, dtype=float)
    if not np.isfinite(force).all() or np.any(np.abs(force) > 1):
        raise ValueError("initial force must be finite and within [-1, 1]")
    work = np.zeros_like(force)
    for increment in np.diff(displacement):
        next_force = np.clip(force + increment, -1, 1)
        elastic = next_force - force
        plastic = increment - elastic
        # The force changes linearly on the elastic portion and is constant
        # on the remainder of this monotone displacement segment.
        work += 0.5 * (force + next_force) * elastic + next_force * plastic
        force = next_force
    return work


@dataclass(frozen=True)
class InitialWorkBounds:
    minimum: float
    maximum: float
    minimising_initial_force: float
    maximising_initial_force: float
    intervals: int


def initial_work_bounds(displacement):
    """Find global extrema by enumerating all initial-force breakpoints.

    Before first yield, force is initial_force plus the cumulative imposed
    displacement. First yield erases dependence of subsequent force on the
    initial force. Breakpoints therefore occur only at +/-1 minus those
    cumulative displacements. Work is at most quadratic between them.
    Endpoints and any interior quadratic stationary points give exact extrema
    up to floating-point evaluation, without optimising a sampled curve.
    """
    displacement = _path(displacement)
    relative = displacement - displacement[0]
    knots = np.unique(np.clip(np.r_[-1.0, 1.0, 1 - relative, -1 - relative], -1, 1))
    left, right = knots[:-1], knots[1:]
    middle = (left + right) / 2
    endpoints = path_work(displacement, knots)
    midpoint = path_work(displacement, middle)
    # Polynomial in t=(initial_force-left)/(right-left), so tiny intervals
    # do not require division by a squared interval width.
    quadratic = 2 * (endpoints[:-1] + endpoints[1:] - 2 * midpoint)
    linear = endpoints[1:] - endpoints[:-1] - quadratic
    scale = np.maximum(1, np.maximum(np.abs(endpoints[:-1]), np.abs(endpoints[1:])))
    curved = np.abs(quadratic) > 128 * np.finfo(float).eps * scale
    position = np.full_like(quadratic, np.nan)
    np.divide(-linear, 2 * quadratic, out=position, where=curved)
    inside = curved & (position > 0) & (position < 1)
    stationary = left[inside] + position[inside] * (right[inside] - left[inside])
    candidates = np.r_[knots, stationary]
    work = np.r_[endpoints, path_work(displacement, stationary)]
    imin, imax = int(np.argmin(work)), int(np.argmax(work))
    return InitialWorkBounds(
        minimum=float(work[imin]),
        maximum=float(work[imax]),
        minimising_initial_force=float(candidates[imin]),
        maximising_initial_force=float(candidates[imax]),
        intervals=len(left),
    )
