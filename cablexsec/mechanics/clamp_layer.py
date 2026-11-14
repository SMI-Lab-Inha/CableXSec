# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Friction boundary layer of one armour wire anchored at a clamp.

The sectional friction laws of the package (:mod:`~cablexsec.mechanics.fast_slip`,
:mod:`~cablexsec.mechanics.distributed_friction`) assume curvature that is
uniform over at least one pitch and no anchorage. At a test-rig clamp or an
end fitting the armour is anchored, so friction feeds a mean wire force that
decays over a boundary layer. This module solves that problem for one wire
under planar bending with a curvature that varies along the cable axis.

Model (normalised; rigid stick, Coulomb bound, axial slip only)::

    n = e + du/dphi,   |dn/dphi| <= rho(t),   dn/dphi = rho sign(du/dt) in slip,
    e = a(t) k_hat(x) cos(theta(x)),   u = 0 at both anchored ends,

with ``phi`` the helix angle along the wire (one pitch is ``2 pi``),
``a = kappa_ref / kappa_c``, ``kappa_c`` the slip-onset curvature of the
layer (:func:`~cablexsec.mechanics.papailiou.slip_onset_curvature`, which
includes the Yun coefficient), ``k_hat(x)`` the curvature shape along the
specimen, ``theta(x)`` the angular position of the wire measured from the
tensile extreme fibre and ``rho(t) = P(t) / P_ref`` the clamping-pressure
ratio. The wire force is ``N = g n`` and its stress ``sigma = sigma_u n`` with
``sigma_u = sigma_0(P_ref) mu |sin a|``, the slope of the linearised
Papailiou cap.

Discretisation: ``M`` segments of length ``h`` in ``phi``; segment forces
``n_j``; node friction ``n_j - n_{j-1}`` bounded by ``rho h``. Each increment
is the Euclidean projection of the trial force ``n + da e_hat`` onto the
open-chain Lipschitz set ``{|n_j - n_{j-1}| <= rho h}``, solved exactly by a
dynamic programme. The set is translation invariant, so the projection keeps
``sum n``, which is the condition that the far anchor does not move.

Closed forms (uniform curvature from the anchor):

* far field: the Kraincanic--Kebadze slip front,
  ``n = sign(c) min(a |c|, arcsin |c|)``, ``c = cos theta``; peak ``min(a, pi/2)``;
* below slip (``a < 1``) the wire sticks and there is no clamp effect;
* ``a >> 1``: averaged mean force at the anchor
  ``|H0| = sqrt((4 a / pi)(x arcsin x + sqrt(1 - x^2) - 1))``, ``x = |sin theta0|``,
  worst on the neutral-axis wire (``x = 1``): ``|H0| = 0.8525 sqrt(a)``, and the
  amplification of the peak friction force ``A = 1 + 0.5427 sqrt(a)``;
* boundary-layer length ``l = sqrt(pi a / 2)`` radians, i.e.
  ``l_x = sqrt(L_s P) / 2`` with the slip length ``L_s = a P / (2 pi)``.

The solver and closed forms are a port of a separately verified reference
implementation. The tests of this package
check the projection against a generic quadratic programme, the far field
against Kraincanic--Kebadze, the stick regime, the energy identity, mesh and
step convergence, and the clamp law at large ``a``.

Limits: rigid stick (the ``xi``-reduced stick stiffness enters through
``kappa_c`` only), constant bound along the wire, axial slip only, one wire
(no interaction between layers), planar proportional loading
``kappa(x, t) = kappa_ref(t) k_hat(x)``. The clamp is the anchoring plane of
the armour; a bonded (epoxy) length inside a fitting is not frictional.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from cablexsec._numba import HAVE_NUMBA, njit
from cablexsec.exceptions import CableInputError

__all__ = [
    "CLAMP_LAW_COEFFICIENT",
    "HAVE_NUMBA",
    "anchored_wire_field_response",
    "anchored_wire_node_dissipation",
    "anchored_wire_response",
    "chain_projection",
    "chain_projection_reference",
    "clamp_amplification",
    "clamp_boundary_layer_length",
    "clamp_mean_force",
    "clamp_offset_potential",
    "clamp_setback_length",
    "gradient_slip_onset_factor",
    "uniform_curvature_force",
]

FloatArray = NDArray[np.float64]

#: ``(2 / pi) sqrt(2 - 4 / pi)``: worst-wire clamp law ``A = 1 + C sqrt(a)``.
CLAMP_LAW_COEFFICIENT = 2.0 / math.pi * math.sqrt(2.0 - 4.0 / math.pi)


# --------------------------------------------------------------------------- #
# Exact open-chain projection                                                  #
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _lower_bound(values: FloatArray, size: int, target: float) -> int:
    lo, hi = 0, size
    while lo < hi:
        mid = (lo + hi) >> 1
        if values[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _upper_bound(values: FloatArray, size: int, target: float) -> int:
    lo, hi = 0, size
    while lo < hi:
        mid = (lo + hi) >> 1
        if values[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _project(
    y: FloatArray,
    bound: float,
    out: FloatArray,
    xstar: FloatArray,
    xs: FloatArray,
    gs: FloatArray,
    xt: FloatArray,
    gt: FloatArray,
) -> None:
    """Dynamic programme on the derivative of the cost-to-come (slope trick).

    Breakpoints outside ``[min y, max y]`` are pruned, which is exact because
    every partial optimum lies in that range.
    """
    m = y.size
    lo = y[0]
    hi = y[0]
    for i in range(1, m):
        lo = min(lo, y[i])
        hi = max(hi, y[i])
    size = 1
    xs[0] = y[0]
    gs[0] = 0.0
    sl = 1.0
    sr = 1.0
    for i in range(m):
        if i > 0:
            x0 = xstar[i - 1]
            k0 = _lower_bound(xs, size, x0)
            k1 = _upper_bound(xs, size, x0)
            t = 0
            for j in range(k0):
                xt[t] = xs[j] - bound
                gt[t] = gs[j]
                t += 1
            xt[t] = x0 - bound
            gt[t] = 0.0
            t += 1
            xt[t] = x0 + bound
            gt[t] = 0.0
            t += 1
            for j in range(k1, size):
                xt[t] = xs[j] + bound
                gt[t] = gs[j]
                t += 1
            yi = y[i]
            for j in range(t):
                gt[j] += xt[j] - yi
            sl += 1.0
            sr += 1.0
            j0 = _lower_bound(xt, t, lo) - 1
            j0 = max(j0, 0)
            j1 = _upper_bound(xt, t, hi) + 1
            j1 = min(j1, t)
            size = j1 - j0
            for j in range(size):
                xs[j] = xt[j0 + j]
                gs[j] = gt[j0 + j]
        k = _lower_bound(gs, size, 0.0)
        if k == 0:
            xstar[i] = xs[0] - gs[0] / sl
        elif k == size:
            xstar[i] = xs[size - 1] - gs[size - 1] / sr
        else:
            x1 = xs[k - 1]
            x2 = xs[k]
            g1 = gs[k - 1]
            g2 = gs[k]
            xstar[i] = x1 - g1 * (x2 - x1) / (g2 - g1)
    out[m - 1] = xstar[m - 1]
    for i in range(m - 2, -1, -1):
        value = xstar[i]
        value = max(value, out[i + 1] - bound)
        value = min(value, out[i + 1] + bound)
        out[i] = value


def chain_projection(y: FloatArray, bound: float) -> FloatArray:
    """``argmin 0.5 |n - y|^2`` subject to ``|n_{i+1} - n_i| <= bound`` (exact).

    Compiled when ``numba`` is available; :func:`chain_projection_reference`
    is the vectorised NumPy reference.
    """
    values = np.ascontiguousarray(y, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise CableInputError("projection input must be a finite, non-empty vector")
    if not math.isfinite(bound) or bound < 0.0:
        raise CableInputError("projection bound must be finite and non-negative")
    if not HAVE_NUMBA:
        return chain_projection_reference(values, float(bound))
    m = values.size
    out = np.empty(m)
    work = np.empty((4, 2 * m + 4))
    _project(values, float(bound), out, np.empty(m), work[0], work[1], work[2], work[3])
    return out


def chain_projection_reference(y: FloatArray, bound: float) -> FloatArray:
    """NumPy reference of :func:`chain_projection` (same programme)."""
    values = np.asarray(y, dtype=np.float64)
    m = values.size
    lo, hi = values.min(), values.max()
    xstar = np.empty(m)
    xs = np.array([values[0]])
    gs = np.array([0.0])
    sl = sr = 1.0
    for i in range(m):
        if i > 0:
            x0 = xstar[i - 1]
            k0 = int(np.searchsorted(xs, x0, side="left"))
            k1 = int(np.searchsorted(xs, x0, side="right"))
            xs = np.concatenate(
                [xs[:k0] - bound, [x0 - bound, x0 + bound], xs[k1:] + bound]
            )
            gs = np.concatenate([gs[:k0], [0.0, 0.0], gs[k1:]])
            gs = gs + (xs - values[i])
            sl += 1.0
            sr += 1.0
            if xs.size > 64:
                j0 = max(int(np.searchsorted(xs, lo, side="left")) - 1, 0)
                j1 = int(np.searchsorted(xs, hi, side="right")) + 1
                xs, gs = xs[j0:j1], gs[j0:j1]
        k = int(np.searchsorted(gs, 0.0))
        if k == 0:
            xstar[i] = xs[0] - gs[0] / sl
        elif k == gs.size:
            xstar[i] = xs[-1] - gs[-1] / sr
        else:
            x1, x2, g1, g2 = xs[k - 1], xs[k], gs[k - 1], gs[k]
            xstar[i] = x1 - g1 * (x2 - x1) / (g2 - g1)
    out = np.empty(m)
    out[-1] = xstar[-1]
    for i in range(m - 2, -1, -1):
        out[i] = min(max(xstar[i], out[i + 1] - bound), out[i + 1] + bound)
    return out


# --------------------------------------------------------------------------- #
# Incremental anchored-wire history                                           #
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _anchored_kernel(
    e_hat: FloatArray,
    cap_hat: FloatArray,
    load: FloatArray,
    rho: FloatArray,
    h: float,
    record: NDArray[np.int64],
    index: NDArray[np.int64],
    out_n: FloatArray,
    out_ref: FloatArray,
) -> float:
    m = e_hat.size
    n = np.zeros(m)
    ref = np.zeros(m)
    y = np.empty(m)
    new = np.empty(m)
    xstar = np.empty(m)
    xs = np.empty(2 * m + 4)
    gs = np.empty(2 * m + 4)
    xt = np.empty(2 * m + 4)
    gt = np.empty(2 * m + 4)
    old = 0.0
    dissipation = 0.0
    row = 0
    for i in range(load.size):
        step = load[i] - old
        bound = rho[i] * h
        for j in range(m):
            y[j] = n[j] + step * e_hat[j]
        _project(y, bound, new, xstar, xs, gs, xt, gt)
        slip = 0.0
        for j in range(m - 1):
            slip += h * (new[j] - y[j])
            dissipation += bound * abs(slip)
        for j in range(m):
            n[j] = new[j]
            limit = rho[i] * cap_hat[j]
            trial = ref[j] + step * e_hat[j]
            ref[j] = min(max(trial, -limit), limit)
        old = load[i]
        if record[i] != 0:
            for k in range(index.size):
                out_n[row, k] = n[index[k]]
                out_ref[row, k] = ref[index[k]]
            row += 1
    return dissipation


def anchored_wire_response(
    e_hat: FloatArray,
    load: FloatArray,
    h: float,
    *,
    rho: FloatArray | None = None,
    cap_hat: FloatArray | None = None,
    record: NDArray[np.bool_] | None = None,
    index: NDArray[np.int64] | None = None,
) -> tuple[FloatArray, FloatArray, float]:
    """Incremental response of one wire anchored at both ends.

    Parameters
    ----------
    e_hat : numpy.ndarray
        Stuck force per unit load at the segment midpoints,
        ``k_hat(x_j) cos(theta(x_j))``.
    load : numpy.ndarray
        Load path ``a(t)`` (normalised curvature at the reference station),
        starting from the virgin state ``a = 0``. Increments must be small
        (about ``0.5`` or less) because slip directions reverse near an anchor.
    h : float
        Segment length in helix angle [rad] (``2 pi`` per pitch).
    rho : numpy.ndarray, optional
        Friction-bound ratio per step (default one).
    cap_hat : numpy.ndarray, optional
        Pointwise cap for the sectional reference, per unit bound
        (``arcsin |cos theta|`` for planar bending; default zero).
    record : numpy.ndarray of bool, optional
        Steps to record (default all).
    index : numpy.ndarray of int, optional
        Segments to record (default all).

    Returns
    -------
    n, n_ref : numpy.ndarray
        Recorded anchored-wire forces and the pointwise elastic--plastic
        (sectional) reference, shape ``(recorded steps, recorded segments)``.
    dissipation : float
        Friction dissipation over the path (normalised: ``g`` times slip
        ``w EA sin a / (g R)``, per wire).
    """
    e = np.ascontiguousarray(e_hat, dtype=np.float64)
    path = np.ascontiguousarray(load, dtype=np.float64)
    if e.ndim != 1 or e.size < 2 or not np.isfinite(e).all():
        raise CableInputError("e_hat must be a finite vector of at least two segments")
    if path.ndim != 1 or not np.isfinite(path).all():
        raise CableInputError("load path must be a finite vector")
    if not math.isfinite(h) or h <= 0.0:
        raise CableInputError("segment length must be positive")
    ratio = (
        np.ones(path.size)
        if rho is None
        else np.ascontiguousarray(rho, dtype=np.float64)
    )
    if ratio.shape != path.shape or not np.isfinite(ratio).all() or np.any(ratio < 0):
        raise CableInputError("rho must be finite, non-negative and match the load")
    cap = np.zeros(e.size) if cap_hat is None else np.ascontiguousarray(cap_hat, float)
    if cap.shape != e.shape:
        raise CableInputError("cap_hat must match e_hat")
    flags = (
        np.ones(path.size, dtype=np.int64)
        if record is None
        else np.asarray(record, dtype=np.int64)
    )
    if flags.shape != path.shape:
        raise CableInputError("record must match the load path")
    idx = (
        np.arange(e.size, dtype=np.int64)
        if index is None
        else np.ascontiguousarray(index, dtype=np.int64)
    )
    if idx.size and (idx.min() < 0 or idx.max() >= e.size):
        raise CableInputError("recorded segment index out of range")
    rows = int(np.count_nonzero(flags))
    out_n = np.zeros((rows, idx.size))
    out_ref = np.zeros((rows, idx.size))
    kernel = _anchored_kernel if HAVE_NUMBA else _anchored_numpy
    dissipation = kernel(e, cap, path, ratio, float(h), flags, idx, out_n, out_ref)
    return out_n, out_ref, float(dissipation)


def _anchored_numpy(
    e_hat: FloatArray,
    cap_hat: FloatArray,
    load: FloatArray,
    rho: FloatArray,
    h: float,
    record: NDArray[np.int64],
    index: NDArray[np.int64],
    out_n: FloatArray,
    out_ref: FloatArray,
) -> float:
    """NumPy path of :func:`_anchored_kernel` (used without ``numba``)."""
    n = np.zeros(e_hat.size)
    ref = np.zeros(e_hat.size)
    old = 0.0
    dissipation = 0.0
    row = 0
    for i in range(load.size):
        step = load[i] - old
        bound = rho[i] * h
        y = n + step * e_hat
        n = chain_projection_reference(y, bound)
        dissipation += bound * float(np.abs(np.cumsum(h * (n - y))[:-1]).sum())
        limit = rho[i] * cap_hat
        ref = np.clip(ref + step * e_hat, -limit, limit)
        old = load[i]
        if record[i]:
            out_n[row] = n[index]
            out_ref[row] = ref[index]
            row += 1
    return dissipation


# --------------------------------------------------------------------------- #
# Non-proportional curvature fields (gap P5)                                  #
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _field_kernel(
    drive: FloatArray,
    c_hat: FloatArray,
    cap_hat: FloatArray,
    counts: NDArray[np.int64],
    rho: FloatArray,
    h: float,
    index: NDArray[np.int64],
    out_n: FloatArray,
    out_ref: FloatArray,
) -> float:
    n_s, m = drive.shape
    n = np.zeros(m)
    ref = np.zeros(m)
    y = np.empty(m)
    new = np.empty(m)
    de = np.empty(m)
    xstar = np.empty(m)
    xs = np.empty(2 * m + 4)
    gs = np.empty(2 * m + 4)
    xt = np.empty(2 * m + 4)
    gt = np.empty(2 * m + 4)
    dissipation = 0.0
    for i in range(n_s):
        k = counts[i]
        r0 = rho[i - 1] if i > 0 else rho[0]
        for j in range(m):
            prev = drive[i - 1, j] if i > 0 else 0.0
            de[j] = (drive[i, j] - prev) * c_hat[j] / k
        for q in range(k):
            w = (q + 1.0) / k
            bound = ((1.0 - w) * r0 + w * rho[i]) * h
            for j in range(m):
                y[j] = n[j] + de[j]
            _project(y, bound, new, xstar, xs, gs, xt, gt)
            slip = 0.0
            for j in range(m - 1):
                slip += h * (new[j] - y[j])
                dissipation += bound * abs(slip)
            for j in range(m):
                n[j] = new[j]
                limit = bound / h * cap_hat[j]
                trial = ref[j] + de[j]
                ref[j] = min(max(trial, -limit), limit)
        for p in range(index.size):
            out_n[i, p] = n[index[p]]
            out_ref[i, p] = ref[index[p]]
    return dissipation


@njit(cache=True)
def _node_dissipation_kernel(
    drive: FloatArray,
    c_hat: FloatArray,
    counts: NDArray[np.int64],
    h: float,
    start: int,
    out: FloatArray,
) -> None:
    n_s, m = drive.shape
    n = np.zeros(m)
    y = np.empty(m)
    new = np.empty(m)
    de = np.empty(m)
    xstar = np.empty(m)
    xs = np.empty(2 * m + 4)
    gs = np.empty(2 * m + 4)
    xt = np.empty(2 * m + 4)
    gt = np.empty(2 * m + 4)
    for i in range(n_s):
        k = counts[i]
        for j in range(m):
            prev = drive[i - 1, j] if i > 0 else 0.0
            de[j] = (drive[i, j] - prev) * c_hat[j] / k
        for _q in range(k):
            for j in range(m):
                y[j] = n[j] + de[j]
            _project(y, h, new, xstar, xs, gs, xt, gt)
            slip = 0.0
            for j in range(m - 1):
                slip += h * (new[j] - y[j])
                if i >= start:
                    out[j] += h * abs(slip)
            for j in range(m):
                n[j] = new[j]


def _node_dissipation_numpy(
    drive: FloatArray,
    c_hat: FloatArray,
    counts: NDArray[np.int64],
    h: float,
    start: int,
    out: FloatArray,
) -> None:
    """NumPy path of :func:`_node_dissipation_kernel` (used without ``numba``)."""
    n = np.zeros(c_hat.size)
    previous = np.zeros(c_hat.size)
    for i in range(drive.shape[0]):
        k = int(counts[i])
        de = (drive[i] - previous) * c_hat / k
        for _q in range(k):
            y = n + de
            n = chain_projection_reference(y, h)
            if i >= start:
                out += h * np.abs(np.cumsum(h * (n - y))[:-1])
        previous = drive[i]


def anchored_wire_node_dissipation(
    drive: FloatArray,
    c_hat: FloatArray,
    h: float,
    *,
    from_sample: int = 0,
    max_increment: float = 0.5,
) -> FloatArray:
    """Friction dissipation at every node of a wire anchored at both ends.

    The path of :func:`anchored_wire_field_response` (bound ratio one, no
    sectional cap), with the dissipation kept node by node instead of summed:
    node ``j`` lies between segments ``j`` and ``j + 1``. Only substeps of
    samples ``i >= from_sample`` are counted, so a window over the last cycle
    of a periodic record gives the dissipation per cycle along the wire.
    Normalised as :func:`anchored_wire_response`; with ``from_sample=0`` the
    nodes sum to its total dissipation.
    """
    a = np.ascontiguousarray(drive, dtype=np.float64)
    c = np.ascontiguousarray(c_hat, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] < 2 or not np.isfinite(a).all():
        raise CableInputError("drive must be a finite (samples, segments) array")
    if c.shape != (a.shape[1],) or not np.isfinite(c).all():
        raise CableInputError("c_hat must match the segments of the drive")
    if not math.isfinite(h) or h <= 0.0:
        raise CableInputError("segment length must be positive")
    if not math.isfinite(max_increment) or max_increment <= 0.0:
        raise CableInputError("max_increment must be positive")
    if not 0 <= from_sample < a.shape[0]:
        raise CableInputError("from_sample must index a sample of the drive")
    previous = np.vstack((np.zeros((1, c.size)), a[:-1]))
    step = np.max(np.abs((a - previous) * c[None, :]), axis=1)
    counts = np.maximum(1, np.ceil(step / max_increment)).astype(np.int64)
    out = np.zeros(c.size - 1)
    kernel = _node_dissipation_kernel if HAVE_NUMBA else _node_dissipation_numpy
    kernel(a, c, counts, float(h), int(from_sample), out)
    return out


def _field_numpy(
    drive: FloatArray,
    c_hat: FloatArray,
    cap_hat: FloatArray,
    counts: NDArray[np.int64],
    rho: FloatArray,
    h: float,
    index: NDArray[np.int64],
    out_n: FloatArray,
    out_ref: FloatArray,
) -> float:
    """NumPy path of :func:`_field_kernel` (used without ``numba``)."""
    n = np.zeros(c_hat.size)
    ref = np.zeros(c_hat.size)
    dissipation = 0.0
    previous = np.zeros(c_hat.size)
    for i in range(drive.shape[0]):
        k = int(counts[i])
        r0 = rho[i - 1] if i > 0 else rho[0]
        de = (drive[i] - previous) * c_hat / k
        for q in range(k):
            w = (q + 1.0) / k
            bound = ((1.0 - w) * r0 + w * rho[i]) * h
            y = n + de
            n = chain_projection_reference(y, bound)
            dissipation += bound * float(np.abs(np.cumsum(h * (n - y))[:-1]).sum())
            limit = bound / h * cap_hat
            ref = np.clip(ref + de, -limit, limit)
        previous = drive[i]
        out_n[i] = n[index]
        out_ref[i] = ref[index]
    return dissipation


def anchored_wire_field_response(
    drive: FloatArray,
    c_hat: FloatArray,
    h: float,
    *,
    max_increment: float = 0.5,
    rho: FloatArray | None = None,
    cap_hat: FloatArray | None = None,
    index: NDArray[np.int64] | None = None,
    substeps: NDArray[np.int64] | None = None,
) -> tuple[FloatArray, FloatArray, float]:
    """Anchored wire under a non-proportional curvature field (gap P5).

    The stuck force increment of segment ``j`` between samples is
    ``(a_j(t_i) - a_j(t_{i-1})) c_j`` with ``a_j = kappa(x_j, t) / kappa_c``
    (``drive``, shape ``(samples, segments)``) and ``c_j = cos theta(x_j)``:
    the proportional solver of :func:`anchored_wire_response` with
    ``e_hat`` replaced by a field that changes shape in time (a restrictor
    locking joint by joint). Each sample interval is split into equal
    substeps so that no segment's normalised increment exceeds
    ``max_increment``; the bound ratio ``rho`` (per sample) is interpolated
    linearly. The path starts from the virgin state. Records are taken at
    every sample. ``substeps`` (per sample, at least the automatic count)
    fixes the subdivision, so that two runs can share one path.

    Returns ``(n, n_ref, dissipation)`` as :func:`anchored_wire_response`.
    """
    a = np.ascontiguousarray(drive, dtype=np.float64)
    c = np.ascontiguousarray(c_hat, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] < 2 or not np.isfinite(a).all():
        raise CableInputError("drive must be a finite (samples, segments) array")
    if c.shape != (a.shape[1],) or not np.isfinite(c).all():
        raise CableInputError("c_hat must match the segments of the drive")
    if not math.isfinite(h) or h <= 0.0:
        raise CableInputError("segment length must be positive")
    if not math.isfinite(max_increment) or max_increment <= 0.0:
        raise CableInputError("max_increment must be positive")
    ratio = (
        np.ones(a.shape[0])
        if rho is None
        else np.ascontiguousarray(rho, dtype=np.float64)
    )
    if (
        ratio.shape != (a.shape[0],)
        or not np.isfinite(ratio).all()
        or np.any(ratio < 0)
    ):
        raise CableInputError("rho must be finite, non-negative and match the samples")
    cap = np.zeros(c.size) if cap_hat is None else np.ascontiguousarray(cap_hat, float)
    if cap.shape != c.shape:
        raise CableInputError("cap_hat must match the segments")
    idx = (
        np.arange(c.size, dtype=np.int64)
        if index is None
        else np.ascontiguousarray(index, dtype=np.int64)
    )
    if idx.size and (idx.min() < 0 or idx.max() >= c.size):
        raise CableInputError("recorded segment index out of range")
    previous = np.vstack((np.zeros((1, c.size)), a[:-1]))
    step = np.max(np.abs((a - previous) * c[None, :]), axis=1)
    counts = np.maximum(1, np.ceil(step / max_increment)).astype(np.int64)
    if substeps is not None:
        given = np.asarray(substeps, dtype=np.int64)
        if given.shape != counts.shape or np.any(given < 1):
            raise CableInputError("substeps must be positive, one per sample")
        counts = np.ascontiguousarray(np.maximum(counts, given))
    out_n = np.zeros((a.shape[0], idx.size))
    out_ref = np.zeros((a.shape[0], idx.size))
    kernel = _field_kernel if HAVE_NUMBA else _field_numpy
    dissipation = kernel(a, c, cap, counts, ratio, float(h), idx, out_n, out_ref)
    return out_n, out_ref, float(dissipation)


# --------------------------------------------------------------------------- #
# Closed forms                                                                 #
# --------------------------------------------------------------------------- #
def uniform_curvature_force(theta: FloatArray, a: float) -> FloatArray:
    """Kraincanic--Kebadze monotonic field far from any anchor (normalised).

    ``n = sign(c) min(a |c|, arcsin |c|)`` with ``c = cos theta``.
    """
    c = np.cos(np.asarray(theta, dtype=np.float64))
    return np.asarray(
        np.sign(c) * np.minimum(a * np.abs(c), np.arcsin(np.clip(np.abs(c), 0.0, 1.0)))
    )


def clamp_offset_potential(c: FloatArray | float, a: FloatArray | float) -> FloatArray:
    """Averaged friction potential ``Phi(c; a)`` of a mean slip offset ``c``.

    ``Phi = (2 a / pi)(x arcsin x + sqrt(1 - x^2) - 1)``, ``x = |c| / a`` for
    ``x <= 1``, continued with unit slope; ``Phi = |c|`` for ``a = 0``.
    """
    c_in, a_in = np.broadcast_arrays(
        np.asarray(c, dtype=np.float64), np.asarray(a, dtype=np.float64)
    )
    shape = c_in.shape
    cc, aa = c_in.ravel(), a_in.ravel()
    out = np.abs(cc).astype(np.float64)
    positive = aa > 0
    x = np.zeros_like(out)
    x[positive] = np.abs(cc[positive]) / aa[positive]
    inside = positive & (x <= 1.0)
    xi = x[inside]
    out[inside] = (2 * aa[inside] / math.pi) * (
        xi * np.arcsin(xi) + np.sqrt(1 - xi * xi) - 1
    )
    beyond = positive & (x > 1.0)
    out[beyond] = (2 * aa[beyond] / math.pi) * (math.pi / 2 - 1) + (
        np.abs(cc[beyond]) - aa[beyond]
    )
    return np.asarray(out.reshape(shape))


def clamp_mean_force(a: FloatArray | float, x: FloatArray | float = 1.0) -> FloatArray:
    """Averaged mean wire force at the anchor, ``|H0| = sqrt(2 Phi(x a; a))``.

    ``x = |sin theta0|``; the neutral-axis wire (``x = 1``) gives
    ``sqrt(a (2 - 4 / pi)) = 0.8525 sqrt(a)``. Valid for ``a >> 1``.
    """
    aa = np.asarray(a, dtype=np.float64)
    return np.asarray(np.sqrt(2.0 * clamp_offset_potential(np.asarray(x) * aa, aa)))


def clamp_amplification(
    a: FloatArray | float, x: FloatArray | float = 1.0
) -> FloatArray:
    """Clamp law: peak friction force over the sectional peak, ``1 + |H0| / (pi/2)``.

    For the worst wire ``A = 1 + 0.5427 sqrt(a)``. The averaged form is within
    a few per cent of the exact solver for ``a >= 30`` and over-predicts for
    ``a <~ 10``; it applies to full slip (``a > pi / 2``) only, below which
    ``A = 1``.
    """
    aa = np.asarray(a, dtype=np.float64)
    law = 1.0 + clamp_mean_force(aa, x) / (math.pi / 2)
    return np.asarray(np.where(aa > math.pi / 2, law, 1.0))


def clamp_boundary_layer_length(
    a: FloatArray | float, pitch_m: float | None = None
) -> FloatArray:
    """Decay length of the mean force, ``sqrt(pi a / 2)`` radians of helix angle.

    With ``pitch_m`` the axial length ``sqrt(pi a / 2) P / (2 pi) =
    sqrt(L_s P) / 2`` is returned [m].
    """
    rad = np.sqrt(math.pi * np.asarray(a, dtype=np.float64) / 2)
    if pitch_m is None:
        return np.asarray(rad)
    return np.asarray(rad * pitch_m / (2 * math.pi))


def clamp_setback_length(a: FloatArray | float, pitch_m: float) -> FloatArray:
    """Set-back from a curvature step beyond which the anchor adds nothing [m].

    ``d* = 0.4 sqrt(L_s P)``; a regime-map fit of the reference study, not a
    closed form.
    """
    slip_length = np.asarray(a, dtype=np.float64) * pitch_m / (2 * math.pi)
    return np.asarray(0.4 * np.sqrt(slip_length * pitch_m))


def gradient_slip_onset_factor(pitch_m: float, gradient_length_m: float) -> float:
    """Onset curvature at an anchor with exponential decay over ``L_g``.

    ``kappa_onset / kappa_c = 1 / sqrt(1 + (P / (2 pi L_g))^2)`` (exact).
    """
    if gradient_length_m <= 0 or pitch_m <= 0:
        raise CableInputError("pitch and gradient length must be positive")
    return 1.0 / math.sqrt(1.0 + (pitch_m / (2 * math.pi * gradient_length_m)) ** 2)
