# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Compiled rainflow counting and Miner damage for many stress histories.

The kernel reproduces :func:`cablexsec.analysis.fatigue.rainflow_cycles`
followed by the Miner and equivalent-range sums of a per-fibre reference
(``fatigue_pair`` in ``tests/test_fast_rainflow.py``), cycle for cycle and in
the same summation order:

1. interior samples of monotonic runs are removed, keeping the last index of
   each turning plateau, with the first and last samples always retained;
2. the reversal scan of the ``rainflow`` package (ASTM E1049-85, 5.4.4) is
   applied to that reduced record;
3. the three-point stack algorithm counts full cycles and residual half
   cycles; zero-range entries are skipped;
4. two-sample and monotonic records give one residual half cycle.

Damage uses a stress-range S-N curve in MPa with an optional second branch,
knee and threshold, exactly as :meth:`SNcurve.cycles_to_failure`.

The pure-Python path remains the reference implementation; regression tests
compare the two. Without numba (or with ``CABLEXSEC_DISABLE_NUMBA=1``) the same
kernels run as interpreted Python, giving the same results much more slowly;
:func:`cablexsec.analysis.fatigue.compute_fatigue` then uses the reference
counter instead (see :mod:`cablexsec._numba`).

:func:`miner_rows` applies the same counter to the bulk Miner damage of
:func:`cablexsec.analysis.fatigue.miner_damage`, including a constant
stress-range factor, and flags any row for which the reference would raise
(non-finite or non-positive amplified range, S-N life outside floating-point
range) so that the caller can rerun it on the reference path.

:func:`fibre_miner_damage` gives the same sums at every fixed fibre of a
(wire, time) stress field without forming the fibre histories: each history
is reduced to its turning points while it is formed from the field
components, and one count serves every stress-range factor.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from cablexsec._numba import HAVE_NUMBA, njit, prange

if TYPE_CHECKING:
    from cablexsec.materials.sn_curves import SNcurve

__all__ = [
    "HAVE_NUMBA",
    "fatigue_rows",
    "fibre_fatigue",
    "fibre_miner_damage",
    "miner_rows",
    "rainflow_table",
]

FloatArray = NDArray[np.float64]
CurveParameters = tuple[float, float, float, float, float, float]


@njit(cache=True)
def _cycles_to_failure(
    stress: float,
    m1: float,
    log_c1: float,
    m2: float,
    log_c2: float,
    knee: float,
    threshold: float,
) -> float:
    """S-N life; ``m2 <= 0``: single slope; ``knee``/``threshold`` < 0: none."""
    if stress <= 0.0:
        return math.inf
    if threshold >= 0.0 and stress <= threshold:
        return math.inf
    m, log_c = m1, log_c1
    if m2 > 0.0 and stress < knee:
        m, log_c = m2, log_c2
    life: float = 10.0 ** (log_c - m * math.log10(stress))
    return life


@njit(cache=True)
def _add_cycle(
    span: float,
    count: float,
    acc: FloatArray,
    factor: float,
    m1: float,
    log_c1: float,
    m2: float,
    log_c2: float,
    knee: float,
    threshold: float,
) -> None:
    """Accumulate damage ``acc[0]`` and power sum ``acc[1]``; flag ``acc[2]``.

    The range is scaled as ``span / 1e6 * factor`` (bitwise ``span / 1e6`` for
    ``factor == 1``). ``acc[2]`` is set where the reference Miner sum raises:
    a non-finite or non-positive scaled range, or a life (above any
    threshold) that is zero or infinite through floating-point range.
    """
    stress = span / 1.0e6 * factor
    life = _cycles_to_failure(stress, m1, log_c1, m2, log_c2, knee, threshold)
    thresholded = threshold >= 0.0 and stress <= threshold
    if not (0.0 < stress < math.inf) or not (thresholded or 0.0 < life < math.inf):
        acc[2] = 1.0
    acc[0] += count / life
    acc[1] += count * span**m1


@njit(cache=True)
def _fatigue_one(
    s: FloatArray,
    m1: float,
    log_c1: float,
    m2: float,
    log_c2: float,
    knee: float,
    threshold: float,
) -> tuple[float, float]:
    """Return (damage, power sum) for one history in Pa."""
    acc = np.zeros(3)
    _count_one(s, 1.0, acc, m1, log_c1, m2, log_c2, knee, threshold)
    return acc[0], acc[1]


@njit(cache=True)
def _count_one(
    s: FloatArray,
    factor: float,
    acc: FloatArray,
    m1: float,
    log_c1: float,
    m2: float,
    log_c2: float,
    knee: float,
    threshold: float,
) -> None:
    """Rainflow-count one history in Pa into ``acc`` (see :func:`_add_cycle`)."""
    n = s.size
    if n < 2:
        return
    if n == 2:
        span = abs(s[1] - s[0])
        if span > 0:
            _add_cycle(span, 0.5, acc, factor, m1, log_c1, m2, log_c2, knee, threshold)
        return
    # 1. plateau-preserving turning points (indices of the reduced record)
    red = np.empty(n)
    k = 0
    red[k] = s[0]
    k += 1
    prev_rising = False
    any_change = False
    for i in range(n - 1):
        if s[i + 1] != s[i]:
            rising = s[i + 1] > s[i]
            if any_change and rising != prev_rising:
                red[k] = s[i]
                k += 1
            prev_rising = rising
            any_change = True
    if not any_change:
        return
    red[k] = s[n - 1]
    k += 1
    if k == 2:
        # Monotonic record: one residual half cycle (a zero range adds nothing).
        span = abs(red[1] - red[0])
        if span > 0:
            _add_cycle(span, 0.5, acc, factor, m1, log_c1, m2, log_c2, knee, threshold)
        return
    # 2. reversal scan of the rainflow package on the reduced record
    rev = np.empty(k)
    r = 0
    x_last = red[0]
    x = red[1]
    d_last = x - x_last
    rev[r] = x_last
    r += 1
    x_next = x
    for j in range(2, k):
        x_next = red[j]
        if x_next == x:
            continue
        d_next = x_next - x
        if d_last * d_next < 0:
            rev[r] = x
            r += 1
        x_last = x
        x = x_next
        d_last = d_next
    rev[r] = x_next
    r += 1
    # 3. three-point stack (deque emulated with a head index)
    stack = np.empty(r)
    head = 0
    top = 0
    for j in range(r):
        stack[top] = rev[j]
        top += 1
        while top - head >= 3:
            x1 = stack[top - 3]
            x2 = stack[top - 2]
            x3 = stack[top - 1]
            big_x = abs(x3 - x2)
            big_y = abs(x2 - x1)
            if big_x < big_y:
                break
            elif top - head == 3:
                span = abs(stack[head] - stack[head + 1])
                if span > 0:
                    _add_cycle(
                        span, 0.5, acc, factor, m1, log_c1, m2, log_c2, knee, threshold
                    )
                head += 1
            else:
                span = abs(x1 - x2)
                if span > 0:
                    _add_cycle(
                        span, 1.0, acc, factor, m1, log_c1, m2, log_c2, knee, threshold
                    )
                last = stack[top - 1]
                top -= 3
                stack[top] = last
                top += 1
    while top - head > 1:
        span = abs(stack[head] - stack[head + 1])
        if span > 0:
            _add_cycle(span, 0.5, acc, factor, m1, log_c1, m2, log_c2, knee, threshold)
        head += 1


@njit(cache=True, parallel=True)
def _fatigue_rows_parallel(
    stress: FloatArray,
    m1: float,
    log_c1: float,
    m2: float,
    log_c2: float,
    knee: float,
    threshold: float,
) -> tuple[FloatArray, FloatArray]:
    rows = stress.shape[0]
    damage = np.empty(rows)
    power = np.empty(rows)
    for i in prange(rows):
        damage[i], power[i] = _fatigue_one(
            stress[i], m1, log_c1, m2, log_c2, knee, threshold
        )
    return damage, power


@njit(cache=True)
def _fatigue_rows_serial(
    stress: FloatArray,
    m1: float,
    log_c1: float,
    m2: float,
    log_c2: float,
    knee: float,
    threshold: float,
) -> tuple[FloatArray, FloatArray]:
    rows = stress.shape[0]
    damage = np.empty(rows)
    power = np.empty(rows)
    for i in range(rows):
        damage[i], power[i] = _fatigue_one(
            stress[i], m1, log_c1, m2, log_c2, knee, threshold
        )
    return damage, power


@njit(cache=True)
def _fibre_one_wire(
    w: int,
    mean: FloatArray,
    radial: FloatArray,
    tangential: FloatArray,
    cosines: FloatArray,
    sines: FloatArray,
    out_d: FloatArray,
    out_p: FloatArray,
    out_lo: FloatArray,
    out_hi: FloatArray,
    m1: float,
    log_c1: float,
    m2: float,
    log_c2: float,
    knee: float,
    threshold: float,
) -> None:
    n_t = mean.shape[1]
    buf = np.empty(n_t)
    for f in range(cosines.size):
        c = cosines[f]
        s = sines[f]
        lo = math.inf
        hi = -math.inf
        for i in range(n_t):
            v = mean[w, i] + radial[w, i] * c + tangential[w, i] * s
            buf[i] = v
            if v < lo:
                lo = v
            if v > hi:
                hi = v
        d, p = _fatigue_one(buf, m1, log_c1, m2, log_c2, knee, threshold)
        out_d[w, f] = d
        out_p[w, f] = p
        out_lo[w, f] = lo
        out_hi[w, f] = hi


@njit(cache=True)
def _fibres_serial(
    mean: FloatArray,
    radial: FloatArray,
    tangential: FloatArray,
    cosines: FloatArray,
    sines: FloatArray,
    m1: float,
    log_c1: float,
    m2: float,
    log_c2: float,
    knee: float,
    threshold: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    n_w = mean.shape[0]
    n_f = cosines.size
    out_d = np.empty((n_w, n_f))
    out_p = np.empty((n_w, n_f))
    out_lo = np.empty((n_w, n_f))
    out_hi = np.empty((n_w, n_f))
    for w in range(n_w):
        _fibre_one_wire(
            w,
            mean,
            radial,
            tangential,
            cosines,
            sines,
            out_d,
            out_p,
            out_lo,
            out_hi,
            m1,
            log_c1,
            m2,
            log_c2,
            knee,
            threshold,
        )
    return out_d, out_p, out_lo, out_hi


@njit(cache=True, parallel=True)
def _fibres_parallel(
    mean: FloatArray,
    radial: FloatArray,
    tangential: FloatArray,
    cosines: FloatArray,
    sines: FloatArray,
    m1: float,
    log_c1: float,
    m2: float,
    log_c2: float,
    knee: float,
    threshold: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    n_w = mean.shape[0]
    n_f = cosines.size
    out_d = np.empty((n_w, n_f))
    out_p = np.empty((n_w, n_f))
    out_lo = np.empty((n_w, n_f))
    out_hi = np.empty((n_w, n_f))
    for w in prange(n_w):
        _fibre_one_wire(
            w,
            mean,
            radial,
            tangential,
            cosines,
            sines,
            out_d,
            out_p,
            out_lo,
            out_hi,
            m1,
            log_c1,
            m2,
            log_c2,
            knee,
            threshold,
        )
    return out_d, out_p, out_lo, out_hi


def fibre_fatigue(
    mean_Pa: FloatArray,
    radial_Pa: FloatArray,
    tangential_Pa: FloatArray,
    angles_rad: FloatArray,
    sn_curve: SNcurve,
    duration: float,
    *,
    parallel: bool = False,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Damage, 1 Hz equivalent range [MPa] and stress extrema [MPa] at fixed fibres.

    The fibre stress is ``mean + radial cos(angle) + tangential sin(angle)``
    (as :meth:`WireStressField.at_fibre`) for every wire (rows) and fibre
    angle; histories are counted in one pass without allocating them.
    """
    arrays = [
        np.ascontiguousarray(a, dtype=np.float64)
        for a in (mean_Pa, radial_Pa, tangential_Pa)
    ]
    if arrays[0].ndim != 2 or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("matching two-dimensional stress components required")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("positive finite duration required")
    angles = np.asarray(angles_rad, dtype=np.float64)
    cosines = np.array([math.cos(a) for a in angles])
    sines = np.array([math.sin(a) for a in angles])
    params = _curve_parameters(sn_curve)
    kernel = _fibres_parallel if parallel else _fibres_serial
    mean, radial, tangential = arrays
    damage, power, lo, hi = kernel(mean, radial, tangential, cosines, sines, *params)
    equivalent = (power / duration) ** (1.0 / params[0]) / 1.0e6
    return damage, equivalent, lo / 1.0e6, hi / 1.0e6


def _curve_parameters(sn_curve: SNcurve) -> CurveParameters:
    if sn_curve.stress_unit != "MPa":
        raise ValueError("a stress-range S-N curve in MPa is required")
    bilinear = sn_curve.m2 is not None and sn_curve.log_C2 is not None
    m2 = sn_curve.m2 if bilinear and sn_curve.m2 is not None else -1.0
    log_c2 = sn_curve.log_C2 if bilinear and sn_curve.log_C2 is not None else 0.0
    transition = sn_curve.transition_stress()  # None exactly when single-slope
    knee = -1.0 if transition is None else transition
    threshold = -1.0 if sn_curve.threshold is None else sn_curve.threshold
    return (
        float(sn_curve.m1),
        float(sn_curve.log_C1),
        float(m2),
        float(log_c2),
        float(knee),
        float(threshold),
    )


def fatigue_rows(
    stress_Pa: FloatArray,
    sn_curve: SNcurve,
    duration: float,
    *,
    parallel: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Miner damage and 1 Hz equivalent range [MPa] for each row of ``stress_Pa``.

    Equivalent to calling ``fatigue_pair`` on every row. ``parallel=True``
    distributes rows over threads; the per-row arithmetic is unchanged.
    """
    stress = np.ascontiguousarray(stress_Pa, dtype=np.float64)
    if stress.ndim != 2 or stress.shape[1] < 2 or not np.isfinite(stress).all():
        raise ValueError("finite two-dimensional stress histories required")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("positive finite duration required")
    params = _curve_parameters(sn_curve)
    kernel = _fatigue_rows_parallel if parallel else _fatigue_rows_serial
    damage, power = kernel(stress, *params)
    return damage, (power / duration) ** (1.0 / params[0]) / 1.0e6


@njit(cache=True)
def _miner_rows_serial(
    stress: FloatArray,
    factor: float,
    m1: float,
    log_c1: float,
    m2: float,
    log_c2: float,
    knee: float,
    threshold: float,
) -> tuple[FloatArray, FloatArray]:
    rows = stress.shape[0]
    damage = np.empty(rows)
    flags = np.empty(rows)
    acc = np.zeros(3)
    for i in range(rows):
        acc[0] = 0.0
        acc[1] = 0.0
        acc[2] = 0.0
        _count_one(stress[i], factor, acc, m1, log_c1, m2, log_c2, knee, threshold)
        damage[i] = acc[0]
        flags[i] = acc[2]
    return damage, flags


def miner_rows(
    stress_Pa: FloatArray,
    sn_curve: SNcurve,
    stress_range_factor: float = 1.0,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Miner damage of each row and a flag where the reference sum would raise.

    Counts every row as :func:`cablexsec.analysis.fatigue.miner_damage` does,
    cycle for cycle and in the same summation order, with ranges scaled as
    ``range / 1e6 * stress_range_factor``. Flagged rows (and rows with a
    non-finite total) must be recomputed on the reference path, which raises
    the documented input error; their damage values here are meaningless.
    Inputs are assumed validated (finite, two-dimensional, curve in MPa).
    """
    stress = np.ascontiguousarray(stress_Pa, dtype=np.float64)
    params = _curve_parameters(sn_curve)
    try:
        damage, flags = _miner_rows_serial(stress, float(stress_range_factor), *params)
    except (ZeroDivisionError, OverflowError):
        # Interpreted kernels raise where compiled arithmetic gives 0 or inf;
        # either way the reference path decides. Flag every row.
        rows = stress.shape[0]
        return np.zeros(rows), np.ones(rows, dtype=np.bool_)
    return damage, (flags != 0.0) | ~np.isfinite(damage)


# --------------------------------------------------------------------------- #
# Fused fixed-fibre Miner damage (compute_fatigue)                             #
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _cycle_damage(
    span: float,
    factor: float,
    m1: float,
    log_c1: float,
    m2: float,
    log_c2: float,
    knee: float,
    threshold: float,
) -> tuple[float, bool]:
    """``(1 / life, flag)`` of one counted range, as :func:`_add_cycle`.

    The range is scaled as ``span / 1e6 * factor``; the flag marks where the
    reference Miner sum raises (see :func:`_add_cycle`).
    """
    stress = span / 1.0e6 * factor
    life = _cycles_to_failure(stress, m1, log_c1, m2, log_c2, knee, threshold)
    thresholded = threshold >= 0.0 and stress <= threshold
    flag = not (0.0 < stress < math.inf) or not (thresholded or 0.0 < life < math.inf)
    return life, flag


@njit(cache=True)
def _count_turning_points(
    red: FloatArray,
    k: int,
    stack: FloatArray,
    factor0: float,
    factor1: float,
    two: bool,
    out: FloatArray,
    m1: float,
    log_c1: float,
    m2: float,
    log_c2: float,
    knee: float,
    threshold: float,
) -> None:
    """Steps 2 and 3 of :func:`_count_one` on the turning points ``red[:k]``.

    The reversal scan feeds the three-point stack directly, point by point,
    and the Miner sums of one or two range factors are accumulated in cycle
    order: ``out = (damage0, damage1, flag0, flag1)``. ``k >= 2``.
    """
    d0 = 0.0
    d1 = 0.0
    g0 = False
    g1 = False
    head = 0
    top = 0
    x = red[1]
    d_last = x - red[0]
    x_next = x
    for j in range(1, k + 1):
        # The reversal scan: red[0], the reversals of red[1:k-1], red[k-1].
        if j == 1:
            value = red[0]
        elif j < k:
            x_next = red[j]
            if x_next == x:
                continue
            d_next = x_next - x
            reversal = d_last * d_next < 0
            value = x
            x = x_next
            d_last = d_next
            if not reversal:
                continue
        else:
            value = x_next
        stack[top] = value
        top += 1
        while top - head >= 3:
            x1 = stack[top - 3]
            x2 = stack[top - 2]
            x3 = stack[top - 1]
            if abs(x3 - x2) < abs(x2 - x1):
                break
            if top - head == 3:
                span = abs(stack[head] - stack[head + 1])
                count = 0.5
                head += 1
            else:
                span = abs(x1 - x2)
                count = 1.0
                top -= 3
                stack[top] = x3
                top += 1
            if span > 0:
                life, flag = _cycle_damage(
                    span, factor0, m1, log_c1, m2, log_c2, knee, threshold
                )
                g0 = g0 or flag
                d0 += count / life
                if two:
                    life, flag = _cycle_damage(
                        span, factor1, m1, log_c1, m2, log_c2, knee, threshold
                    )
                    g1 = g1 or flag
                    d1 += count / life
    while top - head > 1:
        span = abs(stack[head] - stack[head + 1])
        if span > 0:
            life, flag = _cycle_damage(
                span, factor0, m1, log_c1, m2, log_c2, knee, threshold
            )
            g0 = g0 or flag
            d0 += 0.5 / life
            if two:
                life, flag = _cycle_damage(
                    span, factor1, m1, log_c1, m2, log_c2, knee, threshold
                )
                g1 = g1 or flag
                d1 += 0.5 / life
        head += 1
    out[0] = d0
    out[1] = d1
    out[2] = 1.0 if g0 else 0.0
    out[3] = 1.0 if g1 else 0.0


@njit(cache=True)
def _fibre_miner_kernel(
    mean: FloatArray,
    radial: FloatArray,
    tangential: FloatArray,
    normal_offsets: FloatArray,
    lateral_offsets: FloatArray,
    start: int,
    factor0: float,
    factor1: float,
    two: bool,
    m1: float,
    log_c1: float,
    m2: float,
    log_c2: float,
    knee: float,
    threshold: float,
) -> tuple[FloatArray, FloatArray]:
    """Miner damage of every (factor, wire, fibre) without forming histories.

    The fibre stress ``(mean + radial n) + tangential l`` (the arithmetic of
    :meth:`WireStressField.at_fibre`) of samples ``start:`` is reduced to its
    plateau-preserving turning points while it is formed (step 1 of
    :func:`_count_one`, without branches), then counted by
    :func:`_count_turning_points`. ``n_t - start >= 2``.
    """
    n_w, n_t = mean.shape
    n_f = normal_offsets.size
    damage = np.zeros((2, n_w, n_f))
    flags = np.zeros((2, n_w, n_f))
    red = np.empty(n_t - start + 1)
    stack = np.empty(n_t - start + 1)
    out = np.zeros(4)
    for w in range(n_w):
        mw = mean[w]
        rw = radial[w]
        tw = tangential[w]
        # Largest component magnitudes (NaN gives inf): a fibre whose bound is
        # not finite may hold a non-finite sample and is left to the reference.
        big_m = 0.0
        big_r = 0.0
        big_t = 0.0
        for i in range(start, n_t):
            a = abs(mw[i])
            if not a <= big_m:
                big_m = a if a == a else math.inf
            a = abs(rw[i])
            if not a <= big_r:
                big_r = a if a == a else math.inf
            a = abs(tw[i])
            if not a <= big_t:
                big_t = a if a == a else math.inf
        for f in range(n_f):
            c = normal_offsets[f]
            sl = lateral_offsets[f]
            unsafe = not big_m + big_r * abs(c) + big_t * abs(sl) < 1.0e300
            prev = mw[start] + rw[start] * c + tw[start] * sl
            red[0] = prev
            k = np.int64(1)
            rising_before = np.int64(0)
            changed = np.int64(0)
            for i in range(start + 1, n_t):
                x = mw[i] + rw[i] * c + tw[i] * sl
                moved = np.int64(x != prev)
                rising = np.int64(x > prev)
                # A turning point at the previous sample: the record moved,
                # it had moved before, and the direction changed.
                red[k] = prev
                k += moved & changed & (rising ^ rising_before)
                rising_before = (rising_before & (1 - moved)) | (rising & moved)
                changed |= moved
                prev = x
            red[k] = prev
            k += 1
            # A record that never moves gives one zero range, which is skipped.
            _count_turning_points(
                red,
                int(k),
                stack,
                factor0,
                factor1,
                two,
                out,
                m1,
                log_c1,
                m2,
                log_c2,
                knee,
                threshold,
            )
            damage[0, w, f] = out[0]
            damage[1, w, f] = out[1]
            flags[0, w, f] = 1.0 if unsafe or out[2] != 0.0 else 0.0
            flags[1, w, f] = 1.0 if unsafe or out[3] != 0.0 else 0.0
    return damage, flags


def fibre_miner_damage(
    mean_Pa: FloatArray,
    radial_Pa: FloatArray,
    tangential_Pa: FloatArray,
    normal_offsets: FloatArray,
    lateral_offsets: FloatArray,
    sn_curve: SNcurve,
    stress_range_factors: Sequence[float],
    *,
    start: int = 0,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Miner damage at fixed fibres, shape (factor, wire, fibre), and rerun flags.

    The history of fibre ``f`` of wire ``w`` is
    ``(mean + radial * normal_offsets[f]) + tangential * lateral_offsets[f]``
    over samples ``start:``, the arithmetic of
    :meth:`cablexsec.analysis.fatigue.WireStressField.at_fibre`; it is counted
    as :func:`miner_rows` counts that history with each of the one or two
    stress-range factors, cycle for cycle and in the same summation order,
    without forming it, and one count serves both factors. Flagged entries
    (and entries with a non-finite total) must be recomputed on the
    reference path, as for :func:`miner_rows`. Inputs are assumed validated
    (matching two-dimensional components, curve in MPa, finite positive
    factors).
    """
    arrays = [
        np.ascontiguousarray(a, dtype=np.float64)
        for a in (mean_Pa, radial_Pa, tangential_Pa)
    ]
    if arrays[0].ndim != 2 or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("matching two-dimensional stress components required")
    normal = np.ascontiguousarray(normal_offsets, dtype=np.float64)
    lateral = np.ascontiguousarray(lateral_offsets, dtype=np.float64)
    if normal.ndim != 1 or lateral.shape != normal.shape:
        raise ValueError("one normal and one lateral offset per fibre required")
    factors = [float(value) for value in stress_range_factors]
    if len(factors) not in (1, 2):
        raise ValueError("one or two stress-range factors required")
    if not 0 <= start <= arrays[0].shape[1] - 2:
        raise ValueError("at least two samples from start are required")
    params = _curve_parameters(sn_curve)
    mean, radial, tangential = arrays
    two = len(factors) == 2
    rows = slice(0, 2 if two else 1)
    try:
        damage, flags = _fibre_miner_kernel(
            mean,
            radial,
            tangential,
            normal,
            lateral,
            int(start),
            factors[0],
            factors[-1],
            two,
            *params,
        )
    except (ZeroDivisionError, OverflowError):
        # Interpreted kernels raise where compiled arithmetic gives 0 or inf.
        shape = (len(factors), mean.shape[0], normal.size)
        return np.zeros(shape), np.ones(shape, dtype=np.bool_)
    damage = damage[rows]
    return damage, (flags[rows] != 0.0) | ~np.isfinite(damage)


# --------------------------------------------------------------------------- #
# Cycle table with sample indices (fretting regime assessment)                 #
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _rainflow_table(
    s: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.int64], NDArray[np.int64]]:
    """Cycles of :func:`cablexsec.analysis.fatigue.rainflow_cycles` as arrays.

    Returns ``(range, mean, count, start, end)`` in the order and with the
    arithmetic of the reference generator (same reduction, reversal scan and
    three-point stack), with ``start``/``end`` indices into ``s``.
    """
    n = s.size
    cap = 2 * n + 2
    rng = np.empty(cap)
    mean = np.empty(cap)
    count = np.empty(cap)
    start = np.empty(cap, dtype=np.int64)
    end = np.empty(cap, dtype=np.int64)
    m = 0
    if n < 2:
        return rng[:0], mean[:0], count[:0], start[:0], end[:0]
    if n == 2:
        span = abs(s[1] - s[0])
        if span > 0:
            rng[0] = span
            mean[0] = 0.5 * s[0] + 0.5 * s[1]
            count[0] = 0.5
            start[0] = 0
            end[0] = 1
            m = 1
        return rng[:m], mean[:m], count[:m], start[:m], end[:m]
    # 1. plateau-preserving turning points (last index of each turning plateau)
    idx = np.empty(n, dtype=np.int64)
    k = 0
    idx[k] = 0
    k += 1
    prev_rising = False
    any_change = False
    for i in range(n - 1):
        if s[i + 1] != s[i]:
            rising = s[i + 1] > s[i]
            if any_change and rising != prev_rising:
                idx[k] = i
                k += 1
            prev_rising = rising
            any_change = True
    if not any_change:
        return rng[:0], mean[:0], count[:0], start[:0], end[:0]
    idx[k] = n - 1
    k += 1
    if k == 2:
        first = s[0]
        last = s[n - 1]
        rng[0] = abs(last - first)
        mean[0] = 0.5 * (first + last)
        count[0] = 0.5
        start[0] = 0
        end[0] = n - 1
        return rng[:1], mean[:1], count[:1], start[:1], end[:1]
    # 2. reversal scan of the rainflow package on the reduced record
    rev_i = np.empty(k, dtype=np.int64)
    rev_x = np.empty(k)
    r = 0
    x_last = s[idx[0]]
    x = s[idx[1]]
    d_last = x - x_last
    rev_i[r] = 0
    rev_x[r] = x_last
    r += 1
    x_next = x
    j_last = 1
    for j in range(2, k):
        x_next = s[idx[j]]
        j_last = j
        if x_next == x:
            continue
        d_next = x_next - x
        if d_last * d_next < 0:
            rev_i[r] = j - 1
            rev_x[r] = x
            r += 1
        x_last = x
        x = x_next
        d_last = d_next
    rev_i[r] = j_last
    rev_x[r] = x_next
    r += 1
    # 3. three-point stack (deque emulated with a head index)
    st_i = np.empty(r, dtype=np.int64)
    st_x = np.empty(r)
    head = 0
    top = 0
    for j in range(r):
        st_i[top] = rev_i[j]
        st_x[top] = rev_x[j]
        top += 1
        while top - head >= 3:
            x1 = st_x[top - 3]
            x2 = st_x[top - 2]
            x3 = st_x[top - 1]
            if abs(x3 - x2) < abs(x2 - x1):
                break
            elif top - head == 3:
                span = abs(st_x[head] - st_x[head + 1])
                if span > 0:
                    rng[m] = span
                    mean[m] = 0.5 * (st_x[head] + st_x[head + 1])
                    count[m] = 0.5
                    start[m] = idx[st_i[head]]
                    end[m] = idx[st_i[head + 1]]
                    m += 1
                head += 1
            else:
                span = abs(x1 - x2)
                if span > 0:
                    rng[m] = span
                    mean[m] = 0.5 * (x1 + x2)
                    count[m] = 1.0
                    start[m] = idx[st_i[top - 3]]
                    end[m] = idx[st_i[top - 2]]
                    m += 1
                last_i = st_i[top - 1]
                last_x = st_x[top - 1]
                top -= 3
                st_i[top] = last_i
                st_x[top] = last_x
                top += 1
    while top - head > 1:
        span = abs(st_x[head] - st_x[head + 1])
        if span > 0:
            rng[m] = span
            mean[m] = 0.5 * (st_x[head] + st_x[head + 1])
            count[m] = 0.5
            start[m] = idx[st_i[head]]
            end[m] = idx[st_i[head + 1]]
            m += 1
        head += 1
    return rng[:m], mean[:m], count[:m], start[:m], end[:m]


def rainflow_table(
    history: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.int64], NDArray[np.int64]]:
    """Rainflow cycles ``(range, mean, count, start, end)`` of one history.

    Compiled equivalent of
    :func:`cablexsec.analysis.fatigue.rainflow_cycles`, cycle for cycle and in
    the same order (``tests/test_fretting_regime.py``). ``start``/``end`` are
    the sample indices of the two turning points of each cycle.
    """
    s = np.ascontiguousarray(history, dtype=np.float64)
    if s.ndim != 1 or not np.isfinite(s).all():
        raise ValueError("a finite one-dimensional history is required")
    return _rainflow_table(s)
