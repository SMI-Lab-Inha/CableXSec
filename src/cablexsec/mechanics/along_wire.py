# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Slip redistribution along helical wires through a non-uniform curvature field.

The sectional friction laws of the package treat every cable section as if
the curvature were uniform over the whole length of every wire: the
direction-tied cap (:mod:`~cablexsec.mechanics.fast_slip`) pointwise, the
distributed helix model (:mod:`~cablexsec.mechanics.distributed_friction`)
periodically over one pitch. The clamp-layer solver
(:mod:`~cablexsec.mechanics.clamp_layer`) adds one anchor under a
proportionally scaled curvature shape with a rigid interface. This module
removes those restrictions. Each wire is followed along its own helix through
the stations of a specimen, a test rig or a termination, at their own and
time-varying curvature, between ends that are anchored or free.

Model (one wire, axial slip only; normalised as the distributed model)
------------------------------------------------------------------------
With ``phi`` the helix angle along the wire (``2 pi`` per pitch ``P``),
``x = x_0 + phi P / (2 pi)`` the axial position and
``theta(x) = theta_0 + s 2 pi x / P`` the azimuth of the wire (``s = +-1``,
the sense in which the wire turns along ``+x``)::

    n = e + du/dphi,      dn/dphi = t,      t = k_hat (u - p),     |t| <= b,
    e(x, t) = (kappa_x(x, t) cos theta(x) + kappa_y(x, t) sin theta(x)) / kappa_c,
    b = rho(t) g(theta) + lam |n|,          dp/dt != 0 only where |t| = b,

``n`` the wire force above the pre-tension share in units of ``g_0``
(stress ``sigma = sigma_u n``), ``u`` the axial displacement of the wire
relative to the core surface, ``p`` its sliding part, ``k_hat`` the
normalised interface shear stiffness, ``lam`` the capstan coefficient and
``g`` the azimuthal contact shape. Ends: anchored (``u = 0``: end fitting,
grip, clamped stiffener root; the reaction is free) or free (``n = 0``).

Stick stiffness (explicit, :data:`STICK_CONVENTIONS`)
-----------------------------------------------------
The package's sectional laws reduce the stick stress by the Yun coefficient
``xi`` (``k_f = xi E cos^2 a``). Along a wire this reduction can be
represented in three ways, which agree in the far field below slip and differ
near anchors and gradients and in the slip demand:

* ``"spring"`` (default; the distributed model): an elastic interface
  ``k_hat = xi / (1 - xi)`` with the full imposed strain
  (``kappa_c = xi kappa_on``, the rigid-stick onset curvature). The uniform
  far field sticks at ``n = xi a cos theta`` (the package's stick stress);
  near an anchor the shear lag restores the full strain over
  ``1 / sqrt(k_hat)`` radians (Lutchansky 1969); in slip the demand is the
  full geometric mismatch.
* ``"rigid"``: ``xi = 1``. Rigid interface (``k_hat`` = :data:`RIGID_K_HAT`),
  full strain; the far field is the sectional law with ``xi = 1``.
* ``"scaled"``: rigid interface with the imposed strain scaled by ``xi``
  (``kappa_c = kappa_on``). This is the convention of
  :mod:`~cablexsec.analysis.termination` (its clamp increment) and of the
  direction-tied cap: the far field equals the package law, but the slip
  demand near an anchor is reduced by ``xi`` as well.

For ``xi`` near 0.2 (the armour of the packaged power cables) the slip
demand, and with it the anchored-wire amplification, differs by about
``1 / xi`` between ``"scaled"`` and the two others.

Discretisation
--------------
``M`` segments of ``h`` radians (``segments_per_pitch`` per pitch) carry the
forces ``n_j`` at their midpoints; ``M + 1`` nodes carry ``u``, ``p`` and the
traction ``t`` over a tributary length ``w = h`` (``h / 2`` at the ends).
Each increment is backward Euler: semismooth Newton on the active set
(tridiagonal, Thomas) with Armijo backtracking on the convex incremental
potential ``sum h n^2 / 2 + sum w H_b(u - p)`` (``H_b`` the Huber envelope of
the dissipation) for a fixed bound, and a fixed point on the capstan bound
(the Picard iteration of a Volterra problem along each slip zone). Anchored
end nodes are fixed; a chain with two free ends in full slip takes the node
nearest to sticking as a stick row (the tractions must balance), as the
periodic model does. Energy (normalised, per wire):
``W_be = sum h n de`` equals ``Delta U + D + R`` to round-off every
increment, with ``U = sum h n^2 / 2 + sum w t^2 / (2 k_hat)``, ``D`` the
non-negative friction dissipation and ``R`` the backward-Euler remainder.
One normalised energy unit is ``g_0^2 R / (E A sin a)`` joules per wire.

Limits
------
Axial slip only (no transverse slip, wire bending or torsion), one wire on a
rigid core (no interaction between layers), quasi-static, contact pressure
prescribed (scaled in time by ``rho``; no bending-induced redistribution),
Bernoulli kinematics of the core (``e`` from the local curvature only).
:data:`RIGID_K_HAT` stands for the rigid interface (Lutchansky length
``1e-2`` rad, far-field stick factor ``0.9999``). The capstan term uses
``|n|`` like the package cap (a compressive wire is not unloaded). The
increment must be small where slip directions reverse (near anchors under
proportional loading): ``max_increment`` bounds it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from cablexsec._numba import HAVE_NUMBA, njit
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.clamp_layer import _lower_bound, _upper_bound
from cablexsec.mechanics.tendon_bending import LateralRelief

if TYPE_CHECKING:
    from cablexsec.mechanics.mkappa import HelicalLayerSpec
    from cablexsec.mechanics.strand_slip import ConductorRing, StrandSlipParameters

__all__ = [
    "END_CONDITIONS",
    "HAVE_NUMBA",
    "RIGID_K_HAT",
    "STICK_CONVENTIONS",
    "AlongWireHistory",
    "WireChain",
    "WireChainLayer",
    "along_wire_history",
    "chain_layer_jacobian",
    "chain_layer_step",
]

FloatArray = NDArray[np.float64]

#: Stick-stiffness conventions (module notes).
STICK_CONVENTIONS = ("spring", "rigid", "scaled")
#: Wire end conditions.
END_CONDITIONS = ("anchored", "free")
#: Normalised interface stiffness that stands for a rigid interface.
RIGID_K_HAT = 1.0e4

_MAX_NEWTON = 400
_MAX_FIXED_POINT = 400
_MAX_HALVINGS = 10
_BIG = 1.0e30


# --------------------------------------------------------------------------- #
# Compiled kernel                                                              #
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _thomas(
    lo: FloatArray,
    d: FloatArray,
    up: FloatArray,
    rhs: FloatArray,
    x: FloatArray,
    cp: FloatArray,
    dp: FloatArray,
    m: int,
) -> bool:
    """Tridiagonal solve, row ``i``: ``lo_i x_{i-1} + d_i x_i + up_i x_{i+1}``."""
    bet = d[0]
    if bet == 0.0 or not math.isfinite(bet):
        return False
    cp[0] = up[0] / bet
    dp[0] = rhs[0] / bet
    for i in range(1, m):
        bet = d[i] - lo[i] * cp[i - 1]
        if bet == 0.0 or not math.isfinite(bet):
            return False
        cp[i] = up[i] / bet
        dp[i] = (rhs[i] - lo[i] * dp[i - 1]) / bet
    x[m - 1] = dp[m - 1]
    for i in range(m - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return True


@njit(cache=True)
def _clip(value: float, bound: float) -> float:
    if value > bound:
        return bound
    if value < -bound:
        return -bound
    return value


@njit(cache=True)
def _node_force_pair(
    y: FloatArray, v: FloatArray, h: float, m: int, i: int
) -> tuple[float, float]:
    """Forces of the segments right and left of node ``i`` (ghosts zero)."""
    right = y[i] + (v[i + 1] - v[i]) / h if i < m else 0.0
    left = y[i - 1] + (v[i] - v[i - 1]) / h if i > 0 else 0.0
    return right, left


@njit(cache=True)
def _residual(
    y: FloatArray,
    z: FloatArray,
    v: FloatArray,
    b: FloatArray,
    k: float,
    h: float,
    w: FloatArray,
    fixed: NDArray[np.bool_],
    m: int,
) -> float:
    """Largest nodal equilibrium residual ``|n_i - n_{i-1} - w_i t_i|``."""
    worst = 0.0
    for i in range(m + 1):
        if fixed[i]:
            continue
        right, left = _node_force_pair(y, v, h, m, i)
        t = _clip(k * (z[i] + v[i]), b[i])
        r = abs(right - left - w[i] * t)
        if r > worst:
            worst = r
    return worst


@njit(cache=True)
def _objective(
    y: FloatArray,
    z: FloatArray,
    v: FloatArray,
    b: FloatArray,
    k: float,
    h: float,
    w: FloatArray,
    fixed: NDArray[np.bool_],
    m: int,
) -> float:
    f = 0.0
    for j in range(m):
        nj = y[j] + (v[j + 1] - v[j]) / h
        f += 0.5 * h * nj * nj
    for i in range(m + 1):
        if fixed[i]:
            continue
        r = z[i] + v[i]
        if abs(k * r) <= b[i]:
            f += 0.5 * w[i] * k * r * r
        else:
            f += w[i] * (b[i] * abs(r) - b[i] * b[i] / (2.0 * k))
    return f


@njit(cache=True)
def _directional(
    y: FloatArray,
    z: FloatArray,
    v: FloatArray,
    dv: FloatArray,
    b: FloatArray,
    k: float,
    h: float,
    w: FloatArray,
    fixed: NDArray[np.bool_],
    m: int,
) -> float:
    s = 0.0
    for i in range(m + 1):
        if fixed[i]:
            continue
        right, left = _node_force_pair(y, v, h, m, i)
        t = _clip(k * (z[i] + v[i]), b[i])
        s += (left - right + w[i] * t) * dv[i]
    return s


@njit(cache=True)
def _same_active_set(
    z: FloatArray,
    v: FloatArray,
    vn: FloatArray,
    b: FloatArray,
    k: float,
    fixed: NDArray[np.bool_],
    m: int,
) -> bool:
    for i in range(m + 1):
        if fixed[i]:
            continue
        r0 = z[i] + v[i]
        r1 = z[i] + vn[i]
        e0 = abs(k * r0) < b[i]
        e1 = abs(k * r1) < b[i]
        if e0 != e1:
            return False
        if not e0 and (r0 > 0.0) != (r1 > 0.0):
            return False
    return True


@njit(cache=True)
def _rows(
    y: FloatArray,
    z: FloatArray,
    v: FloatArray,
    b: FloatArray,
    k: float,
    h: float,
    w: FloatArray,
    fixed: NDArray[np.bool_],
    m: int,
    nearest: int,
    secant: bool,
    any_elastic: bool,
    lo: FloatArray,
    d: FloatArray,
    up: FloatArray,
    rhs: FloatArray,
) -> None:
    """Newton rows of the fixed-bound increment on the current active set."""
    for i in range(m + 1):
        if fixed[i]:
            lo[i] = 0.0
            up[i] = 0.0
            d[i] = 1.0
            rhs[i] = 0.0
            continue
        c = 0.0
        f = 0.0
        lo[i] = 0.0
        up[i] = 0.0
        if i > 0:
            c += 1.0
            f -= y[i - 1]
            if not fixed[i - 1]:
                lo[i] = -1.0 / h
        if i < m:
            c += 1.0
            f += y[i]
            if not fixed[i + 1]:
                up[i] = -1.0 / h
        r = z[i] + v[i]
        if abs(k * r) < b[i]:
            d[i] = c / h + w[i] * k
            rhs[i] = f - w[i] * k * z[i]
        elif secant:
            # half-quadratic (majorising) weights: a descent step always
            wt = 1e-12 * k
            if b[i] > 0.0 and r != 0.0 and b[i] / abs(r) > wt:
                wt = b[i] / abs(r)
            d[i] = c / h + w[i] * wt
            rhs[i] = f - w[i] * wt * z[i]
        elif not any_elastic and i == nearest:
            d[i] = c / h + w[i] * k
            rhs[i] = f - w[i] * k * z[i]
        else:
            d[i] = c / h
            rhs[i] = f - w[i] * (b[i] if r > 0.0 else -b[i])


@njit(cache=True)
def _nearest_stick(
    z: FloatArray,
    v: FloatArray,
    b: FloatArray,
    k: float,
    fixed: NDArray[np.bool_],
    m: int,
    any_fixed: bool,
) -> tuple[bool, int]:
    """(any node elastic, node nearest to sticking or -1 when not needed)."""
    nearest = -1
    gap = math.inf
    for i in range(m + 1):
        if fixed[i]:
            continue
        excess = abs(k * (z[i] + v[i])) - b[i]
        if excess < 0.0:
            return True, -1
        if excess < gap:
            gap = excess
            nearest = i
    if any_fixed:
        return False, -1
    return False, nearest


@njit(cache=True)
def _solve_fixed_bound(
    y: FloatArray,
    z: FloatArray,
    b: FloatArray,
    k: float,
    h: float,
    w: FloatArray,
    fixed: NDArray[np.bool_],
    any_fixed: bool,
    m: int,
    tol: float,
    v: FloatArray,
    work: FloatArray,
) -> int:
    """Increment ``v`` for a fixed bound; Newton iterations or -1."""
    vn = work[0]
    vt = work[1]
    dv = work[2]
    lo = work[3]
    d = work[4]
    up = work[5]
    rhs = work[6]
    cp = work[7]
    dp = work[8]
    mn = m + 1
    bmax = 0.0
    for i in range(mn):
        if not fixed[i] and b[i] > bmax:
            bmax = b[i]
    if bmax == 0.0 and not any_fixed:
        # No friction and no anchor: the wire slides freely (n = 0).
        v[0] = 0.0
        for j in range(m):
            v[j + 1] = v[j] - h * y[j]
        return 0
    res_old = _residual(y, z, v, b, k, h, w, fixed, m)
    if res_old <= tol:
        return 0
    # Round-off floor of long chains in slip (ill-conditioned Laplacian
    # between sparse stick points): accepted when no descent is left.
    loose = tol * (1.0 + 1e3 * m)
    secant = False
    for it in range(_MAX_NEWTON):
        any_elastic, nearest = _nearest_stick(z, v, b, k, fixed, m, any_fixed)
        _rows(
            y,
            z,
            v,
            b,
            k,
            h,
            w,
            fixed,
            m,
            nearest,
            secant,
            any_elastic,
            lo,
            d,
            up,
            rhs,
        )
        if not _thomas(lo, d, up, rhs, vn, cp, dp, mn):
            if secant:
                return -1
            secant = True
            continue
        res = _residual(y, z, vn, b, k, h, w, fixed, m)
        if res <= tol or (
            res <= 1e6 * tol and _same_active_set(z, v, vn, b, k, fixed, m)
        ):
            for i in range(mn):
                v[i] = vn[i]
            return it + 1
        f0 = _objective(y, z, v, b, k, h, w, fixed, m)
        f1 = _objective(y, z, vn, b, k, h, w, fixed, m)
        if f1 <= f0 - 1e-14 * abs(f0) or (
            res < 0.5 * res_old and f1 <= f0 + 1e-12 * abs(f0)
        ):
            # descent, or a refinement step at the round-off level of f
            for i in range(mn):
                v[i] = vn[i]
            res_old = res
            secant = False
            continue
        for i in range(mn):
            dv[i] = vn[i] - v[i]
        slope = _directional(y, z, v, dv, b, k, h, w, fixed, m)
        accepted = False
        step = 0.5
        if slope < 0.0:
            for _ in range(50):
                for i in range(mn):
                    vt[i] = v[i] + step * dv[i]
                if (
                    _objective(y, z, vt, b, k, h, w, fixed, m)
                    <= f0 + 1e-4 * step * slope
                ):
                    accepted = True
                    break
                step *= 0.5
        if not accepted:
            if res_old <= 1e6 * tol or (secant and res_old <= loose):
                return it + 1
            if secant:
                return -1
            secant = True
            continue
        for i in range(mn):
            v[i] = vt[i]
        res_old = _residual(y, z, v, b, k, h, w, fixed, m)
        secant = False
    return -1


@njit(cache=True)
def _bounds(
    nseg: FloatArray, gb: FloatArray, lam: float, m: int, out: FloatArray
) -> None:
    """Node bounds ``gb + lam (|n_left| + |n_right|) / 2`` (ghost forces zero)."""
    for i in range(m + 1):
        left = abs(nseg[i - 1]) if i > 0 else 0.0
        right = abs(nseg[i]) if i < m else 0.0
        out[i] = gb[i] + lam * 0.5 * (left + right)


@njit(cache=True)
def _project_var(
    y: FloatArray,
    cap: FloatArray,
    left_free: bool,
    right_free: bool,
    out: FloatArray,
    xstar: FloatArray,
    xs: FloatArray,
    gs: FloatArray,
    xt: FloatArray,
    gt: FloatArray,
) -> None:
    """Exact ``argmin 0.5 |n - y|^2`` over ``|n_i - n_{i-1}| <= cap_i``.

    ``cap`` has one entry per node (``M + 1``); interior node ``i`` lies
    between segments ``i - 1`` and ``i``. A free end adds the ghost force
    zero beyond it (``|n_0| <= cap_0``, ``|n_{M-1}| <= cap_M``); an anchored
    end adds nothing. Slope-trick dynamic programme of
    :func:`cablexsec.mechanics.clamp_layer.chain_projection` with node-wise
    bounds and walls for free ends.
    """
    m = y.size
    lo = y[0]
    hi = y[0]
    for i in range(1, m):
        lo = min(lo, y[i])
        hi = max(hi, y[i])
    if left_free:
        lo = min(lo, -cap[0])
        hi = max(hi, cap[0])
        if cap[0] > 0.0:
            size = 2
            xs[0] = -cap[0]
            gs[0] = -cap[0] - y[0]
            xs[1] = cap[0]
            gs[1] = cap[0] - y[0]
        else:
            size = 1
            xs[0] = 0.0
            gs[0] = -y[0]
        sl = _BIG
        sr = _BIG
    else:
        size = 1
        xs[0] = y[0]
        gs[0] = 0.0
        sl = 1.0
        sr = 1.0
    if right_free:
        lo = min(lo, -cap[m])
        hi = max(hi, cap[m])
    for i in range(m):
        if i > 0:
            bound = cap[i]
            x0 = xstar[i - 1]
            k0 = _lower_bound(xs, size, x0)
            k1 = _upper_bound(xs, size, x0)
            # one-sided derivatives at the minimiser: a wall (free end) makes
            # the derivative jump there, so they need not vanish
            if k1 > k0:
                g_left = gs[k0]
                g_right = gs[k1 - 1]
            elif k0 == 0:
                g_left = g_right = gs[0] + sl * (x0 - xs[0])
            elif k0 == size:
                g_left = g_right = gs[size - 1] + sr * (x0 - xs[size - 1])
            else:
                x1 = xs[k0 - 1]
                x2 = xs[k0]
                g_left = g_right = gs[k0 - 1] + (gs[k0] - gs[k0 - 1]) * (x0 - x1) / (
                    x2 - x1
                )
            g_left = min(g_left, 0.0)
            g_right = max(g_right, 0.0)
            t = 0
            for j in range(k0):
                xt[t] = xs[j] - bound
                gt[t] = gs[j]
                t += 1
            if g_left < 0.0:
                xt[t] = x0 - bound
                gt[t] = g_left
                t += 1
            xt[t] = x0 - bound
            gt[t] = 0.0
            t += 1
            xt[t] = x0 + bound
            gt[t] = 0.0
            t += 1
            if g_right > 0.0:
                xt[t] = x0 + bound
                gt[t] = g_right
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
    last = xstar[m - 1]
    if right_free:
        last = _clip(last, cap[m])
    out[m - 1] = last
    for i in range(m - 2, -1, -1):
        value = xstar[i]
        value = max(value, out[i + 1] - cap[i + 1])
        value = min(value, out[i + 1] + cap[i + 1])
        out[i] = value


@njit(cache=True)
def _increment(
    e_new: FloatArray,
    e_old: FloatArray,
    gb: FloatArray,
    k: float,
    lam: float,
    h: float,
    w: FloatArray,
    fixed: NDArray[np.bool_],
    any_fixed: bool,
    rigid: bool,
    u_old: FloatArray,
    z_old: FloatArray,
    n_old: FloatArray,
    t_old: FloatArray,
    v: FloatArray,
    u_new: FloatArray,
    z_new: FloatArray,
    n_new: FloatArray,
    t_new: FloatArray,
    energy: FloatArray,
    work: FloatArray,
    pwork: FloatArray,
    stats: NDArray[np.int64],
) -> bool:
    """One backward-Euler increment of one wire (old state is not modified).

    Elastic interface: ``v`` holds the initial guess of the slip increment
    and returns the solution. Rigid interface (``rigid``): exact projection
    of ``n_old + de`` (``v`` returns the slip increment from the forces).
    ``energy`` returns ``[W_be, D, R]`` of the increment.
    """
    m = e_new.size
    mn = m + 1
    y = work[9]
    b = work[10]
    ncur = work[11]
    scale = 1.0
    for j in range(m):
        if rigid:
            y[j] = n_old[j] + (e_new[j] - e_old[j])
        else:
            y[j] = e_new[j] + (u_old[j + 1] - u_old[j]) / h
        if abs(y[j]) > scale:
            scale = abs(y[j])
        if abs(e_new[j]) > scale:
            scale = abs(e_new[j])
    for i in range(mn):
        if fixed[i]:
            v[i] = 0.0
        if gb[i] > scale:
            scale = gb[i]
    tol = 1e-12 * h * scale
    if rigid:
        for j in range(m):
            ncur[j] = n_old[j]
    else:
        for j in range(m):
            ncur[j] = y[j] + (v[j + 1] - v[j]) / h
    cap = pwork[5]
    nn = pwork[6]
    converged = False
    for it in range(_MAX_FIXED_POINT):
        _bounds(ncur, gb, lam, m, b)
        if rigid:
            for i in range(mn):
                cap[i] = w[i] * b[i]
            _project_var(
                y[:m],
                cap,
                not fixed[0],
                not fixed[m],
                nn,
                pwork[0],
                pwork[1],
                pwork[2],
                pwork[3],
                pwork[4],
            )
        else:
            iters = _solve_fixed_bound(
                y, z_old, b, k, h, w, fixed, any_fixed, m, tol, v, work
            )
            if iters < 0:
                stats[2] += 1
                return False
            if iters > stats[0]:
                stats[0] = iters
            for j in range(m):
                nn[j] = y[j] + (v[j + 1] - v[j]) / h
        diff = 0.0
        big = 1.0
        for j in range(m):
            if abs(nn[j] - ncur[j]) > diff:
                diff = abs(nn[j] - ncur[j])
            if abs(nn[j]) > big:
                big = abs(nn[j])
            ncur[j] = nn[j]
        if not math.isfinite(diff):
            return False
        if lam == 0.0 or diff <= 1e-11 * big:
            if it + 1 > stats[1]:
                stats[1] = it + 1
            converged = True
            break
    if not converged:
        stats[2] += 1
        return False
    _bounds(ncur, gb, lam, m, b)
    w_be = 0.0
    rem = 0.0
    for j in range(m):
        n_new[j] = ncur[j]
        w_be += h * ncur[j] * (e_new[j] - e_old[j])
        dn = ncur[j] - n_old[j]
        rem += 0.5 * h * dn * dn
    if rigid:
        # Slip increment from the forces, zero at an anchored node or at the
        # node farthest from its bound; dissipation from the energy identity.
        ref = -1
        for i in range(mn):
            if fixed[i]:
                ref = i
                break
        if ref < 0:
            best = -math.inf
            for i in range(mn):
                right = ncur[i] if i < m else 0.0
                left = ncur[i - 1] if i > 0 else 0.0
                slack = w[i] * b[i] - abs(right - left)
                if slack > best:
                    best = slack
                    ref = i
        acc = 0.0
        for i in range(mn):
            if i > 0:
                acc += h * (ncur[i - 1] - y[i - 1])
            v[i] = acc
        shift = v[ref]
        stored0 = 0.0
        stored1 = 0.0
        for j in range(m):
            stored0 += 0.5 * h * n_old[j] * n_old[j]
            stored1 += 0.5 * h * ncur[j] * ncur[j]
        for i in range(mn):
            v[i] -= shift
            u_new[i] = u_old[i] + v[i]
            z_new[i] = 0.0
            right = ncur[i] if i < m else 0.0
            left = ncur[i - 1] if i > 0 else 0.0
            t_new[i] = 0.0 if fixed[i] else (right - left) / w[i]
        energy[0] = w_be
        energy[1] = w_be - (stored1 - stored0) - rem
        energy[2] = rem
        return True
    diss = 0.0
    for i in range(mn):
        u_new[i] = u_old[i] + v[i]
        if fixed[i]:
            z_new[i] = 0.0
            t_new[i] = 0.0
            continue
        t = _clip(k * (z_old[i] + v[i]), b[i])
        zn = t / k
        dpl = v[i] - (zn - z_old[i])
        diss += w[i] * t * dpl
        dt = t - t_old[i]
        rem += 0.5 * w[i] * dt * dt / k
        z_new[i] = zn
        t_new[i] = t
    energy[0] = w_be
    energy[1] = diss
    energy[2] = rem
    return True


@njit(cache=True)
def _stick_mask(
    z_old: FloatArray,
    v: FloatArray,
    n_new: FloatArray,
    b: FloatArray,
    k: float,
    w: FloatArray,
    fixed: NDArray[np.bool_],
    any_fixed: bool,
    rigid: bool,
    out: NDArray[np.bool_],
) -> None:
    """Converged stick set (one node sticks when nothing else holds a free
    chain in full slip)."""
    m = n_new.size
    any_stick = False
    nearest = -1
    gap = math.inf
    for i in range(m + 1):
        out[i] = False
        if fixed[i]:
            continue
        if rigid:
            right = n_new[i] if i < m else 0.0
            left = n_new[i - 1] if i > 0 else 0.0
            excess = abs(right - left) - w[i] * b[i] * (1.0 - 1e-9)
        else:
            excess = abs(k * (z_old[i] + v[i])) - b[i] * (1.0 - 1e-12)
        if excess < 0.0:
            out[i] = True
            any_stick = True
        elif excess < gap:
            gap = excess
            nearest = i
    if not any_stick and not any_fixed and nearest >= 0:
        out[nearest] = True


@njit(cache=True)
def _tangent_rows(
    stick: NDArray[np.bool_],
    k: float,
    h: float,
    w: FloatArray,
    fixed: NDArray[np.bool_],
    m: int,
    lo: FloatArray,
    d: FloatArray,
    up: FloatArray,
) -> None:
    for i in range(m + 1):
        lo[i] = 0.0
        up[i] = 0.0
        if fixed[i]:
            d[i] = 1.0
            continue
        c = 0.0
        if i > 0:
            c += 1.0
            if not fixed[i - 1]:
                lo[i] = -1.0 / h
        if i < m:
            c += 1.0
            if not fixed[i + 1]:
                up[i] = -1.0 / h
        d[i] = c / h + (w[i] * k if stick[i] else 0.0)


@njit(cache=True)
def _layer_step(
    e_new: FloatArray,
    e_old: FloatArray,
    gb: FloatArray,
    k: float,
    lam: float,
    h: float,
    w: FloatArray,
    fixed: NDArray[np.bool_],
    any_fixed: bool,
    rigid: bool,
    u_old: FloatArray,
    z_old: FloatArray,
    n_old: FloatArray,
    t_old: FloatArray,
    v: FloatArray,
    u_new: FloatArray,
    z_new: FloatArray,
    n_new: FloatArray,
    t_new: FloatArray,
    energy: FloatArray,
    stats: NDArray[np.int64],
) -> int:
    """One increment of every wire of a layer; 0, or ``-(wire + 1)``."""
    n_w = e_new.shape[0]
    m = e_new.shape[1]
    work = np.empty((12, m + 1))
    pwork = np.empty((7, 4 * m + 8))
    for q in range(n_w):
        ok = _increment(
            e_new[q],
            e_old[q],
            gb[q],
            k,
            lam,
            h,
            w,
            fixed,
            any_fixed,
            rigid,
            u_old[q],
            z_old[q],
            n_old[q],
            t_old[q],
            v[q],
            u_new[q],
            z_new[q],
            n_new[q],
            t_new[q],
            energy[q],
            work,
            pwork,
            stats,
        )
        if not ok:
            return -(q + 1)
    return 0


@njit(cache=True)
def _layer_jacobian(
    z_old: FloatArray,
    v: FloatArray,
    n_new: FloatArray,
    gb: FloatArray,
    k: float,
    lam: float,
    h: float,
    w: FloatArray,
    fixed: NDArray[np.bool_],
    any_fixed: bool,
    rigid: bool,
    cos_seg: FloatArray,
    seg_i0: NDArray[np.int64],
    seg_f: FloatArray,
    st_j0: NDArray[np.int64],
    st_f: FloatArray,
    cos_st: FloatArray,
    scale: float,
    out: FloatArray,
) -> bool:
    """Station Jacobian of the layer moment on the converged active set.

    Column ``j`` is the station moment response (``scale`` times
    ``sum_q cos theta_q n_q`` at every station) to a unit curvature at
    station ``j`` with linear interpolation to the segments, the bound held
    fixed (a rigid interface enters with ``k``). ``out`` is overwritten.
    """
    n_w = cos_seg.shape[0]
    m = cos_seg.shape[1]
    mn = m + 1
    n_s = out.shape[0]
    b = np.empty(mn)
    de = np.zeros(m)
    dn = np.zeros(m)
    lo = np.empty(mn)
    d = np.empty(mn)
    up = np.empty(mn)
    rhs = np.empty(mn)
    cp = np.empty(mn)
    dp = np.empty(mn)
    dv = np.empty(mn)
    piv = np.empty(mn)
    stick = np.zeros(mn, dtype=np.bool_)
    for r in range(n_s):
        for c in range(n_s):
            out[r, c] = 0.0
    first = np.full(n_s, m, dtype=np.int64)
    last = np.full(n_s, -1, dtype=np.int64)
    for s in range(m):
        for jj in (seg_i0[s], seg_i0[s] + 1):
            if jj < n_s:
                if s < first[jj]:
                    first[jj] = s
                if s > last[jj]:
                    last[jj] = s
    for q in range(n_w):
        _bounds(n_new[q], gb[q], lam, m, b)
        _stick_mask(z_old[q], v[q], n_new[q], b, k, w, fixed, any_fixed, rigid, stick)
        _tangent_rows(stick, k, h, w, fixed, m, lo, d, up)
        bet = d[0]
        if bet == 0.0 or not math.isfinite(bet):
            return False
        cp[0] = up[0] / bet
        piv[0] = bet
        for i in range(1, mn):
            bet = d[i] - lo[i] * cp[i - 1]
            if bet == 0.0 or not math.isfinite(bet):
                return False
            piv[i] = bet
            cp[i] = up[i] / bet
        for j in range(n_s):
            if last[j] < 0:
                continue
            for s in range(m):
                de[s] = 0.0
            for s in range(first[j], last[j] + 1):
                wgt = 0.0
                if seg_i0[s] == j:
                    wgt += 1.0 - seg_f[s]
                if seg_i0[s] + 1 == j:
                    wgt += seg_f[s]
                de[s] = cos_seg[q, s] * wgt
            for i in range(mn):
                if fixed[i]:
                    rhs[i] = 0.0
                    continue
                f = 0.0
                if i > 0:
                    f -= de[i - 1]
                if i < m:
                    f += de[i]
                rhs[i] = f
            dp[0] = rhs[0] / piv[0]
            for i in range(1, mn):
                dp[i] = (rhs[i] - lo[i] * dp[i - 1]) / piv[i]
            dv[mn - 1] = dp[mn - 1]
            for i in range(mn - 2, -1, -1):
                dv[i] = dp[i] - cp[i] * dv[i + 1]
            for s in range(m):
                dn[s] = de[s] + (dv[s + 1] - dv[s]) / h
            for r in range(n_s):
                j0 = st_j0[r]
                val = (1.0 - st_f[r]) * dn[j0] + st_f[r] * dn[j0 + 1]
                out[r, j] += scale * cos_st[q, r] * val
    return True


def chain_layer_jacobian(
    chain: WireChain,
    state_old: tuple[FloatArray, FloatArray, FloatArray, FloatArray],
    solution: FloatArray,
    n_new: FloatArray,
    rho: float,
    seg_i0: NDArray[np.int64],
    seg_f: FloatArray,
    st_j0: NDArray[np.int64],
    st_f: FloatArray,
    cos_st: FloatArray,
) -> FloatArray:
    """Station Jacobian ``dM_s / dkappa_j`` of one layer [N m^2] on the
    converged active set of an increment (``solution`` the slip increment
    returned in the ``guess`` of :func:`chain_layer_step`)."""
    lay = chain.layer
    n_s = cos_st.shape[1]
    out = np.zeros((n_s, n_s))
    ok = _layer_jacobian(
        state_old[1],
        np.ascontiguousarray(solution),
        np.ascontiguousarray(n_new),
        np.ascontiguousarray(rho * chain.g_node),
        float(lay.k_hat),
        float(lay.lam),
        float(chain.h),
        chain.weights,
        chain.fixed,
        chain.any_fixed,
        lay.rigid,
        chain.cos_seg,
        seg_i0,
        seg_f,
        st_j0,
        st_f,
        np.ascontiguousarray(cos_st),
        lay.lever_Nm / lay.kappa_c_1_m,
        out,
    )
    if not ok:
        raise CableInputError("along-wire Jacobian is singular")
    return out


@njit(cache=True)
def _wire_history(
    kx: FloatArray,
    ky: FloatArray,
    cs: FloatArray,
    sn: FloatArray,
    gnode: FloatArray,
    rho: FloatArray,
    k: float,
    lam: float,
    h: float,
    w: FloatArray,
    fixed: NDArray[np.bool_],
    any_fixed: bool,
    max_increment: float,
    rigid: bool,
    rec_j: NDArray[np.int64],
    rec_f: FloatArray,
    out_n: FloatArray,
    out_e: FloatArray,
    stats: NDArray[np.int64],
) -> int:
    """History of one wire driven by segment curvatures (units of kappa_c).

    ``kx, ky`` have shape (samples, segments), or (samples, 1) for a load
    common to all segments (``cs, sn`` then carry the shape); the imposed
    strain is ``kx cs + ky sn``. ``out_n`` (samples, records); ``out_e``
    (samples, 4) cumulative ``[W_be, D, U, R]``. Returns 0, or
    ``-(sample + 1)`` for the first sample that could not be advanced.
    """
    n_t = kx.shape[0]
    m = cs.size
    mk = kx.shape[1]
    mn = m + 1
    u = np.zeros(mn)
    z = np.zeros(mn)
    nf = np.zeros(m)
    t = np.zeros(mn)
    e = np.zeros(m)
    u2 = np.zeros(mn)
    z2 = np.zeros(mn)
    n2 = np.zeros(m)
    t2 = np.zeros(mn)
    v = np.zeros(mn)
    su = np.zeros(mn)
    sz = np.zeros(mn)
    snf = np.zeros(m)
    st = np.zeros(mn)
    se = np.zeros(m)
    enew = np.zeros(m)
    gb = np.zeros(mn)
    inc = np.zeros(3)
    energy = np.zeros(3)
    senergy = np.zeros(3)
    work = np.empty((12, mn))
    pwork = np.empty((7, 4 * m + 8))
    kx0 = np.zeros(mk)
    ky0 = np.zeros(mk)
    rho0 = rho[0] if n_t > 0 else 1.0
    for i in range(n_t):
        step = 0.0
        for j in range(m):
            jj = j if mk > 1 else 0
            de = abs((kx[i, jj] - kx0[jj]) * cs[j] + (ky[i, jj] - ky0[jj]) * sn[j])
            if de > step:
                step = de
        nsub = 1
        if max_increment > 0.0 and step > max_increment:
            nsub = int(math.ceil(step / max_increment))
        for j in range(mn):
            su[j] = u[j]
            sz[j] = z[j]
            st[j] = t[j]
        for j in range(m):
            snf[j] = nf[j]
            se[j] = e[j]
        for q in range(3):
            senergy[q] = energy[q]
        ok = False
        for _attempt in range(_MAX_HALVINGS + 1):
            ok = True
            for s in range(1, nsub + 1):
                f = s / nsub
                for j in range(m):
                    jj = j if mk > 1 else 0
                    enew[j] = (kx0[jj] + f * (kx[i, jj] - kx0[jj])) * cs[j] + (
                        ky0[jj] + f * (ky[i, jj] - ky0[jj])
                    ) * sn[j]
                r = rho0 + f * (rho[i] - rho0)
                for j in range(mn):
                    gb[j] = r * gnode[j]
                    v[j] = 0.0
                if not _increment(
                    enew,
                    e,
                    gb,
                    k,
                    lam,
                    h,
                    w,
                    fixed,
                    any_fixed,
                    rigid,
                    u,
                    z,
                    nf,
                    t,
                    v,
                    u2,
                    z2,
                    n2,
                    t2,
                    inc,
                    work,
                    pwork,
                    stats,
                ):
                    ok = False
                    break
                for j in range(mn):
                    u[j] = u2[j]
                    z[j] = z2[j]
                    t[j] = t2[j]
                for j in range(m):
                    nf[j] = n2[j]
                    e[j] = enew[j]
                for q in range(3):
                    energy[q] += inc[q]
            if ok:
                break
            for j in range(mn):
                u[j] = su[j]
                z[j] = sz[j]
                t[j] = st[j]
            for j in range(m):
                nf[j] = snf[j]
                e[j] = se[j]
            for q in range(3):
                energy[q] = senergy[q]
            nsub *= 2
        if not ok:
            return -(i + 1)
        stats[3] += nsub
        for j in range(mk):
            kx0[j] = kx[i, j]
            ky0[j] = ky[i, j]
        rho0 = rho[i]
        stored = 0.0
        for j in range(m):
            stored += 0.5 * h * nf[j] * nf[j]
        for j in range(mn):
            if not fixed[j]:
                stored += 0.5 * w[j] * k * z[j] * z[j]
        out_e[i, 0] = energy[0]
        out_e[i, 1] = energy[1]
        out_e[i, 2] = stored
        out_e[i, 3] = energy[2]
        for r_ in range(rec_j.size):
            j0 = rec_j[r_]
            fr = rec_f[r_]
            if fr == 0.0:
                out_n[i, r_] = nf[j0]
            else:
                out_n[i, r_] = (1.0 - fr) * nf[j0] + fr * nf[j0 + 1]
    return 0


# --------------------------------------------------------------------------- #
# Layer description                                                            #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WireChainLayer:
    """Normalised along-wire description of one helical layer.

    Attributes
    ----------
    name : str
        Label.
    n_wires : int
        Wires of the layer (wire ``q`` at azimuth ``phase0 + 2 pi q / n_w``
        at ``x = 0``).
    r_helix_m, lay_angle_rad : float
        Helix radius and lay angle from the cable axis.
    kappa_c_1_m : float
        Curvature scale of the imposed strain, ``e = kappa cos theta / kappa_c``.
    k_hat : float
        Normalised interface shear stiffness (``RIGID_K_HAT`` for rigid).
    lam : float
        Capstan coefficient of the bound (zero: constant bound).
    force_unit_N : float
        Wire force per unit ``n`` (``g_0``).
    wire_area_m2, ea_N : float
        Wire area and axial stiffness (energy unit).
    stick : str
        Stick convention used to build the layer (information).
    xi : float
        Yun coefficient of the source layer (information).
    shape_g : tuple of float or None
        Azimuthal contact shape at the wire azimuths (mean one); the bound at
        a node is ``rho g(theta)``, periodic linear interpolation.
    sense : int
        ``+1`` when the wire azimuth increases along ``+x``, ``-1`` otherwise.
    phase0_rad : float
        Azimuth of wire 0 at ``x = 0``, measured from the tensile extreme
        fibre of ``kappa_x`` (``theta = 0``).
    """

    name: str
    n_wires: int
    r_helix_m: float
    lay_angle_rad: float
    kappa_c_1_m: float
    k_hat: float
    lam: float
    force_unit_N: float
    wire_area_m2: float
    ea_N: float
    stick: str = "spring"
    xi: float = 1.0
    shape_g: tuple[float, ...] | None = None
    sense: int = 1
    phase0_rad: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_wires, (bool, np.bool_))
            or not isinstance(self.n_wires, (int, np.integer))
            or self.n_wires < 1
        ):
            raise CableInputError("n_wires must be a positive integer")
        for name in (
            "r_helix_m",
            "kappa_c_1_m",
            "k_hat",
            "force_unit_N",
            "wire_area_m2",
            "ea_N",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise CableInputError(f"{name} must be finite and positive")
        if not 0.0 < self.lay_angle_rad < math.pi / 2:
            raise CableInputError("lay_angle_rad must lie in (0, pi/2)")
        if not math.isfinite(self.lam) or self.lam < 0:
            raise CableInputError("lam must be finite and non-negative")
        if self.sense not in (1, -1):
            raise CableInputError("sense must be +1 or -1")
        if self.stick not in STICK_CONVENTIONS:
            raise CableInputError(f"stick must be one of {STICK_CONVENTIONS}")
        if not math.isfinite(self.phase0_rad):
            raise CableInputError("phase0_rad must be finite")
        if self.shape_g is not None:
            g = np.asarray(self.shape_g, dtype=np.float64)
            if g.shape != (self.n_wires,) or not np.isfinite(g).all() or np.any(g < 0):
                raise CableInputError(
                    "shape_g must be finite, non-negative, one per wire"
                )
            object.__setattr__(self, "shape_g", tuple(float(x) for x in g))

    @property
    def rigid(self) -> bool:
        """Rigid interface: increments are exact projections (no spring)."""
        return self.k_hat >= RIGID_K_HAT

    @property
    def pitch_m(self) -> float:
        """Lay length ``2 pi R / tan a`` [m]."""
        return 2.0 * math.pi * self.r_helix_m / math.tan(self.lay_angle_rad)

    @property
    def stress_unit_Pa(self) -> float:
        """Wire stress per unit ``n`` [Pa]."""
        return self.force_unit_N / self.wire_area_m2

    @property
    def lever_Nm(self) -> float:
        """Bending moment of one wire per unit ``n``, ``g_0 R cos a`` [N m]."""
        return self.force_unit_N * self.r_helix_m * math.cos(self.lay_angle_rad)

    @property
    def energy_unit_J(self) -> float:
        """Joules per normalised energy unit and wire, ``g_0^2 R / (EA sin a)``."""
        return (
            self.force_unit_N**2
            * self.r_helix_m
            / (self.ea_N * math.sin(self.lay_angle_rad))
        )

    def azimuth(self, x_m: FloatArray, wire: int | FloatArray) -> FloatArray:
        """Azimuth of wire(s) at axial position(s) ``x`` [rad]."""
        q = np.asarray(wire, dtype=np.float64)
        return np.asarray(
            self.phase0_rad
            + 2.0 * math.pi * q / self.n_wires
            + self.sense
            * 2.0
            * math.pi
            * np.asarray(x_m, dtype=np.float64)
            / self.pitch_m
        )

    def shape_at(self, theta: FloatArray) -> FloatArray:
        """Contact shape ``g`` at azimuths ``theta`` (one when uniform)."""
        th = np.asarray(theta, dtype=np.float64)
        if self.shape_g is None:
            return np.ones_like(th)
        g = np.asarray(self.shape_g, dtype=np.float64)
        n_w = g.size
        pos = np.mod(th, 2.0 * math.pi) * n_w / (2.0 * math.pi)
        i0 = np.floor(pos).astype(np.int64) % n_w
        f = pos - np.floor(pos)
        return np.asarray((1.0 - f) * g[i0] + f * g[(i0 + 1) % n_w])

    @classmethod
    def from_layer_spec(
        cls,
        spec: HelicalLayerSpec,
        *,
        stick: str = "spring",
        capstan: bool = True,
        sense: int = 1,
        name: str = "",
        phase0_rad: float = 0.0,
    ) -> WireChainLayer:
        """The layer of a package armour specification.

        Friction unit, contact shape and capstan coefficient as in the
        distributed model; the stick convention sets ``kappa_c`` and
        ``k_hat`` (module notes). ``"spring"`` needs ``xi < 1``; with
        ``xi = 1`` it equals ``"rigid"``.
        """
        if stick not in STICK_CONVENTIONS:
            raise CableInputError(f"stick must be one of {STICK_CONVENTIONS}")
        alpha = float(spec.lay_angle_rad)
        sin_a = abs(math.sin(alpha))
        mu = float(spec.mu)
        p_ref = float(spec.contact_pressure_Pa)
        n_w = int(spec.n_wires)
        xi = float(spec.xi)
        if not (sin_a > 1e-12 and mu > 0 and p_ref > 0 and 0 < xi <= 1):
            raise CableInputError(
                "the layer has no friction (mu, contact pressure, lay angle or xi)"
            )
        width = 2.0 * math.pi * spec.r_helix_m / n_w
        sigma0 = (
            spec.r_helix_m
            * p_ref
            * width
            * math.cos(alpha)
            / (spec.A_wire_m2 * sin_a**2)
        )
        sigma_u = sigma0 * mu * sin_a
        e_cos2 = float(spec.E_wire_Pa) * math.cos(alpha) ** 2
        kappa_rigid = sigma_u / (e_cos2 * spec.r_helix_m)
        if stick == "scaled":
            kappa_c, k_hat = kappa_rigid / xi, RIGID_K_HAT
        elif stick == "rigid" or xi >= 1.0:
            kappa_c, k_hat = kappa_rigid, RIGID_K_HAT
        else:
            kappa_c, k_hat = kappa_rigid, xi / (1.0 - xi)
        shape = None if spec.shape_g is None else tuple(float(g) for g in spec.shape_g)
        # "scaled" reduces the wire-core strain transfer by xi: the work-conjugate
        # axial stiffness of the chain is xi E A.
        ea = float(spec.E_wire_Pa) * float(spec.A_wire_m2)
        return cls(
            name=name,
            n_wires=n_w,
            r_helix_m=float(spec.r_helix_m),
            lay_angle_rad=alpha,
            kappa_c_1_m=kappa_c,
            k_hat=k_hat,
            lam=mu * sin_a if capstan else 0.0,
            force_unit_N=sigma_u * float(spec.A_wire_m2),
            wire_area_m2=float(spec.A_wire_m2),
            ea_N=ea * xi if stick == "scaled" else ea,
            stick=stick,
            xi=xi,
            shape_g=shape,
            sense=sense,
            phase0_rad=phase0_rad,
        )

    @classmethod
    def from_conductor_ring(
        cls,
        ring: ConductorRing,
        E_Pa: float,
        parameters: StrandSlipParameters,
        *,
        tension_N: float = 0.0,
        name: str = "",
        phase0_rad: float = 0.0,
    ) -> WireChainLayer:
        """A conductor ring with one slipping order (Papailiou cap shape).

        The ring's centroid parameters (``conductor_ring_parameters(...,
        fibre="centroid")``) give the friction strain amplitude ``eps_f`` and
        the stick and slip slopes ``beta_0, beta_1``. With the Papailiou
        shape the sectional cap is ``eps_f (2 / pi)`` per radian from the
        neutral axis, so the chain has a constant bound with
        ``g_0 = E A eps_f (2 / pi)`` and ``kappa_c = g_0 / (E A (beta_0 -
        beta_1))``, rigid stick (the operator has none) and no capstan term.
        The slip-slope part ``beta_1`` is elastic and is not in the chain.
        """
        if ring.r_helix_m <= 0:
            raise CableInputError("the centre wire has no helix")
        if parameters.n_orders != 1:
            raise CableInputError("one slipping order per ring is supported")
        eps = float(parameters.friction_amplitude(np.array([float(tension_N)]))[0, 0])
        beta0, beta1 = float(parameters.beta_m[0]), float(parameters.beta_m[1])
        if not (eps > 0 and beta0 > beta1):
            raise CableInputError("the ring has no friction capacity")
        area = ring.area_m2
        g0 = E_Pa * area * eps * 2.0 / math.pi
        return cls(
            name=name,
            n_wires=int(ring.n_wires),
            r_helix_m=float(ring.r_helix_m),
            lay_angle_rad=float(ring.lay_angle_rad),
            kappa_c_1_m=g0 / (E_Pa * area * (beta0 - beta1)),
            k_hat=RIGID_K_HAT,
            lam=0.0,
            force_unit_N=g0,
            wire_area_m2=area,
            ea_N=E_Pa * area,
            stick="rigid",
            xi=1.0,
            sense=1 if ring.lay_sign > 0 else -1,
            phase0_rad=phase0_rad,
        )


# --------------------------------------------------------------------------- #
# Discretised chain of one layer                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WireChain:
    """The wires of one layer discretised between two ends.

    ``x_seg_m`` are segment midpoints, ``x_node_m`` nodes (``M + 1``);
    ``cos_seg``/``sin_seg`` and ``g_node`` have one row per wire.
    """

    layer: WireChainLayer
    x_start_m: float
    x_end_m: float
    ends: tuple[str, str]
    segments: int
    h: float
    x_seg_m: FloatArray
    x_node_m: FloatArray
    weights: FloatArray
    fixed: NDArray[np.bool_]
    cos_seg: FloatArray
    sin_seg: FloatArray
    g_node: FloatArray

    @classmethod
    def build(
        cls,
        layer: WireChainLayer,
        x_start_m: float,
        x_end_m: float,
        *,
        ends: tuple[str, str] = ("anchored", "anchored"),
        segments_per_pitch: int = 64,
        wires: Sequence[int] | None = None,
    ) -> WireChain:
        """Discretise ``[x_start, x_end]`` with at least ``segments_per_pitch``
        segments per pitch (rounded to a whole number over the length)."""
        if not (math.isfinite(x_start_m) and math.isfinite(x_end_m)) or not (
            x_end_m > x_start_m
        ):
            raise CableInputError("the wire needs x_end_m > x_start_m")
        ends_t = tuple(ends)
        if len(ends_t) != 2 or any(e not in END_CONDITIONS for e in ends_t):
            raise CableInputError(f"ends must be two of {END_CONDITIONS}")
        if (
            isinstance(segments_per_pitch, (bool, np.bool_))
            or not isinstance(segments_per_pitch, (int, np.integer))
            or segments_per_pitch < 8
        ):
            raise CableInputError("segments_per_pitch must be an integer of at least 8")
        pitch = layer.pitch_m
        length = x_end_m - x_start_m
        m = max(2, int(math.ceil(length / pitch * segments_per_pitch - 1e-9)))
        hx = length / m
        h = 2.0 * math.pi * hx / pitch
        x_node = np.asarray(x_start_m + hx * np.arange(m + 1), dtype=np.float64)
        x_seg = np.asarray(x_start_m + hx * (np.arange(m) + 0.5), dtype=np.float64)
        w = np.full(m + 1, h)
        w[0] = w[-1] = 0.5 * h
        fixed = np.zeros(m + 1, dtype=np.bool_)
        fixed[0] = ends_t[0] == "anchored"
        fixed[-1] = ends_t[1] == "anchored"
        q = (
            np.arange(layer.n_wires)
            if wires is None
            else np.asarray(wires, dtype=np.int64)
        )
        if q.ndim != 1 or q.size == 0 or q.min() < 0 or q.max() >= layer.n_wires:
            raise CableInputError("wire indices out of range")
        th_seg = layer.azimuth(x_seg[None, :], q[:, None].astype(np.float64))
        th_node = layer.azimuth(x_node[None, :], q[:, None].astype(np.float64))
        return cls(
            layer=layer,
            x_start_m=float(x_start_m),
            x_end_m=float(x_end_m),
            ends=(str(ends_t[0]), str(ends_t[1])),
            segments=m,
            h=h,
            x_seg_m=x_seg,
            x_node_m=x_node,
            weights=w,
            fixed=fixed,
            cos_seg=np.ascontiguousarray(np.cos(th_seg)),
            sin_seg=np.ascontiguousarray(np.sin(th_seg)),
            g_node=np.ascontiguousarray(layer.shape_at(th_node)),
        )

    @property
    def any_fixed(self) -> bool:
        return bool(self.fixed.any())

    def record_weights(self, x_m: FloatArray) -> tuple[NDArray[np.int64], FloatArray]:
        """Linear interpolation of segment forces to positions (held at ends)."""
        x = np.asarray(x_m, dtype=np.float64)
        hx = (self.x_end_m - self.x_start_m) / self.segments
        pos = np.clip((x - self.x_start_m) / hx - 0.5, 0.0, self.segments - 1.0)
        j0 = np.minimum(np.floor(pos).astype(np.int64), self.segments - 2)
        j0 = np.maximum(j0, 0)
        f = np.clip(pos - j0, 0.0, 1.0)
        return j0, np.asarray(f, dtype=np.float64)


def chain_layer_step(
    chain: WireChain,
    e_new: FloatArray,
    e_old: FloatArray,
    rho: float,
    state: tuple[FloatArray, FloatArray, FloatArray, FloatArray],
    guess: FloatArray,
) -> tuple[tuple[FloatArray, FloatArray, FloatArray, FloatArray], FloatArray]:
    """One increment of every wire of a chain (state ``(u, z, n, t)``).

    The old state is not modified; ``guess`` (slip increment per node, one
    row per wire) is the initial iterate of the elastic-interface solve and
    is overwritten with the slip increment. Returns the new state and the
    increment energies ``(wire, 3) = [W_be, D, R]`` (normalised).
    """
    lay = chain.layer
    u0, z0, n0, t0 = state
    n_w = e_new.shape[0]
    u1 = np.empty_like(u0)
    z1 = np.empty_like(z0)
    n1 = np.empty_like(n0)
    t1 = np.empty_like(t0)
    energy = np.zeros((n_w, 3))
    stats = np.zeros(4, dtype=np.int64)
    gb = np.ascontiguousarray(rho * chain.g_node)
    status = _layer_step(
        np.ascontiguousarray(e_new),
        np.ascontiguousarray(e_old),
        gb,
        float(lay.k_hat),
        float(lay.lam),
        float(chain.h),
        chain.weights,
        chain.fixed,
        chain.any_fixed,
        lay.rigid,
        u0,
        z0,
        n0,
        t0,
        guess,
        u1,
        z1,
        n1,
        t1,
        energy,
        stats,
    )
    if status != 0:
        raise CableInputError(
            f"along-wire increment did not converge (wire {-status - 1})"
        )
    return (u1, z1, n1, t1), energy


# --------------------------------------------------------------------------- #
# One-way history driver                                                        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AlongWireHistory:
    """Wire stress histories of one layer along a specimen or termination.

    ``stress_Pa`` has shape (wires, record positions, samples); ``force``
    the normalised force. ``energy_J`` (wires, samples, 4) holds the
    cumulative ``[W_be, D, U, R]`` per wire in joules (``W_be - D - U - R``
    is round-off). ``azimuth_rad`` (wires, record positions) is the azimuth
    of each recorded wire at each position.
    """

    stress_Pa: FloatArray
    force: FloatArray
    record_x_m: FloatArray
    azimuth_rad: FloatArray
    wires: NDArray[np.int64]
    energy_J: FloatArray
    chain: WireChain
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def balance_residual_J(self) -> FloatArray:
        """``W_be - D - U - R`` per wire and sample [J]."""
        e = self.energy_J
        return np.asarray(e[..., 0] - e[..., 1] - e[..., 2] - e[..., 3])


def along_wire_history(
    layer: WireChainLayer,
    x_m: FloatArray,
    kappa_x: FloatArray,
    kappa_y: FloatArray | None = None,
    *,
    extent_m: tuple[float, float] | None = None,
    ends: tuple[str, str] = ("anchored", "anchored"),
    segments_per_pitch: int = 64,
    contact_ratio: FloatArray | None = None,
    record_x_m: FloatArray | None = None,
    wires: Sequence[int] | None = None,
    max_increment: float = 0.5,
    relief: LateralRelief | None = None,
) -> AlongWireHistory:
    """Wire forces along a non-uniform, time-varying curvature field.

    Parameters
    ----------
    layer : WireChainLayer
        The layer (:meth:`WireChainLayer.from_layer_spec`).
    x_m : numpy.ndarray
        Increasing axial positions at which the curvature is given [m].
    kappa_x, kappa_y : numpy.ndarray
        Curvature components (axial-strain-gradient convention), shape
        (positions, samples) [1/m]; linear in ``x`` between positions, held at
        the end values outside. ``kappa_y`` defaults to zero (planar).
        The history starts from the unloaded state.
    extent_m : (float, float), optional
        Wire ends ``(x_start, x_end)`` (default the first and last position).
    ends : (str, str)
        ``"anchored"`` or ``"free"`` at each end.
    segments_per_pitch : int
        Segments per pitch (minimum).
    contact_ratio : numpy.ndarray, optional
        ``rho(t) = P(t) / P_ref`` per sample (default one).
    record_x_m : numpy.ndarray, optional
        Positions at which forces are recorded (default the segment
        midpoints).
    wires : sequence of int, optional
        Wires to solve (default all).
    max_increment : float
        Largest change of the imposed strain per substep, in units of
        ``kappa_c`` (``inf``: samples only).
    relief : LateralRelief, optional
        Vector-Coulomb relief of the bound by the transverse load (gap G1,
        :class:`~cablexsec.mechanics.tendon_bending.LateralRelief`): the bound
        at a node is multiplied by its factor at the node's azimuth, the
        largest curvature magnitude and the largest ``|dkappa/dx|`` of the
        history at that position, and the distance to the nearest anchored
        end (``None``: unchanged).
    """
    x = np.asarray(x_m, dtype=np.float64)
    kx = np.asarray(kappa_x, dtype=np.float64)
    if x.ndim != 1 or x.size < 1 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise CableInputError("x_m must be finite and strictly increasing")
    if kx.ndim != 2 or kx.shape[0] != x.size or not np.isfinite(kx).all():
        raise CableInputError("kappa_x must be finite with shape (positions, samples)")
    ky = np.zeros_like(kx) if kappa_y is None else np.asarray(kappa_y, dtype=np.float64)
    if ky.shape != kx.shape or not np.isfinite(ky).all():
        raise CableInputError("kappa_y must match kappa_x")
    n_t = kx.shape[1]
    rho = (
        np.ones(n_t)
        if contact_ratio is None
        else np.ascontiguousarray(contact_ratio, dtype=np.float64)
    )
    if rho.shape != (n_t,) or not np.isfinite(rho).all() or np.any(rho < 0):
        raise CableInputError(
            "contact_ratio must be finite, non-negative, one per sample"
        )
    inc = float(max_increment)
    if math.isnan(inc) or inc <= 0:
        raise CableInputError("max_increment must be positive (inf: samples only)")
    lo, hi = (float(x[0]), float(x[-1])) if extent_m is None else extent_m
    chain = WireChain.build(
        layer, lo, hi, ends=ends, segments_per_pitch=segments_per_pitch, wires=wires
    )
    kc = layer.kappa_c_1_m

    def at_segments(field_: FloatArray) -> FloatArray:
        if x.size == 1:
            return np.repeat(field_.T, chain.segments, axis=1) / kc
        out = np.empty((n_t, chain.segments))
        for i in range(n_t):
            out[i] = np.interp(chain.x_seg_m, x, field_[:, i])
        return np.ascontiguousarray(out / kc)

    kx_s = at_segments(kx)
    ky_s = at_segments(ky)
    g_node = chain.g_node
    if relief is not None:
        if not isinstance(relief, LateralRelief):
            raise CableInputError("relief must be a LateralRelief")
        magnitude = np.hypot(kx, ky)
        amp_x = np.max(magnitude, axis=1)
        if x.size > 1:
            grad_x = np.max(
                np.hypot(np.gradient(kx, x, axis=0), np.gradient(ky, x, axis=0)),
                axis=1,
            )
        else:
            grad_x = np.zeros(1)
        anchors = [
            end
            for end, kind in zip(
                (chain.x_start_m, chain.x_end_m), chain.ends, strict=True
            )
            if kind == "anchored"
        ]
        x_node = chain.x_node_m
        dist = (
            np.min(np.abs(x_node[None, :] - np.asarray(anchors)[:, None]), axis=0)
            if anchors
            else None
        )
        q_rows = (
            np.arange(layer.n_wires) if wires is None else np.asarray(wires)
        ).astype(np.float64)
        theta = layer.azimuth(x_node[None, :], q_rows[:, None])
        amp = np.interp(x_node, x, amp_x)[None, :]
        grad = np.interp(x_node, x, grad_x)[None, :]
        g_node = np.ascontiguousarray(
            chain.g_node
            * relief.factor(
                theta,
                np.broadcast_to(amp, theta.shape),
                np.broadcast_to(grad, theta.shape),
                None if dist is None else np.broadcast_to(dist[None, :], theta.shape),
            )
        )
    rec = (
        chain.x_seg_m
        if record_x_m is None
        else np.asarray(record_x_m, dtype=np.float64)
    )
    if rec.ndim != 1 or rec.size == 0 or not np.isfinite(rec).all():
        raise CableInputError("record_x_m must be a finite non-empty vector")
    rec_j, rec_f = chain.record_weights(rec)
    q_ids = (
        np.arange(layer.n_wires, dtype=np.int64)
        if wires is None
        else np.asarray(wires, dtype=np.int64)
    )
    n_rec = rec.size
    force = np.zeros((q_ids.size, n_rec, n_t))
    energy = np.zeros((q_ids.size, n_t, 4))
    stats = np.zeros(4, dtype=np.int64)
    k_hat = float(layer.k_hat)
    for row in range(q_ids.size):
        out_n = np.zeros((n_t, n_rec))
        out_e = np.zeros((n_t, 4))
        status = _wire_history(
            kx_s,
            ky_s,
            chain.cos_seg[row],
            chain.sin_seg[row],
            np.ascontiguousarray(g_node[row]),
            rho,
            k_hat,
            float(layer.lam),
            float(chain.h),
            chain.weights,
            chain.fixed,
            chain.any_fixed,
            inc,
            layer.rigid,
            rec_j,
            rec_f,
            out_n,
            out_e,
            stats,
        )
        if status != 0:
            raise CableInputError(
                f"along-wire history did not converge at sample {-status - 1} "
                f"(wire {int(q_ids[row])})"
            )
        force[row] = out_n.T
        energy[row] = out_e * layer.energy_unit_J
    azimuth = layer.azimuth(rec[None, :], q_ids[:, None].astype(np.float64))
    return AlongWireHistory(
        stress_Pa=layer.stress_unit_Pa * force,
        force=force,
        record_x_m=rec,
        azimuth_rad=np.asarray(azimuth),
        wires=q_ids,
        energy_J=energy,
        chain=chain,
        parameters={
            "segments": chain.segments,
            "h_rad": chain.h,
            "kappa_c_1_m": kc,
            "k_hat": k_hat,
            "lam": float(layer.lam),
            "stick": layer.stick,
            "ends": chain.ends,
            "max_increment": inc,
            "substeps": int(stats[3]),
            "max_newton": int(stats[0]),
            "max_fixed_point": int(stats[1]),
            "compiled": HAVE_NUMBA,
        },
    )
