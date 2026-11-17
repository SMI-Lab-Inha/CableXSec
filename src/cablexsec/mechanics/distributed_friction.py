# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Distributed elastic--Coulomb friction along the helix of one armour layer.

Alternative to the direction-tied pointwise cap of
:mod:`cablexsec.mechanics.fast_slip` (``friction_model="direction_tied"``).
Here the axial force of a wire is solved as a field along one pitch of its
helix, with interface shear compliance and the Papailiou capstan bound, so
that stick zones anchor the force distribution for arbitrary biaxial
curvature histories. No bending direction is needed: the model has no
held-direction rule. The fatigue option ``friction_model="auto"`` uses it
for every non-planar curvature path and the direction-tied cap for planar
ones.

Model (normalised; one wire on a rigid core, uniform curvature, periodic over
one pitch, axial slip only)::

    n = e + du/dphi,   dn/dphi = t,   t = k_hat (u - p),   |t| <= b,
    b = rho(t) g(phi) + lam n,        dp/dt != 0 only where |t| = b,

with ``e = a_x cos(phi) + a_y sin(phi)``, ``a = kappa / kappa_c``,
``kappa_c = xi * kappa_on`` the rigid-stick onset curvature,
``kappa_on = sigma_0 mu |sin a| / (k_f r)`` the slip-onset curvature of the
package (``k_f = xi E cos^2 a``), ``k_hat = xi / (1 - xi)`` (Yun ``xi``;
below slip the periodic solution is ``n = a xi cos(phi)``, i.e. the
``xi``-reduced stick stiffness of the package), ``lam = mu |sin a|`` (capstan
bound on the signed force: a rising force raises its normal load, a falling
one lowers it, as in [Papailiou1997] Eqs. 9-14 with the signed helix angle;
``b`` is floored at zero; ``lam -> 0`` is the constant bound, available as
``bound="uniform"``),
``rho = P(t) / P_ref`` the clamping-pressure ratio and ``g(phi)`` the
optional azimuthal contact shape (periodic linear interpolation of
``shape_g``). The wire stress is
``sigma = (g_0 / A) n`` with ``g_0 / A = sigma_0(P_ref) mu |sin a|``, the
slope of the linearised Papailiou cap. In planar monotonic loading to full slip
``dn/dphi = +-(g + lam n)`` between the extreme fibres and the zero pitch mean
of ``n`` (periodic ``u``) give, for ``g = 1``,
``sigma = sigma_0 (C exp(lam x) - 1)``, ``C = lam pi / (2 sinh(lam pi / 2))``,
``x`` the azimuth from the geometric neutral axis toward the tension side: the
neutral axis moves toward the tension side and the extreme-fibre range is
``2 sigma_0 sinh(lam pi / 2) C = sigma_0 lam pi``, the linear Coulomb value.
The pointwise model holds each wire to its own interval instead (zero at the
geometric neutral axis); in full slip the two differ by ``O(lam^2)``.

Discretisation: ``N`` nodes per pitch, ``h = 2 pi / N``; segment ``j`` sits
at the section azimuth ``j h`` and carries ``n_j``; node ``j`` (between
segments ``j - 1`` and ``j``) carries ``u_j``, ``p_j`` and the traction
``t_j``. Wire ``k`` at azimuth ``2 pi k / n_w`` samples the segment field by
phase (exactly when ``N`` is a multiple of ``n_w``, otherwise by periodic
linear interpolation). Each increment is backward Euler and solves the nodal
equilibrium ``n_j - n_{j-1} = h clip(k_hat (u_j - p_j + v_j), -b_j, b_j)`` with
the bound evaluated at the end of the increment. With the capstan term a
semismooth Newton iteration on the tridiagonal generalised Jacobian (including
``db/dn``) is used first, with a cyclic tridiagonal solve (Thomas plus
Sherman--Morrison, O(N)) per iteration. If it fails, the scheme of the
independently verified reference solver (``stick_check``) is used: for a fixed bound,
semismooth Newton on the convex function
``sum_j h n_j(v)^2 / 2 + h sum_j H_b(u_j - p_j + v_j)`` (``H_b`` the Huber
envelope of the friction dissipation) with Armijo backtracking, and a fixed
point on the bound. Both converge to the same equations (tolerance
``1e-12 h`` times the largest force term). Sample increments larger than
``max_increment`` (in units of ``kappa_on``) are subdivided by linear
interpolation of curvature and pressure; an increment that fails to converge is
repeated with twice as many substeps. With numba the increments are solved by
a faster kernel with the same equations, tests and fall-backs: Newton starts
from the increment of the last substep and the factorisation of its Jacobian
is kept between solves (section "Fast compiled kernel"); the results agree
with the reference kernel to round-off.

Full slip: when every node of an increment is on its bound, the equations fix
the forces, the tractions and ``u - p``, but not a slide ``c`` common to the
displacement and the slip of every node. Any ``c`` that keeps each node
slipping in the direction of its traction is in equilibrium and does the same
friction work (the tractions sum to zero over the pitch, so ``sum h t c = 0``);
the solver alone would leave one node exactly on its bound, a choice made at
round-off between candidate nodes. The increment takes the slide that gives
the step slip a zero pitch mean, or the admissible slide nearest to it. A
uniform curvature, which puts no net axial force on the wire over a pitch,
then slides no wire as a whole; this is the convention of the pointwise laws
in :mod:`cablexsec.analysis.kinematics`, where the pitch mean of the slip keeps
its value over a step in which no station sticks. In partial slip the stick
nodes fix the slide and nothing changes. The kept displacement ``u``, slip
``p`` and step slip are then unique. Forces, stresses and energies do not
depend on the slide, except through the healing of a friction law, where a
node counts as slipping when its traction is on the bound.

Energy (normalised per pitch, per wire): with
``U = h sum n^2/2 + h sum t^2/(2 k_hat)``, the identity
``W_be - D - U - R = 0`` holds every increment, ``W_be = sum h n . de``,
``D = sum h t . dp`` (friction dissipation, non-negative node by node) and
``R = sum h dn^2/2 + h dt^2/(2 k_hat)`` the backward-Euler remainder. Values
are converted to J per metre of cable for the whole layer.

Assumptions shared with the pointwise model: mean clamping prescribed per
sample (no bending-induced redistribution), uniform curvature along the pitch
(no end effects), axial slip only (no transverse slip), quasi-static. The
stick representation differs: the pointwise model is an independent spring
slider per wire, this model a shear-lag interface along the helix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from cablexsec._numba import HAVE_NUMBA, njit
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.friction_law import (
    FrictionLaw,
    _pressure_factor,
    active_friction_law,
    friction_law_time_steps,
)

if TYPE_CHECKING:
    from cablexsec.mechanics.mkappa import HelicalLayerSpec

__all__ = [
    "BOUND_FORMS",
    "DEFAULT_MAX_INCREMENT",
    "DEFAULT_NODES_PER_WIRE",
    "FRICTION_MODELS",
    "HAVE_NUMBA",
    "DistributedFrictionHistory",
    "distributed_friction_history",
    "distributed_wire_friction_stress",
]

FloatArray = NDArray[np.float64]

#: Accepted values of the ``friction_model`` option of the fatigue path
#: (``"auto"`` is resolved per layer by :mod:`cablexsec.analysis.fatigue`).
FRICTION_MODELS = ("direction_tied", "auto", "distributed", "progressive")
#: Forms of the friction bound of the distributed model: ``"signed"`` (capstan
#: term ``lam n`` on the signed bending force of the wire, the default) and
#: ``"uniform"`` (no capstan term, ``b = rho g``).
BOUND_FORMS = ("signed", "uniform")
#: Default helix nodes per wire (``N = DEFAULT_NODES_PER_WIRE * n_wires``).
DEFAULT_NODES_PER_WIRE = 4
#: Default largest curvature increment per substep, in units of ``kappa_on``.
DEFAULT_MAX_INCREMENT = 0.1

_MAX_NEWTON = 60
_MAX_FIXED_POINT = 100
_MAX_HALVINGS = 12
#: Relative margin within which a node counts as on its bound for the slide
#: of a full-slip increment (:func:`_full_slip_slide`); the node that takes
#: the stick row in full slip sits on its bound to round-off.
_ON_BOUND = 1e-9
#: Use :func:`_distributed_kernel_fast` (numba active) instead of the
#: reference :func:`_distributed_kernel`; both solve the same increments and
#: agree to round-off (tests/test_distributed_fast.py).
_FAST_KERNEL = HAVE_NUMBA


# --------------------------------------------------------------------------- #
# Compiled kernel                                                              #
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _cyclic_solve(
    d: FloatArray,
    c: float,
    rhs: FloatArray,
    x: FloatArray,
    cp: FloatArray,
    wy: FloatArray,
    wz: FloatArray,
) -> bool:
    """Solve ``A x = rhs``, ``A`` = diag(d) + c on both off-diagonals and corners.

    Thomas algorithm on the corner-free matrix, corrected by Sherman--Morrison
    (O(N)); same factorisation as ``stick_check.cyclic_tridiag_solve``. Returns
    False for a (numerically) singular system.
    """
    n = d.size
    gam = -d[0]
    bet = d[0] - gam
    if bet == 0.0:
        return False
    cp[0] = c / bet
    wy[0] = rhs[0] / bet
    wz[0] = gam / bet
    for i in range(1, n):
        di = d[i]
        ui = 0.0
        if i == n - 1:
            di = d[i] - c * c / gam
            ui = c
        bet = di - c * cp[i - 1]
        if bet == 0.0:
            return False
        cp[i] = c / bet
        wy[i] = (rhs[i] - c * wy[i - 1]) / bet
        wz[i] = (ui - c * wz[i - 1]) / bet
    for i in range(n - 2, -1, -1):
        wy[i] -= cp[i] * wy[i + 1]
        wz[i] -= cp[i] * wz[i + 1]
    vy = wy[0] + c / gam * wy[n - 1]
    vz = wz[0] + c / gam * wz[n - 1]
    den = 1.0 + vz
    if den == 0.0 or not math.isfinite(den) or not math.isfinite(vy):
        return False
    fac = vy / den
    for i in range(n):
        x[i] = wy[i] - fac * wz[i]
    return True


@njit(cache=True)
def _cyclic_solve_general(
    a: FloatArray,
    d: FloatArray,
    c: FloatArray,
    rhs: FloatArray,
    x: FloatArray,
    cp: FloatArray,
    wy: FloatArray,
    wz: FloatArray,
) -> bool:
    """Solve a cyclic tridiagonal system with row ``j`` = ``(a_j, d_j, c_j)``
    multiplying ``(x_{j-1}, x_j, x_{j+1})`` (indices modulo N), O(N).

    Thomas algorithm plus Sherman--Morrison for the corners ``a_0`` and
    ``c_{N-1}``; returns False for a (numerically) singular pivot.
    """
    n = d.size
    beta = a[0]
    alpha = c[n - 1]
    gam = -d[0]
    bet = d[0] - gam
    if bet == 0.0:
        return False
    cp[0] = c[0] / bet
    wy[0] = rhs[0] / bet
    wz[0] = gam / bet
    for i in range(1, n):
        di = d[i]
        ui = 0.0
        if i == n - 1:
            di = d[i] - alpha * beta / gam
            ui = alpha
        bet = di - a[i] * cp[i - 1]
        if bet == 0.0:
            return False
        cp[i] = c[i] / bet if i < n - 1 else 0.0
        wy[i] = (rhs[i] - a[i] * wy[i - 1]) / bet
        wz[i] = (ui - a[i] * wz[i - 1]) / bet
    for i in range(n - 2, -1, -1):
        wy[i] -= cp[i] * wy[i + 1]
        wz[i] -= cp[i] * wz[i + 1]
    vy = wy[0] + beta / gam * wy[n - 1]
    vz = wz[0] + beta / gam * wz[n - 1]
    den = 1.0 + vz
    if den == 0.0 or not math.isfinite(den):
        return False
    fac = vy / den
    for i in range(n):
        x[i] = wy[i] - fac * wz[i]
    return True


@njit(cache=True)
def _solve_capstan_newton(
    y: FloatArray,
    z0: FloatArray,
    rho: float,
    gnode: FloatArray,
    khat: float,
    lam: float,
    h: float,
    tol: float,
    v: FloatArray,
    b: FloatArray,
    nn: FloatArray,
    la: FloatArray,
    d: FloatArray,
    uc: FloatArray,
    rhs: FloatArray,
    dv: FloatArray,
    cp: FloatArray,
    wy: FloatArray,
    wz: FloatArray,
    prev: FloatArray,
    mfac: FloatArray,
) -> int:
    """Coupled semismooth Newton for the capstan bound ``b = b(n(v))``.

    Solves ``n_j - n_{j-1} = h clip(k_hat (z0_j + v_j), -b_j, b_j)`` with
    ``b_j = m_j (rho g_j + lam (n_j + n_{j-1}) / 2)`` directly (the same
    equations as the fixed point on ``b``; ``m_j`` the friction-law factor of
    the node, one without a law), using the tridiagonal generalised
    Jacobian. Returns iterations, or -1 (caller falls back to the fixed point).
    """
    n = y.size
    last_res = math.inf
    for it in range(_MAX_NEWTON):
        for j in range(n):
            jp = j + 1 if j + 1 < n else 0
            nn[j] = y[j] + (v[jp] - v[j]) / h
        res = 0.0
        any_elastic = False
        same = it > 0
        nearest = 0
        gap = math.inf
        for j in range(n):
            jm = j - 1 if j > 0 else n - 1
            b[j] = mfac[j] * (rho * gnode[j] + lam * 0.5 * (nn[j] + nn[jm]))
            if b[j] < 0.0:
                b[j] = 0.0
            excess = abs(khat * (z0[j] + v[j])) - b[j]
            if excess < gap:
                gap = excess
                nearest = j
        for j in range(n):
            jm = j - 1 if j > 0 else n - 1
            r = z0[j] + v[j]
            kr = khat * r
            if abs(kr) < b[j]:
                state = 0.0
                t = kr
                any_elastic = True
            else:
                state = 1.0 if r > 0.0 else -1.0
                t = state * b[j]
            # Signed capstan: db/dn = m lam / 2 on both segments (zero if floored).
            lm = lam * mfac[j]
            sm = 1.0 if b[j] > 0.0 else 0.0
            sj = sm
            # The equations are linear for a fixed stick/slip state and sign
            # pattern of n; a repeated pattern makes the last step exact.
            key = state + 4.0 * sj
            if key != prev[j]:
                same = False
            prev[j] = key
            g = nn[j] - nn[jm] - h * t
            rhs[j] = g
            if abs(g) > res:
                res = abs(g)
            if state == 0.0 or (gap >= 0.0 and j == nearest):
                # Stick row; in full slip the node nearest to sticking takes
                # the stick row so that the tractions can balance.
                la[j] = -1.0 / h
                d[j] = 2.0 / h + h * khat
                uc[j] = -1.0 / h
            else:
                la[j] = -(1.0 / h + 0.5 * lm * state * sm)
                d[j] = 2.0 / h + 0.5 * lm * state * (sm - sj)
                uc[j] = -(1.0 / h - 0.5 * lm * state * sj)
        if res <= tol or (same and any_elastic and res <= 1e6 * tol):
            return it
        if not math.isfinite(res):
            return -1
        if it > 3 and res > last_res:
            return -1
        last_res = res
        if not _cyclic_solve_general(la, d, uc, rhs, dv, cp, wy, wz):
            return -1
        for j in range(n):
            v[j] += dv[j]
    return -1


@njit(cache=True)
def _residual(
    y: FloatArray, z0: FloatArray, v: FloatArray, b: FloatArray, khat: float, h: float
) -> float:
    """Largest nodal equilibrium residual ``|n_j - n_{j-1} - h t_j|``."""
    n = y.size
    worst = 0.0
    for j in range(n):
        jp = j + 1 if j + 1 < n else 0
        jm = j - 1 if j > 0 else n - 1
        nj = y[j] + (v[jp] - v[j]) / h
        nm = y[jm] + (v[j] - v[jm]) / h
        t = khat * (z0[j] + v[j])
        if t > b[j]:
            t = b[j]
        elif t < -b[j]:
            t = -b[j]
        r = abs(nj - nm - h * t)
        if r > worst:
            worst = r
    return worst


@njit(cache=True)
def _objective(
    y: FloatArray, z0: FloatArray, v: FloatArray, b: FloatArray, khat: float, h: float
) -> float:
    n = y.size
    f = 0.0
    for j in range(n):
        jp = j + 1 if j + 1 < n else 0
        nj = y[j] + (v[jp] - v[j]) / h
        f += 0.5 * h * nj * nj
        r = z0[j] + v[j]
        if abs(khat * r) <= b[j]:
            f += 0.5 * h * khat * r * r
        else:
            f += h * (b[j] * abs(r) - b[j] * b[j] / (2.0 * khat))
    return f


@njit(cache=True)
def _directional(
    y: FloatArray,
    z0: FloatArray,
    v: FloatArray,
    dv: FloatArray,
    b: FloatArray,
    khat: float,
    h: float,
) -> float:
    """Directional derivative of the objective at ``v`` along ``dv``."""
    n = y.size
    s = 0.0
    for j in range(n):
        jp = j + 1 if j + 1 < n else 0
        jm = j - 1 if j > 0 else n - 1
        nj = y[j] + (v[jp] - v[j]) / h
        nm = y[jm] + (v[j] - v[jm]) / h
        t = khat * (z0[j] + v[j])
        if t > b[j]:
            t = b[j]
        elif t < -b[j]:
            t = -b[j]
        s += (nm - nj + h * t) * dv[j]
    return s


@njit(cache=True)
def _same_active_set(
    z0: FloatArray, v: FloatArray, vn: FloatArray, b: FloatArray, khat: float
) -> bool:
    """True when ``v`` and ``vn`` give the same stick nodes and slip signs."""
    for j in range(z0.size):
        r0 = z0[j] + v[j]
        r1 = z0[j] + vn[j]
        e0 = abs(khat * r0) < b[j]
        e1 = abs(khat * r1) < b[j]
        if e0 != e1:
            return False
        if not e0 and (r0 > 0.0) != (r1 > 0.0):
            return False
    return True


@njit(cache=True)
def _solve_fixed_bound(
    y: FloatArray,
    z0: FloatArray,
    b: FloatArray,
    khat: float,
    h: float,
    tol: float,
    v: FloatArray,
    vn: FloatArray,
    vt: FloatArray,
    dv: FloatArray,
    d: FloatArray,
    rhs: FloatArray,
    cp: FloatArray,
    wy: FloatArray,
    wz: FloatArray,
) -> int:
    """Increment ``v`` for a fixed bound; returns Newton iterations or -1."""
    n = y.size
    bmax = 0.0
    for j in range(n):
        if b[j] > bmax:
            bmax = b[j]
    if bmax == 0.0:
        # No friction anywhere: the force is uniform (its pitch mean).
        mean = 0.0
        for j in range(n):
            mean += y[j]
        mean /= n
        v[0] = 0.0
        for j in range(n - 1):
            v[j + 1] = v[j] + h * (mean - y[j])
        return 0
    if _residual(y, z0, v, b, khat, h) <= tol:
        return 0
    secant = False
    for it in range(_MAX_NEWTON):
        any_elastic = False
        nearest = 0
        gap = math.inf
        for j in range(n):
            excess = abs(khat * (z0[j] + v[j])) - b[j]
            if excess < 0.0:
                any_elastic = True
                break
            if excess < gap:
                gap = excess
                nearest = j
        for j in range(n):
            jm = j - 1 if j > 0 else n - 1
            ftr = y[j] - y[jm]
            r = z0[j] + v[j]
            if abs(khat * r) < b[j] or (
                not any_elastic and not secant and j == nearest
            ):
                # Stick, or (all nodes at the bound) the node nearest to
                # sticking: periodicity requires the tractions to balance,
                # which one node off the bound provides in full slip.
                d[j] = 2.0 / h + h * khat
                rhs[j] = ftr - h * khat * z0[j]
            elif any_elastic or not secant:
                d[j] = 2.0 / h
                rhs[j] = ftr - h * (b[j] if r > 0.0 else -b[j])
            else:
                # Fallback: secant (half-quadratic) weights keep the system
                # definite and give a descent direction.
                w = 1e-12 * khat
                if b[j] > 0.0 and r != 0.0 and b[j] / abs(r) > w:
                    w = b[j] / abs(r)
                d[j] = 2.0 / h + h * w
                rhs[j] = ftr - h * w * z0[j]
        if not _cyclic_solve(d, -1.0 / h, rhs, vn, cp, wy, wz):
            if any_elastic or secant:
                return -1
            secant = True
            continue
        res = _residual(y, z0, vn, b, khat, h)
        # Exact for its active set when that set is reproduced (round-off
        # aside); the loose bound only guards against a failed solve.
        if res <= tol or (
            any_elastic and res <= 1e6 * tol and _same_active_set(z0, v, vn, b, khat)
        ):
            for j in range(n):
                v[j] = vn[j]
            return it + 1
        f0 = _objective(y, z0, v, b, khat, h)
        f1 = _objective(y, z0, vn, b, khat, h)
        if f1 <= f0 - 1e-14 * abs(f0):
            for j in range(n):
                v[j] = vn[j]
            secant = False
            continue
        for j in range(n):
            dv[j] = vn[j] - v[j]
        slope = _directional(y, z0, v, dv, b, khat, h)
        accepted = False
        step = 0.5
        if slope < 0.0:
            for _ in range(50):
                for j in range(n):
                    vt[j] = v[j] + step * dv[j]
                if _objective(y, z0, vt, b, khat, h) <= f0 + 1e-4 * step * slope:
                    accepted = True
                    break
                step *= 0.5
        if not accepted:
            if any_elastic or secant:
                return -1
            secant = True  # retry this iterate with the secant weights
            continue
        for j in range(n):
            v[j] = vt[j]
        secant = False
    return -1


@njit(cache=True)
def _full_slip_slide(
    z0: FloatArray,
    v: FloatArray,
    nseg: FloatArray,
    mfac: FloatArray,
    rho: float,
    gnode: FloatArray,
    khat: float,
    lam: float,
) -> None:
    """Fix the common slide of ``u`` and ``p`` of a full-slip increment.

    With every node on its bound (none sticks clear of it, by the relative
    margin ``_ON_BOUND``) the equations fix the segment forces ``nseg`` and
    the tractions, but not a slide ``c`` added to the increment ``v`` (and so
    to the step slip ``dp``) at every node: any ``c`` that keeps each node
    slipping in the direction of its traction is in equilibrium and does the
    same friction work, the tractions summing to zero over the pitch. ``v``
    takes the ``c`` that gives the step slip a zero pitch mean, or the
    admissible ``c`` nearest to it (module notes). No change when a node
    sticks.
    """
    n = z0.size
    lo = -math.inf
    hi = math.inf
    mean = 0.0
    for j in range(n):
        jm = j - 1 if j > 0 else n - 1
        bj = mfac[j] * (rho * gnode[j] + lam * 0.5 * (nseg[j] + nseg[jm]))
        if bj < 0.0:
            bj = 0.0
        r = z0[j] + v[j]
        kr = khat * r
        if abs(kr) < bj * (1.0 - _ON_BOUND):
            return  # a node sticks: it fixes the slide
        t = kr
        if t > bj:
            t = bj
        elif t < -bj:
            t = -bj
        dp = r - t / khat
        mean += dp
        if bj > 0.0:
            # Slip in the direction of the traction: dp + c >= 0 where t > 0.
            if kr > 0.0:
                lo = max(lo, -dp)
            else:
                hi = min(hi, -dp)
    # lo <= 0 <= hi (the solved slide is admissible) to round-off.
    c = -mean / n
    if c < lo:
        c = lo
    if c > hi:
        c = hi
    for j in range(n):
        v[j] += c


@njit(cache=True)
def _advance(
    ax: float,
    ay: float,
    rho: float,
    cs: FloatArray,
    sn: FloatArray,
    gnode: FloatArray,
    khat: float,
    lam: float,
    h: float,
    u: FloatArray,
    p: FloatArray,
    nf: FloatArray,
    tau: FloatArray,
    e: FloatArray,
    energy: FloatArray,
    work: FloatArray,
    stats: NDArray[np.int64],
    capstan_newton: bool,
    mfac: FloatArray,
    law: FloatArray,
    lstate: FloatArray,
    ptab: FloatArray,
    p_ref: float,
) -> bool:
    """One backward-Euler increment to ``(ax, ay, rho)``; False on failure.

    The capstan bound is solved by the coupled Newton iteration when
    ``capstan_newton`` (falling back to the fixed point of the reference on
    failure), otherwise by the fixed point alone.

    ``energy`` accumulates [W_be, D, R]; ``work`` holds N-sized scratch rows.
    ``mfac`` is the friction-law factor of every node's bound; ``law`` =
    ``[active, m_s, m_d, t_stick, dt]`` and ``lstate`` = rows ``[healed,
    timer]`` carry the static/dynamic state. A healed node whose trial
    traction leaves its static bound breaks away: its factor becomes ``m_d``
    and the increment is solved again, until no node breaks away (each node
    at most once per increment).

    ``ptab`` (rows ``[P, f]``, shape (2, K); K = 0 for none) is the pressure
    factor of the law: node ``j`` carries ``m_j f(p_ref rho g_j)`` on its
    bound, with the pressure of the end of the increment.
    """
    n = u.size
    y = work[0]
    z0 = work[1]
    b = work[2]
    v = work[3]
    vn = work[4]
    vt = work[5]
    dv = work[6]
    d = work[7]
    rhs = work[8]
    cp = work[9]
    wy = work[10]
    wz = work[11]
    ncur = work[12]
    enew = work[13]
    nn = work[14]
    la = work[15]
    uc = work[16]
    prev = work[17]
    broke = work[18]
    bend = work[19]
    meff = work[20]
    pfac = work[21]
    has_table = ptab.shape[1] > 0
    for j in range(n):
        if has_table:
            pfac[j] = _pressure_factor(p_ref * rho * gnode[j], ptab)
            meff[j] = mfac[j] * pfac[j]
        else:
            pfac[j] = 1.0
            meff[j] = mfac[j]
    # Residual scale: the largest term entering the force differences.
    scale = 1.0
    for j in range(n):
        jp = j + 1 if j + 1 < n else 0
        enew[j] = ax * cs[j] + ay * sn[j]
        y[j] = enew[j] + (u[jp] - u[j]) / h
        z0[j] = u[j] - p[j]
        v[j] = 0.0
        ncur[j] = nf[j]
        if abs(y[j]) > scale:
            scale = abs(y[j])
        if abs(enew[j]) > scale:
            scale = abs(enew[j])
    if rho > scale:
        scale = rho
    tol = 1e-12 * h * scale
    law_on = law[0] != 0.0 and law[1] != law[2]
    for j in range(n):
        broke[j] = 0.0
    for attempt in range(n + 1):
        if attempt > 0:
            for j in range(n):
                v[j] = 0.0
                ncur[j] = nf[j]
        if not _solve_increment(
            y,
            z0,
            rho,
            gnode,
            khat,
            lam,
            h,
            tol,
            v,
            b,
            nn,
            la,
            d,
            uc,
            rhs,
            dv,
            cp,
            wy,
            wz,
            prev,
            vn,
            vt,
            ncur,
            stats,
            capstan_newton,
            meff,
        ):
            return False
        if not law_on:
            break
        changed = False
        for j in range(n):
            if lstate[0, j] != 0.0:
                jm = j - 1 if j > 0 else n - 1
                bj = meff[j] * (rho * gnode[j] + lam * 0.5 * (ncur[j] + ncur[jm]))
                if bj < 0.0:
                    bj = 0.0
                if abs(khat * (z0[j] + v[j])) > bj:
                    mfac[j] = law[2]
                    meff[j] = mfac[j] * pfac[j] if has_table else mfac[j]
                    lstate[0, j] = 0.0
                    broke[j] = 1.0
                    changed = True
        if not changed:
            break
    _full_slip_slide(z0, v, ncur, meff, rho, gnode, khat, lam)
    w_be = 0.0
    diss = 0.0
    rem = 0.0
    for j in range(n):
        jm = j - 1 if j > 0 else n - 1
        bj = meff[j] * (rho * gnode[j] + lam * 0.5 * (ncur[j] + ncur[jm]))
        if bj < 0.0:
            bj = 0.0
        bend[j] = bj
        t = khat * (z0[j] + v[j])
        slipped = False
        if t > bj:
            t = bj
            slipped = True
        elif t < -bj:
            t = -bj
            slipped = True
        unew = u[j] + v[j]
        dp = (unew - t / khat) - p[j]
        diss += h * t * dp
        w_be += h * ncur[j] * (enew[j] - e[j])
        dn = ncur[j] - nf[j]
        dt = t - tau[j]
        rem += 0.5 * h * dn * dn + 0.5 * h * dt * dt / khat
        u[j] = unew
        p[j] += dp
        tau[j] = t
        nf[j] = ncur[j]
        e[j] = enew[j]
        if law_on:
            if slipped or broke[j] != 0.0:
                lstate[0, j] = 0.0
                lstate[1, j] = 0.0
                mfac[j] = law[2]
            elif lstate[0, j] == 0.0:
                lstate[1, j] += law[4]
                if lstate[1, j] >= law[3]:
                    lstate[0, j] = 1.0
                    mfac[j] = law[1]
    energy[0] += w_be
    energy[1] += diss
    energy[2] += rem
    return True


@njit(cache=True)
def _solve_increment(
    y: FloatArray,
    z0: FloatArray,
    rho: float,
    gnode: FloatArray,
    khat: float,
    lam: float,
    h: float,
    tol: float,
    v: FloatArray,
    b: FloatArray,
    nn: FloatArray,
    la: FloatArray,
    d: FloatArray,
    uc: FloatArray,
    rhs: FloatArray,
    dv: FloatArray,
    cp: FloatArray,
    wy: FloatArray,
    wz: FloatArray,
    prev: FloatArray,
    vn: FloatArray,
    vt: FloatArray,
    ncur: FloatArray,
    stats: NDArray[np.int64],
    capstan_newton: bool,
    mfac: FloatArray,
) -> bool:
    """Solve the increment ``v`` (and the segment forces ``ncur``) for the
    node factors ``mfac``; False if neither scheme converges."""
    n = y.size
    converged = False
    if lam > 0.0 and capstan_newton:
        for j in range(n):
            prev[j] = 2.0
        iters = _solve_capstan_newton(
            y,
            z0,
            rho,
            gnode,
            khat,
            lam,
            h,
            tol,
            v,
            b,
            nn,
            la,
            d,
            uc,
            rhs,
            dv,
            cp,
            wy,
            wz,
            prev,
            mfac,
        )
        if iters >= 0:
            if iters > stats[1]:
                stats[1] = iters
            stats[4] += iters
            for j in range(n):
                jp = j + 1 if j + 1 < n else 0
                ncur[j] = y[j] + (v[jp] - v[j]) / h
            converged = True
        else:
            stats[6] += 1
            for j in range(n):
                v[j] = 0.0
    for it in range(0 if converged else _MAX_FIXED_POINT):
        for j in range(n):
            jm = j - 1 if j > 0 else n - 1
            b[j] = mfac[j] * (rho * gnode[j] + lam * 0.5 * (ncur[j] + ncur[jm]))
            if b[j] < 0.0:
                b[j] = 0.0
        iters = _solve_fixed_bound(
            y, z0, b, khat, h, tol, v, vn, vt, dv, d, rhs, cp, wy, wz
        )
        if iters < 0:
            return False
        if iters > stats[1]:
            stats[1] = iters
        stats[4] += iters
        diff = 0.0
        big = 1.0
        for j in range(n):
            jp = j + 1 if j + 1 < n else 0
            n_new = y[j] + (v[jp] - v[j]) / h
            if abs(n_new - ncur[j]) > diff:
                diff = abs(n_new - ncur[j])
            if abs(n_new) > big:
                big = abs(n_new)
            ncur[j] = n_new
        if not math.isfinite(diff):
            return False
        if lam == 0.0 or diff <= 1e-12 * big:
            if it + 1 > stats[2]:
                stats[2] = it + 1
            stats[5] += it + 1
            converged = True
            break
    return converged


@njit(cache=True)
def _distributed_kernel(
    ax_path: FloatArray,
    ay_path: FloatArray,
    rho_path: FloatArray,
    khat: float,
    lam: float,
    gnode: FloatArray,
    phase0: float,
    xi: float,
    max_increment: float,
    wire_index: NDArray[np.int64],
    wire_frac: FloatArray,
    record_field: bool,
    capstan_newton: bool = True,
    record_index: NDArray[np.int64] | None = None,
    kinematics: FloatArray | None = None,
    law: FloatArray | None = None,
    dt_path: FloatArray | None = None,
    ptab: FloatArray | None = None,
    p_ref: float = 0.0,
) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.int64], FloatArray]:
    """Normalised wire force history.

    Returns ``(n_wires (n_w, T), energy (T, 4) = [W_be, D, U, R],
    slip_fraction (T), stats, field (T or 0, N))``; ``stats = [status,
    max Newton, max fixed point, substeps, ...]`` with status 0 on success and
    ``-(sample + 1)`` for the first sample that could not be advanced, then
    total Newton and fixed-point iterations, capstan Newton fall-backs].

    ``record_index`` (one entry per sample, ``-1``: not kept) and
    ``kinematics`` (kept samples, 4, N) keep the helix field of chosen
    samples: the nodal displacement ``u``, the nodal slip ``p``, the slip of
    the step to the sample ``p - p_before`` and the segment force ``n``.

    ``law`` = ``[m_s, m_d, t_stick]`` (static and dynamic factors, healing
    time) and ``dt_path`` (the duration of every sample step, divided equally
    among its substeps) switch on the static/dynamic friction law; ``None``
    keeps a unit factor on every node. ``ptab`` (rows ``[P, f]``) and the
    reference pressure ``p_ref`` [Pa] add the pressure factor of the law at
    every node's local pressure ``p_ref rho g_j``.
    """
    n = gnode.size
    h = 2.0 * math.pi / n
    n_t = ax_path.size
    n_w = wire_index.size
    cs = np.empty(n)
    sn = np.empty(n)
    for j in range(n):
        cs[j] = math.cos(phase0 + j * h)
        sn[j] = math.sin(phase0 + j * h)
    u = np.zeros(n)
    p = np.zeros(n)
    nf = np.zeros(n)
    tau = np.zeros(n)
    e = np.zeros(n)
    su = np.empty(n)
    sp = np.empty(n)
    snf = np.empty(n)
    stau = np.empty(n)
    se = np.empty(n)
    work = np.zeros((22, n))
    table = np.zeros((2, 0))
    if ptab is not None:
        table = ptab
    mfac = np.ones(n)
    smfac = np.ones(n)
    lstate = np.zeros((2, n))
    slstate = np.zeros((2, n))
    lparams = np.zeros(5)
    if law is not None:
        lparams[0] = 1.0
        lparams[1] = law[0]
        lparams[2] = law[1]
        lparams[3] = law[2]
        for j in range(n):
            mfac[j] = law[0]
            lstate[0, j] = 1.0
    energy = np.zeros(3)
    senergy = np.zeros(3)
    stats = np.zeros(7, dtype=np.int64)
    out = np.empty((n_w, n_t))
    hist = np.empty((n_t, 4))
    slip = np.empty(n_t)
    fld = np.empty((n_t if record_field else 0, n))
    ax0 = 0.0
    ay0 = 0.0
    rho0 = rho_path[0] if n_t > 0 else 0.0
    for i in range(n_t):
        ax1 = ax_path[i]
        ay1 = ay_path[i]
        rho1 = rho_path[i]
        step = math.sqrt((ax1 - ax0) ** 2 + (ay1 - ay0) ** 2) * xi
        nsub = 1
        if max_increment > 0.0 and step > max_increment:
            nsub = int(math.ceil(step / max_increment))
        for j in range(n):
            su[j] = u[j]
            sp[j] = p[j]
            snf[j] = nf[j]
            stau[j] = tau[j]
            se[j] = e[j]
        for k in range(3):
            senergy[k] = energy[k]
        for j in range(n):
            smfac[j] = mfac[j]
            slstate[0, j] = lstate[0, j]
            slstate[1, j] = lstate[1, j]
        ok = False
        for _attempt in range(_MAX_HALVINGS + 1):
            ok = True
            if dt_path is not None:
                lparams[4] = dt_path[i] / nsub
            for s in range(1, nsub + 1):
                f = s / nsub
                if not _advance(
                    ax0 + f * (ax1 - ax0),
                    ay0 + f * (ay1 - ay0),
                    rho0 + f * (rho1 - rho0),
                    cs,
                    sn,
                    gnode,
                    khat,
                    lam,
                    h,
                    u,
                    p,
                    nf,
                    tau,
                    e,
                    energy,
                    work,
                    stats,
                    capstan_newton,
                    mfac,
                    lparams,
                    lstate,
                    table,
                    p_ref,
                ):
                    ok = False
                    break
            if ok:
                break
            for j in range(n):
                u[j] = su[j]
                p[j] = sp[j]
                nf[j] = snf[j]
                tau[j] = stau[j]
                e[j] = se[j]
            for k in range(3):
                energy[k] = senergy[k]
            for j in range(n):
                mfac[j] = smfac[j]
                lstate[0, j] = slstate[0, j]
                lstate[1, j] = slstate[1, j]
            nsub *= 2
        if not ok:
            stats[0] = -(i + 1)
            return out, hist, slip, stats, fld
        stats[3] += nsub
        ax0 = ax1
        ay0 = ay1
        rho0 = rho1
        stored = 0.0
        n_slip = 0
        for j in range(n):
            jm = j - 1 if j > 0 else n - 1
            stored += 0.5 * h * (nf[j] * nf[j] + tau[j] * tau[j] / khat)
            if law is None:
                bj = rho1 * gnode[j] + lam * 0.5 * (nf[j] + nf[jm])
                if bj < 0.0:
                    bj = 0.0
            else:
                bj = work[19, j]  # bound of the last increment
            if bj > 0.0 and abs(tau[j]) >= bj * (1.0 - 1e-9):
                n_slip += 1
        hist[i, 0] = energy[0]
        hist[i, 1] = energy[1]
        hist[i, 2] = stored
        hist[i, 3] = energy[2]
        slip[i] = n_slip / n
        for k in range(n_w):
            j0 = wire_index[k]
            j1 = j0 + 1 if j0 + 1 < n else 0
            out[k, i] = (1.0 - wire_frac[k]) * nf[j0] + wire_frac[k] * nf[j1]
        if record_field:
            for j in range(n):
                fld[i, j] = nf[j]
        if record_index is not None and kinematics is not None:
            row = record_index[i]
            if row >= 0:
                for j in range(n):
                    kinematics[row, 0, j] = u[j]
                    kinematics[row, 1, j] = p[j]
                    kinematics[row, 2, j] = p[j] - sp[j]
                    kinematics[row, 3, j] = nf[j]
    return out, hist, slip, stats, fld


# --------------------------------------------------------------------------- #
# Fast compiled kernel                                                         #
# --------------------------------------------------------------------------- #
# Same increments, equations, convergence tests, fall-backs and substep rule
# as the reference kernel above; only the way each increment is solved
# differs, so the results agree with the reference to round-off
# (tests/test_distributed_fast.py; with a finite healing time of a friction
# law, the heal and break-away events of a node can move by a substep under a
# round-off change, in either kernel alone as much as between them):
#
# - Newton starts from the increment of the last substep instead of zero
#   (the substeps of a sample are equal steps), so the first linearisation
#   usually has the converged stick/slip pattern; an increment is solved again
#   from zero, as in the reference, if that start fails. Both kernels then
#   fix the slide of a full-slip increment by the same rule
#   (:func:`_full_slip_slide`), so the start does not reach the kept fields;
# - the factorisation of the Newton Jacobian is kept between solves. A row is
#   a function of its node's stick/slip key and friction factor only, and
#   the system is eliminated from both ends towards the middle row (twisted
#   factorisation), so a changed row invalidates only the pivots between it
#   and the middle; the stick zones move by a node or two per substep;
# - the residual, bound and pattern of an iterate are formed in one pass, the
#   converged pass gives the segment forces, the two halves of every sweep
#   run as independent recurrences, and divisions by ``h``, ``k_hat`` and the
#   pivots are multiplications by their reciprocals.
#
# Rows of the factorisation cache ``mat`` (shape (_M_ROWS, N)); rows below
# the middle row ``m = N // 2`` hold the top-down elimination, rows above it
# the bottom-up one.
_M_LOWER = 0  # coefficient of v_{j-1}
_M_DIAG = 1  # coefficient of v_j
_M_UPPER = 2  # coefficient of v_{j+1}
_M_PC = 3  # eliminated neighbour coefficient over the pivot
_M_PIV = 4  # 1 / pivot
_M_OFF = 5  # eliminated coefficient over the pivot
_M_ZS = 6  # Sherman--Morrison column, elimination sweep
_M_ZB = 7  # Sherman--Morrison column, solved
_M_KEY = 8  # key the row was built from (_OVERRIDE: stick row in full slip)
_M_FAC = 9  # friction factor the row was built with
_M_ROWS = 10
#: Row key of the node that takes the stick row in full slip.
_OVERRIDE = 8.0
# ``mstate`` = [first stale row of the top half, last stale row of the bottom
# half, 1 + v.z, beta / gam, 1 / middle pivot, a node of the solution sticks].
_CLEAN_LO = 1 << 60
_CLEAN_HI = -1


@njit(cache=True)
def _set_row(
    j: int,
    key: float,
    mj: float,
    lam: float,
    ih: float,
    stick_diag: float,
    mat: FloatArray,
) -> None:
    """Jacobian row ``j`` of :func:`_solve_capstan_newton` for node key
    ``key`` (``state + 4 sm``) and friction factor ``mj``."""
    mat[_M_KEY, j] = key
    mat[_M_FAC, j] = mj
    if key == 4.0 or key == _OVERRIDE:
        mat[_M_LOWER, j] = -ih
        mat[_M_DIAG, j] = stick_diag
        mat[_M_UPPER, j] = -ih
    else:
        sm = 1.0 if key > 2.0 else 0.0
        state = key - 4.0 * sm
        q = 0.5 * (lam * mj) * state * sm
        mat[_M_LOWER, j] = -(ih + q)
        mat[_M_DIAG, j] = 2.0 * ih
        mat[_M_UPPER, j] = -(ih - q)


@njit(cache=True)
def _twisted_refactor(n: int, mat: FloatArray, mstate: FloatArray) -> bool:
    """Update the stale pivots of the twisted factorisation of ``mat``.

    The cyclic system is the tridiagonal one with modified corners plus a
    Sherman--Morrison correction, as in :func:`_cyclic_solve_general`. Top
    rows ``i < m`` are eliminated downwards, bottom rows ``i > m`` upwards;
    rows ``mstate[0] .. m - 1`` and ``m + 1 .. mstate[1]`` are recomputed.
    False for a (numerically) singular system.
    """
    lo = mat[_M_LOWER]
    dg = mat[_M_DIAG]
    up = mat[_M_UPPER]
    pc = mat[_M_PC]
    piv = mat[_M_PIV]
    off = mat[_M_OFF]
    zs = mat[_M_ZS]
    zb = mat[_M_ZB]
    m = n // 2
    s = int(min(mstate[0], m))
    e = int(max(mstate[1], m))
    if s == 0:
        e = n - 1  # row 0 sets the corners of the last row
    beta = lo[0]
    alpha = up[n - 1]
    gam = -dg[0]
    if s == 0:
        bet = dg[0] - gam
        if bet == 0.0:
            return False
        pc[0] = up[0] / bet
        r = 1.0 / bet
        piv[0] = r
        off[0] = 0.0
        zs[0] = gam * r
        s = 1
    if e == n - 1:
        g = dg[n - 1] - alpha * beta / gam
        if g == 0.0:
            return False
        pc[n - 1] = lo[n - 1] / g
        r = 1.0 / g
        piv[n - 1] = r
        off[n - 1] = 0.0
        zs[n - 1] = alpha * r
        e = n - 2
    top = m - s if m > s else 0
    bottom = e - m if e > m else 0
    both = min(top, bottom)
    pct = pc[s - 1] if s > 0 else 0.0
    zt = zs[s - 1] if s > 0 else 0.0
    pcb = pc[e + 1] if e + 1 < n else 0.0
    zbt = zs[e + 1] if e + 1 < n else 0.0
    for k in range(both):
        i = s + k
        a = lo[i]
        bet = dg[i] - a * pct
        i2 = e - k
        c = up[i2]
        g = dg[i2] - c * pcb
        if bet == 0.0 or g == 0.0:
            return False
        pct = up[i] / bet
        pcb = lo[i2] / g
        r = 1.0 / bet
        r2 = 1.0 / g
        o = a * r
        o2 = c * r2
        zt = -o * zt
        zbt = -o2 * zbt
        pc[i] = pct
        piv[i] = r
        off[i] = o
        zs[i] = zt
        pc[i2] = pcb
        piv[i2] = r2
        off[i2] = o2
        zs[i2] = zbt
    for k in range(both, top):
        i = s + k
        a = lo[i]
        bet = dg[i] - a * pct
        if bet == 0.0:
            return False
        pct = up[i] / bet
        r = 1.0 / bet
        o = a * r
        zt = -o * zt
        pc[i] = pct
        piv[i] = r
        off[i] = o
        zs[i] = zt
    for k in range(both, bottom):
        i2 = e - k
        c = up[i2]
        g = dg[i2] - c * pcb
        if g == 0.0:
            return False
        pcb = lo[i2] / g
        r2 = 1.0 / g
        o2 = c * r2
        zbt = -o2 * zbt
        pc[i2] = pcb
        piv[i2] = r2
        off[i2] = o2
        zs[i2] = zbt
    gm = dg[m] - lo[m] * pc[m - 1] - up[m] * pc[m + 1]
    if gm == 0.0:
        return False
    rgm = 1.0 / gm
    zt = (-(lo[m] * zs[m - 1]) - up[m] * zs[m + 1]) * rgm
    zb[m] = zt
    zbt = zt
    left = m
    right = n - 1 - m
    both = min(left, right)
    for k in range(1, both + 1):
        zt = zs[m - k] - pc[m - k] * zt
        zbt = zs[m + k] - pc[m + k] * zbt
        zb[m - k] = zt
        zb[m + k] = zbt
    for k in range(both + 1, left + 1):
        zt = zs[m - k] - pc[m - k] * zt
        zb[m - k] = zt
    for k in range(both + 1, right + 1):
        zbt = zs[m + k] - pc[m + k] * zbt
        zb[m + k] = zbt
    bg = beta / gam
    den = 1.0 + (zb[0] + bg * zb[n - 1])
    if den == 0.0 or not math.isfinite(den):
        return False
    mstate[0] = _CLEAN_LO
    mstate[1] = _CLEAN_HI
    mstate[2] = den
    mstate[3] = bg
    mstate[4] = rgm
    return True


@njit(cache=True)
def _twisted_solve(
    rhs: FloatArray, v: FloatArray, wy: FloatArray, mat: FloatArray, mstate: FloatArray
) -> bool:
    """``v += A^-1 rhs`` for the cyclic tridiagonal Jacobian held in ``mat``
    (stale pivots are updated first); False for a singular system."""
    n = rhs.size
    stale = mstate[0] < n or mstate[1] >= 0
    if stale and not _twisted_refactor(n, mat, mstate):
        mstate[0] = 0.0
        mstate[1] = n - 1
        return False
    lo = mat[_M_LOWER]
    up = mat[_M_UPPER]
    pc = mat[_M_PC]
    piv = mat[_M_PIV]
    off = mat[_M_OFF]
    zb = mat[_M_ZB]
    m = n // 2
    top = m
    bottom = n - 1 - m
    both = min(top, bottom)
    wt = 0.0
    wb = 0.0
    for k in range(both):
        i2 = n - 1 - k
        wt = rhs[k] * piv[k] - off[k] * wt
        wb = rhs[i2] * piv[i2] - off[i2] * wb
        wy[k] = wt
        wy[i2] = wb
    for k in range(both, top):
        wt = rhs[k] * piv[k] - off[k] * wt
        wy[k] = wt
    for k in range(both, bottom):
        i2 = n - 1 - k
        wb = rhs[i2] * piv[i2] - off[i2] * wb
        wy[i2] = wb
    xm = (rhs[m] - lo[m] * wy[m - 1] - up[m] * wy[m + 1]) * mstate[4]
    wy[m] = xm
    wt = xm
    wb = xm
    for k in range(1, both + 1):
        wt = wy[m - k] - pc[m - k] * wt
        wb = wy[m + k] - pc[m + k] * wb
        wy[m - k] = wt
        wy[m + k] = wb
    for k in range(both + 1, top + 1):
        wt = wy[m - k] - pc[m - k] * wt
        wy[m - k] = wt
    for k in range(both + 1, bottom + 1):
        wb = wy[m + k] - pc[m + k] * wb
        wy[m + k] = wb
    fac = (wy[0] + mstate[3] * wy[n - 1]) / mstate[2]
    for i in range(n):
        v[i] += wy[i] - fac * zb[i]
    return True


@njit(cache=True)
def _capstan_newton_fast(
    y: FloatArray,
    z0: FloatArray,
    rho: float,
    gnode: FloatArray,
    khat: float,
    lam: float,
    h: float,
    tol: float,
    v: FloatArray,
    b: FloatArray,
    nn: FloatArray,
    rhs: FloatArray,
    wy: FloatArray,
    prev: FloatArray,
    mfac: FloatArray,
    mat: FloatArray,
    mstate: FloatArray,
) -> int:
    """:func:`_solve_capstan_newton` with the kept twisted factorisation.

    On return ``nn`` and ``b`` hold the segment forces and bounds of the
    returned ``v``, and ``mstate[5]`` is one if a node sticks clear of its
    bound there (by the margin of :func:`_full_slip_slide`, which then leaves
    the increment as it is). Returns iterations, or -1 (caller falls back to
    the fixed point).
    """
    n = y.size
    ih = 1.0 / h
    stick_diag = 2.0 * ih + h * khat
    rk = mat[_M_KEY]
    rm = mat[_M_FAC]
    last_res = math.inf
    for it in range(_MAX_NEWTON):
        lo_s = mstate[0]
        hi_s = mstate[1]
        res = 0.0
        any_elastic = False
        same = it > 0
        firm = False
        nearest = 0
        gap = math.inf
        n_prev = y[n - 1] + (v[0] - v[n - 1]) * ih
        for j in range(n):
            vj = v[j]
            nj = y[j] + ((v[j + 1] if j + 1 < n else v[0]) - vj) * ih
            nn[j] = nj
            mj = mfac[j]
            bj = mj * (rho * gnode[j] + lam * 0.5 * (nj + n_prev))
            if bj < 0.0:
                bj = 0.0
            b[j] = bj
            r = z0[j] + vj
            kr = khat * r
            excess = abs(kr) - bj
            if excess < gap:
                gap = excess
                nearest = j
            if excess < 0.0:
                key = 4.0
                t = kr
                any_elastic = True
                if abs(kr) < bj * (1.0 - _ON_BOUND):
                    firm = True
            else:
                key = 5.0 if r > 0.0 else 3.0
                t = bj if r > 0.0 else -bj
            if not bj > 0.0:
                key -= 4.0
            if key != prev[j]:
                same = False
            prev[j] = key
            if key != rk[j] or ((key == 5.0 or key == 3.0) and mj != rm[j]):
                _set_row(j, key, mj, lam, ih, stick_diag, mat)
                if j < lo_s:
                    lo_s = j
                if j > hi_s:
                    hi_s = j
            g = nj - n_prev - h * t
            rhs[j] = g
            if abs(g) > res:
                res = abs(g)
            n_prev = nj
        mstate[0] = lo_s
        mstate[1] = hi_s
        if res <= tol or (same and any_elastic and res <= 1e6 * tol):
            mstate[5] = 1.0 if firm else 0.0
            return it
        if not math.isfinite(res):
            return -1
        if it > 3 and res > last_res:
            return -1
        last_res = res
        if gap >= 0.0:
            # Full slip: the node nearest to sticking takes the stick row so
            # that the tractions can balance.
            _set_row(nearest, _OVERRIDE, mfac[nearest], lam, ih, stick_diag, mat)
            if nearest < mstate[0]:
                mstate[0] = nearest
            if nearest > mstate[1]:
                mstate[1] = nearest
        if not _twisted_solve(rhs, v, wy, mat, mstate):
            return -1
    return -1


@njit(cache=True)
def _advance_fast(
    ax: float,
    ay: float,
    rho: float,
    cs: FloatArray,
    sn: FloatArray,
    gnode: FloatArray,
    khat: float,
    lam: float,
    h: float,
    u: FloatArray,
    p: FloatArray,
    nf: FloatArray,
    tau: FloatArray,
    e: FloatArray,
    energy: FloatArray,
    work: FloatArray,
    stats: NDArray[np.int64],
    capstan_newton: bool,
    mfac: FloatArray,
    law: FloatArray,
    lstate: FloatArray,
    ptab: FloatArray,
    p_ref: float,
    vlast: FloatArray,
    mat: FloatArray,
    mstate: FloatArray,
) -> bool:
    """:func:`_advance` started from ``vlast``, the increment of the last
    substep, with the kept Newton factorisation; ``vlast`` receives the
    increment of this one."""
    n = u.size
    ih = 1.0 / h
    ik = 1.0 / khat
    y = work[0]
    z0 = work[1]
    b = work[2]
    v = work[3]
    vn = work[4]
    vt = work[5]
    dv = work[6]
    d = work[7]
    rhs = work[8]
    cp = work[9]
    wy = work[10]
    wz = work[11]
    ncur = work[12]
    enew = work[13]
    nn = work[14]
    prev = work[17]
    broke = work[18]
    bend = work[19]
    pfac = work[21]
    has_table = ptab.shape[1] > 0
    law_on = law[0] != 0.0 and law[1] != law[2]
    newton = lam > 0.0 and capstan_newton
    # Without a pressure table the node factors are the law factors.
    meff = work[20] if has_table else mfac
    if has_table:
        for j in range(n):
            pfac[j] = _pressure_factor(p_ref * rho * gnode[j], ptab)
            meff[j] = mfac[j] * pfac[j]
    if law_on:
        for j in range(n):
            broke[j] = 0.0
    scale = 1.0
    for j in range(n):
        ej = ax * cs[j] + ay * sn[j]
        enew[j] = ej
        yj = ej + ((u[j + 1] if j + 1 < n else u[0]) - u[j]) * ih
        y[j] = yj
        z0[j] = u[j] - p[j]
        v[j] = vlast[j]
        if abs(yj) > scale:
            scale = abs(yj)
        if abs(ej) > scale:
            scale = abs(ej)
    if rho > scale:
        scale = rho
    tol = 1e-12 * h * scale
    # The segment forces of the solution: those of the converged Newton pass,
    # or of the fixed point.
    nsol = nn
    sticks = False
    for attempt in range(n + 1):
        if attempt > 0:
            for j in range(n):
                v[j] = 0.0
        # Started from the prediction; solved again from zero (the start of
        # the reference) if that fails.
        predicted = attempt == 0
        while True:
            converged = False
            if newton:
                iters = _capstan_newton_fast(
                    y,
                    z0,
                    rho,
                    gnode,
                    khat,
                    lam,
                    h,
                    tol,
                    v,
                    b,
                    nn,
                    rhs,
                    wy,
                    prev,
                    meff,
                    mat,
                    mstate,
                )
                if iters >= 0:
                    if iters > stats[1]:
                        stats[1] = iters
                    stats[4] += iters
                    nsol = nn
                    converged = True
                    sticks = mstate[5] != 0.0
                elif predicted:
                    predicted = False
                    for j in range(n):
                        v[j] = 0.0
                    continue
                else:
                    stats[6] += 1
                    for j in range(n):
                        v[j] = 0.0
            if not converged:
                for j in range(n):
                    ncur[j] = nf[j]
                if not _fixed_point(
                    y,
                    z0,
                    rho,
                    gnode,
                    khat,
                    lam,
                    h,
                    tol,
                    v,
                    b,
                    d,
                    rhs,
                    dv,
                    cp,
                    wy,
                    wz,
                    vn,
                    vt,
                    ncur,
                    stats,
                    meff,
                ):
                    if not predicted:
                        return False
                    predicted = False
                    for j in range(n):
                        v[j] = 0.0
                    continue
                nsol = ncur
                sticks = False
            break
        if not law_on:
            break
        changed = False
        for j in range(n):
            if lstate[0, j] != 0.0:
                jm = j - 1 if j > 0 else n - 1
                bj = meff[j] * (rho * gnode[j] + lam * 0.5 * (nsol[j] + nsol[jm]))
                if bj < 0.0:
                    bj = 0.0
                if abs(khat * (z0[j] + v[j])) > bj:
                    mfac[j] = law[2]
                    if has_table:
                        meff[j] = mfac[j] * pfac[j]
                    lstate[0, j] = 0.0
                    broke[j] = 1.0
                    changed = True
        if not changed:
            break
    if not sticks:
        # A node that sticks fixes the slide; the Newton pass has found one.
        _full_slip_slide(z0, v, nsol, meff, rho, gnode, khat, lam)
    w_be = 0.0
    diss = 0.0
    rem = 0.0
    n_prev = nsol[n - 1]
    for j in range(n):
        nj = nsol[j]
        bj = meff[j] * (rho * gnode[j] + lam * 0.5 * (nj + n_prev))
        if bj < 0.0:
            bj = 0.0
        bend[j] = bj
        vj = v[j]
        t = khat * (z0[j] + vj)
        slipped = False
        if t > bj:
            t = bj
            slipped = True
        elif t < -bj:
            t = -bj
            slipped = True
        unew = u[j] + vj
        dp = (unew - t * ik) - p[j]
        diss += h * t * dp
        ej = enew[j]
        w_be += h * nj * (ej - e[j])
        dn = nj - nf[j]
        dt = t - tau[j]
        rem += 0.5 * h * dn * dn + 0.5 * h * dt * dt * ik
        u[j] = unew
        p[j] += dp
        tau[j] = t
        nf[j] = nj
        e[j] = ej
        vlast[j] = vj
        n_prev = nj
        if law_on:
            if slipped or broke[j] != 0.0:
                lstate[0, j] = 0.0
                lstate[1, j] = 0.0
                mfac[j] = law[2]
            elif lstate[0, j] == 0.0:
                lstate[1, j] += law[4]
                if lstate[1, j] >= law[3]:
                    lstate[0, j] = 1.0
                    mfac[j] = law[1]
    energy[0] += w_be
    energy[1] += diss
    energy[2] += rem
    return True


@njit(cache=True)
def _fixed_point(
    y: FloatArray,
    z0: FloatArray,
    rho: float,
    gnode: FloatArray,
    khat: float,
    lam: float,
    h: float,
    tol: float,
    v: FloatArray,
    b: FloatArray,
    d: FloatArray,
    rhs: FloatArray,
    dv: FloatArray,
    cp: FloatArray,
    wy: FloatArray,
    wz: FloatArray,
    vn: FloatArray,
    vt: FloatArray,
    ncur: FloatArray,
    stats: NDArray[np.int64],
    mfac: FloatArray,
) -> bool:
    """Fixed point on the bound of :func:`_solve_increment` (the reference
    scheme; used for ``lam = 0``, ``capstan_newton=False`` and fall-backs)."""
    n = y.size
    for it in range(_MAX_FIXED_POINT):
        for j in range(n):
            jm = j - 1 if j > 0 else n - 1
            b[j] = mfac[j] * (rho * gnode[j] + lam * 0.5 * (ncur[j] + ncur[jm]))
            if b[j] < 0.0:
                b[j] = 0.0
        iters = _solve_fixed_bound(
            y, z0, b, khat, h, tol, v, vn, vt, dv, d, rhs, cp, wy, wz
        )
        if iters < 0:
            return False
        if iters > stats[1]:
            stats[1] = iters
        stats[4] += iters
        diff = 0.0
        big = 1.0
        for j in range(n):
            jp = j + 1 if j + 1 < n else 0
            n_new = y[j] + (v[jp] - v[j]) / h
            if abs(n_new - ncur[j]) > diff:
                diff = abs(n_new - ncur[j])
            if abs(n_new) > big:
                big = abs(n_new)
            ncur[j] = n_new
        if not math.isfinite(diff):
            return False
        if lam == 0.0 or diff <= 1e-12 * big:
            if it + 1 > stats[2]:
                stats[2] = it + 1
            stats[5] += it + 1
            return True
    return False


@njit(cache=True)
def _distributed_kernel_fast(
    ax_path: FloatArray,
    ay_path: FloatArray,
    rho_path: FloatArray,
    khat: float,
    lam: float,
    gnode: FloatArray,
    phase0: float,
    xi: float,
    max_increment: float,
    wire_index: NDArray[np.int64],
    wire_frac: FloatArray,
    record_field: bool,
    capstan_newton: bool = True,
    record_index: NDArray[np.int64] | None = None,
    kinematics: FloatArray | None = None,
    law: FloatArray | None = None,
    dt_path: FloatArray | None = None,
    ptab: FloatArray | None = None,
    p_ref: float = 0.0,
) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.int64], FloatArray]:
    """:func:`_distributed_kernel` with the fast increment solver (same
    arguments and results, to round-off)."""
    n = gnode.size
    if n < 3:
        return _distributed_kernel(
            ax_path,
            ay_path,
            rho_path,
            khat,
            lam,
            gnode,
            phase0,
            xi,
            max_increment,
            wire_index,
            wire_frac,
            record_field,
            capstan_newton,
            record_index,
            kinematics,
            law,
            dt_path,
            ptab,
            p_ref,
        )
    h = 2.0 * math.pi / n
    n_t = ax_path.size
    n_w = wire_index.size
    cs = np.empty(n)
    sn = np.empty(n)
    for j in range(n):
        cs[j] = math.cos(phase0 + j * h)
        sn[j] = math.sin(phase0 + j * h)
    u = np.zeros(n)
    p = np.zeros(n)
    nf = np.zeros(n)
    tau = np.zeros(n)
    e = np.zeros(n)
    vlast = np.zeros(n)
    su = np.empty(n)
    sp = np.empty(n)
    snf = np.empty(n)
    stau = np.empty(n)
    se = np.empty(n)
    svlast = np.empty(n)
    work = np.zeros((22, n))
    mat = np.full((_M_ROWS, n), np.nan)
    mstate = np.zeros(6)
    mstate[1] = n - 1
    table = np.zeros((2, 0))
    if ptab is not None:
        table = ptab
    mfac = np.ones(n)
    smfac = np.ones(n)
    lstate = np.zeros((2, n))
    slstate = np.zeros((2, n))
    lparams = np.zeros(5)
    if law is not None:
        lparams[0] = 1.0
        lparams[1] = law[0]
        lparams[2] = law[1]
        lparams[3] = law[2]
        for j in range(n):
            mfac[j] = law[0]
            lstate[0, j] = 1.0
    energy = np.zeros(3)
    senergy = np.zeros(3)
    stats = np.zeros(7, dtype=np.int64)
    out = np.empty((n_w, n_t))
    hist = np.empty((n_t, 4))
    slip = np.empty(n_t)
    fld = np.empty((n_t if record_field else 0, n))
    ax0 = 0.0
    ay0 = 0.0
    rho0 = rho_path[0] if n_t > 0 else 0.0
    ik = 1.0 / khat
    for i in range(n_t):
        ax1 = ax_path[i]
        ay1 = ay_path[i]
        rho1 = rho_path[i]
        step = math.sqrt((ax1 - ax0) ** 2 + (ay1 - ay0) ** 2) * xi
        nsub = 1
        if max_increment > 0.0 and step > max_increment:
            nsub = int(math.ceil(step / max_increment))
        for j in range(n):
            su[j] = u[j]
            sp[j] = p[j]
            snf[j] = nf[j]
            stau[j] = tau[j]
            se[j] = e[j]
            svlast[j] = vlast[j]
        for k in range(3):
            senergy[k] = energy[k]
        for j in range(n):
            smfac[j] = mfac[j]
            slstate[0, j] = lstate[0, j]
            slstate[1, j] = lstate[1, j]
        ok = False
        for _attempt in range(_MAX_HALVINGS + 1):
            ok = True
            if dt_path is not None:
                lparams[4] = dt_path[i] / nsub
            for s in range(1, nsub + 1):
                f = s / nsub
                if not _advance_fast(
                    ax0 + f * (ax1 - ax0),
                    ay0 + f * (ay1 - ay0),
                    rho0 + f * (rho1 - rho0),
                    cs,
                    sn,
                    gnode,
                    khat,
                    lam,
                    h,
                    u,
                    p,
                    nf,
                    tau,
                    e,
                    energy,
                    work,
                    stats,
                    capstan_newton,
                    mfac,
                    lparams,
                    lstate,
                    table,
                    p_ref,
                    vlast,
                    mat,
                    mstate,
                ):
                    ok = False
                    break
            if ok:
                break
            for j in range(n):
                u[j] = su[j]
                p[j] = sp[j]
                nf[j] = snf[j]
                tau[j] = stau[j]
                e[j] = se[j]
                vlast[j] = svlast[j]
            for k in range(3):
                energy[k] = senergy[k]
            for j in range(n):
                mfac[j] = smfac[j]
                lstate[0, j] = slstate[0, j]
                lstate[1, j] = slstate[1, j]
            nsub *= 2
        if not ok:
            stats[0] = -(i + 1)
            return out, hist, slip, stats, fld
        stats[3] += nsub
        ax0 = ax1
        ay0 = ay1
        rho0 = rho1
        stored = 0.0
        n_slip = 0
        for j in range(n):
            jm = j - 1 if j > 0 else n - 1
            stored += 0.5 * h * (nf[j] * nf[j] + tau[j] * tau[j] * ik)
            if law is None:
                bj = rho1 * gnode[j] + lam * 0.5 * (nf[j] + nf[jm])
                if bj < 0.0:
                    bj = 0.0
            else:
                bj = work[19, j]  # bound of the last increment
            if bj > 0.0 and abs(tau[j]) >= bj * (1.0 - 1e-9):
                n_slip += 1
        hist[i, 0] = energy[0]
        hist[i, 1] = energy[1]
        hist[i, 2] = stored
        hist[i, 3] = energy[2]
        slip[i] = n_slip / n
        for k in range(n_w):
            j0 = wire_index[k]
            j1 = j0 + 1 if j0 + 1 < n else 0
            out[k, i] = (1.0 - wire_frac[k]) * nf[j0] + wire_frac[k] * nf[j1]
        if record_field:
            for j in range(n):
                fld[i, j] = nf[j]
        if record_index is not None and kinematics is not None:
            row = record_index[i]
            if row >= 0:
                for j in range(n):
                    kinematics[row, 0, j] = u[j]
                    kinematics[row, 1, j] = p[j]
                    kinematics[row, 2, j] = p[j] - sp[j]
                    kinematics[row, 3, j] = nf[j]
    return out, hist, slip, stats, fld


# --------------------------------------------------------------------------- #
# Physical wrapper                                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DistributedFrictionHistory:
    """Wire friction stress and energy of the distributed model.

    ``stress_Pa`` has shape (wires, samples). Energies are cumulative from the
    stress-free start, per metre of cable, for the whole layer:
    ``endpoint_work - dissipation - stored - integration_remainder =
    balance_residual`` (round-off). ``slip_fraction`` is the fraction of helix
    nodes at the friction bound after each sample.

    ``kinematic_field`` (kept samples, 4, N) is the helix field at the samples
    ``kinematic_samples`` (``field_samples``), normalised: the nodal
    displacement ``u`` of the wire from its plane-section position, the nodal
    slip ``p`` (the Coulomb part of ``u``; ``u - p`` is the elastic shear of
    the interface), the slip of the step to the sample, and the segment force
    ``n``. Node ``j`` sits at the azimuth ``(j - 1/2) h``, segment ``j`` at
    ``j h``. ``parameters["length_unit_m"]`` turns ``u`` and ``p`` into metres
    along the wire (positive where the azimuth increases along it) and
    ``parameters["stress_unit_Pa"]`` turns ``n`` into the wire stress. It is
    ``None`` where the bound is zero (no friction, no clamping or no interface
    stiffness): the wire then carries no friction stress and slides freely.
    """

    stress_Pa: FloatArray
    dissipation_J_per_m: FloatArray
    stored_J_per_m: FloatArray
    endpoint_work_J_per_m: FloatArray
    integration_remainder_J_per_m: FloatArray
    balance_residual_J_per_m: FloatArray
    slip_fraction: FloatArray
    parameters: dict[str, Any] = field(default_factory=dict)
    field_normalised: FloatArray | None = None
    kinematic_samples: NDArray[np.int64] | None = None
    kinematic_field: FloatArray | None = None


def _validated_nodes(spec: HelicalLayerSpec, nodes_per_pitch: int | None) -> int:
    if nodes_per_pitch is None:
        return DEFAULT_NODES_PER_WIRE * int(spec.n_wires)
    if (
        isinstance(nodes_per_pitch, (bool, np.bool_))
        or not isinstance(nodes_per_pitch, (int, np.integer))
        or nodes_per_pitch < 8
    ):
        raise CableInputError("nodes_per_pitch must be an integer of at least 8")
    return int(nodes_per_pitch)


def _validated_increment(max_increment: float) -> float:
    value = float(max_increment)
    if math.isnan(value) or value <= 0.0:
        raise CableInputError("max_increment must be positive (inf disables substeps)")
    return value


def distributed_friction_history(
    spec: HelicalLayerSpec,
    kappa_x: FloatArray,
    kappa_y: FloatArray,
    *,
    contact_pressure_Pa: FloatArray | None = None,
    stick_modulus_Pa: FloatArray | None = None,
    nodes_per_pitch: int | None = None,
    max_increment: float = DEFAULT_MAX_INCREMENT,
    record_field: bool = False,
    field_samples: NDArray[np.int64] | None = None,
    friction_law: FrictionLaw | None = None,
    time_s: FloatArray | None = None,
    bound: str = "signed",
) -> DistributedFrictionHistory:
    """Friction stress of every wire from the distributed helix model [Pa].

    Parameters
    ----------
    spec : HelicalLayerSpec
        Layer; ``xi`` (Yun) sets ``k_hat = xi / (1 - xi)``, ``mu`` and the lay
        angle set ``lam = mu |sin a|``, ``contact_pressure_Pa`` is the
        reference clamping ``P_ref``.
    kappa_x, kappa_y : numpy.ndarray
        Axial-strain-gradient curvature history [1/m], as for the direction-tied
        kernel. The state starts stress free; the step to the first sample is
        subdivided like any other.
    contact_pressure_Pa : numpy.ndarray, optional
        Clamping pressure history ``P(t)`` (``P1 + P2``) [Pa]; it scales the
        external part of the bound, ``rho = P / P_ref``.
    stick_modulus_Pa : numpy.ndarray, optional
        Accepted only when constant and equal to ``spec.k_f``: a time-varying
        interface or wire stiffness is not implemented for this model.
    nodes_per_pitch : int, optional
        Helix nodes ``N`` per pitch (default ``4 n_wires``). Wires sample the
        field by phase; multiples of ``n_wires`` avoid interpolation.
    max_increment : float, optional
        Largest curvature increment per substep in units of the slip-onset
        curvature ``kappa_on`` (default 0.1); ``inf`` uses the samples only.
    record_field : bool, optional
        Also return the normalised segment field, shape (samples, N).
    field_samples : numpy.ndarray of int, optional
        Samples (increasing indices) whose helix field of displacement, slip
        and force is kept (``kinematic_field``). The results are otherwise
        unchanged.
    friction_law : FrictionLaw, optional
        Static/dynamic friction law
        (:class:`~cablexsec.mechanics.friction_law.FrictionLaw`): every helix
        node carries its own factor ``m_s`` or ``m_d`` on the whole bound
        ``m (rho g + lam n)`` and its own stick timer. A healed node whose
        trial traction leaves its static bound breaks away and the increment
        is solved again with the dynamic factor (each node at most once per
        increment). A pressure factor multiplies each node's bound at its
        local pressure ``P g(phi)`` (``P`` the sample's clamping pressure,
        linear within substeps). Rate-dependent terms are not implemented for
        this model and are refused. ``None`` or a neutral law is the
        rate-independent model, bit for bit.
    time_s : numpy.ndarray, optional
        Sample times [s] for a finite positive healing time; each sample step
        is divided equally among its substeps.
    bound : {"signed", "uniform"}, optional
        Form of the friction bound. ``"signed"`` (default) is
        ``b = rho g + lam n`` with the capstan term on the signed bending
        force ``n`` of the wire (module notes). ``"uniform"`` drops the
        capstan term, ``b = rho g``: the linear Coulomb bound of the clamping
        alone, for sensitivity studies and for comparison with solvers that
        carry a constant bound (:mod:`cablexsec.mechanics.clamp_layer`). The
        stress unit ``sigma_0 lam`` and the slip-onset curvature are the same
        in both forms. A bound on the total wire force (bending plus tension
        share) is not offered: the clamping history ``P(t)`` that scales ``g``
        already contains the radial load of the tension share, which a
        capstan term on the tension share would count a second time.
    """
    if bound not in BOUND_FORMS:
        raise CableInputError(f"bound must be one of {BOUND_FORMS}, got {bound!r}")
    kx = np.ascontiguousarray(kappa_x, dtype=np.float64)
    ky = np.ascontiguousarray(kappa_y, dtype=np.float64)
    if (
        kx.ndim != 1
        or ky.shape != kx.shape
        or not kx.size
        or not np.isfinite(kx).all()
        or not np.isfinite(ky).all()
    ):
        raise CableInputError(
            "finite matching nonempty curvature histories are required"
        )
    count = kx.size
    n_nodes = _validated_nodes(spec, nodes_per_pitch)
    increment = _validated_increment(max_increment)
    law = active_friction_law(friction_law)
    if law is not None and law.rate_dependent:
        raise CableInputError(
            "the distributed friction model takes the static/dynamic option of a "
            "friction law only; rate-dependent terms are implemented for the "
            "direction-tied model"
        )
    steps = friction_law_time_steps(law, time_s, count)
    pressure = (
        np.full(count, float(spec.contact_pressure_Pa))
        if contact_pressure_Pa is None
        else np.ascontiguousarray(contact_pressure_Pa, dtype=np.float64)
    )
    if pressure.shape != kx.shape or not np.isfinite(pressure).all():
        raise CableInputError("pressure must be finite and match curvature histories")
    if np.any(pressure < 0.0):
        raise CableInputError("nonnegative pressure is required")
    if stick_modulus_Pa is not None:
        stick = np.asarray(stick_modulus_Pa, dtype=np.float64)
        if stick.shape != kx.shape or not np.isfinite(stick).all():
            raise CableInputError(
                "stiffness must be finite and match curvature histories"
            )
        if not np.allclose(stick, spec.k_f, rtol=1e-12, atol=0.0):
            raise CableInputError(
                "the distributed friction model requires a constant stick "
                "modulus equal to the layer's xi E cos^2(a); a varying interface "
                "or wire stiffness is not implemented for this model"
            )
    xi = float(spec.xi)
    if not math.isfinite(xi) or xi < 0.0 or xi >= 1.0:
        raise CableInputError(
            "the distributed friction model requires 0 <= xi < 1 (finite interface "
            "shear stiffness); the rigid-stick limit is not implemented"
        )
    alpha = float(spec.lay_angle_rad)
    sin_a = abs(math.sin(alpha))
    mu = float(spec.mu)
    p_ref = float(spec.contact_pressure_Pa)
    if p_ref <= 0.0:
        p_ref = float(pressure.max()) if pressure.size else 0.0
    n_w = int(spec.n_wires)
    width = 2.0 * math.pi * spec.r_helix_m / n_w
    lam = mu * sin_a
    # Capstan coefficient of the bound (zero for the uniform bound); ``lam``
    # itself also sets the stress unit sigma_0 lam of both forms.
    capstan = lam if bound == "signed" else 0.0
    zero = xi == 0.0 or mu == 0.0 or sin_a < 1.0e-12 or p_ref <= 0.0
    sigma0 = (
        0.0
        if zero
        else spec.r_helix_m
        * p_ref
        * width
        * math.cos(alpha)
        / (spec.A_wire_m2 * sin_a**2)
    )
    stress_unit = sigma0 * lam  # g_0 / A [Pa per unit n]
    params: dict[str, Any] = {
        "friction_model": "distributed",
        "nodes_per_pitch": n_nodes,
        "max_increment": increment,
        "xi": xi,
        "k_hat": math.inf if xi >= 1.0 else xi / (1.0 - xi),
        "lam": lam,
        "bound": bound,
        "reference_pressure_Pa": p_ref,
        "stress_unit_Pa": stress_unit,
    }
    if law is not None:
        params["friction_law"] = law.describe()
    if zero or stress_unit == 0.0:
        zeros = np.zeros(count)
        params.update(kappa_on=math.inf, substeps=0, max_newton=0, max_fixed_point=0)
        return DistributedFrictionHistory(
            np.zeros((n_w, count)),
            zeros,
            zeros.copy(),
            zeros.copy(),
            zeros.copy(),
            zeros.copy(),
            zeros.copy(),
            params,
            np.zeros((count, n_nodes)) if record_field else None,
        )
    k_f = float(spec.k_f)
    kappa_on = stress_unit / (k_f * spec.r_helix_m)
    kappa_c = xi * kappa_on
    khat = xi / (1.0 - xi)
    shape = np.ones(n_w) if spec.shape_g is None else np.asarray(spec.shape_g, float)
    if shape.shape != (n_w,) or not np.isfinite(shape).all() or np.any(shape < 0):
        raise CableInputError("shape_g must be finite, nonnegative and one per wire")
    h = 2.0 * math.pi / n_nodes
    # Node j sits at azimuth (j - 1/2) h; wires at 2 pi k / n_w.
    node_pos = (
        ((np.arange(n_nodes) - 0.5) * h) % (2.0 * math.pi) * (n_w / (2 * math.pi))
    )
    i0 = np.floor(node_pos).astype(np.int64) % n_w
    gnode = np.ascontiguousarray(
        (1.0 - (node_pos - np.floor(node_pos))) * shape[i0]
        + (node_pos - np.floor(node_pos)) * shape[(i0 + 1) % n_w]
    )
    seg = np.arange(n_w) * (n_nodes / n_w)
    wire_index = np.floor(seg + 1e-12).astype(np.int64) % n_nodes
    wire_frac = np.clip(seg - np.floor(seg + 1e-12), 0.0, 1.0)
    kept: NDArray[np.int64] | None = None
    if field_samples is not None:
        kept = np.asarray(field_samples, dtype=np.int64).reshape(-1)
        if kept.size and (
            np.any(np.diff(kept) <= 0) or kept[0] < 0 or kept[-1] >= count
        ):
            raise CableInputError(
                "field samples must be increasing indices of the history"
            )
    # The fast kernel when compiled; the reference kernel otherwise (it is
    # the faster of the two as Python) and on request (tests).
    kernel = _distributed_kernel_fast if _FAST_KERNEL else _distributed_kernel
    if kept is None and law is None:
        out, hist, slip, stats, fld = kernel(
            kx / kappa_c,
            ky / kappa_c,
            pressure / p_ref,
            khat,
            capstan,
            gnode,
            0.0,
            xi,
            increment,
            wire_index,
            np.ascontiguousarray(wire_frac),
            bool(record_field),
        )
        kinematics = None
    elif law is None:
        assert kept is not None
        index = np.full(count, -1, dtype=np.int64)
        index[kept] = np.arange(kept.size)
        kinematics = np.zeros((kept.size, 4, n_nodes))
        out, hist, slip, stats, fld = kernel(
            kx / kappa_c,
            ky / kappa_c,
            pressure / p_ref,
            khat,
            capstan,
            gnode,
            0.0,
            xi,
            increment,
            wire_index,
            np.ascontiguousarray(wire_frac),
            bool(record_field),
            True,
            index,
            kinematics,
        )
    else:
        index = np.full(count, -1, dtype=np.int64)
        kinematics = None
        if kept is not None:
            index[kept] = np.arange(kept.size)
            kinematics = np.zeros((kept.size, 4, n_nodes))
        out, hist, slip, stats, fld = kernel(
            kx / kappa_c,
            ky / kappa_c,
            pressure / p_ref,
            khat,
            capstan,
            gnode,
            0.0,
            xi,
            increment,
            wire_index,
            np.ascontiguousarray(wire_frac),
            bool(record_field),
            True,
            index,
            np.zeros((0, 4, n_nodes)) if kinematics is None else kinematics,
            np.array([law.static_factor, law.dynamic_factor, law.stick_time_s]),
            np.zeros(count) if steps is None else steps,
            law.pressure_table,
            p_ref,
        )
    if stats[0] != 0:
        raise CableInputError(
            "distributed friction increment did not converge at sample "
            f"{-int(stats[0]) - 1}"
        )
    a_wire = float(spec.A_wire_m2)
    g0 = stress_unit * a_wire
    ea = float(spec.E_wire_Pa) * a_wire
    energy_unit = n_w * g0 * g0 / (2.0 * math.pi * ea * math.cos(alpha))
    work, diss, stored, rem = (energy_unit * hist[:, k] for k in range(4))
    params.update(
        kappa_on=kappa_on,
        substeps=int(stats[3]),
        max_newton=int(stats[1]),
        max_fixed_point=int(stats[2]),
        newton_iterations=int(stats[4]),
        fixed_point_iterations=int(stats[5]),
        capstan_newton_fallbacks=int(stats[6]),
        energy_unit_J_per_m=energy_unit,
        # n = e + du/dphi with e E/stress_unit the plane-section wire strain
        # and ds = r dphi / |sin a|: u is in units of this length.
        length_unit_m=stress_unit * spec.r_helix_m / (float(spec.E_wire_Pa) * sin_a),
    )
    return DistributedFrictionHistory(
        stress_unit * out,
        diss,
        stored,
        work,
        rem,
        work - diss - stored - rem,
        slip,
        params,
        fld if record_field else None,
        kept,
        kinematics,
    )


def distributed_wire_friction_stress(
    spec: HelicalLayerSpec,
    kappa_x: FloatArray,
    kappa_y: FloatArray,
    *,
    contact_pressure_Pa: FloatArray | None = None,
    stick_modulus_Pa: FloatArray | None = None,
    nodes_per_pitch: int | None = None,
    max_increment: float = DEFAULT_MAX_INCREMENT,
    friction_law: FrictionLaw | None = None,
    time_s: FloatArray | None = None,
    bound: str = "signed",
) -> FloatArray:
    """Friction stress [Pa], shape (wires, samples); see
    :func:`distributed_friction_history`."""
    return distributed_friction_history(
        spec,
        kappa_x,
        kappa_y,
        contact_pressure_Pa=contact_pressure_Pa,
        stick_modulus_Pa=stick_modulus_Pa,
        nodes_per_pitch=nodes_per_pitch,
        max_increment=max_increment,
        friction_law=friction_law,
        time_s=time_s,
        bound=bound,
    ).stress_Pa
