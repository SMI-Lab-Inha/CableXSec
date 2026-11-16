# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""State- and rate-dependent friction coefficient of the armour slip kernels.

The slip kernels of the package use one rate-independent Coulomb coefficient
``mu`` per contact pair. A :class:`FrictionLaw` replaces it by an effective
coefficient ``m mu`` with a dimensionless factor ``m`` that depends on the
slip state and the slip rate of every wire (direction-tied kernel) or helix
node (distributed kernel). The factor multiplies the coefficient where the
kernel applies it: inside the capstan exponent of the direction-tied bounds
(``sigma_0 g expm1(m lam psi)`` and ``sigma_0 g (1 - exp(-m lam psi))``,
:func:`~cablexsec.mechanics.papailiou.wire_slip_stress_bounds`) and on the
whole node bound ``b = m (rho g + lam n)`` of the distributed model, where
both terms are proportional to ``mu``.

Static and dynamic coefficient
------------------------------
``static_factor`` ``m_s`` holds until a wire first slips in a stick episode,
``dynamic_factor`` ``m_d`` while it slips, and ``m_s`` again once it has
stuck for ``stick_time_s`` (the healing time ``t_stick``). Each increment is
a backward-Euler return map with the factor of the end of the step: a wire
whose trial stress leaves the interval of its current factor (``m_s`` if
healed, ``m_d`` otherwise) starts or continues a slip episode, and its stress
is returned to the interval of ``m_d``. With ``m_s > m_d`` the loop shows a
breakaway peak (the stress drops from the static to the dynamic bound in the
step of breakaway). With ``m_s < m_d`` (velocity strengthening) a trial
stress between the two bounds ends the stick episode without slip and the
wire continues elastically up to the dynamic bound. The stick timer counts
the time of the completed stick steps after the last slip step;
``stick_time_s = inf`` (default) never heals, ``0`` heals after one stick
step.

Rate dependence
---------------
At slip rate ``v`` the dynamic factor becomes ``m_d R(v)`` with

.. math::

    R(v) = 1 + c \ln(1 + v / v_0) + v / v_\eta ,

``c`` = ``rate_coefficient`` (``mu(v) = mu_d (1 + c ln(1 + v/v_0))``: the
logarithmic law with ``mu_0 = mu_d`` and absolute slope ``c mu_d``),
``v_0`` = ``rate_reference_m_per_s`` and ``v_eta`` =
``viscous_velocity_m_per_s``. The last term is a regularised (viscous)
Coulomb law ``t = b sign(v) + eta v`` with ``eta = b / v_eta``: a Perzyna
overstress proportional to the rate-independent bound ``b``, so that the
parameter is a velocity in every kernel and an interface without contact
has no viscous resistance. The return is backward Euler: the slip increment
``dp`` of a slipping wire solves

.. math::

    |\sigma^{tr}| - k\, dp = B\big(m_d R(L\, dp / \Delta t)\big),

``B(m)`` the bound of the side of the trial stress, by a safeguarded Newton
iteration; the left side falls and the right side rises with ``dp``, so the
root is unique and exact for the step (to round-off). Only ``c >= 0`` is
accepted: velocity weakening loses that uniqueness. The rate is that of the
kernel's slip variable in metres per second. The direction-tied kernel slips
in a reduced strain (the plastic part of the wire strain); it is converted
with ``L = r / |sin a|``, the wire length per radian of helix, which turns
the extreme-fibre strain ``r kappa`` into the largest slip displacement of a
freely sliding wire (at the neutral axis, ``r^2 kappa / |sin a|``). This
gives a representative, not a local, slip velocity of the wire.

Energy. The work ``f dp`` of a slipping wire splits into the rate-independent
friction work ``s B(m_d) dp`` (``s`` the slip sign) and the rate-dependent
(viscous) part ``s [B(m_d R) - B(m_d)] dp``, both non-negative. The
incremental balance ``W = U + D_f + D_v + R`` (input work, stored energy,
friction and viscous dissipation, backward-Euler remainder) holds to
round-off for every law.

Time base. A rate-dependent law, or a static/dynamic law with a finite
positive healing time, needs the sample times ``time_s`` (finite, strictly
rising). The stress-free start lies one sampling interval ``time_s[1] -
time_s[0]`` before the first sample. Entry points without a time base
refuse such a law.

Neutral parameters (``m_s = m_d = 1``, ``c = 0``, ``v_eta = inf``) leave
every kernel on its rate-independent code path, bit for bit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cablexsec._numba import njit
from cablexsec.exceptions import CableInputError

__all__ = [
    "FrictionLaw",
    "SliderHistory",
    "active_friction_law",
    "friction_law_time_steps",
    "slider_history",
]

FloatArray = NDArray[np.float64]

#: Largest exponent argument of the bounds: e^600 times any stress, slope or
#: rate scale of the kernels stays finite in binary64 (a capped bound is far
#: above every trial stress, so the bracket of the return is unchanged).
_EXP_LIMIT = 600.0
_MAX_NEWTON = 400
#: Converged residual of the return, relative to the trial stress.
_RESIDUAL = 2.0e-15
#: Converged bracket, relative to its upper end.
_BRACKET = 4.0e-16


@dataclass(frozen=True)
class FrictionLaw:
    """Static/dynamic and rate-dependent friction coefficient (module notes).

    Parameters
    ----------
    static_factor : float
        ``m_s = mu_s / mu``: factor of a healed (sticking) contact, positive.
    dynamic_factor : float
        ``m_d = mu_d / mu``: factor while the contact slips, positive.
    stick_time_s : float
        Healing time ``t_stick`` [s] of stick after which the static factor
        returns; ``inf`` (default) never heals, ``0`` heals after one stick
        step.
    rate_coefficient : float
        ``c >= 0`` of ``R(v) = 1 + c ln(1 + v/v_0) + v/v_eta``.
    rate_reference_m_per_s : float
        ``v_0 > 0`` [m/s] of the logarithmic term.
    viscous_velocity_m_per_s : float
        ``v_eta > 0`` [m/s] of the viscous term (``eta = b / v_eta``);
        ``inf`` (default) has no viscous term.
    """

    static_factor: float = 1.0
    dynamic_factor: float = 1.0
    stick_time_s: float = math.inf
    rate_coefficient: float = 0.0
    rate_reference_m_per_s: float = 1.0e-3
    viscous_velocity_m_per_s: float = math.inf

    def __post_init__(self) -> None:
        values = {}
        for name in (
            "static_factor",
            "dynamic_factor",
            "stick_time_s",
            "rate_coefficient",
            "rate_reference_m_per_s",
            "viscous_velocity_m_per_s",
        ):
            raw = getattr(self, name)
            if isinstance(raw, (bool, np.bool_)) or not isinstance(
                raw, (int, float, np.integer, np.floating)
            ):
                raise CableInputError(f"friction law {name} must be a real number")
            values[name] = float(raw)
        for name in ("static_factor", "dynamic_factor", "rate_reference_m_per_s"):
            if not math.isfinite(values[name]) or values[name] <= 0.0:
                raise CableInputError(
                    f"friction law {name} must be finite and positive"
                )
        if math.isnan(values["stick_time_s"]) or values["stick_time_s"] < 0.0:
            raise CableInputError("friction law stick_time_s must be >= 0 (or inf)")
        rate = values["rate_coefficient"]
        if not math.isfinite(rate) or rate < 0.0:
            raise CableInputError(
                "friction law rate_coefficient must be finite and >= 0 (velocity "
                "weakening makes the backward-Euler return non-unique)"
            )
        viscous = values["viscous_velocity_m_per_s"]
        if math.isnan(viscous) or viscous <= 0.0:
            raise CableInputError(
                "friction law viscous_velocity_m_per_s must be positive (or inf)"
            )
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def state_dependent(self) -> bool:
        """True when the static and dynamic factors differ."""
        return self.static_factor != self.dynamic_factor

    @property
    def rate_dependent(self) -> bool:
        """True when ``R(v)`` is not identically one."""
        return self.rate_coefficient > 0.0 or math.isfinite(
            self.viscous_velocity_m_per_s
        )

    @property
    def is_neutral(self) -> bool:
        """True when the law reproduces the rate-independent coefficient."""
        return (
            self.static_factor == 1.0
            and self.dynamic_factor == 1.0
            and not self.rate_dependent
        )

    @property
    def needs_time(self) -> bool:
        """True when the law needs the sample times of the history."""
        healing = self.state_dependent and 0.0 < self.stick_time_s < math.inf
        return self.rate_dependent or healing

    @property
    def minimum_factor(self) -> float:
        """Smallest factor the law can give (``R >= 1``)."""
        return min(self.static_factor, self.dynamic_factor)

    @property
    def inverse_viscous_velocity(self) -> float:
        """``1 / v_eta`` [s/m] (zero without a viscous term)."""
        return 1.0 / self.viscous_velocity_m_per_s

    def rate_factor(self, slip_rate_m_per_s: float | FloatArray) -> FloatArray:
        """``R(v)`` at the slip rate magnitude ``v`` [m/s]."""
        v = np.abs(np.asarray(slip_rate_m_per_s, dtype=np.float64))
        return np.asarray(
            1.0
            + self.rate_coefficient * np.log1p(v / self.rate_reference_m_per_s)
            + v * self.inverse_viscous_velocity
        )

    def describe(self) -> str:
        """Short provenance text."""
        parts = []
        if self.state_dependent or self.static_factor != 1.0:
            parts.append(
                f"static factor {self.static_factor:g}, dynamic factor "
                f"{self.dynamic_factor:g}, healing time {self.stick_time_s:g} s"
            )
        if self.rate_coefficient > 0.0:
            parts.append(
                f"logarithmic rate term c = {self.rate_coefficient:g} with "
                f"v0 = {self.rate_reference_m_per_s:g} m/s"
            )
        if math.isfinite(self.viscous_velocity_m_per_s):
            parts.append(
                f"viscous term eta = b / {self.viscous_velocity_m_per_s:g} m/s"
            )
        return "; ".join(parts) if parts else "neutral (rate-independent Coulomb)"


def active_friction_law(friction_law: FrictionLaw | None) -> FrictionLaw | None:
    """The law, or ``None`` when it is absent or neutral (checked type)."""
    if friction_law is None:
        return None
    if not isinstance(friction_law, FrictionLaw):
        raise CableInputError("friction_law must be a FrictionLaw or None")
    return None if friction_law.is_neutral else friction_law


def friction_law_time_steps(
    friction_law: FrictionLaw | None, time_s: FloatArray | None, count: int
) -> FloatArray | None:
    """Step durations [s] of a history for the law, or ``None`` when unused.

    ``dt[0] = time_s[1] - time_s[0]`` (the stress-free start one interval
    before the first sample), ``dt[i] = time_s[i] - time_s[i - 1]``.
    A law that needs time refuses a missing time base; ``time_s`` without
    such a law is refused too (it would have no effect).
    """
    law = active_friction_law(friction_law)
    needs = law is not None and law.needs_time
    if time_s is None:
        if needs:
            raise CableInputError(
                "this friction law depends on time (slip rate or healing time): "
                "give the sample times time_s"
            )
        return None
    if not needs:
        raise CableInputError(
            "time_s is used only by a rate-dependent friction law or one with a "
            "finite positive healing time"
        )
    if np.iscomplexobj(time_s):
        raise CableInputError("time_s must be real")
    t = np.asarray(time_s, dtype=np.float64)
    if t.ndim != 1 or t.size != count or count < 2:
        raise CableInputError(
            "time_s must be a 1-D history matching the curvature samples (>= 2)"
        )
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0.0):
        raise CableInputError("time_s must be finite and rise strictly")
    return np.ascontiguousarray(np.concatenate(([t[1] - t[0]], np.diff(t))))


# --------------------------------------------------------------------------- #
# Scalar return (compiled with numba when available)                          #
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _em1(y: float) -> float:
    return math.expm1(min(y, _EXP_LIMIT))


@njit(cache=True)
def _bound(
    m: float, side: float, c0: float, x0: float, c1: float, x1: float, clin: float
) -> float:
    """Bound of factor ``m``: ``c0 E(m x0) + c1 E(m x1) + clin m``.

    ``E(y) = expm1(y)`` on the rising side (``side > 0``) and
    ``-expm1(-y)`` on the falling side (signed capstan).
    """
    if side > 0.0:
        return c0 * _em1(m * x0) + c1 * _em1(m * x1) + clin * m
    return -(c0 * math.expm1(-m * x0) + c1 * math.expm1(-m * x1)) + clin * m


@njit(cache=True)
def _bound_slope(
    m: float, side: float, c0: float, x0: float, c1: float, x1: float, clin: float
) -> float:
    if side > 0.0:
        return (
            c0 * x0 * math.exp(min(m * x0, _EXP_LIMIT))
            + c1 * x1 * math.exp(min(m * x1, _EXP_LIMIT))
            + clin
        )
    return c0 * x0 * math.exp(-m * x0) + c1 * x1 * math.exp(-m * x1) + clin


@njit(cache=True)
def _slip_magnitude(
    a: float,
    k: float,
    md: float,
    side: float,
    c0: float,
    x0: float,
    c1: float,
    x1: float,
    clin: float,
    speed: float,
    rc: float,
    v0: float,
    inv_veta: float,
) -> float:
    """Slip ``dp >= 0`` of ``a - k dp = B(md R(speed dp))`` (zero if none).

    ``a`` is the trial stress magnitude, ``speed`` converts the slip
    increment into the slip rate. The residual ``g(dp) = a - k dp - B`` falls
    strictly from ``g(0) > 0`` to ``g(hi) <= 0`` at the rate-free root
    ``hi = (a - B(md)) / k``, so the root is bracketed. Newton steps are taken
    only inside the bracket and only while the bracket at least halves every
    second iteration; otherwise the step bisects. The bracket therefore
    shrinks geometrically however steep the bound is in ``dp`` (a strong
    viscous term or a short step), and the iteration ends when the residual
    is at round-off of ``a`` or the bracket at round-off of its upper end
    (the slip itself, not the rate-free slip). NaN is
    returned if neither is reached (callers refuse it).
    """
    hi = (a - _bound(md, side, c0, x0, c1, x1, clin)) / k
    if not hi > 0.0:
        return 0.0
    if rc == 0.0 and inv_veta == 0.0:
        return hi
    lo = 0.0
    x = hi
    previous = 4.0 * hi
    for _ in range(_MAX_NEWTON):
        v = speed * x
        m = md * (1.0 + rc * math.log1p(v / v0) + v * inv_veta)
        g = a - k * x - _bound(m, side, c0, x0, c1, x1, clin)
        if g > 0.0:
            if g <= _RESIDUAL * a:
                return x
            lo = x
        elif g == 0.0 or -g <= _RESIDUAL * a:
            return x
        else:  # negative, or not finite: the bound exceeds the trial stress
            hi = x
        width = hi - lo
        if width <= _BRACKET * hi:
            return lo
        slope = -k - _bound_slope(m, side, c0, x0, c1, x1, clin) * md * speed * (
            rc / (v0 + v) + inv_veta
        )
        xn = x - g / slope
        if not (lo < xn < hi) or width > 0.5 * previous:
            xn = 0.5 * (lo + hi)
        previous = width
        x = xn
    return math.nan


@njit(cache=True)
def _return_stress(
    a: float,
    md: float,
    side: float,
    c0: float,
    x0: float,
    c1: float,
    x1: float,
    clin: float,
    speed: float,
    rc: float,
    v0: float,
    inv_veta: float,
    dp: float,
) -> float:
    """Stress magnitude at the end of a slip step: the bound of the slip rate,
    never above the trial stress ``a``."""
    v = speed * dp
    m = md * (1.0 + rc * math.log1p(v / v0) + v * inv_veta)
    return min(_bound(m, side, c0, x0, c1, x1, clin), a)


def _side_bound(
    factor: FloatArray,
    side: FloatArray,
    c0: FloatArray,
    x0: FloatArray,
    c1: FloatArray,
    x1: FloatArray,
) -> FloatArray:
    """Bound of every element on its own side at its factor (NumPy).

    ``c0 E(m x0) + c1 E(m x1)`` with ``E = expm1`` on the rising side
    (``side > 0``) and ``E(y) = -expm1(-y)`` on the falling side, as
    :func:`_bound`. Only the element's own side is evaluated: at a large rate
    factor the falling bound saturates while the rising exponential would
    overflow. The rising exponent is capped at ``_EXP_LIMIT`` like the scalar
    form.
    """
    rising = side > 0.0
    up_m = np.where(rising, factor, 0.0)
    down_m = np.where(rising, 0.0, factor)
    up = c0 * np.expm1(np.minimum(up_m * x0, _EXP_LIMIT)) + c1 * np.expm1(
        np.minimum(up_m * x1, _EXP_LIMIT)
    )
    down = -(c0 * np.expm1(-down_m * x0) + c1 * np.expm1(-down_m * x1))
    return np.asarray(np.where(rising, up, down))


def _slip_magnitude_array(
    a: FloatArray,
    k: float,
    md: float,
    side: FloatArray,
    c0: FloatArray,
    x0: FloatArray,
    c1: FloatArray,
    x1: FloatArray,
    speed: float,
    rc: float,
    v0: float,
    inv_veta: float,
) -> FloatArray:
    """NumPy form of :func:`_slip_magnitude` over a population (reference).

    The same bracketed, bisection-safeguarded Newton iteration, element by
    element; an element that does not converge is NaN.
    """

    up_side = side > 0.0

    def bound(m: FloatArray) -> FloatArray:
        return _side_bound(m, side, c0, x0, c1, x1)

    def slope(m: FloatArray) -> FloatArray:
        # Each element on its own side only (the other side's exponential
        # can be far out of range at a large rate factor).
        mu = np.where(up_side, m, 0.0)
        md_ = np.where(up_side, 0.0, m)
        up = c0 * x0 * np.exp(np.minimum(mu * x0, _EXP_LIMIT)) + c1 * x1 * np.exp(
            np.minimum(mu * x1, _EXP_LIMIT)
        )
        down = c0 * x0 * np.exp(-md_ * x0) + c1 * x1 * np.exp(-md_ * x1)
        return np.asarray(np.where(up_side, up, down))

    hi = np.maximum((a - bound(np.full(a.shape, md))) / k, 0.0)
    if (rc == 0.0 and inv_veta == 0.0) or not np.any(hi > 0.0):
        return np.asarray(hi)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        lo = np.zeros_like(hi)
        x = hi.copy()
        out = np.where(hi > 0.0, np.nan, 0.0)
        previous = 4.0 * hi
        active = hi > 0.0
        for _ in range(_MAX_NEWTON):
            v = speed * x
            m = md * (1.0 + rc * np.log1p(v / v0) + v * inv_veta)
            g = a - k * x - bound(m)
            small = np.abs(g) <= _RESIDUAL * a
            done = active & small
            out = np.where(done, x, out)
            active = active & ~small
            gains = active & (g > 0.0)
            lo = np.where(gains, x, lo)
            hi = np.where(active & ~(g > 0.0), x, hi)
            width = hi - lo
            closed = active & (width <= _BRACKET * hi)
            out = np.where(closed, lo, out)
            active = active & ~closed
            if not np.any(active):
                break
            d = -k - slope(m) * md * speed * (rc / (v0 + v) + inv_veta)
            xn = x - g / d
            bisect = ~((lo < xn) & (xn < hi)) | (width > 0.5 * previous)
            xn = np.where(bisect, 0.5 * (lo + hi), xn)
            previous = np.where(active, width, previous)
            x = np.where(active, xn, x)
    return np.asarray(out)


# --------------------------------------------------------------------------- #
# Single spring-slider                                                         #
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _slider_kernel(
    drive: FloatArray,
    bound: FloatArray,
    k: float,
    dt: FloatArray,
    ms: float,
    md: float,
    t_stick: float,
    rc: float,
    v0: float,
    inv_veta: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Force, total slip, friction and viscous work, remainder of one slider."""
    n = drive.size
    force = np.empty(n)
    slip = np.empty(n)
    fric = np.empty(n)
    visc = np.empty(n)
    rem = np.empty(n)
    f = 0.0
    p = 0.0
    d_old = 0.0
    healed = True
    timer = 0.0
    wf = 0.0
    wv = 0.0
    wr = 0.0
    for i in range(n):
        b = bound[i]
        trial = f + k * (drive[i] - d_old)
        mav = ms if healed else md
        lim = mav * b
        if trial >= -lim and trial <= lim:
            fn = trial
            dp = 0.0
            if not healed:
                timer += dt[i]
                if timer >= t_stick:
                    healed = True
        else:
            side = 1.0 if trial > 0.0 else -1.0
            mag = _slip_magnitude(
                abs(trial),
                k,
                md,
                side,
                0.0,
                0.0,
                0.0,
                0.0,
                b,
                1.0 / dt[i],
                rc,
                v0,
                inv_veta,
            )
            fn = trial
            if not mag >= 0.0:
                fn = math.nan  # unresolved return: refused by the caller
            elif mag > 0.0:
                # On the end-of-step bound of the slip rate; the slip closes
                # the step.
                fn = side * _return_stress(
                    abs(trial),
                    md,
                    side,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    b,
                    1.0 / dt[i],
                    rc,
                    v0,
                    inv_veta,
                    mag,
                )
                mag = (abs(trial) - abs(fn)) / k
            dp = side * mag
            wf += md * b * mag
            wv += (abs(fn) - md * b) * mag
            healed = False
            timer = 0.0
        wr += (fn - f) * (fn - f) / (2.0 * k)
        f = fn
        p += dp
        d_old = drive[i]
        force[i] = f
        slip[i] = p
        fric[i] = wf
        visc[i] = wv
        rem[i] = wr
    return force, slip, fric, visc, rem


@dataclass(frozen=True)
class SliderHistory:
    """History of one spring-slider (:func:`slider_history`).

    ``force_N`` and ``slip_m`` (total slip) per sample; cumulative works [J]:
    ``input_work_J`` (``sum f dd``), ``stored_J`` (``f^2 / 2k``),
    ``friction_J`` (rate-independent part), ``viscous_J`` (rate-dependent
    part) and ``remainder_J`` (backward-Euler, ``sum df^2 / 2k``), with
    ``balance_J = input - stored - friction - viscous - remainder``.
    """

    force_N: FloatArray
    slip_m: FloatArray
    input_work_J: FloatArray
    stored_J: FloatArray
    friction_J: FloatArray
    viscous_J: FloatArray
    remainder_J: FloatArray
    balance_J: FloatArray


def slider_history(
    displacement_m: FloatArray,
    bound_N: float | FloatArray,
    stiffness_N_per_m: float,
    *,
    friction_law: FrictionLaw | None = None,
    time_s: FloatArray | None = None,
) -> SliderHistory:
    """One elastic spring in series with a slider under the friction law.

    The spring of stiffness ``k`` is driven by the displacement history from
    a stress-free start; the slider's rate-independent bound ``b`` (constant
    or per sample) is scaled by the law's factor, ``|f| <= m b``, with the
    viscous term ``f = m_d b sign(v) + (b / v_eta) v`` in slip. The return
    map is the one of the armour kernels (module notes), here with a bound
    linear in the factor. ``None`` is the rate-independent slider.
    """
    if np.iscomplexobj(displacement_m):
        raise CableInputError("displacement must be real")
    drive = np.ascontiguousarray(displacement_m, dtype=np.float64)
    if drive.ndim != 1 or not drive.size or not np.isfinite(drive).all():
        raise CableInputError("displacement must be a finite nonempty 1-D history")
    bound = np.ascontiguousarray(
        np.broadcast_to(np.asarray(bound_N, dtype=np.float64), drive.shape)
    )
    if not np.isfinite(bound).all() or np.any(bound < 0.0):
        raise CableInputError("slider bound must be finite and non-negative")
    k = float(stiffness_N_per_m)
    if not math.isfinite(k) or k <= 0.0:
        raise CableInputError("slider stiffness must be finite and positive")
    law = active_friction_law(friction_law)
    steps = friction_law_time_steps(law, time_s, drive.size)
    if law is None:
        law = FrictionLaw()
    dt = np.ones(drive.size) if steps is None else steps
    force, slip, fric, visc, rem = _slider_kernel(
        drive,
        bound,
        k,
        dt,
        law.static_factor,
        law.dynamic_factor,
        law.stick_time_s,
        law.rate_coefficient,
        law.rate_reference_m_per_s,
        law.inverse_viscous_velocity,
    )
    if not np.isfinite(force).all():
        raise CableInputError(
            "the friction-law return did not converge (slip rate out of range)"
        )
    work = np.cumsum(force * np.diff(drive, prepend=0.0))
    stored = force**2 / (2.0 * k)
    return SliderHistory(
        force,
        slip,
        work,
        stored,
        fric,
        visc,
        rem,
        work - stored - fric - visc - rem,
    )
