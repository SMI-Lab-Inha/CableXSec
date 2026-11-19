# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Fretting regime, dissipated energy, wear and fatigue knock-down per contact.

Each contact (armour crossing, armour on a polymer bedding or sheath,
conductor trellis point, screen wire on tape) is described by a
:class:`FrettingContact`: its friction coefficient, normal load, Hertz
half-width, peak pressure and the *transition amplitude* ``delta_t`` at which
the tangential load reaches ``mu P``. A relative tangential displacement
history (the kinematic slip of the existing contact maps) is rainflow counted;
every counted cycle of amplitude ``delta_a`` is then classified and given a
dissipated energy.

Contact laws
------------
``"mindlin"`` (point contact, compliance of the Hertz contact itself)
    Cattaneo--Mindlin sphere ([Johnson1985] eqs. 7.39, 7.42, 7.60;
    [Fouvry1995] eqs. 3, 10, 19-31). With ``K_t = 8 G* a`` (``1/G* = sum
    (2 - nu_i)/G_i``) the transition amplitude is ``delta_t = 3 mu P/(2 K_t)``
    (Fouvry's ``K1 mu P / a``). For ``delta_a <= delta_t`` the load ratio is
    ``x = Q*/(mu P) = 1 - (1 - delta_a/delta_t)^{3/2}`` and the energy per
    cycle is

    .. math::

        W = \tfrac{24}{5}\,\mu P\,\delta_t\, f(x), \quad
        f = 1 - (1-x)^{5/3} - \tfrac{5x}{6}\,[1 + (1-x)^{2/3}]
          = \tfrac16 (1-y)^3 (1 + 3y + y^2),\ y = (1-x)^{1/3},

    the second form being an exact rearrangement free of cancellation. In
    gross slip the Masing loop gives ``W = 4 mu P delta_a - 3.2 mu P
    delta_t``. The Fouvry ratios follow: energy ratio ``A = W/(4 delta_a
    Q*)`` with ``A_t = 0.2``, sliding ratio ``D = delta_0/delta_a`` with
    ``D_t = 2 (1/2)^{2/3} - 1 = 0.26`` and ``B = A/D`` with ``B_t = 0.77``
    ([Fouvry1995] eqs. 23-31, 44-46; called A, B, C in [Fouvry1996]).

``"linear_support"`` (line contact on a compliant support)
    A helical wire on a polymer bedding or sheath, or a screen wire on a
    tape. Its tangential compliance is that of the support, as in the section
    model: the elastic-Coulomb law with line stiffness ``k_G = EA q^2
    xi/(1 - xi)`` (``q = sin(alpha)/r_h``, Yun ``xi``). The tangential load
    per unit length is ``Q = k_G delta`` up to the friction limit, so ``x =
    delta_a/delta_t`` with ``delta_t = (mu q)_total / k_G``, the faces of one
    wire sharing the traction in proportion to their friction limits. The
    two-dimensional tangential displacement of a line contact depends on a
    reference point ([Johnson1985] p. 214), so no Hertz compliance enters the
    regime boundary. Inside the contact the Cattaneo strip has stick zone
    ``c/b = (1 - x)^{1/2}`` ([Johnson1985] eq. 7.29). The micro-slip energy
    per unit length and cycle is derived here from that traction with the
    half-plane surface-displacement integral ([Johnson1985] eq. 2.25b,
    similar-material form, normal-tangential coupling neglected) and the
    Masing rule:

    .. math::

        W_{CM} = \frac{32}{\pi^2}\frac{(\mu q)^2}{E^*} I(\gamma),\quad
        I = \int_\gamma^1 \sqrt{1-\xi^2}\,
        [\xi\sqrt{\xi^2-\gamma^2} - \gamma^2
        \ln\tfrac{\xi + \sqrt{\xi^2-\gamma^2}}{\gamma}]\,d\xi ,

    with ``gamma = c/b`` and ``I(0) = pi/16``. In gross slip the sliding part
    ``4 mu q (delta_a - delta_t)`` is added. It is independent of the wire
    radius and small; the regime is set by ``delta_t``.

Regimes and two-dimensional paths
---------------------------------
A cycle is *gross slip* when ``delta_a > delta_t`` (for the Mindlin law this
is ``A > 0.2``), *partial slip* when ``0 < delta_a <= delta_t``, and *stick*
when the contact cannot slip at all (``mu = inf``) or the cycle amplitude is
zero. The vocabulary follows [VingsboSoderberg1988] and [ZhouVincent1995];
the mixed regime of [Fouvry1996] (a contact whose regime changes during the
life) is flagged when both partial- and gross-slip cycles carry at least
``mixed_fraction`` of the counted cycles.

A planar displacement path is counted on its principal projection. For a
path that keeps one direction (the loxodromic and geodesic maps of
:func:`cablexsec.mechanics.mindlin_contact.slip_increment`) this is exact.
Otherwise each cycle's amplitude is ``delta* = half the largest chord`` of
the path between its two turning points
(:func:`cablexsec.mechanics.mindlin_contact._trajectory_diameter`), and the
energy is the uniaxial value at that amplitude. Exact dissipation on rotating
paths is path dependent ([Ciavarella2014]: up to about 15% more for a
circular load at low ``Q/(mu P)``; memory rules of [AleshinBouMatar2016]);
it is not computed here. Both amplitudes are invariant under a rotation of
the path.

Wear and fatigue
----------------
Wear volume follows the energy approach ``V = alpha sum W``
([Fouvry1996]; [LiskiewiczFouvry2005] eq. 4) with the material-pair
coefficient of :mod:`cablexsec.materials.wear`, by default only for
gross-slip cycles (partial-slip wear is negligible, [Fouvry1996] Section 3.1).
The reported depth is the mean over the swept area of each cycle: ``2 (b +
delta_a)`` per unit length for a line contact and ``pi r^2 + 4 r delta_a``
with ``r = sqrt(a b)`` for a point contact. The contact geometry is not
updated as it wears; the unworn width gives the larger depth rate.

The fatigue knock-down multiplies the stress range of each counted stress
cycle at the contact fibre by a regime factor (:class:`FrettingKnockdown`),
the regime being that of the largest slip amplitude within the cycle's
window. Partial slip carries the Nishioka--Hirakawa fretting-fatigue factor
([Nakazawa1994] Eq. 1, with the slip saturation and an assumed slip
constant), gross slip is wear dominated (cracks are worn away,
[VingsboSoderberg1988] Fig. 10; [Fouvry1996] Section 3.1), optionally with
the stress increase of the worn wire section. The crossed-wire wear test of
[Poon2024] (data of [Cruzado2013]) is reproduced in volume (gate WEAR-CW).

References
----------
[Johnson1985] eqs. 2.25, 7.29, 7.39, 7.42, 7.60, 7.63; [Fouvry1995];
[Fouvry1996]; [LiskiewiczFouvry2005]; [VingsboSoderberg1988];
[ZhouVincent1995]; [Ciavarella2014]; [AleshinBouMatar2016]; [Nakazawa1994];
[Poon2024]; [Cruzado2013].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cablexsec._numba import njit
from cablexsec.analysis.fast_rainflow import (
    _curve_parameters,
    _cycles_to_failure,
    rainflow_table,
)
from cablexsec.exceptions import CableInputError
from cablexsec.materials.sn_curves import SNcurve
from cablexsec.materials.wear import FrettingKnockdown, WearCoefficient
from cablexsec.mechanics.hertz_contact import (
    reduced_modulus,
    reduced_shear_modulus,
    solve_contact,
)
from cablexsec.mechanics.mindlin_contact import _trajectory_diameter

__all__ = [
    "REGIMES",
    "ContactFrettingResult",
    "ContactLaw",
    "FrettingContact",
    "KnockdownResult",
    "assess_contact",
    "cattaneo_line_energy",
    "fouvry_ratios",
    "fretting_fatigue_damage",
    "line_contact",
    "linear_support_cycle_energy",
    "mindlin_cycle_energy",
    "mindlin_load_ratio",
    "mindlin_transition_amplitude",
    "point_contact",
    "range_at_cycles",
    "worn_section_fraction",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ContactLaw = Literal["mindlin", "linear_support"]
ContactKind = Literal["point", "line"]

#: Regime names indexed by the integer codes of the result arrays.
REGIMES: tuple[str, str, str] = ("stick", "partial_slip", "gross_slip")
STICK, PARTIAL, GROSS = 0, 1, 2

#: Fouvry transition values for the Mindlin sphere ([Fouvry1995] eqs. 28, 31, 46).
A_TRANSITION = 0.2
D_TRANSITION = 2.0 * 0.5 ** (2.0 / 3.0) - 1.0

# Gauss-Legendre nodes on [0, pi/2] for the strip micro-slip integral; the
# substitution xi = gamma + (1 - gamma) sin^2(theta) removes both end
# singularities, so 64 nodes reach rounding level (tests/test_fretting_regime.py).
_GL_T, _GL_W = np.polynomial.legendre.leggauss(64)
_THETA = 0.25 * math.pi * (_GL_T + 1.0)
_THETA_W = 0.25 * math.pi * _GL_W


# --------------------------------------------------------------------------- #
# Closed forms                                                                #
# --------------------------------------------------------------------------- #
def mindlin_transition_amplitude(mu: float, normal_force_N: float, K_t: float) -> float:
    """Partial/gross-slip transition amplitude ``3 mu P/(2 K_t)`` [m].

    Equal to [Fouvry1995] eq. 19, ``delta_t = K1 mu P / a`` with ``K1 = (3/16)
    sum (2 - nu_i)/G_i``, since ``K_t = 8 G* a``; and to the gross-slip
    displacement of [Johnson1985] eq. 7.42.
    """
    if not all(math.isfinite(v) for v in (normal_force_N, K_t)) or math.isnan(mu):
        raise CableInputError("transition inputs must be finite (mu may be inf)")
    if mu < 0 or normal_force_N < 0 or K_t <= 0:
        raise CableInputError("nonnegative friction and load, positive stiffness")
    if mu == math.inf:
        return math.inf
    return 1.5 * mu * normal_force_N / K_t


def mindlin_load_ratio(delta_a: ArrayLike, delta_t: float) -> FloatArray:
    """``x = Q*/(mu P) = 1 - (1 - delta_a/delta_t)^{3/2}``, capped at one."""
    d = np.asarray(delta_a, dtype=np.float64)
    if delta_t == math.inf:
        return np.zeros_like(d)
    if delta_t <= 0:
        return np.where(d > 0, 1.0, 0.0)
    r = np.minimum(d / delta_t, 1.0)
    with np.errstate(divide="ignore"):
        return np.asarray(-np.expm1(1.5 * np.log1p(-r)), dtype=np.float64)


def _mindlin_f(x: FloatArray) -> FloatArray:
    """``f(x)`` of the Mindlin--Deresiewicz energy, cancellation free."""
    with np.errstate(divide="ignore"):
        one_minus_y = -np.expm1(np.log1p(-np.minimum(x, 1.0)) / 3.0)
    y = 1.0 - one_minus_y
    return np.asarray(one_minus_y**3 * (1.0 + 3.0 * y + y * y) / 6.0)


def mindlin_cycle_energy(
    delta_a: ArrayLike, mu: float, normal_force_N: float, delta_t: float
) -> FloatArray:
    """Energy dissipated per cycle of amplitude ``delta_a`` [J].

    Partial slip: ``(24/5) mu P delta_t f(x)`` ([Fouvry1995] eq. 10 with eq.
    19; [Johnson1985] eq. 7.60 with the ``+`` sign inside the braces, the
    form consistent with Johnson's eq. 7.63). Gross slip: ``4 mu P delta_a -
    3.2 mu P delta_t`` (Masing loop of the same law). Zero for ``mu = inf``.
    """
    d = np.asarray(delta_a, dtype=np.float64)
    if mu == math.inf or delta_t == math.inf:
        return np.zeros_like(d)
    cap = mu * normal_force_N
    if delta_t <= 0:
        return np.asarray(4.0 * cap * d)
    x = mindlin_load_ratio(d, delta_t)
    partial = 4.8 * cap * delta_t * _mindlin_f(x)
    gross = 4.0 * cap * d - 3.2 * cap * delta_t
    return np.asarray(np.where(d > delta_t, gross, partial), dtype=np.float64)


def fouvry_ratios(
    delta_a: ArrayLike, delta_t: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Fouvry ratios ``(A, D, B)`` of the Mindlin law at amplitude ``delta_a``.

    ``A = W/(4 delta_a Q*)`` (energy ratio), ``D = delta_0/delta_a`` (sliding
    ratio, ``delta_0`` the loop opening at ``Q = 0``) and ``B = A/D``
    ([Fouvry1995] eqs. 23-24, 29-30, 44-45). Transition values: 0.2, 0.26,
    0.77. Zero-amplitude entries return zero.
    """
    d = np.asarray(delta_a, dtype=np.float64)
    a_ratio = np.zeros_like(d)
    d_ratio = np.zeros_like(d)
    if delta_t == math.inf or delta_t <= 0:
        moving = d > 0
        if delta_t <= 0:
            a_ratio[moving] = 1.0
            d_ratio[moving] = 1.0
        return a_ratio, d_ratio, np.where(d_ratio > 0, 1.0, 0.0)
    x = mindlin_load_ratio(d, delta_t)
    energy = mindlin_cycle_energy(d, 1.0, 1.0, delta_t)
    moving = (d > 0) & (x > 0)
    a_ratio[moving] = energy[moving] / (4.0 * d[moving] * x[moving])
    # Upper Masing branch crosses Q = 0 at delta = -delta_a + 2 v with
    # F(v) = Q*/2, i.e. v = delta_t [1 - (1 - x/2)^{2/3}].
    v = -delta_t * np.expm1((2.0 / 3.0) * np.log1p(-0.5 * x))
    d_ratio[moving] = (d[moving] - 2.0 * v[moving]) / d[moving]
    b_ratio = np.zeros_like(d)
    positive = d_ratio > 0
    b_ratio[positive] = a_ratio[positive] / d_ratio[positive]
    return a_ratio, d_ratio, b_ratio


def _strip_integral(gamma: FloatArray) -> FloatArray:
    """``I(gamma)`` of the strip micro-slip energy (module notes)."""
    g = np.asarray(gamma, dtype=np.float64)[..., None]
    s = np.sin(_THETA)
    c = np.cos(_THETA)
    span = 1.0 - g
    xi = g + span * s * s
    root = np.sqrt(np.maximum(xi * xi - g * g, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.where(
            g > 0, g * g * np.log((xi + root) / np.where(g > 0, g, 1.0)), 0.0
        )
    slip = xi * root - log_term
    pressure = np.sqrt(np.maximum(1.0 - xi * xi, 0.0))
    jac = 2.0 * span * s * c
    return np.asarray(np.sum(pressure * slip * jac * _THETA_W, axis=-1))


def cattaneo_line_energy(
    load_ratio: ArrayLike, mu: float, line_load_N_per_m: float, E_star_Pa: float
) -> FloatArray:
    """Micro-slip energy per unit length and cycle of a Cattaneo strip [J/m].

    ``(32/pi^2) (mu q)^2/E* I(gamma)`` with ``gamma = (1 - x)^{1/2}``
    (derivation in the module notes); ``(2/pi)(mu q)^2/E*`` at ``x = 1``.
    """
    x = np.minimum(np.asarray(load_ratio, dtype=np.float64), 1.0)
    if mu == math.inf:
        return np.zeros_like(x)
    gamma = np.sqrt(np.maximum(1.0 - x, 0.0))
    energy = 32.0 / math.pi**2 * (mu * line_load_N_per_m) ** 2 / E_star_Pa
    return np.asarray(np.where(x > 0, energy * _strip_integral(gamma), 0.0))


def linear_support_cycle_energy(
    delta_a: ArrayLike,
    mu: float,
    line_load_N_per_m: float,
    E_star_Pa: float,
    delta_t: float,
) -> FloatArray:
    """Energy per unit length and cycle of a line contact on a linear support.

    ``x = min(delta_a/delta_t, 1)``; ``W = W_CM(x) + 4 mu q max(delta_a -
    delta_t, 0)`` [J/m].
    """
    d = np.asarray(delta_a, dtype=np.float64)
    if mu == math.inf or delta_t == math.inf:
        return np.zeros_like(d)
    x = np.where(d > 0, 1.0, 0.0) if delta_t <= 0 else np.minimum(d / delta_t, 1.0)
    sliding = 4.0 * mu * line_load_N_per_m * np.maximum(d - max(delta_t, 0.0), 0.0)
    micro = cattaneo_line_energy(x, mu, line_load_N_per_m, E_star_Pa)
    return np.asarray(micro + sliding)


def worn_section_fraction(depth_m: ArrayLike, wire_radius_m: float) -> FloatArray:
    """Area fraction of a circular wire removed by a flat wear scar of ``depth``.

    Circular-segment area ``r^2 acos(1 - h/r) - (r - h) sqrt(2 r h - h^2)``
    over ``pi r^2``; one for ``h >= 2 r``.
    """
    h = np.clip(np.asarray(depth_m, dtype=np.float64), 0.0, 2.0 * wire_radius_m)
    r = wire_radius_m
    seg = r * r * np.arccos(1.0 - h / r) - (r - h) * np.sqrt(
        np.maximum(2.0 * r * h - h * h, 0.0)
    )
    return np.asarray(seg / (math.pi * r * r))


# --------------------------------------------------------------------------- #
# Contact description                                                         #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FrettingContact:
    """One fretting contact and the data its regime and wear need.

    Attributes
    ----------
    name : str
        Label (for example ``"outer_armour/inner_face"``).
    kind : {"point", "line"}
        Point contact (loads in N, energies in J, wear area in m^2) or line
        contact (per unit length: N/m, J/m, m).
    law : {"mindlin", "linear_support"}
        Tangential law (module notes).
    mu : float
        Friction coefficient of this contact (``inf`` = no slip possible).
    normal_load : float
        Normal force [N] or line load [N/m].
    transition_amplitude_m : float
        ``delta_t`` [m].
    half_width_m : float
        Hertz half-width (line) or equivalent radius ``sqrt(a b)`` (point) [m].
    p0_Pa, E_star_Pa : float
        Peak Hertz pressure and reduced modulus.
    wear : WearCoefficient or None
        Energy wear coefficient of the material pair; ``None`` skips wear.
    wire_radius_m : float or None
        Radius of the wire that wears, for the section-loss fraction.
    """

    name: str
    kind: ContactKind
    law: ContactLaw
    mu: float
    normal_load: float
    transition_amplitude_m: float
    half_width_m: float
    p0_Pa: float
    E_star_Pa: float
    wear: WearCoefficient | None = None
    wire_radius_m: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("point", "line") or self.law not in (
            "mindlin",
            "linear_support",
        ):
            raise CableInputError("unknown contact kind or law")
        if math.isnan(self.mu) or self.mu < 0:
            raise CableInputError("friction coefficient must be nonnegative")
        values = (self.normal_load, self.half_width_m, self.p0_Pa, self.E_star_Pa)
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise CableInputError("contact loads and dimensions must be finite, >= 0")
        if math.isnan(self.transition_amplitude_m) or self.transition_amplitude_m < 0:
            raise CableInputError("transition amplitude must be nonnegative")

    def cycle_energy(self, delta_a: ArrayLike) -> FloatArray:
        """Energy per cycle [J or J/m] at the given amplitudes."""
        if self.law == "mindlin":
            return mindlin_cycle_energy(
                delta_a, self.mu, self.normal_load, self.transition_amplitude_m
            )
        return linear_support_cycle_energy(
            delta_a,
            self.mu,
            self.normal_load,
            self.E_star_Pa,
            self.transition_amplitude_m,
        )

    def load_ratio(self, delta_a: ArrayLike) -> FloatArray:
        """``Q*/(mu P)`` at the given amplitudes."""
        d = np.asarray(delta_a, dtype=np.float64)
        dt = self.transition_amplitude_m
        if self.law == "mindlin":
            return mindlin_load_ratio(d, dt)
        if dt == math.inf:
            return np.zeros_like(d)
        if dt <= 0:
            return np.where(d > 0, 1.0, 0.0)
        return np.asarray(np.minimum(d / dt, 1.0))

    def regime(self, delta_a: ArrayLike) -> NDArray[np.int8]:
        """Regime code per amplitude (:data:`REGIMES`)."""
        d = np.asarray(delta_a, dtype=np.float64)
        code = np.full(d.shape, PARTIAL, dtype=np.int8)
        code[d > self.transition_amplitude_m] = GROSS
        if self.mu == math.inf or self.transition_amplitude_m == math.inf:
            code[:] = STICK
        code[d <= 0] = STICK
        return code

    @property
    def mean_pressure_Pa(self) -> float:
        """Mean Hertz pressure: ``(2/3) p0`` (point) or ``(pi/4) p0`` (line)."""
        return (2.0 / 3.0 if self.kind == "point" else math.pi / 4.0) * self.p0_Pa

    def swept_area(self, delta_a: ArrayLike) -> FloatArray:
        """Worn area per cycle: ``2(b + delta)`` [m] or ``pi r^2 + 4 r delta``."""
        d = np.asarray(delta_a, dtype=np.float64)
        r = self.half_width_m
        if self.kind == "line":
            return np.asarray(2.0 * (r + d))
        return np.asarray(math.pi * r * r + 4.0 * r * d)


def point_contact(
    name: str,
    normal_force_N: float,
    A_curv: float,
    B_curv: float,
    E1_Pa: float,
    nu1: float,
    E2_Pa: float,
    nu2: float,
    mu: float,
    *,
    wear: WearCoefficient | None = None,
    wire_radius_m: float | None = None,
) -> FrettingContact:
    """Crossed-wire (or trellis) point contact with the Mindlin law.

    Hertz ellipse from :func:`cablexsec.mechanics.hertz_contact.solve_contact`
    with the area-equivalent ``K_t = 8 G* sqrt(a b)``; the circular Mindlin
    closed forms are applied to that equivalent contact (an approximation for
    elongated ellipses).
    """
    e_star = reduced_modulus(E1_Pa, nu1, E2_Pa, nu2)
    g_star = reduced_shear_modulus(E1_Pa, nu1, E2_Pa, nu2)
    hz = solve_contact(normal_force_N, A_curv, B_curv, e_star, g_star)
    return FrettingContact(
        name=name,
        kind="point",
        law="mindlin",
        mu=mu,
        normal_load=normal_force_N,
        transition_amplitude_m=mindlin_transition_amplitude(
            mu, normal_force_N, hz.K_t_N_per_m
        ),
        half_width_m=math.sqrt(hz.a_m * hz.b_m),
        p0_Pa=hz.p0_Pa,
        E_star_Pa=e_star,
        wear=wear,
        wire_radius_m=wire_radius_m,
    )


def line_contact(
    name: str,
    line_load_N_per_m: float,
    wire_radius_m: float,
    E1_Pa: float,
    nu1: float,
    E2_Pa: float,
    nu2: float,
    mu: float,
    support_stiffness_N_per_m2: float,
    *,
    friction_limit_N_per_m: float | None = None,
    wear: WearCoefficient | None = None,
) -> FrettingContact:
    """Wire-on-flat line contact on a linear tangential support.

    Hertz cylinder on a flat: ``b = (4 q R/(pi E*))^{1/2}``, ``p0 = 2 q/(pi
    b)`` ([Johnson1985] eqs. 4.43, 4.45 with the flat's curvature zero).
    ``delta_t = friction_limit / k`` where the friction limit is the total
    ``sum mu q`` of all faces sharing the support (default this face's ``mu
    q``). A zero line load has no contact (stick, no energy).
    """
    if not all(math.isfinite(v) and v >= 0 for v in (line_load_N_per_m, wire_radius_m)):
        raise CableInputError("line load and wire radius must be finite and >= 0")
    if math.isnan(support_stiffness_N_per_m2) or support_stiffness_N_per_m2 <= 0:
        raise CableInputError("support stiffness must be positive (inf allowed)")
    e_star = reduced_modulus(E1_Pa, nu1, E2_Pa, nu2)
    if line_load_N_per_m == 0.0:
        return FrettingContact(
            name,
            "line",
            "linear_support",
            math.inf,
            0.0,
            math.inf,
            0.0,
            0.0,
            e_star,
            wear,
            wire_radius_m,
        )
    half = math.sqrt(4.0 * line_load_N_per_m * wire_radius_m / (math.pi * e_star))
    p0 = 2.0 * line_load_N_per_m / (math.pi * half)
    limit = (
        mu * line_load_N_per_m
        if friction_limit_N_per_m is None
        else (friction_limit_N_per_m)
    )
    if mu == math.inf or limit == math.inf:
        delta_t = math.inf
    else:
        delta_t = limit / support_stiffness_N_per_m2
    return FrettingContact(
        name=name,
        kind="line",
        law="linear_support",
        mu=mu,
        normal_load=line_load_N_per_m,
        transition_amplitude_m=delta_t,
        half_width_m=half,
        p0_Pa=p0,
        E_star_Pa=e_star,
        wear=wear,
        wire_radius_m=wire_radius_m,
    )


# --------------------------------------------------------------------------- #
# Cycle extraction                                                            #
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _window_half_range(u: FloatArray, start: IntArray, end: IntArray) -> FloatArray:
    """Half the range of ``u`` over each closed window ``[start, end]``."""
    out = np.empty(start.size)
    for k in range(start.size):
        i0 = min(start[k], end[k])
        i1 = max(start[k], end[k])
        lo = u[i0]
        hi = u[i0]
        for i in range(i0 + 1, i1 + 1):
            v = u[i]
            if v < lo:
                lo = v
            elif v > hi:
                hi = v
        out[k] = 0.5 * (hi - lo)
    return out


def _as_path(displacement: ArrayLike) -> tuple[FloatArray, FloatArray | None]:
    """Principal projection and, for a genuinely planar path, the 2-D points."""
    u = np.asarray(displacement, dtype=np.float64)
    if np.iscomplexobj(displacement) or not np.isfinite(u).all():
        raise CableInputError("displacement history must be real and finite")
    if u.ndim == 1:
        return np.ascontiguousarray(u), None
    if u.ndim != 2 or u.shape[1] != 2:
        raise CableInputError("displacement must have shape (n,) or (n, 2)")
    centred = u - u.mean(axis=0)
    _uu, sv, vt = np.linalg.svd(centred, full_matrices=False)
    projection = np.ascontiguousarray(centred @ vt[0])
    planar = sv.size > 1 and sv[1] > 1.0e-9 * max(sv[0], 1e-300)
    return projection, (u if planar else None)


def _cycle_amplitudes(
    displacement: ArrayLike,
) -> tuple[FloatArray, FloatArray, IntArray, IntArray, FloatArray, FloatArray | None]:
    """Cycle amplitudes ``delta_a``, counts, windows, projection and 2-D path."""
    projection, planar = _as_path(displacement)
    if projection.size < 2:
        empty = np.empty(0)
        idx = np.empty(0, dtype=np.int64)
        return empty, empty, idx, idx, projection, planar
    rng, _mean, count, start, end = rainflow_table(projection)
    if planar is None:
        amplitude = 0.5 * rng
    else:
        amplitude = np.array(
            [
                0.5 * _trajectory_diameter(planar[min(a, b) : max(a, b) + 1])
                for a, b in zip(start, end, strict=True)
            ],
            dtype=np.float64,
        )
    return amplitude, count, start, end, projection, planar


# --------------------------------------------------------------------------- #
# Per-contact assessment                                                      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ContactFrettingResult:
    """Regime, energy and wear of one contact over one record.

    Per-cycle arrays (``amplitude_m``, ``count``, ``regime``, ``load_ratio``,
    ``energy``, ``energy_ratio_A``) follow the rainflow order. Energies are in
    J (point) or J/m (line); ``wear_depth_m`` is the mean depth worn over the
    record. ``count_by_regime`` and ``energy_by_regime`` are indexed by
    :data:`REGIMES`.
    """

    contact: FrettingContact
    amplitude_m: FloatArray
    count: FloatArray
    regime: NDArray[np.int8]
    load_ratio: FloatArray
    energy: FloatArray
    energy_ratio_A: FloatArray
    count_by_regime: FloatArray
    energy_by_regime: FloatArray
    wear_depth_m: float
    wear_volume: float
    planar_path: bool
    mixed: bool
    dominant_regime: str = field(default="stick")

    @property
    def max_amplitude_m(self) -> float:
        """Largest cycle amplitude ``delta*`` [m]."""
        return float(self.amplitude_m.max()) if self.amplitude_m.size else 0.0

    @property
    def slip_to_width(self) -> float:
        """Largest ``delta*/half-width`` (``nan`` without a loaded contact)."""
        if self.contact.half_width_m <= 0:
            return math.nan
        return self.max_amplitude_m / self.contact.half_width_m

    @property
    def total_energy(self) -> float:
        """Dissipated energy over the record [J or J/m]."""
        return float(np.sum(self.count * self.energy))


def assess_contact(
    displacement: ArrayLike,
    contact: FrettingContact,
    *,
    partial_slip_wear: bool = False,
    mixed_fraction: float = 0.05,
) -> ContactFrettingResult:
    """Classify, dissipate and wear one contact under a displacement history.

    Parameters
    ----------
    displacement : array_like, shape (n,) or (n, 2)
        Relative tangential displacement history at the contact [m].
    contact : FrettingContact
    partial_slip_wear : bool, optional
        Also wear on partial-slip energy (default: gross slip only).
    mixed_fraction : float, optional
        Minimum share of the counted cycles in each of partial and gross slip
        for the contact to be flagged as mixed.
    """
    if not 0.0 <= mixed_fraction <= 0.5:
        raise CableInputError("mixed_fraction must lie in [0, 0.5]")
    amplitude, count, _s, _e, _p, planar = _cycle_amplitudes(displacement)
    regime = contact.regime(amplitude)
    load_ratio = contact.load_ratio(amplitude)
    energy = contact.cycle_energy(amplitude)
    energy = np.where(regime == STICK, 0.0, energy)
    q_star = load_ratio
    a_ratio = np.zeros_like(amplitude)
    moving = (amplitude > 0) & (q_star > 0) & (regime != STICK)
    cap = contact.mu * contact.normal_load
    if cap > 0 and math.isfinite(cap):
        a_ratio[moving] = energy[moving] / (
            4.0 * amplitude[moving] * q_star[moving] * cap
        )
    count_by = np.array([float(np.sum(count[regime == c])) for c in range(3)])
    energy_by = np.array(
        [float(np.sum((count * energy)[regime == c])) for c in range(3)]
    )
    wearing = regime == GROSS
    if partial_slip_wear:
        wearing = wearing | (regime == PARTIAL)
    depth = volume = 0.0
    if contact.wear is not None and np.any(wearing):
        alpha = contact.wear.alpha_m3_per_J
        w = count[wearing] * energy[wearing]
        volume = float(alpha * np.sum(w))
        depth = float(alpha * np.sum(w / contact.swept_area(amplitude[wearing])))
    total = float(np.sum(count))
    mixed = bool(
        total > 0
        and count_by[PARTIAL] >= mixed_fraction * total
        and count_by[GROSS] >= mixed_fraction * total
    )
    dominant = REGIMES[int(np.argmax(count_by))] if total > 0 else "stick"
    return ContactFrettingResult(
        contact=contact,
        amplitude_m=amplitude,
        count=count,
        regime=regime,
        load_ratio=load_ratio,
        energy=energy,
        energy_ratio_A=a_ratio,
        count_by_regime=count_by,
        energy_by_regime=energy_by,
        wear_depth_m=depth,
        wear_volume=volume,
        planar_path=planar is not None,
        mixed=mixed,
        dominant_regime=dominant,
    )


# --------------------------------------------------------------------------- #
# Fatigue knock-down                                                          #
# --------------------------------------------------------------------------- #
def range_at_cycles(sn_curve: SNcurve, cycles: float) -> float:
    """Stress range [Pa] of an S-N curve in MPa at ``cycles`` (bisection).

    Below a fatigue threshold the life is infinite; the threshold is then
    returned.
    """
    if sn_curve.stress_unit != "MPa":
        raise CableInputError("a stress-range S-N curve in MPa is required")
    if isinstance(cycles, bool) or not (math.isfinite(cycles) and cycles > 0.0):
        raise CableInputError(f"cycles must be finite and positive: {cycles!r}")
    lo, hi = -6.0, 6.0  # log10 of the range in MPa
    target = math.log10(cycles)

    def log_life(log_range: float) -> float:
        # log10 lives, which a steep curve keeps representable where the life
        # itself would overflow.
        return float(sn_curve.log10_life(np.array([10.0**log_range]))[0])

    def longer(log_range: float) -> bool:
        return log_life(log_range) > target

    # The bisection holds the range inside [1e-6, 1e6] MPa; a root outside it
    # would come back as the bound.
    if longer(hi):
        raise CableInputError(
            f"the S-N curve gives more than {cycles:g} cycles even at 1e6 MPa: "
            "no stress range of that life"
        )
    if math.isclose(log_life(lo), target, rel_tol=1e-12, abs_tol=1e-12):
        return float(10.0**lo * 1.0e6)  # the root on the bracket's lower end
    if not longer(lo):
        raise CableInputError(
            f"the S-N curve gives at most {cycles:g} cycles even at 1e-6 MPa: "
            "no stress range of that life"
        )
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if longer(mid):
            lo = mid
        else:
            hi = mid
    return float(10.0**hi * 1.0e6)


@dataclass(frozen=True)
class KnockdownResult:
    """Plain and fretting-adjusted Miner damage at one contact fibre.

    ``damage_by_regime`` splits the adjusted damage by the regime assigned to
    each stress cycle (:data:`REGIMES`). ``section_loss_factor`` is the
    stress multiplier of the worn section applied to every cycle.
    """

    damage_plain: float
    damage_fretting: float
    damage_by_regime: FloatArray
    count_by_regime: FloatArray
    section_loss_factor: float


@njit(cache=True)
def _knockdown_damage(
    rng: FloatArray,
    count: FloatArray,
    factor: FloatArray,
    m1: float,
    log_c1: float,
    m2: float,
    log_c2: float,
    knee: float,
    threshold: float,
) -> tuple[FloatArray, FloatArray]:
    plain = np.zeros(rng.size)
    adjusted = np.zeros(rng.size)
    for k in range(rng.size):
        stress = rng[k] / 1.0e6
        plain[k] = count[k] / _cycles_to_failure(
            stress, m1, log_c1, m2, log_c2, knee, threshold
        )
        adjusted[k] = count[k] / _cycles_to_failure(
            stress * factor[k],
            m1,
            log_c1,
            m2,
            log_c2,
            knee,
            threshold,
        )
    return plain, adjusted


def fretting_fatigue_damage(
    stress_Pa: ArrayLike,
    displacement: ArrayLike,
    contact: FrettingContact,
    sn_curve: SNcurve,
    knockdown: FrettingKnockdown,
    *,
    section_loss_factor: float = 1.0,
) -> KnockdownResult:
    """Miner damage at a contact fibre with regime-dependent stress factors.

    Each rainflow cycle of ``stress_Pa`` takes the regime of the largest slip
    amplitude within its window (the half range of the principal displacement
    projection, or the half chord of a planar path), and its range is
    multiplied by ``knockdown.gross_slip_factor`` (gross slip), by the
    partial-slip factor :meth:`FrettingKnockdown.notch_factor` at the
    contact's mean pressure and that window amplitude (partial slip; the
    reference range is the S-N range at ``knockdown.reference_cycles``) or by
    one (stick), and by ``section_loss_factor``. The same S-N curve is used
    for both sums.
    """
    stress = np.ascontiguousarray(stress_Pa, dtype=np.float64)
    if stress.ndim != 1 or not np.isfinite(stress).all():
        raise CableInputError("stress history must be finite and one-dimensional")
    if not math.isfinite(section_loss_factor) or section_loss_factor < 1.0:
        raise CableInputError("section_loss_factor must be finite and >= 1")
    projection, planar = _as_path(displacement)
    if projection.shape != stress.shape:
        raise CableInputError("stress and displacement histories must match")
    rng, _mean, count, start, end = rainflow_table(stress)
    if planar is None:
        window = _window_half_range(projection, start, end)
    else:
        window = np.array(
            [
                0.5 * _trajectory_diameter(planar[min(a, b) : max(a, b) + 1])
                for a, b in zip(start, end, strict=True)
            ],
            dtype=np.float64,
        )
    regime = contact.regime(window)
    factor = np.ones_like(rng)
    partial = regime == PARTIAL
    if np.any(partial):
        factor[partial] = knockdown.notch_factor(
            contact.mu,
            contact.mean_pressure_Pa,
            window[partial],
            range_at_cycles(sn_curve, knockdown.reference_cycles),
        )
    factor[regime == GROSS] = knockdown.gross_slip_factor
    factor *= section_loss_factor
    params = _curve_parameters(sn_curve)
    plain, adjusted = _knockdown_damage(rng, count, factor, *params)
    by_regime = np.array([float(np.sum(adjusted[regime == c])) for c in range(3)])
    counts = np.array([float(np.sum(count[regime == c])) for c in range(3)])
    return KnockdownResult(
        damage_plain=float(np.sum(plain)),
        damage_fretting=float(np.sum(adjusted)),
        damage_by_regime=by_regime,
        count_by_regime=counts,
        section_loss_factor=section_loss_factor,
    )
