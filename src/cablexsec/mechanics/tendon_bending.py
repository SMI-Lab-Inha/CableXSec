# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Local tendon bending on a bent pipe: normal and lateral curvature,
transverse slip towards the geodesic and the end-restraint layer (gap G1).

Curvature increments
--------------------
A helical tendon (lay angle ``a`` from the pipe axis, radius ``r``) on a pipe
bent to curvature ``kappa`` changes its normal (out-of-surface, weak-axis for
a flat tendon) and lateral (in-surface, strong-axis) curvature. With the wire
azimuth ``phi`` measured from the tensile extreme fibre and the projections
``p_n = kx cos(phi) + ky sin(phi)`` (normal phase) and
``p_l = -kx sin(phi) + ky cos(phi)`` (lateral phase):

* loxodrome (no transverse slip): lateral ``cos a (1 + sin^2 a) p_l``
  (Sævik 1992 Eq. 2.65; Sævik 2011 Eq. 51; Lukassen 2019 Eq. 16), normal
  ``cos^2 a cos 2a p_n`` (the curve curvature of the deformed helix, Out and
  von Morgen 1997 Eq. 41, used throughout CableXSec; Sævik's material value
  for a stuck tendon is ``cos^4 a``, Eq. 2.66, which differs by the term of
  the axial strain ``sin^2 a cos^2 a``, Skeie 2026 Eq. 15);
* geodesic (free transverse slip): lateral ``0``, normal ``3 cos^2 a p_n``
  (Sævik 1992 Eq. 2.32; Out and von Morgen Eq. 39; Lukassen Eq. 15);
* partial transverse slip ``v = eta w_g p_l`` towards the geodesic offset
  ``w_g = r^2 cos a (1 + sin^2 a) / sin^2 a`` per unit curvature (Sævik 1992
  Eq. 2.30; Feret and Bournazel 1987 Eq. 18; the geodesic map of
  :func:`~cablexsec.mechanics.mindlin_contact.transverse_geodesic_factor`):
  lateral ``(1 - eta) cos a (1 + sin^2 a)``, normal
  ``cos^2 a cos 2a + 2 eta cos^2 a (1 + sin^2 a)`` (the rotation of the
  wire tangent by ``v'`` on the cylinder adds ``sin 2a v' / r``). Both ends
  are exact: ``eta = 0`` is the loxodrome, ``eta = 1`` the geodesic.

Transverse equilibrium (field)
------------------------------
A wire held at the loxodrome needs an in-surface line load
``q = K (w_g kappa p_l)`` with ``K = (N + E I_l k^2) k^2``, ``k = sin a / r``
the wavenumber along the wire, ``N`` the wire tension and ``E I_l`` the
lateral bending stiffness (the transverse friction cell of the companion article;
Sævik 1992 Eq. 4.1; Dong et al. 2018 Eq. 31 and 2025 Eq. 62). The torsion
and normal-bending stiffness resist the move to the geodesic with
``N_r = (G J cos^2 2a + E I_n sin^2 2a) / r^2`` (Dong et al. 2018 Eq. 32), so
without friction the wire stops at ``eta_0 = (N + E I_l k^2) / (N + E I_l
k^2 + N_r)``. Friction (capacity ``c`` per unit wire length, the same
``mu`` times the face line loads as the axial friction) holds it: the
transverse slip is a Coulomb (Jenkin) element of stiffness
``K_d = (N + E I_l k^2 + N_r) k^2`` and target ``eta_0 w_g kappa p_l``.
For a monotonic curvature its amplitude is
``eta = eta_0 - c / (K_d w_g kappa)``, which is Sævik (1992) Eq. 4.1 with the
Galerkin factor ``4 / pi`` of his single sine shape replaced by the
point-wise amplitude, and ``mu`` below Sævik Eq. 4.2 for a tension-clamped
wire with ``E I_l = 0``.

A curvature gradient along the pipe ``dkappa/dx`` adds a holding load in the
other phase, ``B = (4 E I_l k^3 + 2 N k) cos a w_g dkappa/dx p_n /
|kappa|``-type term (leading order of the beam operator acting on
``w_g kappa(x) sin(k s)``); it is what puts transverse motion, and lateral
bending, at the bending-plane fibres of a bellmouth (Gjøsteen et al. 2027,
station 3.1).

Vector Coulomb relief of the axial friction
-------------------------------------------
While the wire slides axially, the friction vector has magnitude ``c`` and
must carry the transverse load ``q``: the axial friction available is
``sqrt(c^2 - q^2)``. When the wire slides transversely as well, the axial
share is at least ``1 / sqrt(1 + R_g^2)`` with ``R_g = (1 + sin^2 a) /
(sin a cos a)`` the frictionless transverse-to-axial slip ratio (the wire
cannot move faster than the geodesic). :class:`LateralRelief` turns the
largest demand ratio of a history into a bound factor on the axial friction,

    g_rel = max(sqrt(1 - rho^2), 1 / sqrt(1 + R_g^2)),
    rho = min(1, (|A sin(theta)| + |B cos(theta)|) / c),

applied for the whole history (the most favourable relief for the history;
an upper estimate of the relief, never a lower one), and within the
end-restraint layer of length ``L_b`` (friction fully mobilised transversely,
Dong et al. 2018) ``g_rel`` takes its floor.

End restraint (Dong et al. 2018, 2025)
--------------------------------------
At an anchor the tendon cannot move transversely (``y = y' = 0``). The
lateral bending then rises in a layer of decay length
``1 / lambda = sqrt(E I_l / N*)``; with friction the layer ends at the stick
boundary ``L_b``. :func:`dong_anchor_tension` is the closed form of Dong et
al. (2018) for tension; :func:`dong_anchor_bending` solves the boundary-value
problem of Dong et al. (2025, Eqs. 62, 76-78) for tension and uniform
bending. Both reproduce the published examples (tests).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.mindlin_contact import transverse_geodesic_factor
from cablexsec.mechanics.tendon_section import WireSection

__all__ = [
    "LateralRelief",
    "TendonCurvature",
    "TransverseSlipModel",
    "curvature_coefficients",
    "dong_anchor_bending",
    "dong_anchor_tension",
    "geodesic_slip_ratio",
    "saevik_transverse_fraction",
]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TendonCurvature:
    """Curvature-change coefficients per unit pipe curvature.

    ``normal`` multiplies the normal projection ``p_n``, ``lateral`` the
    lateral projection ``p_l`` (module notes), ``torsion`` the normal
    projection (Sævik 1992 Eqs. 2.64 and 2.31, interpolated; information).
    """

    normal: float
    lateral: float
    torsion: float


def curvature_coefficients(
    lay_angle_rad: float, transverse_fraction: float = 0.0
) -> TendonCurvature:
    """Normal, lateral and torsion coefficients at transverse slip ``eta``.

    ``eta = 0`` returns the coefficients of
    :func:`~cablexsec.mechanics.wire_stress.local_bending_fibre_stress`
    (loxodrome), ``eta = 1`` the geodesic (module notes).
    """
    eta = float(transverse_fraction)
    if not (math.isfinite(eta) and 0.0 <= eta <= 1.0):
        raise CableInputError("transverse_fraction must lie in [0, 1]")
    c, s = math.cos(lay_angle_rad), math.sin(lay_angle_rad)
    lateral_lox = c * (1 + s * s)
    normal = c * c * math.cos(2 * lay_angle_rad) + 2 * eta * c * lateral_lox
    torsion_lox = s * c**3
    torsion_geo = -s * c * (1 / (s * s) - 3) if s != 0 else 0.0
    return TendonCurvature(
        normal=normal,
        lateral=(1 - eta) * lateral_lox,
        torsion=(1 - eta) * torsion_lox + eta * torsion_geo,
    )


def geodesic_slip_ratio(lay_angle_rad: float) -> float:
    """Frictionless transverse-to-axial slip ratio ``(1 + s^2) / (s c)``."""
    s, c = abs(math.sin(lay_angle_rad)), math.cos(lay_angle_rad)
    if s <= 1e-12 or c <= 1e-12:
        raise CableInputError("the slip ratio needs 0 < lay angle < 90 deg")
    return (1 + s * s) / (s * c)


def saevik_transverse_fraction(
    mu: float,
    tension_N: float,
    E_Pa: float,
    I_lateral_m4: float,
    r_helix_m: float,
    lay_angle_rad: float,
    kappa_1_m: float,
) -> float:
    """Sævik (1992) Eq. 4.1: transverse slip over its geodesic value.

    ``u3 = [1 - (4/pi) mu Q1 R rho / (E I2 cos a (1 + sin^2 a))
    + Q1 R^2 / (E I2 sin^2 a)] / [1 + Q1 R^2 / (sin^2 a E I2)]`` with
    ``rho = 1 / kappa`` and ``I2`` the strong-axis (lateral) inertia;
    clipped to ``[0, 1]``. Valid for an outer, tension-clamped layer.
    """
    if kappa_1_m == 0:
        return 0.0
    s2 = math.sin(lay_angle_rad) ** 2
    c = math.cos(lay_angle_rad)
    ei = E_Pa * I_lateral_m4
    rho = 1.0 / abs(kappa_1_m)
    x = tension_N * r_helix_m**2 / (ei * s2)
    num = 1 - 4 / math.pi * mu * tension_N * r_helix_m * rho / (ei * c * (1 + s2)) + x
    return min(1.0, max(0.0, num / (1 + x)))


@dataclass(frozen=True)
class TransverseSlipModel:
    """Transverse (in-surface) equilibrium of one armour wire.

    Parameters
    ----------
    section : WireSection
    E_Pa, G_Pa : float
        Wire moduli.
    r_helix_m, lay_angle_rad : float
    tension_N : float
        Wire tension ``N`` (static).
    capacity_N_per_m : float
        Friction capacity per unit wire length, ``mu`` times the sum of the
        face line loads (the axial friction capacity of the same wire).
    sense : int
        ``+1``/``-1``: sign of the gradient term (lay hand and direction of
        increasing distance); ranges do not depend on it.
    """

    section: WireSection
    E_Pa: float
    G_Pa: float
    r_helix_m: float
    lay_angle_rad: float
    tension_N: float
    capacity_N_per_m: float
    sense: int = 1

    def __post_init__(self) -> None:
        for name in ("E_Pa", "G_Pa", "r_helix_m"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0):
                raise CableInputError(f"{name} must be finite and positive")
        if not 0 < abs(self.lay_angle_rad) < math.pi / 2:
            raise CableInputError("lay angle must lie in (0, 90) deg")
        for name in ("tension_N", "capacity_N_per_m"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value >= 0):
                raise CableInputError(f"{name} must be finite and non-negative")
        if self.sense not in (1, -1):
            raise CableInputError("sense must be +1 or -1")

    @property
    def wavenumber_1_m(self) -> float:
        """``k = sin a / r`` along the wire [1/m]."""
        return abs(math.sin(self.lay_angle_rad)) / self.r_helix_m

    @property
    def geodesic_offset_m2(self) -> float:
        """``w_g`` per unit curvature [m^2] (Sævik 1992 Eq. 2.30)."""
        return abs(transverse_geodesic_factor(self.r_helix_m, self.lay_angle_rad))

    @property
    def driving_force_N(self) -> float:
        """``N + E I_l k^2`` [N]: pulls the wire to the geodesic."""
        k = self.wavenumber_1_m
        return self.tension_N + self.E_Pa * self.section.I_lateral * k * k

    @property
    def resisting_force_N(self) -> float:
        """``(G J cos^2 2a + E I_n sin^2 2a) / r^2`` [N] (Dong et al. 2018 Eq. 32)."""
        a = self.lay_angle_rad
        return (
            self.G_Pa * self.section.J * math.cos(2 * a) ** 2
            + self.E_Pa * self.section.I_normal * math.sin(2 * a) ** 2
        ) / self.r_helix_m**2

    @property
    def frictionless_fraction(self) -> float:
        """``eta_0``: transverse slip of a frictionless wire over ``w_g``."""
        d = self.driving_force_N
        total = d + self.resisting_force_N
        return d / total if total > 0 else 0.0

    @property
    def stiffness_N_per_m2(self) -> float:
        """Jenkin stiffness ``K_d = (N + E I_l k^2 + N_r) k^2`` [N/m^2]."""
        k = self.wavenumber_1_m
        return (self.driving_force_N + self.resisting_force_N) * k * k

    def holding_load(
        self, kappa_lateral: FloatArray, gradient_normal: FloatArray | None = None
    ) -> FloatArray:
        """Line load [N/m] that holds the wire at the loxodrome.

        ``kappa_lateral`` is the lateral projection ``p_l`` of the curvature,
        ``gradient_normal`` the normal projection of ``dkappa/dx`` [1/m^2].
        """
        k = self.wavenumber_1_m
        w = self.geodesic_offset_m2
        q = self.driving_force_N * k * k * w * np.asarray(kappa_lateral, dtype=float)
        if gradient_normal is not None:
            ei = self.E_Pa * self.section.I_lateral
            b = (4 * ei * k**3 + 2 * self.tension_N * k) * math.cos(self.lay_angle_rad)
            q = q + self.sense * b * w * np.asarray(gradient_normal, dtype=float)
        return np.asarray(q, dtype=np.float64)

    def slip_history(
        self,
        kappa_lateral: FloatArray,
        gradient_normal: FloatArray | None = None,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Jenkin history of the transverse slip along the last axis.

        Returns ``(v, q, kappa_l)``: transverse displacement from the
        loxodrome [m], contact line load [N/m] and lateral curvature change
        ``cos a (1 + sin^2 a) p_l - k^2 v`` [1/m]. Leading axes (wires) are
        independent; the first sample starts from ``v = 0``.
        """
        q_hold = self.holding_load(kappa_lateral, gradient_normal)
        kd = self.stiffness_N_per_m2
        # Frictionless target v* = q_hold / K_d = eta_0 w_g p_l (+ gradient).
        v_star = q_hold / kd
        cap = self.capacity_N_per_m / kd
        v_star = np.atleast_1d(v_star)
        v = np.zeros_like(v_star)
        cur = np.zeros(v_star.shape[:-1])
        for j in range(v_star.shape[-1]):
            cur = np.clip(cur, v_star[..., j] - cap, v_star[..., j] + cap)
            v[..., j] = cur
        q = kd * (v_star - v)
        c, s = math.cos(self.lay_angle_rad), math.sin(self.lay_angle_rad)
        k = self.wavenumber_1_m
        kl = (
            c * (1 + s * s) * np.atleast_1d(np.asarray(kappa_lateral, float))
            - k * k * v
        )
        return v, q, kl

    def demand_ratio(
        self, kappa_amplitude: float, gradient_amplitude: float, theta: FloatArray
    ) -> FloatArray:
        """``rho(theta)`` of :class:`LateralRelief` (module notes), in [0, 1]."""
        th = np.asarray(theta, dtype=np.float64)
        k = self.wavenumber_1_m
        w = self.geodesic_offset_m2
        a = self.driving_force_N * k * k * w * abs(kappa_amplitude)
        ei = self.E_Pa * self.section.I_lateral
        b = (
            (4 * ei * k**3 + 2 * self.tension_N * k)
            * math.cos(self.lay_angle_rad)
            * w
            * abs(gradient_amplitude)
        )
        if self.capacity_N_per_m <= 0:
            return np.ones_like(th)
        demand = a * np.abs(np.sin(th)) + b * np.abs(np.cos(th))
        return np.asarray(np.minimum(1.0, demand / self.capacity_N_per_m))

    @property
    def relief_floor(self) -> float:
        """``1 / sqrt(1 + R_g^2)``: least axial share of a sliding wire."""
        return 1.0 / math.sqrt(1.0 + geodesic_slip_ratio(self.lay_angle_rad) ** 2)


@dataclass(frozen=True)
class LateralRelief:
    """Bound factor of the axial friction from the transverse load (option).

    ``factor`` returns ``g_rel(theta)`` for a curvature amplitude
    ``kappa_amplitude`` and gradient amplitude ``gradient_amplitude`` at a
    position ``distance_m`` from an anchor (module notes). ``theta`` is the
    azimuth from the tensile extreme fibre of the bending plane at angle
    ``direction_rad``. ``anchor_layer_m`` is the end-restraint layer
    (``0``: none). ``use_gradient=False`` drops the gradient term.
    """

    model: TransverseSlipModel
    anchor_layer_m: float = 0.0
    use_gradient: bool = True
    direction_rad: float = 0.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.anchor_layer_m) and self.anchor_layer_m >= 0):
            raise CableInputError("anchor_layer_m must be finite and >= 0")

    def factor(
        self,
        theta: FloatArray,
        kappa_amplitude: float | FloatArray,
        gradient_amplitude: float | FloatArray = 0.0,
        distance_m: float | FloatArray | None = None,
    ) -> FloatArray:
        th = np.asarray(theta, dtype=np.float64) - self.direction_rad
        kap = np.broadcast_to(np.asarray(kappa_amplitude, dtype=float), th.shape)
        grad = np.broadcast_to(
            np.asarray(gradient_amplitude if self.use_gradient else 0.0, dtype=float),
            th.shape,
        )
        m = self.model
        k = m.wavenumber_1_m
        w = m.geodesic_offset_m2
        a = m.driving_force_N * k * k * w * np.abs(kap)
        ei = m.E_Pa * m.section.I_lateral
        b = (4 * ei * k**3 + 2 * m.tension_N * k) * math.cos(m.lay_angle_rad) * w
        demand = a * np.abs(np.sin(th)) + b * np.abs(grad) * np.abs(np.cos(th))
        if m.capacity_N_per_m > 0:
            rho = np.minimum(1.0, demand / m.capacity_N_per_m)
        else:
            rho = np.ones_like(th)
        g = np.maximum(np.sqrt(1.0 - rho * rho), m.relief_floor)
        if distance_m is not None and self.anchor_layer_m > 0:
            d = np.broadcast_to(np.asarray(distance_m, dtype=float), th.shape)
            g = np.where(np.abs(d) < self.anchor_layer_m, m.relief_floor, g)
        return np.asarray(g, dtype=np.float64)


# --------------------------------------------------------------------------- #
# End restraint: Dong et al. (2018, 2025)                                      #
# --------------------------------------------------------------------------- #
def _dong_terms(
    section: WireSection,
    E_Pa: float,
    G_Pa: float,
    r_helix_m: float,
    lay_angle_rad: float,
    tension_N: float,
    J_m4: float | None,
) -> tuple[float, float]:
    """``N*`` (Dong 2018 Eq. 32) and ``E I_l``; Dong's angle is ``90 - a``."""
    a = lay_angle_rad
    j = section.J if J_m4 is None else J_m4
    t_star = (
        tension_N
        + G_Pa * j * math.cos(2 * a) ** 2 / r_helix_m**2
        + E_Pa * section.I_normal * math.sin(2 * a) ** 2 / r_helix_m**2
    )
    return t_star, E_Pa * section.I_lateral


def dong_anchor_tension(
    section: WireSection,
    E_Pa: float,
    G_Pa: float,
    r_helix_m: float,
    lay_angle_rad: float,
    pipe_strain: float,
    mu_inner: float,
    mu_outer: float = 0.0,
    p_outer_Pa: float = 0.0,
    *,
    J_m4: float | None = None,
) -> dict[str, float]:
    """Lateral bending at an anchored tendon under pipe tension (Dong 2018).

    Closed form (Dong, Huang, Zhang and Liu 2018, JOMAE 140 051701, Eqs.
    31-38): ``E I_l y'''' - N* y'' = f`` with ``y(0) = y'(0) = 0`` at the
    anchor, ``y = beta s``, ``y' = beta`` and ``y'' = 0`` at the stick
    boundary ``L_b`` (found by a root search), ``f = mu_i N sin^2 a / r +
    (mu_i + mu_o) p_o b``, ``beta = eps sin a cos a``, ``N = E A eps cos^2 a``.
    Returns ``L_b`` [m along the wire], the decay length ``1 / lambda`` [m],
    the edge stress at the anchor ``E y''(0) b / 2`` [Pa] and the inputs.
    """
    from scipy.optimize import brentq  # noqa: PLC0415

    a = lay_angle_rad
    tension = E_Pa * section.area * pipe_strain * math.cos(a) ** 2
    t_star, ei = _dong_terms(section, E_Pa, G_Pa, r_helix_m, a, tension, J_m4)
    if not (t_star > 0 and ei > 0):
        raise CableInputError("the anchor layer needs a positive tension")
    lam = math.sqrt(t_star / ei)
    f = (
        mu_inner * tension * math.sin(a) ** 2 / r_helix_m
        + (mu_inner + mu_outer) * p_outer_Pa * section.surface_width_m
    )
    beta = pipe_strain * math.sin(a) * math.cos(a)
    big_b = f / (ei * lam**4)

    def constants(lb: float) -> FloatArray:
        x = lam * lb
        matrix = np.array(
            [
                [0.0, 1.0, 0.0, 1.0],
                [lam, 0.0, lam, 0.0],
                [math.sinh(x), math.cosh(x), x, 1.0],
                [lam * math.cosh(x), lam * math.sinh(x), lam, 0.0],
            ]
        )
        rhs = np.array(
            [
                -big_b,
                0.0,
                beta * lb - big_b * math.cosh(x) + 0.5 * big_b * x * x,
                beta - big_b * lam * math.sinh(x) + big_b * lam * x,
            ]
        )
        return np.asarray(np.linalg.solve(matrix, rhs))

    def curvature_end(lb: float) -> float:
        c1, c2, _c3, _c4 = constants(lb)
        x = lam * lb
        return float(lam**2 * (c1 * math.sinh(x) + (big_b + c2) * math.cosh(x) - big_b))

    if f <= 0:
        raise CableInputError("the closed form needs friction (f > 0)")
    grid = np.linspace(1e-4 / lam, 40.0 / lam, 4000)
    values = np.array([curvature_end(x) for x in grid])
    hits = np.nonzero(np.sign(values[:-1]) * np.sign(values[1:]) < 0)[0]
    if hits.size == 0:
        raise CableInputError("no stick boundary found")
    lb = float(brentq(curvature_end, grid[hits[0]], grid[hits[0] + 1], xtol=1e-14))
    _c1, c2, _c3, _c4 = constants(lb)
    curvature0 = lam**2 * c2
    return {
        "stick_boundary_m": lb,
        "decay_length_m": 1.0 / lam,
        "edge_stress_Pa": abs(E_Pa * curvature0 * section.half_width_m),
        "tension_N": tension,
        "effective_tension_N": t_star,
        "friction_N_per_m": f,
        "beta": beta,
    }


def dong_anchor_bending(
    section: WireSection,
    E_Pa: float,
    G_Pa: float,
    r_helix_m: float,
    lay_angle_rad: float,
    pipe_strain: float,
    kappa_1_m: float,
    theta0_rad: float,
    mu_inner: float,
    mu_outer: float = 0.0,
    p_outer_Pa: float = 0.0,
    *,
    J_m4: float | None = None,
    pitches: float = 1.0,
    nodes: int = 4001,
    regularisation_m: float = 1e-8,
) -> dict[str, object]:
    """Anchored tendon under tension and uniform bending (Dong et al. 2025).

    Boundary-value problem of Dong, Zhang, Huang et al. (2025, Mar. Struct.
    102 103761), Eqs. 62 and 76-78, in their notation with ``alpha_0 = 90 deg
    - a``: ``E I_l y'''' - N* y'' + f (y - Y)/sqrt((y - Y)^2 + S1^2) = C2
    sin(k s + theta0)``, ``y(0) = y'(0) = 0``, ``y(L) = Y(L)``, ``y''(L) =
    Y''(L)``, solved over ``pitches`` wire pitches with
    :func:`scipy.integrate.solve_bvp`. ``theta0`` is the wire's phase at the
    anchor in Dong's convention (``pi`` = the extreme fibre of his Table 1
    case). Returns the lateral curvature change along the wire, the stress
    ``E dk2 (-b/2)`` at the edge ``x3 = -b/2`` [Pa] (Dong's Eq. 80 sign) and
    its anchor value.
    """
    from scipy.integrate import solve_bvp  # noqa: PLC0415

    a0 = math.pi / 2 - lay_angle_rad
    sa, ca = math.sin(a0), math.cos(a0)
    k = ca / r_helix_m
    length = pitches * 2 * math.pi / k
    t00 = E_Pa * section.area * pipe_strain * sa * sa
    t_star, ei = _dong_terms(section, E_Pa, G_Pa, r_helix_m, lay_angle_rad, t00, J_m4)
    f = (
        mu_inner * t00 * ca * ca / r_helix_m
        + (mu_inner + mu_outer) * p_outer_Pa * section.surface_width_m
    )
    c2 = t00 * kappa_1_m * sa
    eps, kap, th0, r = pipe_strain, kappa_1_m, theta0_rad, r_helix_m

    def big_y(s: FloatArray) -> FloatArray:
        return np.asarray(
            sa * ca * (eps * s - kap * r / k * (np.sin(k * s + th0) - math.sin(th0)))
        )

    def big_y2(s: FloatArray | float) -> FloatArray:
        return np.asarray(sa * ca * kap * r * k * np.sin(k * np.asarray(s) + th0))

    def slip1(s: FloatArray) -> FloatArray:
        return np.asarray(kap * r * sa * sa / k * (np.sin(k * s + th0) - math.sin(th0)))

    def rhs(s: FloatArray, y: FloatArray) -> FloatArray:
        d = y[0] - big_y(s)
        sgn = d / np.sqrt(d * d + slip1(s) ** 2 + regularisation_m**2)
        y4 = (t_star * y[2] - f * sgn + c2 * np.sin(k * s + th0)) / ei
        return np.vstack((y[1], y[2], y[3], y4))

    def bc(ya: FloatArray, yb: FloatArray) -> FloatArray:
        return np.array(
            [ya[0], ya[1], yb[0] - big_y(np.array(length)), yb[2] - big_y2(length)]
        )

    s = np.linspace(0.0, length, nodes)
    guess = np.vstack(
        (
            big_y(s),
            sa * ca * (eps - kap * r * np.cos(k * s + th0)),
            big_y2(s),
            np.zeros_like(s),
        )
    )
    sol = solve_bvp(rhs, bc, s, guess, tol=1e-6, max_nodes=200000)
    if sol.status != 0:
        raise CableInputError(f"Dong (2025) boundary-value problem: {sol.message}")
    y = sol.sol(s)
    dk2 = -y[2] - kap * sa * np.sin(k * s + th0)
    stress = E_Pa * dk2 * (-section.half_width_m)
    return {
        "s_m": s,
        "lateral_curvature_1_m": np.asarray(dk2),
        "edge_stress_Pa": np.asarray(stress),
        "anchor_edge_stress_Pa": float(stress[0]),
        "transverse_slip_m": np.asarray(y[0] - big_y(s)),
        "effective_tension_N": t_star,
        "friction_N_per_m": f,
        "decay_length_m": math.sqrt(ei / t_star),
    }
