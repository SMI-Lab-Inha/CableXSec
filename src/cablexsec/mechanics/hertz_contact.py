# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Hertzian elliptical contact for crossed wires.

Two armour wires crossing at an oblique angle form an *elliptical* Hertz
contact, not a circular one. This module solves the full elliptical contact
using Carlson elliptic integrals and elastic displacement compatibility.

Source-label note
-----------------
The project spec referred to "Table 4.1" for the ellipse aspect ratio and
"Table 7.1" for the tangential stiffness. Reading [Johnson1985] shows these
labels are imprecise: the ellipse geometry follows the eccentricity equation
(Johnson eqs 4.25-4.33, plotted in Fig. 4.4), Johnson's numbered Table 4.1 is
the sub-surface shear ratio, and the tangential stiffness is eq. (7.44)
(Mindlin), with no numbered "Table 7.1". We therefore implement the exact
elliptic-integral equations rather than splining a table, including the circular
limit. Tangential stiffness remains an area-equivalent approximation.

Geometry
--------
The undeformed gap between the bodies is ``h = A x^2 + B y^2`` with relative
principal curvatures ``A <= B`` [1/m]. The contact ellipse has semi-axes
``a >= b`` (``a`` along the low-curvature ``A`` direction), eccentricity
``e = sqrt(1 - (b/a)^2)``, fixed by the curvature ratio through

    B/A = [ (a/b)^2 E(e) - K(e) ] / [ K(e) - E(e) ]                  (Johnson 4.x)

with ``K``, ``E`` the complete elliptic integrals (modulus ``e``). The code
retains the axis ratio instead of reconstructing it from eccentricity. Size
follows the two displacement-curvature conditions directly:

    A = F_n RD(0, b^2, a^2) / (2 pi E*)
    B = F_n RD(0, a^2, b^2) / (2 pi E*)
    p0  = 3 F_n / (2 pi a b)                                          (Johnson 4.27)

For two equal cylinders of radius r, A + B = 1/r. These are the
coefficients of the quadratic gap, not the full gap Hessian.

References
----------
[Johnson1985] *Contact Mechanics*, §4.2(b) eqs (4.25)-(4.33), (4.27); §7
eq. (7.44) (tangential compliance).
[Greenwood2018] *Hertz Theory and Carlson Elliptic Integrals*, eqs (14)-(20).
https://api.repository.cam.ac.uk/server/api/core/bitstreams/
fb95720d-af5c-443b-9065-66703df5f19a/content
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq
from scipy.special import elliprd

from cablexsec.constants import TAU_MAX_OVER_P0_NU03
from cablexsec.exceptions import CableInputError

__all__ = [
    "HertzContact",
    "crossed_cylinder_curvatures",
    "layer_crossing_angle_rad",
    "layer_crossing_density_per_m",
    "reduced_modulus",
    "reduced_shear_modulus",
    "solve_contact",
]


def reduced_modulus(E1: float, nu1: float, E2: float, nu2: float) -> float:
    """Reduced (plane-strain) contact modulus ``E*`` [Pa].

    ``1/E* = (1 - nu1^2)/E1 + (1 - nu2^2)/E2``  ([Johnson1985] eq. 4.20).
    """
    _validate_materials(E1, nu1, E2, nu2)
    scale = min(E1, E2)
    compliance = (scale / E1) * (1 - nu1) * (1 + nu1) + (scale / E2) * (1 - nu2) * (
        1 + nu2
    )
    return _positive_result(scale / compliance, "reduced normal modulus")


def reduced_shear_modulus(E1: float, nu1: float, E2: float, nu2: float) -> float:
    """Reduced shear modulus ``G*`` for tangential stiffness [Pa].

    ``1/G* = (2 - nu1)/G1 + (2 - nu2)/G2`` with ``G = E / (2(1+nu))``, so that
    the circular tangential stiffness is ``K_t = 8 G* a`` ([Johnson1985] 7.44).
    """
    _validate_materials(E1, nu1, E2, nu2)
    scale = min(E1, E2)
    compliance = (scale / E1) * 2 * (2 - nu1) * (1 + nu1) + (scale / E2) * 2 * (
        2 - nu2
    ) * (1 + nu2)
    return _positive_result(scale / compliance, "reduced tangential modulus")


def _validate_materials(E1: float, nu1: float, E2: float, nu2: float) -> None:
    if not all(math.isfinite(x) for x in (E1, nu1, E2, nu2)):
        raise CableInputError("finite elastic moduli and Poisson ratios are required")
    if min(E1, E2) <= 0 or not (-1 < nu1 <= 0.5 and -1 < nu2 <= 0.5):
        raise CableInputError(
            "positive elastic moduli and -1 < Poisson ratio <= 0.5 required"
        )


def _positive_result(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise CableInputError(f"{name} exceeds the positive floating-point range")
    return value


def layer_crossing_angle_rad(alpha_inner_rad: float, alpha_outer_rad: float) -> float:
    """Angle between wires of two oppositely-laid helical layers, rad.

    Opposite-lay wires at lay angles ``alpha_in``/``alpha_out`` cross at
    ``theta = alpha_in + alpha_out`` (``cos theta = cos a_in cos a_out -
    sin a_in sin a_out``). ``theta = pi/2`` gives a perpendicular (circular)
    contact.
    """
    return alpha_inner_rad + alpha_outer_rad


def layer_crossing_density_per_m(
    n_partner: int,
    r_this_m: float,
    alpha_this_rad: float,
    lay_sign_this: float,
    r_partner_m: float,
    alpha_partner_rad: float,
    lay_sign_partner: float,
) -> float:
    """Number of crossings per metre for one wire against a partner layer.

    A helical wire advances in azimuth at
    ``omega = lay_sign * tan(alpha) / r`` [rad/m].  A wire crosses the
    uniformly phased population of ``n_partner`` wires at density
    ``n_partner * abs(omega_this - omega_partner) / (2*pi)``. Opposite lays
    therefore add their winding rates. The result converts a normal line load
    ``q_n`` [N/m] to the mean force per discrete crossing ``Q_n=q_n/lambda``.
    """
    if n_partner < 1 or r_this_m <= 0.0 or r_partner_m <= 0.0:
        raise CableInputError("positive wire count and helix radii are required")
    omega_this = lay_sign_this * math.tan(alpha_this_rad) / r_this_m
    omega_partner = lay_sign_partner * math.tan(alpha_partner_rad) / r_partner_m
    return n_partner * abs(omega_this - omega_partner) / (2.0 * math.pi)


def crossed_cylinder_curvatures(
    d_wire_m: float,
    crossing_angle_rad: float,
    *,
    partner_diameter_m: float | None = None,
) -> tuple[float, float]:
    """Quadratic gap coefficients A <= B [1/m] for crossed cylinders.

    The gap is h = x.T H x, where H is the sum of the two rotated
    matrices diag(0, 1/d). Its eigenvalues are A and B. For equal
    diameters these reduce to (1 - abs(cos(theta)))/d and
    (1 + abs(cos(theta)))/d. Unequal diameters are supported explicitly.
    """
    d2 = d_wire_m if partner_diameter_m is None else partner_diameter_m
    if not all(math.isfinite(x) for x in (d_wire_m, d2, crossing_angle_rad)):
        raise CableInputError(
            "finite cylinder diameters and crossing angle are required"
        )
    if min(d_wire_m, d2) <= 0:
        raise CableInputError("cylinder diameters must be positive")
    angle = math.remainder(crossing_angle_rad, math.pi)
    scale = min(d_wire_m, d2)
    u, v = scale / d_wire_m, scale / d2
    difference = math.hypot(u - v, 2 * math.sqrt(u * v) * math.cos(angle))
    b_scaled = (u + v + difference) / 2
    b_curv = _positive_result(b_scaled / scale, "larger curvature")
    # Determinant / larger eigenvalue, cancelling the common diameter scale.
    # Exactly parallel cylinders are line contact, with zero smaller curvature.
    a_curv = (math.sin(angle) ** 2 / max(d_wire_m, d2)) / b_scaled
    if angle != 0:
        _positive_result(a_curv, "smaller curvature")
    return a_curv, b_curv


def _log_aspect_ratio(a_curv: float, b_curv: float) -> float:
    """Solve the two curvature equations without subtracting K-E near a circle."""
    if a_curv == b_curv:
        return 0.0
    ratio = b_curv / a_curv
    log_ratio = (
        math.log(ratio) if math.isfinite(ratio) else math.log(b_curv) - math.log(a_curv)
    )

    def residual(log_aspect: float) -> float:
        q = math.exp(-2 * log_aspect)
        return math.log(elliprd(0, 1, q)) - math.log(elliprd(0, q, 1)) - log_ratio

    # q >= exp(-700) keeps both Carlson evaluations finite in binary64.
    # An aspect ratio beyond this bracket is explicitly outside this solver's
    # numerical range, rather than silently clipping eccentricity below one.
    upper = min(log_ratio, 350.0)
    if residual(upper) < 0:
        raise CableInputError(
            "contact aspect ratio exceeds the floating-point solver range"
        )
    return float(brentq(residual, 0, upper, xtol=5e-324, rtol=1e-14))


@dataclass(frozen=True)
class HertzContact:
    """Result of an elliptical Hertz contact solve.

    Attributes
    ----------
    a_m, b_m : float
        Contact-ellipse semi-axes (``a >= b``) [m].
    p0_Pa : float
        Peak contact pressure [Pa].
    eccentricity : float
        Ellipse eccentricity ``e = sqrt(1 - (b/a)^2)``. It can round to one
        for a very slender ellipse; a_m and b_m retain the resolved geometry.
    tau_max_Pa : float
        Legacy circular-contact estimate ``0.31 p0`` (nu = 0.3) [Pa]. This is
        not the resolved maximum for an arbitrary ellipse or material. Use
        compute_hertz_axis_stress for depth-dependent normal-contact tensors.
    K_t_N_per_m : float
        Area-equivalent isotropic tangential stiffness ``8 G* sqrt(ab)``
        [N/m]. It recovers the circular Mindlin limit exactly; an elliptical
        contact has direction-dependent compliance not resolved by this scalar.
    """

    a_m: float
    b_m: float
    p0_Pa: float
    eccentricity: float
    tau_max_Pa: float
    K_t_N_per_m: float

    @property
    def aspect_ratio(self) -> float:
        """``a / b`` (>= 1)."""
        return self.a_m / self.b_m


def solve_contact(
    F_n: float,
    A_curv: float,
    B_curv: float,
    E_star: float,
    G_star: float | None = None,
) -> HertzContact:
    """Solve the elliptical Hertz contact for a normal load ``F_n``.

    Parameters
    ----------
    F_n : float
        Normal contact force [N].
    A_curv, B_curv : float
        Relative principal curvatures [1/m]; reordered so ``A <= B``.
    E_star : float
        Reduced contact modulus [Pa] (see :func:`reduced_modulus`).
    G_star : float or None, optional
        Reduced shear modulus [Pa] for the tangential stiffness. If ``None``,
        ``K_t`` is reported as ``0`` (not needed for the normal-contact gates).

    Returns
    -------
    HertzContact
    """
    if not all(math.isfinite(x) for x in (F_n, A_curv, B_curv, E_star)):
        raise CableInputError(
            "finite normal force, curvatures and modulus are required"
        )
    if F_n <= 0.0:
        raise CableInputError("normal force F_n must be positive")
    a_c, b_c = (A_curv, B_curv) if A_curv <= B_curv else (B_curv, A_curv)
    if a_c <= 0.0:
        raise CableInputError("relative curvatures must be positive")

    if E_star <= 0:
        raise CableInputError("reduced normal modulus must be positive")
    if G_star is not None and (not math.isfinite(G_star) or G_star < 0):
        raise CableInputError("reduced shear modulus must be finite and nonnegative")

    log_aspect = _log_aspect_ratio(a_c, b_c)
    q = math.exp(-2 * log_aspect)
    log_a = (
        math.log(F_n)
        + math.log(elliprd(0, q, 1))
        - math.log(2 * math.pi)
        - math.log(E_star)
        - math.log(a_c)
    ) / 3
    log_b = log_a - log_aspect
    try:
        a = _positive_result(math.exp(log_a), "contact major semi-axis")
        b = _positive_result(math.exp(log_b), "contact minor semi-axis")
        p0 = _positive_result(
            math.exp(math.log(F_n) + math.log(3 / (2 * math.pi)) - log_a - log_b),
            "peak contact pressure",
        )
        k_t = (
            _positive_result(
                math.exp(math.log(8) + math.log(G_star) + 0.5 * (log_a + log_b)),
                "contact tangential stiffness",
            )
            if G_star is not None and G_star > 0
            else 0.0
        )
    except OverflowError as error:
        raise CableInputError("contact result exceeds floating-point range") from error
    e = math.sqrt(-math.expm1(-2 * log_aspect))
    return HertzContact(
        a_m=a,
        b_m=b,
        p0_Pa=p0,
        eccentricity=e,
        tau_max_Pa=TAU_MAX_OVER_P0_NU03 * p0,
        K_t_N_per_m=k_t,
    )
