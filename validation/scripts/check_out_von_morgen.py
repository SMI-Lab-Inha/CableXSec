# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Analytical check of armour slip against Out and von Morgen (1997).

Run as ``python validation/scripts/check_out_von_morgen.py``. This is a verification
module: nothing here is used by an analysis path.

Sources (read from the library copies)
--------------------------------------
* Out and von Morgen (1997), Eng. Struct. 19(6) 507-515. A helical wire that
  is free to move on a bent cylinder (no friction) follows the geodesic of the
  toroid. To first order in ``k = kappa r`` the slip of each material point
  relative to the "deformed helix" (the helix marked on the bent surface) is,
  in the coordinates ``(psi, theta) = (axial/r, circumferential angle)``:

  - wire fixed at the inside of the bend (intrados, ``theta_0 = pi/2``; the
    condition for a wire remote from end fittings), Eqs. 26-28::

        theta_g - theta_h = k cos(theta_h) / tan^2(a)
        psi_g - psi_h = -2 k cos(theta_h) / tan(a)
        |Delta| / r = |k| |cos(theta_h)| sqrt(1 + 4 tan^2 a) / tan^2 a

  - wire fixed at the side (neutral axis, ``theta_0 = 0``; the extreme
    influence of an end fitting), Eqs. 20-22: the same with ``cos(theta_h)``
    replaced by ``cos(theta_h) - 1``;
  - slip direction, Eqs. 23 and 29: ``gamma = arctan(-1/(2 tan a))`` from the
    pipe axis, independent of ``theta`` and of the curvature;
  - relative slip of two contra-wound layers, Eq. 30:
    ``|kappa| r^2 4 / tan(a)``;
  - curvature change of the deformed helix, Eqs. 41-42.

  ``theta = pi/2`` is the inside of the bend (Eq. 1: distance from the bending
  centre ``rho - r sin(theta)``), ``theta = 0, pi`` the neutral axis. The
  lay angle is measured from the pipe axis (Eq. 12: ``psi = theta/tan a``).
* Saevik (1992), Dr.ing. thesis, NTH, Eqs. 2.29-2.30 (longitudinal and
  transverse displacement to the geodesic, which this module shows to be the
  wire-frame components of Out and von Morgen Eqs. 26-27) and Eq. 4.2 (the
  friction coefficient below which a tension-clamped outer wire slips
  transversely).
* Out (1989), OTC 6165: effective curvature change ``dk - 2 kappa_s`` (Eq. 1)
  and the damage ratios of a curvature change with a uniformly distributed
  starting point on the no-slip line (Appendix Eqs. A8 and A11).

What is compared
----------------
1. The closed forms against an exact numerical geodesic on the toroid
   (Euler first integral, Eq. 7, solved by quadrature and root finding), for
   both boundary conditions. Expected error is first order in ``k``.
2. CableXSec's contact slip at ``mu -> 0`` (``MindlinContact`` driven by
   ``slip_increment``) against the wire-frame components of the geodesic slip,
   for the legacy transverse map (in quadrature with the geodesic slip) and
   the geodesic map (``transverse_map="geodesic"``, gap 1.4b), which
   reproduces Eqs. 26-28 at ``eta = 1`` and the exact geodesic to first order.
3. The equilibrium transverse-slip cell of the companion article
   (:mod:`transverse_friction_cell`) for decreasing friction.
4. CableXSec's local-bending curvature coefficients
   (:func:`cablexsec.mechanics.wire_stress.local_bending_fibre_stress`)
   against Eqs. 41-42 and against a numerical curvature of the deformed helix.
5. Out (1989) Eq. 1 and Eqs. A8/A11 reproduced by CableXSec's Coulomb contact
   element in curvature units.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy import integrate, optimize
from scipy.interpolate import CubicSpline

from cablexsec.mechanics.mindlin_contact import (
    MindlinContact,
    geometric_slip_factor,
    slip_increment,
)
from cablexsec.mechanics.wire_stress import local_bending_fibre_stress

FloatArray = NDArray[np.float64]

INSIDE = "inside"
SIDE = "side"
_THETA0 = {INSIDE: 0.5 * math.pi, SIDE: 0.0}


# --------------------------------------------------------------------------- #
# Closed forms                                                                #
# --------------------------------------------------------------------------- #
def _boundary_factor(theta: FloatArray, boundary: str) -> FloatArray:
    if boundary == INSIDE:
        return np.cos(theta)
    if boundary == SIDE:
        return np.cos(theta) - 1.0
    raise ValueError("boundary must be 'inside' or 'side'")


def ovm_slip(
    kappa: float, r: float, alpha: float, theta: float | FloatArray, boundary: str
) -> FloatArray:
    """Geodesic slip ``(axial, circumferential)`` [m], Eqs. 20-21 and 26-27.

    Axial ``r (psi_g - psi_h)`` and circumferential ``r (theta_g - theta_h)``
    displacement of the material point from the deformed helix.
    """
    f = _boundary_factor(np.asarray(theta, dtype=np.float64), boundary)
    t = math.tan(alpha)
    scale = kappa * r * r
    return np.stack((-2.0 * scale * f / t, scale * f / t**2), axis=-1)


def ovm_slip_magnitude(
    kappa: float, r: float, alpha: float, theta: float | FloatArray, boundary: str
) -> FloatArray:
    """``|Delta|`` [m], Eq. 22 (side) and Eq. 28 (inside)."""
    f = np.abs(_boundary_factor(np.asarray(theta, dtype=np.float64), boundary))
    t = math.tan(alpha)
    return abs(kappa) * r * r * f * math.sqrt(1.0 + 4.0 * t * t) / t**2


def ovm_slip_angle(alpha: float) -> float:
    """Angle of the slip vector from the pipe axis, Eqs. 23 and 29 [rad]."""
    return math.atan(-1.0 / (2.0 * math.tan(alpha)))


def ovm_relative_slip_max(kappa: float, r: float, alpha: float) -> float:
    """Largest relative slip of two contra-wound layers, Eq. 30 [m]."""
    return abs(kappa) * r * r * 4.0 / math.tan(alpha)


def wire_frame(vector: FloatArray, alpha: float) -> tuple[FloatArray, FloatArray]:
    """Components along the wire and transverse to it, in the surface."""
    v = np.asarray(vector, dtype=np.float64)
    c, s = math.cos(alpha), math.sin(alpha)
    return v[..., 0] * c + v[..., 1] * s, -v[..., 0] * s + v[..., 1] * c


def transverse_geodesic_scale(r: float, alpha: float) -> float:
    """``r^2 cos(a)(1 + sin^2 a)/sin^2 a`` [m^2]: Saevik (1992) Eq. 2.30.

    Transverse slip to the geodesic per unit curvature; equal to ``Delta . n``
    of Out and von Morgen Eqs. 26-27 and to the ``w_g = dk_g / q^2`` attractor
    of the companion article's migration check.
    """
    s = math.sin(alpha)
    return r * r * math.cos(alpha) * (1.0 + s * s) / (s * s)


def geodesic_transverse_ratio(alpha: float) -> float:
    """Frictionless transverse/longitudinal slip ratio ``(1+sin^2 a)/(sin a cos a)``.

    The value of ``transverse_ratio`` whose amplitude would equal the
    geodesic transverse amplitude.
    """
    return (1.0 + math.sin(alpha) ** 2) / (math.sin(alpha) * math.cos(alpha))


def saevik_limit_friction(kappa: float, r: float, alpha: float) -> float:
    """Saevik (1992) Eq. 4.2: no transverse slip above this friction."""
    return (
        math.cos(alpha)
        * (1 + math.sin(alpha) ** 2)
        * abs(kappa)
        * r
        / math.sin(alpha) ** 2
    )


def deformed_helix_curvature_coefficients(alpha: float) -> dict[str, float]:
    """Normal and binormal curvature change per unit ``kappa sin/cos theta``.

    ``eq41``: normal, ``sin^2 a cos 2a / tan^2 a = cos^2 a cos 2a`` (Eq. 41).
    ``eq42_printed``: binormal as printed, ``cos a sqrt(3 sin^2 a + 1)``.
    ``lutchansky``: the binormal coefficient ``cos a (1 + sin^2 a)`` used by
    CableXSec and the companion article.
    """
    c, s = math.cos(alpha), math.sin(alpha)
    return {
        "eq41": s * s * math.cos(2 * alpha) / math.tan(alpha) ** 2,
        "eq42_printed": c * math.sqrt(3 * s * s + 1),
        "lutchansky": c * (1 + s * s),
    }


def out1989_effective_change(delta_kappa: float, kappa_s: float) -> float:
    """Out (1989) Eq. 1: effective curvature change ``dk - 2 kappa_s`` (>= 0)."""
    return max(delta_kappa - 2.0 * kappa_s, 0.0)


def out1989_damage_ratio(a: float, c: float) -> float:
    """Out (1989) Eqs. A8 (``a <= 1``) and A11 (``a > 1``), ``a = dk/(2 kappa_s)``."""
    if a <= 0.0:
        return 0.0
    if a <= 1.0:
        return a / (c + 1.0)
    return (a ** (c + 1.0) - (a - 1.0) ** (c + 1.0)) / (a**c * (c + 1.0))


# --------------------------------------------------------------------------- #
# Independent numerical references                                            #
# --------------------------------------------------------------------------- #
def exact_geodesic_slip(
    k: float, alpha: float, boundary: str, theta_h: FloatArray, nodes: int = 2001
) -> FloatArray:
    """Slip ``(axial, circumferential)/r`` from the exact geodesic, Eqs. 7 and 9.

    The geodesic through ``(0, theta_0)`` and ``(2 pi/tan a, theta_0 + 2 pi)``
    is found from the first integral ``theta' = (1 - k sin theta)
    sqrt(C (1 - k sin theta)^2 - 1)`` by quadrature of ``psi(theta)`` and a
    root search on ``C``. Material points are matched by arc length along the
    undeformed helix (Eq. 13), as in the source.
    """
    theta0 = _THETA0[boundary]
    tan_a = math.tan(alpha)
    psi_end = 2.0 * math.pi / tan_a

    def dpsi(theta: float, constant: float) -> float:
        f = 1.0 - k * math.sin(theta)
        return 1.0 / (f * math.sqrt(constant * f * f - 1.0))

    def psi_end_error(constant: float) -> float:
        with warnings.catch_warnings():
            # Bracketing evaluations far from the root may not reach 1e-13.
            warnings.simplefilter("ignore", integrate.IntegrationWarning)
            value = integrate.quad(
                dpsi,
                theta0,
                theta0 + 2 * math.pi,
                args=(constant,),
                epsabs=1e-15,
                epsrel=1e-13,
                limit=400,
            )[0]
        return value - psi_end

    c0 = 1.0 + tan_a * tan_a
    lower = (1.0 + 1e-12) / (1.0 - abs(k)) ** 2
    constant = optimize.brentq(
        psi_end_error, max(lower, 0.5 * c0), 2.0 * c0, xtol=1e-15, rtol=1e-15
    )

    def ds(theta: float) -> float:
        f = 1.0 - k * math.sin(theta)
        p = dpsi(theta, constant)
        return math.sqrt(f * f * p * p + 1.0)

    grid = np.linspace(theta0, theta0 + 2 * math.pi, nodes)
    arc = np.zeros_like(grid)
    psi = np.zeros_like(grid)
    with warnings.catch_warnings():
        # Panels this short reach the round-off floor before epsabs.
        warnings.simplefilter("ignore", integrate.IntegrationWarning)
        for i in range(1, nodes):
            a, b = grid[i - 1], grid[i]
            arc[i] = (
                arc[i - 1] + integrate.quad(ds, a, b, epsabs=1e-16, epsrel=1e-14)[0]
            )
            psi[i] = (
                psi[i - 1]
                + integrate.quad(
                    dpsi, a, b, args=(constant,), epsabs=1e-16, epsrel=1e-14
                )[0]
            )
    arc_of, psi_of = CubicSpline(grid, arc), CubicSpline(grid, psi)
    helix_arc = math.sqrt(1.0 + 1.0 / tan_a**2)
    out = []
    for theta in np.asarray(theta_h, dtype=np.float64):
        s_o = helix_arc * (theta - theta0)
        theta_g = optimize.brentq(
            lambda x, s=s_o: float(arc_of(x)) - s, grid[0], grid[-1], xtol=1e-15
        )
        out.append((float(psi_of(theta_g)) - (theta - theta0) / tan_a, theta_g - theta))
    return np.asarray(out)


def _torus_point(theta: float, psi: float, k: float) -> FloatArray:
    """Eq. 1 with ``r = 1``: ``(rho - sin theta) sin(k psi)`` and so on."""
    rho = 1.0 / k
    w = k * psi
    return np.array(
        [
            (rho - math.sin(theta)) * math.sin(w),
            rho - (rho - math.sin(theta)) * math.cos(w),
            math.cos(theta),
        ]
    )


def _helix_curvature(alpha: float, k: float, theta: float, h: float = 1e-3):
    """Normal and in-surface curvature of the deformed helix, ``r = 1``."""
    tan_a = math.tan(alpha)
    pts = [
        _torus_point(theta + j * h, (theta + j * h) / tan_a, k)
        for j in (-2, -1, 0, 1, 2)
    ]
    d1 = (pts[0] - 8 * pts[1] + 8 * pts[3] - pts[4]) / (12 * h)
    d2 = (-pts[0] + 16 * pts[1] - 30 * pts[2] + 16 * pts[3] - pts[4]) / (12 * h * h)
    speed = float(np.linalg.norm(d1))
    tangent = d1 / speed
    curvature = (d2 - (d2 @ tangent) * tangent) / speed**2
    w = k * theta / tan_a
    normal = np.array(
        [
            math.sin(theta) * math.sin(w),
            -math.sin(theta) * math.cos(w),
            -math.cos(theta),
        ]
    )  # Eq. 33, inward
    binormal = np.cross(normal, tangent)
    return float(curvature @ normal), float(curvature @ binormal)


def numerical_curvature_coefficients(alpha: float, k: float = 1e-3) -> dict[str, float]:
    """``d kappa_n / dk`` at the intrados and ``d kappa_g / dk`` at the side.

    Central difference in ``k`` of the exact curvature of the deformed helix
    on the toroid (``r = 1``). Returned as positive coefficients comparable
    with :func:`deformed_helix_curvature_coefficients`.
    """
    kn_p, _ = _helix_curvature(alpha, k, 0.5 * math.pi)
    kn_m, _ = _helix_curvature(alpha, -k, 0.5 * math.pi)
    _, kg_p = _helix_curvature(alpha, k, math.pi)
    _, kg_m = _helix_curvature(alpha, -k, math.pi)
    return {"normal": -(kn_p - kn_m) / (2 * k), "binormal": (kg_p - kg_m) / (2 * k)}


def exact_geodesic_normal_curvature(
    k: float, alpha: float, theta: float = 0.5 * math.pi, nodes: int = 4001
) -> float:
    """Normal curvature of the exact geodesic on the toroid at ``theta``, ``r = 1``.

    The geodesic is the one of :func:`exact_geodesic_slip` for the boundary
    ``inside`` (first integral, Eq. 7, closed over one pitch); its curve is
    differentiated with a cubic spline of ``psi(theta)`` and the curvature
    vector is projected on the inward surface normal (Eq. 33).
    """
    theta0 = _THETA0[INSIDE]
    tan_a = math.tan(alpha)
    psi_end = 2.0 * math.pi / tan_a

    def dpsi(th: float, constant: float) -> float:
        f = 1.0 - k * math.sin(th)
        return 1.0 / (f * math.sqrt(constant * f * f - 1.0))

    def psi_end_error(constant: float) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", integrate.IntegrationWarning)
            value = integrate.quad(
                dpsi,
                theta0,
                theta0 + 2 * math.pi,
                args=(constant,),
                epsabs=1e-15,
                epsrel=1e-13,
                limit=400,
            )[0]
        return value - psi_end

    c0 = 1.0 + tan_a * tan_a
    lower = (1.0 + 1e-12) / (1.0 - abs(k)) ** 2
    constant = optimize.brentq(
        psi_end_error, max(lower, 0.5 * c0), 2.0 * c0, xtol=1e-15, rtol=1e-15
    )
    grid = np.linspace(theta0, theta0 + 2 * math.pi, nodes)
    psi = np.zeros_like(grid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", integrate.IntegrationWarning)
        for i in range(1, nodes):
            psi[i] = (
                psi[i - 1]
                + integrate.quad(
                    dpsi,
                    grid[i - 1],
                    grid[i],
                    args=(constant,),
                    epsabs=1e-16,
                    epsrel=1e-14,
                )[0]
            )
    spline = CubicSpline(grid, psi)
    h = 1e-3
    pts = [
        _torus_point(theta + j * h, float(spline(theta + j * h)), k)
        for j in (-2, -1, 0, 1, 2)
    ]
    d1 = (pts[0] - 8 * pts[1] + 8 * pts[3] - pts[4]) / (12 * h)
    d2 = (-pts[0] + 16 * pts[1] - 30 * pts[2] + 16 * pts[3] - pts[4]) / (12 * h * h)
    speed = float(np.linalg.norm(d1))
    tangent = d1 / speed
    curvature = (d2 - (d2 @ tangent) * tangent) / speed**2
    w = k * float(spline(theta))
    normal = np.array(
        [
            math.sin(theta) * math.sin(w),
            -math.sin(theta) * math.cos(w),
            -math.cos(theta),
        ]
    )
    return float(curvature @ normal)


def geodesic_normal_coefficient(alpha: float, k: float = 1e-3) -> float:
    """``-d kappa_n / dk`` of the exact geodesic at the intrados (positive form).

    Compared with Out and von Morgen (1997) Eq. 39 and Sævik (1992) Eq. 2.32
    (``3 cos^2 a``) and with
    :func:`cablexsec.mechanics.tendon_bending.curvature_coefficients` at full
    transverse slip.
    """
    plus = exact_geodesic_normal_curvature(k, alpha)
    minus = exact_geodesic_normal_curvature(-k, alpha)
    return -(plus - minus) / (2 * k)


def tendon_bending_checks(alphas_deg=(20.0, 25.0, 35.0, 50.0)) -> dict[str, float]:
    """Gap G1: CableXSec tendon curvature coefficients against the closed forms.

    * ``eta = 0``: lateral = Lutchansky ``cos a (1 + sin^2 a)`` (Sævik 1992
      Eq. 2.65; Out and von Morgen Eq. 42 is printed differently, see the
      binormal rows of :func:`main`), normal = Eq. 41;
    * ``eta = 1``: lateral 0, normal ``3 cos^2 a`` (Eq. 39; Sævik Eq. 2.32)
      and the numerical geodesic;
    * geodesic offset ``w_g`` against Sævik Eq. 2.30, and the transverse
      Jenkin model at its slip threshold against Sævik Eq. 4.2.
    """
    from cablexsec.mechanics.tendon_bending import (  # noqa: PLC0415
        TransverseSlipModel,
        curvature_coefficients,
    )
    from cablexsec.mechanics.tendon_section import WireSection  # noqa: PLC0415

    worst = {
        "lox": 0.0,
        "geo_closed": 0.0,
        "geo_numeric": 0.0,
        "offset": 0.0,
        "eq42": 0.0,
    }
    for adeg in alphas_deg:
        a = math.radians(adeg)
        c, s = math.cos(a), math.sin(a)
        lox = curvature_coefficients(a, 0.0)
        geo = curvature_coefficients(a, 1.0)
        closed = deformed_helix_curvature_coefficients(a)
        worst["lox"] = max(
            worst["lox"],
            abs(lox.normal - closed["eq41"]) + abs(lox.lateral - closed["lutchansky"]),
        )
        worst["geo_closed"] = max(
            worst["geo_closed"], abs(geo.normal - 3 * c * c) + abs(geo.lateral)
        )
        numeric = geodesic_normal_coefficient(a)
        worst["geo_numeric"] = max(
            worst["geo_numeric"], abs(numeric - geo.normal) / geo.normal
        )
        r = 0.1
        section = WireSection(shape="rectangular", width_m=1e-6, thickness_m=1e-6)
        model = TransverseSlipModel(
            section, 1e3, 1e3, r, a, tension_N=1.0e4, capacity_N_per_m=1.0
        )
        worst["offset"] = max(
            worst["offset"],
            abs(model.geodesic_offset_m2 - transverse_geodesic_scale(r, a))
            / transverse_geodesic_scale(r, a),
        )
        # Sævik Eq. 4.2: a tension-clamped wire (capacity mu N sin^2 a / r,
        # no bending or torsion stiffness) starts to slip at mu_lim.
        kappa, mu = 0.05, 0.08
        limit = saevik_limit_friction(kappa, r, a)
        cap = mu * 1.0e4 * s * s / r
        clamped = TransverseSlipModel(section, 1e3, 1e3, r, a, 1.0e4, cap)
        rho = float(clamped.demand_ratio(kappa, 0.0, np.array([0.5 * math.pi]))[0])
        expected = min(1.0, limit / mu)
        worst["eq42"] = max(worst["eq42"], abs(rho - expected))
    return worst


# --------------------------------------------------------------------------- #
# CableXSec quantities                                                        #
# --------------------------------------------------------------------------- #
def cablexsec_cycle_slip(
    kappa: float,
    r: float,
    alpha: float,
    phi: float,
    *,
    mu: float,
    eta: float = 0.0,
    transverse_map: str = "legacy",
    K_t: float = 1.0e7,  # noqa: N803
    F_n: float = 100.0,  # noqa: N803
    steps: int = 20,
) -> FloatArray:
    """Half ranges of CableXSec's contact slip over a ``0, +k, -k, +k`` cycle.

    Returns ``(longitudinal, transverse, delta*)`` [m] of the plastic slip
    trajectory of ``MindlinContact`` driven by ``slip_increment`` with the
    given transverse map.
    """
    c_geo = geometric_slip_factor(r, alpha)
    contact = MindlinContact(K_t=K_t, mu=mu)
    path = np.concatenate(
        (
            np.linspace(0, kappa, steps + 1),
            np.linspace(kappa, -kappa, 2 * steps + 1)[1:],
            np.linspace(-kappa, kappa, 2 * steps + 1)[1:],
        )
    )
    total = np.zeros(2)
    trajectory = [np.zeros(2)]
    for dk in np.diff(path):
        d = slip_increment(
            float(dk),
            0.0,
            phi,
            c_geo,
            transverse_ratio=eta,
            transverse_map=transverse_map,  # type: ignore[arg-type]
            lay_angle_rad=alpha,
        )
        force, _, _ = contact.update(d, F_n)
        total = total + d
        trajectory.append(total - force / K_t)
    s = np.asarray(trajectory)
    return np.array(
        [0.5 * np.ptp(s[:, 0]), 0.5 * np.ptp(s[:, 1]), contact.fretting_amplitude]
    )


def compare_frictionless_contact(
    alphas_deg=(15.0, 25.0, 35.0, 45.0, 55.0),
    etas=(0.0, 0.5, 1.0, 2.0),
    kappa: float = 0.1,
    r: float = 0.05,
    azimuths: int = 24,
) -> dict[str, float]:
    """CableXSec contact slip at ``mu = 0`` against the geodesic slip.

    Azimuth map: CableXSec ``phi`` is measured from the extension side, Out and
    von Morgen's ``theta = 3 pi/2`` is the outside of the bend, so
    ``theta = phi + 3 pi/2`` and ``|cos theta| = |sin phi|``. Signs depend on
    the orientation conventions; magnitudes are compared.
    """
    worst = {
        # eta = 0: |CableXSec longitudinal| against |Delta . t| (identity).
        "longitudinal_identity": 0.0,
        # eta = 0: largest CableXSec delta* / |Delta| (inside, remote wire).
        "magnitude_ratio_inside": 0.0,
        # eta = 0: largest CableXSec transverse slip (expected zero).
        "transverse_eta0_m": 0.0,
        # eta > 0: transverse slip at the extreme fibres, where |Delta . n| = 0.
        "transverse_at_extreme_fibre_over_c_geo_kappa": 0.0,
        # eta > 0: transverse slip at the neutral axis / |Delta . n| there.
        "transverse_at_neutral_axis_over_geodesic": 0.0,
        # All eta: components against the end-fitting envelope (1 + |cos|).
        "envelope_ratio": 0.0,
    }
    phis = np.linspace(0.0, 2 * math.pi, azimuths, endpoint=False)
    for adeg in alphas_deg:
        alpha = math.radians(adeg)
        c_geo = geometric_slip_factor(r, alpha)
        s_g = transverse_geodesic_scale(r, alpha)
        for eta in etas:
            for phi in phis:
                cx = cablexsec_cycle_slip(kappa, r, alpha, float(phi), mu=0.0, eta=eta)
                theta = phi + 1.5 * math.pi
                lon, tra = wire_frame(ovm_slip(kappa, r, alpha, theta, INSIDE), alpha)
                lon, tra = abs(float(lon)), abs(float(tra))
                envelope = (1.0 + abs(math.cos(theta))) * kappa
                worst["envelope_ratio"] = max(
                    worst["envelope_ratio"],
                    cx[0] / (c_geo * envelope),
                    cx[1] / (s_g * envelope),
                )
                if eta == 0.0:
                    worst["longitudinal_identity"] = max(
                        worst["longitudinal_identity"],
                        abs(cx[0] - lon) / (c_geo * kappa),
                    )
                    worst["transverse_eta0_m"] = max(worst["transverse_eta0_m"], cx[1])
                    if lon > 1e-6 * c_geo * kappa:
                        worst["magnitude_ratio_inside"] = max(
                            worst["magnitude_ratio_inside"],
                            cx[2] / math.hypot(lon, tra),
                        )
                    continue
                if abs(math.cos(phi)) > 1 - 1e-12:
                    worst["transverse_at_extreme_fibre_over_c_geo_kappa"] = max(
                        worst["transverse_at_extreme_fibre_over_c_geo_kappa"],
                        cx[1] / (c_geo * kappa),
                    )
                if abs(math.sin(phi)) > 1 - 1e-12:
                    worst["transverse_at_neutral_axis_over_geodesic"] = max(
                        worst["transverse_at_neutral_axis_over_geodesic"], cx[1] / tra
                    )
    return worst


def cablexsec_geodesic_slip(
    kappa: float,
    r: float,
    alpha: float,
    phi: float | FloatArray,
    eta: float = 1.0,
    kappa_y: float = 0.0,
) -> FloatArray:
    """CableXSec geodesic-map slip ``(longitudinal, transverse)`` [m].

    One increment from zero curvature to ``(kappa, kappa_y)``; the map is
    linear in the curvature, so this is the slip of a frictionless contact.
    """
    return slip_increment(
        kappa,
        kappa_y,
        phi,
        geometric_slip_factor(r, alpha),
        eta,
        transverse_map="geodesic",
        lay_angle_rad=alpha,
    )


def compare_geodesic_map(
    alphas_deg=(15.0, 25.0, 35.0, 45.0, 55.0),
    kappa: float = 0.1,
    r: float = 0.05,
    azimuths: int = 24,
) -> dict[str, float]:
    """CableXSec geodesic map (``eta = 1``) against Out and von Morgen.

    ``closed_form_residual``: largest ``|s_cx + (Delta . t, Delta . n)|``
    over ``max |Delta|`` (CableXSec's curvature sign is opposite to the
    source's, so the slip vectors are equal and opposite). The cycle checks
    drive ``MindlinContact`` at ``mu = 0`` over ``0, +k, -k, +k``.
    """
    worst = {
        "closed_form_residual": 0.0,
        "cycle_amplitude_over_eq28_min": math.inf,
        "cycle_amplitude_over_eq28_max": 0.0,
        "transverse_at_extreme_fibre_over_geodesic": 0.0,
        "transverse_at_neutral_axis_over_geodesic_min": math.inf,
        "transverse_at_neutral_axis_over_geodesic_max": 0.0,
    }
    phis = np.linspace(0.0, 2 * math.pi, azimuths, endpoint=False)
    for adeg in alphas_deg:
        alpha = math.radians(adeg)
        s_g = transverse_geodesic_scale(r, alpha)
        theta = phis + 1.5 * math.pi
        cx = cablexsec_geodesic_slip(kappa, r, alpha, phis)
        lon, tra = wire_frame(ovm_slip(kappa, r, alpha, theta, INSIDE), alpha)
        bound = float(np.max(ovm_slip_magnitude(kappa, r, alpha, theta, INSIDE)))
        residual = np.hypot(cx[:, 0] + lon, cx[:, 1] + tra)
        worst["closed_form_residual"] = max(
            worst["closed_form_residual"], float(np.max(residual)) / bound
        )
        for phi in phis:
            cycle = cablexsec_cycle_slip(
                kappa, r, alpha, float(phi), mu=0.0, eta=1.0, transverse_map="geodesic"
            )
            magnitude = float(
                ovm_slip_magnitude(kappa, r, alpha, phi + 1.5 * math.pi, INSIDE)
            )
            if magnitude > 1e-9 * bound:
                ratio = cycle[2] / magnitude
                worst["cycle_amplitude_over_eq28_min"] = min(
                    worst["cycle_amplitude_over_eq28_min"], ratio
                )
                worst["cycle_amplitude_over_eq28_max"] = max(
                    worst["cycle_amplitude_over_eq28_max"], ratio
                )
            if abs(math.cos(phi)) > 1 - 1e-12:
                worst["transverse_at_extreme_fibre_over_geodesic"] = max(
                    worst["transverse_at_extreme_fibre_over_geodesic"],
                    cycle[1] / (s_g * kappa),
                )
            if abs(math.sin(phi)) > 1 - 1e-12:
                ratio = cycle[1] / (s_g * kappa)
                worst["transverse_at_neutral_axis_over_geodesic_min"] = min(
                    worst["transverse_at_neutral_axis_over_geodesic_min"], ratio
                )
                worst["transverse_at_neutral_axis_over_geodesic_max"] = max(
                    worst["transverse_at_neutral_axis_over_geodesic_max"], ratio
                )
    return worst


def geodesic_map_exact_error(
    k: float, alpha: float, points: int = 9, nodes: int = 2001
) -> float:
    """Geodesic map against the exact toroid geodesic, ``r = 1``, ``kappa = k``.

    Largest wire-frame difference over ``max |Delta|``; first order in ``k``.
    """
    theta = _THETA0[INSIDE] + np.linspace(0.2, 2 * math.pi - 0.2, points)
    phi = theta - 1.5 * math.pi
    exact = exact_geodesic_slip(k, alpha, INSIDE, theta, nodes=nodes)
    lon, tra = wire_frame(exact, alpha)
    cx = cablexsec_geodesic_slip(k, 1.0, alpha, phi)
    scale = float(np.max(ovm_slip_magnitude(k, 1.0, alpha, theta, INSIDE)))
    return float(np.max(np.hypot(cx[:, 0] + lon, cx[:, 1] + tra))) / scale


def contact_approach(
    kappa: float = 0.1, r: float = 0.05, alpha: float = math.radians(35.0)
) -> list[tuple[float, float]]:
    """Neutral-axis longitudinal slip / geodesic component as ``mu -> 0``."""
    target = geometric_slip_factor(r, alpha) * kappa
    rows = []
    for mu in (0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0):
        cx = cablexsec_cycle_slip(kappa, r, alpha, 0.5 * math.pi, mu=mu)
        rows.append((mu, float(cx[0] / target)))
    return rows


def cell_friction_sweep(
    factors=(1.0, 0.5, 0.2, 0.1, 0.075, 0.06, 0.05),
    kappa: float = 0.10,
    increments: int = 200,
    elements: int = 64,
) -> list[dict[str, float | str]]:
    """Equilibrium transverse-slip cell of the companion article for decreasing ``mu``.

    Published three-core cable (Menard and Cartraud 2023) at the companion article's
    ``mu = 0.2``, monotonic ramp to ``kappa``, geodesic feedback on. The
    transverse plastic slip is compared with the geodesic slip
    ``transverse_geodesic_scale * kappa * |sin phi|`` station by station, the
    longitudinal with ``c_geo * kappa``.
    """
    from transverse_friction_cell import solve_transverse_cell  # noqa: PLC0415
    from transverse_friction_runs import layer_inputs  # noqa: PLC0415

    import cablexsec as cx  # noqa: PLC0415
    from cablexsec.cable_adapters import cable_to_mkappa_specs  # noqa: PLC0415

    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    layers = cable_to_mkappa_specs(cable, 0.0, mu=0.2)
    history = np.linspace(0.0, kappa, increments + 1)
    rows: list[dict[str, float | str]] = []
    for index, layer in enumerate(layers):
        inputs = layer_inputs(layer)
        alpha, radius = inputs["lay_angle_rad"], inputs["helix_radius_m"]
        cap = inputs["friction_limit_N_m"]
        w_g = transverse_geodesic_scale(radius, alpha) * kappa
        s_l = geometric_slip_factor(radius, alpha) * kappa
        for factor in factors:
            inputs["friction_limit_N_m"] = cap * factor
            row: dict[str, float | str] = {
                "layer": getattr(layer, "name", str(index)),
                "lay_angle_deg": math.degrees(alpha),
                "mu": 0.2 * factor,
            }
            try:
                res = solve_transverse_cell(
                    **inputs,
                    curvature_1_m=history,
                    mode="equilibrium",
                    elements=elements,
                    geodesic_feedback=True,
                )
            except RuntimeError as error:
                row["status"] = f"stopped: {error}"
                rows.append(row)
                break
            pt = np.abs(res.transverse_plastic_m[-1])
            pl = np.abs(res.longitudinal_plastic_m[-1])
            bound = w_g * np.abs(np.sin(res.phi_rad))
            mask = bound > 1e-6 * w_g
            row.update(
                status="ok",
                transverse_over_geodesic=float(pt.max() / w_g),
                pointwise_transverse_over_geodesic=float(
                    np.max(pt[mask] / bound[mask])
                ),
                longitudinal_over_loxodromic=float(pl.max() / s_l),
                demand_ratio=float(res.demand_ratio_max.max()),
            )
            rows.append(row)
    return rows


def geodesic_attractor_residual(alpha: float, r: float, kappa: float) -> float:
    """In-surface curvature left on the cell's path when ``w = w_g sin(phi)``.

    The cell's demand uses ``dk_g = kappa cos a (1 + sin^2 a) sin(phi) + w''``
    with ``d/ds = q d/dphi``; this evaluates it for the geodesic slip of
    Saevik Eq. 2.30 / Out and von Morgen Eq. 26-27.
    """
    q = math.sin(alpha) / r
    phi = np.linspace(0.0, math.pi, 257)
    w_g = transverse_geodesic_scale(r, alpha) * kappa
    loxodrome = kappa * math.cos(alpha) * (1 + math.sin(alpha) ** 2) * np.sin(phi)
    second = -(q**2) * w_g * np.sin(phi)
    scale = kappa * math.cos(alpha) * (1 + math.sin(alpha) ** 2)
    return float(np.max(np.abs(loxodrome + second)) / scale)


def out1989_contact_checks(
    c_values=(1.0, 2.0, 3.0),
    a_values=(0.25, 0.5, 1.0, 1.5, 3.0, 9.0),
    points: int = 2000,
) -> dict[str, float]:
    """Out (1989) Eq. 1 and A8/A11 through ``MindlinContact`` (``eta = 0``).

    A contact at azimuth ``phi`` with stiffness ``K_t``, friction ``mu`` and
    normal force ``F_n`` has the slip curvature
    ``kappa_s = mu F_n / (K_t c_geo |sin phi|)``; its plastic slip divided by
    ``c_geo |sin phi|`` is the effective curvature change.
    """
    r, alpha, phi = 0.05, math.radians(35.0), 1.1
    K_t, mu, F_n = 2.0e7, 0.3, 150.0  # noqa: N806
    c_geo = geometric_slip_factor(r, alpha)
    lever = c_geo * abs(math.sin(phi))
    kappa_s = mu * F_n / (K_t * lever)

    def drive(contact: MindlinContact, state: dict, dk: float) -> None:
        d = slip_increment(dk, 0.0, phi, c_geo, transverse_ratio=0.0)
        force, _, _ = contact.update(d, F_n)
        state["kappa"] += dk
        state["total"] = state["total"] + d
        state["plastic"] = state["total"] - force / K_t

    def fresh() -> tuple[MindlinContact, dict]:
        state = {"kappa": 0.0, "total": np.zeros(2), "plastic": np.zeros(2)}
        return MindlinContact(K_t=K_t, mu=mu), state

    # Eq. 1: constant-amplitude cycling between -dk/2 and +dk/2.
    eq1 = 0.0
    for dk in (0.5 * kappa_s, 2.0 * kappa_s, 3.0 * kappa_s, 10.0 * kappa_s):
        contact, state = fresh()
        for target in (0.5 * dk, -0.5 * dk, 0.5 * dk, -0.5 * dk):
            step = (target - state["kappa"]) / 40
            for _ in range(40):
                drive(contact, state, step)
        effective = 2.0 * contact.fretting_amplitude / lever
        expected = out1989_effective_change(dk, kappa_s)
        eq1 = max(eq1, abs(effective - expected) / dk)

    # A8/A11: starting points uniform on the no-slip line (midpoint rule).
    damage = 0.0
    starts = (np.arange(points) + 0.5) / points * 2.0 * kappa_s
    for a in a_values:
        dk = 2.0 * a * kappa_s
        effective = np.empty(points)
        for i, x in enumerate(starts):
            contact, state = fresh()
            drive(contact, state, -(dk + 4.0 * kappa_s))  # onto the lower slip line
            drive(contact, state, float(x))  # start point on the no-slip line
            start = state["plastic"].copy()
            drive(contact, state, dk)
            effective[i] = abs(state["plastic"][0] - start[0]) / lever
        for c in c_values:
            ratio = float(np.mean(effective**c)) / dk**c
            damage = max(damage, abs(ratio - out1989_damage_ratio(a, c)))
    return {"eq1_residual": eq1, "damage_ratio_abs_error": damage, "kappa_s": kappa_s}


def closed_form_errors(
    k: float = 1e-4, alphas_deg=(25.0, 35.0, 45.0)
) -> dict[str, float]:
    """Largest relative difference, closed forms against the exact geodesic."""
    out = {}
    for boundary in (INSIDE, SIDE):
        worst = 0.0
        for adeg in alphas_deg:
            alpha = math.radians(adeg)
            theta = _THETA0[boundary] + np.linspace(0.2, 2 * math.pi - 0.2, 9)
            exact = exact_geodesic_slip(k, alpha, boundary, theta)
            closed = ovm_slip(k, 1.0, alpha, theta, boundary)  # r = 1, kappa = k
            scale = float(np.max(ovm_slip_magnitude(k, 1.0, alpha, theta, boundary)))
            worst = max(
                worst, float(np.max(np.linalg.norm(exact - closed, axis=1))) / scale
            )
        out[boundary] = worst
    return out


def main() -> None:
    print("Out and von Morgen (1997) frictionless slip bounds")
    print("-" * 72)
    for k in (1e-2, 1e-3, 1e-4):
        errors = closed_form_errors(k)
        print(
            f"closed forms vs exact geodesic, k = {k:g}: inside {errors[INSIDE]:.2e}, "
            f"side {errors[SIDE]:.2e} (relative)"
        )
    for adeg in (25.0, 35.0, 45.0):
        a = math.radians(adeg)
        print(
            f"a = {adeg:4.1f} deg: slip angle {math.degrees(ovm_slip_angle(a)):6.2f} deg; "
            f"transverse/longitudinal {geodesic_transverse_ratio(a):.3f}; "
            f"loxodromic/geodesic magnitude {math.sin(a) / math.sqrt(1 + 4 * math.tan(a) ** 2):.3f}; "
            f"loxodromic/Eq. 30 relative slip {math.cos(a) ** 2 / 2:.3f}"
        )
    worst = compare_frictionless_contact()
    print("CableXSec contact slip at mu = 0 against the geodesic slip (legacy map):")
    for key, value in worst.items():
        print(f"  {key}: {value:.3e}")
    print("CableXSec geodesic transverse map (eta = 1) against the geodesic slip:")
    for key, value in compare_geodesic_map().items():
        print(f"  {key}: {value:.3e}")
    for k in (1e-2, 1e-3, 1e-4):
        print(
            f"  geodesic map vs exact geodesic, k = {k:g}, 35 deg: "
            f"{geodesic_map_exact_error(k, math.radians(35.0)):.2e} (relative)"
        )
    print("Neutral-axis longitudinal slip / geodesic component as mu -> 0 (35 deg):")
    print("  " + ", ".join(f"mu {m:g}: {v:.4f}" for m, v in contact_approach()))
    print(
        "Geodesic attractor residual of the cell demand:",
        f"{geodesic_attractor_residual(math.radians(35.0), 0.05, 0.1):.1e}",
    )
    print("Equilibrium cell, published cable, ramp to 0.10 1/m:")
    for row in cell_friction_sweep():
        print("  ", row)
    print("Deformed-helix curvature coefficients (per unit kappa):")
    for adeg in (20.0, 35.0, 50.0):
        a = math.radians(adeg)
        closed = deformed_helix_curvature_coefficients(a)
        numeric = numerical_curvature_coefficients(a)
        cx_radial = float(local_bending_fibre_stress(1.0, 0.0, 0.0, 2.0, 1.0, a, 0.0))
        cx_tang = float(
            local_bending_fibre_stress(
                1.0, 0.0, -0.5 * math.pi, 2.0, 1.0, a, 0.5 * math.pi
            )
        )
        print(
            f"  a = {adeg:4.1f}: normal numeric {numeric['normal']:+.5f}, Eq. 41 "
            f"{closed['eq41']:+.5f}, CableXSec {cx_radial:+.5f}; binormal numeric "
            f"{numeric['binormal']:+.5f}, Eq. 42 as printed {closed['eq42_printed']:+.5f}, "
            f"CableXSec {cx_tang:+.5f}"
        )
    print("Out (1989):", out1989_contact_checks())
    print("Gap G1 tendon curvature and transverse checks (abs/rel errors):")
    print("  ", tendon_bending_checks())
    for adeg in (25.0, 35.0):
        a = math.radians(adeg)
        print(
            f"  a = {adeg:4.1f}: geodesic normal coefficient numeric "
            f"{geodesic_normal_coefficient(a):+.5f}, 3 cos^2 a {3 * math.cos(a) ** 2:+.5f}"
        )


if __name__ == "__main__":
    main()
