# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Per-wire stress components for helical layers.

The total axial stress in a helical wire during cable bending decomposes into
([MenardCartraud2023] Eq. 13):

* ``sigma_A`` — the global axisymmetric/extension stress ``E_w * eps_w`` from
  the helix kinematics ([Costello1997]; the curvature term is the bent-cable
  wire strain of [Knapp1979], [FeretBournazel1987]).
* ``sigma_B`` — the local wire bending stress at a specified material fibre,
  using both radial and tangential curvature components
  ([MenardCartraud2023] Eq. 14), with the normal (radial) component
  following the wire's axial slip state (below).
* ``sigma_f`` — the friction (stick-slip) stress, the universal Coulomb
  return-mapping element (R1 Instantiation A). Its driving variable is the
  signed curvature projection ``kappa cos(phi - phi_bend)``, its elastic
  stiffness ``k_f = xi E_w cos^2(a)`` (the stick stress is
  ``k_f r_h kappa cos(phi - phi_bend)``), and its yield limit
  ``mu F_n / A_w``. See :mod:`cablexsec.mechanics.mkappa` for the time-series
  driver.
* ``sigma_T`` — the thermal stress ``E_w alpha_T (T(r_h) - T_amb)`` (supplied
  by the thermal solver; zero in an isothermal analysis).

Normal curvature and the slip state
-----------------------------------
The normal (out-of-surface, weak-axis for a flat wire) curvature change of
a wire on the deformed helix is ``cos^2(a) cos(2a) p_n`` per unit current
arc length, ``p_n = kappa_x cos(phi) + kappa_y sin(phi)`` (the curve
curvature of the helix carried with the bent surface: Out and von Morgen
1997 Eq. 41; Leroy and Estrier 2001 Eq. 13 without transverse slip). The
bending strain of the wire is set by the rotation of its cross-section per
unit material length, the curve curvature times the stretch ``1 + eps_w``.
The helix starts with the normal curvature ``sin^2(a) / r_h``, so to first
order

    dkappa_n = cos^2(a) cos(2a) p_n + (sin^2(a) / r_h) eps_w,

with ``eps_w = sigma_f / E_w`` the bending-induced axial strain the wire
carries (:func:`slip_state_normal_curvature`). A stuck wire with the full
plane-section strain ``eps_w = r_h cos^2(a) p_n`` takes ``cos^4(a) p_n``
(the prescribed normal curvature of the stuck helix, [Saevik2011] Eq. 50);
with the shear-lag stick coefficient ``xi`` (``k_f = xi E_w cos^2 a``) it
takes ``cos^2(a) cos(2a) + xi sin^2(a) cos^2(a)``; a wire sliding freely
along its helix (``sigma_f = 0``) takes ``cos^2(a) cos(2a)``; a wire in
Coulomb slip keeps the slope ``cos^2(a) cos(2a)`` with the offset of its
friction stress. On the outer face the term is ``(t/2) (sin^2(a) / r_h)
sigma_f``, independent of the modulus. The lateral (in-surface) component
``cos(a) (1 + sin^2 a) p_l`` does not change: the helix starts with zero
geodesic curvature. The stick and slip limits were checked against an
independent finite-element model of a wire on a bent support (not
published).

The legacy axial/contact indicator combines the axial total with the circular
contact shear estimate ``tau_H = 0.31 p0`` ([Johnson1985], nu = 0.3):

    sigma_vm = sqrt(sigma_total^2 + 3 tau_H^2).

Despite its retained API name, this is not a resolved Hertz von Mises stress:
the terms need not occur at the same point, and it is not a proven upper bound.
Use coincident tensors for local invariants, e.g. compute_hertz_axis_stress.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from cablexsec.constants import TAU_MAX_OVER_P0_NU03
from cablexsec.exceptions import CableInputError

__all__ = [
    "friction_stiffness",
    "friction_yield_stress",
    "local_bending_stress",
    "local_bending_amplitude",
    "local_bending_fibre_stress",
    "slip_state_normal_curvature",
    "von_mises_stress",
]

FloatArray = NDArray[np.float64]


def slip_state_normal_curvature(
    friction_stress_Pa: float | FloatArray,
    E_wire_Pa: float | FloatArray,
    lay_angle_rad: float,
    r_helix_m: float,
) -> FloatArray:
    """Normal curvature of the wire's axial slip state [1/m] (module notes).

    ``(sin^2(a) / r_h) sigma_f / E_w``: the initial normal curvature of the
    helix times the bending-induced axial strain ``sigma_f / E_w`` of the
    wire. Added to the deformed-helix term ``cos^2(a) cos(2a) p_n`` it gives
    ``cos^4(a) p_n`` for a stuck wire (``sigma_f = E_w r_h cos^2(a) p_n``),
    ``cos^2(a) cos(2a) + xi sin^2(a) cos^2(a)`` with the stick coefficient
    ``xi`` and nothing for a wire without friction stress.
    """
    if not (math.isfinite(r_helix_m) and r_helix_m > 0.0):
        raise CableInputError("the helix radius must be finite and positive")
    coefficient = math.sin(lay_angle_rad) ** 2 / r_helix_m
    return np.asarray(
        coefficient * (np.asarray(friction_stress_Pa) / np.asarray(E_wire_Pa)),
        dtype=np.float64,
    )


def _slip_term(
    friction_stress_Pa: float | FloatArray | None,
    E_wire_Pa: float | FloatArray,
    lay_angle_rad: float,
    r_helix_m: float | None,
) -> FloatArray | None:
    """:func:`slip_state_normal_curvature`, or ``None`` without a friction stress."""
    if friction_stress_Pa is None:
        return None
    if r_helix_m is None:
        raise CableInputError("a friction stress needs the helix radius r_helix_m")
    return slip_state_normal_curvature(
        friction_stress_Pa, E_wire_Pa, lay_angle_rad, r_helix_m
    )


def local_bending_fibre_stress(
    kappa_x: float | FloatArray,
    kappa_y: float | FloatArray,
    phi: float | FloatArray,
    E_wire_Pa: float | FloatArray,
    d_wire_m: float,
    lay_angle_rad: float,
    fibre_angle_rad: float | FloatArray = 0.0,
    *,
    friction_stress_Pa: float | FloatArray | None = None,
    r_helix_m: float | None = None,
) -> FloatArray:
    """Signed bending stress on a fixed circular-wire material fibre [Pa].

    Curvature components give cable bending strain ``kx*x + ky*y``. Phi is
    the wire-centre azimuth in that material frame. The fibre angle is measured
    from the outward radial direction towards ``wire_tangent cross radial``;
    it must not follow the instantaneous bending direction in a history.
    Array arguments follow NumPy broadcasting, e.g. wire azimuths ``[:, None]``
    and curvature samples ``[None, :]``. No rotation of a critical fibre or
    instantaneous perimeter maximisation is used here.

    ``friction_stress_Pa`` (with the helix radius ``r_helix_m``) is the
    bending-induced axial stress of the wire, which sets the normal
    curvature of its slip state (:func:`slip_state_normal_curvature`, module
    notes). Without it the normal coefficient is the slip limit
    ``cos^2(a) cos(2a)`` ([MenardCartraud2023] Eq. 14), the curvature of a
    wire that carries no friction stress.
    """
    kx, ky = np.asarray(kappa_x), np.asarray(kappa_y)
    angle, theta = np.asarray(phi), np.asarray(fibre_angle_rad)
    c, s = math.cos(lay_angle_rad), math.sin(lay_angle_rad)
    radial = (
        c * c * math.cos(2 * lay_angle_rad) * (kx * np.cos(angle) + ky * np.sin(angle))
    )
    slip = _slip_term(friction_stress_Pa, E_wire_Pa, lay_angle_rad, r_helix_m)
    if slip is not None:
        radial = radial + slip
    tangential = c * (1 + s * s) * (-kx * np.sin(angle) + ky * np.cos(angle))
    return np.asarray(
        np.asarray(E_wire_Pa)
        * d_wire_m
        / 2
        * (radial * np.cos(theta) + tangential * np.sin(theta)),
        dtype=np.float64,
    )


def friction_stiffness(xi: float, E_wire_Pa: float, lay_angle_rad: float) -> float:
    """Stick-branch friction stiffness ``k_f = xi E_w cos^2(a)`` [Pa].

    Relates the friction stress ``sigma_f`` to the driving wire axial strain
    ``u = kappa r_h cos(phi - phi_bend)`` ([Kim2020] stick branch;
    ``xi`` from [Yun2020]). The moment arm carries ``r_h cos(a)`` separately,
    so the net EI_max scaling is cos^3(a) per [WitzTan1992b].
    """
    return xi * E_wire_Pa * math.cos(lay_angle_rad) ** 2


def friction_yield_stress(
    mu: float, F_n_N: float | FloatArray, A_wire_m2: float
) -> FloatArray:
    """Coulomb saturation stress ``f_y = mu F_n / A_w`` [Pa] ([Saevik2011] §5.2)."""
    return np.asarray(mu * np.asarray(F_n_N, dtype=np.float64) / A_wire_m2)


def local_bending_stress(
    kappa: float,
    phi: float | FloatArray,
    E_wire_Pa: float,
    d_wire_m: float,
    lay_angle_rad: float,
    phi_bend: float = 0.0,
    *,
    friction_stress_Pa: float | FloatArray | None = None,
    r_helix_m: float | None = None,
) -> FloatArray:
    """Local stress on the outward radial fibre ([MenardCartraud2023] Eq. 14).

    Phi is measured from the positive extension side, not from the source's
    bending axis. The radial coefficient is cos²(a)*cos(2a), plus the normal
    curvature of the wire's slip state when its friction stress is given
    (:func:`slip_state_normal_curvature`); the tangential coefficient enters
    the perimeter envelope in local_bending_amplitude.
    """
    a = lay_angle_rad
    phi_a = np.asarray(phi, dtype=np.float64)
    slip = _slip_term(friction_stress_Pa, E_wire_Pa, a, r_helix_m)
    if slip is None:
        return np.asarray(
            E_wire_Pa
            * kappa
            * math.cos(a) ** 2
            * math.cos(2 * a)
            * (d_wire_m / 2.0)
            * np.cos(phi_a - phi_bend)
        )
    normal = kappa * math.cos(a) ** 2 * math.cos(2 * a) * np.cos(phi_a - phi_bend)
    return np.asarray(E_wire_Pa * (d_wire_m / 2.0) * (normal + slip))


def local_bending_amplitude(
    kappa: float,
    phi: float | FloatArray,
    E_wire_Pa: float,
    d_wire_m: float,
    lay_angle_rad: float,
    phi_bend: float = 0.0,
    *,
    friction_stress_Pa: float | FloatArray | None = None,
    r_helix_m: float | None = None,
) -> FloatArray:
    """Maximum local bending stress over a circular wire perimeter [Pa].

    This is the norm of the two Lutchansky bending coefficients, not the
    signed stress at one chosen fibre. Add it to abs(mean axial stress) when
    screening the maximum absolute axial stress at a section. A given
    friction stress (with ``r_helix_m``) adds the normal curvature of the
    wire's slip state (:func:`slip_state_normal_curvature`) to the normal
    component, with the sign of the signed curvature ``kappa``.
    """
    angle = np.asarray(phi, dtype=np.float64) - phi_bend
    c, s = math.cos(lay_angle_rad), math.sin(lay_angle_rad)
    slip = _slip_term(friction_stress_Pa, E_wire_Pa, lay_angle_rad, r_helix_m)
    if slip is None:
        return np.asarray(
            E_wire_Pa
            * abs(kappa)
            * d_wire_m
            / 2
            * np.hypot(
                c * c * math.cos(2 * lay_angle_rad) * np.cos(angle),
                c * (1 + s * s) * np.sin(angle),
            )
        )
    return np.asarray(
        E_wire_Pa
        * d_wire_m
        / 2
        * np.hypot(
            kappa * c * c * math.cos(2 * lay_angle_rad) * np.cos(angle) + slip,
            kappa * c * (1 + s * s) * np.sin(angle),
        )
    )


def von_mises_stress(
    sigma_axial_Pa: float | FloatArray, p0_Pa: float | FloatArray
) -> FloatArray:
    """Legacy axial/contact screening indicator [Pa], retaining its API name.

    ``sigma_vm = sqrt(sigma_axial^2 + 3 tau_H^2)`` with ``tau_H = 0.31 p0``
    ([Johnson1985], nu = 0.3).

    The Hertz stress state is not pure shear and these maxima need not be
    coincident. This is neither a resolved tensor invariant nor a guaranteed
    conservative contact-stress bound. With zero pressure it gives |sigma|.
    """
    tau_h = TAU_MAX_OVER_P0_NU03 * np.asarray(p0_Pa, dtype=np.float64)
    sa = np.asarray(sigma_axial_Pa, dtype=np.float64)
    return np.asarray(np.sqrt(sa**2 + 3.0 * tau_h**2))
