# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Tangential Coulomb return map with a Mindlin initial stiffness.

This module implements a two-component contact screen. The
tangential force at each modelled wire contact follows an incremental return
mapping onto the Coulomb circle, so non-proportional and biaxial inputs can be
represented while the contact normal force ``F_n`` is refreshed every step.
This elastic-perfectly plastic update is not the full Cattaneo--Mindlin
partial-slip history solution; Mindlin enters through the initial stiffness
and the stuck-zone expression is a state diagnostic.

Source-label note
-----------------
The project spec cited "[Mindlin1949] Eq. 7" for the monotonic
force-displacement law. Reading the sources shows Eq. (7) of Mindlin (1949) is
the *normal* compliance; the monotonic tangential (Cattaneo-Mindlin) law is
Johnson (1985) eq. (7.42). It is implemented here as :func:`mindlin_monotonic`
with the correct gross-slip displacement ``delta* = 3 mu F_n / (2 K_t)`` (which
gives an initial tangential stiffness of exactly ``K_t``).

Transverse maps
---------------
Two transverse maps are available in :func:`slip_increment`. Both reduce to
the published loxodromic displacement at ``eta = 0``.

``transverse_map="geodesic"`` (recommended)
    The frictionless slip of a wire on the bent cylinder, Out and von Morgen
    (1997) Eqs. 26-27 in the wire frame (Saevik 1992, Eqs. 2.29-2.30)::

        d(delta_l) = c_geo (d_kx sin phi - d_ky cos phi)
        d(delta_q) = -eta (1 + sin^2 alpha)/(sin alpha cos alpha) d(delta_l)

    The ratio is ``s_geo/c_geo`` with ``s_geo = r_h^2 cos(alpha)(1 + sin^2
    alpha)/sin^2(alpha)`` (:func:`transverse_geodesic_factor`, Saevik 1992
    Eq. 2.30), signed with the lay angle. The transverse component is in
    phase with the longitudinal one: both vanish at the extreme fibres and
    peak at the neutral axis, and the slip direction is fixed (Eqs. 23/29).
    ``eta = 1`` is the frictionless geodesic, ``0 <= eta <= 1`` scales the
    transverse component between the loxodromic and geodesic limits.

``transverse_map="legacy"``
    The earlier constant-map sensitivity family::

        d(delta_l) = c_geo (d_kx sin phi - d_ky cos phi)
        d(delta_q) = eta c_geo (d_kx cos phi + d_ky sin phi)

    Its transverse component is in quadrature with the geodesic transverse
    slip: it is largest at the extreme fibres, where the frictionless slip
    is zero, and zero at the neutral axis, where the frictionless transverse
    slip is 2.8-4.3 times the longitudinal one for lay angles of 15-55 deg
    (``scripts/check_out_von_morgen.py``). It is kept to reproduce earlier
    results; selecting it implicitly (``transverse_map=None`` with
    ``eta > 0``) issues :class:`TransverseMapWarning`.

Rotation-invariance theorem (legacy map)
----------------------------------------
With ``c_geo = r_h^2 cos^2(alpha) / sin(alpha)``, ``eta=0`` is the published
loxodromic displacement; ``eta=1`` is the equal-scale isotropic member.
Only for ``eta=1`` is the norm independent of azimuth. Then

    |d delta_t|^2 = c_geo^2 [ (d_kx sin phi - d_ky cos phi)^2
                            + (d_kx cos phi + d_ky sin phi)^2 ]
                  = c_geo^2 [ d_kx^2 (sin^2+cos^2) + d_ky^2 (cos^2+sin^2)
                            + 2 d_kx d_ky (-sin cos + cos sin) ]
                  = c_geo^2 (d_kx^2 + d_ky^2),

independent of ``phi`` — the cross terms cancel identically. The complete
increment histories at two azimuths therefore differ only by an orthogonal
rotation. An isotropic return map preserves their trajectory amplitude, unlike
a scalar loxodromic projection, subject to the contact-state conditions below.

The loxodromic and geodesic maps are instead covariant: rotating the bending
direction by ``beta`` moves the whole azimuthal pattern by ``beta``, and the
increment at each azimuth stays on one fixed line of the contact plane.

Scope condition
---------------
At ``eta=1``, the uniform-amplitude consequence holds only when ``K_t``, ``mu``
and the complete ``F_n(t)`` history are identical at every azimuth. A
three-core contact shape, anisotropy, lateral contact or material variation
breaks that condition. For ``eta<1``, the kinematic norm itself varies with
azimuth. Values above one are allowed; unity is not a physical upper bound.
This is a constant-map sensitivity family, not a uniform-damage law or a
proof that different mechanically coupled contact states have equal moments.

References
----------
[Mindlin1949]; [Cattaneo1938]; [Johnson1985] §7 (eqs 7.42, 7.39);
Out and von Morgen (1997), Eng. Struct. 19(6) 507-515, Eqs. 23-29;
Saevik (1992), Dr.ing. thesis, NTH, Eqs. 2.29-2.30.

The module computes the slip amplitude ``delta*``
(:attr:`MindlinContact.fretting_amplitude`) and the shear-slip product
``Phi = mu p0 delta*`` (:meth:`MindlinContact.shear_slip_product`); the Hertz
half-width comes from :mod:`cablexsec.mechanics.hertz_contact`. ``Phi`` uses
the nominal frictional shear scale ``mu p0`` with the peak Hertz pressure. It
is not the Ruiz parameter of [Ruiz1984], which is formed from the local
surface shear traction and slip and, in its second form, the local tangential
tensile stress; neither Ruiz parameter is computed here.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.return_mapping import CoulombElement, coulomb_return_map

__all__ = [
    "TRANSVERSE_MAPS",
    "MindlinContact",
    "TransverseMap",
    "TransverseMapWarning",
    "geometric_slip_factor",
    "loxodromic_slip_increment",
    "mindlin_monotonic",
    "slip_increment",
    "stuck_zone_ratio",
    "transverse_geodesic_factor",
    "transverse_map_or_warn",
]

FloatArray = NDArray[np.float64]
TransverseMap = Literal["geodesic", "legacy"]
TRANSVERSE_MAPS: tuple[TransverseMap, ...] = ("geodesic", "legacy")


class TransverseMapWarning(UserWarning):
    """A non-zero transverse ratio selected the legacy map implicitly.

    The legacy transverse map is in quadrature with the frictionless
    geodesic transverse slip. Pass ``transverse_map="geodesic"`` for the
    in-phase map, or ``transverse_map="legacy"`` to reproduce earlier results
    without this warning.
    """


_LEGACY_NOTE = (
    "transverse_ratio > 0 without transverse_map uses the legacy transverse "
    "map, which is in quadrature with the frictionless geodesic slip (zero "
    "at the neutral axis, largest at the extreme fibres). Pass "
    "transverse_map='geodesic' for the in-phase map, or "
    "transverse_map='legacy' to keep the earlier results."
)


def transverse_map_or_warn(
    transverse_ratio: float,
    transverse_map: TransverseMap | None,
    *,
    stacklevel: int = 3,
) -> TransverseMap:
    """Validate a ``(transverse_ratio, transverse_map)`` pair; resolve ``None``.

    ``None`` resolves to ``"legacy"``, with :class:`TransverseMapWarning` when
    ``transverse_ratio > 0``. The geodesic map accepts ``0 <= eta <= 1``.
    """
    if not math.isfinite(transverse_ratio) or transverse_ratio < 0.0:
        raise CableInputError("transverse_ratio must be finite and non-negative")
    if transverse_map is None:
        if transverse_ratio > 0.0:
            warnings.warn(_LEGACY_NOTE, TransverseMapWarning, stacklevel=stacklevel)
        return "legacy"
    if transverse_map not in TRANSVERSE_MAPS:
        raise CableInputError(
            f"transverse_map must be one of {TRANSVERSE_MAPS}, not {transverse_map!r}"
        )
    if transverse_map == "geodesic" and transverse_ratio > 1.0:
        raise CableInputError(
            "the geodesic transverse map takes 0 <= transverse_ratio <= 1 "
            "(1 is the frictionless geodesic slip)"
        )
    return transverse_map


def _cross(o: FloatArray, a: FloatArray, b: FloatArray) -> float:
    """Signed twice-area of triangle ``(o, a, b)``."""
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _convex_hull(points: FloatArray) -> FloatArray:
    """Andrew monotone-chain hull of planar points in O(n log n)."""
    pts = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if pts.shape[0] <= 2:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    ordered = pts[order]
    lower: list[FloatArray] = []
    for point in ordered:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[FloatArray] = []
    for point in ordered[::-1]:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


_DIRECT_HULL_LIMIT = 4096


def _direct_diameter(points: FloatArray) -> float:
    """Maximum chord by pairwise search over the hull in the principal frame.

    Rounding can make the monotone-chain hull of an almost collinear path
    (any slip map whose increments keep one direction) non-convex, which
    stops the calipers early. In the principal frame the sort order follows
    the path, so a vertex dropped by a rounding sign lies between its
    neighbours and the pairwise maximum over the remaining vertices is the
    diameter to rounding. Memory is O(h).
    """
    pts = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    centred = pts - pts.mean(axis=0)
    _values, vectors = np.linalg.eigh(centred.T @ centred)
    hull = _convex_hull(centred @ vectors)
    best = 0.0
    for i in range(hull.shape[0] - 1):
        delta = hull[i + 1 :] - hull[i]
        best = max(best, float(np.max(delta[:, 0] ** 2 + delta[:, 1] ** 2)))
    return math.sqrt(best)


def _trajectory_diameter(points: FloatArray) -> float:
    """Exact maximum chord of planar points via hull + rotating calipers.

    Memory is O(n), and runtime is O(n log n), replacing the former dense
    ``(n, n, 2)`` difference tensor. This is the planar point-set diameter;
    interior trajectory points cannot define a farther pair than hull vertices.
    Hulls of up to ``_DIRECT_HULL_LIMIT`` vertices are searched pairwise
    (:func:`_direct_diameter`), which is robust for almost collinear paths;
    larger hulls use rotating calipers.
    """
    hull = _convex_hull(points)
    n = hull.shape[0]
    if n < 2:
        return 0.0
    if n == 2:
        return float(np.linalg.norm(hull[1] - hull[0]))
    if n <= _DIRECT_HULL_LIMIT:
        return _direct_diameter(points)

    max_d2 = 0.0
    j = 1
    for i in range(n):
        ni = (i + 1) % n
        edge = hull[ni] - hull[i]
        while True:
            nj = (j + 1) % n
            vector_now = hull[j] - hull[i]
            vector_next = hull[nj] - hull[i]
            area_now = abs(float(edge[0] * vector_now[1] - edge[1] * vector_now[0]))
            area_next = abs(float(edge[0] * vector_next[1] - edge[1] * vector_next[0]))
            if area_next <= area_now:
                break
            j = nj
        for a, b in ((i, j), (ni, j)):
            delta = hull[a] - hull[b]
            max_d2 = max(max_d2, float(delta @ delta))
    return math.sqrt(max_d2)


def mindlin_monotonic(
    delta: float | FloatArray, mu: float, F_n: float, K_t: float
) -> FloatArray:
    """Monotonic Cattaneo-Mindlin tangential force [N] ([Johnson1985] 7.42).

    ``F_t = mu F_n [1 - (1 - delta/delta*)^{3/2}]`` for ``0 <= delta <= delta*``,
    and ``F_t = mu F_n`` (gross slip) beyond, with gross-slip displacement
    ``delta* = 3 mu F_n / (2 K_t)``. The initial slope ``dF_t/d(delta)`` at
    ``delta = 0`` is exactly the no-slip tangential stiffness ``K_t``.
    """
    if np.iscomplexobj(delta) or np.iscomplexobj([mu, F_n, K_t]):
        raise CableInputError("Mindlin inputs must be real")
    d = np.asarray(delta, dtype=np.float64)
    if not np.isfinite(d).all() or not np.isfinite([mu, F_n, K_t]).all():
        raise CableInputError("Mindlin inputs must be finite")
    if mu < 0 or F_n < 0 or K_t <= 0:
        raise CableInputError(
            "friction and normal force must be non-negative; stiffness positive"
        )
    if mu == 0 or F_n == 0:
        return np.zeros_like(d)
    result = np.zeros_like(d)
    positive = d > 0
    if not np.any(positive):
        return result
    log_cap = math.log(mu) + math.log(F_n)
    log_stiffness = math.log(K_t)
    log_displacement = np.log(d[positive])
    log_ratio = math.log(2 / 3) + log_stiffness + log_displacement - log_cap
    ratio = np.exp(np.minimum(log_ratio, 0))
    small = log_ratio < math.log(1e-8)
    interior = (~small) & (ratio < 1)
    log_force = np.full_like(log_ratio, log_cap)
    # Small-r expansion has relative neglected error below 5e-18 here. Using
    # k*delta in log form avoids cancellation and an underflowed delta/delta*.
    log_force[small] = (
        log_stiffness + log_displacement[small] + np.log1p(-0.25 * ratio[small])
    )
    log_force[interior] += np.log(-np.expm1(1.5 * np.log1p(-ratio[interior])))
    with np.errstate(over="raise", invalid="raise"):
        try:
            force = np.exp(log_force)
        except FloatingPointError as error:
            raise CableInputError(
                "Mindlin force exceeds floating-point range"
            ) from error
    if np.any(force == 0) or not np.isfinite(force).all():
        raise CableInputError("Mindlin force is outside floating-point range")
    result[positive] = force
    return result


def stuck_zone_ratio(F_t_mag: float, F_n: float, mu: float) -> float:
    """Stuck-zone radius ratio ``c/a`` ([Johnson1985] 7.39).

    ``c/a = (1 - |F_t|/(mu F_n))^{1/3}``; 1.0 = full stick, 0.0 = gross slip.
    """
    if not all(math.isfinite(value) for value in (F_t_mag, F_n, mu)):
        raise CableInputError("stuck-zone inputs must be finite")
    if mu < 0 or F_n < 0:
        raise CableInputError("friction and normal force must be non-negative")
    if F_n == 0.0 or mu == 0.0:
        return 0.0
    if F_t_mag == 0:
        return 1.0
    cap = mu * F_n
    if 0 < cap < math.inf:
        frac = min(abs(F_t_mag) / cap, 1.0)
    else:
        frac = math.exp(min(math.log(abs(F_t_mag)) - math.log(mu) - math.log(F_n), 0))
    return float((1.0 - frac) ** (1.0 / 3.0))


def geometric_slip_factor(r_helix_m: float, lay_angle_rad: float) -> float:
    """Loxodromic axial-slip scale ``r_h^2 cos^2(alpha)/sin(alpha)`` [m²].

    The scale follows the standard simplified axial-slip displacement
    ``u = c_geo * kappa * cos(phi)``.  The lay angle is measured from the cable
    axis and must be non-zero; a straight component has no finite helical-slip
    coordinate of this form.
    """
    sine = math.sin(lay_angle_rad)
    if abs(sine) <= 1.0e-12:
        raise CableInputError("loxodromic slip requires a non-zero helical lay angle")
    return r_helix_m**2 * math.cos(lay_angle_rad) ** 2 / abs(sine)


def transverse_geodesic_factor(r_helix_m: float, lay_angle_rad: float) -> float:
    """Geodesic transverse-slip scale ``r_h^2 cos(a)(1 + sin^2 a)/sin^2(a)`` [m²].

    In-surface transverse displacement of a frictionless wire from the
    loxodrome to the geodesic per unit curvature, at the azimuth of largest
    longitudinal slip (Saevik 1992, Eq. 2.30). It equals ``Delta . n`` of the
    slip vector of Out and von Morgen (1997, Eqs. 26-27) with ``n`` the
    in-surface normal to the wire. Its ratio to :func:`geometric_slip_factor`
    is ``(1 + sin^2 a)/(sin a cos a)``. Signed with ``sin(a)/|sin(a)|``.
    """
    sine = math.sin(lay_angle_rad)
    if abs(sine) <= 1.0e-12:
        raise CableInputError("geodesic slip requires a non-zero helical lay angle")
    return (
        r_helix_m**2
        * math.cos(lay_angle_rad)
        * (1.0 + sine * sine)
        / (sine * abs(sine))
    )


def slip_increment(
    d_kappa_x: float,
    d_kappa_y: float,
    phi: float | FloatArray,
    c_geo: float,
    transverse_ratio: float = 1.0,
    *,
    transverse_map: TransverseMap | None = None,
    lay_angle_rad: float | None = None,
) -> FloatArray:
    """Kinematic slip increment at azimuth(s) ``phi`` (shape ``(..., 2)``).

    Components are ``(longitudinal, transverse)`` in the contact plane.
    ``transverse_ratio=0`` returns the loxodromic increment for either map,
    bitwise identical to the earlier implementation.

    Parameters
    ----------
    d_kappa_x, d_kappa_y : float
        Curvature increments [1/m].
    phi : float or numpy.ndarray
        Wire azimuth(s) [rad].
    c_geo : float
        Loxodromic scale :func:`geometric_slip_factor` [m²].
    transverse_ratio : float, optional
        ``eta``. For ``transverse_map="geodesic"`` it scales the frictionless
        geodesic transverse slip, ``0 <= eta <= 1``. For the legacy map any
        ``eta >= 0`` is accepted and unity is the equal-scale member.
    transverse_map : {"geodesic", "legacy"} or None, optional
        ``"geodesic"``: transverse slip in phase with the longitudinal slip,
        ``-eta (s_geo/c_geo) d(delta_l)`` with ``s_geo`` from
        :func:`transverse_geodesic_factor` (Out and von Morgen 1997, Eqs.
        26-27; ``eta = 1`` reproduces their frictionless slip vector, with
        the curvature sign of CableXSec). ``"legacy"``: the former map
        ``eta c_geo (d_kx cos phi + d_ky sin phi)``, which is in quadrature
        with the geodesic transverse slip (zero at the neutral axis, largest
        at the extreme fibres). ``None`` selects the legacy map and warns
        (:class:`TransverseMapWarning`) when ``eta > 0``.
    lay_angle_rad : float, optional
        Lay angle from the cable axis [rad]; required by the geodesic map
        when ``eta > 0``.
    """
    resolved = transverse_map_or_warn(transverse_ratio, transverse_map)
    phi_a = np.asarray(phi, dtype=np.float64)
    d1 = c_geo * (d_kappa_x * np.sin(phi_a) - d_kappa_y * np.cos(phi_a))
    if resolved == "geodesic" and transverse_ratio > 0.0:
        if lay_angle_rad is None:
            raise CableInputError("the geodesic transverse map needs lay_angle_rad")
        sine = math.sin(lay_angle_rad)
        cosine = math.cos(lay_angle_rad)
        if abs(sine) <= 1.0e-12 or abs(cosine) <= 1.0e-12:
            raise CableInputError(
                "the geodesic transverse map needs 0 < |lay angle| < 90 deg"
            )
        ratio = (1.0 + sine * sine) / (sine * cosine)
        d2 = (-transverse_ratio * ratio) * d1
        return np.stack([d1, d2], axis=-1)
    d2 = (
        transverse_ratio
        * c_geo
        * (d_kappa_x * np.cos(phi_a) + d_kappa_y * np.sin(phi_a))
    )
    return np.stack([d1, d2], axis=-1)


def loxodromic_slip_increment(
    d_kappa_x: float,
    d_kappa_y: float,
    phi: float | FloatArray,
    c_geo: float,
) -> FloatArray:
    """Scalar loxodromic projection embedded in the contact plane.

    This is :func:`slip_increment` with ``transverse_ratio=0`` (either
    map). For fixed-plane bending its magnitude is
    ``c_geo * abs(d_kappa_x * sin(phi))`` when ``phi=0`` is the extreme
    fibre; it is maximal at the neutral axis. This 90-degree phase relative
    to bending strain follows by integrating the wire axial strain along the
    helix.
    """
    phi_a = np.asarray(phi, dtype=np.float64)
    axial = c_geo * (d_kappa_x * np.sin(phi_a) - d_kappa_y * np.cos(phi_a))
    return np.stack([axial, np.zeros_like(axial)], axis=-1)


@dataclass
class MindlinContact:
    """Stateful 2-D Coulomb contact with Mindlin initial stiffness.

    Parameters
    ----------
    K_t : float
        Area-equivalent tangential (no-slip) stiffness [N/m] (from Hertz,
        ``8 G* sqrt(a b)``).
    mu : float
        Friction coefficient.

    Notes
    -----
    The contact normal force is supplied per ``update`` (it varies with the
    cable state), so non-proportional load paths can be represented. The slip
    (plastic) trajectory ``s(t) = delta_t(t) - F_t(t)/K_t`` is accumulated; the
    fretting amplitude ``delta*`` is half the maximum chord of that trajectory.
    """

    K_t: float
    mu: float
    _element: CoulombElement = field(init=False)
    _delta_total: FloatArray = field(init=False)
    _slip_traj: list[FloatArray] = field(init=False, default_factory=list)
    _p0: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if not math.isfinite(self.mu) or self.mu < 0:
            raise CableInputError("friction coefficient must be finite and nonnegative")
        self._element = CoulombElement(k=self.K_t, vector=True)
        self._delta_total = np.zeros(2, dtype=np.float64)
        self._slip_traj = [np.zeros(2)]

    def update(
        self, d_delta_t: FloatArray, F_n_current: float, p0_current: float = 0.0
    ) -> tuple[FloatArray, str, float]:
        """Advance one increment.

        Parameters
        ----------
        d_delta_t : numpy.ndarray, shape (2,)
            Tangential slip increment.
        F_n_current : float
            Current contact normal force [N] (sets the Coulomb limit
            ``mu F_n`` this step — non-proportional handling).
        p0_current : float, optional
            Current peak Hertz pressure [Pa] (for the shear-slip product).

        Returns
        -------
        F_t : numpy.ndarray, shape (2,)
            Updated tangential force vector [N].
        state : str
            ``"stick"`` or ``"slip"``.
        c_over_a : float
            Stuck-zone ratio.
        """
        if np.iscomplexobj(d_delta_t):
            raise CableInputError("contact increment must be real")
        d = np.asarray(d_delta_t, dtype=np.float64)
        if d.shape != (2,) or not np.isfinite(d).all():
            raise CableInputError("contact increment must have two finite components")
        if not all(
            math.isfinite(v) and v >= 0 for v in (self.mu, F_n_current, p0_current)
        ):
            raise CableInputError(
                "friction, normal load and peak pressure must be finite and nonnegative"
            )
        if self.K_t != self._element.k:
            raise CableInputError(
                "contact stiffness must remain fixed after construction"
            )
        limit = self.mu * F_n_current
        f_t, slip, slipping = coulomb_return_map(
            self._element.f, d, self.K_t, limit, vector=True
        )
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                delta = self._delta_total + d
                total_slip = self._element.slip_total + slip
                s = delta - f_t / self.K_t
            except FloatingPointError as error:
                raise CableInputError(
                    "contact state exceeds floating-point range"
                ) from error
        if not all(np.isfinite(v).all() for v in (delta, total_slip, s)):
            raise CableInputError("contact state must be finite")
        c_a = stuck_zone_ratio(float(np.hypot(f_t[0], f_t[1])), F_n_current, self.mu)
        # Commit only after every validation and derived calculation succeeds.
        self._element.f, self._element.slip_total = f_t, total_slip
        self._delta_total = delta
        self._p0 = max(self._p0, p0_current)
        self._slip_traj.append(s.copy())
        return f_t, ("slip" if bool(slipping) else "stick"), c_a

    @property
    def fretting_amplitude(self) -> float:
        """``delta* = 1/2 max_{i,j} |s(t_i) - s(t_j)|`` (half the max slip chord)."""
        traj = np.array(self._slip_traj)
        if traj.shape[0] < 2:
            return 0.0
        return 0.5 * _trajectory_diameter(traj)

    def shear_slip_product(self) -> float:
        """Shear-slip product ``Phi = mu p0 delta*`` [N/m] for this contact.

        ``mu`` is the friction coefficient, ``p0`` the largest peak Hertz
        pressure supplied to :meth:`update` and ``delta*`` the slip amplitude
        :attr:`fretting_amplitude`. It is a contact-loading indicator for
        ranking wires and comparing designs. It is not the Ruiz parameter
        ([Ruiz1984]), which uses the local surface shear traction and slip
        and, in its second form, the local tangential tensile stress, and it
        is not a fretting life.
        """
        return self.mu * self._p0 * self.fretting_amplitude

    def ruiz_parameter(self) -> float:
        """Deprecated alias of :meth:`shear_slip_product`, kept for callers.

        Returns ``Phi = mu p0 delta*``. Despite its name it does not compute
        the Ruiz parameter of [Ruiz1984]; see :meth:`shear_slip_product`.
        """
        return self.shear_slip_product()
