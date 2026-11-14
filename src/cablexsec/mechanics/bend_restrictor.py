# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Curvature imposed by an articulated bend restrictor or a bend stiffener.

Gap P5. A cable that leaves a fixed root (a hang-off, a J-tube exit or a
test-rig headstock) at an angle ``theta`` to the root axis bends over a
length set by its own bending stiffness ``EI`` and tension ``T``. A bend
restrictor or a bend stiffener changes that curvature field ``kappa(x, t)``,
and the change is *not proportional*: a restrictor locks joint by joint, so
the bent length grows with the angle. This module returns the field for a
history of the root angle, for the rig and bending-test simulators
(:func:`~cablexsec.analysis.rig_test.simulate_rig_test`,
:func:`~cablexsec.analysis.beam_test.simulate_bending_test`) and the
termination clamp correction
(:class:`~cablexsec.analysis.termination.TerminationSpec`).

Articulated restrictor
----------------------
A chain of ``n`` rigid elements (vertebrae) of pitch ``p`` (joint spacing:
element length minus overlap) and bore diameter ``D_b``. Adjacent elements
rotate freely up to the locking angle ``theta_L`` and are rigid beyond it.
Rigid-link chain geometry: the locked chain is a regular polygon with
vertices at the joints, circumradius ``R_L = p / (2 sin(theta_L / 2))``,
equal to the arc radius ``p / theta_L`` within ``theta_L^2 / 24``
(:attr:`ArticulatedRestrictor.locking_radius_m`).

The cable centreline may deviate from the restrictor axis by the bore
clearance ``c = (D_b - D_c) / 2``. In a locked zone it lies on a circle
concentric with the polygon whose radius ``R_c`` stays inside the band
``[R_L - c, R_L cos(theta_L/2) + c]`` (outside the link at mid-element,
inside at the joints; feasible when the chord sagitta
``R_L (1 - cos(theta_L/2)) <= 2 c``). Of the circles in the band the cable
takes the one of least bending plus tension energy per unit turning,
``EI / (2 R_c) + T R_c``, i.e. ``R_c = sqrt(EI / (2 T))`` clipped to the
band (:meth:`ArticulatedRestrictor.cable_locking_radius_m`). Its curvature
``kappa_L = 1 / R_c`` is the cap on the cable curvature inside the
restrictor.

Outside the locked zone the restrictor carries no moment, so the cable is a
small-deflection tensioned beam, ``EI w'''' - T w'' = 0``, between the end
of the locked zone and the far end of the span (``w = w' = 0`` there; the
free length is ``span_m``), with decay length ``lambda = sqrt(EI / T)``
(the untensioned beam for ``lambda > 20 span``). The locked zone
``[0, a]`` carries the cable on the lock curvature, so at its end the cable
has turned by ``kappa_L a`` and the free beam starts with the remaining
rotation ``theta - kappa_L a``. Two lock rules:

``"joints"`` (default)
    The element ``k`` turns relative to its neighbour by the cable rotation
    over its pitch, ``int kappa dx`` over ``[(k-1) p, k p]``; it locks when
    that reaches ``kappa_L p`` (``theta_L`` for zero clearance). The locked
    zone is a whole number of elements, the smallest for which no free
    element inside the restrictor exceeds its lock: the bent length grows
    element by element as the angle grows.
``"smeared"``
    Continuous lock front ``a``: the cable moment is continuous at the front
    (``EI kappa(a+) = EI kappa_L``, the restrictor carries no moment there).
    For a long span it has the closed form ``a = theta / kappa_L - lambda``
    for ``theta > kappa_L lambda`` (and no lock below).

The lock limits the turning of each element, not the local curvature: in
the free element next to the lock front (or at the root before any lock)
the cable may curve more tightly than ``kappa_L`` when ``lambda`` is
comparable with ``p``. A straight bore of length ``p`` admits a cable arc of
curvature up to ``16 c / p^2`` (chord deviation ``kappa p^2 / 8`` within the
band ``2 c``); a larger peak is flagged.

When every element is locked the cable beyond the tip carries the remaining
rotation and its curvature may exceed ``kappa_L``
(:attr:`CurvatureField.tip_overbend`): the classic over-bending at a
restrictor tip.

Bend stiffener
--------------
A conical polyurethane sleeve of length ``L_s``, root and tip outer
diameters ``D_r``, ``D_t``, bore equal to the cable diameter and modulus
``E_s`` (linear elastic) adds ``E_s pi (D(s)^4 - D_c^4) / 64`` to the cable
stiffness. The cable and stiffener are a large-deflection Euler beam,
``(EI(s) phi')' = -T sin(theta - phi)``, clamped at the root
(``phi(0) = 0``) and pulled at its far end by the tension ``T`` at the angle
``theta`` (``EI phi' = 0`` there), the problem of [CaireVaz2007] and Boef
and Out (1990) without the polymer's non-linearity. Without stiffener and
for small angles it reduces to ``phi = theta (1 - exp(-s / lambda))``.

Assumptions (stated)
--------------------
* Planar bending in one plane; the curvature has the sign of ``theta``.
* Restrictor: small rotations, linear cable ``EI`` (a secant value for a
  hysteretic cable), frictionless rigid elements, no restrictor
  self-weight or contact friction; the cable curvature in a locked zone is
  the concentric-arc value above (local curvature peaks at the element
  edges are not resolved). The field is a function of the current angle and
  tension only (no hysteresis of the restrictor).
* Stiffener: linear elastic polyurethane, perfect bond to the cable, no
  temperature or rate dependence, rigid root.

References
----------
[Thies2016] Sec. 2.2 and Table 1 (restrictor data); [CaireVaz2007] (bend
stiffener equations); [LuVazCaire2022] (stiffener and armour stress).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cablexsec.exceptions import CableInputError, CableSolverError
from cablexsec.provenance import Provenance

__all__ = [
    "LOCK_RULES",
    "ArticulatedRestrictor",
    "BendStiffener",
    "CurvatureField",
    "restrictor_curvature_field",
    "stiffener_curvature_field",
    "tensioned_beam_profile",
]

FloatArray = NDArray[np.float64]
LockRule = Literal["joints", "smeared"]
LOCK_RULES: tuple[str, ...] = ("joints", "smeared")

#: Above ``POLYNOMIAL_RATIO * span`` the decay length is treated as infinite
#: (untensioned beam), as in :func:`cablexsec.analysis.rig_test.end_rotation_profile`.
POLYNOMIAL_RATIO = 20.0


def _positive(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise CableInputError(f"{name} must be finite and positive")
    return float(value)


@dataclass(frozen=True)
class ArticulatedRestrictor:
    """Chain of rigid elements with a locking angle (module notes).

    Attributes
    ----------
    pitch_m : float
        Joint spacing along the chain [m] (element length minus overlap).
    lock_angle_rad : float
        Relative rotation of two adjacent elements at locking [rad].
    n_elements : int
        Number of elements (the restrictor length is ``n p``).
    bore_diameter_m : float
        Inner diameter of the elements [m].
    source : str
        Where the data come from.
    stated_mbr_m : float or None
        Minimum bend radius stated by the supplier, reported beside the
        rigid-link value (they need not agree).
    """

    pitch_m: float
    lock_angle_rad: float
    n_elements: int
    bore_diameter_m: float
    source: str = ""
    stated_mbr_m: float | None = None

    def __post_init__(self) -> None:
        _positive(self.pitch_m, "pitch_m")
        _positive(self.bore_diameter_m, "bore_diameter_m")
        if not (
            math.isfinite(self.lock_angle_rad) and 0 < self.lock_angle_rad < math.pi / 2
        ):
            raise CableInputError("lock_angle_rad must lie in (0, pi/2)")
        if (
            isinstance(self.n_elements, bool)
            or not isinstance(self.n_elements, (int, np.integer))
            or self.n_elements < 1
        ):
            raise CableInputError("n_elements must be a positive integer")
        if self.stated_mbr_m is not None:
            _positive(self.stated_mbr_m, "stated_mbr_m")

    @classmethod
    def from_mbr(
        cls,
        mbr_m: float,
        *,
        pitch_m: float,
        n_elements: int,
        bore_diameter_m: float,
        source: str = "",
    ) -> ArticulatedRestrictor:
        """Restrictor whose rigid-link locking radius equals ``mbr_m``.

        ``theta_L = 2 asin(p / (2 MBR))``.
        """
        _positive(mbr_m, "mbr_m")
        _positive(pitch_m, "pitch_m")
        if pitch_m >= 2.0 * mbr_m:
            raise CableInputError("the pitch must be shorter than the MBR diameter")
        return cls(
            pitch_m=pitch_m,
            lock_angle_rad=2.0 * math.asin(pitch_m / (2.0 * mbr_m)),
            n_elements=n_elements,
            bore_diameter_m=bore_diameter_m,
            source=source,
            stated_mbr_m=mbr_m,
        )

    @property
    def length_m(self) -> float:
        """Restrictor length ``n p`` [m]."""
        return float(self.n_elements * self.pitch_m)

    @property
    def locking_radius_m(self) -> float:
        """Rigid-link chain: circumradius ``p / (2 sin(theta_L / 2))`` [m]."""
        return self.pitch_m / (2.0 * math.sin(0.5 * self.lock_angle_rad))

    @property
    def arc_locking_radius_m(self) -> float:
        """Arc approximation ``p / theta_L`` [m]."""
        return self.pitch_m / self.lock_angle_rad

    @property
    def sagitta_m(self) -> float:
        """Chord departure from the locking circle, ``R_L (1 - cos(theta_L/2))``."""
        return self.locking_radius_m * (1.0 - math.cos(0.5 * self.lock_angle_rad))

    def clearance_m(self, cable_od_m: float) -> float:
        """Radial bore clearance ``(D_b - D_c) / 2`` [m] (must be >= 0)."""
        _positive(cable_od_m, "cable_od_m")
        c = 0.5 * (self.bore_diameter_m - cable_od_m)
        if c < 0.0:
            raise CableInputError(
                f"cable diameter {cable_od_m:.6g} m exceeds the restrictor bore "
                f"{self.bore_diameter_m:.6g} m"
            )
        return c

    def cable_locking_radius_m(
        self, cable_od_m: float, decay_length_m: float = math.inf
    ) -> float:
        """Radius of the cable in a locked zone (module notes) [m].

        ``sqrt(EI / (2 T)) = decay_length / sqrt(2)`` clipped to the band
        ``[R_L - c, R_L cos(theta_L/2) + c]``; an infinite decay length
        (no tension) gives the outer end of the band. With a band narrower
        than the chord sagitta (``R_L (1 - cos) > 2 c``) the cable cannot
        follow a circle; the rigid-link radius ``R_L`` is returned and the
        field notes flag it.
        """
        c = self.clearance_m(cable_od_m)
        r_l = self.locking_radius_m
        lower = r_l - c
        upper = r_l * math.cos(0.5 * self.lock_angle_rad) + c
        if upper < lower:
            return r_l
        preferred = (
            math.inf
            if not math.isfinite(decay_length_m)
            else decay_length_m / math.sqrt(2.0)
        )
        return float(min(max(preferred, lower), upper))

    def summary(self, cable_od_m: float | None = None) -> dict[str, float]:
        """Geometry figures: radii, length, sagitta, clearance."""
        out = {
            "pitch_m": self.pitch_m,
            "lock_angle_deg": math.degrees(self.lock_angle_rad),
            "length_m": self.length_m,
            "locking_radius_m": self.locking_radius_m,
            "arc_locking_radius_m": self.arc_locking_radius_m,
            "sagitta_m": self.sagitta_m,
        }
        if self.stated_mbr_m is not None:
            out["stated_mbr_m"] = self.stated_mbr_m
            out["lock_angle_from_stated_mbr_deg"] = math.degrees(
                2.0 * math.asin(min(1.0, self.pitch_m / (2.0 * self.stated_mbr_m)))
            )
        if cable_od_m is not None:
            out["clearance_m"] = self.clearance_m(cable_od_m)
            out["cable_locking_radius_m"] = self.cable_locking_radius_m(cable_od_m)
        return out


@dataclass(frozen=True)
class BendStiffener:
    """Conical polyurethane bend stiffener (module notes).

    Attributes
    ----------
    length_m : float
        Length of the conical part from the root [m].
    root_od_m, tip_od_m : float
        Outer diameter at the root and at the tip [m].
    E_Pa : float
        Polyurethane modulus (linear) [Pa].
    source : str
        Where the data come from.
    """

    length_m: float
    root_od_m: float
    tip_od_m: float
    E_Pa: float
    source: str = ""

    def __post_init__(self) -> None:
        for name in ("length_m", "root_od_m", "tip_od_m", "E_Pa"):
            _positive(getattr(self, name), name)
        if self.tip_od_m > self.root_od_m:
            raise CableInputError("the stiffener tip may not be wider than the root")

    def bending_stiffness(
        self, s_m: ArrayLike, cable_od_m: float, cable_EI_Nm2: float
    ) -> FloatArray:
        """``EI_cable + E_s pi (D(s)^4 - D_c^4) / 64`` along the cable [N m^2]."""
        _positive(cable_od_m, "cable_od_m")
        _positive(cable_EI_Nm2, "cable_EI_Nm2")
        if cable_od_m >= self.tip_od_m:
            raise CableInputError("the stiffener tip must be wider than the cable")
        s = np.asarray(s_m, dtype=np.float64)
        frac = np.clip(s / self.length_m, 0.0, 1.0)
        diameter = self.root_od_m + (self.tip_od_m - self.root_od_m) * frac
        added = self.E_Pa * math.pi * (diameter**4 - cable_od_m**4) / 64.0
        return np.asarray(cable_EI_Nm2 + np.where(s <= self.length_m, added, 0.0))


@dataclass(frozen=True)
class CurvatureField:
    """Curvature ``kappa(x, t)`` along the cable from the root [1/m].

    ``kappa_1_m`` has shape ``(x, time)``; ``x_m`` increases from the root
    (``x = 0``). ``locked_length_m`` is the locked (restrictor) length per
    sample, ``n_locked`` the number of locked elements, ``tip_overbend``
    flags samples whose curvature beyond a fully locked restrictor exceeds
    the lock curvature. Values between stations are linear in ``x``.
    """

    x_m: FloatArray
    kappa_1_m: FloatArray
    time_s: FloatArray | None
    kind: str
    locked_length_m: FloatArray
    n_locked: NDArray[np.int64]
    tip_overbend: NDArray[np.bool_]
    lock_curvature_1_m: float
    decay_length_m: FloatArray
    provenance: Provenance
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        x = np.asarray(self.x_m, dtype=np.float64)
        k = np.asarray(self.kappa_1_m, dtype=np.float64)
        if x.ndim != 1 or x.size < 2 or np.any(np.diff(x) <= 0) or x[0] < 0:
            raise CableInputError("field stations must start at x >= 0 and increase")
        if k.ndim != 2 or k.shape[0] != x.size or not np.isfinite(k).all():
            raise CableInputError("kappa must be a finite (x, time) array")

    @property
    def n_samples(self) -> int:
        return int(self.kappa_1_m.shape[1])

    def at(self, x_m: float) -> FloatArray:
        """Curvature history at one station [1/m] (zero beyond the stations)."""
        x = float(x_m)
        xs = self.x_m
        if x < xs[0] or x > xs[-1]:
            return np.zeros(self.n_samples)
        j = int(np.searchsorted(xs, x, side="right")) - 1
        j = min(max(j, 0), xs.size - 2)
        w = (x - xs[j]) / (xs[j + 1] - xs[j])
        return np.asarray((1.0 - w) * self.kappa_1_m[j] + w * self.kappa_1_m[j + 1])

    def sample(self, x_m: ArrayLike) -> FloatArray:
        """Curvature at stations ``x_m`` [1/m], shape ``(x, time)``."""
        xs = np.atleast_1d(np.asarray(x_m, dtype=np.float64))
        return np.asarray(np.stack([self.at(float(v)) for v in xs]))

    def peak(self) -> float:
        """Largest ``|kappa|`` over the field [1/m]."""
        return float(np.max(np.abs(self.kappa_1_m)))

    def forward_peak(self) -> float:
        """Largest curvature with the sign it has at the root (reverse
        bending near the far end excluded) [1/m]."""
        root = self.kappa_1_m[0]
        sign = np.where(root != 0.0, np.sign(root), 1.0)
        return float(max(0.0, float(np.max(self.kappa_1_m * sign[None, :]))))

    def bent_length_m(self, fraction: float = 0.5) -> FloatArray:
        """Per sample, the end of the zone around the forward peak (the
        curvature with the sign it has at the root) where the curvature
        keeps at least ``fraction`` of that peak [m] (zero for a straight
        sample); reverse bending near the far end of the span is not
        counted."""
        if not 0.0 < fraction <= 1.0:
            raise CableInputError("fraction must lie in (0, 1]")
        out = np.zeros(self.n_samples)
        for t in range(self.n_samples):
            col = self.kappa_1_m[:, t]
            first = float(col[0])
            if first == 0.0:
                first = float(col[np.argmax(np.abs(col))])
            if first == 0.0:
                continue
            forward = col * math.copysign(1.0, first)
            i = int(np.argmax(forward))
            top = float(forward[i])
            keep = forward >= fraction * top * (1.0 - 1e-12)
            j = i
            while j + 1 < col.size and keep[j + 1]:
                j += 1
            out[t] = float(self.x_m[j])
        return out


# --------------------------------------------------------------------------- #
# Tensioned beam                                                               #
# --------------------------------------------------------------------------- #
def _free_beam(
    a: float,
    length: float,
    w_a: float,
    phi_a: float,
    lam: float,
) -> tuple[FloatArray, bool]:
    """Coefficients of the free beam on ``[a, L]`` (``w = w' = 0`` at ``L``).

    Exponential basis ``c0 + c1 s + c2 lam^2 e^{-s/lam} + c3 lam^2 e^{-(l-s)/lam}``
    or, for an untensioned beam, ``c0 + c1 s + c2 s^2 + c3 s^3`` with
    ``s = x - a``, ``l = L - a``.
    """
    ell = length - a
    if ell <= 0.0:
        return np.zeros(4), True
    poly = not math.isfinite(lam) or lam > POLYNOMIAL_RATIO * length
    if poly:
        matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, ell, ell**2, ell**3],
                [0.0, 1.0, 2.0 * ell, 3.0 * ell**2],
            ]
        )
    else:
        e = math.exp(-ell / lam)
        matrix = np.array(
            [
                [1.0, 0.0, lam**2, lam**2 * e],
                [0.0, 1.0, -lam, lam * e],
                [1.0, ell, lam**2 * e, lam**2],
                [0.0, 1.0, -lam * e, lam],
            ]
        )
    coef = np.linalg.solve(matrix, np.array([w_a, phi_a, 0.0, 0.0]))
    return np.asarray(coef), poly


def _free_eval(
    coef: FloatArray, poly: bool, a: float, length: float, lam: float, x: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``w, w', w''`` of the free beam at ``x >= a``."""
    s = x - a
    ell = length - a
    if poly:
        w = coef[0] + coef[1] * s + coef[2] * s**2 + coef[3] * s**3
        dw = coef[1] + 2 * coef[2] * s + 3 * coef[3] * s**2
        d2 = 2 * coef[2] + 6 * coef[3] * s
    else:
        e1 = np.exp(-s / lam)
        e2 = np.exp(-(ell - s) / lam)
        w = coef[0] + coef[1] * s + lam**2 * (coef[2] * e1 + coef[3] * e2)
        dw = coef[1] - lam * coef[2] * e1 + lam * coef[3] * e2
        d2 = coef[2] * e1 + coef[3] * e2
    return np.asarray(w), np.asarray(dw), np.asarray(d2)


def tensioned_beam_profile(
    x_m: ArrayLike,
    span_m: float,
    decay_length_m: float,
    rotation_rad: float = 1.0,
) -> FloatArray:
    """Curvature of the free tensioned beam under a root rotation [1/m].

    ``EI w'''' - T w'' = 0`` on ``[0, L]``, ``w(0) = 0``, ``w'(0) = theta``,
    ``w(L) = w'(L) = 0``; returns ``-w''`` (positive for positive
    ``theta``). Equals ``-theta`` times
    :func:`cablexsec.analysis.rig_test.end_rotation_profile`.
    """
    x = np.asarray(x_m, dtype=np.float64)
    coef, poly = _free_beam(
        0.0, float(span_m), 0.0, float(rotation_rad), decay_length_m
    )
    return np.asarray(-_free_eval(coef, poly, 0.0, float(span_m), decay_length_m, x)[2])


def _lock_state(
    theta: float,
    lam: float,
    span: float,
    kappa_lock: float,
    pitch: float,
    n_el: int,
    rule: str,
) -> tuple[float, int]:
    """Locked length ``a`` and element count for ``theta >= 0``."""
    if theta <= 0.0:
        return 0.0, 0
    restrictor_end = min(n_el * pitch, span)

    def front_curvature(a: float) -> float:
        coef, poly = _free_beam(
            a, span, theta * a - 0.5 * kappa_lock * a * a, theta - kappa_lock * a, lam
        )
        return float(-_free_eval(coef, poly, a, span, lam, np.array([a]))[2][0])

    if rule == "smeared":
        if front_curvature(0.0) <= kappa_lock:
            return 0.0, 0
        if front_curvature(restrictor_end) > kappa_lock:
            return restrictor_end, n_el
        lo, hi = 0.0, restrictor_end
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if front_curvature(mid) > kappa_lock:
                lo = mid
            else:
                hi = mid
            if hi - lo <= 1e-12 * max(1.0, span):
                break
        a = 0.5 * (lo + hi)
        return a, int(math.floor(a / pitch + 1e-9))
    tol = 1e-12 * kappa_lock * pitch
    n_max = min(n_el, int(math.floor(span / pitch + 1e-9)))
    for j in range(n_max + 1):
        a = j * pitch
        coef, poly = _free_beam(
            a, span, theta * a - 0.5 * kappa_lock * a * a, theta - kappa_lock * a, lam
        )
        joints = np.arange(j, n_max + 1, dtype=np.float64) * pitch
        if joints.size < 2:
            return a, j
        dw = _free_eval(coef, poly, a, span, lam, joints)[1]
        turning = -np.diff(dw)
        if np.all(turning <= kappa_lock * pitch + tol):
            return a, j
    return n_max * pitch, n_max


def restrictor_curvature_field(
    restrictor: ArticulatedRestrictor,
    rotation_rad: ArrayLike,
    *,
    cable_od_m: float,
    bending_stiffness_Nm2: float,
    tension_N: ArrayLike = 0.0,
    span_m: float | None = None,
    x_m: ArrayLike | None = None,
    time_s: ArrayLike | None = None,
    rule: LockRule = "joints",
) -> CurvatureField:
    """Cable curvature inside and beyond an articulated restrictor (module notes).

    Parameters
    ----------
    restrictor : ArticulatedRestrictor
    rotation_rad : array_like
        Root angle history ``theta(t)`` [rad]: the cable direction at the
        root relative to its direction at the far end of the span.
    cable_od_m : float
        Cable outer diameter [m] (bore clearance).
    bending_stiffness_Nm2 : float
        Cable bending stiffness ``EI`` (a secant value for a hysteretic
        cable) [N m^2].
    tension_N : array_like
        Axial load of the cable and restrictor that acts on the bent shape
        (scalar or per sample) [N]; zero gives the untensioned beam.
    span_m : float or None
        Free length from the root to the far end where the cable direction
        is held (default: the restrictor length plus ten decay lengths at
        the smallest positive tension, or twice the restrictor length).
    x_m : array_like or None
        Output stations (default 401 stations over the span).
    time_s : array_like or None
        Sample times carried with the field.
    rule : {"joints", "smeared"}
        Lock rule (module notes).
    """
    if rule not in LOCK_RULES:
        raise CableInputError(f"rule must be one of {LOCK_RULES}")
    theta = np.atleast_1d(np.asarray(rotation_rad, dtype=np.float64))
    if theta.ndim != 1 or not np.isfinite(theta).all():
        raise CableInputError("rotation history must be a finite 1-D array")
    tension = np.broadcast_to(np.asarray(tension_N, dtype=np.float64), theta.shape)
    if not np.isfinite(tension).all() or np.any(tension < 0):
        raise CableInputError("tension must be finite and non-negative")
    ei = _positive(bending_stiffness_Nm2, "bending_stiffness_Nm2")
    restrictor.clearance_m(cable_od_m)
    lam_samples = np.where(
        tension > 0, np.sqrt(ei / np.where(tension > 0, tension, 1.0)), math.inf
    )
    if span_m is None:
        finite = lam_samples[np.isfinite(lam_samples)]
        span = (
            restrictor.length_m + 10.0 * float(np.max(finite))
            if finite.size
            else 2.0 * restrictor.length_m
        )
    else:
        span = _positive(span_m, "span_m")
    xs = (
        np.linspace(0.0, span, 401)
        if x_m is None
        else np.asarray(x_m, dtype=np.float64)
    )
    if (
        xs.ndim != 1
        or xs.size < 2
        or np.any(np.diff(xs) <= 0)
        or xs[0] < 0
        or xs[-1] > span
    ):
        raise CableInputError("x_m must increase within [0, span]")
    pitch = restrictor.pitch_m
    n_el = restrictor.n_elements
    kappa = np.zeros((xs.size, theta.size))
    locked = np.zeros(theta.size)
    count = np.zeros(theta.size, dtype=np.int64)
    overbend = np.zeros(theta.size, dtype=bool)
    lock_curv = np.zeros(theta.size)
    cache: dict[tuple[float, float], tuple[FloatArray, float, int, bool, float]] = {}
    for t in range(theta.size):
        key = (abs(float(theta[t])), float(tension[t]))
        if key not in cache:
            mag, ten = key
            lam = float(lam_samples[t])
            k_lock = 1.0 / restrictor.cable_locking_radius_m(cable_od_m, lam)
            a, j = _lock_state(mag, lam, span, k_lock, pitch, n_el, rule)
            coef, poly = _free_beam(
                a, span, mag * a - 0.5 * k_lock * a * a, mag - k_lock * a, lam
            )
            free = -_free_eval(coef, poly, a, span, lam, np.maximum(xs, a))[2]
            row = np.where(xs < a, k_lock, free)
            end = restrictor.length_m
            tip = bool(
                j >= n_el
                and end < span
                and float(-_free_eval(coef, poly, a, span, lam, np.array([end]))[2][0])
                > k_lock * (1.0 + 1e-9)
            )
            cache[key] = (np.asarray(row), a, j, tip, k_lock)
        row, a, j, tip, k_lock = cache[key]
        sign = 1.0 if theta[t] >= 0 else -1.0
        kappa[:, t] = sign * row
        locked[t] = a
        count[t] = j
        overbend[t] = tip
        lock_curv[t] = k_lock
    notes: list[str] = []
    clearance = restrictor.clearance_m(cable_od_m)
    if restrictor.sagitta_m > 2.0 * clearance:
        notes.append(
            f"chord sagitta {restrictor.sagitta_m * 1e3:.3g} mm exceeds twice the "
            f"bore clearance ({clearance * 1e3:.3g} mm): the cable cannot follow a "
            "circle in a locked zone; the rigid-link radius is used and local "
            "curvature peaks at the joints are not resolved"
        )
    within = xs <= restrictor.length_m
    # A free (unlocked) element is a straight bore: the cable arc inside it
    # deviates from the chord by kappa p^2 / 8, at most the band 2 c.
    bore_limit = 16.0 * clearance / pitch**2
    peak_free = float(np.max(np.abs(kappa[within]))) if np.any(within) else 0.0
    if peak_free > bore_limit * (1 + 1e-9):
        notes.append(
            f"peak cable curvature {peak_free:.4g} 1/m inside the restrictor exceeds "
            f"the straight-bore limit 16 c / p^2 = {bore_limit:.4g} 1/m: the cable "
            "would bear on the element bore (not modelled)"
        )
    reverse = np.any(
        kappa[within] * np.sign(theta)[None, :] < -lock_curv[None, :] * (1 + 1e-9)
    )
    if reverse:
        notes.append(
            "reverse curvature towards the far end of the span exceeds the lock "
            "curvature inside the restrictor: a lock there is not modelled "
            "(locking proceeds from the root only)"
        )
    if np.any(overbend):
        notes.append(
            "every element locked: the cable beyond the restrictor tip bends "
            "tighter than the lock radius (over-bending at the tip)"
        )
    if restrictor.stated_mbr_m is not None and not math.isclose(
        restrictor.stated_mbr_m, restrictor.locking_radius_m, rel_tol=0.02
    ):
        notes.append(
            f"stated MBR {restrictor.stated_mbr_m:.4g} m differs from the "
            f"rigid-link radius {restrictor.locking_radius_m:.4g} m of the pitch "
            "and locking angle: run both"
        )
    k_ref = float(lock_curv[np.argmax(np.abs(theta))]) if theta.size else 0.0
    return CurvatureField(
        x_m=np.asarray(xs),
        kappa_1_m=kappa,
        time_s=None if time_s is None else np.asarray(time_s, dtype=np.float64),
        kind=f"articulated restrictor ({rule} lock rule)",
        locked_length_m=locked,
        n_locked=count,
        tip_overbend=overbend,
        lock_curvature_1_m=k_ref,
        decay_length_m=np.asarray(lam_samples),
        provenance=Provenance(
            method=(
                "articulated bend restrictor: rigid-link chain locking at "
                f"{math.degrees(restrictor.lock_angle_rad):.4g} deg per joint "
                f"(R_L {restrictor.locking_radius_m:.4g} m), cable on the "
                "concentric locked arc inside the bore clearance, small-"
                "deflection tensioned beam beyond the lock front"
            ),
            citations=("[Thies2016] Sec. 2.2", "rigid-link chain geometry"),
            gates=(
                "no lock: equal to the tensioned-beam end-rotation profile",
                "smeared rule, long span: lock front a = theta / kappa_L - lambda",
                "short decay length: one element locks per theta_L of rotation",
            ),
            limitations=(
                "small rotations, linear cable EI, frictionless rigid elements",
                "locking proceeds from the root; a lock of the reverse curvature "
                "near the far end of the span is flagged, not modelled",
                "planar bending; curvature is a function of the current angle "
                "and tension (no restrictor hysteresis)",
                "local curvature peaks at the element edges not resolved",
            ),
        ),
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------- #
# Bend stiffener                                                               #
# --------------------------------------------------------------------------- #
def _stiffener_solution(
    stiffener: BendStiffener | None,
    angle: float,
    tension: float,
    cable_od_m: float,
    ei_cable: float,
    s_end: float,
    mesh: FloatArray,
    guess: FloatArray | None,
) -> tuple[FloatArray, FloatArray]:
    """``(phi, phi')`` on ``mesh`` of the large-deflection stiffener problem.

    Conservative finite differences, ``(F_{i+1/2} - F_{i-1/2}) / hbar_i +
    T sin(theta - phi_i) = 0`` with ``F = EI phi'`` at the cell midpoints
    (stiffness jumps are exact at the nodes), ``phi_0 = 0`` and ``F = 0``
    beyond the last node; Newton iterations on the tridiagonal system.
    """
    from scipy.linalg import solve_banded  # noqa: PLC0415

    h = np.diff(mesh)
    mid = 0.5 * (mesh[1:] + mesh[:-1])
    stiff = (
        np.full(mid.size, ei_cable)
        if stiffener is None
        else stiffener.bending_stiffness(mid, cable_od_m, ei_cable)
    )
    g = stiff / h  # conductance of each cell
    n = mesh.size - 1  # unknowns phi_1 .. phi_n
    hbar = np.empty(n)
    hbar[:-1] = 0.5 * (h[:-1] + h[1:])
    hbar[-1] = 0.5 * h[-1]
    if guess is None:
        lam = math.sqrt(ei_cable / tension)
        phi = angle * (1.0 - np.exp(-mesh / lam))
    else:
        phi = np.array(guess, dtype=np.float64)
    phi[0] = 0.0
    for _ in range(100):
        flux = g * np.diff(phi)  # F at midpoints, size n
        inner = np.empty(n)
        inner[:-1] = flux[1:] - flux[:-1]
        inner[-1] = -flux[-1]
        residual = inner / hbar + tension * np.sin(angle - phi[1:])
        diag = np.empty(n)
        diag[:-1] = -(g[:-1] + g[1:])
        diag[-1] = -g[-1]
        diag = diag / hbar - tension * np.cos(angle - phi[1:])
        upper = np.zeros(n)
        lower = np.zeros(n)
        upper[1:] = g[1:] / hbar[:-1]
        lower[:-1] = g[1:] / hbar[1:]
        bands = np.vstack((upper, diag, lower))
        step = solve_banded((1, 1), bands, -residual)
        phi[1:] += step
        if float(np.max(np.abs(step))) <= 1e-12 * max(1.0, abs(angle)):
            break
    else:
        raise CableSolverError("bend-stiffener solution did not converge")
    curvature = np.gradient(phi, mesh, edge_order=2)
    return np.asarray(phi), np.asarray(curvature)


def stiffener_curvature_field(
    stiffener: BendStiffener | None,
    angle_rad: ArrayLike,
    *,
    cable_od_m: float,
    bending_stiffness_Nm2: float,
    tension_N: float,
    x_m: ArrayLike | None = None,
    time_s: ArrayLike | None = None,
    levels: int = 41,
) -> CurvatureField:
    """Cable curvature through a bend stiffener (module notes) [1/m].

    The large-deflection problem is solved at ``levels`` equally spaced
    angles over ``[0, max |theta|]`` (constant tension) with continuation and
    interpolated linearly in the angle; the field is odd in the angle.
    ``stiffener=None`` gives the bare cable.
    """
    angles = np.atleast_1d(np.asarray(angle_rad, dtype=np.float64))
    if angles.ndim != 1 or not np.isfinite(angles).all():
        raise CableInputError("angle history must be a finite 1-D array")
    if np.max(np.abs(angles)) >= math.pi / 2:
        raise CableInputError("angles must stay below 90 degrees")
    ei = _positive(bending_stiffness_Nm2, "bending_stiffness_Nm2")
    ten = _positive(tension_N, "tension_N")
    _positive(cable_od_m, "cable_od_m")
    if isinstance(levels, bool) or levels < 2:
        raise CableInputError("levels must be at least 2")
    lam = math.sqrt(ei / ten)
    length = 0.0 if stiffener is None else stiffener.length_m
    s_end = length + 12.0 * lam
    if stiffener is not None:
        stiffener.bending_stiffness(np.array([0.0]), cable_od_m, ei)
        lam_root = math.sqrt(
            float(stiffener.bending_stiffness(np.array([0.0]), cable_od_m, ei)[0]) / ten
        )
        s_end = length + 12.0 * lam + 2.0 * lam_root
    xs = (
        np.linspace(0.0, length + 6.0 * lam, 401)
        if x_m is None
        else np.asarray(x_m, dtype=np.float64)
    )
    if (
        xs.ndim != 1
        or xs.size < 2
        or np.any(np.diff(xs) <= 0)
        or xs[0] < 0
        or xs[-1] > s_end
    ):
        raise CableInputError("x_m must increase within the solved length")
    step = min(lam, length if length > 0 else lam) / 200.0
    mesh = np.unique(
        np.concatenate([np.arange(0.0, s_end, step), [s_end], np.asarray([length])])
    )
    top = float(np.max(np.abs(angles)))
    grid = np.linspace(0.0, top, levels)
    table = np.zeros((levels, xs.size))
    guess: FloatArray | None = None
    for i, a in enumerate(grid[1:], start=1):
        phi, kap = _stiffener_solution(
            stiffener, float(a), ten, cable_od_m, ei, s_end, mesh, guess
        )
        guess = phi
        table[i] = np.interp(xs, mesh, kap)
    mags = np.abs(angles)
    kappa = np.zeros((xs.size, angles.size))
    for t, a in enumerate(mags):
        if top == 0.0:
            break
        pos = a / top * (levels - 1)
        i = min(int(math.floor(pos)), levels - 2)
        w = pos - i
        kappa[:, t] = np.sign(angles[t]) * ((1 - w) * table[i] + w * table[i + 1])
    return CurvatureField(
        x_m=np.asarray(xs),
        kappa_1_m=kappa,
        time_s=None if time_s is None else np.asarray(time_s, dtype=np.float64),
        kind="bend stiffener" if stiffener is not None else "bare tensioned cable",
        locked_length_m=np.zeros(angles.size),
        n_locked=np.zeros(angles.size, dtype=np.int64),
        tip_overbend=np.zeros(angles.size, dtype=bool),
        lock_curvature_1_m=math.inf,
        decay_length_m=np.full(angles.size, lam),
        provenance=Provenance(
            method=(
                "bend stiffener: large-deflection Euler beam, linear elastic "
                "polyurethane cone bonded to the cable, tension at the far end "
                f"({levels} angle levels, linear interpolation)"
            ),
            citations=("[CaireVaz2007]", "[LuVazCaire2022]"),
            gates=("bare cable, small angle: kappa = theta / lambda exp(-s / lambda)",),
            limitations=(
                "linear elastic polyurethane (no temperature, rate or strain "
                "non-linearity), rigid root, constant tension, planar bending",
            ),
        ),
    )
