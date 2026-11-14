# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Multilinear stick-slip strain operator for stranded conductor wires.

The operator follows the Adaptive Hysteresis Model of [KarlsenHusoy2026],
Eqs. 2-10. For an element at fixed azimuth ``theta`` the strain is

.. math::

    \varepsilon = \alpha T + \varepsilon_B ,

where the bending part ``eps_B`` is built order by order. Order zero is the
full-stick line ``eps_0 = beta_0 kappa``. Order ``n >= 1`` has the friction
boundaries ``beta_n kappa +- eps_f,n`` (Eqs. 7-8) and the stick predictor
``eps_n(t-1) + eps_(n-1)(t) - eps_(n-1)(t-1)`` (Eq. 9); the state is the
predictor clamped to the boundaries (Eq. 6). The highest order is ``eps_B``.
Two orders give the bilinear stick-slip loop of one helical layer, three the
trilinear loop of a strand inside a conductor inside a cable.

The friction strain amplitude depends on tension and on the angle ``phi``
between the element and the curvature resultant (Eq. 4)

.. math::

    \varepsilon_{f,n}(\phi, T) =
    \left[\varepsilon_{f,n}^{0} + (\varepsilon_{f,n}(T_0) -
    \varepsilon_{f,n}^{0}) |T|/T_0\right] |\cos\phi| ,

where the pressure term ``eps_f^0`` is an extension of this module (zero
reproduces Eq. 4 exactly) for conductors held by an extruded screen or
sheath without tension. Compression uses ``|T|``, the reversed inequality
set recommended in [KarlsenHusoy2026] Section 2.3.

Curvature projection (Eq. 10) uses the material frame of this package: the
bending strain at azimuth ``theta`` is ``r (kappa_x cos theta + kappa_y sin
theta)``. [KarlsenHusoy2026] write ``kappa_X sin psi - kappa_Y cos psi``, which
is the same projection with ``psi = theta + 90 deg``. ``|cos phi|`` is
``|kappa_theta| / kappa_R``; while the resultant curvature is zero the last
direction is held.

:func:`conductor_ring_parameters` derives bilinear parameters for each
resolved wire ring of a stranded conductor from geometry, friction and
contact load. The stick slope is the plane-section strain ``R cos^2 alpha``;
the slip slope is the local wire bending ``(d/2) cos^2 alpha cos 2 alpha``
([Nasution2014b] Eq. 6); the friction amplitude integrates the available
friction force per unit wire length over a quarter pitch
([Nasution2014b] Eq. 7). The interface line loads are the inward helix loads
of all rings outside the interface (with the ring's own load on its inner
face), the reading of [Nasution2014b] Eq. 9 adopted here, plus an optional
external pressure. In-layer (side) contact is neglected, as in that source.

References
----------
[KarlsenHusoy2026] Eqs. 2-10, Tables 3-4; [Nasution2014b] Eqs. 4-9;
[Papailiou1997].
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cablexsec._numba import HAVE_NUMBA, njit
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.core_lay import bending_factor
from cablexsec.provenance import Provenance

__all__ = [
    "CAP_SHAPES",
    "ConductorRing",
    "PressureTransmission",
    "StrandSlipParameters",
    "StrandSlipResult",
    "conductor_axial_stiffness",
    "conductor_bending_moment",
    "conductor_ring_parameters",
    "interface_line_loads",
    "strand_slip_strain",
]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
PressureTransmission = Literal["all_interfaces", "outer_interface"]
CapShape = Literal["karlsen_husoy", "papailiou"]
#: Accepted azimuthal shapes of the friction amplitude (:func:strand_slip_strain).
CAP_SHAPES: tuple[str, ...] = ("karlsen_husoy", "papailiou")

#: Resultant curvatures below this value [1/m] hold the previous direction.
_DIRECTION_TOLERANCE = 1.0e-14


@dataclass(frozen=True)
class StrandSlipParameters:
    """Strain parameters of one helically laid element ([KarlsenHusoy2026]).

    Attributes
    ----------
    alpha_per_N : float
        Axial strain per unit cable tension [1/N] (Eq. 3; ``alpha`` of the
        source is given per kN).
    beta_m : tuple of float
        ``beta_0 ... beta_n`` [m]: stick (order 0) and slip slopes (Eq. 5).
    eps_f_ref : tuple of float
        ``eps_f,1 ... eps_f,n`` at ``T0_N`` on the curvature resultant (Eq. 4).
    T0_N : float
        Reference tension of ``eps_f_ref`` [N].
    eps_f_zero_tension : tuple of float
        Friction strain amplitudes at zero tension (pressure-held contact).
        Empty means zero, which is Eq. 4 exactly.
    """

    alpha_per_N: float
    beta_m: tuple[float, ...]
    eps_f_ref: tuple[float, ...]
    T0_N: float
    eps_f_zero_tension: tuple[float, ...] = field(default=())

    def __post_init__(self) -> None:
        beta = tuple(float(x) for x in self.beta_m)
        ref = tuple(float(x) for x in self.eps_f_ref)
        zero = tuple(float(x) for x in self.eps_f_zero_tension) or (0.0,) * len(ref)
        if len(beta) < 1 or len(ref) != len(beta) - 1 or len(zero) != len(ref):
            raise CableInputError(
                "strand-slip parameters need beta_0..beta_n and n friction amplitudes"
            )
        values = (self.alpha_per_N, self.T0_N, *beta, *ref, *zero)
        if not all(math.isfinite(x) for x in values):
            raise CableInputError("strand-slip parameters must be finite")
        if min(ref + zero, default=0.0) < 0:
            raise CableInputError("friction strain amplitudes must be nonnegative")
        if self.T0_N < 0 or (self.T0_N == 0 and ref != zero):
            raise CableInputError(
                "a positive reference tension is required for tension-dependent "
                "friction strain"
            )
        object.__setattr__(self, "beta_m", beta)
        object.__setattr__(self, "eps_f_ref", ref)
        object.__setattr__(self, "eps_f_zero_tension", zero)

    @property
    def n_orders(self) -> int:
        """Number of slipping orders (1 = bilinear, 2 = trilinear)."""
        return len(self.eps_f_ref)

    @classmethod
    def from_karlsen_husoy(
        cls,
        alpha_per_kN: float,
        beta_m: Sequence[float],
        eps_f_ref: Sequence[float],
        T0_kN: float,
    ) -> StrandSlipParameters:
        """Parameters in the units of [KarlsenHusoy2026] Tables 3-4."""
        return cls(
            alpha_per_N=alpha_per_kN * 1.0e-3,
            beta_m=tuple(beta_m),
            eps_f_ref=tuple(eps_f_ref),
            T0_N=T0_kN * 1.0e3,
        )

    def friction_amplitude(self, tension_N: FloatArray) -> FloatArray:
        """``eps_f,n`` on the curvature resultant, shape ``(n_orders, n_steps)``."""
        t = np.abs(np.asarray(tension_N, dtype=np.float64))
        zero = np.asarray(self.eps_f_zero_tension)[:, None]
        ref = np.asarray(self.eps_f_ref)[:, None]
        if self.T0_N == 0:
            return np.broadcast_to(zero, (zero.shape[0], t.size)).copy()
        return np.asarray(zero + (ref - zero) * (t[None, :] / self.T0_N))


@dataclass(frozen=True)
class StrandSlipResult:
    """Element strain histories of the multilinear operator.

    Arrays are indexed ``(element, time)``; per-order arrays
    ``(order, element, time)`` with order 1 first. ``mobilisation`` is
    ``(eps_n - beta_n kappa) / eps_f,n`` in ``[-1, 1]``: the fraction of the
    friction capacity in use, signed (zero where that capacity is zero).
    ``slipping`` flags increments in which the order's stick predictor left
    its friction band.
    """

    theta_rad: FloatArray
    curvature_projection: FloatArray
    axial_strain: FloatArray
    bending_strain: FloatArray
    strain: FloatArray
    order_strain: FloatArray
    mobilisation: FloatArray
    slipping: BoolArray
    parameters: StrandSlipParameters
    provenance: Provenance


@njit(cache=True)
def _cascade(
    kappa_e: FloatArray, eps_f: FloatArray, beta: FloatArray
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """Eqs. 6-9 for all elements; ``eps_f`` has shape (order, element, time)."""
    n_ord = beta.size - 1
    n_e, n_t = kappa_e.shape
    strain = np.zeros((n_ord, n_e, n_t))
    ratio = np.zeros((n_ord, n_e, n_t))
    slip = np.zeros((n_ord, n_e, n_t), dtype=np.bool_)
    for e in range(n_e):
        prev_lower = 0.0  # order n-1 strain at t-1 (virgin state at t = -1)
        prev = np.zeros(n_ord)
        for t in range(n_t):
            k = kappa_e[e, t]
            lower_now = beta[0] * k
            lower_before = prev_lower
            prev_lower = lower_now
            for n in range(n_ord):
                predictor = prev[n] + (lower_now - lower_before)
                centre = beta[n + 1] * k
                cap = eps_f[n, e, t]
                low = centre - cap
                high = centre + cap
                value = predictor
                slipped = False
                if predictor > high:
                    value = high
                    slipped = True
                elif predictor < low:
                    value = low
                    slipped = True
                lower_before = prev[n]
                lower_now = value
                prev[n] = value
                strain[n, e, t] = value
                slip[n, e, t] = slipped
                ratio[n, e, t] = (value - centre) / cap if cap > 0 else 0.0
    return strain, ratio, slip


def _direction_cosine(
    kappa_x: FloatArray, kappa_y: FloatArray, kappa_e: FloatArray
) -> FloatArray:
    """``|cos phi|`` per element and step, holding the last resultant direction."""
    resultant = np.hypot(kappa_x, kappa_y)
    valid = resultant > _DIRECTION_TOLERANCE
    cosine = np.ones_like(kappa_e)
    if not np.any(valid):
        return cosine
    raw = np.zeros_like(kappa_e)
    raw[:, valid] = np.abs(kappa_e[:, valid]) / resultant[valid]
    # Hold the last valid direction; before the first one use the first.
    index = np.where(valid, np.arange(valid.size), -1)
    index = np.maximum.accumulate(index)
    first = int(np.argmax(valid))
    index[index < 0] = first
    cosine = np.minimum(raw[:, index], 1.0)
    return np.asarray(cosine)


def strand_slip_strain(
    parameters: StrandSlipParameters,
    kappa_x: ArrayLike,
    kappa_y: ArrayLike,
    tension_N: ArrayLike | None = None,
    *,
    theta_rad: ArrayLike | None = None,
    n_orientations: int = 8,
    cap_shape: CapShape = "karlsen_husoy",
) -> StrandSlipResult:
    """Strain histories of elements at fixed azimuths ([KarlsenHusoy2026]).

    Parameters
    ----------
    parameters : StrandSlipParameters
        Element parameters (orders, friction, axial strain coefficient).
    kappa_x, kappa_y : array_like
        Curvature components [1/m] in the package material frame (bending
        strain ``r (kappa_x cos theta + kappa_y sin theta)``).
    tension_N : array_like or None, optional
        Cable tension history [N]; ``None`` is zero tension.
    theta_rad : array_like or None, optional
        Element azimuths [rad]. Default: ``n_orientations`` equally spaced
        positions from zero (45 degree steps for eight, as in the source).
    n_orientations : int, optional
        Number of default azimuths.
    cap_shape : {"karlsen_husoy", "papailiou"}, optional
        Azimuthal shape of the friction amplitude. `"karlsen_husoy"`
        (default) is Eq. 4, `eps_f |cos phi|`: every azimuth of a ring
        reaches its band at the same curvature, a sharp ring transition.
        `"papailiou"` is the neutral-axis slip front of [Papailiou1997]
        with the same quarter-pitch capacity, `eps_f (2/pi) asin|cos phi|`:
        slip starts at the neutral axis and spreads to the extreme fibre
        between the onset curvature and `pi/2` times it (gap G8). Both are
        functions of `|kappa_theta| / kappa_R` only, so a rotated history
        rotates the response.

    Returns
    -------
    StrandSlipResult
    """
    kx = np.asarray(kappa_x, dtype=np.float64)
    ky = np.asarray(kappa_y, dtype=np.float64)
    if np.iscomplexobj(kappa_x) or np.iscomplexobj(kappa_y):
        raise CableInputError("curvatures must be real")
    if kx.ndim != 1 or kx.shape != ky.shape or kx.size < 1:
        raise CableInputError("kappa_x and kappa_y must be equal 1-D histories")
    if not (np.isfinite(kx).all() and np.isfinite(ky).all()):
        raise CableInputError("curvature histories must be finite")
    if tension_N is None:
        tension = np.zeros_like(kx)
    else:
        tension = np.asarray(tension_N, dtype=np.float64)
        if tension.shape != kx.shape or not np.isfinite(tension).all():
            raise CableInputError("tension must be finite and match the curvatures")
    if theta_rad is None:
        if (
            isinstance(n_orientations, bool)
            or not isinstance(n_orientations, (int, np.integer))
            or n_orientations < 1
        ):
            raise CableInputError("n_orientations must be a positive integer")
        theta = 2.0 * math.pi * np.arange(n_orientations) / n_orientations
    else:
        theta = np.atleast_1d(np.asarray(theta_rad, dtype=np.float64))
        if theta.ndim != 1 or theta.size < 1 or not np.isfinite(theta).all():
            raise CableInputError("theta_rad must be a finite 1-D array")
    kappa_e = (
        np.cos(theta)[:, None] * kx[None, :] + np.sin(theta)[:, None] * ky[None, :]
    )
    if cap_shape not in CAP_SHAPES:
        raise CableInputError(f"cap_shape must be one of {CAP_SHAPES}")
    cosine = _direction_cosine(kx, ky, kappa_e)
    if cap_shape == "papailiou":
        cosine = (2.0 / math.pi) * np.arcsin(np.clip(cosine, 0.0, 1.0))
    amplitude = parameters.friction_amplitude(tension)  # (order, time)
    eps_f = amplitude[:, None, :] * cosine[None, :, :]
    beta = np.asarray(parameters.beta_m, dtype=np.float64)
    if parameters.n_orders == 0:
        order_strain = np.zeros((0, theta.size, kx.size))
        ratio = np.zeros_like(order_strain)
        slip = np.zeros(order_strain.shape, dtype=np.bool_)
        bending = beta[0] * kappa_e
    else:
        order_strain, ratio, slip = _cascade(
            np.ascontiguousarray(kappa_e), np.ascontiguousarray(eps_f), beta
        )
        bending = order_strain[-1]
    axial = np.broadcast_to(parameters.alpha_per_N * tension, kappa_e.shape).copy()
    provenance = Provenance(
        method=(
            "multilinear stick-slip strain operator (Adaptive Hysteresis Model, "
            f"{parameters.n_orders} slipping order(s))"
            + (" compiled" if HAVE_NUMBA else "")
        ),
        citations=("[KarlsenHusoy2026] Eqs. 2-10",),
        limitations=(
            "rate-independent multilinear stick-slip at each fixed azimuth; "
            "friction amplitude follows Eq. 4 (|cos phi| and tension scaling) "
            "with the optional pressure term of this package",
            "no contact compliance (Mindlin) or wear; parameters must come from "
            "a local cross-section analysis or conductor_ring_parameters",
        ),
    )
    return StrandSlipResult(
        theta_rad=theta,
        curvature_projection=kappa_e,
        axial_strain=axial,
        bending_strain=bending,
        strain=axial + bending,
        order_strain=order_strain,
        mobilisation=ratio,
        slipping=slip,
        parameters=parameters,
        provenance=provenance,
    )


# --------------------------------------------------------------------------- #
# Stranded-conductor parameters from geometry                                 #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConductorRing:
    """One wire ring of a stranded conductor (centre wire: ``r_helix_m = 0``).

    ``lay_angle_rad`` is measured from the conductor axis and is nonnegative;
    ``lay_sign`` is +1 for right-hand and -1 for left-hand lay.
    """

    n_wires: int
    d_wire_m: float
    r_helix_m: float
    lay_angle_rad: float
    lay_sign: float = 1.0
    wire_area_m2: float | None = None
    compaction_scf: float = 1.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_wires, bool)
            or not isinstance(self.n_wires, (int, np.integer))
            or self.n_wires < 1
        ):
            raise CableInputError("a ring needs a positive integer wire count")
        values = (self.d_wire_m, self.r_helix_m, self.lay_angle_rad, self.lay_sign)
        if not all(math.isfinite(x) for x in values):
            raise CableInputError("ring geometry must be finite")
        if self.d_wire_m <= 0 or self.r_helix_m < 0:
            raise CableInputError("positive diameter and nonnegative radius required")
        if not 0 <= self.lay_angle_rad < math.pi / 2:
            raise CableInputError("lay angle must lie in [0, pi/2)")
        if self.r_helix_m > 0 and self.lay_angle_rad == 0:
            raise CableInputError("a helical ring needs a positive lay angle")
        if self.lay_sign not in (1.0, -1.0):
            raise CableInputError("lay_sign must be +1 or -1")
        if self.wire_area_m2 is not None and not (
            math.isfinite(self.wire_area_m2) and self.wire_area_m2 > 0
        ):
            raise CableInputError("wire_area_m2 must be finite and positive")
        if not (math.isfinite(self.compaction_scf) and self.compaction_scf >= 1.0):
            raise CableInputError("compaction_scf must be finite and at least 1")

    @property
    def area_m2(self) -> float:
        """Area of one wire normal to its axis [m^2] (compacted area if given)."""
        if self.wire_area_m2 is not None:
            return self.wire_area_m2
        return math.pi * self.d_wire_m**2 / 4.0

    @classmethod
    def from_pitch(
        cls,
        n_wires: int,
        d_wire_m: float,
        r_helix_m: float,
        pitch_m: float | None,
        lay_sign: float = 1.0,
        wire_area_m2: float | None = None,
        compaction_scf: float = 1.0,
    ) -> ConductorRing:
        """Ring from its pitch; a centre wire has no pitch."""
        angle = 0.0 if not pitch_m else math.atan2(2 * math.pi * r_helix_m, pitch_m)
        return cls(
            n_wires, d_wire_m, r_helix_m, angle, lay_sign, wire_area_m2, compaction_scf
        )


def conductor_axial_stiffness(rings: Sequence[ConductorRing], E_Pa: float) -> float:
    """``E A (sum of n cos^3 alpha)`` of the rings [N] ([Nasution2014b] Eq. 4)."""
    return E_Pa * sum(
        r.n_wires * r.area_m2 * math.cos(r.lay_angle_rad) ** 3 for r in rings
    )


def _helix_line_load(ring: ConductorRing, E_Pa: float, strain: float) -> float:
    """Inward line load of a ring per unit conductor length [N/m].

    Each wire carries ``T_w = E A cos^2 alpha eps``; its helix curvature
    ``sin^2 alpha / R`` gives ``T_w sin^2 alpha / R`` per unit wire length,
    that is ``/ cos alpha`` per unit conductor length.
    """
    if ring.r_helix_m == 0:
        return 0.0
    c, s = math.cos(ring.lay_angle_rad), math.sin(ring.lay_angle_rad)
    wire_tension = E_Pa * ring.area_m2 * c * c * strain
    return ring.n_wires * wire_tension * s * s / (ring.r_helix_m * c)


def interface_line_loads(
    rings: Sequence[ConductorRing],
    E_Pa: float,
    strain: float,
    *,
    external_pressure_Pa: float = 0.0,
    r_outer_m: float | None = None,
    pressure_transmission: PressureTransmission = "outer_interface",
) -> tuple[FloatArray, FloatArray]:
    """Line loads per unit conductor length on the outer and inner face of each ring.

    The outer face of ring ``i`` carries the helix loads of every ring outside
    it; the inner face adds ring ``i``'s own load ([Nasution2014b] Eq. 9 as
    read here). External pressure acts on the conductor surface. With
    ``"outer_interface"`` (default) it loads only the outer face of the
    outermost ring, whose side contacts are taken to carry it as a ring; with
    ``"all_interfaces"`` its resultant ``2 pi r_outer p`` passes to every
    interface (no in-layer arching). The default agrees better with the
    loops of Hu et al. (2022) (see ``scripts/check_conductor_strand_slip.py``)
    and with the inward decrease of contact stress reported by Nam et al.
    (2024); it is an assumption, not a solved radial equilibrium.
    """
    if pressure_transmission not in ("all_interfaces", "outer_interface"):
        raise CableInputError(
            "pressure_transmission must be 'all_interfaces' or 'outer_interface'"
        )
    n = len(rings)
    own = np.array([_helix_line_load(r, E_Pa, strain) for r in rings])
    if external_pressure_Pa != 0.0:
        if r_outer_m is None:
            last = rings[-1]
            r_outer_m = last.r_helix_m + last.d_wire_m / 2
        external = 2.0 * math.pi * r_outer_m * external_pressure_Pa
    else:
        external = 0.0
    outer = np.zeros(n)
    inner = np.zeros(n)
    for i in range(n):
        from_outside = float(np.sum(own[i + 1 :]))
        if pressure_transmission == "all_interfaces" or i == n - 1:
            from_outside += external
        outer[i] = from_outside
        inner[i] = from_outside + own[i]
        if pressure_transmission == "outer_interface" and i == n - 1:
            inner[i] = own[i] + float(np.sum(own[i + 1 :]))
    return outer, inner


def _friction_strain(
    ring: ConductorRing,
    outer_load: float,
    inner_load: float,
    mu_outer: float,
    mu_inner: float,
    E_Pa: float,
) -> float:
    """Friction strain amplitude ``pi R tau / (2 sin alpha A E)`` of one ring."""
    if ring.r_helix_m == 0:
        return 0.0
    c, s = math.cos(ring.lay_angle_rad), math.sin(ring.lay_angle_rad)
    tau = (mu_outer * outer_load + mu_inner * inner_load) * c / ring.n_wires
    return math.pi * ring.r_helix_m * tau / (2.0 * s * ring.area_m2 * E_Pa)


def conductor_ring_parameters(
    rings: Sequence[ConductorRing],
    E_Pa: float,
    *,
    mu_wire: float,
    reference_tension_N: float,
    mu_outer: float | None = None,
    strain_per_tension: float | None = None,
    external_pressure_Pa: float = 0.0,
    r_outer_m: float | None = None,
    pressure_transmission: PressureTransmission = "outer_interface",
    fibre: Literal["outer", "centroid"] = "outer",
) -> tuple[StrandSlipParameters, ...]:
    """Bilinear [KarlsenHusoy2026] parameters for every ring of a conductor.

    Parameters
    ----------
    rings : sequence of ConductorRing
        Rings from the centre outwards.
    E_Pa : float
        Wire modulus.
    mu_wire : float
        Wire-wire friction coefficient.
    reference_tension_N : float
        Tension ``T0`` at which ``eps_f_ref`` is evaluated (positive).
    mu_outer : float or None, optional
        Friction coefficient on the outer face of the outermost ring (screen,
        sheath or insulation); defaults to ``mu_wire``.
    strain_per_tension : float or None, optional
        Conductor axial strain per newton of tension. Defaults to the bare
        conductor ``1 / (E A sum n cos^3 alpha)``; inside a cable pass
        ``1 / EA_cable``.
    external_pressure_Pa, r_outer_m, pressure_transmission
        External contact pressure on the conductor surface and its
        transmission (see :func:`interface_line_loads`).
    fibre : {"outer", "centroid"}, optional
        ``"outer"`` gives the outer-fibre strain used for fatigue (local wire
        bending included); ``"centroid"`` the wire-axis strain used for the
        bending moment.

    Returns
    -------
    tuple of StrandSlipParameters
        One entry per ring. The centre wire has no slipping order.
    """
    if not rings:
        raise CableInputError("at least one ring is required")
    values = (E_Pa, mu_wire, reference_tension_N, external_pressure_Pa)
    if not all(math.isfinite(x) for x in values) or E_Pa <= 0 or mu_wire < 0:
        raise CableInputError(
            "finite positive modulus and nonnegative friction required"
        )
    if reference_tension_N <= 0:
        raise CableInputError("reference tension must be positive")
    if fibre not in ("outer", "centroid"):
        raise CableInputError("fibre must be 'outer' or 'centroid'")
    mu_out = mu_wire if mu_outer is None else float(mu_outer)
    if not math.isfinite(mu_out) or mu_out < 0:
        raise CableInputError("mu_outer must be finite and nonnegative")
    compliance = (
        1.0 / conductor_axial_stiffness(rings, E_Pa)
        if strain_per_tension is None
        else float(strain_per_tension)
    )
    if not math.isfinite(compliance) or compliance <= 0:
        raise CableInputError("strain_per_tension must be finite and positive")
    reference_strain = compliance * reference_tension_N
    out_ref, in_ref = interface_line_loads(
        rings,
        E_Pa,
        reference_strain,
        external_pressure_Pa=external_pressure_Pa,
        r_outer_m=r_outer_m,
        pressure_transmission=pressure_transmission,
    )
    out_0, in_0 = interface_line_loads(
        rings,
        E_Pa,
        0.0,
        external_pressure_Pa=external_pressure_Pa,
        r_outer_m=r_outer_m,
        pressure_transmission=pressure_transmission,
    )
    parameters = []
    last = len(rings) - 1
    for i, ring in enumerate(rings):
        c = math.cos(ring.lay_angle_rad)
        local = 0.5 * ring.d_wire_m * c * c * math.cos(2.0 * ring.lay_angle_rad)
        slip_slope = local if fibre == "outer" else 0.0
        alpha = c * c * compliance
        if ring.r_helix_m == 0:
            parameters.append(
                StrandSlipParameters(alpha, (slip_slope,), (), reference_tension_N)
            )
            continue
        stick_slope = ring.r_helix_m * c * c + slip_slope
        face_mu = mu_out if i == last else mu_wire
        eps_ref = _friction_strain(ring, out_ref[i], in_ref[i], face_mu, mu_wire, E_Pa)
        eps_zero = _friction_strain(ring, out_0[i], in_0[i], face_mu, mu_wire, E_Pa)
        parameters.append(
            StrandSlipParameters(
                alpha_per_N=alpha,
                beta_m=(stick_slope, slip_slope),
                eps_f_ref=(eps_ref,),
                T0_N=reference_tension_N,
                eps_f_zero_tension=(eps_zero,),
            )
        )
    return tuple(parameters)


def conductor_bending_moment(
    rings: Sequence[ConductorRing],
    E_Pa: float,
    centroid_parameters: Sequence[StrandSlipParameters],
    kappa_x: ArrayLike,
    kappa_y: ArrayLike,
    tension_N: ArrayLike | None = None,
    *,
    additional_EI_Nm2: float = 0.0,
    cap_shape: CapShape = "karlsen_husoy",
) -> tuple[FloatArray, FloatArray]:
    """Bending moment ``(M_x, M_y)`` [N m] of a conductor from its wire strains.

    Each wire of each ring is an element at its own azimuth. The axial force
    ``E A eps cos alpha`` of the wire-axis strain from ``centroid_parameters``
    (``fibre="centroid"``) contributes ``F r (cos theta, sin theta)``; the wire
    bending stiffness ``sum E I_w`` times the helix bending factor
    (:func:`cablexsec.mechanics.core_lay.bending_factor`) and
    ``additional_EI_Nm2`` (for example a sheath) are added elastically.
    ``cap_shape`` selects the azimuthal friction amplitude of
    :func:`strand_slip_strain` (default Karlsen-Husoy Eq. 4).
    """
    if len(centroid_parameters) != len(rings):
        raise CableInputError("one parameter set per ring is required")
    kx = np.asarray(kappa_x, dtype=np.float64)
    ky = np.asarray(kappa_y, dtype=np.float64)
    m_x = np.zeros_like(kx)
    m_y = np.zeros_like(ky)
    wire_ei = 0.0
    for ring, parameters in zip(rings, centroid_parameters, strict=True):
        wire_ei += (
            E_Pa
            * ring.n_wires
            * math.pi
            * ring.d_wire_m**4
            / 64.0
            * bending_factor(ring.lay_angle_rad)
        )
        if ring.r_helix_m == 0:
            continue
        theta = 2.0 * math.pi * np.arange(ring.n_wires) / ring.n_wires
        result = strand_slip_strain(
            parameters, kx, ky, tension_N, theta_rad=theta, cap_shape=cap_shape
        )
        force = E_Pa * ring.area_m2 * math.cos(ring.lay_angle_rad) * result.strain
        m_x += ring.r_helix_m * np.sum(force * np.cos(theta)[:, None], axis=0)
        m_y += ring.r_helix_m * np.sum(force * np.sin(theta)[:, None], axis=0)
    elastic = wire_ei + additional_EI_Nm2
    return m_x + elastic * kx, m_y + elastic * ky
