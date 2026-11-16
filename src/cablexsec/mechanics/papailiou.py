# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Papailiou (1997) axial slip-front bending model.

The cross-sectional Coulomb population (R1.2) gives the correct EI bounds but a
too-sharp knee, because it models slip onset at the *extreme fibres* (max axial
stress). In a real conductor the relative inter-wire slip is largest at the
*neutral axis*, so slip starts there and the slip front propagates outward
along the cable length as curvature grows — the [Papailiou1997] mechanism. Its
closed form gives the smooth knee.

Per [Papailiou1997], a wire at helix angle ``phi`` (Fig. 1, ``phi`` in
``[-pi/2, pi/2]`` measured from the neutral axis):

* stick (parallel-axis) stiffness, Eq. 20:
  ``EJ_stick(phi) = E A (r sin phi)^2 cos^3(beta)``
* slip (frozen friction) stress, Eq. 14, signed helix angle:
  ``sigma_slip(phi) = sigma_0 (exp(mu sin(beta) phi) - 1)``
* critical curvature, Eq. 15:
  ``kappa_c(phi) = sigma_0 (exp(mu sin(beta) phi) - 1) / (E r sin(phi) cos^2 beta)``

Eq. 11 integrates the capstan equation ``dZ = mu Z sin(beta) dphi`` (Eq. 9)
from the neutral axis with the wire's own force ``Z`` and the signed helix
angle: on the tension side (``phi > 0``) the force can grow to
``Z_T exp(lam phi)``, on the compression side it can fall only to
``Z_T exp(-lam |phi|)``, because a falling wire force also lowers the normal
load it puts on the support (Papailiou: the slip stress increases or decreases
``sigma_T`` "depending on the sign of the wire helix angle"). With a distance
``psi`` from the neutral axis the admissible friction stress of a wire is
therefore the asymmetric interval

    -sigma_0 (1 - exp(-lam psi)) <= sigma <= sigma_0 (exp(lam psi) - 1),

``lam = mu |sin(beta)|`` (:func:`wire_slip_stress_bounds`). The bounds belong
to the wire, not to its side of the neutral axis: a rising stress is bounded
by the upper value and a falling one by the lower value, wherever the wire
lies. The full reversal range is ``2 sigma_0 sinh(lam psi)``, odd in ``lam``,
so the capstan term changes it only at ``O(lam^3)`` relative to the linear
Coulomb range ``2 sigma_0 lam psi``.

``sigma_0`` is the friction-setting stress. The friction-continuity balance
uses normal line load ``q_wire`` per unit *wire-axis* length, not pressure:

    sigma_0 = q_wire / (A_wire * kappa_wire)
    kappa_wire = sin^2(alpha) / r
    q_wire = (P1 + P2) * (2 pi r / n_wires) * cos(alpha)

Here ``P1 + P2`` is the sum of mean pressures on the two cylindrical armour
interfaces. The tributary width and wire-length projection convert it to line
load. A pressure formula for rectangular strips cannot be applied to round
wires by substituting wire diameter for strip thickness. In the small-angle
station limit, the onset curvature is
``mu*q_wire/(E*A_wire*cos^2(alpha)*sin(alpha))``. For a single helical layer
held at prescribed winding stress, the Feret--Bournazel pressure jump gives
``sigma_0`` equal to that stress, providing an independent force-balance check.

Contact-pressure assumption
---------------------------
The slip-front model is evaluated at a prescribed mean clamping pressure.
Its exponential represents the longitudinal friction-continuity mechanism;
it does not solve azimuthally varying normal contact, separation or ovalisation
under bending. An externally calculated change of mean pressure may be passed
to the physical-wire stress function, but bending-induced redistribution and
contact compliance require a coupled section model. The prescribed-pressure
assumption must be stated when comparing with experiment.

Implementation
--------------
Each helix-angle station is an elastic--Coulomb spring-slider with
per-station stiffness ``k_phi = EJ_stick(phi)`` and the asymmetric yield
interval of its wire (:meth:`PapailiouLayer.station_bounds`): the station
force is the moment share ``sigma A r sin(phi) cos(beta)``, so the tension
bound ``sigma_0 expm1(lam |phi|) A r |sin phi| cos(beta)`` limits a positive
force for ``phi > 0`` and a negative one for ``phi < 0``. A station
therefore slips exactly at ``kappa_c(phi)``. The slip front is encoded
entirely in the distribution of stiffness and bounds. Stations are integrated
over ``phi`` with density ``n_w / pi`` so the full-stick limit recovers the
Witz-Tan bound ``1/2 n_w E A r^2 cos^3(beta)`` (gate A8b).

References
----------
[Papailiou1997] Eqs. (9)-(15), (19)-(23).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.friction_law import (
    FrictionLaw,
    _side_bound,
    _slip_magnitude_array,
    active_friction_law,
    friction_law_time_steps,
)
from cablexsec.mechanics.mkappa import HelicalLayerSpec
from cablexsec.mechanics.return_mapping import CoulombElement

if TYPE_CHECKING:
    from cablexsec.mechanics.progressive_friction import ProgressiveFriction

__all__ = [
    "PapailiouLayer",
    "PapailiouResult",
    "PapailiouWireResult",
    "compute_mkappa_papailiou",
    "DIRECTION_HOLD_FRACTION",
    "compute_mkappa_wire_slip_front",
    "direction_threshold_curvature",
    "full_slip_curvature",
    "slip_onset_curvature",
    "wire_slip_stress_bounds",
    "wire_slip_stress_cap",
    "wire_slip_history",
]

FloatArray = NDArray[np.float64]

#: Default fraction ``f`` of the smallest slip-onset curvature below which the
#: bending direction of the slip cap is held and blended (see
#: :func:`wire_slip_history`). At ``|kappa| = 0.1 kappa_on`` the stick stress of
#: a neutral-axis wire is a tenth of its slip resistance, and a perturbation of
#: 1 % of ``kappa_on`` turns the direction by at most 0.1 rad. ``f = 0``
#: reproduces the earlier behaviour, in which every nonzero sample re-oriented
#: the cap.
DIRECTION_HOLD_FRACTION = 0.1


@dataclass(frozen=True)
class PapailiouLayer:
    """One helical layer for the Papailiou slip-front M-κ model.

    Parameters
    ----------
    n_wires : int
    r_layer_m, lay_angle_rad : float
    E_wire_Pa, A_wire_m2, d_wire_m : float
    mu : float
        Friction coefficient for this layer's contact pair.
    contact_pressure_Pa : float
        Sum of mean clamping pressures on the cylindrical interfaces bounding
        the layer. Converted to wire-axis line load using tributary width
        ``2*pi*r/n`` and projection ``cos(alpha)``. This prescribed state may
        include manufacturing, external-pressure and axial-load contributions;
        bending-induced redistribution is not solved by this class.
    n_stations : int
        Number of helix-angle stations integrated over ``[-pi/2, pi/2]``.
    """

    n_wires: int
    r_layer_m: float
    lay_angle_rad: float
    E_wire_Pa: float
    A_wire_m2: float
    d_wire_m: float
    mu: float
    contact_pressure_Pa: float
    n_stations: int = 400

    def __post_init__(self) -> None:
        if (
            not isinstance(self.n_wires, int)
            or isinstance(self.n_wires, bool)
            or self.n_wires < 1
            or not isinstance(self.n_stations, int)
            or isinstance(self.n_stations, bool)
            or self.n_stations < 2
        ):
            raise CableInputError(
                "positive integer wire and station counts are required"
            )
        positive = [self.r_layer_m, self.E_wire_Pa, self.A_wire_m2, self.d_wire_m]
        if not np.isfinite(positive).all() or min(positive) <= 0:
            raise CableInputError(
                "wire geometry and elastic modulus must be finite and positive"
            )
        if (
            not np.isfinite([self.mu, self.contact_pressure_Pa]).all()
            or min(self.mu, self.contact_pressure_Pa) < 0
        ):
            raise CableInputError(
                "friction and clamping pressure must be finite and non-negative"
            )
        if (
            not math.isfinite(self.lay_angle_rad)
            or not 0 < abs(self.lay_angle_rad) < math.pi / 2
        ):
            raise CableInputError(
                "slip-front model requires non-zero lay angle below pi/2"
            )

    @property
    def sigma0_Pa(self) -> float:
        """Stress q_wire/(A kappa_wire), with round-wire tributary width.

        The clamping pressure is averaged over a cylindrical interface. Its
        line load per wire-axis length is P*(2 pi r/n)*cos(alpha). This must
        not be replaced by P*d as for a rectangular strip without accounting
        for wire area and circumferential coverage.
        """
        width = 2.0 * math.pi * self.r_layer_m / self.n_wires
        return (
            self.r_layer_m
            * self.contact_pressure_Pa
            * width
            * math.cos(self.lay_angle_rad)
            / (self.A_wire_m2 * math.sin(self.lay_angle_rad) ** 2)
        )

    def station_params(self) -> tuple[FloatArray, FloatArray, float]:
        """Return ``(k_phi, f_y_phi, weight)`` over the helix-angle stations.

        ``weight`` is the integration weight ``(n_w/pi) * d_phi`` applied to each
        station's moment contribution.
        """
        beta = self.lay_angle_rad
        # midpoint stations on (-pi/2, pi/2), avoiding the phi = 0 singularity.
        edges = np.linspace(-math.pi / 2, math.pi / 2, self.n_stations + 1)
        phi = 0.5 * (edges[:-1] + edges[1:])
        d_phi = math.pi / self.n_stations
        sin_abs = np.abs(np.sin(phi))
        cos_b = math.cos(beta)
        k_phi = (
            self.E_wire_Pa
            * self.A_wire_m2
            * (self.r_layer_m * np.sin(phi)) ** 2
            * (cos_b**3)
        )
        sigma_slip = self.sigma0_Pa * (
            np.expm1(self.mu * abs(math.sin(beta)) * np.abs(phi))
        )
        f_y_phi = sigma_slip * self.A_wire_m2 * self.r_layer_m * sin_abs * cos_b
        weight = (self.n_wires / math.pi) * d_phi
        return np.asarray(k_phi), np.asarray(f_y_phi), weight

    def station_bounds(self) -> tuple[FloatArray, FloatArray, FloatArray, float]:
        """Return ``(k_phi, upper, lower, weight)``: ``-lower <= f <= upper``.

        ``f_y_phi`` of :meth:`station_params` is the tension bound of the
        station's wire in moment units; the compression bound is
        ``f_y_phi exp(-lam |phi|)`` (``sigma_0 (1 - exp(-lam |phi|))``). The
        station force is the moment share ``sigma A r sin(phi) cos(beta)``:
        for ``phi > 0`` a positive force is a tensile wire stress, for
        ``phi < 0`` a compressive one, so the two bounds swap there.
        """
        k_phi, f_y_phi, weight = self.station_params()
        edges = np.linspace(-math.pi / 2, math.pi / 2, self.n_stations + 1)
        phi = 0.5 * (edges[:-1] + edges[1:])
        lam = self.mu * abs(math.sin(self.lay_angle_rad))
        reduced = f_y_phi * np.exp(-lam * np.abs(phi))
        upper = np.where(phi > 0.0, f_y_phi, reduced)
        lower = np.where(phi > 0.0, reduced, f_y_phi)
        return k_phi, np.asarray(upper), np.asarray(lower), weight

    @property
    def EI_stick(self) -> float:
        """Full-stick parallel-axis stiffness ``1/2 n E A r^2 cos^3(beta)``."""
        return (
            0.5
            * self.n_wires
            * self.E_wire_Pa
            * self.A_wire_m2
            * self.r_layer_m**2
            * math.cos(self.lay_angle_rad) ** 3
        )


@dataclass(frozen=True)
class BendingWork:
    """Work per cable-axis length, starting from a stress-free reference.

    With fixed material properties, endpoint work equals stored energy plus
    frictional work plus the backward-Euler integration remainder. A
    WireSlipHistory additionally reports prescribed stiffness-change work;
    that contribution enters the balance on the input side. Neither it nor
    the integration remainder is frictional dissipation. Open-path mechanical
    work must not be reported as loop loss.

    ``viscous_J_per_m`` is the rate-dependent part of the dissipation under a
    rate-dependent :class:`~cablexsec.mechanics.friction_law.FrictionLaw`
    (``None`` otherwise); ``friction_J_per_m`` is then the rate-independent
    part, and the balance subtracts both.
    """

    stored_J_per_m: FloatArray
    friction_J_per_m: FloatArray
    endpoint_work_J_per_m: FloatArray
    integration_remainder_J_per_m: FloatArray
    balance_residual_J_per_m: FloatArray
    viscous_J_per_m: FloatArray | None = None

    @property
    def dissipation_J_per_m(self) -> FloatArray:
        """Total interface dissipation, friction plus viscous [J/m]."""
        if self.viscous_J_per_m is None:
            return self.friction_J_per_m
        return np.asarray(self.friction_J_per_m + self.viscous_J_per_m)


@dataclass(frozen=True)
class PapailiouResult:
    """Moment-curvature loop from the Papailiou slip-front model."""

    kappa: FloatArray
    moment_Nm: FloatArray
    W_diss_J_per_m: float
    EI_max_Nm2: float
    EI_min_Nm2: float
    work: BendingWork


@dataclass(frozen=True)
class PapailiouWireResult:
    """Physical-wire Papailiou loop used by every public stress path."""

    kappa: FloatArray
    moment_Nm: FloatArray
    W_diss_J_per_m: float
    EI_max_Nm2: float
    EI_min_Nm2: float
    slip_fraction: FloatArray
    work: BendingWork


def wire_slip_stress_cap(
    layer: HelicalLayerSpec,
    phi_bend: float,
    *,
    contact_pressure_Pa: float | None = None,
) -> FloatArray:
    r"""Papailiou friction-stress cap at every physical wire [Pa].

    The coordinate transform ``sin(psi)=|cos(phi-phi_bend)|`` maps physical
    wire azimuth onto Papailiou's neutral-axis coordinate. The optional
    three-core shape multiplies the mean ``P1+P2`` contact state and therefore
    changes the local slip front without changing its circumferential mean.
    """
    pressure = (
        layer.contact_pressure_Pa
        if contact_pressure_Pa is None
        else contact_pressure_Pa
    )
    if pressure <= 0.0:
        return np.zeros(layer.n_wires, dtype=np.float64)
    sin_a = abs(math.sin(layer.lay_angle_rad))
    if abs(sin_a) < 1.0e-12:
        return np.zeros(layer.n_wires, dtype=np.float64)
    width = 2.0 * math.pi * layer.r_helix_m / layer.n_wires
    sigma0 = (
        layer.r_helix_m
        * pressure
        * width
        * math.cos(layer.lay_angle_rad)
        / (layer.A_wire_m2 * sin_a**2)
    )
    g = (
        np.ones(layer.n_wires, dtype=np.float64)
        if layer.shape_g is None
        else np.asarray(layer.shape_g, dtype=np.float64)
    )
    cphi = np.cos(layer.phi_wires - phi_bend)
    psi = np.arcsin(np.clip(np.abs(cphi), 0.0, 1.0))
    return np.asarray(sigma0 * g * np.expm1(layer.mu * sin_a * psi))


def wire_slip_stress_bounds(
    layer: HelicalLayerSpec,
    phi_bend: float,
    *,
    contact_pressure_Pa: float | None = None,
) -> tuple[FloatArray, FloatArray]:
    r"""Tension and compression friction-stress bounds of every wire [Pa].

    Returns ``(upper, lower)`` magnitudes with ``-lower <= sigma <= upper``:
    ``upper = sigma0 g expm1(lam psi)`` (:func:`wire_slip_stress_cap`) and
    ``lower = -sigma0 g expm1(-lam psi) = upper exp(-lam psi)``. The capstan
    growth acts on the wire's own force, so a wire whose force falls also
    loses normal load ([Papailiou1997] Eqs. 9-14 with signed ``phi``).
    """
    pressure = (
        layer.contact_pressure_Pa
        if contact_pressure_Pa is None
        else contact_pressure_Pa
    )
    sin_a = abs(math.sin(layer.lay_angle_rad))
    if pressure <= 0.0 or sin_a < 1.0e-12:
        zero = np.zeros(layer.n_wires, dtype=np.float64)
        return zero, zero.copy()
    width = 2.0 * math.pi * layer.r_helix_m / layer.n_wires
    sigma0 = (
        layer.r_helix_m
        * pressure
        * width
        * math.cos(layer.lay_angle_rad)
        / (layer.A_wire_m2 * sin_a**2)
    )
    g = (
        np.ones(layer.n_wires, dtype=np.float64)
        if layer.shape_g is None
        else np.asarray(layer.shape_g, dtype=np.float64)
    )
    cphi = np.cos(layer.phi_wires - phi_bend)
    psi = np.arcsin(np.clip(np.abs(cphi), 0.0, 1.0))
    upper = sigma0 * g * np.expm1(layer.mu * sin_a * psi)
    lower = -sigma0 * g * np.expm1(-layer.mu * sin_a * psi)
    return np.asarray(upper), np.asarray(lower)


def slip_onset_curvature(
    layer: HelicalLayerSpec,
    *,
    contact_pressure_Pa: float | None = None,
    stick_modulus_Pa: float | None = None,
) -> float:
    r"""Smallest curvature at which any wire of the layer starts to slip [1/m].

    A wire at azimuth ``phi`` relative to the bending direction carries the
    stick stress ``k r kappa c`` with ``c = |cos(phi - phi_bend)|`` and slips
    when it reaches the cap ``sigma0 g expm1(mu |sin a| asin c)`` of
    :func:`wire_slip_stress_cap`. The onset curvature
    ``sigma0 g expm1(mu |sin a| asin c) / (k r c)`` increases with ``c``, so its
    infimum is the neutral-axis limit ``c -> 0`` (Papailiou: slip starts at the
    neutral axis):

    .. math::

        \kappa_{on} = \frac{\mu |\sin a| \sigma_0 g_{min}}{k r}
                    = \frac{\mu P (2\pi r/n) \cos a \, g_{min}}
                           {A |\sin a| \, k},

    with ``k = xi E cos^2 a`` the stick stiffness. For ``xi = 1`` this is the
    module-level small-angle onset ``mu q_wire / (E A cos^2 a sin a)``. It is a
    lower bound for a finite wire population on the tension side. On the
    compression side the bound ``sigma0 g (1 - exp(-lam psi))`` gives the onset
    ``kappa_on (1 - exp(-lam psi)) / (lam sin psi)``, whose minimum
    ``kappa_on m(lam)``, ``m = 1 - 3 lam^2 / 8 + O(lam^3)``, lies at
    ``psi ~ 3 lam / 2`` (:func:`_compression_onset_factor`); the value returned
    here stays the neutral-axis unit used to normalise curvature. Zero
    clamping, zero friction or a vanishing shape factor gives zero.
    """
    pressure = (
        layer.contact_pressure_Pa
        if contact_pressure_Pa is None
        else float(contact_pressure_Pa)
    )
    stiffness = layer.k_f if stick_modulus_Pa is None else float(stick_modulus_Pa)
    sin_a = abs(math.sin(layer.lay_angle_rad))
    if pressure <= 0.0 or sin_a < 1.0e-12 or stiffness <= 0.0:
        return 0.0
    g_min = 1.0 if layer.shape_g is None else float(np.min(layer.shape_g))
    return _onset_coefficient(layer, max(g_min, 0.0)) * pressure / stiffness


def _compression_onset_factor(lam: float) -> float:
    """``min_psi (1 - exp(-lam psi)) / (lam sin psi)`` over ``(0, pi/2]``.

    Ratio of the compression-side first-slip curvature to the neutral-axis
    onset; 1 for ``lam -> 0``.
    """
    if lam <= 0.0:
        return 1.0

    def ratio(p: float) -> float:
        return -math.expm1(-lam * p) / (lam * math.sin(p))

    psi = np.linspace(1.0e-6, math.pi / 2, 4001)
    values = -np.expm1(-lam * psi) / (lam * np.sin(psi))
    i = int(np.argmin(values))
    lo, hi = psi[max(i - 1, 0)], psi[min(i + 1, psi.size - 1)]
    for _ in range(80):  # golden-section refinement of the bracketed minimum
        a = hi - 0.6180339887498949 * (hi - lo)
        b = lo + 0.6180339887498949 * (hi - lo)
        if ratio(a) < ratio(b):
            hi = b
        else:
            lo = a
    return float(min(1.0, values[i], ratio(0.5 * (lo + hi))))


def _onset_coefficient(layer: HelicalLayerSpec, g_min: float) -> float:
    """``kappa_on * k / P``; shared by the reference and compiled updates."""
    sin_a = abs(math.sin(layer.lay_angle_rad))
    if sin_a < 1.0e-12:
        return 0.0
    width = 2.0 * math.pi * layer.r_helix_m / layer.n_wires
    return (
        layer.mu
        * width
        * math.cos(layer.lay_angle_rad)
        * g_min
        / (layer.A_wire_m2 * sin_a)
    )


def full_slip_curvature(
    layer: HelicalLayerSpec,
    *,
    phi_bend: float = 0.0,
    contact_pressure_Pa: float | None = None,
    stick_modulus_Pa: float | None = None,
) -> float:
    r"""Curvature at which the last wire of the layer starts to slip [1/m].

    Under planar bending of increasing magnitude from a stress-free state,
    wire ``k`` sticks until its stick stress ``k_f r kappa c_k`` reaches the
    bound on its side from :func:`wire_slip_stress_bounds` (upper on the
    tension side, lower on the compression side), with
    ``c_k = |cos(phi_k - phi_bend)|``. It slips from

    .. math::

        \kappa_k = \frac{\mathrm{cap}_k}{k_f\, r\, c_k},

    and every wire is slipping from ``max_k kappa_k``, which this function
    returns. Wires on the neutral axis (``c_k = 0``) carry no stick stress
    and are left out. For a uniform contact (``g = 1``) the largest value is
    the tension-side extreme-fibre wire (``c = 1``),
    ``sigma0 expm1(mu |sin a| pi/2) / (k_f r)``, when a wire lies on that
    fibre (``phi_bend`` on a wire azimuth). The ratio to
    :func:`slip_onset_curvature` is ``expm1(mu |sin a| pi/2) / (mu |sin a|)``,
    which tends to ``pi/2`` as the friction tends to zero. Zero clamping,
    zero friction or zero stiffness gives zero.
    """
    stiffness = layer.k_f if stick_modulus_Pa is None else float(stick_modulus_Pa)
    if not math.isfinite(phi_bend):
        raise CableInputError("bending direction must be finite")
    if stiffness <= 0.0:
        return 0.0
    upper, lower = wire_slip_stress_bounds(
        layer, phi_bend, contact_pressure_Pa=contact_pressure_Pa
    )
    cosine = np.cos(layer.phi_wires - phi_bend)
    cap = np.where(cosine >= 0.0, upper, lower)
    c = np.abs(cosine)
    loaded = c > 1.0e-12
    if not np.any(loaded) or not np.any(cap > 0.0):
        return 0.0
    curvature = cap[loaded] / (stiffness * layer.r_helix_m * c[loaded])
    return float(np.max(curvature))


def direction_threshold_curvature(
    layer: HelicalLayerSpec,
    hold_fraction: float = DIRECTION_HOLD_FRACTION,
    *,
    contact_pressure_Pa: float | None = None,
    stick_modulus_Pa: float | None = None,
) -> float:
    """Curvature ``f m(lam) kappa_on`` below which the bending direction is held.

    ``m(lam)`` (:func:`_compression_onset_factor`) is the compression-side
    first-slip factor, so that every wire, on either side, is in stick below
    the threshold.
    """
    hold_fraction = float(hold_fraction)
    if not math.isfinite(hold_fraction) or not 0.0 <= hold_fraction <= 1.0:
        raise CableInputError("direction hold fraction must lie in [0, 1]")
    lam = layer.mu * abs(math.sin(layer.lay_angle_rad))
    return (
        hold_fraction
        * _compression_onset_factor(lam)
        * slip_onset_curvature(
            layer,
            contact_pressure_Pa=contact_pressure_Pa,
            stick_modulus_Pa=stick_modulus_Pa,
        )
    )


def _threshold_coefficient(layer: HelicalLayerSpec, hold_fraction: float) -> float:
    """``f m kappa_on k / P``: the per-sample threshold is this times ``P/k``."""
    if not math.isfinite(hold_fraction) or not 0.0 <= hold_fraction <= 1.0:
        raise CableInputError("direction hold fraction must lie in [0, 1]")
    if hold_fraction == 0.0:
        return 0.0
    g_min = 1.0 if layer.shape_g is None else float(np.min(layer.shape_g))
    lam = layer.mu * abs(math.sin(layer.lay_angle_rad))
    return (
        hold_fraction
        * _compression_onset_factor(lam)
        * _onset_coefficient(layer, max(g_min, 0.0))
    )


def compute_mkappa_wire_slip_front(
    layers: list[HelicalLayerSpec],
    kappa_history: FloatArray,
    *,
    phi_bend: float = 0.0,
    EI_min_Nm2: float = 0.0,
    progressive: ProgressiveFriction | None = None,
    contact_pressure_Pa: Sequence[FloatArray | None] | None = None,
    friction_law: FrictionLaw | None = None,
    time_s: FloatArray | None = None,
) -> PapailiouWireResult:
    """Integrate one shared Papailiou return map over physical cable wires.

    Unlike the legacy extreme-fibre Coulomb population, the yield cap follows
    Papailiou's neutral-axis slip-front law. Using physical wire azimuths makes
    bending-plane rotation and a non-axisymmetric ``g(phi)`` contact field
    explicit and lets M-kappa, stress and fatigue share the same kernel.
    ``progressive`` selects the optional progressive (Iwan) law of
    :mod:`cablexsec.mechanics.progressive_friction`; its slip fraction is the
    share of slipping sub-elements. ``None`` keeps the default law.

    ``contact_pressure_Pa`` gives, per layer and in the order of ``layers``,
    a per-sample friction-setting pressure ``P1+P2`` [Pa] for the return map
    (the kernels' ``contact_pressure_Pa``), or ``None`` for the layer's own
    fixed ``contact_pressure_Pa``. The pressure moves only the friction bound;
    stick stiffness and ``EI_min`` stay those of the layer specifications.
    ``None`` for the whole argument keeps the fixed-pressure law.

    ``friction_law`` (:class:`~cablexsec.mechanics.friction_law.FrictionLaw`)
    and ``time_s`` pass to :func:`wire_slip_history`; a non-neutral law is
    refused with ``progressive``. ``work.viscous_J_per_m`` and
    ``W_diss_J_per_m`` then include the rate-dependent dissipation.
    """
    kappa = np.asarray(kappa_history, dtype=np.float64)
    law = active_friction_law(friction_law)
    if law is not None and progressive is not None:
        raise CableInputError(
            "a friction law acts on the single spring-slider per wire; the "
            "progressive law is not implemented with it"
        )
    moment = np.zeros(kappa.size, dtype=np.float64)
    slip_fraction = np.zeros(kappa.size, dtype=np.float64)
    stored, friction, endpoint, remainder = (np.zeros(kappa.size) for _ in range(4))
    viscous = None if law is None or not law.rate_dependent else np.zeros(kappa.size)
    total_wires = sum(layer.n_wires for layer in layers) or 1
    if contact_pressure_Pa is not None and len(contact_pressure_Pa) != len(layers):
        raise CableInputError("one contact-pressure history (or None) per layer")
    pressures = (
        [None] * len(layers) if contact_pressure_Pa is None else contact_pressure_Pa
    )
    for layer, pressure in zip(layers, pressures, strict=True):
        if progressive is None:
            history = wire_slip_history(
                layer,
                kappa * math.cos(phi_bend),
                kappa * math.sin(phi_bend),
                contact_pressure_Pa=pressure,
                initial_bending_angle=phi_bend,
                friction_law=law,
                time_s=time_s,
            )
            slipping: FloatArray | NDArray[np.int64] = history.slipping_count
        else:
            from cablexsec.mechanics.progressive_friction import (  # noqa: PLC0415
                progressive_wire_history,
            )

            progressive_history = progressive_wire_history(
                layer,
                kappa * math.cos(phi_bend),
                kappa * math.sin(phi_bend),
                progressive=progressive,
                contact_pressure_Pa=pressure,
                initial_bending_angle=phi_bend,
            )
            history = progressive_history
            slipping = progressive_history.slipping_share
        moment += history.moment_Nm @ np.array([math.cos(phi_bend), math.sin(phi_bend)])
        stored += history.work.stored_J_per_m
        friction += history.work.friction_J_per_m
        endpoint += history.work.endpoint_work_J_per_m
        remainder += history.work.integration_remainder_J_per_m
        if viscous is not None and history.work.viscous_J_per_m is not None:
            viscous += history.work.viscous_J_per_m
        slip_fraction += slipping / total_wires
    increments = np.diff(kappa, prepend=0.0)
    moment += EI_min_Nm2 * kappa
    stored += 0.5 * EI_min_Nm2 * kappa**2
    endpoint += np.cumsum(EI_min_Nm2 * kappa * increments)
    remainder += np.cumsum(0.5 * EI_min_Nm2 * increments**2)
    if viscous is None:
        work = BendingWork(
            stored,
            friction,
            endpoint,
            remainder,
            endpoint - stored - friction - remainder,
        )
    else:
        work = BendingWork(
            stored,
            friction,
            endpoint,
            remainder,
            endpoint - stored - friction - viscous - remainder,
            viscous,
        )
    total = work.dissipation_J_per_m
    return PapailiouWireResult(
        kappa=kappa,
        moment_Nm=moment,
        W_diss_J_per_m=float(total[-1]) if total.size else 0.0,
        EI_max_Nm2=sum(layer.EI_stick for layer in layers) + EI_min_Nm2,
        EI_min_Nm2=EI_min_Nm2,
        slip_fraction=slip_fraction,
        work=work,
    )


@dataclass(frozen=True)
class WireSlipHistory:
    """One wire population supplies stress, moment and the same friction work.

    Stresses have shape (wires, samples), moments (samples, two). The plastic
    drive is a reduced axial-strain variable, not a crossing slip in metres.
    Material work accounts for a prescribed change of elastic/shear stiffness
    at fixed strain and plastic state; it is not frictional dissipation.
    ``cap_Pa`` is the friction bound on the side of each wire's stress at each
    sample, shape (wires, samples), kept only on request (``keep_cap=True``).
    """

    stress_Pa: FloatArray
    moment_Nm: FloatArray
    slipping_count: NDArray[np.int64]
    final_plastic_drive: FloatArray
    material_work_J_per_m: FloatArray
    work: BendingWork
    cap_Pa: FloatArray | None = None


def wire_slip_history(
    layer: HelicalLayerSpec,
    kappa_x: FloatArray,
    kappa_y: FloatArray,
    *,
    contact_pressure_Pa: FloatArray | None = None,
    stick_modulus_Pa: FloatArray | None = None,
    initial_bending_angle: float = 0.0,
    direction_hold_fraction: float = DIRECTION_HOLD_FRACTION,
    keep_cap: bool = False,
    friction_law: FrictionLaw | None = None,
    time_s: FloatArray | None = None,
) -> WireSlipHistory:
    """Shared small-strain, mean-clamping update for bending and bulk fatigue.

    No spatial contact solve is introduced. The Papailiou bounds of
    :func:`wire_slip_stress_bounds` and a scalar spring-slider describe each
    physical wire: its friction stress stays in ``[-lower, upper]``. Their
    virtual work yields both bending-moment components.

    Bending direction of the slip cap. Near zero curvature the direction
    ``atan2(kappa_y, kappa_x)`` is not a physical quantity; re-orienting the
    cap there clipped the locked friction stress to the cap of an arbitrary
    neutral axis, so that results jumped under perturbations of 1e-12 1/m.
    The held direction ``phi_h`` is therefore updated only when
    ``|kappa| >= kappa_th = f m(lam) kappa_on(P_i, k_i)``
    (``f = direction_hold_fraction``, :func:`slip_onset_curvature`,
    :func:`direction_threshold_curvature`). Below ``kappa_th`` each bound is
    ``(1 - w) bound(phi_h) + w bound(phi)`` with ``w = |kappa|/kappa_th`` and
    ``phi`` the current direction. The bounds are then continuous in the
    curvature, equal to ``bound(phi_h)`` at zero, and, because the smaller
    (compression) bound of a wire is at least ``m kappa_on k r |cos(phi_k -
    phi)|``, ``w bound(phi) >= k r |kappa| |cos(phi_k - phi)| / f``
    guarantees that a sub-threshold path from a stress-free state stays in
    stick for any ``0 <= f <= 1`` (a plain hold would make wires on the held
    neutral axis slip under rotating sub-threshold curvature). Histories
    whose nonzero samples all satisfy ``|kappa| >= kappa_th`` are reproduced
    bitwise, and ``f = 0`` reproduces the earlier update exactly.
    Changing clamping still updates the return map at held curvature.

    ``keep_cap=True`` also returns, for every wire and sample, the bound on
    the side of its current stress (``cap_Pa``: ``upper`` for a nonnegative
    stress, ``lower`` for a negative one); a wire whose stress magnitude is
    below it is in stick. The other results are unchanged.

    Friction work under rotating curvature. ``work.friction_J_per_m`` is the
    dissipation of this population. In planar bending it agrees with the
    distributed helix model
    (:func:`~cablexsec.mechanics.distributed_friction.distributed_friction_history`)
    within 3 % from twice the slip onset, but under rotating curvature it is
    too low. The cap is the friction stress accumulated from the current
    neutral axis, so it turns with the bending direction. On a circular orbit
    a wire passes the extreme fibre, where its cap is largest, while its
    drive is stationary, and it slips mostly near the neutral axis, where its
    cap is close to zero. For a circular orbit of radius ``R`` much larger
    than the slip onset a wire sits at the bound of the sign of its drive
    rate and dissipates ``4 sigma_0 r R (sinh x0 - lam) / (1 + lam^2)`` per
    revolution (times ``A cos a``; ``x0 = lam pi / 2``, ``lam = mu |sin a|``),
    against a wire mean of ``(16 / pi) sigma_0 r R lam cosh x0 / (1 + lam^2)``
    for the two planar cycles of amplitude ``R`` about the two axes: ratio
    ``pi (sinh x0 - lam) / (4 lam cosh x0) = pi (pi/2 - 1) / 4 + O(lam^2)``,
    0.448. In the distributed model every helix node slips through the full
    ``4 R`` per revolution at its own traction bound; its rotating full-slip
    field is the planar one turned to the slip-reversal axes, and the ratio is
    ``pi / 4`` for any ``lam``. The direction-tied work is thus 0.57 of the
    distributed work at large amplitude, and less closer to slip onset (0.16
    at twice the onset and 0.39 at eight times on the 66 kV template). The
    per-wire stress ranges, which fatigue uses, agree with the distributed
    model within 10 % on the same orbits: the cap bounds the stress amplitude
    much as the distributed bound does, but narrows the stress-drive loop of
    each wire. Frictional heat uses the distributed dissipation
    (:func:`~cablexsec.analysis.thermal_coupling.record_friction_work`).

    Friction law. ``friction_law``
    (:class:`~cablexsec.mechanics.friction_law.FrictionLaw`) scales the
    coefficient of every wire by its static, dynamic and rate factor: the
    bounds become ``sigma0 g expm1(m lam psi)`` and ``sigma0 g (1 - exp(-m lam
    psi))`` (blended below the direction threshold as above), each wire keeps
    its own stick timer, and the slip rate is ``(r / |sin a|) dp / dt`` with
    ``dp`` the increment of the plastic drive (module notes of
    :mod:`cablexsec.mechanics.friction_law`). The direction threshold uses
    the smallest factor of the law. ``time_s`` gives the sample times when the
    law needs them. ``work.viscous_J_per_m`` is the rate-dependent
    dissipation of a rate-dependent law. ``None`` or a neutral law keeps the
    rate-independent update, bit for bit.
    """
    kx, ky = np.asarray(kappa_x, dtype=float), np.asarray(kappa_y, dtype=float)
    if (
        kx.ndim != 1
        or ky.shape != kx.shape
        or not kx.size
        or not np.isfinite([kx, ky]).all()
    ):
        raise CableInputError(
            "finite matching nonempty curvature histories are required"
        )
    if not math.isfinite(initial_bending_angle):
        raise CableInputError("initial bending angle must be finite")
    law = active_friction_law(friction_law)
    steps = friction_law_time_steps(law, time_s, kx.size)
    threshold_coefficient = _threshold_coefficient(
        layer if law is None else _law_threshold_layer(layer, law),
        float(direction_hold_fraction),
    )
    count = kx.size
    pressure = (
        np.full(count, layer.contact_pressure_Pa)
        if contact_pressure_Pa is None
        else np.asarray(contact_pressure_Pa, dtype=float)
    )
    stiffness = (
        np.full(count, layer.k_f)
        if stick_modulus_Pa is None
        else np.asarray(stick_modulus_Pa, dtype=float)
    )
    for label, values in (("pressure", pressure), ("stiffness", stiffness)):
        if values.shape != kx.shape or not np.isfinite(values).all():
            raise CableInputError(
                f"{label} must be finite and match curvature histories"
            )
    if np.any(pressure < 0) or np.any(stiffness <= 0):
        raise CableInputError(
            "nonnegative pressure and positive stiffness are required"
        )
    if law is not None:
        return _law_wire_slip_history(
            layer,
            kx,
            ky,
            pressure,
            stiffness,
            law,
            steps,
            initial_bending_angle,
            threshold_coefficient,
            keep_cap,
        )
    element = CoulombElement(k=float(stiffness[0]), shape=(layer.n_wires,))
    directions = np.column_stack((np.cos(layer.phi_wires), np.sin(layer.phi_wires)))
    drive_map = layer.r_helix_m * directions
    weight = layer.A_wire_m2 * math.cos(layer.lay_angle_rad)
    stress = np.empty((layer.n_wires, count))
    caps = np.empty((layer.n_wires, count)) if keep_cap else None
    moment = np.empty((count, 2))
    slipping_count = np.empty(count, dtype=np.int64)
    stored, friction, endpoint, remainder, material = (
        np.zeros(count) for _ in range(5)
    )
    totals = np.zeros(
        4
    )  # friction, endpoint work, integration remainder, material work
    old_drive = np.zeros(layer.n_wires)
    held = current = initial_bending_angle
    zero = np.zeros(layer.n_wires)
    caches: dict[
        str, tuple[tuple[float, float] | None, tuple[FloatArray, FloatArray]]
    ] = {
        "held": (None, (zero, zero)),
        "current": (None, (zero, zero)),
    }

    def oriented_cap(
        slot: str, angle: float, load: float
    ) -> tuple[FloatArray, FloatArray]:
        key, value = caches[slot]
        if key != (angle, load):
            value = wire_slip_stress_bounds(layer, angle, contact_pressure_Pa=load)
            for part in value:
                if (
                    part.shape != (layer.n_wires,)
                    or not np.isfinite(part).all()
                    or np.any(part < 0)
                ):
                    raise CableInputError(
                        "wire caps must be finite, nonnegative and match the population"
                    )
            caches[slot] = ((angle, load), value)
        return value

    # Histories/stiffness are validated above. Check derived inputs when made;
    # finite local state then follows inductively from trapped arithmetic.
    # No caller-owned state is changed or exposed before successful completion.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            for index, (x, y) in enumerate(zip(kx, ky, strict=True)):
                current_stiffness = float(stiffness[index])
                if current_stiffness != element.k:
                    elastic_strain = element.f / element.k
                    totals[3] += (
                        0.5
                        * weight
                        * (current_stiffness - element.k)
                        * (elastic_strain**2).sum()
                    )
                    element.set_stiffness(current_stiffness)
                old_stress, old_plastic = element.f, element.slip_total
                load = float(pressure[index])
                if x != 0 or y != 0:
                    current = math.atan2(y, x)
                threshold = threshold_coefficient * load / current_stiffness
                magnitude = math.sqrt(x * x + y * y)
                if magnitude >= threshold:
                    held = current
                    cap, cap_lo = oriented_cap("held", held, load)
                else:
                    # Ill-defined direction: blend the cap of the last
                    # well-defined axis with that of the current direction.
                    blend = magnitude / threshold
                    cap, cap_lo = oriented_cap("held", held, load)
                    if blend > 0.0:
                        cur_hi, cur_lo = oriented_cap("current", current, load)
                        cap = (1.0 - blend) * cap + blend * cur_hi
                        cap_lo = (1.0 - blend) * cap_lo + blend * cur_lo
                drive = drive_map @ np.array([x, y])
                increment = drive - old_drive
                if not np.isfinite(increment).all():
                    raise CableInputError("derived wire increment must be finite")
                force, slip, slipping = _station_return(
                    element.f, increment, element.k, cap, cap_lo
                )
                # Active bound for ``keep_cap`` (the side of the current force).
                cap = np.where(force < 0.0, cap_lo, cap)
                element.f = force
                element.slip_total = element.slip_total + slip
                stress[:, index] = force
                if caps is not None:
                    caps[:, index] = cap
                moment[index] = weight * (force @ drive_map)
                slipping_count[index] = np.count_nonzero(slipping)
                stored[index] = weight * (force**2).sum() / (2 * current_stiffness)
                totals[0] += weight * (force * (element.slip_total - old_plastic)).sum()
                totals[1] += weight * (force * increment).sum()
                totals[2] += (
                    weight * ((force - old_stress) ** 2).sum() / (2 * current_stiffness)
                )
                friction[index], endpoint[index], remainder[index], material[index] = (
                    totals
                )
                old_drive = drive
            balance = endpoint + material - stored - friction - remainder
        except FloatingPointError as error:
            raise CableInputError(
                "wire history exceeds floating-point range"
            ) from error
    if not all(
        np.isfinite(v).all()
        for v in (
            stress,
            moment,
            element.slip_total,
            stored,
            friction,
            endpoint,
            remainder,
            material,
            balance,
        )
    ):
        raise CableInputError("wire history produced nonfinite results")
    return WireSlipHistory(
        stress,
        moment,
        slipping_count,
        element.slip_total.copy(),
        material,
        BendingWork(
            stored,
            friction,
            endpoint,
            remainder,
            balance,
        ),
        caps,
    )


def _law_threshold_layer(layer: HelicalLayerSpec, law: FrictionLaw) -> HelicalLayerSpec:
    """Layer with the smallest coefficient of the law (direction threshold)."""
    factor = law.minimum_factor
    if factor == 1.0:
        return layer
    return dataclasses.replace(layer, mu=layer.mu * factor)


def _bound_parts(
    layer: HelicalLayerSpec, angle: float, load: float
) -> tuple[FloatArray, FloatArray]:
    """``(sigma0 g, mu |sin a| psi)`` of every wire: the bounds of factor ``m``
    are ``sigma0 g expm1(m x)`` and ``-sigma0 g expm1(-m x)``."""
    sin_a = abs(math.sin(layer.lay_angle_rad))
    if load <= 0.0 or sin_a < 1.0e-12:
        zero = np.zeros(layer.n_wires, dtype=np.float64)
        return zero, zero.copy()
    width = 2.0 * math.pi * layer.r_helix_m / layer.n_wires
    sigma0 = (
        layer.r_helix_m
        * load
        * width
        * math.cos(layer.lay_angle_rad)
        / (layer.A_wire_m2 * sin_a**2)
    )
    g = (
        np.ones(layer.n_wires, dtype=np.float64)
        if layer.shape_g is None
        else np.asarray(layer.shape_g, dtype=np.float64)
    )
    cphi = np.cos(layer.phi_wires - angle)
    psi = np.arcsin(np.clip(np.abs(cphi), 0.0, 1.0))
    scale = np.asarray(sigma0 * g, dtype=np.float64)
    x = np.asarray(layer.mu * sin_a * psi, dtype=np.float64)
    if (
        scale.shape != (layer.n_wires,)
        or not np.isfinite(scale).all()
        or np.any(scale < 0)
        or not np.isfinite(x).all()
    ):
        raise CableInputError(
            "wire caps must be finite, nonnegative and match the population"
        )
    return scale, x


def _law_bounds(
    factor: FloatArray,
    c0: FloatArray,
    x0: FloatArray,
    c1: FloatArray,
    x1: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Rising and falling bounds of every wire at its factor."""
    upper = c0 * np.expm1(factor * x0) + c1 * np.expm1(factor * x1)
    lower = -(c0 * np.expm1(-factor * x0) + c1 * np.expm1(-factor * x1))
    return np.asarray(upper), np.asarray(lower)


def _law_wire_slip_history(
    layer: HelicalLayerSpec,
    kx: FloatArray,
    ky: FloatArray,
    pressure: FloatArray,
    stiffness: FloatArray,
    law: FrictionLaw,
    steps: FloatArray | None,
    initial_bending_angle: float,
    threshold_coefficient: float,
    keep_cap: bool,
) -> WireSlipHistory:
    """:func:`wire_slip_history` under a non-neutral friction law (reference)."""
    count = kx.size
    n_w = layer.n_wires
    element = CoulombElement(k=float(stiffness[0]), shape=(n_w,))
    directions = np.column_stack((np.cos(layer.phi_wires), np.sin(layer.phi_wires)))
    drive_map = layer.r_helix_m * directions
    weight = layer.A_wire_m2 * math.cos(layer.lay_angle_rad)
    # Slip rate per plastic strain rate. A zero lay angle has no capstan
    # friction (zero bounds, :func:`_bound_parts`): its rate does not enter.
    sin_a = abs(math.sin(layer.lay_angle_rad))
    slip_length = layer.r_helix_m / sin_a if sin_a >= 1.0e-12 else 0.0
    stress = np.empty((n_w, count))
    caps = np.empty((n_w, count)) if keep_cap else None
    moment = np.empty((count, 2))
    slipping_count = np.empty(count, dtype=np.int64)
    stored, friction, viscous, endpoint, remainder, material = (
        np.zeros(count) for _ in range(6)
    )
    # friction, viscous, endpoint work, integration remainder, material work
    totals = np.zeros(5)
    old_drive = np.zeros(n_w)
    held = current = initial_bending_angle
    healed = np.ones(n_w, dtype=bool)
    timer = np.zeros(n_w)
    ms, md = law.static_factor, law.dynamic_factor
    rc, v0 = law.rate_coefficient, law.rate_reference_m_per_s
    inv_veta = law.inverse_viscous_velocity
    rate = law.rate_dependent
    cache: dict[str, tuple[tuple[float, float] | None, tuple[FloatArray, FloatArray]]]
    zero = np.zeros(n_w)
    cache = {"held": (None, (zero, zero)), "current": (None, (zero, zero))}

    def parts(slot: str, angle: float, load: float) -> tuple[FloatArray, FloatArray]:
        key, value = cache[slot]
        if key != (angle, load):
            value = _bound_parts(layer, angle, load)
            cache[slot] = ((angle, load), value)
        return value

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            for index, (x, y) in enumerate(zip(kx, ky, strict=True)):
                current_stiffness = float(stiffness[index])
                if current_stiffness != element.k:
                    elastic_strain = element.f / element.k
                    totals[4] += (
                        0.5
                        * weight
                        * (current_stiffness - element.k)
                        * (elastic_strain**2).sum()
                    )
                    element.set_stiffness(current_stiffness)
                old_stress = element.f
                load = float(pressure[index])
                if x != 0 or y != 0:
                    current = math.atan2(y, x)
                threshold = threshold_coefficient * load / current_stiffness
                magnitude = math.sqrt(x * x + y * y)
                blend = 0.0
                if magnitude >= threshold:
                    held = current
                else:
                    blend = magnitude / threshold
                scale, x_held = parts("held", held, load)
                x_cur = x_held
                if blend > 0.0:
                    x_cur = parts("current", current, load)[1]
                c0 = (1.0 - blend) * scale
                c1 = blend * scale
                drive = drive_map @ np.array([x, y])
                increment = drive - old_drive
                if not np.isfinite(increment).all():
                    raise CableInputError("derived wire increment must be finite")
                k = element.k
                trial = element.f + k * increment
                available = np.where(healed, ms, md)
                upper, lower = _law_bounds(available, c0, x_held, c1, x_cur)
                leaves = (trial > upper) | (trial < -lower)
                side = np.where(trial > 0.0, 1.0, -1.0)
                size = np.abs(trial)
                dyn_upper, dyn_lower = _law_bounds(
                    np.full(n_w, md), c0, x_held, c1, x_cur
                )
                dynamic = np.where(side > 0.0, dyn_upper, dyn_lower)
                slips = leaves & (size > dynamic)
                magnitude_slip = np.zeros(n_w)
                if np.any(slips):
                    dt = 1.0 if steps is None else float(steps[index])
                    with np.errstate(over="ignore", invalid="ignore"):
                        magnitude_slip[slips] = _slip_magnitude_array(
                            size[slips],
                            k,
                            md,
                            side[slips],
                            c0[slips],
                            x_held[slips],
                            c1[slips],
                            x_cur[slips],
                            slip_length / dt,
                            rc,
                            v0,
                            inv_veta,
                        )
                    if not np.isfinite(magnitude_slip).all():
                        raise CableInputError(
                            "the friction-law return did not converge (slip rate "
                            "out of range)"
                        )
                # Factor at the end of the step: the available one in stick,
                # the dynamic one (times the rate factor) after leaving it.
                end_factor = np.where(leaves, md, available)
                if rate and np.any(slips):
                    dt = 1.0 if steps is None else float(steps[index])
                    end_factor = np.where(
                        slips,
                        md * law.rate_factor(slip_length * magnitude_slip / dt),
                        end_factor,
                    )
                # A slipping wire sits on its end-of-step bound (exactly, as
                # the rate-independent clip); the slip closes the step.
                # Never above the trial stress (the bound at a converged root).
                # Only the side of the slip is evaluated: at a large rate
                # factor the falling bound saturates while the rising
                # exponential of the same factor overflows.
                end_bound = _side_bound(end_factor, side, c0, x_held, c1, x_cur)
                force = np.where(slips, side * np.minimum(end_bound, size), trial)
                slip = np.where(slips, (trial - force) / k, 0.0)
                element.f = force
                element.slip_total = element.slip_total + slip
                stress[:, index] = force
                if caps is not None:
                    # Wires without slip carry no rate factor: both of their
                    # bounds are in range; a slipping wire's cap is its bound.
                    still_upper, still_lower = _law_bounds(
                        np.where(slips, md, end_factor), c0, x_held, c1, x_cur
                    )
                    caps[:, index] = np.where(
                        slips,
                        end_bound,
                        np.where(force < 0.0, still_lower, still_upper),
                    )
                moment[index] = weight * (force @ drive_map)
                slipping_count[index] = np.count_nonzero(force != trial)
                stored[index] = weight * (force**2).sum() / (2 * current_stiffness)
                dissipated = weight * (force * slip).sum()
                if rate:
                    coulomb = weight * (dynamic * np.abs(slip))[slips].sum()
                    totals[0] += coulomb
                    totals[1] += dissipated - coulomb
                else:
                    totals[0] += dissipated
                totals[2] += weight * (force * increment).sum()
                totals[3] += (
                    weight * ((force - old_stress) ** 2).sum() / (2 * current_stiffness)
                )
                (
                    friction[index],
                    viscous[index],
                    endpoint[index],
                    remainder[index],
                    material[index],
                ) = totals
                # Stick timers: a step that leaves the current interval ends
                # the stick episode; completed stick steps count towards
                # healing.
                dt_stick = 0.0 if steps is None else float(steps[index])
                waiting = ~leaves & ~healed
                timer = np.where(
                    leaves, 0.0, np.where(waiting, timer + dt_stick, timer)
                )
                healed = (healed & ~leaves) | (waiting & (timer >= law.stick_time_s))
                old_drive = drive
            balance = endpoint + material - stored - friction - viscous - remainder
        except FloatingPointError as error:
            raise CableInputError(
                "wire history exceeds floating-point range"
            ) from error
    if not all(
        np.isfinite(v).all()
        for v in (
            stress,
            moment,
            element.slip_total,
            stored,
            friction,
            viscous,
            endpoint,
            remainder,
            material,
            balance,
        )
    ):
        raise CableInputError("wire history produced nonfinite results")
    return WireSlipHistory(
        stress,
        moment,
        slipping_count,
        element.slip_total.copy(),
        material,
        BendingWork(
            stored,
            friction,
            endpoint,
            remainder,
            balance,
            viscous if rate else None,
        ),
        caps,
    )


def _station_return(
    previous: FloatArray,
    increment: float | FloatArray,
    stiffness: float | FloatArray,
    upper: FloatArray,
    lower: FloatArray,
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    """Return map of spring-sliders with the yield interval ``[-lower, upper]``."""
    trial = previous + stiffness * increment
    force = np.minimum(np.maximum(trial, -lower), upper)
    slipping = force != trial
    slip = np.where(slipping, increment - (force - previous) / stiffness, 0.0)
    return np.asarray(force), np.asarray(slip), np.asarray(slipping)


def compute_mkappa_papailiou(
    layers: list[PapailiouLayer],
    kappa_history: FloatArray,
    *,
    EI_min_Nm2: float = 0.0,
) -> PapailiouResult:
    """Integrate the Papailiou slip-front M-κ loop over a curvature history.

    Each helix-angle station is advanced as a spring-slider with the signed
    capstan interval of :meth:`PapailiouLayer.station_bounds`; the smooth
    knee arises from the ``kappa_c(phi)`` distribution (slip starts at the
    neutral axis).
    """
    kappa = np.asarray(kappa_history, dtype=np.float64)
    n_steps = kappa.size
    moment = np.zeros(n_steps, dtype=np.float64)

    k_phis: list[FloatArray] = []
    f_y_phis: list[tuple[FloatArray, FloatArray]] = []
    weights: list[float] = []
    forces: list[FloatArray] = []
    for lyr in layers:
        k_phi, upper, lower, weight = lyr.station_bounds()
        # Odd station counts include the neutral axis. Its zero stiffness,
        # zero force and zero moment carry no energy; omit it from division.
        active = k_phi > 0.0
        k_phi = k_phi[active]
        k_phis.append(k_phi)
        # (upper, lower) magnitudes of the station interval, active stations.
        f_y_phis.append((upper[active], lower[active]))
        weights.append(weight)
        forces.append(np.zeros_like(k_phi))

    kappa_prev = 0.0
    stored, friction, endpoint, remainder = (np.zeros(n_steps) for _ in range(4))
    cumulative_dissipation = cumulative_work = cumulative_remainder = 0.0
    for t in range(n_steps):
        d_kappa = kappa[t] - kappa_prev
        kappa_prev = kappa[t]
        m_t = 0.0
        stored_i = 0.5 * EI_min_Nm2 * kappa[t] ** 2
        cumulative_remainder += 0.5 * EI_min_Nm2 * d_kappa**2
        for i in range(len(layers)):
            f_new, slip, _slipping = _station_return(
                forces[i], d_kappa, k_phis[i], *f_y_phis[i]
            )
            stored_i += float(np.sum(f_new**2 / (2 * k_phis[i])) * weights[i])
            cumulative_dissipation += float(np.sum(f_new * slip) * weights[i])
            cumulative_remainder += float(
                np.sum((f_new - forces[i]) ** 2 / (2 * k_phis[i])) * weights[i]
            )
            forces[i] = f_new
            m_t += float(np.sum(f_new)) * weights[i]
        moment[t] = m_t + EI_min_Nm2 * kappa[t]
        cumulative_work += moment[t] * d_kappa
        stored[t], friction[t] = stored_i, cumulative_dissipation
        endpoint[t], remainder[t] = cumulative_work, cumulative_remainder

    ei_max = sum(lyr.EI_stick for lyr in layers) + EI_min_Nm2
    return PapailiouResult(
        kappa=kappa,
        moment_Nm=moment,
        W_diss_J_per_m=cumulative_dissipation,
        EI_max_Nm2=ei_max,
        EI_min_Nm2=EI_min_Nm2,
        work=BendingWork(
            stored,
            friction,
            endpoint,
            remainder,
            endpoint - stored - friction - remainder,
        ),
    )
