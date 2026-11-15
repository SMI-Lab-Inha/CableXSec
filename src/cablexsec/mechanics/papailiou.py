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
* slip (frozen friction) stress, Eq. 14:
  ``sigma_slip(phi) = sigma_0 (exp(mu sin(beta) |phi|) - 1)``
* critical curvature, Eq. 15:
  ``kappa_c(phi) = sigma_0 (exp(mu sin(beta)|phi|) - 1) / (E r sin|phi| cos^2 beta)``

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
Each helix-angle station uses the same verified Coulomb return-mapping element
(:func:`coulomb_return_map`), unchanged, with per-station stiffness
``k_phi = EJ_stick(phi)`` and yield ``f_y_phi = sigma_slip(phi) A r |sin phi|
cos(beta)``; it therefore slips exactly at ``kappa_c(phi)``. The slip front is
encoded entirely in the ``(k_phi, f_y_phi)`` distribution. Stations are
integrated over ``phi`` with density ``n_w / pi`` so the full-stick limit
recovers the Witz-Tan bound ``1/2 n_w E A r^2 cos^3(beta)`` (gate A8b).

References
----------
[Papailiou1997] Eqs. (14), (15), (19)-(23).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.mkappa import HelicalLayerSpec
from cablexsec.mechanics.return_mapping import (
    CoulombElement,
    _scalar_return_map,
    coulomb_return_map,
)

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
    """

    stored_J_per_m: FloatArray
    friction_J_per_m: FloatArray
    endpoint_work_J_per_m: FloatArray
    integration_remainder_J_per_m: FloatArray
    balance_residual_J_per_m: FloatArray


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
    lower bound for a finite wire population. Zero clamping, zero friction or
    a vanishing shape factor gives zero.
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
    wire ``k`` sticks until its stick stress ``k_f r kappa c_k`` reaches its
    cap from :func:`wire_slip_stress_cap`, with
    ``c_k = |cos(phi_k - phi_bend)|``. It slips from

    .. math::

        \kappa_k = \frac{\mathrm{cap}_k}{k_f\, r\, c_k},

    and every wire is slipping from ``max_k kappa_k``, which this function
    returns. Wires on the neutral axis (``c_k = 0``) carry no stick stress
    and are left out. For a uniform contact (``g = 1``) the largest value is
    the extreme-fibre wire (``c = 1``),
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
    cap = wire_slip_stress_cap(layer, phi_bend, contact_pressure_Pa=contact_pressure_Pa)
    c = np.abs(np.cos(layer.phi_wires - phi_bend))
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
    """Curvature ``f * kappa_on`` below which the bending direction is held."""
    hold_fraction = float(hold_fraction)
    if not math.isfinite(hold_fraction) or not 0.0 <= hold_fraction <= 1.0:
        raise CableInputError("direction hold fraction must lie in [0, 1]")
    return hold_fraction * slip_onset_curvature(
        layer,
        contact_pressure_Pa=contact_pressure_Pa,
        stick_modulus_Pa=stick_modulus_Pa,
    )


def _threshold_coefficient(layer: HelicalLayerSpec, hold_fraction: float) -> float:
    """``f * kappa_on * k / P``: the per-sample threshold is this times ``P/k``."""
    if not math.isfinite(hold_fraction) or not 0.0 <= hold_fraction <= 1.0:
        raise CableInputError("direction hold fraction must lie in [0, 1]")
    if hold_fraction == 0.0:
        return 0.0
    g_min = 1.0 if layer.shape_g is None else float(np.min(layer.shape_g))
    return hold_fraction * _onset_coefficient(layer, max(g_min, 0.0))


def compute_mkappa_wire_slip_front(
    layers: list[HelicalLayerSpec],
    kappa_history: FloatArray,
    *,
    phi_bend: float = 0.0,
    EI_min_Nm2: float = 0.0,
    progressive: ProgressiveFriction | None = None,
) -> PapailiouWireResult:
    """Integrate one shared Papailiou return map over physical cable wires.

    Unlike the legacy extreme-fibre Coulomb population, the yield cap follows
    Papailiou's neutral-axis slip-front law. Using physical wire azimuths makes
    bending-plane rotation and a non-axisymmetric ``g(phi)`` contact field
    explicit and lets M-kappa, stress and fatigue share the same kernel.
    ``progressive`` selects the optional progressive (Iwan) law of
    :mod:`cablexsec.mechanics.progressive_friction`; its slip fraction is the
    share of slipping sub-elements. ``None`` keeps the default law.
    """
    kappa = np.asarray(kappa_history, dtype=np.float64)
    moment = np.zeros(kappa.size, dtype=np.float64)
    slip_fraction = np.zeros(kappa.size, dtype=np.float64)
    stored, friction, endpoint, remainder = (np.zeros(kappa.size) for _ in range(4))
    total_wires = sum(layer.n_wires for layer in layers) or 1
    for layer in layers:
        if progressive is None:
            history = wire_slip_history(
                layer,
                kappa * math.cos(phi_bend),
                kappa * math.sin(phi_bend),
                initial_bending_angle=phi_bend,
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
                initial_bending_angle=phi_bend,
            )
            history = progressive_history
            slipping = progressive_history.slipping_share
        moment += history.moment_Nm @ np.array([math.cos(phi_bend), math.sin(phi_bend)])
        stored += history.work.stored_J_per_m
        friction += history.work.friction_J_per_m
        endpoint += history.work.endpoint_work_J_per_m
        remainder += history.work.integration_remainder_J_per_m
        slip_fraction += slipping / total_wires
    increments = np.diff(kappa, prepend=0.0)
    moment += EI_min_Nm2 * kappa
    stored += 0.5 * EI_min_Nm2 * kappa**2
    endpoint += np.cumsum(EI_min_Nm2 * kappa * increments)
    remainder += np.cumsum(0.5 * EI_min_Nm2 * increments**2)
    return PapailiouWireResult(
        kappa=kappa,
        moment_Nm=moment,
        W_diss_J_per_m=float(friction[-1]) if friction.size else 0.0,
        EI_max_Nm2=sum(layer.EI_stick for layer in layers) + EI_min_Nm2,
        EI_min_Nm2=EI_min_Nm2,
        slip_fraction=slip_fraction,
        work=BendingWork(
            stored,
            friction,
            endpoint,
            remainder,
            endpoint - stored - friction - remainder,
        ),
    )


@dataclass(frozen=True)
class WireSlipHistory:
    """One wire population supplies stress, moment and the same friction work.

    Stresses have shape (wires, samples), moments (samples, two). The plastic
    drive is a reduced axial-strain variable, not a crossing slip in metres.
    Material work accounts for a prescribed change of elastic/shear stiffness
    at fixed strain and plastic state; it is not frictional dissipation.
    ``cap_Pa`` is the friction cap each wire was held to at each sample, shape
    (wires, samples), kept only on request (``keep_cap=True``).
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
) -> WireSlipHistory:
    """Shared small-strain, mean-clamping update for bending and bulk fatigue.

    No spatial contact solve is introduced. The existing Papailiou cap and
    scalar spring-slider describe each physical wire. Their virtual work
    yields both bending-moment components.

    Bending direction of the slip cap. Near zero curvature the direction
    ``atan2(kappa_y, kappa_x)`` is not a physical quantity; re-orienting the
    cap there clipped the locked friction stress to the cap of an arbitrary
    neutral axis, so that results jumped under perturbations of 1e-12 1/m.
    The held direction ``phi_h`` is therefore updated only when
    ``|kappa| >= kappa_th = f * kappa_on(P_i, k_i)``
    (``f = direction_hold_fraction``, :func:`slip_onset_curvature`). Below
    ``kappa_th`` the cap is ``(1 - w) cap(phi_h) + w cap(phi)`` with
    ``w = |kappa|/kappa_th`` and ``phi`` the current direction. The cap is then
    continuous in the curvature, equal to ``cap(phi_h)`` at zero, and
    ``w cap(phi) >= k r |kappa| |cos(phi_k - phi)| / f`` guarantees that a
    sub-threshold path from a stress-free state stays in stick for any
    ``0 <= f <= 1``, exactly as before (a plain hold would make wires on the
    held neutral axis slip under rotating sub-threshold curvature). Histories
    whose nonzero samples all satisfy ``|kappa| >= kappa_th`` are reproduced
    bitwise, and ``f = 0`` reproduces the earlier update exactly.
    Changing clamping still updates the return map at held curvature.

    ``keep_cap=True`` also returns the cap of every wire at every sample
    (``cap_Pa``); a wire whose stress is below its cap is in stick. The
    other results are unchanged.

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
    than the slip onset, with the linearised cap
    ``sigma_0 lam asin|cos(phi - phi_bend)|`` (``lam = mu |sin a|``), a wire
    dissipates ``4 (pi/2 - 1) sigma_0 lam r R`` per revolution (times
    ``A cos a``), against an average of ``16 sigma_0 lam r R / pi`` for the
    two planar cycles of amplitude ``R`` about the two axes: ratio
    ``pi (pi/2 - 1) / 4 = 0.448``. In the distributed model every helix node
    slips through the full ``4 R`` per revolution at its own traction bound,
    and the ratio is ``pi / 4``. The
    direction-tied work is thus 0.57 of the distributed work at large
    amplitude, and less closer to slip onset (0.16 at twice the onset and 0.38
    at eight times on the 66 kV template). The per-wire stress ranges, which
    fatigue uses, agree with the distributed model within 10 % on the same
    orbits: the cap bounds the stress amplitude much as the distributed bound
    does, but narrows the stress-drive loop of each wire. Frictional heat
    uses the distributed dissipation
    (:func:`~cablexsec.analysis.thermal_coupling.record_friction_work`).
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
    threshold_coefficient = _threshold_coefficient(
        layer, float(direction_hold_fraction)
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
    caches: dict[str, tuple[tuple[float, float] | None, FloatArray]] = {
        "held": (None, np.zeros(layer.n_wires)),
        "current": (None, np.zeros(layer.n_wires)),
    }

    def oriented_cap(slot: str, angle: float, load: float) -> FloatArray:
        key, value = caches[slot]
        if key != (angle, load):
            value = wire_slip_stress_cap(layer, angle, contact_pressure_Pa=load)
            if (
                value.shape != (layer.n_wires,)
                or not np.isfinite(value).all()
                or np.any(value < 0)
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
                    cap = oriented_cap("held", held, load)
                else:
                    # Ill-defined direction: blend the cap of the last
                    # well-defined axis with that of the current direction.
                    blend = magnitude / threshold
                    cap = oriented_cap("held", held, load)
                    if blend > 0.0:
                        cap = (1.0 - blend) * cap + blend * oriented_cap(
                            "current", current, load
                        )
                drive = drive_map @ np.array([x, y])
                increment = drive - old_drive
                if not np.isfinite(increment).all():
                    raise CableInputError("derived wire increment must be finite")
                force, slip, slipping = _scalar_return_map(
                    element.f, increment, element.k, cap
                )
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


def compute_mkappa_papailiou(
    layers: list[PapailiouLayer],
    kappa_history: FloatArray,
    *,
    EI_min_Nm2: float = 0.0,
) -> PapailiouResult:
    """Integrate the Papailiou slip-front M-κ loop over a curvature history.

    Each helix-angle station is advanced with the unchanged Coulomb
    return-mapping element; the smooth knee arises from the ``kappa_c(phi)``
    distribution (slip starts at the neutral axis).
    """
    kappa = np.asarray(kappa_history, dtype=np.float64)
    n_steps = kappa.size
    moment = np.zeros(n_steps, dtype=np.float64)

    k_phis: list[FloatArray] = []
    f_y_phis: list[FloatArray] = []
    weights: list[float] = []
    forces: list[FloatArray] = []
    for lyr in layers:
        k_phi, f_y_phi, weight = lyr.station_params()
        # Odd station counts include the neutral axis. Its zero stiffness,
        # zero force and zero moment carry no energy; omit it from division.
        active = k_phi > 0.0
        k_phi, f_y_phi = k_phi[active], f_y_phi[active]
        k_phis.append(k_phi)
        f_y_phis.append(f_y_phi)
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
            f_new, slip, _slp = coulomb_return_map(
                forces[i], d_kappa, k_phis[i], f_y_phis[i]
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
