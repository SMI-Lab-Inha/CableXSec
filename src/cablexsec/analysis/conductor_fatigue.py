# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Stranded-conductor fatigue with interlayer strand slip and trellis fretting.

Each resolved wire ring of a conductor receives bilinear stick-slip
parameters (:func:`cablexsec.mechanics.strand_slip.conductor_ring_parameters`)
and its outer-fibre strain is computed at fixed azimuths by the multilinear
operator of [KarlsenHusoy2026] (Eqs. 2-10). Two damage terms follow:

* **bulk** — the elastic wire stress ``E eps`` at every azimuth is rainflow
  counted against the registered stress-range curve (Miner sum); the ring
  value is the most damaged azimuth, as in [KarlsenHusoy2026] Section 2.4;
* **trellis fretting** (opt-in, not validated; see below) — at each crossing
  of oppositely laid neighbouring rings the tangential force is
  ``Q = r mu P`` with ``P`` the crossing force at the reference tension and
  ``r`` the contact force ratio from the local shear flow (below). Cycles of
  ``Q`` are counted; cycles in which the crossing does not slip are skipped
  and every slipping cycle receives its own SWT driver from its amplitude and
  the contact pressure
  (:func:`cablexsec.analysis.fretting_swt.swt_slip_cycle_damage`). Rings laid
  in the same direction touch along lines, not at trellis points, and the
  centre-wire contact is a line contact; neither is assessed for fretting.

Trellis contact model (gap 1.3c)
--------------------------------
*Crossing force.* The interface line load of
:func:`~cablexsec.mechanics.strand_slip.interface_line_loads` (helix clench
``n T_w sin^2 alpha / (R cos alpha)`` of every ring outside the interface,
plus the ring's own on its inner face; [Nasution2014b] Eq. 9 as read here) is
divided by the number of crossings per unit conductor length,
``n_i n_j |omega_i - omega_j| / (2 pi)``
(:func:`~cablexsec.mechanics.hertz_contact.layer_crossing_density_per_m`).
This is the trellis load of [RaoofHobbs1988] Eq. 23 with the patch spacing of
Eqs. 24-25. The in-layer (hoop) share ``2 P_MS cos beta`` of their Eq. 19 is
not included: it vanishes for layers with gaps, as reported for the outer
layers of the 300 mm^2 conductor ([Nasution2014b] p. 246, Table 4). The FE
point forces of [Nasution2014b] Table 4 (43.6 and 21.8 N against 177.6 and
41.9 N here) are not used: with the FE wire forces (Fig. 30) they carry about
a quarter to a half of the outer-layer clench, so they are not forces per
physical crossing.

*Tangential force.* In stick the wire force at azimuth ``theta`` follows
plane sections, ``F = E A cos^2 alpha R kappa_theta``; its gradient along the
wire is the shear flow ``E A cos^2 alpha sin alpha kappa_n`` with
``kappa_n = -kappa_x sin theta + kappa_y cos theta``, zero at the extreme
fibre and largest at the neutral axis, where slip starts ([Papailiou1997]
p. 1577). The crossings of both faces share it in proportion to their
friction capacity, so ``r`` is the stop operator of ``kappa_n / kappa_c``
clamped to +-1, with ``kappa_c = tau / (E A cos^2 alpha sin alpha) =
2 eps_f / (pi R cos^2 alpha)`` and ``tau`` the ring's friction force per unit
wire length ([Nasution2014b] Eq. 7). At the neutral axis this gives
[Papailiou1997]'s critical curvature; the gross-slip range ``pi kappa_c`` of
the Karlsen-Husøy operator is reached at the extreme fibre. Spreading of slip
from the neutral axis by accumulated wire displacement ([Papailiou1997]
Eq. 15) is not represented at the contact.

*Pressure saturation.* The elastic Hertz pressure of these crossings
(1.4-2.0 GPa in the Nasution tests) is several times the copper wire yield
stress. The peak pressure is limited to ``c sigma_y`` with ``c = 3`` (the
fully plastic limit of the indentation pressure, [Johnson1985] §6.3 p. 176
and Fig. 6.14; plastic flow flattens the distribution, p. 172); when the
Hertz mean pressure exceeds the limit the contact area grows to
``P / (c sigma_y)``.

*Status.* With these corrections the trellis SWT rule still predicts the
dry Nasution tests one to two orders of magnitude short and cannot see the
bending radius (gate G1 of gap 1.3c, ``doc/benchmarks.md``). It is kept as
the legacy rule ``trellis_model="surface_swt"`` of
:func:`cablexsec.api.compute_conductor_fatigue`.

Version 2 (gap CF2)
-------------------
The bulk count with the SWT mean-stress correction against each conductor's
own wire curve reproduces the dry Nasution tension-bending groups and both
radius trends. The trellis rule of version 2 (:class:`CriticalDistanceSpec`)
evaluates the contact field after plastic indentation plus the bulk stress at
half the critical distance below each crossing ([Taylor2007] point method,
threshold from [TromansSun1996]) on critical planes with the same correction
and curve; the surface evaluation without a material length was the cause of
the version-1 failure.

References
----------
[KarlsenHusoy2026]; [Nasution2014b]; [RaoofHobbs1988]; [Papailiou1997];
[Johnson1985] §3.6, §6.3; [Poon2022]; [Poon2023]; [HobbsRaoof1994];
[ASTM E1049]; [Miner1945]; [SmithWatsonTopper1970];
[Liaw1982] P.K. Liaw, T.R. Leax, R.S. Williams, M.G. Peck, Near-threshold
fatigue crack growth behavior in copper, Metall. Trans. A 13 (1982)
1607-1618;
[TromansSun1996] D. Tromans, R.H. Sun, Intergranular/transgranular fatigue of
copper: influence of environment on crack path and propagation rates, Mater.
Sci. Eng. A 219 (1996) 56-65; [Taylor2007] D. Taylor, The Theory of Critical
Distances, Elsevier, 2007.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cablexsec._numba import njit
from cablexsec.analysis.fatigue import (
    RainflowCounter,
    _miner_damage_rows,
    _use_compiled_counter,
)
from cablexsec.analysis.fretting_regime import range_at_cycles
from cablexsec.analysis.fretting_swt import PartialSlipCycles, swt_slip_cycle_damage
from cablexsec.analysis.sn_assessment import (
    ComponentSCF,
    corrected_damage_rows,
    needs_correction,
    scaled_component,
)
from cablexsec.exceptions import CableInputError
from cablexsec.materials.mean_stress import MeanStressSpec
from cablexsec.materials.sn_curves import SNcurve, SWTLifeConstants
from cablexsec.materials.strain_life import (
    MeanStressCorrection,
    StrainLifeConstants,
    strain_life_damage_map,
)
from cablexsec.mechanics.hertz_contact import (
    HertzContact,
    crossed_cylinder_curvatures,
    layer_crossing_angle_rad,
    layer_crossing_density_per_m,
    reduced_modulus,
    solve_contact,
)
from cablexsec.mechanics.hertz_field import compute_hertz_subsurface_stress
from cablexsec.mechanics.strand_slip import (
    ConductorRing,
    PressureTransmission,
    StrandSlipParameters,
    StrandSlipResult,
    conductor_axial_stiffness,
    conductor_ring_parameters,
    interface_line_loads,
    strand_slip_strain,
)
from cablexsec.provenance import Provenance, UnvalidatedModelWarning

__all__ = [
    "DEFAULT_CONTACT_YIELD_PA",
    "CriticalDistanceSpec",
    "FULLY_PLASTIC_PRESSURE_FACTOR",
    "ConductorFatigueResult",
    "TrellisContact",
    "assess_conductor_fatigue",
    "ring_twist_strain_per_rate",
    "trellis_contact",
    "trellis_force_ratio",
]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

#: Fully plastic upper limit of the mean indentation pressure as a multiple of
#: the yield stress, "~3Y" ([Johnson1985] §6.3, p. 176 and Fig. 6.14).
FULLY_PLASTIC_PRESSURE_FACTOR = 3.0

#: Default wire yield stress for the trellis pressure limit [Pa]: 0.2 % proof
#: stress of the outer wire of a 300 mm^2 compacted copper conductor,
#: about 300 MPa ([Nasution2014b] Fig. 9). Annealed wire yields far lower;
#: pass the measured value of the conductor at hand.
DEFAULT_CONTACT_YIELD_PA = 300.0e6


@dataclass(frozen=True)
class ConductorFatigueResult:
    """Ring-level fatigue damage of a stranded conductor.

    Rings are numbered from the centre wire (index 0) outwards. Per-ring
    values are the maxima over the analysed azimuths. ``contact_p0_Pa`` (the
    pressure used by the fretting driver, after saturation),
    ``contact_p0_elastic_Pa`` (Hertz) and ``crossing_force_N`` are the largest
    values over the ring's assessed trellis faces; ``critical_curvature_1_m``
    is the neutral-axis slip curvature ``kappa_c`` of the crossings. All are
    zero (``nan`` for ``slip_cycle_fraction``) where trellis fretting is not
    evaluated; ``fretting_evaluated`` says whether it was requested.
    ``D_strain_life_per_ring`` (gap P7, with ``strain_life``) is the
    strain-life damage of the outer-fibre strain, maximum over the azimuths
    (``None`` otherwise).
    """

    orientations_deg: FloatArray
    D_bulk_per_ring: FloatArray
    D_bulk_per_orientation: FloatArray
    D_fretting_per_ring: FloatArray
    D_total_per_ring: FloatArray
    stress_range_max_Pa: FloatArray
    contact_p0_Pa: FloatArray
    contact_p0_elastic_Pa: FloatArray
    crossing_force_N: FloatArray
    critical_curvature_1_m: FloatArray
    slip_cycle_fraction: FloatArray
    critical_ring: int
    governing: str
    fretting_evaluated: bool
    parameters: tuple[StrandSlipParameters, ...]
    provenance: Provenance
    D_strain_life_per_ring: FloatArray | None = None
    critical_distance_m: float | None = None
    trellis_azimuth_deg: FloatArray | None = None
    trellis_x_over_a: FloatArray | None = None
    trellis_face: FloatArray | None = None


@dataclass(frozen=True)
class TrellisContact:
    """Normal state of one trellis crossing at the reference tension.

    Attributes
    ----------
    force_N : float
        Crossing force ``P``.
    p0_Pa : float
        Peak pressure for the fretting driver: the Hertz value, limited to
        ``c sigma_y`` when a yield stress is given.
    p0_elastic_Pa : float
        Elastic Hertz peak pressure.
    mean_pressure_Pa : float
        ``P`` over the contact area.
    area_m2 : float
        Contact area: the Hertz ellipse, or ``P / (c sigma_y)`` when the
        Hertz mean pressure exceeds the limit.
    saturated : bool
        True when the peak pressure is limited.
    """

    force_N: float
    p0_Pa: float
    p0_elastic_Pa: float
    mean_pressure_Pa: float
    area_m2: float
    saturated: bool


def trellis_contact(
    ring: ConductorRing,
    partner: ConductorRing,
    line_load_N_per_m: float,
    E_Pa: float,
    nu: float,
    *,
    yield_stress_Pa: float | None = DEFAULT_CONTACT_YIELD_PA,
    pressure_factor: float = FULLY_PLASTIC_PRESSURE_FACTOR,
) -> TrellisContact | None:
    """Crossing force and contact pressure of a ring against a partner ring.

    ``P = q / (n_ring lambda)`` with ``q`` the interface line load per unit
    conductor length and ``lambda`` the crossings per unit length of one wire
    of ``ring`` against ``partner`` ([RaoofHobbs1988] Eqs. 23-25). The Hertz
    ellipse of the crossed cylinders gives ``p0``; with ``yield_stress_Pa``
    the peak is limited to ``pressure_factor * yield_stress_Pa`` and the area
    grows once the Hertz mean pressure exceeds that limit ([Johnson1985]
    §6.3). ``yield_stress_Pa=None`` keeps the elastic Hertz contact.
    Returns ``None`` where no trellis contact exists (centre wire, same lay,
    or no load).
    """
    if yield_stress_Pa is not None and (
        not math.isfinite(yield_stress_Pa) or yield_stress_Pa <= 0
    ):
        raise CableInputError("contact yield stress must be finite and positive")
    if not math.isfinite(pressure_factor) or pressure_factor <= 0:
        raise CableInputError("contact pressure factor must be finite and positive")
    if (
        ring.r_helix_m == 0
        or partner.r_helix_m == 0
        or ring.lay_sign == partner.lay_sign
        or line_load_N_per_m <= 0
    ):
        return None
    density = layer_crossing_density_per_m(
        partner.n_wires,
        ring.r_helix_m,
        ring.lay_angle_rad,
        ring.lay_sign,
        partner.r_helix_m,
        partner.lay_angle_rad,
        partner.lay_sign,
    )
    if density <= 0:
        return None
    force = line_load_N_per_m / ring.n_wires / density
    theta = layer_crossing_angle_rad(ring.lay_angle_rad, partner.lay_angle_rad)
    a_curv, b_curv = crossed_cylinder_curvatures(
        ring.d_wire_m, theta, partner_diameter_m=partner.d_wire_m
    )
    contact = solve_contact(force, a_curv, b_curv, reduced_modulus(E_Pa, nu, E_Pa, nu))
    area = math.pi * contact.a_m * contact.b_m
    p0 = contact.p0_Pa
    saturated = False
    if yield_stress_Pa is not None:
        limit = pressure_factor * yield_stress_Pa
        if p0 > limit:
            p0 = limit
            saturated = True
            area = max(area, force / limit)
    return TrellisContact(
        force_N=force,
        p0_Pa=p0,
        p0_elastic_Pa=contact.p0_Pa,
        mean_pressure_Pa=force / area,
        area_m2=area,
        saturated=saturated,
    )


def _trellis_contact(
    ring: ConductorRing,
    partner: ConductorRing,
    line_load_N_per_m: float,
    E_Pa: float,
    nu: float,
) -> tuple[float, float] | None:
    """Crossing force [N] and elastic Hertz peak pressure [Pa], or ``None``.

    Kept for scripts written against gap 1.3; :func:`trellis_contact` is the
    current form.
    """
    contact = trellis_contact(
        ring, partner, line_load_N_per_m, E_Pa, nu, yield_stress_Pa=None
    )
    return None if contact is None else (contact.force_N, contact.p0_Pa)


@njit(cache=True)
def _stop_operator(drive: FloatArray) -> tuple[FloatArray, BoolArray]:
    """Elastic-perfectly-plastic stop operator with unit limits, row by row.

    The state starts from zero with a zero drive (virgin contact); a sample is
    flagged when its increment reaches a limit.
    """
    n_e, n_t = drive.shape
    ratio = np.zeros((n_e, n_t))
    slip = np.zeros((n_e, n_t), dtype=np.bool_)
    for e in range(n_e):
        state = 0.0
        previous = 0.0
        for t in range(n_t):
            value = state + (drive[e, t] - previous)
            previous = drive[e, t]
            slipped = False
            if value > 1.0:
                value = 1.0
                slipped = True
            elif value < -1.0:
                value = -1.0
                slipped = True
            state = value
            ratio[e, t] = value
            slip[e, t] = slipped
    return ratio, slip


def trellis_force_ratio(
    kappa_x: ArrayLike,
    kappa_y: ArrayLike,
    theta_rad: ArrayLike,
    kappa_c: float,
) -> tuple[FloatArray, BoolArray]:
    """Tangential force ratio ``Q / (mu P)`` of trellis crossings from the shear flow.

    Parameters
    ----------
    kappa_x, kappa_y : array_like, shape (n,)
        Curvature history of the conductor axis [1/m].
    theta_rad : array_like, shape (m,)
        Azimuths of the crossings in the material frame (``theta = 0`` on the
        ``x`` axis).
    kappa_c : float
        Neutral-axis slip curvature of the ring, ``tau / (E A cos^2 alpha
        sin alpha)`` [1/m], positive.

    Returns
    -------
    ratio : numpy.ndarray, shape (m, n)
        Signed ``Q / (mu P)`` in ``[-1, 1]``.
    slipping : numpy.ndarray of bool, shape (m, n)
        True where the crossing slides during the increment ending at that
        sample.

    Notes
    -----
    The drive is the curvature component normal to the azimuth,
    ``kappa_n = -kappa_x sin theta + kappa_y cos theta``, which sets the
    stick shear flow ``E A cos^2 alpha sin alpha kappa_n`` along the wire
    ([Papailiou1997] Eqs. 7-9 differentiated along the helix). A reversal of
    ``kappa_n`` by more than ``2 kappa_c`` slides the crossing (Cattaneo-
    Mindlin unloading: the force changes by twice the limit).
    """
    kx = np.asarray(kappa_x, dtype=np.float64)
    ky = np.asarray(kappa_y, dtype=np.float64)
    theta = np.atleast_1d(np.asarray(theta_rad, dtype=np.float64))
    if kx.ndim != 1 or kx.shape != ky.shape or theta.ndim != 1:
        raise CableInputError("curvatures must be equal 1-D histories")
    if not (
        np.isfinite(kx).all() and np.isfinite(ky).all() and np.isfinite(theta).all()
    ):
        raise CableInputError("curvatures and azimuths must be finite")
    if not math.isfinite(kappa_c) or kappa_c <= 0:
        raise CableInputError("kappa_c must be finite and positive")
    drive = (
        -np.sin(theta)[:, None] * kx[None, :] + np.cos(theta)[:, None] * ky[None, :]
    ) / kappa_c
    ratio, slip = _stop_operator(np.ascontiguousarray(drive))
    return np.asarray(ratio, dtype=np.float64), np.asarray(slip, dtype=np.bool_)


@dataclass(frozen=True)
class CriticalDistanceSpec:
    """Material length and sampling of the trellis critical-distance rule.

    The critical distance is ``L = (1/pi) (dK_th / dsigma_0)^2``
    ([Taylor2007], point method) with ``dsigma_0`` the stress range of the bulk
    curve at ``reference_cycles`` (its own stress ratio), and the stress is
    taken at depth ``L/2`` (point method). ``threshold_MPa_sqrt_m`` defaults
    to 2.55 MPa m^0.5, full-hard ETP copper at R = 0.1 ([Liaw1982] Table III;
    2.35 at R = 0.3): conductor wire is cold-worked (proof stress about
    300 MPa), and R = 0.1 is the ratio of the published conductor-wire
    curves. Annealed copper has 6.35 at R = 0.1 ([Liaw1982]) and 2.1-3.8 at
    R = 0.33 by environment ([TromansSun1996]); these do not apply to
    conductor wire (gap CF3). ``hardness_factor`` times the wire
    yield stress is the mean pressure of a crossing that has indented
    plastically ([Johnson1985] §6.3, about 3).
    """

    threshold_MPa_sqrt_m: float = 2.55
    reference_cycles: float = 1.0e7
    hardness_factor: float = FULLY_PLASTIC_PRESSURE_FACTOR
    n_points: int = 9
    n_planes: int = 18
    source: str = (
        "[Liaw1982] Metall. Trans. A 13, 1607-1618, Table III (full-hard ETP "
        "copper, dK_th 2.55 at R = 0.1); [Taylor2007] point method"
    )

    def __post_init__(self) -> None:
        values = (
            self.threshold_MPa_sqrt_m,
            self.reference_cycles,
            self.hardness_factor,
        )
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise CableInputError(
                "critical-distance inputs must be finite and positive"
            )
        if self.n_points < 1 or self.n_planes < 1:
            raise CableInputError("at least one point and one plane are required")

    def length_m(self, curve: SNcurve) -> float:
        """Critical distance ``L`` [m] for a stress-range curve in MPa."""
        dsigma0 = range_at_cycles(curve, self.reference_cycles) / 1e6
        return (self.threshold_MPa_sqrt_m / dsigma0) ** 2 / math.pi


def _shakedown_contact(
    ring: ConductorRing,
    partner: ConductorRing,
    force_N: float,
    E_Pa: float,
    nu: float,
    yield_stress_Pa: float | None,
    hardness_factor: float,
) -> tuple[float, float, float, bool]:
    """Semi-axes and peak pressure of a crossing after plastic indentation.

    The elastic Hertz ellipse is kept while its mean pressure is below
    ``hardness_factor * yield``; otherwise the crossing indents until the
    mean pressure equals that hardness, the ellipse keeps its aspect ratio,
    and elastic reloading of the conforming profile carries the Hertz
    distribution ``p0 = 1.5 P / (pi a b)`` on the enlarged patch.
    """
    theta = layer_crossing_angle_rad(ring.lay_angle_rad, partner.lay_angle_rad)
    a_curv, b_curv = crossed_cylinder_curvatures(
        ring.d_wire_m, theta, partner_diameter_m=partner.d_wire_m
    )
    hz = solve_contact(force_N, a_curv, b_curv, reduced_modulus(E_Pa, nu, E_Pa, nu))
    a, b = hz.a_m, hz.b_m
    plastic = False
    if yield_stress_Pa is not None:
        area = force_N / (hardness_factor * yield_stress_Pa)
        if area > math.pi * a * b:
            scale = math.sqrt(area / (math.pi * a * b))
            a, b = a * scale, b * scale
            plastic = True
    return a, b, 1.5 * force_N / (math.pi * a * b), plastic


def _trellis_critical_distance(
    contact_axes: tuple[float, float, float],
    depth_m: float,
    nu: float,
    bulk_stress_Pa: FloatArray,
    traction_ratio: FloatArray,
    curve: SNcurve,
    spec: CriticalDistanceSpec,
    compiled: bool,
) -> tuple[float, int, float]:
    """Largest critical-plane damage at depth ``L/2`` below one crossing.

    ``bulk_stress_Pa`` and ``traction_ratio`` (``Q/P``) are ``(azimuth,
    time)`` arrays. The contact frame has ``x`` along the major axis, taken
    as the wire axis (crossing angles are small), and the bulk stress acts
    along ``x``. Every plane normal in the ``x``-``z`` plane is rainflow
    counted with the SWT mean-stress correction against ``curve``. Returns
    the damage, the azimuth index and ``x/a`` of the critical point.
    """
    a, b, p0 = contact_axes
    contact = HertzContact(a, b, p0, math.sqrt(max(0.0, 1.0 - (b / a) ** 2)), 0.0, 1.0)
    xs = np.linspace(-1.3 * a, 1.3 * a, spec.n_points)
    points = np.column_stack([xs, np.zeros_like(xs), np.full_like(xs, depth_m)])
    # Slender, shallow cases need the finest quadrature; a relative tolerance
    # of 1e-4 of the stress scale is ample for a fatigue driver.
    normal = compute_hertz_subsurface_stress(
        contact, points, poisson_ratio=nu, rtol=1e-4, atol_Pa=1e3, max_order=1024
    ).total_stress_Pa
    unit_shear = (
        compute_hertz_subsurface_stress(
            contact,
            points,
            poisson_ratio=nu,
            tangential_traction_ratio=(1.0, 0.0),
            rtol=1e-4,
            atol_Pa=1e3,
            max_order=1024,
        ).total_stress_Pa
        - normal
    )
    psi = np.pi * np.arange(spec.n_planes) / spec.n_planes
    normals = np.stack([np.cos(psi), np.zeros_like(psi), np.sin(psi)], axis=1)
    # Normal stress on each plane: static part, part per unit Q/P, part per unit bulk.
    s_static = np.einsum("pi,kij,pj->kp", normals, normal, normals)
    s_shear = np.einsum("pi,kij,pj->kp", normals, unit_shear, normals)
    s_bulk = normals[:, 0] ** 2
    swt = MeanStressSpec(method="swt")
    best, best_e, best_x = 0.0, 0, 0.0
    for e in range(bulk_stress_Pa.shape[0]):
        rows = (
            s_static[:, :, None]
            + s_shear[:, :, None] * traction_ratio[e][None, None, :]
            + s_bulk[None, :, None] * bulk_stress_Pa[e][None, None, :]
        ).reshape(-1, bulk_stress_Pa.shape[1])
        damage = corrected_damage_rows(rows, curve, mean_stress=swt, compiled=compiled)
        k = int(np.argmax(damage))
        if damage[k] > best:
            best, best_e, best_x = (
                float(damage[k]),
                e,
                float(xs[k // spec.n_planes] / a),
            )
    return best, best_e, best_x


def ring_twist_strain_per_rate(
    ring: ConductorRing, axis_strain_per_twist_m: float
) -> float:
    """Axial wire strain of a ring per unit twist rate [m] ([Knapp1981] Eq. 11).

    ``cos^2(a) e_tau + s r sin(a) cos(a)``: the conductor axis strain
    ``e_tau tau`` resolved along the wire and the helix strain of the ring
    twisting with the conductor (fixed radii, radially rigid support).
    """
    a = ring.lay_angle_rad
    c = math.cos(a)
    return (
        c * c * axis_strain_per_twist_m
        + ring.lay_sign * ring.r_helix_m * math.sin(a) * c
    )


def _with_twist_strain(
    ring: ConductorRing,
    result: StrandSlipResult,
    twist: FloatArray,
    axis_strain_per_twist_m: float,
) -> StrandSlipResult:
    """Add the uniform twist strain of a ring to its axial and total strain."""
    strain = ring_twist_strain_per_rate(ring, axis_strain_per_twist_m) * twist
    return dataclasses.replace(
        result,
        axial_strain=result.axial_strain + strain[None, :],
        strain=result.strain + strain[None, :],
    )


def assess_conductor_fatigue(
    rings: Sequence[ConductorRing],
    E_Pa: float,
    nu: float,
    kappa_x: ArrayLike,
    kappa_y: ArrayLike,
    tension_N: ArrayLike | None = None,
    *,
    sn_bulk: SNcurve,
    swt_const: SWTLifeConstants | None,
    mu_wire: float,
    mu_outer: float | None = None,
    strain_per_tension: float | None = None,
    reference_tension_N: float | None = None,
    external_pressure_Pa: float = 0.0,
    r_outer_m: float | None = None,
    pressure_transmission: PressureTransmission = "outer_interface",
    n_orientations: int = 8,
    partial_slip_cycles: PartialSlipCycles = "skip",
    rainflow_counter: RainflowCounter = "auto",
    contact_yield_Pa: float | None = DEFAULT_CONTACT_YIELD_PA,
    contact_pressure_factor: float = FULLY_PLASTIC_PRESSURE_FACTOR,
    strain_life: StrainLifeConstants | None = None,
    correction: MeanStressCorrection = "swt",
    mean_stress: MeanStressSpec | None = None,
    scf: ComponentSCF | None = None,
    critical_distance: CriticalDistanceSpec | None = None,
    twist_rate_per_m: ArrayLike | None = None,
    axis_strain_per_twist_m: float | None = None,
) -> ConductorFatigueResult:
    """Bulk and trellis-fretting damage of every ring of a stranded conductor.

    Parameters
    ----------
    rings : sequence of ConductorRing
        Wire rings from the centre outwards.
    E_Pa, nu : float
        Wire modulus and Poisson ratio.
    kappa_x, kappa_y : array_like
        Curvature history of the conductor axis [1/m].
    tension_N : array_like or None, optional
        Tension history [N]; with ``strain_per_tension`` it gives the
        conductor axial strain.
    sn_bulk : SNcurve
        Stress-range curve in MPa for the wire bulk stress.
    swt_const : SWTLifeConstants or None
        Strain-life constants for trellis fretting; ``None`` is bulk only.
        Trellis fretting is not validated (gap 1.3c): requesting it emits
        :class:`~cablexsec.provenance.UnvalidatedModelWarning`.
    mu_wire, mu_outer : float
        Wire-wire friction and the friction on the conductor surface.
    strain_per_tension : float or None, optional
        Axial strain per newton; defaults to the bare conductor.
    reference_tension_N : float or None, optional
        Tension for the friction parameters and crossing forces; defaults to
        the largest absolute tension of the history, or 1 N without tension.
    external_pressure_Pa, r_outer_m, pressure_transmission
        Contact pressure on the conductor surface (screen, sheath or
        insulation) and its transmission to inner interfaces.
    n_orientations : int, optional
        Equally spaced azimuths per ring (default 8, as [KarlsenHusoy2026]).
    partial_slip_cycles : {"skip", "cattaneo_mindlin"}, optional
        Treatment of fretting cycles without slip.
    rainflow_counter : {"auto", "compiled", "reference"}, optional
        Counter for the bulk Miner sums.
    contact_yield_Pa : float or None, optional
        Wire yield stress for the trellis pressure limit (default
        :data:`DEFAULT_CONTACT_YIELD_PA`, 300 MPa, [Nasution2014b] Fig. 9);
        ``None`` keeps the elastic Hertz pressure.
    contact_pressure_factor : float, optional
        Limit of the contact pressure over the yield stress (default
        :data:`FULLY_PLASTIC_PRESSURE_FACTOR`, 3, [Johnson1985] §6.3).
    strain_life : StrainLifeConstants or None, optional
        Gap P7: also assess the outer-fibre strain of every azimuth by the
        strain-life route (Masing cyclic curve, Coffin--Manson--Basquin), so
        that ranges above yield are assessed; ``None`` (default) leaves the
        result unchanged.
    correction : {"swt", "morrow", "none"}, optional
        Mean-stress correction of the strain-life route.
    mean_stress : MeanStressSpec or None, optional
        Mean-stress correction of the bulk stress cycles; ``None`` (default)
        counts the unmodified ranges.
    scf : ComponentSCF or None, optional
        Factors on the axial wire stress (axial factors) and on the
        stick-slip bending stress of the operator (friction factors), mean
        and dynamic.
    critical_distance : CriticalDistanceSpec or None, optional
        Trellis rule of version 2 (gap CF2): the stress at depth ``L/2``
        below each crossing, the contact field after plastic indentation
        plus the bulk wire stress, is counted on critical planes with the SWT
        mean-stress correction against ``sn_bulk``. The ring damage is then
        the larger of the bulk and the trellis value (the trellis value
        already contains the bulk stress). Exclusive with ``swt_const``.
    twist_rate_per_m : array_like or None, optional
        Twist rate of the conductor [rad/m], positive in the sense that
        tightens a right-hand lay; it adds the uniform axial strain
        ``cos^2(a) e_tau tau + s r sin(a) cos(a) tau`` to every wire of a
        ring ([Knapp1981] Eqs. 2 and 11, fixed radii), with ``e_tau =
        axis_strain_per_twist_m`` the strain of the conductor axis per unit
        twist rate (required with the twist rate). It does not change the
        ring contact. ``None`` (default) leaves the result unchanged, bit for
        bit.
    axis_strain_per_twist_m : float or None, optional
        Axial strain of the conductor axis per unit twist rate [m]: for a
        laid core of a cable, ``r_c tan(b) - K_et/K_ee`` with the tension held
        (:func:`cablexsec.compute_conductor_fatigue`).

    Returns
    -------
    ConductorFatigueResult
    """
    if swt_const is not None and critical_distance is not None:
        raise CableInputError("choose the surface SWT or the critical-distance rule")
    if sn_bulk.stress_unit != "MPa":
        raise CableInputError("conductor bulk fatigue requires an S-N curve in MPa")
    kx = np.asarray(kappa_x, dtype=np.float64)
    ky = np.asarray(kappa_y, dtype=np.float64)
    if kx.ndim != 1 or kx.shape != ky.shape or kx.size < 2:
        raise CableInputError("curvature histories must be equal 1-D arrays")
    tension = None if tension_N is None else np.asarray(tension_N, dtype=np.float64)
    if tension is not None and tension.shape != kx.shape:
        raise CableInputError("tension must match the curvature histories")
    if reference_tension_N is None:
        peak = 0.0 if tension is None else float(np.max(np.abs(tension)))
        reference_tension_N = peak if peak > 0 else 1.0
    twist = None
    if twist_rate_per_m is not None:
        twist = np.asarray(twist_rate_per_m, dtype=np.float64)
        if twist.shape != kx.shape or not np.isfinite(twist).all():
            raise CableInputError(
                "twist_rate_per_m must be finite and match the curvature histories"
            )
        if axis_strain_per_twist_m is None or not math.isfinite(
            axis_strain_per_twist_m
        ):
            raise CableInputError(
                "twist_rate_per_m requires a finite axis_strain_per_twist_m"
            )
    elif axis_strain_per_twist_m is not None:
        raise CableInputError(
            "axis_strain_per_twist_m is used only with twist_rate_per_m"
        )
    if swt_const is not None:
        warnings.warn(
            "conductor trellis fretting (per-slip-cycle SWT) is not validated: "
            "on the published Nasution conductor tests it predicts the dry "
            "lives one to two orders of magnitude short and does not resolve "
            "the bending radius (gap 1.3c, doc/benchmarks.md). Use the bulk "
            "S-N damage for assessment.",
            UnvalidatedModelWarning,
            stacklevel=2,
        )
    compiled = _use_compiled_counter(rainflow_counter)
    corrected = needs_correction(sn_bulk, mean_stress, scf)
    scf_parts = scf if scf is not None else ComponentSCF()
    parameters = conductor_ring_parameters(
        rings,
        E_Pa,
        mu_wire=mu_wire,
        reference_tension_N=reference_tension_N,
        mu_outer=mu_outer,
        strain_per_tension=strain_per_tension,
        external_pressure_Pa=external_pressure_Pa,
        r_outer_m=r_outer_m,
        pressure_transmission=pressure_transmission,
        fibre="outer",
    )
    compliance = (
        1.0 / conductor_axial_stiffness(rings, E_Pa)
        if strain_per_tension is None
        else strain_per_tension
    )
    outer_load, inner_load = interface_line_loads(
        rings,
        E_Pa,
        compliance * reference_tension_N,
        external_pressure_Pa=external_pressure_Pa,
        r_outer_m=r_outer_m,
        pressure_transmission=pressure_transmission,
    )
    n_rings = len(rings)
    d_life = np.zeros(n_rings)
    d_bulk_orient = np.zeros((n_rings, n_orientations))
    d_fret = np.zeros(n_rings)
    stress_range = np.zeros(n_rings)
    p0_max = np.zeros(n_rings)
    p0_elastic_max = np.zeros(n_rings)
    force_max = np.zeros(n_rings)
    kappa_c = np.zeros(n_rings)
    slip_fraction = np.full(n_rings, np.nan)
    theta = 2.0 * math.pi * np.arange(n_orientations) / n_orientations
    cd_length = (
        critical_distance.length_m(sn_bulk) if critical_distance is not None else 0.0
    )
    cd_azimuth = np.full(n_rings, np.nan)
    cd_x = np.full(n_rings, np.nan)
    cd_face = np.full(n_rings, np.nan)
    for i, (ring, params) in enumerate(zip(rings, parameters, strict=True)):
        result = strand_slip_strain(params, kx, ky, tension, theta_rad=theta)
        if twist is not None:
            assert axis_strain_per_twist_m is not None  # checked above
            result = _with_twist_strain(ring, result, twist, axis_strain_per_twist_m)
        # Compaction indentations raise the stress of the whole wire section
        # (axial and bending parts alike) by the ring's factor (gap CF3).
        compaction = ring.compaction_scf
        stress = E_Pa * compaction * result.strain
        stress_range[i] = float(np.max(stress.max(axis=1) - stress.min(axis=1)))
        if corrected:
            corrected_stress = stress
            if not scf_parts.is_identity:
                axial = np.broadcast_to(
                    E_Pa * compaction * result.axial_strain, stress.shape
                )
                bending = np.broadcast_to(
                    E_Pa * compaction * result.bending_strain, stress.shape
                )
                corrected_stress = scaled_component(
                    axial, scf_parts.axial_mean, scf_parts.axial_dynamic
                ) + scaled_component(
                    bending, scf_parts.friction_mean, scf_parts.friction_dynamic
                )
            d_bulk_orient[i] = corrected_damage_rows(
                corrected_stress,
                sn_bulk,
                mean_stress=mean_stress,
                thickness_m=ring.d_wire_m,
                compiled=compiled,
            )
        else:
            d_bulk_orient[i] = _miner_damage_rows(stress, sn_bulk, 0.0, 1.0, compiled)
        if strain_life is not None:
            d_life[i] = float(
                np.max(
                    strain_life_damage_map(
                        result.strain, strain_life, correction=correction
                    ).damage
                )
            )
        if critical_distance is not None and params.n_orders > 0:
            c2 = math.cos(ring.lay_angle_rad) ** 2
            kappa_c[i] = 2.0 * params.eps_f_ref[0] / (math.pi * ring.r_helix_m * c2)
            if kappa_c[i] > 0:
                ratio, _slipping = trellis_force_ratio(kx, ky, theta, float(kappa_c[i]))
                for face, partner, load in (
                    (1.0, rings[i + 1] if i + 1 < n_rings else None, outer_load[i]),
                    (-1.0, rings[i - 1] if i > 0 else None, inner_load[i]),
                ):
                    if partner is None:
                        continue
                    contact = trellis_contact(
                        ring, partner, float(load), E_Pa, nu, yield_stress_Pa=None
                    )
                    if contact is None:
                        continue
                    a, b, p0, _plastic = _shakedown_contact(
                        ring,
                        partner,
                        contact.force_N,
                        E_Pa,
                        nu,
                        contact_yield_Pa,
                        critical_distance.hardness_factor,
                    )
                    force_max[i] = max(force_max[i], contact.force_N)
                    p0_max[i] = max(p0_max[i], p0)
                    p0_elastic_max[i] = max(p0_elastic_max[i], contact.p0_elastic_Pa)
                    damage, e_crit, x_crit = _trellis_critical_distance(
                        (a, b, p0),
                        0.5 * cd_length,
                        nu,
                        stress,
                        mu_wire * ratio,
                        sn_bulk,
                        critical_distance,
                        compiled,
                    )
                    if damage > d_fret[i]:
                        d_fret[i] = damage
                        cd_azimuth[i] = float(np.degrees(theta[e_crit]))
                        cd_x[i] = x_crit
                        cd_face[i] = face
            continue
        if swt_const is None or params.n_orders == 0:
            continue
        faces = []
        if i + 1 < n_rings:
            faces.append((rings[i + 1], outer_load[i], mu_wire))
        if i > 0:
            faces.append((rings[i - 1], inner_load[i], mu_wire))
        c2 = math.cos(ring.lay_angle_rad) ** 2
        # Neutral-axis slip curvature tau / (E A cos^2 a sin a)
        # = 2 eps_f / (pi R cos^2 a).
        kappa_c[i] = 2.0 * params.eps_f_ref[0] / (math.pi * ring.r_helix_m * c2)
        if kappa_c[i] <= 0:
            continue
        ratio, slipping = trellis_force_ratio(kx, ky, theta, float(kappa_c[i]))
        counted = evaluated = 0.0
        for partner, load, mu in faces:
            contact = trellis_contact(
                ring,
                partner,
                float(load),
                E_Pa,
                nu,
                yield_stress_Pa=contact_yield_Pa,
                pressure_factor=contact_pressure_factor,
            )
            if contact is None:
                continue
            force, p0 = contact.force_N, contact.p0_Pa
            p0_max[i] = max(p0_max[i], p0)
            p0_elastic_max[i] = max(p0_elastic_max[i], contact.p0_elastic_Pa)
            force_max[i] = max(force_max[i], force)
            for e in range(n_orientations):
                cycles = swt_slip_cycle_damage(
                    ratio[e] * mu * force,
                    slipping[e],
                    force,
                    p0,
                    mu,
                    E_Pa,
                    nu,
                    swt_const,
                    partial_slip=partial_slip_cycles,
                )
                d_fret[i] = max(d_fret[i], cycles.damage)
                counted += cycles.counted_cycles
                evaluated += cycles.evaluated_cycles
        if counted > 0:
            slip_fraction[i] = evaluated / counted
    d_bulk = d_bulk_orient.max(axis=1)
    if critical_distance is not None:
        d_total = np.maximum(d_bulk, d_fret)
    else:
        d_total = d_bulk + d_fret
    critical = int(np.argmax(d_total))
    governing = "fretting" if d_fret[critical] > d_bulk[critical] else "bulk"
    fretting = swt_const is not None
    if critical_distance is not None:
        governing = "trellis" if d_fret[critical] > d_bulk[critical] else "bulk"
    if fretting:
        method = (
            "stranded-conductor fatigue: bilinear stick-slip strain per ring "
            "(Adaptive Hysteresis Model, [KarlsenHusoy2026] Eqs. 2-10) with "
            "friction amplitudes from [Nasution2014b] Eqs. 7-9 as read here; "
            "rainflow + Miner on elastic outer-fibre stress; trellis fretting "
            "(NOT VALIDATED) by per-cycle SWT on slipping cycles, tangential "
            "force from the stick shear flow, pressure limited to "
            f"{contact_pressure_factor:g} sigma_y"
        )
    else:
        method = (
            "stranded-conductor fatigue: bilinear stick-slip strain per ring "
            "(Adaptive Hysteresis Model, [KarlsenHusoy2026] Eqs. 2-10) with "
            "friction amplitudes from [Nasution2014b] Eqs. 7-9 as read here; "
            "rainflow + Miner on elastic outer-fibre stress (bulk S-N only)"
        )
    citations = [
        "[KarlsenHusoy2026] Eqs. 2-10",
        "[Nasution2014b] Eqs. 4-9",
        "[ASTM E1049]",
        "[Miner1945]",
    ]
    limitations = [
        "the power core is taken in full slip inside the cable; the "
        "core-level (order-zero) stick of [KarlsenHusoy2026] is not included",
        "in-layer (side) contact is neglected and the interface loads follow "
        "the helix-tension reading of [Nasution2014b] Eq. 9; external pressure "
        f"transmission '{pressure_transmission}' is an assumption",
        "elastic wire stress; copper plasticity, compaction stress "
        "concentration and wear are not modelled",
    ]
    if corrected:
        limitations.append(
            f"user S-N options: curve {sn_bulk.name} ({sn_bulk.reference}); "
            + (
                mean_stress.describe()
                if mean_stress is not None
                else "no mean-stress correction"
            )
            + "; "
            + scf_parts.describe()
            + " (friction factors on the stick-slip bending stress)"
        )
    if fretting:
        citations[2:2] = [
            "[RaoofHobbs1988] Eqs. 19, 23-25",
            "[Papailiou1997] Eqs. 7-9, 15",
            "[Johnson1985] §6.3",
            "[Poon2022] Eq. 30",
            "[Poon2023] Eqs. 7-9",
            "[HobbsRaoof1994]",
        ]
        limitations += [
            "trellis fretting is NOT VALIDATED: on the Nasution et al. (2014, "
            "2013) conductor tests it predicts the dry lives one to two orders "
            "of magnitude short and does not resolve the bending radius "
            "(gap 1.3c gate G1, doc/benchmarks.md)",
            "crossing force at the reference tension without the in-layer hoop "
            "share of [RaoofHobbs1988] Eq. 19; tangential force from the local "
            "stick shear flow at constant friction capacity (no spreading of "
            "slip by accumulated wire displacement, [Papailiou1997] Eq. 15)",
            "contact pressure limited to "
            + (
                "the elastic Hertz value (no yield limit)"
                if contact_yield_Pa is None
                else f"{contact_pressure_factor:g} x {contact_yield_Pa / 1e6:g} MPa"
            )
            + "; the SWT driver is a closed-form line-contact surface stress, "
            "not a critical-plane search",
            "non-slipping fretting cycles are "
            + (
                "skipped"
                if partial_slip_cycles == "skip"
                else "given partial-slip drivers"
            ),
        ]
    if critical_distance is not None:
        method = (
            "stranded-conductor fatigue: bilinear stick-slip strain per ring "
            "(Adaptive Hysteresis Model, [KarlsenHusoy2026] Eqs. 2-10) with "
            "friction amplitudes from [Nasution2014b] Eqs. 7-9 as read here; "
            "bulk rainflow + Miner on elastic outer-fibre stress; trellis "
            "crossings by the point method of critical distances: contact field "
            "after plastic indentation plus bulk stress at depth L/2, critical "
            f"planes, SWT mean-stress correction, L = {cd_length * 1e6:.0f} um"
        )
        citations[2:2] = [
            "[RaoofHobbs1988] Eqs. 23-25",
            "[Papailiou1997] Eqs. 7-9",
            "[Johnson1985] §3.6, §6.3",
            "[TromansSun1996]",
            "[Taylor2007]",
            "[SmithWatsonTopper1970]",
        ]
        limitations += [
            "trellis threshold dK_th from annealed electrolytic copper "
            f"({critical_distance.threshold_MPa_sqrt_m:g} MPa m^0.5, band 2.1-3.8); "
            "no value for cold-drawn conductor wire was found",
            "tangential traction taken along the contact major axis (the wire "
            "axis for small crossing angles), with the sliding (Hertz-shaped) "
            "distribution in every cycle; residual stresses of the indentation "
            "are not included in the mean stress",
            "crossing force at the reference tension; wear of the crossings is "
            "not modelled",
        ]
    provenance = Provenance(
        method=method,
        citations=tuple(citations),
        limitations=tuple(limitations),
    )
    return ConductorFatigueResult(
        orientations_deg=np.degrees(theta),
        D_bulk_per_ring=d_bulk,
        D_bulk_per_orientation=d_bulk_orient,
        D_fretting_per_ring=d_fret,
        D_total_per_ring=d_total,
        stress_range_max_Pa=stress_range,
        contact_p0_Pa=p0_max,
        contact_p0_elastic_Pa=p0_elastic_max,
        crossing_force_N=force_max,
        critical_curvature_1_m=kappa_c,
        slip_cycle_fraction=slip_fraction,
        critical_ring=critical,
        governing=governing,
        fretting_evaluated=fretting or critical_distance is not None,
        parameters=parameters,
        provenance=provenance,
        D_strain_life_per_ring=None if strain_life is None else d_life,
        critical_distance_m=cd_length if critical_distance is not None else None,
        trellis_azimuth_deg=cd_azimuth if critical_distance is not None else None,
        trellis_x_over_a=cd_x if critical_distance is not None else None,
        trellis_face=cd_face if critical_distance is not None else None,
    )
