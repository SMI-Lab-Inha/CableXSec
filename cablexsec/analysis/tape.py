# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Helical tape layers: strain along the tape, slip, overlap, edge strain, fatigue.

Gap P2. A metallic tape (a copper screen tape, a binder) is wound as a
helix of lay angle ``beta`` (to the layer axis), width ``w`` and thickness
``t`` on a substrate of mid radius ``rho``
(:class:`~cablexsec.cables.schema.TapeWinding`). The substrate surface
carries the longitudinal strain field of its layer,

.. math::

    \varepsilon_z(\theta, t) = \varepsilon_c(t) + \rho\,[g_r(t)\cos\theta
    + g_t(t)\sin\theta], \qquad \varepsilon_\theta = h\,\varepsilon_z ,

with ``h`` the hoop-to-axial ratio of the substrate surface (``-nu`` of the
polymer underneath by default). For a tape of a power core ``epsilon_c``,
``g_r``, ``g_t`` are the core-axis strain and core bending gradients of the
declared core slip state (:mod:`cablexsec.mechanics.core_kinematics`,
``theta = 0`` away from the cable axis), so per-core tapes follow the core
slip state; for a concentric tape they are the cable axial strain and
curvature (``theta = 0`` on the ``x`` axis). The strain along the tape
direction ``(cos beta, sin beta)`` of a fibre bonded to the surface is
``epsilon_z (cos^2 beta + h sin^2 beta)`` (no surface shear in bending).

Tape states
-----------
``"stuck"``
    The tape follows the surface (bonded, or stuck by friction): upper
    bound. A fibre of the tape at the transverse offset ``eta`` sits at
    ``theta + eta cos(beta) / rho``, so the two edges
    (``eta = +-w/2``) see the surface strain at angles ``+-w cos(beta) /
    (2 rho)`` from the centre line: the edge strain.
``"free"``
    The tape slides freely along its helix (no friction): its membrane
    strain is the uniform part ``epsilon_c (cos^2 beta + h sin^2 beta)``
    (common to the layer, no slip demand) and every fibre carries the
    tape's own bending, flat-wise ``+-(t/2) cos^2(beta) cos(2 beta) kappa_r``
    (the radial Lutchansky component of a helix, [MenardCartraud2023]
    Eq. 14, with ``kappa_r`` the bending curvature component towards the
    fibre's azimuth). The edge-wise component ``cos(beta)(1 + sin^2 beta)
    kappa_t`` acts on the edges as ``+-(w/2)`` times it only if the tape
    keeps its loxodromic path (``path="loxodromic"``, an upper bound for a
    wide tape); a tape that also slides transversely onto the geodesic path
    (``path="geodesic"``, default) has no edge-wise bending ([Saevik1992]
    quantified this transverse slip for tendons).
``"frictional"``
    Elastic--Coulomb stick-slip along the tape on the declared
    friction-setting pressure ``P1 + P2`` and friction coefficient: the
    bending part of the stuck strain, per fibre line of the tape
    (centre and edges), is returned through the Papailiou cap
    ``sigma_0 expm1(mu sin(beta) psi)`` with
    ``sigma_0 = rho (P1 + P2) / (t sin^2 beta)`` and ``psi`` the helix
    angle from the neutral line of the current bending direction
    ([Papailiou1997]; the per-unit-length friction force of a tape is
    ``mu (P1 + P2) w``), plus the uniform part.

Slip and overlap
----------------
The slip of the tape along its length relative to the substrate is the
integral of the strain the substrate imposes minus the strain the tape
carries, ``u(theta) = int (eps_stuck - eps_tape) rho / sin(beta) dtheta``
(zero mean over a turn); its amplitude is reported. The axial gap between
neighbouring turns is ``P / n - w / sin(beta)`` with ``P = 2 pi rho /
tan(beta)``; negative values are overlaps (``overlap_fraction`` of the
axial tape width). In an overlap the outer turn lies one thickness further
out and slides on the inner turn by the difference of their slips,
``(t / rho) |u|`` (reported as ``overlap_slip_amplitude_m``).

Fatigue
-------
The tape strain histories at the four fibres (two faces at the centre line
and the two edges) of every station and azimuth are rainflow counted and
assessed by the strain-life route (:mod:`cablexsec.materials.strain_life`:
Masing stresses, Coffin--Manson--Basquin with SWT or Morrow). The tape is
strain controlled in the stuck and free states. Copper tape uses the
[Poon2022] constants (``copper_poon2022``, a transfer from conductor wire;
no copper-tape strain-life data in the library); other tapes need their own
constants (an assumed set is labelled by its ``basis``).

Limitations (stated)
--------------------
* Thin tape, small strains, one tape layer; no tape-on-tape friction in an
  overlap (only its slip measure), no winding tension history.
* The edge stress from wires or tapes bearing on the tape edges (contact
  stress concentration) is not modelled: only the kinematic edge strain.
* Hoop strain of the substrate as ``h epsilon_z``; surface shear from
  torsion not included.

References
----------
[Papailiou1997]; [MenardCartraud2023] Eq. 14; [Saevik1992]; [Poon2022]
Table 4 and Eq. 30; [Dowling2013] Ch. 14; [Thies2016] Sec. 3.3.3 (tape
fractures from the edges).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cablexsec.analysis.sn_assessment import (
    ComponentSCF,
    corrected_damage_rows,
    scaled_component,
)
from cablexsec.cables.schema import (
    CableData,
    CylindricalLayer,
    CylindricalOffsetLayer,
    TapeWinding,
)
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.materials import library
from cablexsec.materials.mean_stress import MeanStressSpec
from cablexsec.materials.sn_curves import SNcurve, get_sn_curve
from cablexsec.materials.strain_life import (
    MEAN_STRESS_CORRECTIONS,
    MeanStressCorrection,
    StrainLifeConstants,
    get_strain_life,
    strain_life_damage_map,
)
from cablexsec.mechanics.core_friction import CoreFrictionSpec
from cablexsec.mechanics.core_kinematics import CoreSlipState, core_kinematics
from cablexsec.mechanics.radial_cable import CableRadialState
from cablexsec.provenance import Provenance

__all__ = [
    "TAPE_PATHS",
    "TAPE_STATES",
    "TapeGeometry",
    "TapeResult",
    "compute_tape_fatigue",
    "tape_geometry",
    "tape_strain",
]

FloatArray = NDArray[np.float64]
TapeState = Literal["stuck", "free", "frictional"]
TapePath = Literal["geodesic", "loxodromic"]
TAPE_STATES: tuple[str, ...] = ("stuck", "free", "frictional")
TAPE_PATHS: tuple[str, ...] = ("geodesic", "loxodromic")
#: Fibres of the tape: (name, transverse position in units of w/2, face +-1).
FIBRES: tuple[tuple[str, float, float], ...] = (
    ("centre_outer", 0.0, 1.0),
    ("centre_inner", 0.0, -1.0),
    ("edge_plus", 1.0, 1.0),
    ("edge_minus", -1.0, 1.0),
)


@dataclass(frozen=True)
class TapeGeometry:
    """Helix of a tape: radius, thickness, angle, pitch, gap and overlap."""

    radius_m: float
    thickness_m: float
    width_m: float
    lay_angle_rad: float
    n_tapes: int
    pitch_m: float
    axial_gap_m: float
    coverage: float
    overlap_fraction: float
    edge_angle_rad: float

    @property
    def open_helix(self) -> bool:
        return self.axial_gap_m > 0.0


def tape_geometry(
    radius_m: float, thickness_m: float, winding: TapeWinding
) -> TapeGeometry:
    """Pitch ``2 pi rho / tan beta``, axial gap ``P / n - w / sin beta``,
    coverage ``n w / (P sin beta)``, overlap fraction and edge angle
    ``w cos(beta) / (2 rho)``."""
    if not (math.isfinite(radius_m) and radius_m > 0):
        raise CableInputError("tape radius must be positive")
    if not (math.isfinite(thickness_m) and thickness_m > 0):
        raise CableInputError("tape thickness must be positive")
    beta = math.radians(winding.lay_angle_deg)
    pitch = 2.0 * math.pi * radius_m / math.tan(beta)
    axial_width = winding.width_m / math.sin(beta)
    gap = pitch / winding.n_tapes - axial_width
    coverage = winding.n_tapes * axial_width / pitch
    return TapeGeometry(
        radius_m=radius_m,
        thickness_m=thickness_m,
        width_m=winding.width_m,
        lay_angle_rad=beta,
        n_tapes=winding.n_tapes,
        pitch_m=pitch,
        axial_gap_m=gap,
        coverage=coverage,
        overlap_fraction=max(0.0, -gap) / axial_width,
        edge_angle_rad=0.5 * winding.width_m * math.cos(beta) / radius_m,
    )


def _papailiou_return(
    drive: FloatArray,
    direction: FloatArray,
    theta: FloatArray,
    stiffness: float,
    sigma0: float,
    mu_sin: float,
) -> FloatArray:
    """Direction-tied elastic--Coulomb return map, shape (fibre line, time).

    ``drive`` is the stuck bending strain per line and time; the cap of a
    line at ``theta`` is ``sigma0 expm1(mu sin(beta) asin|cos(theta - b)|)``
    with ``b`` the current bending direction (held while the bending
    vanishes).
    """
    n_l, n_t = drive.shape
    out = np.zeros((n_l, n_t))
    force = np.zeros(n_l)
    previous = np.zeros(n_l)
    for i in range(n_t):
        psi = np.arcsin(np.clip(np.abs(np.cos(theta - direction[i])), 0.0, 1.0))
        cap = sigma0 * np.expm1(mu_sin * psi)
        trial = force + stiffness * (drive[:, i] - previous)
        force = np.clip(trial, -cap, cap)
        previous = drive[:, i]
        out[:, i] = force
    return out


def tape_strain(
    geometry: TapeGeometry,
    axis_strain: ArrayLike,
    gradient_radial_per_m: ArrayLike,
    gradient_tangential_per_m: ArrayLike,
    theta_rad: ArrayLike,
    *,
    state: TapeState,
    hoop_ratio: float,
    path: TapePath = "geodesic",
    E_Pa: float | None = None,
    contact_pressure_Pa: float | None = None,
    mu: float | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Tape strain at the fibres and the slip along the tape.

    ``axis_strain``, ``gradient_*`` have shape ``(station, time)``;
    ``theta_rad`` are the centre-line azimuths of the tape on one turn.
    Returns ``(strain, slip_m)``: strain of shape
    ``(station, theta, fibre, time)`` over :data:`FIBRES`, slip of the
    centre line along the tape of shape ``(station, theta, time)``.
    ``E_Pa``, ``contact_pressure_Pa`` and ``mu`` are needed for the
    frictional state.
    """
    if state not in TAPE_STATES:
        raise CableInputError(f"tape state must be one of {TAPE_STATES}")
    if path not in TAPE_PATHS:
        raise CableInputError(f"tape path must be one of {TAPE_PATHS}")
    if not (math.isfinite(hoop_ratio) and -1.0 <= hoop_ratio <= 1.0):
        raise CableInputError("hoop_ratio must lie in [-1, 1]")
    eps_c = np.atleast_2d(np.asarray(axis_strain, dtype=np.float64))
    g_r = np.atleast_2d(np.asarray(gradient_radial_per_m, dtype=np.float64))
    g_t = np.atleast_2d(np.asarray(gradient_tangential_per_m, dtype=np.float64))
    if not (eps_c.shape == g_r.shape == g_t.shape) or eps_c.ndim != 2:
        raise CableInputError(
            "strain and gradients must share the (station, time) shape"
        )
    theta = np.atleast_1d(np.asarray(theta_rad, dtype=np.float64))
    beta = geometry.lay_angle_rad
    c2, s2 = math.cos(beta) ** 2, math.sin(beta) ** 2
    factor = c2 + hoop_ratio * s2
    rho = geometry.radius_m
    half_t = 0.5 * geometry.thickness_m
    half_w = 0.5 * geometry.width_m
    n_s, n_t = eps_c.shape
    out = np.zeros((n_s, theta.size, len(FIBRES), n_t))
    slip = np.zeros((n_s, theta.size, n_t))
    if state == "frictional":
        if E_Pa is None or contact_pressure_Pa is None or mu is None:
            raise CableInputError(
                "the frictional tape state needs E_Pa, contact_pressure_Pa and mu"
            )
        if contact_pressure_Pa < 0 or mu < 0:
            raise CableInputError("contact pressure and mu must be non-negative")
    lines = sorted({w for _, w, _ in FIBRES})
    for s in range(n_s):
        direction = np.arctan2(g_t[s], g_r[s])
        hold = np.hypot(g_t[s], g_r[s]) == 0.0
        if np.any(hold):
            last = 0.0
            for i in range(n_t):
                if hold[i]:
                    direction[i] = last
                else:
                    last = direction[i]
        line_strain: dict[float, FloatArray] = {}
        for w_pos in lines:
            ang = theta + w_pos * geometry.edge_angle_rad
            bend_stuck = rho * (
                np.cos(ang)[:, None] * g_r[s][None, :]
                + np.sin(ang)[:, None] * g_t[s][None, :]
            )
            uniform = factor * eps_c[s][None, :] * np.ones((ang.size, 1))
            if state == "stuck":
                membrane = uniform + factor * bend_stuck
            elif state == "free":
                membrane = uniform
            else:
                assert E_Pa is not None and contact_pressure_Pa is not None
                assert mu is not None
                k = E_Pa * factor
                sigma0 = (
                    rho
                    * contact_pressure_Pa
                    / (geometry.thickness_m * math.sin(beta) ** 2)
                )
                stress = _papailiou_return(
                    bend_stuck, direction, ang, k, sigma0, mu * math.sin(beta)
                )
                membrane = uniform + stress / E_Pa
            line_strain[w_pos] = membrane
            if w_pos == 0.0:
                # slip along the tape: integral of (stuck - carried) strain
                demand = uniform + factor * bend_stuck - membrane
                order = np.argsort(theta)
                th = theta[order]
                d = demand[order]
                ds = rho / math.sin(beta) * np.diff(th)
                u = np.concatenate(
                    [
                        np.zeros((1, n_t)),
                        np.cumsum(0.5 * (d[1:] + d[:-1]) * ds[:, None], axis=0),
                    ]
                )
                u -= u.mean(axis=0, keepdims=True)
                back = np.empty_like(u)
                back[order] = u
                slip[s] = back
        # own bending of the tape (free and frictional states slide)
        kr = (
            np.cos(theta)[:, None] * g_r[s][None, :]
            + np.sin(theta)[:, None] * g_t[s][None, :]
        )
        kt = (
            -np.sin(theta)[:, None] * g_r[s][None, :]
            + np.cos(theta)[:, None] * g_t[s][None, :]
        )
        flat = c2 * math.cos(2 * beta) * kr
        edge = math.cos(beta) * (1.0 + s2) * kt
        for f, (_, w_pos, face) in enumerate(FIBRES):
            value = line_strain[w_pos]
            if state != "stuck":
                value = value + face * half_t * flat
                if path == "loxodromic" and w_pos != 0.0:
                    value = value + w_pos * half_w * edge
            else:
                value = value + face * half_t * factor * (
                    np.cos(theta + w_pos * geometry.edge_angle_rad)[:, None]
                    * g_r[s][None, :]
                    + np.sin(theta + w_pos * geometry.edge_angle_rad)[:, None]
                    * g_t[s][None, :]
                )
            out[s, :, f, :] = value
    return out, slip


@dataclass(frozen=True)
class TapeResult:
    """Tape strain, slip and strain-life damage of a tape layer.

    ``strain`` has shape ``(station, theta, fibre, time)`` over
    :data:`FIBRES`; ``D_map`` (Miner damage of the record) has shape
    ``(station, theta, fibre)``. ``phase_deg`` are the core stations (one
    for a concentric tape), ``theta_deg`` the tape azimuths.
    """

    layer_name: str
    state: str
    path: str
    core_state: str | None
    geometry: TapeGeometry
    hoop_ratio: float
    phase_deg: FloatArray
    theta_deg: FloatArray
    strain: FloatArray
    slip_amplitude_m: FloatArray
    overlap_slip_amplitude_m: FloatArray
    strain_amplitude_max: float
    edge_to_centre_ratio: float
    D_map: FloatArray
    D_miner: float
    critical: dict[str, float | str]
    constants: StrainLifeConstants
    correction: str
    provenance: Provenance
    D_sn_map: FloatArray | None = None
    D_sn: float | None = None
    sn_curve: str | None = None


def _tape_layer(
    cable: CableData, layer_name: str
) -> CylindricalOffsetLayer | CylindricalLayer:
    try:
        layer = cable.get_layer(layer_name)
    except KeyError as exc:
        raise CableSchemaError(str(exc)) from None
    if not isinstance(layer, (CylindricalOffsetLayer, CylindricalLayer)):
        raise CableSchemaError(
            f"layer {layer_name!r} is a {type(layer).__name__}; a tape is a "
            "cylindrical or cylindrical_offset layer with a tape winding"
        )
    return layer


def _substrate_nu(
    cable: CableData, layer: CylindricalOffsetLayer | CylindricalLayer
) -> float:
    """Poisson ratio of the outermost polymer layer under the tape.

    The surface hoop strain is set by the polymer body the tape is wound
    over (a wire screen in between follows it); 0.5 if none is found.
    """
    best: tuple[float, float] | None = None
    limit = float(layer.r_inner_m or 0.0) + 1e-9
    for other in cable.layers:
        r_out = getattr(other, "r_outer_m", None)
        if other is layer or r_out is None or float(r_out) > limit:
            continue
        if isinstance(layer, CylindricalOffsetLayer) != isinstance(
            other, CylindricalOffsetLayer
        ):
            continue
        if not library.is_polymer(other.material):
            continue
        try:
            nu = float(library.get_property(other.material, "nu"))
        except KeyError:
            continue
        if best is None or float(r_out) > best[0]:
            best = (float(r_out), nu)
    return 0.5 if best is None else best[1]


def compute_tape_fatigue(
    cable: CableData,
    layer_name: str,
    kappa_x: ArrayLike,
    kappa_y: ArrayLike | None = None,
    *,
    state: TapeState,
    core_state: CoreSlipState | None = None,
    path: TapePath = "geodesic",
    axial_strain: ArrayLike | None = None,
    hoop_ratio: float | None = None,
    n_theta: int = 36,
    n_stations: int | None = None,
    contact_pressure_Pa: float | None = None,
    mu: float | None = None,
    strain_life: str | StrainLifeConstants | None = None,
    correction: MeanStressCorrection = "swt",
    core_friction: CoreFrictionSpec | None = None,
    contact_tension_N: float = 0.0,
    contact_model: str = "bounded",
    radial_state: CableRadialState | None = None,
    sn_curve: str | SNcurve | None = None,
    mean_stress: MeanStressSpec | None = None,
    scf: ComponentSCF | None = None,
) -> TapeResult:
    """Strain, slip and strain-life damage of a helical tape layer (gap P2).

    With ``sn_curve`` (a registered name or a curve) the elastic tape stress
    ``E eps`` is also assessed against that stress-range curve, with the
    optional ``mean_stress`` correction and the axial factors of ``scf``
    (mean and dynamic), giving ``D_sn_map`` and ``D_sn``. The strain-life
    damage is unchanged.

    Parameters
    ----------
    cable : CableData
    layer_name : str
        A ``cylindrical`` or ``cylindrical_offset`` layer with ``tape``.
    kappa_x, kappa_y : array_like
        Cable curvature history [1/m].
    state : {"stuck", "free", "frictional"}
        Tape state (module notes); required.
    core_state : {"stuck", "free", "frictional"} or None
        Core slip state for a tape of a power core (required there).
    path : {"geodesic", "loxodromic"}
        Path of a sliding tape (edge-wise bending only on the loxodromic).
    axial_strain : array_like or None
        Cable axial strain history.
    hoop_ratio : float or None
        Hoop-to-axial strain ratio of the substrate surface (default
        ``-nu`` of the layer under the tape).
    n_theta, n_stations : int
        Azimuths along one tape turn; core stations along the lay.
    contact_pressure_Pa, mu : float or None
        Friction-setting pressure ``P1 + P2`` on the tape and its friction
        coefficient (frictional state; no defaults).
    strain_life : str or StrainLifeConstants or None
        Strain-life constants (default: the layer's ``strain_life``).
    correction : {"swt", "morrow", "none"}
        Mean-stress correction.
    core_friction, contact_tension_N, contact_model, radial_state
        Inputs of the frictional core law (core_state="frictional").
    """
    layer = _tape_layer(cable, layer_name)
    if layer.tape is None:
        raise CableSchemaError(f"layer {layer_name!r} declares no tape winding")
    if layer.r_inner_m is None or layer.r_outer_m is None:
        raise CableSchemaError(f"layer {layer_name!r} has no complete radii")
    if correction not in MEAN_STRESS_CORRECTIONS:
        raise CableInputError(
            f"mean-stress correction must be one of {MEAN_STRESS_CORRECTIONS}"
        )
    if isinstance(n_theta, bool) or not isinstance(n_theta, int) or n_theta < 4:
        raise CableInputError("n_theta must be an integer of at least 4")
    key = strain_life if strain_life is not None else layer.strain_life
    if not key:
        raise CableSchemaError(
            f"layer {layer_name!r} declares no strain_life constants; pass strain_life="
        )
    constants = get_strain_life(key) if isinstance(key, str) else key
    radius = 0.5 * (layer.r_inner_m + layer.r_outer_m)
    geometry = tape_geometry(radius, layer.r_outer_m - layer.r_inner_m, layer.tape)
    kx = np.asarray(kappa_x, dtype=np.float64)
    ky = np.zeros_like(kx) if kappa_y is None else np.asarray(kappa_y, dtype=np.float64)
    if (
        kx.ndim != 1
        or ky.shape != kx.shape
        or not (np.isfinite(kx).all() and np.isfinite(ky).all())
    ):
        raise CableInputError("finite equal 1-D curvature histories are required")
    strain_in = None
    if axial_strain is not None:
        strain_in = np.asarray(axial_strain, dtype=np.float64)
        if strain_in.shape != kx.shape or not np.isfinite(strain_in).all():
            raise CableInputError("axial strain must be finite and match the curvature")
    if isinstance(layer, CylindricalOffsetLayer):
        if core_state is None:
            raise CableInputError(
                f"tape {layer_name!r} is inside a power core: give core_state"
            )
        kin = core_kinematics(
            cable,
            kx,
            ky,
            state=core_state,
            axial_strain=strain_in,
            n_stations=n_stations if core_state != "frictional" else None,
            core_friction=core_friction,
            contact_tension_N=contact_tension_N,
            contact_model=contact_model,
            radial_state=radial_state,
        )
        eps_c = kin.axis_strain
        g_r = kin.gradient_radial_per_m
        g_t = kin.gradient_tangential_per_m
        phase = np.degrees(kin.phase_rad)
        if core_state == "frictional" and n_stations is not None:
            pick = np.unique(
                np.round(np.arange(n_stations) * kin.n_stations / n_stations).astype(
                    int
                )
            )
            eps_c, g_r, g_t, phase = eps_c[pick], g_r[pick], g_t[pick], phase[pick]
    else:
        if core_state is not None:
            raise CableInputError("core_state applies to a tape of a power core")
        eps_c = (np.zeros_like(kx) if strain_in is None else strain_in)[None, :]
        g_r = kx[None, :]
        g_t = ky[None, :]
        phase = np.zeros(1)
    h = -_substrate_nu(cable, layer) if hoop_ratio is None else float(hoop_ratio)
    try:
        e_tape = library.get_property(layer.material, "E_Pa")
    except KeyError:
        e_tape = constants.E_Pa
    theta = 2.0 * np.pi * np.arange(n_theta) / n_theta
    strain, slip = tape_strain(
        geometry,
        eps_c,
        g_r,
        g_t,
        theta,
        state=state,
        hoop_ratio=h,
        path=path,
        E_Pa=e_tape,
        contact_pressure_Pa=contact_pressure_Pa,
        mu=mu,
    )
    n_s, n_th, n_f, n_t = strain.shape
    life = strain_life_damage_map(
        strain.reshape(n_s * n_th * n_f, n_t), constants, correction=correction
    )
    d_map = life.damage.reshape(n_s, n_th, n_f)
    s, t, f = np.unravel_index(int(np.argmax(d_map)), d_map.shape)
    d_sn_map = None
    curve = None
    if sn_curve is not None:
        curve = get_sn_curve(sn_curve) if isinstance(sn_curve, str) else sn_curve
        stress = e_tape * strain.reshape(n_s * n_th * n_f, n_t)
        if scf is not None:
            stress = scaled_component(stress, scf.axial_mean, scf.axial_dynamic)
        d_sn_map = corrected_damage_rows(
            stress,
            curve,
            mean_stress=mean_stress,
            thickness_m=geometry.thickness_m,
        ).reshape(n_s, n_th, n_f)
    elif mean_stress is not None or scf is not None:
        raise CableInputError("mean_stress and scf of a tape need sn_curve")
    amplitude = 0.5 * np.ptp(strain, axis=3)
    centre = float(np.max(amplitude[:, :, 0]))
    edges = float(np.max(amplitude[:, :, 2:]))
    slip_amp = 0.5 * np.ptp(slip, axis=2)
    overlap_slip = (
        slip_amp * geometry.thickness_m / geometry.radius_m
        if geometry.overlap_fraction > 0
        else np.zeros_like(slip_amp)
    )
    limitations = [
        "thin tape, small strains; edge contact stress from wires or tapes "
        "bearing on the tape edge not modelled",
        f"hoop-to-axial ratio of the substrate {h:.3g}",
    ]
    if curve is not None:
        limitations.append(
            f"stress-range S-N check of the elastic tape stress E eps: curve "
            f"{curve.name} ({curve.reference}); "
            + (
                mean_stress.describe()
                if mean_stress is not None
                else "no mean-stress correction"
            )
            + "; "
            + (scf.describe() if scf is not None else "no component SCF")
            + " (axial factors); elastic stress above yield is not corrected"
        )
    if constants.basis != "published":
        limitations.append(
            f"strain-life constants {constants.name!r} are {constants.basis}: a bound"
        )
    if "copper" in layer.material.lower() and constants.name == "copper_poon2022":
        limitations.append(
            "copper conductor-wire strain-life constants transferred to the tape "
            "(no copper-tape data in the reference library)"
        )
    if state == "frictional":
        limitations.append(
            f"friction-setting pressure {contact_pressure_Pa:.6g} Pa and mu {mu:.3g} "
            "are inputs"
        )
    return TapeResult(
        layer_name=layer.name,
        state=state,
        path=path,
        core_state=core_state,
        geometry=geometry,
        hoop_ratio=h,
        phase_deg=np.asarray(phase, dtype=np.float64),
        theta_deg=np.degrees(theta),
        strain=strain,
        slip_amplitude_m=np.asarray(slip_amp),
        overlap_slip_amplitude_m=np.asarray(overlap_slip),
        strain_amplitude_max=float(np.max(amplitude)),
        edge_to_centre_ratio=edges / centre if centre > 0 else math.inf,
        D_map=d_map,
        D_miner=float(d_map[s, t, f]),
        critical={
            "phase_deg": float(phase[s]),
            "theta_deg": float(np.degrees(theta[t])),
            "fibre": FIBRES[f][0],
            "strain_amplitude": float(amplitude[s, t, f]),
        },
        constants=constants,
        correction=correction,
        provenance=Provenance(
            method=(
                f"helical tape ({state} state, {path} path): strain along the "
                "tape from the substrate strain field"
                + (f" of the {core_state!r} core state" if core_state else "")
                + "; rainflow, Masing stresses and strain-life damage"
            ),
            citations=(
                "[MenardCartraud2023] Eq. 14",
                "[Papailiou1997]",
                "[Dowling2013] Ch. 14",
                constants.reference,
            ),
            gates=(
                "stuck tape at beta -> 0 equals the substrate strain",
                "free tape: no bending membrane strain; slip = integral of the "
                "stuck strain",
                "frictional tape: stuck below slip onset, free-tape slip at zero "
                "pressure",
            ),
            limitations=tuple(limitations),
        ),
        D_sn_map=d_sn_map,
        D_sn=None if d_sn_map is None else float(np.max(d_sn_map)),
        sn_curve=None if curve is None else curve.name,
    )
