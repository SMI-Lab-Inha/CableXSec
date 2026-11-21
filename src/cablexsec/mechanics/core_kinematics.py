# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Slip state of the laid power cores and the strain field inside a core.

Gap P8. A metallic sheath, a laminate foil or a wire screen of a three-core
cable sits inside one power core. Its longitudinal strain is the strain of
the core axis plus the strain of the core's own bending,

.. math::

    \varepsilon(\varphi, \rho, \theta, t) = \varepsilon_c(\varphi, t)
    + \rho \left[g_r(\varphi, t) \cos\theta + g_t(\varphi, t)\sin\theta\right],

where ``phi`` is the azimuth of the core axis on the pitch circle (the
station of the core along its lay, measured from the positive extension side
of ``kappa_x``), ``rho`` and ``theta`` are polar coordinates in the core
cross-section (``theta = 0`` points radially away from the cable axis,
``theta = pi/2`` along the lay circle in the direction of increasing
``phi``) and ``g_r``, ``g_t`` are the strain gradients of the core's own
bending: the two Lutchansky components of the helical core axis,
``g_r = cos^2(beta) cos(2 beta) kappa_r + (sin^2(beta) / r_c) epsilon_b`` and
``g_t = cos(beta) (1 + sin^2 beta) kappa_t``
(:func:`cablexsec.mechanics.core_lay.core_bending_components`, after
[MenardCartraud2023] Eq. 14), where ``epsilon_b`` is the bending part of the
core-axis strain (the normal curvature of a helical rod follows its axial
slip state, :func:`cablexsec.mechanics.wire_stress.slip_state_normal_curvature`:
``g_r = cos^4(beta) kappa_r`` stuck, ``cos^2(beta) cos(2 beta) kappa_r``
free). The slip state sets the core-axis strain ``epsilon_c`` and, through
it, the normal core curvature:

``"stuck"``
    Plane-section bound: the core is bonded to the cable cross-section,
    ``epsilon_c = r_c cos^2(beta) (kappa_x cos phi + kappa_y sin phi)``
    (``r_c`` pitch-circle radius, ``beta`` core lay angle; the strain of a
    helical fibre of the cable section, as for a stuck armour wire with
    ``xi = 1``). Upper bound of the core-axis strain.
``"free"``
    Full-slip bound: the core slides freely along its helix, its axis keeps
    zero bending strain, ``epsilon_c = 0``; every layer bends about the core
    axis only, ``epsilon = rho (g_r cos theta + g_t sin theta)``.
``"frictional"``
    The core stick-slip law of :mod:`cablexsec.mechanics.core_friction`
    (gap R1b): every station is a helical core of rigidity ``EA_core`` with
    the Papailiou cap set by the core contact loads (bounded estimate or the
    solved trefoil contact), the shear-lag stick coefficient ``xi`` and the
    elastic--Coulomb return map;
    ``epsilon_c = sigma_c / E_core`` with ``sigma_c`` the station stress of
    :func:`cablexsec.mechanics.papailiou.wire_slip_history`.

A prescribed cable axial strain ``epsilon_a`` adds ``epsilon_a cos^2 beta``
to the core axis in every state (equal strain of the laid core).

Assumptions (stated)
--------------------
* The layers of one core are bonded and share the core-axis strain; a layer
  that can slip on its neighbours inside the core (a wire screen) receives
  the core-axis strain through its contact and is assessed separately
  (:mod:`cablexsec.analysis.core_screen`).
* Small strains; the core is a rod (no ovalisation, no hoop contact change).
* In the stuck bound the core is fully constrained (``xi = 1``); the
  frictional state reaches ``xi`` times this value in stick.

References
----------
[Papailiou1997]; [Yun2020]; [MenardCartraud2023] Eq. 14 and Section 5;
[WitzTan1992b] (plane-section helical strain).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cablexsec.cables.schema import CableData, CylindricalOffsetLayer
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.mechanics.core_friction import (
    CoreFrictionModel,
    CoreFrictionSpec,
    _core_geometry,
    build_core_friction,
    core_station_history,
)
from cablexsec.mechanics.core_lay import core_angle
from cablexsec.mechanics.radial_cable import CableRadialState
from cablexsec.mechanics.section_state import SectionState
from cablexsec.mechanics.wire_stress import slip_state_normal_curvature

__all__ = [
    "CORE_SLIP_STATES",
    "CoreKinematics",
    "CoreSlipState",
    "core_kinematics",
    "core_layer",
]

FloatArray = NDArray[np.float64]

#: Kinematic state of the laid cores: plane-section bound, full-slip bound or
#: the frictional stick-slip law.
CoreSlipState = Literal["stuck", "free", "frictional"]
CORE_SLIP_STATES: tuple[str, ...] = get_args(CoreSlipState)

#: Default number of equally spaced core stations along one lay (stuck/free).
DEFAULT_CORE_STATIONS = 36


@dataclass(frozen=True)
class CoreKinematics:
    """Core-axis strain and core bending gradients per station and time.

    Arrays have shape ``(station, time)``. ``phase_rad`` is the azimuth of the
    core axis on the pitch circle at each station. ``slipping`` flags the
    stations in slip (frictional state only, else ``None``). ``xi`` is the
    stick coefficient of the core-axis strain (1 stuck, 0 free, the shear-lag
    value of the frictional law).
    """

    state: str
    n_cores: int
    r_offset_m: float
    lay_angle_rad: float
    phase_rad: FloatArray
    axis_strain: FloatArray
    gradient_radial_per_m: FloatArray
    gradient_tangential_per_m: FloatArray
    xi: float
    slipping: NDArray[np.bool_] | None = None
    model: CoreFrictionModel | None = None

    @property
    def n_stations(self) -> int:
        return int(self.phase_rad.size)

    def strain(self, radius_m: float, theta_rad: ArrayLike) -> FloatArray:
        """Longitudinal strain at core radius ``rho`` and angles ``theta``.

        Returns shape ``(station, theta, time)``.
        """
        if not math.isfinite(radius_m) or radius_m < 0.0:
            raise CableInputError("core radius must be finite and non-negative")
        theta = np.atleast_1d(np.asarray(theta_rad, dtype=np.float64))
        if theta.ndim != 1 or not np.isfinite(theta).all():
            raise CableInputError("theta must be a finite 1-D array")
        c = np.cos(theta)[None, :, None]
        s = np.sin(theta)[None, :, None]
        return np.asarray(
            self.axis_strain[:, None, :]
            + radius_m
            * (
                self.gradient_radial_per_m[:, None, :] * c
                + self.gradient_tangential_per_m[:, None, :] * s
            )
        )

    def mean_strain(self) -> FloatArray:
        """Mean strain over any circle about the core axis: the axis strain."""
        return self.axis_strain


def core_layer(cable: CableData, layer_name: str) -> CylindricalOffsetLayer:
    """The named layer of a laid power core (``cylindrical_offset``)."""
    try:
        layer = cable.get_layer(layer_name)
    except KeyError as exc:
        raise CableSchemaError(str(exc)) from None
    if not isinstance(layer, CylindricalOffsetLayer):
        raise CableSchemaError(
            f"layer {layer_name!r} is a {type(layer).__name__}, not a layer of a "
            "power core (cylindrical_offset)"
        )
    return layer


def _histories(
    kappa_x: ArrayLike, kappa_y: ArrayLike | None
) -> tuple[FloatArray, FloatArray]:
    kx = np.asarray(kappa_x, dtype=np.float64)
    ky = np.zeros_like(kx) if kappa_y is None else np.asarray(kappa_y, dtype=np.float64)
    if kx.ndim != 1 or ky.shape != kx.shape or kx.size < 1:
        raise CableInputError(
            "kappa_x and kappa_y must be equal non-empty 1-D histories"
        )
    if not (np.isfinite(kx).all() and np.isfinite(ky).all()):
        raise CableInputError("curvature histories must be finite")
    return kx, ky


def _laid_geometry(cable: CableData) -> tuple[int, float, float]:
    """``(n_cores, r_c, beta)``; aligned cores (no ``core_lay``) give beta 0."""
    cores = [x for x in cable.layers if isinstance(x, CylindricalOffsetLayer)]
    offsets = {x.r_offset_m for x in cores}
    counts = {x.n_cores for x in cores}
    if not cores or len(offsets) != 1 or len(counts) != 1 or None in offsets:
        raise CableSchemaError(
            "core kinematics need one family of power cores with a common "
            "pitch-circle radius and core count"
        )
    r_c = float(next(iter(offsets)))  # type: ignore[arg-type]
    return int(next(iter(counts))), r_c, abs(core_angle(cable, cores[0]))


def core_kinematics(
    cable: CableData,
    kappa_x: ArrayLike,
    kappa_y: ArrayLike | None = None,
    *,
    state: CoreSlipState,
    axial_strain: ArrayLike | None = None,
    n_stations: int | None = None,
    core_friction: CoreFrictionSpec | None = None,
    contact_tension_N: float = 0.0,
    contact_model: str = "bounded",
    radial_state: CableRadialState | None = None,
    section_state: SectionState | None = None,
    initial_bending_angle: float = 0.0,
) -> CoreKinematics:
    """Core-axis strain and core bending of the laid cores under a history.

    Parameters
    ----------
    cable : CableData
        Cable with one family of power cores (``cylindrical_offset`` layers).
    kappa_x, kappa_y : array_like
        Cable curvature history [1/m] (``kappa_y`` default zero).
    state : {"stuck", "free", "frictional"}
        Core slip state (module notes). Required: the choice is explicit.
    axial_strain : array_like or None
        Cable axial strain history, added as ``epsilon cos^2 beta``.
    n_stations : int or None
        Stations along one lay for the bounds (default 36). The frictional
        state uses the population of the core law
        (``n_cores * core_friction.n_stations_per_core``) and rejects a
        different value.
    core_friction : CoreFrictionSpec or None
        Inputs of the frictional law (default :class:`CoreFrictionSpec`).
    contact_tension_N, contact_model, radial_state, section_state
        Normal contact loads of the frictional law
        (:func:`~cablexsec.mechanics.core_friction.build_core_friction`):
        ``contact_model="solved"`` uses the solved trefoil contact.
    initial_bending_angle : float
        Held bending direction at the start (frictional state).
    """
    if state not in CORE_SLIP_STATES:
        raise CableInputError(
            f"core slip state must be one of {', '.join(CORE_SLIP_STATES)}"
        )
    kx, ky = _histories(kappa_x, kappa_y)
    eps_axial = None
    if axial_strain is not None:
        eps_axial = np.asarray(axial_strain, dtype=np.float64)
        if eps_axial.shape != kx.shape or not np.isfinite(eps_axial).all():
            raise CableInputError("axial strain must be finite and match the curvature")
    model: CoreFrictionModel | None = None
    slipping: NDArray[np.bool_] | None = None
    if state == "frictional":
        n_cores, r_c, beta, _ = _core_geometry(cable)
        spec = core_friction if core_friction is not None else CoreFrictionSpec()
        model = build_core_friction(
            cable,
            spec,
            contact_tension_N,
            contact_model=contact_model,
            radial_state=radial_state,
            section_state=section_state,
        )
        population = model.population
        if n_stations is not None and n_stations != population.n_wires:
            raise CableInputError(
                "the frictional state uses n_cores * n_stations_per_core stations "
                f"({population.n_wires}); set CoreFrictionSpec.n_stations_per_core"
            )
        history = core_station_history(
            model, kx, ky, initial_bending_angle=initial_bending_angle
        )
        phase = np.asarray(population.phi_wires, dtype=np.float64)
        axis = np.asarray(history.stress_Pa / population.E_wire_Pa)
        # A station slips in a step whose stress increment is not the elastic
        # stick increment k_f r (d kappa . e_phi) of the return map.
        stick = (
            population.k_f
            * population.r_helix_m
            * (
                np.cos(phase)[:, None] * kx[None, :]
                + np.sin(phase)[:, None] * ky[None, :]
            )
        )
        d_stress = np.diff(history.stress_Pa, axis=1, prepend=0.0)
        d_stick = np.diff(stick, axis=1, prepend=0.0)
        scale = float(np.max(np.abs(stick))) if stick.size else 0.0
        slipping = np.asarray(np.abs(d_stress - d_stick) > 1e-9 * (scale + 1.0))
        xi = float(model.xi)
    else:
        n_cores, r_c, beta = _laid_geometry(cable)
        count = DEFAULT_CORE_STATIONS if n_stations is None else n_stations
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise CableInputError("n_stations must be a positive integer")
        phase = 2.0 * np.pi * np.arange(count) / count
        if state == "stuck":
            axis = (
                r_c
                * math.cos(beta) ** 2
                * (
                    np.cos(phase)[:, None] * kx[None, :]
                    + np.sin(phase)[:, None] * ky[None, :]
                )
            )
            xi = 1.0
        else:
            axis = np.zeros((count, kx.size))
            xi = 0.0
    # The normal curvature of the core follows its slip state through the
    # bending part of the core-axis strain (before the axial strain is added).
    held = slip_state_normal_curvature(axis, 1.0, beta, r_c) if r_c > 0.0 else 0.0
    if eps_axial is not None:
        axis = axis + math.cos(beta) ** 2 * eps_axial[None, :]
    c, s = math.cos(beta), math.sin(beta)
    kr = np.cos(phase)[:, None] * kx[None, :] + np.sin(phase)[:, None] * ky[None, :]
    kt = -np.sin(phase)[:, None] * kx[None, :] + np.cos(phase)[:, None] * ky[None, :]
    return CoreKinematics(
        state=state,
        n_cores=n_cores,
        r_offset_m=r_c,
        lay_angle_rad=beta,
        phase_rad=phase,
        axis_strain=np.asarray(axis, dtype=np.float64),
        gradient_radial_per_m=np.asarray(c * c * math.cos(2.0 * beta) * kr + held),
        gradient_tangential_per_m=np.asarray(c * (1.0 + s * s) * kt),
        xi=xi,
        slipping=slipping,
        model=model,
    )
