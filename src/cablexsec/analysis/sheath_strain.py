# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Strain field of the layers of a power core and strain-life sheath fatigue.

Gaps P8 and P6.

:func:`compute_core_layer_strain` gives the longitudinal strain of any layer
inside a laid power core (screen, foil, sheath) under the three core slip
states of :mod:`cablexsec.mechanics.core_kinematics`.

:func:`strain_life_sheath_fatigue` is the strain-life route of
:func:`cablexsec.compute_shell_fatigue`. The local strain at the critical
fibre of a metallic sheath, foil or tape is

* a concentric ``cylindrical_shell`` sheath of a cable:
  ``epsilon = K (kappa_x cos theta + kappa_y sin theta) r_mid + epsilon_a``
  with the strain concentration ``K = SCF f_corr`` of a corrugated sheath
  (the elastic root-strain ratio of the existing stress route,
  ``sigma_peak / E = SCF f_corr kappa r``, applied to strain; a lower
  estimate once the root yields, see the limitations) and ``K = 1`` for a
  smooth or bonded sheath;
* a layer of a power core (``cylindrical_offset``): the core-layer strain of
  the declared core slip state at the layer's mid radius.

A ``bonded_laminate`` foil follows the polymer it is bonded to
([Worzyk2009] Sec. 2.3.2): it is strain controlled in both directions, with
the hoop strain of the polymer ``epsilon_theta = -nu_p epsilon_z``. Its
von Mises equivalent strain (plastic incompressibility) is
``epsilon_eq = (2/sqrt 3) sqrt(1 + lambda + lambda^2) epsilon_z`` with
``lambda = -nu_p`` (1.0017 for ``nu_p = 0.45``; 1 for ``nu_p = 0.5``).
Each fibre's strain history is rainflow counted, the Masing stresses give
the maximum and mean stress of each cycle, and the Coffin--Manson--Basquin
curve with the SWT or Morrow correction gives the life
(:mod:`cablexsec.materials.strain_life`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cablexsec.cables.schema import (
    CableData,
    CylindricalOffsetLayer,
    CylindricalShellLayer,
)
from cablexsec.cables.schema import scope_entry_points as _scope_entry_points
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.materials import library
from cablexsec.materials.strain_life import (
    MEAN_STRESS_CORRECTIONS,
    MeanStressCorrection,
    StrainLifeConstants,
    get_strain_life,
    strain_life_damage_map,
)
from cablexsec.mechanics.core_friction import CoreFrictionSpec
from cablexsec.mechanics.core_kinematics import (
    CoreKinematics,
    CoreSlipState,
    core_kinematics,
    core_layer,
)
from cablexsec.mechanics.radial_cable import CableRadialState
from cablexsec.provenance import Provenance

__all__ = [
    "CoreLayerStrainResult",
    "SheathStrainLifeResult",
    "compute_core_layer_strain",
    "laminate_strain_factor",
    "strain_life_sheath_fatigue",
]

FloatArray = NDArray[np.float64]
DEFAULT_THETA = 36


@dataclass(frozen=True)
class CoreLayerStrainResult:
    """Longitudinal strain of a core layer, shape ``(station, theta, time)``.

    ``phase_deg``: core station on the pitch circle (0 deg extension side of
    ``kappa_x``); ``theta_deg``: angle in the core section (0 deg away from
    the cable axis). ``amplitude`` is half the range over the history.
    """

    layer_name: str
    core_state: str
    radius_m: float
    phase_deg: FloatArray
    theta_deg: FloatArray
    strain: FloatArray
    kinematics: CoreKinematics
    provenance: Provenance

    @property
    def amplitude(self) -> FloatArray:
        return np.asarray(0.5 * np.ptp(self.strain, axis=2))

    @property
    def mean_over_section(self) -> FloatArray:
        """Strain averaged over the circle ``theta``: the core-axis strain."""
        return np.asarray(np.mean(self.strain, axis=1))

    def maximum(self) -> dict[str, float]:
        """Largest strain amplitude and its station and angle."""
        amp = self.amplitude
        s, t = np.unravel_index(int(np.argmax(amp)), amp.shape)
        return {
            "strain_amplitude": float(amp[s, t]),
            "phase_deg": float(self.phase_deg[s]),
            "theta_deg": float(self.theta_deg[t]),
        }


def _layer_radius(layer: CylindricalOffsetLayer) -> float:
    if layer.strand_layers:
        rings = [r.r_helix_m for r in layer.strand_layers if r.r_helix_m > 0]
        if len(rings) == 1:
            return float(rings[0])
    if layer.r_inner_m is None or layer.r_outer_m is None:
        raise CableSchemaError(f"layer {layer.name!r} has no complete radii")
    return 0.5 * (layer.r_inner_m + layer.r_outer_m)


def compute_core_layer_strain(
    cable: CableData,
    layer_name: str,
    kappa_x: ArrayLike,
    kappa_y: ArrayLike | None = None,
    *,
    core_state: CoreSlipState,
    radius_m: float | None = None,
    n_theta: int = DEFAULT_THETA,
    n_stations: int | None = None,
    axial_strain: ArrayLike | None = None,
    core_friction: CoreFrictionSpec | None = None,
    contact_tension_N: float = 0.0,
    contact_model: str = "bounded",
    radial_state: CableRadialState | None = None,
) -> CoreLayerStrainResult:
    """Strain field of a layer inside a laid power core (gap P8).

    Parameters
    ----------
    cable : CableData
    layer_name : str
        A ``cylindrical_offset`` layer of the power cores.
    kappa_x, kappa_y : array_like
        Cable curvature history [1/m].
    core_state : {"stuck", "free", "frictional"}
        Core slip state; required (no default).
    radius_m : float or None
        Radius in the core section (default: the layer's mid radius, or the
        helix radius of a single resolved ring).
    n_theta : int
        Equally spaced angles in the core section (default 36).
    n_stations, axial_strain, core_friction, contact_tension_N,
    contact_model, radial_state
        As :func:`~cablexsec.mechanics.core_kinematics.core_kinematics`.
    """
    layer = core_layer(cable, layer_name)
    if isinstance(n_theta, bool) or not isinstance(n_theta, int) or n_theta < 1:
        raise CableInputError("n_theta must be a positive integer")
    radius = _layer_radius(layer) if radius_m is None else float(radius_m)
    kin = core_kinematics(
        cable,
        kappa_x,
        kappa_y,
        state=core_state,
        axial_strain=axial_strain,
        n_stations=n_stations,
        core_friction=core_friction,
        contact_tension_N=contact_tension_N,
        contact_model=contact_model,
        radial_state=radial_state,
    )
    theta = 2.0 * np.pi * np.arange(n_theta) / n_theta
    return CoreLayerStrainResult(
        layer_name=layer.name,
        core_state=core_state,
        radius_m=radius,
        phase_deg=np.degrees(kin.phase_rad),
        theta_deg=np.degrees(theta),
        strain=kin.strain(radius, theta),
        kinematics=kin,
        provenance=Provenance(
            method=(
                "core-layer strain: core-axis strain of the "
                f"{core_state!r} core state plus the core's own bending"
            ),
            citations=(
                "[MenardCartraud2023] Eq. 14",
                "[Papailiou1997]",
                "[Yun2020]",
            ),
            gates=(
                "stuck core: plane-section strain of the cable section",
                "free core: zero mean strain over the core section",
            ),
            limitations=(
                "layers of a core are bonded and share the core-axis strain",
                "stuck is the plane-section upper bound and free the full-slip "
                "lower bound of the core-axis strain",
            ),
        ),
    )


def laminate_strain_factor(hoop_ratio: float) -> float:
    """Equivalent-strain factor ``(2/sqrt 3) sqrt(1 + l + l^2)`` (``l`` hoop ratio)."""
    if not math.isfinite(hoop_ratio) or not -1.0 <= hoop_ratio <= 1.0:
        raise CableInputError("hoop ratio must lie in [-1, 1]")
    return 2.0 / math.sqrt(3.0) * math.sqrt(1.0 + hoop_ratio + hoop_ratio**2)


def _bonded_polymer_nu(
    cable: CableData, r_outer: float, family: object
) -> float | None:
    """Poisson ratio of the polymer layer bonded over a foil, if found."""
    for other in cable.layers:
        if other is family:
            continue
        ri = getattr(other, "r_inner_m", None)
        if ri is None or abs(ri - r_outer) > 1e-6:
            continue
        if not library.is_polymer(other.material):
            continue
        try:
            return float(library.get_property(other.material, "nu"))
        except KeyError:
            continue
    return None


@dataclass(frozen=True)
class SheathStrainLifeResult:
    """Strain-life damage map of a sheath, foil or tape.

    ``D_map`` has shape ``(station, theta)`` (one station for a concentric
    sheath); ``D_miner`` and ``n_cycles`` are those of its critical point.
    ``strain_factor`` is the strain concentration times the laminate
    equivalent-strain factor.
    """

    D_miner: float
    n_cycles: float
    strain_amplitude_max: float
    stress_max_Pa: float
    D_map: FloatArray
    phase_deg: FloatArray
    theta_deg: FloatArray
    critical_phase_deg: float
    critical_theta_deg: float
    strain_factor: float
    constants: StrainLifeConstants
    correction: str
    core_state: str | None


def strain_life_sheath_fatigue(
    cable: CableData,
    layer_name: str,
    kappa_x: ArrayLike,
    kappa_y: ArrayLike | None = None,
    *,
    strain_life: str | StrainLifeConstants | None = None,
    correction: MeanStressCorrection = "swt",
    axial_strain: ArrayLike | None = None,
    core_state: CoreSlipState | None = None,
    n_theta: int = DEFAULT_THETA,
    n_stations: int | None = None,
    core_friction: CoreFrictionSpec | None = None,
    contact_tension_N: float = 0.0,
    contact_model: str = "bounded",
    radial_state: CableRadialState | None = None,
    hoop_ratio: float | None = None,
) -> SheathStrainLifeResult:
    """Strain-life damage of a metallic sheath, foil or tape (gap P6)."""
    try:
        layer = cable.get_layer(layer_name)
    except KeyError as exc:
        raise CableSchemaError(str(exc)) from None
    if correction not in MEAN_STRESS_CORRECTIONS:
        raise CableInputError(
            f"mean-stress correction must be one of {MEAN_STRESS_CORRECTIONS}"
        )
    if isinstance(n_theta, bool) or not isinstance(n_theta, int) or n_theta < 1:
        raise CableInputError("n_theta must be a positive integer")
    key = strain_life if strain_life is not None else getattr(layer, "strain_life", "")
    if not key:
        raise CableSchemaError(
            f"layer {layer_name!r} declares no strain_life constants; pass "
            "strain_life= or set the layer's strain_life key"
        )
    constants = get_strain_life(key) if isinstance(key, str) else key
    kx = np.asarray(kappa_x, dtype=np.float64)
    ky = np.zeros_like(kx) if kappa_y is None else np.asarray(kappa_y, dtype=np.float64)
    if kx.ndim != 1 or ky.shape != kx.shape or not np.isfinite([kx, ky]).all():
        raise CableInputError("finite equal 1-D curvature histories are required")
    theta = 2.0 * np.pi * np.arange(n_theta) / n_theta
    bonded = getattr(layer, "construction", "") == "bonded_laminate"
    if isinstance(layer, CylindricalShellLayer):
        if core_state is not None:
            raise CableInputError(
                "core_state applies to a layer of a power core, not to a "
                "concentric cylindrical_shell"
            )
        assert layer.r_mid_m is not None
        concentration = 1.0 if bonded else layer.scf * layer.f_corrugation
        strain = (
            concentration
            * layer.r_mid_m
            * (
                np.cos(theta)[:, None] * kx[None, :]
                + np.sin(theta)[:, None] * ky[None, :]
            )
        )
        if axial_strain is not None:
            strain = strain + np.asarray(axial_strain, dtype=np.float64)[None, :]
        strain = strain[None, :, :]
        phase = np.zeros(1)
        r_outer = layer.r_mid_m + 0.5 * (layer.t_m or 0.0)
        state: str | None = None
    elif isinstance(layer, CylindricalOffsetLayer):
        if core_state is None:
            raise CableInputError(
                f"layer {layer_name!r} is inside a power core: give core_state "
                "('stuck', 'free' or 'frictional')"
            )
        kin = core_kinematics(
            cable,
            kx,
            ky,
            state=core_state,
            axial_strain=axial_strain,
            n_stations=n_stations,
            core_friction=core_friction,
            contact_tension_N=contact_tension_N,
            contact_model=contact_model,
            radial_state=radial_state,
        )
        strain = kin.strain(_layer_radius(layer), theta)
        phase = kin.phase_rad
        concentration = 1.0
        r_outer = float(layer.r_outer_m or 0.0)
        state = core_state
    else:
        raise CableSchemaError(
            f"layer {layer_name!r} is a {type(layer).__name__}; the strain-life "
            "route needs a cylindrical_shell or a layer of a power core"
        )
    biaxial = 1.0
    if hoop_ratio is not None:
        biaxial = laminate_strain_factor(hoop_ratio)
    elif bonded:
        found = _bonded_polymer_nu(cable, r_outer, layer)
        biaxial = laminate_strain_factor(-0.5 if found is None else -found)
    strain = strain * biaxial
    factor = concentration * biaxial
    n_st, n_th = strain.shape[0], strain.shape[1]
    life_map = strain_life_damage_map(
        strain.reshape(n_st * n_th, -1), constants, correction=correction
    )
    d_map = life_map.damage.reshape(n_st, n_th)
    amp_max = float(np.max(life_map.strain_amplitude_max))
    stress_max = float(np.max(life_map.stress_max_Pa))
    s, t = np.unravel_index(int(np.argmax(d_map)), d_map.shape)
    cycles = float(life_map.n_cycles.reshape(n_st, n_th)[s, t])
    return SheathStrainLifeResult(
        D_miner=float(d_map[s, t]),
        n_cycles=cycles,
        strain_amplitude_max=amp_max,
        stress_max_Pa=stress_max,
        D_map=d_map,
        phase_deg=np.degrees(phase),
        theta_deg=np.degrees(theta),
        critical_phase_deg=float(np.degrees(phase[s])),
        critical_theta_deg=float(np.degrees(theta[t])),
        strain_factor=factor,
        constants=constants,
        correction=correction,
        core_state=state,
    )


# Public entry points read the cable's own materials when called directly.
_scope_entry_points(globals(), __all__)
