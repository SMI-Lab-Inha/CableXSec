# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Torque balance of helical armour and twist under tension (gap 3.5).

A tensioned helical layer carries a torque about the cable axis. Each wire
of layer ``i`` (``n_i`` wires of area ``A_i`` at helix radius ``r_i`` and
lay angle ``alpha_i`` from the cable axis, lay sign ``s_i = +1`` for a
right-hand and ``-1`` for a left-hand lay) carries the wire force
``A_i sigma_i``, whose tangential component ``A_i sigma_i sin(alpha_i)``
acts at the radius ``r_i``:

.. math::

    M_i = s_i\,n_i A_i \sigma_i r_i \sin\alpha_i ,

(Knapp 1981, Eq. 10; Feyrer 2015, Eq. 2.11 with ``S_i = F_i cos(alpha_i)``).
For a fixed-radius section with the ends restrained against twist,
``sigma_i = E_i epsilon cos^2(alpha_i)``, so that the torque per unit
tension is ``K_et / K_ee`` of the Witz-Tan matrix
(:func:`cablexsec.mechanics.stiffness.axial_torsional_matrix`), which is
Feyrer's torque constant ``c_1S d_S`` (Eq. 2.74). Two contra-helical layers
are torque balanced when their torques cancel (Knapp 1981, Eqs. 22-24).

Here the layer stresses come from the coupled radial-axial-torsional section
(:func:`cablexsec.mechanics.radial_cable.solve_cable_radial_states`), the same
solved state as the contact, fatigue and thermal routes. It carries the
radial contraction of the core and the unilateral contacts between layers
(a layer that lifts off stops sharing load), laid power cores with their own
lay torque, polymer torsion and the declared manufacturing clamping, which is
identified as residual wire stress at zero tension and carried through every
state. The section is solved twice at every tension:

* **fixed twist** (both ends restrained): the end torque ``M(T)``; the
  torque/tension coefficient is ``(M(T) - M(0)) / T``;
* **free twist** (zero end torque): the twist ``tau(T)``; the twist/tension
  coefficient is ``(tau(T) - tau(0)) / T``.

The tangent coefficients ``K_et/K_ee`` and ``-K_et/(K_ee K_tt - K_et^2)`` of
each solved state are reported as well; they equal the secants while the
contact set does not change. Signs: a positive torque or twist turns in the
sense that tightens a right-hand lay; a free end turns opposite to the
fixed-end torque.

The torque-balance ratio of the armour is ``-dM_outer / sum(dM_inner)`` of
the fixed-twist torque increments: 1 when the outermost structural layer
balances the layers inside it. Laid power cores contribute their own torque
(``core_torque_fixed_Nm``), which a design must include. The free-twist
redistribution of layer stress is ``sigma_free - sigma_fixed``.

:func:`design_torque_balance` varies the lay angle, wire count or wire
diameter of one armour layer (the outermost by default) to set the
torque/tension coefficient to a target (zero by default) at a tension, or to
minimise its largest deviation over a tension range, within bounds and a
lateral fill constraint ``n d / (2 pi r cos(alpha)) <= max_fill`` (adjacent
wires of the layer touch at a fill of 1).

Excluded: bending and twisting moments of the individual wires (Costello
1997, Eqs. 3.39-3.40), which lower the torque of a seven-wire strand by
about 5% (Costello 1997, Example 3.1) and scale with ``(d / 2 r)^2``;
friction and slip; wire flattening; large-rotation geometry.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, minimize_scalar

from cablexsec.cables.schema import (
    CableData,
    CylindricalLayer,
    CylindricalShellLayer,
    HelicalLayer,
    HelicalStrandedLayer,
    Layer,
    uses_cable_materials,
)
from cablexsec.cables.schema import scope_entry_points as _scope_entry_points
from cablexsec.clamping import ClampingState, clamping_state
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.materials import library
from cablexsec.mechanics.radial_cable import (
    CableRadialState,
    solve_cable_radial_states,
)
from cablexsec.mechanics.stiffness import axial_torsional_matrix
from cablexsec.provenance import Provenance

__all__ = [
    "KNAPP_BRANCH_LIMIT_DEG",
    "FixedRadiusBalance",
    "TorqueBalanceDesign",
    "TorqueBalanceResult",
    "armour_fill_ratio",
    "design_torque_balance",
    "fixed_radius_balance",
    "torque_balance",
]

FloatArray = NDArray[np.float64]
Vary = Literal["outer_lay_angle", "outer_wire_count", "outer_wire_diameter"]

#: Lay angle [deg] of the maximum of ``sin(a) cos^2(a)``; the layer torque
#: per unit cable strain rises below it and falls above it (Knapp 1981,
#: Fig. 3: two lay angles balance a given torque).
KNAPP_BRANCH_LIMIT_DEG = math.degrees(math.atan(1.0 / math.sqrt(2.0)))

_CITATIONS = (
    "[Knapp1981] Eqs. 10, 22-24",
    "[Feyrer2015] Eqs. 2.11, 2.69-2.74",
    "[WitzTan1992a]",
    "[KimLee2017] Section 5",
)
_LIMITATIONS = (
    "linear elastic wires; bending and twisting moments of individual wires "
    "omitted (about -5% torque for a seven-wire strand, Costello 1997 "
    "Example 3.1; scales with (d/2r)^2, negligible for armour wires)",
    "frictionless unilateral radial contact without slip, wire flattening "
    "or lateral (hoop) wire contact; small rotations",
    "loose (non-structural) core screens follow loose_layers; the default "
    "'neighbour' meshes them with the material of the softer structural "
    "neighbour, as the fixed-radius stiffness gives them no axial load",
    "lay directions not stated by a source are assumptions of the cable "
    "definition and decide the sign of every torque",
)


# --------------------------------------------------------------------------- #
# Layer helpers                                                               #
# --------------------------------------------------------------------------- #
def _lay_sign(layer: HelicalLayer) -> float:
    return 1.0 if layer.lay_direction == "right_hand" else -1.0


def _angle(layer: HelicalLayer) -> float:
    return abs(float(layer.lay_angle_deg or 0.0))


def _torque_per_stress(layer: HelicalLayer) -> float:
    """Axial torque of the layer per unit wire stress [N m / Pa]."""
    assert layer.n_wires and layer.d_wire_m and layer.r_helix_m
    area = math.pi * layer.d_wire_m**2 / 4.0
    return (
        _lay_sign(layer)
        * layer.n_wires
        * area
        * layer.r_helix_m
        * math.sin(math.radians(_angle(layer)))
    )


def armour_fill_ratio(
    n_wires: float, d_wire_m: float, r_helix_m: float, lay_angle_deg: float
) -> float:
    """Lateral fill ``n d / (2 pi r cos(alpha))`` of a helical layer.

    The circumferential pitch of adjacent wires normal to their axes is
    ``2 pi r cos(alpha) / n``; the wires touch at a fill of 1 (small
    ``d / r``; Kim and Lee 2017, Section 5.1, Step 1).
    """
    values = (n_wires, d_wire_m, r_helix_m)
    if not all(math.isfinite(v) and v > 0 for v in values) or not (
        0.0 <= abs(lay_angle_deg) < 90.0
    ):
        raise CableInputError("fill ratio needs positive geometry and |angle| < 90")
    return (
        n_wires
        * d_wire_m
        / (2.0 * math.pi * r_helix_m * math.cos(math.radians(lay_angle_deg)))
    )


def _structural_armour(cable: CableData) -> list[HelicalLayer]:
    layers = [x for x in cable.helical_layers if x.structural]
    return sorted(layers, key=lambda x: float(x.r_helix_m or 0.0))


# --------------------------------------------------------------------------- #
# Result                                                                      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TorqueBalanceResult:
    """Torque and twist of a cable under tension at fixed and free twist.

    Arrays are indexed like ``tensions_N``, whose first entry is the
    zero-tension reference. Torques are about the cable axis [N m]; twist is
    per unit length [rad/m]; layer quantities are keyed by layer name.

    Attributes
    ----------
    tensions_N : numpy.ndarray
        True wall tensions, starting with 0.
    torque_fixed_Nm : numpy.ndarray
        End torque with the twist restrained, including the residual torque
        of the manufacturing clamping at zero tension.
    twist_free_rad_per_m : numpy.ndarray
        Twist of a free end relative to the free state at zero tension.
    torque_per_tension_m, twist_per_tension_rad_per_N_m : numpy.ndarray
        Secant coefficients from zero tension (NaN at zero tension).
    torque_per_tension_tangent_m, twist_per_tension_tangent_rad_per_N_m
        Tangent coefficients of each solved state.
    ea_fixed_N, ea_free_N, gj_N_m2 : numpy.ndarray
        Tangent axial rigidity at fixed and free twist and the torsional
        rigidity at fixed extension, all on the solved contact set.
    layer_torque_fixed_Nm : dict[str, numpy.ndarray]
        Torque increment of each helical layer at fixed twist.
    core_torque_fixed_Nm : numpy.ndarray
        Remaining torque increment at fixed twist (laid cores, other layers).
    wire_stress_fixed_Pa, wire_stress_free_Pa : dict[str, numpy.ndarray]
        Wire stress of each helical layer (including the residual stress).
    balance_ratio : numpy.ndarray
        ``-dM_outer / sum(dM_inner)`` of the structural armour at fixed twist.
    fixed_radius : dict[str, float]
        The fixed-radius Witz-Tan values used by :func:`cablexsec.xsec_stiffness`.
    open_interfaces_free : tuple[str, ...]
        Interfaces ``inner/outer`` without contact force at the largest
        tension with a free end (open, or closed without load).
    """

    cable_name: str
    tensions_N: FloatArray
    torque_fixed_Nm: FloatArray
    twist_free_rad_per_m: FloatArray
    axial_strain_fixed: FloatArray
    axial_strain_free: FloatArray
    torque_per_tension_m: FloatArray
    torque_per_tension_tangent_m: FloatArray
    twist_per_tension_rad_per_N_m: FloatArray
    twist_per_tension_tangent_rad_per_N_m: FloatArray
    ea_fixed_N: FloatArray
    ea_free_N: FloatArray
    gj_N_m2: FloatArray
    layer_torque_fixed_Nm: Mapping[str, FloatArray]
    core_torque_fixed_Nm: FloatArray
    wire_stress_fixed_Pa: Mapping[str, FloatArray]
    wire_stress_free_Pa: Mapping[str, FloatArray]
    outer_layer: str | None
    inner_layers: tuple[str, ...]
    balance_ratio: FloatArray
    residual_torque_Nm: float
    fixed_radius: Mapping[str, float]
    active_contacts_fixed: NDArray[np.int64]
    active_contacts_free: NDArray[np.int64]
    open_interfaces_free: tuple[str, ...]
    clamping: ClampingState
    loose_layers: str
    notes: tuple[str, ...] = field(default=())
    provenance: Provenance | None = None

    # ------------------------------------------------------------------ #
    @property
    def torque_coefficient_m(self) -> float:
        """Secant torque/tension at fixed twist at the largest tension [m]."""
        return float(self.torque_per_tension_m[-1])

    @property
    def twist_coefficient_rad_per_N_m(self) -> float:
        """Secant twist/tension of a free end at the largest tension."""
        return float(self.twist_per_tension_rad_per_N_m[-1])

    @property
    def twist_per_100kN_deg_per_m(self) -> float:
        """Free-end twist per 100 kN of tension [deg/m] (secant)."""
        return math.degrees(self.twist_coefficient_rad_per_N_m) * 1.0e5

    @property
    def stress_redistribution_Pa(self) -> dict[str, FloatArray]:
        """``sigma_free - sigma_fixed`` of each helical layer [Pa]."""
        return {
            name: self.wire_stress_free_Pa[name] - self.wire_stress_fixed_Pa[name]
            for name in self.wire_stress_fixed_Pa
        }

    @property
    def stress_increment_ratio(self) -> dict[str, FloatArray]:
        """Free over fixed wire-stress increment from zero tension per layer."""
        out: dict[str, FloatArray] = {}
        for name, fixed in self.wire_stress_fixed_Pa.items():
            free = self.wire_stress_free_Pa[name]
            d_fixed = fixed - fixed[0]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(d_fixed != 0.0, (free - free[0]) / d_fixed, np.nan)
            out[name] = np.asarray(ratio, dtype=np.float64)
        return out

    def summary(self) -> str:
        """Plain-text table of the result at every tension."""
        torque_mm = self.fixed_radius["torque_per_tension_m"] * 1e3
        twist = self.fixed_radius["twist_per_tension_rad_per_N_m"]
        lines = [
            f"Torque balance of {self.cable_name!r}",
            f"  fixed-radius torque/tension {torque_mm:.4f} mm; "
            f"twist/tension {twist:.4e} rad/(N m)",
            "  residual torque at zero tension (fixed twist) "
            f"{self.residual_torque_Nm:.4g} N m",
            "  T [kN]  M_fixed [N m]  M/T [mm]  twist [deg/m]  ratio",
        ]
        for k, t in enumerate(self.tensions_N):
            lines.append(
                f"  {t / 1e3:7.1f}  {self.torque_fixed_Nm[k]:13.4g}  "
                f"{self.torque_per_tension_m[k] * 1e3:8.4f}  "
                f"{math.degrees(self.twist_free_rad_per_m[k]):13.5f}  "
                f"{self.balance_ratio[k]:6.3f}"
            )
        for name in self.wire_stress_fixed_Pa:
            fixed = self.wire_stress_fixed_Pa[name][-1] / 1e6
            free = self.wire_stress_free_Pa[name][-1] / 1e6
            lines.append(
                f"  {name}: sigma fixed {fixed:.4g} MPa, free {free:.4g} MPa "
                f"at {self.tensions_N[-1] / 1e3:g} kN"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Analysis                                                                    #
# --------------------------------------------------------------------------- #
def _tension_grid(
    tension_range: tuple[float, float] | None,
    n_tensions: int,
    tensions_N: Sequence[float] | None,
) -> FloatArray:
    if tensions_N is not None:
        values = np.asarray([float(v) for v in tensions_N], dtype=np.float64)
    else:
        if tension_range is None or len(tension_range) != 2:
            raise CableInputError("tension_range must be a (minimum, maximum) pair")
        lo, hi = float(tension_range[0]), float(tension_range[1])
        if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 <= lo < hi):
            raise CableInputError("tension_range needs 0 <= minimum < maximum")
        if n_tensions < 2:
            raise CableInputError("n_tensions must be at least 2")
        values = np.linspace(lo, hi, int(n_tensions))
    if values.size == 0 or not np.isfinite(values).all() or np.any(values < 0):
        raise CableInputError("tensions must be finite and non-negative")
    positive = np.unique(values[values > 0.0])
    if positive.size == 0:
        raise CableInputError("at least one positive tension is required")
    return np.concatenate(([0.0], positive))


def _solver_options(
    *,
    temperature_C: float | Mapping[str, float] | None,
    T_ref_C: float,
    external_pressure_Pa: float | None,
    flooded_pressure_Pa: float,
    include_manufacturing: bool,
    loose_layers: str,
    filler: str,
    angular_cells: int,
    youngs_modulus_Pa: Mapping[str, float] | None,
) -> dict[str, object]:
    if loose_layers not in {"declared", "neighbour"}:
        raise CableInputError("loose_layers must be 'declared' or 'neighbour'")
    return {
        "temperature_C": temperature_C,
        "T_ref_C": T_ref_C,
        "external_pressure_Pa": external_pressure_Pa,
        "flooded_pressure_Pa": flooded_pressure_Pa,
        "include_manufacturing": include_manufacturing,
        "loose_layers": loose_layers,
        "filler": filler,
        "angular_cells": angular_cells,
        "youngs_modulus_Pa": youngs_modulus_Pa,
    }


def _solve(
    cable: CableData, tensions: FloatArray, free: bool, options: Mapping[str, object]
) -> list[CableRadialState]:
    return solve_cable_radial_states(
        cable,
        tensions,
        prescribed_twist_per_m=None if free else 0.0,
        torque_Nm=0.0,
        **options,  # type: ignore[arg-type]
    )


def _require_complete(cable: CableData) -> None:
    missing = cable.missing_fields()
    if missing:
        raise CableSchemaError(
            f"torque_balance: cable {cable.cable.name!r} is not fully specified "
            f"(missing {', '.join(missing[:6])}); supply the geometry override"
        )
    if any(isinstance(x, HelicalStrandedLayer) for x in cable.layers):
        raise CableSchemaError(
            "torque_balance: nested helical (stranded) layers need a strand law"
        )
    if not cable.helical_layers:
        raise CableSchemaError("torque_balance: the cable has no helical layer")


def _fixed_radius(cable: CableData) -> dict[str, float]:
    at = axial_torsional_matrix(cable)
    det = at.K_ee * at.K_tt - at.K_et**2
    return {
        "K_ee_N": at.K_ee,
        "K_et_Nm": at.K_et,
        "K_tt_Nm2": at.K_tt,
        "torque_per_tension_m": at.K_et / at.K_ee if at.K_ee > 0 else math.nan,
        "twist_per_tension_rad_per_N_m": -at.K_et / det if det > 0 else math.nan,
    }


@dataclass(frozen=True)
class FixedRadiusBalance:
    """Torque balance of the armour in the fixed-radius assembly (no contact).

    The quick form of :func:`torque_balance` for a model check: the wire
    stress at fixed twist is ``E epsilon cos^2(alpha)`` (module notes), so
    each layer's torque per unit cable strain is its share of ``K_et`` of
    the Witz-Tan matrix.

    Attributes
    ----------
    layer_torque_per_strain_Nm : dict[str, float]
        Torque per unit strain of each structural helical layer at fixed
        twist [N m], signed by the lay (right-hand positive).
    outer_layer, inner_layers
        The outermost structural helical layer and those inside it.
    balance_ratio : float
        ``-dM_outer / sum(dM_inner)``, the ratio of :func:`torque_balance`
        at fixed radius: 1 when the outer layer balances the layers inside
        it; ``NaN`` with fewer than two layers.
    torque_per_tension_m, twist_per_tension_rad_per_N_m : float
        ``K_et / K_ee`` and the free-end twist per tension of the whole
        section (laid cores included).
    """

    layer_torque_per_strain_Nm: Mapping[str, float]
    outer_layer: str | None
    inner_layers: tuple[str, ...]
    balance_ratio: float
    torque_per_tension_m: float
    twist_per_tension_rad_per_N_m: float


@uses_cable_materials
def fixed_radius_balance(cable: CableData) -> FixedRadiusBalance:
    """Torque balance of ``cable``'s armour at fixed radius (no solve).

    Needs a fully specified cable whose helical layers have a Young's
    modulus; for a model check, where :func:`torque_balance` (two solved
    sections at every tension) would be too slow.
    """
    _require_complete(cable)
    armour = _structural_armour(cable)
    torques = {
        layer.name: _torque_per_stress(layer)
        * library.get_property(layer.material, "E_Pa")
        * math.cos(math.radians(_angle(layer))) ** 2
        for layer in armour
    }
    outer = armour[-1].name if armour else None
    inner = tuple(layer.name for layer in armour[:-1])
    denominator = sum(torques[name] for name in inner)
    ratio = (
        -torques[outer] / denominator
        if outer is not None and inner and denominator != 0.0
        else math.nan
    )
    whole = _fixed_radius(cable)
    return FixedRadiusBalance(
        layer_torque_per_strain_Nm=torques,
        outer_layer=outer,
        inner_layers=inner,
        balance_ratio=float(ratio),
        torque_per_tension_m=whole["torque_per_tension_m"],
        twist_per_tension_rad_per_N_m=whole["twist_per_tension_rad_per_N_m"],
    )


def torque_balance(
    cable: CableData,
    tension_range: tuple[float, float] | None = (0.0, 100.0e3),
    *,
    n_tensions: int = 5,
    tensions_N: Sequence[float] | None = None,
    temperature_C: float | Mapping[str, float] | None = None,
    T_ref_C: float = 20.0,
    external_pressure_Pa: float | None = None,
    flooded_pressure_Pa: float = 0.0,
    include_manufacturing: bool = True,
    loose_layers: str = "neighbour",
    filler: str = "auto",
    angular_cells: int = 48,
    youngs_modulus_Pa: Mapping[str, float] | None = None,
) -> TorqueBalanceResult:
    """Torque/tension and twist/tension of a cable at fixed and free twist.

    Parameters
    ----------
    cable : CableData
        A fully specified cable with at least one helical layer.
    tension_range : (float, float), optional
        Minimum and maximum true wall tension [N], sampled at
        ``n_tensions`` points. Zero tension is always added as the
        reference state.
    tensions_N : sequence of float, optional
        Explicit tensions instead of ``tension_range``.
    temperature_C, T_ref_C, external_pressure_Pa, flooded_pressure_Pa
        Load state, as :func:`cablexsec.solve_cable_radial_state`. The
        default external pressure is the cable's declared initial state.
    include_manufacturing : bool, optional
        Carry the declared manufacturing clamping (residual wire stress).
    loose_layers : {"neighbour", "declared"}, optional
        Representation of loose core screens in the solved section (see
        :func:`cablexsec.mechanics.radial_cable.build_radial_section`). The
        default meshes them with the material of their softer structural
        neighbour, as the fixed-radius stiffness gives them no axial load.
    filler, angular_cells, youngs_modulus_Pa
        Passed to the solved section.

    Returns
    -------
    TorqueBalanceResult
    """
    _require_complete(cable)
    tensions = _tension_grid(
        tension_range if tensions_N is None else None, n_tensions, tensions_N
    )
    options = _solver_options(
        temperature_C=temperature_C,
        T_ref_C=T_ref_C,
        external_pressure_Pa=external_pressure_Pa,
        flooded_pressure_Pa=flooded_pressure_Pa,
        include_manufacturing=include_manufacturing,
        loose_layers=loose_layers,
        filler=filler,
        angular_cells=angular_cells,
        youngs_modulus_Pa=youngs_modulus_Pa,
    )
    fixed = _solve(cable, tensions, False, options)
    free = _solve(cable, tensions, True, options)
    return _assemble(cable, tensions, fixed, free, loose_layers)


def _assemble(
    cable: CableData,
    tensions: FloatArray,
    fixed: Sequence[CableRadialState],
    free: Sequence[CableRadialState],
    loose_layers: str,
) -> TorqueBalanceResult:
    torque = np.array([s.result.torque_Nm for s in fixed])
    twist_abs = np.array([s.result.twist_per_m for s in free])
    twist = twist_abs - twist_abs[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        secant_m = np.where(tensions > 0, (torque - torque[0]) / tensions, np.nan)
        secant_t = np.where(tensions > 0, twist / tensions, np.nan)
    tan_fixed = np.array([s.result.tangent for s in fixed])
    tan_free = np.array([s.result.tangent for s in free])
    det_free = tan_free[:, 0, 0] * tan_free[:, 1, 1] - tan_free[:, 0, 1] ** 2
    helical = list(cable.helical_layers)
    stress_fixed = {
        x.name: np.array([s.result.wire_stress_Pa[x.name] for s in fixed])
        for x in helical
    }
    stress_free = {
        x.name: np.array([s.result.wire_stress_Pa[x.name] for s in free])
        for x in helical
    }
    layer_torque = {
        x.name: _torque_per_stress(x) * (stress_fixed[x.name] - stress_fixed[x.name][0])
        for x in helical
    }
    core_torque = (torque - torque[0]) - sum(
        layer_torque.values(), np.zeros_like(torque)
    )
    armour = _structural_armour(cable)
    outer = armour[-1].name if armour else None
    inner = tuple(x.name for x in armour[:-1])
    if outer is not None and inner:
        denominator = sum((layer_torque[n] for n in inner), np.zeros_like(torque))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                denominator != 0.0, -layer_torque[outer] / denominator, np.nan
            )
    else:
        ratio = np.full_like(torque, np.nan)
    last = free[-1].result
    open_free = tuple(
        f"{a}/{b}"
        for (a, b), force in zip(
            last.interface_names, last.interface_line_force_N_per_m, strict=True
        )
        if force <= 0.0
    )
    notes = list(fixed[0].notes)
    if open_free:
        notes.append(
            "interfaces without contact force at the largest tension with a "
            "free end: " + ", ".join(open_free)
        )
    state = clamping_state(cable)
    notes.append("manufacturing clamping: " + state.describe())
    return TorqueBalanceResult(
        cable_name=cable.cable.name,
        tensions_N=tensions,
        torque_fixed_Nm=torque,
        twist_free_rad_per_m=twist,
        axial_strain_fixed=np.array([s.result.axial_strain for s in fixed]),
        axial_strain_free=np.array([s.result.axial_strain for s in free]),
        torque_per_tension_m=np.asarray(secant_m, dtype=np.float64),
        torque_per_tension_tangent_m=tan_fixed[:, 0, 1] / tan_fixed[:, 0, 0],
        twist_per_tension_rad_per_N_m=np.asarray(secant_t, dtype=np.float64),
        twist_per_tension_tangent_rad_per_N_m=-tan_free[:, 0, 1] / det_free,
        ea_fixed_N=tan_fixed[:, 0, 0],
        ea_free_N=tan_free[:, 0, 0] - tan_free[:, 0, 1] ** 2 / tan_free[:, 1, 1],
        gj_N_m2=tan_fixed[:, 1, 1],
        layer_torque_fixed_Nm=layer_torque,
        core_torque_fixed_Nm=np.asarray(core_torque, dtype=np.float64),
        wire_stress_fixed_Pa=stress_fixed,
        wire_stress_free_Pa=stress_free,
        outer_layer=outer,
        inner_layers=inner,
        balance_ratio=np.asarray(ratio, dtype=np.float64),
        residual_torque_Nm=float(torque[0]),
        fixed_radius=_fixed_radius(cable),
        active_contacts_fixed=np.array(
            [s.result.active_contacts for s in fixed], dtype=np.int64
        ),
        active_contacts_free=np.array(
            [s.result.active_contacts for s in free], dtype=np.int64
        ),
        open_interfaces_free=open_free,
        clamping=state,
        loose_layers=loose_layers,
        notes=tuple(notes),
        provenance=Provenance(
            method="layer torques from the coupled radial-axial-torsional "
            "section solved at fixed and at free twist (secant and tangent "
            "coefficients)",
            citations=_CITATIONS,
            gates=(
                "TQ1 Feyrer Ex. 2.8 closed form",
                "TQ2 Knapp core factor (coupled section)",
                "TQ3 Kim-Lee two-layer balance angles",
                "TQ4 Utting-Jones fixed-end torque",
                "TQ5 mirror lay symmetry",
                "TQ6 xsec_stiffness consistency",
            ),
            limitations=_LIMITATIONS,
        ),
    )


# --------------------------------------------------------------------------- #
# Design helper                                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TorqueBalanceDesign:
    """Armour layer parameter that balances the torque of a cable.

    ``value`` is in the natural unit of ``vary``: degrees for the lay angle,
    a count for the wire count, metres for the wire diameter. ``coefficient_m``
    is the achieved torque/tension coefficient at ``tension_N`` (the largest
    deviation from ``target_m`` over the tensions for a range objective).
    ``cable`` is the modified definition and ``result`` its full torque
    balance. ``converged`` is False when no admissible value meets the
    target; ``value`` is then the admissible value closest to it.
    """

    vary: str
    layer: str
    value: float
    initial_value: float
    target_m: float
    tensions_N: tuple[float, ...]
    objective: str
    coefficient_m: float
    initial_coefficient_m: float
    converged: bool
    message: str
    bounds: tuple[float, float]
    fill_ratio: float
    lateral_gap_m: float
    max_fill: float
    max_wire_stress_Pa: float
    stress_limit_Pa: float | None
    stress_ok: bool | None
    history: tuple[tuple[float, float], ...]
    cable: CableData
    result: TorqueBalanceResult
    continuous_value: float | None = None
    provenance: Provenance | None = None


def _outside(layer: Layer, radius: float) -> bool:
    tol = 1e-9
    if isinstance(layer, HelicalLayer):
        depth = layer.wire_radial_thickness_m if layer.d_wire_m else 0.0
        return float(layer.r_helix_m or 0.0) - float(depth) / 2 >= (radius - tol)
    if isinstance(layer, CylindricalShellLayer):
        return float(layer.r_mid_m or 0.0) - float(layer.t_m or 0.0) / 2 >= radius - tol
    if isinstance(layer, CylindricalLayer):
        inner = float(layer.r_inner_m or 0.0)
        outer = float(layer.r_outer_m or 0.0)
        if inner < radius - tol < outer - 2 * tol:
            raise CableSchemaError(
                f"{layer.name}: layer straddles the armour; cannot shift it"
            )
        return inner >= radius - tol
    return False


def _shift(layer: Layer, delta: float) -> Layer:
    if isinstance(layer, HelicalLayer):
        return layer.model_copy(
            update={"r_helix_m": float(layer.r_helix_m or 0) + delta}
        )
    if isinstance(layer, CylindricalShellLayer):
        return layer.model_copy(update={"r_mid_m": float(layer.r_mid_m or 0) + delta})
    if isinstance(layer, CylindricalLayer):
        return layer.model_copy(
            update={
                "r_inner_m": float(layer.r_inner_m or 0) + delta,
                "r_outer_m": float(layer.r_outer_m or 0) + delta,
            }
        )
    raise CableSchemaError("only concentric layers can be shifted")


def _with_parameter(
    cable: CableData, layer_name: str, vary: str, value: float
) -> CableData:
    """Copy of ``cable`` with the parameter of one helical layer changed.

    A changed wire diameter keeps the layer's inner face and moves every
    concentric layer outside it radially by the change of diameter.
    """
    layer = cable.get_layer(layer_name)
    assert isinstance(layer, HelicalLayer)
    layers = list(cable.layers)
    index = layers.index(layer)
    if vary == "outer_lay_angle":
        sign = -1.0 if float(layer.lay_angle_deg or 0.0) < 0 else 1.0
        layers[index] = layer.model_copy(update={"lay_angle_deg": sign * value})
    elif vary == "outer_wire_count":
        layers[index] = layer.model_copy(update={"n_wires": int(round(value))})
    else:
        d0 = float(layer.d_wire_m or 0.0)
        r0 = float(layer.r_helix_m or 0.0)
        delta = value - d0
        face = r0 + d0 / 2
        for j, other in enumerate(layers):
            if j != index and other.name != layer_name and _outside(other, face):
                layers[j] = _shift(other, delta)
        layers[index] = layer.model_copy(
            update={"d_wire_m": value, "r_helix_m": r0 + delta / 2}
        )
    return cable.model_copy(update={"layers": tuple(layers)})


def _fill_bounds(
    layer: HelicalLayer, vary: str, max_fill: float, min_fill: float
) -> tuple[float, float]:
    """Admissible interval of the varied parameter from the fill limits."""
    assert layer.n_wires and layer.d_wire_m and layer.r_helix_m
    n, d, r = float(layer.n_wires), float(layer.d_wire_m), float(layer.r_helix_m)
    c = math.cos(math.radians(_angle(layer)))
    if vary == "outer_lay_angle":
        x = n * d / (2 * math.pi * r * max_fill)
        hi = math.degrees(math.acos(x)) if x <= 1.0 else -math.inf
        lo = 0.0
        if min_fill > 0.0:
            y = n * d / (2 * math.pi * r * min_fill)
            lo = math.degrees(math.acos(min(1.0, y)))
        return lo, hi
    if vary == "outer_wire_count":
        room = 2 * math.pi * r * c / d
        hi_n = (
            math.floor(room * max_fill + 1e-9) if math.isfinite(max_fill) else math.inf
        )
        lo_n = math.ceil(room * min_fill - 1e-9) if min_fill else 1
        return float(max(lo_n, 1)), float(hi_n)
    r_in = r - d / 2

    def limit(fill: float) -> float:
        k = math.pi * c * fill
        return 2 * k * r_in / (n - k) if n > k else math.inf

    return (limit(min_fill) if min_fill > 0.0 else 0.0), limit(max_fill)


def _default_bounds(layer: HelicalLayer, vary: str) -> tuple[float, float]:
    if vary == "outer_lay_angle":
        return 1.0, KNAPP_BRANCH_LIMIT_DEG
    if vary == "outer_wire_count":
        count = float(layer.n_wires or 0)
        return max(1.0, math.ceil(count / 4)), 4.0 * count
    d = float(layer.d_wire_m or 0.0)
    return 0.5 * d, 2.0 * d


def _parameter(layer: HelicalLayer, vary: str) -> float:
    if vary == "outer_lay_angle":
        return _angle(layer)
    if vary == "outer_wire_count":
        return float(layer.n_wires or 0)
    return float(layer.d_wire_m or 0.0)


def _integer_root(
    f: Callable[[float], float], lo: int, hi: int, f_lo: float, f_hi: float
) -> tuple[int, float, float]:
    """Bracketing integer bisection; returns the best integer and its value."""
    while hi - lo > 1:
        mid = (lo + hi) // 2
        f_mid = f(float(mid))
        if (f_mid > 0) == (f_lo > 0):
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
    continuous = lo + f_lo / (f_lo - f_hi) if f_lo != f_hi else float(lo)
    best = lo if abs(f_lo) <= abs(f_hi) else hi
    return best, (f_lo if best == lo else f_hi), continuous


def design_torque_balance(
    cable: CableData,
    vary: Vary = "outer_lay_angle",
    *,
    target: float = 0.0,
    tension_N: float | None = None,
    tension_range: tuple[float, float] | None = None,
    n_tensions: int = 5,
    bounds: tuple[float, float] | None = None,
    layer: str | None = None,
    max_fill: float = 1.0,
    min_fill: float = 0.0,
    stress_limit_Pa: float | None = None,
    xtol: float | None = None,
    temperature_C: float | Mapping[str, float] | None = None,
    T_ref_C: float = 20.0,
    external_pressure_Pa: float | None = None,
    flooded_pressure_Pa: float = 0.0,
    include_manufacturing: bool = True,
    loose_layers: str = "neighbour",
    filler: str = "auto",
    angular_cells: int = 48,
    youngs_modulus_Pa: Mapping[str, float] | None = None,
) -> TorqueBalanceDesign:
    """Armour parameter that zeroes (or minimises) the torque coefficient.

    Parameters
    ----------
    cable : CableData
        A fully specified cable with at least two structural helical layers
        (or one, if the laid cores supply the opposing torque).
    vary : {"outer_lay_angle", "outer_wire_count", "outer_wire_diameter"}
        Parameter of ``layer`` (default: the outermost structural helical
        layer). A changed diameter keeps the layer's inner face and moves
        the layers outside it.
    target : float, optional
        Target torque/tension coefficient at fixed twist [m] (default 0).
    tension_N : float, optional
        Tension [N] at which the target is met (default 100 kN).
    tension_range : (float, float), optional
        Minimise instead the largest deviation from ``target`` over
        ``n_tensions`` tensions of this range.
    bounds : (float, float), optional
        Search interval in the unit of ``vary`` (deg, count, m). Defaults:
        1 deg to :data:`KNAPP_BRANCH_LIMIT_DEG` (the lower torque branch),
        a quarter to four times the wire count, and half to twice the wire
        diameter. Always intersected with the fill limits.
    max_fill, min_fill : float, optional
        Lateral fill limits ``n d / (2 pi r cos(alpha))`` of the layer;
        adjacent wires touch at 1.
    stress_limit_Pa : float, optional
        Wire-stress limit checked (not optimised) at the largest tension at
        fixed and free twist, as the stress check of Kim and Lee (2017).

    The declared manufacturing clamping of the varied layer is kept
    unchanged. The other keywords are those of :func:`torque_balance`.

    Returns
    -------
    TorqueBalanceDesign
    """
    if vary not in ("outer_lay_angle", "outer_wire_count", "outer_wire_diameter"):
        raise CableInputError(
            "vary must be 'outer_lay_angle', 'outer_wire_count' or "
            "'outer_wire_diameter'"
        )
    if not (math.isfinite(target) and max_fill > 0.0 and 0.0 <= min_fill < max_fill):
        raise CableInputError("target must be finite and 0 <= min_fill < max_fill")
    _require_complete(cable)
    armour = _structural_armour(cable)
    if layer is None:
        if not armour:
            raise CableSchemaError("the cable has no structural helical layer")
        layer = armour[-1].name
    chosen = cable.get_layer(layer)
    if not isinstance(chosen, HelicalLayer):
        raise CableSchemaError(f"{layer!r} is not a helical layer")
    if any(x.wire_shape != "round" for x in cable.helical_layers):
        raise CableSchemaError(
            "the torque-balance design varies round wires; rectangular or "
            "profiled wires (gap G1) are not supported by it"
        )
    if tension_range is not None:
        grid = _tension_grid(tension_range, n_tensions, None)[1:]
        objective = "minimax"
    else:
        t = 100.0e3 if tension_N is None else float(tension_N)
        if not (math.isfinite(t) and t > 0):
            raise CableInputError("tension_N must be finite and positive")
        grid = np.array([t])
        objective = "zero"
    options = _solver_options(
        temperature_C=temperature_C,
        T_ref_C=T_ref_C,
        external_pressure_Pa=external_pressure_Pa,
        flooded_pressure_Pa=flooded_pressure_Pa,
        include_manufacturing=include_manufacturing,
        loose_layers=loose_layers,
        filler=filler,
        angular_cells=angular_cells,
        youngs_modulus_Pa=youngs_modulus_Pa,
    )
    lo_b, hi_b = _default_bounds(chosen, vary) if bounds is None else bounds
    lo_f, hi_f = _fill_bounds(chosen, vary, max_fill, min_fill)
    lo, hi = max(float(lo_b), lo_f), min(float(hi_b), hi_f)
    if vary == "outer_wire_count":
        lo, hi = float(math.ceil(lo)), float(math.floor(hi))
    if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi):
        raise CableInputError(
            f"no admissible {vary} within bounds ({lo_b:g}, {hi_b:g}) and the "
            f"fill limits ({min_fill:g}, {max_fill:g})"
        )
    if vary == "outer_lay_angle" and hi >= 90.0:
        raise CableInputError("lay angle bounds must lie below 90 deg")
    history: list[tuple[float, float]] = []
    cache: dict[float, FloatArray] = {}

    def coefficients(x: float) -> FloatArray:
        key = round(x, 12)
        if key not in cache:
            trial = _with_parameter(cable, layer, vary, x)
            states = _solve(trial, np.concatenate(([0.0], grid)), False, options)
            m0 = states[0].result.torque_Nm
            cache[key] = np.array(
                [
                    (s.result.torque_Nm - m0) / t
                    for s, t in zip(states[1:], grid, strict=True)
                ]
            )
        return cache[key]

    def deviation(x: float) -> float:
        c = coefficients(x) - target
        value = float(c[0]) if objective == "zero" else float(np.max(np.abs(c)))
        history.append((x, value))
        return value

    x0 = _parameter(chosen, vary)
    c_initial = float(coefficients(x0)[np.argmax(np.abs(coefficients(x0) - target))])
    continuous: float | None = None
    if objective == "zero":
        f_lo, f_hi = deviation(lo), deviation(hi)
        if f_lo == 0.0 or f_hi == 0.0:
            x_best, converged = (lo if f_lo == 0.0 else hi), True
            message = "target met at a bound"
        elif (f_lo > 0) != (f_hi > 0):
            if vary == "outer_wire_count":
                best, _, continuous = _integer_root(
                    deviation, int(lo), int(hi), f_lo, f_hi
                )
                x_best, converged = float(best), True
                message = "nearest whole wire count to the zero"
            else:
                tol = xtol or (1e-5 if vary == "outer_lay_angle" else 1e-8)
                x_best = float(brentq(deviation, lo, hi, xtol=tol))
                converged, message = True, "zero of the torque coefficient found"
        else:
            x_best = lo if abs(f_lo) <= abs(f_hi) else hi
            converged = False
            message = (
                "no zero of the torque coefficient within the admissible "
                "interval; the closest bound is returned"
            )
    else:
        if vary == "outer_wire_count":
            values = {
                n: abs(deviation(float(n)))
                for n in _integer_candidates(deviation, int(lo), int(hi))
            }
            x_best = float(min(values, key=lambda n: values[n]))
        else:
            tol = xtol or (1e-4 if vary == "outer_lay_angle" else 1e-7)
            found = minimize_scalar(
                lambda x: abs(deviation(float(x))),
                bounds=(lo, hi),
                method="bounded",
                options={"xatol": tol},
            )
            x_best = float(found.x)
        converged, message = True, "largest deviation over the tensions minimised"
    design = _with_parameter(cable, layer, vary, x_best)
    result = torque_balance(
        design,
        tensions_N=list(grid),
        temperature_C=temperature_C,
        T_ref_C=T_ref_C,
        external_pressure_Pa=external_pressure_Pa,
        flooded_pressure_Pa=flooded_pressure_Pa,
        include_manufacturing=include_manufacturing,
        loose_layers=loose_layers,
        filler=filler,
        angular_cells=angular_cells,
        youngs_modulus_Pa=youngs_modulus_Pa,
    )
    final = coefficients(x_best)
    achieved = (
        float(final[0])
        if objective == "zero"
        else float(final[np.argmax(np.abs(final - target))])
    )
    varied = design.get_layer(layer)
    assert isinstance(varied, HelicalLayer)
    assert varied.n_wires and varied.d_wire_m and varied.r_helix_m
    fill = armour_fill_ratio(
        varied.n_wires, varied.d_wire_m, varied.r_helix_m, _angle(varied)
    )
    gap = (
        2
        * math.pi
        * varied.r_helix_m
        * math.cos(math.radians(_angle(varied)))
        / varied.n_wires
        - varied.d_wire_m
    )
    peak = max(
        float(np.max(np.abs(s[1:])))
        for s in (
            *result.wire_stress_fixed_Pa.values(),
            *result.wire_stress_free_Pa.values(),
        )
    )
    return TorqueBalanceDesign(
        vary=vary,
        layer=layer,
        value=x_best,
        initial_value=x0,
        target_m=float(target),
        tensions_N=tuple(float(t) for t in grid),
        objective=objective,
        coefficient_m=achieved,
        initial_coefficient_m=c_initial,
        converged=converged,
        message=message,
        bounds=(lo, hi),
        fill_ratio=fill,
        lateral_gap_m=gap,
        max_fill=max_fill,
        max_wire_stress_Pa=peak,
        stress_limit_Pa=stress_limit_Pa,
        stress_ok=None if stress_limit_Pa is None else peak <= stress_limit_Pa,
        history=tuple(history),
        cable=design,
        result=result,
        continuous_value=continuous,
        provenance=Provenance(
            method=f"{vary} of {layer!r} for a torque/tension coefficient of "
            f"{target:g} m at fixed twist ({objective}) on the coupled section",
            citations=_CITATIONS,
            gates=("TQ3 Kim-Lee two-layer balance angles", "TQ7 design round trip"),
            limitations=(
                *_LIMITATIONS,
                "the declared manufacturing clamping of the varied layer is "
                "kept unchanged; bending stiffness is not part of the objective",
            ),
        ),
    )


def _integer_candidates(f: Callable[[float], float], lo: int, hi: int) -> list[int]:
    """Integers near the minimum of a unimodal ``|f|`` (ternary search)."""
    a, b = lo, hi
    while b - a > 4:
        m1 = a + (b - a) // 3
        m2 = b - (b - a) // 3
        if abs(f(float(m1))) <= abs(f(float(m2))):
            b = m2
        else:
            a = m1
    return list(range(a, b + 1))


# Public entry points read the cable's own materials when called directly.
_scope_entry_points(globals(), __all__)
