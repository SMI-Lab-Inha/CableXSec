# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""IEC 60287 radial thermal solver (1-D Crank-Nicolson) + analytical gate.

The cable temperature field sets the thermal stress ``sigma_T`` and the
temperature-dependent polymer modulus. This module computes the steady-state
and transient radial temperature profile of a concentric layer stack from the
conductor Joule loss (and, above 100 kV, the dielectric loss).

Two routines, disciplined in the A11/A12/A13 order — the analytical invariant
first, the numerical solver gated against it:

* :func:`steady_state_profile` — the **closed-form** concentric-cylinder
  steady state. With the loss ``W_c`` generated at the conductor, the
  radial heat flux through every outer annulus equals ``W_c``, so the
  temperature rise across annulus *i* is ``W_c R_i`` with the IEC conduction
  thermal resistance ([IEC60287-2-1] Eq. for ``T1``)

  .. math::  R_i = \frac{1}{2\pi\lambda_i}\ln\frac{r_{o,i}}{r_{i,i}},

  and ``T(r)`` is piecewise-logarithmic: ``T_conductor = T_amb + W_c Σ_i R_i``.

* :func:`solve_crank_nicolson` — the 1-D radial transient solver. The cylindrical
  heat equation ``ρc ∂T/∂t = (1/r)∂_r(r λ ∂_r T) + q`` is integrated by
  Crank-Nicolson; its ``t → ∞`` limit reproduces :func:`steady_state_profile`
  to < 0.1 K (the discipline gate).

For current-rating problems where the surroundings are represented by a lumped
external resistance rather than a fixed surface temperature,
:func:`iec_conductor_temperature` evaluates the IEC ladder
([IEC60287-1-1] Eq. 1; [Poradowski2025] Eq. 2) with ``T4`` as an external
**Robin-type series resistance** (soil for a buried cable; bend-stiffener +
water gap for a dynamic cable section). It reduces to the pure conduction ladder above
when ``W_d = λ1 = 0`` and ``n = 1`` (used as the three-core reduction gate).
:func:`compute_thermal` builds that ladder from a cable definition and returns
a radial profile in which the heat flow through each annulus carries the
losses generated inside it ([IEC60287-1-1] 1.4.1.1 Eq. 1; [NeherMcGrath1957]
Eqs. 1-2), so the profile ends equal ``θ_c`` and ``θ_a + W_tot T4``.

Heat sources ([IEC60287-1-1]):

* Conductor Joule ``W_c = I^2 R_ac`` per conductor with
  ``R_ac = R_dc(T)(1 + y_s + y_p)``,
  ``R_dc(T) = R_dc,20 (1 + α_20 (T - 20))`` (:func:`ac_resistance`,
  :func:`skin_effect_factor`, :func:`proximity_effect_factor`).
* Dielectric ``W_d = U_0^2 ω C tanδ`` — active only above 100 kV
  (:func:`dielectric_loss`).
* Sheath ``λ1`` circulating-current loss for bonded metallic sheaths
  (:func:`sheath_circulating_loss_factor`).
* Armour ``λ2`` ([IEC60287-1-1] 2.4): supplied by the caller, not computed
  (the full standard is not held; finite element values for three-core
  armoured cables lie well below the IEC estimate, e.g. Goddard et al. 2015,
  Table V), released at the armour radii in equal shares.
* **Not implemented:** the sheath eddy-current loss ``λ1''``.

References
----------
[IEC60287-1-1] (losses, ``R_ac``, dielectric, sheath ``λ1``, ladder Eq. 1);
[IEC60287-2-1] (thermal resistances ``T1..T3``); [Poradowski2025] (external
``T4`` for dynamic/buried submarine cables, Eq. 2, which, like the ladder
implemented here, carries no armour loss factor ``λ2``);
[NeherMcGrath1957] J. H. Neher and M. H. McGrath, "The calculation of the
temperature rise and load capability of cable systems", AIEE Trans. Power
App. Syst. 76(3), 752--772 (1957), doi:10.1109/AIEEPAS.1957.4499653
(thermal-circuit basis of the IEC 60287 ladder: heat flow increasing in steps
where the losses enter, Eqs. 1-2; dielectric loss acting at the middle of the
conductor-to-sheath resistance, paragraph after Eq. 5A).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from cablexsec.cables.schema import uses_cable_materials
from cablexsec.exceptions import CableInputError
from cablexsec.materials.library import get_property
from cablexsec.provenance import Provenance, ValidatedRange, check_range

if TYPE_CHECKING:
    from cablexsec.cables.schema import CableData, Layer

__all__ = [
    "ConductionLayer",
    "ThermalResult",
    "ac_resistance",
    "compute_thermal",
    "conductor_layers",
    "conductor_material_of",
    "dielectric_loss",
    "iec_conductor_temperature",
    "proximity_effect_factor",
    "sheath_circulating_loss_factor",
    "skin_effect_factor",
    "solve_crank_nicolson",
    "steady_state_profile",
    "thermal_resistance",
]

FloatArray = NDArray[np.float64]

# Classification threshold (not a material property) separating metallic
# sheaths/screens from polymer and filler layers in the thermal ladder: the
# library's metals have λ ≥ 35 W/(m·K), its polymers and fillers λ ≤ 0.5.
_METALLIC_LAMBDA_MIN_W_mK = 10.0
# Radial tolerance for layer-interface comparisons [m].
_RADIUS_TOL_M = 1.0e-12


@dataclass(frozen=True)
class ConductionLayer:
    """One concentric conduction annulus.

    Parameters
    ----------
    r_inner_m, r_outer_m : float
        Annulus bore and outer radius [m].
    lambda_W_mK : float
        Thermal conductivity [W/(m·K)].
    rho_cp_J_m3K : float
        Volumetric heat capacity [J/(m³·K)] (transient only).
    """

    r_inner_m: float
    r_outer_m: float
    lambda_W_mK: float
    rho_cp_J_m3K: float = 0.0


@dataclass(frozen=True)
class ThermalResult:
    """Radial temperature profile and key node temperatures.

    Attributes
    ----------
    r_m, T_C : numpy.ndarray
        Radial node coordinates [m] and steady-state temperatures [°C]. For a
        multi-core cable from :func:`compute_thermal`, ``r_m`` runs over the
        per-core radii (from the core axis) and then the cable-level radii.
    T_conductor_C, T_armour_C : float
        Conductor and armour temperatures [°C].
    P_total_W_m : float
        Total heat generation per metre [W/m].
    provenance : Provenance or None
        Backing citations/gates and active limitations.
    """

    r_m: FloatArray
    T_C: FloatArray
    T_conductor_C: float
    T_armour_C: float
    P_total_W_m: float
    provenance: Provenance | None = None


def thermal_resistance(r_inner_m: float, r_outer_m: float, lambda_W_mK: float) -> float:
    """Cylindrical conduction thermal resistance ``ln(r_o/r_i)/(2π λ)`` [K·m/W].

    ([IEC60287-2-1], thermal resistance of a concentric layer ``T1``.)
    """
    return math.log(r_outer_m / r_inner_m) / (2.0 * math.pi * lambda_W_mK)


def steady_state_profile(
    layers: list[ConductionLayer],
    W_conductor_W_m: float,
    T_ambient_C: float,
    *,
    n_per_layer: int = 1,
) -> tuple[FloatArray, FloatArray]:
    """Closed-form steady-state radial temperature profile.

    Conductor loss ``W_conductor`` only; the flux through every annulus equals
    ``W_conductor``, so ``T(r) = T_outer + (W/(2π λ)) ln(r_outer/r)`` per annulus
    and ``T_conductor = T_ambient + W Σ_i R_i``. Returns ``(r, T)`` from the
    innermost bore out to the outer surface.

    Parameters
    ----------
    layers : list of ConductionLayer
        Concentric annuli, inner→outer.
    W_conductor_W_m : float
        Conductor loss per unit length [W/m].
    T_ambient_C : float
        Temperature at the outer surface [°C].
    n_per_layer : int, optional
        Sub-nodes per layer for a smooth profile (the temperatures are exact at
        every node regardless of ``n_per_layer``).
    """
    return _piecewise_log_profile(
        layers, [W_conductor_W_m] * len(layers), T_ambient_C, n_per_layer=n_per_layer
    )


def _piecewise_log_profile(
    layers: list[ConductionLayer],
    heat_flows_W_m: Sequence[float],
    T_outer_C: float,
    *,
    n_per_layer: int = 1,
) -> tuple[FloatArray, FloatArray]:
    """Steady conduction profile with a (possibly different) heat flow per annulus.

    Annulus *i* carries the radial heat flow ``q_i`` [W/m], so
    ``T(r) = T(r_o,i) + (q_i/(2π λ_i)) ln(r_o,i/r)`` within it, integrated
    inward from ``T_outer_C`` at the outer surface of the last annulus. Gaps
    between annuli carry no resistance (temperature continuous across them).
    Returns ``(r, T)`` inner → outer with duplicate interface nodes removed.
    """
    # Outer-surface temperature is the boundary; integrate inward.
    r_nodes: list[float] = []
    t_nodes: list[float] = []
    t_outer = T_outer_C
    for lyr, q in zip(reversed(layers), reversed(heat_flows_W_m), strict=True):
        edges = np.linspace(lyr.r_outer_m, lyr.r_inner_m, n_per_layer + 1)
        for r in edges:
            # T(r) = t_outer + (q/(2π λ)) ln(r_outer/r)
            t = t_outer + q / (2.0 * math.pi * lyr.lambda_W_mK) * math.log(
                lyr.r_outer_m / r
            )
            r_nodes.append(float(r))
            t_nodes.append(float(t))
        t_outer = t_outer + q * thermal_resistance(
            lyr.r_inner_m, lyr.r_outer_m, lyr.lambda_W_mK
        )
    # Reverse to inner -> outer and drop duplicate interface nodes.
    r_arr = np.array(r_nodes[::-1], dtype=np.float64)
    t_arr = np.array(t_nodes[::-1], dtype=np.float64)
    _, keep = np.unique(np.round(r_arr, 12), return_index=True)
    return r_arr[keep], t_arr[keep]


def ac_resistance(
    rho_ohm_m: float,
    area_m2: float,
    T_C: float,
    alpha_R_per_K: float,
    *,
    y_s: float = 0.0,
    y_p: float = 0.0,
) -> float:
    """Conductor AC resistance per unit length [Ω/m] ([IEC60287-1-1]).

    ``R_ac = R_dc(T) (1 + y_s + y_p)`` with
    ``R_dc(T) = (ρ_20/A)(1 + α_20 (T - 20))``. Skin ``y_s`` and proximity
    ``y_p`` factors default to zero (DC limit) and are supplied by the caller
    when the IEC frequency-dependent terms are wanted.
    """
    r_dc = (rho_ohm_m / area_m2) * (1.0 + alpha_R_per_K * (T_C - 20.0))
    return r_dc * (1.0 + y_s + y_p)


def dielectric_loss(
    voltage_kV: float,
    epsilon_r: float,
    tan_delta: float,
    r_inner_m: float,
    r_outer_m: float,
    *,
    frequency_Hz: float = 50.0,
    n_phases: int = 1,
) -> float:
    """Dielectric loss per unit length [W/m] ([IEC60287-1-1] Eq. 18).

    ``W_d = n U_0^2 ω C tanδ`` with phase voltage ``U_0 = V/√3`` and coaxial
    capacitance ``C = 2π ε_0 ε_r / ln(r_o/r_i)``. Returns 0 below 100 kV (the
    IEC threshold for including dielectric loss in ratings).
    """
    if voltage_kV < 100.0:
        return 0.0
    eps0 = 8.854e-12
    u0 = voltage_kV * 1.0e3 / math.sqrt(3.0)
    cap = 2.0 * math.pi * eps0 * epsilon_r / math.log(r_outer_m / r_inner_m)
    omega = 2.0 * math.pi * frequency_Hz
    return n_phases * u0**2 * omega * cap * tan_delta


def skin_effect_factor(
    r_dc_ohm_m: float, frequency_Hz: float, *, k_s: float = 1.0
) -> float:
    r"""Skin-effect factor ``y_s`` ([IEC60287-1-1] Eq. 5).

    ``x_s^2 = (8πf/R')·10^{-7}·k_s`` and
    ``y_s = x_s^4 / (192 + 0.8 x_s^4)``, with ``R'`` the DC resistance per
    unit length [Ω/m] at operating temperature and ``k_s`` the conductor
    construction coefficient (``1`` for round/compact stranded;
    [IEC60287-1-1] Table 2 lists segmental values).
    """
    xs2 = (8.0 * math.pi * frequency_Hz / r_dc_ohm_m) * 1.0e-7 * k_s
    xs4 = xs2 * xs2
    return xs4 / (192.0 + 0.8 * xs4)  # [IEC60287-1-1] Eq. 5


def proximity_effect_factor(
    r_dc_ohm_m: float,
    frequency_Hz: float,
    d_conductor_m: float,
    spacing_m: float,
    *,
    k_p: float = 1.0,
) -> float:
    r"""Proximity-effect factor ``y_p`` for three-core cables ([IEC60287-1-1] Eq. 7).

    With ``x_p^4`` formed as in :func:`skin_effect_factor` (using ``k_p``) and
    ``F = x_p^4/(192 + 0.8 x_p^4)``,

    .. math::
        y_p = F\,(d_c/s)^2\left[0.312\,(d_c/s)^2
              + \frac{1.18}{F + 0.27}\right],

    where ``d_c`` is the conductor diameter and ``s`` the axial spacing of
    adjacent conductors (touching trefoil: ``s`` = core overall diameter).
    """
    xp2 = (8.0 * math.pi * frequency_Hz / r_dc_ohm_m) * 1.0e-7 * k_p
    xp4 = xp2 * xp2
    f_p = xp4 / (192.0 + 0.8 * xp4)
    ratio = d_conductor_m / spacing_m
    # [IEC60287-1-1] Eq. 7 (three conductors).
    return f_p * ratio**2 * (0.312 * ratio**2 + 1.18 / (f_p + 0.27))


def sheath_circulating_loss_factor(
    r_sheath_ohm_m: float, r_ac_ohm_m: float, reactance_ohm_m: float
) -> float:
    r"""Sheath circulating-current loss factor ``λ1'`` ([IEC60287-1-1] Eq. 24).

    For metallic sheaths bonded at both ends (trefoil),

    .. math::  \lambda_1' = \frac{R_s}{R}\,\frac{1}{1 + (R_s/X)^2},

    with ``R_s`` the sheath AC resistance per unit length, ``R`` the conductor
    AC resistance, and ``X = 2ω·10^{-7}\ln(2s/d_s)`` the sheath reactance
    ([IEC60287-1-1] Eq. 25). The eddy-current term ``λ1''`` is additive and not
    computed here; for solidly-bonded sheaths the circulating term dominates.
    """
    return (r_sheath_ohm_m / r_ac_ohm_m) / (
        1.0 + (r_sheath_ohm_m / reactance_ohm_m) ** 2
    )


def iec_conductor_temperature(
    T1: float,
    T2: float,
    T3: float,
    T4: float,
    *,
    W_c: float,
    W_d: float = 0.0,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
    n_conductors: int = 1,
    T_ambient_C: float,
) -> float:
    r"""Steady-state conductor temperature, IEC ladder with external ``T4``.

    Implements the IEC 60287 current-rating temperature relation
    ([IEC60287-1-1] Eq. 1; [Poradowski2025] Eq. 2),

    .. math::
        \theta_c = \theta_a + (W_c + \tfrac12 W_d)\,T_1
        + n\,\bigl(W_c(1+\lambda_1) + W_d\bigr)\,T_2
        + n\,\bigl(W_c(1+\lambda_1+\lambda_2) + W_d\bigr)\,(T_3 + T_4),

    where ``T4`` is the **external (Robin-type) series thermal resistance** of
    the surroundings — soil for a buried cable, bend-stiffener + water for a
    dynamic cable section ([Poradowski2025]). The conductor loss ``W_c`` and dielectric
    loss ``W_d`` ([IEC60287-1-1]) inject at the conductor and the insulation
    mid-point respectively; ``λ1`` is the sheath loss factor, ``λ2`` the
    armour loss factor ([IEC60287-1-1] 2.4; ``T3`` lies outside the armour)
    and ``n`` the number of loaded conductors. With ``λ2 = 0`` the three
    outer resistances carry the same heat flow.

    Reduction invariant (gate): with ``W_d = 0``, ``λ1 = λ2 = 0``, ``n = 1``
    this is ``θ_a + W_c (T1 + T2 + T3 + T4)`` — the pure conduction ladder of
    :func:`steady_state_profile` with ``T4`` as the final
    annulus.
    """
    half_dielectric = W_c + 0.5 * W_d
    outer_path = n_conductors * (W_c * (1.0 + lambda1) + W_d)
    if lambda2 == 0.0:
        return T_ambient_C + half_dielectric * T1 + outer_path * (T2 + T3 + T4)
    armour_path = outer_path + n_conductors * W_c * lambda2
    return (
        T_ambient_C + half_dielectric * T1 + outer_path * T2 + armour_path * (T3 + T4)
    )


def solve_crank_nicolson(
    layers: list[ConductionLayer],
    W_conductor_W_m: float,
    T_ambient_C: float,
    *,
    n_per_layer: int = 40,
    dt_s: float = 50.0,
    t_end_s: float = 3.0e6,
) -> tuple[FloatArray, FloatArray]:
    """1-D radial Crank-Nicolson transient solver; returns the steady profile.

    Integrates ``ρc ∂T/∂t = (1/r) ∂_r(r λ ∂_r T)`` with the conductor loss as a
    fixed-flux inner boundary (``2π r_in λ ∂_r T = -W``) and ``T = T_ambient`` at
    the outer surface, until steady state. Returns ``(r, T)`` at the final time;
    its ``t → ∞`` limit reproduces :func:`steady_state_profile` (gate: < 0.1 K).

    Parameters
    ----------
    layers : list of ConductionLayer
        Concentric annuli with ``rho_cp_J_m3K`` set for the transient.
    W_conductor_W_m, T_ambient_C : float
        Conductor loss [W/m] and outer-surface temperature [°C].
    n_per_layer : int, optional
        Radial nodes per layer.
    dt_s, t_end_s : float, optional
        Time step and integration horizon [s].
    """
    # Build a radial node grid with per-node lambda / rho_cp.
    r_list: list[float] = []
    lam_list: list[float] = []
    rc_list: list[float] = []
    for lyr in layers:
        edges = np.linspace(lyr.r_inner_m, lyr.r_outer_m, n_per_layer + 1)[:-1]
        for r in edges:
            r_list.append(float(r))
            lam_list.append(lyr.lambda_W_mK)
            rc_list.append(max(lyr.rho_cp_J_m3K, 1.0))
    r_list.append(layers[-1].r_outer_m)
    lam_list.append(layers[-1].lambda_W_mK)
    rc_list.append(max(layers[-1].rho_cp_J_m3K, 1.0))

    r = np.array(r_list, dtype=np.float64)
    lam = np.array(lam_list, dtype=np.float64)
    rho_cp = np.array(rc_list, dtype=np.float64)
    n = r.size

    # Finite-volume conductances between adjacent nodes: G_i = 2π λ / ln(r_{i+1}/r_i).
    g = np.zeros(n - 1, dtype=np.float64)
    for i in range(n - 1):
        lam_face = 0.5 * (lam[i] + lam[i + 1])
        g[i] = 2.0 * math.pi * lam_face / math.log(r[i + 1] / r[i])

    # Nodal heat capacity per unit length: C_i = ρc_i * π (r_{i+1/2}^2 - r_{i-1/2}^2).
    cap = np.zeros(n, dtype=np.float64)
    for i in range(n):
        r_lo = r[i] if i == 0 else 0.5 * (r[i - 1] + r[i])
        r_hi = r[i] if i == n - 1 else 0.5 * (r[i] + r[i + 1])
        cap[i] = rho_cp[i] * math.pi * (r_hi**2 - r_lo**2)

    # Conductance (Laplacian) matrix K: K T = sources at steady state.
    k_mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        k_mat[i, i] += g[i]
        k_mat[i, i + 1] -= g[i]
        k_mat[i + 1, i + 1] += g[i]
        k_mat[i + 1, i] -= g[i]

    source = np.zeros(n, dtype=np.float64)
    source[0] = W_conductor_W_m  # fixed-flux conductor loss at the inner node

    # Dirichlet outer boundary: pin the last node to ambient.
    def apply_outer_bc(mat: FloatArray, rhs: FloatArray) -> None:
        mat[-1, :] = 0.0
        mat[-1, -1] = 1.0
        rhs[-1] = T_ambient_C

    # Crank-Nicolson: (C/dt + K/2) T^{n+1} = (C/dt - K/2) T^n + S.
    c_over_dt = np.diag(cap / dt_s)
    a_mat = c_over_dt + 0.5 * k_mat
    b_mat = c_over_dt - 0.5 * k_mat
    apply_outer_bc(a_mat, np.zeros(n))  # fix LHS rows for BC
    a_lu = np.linalg.inv(a_mat)

    temp = np.full(n, T_ambient_C, dtype=np.float64)
    n_steps = int(t_end_s / dt_s)
    for _ in range(n_steps):
        rhs = b_mat @ temp + source
        rhs[-1] = T_ambient_C  # enforce BC on RHS
        temp = a_lu @ rhs
    return r, temp


@dataclass(frozen=True)
class _LadderAnnulus:
    """One conduction annulus of the :func:`compute_thermal` ladder (private)."""

    layer: ConductionLayer
    name: str
    per_core: bool  # radius measured from a core axis (multi-core cables only)
    metallic: bool
    layer_name: str = ""  # cable layer name as declared (``name`` is lower case)
    material: str = ""


@dataclass(frozen=True)
class _LadderGeometry:
    """Heat path of a cable definition shared by the steady and transient models.

    ``sheath_found`` is False when no metallic sheath/screen lies outside the
    insulation: the ``T1``/``T2`` split (``in_t1``) then ends at the outer
    radius of the insulation, which is an assumption, and
    :func:`compute_thermal` refuses the losses that depend on the split.
    """

    path: list[_LadderAnnulus]
    in_t1: list[bool]
    n: int
    n_geo: int
    multi_core: bool
    r_core: float
    geometry_notes: list[str]
    sheath_found: bool = True


def _layer_heat_flows(
    path: list[_LadderAnnulus],
    *,
    in_t1: list[bool],
    W_c: float,
    W_d: float,
    lambda1: float,
    n_conductors: int,
    multi_core: bool,
    lambda2: float = 0.0,
    armour_radii_m: tuple[float, ...] = (),
) -> tuple[list[float], float, float]:
    r"""Per-annulus heat flow of the IEC ladder and the matching ``T1``, ``T2``.

    The allocation follows the structure of [IEC60287-1-1] 1.4.1.1 Eq. 1,

    .. math::
        \Delta\theta = (W_c + \tfrac12 W_d)\,T_1
        + [W_c(1+\lambda_1) + W_d]\,n\,T_2
        + [W_c(1+\lambda_1+\lambda_2) + W_d]\,n\,(T_3 + T_4),

    i.e. the heat flow increases in steps at the radii where losses enter
    ([NeherMcGrath1957] Eqs. 1-2): per core, ``W_c + ½W_d`` through every
    annulus between the conductor and the metallic sheath/screen (the IEC
    lumping of the dielectric loss, exact for a uniform insulation wall because
    the loss density varies as ``1/r²``; [NeherMcGrath1957], paragraph after
    Eq. 5A), and ``W_c(1+λ1) + W_d`` outside the sheath. The armour loss
    ``λ2 W_c`` per core enters at the armour radii ``armour_radii_m`` (cable
    axis), shared equally between the armour layers: a cable-level annulus
    outside ``k`` of ``m`` armour layers carries ``W_c(1+λ1+λ2 k/m) + W_d``
    per core. With ``λ2 = 0`` ``T2`` and ``T3`` carry the same flow and are
    lumped; the returned ``T2`` then holds both.

    An annulus common to all cores (a cable-level layer of a multi-core cable,
    or any layer outside the sheath of a single-path ladder) carries the losses
    of the ``n`` loaded cores; a per-core annulus carries those of one core.
    The returned ``T1``, ``T2`` are therefore the per-core equivalents for Eq. 1:
    ``T1 = Σ_{T1} m_i R_i`` and ``T2 = Σ_{outer} m_i R_i / n`` with
    ``m_i ∈ {1, n}``, so that ``Σ_i q_i R_i = (W_c + ½W_d)T1 + n[W_c(1+λ1) +
    W_d]T2`` holds term by term.
    """
    q_t1 = W_c + 0.5 * W_d
    q_outer = W_c * (1.0 + lambda1) + W_d
    flows: list[float] = []
    t1 = 0.0
    t2_scaled = 0.0
    for annulus, inner in zip(path, in_t1, strict=True):
        if multi_core:
            mult = 1.0 if annulus.per_core else float(n_conductors)
        else:
            mult = 1.0 if inner else float(n_conductors)
        r_i = thermal_resistance(
            annulus.layer.r_inner_m, annulus.layer.r_outer_m, annulus.layer.lambda_W_mK
        )
        if inner:
            flows.append(mult * q_t1)
            t1 += mult * r_i
        else:
            share = 0.0
            if armour_radii_m and not annulus.per_core:
                share = sum(
                    1.0
                    for r_a in armour_radii_m
                    if r_a <= annulus.layer.r_inner_m + 1e-12
                ) / len(armour_radii_m)
            flows.append(mult * (q_outer + W_c * lambda2 * share))
            t2_scaled += mult * r_i
    return flows, t1, t2_scaled / float(n_conductors)


#: Absolute zero [°C]: the floor of a temperature input.
_ABSOLUTE_ZERO_C = -273.15


@uses_cable_materials
def compute_thermal(
    cable: CableData,
    current_A: float,
    conductor_area_m2: float,
    *,
    n_cores: int | None = None,
    conductor_material: str | None = None,
    conductor_temp_C: float = 90.0,
    y_s: float = 0.0,
    y_p: float = 0.0,
    lambda1: float = 0.0,
    T4_external_Km_W: float | None = None,
    ambient_C: float | None = None,
    lambda2: float = 0.0,
) -> ThermalResult:
    r"""Steady-state IEC 60287 conductor temperature for a cable definition.

    Assembles the concentric conduction ladder from the cable's annular layers
    (each with ``r_inner_m``/``r_outer_m`` and a material conductivity) and
    closes it with the external Robin resistance ``T4`` from the cable's
    ``environment`` block (or the override argument). The ladder is split as in
    [IEC60287-1-1] 1.4.1.1 Eq. 1: ``T1`` is the resistance between the
    conductor and the metallic sheath/screen, ``T2``/``T3`` the resistance
    outside it ([IEC60287-2-1] for the annulus resistances). The conductor
    temperature follows :func:`iec_conductor_temperature` ([Poradowski2025]
    Eq. 2).

    The conductor Joule loss is ``W_c = I^2 R_ac`` per conductor with ``R_ac``
    from :func:`ac_resistance` at ``conductor_temp_C`` (skin/proximity ``y_s``,
    ``y_p`` supplied by the caller via :func:`skin_effect_factor` /
    :func:`proximity_effect_factor`); dielectric loss is added above 100 kV.

    Radial profile
    --------------
    The returned ``T(r)`` is piecewise logarithmic, with the heat flow through
    each annulus carrying the losses generated inside it (see
    :func:`_layer_heat_flows`): ``W_c + ½W_d`` per core between the conductor
    and the sheath, ``W_c(1+λ1) + W_d`` per core outside the sheath, ``n`` times
    that through layers common to all cores. Integrating inward from the
    surface temperature ``θ_a + W_tot T4`` (``W_tot = n[W_c(1+λ1) + W_d]``)
    therefore returns exactly ``θ_c`` at the conductor surface. The sheath
    radius is the inner radius of the first metallic layer (a
    ``cylindrical_shell``, or an annulus with ``λ ≥ 10 W/(m·K)``) outside the
    insulation; per-core metallic layers take precedence over cable-level ones.
    The insulation is identified from the layer stack, not from layer names:
    it is the run of non-metallic annuli that starts at the first non-metallic
    annulus outside the conductor (in the per-core stack of a multi-core
    cable) and ends at the next metallic layer. If no metallic layer follows
    it, the split of ``T1`` and ``T2`` is undefined: a call with a dielectric
    loss or ``lambda1 > 0``, which depend on the split, is refused; without
    them the split does not change the result. Metallic shells and helical
    armour carry no conduction resistance in the ladder (temperature
    continuous across them). For a multi-core cable ``r_m`` runs over the
    per-core radii (from the core axis) and then over the cable-level radii
    (from the cable axis).

    Armour temperature
    ------------------
    ``T_armour_C`` is the profile temperature at the helix radius of the
    innermost structural ``helical`` layer. A cable without a structural
    helical layer has no armour temperature (``NaN``, recorded in the
    provenance); a structural helical layer without ``r_helix_m`` is refused.

    Conductor material
    ------------------
    Without ``conductor_material`` the material of the cable's conductor
    layer is used: the ``cylindrical_offset`` layer at the core centre
    (``r_inner_m = 0``), else the ``helical_stranded`` layer. A cable with no
    such layer, or with several of different materials, needs
    ``conductor_material``. The choice is recorded in the provenance.

    Interstitial filler (multi-core cables)
    ---------------------------------------
    A cable-level layer with ``r_inner_m = 0`` in a cable with more than one
    core is the filler around the cores, not a heat source. It is **not**
    dropped: until the [IEC60287-2-1] filler/bedding geometric factor is
    implemented, it is represented by the annulus of equal filler area,
    ``√n_geo · r_core ≤ r ≤ r_outer``, with ``r_core`` the largest per-core
    outer radius and ``n_geo`` the number of cores. This explicit approximation
    is recorded in the provenance limitations. As an order-of-magnitude check
    only (filler conductivity of [Poradowski2025] not stated), the CIGRE TB 880
    geometry (touching cores of 65.9 mm radius in a 141.95 mm filler, library
    ``pp_filler`` and ``hdpe``) gives 0.166 K·m/W for filler plus armour
    bedding, about 1.5 times the tabulated ``T2 = 0.1121`` K·m/W of
    [Poradowski2025] Table 1, i.e. on the high (conservative) side; the
    previous behaviour (filler omitted) gave about 0.008. A per-core layer with
    ``r_inner_m = 0`` (and any such layer of a single-core cable) is the
    conductor, the heat source, and is excluded from the ladder.

    Three-core reduction gate
    -------------------------
    With ``n_cores = 1``, no dielectric loss and ``lambda1 = 0`` the ladder
    collapses to ``θ_a + W_c (ΣR_internal + T4)`` — i.e. exactly the 1-D
    primitive :func:`steady_state_profile` evaluated on the
    same annuli with the cable surface raised by ``W_c T4``. The three-core
    mutual-heating geometric factor of [IEC60287-2-1] is **not** applied; for
    ``n_cores > 1`` the result follows the lumped IEC ladder, and the
    geometric correction is left as a documented refinement.

    Limitations
    -----------
    The armour loss factor ``λ2`` ([IEC60287-1-1] 2.4) is not computed: it is
    the caller's ``lambda2`` (default 0, armour losses then absent from the
    heat flow through ``T3`` and ``T4`` and from ``P_total_W_m``), released at
    the armour radii in equal shares. The sheath eddy-current loss ``λ1''`` is
    not computed (``lambda1`` is whatever the caller supplies).

    Parameters
    ----------
    cable : CableData
        Cable whose concentric annuli define the conduction ladder.
    current_A : float
        Conductor current [A].
    conductor_area_m2 : float
        Conductor cross-sectional area [m²] for ``R_dc``.
    n_cores : int, optional
        Number of loaded conductors ``n``. Defaults to the count of
        ``cylindrical_offset`` cores, else 1.
    conductor_material : str, optional
        Material key for ``rho_ohm_m`` and ``alpha_R_per_K``; defaults to the
        material of the cable's conductor layer (see above).
    conductor_temp_C : float, optional
        Temperature at which ``R_ac`` is evaluated [°C] (default 90 = XLPE max).
    y_s, y_p : float, optional
        Skin and proximity factors (default 0 = DC limit).
    lambda1 : float, optional
        Sheath loss factor (default 0).
    T4_external_Km_W, ambient_C : float, optional
        Override the cable ``environment`` block.
    lambda2 : float, optional
        Armour loss factor ([IEC60287-1-1] 2.4; default 0), released at the
        radii of the structural armour layers in equal shares.
    """
    env = cable.environment
    t4 = T4_external_Km_W if T4_external_Km_W is not None else env.T4_external_Km_W
    t_amb = ambient_C if ambient_C is not None else env.ambient_C
    if t4 is None or t_amb is None:
        raise CableInputError(
            "compute_thermal needs T4_external_Km_W and ambient_C "
            "(set them in the cable 'environment' block or pass overrides)."
        )
    checks = {
        "current_A": (current_A, None),
        "conductor_temp_C": (conductor_temp_C, False),
        "ambient_C": (t_amb, False),
        "conductor_area_m2": (conductor_area_m2, True),
        "y_s": (y_s, None),
        "y_p": (y_p, None),
        "lambda1": (lambda1, None),
        "lambda2": (lambda2, None),
        "T4_external_Km_W": (t4, None),
    }
    for name, (value, positive) in checks.items():
        if not math.isfinite(value):
            raise CableInputError(f"compute_thermal: {name} must be finite")
        if positive is True and value <= 0.0:
            raise CableInputError(f"compute_thermal: {name} must be positive")
        if positive is None and value < 0.0:
            raise CableInputError(f"compute_thermal: {name} must not be negative")
    for name, value in (("conductor_temp_C", conductor_temp_C), ("ambient_C", t_amb)):
        if value < _ABSOLUTE_ZERO_C:
            raise CableInputError(
                f"compute_thermal: {name} = {value:g} °C is below absolute zero"
            )
    geo = _ladder_geometry(cable, n_cores)
    path, in_t1, n = geo.path, geo.in_t1, geo.n
    multi_core, geometry_notes = geo.multi_core, list(geo.geometry_notes)
    ladder = [a.layer for a in path]

    # Losses.
    material, material_basis = conductor_material_of(cable, conductor_material)
    try:
        rho = get_property(material, "rho_ohm_m")
        alpha = get_property(material, "alpha_R_per_K")
    except KeyError as error:
        raise CableInputError(
            f"compute_thermal: conductor material {material!r} needs rho_ohm_m "
            "and alpha_R_per_K (give them in the cable 'materials' block or "
            f"pass conductor_material): {error}"
        ) from None
    r_ac = ac_resistance(
        rho, conductor_area_m2, conductor_temp_C, alpha, y_s=y_s, y_p=y_p
    )
    w_c = current_A**2 * r_ac
    w_d = _cable_dielectric_loss(cable)
    if not geo.sheath_found and (w_d > 0.0 or lambda1 > 0.0):
        raise CableInputError(
            "compute_thermal: no metallic sheath or screen lies outside the "
            "insulation, so the T1/T2 split that the dielectric loss and "
            "lambda1 need is undefined; declare the metallic sheath/screen "
            "layer (cylindrical_shell or a metallic annulus)"
        )

    armour_radii = _armour_radii(cable) if lambda2 > 0.0 else ()
    if lambda2 > 0.0 and not armour_radii:
        raise CableInputError(
            "compute_thermal: lambda2 > 0 needs a structural armour layer"
        )
    flows, t1, t2 = _layer_heat_flows(
        path,
        in_t1=in_t1,
        W_c=w_c,
        W_d=w_d,
        lambda1=lambda1,
        n_conductors=n,
        multi_core=multi_core,
        lambda2=lambda2,
        armour_radii_m=armour_radii,
    )
    # Total heat leaving the cable and the surface temperature.
    p_total = n * (w_c * (1.0 + lambda1 + lambda2) + w_d)
    theta_surface = t_amb + p_total * t4
    if lambda2 == 0.0:
        theta_c = iec_conductor_temperature(
            t1,
            t2,
            0.0,
            t4,
            W_c=w_c,
            W_d=w_d,
            lambda1=lambda1,
            n_conductors=n,
            T_ambient_C=t_amb,
        )
    else:
        # Eq. 1 with T2 and T3 separated at the armour: the sum of the
        # annulus drops, each carrying its own heat flow.
        theta_c = theta_surface + sum(
            q
            * thermal_resistance(
                a.layer.r_inner_m, a.layer.r_outer_m, a.layer.lambda_W_mK
            )
            for q, a in zip(flows, path, strict=True)
        )

    # Radial profile integrated inward from the surface with the per-annulus
    # heat flows; its conductor node reproduces theta_c to round-off.
    r_m, t_profile = _piecewise_log_profile(
        ladder, flows, theta_surface, n_per_layer=10
    )

    # Armour temperature: profile value at the innermost structural armour.
    armour_r, armour_note = _armour_temperature_radius(cable)
    t_armour = (
        float(np.interp(armour_r, r_m, t_profile)) if armour_r is not None else math.nan
    )

    # Soft validated-range guard: the conduction ladder + Robin T4 was validated
    # (BM5, Poradowski 2025) at the XLPE continuous-operation limit of 90 degC.
    check_range(
        float(theta_c),
        ValidatedRange(
            "conductor temperature [degC]",
            t_amb,
            90.0,
            "[IEC60287] XLPE 90 degC continuous limit; BM5 [Poradowski2025] "
            "validated at 90 degC",
        ),
    )
    limitations = [
        material_basis,
        "external resistance T4 taken verbatim as input (not derived from "
        "installation geometry)",
    ]
    if armour_note:
        limitations.append(armour_note)
    if n > 1:
        limitations.append(
            "three-core mutual-heating geometric factor [IEC60287-2-1] NOT "
            "applied; lumped IEC ladder used (n>1)"
        )
    limitations.extend(geometry_notes)
    limitations.append(
        "sheath eddy-loss lambda1'' ([IEC60287-1-1] 2.3.6) not included "
        "(circulating lambda1' only); BM5 residual ~5 K when lambda1''=0"
    )
    limitations.append(
        "armour loss factor lambda2 ([IEC60287-1-1] 2.4) supplied by the "
        "caller, not computed; released at the armour radii in equal shares"
        if lambda2 > 0.0
        else "armour loss factor lambda2 ([IEC60287-1-1] 2.4) zero: armour "
        "losses absent from the T3/T4 heat flow and P_total_W_m"
    )
    prov = Provenance(
        method="IEC 60287 conduction ladder with external Robin resistance T4",
        citations=(
            "[IEC60287-1-1] 1.4.1.1 Eq.1",
            "[IEC60287-2-1]",
            "[Poradowski2025] Eq.2",
            "[NeherMcGrath1957] Eqs.1-2",
        ),
        gates=(
            "A9 reduction (n=1) <1e-12",
            "BM5 90 degC",
            "profile ends T(r_c)=theta_c, T(r_s)=theta_a+P_total*T4 <1e-12",
        ),
        limitations=tuple(limitations),
    )
    return ThermalResult(
        r_m, t_profile, float(theta_c), t_armour, float(p_total), provenance=prov
    )


def _cable_dielectric_loss(cable: CableData) -> float:
    """Per-core dielectric loss of the first insulation with electrical data."""
    voltage = cable.cable.voltage_kV
    if voltage is None or voltage < 100.0:
        return 0.0
    for layer in cable.layers:
        elec = getattr(layer, "electrical", None)
        r_in = getattr(layer, "r_inner_m", None)
        r_out = getattr(layer, "r_outer_m", None)
        if elec is None or r_in is None or r_out is None:
            continue
        tan_d = elec.loss_factor_tan_delta
        eps_r = elec.permittivity_relative
        if tan_d and eps_r:
            return dielectric_loss(
                voltage,
                eps_r,
                tan_d,
                r_in,
                r_out,
                frequency_Hz=cable.environment.frequency_Hz,
            )
    # At this voltage the dielectric loss is part of the ladder; a silent zero
    # would understate the conductor temperature.
    raise CableInputError(
        f"the cable is rated {voltage:g} kV (100 kV or more), so the ladder "
        "includes the dielectric loss, but no insulation layer gives "
        "'electrical: {permittivity_relative, loss_factor_tan_delta}'"
    )


def _armour_radii(cable: CableData) -> tuple[float, ...]:
    """Helix radii of the structural armour layers (cable axis) [m]."""
    return tuple(
        float(layer.r_helix_m)
        for layer in cable.helical_layers
        if layer.structural and layer.r_helix_m
    )


def _armour_temperature_radius(cable: CableData) -> tuple[float | None, str]:
    """Radius of the armour temperature and a provenance note.

    The innermost structural ``helical`` layer is the armour (schema layer
    type and ``structural`` flag, not the layer name). ``None`` and a note
    when the cable has none; a structural armour without ``r_helix_m`` is an
    input error.
    """
    armour = [layer for layer in cable.helical_layers if layer.structural]
    if not armour:
        return None, "no structural helical (armour) layer: T_armour_C is NaN"
    missing = [layer.name for layer in armour if not layer.r_helix_m]
    if missing:
        raise CableInputError(
            "compute_thermal: the armour temperature needs r_helix_m of the "
            f"structural helical layers {missing}"
        )
    return min(float(layer.r_helix_m or 0.0) for layer in armour), ""


def conductor_layers(cable: CableData) -> list[Layer]:
    """The conductor layers of a cable, identified from the schema.

    The ``cylindrical_offset`` layers at the core centre (``r_inner_m = 0``);
    without them, the ``helical_stranded`` layers. Empty when the cable has
    no conductor (a flexible pipe, for instance).
    """
    centre: list[Layer] = [
        layer
        for layer in cable.layers
        if layer.layer_type == "cylindrical_offset"
        and (getattr(layer, "r_inner_m", None) or 0.0) <= 0.0
    ]
    if centre:
        return centre
    return [layer for layer in cable.layers if layer.layer_type == "helical_stranded"]


def conductor_material_of(cable: CableData, given: str | None) -> tuple[str, str]:
    """Conductor material key and a provenance note stating where it came from.

    ``given`` (an explicit argument) wins; otherwise the single material of
    :func:`conductor_layers`. No conductor layer, or conductor layers of
    different materials, raise :class:`CableInputError`.
    """
    if given is not None:
        if not isinstance(given, str) or not given.strip():
            raise CableInputError("conductor_material must be a non-empty string")
        return given, f"conductor material {given!r} given explicitly"
    layers = conductor_layers(cable)
    materials = sorted({layer.material for layer in layers})
    if len(materials) != 1:
        found = (
            "no conductor layer (a cylindrical_offset layer at the core centre "
            "or a helical_stranded layer)"
            if not materials
            else f"conductor layers of different materials {materials}"
        )
        raise CableInputError(
            "the conductor material cannot be determined from the cable: "
            f"{found}; pass conductor_material"
        )
    names = ", ".join(repr(layer.name) for layer in layers)
    return materials[0], (
        f"conductor material {materials[0]!r} from the conductor layer {names}"
    )


def _ladder_geometry(cable: CableData, n_cores: int | None) -> _LadderGeometry:
    """Concentric heat path, ``T1`` split and core counts of a cable definition."""
    if n_cores is not None and (
        isinstance(n_cores, bool) or not isinstance(n_cores, Integral) or n_cores < 1
    ):
        raise CableInputError(
            f"n_cores must be a positive integer (at least 1): {n_cores!r}"
        )
    n_geo = (
        max((getattr(layer, "n_cores", 0) for layer in cable.layers), default=0) or 1
    )
    n = n_cores if n_cores is not None else n_geo
    multi_core = n_geo > 1
    r_core = max(
        (
            float(getattr(layer, "r_outer_m", None) or 0.0)
            for layer in cable.layers
            if layer.layer_type == "cylindrical_offset"
        ),
        default=0.0,
    )

    # Concentric conduction annuli (layers exposing r_inner_m/r_outer_m) and
    # metallic sheath candidates (r_inner, per_core).
    annuli: list[_LadderAnnulus] = []
    sheath_candidates: list[tuple[float, bool, str]] = []
    geometry_notes: list[str] = []
    for layer in cable.layers:
        name = layer.name.lower()
        per_core = multi_core and layer.layer_type == "cylindrical_offset"
        if layer.layer_type == "cylindrical_shell":
            r_mid = getattr(layer, "r_mid_m", None)
            t_shell = getattr(layer, "t_m", None)
            if r_mid is not None and t_shell is not None:
                sheath_candidates.append((r_mid - 0.5 * t_shell, False, name))
            continue
        r_in = getattr(layer, "r_inner_m", None)
        r_out = getattr(layer, "r_outer_m", None)
        if r_in is None or r_out is None or r_out <= r_in:
            continue
        lam = get_property(layer.material, "lambda_W_mK")
        if r_in <= 0.0:
            if per_core or not multi_core:
                # The conductor (r_inner = 0) is the heat source, not a
                # ln(r_o/r_i) conduction annulus; T1 starts at its surface.
                continue
            # Cable-level interstitial filler around the cores: equal-filler-
            # area annulus (explicit approximation; see docstring).
            r_eq = math.sqrt(n_geo) * r_core
            if not 0.0 < r_eq < r_out:
                raise CableInputError(
                    f"Filler layer {layer.name!r} (r_inner_m = 0, r_outer_m = "
                    f"{r_out:g} m) cannot enclose {n_geo} cores of outer radius "
                    f"{r_core:g} m; check the core and filler radii."
                )
            geometry_notes.append(
                f"interstitial filler {layer.name!r} (r_inner_m = 0) represented "
                f"by the equal-filler-area annulus sqrt(n)*r_core = "
                f"{r_eq * 1e3:.2f} mm to {r_out * 1e3:.2f} mm (n = {n_geo}); "
                "[IEC60287-2-1] filler/bedding geometric factor NOT applied"
            )
            r_in = r_eq
        metallic = lam >= _METALLIC_LAMBDA_MIN_W_mK
        annuli.append(
            _LadderAnnulus(
                ConductionLayer(r_in, r_out, lam),
                name,
                per_core,
                metallic,
                layer_name=layer.name,
                material=layer.material,
            )
        )
        if metallic:
            sheath_candidates.append((r_in, per_core, name))
    if not annuli:
        raise CableInputError(
            "Cable has no concentric annular layers for a thermal ladder."
        )
    # Heat path: per-core annuli (core-axis radii), then cable-level annuli.
    path = sorted(annuli, key=lambda a: (not a.per_core, a.layer.r_inner_m))
    for inner_a, outer_a in itertools.pairwise(path):
        if outer_a.layer.r_inner_m < inner_a.layer.r_outer_m - _RADIUS_TOL_M:
            raise CableInputError(
                f"Thermal annuli {inner_a.name!r} and {outer_a.name!r} overlap "
                f"({outer_a.layer.r_inner_m:g} m < {inner_a.layer.r_outer_m:g} m)."
            )
    # T1 boundary: the metallic sheath/screen outside the insulation. The
    # insulation is found from the layer stack (material conductivity and
    # position), not from layer names: the run of non-metallic annuli that
    # starts at the first non-metallic annulus of the innermost stack (the
    # per-core stack of a multi-core cable) and ends at the next metallic
    # layer (annulus or shell) of that stack.
    ins_per_core = multi_core and any(a.per_core for a in path)
    stack = sorted(
        [
            (a.layer.r_inner_m, a.layer.r_outer_m, a.metallic)
            for a in path
            if a.per_core == ins_per_core
        ]
        + [(c[0], c[0], True) for c in sheath_candidates if c[1] == ins_per_core]
    )
    r_ins_out: float | None = None
    for _r_inner, r_outer, metallic in stack:
        if not metallic:
            r_ins_out = r_outer
        elif r_ins_out is not None:
            break
    sheath_found = False
    if r_ins_out is not None:
        ins_out = r_ins_out

        def _outside_insulation(r_inner: float, per_core: bool) -> bool:
            if per_core == ins_per_core:
                return r_inner >= ins_out - _RADIUS_TOL_M
            return not per_core  # cable-level layers lie outside the cores

        candidates = [c for c in sheath_candidates if _outside_insulation(c[0], c[1])]
        preferred = [c for c in candidates if c[1]] or candidates
        if preferred:
            r_sheath, sheath_per_core, _sheath_name = min(preferred)
            sheath_found = True
        else:
            r_sheath, sheath_per_core = ins_out, ins_per_core
            geometry_notes.append(
                "no metallic sheath/screen identified outside the insulation; "
                "the T1/T2 split ends at the insulation outer radius (no "
                "dielectric loss or lambda1 accepted)"
            )

        def _in_t1(a: _LadderAnnulus) -> bool:
            if not multi_core or a.per_core == sheath_per_core:
                return a.layer.r_outer_m <= r_sheath + _RADIUS_TOL_M
            return a.per_core  # per-core layer inside a cable-level sheath

        in_t1 = [_in_t1(a) for a in path]
    else:
        in_t1 = [False] * len(path)  # no insulation identified: T1 = 0
        geometry_notes.append(
            "no non-metallic (insulation) annulus in the heat path: T1 = 0 "
            "(no dielectric loss or lambda1 accepted)"
        )

    return _LadderGeometry(
        path, in_t1, n, n_geo, multi_core, r_core, geometry_notes, sheath_found
    )
