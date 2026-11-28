# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""IEC 60287 radial thermal solver (1-D Crank-Nicolson) + analytical gate.

The cable temperature field sets the thermal stress ``sigma_T`` and the
temperature-dependent polymer modulus. This module computes the steady-state
and transient radial temperature profile of a concentric layer stack from the
conductor Joule loss, the dielectric loss and the sheath loss, following IEC
60287-1-1:2023 and IEC 60287-2-1:2023.

Two routines, disciplined in the A11/A12/A13 order — the analytical invariant
first, the numerical solver gated against it:

* :func:`steady_state_profile` — the **closed-form** concentric-cylinder
  steady state. With the loss ``W_c`` generated at the conductor, the
  radial heat flux through every outer annulus equals ``W_c``, so the
  temperature rise across annulus *i* is ``W_c R_i`` with the IEC conduction
  thermal resistance of a concentric layer ([IEC60287-2-1] 4.1.3)

  .. math::  R_i = \frac{1}{2\pi\lambda_i}\ln\frac{r_{o,i}}{r_{i,i}},

  and ``T(r)`` is piecewise-logarithmic: ``T_conductor = T_amb + W_c Σ_i R_i``.

* :func:`solve_crank_nicolson` — the 1-D radial transient solver. The cylindrical
  heat equation ``ρc ∂T/∂t = (1/r)∂_r(r λ ∂_r T) + q`` is integrated by
  Crank-Nicolson; its ``t → ∞`` limit reproduces :func:`steady_state_profile`
  to < 0.1 K (the discipline gate).

For current-rating problems where the surroundings are represented by a lumped
external resistance rather than a fixed surface temperature,
:func:`iec_conductor_temperature` evaluates the IEC ladder
([IEC60287-1-1] 4.2.1, Formula (1); [Poradowski2025] Eq. 2) with ``T4`` as an external
**Robin-type series resistance** (soil for a buried cable; bend-stiffener +
water gap for a dynamic cable section). It reduces to the pure conduction ladder above
when ``W_d = λ1 = 0`` and ``n = 1`` (used as the three-core reduction gate).
:func:`compute_thermal` builds that ladder from a cable definition and returns
a radial profile in which the heat flow through each annulus carries the
losses generated inside it ([IEC60287-1-1] 4.2.1, Formula (1);
[NeherMcGrath1957] Eqs. 1-2), so the profile ends equal ``θ_c`` and
``θ_a + W_tot T4``.

Heat sources ([IEC60287-1-1] Clause 5), computed from the cable unless given:

* Conductor Joule ``W_c = I^2 R_C`` per conductor with
  ``R_C = R'(1 + y_s + y_p)``, ``R' = R_0 (1 + α_20 (θ - 20))`` (5.1,
  Formula (7)); ``R_0`` from [IEC60228] Table 4, per length of the complete
  cable (:func:`conductor_resistance_20C`); ``y_s`` over the three ranges of
  5.1.3 and ``y_p`` of 5.1.4/5.1.5 (:func:`skin_effect_factor`,
  :func:`proximity_effect_factor`).
* Dielectric ``W_d = ω C U_0^2 tanδ`` with ``C`` times the lay-up factor
  ``C_LL`` (5.2, Annex A), counted from the ``U_0`` of the insulation type
  (127 kV for XLPE; :func:`dielectric_loss`).
* Sheath ``λ1'`` of the per-core screens of an SL-type cable (5.3.2, or
  5.3.11 under magnetic armour; :func:`sheath_circulating_loss_factor`).
  ``λ1'' = 0`` as in those clauses; :func:`sheath_eddy_loss_factor` gives
  the eddy loss of 5.3.7.1 that CIGRE TB 880 (2022) Guidance Points 6 and 31
  keep, for a caller who adds it.
* Armour ``λ2`` (5.4): supplied by the caller, not computed (finite element
  values for three-core armoured cables lie well below the IEC estimate,
  e.g. Goddard et al. 2015, Table V), released at the armour radii in equal
  shares.

Thermal resistances ([IEC60287-2-1] 4.1): per-core annuli divided by
``C_LL`` (4.1.2.1, 4.1.3), ``T1`` from the conductor diameter ``d_c``, the
semiconducting screens with the insulation (4.1.1), and the fillers and
bedding of an armoured three-core SL-type cable by the geometric factor
``G-bar`` (4.1.4.2, 5.6; CIGRE TB 880 Guidance Point 45).

References
----------
[IEC60287-1-1] IEC 60287-1-1:2023 (losses, ``R_C``, dielectric, sheath
``λ1``, ladder Formula (1), lay-up factor Annex A); [IEC60287-2-1] IEC
60287-2-1:2023 (thermal resistances ``T1..T3``, Table 1 resistivities);
[IEC60228] IEC 60228:2023 (conductor resistance); [Poradowski2025] (external
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

from cablexsec.cables.schema import scope_entry_points as _scope_entry_points
from cablexsec.cables.schema import uses_cable_materials
from cablexsec.exceptions import CableInputError
from cablexsec.materials.library import get_property
from cablexsec.provenance import Provenance, ValidatedRange, check_range

if TYPE_CHECKING:
    from cablexsec.cables.schema import CableData, Layer

__all__ = [
    "DIELECTRIC_LOSS_U0_KV",
    "ConductionLayer",
    "ThermalResult",
    "ac_resistance",
    "compute_thermal",
    "conductor_layers",
    "conductor_material_of",
    "conductor_resistance_20C",
    "dielectric_loss",
    "iec_conductor_temperature",
    "lay_length_factor",
    "proximity_effect_factor",
    "sheath_circulating_loss_factor",
    "sheath_eddy_loss_factor",
    "sheath_reactance",
    "skin_effect_factor",
    "sl_filler_geometric_factor",
    "solve_crank_nicolson",
    "steady_state_profile",
    "thermal_resistance",
]

FloatArray = NDArray[np.float64]

#: [IEC60228] (2023) Table 4, class 2 stranded conductors: maximum DC
#: resistance at 20 degC [ohm/km] of plain annealed copper and of aluminium,
#: by nominal cross-section [mm^2] (aluminium from 10 mm^2). Per km of the
#: complete cable ([IEC60228] Annex A), so no lay-up factor is applied to it.
_IEC60228_CLASS2_OHM_KM: dict[float, tuple[float, float | None]] = {
    0.75: (24.5, None),
    1.0: (18.1, None),
    1.5: (12.1, None),
    2.5: (7.41, None),
    4.0: (4.61, None),
    6.0: (3.08, None),
    10.0: (1.83, 3.08),
    16.0: (1.15, 1.91),
    25.0: (0.727, 1.20),
    35.0: (0.524, 0.868),
    50.0: (0.387, 0.641),
    70.0: (0.268, 0.443),
    95.0: (0.193, 0.320),
    120.0: (0.153, 0.253),
    150.0: (0.124, 0.206),
    185.0: (0.0991, 0.164),
    240.0: (0.0754, 0.125),
    300.0: (0.0601, 0.100),
    400.0: (0.0470, 0.0778),
    500.0: (0.0366, 0.0605),
    630.0: (0.0283, 0.0469),
    800.0: (0.0221, 0.0367),
    1000.0: (0.0176, 0.0291),
    1200.0: (0.0151, 0.0247),
    1400.0: (0.0129, 0.0212),
    1600.0: (0.0113, 0.0186),
    1800.0: (0.0101, 0.0165),
    2000.0: (0.0090, 0.0149),
    2500.0: (0.0072, 0.0119),
    3000.0: (0.0060, 0.0099),
    3200.0: (0.0056, 0.0093),
    3500.0: (0.0051, 0.0085),
}
#: [IEC60287-1-1] (2023) Table 1 conductor resistivities at 20 degC [ohm m],
#: used to recognise a copper or an aluminium conductor material.
_RHO_COPPER, _RHO_ALUMINIUM = 1.7241e-8, 2.8264e-8
#: [IEC60287-1-1] (2023) Annex A, Table A.1: neutral-axis factor C_fL of
#: laid-up round cores, by number of cores.
_CFL_BY_CORES: dict[int, float] = {2: 1.16, 3: 1.29, 4: 1.53, 5: 1.80, 6: 2.08}
#: [IEC60287-1-1] (2023) 5.2: U0 [kV] from which the dielectric loss is
#: taken into account, by insulation type.
DIELECTRIC_LOSS_U0_KV: dict[str, float] = {
    "paper_solid": 38.0,
    "paper_fluid": 63.5,
    "ppl": 63.5,
    "butyl_rubber": 18.0,
    "epr": 63.5,
    "pvc": 6.0,
    "pe": 127.0,
    "xlpe": 127.0,
    "xlpe_filled": 63.5,
}
#: An armour material with ``relative_permeability`` at or below this value
#: is non-magnetic (stainless steel, copper, aluminium); one without the
#: property is taken as magnetic steel ([IEC60287-1-1] (2023) 5.4.3.1 d)
#: gives 400 for steel wires).
_MAGNETIC_PERMEABILITY_MIN = 1.5

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
    losses : dict or None
        Loss terms of one core used by :func:`compute_thermal`: ``R0_ohm_m``,
        ``R_ac_ohm_m``, ``y_s``, ``y_p``, ``lay_length_factor``, ``W_c_W_m``,
        ``W_d_W_m``, ``lambda1``, ``lambda2``.
    """

    r_m: FloatArray
    T_C: FloatArray
    T_conductor_C: float
    T_armour_C: float
    P_total_W_m: float
    provenance: Provenance | None = None
    losses: dict[str, float] | None = None


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
    insulation_type: str = "xlpe",
    lay_length_factor: float = 1.0,
    include: bool | None = None,
) -> float:
    r"""Dielectric loss per unit length [W/m] ([IEC60287-1-1] 5.2).

    ``W_d = n ω C U_0^2 tanδ`` with the voltage to earth ``U_0 = U/√3``
    (``voltage_kV`` is the rated phase-to-phase voltage ``U``) and the
    capacitance ``C = 2π ε_0 ε_r C_LL / ln(D_i/d_c)`` per core, ``C_LL`` the
    lay-up factor of laid-up cores ([IEC60287-1-1] Annex A); ``r_inner_m`` is
    the radius over the conductor screen and ``r_outer_m`` the radius over the
    insulation without its screen.

    ``include=None`` follows [IEC60287-1-1] 5.2: the loss is taken into
    account from the ``U_0`` of :data:`DIELECTRIC_LOSS_U0_KV` for the
    ``insulation_type`` (127 kV for unfilled XLPE) and is zero below it.
    ``include=True`` always evaluates it, as CIGRE TB 880 (2022) Guidance
    Point 7 recommends; ``False`` returns zero.
    """
    u0_kV = voltage_kV / math.sqrt(3.0)
    if include is None:
        try:
            threshold = DIELECTRIC_LOSS_U0_KV[insulation_type]
        except KeyError:
            raise CableInputError(
                f"unknown insulation_type {insulation_type!r}; one of "
                f"{sorted(DIELECTRIC_LOSS_U0_KV)}"
            ) from None
        include = u0_kV >= threshold
    if not include:
        return 0.0
    eps0 = 8.854e-12  # [IEC60287-1-1] 5.2
    cap = (
        2.0
        * math.pi
        * eps0
        * epsilon_r
        * lay_length_factor
        / math.log(r_outer_m / r_inner_m)
    )
    omega = 2.0 * math.pi * frequency_Hz
    return n_phases * (u0_kV * 1.0e3) ** 2 * omega * cap * tan_delta


def _bessel_argument_squared(r_dc_ohm_m: float, frequency_Hz: float, k: float) -> float:
    """``x^2 = (8 π f / R') 10^-7 k`` ([IEC60287-1-1] 5.1.3 to 5.1.5)."""
    return (8.0 * math.pi * frequency_Hz / r_dc_ohm_m) * 1.0e-7 * k


def skin_effect_factor(
    r_dc_ohm_m: float, frequency_Hz: float, *, k_s: float = 1.0
) -> float:
    r"""Skin-effect factor ``y_s`` ([IEC60287-1-1] 5.1.3).

    With ``x_s^2 = (8πf/R')·10^{-7}·k_s``, ``R'`` the DC resistance per unit
    length [Ω/m] at the operating temperature and ``k_s`` the coefficient of
    [IEC60287-1-1] Table 2 (1 for round stranded conductors):

    * ``0 < x_s ≤ 2.8``: ``y_s = x_s^4 / (192 + 0.8 x_s^4)``;
    * ``2.8 < x_s ≤ 3.8``: ``y_s = -0.136 - 0.0177 x_s + 0.0563 x_s^2``;
    * ``x_s > 3.8``: ``y_s = 0.354 x_s - 0.733``.
    """
    xs2 = _bessel_argument_squared(r_dc_ohm_m, frequency_Hz, k_s)
    xs = math.sqrt(xs2)
    if xs <= 2.8:
        return xs2 * xs2 / (192.0 + 0.8 * xs2 * xs2)
    if xs <= 3.8:
        return -0.136 - 0.0177 * xs + 0.0563 * xs2
    return 0.354 * xs - 0.733


def proximity_effect_factor(
    r_dc_ohm_m: float,
    frequency_Hz: float,
    d_conductor_m: float,
    spacing_m: float,
    *,
    k_p: float = 1.0,
    n_conductors: int = 3,
) -> float:
    r"""Proximity-effect factor ``y_p`` ([IEC60287-1-1] 5.1.4 and 5.1.5.1).

    With ``x_p^2 = (8πf/R')·10^{-7}·k_p`` and ``F = x_p^4/(192 + 0.8 x_p^4)``,
    three conductors (5.1.5.1, a three-core cable or three single-core
    cables):

    .. math::
        y_p = F\,(d_c/s)^2\left[0.312\,(d_c/s)^2
              + \frac{1.18}{F + 0.27}\right],

    and two conductors (5.1.4): ``y_p = 2.9 F (d_c/s)^2``. ``d_c`` is the
    conductor diameter and ``s`` the distance between conductor axes. The
    formulae hold for ``x_p ≤ 2.8``.
    """
    if n_conductors not in (2, 3):
        raise CableInputError(
            "the IEC proximity-effect factor is given for two or three "
            f"conductors, not {n_conductors}; pass y_p"
        )
    xp2 = _bessel_argument_squared(r_dc_ohm_m, frequency_Hz, k_p)
    xp4 = xp2 * xp2
    f_p = xp4 / (192.0 + 0.8 * xp4)
    ratio = d_conductor_m / spacing_m
    if n_conductors == 2:
        return 2.9 * f_p * ratio**2  # [IEC60287-1-1] 5.1.4
    return f_p * ratio**2 * (0.312 * ratio**2 + 1.18 / (f_p + 0.27))


def sheath_circulating_loss_factor(
    r_sheath_ohm_m: float,
    r_ac_ohm_m: float,
    reactance_ohm_m: float,
    *,
    sl_armoured: bool = False,
) -> float:
    r"""Sheath circulating-current loss factor ``λ1'``.

    Two or three single-core cables in trefoil, sheaths bonded at both ends,
    and the cores of an unarmoured SL-type cable ([IEC60287-1-1] 5.3.2):

    .. math::  \lambda_1' = \frac{R_s}{R}\,\frac{1}{1 + (R_s/X)^2},

    with ``R_s`` the sheath resistance and ``R`` the conductor AC resistance
    per unit length at their operating temperatures, and ``X = 2ω·10^{-7}
    \ln(2s/d)`` the sheath reactance (``s`` the distance between conductor
    axes, ``d`` the mean sheath diameter). An SL-type cable under magnetic
    armour (``sl_armoured=True``, [IEC60287-1-1] 5.3.11) has the factor 1.5
    on this expression and ``X`` multiplied by the lay-up factor ``C_LL``,
    which the caller includes in ``reactance_ohm_m``. The eddy-current term
    ``λ1''`` is zero in both clauses.
    """
    factor = 1.5 if sl_armoured else 1.0
    return (
        factor
        * (r_sheath_ohm_m / r_ac_ohm_m)
        / (1.0 + (r_sheath_ohm_m / reactance_ohm_m) ** 2)
    )


def sheath_eddy_loss_factor(
    r_sheath_ohm_m: float,
    r_ac_ohm_m: float,
    *,
    sheath_mean_diameter_m: float,
    sheath_outer_diameter_m: float,
    sheath_thickness_m: float,
    sheath_resistivity_ohm_m: float,
    spacing_m: float,
    frequency_Hz: float = 50.0,
    solid_bonding_reactance_ohm_m: float | None = None,
) -> float:
    r"""Sheath eddy-current loss factor ``λ1''`` of three cables in trefoil.

    [IEC60287-1-1] 5.3.7.1 (sheaths single-point bonded or cross-bonded):

    .. math::
        \lambda_1'' = \frac{R_s}{R}\left[g_s \lambda_0 (1 + \Delta_1 + \Delta_2)
        + \frac{(\beta_1 t_s)^4}{12\cdot10^{12}}\right],

    with ``m = ω 10^-7 / R_s``, ``λ0 = 3 m²/(1+m²) (d/2s)²``, ``Δ1 = (1.14
    m^2.45 + 0.33)(d/2s)^(0.92 m + 1.66)``, ``Δ2 = 0``, ``β1 = sqrt(4 π ω /
    (10^7 ρ_s))`` and ``g_s = 1 + (t_s/D_s)^1.74 (β1 D_s 10^-3 - 1.6)``
    (lengths in mm inside the brackets, as in the standard).

    IEC 60287-1-1 sets ``λ1'' = 0`` for sheaths bonded at both ends and for SL
    cables (5.3.2, 5.3.11); :func:`compute_thermal` follows it. CIGRE TB 880
    (2022) Guidance Point 6 keeps the eddy loss in every case and Guidance
    Point 31 multiplies it, under solid bonding, by the factor ``F`` of
    [IEC60287-1-1] 5.3.6 with ``M = N = R_s/X``: pass the sheath reactance
    ``X`` as ``solid_bonding_reactance_ohm_m`` for that product. The value
    can then be added to a computed ``λ1'`` and passed as ``lambda1``.
    """
    omega = 2.0 * math.pi * frequency_Hz
    m = omega * 1.0e-7 / r_sheath_ohm_m
    q = sheath_mean_diameter_m / (2.0 * spacing_m)
    lam0 = 3.0 * (m**2 / (1.0 + m**2)) * q**2
    delta1 = (1.14 * m**2.45 + 0.33) * q ** (0.92 * m + 1.66)
    beta1 = math.sqrt(4.0 * math.pi * omega / (1.0e7 * sheath_resistivity_ohm_m))
    ts_mm, ds_mm = sheath_thickness_m * 1e3, sheath_outer_diameter_m * 1e3
    g_s = 1.0 + (ts_mm / ds_mm) ** 1.74 * (beta1 * ds_mm * 1.0e-3 - 1.6)
    lam = (r_sheath_ohm_m / r_ac_ohm_m) * (
        g_s * lam0 * (1.0 + delta1) + (beta1 * ts_mm) ** 4 / 12.0e12
    )
    if solid_bonding_reactance_ohm_m is not None:
        mm = r_sheath_ohm_m / solid_bonding_reactance_ohm_m
        lam *= (4.0 * mm**4 + (2.0 * mm) ** 2) / (4.0 * (mm**2 + 1.0) ** 2)
    return float(lam)


def sheath_reactance(
    spacing_m: float,
    sheath_mean_diameter_m: float,
    *,
    frequency_Hz: float = 50.0,
    lay_length_factor: float = 1.0,
) -> float:
    """Sheath reactance ``X = 2 ω 10^-7 ln(2s/d) C_LL`` [Ω/m] ([IEC60287-1-1] 5.3)."""
    omega = 2.0 * math.pi * frequency_Hz
    return (
        2.0
        * omega
        * 1.0e-7
        * math.log(2.0 * spacing_m / sheath_mean_diameter_m)
        * lay_length_factor
    )


def lay_length_factor(cable: CableData) -> tuple[float, str]:
    r"""Lay-up factor ``C_LL`` of laid-up cores and a provenance note.

    [IEC60287-1-1] (2023) Annex A, Formula (A.1) (also [IEC60287-2-1] (2023)
    Annex A):

    .. math::  C_{LL} = \sqrt{1 + \left(\frac{\pi C_{fL} D_p}{L_L}\right)^2},

    with ``D_p`` the diameter of a core (over its outermost per-core layer),
    ``L_L`` the core lay pitch (``core_lay.pitch_m``) and ``C_fL`` from Table
    A.1 by the number of cores. One for a single-core cable and for cores laid
    straight (no ``core_lay`` block). It multiplies a resistance, capacitance
    or reactance computed for a straight core and divides a per-core thermal
    resistance.
    """
    n_cores = max((getattr(layer, "n_cores", 0) for layer in cable.layers), default=0)
    if n_cores <= 1:
        return 1.0, ""
    lay = cable.core_lay
    if lay is None:
        return 1.0, "cores laid straight (no core_lay): lay-up factor C_LL = 1"
    if lay.pitch_m is None:
        return 1.0, (
            "core_lay.pitch_m withheld: lay-up factor C_LL ([IEC60287-1-1] "
            "Annex A) not applied (C_LL = 1)"
        )
    try:
        cfl = _CFL_BY_CORES[n_cores]
    except KeyError:
        raise CableInputError(
            f"[IEC60287-1-1] Table A.1 gives C_fL for 2 to 6 cores, not {n_cores}"
        ) from None
    d_core = 2.0 * max(
        (
            float(getattr(layer, "r_outer_m", None) or 0.0)
            for layer in cable.layers
            if layer.layer_type == "cylindrical_offset"
        ),
        default=0.0,
    )
    if d_core <= 0.0:
        raise CableInputError("the lay-up factor needs the outer radius of the cores")
    c_ll = math.sqrt(1.0 + (math.pi * cfl * d_core / lay.pitch_m) ** 2)
    return c_ll, (
        f"lay-up factor C_LL = {c_ll:.4f} ([IEC60287-1-1] Annex A: C_fL {cfl}, "
        f"core diameter {d_core * 1e3:.2f} mm, lay pitch {lay.pitch_m:g} m)"
    )


def conductor_resistance_20C(
    cable: CableData,
    conductor_area_m2: float,
    material: str,
    *,
    lay_factor: float = 1.0,
) -> tuple[float, str]:
    """DC resistance of one conductor at 20 degC per metre of cable [Ω/m] and its basis.

    [IEC60287-1-1] (2023) 5.1.2: ``R_0`` is taken from [IEC60228]: a copper
    or aluminium conductor (recognised by its resistivity, within 3 % of the
    [IEC60287-1-1] Table 1 value) of a nominal class 2 cross-section takes the
    maximum resistance of [IEC60228] (2023) Table 4, which holds per length of
    the complete cable ([IEC60228] Annex A); a cross-section between two
    nominal sizes is interpolated linearly (CIGRE TB 880 (2022), Section 2.5).
    Otherwise the resistance is ``ρ/A`` with the material resistivity, times
    the lay-up factor ``lay_factor`` for the length of a laid-up core
    ([IEC60287-1-1] 5.1.2 and Annex A).
    """
    rho = get_property(material, "rho_ohm_m")
    area_mm2 = conductor_area_m2 * 1.0e6
    column = None
    if abs(rho / _RHO_COPPER - 1.0) <= 0.03:
        column, metal = 0, "copper"
    elif abs(rho / _RHO_ALUMINIUM - 1.0) <= 0.03:
        column, metal = 1, "aluminium"
    solid = any(
        getattr(layer, "construction", "") == "solid"
        for layer in conductor_layers(cable)
    )
    if column is not None and not solid:
        sizes = sorted(
            a for a, row in _IEC60228_CLASS2_OHM_KM.items() if row[column] is not None
        )
        values = [_IEC60228_CLASS2_OHM_KM[a][column] for a in sizes]
        if sizes[0] <= area_mm2 <= sizes[-1]:
            r_km = float(np.interp(area_mm2, sizes, values))  # type: ignore[arg-type]
            nominal = any(math.isclose(area_mm2, a, rel_tol=1e-6) for a in sizes)
            how = (
                "nominal size"
                if nominal
                else "interpolated between nominal sizes (CIGRE TB 880, 2.5)"
            )
            return r_km * 1.0e-3, (
                f"R_0 = {r_km:.4g} ohm/km of the complete cable from [IEC60228] "
                f"Table 4, class 2 {metal}, {area_mm2:g} mm2 ({how})"
            )
    r20 = rho / conductor_area_m2 * lay_factor
    reason = (
        "solid conductor"
        if solid
        else "material is not the copper or aluminium of [IEC60287-1-1] Table 1"
        if column is None
        else "cross-section outside [IEC60228] Table 4"
    )
    return r20, (
        f"R_0 = rho/A x C_LL = {r20 * 1e3:.4g} ohm/km ({reason}; [IEC60287-1-1] "
        "5.1.2 and Annex A)"
    )


@dataclass(frozen=True)
class _ConductorLosses:
    """Conductor resistance and loss factors of one core (private)."""

    r20_ohm_m: float
    alpha_R_per_K: float
    y_s: float
    y_p: float
    lay_factor: float
    notes: tuple[str, ...]


def _conductor_offset_layer(cable: CableData) -> Layer | None:
    for layer in conductor_layers(cable):
        if layer.layer_type == "cylindrical_offset":
            return layer
    return None


def _conductor_losses(
    cable: CableData,
    conductor_area_m2: float,
    material: str,
    temperature_C: float,
    *,
    y_s: float | None,
    y_p: float | None,
    r20_ohm_m: float | None,
    k_s: float | None,
    k_p: float | None,
) -> _ConductorLosses:
    """``R_0`` and the skin and proximity factors at ``temperature_C`` (IEC 5.1)."""
    notes: list[str] = []
    c_ll, c_note = lay_length_factor(cable)
    if c_note:
        notes.append(c_note)
    alpha = get_property(material, "alpha_R_per_K")
    if r20_ohm_m is None:
        r20, r_note = conductor_resistance_20C(
            cable, conductor_area_m2, material, lay_factor=c_ll
        )
        notes.append(r_note)
    else:
        r20 = float(r20_ohm_m)
        notes.append(f"R_0 = {r20 * 1e3:.4g} ohm/km given explicitly")
    freq = cable.environment.frequency_Hz
    r_dc = r20 * (1.0 + alpha * (temperature_C - 20.0))
    aluminium = abs(get_property(material, "rho_ohm_m") / _RHO_ALUMINIUM - 1.0) <= 0.03
    solid = any(
        getattr(layer, "construction", "") == "solid"
        for layer in conductor_layers(cable)
    )
    # [IEC60287-1-1] Table 2, round conductor, extruded insulation: k_s = 1;
    # k_p = 1, except 0.8 for stranded aluminium.
    ks = 1.0 if k_s is None else float(k_s)
    kp = (0.8 if aluminium and not solid else 1.0) if k_p is None else float(k_p)
    if (y_s is None and k_s is None) or (y_p is None and k_p is None):
        notes.append(
            f"k_s = {ks:g}, k_p = {kp:g} from [IEC60287-1-1] Table 2 (round "
            "conductor, extruded insulation; pass k_s/k_p for Milliken or "
            "hollow conductors)"
        )
    if y_s is None:
        y_s = skin_effect_factor(r_dc, freq, k_s=ks)
    if y_p is None:
        n_geo = max((getattr(layer, "n_cores", 0) for layer in cable.layers), default=0)
        core = _conductor_offset_layer(cable)
        if n_geo <= 1 or core is None:
            y_p = 0.0
            notes.append(
                "single-core cable: the proximity effect depends on the spacing "
                "of the other cables of the circuit and is not computed (y_p = 0; "
                "pass y_p, [IEC60287-1-1] 5.1.4/5.1.5)"
            )
        else:
            r_offset = float(getattr(core, "r_offset_m", None) or 0.0)
            spacing = 2.0 * r_offset * math.sin(math.pi / n_geo)
            d_c = float(
                getattr(core, "d_conductor_m", None)
                or 2.0 * float(getattr(core, "r_outer_m", 0.0) or 0.0)
            )
            if spacing <= 0.0 or d_c <= 0.0:
                raise CableInputError(
                    "the proximity effect needs the core offset and the conductor "
                    "diameter; pass y_p"
                )
            y_p = proximity_effect_factor(
                r_dc, freq, d_c, spacing, k_p=kp, n_conductors=n_geo
            )
            notes.append(
                f"y_p from [IEC60287-1-1] 5.1.{4 if n_geo == 2 else 5} with d_c = "
                f"{d_c * 1e3:.2f} mm and s = {spacing * 1e3:.2f} mm"
            )
    return _ConductorLosses(r20, alpha, float(y_s), float(y_p), c_ll, tuple(notes))


def _sheath_loss_factor(
    cable: CableData,
    geo: _LadderGeometry,
    *,
    r_ac_ohm_m: float,
    sheath_temperature_C: float,
    lay_factor: float,
) -> tuple[float, str]:
    """Sheath loss factor ``λ1`` of the cable from its geometry, and a note.

    Computed for the per-core metallic screens of a two- or three-core
    (SL-type) cable: [IEC60287-1-1] 5.3.11 under magnetic armour (factor 1.5,
    ``X`` times ``C_LL``) and 5.3.2 otherwise, following CIGRE TB 880 (2022)
    Guidance Point 38 for non-magnetic armour; ``λ1'' = 0`` in both clauses.
    ``R_s`` is that of one core's screen at the sheath temperature, times
    ``C_LL``: the wires of a wire screen (``strand_layers``) over their own
    lay (CIGRE TB 880 Guidance Point 27), a solid sheath or foil by its
    annulus, and further per-core metallic layers over the screen (a laminated
    foil) in parallel with it, as in TB 880 Case 2; metal tapes are left out
    (Guidance Point 29, counter helix tapes). A single-core cable (bonding and
    spacing of the circuit) and a common sheath over the cores are not
    computed: ``λ1 = 0`` with a note, and the caller passes ``lambda1``.
    """
    if not geo.sheath_found:
        return 0.0, ""
    if not geo.multi_core:
        return 0.0, (
            "single-core cable: the sheath loss depends on the bonding and the "
            "spacing of the circuit and is not computed (lambda1 = 0; pass "
            "lambda1, [IEC60287-1-1] 5.3.2 to 5.3.7)"
        )
    if not geo.sheath_per_core or geo.n_geo not in (2, 3):
        return 0.0, (
            "metallic sheath common to the cores: the sheath loss "
            "([IEC60287-1-1] 5.3.8 to 5.3.10) is not computed (lambda1 = 0; "
            "pass lambda1)"
        )
    layer = next(x for x in cable.layers if x.name == geo.sheath_name)
    by_name = {x.name: x for x in cable.layers}
    r_screen_in = float(getattr(layer, "r_inner_m", 0.0) or 0.0)
    counted = [layer]
    tapes: list[str] = []
    for a in geo.path:
        other = by_name.get(a.layer_name)
        if (
            other is None
            or other is layer
            or not (a.metallic and a.per_core)
            or a.layer.r_inner_m < r_screen_in - _RADIUS_TOL_M
        ):
            continue
        if getattr(other, "tape", None) is not None:
            tapes.append(other.name)
        else:
            counted.append(other)
    conductance = 0.0  # 1 / R_s per unit length of the core
    diameters: list[float] = []
    parts: list[str] = []
    for item in counted:
        r_in = float(getattr(item, "r_inner_m", 0.0) or 0.0)
        r_out = float(getattr(item, "r_outer_m", 0.0) or 0.0)
        try:
            rho_s = get_property(item.material, "rho_ohm_m")
            alpha_s = get_property(item.material, "alpha_R_per_K")
        except KeyError:
            raise CableInputError(
                f"the sheath loss needs rho_ohm_m and alpha_R_per_K of the screen "
                f"material {item.material!r} ({item.name!r}); give them or pass "
                "lambda1"
            ) from None
        rings = tuple(getattr(item, "strand_layers", ()) or ())
        if rings:
            # Wires in parallel, each ring over its own helix length per unit
            # core length (1 / cos of its lay angle; TB 880 GP 27).
            area = sum(ring.area_m2 * math.cos(ring.lay_angle_rad) for ring in rings)
            what = (
                f"{sum(ring.n_wires for ring in rings)} wires of "
                f"{sum(ring.area_m2 for ring in rings) * 1e6:.4g} mm2 over their lay"
            )
        else:
            area = math.pi * (r_out**2 - r_in**2)
            what = f"annulus of {area * 1e6:.4g} mm2"
            if getattr(item, "construction", "") == "resolved_wire_screen":
                what += " (a wire screen without strand_layers)"
        r_item = rho_s / area * (1.0 + alpha_s * (sheath_temperature_C - 20.0))
        conductance += 1.0 / r_item
        diameters.append(r_in + r_out)
        parts.append(f"{item.name!r}: {what}")
    r_s = lay_factor / conductance
    # Mean diameter of the screens in parallel: the mean of their own mean
    # diameters (TB 880 Case 2, wires and laminated foil).
    d_mean = sum(diameters) / len(diameters)
    metal = (
        f" ({'; '.join(parts)}"
        + ("; in parallel" if len(counted) > 1 else "")
        + (
            f"; metal tape(s) {', '.join(tapes)} not counted, TB 880 GP 29"
            if tapes
            else ""
        )
        + ")"
    )
    core = _conductor_offset_layer(cable)
    r_offset = float(getattr(core, "r_offset_m", None) or 0.0) if core else 0.0
    spacing = 2.0 * r_offset * math.sin(math.pi / geo.n_geo)
    armour = [layer for layer in cable.helical_layers if layer.structural]
    magnetic = [
        a
        for a in armour
        if _relative_permeability(a.material) > _MAGNETIC_PERMEABILITY_MIN
    ]
    sl_armoured = geo.n_geo == 3 and bool(magnetic)
    x = sheath_reactance(
        spacing,
        d_mean,
        frequency_Hz=cable.environment.frequency_Hz,
        lay_length_factor=lay_factor if sl_armoured else 1.0,
    )
    lam1 = sheath_circulating_loss_factor(r_s, r_ac_ohm_m, x, sl_armoured=sl_armoured)
    clause = (
        "5.3.11 (SL type under magnetic armour: factor 1.5, X times C_LL)"
        if sl_armoured
        else "5.3.2 (SL type, unarmoured or non-magnetic armour; CIGRE TB 880 GP 38)"
    )
    screen = metal
    common = [a.layer_name for a in geo.path if a.metallic and not a.per_core] + [
        x.name for x in cable.layers if x.layer_type == "cylindrical_shell"
    ]
    common_note = (
        f"; the losses of the metallic layer(s) over the cores ({', '.join(common)}) "
        "are not computed (pass lambda1 to include them)"
        if common
        else ""
    )
    return lam1, (
        f"lambda1' = {lam1:.4f} from [IEC60287-1-1] {clause}: R_s = "
        f"{r_s * 1e3:.4g} ohm/km of {layer.name!r} at {sheath_temperature_C:.1f} "
        f"degC{screen}, s = {spacing * 1e3:.2f} mm, d = {d_mean * 1e3:.2f} mm; "
        "lambda1'' = 0 (IEC; CIGRE TB 880 GP 6 would add the eddy loss)" + common_note
    )


def _relative_permeability(material: str) -> float:
    """``relative_permeability`` of an armour material; magnetic steel if absent."""
    try:
        return get_property(material, "relative_permeability")
    except KeyError:
        return 400.0  # [IEC60287-1-1] 5.4.3.1 d), steel wires


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
    armour loss factor ([IEC60287-1-1] 5.4; ``T3`` lies outside the armour)
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
    sheath_name: str = ""
    sheath_per_core: bool = False
    lay_factor: float = 1.0


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

    The allocation follows the structure of [IEC60287-1-1] 4.2.1, Formula (1),

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
    y_s: float | None = None,
    y_p: float | None = None,
    lambda1: float | None = None,
    T4_external_Km_W: float | None = None,
    ambient_C: float | None = None,
    lambda2: float = 0.0,
    conductor_resistance_20C_ohm_m: float | None = None,
    k_s: float | None = None,
    k_p: float | None = None,
    include_dielectric_loss: bool | None = None,
) -> ThermalResult:
    r"""Steady-state IEC 60287 conductor temperature for a cable definition.

    Assembles the concentric conduction ladder from the cable's annular layers
    (each with ``r_inner_m``/``r_outer_m`` and a material conductivity) and
    closes it with the external Robin resistance ``T4`` from the cable's
    ``environment`` block (or the override argument). The ladder is split as in
    [IEC60287-1-1] 4.2.1, Formula (1): ``T1`` is the resistance between the
    conductor and the metallic sheath/screen, ``T2``/``T3`` the resistance
    outside it ([IEC60287-2-1] 4.1 for the annulus resistances). The
    conductor temperature follows :func:`iec_conductor_temperature`
    ([Poradowski2025] Eq. 2).

    Losses
    ------
    The conductor Joule loss is ``W_c = I^2 R_C`` per conductor with
    ``R_C = R_0 (1 + α_20 (θ - 20)) (1 + y_s + y_p)`` at ``conductor_temp_C``
    ([IEC60287-1-1] 5.1, Formula (7)). Each loss term not given is computed
    from the cable:

    * ``R_0`` from [IEC60228] Table 4 for a copper or aluminium class 2
      conductor (per length of the complete cable), else ``ρ/A`` times the
      lay-up factor ``C_LL`` (:func:`conductor_resistance_20C`,
      :func:`lay_length_factor`);
    * ``y_s`` from [IEC60287-1-1] 5.1.3, all three ranges of ``x_s``;
    * ``y_p`` of a two- or three-core cable from 5.1.4/5.1.5.1 with the
      conductor diameter ``d_conductor_m`` (else ``2 r_outer_m``) and the
      distance between conductor axes; a single-core cable gets ``y_p = 0``
      (the spacing of the circuit is not part of the cable), recorded;
    * ``λ1`` of the per-core screens of a two- or three-core (SL-type)
      cable: [IEC60287-1-1] 5.3.11 under magnetic armour (factor 1.5),
      5.3.2 otherwise, ``λ1'' = 0`` (IEC; CIGRE TB 880 Guidance Point 6 would
      add the eddy loss), the screen resistance at the sheath temperature
      ``θ - (W_c + ½W_d) T1`` (5.3.1). A single-core cable or a common sheath
      gets ``λ1 = 0``, recorded; pass ``lambda1``;
    * the dielectric loss ``W_d`` per core ([IEC60287-1-1] 5.2) from the
      first insulation layer with ``electrical`` data, with ``C`` times
      ``C_LL``, counted from the ``U_0`` of its ``insulation_type`` (127 kV for
      unfilled XLPE) unless ``include_dielectric_loss`` says otherwise.

    The values used are in ``ThermalResult.losses`` and their basis in the
    provenance limitations.

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
    is recorded in the provenance limitations. A per-core layer with
    ``r_inner_m = 0`` (and any such layer of a single-core cable) is the
    conductor, the heat source, and is excluded from the ladder.

    Three-core reduction gate
    -------------------------
    With ``n_cores = 1``, no dielectric loss and ``lambda1 = 0`` the ladder
    collapses to ``θ_a + W_c (ΣR_internal + T4)`` — i.e. exactly the 1-D
    primitive :func:`steady_state_profile` evaluated on the
    same annuli with the cable surface raised by ``W_c T4``. For
    ``n_cores > 1`` the result follows the lumped IEC ladder: per-core
    annuli for each core and, in an armoured three-core SL-type cable, the
    filler resistance of the [IEC60287-2-1] geometric factor; the belted and
    screened-cable factors of [IEC60287-2-1] 4.1.2.2 and 4.1.2.3 are not
    implemented.

    Limitations
    -----------
    The armour loss factor ``λ2`` ([IEC60287-1-1] 5.4) is not computed: it is
    the caller's ``lambda2`` (default 0, armour losses then absent from the
    heat flow through ``T3`` and ``T4`` and from ``P_total_W_m``), released at
    the armour radii in equal shares. The skin and proximity factors are not
    raised by 1.5 for magnetic armour (CIGRE TB 880 Guidance Point 25 does;
    [IEC60287-1-1] 5.1.6 does it for pipe-type cables only).

    Parameters
    ----------
    cable : CableData
        Cable whose concentric annuli define the conduction ladder.
    current_A : float
        Conductor current [A].
    conductor_area_m2 : float
        Conductor cross-section [m²]: the [IEC60228] nominal size of ``R_0``.
    n_cores : int, optional
        Number of loaded conductors ``n``. Defaults to the count of
        ``cylindrical_offset`` cores, else 1.
    conductor_material : str, optional
        Material key for ``rho_ohm_m`` and ``alpha_R_per_K``; defaults to the
        material of the cable's conductor layer (see above).
    conductor_temp_C : float, optional
        Temperature at which ``R_C`` is evaluated [°C] (default 90 = XLPE max).
    y_s, y_p : float, optional
        Skin and proximity factors; computed when not given (see Losses).
    lambda1 : float, optional
        Sheath loss factor; computed when not given (see Losses).
    T4_external_Km_W, ambient_C : float, optional
        Override the cable ``environment`` block.
    lambda2 : float, optional
        Armour loss factor ([IEC60287-1-1] 5.4; default 0), released at the
        radii of the structural armour layers in equal shares.
    conductor_resistance_20C_ohm_m : float, optional
        ``R_0`` per metre of cable [Ω/m] (a datasheet or measured value),
        instead of [IEC60228].
    k_s, k_p : float, optional
        Skin and proximity coefficients ([IEC60287-1-1] Table 2); default 1
        and 1 (0.8 for stranded aluminium), round conductor, extruded
        insulation.
    include_dielectric_loss : bool, optional
        ``None`` (default): the [IEC60287-1-1] 5.2 voltage threshold;
        ``True``: always (CIGRE TB 880 Guidance Point 7); ``False``: never.
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
        "conductor_resistance_20C_ohm_m": (conductor_resistance_20C_ohm_m, True),
        "k_s": (k_s, True),
        "k_p": (k_p, True),
    }
    for name, (value, positive) in checks.items():
        if value is None:
            continue
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
        get_property(material, "rho_ohm_m")
        get_property(material, "alpha_R_per_K")
    except KeyError as error:
        raise CableInputError(
            f"compute_thermal: conductor material {material!r} needs rho_ohm_m "
            "and alpha_R_per_K (give them in the cable 'materials' block or "
            f"pass conductor_material): {error}"
        ) from None
    cond = _conductor_losses(
        cable,
        conductor_area_m2,
        material,
        conductor_temp_C,
        y_s=y_s,
        y_p=y_p,
        r20_ohm_m=conductor_resistance_20C_ohm_m,
        k_s=k_s,
        k_p=k_p,
    )
    r_ac = (
        cond.r20_ohm_m
        * (1.0 + cond.alpha_R_per_K * (conductor_temp_C - 20.0))
        * (1.0 + cond.y_s + cond.y_p)
    )
    w_c = current_A**2 * r_ac
    w_d, w_d_note = _cable_dielectric_loss(
        cable, lay_factor=cond.lay_factor, include=include_dielectric_loss
    )
    lambda1_note = ""
    if lambda1 is None:
        t1_core = sum(
            thermal_resistance(
                a.layer.r_inner_m, a.layer.r_outer_m, a.layer.lambda_W_mK
            )
            for a, inner in zip(path, in_t1, strict=True)
            if inner
        )
        lambda1, lambda1_note = _sheath_loss_factor(
            cable,
            geo,
            r_ac_ohm_m=r_ac,
            sheath_temperature_C=conductor_temp_C - (w_c + 0.5 * w_d) * t1_core,
            lay_factor=cond.lay_factor,
        )
    elif not geo.sheath_found and lambda1 > 0.0:
        raise CableInputError(
            "compute_thermal: no metallic sheath or screen lies outside the "
            "insulation, so the T1/T2 split that lambda1 needs is undefined; "
            "declare the metallic sheath/screen layer (cylindrical_shell or a "
            "metallic annulus)"
        )
    if not geo.sheath_found and w_d > 0.0:
        raise CableInputError(
            "compute_thermal: no metallic sheath or screen lies outside the "
            "insulation, so the T1/T2 split that the dielectric loss needs is "
            "undefined; declare the metallic sheath/screen layer "
            "(cylindrical_shell or a metallic annulus)"
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

    # Soft validated-range guard: the IEC ladder applies up to the XLPE
    # continuous-operation limit of 90 degC.
    check_range(
        float(theta_c),
        ValidatedRange(
            "conductor temperature [degC]",
            t_amb,
            90.0,
            "[IEC60287-1-1] 5.1.2: XLPE maximum operating temperature 90 degC",
        ),
    )
    limitations = [
        material_basis,
        *cond.notes,
        "external resistance T4 taken verbatim as input (not derived from "
        "installation geometry)",
    ]
    if w_d_note:
        limitations.append(w_d_note)
    if lambda1_note:
        limitations.append(lambda1_note)
    if armour_note:
        limitations.append(armour_note)
    g_bar_applied = any("G-bar =" in note for note in geometry_notes)
    if n > 1 and not g_bar_applied:
        limitations.append(
            "three-core geometric factors of [IEC60287-2-1] (belted, screened "
            "or SL-type fillers) NOT applied; lumped IEC ladder used (n>1)"
        )
    limitations.extend(geometry_notes)
    limitations.append(
        "armour loss factor lambda2 ([IEC60287-1-1] 5.4) supplied by the "
        "caller, not computed; released at the armour radii in equal shares"
        if lambda2 > 0.0
        else "armour loss factor lambda2 ([IEC60287-1-1] 5.4) zero: armour "
        "losses absent from the T3/T4 heat flow and P_total_W_m"
    )
    prov = Provenance(
        method="IEC 60287 conduction ladder with external Robin resistance T4",
        citations=(
            "[IEC60287-1-1] 4.2.1 Formula (1), 5.1 to 5.3, Annex A",
            "[IEC60228] Table 4",
            "[IEC60287-2-1] 4.1",
            "[Poradowski2025] Eq.2",
            "[NeherMcGrath1957] Eqs.1-2",
        ),
        gates=(
            "A9 reduction (n=1) <1e-12",
            "profile ends T(r_c)=theta_c, T(r_s)=theta_a+P_total*T4 <1e-12",
        ),
        limitations=tuple(limitations),
    )
    losses = {
        "R0_ohm_m": cond.r20_ohm_m,
        "R_ac_ohm_m": r_ac,
        "y_s": cond.y_s,
        "y_p": cond.y_p,
        "lay_length_factor": cond.lay_factor,
        "W_c_W_m": w_c,
        "W_d_W_m": w_d,
        "lambda1": float(lambda1),
        "lambda2": float(lambda2),
    }
    return ThermalResult(
        r_m,
        t_profile,
        float(theta_c),
        t_armour,
        float(p_total),
        provenance=prov,
        losses=losses,
    )


def _cable_dielectric_loss(
    cable: CableData, *, lay_factor: float = 1.0, include: bool | None = None
) -> tuple[float, str]:
    """Per-core dielectric loss of the first insulation with electrical data, a note.

    [IEC60287-1-1] 5.2: counted from the ``U_0`` of the insulation type
    (``include=None``), always (``True``) or never (``False``).
    """
    voltage = cable.cable.voltage_kV
    if voltage is None or include is False:
        return 0.0, ""
    incomplete: list[tuple[str, str]] = []  # (layer, insulation type) lacking data
    for layer in cable.layers:
        elec = getattr(layer, "electrical", None)
        r_in = getattr(layer, "r_inner_m", None)
        r_out = getattr(layer, "r_outer_m", None)
        if elec is None:
            continue
        tan_d = elec.loss_factor_tan_delta
        eps_r = elec.permittivity_relative
        if not (tan_d and eps_r) or r_in is None or r_out is None:
            if elec.insulation_type:
                incomplete.append((layer.name, elec.insulation_type))
            continue
        kind = elec.insulation_type or "xlpe"
        w_d = dielectric_loss(
            voltage,
            eps_r,
            tan_d,
            r_in,
            r_out,
            frequency_Hz=cable.environment.frequency_Hz,
            insulation_type=kind,
            lay_length_factor=lay_factor,
            include=include,
        )
        u0 = voltage / math.sqrt(3.0)
        if include is True:
            basis = "always included (CIGRE TB 880 Guidance Point 7)"
        else:
            threshold = DIELECTRIC_LOSS_U0_KV[kind]
            side = "at or above" if w_d > 0.0 else "below"
            basis = (
                f"U0 = {u0:.1f} kV {side} the {threshold:g} kV of "
                f"{kind}{'' if elec.insulation_type else ' (assumed)'} "
                "([IEC60287-1-1] 5.2)"
            )
        return w_d, f"dielectric loss W_d = {w_d:.4g} W/m per core: {basis}"
    # The threshold of the declared insulation type of a layer without the
    # data, else that of XLPE (the default type).
    kinds = [kind for _, kind in incomplete] or ["xlpe"]
    kind = min(kinds, key=lambda k: DIELECTRIC_LOSS_U0_KV[k])
    threshold = DIELECTRIC_LOSS_U0_KV[kind]
    needed = include is True or voltage / math.sqrt(3.0) >= threshold
    if not needed:
        return 0.0, ""
    # Where the dielectric loss may count, a silent zero would understate the
    # conductor temperature.
    declared = (
        "; "
        + ", ".join(
            f"{name!r} declares {k} without both values" for name, k in incomplete
        )
        if incomplete
        else ""
    )
    reason = (
        "the dielectric loss is requested (include_dielectric_loss=True)"
        if include is True
        else (
            f"U0 = {voltage / math.sqrt(3.0):.1f} kV reaches the {threshold:g} kV "
            f"of {kind} from which the dielectric loss counts ([IEC60287-1-1] 5.2)"
        )
    )
    raise CableInputError(
        f"the cable is rated {voltage:g} kV and {reason}, but no insulation "
        "layer gives 'electrical: {permittivity_relative, loss_factor_tan_delta}' "
        "with its radii" + declared
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
    # Laid-up cores: a per-core thermal resistance per metre of cable is the
    # straight-core value divided by C_LL ([IEC60287-2-1] 4.1.2.1, 4.1.3).
    c_ll, _ = lay_length_factor(cable)
    # T1 starts at the conductor diameter d_c ([IEC60287-2-1] 4.1.2.1), not
    # at the radius of an area-equivalent conductor.
    core = _conductor_offset_layer(cable)
    d_c = getattr(core, "d_conductor_m", None) if core is not None else None
    r_env = 0.5 * float(d_c) if d_c else 0.0

    # Concentric conduction annuli (layers exposing r_inner_m/r_outer_m) and
    # metallic sheath candidates (r_inner, per_core).
    annuli: list[_LadderAnnulus] = []
    sheath_candidates: list[tuple[float, bool, str]] = []
    geometry_notes: list[str] = []
    if c_ll != 1.0:
        geometry_notes.append(
            f"per-core thermal resistances divided by the lay-up factor C_LL = "
            f"{c_ll:.4f} ([IEC60287-2-1] 4.1.2.1, 4.1.3)"
        )
    for layer in cable.layers:
        name = layer.name.lower()
        per_core = multi_core and layer.layer_type == "cylindrical_offset"
        if layer.layer_type == "cylindrical_shell":
            r_mid = getattr(layer, "r_mid_m", None)
            t_shell = getattr(layer, "t_m", None)
            if r_mid is not None and t_shell is not None:
                sheath_candidates.append((r_mid - 0.5 * t_shell, False, layer.name))
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
                f"{r_eq * 1e3:.2f} mm to {r_out * 1e3:.2f} mm (n = {n_geo}) "
                "for its heat capacity; the SL-type geometric factor sets its "
                "resistance where it applies"
            )
            r_in = r_eq
        core_stack = per_core or not multi_core
        if core_stack and r_env > 0.0 and r_in < r_env - _RADIUS_TOL_M:
            if r_out <= r_env + _RADIUS_TOL_M:
                geometry_notes.append(
                    f"{layer.name!r} lies inside the conductor diameter d_c = "
                    f"{2e3 * r_env:.2f} mm (stranding voids): no thermal "
                    "resistance ([IEC60287-2-1] 4.1.2.1)"
                )
                continue
            geometry_notes.append(
                f"{layer.name!r} starts at the conductor diameter d_c = "
                f"{2e3 * r_env:.2f} mm, not at {2e3 * r_in:.2f} mm "
                "([IEC60287-2-1] 4.1.2.1)"
            )
            r_in = r_env
        metallic = lam >= _METALLIC_LAMBDA_MIN_W_mK
        annuli.append(
            _LadderAnnulus(
                ConductionLayer(r_in, r_out, lam * c_ll if per_core else lam),
                name,
                per_core,
                metallic,
                layer_name=layer.name,
                material=layer.material,
            )
        )
        if metallic:
            sheath_candidates.append((r_in, per_core, layer.name))
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
    sheath_name, sheath_per_core = "", False
    if r_ins_out is not None:
        ins_out = r_ins_out

        def _outside_insulation(r_inner: float, per_core: bool) -> bool:
            if per_core == ins_per_core:
                return r_inner >= ins_out - _RADIUS_TOL_M
            return not per_core  # cable-level layers lie outside the cores

        candidates = [c for c in sheath_candidates if _outside_insulation(c[0], c[1])]
        preferred = [c for c in candidates if c[1]] or candidates
        if preferred:
            r_sheath, sheath_per_core, sheath_name = min(preferred)
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

    sl_type = sheath_found and multi_core and bool(sheath_per_core)
    if sl_type:
        path = _sl_filler_path(cable, path, n_geo, r_core, geometry_notes)
        screen = next((x for x in cable.layers if x.name == sheath_name), None)
        if getattr(screen, "construction", "") == "resolved_wire_screen":
            geometry_notes.append(
                f"core screen {sheath_name!r} is a wire screen: [IEC60287-2-1] "
                "4.1.2.2.4 treats a three-core cable with screens of spaced "
                "wires as belted (T1 from the belted geometric factor); this "
                "ladder takes the screen as an isotherm, which gives a lower T1"
            )
    return _LadderGeometry(
        path,
        in_t1,
        n,
        n_geo,
        multi_core,
        r_core,
        geometry_notes,
        sheath_found,
        sheath_name if sheath_found else "",
        bool(sheath_per_core) if sheath_found else False,
        c_ll,
    )


def sl_filler_geometric_factor(x: float, *, sheaths_touching: bool = True) -> float:
    """Geometric factor ``G-bar`` of SL-type fillers ([IEC60287-2-1] 5.6).

    ``x`` is the thickness of material between the sheaths (or core jackets)
    and the armour as a fraction of their outer diameter, ``0 < x <= 0.15``.
    The upper curve of Figure 6 is for sheaths touching, the lower one for
    equal thicknesses between the sheaths and between sheath and armour;
    CIGRE TB 880 (2022) Guidance Point 45 uses the upper curve. Each is
    ``2 pi`` times the polynomial of [IEC60287-2-1] 5.6 (error below 1 %).
    """
    if not 0.0 < x <= 0.15:
        raise CableInputError(
            f"the filler geometric factor is given for 0 < x <= 0.15, not {x:g}"
        )
    if sheaths_touching:
        poly = (
            0.00022619 + 2.11429 * x - 20.4762 * x**2
            if x <= 0.03
            else 0.0142108 + 1.17533 * x - 4.49737 * x**2 + 10.6352 * x**3
        )
    else:
        poly = (
            0.000202380 + 2.03214 * x - 21.6667 * x**2
            if x <= 0.03
            else 0.0126529 + 1.101 * x - 4.56104 * x**2 + 11.5093 * x**3
        )
    return 2.0 * math.pi * poly


def _sl_filler_path(
    cable: CableData,
    path: list[_LadderAnnulus],
    n_geo: int,
    r_core: float,
    notes: list[str],
) -> list[_LadderAnnulus]:
    """Scale the filler and bedding under the armour to ``rho/(6 pi) G-bar``.

    [IEC60287-2-1] 4.1.4.2 and 5.6 with CIGRE TB 880 (2022) Guidance Point
    45: in a three-core SL-type cable the material between the core jackets
    and the armour (fillers, binders, an inner sheath) has the thermal
    resistance ``T2 = rho/(6 pi) G-bar`` per cable, ``rho`` the highest
    resistivity of these materials and ``x`` the smallest distance between a
    core jacket and the armour over the core diameter. The annuli keep their
    geometry (and heat capacity); their resistances are scaled in proportion
    so that they sum to ``T2``. Without armour, with other than three cores or
    outside ``0 < x <= 0.15`` the annuli stay as they are, recorded.
    """
    armour = [a for a in cable.helical_layers if a.structural and a.r_helix_m]
    if n_geo != 3 or not armour:
        notes.append(
            "SL-type filler geometric factor ([IEC60287-2-1] 4.1.4.2) not "
            "applied: it is given for an armoured three-core cable"
        )
        return path
    r_arm = min(
        float(a.r_helix_m or 0.0)
        - 0.5 * float(getattr(a, "wire_thickness_m", None) or a.d_wire_m or 0.0)
        for a in armour
    )
    core = _conductor_offset_layer(cable)
    r_offset = float(getattr(core, "r_offset_m", None) or 0.0) if core else 0.0
    region = [
        i
        for i, a in enumerate(path)
        if not a.per_core and not a.metallic and a.layer.r_outer_m <= r_arm + 1e-9
    ]
    x = (r_arm - r_offset - r_core) / (2.0 * r_core) if r_core > 0.0 else 0.0
    if not region or not 0.0 < x <= 0.15:
        notes.append(
            f"SL-type filler geometric factor ([IEC60287-2-1] 5.6) not applied: "
            f"x = {x:.4f} is outside 0 < x <= 0.15 or no filler lies under the "
            "armour; the annuli are kept"
        )
        return path
    touching = 2.0 * r_offset * math.sin(math.pi / 3.0) <= 2.0 * r_core * 1.01
    g_bar = sl_filler_geometric_factor(x, sheaths_touching=touching)
    rho = max(1.0 / path[i].layer.lambda_W_mK for i in region)
    t2 = rho / (6.0 * math.pi) * g_bar
    current = sum(
        thermal_resistance(
            path[i].layer.r_inner_m, path[i].layer.r_outer_m, path[i].layer.lambda_W_mK
        )
        for i in region
    )
    scale = t2 / current
    new_path = list(path)
    for i in region:
        a = path[i]
        new_path[i] = _LadderAnnulus(
            ConductionLayer(
                a.layer.r_inner_m,
                a.layer.r_outer_m,
                a.layer.lambda_W_mK / scale,
                a.layer.rho_cp_J_m3K,
            ),
            a.name,
            a.per_core,
            a.metallic,
            layer_name=a.layer_name,
            material=a.material,
        )
    names = ", ".join(repr(path[i].layer_name) for i in region)
    curve = "sheaths touching" if touching else "equal thicknesses"
    notes.append(
        f"fillers and bedding under the armour ({names}) carry T2 = rho/(6 pi) "
        f"G-bar = {t2:.4f} K m/W ([IEC60287-2-1] 4.1.4.2, 5.6, {curve}; "
        f"CIGRE TB 880 GP 45): x = {x:.4f}, G-bar = {g_bar:.4f}, rho = "
        f"{rho:.2f} K m/W (highest of these materials); the equal-area annuli "
        f"gave {current:.4f} K m/W"
    )
    return new_path


# Public entry points read the cable's own materials when called directly.
_scope_entry_points(globals(), __all__)
