# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Ultimate-limit-state screens for armour in axial compression.

Two instability modes of the tensile armour are screened, together with wire
yield, for a cable whose effective (or true wall) tension is negative.

Lateral buckling
    Transverse instability of the wires on their supporting cylinder after
    cyclic bending has removed the friction restraint. The lower bound is the
    friction-free, buckling-length-independent capacity of [SaevikThorsen2017]
    Eq. 19 for each layer:

    .. math::
        P_1 = n\frac{\cos α}{R^2}\bigl[GJ\sin^4 α
              + (4EI_2 + EI_3 - GJ)\sin^2 α\cos^2 α\bigr].

    For round wires the analytical limit of [LiVaz2021a] Eq. 11 (after Li and
    Vaz 2019) is also evaluated,

    .. math::
        P_{cr} = n\frac{EπD^4\sin^2 2φ\cos φ}{64 r^2}
                 \Bigl(1 + \frac{2G\cos^2 φ}{E\sin^2 φ + 2G}\Bigr),

    with the cyclic-curvature knock-down ``P*_cr/P_cr`` of the finite-element
    study [LiVaz2021b] Table 3 (0.91, 0.88, 0.83 at ``κ_max r`` = 0.0036,
    0.0072, 0.0107). The lateral capacity is the smaller of ``P_1`` and the
    knocked-down ``P_cr``. Friction enters as a gate: the wires can migrate
    laterally only if the cyclic curvature exceeds the first-slip curvature at
    the neutral axis ([LiVaz2021a] Eqs. 12–15),

    .. math::
        κ_{slip} = \frac{μ\,(Q\sin^2 α / r + q_{ext})}{E A \sin α\cos^2 α},

    evaluated at the lateral capacity, with ``Q`` the helix-direction wire
    compression and ``q_ext`` the line load from a sealed annulus. Above it
    the capacity does not depend on ``μ`` ([LiVaz2021a] Table 4: 0.83–0.85
    ``P_cr`` for μ = 0.05–0.15).

Radial (bird-cage) buckling
    Buckling of the armour stack as a beam on the elastic foundation of the
    outboard hoop layers, [SaevikThorsen2017] Eq. 22 (straight beam,
    default and conservative) or Eqs. 29–32 (curved beam), with the
    foundation stiffness of each sheath (Eq. 23) and high-strength tape
    (Eq. 24). It interacts with the hoop-capacity failure ``P_3`` (Eq. 33)
    and wire yield ``P_4`` (Eq. 34) through Eq. 35,
    ``1/P_2^* = 1/P_2 + 1/P_3 + 1/P_4``. For layers of different lay angles
    the hoop demand is summed over the layers with their axial load shares,
    ``P_3 = 2πH / Σ_i s_i \tan^2 α_i / R_i``, which reduces to Eq. 33 for two
    equal and opposite layers. ``P_3`` needs a sourced hoop strength of the
    outboard sheath or tape; without one it is unavailable (``inf``) and the
    result says so.

Load path and conventions
    Tension is positive. A flooded annulus is driven by the effective tension;
    a sealed (dry) annulus by the true wall tension ``T_w = T_e - p_e A_e``
    ([SaevikThorsen2017] Eq. 1 with no bore). The cable force is shared
    among the layers by equal axial strain (the EA-weighted fixed-twist load
    sharing used by the stress and capacity paths). Capacities are reported
    as armour axial forces and as the cable axial compression that produces
    them.

Design factor
    No power-cable code gives a compression capacity. The default design
    utilisation factors are those of DNV-RP-F401 Table 3-1 for steel armour
    (0.67 normal operation, 0.78 installation, 1.00 abnormal), here applied
    to the buckling capacity as well as to yield. That transfer is an
    assumption of this screen.

References
----------
[SaevikThorsen2017] S. Sævik, M.J. Thorsen, J. Offshore Mech. Arct. Eng.
139(4):041701 (2017), Eqs. 1, 19, 22–24, 29–36.
[LiVaz2021a] X. Li, M.A. Vaz, Ocean Eng. 220:108306 (2021), Eqs. 11–15,
Tables 2 and 4.
[LiVaz2021b] X. Li, M.A. Vaz, Mar. Struct. 76:102895 (2021), Eq. 12, Table 3.
[Ostergaard2012] N.H. Østergaard et al., Mar. Struct. 27:64–81 (2012).
[DNV-RP-F401] Electrical power cables in subsea applications (2019),
Table 3-1.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from cablexsec.cables.schema import (
    CableData,
    CylindricalLayer,
    CylindricalOffsetLayer,
    CylindricalShellLayer,
    HelicalLayer,
    HelicalStrandedLayer,
)
from cablexsec.cables.schema import scope_entry_points as _scope_entry_points
from cablexsec.exceptions import CableInputError
from cablexsec.materials import library
from cablexsec.mechanics.stiffness import axial_stiffness_by_layer
from cablexsec.provenance import Provenance, ValidatedRange, check_range

__all__ = [
    "ARMOUR_UTILISATION_FACTORS",
    "missing_armour_message",
    "LI_VAZ_KNOCKDOWN_TABLE",
    "CompressionCapacityResult",
    "CompressionScreenResult",
    "LateralBucklingLayer",
    "RadialBucklingStack",
    "TapeLayer",
    "WireSection",
    "combined_radial_load",
    "compression_screen",
    "compute_compression_capacity",
    "cyclic_curvature_knockdown",
    "first_slip_curvature",
    "lateral_buckling_load",
    "lateral_buckling_load_round",
    "radial_buckling_load_curved",
    "radial_buckling_load_straight",
    "radial_capacity_failure_load",
    "rectangular_wire_section",
    "round_wire_section",
    "sheath_foundation_stiffness",
    "tape_foundation_stiffness",
    "true_wall_tension",
    "wire_yield_load",
]

FloatArray = NDArray[np.float64]
Annulus = Literal["flooded", "sealed"]
RadialModel = Literal["straight", "curved"]

#: DNV-RP-F401 (September 2019, amended September 2021) Sec. 3.8.1, Table 3-1,
#: p. 13: utilisation factors for steel armour (load over the lesser of the
#: yield strength and 90 % of the tensile strength).
ARMOUR_UTILISATION_FACTORS: Mapping[str, float] = {
    "normal": 0.67,
    "installation": 0.78,
    "abnormal": 1.00,
}

#: [LiVaz2021b] Table 3, cyclic curvature from zero: (κ_max r, P*_cr/P_cr).
#: Wire radius r = 35.75 mm, κ_max = 0.1, 0.2 and 0.3 1/m.
LI_VAZ_KNOCKDOWN_TABLE: tuple[tuple[float, float], ...] = (
    (0.1 * 0.03575, 0.91),
    (0.2 * 0.03575, 0.88),
    (0.3 * 0.03575, 0.83),
)

#: Lay angles of the published cases reproduced by the gates (Li and Vaz 20
#: deg; Østergaard pipes 26.2-31.5 deg; Yang 30 deg; de Sousa 35 deg; Malta
#: 36-38 deg). Both lateral closed forms tend to zero with the lay angle.
LAY_ANGLE_RANGE = ValidatedRange(
    name="armour lay angle [deg]",
    low=20.0,
    high=38.0,
    bound_cite="lay angles of the reproduced cases (gates ST17, LV21, YANG15)",
)

_CITATIONS = (
    "[SaevikThorsen2017] Eqs. 1, 19, 22-24, 29-36",
    "[LiVaz2021a] Eqs. 11-15",
    "[LiVaz2021b] Table 3",
    "[DNV-RP-F401] Table 3-1",
)


# --------------------------------------------------------------------------- #
# Wire sections                                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WireSection:
    """Bending and torsion rigidities of one armour wire.

    ``EI2`` bends the wire radially (about the axis along the flat side, the
    weak axis of a flat wire) and ``EI3`` laterally (strong axis), as in
    [SaevikThorsen2017] Eqs. 14–16.

    Attributes
    ----------
    E_Pa, G_Pa : float
        Young's and shear moduli [Pa].
    area_m2 : float
        Cross-section area [m²].
    EI2_Nm2, EI3_Nm2, GJ_Nm2 : float
        Radial and lateral bending rigidities and torsion rigidity [N·m²].
    """

    E_Pa: float
    G_Pa: float
    area_m2: float
    EI2_Nm2: float
    EI3_Nm2: float
    GJ_Nm2: float


def _positive(name: str, value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise CableInputError(f"{name} must be finite and > 0")
    return float(value)


def round_wire_section(E_Pa: float, nu: float, d_m: float) -> WireSection:
    """Round wire: ``I2 = I3 = πd⁴/64``, ``J = 2I``, ``G = E/[2(1+ν)]``."""
    _positive("E_Pa", E_Pa)
    _positive("d_m", d_m)
    if not -1.0 < nu < 0.5:
        raise CableInputError("Poisson ratio must lie in (-1, 0.5)")
    g = E_Pa / (2.0 * (1.0 + nu))
    i = math.pi * d_m**4 / 64.0
    return WireSection(
        E_Pa=E_Pa,
        G_Pa=g,
        area_m2=math.pi * d_m**2 / 4.0,
        EI2_Nm2=E_Pa * i,
        EI3_Nm2=E_Pa * i,
        GJ_Nm2=g * 2.0 * i,
    )


def rectangular_wire_section(
    E_Pa: float, nu: float, thickness_m: float, width_m: float
) -> WireSection:
    """Flat wire of radial ``thickness`` and circumferential ``width``.

    ``I2 = w t³/12`` (radial bending), ``I3 = t w³/12`` (lateral bending) and
    the torsion constant ``J = b c³[1/3 - 0.21 (c/b)(1 - c⁴/(12 b⁴))]`` with
    ``b ≥ c`` the long and short sides (Roark's approximation; the source
    papers do not state theirs).
    """
    _positive("E_Pa", E_Pa)
    _positive("thickness_m", thickness_m)
    _positive("width_m", width_m)
    if not -1.0 < nu < 0.5:
        raise CableInputError("Poisson ratio must lie in (-1, 0.5)")
    g = E_Pa / (2.0 * (1.0 + nu))
    b, c = max(thickness_m, width_m), min(thickness_m, width_m)
    j = b * c**3 * (1.0 / 3.0 - 0.21 * (c / b) * (1.0 - c**4 / (12.0 * b**4)))
    return WireSection(
        E_Pa=E_Pa,
        G_Pa=g,
        area_m2=thickness_m * width_m,
        EI2_Nm2=E_Pa * width_m * thickness_m**3 / 12.0,
        EI3_Nm2=E_Pa * thickness_m * width_m**3 / 12.0,
        GJ_Nm2=g * j,
    )


# --------------------------------------------------------------------------- #
# Closed forms                                                                #
# --------------------------------------------------------------------------- #
def lateral_buckling_load(
    n_wires: float, r_helix_m: float, lay_angle_rad: float, section: WireSection
) -> float:
    r"""Friction-free lateral buckling lower bound ``P_1`` [N] of one layer.

    [SaevikThorsen2017] Eq. 19, the axial force of ``n_wires`` wires on a
    cylinder of radius ``r_helix_m``; ``inf`` at zero lay angle.
    """
    _positive("n_wires", n_wires)
    _positive("r_helix_m", r_helix_m)
    if math.isclose(lay_angle_rad, 0.0, abs_tol=1e-15):
        return math.inf
    s, c = math.sin(lay_angle_rad), math.cos(lay_angle_rad)
    gj, ei2, ei3 = section.GJ_Nm2, section.EI2_Nm2, section.EI3_Nm2
    bracket = gj * s**4 + (4.0 * ei2 + ei3 - gj) * s**2 * c**2
    return float(n_wires * c / r_helix_m**2 * bracket)


def lateral_buckling_load_round(
    n_wires: float, r_helix_m: float, lay_angle_rad: float, section: WireSection
) -> float:
    r"""Round-wire lateral buckling limit [N] of [LiVaz2021a] Eq. 11.

    ``n_wires`` times the single-wire limit in the cable axial direction.
    Valid for circular wires (``EI2 = EI3``); ``inf`` at zero lay angle.
    """
    _positive("n_wires", n_wires)
    _positive("r_helix_m", r_helix_m)
    if not math.isclose(section.EI2_Nm2, section.EI3_Nm2, rel_tol=1e-12):
        raise CableInputError("the Li-Vaz limit applies to round wires only")
    if math.isclose(lay_angle_rad, 0.0, abs_tol=1e-15):
        return math.inf
    e, g, ei = section.E_Pa, section.G_Pa, section.EI2_Nm2
    phi = lay_angle_rad
    base = ei * math.sin(2.0 * phi) ** 2 * math.cos(phi) / r_helix_m**2
    factor = 1.0 + 2.0 * g * math.cos(phi) ** 2 / (e * math.sin(phi) ** 2 + 2.0 * g)
    return float(n_wires * base * factor)


def cyclic_curvature_knockdown(
    kappa_1_m: float, r_helix_m: float
) -> tuple[float, bool]:
    """Knock-down ``P*_cr/P_cr`` and whether ``κ r`` lies outside the table.

    Linear interpolation in ``κ_max r`` of [LiVaz2021b] Table 3. Below the
    first point the first value (0.91) is held, not raised towards one; above
    the last point the last value (0.83) is held and the flag is ``True``,
    because the source shows the ratio still falling there.
    """
    if not math.isfinite(kappa_1_m) or kappa_1_m < 0.0:
        raise CableInputError("curvature must be finite and >= 0")
    _positive("r_helix_m", r_helix_m)
    x = kappa_1_m * r_helix_m
    xs = [p[0] for p in LI_VAZ_KNOCKDOWN_TABLE]
    ys = [p[1] for p in LI_VAZ_KNOCKDOWN_TABLE]
    return float(np.interp(x, xs, ys)), x > xs[-1] * (1.0 + 1e-12)


def first_slip_curvature(
    mu: float,
    wire_compression_N: float,
    lay_angle_rad: float,
    section: WireSection,
    r_helix_m: float,
    *,
    external_line_load_N_m: float = 0.0,
) -> float:
    r"""Curvature at first lateral slip of a compressed wire [1/m].

    [LiVaz2021a] Eqs. 12–15 generalised to any section and an external
    normal line load per unit wire length: friction ``μ p_n`` with
    ``p_n = Q sin²α / r + q_ext`` balances the neutral-axis strain gradient
    ``E A κ sin α cos² α``. Zero friction gives zero; zero lay angle gives
    ``inf``.
    """
    if not math.isfinite(mu) or mu < 0.0:
        raise CableInputError("friction coefficient must be finite and >= 0")
    if wire_compression_N < 0.0 or external_line_load_N_m < 0.0:
        raise CableInputError("wire compression and line load must be >= 0")
    _positive("r_helix_m", r_helix_m)
    s, c = math.sin(lay_angle_rad), math.cos(lay_angle_rad)
    if math.isclose(s, 0.0, abs_tol=1e-15):
        return math.inf
    p_n = wire_compression_N * s**2 / r_helix_m + external_line_load_N_m
    return float(mu * p_n / (section.E_Pa * section.area_m2 * s * c**2))


def sheath_foundation_stiffness(
    n_wires: float, r_m: float, lay_angle_rad: float, E_s_Pa: float, t_s_m: float
) -> float:
    """Radial foundation stiffness of a sheath per wire [Pa], Eq. 23."""
    _positive("n_wires", n_wires)
    _positive("r_m", r_m)
    if E_s_Pa < 0.0 or t_s_m < 0.0:
        raise CableInputError("sheath modulus and thickness must be >= 0")
    return float(
        2.0 * math.pi / n_wires * E_s_Pa * t_s_m / r_m * math.cos(lay_angle_rad)
    )


def tape_foundation_stiffness(
    n_wires: float,
    r_m: float,
    lay_angle_rad: float,
    n_plies: float,
    EA_tape_N: float,
    tape_angle_rad: float,
) -> float:
    """Radial foundation stiffness of a tape per wire [Pa], Eq. 24."""
    _positive("n_wires", n_wires)
    _positive("r_m", r_m)
    ct = math.cos(tape_angle_rad)
    if math.isclose(ct, 0.0, abs_tol=1e-15):
        raise CableInputError("tape lay angle must be below 90 degrees")
    return float(
        n_plies
        / n_wires
        * EA_tape_N
        * math.sin(tape_angle_rad) ** 4
        / r_m**2
        * math.cos(lay_angle_rad)
        / ct
    )


def radial_buckling_load_straight(
    n_wires: float, lay_angle_rad: float, section: WireSection, c_Pa: float
) -> float:
    """Straight-beam radial buckling load ``P_2 = 2n cos α √(EI_2 c)`` [N], Eq. 22."""
    if c_Pa <= 0.0:
        return math.inf
    return float(
        2.0 * n_wires * math.cos(lay_angle_rad) * math.sqrt(section.EI2_Nm2 * c_Pa)
    )


def radial_buckling_load_curved(
    n_wires: float,
    r_helix_m: float,
    lay_angle_rad: float,
    section: WireSection,
    c_Pa: float,
) -> tuple[float, float]:
    r"""Curved-beam radial buckling load [N] and half-wavelength [m].

    [SaevikThorsen2017] Eqs. 29–32. With ``k = π m/l`` the critical wire
    force is ``EI_2 (k⁴ - a_1 k² + a_2)/(k² - a_3)``; its minimum lies at
    ``k² = a_3 + √(a_3² - a_1 a_3 + a_2)`` (Eq. 31) where it equals
    ``EI_2 (2k² - a_1)``. Returns ``(inf, inf)`` without a foundation.
    """
    if c_Pa <= 0.0:
        return math.inf, math.inf
    _positive("r_helix_m", r_helix_m)
    s, c = math.sin(lay_angle_rad), math.cos(lay_angle_rad)
    k1 = s * c / r_helix_m
    k2 = s**2 / r_helix_m
    ei2, ei3, gj = section.EI2_Nm2, section.EI3_Nm2, section.GJ_Nm2
    a1 = k2**2 - 2.0 * k1**2 - 4.0 * ei3 / ei2 * k1**2
    a2 = c_Pa / ei2 + 2.0 * gj / ei2 * k1**2 * k2**2 + k1**4 - k1**2 * k2**2
    a3 = k2**2 - k1**2
    disc = a3**2 - a1 * a3 + a2
    y = a3 + math.sqrt(disc)
    if y <= 0.0:  # pragma: no cover - not reached for c > 0 and real geometry
        raise CableInputError("curved-beam radial mode has no positive wavenumber")
    q_cr = ei2 * (2.0 * y - a1)
    return float(n_wires * c * q_cr), float(math.pi / math.sqrt(y))


def radial_capacity_failure_load(
    hoop_strength_N_m: float, tan2_over_r_1_m: float
) -> float:
    r"""Hoop-capacity (bird-cage) failure load ``P_3`` [N].

    ``P_3 = 2π H / Σ_i s_i tan²α_i / R_i`` with ``H`` the hoop force capacity
    per unit length of the outboard layers and ``s_i`` the armour load share
    of each layer. For one radius and lay angle this is [SaevikThorsen2017]
    Eq. 33, ``R/tan²α (n_t σ_ut A_t sin²α_t/(cos α_t R_t) + 2π σ_s t_s)``.
    Returns ``inf`` when no hoop capacity is available.
    """
    if hoop_strength_N_m <= 0.0 or tan2_over_r_1_m <= 0.0:
        return math.inf
    return float(2.0 * math.pi * hoop_strength_N_m / tan2_over_r_1_m)


def wire_yield_load(
    n_wires: float, lay_angle_rad: float, sigma_y_Pa: float, area_m2: float
) -> float:
    """Axial force at wire yield ``P_4 = n cos α σ_y A`` [N], Eq. 34."""
    return float(n_wires * math.cos(lay_angle_rad) * sigma_y_Pa * area_m2)


def combined_radial_load(p2_N: float, p3_N: float, p4_N: float) -> float:
    """Interaction ``1/P_2^* = 1/P_2 + 1/P_3 + 1/P_4`` [N], Eq. 35.

    Infinite terms (unavailable or unrestrained modes) drop out.
    """
    if any(p <= 0.0 for p in (p2_N, p3_N, p4_N)):
        raise CableInputError("capacities must be > 0")
    total = sum(1.0 / p for p in (p2_N, p3_N, p4_N) if math.isfinite(p))
    return math.inf if total == 0.0 else 1.0 / total


def true_wall_tension(
    effective_tension_N: float,
    p_ext_Pa: float,
    area_ext_m2: float,
    p_int_Pa: float = 0.0,
    area_int_m2: float = 0.0,
) -> float:
    """True wall tension ``T_w = T_e - p_e A_e + p_i A_i`` [N], Eq. 1."""
    return float(effective_tension_N - p_ext_Pa * area_ext_m2 + p_int_Pa * area_int_m2)


# --------------------------------------------------------------------------- #
# Cable-level screen                                                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TapeLayer:
    """A high-strength tape outboard of the armour (rare in power cables).

    Attributes
    ----------
    name : str
        Label.
    E_Pa : float
        Tape modulus along the tape [Pa].
    area_m2 : float
        Cross-section of one ply (thickness × width) [m²].
    lay_angle_rad : float
        Tape lay angle from the cable axis (close to 90°) [rad].
    n_plies : int
        Number of plies (windings) ``n_t``.
    r_m : float
        Tape radius ``R_t`` [m].
    sigma_ult_Pa : float
        Ultimate tensile stress ``σ_ut`` [Pa].
    source : str
        Citation or test record for the tape properties.
    """

    name: str
    E_Pa: float
    area_m2: float
    lay_angle_rad: float
    n_plies: int
    r_m: float
    sigma_ult_Pa: float
    source: str

    def __post_init__(self) -> None:
        for key in ("E_Pa", "area_m2", "r_m", "sigma_ult_Pa"):
            _positive(key, float(getattr(self, key)))
        if self.n_plies < 1:
            raise CableInputError("tape needs at least one ply")
        if not 0.0 < self.lay_angle_rad < math.pi / 2.0:
            raise CableInputError("tape lay angle must lie in (0, 90) degrees")
        if not self.source.strip():
            raise CableInputError("tape properties need a source")

    @property
    def ultimate_strain(self) -> float:
        """Tape failure strain ``σ_ut / E_t``."""
        return self.sigma_ult_Pa / self.E_Pa

    @property
    def hoop_strength_N_m(self) -> float:
        """Hoop force capacity per unit length, ``n_t σ A sin²α_t/(2π cos α_t R_t)``."""
        a = self.lay_angle_rad
        return (
            self.n_plies
            * self.sigma_ult_Pa
            * self.area_m2
            * math.sin(a) ** 2
            / (2.0 * math.pi * math.cos(a) * self.r_m)
        )


@dataclass(frozen=True)
class LateralBucklingLayer:
    """Lateral buckling screen of one armour layer.

    Attributes
    ----------
    layer : str
        Armour layer name.
    load_share : float
        Fraction of the cable axial force carried by the layer.
    demand_N : float
        Layer axial compression (``>= 0``) [N].
    P1_saevik_N : float
        Friction-free lower bound, Eq. 19 [N].
    P_li_vaz_N : float
        Round-wire limit of [LiVaz2021a] Eq. 11 [N].
    knockdown : float
        [LiVaz2021b] cyclic-curvature factor applied to ``P_li_vaz_N``.
    knockdown_beyond_table : bool
        ``κ r`` above the tabulated range (last value held).
    capacity_N : float
        ``min(P1, knockdown × P_li_vaz)`` [N].
    first_slip_curvature_1_m : float
        Neutral-axis first-slip curvature at the capacity load [1/m].
    active : bool
        Cyclic curvature reaches first slip (``True`` when no curvature given).
    utilisation : float
        ``demand / capacity`` when active, else 0.
    """

    layer: str
    load_share: float
    demand_N: float
    P1_saevik_N: float
    P_li_vaz_N: float
    knockdown: float
    knockdown_beyond_table: bool
    capacity_N: float
    first_slip_curvature_1_m: float
    active: bool
    utilisation: float


@dataclass(frozen=True)
class RadialBucklingStack:
    """Radial (bird-cage) screen of the armour stack.

    Attributes
    ----------
    layers : tuple[str, ...]
        Armour layers of the stack.
    restraint_layers : tuple[str, ...]
        Outboard continuous annuli and tapes providing the foundation.
    n_wires_total : int
        Wires of all stack layers.
    mean_radius_m : float
        Wire-count weighted mean helix radius [m].
    foundation_c_Pa : tuple[float, ...]
        Foundation stiffness per wire of each layer (Eqs. 23–24) [Pa].
    radial_model : str
        ``"straight"`` (Eq. 22) or ``"curved"`` (Eqs. 29–32).
    P2_N : float
        Elastic radial buckling load [N] (``inf`` without restraint).
    half_wavelength_m : float
        Buckle half-wavelength of the first layer [m].
    P3_N : float
        Hoop-capacity failure load [N] (``inf`` when no strength given).
    P4_N : float
        Wire yield load [N].
    P2_star_N : float
        Interaction capacity, Eq. 35 [N].
    demand_N : float
        Armour axial compression [N].
    utilisation : float
        ``demand / P2_star``.
    """

    layers: tuple[str, ...]
    restraint_layers: tuple[str, ...]
    n_wires_total: int
    mean_radius_m: float
    foundation_c_Pa: tuple[float, ...]
    radial_model: str
    P2_N: float
    half_wavelength_m: float
    P3_N: float
    P4_N: float
    P2_star_N: float
    demand_N: float
    utilisation: float


@dataclass(frozen=True)
class CompressionScreenResult:
    """Armour compression screen at one cable tension and curvature.

    Attributes
    ----------
    tension_N : float
        Input effective tension (positive in tension) [N].
    driving_tension_N : float
        Tension that loads the armour: effective (flooded) or true wall
        (sealed) [N].
    curvature_1_m : float or None
        Maximum cyclic curvature from straight; ``None`` = not specified
        (lateral mode taken as active, lowest tabulated knock-down).
    annulus : str
        ``"flooded"`` or ``"sealed"``.
    friction_coefficient : float
        ``μ`` used in the first-slip gate.
    armour_share : float
        Fraction of the cable axial force carried by the armour stack.
    lateral : tuple[LateralBucklingLayer, ...]
        Per-layer lateral screens.
    radial : RadialBucklingStack
        Stack radial screen.
    utilisation : float
        Largest characteristic utilisation (demand / capacity).
    governing : str
        ``"lateral:<layer>"``, ``"radial"`` or ``"none"`` (no compression).
    compression_capacity_N : float
        Cable axial compression at characteristic utilisation one [N].
    utilisation_factor : float
        Design utilisation factor η (allowable = η × capacity).
    design_utilisation : float
        ``utilisation / η``.
    allowable_compression_N : float
        ``η × compression_capacity_N`` [N].
    margin_N : float
        ``allowable_compression_N + driving_tension_N``; positive = within
        the allowable [N].
    assumptions : tuple[str, ...]
        Inputs that are assumptions rather than cable data.
    provenance : Provenance
        Method, sources, gates and limitations.
    """

    tension_N: float
    driving_tension_N: float
    curvature_1_m: float | None
    annulus: str
    friction_coefficient: float
    armour_share: float
    lateral: tuple[LateralBucklingLayer, ...]
    radial: RadialBucklingStack
    utilisation: float
    governing: str
    compression_capacity_N: float
    utilisation_factor: float
    design_utilisation: float
    allowable_compression_N: float
    margin_N: float
    assumptions: tuple[str, ...]
    provenance: Provenance


_ARMOUR_GEOMETRY = (
    ("n_wires", "wire count"),
    ("d_wire_m", "wire diameter"),
    ("r_helix_m", "helix radius"),
    ("lay_angle_rad", "lay angle"),
)


def _armour_gaps(layer: HelicalLayer) -> list[str]:
    """What an armour layer lacks for the capacity checks, in words."""
    gaps = [words for name, words in _ARMOUR_GEOMETRY if getattr(layer, name) is None]
    try:
        library.get_property(layer.material, "sigma_y_Pa")
    except KeyError:
        gaps.append(f"a published yield stress of {layer.material}")
    return gaps


def missing_armour_message(cable: CableData) -> str:
    """Why no armour layer of ``cable`` can enter the capacity checks.

    The checks need an armour layer with its wire count, wire diameter, helix
    radius and lay angle, of a material with a published yield stress. The
    message names what each armour layer lacks.
    """
    need = (
        "the capacity needs an armour layer with its wire count, wire "
        "diameter, helix radius and lay angle, of a material with a published "
        "yield stress"
    )
    layers = cable.helical_layers
    if not layers:
        return f"{need}; the cable has no armour layer"
    gaps = "; ".join(
        f"{layer.name} lacks {', '.join(_armour_gaps(layer))}" for layer in layers
    )
    return f"{need}; no armour layer of this cable has them all ({gaps})"


def _armour_stack(cable: CableData) -> list[HelicalLayer]:
    out = [lyr for lyr in cable.helical_layers if not _armour_gaps(lyr)]
    if not out:
        raise CableInputError(missing_armour_message(cable))
    return out


def _outer_radius(cable: CableData) -> float:
    radii: list[float] = []
    for lyr in cable.layers:
        if isinstance(lyr, CylindricalLayer) and lyr.r_outer_m is not None:
            radii.append(lyr.r_outer_m)
        elif isinstance(lyr, CylindricalShellLayer):
            if lyr.r_mid_m is not None and lyr.t_m is not None:
                radii.append(lyr.r_mid_m + lyr.t_m / 2.0)
        elif isinstance(lyr, HelicalLayer):
            if lyr.r_helix_m is not None and lyr.d_wire_m is not None:
                radii.append(lyr.r_helix_m + lyr.wire_radial_thickness_m / 2.0)
        elif isinstance(lyr, HelicalStrandedLayer):
            if lyr.r_cable_m is not None and lyr.r_strand_m is not None:
                d_strand = lyr.d_strand_m or 0.0
                radii.append(lyr.r_cable_m + lyr.r_strand_m + 0.5 * d_strand)
        elif (
            isinstance(lyr, CylindricalOffsetLayer)
            and lyr.r_offset_m is not None
            and lyr.r_outer_m is not None
        ):
            radii.append(lyr.r_offset_m + lyr.r_outer_m)
    return max(radii)


def _outboard_annuli(
    cable: CableData, r_stack_m: float
) -> list[tuple[str, float, float, float]]:
    """``(name, E, t, r_inner)`` of continuous annuli outside ``r_stack_m``."""
    out: list[tuple[str, float, float, float]] = []
    for lyr in cable.layers:
        if isinstance(lyr, CylindricalLayer):
            if lyr.r_inner_m is None or lyr.r_outer_m is None:
                continue
            r_in, t = lyr.r_inner_m, lyr.r_outer_m - lyr.r_inner_m
        elif isinstance(lyr, CylindricalShellLayer):
            if lyr.r_mid_m is None or lyr.t_m is None:
                continue
            r_in, t = lyr.r_mid_m - lyr.t_m / 2.0, lyr.t_m
        else:
            continue
        if r_in < r_stack_m:
            continue
        out.append((lyr.name, library.get_property(lyr.material, "E_Pa"), t, r_in))
    return out


def _hoop_strengths(
    names: Sequence[str],
    strength: float | Mapping[str, float] | None,
) -> dict[str, float]:
    if strength is None:
        return {}
    if isinstance(strength, Mapping):
        unknown = set(strength) - set(names)
        if unknown:
            raise CableInputError(
                f"sheath strengths given for layers that are not outboard annuli: "
                f"{sorted(unknown)}"
            )
        values = {k: float(v) for k, v in strength.items()}
    else:
        values = dict.fromkeys(names, float(strength))
    for key, value in values.items():
        _positive(f"hoop strength of {key}", value)
    return values


def _check_options(
    annulus: str,
    p_ext_Pa: float,
    radial_model: str,
    utilisation_factor: float,
    sheath_hoop_strength_Pa: float | Mapping[str, float] | None,
    sheath_strength_source: str | None,
    curvature_1_m: float | None,
) -> None:
    if annulus not in ("flooded", "sealed"):
        raise CableInputError("annulus must be 'flooded' or 'sealed'")
    if not math.isfinite(p_ext_Pa) or p_ext_Pa < 0.0:
        raise CableInputError("p_ext_Pa must be finite and >= 0")
    if radial_model not in ("straight", "curved"):
        raise CableInputError("radial_model must be 'straight' or 'curved'")
    if not math.isfinite(utilisation_factor) or not 0.0 < utilisation_factor <= 1.0:
        raise CableInputError("utilisation_factor must lie in (0, 1]")
    if (sheath_hoop_strength_Pa is None) != (sheath_strength_source is None):
        raise CableInputError(
            "sheath_hoop_strength_Pa and sheath_strength_source must be "
            "supplied together"
        )
    if sheath_strength_source is not None and not sheath_strength_source.strip():
        raise CableInputError("sheath_strength_source must be non-empty")
    if curvature_1_m is not None and (
        not math.isfinite(curvature_1_m) or curvature_1_m < 0.0
    ):
        raise CableInputError("curvature_1_m must be finite and >= 0")


@dataclass(frozen=True)
class _Options:
    """Validated screen options shared by the screen and the capacity curve."""

    annulus: str
    p_ext_Pa: float
    mu: float | None
    radial_model: str
    sheath_hoop_strength_Pa: float | Mapping[str, float] | None
    sheath_strength_source: str | None
    tapes: tuple[TapeLayer, ...]
    utilisation_factor: float


def compression_screen(
    cable: CableData,
    tension_N: float,
    curvature_1_m: float | None = None,
    *,
    annulus: Annulus = "flooded",
    p_ext_Pa: float = 0.0,
    mu: float | None = None,
    radial_model: RadialModel = "straight",
    sheath_hoop_strength_Pa: float | Mapping[str, float] | None = None,
    sheath_strength_source: str | None = None,
    tapes: Sequence[TapeLayer] = (),
    utilisation_factor: float = 1.0,
) -> CompressionScreenResult:
    """Lateral and radial buckling screen of the armour at one load state.

    Parameters
    ----------
    cable : CableData
        Cable with fully specified (round-wire) armour layers.
    tension_N : float
        Effective tension, positive in tension [N].
    curvature_1_m : float, optional
        Maximum curvature reached in cyclic bending from straight [1/m].
        ``None`` (default) treats lateral slip as active and applies the
        lowest tabulated knock-down (0.83).
    annulus : {"flooded", "sealed"}
        Flooded: the effective tension loads the armour and external pressure
        gives no clamping. Sealed: the true wall tension
        ``T_e - p_ext π r_o²`` loads the armour and the pressure adds the
        normal line load of the first-slip gate.
    p_ext_Pa : float
        External hydrostatic pressure [Pa] (used only with ``"sealed"``).
    mu : float, optional
        Armour friction coefficient for the first-slip gate (default: the
        armour material ``mu_on_hdpe``).
    radial_model : {"straight", "curved"}
        Elastic radial buckling model, Eq. 22 (default, conservative in the
        source's comparison) or Eqs. 29–32.
    sheath_hoop_strength_Pa : float or mapping, optional
        Hoop strength of the outboard sheath(s) for ``P_3`` (one value for all
        outboard annuli, or per layer name). Must come with
        ``sheath_strength_source``. Without it ``P_3`` is unavailable.
    sheath_strength_source : str, optional
        Citation or test record for the strength.
    tapes : sequence of TapeLayer
        High-strength tapes outboard of the armour (none in most power
        cables).
    utilisation_factor : float
        Design utilisation factor η in (0, 1], e.g.
        ``ARMOUR_UTILISATION_FACTORS["normal"]``.

    Returns
    -------
    CompressionScreenResult
    """
    opts = _Options(
        annulus=annulus,
        p_ext_Pa=p_ext_Pa,
        mu=mu,
        radial_model=radial_model,
        sheath_hoop_strength_Pa=sheath_hoop_strength_Pa,
        sheath_strength_source=sheath_strength_source,
        tapes=tuple(tapes),
        utilisation_factor=utilisation_factor,
    )
    _check_options(
        annulus,
        p_ext_Pa,
        radial_model,
        utilisation_factor,
        sheath_hoop_strength_Pa,
        sheath_strength_source,
        curvature_1_m,
    )
    return _screen(cable, tension_N, curvature_1_m, opts)


def _screen(
    cable: CableData,
    tension_N: float,
    curvature_1_m: float | None,
    opts: _Options,
) -> CompressionScreenResult:
    annulus, p_ext_Pa, mu = opts.annulus, opts.p_ext_Pa, opts.mu
    radial_model, tapes = opts.radial_model, opts.tapes
    sheath_hoop_strength_Pa = opts.sheath_hoop_strength_Pa
    sheath_strength_source = opts.sheath_strength_source
    utilisation_factor = opts.utilisation_factor
    if not math.isfinite(tension_N):
        raise CableInputError("tension_N must be finite")
    stack = _armour_stack(cable)
    r_out = _outer_radius(cable)
    sealed = annulus == "sealed"
    t_drive = (
        true_wall_tension(tension_N, p_ext_Pa, math.pi * r_out**2)
        if sealed
        else float(tension_N)
    )
    compression = -t_drive if t_drive < 0.0 else 0.0

    ea = axial_stiffness_by_layer(cable)
    ea_total = sum(ea.values())
    shares = [ea[lyr.name] / ea_total for lyr in stack]
    armour_share = sum(shares)
    sections: list[WireSection] = []
    for lyr in stack:
        assert lyr.d_wire_m is not None
        e_w = library.get_property(lyr.material, "E_Pa")
        nu_w = library.get_property(lyr.material, "nu")
        if lyr.wire_shape == "round":
            sections.append(round_wire_section(e_w, nu_w, lyr.d_wire_m))
            continue
        # Rectangular or profiled wire (gap G1): its own inertias.
        shape = lyr.wire_section
        g_w = e_w / (2.0 * (1.0 + nu_w))
        sections.append(
            WireSection(
                E_Pa=e_w,
                G_Pa=g_w,
                area_m2=shape.area,
                EI2_Nm2=e_w * shape.I_normal,
                EI3_Nm2=e_w * shape.I_lateral,
                GJ_Nm2=g_w * shape.J,
            )
        )

    # ---- friction coefficient ------------------------------------------------
    if mu is None:
        try:
            mu_eff = library.get_property(stack[0].material, "mu_on_hdpe")
        except KeyError as exc:
            raise CableInputError(
                "no armour friction coefficient in the library; pass mu"
            ) from exc
    else:
        mu_eff = float(mu)
    if not math.isfinite(mu_eff) or mu_eff < 0.0:
        raise CableInputError("mu must be finite and >= 0")

    # ---- lateral, per layer --------------------------------------------------
    outside = [
        lyr.name
        for lyr in stack
        if not check_range(
            math.degrees(float(lyr.lay_angle_rad or 0.0)), LAY_ANGLE_RANGE
        )
    ]
    lateral: list[LateralBucklingLayer] = []
    for lyr, share, sec in zip(stack, shares, sections, strict=True):
        assert lyr.n_wires is not None and lyr.r_helix_m is not None
        assert lyr.lay_angle_rad is not None
        n, r, a = lyr.n_wires, lyr.r_helix_m, lyr.lay_angle_rad
        p1 = lateral_buckling_load(n, r, a, sec)
        p_lv = lateral_buckling_load_round(n, r, a, sec)
        if curvature_1_m is None:
            eta, beyond = LI_VAZ_KNOCKDOWN_TABLE[-1][1], False
        else:
            eta, beyond = cyclic_curvature_knockdown(curvature_1_m, r)
        cap = min(p1, eta * p_lv)
        q_ext = 2.0 * math.pi * r_out * p_ext_Pa * math.cos(a) / n if sealed else 0.0
        q_wire = cap / (n * math.cos(a)) if math.isfinite(cap) else 0.0
        k_slip = first_slip_curvature(
            mu_eff, q_wire, a, sec, r, external_line_load_N_m=q_ext
        )
        active = curvature_1_m is None or curvature_1_m >= k_slip
        demand = share * compression
        uf = demand / cap if (active and math.isfinite(cap)) else 0.0
        lateral.append(
            LateralBucklingLayer(
                layer=lyr.name,
                load_share=share,
                demand_N=demand,
                P1_saevik_N=p1,
                P_li_vaz_N=p_lv,
                knockdown=eta,
                knockdown_beyond_table=beyond,
                capacity_N=cap,
                first_slip_curvature_1_m=k_slip,
                active=active,
                utilisation=uf,
            )
        )

    # ---- radial, armour stack ------------------------------------------------
    radial = _radial_stack(
        cable,
        stack,
        sections,
        shares,
        radial_model,
        sheath_hoop_strength_Pa,
        tapes,
        compression,
    )

    # ---- governing -----------------------------------------------------------
    cap_candidates: dict[str, float] = {}
    for lat in lateral:
        if lat.active and math.isfinite(lat.capacity_N):
            cap_candidates[f"lateral:{lat.layer}"] = lat.capacity_N / lat.load_share
    if math.isfinite(radial.P2_star_N):
        cap_candidates["radial"] = radial.P2_star_N / armour_share
    capacity = min(cap_candidates.values()) if cap_candidates else math.inf
    ufs = {f"lateral:{lat.layer}": lat.utilisation for lat in lateral}
    ufs["radial"] = radial.utilisation
    if compression > 0.0 and cap_candidates:
        governing = min(cap_candidates, key=lambda k: cap_candidates[k])
    else:
        governing = "none"
    utilisation = max(ufs.values())
    allowable = utilisation_factor * capacity

    assumptions = _assumptions(
        outside,
        cable,
        radial,
        mu,
        mu_eff,
        curvature_1_m,
        annulus,
        sheath_hoop_strength_Pa,
        sheath_strength_source,
        tapes,
        lateral,
    )
    return CompressionScreenResult(
        tension_N=float(tension_N),
        driving_tension_N=t_drive,
        curvature_1_m=curvature_1_m,
        annulus=annulus,
        friction_coefficient=mu_eff,
        armour_share=armour_share,
        lateral=tuple(lateral),
        radial=radial,
        utilisation=utilisation,
        governing=governing,
        compression_capacity_N=capacity,
        utilisation_factor=utilisation_factor,
        design_utilisation=utilisation / utilisation_factor,
        allowable_compression_N=allowable,
        margin_N=allowable + t_drive,
        assumptions=assumptions,
        provenance=_provenance(radial_model),
    )


def _radial_stack(
    cable: CableData,
    stack: Sequence[HelicalLayer],
    sections: Sequence[WireSection],
    shares: Sequence[float],
    radial_model: str,
    sheath_hoop_strength_Pa: float | Mapping[str, float] | None,
    tapes: Sequence[TapeLayer],
    compression: float,
) -> RadialBucklingStack:
    n_w = [int(lyr.n_wires or 0) for lyr in stack]
    r_h = [float(lyr.r_helix_m or 0.0) for lyr in stack]
    alphas = [float(lyr.lay_angle_rad or 0.0) for lyr in stack]
    n_tot = sum(n_w)
    r_mean = sum(n * r for n, r in zip(n_w, r_h, strict=True)) / n_tot
    annuli = _outboard_annuli(cable, max(r_h))
    names = [a[0] for a in annuli]
    strengths = _hoop_strengths(names, sheath_hoop_strength_Pa)
    restraint = tuple(names) + tuple(t.name for t in tapes)

    c_layers: list[float] = []
    p2 = 0.0
    half_wave = math.inf
    for k, (sec, a) in enumerate(zip(sections, alphas, strict=True)):
        c = sum(
            sheath_foundation_stiffness(n_tot, r_mean, a, e_s, t_s)
            for _, e_s, t_s, _ in annuli
        )
        c += sum(
            tape_foundation_stiffness(
                n_tot, r_mean, a, t.n_plies, t.E_Pa * t.area_m2, t.lay_angle_rad
            )
            for t in tapes
        )
        c_layers.append(c)
        if radial_model == "curved":
            p2_k, hw = radial_buckling_load_curved(n_w[k], r_mean, a, sec, c)
        else:
            p2_k = radial_buckling_load_straight(n_w[k], a, sec, c)
            hw = math.pi * (sec.EI2_Nm2 / c) ** 0.25 if c > 0.0 else math.inf
        p2 += p2_k
        if k == 0:
            half_wave = hw

    # Hoop capacity: tapes at failure, sheaths at min(E ε_ut, strength).
    hoop = sum(t.hoop_strength_N_m for t in tapes)
    eps_tape = min((t.ultimate_strain for t in tapes), default=math.inf)
    for name, e_s, t_s, _ in annuli:
        if name in strengths:
            hoop += min(e_s * eps_tape, strengths[name]) * t_s
    s_arm = sum(shares)
    tan2_over_r = sum(
        (s / s_arm) * math.tan(a) ** 2 / r
        for s, a, r in zip(shares, alphas, r_h, strict=True)
    )
    p3 = radial_capacity_failure_load(hoop, tan2_over_r)
    p4 = 0.0
    for lyr, sec in zip(stack, sections, strict=True):
        assert lyr.n_wires is not None and lyr.lay_angle_rad is not None
        p4 += wire_yield_load(
            lyr.n_wires,
            lyr.lay_angle_rad,
            library.get_property(lyr.material, "sigma_y_Pa"),
            sec.area_m2,
        )
    p2_star = combined_radial_load(p2 if p2 > 0.0 else math.inf, p3, p4)
    demand = s_arm * compression
    return RadialBucklingStack(
        layers=tuple(lyr.name for lyr in stack),
        restraint_layers=restraint,
        n_wires_total=n_tot,
        mean_radius_m=r_mean,
        foundation_c_Pa=tuple(c_layers),
        radial_model=radial_model,
        P2_N=p2 if p2 > 0.0 else math.inf,
        half_wavelength_m=half_wave,
        P3_N=p3,
        P4_N=p4,
        P2_star_N=p2_star,
        demand_N=demand,
        utilisation=demand / p2_star if math.isfinite(p2_star) else 0.0,
    )


def _assumptions(
    outside: Sequence[str],
    cable: CableData,
    radial: RadialBucklingStack,
    mu: float | None,
    mu_eff: float,
    curvature_1_m: float | None,
    annulus: str,
    strength: float | Mapping[str, float] | None,
    source: str | None,
    tapes: Sequence[TapeLayer],
    lateral: Sequence[LateralBucklingLayer],
) -> tuple[str, ...]:
    out = [
        "axial force shared among layers by equal axial strain (EA-weighted)",
        f"annulus taken as {annulus} (caller's choice, not inferred from the cable)",
    ]
    if cable.water_ingress is not None and (
        (cable.water_ingress.armour_annulus == "flooded") != (annulus == "flooded")
    ):
        out.append(
            f"annulus differs from the cable declaration "
            f"({cable.water_ingress.armour_annulus})"
        )
    if mu is None:
        out.append(f"friction coefficient {mu_eff:g} from the armour material library")
    if curvature_1_m is None:
        out.append(
            "cyclic curvature not given: lateral slip assumed and the lowest "
            "tabulated knock-down (0.83) applied"
        )
    if any(lat.knockdown_beyond_table for lat in lateral):
        out.append(
            "curvature beyond the Li-Vaz knock-down table; last value (0.83) held"
        )
    if outside:
        out.append(
            "lay angle outside the 20-38 deg range of the reproduced cases for "
            f"{', '.join(outside)}: the lateral lower bound falls with sin^2(alpha) "
            "and omits the buckle-length term, so it is increasingly conservative"
        )
    if not radial.restraint_layers:
        out.append("no outboard hoop layer: radial elastic mode not bounded (P2 = inf)")
    if strength is None and not tapes:
        out.append("no sourced sheath or tape hoop strength: P3 unavailable (inf)")
    elif source is not None:
        out.append(f"sheath hoop strength from: {source}")
    out.extend(f"tape {t.name} from: {t.source}" for t in tapes)
    return tuple(out)


def _provenance(radial_model: str) -> Provenance:
    return Provenance(
        method=(
            "armour compression screen: Saevik-Thorsen lateral lower bound "
            "(Eq. 19) and Li-Vaz round-wire limit with cyclic-curvature "
            "knock-down, gated by first slip; radial "
            f"{radial_model}-beam buckling on the outboard sheath/tape "
            "foundation with hoop-capacity and yield interaction (Eq. 35)"
        ),
        citations=_CITATIONS,
        gates=("ST17-T3", "OST12", "LV21", "ST17-P34", "ST17-SOUSA", "YANG15"),
        limitations=(
            "analytical screen; no imperfections, plasticity in the buckling "
            "modes or inter-wire contact",
            "lateral capacity is the friction-free lower bound unless the "
            "cyclic curvature is below first slip; the first-slip curvature "
            "ignores elastic slip, which delays slip in the source FE",
            "radial foundation from thin-ring hoop stiffness of outboard "
            "annuli at the mean armour radius; the inner layers of the stack "
            "are not credited",
            "design factor transfers the DNV-RP-F401 armour utilisation to "
            "buckling capacity (assumption)",
        ),
    )


# --------------------------------------------------------------------------- #
# Combined capacity curve                                                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CompressionCapacityResult:
    """Curvature capacity over a signed tension sweep including compression.

    Attributes
    ----------
    tension_N : numpy.ndarray
        Effective tensions (negative = compression) [N].
    driving_tension_N : numpy.ndarray
        Effective (flooded) or true wall (sealed) tensions [N].
    kappa_max_1_m : numpy.ndarray
        Governing curvature capacity [1/m].
    kappa_yield_1_m : numpy.ndarray
        First-yield curvature at the design usage η [1/m].
    kappa_lateral_1_m : numpy.ndarray
        Smallest cyclic curvature at which lateral buckling is reached
        (``inf`` if never) [1/m].
    uf_radial : numpy.ndarray
        Radial design utilisation (demand / η P2*), curvature independent.
    governing : list[str]
        ``"wire_yield"``, ``"lateral_buckling"``, ``"radial_buckling"`` or
        ``"component_limit"``.
    utilisation_factor : float
        Design utilisation factor η.
    compression_capacity_N : float
        Characteristic cable compression capacity with lateral slip active
        at the largest tabulated knock-down [N].
    provenance : Provenance
    kappa_component_1_m : numpy.ndarray
        Curvature at which the design utilisation of the components other
        than the armour reaches one (``inf`` where never; empty when not
        checked) [1/m].
    governing_component : tuple[tuple[str, str], ...]
        The component and its limit at that curvature.
    """

    tension_N: FloatArray
    driving_tension_N: FloatArray
    kappa_max_1_m: FloatArray
    kappa_yield_1_m: FloatArray
    kappa_lateral_1_m: FloatArray
    uf_radial: FloatArray
    governing: list[str]
    utilisation_factor: float
    compression_capacity_N: float
    provenance: Provenance
    kappa_component_1_m: FloatArray = field(default_factory=lambda: np.zeros(0))
    governing_component: tuple[tuple[str, str], ...] = ()


def _lateral_curvature(
    cable: CableData,
    tension_N: float,
    opts: _Options,
    kappa_hi: float,
) -> float:
    """Smallest cyclic curvature at which a layer reaches η × lateral capacity.

    The lateral capacity does not rise with curvature (knock-down) and the
    gate opens once the curvature passes first slip, so failure is monotone in
    curvature and a bisection applies.
    """
    eta_design = opts.utilisation_factor

    def fails(kappa: float) -> bool:
        res = _screen(cable, tension_N, kappa, opts)
        return any(lat.active and lat.utilisation >= eta_design for lat in res.lateral)

    if fails(0.0):
        return 0.0
    if not fails(kappa_hi):
        return math.inf
    lo, hi = 0.0, kappa_hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if fails(mid):
            hi = mid
        else:
            lo = mid
    return hi


def compute_compression_capacity(
    cable: CableData,
    tension_N: FloatArray,
    *,
    annulus: Annulus = "flooded",
    p_ext_Pa: float = 0.0,
    mu: float | None = None,
    radial_model: RadialModel = "straight",
    sheath_hoop_strength_Pa: float | Mapping[str, float] | None = None,
    sheath_strength_source: str | None = None,
    tapes: Sequence[TapeLayer] = (),
    utilisation_factor: float = 1.0,
    kappa_search_max_1_m: float = 2.0,
    component_limits: bool = True,
    component_utilisation_factor: float | None = None,
) -> CompressionCapacityResult:
    """Tension–curvature and compression–curvature capacity curve.

    At each tension the curvature capacity is the smaller of the first-yield
    curvature (the common slip-front wire stress of
    :func:`cablexsec.analysis.capacity.wire_yield_curvature`, here at usage
    η) and, in compression, the cyclic curvature at which a layer reaches η
    times its lateral capacity (searched up to ``kappa_search_max_1_m``). A
    radial design utilisation of one or more sets the capacity to zero. At
    η = 1 and positive tension the curve is the wire-yield branch of
    :func:`cablexsec.compute_capacity`. Options are those of
    :func:`compression_screen`. With ``component_limits`` (default) the
    limits of the components other than the armour bound the curvature too,
    as in :func:`~cablexsec.analysis.capacity_check.build_capacity_envelope`,
    with ``component_utilisation_factor`` (default: the factor of the
    lifecycle phase of ``utilisation_factor``,
    :func:`~cablexsec.analysis.component_limits.component_factor_for`).
    """
    from cablexsec.analysis.capacity import wire_yield_curvature  # noqa: PLC0415
    from cablexsec.analysis.component_limits import (  # noqa: PLC0415
        component_factor_for,
    )
    from cablexsec.analysis.components import (  # noqa: PLC0415
        component_usage_curve,
    )

    eta_c = (
        component_factor_for(utilisation_factor)
        if component_utilisation_factor is None
        else float(component_utilisation_factor)
    )
    if not (math.isfinite(eta_c) and 0.0 < eta_c <= 1.0):
        raise CableInputError("component_utilisation_factor must lie in (0, 1]")

    _check_options(
        annulus,
        p_ext_Pa,
        radial_model,
        utilisation_factor,
        sheath_hoop_strength_Pa,
        sheath_strength_source,
        None,
    )
    _positive("kappa_search_max_1_m", kappa_search_max_1_m)
    opts = _Options(
        annulus=annulus,
        p_ext_Pa=p_ext_Pa,
        mu=mu,
        radial_model=radial_model,
        sheath_hoop_strength_Pa=sheath_hoop_strength_Pa,
        sheath_strength_source=sheath_strength_source,
        tapes=tuple(tapes),
        utilisation_factor=utilisation_factor,
    )
    tensions = np.asarray(tension_N, dtype=np.float64).ravel()
    n = tensions.size
    drive = np.zeros(n)
    k_yield = np.zeros(n)
    k_lat = np.full(n, math.inf)
    uf_rad = np.zeros(n)
    k_max = np.zeros(n)
    k_comp = np.full(n, math.inf)
    comp_governing: list[tuple[str, str]] = []
    governing: list[str] = []
    capacity = _screen(cable, 0.0, None, opts).compression_capacity_N
    for i, t in enumerate(tensions):
        base = _screen(cable, float(t), None, opts)
        drive[i] = base.driving_tension_N
        uf_rad[i] = base.radial.utilisation / utilisation_factor
        k_yield[i] = wire_yield_curvature(
            cable, base.driving_tension_N, mu=mu, usage_limit=utilisation_factor
        )
        if base.driving_tension_N < 0.0:
            k_lat[i] = _lateral_curvature(cable, float(t), opts, kappa_search_max_1_m)
        if component_limits:
            reach = k_yield[i] if math.isfinite(k_yield[i]) else 0.0
            curve = component_usage_curve(
                cable,
                base.driving_tension_N,
                1.5 * max(reach, 0.2),
                mu=mu,
                factor=eta_c,
            )
            k_comp[i] = curve.root(1.0)
            comp_governing.append(
                curve.governing(k_comp[i]) if curve.assessed else ("", "")
            )
        else:
            comp_governing.append(("", ""))
        if uf_rad[i] >= 1.0:
            k_max[i] = 0.0
            governing.append("radial_buckling")
        elif k_comp[i] < min(k_lat[i], k_yield[i]):
            k_max[i] = k_comp[i]
            governing.append("component_limit")
        elif k_lat[i] < k_yield[i]:
            k_max[i] = k_lat[i]
            governing.append("lateral_buckling")
        else:
            k_max[i] = k_yield[i]
            governing.append("wire_yield")
    return CompressionCapacityResult(
        tension_N=tensions,
        driving_tension_N=drive,
        kappa_max_1_m=k_max,
        kappa_yield_1_m=k_yield,
        kappa_lateral_1_m=k_lat,
        uf_radial=uf_rad,
        governing=governing,
        utilisation_factor=utilisation_factor,
        compression_capacity_N=capacity,
        provenance=_provenance(radial_model),
        kappa_component_1_m=k_comp if component_limits else np.zeros(0),
        governing_component=tuple(comp_governing),
    )


# Public entry points read the cable's own materials when called directly.
_scope_entry_points(globals(), __all__)
