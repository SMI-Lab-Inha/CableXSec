# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Energy wear coefficients and fretting-fatigue knock-down laws.

Every entry states its source or is labelled an assumption. Archard
coefficients ``k`` (volume per unit normal load and sliding distance) are
converted to energy coefficients with ``alpha = k / mu`` at the friction
coefficient of the same test, since the dissipated energy of a sliding cycle
is ``mu P`` times its sliding distance ([Fouvry1996] Eq. 20;
[LiskiewiczFouvry2005] Eq. 4). ``1 MPa^-1 = 1 mm^3/(N mm) = 1e-6 m^3/(N m)``.

The steel-on-polymer coefficients are **assumptions**: the reference library
has no fretting wear coefficient for steel on polyethylene, polypropylene or
polyamide. Wear depths of those contacts scale linearly with the value and
are order-of-magnitude indications only.

References
----------
[Fouvry1996] Wear 200, 186-205, Table 2; [LiskiewiczFouvry2005];
[Cruzado2013] Table 3; [Poon2023] Table 1; [Poon2024] Table 3;
[OHalloran2017] (riser pressure-armour steel, summary table);
[Nakazawa1994] Eq. 1 and Fig. 3 (Nishioka--Hirakawa); [Raoof1983] Eq. 6.3;
[VingsboSoderberg1988]
Fig. 10; [Poon2024] p. 11.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cablexsec.exceptions import CableInputError, CableLookupError

__all__ = [
    "FRETTING_KNOCKDOWNS",
    "WEAR_COEFFICIENTS",
    "FrettingKnockdown",
    "WearCoefficient",
    "get_knockdown",
    "get_wear_coefficient",
    "wear_pair_for",
]

_MPA_INV = 1.0e-6  # m^3/(N m) per MPa^-1


@dataclass(frozen=True)
class WearCoefficient:
    """Energy wear coefficient of one material pair [m^3/J].

    ``alpha_m3_per_J`` is the nominal value and ``low``/``high`` bound the
    scatter of the cited data (or the assumed band). ``worn_body`` names the
    body whose depth is reported.
    """

    pair: str
    alpha_m3_per_J: float
    low_m3_per_J: float
    high_m3_per_J: float
    source: str
    assumed: bool
    worn_body: str

    def __post_init__(self) -> None:
        values = (self.alpha_m3_per_J, self.low_m3_per_J, self.high_m3_per_J)
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise CableInputError("wear coefficients must be finite and >= 0")
        if not self.low_m3_per_J <= self.alpha_m3_per_J <= self.high_m3_per_J:
            raise CableInputError("nominal wear coefficient must lie in its band")

    def scaled(self, alpha_m3_per_J: float) -> WearCoefficient:
        """Copy with another nominal value (a sensitivity case)."""
        return WearCoefficient(
            self.pair,
            alpha_m3_per_J,
            min(self.low_m3_per_J, alpha_m3_per_J),
            max(self.high_m3_per_J, alpha_m3_per_J),
            self.source + "; nominal value replaced by the caller",
            True,
            self.worn_body,
        )


WEAR_COEFFICIENTS: dict[str, WearCoefficient] = {
    "steel_steel": WearCoefficient(
        pair="steel_steel",
        # Cruzado 2013 Table 3: k = 1.57-2.34e-8 MPa^-1 at mu = 0.7 (crossed
        # 0.45 mm eutectoid steel wires); alpha = k/mu.
        alpha_m3_per_J=2.0e-8 * _MPA_INV / 0.7,
        low_m3_per_J=6.3e-9 * _MPA_INV,
        high_m3_per_J=5.5e-8 * _MPA_INV,
        source=(
            "[Cruzado2013] Table 3, Archard k = 1.57-2.34e-8 MPa^-1 at mu = 0.7 "
            "(crossed steel wires), alpha = k/mu with k = 2.0e-8; band from the "
            "energy coefficients of riser pressure-armour steel, 6.3e-9 to "
            "5.5e-8 MPa^-1 ([OHalloran2017] summary table). Same order as "
            "[Fouvry1996] Table 2, 20 850 um^3/J for high-speed steel on alumina"
        ),
        assumed=False,
        worn_body="each steel wire",
    ),
    "copper_copper": WearCoefficient(
        pair="copper_copper",
        # Poon 2023 Table 1 test 6: k' = 8.55e-7 MPa^-1, ECoF 0.91.
        alpha_m3_per_J=8.55e-7 * _MPA_INV / 0.91,
        low_m3_per_J=2.27e-7 * _MPA_INV / 1.0,
        high_m3_per_J=1.25e-6 * _MPA_INV / 0.91,
        source=(
            "[Poon2023] Table 1 (C101 copper, crossed cylinders, gross slip): "
            "k' = 8.42e-7 to 1.25e-6 MPa^-1 at energy friction coefficients "
            "0.91-1.47, alpha = k'/mu with test 6 (8.55e-7, 0.91); lower bound "
            "from the bottom-surface re-evaluation of [Poon2024] Table 3 "
            "(2.27e-7 MPa^-1, mu taken as 1.0)"
        ),
        assumed=False,
        worn_body="each copper wire",
    ),
    "steel_polyethylene": WearCoefficient(
        pair="steel_polyethylene",
        alpha_m3_per_J=1.0e-13,
        low_m3_per_J=1.0e-14,
        high_m3_per_J=1.0e-12,
        source=(
            "ASSUMPTION: no steel-on-polyethylene fretting wear coefficient in "
            "the reference library; order-of-magnitude placeholder with a "
            "decade either side"
        ),
        assumed=True,
        worn_body="polymer bedding or sheath",
    ),
    "steel_polypropylene": WearCoefficient(
        pair="steel_polypropylene",
        alpha_m3_per_J=1.0e-13,
        low_m3_per_J=1.0e-14,
        high_m3_per_J=1.0e-12,
        source=(
            "ASSUMPTION: no steel-on-polypropylene fretting wear coefficient in "
            "the reference library; order-of-magnitude placeholder"
        ),
        assumed=True,
        worn_body="polymer bedding or tape",
    ),
    "steel_polyamide": WearCoefficient(
        pair="steel_polyamide",
        alpha_m3_per_J=1.0e-13,
        low_m3_per_J=1.0e-14,
        high_m3_per_J=1.0e-12,
        source=(
            "ASSUMPTION: no steel-on-polyamide fretting wear coefficient in the "
            "reference library; order-of-magnitude placeholder"
        ),
        assumed=True,
        worn_body="polymer tape or sheath",
    ),
    "steel_fluoroelastomer": WearCoefficient(
        pair="steel_fluoroelastomer",
        alpha_m3_per_J=1.0e-13,
        low_m3_per_J=1.0e-14,
        high_m3_per_J=1.0e-12,
        source=(
            "ASSUMPTION: no steel-on-elastomer fretting wear coefficient in the "
            "reference library; order-of-magnitude placeholder"
        ),
        assumed=True,
        worn_body="elastomer bedding",
    ),
}

# Material-name fragments mapped to wear families (checked in order).
_FAMILIES: tuple[tuple[str, str], ...] = (
    ("steel", "steel"),
    ("copper", "copper"),
    ("cu_", "copper"),
    ("hdpe", "polyethylene"),
    ("mdpe", "polyethylene"),
    ("pe_", "polyethylene"),
    ("xlpe", "polyethylene"),
    ("pfr", "fluoroelastomer"),
    ("ppy", "polypropylene"),
    ("pp_", "polypropylene"),
    ("pa11", "polyamide"),
    ("pa12", "polyamide"),
    ("polyamide", "polyamide"),
)


def _family(material: str) -> str:
    name = material.lower()
    for fragment, family in _FAMILIES:
        if fragment in name or name.startswith(fragment.rstrip("_")):
            return family
    raise CableLookupError(f"no wear family for material {material!r}")


def wear_pair_for(material_1: str, material_2: str) -> str:
    """Registry key for two library materials (order independent)."""
    families = sorted({_family(material_1), _family(material_2)})
    if len(families) == 1:
        key = f"{families[0]}_{families[0]}"
    else:
        metal = [f for f in families if f in ("steel", "copper")]
        other = [f for f in families if f not in ("steel", "copper")]
        if len(metal) == 1 and len(other) == 1:
            key = f"{metal[0]}_{other[0]}"
        else:
            key = "_".join(families)
    if key not in WEAR_COEFFICIENTS:
        raise CableLookupError(f"no wear coefficient registered for {key!r}")
    return key


def get_wear_coefficient(pair: str) -> WearCoefficient:
    """Registered :class:`WearCoefficient` by key."""
    try:
        return WEAR_COEFFICIENTS[pair]
    except KeyError:
        raise CableLookupError(
            f"unknown wear pair {pair!r}; known: {sorted(WEAR_COEFFICIENTS)}"
        ) from None


@dataclass(frozen=True)
class FrettingKnockdown:
    r"""Regime-dependent stress-range factor of the S-N assessment.

    A stress cycle in gross slip has its range multiplied by
    ``gross_slip_factor``. A cycle in partial slip is multiplied by
    ``partial_slip_factor`` and, with ``nishioka_hirakawa=True``, by the
    fretting-fatigue notch factor of the Nishioka--Hirakawa relation
    ([Nakazawa1994] Eq. 1; [Raoof1983] Eq. 6.3)

    .. math::

        K_f = 1 + \frac{2 \mu p_m}{\sigma_{a,ref}}
              \left(1 - e^{-\delta/k}\right),

    the reduction ``2 mu p_m (1 - exp(-delta/k))`` of the fatigue strength
    expressed as a factor at the reference amplitude ``sigma_a,ref``: half
    the S-N range at ``reference_cycles``. ``p_m`` is the mean contact
    pressure, clamped to ``pressure_cap_Pa`` (the range of the source tests),
    and ``delta`` the slip amplitude of the cycle. A factor (rather than an
    added amplitude) keeps cycles far below the fatigue strength from being
    lifted onto the curve by a constant term.
    """

    name: str
    partial_slip_factor: float
    gross_slip_factor: float
    nishioka_hirakawa: bool
    pressure_cap_Pa: float
    slip_constant_m: float
    reference_cycles: float
    source: str
    assumed: bool

    def __post_init__(self) -> None:
        values = (self.partial_slip_factor, self.gross_slip_factor)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise CableInputError("knock-down factors must be finite and positive")
        if math.isnan(self.pressure_cap_Pa) or self.pressure_cap_Pa < 0:
            raise CableInputError("pressure cap must be nonnegative")
        if not self.slip_constant_m > 0 or not self.reference_cycles > 1:
            raise CableInputError("slip constant and reference cycles must be positive")

    def notch_factor(
        self,
        mu: float,
        mean_pressure_Pa: float,
        slip_amplitude_m: NDArray[np.float64],
        reference_range_Pa: float,
    ) -> NDArray[np.float64]:
        """Partial-slip factor ``partial_slip_factor * K_f`` per cycle."""
        delta = np.asarray(slip_amplitude_m, dtype=np.float64)
        if not self.nishioka_hirakawa or not math.isfinite(mu) or mu <= 0:
            return np.full(delta.shape, self.partial_slip_factor)
        amplitude = 0.5 * reference_range_Pa
        if not math.isfinite(amplitude) or amplitude <= 0:
            raise CableInputError("reference stress range must be finite and positive")
        reduction = 2.0 * mu * min(mean_pressure_Pa, self.pressure_cap_Pa) / amplitude
        saturation = -np.expm1(-np.maximum(delta, 0.0) / self.slip_constant_m)
        return np.asarray(self.partial_slip_factor * (1.0 + reduction * saturation))


_NAKAZAWA = (
    "[Nakazawa1994] Eq. 1 (Nishioka-Hirakawa) and Fig. 3: high-strength "
    "steel, contact pressures 40-160 MPa (cap), minimum life at 3-8 um slip. "
    "ASSUMED: slip constant k = 2.7 um (95% of the reduction at 8 um) and the "
    "S-N range at 1e7 cycles as the reference. Gross slip: no crack knock-down, "
    "wear removes initiated cracks ([VingsboSoderberg1988] Fig. 10; "
    "[Fouvry1996] Section 3.1; gross-slip lives rise with wear in [Poon2023] "
    "Fig. 17 and [Poon2024] p. 11)"
)

FRETTING_KNOCKDOWNS: dict[str, FrettingKnockdown] = {
    "steel_nishioka_hirakawa": FrettingKnockdown(
        name="steel_nishioka_hirakawa",
        partial_slip_factor=1.0,
        gross_slip_factor=1.0,
        nishioka_hirakawa=True,
        pressure_cap_Pa=160.0e6,
        slip_constant_m=2.7e-6,
        reference_cycles=1.0e7,
        source=_NAKAZAWA,
        assumed=True,
    ),
    "copper_nishioka_hirakawa": FrettingKnockdown(
        name="copper_nishioka_hirakawa",
        partial_slip_factor=1.0,
        gross_slip_factor=1.0,
        nishioka_hirakawa=True,
        pressure_cap_Pa=160.0e6,
        slip_constant_m=2.7e-6,
        reference_cycles=1.0e7,
        source=(
            "ASSUMPTION: the steel relation applied to copper; no copper "
            "partial-slip knock-down in the library ([Jiang2022]: copper fretting "
            "less dominant than in steel). Library search of 18 Oct 2026 (gap "
            "P7): no plain-versus-fretting copper S-N data or copper knock-down "
            "([Nasution2014b] and [Nasution2013] report no fretting on the "
            "failed wires; [Karlsen2010] p. 6 'fretting plays a less dominating "
            "role'; [Poon2022], [Poon2023], [Wang2026] give FE SWT lives only). "
            "The SWT trellis branch of compute_conductor_fatigue is the sourced "
            "copper route. " + _NAKAZAWA
        ),
        assumed=True,
    ),
    "none": FrettingKnockdown(
        name="none",
        partial_slip_factor=1.0,
        gross_slip_factor=1.0,
        nishioka_hirakawa=False,
        pressure_cap_Pa=0.0,
        slip_constant_m=1.0,
        reference_cycles=1.0e7,
        source="no fretting adjustment (plain S-N)",
        assumed=False,
    ),
}


def get_knockdown(name: str) -> FrettingKnockdown:
    """Registered :class:`FrettingKnockdown` by key."""
    try:
        return FRETTING_KNOCKDOWNS[name]
    except KeyError:
        raise CableLookupError(
            f"unknown knock-down {name!r}; known: {sorted(FRETTING_KNOCKDOWNS)}"
        ) from None
