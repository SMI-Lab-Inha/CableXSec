# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""The material properties a cable file may define, with units and ranges.

The material library (:mod:`cablexsec.materials.library`) and the
``materials:`` block of a cable file use the same unit-suffixed property
names in strict SI units. This module defines each name once: its label,
group, SI unit and the physically admissible range that a cable file is
checked against. The range is a plausibility bound, not a design limit.

Examples
--------
>>> MATERIAL_PROPERTIES["E_Pa"].unit
'Pa'
>>> MATERIAL_PROPERTIES["nu"].problem(0.6)
'nu = 0.6 is outside 0 <= value < 0.5'
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

__all__ = [
    "ELASTIC",
    "ELECTRICAL",
    "FRICTION",
    "MATERIAL_PROPERTIES",
    "PROPERTY_GROUPS",
    "SOURCE_EXAMPLES",
    "STRENGTH",
    "THERMAL",
    "PropertyDefinition",
]

ELASTIC = "Elastic"
STRENGTH = "Strength"
FRICTION = "Friction"
THERMAL = "Thermal"
ELECTRICAL = "Electrical"

PROPERTY_GROUPS: tuple[str, ...] = (ELASTIC, STRENGTH, FRICTION, THERMAL, ELECTRICAL)
"""Groups of the material properties, in the order they are shown."""

SOURCE_EXAMPLES: tuple[str, ...] = ("user input", "model assumption")
"""Source texts for a value that has no literature reference."""


@dataclass(frozen=True)
class PropertyDefinition:
    """One material property: name, meaning, SI unit and admissible range."""

    name: str
    """Unit-suffixed key, as in the library (``E_Pa``)."""
    label: str
    """What a user reads (``Young's modulus``)."""
    group: str
    unit: str
    """SI unit symbol of the stored value; empty for a ratio."""
    lower: float
    upper: float
    lower_open: bool = False
    """Whether ``lower`` itself is excluded."""
    upper_open: bool = False
    """Whether ``upper`` itself is excluded."""
    help: str = ""

    def admits(self, value: float) -> bool:
        """Whether ``value`` is finite and within the range."""
        if not math.isfinite(value):
            return False
        above = value > self.lower if self.lower_open else value >= self.lower
        below = value < self.upper if self.upper_open else value <= self.upper
        return above and below

    @property
    def range_text(self) -> str:
        """The range as text: ``0 < value <= 1e+12 Pa``."""
        low = "<" if self.lower_open else "<="
        high = "<" if self.upper_open else "<="
        unit = f" {self.unit}" if self.unit else ""
        return f"{self.lower:g} {low} value {high} {self.upper:g}{unit}"

    def problem(self, value: float) -> str | None:
        """Why ``value`` is not admitted, or ``None`` when it is."""
        if self.admits(value):
            return None
        return f"{self.name} = {value:g} is outside {self.range_text}"


def _p(
    name: str,
    label: str,
    group: str,
    unit: str,
    bounds: tuple[float, float],
    *,
    open_: str = "lower",
    help: str = "",
) -> PropertyDefinition:
    return PropertyDefinition(
        name,
        label,
        group,
        unit,
        bounds[0],
        bounds[1],
        lower_open=open_ in ("lower", "both"),
        upper_open=open_ in ("upper", "both"),
        help=help,
    )


_DEFINITIONS: tuple[PropertyDefinition, ...] = (
    _p("E_Pa", "Young's modulus", ELASTIC, "Pa", (0.0, 1.0e12)),
    _p(
        "E_eff_axial_Pa",
        "Effective axial modulus",
        ELASTIC,
        "Pa",
        (0.0, 1.0e12),
        help="Axial modulus of the wires of a stranded conductor; a material "
        "that gives it without a Young's modulus is treated as a conductor.",
    ),
    _p(
        "E_eff_bending_Pa",
        "Effective bending modulus",
        ELASTIC,
        "Pa",
        (0.0, 1.0e12),
        help="Bending modulus of an individual conductor, if it differs from "
        "the axial modulus.",
    ),
    _p(
        "E_corrugated_Pa",
        "Corrugated wall modulus",
        ELASTIC,
        "Pa",
        (0.0, 1.0e12),
        help="Equivalent modulus of a corrugated metallic sheath wall.",
    ),
    _p("nu", "Poisson's ratio", ELASTIC, "", (0.0, 0.5), open_="upper"),
    _p("rho_kg_m3", "Density", ELASTIC, "kg/m³", (0.0, 25000.0)),
    _p(
        "creep_coeff",
        "Norton creep coefficient",
        ELASTIC,
        "",
        (0.0, 1.0),
        help="Coefficient of the Norton creep law, in the units of its source.",
    ),
    _p(
        "creep_exponent",
        "Norton creep exponent",
        ELASTIC,
        "",
        (1.0, 20.0),
        open_="none",
    ),
    _p(
        "sigma_y_Pa",
        "Yield stress",
        STRENGTH,
        "Pa",
        (0.0, 1.0e10),
        help="Yield stress, or the 0.2 % proof stress of a metal without a "
        "yield point (copper, aluminium): the stress limit of the component.",
    ),
    _p("sigma_UTS_Pa", "Tensile strength", STRENGTH, "Pa", (0.0, 1.0e10)),
    _p(
        "epsilon_allow",
        "Allowable strain",
        STRENGTH,
        "",
        (0.0, 1.0),
        help="Largest longitudinal strain a component of this material may "
        "take, in tension or compression (insulation, semiconducting screen, "
        "polymer sheath, metallic screen or sheath), as a ratio (0.077 is "
        "7.7 %). Without it the component has no strain limit, which the "
        "results say.",
    ),
    _p(
        "mu_on_hdpe",
        "Friction on polyethylene",
        FRICTION,
        "",
        (0.0, 1.5),
        open_="none",
        help="Coulomb friction coefficient of this material sliding on a "
        "polymer sheath or tape.",
    ),
    _p(
        "mu_on_steel",
        "Friction on steel",
        FRICTION,
        "",
        (0.0, 1.5),
        open_="none",
        help="Coulomb friction coefficient of this material sliding on steel wires.",
    ),
    _p(
        "mu_on_cu",
        "Friction on copper",
        FRICTION,
        "",
        (0.0, 1.5),
        open_="none",
        help="Coulomb friction coefficient of this material sliding on copper wires.",
    ),
    _p(
        "alpha_T_K",
        "Thermal expansion coefficient",
        THERMAL,
        "1/K",
        (-1.0e-4, 1.0e-3),
        open_="none",
    ),
    _p("lambda_W_mK", "Thermal conductivity", THERMAL, "W/(m·K)", (0.0, 2000.0)),
    _p(
        "rho_cp_J_m3K",
        "Volumetric heat capacity",
        THERMAL,
        "J/(m³·K)",
        (0.0, 2.0e7),
    ),
    _p("rho_ohm_m", "Electrical resistivity", ELECTRICAL, "Ω·m", (0.0, 1.0e-5)),
    _p(
        "alpha_R_per_K",
        "Temperature coefficient of resistivity",
        ELECTRICAL,
        "1/K",
        (0.0, 0.02),
        open_="none",
    ),
    _p(
        "epsilon_r",
        "Relative permittivity",
        ELECTRICAL,
        "",
        (1.0, 100.0),
        open_="none",
    ),
)

MATERIAL_PROPERTIES: Mapping[str, PropertyDefinition] = MappingProxyType(
    {definition.name: definition for definition in _DEFINITIONS}
)
"""Every material property a cable file may define, by name, in group order."""
