# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Limits of the cable components and the utilisation against them.

Every component of the section is checked against the limits its material
gives, each with its source. The guidance of the OWA MecLim II guideline
(Offshore Wind Accelerator, Carbon Trust, July 2025) [MecLim2025] is followed:
the onset of damage of a component is "the yield strength of a metallic
material or [the] strain limit for a non-metallic material", applicable to the
combination of material and component, with "a reduced strain limit ...
typically applicable to conductor insulation" ([MecLim2025] Sec. 3.4, p. 15),
and loading is applied "until a utilisation of 1.0 is achieved in each
component" (Sec. 3.2, p. 9).

======================  ======================================  ===================
Limit (material key)     Utilisation                             Components
======================  ======================================  ===================
``sigma_y_Pa``           largest stress magnitude / limit        metals: conductor
                         (yield, or the 0.2 % proof stress of    (0.2 % proof), metal
                         a metal without a yield point)          screens and sheaths
``sigma_UTS_Pa``         net-section stress / limit              conductor (fracture)
``epsilon_allow``        largest strain magnitude / limit        any material
======================  ======================================  ===================

A component whose material gives none of them has no utilisation: the
results say "no limit given" and name the property to enter. No limit is
substituted: DNVGL-ST-0119 (July 2018) 16.2.2, p. 149, states that the main
shortcoming of the available codes is "the lack of acceptance criteria, i.e.
allowable strain or utilization of yield, of the various materials and
components of a power cable".

Published limits in the material library
----------------------------------------
* Copper conductor of the IEA Task 49 template: 0.2 % proof stress
  (plastic limit) 120 MPa of ETP copper, after Robinson; [Janocha2024]
  Sec. 2.1, p. 5. The 250 MPa yield strength of the same sentence is not
  used: the source itself takes the 120 MPa plastic limit as the conductor
  capacity (Sec. 3.3, Fig. 21).
* Polyethylene sheaths: maximum allowable strain 7.7 % of PE and PA sheaths of
  unbonded flexible pipe, [API17J] (3rd ed. 2008 / ISO 13628-2:2006) Table 6,
  p. 19. It is the only published allowable strain of a PE sheath in the
  reference library; DNVGL-ST-0119 16.2.2 (p. 149) holds that the principles of
  the flexible-pipe codes may for the most part be applied to power cables.
* Steel armour yield and the design utilisation factors of DNV-RP-F401
  (September 2019, amended September 2021) Sec. 3.8.1, Table 3-1, p. 13: 0.67
  normal operation, 0.78 installation, 1.00 abnormal operation, on the lesser
  of the yield strength and 90 % of the tensile strength. API 17J Table 6
  (p. 19) uses the same 0.67 for the tensile armour of flexible pipe in normal
  operation; the citation of the armour factor is DNV-RP-F401.

No published value (user input, the result says "no limit given"): the strain
limit of XLPE insulation and of semiconducting screens ([MecLim2025] p. 15
names a reduced limit without a value); the strain limit of lead sheaths
(CIGRE TB 610 (2015) Sec. 5.4.2, p. 75: "setting practical criteria for
maximum permissible strain on the lead sheath by calculation is very
difficult"); the strength of copper tape screens.

Utilisation factors of the components other than the steel armour
------------------------------------------------------------------
[MecLim2025] Table 5, p. 19: 0.80 in normal operation, 1.00 in installation
under load control (0.75 under displacement control) and 1.00 in abnormal
operation, set by the MecLim II JIP after API 17E (ISO 13628-5) for umbilicals.
:func:`component_factor_for` maps an armour factor of DNV-RP-F401 Table 3-1
to the factor of the same lifecycle phase.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cablexsec.materials import library

__all__ = [
    "COMPONENT_FACTOR_SOURCE",
    "COMPONENT_UTILISATION_FACTORS",
    "NO_LIMIT",
    "Assessment",
    "Limit",
    "assess",
    "component_factor_for",
    "component_limits",
    "limit_text",
    "point_usage",
    "stress_limit",
]

FloatArray = NDArray[np.float64]

#: Utilisation factors of the cable components other than the steel armour,
#: by lifecycle phase ([MecLim2025] Table 5, p. 19; installation under load
#: control: 0.75 under displacement control).
COMPONENT_UTILISATION_FACTORS: Mapping[str, float] = {
    "normal": 0.80,
    "installation": 1.00,
    "abnormal": 1.00,
}

COMPONENT_FACTOR_SOURCE = (
    "OWA MecLim II guideline (2025) Table 5, p. 19: components other than the "
    "steel armour"
)

NO_LIMIT = "no limit given"
"""Failure mode of a component whose material gives no limit."""

#: Armour factors of DNV-RP-F401 Table 3-1 by phase, for the phase mapping.
_ARMOUR_BY_PHASE: Mapping[str, float] = {
    "normal": 0.67,
    "installation": 0.78,
    "abnormal": 1.00,
}


def component_factor_for(armour_factor: float) -> float:
    """The component utilisation factor of the phase of an armour factor.

    An armour factor of DNV-RP-F401 Table 3-1 (0.67, 0.78, 1.00) gives the
    [MecLim2025] Table 5 factor of the same phase (0.80, 1.00, 1.00); any
    other armour factor is used for the components too.
    """
    for phase, value in _ARMOUR_BY_PHASE.items():
        if math.isclose(armour_factor, value, abs_tol=1e-9):
            return COMPONENT_UTILISATION_FACTORS[phase]
    return float(armour_factor)


@dataclass(frozen=True)
class Limit:
    """One limit of a component: its mode, value, words and source.

    ``mode`` is ``"yielding"`` (stress), ``"tensile fracture"`` (net-section
    stress, conductors) or ``"strain limit"``; ``value`` is in Pa for a
    stress and a ratio for a strain.
    """

    mode: str
    value: float
    words: str
    source: str

    @property
    def text(self) -> str:
        """The limit in words with its value and source."""
        if self.mode == "strain limit":
            amount = f"{100.0 * self.value:.3g} %"
        else:
            amount = f"{self.value / 1e6:.4g} MPa"
        return f"{self.words} {amount} ({self.source})"


def _value(material: str, key: str) -> tuple[float, str] | None:
    try:
        value = float(library.get_property(material, key))
    except (KeyError, ValueError, TypeError):
        return None
    if not (math.isfinite(value) and value > 0.0):
        return None
    try:
        source = library.get_citation(material, key)
    except KeyError:
        source = ""
    return value, source


def component_limits(material: str, component_type: str) -> tuple[Limit, ...]:
    """The limits the material of a component gives, in the order checked.

    The yield stress (for a conductor, its 0.2 % proof stress), the tensile
    strength of a conductor (net section) and the allowable strain. Armour
    wires are checked against their yield stress elsewhere and are not
    covered here.
    """
    limits: list[Limit] = []
    conductor = component_type == "conductor"
    found = _value(material, "sigma_y_Pa")
    if found is not None:
        words = "0.2 % proof stress" if conductor else "yield stress"
        limits.append(Limit("yielding", found[0], words, found[1]))
    if conductor:
        found = _value(material, "sigma_UTS_Pa")
        if found is not None:
            limits.append(
                Limit(
                    "tensile fracture",
                    found[0],
                    "tensile strength on the net section",
                    found[1],
                )
            )
    found = _value(material, "epsilon_allow")
    if found is not None:
        limits.append(Limit("strain limit", found[0], "allowable strain", found[1]))
    return tuple(limits)


def limit_text(limits: tuple[Limit, ...]) -> str:
    """The limits in words, ``no limit given`` when there are none."""
    if not limits:
        return NO_LIMIT
    return "; ".join(limit.text for limit in limits)


@dataclass(frozen=True)
class Assessment:
    """Utilisation of a component against all its limits.

    ``utilisation`` is the largest of ``utilisations`` (mode, value), the
    governing ``mode`` names it; NaN and :data:`NO_LIMIT` without a limit.
    ``limits`` says what was checked, with values and sources.
    """

    utilisation: float
    mode: str
    utilisations: tuple[tuple[str, float], ...]
    limits: str


def assess(
    limits: tuple[Limit, ...],
    *,
    peak_stress_Pa: float,
    net_section_Pa: float = math.nan,
    peak_strain: float = math.nan,
) -> Assessment:
    """The utilisation of a component at its largest stress and strain.

    ``peak_stress_Pa`` is the largest stress magnitude (or the equivalent
    stress of a metallic sheath), ``net_section_Pa`` the largest tensile
    net-section stress of a conductor, ``peak_strain`` the largest mechanical
    strain magnitude.
    """
    values: list[tuple[str, float]] = []
    for limit in limits:
        if limit.mode == "yielding" and math.isfinite(peak_stress_Pa):
            values.append((limit.mode, abs(peak_stress_Pa) / limit.value))
        elif limit.mode == "tensile fracture" and math.isfinite(net_section_Pa):
            values.append((limit.mode, max(net_section_Pa, 0.0) / limit.value))
        elif limit.mode == "strain limit" and math.isfinite(peak_strain):
            values.append((limit.mode, abs(peak_strain) / limit.value))
    if not values:
        return Assessment(math.nan, NO_LIMIT, (), limit_text(limits))
    mode, usage = max(values, key=lambda item: item[1])
    return Assessment(usage, mode, tuple(values), limit_text(limits))


def point_usage(
    limits: tuple[Limit, ...],
    stress_Pa: FloatArray,
    modulus_Pa: float,
    net_section_Pa: FloatArray | float = math.nan,
) -> FloatArray:
    """The utilisation at every point of a field (NaN without a limit).

    The stress-based limits act on each point's stress, the strain limit on
    its mechanical strain ``stress / modulus``, the fracture limit on the
    net-section stress of the point's instance (broadcast against
    ``stress_Pa``). The largest value over the points is the component's
    utilisation from :func:`assess` on the same values.
    """
    stress = np.asarray(stress_Pa, dtype=np.float64)
    out = np.full(stress.shape, -math.inf)
    used = False
    net = np.broadcast_to(np.asarray(net_section_Pa, dtype=np.float64), stress.shape)
    for limit in limits:
        if limit.mode == "yielding":
            out = np.maximum(out, np.abs(stress) / limit.value)
            used = True
        elif limit.mode == "tensile fracture" and np.all(np.isfinite(net)):
            out = np.maximum(out, np.maximum(net, 0.0) / limit.value)
            used = True
        elif limit.mode == "strain limit" and modulus_Pa > 0.0:
            out = np.maximum(out, np.abs(stress) / modulus_Pa / limit.value)
            used = True
    if not used:
        return np.full(stress.shape, math.nan)
    return out


def stress_limit(limits: tuple[Limit, ...], modulus_Pa: float) -> tuple[float, str]:
    """The smallest stress magnitude at which a point reaches a limit.

    The yield stress, or the allowable strain times the modulus; the
    net-section fracture limit is not a point limit. Returns NaN and an empty
    text when the component has no point limit.
    """
    best = (math.nan, "")
    for limit in limits:
        if limit.mode == "yielding":
            value = limit.value
        elif limit.mode == "strain limit" and modulus_Pa > 0.0:
            value = limit.value * modulus_Pa
        else:
            continue
        if not math.isfinite(best[0]) or value < best[0]:
            best = (value, limit.text)
    return best
