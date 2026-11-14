# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Cable axial-torsion stiffness assembled from per-element stiffnesses.

A **separate** path from :mod:`cablexsec.mechanics.stiffness` (which builds the
6×6 from raw geometry + materials in a :class:`CableData`). This assembler
consumes per-element stiffnesses directly — axial ``EA_i``, torsional ``GJ_i``,
bending ``EI_i``, centroid radius ``R_i`` and helical pitch ``p_i`` — as
tabulated by sources such as Sævik & Bruaseth (2005) Appendix A. It does **not**
reconstruct raw geometry (wire counts, areas, moduli), so it never gap-fills.

Each helical element at radius ``R`` with pitch ``p`` has lay angle
``α = atan(2π R / |p|)`` (``α = 0`` for a straight element, ``p = 0``). Under a
cable axial strain ``ε`` and twist rate ``φ'`` the element axial strain is
``ε_w = ε cos²α + φ' R sinα cosα`` ([WitzTan1992a]; Knapp helix kinematics); its
axial force ``EA_i ε_w`` projects onto the cable axis (``×cosα``) and torque
(``×R sinα``), giving the tension–torsion sub-block of the 6×6:

.. math::
    K_{11} = \sum_i EA_i \cos^3α_i, \qquad
    K_{12} = \sum_i s_i\,EA_i R_i \sinα_i \cos^2α_i,

.. math::
    K_{22} = \sum_i \big(EA_i R_i^2 \sin^2α_i \cosα_i + GJ_i \cosα_i\big),

with ``s_i = sign(p_i)`` the lay direction. The **tension–torsion-fixed** axial
stiffness (twist restrained, ``φ' = 0``) is ``EA_fixed = K_{11}``; the
**tension–torsion-free** stiffness (torque-free, Schur reduction) is
``EA_free = K_{11} - K_{12}^2 / K_{22}``.

This is the **no-radial-interaction** stiffness: it omits the inter-layer radial
contraction coupling (wires pressing inward, core compressing) that is the
central effect of [SaevikBruaseth2005] and reduces the real axial stiffness.
That reduction needs cross-section radial-compliance data not carried by the
per-element ``(EA_i, GJ_i, EI_i, R_i, p_i)`` set; this assembler therefore gives
the analytical upper bound.

References
----------
[WitzTan1992a] (helix axial-torsion kinematics); [SaevikBruaseth2005]
(per-element tabulation, the radial-interaction reduction this omits).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "HelicalElement",
    "axial_torsion_2x2",
    "ea_fixed_from_elements",
    "ea_free_from_elements",
    "lay_angle_rad",
]


@dataclass(frozen=True)
class HelicalElement:
    """One structural element of a cable cross-section (tabulated stiffnesses).

    Parameters
    ----------
    centroid_radius_m : float
        Helix (centroid) radius ``R`` [m]; 0 for a straight central element.
    pitch_m : float
        Helical lay pitch ``p`` [m], signed for lay direction; 0 = straight.
    EA_N : float
        Axial stiffness ``E A`` of the element [N].
    GJ_Nm2 : float
        Torsional stiffness ``G J`` of the element [N·m²].
    EI_Nm2 : float
        Bending stiffness ``E I`` of the element [N·m²] (recorded; not used in
        the tension–torsion sub-block).
    """

    centroid_radius_m: float
    pitch_m: float
    EA_N: float
    GJ_Nm2: float
    EI_Nm2: float


def lay_angle_rad(element: HelicalElement) -> float:
    """Lay angle ``α = atan(2π R / |p|)`` [rad] (0 for a straight element)."""
    if element.pitch_m == 0.0 or element.centroid_radius_m == 0.0:
        return 0.0
    return math.atan(2.0 * math.pi * element.centroid_radius_m / abs(element.pitch_m))


def axial_torsion_2x2(
    elements: list[HelicalElement],
) -> tuple[float, float, float]:
    """Assemble the cable tension–torsion sub-block ``(K11, K12, K22)``."""
    k11 = k12 = k22 = 0.0
    for el in elements:
        a = lay_angle_rad(el)
        ca, sa = math.cos(a), math.sin(a)
        sign = 1.0 if el.pitch_m >= 0.0 else -1.0
        k11 += el.EA_N * ca**3
        k12 += sign * el.EA_N * el.centroid_radius_m * sa * ca**2
        k22 += el.EA_N * el.centroid_radius_m**2 * sa**2 * ca + el.GJ_Nm2 * ca
    return k11, k12, k22


def ea_fixed_from_elements(elements: list[HelicalElement]) -> float:
    """Tension–torsion-**fixed** axial stiffness ``EA_fixed = K11`` [N].

    Twist restrained (``φ' = 0``). The no-radial-interaction analytical value
    (see the module docstring).
    """
    return axial_torsion_2x2(elements)[0]


def ea_free_from_elements(elements: list[HelicalElement]) -> float:
    """Tension–torsion-**free** axial stiffness ``K11 - K12²/K22`` [N] (Schur)."""
    k11, k12, k22 = axial_torsion_2x2(elements)
    return k11 - k12**2 / k22 if k22 > 0.0 else k11
