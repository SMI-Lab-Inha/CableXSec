# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Fixed-radius helical-rod transfer for assembled power cores.

Axial energy is transformed from core length to cable length. Local bending
uses the same two Lutchansky curvature components as individual wire stress.
This is a small-increment rod reduction, not a three-core radial contact or
internal-slip solution. Core cross-section torsion/warping is not condensed
into the extension-only assembly transform.
"""

from __future__ import annotations

import math

from cablexsec.cables.schema import CableData, CylindricalOffsetLayer
from cablexsec.exceptions import CableSchemaError


def core_angle(cable: CableData, layer: CylindricalOffsetLayer) -> float:
    """Signed assembly angle at the core-axis radius [rad]."""
    if cable.core_lay is None:
        return 0.0
    if cable.core_lay.pitch_m is None or layer.r_offset_m is None:
        raise CableSchemaError("complete assembly pitch and core radius are required")
    sign = 1 if cable.core_lay.lay_direction == "right_hand" else -1
    return sign * math.atan2(2 * math.pi * layer.r_offset_m, cable.core_lay.pitch_m)


def bending_factor(angle: float) -> float:
    """Mean helical-rod bending energy relative to straight bending."""
    c, s = math.cos(angle), math.sin(angle)
    return (c * c * (1 + s * s) ** 2 + (c * c * math.cos(2 * angle)) ** 2) / (2 * c)


def core_bending_components(
    angle: float, phase: float, kx: float, ky: float
) -> tuple[float, float]:
    """Longitudinal bending-strain gradients in the radial/tangential frame.

    The azimuth is measured from the positive extension side of kx, as in
    the global core-axis strain. Ménard and Cartraud (2023), Eq. (14), use
    an azimuth measured from the bending axis instead. Its second coefficient
    is cos²(angle)*cos(2*angle), the curve curvature of the deformed helix:
    the normal curvature of a core that carries no bending axial strain
    (free). The slip state adds ``(sin²(angle) / r_c) epsilon_b`` of the
    bending part of the core-axis strain
    (:func:`~cablexsec.mechanics.wire_stress.slip_state_normal_curvature`),
    which makes a stuck core ``cos⁴(angle)``; the callers add it.
    """
    c, s = math.cos(angle), math.sin(angle)
    radial = kx * math.cos(phase) + ky * math.sin(phase)
    tangential = -kx * math.sin(phase) + ky * math.cos(phase)
    return c * c * math.cos(2 * angle) * radial, c * (1 + s * s) * tangential
