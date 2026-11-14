# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Resolved internal wire rings for straight, fixed-radius power-core axes.

The axial energy per core length is sum(E A (c² epsilon-eT)² / (2c)).
It gives both the fixed-twist rigidity and the thermal force coefficient.
Internal wire bending follows the same two Lutchansky curvature coefficients
as wire_stress.local_bending_amplitude, integrated over a complete wire lay.
The internal parallel-axis term is a separate, fully constrained reference;
it is not inserted into the armour-only friction history as an elastic spring.

This reduction does not solve core lay, core twist, radial contact, or internal
friction. Those require additional degrees of freedom and contact states.
"""

from __future__ import annotations

import math

from cablexsec.cables.schema import CoreStrandRing, CylindricalOffsetLayer
from cablexsec.mechanics.core_lay import bending_factor


def axial_area(layer: CylindricalOffsetLayer) -> float:
    """Geometric coefficient of E epsilon in core-axis force, per core [m²]."""
    return sum(r.area_m2 * math.cos(r.lay_angle_rad) ** 3 for r in layer.strand_layers)


def thermal_force_area(layer: CylindricalOffsetLayer) -> float:
    """Geometric coefficient of E eT in core-axis force, per core [m²]."""
    return sum(r.area_m2 * math.cos(r.lay_angle_rad) for r in layer.strand_layers)


def wire_bending_inertia(ring: CoreStrandRing) -> float:
    """Individual-wire bending-energy coefficient per core length [m⁴].

    For wire azimuth phi the local curvature components are
    kappa*c²*cos(2*alpha)*cos(phi) and -kappa*c*(1+s²)*sin(phi), with phi
    measured from the extension side. Average their squared norm
    over a lay and convert wire length to core length using 1/c.
    """
    return (
        ring.n_wires
        * math.pi
        * ring.d_wire_m**4
        / 64
        * bending_factor(ring.lay_angle_rad)
    )


def internal_parallel_inertia(layer: CylindricalOffsetLayer) -> float:
    """Fully constrained internal ring-axis contribution per core [m⁴]."""
    return sum(
        0.5 * r.area_m2 * r.r_helix_m**2 * math.cos(r.lay_angle_rad) ** 3
        for r in layer.strand_layers
    )
