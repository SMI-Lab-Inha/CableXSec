# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Material areas and volumes for the declared fixed-axis section geometry.

Metal area normal to a wire, material volume per cable length, and a bundle's
outer envelope are distinct quantities. In particular, filler does not occupy
the power-core envelopes. No unspecified gaps are automatically filled.
"""

from __future__ import annotations

import math
from functools import lru_cache

from scipy.integrate import quad

from cablexsec.cables.schema import (
    CableData,
    CylindricalLayer,
    CylindricalOffsetLayer,
    CylindricalShellLayer,
    HelicalLayer,
    HelicalStrandedLayer,
    Layer,
)
from cablexsec.exceptions import CableSchemaError
from cablexsec.mechanics.core_lay import core_angle


def axial_cosine(layer: Layer) -> float:
    """Wire-to-axis projection for a single declared lay, else unity.

    Resolved multi-ring cores use ring-specific projections, not this helper.
    """
    if isinstance(layer, HelicalLayer):
        return math.cos(layer.lay_angle_rad or 0.0)
    if isinstance(layer, CylindricalOffsetLayer):
        return math.cos(math.radians(layer.strand_lay_angle_deg or 0.0))
    return 1.0


def offset_metal_area(layer: CylindricalOffsetLayer) -> float:
    """Actual metal area per core, normal to strand axes when specified [m²]."""
    if layer.strand_layers:
        return sum(r.area_m2 for r in layer.strand_layers)
    if layer.conductor_area_mm2 is not None:
        return layer.conductor_area_mm2 * 1e-6
    if layer.r_inner_m is None or layer.r_outer_m is None:
        raise CableSchemaError("component area requires complete radii")
    return math.pi * (layer.r_outer_m**2 - layer.r_inner_m**2)


def _circle_overlap(radius: float, other: float, distance: float) -> float:
    """Intersection area of two disks; all arguments nonnegative [m²]."""
    if radius == 0 or other == 0 or distance >= radius + other:
        return 0.0
    if distance <= abs(radius - other):
        return math.pi * min(radius, other) ** 2
    left = (distance**2 + radius**2 - other**2) / (2 * distance * radius)
    right = (distance**2 + other**2 - radius**2) / (2 * distance * other)
    radicand = (
        (-distance + radius + other)
        * (distance + radius - other)
        * (distance - radius + other)
        * (distance + radius + other)
    )
    return (
        radius**2 * math.acos(max(-1.0, min(1.0, left)))
        + other**2 * math.acos(max(-1.0, min(1.0, right)))
        - 0.5 * math.sqrt(max(0.0, radicand))
    )


@lru_cache(maxsize=256)
def _ellipse_annulus_overlap(
    inner: float, outer: float, offset: float, radial: float, tangential: float
) -> float:
    """One tilted tubular core envelope intersected with an axial annulus.

    Integrate the ellipse's radial intervals. The ellipse is a local tangent
    approximation to the core tube, with tangent-plane area pi*a*b.
    """
    if offset <= radial:
        raise CableSchemaError("laid core envelope must not include the cable axis")
    limit = math.atan2(tangential, math.sqrt(offset * offset - radial * radial))

    def integrand(theta: float) -> float:
        c, s = math.cos(theta), math.sin(theta)
        a = c * c / radial**2 + s * s / tangential**2
        b = offset * c / radial**2
        d = b * b - a * (offset * offset / radial**2 - 1)
        root = math.sqrt(max(0.0, d))
        lo, hi = max(inner, (b - root) / a), min(outer, (b + root) / a)
        return max(0.0, hi * hi - lo * lo)

    return float(quad(integrand, 0, limit, epsabs=1e-14, epsrel=1e-11, limit=200)[0])


LAID_CORE_OVERLAP = 0.005
"""Overlap of neighbouring laid-core section ellipses, as a share of their
half-width, accepted as touching in the filler area (see :func:`filler_area`)."""


def is_interstitial_filler(layer: Layer, cable: CableData) -> bool:
    """True for a cylindrical layer that fills the space around power cores.

    Declared either by the ``pp_filler`` material or, for any filler material,
    as a non-structural annulus starting at the axis (``r_inner_m = 0``) of a
    multi-core cable. Such a disc necessarily encloses the cores, so its full
    area would count the core envelopes twice.
    """
    if not isinstance(layer, CylindricalLayer):
        return False
    if layer.material == "pp_filler":
        return True
    return (
        not layer.structural
        and layer.r_inner_m == 0.0
        and any(
            isinstance(x, CylindricalOffsetLayer) and x.n_cores > 1
            for x in cable.layers
        )
    )


def filler_area(layer: CylindricalLayer, cable: CableData) -> float:
    """Declared filler annulus minus its intersection with core envelopes.

    All offset layers must belong to one concentric or uniformly spaced
    core family. Distinct overlapping families need an explicit geometry
    description; subtracting them separately would count overlaps twice.
    """
    if layer.r_inner_m is None or layer.r_outer_m is None:
        raise CableSchemaError("filler area requires complete radii")
    groups: dict[tuple[int, float], float] = {}
    for core in cable.layers:
        if isinstance(core, CylindricalOffsetLayer):
            if core.r_offset_m is None or core.r_outer_m is None:
                raise CableSchemaError("filler area requires complete core envelopes")
            key = (core.n_cores, core.r_offset_m)
            groups[key] = max(groups.get(key, 0.0), core.r_outer_m)
    if len(groups) > 1:
        raise CableSchemaError(
            "filler area requires one explicitly defined core family"
        )
    area = math.pi * (layer.r_outer_m**2 - layer.r_inner_m**2)
    for (count, offset), radius in groups.items():
        prototype = next(
            core
            for core in cable.layers
            if isinstance(core, CylindricalOffsetLayer)
            and core.n_cores == count
            and core.r_offset_m == offset
        )
        beta = core_angle(cable, prototype)
        if count > 1 and 2 * offset * math.sin(math.pi / count) < 2 * radius * (
            1 - 1e-6
        ):
            raise CableSchemaError(
                "overlapping power-core envelopes in filler geometry"
            )
        if beta == 0:
            excluded = count * (
                _circle_overlap(layer.r_outer_m, radius, offset)
                - _circle_overlap(layer.r_inner_m, radius, offset)
            )
        else:
            major = radius / math.cos(beta)
            if count > 1:
                direction = math.pi / 2 + math.pi / count
                half_width = math.hypot(
                    radius * math.cos(direction), major * math.sin(direction)
                )
                # Touching laid cores: the inclined-cylinder ellipses of their
                # sections overlap slightly near the contact, an artefact of
                # the approximation (the true sections touch). Up to
                # LAID_CORE_OVERLAP of the half-width the shared lens, counted
                # twice, stays below 0.1% of the filler area (Menard 2023:
                # 0.31% overlap, 0.04% of the area).
                if offset * math.sin(math.pi / count) < half_width * (
                    1 - LAID_CORE_OVERLAP
                ):
                    raise CableSchemaError(
                        "laid core envelopes require separation or a "
                        "contact-deformation solution"
                    )
            excluded = count * _ellipse_annulus_overlap(
                layer.r_inner_m, layer.r_outer_m, offset, radius, major
            )
        area -= excluded
    if area < -1e-12:
        raise CableSchemaError("negative filler area after core-envelope subtraction")
    return max(0.0, area)


def material_volume_per_length(layer: Layer, cable: CableData) -> float:
    """Material volume per cable-axis length [m²], including wire lay length."""
    if isinstance(layer, HelicalLayer):
        assert layer.n_wires and layer.d_wire_m and layer.lay_angle_rad is not None
        return (
            layer.n_wires
            * math.pi
            * layer.d_wire_m**2
            / 4
            / math.cos(layer.lay_angle_rad)
        )
    if isinstance(layer, HelicalStrandedLayer):
        raise CableSchemaError(
            "nested-helix material volume needs its resolved geometry"
        )
    if isinstance(layer, CylindricalOffsetLayer):
        projection = math.cos(core_angle(cable, layer))
        if layer.strand_layers:
            return (
                layer.n_cores
                / projection
                * sum(
                    ring.area_m2 / math.cos(ring.lay_angle_rad)
                    for ring in layer.strand_layers
                )
            )
        return (
            layer.n_cores
            * offset_metal_area(layer)
            / math.cos(math.radians(layer.strand_lay_angle_deg or 0.0))
            / projection
        )
    if isinstance(layer, CylindricalShellLayer):
        assert layer.r_mid_m and layer.t_m
        # Mean-radius smooth-wall volume. The stiffness reduction is not a
        # material-volume multiplier; resolved corrugation mass needs its profile.
        return 2 * math.pi * layer.r_mid_m * layer.t_m
    if isinstance(layer, CylindricalLayer):
        if is_interstitial_filler(layer, cable):
            return filler_area(layer, cable)
        assert layer.r_inner_m is not None and layer.r_outer_m is not None
        return math.pi * (layer.r_outer_m**2 - layer.r_inner_m**2)
    raise CableSchemaError("unsupported material-volume geometry")
