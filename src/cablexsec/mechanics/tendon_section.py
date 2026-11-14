# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Cross-sections of helical armour wires: round, rectangular and profiled.

Armour tendons of flexible pipes and many power-cable armours are flat strips
(Sævik 2011, Table 1: 5 x 12.5 mm; Witz 1996, Table 2: 6 x 3 mm), not round
wires. A :class:`WireSection` carries what the section mechanics needs from
the shape of one wire:

* contact geometry: the radial thickness ``t`` (the wire occupies
  ``r_helix +- t/2``), the width ``b`` in the cylinder surface, normal to the
  wire, and the circumferential fill ``n b / (2 pi r cos a)``;
* section properties: area, the two principal second moments and the
  Saint-Venant torsion constant;
* stress recovery: the position of any perimeter point, so that the local
  bending stress of the normal (out-of-surface) curvature and of the lateral
  (in-surface) curvature can be recovered at the corners and edges.

Axes (CableXSec wire frame): ``y_n`` along the outward surface normal (the
radial direction, through the thickness) and ``y_l`` in the cylinder surface,
normal to the wire (along the width), with ``(wire tangent, y_n, y_l)``
right-handed as in
:func:`~cablexsec.mechanics.wire_stress.local_bending_fibre_stress`, whose
fibre angle ``theta`` is measured from ``+y_n`` towards ``+y_l``.

* ``I_normal = int y_n^2 dA`` resists the normal curvature (stress varies
  through the thickness): ``b t^3 / 12`` for a sharp rectangle;
* ``I_lateral = int y_l^2 dA`` resists the lateral curvature (stress varies
  across the width): ``t b^3 / 12``.

Rounded corners (radius ``r_c``): area ``b t - (4 - pi) r_c^2``; second
moments by composition (rectangle minus four corner squares plus four quarter
discs, parallel-axis theorem). The torsion constant is the exact series of
the sharp rectangle (Timoshenko and Goodier 1970, Sec. 109)::

    J = (a c^3 / 3) [1 - (192 / pi^5) (c / a) sum_{n odd} tanh(n pi a / 2c) / n^5]

with ``a >= c`` the long and short sides; the corner rounding is not
included in ``J`` (an upper value).

A ``profiled`` section (interlocked or Z/C profiles, Witz 1996 Table 2)
declares area, inertias and torsion constant from its source and uses the
envelope ``b x t`` (with an optional corner radius) for contact geometry and
for stress-point positions.

The round section reproduces the earlier circular-wire expressions exactly
(``pi d^2 / 4``, ``pi d^4 / 64``, ``pi d^4 / 32``, perimeter point
``(d/2)(cos theta, sin theta)``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from cablexsec.exceptions import CableInputError

__all__ = [
    "WIRE_SHAPES",
    "WireSection",
    "rectangle_torsion_constant",
    "rounded_rectangle_properties",
]

FloatArray = NDArray[np.float64]
WireShape = Literal["round", "rectangular", "profiled"]
WIRE_SHAPES: tuple[WireShape, ...] = ("round", "rectangular", "profiled")


def rectangle_torsion_constant(a_m: float, b_m: float, terms: int = 50) -> float:
    """Saint-Venant torsion constant of a sharp ``a x b`` rectangle [m^4].

    Exact series (Timoshenko and Goodier 1970, Sec. 109), ``terms`` odd
    terms; 50 terms converge to round-off for every aspect ratio.
    """
    if not (a_m > 0 and b_m > 0 and math.isfinite(a_m) and math.isfinite(b_m)):
        raise CableInputError("rectangle sides must be finite and positive")
    long, short = max(a_m, b_m), min(a_m, b_m)
    total = 0.0
    for k in range(terms):
        n = 2 * k + 1
        total += math.tanh(n * math.pi * long / (2 * short)) / n**5
    return long * short**3 / 3 * (1 - 192 / math.pi**5 * short / long * total)


def rounded_rectangle_properties(
    width_m: float, thickness_m: float, corner_radius_m: float = 0.0
) -> tuple[float, float, float]:
    """Area and the two centroidal second moments of a rounded rectangle.

    Returns ``(area, I_normal, I_lateral)`` with ``I_normal`` about the axis
    along the width (``int y_n^2 dA``, ``y_n`` through the thickness) and
    ``I_lateral`` about the axis through the thickness.
    """
    b, t, r = width_m, thickness_m, corner_radius_m
    if not (b > 0 and t > 0 and r >= 0):
        raise CableInputError("width and thickness must be positive, radius >= 0")
    if r > 0.5 * min(b, t) * (1 + 1e-12):
        raise CableInputError("corner radius exceeds half the smaller side")
    area = b * t - (4 - math.pi) * r * r
    disc = math.pi * r * r / 4
    own_disc = (math.pi / 16 - 4 / (9 * math.pi)) * r**4
    own_square = r**4 / 12

    def second_moment(along: float, across: float) -> float:
        # Moment about the axis along ``along``; distances measured across.
        base = along * across**3 / 12
        if r == 0:
            return base
        square_arm = across / 2 - r / 2
        disc_arm = across / 2 - r + 4 * r / (3 * math.pi)
        corner = own_disc + disc * disc_arm**2 - own_square - r * r * square_arm**2
        return base + 4 * corner

    return area, second_moment(b, t), second_moment(t, b)


@dataclass(frozen=True)
class WireSection:
    """Cross-section of one helical wire.

    Parameters
    ----------
    shape : {"round", "rectangular", "profiled"}
    diameter_m : float or None
        Round wires only.
    width_m, thickness_m : float or None
        Rectangular and profiled sections: width in the cylinder surface
        (normal to the wire) and radial thickness.
    corner_radius_m : float
        Corner radius (rectangular, and the stress-point envelope of a
        profiled section).
    area_m2, I_normal_m4, I_lateral_m4, J_m4 : float or None
        Declared properties of a profiled section (required for it, not
        accepted for the other shapes).
    """

    shape: WireShape = "round"
    diameter_m: float | None = None
    width_m: float | None = None
    thickness_m: float | None = None
    corner_radius_m: float = 0.0
    area_m2: float | None = None
    I_normal_m4: float | None = None  # noqa: N815
    I_lateral_m4: float | None = None  # noqa: N815
    J_m4: float | None = None  # noqa: N815

    def __post_init__(self) -> None:
        if self.shape not in WIRE_SHAPES:
            raise CableInputError(f"wire shape must be one of {WIRE_SHAPES}")
        declared = (self.area_m2, self.I_normal_m4, self.I_lateral_m4, self.J_m4)
        if self.shape == "round":
            d = self.diameter_m
            if d is None or not math.isfinite(d) or d <= 0:
                raise CableInputError("a round wire needs a positive diameter")
            if any(v is not None for v in declared) or self.width_m or self.thickness_m:
                raise CableInputError("a round wire takes its diameter only")
            return
        for name in ("width_m", "thickness_m"):
            value = getattr(self, name)
            if value is None or not math.isfinite(value) or value <= 0:
                raise CableInputError(f"a {self.shape} wire needs a positive {name}")
        r = self.corner_radius_m
        if not math.isfinite(r) or r < 0:
            raise CableInputError("corner_radius_m must be finite and >= 0")
        assert self.width_m is not None and self.thickness_m is not None
        if r > 0.5 * min(self.width_m, self.thickness_m) * (1 + 1e-12):
            raise CableInputError("corner radius exceeds half the smaller side")
        if self.shape == "rectangular":
            if any(v is not None for v in declared):
                raise CableInputError(
                    "a rectangular wire computes its properties; declare a "
                    "profiled section to give them"
                )
            return
        if any(v is None or not math.isfinite(v) or v <= 0 for v in declared):
            raise CableInputError(
                "a profiled wire needs positive area_m2, I_normal_m4, "
                "I_lateral_m4 and J_m4 from its source"
            )
        assert self.area_m2 is not None
        if self.area_m2 > self.width_m * self.thickness_m * (1 + 1e-9):
            raise CableInputError(
                "profiled area exceeds its width x thickness envelope"
            )

    # ------------------------------------------------------------------ #
    # Properties                                                          #
    # ------------------------------------------------------------------ #
    @property
    def is_round(self) -> bool:
        return self.shape == "round"

    @property
    def area(self) -> float:
        """Cross-sectional area [m^2]."""
        if self.shape == "round":
            assert self.diameter_m is not None
            return math.pi * self.diameter_m**2 / 4.0
        if self.shape == "profiled":
            assert self.area_m2 is not None
            return self.area_m2
        assert self.width_m is not None and self.thickness_m is not None
        return rounded_rectangle_properties(
            self.width_m, self.thickness_m, self.corner_radius_m
        )[0]

    @property
    def equivalent_diameter_m(self) -> float:
        """Diameter of the round wire of the same area [m]."""
        if self.shape == "round":
            assert self.diameter_m is not None
            return self.diameter_m
        return math.sqrt(4.0 * self.area / math.pi)

    @property
    def radial_thickness_m(self) -> float:
        """Radial extent of the wire (contact geometry) [m]."""
        if self.shape == "round":
            assert self.diameter_m is not None
            return self.diameter_m
        assert self.thickness_m is not None
        return self.thickness_m

    @property
    def surface_width_m(self) -> float:
        """Width in the cylinder surface, normal to the wire [m]."""
        if self.shape == "round":
            assert self.diameter_m is not None
            return self.diameter_m
        assert self.width_m is not None
        return self.width_m

    @property
    def I_normal(self) -> float:  # noqa: N802
        """Second moment for normal (out-of-surface) curvature [m^4]."""
        if self.shape == "round":
            assert self.diameter_m is not None
            return math.pi * self.diameter_m**4 / 64.0
        if self.shape == "profiled":
            assert self.I_normal_m4 is not None
            return self.I_normal_m4
        assert self.width_m is not None and self.thickness_m is not None
        return rounded_rectangle_properties(
            self.width_m, self.thickness_m, self.corner_radius_m
        )[1]

    @property
    def I_lateral(self) -> float:  # noqa: N802
        """Second moment for lateral (in-surface) curvature [m^4]."""
        if self.shape == "round":
            assert self.diameter_m is not None
            return math.pi * self.diameter_m**4 / 64.0
        if self.shape == "profiled":
            assert self.I_lateral_m4 is not None
            return self.I_lateral_m4
        assert self.width_m is not None and self.thickness_m is not None
        return rounded_rectangle_properties(
            self.width_m, self.thickness_m, self.corner_radius_m
        )[2]

    @property
    def J(self) -> float:  # noqa: N802
        """Saint-Venant torsion constant [m^4] (sharp rectangle; module notes)."""
        if self.shape == "round":
            assert self.diameter_m is not None
            return math.pi * self.diameter_m**4 / 32.0
        if self.shape == "profiled":
            assert self.J_m4 is not None
            return self.J_m4
        assert self.width_m is not None and self.thickness_m is not None
        return rectangle_torsion_constant(self.width_m, self.thickness_m)

    @property
    def half_thickness_m(self) -> float:
        """Distance of the outer face from the centroid along ``y_n`` [m]."""
        return 0.5 * self.radial_thickness_m

    @property
    def half_width_m(self) -> float:
        """Distance of the edge from the centroid along ``y_l`` [m]."""
        return 0.5 * self.surface_width_m

    def fill_factor(
        self, n_wires: int, r_helix_m: float, lay_angle_rad: float
    ) -> float:
        """Circumferential fill ``n b / (2 pi r cos a)`` of a layer of these wires."""
        return (
            n_wires
            * self.surface_width_m
            / (2.0 * math.pi * r_helix_m * math.cos(lay_angle_rad))
        )

    # ------------------------------------------------------------------ #
    # Stress points                                                       #
    # ------------------------------------------------------------------ #
    def perimeter_offsets(
        self, theta_rad: float | FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Normalised position ``(y_n / (t/2), y_l / (b/2))`` of perimeter points.

        The point is where the ray from the centroid at fibre angle ``theta``
        (from ``+y_n`` towards ``+y_l``) leaves the section. For a round wire
        this is ``(cos theta, sin theta)``. Multiplying by the stresses at the
        outer face and at the edge gives the bending stress at that point.
        """
        th = np.asarray(theta_rad, dtype=np.float64)
        c, s = np.cos(th), np.sin(th)
        if self.shape == "round":
            return c, s
        hn, hl = self.half_thickness_m, self.half_width_m
        r = self.corner_radius_m
        with np.errstate(divide="ignore", invalid="ignore"):
            to_face = np.where(np.abs(c) > 0, hn / np.abs(c), np.inf)
            to_edge = np.where(np.abs(s) > 0, hl / np.abs(s), np.inf)
        dist = np.minimum(to_face, to_edge)
        pn, pl = dist * c, dist * s
        if r > 0:
            cn = np.sign(c) * (hn - r)
            cl = np.sign(s) * (hl - r)
            corner = (np.abs(pn) > hn - r) & (np.abs(pl) > hl - r)
            dot = c * cn + s * cl
            disc = np.maximum(dot * dot - (cn * cn + cl * cl) + r * r, 0.0)
            dist_c = dot + np.sqrt(disc)
            pn = np.where(corner, dist_c * c, pn)
            pl = np.where(corner, dist_c * s, pl)
        return pn / hn, pl / hl

    def corner_offsets(self) -> tuple[FloatArray, FloatArray]:
        """The four extreme stress points of a rectangle, normalised.

        For a sharp rectangle these are the corners ``(+-1, +-1)``. With a
        corner radius the extreme point of a linear field lies on the arc;
        the 45-degree arc points are returned (the exact extreme depends on
        the stress ratio, see :meth:`max_bending_stress`). Round wires return
        the four points at 45 degrees on the circle.
        """
        sign_n = np.array([1.0, 1.0, -1.0, -1.0])
        sign_l = np.array([1.0, -1.0, 1.0, -1.0])
        if self.shape == "round":
            h = math.sqrt(0.5)
            return sign_n * h, sign_l * h
        hn, hl, r = self.half_thickness_m, self.half_width_m, self.corner_radius_m
        k = r * (1 - math.sqrt(0.5))
        return sign_n * (hn - k) / hn, sign_l * (hl - k) / hl

    def max_bending_stress(
        self, sigma_face: float | FloatArray, sigma_edge: float | FloatArray
    ) -> FloatArray:
        """Largest ``|sigma|`` over the section of a linear bending field.

        ``sigma_face`` is the stress at the outer face (``y_n = t/2``) and
        ``sigma_edge`` at the edge (``y_l = b/2``) of the same field. Round:
        ``hypot``. Rounded rectangle:
        ``|a| (t/2 - r) + |c| (b/2 - r) + r hypot(a, c)`` with ``a, c`` the
        stress gradients.
        """
        f = np.abs(np.asarray(sigma_face, dtype=np.float64))
        e = np.abs(np.asarray(sigma_edge, dtype=np.float64))
        if self.shape == "round":
            return np.asarray(np.hypot(f, e))
        hn, hl, r = self.half_thickness_m, self.half_width_m, self.corner_radius_m
        a, c = f / hn, e / hl
        return np.asarray(a * (hn - r) + c * (hl - r) + r * np.hypot(a, c))


def round_section(d_wire_m: float) -> WireSection:
    """Round wire of diameter ``d_wire_m``."""
    return WireSection(shape="round", diameter_m=d_wire_m)
