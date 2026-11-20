# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Support compliance of a stuck helical wire: layered annulus and contact strip.

A stuck wire follows the plane-section strain of the body that supports it,
less the relative displacement taken up by the polymer between them. With a
per-wire, per-length stiffness ``K`` the stick coefficient is
``xi = K / (K + EA sin^2(alpha) / r^2)``
(:func:`~cablexsec.mechanics.yun_coefficient.yun_xi`).

[Yun2020] take ``K`` as thin-layer strips, ``1/K = sum t_i / (G_i b)``.
That form assumes a support that follows plane sections under the stack, and
local shear of a strip of the wire's width.

This module supplies the ``"annulus"`` stick support of
:func:`~cablexsec.mechanics.stiffness.xi_for_layer`. On average along the
cable, the friction tractions of a stuck layer do not depend on the axial
position and vary as ``sin(phi)`` around it. The exact elastic response of a
layered annulus to that mode-1 axial surface traction is the antiplane field
``w = (A r + B / r) sin(phi)`` in each layer. Its term ``A r sin(phi)`` is
the transverse shear tilt of the supporting body.

:func:`annulus_mode1_compliance` gives the surface compliance for three inner
boundaries:

- a rigid support (``w = 0``): a helical layer or laid cores inside;
- a free bore (no shear stress on an empty inner surface);
- the axis (regular solution).

For a thin layer on a rigid support it reduces to ``t / G``. A hollow polymer
core, which shears as a beam, is several times softer.

The traction under each wire is concentrated on a strip of the wire's surface
width. :func:`strip_concentration_compliance` adds the ``k >= 1`` harmonics of
that periodic strip load on the layer under the wire.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np

from cablexsec.cables.schema import _radial_tol

__all__ = [
    "SupportCondition",
    "annulus_mode1_compliance",
    "strip_concentration_compliance",
]

SupportCondition = Literal["rigid", "bore", "axis"]


def _propagate(
    w: float, sigma: float, r_a: float, r_b: float, g: float
) -> tuple[float, float]:
    """Carry (w, sigma_rz) from ``r_a`` to ``r_b`` through one layer."""
    a = 0.5 * (w / r_a + sigma / g)
    b = 0.5 * r_a * r_a * (w / r_a - sigma / g)
    return a * r_b + b / r_b, g * (a - b / (r_b * r_b))


def annulus_mode1_compliance(
    layers: Sequence[tuple[float, float, float]], support: SupportCondition
) -> float:
    """Surface compliance ``w(r_c) / tau_z`` [m/Pa] of a layered annulus.

    Parameters
    ----------
    layers : sequence of (r_inner, r_outer, G)
        Layers from the outside in: radii [m], shear modulus [Pa]. A radial
        gap between consecutive layers is rigid: it keeps the displacement
        and the shear flow ``sigma r``. An overlap within the cable schema's
        radial tolerance is a touching interface; a larger one is refused.
    support : {"rigid", "bore", "axis"}
        Inner boundary: ``w = 0``; zero shear stress; or regular at ``r = 0``
        (then the innermost layer must start at the axis).

    Returns
    -------
    float
        Axial surface displacement amplitude per unit axial traction
        amplitude at the outer radius; ``0`` for an empty stack.
    """
    if not layers:
        return 0.0
    seq = list(reversed(layers))
    for r_in, r_out, g in seq:
        if not (math.isfinite(r_in) and math.isfinite(r_out) and math.isfinite(g)):
            raise ValueError("layer radii and shear modulus must be finite")
        if r_out <= r_in or r_in < 0.0 or g <= 0.0:
            raise ValueError("each layer needs 0 <= r_inner < r_outer and G > 0")
    if support == "axis":
        if seq[0][0] != 0.0:
            raise ValueError("an axis support needs an innermost layer at r = 0")
        w, sigma = seq[0][1], seq[0][2]  # w = A r with A = 1
        r_prev = seq[0][1]
        seq = seq[1:]
    elif support == "rigid":
        w, sigma, r_prev = 0.0, 1.0, seq[0][0]
    elif support == "bore":
        w, sigma, r_prev = 1.0, 0.0, seq[0][0]
    else:
        raise ValueError(f"unknown support condition {support!r}")
    if support != "axis" and r_prev <= 0.0:
        raise ValueError("a rigid or bore support needs an inner radius above zero")
    for r_in, r_out, g in seq:
        if r_in < r_prev:
            # Interfaces the cable schema accepts as touching (radius
            # rounding) start at the previous outer radius; a genuine
            # overlap is refused.
            if r_prev - r_in > _radial_tol(r_prev):
                raise ValueError("layers must not overlap")
            r_in = r_prev
        if r_in > r_prev:
            sigma *= r_prev / r_in
        w, sigma = _propagate(w, sigma, r_in, r_out, g)
        r_prev = r_out
    return w / sigma


def strip_concentration_compliance(
    spacing_m: float,
    half_width_m: float,
    thickness_m: float,
    shear_modulus_Pa: float,
) -> float:
    """Per-wire compliance [m^2/N] of the concentrated part of the contact load.

    Uniform traction on strips of half-width ``a`` at spacing ``w``, with the
    displacement averaged over the strip, on a layer of thickness ``t`` over
    a rigid base (antiplane):
    ``(2 / (G w)) sum_k sinc^2(q_k a) tanh(q_k t) / q_k`` with
    ``q_k = 2 pi k / w``. Zero for full coverage (``2 a >= w``). The mean
    (``k = 0``) part is in :func:`annulus_mode1_compliance`. The sum runs to
    ``k = max(4000, 200 w / a)`` (at most two million terms), where the strip
    weight has fallen below ``1e-6`` of its first value.
    """
    w, a, t, g = spacing_m, half_width_m, thickness_m, shear_modulus_Pa
    if not all(math.isfinite(x) for x in (w, a, t, g)):
        raise ValueError("strip inputs must be finite")
    if w <= 0.0 or a <= 0.0 or t <= 0.0 or g <= 0.0:
        raise ValueError("spacing, half-width, thickness and modulus must be positive")
    if 2.0 * a >= w:
        return 0.0
    kmax = int(min(2_000_000, max(4000, math.ceil(200.0 * w / a))))
    k = np.arange(1, kmax + 1, dtype=np.float64)
    q = 2.0 * np.pi * k / w
    weight = np.sinc(q * a / np.pi) ** 2
    total = float(np.sum(weight * np.tanh(q * t) / q))
    return 2.0 * total / (g * w)
