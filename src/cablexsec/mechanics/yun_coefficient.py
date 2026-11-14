# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Yun (2020) shear coefficient ``xi`` for the stick-branch stiffness.

The compliant polymer layers between a helical (armour) layer and the cable
axis shear under bending, so the wire axial strain is less than the
plane-section (Witz-Tan) value. The reduction factor is

    xi = K_G / (K_G + EA_w sin^2(alpha) / r_h^2)                 ([Yun2020])

where ``K_G`` is the through-thickness shear stiffness of the intervening
polymer stack, combined as series springs ([Yun2020] Eq. 57/58 — the survey
locates the layer-compliance summation here, not the spec's "Eq. 49"):

    K_G = 1 / sum_i ( t_i / (G_i b_i) ),   G_i = E_i / (2 (1 + nu_i)).

Both K_G and EA_w refer to a single wire. The supporting width b_i is the
width assigned to that wire, never the summed width of the helical layer.

Limits: no polymer -> ``xi = 1`` (plane sections remain plane, Witz-Tan upper
bound EI_max); very soft polymer -> ``xi -> 0`` (full shear compliance).
"""

from __future__ import annotations

import math

__all__ = ["series_shear_stiffness", "shear_modulus", "yun_xi"]


def shear_modulus(E_Pa: float, nu: float) -> float:
    """Shear modulus ``G = E / (2 (1 + nu))`` [Pa]."""
    return E_Pa / (2.0 * (1.0 + nu))


def series_shear_stiffness(layers: list[tuple[float, float, float]]) -> float:
    """Series-combined shear stiffness ``K_G = 1 / sum(t_i/(G_i b_i))`` [N/m²].

    Parameters
    ----------
    layers : list of (t_i, G_i, b_i)
        Per polymer layer: thickness ``t_i`` [m], shear modulus ``G_i`` [Pa],
        and width ``b_i`` [m]. Layers in series add compliances ([Yun2020]
        Eq. 58).

    Returns
    -------
    float
        ``K_G``; ``inf`` for an empty stack (no polymer -> rigid -> xi = 1).
    """
    if not layers:
        return math.inf
    compliance = sum(t / (g * b) for (t, g, b) in layers)
    return 1.0 / compliance if compliance > 0 else math.inf


def yun_xi(
    K_G: float,
    EA_wire_N: float,
    lay_angle_rad: float,
    r_helix_m: float,
) -> float:
    """Yun shear coefficient ``xi`` ([Yun2020]).

    ``xi = K_G / (K_G + EA_w sin^2(alpha) / r_h^2)``. Returns 1.0 for an
    infinitely stiff stack (``K_G -> inf``).

    Parameters
    ----------
    K_G : float
        Series shear stiffness per wire and per wire-axis length [N/m²].
    EA_wire_N : float
        Axial stiffness of a single wire ``E_w A_w`` [N].
    lay_angle_rad : float
        Helical layer lay angle [rad].
    r_helix_m : float
        Helix radius [m].
    """
    if math.isinf(K_G):
        return 1.0
    coupling = EA_wire_N * math.sin(lay_angle_rad) ** 2 / r_helix_m**2
    return K_G / (K_G + coupling)
