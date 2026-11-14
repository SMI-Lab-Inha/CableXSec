# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Gap G1: rectangular and profiled wire sections (closed forms, V1)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.tendon_section import (
    WireSection,
    rectangle_torsion_constant,
    rounded_rectangle_properties,
)


def _grid_properties(b: float, t: float, r: float, n: int = 2001):
    """Area and second moments of a rounded rectangle on a fine grid."""
    yn = (np.arange(n) + 0.5) / n * t - t / 2
    yl = (np.arange(n) + 0.5) / n * b - b / 2
    gn, gl = np.meshgrid(yn, yl, indexing="ij")
    inside = np.ones_like(gn, dtype=bool)
    if r > 0:
        cn, cl = t / 2 - r, b / 2 - r
        corner = (np.abs(gn) > cn) & (np.abs(gl) > cl)
        far = (np.abs(gn) - cn) ** 2 + (np.abs(gl) - cl) ** 2 > r * r
        inside &= ~(corner & far)
    da = (t / n) * (b / n)
    return (
        float(inside.sum() * da),
        float((gn**2 * inside).sum() * da),
        float((gl**2 * inside).sum() * da),
    )


def test_sharp_rectangle_closed_forms():
    s = WireSection(shape="rectangular", width_m=0.0125, thickness_m=0.005)
    assert s.area == pytest.approx(0.0125 * 0.005, rel=1e-15)
    assert s.I_normal == pytest.approx(0.0125 * 0.005**3 / 12, rel=1e-15)
    assert s.I_lateral == pytest.approx(0.005 * 0.0125**3 / 12, rel=1e-15)
    assert s.radial_thickness_m == 0.005
    assert s.surface_width_m == 0.0125
    assert s.equivalent_diameter_m == pytest.approx(
        math.sqrt(4 * 0.0125 * 0.005 / math.pi), rel=1e-15
    )


@pytest.mark.parametrize("r", [0.0, 0.0005, 0.001, 0.0015])
def test_rounded_rectangle_against_grid(r):
    b, t = 0.006, 0.003
    area, i_n, i_l = rounded_rectangle_properties(b, t, r)
    assert area == pytest.approx(b * t - (4 - math.pi) * r * r, rel=1e-14)
    g_area, g_n, g_l = _grid_properties(b, t, r)
    assert area == pytest.approx(g_area, rel=2e-4)
    assert i_n == pytest.approx(g_n, rel=5e-4)
    assert i_l == pytest.approx(g_l, rel=5e-4)


def test_full_round_corners_give_a_stadium():
    # r = t/2: two half discs and a rectangle; I about the long axis exact.
    b, t = 0.006, 0.002
    r = t / 2
    area, i_n, _ = rounded_rectangle_properties(b, t, r)
    core = (b - t) * t
    assert area == pytest.approx(core + math.pi * r * r, rel=1e-14)
    assert i_n == pytest.approx((b - t) * t**3 / 12 + math.pi * r**4 / 4, rel=1e-13)


def test_torsion_constant_series():
    # Square: J = 0.140577 a^4 (Timoshenko and Goodier 1970, Table 109).
    assert rectangle_torsion_constant(1.0, 1.0) == pytest.approx(0.140577, abs=2e-6)
    # Thin strip: J -> a c^3 / 3.
    assert rectangle_torsion_constant(1.0, 1e-3) == pytest.approx(1e-9 / 3, rel=1e-3)
    for ratio in (1.0, 1.5, 2.0, 2.5, 4.0, 10.0):
        b, c = ratio, 1.0
        roark = b * c**3 * (1 / 3 - 0.21 * (c / b) * (1 - c**4 / (12 * b**4)))
        assert rectangle_torsion_constant(b, c) == pytest.approx(roark, rel=5e-3)


def test_round_section_reproduces_circle_expressions():
    d = 0.0089206
    s = WireSection(shape="round", diameter_m=d)
    assert s.area == math.pi * d**2 / 4.0
    assert s.I_normal == math.pi * d**4 / 64.0 == s.I_lateral
    assert math.pi * d**4 / 32.0 == s.J
    theta = np.linspace(0, 2 * math.pi, 13)
    n, lat = s.perimeter_offsets(theta)
    assert np.array_equal(n, np.cos(theta)) and np.array_equal(lat, np.sin(theta))


def test_perimeter_offsets_and_corners():
    s = WireSection(shape="rectangular", width_m=0.0125, thickness_m=0.005)
    n, lat = s.perimeter_offsets(
        np.array([0.0, math.pi / 2, math.atan2(0.0125, 0.005)])
    )
    assert np.allclose(n, [1.0, 0.0, 1.0]) and np.allclose(lat, [0.0, 1.0, 1.0])
    # every perimeter point lies on the boundary
    theta = np.linspace(0, 2 * math.pi, 361)
    n, lat = s.perimeter_offsets(theta)
    assert np.allclose(np.maximum(np.abs(n), np.abs(lat)), 1.0)
    rounded = WireSection(
        shape="rectangular", width_m=0.006, thickness_m=0.003, corner_radius_m=0.001
    )
    n, lat = rounded.perimeter_offsets(theta)
    yn, yl = n * 0.0015, lat * 0.003
    cn, cl = 0.0005, 0.002
    corner = (np.abs(yn) > cn) & (np.abs(yl) > cl)
    dist = np.hypot(np.abs(yn) - cn, np.abs(yl) - cl)
    assert np.allclose(dist[corner], 0.001)


@pytest.mark.parametrize("r", [0.0, 0.001])
def test_max_bending_stress_is_the_perimeter_maximum(r):
    s = WireSection(
        shape="rectangular", width_m=0.006, thickness_m=0.003, corner_radius_m=r
    )
    theta = np.linspace(0, 2 * math.pi, 200001)
    n, lat = s.perimeter_offsets(theta)
    for face, edge in ((100.0, 0.0), (0.0, 50.0), (80.0, -60.0), (-10.0, 300.0)):
        brute = float(np.max(np.abs(face * n + edge * lat)))
        # the ray grid samples the sharp corner only approximately
        assert float(s.max_bending_stress(face, edge)) == pytest.approx(brute, rel=1e-4)
        assert float(s.max_bending_stress(face, edge)) >= brute * (1 - 1e-12)


def test_fill_factor_and_validation():
    s = WireSection(shape="rectangular", width_m=0.0125, thickness_m=0.005)
    fill = s.fill_factor(55, 0.132, math.atan(math.pi * 0.264 / 1.795))
    assert fill == pytest.approx(0.913, abs=1e-3)
    with pytest.raises(CableInputError):
        WireSection(
            shape="rectangular", width_m=0.006, thickness_m=0.003, corner_radius_m=0.002
        )
    with pytest.raises(CableInputError):
        WireSection(shape="profiled", width_m=0.006, thickness_m=0.003)
    with pytest.raises(CableInputError):
        WireSection(shape="round", diameter_m=0.004, width_m=0.004)
    p = WireSection(
        shape="profiled",
        width_m=0.00925,
        thickness_m=0.0062,
        area_m2=51.5e-6,
        I_normal_m4=100e-12,
        I_lateral_m4=771e-12,
        J_m4=204.6e-12,
    )
    assert (p.area, p.I_normal, p.I_lateral, p.J) == (
        51.5e-6,
        100e-12,
        771e-12,
        204.6e-12,
    )
    assert p.radial_thickness_m == 0.0062
