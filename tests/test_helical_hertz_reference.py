# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Independent implicit-surface derivatives and local normal-gap curvature."""

import math

import numpy as np
import pytest
from helical_hertz_reference import (
    circle_pair_geometry,
    hertz_line,
    levelset_derivatives,
)
from scipy.integrate import quad
from scipy.optimize import brentq


def implicit_surface(p, centre, radius, q):
    # Direct geometric construction, not the implemented derivative formula.
    x, y, z = p
    angle = -q * z
    local = np.array(
        [
            math.cos(angle) * x - math.sin(angle) * y,
            math.sin(angle) * x + math.cos(angle) * y,
        ]
    )
    return np.sum((local - centre) ** 2) - radius**2


@pytest.mark.parametrize("q", [-9.0, 0.0, 9.0])
def test_hessian_by_independent_cartesian_differentiation(q):
    p, c, r, h = np.array([0.021, 0.006, 0.0]), np.array([0.01, -0.004]), 0.0153, 1e-5
    g, H = levelset_derivatives(p[:2], c, r, q)
    basis = np.eye(3) * h

    def f(v):
        return implicit_surface(v, c, r, q)

    gradient = [(f(p + e) - f(p - e)) / (2 * h) for e in basis]
    hessian = np.array(
        [
            [
                (f(p + a + b) - f(p + a - b) - f(p - a + b) + f(p - a - b))
                / (4 * h * h)
                for b in basis
            ]
            for a in basis
        ]
    )
    np.testing.assert_allclose(g, gradient, atol=1e-10, rtol=0)
    np.testing.assert_allclose(H, hessian, atol=2e-8, rtol=0)


@pytest.mark.parametrize("q", [-9.0, 0.0, 9.0])
@pytest.mark.parametrize("cavity", [False, True])
def test_normal_gap_curvature_by_solving_surface_intersections(q, cavity):
    r = 0.0153
    radius = 2 * r / np.sqrt(3)
    ca = np.array([radius, 0.0])
    cb = np.zeros(2) if cavity else radius * np.array([-0.5, np.sqrt(3) / 2])
    rb = radius + r if cavity else r
    p = np.array([rb, 0.0]) if cavity else (ca + cb) / 2
    geometry = circle_pair_geometry(p, ca, r, cb, rb, q, cavity_b=cavity)
    s = 5e-6

    def normal_coordinate(c, rad, offset):
        point = np.r_[p, 0.0] + offset * geometry.transverse_tangent
        return brentq(
            lambda distance: implicit_surface(
                point + distance * geometry.outward_normal_a, c, rad, q
            ),
            -1e-5,
            1e-5,
            xtol=1e-17,
        )

    measured = (
        sum(normal_coordinate(cb, rb, v) - normal_coordinate(ca, r, v) for v in (-s, s))
        / s**2
    )
    assert geometry.relative_curvature_per_m == pytest.approx(measured, rel=2e-6)
    assert geometry.outward_normal_a @ geometry.line_tangent == pytest.approx(
        0, abs=1e-14
    )
    if q == 0:
        assert geometry.relative_curvature_per_m == pytest.approx(
            1 / r + (-1 if cavity else 1) / rb
        )
    if cavity:
        assert geometry.relative_curvature_per_m == pytest.approx(
            (1 / r - 1 / rb) * (1 + (q * rb) ** 2)
        )


def test_hertz_pressure_integrates_to_physical_line_load():
    result = hertz_line(2800, 130, 1.38e9, 0.4, 1.38e9, 0.4)
    b, p = result["half_width_m"], result["max_pressure_Pa"]
    force = quad(lambda x: p * np.sqrt(max(0, 1 - (x / b) ** 2)), -b, b, epsabs=1e-7)[0]
    assert force == pytest.approx(2800, rel=1e-12)
    assert result["reduced_modulus_Pa"] == pytest.approx(1.38e9 / (2 * (1 - 0.4**2)))
    assert hertz_line(0, 130, 1.38e9, 0.4, 1.38e9, 0.4)["max_pressure_Pa"] == 0


def test_incompatible_geometry_and_constants_rejected():
    with pytest.raises(ValueError, match="both circles"):
        circle_pair_geometry([0, 0], [-1, 0], 1, [2, 0], 1, 0)
    with pytest.raises(ValueError, match="opposite"):
        circle_pair_geometry([1, 0], [0, 0], 1, [0, 0], 1, 0)
    with pytest.raises(ValueError, match="elastic"):
        hertz_line(1, 1, 1, 0.5, 1, 0.3)
