# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Independent local Hertz reference for tangent screw-swept circular sections.

No CableXSec mechanics imports. Each surface is defined by
phi(x,y,z)=|R(-q*z)(x,y)-c|^2-r^2. The relative normal curvature follows from
the level-set Hessians, projected into the common tangent plane. A cavity
uses -phi. Hertz then treats a small neighbourhood as elastic half-spaces.
It does not determine the load partition of the assembled cable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContactGeometry:
    point_m: np.ndarray
    outward_normal_a: np.ndarray
    transverse_tangent: np.ndarray
    line_tangent: np.ndarray
    line_length_per_axis_m: float
    relative_curvature_per_m: float


def levelset_derivatives(point_m, centre_m, radius_m, q_per_m, *, cavity=False):
    """Gradient and Hessian at z=0, with material on the negative side."""
    p, c = np.asarray(point_m, dtype=float), np.asarray(centre_m, dtype=float)
    if (
        p.shape != (2,)
        or c.shape != (2,)
        or not np.isfinite(np.r_[p, c, radius_m, q_per_m]).all()
        or radius_m <= 0
    ):
        raise ValueError(
            "finite two-dimensional geometry, twist and positive radius required"
        )
    q = q_per_m
    gradient = 2 * np.r_[p - c, q * (c[1] * p[0] - c[0] * p[1])]
    hessian = np.array(
        [
            [2, 0, 2 * q * c[1]],
            [0, 2, -2 * q * c[0]],
            [2 * q * c[1], -2 * q * c[0], 2 * q * q * (c @ p)],
        ]
    )
    sign = -1 if cavity else 1
    return sign * gradient, sign * hessian


def circle_pair_geometry(
    point_m, centre_a_m, radius_a_m, centre_b_m, radius_b_m, q_per_m, *, cavity_b=False
):
    p = np.asarray(point_m, dtype=float)
    ga, ha = levelset_derivatives(p, centre_a_m, radius_a_m, q_per_m)
    gb, hb = levelset_derivatives(p, centre_b_m, radius_b_m, q_per_m, cavity=cavity_b)
    for c, r in ((centre_a_m, radius_a_m), (centre_b_m, radius_b_m)):
        if not np.isclose(np.linalg.norm(p - c), r, rtol=1e-10, atol=0):
            raise ValueError("declared tangency must lie on both circles")
    na, nb = ga / np.linalg.norm(ga), gb / np.linalg.norm(gb)
    if np.linalg.norm(na + nb) > 1e-10:
        raise ValueError("material normals must be opposite at tangency")
    tangent = np.array([-q_per_m * p[1], q_per_m * p[0], 1.0])
    metric = np.linalg.norm(tangent)
    tangent /= metric
    transverse = np.cross(tangent, na)
    transverse /= np.linalg.norm(transverse)
    relative = ha / np.linalg.norm(ga) + hb / np.linalg.norm(gb)
    curvature = float(transverse @ relative @ transverse)
    if curvature <= 0:
        raise ValueError("positive relative curvature required for Hertz line contact")
    return ContactGeometry(p, na, transverse, tangent, float(metric), curvature)


def hertz_line(
    force_per_contact_length_N_per_m,
    relative_curvature_per_m,
    young_a_Pa,
    poisson_a,
    young_b_Pa,
    poisson_b,
):
    values = [
        force_per_contact_length_N_per_m,
        relative_curvature_per_m,
        young_a_Pa,
        poisson_a,
        young_b_Pa,
        poisson_b,
    ]
    if (
        not np.isfinite(values).all()
        or force_per_contact_length_N_per_m < 0
        or relative_curvature_per_m <= 0
        or min(young_a_Pa, young_b_Pa) <= 0
        or not (-1 < poisson_a < 0.5 and -1 < poisson_b < 0.5)
    ):
        raise ValueError(
            "finite elastic constants, nonnegative load and positive curvature required"
        )
    reduced = 1 / ((1 - poisson_a**2) / young_a_Pa + (1 - poisson_b**2) / young_b_Pa)
    half_width = np.sqrt(
        4
        * force_per_contact_length_N_per_m
        / (np.pi * reduced * relative_curvature_per_m)
    )
    maximum = (
        2 * force_per_contact_length_N_per_m / (np.pi * half_width)
        if half_width
        else 0.0
    )
    return {
        "reduced_modulus_Pa": float(reduced),
        "half_width_m": float(half_width),
        "max_pressure_Pa": float(maximum),
    }
