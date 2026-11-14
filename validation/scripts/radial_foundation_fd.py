# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Independent periodic finite-difference solution of a unilateral radial ring.

No CableXSec mechanics are imported. This checks the declared radial operator,
not the accuracy of a Winkler foundation as a physical three-core cable.
The unknown displacement is sampled on a full circular mesh without imposing
three-fold or reflection symmetry. Foundation steps are integrated over each
nodal control interval. Compression-only reactions enter the solved energy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, diags, eye
from scipy.sparse.linalg import spsolve

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RadialFDResult:
    phi_rad: FloatArray
    displacement_m: FloatArray
    foundation_pressure_Pa: FloatArray  # noqa: N815
    stored_energy_J_per_m: float  # noqa: N815
    pressure_work_J_per_m: float  # noqa: N815
    relative_equilibrium_residual: float
    contact_fraction: float


def solve_radial_foundation_fd(
    *,
    pressure_Pa: float,
    core_stiffness_N_per_m3: float,
    filler_stiffness_N_per_m3: float,
    support_width_rad: float,
    rigidity_Nm: float,
    radius_m: float,
    core_phase_rad: float = 0.0,
    nodes: int = 1536,
) -> RadialFDResult:
    """Solve inward-positive radial displacement with pressure-controlled load."""
    values = [
        pressure_Pa,
        core_stiffness_N_per_m3,
        filler_stiffness_N_per_m3,
        support_width_rad,
        rigidity_Nm,
        radius_m,
        core_phase_rad,
    ]
    if not np.isfinite(values).all() or min(values[:6]) <= 0:
        raise ValueError(
            "finite positive pressure, stiffness, width, rigidity and radius required"
        )
    if support_width_rad > 2 * math.pi:
        raise ValueError("support width must not exceed 2pi")
    if isinstance(nodes, bool) or not isinstance(nodes, int) or nodes < 24:
        raise ValueError("at least 24 nodes required")
    h = 2 * math.pi / nodes
    phi = (np.arange(nodes) + 0.5) * h
    coverage = np.zeros(nodes)
    if support_width_rad >= 2 * math.pi / 3:
        coverage[:] = 1.0
    else:
        centres = (core_phase_rad + np.arange(3) * 2 * math.pi / 3) % (2 * math.pi)
        for centre in centres:
            for shift in (-2 * math.pi, 0.0, 2 * math.pi):
                lo = centre + shift - support_width_rad / 2
                hi = centre + shift + support_width_rad / 2
                coverage += (
                    np.maximum(
                        0.0, np.minimum(phi + h / 2, hi) - np.maximum(phi - h / 2, lo)
                    )
                    / h
                )
    support = filler_stiffness_N_per_m3 + coverage * (
        core_stiffness_N_per_m3 - filler_stiffness_N_per_m3
    )
    scale = float(np.mean(support))
    support /= scale
    rows = np.repeat(np.arange(nodes), 3)
    columns = np.column_stack(
        (
            (np.arange(nodes) - 1) % nodes,
            np.arange(nodes),
            (np.arange(nodes) + 1) % nodes,
        )
    ).ravel()
    data = np.tile(np.array([1.0, -2.0, 1.0]) / h**2, nodes)
    laplace = coo_matrix((data, (rows, columns)), shape=(nodes, nodes)).tocsr()
    operator = laplace + eye(nodes)
    ring = (rigidity_Nm / radius_m**4 / scale) * (operator.T @ operator)
    rhs = np.ones(nodes)
    displacement = spsolve(ring + diags(support), rhs)

    def potential(u: FloatArray) -> float:
        return float(
            0.5 * (u @ (ring @ u) + support @ np.maximum(u, 0.0) ** 2) - rhs @ u
        )

    for _ in range(60):
        reaction = support * np.maximum(displacement, 0.0)
        residual = ring @ displacement + reaction - rhs
        # Sparse fourth derivatives lose several digits on very fine meshes.
        error = float(np.max(abs(residual)))
        if error < 2e-7:
            break
        tangent = ring + diags(support * (displacement > 0))
        direction = spsolve(tangent, -residual)
        old_energy = potential(displacement)
        slope = float(residual @ direction)
        step = 1.0
        for _ in range(30):
            candidate = displacement + step * direction
            if potential(candidate) <= old_energy + 1e-4 * step * slope + 1e-12 * abs(
                old_energy
            ):
                displacement = candidate
                break
            step *= 0.5
        else:
            raise RuntimeError("independent ring energy minimisation failed")
    else:
        raise RuntimeError("independent ring force equilibrium not obtained")
    physical = displacement * pressure_Pa / scale
    foundation = scale * support * np.maximum(physical, 0.0)
    bend = operator @ physical
    stored = (
        0.5
        * radius_m
        * h
        * (
            rigidity_Nm / radius_m**4 * (bend @ bend)
            + scale * support @ np.maximum(physical, 0.0) ** 2
        )
    )
    work = radius_m * h * pressure_Pa * np.sum(physical)
    return RadialFDResult(
        phi,
        physical,
        foundation,
        float(stored),
        float(work),
        error,
        float(np.mean(physical > 0)),
    )
