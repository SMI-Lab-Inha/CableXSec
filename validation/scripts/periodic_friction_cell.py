# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Independent spatial equilibrium of a helical rod on frictional support.

No CableXSec mechanics are imported. The unknown longitudinal warping u(s)
minimises rod strain energy plus the incremental spring-slider potential.
Rod strain is kappa*r*cos(alpha)^2*cos(q*s) + du/ds, q=sin(alpha)/r.
Support stiffness and Coulomb line-force limit are uniform prescribed inputs.
This is a fixed-normal-load, planar, small-strain reference, not a resolved
three-core cable or an evolving normal-contact model.

Reflection symmetry gives u=0 at phases zero and pi; only that half-turn is
meshed. This remains valid under planar reversals with uniform support and
the initially unstrained state used here. It is not an arbitrary warping
constraint for nonuniform support or non-proportional bending.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_banded

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FrictionCellResult:
    curvature_1_m: FloatArray
    moment_Nm: FloatArray  # noqa: N815 -- SI unit suffix
    rod_energy_J_m: FloatArray  # noqa: N815
    support_energy_J_m: FloatArray  # noqa: N815
    friction_J_m: FloatArray  # noqa: N815
    endpoint_work_J_m: FloatArray  # noqa: N815
    incremental_remainder_J_m: FloatArray  # noqa: N815
    equilibrium_relative_residual: FloatArray
    iterations: NDArray[np.int64]
    phi_rad: FloatArray
    final_warping_m: FloatArray
    final_strain: FloatArray
    final_support_traction_N_m: FloatArray  # noqa: N815
    final_plastic_displacement_m: FloatArray


def solve_friction_cell(
    *,
    axial_rigidity_N: float,
    helix_radius_m: float,
    lay_angle_rad: float,
    wire_count: int,
    support_N_m2: float,
    friction_limit_N_m: float,
    curvature_1_m: FloatArray,
    elements: int = 64,
    tolerance: float = 1e-10,
    maximum_iterations: int = 60,
) -> FrictionCellResult:
    """Solve uniform elastic-frictional support, retaining its plastic history.

    Linear rod elements use four Gauss points for the prescribed sinusoidal
    strain and distributed support. Plastic displacements reside at those
    integration points. A semismooth Newton method minimises the convex
    incremental potential with an Armijo line search. No fitted displacement
    factor or section-level clipping supplies the equilibrium solution.
    """
    scalars = [
        axial_rigidity_N,
        helix_radius_m,
        lay_angle_rad,
        support_N_m2,
        friction_limit_N_m,
        tolerance,
    ]
    if not np.isfinite(scalars).all():
        raise ValueError("finite mechanical inputs are required")
    if axial_rigidity_N <= 0 or helix_radius_m <= 0 or support_N_m2 <= 0:
        raise ValueError("positive rigidity, radius and support stiffness required")
    if friction_limit_N_m < 0 or tolerance <= 0:
        raise ValueError("nonnegative friction and positive tolerance required")
    if not 0 < abs(lay_angle_rad) < math.pi / 2:
        raise ValueError("nonzero lay angle below pi/2 required")
    for name, value, minimum in (
        ("wire count", wire_count, 1),
        ("elements", elements, 4),
        ("maximum iterations", maximum_iterations, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    curvature = np.asarray(curvature_1_m, dtype=float)
    if curvature.ndim != 1 or not curvature.size or not np.isfinite(curvature).all():
        raise ValueError("finite one-dimensional curvature history required")
    ea, stiffness, cap = axial_rigidity_N, support_N_m2, friction_limit_N_m
    cosine = math.cos(lay_angle_rad)
    half_length = math.pi * helix_radius_m / abs(math.sin(lay_angle_rad))
    spacing = half_length / elements
    gauss, gauss_weights = np.polynomial.legendre.leggauss(4)
    shape = np.column_stack(((1 - gauss) / 2, (1 + gauss) / 2))
    weights = gauss_weights * spacing / 2
    gradient = np.array([-1.0, 1.0]) / spacing
    phi = (np.arange(elements)[:, None] + (gauss[None, :] + 1) / 2) * math.pi / elements
    reference = helix_radius_m * cosine**2 * np.cos(phi)
    factor = wire_count / (half_length * cosine)
    u = np.zeros(elements + 1)
    plastic = np.zeros((elements, 4))
    old_strain = np.zeros_like(plastic)
    old_elastic = np.zeros_like(plastic)
    rows = np.zeros((len(curvature), 7))
    iterations = np.zeros(len(curvature), dtype=np.int64)
    cumulative_friction = cumulative_work = cumulative_remainder = previous_kappa = 0.0

    def assemble(displacement, imposed_strain, previous_plastic):
        nodal = np.column_stack((displacement[:-1], displacement[1:]))
        strain = imposed_strain + (nodal @ gradient)[:, None]
        slip = nodal @ shape.T
        trial_traction = stiffness * (slip - previous_plastic)
        traction = np.clip(trial_traction, -cap, cap)
        elastic = traction / stiffness
        local_force = ea * (strain @ weights)[:, None] * gradient[None, :] + np.einsum(
            "eq,q,qi->ei", traction, weights, shape
        )
        residual = np.zeros(elements + 1)
        residual[:-1] += local_force[:, 0]
        residual[1:] += local_force[:, 1]
        tangent = stiffness * (np.abs(trial_traction) <= cap)
        local_stiffness = np.einsum("eq,q,qi,qj->eij", tangent, weights, shape, shape)
        local_stiffness += ea * spacing * np.outer(gradient, gradient)[None, :, :]
        diagonal = local_stiffness[:-1, 1, 1] + local_stiffness[1:, 0, 0]
        off = local_stiffness[1:-1, 0, 1]
        banded = np.zeros((3, elements - 1))
        banded[1] = diagonal
        banded[0, 1:] = off
        banded[2, :-1] = off
        friction_potential = 0.5 * traction**2 / stiffness + cap * np.abs(
            slip - previous_plastic - elastic
        )
        potential = float(np.sum((0.5 * ea * strain**2 + friction_potential) @ weights))
        return residual, banded, potential, strain, slip, traction, elastic

    for step, kappa in enumerate(curvature):
        imposed_strain = kappa * reference
        # Fixed scale per station, independent of the current iterate.
        initial_force = ea * (imposed_strain @ weights)[:, None] * gradient[None, :]
        load = initial_force[:-1, 1] + initial_force[1:, 0]
        scale = max(float(np.linalg.norm(load)), 1.0)
        for iteration in range(maximum_iterations):  # noqa: B007 -- retained below
            fields = assemble(u, imposed_strain, plastic)
            residual, banded, potential = fields[:3]
            norm = float(np.linalg.norm(residual[1:-1])) / scale
            if norm <= tolerance:
                break
            increment = solve_banded(
                (1, 1), banded, -residual[1:-1], check_finite=False
            )
            directional = float(residual[1:-1] @ increment)
            step_length = 1.0
            for _ in range(40):
                candidate = u.copy()
                candidate[1:-1] += step_length * increment
                trial = assemble(candidate, imposed_strain, plastic)
                # Near equilibrium, energy differences can round to zero;
                # a strict residual reduction also accepts that Newton step.
                roundoff = (
                    32 * np.finfo(float).eps * max(abs(potential), abs(trial[2]), 1.0)
                )
                if trial[2] <= potential + 1e-4 * step_length * directional or (
                    abs(trial[2] - potential) <= roundoff
                    and np.linalg.norm(trial[0][1:-1])
                    < 0.5 * np.linalg.norm(residual[1:-1])
                ):
                    u = candidate
                    break
                step_length *= 0.5
            else:
                raise RuntimeError(f"line search failed at curvature station {step}")
        else:
            raise RuntimeError(
                f"equilibrium did not converge at curvature station {step}"
            )
        _, _, _, strain, slip, traction, elastic = fields
        next_plastic = slip - elastic
        cumulative_friction += factor * float(
            np.sum(cap * np.abs(next_plastic - plastic) @ weights)
        )
        cumulative_remainder += factor * float(
            np.sum(
                (
                    0.5 * ea * (strain - old_strain) ** 2
                    + 0.5 * stiffness * (elastic - old_elastic) ** 2
                )
                @ weights
            )
        )
        moment = factor * ea * float(np.sum((strain * reference) @ weights))
        cumulative_work += moment * (kappa - previous_kappa)
        rows[step] = [
            moment,
            factor * 0.5 * ea * np.sum(strain**2 @ weights),
            factor * 0.5 * stiffness * np.sum(elastic**2 @ weights),
            cumulative_friction,
            cumulative_work,
            cumulative_remainder,
            norm,
        ]
        iterations[step] = iteration
        old_strain, old_elastic, plastic = strain.copy(), elastic.copy(), next_plastic
        previous_kappa = float(kappa)
    return FrictionCellResult(
        curvature.copy(),
        *(rows[:, i] for i in range(7)),
        iterations,
        phi.copy(),
        slip.copy(),
        strain.copy(),
        traction.copy(),
        plastic.copy(),
    )
