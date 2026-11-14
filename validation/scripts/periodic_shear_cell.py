# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Independent finite-element check of elastic helical-wire shear support.

This verification model imports no CableXSec mechanics. A periodic axial
warping displacement v(s) minimises wire and support energy over one helix
turn. Wire strain is e0(s) + dv/ds, with e0 = r cos(alpha)^2 kappa dot
[cos(phi), sin(phi)]. The foundation energy is K(s) v(s)^2/2 per wire metre.
The unknown displacement is solved, not prescribed from a slip scale.

This is an elastic one-wire support cell, not a three-core solid model or a
frictional crossing network. An azimuthally varying support is a declared
input, not a prediction of trefoil contact. Individual wire bending is not
included. The reported stiffness is the parallel-axis contribution alone.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PeriodicShearResult:
    phi_nodes_rad: FloatArray
    warping_per_curvature_m2: FloatArray
    quadrature_phi_rad: FloatArray
    strain_per_curvature_m: FloatArray
    moment_stiffness_Nm2: FloatArray  # noqa: N815 -- SI unit suffix
    wire_energy_stiffness_Nm2: FloatArray  # noqa: N815 -- SI unit suffix
    support_energy_stiffness_Nm2: FloatArray  # noqa: N815 -- SI unit suffix
    relative_equilibrium_residual: float


def solve_periodic_shear_cell(
    *,
    axial_rigidity_N: float,
    helix_radius_m: float,
    lay_angle_rad: float,
    wire_count: int,
    support_N_per_m2: float | Callable[[FloatArray], FloatArray],
    elements: int = 128,
) -> PeriodicShearResult:
    """Solve both unit-curvature responses using periodic linear elements.

    Support can be a nonnegative scalar or a function of quadrature azimuth.
    Zero support permits free warping; one displacement is fixed as a gauge
    and the returned field is shifted to zero nodal mean. Infinite support
    is not approximated by a large penalty; supply a finite stiffness.
    Positive and negative helix handedness give the same single-wire energy.
    """
    scalars = [axial_rigidity_N, helix_radius_m, lay_angle_rad]
    if not np.isfinite(scalars).all():
        raise ValueError("wire geometry and rigidity must be finite")
    if axial_rigidity_N <= 0 or helix_radius_m <= 0:
        raise ValueError("wire radius and rigidity must be positive")
    if not 0 < abs(lay_angle_rad) < math.pi / 2:
        raise ValueError("nonzero lay angle with magnitude below pi/2 required")
    if (
        isinstance(wire_count, bool)
        or not isinstance(wire_count, int)
        or wire_count < 1
    ):
        raise ValueError("wire_count must be a positive integer")
    if isinstance(elements, bool) or not isinstance(elements, int) or elements < 8:
        raise ValueError("at least eight elements are required")
    cosine = math.cos(lay_angle_rad)
    q = math.sin(lay_angle_rad) / helix_radius_m
    turn_length = 2 * math.pi / abs(q)
    h = turn_length / elements
    gauss, weights = np.polynomial.legendre.leggauss(4)
    shape = np.column_stack(((1 - gauss) / 2, (1 + gauss) / 2))
    gradient = np.array([-1.0, 1.0]) / h
    phi_nodes = np.arange(elements) * q * h
    phi = phi_nodes[:, None] + q * h * (gauss[None, :] + 1) / 2
    support = np.broadcast_to(
        support_N_per_m2(phi) if callable(support_N_per_m2) else support_N_per_m2,
        phi.shape,
    ).astype(float)
    if not np.isfinite(support).all() or np.any(support < 0):
        raise ValueError("support must be finite and nonnegative")
    reference_strain = (
        helix_radius_m * cosine**2 * np.stack((np.cos(phi), np.sin(phi)), axis=-1)
    )
    integration_weights = weights * h / 2
    connectivity = np.column_stack(
        (np.arange(elements), (np.arange(elements) + 1) % elements)
    )
    local = np.broadcast_to(
        axial_rigidity_N * h * np.outer(gradient, gradient), (elements, 2, 2)
    ).copy()
    local += np.einsum("eq,qi,qj,q->eij", support, shape, shape, integration_weights)
    rows = np.broadcast_to(connectivity[:, :, None], local.shape).ravel()
    columns = np.broadcast_to(connectivity[:, None, :], local.shape).ravel()
    matrix = coo_matrix(
        (local.ravel(), (rows, columns)), shape=(elements, elements)
    ).tocsr()
    local_load = -axial_rigidity_N * np.einsum(
        "i,eqk,q->eik", gradient, reference_strain, integration_weights
    )
    load = np.zeros((elements, 2))
    np.add.at(load, connectivity, local_load)
    displacement = np.zeros_like(load)
    unsupported = not np.any(support)
    free = np.arange(1 if unsupported else 0, elements)
    reduced = matrix[free][:, free]
    displacement[free] = spsolve(reduced, load[free])
    if unsupported:
        displacement -= np.mean(displacement, axis=0)
    local_displacement = displacement[connectivity]
    warping = np.einsum("qi,eik->eqk", shape, local_displacement)
    strain = (
        reference_strain
        + np.einsum("i,eik->ek", gradient, local_displacement)[:, None, :]
    )
    factor = wire_count / (turn_length * cosine)
    wire_energy = (
        factor
        * axial_rigidity_N
        * np.einsum("eqi,eqj,q->ij", strain, strain, integration_weights)
    )
    support_energy = factor * np.einsum(
        "eq,eqi,eqj,q->ij", support, warping, warping, integration_weights
    )
    moments = (
        factor
        * axial_rigidity_N
        * np.einsum("eqi,eqj,q->ij", reference_strain, strain, integration_weights)
    )
    residual = float(
        np.linalg.norm(matrix @ displacement - load) / max(np.linalg.norm(load), 1.0)
    )
    return PeriodicShearResult(
        phi_nodes,
        displacement,
        phi,
        strain,
        moments,
        wire_energy,
        support_energy,
        residual,
    )


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ea, radius, angle, count = 1.5e6, 0.05, 0.35, 24
    q = math.sin(angle) / radius
    plane = 0.5 * count * ea * radius**2 * math.cos(angle) ** 3
    rows = []
    for xi in [0.02, 0.2, 0.8, 0.98]:
        support = ea * q * q * xi / (1 - xi)
        for elements in [32, 64, 128, 256, 512]:
            result = solve_periodic_shear_cell(
                axial_rigidity_N=ea,
                helix_radius_m=radius,
                lay_angle_rad=angle,
                wire_count=count,
                support_N_per_m2=support,
                elements=elements,
            )
            calculated = result.moment_stiffness_Nm2[0, 0]
            work_matrix = (
                result.wire_energy_stiffness_Nm2 + result.support_energy_stiffness_Nm2
            )
            rows.append(
                {
                    "xi": xi,
                    "elements": elements,
                    "analytical_parallel_EI_Nm2": plane * xi,
                    "calculated_parallel_EI_Nm2": calculated,
                    "relative_stiffness_error": abs(calculated / (plane * xi) - 1),
                    "relative_work_error": float(
                        np.linalg.norm(result.moment_stiffness_Nm2 - work_matrix)
                        / np.linalg.norm(work_matrix)
                    ),
                    "equilibrium_residual": result.relative_equilibrium_residual,
                }
            )
    result = {
        "definition": "periodic elastic wire on specified shear foundation; not a trefoil or frictional-contact solution",
        "public_example_inputs": {
            "EA_N": ea,
            "radius_m": radius,
            "angle_rad": angle,
            "wire_count": count,
        },
        "comparisons": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([row for row in rows if row["elements"] == 512], indent=2))


if __name__ == "__main__":
    main()
