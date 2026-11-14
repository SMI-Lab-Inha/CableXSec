# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Synthetic prescribed-traction cycle at one fixed subsurface material point.

Run: python -m examples.prescribed_contact_swt
This verifies the tensor-to-SWT workflow, not cable fatigue life. The Hertz
footprint is fixed; oscillating shear is prescribed without a friction/history
law. Neither intermediate traction ratios nor reversal imply partial slip.
"""

from __future__ import annotations

import numpy as np

from cablexsec import compute_hertz_subsurface_stress, compute_swt_cycle
from cablexsec.materials.sn_curves import SWTLifeConstants
from cablexsec.mechanics.hertz_contact import solve_contact


def main() -> None:
    young, poisson = 200e9, 0.3
    material = SWTLifeConstants(800e6, -0.5, 0, -0.6, young)
    footprint = solve_contact(1.0, 100.0, 300.0, young / (2 * (1 - poisson**2)))
    point = [0.6 * footprint.a_m, 0.2 * footprint.b_m, 0.5 * footprint.b_m]
    normal = compute_hertz_subsurface_stress(footprint, point)
    along_x = compute_hertz_subsurface_stress(
        footprint, point, tangential_traction_ratio=(1, 0)
    )
    along_y = compute_hertz_subsurface_stress(
        footprint, point, tangential_traction_ratio=(0, 1)
    )
    phase = np.linspace(0, 2 * np.pi, 721)
    stress = (
        normal.normal_stress_Pa
        + (0.3 * np.sin(phase))[:, None, None] * along_x.tangential_stress_Pa
        + (0.15 * np.cos(phase))[:, None, None] * along_y.tangential_stress_Pa
    )
    # Coincident tensile bulk stress, in the same fixed contact/material frame.
    stress[:, 0, 0] += 200e6 + 250e6 * np.sin(phase + 0.4)
    stress[-1] = stress[0]
    strain = (
        (1 + poisson) * stress
        - poisson * np.trace(stress, axis1=1, axis2=2)[:, None, None] * np.eye(3)
    ) / young
    print("Synthetic prescribed-traction cycle; fixed subsurface point:", point)
    print(
        "Normal, x-shear and y-shear quadrature orders:",
        normal.quadrature_order,
        along_x.quadrature_order,
        along_y.quadrature_order,
    )
    for step in (10, 5, 2):
        polar = np.deg2rad(np.arange(step, 91, step))
        azimuth = np.deg2rad(np.arange(0, 360, step))
        theta, phi = np.meshgrid(polar, azimuth, indexing="ij")
        normals = np.vstack(
            (
                [0, 0, 1],
                np.column_stack(
                    (
                        np.sin(theta).ravel() * np.cos(phi).ravel(),
                        np.sin(theta).ravel() * np.sin(phi).ravel(),
                        np.cos(theta).ravel(),
                    )
                ),
            )
        )
        result = compute_swt_cycle(
            stress,
            strain,
            normals,
            material,
            material_point_id="synthetic-fixed-subsurface-point",
        )
        print(
            f"{step:2d} degree grid ({len(normals)} planes): "
            f"sampled SWT {result.sampled_max_swt_Pa:.6f} Pa; "
            f"formal synthetic-material life {result.life_cycles:.6f} cycles"
        )
    print("Prescribed tractions and elastic strain; no physical life qualification.")


if __name__ == "__main__":
    main()
