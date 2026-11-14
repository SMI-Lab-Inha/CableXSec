# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Synthetic, single-point, one-cycle SWT calculation and sampling refinement.

This is an elastic verification example, not a measured cable load case or a
qualified service-life prediction. The material law is deliberately synthetic.
Run from the checkout with python -m examples.critical_plane_swt.
"""

from __future__ import annotations

import numpy as np

from cablexsec import compute_swt_cycle
from cablexsec.materials.sn_curves import SWTLifeConstants


def main() -> None:
    young, poisson = 200e9, 0.3
    material = SWTLifeConstants(800e6, -0.5, 0, -0.6, young)
    phase = np.linspace(0, 2 * np.pi, 721)
    stress = np.zeros((len(phase), 3, 3))
    stress[:, 0, 0] = 30e6 + 100e6 * np.sin(phase)
    stress[:, 1, 1] = 20e6 * np.cos(phase)
    stress[:, 0, 1] = stress[:, 1, 0] = 40e6 * np.sin(phase + 0.6)
    stress[-1] = stress[0]
    strain = (
        (1 + poisson) * stress
        - poisson * np.trace(stress, axis1=1, axis2=2)[:, None, None] * np.eye(3)
    ) / young
    print("Synthetic repeated cycle; sampled in-plane maxima, formal material life")
    for step in (10, 5, 1):
        angles = np.deg2rad(np.arange(0, 180, step))
        normals = np.column_stack(
            (np.cos(angles), np.sin(angles), np.zeros_like(angles))
        )
        result = compute_swt_cycle(
            stress, strain, normals, material, material_point_id="synthetic-point"
        )
        assert result.critical_plane_index is not None
        angle = np.rad2deg(angles[result.critical_plane_index])
        print(
            f"{step:2d} degree spacing: sampled plane {angle:.1f} degrees, "
            f"SWT {result.sampled_max_swt_Pa:.6f} Pa, "
            f"formal life {result.life_cycles:.6f} cycles"
        )
    print(result.provenance)


if __name__ == "__main__":
    main()
