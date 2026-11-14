# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""How-to 7: SWT critical plane of one closed multiaxial cycle.

Run from a source checkout:

    python doc/howto/scripts/howto_07_swt_critical_plane.py
"""
# [recipe]

import numpy as np

import cablexsec as cx
from cablexsec.materials.sn_curves import get_swt_life

material = get_swt_life("copper_poon2022")
young, poisson = material.E_Pa, 0.34

# One closed cycle at one material point: axial stress with an out-of-phase shear.
phase = np.linspace(0.0, 2.0 * np.pi, 361)
stress = np.zeros((phase.size, 3, 3))
stress[:, 0, 0] = 40e6 + 60e6 * np.sin(phase)
stress[:, 0, 1] = stress[:, 1, 0] = 25e6 * np.sin(phase + 0.5)
stress[-1] = stress[0]
trace = np.trace(stress, axis1=1, axis2=2)[:, None, None]
strain = ((1 + poisson) * stress - poisson * trace * np.eye(3)) / young

# Candidate planes: normals in the x-y plane every 2 degrees.
angles = np.deg2rad(np.arange(0.0, 180.0, 2.0))
normals = np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))
result = cx.compute_swt_cycle(
    stress, strain, normals, material, material_point_id="wire surface point"
)
critical = np.rad2deg(angles[result.critical_plane_index])
print(f"critical plane normal at {critical:.0f} deg to the wire axis")
print(f"sampled SWT {result.sampled_max_swt_Pa / 1e3:.1f} kPa")
print(f"life {result.life_cycles:.3e} cycles")
x_plane = cx.compute_swt_cycle(
    stress, strain, normals[:1], material, material_point_id="wire surface point"
)
print(f"plane normal along the axis only: life {x_plane.life_cycles:.3e} cycles")
