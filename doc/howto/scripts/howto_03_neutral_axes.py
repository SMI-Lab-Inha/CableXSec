# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""How-to 3: neutral axes of a section under a turning curvature.

Run from a source checkout:

    python doc/howto/scripts/howto_03_neutral_axes.py
"""
# [recipe]

import numpy as np

import cablexsec as cx

cable = cx.load_cable(cx.inter_array_three_core_yaml_path())
# Bend in x to 0.05 1/m, then turn the curvature vector through 180 degrees.
ramp = np.linspace(0.0, 0.05, 21)
turn = np.deg2rad(np.linspace(0.0, 180.0, 37))[1:]
kappa_x = np.concatenate([ramp, 0.05 * np.cos(turn)])
kappa_y = np.concatenate([np.zeros_like(ramp), 0.05 * np.sin(turn)])
tension = np.full(kappa_x.size, 100e3)  # force on the armour [N]

result = cx.compute_neutral_axes(cable, kappa_x, kappa_y, tension_N=tension)
outer = result.layers[-1]


def deg(angle_rad: float) -> float:
    return round(float(np.rad2deg(angle_rad)), 1) + 0.0


print(f"strain axis offset {result.strain_axis_offset_m[20] * 1e3:.2f} mm")
print(f"{'kappa':>6} {'theta':>6} {'psi':>6} {'delta':>6} {'psi_L':>6} {'stick':>6}")
for i in (2, 6, 10, 20, 26, 44, 56):
    print(
        f"{result.curvature_1_m[i]:6.3f}"
        f" {deg(result.bending_direction_rad[i]):6.1f}"
        f" {deg(result.moment_direction_rad[i]):6.1f}"
        f" {deg(result.moment_angle_rad[i]):6.1f}"
        f" {deg(outer.stress_direction_rad[i]):6.1f}"
        f" {deg(outer.stick_arc_rad[i]):6.0f}"
    )
