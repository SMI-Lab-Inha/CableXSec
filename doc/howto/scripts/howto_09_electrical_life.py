# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""How-to 9: relative electrical (insulation) life at operating temperatures.

Run from a source checkout:

    python doc/howto/scripts/howto_09_electrical_life.py
"""
# [recipe]

import numpy as np

import cablexsec as cx

cable = cx.load_cable(cx.iea_task49_66kv_yaml_path())
curvature = np.zeros(10)  # curvature history [1/m]; used only with mechanical_beta
for temperature in (50.0, 70.0, 90.0):
    life = cx.compute_electrical_life(
        cable, curvature, T_conductor_C=temperature, T0_C=90.0
    )
    print(
        f"T = {temperature:4.1f} C: field {life.E_field_Vm / 1e6:.2f} kV/mm, "
        f"L/L0 = {life.life_factor:.3g}"
    )
