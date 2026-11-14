# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""How-to 5: check a cable coiled into a static tank.

Run from a source checkout:

    python doc/howto/scripts/howto_05_coiling.py
"""
# [recipe]

import cablexsec as cx

cable = cx.load_cable(cx.chae2023_yaml_path())
for diameter in (14.4, 8.0):  # tank diameter [m]
    result = cx.compute_coiling(cable, tank_diameter_m=diameter, n_per_turn=72)
    armour = result.layers["armour"]
    print(
        f"D = {diameter:4.1f} m: kappa = twist = {result.curvature_per_m:.4f} 1/m, "
        f"torque {result.torque_Nm / 1e3:.2f} kN m"
    )
    print(
        f"  armour wire centre {armour.centre_min_Pa / 1e6:7.1f} to "
        f"{armour.centre_max_Pa / 1e6:6.1f} MPa, "
        f"surface {armour.fibre_min_Pa / 1e6:7.1f} to "
        f"{armour.fibre_max_Pa / 1e6:6.1f} MPa"
    )
    print(
        f"  bird-caging capacity {armour.sigma_cr_Pa / 1e6:.1f} MPa, "
        f"buckling {armour.buckling_utilisation:.3f}, "
        f"yield {armour.yield_utilisation:.3f}"
    )
    print(
        f"  governing: {result.governing} ({armour.governing_mode}), "
        f"utilisation {result.utilisation:.3f}, elastic {result.elastic_valid}"
    )
