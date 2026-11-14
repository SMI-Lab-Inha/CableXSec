# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""How-to 8: frictional heating of an energised cable in cyclic bending.

Run from a source checkout:

    python doc/howto/scripts/howto_08_frictional_heating.py
"""
# [recipe]

import math

import numpy as np

import cablexsec as cx
from cablexsec.analysis.frictional_heating import (
    CyclicBending,
    simulate_frictional_heating,
)

cable = cx.load_cable(cx.iea_task49_66kv_yaml_path())
network = cx.thermal_network(
    cable,
    630e-6,  # conductor area [m^2]
    T4_external_Km_W=1.0 / (math.pi * 0.184 * 8.0),  # in water, h = 8 W/(m^2 K)
    ambient_C=20.0,
)
bending = CyclicBending(
    tension_N=80e3,
    amplitude_1_m=0.05,  # reversed bending about a straight cable
    frequency_Hz=0.2,
    points_per_cycle=48,
)
time_s = np.linspace(0.0, 6 * 3600.0, 7)  # six hours, hourly output
for coupling in ("none", "two_way"):
    result = simulate_frictional_heating(
        cable,
        network,
        bending,
        time_s,
        current_A=600.0,
        coupling=coupling,
        update_interval_s=1800.0,
        max_step_s=600.0,
    )
    armour = result.layer_temperature_C["outer_armour"]
    conductor = result.node_temperature_C[:, 0]
    print(f"coupling {coupling}")
    print(
        f"  friction heat {result.total_friction_heat_W_m[0]:.2f} W/m at start, "
        f"{result.total_friction_heat_W_m[-1]:.2f} W/m at 6 h"
    )
    print(
        f"  after 6 h: conductor {conductor[-1]:.1f} C, outer armour {armour[-1]:.1f} C"
    )
