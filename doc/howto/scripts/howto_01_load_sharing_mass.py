# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""How-to 1: mass properties and the tension carried by each layer.

Run from a source checkout:

    python doc/howto/scripts/howto_01_load_sharing_mass.py
"""
# [recipe]

import cablexsec as cx

cable = cx.load_cable(cx.iea_task49_66kv_yaml_path())

mass = cx.compute_mass_properties(cable)
print(f"outer diameter      {mass.outer_diameter_m * 1e3:7.1f} mm")
print(f"mass                {mass.mass_kg_per_m:7.2f} kg/m")
print(f"weight in water     {mass.submerged_weight_flooded_voids_N_per_m:7.1f} N/m")
print(f"  (voids sealed)    {mass.submerged_weight_dry_voids_N_per_m:7.1f} N/m")

for twist in ("held", "free"):
    sharing = cx.compute_load_sharing(cable, 200e3, n_tensions=3, twist=twist)
    share = sharing.share()
    print(f"share of 200 kN, twist {twist} ({sharing.route} route)")
    for name in sharing.layers:
        if abs(share[name][-1]) >= 0.005:
            print(f"  {name:<18} {share[name][-1]:6.3f}")
    print(f"  {'sum of all layers':<18} {sharing.total_N[-1] / 200e3:6.3f}")
