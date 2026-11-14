# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""How-to 2: the radial contact state of a section.

Run from a source checkout:

    python doc/howto/scripts/howto_02_contact_state.py
"""
# [recipe]

import math

import numpy as np

import cablexsec as cx

# 1. Full cable, trefoil cores included: tension and external pressure.
cable = cx.load_cable(cx.iea_task49_66kv_yaml_path())
for tension, pressure in ((100e3, 0.0), (100e3, 2e6)):
    state = cx.solve_cable_radial_state(cable, tension, external_pressure_Pa=pressure)
    contact = state.armour_contact_pressure()
    wire = state.result.wire_stress_Pa
    print(
        f"T {tension / 1e3:.0f} kN, sealed p_ext {pressure / 1e6:.0f} MPa: "
        f"strain {state.result.axial_strain * 1e6:.0f} ue, "
        f"open gaps {int(np.count_nonzero(state.result.interface_gap_m > 1e-9))}"
    )
    for name in contact:
        print(
            f"  {name:<13} P1+P2 {contact[name] / 1e6:6.3f} MPa, "
            f"wire stress {wire[name] / 1e6:6.1f} MPa"
        )

# 2. Concentric cable from its definition (offset cores are refused).
single = cx.load_cable(cx.inter_array_single_core_yaml_path())
qp = cx.solve_contact_qp(single, 200e3)
print(f"single-core cable at 200 kN: {qp.n_active} contacts closed")
for radius, pressure in zip(qp.interface_radii_m, qp.contact_pressure_Pa, strict=True):
    print(f"  r = {radius * 1e3:5.1f} mm: {pressure / 1e6:6.3f} MPa")

# 3. Layers given directly: polymer core, two counter-wound armour layers.
alpha = math.radians(15.0)
layers = [
    cx.ElasticAnnulus("core", 0.0, 0.030, 1.0e9, 0.45),
    cx.ElasticHelicalRing("inner", 0.0325, 0.005, 36, alpha, 207e9),
    cx.ElasticAnnulus("bedding", 0.035, 0.037, 0.9e9, 0.45),
    cx.ElasticHelicalRing("outer", 0.0395, 0.005, 44, -alpha, 207e9),
    cx.ElasticAnnulus("sheath", 0.042, 0.047, 0.9e9, 0.45),
]
for twist in (0.0, None):  # twist held, then torque-free
    section = cx.solve_concentric_section(
        layers, tension_N=100e3, prescribed_twist_per_m=twist
    )
    held = "held" if twist is not None else "free"
    print(
        f"twist {held}: strain {section.axial_strain * 1e6:.0f} ue, "
        f"twist {section.twist_per_m * 1e3:.2f} mrad/m, "
        f"torque {section.torque_resultant_Nm:.1f} N m"
    )
    print(
        "  contact "
        + ", ".join(f"{p / 1e6:.3f}" for p in section.contact_pressure_Pa)
        + " MPa"
    )
