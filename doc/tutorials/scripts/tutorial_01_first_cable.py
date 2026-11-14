# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tutorial 1: your first cable.

Loads two bundled published cables, prints their layers, stiffness bounds and
armour contact state under tension, and writes one figure. Run from a source
checkout:

    python doc/tutorials/scripts/tutorial_01_first_cable.py [output_folder]

The figure goes to ``doc/tutorials/figures`` unless a folder is given.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import cablexsec as cx
from cablexsec.mechanics.papailiou import (
    full_slip_curvature,
    slip_onset_curvature,
)
from cablexsec.mechanics.radial_cable import solve_cable_radial_states

FIGURES = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path(__file__).resolve().parents[1] / "figures"
)

# [step-load]
cable = cx.load_cable(cx.menard2023_yaml_path())
print(cable.cable.name)
print(f"source: {cable.cable.source_document}")
print(f"{'layer':<18} {'type':<19} {'material':<24} geometry")
for layer in cable.layers:
    if layer.layer_type == "helical":
        geometry = (
            f"{layer.n_wires} wires, d = {layer.d_wire_m * 1e3:.1f} mm, "
            f"lay {layer.lay_angle_deg:.1f} deg, R = {layer.r_helix_m * 1e3:.1f} mm"
        )
    elif layer.layer_type == "cylindrical_offset":
        geometry = (
            f"{layer.n_cores} cores at {layer.r_offset_m * 1e3:.1f} mm, "
            f"r = {layer.r_inner_m * 1e3:.1f}-{layer.r_outer_m * 1e3:.1f} mm"
        )
    else:
        geometry = f"r = {layer.r_inner_m * 1e3:.1f}-{layer.r_outer_m * 1e3:.1f} mm"
    print(f"{layer.name:<18} {layer.layer_type:<19} {layer.material:<24} {geometry}")
# [step-stiffness]
stiffness = cx.compute_stiffness(cable)
print(f"EA, twist fixed      {stiffness.ea_fixed_N / 1e6:8.1f} MN")
print(f"EA, twist free       {stiffness.ea_free_N / 1e6:8.1f} MN")
print(f"GJ (armour lay)      {stiffness.gj_Nm2 / 1e3:8.1f} kN m^2")
print(f"tension-twist K_et   {stiffness.axial_torsion_coupling_Nm / 1e3:8.1f} kN m")
print(f"EI full slip         {stiffness.ei_min_Nm2 / 1e3:8.2f} kN m^2")
print(f"EI no slip (xi)      {stiffness.ei_max_Nm2 / 1e3:8.2f} kN m^2")
print(f"EI plane section     {stiffness.ei_plane_section_Nm2 / 1e3:8.2f} kN m^2")
for name, xi in stiffness.xi_per_layer.items():
    print(f"xi {name:<17} {xi:8.3f}")
# [step-contact]
print(cx.clamping_state(cable).describe())
tensions = (0.0, 50e3, 100e3)  # N
solved = solve_cable_radial_states(cable, tensions)
print(
    f"{'T [kN]':>6} {'layer':<13} {'P bounded':>9} {'P solved':>9}"
    f" {'k onset':>8} {'k full':>8}"
)
for tension, state in zip(tensions, solved, strict=True):
    specs = cx.cable_to_mkappa_specs(cable, tension, contact_model="bounded")
    p_solved = state.armour_contact_pressure()
    for layer, spec in zip(cable.helical_layers, specs, strict=True):
        print(
            f"{tension / 1e3:6.0f} {layer.name:<13}"
            f" {spec.contact_pressure_Pa / 1e6:9.3f} {p_solved[layer.name] / 1e6:9.3f}"
            f" {slip_onset_curvature(spec):8.4f} {full_slip_curvature(spec):8.4f}"
        )
# [step-template]
template = cx.load_cable(cx.iea_task49_66kv_yaml_path())
template_stiffness = cx.compute_stiffness(template)
print(template.cable.name)
print(f"EA, twist fixed      {template_stiffness.ea_fixed_N / 1e6:8.1f} MN")
print(f"EI full slip         {template_stiffness.ei_min_Nm2 / 1e3:8.2f} kN m^2")
print(f"EI no slip (xi)      {template_stiffness.ei_max_Nm2 / 1e3:8.2f} kN m^2")
print(cx.clamping_state(template).describe())
specs = cx.cable_to_mkappa_specs(template, 100e3, contact_model="bounded")
for layer, spec in zip(template.helical_layers, specs, strict=True):
    print(
        f"100 kN {layer.name:<13} P = {spec.contact_pressure_Pa / 1e6:.3f} MPa,"
        f" slip onset {slip_onset_curvature(spec):.5f} 1/m"
    )
# [step-figure]
sweep = np.linspace(0.0, 100e3, 11)  # N
bounded = [cx.cable_to_mkappa_specs(cable, t, contact_model="bounded") for t in sweep]
radial = solve_cable_radial_states(cable, sweep)
fig, (ax_contact, ax_bending) = plt.subplots(1, 2, figsize=(9.0, 3.8))
for i, (layer, colour) in enumerate(
    zip(cable.helical_layers, ("C0", "C1"), strict=True)
):
    p_bounded = [specs[i].contact_pressure_Pa / 1e6 for specs in bounded]
    p_solved = [s.armour_contact_pressure()[layer.name] / 1e6 for s in radial]
    ax_contact.plot(sweep / 1e3, p_bounded, "-", c=colour, label=f"{layer.name}")
    ax_contact.plot(sweep / 1e3, p_solved, "--", c=colour)
ax_contact.set_xlabel("tension [kN]")
ax_contact.set_ylabel("armour contact pressure P1 + P2 [MPa]")
ax_contact.set_title("Solid: bounded chain; dashed: solved")
ax_contact.legend(loc="upper left")
kappa = np.linspace(0.0, 0.02, 401)
for tension, colour in ((0.0, "black"), (100e3, "tab:red")):
    loop = cx.compute_mkappa(cable, kappa, tension_N=tension)
    ax_bending.plot(
        kappa, loop.moment_Nm, color=colour, label=f"T = {tension / 1e3:.0f} kN"
    )
    for spec in cx.cable_to_mkappa_specs(cable, tension, contact_model="bounded"):
        onset = slip_onset_curvature(spec)
        ax_bending.plot(onset, np.interp(onset, kappa, loop.moment_Nm), "o", c=colour)
ax_bending.plot(kappa, stiffness.ei_min_Nm2 * kappa, "--", c="grey", label="full slip")
ax_bending.plot(kappa, stiffness.ei_max_Nm2 * kappa, ":", c="grey", label="no slip")
ax_bending.set_ylim(0.0, 700.0)
ax_bending.set_xlabel("curvature [1/m]")
ax_bending.set_ylabel("bending moment [N m]")
ax_bending.set_title("First loading; dots: slip onset")
ax_bending.legend(loc="lower right")
fig.tight_layout()
FIGURES.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURES / "tutorial_01_first_cable.png", dpi=150)
print(f"wrote {(FIGURES / 'tutorial_01_first_cable.png').name}")
