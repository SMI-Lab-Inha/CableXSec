# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""How-to 4: longitudinal stress over the whole cross-section.

Run from a source checkout:

    python doc/howto/scripts/howto_04_section_stress.py [output_folder]

The figure goes to ``doc/howto/figures`` unless a folder is given.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import cablexsec as cx
from cablexsec.plotting.style import LABELS, SIGNED_CMAP, apply_style

apply_style()  # the MATLAB-style figures of every CableXSec plot
FIGURES = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path(__file__).resolve().parents[1] / "figures"
)

# [recipe]
cable = cx.load_cable(cx.inter_array_three_core_yaml_path())
field = cx.compute_section_stress(
    cable, 0.10, 0.0, tension_N=100e3, n_points=72, rotation_rad=np.pi / 2
)
print(f"core state: {field.core_state}")
print(f"{'layer':<18} {'kind':<8} {'min MPa':>8} {'max MPa':>8} {'use':>6}  limit")
for layer in field.layers:
    if not layer.has_values:
        continue
    usage = layer.utilisation
    peak = f"{np.nanmax(usage):6.3f}" if np.isfinite(usage).any() else "     -"
    limit = "follows the section" if layer.kinematic else layer.failure_mode
    print(
        f"{layer.name:<18} {layer.kind:<8}"
        f" {layer.sigma_min_Pa.min() / 1e6:8.1f} {layer.sigma_max_Pa.max() / 1e6:8.1f}"
        f" {peak}  {limit}"
    )

# Wire fields alone, from a wire-stress result at the same load.
stress = cx.compute_stress(cable, 0.10, 0.0, tension_N=100e3)
for wires in cx.armour_stress_fields(cable, stress, rotation_rad=np.pi / 2):
    centre = wires.sigma_Pa[:, 0, 0]
    print(
        f"{wires.name}: {centre.size} wires, centre line "
        f"{centre.min() / 1e6:.1f} to {centre.max() / 1e6:.1f} MPa"
    )

# [figure]
fig, ax = plt.subplots(figsize=(4.6, 4.0))
for layer in field.layers:
    if not layer.has_values:
        continue
    xy = layer.points_m().reshape(-1, 2) * 1e3
    marks = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=layer.sigma_Pa.reshape(-1) / 1e6,
        s=4 if layer.kind != "wires" else 10,
        cmap=SIGNED_CMAP,
        vmin=-300,
        vmax=300,
    )
fig.colorbar(marks, ax=ax, label=r"Longitudinal stress $\sigma_{zz}$ [MPa]")
ax.set_aspect("equal")
ax.set_xlabel(LABELS["x_mm"])
ax.set_ylabel(LABELS["y_mm"])
ax.set_title(r"$T$ = 100 kN, $\kappa$ = 0.1 m$^{\mathrm{-1}}$ towards $+y$", fontsize=9)
fig.tight_layout()
FIGURES.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURES / "howto_04_section_stress.png")
