# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tutorial 2: bending hysteresis.

Computes the moment-curvature loop of the Ménard and Cartraud (2023) cable on
the measured curvature path of their range (b), compares it with the measured
loop, splits the work into stored and dissipated parts, and shows the effect of
the friction coefficient and the clamping pressure. Run from a source checkout
(the measured loop is in ``validation/paper_data``):

    python doc/tutorials/scripts/tutorial_02_bending_hysteresis.py [output_folder]

The figures go to ``doc/tutorials/figures`` unless a folder is given.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import cablexsec as cx
from cablexsec.analysis.curve_comparison import cycle_work, insert_cycle_crossings
from cablexsec.mechanics.papailiou import slip_onset_curvature
from cablexsec.plotting.style import COLORS, LABELS, apply_style

apply_style()  # the MATLAB-style figures of every CableXSec plot
REPOSITORY = Path(__file__).resolve().parents[3]
MEASURED = REPOSITORY / "validation" / "paper_data" / "menard_fig16b_ordered.csv"
FIGURES = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path(__file__).resolve().parents[1] / "figures"
)

# [step-measured]
cable = cx.load_cable(cx.menard2023_yaml_path())
measured = np.genfromtxt(MEASURED, delimiter=",", names=True)
kappa_test = measured["curvature_1_per_m"]
moment_test = measured["moment_Nm"]
print(f"measured points: {kappa_test.size}")
print(f"curvature {kappa_test.min():.4f} to {kappa_test.max():.4f} 1/m")
print(f"moment    {moment_test.min():.0f} to {moment_test.max():.0f} N m")
# [step-loop]
path, original, crossings = insert_cycle_crossings(kappa_test, crossing=0.0)
loop = cx.compute_mkappa(cable, np.tile(path, 2), tension_N=0.0, mu=0.2)
second = slice(path.size, None)
moment_model = loop.moment_Nm[second][original]
rmse = np.sqrt(np.mean((moment_model - moment_test) ** 2)) / np.ptp(moment_test)
print(f"model moment {moment_model.min():.0f} to {moment_model.max():.0f} N m")
print(f"ordered RMSE / measured range: {100 * rmse:.2f} %")
# [step-work]
stored = loop.work.stored_J_per_m[second]
friction = loop.work.friction_J_per_m[second]
work_model = np.diff((stored + friction)[crossings])
work_test = cycle_work(kappa_test, moment_test, crossing=0.0)
print(f"stored energy at the cycle crossings: {stored[crossings].round(3)} J/m")
print(f"largest stored energy: {stored.max():.2f} J/m")
print(f"friction work per cycle, model:    {work_model.round(1)} J/m")
print(f"loop area per cycle, measured:     {work_test.round(1)} J/m")
difference = work_model.mean() / work_test.mean() - 1.0
print(f"difference of the means: {100 * difference:+.1f} %")
# [step-sensitivity]
kappa = 0.1 * np.sin(np.linspace(0.0, 4.0 * np.pi, 801))  # two cycles, 1/m
cycle = slice(400, None)  # the second, closed cycle


def describe(label, result, spec):
    """Peak moment, friction work of the second cycle and slip onset."""
    work = result.work.friction_J_per_m
    print(
        f"{label:<15} peak {result.moment_Nm[cycle].max():5.0f} N m,"
        f" friction work {work[-1] - work[400]:5.1f} J/m per cycle,"
        f" inner slip onset {slip_onset_curvature(spec):.4f} 1/m"
    )


friction_loops = {}
for mu in (0.1, 0.2, 0.3):
    friction_loops[mu] = cx.compute_mkappa(cable, kappa, mu=mu)
    describe(
        f"mu = {mu}", friction_loops[mu], cx.cable_to_mkappa_specs(cable, 0.0, mu=mu)[0]
    )
clamping_loops = {}
for pressure in (0.15e6, 0.3e6, 0.6e6):
    clamped = cable.model_copy(
        update={
            "initial_state": cable.initial_state.model_copy(
                update={"external_pressure_Pa": pressure}
            )
        }
    )
    clamping_loops[pressure] = cx.compute_mkappa(clamped, kappa)
    describe(
        f"p0 = {pressure / 1e6:.2f} MPa",
        clamping_loops[pressure],
        cx.cable_to_mkappa_specs(clamped, 0.0)[0],
    )
# [step-figures]
FIGURES.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(5.0, 4.0))
ax.plot(
    kappa_test,
    moment_test,
    "o",
    ms=3,
    mfc="none",
    c=COLORS["BLACK"],
    mew=0.6,
    label="measured",
)
ax.plot(path, loop.moment_Nm[second], c=COLORS["BLUE"], label="CableXSec")
ax.set_xlabel(LABELS["curvature"])
ax.set_xticks([-0.1, -0.05, 0.0, 0.05, 0.1])
ax.set_ylabel(LABELS["moment_Nm"])
ax.set_title("Ménard and Cartraud (2023), range (b)")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(FIGURES / "tutorial_02_measured_loop.png")
print(f"wrote {(FIGURES / 'tutorial_02_measured_loop.png').name}")

fig, (ax_mu, ax_p) = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
for mu, result in friction_loops.items():
    ax_mu.plot(kappa[cycle], result.moment_Nm[cycle], label=rf"$\mu$ = {mu}")
for pressure, result in clamping_loops.items():
    label = rf"$p_\mathrm{{0}}$ = {pressure / 1e6:.2f} MPa"
    ax_p.plot(kappa[cycle], result.moment_Nm[cycle], label=label)
ax_mu.set_title(r"Friction coefficient ($p_\mathrm{0}$ = 0.30 MPa)")
ax_p.set_title(r"Initial pressure $p_\mathrm{0}$ ($\mu$ = 0.2)")
ax_mu.set_ylabel(LABELS["moment_Nm"])
for axis in (ax_mu, ax_p):
    axis.set_xlabel(LABELS["curvature"])
    axis.set_xticks([-0.1, -0.05, 0.0, 0.05, 0.1])
    axis.legend(loc="upper left")
fig.tight_layout()
fig.savefig(FIGURES / "tutorial_02_sensitivity.png")
print(f"wrote {(FIGURES / 'tutorial_02_sensitivity.png').name}")
