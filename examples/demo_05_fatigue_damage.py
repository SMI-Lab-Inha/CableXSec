# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Demo 05 — fixed-fibre bulk fatigue over a prescribed biaxial history.

Rainflow counting and Miner summation are applied to fixed material fibres
around each armour wire. The template specifies bulk fatigue only; contact
fatigue life is not calculated without a qualified material/contact law.
No generic Goodman correction is applied to the adopted armour stress-range
curve. These prescribed histories are not a coupled floating-wind simulation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _console import configure_console

import cablexsec as cx
from cablexsec.plotting.style import COLORS, LABELS, apply_style

configure_console()

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)


def main() -> None:
    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    # A 1-hour storm-like curvature history (illustrative, biaxial).
    t = np.linspace(0.0, 3600.0, 4000)
    kappa_x = (
        0.010
        * np.sin(2 * np.pi * 0.1 * t)
        * (0.5 + 0.5 * np.sin(2 * np.pi * 0.005 * t))
    )
    kappa_y = 0.004 * np.sin(2 * np.pi * 0.13 * t)
    fat = cx.compute_fatigue(cable, kappa_x, kappa_y, contact_tension_N=8.0e4)

    print("=== per-wire fatigue (1-hour synthetic history) ===")
    print(f"  critical layer    : {fat.critical_layer}")
    print(
        f"  critical wire φ    : {fat.phi_critical_deg:.0f} deg (above core: {fat.above_core})"
    )
    print(f"  governing branch   : {fat.governing}")
    print(f"  governing fibre   : {fat.fibre_critical_deg:.1f} deg from radial")
    crit = fat.critical_layer
    dmax = float(np.max(fat.D_total_per_wire[crit]))
    print(f"  peak D_total/cycleset: {dmax:.3e}")
    print("  Contact fatigue life is not assessed in this bulk-only example.")

    apply_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    n = len(fat.D_total_per_wire[crit])
    phi = np.degrees(2 * np.pi * np.arange(n) / n)
    ax.bar(
        phi,
        fat.D_bulk_per_wire[crit],
        width=360 / n,
        color=COLORS["BLUE"],
        label=r"$D_\mathrm{bulk}$",
    )
    ax.set_xlabel(LABELS["azimuth"])
    ax.set_ylabel(r"Miner damage over the history, $D$ [–]")
    ax.set_title(f"Fatigue damage per wire — {crit}")
    ax.legend()
    fig.savefig(OUT / "demo_05_fatigue_damage.png")
    print(f"\nsaved {OUT / 'demo_05_fatigue_damage.png'}")


if __name__ == "__main__":
    main()
