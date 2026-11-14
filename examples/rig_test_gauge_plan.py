# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Gauge plan for a combined tension--bending rig test (public 33 kV three-core).

A 4 m specimen of the public three-core 33 kV template is tensioned to
100 kN and bent cyclically in one plane at bending radii of 10, 5 and 3 m,
with the armour anchored at both clamps. For each radius the script prints the
layer friction scales, the recommended gauges with their expected strain
ranges against an assumed 2 microstrain resolution, and the clamp contrast
(range with the clamp minus the sectional range). It also writes the strain
histories of the recommended gauges and the moment--curvature loops to
``examples/output/rig_test_gauge_plan.png``.

The cable definition is a public template, not a manufacturer design; the
numbers illustrate the workflow and are not a prediction for a real product.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from _console import configure_console

import cablexsec as cx
from cablexsec.plotting.style import apply_style

configure_console()

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)


def main() -> None:
    cable = cx.load_cable(cx.inter_array_three_core_yaml_path(), geometry_override=None)
    setup = cx.RigSetup(free_length_m=4.0, profile="uniform", clamp="anchored")
    results = {}
    for radius in (10.0, 5.0, 3.0):
        protocol = cx.RigProtocol.cyclic(
            tension_N=100e3, curvature_amplitude_1_m=1.0 / radius, n_cycles=3
        )
        res = cx.simulate_rig_test(cable, protocol, setup)
        results[radius] = res
        print(f"\n=== bending radius {radius:g} m, tension 100 kN ===")
        print(res.summary())
        for note in res.notes:
            print("note:", note)

    print("\nOuter armour, wire on the neutral axis at the anchor, x = 0.05 m")
    print("(ranges in microstrain; friction amplitude = half range):")
    print("  R [m]  total range  sectional range  friction ampl.  clamp-law peak")
    for radius, res in results.items():
        near = [
            g
            for g in res.gauges
            if g.location.layer == "outer_armour" and g.location.x_m == 0.05
        ]
        g = max(near, key=lambda item: item.clamp_contrast)
        law = next(s for s in res.layers if s.layer == "outer_armour")
        print(
            f"  {radius:5.1f}  {g.ranges['total']:11.0f}  {g.range_sectional:15.0f}"
            f"  {g.ranges['friction'] / 2:14.0f}"
            f"  {law.clamp_law_peak_microstrain:14.0f}"
        )

    res = results[5.0]
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for rec in res.recommendations[:4]:
        g = rec.gauge
        axes[0].plot(
            res.time_s,
            g.total,
            label=f"{g.location.layer} {g.angle_deg:.0f} deg, x={g.location.x_m:.2f} m",
        )
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("wire strain [microstrain]")
    axes[0].legend(fontsize=7)
    for station in res.stations:
        r = res.retained
        axes[1].plot(
            station.kappa_1_m[r],
            station.moment_Nm[r] / 1e3,
            label=f"x = {station.x_m:.1f} m",
        )
    axes[1].set_xlabel("curvature [1/m]")
    axes[1].set_ylabel("moment [kN m]")
    axes[1].legend()
    fig.tight_layout()
    path = OUT / "rig_test_gauge_plan.png"
    fig.savefig(path)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
