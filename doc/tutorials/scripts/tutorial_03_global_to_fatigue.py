# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tutorial 3: from a global time history to armour fatigue.

Writes a synthetic section history in the layout of a time-history CSV export,
reads it back with ``read_time_history_csv``, recovers the armour wire stresses
at fixed fibres, and computes the rainflow/Miner damage of every wire and the
damage-equivalent stress range of the governing fibre. It also reads the
bundled demonstration history of a global analysis. Run from a source
checkout:

    python doc/tutorials/scripts/tutorial_03_global_to_fatigue.py [output_folder]

The figures go to ``doc/tutorials/figures`` unless a folder is given.
"""

import csv
import sys
import tempfile
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import cablexsec as cx
from cablexsec.analysis.fast_rainflow import fibre_fatigue
from cablexsec.analysis.fatigue import rainflow_cycles, wire_stress_field
from cablexsec.cable_adapters import cable_to_fatigue_layers

REPOSITORY = Path(__file__).resolve().parents[3]
FIGURES = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path(__file__).resolve().parents[1] / "figures"
)

# [step-export]
# A synthetic sag-bend record: JONSWAP-shaped in-plane and out-of-plane
# curvature about a static sag curvature, and a dynamic tension that follows
# the in-plane curvature. It stands in for the export of a global analysis.
duration_s = 600.0
record = cx.jonswap_series(
    hs_m=3.0,
    tp_s=9.0,
    duration_s=duration_s,
    dt_s=0.2,
    rms_curvature_1_m=0.01,
    dynamic_tension_rms_N=8.0e3,
    tension_phase_deg=30.0,
    out_of_plane_fraction=0.3,
    seed=7,
)
# Rotation components as a global program writes them: the sag bend of a line
# in a vertical plane is a negative rotation about e2 (y curvature).
y_curvature = -(0.05 + record.curvature_x_1_m)
x_curvature = record.curvature_y_1_m
tension_kn = (60.0e3 + record.tension_N) / 1.0e3

work = tempfile.TemporaryDirectory()
export = Path(work.name) / "sag_bend_history.csv"
with export.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(
        [
            "Time (s)",
            "x curvature (rad/m)",
            "y curvature (rad/m)",
            "Effective tension (kN)",
        ]
    )
    for row in zip(record.time_s, x_curvature, y_curvature, tension_kn, strict=True):
        writer.writerow([f"{value:.6g}" for value in row])
print("".join(export.read_text(encoding="utf-8").splitlines(keepends=True)[:3]), end="")
# [step-read]
series = cx.read_time_history_csv(
    export,
    time_col="Time",
    curvature_x_col="x curvature",
    curvature_y_col="y curvature",
    tension_col="Effective tension",
    curvature_convention="rotation",
    tension_scale_to_N=1.0e3,
)
work.cleanup()
kx, ky = series.curvature_x_1_m, series.curvature_y_1_m
print(f"samples {series.n_steps}, step {series.dt_s:.2f} s")
print(f"tension kind: {series.tension_kind}")
print(f"strain gradient x  {kx.min():+.4f} to {kx.max():+.4f} 1/m")
print(f"strain gradient y  {ky.min():+.4f} to {ky.max():+.4f} 1/m")
print(
    f"tension            {series.tension_N.min() / 1e3:.1f} to "
    f"{series.tension_N.max() / 1e3:.1f} kN"
)
# [step-demo]
demo = REPOSITORY / "validation/paper_data/global_demo/global_history_si.csv"
demo_series = cx.read_time_history_csv(
    demo,
    time_col="time_s",
    curvature_x_col="curvature_x_1_m",
    curvature_y_col="curvature_y_1_m",
    tension_col="effective_tension_N",
    curvature_convention="strain_gradient",
    tension_kind="effective",
    tension_scale_to_N=1.0,
)
same = cx.read_series_csv(demo)
print(
    f"demo history: {demo_series.n_steps} samples, step {demo_series.dt_s:.1f} s, "
    f"largest curvature "
    f"{np.hypot(demo_series.curvature_x_1_m, demo_series.curvature_y_1_m).max():.4f}"
    f" 1/m, tension {demo_series.tension_N.min() / 1e3:.1f} to "
    f"{demo_series.tension_N.max() / 1e3:.1f} kN"
)
print(
    "same arrays as read_series_csv:",
    all(
        np.array_equal(getattr(demo_series, name), getattr(same, name))
        for name in ("time_s", "curvature_x_1_m", "curvature_y_1_m", "tension_N")
    ),
)
# [step-tension]
cable = cx.load_cable(cx.iea_task49_66kv_yaml_path())
print(cable.cable.name)
print(f"armour annulus: {cable.water_ingress.armour_annulus}")
try:
    cx.fatigue_from_series(cable, series)
except cx.CableSchemaError as error:
    print(f"fatigue_from_series: {str(error).split(';')[0]}")
# [step-fatigue]
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    fatigue = cx.compute_fatigue(cable, kx, ky, tension_N=series.tension_N)
print("warnings:", ", ".join(sorted({item.category.__name__ for item in caught})))
for name, damage in fatigue.D_total_per_wire.items():
    layer = cable.get_layer(name)
    wire = int(np.argmax(damage))
    print(
        f"{name}: {layer.n_wires} wires, damage {damage.min():.3e} to "
        f"{damage.max():.3e}, largest on wire {wire} at "
        f"{360.0 * wire / layer.n_wires:.1f} deg, fibre "
        f"{fatigue.fibre_critical_deg_per_wire[name][wire]:.2f} deg"
    )
print(
    f"governing: {fatigue.critical_layer}, wire at {fatigue.phi_critical_deg:.1f} deg, "
    f"fibre {fatigue.fibre_critical_deg:.2f} deg ({fatigue.governing} damage)"
)
# [step-recover]
warnings.simplefilter("ignore", cx.MissingClampingStateWarning)  # reported above
layers = cable_to_fatigue_layers(cable, contact_tension_N=float(series.tension_N[0]))
pressure, _ = cx.armour_contact_histories(cable, series.tension_N)
ea_fixed = cx.compute_stiffness(cable).ea_fixed_N
angles = np.radians(fatigue.fibre_angles_deg)
fields = {}
for layer in layers:
    field = wire_stress_field(
        layer,
        kx,
        ky,
        series.tension_N,
        ea_fixed,
        contact_pressure_Pa=pressure[layer.name],
    )
    damage, equivalent_1hz, _, _ = fibre_fatigue(
        field.mean_Pa,
        field.radial_Pa,
        field.tangential_Pa,
        angles,
        layer.sn_bulk,
        duration_s,
    )
    fields[layer.name] = field
    wire, fibre = np.unravel_index(np.argmax(damage), damage.shape)
    print(
        f"{layer.name}: same damage as compute_fatigue "
        f"{np.allclose(damage, fatigue.D_bulk_per_fibre[layer.name], rtol=1e-10)}, "
        f"1 Hz equivalent range {equivalent_1hz.min():.2f} to "
        f"{equivalent_1hz.max():.2f} MPa"
    )
# [step-equivalent]
name = fatigue.critical_layer
layer = next(item for item in layers if item.name == name)
wire = int(np.argmax(fatigue.D_total_per_wire[name]))
fibre_deg = fatigue.fibre_critical_deg_per_wire[name][wire]
stress = fields[name].at_fibre(np.radians(fibre_deg))[wire]
cycles = list(rainflow_cycles(stress))
ranges_mpa = np.array([cycle[0] for cycle in cycles]) / 1e6
counts = np.array([cycle[2] for cycle in cycles])
m = layer.sn_bulk.m1
counted = counts.sum()
equivalent_mpa = (np.sum(counts * ranges_mpa**m) / counted) ** (1.0 / m)
equivalent_1hz_mpa = (np.sum(counts * ranges_mpa**m) / duration_s) ** (1.0 / m)
print(f"S-N curve {layer.sn_bulk.name}: slope m = {m:g}, {layer.sn_bulk.reference}")
print(
    f"stress at the governing fibre: {stress.min() / 1e6:.1f} to "
    f"{stress.max() / 1e6:.1f} MPa"
)
print(f"rainflow: {counted:.1f} cycles, largest range {ranges_mpa.max():.2f} MPa")
print(f"equivalent range over the counted cycles {equivalent_mpa:.2f} MPa")
print(f"equivalent range at 1 Hz, {duration_s:.0f} cycles {equivalent_1hz_mpa:.2f} MPa")
print(
    f"damage from the equivalent range "
    f"{counted / layer.sn_bulk.cycles_to_failure(equivalent_mpa):.4e}, "
    f"from compute_fatigue {fatigue.D_total_per_wire[name][wire]:.4e}"
)
# [step-figures]
fig, ax = plt.subplots(figsize=(6.4, 3.4), layout="constrained")
for name, damage in fatigue.D_total_per_wire.items():
    count = damage.size
    ax.plot(360.0 * np.arange(count) / count, damage, label=name.replace("_", " "))
ax.axvline(fatigue.phi_critical_deg, color="0.5", linewidth=0.8, linestyle="--")
ax.set_xlabel("wire azimuth from e1 [deg]")
ax.set_ylabel("Miner damage over the record")
ax.set_xlim(0.0, 360.0)
ax.set_xticks(np.arange(0, 361, 45))
ax.set_ylim(bottom=0.0)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.legend(frameon=False, loc="lower center")
fig.savefig(FIGURES / "tutorial_03_damage_circumference.png", dpi=150)
plt.close(fig)
print("wrote tutorial_03_damage_circumference.png")

fig, ax = plt.subplots(figsize=(6.4, 3.0), layout="constrained")
window = series.time_s <= 120.0
ax.plot(series.time_s[window], stress[window] / 1e6, linewidth=0.9)
ax.set_xlabel("time [s]")
ax.set_ylabel("fibre stress [MPa]")
ax.set_xlim(0.0, 120.0)
ax.set_title(
    f"{fatigue.critical_layer.replace('_', ' ')}, wire at "
    f"{fatigue.phi_critical_deg:.0f} deg, fibre {fatigue.fibre_critical_deg:.0f} deg",
    fontsize=10,
)
fig.savefig(FIGURES / "tutorial_03_governing_fibre_stress.png", dpi=150)
plt.close(fig)
print("wrote tutorial_03_governing_fibre_stress.png")
