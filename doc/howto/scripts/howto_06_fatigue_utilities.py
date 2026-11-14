# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""How-to 6: fatigue utilities on one synthetic load record.

Run from a source checkout:

    python doc/howto/scripts/howto_06_fatigue_utilities.py
"""
# [recipe]

import numpy as np

import cablexsec as cx

cable = cx.load_cable(cx.inter_array_three_core_yaml_path())
series = cx.jonswap_series(
    hs_m=3.0,
    tp_s=9.0,
    duration_s=600.0,
    dt_s=0.5,
    rms_curvature_1_m=0.02,
    mean_tension_N=80e3,
    dynamic_tension_rms_N=5e3,
    out_of_plane_fraction=0.3,
    tension_kind="true",
    seed=1,
)

# Armour fatigue at fixed fibres of every wire (rainflow and Miner).
result = cx.fatigue_from_series(cable, series, n_fibres=16)
worst = {name: float(d.max()) for name, d in result.D_total_per_wire.items()}
print(
    f"armour: critical layer {result.critical_layer}, "
    f"wire at {result.phi_critical_deg:.0f} deg, "
    f"fibre at {result.fibre_critical_deg:.0f} deg"
)
for name, damage in worst.items():
    print(f"  {name:<13} damage in 600 s {damage:.3e}")

# Strain-life damage of a copper fibre 25 mm from the neutral axis.
strain = series.curvature_x_1_m * 0.025
copper = cx.strain_life_damage(strain, "copper_poon2022", correction="swt")
print(
    f"copper fibre: {copper.n_cycles:.0f} cycles, "
    f"max amplitude {copper.strain_amplitude_max * 1e6:.0f} ue, "
    f"damage {copper.damage:.3e}"
)

# Gumbel fit of ten 60 s block maxima of the curvature magnitude.
magnitude = np.hypot(series.curvature_x_1_m, series.curvature_y_1_m)
maxima = magnitude.reshape(10, -1).max(axis=1)
fit = cx.gumbel_fit(maxima, fractile=0.9, duration_ratio=60.0)
print(
    f"curvature: mean 60 s maximum {maxima.mean():.4f} 1/m, "
    f"1 h most probable {fit.most_probable:.4f}, "
    f"90 % fractile {fit.fractile_value:.4f} 1/m"
)

# Wire slip along the helix over the first 60 s.
window = slice(0, 120)
kinematics = cx.kinematic_history(
    cable,
    series.curvature_x_1_m[window],
    series.curvature_y_1_m[window],
    time_s=series.time_s[window],
    tension_N=series.tension_N[window],
)
for layer in kinematics.layers:
    slip = layer.slip_m
    print(
        f"{layer.name:<13} slip range {np.ptp(slip, axis=1).max() * 1e6:.1f} um, "
        f"slipping share {layer.slipping.mean():.2f}"
    )
