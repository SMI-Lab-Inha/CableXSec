# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tutorial 4: contact loading and fretting.

Forms the armour contact pressures over a tension history, solves the Hertz
contact of crossing armour wires, follows the slip of every wire round the
section under a curvature history, classifies the fretting regime of every
contact cycle and recovers an SWT critical-plane indicator next to a crossing.
Run from a source checkout:

    python doc/tutorials/scripts/tutorial_04_contact_and_fretting.py [output_folder]

The figures go to ``doc/tutorials/figures`` unless a folder is given.
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import cablexsec as cx
from cablexsec.analysis.fatigue import rainflow_cycles, wire_stress_field
from cablexsec.cable_adapters import armour_crossing_line_loads, cable_to_fatigue_layers
from cablexsec.materials import library
from cablexsec.materials.sn_curves import SWTLifeConstants
from cablexsec.mechanics.hertz_contact import (
    crossed_cylinder_curvatures,
    layer_crossing_angle_rad,
    layer_crossing_density_per_m,
    reduced_modulus,
    reduced_shear_modulus,
    solve_contact,
)

FIGURES = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path(__file__).resolve().parents[1] / "figures"
)

# [step-cable]
cable = cx.load_cable(cx.export_three_core_yaml_path())
print(cable.cable.name)
for layer in cable.helical_layers:
    clamping = layer.manufacturing_clamping
    print(
        f"{layer.name}: {layer.n_wires} wires, d = {layer.d_wire_m * 1e3:.1f} mm, "
        f"R = {layer.r_helix_m * 1e3:.1f} mm, lay {layer.lay_angle_deg:.0f} deg "
        f"{layer.lay_direction.replace('_', ' ')}, P0 = {clamping.value_Pa / 1e6:.1f} "
        f"MPa ({clamping.basis})"
    )
# [step-history]
# A synthetic record at a sag bend: true wall tension and biaxial curvature.
duration_s = 300.0
record = cx.jonswap_series(
    hs_m=3.0,
    tp_s=9.0,
    duration_s=duration_s,
    dt_s=0.2,
    rms_curvature_1_m=0.003,
    dynamic_tension_rms_N=10.0e3,
    tension_phase_deg=30.0,
    out_of_plane_fraction=0.3,
    seed=11,
    tension_kind="true",
)
kx = 0.02 + record.curvature_x_1_m
ky = record.curvature_y_1_m
tension = 150.0e3 + record.tension_N
print(
    f"record: {kx.size} samples, curvature {np.hypot(kx, ky).min():.4f} to "
    f"{np.hypot(kx, ky).max():.4f} 1/m, tension {tension.min() / 1e3:.0f} to "
    f"{tension.max() / 1e3:.0f} kN"
)
# [step-pressure]
pressure, line_load = cx.armour_contact_histories(cable, tension)
for name in pressure:
    print(
        f"{name}: P1+P2 {pressure[name].min() / 1e6:.3f} to "
        f"{pressure[name].max() / 1e6:.3f} MPa, crossing line load "
        f"{line_load[name].min() / 1e3:.2f} to {line_load[name].max() / 1e3:.2f} kN/m"
    )
# [step-hertz]
inner = cable.get_layer("inner_armour")
outer = cable.get_layer("outer_armour")
reference_tension = float(np.mean(tension))
partner, load_per_m = armour_crossing_line_loads(cable, reference_tension)[
    "inner_armour"
]
angle_in, angle_out = (
    math.radians(inner.lay_angle_deg),
    math.radians(outer.lay_angle_deg),
)
crossing = layer_crossing_angle_rad(angle_in, angle_out)
density = layer_crossing_density_per_m(
    outer.n_wires,
    inner.r_helix_m,
    angle_in,
    1.0 if inner.lay_direction == "right_hand" else -1.0,
    outer.r_helix_m,
    angle_out,
    1.0 if outer.lay_direction == "right_hand" else -1.0,
)
crossing_force = load_per_m / density
young, poisson = (library.get_property(inner.material, p) for p in ("E_Pa", "nu"))
curv_a, curv_b = crossed_cylinder_curvatures(
    inner.d_wire_m, crossing, partner_diameter_m=outer.d_wire_m
)
hertz = solve_contact(
    crossing_force,
    curv_a,
    curv_b,
    reduced_modulus(young, poisson, young, poisson),
    reduced_shear_modulus(young, poisson, young, poisson),
)
depth = np.linspace(0.0, 3.0 * hertz.b_m, 601)
axis = cx.compute_hertz_axis_stress(hertz, depth, poisson_ratio=poisson)
print(
    f"crossing with {partner.name}: angle {math.degrees(crossing):.0f} deg, "
    f"{density:.1f} crossings per metre, {crossing_force:.1f} N per crossing"
)
print(
    f"Hertz ellipse a = {hertz.a_m * 1e6:.1f} um, b = {hertz.b_m * 1e6:.1f} um, "
    f"p0 = {hertz.p0_Pa / 1e6:.0f} MPa, K_t = {hertz.K_t_N_per_m / 1e6:.1f} N/um"
)
print(
    f"peak von Mises {axis.sampled_peak_von_mises_Pa / 1e6:.0f} MPa at depth "
    f"{axis.sampled_peak_depth_m * 1e6:.1f} um; wire yield stress "
    f"{library.get_property(inner.material, 'sigma_y_Pa') / 1e6:.0f} MPa"
)
# [step-fretting]
fretting = cx.compute_fretting(cable, kx, ky, duration_s=duration_s, tension_N=tension)
print(
    f"contact loads at {fretting.reference_tension_N / 1e3:.0f} kN; extreme fibre "
    f"of the dynamic bending at {fretting.extreme_fibre_deg:.1f} deg"
)
for key, row in fretting.summary().items():
    if fretting.faces[key].contact.normal_load <= 0.0:
        print(f"{key}: no normal load on this face")
        continue
    print(
        f"{key} on {row['partner']}: mu {row['mu']:.2f}, load "
        f"{row['line_or_point_load']:.1f} N, p0 {row['p0_MPa']:.0f} MPa, "
        f"transition amplitude {row['transition_amplitude_um']:.2f} um"
    )
    print(
        f"  largest slip amplitude {row['max_amplitude_um']:.0f} um; cycles in "
        f"stick {row['share_stick']:.0%}, partial slip "
        f"{row['share_partial_slip']:.0%}, gross slip {row['share_gross_slip']:.0%}"
    )
    print(
        f"  damage per year at the contact fibre {row['max_D_fretting_per_year']:.3f}"
        f" at {row['max_damage_offset_deg']:.0f} deg from the extreme fibre; "
        f"largest wear at {row['max_wear_offset_deg']:.0f} deg"
    )
# [step-slip]
history = cx.kinematic_history(
    cable, kx, ky, tension_N=tension, contact_pressure_Pa=pressure
)
faces = {
    "inner_armour": "inner_armour/outer_face",
    "outer_armour": "outer_armour/inner_face",
}
friction_slip = {}
for field in history.layers:
    amplitude = 0.5 * np.ptp(field.slip_m, axis=1)
    friction_slip[field.name] = (np.degrees(field.azimuth_rad), amplitude)
    full = fretting.faces[faces[field.name]].max_amplitude_m
    print(
        f"{field.name}: full-slip amplitude up to {full.max() * 1e6:.0f} um, "
        f"friction law up to {amplitude.max() * 1e6:.1f} um, "
        f"stations slipping {field.slipping.mean():.1%} of the time"
    )
# [step-index]
face = fretting.faces["inner_armour/outer_face"]
shear_slip = face.contact.mu * face.contact.p0_Pa * face.max_amplitude_m
k = int(np.argmax(shear_slip))
print(
    f"shear-slip product mu p0 delta*: largest {shear_slip.max() / 1e3:.0f} kN/m "
    f"at {face.extreme_fibre_offset_deg[k]:.0f} deg from the extreme fibre"
)
# [step-swt]
layers = cable_to_fatigue_layers(cable, contact_tension_N=float(tension[0]))
layer = next(item for item in layers if item.name == inner.name)
field = wire_stress_field(
    layer,
    kx,
    ky,
    tension,
    cx.compute_stiffness(cable).ea_fixed_N,
    contact_pressure_Pa=pressure[inner.name],
)
wire = int(np.argmax(face.D_fretting_per_year))
contact_fibre = field.at_fibre(0.0)[wire]  # the outer face touches the outer armour
largest = max(rainflow_cycles(contact_fibre), key=lambda cycle: cycle[0])
bulk_range, bulk_mean = largest[0], largest[1]
print(
    f"wire {wire} ({face.dominant_regime[wire].replace('_', ' ')}): largest bulk "
    f"cycle {bulk_range / 1e6:.1f} MPa about {bulk_mean / 1e6:.1f} MPa"
)
# One closed cycle: the bulk stress along the wire axis and a Hertz-shaped
# shear traction of mu p reversing along it, the gross-slip traction.
phase = np.linspace(0.0, 2.0 * np.pi, 181)
axis_t = np.array([math.cos(crossing / 2), math.sin(crossing / 2), 0.0])
bulk = (bulk_mean + 0.5 * bulk_range * np.sin(phase))[:, None, None] * np.outer(
    axis_t, axis_t
)
mu = face.contact.mu
polar, azimuth = np.meshgrid(
    np.radians(np.arange(10, 91, 10)), np.radians(np.arange(0, 360, 10)), indexing="ij"
)
normals = np.vstack(
    (
        [0.0, 0.0, 1.0],
        np.column_stack(
            (
                (np.sin(polar) * np.cos(azimuth)).ravel(),
                (np.sin(polar) * np.sin(azimuth)).ravel(),
                np.cos(polar).ravel(),
            )
        ),
    )
)
# compute_swt_cycle also gives a formal life from a strain-life law. No
# strain-life constants of armour steel are registered, so a placeholder law
# with the wire modulus is passed and no life is printed.
placeholder = SWTLifeConstants(1.0e9, -0.1, 0.5, -0.6, young)


def swt_indicator(stress, label):
    """Largest sampled SWT indicator and its plane for one closed cycle."""
    stress = stress.copy()
    stress[-1] = stress[0]
    trace = np.trace(stress, axis1=1, axis2=2)[:, None, None]
    strain = ((1 + poisson) * stress - poisson * trace * np.eye(3)) / young
    result = cx.compute_swt_cycle(
        stress, strain, normals, placeholder, material_point_id=label
    )
    return result.sampled_max_swt_Pa, result.critical_plane_normal


def contact_cycle(x_over_a, z_over_b):
    """Stress history of one cycle at a point of the contact frame."""
    point = [x_over_a * hertz.a_m, 0.0, z_over_b * hertz.b_m]
    normal = cx.compute_hertz_subsurface_stress(hertz, point, poisson_ratio=poisson)
    shear = cx.compute_hertz_subsurface_stress(
        hertz,
        point,
        poisson_ratio=poisson,
        tangential_traction_ratio=(mu * axis_t[0], mu * axis_t[1]),
    )
    return (
        normal.normal_stress_Pa
        + np.sin(phase)[:, None, None] * shear.tangential_stress_Pa
        + bulk
    )


bulk_only, _ = swt_indicator(bulk, "bulk cycle alone")
print(f"SWT of the bulk cycle alone {bulk_only / 1e6:.4f} MPa")
for x_over_a in (0.0, -1.0, -1.1):
    value, _ = swt_indicator(contact_cycle(x_over_a, 0.5), f"x/a {x_over_a}")
    print(f"x = {x_over_a:+.1f} a, z = 0.5 b: SWT {value / 1e6:.4f} MPa")
depths = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
profile = []
for z_over_b in depths:
    value, plane = swt_indicator(contact_cycle(-1.1, z_over_b), f"z/b {z_over_b}")
    profile.append(value)
    tilt = math.degrees(math.acos(min(1.0, abs(float(plane @ axis_t)))))
    print(
        f"x = -1.1 a, z = {z_over_b:.2f} b: SWT {value / 1e6:.4f} MPa, "
        f"plane normal {tilt:.0f} deg from the wire axis"
    )
# [step-figures]
fig, ax = plt.subplots(figsize=(6.4, 4.0), layout="constrained")
for name, colour in (("inner_armour", "C0"), ("outer_armour", "C1")):
    face_name = faces[name]
    full = fretting.faces[face_name]
    label = name.replace("_", " ")
    ax.plot(
        full.wire_azimuth_deg,
        full.max_amplitude_m * 1e6,
        color=colour,
        linestyle="--",
        label=f"{label}, full slip",
    )
    stations, amplitude = friction_slip[name]
    ax.plot(stations, amplitude * 1e6, color=colour, label=f"{label}, friction law")
ax.set_xlabel("wire azimuth from e1 [deg]")
ax.set_ylabel("slip amplitude [µm]")
ax.set_xlim(0.0, 360.0)
ax.set_xticks(np.arange(0, 361, 45))
ax.set_ylim(bottom=0.0)
fig.legend(frameon=False, fontsize=8, ncols=2, loc="outside lower center")
fig.savefig(FIGURES / "tutorial_04_slip_amplitude.png", dpi=150)
plt.close(fig)
print("wrote tutorial_04_slip_amplitude.png")

fig, ax = plt.subplots(figsize=(6.4, 3.2), layout="constrained")
ax.plot(depths, np.array(profile) / 1e6, marker="o", label="next to the crossing")
ax.axhline(bulk_only / 1e6, color="0.4", linestyle="--", label="bulk cycle alone")
ax.set_xlabel("depth z / b at x = -1.1 a")
ax.set_ylabel("SWT indicator [MPa]")
ax.set_xlim(0.0, 1.05)
ax.set_ylim(bottom=0.0)
ax.legend(frameon=False)
fig.savefig(FIGURES / "tutorial_04_swt_depth.png", dpi=150)
plt.close(fig)
print("wrote tutorial_04_swt_depth.png")
