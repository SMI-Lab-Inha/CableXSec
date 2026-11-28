# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Torque balance of the public 66 kV template and the example cables.

For each packaged public cable the script prints the torque/tension
coefficient with the ends restrained against twist, the twist of a free end
per 100 kN, the torque-balance ratio of the armour and the free-twist
redistribution of wire stress, from the solved section and from the
fixed-radius matrix of ``xsec_stiffness``. For the IEA Wind Task 49 66 kV
template it then finds the outer-armour lay angle, wire count and wire
diameter that zero the torque at 100 kN, and the lay angle for the other core
lay direction and for aligned cores (armour alone).

The cable definitions are public templates and representative examples, not
manufacturer designs; their lay directions are assumptions (see the YAML
headers). Run with one thread: ``python examples/torque_balance_templates.py``.
"""

from __future__ import annotations

import math
import warnings

from _console import configure_console

import cablexsec as cx
from cablexsec.analysis.torque_balance import armour_fill_ratio
from cablexsec.exceptions import CableSchemaError
from cablexsec.mechanics.stiffness import axial_torsional_matrix

configure_console()

TENSION_N = 100.0e3
CABLES = (
    ("IEA 66 kV template", cx.iea_task49_66kv_yaml_path),
    ("33 kV single-core", cx.inter_array_single_core_yaml_path),
    ("33 kV three-core", cx.inter_array_three_core_yaml_path),
    ("220 kV export", cx.export_three_core_yaml_path),
    ("CIGRE TB 880", cx.poradowski2025_220kv_yaml_path),
    ("Menard 2023", cx.menard2023_yaml_path),
)


def fixed_radius(cable: cx.CableData) -> tuple[float, float]:
    at = axial_torsional_matrix(cable)
    twist = -at.K_et / (at.K_ee * at.K_tt - at.K_et**2)
    return at.K_et / at.K_ee, math.degrees(twist) * TENSION_N


def table() -> None:
    print(
        "cable                 M/T solved  M/T fixed-r  twist solved  twist fixed-r"
        "  ratio  fill"
    )
    print("                      [mm]        [mm]         [deg/m/100kN] [deg/m/100kN]")
    for label, path in CABLES:
        cable = cx.load_cable(path(), geometry_override=None)
        c_fr, t_fr = fixed_radius(cable)
        fills = ", ".join(
            f"{armour_fill_ratio(x.n_wires, x.d_wire_m, x.r_helix_m, x.lay_angle_deg):.2f}"
            for x in cable.helical_layers
        )
        try:
            r = cx.torque_balance(cable, tensions_N=[TENSION_N])
        except CableSchemaError as exc:
            print(
                f"{label:20s}  {'-':>10s}  {c_fr * 1e3:11.3f}  {'-':>12s}  "
                f"{t_fr:13.4f}  {'-':>5s}  {fills}   solved section: {exc}"
            )
            continue
        print(
            f"{label:20s}  {r.torque_coefficient_m * 1e3:10.3f}  {c_fr * 1e3:11.3f}  "
            f"{r.twist_per_100kN_deg_per_m:12.4f}  {t_fr:13.4f}  "
            f"{r.balance_ratio[-1]:5.2f}  {fills}"
        )
        shift = ", ".join(
            f"{name} {value[-1] / 1e6:+.2f} MPa"
            for name, value in r.stress_redistribution_Pa.items()
        )
        print(
            f"{'':20s}  free minus fixed wire stress: {shift}; residual "
            f"torque {r.residual_torque_Nm:.3g} N m"
        )
        if r.open_interfaces_free:
            print(
                f"{'':20s}  no contact force with a free end: {', '.join(r.open_interfaces_free)}"
            )


def template_design() -> None:
    cable = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
    r = cx.torque_balance(cable, (0.0, 200.0e3), n_tensions=5)
    print()
    print(r.summary())
    print("\nOuter armour that zeroes the torque at 100 kN (template, core lay RH):")
    for vary, unit, scale in (
        ("outer_lay_angle", "deg", 1.0),
        ("outer_wire_count", "", 1.0),
        ("outer_wire_diameter", "mm", 1e3),
    ):
        d = cx.design_torque_balance(cable, vary, tension_N=TENSION_N)
        print(
            f"  {vary:20s} {d.initial_value * scale:8.3f} -> {d.value * scale:8.3f} "
            f"{unit:3s} fill {d.fill_ratio:.3f}, armour ratio "
            f"{d.result.balance_ratio[-1]:.3f} ({d.message})"
        )
    assert cable.core_lay is not None
    left = cable.model_copy(
        update={
            "core_lay": cable.core_lay.model_copy(update={"lay_direction": "left_hand"})
        }
    )
    aligned = cable.model_copy(update={"core_lay": None})
    for label, variant in (("core lay LH", left), ("aligned cores", aligned)):
        r = cx.torque_balance(variant, tensions_N=[TENSION_N])
        d = cx.design_torque_balance(variant, tension_N=TENSION_N)
        print(
            f"  {label:13s}: M/T {r.torque_coefficient_m * 1e3:+.3f} mm, twist "
            f"{r.twist_per_100kN_deg_per_m:+.4f} deg/m per 100 kN; balancing outer "
            f"lay angle {d.value:.2f} deg (fill {d.fill_ratio:.3f})"
        )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table()
        template_design()
