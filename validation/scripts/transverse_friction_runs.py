# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Same-cable comparison of contact-level and section-level bending work.

For the published three-core cable (Menard and Cartraud 2023), each armour
layer is solved with the two-component friction cell of
:mod:`transverse_friction_cell` under a symmetric curvature cycle:

* ``none``: longitudinal friction only (the existing spatial reference);
* ``equilibrium``: the transverse line load needed to keep the wire on its
  loxodromic path is carried by the same Coulomb friction; transverse slip is
  an output, not an input;
* ``prescribed``: the assumed transverse map of Eq. (2.15) with eta values.

The summed contact-level frictional work per cycle is compared with the
section-law work for the same inputs. No parameter is fitted.

Run from the repository root:
``python -m transverse_friction_runs --output <new-directory>``
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from transverse_friction_cell import solve_transverse_cell

import cablexsec as cx
from cablexsec.cable_adapters import cable_to_mkappa_specs
from cablexsec.mechanics.papailiou import compute_mkappa_wire_slip_front
from cablexsec.mechanics.stiffness import ei_min_slip


def layer_inputs(layer):
    alpha, radius, area = layer.lay_angle_rad, layer.r_helix_m, layer.A_wire_m2
    q = abs(math.sin(alpha)) / radius
    ea = layer.E_wire_Pa * area
    diameter = math.sqrt(4 * area / math.pi)
    ei = layer.E_wire_Pa * math.pi * diameter**4 / 64
    stiffness = ea * q * q * layer.xi / (1 - layer.xi)
    line_friction = (
        layer.mu
        * layer.contact_pressure_Pa
        * (2 * math.pi * radius / layer.n_wires)
        * math.cos(alpha)
    )
    return {
        "axial_rigidity_N": ea,
        "bending_rigidity_Nm2": ei,
        "helix_radius_m": radius,
        "lay_angle_rad": alpha,
        "wire_count": layer.n_wires,
        "support_N_m2": stiffness,
        "friction_limit_N_m": line_friction,
    }


def cycle(result, start):
    return {
        "friction_J_m": float(result.friction_J_m[-1] - result.friction_J_m[start]),
        "friction_longitudinal_J_m": float(
            result.friction_longitudinal_J_m[-1]
            - result.friction_longitudinal_J_m[start]
        ),
        "friction_transverse_J_m": float(
            result.friction_transverse_J_m[-1] - result.friction_transverse_J_m[start]
        ),
    }


def run_case(amplitude, mode, eta, elements, increments, tension=0.0):
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    layers = cable_to_mkappa_specs(cable, 0.0, mu=0.2)
    history = amplitude * np.sin(np.linspace(0, 6 * np.pi, 3 * increments + 1))
    start = 2 * increments
    moment = ei_min_slip(cable) * history.copy()
    totals = {
        "friction_J_m": 0.0,
        "friction_longitudinal_J_m": 0.0,
        "friction_transverse_J_m": 0.0,
    }
    demand, slip_ratio, balance = 0.0, 0.0, 0.0
    for layer in layers:
        inputs = layer_inputs(layer)
        result = solve_transverse_cell(
            **inputs,
            curvature_1_m=history,
            mode=mode,
            eta=eta,
            tension_per_wire_N=tension,
            elements=elements,
        )
        for key, value in cycle(result, start).items():
            totals[key] += value
        moment += result.moment_Nm
        demand = max(demand, float(np.max(result.demand_ratio_max[start:])))
        pl = result.longitudinal_plastic_m[start:]
        pt = result.transverse_plastic_m[start:]
        amp_l = 0.5 * float(np.max(np.ptp(pl, axis=0)))
        amp_t = 0.5 * float(np.max(np.ptp(pt, axis=0)))
        if amp_l > 0:
            slip_ratio = max(slip_ratio, amp_t / amp_l)
        if mode != "equilibrium":
            stored_change = result.stored_J_m[-1] - result.stored_J_m[start]
            work = result.endpoint_work_J_m[-1] - result.endpoint_work_J_m[start]
            rem = (
                result.incremental_remainder_J_m[-1]
                - result.incremental_remainder_J_m[start]
            )
            fr = result.friction_J_m[-1] - result.friction_J_m[start]
            balance = max(balance, abs(work - stored_change - fr - rem))
    section = compute_mkappa_wire_slip_front(
        layers, history, EI_min_Nm2=ei_min_slip(cable)
    )
    section_work = float(
        section.work.friction_J_per_m[-1] - section.work.friction_J_per_m[start]
    )
    return {
        "amplitude_1_m": amplitude,
        "mode": mode,
        "eta": eta,
        "tension_per_wire_N": tension,
        "elements": elements,
        "increments_per_cycle": increments,
        **totals,
        "peak_moment_Nm": float(np.max(moment[start:])),
        "section_law_friction_J_m": section_work,
        "section_law_peak_moment_Nm": float(np.max(section.moment_Nm[start:])),
        "max_transverse_demand_ratio": demand,
        "max_transverse_to_longitudinal_slip_ratio": slip_ratio,
        "work_balance_residual_J_m": balance,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--elements", type=int, default=128)
    parser.add_argument("--increments", type=int, default=360)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"preserve existing results: {args.output}")
    args.output.mkdir(parents=True)
    rows = []
    for amplitude in (0.05, 0.10, 0.20):
        cases = [("none", 0.0), ("equilibrium", 0.0)] + [
            ("prescribed", e) for e in (0.25, 0.5, 1.0, 2.0)
        ]
        for mode, eta in cases:
            row = run_case(amplitude, mode, eta, args.elements, args.increments)
            rows.append(row)
            print(json.dumps(row), flush=True)
    (args.output / "transverse_same_cable.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
