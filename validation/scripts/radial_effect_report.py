# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Effect of the solved radial contact state on armour friction and damage.

Compares the default bounded pressure chain with the solved radial contact
model ("solved", filler as declared and filler removed) for the public
three-core templates and the Ménard and Cartraud (2023) cable. Reports the
friction-setting pressure P1+P2, the full-slip Papailiou friction-stress cap
and the bulk armour damage of a fixed synthetic bending history. The history
is a comparison device only (not a design load case); no article number
is produced or changed by this script.

Usage: python validation/scripts/radial_effect_report.py
"""

from __future__ import annotations

import math

import numpy as np

import cablexsec as cx
from cablexsec.cable_adapters import armour_contact_pressure, cable_to_mkappa_specs
from cablexsec.exceptions import CableSchemaError
from cablexsec.mechanics.papailiou import wire_slip_stress_cap
from cablexsec.mechanics.radial_cable import solve_cable_radial_state

CABLES = {
    "three-core 33 kV template": cx.inter_array_three_core_yaml_path,
    "three-core 220 kV template": cx.export_three_core_yaml_path,
    "Menard and Cartraud 2023": cx.menard2023_yaml_path,
    "single-core 33 kV template": cx.inter_array_single_core_yaml_path,
}
TENSIONS_N = (0.0, 100e3)


def _friction_cap(cable, tension, model, state=None):
    specs = cable_to_mkappa_specs(
        cable, tension, contact_model=model, radial_state=state
    )
    names = [x.name for x in cx.cable_adapters.structural_armour_layers(cable)]
    return {
        name: float(np.max(wire_slip_stress_cap(spec, 0.0)))
        for name, spec in zip(names, specs, strict=True)
    }


def _damage(cable, tension, model):
    t = np.linspace(0.0, 20 * math.pi, 2001)
    kx = 0.05 * np.sin(t)
    ky = np.zeros_like(kx)
    res = cx.compute_fatigue(
        cable,
        kx,
        ky,
        tension_N=np.full_like(kx, tension),
        contact_tension_N=tension,
        contact_model=model,
    )
    return {name: float(np.max(d)) for name, d in res.D_bulk_per_wire.items()}


def main() -> None:
    rows = []
    for label, path in CABLES.items():
        cable = cx.load_cable(path(), geometry_override=None)
        for tension in TENSIONS_N:
            entry = {"cable": label, "T_kN": tension / 1e3}
            try:
                entry["bounded"] = armour_contact_pressure(cable, tension)
                entry["cap_bounded"] = _friction_cap(cable, tension, "bounded")
                entry["dmg_bounded"] = _damage(cable, tension, "bounded")
            except CableSchemaError as exc:
                entry["bounded"] = str(exc)
            for filler in ("auto", "void"):
                try:
                    state = solve_cable_radial_state(cable, tension, filler=filler)
                    entry[f"solved_{filler}"] = armour_contact_pressure(
                        cable, tension, contact_model="solved", radial_state=state
                    )
                    entry[f"cap_{filler}"] = _friction_cap(
                        cable, tension, "solved", state
                    )
                    entry[f"notes_{filler}"] = state.notes
                except CableSchemaError as exc:
                    entry[f"solved_{filler}"] = f"unsupported: {exc}"
            try:
                entry["dmg_solved"] = _damage(cable, tension, "solved")
            except CableSchemaError as exc:
                entry["dmg_solved"] = f"unsupported: {exc}"
            rows.append(entry)
    for entry in rows:
        print(f"\n## {entry['cable']}, T = {entry['T_kN']:.0f} kN")
        for key, value in entry.items():
            if key in {"cable", "T_kN"}:
                continue
            if isinstance(value, dict):
                text = ", ".join(f"{k}: {v:.4g}" for k, v in value.items())
            else:
                text = str(value)
            print(f"- {key}: {text}")


if __name__ == "__main__":
    main()


def thermal_and_water_effects() -> None:
    """Load cases that only the solved model can represent."""
    cable = cx.load_cable(cx.inter_array_three_core_yaml_path(), geometry_override=None)
    core_layers = [x.name for x in cable.layers if getattr(x, "n_cores", 1) == 3]
    cases = {
        "reference 20 C, dry, T=50 kN": {},
        "all layers 50 C": {"temperature_C": 50.0},
        "cores 70 C, rest 20 C": {"temperature_C": dict.fromkeys(core_layers, 70.0)},
        "dry external pressure 1 MPa": {"external_pressure_Pa": 1e6},
        "flooded (wet design) 1 MPa": {"flooded_pressure_Pa": 1e6},
    }
    print("\n## three-core 33 kV template: temperature and water pressure, T = 50 kN")
    for label, kwargs in cases.items():
        state = solve_cable_radial_state(cable, 50e3, **kwargs)
        p = state.armour_contact_pressure()
        b = state.result.bundle
        assert b is not None
        print(
            f"- {label}: P1+P2 inner {p['inner_armour'] / 1e6:.3f} MPa, "
            f"outer {p['outer_armour'] / 1e6:.3f} MPa; core-ring "
            f"{b.core_ring_force_N_per_m / 1e3:.2f} kN/m, filler-ring "
            f"{b.filler_ring_force_N_per_m / 1e3:.1f} kN/m"
        )


if __name__ == "__main__":
    thermal_and_water_effects()
