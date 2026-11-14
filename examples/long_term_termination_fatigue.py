# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Illustrative lifetime projections with explicit termination range factors.

The SCFs, loads and probabilities below are synthetic, not qualified design data.
"""

from __future__ import annotations

import hashlib

import numpy as np

import cablexsec as cx


def main() -> None:
    original = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    cable = original.model_copy(
        update={
            "layers": tuple(
                layer.model_copy(update={"termination_scf": 1.5})
                if layer.name in {"inner_armour", "outer_armour"}
                else layer
                for layer in original.layers
            )
        }
    )
    identity = hashlib.sha256(
        (
            cable.model_dump_json() + cx.__version__ + ":default-fatigue-settings"
        ).encode()
    ).hexdigest()
    time = np.linspace(0, 600, 601)
    phase = 2 * np.pi * time / 10
    cases = []
    for name, amplitude, probability in (("mild", 0.005, 0.8), ("severe", 0.02, 0.2)):
        short = cx.compute_fatigue(
            cable,
            amplitude * np.sin(phase),
            0.5 * amplitude * np.cos(phase),
            tension_N=np.full_like(time, 80e3),
        )
        cases.append(cx.FatigueLoadCase(name, short, 600, probability, identity))
    for basis in ("bulk", "bulk_with_termination"):
        result = cx.compute_long_term_fatigue(
            cases,
            exposure_years=25,
            design_fatigue_factor=3,
            damage_basis=basis,
        )
        print(
            f"{result.damage_basis}: design utilization {result.design_utilization:.6g}"
        )
        print(
            f"  {result.critical_layer}, wire {result.critical_wire_index}, "
            f"fibre {result.critical_fibre_deg:g} degrees"
        )
    print(
        "Synthetic demonstration; excludes fretting and is not a validated service life."
    )


if __name__ == "__main__":
    main()
