# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Illustrative stationary bulk fatigue; probabilities are synthetic inputs."""

from __future__ import annotations

import hashlib

import numpy as np

import cablexsec as cx


def main() -> None:
    cable = cx.load_cable(cx.inter_array_single_core_yaml_path())
    # Use the same indexed geometry, material frame, curves and default fatigue
    # settings throughout. Include custom curve/settings inputs in this digest
    # when overriding defaults; a cable digest alone would then be insufficient.
    assessment_id = hashlib.sha256(
        (
            cable.model_dump_json() + cx.__version__ + ":default-fatigue-settings"
        ).encode()
    ).hexdigest()
    time_s = np.linspace(0, 600, 601)
    cases = []
    for name, amplitude, probability in (("mild", 0.002, 0.8), ("severe", 0.008, 0.2)):
        phase = 2 * np.pi * time_s / 10
        result = cx.compute_fatigue(
            cable,
            amplitude * np.sin(phase),
            0.5 * amplitude * np.cos(phase),
            tension_N=np.full_like(time_s, 80e3),
        )
        cases.append(cx.FatigueLoadCase(name, result, 600, probability, assessment_id))
    long_term = cx.compute_long_term_fatigue(
        cases, exposure_years=25, design_fatigue_factor=3
    )
    print(f"Bulk design utilization: {long_term.design_utilization:.6g}")
    print(
        f"Location: {long_term.critical_layer}, wire {long_term.critical_wire_index}, "
        f"fibre {long_term.critical_fibre_deg:g} degrees"
    )
    print(long_term.provenance)


if __name__ == "__main__":
    main()
