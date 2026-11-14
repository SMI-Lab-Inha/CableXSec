# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Depth-resolved normal contact between two obliquely crossed steel wires.

Run from the checkout: python -m examples.hertz_axis_stress
This synthetic elastic example is not a cable fatigue-life assessment.
"""

import math

import numpy as np

from cablexsec import compute_hertz_axis_stress
from cablexsec.mechanics.hertz_contact import (
    crossed_cylinder_curvatures,
    reduced_modulus,
    solve_contact,
)


def main() -> None:
    a_curv, b_curv = crossed_cylinder_curvatures(0.003, math.radians(60))
    footprint = solve_contact(
        10.0, a_curv, b_curv, reduced_modulus(210e9, 0.3, 210e9, 0.3)
    )
    stress = compute_hertz_axis_stress(
        footprint, np.linspace(0, 3 * footprint.a_m, 1001), poisson_ratio=0.3
    )
    print(
        f"Contact semi-axes: {footprint.a_m * 1e6:.3f}, "
        f"{footprint.b_m * 1e6:.3f} micrometres"
    )
    print(f"Peak pressure: {footprint.p0_Pa / 1e6:.3f} MPa")
    print(
        f"Sampled peak von Mises: {stress.sampled_peak_von_mises_Pa / 1e6:.3f} MPa "
        f"at depth {stress.sampled_peak_depth_m * 1e6:.3f} micrometres"
    )
    print(
        f"Sampled peak maximum shear: {np.max(stress.maximum_shear_Pa) / 1e6:.3f} MPa"
    )
    print(
        f"Legacy circular-contact shear estimate: {footprint.tau_max_Pa / 1e6:.3f} MPa"
    )
    print(
        "Normal contact only; supplied depth samples do not establish a global "
        "3-D peak or fatigue life."
    )


if __name__ == "__main__":
    main()
