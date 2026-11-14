# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""How-to 10: read CableDyn and RIFLEX ASCII results into load series.

Uses the files under ``tests/data`` of a source checkout:

    python doc/howto/scripts/howto_10_read_global_results.py
"""
# [recipe]

import math
from pathlib import Path

import numpy as np

import cablexsec as cx

DATA = Path(__file__).resolve().parents[3] / "tests" / "data"

# CableDyn: a lazy-wave run, two material sections by arc length from End A.
read = cx.read_cabledyn_sections(
    DATA / "cabledyn" / "lw128",  # output root
    DATA / "cabledyn" / "lw128.dat",  # deck of the run
    {"hangoff": 0.25, "sag": 58.4},
    frame="section",
)
print(f"curvature from {read.checks['curvature_source']} channels")
outer_area = math.pi * 0.160**2 / 4.0  # outer diameter of the deck line type
for name, history in read.histories.items():
    series = history.to_load_case_series(outer_area)
    kx, ky = series.curvature_x_1_m, series.curvature_y_1_m
    print(
        f"{name:<8} {series.time_s.size} samples, dt {series.time_s[1]:.1f} s, "
        f"Te mean {history.effective_tension_N.mean() / 1e3:.2f} kN, "
        f"true mean {series.tension_N.mean() / 1e3:.2f} kN, "
        f"|kappa| max {np.hypot(kx, ky).max():.4f} 1/m"
    )

# RIFLEX ASCII: positional columns; declare the tension and curvature basis.
series = cx.read_riflex_ascii(
    DATA / "histories" / "synthetic_riflex.asc",
    time_col=1,
    curvature_x_col=2,
    tension_col=3,
    curvature_y_col=4,
    tension_kind="effective",
    curvature_convention="strain_gradient",
)
print(
    f"riflex   {series.time_s.size} samples, {series.time_s[-1]:.1f} s, "
    f"{series.tension_kind} tension {series.tension_N.mean() / 1e3:.2f} kN mean, "
    f"kappa_x {series.curvature_x_1_m.min():.4f} to "
    f"{series.curvature_x_1_m.max():.4f} 1/m"
)
