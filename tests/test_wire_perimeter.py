# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Independent maximisation around the circular wire perimeter."""

import math

import numpy as np
import pytest

from cablexsec.mechanics.wire_stress import local_bending_amplitude


@pytest.mark.parametrize("azimuth", [0.0, 0.3, 1.2, math.pi / 2, 2.4, math.pi])
def test_bending_amplitude_matches_resolved_perimeter(azimuth):
    alpha, radius, curvature, modulus = 0.35, 0.002, 0.07, 210e9
    theta = np.linspace(0, 2 * np.pi, 20001)
    a = math.cos(alpha) ** 2 * math.cos(2 * alpha) * math.cos(azimuth)
    b = -math.cos(alpha) * (1 + math.sin(alpha) ** 2) * math.sin(azimuth)
    direct = modulus * curvature * radius * (a * np.cos(theta) + b * np.sin(theta))
    expected = local_bending_amplitude(curvature, azimuth, modulus, 2 * radius, alpha)
    assert np.max(np.abs(direct)) == pytest.approx(float(expected), rel=2e-8)
    mean_stress = -60e6
    assert np.max(np.abs(direct + mean_stress)) == pytest.approx(
        abs(mean_stress) + float(expected), rel=2e-8
    )
