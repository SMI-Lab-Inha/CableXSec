# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Dai, Sævik and Ye (2020) umbilical in a bellmouth: data, targets, rows.

The DAI20 targets were registered before any model output (doc/benchmarks.md,
steel-tube umbilical section); the pins below catch a later relaxation of the
targets or a silent change of the result (VALIDATION_POLICY Rules 2 and 9).
"""

from __future__ import annotations

import check_dai2020 as dai
import numpy as np
import pytest

pytestmark = pytest.mark.full_scale


def test_digitised_figure_9_is_complete_and_on_the_bellmouth_plateau():
    fig = dai.measured_fig9()
    assert len(fig["S2"]) == 67
    assert len(fig["S3"]) == 45
    s2 = fig["S2"]
    plateau = s2[np.abs(s2[:, 0]) >= 10.0, 1]
    # measured centre-tube curvature on the bellmouth: 0.1 1/m within reading
    assert np.abs(np.abs(plateau) - dai.KAPPA_BELL).max() < 0.0025
    assert (fig["S2"][:, 2] <= fig["S2"][:, 3]).all()


def test_registered_targets_are_pinned():
    meas = dai.measured_targets()
    assert meas["s2_kappa_at_4deg"] == pytest.approx(0.03995, abs=5e-6)
    assert meas["s2_angle_90"] == pytest.approx(7.033, abs=5e-4)
    assert meas["s3_angle_50"] == pytest.approx(14.574, abs=5e-4)
    assert dai.ANGLE_BAND_DEG == 1.0
    assert (dai.IDENTIFY_ANGLE_DEG, dai.S2_LEVEL, dai.S3_LEVEL) == (4.0, 0.9, 0.5)
    rig = dai.RIG
    assert (rig.x_anchor_m, rig.free_length_m, rig.rigid_right_m) == (-2.1, 9.0, 2.2)
    assert (rig.gap_m, rig.bell_end_m, rig.kappa_b0, rig.kappa_b1) == (
        0.003,
        3.0,
        0.1,
        0.0,
    )
    assert not rig.inlet_bounded and rig.x_rotation_m == 0.0
    assert pytest.approx(2.648) == dai.X_S3


def test_bellmouth_surface_starts_at_the_gap_without_an_inlet_tube():
    beam = dai._elastic_beam(50e3)
    inlet = beam.x < -1e-9
    assert np.isinf(beam.upper[inlet]).all()
    entrance = np.isclose(beam.x, 0.0, atol=beam.h / 4)
    assert beam.upper[entrance] == pytest.approx(dai.GAP_M)
    s = beam.x[(beam.x > 0) & (beam.x <= 3.0)]
    upper = beam.upper[(beam.x > 0) & (beam.x <= 3.0)]
    assert upper == pytest.approx(dai.GAP_M + 0.05 * s**2)
