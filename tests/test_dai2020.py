# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Dai, Sævik and Ye (2020) umbilical in a bellmouth: data, targets, rows.

The DAI20 targets were registered before any model output (doc/benchmarks.md,
steel-tube umbilical section); the pins below catch a later relaxation of the
targets or a silent change of the result (VALIDATION_POLICY Rules 2 and 9).
"""

from __future__ import annotations

import math

import check_dai2020 as dai
import numpy as np
import pytest
import run_validation

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


def test_coulomb_range_is_the_source_2d_formula():
    # The sectional full-slip range equals the source's Eq. 14 with the same
    # line loads, apart from the wire-axis projection cos(alpha) and the
    # third-order capstan term of the signed bound.
    cou = dai.coulomb_check()
    alpha = math.radians(dai.LAY_DEG)
    for model, forces in dai.LINE_FORCE_N_M.items():
        for tube, (_n, radius, od, t) in dai.TUBES.items():
            area = math.pi * (od - t) * t
            eq14 = (
                math.pi
                * radius
                * dai.MU
                * forces[tube]
                / (dai.E_STEEL_PA * area * math.sin(alpha))
            )
            got = cou[model][tube]["microstrain"] * 1e-6
            assert got == pytest.approx(eq14, rel=2e-3)


def test_beam_curvature_is_mesh_converged():
    # with the tight solve tolerance the identified stiffness and the two
    # predicted angles move by less than 0.05 deg from 20 to 2.5 mm
    reg = dai.kappa_check()
    for v in dai.mesh_study().values():
        assert abs(v["error_s2_deg"] - reg["error_s2_deg"]) < 0.05
        assert abs(v["error_s3_deg"] - reg["error_s3_deg"]) < 0.05
        assert v["EI_Nm2"] == pytest.approx(reg["EI_Nm2"], rel=0.05)


def test_registered_rows_are_pinned():
    rows = {r.ident: r for r in run_validation._dai2020_rows()}
    kap = rows["DAI20-KAPPA"]
    assert kap.target == "both within +-1.0 deg of the measured 7.03 and 14.57 deg"
    assert kap.value == (
        "EI 59.1 kN m2; S2 6.16 deg (-0.87), S3 13.69 deg (-0.89); finer beam "
        "meshes S2/S3 error 10 mm -0.89/-0.89; 5 mm -0.91/-0.90; 2.5 mm "
        "-0.91/-0.90 deg"
    )
    assert kap.status == "PASS"
    cou = rows["DAI20-COULOMB"]
    assert cou.target == "stated, not gated (measured 200 / 158 microstrain)"
    assert cou.value == (
        "radial 378 / 71 (+89 / -55 %); radial plus hoop 428 / 416 "
        "(+114 / +163 %); the source's 2D values reproduced within 1.1 %"
    )
    assert cou.status == "INFO"
