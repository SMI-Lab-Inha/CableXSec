# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Termination stress amplification must precede S-N curve evaluation."""

import math
from dataclasses import replace

import numpy as np
import pytest

from cablexsec.analysis.fatigue import FatigueLayer, compute_fatigue, miner_damage
from cablexsec.materials.sn_curves import SNcurve
from cablexsec.mechanics.mkappa import HelicalLayerSpec


def layer(curve, scf=2.0):
    return FatigueLayer(
        spec=HelicalLayerSpec(
            n_wires=1,
            r_helix_m=0.05,
            lay_angle_rad=0,
            E_wire_Pa=200e9,
            A_wire_m2=math.pi * 0.002**2,
            xi=1,
            mu=0,
            F_n_axi_N=0,
        ),
        sn_bulk=curve,
        sigma_UTS_Pa=1e9,
        name="wire",
        SCF=scf,
        termination=True,
    )


def run(curve, stress_range_MPa, scf=2.0):
    # One closed cycle in prescribed normal stress, with no bending/friction.
    stress = np.array([0.0, stress_range_MPa * 1e6, 0.0])
    return compute_fatigue(
        [layer(curve, scf)],
        np.zeros(3),
        np.zeros(3),
        normal_stress_Pa={"wire": stress},
        n_fibres=4,
    )


def test_termination_can_cross_fatigue_cutoff_from_zero_bulk_damage():
    curve = SNcurve("cutoff", "MPa", 3, 12, threshold=50)
    result = run(curve, 40)
    assert result.D_bulk_per_wire["wire"][0] == 0
    # One cycle at an amplified 80 MPa range: N = 10**12 / 80**3.
    expected = 80**3 / 1e12
    assert result.D_term_per_wire["wire"][0] == pytest.approx(expected)
    assert result.D_total_per_wire["wire"][0] == pytest.approx(expected)


@pytest.mark.parametrize("stress_range", [20.0, 40.0, 60.0])
def test_bilinear_branch_is_selected_at_amplified_range(stress_range):
    # Continuous branches intersect at S=100 MPa: C2/C1 = S**(5-3).
    curve = SNcurve("bilinear", "MPa", 3, 12, m2=5, log_C2=16)
    result = run(curve, stress_range)
    amplified = 2 * stress_range
    expected = amplified**5 / 1e16 if amplified < 100 else amplified**3 / 1e12
    assert result.D_total_per_wire["wire"][0] == pytest.approx(expected, rel=1e-12)
    assert result.D_bulk_per_wire["wire"][0] == pytest.approx(stress_range**5 / 1e16)


def test_single_slope_scaling_is_preserved():
    curve = SNcurve("power-law", "MPa", 4, 13)
    result = run(curve, 35, scf=1.5)
    assert result.D_total_per_wire["wire"][0] == pytest.approx(52.5**4 / 1e13)
    assert result.D_term_per_wire["wire"][0] == pytest.approx(
        (1.5**4 - 1) * 35**4 / 1e13
    )


def test_scf_one_adds_no_increment():
    curve = SNcurve("cutoff", "MPa", 3, 12, threshold=50)
    result = run(curve, 80, scf=1)
    assert result.D_term_per_wire["wire"][0] == 0
    assert result.D_total_per_wire["wire"][0] == result.D_bulk_per_wire["wire"][0]


def test_unflagged_layer_does_not_apply_termination_factor():
    curve = SNcurve("power-law", "MPa", 3, 12)
    result = compute_fatigue(
        [replace(layer(curve), termination=False)],
        np.zeros(3),
        np.zeros(3),
        normal_stress_Pa={"wire": np.array([0, 40e6, 0])},
        n_fibres=4,
    )
    assert result.D_total_per_wire["wire"][0] == pytest.approx(40**3 / 1e12)


def test_amplification_can_change_the_governing_material_fibre():
    curve = SNcurve("bilinear", "MPa", 3, 12, m2=5, log_C2=16)
    # Twenty small cycles at radial fibres, one large cycle at tangential
    # fibres. The latter governs before amplification; the former afterwards.
    radial = np.array([0.0, 30e6] * 20 + [0.0, 0.0, 0.0])
    tangential = np.array([0.0] * 41 + [70e6, 0.0])
    result = compute_fatigue(
        [layer(curve, scf=4)],
        radial / (200e9 * 0.002),
        tangential / (200e9 * 0.002),
        n_fibres=4,
    )
    assert result.D_bulk_per_wire["wire"][0] == pytest.approx(70**5 / 1e16)
    assert result.D_total_per_wire["wire"][0] == pytest.approx(20 * 120**3 / 1e12)
    assert result.fibre_critical_deg_per_wire["wire"][0] in (90, 270)
    assert result.termination_fibre_critical_deg_per_wire["wire"][0] in (0, 180)
    np.testing.assert_allclose(
        result.D_bulk_with_termination_per_fibre["wire"],
        [[20 * 120**3 / 1e12, 280**3 / 1e12] * 2],
        rtol=1e-12,
    )


def test_explicit_reduction_factor_handles_cutoff_and_signed_adjustment():
    curve = SNcurve("cutoff", "MPa", 3, 12, threshold=50)
    result = run(curve, 80, scf=0.5)
    assert result.D_bulk_per_wire["wire"][0] > 0
    assert result.D_total_per_wire["wire"][0] == 0
    assert result.D_term_per_wire["wire"][0] == -result.D_bulk_per_wire["wire"][0]


@pytest.mark.parametrize("value", [0, -1, math.inf, math.nan])
def test_invalid_factors_fail_before_counting(value):
    curve = SNcurve("power-law", "MPa", 3, 12)
    with pytest.raises(ValueError, match="SCF"):
        layer(curve, scf=value)
    with pytest.raises(ValueError, match="stress_range_factor"):
        miner_damage(np.array([0, 1]), curve, 1e9, stress_range_factor=value)


def test_amplified_stress_overflow_is_explicit():
    curve = SNcurve("power-law", "MPa", 3, 12)
    with pytest.raises(ValueError, match="amplified stress range"):
        miner_damage(np.array([0.0, 100e6]), curve, 1e9, stress_range_factor=1e308)
