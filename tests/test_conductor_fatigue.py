# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Stranded-conductor fatigue: strand slip, bulk Miner and trellis fretting."""

from __future__ import annotations

import math
import warnings

import check_conductor_strand_slip as check
import numpy as np
import pytest

import cablexsec as cx
from cablexsec._numba import kernels_compiled
from cablexsec.analysis.conductor_fatigue import (
    FULLY_PLASTIC_PRESSURE_FACTOR,
    _stop_operator,
    assess_conductor_fatigue,
    trellis_contact,
    trellis_force_ratio,
)
from cablexsec.cables.schema import CoreStrandRing
from cablexsec.exceptions import CableInputError
from cablexsec.materials.sn_curves import get_sn_curve, get_swt_life
from cablexsec.mechanics.hertz_contact import layer_crossing_density_per_m
from cablexsec.mechanics.strand_slip import (
    conductor_axial_stiffness,
    interface_line_loads,
)
from cablexsec.provenance import (
    UnevaluatedFatigueCurveWarning,
    UnvalidatedModelWarning,
)

SN = get_sn_curve("nasution_2014_copper")
SWT = get_swt_life("copper_poon2022")


def _run(kappa: np.ndarray, **kwargs):
    return assess_conductor_fatigue(
        check.nasution_rings(),
        check.NAS_E,
        0.3,
        kappa,
        np.zeros_like(kappa),
        np.full_like(kappa, check.NAS_T),
        sn_bulk=SN,
        mu_wire=0.2,
        reference_tension_N=check.NAS_T,
        **{"swt_const": SWT, **kwargs},
    )


@pytest.mark.filterwarnings("ignore::cablexsec.provenance.UnvalidatedModelWarning")
def test_bulk_damage_and_fretting_on_alternately_laid_rings():
    kappa = np.sin(np.linspace(0, 6 * math.pi, 1801)) / 3.0
    result = _run(kappa)
    assert result.D_bulk_per_ring.shape == (4,)
    assert result.D_bulk_per_orientation.shape == (4, 8)
    assert np.all(result.D_bulk_per_ring > 0)
    # The centre wire has no trellis contact; every helical ring has one.
    assert result.D_fretting_per_ring[0] == 0
    assert np.all(result.D_fretting_per_ring[1:] > 0)
    assert np.all(result.contact_p0_Pa[1:] > 0)
    assert result.critical_ring in (1, 2)
    bulk_only = _run(kappa, swt_const=None)
    np.testing.assert_array_equal(bulk_only.D_fretting_per_ring, 0.0)
    np.testing.assert_array_equal(bulk_only.D_bulk_per_ring, result.D_bulk_per_ring)
    assert "KarlsenHusoy2026" in " ".join(result.provenance.citations)


@pytest.mark.filterwarnings("ignore::cablexsec.provenance.UnvalidatedModelWarning")
def test_stick_history_has_no_slip_cycles_unless_partial_slip_is_requested():
    # Far below the 2nd-layer slip curvature 2 eps_f / (beta_0 - beta_1).
    kappa = 1e-5 * np.sin(np.linspace(0, 20 * math.pi, 2001))
    skipped = _run(kappa)
    np.testing.assert_array_equal(skipped.D_fretting_per_ring, 0.0)
    assert np.all(skipped.slip_cycle_fraction[1:] < 0.05)
    partial = _run(kappa, partial_slip_cycles="cattaneo_mindlin")
    assert np.all(partial.D_fretting_per_ring[1:] > 0)


@pytest.mark.filterwarnings("ignore::cablexsec.provenance.UnvalidatedModelWarning")
def test_same_lay_neighbours_have_no_trellis_contact():
    rings = tuple(
        cx_ring
        if index == 0
        else type(cx_ring)(
            cx_ring.n_wires,
            cx_ring.d_wire_m,
            cx_ring.r_helix_m,
            cx_ring.lay_angle_rad,
            1.0,
        )
        for index, cx_ring in enumerate(check.nasution_rings())
    )
    kappa = np.sin(np.linspace(0, 6 * math.pi, 601)) / 3.0
    result = assess_conductor_fatigue(
        rings,
        check.NAS_E,
        0.3,
        kappa,
        np.zeros_like(kappa),
        np.full_like(kappa, check.NAS_T),
        sn_bulk=SN,
        swt_const=SWT,
        mu_wire=0.2,
    )
    np.testing.assert_array_equal(result.D_fretting_per_ring, 0.0)


def test_public_cable_without_resolved_conductor_is_rejected():
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    kappa = np.zeros(4)
    with pytest.raises(cx.CableSchemaError, match="strand_layers"):
        cx.compute_conductor_fatigue(cable, kappa, kappa)


# --------------------------------------------------------------------------- #
# Gap 1.3c: trellis contact model and default                                  #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def stranded_cable():
    """Public 66 kV template with a synthetic alternately laid conductor."""
    cable = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
    conductor = cable.layers[0]
    d = conductor.r_outer_m / 4.5
    rings = [CoreStrandRing(n_wires=1, d_wire_m=d, r_helix_m=0.0)]
    for i, n in enumerate((6, 12, 18, 24), start=1):
        rings.append(
            CoreStrandRing(
                n_wires=n,
                d_wire_m=d,
                r_helix_m=i * d,
                pitch_m=24 * i * d,
                lay_direction="right_hand" if i % 2 else "left_hand",
            )
        )
    layer = conductor.model_copy(
        update={
            "strand_layers": tuple(rings),
            "conductor_area_mm2": None,
            "construction": "resolved_concentric_stranded",
            "bulk_sn": "nasution_2014_copper",
            "fretting_sn": "copper_poon2022",
        }
    )
    return cable.model_copy(update={"layers": (layer, *cable.layers[1:])})


def test_default_conductor_assessment_is_bulk_only(stranded_cable):
    t = np.linspace(0, 6 * math.pi, 601)
    kx, ky = 0.2 * np.sin(t), 0.05 * np.cos(t)
    tension = np.full_like(t, 2e4)
    with pytest.warns(UnevaluatedFatigueCurveWarning, match="trellis_fretting=True"):
        bulk = cx.compute_conductor_fatigue(stranded_cable, kx, ky, tension_N=tension)
    np.testing.assert_array_equal(bulk.D_fretting_per_ring, 0.0)
    np.testing.assert_array_equal(bulk.D_total_per_ring, bulk.D_bulk_per_ring)
    assert bulk.governing == "bulk"
    assert any("trellis_fretting=True" in note for note in bulk.provenance.limitations)
    opted = cx.compute_conductor_fatigue(
        stranded_cable, kx, ky, tension_N=tension, trellis_fretting=True
    )
    assert opted.fretting_evaluated
    assert "critical distances" in opted.provenance.method
    np.testing.assert_array_equal(opted.D_bulk_per_ring, bulk.D_bulk_per_ring)
    np.testing.assert_array_equal(
        opted.D_total_per_ring,
        np.maximum(opted.D_bulk_per_ring, opted.D_fretting_per_ring),
    )
    with pytest.warns(UnvalidatedModelWarning, match="not validated"):
        legacy = cx.compute_conductor_fatigue(
            stranded_cable,
            kx,
            ky,
            tension_N=tension,
            trellis_fretting=True,
            trellis_model="surface_swt",
        )
    assert np.any(legacy.D_fretting_per_ring > 0)
    assert any("NOT VALIDATED" in n for n in legacy.provenance.limitations)
    no_law = stranded_cable.model_copy(
        update={
            "layers": (
                stranded_cable.layers[0].model_copy(update={"fretting_sn": ""}),
                *stranded_cable.layers[1:],
            )
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cx.compute_conductor_fatigue(no_law, kx, ky, tension_N=tension)
    cx.compute_conductor_fatigue(
        no_law, kx, ky, tension_N=tension, trellis_fretting=True
    )
    with pytest.raises(cx.CableSchemaError, match="fretting_sn"):
        cx.compute_conductor_fatigue(
            no_law,
            kx,
            ky,
            tension_N=tension,
            trellis_fretting=True,
            trellis_model="surface_swt",
        )


def test_requesting_trellis_fretting_warns_at_the_analysis_level():
    kappa = np.sin(np.linspace(0, 2 * math.pi, 201)) / 3.0
    with pytest.warns(UnvalidatedModelWarning, match="gap 1.3c"):
        _run(kappa)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _run(kappa, swt_const=None)


def test_crossing_force_balances_the_interface_line_load():
    rings = check.nasution_rings()
    strain = check.NAS_T / conductor_axial_stiffness(rings, check.NAS_E)
    outer, inner = interface_line_loads(rings, check.NAS_E, strain)
    for i in (1, 2):
        out_face = trellis_contact(rings[i], rings[i + 1], outer[i], check.NAS_E, 0.3)
        in_face = trellis_contact(
            rings[i + 1], rings[i], inner[i + 1], check.NAS_E, 0.3
        )
        assert out_face is not None and in_face is not None
        # One crossing seen from either ring carries the same force.
        assert out_face.force_N == pytest.approx(in_face.force_N, rel=1e-12)
        density = layer_crossing_density_per_m(
            rings[i + 1].n_wires,
            rings[i].r_helix_m,
            rings[i].lay_angle_rad,
            rings[i].lay_sign,
            rings[i + 1].r_helix_m,
            rings[i + 1].lay_angle_rad,
            rings[i + 1].lay_sign,
        )
        # Force x crossings per unit length = line load (radial equilibrium).
        total = out_face.force_N * rings[i].n_wires * density
        assert total == pytest.approx(outer[i], rel=1e-12)
    # Values quoted in the documentation (300 mm^2, 30 kN).
    first = trellis_contact(rings[1], rings[2], outer[1], check.NAS_E, 0.3)
    second = trellis_contact(rings[2], rings[3], outer[2], check.NAS_E, 0.3)
    assert first is not None and second is not None
    assert first.force_N == pytest.approx(177.6, abs=0.1)
    assert second.force_N == pytest.approx(41.9, abs=0.1)


def _contact(yield_stress_Pa):
    rings = check.nasution_rings()
    return trellis_contact(
        rings[1], rings[2], 9.3e4, check.NAS_E, 0.3, yield_stress_Pa=yield_stress_Pa
    )


def test_contact_pressure_saturates_at_the_plastic_limit():
    elastic = _contact(None)
    assert elastic is not None and not elastic.saturated
    assert elastic.p0_Pa == elastic.p0_elastic_Pa
    # Hertz: mean pressure is two thirds of the peak.
    assert elastic.mean_pressure_Pa == pytest.approx(2 * elastic.p0_Pa / 3, rel=1e-9)
    capped = _contact(300e6)
    assert capped is not None and capped.saturated
    assert capped.p0_Pa == pytest.approx(FULLY_PLASTIC_PRESSURE_FACTOR * 300e6)
    assert capped.p0_elastic_Pa == elastic.p0_Pa
    # Mean pressure above the limit: the area grows to P / (3 sigma_y).
    assert elastic.mean_pressure_Pa > 900e6
    assert capped.area_m2 == pytest.approx(capped.force_N / 900e6, rel=1e-12)
    assert capped.mean_pressure_Pa == pytest.approx(900e6, rel=1e-12)
    # Peak between the limit and 1.5 times it: peak limited, Hertz area kept.
    mid = _contact(elastic.p0_Pa / 3 / 1.2)
    assert mid is not None and mid.saturated
    assert mid.area_m2 == pytest.approx(elastic.area_m2, rel=1e-12)
    loose = _contact(1e12)
    assert loose is not None and not loose.saturated
    assert loose.p0_Pa == elastic.p0_Pa
    with pytest.raises(CableInputError):
        _contact(-1.0)


def test_trellis_force_follows_the_shear_flow():
    theta = np.radians([0.0, 30.0, 90.0, 180.0, 270.0])
    kappa_c = 0.1
    t = np.linspace(0, 4 * math.pi, 801)
    zero = np.zeros_like(t)
    small = 0.9 * kappa_c * np.sin(t)  # reversal 1.8 kappa_c < 2 kappa_c
    ratio, slip = trellis_force_ratio(small, zero, theta, kappa_c)
    np.testing.assert_allclose(ratio[0], 0.0, atol=1e-15)  # extreme fibre
    np.testing.assert_allclose(ratio[3], 0.0, atol=1e-15)
    assert not slip.any()
    np.testing.assert_allclose(np.abs(ratio[2]).max(), 0.9, rtol=1e-3)
    # Shear flow ~ sin(theta): half the neutral-axis demand at 30 degrees.
    np.testing.assert_allclose(ratio[1], 0.5 * ratio[2], atol=1e-12)
    large = 1.5 * kappa_c * np.sin(t)  # reversal 3 kappa_c > 2 kappa_c
    ratio, slip = trellis_force_ratio(large, zero, theta, kappa_c)
    assert slip[2].any() and slip[4].any()
    assert not slip[0].any() and not slip[1].any()  # 1.5 sin 30 deg < 1
    np.testing.assert_allclose(ratio[2].max(), 1.0)
    # The projection rotates with the material frame.
    rotated, _ = trellis_force_ratio(zero, large, theta + math.pi / 2, kappa_c)
    np.testing.assert_allclose(rotated, ratio, atol=1e-12)
    with pytest.raises(CableInputError):
        trellis_force_ratio(small, small, theta, 0.0)


@pytest.mark.skipif(not kernels_compiled(), reason="kernels not compiled")
def test_compiled_stop_operator_matches_the_python_reference():
    rng = np.random.default_rng(7)
    drive = np.cumsum(rng.normal(0.0, 0.4, (3, 500)), axis=1)
    compiled = _stop_operator(drive)
    reference = _stop_operator.py_func(drive)  # type: ignore[attr-defined]
    np.testing.assert_array_equal(compiled[0], reference[0])
    np.testing.assert_array_equal(compiled[1], reference[1])


def test_neutral_axis_slip_curvature_is_papailiou_onset():
    kappa = np.sin(np.linspace(0, 2 * math.pi, 401)) / 3.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnvalidatedModelWarning)
        result = _run(kappa)
    for i in (1, 2, 3):
        par = result.parameters[i]
        gross = 2 * par.eps_f_ref[0] / (par.beta_m[0] - par.beta_m[1])
        # Karlsen-Husoy gross-slip range at the extreme fibre = pi kappa_c.
        assert math.pi * result.critical_curvature_1_m[i] == pytest.approx(
            gross, rel=1e-12
        )


def test_corrected_trellis_rule_still_fails_the_nasution_tests():
    """Gap 1.3c gate G1 outcome (registered): the contact rule is not validated."""
    lives = {}
    for radius in (3.0, 6.5):
        damage = []
        for n in (3, 5):
            t = np.linspace(0.0, 2 * n * math.pi, 200 * n + 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UnvalidatedModelWarning)
                damage.append(_run(np.sin(t) / radius).D_total_per_ring)
        per_cycle = (damage[1] - damage[0]) / 2
        lives[radius] = 1.0 / per_cycle.max()
    # Tests: 2.33e5 and 8.32e5 cycles (Nasution 2014b Table 2).
    assert lives[3.0] < 2.33e5 / 3
    assert lives[6.5] < 8.32e5 / 3
    assert lives[6.5] / lives[3.0] < 1.8  # measured 3.6


# --------------------------------------------------------------------------- #
# Gap CF2: SWT mean-stress default and the critical-distance trellis rule      #
# --------------------------------------------------------------------------- #
def _nasution_life(radius: float, *, mean_stress, critical_distance=None) -> float:
    """Steady per-cycle life of the most damaged ring, 300 mm^2 test set-up."""
    damage = []
    for n in (3, 5):
        t = np.linspace(0.0, 2 * n * math.pi, 200 * n + 1)
        result = _run(
            np.sin(t) / radius,
            swt_const=None,
            mean_stress=mean_stress,
            critical_distance=critical_distance,
        )
        damage.append(result.D_total_per_ring)
    return float(1.0 / ((damage[1] - damage[0]) / 2).max())


def test_critical_distance_follows_threshold_and_curve():
    from cablexsec.analysis.conductor_fatigue import CriticalDistanceSpec

    spec = CriticalDistanceSpec(threshold_MPa_sqrt_m=3.0, reference_cycles=1e7)
    dsigma0 = 10 ** ((SN.log_C1 - 7.0) / SN.m1)  # MPa, single-slope curve
    assert spec.length_m(SN) == pytest.approx((3.0 / dsigma0) ** 2 / math.pi, rel=1e-6)
    with pytest.raises(CableInputError):
        CriticalDistanceSpec(threshold_MPa_sqrt_m=0.0)


def test_shakedown_contact_is_hertz_below_the_hardness_and_conforms_above():
    from cablexsec.analysis.conductor_fatigue import _shakedown_contact

    rings = check.nasution_rings()
    ring, partner = rings[1], rings[2]
    a, b, p0, plastic = _shakedown_contact(ring, partner, 1.0, 115e9, 0.3, 300e6, 3.0)
    assert not plastic
    a, b, p0, plastic = _shakedown_contact(ring, partner, 200.0, 115e9, 0.3, 300e6, 3.0)
    assert plastic
    # Mean pressure of the indented patch equals the hardness 3 sigma_y.
    assert 200.0 / (math.pi * a * b) == pytest.approx(900e6, rel=1e-9)
    assert p0 == pytest.approx(1.5 * 900e6, rel=1e-9)


def test_trellis_rule_without_contact_load_is_the_bulk_swt_count():
    """No contact pressure: the plane along the wire carries the bulk stress alone."""
    from cablexsec.analysis.conductor_fatigue import (
        CriticalDistanceSpec,
        _trellis_critical_distance,
    )
    from cablexsec.analysis.sn_assessment import corrected_damage
    from cablexsec.materials.mean_stress import MeanStressSpec

    t = np.linspace(0.0, 6 * math.pi, 601)
    bulk = (100.0 + 110.0 * np.sin(t))[None, :] * 1e6
    damage, _e, _x = _trellis_critical_distance(
        (1e-3, 1e-4, 0.0),
        1e-4,
        0.3,
        bulk,
        0.2 * np.sin(t)[None, :],
        SN,
        CriticalDistanceSpec(n_points=3, n_planes=6),
        False,
    )
    reference = corrected_damage(bulk[0], SN, mean_stress=MeanStressSpec(method="swt"))
    assert damage == pytest.approx(reference, rel=1e-12)


def test_swt_mean_stress_reproduces_the_nasution_bending_tests():
    """Gap CF2 gates CF2-L and CF2-T on the 300 mm^2 groups A1 and A2 (registered)."""
    from cablexsec.materials.mean_stress import MeanStressSpec

    swt = MeanStressSpec(method="swt")
    a1 = _nasution_life(3.0, mean_stress=swt)
    a2 = _nasution_life(6.5, mean_stress=swt)
    # Test geometric means 2.32e5 and 8.32e5 cycles ([Nasution2014b] Table 2).
    assert 2.32e5 / 3 <= a1 <= 3 * 2.32e5
    assert 8.32e5 / 3 <= a2 <= 3 * 8.32e5
    assert 1.8 <= a2 / a1 <= 7.2  # measured 3.6
    # Without the correction the radius trend is overstated (gap 1.3c diagnosis).
    assert (
        _nasution_life(6.5, mean_stress=None) / _nasution_life(3.0, mean_stress=None)
        > 7.2
    )


def test_95mm2_wire_curve_is_registered_from_nasution_2013():
    curve = get_sn_curve("nasution_2013_copper_95")
    assert curve.m1 == 6.238
    assert 10**curve.log_C1 == pytest.approx(6.098e19, rel=1e-12)
    assert curve.R_ratio == 0.1


# --------------------------------------------------------------------------- #
# Compacted wire rings and the designed 66 kV conductor                         #
# --------------------------------------------------------------------------- #
def test_compacted_wire_area_sets_the_ring_area_within_its_cell():
    ring = CoreStrandRing(
        n_wires=6, d_wire_m=3e-3, r_helix_m=3e-3, pitch_m=0.12, wire_area_m2=8e-6
    )
    assert ring.area_m2 == pytest.approx(6 * 8e-6)
    round_ring = CoreStrandRing(n_wires=6, d_wire_m=3e-3, r_helix_m=3e-3, pitch_m=0.12)
    assert round_ring.area_m2 == pytest.approx(6 * math.pi * 3e-3**2 / 4)
    with pytest.raises(ValueError, match="cell"):
        CoreStrandRing(
            n_wires=6, d_wire_m=3e-3, r_helix_m=3e-3, pitch_m=0.12, wire_area_m2=1e-5
        )


@pytest.mark.parametrize("pattern", ["equal_ratio", "equal_pitch"])
def test_designed_66kv_conductor_matches_the_published_data(pattern):
    cable = cx.iea_task49_66kv_stranded(lay_pattern=pattern, outer_lay_ratio=12.0)
    conductor = next(layer for layer in cable.layers if layer.name == "cu_conductor")
    rings = conductor.strand_layers
    assert [r.n_wires for r in rings] == [1, 6, 12, 18, 24]  # Janocha Fig. 4a
    assert sum(r.area_m2 for r in rings) == pytest.approx(630e-6, rel=1e-12)
    outer = rings[-1]
    assert 2 * outer.r_helix_m + outer.d_wire_m == pytest.approx(29.90e-3)
    assert outer.pitch_m / 29.90e-3 == pytest.approx(12.0)
    assert outer.lay_direction == "left_hand"
    assert [r.lay_direction for r in rings[1:]] == [
        "right_hand",
        "left_hand",
        "right_hand",
        "left_hand",
    ]
    ratios = [r.pitch_m / (2 * r.r_helix_m + r.d_wire_m) for r in rings[1:]]
    if pattern == "equal_ratio":
        assert ratios == pytest.approx([12.0] * 4)
    else:
        assert [r.pitch_m for r in rings[1:]] == pytest.approx([outer.pitch_m] * 4)
    t = np.linspace(0, 6 * math.pi, 601)
    result = cx.compute_conductor_fatigue(
        cable, np.sin(t) / 10.0, np.zeros_like(t), tension_N=np.full_like(t, 4e4)
    )
    assert result.critical_ring == 1
    # The default template keeps its area-equivalent solid conductor.
    solid = cx.load_cable(cx.iea_task49_66kv_yaml_path())
    assert not next(
        layer for layer in solid.layers if layer.name == "cu_conductor"
    ).strand_layers


# --------------------------------------------------------------------------- #
# Gap CF3: compaction factors per ring and the corrected wire curve           #
# --------------------------------------------------------------------------- #
def test_compaction_factor_scales_the_ring_stress_and_unit_factor_is_identity():
    from dataclasses import replace

    t = np.linspace(0.0, 6 * math.pi, 601)
    kappa = np.sin(t) / 3.0
    rings = check.nasution_rings()
    base = assess_conductor_fatigue(
        rings,
        check.NAS_E,
        0.3,
        kappa,
        np.zeros_like(kappa),
        np.full_like(kappa, check.NAS_T),
        sn_bulk=SN,
        swt_const=None,
        mu_wire=0.2,
        reference_tension_N=check.NAS_T,
    )
    same = assess_conductor_fatigue(
        tuple(replace(r, compaction_scf=1.0) for r in rings),
        check.NAS_E,
        0.3,
        kappa,
        np.zeros_like(kappa),
        np.full_like(kappa, check.NAS_T),
        sn_bulk=SN,
        swt_const=None,
        mu_wire=0.2,
        reference_tension_N=check.NAS_T,
    )
    np.testing.assert_array_equal(same.D_bulk_per_ring, base.D_bulk_per_ring)
    scaled = assess_conductor_fatigue(
        tuple(replace(r, compaction_scf=1.2) for r in rings),
        check.NAS_E,
        0.3,
        kappa,
        np.zeros_like(kappa),
        np.full_like(kappa, check.NAS_T),
        sn_bulk=SN,
        swt_const=None,
        mu_wire=0.2,
        reference_tension_N=check.NAS_T,
    )
    np.testing.assert_allclose(
        scaled.stress_range_max_Pa, 1.2 * base.stress_range_max_Pa, rtol=1e-12
    )
    with pytest.raises(CableInputError):
        replace(rings[1], compaction_scf=0.9)


def test_corrected_curve_and_compaction_option_of_the_66kv_conductor():
    curve = get_sn_curve("nasution_2013_copper_corrected")
    assert curve.m1 == 8.424 and 10**curve.log_C1 == pytest.approx(2.88e25)
    cable = cx.iea_task49_66kv_stranded(compaction="heavy")
    conductor = next(layer for layer in cable.layers if layer.name == "cu_conductor")
    assert [r.compaction_scf for r in conductor.strand_layers] == [
        1.0,
        1.026,
        1.026,
        1.026,
        1.202,
    ]
    assert conductor.bulk_sn == "nasution_2013_copper_corrected"
    plain = cx.iea_task49_66kv_stranded()
    assert all(
        r.compaction_scf == 1.0
        for r in next(
            lyr for lyr in plain.layers if lyr.name == "cu_conductor"
        ).strand_layers
    )


def test_default_trellis_threshold_is_full_hard_copper():
    from cablexsec.analysis.conductor_fatigue import CriticalDistanceSpec

    assert CriticalDistanceSpec().threshold_MPa_sqrt_m == 2.55
