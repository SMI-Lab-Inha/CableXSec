# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Anchored wire at uniform curvature: Zhu and Tan (2014), Dong et al. (2025).

Reported comparison, not a gate (doc/benchmarks.md, "Anchored wire at uniform
curvature against published FE"). The analytical tests check the claims the
comparison rests on: the Zhu-Tan Table 2 fixes the regime parameter ``a``,
their shifted friction field breaks the Coulomb flow rule, the normalised
anchored stress depends on ``a`` (on the friction coefficient), and wires
anchored at the extreme fibres carry no anchor effect. The pins catch a
silent change of the recorded numbers (VALIDATION_POLICY Rule 9).
"""

from __future__ import annotations

import math

import check_anchored_wire_fe as aw
import numpy as np
import pytest

from cablexsec.mechanics.clamp_layer import (
    anchored_wire_response,
    uniform_curvature_force,
)

#: Coarse discretisation for the suite. The register uses 720 segments per
#: pitch and steps of 0.01; the coarse values differ from it by up to 0.01
#: normalised and 0.3 MPa, so the pins hold both within their tolerance.
COARSE = {"segments_per_pitch": 256, "step": 0.02}


def _a_zt() -> float:
    return aw.a_from_ratio(float(np.mean(aw.zt_ratio_from_table2())))


@pytest.mark.analytical_sanity
def test_zt_table2_rows_give_one_displacement_ratio():
    ratios = aw.zt_ratio_from_table2()
    assert max(ratios) - min(ratios) < 3e-3
    ratio = float(np.mean(ratios))
    assert ratio == pytest.approx(3.987, abs=2e-3)
    for ti, tm, tn in aw.ZT_TABLE2:
        m, n = aw.zt_shift_angles(math.radians(ti), ratio)
        assert math.degrees(m) == pytest.approx(tm, abs=0.06)
        assert math.degrees(n) == pytest.approx(tn, abs=0.06)
    # a = pi^2 U / 8 and the free-field amplitude 4 A / (pi U) (Fig. 14: 58 +- 2.5)
    assert aw.a_from_ratio(ratio) == pytest.approx(4.919, abs=3e-3)
    free = 4 * aw.zt_stick_amplitude_pa() / (math.pi * ratio)
    assert free / 1e6 == pytest.approx(58.0, abs=2.5)


@pytest.mark.analytical_sanity
def test_zt_shifted_field_breaks_the_coulomb_flow_rule():
    a = _a_zt()
    ratio = 8 * a / math.pi**2
    th, f, s = aw.zt_analytical_field(0.0, 0.0, 0.0)
    # the free field is consistent (up to the grid points at the sign changes)
    assert aw.coulomb_violation(th, f, s, a) < math.radians(0.5)
    for ti in (45.0, 90.0, 135.0):
        tm, tn = aw.zt_shift_angles(math.radians(ti), ratio)
        th, f, s = aw.zt_analytical_field(math.radians(ti), tm, tn)
        assert math.degrees(aw.coulomb_violation(th, f, s, a)) > 10.0
    tm, tn = aw.zt_shift_angles(math.pi / 2, ratio)
    th, f, s = aw.zt_analytical_field(math.pi / 2, tm, tn)
    assert s[0] == pytest.approx(2.16, abs=0.01)


@pytest.mark.analytical_sanity
def test_one_projection_is_a_consistent_coulomb_field_below_zhu_tan():
    """The deformation-theory solution obeys the flow rule exactly."""
    a = _a_zt()
    spp = COARSE["segments_per_pitch"]
    h = 2 * math.pi / spp
    phi = (np.arange(2 * spp) + 0.5) * h
    e_hat = np.cos(-math.pi / 2 + phi)
    n = anchored_wire_response(e_hat, np.array([a]), h)[0][-1]
    slip = np.cumsum((n - a * e_hat) * h)[:-1]
    dn = np.diff(n)
    sliding = np.abs(dn) > h * (1 - 1e-9)
    assert np.all(
        np.sign(dn[sliding & (np.abs(slip) > 1e-9)])
        == np.sign(slip[sliding & (np.abs(slip) > 1e-9)])
    )
    assert float(np.abs(slip[~sliding]).max()) < 1e-9
    assert n[0] / (math.pi / 2) < 1.1 < 2.16


@pytest.mark.analytical_sanity
def test_extreme_fibre_anchor_leaves_the_free_field():
    a = _a_zt()
    for theta0 in (0.0, math.pi):
        phi, n = aw.anchored_profile(theta0, a, pitches=2, **COARSE)
        free = uniform_curvature_force(theta0 + phi, a)
        assert float(np.abs(n - free).max()) < 1e-9


@pytest.mark.analytical_sanity
def test_normalised_anchor_stress_depends_on_a():
    rows = aw.peak_versus_a((1.5, 2.0, 4.92, 10.0), **COARSE)
    peaks = [r[2] for r in rows]
    assert peaks[0] == pytest.approx(1.0, abs=1e-3)  # no full slip: no rise
    assert all(p2 > p1 + 0.1 for p1, p2 in zip(peaks, peaks[1:], strict=False))
    assert peaks[2] == pytest.approx(1.70, abs=0.012)
    assert peaks[3] == pytest.approx(2.38, abs=0.01)


@pytest.mark.full_scale
def test_zhu_tan_pins():
    out = {}
    a = _a_zt()
    for ti in (45.0, 90.0, 135.0):
        phi, n = aw.anchored_profile(math.radians(ti) - math.pi, a, pitches=2, **COARSE)
        s = n / (math.pi / 2)
        out[ti] = (s[0], s[phi < 2 * math.pi].max())
    assert out[45.0][1] == pytest.approx(1.32, abs=0.015)
    assert out[90.0][0] == pytest.approx(1.19, abs=0.01)
    assert out[90.0][1] == pytest.approx(1.70, abs=0.012)
    assert out[135.0][0] == pytest.approx(1.76, abs=0.012)
    dig = aw.zt_digitised()
    fea = {ti: aw.curve_features(*dig[(ti, "fea")], 355.0, 4.0) for ti in (45, 90, 135)}
    assert fea[45]["peak"] == pytest.approx(1.59, abs=0.02)
    assert fea[90]["start"] == pytest.approx(2.22, abs=0.02)
    assert fea[135]["start"] == pytest.approx(2.19, abs=0.02)
    x0, y0 = dig[(0.0, "fea")]
    w = (x0 > 30) & (x0 < 1050)
    rms_cos = float(np.sqrt(np.mean((y0[w] + np.cos(np.radians(x0[w]))) ** 2)))
    assert rms_cos < 0.05  # the source's FE far field is a cosine, not a triangle


@pytest.mark.full_scale
def test_dong_pins():
    p = aw.dong_parameters()
    assert p["friction_N_m"] == pytest.approx(3894.5, abs=0.1)
    assert p["tension_stress_Pa"] / 1e6 == pytest.approx(140.91, abs=0.01)
    assert p["a"] == pytest.approx(1.8678, abs=1e-4)
    phi, n = aw.anchored_profile(-math.pi / 2, p["a"], pitches=1, **COARSE)
    stress = (p["tension_stress_Pa"] + p["sigma_u_Pa"] * n) / 1e6
    assert stress[0] == pytest.approx(156.6, abs=0.4)
    assert stress.max() == pytest.approx(207.7, abs=0.4)
    assert phi[np.argmax(stress)] / (2 * math.pi) == pytest.approx(0.217, abs=0.01)
    dig = aw.dong_digitised()
    fe14 = aw.curve_features(*dig[("14", "numerical")], 0.75, 0.004)
    assert fe14["start"] == pytest.approx(156.9, abs=0.5)
    assert fe14["peak"] == pytest.approx(204.5, abs=0.5)
    fe9 = aw.curve_features(*dig[("9", "numerical")], 0.25, 0.004)
    assert fe9["start"] == pytest.approx(194.5, abs=0.5)
    fe25 = aw.curve_features(*dig[("25", "90")], 180.0, 4.0)
    assert fe25["start"] == pytest.approx(164.8, abs=0.5)


def test_digitised_files_name_their_sources():
    for path, digest in (
        (aw.ZT_CSV, "ad2af9e711fedea4ba2404fa6570de6ec18007a5ec8096e6f27ba7bd60e86c60"),
        (
            aw.DONG_CSV,
            "3ebec879ff0edf43c4390ad8832afcac9e2ad6dd13ed4b28ee4b7b44747d113d",
        ),
    ):
        head = path.read_text(encoding="utf-8").split("\n", 20)
        assert any(digest in line for line in head)


@pytest.mark.independent_method
def test_report_rows_are_pinned():
    import run_validation  # noqa: PLC0415

    rows = {r.ident: r for r in run_validation._anchored_wire_fe_rows()}
    dong = rows["AW-DONG25"]
    assert dong.target == "stated, not gated (code-to-code)"
    assert dong.value == (
        "Fig. 14 anchor 156.4 vs FE 156.9 (source theory 147.8); peak 207.8 at "
        "X/P 0.217 vs 204.5 at 0.223; Fig. 25 anchors "
        "81.8/117.6/156.4/191.9/200.0 vs 89.3/121.2/164.8/199.1/195.0"
    )
    assert dong.status == "INFO"
    zt = rows["AW-ZT14"]
    assert zt.value == (
        "90 deg 1.19 / 1.70 vs FE 2.22 / 2.23; 45 deg 0.05 / 1.32 vs 0.97 / 1.59; "
        "FE far field RMS distance to a cosine 0.028, to the full-slip triangle "
        "0.154 (the FE wire is not in slip)"
    )
    assert zt.status == "INFO"
