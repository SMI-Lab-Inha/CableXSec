# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Zeidan (2025) four-point loops of a 112 mm three-core cable (Z-EI, Z-ID).

The digitised records, the facts read from them (doc/benchmarks.md, "Zeidan
(2025) four-point loops") and the parameter-free bound row Z-EI. The
identification row Z-ID needs minutes of beam simulations and is run by
``validation/scripts/check_zeidan2025.py`` and the validation report.
"""

from __future__ import annotations

import json
import math
import warnings

import check_zeidan2025 as zd
import numpy as np
import pytest

import cablexsec as cx

full_scale = pytest.mark.full_scale


@full_scale
def test_digitised_records_carry_their_provenance():
    record = json.loads(
        (zd.DATA / "zeidan2025_digitisation.json").read_text(encoding="utf-8")
    )
    thesis = record["thesis_sha256"]
    for figure in ("5_5", "5_6"):
        for amp in zd.AMPLITUDES_MM:
            path = zd.DATA / f"zeidan2025_fig{figure}_{amp}mm.csv"
            text = path.read_text(encoding="utf-8")
            assert f"Fig. {figure.replace('_', '.')}" in text
            assert thesis in text
            assert "native PDF polyline in plotted order" in text
            assert "reading half-width" in text
    # Paper Fig. 17 is the thesis Fig. 5.6 record: its outlines lie within
    # their own half-thickness of the thesis polylines.
    check = record["paper_fig17_vs_thesis_fig5_6"]
    half = 0.5 * check["outline_thickness_pt"]
    for amp in zd.AMPLITUDES_MM:
        assert check[f"{amp}mm"]["median_pt"] < half
        assert check[f"{amp}mm"]["p95_pt"] < half + 0.05


@full_scale
def test_plotted_curvature_is_eq51_of_the_nominal_displacement():
    for figure in ("5_5", "5_6"):
        for amp in zd.AMPLITUDES_MM:
            kappa, _ = zd.load_record(figure, amp)
            nominal = zd.EQ51 * amp * 1e-3
            assert kappa.max() == pytest.approx(nominal, abs=2e-4)
            assert kappa.min() == pytest.approx(-nominal, abs=2e-4)


@full_scale
def test_fig18_puts_the_nominal_displacement_at_mid_span():
    for amp in zd.AMPLITUDES_MM:
        x, w = zd.load_deflection(amp)
        assert -np.interp(0.999, x, w) == pytest.approx(amp * 1e-3, abs=5e-4)
        ratio = zd.central_curvature(x, w) / (zd.EQ51 * amp * 1e-3)
        assert 0.76 < ratio < 0.80
    assert abs(zd.R2_FACTOR - 1.4246) < 1e-4


@full_scale
def test_measured_loop_metrics_are_those_registered():
    loops = zd.measured_loops()
    registered = {
        30: (297.9, 3.49, 0.71, 0.92),
        40: (445.1, 4.14, 0.66, 1.01),
        50: (592.1, 4.68, 0.66, 1.05),
        60: (751.1, 5.16, 0.69, 1.11),
    }
    for amp, (work, rng, centre, half) in registered.items():
        loop = loops[amp]
        assert loop.work == pytest.approx(work, abs=0.05)
        assert loop.moment_range / 1e3 == pytest.approx(rng, abs=0.005)
        assert loop.centre / 1e3 == pytest.approx(centre, abs=0.005)
        assert loop.half_height / 1e3 == pytest.approx(half, abs=0.005)
        assert len(loop.works) == 4
    first = zd.measured_loops("5_5")
    for amp in zd.AMPLITUDES_MM:
        assert 1.19 < first[amp].work / loops[amp].work < 1.37


@full_scale
def test_published_bilinear_fit_overstates_the_stabilised_loop():
    half = zd.bilinear_half_height()
    assert half == pytest.approx((1 - 9.0 / 106) * 1830.0)
    for amp, loop in zd.measured_loops().items():
        assert loop.half_height < 0.7 * half
        assert zd.bilinear_work(zd.EQ51 * amp * 1e-3) > 1.4 * loop.work


@full_scale
def test_cable_geometry_follows_the_paper_table():
    cable = zd.load_base()
    inner = cable.get_layer("inner_armour")
    outer = cable.get_layer("outer_armour")
    for layer, pitch in ((inner, 0.913), (outer, 1.233)):
        angle = math.degrees(math.atan(2 * math.pi * layer.r_helix_m / pitch))
        assert layer.lay_angle_deg == pytest.approx(angle, abs=0.01)
    assert 2e3 * (outer.r_helix_m + 0.5 * outer.d_wire_m) == pytest.approx(103.0)
    assert 2e3 * (inner.r_helix_m + 0.5 * inner.d_wire_m) == pytest.approx(96.0)
    core = cable.get_layer("core_sheath")
    assert core.r_offset_m == pytest.approx(
        core.r_outer_m / math.sin(math.pi / 3), rel=1e-4
    )
    state = cable.initial_state
    assert state.external_pressure_Pa == zd.SOURCE_PRESSURE_PA
    assert state.identification == "published_calibration"


@full_scale
def test_z_ei_reports_the_registered_outcome():
    """The measured stick secant lies below the packaged no-slip EI_max; the
    measured slip slope (plotted curvature) lies 4-17 % below EI_min with the
    bonded outer face of the packaged input (gap G8): Z-EI fails narrowly.
    With both faces sliding EI_min is 2.72 kN m2 and the slope is about 3.5
    times above it. The shear stack stops at the support: the bedding under
    the inner armour (xi 0.832) and nothing between the armours (xi 1; the
    bonded outer layer keeps it, its support being rigid)."""
    out = zd.z_ei()
    assert out["pass"] is False
    assert out["pass_plane_section"] is True
    assert list(out["xi"].values()) == pytest.approx([0.832, 1.0], abs=1e-3)
    assert out["ei_max_Nm2"] == pytest.approx(311.1e3, rel=1e-3)
    assert out["ei_min_Nm2"] == pytest.approx(10.09e3, rel=1e-3)
    for amp in zd.AMPLITUDES_MM:
        assert max(out["stick_true_Nm2"][amp]) < out["ei_max_Nm2"]
        assert 0.83 < out["slip_Nm2"][amp] / out["ei_min_Nm2"] < 1.0


@full_scale
def test_concentric_support_coefficient_of_the_diagnostic():
    xi = zd.concentric_xi(zd.load_base())
    assert xi == pytest.approx([0.832, 0.897], abs=1e-3)


@pytest.mark.analytical_sanity
def test_section_law_reads_the_cable_materials_outside_a_test():
    """A law built from a cable with a ``materials:`` block reads those
    materials when it is built and when it is evaluated (it used to fall
    back silently outside :func:`simulate_bending_test`)."""
    cable = zd.load_base()
    kappa = np.linspace(0.0, 0.01, 6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        law = cx.SectionLaw.from_cable(cable, mu=0.2)
        reference = cx.compute_mkappa(cable, kappa, mu=0.2).moment_Nm
    np.testing.assert_array_equal(law.moment(kappa), reference)
