# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""L7 — publication figures + the exploratory viewer.

Gates:
* every one of the nine figure functions runs headless, returns a Matplotlib
  object, and saves a non-empty 300-DPI PNG (the gallery);
* the **provenance-annotation rule**: a figure annotates its source result's
  limitations verbatim (a figure cannot present a number more confidently than
  the result allows);
* the **viewer contract**: the dashboard renders result-object numbers +
  provenance and is marked render-only; a result limitation appears verbatim and
  a result number is present (the JS recomputes nothing).
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")  # headless

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import cablexsec as cx  # noqa: E402
import cablexsec.plotting as viz  # noqa: E402

pytestmark = pytest.mark.analytical_sanity


def _cable() -> cx.CableData:
    return cx.load_cable(cx.inter_array_single_core_yaml_path(), geometry_override=None)


@pytest.fixture(scope="module")
def results():
    cab = _cable()
    stress = cx.compute_stress(cab, 0.03, tension_N=0.0)
    mk = cx.compute_mkappa(cab, np.linspace(0, 0.03, 200), tension_N=0.0)
    cap = cx.compute_capacity(cab, np.linspace(-5e5, 5e5, 31))
    k = 0.02 * np.sin(np.linspace(0, 40 * np.pi, 300))
    fat = cx.compute_fatigue(cab, k, np.zeros_like(k))
    ser = cx.jonswap_series(duration_s=120, dt_s=0.5, rms_curvature_1_m=0.01, seed=3)
    tb = cx.load_cable(cx.cigre_tb880_yaml_path(), geometry_override=None)
    th = cx.compute_thermal(tb, 908.45, 1200e-6, n_cores=1)
    return {
        "cab": cab,
        "stress": stress,
        "mk": mk,
        "cap": cap,
        "fat": fat,
        "ser": ser,
        "tb": tb,
        "th": th,
    }


def _save_nonempty(fig, tmp_path, name) -> None:
    p = tmp_path / f"{name}.png"
    viz.save_figure(fig, str(p))
    assert p.stat().st_size > 5000  # a real raster, not a stub


def test_gallery_all_nine_render_and_save(results, tmp_path) -> None:
    import matplotlib.pyplot as plt

    r = results
    _save_nonempty(viz.plot_cross_section(r["cab"], r["stress"]).figure, tmp_path, "cs")
    _save_nonempty(viz.plot_mkappa(r["mk"]).figure, tmp_path, "mk")
    _save_nonempty(viz.plot_stress_distribution(r["stress"]), tmp_path, "sd")
    _save_nonempty(viz.plot_stress_timeseries(r["ser"]), tmp_path, "ts")
    _save_nonempty(viz.plot_capacity_envelope(r["cap"]), tmp_path, "cap")
    _save_nonempty(
        viz.plot_temperature_gradient(r["th"], r["tb"]).figure, tmp_path, "th"
    )
    _save_nonempty(viz.plot_damage_split(r["fat"]), tmp_path, "dmg")
    _save_nonempty(viz.plot_fretting_polar(r["stress"]), tmp_path, "fp")
    _save_nonempty(viz.plot_mkappa_validation_suite({"BM": r["mk"]}), tmp_path, "suite")
    plt.close("all")


def _all_text(fig) -> str:
    return " ".join(
        t.get_text() for t in [*fig.texts, *(t for ax in fig.axes for t in ax.texts)]
    )


@pytest.mark.parametrize("caveats", [None, "absent", (), ("Long caveat. " * 30,) * 3])
def test_stress_panels_and_caveats_do_not_overlap(results, caveats):
    import matplotlib.pyplot as plt

    result = results["stress"]
    if caveats == "absent":
        result = replace(result, provenance=None)
    elif caveats is not None:
        result = replace(
            result, provenance=replace(result.provenance, limitations=caveats)
        )
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*[Tt]ight layout.*")
        fig = viz.plot_stress_distribution(result)
        fig.canvas.draw()
    try:
        assert len(fig.axes) == 6
        renderer = fig.canvas.get_renderer()
        boxes = [ax.get_tightbbox(renderer) for ax in fig.axes]
        for index, box in enumerate(boxes):
            assert all(not box.overlaps(other) for other in boxes[index + 1 :])
        if result.provenance and result.provenance.limitations:
            footer = next(t for t in fig.texts if t.get_text().startswith("•"))
            box = footer.get_window_extent(renderer)
            assert box.y1 < min(panel.y0 for panel in boxes)
            assert fig.bbox.contains(box.x0, box.y0)
            assert fig.bbox.contains(box.x1, box.y1)
            text = " ".join(footer.get_text().split())
            for caveat in result.provenance.limitations[:2]:
                assert " ".join(caveat.split()) in text
            if len(result.provenance.limitations) > 2:
                assert f"(+{len(result.provenance.limitations) - 2} more" in text
    finally:
        plt.close(fig)


def test_figures_annotate_provenance_limitations_verbatim(results) -> None:
    """A figure must carry its result's limitation strings verbatim."""
    r = results
    # M-κ figure: its first limitation (the Papailiou-vs-loop caveat) must appear.
    ax = viz.plot_mkappa(r["mk"])
    fig_text = _all_text(ax.figure)
    lim0 = r["mk"].provenance.limitations[0]
    # the annotation truncates long strings to a prefix; check a verbatim chunk.
    assert lim0[:35] in fig_text

    # Cross-section coloured by stress carries the stress result's limitation.
    ax_cs = viz.plot_cross_section(r["cab"], r["stress"])
    cs_text = _all_text(ax_cs.figure)
    assert r["stress"].provenance.limitations[0][:30] in cs_text


def test_signed_colormap_is_diverging_not_rainbow() -> None:
    """The signed colormap is the blue-white-red house map (no jet/rainbow)."""
    cmap = viz.SIGNED_CMAP
    lo = cmap(0.0)  # blue end
    mid = cmap(0.5)  # white
    hi = cmap(1.0)  # red end
    assert lo[2] > lo[0]  # blue dominant at low
    assert hi[0] > hi[2]  # red dominant at high
    assert min(mid[:3]) > 0.9  # near-white centre


def test_dashboard_is_a_viewer_not_an_engine(results, tmp_path) -> None:
    """The dashboard renders result numbers + provenance and recomputes nothing."""
    r = results
    out = tmp_path / "dash.html"
    p = viz.build_dashboard(
        r["cab"],
        stress=r["stress"],
        mkappa=r["mk"],
        capacity=r["cap"],
        fatigue=r["fat"],
        out_path=out,
    )
    html = p.read_text(encoding="utf-8")

    # render-only contract marker + the explicit "computes nothing" banner.
    assert 'data-viewer="render-only"' in html
    assert "computes no claimed number" in html

    # a result-object NUMBER appears verbatim (came from the result, not JS).
    assert f"{r['mk'].W_diss_J_per_m:.1f}" in html

    # a source result's limitation appears verbatim (provenance carried through).
    assert r["mk"].provenance.limitations[0] in html
    assert r["stress"].provenance.limitations[0] in html


def test_dashboard_geometry_only_without_stress(results, tmp_path) -> None:
    """Without a StressResult the viewer still renders (geometry-only)."""
    out = tmp_path / "geom.html"
    p = viz.build_dashboard(results["cab"], out_path=out)
    html = p.read_text(encoding="utf-8")
    assert "geometry only" in html  # legend states no stress field


def test_cross_section_material_mode_without_stress(results) -> None:
    """The cross-section renders by material when no stress result is given."""
    ax = viz.plot_cross_section(results["cab"])
    assert ax.figure is not None


def test_stress_timeseries_with_overlays(results) -> None:
    """The σ_total / δ* overlay panels render when result-derived arrays given."""
    ser = results["ser"]
    n = ser.n_steps
    sigma_crit = 2.0e8 * np.sin(np.linspace(0, 20 * np.pi, n))
    delta_star = 2.0 + np.abs(np.sin(np.linspace(0, 10 * np.pi, n)))
    fig = viz.plot_stress_timeseries(
        ser, sigma_crit_Pa=sigma_crit, delta_star_um=delta_star, sigma_y_Pa=6.9e8
    )
    assert len(fig.axes) == 4  # κ, σ, δ*, tension
