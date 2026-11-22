# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for the MATLAB-style plotting settings."""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")  # headless backend for CI

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from cablexsec.plotting import style  # noqa: E402

#: MATLAB's default ``lines`` colour order (R2014b and later).
MATLAB_LINES = (
    "#0072BD",
    "#D95319",
    "#EDB120",
    "#7E2F8E",
    "#77AC30",
    "#4DBEEE",
    "#A2142F",
)


# Test-taxonomy category.
pytestmark = pytest.mark.analytical_sanity


def test_colour_order_is_matlab_lines() -> None:
    assert style.MATLAB_ORDER == MATLAB_LINES
    style.apply_style()
    cycle = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    assert tuple(c.upper() for c in cycle) == MATLAB_LINES


def test_named_colours_are_matlab_hues() -> None:
    hues = {k: v for k, v in style.COLORS.items() if k not in ("BLACK", "GREY")}
    assert set(hues.values()) == set(MATLAB_LINES)
    assert style.COLORS["BLUE"] == MATLAB_LINES[0]


def test_apply_style_sets_the_matlab_axes() -> None:
    style.apply_style()
    assert mpl.rcParams["font.family"] == ["sans-serif"]
    assert mpl.rcParams["font.sans-serif"][:2] == ["Helvetica", "Arial"]
    assert mpl.rcParams["font.size"] == 10
    assert mpl.rcParams["axes.grid"] is False
    assert mpl.rcParams["xtick.direction"] == "in"
    assert mpl.rcParams["ytick.direction"] == "in"
    assert mpl.rcParams["xtick.top"] is True
    assert mpl.rcParams["ytick.right"] is True
    assert mpl.rcParams["legend.fancybox"] is False
    assert mpl.rcParams["savefig.dpi"] == 300
    assert mpl.rcParams["text.usetex"] is False
    assert mpl.rcParams["mathtext.fontset"] == "custom"


def test_use_style_restores_rcparams() -> None:
    mpl.rcParams["font.size"] = 7.0
    with style.use_style():
        assert mpl.rcParams["font.size"] == 10
    assert mpl.rcParams["font.size"] == 7.0


def test_standard_labels_render_with_symbols() -> None:
    """Every standard label is valid mathtext and keeps numerals upright."""
    with style.use_style():
        fig, ax = plt.subplots()
        for text in style.LABELS.values():
            ax.set_xlabel(text)
            fig.canvas.draw()  # raises on invalid mathtext
        plt.close(fig)
    assert r"\kappa" in style.LABELS["curvature"]
    assert r"m$^{\mathrm{-1}}$" in style.LABELS["curvature"]
    assert "1/m" not in "".join(style.LABELS.values())
