# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for the MATLAB-style plotting settings."""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")  # headless backend for CI

import matplotlib as mpl  # noqa: E402

from cablexsec.plotting import style  # noqa: E402

EXPECTED_PALETTE = {
    "BLACK": "#000000",
    "RED": "#CC0000",
    "BLUE": "#0000CC",
    "GREEN": "#006600",
    "GREY": "#666666",
    "ORANGE": "#CC6600",
}


# Test-taxonomy category.
pytestmark = pytest.mark.analytical_sanity


def test_palette_exact() -> None:
    assert style.COLORS == EXPECTED_PALETTE


def test_apply_style_sets_rcparams() -> None:
    style.apply_style()
    assert mpl.rcParams["font.size"] == 11
    assert mpl.rcParams["savefig.dpi"] == 300
    assert mpl.rcParams["figure.dpi"] == 300
    assert mpl.rcParams["xtick.direction"] == "in"
    assert mpl.rcParams["ytick.direction"] == "in"
    assert mpl.rcParams["text.usetex"] is False
    assert "Times New Roman" in mpl.rcParams["font.serif"]


def test_use_style_restores_rcparams() -> None:
    mpl.rcParams["font.size"] = 7.0
    with style.use_style():
        assert mpl.rcParams["font.size"] == 11
    assert mpl.rcParams["font.size"] == 7.0


def test_prop_cycle_uses_palette() -> None:
    style.apply_style()
    cycle_colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    assert cycle_colors[0] == "#000000"
    assert set(EXPECTED_PALETTE.values()) == set(cycle_colors)
