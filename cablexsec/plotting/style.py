# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Publication-ready MATLAB-style plotting settings.

All CableXSec figures share one visual style: a fixed six-colour palette,
Times New Roman serif at 11 pt, inward ticks, light grid, and 300 DPI output.
Apply it once at the top of a figure script with :func:`apply_style`, or scope
it to a block with :func:`use_style`.

Examples
--------
>>> from cablexsec.plotting import style
>>> style.apply_style()                      # global, persistent
>>> with style.use_style():                  # scoped, auto-restored
...     ...  # draw here
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

import matplotlib as mpl
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, Normalize
    from matplotlib.figure import Figure

    from cablexsec.provenance import Provenance

__all__ = [
    "COLORS",
    "FIGSIZE",
    "MATERIAL_COLORS",
    "SIGNED_CMAP",
    "STYLE",
    "annotate_provenance",
    "apply_style",
    "material_color",
    "save_figure",
    "signed_norm",
    "use_style",
]

# Standard colour palette. Use these names, never ad-hoc hex codes.
COLORS: dict[str, str] = {
    "BLACK": "#000000",  # primary lines, critical results
    "RED": "#CC0000",  # warning, critical, threshold, hot
    "BLUE": "#0000CC",  # primary data, reference, cold
    "GREEN": "#006600",  # safe, secondary data
    "GREY": "#666666",  # reference lines, minor data
    "ORANGE": "#CC6600",  # tertiary data
}

# Canonical figure sizes [inches] — one width language across the paper.
FIGSIZE: dict[str, tuple[float, float]] = {
    "single": (5.0, 4.2),  # one-panel figure
    "wide": (7.0, 3.6),  # time-series / wide aspect
    "square": (5.0, 5.0),  # cross-section / polar
    "six_panel": (9.0, 6.0),  # 2x3 stress components
    "suite": (9.0, 6.5),  # validation-suite grid
}

# The ONE signed-field colormap: blue (-) -> white (0) -> red (+), built from
# the house palette. Diverging fields use THIS, never rainbow/jet (which encodes
# magnitude as hue and is unreadable for sign).
SIGNED_CMAP: Colormap = LinearSegmentedColormap.from_list(
    "cablexsec_signed", [COLORS["BLUE"], "#FFFFFF", COLORS["RED"]], N=256
)

# Material -> fill colour for the by-material cross-section view.
MATERIAL_COLORS: dict[str, str] = {
    "steel": COLORS["GREY"],
    "copper": COLORS["ORANGE"],
    "cu": COLORS["ORANGE"],
    "xlpe": COLORS["BLUE"],
    "hdpe": "#BBBBBB",
    "pp": "#DDDDDD",
    "lead": "#444444",
    "aluminium": "#AAAAAA",
    "al": "#AAAAAA",
}

# Matplotlib rcParams for the house style.
STYLE: dict[str, object] = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "grid.color": "#999999",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "text.usetex": False,
    # Default colour cycle follows the house palette.
    "axes.prop_cycle": cycler(
        color=[
            COLORS["BLACK"],
            COLORS["RED"],
            COLORS["BLUE"],
            COLORS["GREEN"],
            COLORS["GREY"],
            COLORS["ORANGE"],
        ]
    ),
}


def apply_style() -> None:
    """Apply the CableXSec house style to the global matplotlib rcParams."""
    cast(Any, mpl.rcParams).update(STYLE)


@contextmanager
def use_style() -> Iterator[None]:
    """Context manager applying the house style, restoring rcParams on exit."""
    with mpl.rc_context(cast(Any, STYLE)):
        yield


def signed_norm(vmax_abs: float) -> Normalize:
    """A symmetric :class:`TwoSlopeNorm` centred at 0 for :data:`SIGNED_CMAP`.

    Use for any signed field (stress components, etc.) so zero maps to white and
    ``±vmax_abs`` map to full blue/red. Guards a degenerate all-zero field.
    """
    v = abs(float(vmax_abs)) or 1.0
    return TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)


def material_color(material: str) -> str:
    """House fill colour for a material name (substring match), else mid-grey."""
    m = material.lower()
    for key, hexcode in MATERIAL_COLORS.items():
        if key in m:
            return hexcode
    return COLORS["GREY"]


def annotate_provenance(
    ax: Axes,
    provenance: Provenance | None,
    *,
    loc: str = "lower left",
    max_items: int = 3,
    prefix: str = "",
) -> None:
    """Annotate a figure with a result's limitations **verbatim**.

    A figure must not present a number more confidently than its result object
    allows (the demo-provenance rule, for plots): the limitation strings are
    copied verbatim from ``provenance.limitations``, never paraphrased. Nothing
    is drawn when the result carries no limitations.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    provenance : Provenance or None
        The source result's provenance; its ``limitations`` are rendered.
    loc : {"lower left", "lower right", "upper left", "upper right"}, optional
        Corner for the caveat box.
    max_items : int, optional
        Cap on the number of limitation lines shown (the rest are summarised as
        "+N more"); ``provenance`` is never silently truncated to zero.
    prefix : str, optional
        Optional leading line (e.g. a headline caveat for this figure).
    """
    if provenance is None or not provenance.limitations:
        return
    lims = list(provenance.limitations)
    shown = lims[:max_items]
    lines = [f"• {s}" for s in shown]
    if len(lims) > max_items:
        lines.append(f"• (+{len(lims) - max_items} more — see provenance)")
    if prefix:
        lines.insert(0, prefix)
    text = "\n".join(lines)

    va, ha = (
        ("bottom" if "lower" in loc else "top"),
        ("left" if "left" in loc else "right"),
    )
    x = 0.02 if ha == "left" else 0.98
    y = 0.02 if va == "bottom" else 0.98
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=6.5,
        color=COLORS["GREY"],
        zorder=10,
        wrap=True,
        bbox={
            "facecolor": "white",
            "edgecolor": COLORS["GREY"],
            "boxstyle": "round,pad=0.25",
            "alpha": 0.92,
            "linewidth": 0.5,
        },
    )


def save_figure(fig: Figure, path: str) -> str:
    """Save a figure at the house 300 DPI / tight-box settings; return the path."""
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    return path
