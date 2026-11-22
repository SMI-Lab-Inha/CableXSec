# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""MATLAB-style plotting settings for every CableXSec figure.

The style follows MATLAB's default axes (R2014b and later):

- the seven-colour ``lines`` order (blue, orange, yellow, purple, green, cyan,
  dark red) as the line cycle;
- a sans-serif face (Helvetica, Arial or their metric clones) at 10 pt, axis
  labels and titles at 11 pt;
- a closed box with inward ticks on all four sides, dark-grey axes, no grid;
- a square-cornered, opaque legend box.

Symbols are typeset with matplotlib's mathtext in the same sans-serif face, so
labels carry Greek letters and true sub- and superscripts (``$\\kappa$``,
``$EI_\\mathrm{min}$``, ``m$^{\\mathrm{-1}}$``). Numerals inside math are
set upright with ``\\mathrm``, variables in italic. :data:`LABELS` holds the
standard axis labels. Apply the style once with :func:`apply_style`, or scope it with
:func:`use_style`.

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
    "LABELS",
    "MATERIAL_COLORS",
    "MATLAB_ORDER",
    "SIGNED_CMAP",
    "STYLE",
    "annotate_provenance",
    "apply_style",
    "material_color",
    "save_figure",
    "signed_norm",
    "use_style",
]

#: MATLAB's default ``lines`` colour order (R2014b and later), in cycle order.
MATLAB_ORDER: tuple[str, ...] = (
    "#0072BD",  # blue
    "#D95319",  # orange
    "#EDB120",  # yellow
    "#7E2F8E",  # purple
    "#77AC30",  # green
    "#4DBEEE",  # cyan
    "#A2142F",  # dark red
)

# Named colours. Use these names, never ad-hoc hex codes. The hues are those of
# MATLAB_ORDER; BLACK and GREY are for references, data markers and outlines.
COLORS: dict[str, str] = {
    "BLUE": MATLAB_ORDER[0],  # primary model result
    "ORANGE": MATLAB_ORDER[1],  # second series, measured data
    "YELLOW": MATLAB_ORDER[2],
    "PURPLE": MATLAB_ORDER[3],
    "GREEN": MATLAB_ORDER[4],  # safe, secondary data
    "CYAN": MATLAB_ORDER[5],
    "RED": MATLAB_ORDER[6],  # warning, critical, threshold, hot
    "BLACK": "#000000",  # measured markers, outlines
    "GREY": "#808080",  # reference lines, minor data
}

#: MATLAB's axes and text colour, ``[0.15 0.15 0.15]``.
_AXES_GREY = "#262626"

# Canonical figure sizes [inches].
FIGSIZE: dict[str, tuple[float, float]] = {
    "single": (5.0, 4.2),  # one-panel figure
    "wide": (7.0, 3.6),  # time-series / wide aspect
    "square": (5.0, 5.0),  # cross-section / polar
    "six_panel": (9.0, 6.0),  # 2x3 stress components
    "suite": (9.0, 6.5),  # validation-suite grid
}

#: Standard axis labels, with symbols and SI units.
LABELS: dict[str, str] = {
    "curvature": r"Curvature $\kappa$ [m$^{\mathrm{-1}}$]",
    "curvature_milli": r"Curvature $\kappa$ [$10^{\mathrm{-3}}$ m$^{\mathrm{-1}}$]",
    "moment_Nm": r"Bending moment $M$ [N$\cdot$m]",
    "moment_kNm": r"Bending moment $M$ [kN$\cdot$m]",
    "tension_kN": r"Tension $T$ [kN]",
    "time": r"Time $t$ [s]",
    "azimuth": r"Wire azimuth $\varphi$ [$^\circ$]",
    "radius_mm": r"Radius $r$ [mm]",
    "temperature": r"Temperature $T$ [$^\circ$C]",
    "stress": r"Stress $\sigma$ [MPa]",
    "x_mm": r"$x$ [mm]",
    "y_mm": r"$y$ [mm]",
    "x_m": r"$x$ [m]",
    "y_m": r"$y$ [m]",
    "damage": r"Miner damage $D$ [–]",
    "slip_um": r"Slip amplitude $\delta$ [$\mu$m]",
}

# The ONE signed-field colormap: blue (-) -> white (0) -> dark red (+), from the
# MATLAB hues. Diverging fields use THIS, never rainbow/jet (which encodes
# magnitude as hue and is unreadable for sign).
SIGNED_CMAP: Colormap = LinearSegmentedColormap.from_list(
    "cablexsec_signed", [COLORS["BLUE"], "#FFFFFF", COLORS["RED"]], N=256
)

# Material -> fill colour for the by-material cross-section view.
MATERIAL_COLORS: dict[str, str] = {
    "steel": "#8C8C8C",
    "copper": "#D9874A",
    "cu": "#D9874A",
    "xlpe": "#9CC3E4",
    "epr": "#9CC3E4",
    "hdpe": "#BFBFBF",
    "mdpe": "#CFCFCF",
    "ldpe": "#DCDCDC",
    "pp": "#E6E6E6",
    "lead": "#4D4D4D",
    "aluminium": "#A6A6A6",
    "al": "#A6A6A6",
}

# Matplotlib rcParams for the house style.
STYLE: dict[str, object] = {
    # Text: MATLAB's Helvetica, with Arial and metric clones as fallbacks.
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Helvetica",
        "Arial",
        "Liberation Sans",
        "Nimbus Sans",
        "DejaVu Sans",
    ],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "text.color": _AXES_GREY,
    "text.usetex": False,
    # Symbols in the same sans-serif face as the text.
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans",
    "mathtext.it": "sans:italic",
    "mathtext.bf": "sans:bold",
    "mathtext.sf": "sans",
    "mathtext.default": "it",
    "axes.unicode_minus": True,
    # Axes: closed box, inward ticks on all sides, no grid.
    "axes.linewidth": 0.6,
    "axes.edgecolor": _AXES_GREY,
    "axes.labelcolor": _AXES_GREY,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": False,
    "grid.color": _AXES_GREY,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.5,
    "axes.prop_cycle": cycler(color=list(MATLAB_ORDER)),
    "axes.facecolor": "white",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.color": _AXES_GREY,
    "ytick.color": _AXES_GREY,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    # Lines and markers.
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "lines.markeredgewidth": 0.8,
    "patch.linewidth": 0.6,
    # Legend: square, opaque, thin dark edge.
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.framealpha": 1.0,
    "legend.edgecolor": _AXES_GREY,
    "legend.shadow": False,
    # Figure and output.
    "figure.facecolor": "white",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "savefig.facecolor": "white",
}


def apply_style() -> None:
    """Apply the CableXSec MATLAB-style settings to the global rcParams."""
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
            "boxstyle": "square,pad=0.25",
            "alpha": 0.92,
            "linewidth": 0.5,
        },
    )


def save_figure(fig: Figure, path: str) -> str:
    """Save a figure at the house 300 DPI / tight-box settings; return the path."""
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    return path
