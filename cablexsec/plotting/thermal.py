# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Radial temperature-gradient figure from a ThermalResult."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cablexsec.plotting.style import (
    COLORS,
    FIGSIZE,
    annotate_provenance,
    use_style,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from cablexsec.cables.schema import CableData
    from cablexsec.mechanics.thermal import ThermalResult

__all__ = ["plot_temperature_gradient"]


def plot_temperature_gradient(
    result: ThermalResult,
    cable: CableData | None = None,
    *,
    current_A: float | None = None,
    ax: Axes | None = None,
) -> Axes:
    r"""Radial temperature profile ``T(r)`` from a :class:`ThermalResult`.

    Solid black ``T(r)``; grey dashed layer boundaries (if ``cable`` given);
    conductor (red) and armour (blue) node temperatures as dashed guides; a text
    box with the current, total loss, and conductor temperature. The result's
    limitations are annotated verbatim (e.g. the BM5 three-core caveat).

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    with use_style():
        if ax is None:
            _, ax = plt.subplots(figsize=FIGSIZE["single"])

        r_mm = result.r_m * 1e3
        ax.plot(
            r_mm, result.T_C, color=COLORS["BLACK"], lw=1.6, zorder=3, label=r"$T(r)$"
        )

        if cable is not None:
            for lyr in cable.layers:
                r_o = getattr(lyr, "r_outer_m", None)
                if r_o:
                    ax.axvline(
                        r_o * 1e3, color=COLORS["GREY"], ls="--", lw=0.5, zorder=1
                    )

        ax.axhline(
            result.T_conductor_C,
            color=COLORS["RED"],
            ls="--",
            lw=0.9,
            label=f"conductor {result.T_conductor_C:.1f} °C",
        )
        ax.axhline(
            result.T_armour_C,
            color=COLORS["BLUE"],
            ls="--",
            lw=0.9,
            label=f"armour {result.T_armour_C:.1f} °C",
        )

        ax.set_xlabel("radius [mm]")
        ax.set_ylabel("temperature [°C]")
        ax.set_title("Radial temperature gradient")
        ax.legend(loc="upper right", fontsize=8)

        lines = [
            f"$P$ = {result.P_total_W_m:.1f} W/m",
            f"$T_{{cond}}$ = {result.T_conductor_C:.1f} °C",
        ]
        if current_A is not None:
            lines.insert(0, f"$I$ = {current_A:.0f} A")
        ax.text(
            0.03,
            0.03,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox={
                "facecolor": "white",
                "edgecolor": COLORS["GREY"],
                "boxstyle": "round,pad=0.3",
            },
        )
        annotate_provenance(ax, result.provenance, loc="upper left", max_items=2)
    return ax
