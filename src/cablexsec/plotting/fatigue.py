# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Fatigue figures: damage split and the CMR-vs-loxodromic fretting polar."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cablexsec.plotting.style import (
    COLORS,
    FIGSIZE,
    annotate_provenance,
    use_style,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from cablexsec.analysis.fatigue import FatigueResult
    from cablexsec.analysis.stress import StressResult

__all__ = ["plot_damage_split", "plot_fretting_polar"]


def plot_damage_split(result: FatigueResult) -> Figure:
    r"""Two-panel fatigue damage breakdown from a :class:`FatigueResult`.

    Panel (a): a log-scale grouped bar chart of the per-mechanism damage
    (bulk / fretting / termination / total) at the **critical wire** of each
    layer. Panel (b): a polar plot of ``D_total(φ)`` for the critical layer. The
    result's limitations are annotated verbatim.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    layers = list(result.D_total_per_wire)
    with use_style():
        fig = plt.figure(figsize=(9.0, 4.0))
        axa = fig.add_subplot(1, 2, 1)
        axb = fig.add_subplot(1, 2, 2, projection="polar")

        # (a) per-layer critical-wire damage split (log scale).
        parts = [
            ("bulk", result.D_bulk_per_wire, COLORS["BLUE"]),
            ("fretting", result.D_fretting_per_wire, COLORS["ORANGE"]),
            ("termination", result.D_term_per_wire, COLORS["YELLOW"]),
            ("total", result.D_total_per_wire, COLORS["PURPLE"]),
        ]
        x = np.arange(len(layers))
        w = 0.2
        for j, (label, dct, color) in enumerate(parts):
            crit = [float(np.max(dct[n])) if dct[n].size else 0.0 for n in layers]
            crit = [max(c, 1e-30) for c in crit]  # log floor
            axa.bar(x + (j - 1.5) * w, crit, width=w, color=color, label=label)
        axa.set_yscale("log")
        axa.set_xticks(x)
        axa.set_xticklabels(layers, fontsize=8, rotation=15)
        axa.set_ylabel(r"Miner damage per wire $D$ [–]")
        axa.set_title("(a) Damage split (critical wire)")
        axa.legend(loc="best", fontsize=8)

        # (b) polar D_total(φ) for the critical layer.
        name = result.critical_layer or (layers[0] if layers else "")
        if name:
            d = result.D_total_per_wire[name]
            phi = (
                np.radians(result.wire_azimuth_deg[name])
                if name in result.wire_azimuth_deg
                else np.linspace(0, 2 * np.pi, d.size, endpoint=False)
            )
            axb.plot(
                np.append(phi, phi[0]),
                np.append(d, d[0]),
                color=COLORS["BLUE"],
            )
        axb.set_title(rf"(b) $D_\mathrm{{total}}(\varphi)$ — {name}", fontsize=10)

        annotate_provenance(
            axa,
            result.provenance,
            loc="lower left",
            max_items=2,
            prefix=f"governing: {result.governing}; "
            f"fretting frac {result.D_fretting_fraction:.2f}",
        )
        fig.tight_layout()
    return fig


def plot_fretting_polar(result: StressResult, *, layer: str | None = None) -> Figure:
    r"""Polar fretting distribution: 2D-Mindlin (uniform) vs loxodromic (zeros).

    Visualises the **rotation-invariance theorem**: the 2D vectorial Mindlin slip
    gives a fretting amplitude **uniform** around the cross-section (blue
    circle), while the loxodromic (1D, neutral-axis-projected) model gives an
    amplitude ``∝ |cos(φ − φ_bend)|`` with **zeros at the neutral-axis wires**
    (orange dashed) — a qualitative failure the loxodromic model cannot avoid. The
    per-wire azimuths and bending plane come from the :class:`StressResult`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    name = layer or result.critical_layer or next(iter(result.phi_deg))
    phi = np.radians(result.phi_deg[name])
    phi_bend = result.phi_bend_rad

    # Both normalised to unit peak (this figure shows the SHAPE, not magnitude).
    loxo = np.abs(np.cos(phi - phi_bend))
    uniform = np.ones_like(phi)

    order = np.argsort(phi)
    phi_s = phi[order]
    with use_style():
        fig = plt.figure(figsize=FIGSIZE["square"])
        ax = fig.add_subplot(111, projection="polar")
        ax.plot(
            np.append(phi_s, phi_s[0]),
            np.append(uniform[order], uniform[order][0]),
            color=COLORS["BLUE"],
            label="2D Mindlin (uniform)",
        )
        ax.plot(
            np.append(phi_s, phi_s[0]),
            np.append(loxo[order], loxo[order][0]),
            color=COLORS["ORANGE"],
            ls="--",
            label=r"loxodromic ($|\cos(\varphi - \varphi_\mathrm{b})|$, zeros at NA)",
        )
        # mark the neutral-axis wires (loxodromic zeros)
        na = np.abs(np.abs(np.cos(phi - phi_bend))) < 0.05
        if np.any(na):
            ax.scatter(phi[na], loxo[na], c=COLORS["ORANGE"], s=30, zorder=5)
        ax.set_rlim(0, 1.15)  # type: ignore[attr-defined]  # polar axes
        ax.set_title(f"Fretting amplitude shape — {name}", fontsize=10, pad=14)
        ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.10), fontsize=8)
        annotate_provenance(ax, result.provenance, loc="lower left", max_items=1)
        fig.tight_layout()
    return fig
