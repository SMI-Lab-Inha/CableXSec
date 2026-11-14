# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Capacity-envelope figure from a CapacityResult tension sweep."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cablexsec.plotting.style import (
    COLORS,
    annotate_provenance,
    use_style,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from cablexsec.analysis.capacity import CapacityResult

__all__ = ["plot_capacity_envelope"]

# Mechanism -> house colour (the shared plotting convention).
_MECH_COLOR = {
    "wire_yield": COLORS["BLACK"],
    "bird_caging": COLORS["RED"],
    "polymer_crush": COLORS["BLUE"],
    "conductor": COLORS["GREEN"],
}


def plot_capacity_envelope(result: CapacityResult) -> Figure:
    r"""Two-panel capacity envelope from a :class:`CapacityResult` tension sweep.

    Panel (a): the curvature capacity ``κ_max`` vs tension — the operating
    envelope — with each segment coloured by its governing mechanism. Panel (b):
    the tension-driven usage factors (bird-caging, polymer crush, conductor) vs
    tension, with the UF = 1.0 (thick black) and UF = 0.8 (green dashed)
    reference lines. The result's limitations are annotated verbatim.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    t_kN = result.tension_N / 1e3
    with use_style():
        fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.0, 4.0))

        # (a) curvature capacity envelope, coloured by governing mechanism.
        axa.plot(
            t_kN,
            result.kappa_max_1_m,
            color=COLORS["BLACK"],
            lw=1.6,
            zorder=2,
            label="capacity boundary",
        )
        gov = np.asarray(result.governing)
        for mech, color in _MECH_COLOR.items():
            sel = gov == mech
            if np.any(sel):
                axa.scatter(
                    t_kN[sel],
                    result.kappa_max_1_m[sel],
                    c=color,
                    s=16,
                    label=mech.replace("_", " "),
                    zorder=3,
                )
        axa.set_xlabel("tension [kN]")
        axa.set_ylabel(r"curvature capacity $\kappa_{max}$ [1/m]")
        axa.set_title("(a) capacity envelope")
        axa.legend(loc="best", fontsize=8)

        # (b) tension-driven usage factors.
        axb.plot(
            t_kN,
            result.uf_bird_caging,
            color=COLORS["RED"],
            lw=1.3,
            label="bird-caging",
        )
        if np.any(np.isfinite(result.uf_polymer_crush)):
            axb.plot(
                t_kN,
                result.uf_polymer_crush,
                color=COLORS["BLUE"],
                lw=1.3,
                label="polymer crush",
            )
        else:
            axb.text(
                0.98,
                0.04,
                "polymer crush unavailable\nno sourced project allowable",
                transform=axb.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
                color=COLORS["BLUE"],
            )
        axb.plot(
            t_kN, result.uf_conductor, color=COLORS["GREEN"], lw=1.3, label="conductor"
        )
        axb.axhline(1.0, color=COLORS["BLACK"], lw=1.6, label="UF = 1.0")
        axb.axhline(0.8, color=COLORS["GREEN"], ls="--", lw=0.9, label="UF = 0.8")
        axb.set_xlabel("tension [kN]")
        axb.set_ylabel("usage factor [-]")
        axb.set_title("(b) tension-driven usage factors")
        axb.legend(loc="best", fontsize=8)

        annotate_provenance(axb, result.provenance, loc="lower right", max_items=2)
        fig.tight_layout()
    return fig
