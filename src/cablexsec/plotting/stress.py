# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Per-wire stress figures: 6-panel component distribution and time series."""

from __future__ import annotations

from textwrap import fill
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.colors import Normalize
from numpy.typing import NDArray

from cablexsec.plotting.style import (
    COLORS,
    FIGSIZE,
    LABELS,
    SIGNED_CMAP,
    annotate_provenance,
    signed_norm,
    use_style,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from cablexsec.analysis.stress import StressResult
    from cablexsec.io.series import LoadCaseSeries

__all__ = ["plot_stress_distribution", "plot_stress_timeseries"]

FloatArray = NDArray[np.float64]

# (field, label, signed?) for the six panels, in reading order.
_PANELS = [
    ("sigma_A_Pa", r"$\sigma_\mathrm{A}$ (tension)", True),
    ("sigma_B_Pa", r"$\sigma_\mathrm{B}$ (bending)", True),
    ("sigma_f_Pa", r"$\sigma_\mathrm{f}$ (friction)", True),
    ("sigma_T_Pa", r"$\sigma_\mathrm{T}$ (thermal)", True),
    ("sigma_total_Pa", r"$\sigma_\mathrm{total}$", True),
    ("sigma_vm_Pa", "Axial/contact indicator", False),
]


def plot_stress_distribution(result: StressResult) -> Figure:
    r"""Six-panel per-wire stress distribution vs azimuth (one ``StressResult``).

    Each panel scatters every wire at its azimuth ``φ`` against the component
    value [MPa]; **signed** components (σ_A/σ_B/σ_f/σ_T/σ_total) are coloured with
    the blue-white-red diverging map, the **unsigned** σ_vm with a sequential
    map. The critical wire carries a thick black ring; **slipping** wires are
    drawn as open markers. The result's limitations are annotated verbatim.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    with use_style():
        limitations = result.provenance.limitations if result.provenance else ()
        notes = [
            fill(f"• {text}", width=90, break_long_words=False, break_on_hyphens=False)
            for text in limitations[:2]
        ]
        if len(limitations) > 2:
            notes.append(f"• (+{len(limitations) - 2} more — see provenance)")
        footer = "\n".join(notes)
        # Reserve physical space for the caveats instead of covering stress data.
        # Conservative line width fits the house font at seven points in nine inches.
        footer_height = (len(footer.splitlines()) * 10 + 24) / 72 if footer else 0.0
        width, panel_height = FIGSIZE["six_panel"]
        height = panel_height + footer_height
        fig, axes = plt.subplots(2, 3, figsize=(width, height), squeeze=False)
        flat = axes.ravel()
        layers = list(result.phi_deg)
        for p, (field, label, signed) in enumerate(_PANELS):
            ax = flat[p]
            _panel(ax, result, field, label, signed, layers)
        if footer:
            note = fig.text(
                0.04,
                0.12 / height,
                footer,
                fontsize=7,
                color=COLORS["GREY"],
                ha="left",
                va="bottom",
            )
            note.set_in_layout(False)
        fig.suptitle(
            f"Per-wire stress — {result.critical_layer} critical at "
            f"φ={result.critical_phi_deg:.0f}°, "
            f"screening peak={result.critical_sigma_vm_Pa / 1e6:.0f} MPa",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, footer_height / height, 1, 0.96))
    return fig


def _panel(
    ax: Axes,
    result: StressResult,
    field: str,
    label: str,
    signed: bool,
    layers: list[str],
) -> None:
    """Draw one stress-component panel."""
    import matplotlib.pyplot as plt

    data = getattr(result, field)
    allv = np.concatenate([data[n] for n in layers]) if layers else np.zeros(1)
    if signed:
        norm = signed_norm(float(np.max(np.abs(allv))) if allv.size else 1.0)
        cmap = SIGNED_CMAP
    else:
        vmax = float(np.max(allv)) or 1.0
        norm = Normalize(vmin=0.0, vmax=vmax)
        cmap = plt.get_cmap("Reds")

    for name in layers:
        phi = result.phi_deg[name]
        val = data[name]
        slip = result.slipping[name]
        # stuck = filled by value (diverging/sequential); slipping = open marker.
        ax.scatter(
            phi[~slip],
            val[~slip] / 1e6,
            c=val[~slip],
            cmap=cmap,
            norm=norm,
            edgecolors=COLORS["BLACK"],
            linewidths=0.3,
            s=22,
            zorder=3,
        )
        ax.scatter(
            phi[slip],
            val[slip] / 1e6,
            facecolors="none",
            edgecolors=COLORS["BLACK"],
            linewidths=0.6,
            s=22,
            zorder=3,
        )
        # critical wire ring
        if name == result.critical_layer:
            kc = int(np.argmin(np.abs(phi - result.critical_phi_deg)))
            ax.scatter(
                phi[kc],
                val[kc] / 1e6,
                facecolors="none",
                edgecolors=COLORS["BLACK"],
                linewidths=1.6,
                s=70,
                zorder=4,
            )
    if signed:
        ax.axhline(0.0, color=COLORS["GREY"], lw=0.5)
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel(LABELS["azimuth"], fontsize=9)
    ax.set_ylabel(f"{label} [MPa]", fontsize=9)


def plot_stress_timeseries(
    series: LoadCaseSeries,
    *,
    sigma_crit_Pa: FloatArray | None = None,
    delta_star_um: FloatArray | None = None,
    sigma_y_Pa: float | None = None,
) -> Figure:
    r"""Stacked time-series panels driven by a :class:`LoadCaseSeries`.

    Panel 1: curvature κ_x (black), κ_y (red). Panel 2: critical-wire σ_total
    with ±σ_y bands if given. Panel 3: fretting amplitude δ* with the 3 µm Cu /
    5 µm steel thresholds. Panel 4: effective tension. Optional overlays
    (``sigma_crit_Pa``, ``delta_star_um``) are caller-supplied result-derived
    arrays; the series' own limitations are annotated.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    have_sig = sigma_crit_Pa is not None
    have_del = delta_star_um is not None
    n_panels = 2 + int(have_sig) + int(have_del)
    t = series.time_s

    with use_style():
        fig, axes = plt.subplots(
            n_panels, 1, figsize=FIGSIZE["wide"], sharex=True, squeeze=False
        )
        col = axes[:, 0]
        i = 0
        col[i].plot(
            t,
            series.curvature_x_1_m,
            color=COLORS["BLUE"],
            lw=1.0,
            label=r"$\kappa_x$",
        )
        if np.any(series.curvature_y_1_m):
            col[i].plot(
                t,
                series.curvature_y_1_m,
                color=COLORS["ORANGE"],
                lw=1.0,
                label=r"$\kappa_y$",
            )
        col[i].set_ylabel(r"$\kappa$ [m$^{\mathrm{-1}}$]", fontsize=9)
        col[i].legend(loc="upper right", fontsize=8)
        i += 1

        if have_sig:
            assert sigma_crit_Pa is not None
            col[i].plot(t, sigma_crit_Pa / 1e6, color=COLORS["BLUE"], lw=1.0)
            if sigma_y_Pa is not None:
                for s in (sigma_y_Pa, -sigma_y_Pa):
                    col[i].axhline(s / 1e6, color=COLORS["RED"], ls="--", lw=0.8)
            col[i].set_ylabel(r"$\sigma_\mathrm{total}$ [MPa]", fontsize=9)
            i += 1

        if have_del:
            assert delta_star_um is not None
            col[i].plot(t, delta_star_um, color=COLORS["BLUE"], lw=1.0)
            col[i].axhline(
                3.0, color=COLORS["GREEN"], ls="--", lw=0.8, label="3 µm (Cu)"
            )
            col[i].axhline(
                5.0, color=COLORS["RED"], ls="--", lw=0.8, label="5 µm (steel)"
            )
            col[i].set_ylabel(r"$\delta^*$ [µm]", fontsize=9)
            col[i].legend(loc="upper right", fontsize=8)
            i += 1

        col[i].plot(t, series.tension_N / 1e3, color=COLORS["PURPLE"], lw=1.0)
        col[i].set_ylabel(r"$T$ [kN]", fontsize=9)
        col[-1].set_xlabel(LABELS["time"])
        annotate_provenance(col[0], series.provenance, loc="upper left", max_items=2)
        fig.tight_layout()
    return fig
