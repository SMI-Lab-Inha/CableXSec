# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Moment-curvature figures: single loop and the validation suite."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from cablexsec.plotting.style import (
    COLORS,
    FIGSIZE,
    LABELS,
    annotate_provenance,
    use_style,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from cablexsec.api import MkappaResult

__all__ = ["plot_mkappa", "plot_mkappa_validation_suite"]

FloatArray = NDArray[np.float64]


def plot_mkappa(
    result: MkappaResult,
    *,
    exp_kappa: FloatArray | None = None,
    exp_moment_kNm: FloatArray | None = None,
    title: str = "",
    ax: Axes | None = None,
    annotate: bool = True,
) -> Axes:
    """Plot a model M-κ loop (public ``MkappaResult``), optionally vs experiment.

    Model loop in the first MATLAB colour; grey EI_max (dotted) / EI_min
    (dashed) reference slopes; black open circles for experiment. The
    result's limitations are annotated verbatim (e.g. the Papailiou-vs-Coulomb
    engine caveat).

    Parameters
    ----------
    result : MkappaResult
        From :func:`cablexsec.compute_mkappa` (``kappa_1_m``/``moment_Nm``).
    exp_kappa, exp_moment_kNm : numpy.ndarray, optional
        Experimental curvature [1/m] and moment [kN·m].
    title : str, optional
    ax : matplotlib.axes.Axes, optional
    annotate : bool, optional
        Draw the provenance caveat box (default True).

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    with use_style():
        if ax is None:
            _, ax = plt.subplots(figsize=FIGSIZE["single"])

        kappa = result.kappa_1_m
        moment_kNm = result.moment_Nm / 1.0e3

        ax.plot(
            kappa,
            moment_kNm,
            color=COLORS["BLUE"],
            label="CableXSec model",
            zorder=3,
        )

        k_line = np.array([0.0, float(np.max(np.abs(kappa))) or 1.0])
        ax.plot(
            k_line,
            result.EI_max_Nm2 * k_line / 1e3,
            color=COLORS["GREY"],
            ls=":",
            lw=1.0,
            label=r"$EI_\mathrm{max}$ (stuck)",
        )
        ax.plot(
            k_line,
            result.EI_min_Nm2 * k_line / 1e3,
            color=COLORS["GREY"],
            ls="--",
            lw=1.0,
            label=r"$EI_\mathrm{min}$ (full slip)",
        )

        if exp_kappa is not None and exp_moment_kNm is not None:
            ax.plot(
                exp_kappa,
                exp_moment_kNm,
                ls="none",
                marker="o",
                ms=3.0,
                mfc="none",
                mec=COLORS["BLACK"],
                mew=0.7,
                label="Experiment",
                zorder=2,
            )

        m_peak = float(np.max(np.abs(moment_kNm))) or 1.0
        if exp_moment_kNm is not None and exp_moment_kNm.size:
            m_peak = max(m_peak, float(np.max(np.abs(exp_moment_kNm))))
        ax.set_ylim(-1.3 * m_peak, 1.3 * m_peak)

        ax.axhline(0.0, color=COLORS["GREY"], lw=0.5)
        ax.axvline(0.0, color=COLORS["GREY"], lw=0.5)
        ax.set_xlabel(LABELS["curvature"])
        ax.set_ylabel(LABELS["moment_kNm"])
        if title:
            ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)

        txt = (
            rf"$W_\mathrm{{diss}}$ = {result.W_diss_J_per_m:.0f} J/m" + "\n"
            rf"$EI_\mathrm{{app}}$ = {result.EI_apparent_Nm2 / 1e3:.0f} "
            r"kN$\cdot$m$^\mathrm{2}$"
        )
        ax.text(
            0.97,
            0.03,
            txt,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={
                "facecolor": "white",
                "edgecolor": COLORS["GREY"],
                "boxstyle": "square,pad=0.3",
            },
        )
        if annotate:
            annotate_provenance(ax, result.provenance, loc="upper right", max_items=2)
    return ax


def plot_mkappa_validation_suite(
    results: dict[str, MkappaResult],
    *,
    experiments: dict[str, tuple[FloatArray, FloatArray]] | None = None,
    suptitle: str = r"$M$–$\kappa$ validation suite",
) -> Figure:
    """Grid of M-κ loops (one per benchmark) with an error/summary table.

    Parameters
    ----------
    results : dict[str, MkappaResult]
        Benchmark name -> model result.
    experiments : dict[str, (kappa, moment_kNm)], optional
        Benchmark name -> experimental (curvature [1/m], moment [kN·m]).
    suptitle : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    experiments = experiments or {}
    names = list(results)
    n = len(names)
    ncols = min(3, max(1, n))
    nrows = (n + ncols - 1) // ncols

    with use_style():
        fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE["suite"], squeeze=False)
        flat = axes.ravel()
        for j, name in enumerate(names):
            exp = experiments.get(name)
            plot_mkappa(
                results[name],
                exp_kappa=exp[0] if exp else None,
                exp_moment_kNm=exp[1] if exp else None,
                title=name,
                ax=flat[j],
                annotate=False,
            )
        for j in range(n, len(flat)):
            flat[j].axis("off")
        fig.suptitle(suptitle, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig
