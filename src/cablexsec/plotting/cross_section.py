# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Cable cross-section figure — geometry to scale, optionally stress-coloured."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from cablexsec.cables.schema import (
    CylindricalLayer,
    CylindricalOffsetLayer,
    CylindricalShellLayer,
    HelicalLayer,
)
from cablexsec.plotting.style import (
    COLORS,
    FIGSIZE,
    SIGNED_CMAP,
    annotate_provenance,
    material_color,
    signed_norm,
    use_style,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from cablexsec.analysis.stress import StressResult
    from cablexsec.cables.schema import CableData

__all__ = ["plot_cross_section"]


def plot_cross_section(
    cable: CableData,
    stress_result: StressResult | None = None,
    *,
    component: str = "sigma_total_Pa",
    ax: Axes | None = None,
    title: str = "",
) -> Axes:
    r"""Draw the cable cross-section to scale.

    Layers are drawn concentric/offset to scale. Without ``stress_result`` wires
    and rings are coloured by **material**; with it, helical wires are coloured by
    the signed per-wire ``component`` (blue-white-red diverging), **slipping**
    wires drawn as open circles and **stuck** wires filled. The source result's
    limitations are annotated verbatim.

    Parameters
    ----------
    cable : CableData
        The cable to draw (any fully- or partially-specified geometry).
    stress_result : StressResult, optional
        Per-wire stress to colour the helical wires by.
    component : str, optional
        Which signed per-wire field of ``stress_result`` to colour by.
    ax : matplotlib.axes.Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Annulus, Circle

    with use_style():
        if ax is None:
            _, ax = plt.subplots(figsize=FIGSIZE["square"])

        norm = sm = None
        if stress_result is not None:
            allvals = np.concatenate(
                [np.abs(v) for v in getattr(stress_result, component).values()]
            )
            norm = signed_norm(float(np.max(allvals)) if allvals.size else 1.0)
            sm = plt.cm.ScalarMappable(norm=norm, cmap=SIGNED_CMAP)

        r_max = 0.0
        for lyr in cable.layers:
            r_max = max(r_max, _draw_layer(ax, lyr, stress_result, component, norm))

        if sm is not None:
            from matplotlib.ticker import FuncFormatter

            cbar = ax.figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(_component_label(component), fontsize=9)
            cbar.ax.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _pos: f"{v / 1e6:.0f}")
            )

        lim = 1.08 * (r_max or 1.0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(title or f"{cable.cable.name} — cross-section")
        # Legend for the marker convention only when stress-coloured.
        if stress_result is not None:
            ax.scatter(
                [], [], facecolors=COLORS["GREY"], edgecolors="k", label="stuck wire"
            )
            ax.scatter([], [], facecolors="none", edgecolors="k", label="slipping wire")
            ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.9)
            annotate_provenance(ax, stress_result.provenance, loc="lower left")
        # keep the unused import referenced for static checkers
        _ = (Circle, Annulus)
    return ax


def _draw_layer(
    ax: Axes,
    lyr: object,
    stress: StressResult | None,
    component: str,
    norm: object,
) -> float:
    """Draw one layer; return the outer radius it reaches [m]."""
    from matplotlib.patches import Annulus, Circle

    if isinstance(lyr, HelicalLayer):
        if None in (lyr.n_wires, lyr.r_helix_m, lyr.d_wire_m):
            return 0.0
        assert lyr.n_wires and lyr.r_helix_m and lyr.d_wire_m
        r_w = lyr.d_wire_m / 2.0
        colored = stress is not None and lyr.name in getattr(stress, component)
        if colored:
            assert stress is not None
            phis = np.radians(stress.phi_deg[lyr.name])
            vals = getattr(stress, component)[lyr.name]
            slip = stress.slipping[lyr.name]
        else:
            phis = 2.0 * math.pi * np.arange(lyr.n_wires) / lyr.n_wires
            vals = np.zeros(lyr.n_wires)
            slip = np.zeros(lyr.n_wires, dtype=bool)
        for k, phi in enumerate(phis):
            cx, cy = lyr.r_helix_m * math.cos(phi), lyr.r_helix_m * math.sin(phi)
            face: Any
            if colored and norm is not None:
                face = SIGNED_CMAP(norm(vals[k]))  # type: ignore[operator]
            else:
                face = material_color(lyr.material)
            ax.add_patch(
                Circle(
                    (cx, cy),
                    r_w,
                    facecolor="none" if (colored and slip[k]) else face,
                    edgecolor=COLORS["BLACK"],
                    linewidth=0.5,
                    zorder=4,
                )
            )
        return lyr.r_helix_m + r_w

    if isinstance(lyr, CylindricalShellLayer):
        if None in (lyr.r_mid_m, lyr.t_m):
            return 0.0
        assert lyr.r_mid_m and lyr.t_m
        ax.add_patch(
            Annulus(
                (0, 0),
                lyr.r_mid_m + lyr.t_m / 2,
                lyr.t_m,
                facecolor=material_color(lyr.material),
                edgecolor=COLORS["GREY"],
                linewidth=0.4,
                zorder=2,
            )
        )
        return lyr.r_mid_m + lyr.t_m / 2

    if isinstance(lyr, CylindricalOffsetLayer):
        if None in (lyr.r_offset_m, lyr.r_outer_m):
            return 0.0
        assert lyr.r_offset_m is not None and lyr.r_outer_m is not None
        for k in range(lyr.n_cores):
            ang = math.pi / 2 + 2 * math.pi * k / lyr.n_cores
            cx, cy = lyr.r_offset_m * math.cos(ang), lyr.r_offset_m * math.sin(ang)
            r_in = lyr.r_inner_m or 0.0
            # A ring, not a disc with a white hole: a hole drawn in white would
            # cover the inner layers of the same core drawn before it.
            patch = (
                Annulus((cx, cy), lyr.r_outer_m, lyr.r_outer_m - r_in)
                if r_in > 0.0
                else Circle((cx, cy), lyr.r_outer_m)
            )
            patch.set(
                facecolor=material_color(lyr.material),
                edgecolor=COLORS["GREY"],
                linewidth=0.4,
                zorder=2,
            )
            ax.add_patch(patch)
        return lyr.r_offset_m + lyr.r_outer_m

    if isinstance(lyr, CylindricalLayer):
        if None in (lyr.r_inner_m, lyr.r_outer_m):
            return 0.0
        assert lyr.r_inner_m is not None and lyr.r_outer_m is not None
        width = lyr.r_outer_m - lyr.r_inner_m
        ax.add_patch(
            Annulus(
                (0, 0),
                lyr.r_outer_m,
                max(width, 1e-6),
                facecolor=material_color(lyr.material),
                edgecolor=COLORS["GREY"],
                linewidth=0.3,
                zorder=1,
                alpha=0.9,
            )
        )
        return lyr.r_outer_m

    return 0.0


def _component_label(component: str) -> str:
    """Pretty label for a stress component field name."""
    names = {
        "sigma_total_Pa": r"$\sigma_{total}$ [MPa]",
        "sigma_A_Pa": r"$\sigma_A$ [MPa]",
        "sigma_B_Pa": r"$\sigma_B$ [MPa]",
        "sigma_f_Pa": r"$\sigma_f$ [MPa]",
        "sigma_vm_Pa": "Axial/contact indicator [MPa]",
    }
    return names.get(component, component)
