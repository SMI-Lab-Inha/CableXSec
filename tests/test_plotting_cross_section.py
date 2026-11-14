# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Cross-section drawing: the axes span the outermost drawn layer."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

import cablexsec as cx  # noqa: E402
from cablexsec.cables.schema import (  # noqa: E402
    CylindricalLayer,
    CylindricalOffsetLayer,
    CylindricalShellLayer,
    HelicalLayer,
)
from cablexsec.plotting.cross_section import plot_cross_section  # noqa: E402


def _outer_radius(layer: object) -> float:
    if isinstance(layer, HelicalLayer):
        return layer.r_helix_m + layer.wire_radial_thickness_m / 2
    if isinstance(layer, CylindricalShellLayer):
        return layer.r_mid_m + layer.t_m / 2
    if isinstance(layer, CylindricalOffsetLayer):
        return layer.r_offset_m + layer.r_outer_m
    if isinstance(layer, CylindricalLayer):
        return layer.r_outer_m
    return 0.0


@pytest.mark.parametrize(
    "path", [cx.cigre_tb880_yaml_path(), cx.export_three_core_yaml_path()]
)
def test_axes_span_the_outermost_layer_with_shells_and_offsets(path):
    cable = cx.load_cable(path, geometry_override=None)
    ax = plot_cross_section(cable)
    try:
        r_max = max(_outer_radius(layer) for layer in cable.layers)
        assert ax.get_xlim() == pytest.approx((-1.08 * r_max, 1.08 * r_max))
        assert ax.get_ylim() == pytest.approx((-1.08 * r_max, 1.08 * r_max))
        assert ax.get_aspect() == 1.0
    finally:
        plt.close(ax.figure)


def test_layers_without_geometry_are_not_drawn():
    cable = cx.load_cable(cx.export_three_core_yaml_path(), geometry_override=None)
    blank = []
    for layer in cable.layers:
        if isinstance(layer, HelicalLayer):
            layer = layer.model_copy(update={"r_helix_m": None})
        elif isinstance(layer, CylindricalShellLayer):
            layer = layer.model_copy(update={"t_m": None})
        elif isinstance(layer, CylindricalOffsetLayer):
            layer = layer.model_copy(update={"r_offset_m": None})
        elif isinstance(layer, CylindricalLayer):
            layer = layer.model_copy(update={"r_outer_m": None})
        blank.append(layer)
    ax = plot_cross_section(cable.model_copy(update={"layers": tuple(blank)}))
    try:
        # Nothing drawn: no patch, and the unit default extent.
        assert len(ax.patches) == 0
        assert ax.get_xlim() == pytest.approx((-1.08, 1.08))
    finally:
        plt.close(ax.figure)
