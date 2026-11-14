# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""An interstitial filler lies beside the cores, not in the armour's shear path."""

from __future__ import annotations

from pathlib import Path

import pytest

import cablexsec as cx
from cablexsec.cables import schema
from cablexsec.cables.schema import load_cable
from cablexsec.mechanics.section_geometry import LAID_CORE_OVERLAP, filler_area
from cablexsec.mechanics.stiffness import xi_for_layer

FOLDER = Path(schema.__file__).parent


def test_filler_leaves_the_shear_coefficient_and_stick_limit_unchanged() -> None:
    plain = load_cable(FOLDER / "menard2023.yaml", geometry_override=None)
    filled = load_cable(FOLDER / "menard2023_filled.yaml", geometry_override=None)
    for layer in plain.helical_layers:
        assert xi_for_layer(filled, layer.name) == xi_for_layer(plain, layer.name)
    assert cx.compute_stiffness(filled).ei_max_Nm2 == pytest.approx(
        cx.compute_stiffness(plain).ei_max_Nm2, rel=1e-12
    )
    # The filler still carries axial load over its true area.
    assert (
        cx.compute_stiffness(filled).ea_free_N > cx.compute_stiffness(plain).ea_free_N
    )


def test_touching_laid_cores_give_a_filler_area() -> None:
    filled = load_cable(FOLDER / "menard2023_filled.yaml", geometry_override=None)
    filler = filled.get_layer("pp_filler")
    area = filler_area(filler, filled)  # type: ignore[arg-type]
    assert 0.0 < area < 1.2e-3 and LAID_CORE_OVERLAP == 0.005
