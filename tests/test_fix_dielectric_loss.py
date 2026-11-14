# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""At 100 kV or more the thermal ladder includes the dielectric loss; missing data are an error."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

import cablexsec as cx
from cablexsec.exceptions import CableInputError

SOURCE = Path(cx.__file__).parent / "cables" / "cigre_tb880_220kv.yaml"
CURRENT_A, AREA_M2 = 908.45, 1200e-6


def _variant(
    tmp_path: Path, *, drop_electrical: bool, voltage_kV: float | None
) -> Path:
    data = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    if drop_electrical:
        for layer in data["layers"]:
            layer.pop("electrical", None)
    if voltage_kV is not None:
        data["cable"]["voltage_kV"] = voltage_kV
    target = tmp_path / "variant.yaml"
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return target


def test_high_voltage_cable_without_dielectric_data_is_refused(tmp_path: Path) -> None:
    path = _variant(tmp_path, drop_electrical=True, voltage_kV=None)
    cable = cx.load_cable(path, geometry_override=None)
    with pytest.raises(CableInputError, match="dielectric loss"):
        cx.compute_thermal(cable, CURRENT_A, AREA_M2)


def test_dielectric_loss_raises_the_conductor_temperature(tmp_path: Path) -> None:
    with_loss = cx.compute_thermal(
        cx.load_cable(SOURCE, geometry_override=None), CURRENT_A, AREA_M2
    )
    below = cx.load_cable(
        _variant(tmp_path, drop_electrical=False, voltage_kV=66.0),
        geometry_override=None,
    )
    without_loss = cx.compute_thermal(below, CURRENT_A, AREA_M2)
    assert math.isfinite(with_loss.T_conductor_C)
    assert with_loss.T_conductor_C > without_loss.T_conductor_C
