# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Screen resistance of the sheath loss: wires over their lay, foil in parallel."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

import cablexsec as cx
from cablexsec.materials.library import get_property
from cablexsec.mechanics.thermal import sheath_circulating_loss_factor, sheath_reactance

SOURCE = Path(cx.__file__).parent / "cables" / "thies2016_30kv.yaml"
CURRENT_A, AREA_M2 = 400.0, 240e-6
ENV = {"T4_external_Km_W": 0.5, "ambient_C": 15.0}

pytestmark = pytest.mark.analytical_sanity


def _layer(data: dict, name: str) -> dict:
    return next(x for x in data["layers"] if x["name"] == name)


def _load(tmp_path: Path, data: dict) -> cx.CableData:
    target = tmp_path / "thies.yaml"
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return cx.load_cable(target, geometry_override=None)


def _lambda1_by_hand(res, data: dict, names: list[str]) -> float:
    """lambda1' of [IEC60287-1-1] 5.3.11 with R_s of the named screens in parallel."""
    losses = res.losses
    t_s = _sheath_temperature(res)
    conductance, diameters = 0.0, []
    for name in names:
        layer = _layer(data, name)
        rho = get_property(layer["material"], "rho_ohm_m")
        alpha = get_property(layer["material"], "alpha_R_per_K")
        if "strand_layers" in layer:
            area = 0.0
            for ring in layer["strand_layers"]:
                lay = math.atan2(2 * math.pi * ring["r_helix_m"], ring["pitch_m"])
                area += (
                    ring["n_wires"]
                    * math.pi
                    * ring["d_wire_m"] ** 2
                    / 4
                    * math.cos(lay)
                )
        else:
            area = math.pi * (layer["r_outer_m"] ** 2 - layer["r_inner_m"] ** 2)
        conductance += area / (rho * (1 + alpha * (t_s - 20.0)))
        diameters.append(layer["r_inner_m"] + layer["r_outer_m"])
    c_ll = losses["lay_length_factor"]
    r_s = c_ll / conductance
    spacing = 2 * _layer(data, "cu_screen_wires")["r_offset_m"] * math.sin(math.pi / 3)
    x = sheath_reactance(
        spacing, sum(diameters) / len(diameters), lay_length_factor=c_ll
    )
    return sheath_circulating_loss_factor(
        r_s, losses["R_ac_ohm_m"], x, sl_armoured=True
    )


def _sheath_temperature(res) -> float:
    """Sheath temperature the ladder reports in its lambda1 note [degC]."""
    note = next(n for n in res.provenance.limitations if n.startswith("lambda1'"))
    return float(note.split(" at ")[1].split(" degC")[0])


def test_wire_screen_and_laminate_in_parallel_tape_left_out(tmp_path: Path) -> None:
    # 2026-11-28 (PR #66 review): R_s was the envelope annulus of the wires,
    # 72.8 mm2 instead of their 15.0 mm2. Now the 53 wires over their lay and
    # the bonded Al laminate in parallel (CIGRE TB 880 Case 2), the counter
    # helix Cu tape left out (TB 880 Guidance Point 29).
    data = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    res = cx.compute_thermal(_load(tmp_path, data), CURRENT_A, AREA_M2, **ENV)
    note = next(n for n in res.provenance.limitations if n.startswith("lambda1'"))
    assert "53 wires of 14.99 mm2" in note
    assert "in parallel" in note and "cu_tape not counted" in note
    expected = _lambda1_by_hand(res, data, ["cu_screen_wires", "al_laminate"])
    assert res.losses["lambda1"] == pytest.approx(expected, rel=1e-3)


def test_wire_screen_alone_uses_its_wires(tmp_path: Path) -> None:
    data = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    _layer(data, "al_laminate")["material"] = "hdpe"  # a polymer, not a screen
    res = cx.compute_thermal(_load(tmp_path, data), CURRENT_A, AREA_M2, **ENV)
    expected = _lambda1_by_hand(res, data, ["cu_screen_wires"])
    assert res.losses["lambda1"] == pytest.approx(expected, rel=1e-3)
    # The envelope annulus would give a resistance about five times lower.
    wires = _layer(data, "cu_screen_wires")
    envelope = math.pi * (wires["r_outer_m"] ** 2 - wires["r_inner_m"] ** 2)
    assert envelope / 14.99e-6 > 4.5
