# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Load a cable definition bundled with the package by its file stem."""

from __future__ import annotations

from pathlib import Path

import cablexsec as cx


def bundled(name: str) -> cx.CableData:
    """The packaged ``cablexsec/cables/<name>.yaml``, without geometry override."""
    path = Path(cx.__file__).parent / "cables" / f"{name}.yaml"
    return cx.load_cable(path, geometry_override=None)


def incomplete_cable() -> cx.CableData:
    """The published Menard cable with its inner-armour wire geometry withheld.

    Every withheld value is ``null``, as in a cable definition whose geometry
    is supplied separately; the result is not fully specified.
    """
    import tempfile

    import yaml

    path = Path(cx.__file__).parent / "cables" / "menard2023.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for layer in data["layers"]:
        if layer["name"] == "inner_armour":
            for key in ("n_wires", "d_wire_m", "r_helix_m", "lay_angle_deg"):
                layer[key] = None
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "menard2023_withheld.yaml"
        target.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        return cx.load_cable(target, geometry_override=None)
