# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Stick coefficient of two armoured verification tubes against a line-contact FE model (row XI-FE).

Model to model. The reference is the sectional stick coefficient of a three-dimensional line-contact
finite-element model of each tube (``validation/paper_data/line_contact_tube_stick.csv``; the header of
that file states the model and how the value is extracted). CableXSec's ξ is evaluated with both stick
supports of :func:`~cablexsec.mechanics.stiffness.xi_for_layer`: the default series stack
(``"yun_stack"``) and the opt-in ``"annulus"``. The finite-element wire section is an octagon inscribed
in the round wire, so the comparison uses the meshed wire area: the wire modulus is scaled by the area
factor, which scales E A_w and leaves the support unchanged.

Usage: ``python validation/scripts/check_stick_support_fe.py``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cablexsec as cx
from cablexsec.materials import library
from cablexsec.mechanics.stiffness import xi_for_layer

DATA = Path(__file__).resolve().parents[1] / "paper_data"
ARMOUR = "armour"


def _cases() -> list[dict[str, str]]:
    with (DATA / "line_contact_tube_stick.csv").open(encoding="utf-8") as handle:
        rows = [line for line in handle if not line.startswith("#")]
    return list(csv.DictReader(rows))


def stick_comparison() -> dict[str, dict[str, float]]:
    """ξ of the finite-element model and of both stick supports, per tube."""
    out: dict[str, dict[str, float]] = {}
    for case in _cases():
        cable = cx.load_cable(DATA / case["yaml"], geometry_override=None)
        with cable.material_scope():
            moduli = {
                layer.name: library.get_property(layer.material, "E_Pa")
                for layer in cable.layers
            }
            moduli[ARMOUR] *= float(case["area_factor"])
            values = {
                model: xi_for_layer(
                    cable, ARMOUR, youngs_modulus_Pa=moduli, stick_support=model
                )
                for model in ("yun_stack", "annulus")
            }
        xi_fe = float(case["xi_fe"])
        out[case["case"]] = {
            "xi_fe": xi_fe,
            "yun_stack": values["yun_stack"],
            "annulus": values["annulus"],
            "yun_stack_error": values["yun_stack"] / xi_fe - 1.0,
            "annulus_error": values["annulus"] / xi_fe - 1.0,
        }
    return out


def main() -> None:
    for name, row in stick_comparison().items():
        print(
            f"{name:8s} FE {row['xi_fe']:.4f}  annulus {row['annulus']:.4f} "
            f"({100 * row['annulus_error']:+.1f} %)  Yun stack {row['yun_stack']:.4f} "
            f"({100 * row['yun_stack_error']:+.1f} %)"
        )


if __name__ == "__main__":
    main()
