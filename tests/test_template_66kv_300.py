# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""66 kV three-core 300 mm2 section derived from the public 630 mm2 template.

The IEA Wind Task 49 design basis (Hall et al. 2024, Table 13, p. 35) publishes
aggregate properties only for its 300 mm2 row. The derived section keeps the
layer thicknesses, lay angles, materials and armour coverage of the Janocha et al.
(2024) template and scales the conductor; these tests pin the derivation and
record its agreement with Table 13 (outer diameter and axial stiffness close,
slipping bending stiffness low, as for the 630 mm2 template).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.cables.schema import CableData, load_cable
from cablexsec.mechanics.stiffness import ea_fixed

# [IEATask49] Table 13, 300 mm2 row.
OD_TABLE_M = 0.161
EA_TABLE_N = 469.6e6
EI_SLIP_TABLE_NM2 = 19.92e3


@pytest.fixture(scope="module")
def cable() -> CableData:
    return load_cable(cx.iea_task49_66kv_300_yaml_path(), geometry_override=None)


@pytest.fixture(scope="module")
def template() -> CableData:
    return load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)


def _layer(cab: CableData, name: str):  # type: ignore[no-untyped-def]
    return next(layer for layer in cab.layers if layer.name == name)


def test_conductor_is_300_mm2(cable) -> None:
    cu = _layer(cable, "cu_conductor")
    assert math.pi * cu.r_outer_m**2 == pytest.approx(300e-6, rel=1e-12)


@pytest.mark.parametrize(
    ("inner", "outer"),
    [
        ("conductor_screen", "xlpe_insulation"),
        ("xlpe_insulation", "insulation_screen"),
        ("insulation_screen", "cu_tape_screen"),
        ("cu_tape_screen", "mdpe_core_sheath"),
        ("ppy_bedding_inner", "mdpe_inner_sheath"),
    ],
)
def test_radial_thicknesses_kept_from_the_template(
    cable, template, inner, outer
) -> None:
    def thickness(cab: CableData) -> float:
        return _layer(cab, outer).r_outer_m - _layer(cab, inner).r_outer_m

    assert thickness(cable) == pytest.approx(thickness(template), abs=1e-7)


def test_armour_coverage_and_lay_kept(cable, template) -> None:
    for name in ("inner_armour", "outer_armour"):
        new, old = _layer(cable, name), _layer(template, name)

        def cover(layer) -> float:  # type: ignore[no-untyped-def]
            return (
                layer.n_wires
                * layer.d_wire_m
                / (
                    2
                    * math.pi
                    * layer.r_helix_m
                    * math.cos(math.radians(layer.lay_angle_deg))
                )
            )

        assert new.lay_angle_deg == old.lay_angle_deg
        assert new.d_wire_m == old.d_wire_m
        assert abs(cover(new) - cover(old)) < new.d_wire_m / (
            2 * math.pi * new.r_helix_m
        )
    assert (
        _layer(cable, "inner_armour").n_wires,
        _layer(cable, "outer_armour").n_wires,
    ) == (122, 128)


def test_outer_diameter_and_axial_stiffness_against_table_13(cable) -> None:
    od = 2 * _layer(cable, "hdpe_outer_sheath").r_outer_m
    assert od / OD_TABLE_M - 1 == pytest.approx(0.0187, abs=0.002)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ea = ea_fixed(cable)
    assert abs(ea / EA_TABLE_N - 1) < 0.03


def test_slipping_bending_stiffness_is_recorded_low(cable) -> None:
    """Full-slip EI at 50 kN is 14.5 kN m2 against 19.92 (630 mm2 template: 21.7 against 42.47)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = cx.compute_mkappa(cable, np.linspace(0.0, 0.15, 200), tension_N=50e3)
    assert res.EI_min_Nm2 == pytest.approx(14.47e3, rel=0.01)
    assert res.EI_min_Nm2 < EI_SLIP_TABLE_NM2
