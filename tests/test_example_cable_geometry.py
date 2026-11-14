# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Geometric feasibility of the public 33 kV examples, and their pinned copies.

In October 2026 the two 33 kV examples were corrected: their
armour wires overlapped (lateral fill above 1) and the single-core cable had
undeclared 1 mm radial gaps where the semiconducting screens belong. The
pre-correction files are kept unchanged as ``*_legacy_r2.yaml`` to reproduce
earlier results.
"""

import hashlib

import pytest

import cablexsec as cx
from cablexsec.analysis.torque_balance import armour_fill_ratio
from cablexsec.cables import (
    inter_array_single_core_legacy_r2_yaml_path,
    inter_array_three_core_legacy_r2_yaml_path,
)
from cablexsec.cables.schema import (
    CylindricalLayer,
    CylindricalOffsetLayer,
    HelicalLayer,
)

pytestmark = pytest.mark.analytical_sanity

# SHA-256 of the files at 35f24e1, the state the R2 paths reproduce.
LEGACY_SHA256 = {
    "single": "00901572e85f6b8bb55e111e64baba51e808f284fb828ae3650df890adccf7ec",
    "three": "b396fabf77fd51ac51a8286a2d3a9b7b595643abb4e4db901fe4e1441d0a5ecb",
}
LEGACY = {
    "single": inter_array_single_core_legacy_r2_yaml_path,
    "three": inter_array_three_core_legacy_r2_yaml_path,
}
CORRECTED = {
    "single": cx.inter_array_single_core_yaml_path,
    "three": cx.inter_array_three_core_yaml_path,
}


def _load(path):
    return cx.load_cable(path, geometry_override=None)


def _fills(cable):
    return {
        layer.name: armour_fill_ratio(
            layer.n_wires, layer.d_wire_m, layer.r_helix_m, layer.lay_angle_deg
        )
        for layer in cable.layers
        if isinstance(layer, HelicalLayer)
    }


def _stack_gaps(intervals):
    """Radial gaps between consecutive (inner, outer) intervals of one stack."""
    ordered = sorted(intervals)
    return [b[0] - a[1] for a, b in zip(ordered, ordered[1:], strict=False)]


@pytest.mark.parametrize("name", sorted(LEGACY))
def test_legacy_copies_are_pinned(name):
    data = LEGACY[name]().read_bytes()
    assert hashlib.sha256(data).hexdigest() == LEGACY_SHA256[name]


@pytest.mark.parametrize("name", sorted(LEGACY))
def test_legacy_copies_keep_the_infeasible_r2_fill(name):
    # Why the copies are pinned and not used for new work.
    assert min(_fills(_load(LEGACY[name]())).values()) > 1.1


@pytest.mark.parametrize("name", sorted(CORRECTED))
def test_corrected_armour_wires_fit_side_by_side(name):
    corrected, legacy = _load(CORRECTED[name]()), _load(LEGACY[name]())
    fills = _fills(corrected)
    assert all(0.97 < fill <= 1.0 for fill in fills.values()), fills
    for layer in corrected.layers:
        if isinstance(layer, HelicalLayer):
            old = legacy.get_layer(layer.name)
            # Basis: the declared diameter, radius and lay angle are kept and
            # the count is the largest with fill <= 1.
            assert (layer.d_wire_m, layer.r_helix_m, layer.lay_angle_deg) == (
                old.d_wire_m,
                old.r_helix_m,
                old.lay_angle_deg,
            )
            one_more = armour_fill_ratio(
                layer.n_wires + 1, layer.d_wire_m, layer.r_helix_m, layer.lay_angle_deg
            )
            assert one_more > 1.0


def _core_stack(cable):
    return [
        (layer.r_inner_m, layer.r_outer_m)
        for layer in cable.layers
        if isinstance(layer, CylindricalOffsetLayer) and layer.r_offset_m > 0
    ]


def _concentric_stack(cable, after_radius=0.0):
    intervals = []
    for layer in cable.layers:
        if isinstance(layer, HelicalLayer):
            half = layer.d_wire_m / 2
            intervals.append((layer.r_helix_m - half, layer.r_helix_m + half))
        elif (
            isinstance(layer, CylindricalOffsetLayer)
            and layer.r_offset_m == 0
            or isinstance(layer, CylindricalLayer)
        ):
            intervals.append((layer.r_inner_m, layer.r_outer_m))
    return [i for i in intervals if i[0] >= after_radius]


def test_single_core_has_no_undeclared_radial_gap():
    legacy = _concentric_stack(_load(LEGACY["single"]()))
    assert max(_stack_gaps(legacy)) == pytest.approx(1.0e-3, abs=2e-6)
    gaps = _stack_gaps(_concentric_stack(_load(CORRECTED["single"]())))
    assert max(abs(g) for g in gaps) < 1e-9


def test_three_core_cores_have_no_undeclared_radial_gap():
    cable = _load(CORRECTED["three"]())
    assert max(abs(g) for g in _stack_gaps(_core_stack(cable))) < 1e-9
    # Bundle from the filler outwards: bedding, two armours, sheath.
    bundle = _concentric_stack(cable, after_radius=1e-6)
    assert max(abs(g) for g in _stack_gaps(bundle)) < 1e-9


@pytest.mark.parametrize("name", sorted(CORRECTED))
def test_screens_are_declared_with_insulation_properties(name):
    cable = _load(CORRECTED[name]())
    names = [layer.name for layer in cable.layers]
    assert names.index("conductor_screen") < names.index("xlpe_insulation")
    assert names.index("xlpe_insulation") < names.index("insulation_screen")
    for screen in ("conductor_screen", "insulation_screen"):
        layer = cable.get_layer(screen)
        assert layer.material == cable.get_layer("xlpe_insulation").material
        assert layer.r_outer_m - layer.r_inner_m == pytest.approx(1.0e-3, abs=1e-5)
        assert layer.electrical is None  # not part of the dielectric
