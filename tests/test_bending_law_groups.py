# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""One exported law per friction group, each with its own scale table.

Rows MX-group-B, MX-group-E and MX-group-0 of ``doc/benchmarks.md`` (gap MX),
registered before the run: the 66 kV template with the laid cores, the
filler profiles and armour clamping, whose groups scale differently with
tension. The global model sums the group laws ``s_g B_g(kappa / s_g)``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import cablexsec as cx
import cablexsec.analysis.bending_law as bending_law
from cablexsec.analysis.bending_law import (
    CORE_GROUP,
    FILLER_GROUP,
    SlipGroupLaw,
    export_bending_law,
)
from cablexsec.api import CoreFrictionSpec
from cablexsec.exceptions import CableInputError

pytestmark = [
    pytest.mark.analytical_sanity,
    pytest.mark.filterwarnings(
        "ignore::cablexsec.provenance.MissingClampingStateWarning"
    ),
]

TENSIONS = (1.0e4, 2.5e4, 5.0e4, 1.0e5, 1.5e5, 2.0e5)
AMPLITUDES = (1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 0.1, 0.3)
CORES = CoreFrictionSpec(
    mu_core_filler=0.46, mu_core_core=0.46, mu_core_sheath=0.46, filler_profiles=True
)
#: Registered targets (doc/benchmarks.md, MX-group).
BACKBONE_TOLERANCE = 5e-3
ENERGY_TOLERANCE = 1e-2


@pytest.fixture(scope="module")
def iea66():
    return cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)


def _wound(cable, sigma_Pa):
    """The cable with an armour winding stress on both armour layers."""
    layers = tuple(
        x.model_copy(update={"winding_stress_Pa": sigma_Pa})
        if x.name in {"inner_armour", "outer_armour"}
        else x
        for x in cable.layers
    )
    return cable.model_copy(update={"layers": layers})


@pytest.fixture(
    scope="module", params=[0.0, 20.0e6, 40.0e6], ids=["0MPa", "20MPa", "40MPa"]
)
def case(request, iea66):
    cable = _wound(iea66, request.param) if request.param else iea66
    export = export_bending_law(
        cable,
        tensions_N=TENSIONS,
        kappa_max_1_m=0.4,
        n_points=200,
        slip_scale=True,
        core_friction=CORES,
    )
    return request.param, cable, export


def _cycles(amplitude, per_cycle=200, cycles=3):
    t = np.linspace(0.0, 2.0 * math.pi * cycles, per_cycle * cycles + 1)
    return amplitude * np.sin(t)


def test_groups_and_their_scales_differ_with_clamping(case):
    sigma, _, export = case
    groups = export.slip_group_laws()
    assert [g.name for g in groups] == [
        "inner_armour",
        "outer_armour",
        CORE_GROUP,
        FILLER_GROUP,
    ]
    assert [g.kind for g in groups] == [
        "armour_layer",
        "armour_layer",
        CORE_GROUP,
        FILLER_GROUP,
    ]
    # The first group carries the elastic full-slip part; the others none.
    assert groups[0].law.ei_slip_Nm2 == export.slip_scale.ei_slip_Nm2
    assert all(g.law.ei_slip_Nm2 == 0.0 for g in groups[1:])
    high = np.array([g.scale_at([TENSIONS[-1]])[0] for g in groups])
    if sigma == 0.0:
        np.testing.assert_allclose(high, TENSIONS[-1] / TENSIONS[0], rtol=1e-9)
    else:
        # With clamping the laid-core group scales unlike the armour.
        assert np.ptp(high) / np.min(high) > 0.05


def test_mx_group_b_summed_backbones_reproduce_the_export(case):
    """MX-group-B: within 0.5 % of the exported moment at every curvature."""
    _, _, export = case
    groups = export.slip_group_laws()
    kappa = export.kappa_1_m[1:]
    worst = 0.0
    for i, tension in enumerate(export.tension_N):
        total = sum(g.backbone_at(kappa, float(tension)) for g in groups)
        reference = export.moment_Nm[i, 1:]
        worst = max(worst, float(np.max(np.abs(total / reference - 1.0))))
    assert worst <= BACKBONE_TOLERANCE


def test_mx_group_e_summed_loop_work_reproduces_the_export(case):
    """MX-group-E: work of three symmetric cycles within +-1 %."""
    _, _, export = case
    groups = export.slip_group_laws()
    for i, tension in enumerate(export.tension_N):
        laws = [g.law_at(float(tension)) for g in groups]
        for amplitude in AMPLITUDES:
            history = _cycles(amplitude)
            reference = export.laws[i].dissipated_energy(history)
            summed = sum(law.dissipated_energy(history) for law in laws)
            assert summed == pytest.approx(reference, rel=ENERGY_TOLERANCE)


def test_mx_group_0_table_runs_to_zero_tension(case):
    """MX-group-0: the default table starts at zero with the contact model's caps."""
    sigma, cable, export = case
    table = export.slip_scale
    assert table.tension_N[0] == 0.0
    np.testing.assert_array_equal(table.tension_N[1:], export.tension_N)
    contact, common = bending_law._law_options({"core_friction": CORES})
    reference = table.reference_cap_Nm

    def direct(tension):
        state = bending_law._state_law(cable, tension, None, contact, common)
        return bending_law._group_caps(state) / reference

    if sigma == 0.0:
        assert np.max(np.abs(table.group_scale[:, 0, 0])) <= 1e-12
    else:
        zero = direct(0.0)
        assert np.all(zero > 0.0)
        np.testing.assert_allclose(table.group_scale[:, 0, 0], zero, rtol=1e-12)
    np.testing.assert_allclose(
        table.group_scales([5.0e3])[:, 0], direct(5.0e3), rtol=1e-9, atol=1e-12
    )


def test_single_scale_is_reported_beside_the_groups(case):
    """MX-group-1 (reported): one cap-weighted scale for all groups."""
    sigma, _, export = case
    table = export.slip_scale
    kappa = export.kappa_1_m[1:]
    worst = 0.0
    for i, tension in enumerate(export.tension_N):
        single = table.law_at(float(tension), per_group=False).backbone(kappa)
        worst = max(worst, float(np.max(np.abs(single / export.moment_Nm[i, 1:] - 1))))
    if sigma == 0.0:
        assert worst <= BACKBONE_TOLERANCE
    else:
        assert worst > BACKBONE_TOLERANCE


def test_group_moment_history_and_zero_scale(iea66):
    export = export_bending_law(
        iea66,
        tensions_N=[2.0e4, 6.0e4],
        kappa_max_1_m=2e-3,
        n_points=40,
        slip_scale=True,
    )
    groups = export.slip_group_laws()
    table = export.slip_scale
    t = np.arange(400)
    kappa = 1.5e-3 * np.sin(2 * math.pi * t / 100 + 0.1)
    tension = 2.0e4 + 4.0e4 * np.sin(math.pi * t / 170) ** 2
    total = sum(g.moment(kappa, tension) for g in groups)
    np.testing.assert_allclose(
        total, table.moment(kappa, tension), rtol=0, atol=1e-12 * np.ptp(total)
    )
    # Zero tension: no contact without clamping, the elastic part alone.
    k = np.linspace(0.0, 2e-3, 11)
    np.testing.assert_allclose(
        groups[0].backbone_at(k, 0.0), table.ei_slip_Nm2 * k, rtol=1e-15
    )
    assert np.all(groups[1].backbone_at(k, 0.0) == 0.0)
    assert groups[1].law_at(0.0).stiffness_Nm2.size == 0
    # Virgin backbone at a table state is the merged law at that state.
    merged = table.law_at(6.0e4)
    np.testing.assert_allclose(
        sum(g.backbone_at(k, 6.0e4) for g in groups),
        merged.backbone(k),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        groups[0].moment_Nm, groups[0].law.backbone(export.kappa_1_m)
    )
    np.testing.assert_array_equal(groups[0].tension_N, table.tension_N)
    with pytest.raises(CableInputError, match="tension history"):
        groups[0].moment(kappa, tension[:-1])
    with pytest.raises(CableInputError, match="curvature history"):
        groups[0].moment(np.array([[0.0]]), np.array([[0.0]]))
    with pytest.raises(CableInputError, match="external-pressure"):
        groups[0].moment(kappa, tension, np.zeros(3))


def test_group_csv_files(iea66, tmp_path):
    export = export_bending_law(
        iea66,
        tensions_N=[2.0e4, 6.0e4],
        kappa_max_1_m=2e-3,
        n_points=20,
        slip_scale=True,
        slip_scale_tensions_N=[2.0e4, 6.0e4],
        slip_scale_external_pressure_Pa=[0.0, 1.0e6],
    )
    paths = export.to_group_csv(tmp_path / "groups")
    names = [p.name for p in paths]
    assert names == [
        "inner_armour_backbone.csv",
        "inner_armour_slip_scale.csv",
        "outer_armour_backbone.csv",
        "outer_armour_slip_scale.csv",
    ]
    lines = paths[0].read_text("utf-8").splitlines()
    assert lines[0].startswith("# friction group inner_armour (armour_layer)")
    assert lines[2] == "kappa_1_m,M_Nm"
    data = np.loadtxt(paths[0], delimiter=",", skiprows=3)
    group = export.slip_group_laws()[0]
    np.testing.assert_allclose(data[:, 0], export.kappa_1_m, rtol=1e-8)
    np.testing.assert_allclose(data[:, 1], group.moment_Nm, rtol=1e-8, atol=1e-12)
    scale = paths[3].read_text("utf-8").splitlines()
    assert scale[0] == "tension_N,external_pressure_Pa,s,ds_dT_1_per_N,ds_dpe_1_per_Pa"
    assert len(scale) == 1 + 2 * 2
    row = np.array(scale[2].split(","), dtype=float)
    outer = export.slip_group_laws()[1]
    assert row[:2].tolist() == [2.0e4, 1.0e6]
    assert row[2] == pytest.approx(outer.scale[0, 1], rel=1e-8)
    assert row[4] == pytest.approx(
        export.slip_scale.group_scale_dpe_1_per_Pa[1, 0, 1], rel=1e-8
    )
    tension_only = export_bending_law(
        iea66, tensions_N=[2.0e4], kappa_max_1_m=2e-3, n_points=20, slip_scale=True
    )
    text = tension_only.slip_group_laws()[0].scale_csv_text().splitlines()
    assert text[0] == "tension_N,external_pressure_Pa,s,ds_dT_1_per_N"
    assert [float(x.split(",")[0]) for x in text[1:]] == [0.0, 2.0e4]


def test_group_csv_file_names_stay_in_the_directory(iea66, tmp_path):
    from cablexsec.analysis.bending_law import _file_stem, _file_stems

    assert _file_stem("armour/inner") == "armour_inner"
    assert _file_stem("../../etc") == ".._.._etc"
    assert _file_stem("..") == "group" and _file_stem("") == "group"
    assert _file_stem("C:\\x y") == "C__x_y"
    assert _file_stems(["a/b", "a_b", "A_B", "laid_cores"]) == [
        "a_b",
        "a_b_2",
        "A_B_3",
        "laid_cores",
    ]
    names = {"inner_armour": "armour/inner", "outer_armour": "../outer"}
    layers = tuple(
        x.model_copy(update={"name": names[x.name]}) if x.name in names else x
        for x in iea66.layers
    )
    renamed = iea66.model_copy(update={"layers": layers})
    export = export_bending_law(
        renamed, tensions_N=[2.0e4], kappa_max_1_m=2e-3, n_points=20, slip_scale=True
    )
    folder = tmp_path / "out"
    paths = export.to_group_csv(folder)
    assert [p.name for p in paths] == [
        "armour_inner_backbone.csv",
        "armour_inner_slip_scale.csv",
        ".._outer_backbone.csv",
        ".._outer_slip_scale.csv",
    ]
    assert all(p.parent == folder and p.is_file() for p in paths)
    first = paths[0].read_text("utf-8").splitlines()[0]
    assert first.startswith("# friction group armour/inner (armour_layer)")


def test_group_law_checks(iea66):
    plain = export_bending_law(
        iea66, tensions_N=[2.0e4], kappa_max_1_m=2e-3, n_points=20
    )
    with pytest.raises(CableInputError, match="slip_scale=True"):
        plain.slip_group_laws()
    export = export_bending_law(
        iea66, tensions_N=[2.0e4], kappa_max_1_m=2e-3, n_points=20, slip_scale=True
    )
    table = export.slip_scale
    with pytest.raises(CableInputError, match="group index"):
        table.group_law(5, export.kappa_1_m)
    with pytest.raises(CableInputError, match="export grid"):
        table.group_law(0, export.kappa_1_m[:-1])
    group = table.group_law(1, export.kappa_1_m)
    assert isinstance(group, SlipGroupLaw) and group.index == 1
    # An explicit table axis is used as given (no zero row added).
    explicit = export_bending_law(
        iea66,
        tensions_N=[2.0e4],
        kappa_max_1_m=2e-3,
        n_points=20,
        slip_scale=True,
        slip_scale_tensions_N=[2.0e4, 4.0e4],
    )
    np.testing.assert_array_equal(explicit.slip_scale.tension_N, [2.0e4, 4.0e4])
    # A table that already starts at zero keeps one zero row.
    from_zero = export_bending_law(
        iea66,
        tensions_N=[0.0, 2.0e4],
        kappa_max_1_m=2e-3,
        n_points=20,
        slip_scale=True,
    )
    np.testing.assert_array_equal(from_zero.slip_scale.tension_N, [0.0, 2.0e4])
