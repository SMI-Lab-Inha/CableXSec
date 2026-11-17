# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Gap a2: slip scaling variable of the exported bending law.

The direction-tied armour law has fixed stick stiffnesses and friction bounds
proportional to each layer's ``P1+P2``, so a layer whose bounds scale by
``rho`` obeys ``M(kappa; rho) = rho M(kappa / rho; 1)`` over any history, and
its Iwan elements keep their stiffnesses while their yield curvatures scale.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest

import cablexsec as cx
import cablexsec.analysis.bending_law as bending_law
import cablexsec.mechanics.papailiou as papailiou
from cablexsec._project_codec import pack, unpack
from cablexsec.analysis.bending_law import (
    CORE_GROUP,
    SlipScaleTable,
    export_bending_law,
)
from cablexsec.api import _prepare_mkappa_law
from cablexsec.cable_adapters import armour_contact_pressure
from cablexsec.exceptions import CableInputError

pytestmark = [
    pytest.mark.analytical_sanity,
    pytest.mark.filterwarnings(
        "ignore::cablexsec.provenance.MissingClampingStateWarning"
    ),
]


def _cable(path):
    return cx.load_cable(path, geometry_override=None)


@pytest.fixture(scope="module")
def menard():
    return _cable(cx.menard2023_yaml_path())


@pytest.fixture(scope="module")
def iea66():
    return _cable(cx.iea_task49_66kv_yaml_path())


def _table(cable, kappa, reference_N, tensions_N, pressures=None, **options):
    return bending_law._slip_scale_table(
        cable,
        np.asarray(kappa, dtype=np.float64),
        reference_N=reference_N,
        candidates_N=np.array([reference_N]),
        tensions_N=tensions_N,
        external_pressure_Pa=pressures,
        options=options,
    )


def _grid(kappa_max, n=160):
    return np.concatenate([[0.0], np.geomspace(kappa_max * 1e-4, kappa_max, n - 1)])


def _history(n=1200, amplitude=0.03, phase=0.1):
    t = np.arange(n)
    # Incommensurate bending and tension cycles; the phase keeps the samples
    # off the stick/slip breakpoints.
    kappa = amplitude * np.sin(2.0 * np.pi * t / 200.0 + phase)
    tension = 1.0e4 + 6.0e4 * np.sin(np.pi * t / 370.0) ** 2
    return kappa, tension


def test_section_law_is_homogeneous_in_the_friction_bound(iea66):
    # The 66 kV template has no manufacturing clamping: P1+P2 of both armour
    # layers and the core contact load are proportional to the tension, so
    # every group scales by T2 / T1 and M(kappa; T2) = EI k + rho F(k / rho; T1)
    # over a loop history, cores included.
    t1, t2 = 2.0e4, 7.0e4
    rho = t2 / t1
    kappa = (
        4e-4
        * np.sin(np.linspace(0.0, 5.0 * np.pi, 700) + 0.1)
        * np.linspace(0.3, 1.0, 700)
    )
    table = _table(iea66, _grid(1e-3), t1, [t1, t2], core_friction=True)
    assert table.group_names == ("inner_armour", "outer_armour", CORE_GROUP)
    np.testing.assert_allclose(table.group_scale[:, 1, 0], rho, rtol=1e-12)
    at_t2 = cx.compute_mkappa(iea66, kappa, tension_N=t2, core_friction=True)
    at_t1 = cx.compute_mkappa(iea66, kappa / rho, tension_N=t1, core_friction=True)
    ei = at_t2.EI_min_Nm2
    predicted = ei * kappa + rho * (np.asarray(at_t1.moment_Nm) - ei * kappa / rho)
    scale = np.max(np.abs(at_t2.moment_Nm))
    assert np.max(np.abs(predicted - at_t2.moment_Nm)) < 1e-12 * scale


@pytest.mark.parametrize(
    ("path", "core"),
    [
        (cx.menard2023_yaml_path(), None),
        (cx.inter_array_three_core_yaml_path(), True),
    ],
)
def test_group_yield_curvatures_scale_exactly(path, core):
    # Every group law at T2, decomposed from its own backbone on the grid
    # rho_g kappa, is the reference group law with fixed stiffnesses and
    # yield curvatures times rho_g (layers with different rho_g included).
    cable = _cable(path)
    kappa = _grid(0.3, 200)
    reference = _table(cable, kappa, 1.0e4, [1.0e4, 6.0e4], core_friction=core)
    rho = reference.group_scale[:, 1, 0]
    assert np.ptp(rho) > 1e-3  # the groups do scale differently
    for g, name in enumerate(reference.group_names):
        state = _table(cable, kappa * rho[g], 6.0e4, [6.0e4], core_friction=core)
        a, b = reference.group_laws[g], state.group_laws[g]
        assert state.group_names[g] == name
        k_scale = np.max(a.stiffness_Nm2)
        assert np.max(np.abs(a.stiffness_Nm2 - b.stiffness_Nm2)) < 1e-12 * k_scale
        np.testing.assert_allclose(
            b.yield_curvature_1_m, rho[g] * a.yield_curvature_1_m, rtol=1e-12
        )
        assert state.reference_cap_Nm[g] == pytest.approx(
            rho[g] * reference.reference_cap_Nm[g], rel=1e-12
        )


def test_single_scale_reproduces_the_export_when_groups_scale_alike(iea66):
    # Public route: the law exported at T2 on the curvature grid scaled by s
    # is the reference law with every yield curvature times s.
    t1, t2, kappa_max = 2.0e4, 6.0e4, 2e-3
    s = t2 / t1
    export = export_bending_law(
        iea66,
        tensions_N=[t1, t2],
        kappa_max_1_m=kappa_max,
        n_points=120,
        slip_scale=True,
    )
    table = export.slip_scale
    assert table is not None and table.reference_tension_N == t1
    np.testing.assert_allclose(table.scale[:, 0], [1.0, s], rtol=1e-12)
    # The grid carries the slip curvatures of every exported tension: at
    # [t2, s t2] it is the reference grid times s.
    direct = export_bending_law(
        iea66, tensions_N=[t2, s * t2], kappa_max_1_m=s * kappa_max, n_points=120
    ).laws[0]
    scaled = table.law_at(t2, per_group=False)
    np.testing.assert_allclose(
        scaled.yield_curvature_1_m, direct.yield_curvature_1_m, rtol=1e-12
    )
    # Element stiffnesses are differences of backbone tangents, whose round-off
    # scales with the stick tangent (the stick grid reaches down to 1e-6 1/m).
    k_scale = direct.ei_stick_Nm2
    assert np.max(np.abs(scaled.stiffness_Nm2 - direct.stiffness_Nm2)) < 1e-12 * k_scale
    assert scaled.ei_slip_Nm2 == pytest.approx(direct.ei_slip_Nm2, rel=1e-12)
    # At the reference the merged groups are the exported law.
    merged = table.law_at(t1)
    moment = merged.backbone(export.kappa_1_m)
    assert np.max(np.abs(moment - export.moment_Nm[0])) < 1e-12 * np.ptp(
        export.moment_Nm[0]
    )


def _symmetric_bounds(monkeypatch):
    original = papailiou.wire_slip_stress_bounds

    def symmetric(layer, phi_bend, *, contact_pressure_Pa=None):
        upper, _ = original(layer, phi_bend, contact_pressure_Pa=contact_pressure_Pa)
        return upper, upper.copy()

    monkeypatch.setattr(papailiou, "wire_slip_stress_bounds", symmetric)
    monkeypatch.setattr(bending_law, "wire_slip_stress_bounds", symmetric)
    return symmetric


def _breakpoint_grid(cable, tension_N, bounds, kappa_top):
    """Curvature grid through the slip curvature of every wire."""
    law = _prepare_mkappa_law(
        cable,
        tension_N,
        mu=None,
        phi_bend=0.0,
        contact_model="bounded",
        section_state=None,
        radial_state=None,
        core_spec=None,
        progressive=None,
    )
    yields = []
    for spec in law.layers:
        upper, _ = bounds(spec, 0.0)
        c = np.abs(np.cos(spec.phi_wires))
        loaded = (c > 1e-12) & (upper > 0.0)
        yields.append(upper[loaded] / (spec.k_f * spec.r_helix_m * c[loaded]))
    y = np.sort(np.concatenate(yields))
    # Wires of equal |cos| share a slip curvature up to rounding: keep one
    # point per cluster, so no tangent is formed over a rounding gap.
    start = np.concatenate([[True], np.diff(y) > 1e-9 * y[1:]])
    cluster = np.cumsum(start) - 1
    points = np.array([y[cluster == i].max() for i in range(cluster[-1] + 1)])
    return np.concatenate([[0.0], points, [kappa_top]])


def test_slider_rule_is_the_history_route_for_symmetric_bounds(menard, monkeypatch):
    # The key identity. With symmetric wire bounds every wire is an Iwan
    # element, and on a grid through every wire's slip curvature the
    # decomposition is exact. A tension cycle inside the bending cycles then
    # gives the same moment and friction work through the scaled law (every
    # element's return map with the new cap) as through compute_mkappa with
    # the contact-tension history (every wire's return map with the new
    # bound): two layers that scale differently, caps falling and rising in
    # stick and in slip.
    bounds = _symmetric_bounds(monkeypatch)
    kappa_grid = _breakpoint_grid(menard, 1.0e4, bounds, 0.1)
    table = _table(menard, kappa_grid, 1.0e4, np.linspace(1.0e4, 7.0e4, 7))
    kappa, tension = _history()
    result = cx.compute_mkappa(menard, kappa, contact_tension_history_N=tension)
    moment = table.moment(kappa, tension)
    scale = np.max(np.abs(result.moment_Nm))
    assert np.max(np.abs(moment - result.moment_Nm)) < 1e-12 * scale
    assert result.work is not None
    work = result.work.friction_J_per_m[-1]
    assert table.dissipated_energy(kappa, tension) == pytest.approx(work, rel=1e-12)
    # The rule matters: the law frozen at the reference misses by 10-20 %.
    frozen = table.moment(kappa, np.full_like(tension, 1.0e4))
    assert np.max(np.abs(frozen - result.moment_Nm)) > 0.05 * scale


@pytest.mark.parametrize(
    ("path", "amplitude", "n_points"),
    [
        (cx.menard2023_yaml_path(), 0.03, 300),
        (cx.iea_task49_66kv_yaml_path(), 3e-4, 300),
    ],
)
def test_tension_cycle_adds_nothing_to_the_fixed_tension_residual(
    path, amplitude, n_points
):
    # With the signed capstan bound the falling bound of a wire is its
    # rising bound times exp(-lam psi): its slip interval is asymmetric, so
    # the section law is not exactly of Masing type and every exported Iwan
    # law misses compute_mkappa loops by a residual. Under a tension cycle
    # inside the bending cycles the scaled law misses the history route by no
    # more than that residual at the two ends of the tension range.
    cable = _cable(path)
    kappa, tension = _history(amplitude=amplitude)
    export = export_bending_law(
        cable,
        tensions_N=[1.0e4, 7.0e4],
        kappa_max_1_m=1.2 * amplitude,
        n_points=n_points,
        slip_scale=True,
        slip_scale_tensions_N=np.linspace(1.0e4, 7.0e4, 7),
    )
    table = export.slip_scale
    assert table is not None
    history = np.asarray(
        cx.compute_mkappa(cable, kappa, contact_tension_history_N=tension).moment_Nm
    )
    span = np.ptp(history)
    residual = []
    for level in (1.0e4, 7.0e4):
        fixed = np.asarray(cx.compute_mkappa(cable, kappa, tension_N=level).moment_Nm)
        law = table.moment(kappa, np.full_like(kappa, level))
        residual.append(np.max(np.abs(law - fixed)) / np.ptp(fixed))
    scaled = table.moment(kappa, tension)
    deviation = np.max(np.abs(scaled - history)) / span
    assert deviation <= 1.25 * max(residual)
    assert deviation < 5e-3
    frozen = table.moment(kappa, np.full_like(tension, table.reference_tension_N))
    assert np.max(np.abs(frozen - history)) / span > 20 * deviation


def test_single_scale_error_with_layers_that_scale_differently(menard):
    # Menard, 10 -> 70 kN: inner P1+P2 x 2.3, outer x 1.6. The cap-weighted
    # single scale keeps the stick stiffness and the full-slip moment but
    # misplaces the transition by several percent of the moment range.
    kappa = _grid(0.05, 200)
    table = _table(menard, kappa, 1.0e4, [1.0e4, 7.0e4])
    exact = table.law_at(7.0e4)
    single = table.law_at(7.0e4, per_group=False)
    assert single.ei_stick_Nm2 == pytest.approx(exact.ei_stick_Nm2, rel=1e-12)
    full_slip = float(exact.stiffness_Nm2 @ exact.yield_curvature_1_m)
    assert float(single.stiffness_Nm2 @ single.yield_curvature_1_m) == pytest.approx(
        full_slip, rel=1e-12
    )
    k = np.linspace(0.0, 0.05, 1001)
    error = np.max(np.abs(single.backbone(k) - exact.backbone(k))) / np.ptp(
        exact.backbone(k)
    )
    assert 0.02 < error < 0.2


def test_scale_values_and_derivatives(iea66, menard):
    # Bounded chain: P1+P2 affine in tension, so the cap is too and the
    # central differences give its slope.
    table = _table(iea66, _grid(1e-3), 2.0e4, [0.0, 2.0e4, 1.0e5])
    np.testing.assert_allclose(table.scale[:, 0], [0.0, 1.0, 5.0], atol=1e-12)
    np.testing.assert_allclose(table.scale_dT_1_per_N, 1.0 / 2.0e4, rtol=1e-6)
    np.testing.assert_allclose(table.group_scale_dT_1_per_N, 1.0 / 2.0e4, rtol=1e-6)
    assert table.scale_dpe_1_per_Pa is None
    assert table.group_scale_dpe_1_per_Pa is None
    # Ménard: each layer's slope is the slope of its own P1+P2.
    pressures = [armour_contact_pressure(menard, t) for t in (1.0e4, 7.0e4)]
    table = _table(menard, _grid(0.05), 1.0e4, [1.0e4, 7.0e4])
    for g, name in enumerate(table.group_names):
        slope = (pressures[1][name] - pressures[0][name]) / 6.0e4
        np.testing.assert_allclose(
            table.group_scale_dT_1_per_N[g], slope / pressures[0][name], rtol=1e-6
        )
        np.testing.assert_allclose(
            table.group_scale[g, :, 0],
            [1.0, pressures[1][name] / pressures[0][name]],
            rtol=1e-12,
        )
    # Interpolation between table rows is linear in the caps.
    rho = table.group_scales([4.0e4])[:, 0]
    np.testing.assert_allclose(rho, table.group_scale[:, :, 0].mean(axis=1), rtol=1e-12)
    single = table.group_scales([4.0e4], per_group=False)[:, 0]
    np.testing.assert_allclose(single, table.scale[:, 0].mean(), rtol=1e-12)


def test_external_pressure_axis(menard):
    # A pressure axis evaluates solved section states. At the reference
    # pressure they give the caps of the tension-only table; the bounded
    # chain is affine in the external pressure as well.
    p0 = menard.initial_state.external_pressure_Pa
    pressures = [p0, p0 + 1.0e6, p0 + 2.0e6]
    kappa = _grid(0.05, 60)
    plain = _table(menard, kappa, 1.0e4, [1.0e4, 5.0e4])
    table = _table(menard, kappa, 1.0e4, [1.0e4, 5.0e4], pressures)
    np.testing.assert_allclose(
        table.reference_cap_Nm, plain.reference_cap_Nm, rtol=1e-12
    )
    np.testing.assert_allclose(
        table.group_cap_Nm[:, :, 0], plain.group_cap_Nm[:, :, 0], rtol=1e-12
    )
    for g in range(len(table.group_names)):
        k, k_ref = table.group_laws[g], plain.group_laws[g]
        np.testing.assert_allclose(k.stiffness_Nm2, k_ref.stiffness_Nm2, rtol=1e-12)
    d_p = table.group_scale_dpe_1_per_Pa
    assert d_p is not None and table.scale_dpe_1_per_Pa is not None
    slope = np.diff(table.group_scale, axis=2) / 1.0e6
    np.testing.assert_allclose(d_p[:, :, :2], slope, rtol=1e-6)
    np.testing.assert_allclose(d_p[:, :, 2], slope[:, :, 1], rtol=1e-6)
    # Bilinear interpolation of an affine state is exact.
    rho = table.group_scales([3.0e4], [p0 + 0.5e6])[:, 0]
    expected = table.group_scale[:, :, :2].mean(axis=(1, 2))
    np.testing.assert_allclose(rho, expected, rtol=1e-12)
    # A pressure history drives the moment like a tension history.
    kappa_h, tension_h = _history(400, amplitude=0.02)
    tension_h = np.clip(tension_h, 1.0e4, 5.0e4)
    pressure_h = p0 + 1.5e6 * np.sin(np.linspace(0.0, 3.0, 400)) ** 2
    moment = table.moment(kappa_h, tension_h, pressure_h)
    assert np.isfinite(moment).all()
    with pytest.raises(CableInputError, match="no external-pressure axis"):
        plain.group_scales([2e4], [p0])
    with pytest.raises(CableInputError, match="outside the table"):
        table.group_scales([2e4], [p0 + 3e6])


def test_cap_is_the_full_slip_friction_moment(menard):
    # The analytic cap equals each group backbone far beyond full slip.
    kappa = np.concatenate([[0.0], np.geomspace(1e-5, 2.0, 300)])
    table = _table(menard, kappa, 3.0e4, [3.0e4])
    for g, law in enumerate(table.group_laws):
        assert law.backbone(np.array([2.0]))[0] == pytest.approx(
            table.reference_cap_Nm[g], rel=1e-9
        )


def test_json_csv_and_project_round_trip(iea66, tmp_path):
    plain = export_bending_law(
        iea66, tensions_N=[2e4, 6e4], kappa_max_1_m=2e-3, n_points=30
    )
    record = json.loads(plain.json_text())
    assert record["format"] == "cablexsec.bending_law/1" and "slip_scale" not in record
    assert plain.slip_scale is None
    export = export_bending_law(
        iea66,
        tensions_N=[2e4, 6e4],
        kappa_max_1_m=2e-3,
        n_points=30,
        slip_scale=True,
        slip_scale_tensions_N=[2e4, 4e4, 6e4],
        slip_scale_external_pressure_Pa=[0.0, 1e6],
    )
    table = export.slip_scale
    assert table is not None
    record = json.loads(export.to_json(tmp_path / "law.json").read_text("utf-8"))
    assert record["format"] == "cablexsec.bending_law/2"
    for key in ("tension_N", "moment_Nm", "elements", "ei_slip_Nm2", "kappa_1_m"):
        assert record[key] == json.loads(plain.json_text())[key]
    block = record["slip_scale"]
    assert block["rule"] == "yield_curvature_scaling"
    assert block["variables"] == ["tension_N", "external_pressure_Pa"]
    assert np.asarray(block["scale"]).shape == (3, 2)
    assert [g["name"] for g in block["groups"]] == list(table.group_names)
    assert block["groups"][0]["kind"] == "armour_layer"
    np.testing.assert_allclose(block["dscale_dtension_1_per_N"], table.scale_dT_1_per_N)
    merged = table.law_at(2e4, 0.0)
    assert block["reference_elements"]["yield_curvature_1_m"] == (
        merged.yield_curvature_1_m.tolist()
    )
    text = table.to_csv(tmp_path / "scale.csv").read_text("utf-8").splitlines()
    assert text[0].startswith("tension_N,external_pressure_Pa,s,ds_dT_1_per_N,ds_dpe")
    assert "s_inner_armour" in text[0] and len(text) == 1 + 3 * 2
    row = np.array(text[2].split(","), dtype=float)
    assert row[0] == 2e4 and row[1] == 1e6
    assert row[2] == pytest.approx(table.scale[0, 1], rel=1e-8)
    tension_only = export_bending_law(
        iea66, tensions_N=[2e4, 6e4], kappa_max_1_m=2e-3, n_points=30, slip_scale=True
    )
    assert tension_only.slip_scale is not None
    lines = tension_only.slip_scale.csv_text().splitlines()
    assert lines[0].split(",")[:4] == [
        "tension_N",
        "external_pressure_Pa",
        "s",
        "ds_dT_1_per_N",
    ]
    assert "ds_dpe" not in lines[0]
    # Project codec: the table round-trips, and a payload written before the
    # field existed (no "slip_scale") still decodes.
    decoded = unpack(pack(export))
    assert decoded.slip_scale is not None
    np.testing.assert_array_equal(decoded.slip_scale.group_cap_Nm, table.group_cap_Nm)
    payload = pack(plain)
    del payload["fields"]["slip_scale"]
    assert unpack(payload).slip_scale is None
    missing = pack(plain)
    del missing["fields"]["laws"]
    with pytest.raises(CableInputError):
        unpack(missing)


def test_project_settings_pass_the_slip_scale(iea66):
    from cablexsec.project import BendingLawSettings, CableModel

    model = CableModel(iea66)
    settings = BendingLawSettings(
        tensions_N=(2e4, 6e4),
        kappa_max_1_m=2e-3,
        n_points=20,
        slip_scale=True,
        slip_scale_reference_N=2e4,
    )
    export = model.bending_law(settings)
    assert export.slip_scale is not None
    assert export.slip_scale.reference_tension_N == 2e4


@pytest.mark.parametrize(
    "axes",
    [
        {"slip_scale_tensions_N": [3e4, 6e4]},
        {
            "slip_scale_tensions_N": [2e4, 6e4],
            "slip_scale_external_pressure_Pa": [1e6, 2e6],
        },
    ],
    ids=["tension_axis", "pressure_axis"],
)
def test_reference_outside_the_table_axes_serialises(iea66, axes):
    # The default reference (20 kN, the cable's own external pressure) lies
    # outside these axes: the table refuses it as a state, but the reference
    # elements are the group elements at scale 1 and still serialise.
    export = export_bending_law(
        iea66,
        tensions_N=[2e4, 6e4],
        kappa_max_1_m=2e-3,
        n_points=30,
        slip_scale=True,
        **axes,
    )
    table = export.slip_scale
    assert table is not None and table.reference_tension_N == 2e4
    with pytest.raises(CableInputError):
        table.law_at(
            table.reference_tension_N,
            None
            if table.external_pressure_Pa is None
            else table.reference_external_pressure_Pa,
        )
    block = json.loads(export.json_text())["slip_scale"]
    reference = table.reference_law()
    assert (
        block["reference_elements"]["stiffness_Nm2"] == reference.stiffness_Nm2.tolist()
    )
    assert block["reference_elements"]["yield_curvature_1_m"] == (
        reference.yield_curvature_1_m.tolist()
    )
    # The same merge as law_at on a table that holds the reference state.
    inside = export_bending_law(
        iea66,
        tensions_N=[2e4, 6e4],
        kappa_max_1_m=2e-3,
        n_points=30,
        slip_scale=True,
        slip_scale_tensions_N=[2e4, 6e4],
    ).slip_scale
    assert inside is not None
    at_reference = inside.law_at(inside.reference_tension_N)
    np.testing.assert_allclose(
        inside.reference_law().stiffness_Nm2, at_reference.stiffness_Nm2, rtol=1e-12
    )
    np.testing.assert_allclose(
        inside.reference_law().yield_curvature_1_m,
        at_reference.yield_curvature_1_m,
        rtol=1e-12,
    )


def test_export_checks(iea66):
    args = {"tensions_N": [0.0, 2e4], "kappa_max_1_m": 1e-3, "n_points": 20}
    with pytest.raises(CableInputError, match="need slip_scale=True"):
        export_bending_law(iea66, slip_scale_reference_N=2e4, **args)
    for name in ("contact_tension_history_N", "contact_pressure_Pa"):
        with pytest.raises(CableInputError, match="set by the export"):
            export_bending_law(iea66, **args, **{name: None})
    # No contact at zero tension on this template: the default reference is
    # the first tension with contact, and zero itself is refused.
    export = export_bending_law(iea66, slip_scale=True, **args)
    assert export.slip_scale is not None
    assert export.slip_scale.reference_tension_N == 2e4
    np.testing.assert_allclose(export.slip_scale.scale[:, 0], [0.0, 1.0], atol=1e-15)
    # Without contact the scaled law is the full-slip stiffness alone.
    bare = export.slip_scale.law_at(0.0)
    assert bare.stiffness_Nm2.size == 0
    assert bare.ei_stick_Nm2 == export.slip_scale.ei_slip_Nm2
    with pytest.raises(CableInputError, match="no friction at the reference"):
        export_bending_law(iea66, slip_scale=True, slip_scale_reference_N=0.0, **args)
    with pytest.raises(CableInputError, match="slip_scale_tensions_N"):
        export_bending_law(
            iea66, slip_scale=True, slip_scale_tensions_N=[2e4, 1e4], **args
        )
    with pytest.raises(CableInputError, match="slip_scale_external_pressure_Pa"):
        export_bending_law(
            iea66, slip_scale=True, slip_scale_external_pressure_Pa=[-1.0], **args
        )
    with pytest.raises(CableInputError, match="slip_scale_reference_N"):
        export_bending_law(
            iea66, slip_scale=True, slip_scale_reference_N=math.nan, **args
        )


def test_strand_rings_are_refused():
    cable = _cable(cx.thies2016_yaml_path())
    with pytest.raises(CableInputError, match="strand"):
        export_bending_law(
            cable,
            tensions_N=[2e4],
            kappa_max_1_m=0.05,
            n_points=20,
            slip_scale=True,
            core_friction=True,
        )


def test_history_and_table_checks(iea66):
    table = _table(iea66, _grid(1e-3, 40), 2.0e4, [2.0e4, 6.0e4])
    with pytest.raises(CableInputError, match="outside the table"):
        table.group_scales([1.0e4])
    with pytest.raises(CableInputError, match="finite"):
        table.group_scales([math.nan])
    with pytest.raises(CableInputError, match="1-D"):
        table.group_scales(np.zeros((2, 2)) + 3e4)
    with pytest.raises(CableInputError, match="tension history must match"):
        table.moment(np.zeros(3), np.full(2, 3e4))
    with pytest.raises(CableInputError, match="curvature history"):
        table.moment(np.zeros((2, 2)), np.full(2, 3e4))
    with pytest.raises(CableInputError, match="external-pressure history"):
        table.moment(np.zeros(3), np.full(3, 3e4), np.zeros(2))
    with pytest.raises(CableInputError, match="no external-pressure axis"):
        table.moment(np.zeros(3), np.full(3, 3e4), np.zeros(3))
    one = _table(iea66, _grid(1e-3, 40), 2.0e4, [2.0e4])
    np.testing.assert_allclose(one.group_scales([2.0e4]), 1.0)
    with pytest.raises(CableInputError, match="outside"):
        one.group_scales([2.1e4])
    # Constructor checks.
    good = {f: getattr(table, f) for f in table.__dataclass_fields__}
    cases = [
        ({"tension_N": np.array([2e4, 1e4])}, "strictly ascending"),
        ({"group_names": ("a",)}, "one law per friction group"),
        ({"group_names": ("a", "a")}, "distinct"),
        ({"reference_cap_Nm": np.array([1.0, 0.0])}, "positive reference cap"),
        ({"group_cap_Nm": np.zeros((2, 3, 1))}, "shape"),
        ({"group_cap_Nm": -table.group_cap_Nm}, "non-negative"),
        ({"group_cap_dT_Nm_per_N": table.group_cap_dT_Nm_per_N * np.nan}, "finite"),
        ({"group_cap_dpe_Nm_per_Pa": table.group_cap_dT_Nm_per_N}, "pressure axis"),
        ({"reference_tension_N": -1.0}, "reference state"),
        ({"ei_slip_Nm2": -1.0}, "ei_slip_Nm2"),
        ({"external_pressure_Pa": np.array([0.0])}, "pressure derivative"),
    ]
    for change, match in cases:
        with pytest.raises(CableInputError, match=match):
            SlipScaleTable(**{**good, **change})
    with pytest.raises(CableInputError, match="pressure derivative must be finite"):
        SlipScaleTable(
            **{
                **good,
                "external_pressure_Pa": np.array([0.0]),
                "group_cap_dpe_Nm_per_Pa": np.zeros((2, 1, 1)),
            }
        )


def test_falling_cap_at_fixed_curvature_does_slider_work(iea66):
    # Hold the curvature in full slip and lower the tension: every element
    # is clipped to the new cap; the work of one step is k e1 (e0 - e1).
    table = _table(iea66, _grid(1e-3, 60), 2.0e4, [1.0e4, 2.0e4])
    kappa = np.array([5e-4, 5e-4])
    tension = np.array([2.0e4, 1.0e4])
    law = table.law_at(2.0e4)
    before = table.moment(kappa[:1], tension[:1])
    after = table.moment(kappa, tension)
    full = law.ei_slip_Nm2 * 5e-4 + float(law.stiffness_Nm2 @ law.yield_curvature_1_m)
    assert before[0] == pytest.approx(full, rel=1e-12)
    expected = law.ei_slip_Nm2 * 5e-4 + 0.5 * (full - law.ei_slip_Nm2 * 5e-4)
    assert after[1] == pytest.approx(expected, rel=1e-12)
    k, y = law.stiffness_Nm2, law.yield_curvature_1_m
    step = float(np.sum(k * (0.5 * y) * (0.5 * y)))
    loading = table.dissipated_energy(kappa[:1], tension[:1])
    assert table.dissipated_energy(kappa, tension) - loading == pytest.approx(
        step, rel=1e-12
    )
    # Raising the cap back leaves the clipped elements where they are.
    again = table.moment(np.full(3, 5e-4), np.array([2.0e4, 1.0e4, 2.0e4]))
    assert again[2] == pytest.approx(after[1], rel=1e-12)


def test_replace_keeps_the_table(iea66):
    export = export_bending_law(
        iea66, tensions_N=[2e4], kappa_max_1_m=1e-3, n_points=20, slip_scale=True
    )
    again = replace(export, settings={})
    assert again.slip_scale is export.slip_scale


def _touching_alternate_wires():
    """Three touching armour layers laid alternately (two crossing interfaces in
    the middle one, which the crossed-wire line loads refuse)."""
    from cablexsec.cables.schema import CableData, CableMeta, HelicalLayer

    layers = tuple(
        HelicalLayer(
            name=f"wire-{i}",
            material="galvanised_steel_armour",
            n_wires=12 * (i + 1),
            d_wire_m=0.002,
            r_helix_m=0.02 + 0.002 * i,
            lay_angle_deg=20 + 10 * i,
            lay_direction="left_hand" if i == 1 else "right_hand",
        )
        for i in range(3)
    )
    return CableData(cable=CableMeta(name="touching alternate wires"), layers=layers)


@pytest.mark.parametrize(
    ("contact_model", "pressures"),
    # (the solved model of this sheath-free synthetic stack meets a singular
    # sparse factor in some environments, also without the slip scale)
    [("bounded", None), ("bounded", [0.0, 1e6])],
)
def test_slip_scale_uses_the_pressure_only_route(contact_model, pressures):
    # The scale evaluates P1+P2 with the scalar-tension contact states of
    # compute_mkappa, never the crossed-wire line loads, so a stack whose
    # contact histories are refused still exports it.
    cable = _touching_alternate_wires()
    with pytest.raises(cx.CableSchemaError):
        cx.armour_contact_histories(cable, np.array([1.0e4]))
    export = export_bending_law(
        cable,
        tensions_N=[1.0e4, 3.0e4],
        kappa_max_1_m=0.06,
        n_points=60,
        contact_model=contact_model,
        slip_scale=True,
        slip_scale_external_pressure_Pa=pressures,
    )
    table = export.slip_scale
    assert table is not None
    assert table.group_names == ("wire-0", "wire-1", "wire-2")
    for i, tension in enumerate(export.tension_N):
        got = table.law_at(tension, None if pressures is None else 0.0)
        np.testing.assert_allclose(
            got.ei_stick_Nm2, export.laws[i].ei_stick_Nm2, rtol=1e-12
        )
    if contact_model == "bounded" and pressures is None:
        kappa, tension = _history(600, amplitude=0.05)
        tension = 1.0e4 + 2.0e4 * np.sin(np.pi * np.arange(600) / 370.0) ** 2
        history = np.asarray(
            cx.compute_mkappa(cable, kappa, contact_tension_history_N=tension).moment_Nm
        )
        scaled = table.moment(kappa, tension)
        assert np.max(np.abs(scaled - history)) < 0.02 * np.ptp(history)
