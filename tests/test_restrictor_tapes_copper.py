# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Gaps P5, P2, P7: bend restrictor and stiffener fields, helical tapes,
copper plasticity through strain life, and the copper_stranded friction label.

Closed forms (rigid-link chain, tensioned beam, Coulomb chain), limits of the
tape states and equivalence of the new non-proportional solvers with the
proportional ones on proportional fields; defaults are unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.beam_test import (
    BendingProtocol,
    Cantilever,
    SectionLaw,
    simulate_bending_test,
)
from cablexsec.analysis.rig_test import end_rotation_profile
from cablexsec.analysis.tape import compute_tape_fatigue, tape_geometry, tape_strain
from cablexsec.analysis.termination import (
    TerminationSpec,
    termination_friction_increment,
)
from cablexsec.cables.schema import TapeWinding
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.materials import library
from cablexsec.materials.strain_life import get_strain_life
from cablexsec.mechanics.bend_restrictor import (
    ArticulatedRestrictor,
    BendStiffener,
    CurvatureField,
    restrictor_curvature_field,
    stiffener_curvature_field,
    tensioned_beam_profile,
)
from cablexsec.mechanics.clamp_layer import (
    anchored_wire_field_response,
    anchored_wire_response,
)
from cablexsec.mechanics.mkappa import HelicalLayerSpec
from cablexsec.provenance import Provenance

pytestmark = pytest.mark.analytical_sanity

THETA_L = math.radians(4.75)


def _restrictor(**kw: object) -> ArticulatedRestrictor:
    base: dict[str, object] = {
        "pitch_m": 0.36,
        "lock_angle_rad": THETA_L,
        "n_elements": 15,
        "bore_diameter_m": 0.1375,
    }
    base.update(kw)
    return ArticulatedRestrictor(**base)  # type: ignore[arg-type]


def _field(x: np.ndarray, kappa: np.ndarray) -> CurvatureField:
    n = kappa.shape[1]
    return CurvatureField(
        x,
        kappa,
        None,
        "test",
        np.zeros(n),
        np.zeros(n, dtype=np.int64),
        np.zeros(n, dtype=bool),
        math.inf,
        np.zeros(n),
        Provenance(method="test"),
    )


@pytest.fixture(scope="module")
def thies():
    return cx.load_cable(cx.thies2016_yaml_path(), geometry_override=None)


# --------------------------------------------------------------------------- #
# P5: rigid-link chain geometry and the Thies restrictor data                 #
# --------------------------------------------------------------------------- #
def test_rigid_link_chain_geometry_and_thies_discrepancy():
    r = _restrictor(stated_mbr_m=2.54)
    p = 0.36
    assert r.locking_radius_m == pytest.approx(p / (2 * math.sin(THETA_L / 2)))
    assert r.arc_locking_radius_m == pytest.approx(p / THETA_L)
    # Thies (2016) Table 1: 4.75 deg and pitch 0.36 m give 4.34 m, not 2.54 m
    assert r.locking_radius_m == pytest.approx(4.344, abs=1e-3)
    assert r.length_m == pytest.approx(5.4)
    summary = r.summary(0.123)
    assert summary["lock_angle_from_stated_mbr_deg"] == pytest.approx(8.127, abs=1e-3)
    assert summary["clearance_m"] == pytest.approx(0.00725)
    from_mbr = ArticulatedRestrictor.from_mbr(
        2.54, pitch_m=p, n_elements=15, bore_diameter_m=0.1375
    )
    assert from_mbr.locking_radius_m == pytest.approx(2.54)
    # polygon vs arc differ by theta^2 / 24
    assert r.locking_radius_m / r.arc_locking_radius_m - 1 == pytest.approx(
        THETA_L**2 / 24, rel=1e-3
    )
    f = restrictor_curvature_field(
        r, [0.4], cable_od_m=0.123, bending_stiffness_Nm2=5e3, tension_N=2e4
    )
    assert any("stated MBR" in note for note in f.notes)


def test_cable_locking_radius_band():
    r = _restrictor()
    c = r.clearance_m(0.123)
    lower = r.locking_radius_m - c
    upper = r.locking_radius_m * math.cos(THETA_L / 2) + c
    assert r.cable_locking_radius_m(0.123) == pytest.approx(upper)
    assert r.cable_locking_radius_m(0.123, 0.01) == pytest.approx(lower)
    mid = 0.5 * (lower + upper) * math.sqrt(2.0)
    assert r.cable_locking_radius_m(0.123, mid) == pytest.approx(mid / math.sqrt(2.0))
    with pytest.raises(CableInputError):
        r.clearance_m(0.2)


def test_unlocked_field_is_the_tensioned_beam():
    x = np.linspace(0.0, 5.4, 301)
    for lam in (0.5, math.inf):
        expected = -0.02 * end_rotation_profile(x, 5.4, lam)
        assert np.allclose(tensioned_beam_profile(x, 5.4, lam, 0.02), expected)
    r = _restrictor()
    f = restrictor_curvature_field(
        r,
        [0.02, -0.02],
        cable_od_m=0.123,
        bending_stiffness_Nm2=5e3,
        tension_N=2e4,
        span_m=5.4,
        x_m=x,
    )
    lam = math.sqrt(5e3 / 2e4)
    assert np.allclose(f.kappa_1_m[:, 0], -0.02 * end_rotation_profile(x, 5.4, lam))
    assert np.array_equal(f.kappa_1_m[:, 1], -f.kappa_1_m[:, 0])
    assert f.n_locked.tolist() == [0, 0]


def test_chain_limit_locks_one_element_per_lock_angle():
    """Short decay length: the restrictor behaves as a chain."""
    r = _restrictor()
    for deg in (5.0, 10.0, 20.0, 28.0):
        f = restrictor_curvature_field(
            r,
            [math.radians(deg)],
            cable_od_m=0.123,
            bending_stiffness_Nm2=1.0,
            tension_N=1e6,
            span_m=1e4,
            x_m=np.linspace(0.0, 10.0, 101),
        )
        k_p = f.lock_curvature_1_m * r.pitch_m
        assert f.n_locked[0] == math.ceil(math.radians(deg) / k_p) - 1
        assert f.locked_length_m[0] == pytest.approx(f.n_locked[0] * r.pitch_m)


def test_smeared_lock_front_closed_form():
    r = _restrictor(n_elements=60)
    lam = math.sqrt(5e3 / 2e5)
    theta = math.radians(28.0)
    f = restrictor_curvature_field(
        r,
        [theta],
        cable_od_m=0.123,
        bending_stiffness_Nm2=5e3,
        tension_N=2e5,
        span_m=1e4,
        x_m=np.linspace(0.0, 20.0, 2001),
        rule="smeared",
    )
    k_l = f.lock_curvature_1_m
    # finite span: the far-end deflection condition adds ~ w(a) / span
    assert f.locked_length_m[0] == pytest.approx(theta / k_l - lam, rel=5e-4)
    # cable moment continuous at the front: curvature equals the lock value
    beyond = f.x_m > f.locked_length_m[0]
    x1 = float(f.x_m[beyond][0])
    assert float(f.kappa_1_m[beyond, 0][0]) == pytest.approx(
        k_l * math.exp(-(x1 - f.locked_length_m[0]) / lam), rel=2e-3
    )
    # no lock below kappa_L lambda
    small = restrictor_curvature_field(
        r,
        [0.9 * k_l * lam],
        cable_od_m=0.123,
        bending_stiffness_Nm2=5e3,
        tension_N=2e5,
        span_m=1e4,
        rule="smeared",
    )
    assert small.locked_length_m[0] == 0.0


def test_bent_length_grows_as_joints_lock_and_tip_overbend():
    r = _restrictor(n_elements=5)
    angles = np.radians([2.0, 8.0, 14.0, 20.0, 28.0])
    f = restrictor_curvature_field(
        r,
        angles,
        cable_od_m=0.123,
        bending_stiffness_Nm2=3e3,
        tension_N=2e4,
        span_m=6.0,
    )
    assert np.all(np.diff(f.n_locked) >= 0)
    assert f.n_locked[0] == 0 and f.n_locked[-1] == 5
    assert bool(f.tip_overbend[-1])
    assert any("over-bending" in note for note in f.notes)


def test_bent_length_ignores_reverse_bending_and_bore_limit_note():
    x = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    k = np.array([[1.0], [0.8], [0.4], [0.0], [-0.9]])
    assert _field(x, k).bent_length_m(0.5)[0] == pytest.approx(0.5)
    r = _restrictor(n_elements=3, bore_diameter_m=0.124)
    f = restrictor_curvature_field(
        r,
        [math.radians(10.0)],
        cable_od_m=0.123,
        bending_stiffness_Nm2=1e3,
        tension_N=1e5,
    )
    assert any("straight-bore limit" in note for note in f.notes)


def test_bend_stiffener_closed_form_and_stiffening():
    ei, ten = 5.0e3, 2.0e4
    lam = math.sqrt(ei / ten)
    angle = math.radians(1.0)
    bare = stiffener_curvature_field(
        None, [angle], cable_od_m=0.1, bending_stiffness_Nm2=ei, tension_N=ten
    )
    x = bare.x_m
    assert np.allclose(
        bare.kappa_1_m[:, 0], angle / lam * np.exp(-x / lam), atol=2e-4 * angle / lam
    )
    cone = BendStiffener(length_m=2.0, root_od_m=0.5, tip_od_m=0.15, E_Pa=45e6)
    stiff = stiffener_curvature_field(
        cone,
        np.radians([10.0, -10.0]),
        cable_od_m=0.1,
        bending_stiffness_Nm2=ei,
        tension_N=1e5,
    )
    plain = stiffener_curvature_field(
        None,
        np.radians([10.0]),
        cable_od_m=0.1,
        bending_stiffness_Nm2=ei,
        tension_N=1e5,
    )
    assert stiff.peak() < 0.5 * plain.peak()
    assert np.allclose(stiff.kappa_1_m[:, 1], -stiff.kappa_1_m[:, 0])
    with pytest.raises(CableInputError):
        BendStiffener(length_m=1.0, root_od_m=0.1, tip_od_m=0.2, E_Pa=1e7)


# --------------------------------------------------------------------------- #
# P5: solvers with a non-proportional field                                    #
# --------------------------------------------------------------------------- #
def test_field_kernel_equals_proportional_kernel():
    m = 64
    h = 2 * math.pi / m
    theta = 2 * math.pi * np.arange(3 * m) / m
    shape = np.exp(-np.arange(3 * m) * h / 6.0)
    e_hat = shape * np.cos(theta)
    load = 6.0 * np.sin(np.linspace(0.0, 4 * math.pi, 41))[1:]
    counts = np.full(load.size, 30)
    fine = np.concatenate(
        [
            np.linspace(a, b, c + 1)[1:]
            for a, b, c in zip(np.r_[0.0, load[:-1]], load, counts, strict=True)
        ]
    )
    record = np.zeros(fine.size, dtype=bool)
    record[np.cumsum(counts) - 1] = True
    n_p, r_p, _ = anchored_wire_response(
        e_hat, fine, h, cap_hat=np.abs(np.cos(theta)), record=record
    )
    n_f, r_f, _ = anchored_wire_field_response(
        load[:, None] * shape[None, :],
        np.cos(theta),
        h,
        cap_hat=np.abs(np.cos(theta)),
        substeps=counts,
    )
    assert np.allclose(n_f, n_p, atol=1e-10)
    assert np.allclose(r_f, r_p, atol=1e-10)


def test_rig_field_profile_reproduces_table_profile(thies):
    length = 5.4
    prot = cx.RigProtocol.cyclic(
        tension_N=0.0, curvature_amplitude_1_m=0.3, n_cycles=1, samples_per_cycle=24
    )
    x = np.array([0.0, 1.0, 1.5, length])
    s = np.array([1.0, 1.0, 0.0, 0.0])
    gauges = [cx.GaugeLocation("armour", a, xx) for a in (0, 90) for xx in (0.05, 0.6)]
    table = cx.simulate_rig_test(
        thies,
        prot,
        cx.RigSetup(
            free_length_m=length,
            profile="table",
            profile_x_m=x,
            profile_shape=s,
            reference_x_m=0.5,
        ),
        gauges=gauges,
        contact_model="bounded",
        mu_factors=(),
    )
    xf = np.linspace(0.0, length, 541)
    field = _field(xf, np.interp(xf, x, s)[:, None] * prot.curvature_1_m[None, :])
    run = cx.simulate_rig_test(
        thies,
        prot,
        cx.RigSetup(
            free_length_m=length,
            profile="field",
            curvature_field=field,
            reference_x_m=0.5,
        ),
        gauges=gauges,
        contact_model="bounded",
        mu_factors=(),
    )
    for a, b in zip(table.gauges, run.gauges, strict=True):
        assert b.ranges["total"] == pytest.approx(a.ranges["total"], rel=1e-6, abs=1e-6)
        assert b.range_sectional == pytest.approx(a.range_sectional, rel=1e-9)
    with pytest.raises(CableInputError):
        cx.RigSetup(free_length_m=length, profile="field")
    bad = cx.RigProtocol.cyclic(
        tension_N=0.0, curvature_amplitude_1_m=0.2, n_cycles=1, samples_per_cycle=24
    )
    with pytest.raises(CableInputError, match="reference station"):
        cx.simulate_rig_test(
            thies,
            bad,
            cx.RigSetup(
                free_length_m=length,
                profile="field",
                curvature_field=field,
                reference_x_m=0.5,
            ),
            gauges=gauges,
            contact_model="bounded",
            mu_factors=(),
        )


def test_rig_runs_on_a_restrictor_field(thies):
    r = ArticulatedRestrictor.from_mbr(
        2.54, pitch_m=0.36, n_elements=15, bore_diameter_m=0.1375
    )
    t = np.linspace(0.0, 2.0, 49)
    rotation = math.radians(28.0) * np.sin(2 * np.pi * t)
    field = restrictor_curvature_field(
        r,
        rotation,
        cable_od_m=0.123,
        bending_stiffness_Nm2=4.0e3,
        tension_N=2.0e4,
        span_m=5.4,
        x_m=np.linspace(0.0, 5.4, 541),
    )
    prot = cx.RigProtocol.from_curvature_field(
        field, time_s=t, tension_N=0.0, reference_x_m=0.0, retain_from_s=1.0
    )
    res = cx.simulate_rig_test(
        thies,
        prot,
        cx.RigSetup(
            free_length_m=5.4, profile="field", curvature_field=field, reference_x_m=0.0
        ),
        gauges=[cx.GaugeLocation("armour", 90.0, 0.02)],
        contact_model="bounded",
        mu_factors=(),
    )
    assert res.gauges[0].ranges["total"] > res.gauges[0].range_sectional
    assert any("non-proportional" in note for note in res.notes)


def test_termination_with_uniform_field_equals_default():
    spec = HelicalLayerSpec(
        n_wires=40,
        r_helix_m=0.05,
        lay_angle_rad=math.radians(15),
        E_wire_Pa=210e9,
        A_wire_m2=math.pi * 0.004**2 / 4,
        xi=0.8,
        mu=0.2,
        F_n_axi_N=0.0,
        contact_pressure_Pa=1e6,
        normal_load_is_line=True,
    )
    t = np.linspace(0, 1, 41)
    kx = 0.2 * np.sin(2 * np.pi * t)
    ky = np.zeros_like(kx)
    xs = np.linspace(0.0, 20.0, 801)
    field = _field(xs, np.repeat(kx[None, :], xs.size, axis=0))
    for solver in ("clamp", "along_wire"):
        a = termination_friction_increment(
            spec, kx, ky, TerminationSpec(distance_m=0.3, solver=solver)
        )
        b = termination_friction_increment(
            spec,
            kx,
            ky,
            TerminationSpec(distance_m=0.3, solver=solver, curvature_field=field),
        )
        scale = float(np.max(np.abs(a.stress_Pa)))
        assert scale > 0
        assert np.max(np.abs(a.stress_Pa - b.stress_Pa)) <= 1e-9 * scale
    with pytest.raises(CableInputError):
        termination_friction_increment(
            spec,
            2 * kx,
            ky,
            TerminationSpec(distance_m=0.3, curvature_field=field),
        )
    with pytest.raises(CableInputError):
        TerminationSpec(
            distance_m=0.3,
            curvature_field=field,
            curvature_profile=(np.array([0.0, 1.0]), np.array([1.0, 1.0])),
        )


def test_bending_test_restrictor_caps_curvature():
    ei, length = 1.0e4, 3.0
    r = ArticulatedRestrictor(
        pitch_m=0.2,
        lock_angle_rad=math.radians(5.0),
        n_elements=10,
        bore_diameter_m=0.11,
    )
    law = SectionLaw.elastic(ei)
    loads = np.array([1000.0, 3000.0, 6000.0, 3000.0, 0.0])
    res = simulate_bending_test(
        law,
        Cantilever(length),
        BendingProtocol("load", loads),
        restrictor=r,
        cable_od_m=0.1,
    )
    k_l = 1.0 / r.cable_locking_radius_m(0.1)
    x = res.x_m
    for k, force in enumerate(loads):
        free = force * (length - x) / ei
        expected = np.where(x <= r.length_m + 1e-12, np.minimum(free, k_l), free)
        assert np.allclose(
            res.curvature_field_1_m[:, k], expected, rtol=1e-9, atol=1e-12
        )
    assert res.restrictor_moment_field_Nm is not None
    m_r = res.restrictor_moment_field_Nm[:, 2]
    locked = (6000.0 * (length - x) / ei > k_l) & (x <= r.length_m + 1e-12)
    assert np.allclose(m_r, np.where(locked, 6000.0 * (length - x) - ei * k_l, 0.0))
    assert res.equilibrium_residual < 1e-9
    plain = simulate_bending_test(
        law, Cantilever(length), BendingProtocol("load", loads)
    )
    assert plain.restrictor_moment_field_Nm is None
    with pytest.raises(CableInputError):
        simulate_bending_test(
            law,
            Cantilever(length),
            BendingProtocol("curvature", loads * 1e-4),
            restrictor=r,
            cable_od_m=0.1,
        )


# --------------------------------------------------------------------------- #
# P2: helical tapes                                                            #
# --------------------------------------------------------------------------- #
def test_tape_geometry_open_helix_and_overlap():
    w = TapeWinding(width_m=0.02, lay_angle_deg=50.0, source="test")
    g = tape_geometry(0.02, 1e-4, w)
    beta = math.radians(50.0)
    assert g.pitch_m == pytest.approx(2 * math.pi * 0.02 / math.tan(beta))
    assert g.axial_gap_m == pytest.approx(g.pitch_m - 0.02 / math.sin(beta))
    assert g.open_helix and g.overlap_fraction == 0.0
    lapped = tape_geometry(
        0.02, 1e-4, TapeWinding(width_m=0.1, lay_angle_deg=50.0, source="t")
    )
    assert lapped.axial_gap_m < 0 and lapped.overlap_fraction > 0
    assert g.edge_angle_rad == pytest.approx(0.01 * math.cos(beta) / 0.02)
    with pytest.raises(ValueError):
        TapeWinding(width_m=0.02, lay_angle_deg=95.0, source="t")


def test_tape_states_closed_forms():
    w = TapeWinding(width_m=0.02, lay_angle_deg=50.0, source="test")
    g = tape_geometry(0.02, 1e-4, w)
    k = 0.3 * np.sin(np.linspace(0.0, 2 * math.pi, 65))
    eps0 = 1e-4 * np.ones_like(k)
    theta = 2 * math.pi * np.arange(16) / 16
    beta = math.radians(50.0)
    h = -0.45
    factor = math.cos(beta) ** 2 + h * math.sin(beta) ** 2
    stuck, slip_s = tape_strain(
        g,
        eps0[None],
        k[None],
        np.zeros_like(k)[None],
        theta,
        state="stuck",
        hoop_ratio=h,
    )
    # stuck centre-line outer fibre: surface strain at radius rho + t/2
    expected = factor * (
        eps0[None, :]
        + (g.radius_m + g.thickness_m / 2) * np.cos(theta)[:, None] * k[None, :]
    )
    assert np.allclose(stuck[0, :, 0, :], expected)
    assert np.allclose(slip_s, 0.0)
    free, slip_f = tape_strain(
        g,
        eps0[None],
        k[None],
        np.zeros_like(k)[None],
        theta,
        state="free",
        hoop_ratio=h,
    )
    own = g.thickness_m / 2 * math.cos(beta) ** 2 * math.cos(2 * beta) * np.cos(theta)
    assert np.allclose(
        free[0, :, 0, :], factor * eps0[None, :] + own[:, None] * k[None, :]
    )
    # free slip amplitude: rho^2 kappa |factor| / sin(beta)
    amp = 0.5 * np.ptp(slip_f[0], axis=1).max()
    assert amp == pytest.approx(
        g.radius_m**2 * 0.3 * abs(factor) / math.sin(beta), rel=0.03
    )
    # frictional: zero pressure is the free tape, huge pressure the stuck tape
    lo, _ = tape_strain(
        g,
        eps0[None],
        k[None],
        np.zeros_like(k)[None],
        theta,
        state="frictional",
        hoop_ratio=h,
        E_Pa=117e9,
        contact_pressure_Pa=0.0,
        mu=0.2,
    )
    hi, _ = tape_strain(
        g,
        eps0[None],
        k[None],
        np.zeros_like(k)[None],
        theta,
        state="frictional",
        hoop_ratio=h,
        E_Pa=117e9,
        contact_pressure_Pa=1e12,
        mu=0.2,
    )
    assert np.allclose(lo, free)
    centre = hi[0, :, 0, :] - own[:, None] * k[None, :]
    stuck_mid = factor * (
        eps0[None, :] + g.radius_m * np.cos(theta)[:, None] * k[None, :]
    )
    assert np.allclose(centre, stuck_mid, atol=1e-12)
    lox, _ = tape_strain(
        g,
        eps0[None],
        k[None],
        np.zeros_like(k)[None],
        theta,
        state="free",
        hoop_ratio=h,
        path="loxodromic",
    )
    assert np.max(np.abs(lox[0, :, 2, :])) > np.max(np.abs(free[0, :, 2, :]))
    with pytest.raises(CableInputError):
        tape_strain(
            g, eps0[None], k[None], k[None], theta, state="frictional", hoop_ratio=h
        )


def test_core_tape_follows_core_slip_state(thies):
    k = (1 / 2.54) * np.cos(2 * np.pi * np.linspace(0.0, 2.0, 193))
    free = compute_tape_fatigue(
        thies, "cu_tape", k, state="stuck", core_state="free", n_stations=8
    )
    stuck = compute_tape_fatigue(
        thies, "cu_tape", k, state="stuck", core_state="stuck", n_stations=8
    )
    assert stuck.strain_amplitude_max > free.strain_amplitude_max
    assert stuck.D_miner > free.D_miner > 0
    assert free.constants.name == "copper_poon2022"
    assert free.hoop_ratio == pytest.approx(-library.get_property("hdpe", "nu"))
    sliding = compute_tape_fatigue(
        thies, "cu_tape", k, state="free", core_state="free", n_stations=8
    )
    assert sliding.D_miner < free.D_miner
    assert float(np.max(sliding.slip_amplitude_m)) > 0
    with pytest.raises(CableInputError):
        compute_tape_fatigue(thies, "cu_tape", k, state="stuck")
    with pytest.raises(CableSchemaError):
        compute_tape_fatigue(thies, "al_laminate", k, state="stuck", core_state="free")


# --------------------------------------------------------------------------- #
# P7: copper plasticity, constants and labels                                  #
# --------------------------------------------------------------------------- #
def test_copper_strain_life_sets_registered_with_sources():
    poon = get_strain_life("copper_poon2022")
    derived = get_strain_life("copper_annealed_mus_derived")
    assert derived.basis == "derived" and poon.basis == "published"
    assert "[Wan2022]" in derived.reference and "[Jiang2023]" in derived.reference
    k_c, n_c = poon.cyclic_curve
    assert n_c == pytest.approx(poon.b / poon.c)
    # the two independent copper estimates agree within a decade at 0.2 %
    n1 = float(poon.cycles_to_failure(0.002))
    n2 = float(derived.cycles_to_failure(0.002))
    assert 0.1 < n1 / n2 < 10.0


def test_screen_strain_life_route_default_unchanged(thies):
    k = (1 / 2.54) * np.cos(2 * np.pi * np.linspace(0.0, 2.0, 193))
    kwargs = {"core_state": "free", "duration_s": 2.0, "mu": 0.2, "n_stations": 2}
    base = cx.compute_core_screen_fatigue(thies, k, **kwargs)
    life = cx.compute_core_screen_fatigue(
        thies, k, strain_life="copper_poon2022", **kwargs
    )
    assert base.D_strain_life is None and base.strain_life_constants == ""
    assert np.array_equal(base.D_bulk, life.D_bulk)
    assert np.array_equal(base.stress_range_Pa, life.stress_range_Pa)
    assert life.D_strain_life is not None
    assert life.D_strain_life.shape == base.D_bulk.shape
    assert np.all(life.D_strain_life >= 0)
    assert np.max(life.D_strain_life) > 0
    assert any("strain-life route" in x for x in life.provenance.limitations)


def test_copper_stranded_friction_is_a_labelled_assumption():
    value = library.get_property("copper_stranded", "mu_on_cu")
    citation = library.get_citation("copper_stranded", "mu_on_cu")
    assert value == 0.25
    assert citation.startswith("[ModelAssumption]")
    assert "0.1-0.5" in citation and "no 0.25" in citation
