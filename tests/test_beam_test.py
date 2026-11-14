# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Bending-test simulator (gap T1): closed forms, bitwise zone loop, energy."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.beam_test import (
    BendingProtocol,
    Cantilever,
    FourPoint,
    SectionLaw,
    ThreePoint,
    _legs,
    simulate_bending_test,
)
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.core_lay import bending_factor

EI = 2.0e3
L = 2.0


@pytest.fixture(scope="module")
def menard():
    return cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)


def _closed_form(arrangement, force):
    if isinstance(arrangement, ThreePoint):
        return force * L**3 / (48 * EI)
    if isinstance(arrangement, FourPoint):
        a = 0.5 * (L - arrangement.loading_span_m)
        return 0.5 * force * a**2 * (3 * L - 4 * a) / (6 * EI)
    return force * L**3 / (3 * EI)


@pytest.mark.parametrize(
    "arrangement", [ThreePoint(L), FourPoint(L, 1.0), Cantilever(L)]
)
@pytest.mark.parametrize("control", ["load", "displacement", "curvature", "moment"])
def test_elastic_closed_forms(arrangement, control):
    law = SectionLaw.elastic(EI)
    amplitude = {"load": 100.0, "displacement": 1e-3, "curvature": 0.01, "moment": 50.0}
    protocol = BendingProtocol.cyclic(
        control, amplitude[control], n_cycles=1, samples_per_cycle=40
    )
    r = simulate_bending_test(law, arrangement, protocol, stations_per_segment=5)
    exact = _closed_form(arrangement, r.load_N)
    scale = np.max(np.abs(exact))
    assert np.max(np.abs(r.displacement_m - exact)) <= 1e-12 * scale
    # elastic curvature definitions return the measuring-station curvature
    np.testing.assert_allclose(
        r.curvature_1_m,
        r.moment_Nm / EI,
        rtol=0,
        atol=1e-12 * np.max(np.abs(r.moment_Nm / EI)),
    )
    assert r.equilibrium_residual < 1e-12
    if control == "curvature":
        np.testing.assert_allclose(r.curvature_1_m, protocol.values, atol=1e-14)
    # elastic: no dissipation, external work equals stored energy
    assert r.friction_work_J is not None and np.max(np.abs(r.friction_work_J)) == 0.0


def test_three_point_mid_deflection_formula():
    """Hu Eq. 2: for an elastic beam 12 w / L^2 is the mid-span curvature."""
    law = SectionLaw.elastic(EI)
    p = BendingProtocol("load", np.array([10.0, 20.0]))
    r = simulate_bending_test(law, ThreePoint(L), p)
    np.testing.assert_allclose(r.curvature_1_m, r.load_N * L / 4 / EI, rtol=1e-13)
    np.testing.assert_allclose(r.deflection_m(0.5 * L), r.displacement_m, rtol=1e-13)


def test_section_law_bitwise_compute_mkappa(menard):
    k = 0.08 * np.sin(np.linspace(0, 4 * math.pi, 181))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kw in ({}, {"friction_model": "progressive"}, {"core_friction": True}):
            ref = cx.compute_mkappa(menard, k, mu=0.2, **kw)
            law = SectionLaw.from_cable(menard, mu=0.2, **kw)
            resp = law.evaluate(k)
            assert np.array_equal(resp.moment_Nm, ref.moment_Nm)
            assert np.array_equal(resp.friction_J_per_m, ref.work.friction_J_per_m)


def test_four_point_zone_is_the_section_loop_bitwise(menard):
    """Uniform moment between the loads: every zone station is the section law."""
    k = np.r_[
        np.linspace(0, 0.1, 21)[1:],
        np.linspace(0.1, -0.1, 41)[1:],
        np.linspace(-0.1, 0.1, 41)[1:],
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = cx.compute_mkappa(menard, k, mu=0.2)
        r = simulate_bending_test(
            menard,
            FourPoint(2.5, 1.25),
            BendingProtocol("curvature", k),
            mu=0.2,
            stations_per_segment=6,
            gauge_x_m=(0.9375, 1.25, 1.5625),
        )
    zone = (r.x_m >= 0.625 - 1e-12) & (r.x_m <= 1.875 + 1e-12)
    assert zone.sum() >= 7
    for i in np.flatnonzero(zone):
        assert np.array_equal(r.curvature_field_1_m[i], k)
        assert np.array_equal(r.moment_field_Nm[i], ref.moment_Nm)
    assert np.array_equal(r.station_moment_Nm, ref.moment_Nm)
    # the test's own reduction (Menard Eq. 1, LVDT parabola) reads the same loop
    np.testing.assert_allclose(
        r.moment_Nm, ref.moment_Nm, rtol=0, atol=1e-9 * np.ptp(ref.moment_Nm)
    )
    g = r.gauge_x_m
    vand = np.column_stack((g**2, g, np.ones_like(g)))
    fit = -2.0 * (np.linalg.pinv(vand) @ r.gauge_deflection_m)[0]
    np.testing.assert_allclose(fit, k, atol=1e-12)
    # the "fit" definition under curvature control gives the same loop
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_fit = simulate_bending_test(
            menard,
            FourPoint(2.5, 1.25),
            BendingProtocol("curvature", k),
            mu=0.2,
            stations_per_segment=6,
            curvature_definition="fit",
            gauge_x_m=(0.9375, 1.25, 1.5625),
        )
    np.testing.assert_allclose(r_fit.curvature_1_m, k, atol=1e-10 * 0.1)
    np.testing.assert_allclose(
        r_fit.moment_Nm, ref.moment_Nm, atol=1e-8 * np.ptp(ref.moment_Nm)
    )


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"friction_model": "progressive"},
        {"core_friction": True},
        {"friction_model": "distributed"},
    ],
)
def test_leg_property_and_energy(menard, kw):
    """Equilibrium of the final full-history run and energy convergence."""
    rel = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in (6, 24):
            r = simulate_bending_test(
                menard,
                ThreePoint(2.0),
                BendingProtocol.cyclic("load", 400.0, n_cycles=2, samples_per_cycle=80),
                mu=0.2,
                stations_per_segment=n,
                **kw,
            )
            assert r.equilibrium_residual < 1e-8
            ext, fr = r.cycle_external_work_J(), r.cycle_friction_work_J()
            assert fr is not None and fr > 0
            rel.append(abs(ext - fr) / fr)
            assert r.friction_work_J is not None and np.all(
                np.diff(r.friction_work_J) >= -1e-12
            )
    assert rel[1] < 0.01
    assert rel[1] < 0.5 * rel[0]


def _hu_law(cap):
    from check_conductor_strand_slip import HU_E

    g8 = pytest.importorskip("check_g8_gradual_reversal")
    HU_TENSION, _hu_setup = g8.HU_TENSION, g8._hu_setup

    rings, params, ei = _hu_setup()
    wire = sum(
        HU_E
        * r.n_wires
        * math.pi
        * r.d_wire_m**4
        / 64
        * bending_factor(r.lay_angle_rad)
        for r in rings
    )
    return SectionLaw.from_conductor(
        rings,
        HU_E,
        params,
        tension_N=HU_TENSION,
        additional_EI_Nm2=ei - wire,
        cap_shape=cap,
    )


@pytest.mark.parametrize(
    "cap,area,m_max", [("karlsen_husoy", 5.236, 8.827), ("papailiou", 4.354, 8.110)]
)
def test_hu_three_point_regression(cap, area, m_max):
    """Hu (2022) set-up: three-point, 440 mm, +-6 mm, third cycle (gap T1)."""
    law = _hu_law(cap)
    r = simulate_bending_test(
        law,
        ThreePoint(0.44),
        BendingProtocol.cyclic(
            "displacement", 0.006, n_cycles=3, samples_per_cycle=500
        ),
        stations_per_segment=24,
    )
    assert r.equilibrium_residual < 1e-10
    assert r.loop_work_J_per_m() == pytest.approx(area, abs=2e-3)
    assert float(np.max(r.moment_Nm[r.retained])) == pytest.approx(m_max, abs=2e-3)
    # reduced curvature amplitude is Hu Eq. 2 at 6 mm
    assert float(np.max(r.curvature_1_m)) == pytest.approx(
        12 * 0.006 / 0.44**2, rel=1e-9
    )
    # the uniform-curvature section loop at the same amplitude is smaller
    k = r.curvature_1_m
    m_uniform = law.moment(k)
    uniform = abs(
        np.trapezoid(
            np.r_[m_uniform[r.retained], m_uniform[r.retained][:1]],
            np.r_[k[r.retained], k[r.retained][:1]],
        )
    )
    assert uniform < r.loop_work_J_per_m()


def test_conductor_law_bounds():
    law = _hu_law("karlsen_husoy")
    k = np.r_[0.0, 1e-7, np.linspace(0.01, 3.0, 400)]
    m = law.moment(k)
    slope_end = (m[-1] - m[-2]) / (k[-1] - k[-2])
    slope_start = m[1] / k[1]
    assert slope_end == pytest.approx(law.ei_slip_Nm2, rel=1e-9)
    assert slope_start == pytest.approx(law.ei_stick_Nm2, rel=1e-9)


def test_legs():
    v = np.array([1.0, 2.0, 2.0, 1.0, -1.0, -1.0, 0.0, 3.0])
    assert _legs(v) == [(0, 3), (3, 6), (6, 8)]
    assert _legs(np.array([0.0, 0.0, 1.0])) == [(0, 3)]


def test_input_validation(menard):
    law = SectionLaw.elastic(EI)
    p = BendingProtocol.cyclic("load", 1.0, n_cycles=1, samples_per_cycle=8)
    with pytest.raises(CableInputError):
        FourPoint(1.0, 1.0)._beam()
    with pytest.raises(CableInputError):
        simulate_bending_test(law, ThreePoint(1.0), p, curvature_definition="fit")
    with pytest.raises(CableInputError):
        simulate_bending_test(law, ThreePoint(1.0), p, mu=0.2)
    with pytest.raises(CableInputError):
        BendingProtocol("force", np.array([1.0, 2.0]))  # type: ignore[arg-type]
    with pytest.raises(CableInputError):
        SectionLaw.elastic(-1.0)
    with pytest.raises(CableInputError):
        SectionLaw(lambda k: None, ei_stick_Nm2=1.0, ei_slip_Nm2=2.0, description="")  # type: ignore[arg-type,return-value]
    with pytest.raises(CableInputError):
        SectionLaw.from_cable(menard, friction_model="sticky")  # type: ignore[arg-type]
    with pytest.raises(CableInputError):
        simulate_bending_test(law, ThreePoint(1.0), p, gauge_x_m=(2.0,))
