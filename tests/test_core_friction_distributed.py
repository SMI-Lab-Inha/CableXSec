# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Distributed laid-core law and core contact loads that follow the tension.

Rows CF-D and CF-T of ``doc/benchmarks.md`` (laid-core friction, gap R1b),
registered before the run. Verification only: limits, closed forms, energy,
invariance and identities of the scaling.
"""

from __future__ import annotations

import dataclasses
import math
import warnings

import numpy as np
import pytest

import cablexsec as cx
import cablexsec.mechanics.core_friction as core_friction
from cablexsec._numba import kernels_compiled
from cablexsec.analysis.bending_law import export_bending_law
from cablexsec.analysis.neutral_axis import neutral_axes
from cablexsec.api import CoreFrictionSpec
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.mechanics import distributed_friction as dfm
from cablexsec.mechanics.core_friction import (
    build_core_friction,
    core_friction_history,
    core_load_history,
)
from cablexsec.mechanics.papailiou import slip_onset_curvature
from cablexsec.provenance import FrictionModelFallbackWarning

pytestmark = [
    pytest.mark.analytical_sanity,
    pytest.mark.filterwarnings(
        "ignore::cablexsec.provenance.MissingClampingStateWarning"
    ),
]

PI = math.pi
T = 50.0e3
SPEC = CoreFrictionSpec(
    mu_core_filler=0.46, mu_core_core=0.46, mu_core_sheath=0.46, filler_profiles=True
)
#: Interpreted kernels (no numba) take shorter histories and fewer nodes.
COMPILED = kernels_compiled()
NODES = None if COMPILED else 180
compiled = pytest.mark.skipif(
    not COMPILED, reason="full-slip and fine-node runs need the compiled kernel"
)


@pytest.fixture(scope="module")
def iea66():
    return cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)


@pytest.fixture(scope="module")
def model(iea66):
    return build_core_friction(iea66, SPEC, T)


def _wound(cable, sigma_Pa):
    layers = tuple(
        x.model_copy(update={"winding_stress_Pa": sigma_Pa})
        if x.name in {"inner_armour", "outer_armour"}
        else x
        for x in cable.layers
    )
    return cable.model_copy(update={"layers": layers})


def _walk(seed, size, scale):
    rng = np.random.default_rng(seed)
    return (
        np.cumsum(rng.normal(scale=scale, size=size)),
        np.cumsum(rng.normal(scale=scale, size=size)),
    )


def _short_walk(seed=11):
    return _walk(seed, 300, 0.01) if COMPILED else _walk(seed, 30, 2e-3)


def _loop(amplitude, n=401, cycles=2):
    return amplitude * np.sin(np.linspace(0.0, 2 * cycles * PI, n))


def _scale(history):
    return float(np.max(np.abs(history.moment_Nm)))


# --------------------------------------------------------------------------- #
# CF-D: distributed core law                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("azimuth", [0.0, 0.7])
def test_cf_d_planar_auto_is_the_direction_tied_law(model, iea66, azimuth):
    k = _loop(0.05)
    kx, ky = k * math.cos(azimuth), k * math.sin(azimuth)
    tied = core_friction_history(model, kx, ky, initial_bending_angle=azimuth)
    auto = core_friction_history(
        model, kx, ky, initial_bending_angle=azimuth, friction_model="auto"
    )
    assert auto.friction_model == "direction_tied"
    scale = _scale(tied)
    assert np.max(np.abs(auto.moment_Nm - tied.moment_Nm)) <= 1e-12 * scale
    work = tied.work.friction_J_per_m
    assert np.max(np.abs(auto.work.friction_J_per_m - work)) <= 1e-12 * work[-1]
    tension = np.full(k.size, T)
    section = {
        model_name: cx.compute_neutral_axes(
            iea66,
            kx,
            ky,
            tension_N=tension,
            friction_model=model_name,
            core_friction=SPEC,
        )
        for model_name in ("direction_tied", "auto")
    }
    a, b = section["auto"].moment_Nm, section["direction_tied"].moment_Nm
    assert np.max(np.abs(a - b)) <= 1e-12 * np.max(np.abs(b))
    assert "(auto)" in section["auto"].provenance.method


def test_cf_d_rotation_equivariance(model):
    kx, ky = _short_walk()
    base = core_friction_history(
        model, kx, ky, friction_model="distributed", nodes_per_pitch=NODES
    )
    assert base.friction_model == "distributed"
    scale = _scale(base)
    n_st = model.population.n_wires
    for angle, tol in ((2 * PI * 7 / n_st, 1e-9), (0.3, 1e-3)):
        c, s = math.cos(angle), math.sin(angle)
        rot = core_friction_history(
            model,
            c * kx - s * ky,
            s * kx + c * ky,
            friction_model="distributed",
            nodes_per_pitch=NODES,
        )
        expected = base.moment_Nm @ np.array([[c, s], [-s, c]])
        assert np.max(np.abs(rot.moment_Nm - expected)) <= tol * scale
        assert rot.work.friction_J_per_m[-1] == pytest.approx(
            base.work.friction_J_per_m[-1], rel=tol
        )


def test_cf_d_dissipation_is_nonnegative_and_work_balances(model):
    steps = 400 if COMPILED else 40
    t = np.linspace(0.0, 4 * PI, steps)
    radius = (0.03 if COMPILED else 3e-3) * np.minimum(1.0, t / PI)
    for kx, ky in (_short_walk(3), (radius * np.cos(t), radius * np.sin(t))):
        hist = core_friction_history(
            model, kx, ky, friction_model="distributed", nodes_per_pitch=NODES
        )
        friction = hist.work.friction_J_per_m
        assert friction[-1] > 0.0
        assert np.all(np.diff(friction) >= -1e-12 * friction[-1])
        residual = hist.work.balance_residual_J_per_m
        assert np.max(np.abs(residual)) <= 1e-9 * friction[-1]


def test_cf_d_zero_friction_gives_the_armour_only_section(iea66):
    zero = CoreFrictionSpec(
        mu_core_filler=0.0, mu_core_core=0.0, mu_core_sheath=0.0, filler_profiles=True
    )
    kx, ky = _walk(5, 120, 1e-3) if COMPILED else _walk(5, 12, 1e-4)
    tension = np.full(kx.size, T)
    options = {"tension_N": tension, "friction_model": "distributed"}
    base = cx.compute_neutral_axes(iea66, kx, ky, **options)
    cored = cx.compute_neutral_axes(iea66, kx, ky, core_friction=zero, **options)
    assert np.array_equal(cored.core_moment_Nm, np.zeros((kx.size, 2)))
    assert np.array_equal(cored.moment_Nm, base.moment_Nm)


def test_cf_d_large_friction_tangent_is_the_stick_stiffness(iea66):
    stuck = CoreFrictionSpec(
        mu_core_filler=1e3, mu_core_core=1e3, mu_core_sheath=1e3, filler_profiles=True
    )
    model = build_core_friction(iea66, stuck, T)
    assert 0.0 < model.xi < 1.0 and 0.0 < model.fillers.xi < 1.0
    t = np.linspace(0.0, 2 * PI, 41)[1:]
    kx, ky = 1e-6 * np.cos(t), 1e-6 * np.sin(t)
    hist = core_friction_history(model, kx, ky, friction_model="distributed")
    tangent = np.hypot(*hist.moment_Nm.T) / np.hypot(kx, ky)
    assert np.allclose(tangent, model.stick_EI_Nm2, rtol=1e-5, atol=0.0)
    # No slip: the friction work is round-off of the stored energy.
    stored = float(np.max(hist.work.stored_J_per_m))
    assert np.max(np.abs(hist.work.friction_J_per_m)) <= 1e-12 * stored


def _closed_form(n, q, r, beta, mu):
    lam = mu * math.sin(beta)
    c = lam * PI / (2.0 * math.sinh(lam * PI / 2.0))
    return (
        2.0
        / PI
        * n
        * q
        * r
        * r
        * math.cos(beta) ** 2
        / math.sin(beta) ** 2
        * lam
        * c
        * math.cosh(lam * PI / 2.0)
        / (1.0 + lam * lam)
    )


def _station_sum(population, n, q):
    """Moment of the closed-form full-slip field summed over the stations."""
    lam = population.mu * math.sin(population.lay_angle_rad)
    c = lam * PI / (2.0 * math.sinh(lam * PI / 2.0))
    r, beta = population.r_helix_m, population.lay_angle_rad
    area = population.A_wire_m2 * population.n_wires / n
    sigma0 = q * math.cos(beta) * r / (area * math.sin(beta) ** 2)
    phi = population.phi_wires
    x = PI / 2 - np.abs(np.angle(np.exp(1j * phi)))
    stress = sigma0 * (c * np.exp(lam * x) - 1.0)
    weight = population.A_wire_m2 * math.cos(beta) * r
    return float(weight * np.sum(stress * np.cos(phi)))


def _full_slip(model):
    fillers = model.fillers
    onset = max(
        slip_onset_curvature(model.population),
        slip_onset_curvature(fillers.population),
    )
    kappa = np.linspace(0.0, 1000.0 * onset, 301)[1:]
    hist = core_friction_history(
        model, kappa, np.zeros_like(kappa), friction_model="distributed"
    )
    cases = (
        (
            hist.core_moment_Nm[-1, 0],
            model.population,
            model.n_cores,
            model.loads.total_N_per_m,
        ),
        (
            hist.filler_moment_Nm[-1, 0],
            fillers.population,
            fillers.n_profiles,
            fillers.line_load_N_per_m,
        ),
    )
    out = []
    for moment, population, n, q in cases:
        closed = _closed_form(
            n, q, population.r_helix_m, population.lay_angle_rad, population.mu
        )
        out.append((moment / closed - 1.0, moment / _station_sum(population, n, q) - 1))
    assert abs(hist.core_moment_Nm[-1, 1]) <= 1e-9 * abs(hist.core_moment_Nm[-1, 0])
    assert hist.core_slip_fraction[-1] == 1.0
    return out


@compiled
def test_cf_d_full_slip_moment_matches_the_closed_form(model, iea66):
    """Planar monotonic loading to full slip. The distributed full-slip field
    is ``sigma_0 (C exp(lam x) - 1)``; its moment integrand ``C exp(lam x)
    sin x`` over the pitch gives the closed form, the direction-tied value
    ``a cosh(a pi/2) / (1 + a^2)`` times ``C``.

    Registered target 1e-4: FAIL at the default 60 stations per core
    (1.02e-4 cores, 1.03e-4 profiles; doc/benchmarks.md, CF-D-slip). The
    field is exact (the model reproduces the station sum of the closed-form
    field to 1e-9); the residual is the station quadrature of a field with
    kinks at the extreme fibres, second order in the station spacing (twice
    the stations: 2.6e-5)."""
    default = _full_slip(model)
    for continuous, stations in default:
        assert abs(stations) <= 1e-9
        assert 1.0e-4 < continuous < 1.1e-4  # the documented deviation
    finer = _full_slip(
        build_core_friction(
            iea66, dataclasses.replace(SPEC, n_stations_per_core=120), T
        )
    )
    for (coarse, _), (fine, stations) in zip(default, finer, strict=True):
        assert abs(stations) <= 1e-9
        assert fine <= 1e-4
        assert coarse / fine == pytest.approx(4.0, rel=0.02)


@pytest.mark.parametrize("follow", [False, True])
def test_cf_d_compiled_kernel_is_the_reference(model, iea66, monkeypatch, follow):
    t = np.linspace(0.0, 1.0, 120 if COMPILED else 24)
    amplitude = 0.02 if COMPILED else 2e-3
    kx = amplitude * np.sin(6 * PI * t) * (0.4 + t)
    ky = 0.6 * amplitude * np.sin(4 * PI * t + 1.0)
    loads = (
        core_load_history(iea66, model, T * (1.0 + 0.3 * np.sin(5 * PI * t)))
        if follow
        else None
    )
    runs = []
    for fast in (False, True):
        monkeypatch.setattr(dfm, "_FAST_KERNEL", fast)
        runs.append(
            core_friction_history(
                model,
                kx,
                ky,
                friction_model="distributed",
                loads=loads,
                nodes_per_pitch=NODES,
            )
        )
    ref, fast_run = runs
    scale = _scale(ref)
    assert np.max(np.abs(fast_run.moment_Nm - ref.moment_Nm)) <= 1e-10 * scale
    energy = np.max(np.abs(ref.work.endpoint_work_J_per_m))
    for name in (
        "stored_J_per_m",
        "friction_J_per_m",
        "endpoint_work_J_per_m",
        "integration_remainder_J_per_m",
    ):
        a, b = getattr(fast_run.work, name), getattr(ref.work, name)
        assert np.max(np.abs(a - b)) <= 1e-10 * energy, name


# --------------------------------------------------------------------------- #
# CF-T: core contact loads over a tension history                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("law", ["direction_tied", "distributed"])
def test_cf_t_constant_reference_tension_changes_nothing(model, iea66, law):
    kx, ky = _short_walk(7)
    loads = core_load_history(iea66, model, np.full(kx.size, T))
    fixed = core_friction_history(
        model, kx, ky, friction_model=law, nodes_per_pitch=NODES
    )
    follow = core_friction_history(
        model, kx, ky, friction_model=law, loads=loads, nodes_per_pitch=NODES
    )
    assert follow.loads is loads and fixed.loads is None
    scale = _scale(fixed)
    assert np.max(np.abs(follow.moment_Nm - fixed.moment_Nm)) <= 1e-12 * scale


@pytest.mark.parametrize("sigma", [0.0, 20.0e6], ids=["0MPa", "20MPa"])
@pytest.mark.parametrize("law", ["direction_tied", "distributed"])
def test_cf_t_loads_at_another_level_are_the_law_built_there(iea66, sigma, law):
    cable = _wound(iea66, sigma) if sigma else iea66
    at_ref = build_core_friction(cable, SPEC, T)
    level = 1.2e5
    at_level = build_core_friction(cable, SPEC, level)
    kx, ky = _short_walk(9)
    loads = core_load_history(cable, at_ref, np.full(kx.size, level))
    np.testing.assert_allclose(
        loads.core_N_per_m, at_level.loads.total_N_per_m, rtol=1e-12
    )
    follow = core_friction_history(
        at_ref, kx, ky, friction_model=law, loads=loads, nodes_per_pitch=NODES
    )
    direct = core_friction_history(
        at_level, kx, ky, friction_model=law, nodes_per_pitch=NODES
    )
    tolerance = 1e-12 if law == "direction_tied" else 2e-3
    scale = _scale(direct)
    assert np.max(np.abs(follow.moment_Nm - direct.moment_Nm)) <= tolerance * scale


def _tension_history(n=1200, amplitude=0.03, phase=0.1):
    t = np.arange(n)
    kappa = amplitude * np.sin(2.0 * PI * t / 200.0 + phase)
    tension = 1.0e4 + 6.0e4 * np.sin(PI * t / 370.0) ** 2
    return kappa, tension


@pytest.mark.parametrize("sigma", [0.0, 20.0e6], ids=["0MPa", "20MPa"])
@pytest.mark.parametrize("amplitude", [3e-4, 3e-3])
def test_cf_t_tension_cycle_agrees_with_the_slip_scaled_law(iea66, sigma, amplitude):
    """The MX-scale tension-cycle check with the laid cores and profiles."""
    cable = _wound(iea66, sigma) if sigma else iea66
    kappa, tension = _tension_history(amplitude=amplitude)
    export = export_bending_law(
        cable,
        tensions_N=[1.0e4, 7.0e4],
        kappa_max_1_m=1.2 * amplitude,
        n_points=300,
        slip_scale=True,
        slip_scale_tensions_N=np.linspace(1.0e4, 7.0e4, 7),
        core_friction=SPEC,
    )
    table = export.slip_scale

    def section(**options):
        return np.asarray(
            cx.compute_mkappa(cable, kappa, core_friction=SPEC, **options).moment_Nm
        )

    history = section(contact_tension_history_N=tension)
    span = np.ptp(history)
    residual = []
    for level in (1.0e4, 7.0e4):
        fixed = section(tension_N=level)
        law = table.moment(kappa, np.full_like(kappa, level))
        residual.append(np.max(np.abs(law - fixed)) / np.ptp(fixed))
    scaled = table.moment(kappa, tension)
    deviation = np.max(np.abs(scaled - history)) / span
    assert deviation <= 1.25 * max(residual)
    assert deviation < 5e-3
    # Reported (doc/benchmarks.md, CF-T-cycle): the loads held at the first
    # sample; the comparison is not gated.
    held = section(contact_tension_history_N=tension, core_loads="reference")
    assert np.isfinite(np.max(np.abs(scaled - held)) / span)


# --------------------------------------------------------------------------- #
# Routes, options and checks                                                   #
# --------------------------------------------------------------------------- #
def test_neutral_axes_core_options(iea66):
    # The direction-tied routes take a walk into slip with either kernel.
    kx, ky = _walk(13, 60, 2e-3)
    tension = T * (1.0 + 0.2 * np.sin(np.linspace(0.0, 3.0, kx.size)))
    base = {"tension_N": tension, "core_friction": SPEC}
    follow = cx.compute_neutral_axes(iea66, kx, ky, **base)
    held = cx.compute_neutral_axes(iea66, kx, ky, core_loads="reference", **base)
    assert not np.array_equal(follow.core_moment_Nm, held.core_moment_Nm)
    text = " ".join(follow.provenance.limitations)
    assert "direction-tied core law" in text and "following the tension" in text
    assert "at the reference tension" in " ".join(held.provenance.limitations)
    # Loads held at the first sample: the core law of the reference model.
    model = build_core_friction(iea66, SPEC, float(tension[0]))
    expected = core_friction_history(model, kx, ky).moment_Nm
    np.testing.assert_array_equal(held.core_moment_Nm, expected)
    # contact_tension_N sets the loads when given with a direct strain.
    contact = cx.compute_neutral_axes(
        iea66,
        kx,
        ky,
        axial_strain=np.zeros_like(kx),
        contact_tension_N=tension,
        core_friction=SPEC,
    )
    np.testing.assert_array_equal(contact.core_moment_Nm, follow.core_moment_Nm)
    # The core model can differ from the armour model.
    count = kx.size if COMPILED else 8
    split = cx.compute_neutral_axes(
        iea66,
        kx[:count] * (1.0 if COMPILED else 0.1),
        ky[:count] * (1.0 if COMPILED else 0.1),
        core_friction_model="distributed",
        core_friction=SPEC,
        tension_N=tension[:count],
    )
    assert "distributed core law" in " ".join(split.provenance.limitations)
    progressive = cx.compute_neutral_axes(
        iea66, kx, ky, friction_model="progressive", **base
    )
    np.testing.assert_array_equal(progressive.core_moment_Nm, follow.core_moment_Nm)
    with pytest.raises(CableInputError, match="core_loads"):
        cx.compute_neutral_axes(iea66, kx, ky, core_loads="frozen", **base)
    with pytest.raises(CableInputError, match="core_friction_model must"):
        cx.compute_neutral_axes(iea66, kx, ky, core_friction_model="other", **base)
    with pytest.raises(CableInputError, match="needs core_friction"):
        cx.compute_neutral_axes(iea66, kx, ky, core_friction_model="auto")


def test_auto_falls_back_on_a_rigid_interface(iea66):
    rigid = CoreFrictionSpec(core_shear=False)
    model = build_core_friction(iea66, rigid, T)
    assert model.xi == 1.0
    t = np.linspace(0.0, 2 * PI, 60)
    kx, ky = 0.01 * np.cos(t), 0.01 * np.sin(t)
    with pytest.warns(FrictionModelFallbackWarning, match="laid-core law"):
        auto = core_friction_history(model, kx, ky, friction_model="auto")
    assert auto.friction_model == "direction_tied"
    np.testing.assert_array_equal(
        auto.moment_Nm, core_friction_history(model, kx, ky).moment_Nm
    )
    with pytest.raises(CableInputError, match="compliant interface"):
        core_friction_history(model, kx, ky, friction_model="distributed")
    with pytest.raises(CableInputError, match="core friction model"):
        core_friction_history(model, kx, ky, friction_model="other")  # type: ignore[arg-type]
    # The armour route of the neutral axes resolves "auto" per layer.
    spec = cx.cable_to_fatigue_layers(iea66, contact_tension_N=T)[0].spec
    stiff = dataclasses.replace(spec, xi=1.0)
    with pytest.warns(FrictionModelFallbackWarning, match="stiff_layer"):
        result = neutral_axes([("stiff_layer", stiff)], kx, ky, friction_model="auto")
    assert result.layers[0].friction_model == "direction_tied"
    with warnings.catch_warnings():
        warnings.simplefilter("error", FrictionModelFallbackWarning)
        planar = neutral_axes(
            [("stiff_layer", stiff)], kx, np.zeros_like(kx), friction_model="auto"
        )
    assert planar.layers[0].friction_model == "direction_tied"


def test_core_load_history_routes_and_checks(iea66, model, monkeypatch):
    tension = np.array([1.0e4, 5.0e4, 9.0e4, 5.0e4])
    bounded = core_load_history(iea66, model, tension)
    assert bounded.contact_model == "bounded"
    for i, value in enumerate(tension):
        direct = build_core_friction(iea66, SPEC, float(value))
        assert bounded.core_N_per_m[i] == pytest.approx(
            direct.loads.total_N_per_m, rel=1e-12
        )
        assert bounded.filler_profile_N_per_m[i] == pytest.approx(
            direct.fillers.line_load_N_per_m, rel=1e-12
        )
        assert bounded.core_pressure_Pa[i] == pytest.approx(
            direct.population.contact_pressure_Pa, rel=1e-12
        )
        assert bounded.filler_pressure_Pa[i] == pytest.approx(
            direct.fillers.population.contact_pressure_Pa, rel=1e-12
        )
    np.testing.assert_allclose(bounded.core_mu_effective, 0.46, rtol=1e-12)
    # Zero tension and no clamping: no load, the filler coefficient.
    plain = build_core_friction(iea66, CoreFrictionSpec(mu_core_filler=0.3), T)
    zero = core_load_history(iea66, plain, np.zeros(3))
    assert np.all(zero.core_N_per_m == 0.0) and zero.filler_pressure_Pa is None
    np.testing.assert_array_equal(zero.core_mu_effective, 0.3)
    # Solved model: exact at a few tensions, interpolated beyond the node count.
    solved_model = build_core_friction(iea66, SPEC, T, contact_model="solved")
    solved = core_load_history(
        iea66, solved_model, np.array([T, T]), contact_model="solved"
    )
    np.testing.assert_allclose(
        solved.core_N_per_m, solved_model.loads.total_N_per_m, rtol=1e-12
    )
    monkeypatch.setattr(core_friction, "_SOLVED_LOAD_NODES", 2)
    spread = core_load_history(
        iea66, solved_model, np.array([2.0e4, T, 8.0e4]), contact_model="solved"
    )
    ends = [
        build_core_friction(iea66, SPEC, x, contact_model="solved").loads
        for x in (2.0e4, 8.0e4)
    ]
    assert spread.core_N_per_m[1] == pytest.approx(
        0.5 * (ends[0].total_N_per_m + ends[1].total_N_per_m), rel=1e-12
    )
    for bad in (np.zeros((2, 2)), np.array([]), np.array([np.nan]), np.array([-1.0])):
        with pytest.raises(CableInputError, match="tension history"):
            core_load_history(iea66, model, bad)
    with pytest.raises(CableSchemaError, match="contact_model"):
        core_load_history(iea66, model, tension, contact_model="other")
    with pytest.raises(CableInputError, match="one sample per curvature"):
        core_friction_history(model, np.zeros(3), np.zeros(3), loads=bounded)
    with pytest.raises(CableInputError, match="another core model"):
        core_friction_history(plain, np.zeros(4), np.zeros(4), loads=bounded)


def test_mixed_coefficients_scale_the_bound_with_the_cap(iea66):
    """Contact types loaded in another proportion: the core bound level follows
    the population's full-slip friction moment at the sample's coefficient."""
    mixed = CoreFrictionSpec(mu_core_filler=0.4, mu_core_core=0.1, mu_core_sheath=0.2)
    wound = _wound(iea66, 20.0e6)
    at_ref = build_core_friction(wound, mixed, T)
    level = 1.5e5
    at_level = build_core_friction(wound, mixed, level)
    assert at_level.mu_effective != pytest.approx(at_ref.mu_effective, rel=1e-6)
    loads = core_load_history(wound, at_ref, np.array([level]))
    assert loads.core_mu_effective[0] == pytest.approx(at_level.mu_effective, rel=1e-12)
    caps = [
        core_friction._population_cap_per_pressure(m.population, m.mu_effective)
        * m.population.contact_pressure_Pa
        for m in (at_ref, at_level)
    ]
    ratio = loads.core_pressure_Pa[0] / at_ref.population.contact_pressure_Pa
    assert ratio == pytest.approx(caps[1] / caps[0], rel=1e-12)


@pytest.mark.parametrize("law", ["direction_tied", "distributed"])
def test_zero_reference_coefficient_with_frictional_loads_later(iea66, law):
    """An unloaded reference (zero tension, no clamping) takes the filler
    coefficient, here zero; the core--core loads of the later samples carry
    friction all the same, normalised by the coefficient of the most loaded
    sample."""
    spec = CoreFrictionSpec(mu_core_filler=0.0, mu_core_core=0.2, mu_core_sheath=0.2)
    at_zero = build_core_friction(iea66, spec, 0.0)
    assert at_zero.population.mu == 0.0
    level = 8.0e4
    at_level = build_core_friction(iea66, spec, level)
    kx, ky = _short_walk(17)
    kx, ky = kx - kx[0], ky - ky[0]
    tension = np.full(kx.size, level)
    tension[0] = 0.0
    loads = core_load_history(iea66, at_zero, tension)
    assert loads.core_mu_capstan == pytest.approx(at_level.mu_effective, rel=1e-12)
    follow = core_friction_history(
        at_zero, kx, ky, friction_model=law, loads=loads, nodes_per_pitch=NODES
    )
    direct = core_friction_history(
        at_level, kx, ky, friction_model=law, nodes_per_pitch=NODES
    )
    assert follow.work.friction_J_per_m[-1] > 0.0
    tolerance = 1e-12 if law == "direction_tied" else 2e-3
    scale = _scale(direct)
    assert np.max(np.abs(follow.moment_Nm - direct.moment_Nm)) <= tolerance * scale


def test_compute_mkappa_core_loads_option(iea66):
    kappa, tension = _tension_history(n=300, amplitude=3e-3)
    follow = cx.compute_mkappa(
        iea66, kappa, contact_tension_history_N=tension, core_friction=SPEC
    )
    held = cx.compute_mkappa(
        iea66,
        kappa,
        contact_tension_history_N=tension,
        core_friction=SPEC,
        core_loads="reference",
    )
    assert follow.core_friction.loads is not None and held.core_friction.loads is None
    assert "core contact loads follow the tension history" in (follow.provenance.method)
    assert "laid-core and profile contact loads follow" in " ".join(
        follow.provenance.limitations
    )
    assert not np.array_equal(follow.moment_Nm, held.moment_Nm)
    # Without a tension history the option changes nothing.
    a = cx.compute_mkappa(iea66, kappa, tension_N=T, core_friction=SPEC)
    b = cx.compute_mkappa(
        iea66, kappa, tension_N=T, core_friction=SPEC, core_loads="reference"
    )
    np.testing.assert_array_equal(a.moment_Nm, b.moment_Nm)
    with pytest.raises(CableSchemaError, match="core_loads"):
        cx.compute_mkappa(iea66, kappa, core_loads="frozen")  # type: ignore[arg-type]
