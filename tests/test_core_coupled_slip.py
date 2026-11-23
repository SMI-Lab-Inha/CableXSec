# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Relative core--core slip of laid cores (``core_core_coupling="relative"``).

The coupled station law of :mod:`cablexsec.mechanics.station_coupling` against
the independent field solver ``validation/scripts/coupled_core_slip_reference.py``
(onset, full slip, moment, friction-stress distribution, reversal loop), its
limits, the agreement of the compiled and NumPy return maps, and the default
law, which is unchanged.
"""

from __future__ import annotations

import dataclasses
import math
import warnings

import numpy as np
import pytest
from coupled_core_slip_reference import CoupledSlipReference, run_path

import cablexsec as cx
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.mechanics import station_coupling
from cablexsec.mechanics.core_friction import (
    CoreFrictionSpec,
    build_core_friction,
    core_friction_history,
    core_full_slip_curvature,
    core_full_slip_moment,
    core_load_history,
    core_station_bounds,
    core_station_history,
)
from cablexsec.mechanics.mkappa import HelicalLayerSpec
from cablexsec.mechanics.papailiou import (
    coupled_slip_stress_bounds,
    slip_onset_curvature,
    wire_slip_history,
)
from cablexsec.mechanics.station_coupling import (
    StationCoupling,
    group_view,
    project_station_groups,
)
from cablexsec.provenance import FrictionModelFallbackWarning

T = 50.0e3
RELATIVE = CoreFrictionSpec(core_core_coupling="relative")
#: 2 pi / (3 sqrt 3): full slip over onset of three cores in core-core contact.
FULL_SLIP_RATIO = 2.0 * math.pi / (3.0 * math.sqrt(3.0))


@pytest.fixture(scope="module")
def cable():
    return cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)


def _population(n: int, spc: int, mu: float = 0.2) -> HelicalLayerSpec:
    """Rigid stations of ``n`` cores, lay angle small enough that the capstan
    term of the interlayer bound stays below 2e-4 relative."""
    return HelicalLayerSpec(
        n_wires=n * spc,
        r_helix_m=0.05,
        lay_angle_rad=math.asin(1.0e-3),
        E_wire_Pa=1.0e10,
        A_wire_m2=1.0e-4,
        xi=1.0,
        mu=mu,
        F_n_axi_N=100.0,
        contact_pressure_Pa=1.0e5,
        normal_load_is_line=True,
    )


def _library(share_cc: float, a: np.ndarray, *, n: int = 3, spc: int = 240):
    """Coupled law at ``a = kappa / kappa_on``: interlayer coefficient
    ``mu (1 - share)``, core-core coefficient ``mu`` on ``share`` of the load."""
    layer = _population(n, spc)
    coupling = StationCoupling(
        n, layer.mu * (1.0 - share_cc), layer.mu, layer.contact_pressure_Pa * share_cc
    )
    on = slip_onset_curvature(layer)
    history = wire_slip_history(
        layer,
        a * on,
        np.zeros_like(a),
        coupling=coupling,
        direction_hold_fraction=0.0,
    )
    return history, layer, on


# --------------------------------------------------------------------------- #
# Independent field solver (registered comparison)                             #
# --------------------------------------------------------------------------- #
CASES = {
    # label: (b_a, b_c, core-core share of the capacity)
    "intralayer": (0.0, 1.0, 1.0),
    "mixed": (1.0, 0.5, 0.5),
    "interlayer": (1.0, 0.0, 0.0),
}


@pytest.fixture(scope="module", params=sorted(CASES))
def loop(request):
    """Reference and library over 0 -> 1.7 -> -1.7 -> 1.7 (units of onset)."""
    b_a, b_c, share = CASES[request.param]
    ref = CoupledSlipReference(nodes=240, b_c=b_c, b_a=b_a)
    up = np.concatenate([[0.5, 0.9, 0.98], np.arange(1.0025, 1.7 + 1e-9, 0.0075)])
    down = np.arange(1.7, -1.7 - 1e-9, -0.02)[1:]
    back = np.arange(-1.7, 1.7 + 1e-9, 0.02)[1:]
    path = np.concatenate([up, down, back])
    keep = tuple(int(np.argmin(np.abs(up - v))) for v in (0.98, 1.03, 1.1, 1.2, 1.5))
    out = run_path(ref, path * ref.a_cr, keep=keep)
    history, layer, on = _library(share, path)
    return request.param, ref, path, up.size, keep, out, history, layer, on


@pytest.mark.independent_method
def test_moment_curvature_and_reversal_match_the_field_solver(loop):
    # Q3, Q5: moment along the loop within 3e-3 of the full-slip moment, cycle
    # dissipation within 2 %.
    _, ref, path, n_up, _, out, history, layer, on = loop
    m_ref = out.moment / (0.5 * ref.n_cores * ref.a_cr)
    m_lib = history.moment_Nm[:, 0] / (layer.EI_stick * on)
    full = m_ref[n_up - 1]
    assert np.max(np.abs(m_lib - m_ref)) <= 3e-3 * full
    w_ref = out.dissipation / (2.0 * math.pi * 0.5 * ref.n_cores * ref.a_cr**2)
    w_lib = history.work.friction_J_per_m / (layer.EI_stick * on**2)
    cycle_ref = w_ref[-1] - w_ref[n_up - 1]
    cycle_lib = w_lib[-1] - w_lib[n_up - 1]
    assert cycle_lib == pytest.approx(cycle_ref, rel=0.02)


@pytest.mark.independent_method
def test_friction_stress_distribution_matches_the_field_solver(loop):
    # Q4: the axial friction force along the pitch at 1.03, 1.1, 1.2, 1.5 onset.
    name, ref, path, _, keep, out, history, layer, on = loop
    theta = out.segment_azimuth[0]
    plateau = ref.b_a * math.pi / 2 + 2.0 * ref.b_c * min(math.pi / 3, math.pi / 2)
    phi = layer.phi_wires
    for index in keep[1:]:
        n_ref = np.interp(phi, theta, out.forces[index][0], period=2 * math.pi)
        n_lib = history.stress_Pa[:, index] / (layer.k_f * layer.r_helix_m * on)
        n_lib = n_lib * ref.a_cr
        diff = np.abs(n_lib - n_ref)
        assert np.sqrt(np.mean(diff**2)) <= 1e-2 * plateau, (name, path[index])
        assert np.max(diff) <= 4e-2 * plateau, (name, path[index])


@pytest.mark.independent_method
def test_onset_matches_the_field_solver(loop):
    # Q1: no macroscopic slip of the rigid-contact limit at 0.98 onset, slip at
    # 1.03 onset; the library onset is the absolute one.
    _, ref, path, _, keep, out, history, layer, on = loop
    theta = out.segment_azimuth
    k = min(ref.k_a if ref.b_a > 0 else ref.k_c, ref.k_c if ref.b_c > 0 else ref.k_a)
    for index, below in ((keep[0], True), (keep[1], False)):
        a = path[index] * ref.a_cr
        slip = np.max(np.abs(out.forces[index] - a * np.cos(theta)))
        assert (slip < 20.0 / k) if below else (slip > 100.0 / k)
    assert history.slipping_count[keep[0]] == 0
    assert history.slipping_count[keep[1]] > 0


def _reference_full_slip(b_a: float, b_c: float) -> float:
    """First curvature (units of onset) at which the reference tangent of the
    moment falls below 1e-3 of the stick tangent."""
    ref = CoupledSlipReference(nodes=360, b_c=b_c, b_a=b_a)
    a = np.concatenate(
        [[0.95], np.arange(1.0, 1.15, 0.01), np.arange(1.15, 1.45, 0.002)]
    )
    out = run_path(ref, a * ref.a_cr)
    m = out.moment / (0.5 * ref.n_cores * ref.a_cr)
    tangent = np.diff(m) / np.diff(a)
    flat = np.flatnonzero(tangent < 1e-3)
    return float(a[flat[0]])


def _library_full_slip(share: float) -> float:
    a = np.linspace(0.0, 1.6, 6401)
    history, layer, on = _library(share, a, spc=120)
    history = wire_slip_history(
        layer,
        a * on,
        0 * a,
        coupling=StationCoupling(
            3, layer.mu * (1 - share), layer.mu, layer.contact_pressure_Pa * share
        ),
        direction_hold_fraction=0.0,
        keep_cap=True,
    )
    loaded = np.abs(np.cos(layer.phi_wires)) > 1e-9
    stress = np.abs(history.stress_Pa[loaded])
    on_bound = np.all(stress >= history.cap_Pa[loaded] * (1 - 1e-12), axis=0)
    return float(a[np.argmax(on_bound)])


@pytest.mark.independent_method
@pytest.mark.parametrize(("b_a", "b_c", "share"), [(0.0, 1.0, 1.0), (1.0, 0.5, 0.5)])
def test_full_slip_curvature_matches_the_field_solver(b_a, b_c, share):
    # Q2: within 1 %; the pure intralayer value is 2 pi / (3 sqrt 3).
    reference = _reference_full_slip(b_a, b_c)
    library = _library_full_slip(share)
    assert library == pytest.approx(reference, rel=0.01)
    if share == 1.0:
        assert library == pytest.approx(FULL_SLIP_RATIO, rel=2e-3)
        assert reference == pytest.approx(FULL_SLIP_RATIO, rel=0.01)


def _closed_form_profile(psi: np.ndarray, b_a: float, b_c: float, n: int = 3):
    """Planar full-slip friction force at the angle ``psi`` from the neutral
    axis: interlayer b_a psi plus intralayer 2 b_c min(psi, pi / n)."""
    return b_a * psi + 2.0 * b_c * np.minimum(psi, math.pi / n)


@pytest.mark.independent_method
@pytest.mark.parametrize(("b_a", "b_c", "share"), [(0.0, 1.0, 1.0), (1.0, 0.5, 0.5)])
def test_full_slip_profile_and_moment_closed_form(b_a, b_c, share):
    # Q6: the frozen full-slip profile of the reference and of the library.
    ref = CoupledSlipReference(nodes=360, b_c=b_c, b_a=b_a)
    a = np.array([0.9, 1.2, 1.5, 1.8, 2.0]) * ref.a_cr
    out = run_path(ref, a, keep=(4,))
    theta = out.segment_azimuth[0]
    psi = np.arcsin(np.abs(np.cos(theta)))
    profile = np.sign(np.cos(theta)) * _closed_form_profile(psi, b_a, b_c)
    scale = float(np.max(np.abs(profile)))
    assert np.max(np.abs(out.forces[4][0] - profile)) <= 1e-3 * scale
    history, layer, on = _library(share, a / ref.a_cr)
    psi_l = np.arcsin(np.abs(np.cos(layer.phi_wires)))
    profile_l = np.sign(np.cos(layer.phi_wires)) * _closed_form_profile(psi_l, b_a, b_c)
    n_lib = history.stress_Pa[:, -1] / (layer.k_f * layer.r_helix_m * on) * ref.a_cr
    assert np.max(np.abs(n_lib - profile_l)) <= 1e-3 * scale
    # Pure intralayer: sqrt(3)/2 of the uncoupled full-slip moment.
    if share == 1.0:
        uncoupled = 4.0 / math.pi
        assert out.moment[-1] / (1.5 * ref.a_cr) == pytest.approx(
            math.sqrt(3) / 2 * uncoupled, rel=1e-3
        )


# --------------------------------------------------------------------------- #
# Limits                                                                       #
# --------------------------------------------------------------------------- #
def _planar_and_biaxial():
    t = np.linspace(0.0, 2.0, 121)
    return 2.0e-3 * np.sin(np.pi * t), 0.6e-3 * np.sin(2 * np.pi * t)


@pytest.mark.analytical_sanity
@pytest.mark.parametrize("contact_model", ["bounded", "solved"])
def test_no_core_core_friction_gives_the_interlayer_law(cable, contact_model):
    spec = CoreFrictionSpec(mu_core_core=0.0, filler_profiles=True)
    kx, ky = _planar_and_biaxial()
    absolute = build_core_friction(cable, spec, T, contact_model=contact_model)
    relative = build_core_friction(
        cable,
        dataclasses.replace(spec, core_core_coupling="relative"),
        T,
        contact_model=contact_model,
    )
    a = core_friction_history(absolute, kx, ky)
    b = core_friction_history(relative, kx, ky)
    scale = float(np.max(np.abs(a.moment_Nm)))
    assert np.max(np.abs(a.moment_Nm - b.moment_Nm)) <= 1e-12 * scale
    assert b.work.friction_J_per_m[-1] == pytest.approx(
        a.work.friction_J_per_m[-1], rel=1e-12
    )


@pytest.mark.analytical_sanity
def test_no_core_core_load_gives_the_absolute_law(cable):
    # A central filler takes the inward core loads: no core-core contact.
    spec = CoreFrictionSpec(central_filler=True)
    absolute = build_core_friction(cable, spec, T)
    relative = build_core_friction(
        cable, dataclasses.replace(spec, core_core_coupling="relative"), T
    )
    assert absolute.loads.core_core_N_per_m == 0.0
    kx, ky = _planar_and_biaxial()
    a = core_friction_history(absolute, kx, ky)
    b = core_friction_history(relative, kx, ky)
    scale = float(np.max(np.abs(a.moment_Nm)))
    assert np.max(np.abs(a.moment_Nm - b.moment_Nm)) <= 1e-12 * scale


@pytest.mark.analytical_sanity
def test_pure_intralayer_cores_reach_full_slip_at_the_coupled_ratio(cable):
    spec = CoreFrictionSpec(
        mu_core_filler=0.0,
        mu_core_sheath=0.0,
        core_core_coupling="relative",
        n_stations_per_core=240,
    )
    model = build_core_friction(cable, spec, T)
    assert model.coupling is not None
    assert model.coupling.interlayer_mu == 0.0
    on = slip_onset_curvature(model.population)
    assert core_full_slip_curvature(model) / on == pytest.approx(
        FULL_SLIP_RATIO, rel=2e-3
    )
    # Full-slip moment: sqrt(3)/2 of the uncoupled one with the same
    # (linear) bound.
    upper, lower, cap = coupled_slip_stress_bounds(
        model.population, model.coupling, 0.0
    )
    psi = np.arcsin(np.abs(np.cos(model.population.phi_wires)))
    linear = cap / np.minimum(psi, math.pi / 3).clip(1e-300) * psi
    c = np.abs(np.cos(model.population.phi_wires))
    ratio = np.sum(cap * c) / np.sum(linear * c)
    assert ratio == pytest.approx(math.sqrt(3) / 2, rel=1e-4)
    weight = (
        model.population.A_wire_m2
        * math.cos(model.population.lay_angle_rad)
        * model.population.r_helix_m
    )
    assert core_full_slip_moment(model) == pytest.approx(
        weight * np.sum(cap * c), rel=1e-12
    )


@pytest.mark.analytical_sanity
def test_two_cores_slip_as_the_absolute_law_without_capstan():
    # Two cores slip relative to each other in the sense of their absolute slip.
    layer = _population(2, 90)
    on = slip_onset_curvature(layer)
    a = np.concatenate([np.linspace(0, 3, 61), np.linspace(3, -3, 121)[1:]])
    coupled = wire_slip_history(
        layer,
        a * on,
        0.3 * a * on,
        coupling=StationCoupling(2, 0.0, layer.mu, layer.contact_pressure_Pa),
    )
    absolute = wire_slip_history(layer, a * on, 0.3 * a * on)
    scale = float(np.max(np.abs(absolute.moment_Nm)))
    lam = layer.mu * math.sin(layer.lay_angle_rad)
    assert lam <= 2e-4
    assert np.max(np.abs(coupled.moment_Nm - absolute.moment_Nm)) <= 5 * lam * scale


@pytest.mark.analytical_sanity
def test_cables_without_laid_core_family_are_rejected_alike():
    single = cx.load_cable(cx.inter_array_single_core_yaml_path())
    messages = []
    for spec in (CoreFrictionSpec(), RELATIVE):
        with pytest.raises(CableSchemaError) as error:
            build_core_friction(single, spec, T)
        messages.append(str(error.value))
    assert messages[0] == messages[1]


@pytest.mark.analytical_sanity
def test_stick_state_and_onset_are_those_of_the_absolute_law(cable):
    absolute = build_core_friction(cable, CoreFrictionSpec(), T)
    relative = build_core_friction(cable, RELATIVE, T)
    assert relative.stick_EI_Nm2 == absolute.stick_EI_Nm2
    assert relative.xi == absolute.xi
    on = slip_onset_curvature(absolute.population)
    k = np.linspace(0.0, 0.95 * on, 20)
    a = core_friction_history(absolute, k, 0 * k)
    b = core_friction_history(relative, k, 0 * k)
    assert np.max(np.abs(a.moment_Nm - b.moment_Nm)) <= 1e-12 * float(
        np.max(np.abs(a.moment_Nm))
    )
    # The bounds a station holds in full slip: interlayer plus intralayer.
    upper, lower = core_station_bounds(relative, 0.3)
    u_a, l_a, cap = coupled_slip_stress_bounds(
        relative.population, relative.coupling, 0.3
    )
    np.testing.assert_array_equal(upper, u_a + cap)
    np.testing.assert_array_equal(lower, l_a + cap)


# --------------------------------------------------------------------------- #
# Default law unchanged                                                        #
# --------------------------------------------------------------------------- #
#: The default law on dev e9b3f8f (before the coupling option): moment
#: components at two samples, final friction work, and the same with a
#: tension-following load history (66 kV template, filler profiles).
GOLDEN = {
    "bounded": (
        -51.667575888214344,
        -5.9973057283196995,
        0.20684780454609583,
        -99.55404704200525,
        0.1910175589473319,
    ),
    "solved": (
        -35.8886511509267,
        -6.336827311759066,
        0.17858448424133866,
        -72.95869295260198,
        0.17278280109502678,
    ),
}


@pytest.mark.analytical_sanity
@pytest.mark.parametrize("contact_model", ["bounded", "solved"])
def test_default_law_is_unchanged(cable, contact_model):
    model = build_core_friction(
        cable, CoreFrictionSpec(filler_profiles=True), T, contact_model=contact_model
    )
    assert model.coupling is None
    t = np.linspace(0.0, 2.0, 81)
    kx = 1.5e-3 * np.sin(np.pi * t)
    ky = 0.4e-3 * np.sin(2 * np.pi * t)
    h = core_friction_history(model, kx, ky)
    loads = core_load_history(
        cable, model, 30e3 + 40e3 * t, contact_model=contact_model
    )
    assert loads.core_intralayer_pressure_Pa is None
    h2 = core_friction_history(model, kx, ky, loads=loads)
    values = (
        float(h.moment_Nm[40, 0]),
        float(h.moment_Nm[60, 1]),
        float(h.work.friction_J_per_m[-1]),
        float(h2.moment_Nm[50, 0]),
        float(h2.work.friction_J_per_m[-1]),
    )
    # The solved trefoil comes from an iterative radial solve, which converges
    # to its own tolerance (the compiled and the plain build differ by a few
    # 1e-9); the bounded chain is closed form.
    rtol = 1e-12 if contact_model == "bounded" else 1e-7
    np.testing.assert_allclose(values, GOLDEN[contact_model], rtol=rtol, atol=0)
    # In the same run the default core law is the independent-station
    # update, bit for bit.
    direct = wire_slip_history(
        model.population,
        kx,
        ky,
        direction_hold_fraction=model.spec.direction_hold_fraction,
    )
    np.testing.assert_array_equal(h.core_moment_Nm, direct.moment_Nm)
    station = core_station_history(model, kx, ky)
    np.testing.assert_array_equal(station.stress_Pa, direct.stress_Pa)
    assert core_full_slip_moment(model) > 0.0


# --------------------------------------------------------------------------- #
# Return map: compiled and NumPy paths                                         #
# --------------------------------------------------------------------------- #
def _random_groups(rng, groups: int, n: int):
    upper = rng.uniform(0.0, 1.0, (groups, n)) * (rng.random((groups, n)) > 0.2)
    lower = upper * rng.uniform(0.5, 1.0, (groups, n))
    cap = rng.uniform(0.0, 1.5, (groups, n)) * (rng.random((groups, n)) > 0.1)
    trial = rng.normal(0.0, 2.0, (groups, n))
    return trial, upper, lower, cap


def _stations(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x.T).reshape(-1)


@pytest.mark.analytical_sanity
@pytest.mark.parametrize("n", [2, 3, 4])
def test_compiled_and_numpy_return_maps_agree_and_are_optimal(n):
    rng = np.random.default_rng(7 + n)
    trial, upper, lower, cap = _random_groups(rng, 400, n)
    args = [_stations(x) for x in (trial, upper, lower, cap)]
    fast = project_station_groups(*args, n, compiled=True)
    slow = project_station_groups(*args, n, compiled=False)
    scale = float(np.max(np.abs(args[0])))
    assert np.max(np.abs(fast - slow)) <= 1e-12 * scale
    s = group_view(slow, n)
    # Admissible: a decomposition a + c with sum(c) = 0 exists, and the result
    # is the closest admissible point (variational inequality against random
    # admissible points).
    lo_c = np.maximum(s - upper, -cap)
    hi_c = np.minimum(s + lower, cap)
    tol = 1e-12 * scale
    assert np.all(lo_c <= hi_c + tol)
    assert np.all(lo_c.sum(axis=1) <= tol)
    assert np.all(hi_c.sum(axis=1) >= -tol)
    for _ in range(20):
        a = rng.uniform(-lower, upper)
        c = rng.uniform(-cap, cap)
        c -= c.mean(axis=1, keepdims=True)
        inside = np.all(np.abs(c) <= cap, axis=1)
        y = a + c
        inner = np.sum((trial - s) * (y - s), axis=1)
        assert np.all(inner[inside] <= 1e-10 * scale**2)
    # A point that is admissible is its own projection.
    again = project_station_groups(_stations(s), *args[1:], n, compiled=False)
    assert np.max(np.abs(again - slow)) <= 1e-12 * scale


@pytest.mark.analytical_sanity
def test_core_history_is_the_same_on_both_return_maps(cable, monkeypatch):
    model = build_core_friction(cable, RELATIVE, T)
    kx, ky = _planar_and_biaxial()
    monkeypatch.setattr(station_coupling, "HAVE_NUMBA", True)
    fast = core_friction_history(model, kx, ky)
    monkeypatch.setattr(station_coupling, "HAVE_NUMBA", False)
    slow = core_friction_history(model, kx, ky)
    scale = float(np.max(np.abs(slow.moment_Nm)))
    assert np.max(np.abs(fast.moment_Nm - slow.moment_Nm)) <= 1e-12 * scale
    assert np.max(np.abs(fast.work.friction_J_per_m - slow.work.friction_J_per_m)) <= (
        1e-12 * float(np.max(np.abs(slow.work.friction_J_per_m)))
    )


# --------------------------------------------------------------------------- #
# Routes and inputs                                                            #
# --------------------------------------------------------------------------- #
def test_relative_law_lowers_the_slip_moment_and_keeps_energy(cable):
    absolute = build_core_friction(cable, CoreFrictionSpec(), T)
    relative = build_core_friction(cable, RELATIVE, T)
    assert relative.loads.core_core_N_per_m > 0.0
    on = slip_onset_curvature(absolute.population)
    k = np.concatenate([np.linspace(0, 6 * on, 121), np.linspace(6 * on, -6 * on, 241)])
    a = core_friction_history(absolute, k, 0 * k)
    b = core_friction_history(relative, k, 0 * k)
    assert abs(b.core_moment_Nm[120, 0]) < abs(a.core_moment_Nm[120, 0])
    assert core_full_slip_moment(relative) == pytest.approx(
        abs(b.core_moment_Nm[120, 0]), rel=1e-9
    )
    assert core_full_slip_curvature(relative) < core_full_slip_curvature(absolute)
    balance = b.work.balance_residual_J_per_m
    assert np.max(np.abs(balance)) <= 1e-9 * float(np.max(b.work.endpoint_work_J_per_m))
    assert np.all(np.diff(b.work.friction_J_per_m) >= -1e-15)


def test_relative_load_history_follows_the_core_core_load(cable):
    model = build_core_friction(cable, RELATIVE, T)
    tension = np.linspace(10e3, 120e3, 50)
    loads = core_load_history(cable, model, tension)
    assert loads.core_intralayer_pressure_Pa is not None
    reference = model.coupling.intralayer_pressure_Pa
    i = int(np.argmin(np.abs(tension - T)))
    assert loads.core_intralayer_pressure_Pa[i] == pytest.approx(reference, rel=0.05)
    assert np.all(np.diff(loads.core_intralayer_pressure_Pa) > 0.0)
    on = slip_onset_curvature(model.population)
    k = np.linspace(0.0, 5 * on, 50)
    held = core_friction_history(model, k, 0 * k)
    follow = core_friction_history(model, k, 0 * k, loads=loads)
    assert abs(follow.core_moment_Nm[-1, 0]) > abs(held.core_moment_Nm[-1, 0])
    # A history of the absolute law does not fit the relative model, and back.
    absolute = build_core_friction(cable, CoreFrictionSpec(), T)
    other = core_load_history(cable, absolute, tension)
    for m, h in ((model, other), (absolute, loads)):
        with pytest.raises(CableInputError, match="core-core coupling differs"):
            core_friction_history(m, k, 0 * k, loads=h)


def test_relative_load_history_with_an_unloaded_interlayer_reference(cable):
    # Interlayer coefficient zero at the reference, positive later: the
    # pressure is normalised by the most loaded sample.
    spec = CoreFrictionSpec(mu_core_sheath=0.0, core_core_coupling="relative")
    model = build_core_friction(cable, dataclasses.replace(spec, mu_core_filler=0.0), T)
    assert model.coupling.interlayer_mu == 0.0
    loads = core_load_history(cable, dataclasses.replace(model, spec=spec), [T, 2 * T])
    assert loads.core_mu_capstan > 0.0
    assert np.all(np.isfinite(loads.core_pressure_Pa))


def test_relative_law_is_direction_tied(cable):
    model = build_core_friction(cable, RELATIVE, T)
    t = np.linspace(0.0, 2 * np.pi, 60)
    on = slip_onset_curvature(model.population)
    kx, ky = 3 * on * np.cos(t), 3 * on * np.sin(t)
    with pytest.raises(CableInputError, match="direction-tied law"):
        core_friction_history(model, kx, ky, friction_model="distributed")
    with pytest.warns(FrictionModelFallbackWarning, match="relative core-core"):
        auto = core_friction_history(model, kx, ky, friction_model="auto")
    assert auto.friction_model == "direction_tied"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        planar = core_friction_history(model, kx, 0 * ky, friction_model="auto")
    assert planar.friction_model == "direction_tied"


def test_relative_specification_and_station_inputs():
    with pytest.raises(CableInputError, match="core_core_coupling"):
        CoreFrictionSpec(core_core_coupling="neighbours")  # type: ignore[arg-type]
    for bad in (
        {"group_size": 1},
        {"group_size": True},
        {"interlayer_mu": -0.1},
        {"intralayer_pressure_Pa": math.inf},
    ):
        values = {
            "group_size": 3,
            "interlayer_mu": 0.1,
            "intralayer_mu": 0.2,
            "intralayer_pressure_Pa": 1.0,
        }
        values.update(bad)
        with pytest.raises(CableInputError):
            StationCoupling(**values)
    with pytest.raises(CableInputError, match="groups of 3"):
        group_view(np.zeros(10), 3)
    layer = _population(3, 4)
    coupling = StationCoupling(3, 0.1, 0.2, 1e5)
    k = np.linspace(0.0, 1e-3, 5)
    with pytest.raises(CableInputError, match="friction law"):
        wire_slip_history(
            layer,
            k,
            k,
            coupling=coupling,
            friction_law=cx.FrictionLaw(static_factor=1.2, stick_time_s=0.0),
            time_s=np.arange(5.0),
        )
    with pytest.raises(CableInputError, match="needs a station coupling"):
        wire_slip_history(layer, k, k, intralayer_pressure_Pa=np.ones(5))
    with pytest.raises(CableInputError, match="groups of 5"):
        wire_slip_history(
            _population(1, 12), k, k, coupling=StationCoupling(5, 0.1, 0.2, 1.0)
        )
    with pytest.raises(CableInputError, match="intralayer pressure must"):
        wire_slip_history(layer, k, k, coupling=coupling, intralayer_pressure_Pa=[1.0])
    with pytest.raises(CableInputError, match="nonnegative intralayer"):
        wire_slip_history(
            layer, k, k, coupling=coupling, intralayer_pressure_Pa=-np.ones(5)
        )
    # Zero intralayer friction gives no intralayer bound; keep_cap reports
    # the station bound of the side of the stress.
    upper, lower, cap = coupled_slip_stress_bounds(
        layer, dataclasses.replace(coupling, intralayer_mu=0.0), 0.0
    )
    assert not np.any(cap)
    history = wire_slip_history(
        layer,
        5 * k,
        0 * k,
        coupling=coupling,
        keep_cap=True,
        direction_hold_fraction=0.0,
    )
    assert history.cap_Pa is not None
    assert np.all(np.abs(history.stress_Pa) <= history.cap_Pa * (1 + 1e-12))
    # Below the direction threshold the three bounds are blended.
    tiny = wire_slip_history(layer, 1e-9 * k, 1e-9 * k[::-1], coupling=coupling)
    assert np.all(tiny.slipping_count == 0)


def test_relative_routes_of_the_section(cable):
    from cablexsec.analysis.components import _core_axis
    from cablexsec.mechanics.core_kinematics import core_kinematics
    from cablexsec.mechanics.section_state import solve_section_state

    on = slip_onset_curvature(build_core_friction(cable, RELATIVE, T).population)
    k = np.linspace(0.0, 4 * on, 40)
    for spec in (CoreFrictionSpec(), RELATIVE):
        kin = core_kinematics(
            cable,
            k,
            0.2 * k,
            state="frictional",
            core_friction=spec,
            contact_tension_N=T,
        )
        assert kin.slipping is not None and kin.slipping[:, -1].any()
    state = solve_section_state(cable, tension_N=T, contact_model="bounded")
    layer = next(x for x in cable.layers if getattr(x, "r_offset_m", None))
    absolute = _core_axis(cable, layer, state, 4 * on, 0.0, CoreFrictionSpec())
    relative = _core_axis(cable, layer, state, 4 * on, 0.0, RELATIVE)
    assert "relative core-core slip" in relative.note
    np.testing.assert_allclose(relative.stuck, absolute.stuck)
    # The core on the extreme fibre holds the plateau of the coupled profile.
    assert abs(relative.frictional[0]) < abs(absolute.frictional[0])
    zero = _core_axis(cable, layer, state, 0.0, 0.0, RELATIVE)
    assert not np.any(zero.frictional)


def test_coupled_stations_with_contact_shape_and_changing_stiffness():
    layer = _population(3, 8)
    shaped = dataclasses.replace(layer, shape_g=np.ones(layer.n_wires))
    coupling = StationCoupling(3, 0.1, 0.2, 5e4)
    for part, other in zip(
        coupled_slip_stress_bounds(layer, coupling, 0.4),
        coupled_slip_stress_bounds(shaped, coupling, 0.4),
        strict=True,
    ):
        np.testing.assert_array_equal(part, other)
    on = slip_onset_curvature(layer)
    a = np.concatenate([np.linspace(0, 4, 41), np.linspace(4, -4, 81)[1:]])
    stiffness = layer.k_f * np.linspace(1.0, 0.8, a.size)
    history = wire_slip_history(
        shaped, a * on, 0.2 * a * on, coupling=coupling, stick_modulus_Pa=stiffness
    )
    scale = float(np.max(np.abs(history.work.endpoint_work_J_per_m)))
    assert np.max(np.abs(history.work.balance_residual_J_per_m)) <= 1e-9 * scale
    assert np.any(history.material_work_J_per_m != 0.0)


def test_coupled_full_slip_curvature_inputs(cable):
    free = CoreFrictionSpec(
        mu_core_filler=0.0,
        mu_core_core=0.0,
        mu_core_sheath=0.0,
        core_core_coupling="relative",
    )
    model = build_core_friction(cable, free, T)
    assert core_full_slip_curvature(model) == 0.0
    with pytest.raises(CableInputError, match="finite"):
        core_full_slip_curvature(build_core_friction(cable, RELATIVE, T), math.nan)
    absolute = build_core_friction(cable, CoreFrictionSpec(), T)
    upper, lower = core_station_bounds(absolute, 0.2)
    from cablexsec.mechanics.papailiou import wire_slip_stress_bounds

    reference = wire_slip_stress_bounds(absolute.population, 0.2)
    np.testing.assert_array_equal(upper, reference[0])
    np.testing.assert_array_equal(lower, reference[1])


@pytest.mark.independent_method
def test_skeie2023_table1_is_the_saturation_of_the_peak_friction_stress() -> None:
    """Skeie, Sødahl, Pettersen and Verde (2023) Table 1 (row SK23-T1): with
    the contact load per metre of helix the onset is their Eq. 16; the largest
    friction stress stops growing at pi/3 times the onset (their "full slip",
    1.049 in their model, 1.053 in their FE), the slip zones coalesce at
    2 pi / (3 sqrt 3)."""
    from check_core_friction import SKEIE2023, skeie2023_table1

    sk = skeie2023_table1()
    assert sk["onset_helix_1_mm"] == pytest.approx(sk["eq16_1_mm"], rel=1e-12)
    assert sk["onset_helix_1_mm"] == pytest.approx(
        SKEIE2023["table1_onset_1_mm"], rel=0.01
    )
    assert sk["first_slip_ratio"] == pytest.approx(1.0, abs=2e-4)
    assert sk["saturation_ratio"] == pytest.approx(math.pi / 3, abs=2e-4)
    assert sk["coalescence_ratio"] == pytest.approx(FULL_SLIP_RATIO, abs=2e-4)
    table = SKEIE2023["table1_full_helica_1_mm"] / SKEIE2023["table1_onset_1_mm"]
    assert sk["saturation_ratio"] == pytest.approx(table, rel=0.005)
