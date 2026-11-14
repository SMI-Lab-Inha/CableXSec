# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Fretting regime, energy, wear and knock-down module (gap 3.1, gates FR1-FR9)."""

import math

import numpy as np
import pytest
from scipy.integrate import quad

import cablexsec as cx
from cablexsec.analysis import fretting_regime as fr
from cablexsec.analysis.fatigue import miner_damage, rainflow_cycles
from cablexsec.analysis.fretting_assessment import (
    conductor_trellis_fretting,
    extreme_fibre_offset_deg,
)
from cablexsec.materials.sn_curves import SN_CURVES
from cablexsec.materials.wear import (
    FRETTING_KNOCKDOWNS,
    WEAR_COEFFICIENTS,
    WearCoefficient,
    get_knockdown,
    get_wear_coefficient,
    wear_pair_for,
)
from cablexsec.mechanics.hertz_contact import solve_contact
from cablexsec.mechanics.mindlin_contact import mindlin_monotonic
from cablexsec.mechanics.strand_slip import ConductorRing

STEEL = get_wear_coefficient("steel_steel")


# --------------------------------------------------------------------------- #
# Compiled cycle table (regression against the reference counter)             #
# --------------------------------------------------------------------------- #
def _histories():
    rng = np.random.default_rng(7)
    return [
        np.array([]),
        np.array([1.0]),
        np.array([1.0, 3.0]),
        np.array([2.0, 2.0]),
        np.full(20, 4.0),
        np.linspace(0.0, 5.0, 30),
        np.round(rng.normal(size=3000) * 3) / 3,  # plateaus and ties
        np.cumsum(rng.normal(size=5000)),
        np.r_[0.0, 1.0, 1.0, 1.0, -2.0, -2.0, 3.0, 3.0, 3.0],
    ]


@pytest.mark.parametrize("history", _histories())
def test_rainflow_table_matches_reference(history):
    reference = list(rainflow_cycles(history))
    rng, mean, count, start, end = fr.rainflow_table(history)
    assert rng.size == len(reference)
    for k, (r, m, c, i0, i1) in enumerate(reference):
        assert (rng[k], mean[k], count[k], start[k], end[k]) == (r, m, c, i0, i1)


def test_rainflow_table_python_path_matches_compiled():
    kernel = getattr(fr.rainflow_table.__globals__["_rainflow_table"], "py_func", None)
    if kernel is None:
        pytest.skip("numba not active")
    history = np.cumsum(np.random.default_rng(3).normal(size=2000))
    for a, b in zip(kernel(history), fr.rainflow_table(history), strict=True):
        np.testing.assert_array_equal(a, b)


def test_window_half_range_matches_numpy():
    u = np.cumsum(np.random.default_rng(5).normal(size=500))
    start = np.array([0, 10, 400, 499, 30], dtype=np.int64)
    end = np.array([499, 20, 300, 499, 31], dtype=np.int64)
    got = fr._window_half_range(u, start, end)
    for k, (a, b) in enumerate(zip(start, end, strict=True)):
        w = u[min(a, b) : max(a, b) + 1]
        assert got[k] == 0.5 * (w.max() - w.min())


# --------------------------------------------------------------------------- #
# FR1: Mindlin closed form against the numerical Masing loop                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ratio", [0.05, 0.3, 0.7, 1.0, 1.5, 4.0])
def test_mindlin_energy_closed_form_matches_numeric_loop(ratio):
    mu, force, stiffness = 0.6, 25.0, 3.0e7
    dt = fr.mindlin_transition_amplitude(mu, force, stiffness)
    da = ratio * dt
    q_star = float(mindlin_monotonic(min(da, dt), mu, force, stiffness))
    x = np.linspace(-da, da, 400_001)
    upper = np.minimum(
        -q_star + 2.0 * mindlin_monotonic(0.5 * (x + da), mu, force, stiffness), q_star
    )
    numeric = 2.0 * np.trapezoid(upper, x)  # lower branch is the mirror image
    closed = fr.mindlin_cycle_energy(np.array([da]), mu, force, dt)[0]
    assert closed == pytest.approx(numeric, rel=2e-6)


def test_mindlin_small_amplitude_limit_is_johnson_763():
    # [Johnson1985] eq. 7.63: W = sum((2 - nu)/G) Q*^3/(36 a mu P) = (2/9) Q*^3/(mu P K_t)
    mu, force, stiffness = 0.3, 10.0, 1.0e7
    dt = fr.mindlin_transition_amplitude(mu, force, stiffness)
    for x in (1e-6, 1e-4, 1e-2):
        da = dt * -math.expm1((2.0 / 3.0) * math.log1p(-x))
        w = fr.mindlin_cycle_energy(np.array([da]), mu, force, dt)[0]
        expected = (2.0 / 9.0) * (x * mu * force) ** 3 / (mu * force * stiffness)
        assert w == pytest.approx(expected, rel=2 * x + 1e-9)


def test_transition_amplitude_is_fouvry_eq19():
    # K1 = (3/16) sum((2 - nu)/G); delta_t = K1 mu P / a  ([Fouvry1995] eqs. 3, 19)
    e, nu, radius, force, mu = 210e9, 0.3, 5e-3, 50.0, 0.8
    g = e / (2 * (1 + nu))
    c = fr.point_contact(
        "ball", force, 1 / (2 * radius), 1 / (2 * radius), e, nu, e, nu, mu
    )
    a = c.half_width_m
    k1 = 3.0 / 16.0 * 2 * (2 - nu) / g
    assert c.transition_amplitude_m == pytest.approx(k1 * mu * force / a, rel=1e-12)


# --------------------------------------------------------------------------- #
# FR2: Fouvry transition criteria and the published case                      #
# --------------------------------------------------------------------------- #
def test_fouvry_transition_values():
    a, d, b = fr.fouvry_ratios(np.array([1.0]), 1.0)
    assert a[0] == pytest.approx(0.2, abs=1e-15)
    assert d[0] == pytest.approx(2 * 0.5 ** (2 / 3) - 1, abs=1e-15)  # 0.26
    assert b[0] == pytest.approx(0.2 / (2 * 0.5 ** (2 / 3) - 1), abs=1e-14)  # 0.77
    assert b[0] == pytest.approx(0.77, abs=0.001)


def test_energy_ratio_criterion_equals_amplitude_criterion():
    dt = 2.0e-6
    amplitude = np.geomspace(1e-3, 50, 400) * dt
    a, _d, _b = fr.fouvry_ratios(amplitude, dt)
    contact = fr.FrettingContact("c", "point", "mindlin", 0.5, 1.0, dt, 1e-4, 1e8, 1e11)
    gross = contact.regime(amplitude) == fr.GROSS
    np.testing.assert_array_equal(gross, a > 0.2 + 1e-12)
    assert np.all(np.diff(a) >= -1e-12)


# [Fouvry1995] Tables 3-4: HSS flat / Cr-steel ball, R = 12.7 mm, E* = 119.4 GPa,
# G1 = 159.7, G2 = 146.5 GPa (as printed), nu = 0.28.
FOUVRY_CASES = [  # P [N], mu_t, a [um], delta_t Mindlin [um], measured [um]
    (200.0, 0.97, 252.0, 3.2, 3.6),
    (300.0, 0.95, 288.0, 4.1, 4.4),
    (400.0, 0.87, 317.0, 4.6, 4.9),
    (500.0, 0.85, 342.0, 5.2, 5.4),
]


@pytest.mark.parametrize(
    ("force", "mu", "a_um", "theory_um", "measured_um"), FOUVRY_CASES
)
def test_fouvry_1995_transition_amplitudes(force, mu, a_um, theory_um, measured_um):
    radius, e_star, nu = 12.7e-3, 119.4e9, 0.28
    hz = solve_contact(force, 1 / (2 * radius), 1 / (2 * radius), e_star)
    assert hz.a_m * 1e6 == pytest.approx(a_um, abs=1.0)
    g_star = 1.0 / ((2 - nu) / 159.7e9 + (2 - nu) / 146.5e9)
    k_t = 8.0 * g_star * hz.a_m
    dt = fr.mindlin_transition_amplitude(mu, force, k_t)
    assert dt * 1e6 == pytest.approx(theory_um, abs=0.1)  # one decimal in Table 4
    # With G = E/(2(1 + nu)) the same formula gives ~1.7-1.8x the measured value
    # (documented in BENCHMARKS.md, gate FR2).
    g_true = 1.0 / ((2 - nu) / (230e9 / 2.56) + (2 - nu) / (211e9 / 2.56))
    ratio = (
        fr.mindlin_transition_amplitude(mu, force, 8 * g_true * hz.a_m)
        * 1e6
        / measured_um
    )
    assert 1.5 < ratio < 2.0


# --------------------------------------------------------------------------- #
# FR3: Cattaneo strip micro-slip energy                                       #
# --------------------------------------------------------------------------- #
def test_strip_integral_limits_and_quadrature():
    assert fr._strip_integral(np.array([0.0]))[0] == pytest.approx(
        math.pi / 16, rel=1e-13
    )
    for gamma in (0.05, 0.3, 0.7, 0.95):

        def integrand(xi, g=gamma):
            root = math.sqrt(max(xi * xi - g * g, 0.0))
            slip = xi * root - g * g * math.log((xi + root) / g)
            return math.sqrt(1 - xi * xi) * slip

        expected, _err = quad(integrand, gamma, 1.0, epsabs=0, epsrel=1e-12, limit=200)
        assert fr._strip_integral(np.array([gamma]))[0] == pytest.approx(
            expected, rel=1e-9
        )


def test_strip_slip_profile_matches_gradient_integral():
    # s(X) = (mu p0/(E* b))[X sqrt(X^2-c^2) - c^2 ln((X+sqrt)/c)] integrates
    # ds/dX = 2 mu p0 sqrt(X^2 - c^2)/(E* b) (module notes).
    b, c, mu, p0, e = 1.0, 0.6, 0.3, 1.0, 1.0
    for xx in (0.7, 0.9, 1.0):
        num, _ = quad(lambda t: 2 * mu * p0 * math.sqrt(t * t - c * c) / (e * b), c, xx)
        root = math.sqrt(xx * xx - c * c)
        closed = mu * p0 / (e * b) * (xx * root - c * c * math.log((xx + root) / c))
        assert closed == pytest.approx(num, rel=1e-10)


def test_line_energy_gross_limit_and_cubic_start():
    mu, q, e_star = 0.2, 100.0, 1e9
    w1 = fr.cattaneo_line_energy(np.array([1.0]), mu, q, e_star)[0]
    assert w1 == pytest.approx(2 / math.pi * (mu * q) ** 2 / e_star, rel=1e-13)
    w = fr.cattaneo_line_energy(np.array([1e-3, 2e-3]), mu, q, e_star)
    assert w[1] / w[0] == pytest.approx(8.0, rel=1e-2)  # O(x^3)


# --------------------------------------------------------------------------- #
# FR4: limits                                                                 #
# --------------------------------------------------------------------------- #
def _sine(amplitude, cycles=20, n=40):
    return amplitude * np.sin(np.linspace(0, 2 * math.pi * cycles, cycles * n + 1))


def test_infinite_friction_is_stick_without_wear():
    c = fr.point_contact(
        "p", 10.0, 50.0, 80.0, 210e9, 0.3, 210e9, 0.3, math.inf, wear=STEEL
    )
    r = fr.assess_contact(_sine(50e-6), c, partial_slip_wear=True)
    assert set(r.regime.tolist()) == {fr.STICK}
    assert r.total_energy == 0.0 and r.wear_depth_m == 0.0


def test_large_friction_tends_to_no_energy():
    energies, wear = [], []
    for mu in (5.0, 50.0, 500.0, 5000.0):  # beyond the transition
        c = fr.point_contact(
            "p", 10.0, 50.0, 80.0, 210e9, 0.3, 210e9, 0.3, mu, wear=STEEL
        )
        r = fr.assess_contact(_sine(5e-6), c, partial_slip_wear=True)
        energies.append(r.total_energy)
        wear.append(r.wear_depth_m)
        if mu >= 50:
            assert r.dominant_regime == "partial_slip"
            assert fr.assess_contact(_sine(5e-6), c).wear_depth_m == 0.0
    assert all(b < a for a, b in zip(energies, energies[1:], strict=False))
    assert energies[-1] < 1e-3 * energies[0] and wear[-1] < 1e-3 * wear[0]


def test_zero_load_line_contact_has_no_contact():
    c = fr.line_contact("l", 0.0, 1e-3, 210e9, 0.3, 1e9, 0.45, 0.2, 1e7)
    r = fr.assess_contact(_sine(1e-4), c)
    assert r.total_energy == 0.0 and r.dominant_regime == "stick"
    assert math.isnan(r.slip_to_width)


def test_gross_slip_energy_tends_to_coulomb_work():
    c = fr.line_contact("l", 100.0, 1.5e-3, 210e9, 0.3, 0.8e9, 0.45, 0.2, 1e7)
    r = fr.assess_contact(_sine(1e-3, cycles=10), c)
    expected = 4 * 0.2 * 100.0 * (1e-3 - c.transition_amplitude_m)
    full = r.energy[r.count == 1.0]
    assert np.all(full > expected) and np.all(full < 1.01 * expected)


# --------------------------------------------------------------------------- #
# FR5: rotation invariance of planar paths                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("angle", [0.3, 1.1, 2.5, -0.7])
def test_planar_path_rotation_invariance(angle):
    t = np.linspace(0, 12 * math.pi, 3001)
    rng = np.random.default_rng(11)
    path = np.c_[
        30e-6 * np.sin(t) + 5e-6 * rng.normal(size=t.size), 12e-6 * np.cos(1.3 * t)
    ]
    rot = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    c = fr.point_contact("p", 20.0, 40.0, 70.0, 210e9, 0.3, 210e9, 0.3, 0.4, wear=STEEL)
    a = fr.assess_contact(path, c)
    b = fr.assess_contact(path @ rot.T, c)
    assert a.planar_path and b.planar_path
    np.testing.assert_allclose(b.amplitude_m, a.amplitude_m, rtol=1e-10)
    np.testing.assert_array_equal(b.regime, a.regime)
    assert b.wear_depth_m == pytest.approx(a.wear_depth_m, rel=1e-10)


def test_collinear_planar_path_equals_scalar_path():
    u = _sine(20e-6) + np.linspace(0, 3e-6, 801)
    direction = np.array([0.6, -0.8])
    c = fr.point_contact("p", 20.0, 40.0, 70.0, 210e9, 0.3, 210e9, 0.3, 0.4, wear=STEEL)
    one = fr.assess_contact(u, c)
    two = fr.assess_contact(u[:, None] * direction[None, :], c)
    assert not two.planar_path
    np.testing.assert_allclose(two.amplitude_m, one.amplitude_m, rtol=1e-9)


# --------------------------------------------------------------------------- #
# FR6: Archard / energy wear against a published crossed-wire test            #
# --------------------------------------------------------------------------- #
def test_energy_wear_reproduces_crossed_wire_scar_volume():
    """[Poon2024] Tables 4-5 (data of [Cruzado2013]): 0.45 mm steel wires at 45 deg,
    P = 1 N, mu = 0.7, 50 000 cycles, stroke 130 um (amplitude 65 um),
    k = 2.93e-8 MPa^-1; measured scar 154 x 381 um, depth 18 um."""
    d, theta, force, mu, cycles = 0.45e-3, math.radians(45), 1.0, 0.7, 50_000
    from cablexsec.mechanics.hertz_contact import crossed_cylinder_curvatures

    a_c, b_c = crossed_cylinder_curvatures(d, theta)
    k = 2.93e-8 * 1e-6  # m^3/(N m)
    coefficient = WearCoefficient(
        "test", k / mu, k / mu, k / mu, "[Poon2024]", False, "wire"
    )
    c = fr.point_contact(
        "wire", force, a_c, b_c, 210e9, 0.33, 210e9, 0.33, mu, wear=coefficient
    )
    u = 65e-6 * np.sin(np.linspace(0, 2 * math.pi * 200, 200 * 16 + 1))
    r = fr.assess_contact(u, c)
    assert r.dominant_regime == "gross_slip"
    volume = r.wear_volume * cycles / 200
    measured = math.pi / 8 * 154e-6 * 381e-6 * 18e-6  # elliptic paraboloid cap
    assert 0.8 < volume / measured < 1.1


# --------------------------------------------------------------------------- #
# FR7: knock-down                                                             #
# --------------------------------------------------------------------------- #
def _stress_and_slip():
    t = np.linspace(0, 40 * math.pi, 4001)
    stress = 60e6 * np.cos(t) + 5e6 * np.sin(3.1 * t)
    slip = 8e-6 * np.sin(t)  # neutral-axis phase
    return stress, slip


def test_knockdown_none_reproduces_plain_miner():
    stress, slip = _stress_and_slip()
    curve = SN_CURVES["saevik_2011_flat_steel"]
    c = fr.line_contact("l", 100.0, 1.5e-3, 210e9, 0.3, 0.8e9, 0.45, 0.2, 1e7)
    kd = fr.fretting_fatigue_damage(stress, slip, c, curve, get_knockdown("none"))
    assert kd.damage_fretting == kd.damage_plain
    assert kd.damage_plain == pytest.approx(miner_damage(stress, curve, 1.0), rel=1e-12)


def test_partial_slip_knockdown_raises_damage_gross_does_not():
    stress, slip = _stress_and_slip()
    curve = SN_CURVES["saevik_2011_flat_steel"]
    law = get_knockdown("steel_nishioka_hirakawa")
    partial = fr.line_contact("l", 100.0, 1.5e-3, 210e9, 0.3, 0.8e9, 0.45, 0.2, 1e5)
    gross = fr.line_contact("l", 100.0, 1.5e-3, 210e9, 0.3, 0.8e9, 0.45, 0.2, 1e8)
    kp = fr.fretting_fatigue_damage(stress, slip, partial, curve, law)
    kg = fr.fretting_fatigue_damage(stress, slip, gross, curve, law)
    assert kp.count_by_regime[fr.PARTIAL] > 0 and kp.damage_fretting > kp.damage_plain
    assert kg.count_by_regime[fr.PARTIAL] == 0
    assert kg.damage_fretting == pytest.approx(kg.damage_plain, rel=1e-12)
    loss = fr.fretting_fatigue_damage(
        stress, slip, gross, curve, law, section_loss_factor=1.2
    )
    assert loss.damage_fretting > kg.damage_fretting


def test_worn_section_fraction():
    r = 1e-3
    assert fr.worn_section_fraction(np.array([0.0]), r)[0] == 0.0
    assert fr.worn_section_fraction(np.array([r]), r)[0] == pytest.approx(0.5)
    assert fr.worn_section_fraction(np.array([3 * r]), r)[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# FR8: registry provenance                                                    #
# --------------------------------------------------------------------------- #
def test_every_wear_coefficient_cites_or_declares_assumption():
    for key, entry in WEAR_COEFFICIENTS.items():
        assert entry.pair == key
        assert entry.source.strip()
        assert entry.assumed == entry.source.startswith("ASSUMPTION")
        assert entry.low_m3_per_J <= entry.alpha_m3_per_J <= entry.high_m3_per_J
    for entry in FRETTING_KNOCKDOWNS.values():
        assert entry.source.strip()


def test_wear_pair_mapping():
    assert (
        wear_pair_for("steel_janocha2024", "mdpe_janocha2024") == "steel_polyethylene"
    )
    assert wear_pair_for("hdpe", "galvanised_steel_armour") == "steel_polyethylene"
    assert (
        wear_pair_for("steel_janocha2024", "ppy_janocha2024") == "steel_polypropylene"
    )
    assert wear_pair_for("copper_stranded", "copper_janocha2024") == "copper_copper"
    assert (
        wear_pair_for("galvanised_steel_armour", "steel_janocha2024") == "steel_steel"
    )
    assert wear_pair_for("galvanised_steel_armour", "pfr_menard2023") == (
        "steel_fluoroelastomer"
    )


# --------------------------------------------------------------------------- #
# FR9: cable application (public 66 kV template)                              #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def iea66():
    return cx.load_cable(cx.iea_task49_66kv_yaml_path())


def test_compute_fretting_66kv_locations(iea66):
    t = np.arange(4000) * 0.1
    kx = 0.004 * np.sin(2 * math.pi * t / 9.0)
    ky = np.zeros_like(kx)
    tension = np.full_like(kx, 100e3)
    result = cx.compute_fretting(
        iea66, kx, ky, duration_s=float(t[-1]), tension_N=tension
    )
    assert set(result.faces) == {
        "inner_armour/inner_face",
        "inner_armour/outer_face",
        "outer_armour/inner_face",
        "outer_armour/outer_face",
    }
    assert result.extreme_fibre_deg == pytest.approx(0.0, abs=1e-9)
    face = result.faces["inner_armour/inner_face"]
    assert face.partner == "mdpe_inner_sheath"
    assert any("assumption" in note for note in result.provenance.limitations)
    loc = face.location()
    assert loc["max_wear_offset_deg"] > 80.0  # wear at the neutral axis
    assert loc["max_damage_offset_deg"] < 10.0  # cracking at the extreme fibre
    share = face.regime_fraction
    counted = face.cycles_by_regime.sum(axis=1) > 0
    np.testing.assert_allclose(share[counted].sum(axis=1), 1.0)
    assert not counted[0]  # the extreme-fibre wire does not slip in plane bending
    # Outer sheath of a wet design carries no contact load at zero external pressure.
    outer = result.faces["outer_armour/outer_face"]
    assert outer.contact.normal_load == 0.0
    assert np.all(outer.wear_depth_per_year_m == 0.0)


def test_compute_fretting_wear_override_scales_linearly(iea66):
    t = np.arange(2000) * 0.1
    kx = 0.003 * np.sin(2 * math.pi * t / 7.0)
    ky = 0.001 * np.cos(2 * math.pi * t / 7.0)
    base = cx.compute_fretting(
        iea66, kx, ky, duration_s=200.0, contact_tension_N=80e3, knockdown=None
    )
    entry = get_wear_coefficient("steel_polyethylene")
    scaled = cx.compute_fretting(
        iea66,
        kx,
        ky,
        duration_s=200.0,
        contact_tension_N=80e3,
        knockdown=None,
        wear={"steel_polyethylene": entry.scaled(10 * entry.alpha_m3_per_J)},
    )
    a = base.faces["inner_armour/inner_face"].wear_depth_per_year_m
    b = scaled.faces["inner_armour/inner_face"].wear_depth_per_year_m
    np.testing.assert_allclose(b, 10 * a, rtol=1e-12)
    assert np.isnan(base.faces["inner_armour/inner_face"].D_fretting_per_year).all()


def test_extreme_fibre_offset_folding():
    got = extreme_fibre_offset_deg(np.radians([0, 45, 90, 135, 180, 270, 359]), 0.0)
    np.testing.assert_allclose(got, [0, 45, 90, 45, 0, 90, 1], atol=1e-9)


# --------------------------------------------------------------------------- #
# Conductor trellis                                                           #
# --------------------------------------------------------------------------- #
def _nasution_rings():
    d = 3.21e-3
    layers = ((6, 265e-3, 1), (12, 285e-3, -1), (18, 295e-3, 1))
    rings = [ConductorRing(1, d, 0.0, 0.0)]
    for i, (n, pitch, sign) in enumerate(layers, start=1):
        rings.append(ConductorRing.from_pitch(n, d, i * d, pitch, sign))
    return rings


@pytest.mark.parametrize(("amplitude", "expected"), [(1e-4, "partial"), (0.3, "gross")])
def test_trellis_regimes_follow_slip(amplitude, expected):
    t = np.linspace(0, 40 * math.pi, 2001)
    kx = amplitude * np.sin(t)
    result = conductor_trellis_fretting(
        _nasution_rings(),
        115e9,
        0.34,
        kx,
        np.zeros_like(kx),
        np.full_like(kx, 30e3),
        mu_wire=0.2,
        duration_s=200.0,
        wear=get_wear_coefficient("copper_copper"),
    )
    assert result.contacts[0] is None  # centre wire: line contact, not assessed
    share = result.cycles_by_regime[1:3].sum(axis=(0, 1))
    if expected == "partial":
        assert share[fr.GROSS] == 0 and share[fr.PARTIAL] > 0
        assert np.all(result.wear_depth_per_year_m == 0)
    else:
        assert share[fr.GROSS] > share[fr.PARTIAL]
        assert np.max(result.wear_depth_per_year_m) > 0


def test_nishioka_notch_factor_limits():
    law = get_knockdown("steel_nishioka_hirakawa")
    curve = SN_CURVES["saevik_2011_flat_steel"]
    ref = fr.range_at_cycles(curve, law.reference_cycles)
    assert curve.cycles_to_failure(ref / 1e6) == pytest.approx(1e7, rel=1e-9)
    delta = np.array([0.0, 1e-9, 8e-6, 1e-3])
    k = law.notch_factor(0.2, 50e6, delta, ref)
    full = 1 + 2 * 0.2 * 50e6 / (0.5 * ref)
    assert k[0] == 1.0 and k[1] == pytest.approx(1.0, abs=1e-3)
    assert k[2] == pytest.approx(1 + (full - 1) * (1 - math.exp(-8 / 2.7)), rel=1e-12)
    assert k[3] == pytest.approx(full, rel=1e-12)
    capped = law.notch_factor(0.2, 1e9, delta, ref)[3]
    assert capped == pytest.approx(1 + 2 * 0.2 * 160e6 / (0.5 * ref), rel=1e-12)
    assert np.all(get_knockdown("none").notch_factor(0.2, 50e6, delta, ref) == 1.0)
