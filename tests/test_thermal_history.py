# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Temperature histories coupled into clamping, friction and fatigue (gap 4.1).

Gates:

* thermal network: steady state equals ``compute_thermal`` (BM5 inputs,
  136.06 deg C) and the 66 kV template; discrete energy balance; TR-BDF2
  order of accuracy; coupled ``R_ac(T)`` fixed point;
* radial thermal mismatch: two-layer Lame closed forms (plane strain and
  generalised plane strain), unilateral opening on cooling;
* isothermal limit: all layers at ``T_ref`` reproduce the unenergised solved
  fatigue and rig results bit for bit;
* monotonic trends: clamping, thermal wire stress and slip-onset curvature
  rise with temperature (expansion), the stick modulus falls (softening).
"""

from __future__ import annotations

import itertools
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.thermal_coupling import (
    ThermalHistory,
    ThermalState,
    armour_thermal_state,
    compute_thermal_fatigue,
    thermal_fatigue_inputs,
    thermal_material_state,
)
from cablexsec.cable_adapters import cable_to_fatigue_layers
from cablexsec.cables.schema import (
    _LAYER_ADAPTER,
    CableData,
    CableMeta,
    EnvironmentProperties,
)
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.materials import library
from cablexsec.mechanics.concentric_section import ElasticAnnulus
from cablexsec.mechanics.radial_cable import (
    solve_cable_radial_state,
    solve_cable_radial_states,
)
from cablexsec.mechanics.radial_solution import RadialSection
from cablexsec.mechanics.thermal import (
    compute_thermal,
    proximity_effect_factor,
    skin_effect_factor,
)
from cablexsec.mechanics.thermal_network import thermal_network

sys.path.insert(0, str(Path(__file__).parent))
from test_thermal import _bm5_loss_chain  # noqa: E402

RATED_A = 1093.0  # [IEATask49] Table 13, 630 mm2 (125 MW)


@pytest.fixture(scope="module")
def tb880() -> CableData:
    return cx.load_cable(cx.cigre_tb880_yaml_path(), geometry_override=None)


@pytest.fixture(scope="module")
def t66() -> CableData:
    return cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)


def _bm5_inputs() -> dict[str, float | int]:
    losses = _bm5_loss_chain()
    r_dc = (1.7241e-8 / 1200e-6) * (1.0 + 3.93e-3 * 70.0)
    return {
        "n_cores": 3,
        "lambda1": losses["lambda1"],
        "y_s": skin_effect_factor(r_dc, 50.0),
        "y_p": proximity_effect_factor(r_dc, 50.0, 0.042, 0.13175),
        "T4_external_Km_W": 0.3211,
        "ambient_C": 15.0,
    }


# --------------------------------------------------------------------------- #
# Thermal network                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.analytical_sanity
@pytest.mark.parametrize("n_per_layer", [1, 3, 8])
def test_network_steady_state_equals_compute_thermal_bm5(tb880, n_per_layer):
    """BM5 inputs: the RC ladder's steady state is the IEC profile (136.06 C)."""
    kw = _bm5_inputs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = compute_thermal(tb880, 908.45, 1200e-6, **kw)
    net = thermal_network(tb880, 1200e-6, n_per_layer=n_per_layer, **kw)
    temps = net.steady_state(908.45, resistance_temperature_C=90.0)
    assert ref.T_conductor_C == pytest.approx(136.06, abs=0.01)
    assert temps[0] == pytest.approx(ref.T_conductor_C, rel=1e-12)
    # Annulus edges are nodes of both profiles: identical temperatures there.
    for r, t_ref in zip(ref.r_m, ref.T_C, strict=True):
        hit = np.flatnonzero(np.isclose(net.radius_m, r, rtol=0, atol=1e-12))
        if hit.size:
            assert temps[hit[0]] == pytest.approx(t_ref, rel=1e-11)
    # Energy balance: heat through T4 equals the heat generated.
    heat_out = (temps[-1] - 15.0) / net.external_resistance_Km_W
    heat_in = net.total_heat_W_m(908.45, temps[0], resistance_temperature_C=90.0)
    assert heat_out == pytest.approx(heat_in, rel=1e-10)
    assert heat_in == pytest.approx(ref.P_total_W_m, rel=1e-12)


@pytest.mark.analytical_sanity
def test_network_steady_state_equals_compute_thermal_66kv(t66):
    """Three-core template: conductor and armour temperatures as compute_thermal."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = compute_thermal(
            t66, RATED_A, 630e-6, conductor_material="copper_janocha2024"
        )
    net = thermal_network(t66, 630e-6, conductor_material="copper_janocha2024")
    temps = net.steady_state(RATED_A, resistance_temperature_C=90.0)
    # The 0.10 mm copper tape screens make the chain stiff: agreement to its
    # conditioning.
    assert temps[0] == pytest.approx(ref.T_conductor_C, rel=1e-8)
    layer_t = net.layer_temperatures(temps)
    assert float(layer_t["cu_conductor"]) == temps[0]
    # Layer temperatures fall from the conductor to the sheath.
    order = [
        "cu_conductor",
        "xlpe_insulation",
        "mdpe_core_sheath",
        "mdpe_filler",
        "mdpe_inner_sheath",
        "inner_armour",
        "outer_armour",
        "hdpe_outer_sheath",
    ]
    values = [float(layer_t[name]) for name in order]
    assert all(a > b for a, b in itertools.pairwise(values))
    assert float(layer_t["inner_armour"]) == pytest.approx(ref.T_armour_C, abs=0.5)


@pytest.mark.analytical_sanity
def test_coupled_resistance_is_a_fixed_point(t66):
    """R_ac(T_c) solved implicitly: re-evaluating R_ac at T_c changes nothing."""
    net = thermal_network(t66, 630e-6, conductor_material="copper_janocha2024")
    coupled = net.steady_state(RATED_A)
    fixed = net.steady_state(RATED_A, resistance_temperature_C=float(coupled[0]))
    # To the conditioning of the 0.10 mm copper tape screens.
    np.testing.assert_allclose(coupled, fixed, rtol=1e-8)
    with pytest.raises(CableInputError, match="thermal runaway"):
        net.steady_state(1.0e5)


@pytest.mark.analytical_sanity
def test_crank_nicolson_energy_balance_is_exact(t66):
    """sum C dT = h/2 (Q_n + Q_n+1) - h/2 (q_out,n + q_out,n+1) to round-off."""
    net = thermal_network(t66, 630e-6, conductor_material="copper_janocha2024")
    t = np.linspace(0.0, 6 * 3600.0, 73)
    current = RATED_A * (0.3 + 0.7 * (t > 1800.0))
    temps = net.transient(
        t,
        current,
        initial="ambient",
        scheme="crank-nicolson",
        max_step_s=1e9,
        resistance_temperature_C=90.0,
    )
    c = net.capacitance_J_mK
    g_ext = 1.0 / net.external_resistance_Km_W
    for i in range(t.size - 1):
        h = t[i + 1] - t[i]
        stored = float(np.sum(c * (temps[i + 1] - temps[i])))
        q = [
            net.total_heat_W_m(current[j], temps[j, 0], resistance_temperature_C=90.0)
            - g_ext * (temps[j, -1] - net.ambient_C)
            for j in (i, i + 1)
        ]
        # Round-off of the stiff chain (0.10 mm copper tape screens).
        assert stored == pytest.approx(0.5 * h * (q[0] + q[1]), rel=1e-7, abs=1e-6)


@pytest.mark.analytical_sanity
def test_transient_limits_and_second_order(t66):
    """Steady start stays steady; step response reaches steady; TR-BDF2 order 2."""
    net = thermal_network(t66, 630e-6, conductor_material="copper_janocha2024")
    t = np.linspace(0.0, 4 * 3600.0, 5)
    hold = net.transient(t, RATED_A)
    # Steady to the conditioning of the 0.10 mm copper tape screens [K].
    assert float(np.max(np.abs(hold - hold[0]))) < 1e-6
    long_t = np.array([0.0, 30 * 86400.0])
    step = net.transient(long_t, RATED_A, initial="ambient", max_step_s=600.0)
    np.testing.assert_allclose(step[-1], net.steady_state(RATED_A), atol=1e-3)
    # Second order: errors at h, h/2 against a fine solution fall by about 4.
    t2 = np.array([0.0, 3 * 3600.0])
    ramp = [0.0, RATED_A]
    fine = net.transient(t2, ramp, initial="ambient", max_step_s=5.0)[-1, 0]
    errors = [
        abs(net.transient(t2, ramp, initial="ambient", max_step_s=h)[-1, 0] - fine)
        for h in (600.0, 300.0)
    ]
    assert 3.0 < errors[0] / errors[1] < 5.0
    tau = net.time_constants_s()
    assert tau[0] > 3600.0  # the cable responds over hours (Pi1 << 1)


def test_periodic_daily_cycle(t66):
    hist = ThermalHistory.from_current(
        t66,
        np.linspace(0.0, 86400.0, 25),
        RATED_A * 0.5 * (1 - np.cos(2 * np.pi * np.linspace(0.0, 1.0, 25))),
        periodic=True,
    )
    tc = hist.conductor_temperature_C
    assert tc is not None
    assert abs(tc[-1] - tc[0]) < 1e-3
    assert 15.0 < tc.min() < tc.max() < 90.0
    blocks = hist.blocks()
    assert len(blocks) == 24
    assert sum(b.weight for b in blocks) == pytest.approx(1.0, rel=1e-14)
    # Block mean of a linear segment is its midpoint value.
    name = "inner_armour"
    mid = 0.5 * (hist.layer_temperature_C[name][3] + hist.layer_temperature_C[name][4])
    assert blocks[3].state.layer_temperature_C[name] == pytest.approx(mid, rel=1e-14)
    with pytest.raises(CableInputError, match="periodic"):
        ThermalHistory.from_current(t66, [0.0, 3600.0], [0.0, 100.0], periodic=True)


# --------------------------------------------------------------------------- #
# Radial thermal mismatch: closed forms                                       #
# --------------------------------------------------------------------------- #
_E1, _NU1, _A1 = 1.0e9, 0.40, 1.0e-4  # polymer core
_E2, _NU2, _A2 = 70.0e9, 0.33, 23.0e-6  # aluminium tube
_B, _C = 0.020, 0.022


def _lame(e: float, nu: float) -> tuple[float, float]:
    return e * nu / ((1 + nu) * (1 - 2 * nu)), e / (2 * (1 + nu))


_MATS = (_E1, _NU1, _A1, _E2, _NU2, _A2)


def _closed_form_pressure(
    delta_t: float,
    *,
    free_axial: bool,
    mats: tuple[float, float, float, float, float, float] = _MATS,
) -> float:
    """Contact pressure of a solid core in a tube under uniform heating.

    Plane strain: explicit compatibility of the two Lame solutions. Free
    axial: generalised plane strain with zero axial force (4 x 4 system).
    """
    em1, nu1, a1, em2, nu2, a2 = mats
    e1, e2 = a1 * delta_t, a2 * delta_t
    if not free_axial:
        compliance = (1 + nu1) * (1 - 2 * nu1) / em1 + (1 + nu2) * (
            (1 - 2 * nu2) * _B**2 + _C**2
        ) / (em2 * (_C**2 - _B**2))
        return max(((1 + nu1) * e1 - (1 + nu2) * e2) / compliance, 0.0)
    l1, g1 = _lame(em1, nu1)
    l2, g2 = _lame(em2, nu2)
    # Unknowns: C1, C2, D2, eps. sigma_r = 2(l+g)C - 2g D/r^2 + l eps - (3l+2g)eT
    a = np.array(
        [
            [_B, -_B, -1.0 / _B, 0.0],
            [2 * (l1 + g1), -2 * (l2 + g2), 2 * g2 / _B**2, l1 - l2],
            [0.0, 2 * (l2 + g2), -2 * g2 / _C**2, l2],
            [
                2 * l1 * _B**2,
                2 * l2 * (_C**2 - _B**2),
                0.0,
                (l1 + 2 * g1) * _B**2 + (l2 + 2 * g2) * (_C**2 - _B**2),
            ],
        ]
    )
    rhs = np.array(
        [
            0.0,
            (3 * l1 + 2 * g1) * e1 - (3 * l2 + 2 * g2) * e2,
            (3 * l2 + 2 * g2) * e2,
            (3 * l1 + 2 * g1) * e1 * _B**2 + (3 * l2 + 2 * g2) * e2 * (_C**2 - _B**2),
        ]
    )
    c1, _c2, _d2, eps = np.linalg.solve(a, rhs)
    sigma_r = 2 * (l1 + g1) * c1 + l1 * eps - (3 * l1 + 2 * g1) * e1
    return max(-sigma_r, 0.0)


@pytest.mark.analytical_sanity
@pytest.mark.parametrize("delta_t", [40.0, 10.0, -20.0])
def test_two_layer_thermal_mismatch_pressure_closed_form(delta_t):
    """RadialSection reproduces the Lame contact pressure; cooling opens."""
    section = RadialSection(
        [
            ElasticAnnulus("core", 0.0, _B, _E1, _NU1, _A1 * delta_t),
            ElasticAnnulus("tube", _B, _C, _E2, _NU2, _A2 * delta_t),
        ]
    )
    plane = section.solve(prescribed_axial_strain=0.0)
    free = section.solve(tension_N=0.0)
    for result, free_axial in ((plane, False), (free, True)):
        p = float(result.interface_line_force_N_per_m[0]) / (2 * math.pi * _B)
        expected = _closed_form_pressure(delta_t, free_axial=free_axial)
        if expected == 0.0:
            assert p == pytest.approx(0.0, abs=1e-6)
        else:
            assert p == pytest.approx(expected, rel=1e-9)


def _two_layer_cable() -> CableData:
    layers = [
        {
            "name": "core",
            "layer_type": "cylindrical",
            "material": "hdpe",
            "r_inner_m": 0.0,
            "r_outer_m": _B,
        },
        {
            "name": "tube",
            "layer_type": "cylindrical",
            "material": "aluminium_corrugated",
            "r_inner_m": _B,
            "r_outer_m": _C,
        },
    ]
    return CableData(
        cable=CableMeta(name="two-layer thermal mismatch"),
        layers=tuple(_LAYER_ADAPTER.validate_python(x) for x in layers),
        environment=EnvironmentProperties(ambient_C=15.0, T4_external_Km_W=0.3),
    )


@pytest.mark.analytical_sanity
def test_cable_path_thermal_state_matches_closed_form():
    """ThermalState -> solved radial state equals the Lame pressure (free axial)."""
    cable = _two_layer_cable()
    e1, nu1, a1 = (library.get_property("hdpe", k) for k in ("E_Pa", "nu", "alpha_T_K"))
    e2, nu2, a2 = (
        library.get_property("aluminium_corrugated", k)
        for k in ("E_Pa", "nu", "alpha_T_K")
    )
    delta_t = 30.0
    state = ThermalState(
        {"core": 20.0 + delta_t, "tube": 20.0 + delta_t},
        temperature_dependent_modulus=False,
    )
    material = thermal_material_state(cable, state)
    solved = solve_cable_radial_states(cable, (0.0,), **material.radial_inputs)[0]
    p = solved.interface_line_force("core", "tube") / (2 * math.pi * _B)
    expected = _closed_form_pressure(
        delta_t, free_axial=True, mats=(e1, nu1, a1, e2, nu2, a2)
    )
    assert p == pytest.approx(expected, rel=1e-9)
    assert p > 0


# --------------------------------------------------------------------------- #
# Isothermal limit: bitwise                                                   #
# --------------------------------------------------------------------------- #
def _record(n: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    t = np.arange(n, dtype=float)
    kx = 0.004 * np.sin(2 * np.pi * t / 10) + 4e-4 * rng.standard_normal(n)
    ky = 0.002 * np.sin(2 * np.pi * t / 13 + 0.4)
    tension = 50e3 + 5e3 * np.sin(2 * np.pi * t / 10)
    return t, kx, ky, tension


def _same(a, b) -> bool:
    return all(
        np.array_equal(a.D_total_per_wire[k], b.D_total_per_wire[k])
        and np.array_equal(a.D_bulk_per_fibre[k], b.D_bulk_per_fibre[k])
        for k in a.D_total_per_wire
    )


@pytest.mark.analytical_sanity
def test_isothermal_limit_is_bitwise_fatigue(t66):
    _t, kx, ky, tension = _record()
    iso = ThermalHistory.isothermal(t66)
    base = cx.compute_fatigue(t66, kx, ky, tension_N=tension, contact_model="solved")
    assert _same(base, cx.compute_fatigue(t66, kx, ky, tension_N=tension, thermal=iso))
    base0 = cx.compute_fatigue(t66, kx, ky, contact_model="solved")
    state = iso.blocks()[0].state
    assert state.is_reference
    assert _same(base0, cx.compute_fatigue(t66, kx, ky, thermal=state))
    # A two-node history whose blocks are both at T_ref, placed by time_s.
    two = ThermalHistory.uniform(t66, [0.0, 1000.0], 20.0)
    assert _same(
        base,
        cx.compute_fatigue(
            t66, kx, ky, tension_N=tension, thermal=two, time_s=np.arange(kx.size)
        ),
    )


@pytest.mark.analytical_sanity
def test_identical_blocks_equal_a_single_state(t66):
    """Two blocks with the same state give the single-state result bit for bit."""
    t, kx, ky, tension = _record(300)
    hot = ThermalHistory.uniform(t66, [0.0, 150.0, 300.0], 45.0)
    single = cx.compute_fatigue(
        t66, kx, ky, tension_N=tension, thermal=hot.blocks()[0].state
    )
    multi = cx.compute_fatigue(t66, kx, ky, tension_N=tension, thermal=hot, time_s=t)
    assert _same(single, multi)


@pytest.mark.analytical_sanity
def test_isothermal_limit_is_bitwise_rig(t66):
    proto = cx.RigProtocol.cyclic(
        tension_N=50e3, curvature_amplitude_1_m=0.01, n_cycles=1, samples_per_cycle=16
    )
    setup = cx.RigSetup(free_length_m=4.0, clamp="none")
    gauges = [cx.GaugeLocation("inner_armour", a, 2.0) for a in (0.0, 90.0)]
    kw = {"gauges": gauges, "mu_factors": (), "contact_model": "solved"}
    base = cx.simulate_rig_test(t66, proto, setup, **kw)
    iso = cx.simulate_rig_test(
        t66, proto, setup, thermal=ThermalHistory.isothermal(t66), **kw
    )
    for a, b in zip(base.gauges, iso.gauges, strict=True):
        assert np.array_equal(a.total, b.total)
    hot = cx.simulate_rig_test(
        t66, proto, setup, thermal=ThermalHistory.uniform(t66, [0.0], 50.0), **kw
    )
    # Heating raises the clamping and the sectional friction-strain cap.
    assert hot.layers[0].contact_pressure_Pa > base.layers[0].contact_pressure_Pa
    assert (
        hot.layers[0].sectional_cap_microstrain
        > base.layers[0].sectional_cap_microstrain
    )
    with pytest.raises(CableInputError, match="not both"):
        cx.simulate_rig_test(
            t66,
            cx.RigProtocol.cyclic(
                tension_N=5e4, curvature_amplitude_1_m=0.01, temperature_C=40.0
            ),
            setup,
            thermal=ThermalHistory.isothermal(t66),
            **kw,
        )


# --------------------------------------------------------------------------- #
# Monotonic trends                                                            #
# --------------------------------------------------------------------------- #
def test_heating_raises_clamping_and_thermal_stress(t66):
    """Expansion (template moduli constant): P1+P2, kappa_c and sigma_T rise."""
    rows = [
        armour_thermal_state(
            t66,
            ThermalHistory.uniform(t66, [0.0], temp).blocks()[0].state,
            tension_N=50e3,
        )
        for temp in (25.0, 40.0, 55.0)
    ]
    for name in ("inner_armour", "outer_armour"):
        p = [r[name].contact_pressure_Pa for r in rows]
        kc = [r[name].slip_onset_curvature_1_m for r in rows]
        st = [r[name].thermal_wire_stress_Pa for r in rows]
        assert p[0] < p[1] < p[2]
        assert kc[0] < kc[1] < kc[2]
        assert 0 < st[0] < st[1] < st[2]
        assert all(r[name].pi4 > 0 for r in rows)


def test_softening_lowers_the_stick_modulus(t66):
    """Transferred E(T) without expansion: k_f falls monotonically with T."""
    transfer = {
        "xlpe_janocha2024": "xlpe_qiao2025",
        "mdpe_janocha2024": "hdpe",
        "hdpe_janocha2024": "hdpe",
    }
    stick = []
    for temp in (30.0, 50.0, 70.0):
        state = (
            ThermalHistory.uniform(
                t66, [0.0], temp, thermal_expansion=False, modulus_transfer=transfer
            )
            .blocks()[0]
            .state
        )
        stick.append(
            armour_thermal_state(t66, state, tension_N=50e3)[
                "inner_armour"
            ].stick_modulus_Pa
        )
    assert stick[0] > stick[1] > stick[2]
    # At T_ref the transfer ratio is exactly one.
    ref = ThermalState(
        {layer.name: 20.0 for layer in t66.layers}, modulus_transfer=transfer
    )
    assert thermal_material_state(t66, ref).is_reference


def test_steady_currents_order_the_temperatures(t66):
    temps = [
        ThermalHistory.steady(t66, f * RATED_A).blocks()[0].state
        for f in (0.0, 0.5, 1.0)
    ]
    tc = [s.conductor_temperature_C for s in temps]
    assert tc[0] == pytest.approx(15.0, abs=1e-6)  # no current: ambient
    # Rated current: 97 deg C with T_ref at the ambient; the semiconducting
    # screens count with the insulation (doc/template_66kv.md Section 4).
    assert tc[0] < tc[1] < tc[2] < 100.0
    arm = [s.layer_temperature_C["inner_armour"] for s in temps]
    assert arm[0] < arm[1] < arm[2] < tc[2]


# --------------------------------------------------------------------------- #
# Block-weighted fatigue and inputs                                           #
# --------------------------------------------------------------------------- #
def test_compute_thermal_fatigue_blocks_and_diagnostics(t66):
    _t, kx, ky, tension = _record(300)
    hist = ThermalHistory.from_current(
        t66, [0.0, 3600.0, 7200.0], [0.0, RATED_A, 0.0], initial="ambient"
    )
    reports: list[tuple[float, str]] = []
    res = compute_thermal_fatigue(
        t66,
        kx,
        ky,
        hist,
        tension_N=tension,
        record_duration_s=300.0,
        progress=lambda share, text: reports.append((share, text)),
    )
    assert reports == [
        (0.25, "Block 1 of 2"),
        (0.5, "Block 2 of 2"),
        (0.75, "Unenergised reference"),
        (1.0, "Time-mean temperatures"),
    ]
    assert len(res.blocks) == 2
    assert sum(b.block.weight for b in res.blocks) == pytest.approx(1.0)
    for name, values in res.D_total_per_wire.items():
        expected = sum(
            b.block.weight * b.fatigue.D_total_per_wire[name] for b in res.blocks
        )
        np.testing.assert_allclose(values, expected, rtol=1e-14)
    assert math.isfinite(res.pi1) and res.pi1 < 1e-2
    for b in res.blocks:
        assert b.frictional_heat_W_m > 0.0
        assert 0.0 < b.pi2 < 1e-2  # one-way coupling holds (Pi2 << 1)
        # H2: frictional heating raises the armour temperature negligibly.
        assert 0.0 < b.friction_temperature_rise_K < 0.5
        assert all(v > 0 for v in b.pi3.values())
    assert res.damage_ratio_reference > 0
    assert set(res.hypotheses) == {"H1", "H2", "H3", "H4", "H5", "H6"}
    assert all(v >= 0 for v in res.D_thermal_cycle.values())
    assert res.thermal_stress_range_Pa["inner_armour"] > 0


def test_inputs_vary_between_blocks_and_stick_history(t66):
    t, kx, _ky, tension = _record(200)
    transfer = {"mdpe_janocha2024": "hdpe", "xlpe_janocha2024": "xlpe_qiao2025"}
    hist = ThermalHistory.uniform(
        t66,
        [0.0, 100.0, 101.0, 200.0],
        [25.0, 25.0, 60.0, 60.0],
        modulus_transfer=transfer,
    )
    layers = cable_to_fatigue_layers(
        t66, contact_tension_N=float(tension[0]), contact_model="solved"
    )
    inputs = thermal_fatigue_inputs(t66, layers, t.size, tension, hist, time_s=t)
    assert len(inputs.states) == 3
    assert inputs.stick_modulus_Pa is not None
    assert inputs.thermal_stress_Pa is not None
    assert inputs.contact_pressure_Pa is not None
    for name in ("inner_armour", "outer_armour"):
        k = inputs.stick_modulus_Pa[name]
        assert np.unique(k).size == 3  # one value per block
        assert k[0] > k[-1]  # the hot block is softer
    res = cx.compute_fatigue(t66, kx, kx * 0, tension_N=tension, thermal=hist, time_s=t)
    assert any("thermal coupling" in x for x in res.provenance.limitations)
    with pytest.raises(CableSchemaError, match="constant stick modulus"):
        cx.compute_fatigue(
            t66,
            kx,
            kx * 0,
            tension_N=tension,
            thermal=hist,
            time_s=t,
            friction_model="distributed",
        )


def test_input_validation(t66):
    _t, kx, ky, tension = _record(50)
    iso = ThermalHistory.isothermal(t66)
    with pytest.raises(CableSchemaError, match="solved"):
        cx.compute_fatigue(t66, kx, ky, thermal=iso, contact_model="bounded")
    with pytest.raises(CableSchemaError, match="time_s"):
        cx.compute_fatigue(t66, kx, ky, time_s=np.arange(kx.size))
    two = ThermalHistory.uniform(t66, [0.0, 10.0], [20.0, 30.0])
    with pytest.raises(CableInputError, match="time_s"):
        cx.compute_fatigue(t66, kx, ky, thermal=two)
    with pytest.raises(CableInputError, match="outside"):
        cx.compute_fatigue(t66, kx, ky, thermal=two, time_s=np.arange(kx.size) + 20.0)
    with pytest.raises(CableInputError, match="no temperature"):
        thermal_material_state(t66, ThermalState({"cu_conductor": 50.0}))
    with pytest.raises(CableInputError, match="registered"):
        ThermalState({"a": 20.0}, modulus_transfer={"x": "no_such_law"})
    with pytest.raises(CableInputError, match="increasing"):
        ThermalHistory.uniform(t66, [0.0, 0.0], 20.0)
    # Registered E(T) laws refuse to extrapolate beyond their source range.
    tb = cx.load_cable(cx.cigre_tb880_yaml_path(), geometry_override=None)
    hot = ThermalHistory.uniform(tb, [0.0], 120.0).blocks()[0].state
    with pytest.raises(CableInputError, match="outside the sourced"):
        thermal_material_state(tb, hot)


def test_solve_cable_radial_states_matches_single_solves(t66):
    temps = {layer.name: 45.0 for layer in t66.layers}
    many = solve_cable_radial_states(t66, (0.0, 50e3), temperature_C=temps)
    for tension, state in zip((0.0, 50e3), many, strict=True):
        one = solve_cable_radial_state(t66, tension, temperature_C=temps)
        assert one.armour_contact_pressure() == state.armour_contact_pressure()
        assert dict(one.result.wire_stress_Pa) == dict(state.result.wire_stress_Pa)


def test_network_input_validation(t66):
    net = thermal_network(t66, 630e-6, conductor_material="copper_janocha2024")
    bad_calls = [
        (lambda: net.transient([0.0], 100.0), ">= 2 samples"),
        (lambda: net.transient([0.0, 0.0], 100.0), "strictly increasing"),
        (lambda: net.transient([0.0, 1.0], 100.0, scheme="euler"), "scheme"),
        (lambda: net.transient([0.0, 1.0], 100.0, max_step_s=0.0), "positive"),
        (lambda: net.transient([0.0, 1.0], 100.0, initial="hot"), "initial"),
        (lambda: net.transient([0.0, 1.0], 100.0, initial=np.zeros(3)), "match"),
        (lambda: net.transient([0.0, 1.0], [1.0, 2.0, 3.0]), "current_A"),
        (lambda: net.steady_state(math.nan), "finite"),
        (lambda: net.layer_temperatures(np.zeros(3)), "match"),
        (lambda: net.resistance_to_ambient_Km_W("nope"), "unknown layer"),
    ]
    for call, message in bad_calls:
        with pytest.raises(CableInputError, match=message):
            call()
    kw = {"conductor_material": "copper_janocha2024"}
    for options, message in (
        ({"T4_external_Km_W": -1.0}, "T4"),
        ({"n_per_layer": 0}, "n_per_layer"),
        ({"y_s": -0.1}, "non-negative"),
        ({"n_cores": 0}, "at least 1"),
    ):
        with pytest.raises(CableInputError, match=message):
            thermal_network(t66, 630e-6, **kw, **options)
    with pytest.raises(CableInputError, match="conductor_area_m2"):
        thermal_network(t66, 0.0, **kw)
    # Frictional heat at the inner armour raises it through the outer layers.
    r_arm = net.resistance_to_ambient_Km_W("inner_armour")
    assert net.external_resistance_Km_W < r_arm < 1.0
    # Crank-Nicolson with periodic regime agrees with TR-BDF2.
    tau = np.linspace(0.0, 86400.0, 13)
    cur = 600.0 * (1 - np.cos(2 * np.pi * tau / 86400.0))
    a = net.transient(tau, cur, periodic=True, max_step_s=120.0)
    b = net.transient(
        tau, cur, periodic=True, scheme="crank-nicolson", max_step_s=120.0
    )
    assert float(np.max(np.abs(a - b))) < 0.05


def test_history_and_coupling_validation(t66, tb880):
    with pytest.raises(CableInputError, match="non-empty"):
        ThermalState({})
    with pytest.raises(CableInputError, match="1-D"):
        ThermalHistory(time_s=np.zeros((2, 2)), layer_temperature_C={"a": 1.0})
    with pytest.raises(CableInputError, match="required"):
        ThermalHistory(time_s=[0.0], layer_temperature_C={})
    with pytest.raises(CableInputError, match="match time_s"):
        ThermalHistory(time_s=[0.0, 1.0], layer_temperature_C={"a": [1.0, 2.0, 3.0]})
    with pytest.raises(CableInputError, match="current_A"):
        ThermalHistory(
            time_s=[0.0, 1.0], layer_temperature_C={"a": 20.0}, current_A=[1.0]
        )
    hist = ThermalHistory.uniform(t66, [0.0, 10.0], [20.0, 40.0])
    with pytest.raises(CableInputError, match="block edges"):
        hist.blocks([0.0, 20.0])
    assert hist.with_options(thermal_expansion=False).thermal_expansion is False
    mean = hist.mean_state()
    assert mean.layer_temperature_C["inner_armour"] == pytest.approx(30.0)
    assert hist.state_at(5.0).layer_temperature_C["inner_armour"] == pytest.approx(30.0)
    # One block over the whole span is the time mean.
    edge = hist.blocks([0.0, 10.0])[0].state
    assert edge.layer_temperature_C["inner_armour"] == pytest.approx(30.0)
    # Registered laws act on the TB 880 polymers (citations recorded).
    hot = ThermalHistory.uniform(tb880, [0.0], 50.0).blocks()[0].state
    material = thermal_material_state(tb880, hot)
    assert "xlpe_insulation" in material.modulus_sources
    assert material.xi_moduli_Pa is not None
    no_effect = (
        ThermalHistory.uniform(t66, [0.0], 40.0, thermal_expansion=False)
        .blocks()[0]
        .state
    )
    assert not thermal_material_state(t66, no_effect).radial_inputs
    with pytest.raises(CableInputError, match="conductor_area_m2"):
        ThermalHistory.steady(_two_layer_cable(), 100.0)
    _t, kx, ky, _tension = _record(40)
    layers = cable_to_fatigue_layers(t66, contact_model="solved")
    state = hist.blocks()[0].state
    with pytest.raises(CableInputError, match="only with a ThermalHistory"):
        thermal_fatigue_inputs(
            t66, layers, kx.size, None, state, time_s=np.arange(40.0)
        )
    with pytest.raises(CableInputError, match="ThermalHistory or a ThermalState"):
        thermal_fatigue_inputs(t66, layers, kx.size, None, "hot")  # type: ignore[arg-type]
    with pytest.raises(CableInputError, match="non-decreasing"):
        thermal_fatigue_inputs(t66, layers, kx.size, None, hist, time_s=np.zeros(3))
    # No tension, hot state: pressure history from the solved state at 0 N.
    inputs = thermal_fatigue_inputs(t66, layers, kx.size, None, state)
    assert inputs.contact_pressure_Pa is not None
    for message, call in (
        ("ThermalHistory", lambda: compute_thermal_fatigue(t66, kx, ky, state)),
        (
            "set by compute_thermal_fatigue",
            lambda: compute_thermal_fatigue(t66, kx, ky, hist, contact_model="solved"),
        ),
        (
            "positive",
            lambda: compute_thermal_fatigue(t66, kx, ky, hist, record_duration_s=0.0),
        ),
    ):
        with pytest.raises(CableInputError, match=message):
            call()
