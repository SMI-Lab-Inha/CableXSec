# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Contact-pressure-dependent friction coefficient (``FrictionLaw.pressure_factor``).

Closed forms on one spring-slider under pressure steps, the per-wire full-slip
bound at the local pressure ``(P1+P2) g``, energy identities, a constant or
unit table against the scaled or neutral coefficient, the compiled
direction-tied kernel against the NumPy reference, the distributed kernel and
the entry points (bending, fatigue, exported bending law).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.bending_law import export_bending_law
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.mechanics import friction_law as flm
from cablexsec.mechanics.distributed_friction import distributed_friction_history
from cablexsec.mechanics.fast_slip import wire_friction_stress
from cablexsec.mechanics.friction_law import FrictionLaw, slider_history
from cablexsec.mechanics.mkappa import HelicalLayerSpec
from cablexsec.mechanics.papailiou import full_slip_curvature, wire_slip_history

analytical = pytest.mark.analytical_sanity

#: Factor falling with pressure, as measured for wire on anti-wear tape.
TABLE = ((1.0e6, 1.5), (2.0e6, 1.0), (4.0e6, 0.8))
PRESSURE_ONLY = FrictionLaw(pressure_factor=TABLE)
BREAKAWAY_P = FrictionLaw(static_factor=1.3, dynamic_factor=1.0, pressure_factor=TABLE)
COMBINED_P = FrictionLaw(
    static_factor=1.25,
    dynamic_factor=0.9,
    stick_time_s=0.2,
    rate_coefficient=0.05,
    rate_reference_m_per_s=1.0e-5,
    viscous_velocity_m_per_s=1.0e-3,
    pressure_factor=TABLE,
)
LAWS_P = [PRESSURE_ONLY, BREAKAWAY_P, COMBINED_P]


def _spec(n_wires: int = 12, shape: bool = True, pressure: float = 2.0e6):
    g = None
    if shape:
        phi = 2 * np.pi * np.arange(n_wires) / n_wires
        g = 1.0 + 0.3 * np.cos(phi)
    return HelicalLayerSpec(
        n_wires=n_wires,
        r_helix_m=0.05,
        lay_angle_rad=0.4,
        E_wire_Pa=2.0e11,
        A_wire_m2=2.0e-5,
        xi=0.3,
        mu=0.2,
        F_n_axi_N=0.0,
        shape_g=g,
        contact_pressure_Pa=pressure,
    )


def _history(spec: HelicalLayerSpec, n: int = 1200, seed: int = 4):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 12.0, n)
    big = 3.0 * full_slip_curvature(spec)
    kx = big * np.sin(2 * np.pi * 0.25 * t) + 0.01 * big * rng.standard_normal(n)
    ky = 0.4 * big * np.sin(2 * np.pi * 0.13 * t)
    kx[200:240] = 0.0
    ky[200:240] = 0.0
    # 1.2-2.8 MPa, times g of 0.7-1.3: local pressures below, inside and
    # above the table.
    pressure = spec.contact_pressure_Pa * (1 + 0.4 * np.sin(np.linspace(0, 9, n)))
    return t, kx, ky, pressure


def _time(law: FrictionLaw, t: np.ndarray) -> np.ndarray | None:
    return t if law.needs_time else None


def _constant(factor: float) -> FrictionLaw:
    return FrictionLaw(pressure_factor=((0.0, factor), (1.0e9, factor)))


# --------------------------------------------------------------------------- #
# Settings                                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "table",
    [
        ((1.0e6, 1.0),),
        ((1.0e6, 1.0), (1.0e6, 0.9)),
        ((2.0e6, 1.0), (1.0e6, 0.9)),
        ((-1.0, 1.0), (1.0e6, 0.9)),
        ((0.0, 1.0), (1.0e6, 0.0)),
        ((0.0, 1.0), (math.inf, 0.9)),
        ((0.0, math.nan), (1.0e6, 0.9)),
        ((0.0, 1.0, 2.0), (1.0e6, 0.9, 2.0)),
        "table",
        ((0.0, 1.0 + 1j), (1.0e6, 0.9)),
        ((0.0, "a"), (1.0e6, 0.9)),
        5.0,
    ],
)
def test_pressure_table_is_checked(table):
    with pytest.raises(CableInputError, match="pressure_factor"):
        FrictionLaw(pressure_factor=table)


def test_pressure_table_properties():
    law = FrictionLaw(pressure_factor=np.array(TABLE))
    assert law.pressure_factor == TABLE
    assert law == PRESSURE_ONLY and hash(law) == hash(PRESSURE_ONLY)
    assert law.pressure_dependent and law.coefficient_neutral
    assert not law.is_neutral and not law.needs_time
    assert flm.active_friction_law(law) is law
    assert law.minimum_factor == 0.8
    assert BREAKAWAY_P.minimum_factor == 0.8
    assert COMBINED_P.minimum_factor == pytest.approx(0.72, rel=1e-15)
    assert "pressure factor" in law.describe() and "4 MPa" in law.describe()
    np.testing.assert_allclose(
        law.pressure_factor_at(np.array([0.0, 1.0e6, 1.5e6, 3.0e6, 4.0e6, 1.0e8])),
        [1.5, 1.5, 1.25, 0.9, 0.8, 0.8],
        rtol=1e-15,
    )
    np.testing.assert_array_equal(law.pressure_table, np.array(TABLE).T)
    # A table of unit factors is no pressure dependence.
    unit = FrictionLaw(pressure_factor=((0.0, 1.0), (1.0e6, 1.0)))
    assert unit.is_neutral and not unit.pressure_dependent
    assert unit.pressure_table.shape == (2, 0)
    np.testing.assert_array_equal(unit.pressure_factor_at([1.0, 2.0]), 1.0)
    assert FrictionLaw().pressure_table.shape == (2, 0)


@analytical
def test_compiled_factor_matches_numpy_interp():
    rng = np.random.default_rng(3)
    table = np.ascontiguousarray(
        np.array([[0.5e6, 1.0e6, 2.5e6, 8.0e6], [1.7, 1.3, 1.0, 0.85]])
    )
    law = FrictionLaw(pressure_factor=tuple(map(tuple, table.T)))
    points = np.concatenate([rng.uniform(0.0, 1.0e7, 400), table[0], [0.0, 1e12]])
    got = np.array([flm._pressure_factor(float(p), table) for p in points])
    np.testing.assert_allclose(got, law.pressure_factor_at(points), rtol=0, atol=1e-15)
    assert flm._pressure_factor(3.0e6, np.zeros((2, 0))) == 1.0


# --------------------------------------------------------------------------- #
# One spring-slider                                                            #
# --------------------------------------------------------------------------- #
@analytical
def test_slider_under_pressure_steps():
    """Full slip: the force is ``f(P) c P``; a falling bound clips the
    slipping force at the step, a rising one leaves it in stick until the
    spring reaches the new bound (``c P`` the bound at the coefficient)."""
    k, c, step = 2.0e6, 5.0e-6, 1.0e-6
    law = FrictionLaw(pressure_factor=((1.0e6, 1.6), (4.0e6, 1.0), (8.0e6, 0.9)))
    drive = step * np.arange(1, 601)
    pressure = np.full(drive.size, 8.0e6)
    pressure[200:400] = 1.0e6  # step down
    pressure[400:] = 4.0e6  # step up
    s = slider_history(
        drive, c * pressure, k, friction_law=law, contact_pressure_Pa=pressure
    )
    high, low, mid = 36.0, 8.0, 20.0  # 5e-6 x {8e6 x 0.9, 1e6 x 1.6, 4e6 x 1.0}
    np.testing.assert_allclose(s.force_N[150:200], high, rtol=1e-15)
    np.testing.assert_allclose(s.force_N[200:400], low, rtol=1e-15)
    rise = s.force_N[400:] - low
    elastic = rise < mid - low
    np.testing.assert_allclose(
        rise[elastic], k * step * np.arange(1, elastic.sum() + 1), rtol=1e-12
    )
    np.testing.assert_allclose(s.force_N[400:][~elastic], mid, rtol=1e-15)
    # The clipped force at the step down: (36 - 8)^2 / (2 k) leaves the spring;
    # friction work takes 8 (36 - 8) / k, the remainder the rest.
    assert abs(s.balance_J).max() <= 1e-14 * s.input_work_J[-1]


@analytical
def test_slider_pressure_outside_the_table_and_zero_contact():
    k, c = 2.0e6, 5.0e-6
    drive = 1.0e-6 * np.arange(1, 201)
    for pressure, factor in ((0.5e6, 1.5), (9.0e6, 0.8)):
        s = slider_history(
            drive,
            c * pressure,
            k,
            friction_law=PRESSURE_ONLY,
            contact_pressure_Pa=pressure,
        )
        np.testing.assert_allclose(s.force_N[-50:], factor * c * pressure, rtol=1e-15)
    zero = slider_history(
        drive, 0.0, k, friction_law=PRESSURE_ONLY, contact_pressure_Pa=0.0
    )
    np.testing.assert_array_equal(zero.force_N, 0.0)
    np.testing.assert_array_equal(zero.friction_J, 0.0)


@analytical
def test_slider_breakaway_at_the_pressure_factor():
    """The static peak and dynamic level both carry ``f(P)``."""
    k, step, pressure = 2.0e6, 1.0e-7, 1.5e6
    drive = step * np.arange(1, 401)
    s = slider_history(
        drive, 10.0, k, friction_law=BREAKAWAY_P, contact_pressure_Pa=pressure
    )
    factor = 1.25
    assert 1.3 * factor * 10.0 - k * step < s.force_N.max() <= 1.3 * factor * 10.0
    np.testing.assert_allclose(s.force_N[-50:], factor * 10.0, rtol=1e-15)


@analytical
@pytest.mark.parametrize("law", LAWS_P)
def test_slider_energy_identity_with_a_pressure_history(law):
    rng = np.random.default_rng(11)
    time = np.cumsum(rng.uniform(0.001, 0.02, 3000))
    drive = 2.0e-5 * np.cumsum(rng.standard_normal(3000))
    pressure = 2.0e6 * (1.0 + 0.8 * np.sin(0.7 * time)) + 1.0e5
    s = slider_history(
        drive,
        2.5e-6 * pressure,
        1.0e6,
        friction_law=law,
        time_s=_time(law, time),
        contact_pressure_Pa=pressure,
    )
    assert np.all(np.diff(s.friction_J) >= 0.0)
    assert abs(s.balance_J).max() <= 1e-12 * s.input_work_J.max()


def test_slider_pressure_input_is_checked():
    drive = 1.0e-6 * np.arange(1, 11)
    with pytest.raises(CableInputError, match="give contact_pressure_Pa"):
        slider_history(drive, 1.0, 1.0e6, friction_law=PRESSURE_ONLY)
    with pytest.raises(CableInputError, match="used only"):
        slider_history(drive, 1.0, 1.0e6, contact_pressure_Pa=1.0e6)
    for bad in (-1.0, math.nan, np.ones(3), 1.0e6 + 1j):
        with pytest.raises(CableInputError, match="contact_pressure_Pa"):
            slider_history(
                drive, 1.0, 1.0e6, friction_law=PRESSURE_ONLY, contact_pressure_Pa=bad
            )


# --------------------------------------------------------------------------- #
# Direction-tied kernel                                                        #
# --------------------------------------------------------------------------- #
@analytical
def test_full_slip_bound_at_the_local_pressure():
    """Monotonic planar bending far beyond full slip: every wire sits on its
    capstan bound with the coefficient ``f((P1+P2) g) mu`` of its own local
    pressure (the shape ``g`` spreads the wires over the table)."""
    spec = _spec()
    kappa = np.linspace(0.0, 6.0 * full_slip_curvature(spec), 2001)[1:]
    zero = np.zeros_like(kappa)
    h = wire_slip_history(spec, kappa, zero, friction_law=PRESSURE_ONLY)
    fast = wire_friction_stress(spec, kappa, zero, friction_law=PRESSURE_ONLY)
    g = np.asarray(spec.shape_g)
    sin_a = math.sin(spec.lay_angle_rad)
    width = 2.0 * math.pi * spec.r_helix_m / spec.n_wires
    sigma0 = (
        spec.r_helix_m
        * spec.contact_pressure_Pa
        * width
        * math.cos(spec.lay_angle_rad)
        / (spec.A_wire_m2 * sin_a**2)
    )
    cphi = np.cos(spec.phi_wires)
    x = (
        PRESSURE_ONLY.pressure_factor_at(spec.contact_pressure_Pa * g)
        * spec.mu
        * sin_a
        * np.arcsin(np.abs(cphi))
    )
    expected = np.where(cphi > 0.0, sigma0 * g * np.expm1(x), sigma0 * g * np.expm1(-x))
    driven = np.abs(cphi) > 1e-9
    scale = np.abs(expected).max()
    np.testing.assert_allclose(
        h.stress_Pa[driven, -1], expected[driven], rtol=0, atol=1e-13 * scale
    )
    np.testing.assert_allclose(
        fast[driven, -1], expected[driven], rtol=0, atol=1e-12 * scale
    )
    # The factors span the table: some wires below 2 MPa, some above.
    factors = PRESSURE_ONLY.pressure_factor_at(spec.contact_pressure_Pa * g)
    assert factors.min() < 1.0 < factors.max()


def test_unit_table_is_bitwise_and_constant_table_scales_mu():
    spec = _spec()
    t, kx, ky, pressure = _history(spec)
    law = FrictionLaw(static_factor=1.3, dynamic_factor=1.0)
    unit = dataclasses.replace(law, pressure_factor=((0.0, 1.0), (5.0e6, 1.0)))
    for kwargs in ({}, {"contact_pressure_Pa": pressure}):
        np.testing.assert_array_equal(
            wire_slip_history(spec, kx, ky, friction_law=unit, **kwargs).stress_Pa,
            wire_slip_history(spec, kx, ky, friction_law=law, **kwargs).stress_Pa,
        )
        np.testing.assert_array_equal(
            wire_friction_stress(spec, kx, ky, friction_law=unit, **kwargs),
            wire_friction_stress(spec, kx, ky, friction_law=law, **kwargs),
        )
    # Constant factor 1.3: the rate-independent law with 1.3 mu.
    scaled = dataclasses.replace(spec, mu=1.3 * spec.mu)
    ref = wire_slip_history(scaled, kx, ky, contact_pressure_Pa=pressure)
    got = wire_slip_history(
        spec, kx, ky, contact_pressure_Pa=pressure, friction_law=_constant(1.3)
    )
    fast = wire_friction_stress(
        spec, kx, ky, contact_pressure_Pa=pressure, friction_law=_constant(1.3)
    )
    scale = np.abs(ref.stress_Pa).max()
    np.testing.assert_allclose(got.stress_Pa, ref.stress_Pa, rtol=0, atol=1e-12 * scale)
    np.testing.assert_allclose(fast, ref.stress_Pa, rtol=0, atol=1e-12 * scale)


@pytest.mark.parametrize("law", LAWS_P)
def test_compiled_kernel_matches_reference_with_a_pressure_factor(law):
    spec = _spec()
    t, kx, ky, pressure = _history(spec)
    ref = wire_slip_history(
        spec,
        kx,
        ky,
        contact_pressure_Pa=pressure,
        friction_law=law,
        time_s=_time(law, t),
    )
    fast = wire_friction_stress(
        spec,
        kx,
        ky,
        contact_pressure_Pa=pressure,
        friction_law=law,
        time_s=_time(law, t),
    )
    scale = np.abs(ref.stress_Pa).max()
    np.testing.assert_allclose(fast, ref.stress_Pa, rtol=0, atol=1e-12 * scale)
    # The factor changes the result.
    plain = wire_slip_history(
        spec,
        kx,
        ky,
        contact_pressure_Pa=pressure,
        friction_law=dataclasses.replace(law, pressure_factor=None),
        time_s=_time(law, t),
    )
    assert np.abs(ref.stress_Pa - plain.stress_Pa).max() > 1e-3 * scale


@analytical
@pytest.mark.parametrize("law", LAWS_P)
def test_kernel_energy_identity_with_a_pressure_factor(law):
    spec = _spec()
    t, kx, ky, pressure = _history(spec)
    h = wire_slip_history(
        spec,
        kx,
        ky,
        contact_pressure_Pa=pressure,
        friction_law=law,
        time_s=_time(law, t),
    )
    work = h.work
    assert np.all(np.diff(work.friction_J_per_m) >= 0.0)
    scale = work.dissipation_J_per_m[-1]
    assert np.abs(work.balance_residual_J_per_m).max() <= 1e-12 * scale


def test_zero_contact_with_a_pressure_factor():
    spec = _spec(pressure=0.0)
    t, kx, ky, _ = _history(_spec(), n=200)
    h = wire_slip_history(spec, kx, ky, friction_law=COMBINED_P, time_s=t)
    np.testing.assert_array_equal(h.stress_Pa, 0.0)
    fast = wire_friction_stress(spec, kx, ky, friction_law=COMBINED_P, time_s=t)
    np.testing.assert_array_equal(fast, 0.0)
    d = distributed_friction_history(spec, kx, ky, friction_law=BREAKAWAY_P)
    np.testing.assert_array_equal(d.stress_Pa, 0.0)


# --------------------------------------------------------------------------- #
# Distributed kernel                                                           #
# --------------------------------------------------------------------------- #
def _short(spec, n=160):
    t = np.linspace(0.0, 8.0, n)
    big = 3.0 * full_slip_curvature(spec)
    pressure = spec.contact_pressure_Pa * (1 + 0.4 * np.sin(np.linspace(0, 5, n)))
    return (
        t,
        big * np.sin(2 * np.pi * 0.25 * t),
        0.4 * big * np.cos(2 * np.pi * 0.13 * t),
        pressure,
    )


def test_distributed_constant_factor_and_unit_table():
    """Uniform contact at a fixed pressure: the factor is the constant
    ``f(P)``, the model with ``f(P) mu``; a unit table is bitwise neutral."""
    spec = _spec(shape=False, pressure=1.5e6)
    t, kx, ky, _ = _short(spec)
    got = distributed_friction_history(
        spec, kx, ky, friction_law=PRESSURE_ONLY, max_increment=math.inf
    )
    exp = distributed_friction_history(
        dataclasses.replace(spec, mu=1.25 * spec.mu), kx, ky, max_increment=math.inf
    )
    scale = np.abs(exp.stress_Pa).max()
    np.testing.assert_allclose(got.stress_Pa, exp.stress_Pa, rtol=0, atol=1e-10 * scale)
    np.testing.assert_allclose(
        got.dissipation_J_per_m, exp.dissipation_J_per_m, rtol=1e-10
    )
    assert "pressure factor" in got.parameters["friction_law"]
    law = FrictionLaw(static_factor=1.3, dynamic_factor=1.0)
    unit = dataclasses.replace(law, pressure_factor=((0.0, 1.0), (5.0e6, 1.0)))
    a = distributed_friction_history(_spec(), kx, ky, friction_law=law)
    b = distributed_friction_history(_spec(), kx, ky, friction_law=unit)
    np.testing.assert_array_equal(a.stress_Pa, b.stress_Pa)
    np.testing.assert_array_equal(a.dissipation_J_per_m, b.dissipation_J_per_m)


@analytical
@pytest.mark.parametrize("law", [PRESSURE_ONLY, BREAKAWAY_P])
def test_distributed_energy_with_a_pressure_factor(law):
    spec = _spec()
    t, kx, ky, pressure = _short(spec)
    h = distributed_friction_history(
        spec, kx, ky, contact_pressure_Pa=pressure, friction_law=law
    )
    plain = distributed_friction_history(
        spec,
        kx,
        ky,
        contact_pressure_Pa=pressure,
        friction_law=dataclasses.replace(law, pressure_factor=None),
    )
    scale = h.dissipation_J_per_m[-1]
    assert np.abs(h.balance_residual_J_per_m).max() <= 1e-12 * scale
    assert np.all(np.diff(h.dissipation_J_per_m) >= -1e-14 * scale)
    assert (
        np.abs(h.stress_Pa - plain.stress_Pa).max()
        > 1e-3 * np.abs(plain.stress_Pa).max()
    )


# --------------------------------------------------------------------------- #
# Entry points                                                                 #
# --------------------------------------------------------------------------- #
def _menard():
    return cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)


def test_compute_mkappa_with_a_pressure_factor():
    cable = _menard()
    kappa = 4.0e-3 * np.sin(np.linspace(0.0, 4 * np.pi, 801))
    ref = cx.compute_mkappa(cable, kappa, mu=0.2 * 1.3)
    got = cx.compute_mkappa(cable, kappa, mu=0.2, friction_law=_constant(1.3))
    scale = np.abs(ref.moment_Nm).max()
    np.testing.assert_allclose(got.moment_Nm, ref.moment_Nm, rtol=0, atol=1e-12 * scale)
    assert got.W_diss_J_per_m == pytest.approx(ref.W_diss_J_per_m, rel=1e-12)
    # A tension history moves P1+P2 and the factor with it.
    tension = 1.0e5 * (1.0 + 0.5 * np.sin(np.linspace(0.0, 9.0, kappa.size)))
    big = 12.5 * kappa
    a = cx.compute_mkappa(
        cable, big, contact_tension_history_N=tension, friction_law=PRESSURE_ONLY
    )
    b = cx.compute_mkappa(cable, big, contact_tension_history_N=tension)
    assert np.abs(a.moment_Nm - b.moment_Nm).max() > 1e-3 * np.abs(b.moment_Nm).max()
    assert a.work is not None
    assert np.abs(a.work.balance_residual_J_per_m).max() <= 1e-12 * a.W_diss_J_per_m
    assert a.provenance is not None and "pressure factor" in a.provenance.method
    with pytest.raises(CableSchemaError, match="direction-tied"):
        cx.compute_mkappa(cable, kappa, friction_law=PRESSURE_ONLY, core_friction=True)


def test_compute_fatigue_with_a_pressure_factor():
    cable = _menard()
    time = np.linspace(0.0, 100.0, 300)
    kx = 0.02 * np.sin(2 * np.pi * time / 10.0)
    ky = np.zeros_like(kx)
    for model in ("direction_tied", "distributed"):
        ref = cx.compute_fatigue(cable, kx, ky, friction_model=model)
        got = cx.compute_fatigue(
            cable, kx, ky, friction_model=model, friction_law=PRESSURE_ONLY
        )
        assert got.provenance is not None and "pressure factor" in got.provenance.method
        changed = any(
            not np.array_equal(got.D_total_per_wire[name], damage)
            for name, damage in ref.D_total_per_wire.items()
        )
        assert changed
    with pytest.raises(CableInputError, match="progressive"):
        cx.compute_fatigue(
            cable, kx, ky, friction_law=PRESSURE_ONLY, friction_model="progressive"
        )


def test_bending_law_export_with_a_pressure_factor():
    """Accepted at fixed contact (one law per tension), refused with the slip
    scaling variable (no exact homogeneity in the pressure)."""
    cable = _menard()
    args = {"tensions_N": [0.0, 2.0e5], "kappa_max_1_m": 0.05, "n_points": 20}
    law = export_bending_law(cable, friction_law=PRESSURE_ONLY, **args)
    plain = export_bending_law(cable, **args)
    for i, tension in enumerate(args["tensions_N"]):
        backbone = cx.compute_mkappa(
            cable, law.kappa_1_m, tension_N=tension, friction_law=PRESSURE_ONLY
        ).moment_Nm
        np.testing.assert_array_equal(law.moment_Nm[i, 1:], backbone[1:])
    assert not np.array_equal(law.moment_Nm, plain.moment_Nm)
    with pytest.raises(CableInputError, match="homogeneous"):
        export_bending_law(cable, friction_law=PRESSURE_ONLY, slip_scale=True, **args)
    with pytest.raises(CableInputError, match="Masing"):
        export_bending_law(cable, friction_law=BREAKAWAY_P, **args)
