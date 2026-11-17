# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Armour friction model of the fatigue path under biaxial curvature.

Gates:

* the out-of-plane ratio of a curvature path: zero for a planar path in any
  plane (and below the tolerance after rounding to six digits), ``b/a`` for
  an ellipse, one for a circle, scale invariant;
* the default of ``compute_fatigue`` stays the direction-tied model, bit
  for bit (results and provenance), and warns once per call
  (``DirectionTiedFrictionWarning``, a ``FutureWarning`` announcing the
  change of the default to ``"auto"``) only when it is left at the default
  on a non-planar path;
* ``friction_model="auto"`` (opt-in) is the direction-tied model on planar
  paths, bit for bit, and the distributed helix model on any other path,
  bit for bit;
* under steadily rotating curvature at twice the slip onset the
  direction-tied cap gives worst-wire friction-stress ranges below the
  distributed model (0.86 of them on a near-rigid interface, 0.94 on the
  template interface of the 66 kV inner armour) and too little friction
  work (0.38-0.69 of the planar cycle against 1.8-2.6 for the distributed
  model); on planar and flat elliptical paths its ranges are at or above
  the distributed ones;
* layers the distributed model does not cover keep the direction-tied model
  and say why; a constant stick modulus other than the layer's enters as the
  Yun coefficient of that state;
* the bound form of the distributed model: the uniform bound gives the
  linear Coulomb full-slip field ``sigma_0 lam x``, the same cycle work as
  the signed bound within 1e-3, and the signed default is unchanged;
* the documented pitch-mean offset of the direction-tied cap in planar full
  slip, ``sigma_0 (sinh x0 / x0 - 1)``, ``x0 = lam pi / 2``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.fatigue import (
    PLANAR_PATH_TOLERANCE,
    compute_fatigue,
    curvature_path_out_of_plane_ratio,
    wire_stress_field,
)
from cablexsec.cable_adapters import cable_to_fatigue_layers
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.distributed_friction import (
    HAVE_NUMBA,
    distributed_friction_history,
)
from cablexsec.mechanics.friction_law import FrictionLaw
from cablexsec.mechanics.papailiou import slip_onset_curvature, wire_slip_history
from cablexsec.provenance import (
    DirectionTiedFrictionWarning,
    FrictionModelFallbackWarning,
)

compiled = pytest.mark.skipif(
    not HAVE_NUMBA, reason="long friction histories need the compiled kernels"
)

POINTS = 360  # samples per revolution or cycle
TEMPLATE_XI = None  # keep the template interface (xi = 0.813, inner armour)


@pytest.fixture(scope="module")
def inner():
    cable = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cable_to_fatigue_layers(cable, contact_tension_N=100e3)[0]


def _with_xi(layer, xi):
    return layer if xi is None else replace(layer, spec=replace(layer.spec, xi=xi))


def _orbit(radius, ry, revolutions=4, ramp=90):
    """Ramp to ``(radius, 0)``, then ellipses of semi-axes ``radius``, ``ry radius``."""
    th = np.linspace(0.0, 2.0 * math.pi * revolutions, revolutions * POINTS + 1)
    r = np.linspace(0.0, radius, ramp + 1)[:-1]
    return (
        np.r_[r, radius * np.cos(th)],
        np.r_[0.0 * r, ry * radius * np.sin(th)],
    )


def _worst_range(stress):
    tail = stress[:, -POINTS - 1 :]
    return float(np.max(tail.max(axis=1) - tail.min(axis=1)))


def _friction_range(layer, kx, ky, model):
    field = wire_stress_field(layer, kx, ky, friction_model=model, keep_friction=True)
    assert field.friction_Pa is not None
    return _worst_range(field.friction_Pa)


def _last_cycle(cumulative):
    return float(cumulative[-1] - cumulative[-1 - POINTS])


# --------------------------------------------------------------------------- #
# planarity                                                                    #
# --------------------------------------------------------------------------- #
def test_out_of_plane_ratio():
    t = np.linspace(0.0, 30.0, 3001)
    s = 0.01 * np.sin(t) * (1.0 + 0.3 * np.sin(0.21 * t))
    for angle in (0.0, 0.4, math.pi / 2, 2.0, -2.9):
        kx, ky = s * math.cos(angle), s * math.sin(angle)
        assert curvature_path_out_of_plane_ratio(kx, ky) < 1e-15
        # Output rounded to six significant digits stays planar.
        rx = np.array([float(f"{v:.6g}") for v in kx])
        ry = np.array([float(f"{v:.6g}") for v in ky])
        assert curvature_path_out_of_plane_ratio(rx, ry) < 1e-5
    th = np.linspace(0.0, 2.0 * math.pi, 721)
    for ratio in (1e-4, 0.1, 0.5, 1.0):
        kx, ky = 0.02 * np.cos(th), ratio * 0.02 * np.sin(th)
        # Rotate the ellipse: the measure has no preferred axis.
        c, sn = math.cos(1.1), math.sin(1.1)
        value = curvature_path_out_of_plane_ratio(c * kx - sn * ky, sn * kx + c * ky)
        assert value == pytest.approx(ratio, rel=1e-9, abs=1e-15)
        for scale in (1e-200, 1e200):
            assert curvature_path_out_of_plane_ratio(
                scale * kx, scale * ky
            ) == pytest.approx(ratio, rel=1e-9, abs=1e-15)
    assert curvature_path_out_of_plane_ratio(np.zeros(5), np.zeros(5)) == 0.0
    assert curvature_path_out_of_plane_ratio(np.zeros(0), np.zeros(0)) == 0.0
    with pytest.raises(CableInputError, match="same shape"):
        curvature_path_out_of_plane_ratio(np.zeros(3), np.zeros(4))
    # A static offset in y with cycling in x is not planar.
    assert curvature_path_out_of_plane_ratio(np.sin(th), np.full(th.size, 0.5)) > 0.1
    assert curvature_path_out_of_plane_ratio(np.sin(th), 2e-3 * np.cos(th)) > (
        PLANAR_PATH_TOLERANCE
    )
    # A static bend with cycles normal to its plane is judged against the
    # cycles (or 1e-2 of the bend, if larger), not against the bend: cycles
    # of 8e-4, 8e-3 and 8e-2 of the bend give 0.08, 0.8 and 1.
    static = np.full(th.size, 0.05)
    cycles = 4e-5 * np.sin(th)
    for factor, expected in ((1.0, 0.08), (10.0, 0.8), (100.0, 1.0)):
        assert curvature_path_out_of_plane_ratio(
            static, factor * cycles
        ) == pytest.approx(expected, rel=1e-4)
        assert curvature_path_out_of_plane_ratio(
            -factor * cycles, static
        ) == pytest.approx(expected, rel=1e-4)
    # Rounded to six significant digits, a static bend in an oblique plane
    # stays planar.
    oblique = [float(f"{v:.6g}") for v in (0.05 * math.cos(0.4), 0.05 * math.sin(0.4))]
    noisy = np.array([1.0, 1.0 + 1e-6, 1.0 - 1e-6]) * 0.05
    rx = np.array([float(f"{v * math.cos(0.4):.6g}") for v in noisy] + oblique[:1])
    ry = np.array([float(f"{v * math.sin(0.4):.6g}") for v in noisy] + oblique[1:])
    assert curvature_path_out_of_plane_ratio(rx, ry) < PLANAR_PATH_TOLERANCE
    # Six significant digits round a mantissa near 1 by up to 5e-6 of the
    # value: the worst case stays planar too, at any angle.
    for angle in (0.347, *np.linspace(0.0, math.pi, 181)):
        for radius in (0.015, 0.0100001, 0.0999999):
            noisy = radius * np.array([1.0, 1.000001, 0.999999])
            rx = np.array([float(f"{v * math.cos(angle):.6g}") for v in noisy])
            ry = np.array([float(f"{v * math.sin(angle):.6g}") for v in noisy])
            value = curvature_path_out_of_plane_ratio(rx, ry)
            assert value < PLANAR_PATH_TOLERANCE, (angle, radius, value)
    # The same cycles in the plane of the bend stay planar, in any plane.
    for angle in (0.0, 0.4, 2.0):
        s = static + cycles
        value = curvature_path_out_of_plane_ratio(
            s * math.cos(angle), s * math.sin(angle)
        )
        assert value < 1e-10
    # A constant curvature keeps one plane.
    assert curvature_path_out_of_plane_ratio(np.full(5, 0.01), np.full(5, 0.02)) < 1e-9


# --------------------------------------------------------------------------- #
# the default                                                                  #
# --------------------------------------------------------------------------- #
def _damage_equal(a, b):
    assert a.D_total_per_wire.keys() == b.D_total_per_wire.keys()
    for name in a.D_total_per_wire:
        assert np.array_equal(a.D_total_per_wire[name], b.D_total_per_wire[name])
        assert np.array_equal(a.D_bulk_per_fibre[name], b.D_bulk_per_fibre[name])


def _calls(record):
    return [w for w in record if issubclass(w.category, DirectionTiedFrictionWarning)]


def test_default_stays_direction_tied_and_warns_on_non_planar_paths(inner):
    on = slip_onset_curvature(inner.spec)
    cable = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
    for ry in (0.0, 0.5):
        kx, ky = _orbit(2.0 * on, ry, revolutions=1)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            default = compute_fatigue([inner], kx, ky)
            explicit = compute_fatigue([inner], kx, ky, friction_model="direction_tied")
            public = cx.compute_fatigue(cable, kx, ky, contact_tension_N=1e5)
            public_explicit = cx.compute_fatigue(
                cable, kx, ky, contact_tension_N=1e5, friction_model="direction_tied"
            )
        _damage_equal(default, explicit)
        _damage_equal(public, public_explicit)
        assert default.provenance == explicit.provenance
        assert public.provenance == public_explicit.provenance
        assert "'auto'" not in default.provenance.method
        calls = _calls(record)
        assert not any("non-planar" in x for x in default.provenance.limitations)
        if ry == 0.0:
            assert not calls
        else:
            # Once per call, at the caller's line, for the two default calls.
            assert len(calls) == 2
            assert all(w.filename == __file__ for w in calls)
            text = str(calls[0].message)
            assert "will change from 'direction_tied' to 'auto'" in text
            assert "0.86" in text and "0.94" in text
            assert issubclass(DirectionTiedFrictionWarning, FutureWarning)


def test_auto_is_direction_tied_on_planar_paths(inner):
    on = slip_onset_curvature(inner.spec)
    kx, ky = _orbit(3.0 * on, 0.0, revolutions=2)
    for angle in (0.0, 0.9):
        x, y = kx * math.cos(angle), kx * math.sin(angle)
        auto = compute_fatigue([inner], x, y, friction_model="auto")
        explicit = compute_fatigue([inner], x, y, friction_model="direction_tied")
        _damage_equal(auto, explicit)
        method = auto.provenance.method
        assert "'auto'" in method and "inner_armour: direction_tied" in method
        assert not any("non-planar" in x for x in auto.provenance.limitations)
    field = wire_stress_field(inner, kx, ky, friction_model="auto")
    tied = wire_stress_field(inner, kx, ky)
    assert np.array_equal(field.mean_Pa, tied.mean_Pa)


@compiled
@pytest.mark.independent_method
def test_auto_is_distributed_on_biaxial_paths(inner):
    on = slip_onset_curvature(inner.spec)
    kx, ky = _orbit(2.0 * on, 0.5, revolutions=2)
    default = compute_fatigue([inner], kx, ky, friction_model="auto")
    distributed = compute_fatigue([inner], kx, ky, friction_model="distributed")
    _damage_equal(default, distributed)
    assert "inner_armour: distributed" in default.provenance.method
    assert "distributed elastic-Coulomb helix model" in default.provenance.method
    tied = compute_fatigue([inner], kx, ky, friction_model="direction_tied")
    assert not any("non-planar" in x for x in tied.provenance.limitations)
    field = wire_stress_field(inner, kx, ky, friction_model="auto")
    helix = wire_stress_field(inner, kx, ky, friction_model="distributed")
    assert np.array_equal(field.mean_Pa, helix.mean_Pa)


@compiled
@pytest.mark.independent_method
@pytest.mark.parametrize(
    ("xi", "rotation_ceiling"),
    [(0.99, 0.90), (TEMPLATE_XI, 0.97)],
    ids=["near_rigid", "template"],
)
def test_rotation_direction_tied_below_distributed(inner, xi, rotation_ceiling):
    layer = _with_xi(inner, xi)
    on = slip_onset_curvature(layer.spec)
    ratios = {}
    for label, ry in (("circle", 1.0), ("ellipse", 0.5), ("planar", 0.0)):
        kx, ky = _orbit(2.0 * on, ry)
        tied = _friction_range(layer, kx, ky, "direction_tied")
        helix = _friction_range(layer, kx, ky, "distributed")
        ratios[label] = tied / helix
        assert _friction_range(layer, kx, ky, "auto") == (helix if ry > 0 else tied)
    # Measured: 0.863 (near rigid) and 0.943 (template) on the circle.
    assert ratios["circle"] < rotation_ceiling
    # Flat ellipses and planar paths: the direction-tied ranges are not lower.
    assert ratios["ellipse"] >= 1.0
    assert ratios["planar"] >= 1.0
    assert ratios["planar"] < 1.1


@compiled
@pytest.mark.independent_method
@pytest.mark.parametrize("amplitude", [2.0, 8.0])
def test_rotation_friction_work_against_planar(inner, amplitude):
    layer = _with_xi(inner, 0.99)
    spec = layer.spec
    on = slip_onset_curvature(spec)
    work = {}
    for label, ry in (("circle", 1.0), ("planar", 0.0)):
        kx, ky = _orbit(amplitude * on, ry)
        work[label] = (
            _last_cycle(wire_slip_history(spec, kx, ky).work.friction_J_per_m),
            _last_cycle(distributed_friction_history(spec, kx, ky).dissipation_J_per_m),
        )
    tied = work["circle"][0] / work["planar"][0]
    helix = work["circle"][1] / work["planar"][1]
    # Measured: 0.38 and 0.69 (direction-tied), 2.62 and 1.82 (distributed).
    assert tied < 0.75
    assert helix > 1.75


# --------------------------------------------------------------------------- #
# fallbacks                                                                    #
# --------------------------------------------------------------------------- #
def _short_rotation(layer, n=48):
    on = slip_onset_curvature(layer.spec)
    th = np.linspace(0.0, 2.0 * math.pi, n)
    return 2.0 * on * np.cos(th), 2.0 * on * np.sin(th)


def _fallback_case(result, explicit, reason):
    _damage_equal(result, explicit)
    assert "inner_armour: direction_tied" in result.provenance.method
    notes = [x for x in result.provenance.limitations if "kept the direction" in x]
    assert len(notes) == 1 and reason in notes[0]
    assert any("non-planar" in x for x in result.provenance.limitations)


def test_rigid_interface_keeps_direction_tied(inner):
    rigid = _with_xi(inner, 1.0)
    kx, ky = _short_rotation(rigid)
    with pytest.warns(FrictionModelFallbackWarning, match="rigid") as calls:
        result = compute_fatigue([rigid], kx, ky, friction_model="auto")
    assert len(calls) == 1 and calls[0].filename == __file__
    explicit = compute_fatigue([rigid], kx, ky, friction_model="direction_tied")
    _fallback_case(result, explicit, "rigid")


def test_rate_dependent_law_keeps_direction_tied(inner):
    kx, ky = _short_rotation(inner)
    law = FrictionLaw(rate_coefficient=0.05)
    time = np.linspace(0.0, 10.0, kx.size)
    with pytest.warns(FrictionModelFallbackWarning, match="rate dependent"):
        result = compute_fatigue(
            [inner], kx, ky, friction_law=law, time_s=time, friction_model="auto"
        )
    with pytest.warns(FrictionModelFallbackWarning, match="rate dependent") as calls:
        wire_stress_field(
            inner, kx, ky, friction_law=law, time_s=time, friction_model="auto"
        )
    assert len(calls) == 1 and calls[0].filename == __file__
    explicit = compute_fatigue(
        [inner], kx, ky, friction_law=law, time_s=time, friction_model="direction_tied"
    )
    _fallback_case(result, explicit, "rate dependent")


def test_varying_stick_modulus_keeps_direction_tied(inner):
    kx, ky = _short_rotation(inner)
    stick = {"inner_armour": inner.spec.k_f * np.linspace(1.0, 0.9, kx.size)}
    with pytest.warns(FrictionModelFallbackWarning, match="stick modulus changes"):
        result = compute_fatigue(
            [inner], kx, ky, stick_modulus_Pa=stick, friction_model="auto"
        )
    explicit = compute_fatigue(
        [inner], kx, ky, stick_modulus_Pa=stick, friction_model="direction_tied"
    )
    _fallback_case(result, explicit, "stick modulus changes")


@compiled
def test_constant_stick_modulus_is_the_yun_coefficient(inner):
    kx, ky = _short_rotation(inner, 120)
    factor = 0.9
    stick = {"inner_armour": np.full(kx.size, factor * inner.spec.k_f)}
    result = compute_fatigue(
        [inner], kx, ky, stick_modulus_Pa=stick, friction_model="auto"
    )
    spec = inner.spec
    # The Yun coefficient of that stick modulus, k / (E cos^2 a).
    xi = factor * spec.k_f / (spec.E_wire_Pa * math.cos(spec.lay_angle_rad) ** 2)
    assert xi == pytest.approx(factor * spec.xi, rel=1e-14)
    softer = _with_xi(inner, xi)
    reference = compute_fatigue([softer], kx, ky, friction_model="distributed")
    _damage_equal(result, reference)
    assert "inner_armour: distributed" in result.provenance.method
    assert f"k_hat = {softer.spec.xi / (1 - softer.spec.xi):.6g}" in str(
        result.provenance
    )
    explicit = compute_fatigue(
        [inner], kx, ky, stick_modulus_Pa=stick, friction_model="distributed"
    )
    _damage_equal(explicit, reference)


@pytest.mark.filterwarnings("error::cablexsec.provenance.FrictionModelFallbackWarning")
def test_no_fallback_warning_on_planar_paths_or_explicit_models(inner):
    on = slip_onset_curvature(inner.spec)
    kx, _ = _orbit(3.0 * on, 0.0, revolutions=1)
    law = FrictionLaw(rate_coefficient=0.05)
    time = np.linspace(0.0, 10.0, kx.size)
    # Planar: 'auto' is the direction-tied law, no fallback to report.
    compute_fatigue(
        [inner], kx, 0.0 * kx, friction_law=law, time_s=time, friction_model="auto"
    )
    rx, ry = _short_rotation(inner)
    t = np.linspace(0.0, 10.0, rx.size)
    compute_fatigue(
        [inner], rx, ry, friction_law=law, time_s=t, friction_model="direction_tied"
    )


@compiled
def test_auto_sees_cycles_normal_to_a_static_bend(inner):
    # Cycles at twice the slip onset normal to a static bend 2500 times
    # larger: against the largest curvature they are 8e-4 out of plane (below
    # the tolerance), against their own range fully out of plane.
    on = slip_onset_curvature(inner.spec)
    th = np.linspace(0.0, 4.0 * math.pi, 241)
    kx = np.full(th.size, 2500.0 * 2.0 * on)
    ky = 2.0 * on * np.sin(th)
    assert float(np.max(np.abs(ky))) / float(np.max(np.hypot(kx, ky))) < (
        PLANAR_PATH_TOLERANCE
    )
    auto = wire_stress_field(inner, kx, ky, friction_model="auto")
    helix = wire_stress_field(inner, kx, ky, friction_model="distributed")
    assert np.array_equal(auto.mean_Pa, helix.mean_Pa)


def test_progressive_settings_still_need_progressive(inner):
    from cablexsec.mechanics.progressive_friction import ProgressiveFriction

    kx, ky = _short_rotation(inner)
    with pytest.raises(CableInputError, match="progressive"):
        compute_fatigue(
            [inner], kx, ky, progressive=ProgressiveFriction(), friction_model="auto"
        )


# --------------------------------------------------------------------------- #
# bound form of the distributed model                                         #
# --------------------------------------------------------------------------- #
def test_bound_option_validation(inner):
    kx, ky = _short_rotation(inner, 8)
    with pytest.raises(CableInputError, match="bound"):
        distributed_friction_history(inner.spec, kx, ky, bound="total")
    with pytest.raises(CableInputError, match="distributed_bound"):
        compute_fatigue([inner], kx, ky, distributed_bound="total")
    with pytest.raises(CableInputError, match="distributed_bound"):
        wire_stress_field(inner, kx, ky, distributed_bound="total")


@compiled
@pytest.mark.analytical_sanity
def test_uniform_bound_full_slip_field_is_linear_coulomb(inner):
    spec = inner.spec
    on = slip_onset_curvature(spec)
    amplitude = 400.0 * on
    kx = np.linspace(0.0, amplitude, 801)
    uniform = distributed_friction_history(spec, kx, 0.0 * kx, bound="uniform")
    signed = distributed_friction_history(spec, kx, 0.0 * kx)
    unit = uniform.parameters["stress_unit_Pa"]
    assert unit == signed.parameters["stress_unit_Pa"]
    assert uniform.parameters["bound"] == "uniform"
    assert signed.parameters["bound"] == "signed"
    # Azimuth from the neutral axis toward the tension side (phi = 0).
    phi = np.angle(np.exp(1j * spec.phi_wires))
    x = math.pi / 2.0 - np.abs(phi)
    # The elastic interface rounds the field over O(kappa_on / kappa) radians.
    assert np.max(np.abs(uniform.stress_Pa[:, -1] / unit - x)) < 2e-2
    lam = uniform.parameters["lam"]
    c = lam * math.pi / (2.0 * math.sinh(lam * math.pi / 2.0))
    sigma0 = unit / lam
    closed = sigma0 * (c * np.exp(lam * x) - 1.0)
    assert np.max(np.abs(signed.stress_Pa[:, -1] - closed)) < 2e-2 * unit
    # Both full-slip fields span the linear Coulomb range sigma_0 lam pi.
    for history in (uniform, signed):
        span = np.ptp(history.stress_Pa[:, -1])
        assert span == pytest.approx(unit * math.pi, rel=2e-2)


@compiled
@pytest.mark.independent_method
def test_uniform_bound_cycle_work_matches_signed(inner):
    spec = inner.spec
    on = slip_onset_curvature(spec)
    for amplitude in (2.0, 32.0):
        kx, _ky = _orbit(amplitude * on, 0.0, revolutions=3)
        signed = distributed_friction_history(spec, kx, 0.0 * kx)
        uniform = distributed_friction_history(spec, kx, 0.0 * kx, bound="uniform")
        ratio = _last_cycle(signed.dissipation_J_per_m) / _last_cycle(
            uniform.dissipation_J_per_m
        )
        # Measured: 1.00009 and 0.99965 (inner armour, lam = 0.045).
        assert ratio == pytest.approx(1.0, abs=1e-3)
        assert abs(np.max(np.abs(uniform.balance_residual_J_per_m))) <= 1e-9 * abs(
            uniform.dissipation_J_per_m[-1]
        )
    default = distributed_friction_history(spec, kx, 0.0 * kx)
    assert np.array_equal(default.stress_Pa, signed.stress_Pa)
    result = compute_fatigue(
        [inner], kx, 0.5 * kx[::-1], friction_model="auto", distributed_bound="uniform"
    )
    assert "uniform bound without the capstan term" in result.provenance.method


# --------------------------------------------------------------------------- #
# documented pitch-mean offset of the direction-tied cap                       #
# --------------------------------------------------------------------------- #
@pytest.mark.analytical_sanity
def test_direction_tied_pitch_mean_offset_in_full_slip(inner):
    spec = inner.spec
    on = slip_onset_curvature(spec)
    kx = np.linspace(0.0, 64.0 * on, 65)
    stress = wire_slip_history(spec, kx, 0.0 * kx).stress_Pa[:, -1]
    lam = spec.mu * abs(math.sin(spec.lay_angle_rad))
    x0 = lam * math.pi / 2.0
    width = 2.0 * math.pi * spec.r_helix_m / spec.n_wires
    sigma0 = (
        spec.r_helix_m
        * spec.contact_pressure_Pa
        * width
        * math.cos(spec.lay_angle_rad)
        / (spec.A_wire_m2 * math.sin(spec.lay_angle_rad) ** 2)
    )
    offset = sigma0 * (math.sinh(x0) / x0 - 1.0)
    # Wire sampling of the pitch and the stick zone (O(kappa_on / kappa)).
    assert float(np.mean(stress)) == pytest.approx(offset, rel=3e-2)
    # Leading order lam pi / 12 of the peak tensile friction stress.
    peak = sigma0 * math.expm1(x0)
    assert offset / peak == pytest.approx(lam * math.pi / 12.0, rel=lam)
    # A uniform shift of every wire leaves the moment unchanged (round-off).
    arm = np.cos(spec.phi_wires)
    assert abs(float(np.mean(stress) * arm.sum())) < 1e-12 * float(
        np.abs(stress * arm).sum()
    )
