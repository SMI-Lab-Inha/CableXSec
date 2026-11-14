# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for the IEC 60287 radial thermal solver (§6.10): gate A9 + solver."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from cablexsec.cables.schema import CableData, CableMeta, EnvironmentProperties
from cablexsec.materials.library import get_property
from cablexsec.mechanics.thermal import (
    ConductionLayer,
    ac_resistance,
    compute_thermal,
    dielectric_loss,
    iec_conductor_temperature,
    proximity_effect_factor,
    sheath_circulating_loss_factor,
    skin_effect_factor,
    solve_crank_nicolson,
    steady_state_profile,
    thermal_resistance,
)

# Test-taxonomy category.
pytestmark = pytest.mark.analytical_sanity


def _stack() -> list[ConductionLayer]:
    # Conductor surface 10 mm -> insulation -> sheath, ambient at the outer face.
    return [
        ConductionLayer(0.010, 0.020, 0.286, 2.4e6),  # XLPE insulation
        ConductionLayer(0.020, 0.024, 0.38, 1.9e6),  # HDPE sheath
    ]


# --------------------------------------------------------------------------- #
# A9 — closed-form concentric-cylinder steady state (the analytical invariant) #
# --------------------------------------------------------------------------- #
def test_A9_series_resistance_closed_form() -> None:
    """T_conductor = T_amb + W * sum(R_i), R_i = ln(r_o/r_i)/(2π λ)."""
    layers = _stack()
    w, t_amb = 20.0, 15.0
    r_sum = sum(
        thermal_resistance(lyr.r_inner_m, lyr.r_outer_m, lyr.lambda_W_mK)
        for lyr in layers
    )
    expected_Tc = t_amb + w * r_sum
    r, t = steady_state_profile(layers, w, t_amb, n_per_layer=1)
    assert t[0] == pytest.approx(expected_Tc, rel=1e-12)  # inner node = conductor
    assert t[-1] == pytest.approx(t_amb, rel=1e-12)  # outer node = ambient


def test_A9_profile_is_logarithmic_within_layer() -> None:
    """Within an annulus T(r) = T_o + (W/2πλ) ln(r_o/r) (pure conduction)."""
    layers = [ConductionLayer(0.01, 0.05, 0.3, 2.0e6)]
    w, t_amb = 30.0, 10.0
    r, t = steady_state_profile(layers, w, t_amb, n_per_layer=20)
    for ri, ti in zip(r, t, strict=True):
        expected = t_amb + w / (2.0 * math.pi * 0.3) * math.log(0.05 / ri)
        assert ti == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------- #
# Crank-Nicolson solver disciplined against A9 (< 0.1 K)                       #
# --------------------------------------------------------------------------- #
def test_solver_converges_to_A9_steady_state() -> None:
    """CN t->inf limit reproduces the closed-form A9 conductor temperature."""
    layers = _stack()
    w, t_amb = 20.0, 15.0
    _r_a, t_a = steady_state_profile(layers, w, t_amb, n_per_layer=1)
    r_n, t_n = solve_crank_nicolson(layers, w, t_amb, n_per_layer=30)
    assert abs(t_n[0] - t_a[0]) < 0.1  # conductor temperature within 0.1 K
    assert t_n[-1] == pytest.approx(t_amb, abs=1e-9)  # outer BC held


def test_solver_zero_loss_is_isothermal() -> None:
    layers = _stack()
    _r, t = solve_crank_nicolson(layers, 0.0, 12.0, n_per_layer=20, t_end_s=1.0e5)
    assert np.allclose(t, 12.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# Heat sources ([IEC60287-1-1])                                               #
# --------------------------------------------------------------------------- #
def test_ac_resistance_temperature_dependence() -> None:
    """R_dc(T) = R_dc,20 (1 + α (T-20)); skin/proximity multiply it."""
    rho, area, alpha = 1.724e-8, 886.0e-6, 3.93e-3
    r20 = ac_resistance(rho, area, 20.0, alpha)
    assert r20 == pytest.approx(rho / area, rel=1e-12)
    r90 = ac_resistance(rho, area, 90.0, alpha)
    assert r90 == pytest.approx((rho / area) * (1.0 + alpha * 70.0), rel=1e-12)
    r90_skin = ac_resistance(rho, area, 90.0, alpha, y_s=0.1, y_p=0.05)
    assert r90_skin == pytest.approx(r90 * 1.15, rel=1e-12)


def test_dielectric_loss_threshold_100kV() -> None:
    """Dielectric loss is off below 100 kV (a 66 kV cable) and on above."""
    assert dielectric_loss(66.0, 2.5, 0.001, 0.015, 0.025) == 0.0  # 66 kV: off
    w_d = dielectric_loss(220.0, 2.5, 0.001, 0.025, 0.045)  # export: on
    assert w_d > 0.0


def test_skin_and_proximity_factors_iec_hand_values() -> None:
    """y_s, y_p for the BM5 conductor match an IEC 60287-1-1 hand calculation."""
    r_dc = (1.7241e-8 / 1200e-6) * (1.0 + 3.93e-3 * 70.0)  # 1200 mm^2 Cu at 90 C
    ys = skin_effect_factor(r_dc, 50.0)
    yp = proximity_effect_factor(r_dc, 50.0, 0.042, 0.13175)
    assert ys == pytest.approx(0.2049, abs=2e-3)
    assert yp == pytest.approx(0.0524, abs=2e-3)


# --------------------------------------------------------------------------- #
# IEC ladder (Robin T4) reduces to the conduction primitive — reduction gate  #
# --------------------------------------------------------------------------- #
def test_iec_ladder_reduces_to_conduction_primitive() -> None:
    """θc(n=1, W_d=0, λ1=0) = θ_a + W_c (ΣR_internal + T4) = primitive + W_c·T4."""
    layers = _stack()
    w_c, t_amb, t4 = 20.0, 15.0, 0.30
    r_int = sum(
        thermal_resistance(ly.r_inner_m, ly.r_outer_m, ly.lambda_W_mK) for ly in layers
    )
    # IEC ladder: split the internal resistance arbitrarily into T1 + T2.
    t1 = thermal_resistance(*(_stack()[0].r_inner_m, _stack()[0].r_outer_m, 0.286))
    t2 = r_int - t1
    theta = iec_conductor_temperature(t1, t2, 0.0, t4, W_c=w_c, T_ambient_C=t_amb)
    # Primitive with the cable surface raised by the external rise W_c·T4.
    _r, t_prof = steady_state_profile(layers, w_c, t_amb + w_c * t4, n_per_layer=1)
    assert theta == pytest.approx(t_prof[0], rel=1e-12)
    assert theta == pytest.approx(t_amb + w_c * (r_int + t4), rel=1e-12)


def _bm5_loss_chain() -> dict[str, float]:
    """IEC 60287-1-1 loss chain for the Poradowski-Anders (2025) NOTE-1 cable.

    220 kV 3-core (CIGRE TB 880), stranded Cu 1200 mm², copper corrugated
    sheath. Geometry is from the public benchmark cable.
    """
    f = 50.0
    rho20, alpha = 1.7241e-8, 3.93e-3
    r_dc = (rho20 / 1200e-6) * (1.0 + alpha * (90.0 - 20.0))
    ys = skin_effect_factor(r_dc, f)
    yp = proximity_effect_factor(r_dc, f, 0.042, 0.13175)  # d_c, trefoil spacing
    r_ac = r_dc * (1.0 + ys + yp)
    w_c = 908.45**2 * r_ac
    # Dielectric: U0 = 220/sqrt(3) kV, XLPE eps_r=2.5 tanδ=0.001 (IEC Table 3).
    w_d = dielectric_loss(220.0, 2.5, 0.001, 0.0227, 0.0466, frequency_Hz=f)
    # Sheath λ1' (circulating): Cu ring 111.95/113.95 mm, Fcor=1.3406, ~80 C.
    a_s = math.pi / 4 * (0.11395**2 - 0.11195**2)
    r_s = rho20 * 1.3406 / a_s * (1.0 + alpha * (80.0 - 20.0))
    d_s = 0.5 * (0.11195 + 0.11395)
    x = 2.0 * (2.0 * math.pi * f) * 1e-7 * math.log(2.0 * 0.13175 / d_s)
    lam1 = sheath_circulating_loss_factor(r_s, r_ac, x)
    return {"W_c": w_c, "W_d": w_d, "lambda1": lam1}


@pytest.mark.full_scale
def test_BM5_poradowski_buried_reference_case() -> None:
    """BM5: reproduce Poradowski-Anders (2025) Table 1 NOTE-1 buried 220 kV cable.

    Stated network: T1=0.6579, T2=0.1121, T3=0.0357, T4=0.3211 K.m/W (T4 the
    external soil/Robin resistance, taken verbatim); θ_amb=15 °C, n=3.
    Reference: I=908.45 A → θc=89.62 °C (FEM) / 90 °C (IEC analytical ampacity).

    Active heat sources: conductor Joule (W_c, skin+proximity) and dielectric
    (W_d, 220 kV > 100 kV) per conductor, plus the copper-sheath circulating
    loss factor λ1. The eddy-current term λ1'' ([IEC60287-1-1] §2.3.6, not in
    our reference set) is the one untabulated input; it is the documented
    residual between the circulating-only result and 89.62 °C.
    """
    t1, t2, t3, t4 = 0.6579, 0.1121, 0.0357, 0.3211
    t_amb, n, target = 15.0, 3, 89.62
    losses = _bm5_loss_chain()
    w_c, w_d, lam1 = losses["W_c"], losses["W_d"], losses["lambda1"]

    # IEC circulating-current sheath factor is the dominant non-Joule term.
    assert 1.0 < lam1 < 1.15

    theta_circ = iec_conductor_temperature(
        t1,
        t2,
        t3,
        t4,
        W_c=w_c,
        W_d=w_d,
        lambda1=lam1,
        n_conductors=n,
        T_ambient_C=t_amb,
    )
    # Circulating-only is a lower bound; residual to 89.62 is the eddy term.
    assert 83.0 < theta_circ < target
    assert (target - theta_circ) < 6.0

    # Network correctness: with the λ1 the paper's analytical ampacity implies,
    # the ladder reproduces 89.62 °C exactly (closes the residual via λ1'').
    rhs = target - t_amb - (w_c + 0.5 * w_d) * t1
    lam1_implied = (rhs / (n * (t2 + t3 + t4)) - w_d) / w_c - 1.0
    assert 1.2 < lam1_implied < 1.35
    theta_implied = iec_conductor_temperature(
        t1,
        t2,
        t3,
        t4,
        W_c=w_c,
        W_d=w_d,
        lambda1=lam1_implied,
        n_conductors=n,
        T_ambient_C=t_amb,
    )
    assert theta_implied == pytest.approx(target, abs=0.05)


# --------------------------------------------------------------------------- #
# compute_thermal wrapper — three-core reduction gate                         #
# --------------------------------------------------------------------------- #
def _concentric_cable(*, voltage_kV: float | None = 70.0) -> CableData:
    """A minimal concentric single-core cable for the reduction gate."""
    layers = [
        {
            "name": "xlpe_insulation",
            "layer_type": "cylindrical",
            "material": "xlpe_qiao2025",
            "r_inner_m": 0.010,
            "r_outer_m": 0.020,
        },
        {
            "name": "hdpe_outer_sheath",
            "layer_type": "cylindrical",
            "material": "hdpe",
            "r_inner_m": 0.020,
            "r_outer_m": 0.024,
        },
    ]
    from cablexsec.cables.schema import _LAYER_ADAPTER  # noqa: PLC0415

    return CableData(
        cable=CableMeta(name="concentric test cable", voltage_kV=voltage_kV),
        layers=tuple(_LAYER_ADAPTER.validate_python(x) for x in layers),
        environment=EnvironmentProperties(ambient_C=15.0, T4_external_Km_W=0.30),
    )


def test_compute_thermal_reduces_to_primitive_single_core() -> None:
    """At n_cores=1 (no dielectric, λ1=0) the wrapper == 1-D primitive + W_c·T4."""
    cable = _concentric_cable(voltage_kV=70.0)  # < 100 kV: W_d = 0
    res = compute_thermal(cable, current_A=600.0, conductor_area_m2=300e-6, n_cores=1)

    # Rebuild the primitive on the same annuli with the surface raised by W_c·T4.
    layers = [
        ConductionLayer(
            0.010,
            0.020,
            get_property("xlpe_qiao2025", "lambda_W_mK"),
        ),
        ConductionLayer(0.020, 0.024, get_property("hdpe", "lambda_W_mK")),
    ]
    rho, alpha = 1.724e-8, 3.93e-3
    r_ac = (rho / 300e-6) * (1.0 + alpha * 70.0)
    w_c = 600.0**2 * r_ac
    r_int = sum(
        thermal_resistance(ly.r_inner_m, ly.r_outer_m, ly.lambda_W_mK) for ly in layers
    )
    _r, t_prof = steady_state_profile(layers, w_c, 15.0 + w_c * 0.30, n_per_layer=10)
    assert res.T_conductor_C == pytest.approx(t_prof[0], rel=1e-12)
    assert res.T_conductor_C == pytest.approx(15.0 + w_c * (r_int + 0.30), rel=1e-12)


def test_compute_thermal_requires_environment_boundary() -> None:
    """Without T4/ambient the wrapper refuses (no silent default)."""
    cable = CableData(
        cable=CableMeta(name="bare", voltage_kV=70.0),
        layers=_concentric_cable().layers,
    )
    with pytest.raises(ValueError, match="T4_external_Km_W and ambient_C"):
        compute_thermal(cable, current_A=600.0, conductor_area_m2=300e-6, n_cores=1)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"conductor_area_m2": 0.0}, "conductor_area_m2 must be positive"),
        ({"conductor_area_m2": -1e-6}, "conductor_area_m2 must be positive"),
        ({"current_A": math.nan}, "current_A must be finite"),
        ({"current_A": math.inf}, "current_A must be finite"),
        ({"conductor_temp_C": math.nan}, "conductor_temp_C must be finite"),
        ({"ambient_C": math.nan}, "ambient_C must be finite"),
        ({"T4_external_Km_W": -0.1}, "T4_external_Km_W must not be negative"),
        ({"y_s": -0.1}, "y_s must not be negative"),
        ({"lambda1": math.nan}, "lambda1 must be finite"),
    ],
)
def test_compute_thermal_refuses_inadmissible_inputs(kwargs, match) -> None:
    """Zero area divided by zero and NaN current gave NaN temperatures."""
    import cablexsec as cx

    cable = cx.load_cable(cx.export_three_core_yaml_path(), geometry_override=None)
    call = {"current_A": 600.0, "conductor_area_m2": 300e-6} | kwargs
    with pytest.raises(ValueError, match=match):
        compute_thermal(cable, **call)


# --------------------------------------------------------------------------- #
# compute_thermal profile consistent with the IEC ladder (audit A5)           #
# --------------------------------------------------------------------------- #
_PROFILE_CASES: dict[str, tuple[str, int | None]] = {
    "export_three_core": ("export_three_core_yaml_path", None),
    "inter_array_three_core": ("inter_array_three_core_yaml_path", None),
    "cigre_tb880_single_path": ("cigre_tb880_yaml_path", 1),
    "inter_array_single_core": ("inter_array_single_core_yaml_path", None),
}


def _load_shipped(path_fn: str) -> CableData:
    import cablexsec as cx  # noqa: PLC0415

    return cx.load_cable(getattr(cx, path_fn)(), geometry_override=None)


def _conductor_loss(current_A: float, area_m2: float) -> float:
    return current_A**2 * ac_resistance(
        get_property("copper_stranded", "rho_ohm_m"),
        area_m2,
        90.0,
        get_property("copper_stranded", "alpha_R_per_K"),
    )


@pytest.mark.parametrize("case", list(_PROFILE_CASES))
@pytest.mark.parametrize("lambda1", [0.0, 0.3])
def test_profile_ends_match_iec_ladder(case: str, lambda1: float) -> None:
    """T(r_conductor) = θ_c and T(r_surface) = θ_a + W_tot·T4 to round-off.

    The heat flow through each annulus carries the losses generated inside it
    ([IEC60287-1-1] 1.4.1.1 Eq. 1), so integrating the per-annulus drops from
    the surface inward reproduces the ladder conductor temperature exactly.
    """
    path_fn, n_cores = _PROFILE_CASES[case]
    cable = _load_shipped(path_fn)
    conductor = cable.get_layer("cu_conductor")
    area = (getattr(conductor, "conductor_area_mm2", None) or 1200.0) * 1e-6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the >90 degC guard may fire
        res = compute_thermal(cable, 908.45, area, lambda1=lambda1, n_cores=n_cores)
    env = cable.environment
    assert env.ambient_C is not None and env.T4_external_Km_W is not None
    surface = env.ambient_C + res.P_total_W_m * env.T4_external_Km_W
    assert res.T_C[0] == pytest.approx(res.T_conductor_C, rel=1e-12)
    assert res.T_C[-1] == pytest.approx(surface, rel=1e-12)
    assert np.all(np.diff(res.r_m) > 0.0)  # ordered path (section-state input)
    assert np.all(np.diff(res.T_C) <= 1e-9)  # monotone: heat flows outward
    assert res.provenance is not None
    assert any("lambda2" in lim for lim in res.provenance.limitations)


def _sheathed_single_core() -> CableData:
    """220 kV single core: insulation, water barrier, Cu shell sheath, oversheath."""
    from cablexsec.cables.schema import _LAYER_ADAPTER  # noqa: PLC0415

    layers = [
        {
            "name": "cu_conductor",
            "layer_type": "cylindrical_offset",
            "material": "copper_stranded",
            "n_cores": 1,
            "r_offset_m": 0.0,
            "r_inner_m": 0.0,
            "r_outer_m": 0.020,
        },
        {
            "name": "xlpe_insulation",
            "layer_type": "cylindrical_offset",
            "material": "xlpe_qiao2025",
            "n_cores": 1,
            "r_offset_m": 0.0,
            "r_inner_m": 0.020,
            "r_outer_m": 0.040,
            "electrical": {
                "permittivity_relative": 2.5,
                "loss_factor_tan_delta": 0.001,
            },
        },
        {
            "name": "water_barrier",
            "layer_type": "cylindrical",
            "material": "hdpe",
            "r_inner_m": 0.040,
            "r_outer_m": 0.044,
        },
        {
            "name": "cu_metallic_sheath",
            "layer_type": "cylindrical_shell",
            "material": "copper_tape_screen",
            "r_mid_m": 0.045,
            "t_m": 0.002,
        },
        {
            "name": "hdpe_outer_sheath",
            "layer_type": "cylindrical",
            "material": "hdpe",
            "r_inner_m": 0.046,
            "r_outer_m": 0.052,
        },
    ]
    return CableData(
        cable=CableMeta(name="sheathed single core", voltage_kV=220.0),
        layers=tuple(_LAYER_ADAPTER.validate_python(x) for x in layers),
        environment=EnvironmentProperties(ambient_C=15.0, T4_external_Km_W=0.30),
    )


def test_compute_thermal_heat_flow_steps_at_sheath_hand_calculation() -> None:
    """IEC Eq. 1 by hand: W_c+½W_d inside the sheath, W_c(1+λ1)+W_d outside."""
    cable = _sheathed_single_core()
    lam1, current, area = 0.25, 1000.0, 1000e-6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = compute_thermal(cable, current, area, n_cores=1, lambda1=lam1)

    lam_x = get_property("xlpe_qiao2025", "lambda_W_mK")
    lam_h = get_property("hdpe", "lambda_W_mK")
    w_c = _conductor_loss(current, area)
    w_d = dielectric_loss(220.0, 2.5, 0.001, 0.020, 0.040)
    assert w_d > 0.0
    # Conductor -> sheath (T1): insulation + water barrier inside the sheath.
    t1 = thermal_resistance(0.020, 0.040, lam_x) + thermal_resistance(
        0.040, 0.044, lam_h
    )
    t3 = thermal_resistance(0.046, 0.052, lam_h)  # serving outside the sheath
    q_out = w_c * (1.0 + lam1) + w_d
    surface = 15.0 + q_out * 0.30
    expected = surface + q_out * t3 + (w_c + 0.5 * w_d) * t1
    assert res.P_total_W_m == pytest.approx(q_out, rel=1e-14)
    assert res.T_conductor_C == pytest.approx(expected, rel=1e-12)
    # Sheath temperature: surface plus the rise across the oversheath only.
    t_sheath = float(np.interp(0.046, res.r_m, res.T_C))
    assert t_sheath == pytest.approx(surface + q_out * t3, rel=1e-12)
    assert res.T_C[-1] == pytest.approx(surface, rel=1e-12)
    assert res.provenance is not None
    assert not any("no metallic sheath" in x for x in res.provenance.limitations)


def test_three_core_filler_is_included_and_reported() -> None:
    """The r_inner=0 cable-level filler enters T2 as the equal-filler-area annulus."""
    cable = _load_shipped("export_three_core_yaml_path")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = compute_thermal(cable, 908.45, 1000e-6)
    assert res.provenance is not None
    notes = [lim for lim in res.provenance.limitations if "pp_filler" in lim]
    assert len(notes) == 1
    assert "equal-filler-area" in notes[0]
    assert "geometric factor NOT applied" in notes[0]

    # Hand ladder: per-core XLPE (T1) and lead sheath (one core's heat), then
    # filler (sqrt(3) r_core .. 107.5 mm), bedding, oversheath (three cores).
    w_c = _conductor_loss(908.45, 1000e-6)
    w_d = dielectric_loss(220.0, 2.5, 0.001, 0.0210, 0.0420)
    q_t1, q_out, n = w_c + 0.5 * w_d, w_c + w_d, 3
    r_eq = math.sqrt(3.0) * 0.0455
    lam = {
        m: get_property(m, "lambda_W_mK")
        for m in ("xlpe_qiao2025", "lead_sheath", "pp_filler", "hdpe")
    }
    rise = (
        q_t1 * thermal_resistance(0.0210, 0.0420, lam["xlpe_qiao2025"])
        + q_out * thermal_resistance(0.0430, 0.0455, lam["lead_sheath"])
        + n * q_out * thermal_resistance(r_eq, 0.1075, lam["pp_filler"])
        + n * q_out * thermal_resistance(0.1075, 0.1105, lam["hdpe"])
        + n * q_out * thermal_resistance(0.1270, 0.1330, lam["hdpe"])
        + n * q_out * 0.35
    )
    assert res.T_conductor_C == pytest.approx(15.0 + rise, rel=1e-12)
    # The filler is part of the profile (no silent drop).
    assert np.any(np.isclose(res.r_m, r_eq, rtol=0.0, atol=1e-12))


def test_three_core_filler_that_cannot_enclose_cores_raises() -> None:
    """A filler smaller than the equal-area core bundle is a geometry error."""
    from cablexsec.cables.schema import _LAYER_ADAPTER  # noqa: PLC0415

    cable = _load_shipped("export_three_core_yaml_path")
    layers = []
    for layer in cable.layers:
        if layer.name == "pp_filler":
            data = layer.model_dump()
            data["r_outer_m"] = 0.070  # < sqrt(3) * 45.5 mm = 78.8 mm
            layer = _LAYER_ADAPTER.validate_python(data)
        layers.append(layer)
    bad = cable.model_copy(update={"layers": tuple(layers)})
    with pytest.raises(ValueError, match="cannot enclose 3 cores"):
        compute_thermal(bad, 500.0, 1000e-6)
