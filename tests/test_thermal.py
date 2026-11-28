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
    conductor_resistance_20C,
    dielectric_loss,
    iec_conductor_temperature,
    lay_length_factor,
    proximity_effect_factor,
    sheath_circulating_loss_factor,
    sheath_eddy_loss_factor,
    sheath_reactance,
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


def test_dielectric_loss_threshold_follows_iec_5_2() -> None:
    """[IEC60287-1-1] 5.2: counted from U0 = 127 kV for XLPE, 63.5 kV for EPR."""
    assert dielectric_loss(66.0, 2.5, 0.001, 0.015, 0.025) == 0.0  # U0 38 kV
    assert dielectric_loss(132.0, 2.5, 0.001, 0.015, 0.025) == 0.0  # U0 76 kV
    w_d = dielectric_loss(220.0, 2.5, 0.001, 0.025, 0.045)  # U0 127.0 kV
    assert w_d > 0.0
    epr = dielectric_loss(132.0, 3.0, 0.005, 0.015, 0.025, insulation_type="epr")
    assert epr > 0.0
    # Always (CIGRE TB 880 Guidance Point 7) and never.
    on = dielectric_loss(66.0, 2.5, 0.001, 0.015, 0.025, include=True)
    u0 = 66.0e3 / math.sqrt(3.0)
    cap = 2.0 * math.pi * 8.854e-12 * 2.5 / math.log(0.025 / 0.015)
    assert on == pytest.approx(2.0 * math.pi * 50.0 * cap * u0**2 * 0.001, rel=1e-14)
    assert dielectric_loss(220.0, 2.5, 0.001, 0.025, 0.045, include=False) == 0.0
    # The capacitance of a laid-up core is C_LL times that of a straight one.
    laid = dielectric_loss(
        66.0, 2.5, 0.001, 0.015, 0.025, include=True, lay_length_factor=1.02
    )
    assert laid == pytest.approx(1.02 * on, rel=1e-14)
    with pytest.raises(ValueError, match="insulation_type"):
        dielectric_loss(220.0, 2.5, 0.001, 0.025, 0.045, insulation_type="oil")


def test_skin_effect_factor_has_the_three_iec_ranges() -> None:
    """[IEC60287-1-1] 5.1.3: the formula changes at x_s = 2.8 and 3.8."""
    f = 50.0

    def r_for(xs: float) -> float:  # R' giving x_s at k_s = 1
        return 8.0 * math.pi * f * 1.0e-7 / xs**2

    for xs, expected in (
        (2.0, 2.0**4 / (192.0 + 0.8 * 2.0**4)),
        (3.0, -0.136 - 0.0177 * 3.0 + 0.0563 * 9.0),
        (4.5, 0.354 * 4.5 - 0.733),
    ):
        assert skin_effect_factor(r_for(xs), f) == pytest.approx(expected, rel=1e-12)
    # The ranges join within the rounding of the published coefficients.
    lo, hi = skin_effect_factor(r_for(2.8), f), skin_effect_factor(r_for(2.8001), f)
    assert hi == pytest.approx(lo, abs=3e-3)
    lo, hi = skin_effect_factor(r_for(3.8), f), skin_effect_factor(r_for(3.8001), f)
    assert hi == pytest.approx(lo, abs=3e-3)


def test_iec60228_resistance_and_lay_up_factor() -> None:
    """R0 from [IEC60228] Table 4; C_LL of [IEC60287-1-1] Annex A."""
    import cablexsec as cx  # noqa: PLC0415

    t66 = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
    c_ll, note = lay_length_factor(t66)
    # Core diameter 70.40 mm, pitch 1.4542 m, C_fL 1.29 (three cores).
    assert c_ll == pytest.approx(
        math.sqrt(1.0 + (math.pi * 1.29 * 0.0704 / 1.4542) ** 2), rel=1e-12
    )
    assert c_ll == pytest.approx(1.0191, abs=1e-4)
    assert "Annex A" in note
    r20, basis = conductor_resistance_20C(t66, 630e-6, "copper_janocha2024")
    assert r20 == pytest.approx(0.0283e-3, rel=1e-12)  # per km of the cable
    assert "Table 4" in basis
    # Between two nominal sizes: linear interpolation (CIGRE TB 880, 2.5).
    r_mid, basis = conductor_resistance_20C(t66, 700e-6, "copper_janocha2024")
    expected = 0.0283 + (0.0221 - 0.0283) * (700.0 - 630.0) / (800.0 - 630.0)
    assert r_mid == pytest.approx(expected * 1e-3, rel=1e-12)
    assert "interpolated" in basis
    # Not a standard material: rho/A times C_LL.
    r_other, basis = conductor_resistance_20C(
        t66, 630e-6, "lead_sheath", lay_factor=c_ll
    )
    assert r_other == pytest.approx(21.4e-8 / 630e-6 * c_ll, rel=1e-12)
    assert "rho/A" in basis
    # A single-core cable has no lay-up.
    single = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    assert lay_length_factor(single) == (1.0, "")


def test_three_core_losses_are_computed_from_the_cable() -> None:
    """y_s, y_p, lambda1 and R0 of the 66 kV template (IEC 60287-1-1 5.1, 5.3.11)."""
    import cablexsec as cx  # noqa: PLC0415

    t66 = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = compute_thermal(t66, 1093.0, 630e-6)
    assert res.losses is not None
    r_dc = 0.0283e-3 * (1.0 + 3.93e-3 * 70.0)
    ys = skin_effect_factor(r_dc, 50.0)
    s = math.sqrt(3.0) * 0.04081  # distance between conductor axes
    yp = proximity_effect_factor(r_dc, 50.0, 0.02990, s)
    assert res.losses["y_s"] == pytest.approx(ys, rel=1e-12)
    assert res.losses["y_p"] == pytest.approx(yp, rel=1e-12)
    r_ac = r_dc * (1.0 + ys + yp)
    assert res.losses["R_ac_ohm_m"] == pytest.approx(r_ac, rel=1e-12)
    # [IEATask49] Table 13 gives 0.04 ohm/km for the 630 mm2 cable.
    assert round(r_ac * 1e3, 2) == 0.04
    assert res.losses["W_d_W_m"] == 0.0  # U0 = 38 kV < 127 kV
    lam1 = res.losses["lambda1"]
    assert 0.05 < lam1 < 0.15
    notes = " ".join(res.provenance.limitations)
    assert "5.3.11" in notes and "factor 1.5" in notes
    # A given value replaces the computed one.
    given = compute_thermal(t66, 1093.0, 630e-6, y_s=0.0, y_p=0.0, lambda1=0.0)
    assert given.losses["y_s"] == given.losses["y_p"] == given.losses["lambda1"] == 0.0
    # Non-magnetic armour: the trefoil formula (5.3.2), no factor 1.5.
    import yaml  # noqa: PLC0415

    data = yaml.safe_load(cx.iea_task49_66kv_yaml_path().read_text(encoding="utf-8"))
    data["materials"] = {
        "stainless_armour": {
            "E_Pa": {"value": 2.0e11, "source": "test"},
            "nu": {"value": 0.3, "source": "test"},
            "relative_permeability": {"value": 1.0, "source": "test"},
        }
    }
    for layer in data["layers"]:
        if layer["layer_type"] == "helical":
            layer["material"] = "stainless_armour"
    stainless = cx.CableData.model_validate(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_ss = compute_thermal(stainless, 1093.0, 630e-6)
    assert "5.3.2" in " ".join(res_ss.provenance.limitations)
    assert res_ss.losses["lambda1"] < lam1


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
    """IEC 60287-1-1 (2023) loss chain of the Poradowski-Anders (2025) NOTE-1 cable.

    220 kV three-core cable, stranded Cu 1200 mm2, corrugated copper sheath
    (Table 1 and footnote d), non-magnetic stainless armour (footnote c).
    ``R0`` from [IEC60228] Table 4; ``lambda1'`` of the sheath from 5.3.2,
    which CIGRE TB 880 Guidance Point 38 assigns to an SL cable under
    non-magnetic armour (5.3.11, factor 1.5, is for magnetic armour);
    ``lambda1''`` from 5.3.7.1 with the solid-bonding factor ``F`` of
    5.3.6, as TB 880 Guidance Points 6 and 31 keep it.
    """
    f = 50.0
    r20, alpha = 0.0151e-3, 3.93e-3  # [IEC60228] Table 4, 1200 mm2 Cu
    r_dc = r20 * (1.0 + alpha * (90.0 - 20.0))
    ys = skin_effect_factor(r_dc, f)
    yp = proximity_effect_factor(r_dc, f, 0.042, 0.13175)  # d_c, touching cores
    r_ac = r_dc * (1.0 + ys + yp)
    w_c = 908.45**2 * r_ac
    # Dielectric: U0 = 127.0 kV >= 127 kV, XLPE eps_r 2.5, tan 0.001 (Table 3).
    w_d = dielectric_loss(220.0, 2.5, 0.001, 0.0227, 0.0466, frequency_Hz=f)
    # Sheath: Cu ring 111.95/113.95 mm, Fcor = 1.3406 (footnote d), ~80 degC.
    a_s = math.pi / 4 * (0.11395**2 - 0.11195**2)
    rho_s = 1.7241e-8 * (1.0 + alpha * (80.0 - 20.0))
    r_s = rho_s * 1.3406 / a_s
    d_s = 0.5 * (0.11195 + 0.11395)
    x = sheath_reactance(0.13175, d_s, frequency_Hz=f)
    lam1_c = sheath_circulating_loss_factor(r_s, r_ac, x)
    lam1_e = sheath_eddy_loss_factor(
        r_s,
        r_ac,
        sheath_mean_diameter_m=d_s,
        sheath_outer_diameter_m=0.11395,
        sheath_thickness_m=0.001,
        sheath_resistivity_ohm_m=rho_s,
        spacing_m=0.13175,
        frequency_Hz=f,
        solid_bonding_reactance_ohm_m=x,
    )
    return {"W_c": w_c, "W_d": w_d, "lambda1": lam1_c, "lambda1_eddy": lam1_e}


@pytest.mark.full_scale
def test_BM5_poradowski_buried_reference_case() -> None:
    """BM5: Poradowski-Anders (2025) Table 1 NOTE-1 buried 220 kV cable.

    Stated network: T1=0.6579, T2=0.1121, T3=0.0357, T4=0.3211 K.m/W (T4 the
    external soil/Robin resistance, taken verbatim); theta_amb = 15 degC,
    n = 3. Reference: I = 908.45 A gives 89.62 degC (FEM) and 90 degC (the
    source's IEC analytical ampacity).

    With ``R0`` from IEC 60228 and the circulating sheath loss of IEC 5.3.2
    (``lambda1'' = 0``, IEC) the ladder gives 85.8 degC; the earlier 84.4 degC
    took ``R0 = rho/A``, 5 % low. Adding the eddy loss that CIGRE TB 880
    Guidance Points 6 and 31 keep (lambda1'' = 0.19) gives 90.9 degC. The
    1.5 factor of IEC 5.3.11 belongs to magnetic armour (TB 880 Guidance
    Point 38) and would give 100 degC.
    """
    t1, t2, t3, t4 = 0.6579, 0.1121, 0.0357, 0.3211
    t_amb, n = 15.0, 3
    losses = _bm5_loss_chain()
    w_c, w_d, lam1 = losses["W_c"], losses["W_d"], losses["lambda1"]
    lam1_e = losses["lambda1_eddy"]

    def theta(lambda1: float) -> float:
        return iec_conductor_temperature(
            t1,
            t2,
            t3,
            t4,
            W_c=w_c,
            W_d=w_d,
            lambda1=lambda1,
            n_conductors=n,
            T_ambient_C=t_amb,
        )

    assert lam1 == pytest.approx(1.027, abs=1e-3)
    assert lam1_e == pytest.approx(0.186, abs=1e-3)
    # IEC (lambda1'' = 0): below both references.
    assert theta(lam1) == pytest.approx(85.75, abs=0.01)
    # With the TB 880 eddy loss: within 1.3 K of the FEM and 0.9 K of the IEC
    # reference temperature.
    assert theta(lam1 + lam1_e) == pytest.approx(90.89, abs=0.01)
    # The magnetic-armour factor of 5.3.11 does not apply to stainless armour.
    assert theta(1.5 * lam1) > 99.0


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
    # The cable has no conductor layer: the conductor material is given.
    # The losses of the primitive: rho/A, no skin or proximity effect.
    res = compute_thermal(
        cable,
        current_A=600.0,
        conductor_area_m2=300e-6,
        n_cores=1,
        conductor_material="copper_stranded",
        y_s=0.0,
        y_p=0.0,
        conductor_resistance_20C_ohm_m=1.7241e-8 / 300e-6,
    )

    # Rebuild the primitive on the same annuli with the surface raised by W_c·T4.
    layers = [
        ConductionLayer(
            0.010,
            0.020,
            get_property("xlpe_qiao2025", "lambda_W_mK"),
        ),
        ConductionLayer(0.020, 0.024, get_property("hdpe", "lambda_W_mK")),
    ]
    rho, alpha = 1.7241e-8, 3.93e-3
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
    "poradowski2025_single_path": ("poradowski2025_220kv_yaml_path", 1),
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
        res = compute_thermal(
            cable,
            current,
            area,
            n_cores=1,
            lambda1=lam1,
            y_s=0.0,
            y_p=0.0,
            conductor_resistance_20C_ohm_m=1.7241e-8 / area,
        )

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
    """Hand ladder of the SL-type export template ([IEC60287-2-1] 4.1, 5.6).

    Per-core resistances divided by C_LL; the filler and bedding under the
    armour carry T2 = rho/(6 pi) G-bar, lower curve (cores not touching).
    """
    cable = _load_shipped("export_three_core_yaml_path")
    rho_cu = 1.7241e-8
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = compute_thermal(
            cable,
            908.45,
            1000e-6,
            y_s=0.0,
            y_p=0.0,
            lambda1=0.0,
            conductor_resistance_20C_ohm_m=rho_cu / 1000e-6,
        )
    assert res.provenance is not None
    notes = [lim for lim in res.provenance.limitations if "pp_filler" in lim]
    assert len(notes) == 2
    assert "equal-filler-area" in notes[0]
    assert "G-bar" in notes[1] and "equal thicknesses" in notes[1]

    c_ll = math.sqrt(1.0 + (math.pi * 1.29 * 0.0910 / 2.138) ** 2)
    w_c = 908.45**2 * rho_cu / 1000e-6 * (1.0 + 3.93e-3 * 70.0)
    w_d = dielectric_loss(220.0, 2.5, 0.001, 0.0210, 0.0420, lay_length_factor=c_ll)
    q_t1, q_out, n = w_c + 0.5 * w_d, w_c + w_d, 3
    lam = {
        m: get_property(m, "lambda_W_mK")
        for m in ("xlpe_qiao2025", "lead_sheath", "pp_filler", "hdpe")
    }
    # Filler and bedding: x = (r_armour_in - r_offset - r_core) / D_core.
    inner = cable.get_layer("inner_armour")
    r_arm = float(inner.r_helix_m) - 0.5 * float(inner.d_wire_m)
    x = (r_arm - 0.060 - 0.0455) / 0.091
    poly = 0.0126529 + 1.101 * x - 4.56104 * x**2 + 11.5093 * x**3
    t2 = (
        max(1.0 / lam["pp_filler"], 1.0 / lam["hdpe"])
        / (6.0 * math.pi)
        * (2.0 * math.pi * poly)
    )
    rise = (
        q_t1 * thermal_resistance(0.0190, 0.0430, lam["xlpe_qiao2025"]) / c_ll
        + q_out * thermal_resistance(0.0430, 0.0455, lam["lead_sheath"]) / c_ll
        + n * q_out * t2
        + n * q_out * thermal_resistance(0.1270, 0.1330, lam["hdpe"])
        + n * q_out * 0.35
    )
    assert res.T_conductor_C == pytest.approx(15.0 + rise, rel=1e-12)
    # The filler is part of the profile (no silent drop).
    r_eq = math.sqrt(3.0) * 0.0455
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
