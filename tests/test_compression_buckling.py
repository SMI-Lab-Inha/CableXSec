# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Armour compression screens: lateral and radial (bird-cage) buckling.

Published targets are recorded in ``scripts/check_compression_buckling.py``;
these tests gate them and check limits, the friction trend, units and the
cable-level assembly.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from check_compression_buckling import (
    all_checks,
    li_vaz_lower_bound,
    malta_sheath_only,
    sousa_case,
    sousa_table5,
)

import cablexsec as cx
from cablexsec.analysis.capacity import compute_capacity, wire_yield_curvature
from cablexsec.analysis.compression import (
    LAY_ANGLE_RANGE,
    LI_VAZ_KNOCKDOWN_TABLE,
    TapeLayer,
    combined_radial_load,
    compression_screen,
    compute_compression_capacity,
    cyclic_curvature_knockdown,
    first_slip_curvature,
    lateral_buckling_load,
    lateral_buckling_load_round,
    radial_buckling_load_curved,
    radial_buckling_load_straight,
    radial_capacity_failure_load,
    rectangular_wire_section,
    round_wire_section,
    sheath_foundation_stiffness,
    tape_foundation_stiffness,
    true_wall_tension,
    wire_yield_load,
)
from cablexsec.cables.schema import CableData, CableMeta
from cablexsec.exceptions import CableInputError
from cablexsec.materials import library

pytestmark = pytest.mark.analytical_sanity

ALPHA = math.radians(25.0)
STEEL = round_wire_section(210e9, 0.3, 5e-3)


def _two_layer_cable(
    alpha_deg: float = 25.0, with_sheath: bool = True, water: str | None = None
) -> CableData:
    """Two equal round-wire armour layers inside an HDPE sheath."""
    from cablexsec.cables.schema import (  # noqa: PLC0415
        _LAYER_ADAPTER,
        WaterIngressProperties,
    )

    core = {
        "name": "core",
        "layer_type": "cylindrical",
        "material": "hdpe",
        "r_inner_m": 0.0,
        "r_outer_m": 0.0775,
    }
    inner = {
        "name": "inner_armour",
        "layer_type": "helical",
        "material": "galvanised_steel_armour",
        "n_wires": 48,
        "d_wire_m": 0.005,
        "r_helix_m": 0.080,
        "lay_angle_deg": alpha_deg,
        "lay_direction": "left_hand",
    }
    outer = dict(
        inner, name="outer_armour", r_helix_m=0.085, lay_direction="right_hand"
    )
    sheath = {
        "name": "outer_sheath",
        "layer_type": "cylindrical",
        "material": "hdpe",
        "r_inner_m": 0.0875,
        "r_outer_m": 0.0935,
    }
    layers = [core, inner, outer] + ([sheath] if with_sheath else [])
    return CableData(
        cable=CableMeta(name="two-layer synthetic"),
        layers=tuple(_LAYER_ADAPTER.validate_python(x) for x in layers),
        water_ingress=(
            None
            if water is None
            else WaterIngressProperties(design="wet", armour_annulus=water)  # type: ignore[arg-type]
        ),
    )


# --------------------------------------------------------------------------- #
# Published targets                                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("check", all_checks(), ids=lambda c: c.ident)
def test_published_targets(check) -> None:
    assert check.passed, f"{check.ident}: {check.computed} vs {check.published}"


def test_gate_identifiers_cover_every_source() -> None:
    idents = {c.ident.split("-")[0] for c in all_checks()}
    assert {"ST17", "OST12", "LV21", "LV21b", "YANG15", "MALTA22"} <= idents


def test_saevik_bound_is_below_the_li_vaz_fe_limits() -> None:
    lb = li_vaz_lower_bound()
    assert lb["P1_over_Pcr"] < lb["min_FE_ratio"]  # 0.613 < 0.83


def test_sousa_straight_beam_is_below_curved_beam() -> None:
    """The source ranks the straight beam conservative (Table 5)."""
    assert sousa_case("straight", True).P2_star_N < sousa_case("curved", True).P2_star_N
    for _, _, p2_here, _, star_here in sousa_table5():
        assert p2_here > 0.0 and star_here > 0.0


def test_malta_sheath_only_screen_is_conservative() -> None:
    """Sheath-only P2* lies below the FE limit load with the tape cut (241.69 kN)."""
    m = malta_sheath_only()
    assert m.P2_star_N < 241.69e3
    assert m.P3_N < m.P2_N  # hoop capacity governs without a tape


# --------------------------------------------------------------------------- #
# Limits                                                                      #
# --------------------------------------------------------------------------- #
def test_zero_lay_angle_limits() -> None:
    assert math.isinf(lateral_buckling_load(50, 0.08, 0.0, STEEL))
    assert math.isinf(lateral_buckling_load_round(50, 0.08, 0.0, STEEL))
    assert math.isinf(first_slip_curvature(0.2, 100.0, 0.0, STEEL, 0.08))


def test_lateral_capacity_vanishes_with_lay_angle() -> None:
    small = [lateral_buckling_load(50, 0.08, math.radians(a), STEEL) for a in (1, 2, 4)]
    assert small[0] < small[1] < small[2]
    ratio = small[1] / small[0]
    assert ratio == pytest.approx(4.0, rel=0.01)  # P1 ~ sin^2(alpha)


def test_round_section_torsion_and_rectangle_square_limit() -> None:
    assert STEEL.GJ_Nm2 == pytest.approx(STEEL.EI2_Nm2 / 1.3, rel=1e-12)
    sq = rectangular_wire_section(210e9, 0.3, 0.01, 0.01)
    # exact square torsion constant 0.1406 b^4 (Roark form 0.1408)
    assert sq.GJ_Nm2 / (sq.G_Pa * 0.01**4) == pytest.approx(0.1406, rel=2e-3)
    assert sq.EI2_Nm2 == pytest.approx(sq.EI3_Nm2, rel=1e-12)
    with pytest.raises(CableInputError):
        lateral_buckling_load_round(
            10, 0.08, ALPHA, rectangular_wire_section(2e11, 0.3, 2e-3, 7e-3)
        )


def test_curved_beam_tends_to_straight_beam_for_large_radius() -> None:
    c = 2.0e6
    p_sb = radial_buckling_load_straight(50, ALPHA, STEEL, c)
    p_cb, half_wave = radial_buckling_load_curved(50, 1.0e6, ALPHA, STEEL, c)
    assert p_cb == pytest.approx(p_sb, rel=1e-9)
    assert half_wave == pytest.approx(math.pi * (STEEL.EI2_Nm2 / c) ** 0.25, rel=1e-9)


def test_no_foundation_and_no_hoop_strength_give_infinite_capacity() -> None:
    assert math.isinf(radial_buckling_load_straight(50, ALPHA, STEEL, 0.0))
    assert radial_buckling_load_curved(50, 0.08, ALPHA, STEEL, 0.0) == (
        math.inf,
        math.inf,
    )
    assert math.isinf(radial_capacity_failure_load(0.0, 1.0))
    assert math.isinf(combined_radial_load(math.inf, math.inf, math.inf))
    assert combined_radial_load(2.0, math.inf, 2.0) == pytest.approx(1.0)
    with pytest.raises(CableInputError):
        combined_radial_load(0.0, 1.0, 1.0)


def test_single_layer_hoop_failure_is_eq33() -> None:
    """2πHR/tan²α equals R/tan²α (n_t σ A sin²α_t/(cos α_t R_t) + 2π σ_s t_s)."""
    tape = TapeLayer("t", 15e9, 9e-5, math.radians(84.0), 2, 0.069, 313e6, "test")
    r, a = 0.0675, math.radians(35.0)
    hoop = tape.hoop_strength_N_m + 20e6 * 5e-3
    eq33 = (
        r
        / math.tan(a) ** 2
        * (
            2
            * 313e6
            * 9e-5
            * math.sin(tape.lay_angle_rad) ** 2
            / (math.cos(tape.lay_angle_rad) * 0.069)
            + 2 * math.pi * 20e6 * 5e-3
        )
    )
    assert radial_capacity_failure_load(hoop, math.tan(a) ** 2 / r) == pytest.approx(
        eq33, rel=1e-12
    )


def test_knockdown_table_hold_and_flag() -> None:
    r = 0.03575
    assert cyclic_curvature_knockdown(0.0, r) == (0.91, False)
    eta, beyond = cyclic_curvature_knockdown(0.15, r)
    assert eta == pytest.approx(0.895) and not beyond
    assert cyclic_curvature_knockdown(1.0, r) == (LI_VAZ_KNOCKDOWN_TABLE[-1][1], True)
    with pytest.raises(CableInputError):
        cyclic_curvature_knockdown(-1.0, r)


def test_true_wall_and_yield_load() -> None:
    assert true_wall_tension(1.0e3, 2.0e6, 0.01) == pytest.approx(-19.0e3)
    assert true_wall_tension(0.0, 1.0e6, 0.01, 2.0e6, 0.005) == pytest.approx(0.0)
    assert wire_yield_load(10, 0.0, 1.0e8, 1e-5) == pytest.approx(1.0e4)


# --------------------------------------------------------------------------- #
# Friction trend and units                                                    #
# --------------------------------------------------------------------------- #
def test_first_slip_curvature_is_linear_in_friction_and_pressure() -> None:
    k1 = first_slip_curvature(0.1, 1000.0, ALPHA, STEEL, 0.08)
    k2 = first_slip_curvature(0.2, 1000.0, ALPHA, STEEL, 0.08)
    assert k2 == pytest.approx(2.0 * k1, rel=1e-12)
    assert first_slip_curvature(0.0, 1000.0, ALPHA, STEEL, 0.08) == 0.0
    k_ext = first_slip_curvature(
        0.1, 1000.0, ALPHA, STEEL, 0.08, external_line_load_N_m=500.0
    )
    assert k_ext > k1
    with pytest.raises(CableInputError):
        first_slip_curvature(-0.1, 1000.0, ALPHA, STEEL, 0.08)


def test_lateral_capacity_is_friction_independent_once_slipping() -> None:
    """[LiVaz2021a] Table 4: P*/Pcr = 0.83-0.85 for mu = 0.05-0.15."""
    cable = _two_layer_cable()
    caps = [
        compression_screen(cable, -1.0e4, 0.1, mu=mu).compression_capacity_N
        for mu in (0.05, 0.1, 0.15)
    ]
    assert (
        caps[0]
        == pytest.approx(caps[1], rel=1e-12)
        == pytest.approx(caps[2], rel=1e-12)
    )


def test_friction_gate_closes_the_lateral_mode_below_first_slip() -> None:
    cable = _two_layer_cable()
    base = compression_screen(cable, -1.0e4, 0.1, mu=0.1)
    k_slip = max(lat.first_slip_curvature_1_m for lat in base.lateral)
    low = compression_screen(cable, -1.0e4, 0.5 * k_slip, mu=0.1)
    assert all(not lat.active for lat in low.lateral)
    assert low.governing == "radial"
    assert low.compression_capacity_N > base.compression_capacity_N
    # higher friction raises the first-slip curvature proportionally
    hi = compression_screen(cable, -1.0e4, 0.1, mu=0.2)
    for a, b in zip(base.lateral, hi.lateral, strict=True):
        assert b.first_slip_curvature_1_m == pytest.approx(
            2 * a.first_slip_curvature_1_m
        )
    # a sealed annulus adds hydrostatic clamping to the gate
    sealed = compression_screen(cable, 0.0, 0.1, mu=0.1, annulus="sealed", p_ext_Pa=2e6)
    for a, b in zip(base.lateral, sealed.lateral, strict=True):
        assert b.first_slip_curvature_1_m > a.first_slip_curvature_1_m


def test_closed_forms_are_unit_consistent() -> None:
    """N, mm, MPa inputs give the same forces as N, m, Pa inputs."""
    sec_si = round_wire_section(210e9, 0.3, 5e-3)
    sec_mm = round_wire_section(210e3, 0.3, 5.0)
    assert lateral_buckling_load(50, 80.0, ALPHA, sec_mm) == pytest.approx(
        lateral_buckling_load(50, 0.08, ALPHA, sec_si), rel=1e-12
    )
    assert lateral_buckling_load_round(50, 80.0, ALPHA, sec_mm) == pytest.approx(
        lateral_buckling_load_round(50, 0.08, ALPHA, sec_si), rel=1e-12
    )
    c_si = sheath_foundation_stiffness(96, 0.0825, ALPHA, 1e9, 6e-3)
    c_mm = sheath_foundation_stiffness(96, 82.5, ALPHA, 1e3, 6.0)
    assert c_mm == pytest.approx(c_si * 1e-6, rel=1e-12)  # Pa -> MPa
    assert radial_buckling_load_straight(96, ALPHA, sec_mm, c_mm) == pytest.approx(
        radial_buckling_load_straight(96, ALPHA, sec_si, c_si), rel=1e-12
    )
    k_si = first_slip_curvature(0.1, 1000.0, ALPHA, sec_si, 0.08)
    k_mm = first_slip_curvature(0.1, 1000.0, ALPHA, sec_mm, 80.0)
    assert k_mm == pytest.approx(k_si * 1e-3, rel=1e-12)  # 1/m -> 1/mm
    t_si = tape_foundation_stiffness(96, 0.0825, ALPHA, 2, 1.3e6, math.radians(84.0))
    t_mm = tape_foundation_stiffness(96, 82.5, ALPHA, 2, 1.3e6, math.radians(84.0))
    assert t_mm == pytest.approx(t_si * 1e-6, rel=1e-12)


# --------------------------------------------------------------------------- #
# Cable-level assembly                                                        #
# --------------------------------------------------------------------------- #
def test_screen_reproduces_closed_forms_for_two_equal_layers() -> None:
    cable = _two_layer_cable()
    res = compression_screen(cable, -2.0e4, 0.1)
    e = library.get_property("galvanised_steel_armour", "E_Pa")
    nu = library.get_property("galvanised_steel_armour", "nu")
    sec = round_wire_section(e, nu, 0.005)
    p1 = [lateral_buckling_load(48, r, ALPHA, sec) for r in (0.080, 0.085)]
    assert [lat.P1_saevik_N for lat in res.lateral] == pytest.approx(p1, rel=1e-12)
    r_mean = 0.0825
    c = sheath_foundation_stiffness(
        96, r_mean, ALPHA, library.get_property("hdpe", "E_Pa"), 0.006
    )
    assert res.radial.foundation_c_Pa == pytest.approx((c, c), rel=1e-12)
    assert (
        pytest.approx(radial_buckling_load_straight(96, ALPHA, sec, c), rel=1e-12)
        == res.radial.P2_N
    )
    assert (
        pytest.approx(
            wire_yield_load(
                96,
                ALPHA,
                library.get_property("galvanised_steel_armour", "sigma_y_Pa"),
                sec.area_m2,
            ),
            rel=1e-12,
        )
        == res.radial.P4_N
    )
    assert math.isinf(res.radial.P3_N)  # no sourced strength
    assert res.radial.restraint_layers == ("outer_sheath",)
    # demands: per-layer lateral demands add up to the radial demand
    assert sum(lat.demand_N for lat in res.lateral) == pytest.approx(
        res.radial.demand_N
    )
    assert res.radial.demand_N == pytest.approx(res.armour_share * 2.0e4)
    # utilisation linear in compression
    res2 = compression_screen(cable, -4.0e4, 0.1)
    assert res2.utilisation == pytest.approx(2.0 * res.utilisation, rel=1e-12)
    assert res.compression_capacity_N == pytest.approx(
        2.0e4 / res.utilisation, rel=1e-12
    )


def test_tension_gives_zero_utilisation_and_positive_margin() -> None:
    cable = _two_layer_cable()
    res = compression_screen(cable, 5.0e3, 0.1, utilisation_factor=0.67)
    assert res.utilisation == 0.0 and res.governing == "none"
    assert res.margin_N == pytest.approx(0.67 * res.compression_capacity_N + 5.0e3)
    assert res.provenance.citations


def test_design_factor_scales_allowable_and_utilisation() -> None:
    cable = _two_layer_cable()
    char = compression_screen(cable, -1.0e4, 0.1)
    des = compression_screen(
        cable, -1.0e4, 0.1, utilisation_factor=cx.ARMOUR_UTILISATION_FACTORS["normal"]
    )
    assert des.design_utilisation == pytest.approx(char.utilisation / 0.67)
    assert des.allowable_compression_N == pytest.approx(
        0.67 * char.compression_capacity_N
    )
    assert des.margin_N == pytest.approx(des.allowable_compression_N - 1.0e4)


def test_hoop_strength_tapes_and_curved_model() -> None:
    cable = _two_layer_cable()
    plain = compression_screen(cable, -1.0e4, 0.1)
    with_p3 = compression_screen(
        cable, -1.0e4, 0.1, sheath_hoop_strength_Pa=20e6, sheath_strength_source="test"
    )
    assert math.isfinite(with_p3.radial.P3_N)
    assert with_p3.radial.P2_star_N < plain.radial.P2_star_N
    mapped = compression_screen(
        cable,
        -1.0e4,
        0.1,
        sheath_hoop_strength_Pa={"outer_sheath": 20e6},
        sheath_strength_source="test",
    )
    assert pytest.approx(with_p3.radial.P3_N) == mapped.radial.P3_N
    tape = TapeLayer("aramid", 15e9, 9e-5, math.radians(84.0), 2, 0.094, 313e6, "test")
    taped = compression_screen(
        cable,
        -1.0e4,
        0.1,
        tapes=(tape,),
        sheath_hoop_strength_Pa=20e6,
        sheath_strength_source="test",
    )
    assert taped.radial.restraint_layers == ("outer_sheath", "aramid")
    assert taped.radial.P2_N > with_p3.radial.P2_N
    assert taped.radial.P3_N > with_p3.radial.P3_N
    curved = compression_screen(cable, -1.0e4, 0.1, radial_model="curved")
    assert curved.radial.radial_model == "curved"
    assert pytest.approx(plain.radial.P2_N, rel=1e-6) != curved.radial.P2_N


def test_bare_strand_has_no_radial_elastic_mode() -> None:
    cable = _two_layer_cable(with_sheath=False)
    res = compression_screen(cable, -1.0e4)
    assert math.isinf(res.radial.P2_N)
    assert math.isfinite(res.radial.P2_star_N)  # wire yield still bounds it
    assert any("no outboard hoop layer" in a for a in res.assumptions)


def test_sealed_annulus_uses_true_wall_tension_and_flags_declaration() -> None:
    cable = _two_layer_cable(water="flooded")
    res = compression_screen(cable, 1.0e3, 0.1, annulus="sealed", p_ext_Pa=2.0e6)
    assert res.driving_tension_N == pytest.approx(1.0e3 - 2.0e6 * math.pi * 0.0935**2)
    assert res.utilisation > 0.0
    assert any("differs from the cable declaration" in a for a in res.assumptions)


def test_input_validation() -> None:
    cable = _two_layer_cable()
    bad = [
        {"annulus": "wet"},
        {"p_ext_Pa": -1.0},
        {"radial_model": "beam"},
        {"utilisation_factor": 1.2},
        {"sheath_hoop_strength_Pa": 20e6},
        {"sheath_hoop_strength_Pa": 20e6, "sheath_strength_source": " "},
        {"sheath_hoop_strength_Pa": {"no_such": 20e6}, "sheath_strength_source": "x"},
        {"sheath_hoop_strength_Pa": -1.0, "sheath_strength_source": "x"},
        {"mu": -0.1},
    ]
    for kwargs in bad:
        with pytest.raises(CableInputError):
            compression_screen(cable, -1.0e4, 0.1, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(CableInputError):
        compression_screen(cable, math.nan)
    with pytest.raises(CableInputError):
        compression_screen(cable, -1.0e4, -0.1)
    with pytest.raises(CableInputError):
        TapeLayer("t", 15e9, 9e-5, math.radians(95.0), 2, 0.09, 313e6, "x")
    with pytest.raises(CableInputError):
        TapeLayer("t", 15e9, 9e-5, math.radians(84.0), 0, 0.09, 313e6, "x")
    with pytest.raises(CableInputError):
        TapeLayer("t", 15e9, 9e-5, math.radians(84.0), 2, 0.09, 313e6, " ")
    with pytest.raises(CableInputError):
        round_wire_section(2e11, 0.6, 1e-3)
    with pytest.raises(CableInputError):
        rectangular_wire_section(2e11, 0.3, -1e-3, 1e-3)
    with pytest.raises(CableInputError):
        tape_foundation_stiffness(50, 0.08, ALPHA, 2, 1e6, math.pi / 2)
    with pytest.raises(CableInputError):
        sheath_foundation_stiffness(50, 0.08, ALPHA, -1.0, 1e-3)
    from cablexsec.cables.schema import _LAYER_ADAPTER  # noqa: PLC0415

    polymer_only = CableData(
        cable=CableMeta(name="no armour"),
        layers=(
            _LAYER_ADAPTER.validate_python(
                {
                    "name": "s",
                    "layer_type": "cylindrical",
                    "material": "hdpe",
                    "r_inner_m": 0.0,
                    "r_outer_m": 0.05,
                }
            ),
        ),
    )
    with pytest.raises(CableInputError):
        compression_screen(polymer_only, -1.0e3)


def test_lay_angle_outside_validated_range_warns() -> None:
    cable = _two_layer_cable(alpha_deg=12.0)
    with pytest.warns(cx.ExtrapolationWarning, match="lay angle"):
        res = compression_screen(cable, -1.0e3, 0.1)
    assert any("outside the 20-38 deg range" in a for a in res.assumptions)
    assert LAY_ANGLE_RANGE.contains(25.0)


# --------------------------------------------------------------------------- #
# Capacity curves and the corrected bird-caging usage in compute_capacity     #
# --------------------------------------------------------------------------- #
def test_compute_capacity_birdcage_uses_layer_forces() -> None:
    """Demand and capacity are both layer axial forces (former per-wire mix-up)."""
    cable = _two_layer_cable()
    res = compute_capacity(cable, np.array([-2.0e4, 1.0e4]))
    screen = compression_screen(cable, -2.0e4)
    assert res.uf_bird_caging[0] == pytest.approx(screen.utilisation, rel=1e-12)
    assert res.uf_bird_caging[1] == 0.0
    cap = screen.compression_capacity_N
    over = compute_capacity(cable, np.array([-1.01 * cap]))
    assert over.governing == ["bird_caging"] and over.kappa_max_1_m[0] == 0.0


def test_usage_limit_scales_the_yield_curvature() -> None:
    cable = _two_layer_cable()
    k1 = wire_yield_curvature(cable, 0.0)
    k067 = wire_yield_curvature(cable, 0.0, usage_limit=0.67)
    assert 0.0 < k067 < k1
    with pytest.raises(CableInputError):
        wire_yield_curvature(cable, 0.0, usage_limit=0.0)


def test_capacity_curve_branches() -> None:
    cable = _two_layer_cable()
    cap = compression_screen(cable, -1.0).compression_capacity_N
    tensions = np.array([5.0e4, 0.0, -0.5 * cap, -1.5 * cap])
    # The armour branches; the components' limits are checked elsewhere.
    curve = compute_compression_capacity(cable, tensions, component_limits=False)
    ref = compute_capacity(cable, tensions[:2], component_limits=False)
    assert curve.kappa_max_1_m[:2] == pytest.approx(ref.kappa_max_1_m, rel=1e-12)
    assert curve.governing[:2] == ["wire_yield", "wire_yield"]
    assert math.isinf(curve.kappa_lateral_1_m[0])
    # below the lateral capacity the cyclic curvature never buckles the layer
    assert curve.governing[2] == "wire_yield"
    # above it, lateral buckling sets in once the wires slip
    assert curve.governing[3] == "lateral_buckling"
    assert 0.0 < curve.kappa_max_1_m[3] < curve.kappa_yield_1_m[3]
    gate = compression_screen(cable, -1.5 * cap, curve.kappa_max_1_m[3] * 1.001)
    assert gate.utilisation >= 1.0
    assert curve.compression_capacity_N == pytest.approx(cap)
    design = compute_compression_capacity(
        cable, tensions[:1], utilisation_factor=0.67, component_limits=False
    )
    assert design.kappa_yield_1_m[0] < curve.kappa_yield_1_m[0]


def test_capacity_curve_radial_branch_and_validation() -> None:
    cable = _two_layer_cable(with_sheath=False)
    cap = compression_screen(
        cable, -1.0, 0.0
    ).compression_capacity_N  # stick: radial only
    curve = compute_compression_capacity(cable, np.array([-1.1 * cap]))
    assert curve.governing == ["radial_buckling"] and curve.kappa_max_1_m[0] == 0.0
    with pytest.raises(CableInputError):
        compute_compression_capacity(cable, np.array([0.0]), kappa_search_max_1_m=0.0)


def test_lateral_fails_at_zero_curvature_without_friction() -> None:
    cable = _two_layer_cable()
    cap = compression_screen(cable, -1.0).compression_capacity_N
    curve = compute_compression_capacity(cable, np.array([-1.2 * cap]), mu=0.0)
    assert curve.kappa_lateral_1_m[0] == 0.0
    assert curve.kappa_max_1_m[0] == 0.0


# --------------------------------------------------------------------------- #
# Application regression: public 66 kV template                               #
# --------------------------------------------------------------------------- #
def test_iea_66kv_screen_regression() -> None:
    cable = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = cx.compression_screen(
            cable,
            -1.0e3,
            0.15,
            sheath_hoop_strength_Pa=20.74e6,
            sheath_strength_source="Malta et al. (2022) Table 4 (assumed)",
            utilisation_factor=0.67,
        )
    assert any(issubclass(w.category, cx.ExtrapolationWarning) for w in caught)
    assert res.governing == "lateral:outer_armour"
    assert res.compression_capacity_N == pytest.approx(6.93e3, rel=5e-3)
    assert [lat.P1_saevik_N for lat in res.lateral] == pytest.approx(
        [3.98e3, 2.35e3], rel=5e-3
    )
    assert res.radial.P2_star_N == pytest.approx(281e3, rel=5e-3)
    assert res.allowable_compression_N == pytest.approx(0.67 * 6.93e3, rel=5e-3)


def test_layer_types_restraints_and_missing_friction() -> None:
    """Offset cores, an outboard metallic shell, a non-armour helix, no μ in library."""
    from cablexsec.cables.schema import _LAYER_ADAPTER  # noqa: PLC0415

    def build(armour_material: str) -> CableData:
        raw = [
            {
                "name": "cores",
                "layer_type": "cylindrical_offset",
                "material": "hdpe",
                "n_cores": 3,
                "r_offset_m": 0.03,
                "r_inner_m": 0.0,
                "r_outer_m": 0.02,
            },
            {
                "name": "polymer_helix",
                "layer_type": "helical",
                "material": "hdpe",
                "n_wires": 40,
                "d_wire_m": 0.004,
                "r_helix_m": 0.074,
                "lay_angle_deg": 25.0,
            },
            {
                "name": "armour",
                "layer_type": "helical",
                "material": armour_material,
                "n_wires": 60,
                "d_wire_m": 0.005,
                "r_helix_m": 0.080,
                "lay_angle_deg": 25.0,
            },
            {
                "name": "shell",
                "layer_type": "cylindrical_shell",
                "material": "aluminium_corrugated",
                "r_mid_m": 0.0835,
                "t_m": 0.001,
            },
        ]
        return CableData(
            cable=CableMeta(name="layer types"),
            layers=tuple(_LAYER_ADAPTER.validate_python(x) for x in raw),
        )

    cable = build("galvanised_steel_armour")
    tape = TapeLayer("tape", 15e9, 9e-5, math.radians(84.0), 2, 0.085, 313e6, "test")
    res = compression_screen(cable, -1.0e4, 0.1, tapes=(tape,))
    assert [lat.layer for lat in res.lateral] == ["armour"]  # polymer helix skipped
    assert res.radial.restraint_layers == ("shell", "tape")
    assert math.isfinite(res.radial.P3_N)  # tape strength alone gives P3
    assert any("tape tape from: test" in a for a in res.assumptions)
    with pytest.raises(CableInputError, match="pass mu"):
        compression_screen(build("lead_sheath"), -1.0e4, 0.1)
    assert (
        compression_screen(build("lead_sheath"), -1.0e4, 0.1, mu=0.1).utilisation > 0.0
    )
    with pytest.raises(CableInputError):
        rectangular_wire_section(2e11, 0.7, 1e-3, 1e-3)
    with pytest.raises(CableInputError):
        first_slip_curvature(0.1, -1.0, ALPHA, STEEL, 0.08)


def test_capacity_curve_includes_the_component_limits() -> None:
    cable = _two_layer_cable()
    tensions = np.array([5.0e4, 0.0])
    curve = compute_compression_capacity(cable, tensions)
    ref = compute_capacity(cable, tensions)
    # The same component limit bounds both curves at the same curvature.
    assert curve.governing == ref.governing == ["component_limit"] * 2
    assert curve.kappa_max_1_m == pytest.approx(ref.kappa_max_1_m, rel=1e-9)
    assert curve.governing_component == ref.governing_component
