# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Public 66 kV three-core wet-design template (IEA Wind Task 49 / Janocha 2024).

Checks that the template reproduces the published geometry, declares a wet
design consistently, and that the standard analyses run on it with results in
the range of the published properties. Published values and their sources are
listed in doc/template_66kv.md; figure values are read from the plots and
carry the reading uncertainty stated there.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

import cablexsec as cx
from cablexsec._numba import HAVE_NUMBA
from cablexsec.cables.schema import (
    CableData,
    CableMeta,
    CylindricalLayer,
    CylindricalOffsetLayer,
    WaterIngressProperties,
    load_cable,
)
from cablexsec.mechanics.section_geometry import filler_area, is_interstitial_filler
from cablexsec.mechanics.stiffness import axial_stiffness_by_layer

#: Long friction histories run pure Python without numba (minutes per case).
needs_numba = pytest.mark.skipif(
    not HAVE_NUMBA, reason="long friction histories need the compiled kernels"
)

# Published values ([Janocha2024] Sections 2.1, 2.4, 3.3; [IEATask49] Table 13).
EA_PUBLISHED_N = 658.0e6
GJ_PUBLISHED_NM2 = 152.0e3
EI_SLIP_PUBLISHED_NM2 = 42.47e3
RATED_CURRENT_A = 125.0e6 / (3**0.5 * 66.0e3)  # 125 MW at 66 kV, Table 13


@pytest.fixture(scope="module")
def cable() -> CableData:
    return load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)


# --------------------------------------------------------------------------- #
# Provenance of the geometry                                                  #
# --------------------------------------------------------------------------- #
# Outer diameters of [Janocha2024] Table 1, 66 kV column [mm].
_TABLE1_OD_MM = {
    "conductor_screen": 33.90,
    "xlpe_insulation": 57.70,
    "cu_tape_screen": 64.00,
    "mdpe_core_sheath": 70.40,
    "mdpe_filler": 151.80,
    "ppy_bedding_inner": 152.60,
    "mdpe_inner_sheath": 160.70,
    "ppy_bedding_armour": 167.40,
    "hdpe_outer_sheath": 184.00,
}


@pytest.mark.parametrize(("name", "od_mm"), list(_TABLE1_OD_MM.items()))
def test_layer_outer_diameters_follow_table_1(cable, name, od_mm) -> None:
    layer = cable.get_layer(name)
    assert 2e3 * layer.r_outer_m == pytest.approx(od_mm, abs=1e-9)


def test_screens_follow_the_source_section_model(cable) -> None:
    """Janocha Table 3 merges the screens into the conductor and the insulation.

    Read as solid copper, the screen walls of Table 1 gave 76.3 kg/m against
    the published 55.76 kg/m (IEA Table 13; 547 N/m in air, Janocha Table 5).
    """
    conductor_screen = cable.get_layer("conductor_screen")
    insulation_screen = cable.get_layer("insulation_screen")
    tape = cable.get_layer("cu_tape_screen")
    assert conductor_screen.material == insulation_screen.material == "xlpe_janocha2024"
    assert not conductor_screen.structural and insulation_screen.structural
    assert not tape.structural and tape.material == "copper_tape_screen"
    assert 1e3 * (tape.r_outer_m - tape.r_inner_m) == pytest.approx(0.10, abs=1e-12)
    mass = cx.compute_mass_properties(cable)
    assert mass.mass_kg_per_m == pytest.approx(55.76, rel=0.01)
    assert mass.weight_in_air_N_per_m == pytest.approx(547.0, rel=0.01)
    copper = sum(v for k, v in mass.layer_mass_kg_per_m.items() if k.startswith("cu_"))
    # The conductors and the assumed tape: 17.06 + 0.55 kg/m.
    assert copper == pytest.approx(17.61, abs=0.01)


def test_armour_and_conductor_follow_the_source(cable) -> None:
    inner, outer = cable.get_layer("inner_armour"), cable.get_layer("outer_armour")
    assert (inner.n_wires, outer.n_wires) == (139, 145)  # Fig. 4a count
    assert inner.d_wire_m == outer.d_wire_m == pytest.approx(3.15e-3)
    assert (inner.lay_angle_deg, outer.lay_angle_deg) == (13.0, 10.0)
    assert inner.lay_direction != outer.lay_direction
    # Wires sit on the Table 1 annuli: 160.70-167.00 and 167.40-173.70 mm.
    assert 2e3 * (inner.r_helix_m - inner.d_wire_m / 2) == pytest.approx(160.70)
    assert 2e3 * (inner.r_helix_m + inner.d_wire_m / 2) == pytest.approx(167.00)
    assert 2e3 * (outer.r_helix_m - outer.d_wire_m / 2) == pytest.approx(167.40)
    assert 2e3 * (outer.r_helix_m + outer.d_wire_m / 2) == pytest.approx(173.70)
    conductor = cable.get_layer("cu_conductor")
    assert conductor.conductor_area_mm2 == 630.0
    assert cable.core_lay is not None
    lay = np.degrees(
        np.arctan(2 * np.pi * conductor.r_offset_m / cable.core_lay.pitch_m)
    )
    assert lay == pytest.approx(10.0, abs=0.01)  # Table 1 bundle lay angle
    assert cable.aggregate_properties.OD_mm == 184.0
    assert cable.is_fully_specified()


def test_template_layers_state_their_sources() -> None:
    text = cx.iea_task49_66kv_yaml_path().read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    for layer in raw["layers"]:
        assert layer["source"].strip(), f"{layer['name']} has no source"


# --------------------------------------------------------------------------- #
# Wet-design declaration                                                      #
# --------------------------------------------------------------------------- #
def test_template_declares_wet_design_without_water_barrier(cable) -> None:
    water = cable.water_ingress
    assert water is not None
    assert (water.design, water.armour_annulus) == ("wet", "flooded")
    assert "CIGRE TB 610" in water.source
    assert not any(x.layer_type == "cylindrical_shell" for x in cable.layers)
    # Flooded armour: a seawater S-N curve on every armour layer.
    for layer in cable.helical_layers:
        assert layer.bulk_sn == "saevik_2011_flat_steel"


def _with_water(path, water: dict) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["water_ingress"] = water
    return raw


def _load_raw(tmp_path, raw: dict) -> CableData:
    path = tmp_path / "cable.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return load_cable(path, geometry_override=None)


def test_wet_design_rejects_a_metallic_water_barrier(tmp_path) -> None:
    raw = _with_water(
        cx.export_three_core_yaml_path(),
        {"design": "wet", "armour_annulus": "flooded"},
    )
    with pytest.raises(ValidationError, match="wet design"):
        _load_raw(tmp_path, raw)


def test_dry_design_requires_a_metallic_water_barrier(tmp_path) -> None:
    raw = _with_water(
        cx.inter_array_three_core_yaml_path(),
        {"design": "dry", "armour_annulus": "sealed"},
    )
    with pytest.raises(ValidationError, match="dry design"):
        _load_raw(tmp_path, raw)
    dry = _load_raw(
        tmp_path,
        _with_water(
            cx.export_three_core_yaml_path(),
            {"design": "dry", "armour_annulus": "sealed", "source": "test"},
        ),
    )
    assert dry.water_ingress.design == "dry"


def test_water_ingress_rejects_unknown_values() -> None:
    with pytest.raises(ValidationError):
        WaterIngressProperties(design="semi", armour_annulus="flooded")
    with pytest.raises(ValidationError):
        WaterIngressProperties(design="wet", armour_annulus="flooded", extra=1)


def test_water_ingress_is_optional_and_round_trips(cable) -> None:
    assert (
        load_cable(cx.menard2023_yaml_path(), geometry_override=None).water_ingress
        is None
    )
    again = CableData.model_validate(cable.model_dump(mode="json"))
    assert again == cable


# --------------------------------------------------------------------------- #
# Interstitial filler of any material                                         #
# --------------------------------------------------------------------------- #
def test_filler_of_any_material_excludes_the_core_envelopes(cable) -> None:
    filler = cable.get_layer("mdpe_filler")
    assert is_interstitial_filler(filler, cable)
    area = filler_area(filler, cable)
    disc = np.pi * filler.r_outer_m**2
    assert 0.2 * disc < area < 0.5 * disc
    contribution = axial_stiffness_by_layer(cable)["mdpe_filler"]
    assert contribution == pytest.approx(1.0e9 * area, rel=1e-12)
    # A bedding annulus outside the cores is not a filler.
    assert not is_interstitial_filler(cable.get_layer("ppy_bedding_inner"), cable)


def test_axis_disc_is_not_a_filler_in_a_single_core_cable() -> None:
    core = CylindricalOffsetLayer(
        name="core",
        material="copper_janocha2024",
        n_cores=1,
        r_offset_m=0.0,
        r_inner_m=0.0,
        r_outer_m=0.01,
    )
    disc = CylindricalLayer(
        name="disc",
        material="mdpe_janocha2024",
        r_inner_m=0.0,
        r_outer_m=0.02,
        structural=False,
    )
    # A disc from the axis through a concentric core is not a filler: the
    # cable is refused when it is built.
    with pytest.raises(ValidationError, match="overlaps or lies inside"):
        CableData(cable=CableMeta(name="single"), layers=(core, disc))


# --------------------------------------------------------------------------- #
# Standard analyses against the published properties                         #
# --------------------------------------------------------------------------- #
def test_section_stiffness_against_published_values(cable) -> None:
    s = cx.compute_stiffness(cable)
    # EA: within 5 % of the published 658 MN (end-twist condition not stated).
    assert s.ea_fixed_N == pytest.approx(EA_PUBLISHED_N, rel=0.05)
    assert s.ea_free_N < s.ea_fixed_N
    # GJ here is the extension-only lay term: below the published value, which
    # also contains polymer and core torsion, but of the same size.
    assert 0.75 * GJ_PUBLISHED_NM2 < s.gj_Nm2 < GJ_PUBLISHED_NM2
    # Full-slip EI is a lower bound of the published slipping stiffness.
    assert 0.4 * EI_SLIP_PUBLISHED_NM2 < s.ei_min_Nm2 < EI_SLIP_PUBLISHED_NM2
    assert s.ei_min_Nm2 < s.ei_max_Nm2 <= s.ei_plane_section_Nm2


@pytest.mark.parametrize("contact_model", [None, "solved"])
def test_moment_curvature_loop(cable, contact_model) -> None:
    # Janocha Fig. 6 loading: 50 kN, +-0.2 1/m.
    kappa = 0.2 * np.sin(np.linspace(0.0, 4.0 * np.pi, 401))
    loop = cx.compute_mkappa(cable, kappa, tension_N=50e3, contact_model=contact_model)
    assert loop.W_diss_J_per_m > 0.0
    assert loop.EI_min_Nm2 <= loop.EI_apparent_Nm2 <= loop.EI_max_Nm2
    # Hysteretic: the two branches differ at zero curvature on the second cycle.
    second = slice(len(kappa) // 2, None)
    zero = np.flatnonzero(np.abs(kappa[second]) < 1e-12)
    assert np.ptp(loop.moment_Nm[second][zero]) > 0.0
    # Peak moment below the published loop (about 6.8 kNm, Fig. 6): the
    # friction of the cores, fillers and conductor wires is not modelled.
    assert 3.0e3 < loop.moment_Nm.max() < 6.8e3


def test_tension_stresses_against_published_stress_factors(cable) -> None:
    # Fig. 19b at 1000 kN: armour about 355 MPa (von Mises), conductor about
    # 190 MPa. The wire axial stress is the dominant part.
    stress = cx.compute_stress(cable, 0.0, 0.0, tension_N=1.0e6)
    for name in ("inner_armour", "outer_armour"):
        sigma = float(np.max(stress.sigma_A_Pa[name]))
        assert 0.75 * 355e6 < sigma < 355e6
    cap = cx.compute_capacity(cable, np.array([1.0e6]))
    sigma_cu = cap.uf_conductor[0] * 220e6  # copper_janocha2024 UTS basis
    assert sigma_cu == pytest.approx(190e6, rel=0.15)


def test_capacity_against_published_armour_yield(cable) -> None:
    # Section 3.3: armour yields at about 1 1/m in pure bending and about
    # 1000 kN in pure tension.
    cap = cx.compute_capacity(
        cable, np.array([0.0, 1.0e6, 1.1e6]), component_limits=False
    )
    assert cap.governing[0] == "wire_yield"
    assert cap.kappa_max_1_m[0] == pytest.approx(1.0, rel=0.25)
    assert 0.0 < cap.kappa_max_1_m[1] < 0.25
    assert cap.kappa_max_1_m[2] < 0.01  # capacity practically exhausted at 1.1 MN


@needs_numba
def test_fatigue_runs_for_both_friction_and_contact_models(cable) -> None:
    rng = np.random.default_rng(66)
    n = 240
    t = np.linspace(0.0, 240.0, n)
    kx = 0.004 * np.sin(2 * np.pi * t / 10) + 0.001 * rng.standard_normal(n)
    ky = 0.002 * np.sin(2 * np.pi * t / 13 + 0.4)
    tension = 50e3 + 5e3 * np.sin(2 * np.pi * t / 10 + 0.3)
    damage = {}
    for friction_model in ("direction_tied", "distributed"):
        for contact_model in (None, "solved"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = cx.compute_fatigue(
                    cable,
                    kx,
                    ky,
                    tension_N=tension,
                    friction_model=friction_model,
                    contact_model=contact_model,
                )
            d = max(float(np.max(v)) for v in result.D_total_per_wire.values())
            assert np.isfinite(d) and d > 0.0
            damage[friction_model, contact_model] = d
    # The two friction laws agree closely on this smooth history.
    for contact_model in (None, "solved"):
        ratio = (
            damage["distributed", contact_model]
            / damage["direction_tied", contact_model]
        )
        assert 0.8 < ratio < 1.25


def test_thermal_rating_is_consistent_with_published_power(cable) -> None:
    th = cx.compute_thermal(
        cable,
        current_A=RATED_CURRENT_A,
        conductor_area_m2=630e-6,
        conductor_material="copper_janocha2024",
    )
    # 125 MW (Table 13) is a 90 degC XLPE rating; the lumped ladder with the
    # equal-filler-area annulus and the assumed seabed T4 lands close to it
    # (95.4 degC: the semiconducting screens count with the insulation, and
    # the stranding voids of the area-equivalent conductor add about 1 K).
    assert 85.0 < th.T_conductor_C < 100.0
    limitations = " ".join(th.provenance.limitations)
    assert "equal-filler-area annulus" in limitations
