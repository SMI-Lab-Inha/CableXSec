# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""User S-N curves, mean-stress corrections and component SCFs.

Verification: closed-form Miner sums, the published mean-stress statements of
DNV-RP-C203 (2024) Sec. 2.5, the textbook mean-stress identities (Goodman,
Gerber, Soderberg, SWT, Walker), round trips at the curve's own test
condition, and bit-for-bit defaults.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.fatigue import compute_fatigue as fatigue_specs
from cablexsec.analysis.fatigue import miner_damage, wire_stress_field
from cablexsec.analysis.sn_assessment import (
    ComponentSCF,
    corrected_damage,
    scaled_component,
)
from cablexsec.exceptions import CableInputError, CableLookupError, CableSchemaError
from cablexsec.materials.mean_stress import MeanStressSpec, equivalent_range
from cablexsec.materials.sn_curves import (
    SNcurve,
    get_sn_curve,
    register_sn_curve,
    unregister_sn_curve,
)
from cablexsec.mechanics.stiffness import ea_fixed

MPA = 1.0e6


def _cycles(range_mpa: float, n: int, mean_mpa: float = 0.0) -> np.ndarray:
    """``n`` full cycles of one range about one mean [Pa] (plus closure)."""
    peaks = np.tile([mean_mpa - range_mpa / 2, mean_mpa + range_mpa / 2], n)
    return np.r_[peaks, mean_mpa - range_mpa / 2] * MPA


@pytest.fixture
def user_curve():
    curve = SNcurve.from_knee(
        "test_dnv_d_air",
        stress_unit="MPa",
        m1=3.0,
        log_C1=12.164,
        m2=5.0,
        knee_cycles=1.0e7,
        environment="air",
        reference="[DNV-RP-C203] Table 2-1, D curve (test)",
    )
    register_sn_curve(curve)
    yield curve
    unregister_sn_curve("test_dnv_d_air")


# --------------------------------------------------------------------------- #
# Registry                                                                    #
# --------------------------------------------------------------------------- #
def test_bilinear_from_knee_reproduces_dnv_table(user_curve):
    """DNV-RP-C203 Table 2-1 D curve: fatigue limit 52.63 MPa, log a2 15.606."""
    # Table values are rounded (log a to three decimals): 0.1 % on the limit.
    assert user_curve.transition_stress() == pytest.approx(52.63, rel=1e-3)
    assert user_curve.log_C2 == pytest.approx(15.606, abs=1e-3)
    assert get_sn_curve("test_dnv_d_air") is user_curve
    assert "test_dnv_d_air" in cx.list_sn_curves()
    # Continuity at the knee.
    knee = user_curve.transition_stress()
    assert user_curve.cycles_to_failure(knee * (1 + 1e-12)) == pytest.approx(
        1.0e7, rel=1e-9
    )


def test_registry_rules(user_curve):
    with pytest.raises(CableInputError, match="already registered"):
        register_sn_curve(user_curve)
    register_sn_curve(user_curve, replace_existing=True)
    with pytest.raises(CableInputError, match="built-in"):
        register_sn_curve(
            dataclasses.replace(user_curve, name="saevik_2011_flat_steel")
        )
    with pytest.raises(CableInputError, match="built-in or blocked"):
        register_sn_curve(dataclasses.replace(user_curve, name="cigre_tb623_lead"))
    with pytest.raises(CableInputError, match="source"):
        register_sn_curve(dataclasses.replace(user_curve, name="x_curve", reference=""))
    with pytest.raises(CableInputError, match="environment"):
        register_sn_curve(
            dataclasses.replace(user_curve, name="x_curve", environment=" ")
        )
    with pytest.raises(CableInputError, match="built-in"):
        unregister_sn_curve("saevik_2011_flat_steel")
    with pytest.raises(CableLookupError):
        unregister_sn_curve("never_registered")
    with pytest.raises(CableInputError, match="reference_thickness_m"):
        dataclasses.replace(user_curve, thickness_exponent=0.2)


# --------------------------------------------------------------------------- #
# Closed-form Miner sums                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.analytical_sanity
@pytest.mark.parametrize("compiled", [False, True])
def test_closed_form_miner_bilinear_and_cutoff(user_curve, compiled):
    """Constant-amplitude blocks on both branches against n / N(range)."""
    from cablexsec.analysis.fatigue import rainflow_cycles

    for rng, branch in ((120.0, (3.0, 12.164)), (40.0, (5.0, user_curve.log_C2))):
        history = _cycles(rng, 200)
        counted = sum(c for _r, _m, c, _i, _j in rainflow_cycles(history))
        assert counted == pytest.approx(200.0, abs=1.0)
        life = 10 ** (branch[1] - branch[0] * math.log10(rng))
        damage = corrected_damage(history, user_curve, compiled=compiled)
        assert damage == pytest.approx(counted / life, rel=1e-12)
    mixed = np.r_[_cycles(120.0, 50), _cycles(40.0, 300)]
    assert corrected_damage(mixed, user_curve, compiled=compiled) == pytest.approx(
        miner_damage(mixed, user_curve, 0.0), rel=1e-12
    )
    cut = dataclasses.replace(user_curve, name="cut", threshold=45.0)
    assert corrected_damage(_cycles(40.0, 300), cut, compiled=compiled) == 0.0


@pytest.mark.analytical_sanity
def test_thickness_exponent_scales_single_slope_damage():
    curve = SNcurve(
        name="t_curve",
        stress_unit="MPa",
        m1=3.0,
        log_C1=12.0,
        thickness_exponent=0.25,
        reference_thickness_m=0.025,
        reference="test",
    )
    history = _cycles(80.0, 100)
    thin = corrected_damage(history, curve, thickness_m=0.010)
    thick = corrected_damage(history, curve, thickness_m=0.050)
    assert curve.thickness_factor(0.010) == 1.0  # t below t_ref uses t_ref
    assert thick / thin == pytest.approx((0.050 / 0.025) ** (0.25 * 3), rel=1e-12)
    with pytest.raises(CableInputError, match="thickness"):
        corrected_damage(history, curve)


# --------------------------------------------------------------------------- #
# Mean-stress corrections                                                     #
# --------------------------------------------------------------------------- #
def _eq(method, rng, mean, curve_R=-1.0, **kw):
    spec = MeanStressSpec(method=method, **kw)
    return float(equivalent_range([rng], [mean], spec, curve_R=curve_R)[0])


@pytest.mark.analytical_sanity
def test_textbook_identities_fully_reversed_curve():
    s = {"strength_source": "test"}
    # sigma_a 100, sigma_m 250, S_u 500, S_y 400 (Dowling 2013 Ch. 9 forms).
    assert _eq("goodman", 200.0, 250.0, uts_Pa=500.0, **s) == pytest.approx(400.0)
    assert _eq("gerber", 200.0, 250.0, uts_Pa=500.0, **s) == pytest.approx(200.0 / 0.75)
    assert _eq("soderberg", 200.0, 250.0, yield_Pa=400.0, **s) == pytest.approx(
        200.0 / (1 - 250.0 / 400.0)
    )
    assert _eq("swt", 200.0, 250.0) == pytest.approx(2 * math.sqrt(350.0 * 100.0))
    walker = {"walker_source": "test"}
    assert _eq("walker", 200.0, 250.0, walker_gamma=0.5, **walker) == pytest.approx(
        _eq("swt", 200.0, 250.0)
    )
    assert _eq("walker", 200.0, 250.0, walker_gamma=1.0, **walker) == pytest.approx(
        200.0
    )
    # Zero mean: every correction is the identity on a fully reversed curve.
    for method, kw in (
        ("goodman", {"uts_Pa": 500.0, **s}),
        ("gerber", {"uts_Pa": 500.0, **s}),
        ("soderberg", {"yield_Pa": 400.0, **s}),
        ("swt", {}),
        ("walker", {"walker_gamma": 0.3, **walker}),
    ):
        assert _eq(method, 200.0, 0.0, **kw) == pytest.approx(200.0, rel=1e-14)
    # SWT: no damage without tension in the cycle.
    assert _eq("swt", 200.0, -150.0) == 0.0


def test_compressive_mean_rule_and_static_failure():
    s = {"uts_Pa": 500.0, "strength_source": "test"}
    assert _eq("goodman", 200.0, -250.0, **s) == pytest.approx(200.0)
    assert _eq("goodman", 200.0, -250.0, compressive_mean="signed", **s) == (
        pytest.approx(200.0 / 1.5)
    )
    assert _eq("gerber", 200.0, -250.0, **s) == pytest.approx(200.0)
    with pytest.raises(CableInputError, match="static failure"):
        _eq("goodman", 200.0, 500.0, **s)
    with pytest.raises(CableInputError, match="source"):
        MeanStressSpec(method="goodman", uts_Pa=500.0)
    with pytest.raises(CableInputError, match="uts_Pa"):
        MeanStressSpec(method="gerber")
    with pytest.raises(CableInputError, match="walker"):
        MeanStressSpec(method="walker", walker_gamma=0.5)


@pytest.mark.analytical_sanity
def test_dnv_rp_c203_published_reduction_factors():
    """DNV-RP-C203 (2024) Sec. 2.5.1-2.5.2: f_m = 1.0/0.8/0.0 and 1.0/0.9/0.8."""
    rng = 100.0
    for method, values in (
        ("dnv_base", (1.0, 0.8, 0.0)),
        ("dnv_pwht", (1.0, 0.9, 0.8)),
    ):
        tension = _eq(method, rng, 60.0)
        zero = _eq(method, rng, 0.0)
        compression = _eq(method, rng, -60.0)
        assert (tension / rng, zero / rng, compression / rng) == pytest.approx(values)
    # Eq. (2.15) at sigma_t = 75, sigma_c = -25: (75 + 15) / 100.
    assert _eq("dnv_base", 100.0, 25.0) == pytest.approx(90.0)


@pytest.mark.analytical_sanity
@pytest.mark.parametrize("curve_R", [0.1, 0.5, -0.5])
def test_round_trip_at_the_curve_test_condition(curve_R):
    """A cycle at the curve's own R maps to itself for every correction."""
    amplitude = 80.0
    mean = amplitude * (1 + curve_R) / (1 - curve_R)
    cases = (
        ("goodman", {"uts_Pa": 900.0, "strength_source": "t"}),
        ("gerber", {"uts_Pa": 900.0, "strength_source": "t"}),
        ("soderberg", {"yield_Pa": 700.0, "strength_source": "t"}),
        ("swt", {}),
        ("walker", {"walker_gamma": 0.4, "walker_source": "t"}),
    )
    for method, kw in cases:
        if curve_R < 0 and method in ("goodman", "gerber", "soderberg"):
            kw = {**kw, "compressive_mean": "signed"}
        value = _eq(method, 2 * amplitude, mean, curve_R=curve_R, **kw)
        assert value == pytest.approx(2 * amplitude, rel=1e-12), method
        # Constant-mean curves round trip the same way.
        value = _eq(method, 2 * amplitude, mean, reference_mean_stress_Pa=mean, **kw)
        assert value == pytest.approx(2 * amplitude, rel=1e-9), method


def test_from_material_reads_strength_and_citation():
    from cablexsec.materials import library

    spec = MeanStressSpec.from_material("goodman", "galvanised_steel_armour")
    uts = library.get_property("galvanised_steel_armour", "sigma_UTS_Pa")
    assert spec.uts_Pa == uts
    assert "galvanised_steel_armour sigma_UTS_Pa" in spec.strength_source
    soderberg = MeanStressSpec.from_material("soderberg", "galvanised_steel_armour")
    assert soderberg.yield_Pa == library.get_property(
        "galvanised_steel_armour", "sigma_y_Pa"
    )


# --------------------------------------------------------------------------- #
# Component SCFs and the fatigue routes                                       #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def menard():
    return cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)


def _history(n=240):
    t = np.linspace(0, 12 * np.pi, n)
    return 0.05 * np.sin(t), 0.02 * np.cos(0.7 * t), 4.0e4 + 1.0e4 * np.sin(0.3 * t)


def test_defaults_bit_for_bit(menard):
    kx, ky, tension = _history()
    base = cx.compute_fatigue(menard, kx, ky, tension_N=tension)
    none = cx.compute_fatigue(
        menard,
        kx,
        ky,
        tension_N=tension,
        mean_stress=MeanStressSpec(),
        scf=ComponentSCF(),
    )
    for name, values in base.D_bulk_per_fibre.items():
        assert np.array_equal(values, none.D_bulk_per_fibre[name])
    assert base.provenance == none.provenance


@pytest.mark.analytical_sanity
def test_uniform_scf_scales_damage_by_k_to_the_m(menard):
    kx, ky, tension = _history()
    base = cx.compute_fatigue(menard, kx, ky, tension_N=tension)
    k = 1.3
    scaled = cx.compute_fatigue(
        menard,
        kx,
        ky,
        tension_N=tension,
        scf=ComponentSCF.uniform(axial=k, bending=k, friction=k, source="test"),
    )
    m = get_sn_curve("saevik_2011_flat_steel").m1
    for name, values in base.D_bulk_per_fibre.items():
        assert np.allclose(scaled.D_bulk_per_fibre[name], values * k**m, rtol=1e-9)
    # Mean-only factors leave the ranges, hence the uncorrected damage, alone.
    mean_only = cx.compute_fatigue(
        menard,
        kx,
        ky,
        tension_N=tension,
        scf=ComponentSCF(axial_mean=1.5, friction_mean=1.4, source="test"),
    )
    for name, values in base.D_bulk_per_fibre.items():
        assert np.allclose(mean_only.D_bulk_per_fibre[name], values, rtol=1e-9)
    assert "component SCF" in scaled.provenance.method


def test_component_parts_are_separated(menard):
    kx, ky, tension = _history(120)
    layer = cx.cable_to_fatigue_layers(menard, contact_tension_N=4.0e4)[0]
    field = wire_stress_field(
        layer, kx, ky, tension, ea_fixed(menard), keep_friction=True
    )
    scf = ComponentSCF(friction_dynamic=2.0, bending_normal_dynamic=3.0, source="t")
    theta = 0.7
    n, lat = field.fibre_offsets(theta)
    expected = (
        (field.mean_Pa - field.friction_Pa)
        + scaled_component(field.friction_Pa, 1.0, 2.0)
        + scaled_component(field.radial_Pa * n, 1.0, 3.0)
        + field.tangential_Pa * lat
    )
    assert np.allclose(field.scf_fibre(theta, scf), expected, rtol=0, atol=1e-6)
    identity = field.scf_fibre(theta, ComponentSCF())
    assert np.allclose(identity, field.at_fibre(theta), rtol=1e-12, atol=1e-6)


def test_goodman_and_user_curve_through_compute_fatigue(menard, user_curve):
    kx, ky, tension = _history()
    base = cx.compute_fatigue(menard, kx, ky, tension_N=tension)
    goodman = cx.compute_fatigue(
        menard, kx, ky, tension_N=tension, mean_stress="goodman"
    )
    for name, values in base.D_bulk_per_fibre.items():
        # Tensile mean wire stress: Goodman increases the damage.
        assert np.max(goodman.D_bulk_per_fibre[name]) > np.max(values)
    layers = [lyr.name for lyr in cx.cable_to_fatigue_layers(menard)]
    user = cx.compute_fatigue(
        menard,
        kx,
        ky,
        tension_N=tension,
        sn_curves=dict.fromkeys(layers, "test_dnv_d_air"),
    )
    direct = fatigue_specs(
        [
            dataclasses.replace(lyr, sn_bulk=user_curve)
            for lyr in cx.cable_to_fatigue_layers(menard, contact_tension_N=tension[0])
        ],
        kx,
        ky,
        tension_N=tension,
        EA_full_N=ea_fixed(menard),
        contact_pressure_Pa=None,
    )
    assert set(user.D_bulk_per_fibre) == set(direct.D_bulk_per_fibre)
    with pytest.raises(CableSchemaError, match="not assessed"):
        cx.compute_fatigue(menard, kx, ky, scf={"no_layer": ComponentSCF()})


# --------------------------------------------------------------------------- #
# Conductor, sheath, tape and sea-state batch routes                          #
# --------------------------------------------------------------------------- #
@pytest.mark.analytical_sanity
def test_conductor_route_scf_and_default():
    import check_conductor_strand_slip as check

    from cablexsec.analysis.conductor_fatigue import assess_conductor_fatigue

    sn = get_sn_curve("nasution_2014_copper")
    kappa = 0.15 * np.sin(np.linspace(0, 8 * np.pi, 161))

    def run(**kwargs):
        return assess_conductor_fatigue(
            check.nasution_rings(),
            check.NAS_E,
            0.3,
            kappa,
            np.zeros_like(kappa),
            np.full_like(kappa, check.NAS_T),
            sn_bulk=sn,
            swt_const=None,
            mu_wire=0.2,
            reference_tension_N=check.NAS_T,
            **kwargs,
        )

    base = run()
    same = run(mean_stress=MeanStressSpec(), scf=ComponentSCF())
    assert np.array_equal(base.D_bulk_per_ring, same.D_bulk_per_ring)
    k = 1.2
    scaled = run(scf=ComponentSCF.uniform(axial=k, friction=k, source="test"))
    assert np.allclose(
        scaled.D_bulk_per_ring, base.D_bulk_per_ring * k**sn.m1, rtol=1e-9
    )
    goodman = run(
        mean_stress=MeanStressSpec(
            method="goodman", uts_Pa=235.0e6, strength_source="test"
        )
    )
    # Nasution's curve is at R = 0.1: the bulk cycles of fully reversed
    # bending about the tension mean are mapped to that ratio.
    assert np.all(np.isfinite(goodman.D_bulk_per_ring))
    assert any("user S-N options" in s for s in goodman.provenance.limitations)


def test_sheath_route_options():
    cable = cx.load_cable(cx.export_three_core_yaml_path(), geometry_override=None)
    k = 0.012 * np.sin(np.linspace(0, 10 * np.pi, 201))
    base = cx.compute_shell_fatigue(cable, "al_metallic_sheath", k)
    same = cx.compute_shell_fatigue(
        cable, "al_metallic_sheath", k, mean_stress=MeanStressSpec()
    )
    assert same.D_miner == base.D_miner and same.n_cycles == base.n_cycles
    single = SNcurve(
        name="al_single", stress_unit="MPa", m1=7.0, log_C1=19.2598, reference="t"
    )
    plain = cx.compute_shell_fatigue(cable, "al_metallic_sheath", k, sn_curve=single)
    factor = 1.25
    amplified = cx.compute_shell_fatigue(
        cable,
        "al_metallic_sheath",
        k,
        sn_curve=single,
        scf=ComponentSCF.uniform(bending=factor, source="test"),
    )
    assert amplified.D_miner == pytest.approx(plain.D_miner * factor**7, rel=1e-9)
    with_mean = cx.compute_shell_fatigue(
        cable,
        "al_metallic_sheath",
        k,
        sn_curve=single,
        mean_stress=MeanStressSpec(
            method="goodman", uts_Pa=150.0e6, strength_source="test"
        ),
        static_mean_stress_Pa=30.0e6,
    )
    assert with_mean.D_miner > plain.D_miner
    assert any("user S-N options" in s for s in with_mean.provenance.limitations)
    with pytest.raises(CableInputError, match="elastic_sn"):
        cx.compute_shell_fatigue(
            cable,
            "al_metallic_sheath",
            k,
            method="strain_life",
            strain_life="aluminium_foil_mus_assumed",
            mean_stress=MeanStressSpec(),
        )


def test_tape_route_stress_sn_option():
    thies = cx.load_cable(cx.thies2016_yaml_path(), geometry_override=None)
    k = (1 / 2.54) * np.cos(2 * np.pi * np.linspace(0.0, 2.0, 97))
    base = cx.compute_tape_fatigue(
        thies, "cu_tape", k, state="stuck", core_state="free", n_stations=4
    )
    assert base.D_sn is None and base.D_sn_map is None
    curve = get_sn_curve("nasution_2014_copper")
    plain = cx.compute_tape_fatigue(
        thies,
        "cu_tape",
        k,
        state="stuck",
        core_state="free",
        n_stations=4,
        sn_curve=curve,
    )
    assert plain.D_miner == base.D_miner
    assert plain.D_sn is not None and plain.D_sn > 0
    factor = 1.1
    scaled = cx.compute_tape_fatigue(
        thies,
        "cu_tape",
        k,
        state="stuck",
        core_state="free",
        n_stations=4,
        sn_curve=curve,
        scf=ComponentSCF.uniform(axial=factor, source="test"),
    )
    assert scaled.D_sn == pytest.approx(plain.D_sn * factor**curve.m1, rel=1e-9)
    with pytest.raises(CableInputError, match="sn_curve"):
        cx.compute_tape_fatigue(
            thies,
            "cu_tape",
            k,
            state="stuck",
            core_state="free",
            n_stations=4,
            mean_stress=MeanStressSpec(),
        )


def test_sea_state_batch_passes_the_options(tmp_path):
    from test_sea_state_batch import _runs

    from cablexsec.analysis.sea_state_batch import (
        SeaStateBatchOptions,
        SectionRequest,
        _section_record,
        run_sea_state_batch,
    )

    cable = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
    runs = _runs(tmp_path, cx.compute_stiffness(cable))
    options = SeaStateBatchOptions(tension_basis="true")
    plain = SectionRequest("a", 2.0)
    assert set(_section_record(plain)) == {
        "name",
        "arc_length_m",
        "termination",
        "thermal",
        "fretting",
    }
    scf = ComponentSCF.uniform(axial=1.2, bending=1.2, friction=1.2, source="t")
    request = SectionRequest("a", 2.0, mean_stress="goodman", scf=scf)
    result = run_sea_state_batch(cable, runs, [request], options=options)
    history = result.histories["a"][runs[0].name]
    area = math.pi * result.manifest["external_diameter_m"] ** 2 / 4
    kx, ky = history.strain_gradient()
    direct = cx.compute_fatigue(
        cable,
        kx,
        ky,
        tension_N=history.true_tension_N(area),
        mean_stress="goodman",
        scf=scf,
    )
    record = result.sections["a"].runs[runs[0].name]["record_damage_max"]
    assert record == pytest.approx(
        max(float(np.max(v)) for v in direct.D_bulk_per_fibre.values()), rel=1e-12
    )
    assert result.manifest["sections"][0]["mean_stress"] == "goodman"
