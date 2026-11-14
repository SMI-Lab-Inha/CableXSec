# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Gaps P1, P6, P8: per-core screens, core slip state, sheath strain life.

Closed-form and limit checks; the Thies (2016) example reproduces the
hand calculation of the restrictor reanalysis (kappa r with the modified
universal slopes, and the core-level helical screen).
"""

from __future__ import annotations

import math
import textwrap

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.fatigue import miner_damage, wire_stress_field
from cablexsec.cable_adapters import cable_to_fatigue_layers
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.materials import library
from cablexsec.materials.strain_life import (
    STRAIN_LIFE,
    StrainLifeConstants,
    get_strain_life,
    masing_stresses,
    register_strain_life,
    strain_life_damage,
)
from cablexsec.mechanics.core_friction import CoreFrictionSpec

pytestmark = pytest.mark.analytical_sanity

KAPPA = 1.0 / 2.54


def _sine(amplitude: float, cycles: int = 4, n: int = 96) -> np.ndarray:
    t = np.linspace(0.0, float(cycles), cycles * n + 1)
    return np.asarray(amplitude * np.cos(2.0 * np.pi * t))


@pytest.fixture(scope="module")
def thies():
    return cx.load_cable(cx.thies2016_yaml_path(), geometry_override=None)


@pytest.fixture(scope="module")
def aligned(thies):
    return thies.model_copy(update={"core_lay": None})


# --------------------------------------------------------------------------- #
# P8: core slip state                                                          #
# --------------------------------------------------------------------------- #
def test_stuck_core_gives_plane_section_strain(aligned, thies):
    k = _sine(KAPPA, 1)
    rho = 0.0202
    field = cx.compute_core_layer_strain(
        aligned, "al_laminate", k, core_state="stuck", n_theta=12, n_stations=8
    )
    phi = np.radians(field.phase_deg)[:, None, None]
    theta = np.radians(field.theta_deg)[None, :, None]
    x = 0.02625 * np.cos(phi) + rho * np.cos(phi + theta)
    assert np.allclose(field.strain, x * k[None, None, :], rtol=0, atol=1e-15)
    # Laid core: helical projection cos^2(beta) of the plane-section strain.
    laid = cx.core_kinematics(thies, k, state="stuck", n_stations=8)
    beta = laid.lay_angle_rad
    assert beta == pytest.approx(math.atan2(2 * math.pi * 0.02625, 2.0))
    expected = 0.02625 * math.cos(beta) ** 2 * np.cos(laid.phase_rad)[:, None] * k
    assert np.allclose(laid.axis_strain, expected, rtol=0, atol=1e-16)
    assert laid.xi == 1.0


def test_free_core_gives_zero_mean_strain(thies):
    k = _sine(KAPPA, 1)
    field = cx.compute_core_layer_strain(
        thies, "al_laminate", k, k * 0.3, core_state="free", n_theta=24
    )
    assert np.max(np.abs(field.mean_over_section)) < 1e-17
    assert np.all(field.kinematics.axis_strain == 0.0)
    # Bending about the core axis only: amplitude kappa_core * rho.
    beta = field.kinematics.lay_angle_rad
    c, s = math.cos(beta), math.sin(beta)
    bound = (
        math.hypot(1.0, 0.3)
        * KAPPA
        * 0.0202
        * max(c * c * math.cos(2 * beta), c * (1 + s * s))
    )
    assert np.max(field.amplitude) <= bound * (1 + 1e-12)


def test_frictional_state_lies_between_the_bounds(thies):
    k = _sine(KAPPA, 2)
    stuck = cx.core_kinematics(thies, k, state="stuck", n_stations=180)
    loose = cx.core_kinematics(
        thies,
        k,
        state="frictional",
        core_friction=CoreFrictionSpec(
            mu_core_filler=0.0, mu_core_core=0.0, mu_core_sheath=0.0
        ),
    )
    assert np.max(np.abs(loose.axis_strain)) < 1e-15
    rigid = cx.core_kinematics(
        thies,
        k,
        state="frictional",
        core_friction=CoreFrictionSpec(
            mu_core_filler=50.0,
            mu_core_core=50.0,
            mu_core_sheath=50.0,
            core_shear=False,
        ),
    )
    assert rigid.xi == pytest.approx(1.0)
    assert np.allclose(rigid.axis_strain, stuck.axis_strain, rtol=0, atol=1e-12)
    assert not np.any(rigid.slipping)
    nominal = cx.core_kinematics(thies, k, state="frictional")
    assert np.all(
        np.abs(nominal.axis_strain) <= np.abs(stuck.axis_strain).max() + 1e-15
    )
    assert nominal.slipping is not None and np.any(nominal.slipping)
    with pytest.raises(CableInputError, match="n_stations_per_core"):
        cx.core_kinematics(thies, k, state="frictional", n_stations=7)


def test_core_state_is_required_and_checked(thies):
    k = _sine(KAPPA, 1)
    with pytest.raises(CableInputError, match="core slip state"):
        cx.core_kinematics(thies, k, state="sliding")  # type: ignore[arg-type]
    with pytest.raises(CableSchemaError, match="cylindrical_offset"):
        cx.compute_core_layer_strain(thies, "armour", k, core_state="free")


# --------------------------------------------------------------------------- #
# P1: per-core helical screen                                                  #
# --------------------------------------------------------------------------- #
_CORE_POLYMER = """
  - name: insulation
    layer_type: {kind}
    material: xlpe_qiao2025{offset}
    r_inner_m: 0.0
    r_outer_m: 0.0190
"""

_TAPE = """
  - name: cu_tape
    layer_type: {kind}
    material: copper_tape_screen{offset}
    r_inner_m: 0.0196
    r_outer_m: 0.0197
    structural: false
"""


def _write(tmp_path, name, body):
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        textwrap.dedent(f'cable:\n  name: "{name}"\nlayers:\n') + body, encoding="utf-8"
    )
    return cx.load_cable(path, geometry_override=None)


def test_single_core_screen_reduces_to_the_helical_layer(tmp_path):
    p = 2.0e5
    helical = _write(
        tmp_path,
        "helical",
        _CORE_POLYMER.format(kind="cylindrical", offset="")
        + """
  - name: screen
    layer_type: helical
    material: copper_wire_screen
    n_wires: 53
    d_wire_m: 0.0006
    r_helix_m: 0.0193
    lay_angle_deg: 10.0
    manufacturing_clamping:
      initial_contact_pressure_Pa: 4.0e5
      basis: assumed
      source: "test: P1 + P2 = 2 p"
    bulk_sn: nasution_2014_copper
"""
        + _TAPE.format(kind="cylindrical", offset=""),
    )
    offset = "\n    n_cores: 1\n    r_offset_m: 0.0"
    pitch = 2 * math.pi * 0.0193 / math.tan(math.radians(10.0))
    core = _write(
        tmp_path,
        "core",
        _CORE_POLYMER.format(kind="cylindrical_offset", offset=offset)
        + f"""
  - name: screen
    layer_type: cylindrical_offset
    material: copper_wire_screen
    n_cores: 1
    r_offset_m: 0.0
    r_inner_m: 0.0190
    r_outer_m: 0.0196
    construction: resolved_wire_screen
    strand_layers:
      - {{n_wires: 53, d_wire_m: 0.0006, r_helix_m: 0.0193, pitch_m: {pitch!r}}}
    outer_face_pressure:
      pressure_Pa: {p}
      identification: assumed_bound
      bounds_Pa: [5.0e4, 5.0e5]
      source: "test"
    bulk_sn: nasution_2014_copper
"""
        + _TAPE.format(kind="cylindrical_offset", offset=offset),
    )
    k = _sine(KAPPA, 3)
    ky = 0.2 * _sine(KAPPA, 3)[::-1]
    layer = cable_to_fatigue_layers(helical, mu=0.2)[0]
    reference = wire_stress_field(layer, k, ky)
    for state in ("stuck", "free"):
        result = cx.compute_core_screen_fatigue(
            core, k, ky, core_state=state, mu=0.2, n_stations=1, knockdown=None
        )
        assert result.xi == pytest.approx(layer.spec.xi, rel=1e-12)
        assert result.p_inner_Pa + result.p_outer_Pa == pytest.approx(
            layer.spec.contact_pressure_Pa
        )
        for f, angle in enumerate((0.0, math.pi / 2, math.pi, 1.5 * math.pi)):
            expected = np.ptp(reference.at_fibre(angle), axis=1)
            assert np.allclose(result.stress_range_Pa[0, :, f], expected, rtol=1e-9)
        damage = [
            max(
                miner_damage(reference.at_fibre(a)[w], layer.sn_bulk, 1.0)
                for a in (0.0, math.pi / 2, math.pi, 1.5 * math.pi)
            )
            for w in range(53)
        ]
        assert np.allclose(result.D_bulk[0], damage, rtol=1e-9)


def test_screen_outer_face_carries_the_declared_pressure(thies):
    k = _sine(KAPPA, 2)
    result = cx.compute_core_screen_fatigue(thies, k, core_state="free", n_stations=2)
    assert result.p_outer_Pa == 2.0e5 and result.p_inner_Pa == 2.0e5
    assert set(result.faces[0]) == {"inner_face", "outer_face"}
    outer = result.faces[0]["outer_face"]
    assert outer.partner == "cu_tape" and outer.contact.normal_load > 0.0
    assert result.mu_outer == library.get_property("copper_wire_screen", "mu_on_cu")
    added = cx.compute_core_screen_fatigue(
        thies, k, core_state="free", n_stations=2, core_contact=True
    )
    assert added.p_outer_Pa > result.p_outer_Pa
    assert "assumed_bound" in " ".join(result.provenance.limitations)


def test_stuck_core_adds_the_axis_strain_to_the_screen(thies):
    k = _sine(KAPPA, 2)
    free = cx.compute_core_screen_fatigue(
        thies, k, core_state="free", n_stations=4, knockdown=None
    )
    stuck = cx.compute_core_screen_fatigue(
        thies, k, core_state="stuck", n_stations=4, knockdown=None
    )
    # Station 0: the core axis strain range 2 kappa r_c cos^2 beta times E cos^2 a.
    beta = stuck.kinematics.lay_angle_rad
    assert stuck.axis_strain_range[0] == pytest.approx(
        2 * KAPPA * 0.02625 * math.cos(beta) ** 2
    )
    assert free.axis_strain_range[0] == 0.0
    assert np.max(stuck.stress_range_Pa) > 3 * np.max(free.stress_range_Pa)
    assert np.any(stuck.exceeds_yield)
    # The axis strain is common to the core layers: no extra slip.
    assert np.allclose(stuck.slip_amplitude_m, free.slip_amplitude_m, rtol=1e-12)


def test_thies_screen_reproduces_the_core_level_workaround(aligned):
    """Aligned cores, free state: the hand core-level helical-screen numbers."""
    from cablexsec.analysis.fretting_assessment import SECONDS_PER_YEAR

    t = np.linspace(0.0, 10.0, 961)
    k = KAPPA * np.sin(2 * np.pi * t)
    result = cx.compute_core_screen_fatigue(
        aligned,
        k,
        core_state="free",
        duration_s=10.0,
        mu=0.2,
        mu_outer=0.25,
        n_stations=1,
    )
    assert np.max(result.stress_range_Pa[0, :, 0]) / 1e6 == pytest.approx(
        254.0, abs=1.0
    )
    assert np.max(result.slip_amplitude_m) * 1e6 == pytest.approx(819.0, rel=0.01)
    assert np.max(result.D_bulk) / 10 == pytest.approx(7.2e-6, rel=0.01)
    face = result.faces[0]["outer_face"]
    fretting = face.D_fretting_per_year / (SECONDS_PER_YEAR / 10.0) / 10.0 * 76165
    assert np.max(fretting) == pytest.approx(43.0, rel=0.01)
    assert int(np.sum(fretting >= 1.0)) == 4


def test_screen_requires_a_resolved_ring(thies):
    template = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
    with pytest.raises(CableSchemaError, match="resolved_wire_screen"):
        cx.compute_core_screen_fatigue(template, _sine(KAPPA), core_state="free")
    with pytest.warns(cx.MissingClampingStateWarning):
        layers = tuple(
            x.model_copy(update={"outer_face_pressure": None})
            if x.name == "cu_screen_wires"
            else x
            for x in thies.layers
        )
        cx.compute_core_screen_fatigue(
            thies.model_copy(update={"layers": layers}),
            _sine(KAPPA),
            core_state="free",
            n_stations=1,
            knockdown=None,
        )


# --------------------------------------------------------------------------- #
# Schema and materials                                                         #
# --------------------------------------------------------------------------- #
def test_assumed_bound_pressure_class():
    with pytest.raises(ValueError, match="bounds_Pa"):
        cx.FacePressure(pressure_Pa=1e5, identification="assumed_bound", source="x")
    with pytest.raises(ValueError, match="inside bounds_Pa"):
        cx.FacePressure(
            pressure_Pa=1e6,
            identification="assumed_bound",
            source="x",
            bounds_Pa=(1e4, 1e5),
        )
    from cablexsec.cables.schema import InitialStateProperties

    state = InitialStateProperties(
        external_pressure_Pa=2e5,
        identification="assumed_bound",
        source="tape",
        bounds_Pa=(5e4, 5e5),
    )
    assert state.identification == "assumed_bound"
    with pytest.raises(ValueError, match="bounds_Pa"):
        InitialStateProperties(
            external_pressure_Pa=2e5, identification="assumed_bound", source="tape"
        )


def test_face_pressure_only_on_a_wire_screen(thies):
    layer = thies.get_layer("al_laminate")
    with pytest.raises(ValueError, match="resolved_wire_screen"):
        type(layer).model_validate(
            {
                **layer.model_dump(),
                "outer_face_pressure": {
                    "pressure_Pa": 1e5,
                    "identification": "measured",
                    "source": "x",
                },
            }
        )


def test_copper_screen_material_is_complete():
    for key in ("sigma_UTS_Pa", "sigma_y_Pa", "mu_on_cu", "mu_on_hdpe"):
        assert library.get_property("copper_wire_screen", key) > 0
        assert library.get_citation("copper_wire_screen", key).strip()
    assert "Wan2022" in library.get_citation("copper_wire_screen", "sigma_y_Pa")
    assert "ModelAssumption" in library.get_citation("copper_wire_screen", "mu_on_hdpe")


def test_user_material_registration():
    cx.register_material(
        "user_tape", {"E_Pa": (2.0e9, "[ModelAssumption] tape"), "nu": (0.4, "x")}
    )
    try:
        assert library.get_property("user_tape", "E_Pa") == 2.0e9
        with pytest.raises(CableInputError, match="already registered"):
            cx.register_material("user_tape", {"E_Pa": (1.0, "x")})
        cx.register_material("user_tape", {"E_Pa": (3.0e9, "x")}, replace_existing=True)
        assert library.get_property("user_tape", "E_Pa") == 3.0e9
    finally:
        cx.unregister_material("user_tape")
    with pytest.raises(CableInputError, match="built-in"):
        cx.register_material("hdpe", {"E_Pa": (1.0, "x")})
    with pytest.raises(CableInputError, match="citation"):
        cx.register_material("bad", {"E_Pa": (1.0, " ")})
    with pytest.raises(CableInputError, match="finite"):
        cx.register_material("bad", {"E_Pa": (math.nan, "x")})


# --------------------------------------------------------------------------- #
# P6: strain life                                                             #
# --------------------------------------------------------------------------- #
def _mus_range(two_n, su, e, d):
    n = two_n / 2
    return (
        0.0266 * d**0.155 * (su / e) ** -0.53 * n**-0.56
        + 1.17 * (su / e) ** 0.832 * n**-0.09
    )


def test_modified_universal_slopes_conversion_is_exact():
    c = StrainLifeConstants.from_modified_universal_slopes(
        "t", E_Pa=70e9, sigma_UTS_Pa=90e6, ductility=0.6, reference="test"
    )
    two_n = np.logspace(1, 8, 15)
    assert np.allclose(
        2 * c.strain_amplitude(two_n), _mus_range(two_n, 90e6, 70e9, 0.6), rtol=1e-12
    )
    eps_a = 0.5 * _mus_range(2e4, 90e6, 70e9, 0.6)
    assert c.cycles_to_failure(eps_a) == pytest.approx(1e4, rel=1e-9)


def test_copper_constants_are_the_registered_poon_set():
    from cablexsec.materials.sn_curves import get_swt_life

    a, b = get_strain_life("copper_poon2022"), get_swt_life("copper_poon2022")
    assert (a.sigma_f_prime_Pa, a.b, a.eps_f_prime, a.c, a.E_Pa) == (
        b.sigma_f_prime_Pa,
        b.b,
        b.eps_f_prime,
        b.c,
        b.E_Pa,
    )
    assert a.basis == "published"
    assert get_strain_life("aluminium_foil_mus_assumed").basis == "assumed"


def test_swt_equals_uncorrected_curve_when_fully_reversed():
    c = get_strain_life("copper_poon2022")
    eps = np.array([1e-4, 1e-3, 5e-3, 2e-2])
    plain = c.cycles_to_failure(eps)
    swt = c.cycles_to_failure(
        eps, sigma_max_Pa=c.stress_amplitude(eps), correction="swt"
    )
    assert np.allclose(swt, plain, rtol=1e-9)
    morrow = c.cycles_to_failure(eps, sigma_mean_Pa=0.0, correction="morrow")
    assert np.allclose(morrow, plain, rtol=1e-12)
    # A tensile mean shortens the life under both corrections.
    assert np.all(
        c.cycles_to_failure(eps, sigma_mean_Pa=50e6, correction="morrow") < plain
    )
    assert np.all(
        c.cycles_to_failure(
            eps, sigma_max_Pa=c.stress_amplitude(eps) + 50e6, correction="swt"
        )
        < plain
    )
    # No damage for a compressive maximum stress (SWT).
    assert np.all(
        np.isinf(c.cycles_to_failure(eps, sigma_max_Pa=-1.0, correction="swt"))
    )


def test_masing_branches_close_with_memory():
    c = get_strain_life("copper_poon2022")
    s = masing_stresses([0.005, -0.005, 0.005, -0.002, 0.003, -0.005, 0.006], c)
    amp = float(c.stress_amplitude(0.005))
    assert s[0] == pytest.approx(amp) and s[1] == pytest.approx(-amp)
    assert s[5] == pytest.approx(-amp, rel=1e-12)  # memory: inner loop closed
    assert s[6] == pytest.approx(float(c.stress_amplitude(0.006)))  # back on the curve
    # Doubled curve of the inner branch.
    assert s[4] - s[3] == pytest.approx(
        2 * float(c.stress_amplitude(0.0025)), rel=1e-12
    )


def test_strain_life_damage_counts_cycles():
    c = get_strain_life("copper_poon2022")
    res = strain_life_damage(_sine(0.004, 6), c, correction="none")
    assert res.n_cycles == pytest.approx(6.0)
    assert res.damage == pytest.approx(
        6.0 / float(c.cycles_to_failure(0.004)), rel=1e-6
    )


def test_user_strain_life_registration():
    c = StrainLifeConstants(
        "user_sl", 70e9, 300e6, -0.1, 0.3, -0.6, reference="test", basis="assumed"
    )
    register_strain_life(c)
    try:
        assert get_strain_life("user_sl") == c
        with pytest.raises(CableInputError, match="already registered"):
            register_strain_life(c)
    finally:
        STRAIN_LIFE.pop("user_sl")
    with pytest.raises(CableInputError, match="built-in"):
        register_strain_life(get_strain_life("copper_poon2022"))


def test_shell_fatigue_elastic_route_is_unchanged_and_strict():
    cable = cx.load_cable(cx.export_three_core_yaml_path(), geometry_override=None)
    k = np.array([-0.01, 0.01])
    base = cx.compute_shell_fatigue(cable, "al_metallic_sheath", k)
    assert base.method == "elastic_sn" and base.strain_life is None
    with pytest.raises(CableInputError, match="strain_life"):
        cx.compute_shell_fatigue(cable, "al_metallic_sheath", k, core_state="free")


def test_shell_strain_life_accepts_strain_above_yield():
    cable = cx.load_cable(cx.export_three_core_yaml_path(), geometry_override=None)
    layer = cable.get_layer("al_metallic_sheath")
    k = _sine(0.4)
    with pytest.raises(CableInputError, match="exceeds yield"):
        cx.compute_shell_fatigue(cable, "al_metallic_sheath", k)
    res = cx.compute_shell_fatigue(
        cable,
        "al_metallic_sheath",
        k,
        method="strain_life",
        strain_life="aluminium_foil_mus_assumed",
    )
    eps_a = layer.scf * layer.f_corrugation * 0.4 * layer.r_mid_m
    assert res.strain_life.strain_amplitude_max == pytest.approx(eps_a, rel=1e-12)
    c = get_strain_life("aluminium_foil_mus_assumed")
    assert res.D_miner == pytest.approx(
        4.0 / float(c.cycles_to_failure(eps_a)), rel=1e-6
    )
    assert "corrugation strain concentration" in " ".join(res.provenance.limitations)
    with pytest.raises(CableSchemaError, match="strain_life"):
        cx.compute_shell_fatigue(cable, "al_metallic_sheath", k, method="strain_life")


def test_thies_laminate_reproduces_the_hand_calculation(aligned, thies):
    """Free core, kappa r with the modified universal slopes (nominal inputs)."""
    r_al = 0.0202
    for cable, tol in ((aligned, 2e-3), (thies, 1e-2)):
        res = cx.compute_shell_fatigue(
            cable,
            "al_laminate",
            _sine(KAPPA),
            method="strain_life",
            core_state="free",
        )
        assert res.strain_life.core_state == "free"
        assert 1.0 < res.strain_life.strain_factor < 1.003  # bonded HDPE hoop strain
        per_cycle = res.D_miner / res.strain_life.n_cycles
        two_n = np.logspace(0, 9, 20001)
        target = 2 * KAPPA * r_al
        hand = 1.0 / (
            np.interp(-target, -_mus_range(two_n, 90e6, 70e9, 0.6), two_n) / 2
        )
        assert per_cycle == pytest.approx(hand, rel=tol)
    with pytest.raises(CableInputError, match="core_state"):
        cx.compute_shell_fatigue(
            thies, "al_laminate", _sine(KAPPA), method="strain_life"
        )


def test_66kv_template_has_no_metallic_sheath_or_wire_screen():
    template = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
    assert not [
        x for x in template.layers if isinstance(x, cx.api.CylindricalShellLayer)
    ]
    screen = template.get_layer("cu_wire_screen")
    assert not screen.strand_layers  # a full ring (Janocha Table 1)
    k = _sine(1.0 / 2.76, 1)
    amps = {
        s: cx.compute_core_layer_strain(
            template, "cu_wire_screen", k, core_state=s
        ).maximum()["strain_amplitude"]
        for s in ("free", "frictional", "stuck")
    }
    assert amps["free"] <= amps["frictional"] + 1e-12 <= amps["stuck"] + 2e-12
