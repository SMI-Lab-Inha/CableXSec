# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Stress-free reference, polymer relaxation and ratio reporting (gap 4.1b).

Gates (independent check):

* the network builders default ``T_ref`` to the ambient, so 0 A is the
  unenergised result bit for bit; ``uniform``/``isothermal`` keep 20 deg C;
  the basis is recorded;
* an explicit ``T_ref`` of 20 deg C reproduces the reported 66 kV state
  (P1+P2 294 kPa against 31 kPa, sigma_T +84 MPa at 50 kN and 1093 A);
* polymers without ``E(T)`` above 40 deg C warn;
* the relaxation factor scales the polymer moduli of every radial solve,
  reference included, but not the stick stiffness, and reproduces the
  check's relaxed state (189 against 33 kPa at f = 0.2, T_ref 15 deg C);
* ``sigma_T`` varies with tension within a block;
* every printed damage ratio carries the S-N slope; relaxation bounds are
  reported.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.thermal_coupling import (
    POLYMER_SOFTENING_WARNING_C,
    ThermalHistory,
    ThermalState,
    armour_thermal_state,
    compute_thermal_fatigue,
    thermal_fatigue_inputs,
    thermal_material_state,
)
from cablexsec.cable_adapters import cable_to_fatigue_layers
from cablexsec.cables.schema import CableData
from cablexsec.exceptions import CableInputError, PhysicalRangeError
from cablexsec.materials import library
from cablexsec.materials.sn_curves import get_sn_curve
from cablexsec.provenance import (
    MissingClampingStateWarning,
    UnmodelledPolymerSofteningWarning,
)

RATED_A = 1093.0  # [IEATask49] Table 13
TRANSFER_ALL = {
    "xlpe_janocha2024": "xlpe_qiao2025",
    "mdpe_janocha2024": "hdpe",
    "hdpe_janocha2024": "hdpe",
    "ppy_janocha2024": "hdpe",
}


@pytest.fixture(scope="module")
def t66() -> CableData:
    return cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)


@pytest.fixture(autouse=True)
def _quiet_known_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingClampingStateWarning)
        warnings.simplefilter("ignore", UnmodelledPolymerSofteningWarning)
        yield


def _record(n: int = 80):
    t = np.arange(float(n))
    kx = 0.004 * np.sin(2 * np.pi * t / 10)
    ky = 0.002 * np.sin(2 * np.pi * t / 13 + 0.4)
    tension = 50e3 + 5e3 * np.sin(2 * np.pi * t / 10)
    return kx, ky, tension


def test_builders_record_their_stress_free_reference(t66):
    steady = ThermalHistory.steady(t66, 0.0)
    assert steady.T_ref_C == t66.environment.ambient_C == 15.0
    assert steady.T_ref_basis.startswith("ambient")
    assert steady.blocks()[0].state.is_reference
    tau = np.array([0.0, 3600.0])
    transient = ThermalHistory.from_current(t66, tau, [0.0, 500.0])
    assert transient.T_ref_C == 15.0 and "ambient" in transient.T_ref_basis
    uniform = ThermalHistory.uniform(t66, [0.0], 30.0)
    assert uniform.T_ref_C == library.MODULUS_REFERENCE_TEMPERATURE_C == 20.0
    assert uniform.T_ref_basis.startswith("laboratory")
    assert ThermalHistory.isothermal(t66).blocks()[0].state.is_reference
    explicit = ThermalHistory.steady(t66, 0.0, T_ref_C=20.0)
    assert explicit.T_ref_basis == "explicit, 20 deg C"
    assert not explicit.blocks()[0].state.is_reference


def test_zero_current_at_ambient_is_the_unenergised_result_bitwise(t66):
    kx, ky, tension = _record()
    base = cx.compute_fatigue(t66, kx, ky, tension_N=tension, contact_model="solved")
    cold = cx.compute_fatigue(
        t66, kx, ky, tension_N=tension, thermal=ThermalHistory.steady(t66, 0.0)
    )
    for name, damage in base.D_total_per_wire.items():
        assert np.array_equal(damage, cold.D_total_per_wire[name])


def test_explicit_reference_reproduces_the_reported_state(t66):
    hot20 = ThermalHistory.steady(t66, RATED_A, T_ref_C=20.0).blocks()[0].state
    a = armour_thermal_state(t66, hot20, tension_N=50e3)["inner_armour"]
    # Armour temperature at its gap node (24 Oct 2026; before, it was
    # interpolated into the next annulus: 294.006 kPa, 84.18 MPa).
    assert a.contact_pressure_Pa == pytest.approx(292.957e3, rel=1e-5)
    assert a.contact_pressure_reference_Pa == pytest.approx(31.104e3, rel=1e-4)
    assert a.thermal_wire_stress_Pa == pytest.approx(84.30e6, rel=1e-4)
    cold20 = ThermalHistory.steady(t66, 0.0, T_ref_C=20.0).blocks()[0].state
    c = armour_thermal_state(t66, cold20, tension_N=50e3)["inner_armour"]
    # 5 K below the stress-free state: the sheath shrink lifts cold clamping.
    assert c.contact_pressure_Pa == pytest.approx(51.83e3, rel=1e-4)
    # At the default (ambient) reference the cold state is the reference.
    hot15 = ThermalHistory.steady(t66, RATED_A).blocks()[0].state
    b = armour_thermal_state(t66, hot15, tension_N=50e3)["inner_armour"]
    assert b.contact_pressure_reference_Pa == a.contact_pressure_reference_Pa
    assert b.contact_pressure_Pa > a.contact_pressure_Pa


def test_hot_polymers_without_a_modulus_law_warn(t66):
    def caught(state: ThermalState) -> list[warnings.WarningMessage]:
        with warnings.catch_warnings(record=True) as out:
            warnings.simplefilter("always")
            thermal_material_state(t66, state)
        return [
            w for w in out if issubclass(w.category, UnmodelledPolymerSofteningWarning)
        ]

    below = POLYMER_SOFTENING_WARNING_C - 5.0
    above = POLYMER_SOFTENING_WARNING_C + 5.0
    assert not caught(ThermalHistory.uniform(t66, [0.0], below).state_at(0.0))
    hot = caught(ThermalHistory.uniform(t66, [0.0], above).state_at(0.0))
    assert len(hot) == 1 and "hdpe_outer_sheath" in str(hot[0].message)
    covered = ThermalHistory.uniform(
        t66, [0.0], above, modulus_transfer=TRANSFER_ALL
    ).state_at(0.0)
    assert not caught(covered)
    switched_off = ThermalHistory.uniform(
        t66,
        [0.0],
        above,
        modulus_transfer=TRANSFER_ALL,
        temperature_dependent_modulus=False,
    ).state_at(0.0)
    assert "switched off" in str(caught(switched_off)[0].message)


def test_registered_laws_are_continuous_at_a_non_library_reference():
    cable = cx.load_cable(cx.cigre_tb880_yaml_path(), geometry_override=None)
    laws = [x for x in cable.layers if x.material in library.TEMPERATURE_MODULUS_MODELS]
    assert laws
    near = ThermalState({x.name: 15.0 + 1e-3 for x in cable.layers}, T_ref_C=15.0)
    moduli = thermal_material_state(cable, near).radial_inputs["youngs_modulus_Pa"]
    for layer in laws:
        e_ref = library.get_property(layer.material, "E_Pa")
        assert moduli[layer.name] == pytest.approx(e_ref, rel=1e-3)
    at20 = ThermalState({x.name: 40.0 for x in cable.layers}, T_ref_C=20.0)
    moduli = thermal_material_state(cable, at20).radial_inputs["youngs_modulus_Pa"]
    for layer in laws:
        assert moduli[layer.name] == library.youngs_modulus_at_temperature(
            layer.material, 40.0
        )


@pytest.mark.parametrize("factor", [0.0, -0.1, 1.5, float("nan")])
def test_relaxation_factor_is_validated(t66, factor):
    with pytest.raises((CableInputError, PhysicalRangeError)):
        ThermalHistory.uniform(t66, [0.0], 30.0, polymer_relaxation_factor=factor)


def test_relaxation_scales_the_sustained_state_not_the_stick_stiffness(t66):
    plain = ThermalHistory.steady(t66, RATED_A).blocks()[0].state
    one = ThermalHistory.steady(t66, RATED_A, polymer_relaxation_factor=1.0)
    relaxed = ThermalHistory.steady(t66, RATED_A, polymer_relaxation_factor=0.2)
    a = armour_thermal_state(t66, plain, tension_N=50e3)["inner_armour"]
    b = armour_thermal_state(t66, one.blocks()[0].state, tension_N=50e3)["inner_armour"]
    c = armour_thermal_state(t66, relaxed.blocks()[0].state, tension_N=50e3)[
        "inner_armour"
    ]
    assert a == b
    # The independent check's relaxed variant (T_ref 15 deg C, f = 0.2).
    assert c.contact_pressure_Pa == pytest.approx(188.06e3, rel=1e-3)
    assert c.contact_pressure_reference_Pa == pytest.approx(33.10e3, rel=1e-3)
    assert 0.0 < c.thermal_wire_stress_Pa < a.thermal_wire_stress_Pa
    assert c.stick_modulus_Pa == a.stick_modulus_Pa
    # The relaxed reference state has no thermal stress.
    iso = ThermalHistory.isothermal(t66, T_ref_C=15.0)
    iso = iso.with_options(polymer_relaxation_factor=0.2).blocks()[0].state
    d = armour_thermal_state(t66, iso, tension_N=50e3)["inner_armour"]
    assert d.thermal_wire_stress_Pa == 0.0
    assert d.contact_pressure_Pa == d.contact_pressure_reference_Pa
    assert d.contact_pressure_Pa == pytest.approx(c.contact_pressure_reference_Pa)


def test_thermal_wire_stress_varies_with_tension_within_a_block(t66):
    tension = np.linspace(30e3, 70e3, 9)
    layers = cable_to_fatigue_layers(
        t66, contact_tension_N=50e3, contact_model="solved"
    )
    hot = ThermalHistory.uniform(t66, [0.0], 45.0)
    inputs = thermal_fatigue_inputs(t66, layers, tension.size, tension, hot)
    sigma = inputs.thermal_stress_Pa["inner_armour"]
    assert np.all(inputs.block_index == 0)
    assert np.ptp(sigma) > 0.01 * abs(float(np.mean(sigma)))


def test_ratios_are_reported_with_slope_and_relaxation_bounds(t66):
    kx, ky, tension = _record(60)
    history = ThermalHistory.steady(
        t66, RATED_A, T_ref_C=20.0, polymer_relaxation_factor=0.5
    )
    result = compute_thermal_fatigue(
        t66, kx, ky, history, tension_N=tension, relaxation_factors=(0.05,)
    )
    assert set(result.relaxation_damage_ratio) == {0.05, 0.5, 1.0}
    assert result.relaxation_damage_ratio[0.5] == result.damage_ratio_reference
    bounds = [result.relaxation_damage_ratio[f] for f in (0.05, 0.5, 1.0)]
    assert bounds == sorted(bounds) and bounds[0] < bounds[-1]
    assert result.governing_layer in result.sn_slope
    curve = get_sn_curve(t66.get_layer(result.governing_layer).bulk_sn)
    assert result.sn_slope[result.governing_layer][0] == curve.m1
    text = result.summary()
    ratio_lines = [line for line in text.splitlines() if "damage ratio" in line]
    assert len(ratio_lines) == 2 + 3
    assert all(f"m = {curve.m1:g}" in line for line in ratio_lines)
    assert "T_ref = 20 deg C (explicit" in text and "not_published" in text
    limitations = " ".join(result.provenance.limitations)
    assert "S-N slope m =" in limitations and "T_ref 20 deg C" in limitations
    assert "polymer moduli x 0.5" in limitations
    # The reference result carries the history's relaxation and reference.
    relaxed_ref = cx.compute_fatigue(
        t66,
        kx,
        ky,
        tension_N=tension,
        thermal=ThermalState(
            {x.name: 20.0 for x in t66.layers},
            T_ref_C=20.0,
            polymer_relaxation_factor=0.5,
        ),
    )
    for name, damage in relaxed_ref.D_total_per_wire.items():
        assert np.array_equal(damage, result.reference.D_total_per_wire[name])


def test_rig_test_accepts_a_relaxed_history(t66):
    proto = cx.RigProtocol.cyclic(
        tension_N=50e3, curvature_amplitude_1_m=0.01, n_cycles=1, samples_per_cycle=8
    )
    setup = cx.RigSetup(free_length_m=4.0, clamp="none")
    kw = {"gauges": [], "mu_factors": (), "contact_model": "solved"}
    hot = cx.simulate_rig_test(
        t66, proto, setup, thermal=ThermalHistory.uniform(t66, [0.0], 50.0), **kw
    )
    relaxed = cx.simulate_rig_test(
        t66,
        proto,
        setup,
        thermal=ThermalHistory.uniform(t66, [0.0], 50.0, polymer_relaxation_factor=0.1),
        **kw,
    )
    assert relaxed.layers[0].contact_pressure_Pa < hot.layers[0].contact_pressure_Pa
