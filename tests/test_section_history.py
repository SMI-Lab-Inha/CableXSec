# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Normal-state transfer and independent incremental wire-stress checks."""

import math
from dataclasses import replace

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.fatigue import FatigueLayer, _wire_stress_series
from cablexsec.cable_adapters import cable_to_fatigue_layers
from cablexsec.exceptions import CableSchemaError
from cablexsec.materials.sn_curves import SNcurve, get_sn_curve
from cablexsec.mechanics.mkappa import HelicalLayerSpec
from cablexsec.mechanics.return_mapping import CoulombElement

pytestmark = pytest.mark.analytical_sanity


def _cable():
    return cx.load_cable(cx.inter_array_single_core_yaml_path(), geometry_override=None)


def _layer():
    spec = HelicalLayerSpec(
        n_wires=4,
        r_helix_m=0.05,
        lay_angle_rad=0.4,
        E_wire_Pa=2e11,
        A_wire_m2=math.pi * 0.003**2 / 4,
        xi=0.8,
        mu=0.2,
        F_n_axi_N=1,
        contact_pressure_Pa=1e6,
    )
    return FatigueLayer(
        spec=spec,
        name="wire",
        sn_bulk=get_sn_curve("dnv_rpc203_t_curve"),
        sigma_UTS_Pa=1.5e9,
    )


def test_changing_modulus_preserves_elastic_displacement_and_slip():
    element = CoulombElement(k=100)
    element.update(0.2, 10)
    assert float(element.slip_total) == pytest.approx(0.1)
    element.set_stiffness(50)
    assert float(element.f) == pytest.approx(5)
    assert float(element.slip_total) == pytest.approx(0.1)
    element.update(0, 3)
    assert float(element.f) == pytest.approx(3)
    assert float(element.slip_total) == pytest.approx(0.14)
    element.update(-0.1, 3)
    assert float(element.f) == pytest.approx(-2)
    assert float(element.slip_total) == pytest.approx(0.14)


@pytest.mark.parametrize("invalid", [0, -1, np.nan, np.inf])
def test_invalid_stiffness_change_leaves_state_unchanged(invalid):
    element = CoulombElement(k=100)
    element.update(0.2, 10)
    with pytest.raises(ValueError, match="finite and positive"):
        element.set_stiffness(invalid)
    assert float(element.f) == 10
    assert float(element.slip_total) == pytest.approx(0.1)
    assert element.k == 100


def test_variable_modulus_wire_history_matches_independent_plastic_strain_update():
    layer = _layer()
    spec = layer.spec
    kappa = np.array([0, 0.06, 0.06, -0.04, -0.04, 0.02, 0.09])
    modulus = np.array([2, 2, 1.5, 1.5, 1.2, 1.2, 2]) * 1e11
    stiffness = (
        np.array([0.8, 0.8, 0.7, 0.7, 0.9, 0.9, 0.8]) * modulus * math.cos(0.4) ** 2
    )
    pressure = np.array([1, 1, 0.1, 0.1, 0.3, 1.2, 1]) * 1e6
    normal = np.array([0, 2, 3, 2, 1, -1, 0]) * 1e7
    calculated = _wire_stress_series(
        layer,
        kappa,
        np.zeros_like(kappa),
        None,
        None,
        None,
        pressure,
        normal_stress_Pa=normal,
        wire_modulus_Pa=modulus,
        stick_modulus_Pa=stiffness,
    )
    # Independent constitutive implementation stores plastic strain, not force.
    # At the extreme wire psi=pi/2, so the slip bounds have a direct scalar
    # form: sigma0 (exp(lam pi/2) - 1) for a rising and sigma0 (1 - exp(-lam
    # pi/2)) for a falling stress (signed capstan, [Papailiou1997] Eq. 14).
    sine, cosine = math.sin(0.4), math.cos(0.4)
    width = 2 * math.pi * spec.r_helix_m / spec.n_wires
    sigma0 = spec.r_helix_m * width * cosine / (spec.A_wire_m2 * sine**2) * pressure
    upper = sigma0 * (math.exp(spec.mu * sine * math.pi / 2) - 1)
    lower = sigma0 * (1 - math.exp(-spec.mu * sine * math.pi / 2))
    plastic = 0.0
    expected = []
    for t, value in enumerate(kappa):
        drive = spec.r_helix_m * value
        friction = np.clip(stiffness[t] * (drive - plastic), -lower[t], upper[t])
        plastic = drive - friction / stiffness[t]
        bending = modulus[t] * value * cosine**2 * math.cos(0.8) * 0.003 / 2
        expected.append(normal[t] + friction + bending)
    np.testing.assert_allclose(calculated[0], expected, rtol=1e-12, atol=1e-8)


def test_held_curvature_release_reclamp_and_reversal_retain_slip_memory():
    layer = _layer()
    kappa = np.array([0, 0.1, 0.1, 0.1, 0])
    pressure = np.array([1, 1, 0, 1, 1]) * 1e6
    stress = _wire_stress_series(
        layer,
        kappa,
        np.zeros_like(kappa),
        None,
        None,
        None,
        pressure,
    )
    # Zero normal pressure removes friction. Restoring pressure at held
    # displacement cannot restore the previous friction force without movement.
    assert stress[0, 2] == pytest.approx(stress[0, 3])
    assert stress[0, 1] > stress[0, 2]
    assert stress[0, -1] < 0


def test_pressure_torque_half_cycle_uses_solved_wire_stress_not_tension_over_ea():
    cable = _cable()
    states = [
        cx.solve_section_state(
            cable,
            tension_N=1e5,
            p_ext_Pa=pressure,
            torque_Nm=torque,
            prescribed_twist_per_m=None,
        )
        for pressure, torque in [(0, 0), (2e6, 1000)]
    ]
    result = cx.compute_fatigue(cable, np.zeros(2), np.zeros(2), section_states=states)
    for layer in cable.helical_layers:
        stress_range = abs(
            states[1].layers[layer.name].longitudinal_stress_Pa
            - states[0].layers[layer.name].longitudinal_stress_Pa
        )
        curve = get_sn_curve(layer.bulk_sn)
        expected = 0.5 / curve.cycles_to_failure(stress_range / 1e6)
        assert expected > 0
        np.testing.assert_allclose(
            result.D_bulk_per_wire[layer.name], expected, rtol=1e-12
        )
    # N is constant; the earlier N/EA stress share would incorrectly give no damage.
    assert sum(np.sum(d) for d in result.D_bulk_per_wire.values()) > 0


def test_qp_tension_api_matches_explicit_section_history():
    cable = _cable()
    tension = np.array([1e5, 2e5, 1e5, 2e5, 1e5])
    kappa = np.array([0, 0.04, 0, -0.04, 0])
    states = [cx.solve_section_state(cable, tension_N=float(n)) for n in tension]
    implicit = cx.compute_fatigue(
        cable, kappa, kappa * 0, tension_N=tension, contact_model="qp"
    )
    explicit = cx.compute_fatigue(cable, kappa, kappa * 0, section_states=states)
    for name in implicit.D_bulk_per_wire:
        np.testing.assert_array_equal(
            implicit.D_bulk_per_wire[name], explicit.D_bulk_per_wire[name]
        )


def test_constant_section_state_ramp_matches_target_wire_stress():
    cable = _cable()
    state = cx.solve_section_state(cable, tension_N=1e5, temperature_C=60, p_ext_Pa=1e6)
    kappa = np.linspace(0, 0.04, 50)
    target = cx.compute_stress(cable, kappa[-1], section_state=state)
    layers = cable_to_fatigue_layers(cable, contact_model="qp", section_state=state)
    for layer in layers:
        history = _wire_stress_series(
            layer,
            kappa,
            kappa * 0,
            None,
            None,
            None,
            None,
            normal_stress_Pa=np.full_like(
                kappa, state.layers[layer.name].longitudinal_stress_Pa
            ),
        )
        np.testing.assert_allclose(
            history[:, -1], target.sigma_total_Pa[layer.name], rtol=1e-11, atol=1e-6
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tension_N": np.zeros(2)},
        {"contact_tension_N": 0},
        {"axial_strain": np.zeros(2)},
        {"thermal_stress_Pa": {}},
        {"contact_model": "bounded"},
    ],
)
def test_section_history_rejects_conflicting_inputs(kwargs):
    cable = _cable()
    state = cx.solve_section_state(cable)
    with pytest.raises(CableSchemaError):
        cx.compute_fatigue(
            cable, np.zeros(2), np.zeros(2), section_states=[state, state], **kwargs
        )


def test_section_history_rejects_wrong_length_and_mixed_models():
    cable = _cable()
    state = cx.solve_section_state(cable)
    with pytest.raises(CableSchemaError, match="match the curvature"):
        cx.compute_fatigue(cable, np.zeros(2), np.zeros(2), section_states=[state])
    with pytest.raises(CableSchemaError, match="same contact model"):
        cx.compute_fatigue(
            cable,
            np.zeros(2),
            np.zeros(2),
            section_states=[
                state,
                cx.solve_section_state(cable, contact_model="bounded"),
            ],
        )


def test_variable_section_does_not_silently_freeze_a_contact_life_model(monkeypatch):
    cable = _cable()
    original = cable_to_fatigue_layers

    def contact_layers(*args, **kwargs):
        layers = original(*args, **kwargs)
        # Supply a non-constant synthetic Hertz stiffness to isolate the API
        # rule; this does not invent a fretting curve for a real cable layer.
        return [
            replace(
                layer,
                sn_fretting=SNcurve("synthetic-slip-only", "um", 3, 12),
                K_t_contact_N_per_m=kwargs["section_state"].tension_N,
            )
            for layer in layers
        ]

    monkeypatch.setattr("cablexsec.api.cable_to_fatigue_layers", contact_layers)
    states = [cx.solve_section_state(cable, tension_N=n) for n in (1e5, 2e5)]
    with pytest.raises(CableSchemaError, match="constant normal"):
        cx.compute_fatigue(cable, np.zeros(2), np.zeros(2), section_states=states)
