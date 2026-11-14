# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Analytical gates for the all-component response."""

from __future__ import annotations

import math

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.cables.schema import CableData, CableMeta, CylindricalLayer
from cablexsec.materials import library
from cablexsec.mechanics.stiffness import ea_fixed
from cablexsec.mechanics.thermal import ThermalResult

pytestmark = pytest.mark.analytical_sanity


def _uniform_polymer_cable() -> CableData:
    return CableData(
        cable=CableMeta(name="uniform thermal gate"),
        layers=(
            CylindricalLayer(
                name="inner",
                material="hdpe",
                r_inner_m=0.0,
                r_outer_m=0.01,
            ),
            CylindricalLayer(
                name="outer",
                material="hdpe",
                r_inner_m=0.01,
                r_outer_m=0.02,
            ),
        ),
    )


def test_public_component_surface_is_exported() -> None:
    assert "compute_component_response" in cx.__all__
    assert "ComponentResponse" in cx.__all__
    assert "ComponentState" in cx.__all__


def test_uniform_free_thermal_expansion_has_zero_stress() -> None:
    cable = _uniform_polymer_cable()
    thermal = ThermalResult(
        r_m=np.array([0.0, 0.02]),
        T_C=np.array([60.0, 60.0]),
        T_conductor_C=60.0,
        T_armour_C=60.0,
        P_total_W_m=0.0,
    )
    result = cx.compute_component_response(cable, 0.0, thermal=thermal, T_ref_C=20.0)
    expected = library.get_property("hdpe", "alpha_T_K") * 40.0
    assert result.axial_strain == pytest.approx(expected)
    assert result.free_thermal_axial_strain == pytest.approx(expected)
    for state in result.states:
        assert state.sigma_stick_min_Pa == pytest.approx(0.0, abs=1e-6)
        assert state.sigma_stick_max_Pa == pytest.approx(0.0, abs=1e-6)


def test_mechanical_axial_strain_recovers_ea_relation() -> None:
    cable = _uniform_polymer_cable()
    tension = 125_000.0
    result = cx.compute_component_response(cable, 0.0, tension_N=tension)
    assert result.mechanical_axial_strain == pytest.approx(tension / ea_fixed(cable))
    assert result.axial_strain == pytest.approx(result.mechanical_axial_strain)


def test_offset_core_reports_local_and_compatible_bounds() -> None:
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    result = cx.compute_component_response(cable, 0.02, tension_N=100_000.0)
    conductor = next(state for state in result.states if state.name == "cu_conductor")
    assert conductor.sigma_stick_max_Pa > conductor.sigma_local_max_Pa
    assert conductor.sigma_stick_min_Pa < conductor.sigma_local_min_Pa
    # The source gives no copper strength: no limit, and no substitute.
    assert conductor.failure_mode == "no limit given"
    assert math.isnan(conductor.mechanical_utilization)
    assert conductor.limits == "no limit given"


def test_single_core_zero_offset_bounds_coincide() -> None:
    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    result = cx.compute_component_response(cable, 0.01, tension_N=80_000.0)
    conductor = next(state for state in result.states if state.name == "cu_conductor")
    assert conductor.sigma_stick_min_Pa == pytest.approx(conductor.sigma_local_min_Pa)
    assert conductor.sigma_stick_max_Pa == pytest.approx(conductor.sigma_local_max_Pa)
    assert conductor.failure_mode == "yielding"  # its 0.2 % proof stress
    assert math.isfinite(conductor.mechanical_utilization)


@pytest.mark.parametrize(
    "cable_path", [cx.menard2023_yaml_path(), cx.inter_array_single_core_yaml_path()]
)
def test_component_descriptions_identify_kinematic_assumptions(cable_path) -> None:
    cable = cx.load_cable(cable_path, geometry_override=None)
    result = cx.compute_component_response(cable, 0.02, tension_N=100_000.0)
    conductor = next(
        state for state in result.states if state.component_type == "conductor"
    )
    assert "kinematic assumptions" in conductor.basis
    assert all("bounds" not in state.basis for state in result.states)
    assert result.provenance is not None
    scope = " ".join(result.provenance.limitations)
    assert "two prescribed kinematic assumptions" in scope
    assert "bounds" not in scope


def test_every_layer_is_reported_and_missing_limits_remain_explicit() -> None:
    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    result = cx.compute_component_response(cable, 0.005, tension_N=80_000.0)
    assert {state.name for state in result.states} == {
        layer.name for layer in cable.layers
    }
    insulation = next(
        state for state in result.states if state.name == "xlpe_insulation"
    )
    assert insulation.component_type == "insulation"
    assert insulation.failure_mode == "no limit given"
    assert math.isnan(insulation.mechanical_utilization)
    assert result.provenance is not None
    assert "calibration-free" in result.provenance.method


def test_rejects_insufficient_circumferential_resolution() -> None:
    with pytest.raises(ValueError, match="at least 8"):
        cx.compute_component_response(_uniform_polymer_cable(), 0.0, n_points=4)


def _armour_states(result):
    return [state for state in result.states if state.component_type == "armour wire"]


def test_armour_utilisation_is_first_yield_and_contact_screen_is_separate() -> None:
    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    result = cx.compute_component_response(cable, 0.0)
    armour = _armour_states(result)
    assert armour
    for state in armour:
        sigma_y = library.get_property(state.material, "sigma_y_Pa")
        # No tension, no curvature: no longitudinal wire stress, so no
        # first-yield usage, while crossing contact keeps the screen large.
        assert state.mechanical_utilization == pytest.approx(0.0, abs=1e-9)
        assert state.contact_screen_utilization > 1.0
        assert state.contact_screen_utilization == pytest.approx(
            state.sigma_equivalent_max_Pa / sigma_y, rel=1e-12
        )
    for state in result.states:
        if state.component_type != "armour wire":
            assert math.isnan(state.contact_screen_utilization)


@pytest.mark.parametrize("contact_model", ["bounded", "qp"])
@pytest.mark.parametrize("tension", [0.0, 50_000.0, 150_000.0])
def test_armour_utilisation_is_one_at_capacity_first_yield(
    contact_model: str, tension: float
) -> None:
    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    # The armour alone: the conductor's proof stress may govern the cable.
    capacity = cx.compute_capacity(
        cable,
        np.array([tension]),
        contact_model=contact_model,
        component_limits=False,
    )
    assert capacity.governing == ["wire_yield"]
    kappa_y = float(capacity.kappa_max_1_m[0])
    result = cx.compute_component_response(
        cable, kappa_y, tension_N=tension, contact_model=contact_model
    )
    peak = max(state.mechanical_utilization for state in _armour_states(result))
    # Same stress basis; the residual is the section-state axial stress used
    # here against the frozen load share in compute_capacity (< 0.05 %).
    assert peak == pytest.approx(1.0, abs=2e-3)
    below = cx.compute_component_response(
        cable, 0.9 * kappa_y, tension_N=tension, contact_model=contact_model
    )
    assert max(s.mechanical_utilization for s in _armour_states(below)) < 1.0


def test_non_structural_helical_layer_is_reported_not_assessed() -> None:
    """A non-structural helical layer (Saevik 2011 pressure armour) has no
    wire stress from compute_stress; it is listed as not assessed instead of
    failing the whole response."""
    cable = cx.load_cable(cx.saevik2011_yaml_path())
    response = cx.compute_component_response(cable, 0.1, 0.0, tension_N=5.0e4)
    state = {s.name: s for s in response.states}["pressure_armour"]
    assert state.failure_mode == "not assessed"
    assert math.isnan(state.mechanical_utilization)
    assert state.sigma_equivalent_max_Pa == 0.0
    armours = [
        s
        for s in response.states
        if s.component_type == state.component_type and s is not state
    ]
    assert armours and all(s.sigma_equivalent_max_Pa > 0.0 for s in armours)


def _conductor(result):
    return next(state for state in result.states if state.name == "cu_conductor")


@pytest.fixture(scope="module")
def iea():
    return cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)


def test_laid_cores_are_assessed_in_the_frictional_state(iea) -> None:
    result = cx.compute_component_response(iea, 0.05, tension_N=50e3)
    conductor = _conductor(result)
    assert result.core_state == conductor.core_state == "frictional"
    uts = library.get_property(conductor.material, "sigma_UTS_Pa")
    proof = library.get_property(conductor.material, "sigma_y_Pa")
    checks = dict(conductor.utilisations)
    # Net-section fracture check: axial force over the conductor area.
    assert checks["tensile fracture"] == pytest.approx(
        conductor.sigma_net_section_max_Pa / uts, rel=1e-12
    )
    # 0.2 % proof stress at the extreme wire fibre, the governing limit.
    assert checks["yielding"] == pytest.approx(
        conductor.sigma_equivalent_max_Pa / proof, rel=1e-12
    )
    assert conductor.failure_mode == "yielding"
    assert conductor.mechanical_utilization == max(checks.values())
    assert conductor.mechanical_utilization < 0.3  # a 20 m bend does not yield Cu
    assert conductor.sigma_equivalent_max_Pa == pytest.approx(
        max(abs(conductor.sigma_frictional_min_Pa), conductor.sigma_frictional_max_Pa)
    )
    by_state = {
        state: _conductor(
            cx.compute_component_response(iea, 0.05, tension_N=50e3, core_state=state)
        )
        for state in ("free", "frictional", "stuck")
    }
    net = {k: v.sigma_net_section_max_Pa for k, v in by_state.items()}
    assert net["free"] < net["frictional"] < net["stuck"]
    assert by_state["stuck"].mechanical_utilization > 1.0  # the stuck assumption
    for state in by_state.values():
        assert state.sigma_local_max_Pa == conductor.sigma_local_max_Pa
        assert state.sigma_stick_max_Pa == conductor.sigma_stick_max_Pa
        assert state.sigma_frictional_max_Pa == conductor.sigma_frictional_max_Pa
    with pytest.raises(ValueError, match="core_state"):
        cx.compute_component_response(iea, 0.05, core_state="glued")  # type: ignore[arg-type]


def test_frictional_core_axis_follows_the_core_friction_law(iea) -> None:
    """Below slip the core axis takes xi times the stuck strain; above slip it
    keeps the friction cap, so the net-section stress stops growing."""
    from cablexsec.mechanics.core_friction import CoreFrictionSpec, build_core_friction

    state = cx.solve_section_state(iea, tension_N=50e3, contact_model="bounded")
    model = build_core_friction(
        iea, CoreFrictionSpec(strand_slip=False), 50e3, section_state=state
    )
    mean = state.layers["cu_conductor"].longitudinal_stress_Pa
    small = {
        s: _conductor(
            cx.compute_component_response(iea, 1e-6, section_state=state, core_state=s)
        ).sigma_net_section_max_Pa
        - mean
        for s in ("frictional", "stuck")
    }
    assert small["frictional"] == pytest.approx(model.xi * small["stuck"], rel=1e-9)
    capped = [
        _conductor(
            cx.compute_component_response(iea, k, section_state=state)
        ).sigma_net_section_max_Pa
        for k in (0.01, 0.05, 0.3)
    ]
    assert capped[0] == pytest.approx(capped[1], rel=1e-12)
    assert capped[1] == pytest.approx(capped[2], rel=1e-12)
    # A larger friction coefficient raises the frictional core stress.
    rough = _conductor(
        cx.compute_component_response(
            iea,
            0.05,
            section_state=state,
            core_friction=CoreFrictionSpec(0.46, 0.46, 0.46),
        )
    )
    assert rough.sigma_net_section_max_Pa > capped[1]


def test_aligned_cores_are_stuck_and_say_so() -> None:
    cable = cx.load_cable(cx.export_three_core_yaml_path(), geometry_override=None)
    cable = cable.model_copy(update={"core_lay": None})
    result = cx.compute_component_response(cable, 0.05, tension_N=50e3)
    conductor = _conductor(result)
    assert conductor.core_state == "stuck"
    assert conductor.sigma_frictional_max_Pa == conductor.sigma_stick_max_Pa
    assert "the cores are not laid" in conductor.basis
    sheath = next(s for s in result.states if s.name == "al_metallic_sheath")
    assert "does not relieve" in sheath.basis
    lead = next(s for s in result.states if s.name == "lead_core_sheath")
    assert lead.component_type == "metallic sheath"


def test_conductor_net_section_matches_the_capacity_conductor_check() -> None:
    from cablexsec.analysis.capacity import _conductor_uf

    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    for kappa in (0.0, 0.05, 0.3):
        conductor = _conductor(
            cx.compute_component_response(cable, kappa, tension_N=50e3)
        )
        assert conductor.core_state == ""
        assert dict(conductor.utilisations)["tensile fracture"] == pytest.approx(
            _conductor_uf(cable, 50e3), rel=1e-9
        )


def test_armour_friction_coefficient_reaches_the_wire_stress() -> None:
    cable = cx.load_cable(cx.witz1996_yaml_path(), geometry_override=None)
    with pytest.raises(cx.CableLookupError, match="mu_on"):
        cx.compute_component_response(cable, 0.05, tension_N=1e4)
    result = cx.compute_component_response(cable, 0.05, tension_N=1e4, mu=0.1)
    stress = cx.compute_stress(cable, 0.05, tension_N=1e4, mu=0.1)
    for layer in cable.helical_layers:
        state = next(s for s in result.states if s.name == layer.name)
        assert state.sigma_local_max_Pa == pytest.approx(
            float(np.max(stress.sigma_perimeter_max_Pa[layer.name])), rel=1e-9
        )
    with pytest.raises(ValueError, match="mu"):
        cx.compute_component_response(cable, 0.05, mu=-0.1)


def test_basis_keeps_a_deprecated_alias() -> None:
    result = cx.compute_component_response(_uniform_polymer_cable(), 0.01)
    state = result.states[0]
    with pytest.deprecated_call():
        assert state.evidence == state.basis
    assert state.component_type == "continuous layer"
    assert "_" not in state.basis
