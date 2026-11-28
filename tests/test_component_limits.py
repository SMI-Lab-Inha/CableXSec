# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Limits of every cable component, their utilisation and the capacity they set.

Every limit has a source: the 0.2 % proof stress of the IEA Task 49 copper
(Janocha et al. 2024, Sec. 2.1, p. 5), the allowable strain of polyethylene
sheaths (API 17J Table 6, p. 19), the steel armour factors (DNV-RP-F401
Sec. 3.8.1, Table 3-1, p. 13) and those of the other components (OWA MecLim II
guideline, Table 5, p. 19). A component whose material gives no limit is not
given one: it says so.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from _bundled import bundled

import cablexsec as cx
from cablexsec.analysis.component_limits import (
    NO_LIMIT,
    assess,
    component_factor_for,
    component_limits,
    stress_limit,
)
from cablexsec.analysis.components import (
    KINEMATIC,
    ComponentState,
    capacity_utilisation,
    own_bending_ratio,
)
from cablexsec.analysis.kinematics import kinematic_history
from cablexsec.materials import library
from cablexsec.materials.properties import MATERIAL_PROPERTIES
from cablexsec.mechanics.core_lay import core_angle, core_bending_components

pytestmark = pytest.mark.filterwarnings("ignore")

#: Published values of the IEA Task 49 66 kV cable (NREL/TP-5000-89709,
#: Table 13, 630 mm2 row, p. 35): minimum breaking load and bend radius.
IEA_MBL_N = 659e3
IEA_MBR_M = 2.76


@pytest.fixture(scope="module")
def iea() -> cx.CableData:
    return bundled("iea_task49_66kv")


def _state(response: cx.ComponentResponse, name: str) -> ComponentState:
    return next(s for s in response.states if s.name == name)


# -- limits and their sources --------------------------------------------------


def test_every_library_limit_has_its_source() -> None:
    proof = library.get_property("copper_janocha2024", "sigma_y_Pa")
    assert proof == 120e6
    cited = library.get_citation("copper_janocha2024", "sigma_y_Pa")
    assert "[Janocha2024]" in cited and "p. 5" in cited and "0.2 %" in cited
    for material in ("hdpe", "hdpe_janocha2024", "mdpe_janocha2024", "hdpe_menard2023"):
        assert library.get_property(material, "epsilon_allow") == 0.077
        source = library.get_citation(material, "epsilon_allow")
        assert "[API17J] (2008) Table 6, p. 19" in source
        assert "applied to the PE sheaths of a power cable" in source
    # No public value: none is substituted.
    for material in ("xlpe_janocha2024", "xlpe_qiao2025", "copper_tape_screen"):
        assert component_limits(material, "insulation") == ()
    with pytest.raises(KeyError):
        library.get_property("lead_sheath", "epsilon_allow")


def test_the_strain_limit_is_a_cable_material_property() -> None:
    definition = MATERIAL_PROPERTIES["epsilon_allow"]
    assert definition.unit == "" and definition.group == "Strength"
    assert definition.problem(0.077) is None
    assert definition.problem(1.5) is not None


def test_component_factors_follow_the_lifecycle_phase() -> None:
    assert cx.COMPONENT_UTILISATION_FACTORS == {
        "normal": 0.80,
        "installation": 1.00,
        "abnormal": 1.00,
    }
    assert cx.ARMOUR_UTILISATION_FACTORS["normal"] == 0.67  # DNV-RP-F401 Table 3-1
    assert component_factor_for(0.67) == 0.80
    assert component_factor_for(0.78) == 1.00
    assert component_factor_for(1.00) == 1.00
    assert component_factor_for(0.5) == 0.5  # no phase: the armour factor


def test_assessment_names_the_governing_limit() -> None:
    limits = component_limits("copper_janocha2024", "conductor")
    assert [limit.mode for limit in limits] == ["yielding", "tensile fracture"]
    judged = assess(limits, peak_stress_Pa=60e6, net_section_Pa=20e6)
    assert judged.mode == "yielding" and judged.utilisation == pytest.approx(0.5)
    assert dict(judged.utilisations)["tensile fracture"] == pytest.approx(20 / 220)
    # A conductor with no proof stress: fracture on the net section only.
    fracture = assess(
        tuple(
            limit
            for limit in component_limits("copper_stranded", "conductor")
            if limit.mode != "yielding"
        ),
        peak_stress_Pa=300e6,
        net_section_Pa=110e6,
    )
    assert fracture.mode == "tensile fracture"
    assert fracture.utilisation == pytest.approx(0.5)
    # The generic stranded copper takes the published 0.2 % proof stress.
    assert component_limits("copper_stranded", "conductor")[0].value == 120e6
    strain = assess(
        component_limits("hdpe", "polymer protection"),
        peak_stress_Pa=1e6,
        peak_strain=0.0385,
    )
    assert strain.mode == "strain limit" and strain.utilisation == pytest.approx(0.5)
    none = assess((), peak_stress_Pa=1e9)
    assert math.isnan(none.utilisation) and none.mode == NO_LIMIT
    value, text = stress_limit(component_limits("hdpe_janocha2024", "x"), 1e9)
    assert value == pytest.approx(0.077e9) and text.startswith("allowable strain")


# -- the strand basis of a stranded conductor --------------------------------------


def test_conductor_bending_is_on_the_basis_of_its_bending_stiffness(iea) -> None:
    conductor = iea.get_layer("cu_conductor")
    ratio = own_bending_ratio(conductor)
    assert ratio == pytest.approx(math.sqrt(5.0 / 112.2))
    # A solid conductor (no effective bending modulus) bends as a solid.
    menard = bundled("menard2023")
    assert own_bending_ratio(menard.get_layer("cu_conductor")) == 1.0
    # Free core slip at zero tension: the extreme wire fibre takes
    # E kappa r_eq, r_eq = R sqrt(E_b / E), about the core's own axis.
    kappa = 0.1
    state = cx.solve_section_state(iea, tension_N=0.0, contact_model="bounded")
    response = cx.compute_component_response(
        iea, kappa, section_state=state, core_state="free", n_points=720
    )
    found = _state(response, "cu_conductor")
    beta = core_angle(iea, conductor)
    local = max(
        math.hypot(*core_bending_components(beta, phase, kappa, 0.0))
        for phase in (0.0, 2 * math.pi / 3, 4 * math.pi / 3)
    )
    e = state.youngs_modulus_Pa["cu_conductor"]
    radius = float(conductor.r_outer_m or 0.0)
    assert found.sigma_local_max_Pa == pytest.approx(
        e * radius * ratio * local, rel=1e-4
    )
    assert "strand basis" in found.basis


def test_conductor_against_its_proof_stress_and_fracture(iea) -> None:
    proof = library.get_property("copper_janocha2024", "sigma_y_Pa")
    uts = library.get_property("copper_janocha2024", "sigma_UTS_Pa")
    frictional = _state(
        cx.compute_component_response(iea, 0.1, 0.0, tension_N=80e3), "cu_conductor"
    )
    checks = dict(frictional.utilisations)
    assert frictional.failure_mode == "yielding"
    assert checks["yielding"] == pytest.approx(
        frictional.sigma_equivalent_max_Pa / proof, rel=1e-12
    )
    assert checks["tensile fracture"] == pytest.approx(
        frictional.sigma_net_section_max_Pa / uts, rel=1e-12
    )
    assert 0.3 < frictional.mechanical_utilization < 0.5
    assert "0.2 % proof stress 120 MPa ([Janocha2024]" in frictional.limits
    # Held to the cable axis the core axis takes the section's bending strain:
    # the conductor is far beyond its proof stress and says so.
    stuck = _state(
        cx.compute_component_response(
            iea, 0.1, 0.0, tension_N=80e3, core_state="stuck"
        ),
        "cu_conductor",
    )
    assert stuck.mechanical_utilization > 3.0 and stuck.failure_mode == "yielding"
    # The net-section stress at zero curvature is the tension share of the
    # conductor, 13.7 MPa at 80 kN; bending adds the friction-held core axis.
    straight = _state(
        cx.compute_component_response(iea, 0.0, tension_N=80e3), "cu_conductor"
    )
    assert straight.sigma_net_section_max_Pa == pytest.approx(13.67e6, rel=0.01)
    assert frictional.sigma_net_section_max_Pa > straight.sigma_net_section_max_Pa


def test_loose_screens_are_checked_on_their_kinematic_stress(iea) -> None:
    # A copper wire screen in place of the template's copper tape (which has
    # no yield stress in the library) is checked against its yield.
    layers = tuple(
        x.model_copy(update={"material": "copper_wire_screen"})
        if x.name == "cu_tape_screen"
        else x
        for x in iea.layers
    )
    wired = iea.model_copy(update={"layers": layers})
    response = cx.compute_component_response(wired, 0.1, 0.0, tension_N=80e3)
    wires = _state(response, "cu_tape_screen")
    assert wires.kinematic and wires.basis.startswith(KINEMATIC)
    sigma_y = library.get_property("copper_wire_screen", "sigma_y_Pa")
    assert wires.mechanical_utilization == pytest.approx(
        wires.sigma_equivalent_max_Pa / sigma_y, rel=1e-12
    )
    assert wires.mechanical_utilization > 1.0  # above its yield, flagged
    response = cx.compute_component_response(iea, 0.1, 0.0, tension_N=80e3)
    for name in ("cu_tape_screen", "conductor_screen"):
        tape = _state(response, name)
        assert tape.kinematic and tape.failure_mode == NO_LIMIT
    # Outside the load path: an upper bound, never the critical component.
    assert response.critical_component == "cu_conductor"
    sheath = _state(response, "hdpe_outer_sheath")
    e = library.get_property("hdpe_janocha2024", "E_Pa")
    assert sheath.failure_mode == "strain limit"
    assert sheath.mechanical_utilization == pytest.approx(
        sheath.sigma_equivalent_max_Pa / e / 0.077, rel=1e-12
    )


def test_metallic_screens_and_sheaths_bound_the_capacity_by_strain() -> None:
    def state(utilisations: tuple[tuple[str, float], ...]) -> ComponentState:
        return ComponentState(
            name="al_sheath",
            component_type="metallic sheath",
            material="aluminium_corrugated",
            strain_min=0.0,
            strain_max=0.0,
            sigma_local_min_Pa=0.0,
            sigma_local_max_Pa=0.0,
            sigma_stick_min_Pa=0.0,
            sigma_stick_max_Pa=0.0,
            sigma_equivalent_max_Pa=0.0,
            temperature_C=20.0,
            mechanical_utilization=max(v for _, v in utilisations),
            failure_mode="yielding",
            basis="",
            utilisations=utilisations,
        )

    # First yield of a sheath made to deform plastically does not bound the
    # capacity; its allowable strain does.
    value, mode = capacity_utilisation(state((("yielding", 3.0),)))
    assert math.isnan(value) and mode == NO_LIMIT
    value, mode = capacity_utilisation(
        state((("yielding", 3.0), ("strain limit", 0.4))), 0.8
    )
    assert mode == "strain limit" and value == pytest.approx(0.5)


# -- the capacity envelope and the minimum bend radius -------------------------


def test_iea_envelope_against_the_published_mbl_and_mbr(iea) -> None:
    grid = np.linspace(0.0, 1.0e6, 41)
    envelope = cx.build_capacity_envelope(iea, grid, utilisation_factor=1.0)
    assert envelope.component_utilisation_factor == 1.0
    assert envelope.governing[0] == "component_limit"
    assert envelope.component_governing[0] == ("cu_conductor", "yielding")
    assert envelope.kinematic_layers == ("conductor_screen", "cu_tape_screen")
    fine = np.linspace(0.0, 1.0e6, 4001)
    at_zero = envelope.utilisation(fine, np.zeros(fine.size))[0]
    tension_capacity = float(fine[np.flatnonzero(at_zero >= 1.0)[0]])
    # Characteristic tension capacity, set by the conductor's 0.2 % proof
    # stress: 6.6 % above the published minimum breaking load, against about
    # 1.23 MN for armour yield alone.
    assert tension_capacity / IEA_MBL_N == pytest.approx(1.066, abs=0.02)
    radius = 1.0 / float(envelope.curvature_capacity(np.zeros(1))[0])
    # Bent towards a core the conductor limit gives the published bend radius;
    # the envelope holds for every bending direction, and 30 degrees from a
    # core (halfway to the axis between two) the most loaded conductor reaches
    # its proof stress at a 2.7 % larger radius.
    toward = cx.component_usage_curve(iea, 0.0, 1.0, directions=(0.0,))
    assert 1.0 / toward.root(1.0) == pytest.approx(IEA_MBR_M, rel=0.005)
    assert radius == pytest.approx(1.027 * IEA_MBR_M, rel=0.01)
    assert radius > 1.0 / toward.root(1.0)
    armour = cx.build_capacity_envelope(
        iea, grid, utilisation_factor=1.0, component_limits=False
    )
    assert not armour.has_components
    assert float(armour.curvature_capacity(np.zeros(1))[0]) > 1.1  # armour only
    # Normal operation: 0.67 on the armour, 0.80 on the conductor.
    normal = cx.build_capacity_envelope(iea, grid)
    assert normal.component_utilisation_factor == 0.80
    assert 1.0 / float(normal.curvature_capacity(np.zeros(1))[0]) == pytest.approx(
        radius / 0.8, rel=0.01
    )


def test_governing_switches_on_the_bundled_cables() -> None:
    grid = np.array([0.0, 50e3])
    tb880 = bundled("poradowski2025_220kv")
    cover = cx.build_capacity_envelope(tb880, grid, utilisation_factor=1.0, mu=0.2)
    # The generic stranded copper takes its published 0.2 % proof stress: the
    # conductor now limits the cable before the outer cover reaches 7.7 % at
    # its outer radius (167.55 mm).
    assert cover.component_governing[0] == ("cu_conductor", "yielding")
    assert float(cover.kappa_capacity_1_m[0]) < 0.077 / 0.16755
    menard = bundled("menard2023")
    switched = cx.build_capacity_envelope(menard, grid, utilisation_factor=1.0)
    assert switched.governing == ("component_limit", "wire_yield")
    # Without limits in the materials the armour results are unchanged.
    chae = bundled("chae2023_coiling")
    same = cx.build_capacity_envelope(chae, grid)
    plain = cx.build_capacity_envelope(chae, grid, component_limits=False)
    np.testing.assert_array_equal(same.kappa_capacity_1_m, plain.kappa_capacity_1_m)
    assert not same.has_components
    # The single core: its generic copper has the published 0.2 % proof
    # stress (since 24 Oct 2026), which now limits the cable before the armour.
    single = bundled("inter_array_single_core_33kv")
    governed = cx.compute_capacity(single, grid)
    armour = cx.compute_capacity(single, grid, component_limits=False)
    assert governed.governing == ["component_limit", "component_limit"]
    assert governed.governing_component[0] == ("cu_conductor", "yielding")
    assert np.all(governed.kappa_max_1_m < armour.kappa_max_1_m)


def test_compute_capacity_names_the_governing_component(iea) -> None:
    capacity = cx.compute_capacity(iea, np.array([0.0, 300e3]))
    assert capacity.governing == ["component_limit", "component_limit"]
    assert capacity.governing_component[0] == ("cu_conductor", "yielding")
    # The worst bending direction (30 degrees from a core): 2.7 % above the
    # published bend radius, which bending towards a core gives.
    assert 1.0 / capacity.kappa_max_1_m[0] == pytest.approx(1.027 * IEA_MBR_M, rel=0.01)
    usage = cx.component_usage_curve(iea, 0.0, 1.0)
    assert usage.root(1.0) == pytest.approx(capacity.kappa_component_1_m[0])


def test_coiling_checks_the_components(iea) -> None:
    coil = cx.compute_coiling(iea, tank_diameter_m=4.0)
    value, mode = coil.components["cu_conductor"]
    assert mode == "yielding" and value > 1.0  # a 2 m radius yields the copper
    assert coil.utilisation == pytest.approx(
        max(value, *(layer.utilisation for layer in coil.layers.values()))
    )
    assert coil.governing in ("cu_conductor", *coil.layers)
    # A loose metal screen: outside the load path, and its first yield does not
    # bound the capacity.
    assert "cu_wire_screen" not in coil.components


# -- the application -------------------------------------------------------------


def test_cores_carry_their_utilisation_in_the_motion(iea) -> None:
    samples = 41
    ramp = np.linspace(0.0, 0.1, samples)
    history = kinematic_history(
        iea, ramp, np.zeros(samples), tension_N=80e3, core_friction=True
    )
    cores = history.cores
    assert cores is not None and cores.utilisation is not None
    assert cores.utilisation.shape == (cores.stations, samples)
    direct = _state(
        cx.compute_component_response(iea, 0.1, 0.0, tension_N=80e3), "cu_conductor"
    )
    # At the azimuths of the three cores of the section the component response
    # assesses; along the pitch every core passes every azimuth.
    phases = np.array([0.0, 2 * math.pi / 3, 4 * math.pi / 3])
    at_cores = cores.along(cores.utilisation[:, -1:], phases)
    assert float(np.max(at_cores)) == pytest.approx(
        direct.mechanical_utilization, rel=0.01
    )
    assert float(cores.utilisation[:, -1].max()) >= float(np.max(at_cores))
