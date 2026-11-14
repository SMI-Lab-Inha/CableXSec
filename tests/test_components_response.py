# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Component response on paths the bundled cables do not take by default.

A smooth metallic sheath in the coupled radial section bends about the cable
axis by its stress concentration factor over the plane-section stress; a
friction field carried over from a load history bounds the laid cores
between the free and the stuck state; a resolved wire ring outside the load
path follows the strain of the section through its own and the core lay; the
inputs that cannot go together are refused. Public templates and generic
geometry only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from _bundled import bundled

import cablexsec as cx
from cablexsec.analysis.stress import FrictionField
from cablexsec.cables.schema import (
    CableData,
    CylindricalOffsetLayer,
    CylindricalShellLayer,
)
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.core_lay import core_angle

pytestmark = pytest.mark.filterwarnings("ignore")

SHEATH_MID_M = 0.02012
SHEATH_T_M = 0.0007
SHEATH_SCF = 1.5


@pytest.fixture(scope="module")
def sheathed() -> CableData:
    """The single-core template with a smooth metallic sheath for its screen."""
    base = cx.load_cable(cx.inter_array_single_core_yaml_path(), geometry_override=None)
    sheath = CylindricalShellLayer(
        name="metal_sheath",
        material="copper_tape_screen",
        r_mid_m=SHEATH_MID_M,
        t_m=SHEATH_T_M,
        scf=SHEATH_SCF,
    )
    layers = tuple(
        sheath if layer.name == "cu_screen" else layer for layer in base.layers
    )
    return base.model_copy(update={"layers": layers})


@pytest.fixture(scope="module")
def iea() -> CableData:
    return bundled("iea_task49_66kv")


def _named(states, name):
    return next(item for item in states if item.name == name)


# -- the coupled radial section ----------------------------------------------------


def test_smooth_sheath_bends_by_its_concentration_factor(sheathed) -> None:
    kappa = 0.05
    response = cx.compute_component_response(
        sheathed, kappa, tension_N=1e5, contact_model="qp", n_points=36
    )
    state = response.section_state
    assert state is not None and state.radial is not None
    sheath = _named(response.states, "metal_sheath")
    normal = state.layers["metal_sheath"].longitudinal_stress_Pa
    modulus = state.youngs_modulus_Pa["metal_sheath"]
    outer = SHEATH_MID_M + SHEATH_T_M / 2
    # Symmetric about the coupled normal stress, by SCF E kappa r at the outer
    # fibre of the wall.
    mean = 0.5 * (sheath.sigma_local_min_Pa + sheath.sigma_local_max_Pa)
    half = 0.5 * (sheath.sigma_local_max_Pa - sheath.sigma_local_min_Pa)
    assert mean == pytest.approx(normal, rel=1e-9, abs=1e-3)
    assert half == pytest.approx(SHEATH_SCF * modulus * kappa * outer, rel=1e-12)
    assert sheath.strain_max - sheath.strain_min == pytest.approx(2 * kappa * outer)
    # A sheath has no net-section check, and the material gives no limit.
    assert math.isnan(sheath.sigma_net_section_max_Pa)
    assert sheath.component_type == "metallic sheath"
    assert math.isnan(sheath.mechanical_utilization)
    assert sheath.basis.startswith("coupled normal section stress")


def test_sheath_field_through_the_wall(sheathed) -> None:
    kappa = 0.05
    field = cx.compute_section_stress(
        sheathed, kappa, tension_N=1e5, contact_model="qp", n_points=36, n_radii=4
    )
    direct = cx.compute_component_response(
        sheathed, kappa, tension_N=1e5, contact_model="qp", n_points=36
    )
    layer = field.layer("metal_sheath")
    state = _named(direct.states, "metal_sheath")
    assert layer.kind == "annulus"
    np.testing.assert_allclose(
        layer.radii_m,
        np.linspace(SHEATH_MID_M - SHEATH_T_M / 2, SHEATH_MID_M + SHEATH_T_M / 2, 4),
    )
    assert float(np.max(layer.sigma_max_Pa)) == pytest.approx(
        state.sigma_stick_max_Pa, rel=1e-12
    )
    assert float(np.min(layer.sigma_min_Pa)) == pytest.approx(
        state.sigma_stick_min_Pa, rel=1e-12
    )
    # One radius: the outer fibre alone, where the extremes are.
    outer = cx.compute_section_stress(
        sheathed, kappa, tension_N=1e5, contact_model="qp", n_points=36, n_radii=1
    ).layer("metal_sheath")
    np.testing.assert_allclose(outer.radii_m, [SHEATH_MID_M + SHEATH_T_M / 2])
    assert float(np.max(outer.sigma_Pa)) == pytest.approx(
        float(np.max(layer.sigma_Pa)), rel=1e-12
    )


def test_field_inputs_are_checked(sheathed) -> None:
    with pytest.raises(CableInputError, match="rotation_rad must be finite"):
        cx.compute_section_stress(sheathed, 0.01, rotation_rad=math.nan)
    field = cx.compute_section_stress(
        sheathed, 0.01, tension_N=1e4, contact_model="qp", n_points=12
    )
    with pytest.raises(CableInputError, match="no layer named 'absent'"):
        field.layer("absent")


def test_a_section_state_excludes_separate_section_loads(sheathed) -> None:
    state = cx.solve_section_state(sheathed, tension_N=1e4, contact_model="qp")
    for loads in (
        {"tension_N": 1.0},
        {"p_ext_Pa": 1e5},
        {"torque_Nm": 1.0},
        {"contact_model": "bounded"},
    ):
        with pytest.raises(CableInputError, match="not both"):
            cx.compute_component_response(sheathed, 0.01, section_state=state, **loads)
    # The model the state was solved with may be named again.
    same = cx.compute_component_response(
        sheathed, 0.01, section_state=state, contact_model="qp"
    )
    assert same.tension_N == pytest.approx(1e4)


# -- the laid cores and a friction field -------------------------------------------


def test_core_friction_field_spans_the_free_and_the_stuck_state(iea) -> None:
    kappa = 0.05
    state = cx.solve_section_state(iea, tension_N=50e3, contact_model="bounded")
    layer = next(item for item in iea.layers if item.name == "cu_conductor")
    phases = 2.0 * np.pi * np.arange(layer.n_cores) / layer.n_cores
    stuck_strain = (
        float(layer.r_offset_m or 0.0)
        * math.cos(core_angle(iea, layer)) ** 2
        * kappa
        * np.cos(phases)
    )

    def net(**options) -> float:
        response = cx.compute_component_response(
            iea, kappa, section_state=state, **options
        )
        return _named(response.states, "cu_conductor").sigma_net_section_max_Pa

    free, stuck = net(core_state="free"), net(core_state="stuck")
    # No friction left on the cores: they slide freely.
    released = FrictionField(cores=(phases, np.zeros(layer.n_cores)))
    assert net(friction=released) == pytest.approx(free, rel=1e-12)
    # Friction that holds the core axes on the cable's plane sections.
    held = FrictionField(cores=(phases, stuck_strain))
    assert net(friction=held) == pytest.approx(stuck, rel=1e-9)
    # Halfway between: the net section is halfway, as it is linear in the strain.
    half = FrictionField(cores=(phases, 0.5 * stuck_strain))
    assert net(friction=half) == pytest.approx(0.5 * (free + stuck), rel=1e-9)


# -- a resolved wire ring outside the load path ----------------------------------------


def test_loose_resolved_wires_follow_the_section_strain() -> None:
    thies = bundled("thies2016_30kv")
    ring_layer = next(
        layer
        for layer in thies.layers
        if isinstance(layer, CylindricalOffsetLayer) and layer.strand_layers
    )
    loose = thies.model_copy(
        update={
            "layers": tuple(
                layer.model_copy(update={"structural": False})
                if layer.name == ring_layer.name
                else layer
                for layer in thies.layers
            )
        }
    )
    response = cx.compute_component_response(loose, 0.0, tension_N=5e4)
    state = response.section_state
    assert state is not None
    item = _named(response.states, ring_layer.name)
    assert item.kinematic and item.basis.startswith("kinematic")
    modulus = state.youngs_modulus_Pa[ring_layer.name]
    assembly = math.cos(core_angle(loose, ring_layer))
    expected = [
        modulus * (math.cos(ring.lay_angle_rad) * assembly) ** 2 * state.axial_strain
        for ring in ring_layer.strand_layers
    ]
    # Straight and at the reference temperature: each ring at its own strain.
    assert item.sigma_stick_min_Pa == pytest.approx(min(expected), rel=1e-9)
    assert item.sigma_stick_max_Pa == pytest.approx(max(expected), rel=1e-9)


# -- the usage curve of a cable --------------------------------------------------------


def test_usage_curve_inputs_are_checked() -> None:
    menard = bundled("menard2023")
    for kwargs, match in (
        ({"kappa_top_1_m": 0.0}, "kappa_top_1_m"),
        ({"kappa_top_1_m": math.inf}, "kappa_top_1_m"),
        ({"n_kappa": 2}, "n_kappa"),
        ({"directions": ()}, "bending direction"),
        ({"directions": (0.0, math.nan)}, "bending direction"),
    ):
        options = {"kappa_top_1_m": 0.5, **kwargs}
        with pytest.raises(CableInputError, match=match):
            cx.component_usage_curve(menard, 0.0, **options)


def test_loose_metal_screens_do_not_bound_the_usage(iea) -> None:
    plain = cx.component_usage_curve(iea, 0.0, 0.5, n_kappa=3, directions=(0.0,))
    loose = cx.component_usage_curve(
        iea, 0.0, 0.5, n_kappa=3, directions=(0.0,), include_kinematic=True
    )
    assert (
        plain.kinematic == loose.kinematic == ("cu_conductor_screen", "cu_wire_screen")
    )
    # The screens are metal without a strain limit: their first yield does not
    # bound the capacity, so including them changes nothing.
    assert loose.names == plain.names
    np.testing.assert_array_equal(loose.usage, plain.usage)


def test_a_core_layer_without_an_axial_modulus_adds_no_utilisation(iea) -> None:
    from cablexsec.analysis.components import core_component_utilisation
    from cablexsec.materials import library

    phi = 2.0 * np.pi * np.arange(3) / 3
    samples = np.linspace(0.0, 0.1, 5)
    args = (phi, samples, np.zeros(5), np.full(5, 1e-4), np.zeros((3, 5)))
    # The copper with its proof stress but without a modulus: no stress, so no
    # utilisation follows; it is as if the conductor were outside the load path.
    entry = {
        key: value
        for key, value in library.get_material("copper_janocha2024").items()
        if not key.startswith("E_")
    }
    overlay = library.MaterialOverlay({"copper_without_modulus": entry})
    unknown = iea.model_copy(
        update={
            "layers": tuple(
                layer.model_copy(update={"material": "copper_without_modulus"})
                if layer.name == "cu_conductor"
                else layer
                for layer in iea.layers
            )
        }
    )
    loose = iea.model_copy(
        update={
            "layers": tuple(
                layer.model_copy(update={"structural": False})
                if layer.name == "cu_conductor"
                else layer
                for layer in iea.layers
            )
        }
    )
    with library.material_scope(overlay):
        without = core_component_utilisation(unknown, *args)
    reference = core_component_utilisation(loose, *args)
    full = core_component_utilisation(iea, *args)
    assert without is not None and reference is not None and full is not None
    np.testing.assert_array_equal(without, reference)
    # The conductor governs where it is assessed.
    assert float(full.max()) > float(reference.max())
