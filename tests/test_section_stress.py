# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""The section stress field reproduces the component extremes and utilisations."""

from __future__ import annotations

import math

import numpy as np
import pytest
from _bundled import bundled

import cablexsec as cx
from cablexsec.analysis.components import own_bending_ratio
from cablexsec.analysis.section_stress import KINEMATIC

pytestmark = pytest.mark.filterwarnings("ignore")

_CABLES = (
    "menard2023",
    "inter_array_three_core_33kv",
    "iea_task49_66kv",
    "thies2016_30kv",
    "poradowski2025_220kv",
    "saevik2011",
)


def _cable(name: str) -> cx.CableData:
    return bundled(name)


def _range(state: cx.ComponentState, key: str) -> tuple[float, float]:
    if key == "frictional":
        return state.sigma_frictional_min_Pa, state.sigma_frictional_max_Pa
    if key == "free":
        return state.sigma_local_min_Pa, state.sigma_local_max_Pa
    return state.sigma_stick_min_Pa, state.sigma_stick_max_Pa


@pytest.mark.parametrize("name", _CABLES)
@pytest.mark.parametrize("core_state", ["frictional", "stuck", "free"])
@pytest.mark.parametrize("curvature", [(0.05, 0.0), (0.02, -0.03)])
def test_field_extremes_equal_the_component_extremes(name, core_state, curvature):
    cable = _cable(name)
    field = cx.compute_section_stress(
        cable, *curvature, tension_N=40e3, core_state=core_state, n_points=72
    )
    direct = cx.compute_component_response(
        cable, *curvature, tension_N=40e3, core_state=core_state, n_points=72
    )
    states = {state.name: state for state in direct.states}
    assert [layer.name for layer in field.layers] == list(states)
    for layer in field.layers:
        state = states[layer.name]
        if not layer.has_values:
            assert state.sigma_stick_min_Pa == state.sigma_stick_max_Pa == 0.0
            assert layer.basis
            assert layer.sigma_Pa.size == 0
            continue
        # Outside the load path: the component response carries the same
        # kinematic stress, marked so.
        assert state.kinematic == layer.kinematic
        if layer.kinematic:
            assert layer.basis.startswith(KINEMATIC)
            assert state.basis.startswith(KINEMATIC)
        low, high = _range(state, layer.core_state or "stuck")
        assert float(np.min(layer.sigma_min_Pa)) == pytest.approx(low, rel=1e-12)
        assert float(np.max(layer.sigma_max_Pa)) == pytest.approx(high, rel=1e-12)
        # The centre-line values lie within the fibre range.
        assert np.all(layer.sigma_min_Pa <= layer.sigma_Pa + 1e-6)
        assert np.all(layer.sigma_Pa <= layer.sigma_max_Pa + 1e-6)
        usage = layer.utilisation
        if math.isfinite(state.mechanical_utilization):
            assert float(np.nanmax(usage)) == pytest.approx(
                state.mechanical_utilization, rel=1e-12
            )
        else:
            assert np.all(np.isnan(usage))


def test_field_reproduces_the_coupled_radial_section():
    cable = _cable("inter_array_single_core_33kv")
    field = cx.compute_section_stress(
        cable, 0.04, tension_N=1e5, contact_model="qp", n_points=36
    )
    direct = cx.compute_component_response(
        cable, 0.04, tension_N=1e5, contact_model="qp", n_points=36
    )
    assert direct.section_state is not None and direct.section_state.radial
    for layer, state in zip(field.layers, direct.states, strict=True):
        if not layer.has_values:
            continue
        assert float(np.min(layer.sigma_min_Pa)) == pytest.approx(
            state.sigma_stick_min_Pa, rel=1e-12
        )
        assert float(np.max(layer.sigma_max_Pa)) == pytest.approx(
            state.sigma_stick_max_Pa, rel=1e-12
        )


def test_points_lie_on_the_section_and_turn_with_the_frame():
    cable = _cable("inter_array_three_core_33kv")
    plain = cx.compute_section_stress(cable, 0.05, tension_N=5e4, n_points=36)
    turned = cx.compute_section_stress(
        cable, 0.05, tension_N=5e4, n_points=36, rotation_rad=math.pi / 2
    )
    assert turned.bending_direction_rad == pytest.approx(math.pi / 2)
    core = turned.layer("cu_conductor")
    # The first core is at the top of the turned frame, on the tension side.
    np.testing.assert_allclose(core.centres_m[0], [0.0, 0.026], atol=1e-12)
    np.testing.assert_allclose(core.sigma_Pa, plain.layer("cu_conductor").sigma_Pa)
    wires = turned.layer("inner_armour")
    assert wires.kind == "wires"
    np.testing.assert_allclose(np.hypot(*wires.centres_m.T), 0.053)
    np.testing.assert_allclose(wires.centres_m[0], [0.0, 0.053], atol=1e-12)
    points = core.points_m()
    assert points.shape == (3, 3, 36, 2)
    distance = np.hypot(*(points[0, -1] - core.centres_m[0]).T)
    np.testing.assert_allclose(distance, 0.0097720502380584)
    sheath = turned.layer("hdpe_outer_sheath")
    # Bending stretches the fibres on the side of the bending direction.
    top = sheath.sigma_Pa[0, -1, np.argmin(np.abs(sheath.angles_rad - math.pi / 2))]
    bottom = sheath.sigma_Pa[
        0, -1, np.argmin(np.abs(sheath.angles_rad - 1.5 * math.pi))
    ]
    assert top > bottom


def test_laid_cores_are_placed_in_their_own_frame():
    """Held to the cable axis, every laid core bends with the section.

    The core-lay bending is written in each core's radial and tangential
    frame; placed in that frame, the stretched side of every core faces the
    bending direction, as plane sections give at a small core lay angle.
    """
    cable = _cable("iea_task49_66kv")
    assert cable.core_lay is not None
    field = cx.compute_section_stress(
        cable,
        0.05,
        tension_N=0.0,
        core_state="stuck",
        n_points=72,
        rotation_rad=math.pi / 2,
    )
    core = field.layer("cu_conductor")
    assert core.kind == "annulus"
    np.testing.assert_allclose(core.turns(), [0.0, 2 * math.pi / 3, 4 * math.pi / 3])
    points = core.points_m()
    # The first sample of each core points away from the cable axis.
    for i, centre in enumerate(core.centres_m):
        outward = points[i, -1, 0] - centre
        cross = outward[0] * centre[1] - outward[1] * centre[0]
        assert cross == pytest.approx(0.0, abs=1e-12)
        assert np.dot(outward, centre) > 0.0
        hottest = points[i, -1, int(np.argmax(core.sigma_Pa[i, -1]))]
        direction = hottest - centre
        angle = math.degrees(math.atan2(direction[1], direction[0]))
        assert angle == pytest.approx(90.0, abs=12.0)


def test_layers_without_a_stress_model_say_why():
    cable = _cable("saevik2011")
    field = cx.compute_section_stress(cable, 0.05, tension_N=1e5, n_points=24)
    pressure = field.layer("pressure_armour")
    assert not pressure.has_values
    assert "non-structural" in pressure.basis
    screen = cx.compute_section_stress(
        _cable("inter_array_three_core_33kv"), 0.05, n_points=24
    ).layer("cu_screen")
    assert screen.has_values and screen.kinematic
    assert screen.basis.startswith(KINEMATIC)


@pytest.mark.parametrize("core_state", ["stuck", "free"])
def test_kinematic_layer_is_its_modulus_times_the_strain_it_follows(core_state):
    """A loose screen of aligned cores: E (eps_0 + kappa . p), p from the
    cable axis held to the section, from the core axis in the free state."""
    cable = _cable("inter_array_three_core_33kv").model_copy(update={"core_lay": None})
    kappa = (0.04, -0.02)
    field = cx.compute_section_stress(
        cable, *kappa, tension_N=60e3, core_state=core_state, n_points=36
    )
    state = field.response.section_state
    assert state is not None
    screen = field.layer("cu_screen")
    assert screen.kinematic and screen.kind == "annulus"
    modulus = state.youngs_modulus_Pa["cu_screen"]
    assert modulus > 1e10  # copper
    points = screen.points_m()
    if core_state == "free":
        points = points - screen.centres_m[:, None, None, :]
    strain = state.axial_strain + kappa[0] * points[..., 0] + kappa[1] * points[..., 1]
    np.testing.assert_allclose(screen.sigma_Pa, modulus * strain, rtol=1e-12)
    np.testing.assert_array_equal(screen.sigma_min_Pa, screen.sigma_Pa)
    assert np.all(np.isnan(screen.utilisation))


def test_kinematic_layer_takes_the_strain_of_its_neighbours():
    """In a laid core, a loose screen has the strain of the load-bearing layers
    it lies between, at the shared radius, in every core slip state.

    A stranded conductor is on the strand basis of its bending stiffness: its
    extreme wire fibre is not the strain of its surface, which the screen on
    it follows (plane sections about the core axis); the screen's strain is
    compared with the layer outside it there."""
    cable = _cable("iea_task49_66kv")
    for core_state in ("frictional", "stuck", "free"):
        field = cx.compute_section_stress(
            cable, 0.03, 0.02, tension_N=80e3, core_state=core_state, n_points=36
        )
        state = field.response.section_state
        assert state is not None
        moduli = state.youngs_modulus_Pa
        for loose, inside, outside in (
            ("conductor_screen", "cu_conductor", "xlpe_insulation"),
            ("cu_tape_screen", "insulation_screen", "mdpe_core_sheath"),
        ):
            screen = field.layer(loose)
            assert (
                screen.kinematic and screen.core_state == field.layer(inside).core_state
            )
            strain = screen.sigma_Pa / moduli[loose]
            below = field.layer(inside)
            above = field.layer(outside)
            np.testing.assert_allclose(screen.radii_m[0], below.radii_m[-1])
            np.testing.assert_allclose(screen.radii_m[-1], above.radii_m[0])
            if own_bending_ratio(cable.get_layer(inside)) == 1.0:
                np.testing.assert_allclose(
                    strain[:, 0], below.sigma_Pa[:, -1] / moduli[inside], rtol=1e-9
                )
            np.testing.assert_allclose(
                strain[:, -1], above.sigma_Pa[:, 0] / moduli[outside], rtol=1e-9
            )


def test_kinematic_signs_and_thermal_strain():
    cable = _cable("inter_array_three_core_33kv").model_copy(update={"core_lay": None})
    bent = cx.compute_section_stress(
        cable, 0.05, tension_N=0.0, core_state="stuck", n_points=36
    )
    screen = bent.layer("cu_screen")
    x = screen.points_m()[..., 0]
    # Bending about the y axis stretches the side x > 0 and shortens x < 0.
    assert np.all(screen.sigma_Pa[x > 1e-9] > 0.0)
    assert np.all(screen.sigma_Pa[x < -1e-9] < 0.0)
    hottest = np.unravel_index(np.argmax(screen.sigma_Pa), screen.sigma_Pa.shape)
    assert x[hottest] == pytest.approx(float(np.max(x)))
    warm = cx.ThermalResult(
        r_m=np.array([0.0, 1.0]),
        T_C=np.array([60.0, 60.0]),
        T_conductor_C=60.0,
        T_armour_C=60.0,
        P_total_W_m=0.0,
    )
    heated = cx.compute_section_stress(
        cable, 0.0, tension_N=30e3, thermal=warm, n_points=24
    )
    state = heated.response.section_state
    assert state is not None
    free = state.free_thermal_strain["cu_screen"]
    assert free > 0.0
    expected = state.youngs_modulus_Pa["cu_screen"] * (state.axial_strain - free)
    np.testing.assert_allclose(heated.layer("cu_screen").sigma_Pa, expected, rtol=1e-12)


def test_rejects_bad_sampling():
    cable = _cable("menard2023")
    with pytest.raises(cx.CableInputError):
        cx.compute_section_stress(cable, 0.05, n_radii=0)
