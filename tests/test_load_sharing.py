# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Load sharing: the axial force of every layer of the solved section.

Public cables only. Each layer's force is recovered from its own strain and
material law; the checks are equilibrium (the forces sum to the section's
axial force and to the tension given to it), independent recoveries of single
layers (wire stress, the Lamé solution of an annulus, the core force of the
trefoil model), the fixed-radius limit and the tension sweep.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.section_properties import (
    SectionLoadCase,
    compute_load_sharing,
    compute_section_properties,
)
from cablexsec.cables.schema import HelicalLayer
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.concentric_section import ElasticAnnulus, ElasticHelicalRing
from cablexsec.mechanics.radial_cable import (
    _solid_area,
    build_radial_section,
    solve_cable_radial_states,
)
from cablexsec.mechanics.radial_solution import RadialSection
from cablexsec.mechanics.stiffness import load_sharing

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

SOLVED = (
    "inter_array_single_core_33kv",
    "inter_array_three_core_33kv",
    "iea_task49_66kv",
    "saevik2011_rect",
    "witz1996_rect",
)


def _cable(key: str) -> cx.CableData:
    path = Path(cx.__file__).parent / "cables" / f"{key}.yaml"
    return cx.load_cable(path, geometry_override=None)


@pytest.fixture(scope="module")
def cables() -> dict[str, cx.CableData]:
    return {key: _cable(key) for key in SOLVED}


# --------------------------------------------------------------------------- #
# Equilibrium                                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", SOLVED)
@pytest.mark.parametrize("twist", ["held", "free"])
@pytest.mark.parametrize("flooded", [0.0, 2.0e6])
def test_layer_forces_sum_to_the_tension(
    cables: dict[str, cx.CableData], key: str, twist: str, flooded: float
) -> None:
    cable = cables[key]
    options: dict[str, object] = (
        {"prescribed_twist_per_m": 0.0}
        if twist == "held"
        else {"prescribed_twist_per_m": None, "torque_Nm": 0.0}
    )
    if flooded:
        options.update(flooded_pressure_Pa=flooded, external_pressure_Pa=0.0)
    tensions = [0.0, 20e3, 100e3]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        states = solve_cable_radial_states(
            cable, tensions, loose_layers="neighbour", **options
        )
    names = {layer.name for layer in cable.layers}
    for tension, state in zip(tensions, states, strict=True):
        mechanical = state.result.layer_axial_force_N
        assert set(mechanical) <= names
        # The layers' own rows add up to the section's extension row: an
        # identity, to round-off.
        assert sum(mechanical.values()) == pytest.approx(
            state.result.axial_force_N, rel=1e-12, abs=1e-6
        )
        # And to the tension given to the section, to the solver's
        # equilibrium residual (at most 0.01 N here, from the trefoil LCP).
        true = state.layer_axial_force()
        assert sum(true.values()) == pytest.approx(tension, abs=1e-6 * 1e5)
        if flooded:
            area = state.layer_solid_area_m2
            for name, force in true.items():
                assert force == pytest.approx(
                    mechanical[name] - flooded * area.get(name, 0.0),
                    rel=1e-12,
                    abs=1e-9,
                )


def test_layer_areas_are_the_solid_area(cables: dict[str, cx.CableData]) -> None:
    for key in ("inter_array_single_core_33kv", "inter_array_three_core_33kv"):
        section, _ = build_radial_section(cables[key], loose_layers="neighbour")
        areas = section.layer_solid_areas()
        assert sum(areas.values()) == pytest.approx(_solid_area(section), rel=1e-12)


# --------------------------------------------------------------------------- #
# Independent recoveries of single layers                                      #
# --------------------------------------------------------------------------- #
def test_helical_layer_force_is_its_wire_force(
    cables: dict[str, cx.CableData],
) -> None:
    cable = cables["inter_array_single_core_33kv"]
    state = solve_cable_radial_states(
        cable, [60e3], loose_layers="neighbour", prescribed_twist_per_m=None
    )[0]
    for layer in cable.helical_layers:
        assert isinstance(layer, HelicalLayer)
        assert layer.d_wire_m and layer.n_wires and layer.lay_angle_rad is not None
        area = layer.n_wires * math.pi * layer.d_wire_m**2 / 4
        expected = (
            state.result.wire_stress_Pa[layer.name]
            * area
            * math.cos(layer.lay_angle_rad)
        )
        assert state.result.layer_axial_force_N[layer.name] == pytest.approx(
            expected, rel=1e-12
        )


def test_annulus_force_is_the_lame_solution(
    cables: dict[str, cx.CableData],
) -> None:
    cable = cables["inter_array_single_core_33kv"]
    section, _ = build_radial_section(
        cable, loose_layers="neighbour", temperature_C=55.0, flooded_pressure_Pa=1e6
    )
    result = section.solve(tension_N=40e3, external_pressure_Pa=0.0)
    for layer in section.layers:
        if not isinstance(layer, ElasticAnnulus) or layer.inner_radius_m == 0.0:
            continue
        a, b = layer.inner_radius_m, layer.outer_radius_m
        # u = A r + B / r through the two face displacements.
        coeff_a, _coeff_b = np.linalg.solve(
            np.array([[a, 1 / a], [b, 1 / b]]),
            np.array(
                [
                    result.inner_displacement_m[layer.name],
                    result.outer_displacement_m[layer.name],
                ]
            ),
        )
        e, nu, free = (
            layer.youngs_modulus_Pa,
            layer.poisson_ratio,
            (layer.free_thermal_strain),
        )
        lam = e * nu / ((1 + nu) * (1 - 2 * nu))
        mu = e / (2 * (1 + nu))
        strain = result.axial_strain
        # The radial terms B/r² cancel in the trace: sigma_zz is uniform.
        sigma = lam * (2 * coeff_a + strain - 3 * free) + 2 * mu * (strain - free)
        expected = sigma * math.pi * (b**2 - a**2)
        assert result.layer_axial_force_N[layer.name] == pytest.approx(
            expected, rel=1e-9, abs=1e-6
        )


def test_trefoil_cores_carry_the_bundle_core_force(
    cables: dict[str, cx.CableData],
) -> None:
    cable = cables["inter_array_three_core_33kv"]
    state = solve_cable_radial_states(cable, [50e3], loose_layers="neighbour")[0]
    bundle = state.result.bundle
    assert bundle is not None
    cores = [
        layer.name
        for layer in cable.layers
        if getattr(layer, "n_cores", 1) == 3
        and layer.name in state.result.layer_axial_force_N
    ]
    assert len(cores) >= 4
    total = sum(state.result.layer_axial_force_N[name] for name in cores)
    assert total == pytest.approx(bundle.core_axial_force_N, rel=1e-9)


# --------------------------------------------------------------------------- #
# Fixed-radius limit                                                           #
# --------------------------------------------------------------------------- #
def test_fixed_radius_limit_is_the_equal_strain_share() -> None:
    """With the wires on a rigid bore and an outer sheath without Poisson
    contraction, the solved section is the fixed-radius assembly: each
    helical layer carries n E A cos³(a) e and the sheath E A e, the terms of
    :func:`~cablexsec.mechanics.stiffness.axial_stiffness_by_layer`."""
    d, steel = 5e-3, 2.07e11
    inner = ElasticHelicalRing("inner", 0.05, d, 60, math.radians(30.0), steel)
    outer = ElasticHelicalRing("outer", 0.05 + d, d, 64, -math.radians(28.0), steel)
    sheath = ElasticAnnulus("sheath", 0.05 + 1.5 * d, 0.065, 9.0e8, 0.0)
    section = RadialSection([inner, outer, sheath], rigid_bore=True)
    strain = 2.0e-3
    result = section.solve(prescribed_axial_strain=strain)
    forces = result.layer_axial_force_N
    for ring in (inner, outer):
        c = math.cos(ring.signed_lay_angle_rad)
        assert forces[ring.name] == pytest.approx(
            ring.total_area_m2 * steel * c**3 * strain, rel=1e-12
        )
    area = math.pi * (0.065**2 - (0.05 + 1.5 * d) ** 2)
    assert forces["sheath"] == pytest.approx(9.0e8 * area * strain, rel=1e-12)
    assert sum(forces.values()) == pytest.approx(result.axial_force_N, rel=1e-12)


def test_solved_share_relieves_the_armour_against_fixed_radius(
    cables: dict[str, cx.CableData],
) -> None:
    """On a real cable the two differ: the armour pulls inward as it is
    tensioned and the radial contraction relieves the wires (the s² u/r term of
    the wire strain), so the armour carries less than at fixed radius and the
    core more."""
    cable = cables["inter_array_single_core_33kv"]
    solved = compute_load_sharing(cable, 100e3, 3)
    fixed = load_sharing(cable, 100e3)
    armour = [layer.name for layer in cable.helical_layers]
    assert solved.route == "solved"
    assert sum(solved.force_N[n][-1] for n in armour) < sum(fixed[n] for n in armour)
    assert solved.force_N["cu_conductor"][-1] > fixed["cu_conductor"]


# --------------------------------------------------------------------------- #
# The tension sweep                                                            #
# --------------------------------------------------------------------------- #
def test_sweep_on_the_solved_route(cables: dict[str, cx.CableData]) -> None:
    cable = cables["inter_array_three_core_33kv"]
    result = compute_load_sharing(cable, 80e3, 5, twist="free")
    np.testing.assert_allclose(result.tension_N, [0, 20e3, 40e3, 60e3, 80e3])
    assert result.route == "solved" and not result.route_reason
    assert (
        result.layers[0] == "cu_conductor" and result.layers[-1] == "hdpe_outer_sheath"
    )
    np.testing.assert_allclose(result.total_N, result.tension_N, atol=0.05)
    share = result.share()
    assert all(math.isnan(v[0]) for v in share.values())
    assert sum(v[-1] for v in share.values()) == pytest.approx(1.0, abs=1e-6)
    # The manufacturing clamping locks self-balancing forces in.
    assert sum(result.force_at_zero_N.values()) == pytest.approx(0.0, abs=0.05)
    assert max(result.force_at_zero_N.values()) > 1e3
    # The same forces as the load case of the section properties.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        case = compute_section_properties(
            cable, [SectionLoadCase("A", 80e3, "free")]
        ).cases[0]
    for name, force in case.load_share_N.items():
        assert result.force_N[name][-1] == pytest.approx(force, rel=1e-9, abs=1e-6)


def test_sweep_with_flooded_pressure_and_temperature(
    cables: dict[str, cx.CableData],
) -> None:
    cable = cables["inter_array_single_core_33kv"]
    flooded = compute_load_sharing(
        cable, 50e3, 3, temperature_C=40.0, external_pressure_Pa=3e6
    )
    np.testing.assert_allclose(flooded.total_N, flooded.tension_N, atol=1e-6)
    # Flooded: the wall tension is the effective tension less p A_solid.
    state = solve_cable_radial_states(
        cable,
        [flooded.wall_tension_N[-1]],
        loose_layers="neighbour",
        flooded_pressure_Pa=3e6,
        external_pressure_Pa=0.0,
        temperature_C=40.0,
    )[0]
    assert state.mechanical_axial_force_N == pytest.approx(50e3, rel=1e-12)
    assert sum(state.layer_axial_force().values()) == pytest.approx(
        flooded.wall_tension_N[-1], abs=1e-6
    )
    sealed = compute_load_sharing(
        cable, 50e3, 3, external_pressure_Pa=3e6, annulus="sealed"
    )
    np.testing.assert_allclose(sealed.total_N, sealed.tension_N, atol=1e-6)
    assert sealed.wall_tension_N[0] < 0.0


def test_sweep_falls_back_to_fixed_radius() -> None:
    cable = cx.load_cable(cx.poradowski2025_220kv_yaml_path(), geometry_override=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        held = compute_load_sharing(cable, 40e3, 3)
        free = compute_load_sharing(cable, 40e3, 3, twist="free")
    # The solved route refuses the corrugated sheath (and the stranded
    # conductor, which it would meet next).
    assert held.route == "fixed_radius" and "corrugated shells" in held.route_reason
    expected = load_sharing(cable, 40e3)
    assert set(held.force_N) == {n for n, v in expected.items() if v}
    for name, force in held.force_N.items():
        assert force[-1] == pytest.approx(expected[name], rel=1e-9)
        assert force[1] == pytest.approx(expected[name] / 2, rel=1e-9)
    assert any("no twist freedom" in note for note in free.notes)


def test_sweep_input_checks(cables: dict[str, cx.CableData]) -> None:
    cable = cables["inter_array_single_core_33kv"]
    for args, options in (
        ((0.0,), {}),
        ((math.inf,), {}),
        ((1e4, 1), {}),
        ((1e4, 2.5), {}),
        ((1e4,), {"twist": "loose"}),
        ((1e4,), {"annulus": "wet"}),
        ((1e4,), {"external_pressure_Pa": -1.0}),
    ):
        with pytest.raises(CableInputError):
            compute_load_sharing(cable, *args, **options)  # type: ignore[arg-type]
