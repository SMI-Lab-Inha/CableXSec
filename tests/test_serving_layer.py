# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Serving layers (``construction: serving``): radial and hoop stiffness only.

A served yarn or wound tape transmits contact pressure but carries no axial
force, bending moment or torque. Checked against closed forms: plane-stress
thick-cylinder displacements, the exact removal of the layer's ``E A`` and
``G J`` at zero Poisson ratio, and the section stiffness assembly.
"""

import math

import numpy as np
import pytest
from pydantic import ValidationError

import cablexsec as cx
from cablexsec.cables.schema import CylindricalLayer, is_serving
from cablexsec.materials import library
from cablexsec.mechanics.axisymmetric import lame_thick_wall
from cablexsec.mechanics.concentric_section import (
    ElasticAnnulus,
    annulus_constitutive,
    solve_concentric_section,
)
from cablexsec.mechanics.radial_cable import solve_cable_radial_state
from cablexsec.mechanics.stiffness import (
    axial_stiffness_by_layer,
    bending_stiffness_by_layer,
    ei_min_slip,
)


def _yarn(**extra: object) -> dict[str, object]:
    return {
        "name": "yarn",
        "layer_type": "cylindrical",
        "material": "yarn_chae2023",
        "r_inner_m": 0.05,
        "r_outer_m": 0.052,
        **extra,
    }


def test_a_serving_is_non_structural_and_wound_on_a_layer() -> None:
    layer = CylindricalLayer.model_validate(_yarn(construction="serving"))
    assert layer.is_serving and is_serving(layer) and not layer.structural
    assert CylindricalLayer.model_validate(
        _yarn(construction="serving", structural=False)
    ).is_serving
    with pytest.raises(ValidationError, match="structural must be false"):
        CylindricalLayer.model_validate(_yarn(construction="serving", structural=True))
    with pytest.raises(ValidationError, match="r_inner_m must be > 0"):
        CylindricalLayer.model_validate(_yarn(construction="serving", r_inner_m=0.0))
    with pytest.raises(ValidationError):
        CylindricalLayer.model_validate(_yarn(construction="tape"))
    shrink = {"hoop_strain": 0.001, "basis": "assumed", "source": "test"}
    with pytest.raises(ValidationError, match="not extruded"):
        CylindricalLayer.model_validate(
            _yarn(construction="serving", extrusion_shrinkage=shrink)
        )
    plain = CylindricalLayer.model_validate(_yarn())
    assert not plain.is_serving and plain.structural


def test_plane_stress_constitutive() -> None:
    e, nu = 1.5e9, 0.45
    d = annulus_constitutive(e, nu, carries_axial_load=False)
    plane = e / (1 - nu * nu)
    np.testing.assert_array_equal(d[2], 0.0)
    np.testing.assert_array_equal(d[:, 2], 0.0)
    assert d[0, 0] == d[1, 1] == pytest.approx(plane, rel=1e-15)
    assert d[0, 1] == pytest.approx(plane * nu, rel=1e-15)
    # Static condensation of the axial stress from the 3D law gives it too.
    full = annulus_constitutive(e, nu)
    condensed = full[:2, :2] - np.outer(full[:2, 2], full[2, :2]) / full[2, 2]
    np.testing.assert_allclose(d[:2, :2], condensed, rtol=1e-13)


@pytest.mark.parametrize("tension_N", [0.0, 2.0e5])
def test_serving_under_pressure_is_a_plane_stress_thick_cylinder(
    tension_N: float,
) -> None:
    e, nu, a, b, p = 1.5e9, 0.45, 0.03, 0.035, 2.0e6
    core = ElasticAnnulus("core", 0.0, a, 110e9, 0.34)
    serving = ElasticAnnulus("serving", a, b, e, nu, carries_axial_load=False)
    result = solve_concentric_section(
        [core, serving], tension_N=tension_N, external_pressure_Pa=p
    )
    q = float(result.contact_pressure_Pa[0])
    assert q > 0
    for radius, value in (
        (a, result.inner_displacement_m["serving"]),
        (b, result.outer_displacement_m["serving"]),
    ):
        exact = lame_thick_wall(radius, a, b, q, p, e, nu).u_r_m
        assert value == pytest.approx(exact, rel=1e-10)
    # The core carries the whole tension: under the uniform lateral pressure
    # q its axial stress is E eps - 2 nu q, so eps = (T / A + 2 nu q) / E.
    area = math.pi * a * a
    assert result.axial_resultant_N == pytest.approx(tension_N, abs=1e-6)
    assert result.axial_strain == pytest.approx(
        (tension_N / area + 2 * 0.34 * q) / 110e9, rel=1e-10
    )


def test_zero_poisson_serving_removes_exactly_its_ea_and_gj() -> None:
    e, a, b = 1.5e9, 0.03, 0.035
    core = ElasticAnnulus("core", 0.0, a, 110e9, 0.34)

    def tangent(carries: bool) -> np.ndarray:
        layers = [core, ElasticAnnulus("s", a, b, e, 0.0, carries_axial_load=carries)]
        return solve_concentric_section(
            layers, tension_N=1e5, external_pressure_Pa=1e6
        ).tangent.matrix

    solid, serving = tangent(True), tangent(False)
    assert solid[0, 0] - serving[0, 0] == pytest.approx(
        e * math.pi * (b * b - a * a), rel=1e-11
    )
    assert solid[1, 1] - serving[1, 1] == pytest.approx(
        e / 2 * math.pi * (b**4 - a**4) / 2, rel=1e-11
    )


def _chae(serving: bool) -> cx.CableData:
    cable = cx.load_cable(cx.chae2023_yaml_path())
    layers = []
    for layer in cable.layers:
        if layer.name in {"yarn1", "yarn2"} and not serving:
            layer = CylindricalLayer.model_validate(
                {**layer.model_dump(), "construction": "", "structural": True}
            )
        layers.append(layer)
    return cable.model_copy(update={"layers": tuple(layers)})


def test_packaged_chae_yarns_are_servings() -> None:
    cable = cx.load_cable(cx.chae2023_yaml_path())
    assert {x.name for x in cable.layers if is_serving(x)} == {"yarn1", "yarn2"}


def test_serving_adds_no_section_stiffness() -> None:
    solid, serving = _chae(False), _chae(True)
    e = library.get_property("yarn_chae2023", "E_Pa")
    ea_solid = axial_stiffness_by_layer(solid)
    ea_serving = axial_stiffness_by_layer(serving)
    assert "yarn1" not in ea_serving and "yarn2" not in ea_serving
    removed_ea = removed_ei = 0.0
    for name in ("yarn1", "yarn2"):
        layer = solid.get_layer(name)
        assert isinstance(layer, CylindricalLayer)
        ri, ro = float(layer.r_inner_m or 0), float(layer.r_outer_m or 0)
        removed_ea += e * math.pi * (ro**2 - ri**2)
        removed_ei += e * math.pi * (ro**4 - ri**4) / 4
        assert bending_stiffness_by_layer(serving)[name].plane_section_Nm2 == 0.0
    assert sum(ea_solid.values()) - sum(ea_serving.values()) == pytest.approx(
        removed_ea, rel=1e-12
    )
    assert ei_min_slip(solid) - ei_min_slip(serving) == pytest.approx(
        removed_ei, rel=1e-12
    )
    # The other layers keep their values exactly.
    for name, value in ea_serving.items():
        assert value == ea_solid[name]


def test_three_core_mesh_serving_ring_carries_no_axial_force_or_torque() -> None:
    solid, serving = _chae(False), _chae(True)
    kwargs = {"tension_N": 1.0e5, "external_pressure_Pa": 1.0e5}
    a = solve_cable_radial_state(solid, **kwargs).result
    b = solve_cable_radial_state(serving, **kwargs).result
    forces = b.layer_axial_force_N
    # yarn1 is the meshed ring of the bundle, yarn2 a concentric annulus.
    assert abs(forces["yarn1"]) < 1e-9 * 1.0e5
    assert abs(forces["yarn2"]) < 1e-9 * 1.0e5
    # The layer forces sum to the tension to the contact solve's tolerance.
    assert sum(forces.values()) == pytest.approx(1.0e5, rel=1e-6)
    assert abs(a.layer_axial_force_N["yarn1"]) > 1.0
    # Removing the yarn torsion lowers the solved torsional rigidity.
    assert b.gj_fixed_tension_Nm2 < a.gj_fixed_tension_Nm2
    assert b.ea_fixed_twist_N < a.ea_fixed_twist_N


def test_serving_is_outside_the_component_load_path() -> None:
    cable = cx.load_cable(cx.thies2016_yaml_path())
    assert is_serving(cable.get_layer("pp_yarn_serving"))
    response = cx.compute_component_response(cable, 0.1, tension_N=5.0e4)
    state = next(s for s in response.states if s.name == "pp_yarn_serving")
    # The kinematic stress of the strain it follows, as a loose screen.
    assert state.kinematic
    assert response.critical_component != "pp_yarn_serving"
    e = response.section_state.youngs_modulus_Pa["pp_yarn_serving"]
    assert state.sigma_local_max_Pa == pytest.approx(e * state.strain_max, rel=1e-9)


def test_serving_in_the_solved_concentric_section() -> None:
    # A concentric section (the bundled single-core cable) with its bedding
    # declared as a serving: the coupled path keeps it in the pressure chain
    # as a plane-stress annulus without axial force or torque, and the
    # component and section stress fields agree.
    base = cx.load_cable(cx.inter_array_single_core_yaml_path())
    layers = tuple(
        CylindricalLayer.model_validate(
            {**x.model_dump(), "construction": "serving", "structural": False}
        )
        if x.name == "hdpe_bedding"
        else x
        for x in base.layers
    )
    cable = base.model_copy(update={"layers": layers})
    state = cx.solve_section_state(
        cable, tension_N=2.0e5, p_ext_Pa=1.0e6, contact_model="qp"
    )
    name = "hdpe_bedding"
    normal = state.layers[name]
    assert normal.longitudinal_stress_Pa == 0.0
    assert normal.axial_force_N == 0.0 and normal.torque_Nm == 0.0
    layer = cable.get_layer(name)
    assert isinstance(layer, CylindricalLayer)
    a, b = float(layer.r_inner_m or 0), float(layer.r_outer_m or 0)
    p_a = -(
        normal.radial_stress_constant_Pa - normal.radial_stress_inverse_Pa_m2 / a**2
    )
    p_b = -(
        normal.radial_stress_constant_Pa - normal.radial_stress_inverse_Pa_m2 / b**2
    )
    assert p_a > 0 and p_b > 0
    # Its face displacements are those of a plane-stress thick cylinder under
    # the solved face pressures.
    assert state.radial is not None
    section = state.radial.section
    e = state.youngs_modulus_Pa[name]
    nu = library.get_property(layer.material, "nu")
    for radius, value in (
        (a, section.inner_displacement_m[name]),
        (b, section.outer_displacement_m[name]),
    ):
        exact = lame_thick_wall(radius, a, b, p_a, p_b, e, nu).u_r_m
        assert value == pytest.approx(exact, rel=1e-9)
    response = cx.compute_component_response(
        cable, 0.05, tension_N=2.0e5, p_ext_Pa=1.0e6, contact_model="qp"
    )
    item = next(s for s in response.states if s.name == name)
    assert item.kinematic
    field = cx.compute_section_stress(
        cable, 0.05, tension_N=2.0e5, p_ext_Pa=1.0e6, contact_model="qp"
    )
    tape = next(x for x in field.layers if x.name == name)
    assert float(np.max(tape.sigma_max_Pa)) == pytest.approx(
        item.sigma_local_max_Pa, rel=1e-9
    )
