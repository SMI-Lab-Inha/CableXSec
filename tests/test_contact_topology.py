# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""A bedding layer must not be replaced by a fictitious steel-wire crossing."""

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.cable_adapters import (
    _fretting_contact,
    armour_crossing_line_loads,
    cable_to_mkappa_specs,
)
from cablexsec.mechanics.hertz_contact import crossed_cylinder_curvatures


def test_published_bedding_excludes_direct_steel_hertz_contact():
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    inner, outer = cable.helical_layers
    assert _fretting_contact(inner, outer, 1000) == (0, 0, 0)
    response = cx.compute_stress(cable, 0.02, tension_N=1e5)
    for layer in cable.helical_layers:
        np.testing.assert_allclose(
            response.sigma_vm_Pa[layer.name],
            np.abs(response.sigma_total_Pa[layer.name]),
            rtol=1e-12,
        )


def test_touching_opposed_wires_retain_direct_contact():
    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    inner, outer = cable.helical_layers
    stiffness, pressure, density = _fretting_contact(inner, outer, 1000)
    assert min(stiffness, pressure, density) > 0


@pytest.mark.parametrize("model", ["bounded", "qp"])
def test_shared_crossing_has_equal_force_and_hertz_solution(model):
    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    inner, outer = cable.helical_layers
    contacts = armour_crossing_line_loads(cable, 1e5, contact_model=model)
    partner1, q1 = contacts[inner.name]
    partner2, q2 = contacts[outer.name]
    k1, p1, rho1 = _fretting_contact(inner, partner1, q1)
    k2, p2, rho2 = _fretting_contact(outer, partner2, q2)
    assert q1 * inner.n_wires == pytest.approx(q2 * outer.n_wires, rel=1e-13)
    assert q1 / rho1 == pytest.approx(q2 / rho2, rel=1e-13)
    assert k1 == pytest.approx(k2, rel=1e-12)
    assert p1 == pytest.approx(p2, rel=1e-12)


@pytest.mark.parametrize("angle", [0.01, 0.3, 0.8, 1.5707963267948966])
def test_unequal_cylinder_curvatures_match_quadratic_gap(angle):
    d1, d2 = 0.003, 0.005
    normal1 = np.array([0.0, 1.0])
    normal2 = np.array([-np.sin(angle), np.cos(angle)])
    gap_matrix = np.outer(normal1, normal1) / d1 + np.outer(normal2, normal2) / d2
    expected = np.linalg.eigvalsh(gap_matrix)
    actual = crossed_cylinder_curvatures(d1, angle, partner_diameter_m=d2)
    np.testing.assert_allclose(actual, expected, rtol=1e-11)
    np.testing.assert_allclose(
        actual,
        crossed_cylinder_curvatures(d2, angle, partner_diameter_m=d1),
        rtol=1e-13,
    )


def test_unequal_material_crossing_is_reciprocal():
    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    inner, outer = cable.helical_layers
    other = outer.model_copy(update={"material": "copper_tape_screen"})
    q1, q2 = 1200.0, 1200.0 * inner.n_wires / outer.n_wires
    first = _fretting_contact(inner, other, q1)
    second = _fretting_contact(other, inner, q2)
    np.testing.assert_allclose(first[:2], second[:2], rtol=1e-12)


def test_auxiliary_lumped_law_requires_length_for_normal_line_load():
    from dataclasses import replace

    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    spec = cable_to_mkappa_specs(cable)[0]
    with pytest.raises(ValueError, match="transfer length"):
        _ = spec.f_y
    declared = replace(spec, friction_transfer_length_m=0.2)
    np.testing.assert_allclose(
        declared.f_y, spec.mu * spec.F_n_axi_N * 0.2 / spec.A_wire_m2
    )
