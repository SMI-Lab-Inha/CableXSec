# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Utilisation curves, limits and roots of the component checks in closed form.

The component usage curve is piecewise linear in curvature with a
non-falling extension beyond its last curvature; its root, value and
governing component follow from that by hand. The limits of a component act
on stress, strain and net section as :mod:`component_limits` states. The
root finder of the capacity bisects a monotone usage to the curvature where
it reaches a level, and a wire of rectangular section bends to the stress of
its outer face and its edge.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis import capacity as cap
from cablexsec.analysis.component_limits import (
    NO_LIMIT,
    Limit,
    assess,
    component_limits,
    point_usage,
    stress_limit,
)
from cablexsec.analysis.components import (
    ComponentState,
    ComponentUsageCurve,
    bending_directions,
    capacity_utilisation,
    own_bending_ratio,
    usage_curvature_grid,
)
from cablexsec.cables.schema import (
    CableData,
    CableMeta,
    CylindricalLayer,
    CylindricalShellLayer,
)
from cablexsec.exceptions import CableInputError
from cablexsec.materials import library

pytestmark = pytest.mark.analytical_sanity


def _curve(usage: list[list[float]], grid: list[float]) -> ComponentUsageCurve:
    names = tuple(f"component {i}" for i in range(len(usage)))
    modes = tuple(
        tuple(("yielding" if i == 0 else "strain limit") for _ in grid)
        for i in range(len(usage))
    )
    return ComponentUsageCurve(
        tension_N=0.0,
        kappa_1_m=np.asarray(grid, dtype=np.float64),
        names=names,
        usage=np.asarray(usage, dtype=np.float64),
        modes=modes,
    )


# -- the usage curve -------------------------------------------------------------


def test_curve_without_components_has_no_limit() -> None:
    empty = ComponentUsageCurve(
        tension_N=0.0,
        kappa_1_m=np.array([0.0, 0.5, 1.0]),
        names=(),
        usage=np.zeros((0, 3)),
        modes=(),
    )
    assert not empty.assessed
    np.testing.assert_array_equal(empty.largest(), np.zeros(3))
    assert empty.at(10.0) == 0.0
    assert empty.root(1.0) == math.inf
    assert empty.governing(0.5) == ("", "")


def test_curve_is_linear_between_and_extended_beyond_its_curvatures() -> None:
    curve = _curve([[0.2, 0.5, 0.8], [0.1, 0.6, 0.7]], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(curve.largest(), [0.2, 0.6, 0.8])
    # Linear between the curvatures, the sign of the curvature does not count.
    assert curve.at(1.5) == pytest.approx(0.7)
    assert curve.at(-0.5) == pytest.approx(0.4)
    # Beyond the last curvature: the slope of the last interval, 0.2 per 1/m.
    assert curve.at(3.0) == pytest.approx(1.0)
    # The root inverts the same lines: between, beyond and at zero curvature.
    assert curve.root(0.7) == pytest.approx(1.5)
    assert curve.root(1.0) == pytest.approx(3.0)
    assert curve.root(0.1) == 0.0
    for level in (0.3, 0.75, 1.4):
        assert curve.at(curve.root(level)) == pytest.approx(level)
    # The component that governs, and its limit, at each curvature.
    assert curve.governing(1.0) == ("component 1", "strain limit")
    assert curve.governing(2.0) == ("component 0", "yielding")
    assert curve.governing(50.0) == ("component 0", "yielding")
    assert curve.governing(math.inf) == ("component 0", "yielding")


def test_curve_never_falls_beyond_its_last_curvature() -> None:
    falling = _curve([[0.2, 0.6, 0.5]], [0.0, 1.0, 2.0])
    # A falling last interval is not extrapolated downwards.
    assert falling.at(5.0) == pytest.approx(0.5)
    # A level the curve passes inside the grid is still found ...
    assert falling.root(0.4) == pytest.approx(0.5)
    # ... a level above it is never reached.
    assert falling.root(0.9) == math.inf


def test_curve_at_one_curvature_is_constant() -> None:
    single = _curve([[0.3]], [0.0])
    assert single.at(2.0) == pytest.approx(0.3)
    assert single.root(0.2) == 0.0
    assert single.root(0.5) == math.inf


def test_curvature_grid_holds_the_nodes_in_range() -> None:
    grid = usage_curvature_grid(2.0, n_kappa=5, nodes=(0.37, 3.0, -1.0, math.nan))
    assert grid[0] == 0.0 and grid[-1] == pytest.approx(2.0)
    assert np.all(np.diff(grid) > 0.0)
    assert 0.37 in grid
    assert 3.0 not in grid
    # Geometric from a thousandth of the top, and evenly spaced to it.
    assert grid[1] == pytest.approx(2.0e-3)
    assert 1.0 in grid


def test_bending_directions_span_the_symmetry_sector() -> None:
    single = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    assert bending_directions(single) == (0.0,)
    three = cx.load_cable(cx.inter_array_three_core_yaml_path(), geometry_override=None)
    assert bending_directions(three) == pytest.approx(
        (0.0, math.pi / 6.0, math.pi / 3.0)
    )


def _state(material: str, kind: str, utilisations) -> ComponentState:
    return ComponentState(
        name="component",
        component_type=kind,
        material=material,
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


def test_design_factor_divides_yield_and_strain_but_not_fracture() -> None:
    conductor = _state(
        "copper_janocha2024",
        "conductor",
        (("yielding", 0.4), ("tensile fracture", 0.5)),
    )
    # Characteristic: fracture governs; at 0.5 the proof stress does.
    assert capacity_utilisation(conductor) == (0.5, "tensile fracture")
    value, mode = capacity_utilisation(conductor, 0.5)
    assert mode == "yielding" and value == pytest.approx(0.8)
    for factor in (0.0, 1.5, math.nan):
        with pytest.raises(CableInputError, match=r"factor must lie in \(0, 1\]"):
            capacity_utilisation(conductor, factor)


# -- the limits of a component ---------------------------------------------------


def test_a_non_positive_material_value_gives_no_limit() -> None:
    overlay = library.MaterialOverlay(
        {
            "soft_metal": {
                "sigma_y_Pa": (0.0, "[ModelAssumption] no yield stress"),
                "epsilon_allow": (0.02, "[ModelAssumption] strain limit"),
            }
        }
    )
    with library.material_scope(overlay):
        limits = component_limits("soft_metal", "metallic sheath")
    assert [limit.mode for limit in limits] == ["strain limit"]
    assert limits[0].text == "allowable strain 2 % ([ModelAssumption] strain limit)"


def test_point_usage_uses_only_the_limits_it_can_evaluate() -> None:
    strain = (Limit("strain limit", 0.05, "allowable strain", "source"),)
    stress = np.array([1.0e6, -2.0e6])
    # Strain from stress over the modulus.
    np.testing.assert_allclose(point_usage(strain, stress, 1.0e9), [0.02, 0.04])
    # Without a modulus no strain follows: no limit applies.
    assert np.isnan(point_usage(strain, stress, 0.0)).all()
    fracture = (Limit("tensile fracture", 200e6, "tensile strength", "source"),)
    assert np.isnan(point_usage(fracture, stress, 1.0e9, math.nan)).all()
    np.testing.assert_allclose(
        point_usage(fracture, stress, 1.0e9, np.array([100e6, -50e6])), [0.5, 0.0]
    )
    # The largest point value is the assessment of the same component.
    both = (*strain, Limit("yielding", 10e6, "yield stress", "source"))
    judged = assess(both, peak_stress_Pa=2.0e6, peak_strain=2.0e6 / 1.0e9)
    assert float(np.max(point_usage(both, stress, 1.0e9))) == pytest.approx(
        judged.utilisation
    )


def test_stress_limit_is_the_smallest_point_limit() -> None:
    yielding = Limit("yielding", 120e6, "0.2 % proof stress", "source")
    fracture = Limit("tensile fracture", 220e6, "tensile strength", "source")
    strain = Limit("strain limit", 0.01, "allowable strain", "source")
    # The net-section fracture is not a point limit.
    assert stress_limit((fracture,), 100e9) == (math.nan, "")
    # The allowable strain times the modulus: 1 GPa here, above the yield.
    value, text = stress_limit((yielding, fracture, strain), 100e9)
    assert value == 120e6 and text == yielding.text
    value, text = stress_limit((strain, yielding), 1e9)
    assert value == pytest.approx(10e6) and text == strain.text
    # No modulus: the strain limit gives no stress.
    assert stress_limit((strain,), 0.0) == (math.nan, "")
    assert assess((fracture,), peak_stress_Pa=1e9).mode == NO_LIMIT


def test_own_bending_ratio_is_one_without_a_lower_bending_modulus() -> None:
    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    armour = next(layer for layer in cable.layers if layer.name == "inner_armour")
    assert own_bending_ratio(armour) == 1.0
    overlay = library.MaterialOverlay(
        {"bend_only": {"E_eff_bending_Pa": (1.0e9, "[ModelAssumption] bending")}}
    )
    layer = CylindricalLayer(
        name="tape", material="bend_only", r_inner_m=0.01, r_outer_m=0.02
    )
    # A material with a bending modulus but no axial modulus: no ratio.
    with library.material_scope(overlay):
        assert own_bending_ratio(layer) == 1.0


# -- the capacity root and the wire section --------------------------------------


def test_usage_root_bisects_to_the_level() -> None:
    def linear(kappa: float) -> float:
        return 0.1 + 0.4 * kappa

    assert cap._usage_root(linear, lambda: 1.0, 0.9) == pytest.approx(2.0, rel=1e-12)
    assert cap._usage_root(linear, lambda: math.inf, 0.9) == pytest.approx(
        2.0, rel=1e-12
    )
    # Passed at zero curvature, and never reached.
    assert cap._usage_root(linear, lambda: 1.0, 0.05) == 0.0
    assert cap._usage_root(lambda kappa: 0.5, lambda: 1.0, 0.9) == math.inf


def test_rectangular_wire_bends_to_its_face_and_edge_stress() -> None:
    cable = cx.load_cable(cx.witz1996_yaml_path("rectangular"), geometry_override=None)
    layer = next(
        layer for layer in cable.helical_layers if layer.wire_shape == "rectangular"
    )
    alpha = layer.lay_angle_rad
    assert alpha is not None
    e_w = library.get_property(layer.material, "E_Pa")
    section = layer.wire_section
    c, s = math.cos(alpha), math.sin(alpha)
    kappa = 0.2
    # In the bending plane only the normal curvature acts, at the outer face.
    face = cap._local_bending_max(layer, kappa, np.array(0.3), e_w, alpha, 0.3)
    assert float(face) == pytest.approx(
        e_w * kappa * c * c * abs(math.cos(2 * alpha)) * section.half_thickness_m
    )
    # A quarter turn from it only the lateral curvature acts, at the edge.
    edge = cap._local_bending_max(
        layer, -kappa, np.array(0.3 + math.pi / 2), e_w, alpha, 0.3
    )
    assert float(edge) == pytest.approx(
        e_w * kappa * c * (1 + s * s) * section.half_width_m
    )


def test_layers_without_their_geometry_are_not_annuli() -> None:
    assert cap._annulus_radii(CylindricalLayer(name="a", material="hdpe")) is None
    assert cap._annulus_radii(CylindricalShellLayer(name="b", material="hdpe")) is None
    shell = CylindricalShellLayer(name="c", material="hdpe", r_mid_m=0.1, t_m=0.002)
    assert cap._annulus_radii(shell) == pytest.approx((0.099, 0.101))


def test_closed_form_bracket_needs_an_armour_layer() -> None:
    polymer = CableData(
        cable=CableMeta(name="no armour"),
        layers=(
            CylindricalLayer(
                name="core", material="hdpe", r_inner_m=0.0, r_outer_m=0.02
            ),
        ),
    )
    with pytest.raises(CableInputError, match="armour layer"):
        cap._wire_yield_curvature_closed_form(polymer, 0.0)
