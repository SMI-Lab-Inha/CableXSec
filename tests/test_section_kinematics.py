# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Section quantities around the kinematics: axes, cores, torque and contact.

Closed forms and identities of :mod:`cablexsec.analysis.neutral_axis`,
:mod:`cablexsec.mechanics.core_kinematics`,
:mod:`cablexsec.analysis.torque_balance` (the fixed-radius balance and the
design bounds) and :mod:`cablexsec.mechanics.axisymmetric`, with the reason of
every rejected input.
"""

from __future__ import annotations

import math

import check_torque_balance as check
import numpy as np
import pytest
from _bundled import bundled

import cablexsec as cx
from cablexsec.analysis.neutral_axis import compute_neutral_axes, neutral_axes
from cablexsec.analysis.torque_balance import (
    _with_parameter,
    armour_fill_ratio,
    fixed_radius_balance,
)
from cablexsec.cable_adapters import cable_to_mkappa_specs, structural_armour_layers
from cablexsec.cables.schema import (
    CableData,
    CableMeta,
    CylindricalLayer,
    CylindricalOffsetLayer,
    HelicalLayer,
)
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.materials import library
from cablexsec.mechanics.axisymmetric import (
    _axial_EA,
    compute_winding_contact_pressure,
)
from cablexsec.mechanics.core_kinematics import core_kinematics, core_layer
from cablexsec.mechanics.stiffness import axial_torsional_matrix, ea_fixed

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

CABLE = bundled("iea_task49_66kv")
SINGLE = bundled("inter_array_single_core_33kv")
SAMPLES = 60
KX = 0.04 * np.sin(np.linspace(0.0, 2.5 * np.pi, SAMPLES))
KY = 0.3 * KX


# --------------------------------------------------------------------------- #
# Neutral axes                                                                #
# --------------------------------------------------------------------------- #
def test_strain_with_its_contact_tension_equals_the_tension_input() -> None:
    tension = np.full(SAMPLES, 60e3)
    by_force = compute_neutral_axes(CABLE, KX, KY, tension_N=tension)
    by_strain = compute_neutral_axes(
        CABLE,
        KX,
        KY,
        axial_strain=tension / ea_fixed(CABLE),
        contact_tension_N=tension,
    )
    np.testing.assert_array_equal(by_force.curvature_1_m, np.hypot(KX, KY))
    np.testing.assert_array_equal(by_strain.moment_Nm, by_force.moment_Nm)
    np.testing.assert_array_equal(
        by_strain.strain_axis_offset_m, by_force.strain_axis_offset_m
    )
    for a, b in zip(by_force.layers, by_strain.layers, strict=True):
        np.testing.assert_array_equal(a.resultant_Pa, b.resultant_Pa)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tension_N": np.zeros(3)}, "tension must be finite"),
        ({"contact_tension_N": np.zeros(3)}, "contact tension must be finite"),
        ({"contact_tension_N": np.full(SAMPLES, -1.0)}, "must not be negative"),
    ],
)
def test_compute_neutral_axes_rejects_bad_histories(kwargs: dict, match: str) -> None:
    with pytest.raises(CableInputError, match=match):
        compute_neutral_axes(CABLE, KX, KY, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"ei_min_Nm2": -1.0}, "EI_min"),
        ({"section_radius_m": 0.0}, "section radius"),
        ({"axial_strain": np.zeros(SAMPLES - 1)}, "axial strain"),
    ],
)
def test_neutral_axes_rejects_bad_section_input(kwargs: dict, match: str) -> None:
    layers = [
        (layer.name, spec)
        for layer, spec in zip(
            structural_armour_layers(CABLE),
            cable_to_mkappa_specs(CABLE, 0.0),
            strict=True,
        )
    ]
    with pytest.raises(CableInputError, match=match):
        neutral_axes(layers, KX, KY, **kwargs)


# --------------------------------------------------------------------------- #
# Core kinematics                                                             #
# --------------------------------------------------------------------------- #
def test_core_strain_is_linear_over_the_core_and_its_mean_is_the_axis() -> None:
    eps = np.linspace(0.0, 1e-3, SAMPLES)
    plain = core_kinematics(CABLE, KX, KY, state="stuck", n_stations=6)
    strained = core_kinematics(
        CABLE, KX, KY, state="stuck", n_stations=6, axial_strain=eps
    )
    # The cable strain enters the core axis as eps cos^2(beta).
    beta = plain.lay_angle_rad
    np.testing.assert_allclose(
        strained.axis_strain - plain.axis_strain,
        np.broadcast_to(math.cos(beta) ** 2 * eps, plain.axis_strain.shape),
        rtol=1e-12,
        atol=1e-18,
    )
    theta = 2.0 * np.pi * np.arange(16) / 16
    radius = 0.02
    strain = strained.strain(radius, theta)
    assert strain.shape == (6, 16, SAMPLES)
    # The circle mean of a first harmonic is the axis strain.
    np.testing.assert_allclose(
        strain.mean(axis=1), strained.mean_strain(), rtol=1e-12, atol=1e-18
    )
    np.testing.assert_array_equal(
        strained.strain(0.0, theta), np.repeat(strained.axis_strain[:, None], 16, 1)
    )
    # Opposite fibres: equal and opposite departures from the axis.
    opposite = strained.strain(radius, theta + math.pi)
    np.testing.assert_allclose(
        strain + opposite,
        np.broadcast_to(2.0 * strained.axis_strain[:, None, :], strain.shape),
        rtol=1e-12,
        atol=1e-15,
    )
    with pytest.raises(CableInputError, match="core radius"):
        strained.strain(-1.0, theta)
    with pytest.raises(CableInputError, match="theta"):
        strained.strain(radius, [0.0, math.nan])


@pytest.mark.parametrize(
    ("args", "kwargs", "match"),
    [
        ((KX, KX[:-1]), {}, "equal non-empty 1-D"),
        ((np.r_[KX[:-1], math.inf],), {}, "finite"),
        ((KX,), {"axial_strain": np.zeros(3)}, "axial strain"),
        ((KX,), {"n_stations": 0}, "n_stations"),
        ((KX,), {"n_stations": True}, "n_stations"),
    ],
)
def test_core_kinematics_rejects_bad_input(
    args: tuple, kwargs: dict, match: str
) -> None:
    with pytest.raises(CableInputError, match=match):
        core_kinematics(CABLE, *args, state="stuck", **kwargs)


def test_core_layers_belong_to_a_family_of_power_cores() -> None:
    assert isinstance(core_layer(CABLE, "xlpe_insulation"), CylindricalOffsetLayer)
    with pytest.raises(CableSchemaError, match="no_such_layer"):
        core_layer(CABLE, "no_such_layer")
    with pytest.raises(CableSchemaError, match="not a layer of a power core"):
        core_layer(CABLE, "inner_armour")
    with pytest.raises(CableSchemaError, match="one family of power cores"):
        core_kinematics(bundled("saevik2011"), KX, state="free")


# --------------------------------------------------------------------------- #
# Torque balance at fixed radius and design bounds                             #
# --------------------------------------------------------------------------- #
@pytest.mark.analytical_sanity
@pytest.mark.parametrize("name", ["inter_array_single_core_33kv", "saevik2011"])
def test_fixed_radius_layer_torques_are_the_closed_form(name: str) -> None:
    cable = bundled(name)
    balance = fixed_radius_balance(cable)
    armour = structural_armour_layers(cable)
    for layer in armour:
        assert layer.n_wires and layer.d_wire_m and layer.r_helix_m
        alpha = math.radians(abs(float(layer.lay_angle_deg or 0.0)))
        sign = 1.0 if layer.lay_direction == "right_hand" else -1.0
        expected = (
            sign
            * layer.n_wires
            * math.pi
            * layer.d_wire_m**2
            / 4.0
            * layer.r_helix_m
            * math.sin(alpha)
            * library.get_property(layer.material, "E_Pa")
            * math.cos(alpha) ** 2
        )
        assert balance.layer_torque_per_strain_Nm[layer.name] == pytest.approx(
            expected, rel=1e-12
        )
    torques = balance.layer_torque_per_strain_Nm
    assert balance.outer_layer == armour[-1].name
    inner = sum(torques[name] for name in balance.inner_layers)
    assert balance.balance_ratio == pytest.approx(
        -torques[armour[-1].name] / inner, rel=1e-12
    )
    # Contra-wound layers: the outer layer opposes the inner ones.
    assert balance.balance_ratio > 0.0
    at = axial_torsional_matrix(cable)
    assert balance.torque_per_tension_m == pytest.approx(at.K_et / at.K_ee, rel=1e-12)
    det = at.K_ee * at.K_tt - at.K_et**2
    assert balance.twist_per_tension_rad_per_N_m == pytest.approx(
        -at.K_et / det, rel=1e-12
    )


@pytest.mark.analytical_sanity
def test_fixed_radius_torques_make_up_k_et_without_laid_cores() -> None:
    # No laid cores and no other helical layer: the armour is the whole K_et.
    balance = fixed_radius_balance(SINGLE)
    total = sum(balance.layer_torque_per_strain_Nm.values())
    assert total == pytest.approx(axial_torsional_matrix(SINGLE).K_et, rel=1e-12)
    # One armour layer has nothing to balance.
    single_layer = fixed_radius_balance(bundled("thies2016_30kv"))
    assert single_layer.inner_layers == () and math.isnan(single_layer.balance_ratio)


def test_tension_grid_and_wire_shape_errors() -> None:
    strand = check.kim_strand(78.85, 87.41)
    with pytest.raises(CableInputError, match="minimum, maximum"):
        cx.torque_balance(strand, None)
    with pytest.raises(CableInputError, match="non-negative"):
        cx.torque_balance(strand, tensions_N=[-1.0, 1e3])
    with pytest.raises(CableSchemaError, match="rectangular"):
        cx.design_torque_balance(bundled("saevik2011_rect"))


def test_minimum_fill_sets_the_lower_lay_angle_bound() -> None:
    strand = check.kim_strand(78.85, 87.41)
    outer = strand.get_layer("layer_2")
    assert isinstance(outer, HelicalLayer)
    assert outer.n_wires and outer.d_wire_m and outer.r_helix_m
    # The fill at 30 deg is the minimum fill: below 30 deg the wires are
    # further apart than it allows, so no angle under 20 deg is admissible.
    fill = armour_fill_ratio(outer.n_wires, outer.d_wire_m, outer.r_helix_m, 30.0)
    with pytest.raises(CableInputError, match="no admissible"):
        cx.design_torque_balance(
            strand, bounds=(1.0, 20.0), min_fill=fill, max_fill=fill + 1.0
        )


def test_layers_off_the_axis_are_not_shifted_with_the_armour() -> None:
    outer = CABLE.get_layer("outer_armour")
    assert isinstance(outer, HelicalLayer) and outer.d_wire_m
    delta = 1e-3
    moved = _with_parameter(
        CABLE, "outer_armour", "outer_wire_diameter", outer.d_wire_m + delta
    )
    for before, after in zip(CABLE.layers, moved.layers, strict=True):
        if isinstance(before, CylindricalOffsetLayer):
            # Laid cores sit inside the armour: unchanged.
            assert after == before
    sheath = CABLE.get_layer("hdpe_outer_sheath")
    assert isinstance(sheath, CylindricalLayer)
    assert moved.get_layer("hdpe_outer_sheath").r_inner_m == pytest.approx(
        float(sheath.r_inner_m or 0.0) + delta, rel=1e-12
    )


# --------------------------------------------------------------------------- #
# Axisymmetric contact chain                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.analytical_sanity
def test_axial_stiffness_of_a_layer_is_the_closed_form() -> None:
    armour = CABLE.get_layer("inner_armour")
    assert isinstance(armour, HelicalLayer)
    assert armour.n_wires and armour.d_wire_m
    e = library.get_property(armour.material, "E_Pa")
    alpha = armour.lay_angle_rad or 0.0
    expected = (
        e * armour.n_wires * math.pi * armour.d_wire_m**2 / 4 * math.cos(alpha) ** 3
    )
    assert _axial_EA(armour) == pytest.approx(expected, rel=1e-12)
    sheath = CABLE.get_layer("hdpe_outer_sheath")
    assert isinstance(sheath, CylindricalLayer)
    assert sheath.r_inner_m is not None and sheath.r_outer_m is not None
    area = math.pi * (sheath.r_outer_m**2 - sheath.r_inner_m**2)
    assert _axial_EA(sheath) == pytest.approx(
        library.get_property(sheath.material, "E_Pa") * area, rel=1e-12
    )
    # A material without a modulus carries no axial load.
    stranded = sheath.model_copy(update={"material": "copper_stranded"})
    assert _axial_EA(stranded) == 0.0


def test_winding_pressure_needs_a_helical_layer() -> None:
    bare = CableData(
        cable=CableMeta(name="bare"),
        layers=(
            CylindricalLayer(
                name="sheath", material="hdpe", r_inner_m=0.01, r_outer_m=0.02
            ),
        ),
    )
    with pytest.raises(CableInputError, match="no helical layers"):
        compute_winding_contact_pressure(bare, 1.0e6)
