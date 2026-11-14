# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""The friction a curvature history leaves, carried to the stress route.

Checks of :mod:`cablexsec.analysis.kinematics` and of
:class:`cablexsec.analysis.stress.FrictionField`: a history ramped from
straight leaves exactly the slip front of the stress route; the friction field
is invariant under a turn of the section; equivalent inputs (strain instead of
tension, a constant prescribed clamping, an explicit transverse model) give
identical kinematics; cores that are not laid are held to the cable axis; and
invalid inputs are rejected with their reason.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from _bundled import bundled

import cablexsec as cx
from cablexsec.analysis.kinematics import (
    armour_transverse_model,
    kinematic_history,
    kinematic_slip_amplitude,
    slip_along_helix,
)
from cablexsec.analysis.stress import FrictionField
from cablexsec.cable_adapters import cable_to_mkappa_specs, structural_armour_layers
from cablexsec.exceptions import CableInputError
from cablexsec.materials import library
from cablexsec.mechanics.stiffness import ea_fixed

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

CABLE = bundled("iea_task49_66kv")
THREE_CORE = bundled("inter_array_three_core_33kv")
SAMPLES = 80
CYCLE = 0.04 * np.sin(np.linspace(0.0, 2.5 * np.pi, SAMPLES))
RAMP = np.linspace(0.0, 0.05, 60)


# --------------------------------------------------------------------------- #
# Friction field                                                              #
# --------------------------------------------------------------------------- #
def test_ramp_from_straight_leaves_the_slip_front_of_the_stress_route() -> None:
    tension = 60e3
    history = kinematic_history(CABLE, RAMP, np.zeros(RAMP.size), tension_N=tension)
    np.testing.assert_array_equal(history.curvature_1_m, RAMP)
    field = history.friction_field(RAMP.size - 1)
    assert set(field.wires) == {"inner_armour", "outer_armour"}
    front = cx.compute_stress(CABLE, RAMP[-1], tension_N=tension)
    carried = cx.compute_stress(CABLE, RAMP[-1], tension_N=tension, friction=field)
    for name, values in front.sigma_f_Pa.items():
        np.testing.assert_allclose(
            carried.sigma_f_Pa[name], values, rtol=1e-12, atol=1e-6
        )
        np.testing.assert_array_equal(carried.slipping[name], front.slipping[name])
    assert carried.moment_recovered_Nm == pytest.approx(
        front.moment_recovered_Nm, rel=1e-12
    )


def test_a_layer_absent_from_the_field_keeps_its_slip_front() -> None:
    history = kinematic_history(CABLE, -RAMP, np.zeros(RAMP.size), tension_N=30e3)
    full = history.friction_field(RAMP.size - 1)
    only_inner = FrictionField({"inner_armour": full.wires["inner_armour"]})
    assert only_inner.wire_friction("outer_armour", np.zeros(3)) is None
    assert only_inner.core_strain(np.zeros(3)) is None
    # A stuck history at a small curvature: the friction a history left on
    # the inner armour is replaced by zero, the outer armour keeps the front.
    zero = FrictionField({"inner_armour": (np.array([0.0, math.pi]), np.zeros(2))})
    kappa = 0.01
    front = cx.compute_stress(CABLE, kappa, tension_N=30e3)
    carried = cx.compute_stress(CABLE, kappa, tension_N=30e3, friction=zero)
    np.testing.assert_array_equal(carried.sigma_f_Pa["inner_armour"], 0.0)
    assert not carried.slipping["inner_armour"].any()
    np.testing.assert_array_equal(
        carried.sigma_f_Pa["outer_armour"], front.sigma_f_Pa["outer_armour"]
    )


def test_friction_field_is_invariant_under_a_turn_of_the_section() -> None:
    history = kinematic_history(CABLE, CYCLE, 0.3 * CYCLE, tension_N=60e3)
    field = history.friction_field(SAMPLES // 2)
    # The laid cores without the core lay stage still carry their friction law.
    cores = history.cores
    assert cores is not None and not cores.moving and field.cores is not None
    assert cores.friction_Pa is not None
    np.testing.assert_allclose(
        field.core_strain(cores.azimuth_rad),
        cores.friction_Pa[:, SAMPLES // 2] / cores.friction_modulus_Pa,
        rtol=1e-12,
    )
    angle = 0.7
    turned = field.turned(angle)
    phi = np.linspace(0.0, 2.0 * math.pi, 37)
    for name in field.wires:
        np.testing.assert_allclose(
            turned.wire_friction(name, phi + angle),
            field.wire_friction(name, phi),
            rtol=1e-12,
            atol=1e-6,
        )
    np.testing.assert_allclose(
        turned.core_strain(phi + angle), field.core_strain(phi), rtol=1e-12
    )
    assert FrictionField().turned(angle).cores is None


def test_friction_field_rejects_mismatched_or_infinite_stations() -> None:
    bad_shape = FrictionField({"inner_armour": (np.zeros(3), np.zeros(2))})
    with pytest.raises(CableInputError, match="one value per station"):
        bad_shape.wire_friction("inner_armour", np.zeros(1))
    infinite = FrictionField(cores=(np.array([0.0, 1.0]), np.array([0.0, np.inf])))
    with pytest.raises(CableInputError, match="finite"):
        infinite.core_strain(np.zeros(1))


# --------------------------------------------------------------------------- #
# Equivalent inputs                                                           #
# --------------------------------------------------------------------------- #
def test_strain_and_its_contact_tension_equal_the_tension_input() -> None:
    tension = 60e3
    by_force = kinematic_history(CABLE, CYCLE, 0.3 * CYCLE, tension_N=tension)
    by_strain = kinematic_history(
        CABLE,
        CYCLE,
        0.3 * CYCLE,
        axial_strain=np.full(SAMPLES, tension / ea_fixed(CABLE)),
        contact_tension_N=tension,
    )
    np.testing.assert_array_equal(by_strain.axial_strain, by_force.axial_strain)
    np.testing.assert_array_equal(by_strain.tension_N, 0.0)
    for a, b in zip(by_force.layers, by_strain.layers, strict=True):
        np.testing.assert_array_equal(a.slip_m, b.slip_m)
        np.testing.assert_array_equal(a.stress_Pa, b.stress_Pa)


def test_constant_prescribed_clamping_equals_the_declared_one() -> None:
    tension = 60e3
    specs = cable_to_mkappa_specs(CABLE, tension)
    pressure = {
        layer.name: np.full(SAMPLES, spec.contact_pressure_Pa)
        for layer, spec in zip(structural_armour_layers(CABLE), specs, strict=True)
    }
    declared = kinematic_history(CABLE, CYCLE, 0.3 * CYCLE, tension_N=tension)
    prescribed = kinematic_history(
        CABLE, CYCLE, 0.3 * CYCLE, tension_N=tension, contact_pressure_Pa=pressure
    )
    for a, b in zip(declared.layers, prescribed.layers, strict=True):
        np.testing.assert_array_equal(a.slip_m, b.slip_m)
        np.testing.assert_array_equal(a.stress_Pa, b.stress_Pa)
    with pytest.raises(CableInputError, match="contact pressure of inner_armour"):
        kinematic_history(
            CABLE,
            CYCLE,
            0.3 * CYCLE,
            contact_pressure_Pa={"inner_armour": np.ones(SAMPLES - 1)},
        )


def test_a_named_transverse_model_is_that_of_the_default() -> None:
    curvature = 4 * CYCLE
    specs = cable_to_mkappa_specs(CABLE, 0.0)
    layer = structural_armour_layers(CABLE)[0]
    model = armour_transverse_model(CABLE, layer, specs[0], 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        everywhere = kinematic_history(
            CABLE, curvature, np.zeros(SAMPLES), transverse=True
        )
        named = kinematic_history(
            CABLE, curvature, np.zeros(SAMPLES), transverse={layer.name: model}
        )
    assert named.layers[0].transverse_m is not None
    assert named.layers[1].transverse_m is None
    np.testing.assert_array_equal(
        named.layers[0].transverse_m, everywhere.layers[0].transverse_m
    )


def test_transverse_model_needs_poisson_s_ratio() -> None:
    layer = structural_armour_layers(CABLE)[0]
    name = "test_armour_without_nu"
    # The armour material less its Poisson's ratio.
    properties = {
        key: (float(value), str(citation))
        for key, (value, citation) in library.MATERIALS[layer.material].items()
        if key != "nu"
    }
    library.register_material(name, properties)
    try:
        renamed = layer.model_copy(update={"material": name})
        spec = cable_to_mkappa_specs(CABLE, 0.0)[0]
        with pytest.raises(CableInputError, match="Poisson's ratio"):
            armour_transverse_model(CABLE, renamed, spec, 0.0)
    finally:
        library.unregister_material(name)


# --------------------------------------------------------------------------- #
# Wires of other sections and cores held to the axis                          #
# --------------------------------------------------------------------------- #
def test_rectangular_wires_slip_kinematically_without_a_yield_stress() -> None:
    pipe = bundled("saevik2011_rect")
    kappa = 0.05 * np.sin(np.linspace(0.0, 2.0 * np.pi, 60))
    history = kinematic_history(pipe, kappa, np.zeros(60), mu=0.0, tension_N=1e3)
    for layer in history.layers:
        # No yield stress of the wire material: no utilisation, but stresses.
        assert layer.utilisation is None and layer.bending_Pa is not None
        amplitude = kinematic_slip_amplitude(layer.r_helix_m, layer.lay_angle_rad, 1.0)
        expected = (
            -layer.lay_sign * amplitude * np.sin(layer.azimuth_rad)[:, None] * kappa
        )
        step = 2 * math.pi / layer.stations
        error = np.abs(layer.slip_m - expected).max() / np.abs(expected).max()
        assert error < step**2 / 12 * 1.05 + 1e-12


def test_cores_that_are_not_laid_are_held_to_the_cable_axis() -> None:
    held = THREE_CORE.model_copy(update={"core_lay": None})
    history = kinematic_history(held, RAMP, np.zeros(RAMP.size))
    cores = history.cores
    assert cores is not None
    assert cores.friction_model == "held" and not cores.moving
    assert cores.bodies == 3 and cores.friction_Pa is None
    np.testing.assert_array_equal(cores.slip_m, 0.0)
    utilisation = cores.utilisation
    assert utilisation is not None
    # Straight and unstrained: nothing is used; then the use grows with the
    # curvature, and the two cores mirrored about the bending plane agree.
    np.testing.assert_array_equal(utilisation[:, 0], 0.0)
    assert np.all(np.diff(utilisation, axis=1) > 0.0)
    np.testing.assert_allclose(utilisation[1], utilisation[2], rtol=1e-12)
    assert np.all(utilisation[0] >= utilisation[1])
    # Held cores have no friction law: the friction field carries none.
    assert history.friction_field(RAMP.size - 1).cores is None
    # Without the core utilisation there are no cores at all.
    bare = kinematic_history(held, RAMP, np.zeros(RAMP.size), core_utilisation=False)
    assert bare.cores is None and bare.fields == bare.layers


# --------------------------------------------------------------------------- #
# Slip integral and input checks                                               #
# --------------------------------------------------------------------------- #
def test_slip_of_one_sample_equals_a_column_of_many() -> None:
    azimuth = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
    gamma = np.outer(np.cos(azimuth), [1e-4, -2e-4, 3e-4])
    many, share = slip_along_helix(azimuth, gamma, 0.08, 0.2, -1)
    one, share_one = slip_along_helix(azimuth, gamma[:, 1], 0.08, 0.2, -1)
    scale = np.abs(many).max()
    np.testing.assert_allclose(one[:, 0], many[:, 1], rtol=0.0, atol=1e-12 * scale)
    assert share_one[0] == pytest.approx(share[1], abs=1e-12)
    # Opposite lay: the same field, mirrored.
    mirrored, _ = slip_along_helix(azimuth, gamma, 0.08, 0.2, 1)
    np.testing.assert_allclose(mirrored, -many, rtol=0.0, atol=1e-12 * scale)
    with pytest.raises(CableInputError, match="lay angle"):
        slip_along_helix(azimuth, gamma, 0.08, 0.0)
    with pytest.raises(CableInputError, match="lay sign"):
        slip_along_helix(azimuth, gamma, 0.08, 0.2, 0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"friction_model": "coulomb"}, "friction_model"),
        ({"kappa_x": np.array([0.0, np.nan])}, "finite and not empty"),
        ({"kappa_x": np.array([])}, "finite and not empty"),
        ({"kappa_y": np.zeros(3)}, "kappa_y must be finite"),
        ({"time_s": np.zeros(SAMPLES + 1)}, "time must be finite"),
        ({"tension_N": -1.0}, "must not be negative"),
    ],
)
def test_kinematic_history_rejects_bad_input(kwargs: dict, match: str) -> None:
    arguments = {"kappa_x": CYCLE, "kappa_y": np.zeros(SAMPLES)} | kwargs
    with pytest.raises(CableInputError, match=match):
        kinematic_history(CABLE, **arguments)
