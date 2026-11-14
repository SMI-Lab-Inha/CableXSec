# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Sectional mechanics: slip-cap limits, input rules and stiffness assembly.

Verification: the Papailiou cap and onset curvature vanish in their stated
limits (no clamping, no friction, no stiffness, zero lay angle), the three-core
support shape is uniform without pressure, an isothermal thermal profile equals
uniform layer temperatures, and the metallic-only rigidity leaves out a
polymer shell.
"""

from __future__ import annotations

import math
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.mechanics.contact_pressure import solve_three_core
from cablexsec.mechanics.mkappa import HelicalLayerSpec
from cablexsec.mechanics.papailiou import (
    direction_threshold_curvature,
    full_slip_curvature,
    slip_onset_curvature,
    wire_slip_history,
    wire_slip_stress_cap,
)
from cablexsec.mechanics.section_state import solve_section_state
from cablexsec.mechanics.stiffness import axial_stiffness_by_layer, ei_min_slip


def _spec(**kwargs) -> HelicalLayerSpec:
    base = {
        "n_wires": 60,
        "r_helix_m": 0.06,
        "lay_angle_rad": math.radians(20.0),
        "E_wire_Pa": 2.1e11,
        "A_wire_m2": math.pi * 0.004**2 / 4,
        "xi": 0.8,
        "mu": 0.2,
        "F_n_axi_N": 0.0,
        "contact_pressure_Pa": 1.0e6,
    }
    base.update(kwargs)
    return HelicalLayerSpec(**base)


# --------------------------------------------------------------------------- #
# Papailiou slip cap                                                           #
# --------------------------------------------------------------------------- #
def test_slip_cap_and_onset_vanish_in_their_limits():
    layer = _spec()
    assert slip_onset_curvature(layer) > 0.0
    # A wire laid along the axis carries no capstan friction.
    axial = _spec(lay_angle_rad=0.0)
    np.testing.assert_array_equal(wire_slip_stress_cap(axial, 0.0), 0.0)
    assert slip_onset_curvature(axial) == 0.0
    # No clamping or no stick stiffness: slip from the first increment.
    assert slip_onset_curvature(layer, contact_pressure_Pa=0.0) == 0.0
    np.testing.assert_array_equal(
        wire_slip_stress_cap(layer, 0.0, contact_pressure_Pa=0.0), 0.0
    )
    assert full_slip_curvature(layer, contact_pressure_Pa=0.0) == 0.0
    assert slip_onset_curvature(layer, stick_modulus_Pa=0.0) == 0.0
    assert full_slip_curvature(layer, stick_modulus_Pa=0.0) == 0.0
    with pytest.raises(CableInputError, match="bending direction must be finite"):
        full_slip_curvature(layer, phi_bend=math.nan)


def test_full_slip_to_onset_ratio_is_the_capstan_factor():
    layer = _spec()
    s = abs(math.sin(layer.lay_angle_rad))
    # Bending towards a wire azimuth puts one wire on the extreme fibre.
    ratio = full_slip_curvature(layer, phi_bend=float(layer.phi_wires[0])) / (
        slip_onset_curvature(layer)
    )
    expected = math.expm1(layer.mu * s * math.pi / 2) / (layer.mu * s)
    assert ratio == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("fraction", [-0.1, 1.1, math.nan])
def test_direction_hold_fraction_must_lie_in_the_unit_interval(fraction):
    with pytest.raises(CableInputError, match="must lie in \\[0, 1\\]"):
        direction_threshold_curvature(_spec(), fraction)


def test_threshold_is_the_fraction_of_the_onset_curvature():
    layer = _spec()
    assert direction_threshold_curvature(layer, 0.25) == pytest.approx(
        0.25 * slip_onset_curvature(layer), rel=1e-15
    )


def test_wire_slip_history_input_rules():
    layer = _spec()
    kx = np.linspace(0.0, 0.05, 6)
    ky = np.zeros_like(kx)
    with pytest.raises(CableInputError, match="initial bending angle must be finite"):
        wire_slip_history(layer, kx, ky, initial_bending_angle=math.inf)
    with pytest.raises(CableInputError, match="nonnegative pressure"):
        wire_slip_history(layer, kx, ky, contact_pressure_Pa=np.full(6, -1.0))
    with pytest.raises(CableInputError, match="finite matching nonempty"):
        wire_slip_history(layer, kx, ky[:-1])


# --------------------------------------------------------------------------- #
# Three-core support shape                                                     #
# --------------------------------------------------------------------------- #
_RING = {
    "k_core": 1.0e11,
    "k_filler": 2.0e10,
    "theta_core_rad": math.radians(80.0),
    "EI_ring": 1.0e2,
    "r_bedding_m": 0.0755,
    "n_wires": 90,
    "max_harmonic": 12,
}


def test_three_core_shape_is_independent_of_the_pressure_level():
    # The shape is normalised to mean one; zero pressure is admissible.
    zero = solve_three_core(0.0, **_RING)
    loaded = solve_three_core(5.0e6, **_RING)
    assert float(np.mean(zero.shape_g)) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(zero.shape_g, loaded.shape_g, rtol=1e-9)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"theta_core_rad": 0.0}, "between zero and 2pi"),
        ({"theta_core_rad": 7.0}, "between zero and 2pi"),
        ({"phi_cores": (0.0, 2.0)}, "three finite core-centre angles"),
        ({"phi_cores": (0.0, 2.0, math.nan)}, "three finite core-centre angles"),
    ],
)
def test_three_core_input_rules(kwargs, message):
    with pytest.raises(CableInputError, match=message):
        solve_three_core(1.0e6, **{**_RING, **kwargs})


# --------------------------------------------------------------------------- #
# Section state                                                                #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def pipe():
    return cx.load_cable(cx.saevik2011_yaml_path(), geometry_override=None)


def _state(cable, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return solve_section_state(
            cable, tension_N=1e5, contact_model="bounded", **kwargs
        )


def test_isothermal_profile_equals_uniform_layer_temperature(pipe):
    radii = np.array([0.0, 1.0])
    profile = SimpleNamespace(r_m=radii, T_C=np.full(2, 40.0))
    uniform = _state(pipe, temperature_C=40.0)
    from_profile = _state(pipe, thermal=profile)
    for name, layer in uniform.layers.items():
        assert from_profile.layers[name].axial_force_N == pytest.approx(
            layer.axial_force_N, rel=1e-12, abs=1e-9
        )


def test_section_state_input_rules(pipe):
    with pytest.raises(CableSchemaError, match="contact_model must be"):
        solve_section_state(pipe, contact_model="solved")
    profile = SimpleNamespace(r_m=np.array([0.0, 1.0]), T_C=np.full(2, 40.0))
    with pytest.raises(CableSchemaError, match="not both"):
        solve_section_state(pipe, thermal=profile, temperature_C=40.0)
    unordered = SimpleNamespace(r_m=np.array([1.0, 0.0]), T_C=np.full(2, 40.0))
    with pytest.raises(CableSchemaError, match="finite ordered radial"):
        solve_section_state(pipe, thermal=unordered)


# --------------------------------------------------------------------------- #
# Axial rigidity                                                               #
# --------------------------------------------------------------------------- #
def test_metallic_rigidity_leaves_out_a_polymer_shell():
    cable = cx.load_cable(cx.cigre_tb880_yaml_path(), geometry_override=None)
    shell = cable.get_layer("cu_metallic_sheath")
    plastic = cable.model_copy(
        update={
            "layers": tuple(
                layer.model_copy(update={"material": "hdpe"})
                if layer is shell
                else layer
                for layer in cable.layers
            )
        }
    )
    full = axial_stiffness_by_layer(plastic)
    metals = axial_stiffness_by_layer(plastic, metals_only=True)
    assert "cu_metallic_sheath" in full and "cu_metallic_sheath" not in metals
    assert set(metals) <= set(full)
    for name, value in metals.items():
        assert value == full[name]
    # The copper sheath itself counts as metal.
    assert "cu_metallic_sheath" in axial_stiffness_by_layer(cable, metals_only=True)


def test_a_layer_without_modulus_carries_no_axial_or_bending_load():
    cable = cx.load_cable(cx.cigre_tb880_yaml_path(), geometry_override=None)
    target = next(
        layer
        for layer in cable.layers
        if layer.layer_type == "cylindrical"
        and layer.structural
        and (layer.r_inner_m or 0.0) > 0.0
    )
    cx.register_material(
        "user_no_modulus_bedding", {"rho_kg_m3": (950.0, "[ModelAssumption] test")}
    )
    try:
        bare = cable.model_copy(
            update={
                "layers": tuple(
                    layer.model_copy(update={"material": "user_no_modulus_bedding"})
                    if layer is target
                    else layer
                    for layer in cable.layers
                )
            }
        )
        full = axial_stiffness_by_layer(cable)
        reduced = axial_stiffness_by_layer(bare)
        assert target.name in full and target.name not in reduced
        assert all(reduced[name] == full[name] for name in reduced)
        assert ei_min_slip(bare) < ei_min_slip(cable)
    finally:
        cx.unregister_material("user_no_modulus_bedding")
