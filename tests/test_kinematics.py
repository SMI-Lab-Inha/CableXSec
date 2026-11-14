# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""The kinematics of the cable over a curvature history (analysis.kinematics).

The slip of a wire along its helix comes from the friction law of the stress
route: zero in stick, the full-slip kinematic slip ``r^2 cos^2 a / sin a``
times the curvature in frictionless slip (Kraincanic and Kebadze), continuous
along the helix, and with the stress the same law gives. The distributed law
gives its own slip field.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from _bundled import bundled

import cablexsec as cx
from cablexsec._numba import HAVE_NUMBA
from cablexsec.analysis.fatigue import wire_stress_field
from cablexsec.analysis.kinematics import (
    armour_transverse_model,
    kinematic_history,
    kinematic_slip_amplitude,
    periodic_interpolate,
    slip_along_helix,
)
from cablexsec.cable_adapters import cable_to_mkappa_specs, structural_armour_layers
from cablexsec.exceptions import CableInputError
from cablexsec.materials import library
from cablexsec.mechanics.distributed_friction import distributed_friction_history
from cablexsec.mechanics.wire_stress import local_bending_fibre_stress

#: Long friction histories run pure Python without numba (minutes per case).
needs_numba = pytest.mark.skipif(
    not HAVE_NUMBA, reason="long friction histories need the compiled kernels"
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

CABLE = bundled("iea_task49_66kv")
THREE_CORE = bundled("inter_array_three_core_33kv")
SAMPLES = 240
CYCLE = 0.06 * np.sin(np.linspace(0.0, 2.5 * np.pi, SAMPLES))


def _held(slipping: np.ndarray) -> np.ndarray:
    """Stations in a stick zone: no change there and at both neighbours."""
    return ~slipping & ~np.roll(slipping, 1, axis=0) & ~np.roll(slipping, -1, axis=0)


def test_kraincanic_kebadze_amplitude() -> None:
    r, a, k = 0.08, math.radians(12.0), 0.05
    expected = r**2 * math.cos(a) ** 2 / math.sin(a) * k
    assert kinematic_slip_amplitude(r, a, k) == pytest.approx(expected, rel=1e-15)
    assert kinematic_slip_amplitude(r, -a, -k) == pytest.approx(expected, rel=1e-15)
    with pytest.raises(CableInputError):
        kinematic_slip_amplitude(r, 0.0, k)


@pytest.mark.parametrize("law", ["direction_tied", "progressive", "distributed"])
def test_frictionless_slip_is_the_full_slip_kinematic_slip(law: str) -> None:
    kx = CYCLE
    ky = 0.3 * CYCLE
    history = kinematic_history(CABLE, kx, ky, mu=0.0, friction_model=law)  # type: ignore[arg-type]
    for layer in history.layers:
        amplitude = kinematic_slip_amplitude(layer.r_helix_m, layer.lay_angle_rad, 1.0)
        phi = layer.azimuth_rad[:, None]
        expected = (
            -layer.lay_sign
            * amplitude
            * (np.sin(phi) * kx[None, :] - np.cos(phi) * ky[None, :])
        )
        # Trapezoids over the stations: relative error h^2 / 12.
        step = 2 * math.pi / layer.stations
        error = np.abs(layer.slip_m - expected).max() / np.abs(expected).max()
        assert error < step**2 / 12 * 1.05 + 1e-12, (layer.name, error)
        assert np.all(layer.stress_Pa is not None)
        assert layer.slipping[:, 1:].mean() > 0.95


def test_slip_is_zero_in_stick() -> None:
    small = 2e-5 * np.sin(np.linspace(0.0, 2 * np.pi, 60))
    history = kinematic_history(CABLE, small, np.zeros(60), tension_N=100e3)
    for layer in history.layers:
        assert not layer.slipping.any()
        amplitude = kinematic_slip_amplitude(layer.r_helix_m, layer.lay_angle_rad, 2e-5)
        assert np.abs(layer.slip_m).max() < 1e-12 * amplitude


def test_stick_zones_stay_in_place_in_planar_bending() -> None:
    kappa = 0.004 * np.sin(np.linspace(0.0, 2.5 * np.pi, 400))
    history = kinematic_history(CABLE, kappa, np.zeros(400), tension_N=200e3)
    for layer in history.layers:
        held = _held(layer.slipping)[:, 1:]
        assert held.any() and (~held).any()
        moved = np.abs(np.diff(layer.slip_m, axis=1))[held]
        amplitude = kinematic_slip_amplitude(
            layer.r_helix_m, layer.lay_angle_rad, 0.004
        )
        assert moved.max() < 1e-4 * amplitude
        # Slip while slipping, of the order of the kinematic slip.
        assert 0.3 * amplitude < np.abs(layer.slip_m).max() <= 1.001 * amplitude


@pytest.mark.parametrize("law", ["direction_tied", "progressive"])
def test_stress_is_that_of_the_stress_route(law: str) -> None:
    tension = np.full(SAMPLES, 80e3)
    history = kinematic_history(
        CABLE,
        CYCLE,
        0.2 * CYCLE,
        tension_N=tension,
        friction_model=law,  # type: ignore[arg-type]
    )
    layers = cx.cable_to_fatigue_layers(CABLE, contact_tension_N=80e3)
    ea = cx.compute_stiffness(CABLE).ea_fixed_N
    for kinematic in history.layers:
        layer = next(item for item in layers if item.name == kinematic.name)
        field = wire_stress_field(
            layer,
            CYCLE,
            0.2 * CYCLE,
            tension,
            ea,
            friction_model=law,  # type: ignore[arg-type]
        )
        assert kinematic.stress_Pa is not None
        np.testing.assert_allclose(
            kinematic.stress_Pa, field.mean_Pa, rtol=1e-12, atol=1e-6
        )
        # The plastic drive behind the slip is the one behind that stress.
        spec = layer.spec
        drive = spec.r_helix_m * (
            np.cos(spec.phi_wires)[:, None] * CYCLE[None, :]
            + np.sin(spec.phi_wires)[:, None] * 0.2 * CYCLE[None, :]
        )
        friction = field.mean_Pa - (
            spec.E_wire_Pa * math.cos(spec.lay_angle_rad) ** 2 * tension / ea
        )
        plastic = drive - friction / spec.k_f
        gamma = math.cos(spec.lay_angle_rad) ** 2 * plastic
        integral = np.diff(kinematic.slip_m, axis=0)
        sin_a = abs(math.sin(spec.lay_angle_rad))
        step = 2 * math.pi / spec.n_wires
        expected = (
            -kinematic.lay_sign
            * spec.r_helix_m
            / sin_a
            * 0.5
            * (gamma[:-1] + gamma[1:])
            * step
        )
        mean = (0.5 * (gamma + np.roll(gamma, -1, axis=0)) * step).sum(axis=0) / (
            2 * math.pi
        )
        expected = expected + kinematic.lay_sign * spec.r_helix_m / sin_a * mean * step
        np.testing.assert_allclose(
            integral, expected, atol=1e-9 * np.abs(kinematic.slip_m).max()
        )


@needs_numba
def test_distributed_slip_is_the_law_s_own_field() -> None:
    history = kinematic_history(
        CABLE, CYCLE, np.zeros(SAMPLES), tension_N=60e3, friction_model="distributed"
    )
    specs = cable_to_mkappa_specs(CABLE, 60e3)
    layers = structural_armour_layers(CABLE)
    spec, layer = specs[0], layers[0]
    kept = np.arange(0, SAMPLES, 7)
    own = distributed_friction_history(
        spec, CYCLE, np.zeros(SAMPLES), field_samples=kept
    )
    assert own.kinematic_field is not None
    unit = own.parameters["length_unit_m"]
    sign = 1 if layer.lay_direction == "right_hand" else -1
    kinematic = history.layers[0]
    np.testing.assert_allclose(
        kinematic.slip_m[:, kept], sign * unit * own.kinematic_field[:, 1, :].T
    )
    # The stress at the nodes is the mean of the two segments a node joins
    # (with the tension); the law's wire stress samples the segments.
    full = distributed_friction_history(
        spec, CYCLE, np.zeros(SAMPLES), record_field=True
    )
    assert full.field_normalised is not None
    segments = full.parameters["stress_unit_Pa"] * full.field_normalised.T
    tension = spec.E_wire_Pa * math.cos(spec.lay_angle_rad) ** 2 * 60e3
    tension /= cx.compute_stiffness(CABLE).ea_fixed_N
    nodes = 0.5 * (segments + np.roll(segments, 1, axis=0)) + tension
    np.testing.assert_allclose(kinematic.stress_Pa, nodes, rtol=1e-12)
    # Read at the wires, linear between the nodes, it is the law's wire
    # stress within a thousandth.
    wires = kinematic.along(kinematic.stress_Pa, spec.phi_wires)  # type: ignore[arg-type]
    total = full.stress_Pa + tension
    np.testing.assert_allclose(wires, total, atol=1e-3 * np.abs(total).max())
    # Recording the field leaves the results unchanged.
    np.testing.assert_array_equal(own.stress_Pa, full.stress_Pa)


def test_slip_is_continuous_along_the_helix() -> None:
    history = kinematic_history(CABLE, CYCLE, np.zeros(SAMPLES), tension_N=50e3)
    layer = history.layers[0]
    fine = np.linspace(0.0, 2.0 * math.pi, 20001)
    values = layer.along(layer.slip_m[:, 80], fine)
    # Periodic: the field closes on itself after a pitch.
    assert values[0] == pytest.approx(values[-1], abs=1e-15)
    largest_jump = np.abs(np.diff(values)).max()
    slope = np.abs(np.diff(layer.slip_m[:, 80])).max() / (2 * math.pi / layer.stations)
    assert largest_jump <= slope * (fine[1] - fine[0]) * 1.0001


def test_periodic_interpolation_and_slip_integral() -> None:
    stations = np.array([0.0, 1.0, 2.5, 4.0, 5.5])
    values = np.sin(stations)
    np.testing.assert_allclose(periodic_interpolate(stations, values, stations), values)
    middle = periodic_interpolate(stations, values, np.array([0.5, 6.0]))
    span = 2 * math.pi - 5.5
    assert middle[0] == pytest.approx(0.5 * (values[0] + values[1]))
    assert middle[1] == pytest.approx(values[4] + (values[0] - values[4]) * 0.5 / span)
    # A constant slip strain is a mean: it is taken out, and its share said.
    u, share = slip_along_helix(
        np.linspace(0, 2 * math.pi, 8, endpoint=False), np.ones(8), 0.1, 0.3
    )
    np.testing.assert_allclose(u, 0.0, atol=1e-15)
    assert share[0] == pytest.approx(1.0)
    with pytest.raises(CableInputError):
        periodic_interpolate(np.array([0.0, 0.0]), np.zeros(2), 0.0)


def test_section_and_kept_samples() -> None:
    tension = np.linspace(20e3, 80e3, SAMPLES)
    kept = np.array([0, 10, 120, SAMPLES - 1])
    time = np.linspace(0.0, 12.0, SAMPLES)
    history = kinematic_history(
        CABLE, CYCLE, np.zeros(SAMPLES), time_s=time, tension_N=tension, samples=kept
    )
    whole = kinematic_history(
        CABLE, CYCLE, np.zeros(SAMPLES), time_s=time, tension_N=tension
    )
    np.testing.assert_array_equal(history.samples, kept)
    np.testing.assert_allclose(history.time_s, time[kept])
    np.testing.assert_allclose(
        history.axial_strain, tension[kept] / cx.compute_stiffness(CABLE).ea_fixed_N
    )
    np.testing.assert_array_equal(history.twist_1_m, 0.0)
    for part, full in zip(history.layers, whole.layers, strict=True):
        np.testing.assert_allclose(part.slip_m, full.slip_m[:, kept], atol=1e-15)
        np.testing.assert_array_equal(part.slipping, full.slipping[:, kept])
    with pytest.raises(CableInputError):
        kinematic_history(CABLE, CYCLE, np.zeros(SAMPLES), samples=[5, 2])
    with pytest.raises(CableInputError):
        kinematic_history(
            CABLE,
            CYCLE,
            np.zeros(SAMPLES),
            tension_N=1.0,
            axial_strain=np.zeros(SAMPLES),
        )


def test_utilisation_is_the_largest_fibre_over_yield() -> None:
    history = kinematic_history(CABLE, CYCLE, np.zeros(SAMPLES), tension_N=80e3)
    layer = history.layers[0]
    assert layer.utilisation is not None and layer.bending_Pa is not None
    assert layer.stress_Pa is not None
    material = CABLE.get_layer(layer.name).material
    limit = library.get_property(material, "sigma_y_Pa")
    largest = np.abs(layer.stress_Pa) + layer.bending_Pa
    np.testing.assert_allclose(layer.utilisation, largest / limit, rtol=1e-12)
    # The local bending is that of the stress route at the outer fibres.
    spec = cable_to_mkappa_specs(CABLE, 80e3)[0]
    diameter = 2 * math.sqrt(spec.A_wire_m2 / math.pi)
    radial = local_bending_fibre_stress(
        CYCLE[None, :],
        0.0,
        spec.phi_wires[:, None],
        spec.E_wire_Pa,
        diameter,
        spec.lay_angle_rad,
    )
    tangential = local_bending_fibre_stress(
        CYCLE[None, :],
        0.0,
        spec.phi_wires[:, None],
        spec.E_wire_Pa,
        diameter,
        spec.lay_angle_rad,
        math.pi / 2,
    )
    np.testing.assert_allclose(layer.bending_Pa, np.hypot(radial, tangential))


def test_core_slip_with_the_core_lay_stage() -> None:
    history = kinematic_history(
        CABLE, CYCLE, np.zeros(SAMPLES), tension_N=50e3, core_friction=True
    )
    cores = history.cores
    assert cores is not None and cores.kind == "cores" and cores.bodies == 3
    assert np.abs(cores.slip_m).max() > 0.0
    amplitude = kinematic_slip_amplitude(cores.r_helix_m, cores.lay_angle_rad, 0.06)
    assert np.abs(cores.slip_m).max() <= 1.001 * amplitude
    assert history.fields[-1] is cores
    # A cable without a core lay has no core stage.
    with pytest.raises(cx.CableSchemaError):
        kinematic_history(
            THREE_CORE.model_copy(update={"core_lay": None}),
            CYCLE,
            np.zeros(SAMPLES),
            core_friction=True,
        )


def test_transverse_slip_when_that_model_is_on() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        history = kinematic_history(
            CABLE, 4 * CYCLE, np.zeros(SAMPLES), tension_N=0.0, transverse=True
        )
    for layer in history.layers:
        assert layer.transverse_m is not None
        assert layer.transverse_m.shape == layer.slip_m.shape
    plain = kinematic_history(CABLE, CYCLE, np.zeros(SAMPLES))
    assert all(layer.transverse_m is None for layer in plain.layers)
    spec = cable_to_mkappa_specs(CABLE, 0.0)[0]
    model = armour_transverse_model(
        CABLE, structural_armour_layers(CABLE)[0], spec, 0.0
    )
    assert model.capacity_N_per_m >= 0.0 and model.G_Pa < model.E_Pa
