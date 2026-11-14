# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""A rotated physical support must rotate its solved pressure distribution."""

import math

import numpy as np
import pytest

from cablexsec.mechanics.contact_pressure import solve_three_core

pytestmark = pytest.mark.analytical_sanity


def test_trefoil_pressure_rotates_with_the_core_geometry():
    count = 360
    shift = 30
    rotation = 2 * math.pi * shift / count
    centres = tuple(2 * math.pi * k / 3 for k in range(3))
    kwargs = {
        "k_core": 1e11,
        "k_filler": 2e10,
        "theta_core_rad": math.radians(79.8),
        "EI_ring": 1e2,
        "r_bedding_m": 0.0775,
        "n_wires": count,
    }
    initial = solve_three_core(5e6, phi_cores=centres, **kwargs)
    rotated = solve_three_core(
        5e6, phi_cores=tuple(p + rotation for p in centres), **kwargs
    )
    np.testing.assert_allclose(
        rotated.shape_g, np.roll(initial.shape_g, shift), rtol=1e-9, atol=1e-10
    )


@pytest.mark.parametrize("order", [6, 18, 36])
def test_ring_and_support_energy_match_pressure_work(order):
    pressure, stiffness_core, stiffness_filler = 5e6, 1e11, 2e10
    radius, rigidity, width = 0.0775, 100.0, math.radians(79.8)
    phase = 0.19
    centres = tuple(phase + 2 * math.pi * k / 3 for k in range(3))
    result = solve_three_core(
        pressure,
        k_core=stiffness_core,
        k_filler=stiffness_filler,
        theta_core_rad=width,
        EI_ring=rigidity,
        r_bedding_m=radius,
        n_wires=360,
        phi_cores=centres,
        max_harmonic=order,
    )
    edges = sorted(
        {
            0.0,
            2 * math.pi,
            *((c + s * width / 2) % (2 * math.pi) for c in centres for s in (-1, 1)),
        }
    )
    gauss, weights = np.polynomial.legendre.leggauss(64)
    energy, work = 0.0, 0.0
    modes = np.asarray(result.harmonic_orders)
    for left, right in zip(edges, edges[1:], strict=False):
        phi = (right + left) / 2 + (right - left) / 2 * gauss
        quadrature = weights * (right - left) / 2 * radius
        inside = (
            np.min(
                np.abs(
                    (phi[:, None] - np.asarray(centres) + math.pi) % (2 * math.pi)
                    - math.pi
                ),
                axis=1,
            )
            < width / 2
        )
        support = np.where(inside, stiffness_core, stiffness_filler)
        basis = np.cos((phi[:, None] - result.phase_rad) * modes)
        displacement = basis @ result.displacement_coefficients_m
        ring_strain = (
            basis
            @ ((1 - modes * modes) * result.displacement_coefficients_m)
            / radius**2
        )
        energy += 0.5 * np.sum(
            quadrature * (support * displacement**2 + rigidity * ring_strain**2)
        )
        work += np.sum(quadrature * pressure * displacement)
    assert energy == pytest.approx(0.5 * work, rel=2e-12)
    assert (
        result.foundation_mean_pressure_Pa + result.ring_mean_reaction_Pa
        == pytest.approx(pressure, rel=2e-12)
    )
    assert result.balance_error < 2e-12


def test_uniform_ring_foundation_matches_its_analytical_displacement():
    pressure, stiffness, rigidity, radius = 3e5, 2e10, 100.0, 0.0775
    result = solve_three_core(
        pressure,
        k_core=stiffness,
        k_filler=stiffness,
        theta_core_rad=0.8,
        EI_ring=rigidity,
        r_bedding_m=radius,
        n_wires=99,
    )
    exact = pressure / (stiffness + rigidity / radius**4)
    np.testing.assert_allclose(
        result.radial_displacement(np.linspace(0, 2 * math.pi, 99)), exact, rtol=1e-12
    )


def test_adapter_core_phase_matches_component_geometry():
    import cablexsec as cx
    from cablexsec.cable_adapters import three_core_shape_for_layer

    cable = cx.load_cable(cx.export_three_core_yaml_path(), geometry_override=None)
    layer = cable.helical_layers[0]
    shape = three_core_shape_for_layer(cable, layer)
    assert shape is not None
    # Component geometry places a core on the positive x axis. Reflection
    # in that axis must preserve the normal reaction pattern.
    np.testing.assert_allclose(shape[1:], shape[:0:-1], rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "change",
    [
        {"q0_Pa": -1},
        {"k_core": float("nan")},
        {"EI_ring": 0},
        {"n_wires": 0},
        {"max_harmonic": 5},
        {"phi_cores": (0.0, 1.0, 2.0)},
    ],
)
def test_unsupported_ring_inputs_are_rejected(change):
    values = {
        "q0_Pa": 3e5,
        "k_core": 1e11,
        "k_filler": 2e10,
        "theta_core_rad": 0.8,
        "EI_ring": 100.0,
        "r_bedding_m": 0.0775,
        "n_wires": 99,
    }
    values.update(change)
    with pytest.raises(ValueError):
        solve_three_core(**values)
