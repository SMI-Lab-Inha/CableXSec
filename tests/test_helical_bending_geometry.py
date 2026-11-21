# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Stress gradients from a bent helical centreline, without a stress formula.

The cable section stays rigid. Positive kappa extends its positive x side.
Local bending strain is minus the change of centreline curvature dotted with
the fibre position in the radial/tangential material frame. This comparison
checks the small-increment kinematics, not three-core contact or free slip.
"""

import math

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.materials import library
from cablexsec.mechanics.core_lay import bending_factor, core_bending_components
from cablexsec.mechanics.wire_stress import (
    local_bending_amplitude,
    local_bending_fibre_stress,
    local_bending_stress,
)

pytestmark = pytest.mark.independent_method


def _curvature_from_path(alpha, phi, curvature):
    radius = 0.04
    q = math.tan(alpha) / radius
    # First/second derivatives at cable coordinate zero of a helix wrapped
    # on a circularly bent cable. No small-curvature approximation here.
    tangent = np.array(
        [
            -radius * q * math.sin(phi),
            radius * q * math.cos(phi),
            1 + radius * curvature * math.cos(phi),
        ]
    )
    second = np.array(
        [
            -curvature - radius * q * q * math.cos(phi),
            -radius * q * q * math.sin(phi),
            -2 * radius * curvature * q * math.sin(phi),
        ]
    )
    speed = np.linalg.norm(tangent)
    radial = np.array([math.cos(phi), math.sin(phi), 0.0])
    tangential = np.cross(tangent / speed, radial)
    return np.array([second @ radial, second @ tangential]) / speed**2


def _gradient(alpha, phi, step=1e-4):
    return -(
        _curvature_from_path(alpha, phi, step) - _curvature_from_path(alpha, phi, -step)
    ) / (2 * step)


@pytest.mark.parametrize("alpha", [-1.0, -0.5, -0.15, 0.0, 0.15, 0.5, 1.0])
@pytest.mark.parametrize("phi", [0.0, 0.31, math.pi / 2, 2.3, math.pi])
def test_local_stress_and_core_transfer_from_bent_helix(alpha, phi):
    direct = _gradient(alpha, phi)
    np.testing.assert_allclose(
        _gradient(alpha, phi, 5e-5), direct, rtol=1e-7, atol=5e-10
    )
    np.testing.assert_allclose(
        core_bending_components(alpha, phi, 1.0, 0.0), direct, rtol=1e-7, atol=5e-10
    )
    assert float(local_bending_stress(1.0, phi, 1.0, 2.0, alpha)) == pytest.approx(
        direct[0], rel=1e-7, abs=5e-10
    )
    assert float(local_bending_amplitude(1.0, phi, 1.0, 2.0, alpha)) == pytest.approx(
        np.linalg.norm(direct), rel=1e-7, abs=5e-10
    )
    # Material fibres remain fixed when the bending axis is rotated.
    bend = 0.72
    rotated = _gradient(alpha, phi - bend)
    for theta in (0, 0.37, math.pi / 2, 2.0, math.pi, 5.3):
        expected = rotated @ np.array([math.cos(theta), math.sin(theta)])
        calculated = local_bending_fibre_stress(
            math.cos(bend), math.sin(bend), phi, 1.0, 2.0, alpha, theta
        )
        assert float(calculated) == pytest.approx(expected, rel=1e-7, abs=5e-10)


@pytest.mark.parametrize("alpha", [0.0, 0.15, 0.5, 1.0])
def test_bending_energy_from_geometric_curvature_changes(alpha):
    # Unit EI, mean energy per cable length; reference wire length is 1/cos(a).
    gradients = np.array(
        [_gradient(alpha, p) for p in np.arange(240) * 2 * math.pi / 240]
    )
    direct = np.mean(np.sum(gradients**2, axis=1)) / math.cos(alpha)
    assert bending_factor(alpha) == pytest.approx(direct, rel=1e-8)


def test_armour_stress_extrema_include_all_fibres_not_only_the_radial_fibre():
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    kx, ky = 0.04, 0.03
    result = cx.compute_stress(cable, kx, ky, tension_N=1000)
    components = cx.compute_component_response(cable, kx, ky, tension_N=1000)
    for layer in cable.helical_layers:
        name = layer.name
        mean = (
            result.sigma_A_Pa[name] + result.sigma_f_Pa[name] + result.sigma_T_Pa[name]
        )
        angles = np.radians(result.phi_deg[name]) - math.atan2(ky, kx)
        e = library.get_property(layer.material, "E_Pa")
        kappa = math.hypot(kx, ky)
        # The curve curvature of the bent helix, whose normal component gains
        # the term of each wire's slip state, (sin^2 a / r) sigma_f / E.
        slip = (
            math.sin(layer.lay_angle_rad) ** 2
            / layer.r_helix_m
            * result.sigma_f_Pa[name]
            / (e * kappa)
        )
        direct = np.array(
            [
                np.linalg.norm(_gradient(layer.lay_angle_rad, p) + np.array([s, 0.0]))
                for p, s in zip(angles, slip, strict=True)
            ]
        )
        amplitude = direct * e * kappa * layer.d_wire_m / 2
        np.testing.assert_allclose(
            result.sigma_perimeter_min_Pa[name], mean - amplitude, rtol=1e-7, atol=0.01
        )
        np.testing.assert_allclose(
            result.sigma_perimeter_max_Pa[name], mean + amplitude, rtol=1e-7, atol=0.01
        )
        component = next(v for v in components.states if v.name == name)
        assert component.sigma_local_max_Pa == pytest.approx(
            max(mean + amplitude), rel=1e-7
        )
        assert np.all(
            result.sigma_vm_envelope_Pa[name] >= result.sigma_vm_Pa[name] - 1e-7
        )
