# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Distinct section-face and crossing-interface forces in local fatigue."""

from dataclasses import replace

import numpy as np
import pytest

from cablexsec.analysis.fatigue import (
    FatigueLayer,
    _fretting_amplitudes_um,
    compute_fatigue,
)
from cablexsec.materials.sn_curves import SNcurve, get_sn_curve
from cablexsec.mechanics.mindlin_contact import geometric_slip_factor
from cablexsec.mechanics.mkappa import HelicalLayerSpec


def layer():
    spec = HelicalLayerSpec(
        n_wires=8,
        r_helix_m=0.05,
        lay_angle_rad=0.3,
        E_wire_Pa=200e9,
        A_wire_m2=2e-5,
        xi=0.8,
        mu=0.3,
        F_n_axi_N=100.0,
        contact_pressure_Pa=3e5,
        normal_load_is_line=True,
    )
    return FatigueLayer(
        spec=spec,
        sn_bulk=get_sn_curve("dnv_rpc203_t_curve"),
        sigma_UTS_Pa=1.4e9,
        sn_fretting=SNcurve(
            name="test-only", stress_unit="um", m1=2.0, log_C1=10.0, threshold=0.0
        ),
        K_t_contact_N_per_m=1e7,
        fretting_mu=0.2,
        crossing_density_per_m=20.0,
        contact_line_load_N_per_m=400.0,
        transverse_ratio=1.0,
        transverse_map="legacy",
    )


def test_fretting_does_not_reuse_section_face_pressure_shape():
    reference = layer()
    changed = replace(
        reference, spec=replace(reference.spec, shape_g=np.linspace(0.1, 1.9, 8))
    )
    kappa = np.array([0.0, 0.01, -0.01, 0.01, -0.01])
    a = _fretting_amplitudes_um(reference, kappa, np.zeros_like(kappa), None)
    b = _fretting_amplitudes_um(changed, kappa, np.zeros_like(kappa), None)
    np.testing.assert_array_equal(a, b)


def test_crossing_pressure_and_stiffness_follow_the_same_hertz_force_scale():
    reference = layer()
    shape = np.linspace(0.0, 2.0, reference.spec.n_wires)
    sample = replace(reference, contact_shape_g=shape)
    amplitude = 0.002
    kappa = amplitude * np.array([0.0, 1.0, -1.0, 1.0, -1.0])
    actual = _fretting_amplitudes_um(sample, kappa, np.zeros_like(kappa), None)
    imposed = (
        geometric_slip_factor(sample.spec.r_helix_m, sample.spec.lay_angle_rad)
        * amplitude
    )
    force = sample.contact_line_load_N_per_m / sample.crossing_density_per_m
    elastic = sample.fretting_mu * force / sample.K_t_contact_N_per_m * shape ** (2 / 3)
    expected = np.maximum(imposed - elastic, 0.0) * 1e6
    expected[shape == 0] = 0.0  # No closed contact to accumulate fretting damage.
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


def test_critical_wire_core_location_is_geometric_not_a_pressure_indicator():
    sample = layer()
    kappa = 0.005 * np.sin(np.linspace(0, 6 * np.pi, 181))
    unknown = compute_fatigue([sample], kappa, np.zeros_like(kappa))
    assert unknown.above_core is None
    for value in (False, True):
        known = replace(sample, above_core_mask=np.full(sample.spec.n_wires, value))
        result = compute_fatigue([known], kappa, np.zeros_like(kappa))
        assert result.above_core is value


@pytest.mark.parametrize(
    "shape", [np.zeros(8), np.ones(7), np.full(8, np.nan), np.arange(8) - 2]
)
def test_invalid_shared_interface_distribution_is_rejected(shape):
    with pytest.raises(ValueError, match="contact_shape_g"):
        replace(layer(), contact_shape_g=shape)


def test_shared_interface_distribution_is_copied_and_read_only():
    shape = np.ones(8)
    sample = replace(layer(), contact_shape_g=shape)
    shape[0] = 4.0
    assert sample.contact_shape_g[0] == 1.0
    with pytest.raises(ValueError):
        sample.contact_shape_g[0] = 2.0
