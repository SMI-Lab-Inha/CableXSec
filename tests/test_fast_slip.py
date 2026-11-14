# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""The compiled friction-stress kernel must reproduce wire_slip_history."""

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.cable_adapters import cable_to_fatigue_layers
from cablexsec.mechanics.fast_slip import wire_friction_stress
from cablexsec.mechanics.papailiou import wire_slip_history


def _layers():
    cable = cx.load_cable(cx.export_three_core_yaml_path(), geometry_override=None)
    return [layer.spec for layer in cable_to_fatigue_layers(cable, contact_tension_N=0)]


def _histories(rng, n=4000):
    t = np.linspace(0, 400, n)
    kx = 0.03 * np.sin(2 * np.pi * 0.05 * t) + 0.01 * rng.standard_normal(n)
    ky = 0.02 * np.cos(2 * np.pi * 0.031 * t) + 0.005 * rng.standard_normal(n)
    kx[100:120] = 0.0
    ky[100:120] = 0.0  # held zero curvature keeps the last bending direction
    return kx, ky


@pytest.mark.parametrize("varying_pressure", [False, True])
def test_matches_reference(varying_pressure):
    rng = np.random.default_rng(11)
    kx, ky = _histories(rng)
    for spec in _layers():
        pressure = None
        if varying_pressure:
            pressure = spec.contact_pressure_Pa * (
                1 + 0.3 * np.sin(np.linspace(0, 20, kx.size))
            )
        ref = wire_slip_history(spec, kx, ky, contact_pressure_Pa=pressure).stress_Pa
        fast = wire_friction_stress(spec, kx, ky, contact_pressure_Pa=pressure)
        scale = np.max(np.abs(ref))
        assert np.max(np.abs(fast - ref)) <= 1e-9 * scale
