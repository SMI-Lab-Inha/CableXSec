# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Inner support reaction must not silently become outer armour pressure."""

import numpy as np
import pytest

import cablexsec as cx
from cablexsec import cable_adapters


@pytest.mark.parametrize(
    "path", [cx.inter_array_three_core_yaml_path, cx.export_three_core_yaml_path]
)
def test_default_specs_do_not_substitute_an_inner_reaction_for_armour_pressure(
    path, monkeypatch
):
    def wrong_face(*args, **kwargs):
        raise AssertionError("an inner-foundation reaction was used as armour pressure")

    monkeypatch.setattr(cable_adapters, "three_core_shape_for_layer", wrong_face)
    cable = cx.load_cable(path(), geometry_override=None)
    specs = cable_adapters.cable_to_mkappa_specs(cable)
    assert specs and all(spec.shape_g is None for spec in specs)
    # Exercise the connected public bending, stress and fatigue paths.
    kappa = 0.001 * np.sin(np.linspace(0, 6 * np.pi, 73))
    bending = cx.compute_mkappa(cable, kappa)
    stress = cx.compute_stress(cable, 0.001)
    fatigue = cx.compute_fatigue(cable, kappa, np.zeros_like(kappa))
    assert np.isfinite(bending.moment_Nm).all()
    assert np.isfinite(stress.critical_sigma_vm_Pa)
    assert all(np.isfinite(d).all() for d in fatigue.D_bulk_per_wire.values())


def test_foundation_diagnostic_remains_available_without_becoming_a_load():
    cable = cx.load_cable(cx.export_three_core_yaml_path(), geometry_override=None)
    shape = cable_adapters.three_core_shape_for_layer(cable, cable.helical_layers[0])
    assert shape is not None
    assert float(np.mean(shape)) == pytest.approx(1.0)
    # This large inner-face variation under uniform OUTER loading is precisely
    # why normalising the two pressures to the same mean is not a valid transfer.
    assert float(np.max(shape)) > 5.0
    assert cable_adapters.cable_to_mkappa_specs(cable)[0].shape_g is None
