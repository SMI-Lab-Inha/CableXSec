# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Sheath counting retains residual half cycles just like wire fatigue."""

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.materials.sn_curves import get_sn_curve


def test_two_point_sheath_excursion_retains_half_cycle_damage():
    cable = cx.load_cable(cx.export_three_core_yaml_path(), geometry_override=None)
    slope = cx.compute_shell_stress(cable, "al_metallic_sheath", 1, 0).sigma_peak_Pa
    result = cx.compute_shell_fatigue(
        cable, "al_metallic_sheath", np.array([-0.01, 0.01])
    )
    curve = get_sn_curve("en1999_1_3_aluminium")
    assert result.n_cycles == 0.5
    assert result.D_miner > 0
    assert result.D_miner == pytest.approx(
        0.5 / curve.cycles_to_failure(0.02 * slope / 1e6)
    )


@pytest.mark.parametrize("history", [[], [0.01], [0.01, 0.01], [0.01] * 4])
def test_no_excursion_has_no_cycles(history):
    cable = cx.load_cable(cx.export_three_core_yaml_path(), geometry_override=None)
    result = cx.compute_shell_fatigue(cable, "al_metallic_sheath", np.array(history))
    assert result.n_cycles == 0
    assert result.D_miner == 0


def test_sheath_history_rejects_multiple_dimensions():
    cable = cx.load_cable(cx.export_three_core_yaml_path(), geometry_override=None)
    with pytest.raises(cx.CableSchemaError, match="one-dimensional"):
        cx.compute_shell_fatigue(cable, "al_metallic_sheath", np.zeros((2, 2)))
