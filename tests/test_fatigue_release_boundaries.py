# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Engine-level shape guards also apply to callers bypassing the cable adapter."""

import numpy as np
import pytest

from cablexsec.analysis.fatigue import compute_fatigue


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"n_fibres": 3}, "n_fibres"),
        ({"n_fibres": True}, "n_fibres"),
        ({"tension_N": np.zeros(3)}, "same shape"),
        ({"axial_strain": np.zeros(3)}, "axial_strain"),
        ({"axial_strain": np.zeros(2), "tension_N": np.zeros(2)}, "not both"),
    ],
)
def test_inconsistent_engine_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        compute_fatigue([], np.zeros(2), np.zeros(2), **kwargs)


@pytest.mark.parametrize(
    "name",
    [
        "thermal_stress_Pa",
        "contact_pressure_Pa",
        "contact_force_N",
        "normal_stress_Pa",
        "wire_modulus_Pa",
        "stick_modulus_Pa",
    ],
)
@pytest.mark.parametrize(
    "history,match", [(np.zeros(3), "shape"), (np.array([0, np.nan]), "finite")]
)
def test_bad_layer_histories(name, history, match):
    with pytest.raises(ValueError, match=match):
        compute_fatigue([], np.zeros(2), np.zeros(2), **{name: {"layer": history}})


@pytest.mark.parametrize(
    "x,y,match",
    [
        ([0, 1], [0], "same shape"),
        ([0], [0], "at least 2"),
        ([0, np.nan], [0, 0], "finite"),
    ],
)
def test_bad_curvature_histories(x, y, match):
    with pytest.raises(ValueError, match=match):
        compute_fatigue([], np.array(x), np.array(y))
