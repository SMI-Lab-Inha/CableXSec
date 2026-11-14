# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Helical tapes: geometry and strain checks, fatigue options and routes."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.tape import compute_tape_fatigue, tape_geometry, tape_strain
from cablexsec.exceptions import CableInputError, CableSchemaError

K = 0.05 * np.sin(np.linspace(0.0, 4.0 * math.pi, 121))


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(scope="module")
def thies():
    return cx.load_cable(cx.thies2016_yaml_path(), geometry_override=None)


def _layer(cable, name, **update):
    layers = [
        layer.model_copy(update=update) if layer.name == name else layer
        for layer in cable.layers
    ]
    return cable.model_copy(update={"layers": layers})


def _run(cable, layer="cu_tape", **kwargs):
    kwargs.setdefault("state", "stuck")
    kwargs.setdefault("core_state", "free")
    kwargs.setdefault("n_stations", 4)
    return compute_tape_fatigue(cable, layer, K, **kwargs)


def test_geometry_checks(thies):
    winding = thies.get_layer("cu_tape").tape
    with pytest.raises(CableInputError, match="radius"):
        tape_geometry(0.0, 1e-4, winding)
    with pytest.raises(CableInputError, match="thickness"):
        tape_geometry(0.02, math.nan, winding)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"state": "glued"}, "tape state"),
        ({"path": "spiral"}, "tape path"),
        ({"hoop_ratio": 2.0}, "hoop_ratio"),
        ({"axis_strain": np.zeros((1, 3))}, "shape"),
        (
            {
                "state": "frictional",
                "contact_pressure_Pa": -1.0,
                "E_Pa": 1.1e11,
                "mu": 0.2,
            },
            "non-negative",
        ),
    ],
)
def test_strain_checks(thies, kwargs, match):
    geometry = tape_geometry(0.02, 1e-4, thies.get_layer("cu_tape").tape)
    args = {
        "axis_strain": np.zeros((1, 4)),
        "gradient_radial_per_m": np.zeros((1, 4)),
        "gradient_tangential_per_m": np.zeros((1, 4)),
        "theta_rad": np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False),
        "state": "stuck",
        "hoop_ratio": -0.4,
    }
    args.update(kwargs)
    with pytest.raises(CableInputError, match=match):
        tape_strain(geometry, **args)


@pytest.mark.parametrize(
    ("layer", "kwargs", "match"),
    [
        ("absent", {}, "absent"),
        ("armour", {}, "a tape is a cylindrical"),
        ("cu_tape", {"correction": "goodman"}, "mean-stress correction"),
        ("cu_tape", {"n_theta": 3}, "n_theta"),
        ("cu_tape", {"n_theta": True}, "n_theta"),
        ("cu_tape", {"kappa_y": K[:-1]}, "equal 1-D curvature"),
        ("cu_tape", {"axial_strain": K[:-1]}, "axial strain"),
    ],
)
def test_fatigue_checks(thies, layer, kwargs, match):
    with pytest.raises((CableInputError, CableSchemaError), match=match):
        _run(thies, layer, **kwargs)


def test_layer_requirements(thies):
    with pytest.raises(CableSchemaError, match="no complete radii"):
        _run(_layer(thies, "cu_tape", r_inner_m=None))
    with pytest.raises(CableSchemaError, match="no strain_life constants"):
        _run(_layer(thies, "cu_tape", strain_life=""))


def test_axial_strain_and_frictional_core(thies):
    plain = _run(thies)
    pulled = _run(thies, axial_strain=2e-4 * (1.0 + K / 0.05))
    assert pulled.D_miner > plain.D_miner
    frictional = _run(thies, core_state="frictional", n_stations=3)
    assert frictional.strain_amplitude_max > 0


def test_frictional_tape_and_bound_constants(thies):
    result = _run(
        thies,
        state="frictional",
        contact_pressure_Pa=1e5,
        mu=0.2,
        strain_life="copper_annealed_mus_derived",
    )
    limitations = " ".join(result.provenance.limitations)
    assert "friction-setting pressure 100000 Pa and mu 0.2" in limitations
    assert "are derived: a bound" in limitations


def test_tape_outside_the_cores(thies):
    winding = thies.get_layer("cu_tape").tape
    wrapped = _layer(thies, "bedding", tape=winding, strain_life="copper_poon2022")
    result = compute_tape_fatigue(wrapped, "bedding", K, state="stuck")
    assert result.strain_amplitude_max > 0
    with pytest.raises(CableInputError, match="tape of a power core"):
        compute_tape_fatigue(wrapped, "bedding", K, state="stuck", core_state="free")
