# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Per-core wire screens: input checks, options and the result summary."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.exceptions import CableInputError, CableSchemaError

K = 0.05 * np.sin(np.linspace(0.0, 4.0 * math.pi, 161))
SCREEN = "cu_screen_wires"


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


def _run(cable, **kwargs):
    kwargs.setdefault("core_state", "free")
    kwargs.setdefault("n_stations", 2)
    return cx.compute_core_screen_fatigue(cable, K, **kwargs)


def test_critical_summary_matches_the_arrays(thies):
    result = _run(thies, duration_s=60.0)
    summary = result.critical()
    assert summary["D_bulk_max"] == pytest.approx(float(np.max(result.D_bulk)))
    fretting = max(
        float(np.max(face.D_fretting_per_year))
        for faces in result.faces
        for face in faces.values()
    )
    assert summary["D_fretting_per_year_max"] == pytest.approx(fretting)
    assert summary["fretting_face"] in ("inner_face", "outer_face")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"layer_name": "absent"}, "named 'absent'"),
        ({"duration_s": 0.0}, "duration_s"),
        ({"time_fraction": -0.1}, "time_fraction"),
        ({"tension_N": np.zeros(K.size), "axial_strain": np.zeros(K.size)}, "not both"),
        ({"mu_inner": -0.1}, "friction coefficients"),
        (
            {"strain_life": "copper_poon2022", "correction": "goodman"},
            "mean-stress correction",
        ),
        ({"core_state": "frictional", "n_stations": 100000}, "n_stations must lie"),
    ],
)
def test_input_checks(thies, kwargs, match):
    with pytest.raises((CableInputError, CableSchemaError), match=match):
        _run(thies, **kwargs)


def test_screen_layer_requirements(thies):
    screen = thies.get_layer(SCREEN)
    ring = screen.strand_layers[0]
    two = _layer(thies, SCREEN, strand_layers=[ring, ring])
    with pytest.raises(CableSchemaError, match="exactly one resolved wire ring"):
        _run(two)
    with pytest.raises(CableSchemaError, match="has no bulk_sn curve"):
        _run(_layer(thies, SCREEN, bulk_sn=""))
    with pytest.raises(CableSchemaError, match="unavailable"):
        _run(_layer(thies, SCREEN, bulk_sn="perier_2009_steel"))


def test_axial_loading_routes(thies):
    plain = _run(thies)
    pulled = _run(thies, tension_N=2e5 * (1.0 + K / 0.05))
    strained = _run(thies, axial_strain=1e-3 * (1.0 + K / 0.05))
    assert np.max(pulled.D_bulk) > np.max(plain.D_bulk)
    assert np.max(strained.D_bulk) > np.max(plain.D_bulk)


def test_frictional_core_and_named_options(thies):
    result = _run(
        thies,
        core_state="frictional",
        n_stations=3,
        knockdown="copper_nishioka_hirakawa",
        strain_life="copper_poon2022",
        correction="morrow",
    )
    assert result.D_strain_life is not None
    limitations = " ".join(result.provenance.limitations)
    assert "stuck core" not in limitations and "free core" not in limitations


def test_faces_without_partners(thies):
    loose = _layer(
        _layer(thies, "swelling_tape_inner", r_outer_m=0.0189),
        "cu_tape",
        r_inner_m=0.0197,
    )
    result = _run(loose)
    assert result.p_inner_Pa == 0.0 and result.p_outer_Pa == 0.0
    assert all(not faces for faces in result.faces)
    limitations = " ".join(result.provenance.limitations)
    assert "no core layer touches the inner face" in limitations
    assert "no core layer touches the outer face" in limitations


def test_steel_partner_friction(thies):
    steel = _layer(thies, "cu_tape", material="galvanised_steel_armour")
    try:
        result = _run(steel)
    except CableInputError as error:
        assert "pass mu_inner/mu_outer" in str(error)
        result = _run(steel, mu_outer=0.3)
    assert result.mu_outer > 0
