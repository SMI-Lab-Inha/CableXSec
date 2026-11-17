# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Utilisation bases of the capacity envelope and the copper plastic limit."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.capacity_check import (
    CAPACITY_BASES,
    ISO_13628_5_UTILISATION_FACTORS,
    CapacityEnvelope,
)
from cablexsec.exceptions import CableInputError

GRID = [0.0, 2.0e5]


@pytest.fixture(scope="module")
def cable() -> cx.CableData:
    return cx.load_cable(cx.iea_task49_66kv_yaml_path())


def _envelope(cable: cx.CableData, **kwargs: object) -> CapacityEnvelope:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cx.build_capacity_envelope(cable, GRID, **kwargs)


def test_published_umbilical_factors() -> None:
    assert CAPACITY_BASES == ("power_cable", "iso_13628_5")
    assert dict(ISO_13628_5_UTILISATION_FACTORS) == {
        "normal": 0.80,
        "installation": 1.00,
        "installation_displacement_controlled": 0.75,
        "abnormal": 1.00,
    }
    assert cx.ISO_13628_5_UTILISATION_FACTORS is ISO_13628_5_UTILISATION_FACTORS


def test_umbilical_basis_is_one_factor_for_every_element(cable) -> None:
    iso = _envelope(cable, basis="iso_13628_5")
    assert iso.basis == "iso_13628_5"
    assert iso.utilisation_factor == 0.80
    assert iso.component_utilisation_factor == 0.80
    # The same element utilisations with the factor set by hand on the
    # power-cable basis: the bases differ only in the factors.
    same = _envelope(cable, utilisation_factor=0.8, component_utilisation_factor=0.8)
    assert same.basis == "power_cable"
    for name in (
        "kappa_capacity_1_m",
        "tension_utilisation",
        "levels",
        "kappa_levels_1_m",
        "component_levels_1_m",
        "component_zero_utilisation",
        "lateral_utilisation",
    ):
        np.testing.assert_array_equal(getattr(iso, name), getattr(same, name))
    assert iso.component_governing == same.component_governing
    assert iso.provenance is not None
    assert any("ISO 13628-5" in c for c in iso.provenance.citations)
    assert not any("DNV-RP-F401" in c for c in iso.provenance.citations)
    assert iso.provenance.method.startswith("umbilical (ISO 13628-5) form")
    # Every phase factor applies to armour and components alike.
    installation = _envelope(
        cable,
        basis="iso_13628_5",
        utilisation_factor=ISO_13628_5_UTILISATION_FACTORS[
            "installation_displacement_controlled"
        ],
    )
    assert installation.utilisation_factor == 0.75
    assert installation.component_utilisation_factor == 0.75


def test_power_cable_basis_is_unchanged(cable) -> None:
    default = _envelope(cable)
    explicit = _envelope(cable, basis="power_cable")
    assert default.basis == "power_cable"
    assert default.utilisation_factor == 0.67
    assert default.component_utilisation_factor == 0.80
    np.testing.assert_array_equal(
        default.kappa_capacity_1_m, explicit.kappa_capacity_1_m
    )
    assert default.provenance is not None
    assert any("DNV-RP-F401" in c for c in default.provenance.citations)


def test_basis_inputs_are_checked(cable) -> None:
    with pytest.raises(CableInputError, match="basis must be"):
        _envelope(cable, basis="api_17j")
    with pytest.raises(CableInputError, match="one utilisation factor"):
        _envelope(cable, basis="iso_13628_5", component_utilisation_factor=0.67)
    imported = CapacityEnvelope.from_points(GRID, [0.3, 0.2], source="a report")
    assert imported.basis == "imported"
    with pytest.raises(CableInputError, match="basis must be"):
        CapacityEnvelope(
            tension_N=np.array(GRID),
            kappa_capacity_1_m=np.array([0.3, 0.2]),
            tension_utilisation=np.zeros(2),
            utilisation_factor=1.0,
            tension_kind="effective",
            basis="other",
        )


def test_copper_plastic_limit_governs_the_66kv_template(cable) -> None:
    # The conductor's 0.2 % proof stress (120 MPa), reported as its yielding
    # mode, governs the template's capacity (doc/template_66kv.md, 4.6).
    envelope = _envelope(cable)
    assert all(g == "component_limit" for g in envelope.governing)
    assert envelope.component_governing == (("cu_conductor", "yielding"),) * 2
    proof = cx.materials.library.get_property(
        cable.get_layer("cu_conductor").material, "sigma_y_Pa"
    )
    assert proof == pytest.approx(120.0e6)
