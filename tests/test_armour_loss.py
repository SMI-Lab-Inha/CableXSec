# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Armour loss factor lambda2 ([IEC60287-1-1] 2.4) in the steady ladder and the network."""

from __future__ import annotations

import warnings

import pytest

import cablexsec as cx
from cablexsec.cables.schema import CableData
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.thermal import (
    _armour_radii,
    _ladder_geometry,
    compute_thermal,
    thermal_resistance,
)
from cablexsec.mechanics.thermal_network import thermal_network

RATED_A = 1093.0  # [IEATask49] Table 13, 630 mm2
MATERIAL = "copper_janocha2024"


@pytest.fixture(scope="module")
def t66() -> CableData:
    return cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)


def _steady(cable: CableData, lambda2: float) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute_thermal(
            cable, RATED_A, 630e-6, conductor_material=MATERIAL, lambda2=lambda2
        ).T_conductor_C


def test_zero_armour_loss_leaves_the_ladder_unchanged(t66: CableData) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = compute_thermal(t66, RATED_A, 630e-6, conductor_material=MATERIAL)
    assert _steady(t66, 0.0) == default.T_conductor_C


@pytest.mark.analytical_sanity
def test_armour_loss_adds_the_eq1_term(t66: CableData) -> None:
    """theta_c(lambda2) - theta_c(0) = n lambda2 W_c (T3 + T4), T3 outside the armour."""
    lambda2 = 0.15
    geo = _ladder_geometry(t66, None)
    radii = _armour_radii(t66)
    t3 = 0.0
    for annulus, inner in zip(geo.path, geo.in_t1, strict=True):
        if inner or annulus.per_core:
            continue
        share = sum(1.0 for r in radii if r <= annulus.layer.r_inner_m) / len(radii)
        t3 += share * thermal_resistance(
            annulus.layer.r_inner_m, annulus.layer.r_outer_m, annulus.layer.lambda_W_mK
        )
    net = thermal_network(t66, 630e-6, conductor_material=MATERIAL)
    w_c = net.conductor_loss_W_m(RATED_A, 90.0)
    t4 = t66.environment.T4_external_Km_W
    assert t4 is not None
    expected = geo.n * lambda2 * w_c * (t3 + t4)
    rise = _steady(t66, lambda2) - _steady(t66, 0.0)
    assert rise == pytest.approx(expected, rel=1e-10)
    assert rise > 0.0


@pytest.mark.analytical_sanity
def test_network_with_armour_loss_matches_the_ladder(t66: CableData) -> None:
    lambda2 = 0.15
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = compute_thermal(
            t66, RATED_A, 630e-6, conductor_material=MATERIAL, lambda2=lambda2
        )
    net = thermal_network(t66, 630e-6, conductor_material=MATERIAL, lambda2=lambda2)
    temps = net.steady_state(RATED_A, resistance_temperature_C=90.0)
    # The 0.10 mm copper tape screens make the chain stiff: agreement to
    # its conditioning.
    assert temps[0] == pytest.approx(ref.T_conductor_C, rel=1e-8)
    heat_in = net.total_heat_W_m(RATED_A, temps[0], resistance_temperature_C=90.0)
    heat_out = (temps[-1] - net.ambient_C) / net.external_resistance_Km_W
    # Thin copper tape screens make the chain stiff: agreement to conditioning.
    assert heat_out == pytest.approx(heat_in, rel=1e-8)
    assert heat_in == pytest.approx(ref.P_total_W_m, rel=1e-12)
    w_c = net.conductor_loss_W_m(RATED_A, 90.0)
    assert heat_in == pytest.approx(net.n_conductors * w_c * (1.0 + lambda2))
    # The armour heat raises the armour against the lossless case.
    plain = thermal_network(t66, 630e-6, conductor_material=MATERIAL)
    cold = plain.layer_temperatures(
        plain.steady_state(RATED_A, resistance_temperature_C=90.0)
    )
    hot = net.layer_temperatures(temps)
    assert float(hot["inner_armour"]) > float(cold["inner_armour"])


def test_armour_loss_input_is_checked(t66: CableData) -> None:
    with pytest.raises(CableInputError):
        thermal_network(t66, 630e-6, conductor_material=MATERIAL, lambda2=-0.1)
    with pytest.raises(CableInputError):
        compute_thermal(t66, RATED_A, 630e-6, conductor_material=MATERIAL, lambda2=-0.1)
