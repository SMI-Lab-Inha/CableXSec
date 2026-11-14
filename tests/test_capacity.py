# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for the capacity envelope (§6.11): four collapse mechanisms, max-UF.

Analytical gates:
* N=0 governing mechanism is wire yield, with κ_max matching the closed-form
  inversion of the frozen stress model (< 0.5%).
* κ_max(N) monotone non-increasing over the tension domain N ≥ 0.
* Bird-caging active iff net armour compression (T_layer < 0).
* Signed thermal stress enters the extreme wire-yield comparison directly;
  zero thermal stress recovers the isothermal envelope exactly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cablexsec.analysis.capacity import (
    birdcage_buckling_detail,
    birdcage_critical_force,
    compute_capacity,
    wire_yield_curvature,
)
from cablexsec.cables.schema import CableData, CableMeta
from cablexsec.materials import library

# Test-taxonomy category.
pytestmark = pytest.mark.analytical_sanity


def _synthetic_cable() -> CableData:
    """A galvanised-steel armour layer restrained by an outboard HDPE sheath."""
    from cablexsec.cables.schema import _LAYER_ADAPTER  # noqa: PLC0415

    armour = {
        "name": "outer_armour",
        "layer_type": "helical",
        "material": "galvanised_steel_armour",
        "n_wires": 50,
        "d_wire_m": 0.005,
        "r_helix_m": 0.080,
        "lay_angle_deg": 20.0,
        "lay_direction": "right_hand",
        "pretension_Pa": 0.0,
    }
    sheath = {
        "name": "outer_sheath",
        "layer_type": "cylindrical",
        "material": "hdpe",
        "r_inner_m": 0.0825,
        "r_outer_m": 0.090,
    }
    return CableData(
        cable=CableMeta(name="synthetic armour cable", voltage_kV=70.0),
        layers=(
            _LAYER_ADAPTER.validate_python(armour),
            _LAYER_ADAPTER.validate_python(sheath),
        ),
    )


def _closed_form_yield_kappa_N0() -> float:
    """σ_y / [E cosα (1+sin²α) d/2] for the synthetic armour layer."""
    e = library.get_property("galvanised_steel_armour", "E_Pa")
    sy = library.get_property("galvanised_steel_armour", "sigma_y_Pa")
    a = math.radians(20.0)
    d = 0.005
    slope = e * math.cos(a) * (1.0 + math.sin(a) ** 2) * (d / 2.0)
    return sy / slope


# --------------------------------------------------------------------------- #
# Analytical gate: N=0 wire-yield inversion (< 0.5%)                           #
# --------------------------------------------------------------------------- #
def test_N0_yield_curvature_matches_closed_form() -> None:
    cable = _synthetic_cable()
    k = wire_yield_curvature(cable, 0.0, mu=0.20)
    assert k == pytest.approx(_closed_form_yield_kappa_N0(), rel=5e-3)


def test_N0_governing_mechanism_is_wire_yield() -> None:
    cable = _synthetic_cable()
    res = compute_capacity(cable, np.array([0.0]), component_limits=False)
    assert res.governing[0] == "wire_yield"
    k0 = _closed_form_yield_kappa_N0()
    assert res.kappa_max_1_m[0] == pytest.approx(k0, rel=5e-3)
    # No net compression, no tension stress: all tension-driven UFs are zero.
    assert res.uf_bird_caging[0] == 0.0
    assert np.isnan(res.uf_polymer_crush[0])
    assert res.unavailable_mechanisms == ("polymer_crush",)
    assert res.uf_conductor[0] == 0.0


def test_polymer_crush_requires_an_explicit_sourced_allowable() -> None:
    cable = _synthetic_cable()
    with pytest.raises(ValueError, match="must be supplied together"):
        compute_capacity(
            cable,
            np.array([1.0e5]),
            polymer_crush_allowable_Pa=25.0e6,
        )
    result = compute_capacity(
        cable,
        np.array([1.0e5]),
        polymer_crush_allowable_Pa=25.0e6,
        polymer_crush_source="project compression test TEST-01",
    )
    assert np.isfinite(result.uf_polymer_crush[0])
    assert result.uf_polymer_crush[0] > 0.0
    assert result.unavailable_mechanisms == ()


# --------------------------------------------------------------------------- #
# κ_max(N) monotone non-increasing over tension                               #
# --------------------------------------------------------------------------- #
def test_kappa_max_monotone_non_increasing_in_tension() -> None:
    cable = _synthetic_cable()
    tensions = np.linspace(0.0, 5.0e5, 11)
    res = compute_capacity(cable, tensions)
    dk = np.diff(res.kappa_max_1_m)
    assert np.all(dk <= 1e-12)  # non-increasing
    assert res.kappa_max_1_m[0] > res.kappa_max_1_m[-1]  # strictly drops with tension


# --------------------------------------------------------------------------- #
# Bird-caging active iff net armour compression                               #
# --------------------------------------------------------------------------- #
def test_birdcaging_active_only_under_net_compression() -> None:
    cable = _synthetic_cable()
    # Tension sweep: no compression -> bird-caging UF = 0 everywhere.
    res_t = compute_capacity(cable, np.array([1.0e5, 3.0e5]))
    assert np.all(res_t.uf_bird_caging == 0.0)
    # Compression sweep: bird-caging activates (UF > 0).
    res_c = compute_capacity(cable, np.array([-1.0e5, -3.0e5]))
    assert np.all(res_c.uf_bird_caging > 0.0)


def _saevik_thorsen_P1_P2(cable: CableData, layer) -> tuple[float, float]:
    """Hand-evaluate [SaevikThorsen2017] Eqs. 19/22/23 for a round armour wire."""
    e = library.get_property(layer.material, "E_Pa")
    nu = library.get_property(layer.material, "nu")
    a = layer.lay_angle_rad
    n_w, r, d = layer.n_wires, layer.r_helix_m, layer.d_wire_m
    ei = e * math.pi * d**4 / 64.0
    gj = ei / (1.0 + nu)
    sa, ca = math.sin(a), math.cos(a)
    # Eq. 19 (round wire EI_2 = EI_3 = EI).
    p1 = n_w * (ca / r**2) * (gj * sa**4 + (5.0 * ei - gj) * sa**2 * ca**2)
    # Eq. 23 foundation stiffness summed over outboard annuli; Eq. 22.
    c = 0.0
    for other in cable.layers:
        ri = getattr(other, "r_inner_m", None)
        ro = getattr(other, "r_outer_m", None)
        if ri is None or ro is None or ri < r:
            continue
        e_s = library.get_property(other.material, "E_Pa")
        c += 2.0 * math.pi * e_s * (ro - ri) * ca / (n_w * r)
    p2 = 2.0 * n_w * ca * math.sqrt(ei * c)
    return p1, p2


def test_birdcage_matches_saevik_thorsen_min_p1_p2() -> None:
    """P_crit = min(P1, P2) reproduces hand-evaluated Eqs. 19/22/23 (Eq. 36)."""
    cable = _synthetic_cable()
    layer = cable.helical_layers[0]
    p1, p2 = _saevik_thorsen_P1_P2(cable, layer)
    detail = birdcage_buckling_detail(cable, layer)
    assert detail.P1_lateral_N == pytest.approx(p1, rel=1e-12)
    assert detail.P2_radial_N == pytest.approx(p2, rel=1e-12)
    assert detail.P_crit_N == pytest.approx(min(p1, p2), rel=1e-12)
    assert birdcage_critical_force(cable, layer) == pytest.approx(
        min(p1, p2), rel=1e-12
    )
    assert detail.restraint_layers == ("outer_sheath",)


def test_birdcage_chae_sigma_c1_crosscheck() -> None:
    """σ_c1 = (2 cosα/A)√(EI c) (Chae 2023 Eq. 17) equals P2/(n_w A)."""
    cable = _synthetic_cable()
    layer = cable.helical_layers[0]
    detail = birdcage_buckling_detail(cable, layer)
    e = library.get_property(layer.material, "E_Pa")
    a = layer.lay_angle_rad
    ei = e * math.pi * layer.d_wire_m**4 / 64.0
    area = math.pi * layer.d_wire_m**2 / 4.0
    sigma_c1 = (2.0 * math.cos(a) / area) * math.sqrt(ei * detail.foundation_c_Pa)
    p2_over_nA = detail.P2_radial_N / (layer.n_wires * area)
    assert p2_over_nA == pytest.approx(sigma_c1, rel=1e-12)


def test_birdcage_bare_strand_uses_lateral_only_and_warns() -> None:
    """No outboard annulus ⇒ radial mode N/A: P_crit = P1, runtime warning."""
    from cablexsec.cables.schema import _LAYER_ADAPTER  # noqa: PLC0415

    armour = _LAYER_ADAPTER.validate_python(
        {
            "name": "bare",
            "layer_type": "helical",
            "material": "galvanised_steel_armour",
            "n_wires": 50,
            "d_wire_m": 0.005,
            "r_helix_m": 0.080,
            "lay_angle_deg": 20.0,
        }
    )
    cable = CableData(cable=CableMeta(name="bare strand"), layers=(armour,))
    detail = birdcage_buckling_detail(cable, armour)
    assert detail.restraint_layers == ()
    assert math.isinf(detail.P2_radial_N)
    assert detail.P_crit_N == pytest.approx(detail.P1_lateral_N, rel=1e-12)
    with pytest.warns(UserWarning, match="no outboard annular restraint"):
        birdcage_critical_force(cable, armour)


def test_birdcage_critical_force_infinite_at_zero_lay_angle() -> None:
    from cablexsec.cables.schema import _LAYER_ADAPTER  # noqa: PLC0415

    straight = _LAYER_ADAPTER.validate_python(
        {
            "name": "straight",
            "layer_type": "helical",
            "material": "galvanised_steel_armour",
            "n_wires": 50,
            "d_wire_m": 0.005,
            "r_helix_m": 0.080,
            "lay_angle_deg": 0.0,
        }
    )
    cable = CableData(cable=CableMeta(name="straight strand"), layers=(straight,))
    assert math.isinf(birdcage_critical_force(cable, straight))


# --------------------------------------------------------------------------- #
# Direct thermal stress contribution vs isothermal (isothermal invariant)          #
# --------------------------------------------------------------------------- #
def test_tensile_thermal_stress_reduces_curvature_capacity() -> None:
    cable = _synthetic_cable()
    tensions = np.array([2.0e5])
    iso = compute_capacity(cable, tensions)
    rated = compute_capacity(cable, tensions, sigma_T_armour_Pa=80.0e6)
    assert rated.kappa_max_1_m[0] < iso.kappa_max_1_m[0]


def test_zero_thermal_stress_recovers_isothermal_exactly() -> None:
    """Zero thermal stress recovers the isothermal invariant."""
    cable = _synthetic_cable()
    tensions = np.linspace(0.0, 3.0e5, 7)
    iso = compute_capacity(cable, tensions)
    rated0 = compute_capacity(cable, tensions, sigma_T_armour_Pa=0.0)
    assert np.allclose(iso.kappa_max_1_m, rated0.kappa_max_1_m, rtol=1e-12)


def test_capacity_boundary_matches_public_stress_engine() -> None:
    """Shared mean/friction stress, maximised over the complete wire perimeter."""
    from cablexsec import (  # noqa: PLC0415
        compute_stress,
        export_three_core_yaml_path,
        load_cable,
    )

    cable = load_cable(export_three_core_yaml_path())
    tension = 25.0e3
    kappa = wire_yield_curvature(cable, tension, contact_model="bounded")
    stress = compute_stress(
        cable,
        kappa,
        tension_N=tension,
        contact_model="bounded",
    )
    from cablexsec.mechanics.wire_stress import local_bending_amplitude

    usage = 0.0
    for layer in cable.helical_layers:
        axial = stress.sigma_total_Pa[layer.name] - stress.sigma_B_Pa[layer.name]
        amplitude = local_bending_amplitude(
            kappa,
            np.deg2rad(stress.phi_deg[layer.name]),
            library.get_property(layer.material, "E_Pa"),
            layer.d_wire_m,
            layer.lay_angle_rad,
            stress.phi_bend_rad,
        )
        usage = max(
            usage,
            float(np.max(np.abs(axial) + amplitude))
            / library.get_property(layer.material, "sigma_y_Pa"),
        )
    assert usage == pytest.approx(1.0, rel=2e-6)


# --------------------------------------------------------------------------- #
# Integration                                                                 #
# --------------------------------------------------------------------------- #
