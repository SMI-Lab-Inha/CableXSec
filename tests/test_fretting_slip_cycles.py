# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Per-cycle SWT fretting on the contact slip history (copper branch)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.fatigue import FatigueLayer, compute_fatigue
from cablexsec.analysis.fretting_swt import (
    partial_slip_surface_stress,
    swt_driver_from_contact,
    swt_life,
    swt_slip_cycle_damage,
)
from cablexsec.materials.sn_curves import (
    BLOCKED_SN_CURVES,
    SN_CURVES,
    SWT_LIFE,
    get_sn_curve,
    get_swt_life,
)
from cablexsec.mechanics.mkappa import HelicalLayerSpec

COPPER = get_swt_life("copper_poon2022")
P0, MU, E, NU = 26.0e6, 0.8, 97.0e9, 0.34


def _saturating(n_cycles: int, cap: float, overdrive: float = 3.0):
    """Tangential force clipped at +-cap, with slip flags where it is clipped."""
    t = np.linspace(0.0, 2 * math.pi * n_cycles, 400 * n_cycles + 1)
    trial = overdrive * cap * np.sin(t)
    force = np.clip(trial, -cap, cap)
    return force, np.abs(trial) > cap


def test_surface_stress_closed_form():
    assert partial_slip_surface_stress(P0, MU, 1.0) == pytest.approx(2 * MU * P0)
    assert partial_slip_surface_stress(P0, MU, 0.25) == pytest.approx(MU * P0)
    assert partial_slip_surface_stress(P0, MU, 4.0) == pytest.approx(2 * MU * P0)
    with pytest.raises(ValueError):
        partial_slip_surface_stress(P0, MU, -0.1)


def test_gross_slip_cycles_recover_the_sliding_line_driver():
    normal = 2.0
    force, slipping = _saturating(5, MU * normal)
    result = swt_slip_cycle_damage(force, slipping, normal, P0, MU, E, NU, COPPER)
    _s, _e, swt = swt_driver_from_contact(P0, MU, E, NU)
    life = swt_life(swt, COPPER)
    # Every cycle that reaches the friction limit slips there; only the final
    # elastic unloading half cycle (from -mu P back to zero) sticks.
    assert result.skipped_stick_cycles == 0.5
    assert result.evaluated_cycles == result.counted_cycles - 0.5
    full = result.count == 1.0
    np.testing.assert_allclose(result.amplitude_ratio[full], 1.0)
    np.testing.assert_allclose(result.swt_Pa[full], swt, rtol=1e-12)
    # Full cycles at the sliding-line driver; the virgin half cycle is smaller.
    assert full.sum() / life < result.damage < result.evaluated_cycles / life


def test_stick_cycles_are_skipped_and_partial_slip_is_optional():
    normal = 2.0
    t = np.linspace(0.0, 20 * math.pi, 2001)
    force = 0.3 * MU * normal * np.sin(t)  # never reaches mu P
    slipping = np.zeros(t.size, dtype=bool)
    skipped = swt_slip_cycle_damage(force, slipping, normal, P0, MU, E, NU, COPPER)
    assert skipped.damage == 0.0
    assert skipped.skipped_stick_cycles == pytest.approx(skipped.counted_cycles)
    partial = swt_slip_cycle_damage(
        force, slipping, normal, P0, MU, E, NU, COPPER, partial_slip="cattaneo_mindlin"
    )
    gross_force, gross_slip = _saturating(10, MU * normal)
    gross = swt_slip_cycle_damage(
        gross_force, gross_slip, normal, P0, MU, E, NU, COPPER
    )
    full = partial.count == 1.0
    np.testing.assert_allclose(partial.amplitude_ratio[full], 0.3, rtol=1e-3)
    assert 0.0 < partial.damage < gross.damage


def test_small_stick_cycles_no_longer_count_as_gross_sliding():
    """The audit defect: tiny reversals were each charged one gross-slip cycle."""
    normal = 2.0
    big, big_slip = _saturating(4, MU * normal)
    ripple = 0.02 * MU * normal * np.sin(np.linspace(0, 400 * math.pi, big.size))
    force = np.clip(big + ripple, -MU * normal, MU * normal)
    slipping = big_slip & (np.abs(big + ripple) >= MU * normal)
    mixed = swt_slip_cycle_damage(force, slipping, normal, P0, MU, E, NU, COPPER)
    alone = swt_slip_cycle_damage(big, big_slip, normal, P0, MU, E, NU, COPPER)
    assert mixed.counted_cycles > 10 * alone.counted_cycles
    assert mixed.damage == pytest.approx(alone.damage, rel=0.05)


def test_bulk_stress_superposition_increases_damage():
    normal = 2.0
    force, slipping = _saturating(3, MU * normal)
    bulk = 50e6 * np.sin(np.linspace(0, 6 * math.pi, force.size))
    base = swt_slip_cycle_damage(force, slipping, normal, P0, MU, E, NU, COPPER)
    combined = swt_slip_cycle_damage(
        force, slipping, normal, P0, MU, E, NU, COPPER, bulk_stress_Pa=bulk
    )
    assert combined.damage > base.damage


@pytest.mark.parametrize(
    "kwargs",
    [
        {"normal_force": 0.0},
        {"slipping": np.zeros(3, dtype=int)},
        {"partial_slip": "always"},
        {"bulk_stress_Pa": np.zeros(2)},
    ],
)
def test_slip_cycle_inputs_are_validated(kwargs):
    arguments = {
        "tangential_force": np.array([0.0, 1.0, -1.0]),
        "slipping": np.array([False, True, True]),
        "normal_force": 2.0,
        "p0_Pa": P0,
        "mu": MU,
        "E_Pa": E,
        "nu": NU,
        "const": COPPER,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        swt_slip_cycle_damage(**arguments)


def _copper_layer(n_wires: int = 12, **changes) -> FatigueLayer:
    layer = FatigueLayer(
        spec=HelicalLayerSpec(
            n_wires=n_wires,
            r_helix_m=0.05,
            lay_angle_rad=math.radians(12.0),
            E_wire_Pa=E,
            A_wire_m2=math.pi * 0.003**2 / 4,
            xi=0.98,
            mu=0.2,
            F_n_axi_N=50.0,
        ),
        sn_bulk=get_sn_curve("nasution_2014_copper"),
        sigma_UTS_Pa=220e6,
        name="cu",
        swt_const=COPPER,
        contact_p0_Pa=P0,
        fretting_mu=MU,
        contact_nu=NU,
        E_eff_axial_Pa=E,
        K_t_contact_N_per_m=1.0e6,
        crossing_density_per_m=50.0,
        contact_line_load_N_per_m=100.0,
    )
    return replace(layer, **changes)


def test_pipeline_skips_small_cycles_that_the_legacy_estimate_counted():
    large = 0.01 * np.sin(np.linspace(0, 8 * math.pi, 1601))
    # 200 small cycles whose slip range stays inside the contact's elastic
    # range 2 mu P / K_t: they reverse the bulk stress but never slip.
    small = 1.2e-4 * np.sin(np.linspace(0, 400 * math.pi, 8001))[1:]
    mixed = np.concatenate((large, small))
    layer = _copper_layer(n_wires=4)
    new_large = compute_fatigue([layer], large, np.zeros_like(large))
    new_mixed = compute_fatigue([layer], mixed, np.zeros_like(mixed))
    old_mixed = compute_fatigue(
        [layer],
        mixed,
        np.zeros_like(mixed),
        copper_contact_model="legacy_gross_sliding",
    )
    d_large = new_large.D_fretting_per_wire["cu"]
    d_new = new_mixed.D_fretting_per_wire["cu"]
    d_old = old_mixed.D_fretting_per_wire["cu"]
    # Loxodromic slip vanishes at the extreme fibres (wires 0 and 2, phi = 0
    # and pi): no slip, no fretting cycle. The legacy estimate charged them.
    sliding = np.sin(layer.spec.phi_wires) ** 2 > 1e-12
    assert np.all(d_new[~sliding] == 0) and np.all(d_old[~sliding] > 0)
    assert np.all(d_new[sliding] > 0)
    np.testing.assert_allclose(d_new, d_large, rtol=0.05)
    assert np.all(d_old[sliding] > 20 * d_new[sliding])
    assert "slipping cycles" in new_mixed.provenance.method
    assert "legacy" in old_mixed.provenance.method


def test_pipeline_gross_slip_matches_legacy_for_one_driver_per_cycle():
    t = np.linspace(0, 10 * math.pi, 2001)
    kappa = 0.01 * np.sin(t)
    layer = _copper_layer(contact_shape_g=None)
    new = compute_fatigue([layer], kappa, np.zeros_like(t))
    old = compute_fatigue(
        [layer], kappa, np.zeros_like(t), copper_contact_model="legacy_gross_sliding"
    )
    # Same driver per slipping cycle; counts differ only by end half cycles.
    sliding = np.sin(layer.spec.phi_wires) ** 2 > 1e-12
    np.testing.assert_allclose(
        new.D_fretting_per_wire["cu"][sliding],
        old.D_fretting_per_wire["cu"][sliding],
        rtol=0.06,
    )


def test_pipeline_needs_contact_stiffness_for_slip_cycles():
    layer = _copper_layer(K_t_contact_N_per_m=0.0)
    kappa = 0.01 * np.sin(np.linspace(0, 4 * math.pi, 101))
    with pytest.raises(ValueError, match="slip-cycle"):
        compute_fatigue([layer], kappa, np.zeros_like(kappa))
    legacy = compute_fatigue(
        [layer],
        kappa,
        np.zeros_like(kappa),
        copper_contact_model="legacy_gross_sliding",
    )
    assert np.all(legacy.D_fretting_per_wire["cu"] > 0)


@pytest.mark.parametrize(
    "kwargs",
    [{"copper_contact_model": "gross"}, {"partial_slip_cycles": "never"}],
)
def test_pipeline_option_names_are_checked(kwargs):
    with pytest.raises(ValueError):
        compute_fatigue([], np.zeros(3), np.zeros(3), **kwargs)


def test_packaged_cables_declare_only_registered_curve_names():
    """Every S-N or contact-life name declared by a bundled cable is registered."""
    known = set(SN_CURVES) | set(BLOCKED_SN_CURVES) | set(SWT_LIFE)
    paths = [
        cx.cigre_tb880_yaml_path(),
        cx.menard2023_yaml_path(),
        cx.inter_array_single_core_yaml_path(),
        cx.inter_array_three_core_yaml_path(),
        cx.export_three_core_yaml_path(),
    ]
    for path in paths:
        cable = cx.load_cable(path, geometry_override=None)
        for layer in cable.layers:
            for field_name in ("bulk_sn", "fretting_sn"):
                name = getattr(layer, field_name, "")
                assert not name or name in known, (path.name, layer.name, name)
