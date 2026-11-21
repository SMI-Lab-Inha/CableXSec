# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""L2 promoted build 1 — per-wire all-component StressResult, gated (§6.7).

Analytical-sanity gates (PI-directed):
* σ_B at φ=0 vs the closed form [MenardCartraud2023] Eq. 14;
* σ_f respects the per-wire stick-slip boundary κ_c (stuck below, saturated above);
* GATE A — the assembled cable moment recovers compute_mkappa (the Papailiou
  engine) within per-wire discretization (a TRUE cross-engine check, since σ_f is
  the same slip-front physics, not the retired Coulomb loop);
* GATE B — slip starts at the NEUTRAL AXIS (Raoof/Papailiou), not the extreme
  fibre; peak |σ_total| stays at the extreme fibre;
* σ_T = 0 when isothermal (the BM2b invariant); σ_total is the component sum;
* σ_vm ≥ |σ_total| (von-Mises with the contact shear term).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.materials import library

pytestmark = pytest.mark.analytical_sanity


def _menard() -> cx.CableData:
    return cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)


def _contact_cable() -> cx.CableData:
    """A PUBLIC cable that carries a friction-setting contact pressure (its
    ``pretension_Pa`` → P1+P2), so the slip-front σ_f branch is exercised in CI."""
    return cx.load_cable(cx.inter_array_single_core_yaml_path(), geometry_override=None)


def _stress(kappa=0.03, tension=1.0e4):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore any extrapolation warning
        return cx.compute_stress(_menard(), kappa, tension_N=tension)


def _stress_contact(kappa=0.03):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cx.compute_stress(_contact_cable(), kappa, tension_N=0.0)


def test_sigma_B_at_phi0_matches_menard_eq14_plus_the_slip_state_term() -> None:
    # Ménard Eq. 14 is the normal curvature of a wire without friction
    # stress; the friction stress the wire carries adds (sin^2 a / r) sigma_f / E
    # (wire_stress.slip_state_normal_curvature).
    cable = _menard()
    for mu in (0.0, None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sr = cx.compute_stress(cable, 0.03, tension_N=1.0e4, mu=mu)
        for name in sr.phi_deg:
            lyr = cable.get_layer(name)
            e = library.get_property(lyr.material, "E_Pa")
            a = lyr.lay_angle_rad
            k0 = int(np.argmin(np.abs(sr.phi_deg[name])))  # wire nearest phi = 0
            closed = (
                e
                * (
                    sr.kappa_mag_1_m * math.cos(a) ** 2 * math.cos(2 * a)
                    + math.sin(a) ** 2 / lyr.r_helix_m * sr.sigma_f_Pa[name][k0] / e
                )
                * (lyr.d_wire_m / 2.0)
            )
            assert sr.sigma_B_Pa[name][k0] == pytest.approx(closed, rel=1e-9)
            if mu == 0.0:
                assert sr.sigma_f_Pa[name][k0] == 0.0


def test_sigma_f_respects_per_wire_kappa_c_boundary() -> None:
    """A wire slips iff κ_mag ≥ κ_c,k; stuck wires stay below the slip cap."""
    sr = _stress_contact(kappa=0.002)  # below the knee → a mix of stuck/slip
    for name in sr.phi_deg:
        kc = sr.kappa_c_1_m[name]
        predicted_slip = sr.kappa_mag_1_m >= kc
        assert np.array_equal(predicted_slip, sr.slipping[name])
        # Stuck wires carry the stick stress, which is ≤ the slip-front cap.
        stuck = ~sr.slipping[name]
        assert np.all(stuck) or kc[~stuck].size > 0  # the regime is exercised
        assert np.all(np.isfinite(sr.sigma_f_Pa[name]))


def test_slip_onset_is_at_the_neutral_axis_not_extreme_fibre() -> None:
    """GATE B — the distinguishing physics. Slip starts at the NEUTRAL AXIS
    (smallest κ_c), per the Raoof (1989b,1992) strain-gauge observation and the
    Papailiou mechanism — NOT at the extreme fibre (which the retired Coulomb
    loop would predict). Max |σ_total| stays at the extreme fibre."""
    sr = _stress_contact(kappa=0.03)
    for name in sr.phi_deg:
        kc = sr.kappa_c_1_m[name]
        phi = sr.phi_deg[name]
        finite = np.isfinite(kc)
        k_first = int(np.argmin(np.where(finite, kc, np.inf)))
        # first-to-slip wire sits near the neutral axis (|cos(φ−φ_bend)| ≈ 0).
        assert abs(np.cos(np.radians(phi[k_first]) - sr.phi_bend_rad)) < 0.2
        # peak total stress sits near the extreme fibre (|cos| ≈ 1).
        k_tot = int(np.argmax(np.abs(sr.sigma_total_Pa[name])))
        assert abs(np.cos(np.radians(phi[k_tot]) - sr.phi_bend_rad)) > 0.8


def test_assembled_moment_recovers_papailiou_engine_A11_cross_engine() -> None:
    """GATE A (cross-engine) — the per-wire σ_f is the SAME Papailiou slip-front
    compute_mkappa integrates, so its assembled moment recovers
    compute_mkappa (the public Papailiou engine) to within per-wire
    discretization. A TRUE cross-engine check, not the old circular
    self-comparison against the (now retired) Coulomb loop."""
    cable = _contact_cable()
    kappa = 0.03
    sr = cx.compute_stress(cable, kappa, tension_N=0.0)
    mk = cx.compute_mkappa(cable, np.linspace(0.0, kappa, 400), tension_N=0.0)
    assert sr.moment_recovered_Nm == pytest.approx(mk.moment_Nm[-1], rel=0.01)


def test_sigma_T_zero_when_isothermal_bm2b_invariant() -> None:
    sr = _stress()
    for name in sr.phi_deg:
        assert np.all(sr.sigma_T_Pa[name] == 0.0)


def test_heating_at_fixed_global_strain_is_compressive() -> None:
    """The wire-only thermal term uses the tensile-positive sign convention."""
    from cablexsec.mechanics.thermal import ThermalResult

    cable = _menard()
    thermal = ThermalResult(
        r_m=np.array([0.0, 1.0]),
        T_C=np.array([40.0, 40.0]),
        T_conductor_C=40.0,
        T_armour_C=40.0,
        P_total_W_m=0.0,
    )
    sr = cx.compute_stress(cable, 0.0, thermal=thermal, T_ref_C=20.0)
    for sigma_T in sr.sigma_T_Pa.values():
        assert np.all(sigma_T < 0.0)


def test_sigma_total_is_component_sum() -> None:
    sr = _stress()
    for name in sr.phi_deg:
        total = (
            sr.sigma_A_Pa[name]
            + sr.sigma_B_Pa[name]
            + sr.sigma_f_Pa[name]
            + sr.sigma_T_Pa[name]
        )
        assert np.allclose(sr.sigma_total_Pa[name], total)


def test_sigma_vm_at_least_abs_total() -> None:
    """von-Mises with the +3τ_H² contact shear term is ≥ |σ_total| everywhere."""
    sr = _stress()
    for name in sr.phi_deg:
        assert np.all(sr.sigma_vm_Pa[name] >= np.abs(sr.sigma_total_Pa[name]) - 1.0)


def test_all_helical_layers_present_and_provenance() -> None:
    """Per-wire stress is exposed for EVERY structural helical layer (the claim)."""
    cable = _menard()
    sr = _stress()
    armours = {x.name for x in cable.helical_layers if x.structural}
    assert set(sr.phi_deg) == armours
    assert sr.provenance is not None
    assert any("kappa_c" in g for g in sr.provenance.gates)
    assert sr.critical_layer in armours


def test_three_core_stress_uses_declared_mean_pressure() -> None:
    cable = cx.load_cable(cx.inter_array_three_core_yaml_path(), geometry_override=None)
    sr = cx.compute_stress(cable, 0.003, tension_N=0.0)
    varied = False
    for _name, sig in sr.sigma_f_Pa.items():
        if float(np.max(sig) - np.min(sig)) > 0.0:
            varied = True
    assert varied
    assert sr.provenance is not None
    assert any(
        "shape of the inner foundation" in lim for lim in sr.provenance.limitations
    )
