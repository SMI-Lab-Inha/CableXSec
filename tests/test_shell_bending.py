# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for the metallic-sheath shell bending module (§6.9, R4).

Gates: thin-wall EI matches the exact annulus second moment to < 0.1% for a
thin wall; EA is exact; corrugation reduces stiffness; the five stress
contributions assemble correctly; sheath fatigue stays BLOCKED (stress only).
"""

from __future__ import annotations

import math

import pytest

from cablexsec.cables.schema import CylindricalShellLayer
from cablexsec.materials import library
from cablexsec.mechanics.shell_bending import (
    shell_axial_stiffness,
    shell_bending_stiffness,
    shell_wall_stress,
)

# Test-taxonomy category.
pytestmark = pytest.mark.analytical_sanity


def _lead_shell(*, f_corr: float = 1.0, scf: float = 1.0) -> CylindricalShellLayer:
    return CylindricalShellLayer(
        name="lead_sheath",
        material="lead_sheath",
        r_mid_m=0.050,
        t_m=0.002,  # t/r = 0.04 → thin-wall error (t/2r)^2 = 4e-4
        f_corrugation=f_corr,
        scf=scf,
    )


# --------------------------------------------------------------------------- #
# Thin-wall stiffness gates                                                   #
# --------------------------------------------------------------------------- #
def test_thinwall_EI_matches_exact_annulus_within_0p1pct() -> None:
    layer = _lead_shell()
    e = library.get_property("lead_sheath", "E_Pa")
    r, t = 0.050, 0.002
    r_o, r_i = r + t / 2.0, r - t / 2.0
    ei_exact = e * math.pi / 4.0 * (r_o**4 - r_i**4)
    ei_thin = shell_bending_stiffness(layer)
    assert abs(ei_thin - ei_exact) / ei_exact < 1.0e-3  # < 0.1%


def test_thinwall_EA_is_exact_annulus_area() -> None:
    layer = _lead_shell()
    e = library.get_property("lead_sheath", "E_Pa")
    r, t = 0.050, 0.002
    r_o, r_i = r + t / 2.0, r - t / 2.0
    ea_exact = e * math.pi * (r_o**2 - r_i**2)
    assert shell_axial_stiffness(layer) == pytest.approx(ea_exact, rel=1e-12)


def test_corrugation_factor_reduces_stiffness() -> None:
    full = shell_bending_stiffness(_lead_shell(f_corr=1.0))
    corrugated = shell_bending_stiffness(_lead_shell(f_corr=0.5))
    assert corrugated == pytest.approx(0.5 * full, rel=1e-12)


# --------------------------------------------------------------------------- #
# Five-contribution wall stress                                               #
# --------------------------------------------------------------------------- #
def test_wall_stress_assembles_five_contributions() -> None:
    layer = _lead_shell(scf=1.5)
    e = library.get_property("lead_sheath", "E_Pa")
    alpha = library.get_property("lead_sheath", "alpha_T_K")
    r, t = 0.050, 0.002
    kappa, eps, ea_total = 0.05, 1.0e-3, 1.0e8
    res = shell_wall_stress(
        layer, kappa, eps, ea_total, p_int_Pa=0.0, p_ext_Pa=2.0e6, delta_T_K=30.0
    )
    assert res.sigma_bending_Pa == pytest.approx(e * kappa * r, rel=1e-12)
    assert res.sigma_peak_Pa == pytest.approx(1.5 * e * kappa * r, rel=1e-12)
    assert res.sigma_hoop_Pa == pytest.approx((0.0 - 2.0e6) * r / t, rel=1e-12)
    assert res.sigma_thermal_Pa == pytest.approx(-e * alpha * 30.0, rel=1e-12)
    assert res.sigma_axial_Pa == pytest.approx(e * eps, rel=1e-12)
    expected_longitudinal = (
        res.sigma_peak_Pa + res.sigma_thermal_Pa + res.sigma_axial_Pa
    )
    expected_vm = math.sqrt(
        expected_longitudinal**2
        + res.sigma_hoop_Pa**2
        - expected_longitudinal * res.sigma_hoop_Pa
    )
    assert res.sigma_longitudinal_Pa == pytest.approx(expected_longitudinal, rel=1e-12)
    assert res.sigma_von_mises_Pa == pytest.approx(expected_vm, rel=1e-12)
    assert res.sigma_total_Pa == pytest.approx(expected_vm, rel=1e-12)


def test_zero_loads_give_zero_stress() -> None:
    res = shell_wall_stress(_lead_shell(), 0.0, 0.0, 1.0e8)
    assert res.sigma_total_Pa == 0.0


# --------------------------------------------------------------------------- #
# Lead sheath fatigue stays BLOCKED (Phase-2); Al sheath is now VALIDATED       #
# --------------------------------------------------------------------------- #
def test_lead_sheath_sn_stays_blocked() -> None:
    from cablexsec.materials import sn_curves  # noqa: PLC0415

    with pytest.raises(sn_curves.BlockedSNCurveError):
        sn_curves.get_sn_curve("cigre_tb623_lead")  # Phase-2 limitation
    # the aluminium sheath curve is now a real, registered curve (not blocked)
    assert "en1999_1_3_aluminium" in sn_curves.list_sn_curves()
    assert "bs7910_class_d_aluminium" not in sn_curves.BLOCKED_SN_CURVES


# --------------------------------------------------------------------------- #
# Corrugated-Al sheath fatigue (EN 1999-1-3) — the newly wired path            #
# --------------------------------------------------------------------------- #
def _export_cable():
    import cablexsec as cx  # noqa: PLC0415

    return cx.load_cable(cx.export_three_core_yaml_path(), geometry_override=None)


def test_al_sheath_fatigue_known_answer() -> None:
    """D equals the closed-form Σ count / N_f(Δσ) — curve + pipeline wired right."""
    import numpy as np  # noqa: PLC0415

    import cablexsec as cx  # noqa: PLC0415
    from cablexsec.materials.sn_curves import get_sn_curve  # noqa: PLC0415

    cab = _export_cable()
    # SCF-amplified corrugation-root stress per unit curvature [Pa].
    k_per = cx.compute_shell_stress(cab, "al_metallic_sheath", 1.0, 0.0).sigma_peak_Pa
    kappa0 = 0.01
    # Only ±κ0 (start = end = +κ0): EVERY rainflow cycle has range 2·κ0, so
    # D = Σ count / N_f(2 κ0) = n_cycles / N_f(2 κ0) exactly.
    hist = np.array([kappa0, -kappa0] * 10 + [kappa0])
    res = cx.compute_shell_fatigue(cab, "al_metallic_sheath", hist)

    rng_MPa = 2.0 * k_per * kappa0 / 1.0e6  # peak-to-peak range
    n_f = get_sn_curve("en1999_1_3_aluminium").cycles_to_failure(rng_MPa)
    assert res.D_miner == pytest.approx(res.n_cycles / n_f, rel=1e-9)
    assert res.sn_curve == "en1999_1_3_aluminium"


def test_al_sheath_fatigue_cutoff_zero_damage() -> None:
    """A range below the 40.6 MPa constant-amplitude cut-off → zero damage."""
    import numpy as np  # noqa: PLC0415

    import cablexsec as cx  # noqa: PLC0415

    cab = _export_cable()
    k_per = cx.compute_shell_stress(cab, "al_metallic_sheath", 1.0, 0.0).sigma_peak_Pa
    # choose kappa so the full range 2*k_per*kappa = 0.5 * 40.6 MPa < cut-off.
    kappa0 = 0.25 * 40.6e6 / k_per
    hist = np.array([kappa0, -kappa0] * 8 + [kappa0])
    res = cx.compute_shell_fatigue(cab, "al_metallic_sheath", hist)
    assert 2.0 * k_per * kappa0 / 1e6 < 40.6  # range below cut-off
    assert res.D_miner == 0.0


@pytest.mark.parametrize("curvature", [[0.02, -0.02, 0.02], [0.09, 0.091, 0.09]])
def test_shell_fatigue_rejects_yielded_mean_or_amplitude(curvature):
    import numpy as np

    import cablexsec as cx

    with pytest.raises(ValueError, match="exceeds yield stress"):
        cx.compute_shell_fatigue(
            _export_cable(), "al_metallic_sheath", np.array(curvature)
        )


def test_shell_fatigue_does_not_infer_an_unavailable_elastic_limit(monkeypatch):
    import numpy as np

    import cablexsec as cx

    original = library.get_property

    def missing_limit(material, name):
        if name == "sigma_y_Pa":
            raise KeyError(name)
        return original(material, name)

    monkeypatch.setattr(library, "get_property", missing_limit)
    with pytest.raises(ValueError, match="No yield stress"):
        cx.compute_shell_fatigue(
            _export_cable(), "al_metallic_sheath", np.array([0, 0.001, 0])
        )


def test_lead_shell_fatigue_stays_blocked() -> None:
    """compute_shell_fatigue on a lead sheath raises BlockedSNCurveError (Phase-2)."""
    import numpy as np  # noqa: PLC0415

    import cablexsec as cx  # noqa: PLC0415
    from cablexsec.cables.schema import CableData, CableMeta  # noqa: PLC0415
    from cablexsec.exceptions import BlockedSNCurveError  # noqa: PLC0415

    lead_shell = CylindricalShellLayer(
        name="lead_sheath",
        material="lead_sheath",
        r_mid_m=0.05,
        t_m=0.002,
        bulk_sn="cigre_tb623_lead",
    )
    cab = CableData(cable=CableMeta(name="lead-test"), layers=(lead_shell,))
    with pytest.raises(BlockedSNCurveError):
        cx.compute_shell_fatigue(cab, "lead_sheath", np.array([0.0, 0.01, -0.01]))


@pytest.mark.parametrize("index", range(5))
def test_compute_shell_stress_refuses_non_finite_inputs(index: int) -> None:
    import cablexsec as cx

    cable = cx.load_cable(cx.export_three_core_yaml_path(), geometry_override=None)
    values = [0.01, 1e-4, 0.0, 1e6, 30.0]
    values[index] = math.nan
    kappa, eps, p_int, p_ext, delta_t = values
    with pytest.raises(cx.PhysicalRangeError, match="shell-stress inputs"):
        cx.compute_shell_stress(
            cable,
            "al_metallic_sheath",
            kappa,
            eps,
            p_int_Pa=p_int,
            p_ext_Pa=p_ext,
            delta_T_K=delta_t,
        )
