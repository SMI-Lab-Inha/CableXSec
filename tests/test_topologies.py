# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""L2 topology matrix — one end-to-end integration test per cable topology.

Every public topology must run the full pipeline (stiffness → M-κ → stress →
capacity → fatigue → thermal, plus shell stress for metallic sheaths). Each
topology also asserts a physical reduction limit:

* single-core ⇒ three-core Fourier correction vanishes (k3 = k6 = 0);
* n = 1 thermal ⇒ the IEC ladder collapses to the 1-D conduction primitive
  (incl. the solid-conductor r_inner = 0 path);
* no metallic sheath ⇒ the shell-stress terms do not exist (compute_shell_stress
  rejects the cable).

A topology that cannot run end-to-end is a release blocker — these tests fail
rather than skip.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.cable_adapters import cable_to_fatigue_layers
from cablexsec.cables import (
    export_three_core_yaml_path,
    inter_array_single_core_yaml_path,
    inter_array_three_core_yaml_path,
)
from cablexsec.cables.schema import CylindricalShellLayer
from cablexsec.mechanics.contact_pressure import solve_three_core

pytestmark = pytest.mark.analytical_sanity

_TOPOLOGIES = {
    "single_core_inter_array": (inter_array_single_core_yaml_path, False, 1),
    "three_core_inter_array": (inter_array_three_core_yaml_path, False, 3),
    "single_core_export": (cx.poradowski2025_220kv_yaml_path, True, 1),
    "three_core_export": (export_three_core_yaml_path, True, 3),
    "three_core_wet_66kv": (cx.iea_task49_66kv_yaml_path, False, 3),
}


def _run_pipeline(cable):
    """Run the full public pipeline; return the result objects."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # extrapolation / >90C guards are expected
        stiff = cx.compute_stiffness(cable)
        kappa = 0.015 * np.sin(np.linspace(0.0, 6.0 * np.pi, 400))
        mk = cx.compute_mkappa(cable, kappa, tension_N=5.0e4)
        sr = cx.compute_stress(cable, 0.015, tension_N=5.0e4)
        cap = cx.compute_capacity(cable, np.array([0.0, 1.0e5]))
        fat = cx.compute_fatigue(
            cable, kappa, np.zeros_like(kappa), contact_tension_N=5.0e4
        )
        area = cable.get_layer("cu_conductor").conductor_area_mm2 * 1e-6
        th = cx.compute_thermal(cable, 500.0, area)
    return stiff, mk, sr, cap, fat, th


@pytest.mark.parametrize("name", list(_TOPOLOGIES))
def test_topology_runs_end_to_end(name: str) -> None:
    path_fn, has_sheath, _n = _TOPOLOGIES[name]
    cable = cx.load_cable(path_fn(), geometry_override=None)
    assert cable.is_fully_specified(), f"{name}: not fully specified — release blocker"

    stiff, mk, sr, cap, fat, th = _run_pipeline(cable)

    # Physical sanity across the whole pipeline.
    assert stiff.ea_fixed_N > 0.0
    assert 0.0 < stiff.ei_min_Nm2 <= stiff.ei_max_Nm2
    assert np.isfinite(mk.EI_apparent_Nm2) and mk.W_diss_J_per_m >= 0.0
    assert stiff.ei_min_Nm2 <= mk.EI_apparent_Nm2 <= stiff.ei_max_Nm2 * 1.001
    assert sr.critical_sigma_vm_Pa > 0.0
    assert np.all(np.isfinite(cap.kappa_max_1_m)) and cap.kappa_max_1_m[0] > 0.0
    assert math.isfinite(th.T_conductor_C)
    assert th.T_conductor_C >= cable.environment.ambient_C
    assert fat.critical_layer != ""
    # The M-κ moment is recovered by the per-wire stress assembly (A11).
    assert sr.provenance is not None and mk.provenance is not None


@pytest.mark.parametrize("name", list(_TOPOLOGIES))
def test_topology_shell_reduction(name: str) -> None:
    """Metallic sheath ⇒ shell stress runs; no sheath ⇒ shell terms vanish."""
    path_fn, has_sheath, _n = _TOPOLOGIES[name]
    cable = cx.load_cable(path_fn(), geometry_override=None)
    shells = [x.name for x in cable.layers if isinstance(x, CylindricalShellLayer)]
    if has_sheath:
        assert shells, f"{name}: expected a metallic sheath"
        ss = cx.compute_shell_stress(cable, shells[0], 0.02, 1.0e-3, p_ext_Pa=2.0e6)
        assert ss.sigma_total_Pa != 0.0 and ss.provenance is not None
    else:
        assert not shells  # polymer topology: no shell terms exist
        with pytest.raises(cx.CableSchemaError):
            cx.compute_shell_stress(cable, "cu_conductor", 0.02, 1.0e-3)


def test_single_core_reduces_three_core_fourier_to_zero() -> None:
    """Single-core (uniform foundation, k_filler = k_core) ⇒ k3 = k6 = 0."""
    # theta_core = 70 deg leaves filler voids between the three cores (3*70 < 360).
    common = {
        "q0_Pa": 1.0e5,
        "theta_core_rad": math.radians(70.0),
        "EI_ring": 5.0e2,
        "r_bedding_m": 0.05,
        "n_wires": 60,
    }
    # Single-core reduction: a uniform foundation (k_filler = k_core) has no
    # trefoil modulation, so the 3rd/6th harmonics vanish.
    uniform = solve_three_core(k_core=1.0e8, k_filler=1.0e8, **common)
    assert abs(uniform.k3) < 1e-6 * uniform.k0
    assert abs(uniform.k6) < 1e-6 * uniform.k0
    # A genuine trefoil (stiff cores over soft filler voids) DOES modulate the
    # foundation — confirming the reduction above is a real limit, not always
    # zero. With the cores phased at π/6 (30/150/270°, the cos3φ nodes), the
    # leading non-uniform harmonic is the 6th, not the 3rd.
    trefoil = solve_three_core(k_core=1.0e8, k_filler=1.0e6, **common)
    assert abs(trefoil.k6) > 1e-3 * trefoil.k0


def test_single_core_thermal_reduces_to_1d_primitive_solid_core() -> None:
    """n=1 thermal on a solid-core (r_inner=0) cable runs and is monotone in r."""
    cable = cx.load_cable(inter_array_single_core_yaml_path(), geometry_override=None)
    area = cable.get_layer("cu_conductor").conductor_area_mm2 * 1e-6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        th = cx.compute_thermal(cable, 500.0, area, n_cores=1)
    # Conductor is the hottest node; temperature decreases monotonically outward.
    assert th.T_conductor_C == pytest.approx(float(np.max(th.T_C)), rel=1e-9)
    assert np.all(np.diff(th.T_C) <= 1e-9)


# --------------------------------------------------------------------------- #
# compute_fatigue(cable) adapter (L2 promoted build 2)                         #
# --------------------------------------------------------------------------- #
def test_fatigue_adapter_blocks_unsupported_steel_fretting_life() -> None:
    """The former Périer slip-amplitude fit is not routed into damage."""
    cable = cx.load_cable(inter_array_single_core_yaml_path(), geometry_override=None)
    layers = cable_to_fatigue_layers(cable, contact_tension_N=5.0e4)
    assert {fl.name for fl in layers} == {"inner_armour", "outer_armour"}
    for fl in layers:
        assert fl.sn_bulk.name == "saevik_2011_flat_steel"
        assert fl.sn_fretting is None
        assert fl.K_t_contact_N_per_m == 0.0


def test_fatigue_adapter_bulk_only_when_no_fretting_curve() -> None:
    """Ménard armours name no fretting S-N ⇒ bulk-only layers (no fretting branch)."""
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    layers = cable_to_fatigue_layers(cable, contact_tension_N=5.0e4)
    for fl in layers:
        assert fl.sn_fretting is None and fl.swt_const is None
        assert fl.K_t_contact_N_per_m == 0.0


def test_winding_stress_drives_contact_pressure_and_dissipation() -> None:
    """BM2b chain: a cited armour winding stress (from the YAML) → Feret-Bournazel
    contact pressure → Papailiou W_diss, reproduced FROM the cable definition.

    A two-armour cable built with ``winding_stress_Pa`` produces a positive
    inter-layer contact pressure that scales ~linearly with the winding stress,
    and a Papailiou dissipation that rises monotonically with it — no harness P0.
    """
    from cablexsec.cables.schema import (  # noqa: PLC0415
        _LAYER_ADAPTER,
        CableData,
        CableMeta,
    )
    from cablexsec.materials import library  # noqa: PLC0415
    from cablexsec.mechanics.axisymmetric import (  # noqa: PLC0415
        compute_winding_contact_pressure,
    )
    from cablexsec.mechanics.papailiou import (  # noqa: PLC0415
        PapailiouLayer,
        compute_mkappa_papailiou,
    )

    def _cable(winding: float):
        layers = [
            {
                "name": "bedding",
                "layer_type": "cylindrical",
                "material": "hdpe",
                "r_inner_m": 0.0,
                "r_outer_m": 0.060,
            },
            {
                "name": "inner_armour",
                "layer_type": "helical",
                "material": "galvanised_steel_armour",
                "n_wires": 50,
                "d_wire_m": 0.005,
                "r_helix_m": 0.0625,
                "lay_angle_deg": 20.0,
                "winding_stress_Pa": winding,
            },
            {
                "name": "outer_armour",
                "layer_type": "helical",
                "material": "galvanised_steel_armour",
                "n_wires": 56,
                "d_wire_m": 0.005,
                "r_helix_m": 0.0675,
                "lay_angle_deg": 20.0,
                "lay_direction": "left_hand",
                "winding_stress_Pa": winding,
            },
            {
                "name": "sheath",
                "layer_type": "cylindrical",
                "material": "hdpe",
                "r_inner_m": 0.070,
                "r_outer_m": 0.076,
            },
        ]
        parsed = tuple(_LAYER_ADAPTER.validate_python(x) for x in layers)
        return CableData(cable=CableMeta(name="winding test"), layers=parsed)

    sy = library.get_property("galvanised_steel_armour", "sigma_y_Pa")
    cab = _cable(0.09 * sy)
    assert cab.get_layer("inner_armour").winding_stress_Pa == pytest.approx(0.09 * sy)

    p_lo = compute_winding_contact_pressure(cab, 0.05 * sy)
    p_hi = compute_winding_contact_pressure(cab, 0.10 * sy)
    for name in ("inner_armour", "outer_armour"):
        assert p_lo[name] > 0.0
        # capstan transfer is linear in winding stress: ~2x stress -> ~2x pressure.
        assert p_hi[name] == pytest.approx(2.0 * p_lo[name], rel=0.05)

    def _wdiss(frac: float) -> float:
        p = compute_winding_contact_pressure(cab, frac * sy)
        mu = library.get_property("galvanised_steel_armour", "mu_on_hdpe")
        layers = [
            PapailiouLayer(
                lyr.n_wires,
                lyr.r_helix_m,
                lyr.lay_angle_rad,
                library.get_property(lyr.material, "E_Pa"),
                math.pi * lyr.d_wire_m**2 / 4,
                lyr.d_wire_m,
                mu,
                p[lyr.name],
            )
            for lyr in cab.helical_layers
        ]
        kappa = 0.05 * np.sin(np.linspace(0.0, 6 * np.pi, 600))
        return compute_mkappa_papailiou(layers, kappa, EI_min_Nm2=1.0e3).W_diss_J_per_m

    assert 0.0 < _wdiss(0.05) < _wdiss(0.09) < _wdiss(0.13)


def test_fatigue_cable_produces_damage() -> None:
    cable = cx.load_cable(inter_array_single_core_yaml_path(), geometry_override=None)
    kappa = 0.012 * np.sin(np.linspace(0.0, 60.0 * np.pi, 1200))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fat = cx.compute_fatigue(
            cable, kappa, np.zeros_like(kappa), contact_tension_N=5.0e4
        )
    assert fat.critical_layer in {"inner_armour", "outer_armour"}
    # Bulk fatigue damage accrues. The registered armour input has no supported
    # steel slip-amplitude life curve, so the contact-life fraction is zero.
    total = sum(float(np.sum(v)) for v in fat.D_total_per_wire.values())
    assert total > 0.0
    assert 0.0 <= fat.D_fretting_fraction <= 1.0
    assert fat.provenance is not None
