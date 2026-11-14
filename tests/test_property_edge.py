# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""L8.3 — property / edge tests on every public entry point.

Three families:
* **degenerate-but-valid** inputs (zero curvature, zero tension, single- vs
  three-core, solid-conductor core) return finite, sane results;
* **guards fire**: extrapolation beyond the validated range WARNS naming the
  bound; a physically inadmissible input RAISES PhysicalRangeError;
* **loud failures**: malformed YAML -> CableSchemaError; a BLOCKED S-N curve ->
  BlockedSNCurveError; a non-finite input -> PhysicalRangeError. Never a silent
  wrong number.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import cablexsec as cx
from cablexsec import (
    BlockedSNCurveError,
    CableSchemaError,
    PhysicalRangeError,
)
from cablexsec.provenance import ExtrapolationWarning

pytestmark = pytest.mark.analytical_sanity

_TOPOLOGIES = {
    "single_core_inter_array": (cx.inter_array_single_core_yaml_path, 1),
    "three_core_inter_array": (cx.inter_array_three_core_yaml_path, 3),
    "single_core_export": (cx.cigre_tb880_yaml_path, 1),
    "three_core_export": (cx.export_three_core_yaml_path, 3),
}


def _load(path_fn) -> cx.CableData:
    return cx.load_cable(path_fn(), geometry_override=None)


# --------------------------------------------------------------------------- #
# Degenerate-but-valid inputs                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(_TOPOLOGIES))
def test_every_topology_zero_curvature_zero_tension(name) -> None:
    """Zero curvature + zero tension: finite stiffness, zero-moment M-κ."""
    cable = _load(_TOPOLOGIES[name][0])
    stiff = cx.compute_stiffness(cable)
    assert np.isfinite(stiff.ei_min_Nm2) and stiff.ei_min_Nm2 > 0.0
    assert stiff.ei_min_Nm2 <= stiff.ei_max_Nm2

    mk = cx.compute_mkappa(cable, np.zeros(5), tension_N=0.0)
    assert np.allclose(mk.moment_Nm, 0.0)  # no curvature -> no moment

    sr = cx.compute_stress(cable, 0.0, tension_N=0.0)
    for name_l in sr.sigma_total_Pa:
        assert np.allclose(sr.sigma_total_Pa[name_l], 0.0)


@pytest.mark.parametrize("name", list(_TOPOLOGIES))
def test_every_topology_runs_full_pipeline(name) -> None:
    """Every L2 topology runs stiffness/M-κ/stress/capacity/fatigue end-to-end."""
    path_fn, n_cores = _TOPOLOGIES[name]
    cable = _load(path_fn)
    k = 0.01 * np.sin(np.linspace(0, 8 * np.pi, 60))
    assert np.isfinite(cx.compute_mkappa(cable, k, tension_N=5e4).W_diss_J_per_m)
    assert cx.compute_stress(cable, 0.012, tension_N=5e4).critical_layer
    cap = cx.compute_capacity(cable, np.array([0.0, 1.0e5]))
    assert np.all(np.isfinite(cap.kappa_max_1_m))
    fat = cx.compute_fatigue(cable, k, np.zeros_like(k))
    assert fat.critical_layer


def test_solid_conductor_core_thermal_divide_by_zero_path() -> None:
    """A solid conductor (r_inner=0) must not divide-by-zero in the thermal ladder."""
    cable = _load(cx.inter_array_single_core_yaml_path)
    area = 1.2e-3
    th = cx.compute_thermal(cable, 500.0, area, n_cores=1)
    assert np.all(np.isfinite(th.T_C))
    assert th.T_conductor_C >= th.T_armour_C  # conductor is the hottest node


def test_single_core_vs_three_core_distinguished() -> None:
    """Single- and three-core cables give distinct, finite stiffness."""
    sc = cx.compute_stiffness(_load(cx.inter_array_single_core_yaml_path))
    tc = cx.compute_stiffness(_load(cx.inter_array_three_core_yaml_path))
    assert np.isfinite(sc.ea_fixed_N) and np.isfinite(tc.ea_fixed_N)
    assert sc.ea_fixed_N != tc.ea_fixed_N


# --------------------------------------------------------------------------- #
# Guards fire — soft (warn) and hard (raise)                                   #
# --------------------------------------------------------------------------- #
def test_extrapolation_beyond_validated_range_warns_naming_bound() -> None:
    """A curvature past the wire-yield bound WARNS, naming the bound (soft guard)."""
    cable = _load(cx.inter_array_single_core_yaml_path)
    with pytest.warns(ExtrapolationWarning, match="curvature"):
        cx.compute_mkappa(cable, np.array([0.0, 5.0]), tension_N=0.0)  # κ=5 1/m


@pytest.mark.parametrize(
    "call",
    [
        lambda c: cx.compute_mkappa(c, np.array([0.0, np.nan])),
        lambda c: cx.compute_stress(c, np.inf),
        lambda c: cx.compute_fatigue(c, np.array([np.nan, 0.0]), np.zeros(2)),
        lambda c: cx.xsec_stiffness(c, -0.5),
        lambda c: cx.xsec_stiffness(c, float("nan")),
    ],
)
def test_physically_inadmissible_input_raises(call) -> None:
    """Non-finite / negative-curvature inputs RAISE PhysicalRangeError (hard)."""
    cable = _load(cx.inter_array_single_core_yaml_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(PhysicalRangeError):
            call(cable)


# --------------------------------------------------------------------------- #
# Loud failures — clear, typed, never a silent wrong number                    #
# --------------------------------------------------------------------------- #
def test_malformed_yaml_raises_cable_schema_error(tmp_path) -> None:
    """A cable whose required geometry is missing fails loudly on use."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "cable:\n  name: broken\n"
        "layers:\n"
        "  - name: armour\n    layer_type: helical\n"
        "    material: galvanised_steel_armour\n",  # no n_wires/d_wire/etc.
        encoding="utf-8",
    )
    cable = cx.load_cable(bad, geometry_override=None)
    assert not cable.is_fully_specified()
    with pytest.raises(CableSchemaError, match="not fully specified|missing"):
        cx.compute_mkappa(cable, np.array([0.0, 0.01]))


def test_unparseable_yaml_raises_clearly(tmp_path) -> None:
    """A structurally invalid cable file raises (schema/validation), not silence."""
    bad = tmp_path / "junk.yaml"
    bad.write_text("cable:\n  name: x\nlayers:\n  - not_a_mapping\n", encoding="utf-8")
    with pytest.raises(Exception) as exc:  # noqa: PT011 (pydantic or schema error)
        cx.load_cable(bad, geometry_override=None)
    assert exc.value is not None


def test_blocked_sn_curve_raises_blocked_error() -> None:
    """A BLOCKED S-N curve (deferred standard) raises BlockedSNCurveError."""
    from cablexsec.materials.sn_curves import BLOCKED_SN_CURVES, get_sn_curve

    key = next(iter(BLOCKED_SN_CURVES))
    with pytest.raises(BlockedSNCurveError, match="BLOCKED"):
        get_sn_curve(key)


def test_exception_hierarchy_catchable_as_base() -> None:
    """Every typed error is catchable as CableXSecError (L1.3 contract)."""
    from cablexsec import CableXSecError

    for exc in (CableSchemaError, BlockedSNCurveError, PhysicalRangeError):
        assert issubclass(exc, CableXSecError)
    assert issubclass(PhysicalRangeError, ValueError)
    assert issubclass(BlockedSNCurveError, KeyError)
