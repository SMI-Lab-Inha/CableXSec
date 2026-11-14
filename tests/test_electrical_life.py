# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for the water-tree screening module (§6.13, Mazzanti 2007 Eq. 17).

Gates: the electro-thermal life factor recovers L0 (factor = 1) at the reference
exactly, is monotone non-increasing in field E and temperature T, and carries
the temperature-dependent VEC synergism n(T) = n0 - b*cT. The mechanical
multiplier is a flagged CableXSec extension (not Mazzanti).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cablexsec.analysis.electrical_life import (
    coaxial_field,
    compute_electrical_life,
    mazzanti_life_factor,
    mechanical_derating,
    mechanical_strain,
)
from cablexsec.cables.schema import CableData, CableMeta

# Mazzanti (2007) Table I — XLPE.
_N0, _B, _b, _E0 = 15.0, 12430.0, 4420.0, 5.0e6  # E0 = 5 kV/mm in V/m
_T0K = 293.15


# Test-taxonomy category.
pytestmark = pytest.mark.analytical_sanity


def _f(E: float, T_K: float, b: float = _b) -> float:
    return mazzanti_life_factor(
        E, T_K, E0_Vm=_E0, T0_K=_T0K, n0=_N0, B_K=_B, b_synergism=b
    )


def test_recovers_L0_at_reference_exactly() -> None:
    """Factor = 1 at E=E0, T=T0 (cT=0 -> both terms unity)."""
    assert _f(_E0, _T0K) == pytest.approx(1.0, rel=1e-15)


def test_monotone_decreasing_in_field() -> None:
    facs = [_f(e, _T0K) for e in np.linspace(_E0, 3.0 * _E0, 20)]
    assert np.all(np.diff(facs) < 0.0)


def test_monotone_decreasing_in_temperature_above_E0() -> None:
    # Above the threshold field, raising T shortens life (Mazzanti Fig. 3).
    facs = [_f(2.0 * _E0, t) for t in np.linspace(_T0K, _T0K + 70.0, 20)]
    assert np.all(np.diff(facs) < 0.0)


def test_vec_synergism_makes_exponent_temperature_dependent() -> None:
    """n(T) = n0 - b*cT: dropping b (b=0) changes the factor at T != T0."""
    e, t = 2.0 * _E0, _T0K + 70.0
    assert _f(e, t, b=_b) != pytest.approx(_f(e, t, b=0.0), rel=1e-6)
    # n(90 C) = 15 - 4420*cT ~ 12.1 (the "n=12" of the old constant-n form).
    c_t = 1.0 / _T0K - 1.0 / (273.15 + 90.0)
    assert _N0 - _b * c_t == pytest.approx(12.1, abs=0.2)


def test_mechanical_derating_is_flagged_extension() -> None:
    assert mechanical_derating(0.0, 100.0) == 1.0
    assert mechanical_derating(0.01, 100.0) < 1.0
    assert mechanical_derating(0.02, 100.0) < mechanical_derating(0.01, 100.0)


def test_coaxial_field_closed_form() -> None:
    e = coaxial_field(220.0, 0.0227, 0.0466)
    u0 = 220e3 / math.sqrt(3.0)
    assert e == pytest.approx(u0 / (0.0227 * math.log(0.0466 / 0.0227)), rel=1e-12)


def test_mechanical_strain_is_std_times_radius() -> None:
    series = np.array([-0.01, 0.0, 0.01, 0.0])
    expected = float(np.std(series)) * 0.05
    assert mechanical_strain(series, 0.05) == pytest.approx(expected)


def _xlpe_cable() -> CableData:
    """A 220 kV insulation layer carrying the Mazzanti Table-I electrical block."""
    from cablexsec.cables.schema import _LAYER_ADAPTER  # noqa: PLC0415

    layer = {
        "name": "xlpe_insulation",
        "layer_type": "cylindrical_offset",
        "material": "xlpe_qiao2025",
        "n_cores": 1,
        "r_offset_m": 0.0,
        "r_inner_m": 0.0227,
        "r_outer_m": 0.0466,
        "electrical": {
            "permittivity_relative": 2.5,
            "loss_factor_tan_delta": 0.001,
            "vec_exponent_n": _N0,
            "thermal_b_k": _B,
            "vec_synergism_b": _b,
            "field_threshold_kV_mm": 5.0,
        },
    }
    return CableData(
        cable=CableMeta(name="export insulation cable", voltage_kV=220.0),
        layers=(_LAYER_ADAPTER.validate_python(layer),),
    )


def test_compute_electrical_life_runs_and_ages_with_heat() -> None:
    cable = _xlpe_cable()
    kappa = np.array([-0.02, 0.0, 0.02, 0.0, -0.02, 0.02])
    res = compute_electrical_life(
        cable, kappa, T_conductor_C=85.0, T0_C=20.0, L0_years=30.0
    )
    assert res.life_factor > 0.0
    assert res.E_field_Vm > 0.0
    # Above threshold field and above reference T -> faster ageing than L0.
    assert res.life_factor < 1.0
