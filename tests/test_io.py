# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""L6.1–L6.2 — global-analysis I/O surfaces.

The reader is plumbing; plumbing gets a **known-answer gate** so a parsing bug
cannot silently corrupt a fatigue number:

* a zero-curvature, constant-amplitude tension history makes σ_total = σ_A only
  (σ_B = σ_f = 0), so the per-wire Miner damage is closed-form; the CSV/
  RIFLEX reader path must recover it to < 1e-10;
* write → read round-trips the canonical exchange CSV bit-for-bit, and the
  fatigue result through the reader equals the in-memory result exactly;
* the JONSWAP generator is deterministic and hits its target RMS.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.io.series import LoadCaseSeries
from cablexsec.materials import library
from cablexsec.materials.sn_curves import get_sn_curve

pytestmark = pytest.mark.analytical_sanity


def _menard() -> cx.CableData:
    return cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)


def _const_amplitude_tension_series(
    n_cycles: int = 20, t_lo: float = 2.0e4, t_hi: float = 6.0e4
) -> LoadCaseSeries:
    """Zero curvature; tension a clean [lo, hi, lo, …] square so σ_total = σ_A."""
    seq = [t_lo]
    for _ in range(n_cycles):
        seq += [t_hi, t_lo]
    tension = np.asarray(seq, dtype=np.float64)
    t = np.arange(tension.size, dtype=np.float64)
    z = np.zeros_like(tension)
    return LoadCaseSeries(
        time_s=t,
        curvature_x_1_m=z,
        curvature_y_1_m=z,
        tension_N=tension,
        tension_kind="true",
        source="known-answer-fixture",
    )


def _analytic_sigma_A_damage(cable: cx.CableData, series: LoadCaseSeries) -> dict:
    """Closed-form per-layer Miner damage when σ_total = σ_A (κ = 0)."""
    from cablexsec.mechanics.stiffness import ea_fixed  # noqa: PLC0415

    ea = ea_fixed(cable)
    out: dict[str, float] = {}
    for lyr in cable.helical_layers:
        if not lyr.structural:
            continue
        e_w = library.get_property(lyr.material, "E_Pa")
        uts = library.get_property(lyr.material, "sigma_UTS_Pa")
        a = lyr.lay_angle_rad
        sig = e_w * math.cos(a) ** 2 * series.tension_N / ea  # σ_A(t), uniform
        out[lyr.name] = _miner_closed_form(sig, get_sn_curve(lyr.bulk_sn), uts)
    return out


def _miner_closed_form(sig, sn_curve, uts) -> float:
    """Rainflow + range-based Miner on a 1-D stress series."""
    import rainflow  # noqa: PLC0415

    damage = 0.0
    _ = uts
    for rng, _mean, count, _i0, _i1 in rainflow.extract_cycles(sig.tolist()):
        damage += count / sn_curve.cycles_to_failure(rng / 1.0e6)
    return damage


def test_time_history_reader_known_answer_damage(tmp_path) -> None:
    """Reader → compute_fatigue recovers the closed-form σ_A damage to < 1e-10."""
    cable = _menard()
    series = _const_amplitude_tension_series()
    analytic = _analytic_sigma_A_damage(cable, series)

    # Write the series as a time-history CSV export, read it back via the reader.
    csv = tmp_path / "export.csv"
    lines = ["Time (s),x curvature (1/m),y curvature (1/m),Wall tension (kN)"]
    lines += [
        f"{series.time_s[i]},0.0,0.0,{series.tension_N[i] / 1e3}"
        for i in range(series.n_steps)
    ]
    csv.write_text("\n".join(lines), encoding="utf-8")

    read = cx.read_time_history_csv(csv)  # kN→N default scale
    res = cx.fatigue_from_series(cable, read)
    for name, d_analytic in analytic.items():
        recovered = float(res.D_bulk_per_wire[name][0])  # uniform across the layer
        assert recovered == pytest.approx(d_analytic, rel=1e-10, abs=1e-300)


def test_reader_roundtrip_identity_and_damage(tmp_path) -> None:
    """Exchange CSV round-trips bit-for-bit; fatigue identical through the reader."""
    cable = _menard()
    series = _const_amplitude_tension_series()
    path = cx.write_series_csv(series, tmp_path / "series.csv")
    back = cx.read_series_csv(path)

    assert np.array_equal(series.curvature_x_1_m, back.curvature_x_1_m)
    assert np.array_equal(series.curvature_y_1_m, back.curvature_y_1_m)
    assert np.array_equal(series.tension_N, back.tension_N)

    d_direct = cx.compute_fatigue(
        cable,
        series.curvature_x_1_m,
        series.curvature_y_1_m,
        tension_N=series.tension_N,
    )
    d_reader = cx.fatigue_from_series(cable, back)
    name = d_direct.critical_layer
    assert np.array_equal(
        d_direct.D_total_per_wire[name], d_reader.D_total_per_wire[name]
    )


def test_temperature_history_maps_into_fatigue_sigma_T() -> None:
    from cablexsec.mechanics.stiffness import ea_fixed

    cable = _menard()
    base = _const_amplitude_tension_series()
    hot = LoadCaseSeries(
        time_s=base.time_s,
        curvature_x_1_m=base.curvature_x_1_m,
        curvature_y_1_m=base.curvature_y_1_m,
        tension_N=base.tension_N,
        temperature_C=np.full(base.n_steps, 40.0),
        axial_strain=base.tension_N / ea_fixed(cable),
        axial_strain_kind="direct",
        source="thermal-known-answer",
    )
    cold_res = cx.fatigue_from_series(cable, base)
    hot_res = cx.fatigue_from_series(cable, hot, T_ref_C=0.0)
    name = cold_res.critical_layer
    assert np.array_equal(
        cold_res.D_total_per_wire[name], hot_res.D_total_per_wire[name]
    )
    assert hot_res.provenance is not None
    assert any("temperature_C mapped" in x for x in hot_res.provenance.limitations)


def test_riflex_reader_positional_columns(tmp_path) -> None:
    """RIFLEX positional ASCII reads back the exact arrays (with comments)."""
    asc = tmp_path / "riflex.asc"
    rows = ["# RIFLEX dynamic results", "# time  curv_x  tension"]
    rows += [f"{i * 0.5}  {0.01 * math.sin(i * 0.2)}  {1.2e5}" for i in range(40)]
    asc.write_text("\n".join(rows), encoding="utf-8")

    ser = cx.read_riflex_ascii(asc, time_col=1, curvature_x_col=2, tension_col=3)
    assert ser.n_steps == 40
    assert ser.tension_N[0] == pytest.approx(1.2e5)
    assert ser.curvature_x_1_m[10] == pytest.approx(0.01 * math.sin(10 * 0.2))
    assert ser.temperature_C is None  # isothermal


def test_jonswap_deterministic_and_hits_target_rms() -> None:
    """Same seed → identical series; RMS curvature matches the requested target."""
    kw = {
        "hs_m": 3.0,
        "tp_s": 9.0,
        "duration_s": 600.0,
        "dt_s": 0.5,
        "rms_curvature_1_m": 0.012,
        "seed": 7,
    }
    a = cx.jonswap_series(**kw)
    b = cx.jonswap_series(**kw)
    assert np.array_equal(a.curvature_x_1_m, b.curvature_x_1_m)
    rms = float(np.sqrt(np.mean(a.curvature_x_1_m**2)))
    assert rms == pytest.approx(0.012, rel=1e-12)
    assert np.all(a.curvature_y_1_m == 0.0)  # planar
    assert a.temperature_C is None
