# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Regenerate the validation report, validation/validation_report.md.

The report is calculated from the live source. Analytical agreement, internal
consistency and measured response comparisons are distinct assessments;
their targets and unresolved discrepancies remain visible. Every input is a
published value or a file in this repository.

Usage: ``python validation/scripts/run_validation.py [--check]``.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np

import cablexsec as cx
from cablexsec.analysis.capacity import birdcage_buckling_detail
from cablexsec.analysis.curve_comparison import cycle_work, insert_cycle_crossings
from cablexsec.cables import (
    inter_array_single_core_legacy_r2_yaml_path,
    inter_array_three_core_legacy_r2_yaml_path,
)
from cablexsec.cables.schema import (
    CableData,
    CableMeta,
    CylindricalLayer,
    CylindricalShellLayer,
)
from cablexsec.materials import library
from cablexsec.mechanics.mindlin_contact import (
    MindlinContact,
    geometric_slip_factor,
    loxodromic_slip_increment,
    slip_increment,
)
from cablexsec.mechanics.mkappa import compute_mkappa_loop, energy_closure
from cablexsec.mechanics.shell_bending import shell_wall_stress
from cablexsec.mechanics.stiffness import (
    ea_fixed,
    ei_min_slip,
)
from cablexsec.mechanics.thermal import (
    ThermalResult,
    dielectric_loss,
    iec_conductor_temperature,
    proximity_effect_factor,
    sheath_circulating_loss_factor,
    skin_effect_factor,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation" / "validation_report.md"

_BM1_ANALYTIC_EI_MIN = 3593.0  # [MenardCartraud2023] Table 3


@dataclass
class Row:
    tier: str
    ident: str
    quantity: str
    target: str
    value: str
    status: str

    def line(self) -> str:
        return (
            f"| {self.tier} | {self.ident} | {self.quantity} | {self.target} | "
            f"{self.value} | {self.status} |"
        )


# Staleness check. A value far inside its own tolerance and below an absolute
# ceiling is round-off: its printed digits differ between machines (for
# example 6.33e-13% and 7.39e-12% against < 0.01%). Such values are compared
# by tolerance class; every other number, the text and the status are
# compared exactly, so a real change still marks the report stale. Values at
# a few hundred machine epsilons (the floor) are round-off for any tolerance:
# against < 1e-12, results of 1.07e-15 and 9.48e-16 differ only by platform.
_ROUNDOFF_FRACTION = 1.0e-3
_ROUNDOFF_FLOOR = 1.0e-13
_ROUNDOFF_CEILING = 1.0e-9
_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?(?![\w.])")


def _tolerance(segment: str) -> float | None:
    """Smallest positive number in a target segment, or None."""
    values = [abs(float(m.group())) for m in _NUMBER.finditer(segment)]
    positive = [v for v in values if v > 0.0 and math.isfinite(v)]
    return min(positive) if positive else None


def _value_classes(value: str, target: str) -> tuple[str, ...]:
    """Value text with round-off-level numbers replaced by their class.

    Value and target segments (separated by ';') are paired when their
    counts agree; a single target applies to every value segment; otherwise
    the smallest declared tolerance applies. Without a positive tolerance no
    number is classed as round-off.
    """
    values = value.split(";")
    targets = target.split(";")
    if len(targets) == len(values):
        tolerances = [_tolerance(t) for t in targets]
    else:
        whole = (
            _tolerance(targets[0])
            if len(targets) == 1
            else min(
                (t for t in map(_tolerance, targets) if t is not None), default=None
            )
        )
        tolerances = [whole] * len(values)
    classes = []
    for segment, tol in zip(values, tolerances, strict=True):
        if tol is None:
            classes.append(segment.strip())
            continue
        limit = min(max(_ROUNDOFF_FRACTION * tol, _ROUNDOFF_FLOOR), _ROUNDOFF_CEILING)

        def classify(match: re.Match[str], limit: float = limit) -> str:
            number = match.group()
            return "<round-off>" if abs(float(number)) <= limit else number

        classes.append(_NUMBER.sub(classify, segment).strip())
    return tuple(classes)


def _row_recorded(row: Row, line: str) -> bool:
    """True if a recorded report line states this row up to round-off."""
    if line == row.line():
        return True
    prefix = f"| {row.tier} | {row.ident} | {row.quantity} | {row.target} | "
    suffix = f" | {row.status} |"
    if not (line.startswith(prefix) and line.endswith(suffix)):
        return False
    if len(line) < len(prefix) + len(suffix):
        return False
    recorded = line[len(prefix) : len(line) - len(suffix)]
    return _value_classes(recorded, row.target) == _value_classes(row.value, row.target)


def report_is_stale(
    rows: list[Row], text: str, recorded: str | None, *, public_only: bool
) -> bool:
    """Compare a recorded report with the live one, rows by tolerance class.

    ``public_only`` checks only that every row is recorded; otherwise every
    line must agree, with table rows compared by :func:`_row_recorded`.
    """
    if recorded is None:
        return True
    recorded_lines = recorded.split("\n")
    if public_only:
        public = [r for r in rows if "needs local" not in r.status.lower()]
        return not all(
            any(_row_recorded(r, line) for line in recorded_lines) for r in public
        )
    expected_lines = text.split("\n")
    if len(expected_lines) != len(recorded_lines):
        return True
    by_line = {r.line(): r for r in rows}
    for expected, actual in zip(expected_lines, recorded_lines, strict=True):
        if expected == actual:
            continue
        row = by_line.get(expected)
        if row is None or not _row_recorded(row, actual):
            return True
    return False


def _pct(a: float, b: float) -> float:
    return abs(a - b) / abs(b) * 100.0


def _bm5_conductor_temperature() -> float:
    """Reproduce BM5 via the [Poradowski2025] Table-1 IEC thermal network.

    Uses ``iec_conductor_temperature`` with the paper's *tabulated* T1/T2/T3/T4
    (public, Poradowski Table 1) — NOT the auto-built ladder of compute_thermal,
    which omits the IEC-60287-2-1 three-core geometric factor. Circulating-
    current sheath loss λ1' only; the eddy term λ1″ is BLOCKED (≈5 K residual).
    """
    t1, t2, t3, t4, t_amb = 0.6579, 0.1121, 0.0357, 0.3211, 15.0
    rho = library.get_property("copper_stranded", "rho_ohm_m")
    alpha = library.get_property("copper_stranded", "alpha_R_per_K")
    r_dc = (rho / 1200e-6) * (1.0 + alpha * (90.0 - 20.0))
    y_s = skin_effect_factor(r_dc, 50.0)
    y_p = proximity_effect_factor(r_dc, 50.0, 0.042, 0.13175)
    r_ac = r_dc * (1.0 + y_s + y_p)
    w_c = 908.45**2 * r_ac
    w_d = dielectric_loss(220.0, 2.5, 0.001, 0.0227, 0.0466)
    a_s = math.pi / 4 * (0.11395**2 - 0.11195**2)
    r_s = rho * 1.3406 / a_s * (1.0 + alpha * (80.0 - 20.0))
    x = 2.0 * (2.0 * math.pi * 50.0) * 1e-7 * math.log(2.0 * 0.13175 / 0.1129)
    lam1 = sheath_circulating_loss_factor(r_s, r_ac, x)
    return iec_conductor_temperature(
        t1,
        t2,
        t3,
        t4,
        W_c=w_c,
        W_d=w_d,
        lambda1=lam1,
        n_conductors=3,
        T_ambient_C=t_amb,
    )


def _reader_known_answer_residual() -> float:
    """Max relative residual: time-history CSV reader→fatigue vs closed-form σ_A damage.

    Zero curvature + a constant-amplitude tension square makes σ_total = σ_A
    (σ_B = σ_f = 0), so the per-wire Miner damage is closed-form. Genuinely
    exercises the file parse (writes + reads a time-history CSV export).
    """
    import tempfile  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import rainflow  # noqa: PLC0415

    from cablexsec.materials.sn_curves import get_sn_curve  # noqa: PLC0415
    from cablexsec.mechanics.stiffness import ea_fixed  # noqa: PLC0415

    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    seq = [2.0e4]
    for _ in range(20):
        seq += [6.0e4, 2.0e4]
    tension = np.asarray(seq, dtype=np.float64)
    t = np.arange(tension.size, dtype=np.float64)

    with tempfile.TemporaryDirectory() as d:
        csv_path = Path(d) / "export.csv"
        lines = ["Time (s),x curvature (1/m),y curvature (1/m),Wall tension (kN)"]
        lines += [f"{t[i]},0.0,0.0,{tension[i] / 1e3}" for i in range(tension.size)]
        csv_path.write_text("\n".join(lines), encoding="utf-8")
        series = cx.read_time_history_csv(csv_path)
        res = cx.fatigue_from_series(cable, series)

    ea = ea_fixed(cable)
    worst = 0.0
    for lyr in cable.helical_layers:
        if not lyr.structural:
            continue
        e_w = library.get_property(lyr.material, "E_Pa")
        sig = e_w * math.cos(lyr.lay_angle_rad) ** 2 * tension / ea
        sn = get_sn_curve(lyr.bulk_sn)
        analytic = 0.0
        for rng, _mean, count, _i0, _i1 in rainflow.extract_cycles(sig.tolist()):
            analytic += count / sn.cycles_to_failure(rng / 1.0e6)
        recovered = float(res.D_bulk_per_wire[lyr.name][0])
        if analytic > 0.0:
            worst = max(worst, abs(recovered - analytic) / analytic)
    return worst


def _global_demo_artifact_residual() -> float:
    """Check the tracked global-run history against its recorded run summary."""
    directory = ROOT / "validation" / "paper_data" / "global_demo"
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    series = cx.read_series_csv(directory / "global_history_si.csv")
    magnitude = np.hypot(series.curvature_x_1_m, series.curvature_y_1_m)
    observed = np.array(
        [
            series.n_steps,
            float(np.max(magnitude)),
            float(np.max(series.tension_N)),
            float(np.min(series.tension_N)),
        ],
        dtype=float,
    )
    history = summary["history"]
    recorded = np.array(
        [
            history["samples"],
            history["max_curvature_1_m"],
            history["max_effective_tension_N"],
            history["min_effective_tension_N"],
        ],
        dtype=float,
    )
    return float(np.max(np.abs(observed - recorded) / np.maximum(abs(recorded), 1.0)))


def _shrink_fit_residual() -> float:
    """Rel. error of the QP surface compliances vs the textbook shrink-fit p."""
    from cablexsec.cables.schema import CylindricalLayer  # noqa: PLC0415
    from cablexsec.mechanics.contact_qp import surface_compliances  # noqa: PLC0415

    a, c_r, b, mat = 0.050, 0.055, 0.060, "galvanised_steel_armour"
    e = library.get_property(mat, "E_Pa")
    nu = library.get_property(mat, "nu")
    inner = CylindricalLayer(name="i", material=mat, r_inner_m=a, r_outer_m=c_r)
    outer = CylindricalLayer(name="o", material=mat, r_inner_m=c_r, r_outer_m=b)
    delta = 1.0e-5
    p_qp = delta / (surface_compliances(inner)[1] + surface_compliances(outer)[0])
    bracket = (1 / e) * ((b**2 + c_r**2) / (b**2 - c_r**2) + nu) + (1 / e) * (
        (c_r**2 + a**2) / (c_r**2 - a**2) - nu
    )
    p_cf = delta / (c_r * bracket)
    return abs(p_qp - p_cf) / p_cf


_WITZ_MU = 0.10  # declared before identification (Kraincanic and Kebadze 2001)
_WITZ_P_INT = 30.0e6  # 300 bar gauge [Witz1996] Fig. 6
_WITZ_R_SEAL = 0.0351  # pressure-sheath bore [Witz1996] Table 2; closed-end thrust


def _witz_branches(path: Path) -> dict[str, np.ndarray]:
    rows = [
        line.split(",")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith(("#", "seq"))
    ]
    return {
        b: np.array([[float(r[2]), float(r[3])] for r in rows if r[1] == b])
        for b in ("virgin", "down", "up")
    }


def _shoelace(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _witz_measured(path: Path) -> dict[str, float]:
    """Registered measured metrics of one Witz loop (digitised markers).

    Width at zero curvature: reloading branch minus unloading branch; a
    marker within 0.01 1/m of zero is read directly (the 300-bar record has
    two, giving the interval [first arrival, lower value]). Energy: polygon
    of peak, unloading, reloading and the first-loading markers after the
    reloading branch rejoins them.
    """
    br = _witz_branches(path)
    vir, dn, up = br["virgin"], br["down"], br["up"]
    peak, trough = vir[-1], dn[-1]
    full = np.vstack([peak[None, :], dn])
    near0 = full[np.abs(full[:, 0]) < 0.01]
    rising = np.vstack([trough[None, :], up])
    m_up = float(np.interp(0.0, rising[:, 0], rising[:, 1]))
    if near0.size:
        m_first, m_low = float(near0[0, 1]), float(near0[:, 1].min())
    else:
        order = np.argsort(full[:, 0])
        m_first = m_low = float(np.interp(0.0, full[order, 0], full[order, 1]))
    j = int(np.argmin(np.hypot(vir[:, 0] - up[-1, 0], vir[:, 1] - up[-1, 1])))
    polygon = np.vstack([peak[None, :], dn, up, vir[j + 1 :]])
    return {
        "kappa_max": float(peak[0]),
        "kappa_min": float(trough[0]),
        "width_lo": m_up - m_first,
        "width_hi": m_up - m_low,
        "energy": _shoelace(polygon),
    }


def _witz_loop(
    cable: CableData, measured: dict[str, float], state: cx.SectionState
) -> dict[str, float]:
    """Model loop 0 -> kappa_max -> kappa_min -> kappa_max and its metrics."""
    step = 0.0025
    k_max, k_min = measured["kappa_max"], measured["kappa_min"]
    kappa = np.concatenate(
        [
            np.linspace(0.0, k_max, round(k_max / step) + 1),
            np.linspace(k_max, k_min, round((k_max - k_min) / step) + 1)[1:],
            np.linspace(k_min, k_max, round((k_max - k_min) / step) + 1)[1:],
        ]
    )
    result = cx.compute_mkappa(cable, kappa, mu=_WITZ_MU, section_state=state)
    i_max, i_min = int(np.argmax(kappa)), int(np.argmin(kappa))
    k, m = result.kappa_1_m, result.moment_Nm
    kd, md = k[i_max : i_min + 1], m[i_max : i_min + 1]
    ku, mu = k[i_min:], m[i_min:]
    width = float(np.interp(0.0, ku, mu) - np.interp(0.0, kd[::-1], md[::-1]))
    loop = np.column_stack((np.concatenate([kd, ku[1:]]), np.concatenate([md, mu[1:]])))
    return {
        "width": width,
        "energy": _shoelace(loop),
        "EI_min": float(result.EI_min_Nm2),
    }


def _witz_rows() -> list[Row]:
    """Witz (1996) 2.5-inch pipe: full-slip EI and identify-on-0-bar gate.

    WITZ-EI is parameter-free. WITZ-ID identifies the initial external
    pressure p0 on the 0-bar loop width at zero curvature with mu = 0.10
    declared, then predicts the 300-bar loop (closed ends: wall tension
    p pi r_seal^2, bounded chain) without refitting. Targets were registered
    before any model run (doc/benchmarks.md, Witz section).
    """
    import warnings  # noqa: PLC0415

    from scipy.optimize import brentq  # noqa: PLC0415

    cable = cx.load_cable(cx.witz1996_yaml_path(), geometry_override=None)
    data = ROOT / "validation" / "paper_data"
    m0 = _witz_measured(data / "witz1996_fig5_0bar.csv")
    m3 = _witz_measured(data / "witz1996_fig6_300bar.csv")
    rows: list[Row] = []
    ei = cx.compute_stiffness(cable).ei_min_Nm2
    rows.append(
        Row(
            "2",
            "WITZ-EI",
            "Witz (1996) pipe full-slip EI_min, 0 bar, vs measured gradients "
            "1190 / 1014 N·m² (parameter-free)",
            "within [913, 1309] N·m² (gradients ±10%)",
            f"{ei:.0f} N·m² ({(ei - 1190) / 1190 * 100:+.1f}% / "
            f"{(ei - 1014) / 1014 * 100:+.1f}%)",
            "PASS" if 913.0 <= ei <= 1309.0 else "FAIL",
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", cx.MissingClampingStateWarning)

        def state(p0: float, *, loaded: bool) -> cx.SectionState:
            return cx.solve_section_state(
                cable,
                tension_N=_WITZ_P_INT * math.pi * _WITZ_R_SEAL**2 if loaded else 0.0,
                p_int_Pa=_WITZ_P_INT if loaded else 0.0,
                p_ext_Pa=p0,
                contact_model="bounded",
            )

        p0 = brentq(
            lambda x: (
                _witz_loop(cable, m0, state(x, loaded=False))["width"] - m0["width_lo"]
            ),
            1.0e4,
            1.0e7,
            xtol=1.0,
        )
        zero = _witz_loop(cable, m0, state(p0, loaded=False))
        pred = _witz_loop(cable, m3, state(p0, loaded=True))
    declared = cable.initial_state.external_pressure_Pa
    w_lo, w_hi = 0.85 * m3["width_lo"], 1.15 * m3["width_hi"]
    e_err = (pred["energy"] - m3["energy"]) / m3["energy"]
    w_ok = w_lo <= pred["width"] <= w_hi
    e_ok = abs(e_err) <= 0.15
    rows.append(
        Row(
            "2",
            "WITZ-ID",
            "Witz (1996) 300-bar loop predicted after p0 identified on the "
            "0-bar width at kappa = 0 (mu = 0.10 declared)",
            f"width within [{w_lo:.0f}, {w_hi:.0f}] N·m and energy within 15% "
            f"of {m3['energy']:.0f} J/m",
            f"p0 {p0 / 1e6:.3f} MPa (YAML {declared / 1e6:.3f}); 0-bar energy "
            f"{zero['energy']:.0f} vs {m0['energy']:.0f} J/m; 300-bar width "
            f"{pred['width']:.0f} vs [{m3['width_lo']:.0f}, {m3['width_hi']:.0f}] "
            f"N·m ({(pred['width'] - m3['width_lo']) / m3['width_lo'] * 100:+.1f}% / "
            f"{(pred['width'] - m3['width_hi']) / m3['width_hi'] * 100:+.1f}%), "
            f"energy {pred['energy']:.0f} J/m ({e_err * 100:+.1f}%)",
            "PASS (identified on 0 bar, predicted at 300 bar)"
            if w_ok and e_ok
            else "FAIL",
        )
    )
    return rows


def _saevik_rows() -> list[Row]:
    """Sævik (2011) FBG inner-armour stress ranges (gap 2.5).

    Predicted with mu = 0.20 declared (Sævik 1992 via Dai et al. 2017), the
    bounded section state at the load case's wall tension and pressure, a
    tensioned-beam curvature profile in the bellmouth and the termination
    correction from the end-fitting face; scripts/check_saevik2011.py.
    Targets registered before any model output (doc/benchmarks.md).
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import check_saevik2011 as sv  # noqa: PLC0415

    g = sv.gate_values()
    p, gj = g["plane"], g["gjosteen"]
    r_t = [v["R_termination"] for v in p["cases"].values()]
    r_s = [v["R_sectional"] for v in p["cases"].values()]
    return [
        Row(
            "2",
            "SV11-PLANE",
            "Sævik (2011) Figs 17-23, 14 load cases: normalised inner-armour axial "
            "stress range at the 8 bending-plane FBG stations, termination-"
            "corrected prediction, mu = 0.20 declared",
            "median R in [0.85, 1.18] and >= 12 of 14 R in [0.70, 1.43] "
            "(R = mean predicted / mean measured, bending-plane stations)",
            f"median R {p['median_R_termination']:.2f} ({min(r_t):.2f}-{max(r_t):.2f}), "
            f"{p['count_in_band_termination']} of 14 in band; sectional median "
            f"{p['median_R_sectional']:.2f} ({min(r_s):.2f}-{max(r_s):.2f}), "
            f"{p['count_in_band_sectional']} of 14",
            "PASS" if p["gate_plane"] else "FAIL",
        ),
        Row(
            "2",
            "SV11-GJ",
            "Gjøsteen et al. (2027) Figs 27-35, LC 14 at +-8 deg: absolute axial "
            "range at stations 2.1, 2.5, 3.1, 3.5, termination-corrected, mu = 0.20",
            "mean ratio in [0.72, 1.38]",
            f"{gj['ratio_termination']:.2f} (sectional {gj['ratio_sectional']:.2f}); "
            f"static stress {gj['static_Pa'] / 1e6:.0f} MPa",
            "PASS" if gj["gate"] else "FAIL",
        ),
    ]


def _saevik_split_rows() -> list[Row]:
    """Where the SV11 termination increment comes from (reported, not gated).

    The registered correction (anchor at the end-fitting face, the beam's
    bellmouth curvature profile) split into the end-fitting anchor alone and
    the curvature profile alone; the slip demand at the smallest angles; the
    consistency of the two published bellmouth end points with Eq. 88.
    scripts/check_saevik2011.py (split_values, eq88_consistency).
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import check_saevik2011 as sv  # noqa: PLC0415

    sp = sv.split_values()
    ef, far, reg = sp["ef_only"], sp["no_end_fitting"], sp["registered"]
    small = sp["small_angle"]
    lo, hi = (small[lc] for lc in sv.SMALL_ANGLE_CASES)
    outside = ", ".join(
        f"LC {lc} {sp['R_sectional_outside'][lc]:.2f}"
        for lc in sp["sectional_outside_band"]
    )
    eq = sv.eq88_consistency()
    change = max(far["largest_station_change"], far["largest_gj_change"])
    change_text = "below 1e-12" if change < 1e-12 else f"{change:.1e}"
    return [
        Row(
            "2",
            "SV11-SPLIT",
            "Sævik (2011) termination increment by origin: (a) end-fitting "
            "anchor alone (the correction with uniform curvature from the "
            "end-fitting face), 14 load cases (SV11-PLANE) and LC 14 at +-8 deg "
            "(SV11-GJ); (b) no end fitting (armour anchored 20 m upstream of the "
            "face, bellmouth curvature profile unchanged), LC 14 (SV11-PLANE) and "
            "LC 14 at +-8 deg (SV11-GJ)",
            "stated, not gated",
            f"(a) median R {ef['median_R']:.2f} ({ef['count_in_band']} of 14 in "
            f"band), GJ {ef['gj_ratio']:.2f}; (b) LC 14 R {far['R_lc14']:.2f}, GJ "
            f"{far['gj_ratio']:.2f}, largest change against the registered "
            f"correction {change_text}"
            f"; registered {reg['median_R']:.2f} and {reg['gj_ratio']:.2f}, "
            f"sectional {reg['median_R_sectional']:.2f} and "
            f"{reg['gj_ratio_sectional']:.2f}",
            "INFO",
        ),
        Row(
            "2",
            "SV11-REACH",
            "SV11-PLANE reach of any termination correction: largest slip "
            "demand a = kappa / kappa_c of the correction at 0.5 and 1 deg, and "
            "the load cases whose sectional R is outside [0.70, 1.43]",
            "stated, not gated",
            f"a_max {lo['a_max']:.2f} and {hi['a_max']:.2f}; R corrected "
            f"{lo['R']:.2f} and {hi['R']:.2f} (sectional {lo['R_sectional']:.2f} "
            f"and {hi['R_sectional']:.2f}); sectional outside the band: {outside}",
            "INFO",
        ),
        Row(
            "2",
            "SV11-EQ88",
            "Sævik (2011) Eq. 88, kappa = 0.067 + 0.11 s, at the two published "
            "bellmouth ends (Sævik 1.72 m; Gjøsteen et al. 1.79 m): radius and "
            "slope",
            "Sævik's text: 3.9 m and 16 deg",
            f"1.72 m: {eq['1.72']['radius_m']:.2f} m, "
            f"{eq['1.72']['slope_deg']:.1f} deg; 1.79 m: "
            f"{eq['1.79']['radius_m']:.2f} m, {eq['1.79']['slope_deg']:.1f} deg; "
            f"3.9 m at s = {eq['radius_3.9']['s_m']:.3f} m",
            "INFO",
        ),
    ]


def _anchored_wire_fe_rows() -> list[Row]:
    """Anchored wire at uniform curvature against two published FE studies.

    Code-to-code, reported (doc/benchmarks.md, "Anchored wire at uniform
    curvature against published FE"): the clamp-layer solver against the FE of
    Dong et al. (2025) and of Zhu and Tan (2014). scripts/check_anchored_wire_fe.py.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import check_anchored_wire_fe as aw  # noqa: PLC0415

    r = aw.report()
    dong, zt = r["dong"], r["zt"]
    d90 = dong["cablexsec"][90]
    fe14, th14 = dong["fig14_fe"], dong["fig14_theory"]
    fe25 = dong["fig25_fe"]
    rise = "/".join(
        f"{dong['cablexsec'][t]['anchor']:.1f}" for t in (0, 45, 90, 135, 180)
    )
    rise_fe = "/".join(
        f"{fe25[t]['start']:.1f}" for t in ("0", "45", "90", "135", "180")
    )
    z90, f90 = zt["cablexsec"][90.0], zt["fea"][90.0]
    z45, f45 = zt["cablexsec"][45.0], zt["fea"][45.0]
    rms = zt["fea_far_field_rms"]
    return [
        Row(
            "2",
            "AW-DONG25",
            "Dong et al. (2025) Figs 14 and 25: armour wire anchored at both ends "
            "of one pitch, pipe bent to a uniform 0.005 1/m (a = "
            f"{dong['parameters']['a']:.2f}), clamp-layer solver against the "
            "source's FE: neutral-axis wire (Fig. 14) and anchor stress for "
            "theta0 = 0, 45, 90, 135, 180 deg (Fig. 25) [MPa]",
            "stated, not gated (code-to-code)",
            f"Fig. 14 anchor {d90['anchor']:.1f} vs FE {fe14['start']:.1f} (source "
            f"theory {th14['start']:.1f}); peak {d90['peak']:.1f} at X/P "
            f"{d90['peak_X']:.3f} vs {fe14['peak']:.1f} at {fe14['peak_x']:.3f}; "
            f"Fig. 25 anchors {rise} vs {rise_fe}",
            "INFO",
        ),
        Row(
            "2",
            "AW-ZT14",
            "Zhu and Tan (2014) Figs 4-8: normalised anchor / peak stress of an "
            f"anchored wire at a = {zt['a']:.2f} (from their Table 2), clamp-layer "
            "solver against the source's FE, wires starting at 45 and 90 deg; FE "
            "far field against the full-slip triangle",
            "stated, not gated (code-to-code)",
            f"90 deg {z90['anchor']:.2f} / {z90['peak']:.2f} vs FE "
            f"{f90['start']:.2f} / {f90['peak']:.2f}; 45 deg {z45['anchor']:.2f} / "
            f"{z45['peak']:.2f} vs {f45['start']:.2f} / {f45['peak']:.2f}; FE far "
            f"field RMS distance to a cosine {rms['cosine']:.3f}, to the full-slip "
            f"triangle {rms['triangle']:.3f} (the FE wire is not in slip)",
            "INFO",
        ),
    ]


def _dai2018_rows() -> list[Row]:
    """Static/dynamic friction law against the measured wire-on-tape loops of
    Dai, Sævik and Ye (2018), Fig. 4; scripts/check_dai2018_friction.py.

    Gated on criteria fixed independently of the model errors: the
    digitisation against the printed static coefficients and re-sticking at
    every reversal. The model errors are reported without a target.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import check_dai2018_friction as dai  # noqa: PLC0415

    r = dai.compare()
    maxima = r["digitised_maximum"]
    n = r["strokes"]
    lo, hi = r["excess_range"]
    test1 = [c for c in r["cases"] if c["test"] == 1]
    test3 = [c for c in r["cases"] if c["test"] == 3]

    def errors(case: dict[str, object]) -> str:
        pe, le = case["peak_error_pct"], case["level_error_pct"]
        return (
            f"{case['label']}: peak {pe[0]:+.1f}/{pe[1]:+.1f}%, level "  # type: ignore[index]
            f"{le[0]:+.1f}/{le[1]:+.1f}%, work {case['work_error_pct']:+.1f}%"  # type: ignore[index]
        )

    measured = [m for c in r["cases"] for m in c["measured"]]
    decay = [m.decay_mm for m in measured]
    travel = [
        m.peak_travel_mm / s.peak_travel_mm
        for c in r["cases"]
        for m, s in zip(c["measured"], c["model"], strict=True)
    ]
    levels = ", ".join(f"{v:+.1f}" for v in r["test3_level_change_pct"])
    top = next(c for c in test3 if c["label"] == "0.2 Hz")
    work3 = [c["work_error_pct"] for c in test3]
    return [
        Row(
            "2",
            "DAI18-DIG",
            "Dai et al. (2018) Fig. 4a, Test 1 (armour wire on anti-wear tape, "
            "8/4/1 MPa), digitised: loop maximum vs printed mu_s 0.22/0.24/0.38 "
            "(p. 428)",
            f"within {dai.PRINTED_PRECISION} (printed precision) + reading resolution",
            f"{maxima['8 MPa']:.4f}/{maxima['4 MPa']:.4f}/{maxima['1 MPa']:.4f} "
            f"(largest deviation {r['digitisation_error']:.4f} vs "
            f"{r['digitisation_tolerance']:.4f})",
            "PASS"
            if r["digitisation_error"] <= r["digitisation_tolerance"]
            else "FAIL",
        ),
        Row(
            "2",
            "DAI18-RS",
            "Dai et al. (2018) Fig. 4a/4c, 7 loops: re-stick peak above the dynamic "
            "level at every reversal; spring-slider (FrictionLaw static_factor = "
            "mu_s/mu_d) with stick_time_s = 0 vs the default inf, third cycle",
            f"excess > {dai.PEAK_RESOLUTIONS:g} reading resolutions on all {n} "
            f"measured strokes; stick_time_s = 0 on all {n}; inf on none",
            f"measured {r['measured_peaks']}/{n} (excess {lo:.3f}-{hi:.3f}); "
            f"stick_time_s = 0 {r['model_peaks']}/{n}; inf {r['never_heals_peaks']}/{n}",
            "PASS"
            if r["measured_peaks"] == n
            and r["model_peaks"] == n
            and r["never_heals_peaks"] == 0
            else "FAIL",
        ),
        Row(
            "2",
            "DAI18-T1",
            "Dai et al. (2018) Test 1 loops vs spring-slider with the printed "
            "mu_s/mu_d, stick_time_s = 0 and each loop's stick slope (last 0.5 mm "
            "before each reversal excluded): peak and dynamic level (+x/-x stroke), "
            "stroke work; decay and peak position, Tests 1 and 3",
            "stated, not gated",
            "; ".join(errors(c) for c in test1)
            + f"; decay to 10% of peak-level {min(decay):.1f}-{max(decay):.1f} mm "
            "(model: one step); peak "
            f"{min(travel):.1f}-{max(travel):.1f}x the slider's elastic travel "
            "from the reversal",
            "INFO",
        ),
        Row(
            "2",
            "DAI18-T3",
            "Dai et al. (2018) Test 3 (4 MPa, groove track, 0.01-0.2 Hz) vs the "
            "rate-independent spring-slider with the 0.01 Hz peak and level",
            "stated, not gated",
            f"dynamic level vs 0.01 Hz {levels}% at 0.05/0.1/0.2 Hz (model 0); "
            f"static peak {r['test3_peak_rise_pct']:+.1f}% at 0.2 Hz (model 0: no "
            "rate term on mu_s); 0.2 Hz peak error "
            f"{top['peak_error_pct'][0]:+.1f}/{top['peak_error_pct'][1]:+.1f}%; "
            f"work {min(work3):+.1f} to {max(work3):+.1f}%",
            "INFO",
        ),
    ]


def _friction_pressure_identities() -> tuple[float, float, float]:
    """Pressure factor of the friction law: closed-form slider under pressure
    steps, compiled against NumPy direction-tied kernel under a pressure
    history, and the energy balance of the latter (relative values)."""
    from cablexsec.mechanics.fast_slip import wire_friction_stress  # noqa: PLC0415
    from cablexsec.mechanics.friction_law import (  # noqa: PLC0415
        FrictionLaw,
        slider_history,
    )
    from cablexsec.mechanics.mkappa import HelicalLayerSpec  # noqa: PLC0415
    from cablexsec.mechanics.papailiou import (  # noqa: PLC0415
        full_slip_curvature,
        wire_slip_history,
    )

    table = ((1.0e6, 1.6), (4.0e6, 1.0), (8.0e6, 0.9))
    k, c, step = 2.0e6, 5.0e-6, 1.0e-6
    drive = step * np.arange(1, 601)
    pressure = np.where(np.arange(600) < 200, 8.0e6, 1.0e6)
    pressure[400:] = 4.0e6
    law = FrictionLaw(pressure_factor=table)
    s = slider_history(
        drive, c * pressure, k, friction_law=law, contact_pressure_Pa=pressure
    )
    expected = np.full(600, 36.0)
    expected[:18] = k * drive[:18]
    expected[200:400] = 8.0
    expected[400:] = np.minimum(8.0 + k * (drive[400:] - drive[399]), 20.0)
    slider = float(np.abs(s.force_N - expected).max() / 36.0)
    phi = 2 * np.pi * np.arange(12) / 12
    spec = HelicalLayerSpec(
        n_wires=12,
        r_helix_m=0.05,
        lay_angle_rad=0.4,
        E_wire_Pa=2.0e11,
        A_wire_m2=2.0e-5,
        xi=0.3,
        mu=0.2,
        F_n_axi_N=0.0,
        shape_g=1.0 + 0.3 * np.cos(phi),
        contact_pressure_Pa=2.0e6,
    )
    t = np.linspace(0.0, 12.0, 1200)
    big = 3.0 * full_slip_curvature(spec)
    kx = big * np.sin(2 * np.pi * 0.25 * t)
    ky = 0.4 * big * np.sin(2 * np.pi * 0.13 * t)
    p_hist = 2.0e6 * (1 + 0.4 * np.sin(np.linspace(0, 9, t.size)))
    law_p = FrictionLaw(static_factor=1.3, dynamic_factor=1.0, pressure_factor=table)
    ref = wire_slip_history(
        spec, kx, ky, contact_pressure_Pa=p_hist, friction_law=law_p
    )
    fast = wire_friction_stress(
        spec, kx, ky, contact_pressure_Pa=p_hist, friction_law=law_p
    )
    kernel = float(np.abs(fast - ref.stress_Pa).max() / np.abs(ref.stress_Pa).max())
    balance = float(
        np.abs(ref.work.balance_residual_J_per_m).max()
        / ref.work.dissipation_J_per_m[-1]
    )
    return slider, kernel, balance


_MENARD_BENCH = cx.FourPoint(2.5, 1.25)  # [MenardCartraud2023] Sec. 3, Lg = 4 L1
_MENARD_LVDT = (0.9375, 1.25, 1.5625)  # positions not reported; uniform zone


def _menard_four_point(
    menard: CableData, curvature: np.ndarray, moment: np.ndarray
) -> tuple[float, float]:
    """Range (b) through the four-point arrangement: ordered RMSE and work.

    Curvature control as the test defines it (parabola through the three
    LVDTs); the moment is the test's ``M = Fa L1``; the work is the mechanical
    work of the measuring-zone station over the same augmented cycle as the
    uniform-curvature row.
    """

    def run(path: np.ndarray) -> cx.BendingTestResult:
        return cx.simulate_bending_test(
            menard,
            _MENARD_BENCH,
            cx.BendingProtocol("curvature", path),
            mu=0.2,
            stations_per_segment=12,
            curvature_definition="fit",
            gauge_x_m=_MENARD_LVDT,
        )

    seq = run(np.tile(curvature, 2)).moment_Nm[-len(curvature) :]
    rmse = float(np.sqrt(np.mean((seq - moment) ** 2))) / float(np.ptp(moment)) * 100
    augmented, _, crossings = insert_cycle_crossings(curvature, crossing=0.0)
    res = run(np.tile(augmented, 2))
    i = int(np.argmin(np.abs(res.x_m - res.measuring_x_m)))
    assert res.friction_field_J_per_m is not None
    assert res.stored_field_J_per_m is not None
    mech = (res.friction_field_J_per_m[i] + res.stored_field_J_per_m[i])[
        -len(augmented) :
    ]
    return rmse, float(np.mean(np.diff(mech[crossings])))


def _hu_rows() -> list[Row]:
    """Hu et al. (2022) conductor loop in its three-point arrangement (gap T1).

    Hu Sec. 2.1: hinged supports 440 mm apart, displacement +-6 mm at
    mid-span, third of three cycles, M = F L/4 and kappa = 12 w/L^2
    (Eqs. 1-2). Gates G8-H1 (loop area 3.185 N m/m +-15%) and G8-H2
    (transition share 34% +-10 points), registered before the G8 runs. The
    uniform-curvature comparison of earlier releases is kept, labelled; it
    compares a different quantity.
    """
    from check_conductor_strand_slip import HU_AREA  # noqa: PLC0415
    from check_g8_gradual_reversal import CYCLE, _hu_metrics  # noqa: PLC0415
    from check_t1_test_arrangement import (  # noqa: PLC0415
        HU_SPAN,
        HU_STROKE,
        hu_law,
    )

    caps = (("karlsen_husoy", "KH"), ("papailiou", "Papailiou"))
    arranged: dict[str, dict[str, float]] = {}
    uniform: dict[str, dict[str, float]] = {}
    kappa_rep = 12.0 * HU_STROKE / HU_SPAN**2
    for cap, tag in caps:
        law = hu_law(cap)
        k_u = kappa_rep * np.sin(CYCLE)
        uniform[tag] = _hu_metrics(k_u, law.moment(k_u))
        r = cx.simulate_bending_test(
            law,
            cx.ThreePoint(HU_SPAN),
            cx.BendingProtocol.cyclic(
                "displacement", HU_STROKE, n_cycles=3, samples_per_cycle=500
            ),
            stations_per_segment=24,
        )
        arranged[tag] = _hu_metrics(
            np.r_[0.0, r.curvature_1_m], np.r_[0.0, r.moment_Nm]
        )

    def text(res: dict[str, dict[str, float]]) -> str:
        return "; ".join(
            f"{tag}: area {v['area']:.3f} N·m/m "
            f"({(v['area'] - HU_AREA) / HU_AREA * 100:+.1f}%), "
            f"share {v['share'] * 100:.1f}%, M_max {v['M_max']:.2f} N·m"
            for tag, v in res.items()
        )

    def ok(v: dict[str, float]) -> bool:
        return abs(v["area"] / HU_AREA - 1) <= 0.15 and abs(v["share"] - 0.34) <= 0.10

    target = "loop area 3.185 N·m/m ±15% and transition share 34% ±10 points"
    return [
        Row(
            "2",
            "HU-3PB",
            "Hu et al. (2022) conductor loop, third cycle, in its three-point "
            "arrangement (440 mm, ±6 mm; M = FL/4, kappa = 12w/L²)",
            target,
            text(arranged),
            "PASS" if all(ok(v) for v in arranged.values()) else "FAIL",
        ),
        Row(
            "2",
            "HU-uniform",
            "Hu et al. (2022) conductor loop at uniform curvature "
            "(pre-T1 basis, kept for continuity)",
            target,
            text(uniform),
            "NOT COMPARABLE (the test does not impose uniform curvature)",
        ),
    ]


def _torque_rows() -> list[Row]:
    """Torque balance (gap 3.5); scripts/check_torque_balance.py."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import check_section_torsion as sv  # noqa: PLC0415
    import check_torque_balance as tq  # noqa: PLC0415

    v = tq.verification_checks()
    p = tq.published_checks()
    w = sv.verification_checks()
    # The first two rows are tier 1 and are reported with the analytical rows.
    return [
        Row(
            "1",
            "TQ",
            "Torque balance: Feyrer Ex. 2.8 torque constant c_1S (fixed radius); "
            "coupled section vs Knapp (1981) core factor, theta = 2 nu; mirror "
            "lay reverses torque and twist",
            "0.0345 +/- 0.2%; < 1e-4 relative; < 1e-9 relative",
            f"{v['feyrer_c1S']:.5f} ({v['feyrer_error']:+.2%}); "
            f"{v['knapp_error']:.1e}; {v['mirror']:.1e}",
            "PASS"
            if abs(v["feyrer_error"]) < 2e-3
            and abs(v["knapp_error"]) < 1e-4
            and v["mirror"] < 1e-9
            else "FAIL",
        ),
        Row(
            "1",
            "TQ-SV",
            "Saint-Venant twist rigidity with free warping (trefoil filler lobes): "
            "2 x 1 rectangle vs series solution; bonded annulus as a mirrored "
            "60 deg sector vs polar moment",
            "< 3e-4 relative; < 1e-6 relative",
            f"{w['rectangle']:.1e}; {w['annulus']:.1e}",
            "PASS"
            if abs(w["rectangle"]) < 3e-4 and abs(w["annulus"]) < 1e-6
            else "FAIL",
        ),
        Row(
            "2",
            "TQ-KL",
            "Torque-balance outer helix angle vs Kim and Lee (2017) beam FE, "
            "1+6+12 strand, seven pairs [deg]",
            "abs. difference <= 0.3",
            f"{p['kim_min_deg']:+.2f} to {p['kim_max_deg']:+.2f}",
            "PASS"
            if max(abs(p["kim_min_deg"]), abs(p["kim_max_deg"])) <= 0.3
            else "FAIL",
        ),
        Row(
            "2",
            "TQ-UJ",
            "Fixed-end torque per kN vs Utting and Jones (1987) Part II Table 1, "
            "six strands: model/Machida-Durelli theory; model/test",
            "1.04-1.07; 1.00-1.15",
            f"{p['uj_md_min']:.3f}-{p['uj_md_max']:.3f}; "
            f"{p['uj_test_min']:.3f}-{p['uj_test_max']:.3f}",
            "PASS"
            if p["uj_md_min"] >= 1.04
            and p["uj_md_max"] <= 1.07
            and p["uj_test_min"] >= 1.0
            and p["uj_test_max"] <= 1.15
            else "FAIL",
        ),
        _chae_torsion_row(sv),
    ]


def _chae_torsion_row(sv: ModuleType) -> Row:
    """CH-GJ: model-to-model torsional stiffness of the Chae cable (stated)."""
    gj = sv.chae_checks()
    pitches = sorted(sv.CHAE_TABLE5_GJ)

    def pair(values: dict[float, float]) -> str:
        return " / ".join(f"{values[p] / 1e3:.1f}" for p in pitches)

    tools = " / ".join(
        f"{min(sv.CHAE_TABLE5_GJ[p]) / 1e3:.1f}-{max(sv.CHAE_TABLE5_GJ[p]) / 1e3:.1f}"
        for p in pitches
    )
    return Row(
        "2",
        "CH-GJ",
        "Chae, Nam and Shim (2023) Table 5: free torsional stiffness at "
        "0.125 rad/m (unwinding, zero tension), armour pitch 1.8 / 3.0 m, "
        "against the range of the source's three cross-section tools [kN m^2]: "
        "packaged straight cores; cores laid S as in the source (Sec. 2.1.2) at "
        "an assumed pitch of 2.5 m and 3.0 m (the source gives none)",
        "stated, not gated (model-to-model; no measurement; core pitch not published)",
        f"tools {tools}; straight {pair(gj['straight'])}; S 2.5 m "
        f"{pair(gj['S 2.5 m'])}; S 3.0 m {pair(gj['S 3.0 m'])}",
        "INFO",
    )


def _mkappa_tension_cycle(cable: CableData) -> tuple[float, float]:
    """Tension cycling inside the bending cycles: the M-kappa law against
    the fatigue path (gap a1).

    Bending at +-0.03 1/m (200 samples per cycle) on the Menard cable with
    the tension 10 + 60 sin^2(pi t / 370) kN, incommensurate with the
    bending. Returns the largest moment difference over the largest moment
    and the largest work-balance residual over the total friction work.
    """
    from cablexsec.analysis.fatigue import wire_stress_field  # noqa: PLC0415
    from cablexsec.cable_adapters import cable_to_fatigue_layers  # noqa: PLC0415

    t = np.arange(1200)
    kappa = 0.03 * np.sin(2.0 * np.pi * t / 200.0)
    tension = 1.0e4 + 6.0e4 * np.sin(np.pi * t / 370.0) ** 2
    result = cx.compute_mkappa(cable, kappa, contact_tension_history_N=tension)
    pressure, _ = cx.armour_contact_histories(cable, tension)
    moment = result.EI_min_Nm2 * kappa
    for layer in cable_to_fatigue_layers(cable, contact_tension_N=float(tension[0])):
        friction = wire_stress_field(
            layer,
            kappa,
            np.zeros_like(kappa),
            contact_pressure_Pa=pressure[layer.name],
            keep_friction=True,
        ).friction_Pa
        assert friction is not None
        spec = layer.spec
        lever = (
            spec.A_wire_m2
            * math.cos(spec.lay_angle_rad)
            * spec.r_helix_m
            * np.cos(spec.phi_wires)
        )
        moment = moment + lever @ friction
    deviation = float(
        np.max(np.abs(result.moment_Nm - moment)) / np.max(np.abs(moment))
    )
    assert result.work is not None
    balance = float(
        np.max(np.abs(result.work.balance_residual_J_per_m))
        / result.work.friction_J_per_m[-1]
    )
    return deviation, balance


def _slip_scale_identities(menard: CableData) -> tuple[float, float, float]:
    """Slip scaling variable of the exported bending law (gap a2).

    1. 66 kV template (no manufacturing clamping, so both armour layers and
       the cores scale by T2/T1): compute_mkappa at 70 kN against
       ``EI k + rho F(k / rho)`` from 20 kN over a loop history, cores
       included; largest difference over the largest moment.
    2. Menard, 10 and 60 kN (the layers scale by 1.40 and 2.06): every group
       law decomposed at 60 kN on the grid ``rho_g kappa`` against the 10 kN
       group law with fixed stiffnesses and yield curvatures times ``rho_g``;
       largest relative difference. The element stiffnesses are differences
       of fitted slopes, so this identity holds to a few hundred machine
       epsilons, which differ between platforms (8e-14 and 1.1e-13).
    3. Menard, bending at +-0.03 1/m with the tension 10 + 60 sin^2(pi t /
       370) kN: largest difference of the scaled law from compute_mkappa with
       the contact-tension history, over the larger fixed-tension residual of
       the same Iwan law at 10 and 70 kN (the signed capstan bound makes each
       wire's slip interval asymmetric, which no Masing element carries).
    """
    from cablexsec.analysis.bending_law import (  # noqa: PLC0415
        _slip_scale_table,
    )

    def table(
        cable: CableData, kappa: np.ndarray, reference: float, tensions: list[float]
    ) -> cx.SlipScaleTable:
        return _slip_scale_table(
            cable,
            kappa,
            reference_N=reference,
            candidates_N=np.array([reference]),
            tensions_N=tensions,
            external_pressure_Pa=None,
            options={"core_friction": True} if cable is not menard else {},
        )

    iea = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
    rho = 7.0e4 / 2.0e4
    kappa = (
        4e-4
        * np.sin(np.linspace(0.0, 5.0 * np.pi, 700) + 0.1)
        * np.linspace(0.3, 1.0, 700)
    )
    high = cx.compute_mkappa(iea, kappa, tension_N=7.0e4, core_friction=True)
    low = cx.compute_mkappa(iea, kappa / rho, tension_N=2.0e4, core_friction=True)
    ei = high.EI_min_Nm2
    scaled = ei * kappa + rho * (np.asarray(low.moment_Nm) - ei * kappa / rho)
    homogeneity = float(
        np.max(np.abs(scaled - high.moment_Nm)) / np.max(np.abs(high.moment_Nm))
    )

    grid = np.concatenate([[0.0], np.geomspace(1e-5, 0.3, 199)])
    reference = table(menard, grid, 1.0e4, [1.0e4, 6.0e4])
    group = 0.0
    for g in range(len(reference.group_names)):
        factor = float(reference.group_scale[g, 1, 0])
        state = table(menard, grid * factor, 6.0e4, [6.0e4])
        a, b = reference.group_laws[g], state.group_laws[g]
        group = max(
            group,
            float(np.max(np.abs(a.stiffness_Nm2 - b.stiffness_Nm2)))
            / float(np.max(a.stiffness_Nm2)),
            float(
                np.max(
                    np.abs(factor * a.yield_curvature_1_m - b.yield_curvature_1_m)
                    / b.yield_curvature_1_m
                )
            ),
        )

    t = np.arange(1200)
    bending = 0.03 * np.sin(2.0 * np.pi * t / 200.0 + 0.1)
    tension = 1.0e4 + 6.0e4 * np.sin(np.pi * t / 370.0) ** 2
    export = cx.export_bending_law(
        menard,
        tensions_N=[1.0e4, 7.0e4],
        kappa_max_1_m=0.036,
        n_points=300,
        slip_scale=True,
        slip_scale_tensions_N=np.linspace(1.0e4, 7.0e4, 7),
    )
    law = export.slip_scale
    assert law is not None
    history = np.asarray(
        cx.compute_mkappa(menard, bending, contact_tension_history_N=tension).moment_Nm
    )
    deviation = float(
        np.max(np.abs(law.moment(bending, tension) - history)) / np.ptp(history)
    )
    residual = 0.0
    for level in (1.0e4, 7.0e4):
        fixed = np.asarray(
            cx.compute_mkappa(menard, bending, tension_N=level).moment_Nm
        )
        residual = max(
            residual,
            float(
                np.max(
                    np.abs(law.moment(bending, np.full_like(bending, level)) - fixed)
                )
                / np.ptp(fixed)
            ),
        )
    return homogeneity, group, deviation / residual


def _rows() -> list[Row]:
    rows: list[Row] = _normal_state_rows()
    menard = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    # The 33 kV rows use the pinned pre-correction examples so that the
    # recorded report is reproduced; their geometry is not physically
    # feasible (armour fill above 1) and serves only as an engine input.
    single = cx.load_cable(
        inter_array_single_core_legacy_r2_yaml_path(), geometry_override=None
    )

    # ---------- Tier 1 — analytical / engine-consistency ----------
    # A10 is the central claim of Paper 1: with identical isotropic contact
    # properties and normal-force histories, rotating the wire azimuth rotates
    # the complete vector trajectory without changing its diameter.
    c_geo = geometric_slip_factor(0.08, math.radians(16.0))
    dkx, dky = 0.013, -0.027
    target_norm2 = c_geo**2 * (dkx**2 + dky**2)
    identity_residual = max(
        abs(float(d @ d) - target_norm2) / target_norm2
        for phi in np.linspace(0.0, 2.0 * np.pi, 37)
        for d in [slip_increment(dkx, dky, phi, c_geo, transverse_map="legacy")]
    )
    theta = np.linspace(0.0, 12.0 * np.pi, 1200)
    curvature = np.column_stack(
        (
            0.02 * np.sin(theta) + 0.006 * np.sin(0.37 * theta),
            0.013 * np.cos(0.83 * theta),
        )
    )
    contacts = [MindlinContact(K_t=5.0e6, mu=0.3) for _ in range(2)]
    previous = np.zeros(2)
    for current in curvature:
        delta = current - previous
        previous = current
        for contact, phi in zip(contacts, (0.13, 1.71), strict=True):
            contact.update(
                slip_increment(delta[0], delta[1], phi, c_geo, transverse_map="legacy"),
                80.0,
            )
    amplitudes = [contact.fretting_amplitude for contact in contacts]
    trajectory_residual = abs(amplitudes[0] - amplitudes[1]) / max(amplitudes)
    a10_residual = max(identity_residual, trajectory_residual)
    rows.append(
        Row(
            "1",
            "A10",
            "Conditional vector-slip rotation equivariance",
            "< 1e-12 relative",
            f"{a10_residual:.2e}",
            "PASS" if a10_residual < 1e-12 else "FAIL",
        )
    )

    # Controlled comparator: same fixed-plane increment and contact state; only
    # the kinematic map changes.  This verifies the stated model-class contrast,
    # not the physical adequacy of either map.
    cardinal_phi = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])
    vector_step = slip_increment(
        0.02, 0.0, cardinal_phi, c_geo, transverse_map="legacy"
    )
    loxodromic_step = loxodromic_slip_increment(0.02, 0.0, cardinal_phi, c_geo)
    vector_magnitude = np.linalg.norm(vector_step, axis=-1)
    loxodromic_magnitude = np.linalg.norm(loxodromic_step, axis=-1)
    vector_spread = float(np.ptp(vector_magnitude) / vector_magnitude.mean())
    # With phi=0 at an extreme fibre, the integrated loxodromic displacement is
    # zero at phi={0, pi} and maximal at the neutral axis {pi/2, 3pi/2}.
    extreme_to_neutral_ratio = float(
        max(loxodromic_magnitude[0], loxodromic_magnitude[2])
        / max(loxodromic_magnitude[1], loxodromic_magnitude[3])
    )
    controlled_residual = max(vector_spread, extreme_to_neutral_ratio)
    rows.append(
        Row(
            "1",
            "A10b",
            "Matched fixed-plane vector/loxodromic contrast",
            "< 1e-12 relative",
            f"{controlled_residual:.2e}",
            "PASS" if controlled_residual < 1e-12 else "FAIL",
        )
    )

    # Frictionless slip bounds of Out and von Morgen (1997) and the effective
    # curvature change of Out (1989); scripts/check_out_von_morgen.py.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from check_out_von_morgen import (  # noqa: PLC0415
        closed_form_errors,
        compare_frictionless_contact,
        out1989_contact_checks,
    )

    ovm_closed = max(closed_form_errors(1e-4).values())
    ovm_contact = compare_frictionless_contact(
        alphas_deg=(15.0, 25.0, 35.0, 45.0, 55.0), etas=(0.0,), azimuths=24
    )
    ovm_identity = ovm_contact["longitudinal_identity"]
    ovm_ratio = ovm_contact["magnitude_ratio_inside"]
    rows.append(
        Row(
            "1",
            "OvM",
            "Frictionless contact slip vs Out-von Morgen geodesic (Eqs. 26-28): "
            "loxodromic slip = wire-direction component; magnitude within bound; "
            "closed forms vs exact geodesic",
            "< 1e-12 relative; ratio <= 1; < 1e-3 relative at k = 1e-4",
            f"identity {ovm_identity:.2e}; ratio {ovm_ratio:.3f}; "
            f"closed form {ovm_closed:.2e}",
            "PASS"
            if ovm_identity < 1e-12 and ovm_ratio <= 1.0 and ovm_closed < 1e-3
            else "FAIL",
        )
    )
    from check_out_von_morgen import (  # noqa: PLC0415
        compare_geodesic_map,
        geodesic_map_exact_error,
    )

    geo = compare_geodesic_map(alphas_deg=(15.0, 25.0, 35.0, 45.0, 55.0), azimuths=24)
    geo_closed = geo["closed_form_residual"]
    geo_cycle = max(
        abs(geo["cycle_amplitude_over_eq28_min"] - 1.0),
        abs(geo["cycle_amplitude_over_eq28_max"] - 1.0),
    )
    geo_exact = geodesic_map_exact_error(1e-4, math.radians(35.0))
    rows.append(
        Row(
            "1",
            "OvM-g",
            "Geodesic transverse map (eta = 1) vs Out-von Morgen slip vector "
            "(Eqs. 26-28) at mu = 0; vs exact geodesic",
            "< 1e-12 relative; < 1e-3 relative at k = 1e-4",
            f"closed form {geo_closed:.2e}; delta*/Eq. 28 - 1 {geo_cycle:.1e}; "
            f"exact {geo_exact:.2e}",
            "PASS"
            if geo_closed < 1e-12 and geo_cycle < 1e-12 and geo_exact < 1e-3
            else "FAIL",
        )
    )
    out89 = out1989_contact_checks(points=1000)
    rows.append(
        Row(
            "1",
            "Out89",
            "Contact plastic slip vs Out (1989) effective curvature change "
            "dk - 2 kappa_s (Eq. 1) and damage ratios (Eqs. A8, A11)",
            "< 1e-12 relative; < 1e-4 absolute",
            f"Eq. 1 {out89['eq1_residual']:.2e}; "
            f"A8/A11 {out89['damage_ratio_abs_error']:.1e}",
            "PASS"
            if out89["eq1_residual"] < 1e-12 and out89["damage_ratio_abs_error"] < 1e-4
            else "FAIL",
        )
    )
    # Laid-core stick-slip law (gap R1b); scripts/check_core_friction.py.
    from check_core_friction import verification_checks  # noqa: PLC0415

    cf = verification_checks()
    rows.append(
        Row(
            "1",
            "CF",
            "Laid-core friction: mu = 0 armour-only loop; large mu tangent vs "
            "EI_max; full-slip core moment vs closed form; dissipation >= 0; "
            "rotation equivariance",
            "0 N m; < 1e-9; < 5e-3; >= 0; < 1e-9 relative",
            f"{cf['mu0']:.1e} N m; {cf['inf']:.1e}; {cf['closed']:.1e}; "
            f"{cf['dissipation']:.1e}; {cf['rotation']:.1e}",
            "PASS"
            if cf["mu0"] == 0.0
            and cf["inf"] < 1e-9
            and cf["closed"] < 5e-3
            and cf["dissipation"] >= 0.0
            and cf["rotation"] < 1e-9
            else "FAIL",
        )
    )
    from check_core_friction import wedge_filler_checks  # noqa: PLC0415

    cfw = wedge_filler_checks()
    rows.append(
        Row(
            "1",
            "CFW",
            "Laid-core contact statics (wedged trefoil, central filler) and laid "
            "filler profiles: mu = 0 armour-only loop; full-slip profile moment "
            "vs closed form; stick tangent vs EI_max + E J_f cos^3 / 2",
            "< 1e-12 relative; 0 N m; < 5e-3; < 1e-9",
            f"{cfw['statics']:.1e}; {cfw['mu0']:.1e} N m; {cfw['closed']:.1e}; "
            f"{cfw['stick']:.1e}",
            "PASS"
            if cfw["statics"] < 1e-12
            and cfw["mu0"] == 0.0
            and cfw["closed"] < 5e-3
            and cfw["stick"] < 1e-9
            else "FAIL",
        )
    )

    # Fretting regime and wear (gap 3.1); scripts/check_fretting_regime.py.
    import check_fretting_regime as fret  # noqa: PLC0415

    fv = fret.verification_checks()
    rows.append(
        Row(
            "1",
            "FR",
            "Fretting regime: Mindlin energy vs numerical Masing loop; Fouvry "
            "A/D/B at transition; Cattaneo strip I(0) = pi/16; mu = inf stick "
            "energy; planar-path rotation",
            "< 1e-9; < 1e-12; < 1e-12; 0 J; < 1e-12 relative",
            f"{fv['mindlin']:.1e}; {fv['fouvry']:.1e}; {fv['strip']:.1e}; "
            f"{fv['stick_energy']:.1e} J; {fv['rotation']:.1e}",
            "PASS"
            if fv["mindlin"] < 1e-9
            and fv["fouvry"] < 1e-12
            and fv["strip"] < 1e-12
            and fv["stick_energy"] == 0.0
            and fv["rotation"] < 1e-12
            else "FAIL",
        )
    )

    torque_rows = _torque_rows()
    rows.extend(torque_rows[:2])

    # A11 energy closure on a synthetic dense layer (loop area vs closed form).
    from cablexsec.mechanics.mkappa import HelicalLayerSpec  # noqa: PLC0415

    spec = HelicalLayerSpec(
        180,
        0.082,
        math.radians(16.0),
        210e9,
        math.pi * 0.005**2 / 4,
        0.75,
        0.20,
        1700.0,
    )
    ka = 0.05
    t = np.arange(6000)
    kap = ka * np.sin(2 * np.pi * t / 2000)
    res = compute_mkappa_loop([spec], kap)
    last = slice(4000, 6001)
    w_num = abs(float(np.trapezoid(res.moment_Nm[last], res.kappa[last])))
    w_cf = energy_closure([spec], ka)
    rows.append(
        Row(
            "1",
            "A11",
            "M-κ energy closure (loop area vs closed form)",
            "< 0.1%",
            f"{_pct(w_num, w_cf):.3f}%",
            "PASS" if _pct(w_num, w_cf) < 0.1 else "FAIL",
        )
    )

    # σ_B closed form (Ménard Eq. 14) at φ=0.
    sr = cx.compute_stress(menard, 0.03, tension_N=1.0e4)
    name = next(iter(sr.phi_deg))
    lyr = menard.get_layer(name)
    e = library.get_property(lyr.material, "E_Pa")
    a = lyr.lay_angle_rad
    closed = (
        e * sr.kappa_mag_1_m * math.cos(a) ** 2 * math.cos(2 * a) * (lyr.d_wire_m / 2)
    )
    k0 = int(np.argmin(np.abs(sr.phi_deg[name])))
    rows.append(
        Row(
            "1",
            "σ_B",
            "Radial-fibre bending vs Ménard Eq. 14 (φ=0 from extension side)",
            "< 1e-6%",
            f"{_pct(sr.sigma_B_Pa[name][k0], closed):.2e}%",
            "PASS" if _pct(sr.sigma_B_Pa[name][k0], closed) < 1e-6 else "FAIL",
        )
    )

    # GATE A (cross-engine): the per-wire σ_f assembly is the SAME Papailiou
    # slip-front compute_mkappa integrates, so the assembled moment recovers the
    # public M-κ engine to within per-wire discretization. Evaluated on the
    # PUBLIC single-core cable (it carries a friction-setting contact pressure,
    # so this exercises the σ_f branch — not a degenerate EI_min line).
    sc_kappa = 0.03
    sc_stress = cx.compute_stress(single, sc_kappa, tension_N=0.0)
    sc_mk = cx.compute_mkappa(single, np.linspace(0, sc_kappa, 400), tension_N=0.0)
    m_rec = _pct(sc_stress.moment_recovered_Nm, sc_mk.moment_Nm[-1])
    rows.append(
        Row(
            "1",
            "M-rec",
            "Assembled per-wire moment vs compute_mkappa (cross-engine, single-core)",
            "< 2%",
            f"{m_rec:.3f}%",
            "PASS" if m_rec < 2.0 else "FAIL",
        )
    )

    # Check the offset-core construction with the same mean armour pressure
    # in stress and moment. An inner support reaction is not armour pressure.
    tc = cx.load_cable(
        inter_array_three_core_legacy_r2_yaml_path(), geometry_override=None
    )
    tc_stress = cx.compute_stress(tc, sc_kappa, tension_N=0.0)
    tc_mk = cx.compute_mkappa(tc, np.linspace(0, sc_kappa, 400), tension_N=0.0)
    m_rec_g = _pct(tc_stress.moment_recovered_Nm, tc_mk.moment_Nm[-1])
    rows.append(
        Row(
            "1",
            "M-rec-3core",
            "Cross-engine moment vs compute_mkappa with "
            "offset-core geometry and shared mean armour pressure",
            "< 2%",
            f"{m_rec_g:.3f}%",
            "PASS" if m_rec_g < 2.0 else "FAIL",
        )
    )

    # Tension cycling within the bending cycles: compute_mkappa with a
    # contact-tension history against the moment assembled from the fatigue
    # path's friction stress on the same tension history (the same kernels;
    # the fatigue path may use the compiled recursion), and the incremental
    # work balance under the changing friction bound.
    mk_dev, mk_balance = _mkappa_tension_cycle(menard)
    rows.append(
        Row(
            "1",
            "MK-Tcyc",
            "compute_mkappa with a tension history cycling within the bending "
            "cycles vs the fatigue-path friction stress on the same history; "
            "work balance under the changing friction bound",
            "< 1e-12 relative; < 1e-12 relative",
            f"{mk_dev:.1e}; {mk_balance:.1e}",
            "PASS" if mk_dev < 1e-12 and mk_balance < 1e-12 else "FAIL",
        )
    )

    # Slip scaling variable of the exported bending law (gap a2): the section
    # law is homogeneous in the friction bound; group yield curvatures scale
    # exactly; a tension cycle inside the bending cycles adds nothing to the
    # fixed-tension residual of the Iwan law.
    sc_homog, sc_group, sc_ratio = _slip_scale_identities(menard)
    rows.append(
        Row(
            "1",
            "MX-scale",
            "Exported law with a slip scaling variable: compute_mkappa at T2 vs "
            "the T1 law with the bound scaled (66 kV, cores included); group "
            "elements at T2 vs the T1 elements with yield curvatures scaled "
            "(Menard, two layers); tension cycle inside the bending cycles, "
            "scaled law vs the contact-tension history over the fixed-tension "
            "Iwan residual",
            "< 1e-12 relative; < 1e-9 relative; ratio <= 1.25",
            f"{sc_homog:.1e}; {sc_group:.1e}; {sc_ratio:.2f}",
            "PASS"
            if sc_homog < 1e-12 and sc_group < 1e-9 and sc_ratio <= 1.25
            else "FAIL",
        )
    )

    # Pressure factor of the friction law (mu(P) per contact interface).
    fp_slider, fp_kernel, fp_balance = _friction_pressure_identities()
    rows.append(
        Row(
            "1",
            "FL-P",
            "Pressure-dependent friction coefficient (FrictionLaw.pressure_factor): "
            "spring-slider under pressure steps vs closed form; compiled vs NumPy "
            "direction-tied kernel under a pressure history (static/dynamic law, "
            "local pressures across the table); work balance",
            "< 1e-9 relative; < 1e-9 relative; < 1e-9 relative",
            f"{fp_slider:.1e}; {fp_kernel:.1e}; {fp_balance:.1e}",
            "PASS" if max(fp_slider, fp_kernel, fp_balance) < 1e-9 else "FAIL",
        )
    )

    # GATE B (distinguishing physics): slip starts at the NEUTRAL AXIS, per the
    # Raoof (1989b,1992) strain-gauge observation and Papailiou — not the extreme
    # fibre (which the retired Coulomb loop predicts). The first-to-slip wire's
    # bending lever |cos(φ−φ_bend)| must be ≈ 0 (neutral axis), not ≈ 1. With
    # the signed capstan bound the first slip sits on the compression side at
    # psi ≈ 3 lam / 2 (lam = mu |sin a|), so |cos| is small but not zero.
    name_b = next(iter(sc_stress.phi_deg))
    kc_b = sc_stress.kappa_c_1_m[name_b]
    phi_b = sc_stress.phi_deg[name_b]
    fin = np.isfinite(kc_b)
    k_first = int(np.argmin(np.where(fin, kc_b, np.inf)))
    lever_first = abs(math.cos(math.radians(phi_b[k_first]) - sc_stress.phi_bend_rad))
    rows.append(
        Row(
            "1",
            "slip-onset",
            "First-to-slip wire at the neutral axis "
            "(|cos| → 0, compression side at psi ≈ 3 lam/2; Raoof/Papailiou, "
            "not extreme-fibre loop)",
            "|cos| < 0.2",
            f"|cos|={lever_first:.3f}",
            "PASS" if lever_first < 0.2 else "FAIL",
        )
    )

    # L6.3 coupling bounds gate: xsec_stiffness tangent == EI_max at κ→0 and
    # → EI_min at large κ (ties the CableDyn coupling output to the bounds).
    xs0 = cx.xsec_stiffness(single, 0.0)
    xs_hi = cx.xsec_stiffness(single, 0.5)
    e0 = _pct(xs0.EI_tangent_Nm2, xs0.EI_max_Nm2)
    e1 = _pct(xs_hi.EI_tangent_Nm2, xs_hi.EI_min_Nm2)
    rows.append(
        Row(
            "1",
            "xsec-K",
            "Coupling tangent bounds (κ→0=EI_max, κ→∞=EI_min); slip-front local slope",
            "< 0.01%",
            f"κ0:{e0:.2e}% / κ∞:{e1:.2e}%",
            "PASS" if max(e0, e1) < 0.01 else "FAIL",
        )
    )

    # QP/KKT radial-compliance contact solver: the LCP root satisfies the KKT
    # complementarity q>=0, w>=0, q.w=0 to machine precision (Skeie2012 §2.3 /
    # Eq 2.19) — the radial-compliance formulation the bounded chain is not.
    qp = cx.solve_contact_qp(single, 1.0e5, p_ext_Pa=2.0e6)
    qp_ok = bool(np.all(qp.contact_pressure_Pa >= -1e-9))
    rows.append(
        Row(
            "1",
            "QP-KKT",
            "Concentric contact max|normal line force times gap| "
            "(common extension, twist and radial equilibrium)",
            "< 1e-8 N",
            f"{qp.complementarity_residual:.1e} (q>=0: {qp_ok})",
            "PASS" if qp.complementarity_residual < 1e-8 and qp_ok else "FAIL",
        )
    )

    # QP exact-elasticity gate: the cylinder surface compliances reproduce the
    # textbook compound-cylinder shrink-fit contact pressure (radial-compliance
    # mechanics verified against exact Lamé elasticity, not a thin-shell approx).
    sf = _shrink_fit_residual()
    rows.append(
        Row(
            "1",
            "QP-Lamé",
            "Isolated cylinder surface compliances vs compound-cylinder "
            "shrink-fit closed form",
            "< 1e-6",
            f"{sf:.1e}",
            "PASS" if sf < 1e-6 else "FAIL",
        )
    )

    # All-component gates. These use a two-layer uniform straight cable so the
    # thermoelastic and axial-force balances have closed-form answers and do
    # not depend on a fitted contact or material response.
    component_cable = CableData(
        cable=CableMeta(name="all-component analytical gate"),
        layers=(
            CylindricalLayer(
                name="inner",
                material="hdpe",
                r_inner_m=0.0,
                r_outer_m=0.01,
            ),
            CylindricalLayer(
                name="outer",
                material="hdpe",
                r_inner_m=0.01,
                r_outer_m=0.02,
            ),
        ),
    )
    uniform_temperature = ThermalResult(
        r_m=np.array([0.0, 0.02]),
        T_C=np.array([60.0, 60.0]),
        T_conductor_C=60.0,
        T_armour_C=60.0,
        P_total_W_m=0.0,
    )
    component_thermal = cx.compute_component_response(
        component_cable,
        0.0,
        thermal=uniform_temperature,
        T_ref_C=20.0,
    )
    thermal_residual = max(
        max(abs(state.sigma_stick_min_Pa), abs(state.sigma_stick_max_Pa))
        for state in component_thermal.states
    )
    rows.append(
        Row(
            "1",
            "COMP-T",
            "Uniform free thermal expansion, all components",
            "< 1e-6 Pa residual",
            f"{thermal_residual:.2e} Pa",
            "PASS" if thermal_residual < 1.0e-6 else "FAIL",
        )
    )

    component_tension = 125_000.0
    component_axial = cx.compute_component_response(
        component_cable,
        0.0,
        tension_N=component_tension,
    )
    force_recovered = component_axial.mechanical_axial_strain * ea_fixed(
        component_cable
    )
    force_residual = abs(force_recovered - component_tension) / component_tension
    rows.append(
        Row(
            "1",
            "COMP-N",
            "All-component axial-force equilibrium",
            "< 1e-12 relative",
            f"{force_residual:.2e}",
            "PASS" if force_residual < 1.0e-12 else "FAIL",
        )
    )

    single_component = cx.compute_component_response(
        single,
        0.01,
        tension_N=80_000.0,
    )
    conductor = next(
        state for state in single_component.states if state.name == "cu_conductor"
    )
    offset_residual = max(
        abs(conductor.sigma_stick_min_Pa - conductor.sigma_local_min_Pa),
        abs(conductor.sigma_stick_max_Pa - conductor.sigma_local_max_Pa),
    )
    rows.append(
        Row(
            "1",
            "COMP-O",
            "Zero-offset local and strain-compatible component bounds",
            "< 1e-6 Pa residual",
            f"{offset_residual:.2e} Pa",
            "PASS" if offset_residual < 1.0e-6 else "FAIL",
        )
    )

    shell = CylindricalShellLayer(
        name="lead sheath gate",
        material="lead_sheath",
        r_mid_m=0.04,
        t_m=0.003,
    )
    shell_strain = 2.5e-4
    shell_result = shell_wall_stress(
        shell,
        0.0,
        shell_strain,
        EA_total_N=1.0,
    )
    shell_expected = (
        library.get_property(shell.material, "E_Pa")
        * shell.f_corrugation
        * shell_strain
    )
    shell_residual = abs(shell_result.sigma_axial_Pa - shell_expected) / shell_expected
    rows.append(
        Row(
            "1",
            "SH-AX",
            "Metallic-sheath equal-strain axial stress",
            "< 1e-12 relative",
            f"{shell_residual:.2e}",
            "PASS" if shell_residual < 1.0e-12 else "FAIL",
        )
    )

    # L6.1 reader known-answer: a zero-curvature constant-amplitude tension
    # history makes σ_total = σ_A only, so the per-wire Miner damage is
    # closed-form; the time-history CSV reader path must recover it exactly.
    ka_dmg = _reader_known_answer_residual()
    rows.append(
        Row(
            "1",
            "io-known",
            "Reader→fatigue recovers closed-form σ_A Miner damage (CSV path)",
            "< 1e-10",
            f"{ka_dmg:.2e} rel",
            "PASS" if ka_dmg < 1e-10 else "FAIL",
        )
    )

    global_residual = _global_demo_artifact_residual()
    rows.append(
        Row(
            "1",
            "io-global",
            "Global time-domain SI history reproduces its recorded run summary",
            "< 1e-10",
            f"{global_residual:.2e} rel (201 samples)",
            "PASS" if global_residual < 1.0e-10 else "FAIL",
        )
    )

    # Chae σ_c1 cross-check of the Sævik-Thorsen bird-cage radial mode.
    det = birdcage_buckling_detail(single, single.get_layer("outer_armour"))
    ow = single.get_layer("outer_armour")
    e_w = library.get_property(ow.material, "E_Pa")
    ei = e_w * math.pi * ow.d_wire_m**4 / 64
    area = math.pi * ow.d_wire_m**2 / 4
    chae = (2 * math.cos(ow.lay_angle_rad) / area) * math.sqrt(ei * det.foundation_c_Pa)
    p2_nA = det.P2_radial_N / (ow.n_wires * area)
    rows.append(
        Row(
            "1",
            "σ_c1",
            "Bird-cage Chae Eq.17 cross-check (P2/nA)",
            "< 1e-9%",
            f"{_pct(p2_nA, chae):.2e}%",
            "PASS" if _pct(p2_nA, chae) < 1e-9 else "FAIL",
        )
    )

    # Armour compression screens; scripts/check_compression_buckling.py.
    from check_compression_buckling import all_checks  # noqa: PLC0415

    comp = {c.ident: c for c in all_checks()}

    def _comp_row(tier: str, ident: str, quantity: str, keys: tuple[str, ...]) -> Row:
        checks = [comp[k] for k in keys]
        target = "; ".join(
            f"{c.published:.4g} ± {100 * c.tolerance:g}%" for c in checks
        )
        value = "; ".join(f"{100 * c.error:+.2f}%" for c in checks)
        status = "PASS" if all(c.passed for c in checks) else "FAIL"
        return Row(tier, ident, quantity, target, value, status)

    rows.append(
        _comp_row(
            "2",
            "ST17-T3",
            "Lateral lower bound Eq. 19 vs Saevik-Thorsen Table 3 (6, 8, 14 in) [kN]",
            ("ST17-T3-6in", "ST17-T3-8in", "ST17-T3-14in"),
        )
    )
    fouvry = fret.fouvry_case()
    rows.append(
        Row(
            "2",
            "FOU95",
            "Mindlin transition amplitude vs Fouvry et al. (1995) Table 4 "
            "(HSS / steel ball, 200-500 N, G as printed in Table 3) [um]",
            "; ".join(
                f"{printed} ± {fret.FOUVRY_TOLERANCE_UM}"
                for _p, _mu, printed in fret.FOUVRY_TABLE4
            ),
            "; ".join(f"{row[2]:.2f}" for row in fouvry),
            "PASS"
            if all(
                abs(row[2] - ref[2]) <= fret.FOUVRY_TOLERANCE_UM
                for row, ref in zip(fouvry, fret.FOUVRY_TABLE4, strict=True)
            )
            else "FAIL",
        )
    )
    wear_ratio = fret.wear_case()
    rows.append(
        Row(
            "2",
            "WEAR-CW",
            "Energy wear volume / measured scar volume, crossed 0.45 mm steel "
            "wires at 45 deg (Poon 2024 Tables 4-5, Cruzado 2013 data)",
            f"{fret.WEAR_BAND[0]}-{fret.WEAR_BAND[1]}",
            f"{wear_ratio:.3f}",
            "PASS" if fret.WEAR_BAND[0] <= wear_ratio <= fret.WEAR_BAND[1] else "FAIL",
        )
    )
    rows.append(
        _comp_row(
            "2",
            "LV21",
            "Li-Vaz Eq. 11 single-wire limit [N]; Eq. 15 first slip, mu 0.05/0.1/0.15",
            ("LV21-Pcr", "LV21-slip-0.05", "LV21-slip-0.1", "LV21-slip-0.15"),
        )
    )
    rows.append(
        _comp_row(
            "2",
            "ST17-P34",
            "de Sousa pipe: wire yield P4 (Eq. 34); hoop failure P3 (Eq. 33), "
            "tape / tape + sheath [kN]",
            ("ST17-P4", "ST17-P3t", "ST17-P3ts"),
        )
    )
    rows.append(
        _comp_row(
            "2", "YANG15", "Tape failure load, Yang et al. Eq. 9 [MN]", ("YANG15",)
        )
    )

    # ---------- Tier 2 — experimental (public cables) ----------
    rows.append(
        _comp_row(
            "2",
            "ST17-SOUSA",
            "Straight-beam radial P2* (Eq. 35) vs de Sousa bird-caging test mean [kN]",
            ("ST17-SOUSA",),
        )
    )
    rows.append(
        _comp_row(
            "2",
            "OST12",
            "Eq. 19 per layer vs Ostergaard fixed-free plateaus, Fig. 14 (inner, "
            "outer, sum) [kN]",
            ("OST12-in", "OST12-out", "OST12-sum"),
        )
    )
    rows.extend(torque_rows[2:])
    ei_min = ei_min_slip(menard)
    rows.append(
        Row(
            "2",
            "BM1",
            "Ménard full-slip EI_min (calibration-free)",
            "< 15%",
            f"{_pct(ei_min, _BM1_ANALYTIC_EI_MIN):.1f}% (EI_min {ei_min:.0f} N·m²)",
            "PASS" if _pct(ei_min, _BM1_ANALYTIC_EI_MIN) < 15 else "FAIL",
        )
    )

    # Ménard and Cartraud calibrated mu=0.2 and P=0.3 MPa on the +/-0.05
    # cycle, then held them fixed. Evaluate the separate +/-0.10 cycle as a
    # calibration-transfer validation. The label remains explicit because this
    # is not a parameter-free prediction of the manufacturing state.
    phase_bm1 = np.linspace(0.0, 6.0 * np.pi, 1801)
    kappa_bm1 = 0.1 * np.sin(phase_bm1)
    bm1_prediction = cx.compute_mkappa(
        menard,
        kappa_bm1,
        tension_N=0.0,
        contact_model="bounded",
    )
    measured_path = ROOT / "validation" / "paper_data" / "menard_fig16b_ordered.csv"
    measured = np.genfromtxt(measured_path, delimiter=",", names=True)
    stable_moment = bm1_prediction.moment_Nm[-601:]
    model_min = float(np.min(stable_moment))
    model_max = float(np.max(stable_moment))
    measured_min = float(np.min(measured["moment_Nm"]))
    measured_max = float(np.max(measured["moment_Nm"]))
    measured_curvature = np.asarray(measured["curvature_1_per_m"], dtype=float)
    sequence_prediction = cx.compute_mkappa(
        menard, np.tile(measured_curvature, 2), tension_N=0.0, mu=0.2
    )
    sequence_moment = sequence_prediction.moment_Nm[-len(measured_curvature) :]
    measured_moment = np.asarray(measured["moment_Nm"], dtype=float)
    range_rmse = (
        float(np.sqrt(np.mean((sequence_moment - measured_moment) ** 2)))
        / float(np.ptp(measured_moment))
        * 100.0
    )
    # Table 3 prints energy in J, while M d(kappa) has units J/m. Compare
    # the measured and calculated traces on identical cycles instead of
    # assuming an undocumented length conversion for that table entry.
    augmented, _, crossings = insert_cycle_crossings(measured_curvature, crossing=0.0)
    work_result = cx.compute_mkappa(
        menard, np.tile(augmented, 2), tension_N=0.0, mu=0.2
    )
    work = work_result.work
    mechanical = (work.stored_J_per_m + work.friction_J_per_m)[-len(augmented) :]
    matched_work = float(np.mean(np.diff(mechanical[crossings])))
    measured_work = float(
        np.mean(cycle_work(measured_curvature, measured_moment, crossing=0.0))
    )
    energy_error = _pct(matched_work, measured_work)
    # Gap T1: the same comparison in the test's own arrangement (four-point
    # bench, M = Fa L1, curvature from a parabola through three LVDTs between
    # the actuators; Menard and Cartraud 2023, Sec. 3). The measured curvature
    # lies in the uniform-moment zone, so the result must equal the
    # uniform-curvature row, which is kept for continuity.
    arr_rmse, arr_work = _menard_four_point(menard, measured_curvature, measured_moment)
    arr_error = _pct(arr_work, measured_work)
    for ident, label, rmse_v, work_v, err_v in (
        (
            "BM1-loop",
            "M\u00e9nard +/-0.10 M-kappa after published +/-0.05 calibration, "
            "four-point test arrangement (Sec. 3: M = Fa L1, LVDT parabola)",
            arr_rmse,
            arr_work,
            arr_error,
        ),
        (
            "BM1-loop-uniform",
            "M\u00e9nard +/-0.10 M-kappa after published +/-0.05 calibration, "
            "uniform curvature (pre-T1 basis, kept for continuity)",
            range_rmse,
            matched_work,
            energy_error,
        ),
    ):
        rows.append(
            Row(
                "2",
                ident,
                label,
                "ordered RMSE < 5%; matched experimental cycle work within 10%",
                f"model {model_min:.0f}/{model_max:.0f} N\u00b7m and "
                f"W={work_v:.1f} J/m vs measured {measured_work:.1f} J/m; measured {measured_min:.0f}/"
                f"{measured_max:.0f} N\u00b7m; ordered RMSE {rmse_v:.2f}%; "
                f"matched experimental work difference {err_v:.1f}%",
                "PASS (published calibration transfer)"
                if rmse_v < 5.0 and err_v < 10.0
                else "FAIL",
            )
        )

    # Laid-core law with full contact accounting: P0 identified on range (b),
    # ranges (a), (c)-(f) predicted (gap R1b; registered targets in
    # doc/benchmarks.md).
    from check_core_friction import identify_then_predict  # noqa: PLC0415

    ident = identify_then_predict()
    id_rows = ident["rows"]
    assert isinstance(id_rows, dict)
    rows.append(
        Row(
            "2",
            "BM1-ID",
            "Ménard laid cores, wedged trefoil and filler profiles (mu 0.2): "
            "P0 identified on range (b) cycle work, ranges (a), (c)-(f) predicted",
            "(b) ordered RMSE < 5%; (a), (c)-(f) cycle work within 20% and "
            "ordered RMSE < 10%",
            f"P0 {float(ident['pressure_Pa']) / 1e3:.1f} kPa; "
            + "; ".join(
                f"({tag}) {id_rows[tag]['work_error'] * 100:+.1f}% / "
                f"{id_rows[tag]['rmse_percent']:.2f}%"
                for tag in "abcdef"
            ),
            "PASS (identified, then predicted)" if ident["pass"] else "FAIL",
        )
    )

    rows.extend(_hu_rows())
    rows.extend(_witz_rows())
    rows.extend(_saevik_rows())
    rows.extend(_saevik_split_rows())
    rows.extend(_anchored_wire_fe_rows())
    rows.extend(_dai2018_rows())

    theta_bm5 = _bm5_conductor_temperature()  # paper tabulated network, λ1' only
    rows.append(
        Row(
            "2",
            "BM5",
            "CIGRE TB880 conductor temp (cited Poradowski tabulated T-network, λ1')",
            "≤ 90 °C, ~5 K low",
            f"{theta_bm5:.1f} °C "
            "(FEM 89.62 / IEC 90; λ1″ closes residual). NOT yet from "
            "compute_thermal(cable) — three-core geometric factor pending "
            "IEC 60287-2-1 (not yet implemented)",
            "PARTIAL (cited network; λ1″ + 3-core factor gated)",
        )
    )

    rows.append(
        Row(
            "2",
            "BM4",
            "Saevik-Bruaseth EA population; no-radial-interaction bound",
            "mean model/measured < 1.25",
            "mean ratio about 1.45; U1 > 2.0",
            "NOT VALIDATED (negative finding; radial interaction missing)",
        )
    )
    return rows


def _taxonomy_counts() -> dict[str, str]:
    out: dict[str, str] = {}
    for marker in ("analytical_sanity", "independent_method", "full_scale"):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/",
                "-m",
                marker,
                "-q",
                "-o",
                "addopts=",
                "-p",
                "no:cacheprovider",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            detail = (proc.stdout + "\n" + proc.stderr).strip()
            raise RuntimeError(
                f"pytest taxonomy {marker!r} failed with exit code "
                f"{proc.returncode}:\n{detail}"
            )
        summaries = [line for line in proc.stdout.splitlines() if " passed" in line]
        if not summaries:
            raise RuntimeError(f"pytest taxonomy {marker!r} emitted no pass summary")
        stable_summary = summaries[-1].split(" in ", maxsplit=1)[0]
        # The passed count only: the deselected count is the rest of the suite, so a test
        # added anywhere else would mark the report stale without changing a gate.
        stable_parts = [
            part.strip() for part in stable_summary.split(",") if "passed" in part
        ]
        out[marker] = ", ".join(stable_parts)
    return out


def _normal_state_rows() -> list[Row]:
    """Independent isotropic limits through the public normal-state recovery."""
    material = "galvanised_steel_armour"
    e = library.get_property(material, "E_Pa")
    nu = library.get_property(material, "nu")
    a, b = 0.04, 0.05
    tube = CableData(
        cable=CableMeta(name="normal-state analytical annulus"),
        layers=(
            CylindricalLayer(name="wall", material=material, r_inner_m=a, r_outer_m=b),
        ),
    )
    area = math.pi * (b * b - a * a)
    polar = math.pi / 2 * (b**4 - a**4)
    normal = cx.solve_section_state(
        tube, tension_N=1e5, torque_Nm=150, prescribed_twist_per_m=None
    )
    error_n = abs(normal.axial_strain / (1e5 / (e * area)) - 1)
    error_q = abs(normal.twist_per_m / (150 / (e / (2 * (1 + nu)) * polar)) - 1)
    pressure = 3e6
    closed = cx.solve_section_state(
        tube, tension_N=pressure * math.pi * a * a, p_int_Pa=pressure
    )
    reported = cx.compute_component_response(tube, 0, section_state=closed).states[0]
    exact_vm = math.sqrt(3) * pressure * b * b / (b * b - a * a)
    error_vm = abs(reported.sigma_equivalent_max_Pa / exact_vm - 1)
    thermal = cx.solve_section_state(tube, temperature_C=60)
    hot = cx.compute_component_response(tube, 0, section_state=thermal).states[0]
    alpha = library.get_property(material, "alpha_T_K")
    error_t = abs(thermal.axial_strain / (alpha * 40) - 1)
    # Free expansion leaves no stress; measured against the fully restrained
    # thermal stress E alpha dT, so the gate does not depend on the unit.
    stress_t = hot.sigma_equivalent_max_Pa / (e * alpha * 40)
    return [
        Row(
            "1",
            "STATE-NQ",
            "Annulus extension and Saint-Venant twist",
            "< 1e-10 relative",
            f"extension {error_n:.2e}; twist {error_q:.2e}",
            "PASS" if max(error_n, error_q) < 1e-10 else "FAIL",
        ),
        Row(
            "1",
            "STATE-P",
            "Closed-end thick-tube von Mises stress vs Lame solution",
            "< 1e-10 relative",
            f"{error_vm:.2e}",
            "PASS" if error_vm < 1e-10 else "FAIL",
        ),
        Row(
            "1",
            "STATE-T",
            "Uniform free thermal expansion and stress",
            "< 1e-10 relative strain; < 1e-10 stress relative to E alpha dT",
            f"strain {error_t:.2e}; stress {stress_t:.2e}",
            "PASS" if error_t < 1e-10 and stress_t < 1e-10 else "FAIL",
        ),
    ]


_CORE_LAY_FLAG = "--core-lay-stage"


def _apply_core_lay_stage() -> None:
    """Evaluate every bending law with the core lay stage (sensitivity switch).

    Cables with one family of laid cores get the core lay stage of
    ``compute_mkappa`` with the settings of the Section properties analysis
    (``CoreFrictionSpec(strand_slip=False)``: core friction 0.2, core
    insulation shear in the stick coefficient, no internal ring slip); other
    cables keep the armour-only law. Calls that request core friction
    themselves are left as they are. The rows are printed and nothing is
    written, so the registered report is unchanged.
    """
    import cablexsec.analysis.beam_test as beam_test  # noqa: PLC0415
    import cablexsec.api as api  # noqa: PLC0415
    from cablexsec.exceptions import CableSchemaError  # noqa: PLC0415
    from cablexsec.mechanics.core_friction import (  # noqa: PLC0415
        CoreFrictionSpec,
        _core_geometry,
    )

    original = api._prepare_mkappa_law
    spec = CoreFrictionSpec(strand_slip=False)

    def prepare(cable: CableData, tension_N: float, **options: object) -> object:
        if options.get("core_spec") is None:
            try:
                _core_geometry(cable)
            except CableSchemaError:
                pass
            else:
                options["core_spec"] = spec
        return original(cable, tension_N, **options)  # type: ignore[arg-type]

    api._prepare_mkappa_law = prepare  # type: ignore[assignment]
    beam_test._prepare_mkappa_law = prepare  # type: ignore[assignment]


def main() -> None:
    # Some Windows consoles retain a locale encoding (for example CP949) even
    # when the source and generated artefacts are UTF-8. Reporting must never
    # make a successful validation run fail because a benchmark name contains
    # a character outside that console's code page.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")

    core_lay = _CORE_LAY_FLAG in sys.argv[1:]
    check_only = "--check" in sys.argv[1:] or core_lay
    unknown = [arg for arg in sys.argv[1:] if arg not in ("--check", _CORE_LAY_FLAG)]
    if unknown:
        raise SystemExit(f"unknown argument(s): {' '.join(unknown)}")
    if core_lay:
        _apply_core_lay_stage()
    rows = _rows()
    tax = _taxonomy_counts()
    lines = [
        "<!-- SPDX-License-Identifier: Apache-2.0 -->",
        "",
        "# CableXSec validation report (generated)",
        "",
        "_Generated by `validation/scripts/run_validation.py` from the live code. "
        "Do not edit by hand._",
        "",
        "Every row is computed from published data or closed-form references in "
        "this repository: analytical residuals and benchmark errors on public "
        "cables.",
        "",
        "`PASS` means only that the named gate met its declared target; it does "
        "not by itself mean experimental validation. Interpret every row using "
        "the validation policy in the manual (doc/validation_policy.md).",
        "",
        "| Tier | ID | Quantity | Target | Value | Status |",
        "|------|----|----------|--------|-------|--------|",
    ]
    for r in rows:
        lines.append(r.line())
    lines += [
        "",
        "## Test taxonomy (full gate suite)",
        "",
        f"- analytical_sanity: `{tax['analytical_sanity']}`",
        f"- independent_method: `{tax['independent_method']}`",
        f"- full_scale: `{tax['full_scale']}`",
        "",
        "Run `pytest -m <marker>` to reproduce each group.",
        "",
        "## Benchmark reproducibility audit (cable/registry vs injected input)",
        "",
        "Does each benchmark's pass come from the cable/material definition, or "
        "from an input injected by a harness?",
        "",
        "| BM | Source of inputs | Reproduces from definition? |",
        "|----|------------------|-----------------------------|",
        "| BM1 | `menard2023.yaml` geometry + material library | YES — `ei_min_slip(cable)` |",
        "| BM3 | `nasution_2014_copper` curve in the S-N registry (cited fit) | YES — from the material/S-N definition |",
        "| BM5 | Poradowski Table-1 **tabulated** T1/T2/T3 (cited) | PARTIAL — uses the cited tabulated network; `compute_thermal(cigre_tb880.yaml)` reads high (136 °C at the BM5 inputs, three cores) because the IEC-60287-2-1 three-core geometric factor is not yet applied |",
        "| BM6 | `copper_poon2022` SWT constants (cited registry) + representative contact | curve YES (registry); cable-level contact driver is representative, not from a cable |",
        "| WITZ | `witz1996.yaml` geometry + material library + digitised Witz Figs 5-6 | EI_min YES (parameter-free); WITZ-ID: initial pressure identified on the Fig. 5 width (declared `identified`), Fig. 6 predicted without refitting |",
        "| SV11 | `saevik2011.yaml` geometry (Table 1; outer armour, tapes and sheath assumed) + material library (polymer modulus identified on the Table 1 EI) + rig constants in `validation/scripts/check_saevik2011.py` + digitised Sævik Figs 17-23 and Gjøsteen Figs 27-35 | PARTIAL — section from the cable definition; the curvature profile along the pipe comes from a harness beam model with an assumed rotation point |",
        "",
        "Engine note (RESOLVED): the Kim/Coulomb cross-sectional loop assumes "
        "extreme-fibre full slip and is NOT a valid absolute-dissipation "
        "predictor (it is A11 internal-energy consistent only). BOTH "
        "`compute_mkappa` AND the per-wire `compute_stress` now route through the "
        "single **Papailiou** neutral-axis slip-front engine — experimentally "
        "compared with six public bending ranges. "
        "Corrected work predictions do not meet the historical energy targets. "
        "GATE A (M-rec) confirms the "
        "per-wire σ_f assembly recovers "
        "`compute_mkappa` cross-engine; GATE B (slip-onset) confirms slip starts "
        "at the neutral axis (Raoof/Papailiou), not the extreme fibre. The "
        "Coulomb loop is retired from every public path (A11 reference only).",
        "",
    ]
    text = "\n".join(lines)
    recorded = OUT.read_text(encoding="utf-8") if OUT.is_file() else None
    stale = (
        [OUT]
        if report_is_stale(
            rows,
            text,
            recorded,
            public_only=False,
        )
        else []
    )
    if core_lay:
        # Sensitivity rows differ from the registered report by design.
        print(f"core lay stage sensitivity run ({len(stale)} report(s) differ)")
    elif check_only and stale:
        names = ", ".join(str(path.relative_to(ROOT)) for path in stale)
        # Show what differs, so a stale report can be diagnosed from the log.
        import difflib  # noqa: PLC0415

        diff = difflib.unified_diff(
            (recorded or "").split("\n"),
            text.split("\n"),
            "recorded",
            "live",
            lineterm="",
            n=0,
        )
        for line in list(diff)[:40]:
            print(line)
        raise SystemExit(
            f"validation artefacts are stale: {names}; run validation/scripts/run_validation.py"
        )
    if not check_only:
        OUT.write_text(text, encoding="utf-8")
    failed = [r for r in rows if r.status == "FAIL"]
    review = [r for r in rows if r.status.startswith("REVIEW")]
    flagged = [r for r in rows if "NOT REPRODUCED" in r.status or "PARTIAL" in r.status]
    action = "checked" if check_only else "wrote"
    print(
        f"{action} {OUT.relative_to(ROOT)}: {len(rows)} rows, "
        f"{len(failed)} FAIL, {len(review)} REVIEW, "
        f"{len(flagged)} NOT-REPRODUCED/PARTIAL"
    )
    for r in rows:
        print(f"  [{r.tier}] {r.ident:6s} {r.status:30s} {r.value}")
    for r in review + flagged:
        print(f"\nFLAG [{r.status}] {r.ident}: {r.value}")
    # FAIL and REVIEW are unresolved regressions and fail the executable gate.
    # NOT VALIDATED/PARTIAL rows remain visible limitations and are excluded
    # from validated claims rather than disguised as passing evidence.
    sys.exit(1 if failed or review else 0)


if __name__ == "__main__":
    main()
