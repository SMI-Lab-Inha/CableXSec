# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Gap T1: published loop benchmarks re-run in their test arrangements.

Run as ``python validation/scripts/check_t1_test_arrangement.py [--json PATH] [--skip-witz]``.
Targets were registered before the runs (doc/benchmarks.md, section "Test
arrangement (gap T1)"); nothing is fitted. Each arrangement is taken from the
set-up section of its source:

1. Hu et al. (2022), Sec. 2.1 and Fig. 5: copper conductor between hinged
   supports 440 mm apart, displacement control +-6 mm at mid-span, three
   cycles, the third reported, M = F L/4 and kappa = 12 w/L^2 (Eqs. 1-2).
   Section law: the package's conductor ring parameters with the
   Karlsen-Husoy amplitude (default) or the Papailiou cap. Compared with the
   uniform-curvature section loop at the same reduced amplitude.
2. Menard and Cartraud (2023), Sec. 3: four-point bench 2.5 m, actuators at
   the quarter points, uniform moment between them, M = Fa L1 (Eq. 1),
   curvature by a parabola through three LVDTs between the actuators (their
   positions are not reported; quarter points of the zone are used; the zone
   curvature is uniform, so they do not matter). Driven along the measured
   curvature path of each range with the transferred inputs (mu 0.2).
3. Witz (1996) reports no test equipment or procedure (p. 888). The
   uniform-curvature comparison stays the basis; WITZ-ID is repeated with
   both loops reduced as three-point tests (span-independent) as a
   sensitivity only.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
from check_conductor_strand_slip import HU_AREA, HU_E, NAM_M_MAX
from check_g8_gradual_reversal import (
    CYCLE,
    HU_SHARE,
    HU_TENSION,
    _hu_metrics,
    _hu_setup,
)
from scipy.optimize import brentq

import cablexsec as cx
from cablexsec.analysis.beam_test import (
    BendingProtocol,
    FourPoint,
    SectionLaw,
    ThreePoint,
    simulate_bending_test,
)
from cablexsec.analysis.curve_comparison import cycle_work, insert_cycle_crossings
from cablexsec.mechanics.core_lay import bending_factor

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "validation" / "paper_data"
HU_SPAN = 0.440  # Hu Sec. 2.1
HU_STROKE = 0.006  # Hu Sec. 2.1, +-6 mm
MENARD = FourPoint(2.5, 1.25)  # Menard and Cartraud Sec. 3: Lg = 4 L1 = 2.5 m
MENARD_GAUGES = (0.9375, 1.25, 1.5625)  # assumed; uniform zone (see docstring)


# --------------------------------------------------------------------------- #
# Hu                                                                           #
# --------------------------------------------------------------------------- #
def hu_law(cap: str) -> SectionLaw:
    rings, params, ei = _hu_setup()
    wire = sum(
        HU_E
        * r.n_wires
        * math.pi
        * r.d_wire_m**4
        / 64
        * bending_factor(r.lay_angle_rad)
        for r in rings
    )
    return SectionLaw.from_conductor(
        rings,
        HU_E,
        params,
        tension_N=HU_TENSION,
        additional_EI_Nm2=ei - wire,
        cap_shape=cap,  # type: ignore[arg-type]
    )


def hu_cases(stations: tuple[int, ...] = (24, 48, 96)) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    kappa_rep = 12.0 * HU_STROKE / HU_SPAN**2
    for cap in ("karlsen_husoy", "papailiou"):
        law = hu_law(cap)
        k_u = kappa_rep * np.sin(CYCLE)
        out[f"uniform, {cap}"] = _hu_metrics(k_u, law.moment(k_u))
        for n in stations:
            t0 = time.time()
            r = simulate_bending_test(
                law,
                ThreePoint(HU_SPAN),
                BendingProtocol.cyclic(
                    "displacement", HU_STROKE, n_cycles=3, samples_per_cycle=500
                ),
                stations_per_segment=n,
            )
            k = np.r_[0.0, r.curvature_1_m]
            m = np.r_[0.0, r.moment_Nm]
            met = _hu_metrics(k, m)
            met.update(
                loop_work=r.loop_work_J_per_m(),
                residual=r.equilibrium_residual,
                seconds=time.time() - t0,
            )
            out[f"three-point n={n}, {cap}"] = met
    return out


# --------------------------------------------------------------------------- #
# Menard                                                                       #
# --------------------------------------------------------------------------- #
def _station_work(cable: cx.CableData, path: np.ndarray, definition: str) -> tuple:
    r = simulate_bending_test(
        cable,
        MENARD,
        BendingProtocol("curvature", path),
        mu=0.2,
        stations_per_segment=12,
        curvature_definition=definition,  # type: ignore[arg-type]
        gauge_x_m=MENARD_GAUGES,
    )
    i = int(np.argmin(np.abs(r.x_m - r.measuring_x_m)))
    assert r.friction_field_J_per_m is not None and r.stored_field_J_per_m is not None
    mech = r.friction_field_J_per_m[i] + r.stored_field_J_per_m[i]
    return r, mech


def menard_cases() -> dict[str, dict[str, float]]:
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    out: dict[str, dict[str, float]] = {}
    for rng in "abcdef":
        d = np.genfromtxt(
            DATA / f"menard_fig16{rng}_ordered.csv", delimiter=",", names=True
        )
        k = np.asarray(d["curvature_1_per_m"], float)
        m = np.asarray(d["moment_Nm"], float)
        crossing = 0.0 if rng in "ab" else None
        measured = float(np.mean(cycle_work(k, m, crossing=crossing)))
        aug, _, cross = insert_cycle_crossings(k, crossing=crossing)
        # uniform curvature (the scorecard basis so far)
        seq_u = cx.compute_mkappa(cable, np.tile(k, 2), mu=0.2).moment_Nm[-k.size :]
        work_u = cx.compute_mkappa(cable, np.tile(aug, 2), mu=0.2).work
        mech_u = (work_u.stored_J_per_m + work_u.friction_J_per_m)[-len(aug) :]
        w_u = float(np.mean(np.diff(mech_u[cross])))
        # four-point arrangement, the test's own reduction (Eq. 1, LVDT fit)
        r_seq, _ = _station_work(cable, np.tile(k, 2), "fit")
        seq_a = r_seq.moment_Nm[-k.size :]
        r_w, mech_a = _station_work(cable, np.tile(aug, 2), "fit")
        w_a = float(np.mean(np.diff(mech_a[-len(aug) :][cross])))

        def rmse(s: np.ndarray, m: np.ndarray = m) -> float:
            return float(np.sqrt(np.mean((s - m) ** 2)) / np.ptp(m) * 100)

        out[rng] = {
            "measured_work": measured,
            "rmse_uniform": rmse(seq_u),
            "rmse_four_point": rmse(seq_a),
            "work_uniform": w_u,
            "work_four_point": w_a,
            "work_err_uniform": (w_u - measured) / measured * 100,
            "work_err_four_point": (w_a - measured) / measured * 100,
            "max_moment_difference": float(np.max(np.abs(seq_a - seq_u))),
            "max_curvature_reading_error": float(
                np.max(np.abs(r_seq.curvature_1_m - np.tile(k, 2)))
            ),
            "residual": max(r_seq.equilibrium_residual, r_w.equilibrium_residual),
        }
    return out


# --------------------------------------------------------------------------- #
# Witz (sensitivity only)                                                      #
# --------------------------------------------------------------------------- #
def witz_cases(step: float = 0.0025) -> dict[str, dict[str, float]]:
    import run_validation as rv  # noqa: PLC0415

    cable = cx.load_cable(cx.witz1996_yaml_path(), geometry_override=None)
    m0 = rv._witz_measured(DATA / "witz1996_fig5_0bar.csv")
    m3 = rv._witz_measured(DATA / "witz1996_fig6_300bar.csv")

    def state(p0: float, loaded: bool) -> cx.SectionState:
        return cx.solve_section_state(
            cable,
            tension_N=rv._WITZ_P_INT * math.pi * rv._WITZ_R_SEAL**2 if loaded else 0.0,
            p_int_Pa=rv._WITZ_P_INT if loaded else 0.0,
            p_ext_Pa=p0,
            contact_model="bounded",
        )

    def loop(
        meas: dict[str, float], st: cx.SectionState, arranged: bool
    ) -> tuple[float, float]:
        hi, lo = meas["kappa_max"], meas["kappa_min"]
        kappa = np.concatenate(
            [
                np.linspace(0.0, hi, round(hi / step) + 1),
                np.linspace(hi, lo, round((hi - lo) / step) + 1)[1:],
                np.linspace(lo, hi, round((hi - lo) / step) + 1)[1:],
            ]
        )
        law = SectionLaw.from_cable(cable, mu=rv._WITZ_MU, section_state=st)
        if arranged:
            r = simulate_bending_test(
                law, ThreePoint(1.0), BendingProtocol("curvature", kappa[1:])
            )
            moment = np.r_[0.0, r.moment_Nm]
            kappa = np.r_[0.0, r.curvature_1_m]
        else:
            moment = law.moment(kappa)
        # peaks from the commanded path (the reduced curvature repeats its
        # extremes to round-off, so argmax could pick the last peak)
        i_max = round(hi / step)
        i_min = i_max + round((hi - lo) / step)
        kd, md = kappa[i_max : i_min + 1], moment[i_max : i_min + 1]
        ku, mu = kappa[i_min:], moment[i_min:]
        width = float(np.interp(0.0, ku, mu) - np.interp(0.0, kd[::-1], md[::-1]))
        poly = np.column_stack((np.r_[kd, ku[1:]], np.r_[md, mu[1:]]))
        return width, rv._shoelace(poly)

    out: dict[str, dict[str, float]] = {}
    for arranged in (False, True):
        t0 = time.time()
        p0 = brentq(
            lambda x, arranged=arranged: (
                loop(m0, state(x, False), arranged)[0] - m0["width_lo"]
            ),
            1.0e4,
            1.0e7,
            xtol=10.0,
        )
        e0 = loop(m0, state(p0, False), arranged)[1]
        width, energy = loop(m3, state(p0, True), arranged)
        out["three-point sensitivity" if arranged else "uniform"] = {
            "p0_MPa": p0 / 1e6,
            "energy_0bar": e0,
            "width": width,
            "energy": energy,
            "energy_per_width": energy / width,
            "width_lo": m3["width_lo"],
            "width_hi": m3["width_hi"],
            "energy_measured": m3["energy"],
            "energy_0bar_measured": m0["energy"],
            "seconds": time.time() - t0,
        }
    return out


def _print_witz(result: dict[str, object]) -> None:
    print("Witz (1996) WITZ-ID, uniform and three-point sensitivity", flush=True)
    witz = witz_cases()
    result["witz"] = witz
    for tag, r in witz.items():
        print(
            f"  {tag:24s} p0 {r['p0_MPa']:.4f} MPa, width {r['width']:.0f} N m "
            f"(measured {r['width_lo']:.0f}-{r['width_hi']:.0f}), energy "
            f"{r['energy']:.0f} J/m (measured {r['energy_measured']:.0f}), "
            f"energy/width {r['energy_per_width']:.3f} 1/m; 0-bar energy "
            f"{r['energy_0bar']:.0f} (measured {r['energy_0bar_measured']:.0f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--skip-witz", action="store_true")
    parser.add_argument("--only-witz", action="store_true")
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    result: dict[str, object] = {}
    if args.only_witz:
        _print_witz(result)
        if args.json is not None:
            args.json.write_text(json.dumps(result, indent=1, default=float))
        return
    print("Hu et al. (2022), third cycle, +-6 mm (kappa_rep 0.3719 1/m)")
    print(
        f"  measured: area {HU_AREA} N m/m, M_max {NAM_M_MAX} N m, share {HU_SHARE:.1%}"
    )
    hu = hu_cases()
    result["hu"] = hu
    for name, r in hu.items():
        print(
            f"  {name:34s} area {r['area']:.3f} ({(r['area'] / HU_AREA - 1) * 100:+.1f}%), "
            f"M_max {r['M_max']:.2f}, share {r['share']:.1%}"
        )
    print(
        "Menard (2023), transferred inputs: RMSE [%] and work error [%], uniform -> four-point"
    )
    menard = menard_cases()
    result["menard"] = menard
    for rng, r in menard.items():
        print(
            f"  ({rng}) RMSE {r['rmse_uniform']:.2f} -> {r['rmse_four_point']:.2f}; work "
            f"{r['work_err_uniform']:+.1f} -> {r['work_err_four_point']:+.1f}; "
            f"max moment difference {r['max_moment_difference']:.2e} N m"
        )
    if not args.skip_witz:
        _print_witz(result)
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()
