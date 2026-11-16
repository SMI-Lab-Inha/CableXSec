# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Sensitivity of the Sævik (2011) termination rows to the anchor stiffness.

Reported, not gated (doc/benchmarks.md, "Anchor compliance of the end
fitting"). The registered rows SV11-PLANE and SV11-GJ (termination-corrected,
clamp solver, ``stick="scaled"``) and the along-wire variant of gap T2
(``stick="spring"``, capstan bound, every wire between the two end fittings,
the beam curvature field at every angle sample) are re-run with an elastic
anchor of stiffness ``K`` per tendon at both end fittings, over a grid that
spans the potting estimate of the termination guide (``1e8``-``2e9`` N/m)
down to a free end. Nothing is fitted; the rigid anchor stays the default.

Usage::

    python validation/scripts/check_saevik2011_anchor.py OUTPUT_JSON
        [--k 2e9 5e8 ...] [--skip-chain] [--cases 14 ...]

The rigid anchor is ``inf`` in ``--k`` and ``null`` in the output. Thread
counts are set to one.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_saevik2011 as sv  # noqa: E402

#: Anchor stiffness per tendon [N/m]; inf is the rigid anchor of the rows.
K_GRID = (math.inf, 2e9, 5e8, 1e8, 1e7, 1e6, 0.0)
LEVELS = 60  # angle samples per leg, as the station histories of the rows


def _label(k: float) -> float | None:
    return None if math.isinf(k) else k


def one_way(k: float, mu: float = sv.MU_DECLARED) -> dict:
    """SV11-PLANE and SV11-GJ, termination-corrected, with anchor ``k``."""
    sv.use_anchor(_label(k))
    try:
        values = sv.gate_values(mu)
    finally:
        sv.use_anchor(None)
    plane, gj = values["plane"], values["gjosteen"]
    return {
        "median_R": plane["median_R_termination"],
        "in_band": plane["count_in_band_termination"],
        "R_per_case": {str(lc): v["R_termination"] for lc, v in plane["cases"].items()},
        "gj_ratio": gj["ratio_termination"],
        "gj_MPa": gj["plane_pred_MPa"],
    }


def _field_table(beam: sv.Beam, half_deg: float) -> np.ndarray:
    """Beam curvature at ``LEVELS + 1`` rocker angles from 0 to ``half``."""
    table = np.zeros((LEVELS + 1, beam.n))
    w = None
    for j in range(1, LEVELS + 1):
        w, k = beam.solve(math.radians(half_deg) * j / LEVELS, w)
        table[j] = k
    return table


def _chain_case(
    case: sv.LoadCase,
    k_values: tuple[float, ...],
    mu: float,
    half_deg: float | None,
    names: tuple[str, ...],
    anchor_diagnostic: bool = False,
) -> dict:
    """Along-wire station ranges [Pa] of one load case for every ``k``.

    ``anchor_diagnostic`` adds, for the rigid anchor and every wire, the
    largest wire stress at the near end fitting and its range over the last
    cycle (``anchor_Pa``), against the largest station range.
    """
    from cablexsec.mechanics.along_wire import (  # noqa: PLC0415
        WireChainLayer,
        along_wire_history,
    )

    _state, layer, static = sv.section(case, mu)
    moment, _ei_max, ei_min = sv.backbone(case, mu)
    beam = sv.Beam(moment, ei_min, case.tension_N)
    half = case.angle_range_deg / 2 if half_deg is None else half_deg
    table = _field_table(beam, half)
    angles = sv._history(1.0, LEVELS)  # in units of the half angle
    level = np.rint(np.abs(angles) * LEVELS).astype(np.int64)
    field = (np.sign(angles)[None, :] * table[level].T).astype(np.float64)
    chain_layer = WireChainLayer.from_layer_spec(
        layer.spec, stick="spring", capstan=True
    )
    stations = {s: (x, az) for s, x, az in sv.stations() if s in names}
    n_w = chain_layer.n_wires
    picks: dict[str, tuple[int, int, float]] = {}
    for s, (x_s, az) in stations.items():
        theta0 = float(chain_layer.azimuth(np.array([x_s]), 0)[0])
        pos = ((math.radians(az) - theta0) % (2 * math.pi)) * n_w / (2 * math.pi)
        i0 = int(math.floor(pos)) % n_w
        picks[s] = (i0, (i0 + 1) % n_w, pos - math.floor(pos))
    wires = sorted({w for i0, i1, _f in picks.values() for w in (i0, i1)})
    rec = np.array([stations[s][0] for s in picks])
    last = slice(angles.size - 4 * LEVELS - 1, angles.size)
    out: dict = {"static_Pa": static, "ranges_Pa": {}}
    for k in k_values:
        hist = along_wire_history(
            chain_layer,
            beam.x,
            field,
            extent_m=(sv.X_ANCHOR_M, sv.X_ANCHOR_M + sv.FREE_LENGTH_M),
            ends=("anchored", "anchored"),
            segments_per_pitch=64,
            record_x_m=rec,
            wires=wires,
            max_increment=0.5,
            anchor_stiffness_N_m=(_label(k), _label(k)),
        )
        row = {w: i for i, w in enumerate(hist.wires.tolist())}
        ranges = {}
        for j, (s, (i0, i1, f)) in enumerate(picks.items()):
            stress = (1 - f) * hist.stress_Pa[row[i0], j] + f * hist.stress_Pa[
                row[i1], j
            ]
            ranges[s] = float(np.ptp(stress[last]))
        out["ranges_Pa"][str(_label(k))] = ranges
    if anchor_diagnostic:
        hist = along_wire_history(
            chain_layer,
            beam.x,
            field,
            extent_m=(sv.X_ANCHOR_M, sv.X_ANCHOR_M + sv.FREE_LENGTH_M),
            ends=("anchored", "anchored"),
            segments_per_pitch=64,
            record_x_m=np.array([sv.X_ANCHOR_M]),
            max_increment=0.5,
        )
        at_anchor = hist.stress_Pa[:, 0, :]
        out["anchor_Pa"] = {
            "max_abs": float(np.abs(at_anchor).max()),
            "max_range_last_cycle": float(np.ptp(at_anchor[:, last], axis=1).max()),
            "largest_station_range": max(
                max(r.values()) for r in out["ranges_Pa"].values()
            ),
        }
    return out


def chain(
    k_values: tuple[float, ...], cases: list[int] | None, mu: float
) -> tuple[dict, dict]:
    """T2-SV-PLANE and T2-SV-GJ with the along-wire chain for every ``k``,
    and the anchor diagnostic of LC 14 at +-8 deg."""
    meas = {
        (int(r["load_case"]), r["station"]): float(r["measured"])
        for r in sv.measured_saevik()
    }
    plane = sv.plane_stations()
    per_k: dict[str, dict] = {str(_label(k)): {"R_per_case": {}} for k in k_values}
    for case in sv.load_cases():
        if cases is not None and case.lc not in cases:
            continue
        t0 = time.time()
        res = _chain_case(case, k_values, mu, None, plane)
        m = np.array([meas[(case.lc, s)] for s in plane])
        for key, ranges in res["ranges_Pa"].items():
            pred = np.array([ranges[s] for s in plane]) / res["static_Pa"]
            per_k[key]["R_per_case"][str(case.lc)] = float(pred.mean() / m.mean())
        print(f"chain LC{case.lc} {time.time() - t0:.1f}s", flush=True)
    for values in per_k.values():
        r = np.array(list(values["R_per_case"].values()))
        values["median_R"] = float(np.median(r)) if r.size else math.nan
        values["in_band"] = int(np.sum((r >= 0.70) & (r <= 1.43)))
    if cases is None or 14 in cases:
        lc14 = next(c for c in sv.load_cases() if c.lc == 14)
        gj_names = ("2.1", "2.5", "3.1", "3.5")
        res = _chain_case(
            lc14, k_values, mu, sv.GJ_HALF_ANGLE_DEG, gj_names, anchor_diagnostic=True
        )
        diagnostic = res["anchor_Pa"]
        gj_meas = sv.measured_gjosteen()
        for key, ranges in res["ranges_Pa"].items():
            pred = [ranges[s] / 1e6 for s in gj_names]
            per_k[key]["gj_MPa"] = dict(zip(gj_names, pred, strict=True))
            per_k[key]["gj_ratio"] = float(
                np.mean(pred) / np.mean([gj_meas[s] for s in gj_names])
            )
    else:
        diagnostic = {}
    return per_k, diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("--k", type=float, nargs="*", default=list(K_GRID))
    parser.add_argument("--mu", type=float, default=sv.MU_DECLARED)
    parser.add_argument("--cases", type=int, nargs="*", default=None)
    parser.add_argument("--skip-chain", action="store_true")
    args = parser.parse_args()
    k_values = tuple(float(k) for k in args.k)
    t0 = time.time()
    lay = sv.section(sv.load_cases()[0], args.mu)[1].spec
    ea = lay.E_wire_Pa * lay.A_wire_m2
    out: dict = {
        "mu": args.mu,
        "EA_N": ea,
        "xi": lay.xi,
        "equivalent_length_m": {
            str(_label(k)): (ea / k if k > 0 else math.inf) for k in k_values
        },
        "one_way": {},
    }
    for k in k_values:
        out["one_way"][str(_label(k))] = one_way(k, args.mu)
        v = out["one_way"][str(_label(k))]
        print(
            f"one-way K={k:g}: median R {v['median_R']:.3f}, {v['in_band']} of 14, "
            f"GJ {v['gj_ratio']:.3f}",
            flush=True,
        )
    if not args.skip_chain:
        out["chain"], out["anchor_diagnostic"] = chain(k_values, args.cases, args.mu)
        for key, v in out["chain"].items():
            print(
                f"chain K={key}: median R {v['median_R']:.3f}, {v['in_band']} of "
                f"{len(v['R_per_case'])}, GJ {v.get('gj_ratio', math.nan):.3f}",
                flush=True,
            )
        print("anchor diagnostic (LC 14, +-8 deg, rigid):", out["anchor_diagnostic"])
    out["seconds"] = time.time() - t0
    args.output.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
