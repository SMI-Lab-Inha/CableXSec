# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Fretting regime map, wear and knock-down of public cables under a history.

Applies :func:`cablexsec.compute_fretting` to the public IEA Wind Task 49
66 kV template and the Ménard and Cartraud (2023) cable, and
:func:`cablexsec.analysis.fretting_assessment.conductor_trellis_fretting` to
an ASSUMED 61-wire stranding of the 66 kV conductor (1+6+12+18+24 wires of
the 630 mm2 area, alternate lay, 12 deg; the source tabulates no stranding).

The curvature, tension and time columns are read from section histories
(``<section>_history.npz`` with ``data``/``columns`` arrays, columns
``time_s``, ``true_tension_N``, ``strain_gradient_x_1_m``,
``strain_gradient_y_1_m``). Without ``--history-dir`` a public synthetic
two-component history is used. Results are written as JSON to ``--output``;
keep outputs derived from non-public load histories outside tracked folders.

Run from the repository root::

    python validation/scripts/run_fretting_regime.py --output <file.json>
        [--history-dir <dir> --sections hangoff stiffener_exit ...]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import cablexsec as cx
from cablexsec.analysis.fretting_assessment import conductor_trellis_fretting
from cablexsec.analysis.fretting_regime import REGIMES
from cablexsec.materials import library
from cablexsec.materials.wear import get_wear_coefficient
from cablexsec.mechanics.stiffness import ea_fixed
from cablexsec.mechanics.strand_slip import ConductorRing, conductor_axial_stiffness

CABLES = {
    "iea_task49_66kv": cx.iea_task49_66kv_yaml_path,
    "menard2023": cx.menard2023_yaml_path,
}


def assumed_66kv_conductor() -> list[ConductorRing]:
    """ASSUMED 61-wire stranding of 630 mm2 (wire 3.63 mm, 12 deg, alternate)."""
    d = math.sqrt(630e-6 / 61 * 4 / math.pi)
    rings = [ConductorRing(1, d, 0.0, 0.0)]
    for i, n in enumerate((6, 12, 18, 24), start=1):
        rings.append(ConductorRing(n, d, i * d, math.radians(12.0), (-1.0) ** (i + 1)))
    return rings


def synthetic_history(seconds: float = 600.0, dt: float = 0.1) -> dict[str, np.ndarray]:
    """Public two-component wave-frequency history (no project data)."""
    t = np.arange(0.0, seconds + dt, dt)
    rng = np.random.default_rng(2026)
    kx = np.zeros_like(t)
    ky = np.zeros_like(t)
    for period, amp in ((6.0, 1.0e-3), (8.0, 2.0e-3), (11.0, 1.0e-3), (60.0, 1.5e-3)):
        kx += amp * np.sin(2 * math.pi * t / period + rng.uniform(0, 2 * math.pi))
        ky += 0.4 * amp * np.sin(2 * math.pi * t / period + rng.uniform(0, 2 * math.pi))
    tension = 90e3 + 15e3 * np.sin(2 * math.pi * t / 8.0)
    return {"time_s": t, "kx": kx, "ky": ky, "tension": tension}


def load_history(path: Path) -> dict[str, np.ndarray]:
    record = np.load(path, allow_pickle=False)
    columns = [str(c) for c in record["columns"]]
    data = record["data"]

    def get(name: str) -> np.ndarray:
        return np.asarray(data[:, columns.index(name)], dtype=np.float64)

    return {
        "time_s": get("time_s"),
        "kx": get("strain_gradient_x_1_m"),
        "ky": get("strain_gradient_y_1_m"),
        "tension": np.maximum(get("true_tension_N"), 0.0),
    }


def face_record(face, extreme_deg: float) -> dict:
    share = face.cycles_by_regime.sum(axis=0)
    total = float(share.sum())
    location = face.location()
    offsets = face.extreme_fibre_offset_deg
    bins = {}
    for low, high in ((0, 30), (30, 60), (60, 90.01)):
        sel = (offsets >= low) & (offsets < high)
        cyc = face.cycles_by_regime[sel].sum(axis=0)
        n = float(cyc.sum())
        bins[f"{low}-{int(high)}deg"] = {
            name: (float(cyc[c]) / n if n > 0 else 0.0)
            for c, name in enumerate(REGIMES)
        } | {
            "max_wear_um_per_year": 1e6
            * float(np.max(face.wear_depth_per_year_m[sel], initial=0)),
            "max_D_fretting_per_year": float(
                np.nanmax(face.D_fretting_per_year[sel], initial=0)
            ),
            "max_D_plain_per_year": float(
                np.nanmax(face.D_plain_per_year[sel], initial=0)
            ),
        }
    return {
        "partner": face.partner,
        "mu": face.contact.mu,
        "load_N_per_m": face.contact.normal_load,
        "half_width_um": 1e6 * face.contact.half_width_m,
        "p0_MPa": face.contact.p0_Pa / 1e6,
        "transition_um": 1e6 * face.contact.transition_amplitude_m,
        "max_amplitude_um": 1e6 * float(np.max(face.max_amplitude_m)),
        "slip_to_width_range": [
            float(np.nanmin(face.slip_to_width))
            if np.isfinite(face.slip_to_width).any()
            else None,
            float(np.nanmax(face.slip_to_width))
            if np.isfinite(face.slip_to_width).any()
            else None,
        ],
        "cycle_share": {
            name: (float(share[c]) / total if total > 0 else 0.0)
            for c, name in enumerate(REGIMES)
        },
        "mixed_wires": int(np.sum(face.mixed)),
        "wear_coefficient": None
        if face.contact.wear is None
        else face.contact.wear.pair,
        "wear_assumed": None
        if face.contact.wear is None
        else face.contact.wear.assumed,
        "by_extreme_fibre_offset": bins,
        "location": location,
        "extreme_fibre_deg": extreme_deg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-dir", type=Path, default=None)
    parser.add_argument(
        "--sections", nargs="*", default=["hangoff", "stiffener_exit", "sag_bend"]
    )
    args = parser.parse_args()
    histories: dict[str, dict[str, np.ndarray]] = {}
    if args.history_dir is None:
        histories["synthetic"] = synthetic_history()
    else:
        for section in args.sections:
            histories[section] = load_history(
                args.history_dir / f"{section}_history.npz"
            )
    out: dict = {"histories": {}, "cables": {}}
    for section, h in histories.items():
        out["histories"][section] = {
            "duration_s": float(h["time_s"][-1] - h["time_s"][0]),
            "samples": int(h["time_s"].size),
        }
    for cable_name, path in CABLES.items():
        cable = cx.load_cable(path())
        out["cables"][cable_name] = {}
        for section, h in histories.items():
            start = time.perf_counter()
            duration = float(h["time_s"][-1] - h["time_s"][0])
            result = cx.compute_fretting(
                cable, h["kx"], h["ky"], duration_s=duration, tension_N=h["tension"]
            )
            row = {
                key: face_record(face, result.extreme_fibre_deg)
                for key, face in result.faces.items()
            }
            row["reference_tension_N"] = result.reference_tension_N
            row["runtime_s"] = time.perf_counter() - start
            if cable_name == "iea_task49_66kv":
                rings = assumed_66kv_conductor()
                e_cu = library.get_property("copper_janocha2024", "E_Pa")
                share_t = conductor_axial_stiffness(rings, e_cu) / ea_fixed(cable)
                trellis = conductor_trellis_fretting(
                    rings,
                    e_cu,
                    library.get_property("copper_janocha2024", "nu"),
                    h["kx"],
                    h["ky"],
                    h["tension"] * share_t,
                    mu_wire=library.get_property("copper_janocha2024", "mu_on_cu"),
                    duration_s=duration,
                    wear=get_wear_coefficient("copper_copper"),
                )
                share = trellis.cycles_by_regime.sum(axis=(0, 1))
                row["conductor_trellis_ASSUMED_stranding"] = {
                    "cycle_share": {
                        name: float(share[c] / share.sum()) if share.sum() > 0 else 0.0
                        for c, name in enumerate(REGIMES)
                    },
                    "max_wear_um_per_year": 1e6
                    * float(trellis.wear_depth_per_year_m.max()),
                    "per_ring_gross_share": [
                        float(r[:, 2].sum() / r.sum()) if r.sum() > 0 else 0.0
                        for r in trellis.cycles_by_regime
                    ],
                }
            out["cables"][cable_name][section] = row
            print(f"{cable_name} {section}: {row['runtime_s']:.1f} s", flush=True)
    args.output.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
