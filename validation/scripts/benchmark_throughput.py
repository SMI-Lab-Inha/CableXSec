# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Throughput of the sea-state fatigue path on one core.

Run as ``python validation/scripts/benchmark_throughput.py --output DIR``
with ``NUMBA_NUM_THREADS=1`` (and the other thread variables at ``1``) on a
quiet machine. The workload is a one-hour synthetic sea state on the 66 kV
template (:func:`cablexsec.iea_task49_66kv_yaml_path`): irregular curvature
components ``kappa_x`` and ``kappa_y`` (peak period 10 s, standard deviations
0.004 and 0.002 1/m) and tension ``100 +- 10 kN``, ``--steps`` samples at
``--dt`` seconds. Each case calls :func:`cablexsec.compute_fatigue` on every
armour wire, or on 12 positions per layer, with the direction-tied or the
distributed friction model and 4 or 32 fibres per wire. Timings exclude
imports, cable loading and the compilation of the kernels (a short warm-up
call); every case reports the minimum of ``--repeat`` calls and the time per
series-step (one stress evaluation and its share of the counting at one
fibre of one wire at one sample).

The damage of every case is written to ``damage.npz``; ``--reference`` names
the output directory of an earlier run (another build) and records the
largest absolute and relative differences of every array.

``--batch-workers`` also times :func:`cablexsec.run_sea_state_batch` over
``--batch-runs`` synthetic CableDyn runs (written to a temporary folder) for
each worker count and checks that the annual damage does not depend on it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import warnings
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray
from profile_release import positive_int, source_digest

import cablexsec as cx
from cablexsec._numba import kernels_compiled, numba_version

FloatArray = NDArray[np.float64]

#: compute_fatigue options of each case.
CASES: dict[str, dict[str, Any]] = {
    "direction_tied_f4": {"friction_model": "direction_tied", "n_fibres": 4},
    "direction_tied_f32": {"friction_model": "direction_tied", "n_fibres": 32},
    "direction_tied_12pos_f4": {
        "friction_model": "direction_tied",
        "n_fibres": 4,
        "wire_positions": 12,
    },
    "distributed_f4": {"friction_model": "distributed", "n_fibres": 4},
    "distributed_f32": {"friction_model": "distributed", "n_fibres": 32},
    "distributed_12pos_f4": {
        "friction_model": "distributed",
        "n_fibres": 4,
        "wire_positions": 12,
    },
}
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


def sea_state(
    steps: int, dt: float, seed: int
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Time, ``kappa_x``, ``kappa_y`` [1/m] and tension [N] of one sea state."""
    rng = np.random.default_rng(seed)
    t = np.arange(steps) * dt
    f = np.linspace(0.04, 0.3, 200)
    spectrum = np.exp(-(((f - 0.1) / 0.02) ** 2)) + 0.1 * np.exp(
        -(((f - 0.2) / 0.05) ** 2)
    )
    amplitude = np.sqrt(spectrum)
    amplitude /= np.sqrt(0.5 * (amplitude**2).sum())

    def component(std: float) -> FloatArray:
        phase = rng.uniform(0.0, 2.0 * math.pi, f.size)
        waves = np.cos(2.0 * math.pi * f[None, :] * t[:, None] + phase[None, :])
        return np.asarray(std * (amplitude[None, :] * waves).sum(axis=1))

    return t, component(0.004), component(0.002), 100e3 + component(10e3)


def time_cases(
    cable: cx.CableData,
    names: list[str],
    steps: int,
    dt: float,
    repeat: int,
) -> tuple[list[dict[str, Any]], dict[str, FloatArray]]:
    _, kx, ky, tension = sea_state(500, dt, 1)
    for name in names:  # compile and load every kernel the cases use
        cx.compute_fatigue(cable, kx, ky, tension_N=tension, **CASES[name])
    _, kx, ky, tension = sea_state(steps, dt, 7)
    rows, arrays = [], {}
    for name in names:
        seconds = []
        for _ in range(repeat):
            start = perf_counter()
            result = cx.compute_fatigue(cable, kx, ky, tension_N=tension, **CASES[name])
            seconds.append(perf_counter() - start)
        series = sum(int(v.size) for v in result.D_bulk_per_fibre.values())
        best = min(seconds)
        rows.append(
            {
                "case": name,
                "options": dict(CASES[name]),
                "seconds": seconds,
                "seconds_min": best,
                "series": series,
                "ns_per_series_step": best / (series * steps) * 1e9,
                "D_total_max": max(
                    float(v.max()) for v in result.D_total_per_wire.values()
                ),
            }
        )
        for layer, values in result.D_bulk_per_fibre.items():
            arrays[f"{name}/{layer}/D_bulk_per_fibre"] = values
        for layer, values in result.D_total_per_wire.items():
            arrays[f"{name}/{layer}/D_total_per_wire"] = values
        print(
            f"{name:26s} {best:8.3f} s  {rows[-1]['ns_per_series_step']:8.2f} ns/series-step",
            flush=True,
        )
    return rows, arrays


def differences(
    arrays: dict[str, FloatArray], reference: Path
) -> dict[str, dict[str, float]]:
    """Largest absolute and relative differences from an earlier run."""
    out = {}
    with np.load(reference / "damage.npz") as old:
        for key, values in arrays.items():
            if key not in old.files:
                continue
            before = old[key]
            diff = np.abs(values - before)
            scale = np.abs(before)
            relative = np.divide(diff, scale, out=np.zeros_like(diff), where=scale > 0)
            out[key] = {
                "max_abs": float(diff.max(initial=0.0)),
                "max_rel": float(relative.max(initial=0.0)),
                "bitwise_equal": bool(np.array_equal(values, before)),
            }
    return out


# --------------------------------------------------------------------------- #
# Sea-state batch over synthetic CableDyn runs                                 #
# --------------------------------------------------------------------------- #
def _fmt(values: Any) -> str:
    return "\t".join(f"{v: .16E}" for v in values)


def write_run(
    folder: Path,
    root: str,
    steps: int,
    dt: float,
    seed: int,
    ea: float,
    ei: float,
    n_elements: int = 10,
    length: float = 10.0,
) -> tuple[Path, Path]:
    """One CableDyn run: a bent arc whose curvature and plane follow a sea state."""
    _, a, b, tension = sea_state(steps, dt, seed)
    kappa = 0.03 + 0.5 * a  # positive arc curvature [1/m]
    turn = 20.0 * b  # rotation of the bending plane [rad]
    s = length * np.arange(n_elements + 1) / n_elements
    t0 = np.array([1.0, 0.0, 0.0])
    deck = folder / f"{root}.dat"
    deck.write_text(
        "--------------------- CableDyn Input File ---\n"
        "synthetic throughput benchmark\n"
        "--------------------- LINE TYPES ---\n"
        "TypeName Diam MassDenInAir EA BA EI Cd_n Cd_t Ca_n Ca_t\n"
        "(-) (m) (kg/m) (N) (-) (N-m2) (-) (-) (-) (-)\n"
        f"cable 0.2 40.0 {ea!r} 0.0 {ei!r} 1.2 0.1 1.0 0.0\n"
        "--------------------- POINTS ---\n"
        "ID Type X Y Z\n(-) (-) (m) (m) (m)\n"
        "1 Coupled 0 0 -10\n2 Fixed 10 0 -12\n"
        "--------------------- LINES ---\n"
        "ID NodeA NodeB Outputs\n(-) (-) (-) (-)\n"
        "1 1 2 pt\n"
        "--------------------- SECTIONS ---\n"
        "LineID LineType Length NumSegs\n(-) (-) (m) (-)\n"
        f"1 cable {length} {n_elements}\n"
        "--------------------- OPTIONS ---\n"
        "9.81 g - gravity\n1025.0 rhoW - density\n"
        f"{dt} dtM - step\n{(steps - 1) * dt} TMax - duration\n"
        "--------------------- OUTPUTS ---\n"
        "--------------------- need this line ---\n",
        encoding="utf-8",
    )
    nodes = n_elements + 1
    positions = np.empty((steps, nodes, 3))
    for k in range(steps):
        binormal = np.array([0.0, math.cos(turn[k]), math.sin(turn[k])])
        normal = np.cross(binormal, t0)
        theta = kappa[k] * s
        arc = np.sin(theta)[:, None] * t0 + (1.0 - np.cos(theta))[:, None] * normal
        positions[k] = np.array([0.0, 0.0, -10.0]) + arc / kappa[k]
    time = np.arange(steps) * dt
    with open(folder / f"{root}.out", "w", encoding="utf-8") as f:
        f.write("# CableDyn driver output (synthetic)\nTime(s)\n")
        for k in range(steps):
            f.write(_fmt([time[k]]) + "\n")
    with open(folder / f"{root}.Line1.p.out", "w", encoding="utf-8") as f:
        f.write("# CableDyn line node positions; public node order EndA -> EndB\n")
        names = [f"Node{j}{c}(m)" for j in range(1, nodes + 1) for c in "XYZ"]
        f.write("\t".join(["Time(s)", *names]) + "\n")
        for k in range(steps):
            f.write(_fmt([time[k], *positions[k].ravel()]) + "\n")
    with open(folder / f"{root}.Line1.t.out", "w", encoding="utf-8") as f:
        f.write("# CableDyn segment tensions; public order EndA -> EndB\n")
        names = [f"Segment{e}Tension(N)" for e in range(1, nodes)]
        f.write("\t".join(["Time(s)", *names]) + "\n")
        for k in range(steps):
            f.write(_fmt([time[k], *np.full(nodes - 1, tension[k])]) + "\n")
    with open(folder / f"{root}.static.out", "w", encoding="utf-8") as f:
        f.write("CableDyn standalone static configuration (synthetic)\n")
        f.write("LineID\tNode\tArcLength\tX\tY\tZ\tTension\tCurvature\n")
        f.write("(-)\t(-)\t(m)\t(m)\t(m)\t(m)\t(N)\t(1/m)\n")
        for j in range(nodes):
            f.write(f"1\t{j + 1}\t0.0\t" + _fmt(positions[0, j]) + "\t1.0E+005\t0.0\n")
    return folder / root, deck


def time_batch(
    cable: cx.CableData, workers: list[int], runs: int, steps: int, dt: float
) -> dict[str, Any]:
    stiffness = cx.compute_stiffness(cable)
    options: dict[str, Any] = {
        "friction_model": "direction_tied",
        "tension_basis": "true",
        "chunk_rows": 4096,
    }
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        batch_runs = []
        for i in range(runs):
            root, deck = write_run(
                folder,
                f"ss{i}",
                steps,
                dt,
                100 + i,
                stiffness.ea_fixed_N,
                stiffness.ei_min_Nm2,
            )
            batch_runs.append(cx.SeaStateRun(f"ss{i}", root, deck, 1.0 / runs))
        sections = [cx.SectionRequest("mid", 5.0)]
        rows, reference = [], None
        for count in workers:
            start = perf_counter()
            result = cx.run_sea_state_batch(
                cable,
                batch_runs,
                sections,
                options=cx.SeaStateBatchOptions(workers=count, **options),
            )
            seconds = perf_counter() - start
            damage = result.sections["mid"].long_term.annual_damage_per_fibre
            if reference is None:
                reference = damage
            for layer, values in damage.items():
                np.testing.assert_allclose(
                    values, reference[layer], rtol=1e-10, atol=0.0
                )
            rows.append({"workers": count, "seconds": seconds})
            print(f"batch workers {count:3d}: {seconds:8.2f} s", flush=True)
    base = rows[0]["seconds"] * rows[0]["workers"]
    for row in rows:
        row["speed_up_vs_first"] = rows[0]["seconds"] / row["seconds"]
        row["efficiency"] = base / (row["seconds"] * row["workers"])
    return {
        "runs": runs,
        "steps": steps,
        "dt_s": dt,
        "sections": 1,
        "options": options,
        "rows": rows,
        "annual_damage_independent_of_workers": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=positive_int, default=20000)
    parser.add_argument("--dt", type=float, default=0.18)
    parser.add_argument("--repeat", type=positive_int, default=3)
    parser.add_argument(
        "--cases", nargs="+", choices=sorted(CASES), default=list(CASES)
    )
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--batch-workers", type=positive_int, nargs="*", default=[])
    parser.add_argument("--batch-runs", type=positive_int, default=16)
    parser.add_argument("--batch-steps", type=positive_int, default=3601)
    args = parser.parse_args()
    if args.steps < 500:
        parser.error("at least 500 steps are required")
    args.output.mkdir(parents=True, exist_ok=False)
    warnings.simplefilter("ignore")
    cable = cx.load_cable(cx.iea_task49_66kv_yaml_path())
    report: dict[str, Any] = {
        "utc": datetime.now(UTC).isoformat(),
        "source_sha256": source_digest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_version": cx.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "numba": numba_version(),
        "kernels_compiled": kernels_compiled(),
        "threads": {key: os.environ.get(key) for key in THREAD_VARIABLES},
        "cable": "iea_task49_66kv",
        "wires": sum(int(layer.n_wires) for layer in cable.helical_layers),
        "steps": args.steps,
        "dt_s": args.dt,
        "repeat": args.repeat,
        "history": "sea_state(steps, dt, seed=7) of this script",
    }
    rows, arrays = time_cases(cable, args.cases, args.steps, args.dt, args.repeat)
    report["cases"] = rows
    np.savez(args.output / "damage.npz", **arrays)
    if args.reference is not None:
        report["differences_from_reference"] = differences(arrays, args.reference)
        report["reference"] = str(args.reference)
    if args.batch_workers:
        report["batch"] = time_batch(
            cable, args.batch_workers, args.batch_runs, args.batch_steps, args.dt
        )
    (args.output / "throughput.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
