# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Matched in-process fatigue timings with unreduced and reduced cycle counting.

Run as ``python validation/scripts/benchmark_fatigue_counter.py``. Each pair uses the same mechanics and parameters; only turning-point
reduction changes. All fibre damage and legacy damage fields must match exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import rainflow
from numpy.typing import NDArray
from profile_release import positive_int, source_digest

import cablexsec as cx
from cablexsec.analysis import fatigue


def unreduced_cycles(
    values: NDArray[np.float64],
) -> Iterator[tuple[float, float, float, int, int]]:
    """Previous package wrapper around the installed upstream counter."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("stress history must be one-dimensional and finite")
    if values.size < 2:
        return
    if values.size == 2:
        span = float(abs(values[1] - values[0]))
        if span > 0:
            yield span, float(0.5 * values[0] + 0.5 * values[1]), 0.5, 0, 1
        return
    for cycle in rainflow.extract_cycles(values.tolist()):
        if cycle[0] > 0:
            yield cycle


def compare(first: cx.FatigueResult, second: cx.FatigueResult) -> None:
    for field in (
        "D_bulk_per_wire",
        "D_fretting_per_wire",
        "D_term_per_wire",
        "D_total_per_wire",
        "D_bulk_per_fibre",
        "D_bulk_with_termination_per_fibre",
        "fibre_critical_deg_per_wire",
        "termination_fibre_critical_deg_per_wire",
    ):
        expected, actual = getattr(first, field), getattr(second, field)
        if expected.keys() != actual.keys():
            raise AssertionError(f"different layer sets for {field}")
        for name in expected:
            np.testing.assert_array_equal(
                actual[name], expected[name], err_msg=f"{field}/{name}"
            )
    np.testing.assert_array_equal(first.fibre_angles_deg, second.fibre_angles_deg)
    for field in (
        "phi_critical_deg",
        "critical_layer",
        "above_core",
        "D_fretting_fraction",
        "governing",
        "fibre_critical_deg",
    ):
        if getattr(first, field) != getattr(second, field):
            raise AssertionError(f"different {field}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=positive_int, nargs="+", default=[601, 6001])
    parser.add_argument("--repeat", type=positive_int, default=5)
    args = parser.parse_args()
    if min(args.samples) < 3:
        parser.error("sample counts must be at least three")
    args.output.mkdir(parents=True, exist_ok=False)
    original = fatigue.rainflow_cycles
    counters = {"unreduced": unreduced_cycles, "reduced": original}
    report = {
        "utc": datetime.now(UTC).isoformat(),
        "source_sha256": source_digest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_version": cx.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "rainflow": rainflow.__version__,
        "threads": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "repeat": args.repeat,
        "n_fibres": 32,
        "tension_N": 80000,
        "contact_tension_N": 80000,
        "comparison": "Same current mechanics; previous unreduced counter versus NumPy turning-point reduction. Alternating paired order; import/load/warmup excluded.",
        "history": "phase=linspace(0,6*pi,samples); kx=0.01*sin(phase); ky=0.004*sin(1.3*phase); bounded contact; all SI units",
        "complete": False,
        "cases": [],
    }
    paths = {
        "single_core_inter_array": cx.inter_array_single_core_yaml_path(),
        "three_core_inter_array": cx.inter_array_three_core_yaml_path(),
        "three_core_export": cx.export_three_core_yaml_path(),
    }
    try:
        for name, path in paths.items():
            cable = cx.load_cable(path, geometry_override=None)
            for samples in args.samples:
                phase = np.linspace(0, 6 * np.pi, samples)
                kx = 0.01 * np.sin(phase)
                ky = 0.004 * np.sin(1.3 * phase)
                tension = np.full(samples, 80000.0)
                results = {}
                timings = {label: [] for label in counters}
                for iteration in range(args.repeat + 1):
                    order = (
                        list(counters)
                        if iteration % 2 == 0
                        else list(reversed(counters))
                    )
                    for label in order:
                        fatigue.rainflow_cycles = counters[label]
                        start = perf_counter()
                        results[label] = cx.compute_fatigue(
                            cable,
                            kx,
                            ky,
                            tension_N=tension,
                            contact_tension_N=80000,
                            n_fibres=32,
                        )
                        elapsed = perf_counter() - start
                        if iteration > 0:
                            timings[label].append(elapsed)
                    compare(results["unreduced"], results["reduced"])
                medians = {
                    label: statistics.median(values)
                    for label, values in timings.items()
                }
                row = {
                    "construction": name,
                    "samples": samples,
                    "seconds": timings,
                    "median_seconds": medians,
                    "speedup": medians["unreduced"] / medians["reduced"],
                    "all_damage_fields_exactly_equal": True,
                }
                report["cases"].append(row)
                (args.output / "comparison.json").write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8"
                )
                print(
                    f"{name}, {samples} samples: {row['speedup']:.3f}x; exact damage equality",
                    flush=True,
                )
    finally:
        fatigue.rainflow_cycles = original
    report["complete"] = True
    (args.output / "comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
