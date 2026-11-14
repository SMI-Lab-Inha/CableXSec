# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Repeatable public-input timings and cProfile data for release comparisons.

Run from a checkout: python validation/scripts/profile_release.py.
To measure an installed wheel, run this script by its absolute path from outside
the checkout. The report identifies the imported package, not just the script.
Timings exclude imports, cable loading and warmup. Profiling is a separate run
so instrumentation overhead does not contaminate the timing samples.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import platform
import pstats
import statistics
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

import cablexsec as cx
from cablexsec.analysis.fretting_swt import swt_life
from cablexsec.materials.sn_curves import get_swt_life
from cablexsec.mechanics.hertz_contact import solve_contact


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def workloads(samples: int) -> dict[str, Callable[[], object]]:
    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    phase = np.linspace(0.0, 6.0 * np.pi, samples)
    kx = 0.01 * np.sin(phase)
    ky = 0.004 * np.sin(1.3 * phase)
    fatigue = cx.compute_fatigue(cable, kx, ky, contact_tension_N=8e4)
    footprint = solve_contact(10.0, 100.0, 300.0, 115e9)
    depths = np.linspace(0, 3 * footprint.a_m, 1001)
    field_points = np.array(
        [
            [0.4 * footprint.a_m, -0.2 * footprint.b_m, depth * footprint.b_m]
            for depth in (0.25, 0.5, 1.0)
        ]
    )
    swt_material = get_swt_life("copper_poon2022")
    cycle = np.linspace(0, 2 * np.pi, samples)
    swt_stress = np.zeros((samples, 3, 3))
    swt_stress[:, 0, 0] = 30e6 + 100e6 * np.sin(cycle)
    swt_stress[:, 0, 1] = swt_stress[:, 1, 0] = 40e6 * np.cos(cycle)
    swt_stress[-1] = swt_stress[0]
    swt_strain = (
        1.34 * swt_stress
        - 0.34 * np.trace(swt_stress, axis1=1, axis2=2)[:, None, None] * np.eye(3)
    ) / swt_material.E_Pa
    angles = np.deg2rad(np.arange(0, 180, 5))
    swt_normals = np.column_stack(
        (np.cos(angles), np.sin(angles), np.zeros_like(angles))
    )
    ensemble = tuple(
        cx.FatigueLoadCase(
            f"condition-{i}", fatigue, 600.0, 1.0 / 100, "profile-same-model"
        )
        for i in range(100)
    )
    return {
        "swt_life": lambda: swt_life(2e4, swt_material),
        "swt_cycle_36_planes": lambda: cx.compute_swt_cycle(
            swt_stress,
            swt_strain,
            swt_normals,
            swt_material,
            material_point_id="synthetic-profile-point",
        ),
        "hertz_axis_1001_depths": lambda: cx.compute_hertz_axis_stress(
            footprint, depths, poisson_ratio=0.3
        ),
        "hertz_field_3_points": lambda: cx.compute_hertz_subsurface_stress(
            footprint,
            field_points,
            poisson_ratio=0.3,
            tangential_traction_ratio=(0.3, -0.1),
        ),
        "stiffness": lambda: cx.compute_stiffness(cable),
        "stress": lambda: cx.compute_stress(cable, 0.01, 0.004, tension_N=8e4),
        "bending": lambda: cx.compute_mkappa(
            cable, kx, tension_N=8e4, contact_model="bounded"
        ),
        "fatigue": lambda: cx.compute_fatigue(cable, kx, ky, contact_tension_N=8e4),
        "long_term_100_records": lambda: cx.compute_long_term_fatigue(
            ensemble, exposure_years=25, design_fatigue_factor=3
        ),
        # The same precomputed records exercise the alternate fixed-fibre field;
        # they have unit SCFs, so this measures aggregation, not SCF recovery.
        "long_term_adjusted_100_records": lambda: cx.compute_long_term_fatigue(
            ensemble,
            exposure_years=25,
            design_fatigue_factor=3,
            damage_basis="bulk_with_termination",
        ),
    }


def measure(function: Callable[[], object], repeat: int) -> dict[str, object]:
    function()  # warmup; also fail before reporting timings if a workload fails
    timings = []
    for _ in range(repeat):
        start = perf_counter()
        function()
        timings.append(perf_counter() - start)
    return {
        "seconds": timings,
        "median_seconds": statistics.median(timings),
        "min_seconds": min(timings),
        "max_seconds": max(timings),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=positive_int, default=601)
    parser.add_argument("--repeat", type=positive_int, default=5)
    parser.add_argument(
        "--expected-package-root",
        type=Path,
        help="fail unless the imported cablexsec directory is this exact path",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/release/profile")
    )
    args = parser.parse_args()
    if args.samples < 3:
        parser.error("--samples must be at least 3")
    package_root = Path(cx.__file__).resolve().parent
    if (
        args.expected_package_root is not None
        and package_root != args.expected_package_root.resolve()
    ):
        parser.error(
            f"imported package {package_root} differs from expected "
            f"{args.expected_package_root.resolve()}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "source_version": cx.__version__,
        "package_root": str(package_root),
        "source_sha256": source_digest(),
        "versions": {name: version(name) for name in ("cablexsec", "numpy", "scipy")},
        "threads": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "samples": args.samples,
        "repeat": args.repeat,
        "workloads": {},
    }
    results = {}
    for name, function in workloads(args.samples).items():
        results[name] = measure(function, args.repeat)
        profiler = cProfile.Profile()
        profiler.runcall(function)
        profiler.dump_stats(str(args.output / f"{name}.prof"))
        with (args.output / f"{name}.txt").open("w", encoding="utf-8") as stream:
            pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
                "cumulative"
            ).print_stats(40)
        print(f"{name}: median {results[name]['median_seconds']:.6f} s", flush=True)
    report["workloads"] = results
    (args.output / "timings.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return 0


def source_digest() -> str:
    """Identify measured package contents, including uncommitted modifications."""
    root = Path(cx.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".yaml"):
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
