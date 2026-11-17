#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Line and branch coverage of CableXSec, with the compiled kernels traced.

Functions compiled by numba run outside the coverage tracer. The gate therefore runs
the whole suite once as installed, then runs the kernel tests a second time with
``NUMBA_DISABLE_JIT=1`` so that the kernel bodies execute as Python, appends both to
one coverage record, and fails below the threshold.

Usage: ``python release/check_coverage.py [--fail-under 95]``
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KERNEL_TESTS = (
    "tests/test_kernel_paths.py",
    "tests/test_fast_rainflow.py",
    "tests/test_fast_slip.py",
    "tests/test_compiled_counter.py",
    "tests/test_progressive_friction.py",
    "tests/test_fretting_regime.py",
    "tests/test_strand_slip.py",
    "tests/test_along_wire_test_inputs.py",
    "tests/test_along_wire_chain.py",
    "tests/test_friction_kernels.py",
    "tests/test_friction_law.py",
    "tests/test_friction_pressure.py",
    "tests/test_distributed_slide.py",
)


def run(command: list[str], extra_env: dict[str, str] | None = None) -> int:
    env = os.environ.copy()
    env.update(MPLBACKEND="Agg")
    env.update(extra_env or {})
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fail-under", type=float, default=95.0)
    args = parser.parse_args()
    python = sys.executable
    pytest = [python, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    status = run(
        [*pytest, "--cov", "--cov-branch", "--cov-report=", "--cov-fail-under=0"]
    )
    if status != 0:
        return status
    status = run(
        [
            *pytest,
            "--cov",
            "--cov-branch",
            "--cov-append",
            "--cov-report=",
            "--cov-fail-under=0",
            *KERNEL_TESTS,
        ],
        {"NUMBA_DISABLE_JIT": "1"},
    )
    if status != 0:
        return status
    run([python, "-m", "coverage", "xml", "-q"])
    return run([python, "-m", "coverage", "report", f"--fail-under={args.fail_under}"])


if __name__ == "__main__":
    raise SystemExit(main())
