#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Smoke-test an installed CableXSec distribution outside the source tree.

Run with the interpreter of a clean environment into which only the wheel was
installed::

    python release/smoke_test_wheel.py --version 0.1.0

The check imports the installed package (and fails if it resolves into this
checkout), loads a bundled cable, computes its stiffness and a bending loop,
and runs the ``cablexsec-batch`` command's help.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True, help="expected version")
    args = parser.parse_args()

    import numpy as np

    import cablexsec as cx

    location = Path(cx.__file__).resolve()
    if location.is_relative_to(ROOT / "src" / "cablexsec"):
        print(f"smoke test: FAIL (imported the checkout: {location})", file=sys.stderr)
        return 1
    if cx.__version__ != args.version:
        print(
            f"smoke test: FAIL (version {cx.__version__}, expected {args.version})",
            file=sys.stderr,
        )
        return 1
    cable = cx.load_cable(cx.menard2023_yaml_path())
    stiffness = cx.compute_stiffness(cable)
    curvature = 0.05 * np.sin(np.linspace(0.0, 4.0 * np.pi, 401))
    loop = cx.compute_mkappa(cable, curvature, contact_model="bounded")
    if not (np.isfinite(stiffness.ea_fixed_N) and np.all(np.isfinite(loop.moment_Nm))):
        print("smoke test: FAIL (non-finite results)", file=sys.stderr)
        return 1
    command = shutil.which("cablexsec-batch")
    if command is None:
        print("smoke test: FAIL (cablexsec-batch not on PATH)", file=sys.stderr)
        return 1
    subprocess.run([command, "--help"], check=True, capture_output=True)
    print(
        f"smoke test: PASS (CableXSec {cx.__version__} at {location.parent}; "
        f"EA {stiffness.ea_fixed_N / 1e6:.1f} MN, EI_min {loop.EI_min_Nm2:.0f} N m^2)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
