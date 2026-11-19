# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Command-line entry points.

``cablexsec-batch`` computes the weighted annual armour damage from a set of
CableDyn sea-state runs::

    cablexsec-batch batch.json --output-directory results/batch1

``batch.json`` names the cable YAML, the CableDyn runs with their probabilities
and retained windows, the material sections and the options (see
:func:`cablexsec.analysis.sea_state_batch.load_batch_spec` and the
global-to-local guide of the manual). The output directory must not exist; it
receives ``manifest.json`` (input digests, checks, options and results), the
annual fixed-fibre damage of every section and the section history of every
run.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import replace
from pathlib import Path

_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def batch_main(argv: list[str] | None = None) -> int:
    """Run a sea-state batch from a JSON specification; return the exit status."""
    # Worker processes inherit these; one thread per worker keeps the batch
    # from oversubscribing the processor.
    for name in _THREAD_VARIABLES:
        os.environ.setdefault(name, "1")

    from cablexsec.analysis.sea_state_batch import (
        load_batch_spec,
        run_sea_state_batch,
    )

    parser = argparse.ArgumentParser(
        prog="cablexsec-batch",
        description="Weighted annual armour damage from a set of CableDyn "
        "sea-state runs.",
    )
    parser.add_argument("spec", type=Path, help="JSON batch specification")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        help="worker processes (at least 1, one thread each; at most one per run "
        "is started); overrides the specification",
    )
    args = parser.parse_args(argv)
    cable, runs, sections, options, cable_path = load_batch_spec(args.spec)
    if args.workers is not None:
        options = replace(options, workers=args.workers)
    result = run_sea_state_batch(
        cable,
        runs,
        sections,
        options=options,
        output_directory=args.output_directory,
        cable_source=cable_path,
    )
    print(f"{len(runs)} runs, {len(sections)} sections -> {args.output_directory}")
    print(f"{'section':<16}{'arc [m]':>10}{'annual D':>13}{'life [y]':>12}  layer")
    for name, section in result.sections.items():
        life = "inf" if math.isinf(section.life_years) else f"{section.life_years:.4g}"
        print(
            f"{name:<16}{section.arc_length_m:>10.3f}"
            f"{section.long_term.annual_damage:>13.4e}{life:>12}  "
            f"{section.long_term.critical_layer}"
        )
    print(f"inputs sha256 {result.manifest['inputs_sha256']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(batch_main())
