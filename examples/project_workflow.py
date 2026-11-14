# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Run and reopen a self-contained engineering project from the Python API."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cablexsec import (
    AnalysisRequest,
    CableModel,
    FatigueSettings,
    LoadCaseSeries,
    Project,
    inter_array_three_core_yaml_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = Project("Three-core inter-array assessment")
    project.set_model(
        "design", CableModel.from_yaml(inter_array_three_core_yaml_path())
    )
    time = np.linspace(0, 60, 601)
    project.set_load_case(
        "operating",
        LoadCaseSeries(
            time,
            0.006 * np.sin(time),
            0.002 * np.cos(time),
            np.full_like(time, 8e4),
            tension_kind="true",
            source="synthetic example; SI units in the cable section frame",
        ),
    )
    project.set_analysis(
        AnalysisRequest(
            name="bulk fatigue",
            model="design",
            load_case="operating",
            settings=FatigueSettings(n_fibres=16),
        )
    )
    record = project.run("bulk fatigue")
    project.save(args.output)
    restored = Project.load(args.output)
    assert restored.records[0].fingerprint == record.fingerprint
    assert not restored.is_stale(restored.records[0])
    print(f"Saved and reopened: {args.output}")
    print(f"Run: {record.run_id}; input fingerprint: {record.fingerprint}")


if __name__ == "__main__":
    main()
