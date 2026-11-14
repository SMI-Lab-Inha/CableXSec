# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tutorials 1 and 2 run as written and print the output their pages show.

Each script runs in a subprocess with a temporary figure folder. Its standard
output must equal the concatenated ``text`` blocks that follow the
``<!-- tutorial-output -->`` markers of the page, and every code marker the
page includes must exist in the script.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parents[1] / "doc" / "tutorials"

TUTORIALS = [
    (
        "01_first_cable.md",
        "tutorial_01_first_cable.py",
        ("tutorial_01_first_cable.png",),
    ),
    (
        "02_bending_hysteresis.md",
        "tutorial_02_bending_hysteresis.py",
        ("tutorial_02_measured_loop.png", "tutorial_02_sensitivity.png"),
    ),
]

_OUTPUT = re.compile(r"<!-- tutorial-output -->\s*```text\n(.*?)```", re.DOTALL)
_MARKER = re.compile(r':(?:start-after|end-before|start-at): "([^"]+)"')


def _run(script: Path, folder: Path) -> str:
    env = dict(os.environ)
    env.update(
        MPLBACKEND="Agg",
        OMP_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
        PYTHONIOENCODING="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(script), str(folder)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


@pytest.mark.parametrize(("page", "script", "figures"), TUTORIALS)
def test_tutorial_output_matches_page(
    page: str, script: str, figures: tuple[str, ...], tmp_path: Path
) -> None:
    text = (DOC / page).read_text(encoding="utf-8")
    source = (DOC / "scripts" / script).read_text(encoding="utf-8")
    markers = _MARKER.findall(text)
    assert markers, "the page includes no part of its script"
    for marker in markers:
        assert marker in source, f"marker {marker!r} is not in {script}"
    shown = "".join(_OUTPUT.findall(text))
    assert shown, "the page shows no output block"

    printed = _run(DOC / "scripts" / script, tmp_path)

    assert printed.splitlines() == shown.splitlines()
    for name in figures:
        figure = tmp_path / name
        assert figure.is_file(), name
        assert figure.stat().st_size < 200_000, name
        assert (DOC / "figures" / name).is_file(), f"{name} is not committed"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(figures)
