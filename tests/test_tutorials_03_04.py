# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tutorials 3 and 4 run as written and print the output their pages show.

Each script runs in a subprocess with a temporary figure folder. Its standard
output must equal the concatenated ``text`` blocks that follow the
``<!-- tutorial-output -->`` markers of the page, every code marker the page
includes must exist in the script, and the script must write its figures and
nothing else.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import cablexsec as cx

REPOSITORY = Path(__file__).resolve().parents[1]
DOC = REPOSITORY / "doc" / "tutorials"

TUTORIALS = [
    (
        "03_global_to_fatigue.md",
        "tutorial_03_global_to_fatigue.py",
        (
            "tutorial_03_damage_circumference.png",
            "tutorial_03_governing_fibre_stress.png",
        ),
    ),
    (
        "04_contact_and_fretting.md",
        "tutorial_04_contact_and_fretting.py",
        ("tutorial_04_slip_amplitude.png", "tutorial_04_swt_depth.png"),
    ),
]

_OUTPUT = re.compile(r"<!-- tutorial-output -->\s*```text\n(.*?)```", re.DOTALL)
_MARKER = re.compile(r':(?:start-after|end-before|start-at): "([^"]+)"')


_NEGATIVE_ZERO = re.compile(r"-(0(?:\.0+)?)(?![0-9])")


def _lines(text: str) -> list[str]:
    """Output lines with a printed negative zero read as zero.

    A value that rounds to zero carries no sign; platforms differ in the sign
    of the round-off behind it.
    """
    return [_NEGATIVE_ZERO.sub(r"\1", line) for line in text.splitlines()]


def _run(script: Path, folder: Path) -> str:
    env = dict(os.environ)
    env.update(
        MPLBACKEND="Agg",
        OMP_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
        PYTHONIOENCODING="utf-8",
        PYTHONPATH=os.pathsep.join(
            item for item in (str(REPOSITORY), env.get("PYTHONPATH", "")) if item
        ),
    )
    completed = subprocess.run(
        [sys.executable, str(script), str(folder)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=folder,
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

    assert _lines(printed) == _lines(shown)
    for name in figures:
        figure = tmp_path / name
        assert figure.is_file(), name
        assert figure.stat().st_size < 200_000, name
        assert (DOC / "figures" / name).is_file(), f"{name} is not committed"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(figures)


def test_file_formats_minimal_definition_is_the_bundled_template(
    tmp_path: Path,
) -> None:
    """The minimal YAML of doc/file_formats.md loads and matches its source."""
    text = (REPOSITORY / "doc" / "file_formats.md").read_text(encoding="utf-8")
    section = text.split("### A minimal complete definition", 1)[1]
    block = re.search(r"```yaml\n(.*?)```", section, re.DOTALL)
    assert block is not None
    path = tmp_path / "minimal.yaml"
    path.write_text(block.group(1), encoding="utf-8")

    shown = cx.load_cable(path)
    bundled = cx.load_cable(cx.inter_array_single_core_yaml_path())

    assert not shown.missing_fields()
    assert [layer.name for layer in shown.layers] == [
        layer.name for layer in bundled.layers
    ]
    for a, b in zip(shown.layers, bundled.layers, strict=True):
        assert a.model_dump(exclude={"notes", "source"}) == b.model_dump(
            exclude={"notes", "source"}
        )
    a, b = cx.compute_stiffness(shown), cx.compute_stiffness(bundled)
    for name in ("ea_fixed_N", "gj_Nm2", "ei_min_Nm2", "ei_max_Nm2"):
        assert getattr(a, name) == getattr(b, name), name
