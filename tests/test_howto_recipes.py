# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""The how-to recipes run as written and print the output the page shows.

Each script of ``doc/howto/scripts`` runs in a subprocess with a temporary
figure folder as its working directory and argument. Its standard output must
equal the ``text`` block that follows its ``<!-- howto-output: <script> -->``
marker in ``doc/howto.md``; the page must include the script from its
``# [recipe]`` marker, and the run must leave no file behind except its figures.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parents[1] / "doc"
SCRIPTS = DOC / "howto" / "scripts"
PAGE = DOC / "howto.md"

FIGURES = {"howto_04_section_stress.py": ("howto_04_section_stress.png",)}

_OUTPUT = re.compile(r"<!-- howto-output: (\S+) -->\s*```text\n(.*?)```", re.DOTALL)


_NEGATIVE_ZERO = re.compile(r"-(0(?:\.0+)?)(?![0-9])")


def _lines(text: str) -> list[str]:
    """Output lines with a printed negative zero read as zero.

    A value that rounds to zero carries no sign; platforms differ in the sign
    of the round-off behind it.
    """
    return [_NEGATIVE_ZERO.sub(r"\1", line) for line in text.splitlines()]


def _shown() -> dict[str, str]:
    return dict(_OUTPUT.findall(PAGE.read_text(encoding="utf-8")))


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
        cwd=folder,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_every_script_is_on_the_page() -> None:
    scripts = sorted(p.name for p in SCRIPTS.glob("howto_*.py"))
    assert len(scripts) == 10
    assert sorted(_shown()) == scripts
    text = PAGE.read_text(encoding="utf-8")
    for name in scripts:
        assert f"howto/scripts/{name}" in text, name
        assert "# [recipe]" in (SCRIPTS / name).read_text(encoding="utf-8"), name


@pytest.mark.parametrize("script", sorted(p.name for p in SCRIPTS.glob("howto_*.py")))
def test_recipe_output_matches_page(script: str, tmp_path: Path) -> None:
    shown = _shown()[script]
    printed = _run(SCRIPTS / script, tmp_path)
    assert _lines(printed) == _lines(shown)
    figures = FIGURES.get(script, ())
    for name in figures:
        figure = tmp_path / name
        assert figure.is_file(), name
        assert figure.stat().st_size < 200_000, name
        assert (DOC / "howto" / "figures" / name).is_file(), f"{name} is missing"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(figures)
