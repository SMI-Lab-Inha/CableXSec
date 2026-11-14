# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Sets of files written all or none: rollback of renames and failure reasons."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from cablexsec.exceptions import CableInputError
from cablexsec.io import file_set
from cablexsec.io.file_set import FileSetError, short_file_name, write_all


def _text(content: str):
    return lambda p: p.write_text(content, encoding="utf-8")


def test_long_name_without_extension_is_shortened_whole() -> None:
    name = "x" * 30 + " 66 kV " + "y" * 40
    short = short_file_name(name, limit=24)
    assert len(short) <= 24
    assert "." not in short


def test_writer_that_writes_nothing_leaves_the_folder_unchanged(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    with pytest.raises(FileSetError, match="b.txt was not written"):
        write_all(tmp_path, [("a.txt", _text("new")), ("b.txt", lambda p: None)])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.txt"]
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "A"


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (FileNotFoundError(2, "No such file or directory"), "path was not found"),
        (IsADirectoryError(21, "Is a directory"), "is a directory"),
        (CableInputError("bad table"), "bad table"),
    ],
)
def test_failure_reasons_are_given_in_words(tmp_path: Path, error, reason) -> None:
    def fail(path: Path) -> None:
        raise error

    with pytest.raises(FileSetError, match=reason):
        write_all(tmp_path, [("a.txt", _text("A")), ("b.txt", fail)])
    assert list(tmp_path.iterdir()) == []
    if isinstance(error, FileNotFoundError) and sys.platform == "win32":
        with pytest.raises(FileSetError, match="longer than Windows allows"):
            write_all(tmp_path, [("b.txt", fail)])


def test_failed_rename_restores_the_replaced_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, content in (("a.txt", "old A"), ("b.txt", "old B")):
        (tmp_path / name).write_text(content, encoding="utf-8")
    real_replace = os.replace
    calls = {"n": 0}

    def replace(src: Path, dst: Path) -> None:
        # Fail when the second new file is moved into place.
        if Path(src).name == "~b.txt.part":
            calls["n"] += 1
            raise PermissionError(13, "Permission denied", str(dst))
        real_replace(src, dst)

    monkeypatch.setattr(file_set.os, "replace", replace)
    with pytest.raises(FileSetError, match="could not save b.txt: permission denied"):
        write_all(tmp_path, [("a.txt", _text("new A")), ("b.txt", _text("new B"))])
    monkeypatch.undo()
    assert calls["n"] == 1
    # All or none: both old files are back, no temporary or backup file is left.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.txt", "b.txt"]
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "old A"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "old B"
