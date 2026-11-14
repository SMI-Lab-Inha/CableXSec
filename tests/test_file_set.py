# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Short file names, sets of files written all or none, and number display."""

from __future__ import annotations

from pathlib import Path

import pytest

from cablexsec.io.file_set import (
    FileSetError,
    existing,
    short_file_name,
    short_name,
    unique_names,
    write_all,
)


def test_short_names_keep_the_words_that_identify_the_cable() -> None:
    long = "IEA Task 49 / Janocha 2024 66 kV three-core (public template)"
    assert short_name(long) == "iea-task-49-2024-66kv"
    assert short_name("Single-core inter-array 33 kV (public representative)") == (
        "single-core-33kv"
    )
    assert short_name("Three-core export 220 kV (public representative)") == (
        "three-core-export-220kv"
    )
    assert short_name("Ménard 2023 reference cable") == "menard-2023"
    assert short_name("") == "cable"
    for name in (long, "Saevik 2011 8-inch FBG flexible riser", "x" * 80):
        assert len(short_name(name)) <= 24
    assert short_name(long, 24, separator=" ", lower=False) == "IEA Task 49 2024 66kV"
    assert short_file_name("a" * 100 + ".json").endswith(".json")
    assert len(short_file_name("a" * 100 + ".json")) <= 64
    assert short_file_name("short.csv") == "short.csv"
    assert unique_names(["a.csv", "a.csv", "b"], separator="-") == [
        "a.csv",
        "a-2.csv",
        "b",
    ]
    assert unique_names(["Line", "Line"]) == ["Line", "Line 2"]


def test_write_all_writes_every_file_or_none(tmp_path: Path) -> None:
    written = write_all(
        tmp_path,
        [
            ("a.txt", lambda p: p.write_text("A", encoding="utf-8")),
            ("b.txt", lambda p: p.write_text("B", encoding="utf-8")),
        ],
    )
    assert [p.name for p in written] == ["a.txt", "b.txt"]
    assert existing(tmp_path, ["a.txt", "c.txt"]) == ["a.txt"]

    def fail(path: Path) -> None:
        raise PermissionError(13, "Permission denied", str(path))

    # A failure part-way leaves the folder as it was: no new file, no
    # temporary file, the old files unchanged.
    with pytest.raises(FileSetError, match="permission denied"):
        write_all(
            tmp_path,
            [
                ("a.txt", lambda p: p.write_text("new", encoding="utf-8")),
                ("c.txt", lambda p: p.write_text("C", encoding="utf-8")),
                ("d.txt", fail),
            ],
        )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.txt", "b.txt"]
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "A"
    # Replacing keeps nothing of the old files and leaves no backups.
    write_all(tmp_path, [("a.txt", lambda p: p.write_text("new", encoding="utf-8"))])
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "new"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.txt", "b.txt"]


def test_write_all_checks_the_path_length_first(tmp_path: Path) -> None:
    deep = tmp_path / ("d" * 40)
    deep.mkdir()
    name = "n" * 60 + ".json"
    limit = len(str(deep.absolute())) + 40
    with pytest.raises(FileSetError, match="characters, more than the"):
        write_all(
            deep, [("ok.txt", lambda p: p.write_text("x")), (name, print)], limit=limit
        )
    assert list(deep.iterdir()) == []
    with pytest.raises(FileSetError, match="does not exist"):
        write_all(tmp_path / "missing", [("a", print)])
    with pytest.raises(FileSetError, match="same name"):
        write_all(tmp_path, [("a", print), ("a", print)])
