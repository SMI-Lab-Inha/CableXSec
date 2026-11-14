# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Short file names and writing a set of files all or none.

A long cable name (``IEA Task 49 / Janocha 2024 66 kV three-core (public
template)``) and the names of its load cases made file names of more than a
hundred characters, which in a deep folder pass the 260-character path limit
of Windows. :func:`short_name` keeps the words that identify the cable within
a length; :func:`write_all` checks every path first and writes a set of files
under temporary names, then renames them, so a failure leaves the folder as it
was.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
import unicodedata
from collections.abc import Callable, Sequence
from pathlib import Path

from cablexsec.exceptions import CableXSecError

__all__ = [
    "MAX_PATH",
    "FileSetError",
    "existing",
    "short_file_name",
    "short_name",
    "unique_names",
    "write_all",
]

MAX_PATH = 259
"""Longest full path, in characters, that every program on Windows opens
(260 with the terminating null)."""

_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "the",
        "of",
        "with",
        "for",
        "cable",
        "public",
        "template",
        "representative",
        "reference",
        "example",
    }
)
"""Words that do not tell one cable from another."""


def _words(text: str) -> list[str]:
    """The words of ``text``: ASCII, no parenthesised remarks, units joined."""
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    plain = re.sub(r"\([^)]*\)", " ", plain)
    # A number and its unit are one word: 66 kV reads 66kV.
    plain = re.sub(r"(\d)\s+(kV|V|mm2|mm|m|kN|MPa)\b", r"\1\2", plain)
    words = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", plain)
    return [w for w in words if w.lower() not in _STOP_WORDS]


def short_name(
    text: str, limit: int = 24, *, separator: str = "-", lower: bool = True
) -> str:
    """``text`` as a name of at most ``limit`` characters that still reads.

    Remarks in parentheses and words such as *cable* or *public* are left out;
    while the name is too long, the last word without a digit goes (numbers
    tell cables apart: a voltage, a year), and at the end the name is cut at
    the limit. ``lower`` gives a file-name stem; otherwise the words keep their
    capitals (an object name).
    """
    words = _words(text)
    if lower:
        words = [w.lower() for w in words]
    if not words:
        return "cable" if lower else "Cable"

    def joined(items: Sequence[str]) -> str:
        return separator.join(items)

    while len(joined(words)) > limit:
        # The first word stays: it names the cable (IEA, Menard, Three-core).
        plain = [
            i for i, w in enumerate(words) if i and not any(c.isdigit() for c in w)
        ]
        if not plain:
            break
        words.pop(plain[-1])
    name = joined(words)
    if len(name) > limit:
        name = name[:limit].rstrip(separator + "-")
    return name


def short_file_name(name: str, limit: int = 64) -> str:
    """A file name of at most ``limit`` characters, its extension kept (names
    saved with results before names were shortened)."""
    if len(name) <= limit:
        return name
    stem, dot, suffix = name.rpartition(".")
    if not dot or not stem:
        stem, suffix = name, ""
    extension = f".{suffix}" if suffix else ""
    return short_name(stem, limit - len(extension)) + extension


def unique_names(names: Sequence[str], separator: str = " ") -> list[str]:
    """``names`` with a number added to each one that repeats an earlier one
    (before the extension of a file name: ``a.csv``, ``a-2.csv``)."""
    seen: set[str] = set()
    found = []
    for name in names:
        stem, dot, suffix = name.rpartition(".") if separator == "-" else ("", "", "")
        if not dot or not stem:
            stem, suffix = name, ""
        candidate, number = name, 2
        while candidate in seen:
            candidate = f"{stem}{separator}{number}" + (f".{suffix}" if suffix else "")
            number += 1
        seen.add(candidate)
        found.append(candidate)
    return found


class FileSetError(CableXSecError):
    """A set of files could not be written; the message says why in words."""


def _length(path: Path) -> int:
    return len(str(path.absolute()))


def existing(folder: Path, names: Sequence[str]) -> list[str]:
    """The ``names`` that already exist in ``folder``."""
    return [name for name in names if (folder / name).exists()]


def _part(path: Path) -> Path:
    return path.with_name(f"~{path.name}.part")


def _backup(path: Path) -> Path:
    return path.with_name(f"~{path.name}.old")


def write_all(
    folder: Path,
    writers: Sequence[tuple[str, Callable[[Path], object]]],
    *,
    limit: int | None = None,
) -> list[Path]:
    """Write every file of ``writers`` into ``folder``, or none of them.

    Each writer is a file name and a function that writes the file at the path
    it is given. Every path, temporary ones included, is checked against
    ``limit`` (default :data:`MAX_PATH`) before anything is written; then the
    files are written under temporary names and renamed into place, replacing
    files of the same name.
    On any failure the temporary files are removed, replaced files are put
    back and :class:`FileSetError` says what went wrong.
    """
    limit = MAX_PATH if limit is None else limit
    names = [name for name, _ in writers]
    if len(set(names)) != len(names):
        raise FileSetError("two of the files have the same name")
    if not folder.is_dir():
        raise FileSetError(f"the folder {folder} does not exist")
    longest = max((_length(_backup(folder / name)) for name in names), default=0)
    if longest > limit:
        name = max(names, key=lambda n: _length(_backup(folder / n)))
        raise FileSetError(
            f"the full path of {name} in this folder would have {longest} "
            f"characters, more than the {limit} that Windows programs open; "
            "choose a folder with a shorter path"
        )
    written: list[Path] = []
    try:
        for name, write in writers:
            part = _part(folder / name)
            written.append(part)
            write(part)
            if not part.exists():
                raise FileSetError(f"{name} was not written")
    except (OSError, CableXSecError) as error:
        _remove(written)
        raise FileSetError(f"could not write {name}: {_reason(error)}") from error
    moved: list[tuple[Path, Path]] = []
    placed: list[Path] = []
    try:
        for name in names:
            target = folder / name
            if target.exists():
                os.replace(target, _backup(target))
                moved.append((_backup(target), target))
            os.replace(_part(target), target)
            placed.append(target)
    except OSError as error:
        _remove(placed)
        for backup, target in reversed(moved):
            os.replace(backup, target)
        _remove(written)
        raise FileSetError(f"could not save {name}: {_reason(error)}") from error
    _remove([backup for backup, _ in moved])
    return [folder / name for name in names]


def _remove(paths: Sequence[Path]) -> None:
    for path in paths:
        with contextlib.suppress(OSError):  # best effort
            path.unlink(missing_ok=True)


def _reason(error: BaseException) -> str:
    """The cause of ``error`` in words."""
    if isinstance(error, PermissionError):
        return "permission denied (the file may be open in another program)"
    if isinstance(error, FileNotFoundError):
        return (
            "the path was not found (it may be longer than Windows allows)"
            if sys.platform == "win32"
            else "the path was not found"
        )
    if isinstance(error, OSError) and error.strerror:
        return error.strerror.lower()
    return str(error)
