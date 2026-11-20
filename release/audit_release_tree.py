#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Audit the tracked files or a Git archive intended for a CableXSec release."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]

# Directories of the development repository that never belong in a release.
FORBIDDEN_ROOTS = {
    "artifacts",
    "data",
    "external",
    "packaging",
    "paper",
    "papers",
    "private",
    "publication",
    "scratchpad",
}
# The desktop application is not part of the public package.
FORBIDDEN_PREFIXES = ("src/cablexsec/app/", "src/cablexsec/gui/")
FORBIDDEN_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".docx",
    ".exe",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".pdf",
    ".sim",
    ".synctex.gz",
    ".tex",
    ".xlsx",
}
REQUIRED_PATHS = {
    ".github/workflows/python.yml",
    "ARCHITECTURE.md",
    "CHANGELOG.md",
    "CITATION.cff",
    "LICENSE",
    "NOTICE",
    "README.md",
    "RELEASE.md",
    "SECURITY.md",
    "VALIDATION.md",
    "src/cablexsec/__init__.py",
    "doc/index.rst",
    "pyproject.toml",
    "validation/README.md",
    "validation/RELEASE_0_1_0.md",
    "validation/RELEASE_0_2_0.md",
}
# The only e-mail address a release may carry is the maintainer's.
MAINTAINER_EMAIL = b"jaehoon.seo@inha.ac.kr"
EMAIL = re.compile(
    rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"
)
SENSITIVE_PATTERNS = (
    ("Windows user path", re.compile(rb"[A-Za-z]:[\\/]+Users[\\/]+", re.IGNORECASE)),
    (
        "local repository path",
        re.compile(rb"[A-Za-z]:[\\/]+repos[\\/]+", re.IGNORECASE),
    ),
    ("GitHub personal token", re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}")),
    ("AWS access key", re.compile(rb"AKIA[0-9A-Z]{16}")),
    ("private key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    (
        "co-author trailer",
        re.compile(rb"^[ \t]*Co-" rb"Authored-By:", re.IGNORECASE | re.MULTILINE),
    ),
    ("private manuscript identifier", re.compile(rb"\bOE-D-[0-9-]+\b", re.IGNORECASE)),
    ("withheld cable name", re.compile(rb"kr" rb"70", re.IGNORECASE)),
    ("withheld-material folder", re.compile(rb"_confi" rb"dential", re.IGNORECASE)),
)
# File names that point at withheld or third-party material.
SENSITIVE_NAMES = re.compile(r"(?i)(kr" r"70|confi" r"dential|manufacturer|\bnda\b)")


class AuditError(RuntimeError):
    """The proposed release tree violates a packaging requirement."""


def run_git(*arguments: str, text: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", os.fspath(ROOT), *arguments],
        check=False,
        capture_output=True,
        text=text,
    )
    if result.returncode != 0:
        error = result.stderr if text else result.stderr.decode(errors="replace")
        raise AuditError(f"git {' '.join(arguments)} failed: {error.strip()}")
    return result.stdout


def tracked_files(tree: Path) -> list[tuple[str, Path]]:
    # --cached lists staged files too, so the audit also runs before the first commit.
    output = bytes(run_git("ls-files", "--cached", "-z", text=False))
    result: list[tuple[str, Path]] = []
    for raw in output.split(b"\0"):
        if not raw:
            continue
        relative = raw.decode("utf-8")
        path = tree / Path(*PurePosixPath(relative).parts)
        if path.is_file():
            result.append((relative, path))
    return sorted(result)


def archive_files(reference: str, temporary: Path) -> list[tuple[str, Path]]:
    archive = temporary / "source.zip"
    run_git("archive", "--format=zip", f"--output={archive}", reference)
    tree = temporary / "source"
    tree.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            parts = PurePosixPath(member.filename)
            if parts.is_absolute() or ".." in parts.parts:
                raise AuditError(f"unsafe archive path: {member.filename}")
        bundle.extractall(tree)
    return sorted(
        (path.relative_to(tree).as_posix(), path)
        for path in tree.rglob("*")
        if path.is_file()
    )


def audit(files: list[tuple[str, Path]]) -> tuple[int, int, str]:
    paths = {relative for relative, _ in files}
    problems: list[str] = []
    digest = hashlib.sha256()
    total_bytes = 0

    for relative, path in files:
        parts = PurePosixPath(relative).parts
        lower = relative.lower()
        if parts and parts[0].lower() in FORBIDDEN_ROOTS:
            problems.append(f"excluded root retained: {relative}")
        if lower.startswith(FORBIDDEN_PREFIXES):
            problems.append(f"desktop-application file retained: {relative}")
        if any(lower.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
            problems.append(f"generated or non-source file retained: {relative}")
        if SENSITIVE_NAMES.search(relative):
            problems.append(f"file name points at withheld material: {relative}")

        data = path.read_bytes()
        total_bytes += len(data)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(data).digest())
        if b"\0" not in data and len(data) <= 10_000_000:
            for label, pattern in SENSITIVE_PATTERNS:
                if pattern.search(data) and relative != "release/audit_release_tree.py":
                    problems.append(f"{label} retained in {relative}")
            for address in EMAIL.findall(data):
                lowered = address.lower()
                # Reserved example domains (RFC 2606) are placeholders, not addresses.
                placeholder = lowered.endswith(
                    (b".invalid", b".example", b"@example.com", b"@example.org")
                )
                if lowered != MAINTAINER_EMAIL and not placeholder:
                    problems.append(
                        f"e-mail address other than the maintainer's in {relative}"
                    )

    problems.extend(
        f"required release file missing: {relative}"
        for relative in sorted(REQUIRED_PATHS - paths)
    )
    if problems:
        raise AuditError("release-tree audit failed:\n  - " + "\n  - ".join(problems))
    return len(files), total_bytes, digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    group = result.add_mutually_exclusive_group()
    group.add_argument(
        "--tree",
        type=Path,
        help="audit the current tracked worktree (defaults to the repository root)",
    )
    group.add_argument(
        "--archive-ref",
        metavar="REF",
        help="audit a temporary Git archive created from REF",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    if args.archive_ref:
        with tempfile.TemporaryDirectory(
            prefix="cablexsec-release-audit-"
        ) as directory:
            files = archive_files(args.archive_ref, Path(directory))
            count, total_bytes, digest = audit(files)
        source = args.archive_ref
    else:
        tree = (args.tree or ROOT).resolve()
        if tree != ROOT.resolve():
            raise AuditError(f"--tree must identify this repository root: {ROOT}")
        files = tracked_files(tree)
        count, total_bytes, digest = audit(files)
        source = "tracked worktree"

    print(
        f"release-tree audit: PASS ({source}, {count} files, "
        f"{total_bytes} bytes, SHA-256 {digest})"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as error:
        print(f"release-tree audit: FAIL\n- {error}", file=sys.stderr)
        raise SystemExit(1) from None
