#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Check that CableXSec release identifiers agree across public interfaces."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def capture(relative_path: str, pattern: str, label: str) -> str:
    match = re.search(pattern, read(relative_path), flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"{relative_path}: cannot find {label}")
    return match.group(1)


# Spellings that name a CableXSec release: "CableXSec 0.1.0", "CableXSec v0.1.0",
# "v0.1.0", "cablexsec-0.1.0-py3-none-any.whl" and ".../releases/tag/v0.1.0".
VERSION_REFERENCE = re.compile(
    r"(?i)(?:cablexsec[ _-]+v?|cablexsec-|releases/tag/v|(?<![\w.])v)"
    r"(\d+\.\d+\.\d+)(?![\w.]*\d)"
)
SWEPT_SUFFIXES = {
    ".cff",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".toml",
    ".txt",
    ".yml",
    ".yaml",
}
# Records of a past release keep the version they describe; the hook configuration
# pins tool versions, not CableXSec versions.
HISTORICAL = re.compile(
    r"^(CHANGELOG\.md|validation/RELEASE_\d+_\d+_\d+\.md|\.pre-commit-config\.yaml)$"
)


def tracked_files() -> list[str]:
    try:
        listing = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
        ).stdout.decode("utf-8")
        names = [name for name in listing.split("\0") if name]
    except (OSError, subprocess.CalledProcessError):
        names = [
            path.relative_to(ROOT).as_posix()
            for path in ROOT.rglob("*")
            if path.is_file()
        ]
    return sorted(name for name in names if Path(name).suffix in SWEPT_SUFFIXES)


def stale_version_references(version: str) -> list[str]:
    """``path:line: text`` for every reference to a CableXSec version other than ``version``."""
    stale = []
    for name in tracked_files():
        if HISTORICAL.match(name):
            continue
        path = ROOT / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            for match in VERSION_REFERENCE.finditer(line):
                if match.group(1) != version and match.group(1).startswith("0."):
                    stale.append(f"{name}:{number}: {line.strip()}")
    return stale


def main() -> int:
    try:
        version = capture(
            "pyproject.toml", r'^version = "(\d+\.\d+\.\d+)"$', "project version"
        )
        citation_version = capture(
            "CITATION.cff", r"^version:\s*([^\s]+)$", "citation version"
        )
        release_date = capture(
            "CITATION.cff", r"^date-released:\s*(\d{4}-\d{2}-\d{2})$", "release date"
        )
    except ValueError as error:
        print(f"release metadata: FAIL\n- {error}", file=sys.stderr)
        return 1

    changelog = read("CHANGELOG.md")
    checks = {
        "CITATION.cff version": citation_version == version,
        "Python runtime version": f'__version__ = "{version}"'
        in read("src/cablexsec/__init__.py"),
        "CHANGELOG release heading": f"## [{version}] - {release_date}" in changelog,
        "CHANGELOG link target": "https://github.com/SMI-Lab-Inha/CableXSec"
        in changelog,
        "README release": f"v{version}" in read("README.md"),
        "Release record": (
            ROOT / "validation" / f"RELEASE_{version.replace('.', '_')}.md"
        ).is_file(),
    }
    failed = [name for name, ok in checks.items() if not ok]
    stale = stale_version_references(version)
    if failed or stale:
        print("release metadata: FAIL", file=sys.stderr)
        for name in failed:
            print(f"- {name} does not match {version}", file=sys.stderr)
        for line in stale:
            print(f"- stale version reference: {line}", file=sys.stderr)
        return 1
    print(f"release metadata: PASS (CableXSec {version}, released {release_date})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
