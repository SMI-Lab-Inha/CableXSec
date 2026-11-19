# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Test configuration: make the validation scripts importable by name.

Under pytest-xdist every worker gets its own numba cache directory before numba is
imported, so that concurrent workers (and the subprocesses they start) never write
the same on-disk kernel cache; a serial run keeps numba's default location.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

_WORKER = os.environ.get("PYTEST_XDIST_WORKER")
_NUMBA_CACHE: Path | None = None
if _WORKER and "NUMBA_CACHE_DIR" not in os.environ:
    _RUN = os.environ.get("PYTEST_XDIST_TESTRUNUID", "run")
    _NUMBA_CACHE = Path(tempfile.gettempdir()) / f"cablexsec-numba-{_RUN}-{_WORKER}"
    os.environ["NUMBA_CACHE_DIR"] = str(_NUMBA_CACHE)

_SCRIPTS = Path(__file__).resolve().parents[1] / "validation" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def pytest_unconfigure(config: object) -> None:
    """Remove the worker's own numba cache at the end of the run."""
    if _NUMBA_CACHE is not None:
        shutil.rmtree(_NUMBA_CACHE, ignore_errors=True)
