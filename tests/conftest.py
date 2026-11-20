# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Test configuration: make the validation scripts importable by name.

Under pytest-xdist every worker writes its own numba cache directory, set before
numba is imported: numba numbers a new cache entry from the index it last read, so
two processes that compile the same kernel for different signatures at the same
time can overwrite each other's entries. The directory is a subdirectory of
``NUMBA_CACHE_DIR`` when that is set, else of the temporary directory, and is
removed at the end of the run. A serial run keeps numba's own location.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

_WORKER = os.environ.get("PYTEST_XDIST_WORKER")
_NUMBA_CACHE: Path | None = None
if _WORKER:
    _RUN = os.environ.get("PYTEST_XDIST_TESTRUNUID", "run")
    _ROOT = os.environ.get("NUMBA_CACHE_DIR") or tempfile.gettempdir()
    _NUMBA_CACHE = Path(_ROOT) / f"cablexsec-numba-{_RUN}-{_WORKER}"
    os.environ["NUMBA_CACHE_DIR"] = str(_NUMBA_CACHE)

# The sea-state batch workers keep persistent numba caches in the user cache
# directory; the tests give them a folder of this process, removed at the end.
_BATCH_CACHE: Path | None = None
if "CABLEXSEC_BATCH_CACHE_DIR" not in os.environ:
    _BATCH_CACHE = Path(tempfile.gettempdir()) / f"cablexsec-batch-{os.getpid()}"
    os.environ["CABLEXSEC_BATCH_CACHE_DIR"] = str(_BATCH_CACHE)

_SCRIPTS = Path(__file__).resolve().parents[1] / "validation" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def pytest_unconfigure(config: object) -> None:
    """Remove the numba caches of this process at the end of the run."""
    for folder in (_NUMBA_CACHE, _BATCH_CACHE):
        if folder is not None:
            shutil.rmtree(folder, ignore_errors=True)
