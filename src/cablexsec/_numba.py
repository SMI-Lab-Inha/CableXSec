# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Optional ``numba`` support shared by the compiled kernels.

``numba`` is an optional dependency (extra ``fast``). When it is importable,
:func:`njit` compiles the decorated function; otherwise, or when the
environment variable ``CABLEXSEC_DISABLE_NUMBA`` is set to ``1``, ``true``,
``yes`` or ``on``, :func:`njit` returns the function unchanged and the same
algorithm runs as ordinary Python. Callers that have a faster NumPy or
pure-Python reference path use :data:`HAVE_NUMBA` to select it instead.

The decorator is typed to return the decorated function's own signature, so
the kernels remain checked by ``mypy --strict`` whether or not ``numba`` is
installed.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from typing import Any, TypeVar, cast

__all__ = [
    "DISABLE_ENV",
    "HAVE_NUMBA",
    "kernels_compiled",
    "njit",
    "numba_version",
    "prange",
]

DISABLE_ENV = "CABLEXSEC_DISABLE_NUMBA"

F = TypeVar("F", bound=Callable[..., Any])


def _disabled() -> bool:
    flag = os.environ.get(DISABLE_ENV, "").strip().lower()
    return flag in {"1", "true", "yes", "on"}


_numba: Any = None
if not _disabled():
    try:  # pragma: no cover - availability depends on the installation
        _numba = importlib.import_module("numba")
    except ImportError:  # pragma: no cover
        _numba = None

HAVE_NUMBA: bool = _numba is not None

prange: Callable[[int], range] = range if _numba is None else _numba.prange


def njit(**options: Any) -> Callable[[F], F]:
    """``numba.njit(**options)`` when compiled kernels are active, else identity."""
    if _numba is None:

        def identity(function: F) -> F:
            return function

        return identity
    return cast(Callable[[F], F], _numba.njit(**options))


def kernels_compiled() -> bool:
    """True when the kernels run compiled: numba active and its JIT enabled.

    ``NUMBA_DISABLE_JIT=1`` keeps :data:`HAVE_NUMBA` true but runs every
    kernel as Python (used for the kernel coverage pass).
    """
    return _numba is not None and not bool(_numba.config.DISABLE_JIT)


def numba_version() -> str | None:
    """Installed ``numba`` distribution version, or ``None`` if not installed.

    This reports the installation, independently of ``CABLEXSEC_DISABLE_NUMBA``.
    """
    try:
        return version("numba")
    except PackageNotFoundError:
        return None
