# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Small console-portability helper shared by the executable demos."""

from __future__ import annotations

import sys


def configure_console() -> None:
    """Prevent reporting text from failing on a locale-limited console."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
