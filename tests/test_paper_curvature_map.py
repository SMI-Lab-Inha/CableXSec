# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""The earlier curvature mapping (``paper_r2``) is pinned bit for bit.

The digest below was recorded with the code before the curvature-sign
correction: the strain gradient formed from a fixed synthetic
rotation-curvature history. With ``curvature_map="paper_r2"`` it must be
reproduced bit for bit.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from cablexsec.mechanics.section_conventions import rotation_to_strain_curvature

ROOT = Path(__file__).resolve().parents[1]
GRADIENT_SHA256 = "a7f1674482f494742bbc897f87627703717be31a0d06b700011009c864c9df33"
STATISTICS_SHA256 = {
    "inner_armour": "6812a7875ee2c17197648f4ebec2e53e23c0cb103a6441fe8eb523f717437bdf",
    "outer_armour": "a3a9ac08c8e3da5d05acf108a53d62229b70ab06055f11310b67f29694928780",
}
CRITICAL = {"inner_armour": (40, 1), "outer_armour": (49, 31)}


def _history():
    t = np.arange(1201) * 0.05
    kx = 0.02 * np.sin(2 * np.pi * t / 9.0) + 0.005 * np.sin(2 * np.pi * t / 2.3)
    ky = 0.004 + 0.008 * np.sin(2 * np.pi * t / 13.0 + 0.7)
    strain = 2.0e-4 + 6.0e-5 * np.sin(2 * np.pi * t / 9.0 + 0.3)
    return t, kx, ky, strain


def _digest(arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def test_paper_gradient_is_bitwise_unchanged():
    _, kx, ky, _ = _history()
    paper = rotation_to_strain_curvature(kx, ky, curvature_map="paper_r2")
    assert _digest(paper) == GRADIENT_SHA256
    right = rotation_to_strain_curvature(kx, ky)
    assert _digest(right) != GRADIENT_SHA256
