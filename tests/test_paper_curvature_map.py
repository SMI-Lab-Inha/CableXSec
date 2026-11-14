# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""The earlier curvature mapping (``paper_r2``) is kept exactly.

Before the curvature-sign correction the strain gradient was formed as
(k_y, -k_x) from the rotation curvature. With ``curvature_map="paper_r2"`` that
map must be reproduced exactly; the default right-handed map is (-k_y, k_x).
Both are a swap and a sign change, which are exact in floating point, so the
check holds on every platform.
"""

from __future__ import annotations

import numpy as np

from cablexsec.mechanics.section_conventions import rotation_to_strain_curvature


def _history():
    t = np.arange(1201) * 0.05
    kx = 0.02 * np.sin(2 * np.pi * t / 9.0) + 0.005 * np.sin(2 * np.pi * t / 2.3)
    ky = 0.004 + 0.008 * np.sin(2 * np.pi * t / 13.0 + 0.7)
    return kx, ky


def test_paper_gradient_is_the_earlier_map():
    kx, ky = _history()
    bx, by = rotation_to_strain_curvature(kx, ky, curvature_map="paper_r2")
    assert np.array_equal(bx, ky) and np.array_equal(by, -kx)
    rx, ry = rotation_to_strain_curvature(kx, ky)
    assert np.array_equal(rx, -ky) and np.array_equal(ry, kx)
