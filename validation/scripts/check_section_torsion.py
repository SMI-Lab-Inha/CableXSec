# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Public checks of the Saint-Venant twist rigidity of the trefoil section.

Run as ``python validation/scripts/check_section_torsion.py``. This is a
verification module: nothing here is used by an analysis path.

The interstitial filler of a trefoil bundle touches the cores and the ring
through frictionless contacts, so its lobes twist as separate bodies with
their own warping (``saint_venant_rigidity``). The warping solver is checked
against two closed forms:

* the Saint-Venant torsion constant of a solid rectangle, series solution
  ``J = a b^3 / 3 [1 - 192 b / (pi^5 a) sum_odd tanh(n pi a / 2b) / n^5]``
  (Timoshenko and Goodier, Theory of Elasticity, 3rd ed., Sec. 109), for an
  aspect ratio of 2;
* a bonded annulus meshed as the 60 degree sector of the trefoil model with
  the mirror conditions, whose warping vanishes, so its rigidity is the polar
  moment ``G pi (b^4 - a^4) / 2`` over six.
"""

from __future__ import annotations

import math

import numpy as np

from cablexsec.mechanics.trefoil_section import SECTOR, saint_venant_rigidity


def _t6(grid: np.ndarray, cells: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Six-node triangles on a structured (2 nx + 1) x (2 ny + 1) node grid."""
    nx, ny = cells
    xy = grid.reshape(-1, 2)

    def index(i: int, j: int) -> int:
        return j * (2 * nx + 1) + i

    conn = []
    for j in range(ny):
        for i in range(nx):
            p, q = (2 * i, 2 * j), (2 * i + 2, 2 * j)
            r, s = (2 * i + 2, 2 * j + 2), (2 * i, 2 * j + 2)
            for u, v, w in ((p, q, r), (p, r, s)):
                mids = [
                    ((m[0] + n[0]) // 2, (m[1] + n[1]) // 2)
                    for m, n in ((u, v), (v, w), (w, u))
                ]
                conn.append([index(*x) for x in (u, v, w, *mids)])
    return xy, np.array(conn, dtype=np.int64)


def rectangle_series(a: float, b: float) -> float:
    """Torsion constant of an ``a x b`` rectangle, ``a >= b`` (series)."""
    total = sum(math.tanh(n * math.pi * a / (2 * b)) / n**5 for n in range(1, 400, 2))
    return a * b**3 / 3 * (1 - 192 / math.pi**5 * b / a * total)


def rectangle_error(a: float = 2.0, b: float = 1.0, nx: int = 16) -> float:
    """Relative error of the warping solver on an ``a x b`` rectangle."""
    ny = max(2, round(nx * b / a))
    gx = np.linspace(3.0, 3.0 + a, 2 * nx + 1)  # away from the twist axis
    gy = np.linspace(-7.0, -7.0 + b, 2 * ny + 1)
    grid = np.array([[(x, y) for x in gx] for y in gy])
    xy, conn = _t6(grid, (nx, ny))
    value = saint_venant_rigidity(xy, conn, np.ones(len(conn)), mirror_angles=())
    return value / rectangle_series(a, b) - 1.0


def annulus_error(r_in: float = 0.5, r_out: float = 1.0) -> float:
    """Relative error of a mirrored annulus sector against its polar moment."""
    nr, nt = 6, 24
    radii = np.linspace(r_in, r_out, 2 * nr + 1)
    angles = np.linspace(0.0, SECTOR, 2 * nt + 1)
    grid = np.array(
        [[(r * math.cos(t), r * math.sin(t)) for r in radii] for t in angles]
    )
    xy, conn = _t6(grid, (nr, nt))
    value = saint_venant_rigidity(xy, conn, np.ones(len(conn)))
    return value / (math.pi / 2 * (r_out**4 - r_in**4) / 6) - 1.0


def verification_checks() -> dict[str, float]:
    """Relative errors of the warping solver against the two closed forms."""
    return {"rectangle": rectangle_error(), "annulus": annulus_error()}


def main() -> None:
    for name, value in verification_checks().items():
        print(f"{name}: {value:.2e}")


if __name__ == "__main__":
    main()
