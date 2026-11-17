# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Saint-Venant twist rigidity of the trefoil bundle and its filler."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cablexsec.mechanics.concentric_section import ElasticAnnulus
from cablexsec.mechanics.trefoil_section import (
    SECTOR,
    SYMMETRY_FACTOR,
    ElasticFiller,
    TrefoilBundle,
    build_trefoil_model,
    saint_venant_rigidity,
)


def _rectangle(a: float, b: float, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """Six-node triangles on an a x b rectangle away from the origin."""
    gx = np.linspace(3.0, 3.0 + a, 2 * nx + 1)
    gy = np.linspace(-7.0, -7.0 + b, 2 * ny + 1)
    xy = np.array([(x, y) for y in gy for x in gx])

    def index(i: int, j: int) -> int:
        return j * (2 * nx + 1) + i

    conn = []
    for j in range(ny):
        for i in range(nx):
            p, q = (2 * i, 2 * j), (2 * i + 2, 2 * j)
            r, s = (2 * i + 2, 2 * j + 2), (2 * i, 2 * j + 2)
            for u, v, w in ((p, q, r), (p, r, s)):
                mid = [
                    ((m[0] + n[0]) // 2, (m[1] + n[1]) // 2)
                    for m, n in ((u, v), (v, w), (w, u))
                ]
                conn.append([index(*x) for x in (u, v, w, *mid)])
    return xy, np.array(conn, dtype=np.int64)


def _rectangle_series(a: float, b: float) -> float:
    """Saint-Venant torsion constant of an a x b rectangle (a >= b), series."""
    total = sum(math.tanh(n * math.pi * a / (2 * b)) / n**5 for n in range(1, 400, 2))
    return a * b**3 / 3 * (1 - 192 / math.pi**5 * b / a * total)


@pytest.mark.analytical_sanity
@pytest.mark.parametrize(("a", "nx", "ny"), [(1.0, 16, 16), (2.0, 16, 8), (4.0, 24, 6)])
def test_free_warping_reproduces_the_rectangle_series(
    a: float, nx: int, ny: int
) -> None:
    xy, conn = _rectangle(a, 1.0, nx, ny)
    value = saint_venant_rigidity(xy, conn, np.ones(len(conn)), mirror_angles=())
    # Displacement FE is an upper bound that converges from above.
    assert value >= _rectangle_series(a, 1.0)
    assert value == pytest.approx(_rectangle_series(a, 1.0), rel=3e-4)


@pytest.mark.analytical_sanity
def test_mirrored_annulus_sector_keeps_the_polar_moment() -> None:
    r_in, r_out, nr, nt = 0.5, 1.0, 6, 24
    keys: dict[tuple[int, int], int] = {}
    xy: list[tuple[float, float]] = []

    def node(i: int, j: int) -> int:
        if (i, j) not in keys:
            r = r_in + (r_out - r_in) * i / (2 * nr)
            t = SECTOR * j / (2 * nt)
            keys[(i, j)] = len(xy)
            xy.append((r * math.cos(t), r * math.sin(t)))
        return keys[(i, j)]

    conn = []
    for i in range(nr):
        for j in range(nt):
            p, q = (2 * i, 2 * j), (2 * i + 2, 2 * j)
            r, s = (2 * i + 2, 2 * j + 2), (2 * i, 2 * j + 2)
            for u, v, w in ((p, q, r), (p, r, s)):
                mid = [
                    ((m[0] + n[0]) // 2, (m[1] + n[1]) // 2)
                    for m, n in ((u, v), (v, w), (w, u))
                ]
                conn.append([node(*x) for x in (u, v, w, *mid)])
    value = saint_venant_rigidity(
        np.array(xy), np.array(conn, dtype=np.int64), np.full(len(conn), 2.0)
    )
    exact = 2.0 * math.pi / 2 * (r_out**4 - r_in**4) / SYMMETRY_FACTOR
    assert value == pytest.approx(exact, rel=1e-6)


A = 0.02
RC = 2 * A / math.sqrt(3)


def _bundle(filler: ElasticFiller | None) -> TrefoilBundle:
    return TrefoilBundle(
        (
            ElasticAnnulus("cond", 0.0, 0.01, 117e9, 0.34),
            ElasticAnnulus("ins", 0.01, A, 0.15e9, 0.45),
        ),
        RC,
        (ElasticAnnulus("ring", RC + A, RC + A + 0.003, 0.9e9, 0.46),),
        filler=filler,
        angular_cells=24,
    )


@pytest.mark.analytical_sanity
def test_cores_and_ring_twist_about_their_own_axes() -> None:
    """Straight cores about their centres, the ring about the cable axis."""
    model = build_trefoil_model(_bundle(None))

    def shear(e: float, nu: float) -> float:
        return e / (2 * (1 + nu))

    cores = 3 * (
        shear(117e9, 0.34) * math.pi / 2 * 0.01**4
        + shear(0.15e9, 0.45) * math.pi / 2 * (A**4 - 0.01**4)
    )
    ring = shear(0.9e9, 0.46) * math.pi / 2 * ((RC + A + 0.003) ** 4 - (RC + A) ** 4)
    assert model.stiffness[1, 1] == pytest.approx(cores + ring, rel=1e-5)


@pytest.mark.analytical_sanity
def test_filler_lobes_twist_as_separate_bodies() -> None:
    """Frictionless contacts carry no axial shear, so the filler lobes warp
    freely: they add far less than their polar moment about the cable axis,
    which they would add only if bonded to the ring with no warping."""
    filler = ElasticFiller("filler", 0.9e9, 0.45)
    without = build_trefoil_model(_bundle(None))
    model = build_trefoil_model(_bundle(filler))
    added = model.stiffness[1, 1] - without.stiffness[1, 1]
    group = model.element_groups["filler"]
    mu = model.element_moduli[group] / (2 * (1 + model.element_poisson[group]))
    conn = model.element_nodes[group]
    own = saint_venant_rigidity(model.nodes, conn, mu)
    assert added == pytest.approx(SYMMETRY_FACTOR * own, rel=1e-9)
    # The lobes' polar moment about the cable axis (one-point estimate per
    # element), the rigidity of a filler bonded to the ring without warping.
    xy = model.nodes[conn[:, :3]]
    area = 0.5 * np.abs(
        (xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
        - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1])
    )
    centroid = xy.mean(axis=1)
    bonded = float(np.sum(mu * area * (centroid**2).sum(axis=1)))
    assert 0.0 < own < 0.2 * bonded
