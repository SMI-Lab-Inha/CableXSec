# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Public checks of the Saint-Venant twist rigidity of the trefoil section.

Run as ``python validation/scripts/check_section_torsion.py``. This is a
verification module: nothing here is used by an analysis path.

The interstitial filler of a trefoil bundle touches the cores and the ring
through frictionless contacts, so its lobes twist as separate bodies with
their own warping (``saint_venant_rigidity``). The warping solver is checked
against two closed forms:

* the Saint-Venant torsion constant of a solid rectangle, the exact series
  ``J = a b^3 / 3 [1 - 192 b / (pi^5 a) sum_odd tanh(n pi a / 2b) / n^5]``
  of ``cablexsec.mechanics.tendon_section.rectangle_torsion_constant``, for an
  aspect ratio of 2;
* a bonded annulus meshed as the 60 degree sector of the trefoil model with
  the mirror conditions, whose warping vanishes, so its rigidity is the polar
  moment ``G pi (b^4 - a^4) / 2`` over six.
"""

from __future__ import annotations

import math
import re
import tempfile
from pathlib import Path

import numpy as np

import cablexsec as cx
from cablexsec.cables.schema import CableData
from cablexsec.mechanics.radial_cable import solve_cable_radial_state
from cablexsec.mechanics.tendon_section import rectangle_torsion_constant
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
    """Torsion constant of an ``a x b`` rectangle (series, tendon_section)."""
    return rectangle_torsion_constant(a, b)


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


# Armour pitches [m] of the Chae, Nam and Shim (2023) cable studied in its
# Table 5. No torsional stiffness of the cable is measured; the stiffness is
# reported here only as the sensitivity to the core lay (doc/benchmarks.md).
CHAE_PITCHES = (1.8, 3.0)
CHAE_TWIST = 0.125  # [rad/m], Table 4 (global twist)
CHAE_HELIX_RADIUS = 0.081  # [m], packaged YAML (assumed there)


def chae_cable(pitch_m: float, core_pitch_m: float | None = None) -> CableData:
    """The packaged Chae cable re-pitched; optionally with S-lay cores."""
    text = Path(cx.chae2023_yaml_path()).read_text(encoding="utf-8")
    angle = math.degrees(math.atan(2 * math.pi * CHAE_HELIX_RADIUS / pitch_m))
    text, count = re.subn(
        r"lay_angle_deg: [0-9.]+", f"lay_angle_deg: {angle:.6f}", text
    )
    if count != 1:
        raise RuntimeError("the packaged Chae cable must have one armour lay angle")
    if core_pitch_m is not None:
        # The source lays the cores S, as the armour (Sec. 2.1.2).
        if "\ncore_lay:" in text or text.count("\nlayers:\n") != 1:
            raise RuntimeError("cannot add a core_lay block to the Chae cable")
        block = f"\ncore_lay:\n  pitch_m: {core_pitch_m}\n  lay_direction: left_hand\n"
        text = text.replace("\nlayers:\n", block + "\nlayers:\n", 1)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "chae.yaml"
        path.write_text(text, encoding="utf-8")
        return cx.load_cable(str(path))


def chae_free_torsion(pitch_m: float, core_pitch_m: float | None = None) -> float:
    """Free-end secant torsional stiffness [N m^2] at the source's twist.

    Zero tension, 0.125 rad/m in the sense that unwinds the armour (the
    coiling sense), solved section with the loose-layer neighbour rule of the
    coiling request.
    """
    cable = chae_cable(pitch_m, core_pitch_m)
    (armour,) = cable.helical_layers
    # Positive twist tightens a right-hand lay, so it unwinds a left-hand one.
    sign = 1.0 if armour.lay_direction == "left_hand" else -1.0
    state = solve_cable_radial_state(
        cable,
        0.0,
        prescribed_twist_per_m=sign * CHAE_TWIST,
        loose_layers="neighbour",
    )
    return abs(float(state.result.torque_Nm)) / CHAE_TWIST


def chae_checks() -> dict[str, dict[float, float]]:
    """Core-lay sensitivity of the free torsional stiffness [N m^2]: straight
    cores (packaged) and S-lay cores at 2.5 and 3.0 m (the source gives no core
    pitch), at each armour pitch."""
    out: dict[str, dict[float, float]] = {}
    for label, core in (("straight", None), ("S 2.5 m", 2.5), ("S 3.0 m", 3.0)):
        out[label] = {p: chae_free_torsion(p, core) for p in CHAE_PITCHES}
    return out


def main() -> None:
    for name, value in verification_checks().items():
        print(f"{name}: {value:.2e}")
    for label, values in chae_checks().items():
        print(label, {p: round(v / 1e3, 1) for p, v in values.items()})


if __name__ == "__main__":
    main()
