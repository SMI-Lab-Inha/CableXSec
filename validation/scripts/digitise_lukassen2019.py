# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Digitise the measured two-sided tendon stresses of the Sævik (2011) FBG riser.

Source
------
T. V. Lukassen, "Constitutive behavior of tensile armor wires in unbonded
flexible pipes", PhD thesis, Technical University of Denmark (2019), Ch. 2,
pp. 13-14 (PDF pages 30-31), Figs 2.7-2.11: the inner tensile armour stress
on the two narrow sides of one tendon, sigma_a (panels a) and sigma_b
(panels b), at stations 3.1-3.5 of the test of Sævik (2011) (load case 14:
500 kN, 34 MPa, rocker +-8 deg), against the bending angle theta. Each panel
holds the FE result (red) and the measurement, "experimental OMS" (blue);
the colours are those of the caption glyphs. The sensors have a 40 mm gauge
length on both narrow sides (p. 11).

Procedure (the CSV header records the numbers)
----------------------------------------------
1. Axis calibration per panel: the thin black grid lines are vector lines at
   the labelled ticks (theta -8 ... 8 deg in steps of 2, stress -135 ... 135
   MPa in steps of 30). A straight line is fitted to (PDF coordinate, label)
   per axis; the largest residual is recorded.
2. Measured series: the blue polyline of each panel (one stroked path of
   width 0.4 pt), its vertices mapped through the panel calibration in their
   stored order. The blue circle markers on the same curve are not used.

Reading resolution (half-width, not experimental uncertainty): the larger
calibration residual plus 0.01 pt (the coordinate precision of the PDF),
in data units.

Usage (needs PyMuPDF, ``pip install pymupdf``; not a CableXSec dependency)::

    python validation/scripts/digitise_lukassen2019.py THESIS.pdf

writes ``validation/paper_data/lukassen2019_lc14_two_sides.csv``.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation" / "paper_data" / "lukassen2019_lc14_two_sides.csv"
LF = "\n"

THETA_TICKS = tuple(float(v) for v in range(-8, 9, 2))
STRESS_TICKS = tuple(float(v) for v in range(-135, 136, 30))
#: PDF page index (0-based) -> stations of the panel rows, top to bottom.
PAGES = {29: ("3.1", "3.2", "3.3"), 30: ("3.4", "3.5")}
BLUE = (0.0, 0.0, 1.0)
GRID = (0.0, 0.0, 0.0)


def _fit(coord: list[float], value: tuple[float, ...]) -> tuple[float, float, float]:
    c = np.asarray(coord, dtype=float)
    v = np.asarray(value, dtype=float)
    b, a = np.polyfit(c, v, 1)
    return float(a), float(b), float(np.max(np.abs(a + b * c - v)))


def _clusters(values: list[float], size: int) -> list[list[float]]:
    """Sorted unique coordinates split into consecutive groups of ``size``."""
    u = sorted({round(v, 3) for v in values})
    if len(u) % size:
        raise RuntimeError(f"unexpected grid: {len(u)} lines, groups of {size}")
    return [u[i : i + size] for i in range(0, len(u), size)]


def _page(page: object, stations: tuple[str, ...]) -> list[dict[str, object]]:
    vertical, horizontal, lines = [], [], []
    for d in page.get_drawings():  # type: ignore[attr-defined]
        colour, width, rect = d.get("color"), d.get("width") or 0.0, d["rect"]
        if colour == GRID and abs(width - 0.1) < 0.01 and len(d["items"]) == 1:
            if rect.width < 1e-3:
                vertical.append(rect.x0)
            elif rect.height < 1e-3:
                horizontal.append(rect.y0)
        elif colour == BLUE and abs(width - 0.4) < 0.01 and d["type"] == "s":
            items = d["items"]
            pts = [items[0][1]] + [it[-1] for it in items]
            lines.append(np.array([(p.x, p.y) for p in pts], dtype=float))
    columns = _clusters(vertical, len(THETA_TICKS))
    rows = _clusters(horizontal, len(STRESS_TICKS))
    if len(columns) != 2 or len(rows) != len(stations):
        raise RuntimeError("unexpected panel layout: check the page")
    out = []
    for station, ys in zip(stations, rows, strict=True):
        for side, xs in zip(("a", "b"), columns, strict=True):
            tx = _fit(xs, THETA_TICKS)
            sy = _fit(sorted(ys), tuple(reversed(STRESS_TICKS)))  # PDF y down
            inside = [
                ln
                for ln in lines
                if xs[0] - 5 <= ln[:, 0].mean() <= xs[-1] + 5
                and ys[0] - 5 <= ln[:, 1].mean() <= ys[-1] + 5
            ]
            if len(inside) != 1:
                raise RuntimeError(f"station {station} side {side}: {len(inside)} lines")
            pts = inside[0]
            theta = tx[0] + tx[1] * pts[:, 0]
            stress = sy[0] + sy[1] * pts[:, 1]
            out.append(
                {
                    "station": station,
                    "side": side,
                    "theta": theta,
                    "stress": stress,
                    "fit_theta": tx,
                    "fit_stress": sy,
                    "half_width": (
                        tx[2] + 0.01 * abs(tx[1]),
                        sy[2] + 0.01 * abs(sy[1]),
                    ),
                }
            )
    return out


def main(argv: list[str]) -> None:
    import pymupdf  # noqa: PLC0415

    pdf = Path(argv[1])
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    doc = pymupdf.open(pdf)
    panels = []
    for index, stations in PAGES.items():
        panels.extend(_page(doc[index], stations))
    hw_t = max(p["half_width"][0] for p in panels)  # type: ignore[index]
    hw_s = max(p["half_width"][1] for p in panels)  # type: ignore[index]
    res_t = max(p["fit_theta"][2] for p in panels)  # type: ignore[index]
    res_s = max(p["fit_stress"][2] for p in panels)  # type: ignore[index]
    head = [
        "# Lukassen (2019) PhD thesis, DTU, Ch. 2, Figs 2.7-2.11 (PDF pages 30-31):",
        "# measured (experimental OMS, blue) inner tensile armour stress on the two",
        "# narrow sides of one tendon, sigma_a (side a) and sigma_b (side b), at",
        "# stations 3.1-3.5 of the Saevik (2011) FBG riser, load case 14 (500 kN,",
        "# 34 MPa, rocker +-8 deg); 40 mm gauge length (p. 11). Stresses are the",
        "# bending stress variations from the state under gravity, tension and",
        "# pressure (p. 12).",
        f"# thesis PDF sha256 {sha}",
        "# method: native PDF polyline in stored order"
        " (validation/scripts/digitise_lukassen2019.py)",
        f"# calibration: per panel on the grid lines; largest residual {res_t:.2e} deg,"
        f" {res_s:.2e} MPa",
        f"# reading half-width: {hw_t:.2e} deg, {hw_s:.2e} MPa"
        " (NOT experimental uncertainty)",
        "station,side,sample,theta_deg,stress_MPa",
    ]
    rows = []
    for p in panels:
        for i, (t, s) in enumerate(zip(p["theta"], p["stress"], strict=True)):  # type: ignore[call-overload]
            rows.append(f"{p['station']},{p['side']},{i},{t:.4f},{s:.3f}")
    OUT.write_text(LF.join(head + rows) + LF, encoding="utf-8", newline=LF)
    for p in panels:
        s = np.asarray(p["stress"])
        print(p["station"], p["side"], len(s), f"range {np.ptp(s):.1f} MPa")


if __name__ == "__main__":
    main(sys.argv)
