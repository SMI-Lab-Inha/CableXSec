# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Digitise the four-point bending records of Zeidan (2025) and Zeidan et al. (2026).

Sources
-------
* L. Zeidan, "Multiscale modeling of the nonlinear mechanical behavior of
  submarine power cable for floating wind turbines", PhD thesis, Ecole
  Centrale de Nantes (defended 8 December 2025), Chapter 5:
  Fig. 5.5 (PDF page 104), first series of four-point tests at 30, 40, 50 and
  60 mm, and Fig. 5.6 (PDF page 105), the same specimen after more than 1000
  cycles at +-60 mm ("stabilised"). Both figures are native PDF polylines
  (one stroked path per series, width 1.263 pt), so the record is read in its
  plotted order without a raster step.
* L. Zeidan, P. Cartraud, G. Marckmann, L. Stainier, Int. J. Solids Struct.
  338 (2026) 114089, doi:10.1016/j.ijsolstr.2026.114089: Fig. 17 (page 12)
  repeats the stabilised loops with every stroke converted to a filled
  outline; it is used here only to confirm that it shows the same record as
  thesis Fig. 5.6. Fig. 18 (page 12) is a raster image (1251 x 754 px) of the
  measured and computed deflection along the half span; its series are read
  at the native resolution.

Procedure (the CSV headers record the numbers)
----------------------------------------------
1. Axis calibration. The grey grid lines of each figure are vector lines at
   the labelled ticks (curvature -0.2 ... 0.2 1/m, moment -2 ... 3 kN m, as
   printed). A straight line is fitted to (PDF coordinate, label) per axis;
   the largest residual is recorded.
2. Loops (thesis Figs 5.5 and 5.6). The vertices of each series' polyline
   are mapped through the calibration in their stored order.
3. Cross-check (paper Fig. 17). Its outline vertices are mapped through its
   own grid calibration and their distance to the thesis polyline of the
   same series is measured in Fig. 17 points; the median should not exceed
   half the outline thickness plus the calibration residual.
4. Deflections (paper Fig. 18). Gridlines are found as the grey pixel rows and
   columns of the plot frame; series pixels are classified by colour (the
   default line colours of the plotting library, Euclidean RGB distance
   < 60); the value of a column is the median row of its pixels. The solid
   (measured) and dashed (model) lines of one series lie within about 1 mm
   of each other and are not separated.

Reading resolution (half-width, not experimental uncertainty): for the
polylines, the calibration residual plus 0.01 pt (the coordinate precision of
the PDF); for Fig. 18, the calibration residual plus one pixel.

Usage (needs PyMuPDF, ``pip install pymupdf``; not a CableXSec dependency)::

    python validation/scripts/digitise_zeidan2025.py THESIS.pdf PAPER.pdf

writes ``validation/paper_data/zeidan2025_fig5_{5,6}_{30,40,50,60}mm.csv``,
``zeidan2026_fig18_deflection.csv`` and ``zeidan2025_digitisation.json``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation" / "paper_data"
FloatArray = NDArray[np.float64]

#: Series colours (default line colours of the plotting library) and labels.
SERIES = {
    "30mm": (0.122, 0.467, 0.706),
    "40mm": (1.0, 0.498, 0.055),
    "50mm": (0.173, 0.627, 0.173),
    "60mm": (0.839, 0.153, 0.157),
}
CURVATURE_TICKS = (-0.2, -0.1, 0.0, 0.1, 0.2)
MOMENT_TICKS_KNM = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
THESIS_FIGURES = {"5_5": 104, "5_6": 105}
PAPER_FIG17_PAGE = 12
PAPER_FIG18_PAGE = 12
DEFLECTION_X_TICKS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
DEFLECTION_Y_TICKS = (0.0, -0.02, -0.04, -0.06)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _close(a: tuple[float, ...] | None, b: tuple[float, ...], tol: float) -> bool:
    return a is not None and all(abs(x - y) < tol for x, y in zip(a, b, strict=True))


def _fit(coord: list[float], value: tuple[float, ...]) -> tuple[float, float, float]:
    """Least-squares ``value = a + b coord``; returns (a, b, largest residual)."""
    c = np.asarray(coord, dtype=float)
    v = np.asarray(value, dtype=float)
    b, a = np.polyfit(c, v, 1)
    return float(a), float(b), float(np.max(np.abs(a + b * c - v)))


def _thesis_figure(page: object) -> dict[str, object]:
    """Calibrated, ordered polylines of one thesis loop figure."""
    vertical, horizontal = [], []
    series: dict[str, FloatArray] = {}
    for path in page.get_drawings():  # type: ignore[attr-defined]
        colour = path.get("color")
        items = path["items"]
        if path["type"] == "s" and _close(colour, (0.69, 0.69, 0.69), 0.01):
            (_, p0, p1) = items[0]
            if abs(p0.x - p1.x) < 1e-6:
                vertical.append(p0.x)
            elif abs(p0.y - p1.y) < 1e-6:
                horizontal.append(p0.y)
            continue
        if path["type"] != "s" or len(items) < 100:
            continue
        for name, rgb in SERIES.items():
            if _close(colour, rgb, 0.01):
                pts = [items[0][1]] + [it[-1] for it in items]
                series[name] = np.array([(p.x, p.y) for p in pts], dtype=float)
    vertical.sort()
    horizontal.sort(reverse=True)  # PDF y grows downwards
    if len(vertical) != len(CURVATURE_TICKS) or len(horizontal) != len(
        MOMENT_TICKS_KNM
    ):
        raise RuntimeError("unexpected grid: check the page and figure")
    kx = _fit(vertical, CURVATURE_TICKS)
    my = _fit(horizontal, MOMENT_TICKS_KNM)
    data = {}
    for name, pts in series.items():
        kappa = kx[0] + kx[1] * pts[:, 0]
        moment = (my[0] + my[1] * pts[:, 1]) * 1e3
        data[name] = np.column_stack([kappa, moment])
    return {
        "grid_x_pt": vertical,
        "grid_y_pt": horizontal,
        "curvature_fit": kx,
        "moment_fit_kNm": my,
        "series": data,
    }


def _fig17_cross_check(page: object, thesis: dict[str, object]) -> dict[str, object]:
    """Distance of the paper Fig. 17 outlines to the thesis Fig. 5.6 polylines."""
    from scipy.spatial import cKDTree

    vertical, horizontal = [], []
    outlines: dict[str, FloatArray] = {}
    legend_thickness = []
    for path in page.get_drawings():  # type: ignore[attr-defined]
        fill = path.get("fill")
        rect = path["rect"]
        if _close(fill, (0.69, 0.69, 0.69), 0.01) and len(path["items"]) == 1:
            if rect.width < 1.0:
                vertical.append(0.5 * (rect.x0 + rect.x1))
            elif rect.height < 1.0:
                horizontal.append(0.5 * (rect.y0 + rect.y1))
            continue
        for name, rgb in SERIES.items():
            if not _close(fill, rgb, 0.01):
                continue
            items = path["items"]
            pts = np.array([(it[-1].x, it[-1].y) for it in items], dtype=float)
            if len(items) > 100:
                outlines[name] = pts
            else:
                legend_thickness.append(rect.height)
    vertical.sort()
    horizontal.sort(reverse=True)
    kx = _fit(vertical, CURVATURE_TICKS)
    my = _fit(horizontal, MOMENT_TICKS_KNM)
    series = thesis["series"]
    assert isinstance(series, dict)
    out: dict[str, object] = {
        "outline_thickness_pt": float(np.median(legend_thickness)),
        "calibration_residual": [kx[2], my[2]],
    }
    for name, pts in outlines.items():
        ref = series[name]
        # thesis data -> Fig. 17 points, densified to 0.05 pt
        px = (ref[:, 0] - kx[0]) / kx[1]
        py = (ref[:, 1] / 1e3 - my[0]) / my[1]
        dense = []
        for i in range(len(px) - 1):
            n = max(2, int(np.hypot(px[i + 1] - px[i], py[i + 1] - py[i]) / 0.05) + 1)
            t = np.linspace(0.0, 1.0, n, endpoint=False)
            dense.append(
                np.column_stack(
                    [px[i] + t * (px[i + 1] - px[i]), py[i] + t * (py[i + 1] - py[i])]
                )
            )
        tree = cKDTree(np.vstack(dense))
        dist, _ = tree.query(pts)
        out[name] = {
            "median_pt": float(np.median(dist)),
            "p95_pt": float(np.percentile(dist, 95)),
        }
    return out


def _fig18(doc: object) -> dict[str, object]:
    """Deflection along the half span, paper Fig. 18 (raster, native pixels)."""
    import pymupdf

    page = doc[PAPER_FIG18_PAGE - 1]  # type: ignore[index]
    images = page.get_images(full=True)
    # Fig. 18 is the lower of the two images on the page.
    xref = max(images, key=lambda im: page.get_image_bbox(im).y0)[0]
    pix = pymupdf.Pixmap(doc, xref)
    if pix.n - pix.alpha >= 4:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )[:, :, :3]
    rgb = img.astype(float)
    # Plot frame: the long dark rows and columns (axis lines); the x axis is the
    # lowest such row, the y axis the leftmost such column.
    dark = rgb.max(axis=2) < 60
    rows = dark.sum(axis=1)
    cols = dark.sum(axis=0)
    frame_bottom = int(np.flatnonzero(rows > 0.5 * rows.max())[-1])
    frame_top = int(np.flatnonzero(rows > 0.5 * rows.max())[0])
    frame_left = int(np.flatnonzero(cols > 0.5 * cols.max())[0])
    # Grid: light-grey achromatic pixels.
    mean = rgb.mean(axis=2)
    grey = (np.ptp(rgb, axis=2) < 12) & (mean > 190) & (mean < 230)

    def peaks(fraction: FloatArray, offset: int) -> list[float]:
        idx = np.flatnonzero(fraction > 0.5)
        groups = np.split(idx, np.flatnonzero(np.diff(idx) > 2) + 1)
        return [
            float(offset + np.average(g, weights=fraction[g])) for g in groups if g.size
        ]

    # x grid lines: a band just above the x axis, below every curve.
    band = grey[frame_bottom - 40 : frame_bottom - 4, frame_left + 3 :]
    gx = [float(frame_left)] + peaks(band.mean(axis=0), frame_left + 3)
    xf0 = _fit(gx, DEFLECTION_X_TICKS)
    # y grid lines: columns from the y axis to x = 0.9 m (left of the legends).
    c_end = int(round((0.9 - xf0[0]) / xf0[1]))
    band = grey[frame_top + 2 : frame_bottom - 2, frame_left + 3 : c_end]
    gy = peaks(band.mean(axis=1), frame_top + 2)
    if len(gx) != len(DEFLECTION_X_TICKS):
        raise RuntimeError(f"Fig. 18: found {len(gx)} x grid lines")
    gy = sorted(gy)
    if len(gy) != len(DEFLECTION_Y_TICKS):
        raise RuntimeError(f"Fig. 18: found {len(gy)} y grid lines")
    xf = _fit(gx, DEFLECTION_X_TICKS)
    yf = _fit(gy, DEFLECTION_Y_TICKS)
    colours = {
        "30mm": (31, 119, 180),
        "40mm": (255, 127, 14),
        "50mm": (44, 160, 44),
        "60mm": (214, 39, 40),
    }
    # Legend boxes sit right of x = 1.0 m; read only up to the half span.
    x_end = int(round((1.0 - xf[0]) / xf[1]))
    out: dict[str, object] = {"x_fit": xf, "y_fit": yf}
    for name, c in colours.items():
        mask = np.linalg.norm(rgb - np.asarray(c, dtype=float), axis=2) < 60
        xs, ys = [], []
        for j in range(frame_left + 2, x_end + 1):
            r = np.flatnonzero(mask[:frame_bottom, j])
            if r.size:
                xs.append(xf[0] + xf[1] * j)
                ys.append(yf[0] + yf[1] * float(np.median(r)))
        out[name] = np.column_stack([xs, ys])
    out["pixel_m"] = (abs(xf[1]), abs(yf[1]))
    return out


def _header(lines: list[str]) -> str:
    return "".join(f"# {line}\n" for line in lines)


def main(thesis_pdf: str, paper_pdf: str) -> None:
    import pymupdf

    thesis_path, paper_path = Path(thesis_pdf), Path(paper_pdf)
    thesis = pymupdf.open(thesis_path)
    paper = pymupdf.open(paper_path)
    record: dict[str, object] = {
        "thesis_sha256": _sha256(thesis_path),
        "paper_sha256": _sha256(paper_path),
        "paper_doi": "10.1016/j.ijsolstr.2026.114089",
    }
    figures = {}
    for fig, page_no in THESIS_FIGURES.items():
        res = _thesis_figure(thesis[page_no - 1])
        figures[fig] = res
        kx, my = res["curvature_fit"], res["moment_fit_kNm"]
        assert isinstance(kx, tuple) and isinstance(my, tuple)
        k_half = kx[2] + 0.01 * abs(kx[1])
        m_half = (my[2] + 0.01 * abs(my[1])) * 1e3
        series = res["series"]
        assert isinstance(series, dict)
        record[f"fig{fig}"] = {
            "pdf_page": page_no,
            "curvature_fit": kx,
            "moment_fit_kNm": my,
            "points": {k: int(len(v)) for k, v in series.items()},
            "reading_halfwidth": [k_half, m_half],
        }
        for name, data in series.items():
            label = "first series" if fig == "5_5" else "after >1000 cycles at +-60 mm"
            head = _header(
                [
                    f"Zeidan (2025) PhD thesis, Fig. {fig.replace('_', '.')} "
                    f"(PDF page {page_no}), {name[:2]} mm, {label}",
                    f"thesis PDF sha256 {record['thesis_sha256']}",
                    "method: native PDF polyline in plotted order "
                    "(validation/scripts/digitise_zeidan2025.py)",
                    f"calibration: kappa = {kx[0]:.6g} + {kx[1]:.8g} x_pt "
                    f"(residual {kx[2]:.2e} 1/m); M = ({my[0]:.6g} + "
                    f"{my[1]:.8g} y_pt) kN m (residual {my[2]:.2e} kN m)",
                    f"reading half-width: {k_half:.2e} 1/m, {m_half:.2f} N m",
                    "curvature as plotted (thesis Eq. 5.1); moment as plotted "
                    "(M = F a, a = 0.475 m)",
                ]
            )
            body = "sample,curvature_1_per_m,moment_Nm\n" + "".join(
                f"{i},{k:.7f},{m:.3f}\n" for i, (k, m) in enumerate(data)
            )
            (OUT / f"zeidan2025_fig{fig}_{name}.csv").write_text(
                head + body, encoding="utf-8"
            )
    record["paper_fig17_vs_thesis_fig5_6"] = _fig17_cross_check(
        paper[PAPER_FIG17_PAGE - 1], figures["5_6"]
    )
    defl = _fig18(paper)
    xf, yf = defl["x_fit"], defl["y_fit"]
    assert isinstance(xf, tuple) and isinstance(yf, tuple)
    record["paper_fig18"] = {
        "x_fit": xf,
        "y_fit": yf,
        "pixel_m": defl["pixel_m"],
    }
    rows = []
    for name in SERIES:
        arr = defl[name]
        assert isinstance(arr, np.ndarray)
        rows.extend(f"{name[:2]},{x:.5f},{y:.6f}\n" for x, y in arr)
    head = _header(
        [
            "Zeidan et al. (2026) IJSS 338, 114089, Fig. 18 (page 12): deflection "
            "along the half span, x from the support (0) to mid-span (1.0 m)",
            f"paper PDF sha256 {record['paper_sha256']}",
            "method: native raster pixels, colour classification, median row "
            "per column (measured and model lines not separated; within about 1 mm)",
            f"calibration: x = {xf[0]:.6g} + {xf[1]:.6g} col (residual "
            f"{xf[2]:.1e} m); y = {yf[0]:.6g} + {yf[1]:.6g} row (residual "
            f"{yf[2]:.1e} m)",
        ]
    )
    (OUT / "zeidan2026_fig18_deflection.csv").write_text(
        head + "amplitude_mm,x_m,deflection_m\n" + "".join(rows), encoding="utf-8"
    )
    (OUT / "zeidan2025_digitisation.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
