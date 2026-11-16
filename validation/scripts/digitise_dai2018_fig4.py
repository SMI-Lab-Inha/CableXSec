# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Digitise the measured friction loops of Dai, Sævik and Ye (2018), Fig. 4.

Source: T. Dai, S. Sævik, N. Ye, "An anisotropic friction model in non-bonded
flexible risers", Marine Structures 59 (2018) 423-443, Fig. 4 (p. 429):
panel (a) Test No. 1 (8, 4 and 1 MPa, 0.01 Hz) and panel (c) Test No. 3
(4 MPa, 0.01, 0.05, 0.1 and 0.2 Hz). Panel (b) (clean tape on a steel bar)
is not digitised.

The figure is a raster image embedded in PDF page 7 (JPEG, 1887 x 2063 px,
300 ppi). It is digitised at that native resolution: extract it without
resampling, e.g. ``pdfimages -j -f 7 -l 7 paper.pdf fig4`` (poppler), and
pass the JPEG. Rendering the page only resamples the same raster.

Procedure (reproducible; the CSV headers record the numbers):

1. Axis calibration. Each panel has a left and a bottom axis line with tick
   marks. The tick-mark centroids are found on a 10 px band inside the
   bottom axis (x) and a 4 px band right of the left axis (y); centroids
   within 4 px of the regular spacing between the end ticks are kept, and a
   straight line is fitted to (pixel, label) by least squares per axis.
2. Point picking by colour. Pixels inside the plot frame with a chroma
   (max - min of RGB) above 50 and a saturation above 0.35 are classified by
   hue (red, green, cyan, blue); the legend handles are masked.
3. Branch traces. In every pixel column the series' pixels above the loop
   centre form the upper branch (sliding in +x, force positive) and those
   below it the lower branch; the branch value is their median and the
   extent is their spread (line width plus the plotted noise band). Columns
   whose pixels span more than 0.03 in y are near-vertical stick segments
   and are dropped from the branches.
4. Stick segments. In every pixel row inside the middle half between the
   two branch levels (the elastic part of a reversal) the median x of the
   pixels left (right) of x = 0 is the stick segment after the left (right)
   reversal.

Reading resolution (half-width, not experimental uncertainty): the largest
calibration residual plus one pixel, per axis and panel.

Usage::

    python validation/scripts/digitise_dai2018_fig4.py fig4-000.jpg

writes ``validation/paper_data/dai2018_fig4a_test1.csv`` and
``dai2018_fig4c_test3.csv``.
"""

from __future__ import annotations

import hashlib
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation" / "paper_data"

PDF_SHA256 = "b5f8d199452fe73fb5c6929cd96c25cb6c0c0de79daca42ef2f74abe74ed2fe1"
IMAGE_SHA256 = "616a05a2db2d9d914572160179aa7c3288caf4f19d257e0900323321b01b8a5a"

#: Largest y extent of a branch column; wider columns are stick segments.
BRANCH_EXTENT = 0.03


@dataclass(frozen=True)
class Panel:
    """Pixel geometry of one panel of the figure."""

    key: str
    spine_x: int  # column of the left axis line
    spine_y: int  # row of the bottom axis line
    x_band: tuple[int, int]  # columns searched for x ticks
    y_band: tuple[int, int]  # rows searched for y ticks
    x_labels: tuple[float, ...]  # first label = first tick found
    y_labels: tuple[float, ...]
    x_last: float  # label of the last tick found
    y_last: float
    frame: tuple[int, int, int, int]  # x0, x1, y0, y1 of the plot area
    masked: tuple[tuple[int, int, int, int], ...]  # legend handles
    series: tuple[tuple[str, str], ...]  # (label, colour)


PANELS = (
    Panel(
        "a",
        109,
        850,
        (100, 1000),
        (0, 850),
        (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0),
        (0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4),
        2.0,
        -0.4,
        (111, 935, 5, 846),
        ((555, 625, 30, 125),),
        (("8 MPa", "red"), ("4 MPa", "green"), ("1 MPa", "blue")),
    ),
    Panel(
        "c",
        588,
        1923,
        (580, 1500),
        (1050, 1924),
        (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5),
        (0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3),
        2.0,
        -0.3,
        (590, 1410, 1070, 1920),
        ((1100, 1175, 1100, 1225),),
        (
            ("0.01 Hz", "red"),
            ("0.05 Hz", "green"),
            ("0.1 Hz", "blue"),
            ("0.2 Hz", "cyan"),
        ),
    ),
)

HUES = {
    "red": (340.0, 20.0),
    "green": (90.0, 150.0),
    "cyan": (165.0, 200.0),
    "blue": (215.0, 265.0),
}


def _runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(int), [0]))))
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def _centroids(profile: NDArray[np.float64], threshold: float) -> list[float]:
    out = []
    for a, b in _runs(profile > threshold):
        w = profile[a : b + 1]
        out.append(float((np.arange(a, b + 1) * w).sum() / w.sum()))
    return out


def _matched(
    found: list[float], labels: tuple[float, ...], last: float
) -> list[tuple[float, float]]:
    """Tick centroids within 4 px of the regular spacing between the first
    and the last centroid found (labels ``labels[0]`` and ``last``)."""
    p0, v0, p1, v1 = found[0], labels[0], found[-1], last
    points = []
    for value in labels:
        predicted = p0 + (value - v0) * (p1 - p0) / (v1 - v0)
        near = [f for f in found if abs(f - predicted) <= 4.0]
        if len(near) == 1:
            points.append((near[0], value))
    return points


def _calibration(
    gray: NDArray[np.float64], panel: Panel
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, object]]:
    xs, ys = panel.spine_x, panel.spine_y
    band = 255.0 - gray[ys - 12 : ys - 2, panel.x_band[0] : panel.x_band[1]]
    x_ticks = [panel.x_band[0] + c for c in _centroids(band.mean(axis=0), 120.0)]
    band = 255.0 - gray[panel.y_band[0] : panel.y_band[1], xs + 3 : xs + 7]
    y_ticks = [panel.y_band[0] + c for c in _centroids(band.mean(axis=1), 150.0)]
    # The end ticks anchor the spacing; curves crossing the band near the
    # left axis add spurious centroids, which the spacing test rejects.
    x_points = _matched(x_ticks, panel.x_labels, panel.x_last)
    y_points = _matched(y_ticks, panel.y_labels, panel.y_last)
    fx = np.polyfit([p for p, _ in x_points], [v for _, v in x_points], 1)
    fy = np.polyfit([p for p, _ in y_points], [v for _, v in y_points], 1)
    rx = max(abs(v - np.polyval(fx, p)) for p, v in x_points)
    ry = max(abs(v - np.polyval(fy, p)) for p, v in y_points)
    info: dict[str, object] = {
        "x_points": x_points,
        "y_points": y_points,
        "x_residual": float(rx),
        "y_residual": float(ry),
        "x_resolution": float(rx + abs(fx[0])),
        "y_resolution": float(ry + abs(fy[0])),
    }
    return fx, fy, info


def _hue(rgb: NDArray[np.float64]) -> tuple[NDArray[np.float64], ...]:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    top = rgb.max(axis=-1)
    chroma = top - rgb.min(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        hr = (60.0 * (g - b) / chroma) % 360.0
        hg = 60.0 * (b - r) / chroma + 120.0
        hb = 60.0 * (r - g) / chroma + 240.0
        saturation = np.where(top > 0, chroma / np.maximum(top, 1.0), 0.0)
    hue = np.where(top == r, hr, np.where(top == g, hg, hb))
    return hue, chroma, saturation


def _in_hue(hue: NDArray[np.float64], colour: str) -> NDArray[np.bool_]:
    lo, hi = HUES[colour]
    if lo > hi:
        return np.asarray((hue > lo) | (hue < hi))
    return np.asarray((hue > lo) & (hue < hi))


def digitise(image_path: Path) -> list[tuple[Panel, dict[str, object], list[str]]]:
    from PIL import Image  # noqa: PLC0415  (matplotlib dependency)

    raw = image_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != IMAGE_SHA256:
        raise SystemExit(
            f"unexpected image (SHA-256 {digest}); expected {IMAGE_SHA256}"
        )
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
    gray = rgb.mean(axis=2)
    hue, chroma, saturation = _hue(rgb)
    results = []
    for panel in PANELS:
        fx, fy, info = _calibration(gray, panel)
        x0, x1, y0, y1 = panel.frame
        region = np.zeros(gray.shape, dtype=bool)
        region[y0:y1, x0:x1] = True
        for mx0, mx1, my0, my1 in panel.masked:
            region[my0:my1, mx0:mx1] = False
        coloured = region & (chroma > 50.0) & (saturation > 0.35)
        rows_out: list[str] = []
        for label, colour in panel.series:
            rows, cols = np.nonzero(coloured & _in_hue(hue, colour))
            x = np.polyval(fx, cols)
            y = np.polyval(fy, rows)
            centre = 0.5 * (np.percentile(y, 0.5) + np.percentile(y, 99.5))
            upper: list[tuple[float, float, float]] = []
            lower: list[tuple[float, float, float]] = []
            for col in np.unique(cols):
                values = y[cols == col]
                for branch, sel in ((upper, values > centre), (lower, values < centre)):
                    group = values[sel]
                    extent = float(np.ptp(group)) if group.size else 0.0
                    if group.size >= 2 and extent <= BRANCH_EXTENT:
                        xc = float(np.polyval(fx, col))
                        branch.append((xc, float(np.median(group)), extent))
            level_hi = float(np.median([v for _, v, _ in upper]))
            level_lo = float(np.median([v for _, v, _ in lower]))
            band_lo = level_lo + 0.25 * (level_hi - level_lo)
            band_hi = level_hi - 0.25 * (level_hi - level_lo)
            left: list[tuple[float, float, float]] = []
            right: list[tuple[float, float, float]] = []
            for row in np.unique(rows):
                yr = float(np.polyval(fy, row))
                if not band_lo <= yr <= band_hi:
                    continue
                values = x[rows == row]
                for stick, sel in ((left, values < 0.0), (right, values > 0.0)):
                    group = values[sel]
                    if group.size:
                        stick.append(
                            (float(np.median(group)), yr, float(np.ptp(group)))
                        )
            # Stroke order: +x stroke = rise after the left reversal, then the
            # upper branch; -x stroke = fall after the right reversal, then the
            # lower branch.
            for segment, points in (
                ("stick_left", sorted(left, key=lambda p: p[1])),
                ("upper", sorted(upper)),
                ("stick_right", sorted(right, key=lambda p: -p[1])),
                ("lower", sorted(lower, reverse=True)),
            ):
                for xv, yv, extent in points:
                    rows_out.append(f"{label},{segment},{xv:.4f},{yv:.4f},{extent:.4f}")
        results.append((panel, info, rows_out))
    return results


_HEADERS = {
    "a": (
        "# Dai, Saevik and Ye (2018) Marine Structures 59:423-443, Fig. 4(a) (p. 429), Test No. 1:",
        "# measured friction loops of a tensile armour wire between two anti-wear tapes (in-service",
        "# tape outer surface, oil and rust spots, about 50 % contact), contact pressure 8, 4 and 1 MPa,",
        "# 0.01 Hz, displacement amplitude 2 mm (Sec. 2.3, p. 425; Sec. 4.1, p. 428). Printed values",
        "# (p. 428): mu_s 0.22 / 0.24 / 0.38 and mu_d 0.18 / 0.20 / 0.32 at 8 / 4 / 1 MPa, selected as",
        "# the maximum (in the stick domain) and the minimum of Fig. 4a.",
    ),
    "c": (
        "# Dai, Saevik and Ye (2018) Marine Structures 59:423-443, Fig. 4(c) (p. 429), Test No. 3:",
        "# measured friction loops of a tensile armour wire in the groove track of the in-service tape",
        "# inner surface, contact pressure 4 MPa, frequency 0.01, 0.05, 0.1 and 0.2 Hz, displacement",
        "# amplitude 2 mm (Sec. 4.3, p. 430). No coefficients are printed for this test.",
    ),
}


def _write(panel: Panel, info: dict[str, object], rows: list[str]) -> Path:
    name = {"a": "dai2018_fig4a_test1.csv", "c": "dai2018_fig4c_test3.csv"}[panel.key]
    xp = "; ".join(f"{p:.2f}px={v:g}mm" for p, v in info["x_points"])  # type: ignore[attr-defined]
    yp = "; ".join(f"{p:.2f}px={v:g}" for p, v in info["y_points"])  # type: ignore[attr-defined]
    x_text = f"x calibration (least squares on tick centroids, image pixels): {xp}; "
    x_text += f"largest residual {info['x_residual']:.4f} mm."
    y_text = f"y calibration: {yp}; largest residual {info['y_residual']:.4f}."
    header = [
        *_HEADERS[panel.key],
        "# y = tangential force / (2 x contact force): two contact interfaces (p. 428).",
        f"# Source PDF SHA-256: {PDF_SHA256}",
        "# Digitised from the raster embedded in PDF page 7 at its native resolution (JPEG",
        "# 1887 x 2063 px, 300 ppi, SHA-256",
        f"# {IMAGE_SHA256}) by",
        "# validation/scripts/digitise_dai2018_fig4.py (procedure in its docstring).",
        *("# " + line for line in textwrap.wrap(x_text, 92)),
        *("# " + line for line in textwrap.wrap(y_text, 92)),
        "# Reading resolution (half-width, NOT experimental uncertainty): x +-"
        f"{info['x_resolution']:.3f} mm, y +-{info['y_resolution']:.4f}",
        "#   (largest calibration residual plus one pixel).",
        "# segment: stick_left = elastic rise after the left reversal (one point per pixel row in the",
        "# middle half between the branch levels; extent = x spread of the row [mm]); upper = sliding",
        "# in +x (one point per pixel column, median of the line; extent = y spread of the column:",
        "# line width plus plotted noise); stick_right and lower = the same for the -x stroke. Rows are",
        "# in stroke order. Columns spanning more than 0.03 in y (near-vertical) are not in the",
        "# branches. Hidden parts of overlapping curves are missing, not interpolated.",
        "# The end-of-stroke rises before each reversal, which the paper attributes to the wire's",
        "# out-of-straightness and surface irregularities (pp. 428, 430), are kept as plotted;",
        "# the comparison (validation/scripts/check_dai2018_friction.py) excludes them.",
        "series,segment,x_mm,y,extent",
    ]
    path = OUT / name
    path.write_text("\n".join([*header, *rows]) + "\n", encoding="utf-8")
    return path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    for panel, info, rows in digitise(Path(sys.argv[1])):
        print(
            f"wrote {_write(panel, info, rows).relative_to(ROOT)} ({len(rows)} points)"
        )


if __name__ == "__main__":
    main()
