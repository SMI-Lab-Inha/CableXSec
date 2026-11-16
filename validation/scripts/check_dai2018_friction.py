# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Static/dynamic friction law against measured wire-on-tape friction loops.

Data: Dai, Sævik and Ye, Marine Structures 59 (2018) 423-443, Fig. 4(a) and
4(c) (p. 429), digitised in ``validation/paper_data/dai2018_fig4a_test1.csv``
and ``dai2018_fig4c_test3.csv`` (procedure: ``digitise_dai2018_fig4.py``).
The y axis is the friction force over twice the contact force (two
interfaces), so one spring-slider with the normalised bound ``b = mu_d``
reproduces a loop in the plotted units.

Model: :func:`cablexsec.mechanics.friction_law.slider_history` driven by a
sinusoid between the measured reversal displacements, three cycles from a
stress-free start at the left reversal (the third compared), with
``FrictionLaw(static_factor = mu_s / mu_d, stick_time_s = 0)``, no rate
terms, and the stick stiffness of the loop's digitised elastic segments (the
mean of its two reversals; a secant over the middle half of each elastic
rise). Test 1 takes ``mu_s`` and ``mu_d`` as printed (p. 428). Test 3 prints
none: the 0.01 Hz loop gives them (its digitised peak and dynamic level,
mean of both strokes), held for 0.05-0.2 Hz, so that the higher frequencies
test the rate-independent law.

Measures, for each stroke and identically for model and measurement
(``d`` = travel since the reversal):

- peak: largest ``|y|`` of the branch at ``d <= 0.6 mm``, and its ``d``;
- dynamic level: smallest 0.1 mm running mean of ``|y|`` after the peak,
  before the end-of-stroke zone (the paper's "minimum value");
- decay length: travel from the peak to the first point whose running mean
  is within 10 % of ``peak - level`` above the level;
- stroke work: ``int y dx`` along the stroke from the force at which the
  digitised elastic segment starts (a quarter of the way between the branch
  levels) to the end-of-stroke zone.

End-of-stroke zone: the last ``END_ZONE_MM`` = 0.5 mm before each reversal,
where the paper attributes the rise of the friction force to the wire's
out-of-straightness (Test 1, p. 428) and surface irregularities (Test 3,
p. 430). It is excluded from every measure.

Usage: ``python validation/scripts/check_dai2018_friction.py``.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cablexsec.mechanics.friction_law import FrictionLaw, slider_history

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "validation" / "paper_data"
TEST1_CSV = "dai2018_fig4a_test1.csv"
TEST3_CSV = "dai2018_fig4c_test3.csv"

FloatArray = NDArray[np.float64]

#: Excluded travel before each reversal [mm] (end-of-stroke rises).
END_ZONE_MM = 0.5
#: Travel after a reversal searched for the breakaway peak [mm].
PEAK_ZONE_MM = 0.6
#: Running-mean window of the dynamic level [mm].
LEVEL_WINDOW_MM = 0.1
#: Share of ``peak - level`` that ends the decay.
DECAY_SHARE = 0.1
#: Samples per model cycle.
SAMPLES = 20_000
#: A re-stick peak exceeds the dynamic level by this many reading resolutions.
PEAK_RESOLUTIONS = 5.0
#: Printed precision of the coefficients (two decimals, p. 428).
PRINTED_PRECISION = 0.005

#: Printed coefficients of Test No. 1 (Sec. 4.1, p. 428): (mu_s, mu_d).
TEST1_PRINTED = {"8 MPa": (0.22, 0.18), "4 MPa": (0.24, 0.20), "1 MPa": (0.38, 0.32)}
TEST3_REFERENCE = "0.01 Hz"


@dataclass(frozen=True)
class Stroke:
    """One stroke: polyline in stroke order and its sliding branch."""

    x: FloatArray  # stroke order (elastic segment, then branch) [mm]
    y: FloatArray
    branch_x: FloatArray
    branch_y: FloatArray
    start_mm: float  # reversal displacement at the start
    end_mm: float  # reversal displacement at the end
    sign: float  # +1 for the +x stroke

    def travel(self, x: FloatArray) -> FloatArray:
        return np.asarray(self.sign * (x - self.start_mm))

    @property
    def length_mm(self) -> float:
        return abs(self.end_mm - self.start_mm)


@dataclass(frozen=True)
class StrokeMeasures:
    peak: float
    peak_travel_mm: float
    level: float
    decay_mm: float
    work: float

    @property
    def excess(self) -> float:
        return self.peak - self.level


@dataclass(frozen=True)
class Loop:
    label: str
    strokes: tuple[Stroke, Stroke]
    stick_slope_per_mm: tuple[float, float]
    band: tuple[float, float]  # y band of the digitised elastic segments


def read_loops(name: str) -> dict[str, Loop]:
    """The digitised loops of one CSV, keyed by series label."""
    rows: dict[str, dict[str, list[tuple[float, float]]]] = {}
    with (DATA / name).open(encoding="utf-8") as handle:
        lines = (line for line in handle if not line.startswith("#"))
        for row in csv.DictReader(lines):
            rows.setdefault(row["series"], {}).setdefault(row["segment"], []).append(
                (float(row["x_mm"]), float(row["y"]))
            )
    loops = {}
    for label, segments in rows.items():
        parts = {key: np.array(value) for key, value in segments.items()}
        every = np.concatenate(list(parts.values()))
        left, right = float(every[:, 0].min()), float(every[:, 0].max())
        # x = a + y / k by least squares (x is the measured coordinate of a
        # pixel row of the elastic segment).
        slopes = tuple(
            1.0 / float(np.polyfit(parts[key][:, 1], parts[key][:, 0], 1)[0])
            for key in ("stick_left", "stick_right")
        )
        band = (
            float(parts["stick_left"][:, 1].min()),
            float(parts["stick_left"][:, 1].max()),
        )
        up = np.concatenate([parts["stick_left"], parts["upper"]])
        down = np.concatenate([parts["stick_right"], parts["lower"]])
        strokes = (
            Stroke(up[:, 0], up[:, 1], *parts["upper"].T, left, right, 1.0),
            Stroke(down[:, 0], down[:, 1], *parts["lower"].T, right, left, -1.0),
        )
        loops[label] = Loop(label, strokes, (slopes[0], slopes[1]), band)
    return loops


def measure(stroke: Stroke) -> StrokeMeasures:
    """Peak, dynamic level, decay length and work of one stroke."""
    d = stroke.travel(stroke.branch_x)
    order = np.argsort(d, kind="stable")
    d, mag = d[order], np.abs(stroke.branch_y)[order]
    usable = d <= stroke.length_mm - END_ZONE_MM
    zone = np.flatnonzero(usable & (d <= PEAK_ZONE_MM))
    i_peak = int(zone[np.argmax(mag[zone])])
    peak, d_peak = float(mag[i_peak]), float(d[i_peak])
    after = usable & (d >= d_peak)
    da, ma = d[after], mag[after]
    # Running mean over a travel window (columns are uneven where curves
    # hide each other).
    smooth = np.array(
        [ma[np.abs(da - centre) <= 0.5 * LEVEL_WINDOW_MM].mean() for centre in da]
    )
    level = float(smooth.min())
    below = np.flatnonzero(smooth <= level + DECAY_SHARE * (peak - level))
    decay = float(da[below[0]] - d_peak)
    keep = stroke.travel(stroke.x) <= stroke.length_mm - END_ZONE_MM
    x, y = stroke.x[keep], stroke.y[keep]
    work = float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))
    return StrokeMeasures(peak, d_peak, level, decay, work)


def model_strokes(
    loop: Loop, mu_s: float, mu_d: float, *, stick_time_s: float = 0.0
) -> tuple[Stroke, Stroke]:
    """The slider's third cycle as two strokes in the digitised form.

    The drive runs between the loop's reversal displacements from a
    stress-free start at the left one; the elastic part of each stroke
    starts where the force enters the band of the digitised elastic segments.
    """
    left, right = loop.strokes[0].start_mm, loop.strokes[0].end_mm
    amplitude = 0.5 * (right - left)
    phase = np.linspace(0.0, 3.0, 3 * SAMPLES + 1)
    travel_mm = amplitude * (1.0 - np.cos(2.0 * np.pi * phase))
    stiffness = float(np.mean(loop.stick_slope_per_mm)) * 1.0e3  # per metre
    law = FrictionLaw(static_factor=mu_s / mu_d, stick_time_s=stick_time_s)
    history = slider_history(travel_mm * 1.0e-3, mu_d, stiffness, friction_law=law)
    last = slice(2 * SAMPLES, 3 * SAMPLES + 1)
    x, f = left + travel_mm[last], history.force_N[last]
    half = SAMPLES // 2
    lo, hi = loop.band
    strokes = []
    for part, sign, start, end in (
        (slice(0, half + 1), 1.0, left, right),
        (slice(half, SAMPLES + 1), -1.0, right, left),
    ):
        xs, fs = x[part], f[part]
        entry = int(np.flatnonzero(fs >= lo if sign > 0 else fs <= hi)[0])
        xs, fs = xs[entry:], fs[entry:]
        inside = fs < hi if sign > 0 else fs > lo
        branch = int(np.argmin(inside))  # first point beyond the band
        strokes.append(Stroke(xs, fs, xs[branch:], fs[branch:], start, end, sign))
    return strokes[0], strokes[1]


def reading_resolution(name: str) -> float:
    """The y reading resolution recorded in a CSV header."""
    with (DATA / name).open(encoding="utf-8") as handle:
        for line in handle:
            if "Reading resolution" in line:
                return float(line.rsplit("y +-", 1)[1].split()[0])
    raise ValueError(f"no reading resolution in {name}")


def _relative(model: float, measured: float) -> float:
    return 100.0 * (model / measured - 1.0)


def compare() -> dict[str, Any]:
    """Every measure of the DAI18 rows (doc/benchmarks.md)."""
    test1, test3 = read_loops(TEST1_CSV), read_loops(TEST3_CSV)
    res1, res3 = reading_resolution(TEST1_CSV), reading_resolution(TEST3_CSV)
    reference = [measure(s) for s in test3[TEST3_REFERENCE].strokes]
    peak3 = float(np.mean([m.peak for m in reference]))
    level3 = float(np.mean([m.level for m in reference]))
    cases = []
    for test, loops, resolution in ((1, test1, res1), (3, test3, res3)):
        for label, loop in loops.items():
            mu_s, mu_d = TEST1_PRINTED[label] if test == 1 else (peak3, level3)
            measured = [measure(s) for s in loop.strokes]
            model = [measure(s) for s in model_strokes(loop, mu_s, mu_d)]
            never = [
                measure(s)
                for s in model_strokes(loop, mu_s, mu_d, stick_time_s=math.inf)
            ]
            cases.append(
                {
                    "test": test,
                    "label": label,
                    "mu_s": mu_s,
                    "mu_d": mu_d,
                    "stiffness_per_mm": float(np.mean(loop.stick_slope_per_mm)),
                    "resolution": resolution,
                    "measured": measured,
                    "model": model,
                    "never_heals": never,
                    "peak_error_pct": [
                        _relative(b.peak, a.peak)
                        for a, b in zip(measured, model, strict=True)
                    ],
                    "level_error_pct": [
                        _relative(b.level, a.level)
                        for a, b in zip(measured, model, strict=True)
                    ],
                    "work_error_pct": _relative(
                        sum(m.work for m in model), sum(m.work for m in measured)
                    ),
                }
            )

    def peaks(key: str) -> int:
        return sum(
            m.excess > PEAK_RESOLUTIONS * c["resolution"] for c in cases for m in c[key]
        )

    excess = [m.excess for c in cases for m in c["measured"]]
    # Digitisation against the printed coefficients: the loop maximum is the
    # printed mu_s (p. 428).
    maximum = {
        label: max(measure(s).peak for s in loop.strokes)
        for label, loop in test1.items()
    }
    peaks3 = {
        label: float(np.mean([measure(s).peak for s in loop.strokes]))
        for label, loop in test3.items()
    }
    levels3 = {
        label: float(np.mean([measure(s).level for s in loop.strokes]))
        for label, loop in test3.items()
    }
    return {
        "cases": cases,
        "strokes": 2 * len(cases),
        "measured_peaks": peaks("measured"),
        "model_peaks": peaks("model"),
        "never_heals_peaks": peaks("never_heals"),
        "excess_range": (min(excess), max(excess)),
        "digitised_maximum": maximum,
        "digitisation_error": max(
            abs(maximum[k] - TEST1_PRINTED[k][0]) for k in maximum
        ),
        "digitisation_tolerance": PRINTED_PRECISION + res1,
        "test3_reference": (peak3, level3),
        "test3_peak_rise_pct": 100.0
        * (peaks3["0.2 Hz"] / peaks3[TEST3_REFERENCE] - 1.0),
        "test3_level_change_pct": [
            100.0 * (levels3[k] / levels3[TEST3_REFERENCE] - 1.0)
            for k in levels3
            if k != TEST3_REFERENCE
        ],
    }


def main() -> None:
    r = compare()
    n = r["strokes"]
    print(
        f"re-sticking: measured {r['measured_peaks']}/{n} strokes, "
        f"t_stick = 0 {r['model_peaks']}/{n}, t_stick = inf {r['never_heals_peaks']}/{n}"
    )
    print(f"digitised maxima {r['digitised_maximum']} vs printed mu_s")
    for c in r["cases"]:
        print(
            f"Test {c['test']} {c['label']:8s} mu_s {c['mu_s']:.4f} "
            f"mu_d {c['mu_d']:.4f} k {c['stiffness_per_mm']:.2f}/mm"
        )
        for name in ("measured", "model"):
            for sign, m in zip("+-", c[name], strict=True):
                print(
                    f"   {name:8s} {sign}x peak {m.peak:.4f} at "
                    f"{m.peak_travel_mm:.3f} mm, level {m.level:.4f}, "
                    f"decay {m.decay_mm:.3f} mm, work {m.work:.4f}"
                )
        pe, le = c["peak_error_pct"], c["level_error_pct"]
        print(
            f"   errors: peak {pe[0]:+.1f}/{pe[1]:+.1f} %, level "
            f"{le[0]:+.1f}/{le[1]:+.1f} %, work {c['work_error_pct']:+.1f} %"
        )
    changes = ", ".join(f"{v:+.1f}" for v in r["test3_level_change_pct"])
    print(
        f"Test 3: peak 0.2 Hz vs 0.01 Hz {r['test3_peak_rise_pct']:+.1f} %; "
        f"level changes {changes} %"
    )


if __name__ == "__main__":
    main()
