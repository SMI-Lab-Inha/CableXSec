# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""The synthetic load history is summed by an inverse FFT (walkthrough 5,
finding 1): the same series as the term-by-term sum of cosines, to round-off,
for the same seed, and in a time that grows as n log n, not n²."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from cablexsec.io import synthetic
from cablexsec.io.synthetic import jonswap_series


def _direct(samples: int, amp: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """The former sum, cosine by cosine, on the generator's grid."""
    count = amp.size
    omega = math.pi * np.arange(1, count + 1) / count  # per unit time step
    t = np.arange(samples, dtype=np.float64)
    return np.asarray(
        (amp[None, :] * np.cos(np.outer(t, omega) + phases[None, :])).sum(axis=1)
    )


@pytest.mark.parametrize(
    "options",
    [
        {"duration_s": 600.0, "dt_s": 0.5, "seed": 7},
        {"duration_s": 300.2, "dt_s": 0.2, "seed": 3},  # an odd number of samples
        {"duration_s": 9.0, "dt_s": 1.0, "seed": 1},  # fewer than 32 samples
        {
            "duration_s": 400.0,
            "dt_s": 0.25,
            "seed": 11,
            "dynamic_tension_rms_N": 5e3,
            "tension_phase_deg": 30.0,
            "out_of_plane_fraction": 0.4,
            "mean_tension_N": 8e4,
            "tension_curvature_coupling_N_per_1_m": 1e5,
        },
    ],
)
def test_fft_reproduces_the_sum_of_cosines(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, float]
) -> None:
    fast = jonswap_series(rms_curvature_1_m=0.02, **options)  # type: ignore[arg-type]
    monkeypatch.setattr(synthetic, "_superpose", _direct)
    slow = jonswap_series(rms_curvature_1_m=0.02, **options)  # type: ignore[arg-type]
    for name in ("curvature_x_1_m", "curvature_y_1_m", "tension_N"):
        a, b = getattr(fast, name), getattr(slow, name)
        scale = max(float(np.max(np.abs(b))), 1e-300)
        assert np.max(np.abs(a - b)) <= 1e-10 * scale, name
    np.testing.assert_array_equal(fast.time_s, slow.time_s)


def test_a_three_hour_record_takes_well_under_a_second() -> None:
    start = time.perf_counter()
    series = jonswap_series(
        duration_s=10800.0,
        dt_s=0.2,
        rms_curvature_1_m=0.02,
        dynamic_tension_rms_N=8e3,
        out_of_plane_fraction=0.3,
        seed=1,
    )
    assert time.perf_counter() - start < 1.0
    assert series.time_s.size == 54000
    rms = float(np.sqrt(np.mean(series.curvature_x_1_m**2)))
    assert rms == pytest.approx(0.02, rel=1e-12)
