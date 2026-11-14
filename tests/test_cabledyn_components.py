# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""CableDyn 0.4.0 signed curvature components into the local route.

CableDyn's curvature vector points to the centre of curvature; a fibre at
(x, y) has bending strain -(CurvX x + CurvY y), with x the upper normal at the
hang-off. The local route takes strain-gradient components (g_x, g_y) with
fibre strain g_x x + g_y y.
"""

from __future__ import annotations

import numpy as np
import pytest

from cablexsec.exceptions import CableSchemaError
from cablexsec.io.cabledyn_components import (
    read_cabledyn_station_table,
    series_from_curvature_components,
)

KAPPA = 0.1


def _strain(series, x: float, y: float) -> np.ndarray:  # type: ignore[no-untyped-def]
    return series.curvature_x_1_m * x + series.curvature_y_1_m * y


def _one(cx: float, cy: float, *, right_handed: bool = True):  # type: ignore[no-untyped-def]
    t = np.array([0.0, 0.1, 0.2])
    return series_from_curvature_components(
        t, np.full(3, 2e4), np.full(3, cx), np.full(3, cy), right_handed=right_handed
    )


def test_sag_bend_compresses_the_upper_fibre() -> None:
    # sag bend: the centre of curvature lies above, along the upper normal x
    s = _one(KAPPA, 0.0)
    assert _strain(s, 0.08, 0.0)[0] == pytest.approx(-KAPPA * 0.08)
    assert _strain(s, -0.08, 0.0)[0] == pytest.approx(KAPPA * 0.08)


def test_hog_bend_stretches_the_upper_fibre() -> None:
    s = _one(-KAPPA, 0.0)
    assert _strain(s, 0.08, 0.0)[0] > 0.0


def test_out_of_plane_component_and_handedness() -> None:
    right = _one(0.0, KAPPA)
    left = _one(0.0, KAPPA, right_handed=False)
    assert _strain(right, 0.0, 0.08)[0] == pytest.approx(-KAPPA * 0.08)
    assert _strain(left, 0.0, 0.08)[0] == pytest.approx(KAPPA * 0.08)


def test_rotating_plane_keeps_the_magnitude() -> None:
    phi = np.linspace(0.0, 4 * np.pi, 401)
    s = series_from_curvature_components(
        phi, np.full(phi.size, 3e4), KAPPA * np.cos(phi), KAPPA * np.sin(phi)
    )
    mag = np.hypot(s.curvature_x_1_m, s.curvature_y_1_m)
    assert np.allclose(mag, KAPPA, rtol=0, atol=1e-15)
    assert s.tension_kind == "effective"


@pytest.mark.parametrize("sep", [",", "\t", "  "])
def test_station_table_round_trip(tmp_path, sep: str) -> None:
    t = np.arange(5) * 0.025
    rows = [sep.join(["Time(s)", "Ten(N)", "CurvX(1/m)", "CurvY(1/m)", "PosZ(m)"])]
    rows += [
        sep.join(f"{v:.8g}" for v in (ti, 1e4 + ti, 0.01, -0.02, -5.0)) for ti in t
    ]
    path = tmp_path / "hangoff.txt"
    path.write_text("# station 0.0 m\n" + "\n".join(rows) + "\n", encoding="utf-8")
    s = read_cabledyn_station_table(path)
    assert np.allclose(s.time_s, t)
    assert np.allclose(s.curvature_x_1_m, -0.01)
    assert np.allclose(s.curvature_y_1_m, 0.02)
    assert np.allclose(s.tension_N, 1e4 + t)
    assert s.source == "hangoff.txt"


def test_station_table_missing_column(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("time,Ten,CurvX\n0,1,0\n0.1,1,0\n0.2,1,0\n", encoding="utf-8")
    with pytest.raises(CableSchemaError, match="CurvY"):
        read_cabledyn_station_table(path)
