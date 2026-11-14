# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Station histories with signed curvature components from CableDyn 0.4.0.

CableDyn 0.4.0 writes, per station, the effective tension ``Ten`` and the
curvature vector resolved on two section axes, ``CurvX`` and ``CurvY`` [1/m].
The vector points to the centre of curvature (:math:`\mathrm{d}\mathbf{t}/
\mathrm{d}s`), so a fibre at :math:`(x, y)` has the bending strain
:math:`-(\mathrm{CurvX}\,x + \mathrm{CurvY}\,y)`. CableXSec takes the
axial-strain-gradient components :math:`(g_x, g_y)` with fibre strain
:math:`g_x x + g_y y` on the same axes, hence

.. math:: g_x = -\mathrm{CurvX}, \qquad g_y = -\mathrm{CurvY}.

The section axes are CableDyn's: :math:`x` is the upper normal at the hang-off,
fixed in the hang-off axes and carried along the line by parallel transport;
:math:`(x, y, \mathbf{t})` is right-handed with :math:`\mathbf{t}` from the
hang-off (End A) towards the anchor. Wire azimuths of the local analysis are
then measured from CableDyn's :math:`x` axis. A left-handed triad is accepted
with ``right_handed=False``, which reverses :math:`y` so that the lay
direction of each armour layer keeps its meaning.

This route sits beside :func:`~cablexsec.io.cabledyn.read_cabledyn_sections`,
which rebuilds the curvature vector of the v0.1 output from node tangents; it
does not change that reader.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from cablexsec.exceptions import CableSchemaError
from cablexsec.io.series import LoadCaseSeries
from cablexsec.provenance import Provenance

__all__ = ["read_cabledyn_station_table", "series_from_curvature_components"]

_TIME = ("time_s", "time(s)", "time", "t")
_TEN = ("ten", "ten_n", "ten(n)", "effective_tension_n")
_CURV_X = ("curvx", "curvx_1_m", "curvx(1/m)")
_CURV_Y = ("curvy", "curvy_1_m", "curvy(1/m)")


def series_from_curvature_components(
    time_s: ArrayLike,
    effective_tension_N: ArrayLike,
    curv_x_1_m: ArrayLike,
    curv_y_1_m: ArrayLike,
    *,
    right_handed: bool = True,
    source: str = "CableDyn 0.4.0 station history",
) -> LoadCaseSeries:
    """A :class:`LoadCaseSeries` from CableDyn's signed curvature components.

    Parameters
    ----------
    time_s, effective_tension_N : array_like
        Sample times [s] (uniform step) and effective tension [N].
    curv_x_1_m, curv_y_1_m : array_like
        ``CurvX`` and ``CurvY``: components of the curvature vector (towards
        the centre of curvature) on CableDyn's section axes [1/m].
    right_handed : bool, optional
        ``True`` (default) when ``x × y = t`` with ``t`` from End A to End B.
        ``False`` reverses ``y``.
    source : str, optional
        Provenance label (file name, record, station).
    """
    cx_ = np.asarray(curv_x_1_m, dtype=np.float64)
    cy_ = np.asarray(curv_y_1_m, dtype=np.float64)
    sign_y = 1.0 if right_handed else -1.0
    limitations = [
        "section axes as CableDyn defines them (x the upper normal at End A, "
        "parallel transport); "
        "the right-handedness of (x, y, t) is "
        + (
            "declared by the caller"
            if right_handed
            else "declared left-handed: y reversed"
        ),
        "effective tension: for a sealed armour annulus convert to true tension "
        "before local recovery",
    ]
    return LoadCaseSeries(
        time_s=np.asarray(time_s, dtype=np.float64),
        curvature_x_1_m=-cx_,
        curvature_y_1_m=-sign_y * cy_,
        tension_N=np.asarray(effective_tension_N, dtype=np.float64),
        tension_kind="effective",
        source=source,
        provenance=Provenance(
            method=(
                "CableDyn 0.4.0 curvature components to strain gradient: "
                "g = -(CurvX, CurvY)"
            ),
            citations=(),
            gates=(
                "sag, hog and out-of-plane signs (tests/test_cabledyn_components.py)",
            ),
            limitations=tuple(limitations),
        ),
    )


def _column(fields: Sequence[str], names: Sequence[str], what: str, path: Path) -> str:
    lowered = {f.strip().lower(): f for f in fields}
    hits = [lowered[n] for n in names if n in lowered]
    if len(hits) != 1:
        raise CableSchemaError(
            f"{path.name}: expected one {what} column ({', '.join(names)}), "
            f"found {len(hits)} in {list(fields)}"
        )
    return hits[0]


def read_cabledyn_station_table(
    path: str | Path, *, right_handed: bool = True
) -> LoadCaseSeries:
    """Read one station table written from CableDyn 0.4.0.

    The columns are time, ``Ten``, ``CurvX`` and ``CurvY``.
    The table is comma, tab or whitespace separated with one header row;
    column names are matched without case (``time``/``Time(s)``/``time_s``,
    ``Ten``, ``CurvX``, ``CurvY``, with or without units written without a
    space). Blank lines are skipped and lines starting with ``#`` are comments.
    Every data row must have as many columns as the header. Other columns are
    ignored.
    """
    p = Path(path)
    numbered = [
        (number, ln)
        for number, ln in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if ln.strip() and not ln.startswith("#")
    ]
    if len(numbered) < 3:
        raise CableSchemaError(f"{p.name}: needs a header row and at least two samples")
    lines = [ln for _, ln in numbered]
    if "," in lines[0]:
        rows = list(csv.reader(lines))
    elif "\t" in lines[0]:
        rows = [ln.split("\t") for ln in lines]
    else:
        rows = [ln.split() for ln in lines]
    header = [h.strip() for h in rows[0]]
    for (number, _), row in zip(numbered[1:], rows[1:], strict=True):
        if len(row) != len(header):
            raise CableSchemaError(
                f"{p.name}: line {number} has {len(row)} columns but the header "
                f"has {len(header)} ({header}); in a whitespace-separated "
                "table write units without a space, e.g. Ten(N)"
            )
    cols = {
        "time": _column(header, _TIME, "time", p),
        "ten": _column(header, _TEN, "Ten", p),
        "cx": _column(header, _CURV_X, "CurvX", p),
        "cy": _column(header, _CURV_Y, "CurvY", p),
    }
    index = {k: header.index(v) for k, v in cols.items()}
    try:
        data = np.array(
            [[float(r[i]) for i in index.values()] for r in rows[1:]], dtype=np.float64
        )
    except ValueError as error:
        raise CableSchemaError(f"{p.name}: non-numeric data row") from error
    return series_from_curvature_components(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        data[:, 3],
        right_handed=right_handed,
        source=p.name,
    )
