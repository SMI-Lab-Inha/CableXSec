# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Time-history CSV reader.

Reads a **time-history CSV export** of a global analysis (a header row of
named columns followed by one row per time sample) into a
:class:`~cablexsec.io.series.LoadCaseSeries`. Columns are selected by
case-insensitive header substrings, with unit scales to ``1/m`` and ``N``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from cablexsec.exceptions import CableSchemaError
from cablexsec.io.series import LoadCaseSeries
from cablexsec.mechanics.section_conventions import rotation_to_strain_curvature
from cablexsec.provenance import Provenance

FloatArray = NDArray[np.float64]

__all__ = ["read_time_history_csv"]


def read_time_history_csv(
    path: str | Path,
    *,
    time_col: str = "Time",
    curvature_x_col: str = "x curvature",
    curvature_y_col: str | None = "y curvature",
    tension_col: str = "Wall tension",
    temperature_col: str | None = None,
    axial_strain_col: str | None = None,
    curvature_convention: Literal["rotation", "strain_gradient"] = "rotation",
    tension_kind: Literal["true", "effective", "unspecified"] | None = None,
    axial_strain_kind: Literal["direct", "total", "unspecified"] | None = None,
    curvature_scale_to_1_m: float = 1.0,
    tension_scale_to_N: float = 1.0e3,
    axial_strain_scale: float = 0.01,
) -> LoadCaseSeries:
    r"""Read a time-history CSV export into a :class:`LoadCaseSeries`.

    Column names are matched **case-insensitively** and a header cell matches if
    the requested name is a unique substring (exports often append units and
    object names, e.g. ``"Effective Tension (kN)"``).

    Parameters
    ----------
    path : str or pathlib.Path
        Time-history CSV export.
    time_col, curvature_x_col, tension_col : str
        Header substrings for the required columns.
    curvature_y_col : str or None, optional
        Second signed curvature component. ``None`` explicitly declares the
        omitted input component zero. A requested but absent column is an error.
    curvature_convention : str, optional
        Native right-handed rotation components (default) are mapped to the
        strain gradient ``(-y curvature, x curvature)``, fibre strain
        ``k_x*y - k_y*x``. Use ``strain_gradient`` only for already
        transformed, signed input. Curvature magnitude cannot retain reversals.
    tension_kind : str or None, optional
        Inferred from recognised wall/true/effective-tension headings. A custom
        heading needs an explicit kind before fatigue. An override cannot
        contradict a recognised native heading.
    axial_strain_col, axial_strain_kind : str or None, optional
        Optional strain column. Direct/total native headings retain their kind;
        custom headings need an explicit kind before fatigue.
    temperature_col : str or None, optional
        Header substring for a temperature column; ``None`` → isothermal.
    curvature_scale_to_1_m : float, optional
        Multiplier from the file's curvature unit to ``1/m`` (1.0 for rad/m;
        use e.g. ``1e3`` for 1/mm).
    tension_scale_to_N : float, optional
        Multiplier from the file's tension unit to N (default ``1e3`` for kN).
    axial_strain_scale : float, optional
        Multiplier to dimensionless strain (default 0.01 for native percent).

    Returns
    -------
    LoadCaseSeries
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"time-history CSV not found: {p}")
    if curvature_convention not in ("rotation", "strain_gradient"):
        raise CableSchemaError("declare rotation or strain_gradient curvature")
    if curvature_convention == "rotation" and any(
        name is not None and name.strip().casefold() == "curvature"
        for name in (curvature_x_col, curvature_y_col)
    ):
        raise CableSchemaError("export signed x/y curvature, not curvature magnitude")
    if any(
        not np.isfinite(scale) or scale <= 0
        for scale in (curvature_scale_to_1_m, tension_scale_to_N, axial_strain_scale)
    ):
        raise CableSchemaError("positive finite unit multipliers required")

    rows, header = _read_csv_rows(p)
    idx = _resolve_columns(
        header,
        {
            "time": time_col,
            "kx": curvature_x_col,
            "ky": curvature_y_col,
            "tension": tension_col,
            "temp": temperature_col,
            "strain": axial_strain_col,
        },
        source=p.name,
    )
    cols = _columns_to_arrays(rows, idx, source=p.name)

    kx = cols["kx"] * curvature_scale_to_1_m
    ky = cols["ky"] * curvature_scale_to_1_m if "ky" in cols else np.zeros_like(kx)
    if curvature_convention == "rotation":
        kx, ky = rotation_to_strain_curvature(kx, ky)
    force_kind = _named_quantity_kind(
        header[idx["tension"]],
        tension_kind,
        {
            "wall tension": "true",
            "true tension": "true",
            "effective tension": "effective",
        },
        {"true", "effective", "unspecified"},
    )
    strain_kind = _named_quantity_kind(
        header[idx["strain"]] if "strain" in idx else "",
        axial_strain_kind,
        {"direct tensile strain": "direct", "total mean axial strain": "total"},
        {"direct", "total", "unspecified"},
    )
    return LoadCaseSeries(
        time_s=cols["time"],
        curvature_x_1_m=kx,
        curvature_y_1_m=ky,
        tension_N=cols["tension"] * tension_scale_to_N,
        temperature_C=cols.get("temp"),
        tension_kind=cast(Literal["true", "effective", "unspecified"], force_kind),
        axial_strain=cols["strain"] * axial_strain_scale if "strain" in cols else None,
        axial_strain_kind=cast(Literal["direct", "total", "unspecified"], strain_kind),
        source=p.name,
        provenance=Provenance(
            method="time-history CSV export reader",
            citations=(),
            gates=(
                "known-answer rainflow gate vs a synthetic fixture (L6.1)",
                "round-trip identity through the exchange CSV",
            ),
            limitations=(
                "reads an ASCII/CSV export of a global analysis",
                "column units are taken from the scale arguments; verify they "
                "match the export (curvature->1/m, tension->N)",
                "data origin is the upstream global model",
                f"input curvature convention: {curvature_convention}",
            ),
        ),
    )


# --------------------------------------------------------------------------- #
# Columnar helpers                                                            #
# --------------------------------------------------------------------------- #
def _named_quantity_kind(
    header: str, explicit: str | None, names: dict[str, str], allowed: set[str]
) -> str:
    """Retain recognised native meanings; never silently relabel a quantity."""
    if explicit is not None and explicit not in allowed:
        raise CableSchemaError(f"invalid quantity kind {explicit!r}")
    found = {kind for name, kind in names.items() if name in header.casefold()}
    if len(found) > 1:
        raise CableSchemaError(f"ambiguous quantity heading {header!r}")
    inferred = next(iter(found), "unspecified")
    if inferred != "unspecified" and explicit not in (None, inferred):
        raise CableSchemaError(f"quantity kind contradicts heading {header!r}")
    return inferred if explicit is None else explicit


def _read_csv_rows(path: Path) -> tuple[list[list[str]], list[str]]:
    """Return ``(data_rows, header)`` from a CSV, skipping blank lines.

    The file is read as UTF-8 (with or without a byte-order mark) and, failing
    that, as Windows-1252, the code page of Windows text exports.
    """
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise CableSchemaError(
            f"{path.name}: not a text CSV (neither UTF-8 nor Windows-1252)."
        )
    reader = csv.reader(text.splitlines())
    all_rows = [r for r in reader if any(cell.strip() for cell in r)]
    if not all_rows:
        raise CableSchemaError(f"{path.name}: empty CSV.")
    header = [c.strip() for c in all_rows[0]]
    if len(header) == 1:
        for delimiter in (";", "\t"):
            if delimiter in header[0]:
                raise CableSchemaError(
                    f"{path.name}: comma-separated values are required; the "
                    f"header is separated by {delimiter!r}."
                )
    return all_rows[1:], header


def _resolve_columns(
    header: list[str], wanted: dict[str, str | None], *, source: str
) -> dict[str, int]:
    """Map each wanted key to a header index by case-insensitive substring."""
    low = [h.lower() for h in header]
    idx: dict[str, int] = {}
    for key, name in wanted.items():
        if name is None:
            continue
        if not name.strip():
            raise CableSchemaError(f"{source}: an empty column selector is ambiguous")
        matches = [i for i, h in enumerate(low) if name.lower() in h]
        if not matches:
            raise CableSchemaError(
                f"{source}: required column {name!r} not found in header {header}."
            )
        if len(matches) != 1 or matches[0] in idx.values():
            raise CableSchemaError(f"{source}: ambiguous column selection {name!r}")
        idx[key] = matches[0]
    return idx


def _columns_to_arrays(
    rows: list[list[str]], idx: dict[str, int], *, source: str
) -> dict[str, FloatArray]:
    """Extract float columns by index; raises on a non-numeric / short row."""
    out: dict[str, list[float]] = {k: [] for k in idx}
    for r, row in enumerate(rows):
        for key, col in idx.items():
            if col >= len(row):
                raise CableSchemaError(
                    f"{source}: row {r + 1} has {len(row)} columns, need index {col}."
                )
            try:
                out[key].append(float(row[col]))
            except ValueError as exc:
                raise CableSchemaError(
                    f"{source}: row {r + 1} column {col} is not numeric: "
                    f"{row[col]!r} ({exc})."
                ) from None
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}
