# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""RIFLEX ASCII results reader.

Reads a **RIFLEX whitespace-delimited ASCII time-history** (the column dump from
a RIFLEX dynamic run: optional ``#``/``%`` comment lines, then one whitespace-
separated row per time sample) into a
:class:`~cablexsec.io.series.LoadCaseSeries`. Columns are selected by
**1-based index** (RIFLEX ASCII dumps are positional and rarely carry a usable
header), with unit scales to convert to ``1/m`` and ``N``.

Like the time-history CSV reader this is gated by the L6 known-answer test against a
synthetic fixture; it carries its data-origin and unit caveats in provenance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from cablexsec.exceptions import CableSchemaError
from cablexsec.io.series import LoadCaseSeries
from cablexsec.mechanics.section_conventions import rotation_to_strain_curvature
from cablexsec.provenance import Provenance

FloatArray = NDArray[np.float64]

__all__ = ["read_riflex_ascii"]


def read_riflex_ascii(
    path: str | Path,
    *,
    time_col: int = 1,
    curvature_x_col: int = 2,
    tension_col: int = 3,
    curvature_y_col: int | None = None,
    temperature_col: int | None = None,
    comment_prefixes: tuple[str, ...] = ("#", "%", "!"),
    curvature_scale_to_1_m: float = 1.0,
    tension_scale_to_N: float = 1.0,
    tension_kind: Literal["true", "effective", "unspecified"] = "unspecified",
    curvature_convention: Literal["strain_gradient", "rotation"] = "strain_gradient",
) -> LoadCaseSeries:
    r"""Read a RIFLEX ASCII time-history into a :class:`LoadCaseSeries`.

    Parameters
    ----------
    path : str or pathlib.Path
        RIFLEX ASCII results file.
    time_col, curvature_x_col, tension_col : int
        **1-based** column indices of the required quantities.
    curvature_y_col, temperature_col : int or None, optional
        1-based indices of the optional out-of-plane curvature / temperature;
        ``None`` → planar / isothermal.
    comment_prefixes : tuple[str, ...], optional
        Line prefixes treated as comments and skipped.
    curvature_scale_to_1_m, tension_scale_to_N : float, optional
        Multipliers from the file's units to ``1/m`` and N.
    tension_kind : str, optional
        Declare true or effective tension from the selected result definition.
        An unlabelled positional column is not assumed to be material force.
    curvature_convention : str, optional
        Selected signed columns are strain gradients by default. Choose
        ``rotation`` for right-handed rod rotation components (fibre strain
        ``k_x*y - k_y*x``); the reader then maps them to the section strain
        gradient ``(-k_y, k_x)``. Verify the source convention.

    Returns
    -------
    LoadCaseSeries
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"RIFLEX ASCII not found: {p}")
    if curvature_convention not in ("strain_gradient", "rotation"):
        raise CableSchemaError("declare rotation or strain_gradient curvature")

    wanted = {
        "time": time_col,
        "kx": curvature_x_col,
        "tension": tension_col,
        "ky": curvature_y_col,
        "temp": temperature_col,
    }
    chosen = {k: v for k, v in wanted.items() if v is not None}
    if any(
        isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in chosen.values()
    ):
        raise CableSchemaError(f"column indices must be 1-based integers: {chosen}")
    if len(set(chosen.values())) != len(chosen):
        raise CableSchemaError(f"each quantity needs its own column: {chosen}")
    cols: dict[str, list[float]] = {k: [] for k in chosen}
    raw = p.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("cp1252")
        except UnicodeDecodeError:
            raise CableSchemaError(
                f"{p.name}: not a text file (neither UTF-8 nor Windows-1252)."
            ) from None
    for ln, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if not s or s.startswith(comment_prefixes):
            continue
        fields = s.split()
        for key, col in chosen.items():
            if col > len(fields):
                raise CableSchemaError(
                    f"{p.name}: line {ln} has {len(fields)} fields, need "
                    f"1-based column {col} ({key}). Check the *_col indices."
                )
            try:
                cols[key].append(float(fields[col - 1]))
            except ValueError as exc:
                raise CableSchemaError(
                    f"{p.name}: line {ln} column {col} is not numeric: "
                    f"{fields[col - 1]!r} ({exc})."
                ) from None

    if not cols["time"]:
        raise CableSchemaError(f"{p.name}: no data rows found (all comments?).")

    arr = {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}
    kx = arr["kx"] * curvature_scale_to_1_m
    ky = arr["ky"] * curvature_scale_to_1_m if "ky" in arr else np.zeros_like(kx)
    if curvature_convention == "rotation":
        kx, ky = rotation_to_strain_curvature(kx, ky)
    return LoadCaseSeries(
        time_s=arr["time"],
        curvature_x_1_m=kx,
        curvature_y_1_m=ky,
        tension_N=arr["tension"] * tension_scale_to_N,
        temperature_C=arr.get("temp"),
        tension_kind=tension_kind,
        source=p.name,
        provenance=Provenance(
            method="RIFLEX ASCII time-history reader (positional columns)",
            citations=(),
            gates=(
                "known-answer rainflow gate vs a synthetic fixture (L6.1)",
                "round-trip identity through the exchange CSV",
            ),
            limitations=(
                "positional 1-based columns — verify the *_col indices match the "
                "RIFLEX dump layout",
                "column units are taken from the scale arguments (->1/m, ->N)",
                "data origin is the upstream RIFLEX model",
            ),
        ),
    )
