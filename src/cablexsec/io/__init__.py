# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Integration surfaces: global-analysis I/O.

Readers ingest an effective-tension + curvature (+ optional temperature) time
series from a global dynamic analysis and map it onto the screening-level
cross-section fatigue/stress pipeline. Everything funnels through one typed
:class:`~cablexsec.io.series.LoadCaseSeries`, so a parsing bug cannot silently
corrupt a fatigue number — the reader is plumbing, gated by a known-answer test.

- :func:`~cablexsec.io.synthetic.jonswap_series` — a self-contained JONSWAP-style
  generator so the demos/tests need no proprietary files.
- :func:`~cablexsec.io.time_history_csv.read_time_history_csv` — time-history
  CSV export of a global analysis, columns selected by header.
- :func:`~cablexsec.io.riflex_reader.read_riflex_ascii` — RIFLEX ASCII results.
- :func:`~cablexsec.io.cabledyn.read_cabledyn_sections` — CableDyn driver and
  OpenFAST module outputs: section histories with reconstructed signed
  curvature components (doc/global_local_guide.md).
- :func:`~cablexsec.io.cabledyn.read_cabledyn_range_graph` and
  :func:`~cablexsec.io.cabledyn.read_cabledyn_lift` — CableDyn range graphs
  (``.Line<L>.range.out``) and VIV lift forces (``.Line<L>.V.out``).
- :func:`~cablexsec.io.cabledyn_components.read_cabledyn_station_table` —
  CableDyn 0.4.0 station tables with signed curvature components (``CurvX``,
  ``CurvY``), sign-mapped to the strain gradient.
- :func:`~cablexsec.io.series.fatigue_from_series` — map a series onto
  :func:`cablexsec.compute_fatigue`.
"""

from __future__ import annotations

from cablexsec.io.cabledyn import (
    CableDynLift,
    CableDynRangeGraph,
    CableDynSectionHistory,
    read_cabledyn_deck,
    read_cabledyn_lift,
    read_cabledyn_range_graph,
    read_cabledyn_sections,
    section_channels,
)
from cablexsec.io.cabledyn_components import (
    read_cabledyn_station_table,
    series_from_curvature_components,
)
from cablexsec.io.riflex_reader import read_riflex_ascii
from cablexsec.io.series import (
    LoadCaseSeries,
    fatigue_from_series,
    read_series_csv,
    write_series_csv,
)
from cablexsec.io.synthetic import jonswap_series
from cablexsec.io.time_history_csv import read_time_history_csv

__all__ = [
    "CableDynLift",
    "CableDynRangeGraph",
    "CableDynSectionHistory",
    "LoadCaseSeries",
    "fatigue_from_series",
    "jonswap_series",
    "read_cabledyn_deck",
    "read_cabledyn_lift",
    "read_cabledyn_range_graph",
    "read_cabledyn_sections",
    "read_cabledyn_station_table",
    "read_riflex_ascii",
    "read_series_csv",
    "read_time_history_csv",
    "section_channels",
    "series_from_curvature_components",
    "write_series_csv",
]
