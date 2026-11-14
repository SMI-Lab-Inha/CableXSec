# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Malformed exchanges fail with useful errors before physical calculations."""

import numpy as np
import pytest

import cablexsec as cx


@pytest.mark.parametrize(
    "reader", [cx.read_time_history_csv, cx.read_riflex_ascii, cx.read_series_csv]
)
def test_missing_exchange_file(reader, tmp_path):
    with pytest.raises(FileNotFoundError):
        reader(tmp_path / "missing.csv")


@pytest.mark.parametrize(
    "body,kwargs,match",
    [
        ("", {}, "empty CSV"),
        ("Time,x curvature,y curvature,Wall tension\n0,0,0,bad\n", {}, "not numeric"),
        ("Time,x curvature,y curvature,Wall tension\n0,0\n", {}, "columns"),
        (
            "Time,x curvature,y curvature,Wall tension\n0,0,0,1\n",
            {"time_col": " "},
            "empty column",
        ),
        (
            "Time,x curvature,y curvature,Wall tension\n0,0,0,1\n",
            {"tension_kind": "bad"},
            "invalid quantity",
        ),
        (
            "Time,x curvature,y curvature,Wall tension Effective tension\n0,0,0,1\n",
            {},
            "ambiguous quantity",
        ),
        ("", {"curvature_convention": "bad"}, "declare"),
    ],
)
def test_bad_time_history_csv(tmp_path, body, kwargs, match):
    path = tmp_path / "export.csv"
    path.write_text(body)
    with pytest.raises(cx.CableSchemaError, match=match):
        cx.read_time_history_csv(path, **kwargs)


@pytest.mark.parametrize(
    "body,kwargs,match",
    [
        ("# comment only\n", {}, "no data"),
        ("0 0 bad\n", {}, "not numeric"),
        ("0 0\n", {}, "fields"),
        ("0 0 1\n", {"time_col": 0}, "1-based"),
        ("0 0 1\n", {"curvature_convention": "bad"}, "declare"),
    ],
)
def test_bad_riflex_ascii(tmp_path, body, kwargs, match):
    path = tmp_path / "export.txt"
    path.write_text(body)
    with pytest.raises(cx.CableSchemaError, match=match):
        cx.read_riflex_ascii(path, **kwargs)


_HEADER = "Time,x curvature,y curvature,Wall tension"
_ROWS = "0,0.01,0,50\n0.1,0.02,0,50\n"


def _read_export(path):
    return cx.read_time_history_csv(
        path, curvature_convention="rotation", tension_kind="true"
    )


def test_time_history_csv_reads_windows_1252_header(tmp_path):
    path = tmp_path / "cp1252.csv"
    path.write_bytes((_HEADER + " (\xb0)\n" + _ROWS).encode("cp1252"))
    series = _read_export(path)
    np.testing.assert_allclose(series.time_s, [0.0, 0.1])


def test_time_history_csv_refuses_binary_and_other_delimiters(tmp_path):
    binary = tmp_path / "binary.csv"
    binary.write_bytes(bytes(range(256)) * 4)
    with pytest.raises(cx.CableSchemaError, match="not a text CSV"):
        _read_export(binary)
    for delimiter in (";", "\t"):
        path = tmp_path / "delimited.csv"
        path.write_text((_HEADER + "\n" + _ROWS).replace(",", delimiter))
        with pytest.raises(cx.CableSchemaError, match="comma-separated"):
            _read_export(path)


def test_series_csv_accepts_byte_order_mark_and_refuses_repeated_columns(tmp_path):
    source = tmp_path / "export.csv"
    source.write_text(_HEADER + "\n" + _ROWS)
    written = cx.write_series_csv(_read_export(source), tmp_path / "series.csv")
    text = written.read_text(encoding="utf-8")
    bom = tmp_path / "bom.csv"
    bom.write_text("﻿" + text, encoding="utf-8")
    np.testing.assert_array_equal(cx.read_series_csv(bom).tension_N, [5.0e4, 5.0e4])
    head, *rows = text.splitlines()
    repeated = tmp_path / "repeated.csv"
    repeated.write_text(
        "\n".join([head + ",true_tension_N"] + [row + ",1" for row in rows]) + "\n"
    )
    with pytest.raises(cx.CableSchemaError, match="repeats a column name"):
        cx.read_series_csv(repeated)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"curvature_x_col": 1}, "its own column"),
        ({"time_col": 0}, "1-based integers"),
        ({"tension_col": -1}, "1-based integers"),
        ({"tension_col": 2.0}, "1-based integers"),
    ],
)
def test_riflex_column_indices_checked_before_reading(tmp_path, kwargs, match):
    path = tmp_path / "riflex.asc"
    path.write_text("0.0 0.01 5e4\n0.1 0.02 5e4\n")
    with pytest.raises(cx.CableSchemaError, match=match):
        cx.read_riflex_ascii(path, **kwargs)


def test_riflex_reads_windows_1252_comment(tmp_path):
    path = tmp_path / "riflex.asc"
    path.write_bytes("# angle (\xb0)\n0.0 0.01 5e4\n0.1 0.02 5e4\n".encode("cp1252"))
    np.testing.assert_allclose(cx.read_riflex_ascii(path).time_s, [0.0, 0.1])
