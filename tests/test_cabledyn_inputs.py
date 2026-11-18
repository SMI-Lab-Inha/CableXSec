# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""CableDyn reader: deck parsing, file layouts and input checks .

Each case corrupts one item of the synthetic run of ``test_cabledyn_reader``
and checks that the reader refuses it with a message naming the item, or
reads a documented variant correctly.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from test_cabledyn_reader import (
    _bending_case,
    _chord_tangent_channels,
    _deck,
    _fmt,
    _write_case,
)

from cablexsec.exceptions import CableSchemaError
from cablexsec.io.cabledyn import (
    read_cabledyn_deck,
    read_cabledyn_sections,
    section_channels,
)

NT = 4


def _case(tmp_path, nt=NT, **kwargs):
    positions = _bending_case(np.full(nt, 0.02), np.zeros(nt), nt)
    return _write_case(tmp_path, positions, **kwargs)


def _read(root, deck, **kwargs):
    kwargs.setdefault("pressure", "none")
    return read_cabledyn_sections(root, deck, {"mid": 5.0}, **kwargs)


def _edit(path, old, new):
    text = path.read_text(encoding="utf-8")
    assert old in text, old
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Arguments                                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"frame": "global"}, "frame"),
        ({"curvature_source": "spline"}, "curvature_source"),
        ({"tension_source": "mean"}, "tension_source"),
        ({"pressure": "flooded"}, "pressure"),
        ({"window_nodes": 1}, "window_nodes"),
        ({"window_nodes": True}, "window_nodes"),
        ({"chunk_rows": 8}, "chunk_rows"),
        ({"reference_vector": (0.0, 0.0, 0.0)}, "reference_vector"),
        ({"reference_vector": (1.0, 0.0)}, "reference_vector"),
    ],
)
def test_arguments_are_checked_before_any_file_is_read(tmp_path, kwargs, match):
    with pytest.raises(CableSchemaError, match=match):
        read_cabledyn_sections(
            tmp_path / "none", tmp_path / "none.dat", {"s": 1.0}, **kwargs
        )


def test_at_least_one_section(tmp_path):
    with pytest.raises(CableSchemaError, match="at least one section"):
        read_cabledyn_sections(tmp_path / "none", tmp_path / "none.dat", {})


# --------------------------------------------------------------------------- #
# Deck                                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        (
            "--------------------- LINES ---\n",
            "--------------------- WIRES ---\n",
            "no LINES",
        ),
        ("1.2 0.1 1.0 0.0\n", "1.2 0.1 1.0 0.0 7 8\n", "'cable' has 12 columns"),
        ("1.2 0.1 1.0 0.0\n", "1.2 0.1 1.0 0x\n", "Ca_t of cable is not numeric"),
        ("cable 0.2 40.0", "cable abc 40.0", "not numeric"),
        ("cable 0.2 40.0", "cable inf 40.0", "not finite"),
        (
            "ID Type X Y Z\n(-) (-) (m) (m) (m)\n1 Coupled 0 0 -10\n2 Fixed 10 0 -12\n",
            "",
            "POINTS has no data rows",
        ),
        ("2 Fixed 10 0 -12\n", "2 Fixed 10 0 -12 0\n", "POINTS row '2' has 6"),
        ("1 cable 10.0 100\n", "1 cable 10.0 100 5\n", "SECTIONS row has 5"),
        ("1 cable 10.0 100\n", "1 cable 10.0 1e2\n", "NumSegs is not an integer"),
        ("1 1 2 pt\n", "1 1 2 pt x\n", "LINES row of line 1 has 5 columns"),
        ("1 1 2 pt\n", "1 1 2 pt\n1 1 2 pt\n", "lists line 1 twice"),
        (
            "--------------------- POINTS ---\n",
            "cable 0.2 40.0 5e8 0.0 2e4 1.2 0.1 1.0 0.0\n--------------------- POINTS ---\n",
            "defines 'cable' twice",
        ),
        ("1 cable 10.0 100\n", "2 cable 10.0 100\n", "line 1 has no sections"),
        ("1 cable 10.0 100\n", "1 rope 10.0 100\n", "undefined type"),
        ("1 cable 10.0 100\n", "1 cable 10.0 0\n", "invalid section"),
        (
            "--------------------- OUTPUTS ---",
            "maybe adaptive_mesh - x\n--------------------- OUTPUTS ---",
            "adaptive_mesh 'maybe'",
        ),
        (
            "ID NodeA NodeB Outputs\n(-) (-) (-) (-)\n1 1 2 pt\n",
            "ID LineType AttachA AttachB UnstrLen NumSegs Outputs\n1 cable 1 2 10.0 100 pt\n",
            "7-column row and SECTIONS",
        ),
    ],
)
def test_deck_errors_name_the_item(tmp_path, old, new, match):
    path = _deck(tmp_path / "d.dat")
    _edit(path, old, new)
    with pytest.raises(CableSchemaError, match=match):
        read_cabledyn_deck(path)


def test_missing_deck(tmp_path):
    with pytest.raises(FileNotFoundError, match="deck not found"):
        read_cabledyn_deck(tmp_path / "absent.dat")


def test_seven_column_lines_comments_defaults_and_adaptive_flag(tmp_path):
    path = _deck(tmp_path / "d.dat")
    _edit(
        path,
        "ID NodeA NodeB Outputs\n(-) (-) (-) (-)\n1 1 2 pt\n",
        "ID LineType AttachA AttachB UnstrLen NumSegs Outputs\n"
        "1 cable 1 2 10.0 100 pt # one row per line\n",
    )
    _edit(
        path,
        "--------------------- SECTIONS ---\n",
        "--------------------- UNUSED ---\n",
    )
    _edit(
        path,
        "9.81 g - gravity\n1025.0 rhoW - density\n",
        "yes adaptive_mesh ! refine\n",
    )
    deck = read_cabledyn_deck(path)
    line = deck.line(1)
    assert (line.end_a, line.end_b, line.outputs) == ("1", "2", "pt")
    assert line.n_elements == 100 and line.n_nodes == 101
    assert deck.adaptive_mesh
    assert set(deck.defaults_used) == {"g", "rhoW"}
    assert (deck.g_m_s2, deck.rho_w_kg_m3) == (9.80665, 1025.0)  # documented defaults
    with pytest.raises(CableSchemaError, match="no LINES row with id 7"):
        deck.line(7)
    with pytest.raises(CableSchemaError, match="line type 'rope'"):
        deck.line_type("rope")


def test_line_type_along_two_sections(tmp_path):
    path = _deck(tmp_path / "d.dat")
    _edit(
        path,
        "--------------------- POINTS ---\n",
        "buoy 0.3 50.0 5.0e8 0.0 2.0e4 1.2 0.1 1.0 0.0\n--------------------- POINTS ---\n",
    )
    _edit(path, "1 cable 10.0 100\n", "1 cable 10.0 100\n1 buoy 5.0 10\n")
    line = read_cabledyn_deck(path).line(1)
    assert line.length_m == pytest.approx(15.0) and line.n_elements == 110
    assert line.line_type_at(3.0) == "cable"
    assert line.line_type_at(12.0) == "buoy"
    assert line.line_type_at(40.0) == "buoy"  # beyond the end: last section


def test_section_channels_lists_what_a_deck_must_write(tmp_path):
    deck = _deck(tmp_path / "d.dat")
    names = section_channels(deck, {"mid": 5.05})
    assert names[:2] == ("L1N49Dec", "L1N49Azi")
    assert {"Ten1N51", "Ten1N52", "Curv1N51", "L1N51px", "L1N52pz"} <= set(names)
    assert len(names) == len(set(names))
    bare = section_channels(deck, {"mid": 5.05}, positions=False, check_curvature=False)
    assert not any(n.startswith("Curv") or n.endswith(("px", "py", "pz")) for n in bare)
    torsion = section_channels(_deck(tmp_path / "t.dat", torsion=True), {"mid": 5.05})
    assert {"Twist1N51", "Torq1N52"} <= set(torsion)
    with pytest.raises(CableSchemaError, match="outside the line"):
        section_channels(deck, {"far": 11.0})


# --------------------------------------------------------------------------- #
# Output files                                                                #
# --------------------------------------------------------------------------- #
def test_output_discovery(tmp_path):
    root, deck = _case(tmp_path)
    (tmp_path / "case.CD.out").write_text("Time\n(s)\n", encoding="utf-8")
    with pytest.raises(CableSchemaError, match="name one output root"):
        _read(root, deck)
    with pytest.raises(FileNotFoundError, match="no CableDyn output"):
        _read(tmp_path / "other", deck)


def test_coupled_static_profile_is_recorded(tmp_path):
    nt = 3
    positions = _bending_case(np.full(nt, 0.02), np.zeros(nt), nt)
    channels = _chord_tangent_channels(positions)
    names = [f"L1N{j}{kind}" for j in range(48, 56) for kind in ("Dec", "Azi")]
    names += ["Ten1N51", "Ten1N52"]
    channels["Ten1N51"] = channels["Ten1N52"] = np.full(nt, 1000.0)
    _deck(tmp_path / "of.dat", channels=names)
    with open(tmp_path / "of.CD.out", "w", encoding="utf-8") as f:
        f.write("\t".join(["Time", *names]) + "\n")
        units = ["(deg)" if n.endswith(("Dec", "Azi")) else "(N)" for n in names]
        f.write("\t".join(["(s)", *units]) + "\n")
        for k in range(nt):
            f.write(_fmt([0.1 * k, *[channels[n][k] for n in names]]) + "\n")
    (tmp_path / "of.CD.static.out").write_text(
        "CableDyn coupled static configuration\n"
        "LineID\tNode\tArcLength\tX\tY\tZ\n"
        "(-)\t(-)\t(m)\t(m)\t(m)\t(m)\n"
        + "".join(f"1\t{j}\t0\t0\t0\t0\n" for j in range(1, 102)),
        encoding="utf-8",
    )
    read = read_cabledyn_sections(
        tmp_path / "of",
        tmp_path / "of.dat",
        {"s": 5.05},
        frame="section",
        pressure="none",
    )
    assert {f.role for f in read.files} == {"main", "static", "deck"}


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("# CableDyn driver output\n", "incomplete CableDyn output table"),
        ("# CableDyn driver output\nTime\n0.0\n", "first column must be Time"),
        ("Time\tTen1N1\n(min)\t(N)\n0.0\t1.0\n", "OpenFAST CableDyn table"),
    ],
)
def test_main_table_layout(tmp_path, text, match):
    root, deck = _case(tmp_path)
    (tmp_path / "case.out").write_text(text, encoding="utf-8")
    with pytest.raises(CableSchemaError, match=match):
        _read(root, deck)


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("# CableDyn finite-EI", "# Other finite-EI", "not a CableDyn per-line table"),
        ("public node order EndA -> EndB", "public node order", "public order"),
        (
            "# CableDyn finite-EI Hermite line",
            "# CableDyn static line",
            "no time history",
        ),
    ],
)
def test_per_line_table_title(tmp_path, old, new, match):
    root, deck = _case(tmp_path)
    _edit(tmp_path / "case.Line1.p.out", old, new)
    with pytest.raises(CableSchemaError, match=match):
        _read(root, deck)


def test_non_finite_history_row(tmp_path):
    root, deck = _case(tmp_path)
    path = tmp_path / "case.Line1.p.out"
    lines = path.read_text(encoding="utf-8").splitlines()
    cells = lines[3].split("\t")
    cells[4] = "nan"
    lines[3] = "\t".join(cells)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(CableSchemaError, match="non-finite row"):
        _read(root, deck)


def test_duplicated_channel(tmp_path):
    positions = _bending_case(np.full(NT, 0.02), np.zeros(NT), NT)
    root, deck = _write_case(
        tmp_path, positions, channels=_chord_tangent_channels(positions)
    )
    _edit(tmp_path / "case.out", "\tL1N2Dec\t", "\tL1N1Dec\t")
    with pytest.raises(CableSchemaError, match="L1N1Dec is duplicated"):
        _read(root, deck)


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda rows: rows[:3], "incomplete static profile"),
        (
            # Units row: LineID Node ArcLength X ...; X (column 4) in feet.
            lambda rows: [
                rows[0],
                rows[1],
                "\t".join(
                    "(ft)" if i == 3 else u for i, u in enumerate(rows[2].split("\t"))
                ),
                *rows[3:],
            ],
            "static column X",
        ),
        (
            lambda rows: [
                *rows[:3],
                *[r.replace("1\t1\t", "1\t0\t", 1) for r in rows[3:4]],
                *rows[4:],
            ],
            "rows missing or unordered",
        ),
        (lambda rows: rows[:-1], "different numbers of nodes"),
    ],
)
def test_static_profile(tmp_path, edit, match):
    root, deck = _case(tmp_path)
    path = tmp_path / "case.static.out"
    rows = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(edit(rows)) + "\n", encoding="utf-8")
    with pytest.raises(CableSchemaError, match=match):
        _read(root, deck)


def _elements(
    path,
    arcs,
    *,
    header="LineID\tElement\tReferenceArcStart\tReferenceArcEnd",
    unit="(m)",
    order=None,
    gap=0.0,
):
    order = range(1, arcs.size) if order is None else order
    rows = [
        "CableDyn continuous Hermite-element extrema (public order End A -> End B)",
        header,
        f"(-)\t(-)\t{unit}\t(m)",
    ]
    for e in order:
        start = arcs[e - 1] + (gap if e == 2 else 0.0)
        rows.append(f"1\t{e}\t{float(start)!r}\t{float(arcs[e])!r}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_element_file_confirms_the_mesh(tmp_path):
    root, deck = _case(tmp_path)
    arcs = 10.0 * np.arange(101) / 100
    _elements(tmp_path / "case.elements.out", arcs)
    read = _read(root, deck)
    assert read.checks["arc_lengths"] == "deck sections, confirmed by .elements.out"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"header": "LineID\tElement\tStart\tReferenceArcEnd"},
            "column ReferenceArcStart missing",
        ),
        ({"unit": "(ft)"}, "ReferenceArcStart must be in"),
        ({"order": [1, 3, 2, *range(4, 101)]}, "unordered"),
        ({"gap": 0.01}, "have gaps"),
    ],
)
def test_element_file_layout(tmp_path, kwargs, match):
    root, deck = _case(tmp_path)
    _elements(tmp_path / "case.elements.out", 10.0 * np.arange(101) / 100, **kwargs)
    with pytest.raises(CableSchemaError, match=match):
        _read(root, deck)


def test_mesh_change_is_checked_by_the_element_file(tmp_path):
    root, deck = _case(tmp_path)
    # As many nodes as the deck, at other arc lengths.
    _elements(tmp_path / "case.elements.out", 10.0 * (np.arange(101) / 100) ** 1.1)
    with pytest.raises(CableSchemaError, match="differs from the deck sections"):
        _read(root, deck)
    _elements(tmp_path / "case.elements.out", 10.0 * np.arange(51) / 50)
    with pytest.raises(CableSchemaError, match="files from different runs"):
        _read(root, deck)
    (tmp_path / "case.elements.out").unlink()
    # adaptive_mesh may refine; files listing the deck nodes show it did not.
    _edit(
        deck,
        "--------------------- OUTPUTS ---",
        "true adaptive_mesh - x\n--------------------- OUTPUTS ---",
    )
    read = _read(root, deck)
    assert read.checks["mesh"]["refined"] is False
    assert read.checks["arc_lengths"] == "deck sections"


@pytest.mark.parametrize("ei", [0.0, 2.0e4])
def test_element_file_of_other_lines(tmp_path, ei):
    """``.elements.out`` may list only the finite-EI lines of a run."""
    root, deck = _case(tmp_path, deck_kwargs={"ei": ei})
    path = tmp_path / "case.elements.out"
    _elements(path, 10.0 * np.arange(101) / 100)
    text = path.read_text(encoding="utf-8").replace("\n1\t", "\n2\t")
    path.write_text(text, encoding="utf-8")
    if ei > 0.0:
        with pytest.raises(CableSchemaError, match="elements of line 1"):
            _read(root, deck)
    else:
        assert _read(root, deck).checks["arc_lengths"] == "deck sections"


# --------------------------------------------------------------------------- #
# Histories                                                                   #
# --------------------------------------------------------------------------- #
def test_segment_tensions_need_the_t_file(tmp_path):
    root, deck = _case(tmp_path, outputs="p")
    with pytest.raises(CableSchemaError, match="needs case.Line1.t.out"):
        _read(root, deck, tension_source="segments")


def test_per_line_files_must_match_the_main_table(tmp_path):
    root, deck = _case(tmp_path)
    path = tmp_path / "case.Line1.p.out"
    rows = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(CableSchemaError, match="different row counts"):
        _read(root, deck)
    _case(tmp_path)
    shifted = [rows[0], rows[1]] + [
        "\t".join([f"{float(r.split(chr(9))[0]) + 0.05: .16E}", *r.split("\t")[1:]])
        for r in rows[2:]
    ]
    path.write_text("\n".join(shifted) + "\n", encoding="utf-8")
    with pytest.raises(CableSchemaError, match="time columns"):
        _read(root, deck)


def test_deck_length_must_describe_the_output_mesh(tmp_path):
    positions = _bending_case(np.full(NT, 0.02), np.zeros(NT), NT)
    root, deck = _write_case(tmp_path, positions, deck_kwargs={"length": 5.0})
    with pytest.raises(CableSchemaError, match="do not describe the output mesh"):
        read_cabledyn_sections(root, deck, {"mid": 2.0}, pressure="none")


@pytest.mark.parametrize(
    ("time", "match"),
    [
        (np.array([0.0]), "fewer than two output rows"),
        (np.array([0.0, 0.1, 0.1, 0.3]), "time does not increase"),
        (np.array([0.0, 0.1, 0.25, 0.3]), "not uniform"),
    ],
)
def test_output_time_axis(tmp_path, time, match):
    positions = _bending_case(np.full(time.size, 0.02), np.zeros(time.size), time.size)
    root, deck = _write_case(tmp_path, positions, time=time)
    with pytest.raises(CableSchemaError, match=match):
        _read(root, deck)


def test_frame_seed_and_reversal(tmp_path):
    root, deck = _case(tmp_path)
    with pytest.raises(CableSchemaError, match="reference_vector"):
        _read(root, deck, reference_vector=(1.0, 0.0, 0.0))  # along the tangent
    positions = _bending_case(np.full(NT, 0.02), np.zeros(NT), NT)
    flipped = positions.copy()
    flipped[2:] = -positions[2:]  # point reflection: every tangent reverses
    root, deck = _write_case(tmp_path, flipped, root="flip")
    with pytest.raises(CableSchemaError, match="reverses"):
        _read(root, deck, frame="section")


def test_streamed_chunks_give_the_single_chunk_result(tmp_path):
    nt = 40
    kappa = 0.03 + 0.01 * np.sin(np.arange(nt) / 3.0)
    positions = _bending_case(kappa, 0.2 * np.sin(np.arange(nt) / 5.0), nt)
    root, deck = _write_case(tmp_path, positions, deck_kwargs={"t_max": 4.0})
    for frame in ("section", "arc"):
        one = _read(root, deck, frame=frame).histories["mid"]
        many = _read(root, deck, frame=frame, chunk_rows=16).histories["mid"]
        for item in ("rotation_curvature_x_1_m", "rotation_curvature_y_1_m"):
            np.testing.assert_allclose(
                getattr(many, item), getattr(one, item), rtol=0.0, atol=1e-12
            )


def test_hydrostatic_pressure_needs_an_elevation(tmp_path):
    positions = _bending_case(np.full(NT, 0.02), np.zeros(NT), NT)
    channels = _chord_tangent_channels(positions)
    channels.update({f"Ten1N{j}": np.full(NT, 5000.0) for j in range(1, 102)})
    root, deck = _write_case(
        tmp_path, positions, channels=channels, outputs="-", static=False
    )
    with pytest.raises(
        CableSchemaError, match="section elevation needs node positions"
    ):
        read_cabledyn_sections(
            root, deck, {"mid": 5.05}, frame="section", curvature_source="tangent"
        )


def test_section_history_contract(tmp_path):
    root, deck = _case(tmp_path)
    history = _read(root, deck, pressure="hydrostatic").histories["mid"]
    with pytest.raises(CableSchemaError, match="time must increase strictly"):
        dataclasses.replace(history, time_s=history.time_s[::-1])
    with pytest.raises(CableSchemaError, match="effective_tension_N must be finite"):
        dataclasses.replace(
            history, effective_tension_N=history.effective_tension_N[:-1]
        )
    with pytest.raises(CableSchemaError, match="no external pressure history"):
        dataclasses.replace(history, external_pressure_Pa=None).true_tension_N(0.03)
    with pytest.raises(CableSchemaError, match="fewer than two samples"):
        history.window(start_time_s=10.0)
    assert math.isclose(history.duration_s, 0.1 * (NT - 1))


def test_section_history_exports(tmp_path):
    root, deck = _case(tmp_path, nt=6)
    dry = _read(root, deck).histories["mid"]
    np.testing.assert_array_equal(dry.true_tension_N(0.03), dry.effective_tension_N)
    columns = dry.to_columns(0.03)
    assert {"time_s", "true_tension_N", "strain_gradient_x_1_m", "elevation_m"} <= set(
        columns
    )
    assert "external_pressure_Pa" not in columns
    wet = _read(root, deck, pressure="hydrostatic").histories["mid"]
    np.testing.assert_allclose(
        wet.to_columns(0.03)["true_tension_N"],
        wet.effective_tension_N - wet.external_pressure_Pa * 0.03,
    )
    part = wet.window(0.1, 0.3)
    np.testing.assert_allclose(part.time_s, [0.1, 0.2, 0.3])
    assert part.external_pressure_Pa.shape == (3,)
    with pytest.raises(CableSchemaError, match="before the requested"):
        wet.window(0.0, 9.0)


@pytest.mark.parametrize("token", ["5.0e8|7.0e8", "5e8|0.2|0.3", "SYROPE:owc.txt|2|3"])
def test_non_numeric_axial_stiffness_token(tmp_path, token):
    # A viscoelastic or Syrope constitutive token in the EA column.
    path = _deck(tmp_path / "d.dat")
    _edit(path, "cable 0.2 40.0 500000000.0", f"cable 0.2 40.0 {token}")
    assert read_cabledyn_deck(path).line_type("cable").ea_N is None
    _edit(path, f"cable 0.2 40.0 {token}", "cable 0.2 40.0 visco_table")
    with pytest.raises(CableSchemaError, match="EA of cable is not numeric"):
        read_cabledyn_deck(path)


_ROW_10 = "cable 0.2 40.0 500000000.0 0.0 20000.0 1.2 0.1 1.0 0.0\n"


def _line_types(tmp_path, row):
    path = _deck(tmp_path / "d.dat")
    _edit(path, _ROW_10, row)
    return path


@pytest.mark.parametrize(
    ("ei", "bb", "alpha"),
    [
        ("2.0e4", 0.0, 0.0),
        ("2.0e4|0", 0.0, 0.0),
        ("2.0e4|1.5e3", 1.5e3, 0.0),
        ("2.0e4|-0.05|0.5", 0.05 * 2.0e4 / (math.pi * 0.5), 0.0),
        (
            "2.0e4|-0.02|0.2|2.0",
            0.02 * 2.0e4 / (math.pi * 2.2),
            4.0 * math.pi * 0.02 * 0.2 * 2.0 / 2.2,
        ),
        ("2.0D4|1.5D3", 1.5e3, 0.0),
    ],
)
def test_bending_damping_in_the_ei_column(tmp_path, ei, bb, alpha):
    """``EI|Bb``, ``EI|-zeta|f`` and ``EI|-zeta|f1|f2`` (CableDyn 0.4.x)."""
    path = _line_types(tmp_path, f"cable 0.2 40.0 5e8 0.0 {ei} 1.2 0.1 1.0 0.0\n")
    line_type = read_cabledyn_deck(path).line_type("cable")
    assert line_type.ei_Nm2 == 2.0e4
    assert line_type.bending_damping_Nm2s == pytest.approx(bb, rel=1e-15)
    assert line_type.mass_damping_1_s == pytest.approx(alpha, rel=1e-15)
    assert line_type.ei_token == ei
    assert line_type.gj_Nm2 is None


@pytest.mark.parametrize(
    ("ei", "match"),
    [
        ("2.0e4|-0.05", "needs the frequency"),
        ("2.0e4|-1.5|0.5", "0 < zeta < 1"),
        ("2.0e4|-0.05|0.0", "positive frequencies"),
        ("2.0e4|-0.05|0.2|-1", "positive frequencies"),
        ("0.0|100", "needs a finite-EI cable"),
        ("0.0|-0.05|0.5", "needs a finite-EI cable"),
        ("2e4|1|2|3|4", "5 '\\|'-separated parts, at most 4"),
        ("-2e4", "EI must be >= 0"),
        ("2e4|x", "not numeric"),
    ],
)
def test_bending_damping_errors(tmp_path, ei, match):
    path = _line_types(tmp_path, f"cable 0.2 40.0 5e8 0.0 {ei} 1.2 0.1 1.0 0.0\n")
    with pytest.raises(CableSchemaError, match=match):
        read_cabledyn_deck(path)


def _row_14(gj, *, ea="5e8", ei="2e4", irn="0.2"):
    return f"cable 0.2 40.0 {ea} 0.0 {ei} 1.9e8 {gj} 0.1 {irn} 1.2 0.1 1.0 0.0\n"


@pytest.mark.parametrize(
    ("gj", "values"),
    [
        ("1.5e4", (1.5e4, 0.0, 0.0)),
        ("1.5e4|0", (1.5e4, 0.0, 0.0)),
        ("1.5e4|50.0", (1.5e4, 50.0, 0.0)),
        ("1.5e4|-50.0|-0.01", (1.5e4, -50.0, -0.01)),
        ("1.5e4|0|20.0", (1.5e4, 0.0, 20.0)),
    ],
)
def test_coupling_and_torsional_damping_in_the_gj_column(tmp_path, gj, values):
    """``GJ``, ``GJ|K``, ``GJ|K|Bt`` of the 14-column row (CableDyn 0.4.0)."""
    line_type = read_cabledyn_deck(_line_types(tmp_path, _row_14(gj))).line_type(
        "cable"
    )
    assert (
        line_type.gj_Nm2,
        line_type.torque_tension_coupling_Nm,
        line_type.torsional_damping,
    ) == values
    assert (line_type.ea_N, line_type.ei_Nm2) == (5e8, 2e4)
    assert line_type.polar_inertia_kg_m == 0.2
    assert line_type.gj_token == gj


@pytest.mark.parametrize(
    ("row", "match"),
    [
        (_row_14("1.5e4|1e7"), "K\\^2 < EA GJ"),
        (_row_14("1.5e4|50", ei="0.0"), "coupling \\(GJ\\|K\\) needs"),
        (_row_14("1.5e4|50", ea="5e8|6e8"), "linear-elastic"),
        (_row_14("0|50"), "GJ > 0"),
        (_row_14("1.5e4|0|20", irn="0"), "torsional damping \\(GJ\\|K\\|Bt\\)"),
        (_row_14("1.5e4|1|2|3"), "4 '\\|'-separated parts, at most 3"),
        (_row_14("1.5e4", irn="x"), "Irn of cable is not numeric"),
    ],
)
def test_gj_column_errors(tmp_path, row, match):
    with pytest.raises(CableSchemaError, match=match):
        read_cabledyn_deck(_line_types(tmp_path, row))


def test_line_type_columns_are_read_by_position(tmp_path):
    """CableDyn reads a LINE TYPES row by its token count, not by the header.

    An 11-column row is the 10 columns and the VIV lift coefficient ``Cl``:
    a header naming the 11th column ``GJ`` does not make it one, and a
    stock-MoorDyn header (``Mass/m``, ``Cd Ca CdAx CaAx``) reads the same row.
    """
    path = _deck(tmp_path / "d.dat")
    _edit(
        path,
        "TypeName Diam MassDenInAir EA BA EI Cd_n Cd_t Ca_n Ca_t\n",
        "LineType Diam Mass/m EA BA/-zeta EI Cd Ca CdAx CaAx GJ\n",
    )
    _edit(path, _ROW_10, _ROW_10.replace(" 0.0\n", " 0.0 0.8\n"))
    line_type = read_cabledyn_deck(path).line_type("cable")
    assert line_type.gj_Nm2 is None and line_type.ea_N == 5e8
    _edit(path, " 0.0 0.8\n", " 0.0 1.0e4|50.0\n")
    with pytest.raises(CableSchemaError, match="Cl of cable is not numeric"):
        read_cabledyn_deck(path)


def test_quoted_names_and_header_rows_anywhere(tmp_path):
    path = _deck(tmp_path / "d.dat")
    _edit(path, _ROW_10, _ROW_10.replace("cable", '"armoured cable"'))
    _edit(path, "1 cable 10.0 100\n", '(-) (-) (m) (-)\n1 "armoured cable" 10.0 100\n')
    deck = read_cabledyn_deck(path)
    assert deck.line(1).sections[0].line_type == "armoured cable"
    assert deck.line_type("Armoured Cable").ei_Nm2 == 2e4


@pytest.mark.parametrize(
    ("row", "ends"),
    [
        ("1 A Rigid 1 0 0 Free 0 1 0\n1 B Rigid 1 0 0 Zero 0 1 0\n", set()),
        ("1 A Rigid 1 0 0 0 0 1 0\n1 B Rigid 1 0 0 0.0e0 0 1 0 15\n", set()),
        ("1 EndA Rigid 1 0 0 Inf 0 1 0\n1 End_B Rigid 1 0 0 2.5e3 0 1 0\n", {"a", "b"}),
        ("1 B Pinned 1 0 0 Infinity 0 1 0 0\n", {"b"}),
        ("1 A Rigid 1 0 0\n", set()),
    ],
)
def test_end_connection_torsion_tokens(tmp_path, row, ends):
    """TorsStiffness Free/Zero/0 leaves the end free in torsion (CableDyn)."""
    path = _deck(tmp_path / "d.dat", torsion=True)
    _edit(path, "1 A Rigid 1 0 0 Rigid 0 1 0 0\n1 B Rigid 1 0 0 Rigid 0 1 0 0\n", row)
    line = read_cabledyn_deck(path).line(1)
    assert line.torsion_restrained_ends == ends
    assert line.torsion_restrained == (ends == {"a", "b"})


@pytest.mark.parametrize(
    ("row", "match"),
    [
        ("1 C Rigid 1 0 0 Rigid 0 1 0\n", "End of line 1 must be A or B"),
        ("1 A Rigid 1 0 0 -5 0 1 0\n", "TorsStiffness of line 1 must be"),
        ("1 A Rigid 1 0 0 Rigid 0 1\n", "END CONNECTIONS row has 9 columns"),
    ],
)
def test_end_connection_errors(tmp_path, row, match):
    path = _deck(tmp_path / "d.dat", torsion=True)
    _edit(path, "1 A Rigid 1 0 0 Rigid 0 1 0 0\n1 B Rigid 1 0 0 Rigid 0 1 0 0\n", row)
    with pytest.raises(CableSchemaError, match=match):
        read_cabledyn_deck(path)


def test_anchor_first_swap_also_swaps_the_torsion_ends(tmp_path):
    path = _deck(tmp_path / "d.dat", torsion=True)
    _edit(path, "1 1 2 pt\n", "1 2 1 pt\n")
    _edit(path, "1 B Rigid 1 0 0 Rigid 0 1 0 0\n", "")
    line = read_cabledyn_deck(path).line(1)
    assert line.reversed_by_parser
    assert line.torsion_restrained_ends == {"b"}


def test_finite_ei_line_to_a_rod_end_is_swapped(tmp_path):
    path = _deck(tmp_path / "d.dat")
    _edit(path, "1 1 2 pt\n", "1 2 R1A pt\n")
    assert read_cabledyn_deck(path).line(1).reversed_by_parser
    path = _deck(tmp_path / "d.dat", ei=0.0)
    _edit(path, "1 1 2 pt\n", "1 2 R1A pt\n")
    assert not read_cabledyn_deck(path).line(1).reversed_by_parser


def test_outputs_end_closes_the_channel_list(tmp_path):
    """An unquoted ``END`` closes the list; CableDyn takes a quoted one as a name."""
    path = _deck(tmp_path / "d.dat", channels=["FairTen1", "END", "AnchTen1"])
    assert read_cabledyn_deck(path).channels == ("FairTen1", "END", "AnchTen1")
    _edit(path, '"END"\n', "end\n")
    assert read_cabledyn_deck(path).channels == ("FairTen1",)
    _edit(path, '"FairTen1"\n', '"FairTen1","Ten1N2" Curv1N2\n')
    assert read_cabledyn_deck(path).channels == ("FairTen1", "Ten1N2", "Curv1N2")


# --------------------------------------------------------------------------- #
# Records as CableDyn reads them                                              #
# --------------------------------------------------------------------------- #
def test_comments_and_headers_as_cabledyn_reads_them(tmp_path):
    """``scan_comment`` and ``is_section_header`` of the CableDyn deck reader.

    ``--`` starts a comment at the start of a record or after a separator,
    unless the record holds a ``---`` rule; ``#`` and ``!`` start one
    anywhere. A record holding ``---`` is a header named by its other
    characters, a byte-order mark is dropped and a section met again goes on.
    """
    path = _deck(tmp_path / "d.dat")
    _edit(path, "1 1 2 pt\n", "1 1 2 pt -- the only line\n")
    _edit(path, "--------------------- POINTS ---\n", "POINTS -------- # points\n")
    _edit(path, "1 Coupled 0 0 -10\n", "1 Coupled 0 0 -10\t--hang-off\n")
    _edit(
        path,
        "--------------------- OPTIONS ---\n",
        "--- SECTIONS ---\n1 cable 2.0 20\n--------------------- OPTIONS ---\n",
    )
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    deck = read_cabledyn_deck(path)
    line = deck.line(1)
    assert line.outputs == "pt" and deck.point_types["1"] == "coupled"
    assert [sec.length_m for sec in line.sections] == [10.0, 2.0]
    # Not a comment: '--' inside a token, after which the row has 5 columns.
    _edit(path, "1 1 2 pt -- the only line\n", "1 1 2 pt a--b\n")
    with pytest.raises(CableSchemaError, match="LINES row of line 1 has 5 columns"):
        read_cabledyn_deck(path)


@pytest.mark.parametrize("mark", ["#", "!"])
def test_comment_marks_cut_quoted_tokens_too(tmp_path, mark):
    """CableDyn cuts a record at ``#`` or ``!`` even inside quotes
    (``doc/options.rst``, file paths); the reader does the same."""
    path = _deck(tmp_path / "d.dat")
    _edit(path, _ROW_10, _ROW_10.replace("cable", f'"armoured{mark}cable"'))
    # The cut row holds no number: a header row to CableDyn, so no type is left.
    with pytest.raises(CableSchemaError, match="LINE TYPES has no data rows"):
        read_cabledyn_deck(path)


def test_quoted_token_runs_to_the_next_separator(tmp_path):
    """A token opening with a quote runs past its closing quote to the next
    separator (CableDyn ``token_bounds``), so ``"a b"c`` is one token."""
    path = _deck(tmp_path / "d.dat")
    _edit(path, _ROW_10, _ROW_10.replace("cable", '"armoured cable"x'))
    _edit(path, "1 cable 10.0 100\n", '1 "armoured cable"x 10.0 100\n')
    deck = read_cabledyn_deck(path)
    assert deck.line(1).sections[0].line_type == "armoured cablex"


# --------------------------------------------------------------------------- #
# LINE TYPES checks                                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "row",
    [
        "cable 0.2 40.0 5e8 0.0 2e4 -1.2 0.1 1.0 0.0\n",
        "cable 0.2 40.0 5e8 0.0 2e4 1.2 0.1 1.0 -0.1 0.8\n",
        _row_14("1.5e4").replace(" 1.0 0.0\n", " -1.0 0.0\n"),
    ],
)
def test_negative_drag_or_added_mass_is_refused(tmp_path, row):
    with pytest.raises(CableSchemaError, match="must be non-negative"):
        read_cabledyn_deck(_line_types(tmp_path, row))


@pytest.mark.parametrize(
    "row",
    [
        "cable 0.2 40.0 5e8 0.0 2e4 0 1.5e4 0.1 0.2 1.2 0.1 1.0 0.0\n",
        "cable 0.2 40.0 5e8 0.0 2e4 1.9e8 0 0.1 0.2 1.2 0.1 1.0 0.0\n",
        "cable 0.2 40.0 5e8 0.0 2e4 1.9e8 1.5e4 -0.1 0.2 1.2 0.1 1.0 0.0\n",
        "cable 0.2 40.0 5e8 0.0 2e4 1.9e8 1.5e4 0.1 0 1.2 0.1 1.0 0.0\n",
    ],
)
def test_finite_ei_columns_are_all_positive_or_all_zero(tmp_path, row):
    with pytest.raises(CableSchemaError, match="GAs/GJ/Irt/Irn must all be > 0"):
        read_cabledyn_deck(_line_types(tmp_path, row))


def test_finite_ei_columns_all_zero_give_no_gj(tmp_path):
    row = "cable 0.2 40.0 5e8 0.0 2e4 0 0 0 0 1.2 0.1 1.0 0.0\n"
    line_type = read_cabledyn_deck(_line_types(tmp_path, row)).line_type("cable")
    assert line_type.gj_Nm2 == 0.0 and line_type.polar_inertia_kg_m == 0.0


# --------------------------------------------------------------------------- #
# EQUIVALENT BUOYANCY                                                         #
# --------------------------------------------------------------------------- #
_EQUIVALENT = (
    "--------------------- EQUIVALENT BUOYANCY ---\n"
    "LineType Diam SubmergedWeight\n(-) (m) (N/m)\n"
)


def _equivalent(tmp_path, rows):
    path = _deck(tmp_path / "d.dat")
    _edit(
        path,
        "--------------------- POINTS ---\n",
        _EQUIVALENT + rows + "--------------------- POINTS ---\n",
    )
    return path


def test_equivalent_buoyancy_replaces_diameter_and_mass(tmp_path):
    """CableDyn's ``apply_equivalent_buoyancy``: ``Diam`` becomes the row's and
    the mass ``rhoW pi D^2/4 + w/g``; the LINE TYPES values are kept apart.

    Checked against CableDyn 0.4.2: the ``lw24`` deck with the row
    ``buoy 0.30 -100.0`` and the same deck with ``buoy`` written as
    ``Diam 0.30`` and this mass give byte-identical ``.static.out``,
    ``.out`` and line files.
    """
    path = _equivalent(tmp_path, "cable 0.30 -100.0\n")
    line_type = read_cabledyn_deck(path).line_type("cable")
    mass = 1025.0 * math.pi / 4.0 * 0.30**2 - 100.0 / 9.81
    assert line_type.diameter_m == 0.30
    assert line_type.mass_kg_m == pytest.approx(mass, rel=1e-15)
    assert (line_type.line_types_diameter_m, line_type.line_types_mass_kg_m) == (
        0.2,
        40.0,
    )
    assert line_type.equivalent_submerged_weight_N_m == -100.0
    plain = read_cabledyn_deck(_deck(tmp_path / "p.dat")).line_type("cable")
    assert (plain.diameter_m, plain.line_types_diameter_m) == (0.2, 0.2)
    assert plain.equivalent_submerged_weight_N_m is None
    # Without g and rhoW in the deck, CableDyn's defaults resolve it.
    _edit(path, "9.81 g - gravity\n1025.0 rhoW - density\n", "")
    default = read_cabledyn_deck(path).line_type("cable")
    assert default.mass_kg_m == pytest.approx(
        1025.0 * math.pi / 4.0 * 0.30**2 - 100.0 / 9.80665, rel=1e-15
    )


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        ("cable 0.30\n", "needs 3 columns"),
        ("rope 0.30 -100.0\n", "unknown LineType 'rope'"),
        ("cable 0.30 -100.0\nCABLE 0.31 -90.0\n", "duplicate EQUIVALENT BUOYANCY"),
        ("cable 0.0 -100.0\n", "needs Diam > 0"),
        ("cable 0.1 -1000.0\n", "negative dry mass"),
        ("cable 0.3 x\n", "submerged weight of cable is not numeric"),
    ],
)
def test_equivalent_buoyancy_errors(tmp_path, rows, match):
    with pytest.raises(CableSchemaError, match=match):
        read_cabledyn_deck(_equivalent(tmp_path, rows))


# --------------------------------------------------------------------------- #
# Which lines CableDyn solves in torsion                                      #
# --------------------------------------------------------------------------- #
def _one_end(tmp_path, *, row=None, end="A", extra=""):
    """The synthetic line restrained in torsion at one end only."""
    tmp_path.mkdir(exist_ok=True)
    path = _deck(tmp_path / "d.dat", torsion=True)
    _edit(path, _ROW_10, row or _row_14("1.5e4"))
    kept = f"1 {end} Rigid 1 0 0 Rigid 0 1 0 0\n"
    _edit(
        path,
        "1 A Rigid 1 0 0 Rigid 0 1 0 0\n1 B Rigid 1 0 0 Rigid 0 1 0 0\n",
        kept,
    )
    if extra:
        _edit(path, "--------------------- OPTIONS ---\n", extra)
    return path


def test_one_end_torsion_follows_cabledyns_rules(tmp_path):
    """CableDyn ``resolve_one_end_torsion``: the line is solved in torsion
    only when every section gives GJ, End B is Fixed and a dynamic run is
    force-blended; otherwise it runs as without the columns and section
    channels does not ask for Twist (CableDyn refuses Twist channels there)."""
    line = read_cabledyn_deck(_one_end(tmp_path / "a")).line(1)
    assert line.torsion_solved and line.torsion_note == ""
    assert "Twist1N51" in section_channels(_one_end(tmp_path / "b"), {"mid": 5.05})
    no_gj = _one_end(tmp_path / "c", row=_ROW_10)
    line = read_cabledyn_deck(no_gj).line(1)
    assert not line.torsion_solved and "give no GJ" in line.torsion_note
    assert not any(n.startswith("Twist") for n in section_channels(no_gj, {"m": 5.0}))
    blend = _one_end(
        tmp_path / "d",
        extra="--------------------- OPTIONS ---\nFalse alpha_force_blend\n",
    )
    assert "alpha_force_blend" in read_cabledyn_deck(blend).line(1).torsion_note
    _edit(blend, "1.0 TMax - duration\n", "0.0 TMax - duration\n")
    assert read_cabledyn_deck(blend).line(1).torsion_solved  # a static run
    moving = _one_end(tmp_path / "e")
    _edit(moving, "2 Fixed 10 0 -12\n", "2 Coupled 10 0 -12\n")
    line = read_cabledyn_deck(moving).line(1)
    assert not line.torsion_solved and "not a Fixed point" in line.torsion_note
    both = read_cabledyn_deck(_deck(tmp_path / "both.dat", torsion=True)).line(1)
    assert both.torsion_solved and both.torsion_restrained


_RODS = (
    "--------------------- RODS ---\n"
    "ID RodType Type XA YA ZA XB YB ZB NumSegs Outputs\n"
    "1 tube {kind} 0 0 -12 0 0 -10 4 -\n"
)
_BODIES = (
    "--------------------- BODIES ---\n"
    "ID Type X Y Z Rot1 Rot2 Rot3 Mass Vol C33 C44 C55 CdA Ca\n"
    "1 {kind} 0 0 -10 0 0 0 1e5 0 0 0 0 0 0\n"
)


@pytest.mark.parametrize(
    ("end_a", "block", "solved"),
    [
        ("R1B", _RODS.format(kind="Fixed"), True),
        ("R1B", _RODS.format(kind="Anchor"), True),
        ("Rod1B", _RODS.format(kind="Pinned"), False),
        ("R1B", _RODS.format(kind="Vessel"), False),
        ("R1B", _RODS.format(kind="Body1") + _BODIES.format(kind="Rigid6"), True),
        ("3", _BODIES.format(kind="Rigid6"), True),
        ("3", _BODIES.format(kind="Coupled"), True),
        ("3", _BODIES.format(kind="Point3"), False),
    ],
)
def test_restrained_end_a_on_a_rod_or_body(tmp_path, end_a, block, solved):
    """A restrained End A holds the torque on a Fixed rod's end or a Rigid6
    body only (a rod fixed to a body is a point of the body)."""
    path = _one_end(tmp_path)
    _edit(path, "1 1 2 pt\n", f"1 {end_a} 2 pt\n")
    _edit(path, "2 Fixed 10 0 -12\n", "2 Fixed 10 0 -12\n3 Body1 0 0 -10\n")
    _edit(
        path,
        "--------------------- LINES ---\n",
        block + "--------------------- LINES ---\n",
    )
    line = read_cabledyn_deck(path).line(1)
    assert line.torsion_solved == solved, line.torsion_note


def test_free_point_below_a_fixed_node_a_is_not_swapped(tmp_path):
    """CableDyn swaps a line from a Fixed NodeA to a Free/Connect NodeB only
    when NodeA is not above it: a weight hung from an overhead point keeps
    its written order."""
    path = _deck(tmp_path / "d.dat")
    _edit(
        path,
        "1 Coupled 0 0 -10\n2 Fixed 10 0 -12\n",
        "1 Fixed 0 0 -10\n2 Free 0 0 -20\n",
    )
    assert not read_cabledyn_deck(path).line(1).reversed_by_parser
    _edit(path, "2 Free 0 0 -20\n", "2 Connect 0 0 -5\n")
    assert read_cabledyn_deck(path).line(1).reversed_by_parser


def test_line_restrained_at_one_end_needs_twist_in_a_dynamic_run(tmp_path):
    positions = _bending_case(np.full(3, 0.02), np.zeros(3), 3)
    root, deck = _write_case(tmp_path, positions, deck_kwargs={"torsion": True})
    _edit(deck, _ROW_10, _row_14("1.5e4"))
    _edit(deck, "1 B Rigid 1 0 0 Rigid 0 1 0 0\n", "")
    with pytest.raises(CableSchemaError, match="CableDyn solves its torsion"):
        read_cabledyn_sections(root, deck, {"s": 3.95}, pressure="none")
    # Without GJ CableDyn does not solve the torsion: read twist-free.
    _edit(deck, _row_14("1.5e4"), _ROW_10)
    read = read_cabledyn_sections(root, deck, {"s": 3.95}, pressure="none")
    assert read.histories["s"].twist_rad is None
    assert read.checks["torsion"]["solved"] is False
    assert "give no GJ" in read.checks["torsion"]["not_solved_because"]


@pytest.mark.parametrize(
    ("block", "match"),
    [
        (_RODS.replace(" 4 -\n", "\n").replace(" -12 0", " -12"), "RODS row has 8"),
        (_BODIES.format(kind="Rigid6").replace(" 0 0\n", "\n"), "BODIES row has 13"),
    ],
)
def test_rod_and_body_rows_are_counted(tmp_path, block, match):
    path = _one_end(tmp_path)
    _edit(
        path,
        "--------------------- LINES ---\n",
        block + "--------------------- LINES ---\n",
    )
    with pytest.raises(CableSchemaError, match=match):
        read_cabledyn_deck(path)


def test_zero_length_rod_is_the_point_it_collapses_to(tmp_path):
    """A zero-length rod is a point (CableDyn ``collapse_zero_length_rods``):
    a Fixed one holds a restrained End A like a Fixed point; a Free one
    makes End A a Free point, from which a Fixed NodeB below is no swap."""
    path = _one_end(tmp_path)
    _edit(path, "1 1 2 pt\n", "1 R1A 2 pt\n")
    rods = _RODS.format(kind="Fix").replace(" 4 -\n", " 0 -\n")
    _edit(
        path,
        "--------------------- LINES ---\n",
        rods + "--------------------- LINES ---\n",
    )
    line = read_cabledyn_deck(path).line(1)
    assert line.torsion_solved and not line.reversed_by_parser
    _edit(path, "1 tube Fix ", "1 tube Body2 ")
    line = read_cabledyn_deck(path).line(1)
    assert line.torsion_solved  # a point of body 2, which the deck does not define


def test_alpha_force_blend_must_be_logical(tmp_path):
    path = _one_end(
        tmp_path, extra="--------------------- OPTIONS ---\nmaybe alpha_force_blend\n"
    )
    with pytest.raises(CableSchemaError, match="alpha_force_blend 'maybe'"):
        read_cabledyn_deck(path)
