# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""CableDyn range graphs, VIV lift files, run routes and the hang-off frame.

``tests/data/cabledyn/lw24v`` is the ``lw24`` deck with a VIV lift
coefficient ``Cl 0.8``, a 0.3 m/s current, seabed friction and the ``LINES``
flags ``ptVr``, run for 1 s by the CableDyn driver (development build of
CableDyn 6c469fe); ``lw24t.Line1.range.out`` is the range file of the
``lw24g`` deck (End A restrained in torsion) with plain ``EI`` columns, from
the same build. The hang-off cases are synthetic, with known answers.
"""

from __future__ import annotations

import math
import shutil

import numpy as np
import pytest
from test_cabledyn_reader import DATA, _arcs, _bending_case, _write_case

import cablexsec as cx
from cablexsec.analysis.sea_state_batch import SeaStateRun
from cablexsec.exceptions import CableSchemaError
from cablexsec.io.cabledyn import (
    read_cabledyn_deck,
    read_cabledyn_lift,
    read_cabledyn_range_graph,
    read_cabledyn_sections,
    section_channels,
)


def _copy_run(tmp_path, root="lw24v"):
    for path in DATA.glob(f"{root}.*"):
        shutil.copy(path, tmp_path / path.name)
    return tmp_path / root


def _main_channels(path):
    rows = path.read_text(encoding="utf-8").splitlines()
    header = rows[1].split("\t")
    data = np.array([[float(cell) for cell in row.split("\t")] for row in rows[2:]])
    return {name: data[:, i] for i, name in enumerate(header)}


# --------------------------------------------------------------------------- #
# Range graphs                                                                #
# --------------------------------------------------------------------------- #
def test_range_graph_of_a_real_run_matches_the_node_channels():
    """CableDyn's range extremes equal those of the channel histories exactly
    (doc/outputs.rst); the deck nodes sit on the refined mesh by DeckNode."""
    graph = read_cabledyn_range_graph(DATA / "lw24v.Line1.range.out")
    assert (graph.line_id, graph.n_samples, graph.time_window_s) == (1, 11, (0.0, 1.0))
    assert list(graph.quantities) == [
        "tension",
        "curvature",
        "bend_moment",
        "declination",
        "clearance",
    ]
    assert graph.node.tolist() == list(range(1, 84))
    assert graph.arc_length_m[0] == 0.0 and np.all(np.diff(graph.arc_length_m) > 0)
    lift = read_cabledyn_lift(DATA / "lw24v.Line1.V.out")
    channels = _main_channels(DATA / "lw24v.out")
    for deck_node in (3, 5, 8, 12):
        row = lift.deck_nodes.index(deck_node)
        for name, quantity in (("Curv", "curvature"), ("Ten", "tension")):
            history = channels[f"{name}1N{deck_node}"]
            assert graph.quantity(quantity).maximum[row] == history.max()
            assert graph.quantity(quantity).minimum[row] == history.min()
            assert graph.quantity(quantity).mean[row] == pytest.approx(
                history.mean(), rel=1e-6
            )
    assert graph.quantity("curvature").unit == "1/m"
    with pytest.raises(CableSchemaError, match="no 'torque' range"):
        graph.quantity("torque")
    assert graph.file.role == "range" and len(graph.file.sha256) == 64


def test_range_graph_of_a_line_with_torsion():
    graph = read_cabledyn_range_graph(DATA / "lw24t.Line1.range.out")
    assert list(graph.quantities)[-2:] == ["torque", "twist"]
    twist = graph.quantity("twist")
    assert twist.unit == "deg" and twist.minimum[0] == 0.0 == twist.maximum[0]
    assert np.all(twist.minimum <= twist.mean) and np.all(twist.mean <= twist.maximum)
    assert not graph.quantity("torque").maximum.flags.writeable


def _range_text():
    return (DATA / "lw24v.Line1.range.out").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("CableDyn range graph (line 1;", "CableDyn ranges (line 1;", "not a CableDyn"),
        ("(line 1; 11 samples", "(line 2; 11 samples", "file name line 1"),
        ("11 samples from t =  0.0", "1 samples from t =  0.0", "does not hold 1"),
        ("TensionMin\tTensionMax", "TensionMax\tTensionMin", "column 3"),
        ("\tClearanceMin\tClearanceMax\tClearanceMean", "", "units row"),
        ("(-)\t(m)\t(N)", "(-)\t(m)\t(kN)", "units row"),
        ("\n2\t", "\n3\t", "number the nodes 1, 2"),
        ("\n1\t 0.0000000E+000", "\n1\t 1.0000000E-003", "ArcLength must start at 0"),
        ("\n1\t 0.0000000E+000", "\n1\tnan", "ArcLength is not finite"),
        ("\n1\t 0.0000000E+000", "\n1\t 0.0000000E+000\t1", "18 cells, the header 17"),
        ("\n1\t", "\n1.0\t", "Node is not an integer"),
    ],
)
def test_range_graph_errors_name_the_item(tmp_path, old, new, match):
    text = _range_text()
    assert old in text, old
    path = tmp_path / "lw24v.Line1.range.out"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(CableSchemaError, match=match):
        read_cabledyn_range_graph(path)


def test_range_graph_order_of_min_mean_max_is_checked(tmp_path):
    rows = _range_text().splitlines()
    cells = rows[3].split("\t")
    cells[4] = f" {2 * float(cells[3]):.7E}"  # mean twice the maximum
    rows[3] = "\t".join(cells)
    path = tmp_path / "r.Line1.range.out"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(CableSchemaError, match="Tension at node 1.*not ordered"):
        read_cabledyn_range_graph(path)
    with pytest.raises(FileNotFoundError, match="range file not found"):
        read_cabledyn_range_graph(tmp_path / "absent.range.out")


def test_range_graph_reads_crlf_and_trailing_blank_lines(tmp_path):
    path = tmp_path / "c.Line1.range.out"
    path.write_bytes(_range_text().replace("\n", "\r\n").encode() + b"\r\n\r\n")
    graph = read_cabledyn_range_graph(path)
    assert graph.node.size == 83


def test_sections_reader_checks_the_range_file_against_the_run(tmp_path):
    root = _copy_run(tmp_path)
    read = read_cabledyn_sections(
        root, root.with_suffix(".dat"), {"s": 10.0}, curvature_check_rtol=0.5
    )
    assert read.checks["range_graph"]["samples"] == 11
    assert {f.role for f in read.files} >= {"range", "lift"}
    path = tmp_path / "lw24v.Line1.range.out"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("11 samples", "10 samples", 1), encoding="utf-8")
    with pytest.raises(CableSchemaError, match="files from different runs"):
        read_cabledyn_sections(
            root, root.with_suffix(".dat"), {"s": 10.0}, curvature_check_rtol=0.5
        )
    rows = text.splitlines()
    path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")  # one node less
    with pytest.raises(CableSchemaError, match="different numbers of nodes"):
        read_cabledyn_sections(
            root, root.with_suffix(".dat"), {"s": 10.0}, curvature_check_rtol=0.5
        )


# --------------------------------------------------------------------------- #
# VIV lift                                                                    #
# --------------------------------------------------------------------------- #
def test_lift_file_of_a_real_run():
    lift = read_cabledyn_lift(DATA / "lw24v.Line1.V.out")
    assert (lift.line_id, lift.n_nodes, lift.force_N.shape) == (1, 83, (11, 83, 3))
    assert lift.time_s[0] == 0.0 and lift.time_s[-1] == 1.0
    assert lift.deck_nodes is not None and lift.deck_nodes[0] == 1
    assert lift.deck_nodes[-1] == 25
    assert np.abs(lift.force_N).max() > 1.0  # a current drives the lift
    some = read_cabledyn_lift(DATA / "lw24v.Line1.V.out", nodes=[83, 2])
    assert some.nodes == (83, 2)
    np.testing.assert_array_equal(some.force_N[:, 0], lift.force_N[:, 82])
    np.testing.assert_array_equal(some.force_N[:, 1], lift.force_N[:, 1])
    small = read_cabledyn_lift(DATA / "lw24v.Line1.V.out", chunk_rows=16)
    np.testing.assert_array_equal(small.force_N, lift.force_N)
    assert not lift.force_N.flags.writeable


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"nodes": [0]}, "distinct integers from 1 to 83"),
        ({"nodes": [1, 1]}, "distinct integers"),
        ({"nodes": [84]}, "distinct integers"),
        ({"nodes": [True]}, "distinct integers"),
        ({"chunk_rows": 4}, "chunk_rows"),
    ],
)
def test_lift_arguments_are_checked(kwargs, match):
    with pytest.raises(CableSchemaError, match=match):
        read_cabledyn_lift(DATA / "lw24v.Line1.V.out", **kwargs)


def _lift_text():
    return (DATA / "lw24v.Line1.V.out").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("VIV lift force at the nodes", "VIV lift at the nodes", "not a CableDyn VIV"),
        ("Node1LiftY(N)", "Node1LiftZ(N)", "column 3"),
        ("# DeckNode\t1\t", "# DeckNode\t2\t", "does not number the deck nodes"),
        ("# DeckNode\t1\t", "# DeckNode\t1\t7\t", "entries for 83 nodes"),
        (
            "\n  1.0000000000000001E-001\t",
            "\n  1.5000000000000001E-001\t",
            "not uniform",
        ),
        (
            "\n  1.0000000000000001E-001\t",
            "\n  1.0000000000000001E-001\tabc\t",
            "malformed row",
        ),
    ],
)
def test_lift_file_errors_name_the_item(tmp_path, old, new, match):
    text = _lift_text()
    assert old in text, old
    path = tmp_path / "lw24v.Line1.V.out"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(CableSchemaError, match=match):
        read_cabledyn_lift(path)


def test_lift_file_of_another_mesh_is_refused_by_the_sections_reader(tmp_path):
    root = _copy_run(tmp_path)
    path = tmp_path / "lw24v.Line1.V.out"
    rows = path.read_text(encoding="utf-8").splitlines()
    rows[1] = "\t".join(rows[1].split("\t")[:-1])  # 82 DeckNode entries
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(CableSchemaError, match="82 entries for 83 nodes"):
        read_cabledyn_sections(
            root, root.with_suffix(".dat"), {"s": 10.0}, curvature_check_rtol=0.5
        )


# --------------------------------------------------------------------------- #
# Run routes: a standalone run ends at TMax; OpenFAST rounds dtM              #
# --------------------------------------------------------------------------- #
def test_standalone_static_deck_is_refused_with_its_reason(tmp_path):
    positions = _bending_case(np.full(3, 0.02), np.zeros(3), 3)
    root, deck = _write_case(tmp_path, positions)
    text = deck.read_text(encoding="utf-8")
    deck.write_text(text.replace("0.2 TMax - duration\n", ""), encoding="utf-8")
    with pytest.raises(CableSchemaError, match="there is no TMax, so the standalone"):
        read_cabledyn_sections(root, deck, {"s": 5.0}, pressure="none")
    deck.write_text(text.replace("0.2 TMax", "0.0 TMax"), encoding="utf-8")
    with pytest.raises(CableSchemaError, match="TMax is 0"):
        read_cabledyn_sections(root, deck, {"s": 5.0}, pressure="none")


def test_standalone_run_that_stopped_early_is_refused(tmp_path):
    positions = _bending_case(np.full(3, 0.02), np.zeros(3), 3)
    root, deck = _write_case(tmp_path, positions, deck_kwargs={"t_max": 1.0})
    with pytest.raises(CableSchemaError, match="the run stopped early"):
        read_cabledyn_sections(root, deck, {"s": 5.0}, pressure="none")


def _openfast_case(tmp_path, step, *, dt="0.1", t_max=None):
    """An OpenFAST CableDyn table on ``step`` and a deck with ``dt``."""
    nt = 3
    positions = _bending_case(np.full(nt, 0.02), np.zeros(nt), nt)
    names = [f"L1N{j}p{axis}" for j in range(1, 102) for axis in "xyz"]
    names += ["Ten1N50", "Ten1N51"]
    root, deck = _write_case(tmp_path, positions, root="of", outputs="-", static=False)
    text = deck.read_text(encoding="utf-8")
    text = text.replace("0.1 dtM - step\n", f"{dt} dtM - step\n")
    text = text.replace(
        "0.2 TMax - duration\n", "" if t_max is None else f"{t_max} TMax\n"
    )
    deck.write_text(text, encoding="utf-8")
    (tmp_path / "of.out").unlink()
    with open(tmp_path / "of.CD.out", "w", encoding="utf-8") as f:
        f.write("\t".join(["Time", *names]) + "\n")
        f.write(
            "\t".join(
                ["(s)", *["(N)" if n.startswith("Ten") else "(m)" for n in names]]
            )
            + "\n"
        )
        for k in range(nt):
            values = [*positions[k].ravel(), 1000.0, 1000.0]
            f.write("\t".join(f"{v: .16E}" for v in [step * k, *values]) + "\n")
    return tmp_path / "of", deck


def test_openfast_run_on_a_rounded_step_is_read(tmp_path):
    """OpenFAST rounds dtM to whole glue steps and ignores the deck's TMax
    (CableDyn doc/options.rst): neither is held against the rows."""
    root, deck = _openfast_case(tmp_path, 0.1, dt="0.08", t_max="0.0")
    read = read_cabledyn_sections(root, deck, {"s": 4.95}, pressure="none")
    assert read.checks["route"] == "OpenFAST (.CD.out)"
    assert read.checks["deck_dtM_s"] == 0.08 and read.checks["time_step_s"] == 0.1


# --------------------------------------------------------------------------- #
# The hang-off frame                                                          #
# --------------------------------------------------------------------------- #
_DIP = math.radians(60.0)  # the hang-off tangent dips 60 deg below horizontal
_T0 = np.array([math.cos(_DIP), 0.0, -math.sin(_DIP)])
_N0 = np.array([math.sin(_DIP), 0.0, math.cos(_DIP)])  # bends upward in x-z
_ORIGIN = np.array([0.0, 0.0, -10.0])


def _yawing_case(yaw_deg, kappa=0.05):
    """An arc from a dipping hang-off, turned rigidly about the vertical
    through End A by ``yaw_deg``: the material frame turns with it."""
    s = _arcs()
    theta = kappa * s
    arc = (np.sin(theta)[:, None] * _T0 + (1 - np.cos(theta))[:, None] * _N0) / kappa
    positions = np.empty((yaw_deg.size, s.size, 3))
    for k, yaw in enumerate(np.radians(yaw_deg)):
        c, d = math.cos(yaw), math.sin(yaw)
        turn = np.array([[c, -d, 0.0], [d, c, 0.0], [0.0, 0.0, 1.0]])
        positions[k] = _ORIGIN + arc @ turn.T
    end = np.cos(kappa * s[-1]) * _T0 + np.sin(kappa * s[-1]) * _N0
    return positions, end


def _body_deck(deck, end_b, *, ends="AB"):
    """End A on a Rigid6 body, restrained in torsion (and End B if asked)."""
    text = deck.read_text(encoding="utf-8")
    text = text.replace("1 Coupled 0 0 -10\n", "1 Body1 0 0 -10\n")
    text = text.replace(
        "--------------------- POINTS ---\n",
        "--------------------- BODIES ---\n"
        "ID Type X Y Z Rot1 Rot2 Rot3 Mass Vol C33 C44 C55 CdA Ca\n"
        "1 Rigid6 0 0 -10 0 0 0 1e5 0 0 0 0 0 0\n"
        "--------------------- POINTS ---\n",
    )
    rows = ""
    if "A" in ends:
        rows += "1 A Rigid {} {} {} Rigid 0 1 0 0\n".format(*_T0)
    if "B" in ends:
        rows += "1 B Rigid {} {} {} Rigid 0 1 0 0\n".format(*end_b)
    text = text.replace(
        "1 A Rigid 1 0 0 Rigid 0 1 0 0\n1 B Rigid 1 0 0 Rigid 0 1 0 0\n", rows
    )
    deck.write_text(text, encoding="utf-8")
    return deck


def _turning_body(tmp_path, yaw_deg):
    positions, end = _yawing_case(yaw_deg)
    nt = yaw_deg.size
    channels = {f"Twist1N{j}": np.zeros(nt) for j in (61, 62)}
    channels.update(
        {"Body1Rx": np.zeros(nt), "Body1Ry": np.zeros(nt), "Body1Rz": yaw_deg}
    )
    root, deck = _write_case(
        tmp_path, positions, channels=channels, deck_kwargs={"torsion": True}
    )
    return root, _body_deck(deck, end)


@pytest.mark.analytical_sanity
def test_hang_off_on_a_turning_body_carries_the_material_frame(tmp_path):
    """A line turned rigidly with the body that holds its restrained End A
    keeps its curvature components in the material frame; the twist-free
    transport in time would turn them by the yaw times the tangent's
    vertical component (here -0.87 x 40 deg)."""
    yaw = np.linspace(0.0, 40.0, 9)
    root, deck = _turning_body(tmp_path, yaw)
    read = read_cabledyn_sections(root, deck, {"s": 6.05}, pressure="none")
    history = read.histories["s"]
    assert read.checks["hang_off"]["source"] == "Body1Rx, Body1Ry, Body1Rz channels"
    assert {"Body1Rx", "Body1Ry", "Body1Rz"} <= set(section_channels(deck, {"s": 6.05}))
    for item in ("rotation_curvature_x_1_m", "rotation_curvature_y_1_m"):
        values = getattr(history, item)
        np.testing.assert_allclose(values, values[0], rtol=0.0, atol=1e-9)
    assert abs(history.rotation_curvature_x_1_m[0]) == pytest.approx(0.05, rel=1e-6)
    # The same turn given as attitude matrices.
    turns = np.array(
        [
            [
                [math.cos(a), -math.sin(a), 0.0],
                [math.sin(a), math.cos(a), 0.0],
                [0, 0, 1],
            ]
            for a in np.radians(yaw)
        ]
    )
    given = read_cabledyn_sections(
        root, deck, {"s": 6.05}, pressure="none", hang_off_rotation=turns
    ).histories["s"]
    np.testing.assert_allclose(
        given.rotation_curvature_x_1_m, history.rotation_curvature_x_1_m, atol=1e-12
    )
    # Declared still, the frame follows the End A tangent alone and turns.
    still = read_cabledyn_sections(
        root, deck, {"s": 6.05}, pressure="none", hang_off_rotation="none"
    ).histories["s"]
    assert np.max(np.abs(still.rotation_curvature_x_1_m)) > 0.02


@pytest.mark.analytical_sanity
def test_given_roll_of_the_hang_off_turns_the_section_axes(tmp_path):
    """A static line with a hang-off rolled by psi about its tangent: the
    components turn as (kx + i ky) e^{-i psi}."""
    nt = 5
    kappa = np.full(nt, 0.02)
    positions = _bending_case(kappa, np.zeros(nt), nt)
    channels = {f"Twist1N{j}": np.zeros(nt) for j in (40, 41)}
    root, deck = _write_case(
        tmp_path, positions, channels=channels, deck_kwargs={"torsion": True}
    )
    roll = np.array([10.0, 10.0, 40.0, 70.0, -20.0])  # the first row is the datum
    history = read_cabledyn_sections(
        root, deck, {"s": 3.95}, pressure="none", hang_off_rotation=roll
    ).histories["s"]
    psi = np.radians(roll - roll[0])
    np.testing.assert_allclose(
        history.rotation_curvature_x_1_m, kappa * np.cos(psi), atol=2e-6
    )
    np.testing.assert_allclose(
        history.rotation_curvature_y_1_m, -kappa * np.sin(psi), atol=2e-6
    )


@pytest.mark.analytical_sanity
def test_free_end_a_turns_against_the_restrained_end_b(tmp_path):
    """Restrained at End B only, the free End A turns by -Twist<L> against
    End B's frame carried along the line (CableDyn doc/conventions.rst): a
    still line whose total twist grows turns its section axes by
    Twist(s) - Twist<L>."""
    nt = 4
    kappa = np.full(nt, 0.02)
    positions = _bending_case(kappa, np.zeros(nt), nt)
    total = np.array([0.0, 20.0, 50.0, 90.0])
    local = np.array([0.0, 5.0, 10.0, 15.0])
    channels = {f"Twist1N{j}": local for j in (40, 41)}
    channels["Twist1"] = total
    root, deck = _write_case(
        tmp_path, positions, channels=channels, deck_kwargs={"torsion": True}
    )
    text = deck.read_text(encoding="utf-8")
    text = text.replace("1 A Rigid 1 0 0 Rigid 0 1 0 0\n", "")
    text = text.replace(
        "cable 0.2 40.0 500000000.0 0.0 20000.0 1.2 0.1 1.0 0.0\n",
        "cable 0.2 40.0 5e8 0.0 2e4 1.9e8 1.5e4 0.1 0.2 1.2 0.1 1.0 0.0\n",
    )
    end_tangent = (math.cos(0.2), 0.0, -math.sin(0.2))  # kappa L = 0.2
    text = text.replace(
        "1 B Rigid 1 0 0 Rigid 0 1 0 0\n",
        "1 B Rigid {} {} {} Rigid 0 1 0 0\n".format(*end_tangent),
    )
    deck.write_text(text, encoding="utf-8")
    line = read_cabledyn_deck(deck).line(1)
    assert line.torsion_solved and line.torsion_restrained_ends == {"b"}
    read = read_cabledyn_sections(root, deck, {"s": 3.95}, pressure="none")
    assert read.checks["hang_off"]["mode"] == "end_b"
    assert "Twist1" in section_channels(deck, {"s": 3.95})
    theta = np.radians(local - total)
    history = read.histories["s"]
    np.testing.assert_allclose(
        history.rotation_curvature_x_1_m, 0.02 * np.cos(theta), atol=2e-6
    )
    np.testing.assert_allclose(
        history.rotation_curvature_y_1_m, -0.02 * np.sin(theta), atol=2e-6
    )
    with pytest.raises(
        CableSchemaError, match="is solved in torsion: its section axes"
    ):
        read_cabledyn_sections(
            root, deck, {"s": 3.95}, pressure="none", frame="section"
        )
    with pytest.raises(CableSchemaError, match="End B only, so End A turns"):
        read_cabledyn_sections(
            root, deck, {"s": 3.95}, pressure="none", hang_off_rotation="none"
        )


def test_hang_off_errors_name_the_remedy(tmp_path):
    yaw = np.linspace(0.0, 10.0, 4)
    root, deck = _turning_body(tmp_path, yaw)
    sections = {"s": 6.05}
    with pytest.raises(
        CableSchemaError, match="is solved in torsion: its section axes"
    ):
        read_cabledyn_sections(root, deck, sections, pressure="none", frame="section")
    with pytest.raises(CableSchemaError, match="has 3 rows"):
        read_cabledyn_sections(
            root, deck, sections, pressure="none", hang_off_rotation=np.zeros(3)
        )
    with pytest.raises(CableSchemaError, match="has 5 rows"):
        read_cabledyn_sections(
            root, deck, sections, pressure="none", hang_off_rotation=np.zeros(5)
        )
    with pytest.raises(CableSchemaError, match="must be rotations"):
        read_cabledyn_sections(
            root, deck, sections, pressure="none", hang_off_rotation=np.ones((4, 3, 3))
        )
    for value, match in (
        ("still", "must be None, 'none'"),
        (np.zeros((4, 2)), "one per output row"),
        ([0.0, math.nan, 0.0, 0.0], "finite"),
    ):
        with pytest.raises(CableSchemaError, match=match):
            read_cabledyn_sections(
                root, deck, sections, pressure="none", hang_off_rotation=value
            )
    # Without the body's attitude channels the material frame is unknown.
    text = (tmp_path / "case.out").read_text(encoding="utf-8")
    (tmp_path / "case.out").write_text(
        text.replace("Body1Rz", "Body1Fz"), encoding="utf-8"
    )
    with pytest.raises(CableSchemaError, match="write Body1Rx, Body1Ry, Body1Rz"):
        read_cabledyn_sections(root, deck, sections, pressure="none")


def test_hang_off_on_a_vessel_needs_its_rotation(tmp_path):
    positions = _bending_case(np.full(3, 0.02), np.zeros(3), 3)
    channels = {f"Twist1N{j}": np.zeros(3) for j in (40, 41)}
    root, deck = _write_case(
        tmp_path, positions, channels=channels, deck_kwargs={"torsion": True}
    )
    with pytest.raises(CableSchemaError, match="on a 'coupled' point"):
        read_cabledyn_sections(root, deck, {"s": 3.95}, pressure="none")
    # Not restrained in torsion: the holder's turn does not reach the cable.
    root, deck = _write_case(tmp_path, positions, root="free")
    with pytest.raises(CableSchemaError, match="End A is free in torsion"):
        read_cabledyn_sections(
            root, deck, {"s": 3.95}, pressure="none", hang_off_rotation="none"
        )


def test_sea_state_run_takes_the_hang_off_rotation(tmp_path):
    run = SeaStateRun(
        "r", tmp_path / "o", tmp_path / "d.dat", 1.0, hang_off_rotation=[1.0, 2.0]
    )
    assert run.hang_off_rotation.tolist() == [1.0, 2.0]
    assert (
        SeaStateRun(
            "s", tmp_path / "o", tmp_path / "d.dat", 1.0, hang_off_rotation="none"
        ).hang_off_rotation
        == "none"
    )
    with pytest.raises(CableSchemaError, match="hang_off_rotation must be"):
        SeaStateRun(
            "t", tmp_path / "o", tmp_path / "d.dat", 1.0, hang_off_rotation="off"
        )


def test_new_readers_are_exported():
    for name in ("read_cabledyn_range_graph", "read_cabledyn_lift"):
        assert name in cx.__all__ and callable(getattr(cx, name))


def test_sea_state_options_take_the_curvature_check_tolerances():
    """A coarse run fails the reader's curvature check at its default
    tolerance; the batch could not relax it."""
    from cablexsec.analysis.sea_state_batch import SeaStateBatchOptions
    from cablexsec.exceptions import CableInputError

    options = SeaStateBatchOptions(curvature_check_rtol=0.5)
    assert options.curvature_check_rtol == 0.5
    for value in (math.nan, -0.1, True):
        with pytest.raises(CableInputError, match="curvature_check_rtol"):
            SeaStateBatchOptions(curvature_check_rtol=value)


def _range_file(path, n_nodes, n_samples, t_last):
    head = ["Node", "ArcLength"] + [
        f"{q}{s}"
        for q in ("Tension", "Curvature", "BendMoment", "Declination")
        for s in ("Min", "Max", "Mean")
    ]
    units = ["(-)", "(m)"] + [
        u for u in ("(N)", "(1/m)", "(N.m)", "(deg)") for _ in range(3)
    ]
    rows = [
        "\t".join([str(j), f" {0.1 * (j - 1):.7E}"] + [" 1.0000000E+000"] * 12)
        for j in range(1, n_nodes + 1)
    ]
    path.write_text(
        f"CableDyn range graph (line 1; {n_samples} samples from t =  "
        f"0.0000000000000000E+000 s to t =  {t_last:.16E} s; public node order "
        "End A -> End B)\n"
        + "\t".join(head)
        + "\n"
        + "\t".join(units)
        + "\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )


def test_range_file_alone_does_not_require_a_refined_mesh_map(tmp_path):
    """A read from main-table channels needs only the deck nodes: a range file
    of a refined mesh, which names no deck nodes, is not held against the
    deck there; with another mesh file it joins the node-count check."""
    nt = 3
    positions = _bending_case(np.full(nt, 0.02), np.zeros(nt), nt)
    channels = {
        f"L1N{j + 1}p{axis}": positions[:, j, k]
        for j in range(positions.shape[1])
        for k, axis in enumerate("xyz")
    }
    channels.update({"Ten1N50": np.full(nt, 1e3), "Ten1N51": np.full(nt, 1e3)})
    root, deck = _write_case(
        tmp_path, positions, channels=channels, outputs="-", static=False
    )
    _range_file(tmp_path / "case.Line1.range.out", 106, nt, 0.2)
    read = read_cabledyn_sections(root, deck, {"s": 4.95}, pressure="none")
    assert read.checks["range_graph"]["mesh_checked"] is False
    assert read.checks["range_graph"]["mesh_nodes"] == 106
    root, deck = _write_case(
        tmp_path, positions, channels=channels, outputs="-", root="with_static"
    )
    _range_file(tmp_path / "with_static.Line1.range.out", 106, nt, 0.2)
    with pytest.raises(CableSchemaError, match="different numbers of nodes"):
        read_cabledyn_sections(root, deck, {"s": 4.95}, pressure="none")


# --------------------------------------------------------------------------- #
# A coupled OpenFAST run of a line restrained in torsion                      #
# --------------------------------------------------------------------------- #
# ``vu_torsion``: the lazy-wave power cable of CableDyn's mixed VolturnUS-S
# example deck alone, linear EI, restrained in torsion at both ends (End A on
# the platform, End B on the anchor), 1 s of OpenFAST v5.0.0 with a development
# build of CableDyn after 0.4.2 (CompMooring = 5; the deck's dtM 0.025 s runs
# as 0.02 s, two glue steps). Its CurvX/CurvY channels are CableDyn's own
# components on its section axes.
_VU = {"upper": 21.0, "mid": 59.0, "buoy": 131.0}


def _vu_copy(tmp_path):
    for suffix in ("dat", "CD.out", "CD.static.out"):
        shutil.copy(DATA / f"vu_torsion.{suffix}", tmp_path / f"vu.{suffix}")
    return tmp_path / "vu", tmp_path / "vu.dat"


def _vu_components():
    channels = _main_channels_units(DATA / "vu_torsion.CD.out")
    arcs = read_cabledyn_deck(DATA / "vu_torsion.dat").line(1).node_arc_lengths()
    out = {}
    for name, arc in _VU.items():
        e = int(np.searchsorted(arcs, arc, side="right")) - 1
        w = (arc - arcs[e]) / (arcs[e + 1] - arcs[e])
        out[name] = tuple(
            (1 - w) * channels[f"{kind}1N{e + 1}"] + w * channels[f"{kind}1N{e + 2}"]
            for kind in ("CurvX", "CurvY")
        )
    return out


def _main_channels_units(path):
    rows = path.read_text(encoding="utf-8").splitlines()
    header = rows[0].split("\t")
    data = np.array([[float(cell) for cell in row.split("\t")] for row in rows[2:]])
    return {name: data[:, i] for i, name in enumerate(header)}


def test_openfast_root_is_read_next_to_the_glue_table(tmp_path):
    """OpenFAST always writes its glue-code ``<root>.out`` beside CableDyn's
    ``<root>.CD.out``; only a CableDyn driver table there is ambiguous."""
    root, deck = _vu_copy(tmp_path)
    (tmp_path / "vu.out").write_text(
        "\nPredictions were generated on 19-Nov-2026 using OpenFAST\n\nTime\tWind1VelX\n",
        encoding="utf-8",
    )
    read = read_cabledyn_sections(
        root, deck, {"s": 21.0}, curvature_source="components", pressure="none"
    )
    assert read.checks["route"] == "OpenFAST (.CD.out)"
    assert {f.role for f in read.files} == {"main", "static", "deck"}
    (tmp_path / "vu.out").write_text(
        "# CableDyn driver output\nTime(s)\tFairTen1\n", encoding="utf-8"
    )
    with pytest.raises(CableSchemaError, match="a CableDyn driver table"):
        read_cabledyn_sections(root, deck, {"s": 21.0}, pressure="none")


@pytest.mark.full_scale
def test_torsion_line_reconstruction_uses_cabledyns_section_axes():
    """On a line solved in torsion the reconstructed components agree with
    CableDyn's CurvX/CurvY within the reader's own curvature tolerance (5 %
    of the largest value plus 2e-5 1/m): the axes start at End A's torsion
    normal, not at the reference vector, whose axes stood 90 deg away here."""
    expected = _vu_components()
    read = read_cabledyn_sections(
        DATA / "vu_torsion",
        DATA / "vu_torsion.dat",
        _VU,
        curvature_source="tangent",
        pressure="none",
        hang_off_rotation="none",  # the platform turns by under 0.01 deg in 1 s
    )
    assert read.checks["time_step_s"] == pytest.approx(0.02)
    for name, (curv_x, curv_y) in expected.items():
        gx, gy = read.histories[name].strain_gradient()
        scale = float(np.max(np.hypot(curv_x, curv_y)))
        error = float(np.max(np.hypot(gx + curv_x, gy + curv_y)))
        assert error <= 2e-5 + 0.05 * scale, (name, error, scale)
        assert error <= 0.02 * scale  # 0.3 to 1.2 % in this run


def test_components_route_takes_cabledyns_curvature_components():
    expected = _vu_components()
    read = read_cabledyn_sections(
        DATA / "vu_torsion",
        DATA / "vu_torsion.dat",
        _VU,
        curvature_source="components",
        pressure="none",
    )
    for name, (curv_x, curv_y) in expected.items():
        history = read.histories[name]
        assert history.frame == "cabledyn section axes"
        np.testing.assert_array_equal(history.rotation_curvature_x_1_m, -curv_y)
        np.testing.assert_array_equal(history.rotation_curvature_y_1_m, curv_x)
        gx, gy = history.strain_gradient()
        np.testing.assert_array_equal(gx, -curv_x)
        assert history.twist_rad is not None
    with pytest.raises(CableSchemaError, match="already carry the hang-off"):
        read_cabledyn_sections(
            DATA / "vu_torsion",
            DATA / "vu_torsion.dat",
            _VU,
            curvature_source="components",
            hang_off_rotation="none",
        )


def test_torsion_line_refuses_the_section_frame():
    with pytest.raises(CableSchemaError, match="curvature_source='components'"):
        read_cabledyn_sections(
            DATA / "vu_torsion",
            DATA / "vu_torsion.dat",
            {"s": 21.0},
            frame="section",
            curvature_source="tangent",
            pressure="none",
            hang_off_rotation="none",
        )


def test_components_route_needs_the_component_channels(tmp_path):
    nt = 3
    positions = _bending_case(np.full(nt, 0.02), np.zeros(nt), nt)
    root, deck = _write_case(tmp_path, positions)
    with pytest.raises(CableSchemaError, match="CurvX1N<J> and CurvY1N<J> channels"):
        read_cabledyn_sections(
            root, deck, {"s": 4.95}, curvature_source="components", pressure="none"
        )


def _vu_without(tmp_path, dropped):
    """``vu_torsion`` with the named ``.CD.out`` channels left out."""
    root, deck = _vu_copy(tmp_path)
    path = tmp_path / "vu.CD.out"
    rows = [row.split("\t") for row in path.read_text(encoding="utf-8").splitlines()]
    keep = [i for i, name in enumerate(rows[0]) if name not in dropped]
    path.write_text(
        "\n".join("\t".join(row[i] for i in keep) for row in rows) + "\n",
        encoding="utf-8",
    )
    return root, deck


def test_components_section_on_a_deck_node_reads_that_node_alone(tmp_path):
    """A section on a deck node takes that node's CurvX/CurvY, Ten, Twist and
    Torq without the next node's channels, bit for bit; a section just off
    the node still needs both and is told where the node lies."""
    channels = _main_channels_units(DATA / "vu_torsion.CD.out")
    arcs = read_cabledyn_deck(DATA / "vu_torsion.dat").line(1).node_arc_lengths()
    on_node = float(arcs[10])  # deck node 11
    full = read_cabledyn_sections(
        DATA / "vu_torsion",
        DATA / "vu_torsion.dat",
        {"s": on_node},
        curvature_source="components",
        pressure="none",
    ).histories["s"]
    root, deck = _vu_without(
        tmp_path,
        {f"{kind}1N12" for kind in ("Ten", "Curv", "CurvX", "CurvY", "Twist", "Torq")},
    )
    read = read_cabledyn_sections(
        root, deck, {"s": on_node}, curvature_source="components", pressure="none"
    )
    history = read.histories["s"]
    np.testing.assert_array_equal(
        history.rotation_curvature_x_1_m, -channels["CurvY1N11"]
    )
    np.testing.assert_array_equal(
        history.rotation_curvature_y_1_m, channels["CurvX1N11"]
    )
    np.testing.assert_array_equal(history.effective_tension_N, channels["Ten1N11"])
    np.testing.assert_array_equal(history.twist_rad, np.radians(channels["Twist1N11"]))
    np.testing.assert_array_equal(history.torque_Nm, channels["Torq1N11"])
    for name in (
        "rotation_curvature_x_1_m",
        "effective_tension_N",
        "twist_rad",
        "torque_Nm",
    ):
        np.testing.assert_array_equal(getattr(history, name), getattr(full, name))
    with pytest.raises(CableSchemaError, match="from deck node 11"):
        read_cabledyn_sections(
            root,
            deck,
            {"s": on_node + 1e-6},
            curvature_source="components",
            pressure="none",
        )


def test_components_route_records_the_twist_where_it_is_written(tmp_path):
    """CableDyn's components carry the twist in their axes: a section whose
    nodes write no Twist channel is still read, without a twist history, and
    the other sections keep theirs."""
    arcs = read_cabledyn_deck(DATA / "vu_torsion.dat").line(1).node_arc_lengths()
    root, deck = _vu_without(tmp_path, {"Twist1N30", "Torq1N30"})
    read = read_cabledyn_sections(
        root,
        deck,
        {"upper": float(arcs[10]), "mid": float(arcs[29])},
        curvature_source="components",
        pressure="none",
    )
    assert read.histories["upper"].twist_rad is not None
    assert read.histories["upper"].torque_Nm is not None
    assert read.histories["mid"].twist_rad is None
    assert read.histories["mid"].torque_Nm is None
    assert "'upper'" in read.checks["torsion"]["material_twist"]
    assert "'mid'" not in read.checks["torsion"]["material_twist"]
