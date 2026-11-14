# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""CableDyn output reader: frames, signs, units, failures.

Synthetic files follow the layouts of CableDyn's standalone driver
(doc/outputs.rst): known centrelines with analytic curvature vectors are
written as node positions, chord tangents (``Dec``/``Azi``), element tensions
and curvature magnitudes. ``tests/data/cabledyn`` holds two compact runs of
the CableDyn v0.1.0 driver on the ``lazy_wave_vessel_rao`` example cable
(JONSWAP Hs 4 m, Tp 10 s, 30 deg; RAO-driven hang-off).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from cablexsec.exceptions import CableSchemaError
from cablexsec.io.cabledyn import read_cabledyn_deck, read_cabledyn_sections
from cablexsec.io.time_history_csv import read_time_history_csv

DATA = Path(__file__).parent / "data" / "cabledyn"
LENGTH = 10.0
N_ELEMENTS = 100


def _arcs(n_elements=N_ELEMENTS, length=LENGTH):
    return length * np.arange(n_elements + 1) / n_elements


def _deck(
    path,
    *,
    n_elements=N_ELEMENTS,
    length=LENGTH,
    outputs="pt",
    channels=(),
    torsion=False,
    dt=0.1,
    extra_options="",
    ea=5.0e8,
    ei=2.0e4,
    t_max=1.0,
):
    ends = ""
    if torsion:
        ends = (
            "--------------------- END CONNECTIONS ---\n"
            "LineID End Stiffness EzX EzY EzZ TorsStiffness NxX NxY NxZ Pretwist\n"
            "(-) (-) (N-m/rad) (-) (-) (-) (N-m/rad) (-) (-) (-) (deg)\n"
            "1 A Rigid 1 0 0 Rigid 0 1 0 0\n"
            "1 B Rigid 1 0 0 Rigid 0 1 0 0\n"
        )
    text = (
        "--------------------- CableDyn Input File ---\n"
        "synthetic\n"
        "--------------------- LINE TYPES ---\n"
        "TypeName Diam MassDenInAir EA BA EI Cd_n Cd_t Ca_n Ca_t\n"
        "(-) (m) (kg/m) (N) (-) (N-m2) (-) (-) (-) (-)\n"
        f"cable 0.2 40.0 {ea!r} 0.0 {ei!r} 1.2 0.1 1.0 0.0\n"
        "--------------------- POINTS ---\n"
        "ID Type X Y Z\n(-) (-) (m) (m) (m)\n"
        "1 Coupled 0 0 -10\n2 Fixed 10 0 -12\n"
        "--------------------- LINES ---\n"
        "ID NodeA NodeB Outputs\n(-) (-) (-) (-)\n"
        f"1 1 2 {outputs}\n"
        "--------------------- SECTIONS ---\n"
        "LineID LineType Length NumSegs\n(-) (-) (m) (-)\n"
        f"1 cable {length} {n_elements}\n"
        f"{ends}"
        "--------------------- OPTIONS ---\n"
        "9.81 g - gravity\n1025.0 rhoW - density\n"
        f"{dt} dtM - step\n{t_max} TMax - duration\n{extra_options}"
        "--------------------- OUTPUTS ---\n"
        + "".join(f'"{name}"\n' for name in channels)
        + "--------------------- need this line ---\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


def _fmt(values):
    return "\t".join(f"{v: .16E}" for v in values)


def _write_case(
    tmp_path,
    positions,
    *,
    root="case",
    time=None,
    tension=None,
    channels=None,
    units=False,
    outputs="pt",
    deck_kwargs=None,
    static=True,
):
    """Write deck, main table, per-line tables and static profile.

    positions: (nt, nn, 3); tension: (nt, ne) element tensions.
    channels: dict name -> (nt,) history for the main table.
    """
    nt, nn, _ = positions.shape
    time = np.arange(nt) * 0.1 if time is None else time
    tension = np.full((nt, nn - 1), 5000.0) if tension is None else tension
    channels = {} if channels is None else channels
    deck = _deck(
        tmp_path / f"{root}.dat",
        n_elements=nn - 1,
        outputs=outputs,
        channels=list(channels),
        **(deck_kwargs or {}),
    )
    names = list(channels)
    with open(tmp_path / f"{root}.out", "w", encoding="utf-8") as f:
        f.write(
            "# CableDyn driver output (finite-EI dynamic; production cubic-Hermite route)\n"
        )
        f.write("\t".join(["Time(s)", *names]) + "\n")
        for k in range(nt):
            f.write(_fmt([time[k], *[channels[n][k] for n in names]]) + "\n")
    if "p" in outputs:
        with open(tmp_path / f"{root}.Line1.p.out", "w", encoding="utf-8") as f:
            f.write(
                "# CableDyn finite-EI Hermite line node positions; public node order EndA -> EndB\n"
            )
            f.write(
                "\t".join(
                    ["Time(s)"]
                    + [f"Node{j}{a}(m)" for j in range(1, nn + 1) for a in "XYZ"]
                )
                + "\n"
            )
            for k in range(nt):
                f.write(_fmt([time[k], *positions[k].ravel()]) + "\n")
    if "t" in outputs:
        with open(tmp_path / f"{root}.Line1.t.out", "w", encoding="utf-8") as f:
            f.write(
                "# CableDyn finite-EI Hermite segment tensions; public order EndA -> EndB\n"
            )
            f.write(
                "\t".join(["Time(s)"] + [f"Segment{e}Tension(N)" for e in range(1, nn)])
                + "\n"
            )
            for k in range(nt):
                f.write(_fmt([time[k], *tension[k]]) + "\n")
    if static:
        with open(tmp_path / f"{root}.static.out", "w", encoding="utf-8") as f:
            f.write(
                "CableDyn standalone static configuration (deformed arc; public node order End A -> End B)\n"
            )
            f.write("LineID\tNode\tArcLength\tX\tY\tZ\tTension\tCurvature\n")
            f.write("(-)\t(-)\t(m)\t(m)\t(m)\t(m)\t(N)\t(1/m)\n")
            for j in range(nn):
                f.write(
                    f"1\t{j + 1}\t0.0\t" + _fmt(positions[0, j]) + "\t5000.0\t0.0\n"
                )
    return tmp_path / root, deck


def _arc(curvature, normal, binormal, s, origin=(0.0, 0.0, -10.0), t0=(1.0, 0.0, 0.0)):
    """Circle with start tangent t0, principal normal n and binormal b."""
    t0, n = np.asarray(t0, float), np.asarray(normal, float)
    theta = curvature * s
    r = np.sin(theta)[:, None] * t0 + (1 - np.cos(theta))[:, None] * n
    return np.asarray(origin) + r / curvature


def _chord_tangent_channels(positions, line=1):
    """Dec/Azi of CableDyn's chord tangents (central; one-sided at the ends)."""
    nn = positions.shape[1]
    out = {}
    for j in range(nn):
        a, b = max(j - 1, 0), min(j + 1, nn - 1)
        t = positions[:, b] - positions[:, a]
        t /= np.linalg.norm(t, axis=-1, keepdims=True)
        out[f"L{line}N{j + 1}Dec"] = np.degrees(np.arccos(np.clip(t[:, 2], -1, 1)))
        out[f"L{line}N{j + 1}Azi"] = np.degrees(np.arctan2(t[:, 1], t[:, 0])) % 360.0
    return out


def _bending_case(kappa, phi, nt):
    """Arc of curvature kappa(t) whose binormal turns by phi(t) about t0 = +x."""
    s = _arcs()
    positions = np.empty((nt, s.size, 3))
    for k in range(nt):
        b = np.array([0.0, math.cos(phi[k]), math.sin(phi[k])])
        n = np.cross(b, [1.0, 0.0, 0.0])
        positions[k] = _arc(kappa[k], n, b, s)
    return positions


# --------------------------------------------------------------------------- #
# Frames and signs                                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.analytical_sanity
@pytest.mark.parametrize("source", ["positions", "tangent"])
@pytest.mark.parametrize("frame", ["arc", "section"])
def test_pure_bending_about_each_axis(tmp_path, source, frame):
    """In-plane arc (binormal +y = e2) and horizontal arc (binormal +z = -e1)."""
    nt = 6
    kappa = 0.03 + 0.01 * np.sin(np.arange(nt))
    for phi, expected in ((0.0, (0.0, 1.0)), (math.pi / 2, (-1.0, 0.0))):
        positions = _bending_case(kappa, np.full(nt, phi), nt)
        channels = _chord_tangent_channels(positions) if source == "tangent" else {}
        root, deck = _write_case(
            tmp_path, positions, channels=channels, root=f"c{int(phi * 100)}"
        )
        read = read_cabledyn_sections(
            root,
            deck,
            {"mid": 4.03, "near_a": 0.05},
            curvature_source=source,
            frame=frame,
            pressure="none",
        )
        for history in read.histories.values():
            assert history.curvature_source == source
            assert np.allclose(
                history.rotation_curvature_x_1_m, expected[0] * kappa, atol=2e-6
            )
            assert np.allclose(
                history.rotation_curvature_y_1_m, expected[1] * kappa, atol=2e-6
            )
            assert np.allclose(history.curvature_norm_1_m, kappa, atol=2e-6)


@pytest.mark.analytical_sanity
def test_heading_rotation_leaves_components_unchanged(tmp_path):
    """A rigid rotation about the vertical rotates the seed with the line."""
    nt = 4
    kappa = np.full(nt, 0.04)
    base = _bending_case(kappa, np.zeros(nt), nt)
    psi = math.radians(37.0)
    rot = np.array(
        [
            [math.cos(psi), -math.sin(psi), 0],
            [math.sin(psi), math.cos(psi), 0],
            [0, 0, 1],
        ]
    )
    turned = base @ rot.T
    results = []
    for name, positions in (("base", base), ("turned", turned)):
        root, deck = _write_case(tmp_path, positions, root=name)
        read = read_cabledyn_sections(root, deck, {"mid": 5.0}, pressure="none")
        results.append(read.histories["mid"])
    for item in ("rotation_curvature_x_1_m", "rotation_curvature_y_1_m"):
        assert np.allclose(
            getattr(results[0], item), getattr(results[1], item), atol=1e-9
        )


@pytest.mark.analytical_sanity
def test_bending_plane_rotating_about_the_hang_off_tangent(tmp_path):
    """Arc frame: (kx, ky) = kappa (-sin phi, cos phi) for binormal angle phi."""
    nt = 11
    kappa = np.full(nt, 0.05)
    phi = np.linspace(0.0, math.pi / 2, nt)
    positions = _bending_case(kappa, phi, nt)
    root, deck = _write_case(tmp_path, positions)
    read = read_cabledyn_sections(
        root, deck, {"a": 3.0, "b": 9.0}, frame="arc", pressure="none"
    )
    for history in read.histories.values():
        assert np.allclose(
            history.rotation_curvature_x_1_m, -kappa * np.sin(phi), atol=5e-6
        )
        assert np.allclose(
            history.rotation_curvature_y_1_m, kappa * np.cos(phi), atol=5e-6
        )


@pytest.mark.analytical_sanity
@pytest.mark.parametrize("phi", [0.0, math.pi, math.pi / 2])
def test_tensile_fibre_is_on_the_convex_side(tmp_path, phi):
    """The strain gradient points away from the centre of curvature.

    phi = 0 hogs (centre below), phi = pi sags (centre above), phi = pi/2
    bends horizontally. The material axes are those documented for the
    reader (e2 = z x t normalised, e1 = e2 x t), formed here from the known
    centreline, not from the reader.
    """
    nt = 3
    kappa = np.full(nt, 0.04)
    positions = _bending_case(kappa, np.full(nt, phi), nt)
    root, deck = _write_case(tmp_path, positions)
    arc = 4.0
    history = read_cabledyn_sections(root, deck, {"s": arc}, pressure="none").histories[
        "s"
    ]
    gx, gy = history.strain_gradient()
    angle = np.arctan2(gy[0], gx[0])
    theta = kappa[0] * arc
    b = np.array([0.0, math.cos(phi), math.sin(phi)])
    x0 = np.array([1.0, 0.0, 0.0])
    n0 = np.cross(b, x0)  # principal normal of _arc at End A
    t = math.cos(theta) * x0 + math.sin(theta) * n0
    centre_dir = math.cos(theta) * n0 - math.sin(theta) * x0  # normal at s
    e2 = np.cross([0.0, 0.0, 1.0], t)
    e2 /= np.linalg.norm(e2)
    e1 = np.cross(e2, t)
    tensile = math.cos(angle) * e1 + math.sin(angle) * e2
    assert tensile @ centre_dir == pytest.approx(-1.0, abs=1e-5)
    if phi == math.pi:  # sag bend: tension on the downward fibre, phi = 0
        assert abs(angle) < 1e-3 and tensile[2] < -0.95
    if phi == 0.0:  # hog: tension on the upper fibre, phi = 180 deg
        assert abs(abs(angle) - math.pi) < 1e-3 and tensile[2] > 0.95


@pytest.mark.analytical_sanity
def test_material_twist_turns_the_section_axes(tmp_path):
    nt = 5
    kappa = np.full(nt, 0.02)
    positions = _bending_case(kappa, np.zeros(nt), nt)
    twist_deg = np.linspace(0.0, 60.0, nt)
    channels = {f"Twist1N{j}": twist_deg for j in (40, 41)}
    channels.update({f"Torq1N{j}": np.full(nt, 12.5) for j in (40, 41)})
    root, deck = _write_case(
        tmp_path, positions, channels=channels, deck_kwargs={"torsion": True}
    )
    history = read_cabledyn_sections(
        root, deck, {"s": 3.95}, pressure="none"
    ).histories["s"]
    twist = np.radians(twist_deg)
    assert np.allclose(history.twist_rad, twist)
    assert np.allclose(history.torque_Nm, 12.5)
    assert np.allclose(
        history.rotation_curvature_x_1_m, kappa * np.sin(twist), atol=2e-6
    )
    assert np.allclose(
        history.rotation_curvature_y_1_m, kappa * np.cos(twist), atol=2e-6
    )


def test_torsion_restrained_line_needs_twist_channels(tmp_path):
    positions = _bending_case(np.full(3, 0.02), np.zeros(3), 3)
    root, deck = _write_case(tmp_path, positions, deck_kwargs={"torsion": True})
    with pytest.raises(CableSchemaError, match="restrained in torsion"):
        read_cabledyn_sections(root, deck, {"s": 3.95}, pressure="none")


# --------------------------------------------------------------------------- #
# Tension, pressure, conventions shared with the time-history CSV path        #
# --------------------------------------------------------------------------- #
def test_tension_interpolation_and_hydrostatic_pressure(tmp_path):
    nt = 3
    positions = _bending_case(np.full(nt, 0.02), np.zeros(nt), nt)
    mids = 0.5 * (_arcs()[:-1] + _arcs()[1:])
    tension = np.tile(4000.0 + 100.0 * mids, (nt, 1))
    root, deck = _write_case(tmp_path, positions, tension=tension)
    read = read_cabledyn_sections(root, deck, {"s": 6.37, "end": 10.0})
    history = read.histories["s"]
    assert np.allclose(history.effective_tension_N, 4000.0 + 100.0 * 6.37)
    assert np.allclose(read.histories["end"].effective_tension_N, tension[0, -1])
    z = -10.0 - (1.0 - math.cos(0.02 * 6.37)) / 0.02
    assert np.allclose(history.elevation_m, z, rtol=0.0, atol=1e-7)
    assert np.allclose(history.external_pressure_Pa, 1025.0 * 9.81 * (-z), rtol=1e-9)
    area = math.pi * 0.2**2 / 4
    assert np.allclose(
        history.true_tension_N(area),
        history.effective_tension_N - history.external_pressure_Pa * area,
    )
    assert read.checks["tension_source"].startswith("segment")


def test_csv_round_trip_gives_identical_series_and_damage(tmp_path):
    """The CableDyn history, exported as time-history CSV columns, reads back identically."""
    import cablexsec as cx

    nt = 81
    t = np.arange(nt) * 0.1
    kappa = 0.03 + 0.02 * np.sin(2 * math.pi * t / 2.0)
    phi = 0.4 * np.sin(2 * math.pi * t / 3.0)
    positions = _bending_case(kappa, phi, nt)
    tension = np.tile(
        20000.0 + 3000.0 * np.sin(2 * math.pi * t / 2.5)[:, None], (1, N_ELEMENTS)
    )
    root, deck = _write_case(tmp_path, positions, tension=tension)
    history = read_cabledyn_sections(root, deck, {"s": 2.0}).histories["s"]
    area = math.pi * 0.184**2 / 4
    series = history.to_load_case_series(area)
    csv = tmp_path / "export.csv"
    columns = np.column_stack(
        (
            history.time_s,
            history.rotation_curvature_x_1_m,
            history.rotation_curvature_y_1_m,
            history.true_tension_N(area) / 1e3,
        )
    )
    np.savetxt(
        csv,
        columns,
        delimiter=",",
        header="Time (s),x curvature (rad/m),y curvature (rad/m),Wall tension (kN)",
        comments="",
    )
    via_csv = read_time_history_csv(csv)
    assert np.allclose(
        via_csv.curvature_x_1_m, series.curvature_x_1_m, rtol=1e-12, atol=1e-15
    )
    assert np.allclose(
        via_csv.curvature_y_1_m, series.curvature_y_1_m, rtol=1e-12, atol=1e-15
    )
    assert np.allclose(via_csv.tension_N, series.tension_N, rtol=1e-12)
    cable = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
    a = cx.compute_fatigue(
        cable,
        series.curvature_x_1_m,
        series.curvature_y_1_m,
        tension_N=series.tension_N,
    )
    b = cx.compute_fatigue(
        cable,
        via_csv.curvature_x_1_m,
        via_csv.curvature_y_1_m,
        tension_N=via_csv.tension_N,
    )
    for layer in a.D_bulk_per_fibre:
        assert np.allclose(
            a.D_bulk_per_fibre[layer], b.D_bulk_per_fibre[layer], rtol=1e-9, atol=0
        )


def test_window_selects_the_retained_record(tmp_path):
    positions = _bending_case(np.full(11, 0.02), np.zeros(11), 11)
    root, deck = _write_case(tmp_path, positions)
    history = read_cabledyn_sections(root, deck, {"s": 5.0}).histories["s"]
    kept = history.window(0.3, 0.8)
    assert np.allclose(kept.time_s, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    with pytest.raises(CableSchemaError, match="ends at"):
        history.window(0.3, 5.0)


# --------------------------------------------------------------------------- #
# Failures: missing channels, mismatched files                                #
# --------------------------------------------------------------------------- #
def test_missing_tangent_channels_are_named(tmp_path):
    positions = _bending_case(np.full(3, 0.02), np.zeros(3), 3)
    root, deck = _write_case(tmp_path, positions)
    with pytest.raises(
        CableSchemaError, match=r"L1N<J>Dec and L1N<J>Azi channels at nodes"
    ):
        read_cabledyn_sections(root, deck, {"s": 5.0}, curvature_source="tangent")


def test_missing_tension_source_is_named(tmp_path):
    positions = _bending_case(np.full(3, 0.02), np.zeros(3), 3)
    root, deck = _write_case(tmp_path, positions, outputs="p")
    with pytest.raises(CableSchemaError, match=r"Ten1N51"):
        read_cabledyn_sections(root, deck, {"s": 5.0})


def test_missing_positions_for_arc_frame(tmp_path):
    positions = _bending_case(np.full(3, 0.02), np.zeros(3), 3)
    channels = _chord_tangent_channels(positions)
    root, deck = _write_case(tmp_path, positions, outputs="t", channels=channels)
    with pytest.raises(CableSchemaError, match="frame transport needs node positions"):
        read_cabledyn_sections(root, deck, {"s": 5.0}, pressure="none")
    # The section frame needs only the tangents near the section.
    read = read_cabledyn_sections(
        root, deck, {"s": 5.0}, pressure="none", frame="section"
    )
    assert read.histories["s"].curvature_source == "tangent"


def test_mesh_that_differs_from_the_deck_is_rejected(tmp_path):
    positions = _bending_case(np.full(3, 0.02), np.zeros(3), 3)
    root, deck = _write_case(tmp_path, positions)
    _deck(deck, n_elements=99)
    with pytest.raises(CableSchemaError, match="header does not match"):
        read_cabledyn_sections(root, deck, {"s": 5.0})


def test_files_from_different_runs_are_rejected(tmp_path):
    positions = _bending_case(np.full(3, 0.02), np.zeros(3), 3)
    root, deck = _write_case(tmp_path, positions)
    shifted = positions.copy()
    shifted[0, 10, 2] += 0.01
    _write_case(tmp_path, shifted, static=False)
    with pytest.raises(CableSchemaError, match="static profile"):
        read_cabledyn_sections(root, deck, {"s": 5.0})


def test_reversed_segment_order_is_rejected(tmp_path):
    nt = 3
    positions = _bending_case(np.full(nt, 0.02), np.zeros(nt), nt)
    mids = 0.5 * (_arcs()[:-1] + _arcs()[1:])
    tension = np.tile(4000.0 + 100.0 * mids, (nt, 1))
    nodes = {
        f"Ten1N{j}": np.full(nt, 4000.0 + 100.0 * _arcs()[j - 1]) for j in (20, 70)
    }
    root, deck = _write_case(
        tmp_path, positions, tension=tension[:, ::-1], channels=nodes
    )
    with pytest.raises(CableSchemaError, match="segment order"):
        read_cabledyn_sections(root, deck, {"s": 5.0})
    root, deck = _write_case(
        tmp_path, positions, tension=tension, channels=nodes, root="ok"
    )
    read = read_cabledyn_sections(root, deck, {"s": 5.0})
    assert read.checks["node_tension_vs_segment_mean_max_rel"] < 1e-12


def test_curvature_channel_disagreement_is_rejected(tmp_path):
    nt = 3
    kappa = np.full(nt, 0.02)
    positions = _bending_case(kappa, np.zeros(nt), nt)
    root, deck = _write_case(tmp_path, positions, channels={"Curv1N30": kappa})
    read = read_cabledyn_sections(root, deck, {"s": 5.0})
    assert read.checks["curvature_vs_channels"]["N30"]["max_abs_error_1_m"] < 1e-6
    root, deck = _write_case(
        tmp_path, positions, channels={"Curv1N30": 2 * kappa}, root="bad"
    )
    with pytest.raises(CableSchemaError, match="Curv1N30"):
        read_cabledyn_sections(root, deck, {"s": 5.0})


def test_output_step_must_match_the_deck(tmp_path):
    positions = _bending_case(np.full(3, 0.02), np.zeros(3), 3)
    root, deck = _write_case(tmp_path, positions, time=np.array([0.0, 0.2, 0.4]))
    with pytest.raises(CableSchemaError, match="dtM"):
        read_cabledyn_sections(root, deck, {"s": 5.0})


def test_openfast_units_row_is_checked(tmp_path):
    nt = 3
    positions = _bending_case(np.full(nt, 0.02), np.zeros(nt), nt)
    channels = _chord_tangent_channels(positions)
    names = [f"L1N{j}{kind}" for j in range(48, 56) for kind in ("Dec", "Azi")]
    names += ["Ten1N51", "Ten1N52"]
    channels["Ten1N51"] = channels["Ten1N52"] = np.full(nt, 1000.0)
    _deck(tmp_path / "of.dat", channels=names)
    for unit, ok in (("(deg)", True), ("(rad)", False)):
        with open(tmp_path / "of.CD.out", "w", encoding="utf-8") as f:
            f.write("\t".join(["Time", *names]) + "\n")
            f.write(
                "\t".join(
                    [
                        "(s)",
                        *[
                            (unit if n.endswith(("Dec", "Azi")) else "(N)")
                            for n in names
                        ],
                    ]
                )
                + "\n"
            )
            for k in range(nt):
                f.write(_fmt([0.1 * k, *[channels[n][k] for n in names]]) + "\n")
        call = lambda: read_cabledyn_sections(  # noqa: E731
            tmp_path / "of",
            tmp_path / "of.dat",
            {"s": 5.05},
            frame="section",
            pressure="none",
        )
        if ok:
            assert call().histories["s"].curvature_source == "tangent"
        else:
            with pytest.raises(CableSchemaError, match="unit"):
                call()


def test_section_outside_the_line_is_rejected(tmp_path):
    positions = _bending_case(np.full(3, 0.02), np.zeros(3), 3)
    root, deck = _write_case(tmp_path, positions)
    with pytest.raises(CableSchemaError, match="outside line 1"):
        read_cabledyn_sections(root, deck, {"s": 10.5})


def test_deck_reader_applies_the_anchor_first_swap(tmp_path):
    text = _deck(tmp_path / "d.dat").read_text(encoding="utf-8")
    text = text.replace("1 1 2 pt\n", "1 2 1 pt\n").replace(
        "1 cable 10.0 100\n", "1 cable 4.0 40\n1 cable 6.0 30\n"
    )
    path = tmp_path / "swap.dat"
    path.write_text(text, encoding="utf-8")
    line = read_cabledyn_deck(path).line(1)
    assert line.reversed_by_parser
    assert [s.length_m for s in line.sections] == [6.0, 4.0]
    assert line.end_a == "1"
    assert np.isclose(line.node_arc_lengths()[30], 6.0)


# --------------------------------------------------------------------------- #
# Genuine CableDyn output                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.full_scale
def test_real_cabledyn_run_from_channels():
    """128 elements, tangent and position channels only (no per-line files)."""
    read = read_cabledyn_sections(
        DATA / "lw128",
        DATA / "lw128.dat",
        {"hangoff": 0.25, "sag": 58.4},
        frame="section",
    )
    assert read.checks["curvature_source"] == "tangent"
    assert read.checks["first_row_vs_static_m"] == 0.0
    assert read.checks["time_step_s"] == pytest.approx(0.1)
    for node, record in read.checks["curvature_vs_channels"].items():
        assert record["max_abs_error_1_m"] <= 0.025 * record["max_channel_1_m"], node
    sag = read.histories["sag"]
    # Sag bend of the lazy wave: centre of curvature above, rotation about -e2.
    assert np.all(sag.rotation_curvature_y_1_m < -0.08)
    assert np.std(sag.rotation_curvature_x_1_m) > 1e-4  # 30 deg sea: out of plane
    # Convex (lower) side in tension: the gradient points to +e1, downward.
    gx, gy = sag.strain_gradient()
    assert np.all(gx > 0.08) and np.all(np.abs(np.arctan2(gy, gx)) < 0.2)
    assert sag.external_pressure_Pa.min() > 6e5
    hang = read.histories["hangoff"]
    assert 8e3 < hang.effective_tension_N.mean() < 1.1e4
    roles = {f.role for f in read.files}
    assert roles == {"main", "static", "elements", "deck"}


@pytest.mark.full_scale
def test_real_cabledyn_run_with_line_files():
    """24 elements with .p.out and .t.out: coarse, so the check fails loudly."""
    sections = {"hangoff": 0.25, "sag": 58.4}
    with pytest.raises(CableSchemaError, match="resolve the curvature"):
        read_cabledyn_sections(DATA / "lw24", DATA / "lw24.dat", sections)
    read = read_cabledyn_sections(
        DATA / "lw24", DATA / "lw24.dat", sections, curvature_check_rtol=0.5
    )
    assert read.checks["curvature_source"] == "positions"
    assert read.checks["tension_source"].startswith("segment")
    # Element-midpoint forces oscillate about the element mean on this mesh.
    assert read.checks["node_tension_vs_segment_mean_max_rel"] > 0.5
    assert {f.role for f in read.files} == {
        "main",
        "positions",
        "tensions",
        "static",
        "elements",
        "deck",
    }
