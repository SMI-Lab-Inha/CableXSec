# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""CableDyn output reader: frames, signs, units, failures.

Synthetic files follow the layouts of CableDyn's standalone driver
(doc/outputs.rst): known centrelines with analytic curvature vectors are
written as node positions, node or chord tangents (``Dec``/``Azi``), element
tensions and curvature magnitudes. ``tests/data/cabledyn`` holds compact runs
on the ``lazy_wave_vessel_rao`` example cable (JONSWAP Hs 4 m, Tp 10 s,
30 deg; RAO-driven hang-off): ``lw128`` of the CableDyn 0.4.2 driver (node
tangents; development build 4b94321), ``lw128_v010`` of the v0.1.0 driver
on the same deck (chord tangents), ``lw24`` of the v0.1.0 driver (no
tangent channels) and ``lw24r`` of the CableDyn 0.4.2 driver (development
build 4b94321) on the ``lw24`` deck (1 s, tangent channels), whose static
solve (``tensile_safety True``) now refines the 24 deck elements to 43, and
``lw24g`` of the same build on that deck written with the 0.4.x LINE TYPES
tokens (damping, ``GJ|K|Bt``, End A restrained in torsion).
"""

from __future__ import annotations

import gc
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from cablexsec.exceptions import CableSchemaError
from cablexsec.io.cabledyn import (
    read_cabledyn_deck,
    read_cabledyn_sections,
    section_channels,
)
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
    sections=None,
):
    """A one-line deck; ``sections`` lists (length, NumSegs) from End A."""
    sections = [(length, n_elements)] if sections is None else sections
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
        + "".join(f"1 cable {sec_len} {sec_n}\n" for sec_len, sec_n in sections)
        + f"{ends}"
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
    # The run ends at TMax, as the standalone driver's does.
    deck_kwargs = {"t_max": round(float(time[-1]), 12), **(deck_kwargs or {})}
    deck = _deck(
        tmp_path / f"{root}.dat",
        n_elements=nn - 1,
        outputs=outputs,
        channels=list(channels),
        **deck_kwargs,
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


def _angle_channels(tangents, line=1):
    """Dec/Azi channels [deg] of unit tangents (nt, nn, 3)."""
    out = {}
    for j in range(tangents.shape[1]):
        t = tangents[:, j]
        out[f"L{line}N{j + 1}Dec"] = np.degrees(np.arccos(np.clip(t[:, 2], -1, 1)))
        out[f"L{line}N{j + 1}Azi"] = np.degrees(np.arctan2(t[:, 1], t[:, 0])) % 360.0
    return out


def _chord_tangent_channels(positions, line=1):
    """Dec/Azi of chord tangents: an EI = 0 line, or CableDyn 0.4.1 and earlier.

    Central difference of the node positions; one-sided at the ends.
    """
    nn = positions.shape[1]
    tangents = np.empty_like(positions)
    for j in range(nn):
        a, b = max(j - 1, 0), min(j + 1, nn - 1)
        t = positions[:, b] - positions[:, a]
        tangents[:, j] = t / np.linalg.norm(t, axis=-1, keepdims=True)
    return _angle_channels(tangents, line)


def _bending_case(kappa, phi, nt, s=None):
    """Arc of curvature kappa(t) whose binormal turns by phi(t) about t0 = +x."""
    s = _arcs() if s is None else s
    positions = np.empty((nt, s.size, 3))
    for k in range(nt):
        b = np.array([0.0, math.cos(phi[k]), math.sin(phi[k])])
        n = np.cross(b, [1.0, 0.0, 0.0])
        positions[k] = _arc(kappa[k], n, b, s)
    return positions


def _node_tangent_channels(kappa, phi, nt, s=None, line=1):
    """Dec/Azi of the exact node tangents of ``_bending_case``.

    What a finite-EI line writes since CableDyn 0.4.2: the tangent at the
    node itself, ``dr/ds = cos(kappa s) t0 + sin(kappa s) n``.
    """
    s = _arcs() if s is None else s
    tangents = np.empty((nt, s.size, 3))
    x0 = np.array([1.0, 0.0, 0.0])
    for k in range(nt):
        b = np.array([0.0, math.cos(phi[k]), math.sin(phi[k])])
        n = np.cross(b, x0)
        theta = kappa[k] * s
        tangents[k] = np.cos(theta)[:, None] * x0 + np.sin(theta)[:, None] * n
    return _angle_channels(tangents, line)


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
        channels = (
            _node_tangent_channels(kappa, np.full(nt, phi), nt)
            if source == "tangent"
            else {}
        )
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


# A non-uniform mesh: 0.2 m elements over the first metre, then 0.3 m.
NONUNIFORM = ((1.0, 5), (9.0, 30))
NONUNIFORM_ARCS = np.concatenate(
    ([0.0], 0.2 * np.arange(1, 6), 1.0 + 0.3 * np.arange(1, 31))
)
PLACEMENT_SECTIONS = {"end_a": 0.0, "junction": 1.0, "mid": 4.37, "end_b": 10.0}


@pytest.mark.analytical_sanity
def test_node_tangents_of_a_finite_ei_line_sit_at_the_nodes(tmp_path):
    """CableDyn 0.4.2+: a finite-EI line writes each node's own tangent.

    On a non-uniform mesh, at End A, at the mesh junction and at End B the
    node tangents give the arc curvature. Placed at the chord mid-arcs (the
    reading of CableDyn 0.4.1 output) the End-A tangent sits half an element
    inside the line and the curvature there comes out about twice too large.
    """
    nt = 4
    kappa = 0.05 + 0.01 * np.arange(nt)
    phi = np.zeros(nt)
    s = NONUNIFORM_ARCS
    positions = _bending_case(kappa, phi, nt, s)
    channels = _node_tangent_channels(kappa, phi, nt, s)
    root, deck = _write_case(
        tmp_path, positions, channels=channels, deck_kwargs={"sections": NONUNIFORM}
    )
    read = read_cabledyn_sections(
        root, deck, PLACEMENT_SECTIONS, curvature_source="tangent", pressure="none"
    )
    assert read.checks["tangent_placement"].startswith("node (finite-EI line")
    assert (
        "tangents placed at the node"
        in read.histories["end_a"].provenance.limitations[0]
    )
    for history in read.histories.values():
        assert np.allclose(history.rotation_curvature_x_1_m, 0.0, atol=1e-6)
        assert np.allclose(history.rotation_curvature_y_1_m, kappa, rtol=1e-4)
    chord = read_cabledyn_sections(
        root,
        deck,
        PLACEMENT_SECTIONS,
        curvature_source="tangent",
        tangent_placement="chord",
        pressure="none",
    )
    assert chord.checks["tangent_placement"].startswith("chord mid-arc")
    end_a = chord.histories["end_a"].curvature_norm_1_m
    assert np.all(end_a > 1.5 * kappa)


@pytest.mark.analytical_sanity
@pytest.mark.parametrize(("ei", "placement"), [(0.0, "auto"), (2.0e4, "chord")])
def test_chord_tangents_sit_at_the_chord_mid_arcs(tmp_path, ei, placement):
    """An EI = 0 line (every version) and finite-EI output of CableDyn <= 0.4.1.

    Both write the chord between the neighbouring nodes (one-sided at the
    ends), the tangent of the arc at the chord's mid-arc.
    """
    nt = 4
    kappa = 0.05 + 0.01 * np.arange(nt)
    s = NONUNIFORM_ARCS
    positions = _bending_case(kappa, np.zeros(nt), nt, s)
    root, deck = _write_case(
        tmp_path,
        positions,
        channels=_chord_tangent_channels(positions),
        deck_kwargs={"sections": NONUNIFORM, "ei": ei},
    )
    read = read_cabledyn_sections(
        root,
        deck,
        PLACEMENT_SECTIONS,
        curvature_source="tangent",
        tangent_placement=placement,
        pressure="none",
    )
    expected = "chord mid-arc (EI = 0" if ei == 0.0 else "chord mid-arc (tangent_"
    assert read.checks["tangent_placement"].startswith(expected)
    for history in read.histories.values():
        assert np.allclose(history.rotation_curvature_x_1_m, 0.0, atol=1e-6)
        assert np.allclose(history.rotation_curvature_y_1_m, kappa, rtol=1e-4)


def test_tangent_placement_is_checked(tmp_path):
    positions = _bending_case(np.full(3, 0.02), np.zeros(3), 3)
    root, deck = _write_case(tmp_path, positions)
    with pytest.raises(CableSchemaError, match="tangent_placement"):
        read_cabledyn_sections(root, deck, {"s": 5.0}, tangent_placement="node")


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
    # The synthetic hang-off (a Coupled point) does not turn.
    history = read_cabledyn_sections(
        root, deck, {"s": 3.95}, pressure="none", hang_off_rotation="none"
    ).histories["s"]
    twist = np.radians(twist_deg)
    assert np.allclose(history.twist_rad, twist)
    assert np.allclose(history.torque_Nm, 12.5)
    assert np.allclose(
        history.rotation_curvature_x_1_m, kappa * np.cos(twist), atol=2e-6
    )
    assert np.allclose(
        history.rotation_curvature_y_1_m, -kappa * np.sin(twist), atol=2e-6
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
    channels = _node_tangent_channels(np.full(3, 0.02), np.zeros(3), 3)
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
    with pytest.raises(CableSchemaError, match="the deck and the output differ"):
        read_cabledyn_sections(root, deck, {"s": 5.0})


def test_files_from_different_runs_are_rejected(tmp_path):
    positions = _bending_case(np.full(3, 0.02), np.zeros(3), 3)
    root, deck = _write_case(tmp_path, positions)
    shifted = positions.copy()
    shifted[0, 10, 2] += 0.01
    _write_case(tmp_path, shifted, static=False)
    with pytest.raises(CableSchemaError, match="static profile"):
        read_cabledyn_sections(root, deck, {"s": 5.0})


def _read_without_open_files(*args, **kwargs):
    """Read, collect, and fail on any file the read left open."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        try:
            return read_cabledyn_sections(*args, **kwargs)
        finally:
            gc.collect()
            leaks = [w for w in caught if issubclass(w.category, ResourceWarning)]
            assert not leaks, [str(w.message) for w in leaks]


def test_reader_closes_every_output_file(tmp_path):
    # Main table and both per-line files, read in several chunks.
    nt = 50
    positions = _bending_case(np.full(nt, 0.02), np.zeros(nt), nt)
    root, deck = _write_case(tmp_path, positions, deck_kwargs={"t_max": 4.9})
    read = _read_without_open_files(root, deck, {"s": 5.0}, chunk_rows=16)
    assert read.histories["s"].time_s.size == nt
    # A per-line file with an extra row: rejected, and every file closed.
    with open(f"{root}.Line1.t.out", "a", encoding="utf-8") as f:
        f.write(_fmt([nt * 0.1, *np.full(N_ELEMENTS, 5000.0)]) + "\n")
    with pytest.raises(CableSchemaError, match="different row counts"):
        _read_without_open_files(root, deck, {"s": 5.0}, chunk_rows=16)


def test_deck_options_follow_the_record_forms(tmp_path):
    # Positional records (model first, keyword after the numbers), commentary
    # after a keyword, the keyword-first dynamic_solver, aliases and the last
    # record of a keyword winning.
    extra = (
        "uniform 0.0 0.6 0.0 current\n"
        "airy 2.0 8.0 0.0 waves - regular waves\n"
        "dynamic_solver 1e-8 1e-14 30 12\n"
        "1000.0 WtrDnsty water density\n"
        "35.0 WtrDpth - depth\n"
        "40.0 water_depth\n"
        "true adaptive_mesh\n"
    )
    deck = read_cabledyn_deck(_deck(tmp_path / "o.dat", extra_options=extra))
    assert deck.options["current"] == "uniform 0.0 0.6 0.0"
    assert deck.options["waves"] == "airy 2.0 8.0 0.0"
    assert deck.options["dynamic_solver"] == "1e-8 1e-14 30 12"
    assert not any(key[0].isdigit() for key in deck.options)
    assert deck.rho_w_kg_m3 == 1000.0  # the alias set after rhoW 1025
    assert deck.water_depth_m == 40.0  # the last of WtrDpth / water_depth
    assert deck.g_m_s2 == 9.81 and deck.dt_s == 0.1 and deck.t_max_s == 1.0
    assert deck.adaptive_mesh
    assert deck.defaults_used == ()


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
    channels = _node_tangent_channels(np.full(nt, 0.02), np.zeros(nt), nt)
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
    """128 elements, tangent and position channels only (no per-line files).

    CableDyn 0.4.2 output: node tangents on a mesh of 0.2 m elements over the
    first 6 m, then 1.24 m. Placed at their nodes, the tangents give
    CableDyn's own curvature at the hang-off nodes N2 to N4 and in the sag
    bend to 2.5e-4 of its largest value. Placed at the chord mid-arcs, the
    End-A tangent sits 0.1 m inside the line and N2 comes out 52 % off,
    N3 15 %.
    """
    read = read_cabledyn_sections(
        DATA / "lw128",
        DATA / "lw128.dat",
        {"hangoff": 0.25, "sag": 58.4},
        frame="section",
    )
    assert read.checks["curvature_source"] == "tangent"
    assert read.checks["tangent_placement"].startswith("node")
    assert read.checks["first_row_vs_static_m"] == 0.0
    assert read.checks["time_step_s"] == pytest.approx(0.1)
    records = read.checks["curvature_vs_channels"]
    assert {"N2", "N3", "N4", "N71", "N75"} <= set(records)
    for node, record in records.items():
        assert record["max_abs_error_1_m"] <= 1e-3 * record["max_channel_1_m"], node
    with pytest.raises(CableSchemaError, match="needs the default tangent_placement"):
        read_cabledyn_sections(
            DATA / "lw128",
            DATA / "lw128.dat",
            {"hangoff": 0.25, "sag": 58.4},
            frame="section",
            tangent_placement="chord",
        )
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
def test_real_cabledyn_v010_run_with_chord_tangents():
    """The same deck run by CableDyn v0.1.0, whose tangents are chords.

    ``tangent_placement="chord"`` reads it: the near-end nodes N2 and N3 agree
    with CableDyn's curvature to 1.5 %; the node placement of 0.4.2
    output is 5 to 11 times further off there.
    """
    sections = {"hangoff": 0.25, "sag": 58.4}
    root, deck = DATA / "lw128_v010", DATA / "lw128_v010.dat"
    read = read_cabledyn_sections(
        root, deck, sections, frame="section", tangent_placement="chord"
    )
    assert read.checks["tangent_placement"].startswith("chord mid-arc")
    records = read.checks["curvature_vs_channels"]
    for node, record in records.items():
        assert record["max_abs_error_1_m"] <= 0.025 * record["max_channel_1_m"], node
    assert np.all(read.histories["sag"].rotation_curvature_y_1_m < -0.08)
    node = read_cabledyn_sections(root, deck, sections, frame="section")
    for name in ("N2", "N3"):
        error = node.checks["curvature_vs_channels"][name]["max_abs_error_1_m"]
        assert error > 4.0 * records[name]["max_abs_error_1_m"], name


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


# --------------------------------------------------------------------------- #
# A mesh the static solve refined                                             #
# --------------------------------------------------------------------------- #
def _refined_copy(tmp_path, files=("out", "Line1.p.out", "Line1.t.out", "dat")):
    """Copy of the ``lw24r`` run with only ``files`` (suffixes after the root)."""
    for suffix in files:
        name = f"lw24r.{suffix}"
        (tmp_path / name).write_bytes((DATA / name).read_bytes())
    return tmp_path / "lw24r", tmp_path / "lw24r.dat"


@pytest.mark.full_scale
def test_real_cabledyn_run_on_a_refined_mesh():
    """``tensile_safety True`` refines the 24 deck elements to 43.

    CableDyn 0.4.2 (development build 4b94321) splits deck elements 6 to 24 in
    two; the per-line files, ``.static.out`` and ``.elements.out`` list the
    44 mesh nodes, the node channels name the 25 deck nodes (``# DeckNode``
    of ``.p.out`` and the ``DeckNode`` column of ``.static.out``). From the
    refined positions the curvature meets every ``Curv`` channel within the
    default 5 % (the 24-element ``lw24`` run of v0.1.0 needs 50 %); from the
    deck-node tangents alone within 10 %.
    """
    root, deck = DATA / "lw24r", DATA / "lw24r.dat"
    sections = {"upper": 10.0, "sag": 58.4}
    read = read_cabledyn_sections(root, deck, sections, curvature_source="positions")
    assert read.checks["mesh"] == {
        "deck_nodes": 25,
        "output_nodes": 44,
        "refined": True,
        "deck_nodes_located_by": "DeckNode of lw24r.Line1.p.out, lw24r.static.out",
    }
    assert read.checks["arc_lengths"].startswith("refined mesh of .elements.out")
    assert read.checks["first_row_vs_static_m"] == 0.0
    assert read.checks["tension_source"].startswith("segment")
    records = read.checks["curvature_vs_channels"]
    assert set(records) == {"N2", "N3", "N4", "N8", "N10", "N12"}
    for node in ("N3", "N4", "N8", "N10", "N12"):
        record = records[node]
        assert record["max_abs_error_1_m"] <= 0.025 * record["max_channel_1_m"], node
    sag = read.histories["sag"]
    assert np.all(sag.rotation_curvature_y_1_m < -0.09)
    assert 1.3e3 < sag.effective_tension_N.mean() < 1.5e3

    tangent = read_cabledyn_sections(
        root,
        deck,
        sections,
        curvature_source="tangent",
        curvature_check_rtol=0.1,
    )
    assert tangent.checks["tangent_placement"].startswith("node")
    for node, record in tangent.checks["curvature_vs_channels"].items():
        assert record["max_abs_error_1_m"] <= 0.1 * record["max_channel_1_m"], node
    np.testing.assert_allclose(
        tangent.histories["sag"].rotation_curvature_y_1_m,
        sag.rotation_curvature_y_1_m,
        rtol=0.01,
    )
    # Chords on the refined mesh (CableDyn 0.4.1 placement) do not fit 0.4.2.
    with pytest.raises(CableSchemaError, match="needs the default tangent_placement"):
        read_cabledyn_sections(
            root,
            deck,
            sections,
            curvature_source="tangent",
            curvature_check_rtol=0.1,
            tangent_placement="chord",
        )


def test_refined_mesh_is_located_without_a_deck_node_numbering(tmp_path):
    """Without ``DeckNode`` the deck nodes are found by their arc lengths."""
    root, deck = _refined_copy(
        tmp_path, ("out", "Line1.t.out", "elements.out", "static.out", "dat")
    )
    lines = (DATA / "lw24r.Line1.p.out").read_text(encoding="utf-8").splitlines()
    assert lines[1].startswith("# DeckNode")
    (tmp_path / "lw24r.Line1.p.out").write_text(
        "\n".join([lines[0], *lines[2:]]) + "\n", encoding="utf-8"
    )
    static = (tmp_path / "lw24r.static.out").read_text(encoding="utf-8").splitlines()
    (tmp_path / "lw24r.static.out").write_text(
        "\n".join([static[0], *("\t".join(row.split("\t")[:-1]) for row in static[1:])])
        + "\n",
        encoding="utf-8",
    )
    read = read_cabledyn_sections(
        root, deck, {"sag": 58.4}, curvature_source="positions"
    )
    assert read.checks["mesh"]["deck_nodes_located_by"] == (
        "arc lengths of lw24r.elements.out"
    )
    reference = read_cabledyn_sections(
        DATA / "lw24r", DATA / "lw24r.dat", {"sag": 58.4}, curvature_source="positions"
    )
    np.testing.assert_array_equal(
        read.histories["sag"].rotation_curvature_y_1_m,
        reference.histories["sag"].rotation_curvature_y_1_m,
    )


def test_refined_mesh_without_the_element_file(tmp_path):
    """The multibody march writes no ``.static.out`` or ``.elements.out``.

    The ``# DeckNode`` line of ``.p.out`` still locates the deck nodes, so the
    positions are read there; the segments of ``.t.out`` and chords on the
    refined mesh cannot be placed.
    """
    root, deck = _refined_copy(tmp_path)
    sections = {"upper": 10.0, "sag": 58.4}
    read = read_cabledyn_sections(root, deck, sections, curvature_check_rtol=0.1)
    assert read.checks["curvature_source"] == "tangent"
    assert read.checks["tension_source"] == "node channels Ten<L>N<J>"
    assert "lw24r.Line1.t.out" in read.checks["tension_file_skipped"]
    assert read.checks["arc_lengths"].startswith("deck sections at the deck nodes")
    with pytest.raises(CableSchemaError, match="elements.out places; it is missing"):
        read_cabledyn_sections(root, deck, sections, tension_source="segments")
    with pytest.raises(CableSchemaError, match="elements.out locates; it is missing"):
        read_cabledyn_sections(root, deck, sections, tangent_placement="chord")


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        # One deck element fewer: the DeckNode numbering names 25 nodes.
        ("1       bare      68.114   10", "1       bare      68.114   9", "DeckNode"),
        # Same mesh, another section length: the deck nodes move.
        ("1       buoy      50.000   8", "1       buoy      51.000   8", "differs"),
    ],
)
def test_refined_mesh_of_another_deck_is_rejected(tmp_path, old, new, match):
    root, deck = _refined_copy(
        tmp_path,
        ("out", "Line1.p.out", "Line1.t.out", "elements.out", "static.out", "dat"),
    )
    text = deck.read_text(encoding="utf-8")
    assert old in text
    deck.write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(CableSchemaError, match=match):
        read_cabledyn_sections(root, deck, {"sag": 58.4})


def test_deck_node_numberings_must_agree(tmp_path):
    root, deck = _refined_copy(
        tmp_path, ("out", "Line1.p.out", "Line1.t.out", "static.out", "dat")
    )
    path = tmp_path / "lw24r.Line1.p.out"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace("\t6\t0\t7\t", "\t6\t7\t0\t", 1)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(CableSchemaError, match="DeckNode numberings of line 1 differ"):
        read_cabledyn_sections(root, deck, {"sag": 58.4})


# --------------------------------------------------------------------------- #
# CableDyn 0.4.x LINE TYPES tokens                                            #
# --------------------------------------------------------------------------- #
def _without_twist(tmp_path, *, coupling=True):
    """Copy of the ``lw24g`` run without its ``Twist`` channels."""
    for suffix in ("Line1.p.out", "Line1.t.out", "static.out", "elements.out"):
        name = f"lw24g.{suffix}"
        (tmp_path / name).write_bytes((DATA / name).read_bytes())
    rows = (DATA / "lw24g.out").read_text(encoding="utf-8").splitlines()
    keep = [
        i for i, name in enumerate(rows[1].split("\t")) if not name.startswith("Twist")
    ]
    (tmp_path / "lw24g.out").write_text(
        "\n".join(
            [
                rows[0],
                *("\t".join(row.split("\t")[i] for i in keep) for row in rows[1:]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    text = (DATA / "lw24g.dat").read_text(encoding="utf-8")
    if not coupling:
        text = text.replace("1.5e4|2.0e3|", "1.5e4|0|")
    (tmp_path / "lw24g.dat").write_text(text, encoding="utf-8")
    return tmp_path / "lw24g", tmp_path / "lw24g.dat"


@pytest.mark.full_scale
def test_real_cabledyn_run_with_damped_coupled_line_types(tmp_path):
    """``lw24g``: the ``lw24`` deck in the grammar of CableDyn 0.4.x.

    Rayleigh bending damping ``EI|-zeta|f1|f2`` and ``EI|Bb``, 14-column rows
    with ``GJ|K|Bt`` (``K`` 2 kN m), End A restrained in torsion (``EndA``,
    ``TorsStiffness Rigid``), End B free, and an ``END`` row in OUTPUTS. The
    CableDyn 0.4.2 driver (development build 4b94321) read and ran it for
    1 s; restrained at one end, the coupling twists the line as it stretches.
    """
    deck = read_cabledyn_deck(DATA / "lw24g.dat")
    bare, buoy = deck.line_type("bare"), deck.line_type("buoy")
    assert (bare.ei_Nm2, buoy.ei_Nm2) == (1.99e4, 1.99e4)
    assert bare.bending_damping_Nm2s == pytest.approx(0.02 * 1.99e4 / (math.pi * 2.2))
    assert bare.mass_damping_1_s == pytest.approx(4 * math.pi * 0.02 * 0.4 / 2.2)
    assert (buoy.bending_damping_Nm2s, buoy.mass_damping_1_s) == (500.0, 0.0)
    assert (bare.gj_Nm2, bare.torque_tension_coupling_Nm, bare.torsional_damping) == (
        1.5e4,
        2.0e3,
        -0.01,
    )
    assert (buoy.torsional_damping, buoy.polar_inertia_kg_m) == (20.0, 0.63)
    line = deck.line(1)
    assert line.torsion_restrained_ends == {"a"} and not line.torsion_restrained
    assert deck.channels[-1] == "L1N14Azi" and "Ignored1" not in deck.channels

    sections = {"upper": 10.0, "sag": 58.4}
    # The clamped hang-off turns with the RAO vessel, whose attitude the
    # output does not record: refused unless given, here declared still.
    with pytest.raises(CableSchemaError, match="Pass hang_off_rotation"):
        read_cabledyn_sections(DATA / "lw24g", deck, sections, curvature_check_rtol=0.5)
    read = read_cabledyn_sections(
        DATA / "lw24g",
        deck,
        sections,
        curvature_check_rtol=0.5,
        hang_off_rotation="none",
    )
    assert read.checks["hang_off"]["source"].startswith("declared still")
    rows = (DATA / "lw24g.out").read_text(encoding="utf-8").splitlines()
    header = rows[1].split("\t")
    channel = np.array(
        [
            [float(row.split("\t")[header.index(f"Twist1N{j}")]) for j in (9, 10)]
            for row in rows[2:]
        ]
    )
    weight = (58.4 - 8 * 68.114 / 10) / (68.114 / 10)
    twist = read.histories["sag"].twist_rad
    np.testing.assert_allclose(
        twist, np.radians((1 - weight) * channel[:, 0] + weight * channel[:, 1])
    )
    assert np.all(twist < 0.0)
    assert read.checks["torsion"] == {
        "restrained_ends": ["a"],
        "solved": True,
        "not_solved_because": "",
        "material_twist": "Twist channels",
    }
    # Without the Twist channels the coupled line cannot be read, and with
    # K = 0 neither can its dynamics (lw24k).
    (tmp_path / "k").mkdir()
    (tmp_path / "plain").mkdir()
    with pytest.raises(CableSchemaError, match="couples torque and tension"):
        read_cabledyn_sections(
            *_without_twist(tmp_path / "k"), sections, curvature_check_rtol=0.5
        )
    with pytest.raises(CableSchemaError, match="in the dynamics the line twists"):
        read_cabledyn_sections(
            *_without_twist(tmp_path / "plain", coupling=False),
            sections,
            curvature_check_rtol=0.5,
        )


@pytest.mark.full_scale
def test_real_run_of_a_line_restrained_at_one_end_twists_dynamically(tmp_path):
    """``lw24k``: ``lw24g`` without the coupling (``GJ|0|Bt``), CableDyn 0.4.2.

    End A, clamped and restrained in torsion on the vessel, rolls with it;
    End B is free in torsion. The statics are twist-free (a free end carries
    no torque), but in the dynamics the polar inertia carries torsional waves
    and the line twists (``Twist1N<J>`` up to -1.7e-3 deg in this 1 s run,
    0.18 deg in a 30 s run of the deck): without the channels it is refused,
    with them read as written. A static deck (``TMax 0``) runs to the single
    ``t = 0`` row, so it cannot have produced this output.
    """
    deck = read_cabledyn_deck(DATA / "lw24k.dat")
    line = deck.line(1)
    assert line.torsion_solved and line.torsion_restrained_ends == {"a"}
    assert deck.line_type("bare").torque_tension_coupling_Nm == 0.0
    rows = (DATA / "lw24k.out").read_text(encoding="utf-8").splitlines()
    header = rows[1].split("\t")
    channel = np.array(
        [
            [float(row.split("\t")[header.index(f"Twist1N{j}")]) for j in (9, 10)]
            for row in rows[2:]
        ]
    )
    assert np.all(channel[0] == 0.0) and np.all(channel[1:] < 0.0)
    sections = {"upper": 10.0, "sag": 58.4}
    # The vessel's turn is not in the output: declared still for this test.
    read = read_cabledyn_sections(
        DATA / "lw24k",
        deck,
        sections,
        curvature_check_rtol=0.5,
        hang_off_rotation="none",
    )
    weight = (58.4 - 8 * 68.114 / 10) / (68.114 / 10)
    np.testing.assert_allclose(
        read.histories["sag"].twist_rad,
        np.radians((1 - weight) * channel[:, 0] + weight * channel[:, 1]),
    )
    assert set(section_channels(deck, sections)) >= {"Twist1N9", "Torq1N10"}
    for suffix in ("Line1.p.out", "Line1.t.out", "static.out", "elements.out"):
        name = f"lw24k.{suffix}"
        (tmp_path / name).write_bytes((DATA / name).read_bytes())
    out = (DATA / "lw24k.out").read_text(encoding="utf-8").splitlines()
    keep = [i for i, name in enumerate(out[1].split("\t")) if "Twist" not in name]
    (tmp_path / "lw24k.out").write_text(
        "\n".join(
            [out[0], *("\t".join(r.split("\t")[i] for i in keep) for r in out[1:])]
        )
        + "\n",
        encoding="utf-8",
    )
    text = (DATA / "lw24k.dat").read_text(encoding="utf-8")
    (tmp_path / "lw24k.dat").write_text(text, encoding="utf-8")
    with pytest.raises(CableSchemaError, match="CableDyn solves its torsion"):
        read_cabledyn_sections(
            tmp_path / "lw24k",
            tmp_path / "lw24k.dat",
            sections,
            curvature_check_rtol=0.5,
        )
    (tmp_path / "lw24k.dat").write_text(
        text.replace("1.0      TMax", "0.0      TMax"), encoding="utf-8"
    )
    with pytest.raises(CableSchemaError, match="TMax is 0, so the standalone driver"):
        read_cabledyn_sections(
            tmp_path / "lw24k",
            tmp_path / "lw24k.dat",
            sections,
            curvature_check_rtol=0.5,
        )
