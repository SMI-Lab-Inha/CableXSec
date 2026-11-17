# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""CableDyn output reader: section histories for local fatigue.

Reads the standalone CableDyn driver files (``<root>.out``,
``<root>.Line<L>.p.out``, ``<root>.Line<L>.t.out``, ``<root>.static.out``,
``<root>.elements.out``) or the OpenFAST module file ``<root>.CD.out`` together
with the deck, and returns the history of selected material sections in the
form the time-history CSV path of CableXSec produces: time, effective tension, two
signed bending-curvature components in a local section frame, and, where the
global model provides them, torsion and external pressure.

What CableDyn writes, and what is derived here
----------------------------------------------
CableDyn reports the curvature magnitude ``Curv<L>N<J>`` at nodes; since
0.4.0 a finite-EI line also writes signed components on section axes
(``CurvX<L>N<J>``, ``CurvY<L>N<J>``), which
:mod:`cablexsec.io.cabledyn_components` reads. This reader reconstructs the
curvature *vector* from the solved geometry and uses ``Curv`` as a check:

* ``curvature_source="tangent"``: node tangents from the ``L<L>N<J>Dec`` and
  ``L<L>N<J>Azi`` channels (written with 8 significant digits of the angle).
  Since CableDyn 0.4.2 a finite-EI line (any section with ``EI > 0``) writes
  the node's own tangent, the cubic-Hermite tangent degree of freedom (at an
  end clamped by a ``Rigid`` row, the direction the clamp holds), and each
  value is placed at its node. An ``EI = 0`` line writes the chord between
  the neighbouring nodes (one-sided at the ends), placed at the mid-arc of
  that chord (an end node at the mid-arc of its element). Up to CableDyn
  0.4.1 a finite-EI line also wrote chords; ``tangent_placement="chord"``
  reads such output (the files carry no version marker).
  A cubic spline through the tangents of a few nodes on either
  side of the section gives :math:`\mathbf{t}` and
  :math:`\boldsymbol{\kappa}=\mathbf{t}\times\mathrm{d}\mathbf{t}/\mathrm{d}s`.
* ``curvature_source="positions"``: node positions from ``.Line<L>.p.out`` (or
  ``L<L>N<J>p{x,y,z}`` channels) and
  :math:`\boldsymbol{\kappa}=\mathbf{r}'\times\mathbf{r}''/|\mathbf{r}'|^3`
  from a cubic spline. The 8-digit position format limits the resolution to
  about :math:`\delta/h^2` (:math:`\delta` the last printed digit, :math:`h`
  the element length); the estimate is recorded in ``checks``.

``s`` is the unstretched arc length from End A (the usual global-model arc length).
Curvature is per unit deformed length; the stretch enters only through the
spline parameter (relative effect of order of the axial strain).

Frames and signs
----------------
The global frame is CableDyn's: right-handed, :math:`z` up, still water at
:math:`z=0`. The section tangent :math:`\mathbf{t}` points from End A to End B
(the usual node :math:`z` axis of a global line model). The two transverse
axes :math:`(\mathbf{e}_1,\mathbf{e}_2)` complete a right-handed triad
:math:`\mathbf{e}_1\times\mathbf{e}_2=\mathbf{t}`. At the first sample (the
static state) they are seeded by
:math:`\mathbf{e}_2=\mathbf{a}\times\mathbf{t}/|\mathbf{a}\times\mathbf{t}|`
with the reference vector :math:`\mathbf{a}` (default global :math:`+z`), so
:math:`\mathbf{e}_2` is horizontal and, for a line in a vertical plane, normal
to that plane; :math:`\mathbf{e}_1=\mathbf{e}_2\times\mathbf{t}` lies in the
vertical plane. They then evolve without twist:

* ``frame="arc"`` (default): parallel transport along the line from End A
  (minimal rotation between successive element chords), the End-A frame itself
  transported in time by minimal rotation. This is the material frame of a
  torque-free isotropic rod. A rotation of the hang-off about its own tangent
  (for example vessel yaw at a vertical hang-off) is not visible in the
  geometry and is not represented.
* ``frame="section"``: the section frame transported in time by minimal
  rotation of the section tangent; needs only the nodes near the section.

When the line is restrained in torsion and ``Twist<L>N<J>`` channels bracket
the section, the material axes are :math:`(\mathbf{e}_1,\mathbf{e}_2)` turned
right-handed about :math:`\mathbf{t}` by the material twist from End A. A line
restrained in torsion without these channels is rejected.

The returned ``rotation_curvature_x/y`` are the components of the
rotation-curvature (Darboux bending) vector :math:`\boldsymbol{\kappa}` on the
two material axes, the representation of the ``x curvature`` and
``y curvature`` of a global line model. They enter local recovery through the same
:func:`~cablexsec.mechanics.section_conventions.rotation_to_strain_curvature`
call as the time-history CSV path: fibre strain :math:`\kappa_x y-\kappa_y x`, so a
sag bend (:math:`\kappa_y<0`) has its tensile fibre on :math:`+\mathbf{e}_1`,
the downward side of the static section.

Tension, torsion, pressure
--------------------------
CableDyn's tension is the effective tension of a solid line without contents
(``doc/conventions.rst``): element-mean axial forces in ``.Line<L>.t.out``,
interpolated linearly between element mid-arcs (held at the end-element value
in the first and last half elements), or the node channels ``Ten<L>N<J>``.
CableDyn writes no pressure. ``pressure="hydrostatic"`` forms
:math:`p_o=\rho_w g\max(-z,0)` from the section elevation and the deck's
``rhoW`` and ``g``; wave dynamic pressure is not included. Torque and twist
are taken from ``Torq<L>N<J>`` and ``Twist<L>N<J>`` when written.

Every required file, column and channel is checked by name and unit; a
missing one raises :class:`~cablexsec.exceptions.CableSchemaError` naming it.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from cablexsec.exceptions import CableSchemaError
from cablexsec.io.series import LoadCaseSeries
from cablexsec.mechanics.section_conventions import (
    rotation_to_strain_curvature,
    true_tension_from_effective,
)
from cablexsec.provenance import Provenance

FloatArray = NDArray[np.float64]
Frame = Literal["arc", "section"]
CurvatureSource = Literal["auto", "tangent", "positions"]
PressureModel = Literal["hydrostatic", "none"]
TangentPlacement = Literal["auto", "chord"]

__all__ = [
    "CableDynDeck",
    "CableDynFile",
    "CableDynLine",
    "CableDynLineType",
    "CableDynRead",
    "CableDynSection",
    "CableDynSectionHistory",
    "read_cabledyn_deck",
    "read_cabledyn_sections",
    "section_channels",
]

# CableDyn documented defaults (doc/options.rst) used when the deck omits them.
_DEFAULT_G = 9.80665
_DEFAULT_RHO_W = 1025.0
# Expected units of the OpenFAST ``.CD.out`` units row (doc/outputs.rst).
_UNITS = {
    "curv": "(1/m)",
    "ten": "(N)",
    "torq": "(N.m)",
    "twist": "(deg)",
    "dec": "(deg)",
    "azi": "(deg)",
    "px": "(m)",
    "py": "(m)",
    "pz": "(m)",
}
_NODE_CHANNEL = re.compile(r"^(curv|ten|torq|twist|bendmom)(\d+)n(\d+)$", re.IGNORECASE)
_NODE_GEOMETRY = re.compile(r"^l(\d+)n(\d+)(dec|azi|px|py|pz)$", re.IGNORECASE)
_SECTION_NAMES = {
    "line types": "line types",
    "linetypes": "line types",
    "line dictionary": "line types",
    "points": "points",
    "connection properties": "points",
    "point properties": "points",
    "connects": "points",
    "lines": "lines",
    "line properties": "lines",
    "sections": "sections",
    "end connections": "end connections",
    "end connection": "end connections",
    "options": "options",
    "solver options": "options",
    "outputs": "outputs",
    "output": "outputs",
}
_MOVING_POINTS = ("coupled", "vessel", "free", "connect")


# --------------------------------------------------------------------------- #
# Deck                                                                        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CableDynLineType:
    """One ``LINE TYPES`` row (SI)."""

    name: str
    diameter_m: float
    mass_kg_m: float
    ea_N: float | None
    ei_Nm2: float
    gj_Nm2: float | None


@dataclass(frozen=True)
class CableDynSection:
    """One ``SECTIONS`` row, End A first."""

    line_type: str
    length_m: float
    n_segments: int


@dataclass(frozen=True)
class CableDynLine:
    """A deck line in public order (End A -> End B, after any anchor-first swap)."""

    line_id: int
    end_a: str
    end_b: str
    outputs: str
    sections: tuple[CableDynSection, ...]
    torsion_restrained: bool
    reversed_by_parser: bool

    @property
    def n_elements(self) -> int:
        """Deck element count (sum of ``NumSegs``)."""
        return sum(sec.n_segments for sec in self.sections)

    @property
    def n_nodes(self) -> int:
        """Deck node count."""
        return self.n_elements + 1

    @property
    def length_m(self) -> float:
        """Unstretched length [m]."""
        return math.fsum(sec.length_m for sec in self.sections)

    def node_arc_lengths(self) -> FloatArray:
        """Unstretched arc length of every node from End A [m]."""
        arcs = [0.0]
        for sec in self.sections:
            start = arcs[-1]
            steps = np.arange(1, sec.n_segments + 1, dtype=np.float64)
            arcs.extend((start + sec.length_m * steps / sec.n_segments).tolist())
        return np.asarray(arcs, dtype=np.float64)

    def line_type_at(self, arc_length_m: float) -> str:
        """Line type name of the section containing ``arc_length_m``."""
        start = 0.0
        for sec in self.sections:
            if arc_length_m <= start + sec.length_m + 1e-12:
                return sec.line_type
            start += sec.length_m
        return self.sections[-1].line_type


@dataclass(frozen=True)
class CableDynDeck:
    """The parts of a CableDyn deck the section reader needs.

    ``options`` holds the raw option tokens with lower-case keys.
    ``defaults_used`` names the physical options taken from the CableDyn
    documented defaults because the deck omits them.
    """

    path: Path
    line_types: Mapping[str, CableDynLineType]
    lines: Mapping[int, CableDynLine]
    point_types: Mapping[str, str]
    options: Mapping[str, str]
    channels: tuple[str, ...]
    g_m_s2: float
    rho_w_kg_m3: float
    water_depth_m: float | None
    dt_s: float | None
    t_max_s: float | None
    adaptive_mesh: bool
    defaults_used: tuple[str, ...] = ()

    def line(self, line_id: int) -> CableDynLine:
        """Return a deck line by id, naming the available ids otherwise."""
        try:
            return self.lines[line_id]
        except KeyError:
            raise CableSchemaError(
                f"{self.path.name}: no LINES row with id {line_id}; "
                f"lines {sorted(self.lines)}"
            ) from None

    def line_type(self, name: str) -> CableDynLineType:
        """Return a line type by (case-insensitive) name."""
        try:
            return self.line_types[name.casefold()]
        except KeyError:
            raise CableSchemaError(
                f"{self.path.name}: line type {name!r} is not defined"
            ) from None


def _strip_comment(text: str) -> str:
    for mark in ("#", "!"):
        cut = text.find(mark)
        if cut >= 0:
            text = text[:cut]
    return text.strip()


#: Keywords of the options the reader uses and their CableDyn aliases.
_OPTION_ALIASES: dict[str, tuple[str, ...]] = {
    "g": ("g", "gravity"),
    "rhoW": ("rhoW", "WtrDnsty", "water_density"),
    "WtrDpth": ("WtrDpth", "water_depth"),
    "dtM": ("dtM", "dt"),
    "TMax": ("TMax",),
    "adaptive_mesh": ("adaptive_mesh", "adaptivemesh"),
}


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _option_record(row: Sequence[str]) -> tuple[str, str] | None:
    """Keyword (case-folded) and value text of one OPTIONS row.

    CableDyn's record forms: ``value keyword [commentary]``; positional
    records with the model name first and the keyword after the numbers
    (``uniform 0.5 0 0 current``, ``airy 2.0 8.0 0.0 waves``); and
    ``dynamic_solver`` with its keyword first. A ``-`` token starts a
    description. The keyword is the first token after the first one that
    is not a number; a row without one is not a record.
    """
    tokens = list(row)
    if "-" in tokens:
        tokens = tokens[: tokens.index("-")]
    if not tokens:
        return None
    if tokens[0].casefold() == "dynamic_solver":
        return tokens[0].casefold(), " ".join(tokens[1:])
    for index in range(1, len(tokens)):
        if not _is_number(tokens[index]):
            return tokens[index].casefold(), " ".join(tokens[:index])
    return None


def _float(token: str, what: str, path: Path) -> float:
    try:
        value = float(token)
    except ValueError:
        raise CableSchemaError(
            f"{path.name}: {what} is not numeric: {token!r}"
        ) from None
    if not math.isfinite(value):
        raise CableSchemaError(f"{path.name}: {what} is not finite: {token!r}")
    return value


def _deck_blocks(path: Path) -> dict[str, list[list[str]]]:
    """Split a deck into recognised sections of token rows (comments removed)."""
    blocks: dict[str, list[list[str]]] = {}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8", errors="strict").splitlines():
        line = raw.strip()
        if line.startswith("---"):
            name = " ".join(line.strip("-").split()).casefold()
            if not name or name == "end" or "need" in name or "input file" in name:
                current = None
                continue
            current = _SECTION_NAMES.get(name, f"other:{name}")
            if current in blocks and not current.startswith("other:"):
                raise CableSchemaError(f"{path.name}: duplicate deck section {name!r}")
            blocks.setdefault(current, [])
            continue
        if current is None or not line or line.startswith("--"):
            continue
        text = _strip_comment(line)
        if text:
            blocks[current].append(text.replace('"', " ").replace("'", " ").split())
    return blocks


def _table(
    rows: list[list[str]], path: Path, section: str
) -> tuple[list[str], list[list[str]]]:
    """Header names (case-folded) and data rows; the units row is dropped."""
    if not rows:
        raise CableSchemaError(f"{path.name}: deck section {section} is empty")
    header = [name.casefold() for name in rows[0]]
    data = [row for row in rows[1:] if not row[0].startswith("(")]
    return header, data


def read_cabledyn_deck(path: str | Path) -> CableDynDeck:
    """Read line types, lines, sections, end connections, options and outputs.

    Applies CableDyn's anchor-first swap (a line whose ``NodeA`` is ``Fixed`` and
    whose ``NodeB`` moves is reversed, with its sections), so node 1 is End A
    as in every output file.
    """
    deck_path = Path(path)
    if not deck_path.is_file():
        raise FileNotFoundError(f"CableDyn deck not found: {deck_path}")
    blocks = _deck_blocks(deck_path)
    for required in ("line types", "lines"):
        if required not in blocks:
            raise CableSchemaError(f"{deck_path.name}: no {required.upper()} section")

    header, rows = _table(blocks["line types"], deck_path, "LINE TYPES")
    for column in ("typename", "diam", "massdeninair", "ea", "ei"):
        if column not in header:
            raise CableSchemaError(
                f"{deck_path.name}: LINE TYPES lacks column {column!r}; found {header}"
            )
    line_types: dict[str, CableDynLineType] = {}
    for row in rows:
        values = dict(zip(header, row, strict=False))
        try:
            ea: float | None = float(values["ea"])
        except ValueError:
            ea = None  # viscoelastic or Syrope constitutive token
        gj_token = values.get("gj")
        name = values["typename"]
        line_types[name.casefold()] = CableDynLineType(
            name=name,
            diameter_m=_float(values["diam"], f"Diam of {name}", deck_path),
            mass_kg_m=_float(values["massdeninair"], f"mass of {name}", deck_path),
            ea_N=ea,
            ei_Nm2=_float(values["ei"], f"EI of {name}", deck_path),
            gj_Nm2=None if gj_token is None else _float(gj_token, "GJ", deck_path),
        )

    point_types: dict[str, str] = {}
    if "points" in blocks:
        p_header, p_rows = _table(blocks["points"], deck_path, "POINTS")
        if "id" not in p_header or not ({"type", "attachment"} & set(p_header)):
            raise CableSchemaError(f"{deck_path.name}: POINTS needs ID and Type")
        type_column = "type" if "type" in p_header else "attachment"
        for row in p_rows:
            values = dict(zip(p_header, row, strict=False))
            point_types[values["id"]] = values[type_column].casefold()

    l_header, l_rows = _table(blocks["lines"], deck_path, "LINES")
    seven_column = "linetype" in l_header
    sections_by_line: dict[int, list[CableDynSection]] = {}
    if "sections" in blocks:
        s_header, s_rows = _table(blocks["sections"], deck_path, "SECTIONS")
        for column in ("lineid", "linetype", "length", "numsegs"):
            if column not in s_header:
                raise CableSchemaError(f"{deck_path.name}: SECTIONS lacks {column!r}")
        for row in s_rows:
            values = dict(zip(s_header, row, strict=False))
            sections_by_line.setdefault(int(values["lineid"]), []).append(
                CableDynSection(
                    line_type=values["linetype"],
                    length_m=_float(values["length"], "section Length", deck_path),
                    n_segments=int(values["numsegs"]),
                )
            )
    torsion_ends: dict[int, set[str]] = {}
    if "end connections" in blocks:
        e_header, e_rows = _table(
            blocks["end connections"], deck_path, "END CONNECTIONS"
        )
        if "torsstiffness" in e_header:
            for row in e_rows:
                values = dict(zip(e_header, row, strict=False))
                token = values.get("torsstiffness", "none").casefold()
                if token not in ("none", "-", "0", "0.0"):
                    torsion_ends.setdefault(int(values["lineid"]), set()).add(
                        values["end"].casefold()
                    )
    lines: dict[int, CableDynLine] = {}
    for row in l_rows:
        values = dict(zip(l_header, row, strict=False))
        line_id = int(values["id"])
        if seven_column:
            end_a, end_b = values.get("attacha", ""), values.get("attachb", "")
            sections = [
                CableDynSection(
                    line_type=values["linetype"],
                    length_m=_float(values["unstrlen"], "UnstrLen", deck_path),
                    n_segments=int(values["numsegs"]),
                )
            ]
            if line_id in sections_by_line:
                raise CableSchemaError(
                    f"{deck_path.name}: line {line_id} has a 7-column row and SECTIONS"
                )
        else:
            end_a, end_b = values.get("nodea", ""), values.get("nodeb", "")
            sections = sections_by_line.get(line_id, [])
        if not sections:
            raise CableSchemaError(f"{deck_path.name}: line {line_id} has no sections")
        for sec in sections:
            if sec.line_type.casefold() not in line_types:
                raise CableSchemaError(
                    f"{deck_path.name}: line {line_id} uses undefined type "
                    f"{sec.line_type!r}"
                )
            if sec.n_segments < 1 or sec.length_m <= 0:
                raise CableSchemaError(
                    f"{deck_path.name}: invalid section of {line_id}"
                )
        type_a = point_types.get(end_a, "")
        type_b = point_types.get(end_b, "")
        swap = type_a == "fixed" and (
            type_b in _MOVING_POINTS
            or type_b.startswith("body")
            or type_b.startswith("turbine")
        )
        if swap:
            end_a, end_b = end_b, end_a
            sections = sections[::-1]
        lines[line_id] = CableDynLine(
            line_id=line_id,
            end_a=end_a,
            end_b=end_b,
            outputs=values.get("outputs", "-"),
            sections=tuple(sections),
            torsion_restrained=torsion_ends.get(line_id, set()) >= {"a", "b"},
            reversed_by_parser=swap,
        )

    records = [
        record
        for record in (_option_record(row) for row in blocks.get("options", []))
        if record is not None
    ]
    # Records apply in order: the last one of a keyword wins.
    options: dict[str, str] = {}
    for key, value in records:
        options[key] = value
    channels = tuple(
        token
        for row in blocks.get("outputs", [])
        for token in " ".join(row).replace(",", " ").split()
    )
    defaults = []

    def last(name: str) -> str | None:
        # The last record of the keyword or any of its aliases.
        names = {alias.casefold() for alias in _OPTION_ALIASES.get(name, (name,))}
        return next((value for key, value in reversed(records) if key in names), None)

    def option(name: str, default: float | None) -> float | None:
        value = last(name)
        if value is not None:
            return _float(value, f"option {name}", deck_path)
        if default is not None:
            defaults.append(name)
        return default

    g = option("g", _DEFAULT_G)
    rho = option("rhoW", _DEFAULT_RHO_W)
    adaptive = last("adaptive_mesh") or "false"
    if adaptive.casefold() not in {
        "true",
        "t",
        "yes",
        "y",
        "on",
        "1",
        "false",
        "f",
        "no",
        "n",
        "off",
        "0",
    }:
        raise CableSchemaError(f"{deck_path.name}: adaptive_mesh {adaptive!r}")
    assert g is not None and rho is not None
    return CableDynDeck(
        path=deck_path,
        line_types=MappingProxyType(line_types),
        lines=MappingProxyType(lines),
        point_types=MappingProxyType(point_types),
        options=MappingProxyType(options),
        channels=channels,
        g_m_s2=g,
        rho_w_kg_m3=rho,
        water_depth_m=option("WtrDpth", None),
        dt_s=option("dtM", None),
        t_max_s=option("TMax", None),
        adaptive_mesh=adaptive.casefold() in {"true", "t", "yes", "y", "on", "1"},
        defaults_used=tuple(defaults),
    )


# --------------------------------------------------------------------------- #
# Output files                                                                #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CableDynFile:
    """A file read for a result, with its digest."""

    role: str
    path: Path
    sha256: str
    size_bytes: int


def _digest(path: Path) -> CableDynFile:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return CableDynFile("", path, sha.hexdigest(), path.stat().st_size)


def _head(path: Path, count: int) -> list[str]:
    lines = []
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for _ in range(count):
            text = handle.readline()
            if not text:
                break
            lines.append(text.rstrip("\r\n"))
    return lines


@dataclass(frozen=True)
class _Table:
    """Layout of a tab-separated CableDyn time-history table."""

    path: Path
    skip: int
    header: tuple[str, ...]
    units: tuple[str, ...] | None


def _main_table(path: Path) -> _Table:
    head = _head(path, 3)
    if len(head) < 2:
        raise CableSchemaError(f"{path.name}: incomplete CableDyn output table")
    if head[0].startswith("#"):
        header = tuple(cell.strip() for cell in head[1].split("\t"))
        if header[0] != "Time(s)":
            raise CableSchemaError(f"{path.name}: first column must be Time(s)")
        return _Table(path, 2, header, None)
    header = tuple(cell.strip() for cell in head[0].split("\t"))
    units = tuple(cell.strip() for cell in head[1].split("\t"))
    if header[0] != "Time" or units[0] != "(s)" or len(units) != len(header):
        raise CableSchemaError(
            f"{path.name}: expected an OpenFAST CableDyn table (Time, units row (s))"
        )
    return _Table(path, 2, header, units)


def _line_table(path: Path, title_word: str, columns: Sequence[str]) -> _Table:
    head = _head(path, 2)
    if len(head) < 2 or not head[0].startswith("# CableDyn"):
        raise CableSchemaError(f"{path.name}: not a CableDyn per-line table")
    title = head[0].casefold()
    if title_word not in title or "enda -> endb" not in title:
        raise CableSchemaError(
            f"{path.name}: title must name {title_word!r} in public order "
            f"EndA -> EndB: {head[0]!r}"
        )
    if "static" in title:
        raise CableSchemaError(f"{path.name}: a static table has no time history")
    header = tuple(cell.strip() for cell in head[1].split("\t"))
    expected = ("Time(s)", *columns)
    if header != expected:
        mismatch = next(
            (
                f"column {i + 1}: {a!r} != expected {b!r}"
                for i, (a, b) in enumerate(zip(header, expected, strict=False))
                if a != b
            ),
            f"{len(header)} columns, expected {len(expected)}",
        )
        raise CableSchemaError(
            f"{path.name}: header does not match the line: {mismatch}"
        )
    return _Table(path, 2, header, None)


def _chunks(
    table: _Table, columns: Sequence[int], chunk_rows: int
) -> Generator[FloatArray, None, None]:
    """Rows of a table in chunks; the file is closed when the generator ends
    or is closed."""
    import pandas as pd  # noqa: PLC0415

    order = {column: i for i, column in enumerate(sorted(set(columns)))}
    pick = [order[column] for column in columns]
    with pd.read_csv(
        table.path,
        sep="\t",
        header=None,
        skiprows=table.skip,
        usecols=sorted(set(columns)),
        dtype=np.float64,
        engine="c",
        chunksize=chunk_rows,
        skipinitialspace=True,
    ) as reader:
        for frame in reader:
            values = frame.to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise CableSchemaError(
                    f"{table.path.name}: incomplete or non-finite row in the history"
                )
            yield values[:, pick]


def _channel_index(table: _Table) -> dict[tuple[str, int, int], int]:
    index: dict[tuple[str, int, int], int] = {}
    for column, token in enumerate(table.header[1:], start=1):
        node = _NODE_CHANNEL.match(token)
        geometry = _NODE_GEOMETRY.match(token)
        if node:
            key = (node.group(1).casefold(), int(node.group(2)), int(node.group(3)))
        elif geometry:
            key = (
                geometry.group(3).casefold(),
                int(geometry.group(1)),
                int(geometry.group(2)),
            )
        else:
            continue
        if key in index:
            raise CableSchemaError(f"{table.path.name}: channel {token} is duplicated")
        if (
            table.units is not None
            and table.units[column].casefold()
            != _UNITS.get(key[0], table.units[column]).casefold()
        ):
            raise CableSchemaError(
                f"{table.path.name}: channel {token} has unit {table.units[column]}, "
                f"expected {_UNITS[key[0]]}"
            )
        index[key] = column
    return index


# --------------------------------------------------------------------------- #
# Geometry kernels                                                            #
# --------------------------------------------------------------------------- #
def _unit(vectors: FloatArray, what: str) -> FloatArray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise CableSchemaError(f"degenerate {what}: zero-length vector")
    return np.asarray(vectors / norm)


def _transport(vector: FloatArray, a: FloatArray, b: FloatArray) -> FloatArray:
    """Minimal rotation of ``vector`` (normal to unit ``a``) taking ``a`` to ``b``."""
    dot = np.sum(a * b, axis=-1, keepdims=True)
    if np.any(dot <= -1.0 + 1e-8):
        raise CableSchemaError("tangent reverses between successive samples")
    moved = vector - np.sum(b * vector, axis=-1, keepdims=True) / (1.0 + dot) * (a + b)
    moved = moved - np.sum(moved * b, axis=-1, keepdims=True) * b
    return _unit(moved, "transported frame axis")


def _seed(tangent: FloatArray, reference: FloatArray, where: str) -> FloatArray:
    """``e1 = e2 x t`` with ``e2 = a x t / |a x t|`` at the first sample."""
    e2 = np.cross(reference, tangent)
    if float(np.linalg.norm(e2)) < 1e-3:
        raise CableSchemaError(
            f"{where}: the tangent is within 0.06 deg of the frame reference vector; "
            "choose another reference_vector"
        )
    e2 = e2 / np.linalg.norm(e2)
    return np.asarray(np.cross(e2, tangent))


def _time_transport(
    tangents: FloatArray, start: FloatArray | None, reference: FloatArray, where: str
) -> tuple[FloatArray, FloatArray]:
    """Frame axis e1 at every sample by minimal rotation in time.

    ``start`` is ``(e1, t)`` of the sample before the chunk, or ``None`` at t=0.
    """
    out = np.empty_like(tangents)
    if start is None:
        e1 = _seed(tangents[0], reference, where)
        previous = tangents[0]
    else:
        e1, previous = start[0], start[1]
    # Scalar loop: each step depends on the previous one.
    vx, vy, vz = (float(c) for c in e1)
    ax, ay, az = (float(c) for c in previous)
    rows = tangents.tolist()
    for k, (bx, by, bz) in enumerate(rows):
        dot = ax * bx + ay * by + az * bz
        if dot <= -1.0 + 1e-8:
            raise CableSchemaError(f"{where}: tangent reverses between samples")
        f = (bx * vx + by * vy + bz * vz) / (1.0 + dot)
        vx, vy, vz = vx - f * (ax + bx), vy - f * (ay + by), vz - f * (az + bz)
        g = vx * bx + vy * by + vz * bz
        vx, vy, vz = vx - g * bx, vy - g * by, vz - g * bz
        n = math.sqrt(vx * vx + vy * vy + vz * vz)
        vx, vy, vz = vx / n, vy / n, vz / n
        out[k] = (vx, vy, vz)
        ax, ay, az = bx, by, bz
    return out, np.array(((vx, vy, vz), (ax, ay, az)))


def _spline(
    locations: FloatArray, values: FloatArray, at: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Value, first and second derivative of a not-a-knot spline along axis 1."""
    from scipy.interpolate import CubicSpline  # noqa: PLC0415

    curve = CubicSpline(locations, values, axis=1, bc_type="not-a-knot")
    return (
        np.asarray(curve(at), dtype=np.float64),
        np.asarray(curve(at, 1), dtype=np.float64),
        np.asarray(curve(at, 2), dtype=np.float64),
    )


# --------------------------------------------------------------------------- #
# Section history                                                             #
# --------------------------------------------------------------------------- #
def _frozen(values: FloatArray | None) -> FloatArray | None:
    if values is None:
        return None
    out = np.array(values, dtype=np.float64)
    out.flags.writeable = False
    return out


@dataclass(frozen=True)
class CableDynSectionHistory:
    """History of one material section of a CableDyn line.

    Attributes
    ----------
    rotation_curvature_x_1_m, rotation_curvature_y_1_m : numpy.ndarray
        Components [1/m] of the rotation-curvature vector on the material
        axes (e1, e2); see the module notes for the frame. Transform with
        :meth:`strain_gradient` before local recovery.
    curvature_norm_1_m : numpy.ndarray
        Magnitude of the reconstructed curvature vector [1/m].
    effective_tension_N : numpy.ndarray
        CableDyn effective tension at the section [N].
    elevation_m : numpy.ndarray or None
        Global z of the section centreline [m] (when positions are read).
    external_pressure_Pa : numpy.ndarray or None
        Hydrostatic external pressure [Pa] (``pressure="hydrostatic"``).
    torque_Nm, twist_rad : numpy.ndarray or None
        Torque and material twist from End A, when written.
    """

    name: str
    arc_length_m: float
    line_id: int
    time_s: FloatArray
    effective_tension_N: FloatArray
    rotation_curvature_x_1_m: FloatArray
    rotation_curvature_y_1_m: FloatArray
    curvature_norm_1_m: FloatArray
    elevation_m: FloatArray | None
    external_pressure_Pa: FloatArray | None
    torque_Nm: FloatArray | None
    twist_rad: FloatArray | None
    frame: str
    curvature_source: str
    tension_source: str
    pressure_model: str
    source: str
    provenance: Provenance

    def __post_init__(self) -> None:
        n = np.asarray(self.time_s).size
        for item in (
            "time_s",
            "effective_tension_N",
            "rotation_curvature_x_1_m",
            "rotation_curvature_y_1_m",
            "curvature_norm_1_m",
            "elevation_m",
            "external_pressure_Pa",
            "torque_Nm",
            "twist_rad",
        ):
            values = _frozen(getattr(self, item))
            if values is not None and (
                values.shape != (n,) or not np.isfinite(values).all()
            ):
                raise CableSchemaError(
                    f"{self.name}: {item} must be finite, length {n}"
                )
            object.__setattr__(self, item, values)
        if n < 2 or np.any(np.diff(self.time_s) <= 0):
            raise CableSchemaError(f"{self.name}: time must increase strictly")

    @property
    def duration_s(self) -> float:
        """Record duration [s]."""
        return float(self.time_s[-1] - self.time_s[0])

    def strain_gradient(self) -> tuple[FloatArray, FloatArray]:
        """Axial-strain gradient ``(-k_y, k_x)`` [1/m], as for CSV history input."""
        return rotation_to_strain_curvature(
            self.rotation_curvature_x_1_m, self.rotation_curvature_y_1_m
        )

    def true_tension_N(self, external_area_m2: float) -> FloatArray:
        """Material force ``T_e - p_o A_o`` [N] (CableDyn lines carry no contents)."""
        if self.external_pressure_Pa is None:
            if self.pressure_model != "none":
                raise CableSchemaError(f"{self.name}: no external pressure history")
            return np.array(self.effective_tension_N)
        return true_tension_from_effective(
            self.effective_tension_N, self.external_pressure_Pa, external_area_m2
        )

    def window(
        self, start_time_s: float = 0.0, end_time_s: float | None = None
    ) -> CableDynSectionHistory:
        """Contiguous samples with ``start <= t <= end`` (transient removal)."""
        end = math.inf if end_time_s is None else end_time_s
        keep = (self.time_s >= start_time_s - 1e-9) & (self.time_s <= end + 1e-9)
        if np.count_nonzero(keep) < 2:
            raise CableSchemaError(
                f"{self.name}: fewer than two samples in [{start_time_s}, {end}] s"
            )
        if end_time_s is not None and self.time_s[-1] < end_time_s - 1e-9:
            raise CableSchemaError(
                f"{self.name}: the record ends at {self.time_s[-1]} s, before "
                f"the requested {end_time_s} s"
            )
        changes: dict[str, Any] = {}
        for item in (
            "time_s",
            "effective_tension_N",
            "rotation_curvature_x_1_m",
            "rotation_curvature_y_1_m",
            "curvature_norm_1_m",
            "elevation_m",
            "external_pressure_Pa",
            "torque_Nm",
            "twist_rad",
        ):
            values = getattr(self, item)
            changes[item] = None if values is None else values[keep]
        return replace(self, **changes)

    def to_load_case_series(self, external_area_m2: float) -> LoadCaseSeries:
        """True-force :class:`LoadCaseSeries` (strain-gradient curvatures)."""
        kx, ky = self.strain_gradient()
        return LoadCaseSeries(
            time_s=self.time_s,
            curvature_x_1_m=kx,
            curvature_y_1_m=ky,
            tension_N=self.true_tension_N(external_area_m2),
            source=self.source,
            provenance=self.provenance,
            tension_kind="true",
        )

    def to_columns(self, external_area_m2: float) -> dict[str, FloatArray]:
        """Columns named as in the ``local_response`` histories of local recovery."""
        kx, ky = self.strain_gradient()
        columns = {
            "time_s": self.time_s,
            "effective_tension_N": self.effective_tension_N,
            "true_tension_N": self.true_tension_N(external_area_m2),
            "strain_gradient_x_1_m": kx,
            "strain_gradient_y_1_m": ky,
            "rotation_curvature_x_1_m": self.rotation_curvature_x_1_m,
            "rotation_curvature_y_1_m": self.rotation_curvature_y_1_m,
            "curvature_norm_1_m": self.curvature_norm_1_m,
        }
        for item in ("external_pressure_Pa", "elevation_m", "torque_Nm", "twist_rad"):
            values = getattr(self, item)
            if values is not None:
                columns[item] = values
        return {key: np.asarray(value) for key, value in columns.items()}


@dataclass(frozen=True)
class CableDynRead:
    """Section histories of one run, the files read and the consistency checks."""

    histories: Mapping[str, CableDynSectionHistory]
    files: tuple[CableDynFile, ...]
    deck: CableDynDeck
    line_id: int
    checks: Mapping[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Reader                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class _Plan:
    """Static data of one section (or a curvature probe at a node)."""

    name: str
    arc: float
    element: int  # 0-based element containing the arc
    window: list[int]  # 0-based nodes of the curvature spline
    probe: bool = False
    node: int = -1  # 0-based node of a curvature probe
    data: dict[str, list[FloatArray]] = field(default_factory=dict)


def _element_of(arcs: FloatArray, arc: float) -> int:
    index = int(np.searchsorted(arcs, arc, side="right")) - 1
    return min(max(index, 0), int(arcs.size) - 2)


def _window(element: int, n_nodes: int, half: int) -> list[int]:
    lo = max(0, element - half + 1)
    hi = min(n_nodes - 1, element + half)
    # Widen at the line ends; the loop stops by the time the window spans every node.
    while hi - lo + 1 < min(2 * half, n_nodes):
        if lo > 0:
            lo -= 1
        else:
            hi += 1
    return list(range(lo, hi + 1))


def _discover(out_root: Path, line_id: int) -> dict[str, Path]:
    standalone = Path(f"{out_root}.out")
    coupled = Path(f"{out_root}.CD.out")
    if standalone.is_file() and coupled.is_file():
        raise CableSchemaError(
            f"both {standalone.name} and {coupled.name} exist; name one output root"
        )
    files: dict[str, Path] = {}
    if standalone.is_file():
        files["main"] = standalone
        for role, suffix in (
            ("positions", f".Line{line_id}.p.out"),
            ("tensions", f".Line{line_id}.t.out"),
            ("static", ".static.out"),
            ("elements", ".elements.out"),
        ):
            candidate = Path(f"{out_root}{suffix}")
            if candidate.is_file():
                files[role] = candidate
    elif coupled.is_file():
        files["main"] = coupled
        static = Path(f"{out_root}.CD.static.out")
        if static.is_file():
            files["static"] = static
    else:
        raise FileNotFoundError(
            f"no CableDyn output {standalone.name} or {coupled.name} next to {out_root}"
        )
    return files


def _static_profile(path: Path, line_id: int) -> FloatArray:
    """X, Y, Z of the line's nodes from a static profile table."""
    rows = path.read_text(encoding="utf-8").splitlines()
    if len(rows) < 4:
        raise CableSchemaError(f"{path.name}: incomplete static profile")
    header = [cell.strip() for cell in rows[1].split("\t")]
    units = [cell.strip() for cell in rows[2].split("\t")]
    for name, unit in (
        ("LineID", "(-)"),
        ("Node", "(-)"),
        ("X", "(m)"),
        ("Y", "(m)"),
        ("Z", "(m)"),
    ):
        if name not in header or units[header.index(name)] != unit:
            raise CableSchemaError(f"{path.name}: static column {name} {unit} missing")
    take = [header.index(name) for name in ("LineID", "Node", "X", "Y", "Z")]
    data = np.array(
        [[float(row.split("\t")[i]) for i in take] for row in rows[3:] if row.strip()]
    )
    mine = data[data[:, 0] == line_id]
    if mine.size == 0 or np.any(np.diff(mine[:, 1]) != 1) or mine[0, 1] != 1:
        raise CableSchemaError(f"{path.name}: line {line_id} rows missing or unordered")
    return np.asarray(mine[:, 2:5])


def _element_arcs(path: Path, line_id: int) -> FloatArray:
    rows = path.read_text(encoding="utf-8").splitlines()
    header = [cell.strip() for cell in rows[1].split("\t")]
    units = [cell.strip() for cell in rows[2].split("\t")]
    for name in ("LineID", "Element", "ReferenceArcStart", "ReferenceArcEnd"):
        if name not in header:
            raise CableSchemaError(f"{path.name}: column {name} missing")
    if units[header.index("ReferenceArcStart")] != "(m)":
        raise CableSchemaError(f"{path.name}: ReferenceArcStart must be in (m)")
    take = [
        header.index(n)
        for n in ("LineID", "Element", "ReferenceArcStart", "ReferenceArcEnd")
    ]
    data = np.array(
        [[float(row.split("\t")[i]) for i in take] for row in rows[3:] if row.strip()]
    )
    mine = data[data[:, 0] == line_id]
    if mine.size == 0 or np.any(np.diff(mine[:, 1]) != 1):
        raise CableSchemaError(f"{path.name}: elements of line {line_id} unordered")
    if np.any(np.abs(mine[1:, 2] - mine[:-1, 3]) > 1e-6 * max(1.0, float(mine[-1, 3]))):
        raise CableSchemaError(f"{path.name}: element arcs of line {line_id} have gaps")
    return np.concatenate((mine[:1, 2], mine[:, 3]))


def read_cabledyn_sections(
    out_root: str | Path,
    deck: str | Path | CableDynDeck,
    sections: Mapping[str, float],
    *,
    line_id: int = 1,
    frame: Frame = "arc",
    curvature_source: CurvatureSource = "auto",
    tangent_placement: TangentPlacement = "auto",
    reference_vector: Sequence[float] = (0.0, 0.0, 1.0),
    pressure: PressureModel = "hydrostatic",
    window_nodes: int = 3,
    tension_source: Literal["auto", "segments", "nodes"] = "auto",
    curvature_check_rtol: float = 0.05,
    curvature_check_end_rtol: float = 0.25,
    curvature_check_atol_1_m: float | None = None,
    chunk_rows: int = 4096,
) -> CableDynRead:
    """Read the histories of named material sections from one CableDyn run.

    Parameters
    ----------
    out_root : str or pathlib.Path
        Output root given to the driver (``<root>.out`` and its line files) or
        the OpenFAST root (``<root>.CD.out``).
    deck : str, pathlib.Path or CableDynDeck
        The deck that produced the run.
    sections : mapping of str to float
        Section name -> unstretched arc length from End A [m].
    line_id : int
        Deck ``LINES`` id.
    frame : {"arc", "section"}
        Twist-free transport of the section axes (module notes).
    curvature_source : {"auto", "tangent", "positions"}
        ``"auto"`` uses the tangent channels when every section has them and
        positions otherwise; the choice is recorded.
    tangent_placement : {"auto", "chord"}
        Where the ``Dec``/``Azi`` values sit along the line. ``"auto"``
        (CableDyn 0.4.2 and later): at the node on a finite-EI line, at the
        mid-arc of the chord on an ``EI = 0`` line. ``"chord"``: at the
        mid-arc of the chord on every line, for the output of CableDyn 0.4.1
        and earlier. The placement is recorded in ``checks``.
    reference_vector : sequence of 3 floats
        Seed vector ``a`` of the frame (default global +z).
    pressure : {"hydrostatic", "none"}
        External pressure model; ``"none"`` declares ``p_o = 0`` explicitly.
    window_nodes : int
        Nodes on each side of the section in the curvature spline.
    tension_source : {"auto", "segments", "nodes"}
        ``"segments"``: element-midpoint axial forces of ``.Line<L>.t.out``,
        linear between element mid-arcs; ``"nodes"``: ``Ten<L>N<J>`` channels
        (length-weighted element means; an end node carries the end force),
        linear between nodes. ``"auto"`` uses the segment file when written.
    curvature_check_rtol, curvature_check_end_rtol, curvature_check_atol_1_m
        Tolerance of the check against ``Curv<L>N<J>`` channels: interior and
        end nodes relative to the node's largest curvature, plus an absolute
        floor (default 2e-5 1/m from tangents, three times the estimated
        resolution from positions).
    chunk_rows : int
        Rows per streamed chunk.

    Returns
    -------
    CableDynRead
    """
    if frame not in ("arc", "section"):
        raise CableSchemaError("frame must be 'arc' or 'section'")
    if curvature_source not in ("auto", "tangent", "positions"):
        raise CableSchemaError("curvature_source must be auto, tangent or positions")
    if tangent_placement not in ("auto", "chord"):
        raise CableSchemaError("tangent_placement must be 'auto' or 'chord'")
    if tension_source not in ("auto", "segments", "nodes"):
        raise CableSchemaError("tension_source must be auto, segments or nodes")
    if pressure not in ("hydrostatic", "none"):
        raise CableSchemaError("pressure must be 'hydrostatic' or 'none'")
    if (
        isinstance(window_nodes, bool)
        or not isinstance(window_nodes, int)
        or window_nodes < 2
    ):
        raise CableSchemaError("window_nodes must be an integer of at least 2")
    if (
        isinstance(chunk_rows, bool)
        or not isinstance(chunk_rows, int)
        or chunk_rows < 16
    ):
        raise CableSchemaError("chunk_rows must be an integer of at least 16")
    reference = np.asarray(reference_vector, dtype=np.float64)
    if (
        reference.shape != (3,)
        or not np.isfinite(reference).all()
        or not reference.any()
    ):
        raise CableSchemaError("reference_vector must be a finite nonzero 3-vector")
    reference = reference / np.linalg.norm(reference)
    if not sections:
        raise CableSchemaError("request at least one section")
    deck_obj = deck if isinstance(deck, CableDynDeck) else read_cabledyn_deck(deck)
    line = deck_obj.line(line_id)
    root = Path(out_root)
    paths = _discover(root, line_id)
    checks: dict[str, Any] = {"deck_defaults_used": list(deck_obj.defaults_used)}

    arcs = line.node_arc_lengths()
    if "elements" in paths:
        element_arcs = _element_arcs(paths["elements"], line_id)
        if element_arcs.size != arcs.size or not np.allclose(
            element_arcs, arcs, rtol=0.0, atol=1e-6 * max(1.0, line.length_m)
        ):
            raise CableSchemaError(
                f"{paths['elements'].name}: mesh of line {line_id} differs from the "
                f"deck sections ({element_arcs.size} against {arcs.size} nodes)"
            )
        arcs = element_arcs
        checks["arc_lengths"] = "deck sections, confirmed by .elements.out"
    elif deck_obj.adaptive_mesh:
        raise CableSchemaError(
            "adaptive_mesh may change the mesh; the .elements.out file is needed "
            "to locate the nodes"
        )
    else:
        checks["arc_lengths"] = "deck sections"
    n_nodes = arcs.size
    for name, arc in sections.items():
        if not name.strip() or not math.isfinite(arc) or not 0.0 <= arc <= arcs[-1]:
            raise CableSchemaError(
                f"section {name!r} at {arc} m is outside line {line_id} "
                f"(0 to {arcs[-1]:.6g} m)"
            )

    main = _main_table(paths["main"])
    channels = _channel_index(main)
    positions_table = None
    if "positions" in paths:
        columns = [f"Node{j}{axis}(m)" for j in range(1, n_nodes + 1) for axis in "XYZ"]
        positions_table = _line_table(paths["positions"], "node positions", columns)
    tensions_table = None
    if "tensions" in paths:
        columns = [f"Segment{k}Tension(N)" for k in range(1, n_nodes)]
        tensions_table = _line_table(paths["tensions"], "segment tensions", columns)

    plans = [
        _Plan(
            name,
            float(arc),
            (e := _element_of(arcs, arc)),
            _window(e, n_nodes, window_nodes),
        )
        for name, arc in sections.items()
    ]
    for key in sorted(k for k in channels if k[0] == "curv" and k[1] == line_id):
        node = key[2] - 1
        if 0 <= node < n_nodes:
            element = min(node, n_nodes - 2)
            plans.append(
                _Plan(
                    f"Curv{line_id}N{key[2]}",
                    float(arcs[node]),
                    element,
                    _window(element, n_nodes, window_nodes),
                    probe=True,
                    node=node,
                )
            )

    def has_tangent(node: int) -> bool:
        return all((kind, line_id, node + 1) in channels for kind in ("dec", "azi"))

    def has_position(node: int) -> bool:
        return positions_table is not None or all(
            (kind, line_id, node + 1) in channels for kind in ("px", "py", "pz")
        )

    sections_plans = [plan for plan in plans if not plan.probe]
    if curvature_source == "auto":
        tangent_ok = all(has_tangent(j) for plan in sections_plans for j in plan.window)
        source = "tangent" if tangent_ok else "positions"
    else:
        source = curvature_source
    needed_curvature = has_tangent if source == "tangent" else has_position
    for plan in sections_plans:
        missing = [j + 1 for j in plan.window if not needed_curvature(j)]
        if missing:
            what = (
                f"L{line_id}N<J>Dec and L{line_id}N<J>Azi channels"
                if source == "tangent"
                else f".Line{line_id}.p.out or L{line_id}N<J>p{{x,y,z}} channels"
            )
            raise CableSchemaError(
                f"section {plan.name!r}: curvature from {source} needs {what} "
                f"at nodes {missing}"
            )
    plans = [
        plan
        for plan in plans
        if not plan.probe or all(needed_curvature(j) for j in plan.window)
    ]
    checks["curvature_source"] = source
    finite_ei = any(
        deck_obj.line_type(sec.line_type).ei_Nm2 > 0.0 for sec in line.sections
    )
    if source == "tangent":
        if tangent_placement == "chord":
            checks["tangent_placement"] = (
                "chord mid-arc (tangent_placement='chord': CableDyn 0.4.1 and earlier)"
            )
        elif finite_ei:
            checks["tangent_placement"] = (
                "node (finite-EI line: node tangent, CableDyn 0.4.2 and later)"
            )
        else:
            checks["tangent_placement"] = "chord mid-arc (EI = 0 line)"

    # Nodes whose positions are needed: frame transport (arc), elevation.
    pos_nodes: set[int] = set()
    if frame == "arc":
        last = max(plan.element for plan in sections_plans) + 1
        pos_nodes.update(range(0, last + 1))
    if source == "positions":
        for plan in plans:
            pos_nodes.update(plan.window)
    if pressure == "hydrostatic":
        for plan in sections_plans:
            pos_nodes.update((plan.element, plan.element + 1))
    missing_positions = [j + 1 for j in sorted(pos_nodes) if not has_position(j)]
    if missing_positions:
        reason = (
            "the arc-length frame transport"
            if frame == "arc"
            else "the section elevation"
        )
        raise CableSchemaError(
            f"{reason} needs node positions (.Line{line_id}.p.out or "
            f"L{line_id}N<J>p{{x,y,z}} channels) at nodes "
            f"{missing_positions[:10]}{'...' if len(missing_positions) > 10 else ''}"
        )
    pos_order = sorted(pos_nodes)
    pos_slot = {node: i for i, node in enumerate(pos_order)}
    tan_order = (
        sorted({j for plan in plans for j in plan.window})
        if source == "tangent"
        else []
    )

    # Tension source per section.
    use_segments = tension_source == "segments" or (
        tension_source == "auto" and tensions_table is not None
    )
    if use_segments and tensions_table is None:
        raise CableSchemaError(
            f"tension_source='segments' needs {root.name}.Line{line_id}.t.out "
            "(LINES Outputs flag t)"
        )
    tension_label = (
        "segment tensions (.t.out, element midpoints)"
        if use_segments
        else "node channels Ten<L>N<J>"
    )
    if not use_segments:
        for plan in sections_plans:
            for node in (plan.element, plan.element + 1):
                if ("ten", line_id, node + 1) not in channels:
                    raise CableSchemaError(
                        f"section {plan.name!r}: tension needs .Line{line_id}.t.out "
                        f"or Ten{line_id}N{node + 1} in the main output"
                    )
    torsion_nodes = [(plan.element, plan.element + 1) for plan in sections_plans]
    twist_available = all(
        ("twist", line_id, n + 1) in channels for pair in torsion_nodes for n in pair
    )
    torque_available = all(
        ("torq", line_id, n + 1) in channels for pair in torsion_nodes for n in pair
    )
    if line.torsion_restrained and not twist_available:
        raise CableSchemaError(
            f"line {line_id} is restrained in torsion; the material frame needs "
            f"Twist{line_id}N<J> channels at the nodes bracketing every section"
        )

    # Main-table column plan.
    main_cols: list[int] = [0]
    slot: dict[tuple[str, int], int] = {}

    def want(kind: str, node: int) -> None:
        key = (kind, node)
        if key not in slot:
            slot[key] = len(main_cols)
            main_cols.append(channels[(kind, line_id, node + 1)])

    for node in tan_order:
        want("dec", node)
        want("azi", node)
    if positions_table is None:
        for node in pos_order:
            for kind in ("px", "py", "pz"):
                want(kind, node)
    if not use_segments:
        for plan in sections_plans:
            want("ten", plan.element)
            want("ten", plan.element + 1)
    if twist_available:
        for plan in sections_plans:
            want("twist", plan.element)
            want("twist", plan.element + 1)
    if torque_available:
        for plan in sections_plans:
            want("torq", plan.element)
            want("torq", plan.element + 1)
    probe_nodes = [plan.node for plan in plans if plan.probe]
    for node in probe_nodes:
        want("curv", node)
    tension_check_nodes = []
    if tensions_table is not None:
        tension_check_nodes = [
            k[2] - 1
            for k in channels
            if k[0] == "ten" and k[1] == line_id and 1 < k[2] < n_nodes
        ]
        for node in tension_check_nodes:
            want("ten", node)

    streams: list[Generator[FloatArray, None, None]] = [
        _chunks(main, main_cols, chunk_rows)
    ]
    if positions_table is not None and pos_order:
        pos_cols = [0] + [
            1 + 3 * node + axis for node in pos_order for axis in range(3)
        ]
        streams.append(_chunks(positions_table, pos_cols, chunk_rows))
    elem_t: FloatArray = np.empty((0, 0))
    ten_slot: dict[int, int] = {}
    if tensions_table is not None:
        mirror = [n_nodes - 1 - node for node in tension_check_nodes]
        elements = sorted(
            {
                e
                for plan in sections_plans
                for e in (plan.element - 1, plan.element, plan.element + 1)
                if 0 <= e < n_nodes - 1
            }
            | {e for node in tension_check_nodes + mirror for e in (node - 1, node)}
        )
        ten_slot = {e: i for i, e in enumerate(elements)}
        streams.append(
            _chunks(tensions_table, [0] + [1 + e for e in elements], chunk_rows)
        )

    reference_l0 = np.diff(arcs)
    midpoints = 0.5 * (arcs[:-1] + arcs[1:])
    # Arc length of each Dec/Azi value: the node's own tangent of a finite-EI
    # line (CableDyn 0.4.2+) at the node; a chord (central, one-sided at the
    # ends) at its mid-arc, the point of a smooth arc whose tangent it is.
    if tangent_placement == "auto" and finite_ei:
        tangent_arcs = arcs
    else:
        tangent_arcs = np.array(
            [
                0.5 * (arcs[max(j - 1, 0)] + arcs[min(j + 1, n_nodes - 1)])
                for j in range(n_nodes)
            ]
        )
    state_a: FloatArray | None = None
    state_section: dict[str, FloatArray | None] = {
        plan.name: None for plan in sections_plans
    }
    first = True
    n_rows = 0
    position_floor = math.nan
    max_t0_error = math.nan
    stretch_range = (math.nan, math.nan)
    tension_check = 0.0
    tension_match = tension_mirror = 0.0

    def append(plan: _Plan, key: str, values: FloatArray) -> None:
        plan.data.setdefault(key, []).append(values)

    # Every stream is closed on the way out (its file handle with it), also
    # when a check below raises or the files end at different rows.
    try:
        while True:
            parts = [next(stream, None) for stream in streams]
            if all(part is None for part in parts):
                break
            if any(part is None for part in parts):
                raise CableSchemaError(
                    "CableDyn output files have different row counts"
                )
            chunk = tuple(part for part in parts if part is not None)
            if len({part.shape[0] for part in chunk}) != 1:
                raise CableSchemaError(
                    "CableDyn output files have different row counts"
                )
            time = chunk[0][:, 0]
            for part in chunk[1:]:
                time_tol = 1e-9 * max(1.0, float(np.max(np.abs(time))))
                if not np.allclose(part[:, 0], time, rtol=0.0, atol=time_tol):
                    raise CableSchemaError(
                        "time columns of the main and per-line files differ"
                    )
            values = chunk[0]
            nt = time.size
            positions = np.empty((nt, len(pos_order), 3))
            if positions_table is not None and pos_order:
                positions = chunk[1][:, 1:].reshape(nt, len(pos_order), 3)
            elif pos_order:
                for node in pos_order:
                    for axis, kind in enumerate(("px", "py", "pz")):
                        positions[:, pos_slot[node], axis] = values[
                            :, slot[(kind, node)]
                        ]
            if tensions_table is not None:
                elem_t = chunk[-1][:, 1:]
            if first:
                if "static" in paths and pos_order:
                    static = _static_profile(paths["static"], line_id)
                    if static.shape[0] != n_nodes:
                        raise CableSchemaError(
                            f"{paths['static'].name}: {static.shape[0]} nodes, "
                            f"expected {n_nodes}"
                        )
                    max_t0_error = float(
                        np.max(np.abs(static[pos_order] - positions[0]))
                    )
                    if max_t0_error > 2e-6 * max(1.0, float(np.max(np.abs(static)))):
                        raise CableSchemaError(
                            "first position row differs from the static profile by "
                            f"{max_t0_error:.3g} m: files from different runs?"
                        )
                if len(pos_order) > 1:
                    contiguous = [j for j in pos_order if j + 1 in pos_slot]
                    if contiguous:
                        chord = np.array(
                            [
                                np.linalg.norm(
                                    positions[0, pos_slot[j + 1]]
                                    - positions[0, pos_slot[j]]
                                )
                                for j in contiguous
                            ]
                        )
                        stretch = chord / reference_l0[contiguous] - 1.0
                        stretch_range = (float(stretch.min()), float(stretch.max()))
                        if stretch_range[0] < -0.02 or stretch_range[1] > 0.05:
                            raise CableSchemaError(
                                "element chord/reference length - 1 spans "
                                f"{stretch_range}: "
                                "the deck sections do not describe the output mesh"
                            )
                    magnitude = float(np.max(np.abs(positions[0])))
                    position_floor = (
                        0.5
                        * 10.0 ** (math.floor(math.log10(max(magnitude, 1e-30))) - 7)
                        / float(np.min(reference_l0)) ** 2
                    )
            # Element tension consistency (ordering check).
            if tensions_table is not None and tension_check_nodes:
                for node in tension_check_nodes:
                    channel = values[:, slot[("ten", node)]]
                    scale = np.maximum(np.abs(channel), 1.0)
                    for j, label in ((node, "match"), (n_nodes - 1 - node, "mirror")):
                        left, right = elem_t[:, ten_slot[j - 1]], elem_t[:, ten_slot[j]]
                        w_left, w_right = reference_l0[j - 1], reference_l0[j]
                        mean = (w_left * left + w_right * right) / (w_left + w_right)
                        error = np.abs(channel - mean) / scale
                        if label == "match":
                            tension_check = max(tension_check, float(np.max(error)))
                            tension_match += float(np.sum(error))
                        else:
                            tension_mirror += float(np.sum(error))
            # Frame transport along the arc (all sections share it).
            arc_e1: dict[int, FloatArray] = {}
            if frame == "arc":
                last = max(plan.element for plan in sections_plans)
                chords = _unit(
                    positions[:, [pos_slot[j + 1] for j in range(last + 1)]]
                    - positions[:, [pos_slot[j] for j in range(last + 1)]],
                    "element chord",
                )
                e1_a, state_a = _time_transport(
                    chords[:, 0], state_a, reference, f"line {line_id} End A"
                )
                e1 = e1_a
                arc_e1[0] = e1
                for k in range(1, last + 1):
                    e1 = _transport(e1, chords[:, k - 1], chords[:, k])
                    arc_e1[k] = e1
            for plan in plans:
                if source == "tangent":
                    dec = np.radians(
                        np.stack(
                            [values[:, slot[("dec", j)]] for j in plan.window], axis=1
                        )
                    )
                    azi = np.radians(
                        np.stack(
                            [values[:, slot[("azi", j)]] for j in plan.window], axis=1
                        )
                    )
                    samples = np.stack(
                        (
                            np.sin(dec) * np.cos(azi),
                            np.sin(dec) * np.sin(azi),
                            np.cos(dec),
                        ),
                        axis=-1,
                    )
                    value, d1, _ = _spline(tangent_arcs[plan.window], samples, plan.arc)
                    tangent = _unit(value, "section tangent")
                    kappa = np.cross(tangent, d1)
                else:
                    window = positions[:, [pos_slot[j] for j in plan.window]]
                    value, d1, d2 = _spline(arcs[plan.window], window, plan.arc)
                    speed = np.linalg.norm(d1, axis=-1, keepdims=True)
                    tangent = d1 / speed
                    kappa = np.cross(d1, d2) / speed**3
                norm = np.linalg.norm(kappa, axis=-1)
                append(plan, "norm", norm)
                if plan.probe:
                    node = plan.node
                    append(plan, "channel", values[:, slot[("curv", node)]])
                    continue
                if frame == "arc":
                    axis1 = _transport(
                        arc_e1[plan.element], chords[:, plan.element], tangent
                    )
                else:
                    axis1, state = _time_transport(
                        tangent,
                        state_section[plan.name],
                        reference,
                        f"section {plan.name!r}",
                    )
                    state_section[plan.name] = state
                axis2 = np.cross(tangent, axis1)
                weight = (plan.arc - arcs[plan.element]) / reference_l0[plan.element]
                if twist_available:
                    twist = np.radians(
                        (1 - weight) * values[:, slot[("twist", plan.element)]]
                        + weight * values[:, slot[("twist", plan.element + 1)]]
                    )
                    cos, sin = np.cos(twist)[:, None], np.sin(twist)[:, None]
                    axis1, axis2 = cos * axis1 + sin * axis2, -sin * axis1 + cos * axis2
                    append(plan, "twist", twist)
                if torque_available:
                    append(
                        plan,
                        "torque",
                        (1 - weight) * values[:, slot[("torq", plan.element)]]
                        + weight * values[:, slot[("torq", plan.element + 1)]],
                    )
                append(plan, "kx", np.sum(kappa * axis1, axis=-1))
                append(plan, "ky", np.sum(kappa * axis2, axis=-1))
                if (
                    pos_order
                    and plan.element in pos_slot
                    and plan.element + 1 in pos_slot
                ):
                    if source == "positions":
                        z = value[:, 2]
                    else:
                        z = (1 - weight) * positions[
                            :, pos_slot[plan.element], 2
                        ] + weight * positions[:, pos_slot[plan.element + 1], 2]
                    append(plan, "z", z)
                if use_segments:
                    if plan.arc <= midpoints[0]:
                        ten = elem_t[:, ten_slot[0]]
                    elif plan.arc >= midpoints[-1]:
                        ten = elem_t[:, ten_slot[n_nodes - 2]]
                    else:
                        lower = int(
                            np.searchsorted(midpoints, plan.arc, side="right") - 1
                        )
                        w = (plan.arc - midpoints[lower]) / (
                            midpoints[lower + 1] - midpoints[lower]
                        )
                        ten = (1 - w) * elem_t[:, ten_slot[lower]] + w * elem_t[
                            :, ten_slot[lower + 1]
                        ]
                else:
                    ten = (1 - weight) * values[
                        :, slot[("ten", plan.element)]
                    ] + weight * values[:, slot[("ten", plan.element + 1)]]
                append(plan, "tension", ten)
            append(plans[0], "time", time)
            n_rows += nt
            first = False
    finally:
        for stream in streams:
            stream.close()

    if n_rows < 2:
        raise CableSchemaError(f"{paths['main'].name}: fewer than two output rows")
    time = np.concatenate(plans[0].data["time"])
    steps = np.diff(time)
    if np.any(steps <= 0):
        raise CableSchemaError(f"{paths['main'].name}: time does not increase")
    if not np.allclose(steps, steps[0], rtol=1e-6, atol=1e-9):
        raise CableSchemaError(f"{paths['main'].name}: the output step is not uniform")
    if deck_obj.dt_s is not None and not math.isclose(
        steps[0], deck_obj.dt_s, rel_tol=1e-6
    ):
        raise CableSchemaError(
            f"output step {steps[0]:g} s differs from the deck dtM {deck_obj.dt_s:g} s"
        )
    checks["time_step_s"] = float(steps[0])
    checks["rows"] = n_rows
    checks["first_row_vs_static_m"] = max_t0_error
    checks["element_stretch_t0"] = list(stretch_range)
    if source == "positions":
        checks["position_resolution_curvature_1_m"] = position_floor
    checks["tension_source"] = tension_label
    if tensions_table is not None and tension_check_nodes:
        # The node channel averages element-mean forces, .t.out holds the
        # element-midpoint force: on a coarse mesh the two differ (the
        # pointwise field oscillates about the element mean). The segment
        # order must still fit the node channels better than its mirror.
        checks["node_tension_vs_segment_mean_max_rel"] = tension_check
        if tension_match > tension_mirror:
            raise CableSchemaError(
                "the .t.out segment tensions fit the Ten<L>N<J> channels no better "
                "in their stated order than reversed: the segment order or the mesh "
                "differs from the deck"
            )

    atol = curvature_check_atol_1_m
    if atol is None:
        floor = position_floor if math.isfinite(position_floor) else 1e-6
        atol = 2e-5 if source == "tangent" else 3.0 * max(floor, 1e-6)
    curvature_checks: dict[str, dict[str, Any]] = {}
    for plan in plans:
        if not plan.probe:
            continue
        node = plan.node + 1
        mine = np.concatenate(plan.data["norm"])
        channel = np.concatenate(plan.data["channel"])
        error = float(np.max(np.abs(mine - channel)))
        # End nodes and their neighbours: the spline window is one-sided
        # there (and starts from the one-sided end chord on chord output),
        # CableDyn reports the end element's own curvature.
        end = node <= 2 or node >= n_nodes - 1
        allowed = atol + (
            curvature_check_end_rtol if end else curvature_check_rtol
        ) * float(np.max(np.abs(channel)))
        # At an end node CableDyn reports the end element's curvature, which
        # carries the end condition (clamped or pinned); the spline
        # extrapolation does not impose it. Reported, not enforced.
        enforced = 1 < node < n_nodes
        curvature_checks[f"N{node}"] = {
            "max_abs_error_1_m": error,
            "max_channel_1_m": float(np.max(np.abs(channel))),
            "allowed_1_m": allowed if enforced else None,
            "near_end": end,
        }
    failed = {
        name: item
        for name, item in curvature_checks.items()
        if item["allowed_1_m"] is not None
        and item["max_abs_error_1_m"] > item["allowed_1_m"]
    }
    if failed:
        checked = sum(
            1 for item in curvature_checks.values() if item["allowed_1_m"] is not None
        )

        def share(item: dict[str, Any]) -> float:
            return float(item["max_abs_error_1_m"]) / max(
                float(item["max_channel_1_m"]), 1e-300
            )

        worst_name = max(failed, key=lambda name: share(failed[name]))
        worst = failed[worst_name]
        percent = 100.0 * share(worst)
        rtol = curvature_check_end_rtol if worst["near_end"] else curvature_check_rtol
        placement = ""
        if source == "tangent" and finite_ei:
            placement = (
                f" The tangents were placed at the {checks['tangent_placement']}; "
                + (
                    "output of CableDyn 0.4.2 and later needs the default "
                    "tangent_placement='auto'."
                    if tangent_placement == "chord"
                    else "output of CableDyn 0.4.1 and earlier needs "
                    "tangent_placement='chord'."
                )
            )
        raise CableSchemaError(
            "the curvature reconstructed from the node "
            f"{'tangents' if source == 'tangent' else 'positions'} differs from "
            f"CableDyn's own curvature output (channel Curv{line_id}{worst_name}, "
            "and the other curvature channels) by more than the curvature check "
            f"tolerance ({100.0 * rtol:g} % of the largest "
            "curvature there, plus the resolution of the "
            f"{'angles' if source == 'tangent' else 'positions'}) at "
            f"{len(failed)} of {checked} checked nodes; at node "
            f"{worst_name.removeprefix('N')} by {worst['max_abs_error_1_m']:.3g} "
            f"1/m, {percent:.0f} % of its largest value "
            f"{worst['max_channel_1_m']:.3g} 1/m. Either the deck, the mesh or "
            "the channels do not match the output, or the mesh is too coarse to "
            "resolve the curvature there (CableDyn reports the larger one-sided "
            "element value, which on a coarse mesh departs from the curvature of "
            "the deformed line): refine the mesh, or raise the curvature check "
            f"tolerance if a difference of {percent:.0f} % is acceptable."
            f"{placement}"
        )
    checks["curvature_vs_channels"] = curvature_checks

    files = tuple(
        replace(_digest(path), role=role) for role, path in sorted(paths.items())
    ) + (replace(_digest(deck_obj.path), role="deck"),)
    histories: dict[str, CableDynSectionHistory] = {}
    for plan in sections_plans:
        data = {key: np.concatenate(parts) for key, parts in plan.data.items()}
        elevation = data.get("z")
        p_ext: FloatArray | None = None
        if pressure == "hydrostatic":
            # The section nodes joined the required positions above.
            assert elevation is not None
            p_ext = deck_obj.rho_w_kg_m3 * deck_obj.g_m_s2 * np.maximum(-elevation, 0.0)
        histories[plan.name] = CableDynSectionHistory(
            name=plan.name,
            arc_length_m=plan.arc,
            line_id=line_id,
            time_s=time,
            effective_tension_N=data["tension"],
            rotation_curvature_x_1_m=data["kx"],
            rotation_curvature_y_1_m=data["ky"],
            curvature_norm_1_m=data["norm"],
            elevation_m=elevation,
            external_pressure_Pa=p_ext,
            torque_Nm=data.get("torque"),
            twist_rad=data.get("twist"),
            frame=frame,
            curvature_source=source,
            tension_source=tension_label,
            pressure_model=pressure,
            source=f"{paths['main'].name}:Line{line_id}@{plan.arc:g}m",
            provenance=Provenance(
                method="CableDyn section history (cablexsec.io.cabledyn)",
                citations=(),
                gates=(
                    "synthetic CableDyn-format files with known curvature vectors",
                    "reconstructed curvature magnitude against Curv<L>N<J> channels",
                ),
                limitations=(
                    "curvature vector reconstructed from the solved centreline; "
                    f"source: {source}"
                    + (
                        f"; tangents placed at the {checks['tangent_placement']}"
                        if source == "tangent"
                        else ""
                    ),
                    f"twist-free {frame} frame seeded by e2 = a x t, "
                    f"a = {tuple(float(v) for v in reference)}; hang-off roll about "
                    "its own tangent is not represented",
                    "material twist from Twist channels"
                    if twist_available
                    else "no material twist (line not restrained in torsion)",
                    "hydrostatic external pressure only (no wave dynamic pressure)"
                    if pressure == "hydrostatic"
                    else "external pressure declared zero",
                    "histories inherit the global model's discretisation, "
                    "environmental and boundary assumptions",
                ),
            ),
        )
    return CableDynRead(
        histories=MappingProxyType(histories),
        files=files,
        deck=deck_obj,
        line_id=line_id,
        checks=MappingProxyType(checks),
    )


def section_channels(
    deck: str | Path | CableDynDeck,
    sections: Mapping[str, float],
    *,
    line_id: int = 1,
    window_nodes: int = 3,
    positions: bool = True,
    check_curvature: bool = True,
) -> tuple[str, ...]:
    """``OUTPUTS`` channel names a deck needs for the tangent curvature source.

    For every section: ``L<L>N<J>Dec``/``Azi`` on the spline window,
    ``Ten<L>N<J>`` and (``positions``) ``L<L>N<J>p{x,y,z}`` at the two bracketing
    nodes, ``Curv<L>N<J>`` there for the consistency check, and
    ``Twist``/``Torq`` on a line restrained in torsion. With these channels
    alone use ``frame="section"``; the default ``frame="arc"`` also needs the
    ``p`` flag of the line (node positions from End A to the section).
    """
    deck_obj = deck if isinstance(deck, CableDynDeck) else read_cabledyn_deck(deck)
    line = deck_obj.line(line_id)
    arcs = line.node_arc_lengths()
    names: list[str] = []

    def add(name: str) -> None:
        if name not in names:
            names.append(name)

    for name, arc in sections.items():
        if not math.isfinite(arc) or not 0.0 <= arc <= arcs[-1]:
            raise CableSchemaError(f"section {name!r} at {arc} m is outside the line")
        element = _element_of(arcs, arc)
        for node in _window(element, arcs.size, window_nodes):
            add(f"L{line_id}N{node + 1}Dec")
            add(f"L{line_id}N{node + 1}Azi")
        for node in (element + 1, element + 2):
            add(f"Ten{line_id}N{node}")
            if check_curvature:
                add(f"Curv{line_id}N{node}")
            if positions:
                for axis in "xyz":
                    add(f"L{line_id}N{node}p{axis}")
            if line.torsion_restrained:
                add(f"Twist{line_id}N{node}")
                add(f"Torq{line_id}N{node}")
    return tuple(names)
