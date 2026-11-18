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

Refined meshes
--------------
The static solve may refine the mesh of a finite-EI line (``adaptive_mesh``,
``tensile_safety`` or the automatic refinement; CableDyn ``doc/outputs.rst``,
node channels on a refined line). Refinement splits deck elements, so every
deck node stays a mesh node at its own unstretched arc length. The ``<J>`` of
the node channels names a deck node (since CableDyn 0.2.0); ``.static.out``,
``.elements.out`` and the per-line files list every node of the refined mesh.
The reader locates the deck nodes by the ``DeckNode`` column of
``.static.out`` or the ``# DeckNode`` line of ``.Line<L>.p.out`` (failing both,
by their arc lengths in ``.elements.out``), requires every file to list the
same nodes and the deck nodes to lie at the arc lengths of the deck sections,
and reads positions and segment tensions on the refined mesh, which
``.elements.out`` places. Without that file the positions are read at the deck
nodes only and the segment tensions not at all (``tension_source="auto"``
then takes the node channels).

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
restrained in torsion at both ends without these channels is rejected, and so
is a dynamic run of a line restrained at one end whose torsion CableDyn solves
(:attr:`CableDynLine.torsion_solved`): the polar inertia carries torsional
waves and a roll of the restrained end twists it. Its static state is
twist-free, unless its line type couples torque and tension (``GJ|K``): its
twist then follows the stretch, and the channels are needed in any run. A line
restrained at one end whose torsion CableDyn does not solve (the deck misses
one of CableDyn's rules for it, ``torsion_note``) runs as one without the
torsion columns and is read twist-free.

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
    "bodies": "bodies",
    "body": "bodies",
    "rods": "rods",
    "rod list": "rods",
    "rod properties": "rods",
    "equivalent buoyancy": "equivalent buoyancy",
    "equivalent sections": "equivalent buoyancy",
    "buoyancy sections": "equivalent buoyancy",
}


# --------------------------------------------------------------------------- #
# Deck                                                                        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CableDynLineType:
    """One ``LINE TYPES`` row (SI), read by position as CableDyn reads it.

    ``ea_N`` is ``None`` for a viscoelastic (``Es|Ed``, ``Es|alphaMBL|vbeta``)
    or Syrope ``EA`` column. ``gj_Nm2`` is the ``GJ`` of a 14-column row
    (``None`` on a 10-column row). The damping and coupling parts of the
    ``EI`` and ``GJ`` columns are kept as CableDyn resolves them:
    ``bending_damping_Nm2s`` is the ``Bb`` of ``EI|Bb``, ``EI|-zeta|f``
    (``Bb = zeta EI/(pi f)``) or ``EI|-zeta|f1|f2``
    (``Bb = zeta EI/(pi (f1 + f2))``), ``mass_damping_1_s`` the
    mass-proportional ``alpha = 4 pi zeta f1 f2/(f1 + f2)`` of the Rayleigh
    form, ``torque_tension_coupling_Nm`` the ``K`` of ``GJ|K``
    (``T = EA eps + K tau``, ``M = GJ tau + K eps``) and ``torsional_damping``
    the ``Bt`` of ``GJ|K|Bt`` [N m² s] (negative: ``-zeta`` of the critical
    torsional damping of an element). ``polar_inertia_kg_m`` is the ``Irn``
    [kg m²/m] of a 14-column row. ``ei_token`` and ``gj_token`` are the
    columns as written.

    ``diameter_m`` and ``mass_kg_m`` are the values CableDyn runs with. An
    ``EQUIVALENT BUOYANCY`` row of the type replaces them, as CableDyn does,
    by its diameter ``D`` and the dry mass
    ``rhoW pi D²/4 + w/g`` of its submerged weight ``w`` [N/m]
    (``equivalent_submerged_weight_N_m``; ``None`` without a row), with the
    deck's ``rhoW`` and ``g``. ``line_types_diameter_m`` and
    ``line_types_mass_kg_m`` keep the ``Diam`` and ``MassDenInAir`` of the
    ``LINE TYPES`` row.
    """

    name: str
    diameter_m: float
    mass_kg_m: float
    ea_N: float | None
    ei_Nm2: float
    gj_Nm2: float | None
    bending_damping_Nm2s: float = 0.0
    mass_damping_1_s: float = 0.0
    torque_tension_coupling_Nm: float = 0.0
    torsional_damping: float = 0.0
    polar_inertia_kg_m: float | None = None
    ei_token: str = ""
    gj_token: str | None = None
    line_types_diameter_m: float | None = None
    line_types_mass_kg_m: float | None = None
    equivalent_submerged_weight_N_m: float | None = None

    def __post_init__(self) -> None:
        # A type built by hand runs with the values it is given.
        if self.line_types_diameter_m is None:
            object.__setattr__(self, "line_types_diameter_m", self.diameter_m)
        if self.line_types_mass_kg_m is None:
            object.__setattr__(self, "line_types_mass_kg_m", self.mass_kg_m)


@dataclass(frozen=True)
class CableDynSection:
    """One ``SECTIONS`` row, End A first."""

    line_type: str
    length_m: float
    n_segments: int


@dataclass(frozen=True)
class CableDynLine:
    """A deck line in public order (End A -> End B, after any anchor-first swap).

    ``torsion_restrained`` is a line restrained in torsion at both ends;
    ``torsion_restrained_ends`` names every restrained end (``"a"``, ``"b"``,
    public order). ``torsion_solved`` is a finite-EI line whose torsion
    CableDyn solves: restrained at both ends, or at one end under CableDyn's
    rules for that case (every section gives ``GJ``, End B is ``Fixed``, a
    dynamic run is force-blended, a restrained End A is on no body but a
    Rigid6 one and on no rod end but a ``Fixed`` rod's); otherwise
    ``torsion_note`` gives CableDyn's reason, and the line runs as without
    the torsion columns.
    """

    line_id: int
    end_a: str
    end_b: str
    outputs: str
    sections: tuple[CableDynSection, ...]
    torsion_restrained: bool
    reversed_by_parser: bool
    torsion_restrained_ends: frozenset[str] = frozenset()
    torsion_solved: bool = False
    torsion_note: str = ""

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


#: CableDyn's token separators (``is_space``): space, tab, LF, VT, FF, CR.
_SPACES = " \t\n\v\f\r"


def _strip_comment(text: str) -> str:
    """A record without its comment, as CableDyn's ``scan_comment`` cuts it.

    ``#`` and ``!`` start a comment anywhere, inside a quoted token too (as
    in a file path, CableDyn ``doc/options.rst``); ``--`` starts one at the
    start of the record or after a separator, unless the record holds a
    ``---`` rule (a section header).
    """
    cuts = [cut for cut in (text.find("#"), text.find("!")) if cut >= 0]
    if "---" not in text:
        start = 0
        while (dash := text.find("--", start)) >= 0:
            if dash == 0 or text[dash - 1] in _SPACES:
                cuts.append(dash)
                break
            start = dash + 2
    if cuts:
        text = text[: min(cuts)]
    return text.strip(_SPACES)


#: Keywords of the options the reader uses and their CableDyn aliases.
_OPTION_ALIASES: dict[str, tuple[str, ...]] = {
    "g": ("g", "gravity"),
    "rhoW": ("rhoW", "WtrDnsty", "water_density"),
    "WtrDpth": ("WtrDpth", "water_depth"),
    "dtM": ("dtM", "dt"),
    "TMax": ("TMax",),
    "adaptive_mesh": ("adaptive_mesh", "adaptivemesh"),
    "alpha_force_blend": ("alpha_force_blend", "alphaforceblend"),
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


#: A decimal real as CableDyn spells one: sign, digits, point, exponent e/E/d/D.
_REAL = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?")
_INTEGER = re.compile(r"[+-]?\d+")
_NOT_FINITE = ("inf", "infinity", "nan")
#: Tokens as CableDyn's ``token_bounds`` splits a record: a token that opens
#: with a quote runs to the matching quote (separators kept) and on to the next
#: separator, or to the end of the record without one; any other token is a
#: run of non-separators.
_TOKEN = re.compile(
    r"\"[^\"]*\"[^ \t\n\v\f\r]*|\"[^\"]*$|'[^']*'[^ \t\n\v\f\r]*|'[^']*$"
    r"|[^ \t\n\v\f\r]+"
)


def _unquote(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1]
    return token.replace('"', "").replace("'", "")


def _numeric_shape(token: str) -> bool:
    """CableDyn's test of a number token (the header rows have none)."""
    word = token[1:] if token[:1] in "+-" else token
    return bool(_REAL.fullmatch(token)) or word.casefold() in _NOT_FINITE


def _float(token: str, what: str, path: Path) -> float:
    if not _REAL.fullmatch(token):
        word = token[1:] if token[:1] in "+-" else token
        if word.casefold() in _NOT_FINITE:
            raise CableSchemaError(f"{path.name}: {what} is not finite: {token!r}")
        raise CableSchemaError(f"{path.name}: {what} is not numeric: {token!r}")
    value = float(token.replace("d", "e").replace("D", "e"))
    if not math.isfinite(value):
        raise CableSchemaError(f"{path.name}: {what} is not finite: {token!r}")
    return value


def _int(token: str, what: str, path: Path) -> int:
    if not _INTEGER.fullmatch(token):
        raise CableSchemaError(f"{path.name}: {what} is not an integer: {token!r}")
    return int(token)


def _bars(token: str, what: str, path: Path, most: int) -> list[float]:
    """The ``|``-separated numbers of one column (at most ``most`` of them)."""
    parts = token.split("|")
    if len(parts) > most:
        raise CableSchemaError(
            f"{path.name}: {what} {token!r} has {len(parts)} '|'-separated parts, "
            f"at most {most}"
        )
    return [_float(part, what, path) for part in parts]


def _ei_column(token: str, name: str, path: Path) -> tuple[float, float, float]:
    """``EI``, bending damping ``Bb`` and mass-proportional ``alpha`` of the
    ``EI`` column: ``EI``, ``EI|Bb``, ``EI|-zeta|f`` or ``EI|-zeta|f1|f2``
    (CableDyn ``doc/driver_format.md``, LINE TYPES)."""
    what = f"LINE TYPES EI column of {name}"
    parts = _bars(token, what, path, 4)
    ei, bb, alpha = parts[0], 0.0, 0.0
    if ei < 0.0:
        raise CableSchemaError(f"{path.name}: {what} {token!r}: EI must be >= 0")
    if len(parts) == 2:
        if parts[1] < 0.0:
            raise CableSchemaError(
                f"{path.name}: {what} {token!r}: a negative bending damping is -zeta "
                "and needs the frequency it holds at, EI|-zeta|f"
            )
        bb = parts[1]
    elif len(parts) >= 3:
        zeta, frequencies = -parts[1], parts[2:]
        if not 0.0 < zeta < 1.0 or min(frequencies) <= 0.0:
            raise CableSchemaError(
                f"{path.name}: {what} {token!r}: EI|-zeta|f[|f2] takes a damping ratio "
                "0 < zeta < 1, written negative, and positive frequencies [Hz]"
            )
        if len(parts) == 3:
            bb = zeta * ei / (math.pi * frequencies[0])
        else:
            f1, f2 = frequencies
            bb = zeta * ei / (math.pi * (f1 + f2))
            alpha = 4.0 * math.pi * zeta * f1 * f2 / (f1 + f2)
    if (len(parts) >= 3 or bb > 0.0) and not ei > 0.0:
        raise CableSchemaError(
            f"{path.name}: LINE TYPES {name}: bending damping (EI|Bb) needs a "
            "finite-EI cable (EI > 0)"
        )
    return ei, bb, alpha


def _plain_ea(token: str, path: Path, name: str) -> float | None:
    """The ``EA`` of a linear-elastic type; ``None`` for the viscoelastic
    (``Es|Ed``, ``Es|alphaMBL|vbeta``) and Syrope (``SYROPE:file|a|b``) forms."""
    if "|" in token or token.casefold().startswith("syrope:"):
        return None
    return _float(token, f"EA of {name}", path)


def _data_rows(rows: list[list[str]]) -> list[list[str]]:
    """Rows of a table section with a number token: CableDyn skips the others
    (column-name and units rows), wherever they stand."""
    return [row for row in rows if any(_numeric_shape(token) for token in row)]


def _end_token(token: str, path: Path, line_id: int) -> str:
    end = token.casefold()
    if end in ("a", "enda", "end_a"):
        return "a"
    if end in ("b", "endb", "end_b"):
        return "b"
    raise CableSchemaError(
        f"{path.name}: END CONNECTIONS End of line {line_id} must be A or B: {token!r}"
    )


def _torsion_restrained(token: str, path: Path, line_id: int) -> bool:
    """``TorsStiffness``: Free/Zero/0 (none), Rigid/Infinity/Inf or a stiffness."""
    word = token.casefold()
    if word in ("free", "zero", "none", "-"):
        return False
    if word in ("rigid", "infinity", "inf"):
        return True
    value = _float(token, f"END CONNECTIONS TorsStiffness of line {line_id}", path)
    if value < 0.0:
        raise CableSchemaError(
            f"{path.name}: END CONNECTIONS TorsStiffness of line {line_id} must be "
            f"Free, Rigid or a stiffness >= 0: {token!r}"
        )
    return value > 0.0


#: Token counts of a LINE TYPES row: 10 or 14 columns, either followed by the
#: VIV columns Cl or Cl dF cF.
_LINE_TYPE_COLUMNS = (10, 11, 13, 14, 15, 17)


def _line_type(row: list[str], path: Path) -> CableDynLineType:
    """One LINE TYPES row by position, as CableDyn reads it:
    ``Name Diam MassDenInAir EA BA/-zeta EI [GAs GJ Irt Irn] Cd_n Cd_t Ca_n Ca_t
    [Cl [dF cF]]``."""
    name = row[0]
    if len(row) not in _LINE_TYPE_COLUMNS:
        raise CableSchemaError(
            f"{path.name}: LINE TYPES row {name!r} has {len(row)} columns; CableDyn "
            "reads 10 (Name Diam MassDenInAir EA BA/-zeta EI Cd_n Cd_t Ca_n Ca_t) "
            "or 14 (Name Diam MassDenInAir EA BA/-zeta EI GAs GJ Irt Irn Cd_n Cd_t "
            "Ca_n Ca_t), either followed by Cl [dF cF] (11, 13, 15 or 17)"
        )
    # The plain numeric columns, named by position (BA may hold Bs|Bd).
    finite_ei_row = len(row) >= 14
    names = (["GAs", "GJ", "Irt", "Irn"] if finite_ei_row else []) + [
        "Cd_n",
        "Cd_t",
        "Ca_n",
        "Ca_t",
        "Cl",
        "dF",
        "cF",
    ]
    values = {
        label: _float(column, f"{label} of {name}", path)
        for column, label in zip(row[6:], names, strict=False)
        if label != "GJ"
    }
    ea = _plain_ea(row[3], path, name)
    ei, bb, alpha = _ei_column(row[5], name, path)
    gj: float | None = None
    coupling = torsional_damping = 0.0
    polar_inertia: float | None = None
    gj_token: str | None = None
    if finite_ei_row:
        gj_token = row[7]
        what = f"LINE TYPES GJ column of {name}"
        parts = _bars(gj_token, what, path, 3)
        gj = parts[0]
        coupling = parts[1] if len(parts) > 1 else 0.0
        torsional_damping = parts[2] if len(parts) > 2 else 0.0
        polar_inertia = values["Irn"]
    # Magnitudes: a negative drag or added-mass coefficient would inject energy.
    if min(values[label] for label in ("Cd_n", "Cd_t", "Ca_n", "Ca_t")) < 0.0:
        raise CableSchemaError(
            f"{path.name}: LINE TYPES {name}: drag and added-mass coefficients "
            "(Cd_n Cd_t Ca_n Ca_t) must be non-negative"
        )
    if finite_ei_row:
        assert gj is not None and polar_inertia is not None
        if coupling != 0.0 and not (
            ei > 0.0 and ea is not None and gj > 0.0 and ea * gj > coupling**2
        ):
            raise CableSchemaError(
                f"{path.name}: LINE TYPES {name}: torque-tension coupling (GJ|K) "
                "needs a finite-EI, linear-elastic cable with GJ > 0 and K^2 < EA GJ"
            )
        if torsional_damping != 0.0 and not (
            ei > 0.0 and gj > 0.0 and polar_inertia > 0.0
        ):
            raise CableSchemaError(
                f"{path.name}: LINE TYPES {name}: torsional damping (GJ|K|Bt) needs a "
                "finite-EI cable with GJ > 0 and Irn > 0"
            )
        # CableDyn validates the four finite-EI values together after the row.
        extra = (values["GAs"], gj, values["Irt"], polar_inertia)
        if max(abs(value) for value in extra) > 0.0 and min(extra) <= 0.0:
            raise CableSchemaError(
                f"{path.name}: LINE TYPES {name}: explicit finite-EI GAs/GJ/Irt/Irn "
                f"must all be > 0 (or all 0): {' '.join(row[6:10])}"
            )
    return CableDynLineType(
        name=name,
        diameter_m=_float(row[1], f"Diam of {name}", path),
        mass_kg_m=_float(row[2], f"mass of {name}", path),
        ea_N=ea,
        ei_Nm2=ei,
        gj_Nm2=gj,
        bending_damping_Nm2s=bb,
        mass_damping_1_s=alpha,
        torque_tension_coupling_Nm=coupling,
        torsional_damping=torsional_damping,
        polar_inertia_kg_m=polar_inertia,
        ei_token=row[5],
        gj_token=gj_token,
    )


def _deck_blocks(
    path: Path,
) -> tuple[dict[str, list[list[str]]], dict[str, list[str]]]:
    """Split a deck into recognised sections of token rows (comments removed).

    As CableDyn reads a deck: the comment is cut first (:func:`_strip_comment`),
    a record holding ``---`` is a section header named by its other
    characters (every ``-`` dropped), and a section met again continues with
    its rows. The records of every section are also returned as text.
    """
    blocks: dict[str, list[list[str]]] = {}
    texts: dict[str, list[str]] = {}
    current: str | None = None
    text_all = path.read_text(encoding="utf-8", errors="strict").removeprefix("\ufeff")
    for raw in text_all.split("\n"):
        line = _strip_comment(raw)
        if "---" in line:
            name = " ".join(line.replace("-", "").split()).casefold()
            if not name or name == "end" or "need" in name or "input file" in name:
                current = None
                continue
            current = _SECTION_NAMES.get(name, f"other:{name}")
            blocks.setdefault(current, [])
            texts.setdefault(current, [])
            continue
        if current is None or not line:
            continue
        # A quoted token counts once, separators and all, as in CableDyn;
        # stray quotes elsewhere are dropped.
        blocks[current].append([_unquote(token) for token in _TOKEN.findall(line)])
        texts[current].append(line)
    return blocks, texts


#: MoorDyn spellings of the RODS ``Type`` column (CableDyn ``append_rod``).
_ROD_TYPE_ALIASES = {
    "anchor": "fixed",
    "fix": "fixed",
    "pin": "pinned",
    "point": "free",
    "con": "free",
    "ves": "vessel",
    "cpld": "coupled",
    "cpldpin": "coupledpinned",
    "vesselpinned": "coupledpinned",
    "vespin": "coupledpinned",
}
#: Host-driven or free BODIES types CableDyn makes Rigid6 bodies.
_RIGID6_BODY_TYPES = frozenset(
    {"coupled", "vessel", "cpld", "ves", "coupledpinned", "cpldpin"}
    | {"vesselpinned", "vespin"}
)
_ROD_END = re.compile(r"rod(\d+)([ab])")
_BODY_ID = re.compile(r"body(\d+)")
_LOGICAL = {
    **dict.fromkeys(("true", "t", "yes", "y", "on", "1"), True),
    **dict.fromkeys(("false", "f", "no", "n", "off", "0"), False),
}


def _rods(rows: list[list[str]], path: Path) -> dict[int, tuple[str, int]]:
    """RODS ``ID -> (Type, NumSegs)`` with CableDyn's spelling of the type."""
    rods: dict[int, tuple[str, int]] = {}
    for row in _data_rows(rows):
        if len(row) not in (10, 11):
            raise CableSchemaError(
                f"{path.name}: RODS row has {len(row)} columns; CableDyn reads 10 "
                "plus optional Outputs (ID RodType Type XA YA ZA XB YB ZB NumSegs "
                "[Outputs])"
            )
        kind = row[2].casefold()
        kind = _ROD_TYPE_ALIASES.get(kind, kind)
        if (
            kind.startswith("body")
            and len(kind) > 7
            and kind.endswith("pin")
            and not kind.endswith("pinned")
        ):
            kind += "ned"
        rods[_int(row[0], "RODS ID", path)] = (kind, _int(row[9], "NumSegs", path))
    return rods


def _bodies(rows: list[list[str]], path: Path) -> dict[int, str]:
    """BODIES ``ID -> type``, ``rigid6`` for the types CableDyn makes Rigid6."""
    bodies: dict[int, str] = {}
    for row in _data_rows(rows):
        if len(row) not in (14, 15, 18):
            raise CableSchemaError(
                f"{path.name}: BODIES row has {len(row)} columns; CableDyn reads 14 "
                "(MoorDyn), 15 or 18"
            )
        kind = row[1].casefold()
        # The 14-column MoorDyn row also makes a Free body Rigid6.
        rigid6 = _RIGID6_BODY_TYPES | ({"free"} if len(row) == 14 else set())
        bodies[_int(row[0], "BODIES ID", path)] = "rigid6" if kind in rigid6 else kind
    return bodies


def _end_kind(
    token: str,
    point_types: Mapping[str, str],
    rods: Mapping[int, tuple[str, int]],
    bodies: Mapping[int, str],
) -> str:
    """The point type CableDyn gives a line end: a POINTS type, or a rod end
    (``R<N>A``, ``Rod<N>B`` or a ``Rod<N>A`` point) as ``rod<N><a|b>``; a
    zero-length rod is the point it collapses to, and an end of a rod fixed
    to a body a point of that body (CableDyn ``resolve_rod_end_tokens``)."""
    kind = point_types.get(token, "")
    if not kind:
        word = token.casefold()
        if word.startswith("r") and not word.startswith("rod") and len(word) >= 3:
            word = "rod" + word[1:]
        kind = word if _ROD_END.fullmatch(word) else ""
    match = _ROD_END.fullmatch(kind)
    if match is None or int(match.group(1)) not in rods:
        return kind
    rod_type, n_segments = rods[int(match.group(1))]
    body = _BODY_ID.match(rod_type)
    if body is not None and (n_segments == 0 or int(body.group(1)) in bodies):
        return f"body{int(body.group(1))}"
    if n_segments == 0:
        # CableDyn collapse_zero_length_rods (a CoupledPinned one only on a
        # coupled host).
        return {
            "free": "free",
            "fixed": "fixed",
            "pinned": "fixed",
            "coupled": "coupled",
            "vessel": "coupled",
            "coupledpinned": "coupled",
        }.get(rod_type, kind)
    return kind


def _one_end_torsion_note(
    line: CableDynLine,
    line_types: Mapping[str, CableDynLineType],
    kinds: tuple[str, str],
    rods: Mapping[int, tuple[str, int]],
    bodies: Mapping[int, str],
    *,
    marching: bool,
    force_blend: bool,
) -> str:
    """Why CableDyn does not solve the torsion of a line restrained at one end
    only, or ``""`` when it does (CableDyn ``resolve_one_end_torsion``)."""
    types = [line_types[sec.line_type.casefold()] for sec in line.sections]
    kind_a, kind_b = kinds
    if not all(t.gj_Nm2 is not None and t.gj_Nm2 > 0.0 for t in types):
        return "its LINE TYPES rows give no GJ (the 14-column row)"
    if kind_b != "fixed":
        return "its End B is not a Fixed point (torsion needs a Fixed End B)"
    if marching and not force_blend:
        return (
            "torsion in a dynamic run needs the force-blended generalised-alpha "
            "(alpha_force_blend True)"
        )
    if "a" in line.torsion_restrained_ends:
        rod = _ROD_END.fullmatch(kind_a)
        body = _BODY_ID.fullmatch(kind_a)
        if rod is not None and int(rod.group(1)) in rods:
            rod_type = rods[int(rod.group(1))][0]
            if rod_type in ("coupled", "vessel"):
                return "its restrained End A is on a Coupled/Vessel rod end"
            if rod_type != "fixed":
                return "its restrained End A is on the end of a free or pinned rod"
        elif body is not None and int(body.group(1)) in bodies:
            if bodies[int(body.group(1))] != "rigid6":
                return "its restrained End A is on a body that is not Rigid6"
    return ""


def read_cabledyn_deck(path: str | Path) -> CableDynDeck:
    """Read line types, lines, sections, end connections, options and outputs.

    The tables are read by position, as CableDyn reads them: a row without a
    number token is a column-name or units row and is skipped, and the token
    count selects the row form (LINE TYPES 10, 11, 13, 14, 15 or 17 columns;
    POINTS 5 or 9; LINES 3, 4 or the stock 7; SECTIONS 4; END CONNECTIONS 6,
    10 or 11). The ``EI`` column may carry bending damping (``EI|Bb``,
    ``EI|-zeta|f``, ``EI|-zeta|f1|f2``) and the ``GJ`` column of a 14-column
    row torque-tension coupling and torsional damping (``GJ|K``,
    ``GJ|K|Bt``); :class:`CableDynLineType` keeps them. ``EQUIVALENT
    BUOYANCY`` rows replace a type's diameter and mass, as CableDyn does.
    Records are cut and split as CableDyn's reader does (:func:`_deck_blocks`).

    Applies CableDyn's anchor-first swap (a line whose ``NodeA`` is ``Fixed`` and
    whose ``NodeB`` moves is reversed, with its sections and end connections),
    so node 1 is End A as in every output file. The BODIES and RODS rows are
    read for the types of the line ends only (the swap and
    :attr:`CableDynLine.torsion_solved`).
    """
    deck_path = Path(path)
    if not deck_path.is_file():
        raise FileNotFoundError(f"CableDyn deck not found: {deck_path}")
    blocks, texts = _deck_blocks(deck_path)
    for required in ("line types", "lines"):
        if required not in blocks:
            raise CableSchemaError(f"{deck_path.name}: no {required.upper()} section")

    line_types: dict[str, CableDynLineType] = {}
    for row in _data_rows(blocks["line types"]):
        line_type = _line_type(row, deck_path)
        key = line_type.name.casefold()
        if key in line_types:
            raise CableSchemaError(
                f"{deck_path.name}: LINE TYPES defines {line_type.name!r} twice"
            )
        line_types[key] = line_type
    if not line_types:
        raise CableSchemaError(f"{deck_path.name}: LINE TYPES has no data rows")

    point_types: dict[str, str] = {}
    point_z: dict[str, float] = {}
    point_rows = _data_rows(blocks.get("points", []))
    if "points" in blocks and not point_rows:
        raise CableSchemaError(f"{deck_path.name}: POINTS has no data rows")
    for row in point_rows:
        if len(row) not in (5, 9):
            raise CableSchemaError(
                f"{deck_path.name}: POINTS row {row[0]!r} has {len(row)} columns; "
                "CableDyn reads 5 (ID Type X Y Z) or 9 (ID Type X Y Z Mass Vol CdA Ca)"
            )
        point_types[row[0]] = row[1].casefold()
        point_z[row[0]] = _float(row[4], f"Z of point {row[0]}", deck_path)
    rods = _rods(blocks.get("rods", []), deck_path)
    bodies = _bodies(blocks.get("bodies", []), deck_path)

    sections_by_line: dict[int, list[CableDynSection]] = {}
    for row in _data_rows(blocks.get("sections", [])):
        if len(row) != 4:
            raise CableSchemaError(
                f"{deck_path.name}: SECTIONS row has {len(row)} columns; CableDyn "
                "reads 4 (LineID LineType Length NumSegs)"
            )
        sections_by_line.setdefault(
            _int(row[0], "SECTIONS LineID", deck_path), []
        ).append(
            CableDynSection(
                line_type=row[1],
                length_m=_float(row[2], "section Length", deck_path),
                n_segments=_int(row[3], "section NumSegs", deck_path),
            )
        )
    torsion_ends: dict[int, set[str]] = {}
    for row in _data_rows(blocks.get("end connections", [])):
        if len(row) not in (6, 10, 11):
            raise CableSchemaError(
                f"{deck_path.name}: END CONNECTIONS row has {len(row)} columns; "
                "CableDyn reads 6 (LineID End Stiffness EzX EzY EzZ), or 10 or 11 "
                "with the torsion columns (... TorsStiffness NxX NxY NxZ [Pretwist])"
            )
        line_id = _int(row[0], "END CONNECTIONS LineID", deck_path)
        end = _end_token(row[1], deck_path, line_id)
        if len(row) >= 10 and _torsion_restrained(row[6], deck_path, line_id):
            torsion_ends.setdefault(line_id, set()).add(end)
    lines: dict[int, CableDynLine] = {}
    end_kinds: dict[int, tuple[str, str, bool]] = {}
    for row in _data_rows(blocks["lines"]):
        line_id = _int(row[0], "LINES ID", deck_path)
        if line_id in lines:
            raise CableSchemaError(
                f"{deck_path.name}: LINES lists line {line_id} twice"
            )
        if len(row) == 7:
            # Stock form: ID LineType AttachA AttachB UnstrLen NumSegs Outputs.
            end_a, end_b, outputs = row[2], row[3], row[6]
            sections = [
                CableDynSection(
                    line_type=row[1],
                    length_m=_float(row[4], "UnstrLen", deck_path),
                    n_segments=_int(row[5], "NumSegs", deck_path),
                )
            ]
            if line_id in sections_by_line:
                raise CableSchemaError(
                    f"{deck_path.name}: line {line_id} has a 7-column row and SECTIONS"
                )
        elif len(row) in (3, 4):
            end_a, end_b = row[1], row[2]
            outputs = row[3] if len(row) == 4 else "-"
            sections = sections_by_line.get(line_id, [])
        else:
            raise CableSchemaError(
                f"{deck_path.name}: LINES row of line {line_id} has {len(row)} "
                "columns; CableDyn reads 3 (ID NodeA NodeB), 4 (+Outputs) or the "
                "stock 7 (ID LineType AttachA AttachB UnstrLen NumSegs Outputs)"
            )
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
        finite_ei = any(
            line_types[sec.line_type.casefold()].ei_Nm2 > 0.0 for sec in sections
        )
        type_a = _end_kind(end_a, point_types, rods, bodies)
        type_b = _end_kind(end_b, point_types, rods, bodies)
        # CableDyn normalize_stock_endpoint_order: a Free/Connect NodeB swaps
        # only when it is not below the Fixed NodeA (a weight hung from an
        # overhead point stays as written).
        swap = type_a == "fixed" and (
            type_b in ("coupled", "vessel")
            or type_b.startswith("body")
            or type_b.startswith("turbine")
            or (finite_ei and _ROD_END.fullmatch(type_b) is not None)
            or (
                type_b in ("free", "connect")
                and point_z.get(end_a, math.inf) <= point_z.get(end_b, -math.inf)
            )
        )
        ends = torsion_ends.get(line_id, set())
        if swap:
            end_a, end_b = end_b, end_a
            sections = sections[::-1]
            ends = {"b" if end == "a" else "a" for end in ends}
            type_a, type_b = type_b, type_a
        end_kinds[line_id] = (type_a, type_b, finite_ei)
        lines[line_id] = CableDynLine(
            line_id=line_id,
            end_a=end_a,
            end_b=end_b,
            outputs=outputs,
            sections=tuple(sections),
            torsion_restrained=ends >= {"a", "b"},
            reversed_by_parser=swap,
            torsion_restrained_ends=frozenset(ends),
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
    # As CableDyn's append_channel: names separated by spaces, tabs or
    # commas, quotes dropped; a row whose first token (quotes kept) is END
    # closes the channel list (stock decks).
    output_rows = texts.get("outputs", [])
    stop = next(
        (
            i
            for i, row in enumerate(output_rows)
            if _TOKEN.findall(row)[0].casefold() == "end"
        ),
        len(output_rows),
    )
    channels = tuple(
        name
        for row in output_rows[:stop]
        for name in re.sub("[\"',]", " ", row).split()
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
    t_max = option("TMax", None)
    blend = (last("alpha_force_blend") or "true").casefold()
    if blend not in _LOGICAL:
        raise CableSchemaError(f"{deck_path.name}: alpha_force_blend {blend!r}")

    # EQUIVALENT BUOYANCY replaces a type's diameter and mass, as CableDyn
    # does before it runs (apply_equivalent_buoyancy).
    for row in _data_rows(blocks.get("equivalent buoyancy", [])):
        if len(row) != 3:
            raise CableSchemaError(
                f"{deck_path.name}: EQUIVALENT BUOYANCY row needs 3 columns: "
                "LineType Diam SubmergedWeightNpm"
            )
        key = row[0].casefold()
        if key not in line_types:
            raise CableSchemaError(
                f"{deck_path.name}: EQUIVALENT BUOYANCY references unknown "
                f"LineType {row[0]!r}"
            )
        if line_types[key].equivalent_submerged_weight_N_m is not None:
            raise CableSchemaError(
                f"{deck_path.name}: duplicate EQUIVALENT BUOYANCY row for "
                f"LineType {row[0]!r}"
            )
        diameter = _float(row[1], f"EQUIVALENT BUOYANCY Diam of {row[0]}", deck_path)
        weight = _float(
            row[2], f"EQUIVALENT BUOYANCY submerged weight of {row[0]}", deck_path
        )
        if not diameter > 0.0:
            raise CableSchemaError(
                f"{deck_path.name}: EQUIVALENT BUOYANCY needs Diam > 0: {row[1]!r}"
            )
        if not (g > 0.0 and rho > 0.0):
            raise CableSchemaError(
                f"{deck_path.name}: EQUIVALENT BUOYANCY needs g > 0 and rhoW > 0"
            )
        mass = rho * (math.pi / 4.0) * diameter**2 + weight / g
        if mass < 0.0:
            raise CableSchemaError(
                f"{deck_path.name}: EQUIVALENT BUOYANCY for LineType {row[0]!r}: "
                "the equivalent section would require a negative dry mass per length"
            )
        line_types[key] = replace(
            line_types[key],
            diameter_m=diameter,
            mass_kg_m=mass,
            equivalent_submerged_weight_N_m=weight,
        )

    # Which lines CableDyn solves in torsion (resolve_one_end_torsion). A
    # deck without TMax is taken as a coupled (OpenFAST) deck, which marches.
    marching = t_max is None or t_max > 0.0
    for line_id, line in list(lines.items()):
        kind_a, kind_b, finite_ei = end_kinds[line_id]
        if not line.torsion_restrained_ends or not finite_ei:
            continue
        note = (
            ""
            if line.torsion_restrained
            else _one_end_torsion_note(
                line,
                line_types,
                (kind_a, kind_b),
                rods,
                bodies,
                marching=marching,
                force_blend=_LOGICAL[blend],
            )
        )
        lines[line_id] = replace(line, torsion_solved=not note, torsion_note=note)
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
        t_max_s=t_max,
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


@dataclass(frozen=True)
class _LineHead:
    """Title, optional ``# DeckNode`` line and header row of a per-line table."""

    path: Path
    header: tuple[str, ...]
    deck_nodes: tuple[int, ...] | None
    skip: int


def _line_head(path: Path, title_word: str) -> _LineHead:
    head = _head(path, 3)
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
    deck_nodes = None
    header_row, skip = head[1], 2
    if head[1].startswith("#"):
        # A line whose mesh the static solve refined: the deck node of every
        # node, 0 for a node the refinement added (CableDyn doc/outputs.rst).
        tokens = head[1].split()
        if tokens[:2] != ["#", "DeckNode"] or len(head) < 3:
            raise CableSchemaError(
                f"{path.name}: unexpected second comment line {head[1][:40]!r}; "
                "a CableDyn per-line table has only a '# DeckNode' line there"
            )
        try:
            deck_nodes = tuple(int(token) for token in tokens[2:])
        except ValueError:
            raise CableSchemaError(
                f"{path.name}: malformed '# DeckNode' line"
            ) from None
        header_row, skip = head[2], 3
    header = tuple(cell.strip() for cell in header_row.split("\t"))
    return _LineHead(path, header, deck_nodes, skip)


def _line_table(head: _LineHead, columns: Sequence[str]) -> _Table:
    path, header = head.path, head.header
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
    return _Table(path, head.skip, header, None)


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
    arc: float  # on the grid of the curvature source
    elem_deck: int  # 0-based deck element containing the arc (node channels)
    elem_pos: int  # 0-based element of the position grid containing the arc
    window: list[int]  # 0-based nodes of the curvature spline (its grid)
    probe: bool = False
    node: int = -1  # 0-based deck node of a curvature probe
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


def _static_profile(
    path: Path, line_id: int
) -> tuple[FloatArray, tuple[int, ...] | None]:
    """X, Y, Z of the line's nodes from a static profile table, and the
    ``DeckNode`` column when the profile has one (a refined mesh in the run)."""
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
    names = ["LineID", "Node", "X", "Y", "Z"]
    if "DeckNode" in header:
        if units[header.index("DeckNode")] != "(-)":
            raise CableSchemaError(f"{path.name}: static column DeckNode (-) missing")
        names.append("DeckNode")
    take = [header.index(name) for name in names]
    data = np.array(
        [[float(row.split("\t")[i]) for i in take] for row in rows[3:] if row.strip()]
    )
    mine = data[data[:, 0] == line_id]
    if mine.size == 0 or np.any(np.diff(mine[:, 1]) != 1) or mine[0, 1] != 1:
        raise CableSchemaError(f"{path.name}: line {line_id} rows missing or unordered")
    deck_nodes = None
    if "DeckNode" in names:
        column = mine[:, 5]
        if np.any(column != np.round(column)):
            raise CableSchemaError(f"{path.name}: DeckNode values must be integers")
        deck_nodes = tuple(int(value) for value in column)
    return np.asarray(mine[:, 2:5]), deck_nodes


@dataclass(frozen=True)
class _Mesh:
    """Nodes of the per-line files (the line's final mesh) and its deck nodes.

    ``deck_rows[j]`` is the 0-based mesh node of deck node ``j + 1``, the node
    the ``<J>`` of the node channels names (CableDyn 0.2.0 and later).
    ``arcs`` is ``None`` when a refined mesh has no ``.elements.out`` to place
    the nodes the refinement added.
    """

    n_nodes: int
    arcs: FloatArray | None
    deck_rows: tuple[int, ...]
    refined: bool
    map_source: str


def _deck_rows(
    deck_nodes: Sequence[int], n_deck: int, where: str, line_id: int
) -> tuple[int, ...]:
    """Mesh rows of the deck nodes from a ``DeckNode`` numbering."""
    numbered = [(row, value) for row, value in enumerate(deck_nodes) if value != 0]
    if (
        any(value < 0 for value in deck_nodes)
        or [value for _, value in numbered] != list(range(1, n_deck + 1))
        or not deck_nodes
        or deck_nodes[0] != 1
        or deck_nodes[-1] != n_deck
    ):
        named = max((value for _, value in numbered), default=0)
        raise CableSchemaError(
            f"{where}: the DeckNode numbering of line {line_id} does not name the "
            f"{n_deck} deck nodes of the deck sections in order, End A first and "
            f"End B last (largest deck node named: {named}): the deck and the "
            "output differ"
        )
    return tuple(row for row, _ in numbered)


def _line_mesh(
    line: CableDynLine,
    deck_arcs: FloatArray,
    finite_ei: bool,
    element_arcs: tuple[Path, FloatArray] | None,
    static: tuple[Path, int, tuple[int, ...] | None] | None,
    heads: Sequence[tuple[_LineHead, int]],
) -> _Mesh:
    """The mesh of the per-line files and the mesh row of every deck node.

    A static solve that refines a finite-EI line (``adaptive_mesh``,
    ``tensile_safety`` or the automatic refinement) splits deck elements, so
    every deck node stays a mesh node at its own unstretched arc length. The
    node channels name deck nodes; ``.static.out``, ``.elements.out`` and the
    per-line files list every mesh node. The deck nodes are located by the
    ``DeckNode`` column of ``.static.out`` or the ``# DeckNode`` line of
    ``.Line<L>.p.out`` and, failing both, by their arc lengths in
    ``.elements.out``; every source present must agree.
    """
    line_id = line.line_id
    n_deck = int(deck_arcs.size)
    tol = 1e-6 * max(1.0, line.length_m)
    counts: dict[str, int] = {}
    maps: dict[str, tuple[int, ...]] = {}
    if element_arcs is not None:
        counts[element_arcs[0].name] = int(element_arcs[1].size)
    if static is not None:
        counts[static[0].name] = static[1]
        if static[2] is not None:
            maps[static[0].name] = static[2]
    for head, n_nodes in heads:
        counts[head.path.name] = n_nodes
        if head.deck_nodes is not None:
            maps[head.path.name] = head.deck_nodes
    if len(set(counts.values())) > 1:
        raise CableSchemaError(
            f"the output files list different numbers of nodes of line {line_id} "
            f"({counts}): files from different runs?"
        )
    n_mesh = next(iter(counts.values()), n_deck)
    for name, numbering in maps.items():
        if len(numbering) != n_mesh:
            raise CableSchemaError(
                f"{name}: the DeckNode numbering of line {line_id} has "
                f"{len(numbering)} entries for {n_mesh} nodes"
            )
    if len(set(maps.values())) > 1:
        raise CableSchemaError(
            f"the DeckNode numberings of line {line_id} differ between "
            f"{sorted(maps)}: files from different runs?"
        )
    if maps:
        source, numbering = next(iter(maps.items()))
        deck_rows = _deck_rows(numbering, n_deck, source, line_id)
        map_source = f"DeckNode of {', '.join(sorted(maps))}"
    elif n_mesh == n_deck:
        deck_rows = tuple(range(n_deck))
        map_source = "deck mesh (not refined)"
    elif element_arcs is not None:
        # No numbering: each deck node is the mesh node at its arc length.
        mesh = element_arcs[1]
        rows = np.searchsorted(mesh, deck_arcs - tol)
        found = (rows < mesh.size) & (
            np.abs(mesh[np.minimum(rows, mesh.size - 1)] - deck_arcs) <= tol
        )
        if not found.all():
            raise CableSchemaError(
                f"{element_arcs[0].name}: mesh of line {line_id} differs from the "
                f"deck sections ({mesh.size} nodes, deck node at "
                f"{float(deck_arcs[int(np.argmin(found))]):.6g} m not among them)"
            )
        deck_rows = tuple(int(row) for row in rows)
        map_source = f"arc lengths of {element_arcs[0].name}"
    else:
        raise CableSchemaError(
            f"the output files list {n_mesh} nodes of line {line_id}, the deck "
            f"sections {n_deck}, and no DeckNode numbering locates the deck nodes "
            f"among them ({sorted(counts)}): the deck and the output differ"
        )
    refined = n_mesh != n_deck
    if refined and not finite_ei:
        raise CableSchemaError(
            f"line {line_id} has no finite-EI section, so its mesh is never refined, "
            f"but the output lists {n_mesh} nodes against {n_deck} deck nodes: the "
            "deck and the output differ"
        )
    arcs: FloatArray | None = None
    if element_arcs is not None:
        mesh = element_arcs[1]
        if not np.allclose(mesh[list(deck_rows)], deck_arcs, rtol=0.0, atol=tol):
            worst = int(np.argmax(np.abs(mesh[list(deck_rows)] - deck_arcs)))
            raise CableSchemaError(
                f"{element_arcs[0].name}: mesh of line {line_id} differs from the "
                f"deck sections (deck node {worst + 1} at "
                f"{float(mesh[deck_rows[worst]]):.6g} m against "
                f"{float(deck_arcs[worst]):.6g} m in the deck)"
            )
        arcs = mesh
    elif not refined:
        arcs = deck_arcs
    return _Mesh(n_mesh, arcs, deck_rows, refined, map_source)


def _element_arcs(path: Path, line_id: int, *, required: bool) -> FloatArray | None:
    """Reference arc lengths of the line's element ends, End A first; ``None``
    when the file has no rows of an ``EI = 0`` line (``required=False``)."""
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
    mine = data[data[:, 0] == line_id] if data.size else data.reshape(0, 4)
    if mine.size == 0 and not required:
        return None
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

    deck_arcs = line.node_arc_lengths()
    n_deck = int(deck_arcs.size)
    finite_ei = any(
        deck_obj.line_type(sec.line_type).ei_Nm2 > 0.0 for sec in line.sections
    )
    for name, arc in sections.items():
        if (
            not name.strip()
            or not math.isfinite(arc)
            or not 0.0 <= arc <= deck_arcs[-1]
        ):
            raise CableSchemaError(
                f"section {name!r} at {arc} m is outside line {line_id} "
                f"(0 to {deck_arcs[-1]:.6g} m)"
            )

    # The node channels name deck nodes; .static.out, .elements.out and the
    # per-line files list every node of the line's final mesh, which the static
    # solve may have refined.
    heads: list[tuple[_LineHead, int]] = []
    positions_head: _LineHead | None = None
    tensions_head: _LineHead | None = None
    if "positions" in paths:
        positions_head = _line_head(paths["positions"], "node positions")
        n_columns = len(positions_head.header) - 1
        if n_columns < 6 or n_columns % 3:
            raise CableSchemaError(
                f"{paths['positions'].name}: expected Time(s) and X, Y, Z columns "
                "of at least two nodes"
            )
        heads.append((positions_head, n_columns // 3))
    if "tensions" in paths:
        tensions_head = _line_head(paths["tensions"], "segment tensions")
        if len(tensions_head.header) < 2:
            raise CableSchemaError(f"{paths['tensions'].name}: no segment columns")
        heads.append((tensions_head, len(tensions_head.header)))
    static_profile = (
        _static_profile(paths["static"], line_id) if "static" in paths else None
    )
    # .elements.out may list the finite-EI lines of the run only.
    element_arcs = None
    if "elements" in paths:
        found = _element_arcs(paths["elements"], line_id, required=finite_ei)
        if found is not None:
            element_arcs = (paths["elements"], found)
    mesh = _line_mesh(
        line,
        deck_arcs,
        finite_ei,
        element_arcs,
        None
        if static_profile is None
        else (paths["static"], static_profile[0].shape[0], static_profile[1]),
        heads,
    )
    mesh_arcs = mesh.arcs
    n_mesh = mesh.n_nodes
    checks["mesh"] = {
        "deck_nodes": n_deck,
        "output_nodes": n_mesh,
        "refined": mesh.refined,
        "deck_nodes_located_by": mesh.map_source,
    }
    if element_arcs is not None:
        checks["arc_lengths"] = (
            "refined mesh of .elements.out; deck nodes at the deck-section arcs"
            if mesh.refined
            else "deck sections, confirmed by .elements.out"
        )
    elif mesh.refined:
        checks["arc_lengths"] = (
            "deck sections at the deck nodes; the nodes the refinement added are "
            "not placed (no .elements.out)"
        )
    else:
        checks["arc_lengths"] = "deck sections"

    main = _main_table(paths["main"])
    channels = _channel_index(main)
    positions_table = None
    if positions_head is not None:
        columns = [f"Node{j}{axis}(m)" for j in range(1, n_mesh + 1) for axis in "XYZ"]
        positions_table = _line_table(positions_head, columns)
    tensions_table = None
    if tensions_head is not None:
        columns = [f"Segment{k}Tension(N)" for k in range(1, n_mesh)]
        tensions_table = _line_table(tensions_head, columns)
        if mesh_arcs is None:
            # The segments of a refined mesh cannot be placed along the line.
            if tension_source == "segments":
                raise CableSchemaError(
                    f"tension_source='segments': the segments of "
                    f"{paths['tensions'].name} lie on the refined mesh of line "
                    f"{line_id}, which {root.name}.elements.out places; it is missing"
                )
            tensions_table = None
            checks["tension_file_skipped"] = (
                f"{paths['tensions'].name}: refined mesh without .elements.out"
            )

    # Grid of the node positions: every mesh node of .Line<L>.p.out when the
    # mesh is placed, the deck nodes otherwise (channels, or the deck-node
    # columns of .p.out). pos_rows: mesh row of each grid node (.p.out column
    # and .static.out row).
    if positions_table is not None and mesh_arcs is not None:
        pos_arcs = mesh_arcs
        pos_rows = list(range(n_mesh))
        pos_on_mesh = True
    else:
        pos_arcs = deck_arcs
        pos_rows = list(mesh.deck_rows)
        pos_on_mesh = False
    n_pos = int(pos_arcs.size)
    pos_l0 = np.diff(pos_arcs)

    def has_tangent(node: int) -> bool:
        return all((kind, line_id, node + 1) in channels for kind in ("dec", "azi"))

    def has_position(node: int) -> bool:
        return positions_table is not None or all(
            (kind, line_id, node + 1) in channels for kind in ("px", "py", "pz")
        )

    # Sections and curvature probes: (name, deck arc, grid arc, deck element,
    # grid element, deck node of a probe or -1).
    specs: list[tuple[str, float, float, int, int, int]] = [
        (
            name,
            float(arc),
            float(arc),
            _element_of(deck_arcs, arc),
            _element_of(pos_arcs, arc),
            -1,
        )
        for name, arc in sections.items()
    ]
    for key in sorted(k for k in channels if k[0] == "curv" and k[1] == line_id):
        node = key[2] - 1
        if 0 <= node < n_deck:
            row = mesh.deck_rows[node] if pos_on_mesh else node
            specs.append(
                (
                    f"Curv{line_id}N{key[2]}",
                    float(deck_arcs[node]),
                    float(pos_arcs[row]),
                    min(node, n_deck - 2),
                    min(row, n_pos - 2),
                    node,
                )
            )
    section_specs = [spec for spec in specs if spec[5] < 0]
    if curvature_source == "auto":
        tangent_ok = all(
            has_tangent(j)
            for spec in section_specs
            for j in _window(spec[3], n_deck, window_nodes)
        )
        source = "tangent" if tangent_ok else "positions"
    else:
        source = curvature_source
    needed_curvature = has_tangent if source == "tangent" else has_position
    plans: list[_Plan] = []
    for name, arc_deck, arc_pos, elem_deck, elem_pos, node in specs:
        window = (
            _window(elem_deck, n_deck, window_nodes)
            if source == "tangent"
            else _window(elem_pos, n_pos, window_nodes)
        )
        missing = [j + 1 for j in window if not needed_curvature(j)]
        if missing and node < 0:
            what = (
                f"L{line_id}N<J>Dec and L{line_id}N<J>Azi channels"
                if source == "tangent"
                else f".Line{line_id}.p.out or L{line_id}N<J>p{{x,y,z}} channels"
            )
            raise CableSchemaError(
                f"section {name!r}: curvature from {source} needs {what} "
                f"at nodes {missing}"
            )
        if missing:
            continue  # a probe without its window is not checked
        plans.append(
            _Plan(
                name,
                arc_deck if source == "tangent" else arc_pos,
                elem_deck,
                elem_pos,
                window,
                probe=node >= 0,
                node=node,
            )
        )
    sections_plans = [plan for plan in plans if not plan.probe]
    checks["curvature_source"] = source
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

    # Position-grid nodes needed: frame transport (arc), curvature, elevation.
    pos_nodes: set[int] = set()
    if frame == "arc":
        last = max(plan.elem_pos for plan in sections_plans) + 1
        pos_nodes.update(range(0, last + 1))
    if source == "positions":
        for plan in plans:
            pos_nodes.update(plan.window)
    if pressure == "hydrostatic":
        for plan in sections_plans:
            pos_nodes.update((plan.elem_pos, plan.elem_pos + 1))
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
            for node in (plan.elem_deck, plan.elem_deck + 1):
                if ("ten", line_id, node + 1) not in channels:
                    raise CableSchemaError(
                        f"section {plan.name!r}: tension needs .Line{line_id}.t.out "
                        f"or Ten{line_id}N{node + 1} in the main output"
                    )
    torsion_nodes = [(plan.elem_deck, plan.elem_deck + 1) for plan in sections_plans]
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
    coupled = sorted(
        {
            sec.line_type
            for sec in line.sections
            if deck_obj.line_type(sec.line_type).torque_tension_coupling_Nm != 0.0
        }
    )
    restrained = " and ".join(sorted(line.torsion_restrained_ends)).upper()
    if coupled and line.torsion_solved and not twist_available:
        # Solved in torsion, a coupled line twists as it stretches
        # (Twist = -sum (K/GJ)(l - L) with one end free): the twist-free
        # frame would miss it.
        raise CableSchemaError(
            f"line {line_id} couples torque and tension (GJ|K of {', '.join(coupled)}) "
            f"and is restrained in torsion at End {restrained}: its "
            "material twist follows the stretch, so the material frame needs "
            f"Twist{line_id}N<J> channels at the nodes bracketing every section"
        )
    marching = deck_obj.t_max_s is None or deck_obj.t_max_s > 0.0
    if line.torsion_solved and marching and not twist_available:
        # Restrained at one end, CableDyn solves the nodal twist in the
        # dynamics: the polar inertia Irn carries torsional waves, and a roll
        # or turn of the restrained end's vessel or body twists the line
        # (doc/theory.rst, a free end). Only its statics are twist-free.
        raise CableSchemaError(
            f"line {line_id} is restrained in torsion at End {restrained} and "
            "CableDyn solves its torsion: in the dynamics the line twists "
            "(torsional waves of its polar inertia, the roll of the restrained "
            "end), so the material frame needs "
            f"Twist{line_id}N<J> channels at the nodes bracketing every section"
        )
    checks["torsion"] = {
        "restrained_ends": sorted(line.torsion_restrained_ends),
        "solved": line.torsion_solved,
        "not_solved_because": line.torsion_note,
        "material_twist": "Twist channels" if twist_available else "none (twist-free)",
    }

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
            want("ten", plan.elem_deck)
            want("ten", plan.elem_deck + 1)
    if twist_available:
        for plan in sections_plans:
            want("twist", plan.elem_deck)
            want("twist", plan.elem_deck + 1)
    if torque_available:
        for plan in sections_plans:
            want("torq", plan.elem_deck)
            want("torq", plan.elem_deck + 1)
    probe_nodes = [plan.node for plan in plans if plan.probe]
    for node in probe_nodes:
        want("curv", node)
    # Interior deck nodes with a Ten channel, paired with their mesh node.
    tension_check_nodes: list[tuple[int, int]] = []
    if tensions_table is not None:
        tension_check_nodes = [
            (k[2] - 1, mesh.deck_rows[k[2] - 1])
            for k in channels
            if k[0] == "ten" and k[1] == line_id and 1 < k[2] < n_deck
        ]
        for node, _ in tension_check_nodes:
            want("ten", node)

    streams: list[Generator[FloatArray, None, None]] = [
        _chunks(main, main_cols, chunk_rows)
    ]
    if positions_table is not None and pos_order:
        pos_cols = [0] + [
            1 + 3 * pos_rows[node] + axis for node in pos_order for axis in range(3)
        ]
        streams.append(_chunks(positions_table, pos_cols, chunk_rows))
    elem_t: FloatArray = np.empty((0, 0))
    ten_slot: dict[int, int] = {}
    seg_elem: dict[str, int] = {}
    mesh_l0: FloatArray = np.empty(0)
    midpoints: FloatArray = np.empty(0)
    if tensions_table is not None:
        assert mesh_arcs is not None  # a table on an unplaced mesh was dropped
        mesh_l0 = np.diff(mesh_arcs)
        midpoints = 0.5 * (mesh_arcs[:-1] + mesh_arcs[1:])
        seg_elem = {
            plan.name: _element_of(mesh_arcs, plan.arc) for plan in sections_plans
        }
        mirror = [n_mesh - 1 - row for _, row in tension_check_nodes]
        elements = sorted(
            {
                e
                for element in seg_elem.values()
                for e in (element - 1, element, element + 1)
                if 0 <= e < n_mesh - 1
            }
            | {
                e
                for row in [row for _, row in tension_check_nodes] + mirror
                for e in (row - 1, row)
            }
        )
        ten_slot = {e: i for i, e in enumerate(elements)}
        streams.append(
            _chunks(tensions_table, [0] + [1 + e for e in elements], chunk_rows)
        )

    deck_l0 = np.diff(deck_arcs)
    # Arc length of each Dec/Azi value: the node's own tangent of a finite-EI
    # line (CableDyn 0.4.2+) at the node; a chord (central, one-sided at the
    # ends) at its mid-arc, the point of a smooth arc whose tangent it is. On a
    # refined line a deck node's chord joins its neighbours on the refined mesh.
    tangent_arcs = deck_arcs
    if source == "tangent" and not (tangent_placement == "auto" and finite_ei):
        if not mesh.refined:
            chord_arcs, chord_rows = deck_arcs, list(range(n_deck))
        elif mesh_arcs is not None:
            chord_arcs, chord_rows = mesh_arcs, list(mesh.deck_rows)
        else:
            raise CableSchemaError(
                f"tangent_placement='chord' on the refined line {line_id} places each "
                "chord on the refined mesh, which "
                f"{root.name}.elements.out locates; it is missing"
            )
        last_row = int(chord_arcs.size) - 1
        tangent_arcs = np.array(
            [
                0.5 * (chord_arcs[max(m - 1, 0)] + chord_arcs[min(m + 1, last_row)])
                for m in chord_rows
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
                if static_profile is not None and pos_order:
                    static = static_profile[0]
                    max_t0_error = float(
                        np.max(
                            np.abs(
                                static[[pos_rows[node] for node in pos_order]]
                                - positions[0]
                            )
                        )
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
                        stretch = chord / pos_l0[contiguous] - 1.0
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
                        / float(np.min(pos_l0)) ** 2
                    )
            # Element tension consistency (ordering check).
            if tensions_table is not None and tension_check_nodes:
                for node, row in tension_check_nodes:
                    channel = values[:, slot[("ten", node)]]
                    scale = np.maximum(np.abs(channel), 1.0)
                    for j, label in ((row, "match"), (n_mesh - 1 - row, "mirror")):
                        left, right = elem_t[:, ten_slot[j - 1]], elem_t[:, ten_slot[j]]
                        w_left, w_right = mesh_l0[j - 1], mesh_l0[j]
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
                last = max(plan.elem_pos for plan in sections_plans)
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
                    window_xyz = positions[:, [pos_slot[j] for j in plan.window]]
                    value, d1, d2 = _spline(pos_arcs[plan.window], window_xyz, plan.arc)
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
                        arc_e1[plan.elem_pos], chords[:, plan.elem_pos], tangent
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
                # Linear weights between the bracketing deck nodes (node
                # channels) and position-grid nodes (positions).
                weight = (plan.arc - deck_arcs[plan.elem_deck]) / deck_l0[
                    plan.elem_deck
                ]
                weight_pos = (plan.arc - pos_arcs[plan.elem_pos]) / pos_l0[
                    plan.elem_pos
                ]
                if twist_available:
                    twist = np.radians(
                        (1 - weight) * values[:, slot[("twist", plan.elem_deck)]]
                        + weight * values[:, slot[("twist", plan.elem_deck + 1)]]
                    )
                    cos, sin = np.cos(twist)[:, None], np.sin(twist)[:, None]
                    axis1, axis2 = cos * axis1 + sin * axis2, -sin * axis1 + cos * axis2
                    append(plan, "twist", twist)
                if torque_available:
                    append(
                        plan,
                        "torque",
                        (1 - weight) * values[:, slot[("torq", plan.elem_deck)]]
                        + weight * values[:, slot[("torq", plan.elem_deck + 1)]],
                    )
                append(plan, "kx", np.sum(kappa * axis1, axis=-1))
                append(plan, "ky", np.sum(kappa * axis2, axis=-1))
                if (
                    pos_order
                    and plan.elem_pos in pos_slot
                    and plan.elem_pos + 1 in pos_slot
                ):
                    if source == "positions":
                        z = value[:, 2]
                    else:
                        z = (1 - weight_pos) * positions[
                            :, pos_slot[plan.elem_pos], 2
                        ] + weight_pos * positions[:, pos_slot[plan.elem_pos + 1], 2]
                    append(plan, "z", z)
                if use_segments:
                    if plan.arc <= midpoints[0]:
                        ten = elem_t[:, ten_slot[0]]
                    elif plan.arc >= midpoints[-1]:
                        ten = elem_t[:, ten_slot[n_mesh - 2]]
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
                        :, slot[("ten", plan.elem_deck)]
                    ] + weight * values[:, slot[("ten", plan.elem_deck + 1)]]
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
        end = node <= 2 or node >= n_deck - 1
        allowed = atol + (
            curvature_check_end_rtol if end else curvature_check_rtol
        ) * float(np.max(np.abs(channel)))
        # At an end node CableDyn reports the end element's curvature, which
        # carries the end condition (clamped or pinned); the spline
        # extrapolation does not impose it. Reported, not enforced.
        enforced = 1 < node < n_deck
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
    ``Twist``/``Torq`` on a line CableDyn solves in torsion
    (:attr:`CableDynLine.torsion_solved`; CableDyn refuses them on any other
    line). With these
    channels alone use ``frame="section"``; the default ``frame="arc"`` also needs the
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
            if line.torsion_solved:
                add(f"Twist{line_id}N{node}")
                add(f"Torq{line_id}N{node}")
    return tuple(names)
