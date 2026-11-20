<!-- SPDX-License-Identifier: Apache-2.0 -->

# Global–local guide: CableDyn runs to armour fatigue

Stage 1 of the global–local coupling is one way: CableDyn solves the global
cable, CableXSec reads the histories of selected material sections and runs
its local fatigue on them, as it does on other global results. The global bending
law stays CableDyn's (linear elastic: the `EI` of the line type, tapered under a
bend stiffener); the hysteretic two-way element is stage 2 (Section 7).

| Piece | Where |
|---|---|
| Reader | `cablexsec.io.cabledyn`: `read_cabledyn_deck`, `read_cabledyn_sections`, `section_channels`, `CableDynSectionHistory`, `read_cabledyn_range_graph`, `read_cabledyn_lift` |
| Sea-state batch | `cablexsec.analysis.sea_state_batch`: `run_sea_state_batch`, `SeaStateRun`, `SectionRequest`, `SeaStateBatchOptions`, `load_batch_spec` |
| Command line | `cablexsec-batch batch.json --output-directory <new folder>` |
| Tests | `tests/test_cabledyn_reader.py`, `tests/test_cabledyn_outputs.py`, `tests/test_sea_state_batch.py`; genuine CableDyn output in `tests/data/cabledyn/` |

## 1. What CableDyn writes

From the CableDyn manual (`doc/driver_format.md`, `doc/outputs.rst`,
`doc/theory.rst`) for CableDyn 0.4.x. The genuine runs in `tests/data/cabledyn/`
were written by the CableDyn 0.1.0 driver (`lw24`, `lw128_v010`) and a
CableDyn 0.4.2 build (`lw128`; `lw24r` on a mesh the static solve
refined; `lw24g` with the 0.4.x LINE TYPES tokens) and a development build
of CableDyn after 0.4.2 (`lw24v`, the `lw24` deck with VIV lift in a current
and the flags `ptVr`; `lw24t.Line1.range.out` of a line restrained in
torsion); where a later release changed a channel, the item says so.

- **Files.** Standalone driver: `<root>.out` (title line, header
  `Time(s)` and the deck `OUTPUTS` tokens, no units row);
  `<root>.Line<L>.p.out` (node positions, `Node<J>X(m)` …, flag `p`);
  `<root>.Line<L>.t.out` (element tensions, `Segment<k>Tension(N)`, flag `t`);
  `<root>.static.out` and `<root>.elements.out` (initial state and element
  reference arcs, cubic-Hermite route). OpenFAST (`CompMooring = 5`):
  `<root>.CD.out` with a header `Time` and a units row, and
  `<root>.CD.static.out`; there are no per-line position or tension files.
  Fields are
  TAB-separated; values have 8 significant digits, time 17. The standalone
  output step is `dtM` and a run ends at `TMax` (a deck without `TMax`, or with
  `TMax 0`, runs static: one `t = 0` row); OpenFAST rounds `dtM` to whole glue
  steps and runs on the host's clock.
- **Range graphs and VIV lift.** The flag `r` writes
  `<root>.Line<L>.range.out` (OpenFAST `<root>.CD.Line<L>.range.out`): a title
  with the sample count and the window from `RangeStart`, a header `Node
  ArcLength` and `<Quantity>Min Max Mean` triples (tension, curvature, bend
  moment, declination, with a seabed the clearance, with torsion the torque and
  twist), a units row and a row per node of the final mesh. The flag `V`
  writes `<root>.Line<L>.V.out`, the VIV force `Node<J>LiftX/Y/Z(N)` at every
  node, with a `# DeckNode` line on a refined mesh. `read_cabledyn_range_graph`
  and `read_cabledyn_lift` read them, checking every column, unit and node;
  the range extremes equal those of the node channels exactly
  (`tests/test_cabledyn_outputs.py`). When either file lies next to a run,
  `read_cabledyn_sections` checks its mesh, and the range window against the
  output rows, and records its digest.
- **Order and frame.** Node 1 is End A (hang-off), node N End B; the parser
  swaps a line listed anchor first. Global axes are right-handed, `z` up,
  still water at `z = 0`, all SI, angles in degrees.
- **Curvature.** `Curv<L>N<J>` is the curvature magnitude at a node. On a
  finite-EI line it is recovered from nodal equilibrium (since CableDyn 0.2.0):
  the geometric curvature of the stretched line, `(1 + ε) BendMom / EI`, the
  larger of the two sides at a section junction (`OutputRecovery element`
  restores the larger one-sided element curvature). Since CableDyn 0.4.0 a
  finite-EI line also writes the signed components `CurvX<L>N<J>` and
  `CurvY<L>N<J>` on section axes (Section 2.1), with `CurvX² + CurvY² =
  Curv²`. The reader of Section 2 rebuilds the vector from the geometry and
  uses `Curv` as a check.
- **Tangent.** `L<L>N<J>Dec` and `L<L>N<J>Azi` give a direction from End A to
  End B. Up to CableDyn 0.4.1 it is the chord between the neighbouring nodes
  (one-sided at the ends). From CableDyn 0.4.2 a finite-EI line writes the
  node's own tangent, the cubic-Hermite tangent degree of freedom (at an end
  clamped by a `Rigid` row, the direction the clamp holds); an `EI = 0` line
  keeps the chord. The output files carry no version marker. The reader
  (Section 2) places a node tangent at its node and a chord at its mid-arc;
  `tangent_placement="chord"` reads finite-EI output of CableDyn 0.4.1 and
  earlier.
- **Tension.** Effective tension of a solid line without contents
  (`T_e = EA ε`). On a finite-EI line `Ten<L>N<J>` is the axial component of
  the section force recovered from nodal equilibrium (since CableDyn 0.2.0;
  `OutputRecovery element` restores the length-weighted mean of the two
  neighbouring element-mean forces), and `.t.out` holds the axial resultant at
  each segment midpoint; an end node reports the end force. On a curved coarse
  element the two can differ (Section 5).
- **Torsion.** On a line solved in torsion: restrained at both ends or, since
  CableDyn 0.4.0, at one end with the other free to twist. `Torq<L>N<J>`
  (uniform along the line in statics; with the nodal twist of the 0.4.0
  dynamics it varies along the line) and `Twist<L>N<J>`, the material twist
  accumulated from End A in degrees.
- **Pressure.** Not written.
- **Deck.** `read_cabledyn_deck` reads the tables by position, as CableDyn
  does: a row without a number token is a column-name or units row, and the
  token count selects the row form (LINE TYPES 10 or 14 columns, either
  followed by `Cl [dF cF]`; POINTS 5 or 9; LINES 3, 4 or the stock 7;
  SECTIONS 4; END CONNECTIONS 6, 10 or 11). The `EI` column may carry
  bending damping (`EI|Bb`, `EI|-zeta|f`, `EI|-zeta|f1|f2`), the `GJ` column
  of a 14-column row the torque–tension coupling `K` and torsional damping
  `Bt` (`GJ|K`, `GJ|K|Bt`); the line type keeps them, resolved as CableDyn
  resolves them (`bending_damping_Nm2s`, `mass_damping_1_s`,
  `torque_tension_coupling_Nm`, `torsional_damping`), and they leave the
  reconstructed histories unchanged, since these come from CableDyn's
  solution. `TorsStiffness` `Free`, `Zero` or `0` leaves an end free in
  torsion. `CableDynLine.end_connections` keeps every END CONNECTIONS row
  (bending `Pinned`/`Rigid`/stiffness, `Ez`, torsion, `Nx` orthonormalised
  against `Ez`, `Pretwist`), checked as CableDyn checks it (one row per line
  end, a finite-EI line, non-zero `Ez`, a usable `Nx` at a restrained end); the
  anchor-first swap negates `Ez` and `Pretwist`. `end_kinds` names what holds
  each end. `CableDynLine.torsion_solved` tells whether CableDyn solves a
  line's torsion: restrained at both ends, or at one end when every section
  gives `GJ`, End B is `Fixed`, a dynamic run is force-blended and a
  restrained End A on a rod or body is on a `Fixed` rod or a Rigid6 body
  (CableDyn's rules; otherwise `torsion_note` gives the reason and the line
  runs as without the torsion columns). A LINE TYPES row with a negative drag
  or added-mass coefficient, or a 14-column row whose `GAs`, `GJ`, `Irt` and
  `Irn` are neither all zero nor all positive, is refused, as by CableDyn.
- **Equivalent buoyancy.** An `EQUIVALENT BUOYANCY` row
  (`LineType Diam SubmergedWeight`) makes CableDyn run the type with the
  row's diameter `D` and the dry mass `rhoW π D²/4 + w/g` of its submerged
  weight `w` [N/m]. The line type's `diameter_m` and `mass_kg_m` are these
  values; `line_types_diameter_m` and `line_types_mass_kg_m` keep the LINE
  TYPES row, and `equivalent_submerged_weight_N_m` the row's `w`. The deck's
  `rhoW` and `g` (or CableDyn's defaults) resolve it; in a coupled run whose
  host sets other values, CableDyn uses the host's.
- **Records.** As in CableDyn, `#` and `!` start a comment anywhere in a
  record, inside a quoted token too (a quoted file path cannot hold them),
  and `--` starts one at the start of a record or after a space or tab,
  unless the record holds a `---` rule. A record holding `---` is a section
  header, named by its other characters; a section met again continues. A
  token that opens with a quote runs to its closing quote, spaces included,
  and on to the next separator. OUTPUTS names are separated by spaces, tabs
  or commas, with the quotes dropped; an unquoted `END` closes the list. A
  header naming no CableDyn section, a record longer than 512 bytes without
  a comment in them, a NUL or non-ASCII whitespace character outside a comment
  and a subnormal number stop the reader with the deck line, as they stop
  CableDyn; bytes that are not UTF-8 are kept.

## 2. Section history

`read_cabledyn_sections(out_root, deck, {"hangoff": 0.25, "sag": 58.4})`
returns, per section at unstretched arc length `s` from End A:
`time_s`, `effective_tension_N`, `rotation_curvature_x_1_m`,
`rotation_curvature_y_1_m`, `curvature_norm_1_m`, `elevation_m`,
`external_pressure_Pa`, `torque_Nm`, `twist_rad`, `twist_rate_per_m`
(Section 2.2), the labels of the frame, curvature source and tension source,
and a provenance record. The read also
returns the digests of every file read and the consistency checks.

**Curvature vector.** `curvature_source="tangent"` splines the tangents of
the `Dec`/`Azi` channels through a few nodes on either side (`window_nodes`,
default 3) and forms `κ = t × dt/ds`. With the default
`tangent_placement="auto"` (CableDyn 0.4.2 and later) the tangent of a
finite-EI line (any section with `EI > 0`) sits at its node and the chord of an
`EI = 0` line at the mid-arc of the chord (an end node at the mid-arc of its
element); `tangent_placement="chord"` places every value at the mid-arc of its
chord, for finite-EI output of CableDyn 0.4.1 and earlier. The placement is
recorded in `checks["tangent_placement"]`. Reading one version's output as the
other's moves the End-A tangent by half an element: 0.4.2 output of the
128-element lazy-wave test run (0.2 m elements at the hang-off) read with
`"chord"` puts the curvature at node 2 52 % away from CableDyn's `Curv`, and
the curvature check fails with a message that names the option.
`"positions"` splines node positions and forms `κ = r' × r'' / |r'|³`; the 8-digit position format limits its resolution to
about `δ/h²` (`δ` the last printed digit, `h` the element length: 2.2e-4 1/m
for the 0.15 m elements of the 80 m lazy-wave example), recorded in the checks.
`"auto"` uses tangents when all needed channels are present.

**Frame and signs.** `t` points from End A to End B. The transverse axes
satisfy `e1 × e2 = t`. At the first sample (the static state)
`e2 = a × t / |a × t|`, with `a` the reference vector (default global `+z`):
`e2` is horizontal, normal to the plane of a line hanging in a vertical
plane, and `e1 = e2 × t` lies in that plane. The axes then evolve without
twist: `frame="arc"` (default) transports them along the line from End A by
minimal rotations between element chords, with the End-A axes transported in
time; this is the material frame of a torque-free isotropic rod.
`frame="section"` transports the section's own axes in time and needs only
the channels near the section. Where `Twist` channels bracket a section the
axes are turned right-handed about `t` by the material twist from End A. A
line restrained in torsion at both ends without these channels is rejected.
So is a dynamic run of a line restrained at one end whose torsion CableDyn
solves: its statics are twist-free (a free end carries no torque), but in the
dynamics its polar inertia `Irn` carries torsional waves and a roll of the
restrained end's vessel or body twists it (CableDyn `doc/theory.rst`, a free
end). The CableDyn 0.4.2 run `tests/data/cabledyn/lw24k`, a lazy wave clamped
and restrained at End A on a vessel, free in torsion at End B and without
torque–tension coupling, twists by up to −1.7e-3° at the sag bend in 1 s and
0.18° in a 30 s run. With coupling (`GJ|K`) the stretch twists the line in
the statics too, so its channels are needed in any run. A line restrained at
one end whose torsion CableDyn does not solve runs as one without the torsion
columns and is read twist-free; `checks["torsion"]` records which case was
read.

**Hang-off rotation.** The `Twist` channels start at 0 at End A, and the
hang-off's turn about its own tangent (vessel yaw at a vertical hang-off) is
not visible in the geometry. It turns the material frame when CableDyn solves
the line in torsion with End A restrained: End A's frame is then held by what
holds it, which turns `Nx` (with `Ez`), carried to the line tangent by the
smallest rotation. With `frame="arc"` the reader follows it from the first
row: a `Fixed` point or `Fixed` rod holds it still, a `Body<N>` turns it with
its `Body<N>Rx/Ry/Rz` channels, and a `Coupled`/`Vessel` point or a turbine,
whose vessel or platform attitude the output does not record, needs
`hang_off_rotation`: the holder's attitude matrices or its roll about the End
A tangent at every output row (`ψ = ∫ω·t dt`; `yaw cos D` at a hang-off of
declination `D` on a vessel turning in yaw only), or `"none"` to declare it
still. Without it such a run is refused. A line turned rigidly with its body
keeps its components to round-off (`tests/test_cabledyn_outputs.py`); the
twist-free transport in time would turn them by the yaw times the vertical
component of the hang-off tangent. Restrained at End B only, the free End A
turns by `−Twist<L>` against End B's frame carried along the line, which the
reader builds from the positions of every node and the `Twist<L>` channel.
`frame="section"` represents neither and is refused where either applies;
`checks["hang_off"]` records the rule and the extent of the turn.

**Section axes of a torsion line.** On a line CableDyn solves in torsion the
reader's axes are CableDyn's section axes, those of `CurvX`/`CurvY`: the first
axis starts at End A's torsion normal `Nx` (End B's, rolled by its `Pretwist`,
carried back on a line restrained at End B only), is carried along the line and
turned by the twist; the reference vector is not used there and
`frame="section"` is refused. The reconstructed components are then
`(-CurvY, CurvX)` to the reconstruction's accuracy: 0.3 to 1.2 % on the coupled
VolturnUS-S run `tests/data/cabledyn/vu_torsion` (1.15 % at 21 m, 0.33 % at 59 m
and 131 m), where axes seeded by the reference vector stood 90° away.
`curvature_source="components"` reads `CurvX`/`CurvY` at the deck nodes around
each section directly: exact on CableDyn's axes, with no node positions or
angles (on an OpenFAST run only the channels written). OpenFAST writes its own
`<root>.out` beside `<root>.CD.out`; the reader takes the CableDyn file and
refuses only a CableDyn driver table there.

`rotation_curvature_x/y` are the components of `κ` on (`e1`, `e2`): a lazy
wave in the `x–z` plane bends about `e2` (sag bend `κ·e2 < 0`, hog bend
`> 0`); out-of-plane bending appears in `κ·e1`. `strain_gradient()` applies
the same `rotation_to_strain_curvature` as the time-history CSV path, so all global
codes enter `compute_fatigue` through one convention: the right-handed fibre
strain `k_x·y − k_y·x`, strain gradient `(−k_y, k_x)`, with the fibre angle
`φ` measured from `e1` towards `e2`. A sag bend (`k_y < 0`) is in tension on
`+e1`, the downward side of the static section (`φ = 0`); a hog bend on the
upper side (`φ = 180°`).

**Sign check.** Four static models of a global line program (a
10 m line in air, pinned at both ends and sagging, or clamped at End A and
hogging as a cantilever, each with End A gamma 0° and 90°) were read for
`x/y curvature`, `x/y bend moment` and the outer `ZZ stress` around the
circumference. In every case the stress follows `k_x·y − k_y·x`: the program's
components are those of the right-handed rotation vector `t × dt/ds`, and its
bend moments are `M = (Σf·y, −Σf·x)`, mapped to the gradient frame as
`(−M_y, M_x)` (`rotation_to_strain_moment`). The cantilever with gamma 0
(`(k_x, k_y) = (0, +0.372)` 1/m) has its largest tension at `θ = 180°` and the
sag (`(0, −0.136)`) at `θ = 0°`, as the concavity of each line requires. Earlier
versions of `rotation_to_strain_curvature` returned `(k_y, −k_x)`, the negative,
which put the tensile fibre half a turn from its physical place in the
time-history CSV, CableDyn and RIFLEX-rotation paths. The probe numbers are in
`tests/data/curvature_sign_probe.json`
(`tests/test_curvature_sign_probe.py`; no global solver is needed to run it),
and the CableDyn gates in `tests/test_cabledyn_reader.py` check that the
reader's tensile fibre lies on the convex side for sag, hog and horizontal
bends and on the downward side of the genuine lazy-wave sag bend. The earlier
mapping survives only as `curvature_map="paper_r2"` for reproducing earlier
results computed with it.

**Tension, pressure, material force.** `tension_source="auto"` takes the
element-midpoint forces of `.t.out` (linear between element mid-arcs, the
end-element value in the first and last half elements) and otherwise the node
channels. `pressure="hydrostatic"` forms `p_o = ρ_w g max(−z, 0)` from the
section elevation and the deck's `rhoW` and `g` (wave dynamic pressure is not
included). `true_tension_N(A_o)` returns `T_e − p_o A_o`.

**Checks (fail loudly).** Every file header is matched column by column;
OpenFAST units are checked per channel; a missing file or channel raises
`CableSchemaError` naming it (for example the `Dec`/`Azi` nodes of the spline
window, `Ten<L>N<J>`, the `p` flag for the arc frame, `Twist` on a restrained
line). The reader also requires: every file of the run listing the same
nodes of the line, and the deck nodes at the arc lengths of the deck
sections (below); first position row equal to
the static profile (files of one run); element chord within −2% to +5% of
its reference length at `t = 0`; uniform output step equal to `dtM`;
`.t.out` order fitting the `Ten` channels better than its mirror; and the
reconstructed curvature magnitude within 5% of each `Curv<L>N<J>` channel's
largest value (25% at nodes 2 and N−1; end nodes reported only, since there
CableDyn reports the end element's curvature with its clamped or pinned end
condition). A mesh too coarse for the curvature fails this check; state a
tolerance or refine the mesh.

**Refined meshes.** The static solve may refine a finite-EI line
(`adaptive_mesh`, `tensile_safety` or the automatic refinement). Each deck
node stays a node of the refined mesh at its own unstretched arc length; the
`<J>` of every node channel names a deck node, while `.static.out`,
`.elements.out` and the per-line files list every node of the refined mesh
(CableDyn `doc/outputs.rst`, node channels on a refined line). The reader
locates the deck nodes by the `DeckNode` column of `.static.out` or the
`# DeckNode` line of `.Line<L>.p.out` (failing both, by their arc lengths in
`.elements.out`); every numbering must name the deck nodes in order and agree
with the others, and the deck nodes must lie at the deck-section arc lengths
of `.elements.out`. Positions and segment tensions are read on the refined
mesh, the node channels at the deck nodes; `checks["mesh"]` records the node
counts and the source of the numbering. Without `.elements.out` the added
nodes cannot be placed: the positions are read at the deck nodes,
`tension_source="auto"` takes the node channels, and
`tension_source="segments"` or `tangent_placement="chord"` is refused.

**Deck outputs.** `section_channels(deck, sections)` lists the `OUTPUTS`
channels the tangent source needs for the section frame: `Dec`/`Azi` on the
window, `Ten`, `Curv` and `p{x,y,z}` at the bracketing nodes, and
`Twist`/`Torq` on a line CableDyn solves in torsion (`torsion_solved`;
CableDyn refuses them on any other line), with `Twist` also at the node before
a section that lies on a deck node (its twist rate, Section 2.2). The arc frame
also needs the line's `p` flag. Adding `t` gives the element tensions.

### 2.1 CableDyn 0.4.0 signed curvature components

Since CableDyn 0.4.0 a finite-EI line writes, at a node, the effective
tension `Ten<L>N<J>` and the curvature vector on two section axes,
`CurvX<L>N<J>` and `CurvY<L>N<J>` [1/m]. The vector points to the centre of
curvature, so a fibre at `(x, y)` has bending strain `−(CurvX·x + CurvY·y)`.
On a line not restrained in torsion `x` starts as the upper normal at the
hang-off, fixed in the hang-off axes (in global axes at a pinned End A), and
is parallel-transported along the line; on a line restrained in torsion it
starts from the restrained end's torsion normal `Nx` (carried to End A when
only End B is restrained) and is turned by the line's twist (CableDyn
`doc/theory.rst`, curvature on the section axes). The
local route takes the strain gradient on the same axes:

```text
g_x = −CurvX,   g_y = −CurvY
```

`cablexsec.io.read_cabledyn_station_table(path)` reads one station table
(comma, tab or whitespace separated; columns `time`, `Ten`, `CurvX`, `CurvY`,
names matched without case and with or without units) and
`cablexsec.io.series_from_curvature_components(t, Ten, CurvX, CurvY)` builds
the same `LoadCaseSeries` from arrays; both mark the tension as effective.
`(x, y, t)` must be right-handed with `t` from End A to End B;
`right_handed=False` reverses `y` for a left-handed triad. Wire azimuths are
then measured from CableDyn's `x` axis (the upper side at the hang-off), half
a turn from the `e1` of Section 2. Because the axes are fixed in the hang-off
axes, the hang-off's rotation about its own tangent, which the tangent
reconstruction of Section 2 cannot see, is carried. Tests:
`tests/test_cabledyn_components.py` (sag, hog and out-of-plane signs, a
rotating plane, separators, a missing column). The station reader takes one
station's columns under the names above; a CableDyn `.out` file holds them as
`Ten<L>N<J>`, `CurvX<L>N<J>` and `CurvY<L>N<J>` for every requested node, so a
station is passed to `series_from_curvature_components` from its three
columns. `read_cabledyn_sections(..., curvature_source="components")` reads
them at the deck nodes around each section (Section 2); a section placed on a
deck node, at the arc length `CableDynLine.node_arc_lengths()` gives it, needs
that node's `CurvX`, `CurvY` and `Ten` channels only (and its `Twist`/`Torq`,
recorded where written). A station given by a rounded arc length a few
micrometres off its node needs the next node's channels too; the error names
the node and its arc length.

### 2.2 Twist rate and the torsional wire stress

A twist of the cable stresses its helical wires, not only the frame. With plane
sections and fixed layer radii a wire of lay angle `a`, helix radius `R` and lay
sign `s` (+1 right hand, Z; −1 left hand, S) strains by

```text
eps_w = eps cos^2(a) + s R tau sin(a) cos(a)
```

under the cable strain `eps` and the twist rate `tau` (Knapp 1981, Eq. 11 with a
radially rigid core and the twist strain of its Eq. 2). `compute_fatigue(...,
twist_rate_per_m=tau)` adds this stress to every armour wire, the same for every
wire of a layer: the bending slip and the friction stress are unchanged, on
every friction model. With `tension_N` the tension is held, so the strain also
changes by `−K_et tau / K_ee` with the fixed-radius coupling `K_et` and
`EA_fixed = K_ee` of `compute_stiffness`. This is the section law `T = EA eps +
K tau`, `M = GJ tau + K eps` that CableDyn solves with `GJ|K`, so a deck
whose `EA`, `GJ` and `K` are the section's `K_ee`, `K_tt` and `K_et` gives
CableXSec the strain CableDyn has. `compute_conductor_fatigue(...,
twist_rate_per_m=tau)` adds the twist strain of the laid core, `(r_c tan(b) −
K_et/K_ee) tau`, to the conductor axis, and the twist of the ring wires about it.

**Input: the twist rate, not the torque.** The reader forms
`twist_rate_per_m` from `Twist<L>N<J>` alone. CableDyn's element twist is
`Twist_{k+1} − Twist_k = M L_0/GJ − (K/GJ)(l − L_0)`, the material twist of
the element, so `(Twist_{k+1} − Twist_k)/L_0` is its kinematic twist rate per
unstretched length. A section inside an element takes that element's rate, from
the two `Twist` channels the frame needs already. A section on a deck node takes
the mean of its two elements' rates, the definition CableDyn uses for
`Torq<L>N<J>`, and needs `Twist` at the node and at both neighbours (one
element at a line end). The torque is not converted:

- it includes the Kelvin–Voigt torsional damping (`GJ|K|Bt`);
- under a bend stiffener it is the section torque, not the cable's share;
- its conversion would need a torsional law, while the twist rate is the global
  model's own kinematics, like the curvature.

`Torq` is a consistency check instead: `Torq = GJ tau + K eps` with `eps =
(Ten − K tau)/EA`, to the print precision of the `Twist` channels.

**Signs.** CableDyn's `Twist` is right-handed about the End A → End B
tangent, so `tau > 0` tightens a right-hand lay: CableXSec's sign (Section 12
of the local analysis guide), with no flip. A right-hand (Z) armour layer takes
tension and a left-hand (S) layer compression under a positive twist rate.

**Deck outputs for the torsional stress.** At every station on deck node `J`
write `Twist<L>N<J−1>`, `Twist<L>N<J>` and `Twist<L>N<J+1>` (the end node and
its neighbour at a line end), and `Torq<L>N<J>` for the check. Where CableDyn
writes twist or torque at a section but not these channels, the reader warns
(`TorsionalStressWarning`, naming them) and `twist_rate_per_m` is `None`.
`checks["twist_rate"]` records the elements used for each section.

**Limits.** The term is the linear fixed-radius helix strain. It leaves out
the radial contraction of the layers (on the 66 kV template the solved radial
section gives the same twist stress within 6 %), the opening of a contact
under a twist that unloads a layer, the twist's change of the interlayer
contact pressure (the friction caps follow the tension), and the bending and
twisting of the individual wires. No measured wire strain under torsion is in
the validation set; row TQ-FT verifies the arithmetic and the consistency with
`K_et` and `K_tt` ([benchmarks.md](benchmarks.md)).

## 3. Sea-state batch

```python
import cablexsec as cx
from cablexsec.analysis.termination import TerminationSpec

cable = cx.load_cable("my_cable.yaml", geometry_override=None)   # needs your own file
runs = [
    cx.SeaStateRun("hs2_tp8", "runs/ss1", "runs/ss1.dat", 0.5, start_time_s=20.0),
    cx.SeaStateRun("hs4_tp10", "runs/ss2", "runs/ss2.dat", 0.3, start_time_s=20.0),
    cx.SeaStateRun("hs3_tp9_30deg", "runs/ss3", "runs/ss3.dat", 0.2, start_time_s=20.0),
]
sections = [
    cx.SectionRequest("hangoff", 0.25, termination=TerminationSpec(distance_m=0.25)),
    cx.SectionRequest("sag", 58.4, fretting=True),
]
result = cx.run_sea_state_batch(
    cable, runs, sections,
    options=cx.SeaStateBatchOptions(workers=2, exposure_years=25.0),
    output_directory="results/batch1", cable_source="my_cable.yaml",
)
result.sections["sag"].long_term.annual_damage, result.sections["sag"].life_years
```

- Each run is read once for all sections; the retained window drops the
  transient (the frame is still transported from `t = 0`).
- `compute_fatigue` runs per section with the material-force history, so the
  armour contact pressure follows the tension history; nothing is frozen at
  zero tension. `tension_basis="auto"` takes the effective tension for a
  cable whose `water_ingress.armour_annulus` is `flooded` and `T_e − p_o A_o`
  (cable outer diameter, hydrostatic pressure) for `sealed`; a cable that
  declares neither needs an explicit choice. A negative material force is
  rejected with the run, section and time.
- Section options: `termination` (clamp correction near an end fitting or a
  clamped stiffener root; the default clamp solver settings take minutes per
  section and record), `thermal` (`ThermalState`/`ThermalHistory`, solved
  contact model), `fretting` (`compute_fretting` with the run probability as
  time fraction; wear and fretting damage are summed per wire over the runs),
  and the user S–N options `sn_curves`, `mean_stress` and `scf`
  ([local_analysis_guide.md](local_analysis_guide.md) Section 4.2; in JSON a
  method name or the `MeanStressSpec` fields, and the `ComponentSCF` fields
  with `source`). They enter the section record only when set.
- Capacity check: `run_sea_state_batch(..., capacity=cx.CapacityBatchCheck(
  envelope, thresholds=(1.0,), groups={"hs4_tp10": [...seeds...]}))` with an
  envelope from `cx.build_capacity_envelope` whose tension grid spans the
  runs gives, per section and run, the utilisation history (written as
  `<section>_<run>_utilisation.npz`), its maximum and time, exceedances and
  the minimum bend radius check, plus a Gumbel fit of the maxima per group of
  seeds ([local_analysis_guide.md](local_analysis_guide.md) Section 8.1). The envelope's tension kind
  selects the effective tension or the material force of the tension basis.
  Fatigue results and their digests are unchanged by it.
- Annual damage: `compute_long_term_fatigue` at every fixed fibre,
  `Σ p_i s_year D_i / T_i`, before the maximum is taken; life
  `1/(DFF·D_year)`. Probabilities must sum to one (divide a condition's
  probability among its seeds).
- Before reading, the deck `EA` at each section is compared with the
  section's `EA_fixed` and the deck `EI` with the section's full-slip to
  no-slip range (`property_rtol`, default 5%). On a 14-column LINE TYPES row
  the deck `GJ` is compared with the section's lay torsional stiffness
  `K_tt` and its torque–tension coupling `K` with `K_et`, as the coupling
  ratio `c = K/sqrt(EA GJ)` against `K_et/sqrt(EA_fixed K_tt)` (`|c| < 1`;
  the free-twist axial stiffness is `EA (1 − c²)`), both within
  `torsion_rtol` (default 25 %). `K_tt` leaves out the components' own
  torsion and the radial relaxation of the layers, which move a section's
  torsional stiffness by tens of percent either way (the core-lay sensitivity
  of the coiling cable in the benchmark register), hence
  the wider tolerance; a 10-column row gives no `GJ` and is not compared. A
  mismatch stops the batch unless `allow_property_mismatch=True` is
  declared, and the comparison is written to the manifest either way.
- Torsional wire stress: `twist_stress=True` passes each section's
  `twist_rate_per_m` (Section 2.2) to `compute_fatigue`; on a line CableDyn
  solves in torsion a section without it is refused. With the default `False`
  a section whose history has a twist rate warns (`TorsionalStressWarning`)
  that the stress is omitted, and the result is that of earlier releases.
- Workers: `workers` processes, one numerical thread each, spawned, at most
  one per run; there is no upper limit, so choose the count for the machine
  (the runs are independent and the batch scales with the worker count while
  there are runs for every worker). Results do not depend on the worker count.
  Each worker keeps its compiled kernels in a cache folder of its own,
  `slot-<i>` under `cablexsec/numba-batch` in the user cache directory (set
  `CABLEXSEC_BATCH_CACHE_DIR` to move it), so later batches start warm; a
  slot is held by one worker at a time, a worker beyond the 16 slots gets a
  temporary cache, and deleting the folder only costs a recompilation.
- Output folder (must not exist): `manifest.json` (SHA-256 of every CableDyn
  file and deck, the cable definition and YAML file, options, property check,
  reader checks per run, results, and `inputs_sha256` over all of them);
  `<section>_annual_damage.npz` (per layer, wire and fibre);
  `<section>_<run>_history.npz` (columns as the `local_response`
  histories of local recovery: `time_s`, `effective_tension_N`, `true_tension_N`,
  `strain_gradient_x/y_1_m`, plus rotation components, norm, pressure,
  elevation, torque, twist).

The command line takes the same inputs as JSON:

```json
{
  "cable": "bundled:iea_task49_66kv",
  "runs": [
    {"name": "ss1", "out_root": "runs/ss1", "deck": "runs/ss1.dat",
     "probability": 0.5, "start_time_s": 20.0},
    {"name": "ss2", "out_root": "runs/ss2", "deck": "runs/ss2.dat",
     "probability": 0.5, "start_time_s": 20.0}
  ],
  "sections": [
    {"name": "hangoff", "arc_length_m": 0.25,
     "termination": {"distance_m": 0.25}},
    {"name": "sag", "arc_length_m": 58.4, "fretting": true}
  ],
  "options": {"workers": 2, "exposure_years": 25.0}
}
```

Paths are relative to the JSON file; unknown keys are rejected.

## 4. Verification

| Check | Result |
|---|---|
| Pure bending about `e2` (arc in a vertical plane) and about `e1` (horizontal arc), both curvature sources and both frames | components equal `(0, κ)` and `(−κ, 0)` within 2e-6 1/m |
| Rigid heading rotation of the configuration | components unchanged (1e-9) |
| Bending plane turning about the hang-off tangent by `φ(t)` (arc frame) | `κ(−sin φ, cos φ)` within 5e-6 1/m |
| Material twist of a restrained line | axes turned by the `Twist` channel; components `κ(sin φ, cos φ)` |
| Twist rate from `Twist` channels of two element rates (1e-3 and 3e-3 rad/m) | inside each element its rate, on the node their mean (1e-12); missing neighbour channels named |
| Tangent placement on a non-uniform mesh (0.2 m then 0.3 m elements): node tangents of a finite-EI line; chords of an `EI = 0` line and, with `tangent_placement="chord"`, of a finite-EI line | curvature within 1e-4 of `κ` at End A, the mesh junction and End B; node tangents read as chords give more than 1.5 `κ` at End A |
| Tension between element mid-arcs, hydrostatic pressure, `T_e − p_o A_o` | exact for a linear field; elevation within 1e-7 m |
| Round trip through the time-history CSV path: section history written as a CSV export (`x curvature`, `y curvature`, `Wall tension`) and read with `read_time_history_csv` | identical `LoadCaseSeries` (1e-12) and identical fixed-fibre damage |
| Batch annual damage against `Σ p_i s_year D_i/T_i` from direct `compute_fatigue` | equal to 1e-12 at every fibre |
| Worker count 1 against 2 | bitwise equal damage and equal input digests |
| Failure cases: missing channels or files, mesh or run mismatch, reversed segment order, curvature channel disagreement, step against `dtM`, OpenFAST units | each raises, naming the item |

Genuine CableDyn output (v0.1.0 driver, `build-rel`; deck from CableDyn's
`examples/lazy_wave_vessel_rao.dat`: Lozon et al. 80 m lazy-wave cable,
1024 elements, clamped hang-off on a vessel moving with illustrative RAOs,
`dtM = 0.05 s`, 120 s; JONSWAP seas Hs 2 m/Tp 8 s/0°, Hs 4 m/Tp 10 s/0° and
Hs 3 m/Tp 9 s/30°, weights 0.5/0.3/0.2, first 20 s discarded):

- Curvature magnitude from the tangent channels against `Curv1N<J>` at 18
  interior nodes, relative to each node's largest value: 0.007–0.1% along the
  line, 0.2–1.1% within 1.5 m of the clamped hang-off, 0.7% at the
  buoyancy-section junction and 5% at the node next to the pinned anchor
  (absolute errors 1e-6 to 1.5e-4 1/m). From positions: about 2e-4 1/m
  everywhere, the resolution estimate above.
- Head-on seas give `κ·e1 = 0` exactly (planar response); the 30° sea gives
  out-of-plane components with a standard deviation of 0.5–1.7e-3 1/m.
- The two frames agree to 1e-7 1/m at the hang-off; at the sag and hog bends
  of the 30° sea they differ by up to 1.1e-3 1/m in `κ·e1` (a frame rotation
  of about 0.6° on an in-plane curvature of 0.1 1/m). Weighted annual damage
  with the section frame is 1.000–1.007 times the arc-frame value.
- The position source gives 0.98–1.31 times the tangent-source damage; the
  largest ratio is at the section with the smallest curvature range, where
  the resolution limit adds small cycles. Write the tangent channels.
- `.t.out` midpoint forces against the `Ten` channels: 8e-4 relative on this
  mesh. On the 24-element test run in `tests/data/cabledyn` they differ by up
  to 195% (the pointwise force oscillates about the element mean on long
  curved elements), and that mesh also fails the curvature check by default.
- Run time: 4 sections × 3 runs (2401 rows each; reading about 2 s per run
  including file digests), 10.4 s on one worker, 8.4 s on two.

CableDyn 0.4.2 output (the 128-element run `lw128` in `tests/data/cabledyn`:
the same cable with 0.2 m elements over the first 6 m and 1.24 m beyond,
Hs 4 m/Tp 10 s/30°, 20 s): the curvature from the node tangents agrees with
`Curv1N<J>` at the hang-off nodes 2 to 4 and the sag-bend nodes 71 to 75 to
2.5e-4 of each node's largest value. Read with `tangent_placement="chord"`,
nodes 2 and 3 depart by 52 % and 15 %, the largest hang-off curvature at
0.25 m is 14.5 % higher (9.59e-3 against 8.37e-3 1/m), and the read fails
the curvature check. The v0.1.0 run of the same deck (`lw128_v010`) read
with `tangent_placement="chord"` agrees to 1.5 % at nodes 2 and 3.

The IEA Task 49 66 kV template stood in for the local section; its `EA` and
`EI` differ from the Lozon cable of the deck, so these runs check the
plumbing and the conventions, not a design result
(`allow_property_mismatch=True`).

## 5. Practical notes

- Request the tangent channels around every section (`section_channels`), and
  the `p` and `t` flags for the arc frame and element tensions.
- On coarse meshes prefer `tension_source="nodes"` and refine the mesh where
  the curvature check fails; the check compares against CableDyn's own
  element curvature, so its failure means the global mesh, not the reader,
  limits the result.
- Sections within the first element of a clamped end are extrapolated by the
  spline; the end node itself is not checked.
- A run that stopped early leaves only committed rows; the reader accepts the
  rows present and the retained window must lie inside them.

## 6. Comparison still open

The sign convention of the global `x/y curvature` is now fixed by the static
probe of Section 2. No independent global run of the matched lazy-wave deck
was made for this stage. The CableDyn validation records (CableDyn's
`VALIDATION.md` L3-6, L3-6c, L3-6e, L4-1c) compare curvature magnitude and
tension with a reference global solver, but not signed components. Remaining:
the same lazy-wave deck in an independent global solver,
`x/y curvature` and wall tension histories at the sections above, compared
with this reader's components (sign convention included) and the resulting
damage.

## 7. Stage 2 and limitations

Stage 1 is one way and uses CableDyn's linear-elastic bending (the `EI` of
the line type, tapered under a bend stiffener). Stage 2 is the
two-way hysteretic element: the section's slip-dependent moment–curvature
law (`xsec_stiffness`, the backbone of `compute_mkappa`) returned to CableDyn
as an element constitutive law with history (biaxial stick–slip state per
element and Gauss point, consistent tangent for the implicit integrator,
state transfer between steps and restarts), followed by the local recovery
of this guide on the consistently bent histories. Also open: wave dynamic
pressure, temperature histories from a coupled thermal run, and conductor
fatigue in the batch. (The hang-off roll about the cable axis is read since
the `hang_off_rotation` input and `curvature_source="components"`, Section 2.)

Two findings shape stage 2:

- **Biaxial law.** The bending plane of a dynamic cable can turn through full
  revolutions at the hang-off and at the stiffener tip in severe seas, so the
  planar Iwan export (Section 8) is not enough there. Its direct biaxial form,
  each element a two-component elastic–perfectly plastic element with a
  circular yield surface and radial return, reproduces the planar law exactly
  for a fixed plane and adds no parameter. Whether it reproduces the per-wire
  section under a swinging plane is to be checked against `compute_fatigue`'s
  biaxial wire response (moment components and dissipated work on rotating
  and irregular histories) before it is adopted.
- **Anchored hang-off.** A clamped end anchors the armour over a boundary layer
  of one to several metres, depending on the contact pressure, and stiffens the
  cable there. Correcting the stress locally after a global run that does not
  carry this stiffening over-predicts the measured near-end stresses (Sævik
  2011; [benchmarks.md](benchmarks.md), two-way anchored termination). The hang-off clamp
  belongs in the two-way element as an anchored boundary condition of the
  cable law.

## 8. Hysteretic bending law for global analysis

`export_bending_law` turns the section law into a bending law a global solver can carry. At a
set of tensions it computes the virgin moment–curvature backbone with `compute_mkappa` and decomposes it into
parallel elastic, perfectly plastic elements (an Iwan model, Iwan 1966) plus the full-slip stiffness. For planar
bending at fixed contact pressure the default and the progressive armour laws are themselves sets of parallel
spring–sliders, so the Iwan law follows Masing's rules and reproduces `compute_mkappa` on planar histories,
including the memory of earlier reversals. It is exact when every wire's slip interval is symmetric. The signed
capstan bound (Papailiou 1997) makes it asymmetric: the falling bound of a wire is its rising bound times
`exp(-λψ)`, which no Masing element carries. On bending cycles of one amplitude near full slip the residual
is about 0.1 % of the moment range (Ménard ±0.03 1/m: 0.14 % at 10 kN, 0.16 % at 70 kN; 66 kV template
0.08–0.11 %; 33 kV three-core 0.26 %) and falls with the friction coefficient (66 kV at μ = 0.02: 0.02 %).

On nested irregular histories it is larger. It peaks on reloading branches inside the doubled (Masing)
transition zone, where a reversal of less than twice the full-slip curvature leaves the wires partly
slipping, and it grows with the capstan factor λ = μ |sin α|. Random turning-point histories up to three
times the full-slip curvature were run on the Ménard cable (10 kN), the 66 kV template and the 33 kV
three-core cable (50 kN), each at μ = 0.085, 0.2 and 0.25. The largest residuals were:

| λ | Moment range | Full-slip friction moment |
| --- | --- | --- |
| about 0.03 | 0.1 % | 0.2 % |
| about 0.07 | 0.3–0.4 % | 0.6–0.8 % |
| about 0.09 | 0.4–0.6 % | 0.9–1.2 % |

The full-slip friction moment is the backbone less `EI_slip κ`. The largest values lie 0.3–1.8 full-slip
curvatures after a reversal. Once a branch leaves that zone the residual returns to the level of a single
cycle (0.01 % of the range), so the memory of earlier reversals, the wipe-out of closed cycles, is unaffected.

```python
import numpy as np
import cablexsec as cx

cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
export = cx.export_bending_law(
    cable, tensions_N=np.arange(0.0, 100e3 + 1, 20e3), kappa_max_1_m=0.2, n_points=200
)
law = export.law_at(35e3)              # linear interpolation between tension levels
moment = law.moment(0.1 * np.sin(np.linspace(0, 6 * np.pi, 600)))
export.to_json("bending_law.json")     # backbones, elements, settings, limitations
export.to_csv("bending_law.csv")       # kappa against M at every tension
pairs = export.backbone_table_kNm(35e3)  # (curvature rad/m, moment kN m) of the backbone
```

Curvature grid. The backbone is computed on a geometric grid from `1e-4 κ_max` to `κ_max` (`n_points`), extended
below at the same ratio until its first point lies under half the smallest first-slip curvature of any exported
tension, plus the slip onset and full-slip curvature of every armour layer (and of the laid cores) at every tension
and at the slip-scale reference. The first chord of every backbone is then the stick stiffness of the section, and
each stick-to-slip transition is resolved at the grid's density; `kappa_1_m` holds more than `n_points` points. A
low tension moves the onset far down (66 kV template: 7.3e-6 1/m for the outer armour at 10 kN), below a grid that
starts at `1e-4 κ_max`. With `spacing="linear"` the export adds one stick point and the onset and full-slip points
and warns when a transition is narrower than the step.

Checks (targets registered first; [benchmarks.md](benchmarks.md), hysteretic bending law): the Iwan law
reproduces `compute_mkappa` within 0.05 % of the moment range and the work within 0.02 % on symmetric cycles and
an irregular history, for the Ménard cable at 0 and 50 kN and the 66 kV template at 50 kN; the decomposition
identities hold to round-off. Interpolation in tension is the limiting step: between levels 100 kN apart the
backbone at the midpoint is missed by 3.9 % of the moment range (Ménard; registered target FAIL), 1.5 % at 50 kN,
0.27 % at 20 kN and 0.07 % at 10 kN. Export tension levels no more than about 20 kN apart, or use the slip scale
below.

### 8.1 Slip scaling variable

The tension and the external pressure change the armour contact pressure `P1+P2`, not the stick stiffness
`k_f = ξ E cos²α`. In the direction-tied law every wire is a spring–slider whose friction bounds are proportional to
its layer's `P1+P2`, and the return map `f ← clip(f + k Δd, −ρ l, ρ u)` is positively homogeneous. A layer whose
bounds scale by `ρ` therefore obeys `M(κ; ρ) = ρ M(κ/ρ; 1)` over any curvature history, loops and the asymmetric
capstan bound included, and its Iwan elements keep their stiffnesses `k_i` while their yield curvatures become
`ρ κ_y,i`. With `slip_scale=True` the export carries this form as `BendingLawExport.slip_scale`, a `SlipScaleTable`:

- the law at a reference tension split into friction groups, one per structural armour layer, `laid_cores` for
  the core lay stage and `filler_profiles` for its laid filler profiles, each with its elements (`k_i`, `κ_y,i`), plus the full-slip stiffness `EI_slip`;
- the cap `R_g` of every group (its planar full-slip friction moment) over a table of tensions and, optionally,
  external pressures, the group scale `ρ_g = R_g / R_g,ref`, the single cap-weighted scale
  `s = Σ R_g / Σ R_g,ref`, and their derivatives `∂/∂T` and `∂/∂p_e` (central differences of the contact model,
  one-sided at zero; the bounded chain is affine, so they are its slopes).

By default the table runs over the exported tensions with zero tension in front. At zero tension the caps are those
of the contact model: zero without manufacturing clamping (every contact load then comes from the tension, so
`s → 0` and the law tends to its elastic part), the clamping caps otherwise (66 kV template with cores and profiles
at 20 MPa winding stress: `ρ_g(0)` 0.82–0.87 of the 10 kN reference). A global model then interpolates the scale
down to zero tension instead of holding its lowest value. An explicit `slip_scale_tensions_N` is used as given.

```python
export = cx.export_bending_law(
    cable, tensions_N=[10e3, 70e3], kappa_max_1_m=0.036, n_points=300,
    slip_scale=True, slip_scale_tensions_N=np.linspace(10e3, 70e3, 7),
    slip_scale_external_pressure_Pa=[0.0, 1e6, 2e6],   # optional pressure axis
)
table = export.slip_scale
law = table.law_at(35e3, 0.5e6)                  # scaled law at one state
moment = table.moment(kappa, tension, pressure)  # sliders follow the changing caps
table.to_csv("slip_scale.csv")                   # s, rho_g and derivatives per state
groups = export.slip_group_laws()                # one law per friction group
moment = sum(g.backbone_at(kappa, 35e3) for g in groups)   # sum of s_g B_g(kappa / s_g)
export.to_group_csv("groups")                    # <group>_backbone.csv, <group>_slip_scale.csv
```

One law per group. A global model that takes one scale per law can still carry the per-group scaling: it takes one
law per friction group, `M_g(κ; T) = s_g(T) B_g(κ / s_g(T))`, with `B_g` the backbone of the group at the reference
(`SlipGroupLaw`, from `BendingLawExport.slip_group_laws()`) and `s_g = ρ_g` its own scale, and sums the group laws.
The first group carries the elastic part `EI_slip κ` (the scaling leaves it unchanged); the others end on the
plateau of their full-slip friction moment. `to_group_csv` writes each backbone (`kappa_1_m,M_Nm`, with two `# `
lines that name the group and the slope beyond the last point) and each scale table (`tension_N,
external_pressure_Pa, s, ds_dT_1_per_N`, plus `ds_dpe_1_per_Pa` with a pressure axis).

Slider update. When the state changes, every element follows the return map with the new cap,
`e_i ← clip(e_i + Δκ, −ρ_g κ_y,i, ρ_g κ_y,i)` and `M = EI_slip κ + Σ k_i e_i`: the slip state is carried across, a
falling cap clips a sticking or slipping element at fixed curvature (friction work, `k_i |e_1| |e_0 − e_1|` in the
backward-Euler form the section law reports) and a rising cap leaves it where it is. This is the rule the fatigue
kernels and `compute_mkappa(..., contact_tension_history_N=...)` apply to every wire.

Checks (`tests/test_bending_law_slip_scale.py`; row MX-scale of the [benchmark register](benchmarks.md)):

- `compute_mkappa` at 70 kN equals the 20 kN law with the bound scaled, `EI κ + ρ F(κ/ρ)`, over a loop history on
  the 66 kV template (no manufacturing clamping, so both layers and the cores scale by `T`): 4e-16 of the moment.
- Every group law decomposed at 60 kN on the grid `ρ_g κ` is the 10 kN group law with fixed `k_i` and yield
  curvatures times `ρ_g` (Ménard, layers scaling by 2.06 and 1.40; 33 kV three-core with its cores): below 1e-13.
- With symmetric wire bounds and a grid through every wire's slip curvature, the scaled law under a tension cycle
  (10–70 kN) inside bending cycles is the contact-tension history of `compute_mkappa`, moment and friction work,
  to 1e-14. With the signed bound the scaled law misses the history route by at most 1.02 times the
  fixed-tension residual above (Ménard 0.17 %, 66 kV 0.11 %); the law frozen at the reference misses by 19 % and
  43 %.

One scale or one per group. The single scale `s` keeps the stick stiffness and the full-slip moment exact and is
exact when every group scales alike. Otherwise it misplaces the transition. With armour clamping the laid cores
scale unlike the armour: on the 66 kV template with cores and filler profiles (μ 0.46) at 20 and 40 MPa winding
stress, 10–200 kN, one scale misses the exported backbone by up to 5.5 % and the cycle work by up to 7 %, while the
summed group laws stay within 4.6e-4 and 0.06 % (rows MX-group-B, MX-group-E and MX-group-1 of the
[benchmark register](benchmarks.md)). Largest error of the single-scale law
against the per-group law over 10–150 kN, reference 10 kN, on the virgin backbone and on cycles up to 1.5 times the
full-slip curvature (moment in % of the range; work per cycle at 1.5 times full slip):

| Template | Tension only | Tension and p_e 0–2 MPa |
| --- | --- | --- |
| 66 kV IEA Task 49 | 0 (both layers ∝ T) | 21 %; work 32 % |
| 66 kV with cores | 0 (cores ∝ T too) | not run |
| 33 kV three-core | 0.10 %; work 0.14 % | 12 %; work 16 % |
| 33 kV three-core with cores | 0.29 %; work 0.36 % | not run |
| Ménard | 12 %; work 11 % | 19 %; work 19 % |

The external pressure adds the same `P1+P2` to layers whose tension parts differ (on the 66 kV template, with no
clamping, `ρ` is 467 for the inner and 1 735 for the outer layer at 10 kN and 2 MPa), so a pressure axis needs the
per-group scales. A group of scale `ρ` reproduces its law at the state up to `ρ κ_max`; beyond, the full-slip slope
continues, as in every exported law. The default reference is the smallest exported tension at which every group
has contact, so caps that rise with tension give `ρ ≥ 1`.

Limits of the scale. The laid-core group scales with its own contact load; it is exact while the load-weighted
friction coefficient of its contacts is unchanged (one contact type, or equal coefficients, as by default). The
internal-ring slip of resolved strands (`strand_layers`) has no cap scaling and is refused. Torque is not a table
variable: the section states that carry a torque (the coupled qp and solved models with a free twist) differ at
zero torque from the fixed-twist states of the backbones, and a torque moves the `P1+P2` of contra-wound layers in
opposite senses. The stick stiffness and `EI_slip` are those of the reference state; at a different temperature the
moduli change, which the scale does not represent.

Limits. The law is planar. CableXSec's biaxial response is per wire and tied to the current bending direction; a
global solver that applies an isotropic hysteretic law to the resultant curvature makes an assumption the section
model does not. Tension and annulus state are fixed within one law of the tension table; the slip scale (Section
8.1) carries the change of the friction caps with tension and external pressure. Whether a global program's
hysteretic bending option applies Masing's rules to the exported table is target MX-5, open until checked; CableDyn
does not yet have a bending-law interface (its stage 2, Section 7).
