<!-- SPDX-License-Identifier: Apache-2.0 -->

# Global–local guide: CableDyn runs to armour fatigue

Stage 1 of the global–local coupling is one way: CableDyn solves the global
cable, CableXSec reads the histories of selected material sections and runs
its local fatigue on them, as it does on other global results. The global bending
law stays CableDyn's (constant `EI`); the hysteretic two-way element is
stage 2 (Section 7).

| Piece | Where |
|---|---|
| Reader | `cablexsec.io.cabledyn`: `read_cabledyn_deck`, `read_cabledyn_sections`, `section_channels`, `CableDynSectionHistory` |
| Sea-state batch | `cablexsec.analysis.sea_state_batch`: `run_sea_state_batch`, `SeaStateRun`, `SectionRequest`, `SeaStateBatchOptions`, `load_batch_spec` |
| Command line | `cablexsec-batch batch.json --output-directory <new folder>` |
| Tests | `tests/test_cabledyn_reader.py`, `tests/test_sea_state_batch.py`; genuine CableDyn output in `tests/data/cabledyn/` |

## 1. What CableDyn writes

From the CableDyn manual (`doc/outputs.rst`, `doc/conventions.rst`) and the
writers in `src/CableDyn_DeckDriver.f90` (CableDyn v0.1.x):

- **Files.** Standalone driver: `<root>.out` (title line, header
  `Time(s)` and the deck `OUTPUTS` tokens, no units row);
  `<root>.Line<L>.p.out` (node positions, `Node<J>X(m)` …, flag `p`);
  `<root>.Line<L>.t.out` (element tensions, `Segment<k>Tension(N)`, flag `t`);
  `<root>.static.out` and `<root>.elements.out` (initial state and element
  reference arcs, cubic-Hermite route). OpenFAST (`CompMooring = 5`):
  `<root>.CD.out` with a header `Time` and a units row, and
  `<root>.CD.static.out`; there are no per-line files. Fields are
  TAB-separated; values have 8 significant digits, time 17. The output step is
  always `dtM`.
- **Order and frame.** Node 1 is End A (hang-off), node N End B; the parser
  swaps a line listed anchor first. Global axes are right-handed, `z` up,
  still water at `z = 0`, all SI, angles in degrees.
- **Curvature.** `Curv<L>N<J>` is a magnitude at nodes only (for finite-EI
  lines the larger of the two one-sided element curvatures at an interior
  node). No signed components and no material frame are written.
- **Tangent.** `L<L>N<J>Dec` and `L<L>N<J>Azi` give the direction of the chord
  between the neighbouring nodes (one-sided at the ends), End A to End B.
- **Tension.** Effective tension of a solid line without contents
  (`T_e = EA ε`). `.t.out` holds the element-midpoint axial force;
  `Ten<L>N<J>` the length-weighted mean of the two neighbouring element-mean
  forces (an end node: the end force). On a curved coarse element the two
  differ (Section 5).
- **Torsion.** Only on a line restrained in torsion at both ends:
  `Torq<L>N<J>` (uniform along the line) and `Twist<L>N<J>`, the material twist
  from End A in degrees.
- **Pressure.** Not written.

## 2. Section history

`read_cabledyn_sections(out_root, deck, {"hangoff": 0.25, "sag": 58.4})`
returns, per section at unstretched arc length `s` from End A:
`time_s`, `effective_tension_N`, `rotation_curvature_x_1_m`,
`rotation_curvature_y_1_m`, `curvature_norm_1_m`, `elevation_m`,
`external_pressure_Pa`, `torque_Nm`, `twist_rad`, the labels of the frame,
curvature source and tension source, and a provenance record. The read also
returns the digests of every file read and the consistency checks.

**Curvature vector.** `curvature_source="tangent"` splines the chord tangents
of the `Dec`/`Azi` channels (each placed at the mid-arc of its chord) through a
few nodes on either side (`window_nodes`, default 3) and forms
`κ = t × dt/ds`. `"positions"` splines node positions and forms
`κ = r' × r'' / |r'|³`; the 8-digit position format limits its resolution to
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
the channels near the section. On a line restrained in torsion the axes are
turned right-handed about `t` by the `Twist` channel (a restrained line
without these channels is rejected). The hang-off's rotation about its own
tangent (vessel yaw at a vertical hang-off) is not visible in the geometry and
is not represented.

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
line). The reader also requires: mesh of `.elements.out` equal to the deck
sections (an adaptive mesh needs that file); first position row equal to
the static profile (files of one run); element chord within −2% to +5% of
its reference length at `t = 0`; uniform output step equal to `dtM`;
`.t.out` order fitting the `Ten` channels better than its mirror; and the
reconstructed curvature magnitude within 5% of each `Curv<L>N<J>` channel's
largest value (25% at nodes 2 and N−1; end nodes reported only, since there
CableDyn reports the end element's curvature with its clamped or pinned end
condition). A mesh too coarse for the curvature fails this check; state a
tolerance or refine the mesh.

**Deck outputs.** `section_channels(deck, sections)` lists the `OUTPUTS`
channels the tangent source needs for the section frame: `Dec`/`Azi` on the
window, `Ten`, `Curv` and `p{x,y,z}` at the bracketing nodes, and
`Twist`/`Torq` on a restrained line. The arc frame also needs the line's `p`
flag. Adding `t` gives the element tensions.

### 2.1 CableDyn 0.4.0 signed curvature components

CableDyn 0.4.0 writes, per station, the effective
tension `Ten` and the curvature vector on two section axes, `CurvX` and
`CurvY` [1/m]. The vector points to the centre of curvature, so a fibre at
`(x, y)` has bending strain `−(CurvX·x + CurvY·y)`; `x` is the upper normal at
the hang-off, fixed in the hang-off axes and parallel-transported along the
line. The local route takes the strain gradient on the same axes:

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
rotating plane, separators, a missing column). Reading the 0.4.0 `.out` files
directly follows once their channel names are fixed by a release.

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
  no-slip range (`property_rtol`, default 5%); a mismatch stops the batch
  unless `allow_property_mismatch=True` is declared, and the comparison is
  written to the manifest either way.
- Workers: `workers` processes (at most `MAX_WORKERS = 8`), one numerical
  thread each, spawned; results do not depend on the worker count.
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

Stage 1 is one way and uses CableDyn's constant `EI`. Stage 2 is the
two-way hysteretic element: the section's slip-dependent moment–curvature
law (`xsec_stiffness`, the backbone of `compute_mkappa`) returned to CableDyn
as an element constitutive law with history (biaxial stick–slip state per
element and Gauss point, consistent tangent for the implicit integrator,
state transfer between steps and restarts), followed by the local recovery
of this guide on the consistently bent histories. Also open: hang-off roll
about the cable axis (from the vessel attitude), wave dynamic pressure,
temperature histories from a coupled thermal run, and conductor fatigue in
the batch.

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
spring–sliders, so the Iwan law follows Masing's rules exactly and reproduces `compute_mkappa` on any planar
history, including the memory of earlier reversals.

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

Checks (targets registered first; [benchmarks.md](benchmarks.md), hysteretic bending law): the Iwan law
reproduces `compute_mkappa` within 0.05 % of the moment range and the work within 0.02 % on symmetric cycles and
an irregular history, for the Ménard cable at 0 and 50 kN and the 66 kV template at 50 kN; the decomposition
identities hold to round-off. Interpolation in tension is the limiting step: between levels 100 kN apart the
backbone at the midpoint is missed by 3.9 % of the moment range (Ménard; registered target FAIL), 1.5 % at 50 kN,
0.27 % at 20 kN and 0.07 % at 10 kN. Export tension levels no more than about 20 kN apart.

Limits. The law is planar. CableXSec's biaxial response is per wire and tied to the current bending direction; a
global solver that applies an isotropic hysteretic law to the resultant curvature makes an assumption the section
model does not. Tension and annulus state are fixed within one law. Whether a global program's hysteretic bending option
applies Masing's rules to the exported table is target MX-5, open until checked; CableDyn does not yet have a
bending-law interface (its stage 2, Section 7).
