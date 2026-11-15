<!-- SPDX-License-Identifier: Apache-2.0 -->

# Key concepts

This is the model behind every CableXSec calculation. Read it once and the
tutorials, the recipes and the two guides read as variations on it.

## CableXSec is a section model under a line model

A {term}`global line model` such as CableDyn treats the cable as a slender rod.
It solves the shape and motion of the whole line under waves, current and
vessel motion, and it needs section properties to do so: axial stiffness `EA`,
bending stiffness `EI`, torsional stiffness `GJ`, mass and hydrodynamic
coefficients. At every point along the line it returns a load history:
tension, two curvature components, and possibly pressure and temperature.

A {term}`section model` works at one of those points. CableXSec resolves the
cross-section there layer by layer: the conductors and insulation of the
cores, the polymer sheaths and beddings, and every armour wire with its
position, {term}`lay angle` and contact with its neighbours. The two models pass
results in both directions:

```text
  global line model  ─── T, κx, κy at one section ───►  section model
  whole cable, waves,                                   one cross-section,
  current, vessel    ◄── EA, GJ, EI bounds, ──────────  every wire and layer
                         hysteretic bending law                │
                                                               ▼
                                                    stress at fixed fibres
                                                               │
                                                               ▼
                                                    rainflow and Miner damage
```

From the history at a section CableXSec gives the section properties the
global model needs, the hysteretic bending law of the armour included; the
stress of every wire and component; contact pressures, slip and fretting
quantities at the armour contacts; temperatures; and the fatigue damage of the
armour, conductor, screens, sheaths and tapes.

The section law treats each section on its own, as if the curvature were the
same along the wires over about one lay length. Where curvature changes faster
than that, in a bending test or a short specimen, the test-arrangement and
along-wire models take over (Sections 6.2 and 6.3 of the
[local analysis guide](local_analysis_guide.md)). Sections near an anchor are
handled in Step 6 of its Section 5.

## A cable is a YAML file of layers

`cx.load_cable` reads the file and checks it against a schema. The layer
types cover helical wire layers (round wires, rectangular tendons and profiled
wires), stranded conductors, laid power cores off the cable axis, concentric
polymer layers, corrugated metallic sheaths and helical tapes.

Every material property and dimension carries a cited source, and the bundled
published cables state theirs layer by layer. Geometry that cannot be shared
may be left as `null` and supplied from a separate file. A definition that is
incomplete or asks for something unsupported stops with a named error. It
never runs approximately. The full layout is on the
[file formats](file_formats.md) page.

## Armour wires stick, then slip

When the cable bends, a helical armour wire is stretched on the tensile side
of the bend and shortened on the compressive side. A wire that **sticks** to
its neighbours follows the bending of the section and carries the axial
stress `k r κ c`. Here `r` is the {term}`helix radius`, `κ` the curvature,
`c = |cos(φ − φ_b)|` for a wire at azimuth `φ` relative to the bending
direction `φ_b`, and `k = ξ E cos² α` the stick stiffness of a wire of modulus
`E` and lay angle `α`.

Friction has to carry that stress along the wire, and friction is capped
(Papailiou 1997). The cap is proportional to the {term}`contact pressure`,
through the friction-setting stress `σ0`, and changes along the curved wire
with the wire's own force (the capstan term), with `λ = μ |sin α|`, `μ` the
friction coefficient and `ψ = asin c`. A rising wire
stress presses the wire harder onto its bed, so it is capped at
`σ0 expm1(λψ)`; a falling one unloads the bed, so it can fall only to
`−σ0 (1 − exp(−λψ))` (Papailiou's Eq. 14 with the signed helix angle). Both
are zero on the {term}`neutral axis` of bending and largest on the extreme
fibres, and a full reversal spans `2 σ0 sinh(λψ)`. As the curvature grows:

1. Below the slip-onset curvature `κ_on` every wire sticks and the section
   bends with the no-slip stiffness.
2. At `κ_on` the wires next to the neutral axis reach their cap and start to
   slip (on the compression side slightly earlier, at `(1 − 3λ²/8) κ_on`).
   `κ_on` is proportional to the friction coefficient and the contact
   pressure.
3. The slip front spreads from the neutral axis towards the extreme fibres. A
   slipping wire keeps the friction stress of its cap (Coulomb saturation) and
   from then on adds only its own bending stiffness.
4. When the wire on the tension-side extreme fibre slips, at
   `κ_on expm1(λ π/2) / λ`, the whole layer is in full slip and the moment
   rises with the full-slip stiffness.

On a reversal the wires stick again, and slip in the opposite sense once the
change of curvature is large enough. The moment–curvature path is therefore a
{term}`hysteresis loop`, and its area is the work friction dissipates in one
cycle.

`cx.compute_mkappa` returns the loop, the stored elastic energy and the
cumulative friction work (`result.work`). The two add up to the work done on
the section. `slip_onset_curvature` and `full_slip_curvature` in
`cablexsec.mechanics.papailiou` return the two characteristic curvatures of a
layer. [Tutorial 1](tutorials/01_first_cable.md) and
[Tutorial 2](tutorials/02_bending_hysteresis.md) work through both.

Each wire is a spring–slider with its own state, so the law follows any
history of the two curvature components. Three friction laws are available:

- `"direction_tied"`, the default: the cap of every wire follows the current
  bending direction.
- `"progressive"`: each wire is split into parallel spring–sliders, so it
  leaves stick gradually.
- `"distributed"`, used by the fatigue functions: the friction force is solved
  along one pitch, which removes the dependence on a bending direction.

Section 9 of the local analysis guide helps choose between them.

The laws give the same stress ranges but not the same friction work when the
bending direction rotates. The direction-tied cap turns with the bending
direction, so on a circular orbit a wire slips mostly near the neutral axis,
where its cap is small. Its work per revolution is then 0.16 of the
distributed law's at twice the slip onset and 0.57 at large amplitude, while
the wire stress ranges agree within 10 %. The frictional heat of
`compute_thermal_fatigue` is therefore always the dissipation of the
distributed law.

## Two bounds bracket the bending stiffness

`cx.compute_stiffness` returns two bending stiffnesses, and the hysteretic
response lies between them.

**Full slip**, `ei_min_Nm2` (Witz and Tan 1992). Every wire, core and layer
bends about its own axis. This is the slope of the loop once all wires slip.
It barely depends on friction or pressure, which makes it a calibration-free
check against a test.

**No slip**, `ei_max_Nm2`. The wires stick and follow the bending of the
section. Their contribution is reduced by the {term}`shear coefficient` ξ of each
armour layer (Yun et al. 2020), which accounts for the shear compliance of the
polymer layers that carry the wire forces. `ei_plane_section_Nm2` is the same
assembly with ξ = 1. The loop starts with the no-slip slope.

On the Ménard and Cartraud (2023) cable the bounds are 3.57 and 59.1 kN m², a
factor of about 17. Neither is the stiffness of a cable in service, which
depends on the curvature amplitude, the history and the contact pressure. For
a global analysis, export the hysteretic law or choose a secant stiffness for
the curvature range of interest.

## Friction needs a contact pressure

CableXSec works with the armour contact pressure `P1 + P2` on the two faces of
each wire layer. It sets the friction cap, and through the cap the slip-onset
curvature, the loop width, the friction stress of every wire and the fretting
loads. The pressure has four sources.

**Manufacturing clamping.** The winding stress retained in the armour, or a
contact pressure left by extrusion and sheathing. It is an input, not a
prediction. Each armour layer declares it with a basis (measured,
specification, published calibration, assumed, representative or not
published) and a source. A cable-level `initial_state` pressure covers every
layer; the Ménard cable carries 0.3 MPa, a published calibration. A cable
without a declaration, or with zero clamping because none is published, runs
with a `MissingClampingStateWarning`, and its friction results rest on the
tension alone. Section 4.4 of the local analysis guide shows how to declare
it.

**Tension.** A helical layer under tension presses inwards, so the pressure
rises with tension.

**External pressure.** Its effect depends on the {term}`annulus`. In a sealed,
dry annulus the hydrostatic pressure acts on the outer sheath and clamps the
armour. In a flooded annulus seawater reaches the armour and hydrostatic
pressure adds almost no net clamping. The annulus also decides which S–N curve
applies to the armour: seawater for flooded, in air for sealed and intact.
Both the pressure and the curve are explicit inputs (Section 7.2 of the local
analysis guide).

**Temperature.** Differential thermal expansion of the layers changes the
clamping of an energised cable (Section 7).

Three contact models turn these loads into pressures:

| Model | What it is | Use |
| --- | --- | --- |
| `"bounded"` | fixed-radius chain that passes the pressure from layer to layer | default of the bending, stress and fatigue functions; supports offset cores |
| `"qp"` | coupled equilibrium of concentric layers with Poisson effects and contact opening | concentric cables; default of `solve_section_state` |
| `"solved"` | elastic contact problem of the whole section, three-core bundles included | `solve_cable_radial_state`; [solved radial contact state](radial_model.md) |

On the same cable the models can differ by tens of per cent, as Tutorial 1
shows. By default a bending calculation holds the pressure of one contact
state. `compute_mkappa` can let it follow a tension history
(`contact_tension_history_N`) or take a per-sample pressure for each armour
layer (`contact_pressure_Pa`), as a fatigue calculation does with its tension
history. The friction bound of each wire then changes from sample to sample in
the same return map, with the slip state carried across. A falling bound clips
a slipping wire's friction stress, and the released elastic energy is friction
work; a rising bound leaves a sticking wire unchanged. Only the armour bound
follows the history: the stick stiffness, the full-slip stiffness and the
contact loads of the laid cores stay at the reference state. The bending-test
simulator and the exported bending law keep a fixed pressure.

## Fatigue is counted at fixed material points

A fatigue crack starts at a point of the material, so CableXSec counts fatigue
at {term}`fixed material points`. Every wire sits at its own azimuth round the
cable, and on every wire a set of fibres sits round its perimeter: 32 by
default, at angles measured from the outward radial direction. The stress
history of each point is the axial stress from tension, plus the local
bending stress of the wire, plus the friction stress, each with the slip state
of that wire. Each point is rainflow-counted and its Miner damage summed on its
own. The result names the critical wire and fibre.

Counting the largest stress round the circumference instead would be wrong.
When the bending direction rotates, that maximum can stay constant while every
material point goes through a full stress cycle. The envelope then misses the
very cycles that cause the damage. For the same reason, seeds and sea states
are combined only at identical (wire, fibre) points.

The friction state of a history has to be built up before it is counted.
Discard a lead-in with `count_from_sample`; Section 5 of the local analysis
guide works through an example.

## Curvature is an axial-strain gradient

CableXSec describes curvature by the axial-strain gradient `(kx, ky)`. A fibre
at `(x, y) = r (cos φ, sin φ)` has the bending strain `kx x + ky y`, so the
vector points to the most tensile fibre.

Global programs usually report the right-handed rotation curvature
`(k_x, k_y)`, for which the fibre strain is `k_x y − k_y x`. The readers
convert one to the other:

```text
strain gradient = (−k_y, k_x)        bend moments: (−M_y, M_x)
```

CableDyn 0.4.0 writes `CurvX` and `CurvY`, a vector that points to the centre
of curvature, and the strain gradient is `(−CurvX, −CurvY)`. A curvature
magnitude alone loses the reversals and the rotation of the bending direction,
and cannot be used for fatigue. The sign checks behind these mappings are in
Section 4.3 of the local analysis guide and Section 2 of the
[global–local guide](global_local_guide.md).

## Fatigue needs the true wall tension

The armour carries the true wall tension. The fatigue functions reject a
history marked as {term}`effective tension` until it is converted with
`series.with_true_tension(...)`, and they do not assume either kind for an
unqualified force column. The `cablexsec-batch` command is the exception: for a
flooded annulus it uses the effective tension, which is then the force the
armour carries.

## Results travel from the global model and back

The usual workflow runs in one direction:

1. Run the global analysis and write the history at the sections of interest.
2. Read each section into a `LoadCaseSeries` with `cx.read_time_history_csv`,
   `cx.read_riflex_ascii`, or `cx.read_cabledyn_sections` followed by
   `.histories[name].to_load_case_series(area)`.
3. Compute the stress and damage at the fixed material points with
   `cx.fatigue_from_series` or `cx.compute_fatigue`.
4. Weight the sea states with `cx.compute_long_term_fatigue`, or run a set of
   CableDyn sea-state runs with the `cablexsec-batch` command (Section 3 of the
   global–local guide).

The section law also goes back. `cx.compute_stiffness` gives linear
properties. `cx.export_bending_law` computes the first-loading
moment–curvature curve at a set of tensions and decomposes it into parallel
elastic–perfectly-plastic elements, an Iwan model. For planar bending at fixed
contact pressure the armour law is itself a set of parallel spring–sliders,
so the exported law follows Masing's rules and reproduces `compute_mkappa`,
memory of earlier reversals included. The law is planar, and tension and
annulus state are fixed within one law. Export tension levels no more than
about 20 kN apart (Section 8 of the global–local guide).

## Every result says where it came from

Every result carries a `provenance` record: the method, the cited sources, the
verification gates and the limitations of the calculation. A run that
succeeds but rests on an assumption, or leaves its validated range, says so
with a warning such as `MissingClampingStateWarning` or
`ExtrapolationWarning`. Comparisons with published measurements, including
those that miss their targets, are listed in the
[validation record](validation.md).

## Where to go next

| Page | What it covers |
| --- | --- |
| [Your first cable](tutorials/01_first_cable.md) | Layers, stiffness bounds and contact state of bundled cables |
| [Bending hysteresis](tutorials/02_bending_hysteresis.md) | Moment–curvature loop against a measured loop; work; friction and clamping |
| [User guide](user_guide.md) | The workflow through the Python interface |
| [Local analysis guide](local_analysis_guide.md) | Inputs, conventions, model choices and limits in depth |
| [Global–local guide](global_local_guide.md) | From global runs to armour fatigue; the bending-law export |
| [Solved radial contact state](radial_model.md) | The contact models and their verification |
