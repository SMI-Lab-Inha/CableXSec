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
- `"distributed"`: the friction force is solved along one pitch, which
  removes the dependence on a bending direction.

The fatigue functions also take `"auto"`: the direction-tied law for a planar
curvature path, the distributed law for any other. Their default is
`"direction_tied"` in this release and changes to `"auto"` in the next one;
until then the default warns (`DirectionTiedFrictionWarning`, a
`FutureWarning`) when the path is not planar, the only case in which the two
differ. Section 9 of the local analysis guide helps choose
between them.

The laws differ when the bending direction rotates. The direction-tied cap
turns with the bending direction, so on a circular orbit a wire slips mostly
near the neutral axis, where its cap is small. Its work per revolution is then
0.16 of the distributed law's at twice the slip onset and 0.57 at large
amplitude, and its largest wire stress range is 0.94 of the distributed one
at twice the slip onset on the 66 kV template interface and 0.86 on a
near-rigid interface (bulk damage about 0.55 to 0.8 times at an S–N slope of
4). On planar paths its stress ranges are at or above the distributed ones.
For a non-planar record use `friction_model="auto"` or `"distributed"`; the
frictional heat of `compute_thermal_fatigue` is always the dissipation of the
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
layers between the wires and the surface that supports them: the concentric
layers down to the next helical layer inside or to the laid cores.
`ei_plane_section_Nm2` is the same assembly with ξ = 1. The loop starts with
the no-slip slope.

On the Ménard and Cartraud (2023) cable the bounds are 3.57 and 190 kN m², a
factor of about 53. Neither is the stiffness of a cable in service, which
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

## The friction coefficient can depend on slip state and rate

By default every contact pair has one rate-independent Coulomb coefficient
`mu`. A `FrictionLaw` replaces it by an effective coefficient `m mu`, with a
factor `m` that each wire (direction-tied model) or helix node (distributed
model) carries on its own. The factor enters where the kernel uses `mu`: in
the capstan exponent of the wire bounds, `sigma0 g expm1(m lam psi)` and
`sigma0 g (1 - exp(-m lam psi))`, and on the whole node bound
`m (rho g + lam n)` of the distributed model.

**Static and dynamic coefficient.** A wire keeps the static factor
`static_factor` (`mu_s / mu`) until it slips, then the dynamic factor
`dynamic_factor` (`mu_d / mu`) while it slips, and returns to the static one
after `stick_time_s` of stick. Each step is a backward-Euler return map with
the factor of the end of the step. A wire whose trial stress leaves the
interval of its current factor starts or continues a slip episode and is
returned to the dynamic interval. With `mu_s > mu_d` the loop has a breakaway
peak: the stress drops from the static to the dynamic bound in the step of
breakaway. With `mu_s < mu_d` (velocity strengthening) the stick episode ends
at the static bound without slip, and the wire continues elastically up to
the dynamic bound. A healing time of `inf` (default) never heals, `0` heals
after one stick step.

**Rate dependence.** At slip rate `v` the dynamic factor becomes `m_d R(v)`,

$$
R(v) = 1 + c \ln(1 + v / v_0) + v / v_\eta ,
$$

the logarithmic law `mu(v) = mu_d (1 + c ln(1 + v/v_0))` plus a regularised
(viscous) Coulomb term `t = b sign(v) + eta v` with `eta = b / v_eta`, a
Perzyna overstress proportional to the rate-independent bound `b`. The slip
increment `dp` of a slipping wire solves
`|sigma_trial| - k dp = B(m_d R(L dp / dt))` with `B` the bound on its side;
the left side falls and the right side rises with `dp`, so the root is
unique and the return is exact for the step. Only `c >= 0` is accepted
(velocity weakening loses this uniqueness). The direction-tied wire slips in
a reduced strain; its slip rate is that strain rate times `L = r / |sin a|`,
the wire length per radian of helix, which turns the extreme-fibre strain
`r kappa` into the largest slip displacement of a freely sliding wire. It is
a representative wire velocity, not a local one. The distributed model takes
the static/dynamic option only.

**Contact-pressure dependence.** `pressure_factor` is a table of
`(P, f)` pairs, a further factor `f(P)` on the coefficient, piecewise linear in
the local friction-setting pressure `P` and clamped to the end factors
outside the table; the effective coefficient is `m f(P) mu`. `P` is the
pressure with which the kernel evaluates the bound at that sample:
`(P1 + P2) g` of the wire or helix node, the sum of the two face pressures of
the armour layer times the contact shape, so it follows a tension or pressure
history. A rig that presses one wire between two faces at the same pressure
`p`, as in the tests below, has `P = 2 p`. Where `P1` and `P2` differ, one
factor at their sum stands for both faces (exact for equal faces or a linear
factor). Zero contact gives a zero bound whatever the factor. The factor is
rate independent: it needs no time base, the distributed model takes it, and
the exported bending law accepts a law that has only a pressure factor at the
fixed contact state of each exported tension. With the slip scaling variable
it is refused: the scaling rests on `M(kappa; rho) = rho M(kappa/rho; 1)`,
which holds when every bound is proportional to `P1 + P2` at a fixed
coefficient, and `f(P)` moves the coefficient inside the capstan exponent
with the pressure.

**Measured armour friction.** Friction loops of a tensile armour wire
between two anti-wear tapes cut from a flexible pipe (Dai, Sævik and Ye,
Marine Structures 59 (2018) 423–443, Sec. 4 and Fig. 4, pp. 428–430)
re-stick at every reversal: each stroke starts with a breakaway peak above
the dynamic level, at 0.01 Hz as at 0.2 Hz. Their law is `stick_time_s = 0`,
not the default `inf`. The printed coefficients of the in-service surfaces
are `mu_s` 0.38, 0.24, 0.22 and `mu_d` 0.32, 0.20, 0.18 at 1, 4 and 8 MPa
contact pressure (p. 428): a static to dynamic ratio of 1.19–1.22, and a
coefficient that falls with pressure by a factor 1.8 on `mu_d` from 1 to
8 MPa. Over 0.01–0.2 Hz at 2 mm amplitude (peak sliding speeds 0.13–2.5
mm/s) the dynamic level shows no rate effect (`c = 0`, no viscous term),
while the static peak rises by about 10 % (8 % in the paper, p. 430), which
the law cannot represent: its rate factor acts on the dynamic factor only.
The measured force also falls from the peak to the dynamic level over a
travel of 0.6–2.3 mm, where the law drops in one step. The comparison is
the DAI18 rows of the benchmark register (one spring-slider, not a cable
section). With `mu = 0.20` (the library value) these data read as follows;
the rig's face pressures of 1, 4 and 8 MPa are `P = 2, 8, 16 MPa`:

```python
law = cx.FrictionLaw(
    static_factor=1.2,  # mu_s / mu_d, 1.19-1.22
    stick_time_s=0.0,  # re-stick at every reversal
    pressure_factor=((2e6, 1.6), (8e6, 1.0), (16e6, 0.9)),  # mu_d / 0.20
)
```

One specimen per condition was tested, on one pipe's tape; the factors are
a parameter choice for a similar interface, not a property of every tape.

**Energy.** The work of a slipping wire splits into the rate-independent
friction work `B(m_d) |dp|` and the rate-dependent (viscous) part
`[B(m_d R) - B(m_d)] |dp|`; `BendingWork.viscous_J_per_m` reports the second,
and the balance input work = stored energy + friction + viscous dissipation +
backward-Euler remainder closes to round-off. One slider driven at full slip
by `x0 sin(omega t)` dissipates `4 b x0` by friction and `pi eta omega x0^2`
by the viscous term per cycle (`slider_history` reproduces both).

**Where it applies.** `compute_mkappa` and `compute_fatigue` take
`friction_law` with the sample times `time_s` when the law needs them (a rate
term or a finite positive healing time); without a time base such a law is
refused, and `time_s` without one is refused too. The stress-free start lies
one sampling interval before the first sample. The progressive (Iwan) model,
the laid-core friction, the termination clamp increment and the bending-test
simulator keep the rate-independent coefficient and refuse a non-neutral law.
The exported bending law refuses a static/dynamic or rate-dependent law as
well: its parallel elastic-plastic elements with Masing's rules are rate
independent and cannot represent it (a pressure factor alone is accepted, as
above). Neutral parameters (`mu_s = mu_d = mu`, `c = 0`, `v_eta = inf`, no
pressure factor) leave every result unchanged, bit for bit. The validation
set holds the spring-slider comparison above; no rate law and no
static/dynamic comparison on a cable section.

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
memory of earlier reversals included. The residual comes from the asymmetric
slip interval of the signed capstan bound. It is about 0.1 % of the moment
range on cycles of one amplitude near full slip. On nested irregular
histories, on reloading branches in the doubled transition zone, it grows
with the capstan factor, to 0.4–0.6 % of the moment range at λ ≈ 0.09
(Section 8 of the global–local guide). The law is planar, and
tension and annulus state are fixed within one law. Export tension levels no
more than about 20 kN apart, or export the slip scale (`slip_scale=True`):
tension and external pressure scale the friction caps of each armour layer and
of the cores, not their stiffness, so one law with scaled yield curvatures
covers the table and its sliders follow a changing tension (Section 8 of the
global–local guide).

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
