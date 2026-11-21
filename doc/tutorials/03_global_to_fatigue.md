<!-- SPDX-License-Identifier: Apache-2.0 -->

# From a global time history to armour fatigue

**Goal:** read the curvature and tension history of a cable section from a
{term}`global analysis` export, recover the stress of every armour wire at fixed
material fibres, count the cycles of every fibre and sum the Miner damage, and
find the governing wire and fibre with its
{term}`damage-equivalent stress range`.

**Script:**
[`doc/tutorials/scripts/tutorial_03_global_to_fatigue.py`](https://github.com/SMI-Lab-Inha/CableXSec/blob/main/doc/tutorials/scripts/tutorial_03_global_to_fatigue.py)
· **Run time:** about ten seconds

Run it from a source checkout:

```text
python doc/tutorials/scripts/tutorial_03_global_to_fatigue.py
```

The code blocks below are taken from the script and the output blocks are what
it prints. `tests/test_tutorials_03_04.py` runs the script and checks that the
two still agree.

## Imports

```{literalinclude} scripts/tutorial_03_global_to_fatigue.py
:language: python
:start-at: "import csv"
:end-before: "[step-export]"
```

`fibre_fatigue`, `rainflow_cycles`, `wire_stress_field` and
`cable_to_fatigue_layers` are the building blocks of `compute_fatigue`. The
tutorial calls them directly to show the stresses behind the damage.

## 1. A time-history export

The script first writes a section history in the layout of a time-history CSV
export: one header row of named columns, then one row per time sample.

The record is synthetic. `cx.jonswap_series` shapes the curvature with a
{term}`JONSWAP` spectrum (significant wave height 3 m, peak period 9 s) and
scales it to a chosen RMS. The script adds a static sag-bend curvature of
0.05 1/m, an out-of-plane component with 30 % of the in-plane RMS, and a
tension that follows the in-plane curvature. It illustrates the route; it is
not the response of a global model.

```{literalinclude} scripts/tutorial_03_global_to_fatigue.py
:language: python
:start-after: "[step-export]"
:end-before: "[step-read]"
```

<!-- tutorial-output -->
```text
Time (s),x curvature (rad/m),y curvature (rad/m),Effective tension (kN)
0,-0.000958504,-0.0483883,59.2063
0.2,-0.00121411,-0.0488292,59.5659
```

The curvature columns are the components of the {term}`rotation curvature`
vector `t × dt/ds` on the section axes `e1` and `e2`, as global line programs
write them. A line hanging in the vertical `x–z` plane bends about `e2`, so its
sag bend has a negative `y curvature`. The tension is effective tension in kN.

## 2. The reader picks columns by their headings

```{literalinclude} scripts/tutorial_03_global_to_fatigue.py
:language: python
:start-after: "[step-read]"
:end-before: "[step-demo]"
```

<!-- tutorial-output -->
```text
samples 3000, step 0.20 s
tension kind: effective
strain gradient x  +0.0232 to +0.0784 1/m
strain gradient y  -0.0089 to +0.0082 1/m
tension            36.8 to 82.3 kN
```

`read_time_history_csv` selects each column by a piece of its heading:

| Argument | Default | Meaning |
| --- | --- | --- |
| `time_col` | `"Time"` | time [s] |
| `curvature_x_col`, `curvature_y_col` | `"x curvature"`, `"y curvature"` | the two signed curvature components; `curvature_y_col=None` declares a planar record |
| `tension_col` | `"Wall tension"` | axial force |
| `axial_strain_col`, `temperature_col` | `None` | optional direct tensile strain and temperature |
| `curvature_scale_to_1_m` | `1.0` | multiplier to 1/m (`1e3` for a file in 1/mm) |
| `tension_scale_to_N` | `1e3` | multiplier to N (the default reads kN) |
| `curvature_convention` | `"rotation"` | `"rotation"` or `"strain_gradient"` |
| `tension_kind` | inferred | `"true"`, `"effective"` or `"unspecified"` |

A selector matches a heading that contains it, without regard to case, so
`"Effective tension"` finds `Effective tension (kN)`. A selector that matches
no heading, or more than one, stops the read with a `CableSchemaError` that
names the headings. Units are never read from the heading. The scale
arguments convert them, so a file in N needs `tension_scale_to_N=1.0`.

The reader converts the rotation components to the {term}`strain gradient` that
CableXSec uses everywhere:

```text
(g_x, g_y) = (−y curvature, x curvature)
```

A fibre at `(x, y) = r (cos φ, sin φ)` then has the bending strain
`g_x x + g_y y`, and the gradient points to the most tensile fibre. The sag
curvature of the record appears as a positive `g_x` throughout: the tensile
side of the bend is at `φ = 0`, on `e1`, the downward side of the section. A
curvature magnitude column is refused, because it cannot show the reversals
of the bending direction. Section 4.3 of the
[local analysis guide](../local_analysis_guide.md) and Section 2 of the
[global–local guide](../global_local_guide.md) give the sign checks behind
this mapping.

The {term}`tension kind` comes from the heading. `Wall tension` and
`True tension` are read as true tension, `Effective tension` as effective
tension. Any other heading gives the kind `unspecified`, which the fatigue
routes refuse, so declare it with `tension_kind=`. An explicit kind that
contradicts a recognised heading is refused.

### The bundled demonstration history

The repository holds a short history from a global time-domain analysis of the
public 220 kV representative export cable, at the most curved section of its
line (`validation/paper_data/global_demo/`). The file is in CableXSec's own
exchange format, with SI units and strain-gradient columns, so the same reader
needs other selectors, another unit scale and another convention:

```{literalinclude} scripts/tutorial_03_global_to_fatigue.py
:language: python
:start-after: "[step-demo]"
:end-before: "[step-tension]"
```

<!-- tutorial-output -->
```text
demo history: 201 samples, step 0.1 s, largest curvature 0.0349 1/m, tension 25.1 to 25.4 kN
same arrays as read_series_csv: True
```

`read_series_csv` reads the exchange format directly, and both routes give the
same arrays. The demonstration record is 20 s long and nearly static, so the
rest of the tutorial uses the synthetic record.

## 3. The annulus decides which force loads the armour

```{literalinclude} scripts/tutorial_03_global_to_fatigue.py
:language: python
:start-after: "[step-tension]"
:end-before: "[step-fatigue]"
```

<!-- tutorial-output -->
```text
IEA Task 49 / Janocha 2024 66 kV three-core (public template)
armour annulus: flooded
fatigue_from_series: effective tension is not a material force
```

The cable is the public 66 kV three-core design of Janocha et al. (2024) and
IEA Wind Task 49 ([template page](../template_66kv.md)). The wire stress needs
the force that the armour carries. `fatigue_from_series` accepts only true
force, so it refuses the {term}`effective tension` of the export. The right force
depends on the armour {term}`annulus`.

In a flooded annulus, as on this cable, seawater reaches the armour and the
hydrostatic pressure acts on every face of the layers, so the effective
tension loads the armour. Pass it to `compute_fatigue` as `tension_N`, as the
sea-state batch does with `tension_basis="effective"`.

In a sealed annulus, convert first with
`series.with_true_tension(external_pressure_Pa, external_area_m2)`, which
gives `T_e − p_o A_o`, and use `fatigue_from_series`.

## 4. Every wire gets its own damage

```{literalinclude} scripts/tutorial_03_global_to_fatigue.py
:language: python
:start-after: "[step-fatigue]"
:end-before: "[step-recover]"
```

<!-- tutorial-output -->
```text
warnings: DirectionTiedFrictionWarning, MissingClampingStateWarning
inner_armour: 139 wires, damage 7.715e-09 to 1.382e-08, largest on wire 70 at 181.3 deg, fibre 180.00 deg
outer_armour: 145 wires, damage 7.429e-09 to 8.884e-09, largest on wire 73 at 181.2 deg, fibre 180.00 deg
governing: inner_armour, wire at 181.3 deg, fibre 180.00 deg (bulk damage)
```

`compute_fatigue` follows the stick and slip of every wire through the whole
record, with the armour contact pressure following the tension history. It
then rainflow-counts the stress at 32 fixed fibres round each wire and sums
the Miner damage against the layer's {term}`S–N curve` (`bulk_sn` in the cable
file).

`D_bulk_per_fibre[layer]` keeps the damage at every wire and fibre, and
`D_bulk_per_wire[layer]` the largest fibre damage of each wire.
`D_total_per_wire[layer]` adds the contact and termination terms. Both are
zero here: the two armour layers declare `fretting_sn: ""`, bulk fatigue only,
and no termination is requested.

Wire `k` of a layer of `n` wires sits at the azimuth `360 k / n` degrees from
`e1`. Fibre angles are measured round the wire from its outward radial fibre,
so fibre 180° is the inward fibre.

The 66 kV template declares zero {term}`manufacturing clamping` because its source
publishes none, and the run says so with `MissingClampingStateWarning`. The
armour friction then comes from the tension alone, so the friction stresses
and damages sit at their low end (Section 4.4 of the local analysis guide).

`DirectionTiedFrictionWarning` announces a change of default: the record
bends the cable in more than one plane, and a later release will solve such
records with `friction_model="auto"`, the distributed helix friction model,
instead of the direction-tied law used here (Section 9 of the local analysis
guide). Pass `friction_model` explicitly to choose the law and silence it.

```{figure} figures/tutorial_03_damage_circumference.png
:alt: Miner damage of every inner and outer armour wire against its azimuth.
:width: 100%

Damage of every wire over the 600 s record. The dashed line marks the
governing wire.
```

The inner armour damage is largest on the wires in the plane of the sag bend,
near 0° and 180°, and smallest near the neutral axis, at 90° and 270°. The
wire on the concave side, at 181°, governs, at its inward fibre. The outer
armour damage changes little round the section.

## 5. The stresses behind the damage can be recovered

`wire_stress_field` gives, for every wire and sample, the stress of the wire
axis from tension and friction, and the two components of the bending of the
wire about its own axes. `at_fibre(angle)` combines them at any fibre.
`fibre_fatigue` counts every fibre in one pass and returns the damage and the
1 Hz damage-equivalent stress range.

```{literalinclude} scripts/tutorial_03_global_to_fatigue.py
:language: python
:start-after: "[step-recover]"
:end-before: "[step-equivalent]"
```

<!-- tutorial-output -->
```text
inner_armour: same damage as compute_fatigue True, 1 Hz equivalent range 2.18 to 12.39 MPa
outer_armour: same damage as compute_fatigue True, 1 Hz equivalent range 2.11 to 11.09 MPa
```

With the contact pressure history of `armour_contact_histories` and the
reference tension of the first sample, the recovered fields reproduce the
damage of `compute_fatigue` at every wire and fibre. A recovery started this
way can also drop a lead-in from the count; see Step 3 in Section 5 of the
local analysis guide, and the `count_from_sample` argument of
`compute_fatigue`.

## 6. One range stands for the whole history

```{literalinclude} scripts/tutorial_03_global_to_fatigue.py
:language: python
:start-after: "[step-equivalent]"
:end-before: "[step-figures]"
```

<!-- tutorial-output -->
```text
S-N curve saevik_2011_flat_steel: slope m = 4, [Saevik2011] Eq. (89)
stress at the governing fibre: 16.1 to 48.8 MPa
rainflow: 153.0 cycles, largest range 32.73 MPa
equivalent range over the counted cycles 17.44 MPa
equivalent range at 1 Hz, 600 cycles 12.39 MPa
damage from the equivalent range 1.3816e-08, from compute_fatigue 1.3816e-08
```

For an S–N curve of a single slope `m`, the damage-equivalent stress range
over `N_eq` cycles is

$$
\Delta\sigma_\mathrm{eq} = \left(\frac{\sum_i n_i\,\Delta\sigma_i^{\,m}}{N_\mathrm{eq}}\right)^{1/m},
$$

with `n_i` the rainflow counts (half cycles count 0.5) and `Δσ_i` their ranges.
With `N_eq` equal to the counted cycles, it is the constant range that does the
same damage in the same number of cycles. With one cycle per second, it is the
1 Hz range that `fibre_fatigue` reports. `N_eq / N_f(Δσ_eq)` returns the Miner
damage of the fibre.

A curve with a knee or a cut-off has no single slope. The equivalent range
then belongs to one slope only, and the damage must be summed cycle by cycle,
as `compute_fatigue` does.

```{figure} figures/tutorial_03_governing_fibre_stress.png
:alt: Stress history of the governing fibre over the first 120 s.
:width: 100%

Stress at the governing fibre over the first 120 s of the record.
```

## 7. Figures

```{literalinclude} scripts/tutorial_03_global_to_fatigue.py
:language: python
:start-after: "[step-figures]"
```

<!-- tutorial-output -->
```text
wrote tutorial_03_damage_circumference.png
wrote tutorial_03_governing_fibre_stress.png
```

## Where next

A design life combines many sea states. `compute_long_term_fatigue` sums the
damage of a scatter of records at every fixed fibre (Step 5 in Section 5 of
the local analysis guide). For CableDyn runs, `run_sea_state_batch` and the
`cablexsec-batch` command read each section and sum over the sea states, as
the [global–local guide](../global_local_guide.md) describes.

Sections near an end fitting or a clamped bend-stiffener root need the
{term}`termination correction`, given through `termination=` of
`compute_fatigue`.

The contact side of the same armour, with slip, fretting regimes and
critical-plane indicators, is the subject of
[Tutorial 4](04_contact_and_fretting.md). The column layouts that every reader
accepts are on the [file formats](../file_formats.md) page.
