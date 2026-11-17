<!-- SPDX-License-Identifier: Apache-2.0 -->

# How-to recipes

Each recipe does one job. It states the goal, gives the code, shows what the
code prints and names the assumption that matters most.

The code is the script under `doc/howto/scripts/`, from its `# [recipe]`
marker on. Run a script from a source checkout with
`python doc/howto/scripts/<script>.py`. Every recipe uses bundled cables and
the public data under `tests/data`, and `tests/test_howto_recipes.py` checks
that each script prints exactly the output shown here. The
[user guide](user_guide.md) places these functions in a project workflow.

```{contents}
:local:
:depth: 2
```

## Section properties and contact

### Load sharing and mass properties

**Goal:** the mass and weight in water of a cable, and the share of the
tension each layer carries with the twist held or free.

```{literalinclude} howto/scripts/howto_01_load_sharing_mass.py
:language: python
:start-after: "# [recipe]"
```

<!-- howto-output: howto_01_load_sharing_mass.py -->
```text
outer diameter        184.0 mm
mass                  76.33 kg/m
weight in water       491.7 N/m
  (voids sealed)      481.2 N/m
share of 200 kN, twist held (solved route)
  cu_conductor        0.312
  mdpe_filler         0.006
  inner_armour        0.319
  outer_armour        0.349
  sum of all layers   1.000
share of 200 kN, twist free (solved route)
  cu_conductor        0.301
  mdpe_filler         0.006
  inner_armour        0.357
  outer_armour        0.322
  sum of all layers   1.000
```

Layers below half a percent of the tension are left out of the printout but
counted in the sum. The masses use the densities of the material library. The
shares come from the solved radial section (`route` is `"solved"`), or from
equal axial strain of every layer (`"fixed_radius"`) where that solver does not
accept the cable.

### Radial contact state

**Goal:** the contact pressures, open gaps and wire stresses of the solved
radial section, from a cable definition or from layers given directly.

```{literalinclude} howto/scripts/howto_02_contact_state.py
:language: python
:start-after: "# [recipe]"
```

<!-- howto-output: howto_02_contact_state.py -->
```text
T 100 kN, sealed p_ext 0 MPa: strain 114 ue, open gaps 1
  inner_armour  P1+P2  0.062 MPa, wire stress   21.0 MPa
  outer_armour  P1+P2  0.016 MPa, wire stress   21.7 MPa
T 100 kN, sealed p_ext 2 MPa: strain 177 ue, open gaps 0
  inner_armour  P1+P2  4.329 MPa, wire stress   27.9 MPa
  outer_armour  P1+P2  4.107 MPa, wire stress   31.0 MPa
single-core cable at 200 kN: 7 contacts closed
  r =   9.8 mm:  2.110 MPa
  r =  10.8 mm:  2.074 MPa
  r =  18.8 mm:  1.961 MPa
  r =  19.8 mm:  1.955 MPa
  r =  20.5 mm:  8.622 MPa
  r =  22.5 mm:  8.687 MPa
  r =  26.5 mm:  3.798 MPa
  r =  30.5 mm:  0.000 MPa
twist held: strain 365 ue, twist 0.00 mrad/m, torque -196.6 N m
  contact 1.038, 0.435, 0.437, 0.000 MPa
twist free: strain 378 ue, twist 6.91 mrad/m, torque 0.0 N m
  contact 1.062, 0.358, 0.361, 0.000 MPa
```

`solve_cable_radial_state` takes trefoil cores and the declared manufacturing
clamping. Here the external pressure acts on the outer sheath with the
{term}`annulus` sealed; `flooded_pressure_Pa` floods it. `solve_contact_qp` and
`solve_concentric_section` accept concentric layers only and leave out the
{term}`manufacturing clamping`. All three solve a state of tension, twist,
pressure and temperature without bending, as the
[radial model](radial_model.md) notes explain.

### Neutral axes under a turning curvature

**Goal:** the {term}`bending direction`, moment direction, stress
{term}`neutral axis` and stick zone of the armour as the curvature vector turns.

```{literalinclude} howto/scripts/howto_03_neutral_axes.py
:language: python
:start-after: "# [recipe]"
```

<!-- howto-output: howto_03_neutral_axes.py -->
```text
strain axis offset -3.82 mm
 kappa  theta    psi  delta  psi_L  stick
 0.005    0.0    0.0    0.0    0.0    360
 0.015    0.0    0.0    0.0    0.0    242
 0.025    0.0    0.0    0.0    0.0      0
 0.050    0.0    0.0    0.0    0.0      0
 0.050   30.0   34.0    4.0   43.0    145
 0.050  120.0  124.1    4.1  133.4    145
 0.050  180.0 -175.9    4.1 -166.6    145
```

Angles are in degrees from the section x axis. `theta` is the bending
direction, `psi` the moment direction and `delta` the lead of the moment.
`psi_L` is the direction of the outer armour's first stress harmonic; its
stress neutral axis is perpendicular to it. `stick` is the arc of wires in
stick.

The stick zone is given for the direction-tied friction law only. Under a
turning curvature the stress neutral axis depends on the friction law
(`friction_model`).

## Stress and installation

### Stress over the whole cross-section

**Goal:** the longitudinal stress at points over every component at one
tension and curvature, for a section stress map.

```{literalinclude} howto/scripts/howto_04_section_stress.py
:language: python
:start-after: "# [recipe]"
:end-before: "# [figure]"
```

<!-- howto-output: howto_04_section_stress.py -->
```text
core state: frictional
layer              kind      min MPa  max MPa    use  limit
cu_conductor       annulus     -34.9    107.1  0.893  yielding
conductor_screen   annulus      -0.2      0.3      -  no limit given
xlpe_insulation    annulus      -0.3      0.4      -  no limit given
insulation_screen  annulus      -0.3      0.4      -  no limit given
cu_screen          annulus    -249.7    309.0      -  follows the section
hdpe_core_sheath   annulus      -2.1      2.5  0.037  strain limit
pp_filler          annulus      -7.0      7.6      -  no limit given
hdpe_bedding       annulus      -4.3      4.6  0.068  strain limit
inner_armour       wires       -71.4    152.2  0.221  yielding
outer_armour       wires       -25.7     97.8  0.142  yielding
hdpe_outer_sheath  annulus      -5.5      5.9  0.086  strain limit
inner_armour: 61 wires, centre line -38.9 to 119.7 MPa
outer_armour: 67 wires, centre line 7.9 to 65.3 MPa
```

```{figure} howto/figures/howto_04_section_stress.png
:alt: Longitudinal stress at points over the three-core cable section
:width: 420px

Longitudinal stress over the section; the plotting code is in the script.
```

The fields reproduce the extremes and utilisations of
`compute_component_response` at the same inputs, in one {term}`core slip state`
(`core_state`). The copper wire screen is outside the load path of the model.
Its stress is its modulus times the strain it follows.

### Coiling into a tank

**Goal:** the armour stresses, {term}`bird-caging` and first-yield checks of a
cable coiled into a static tank with one twist per turn.

```{literalinclude} howto/scripts/howto_05_coiling.py
:language: python
:start-after: "# [recipe]"
```

<!-- howto-output: howto_05_coiling.py -->
```text
D = 14.4 m: kappa = twist = 0.1389 1/m, torque 14.79 kN m
  armour wire centre  -244.0 to -205.4 MPa, surface  -299.8 to -150.6 MPa
  bird-caging capacity 231.7 MPa, buckling 1.053, yield 0.750
  governing: armour (radial), utilisation 1.053, elastic True
D =  8.0 m: kappa = twist = 0.2500 1/m, torque 26.62 kN m
  armour wire centre  -439.1 to -369.7 MPa, surface  -539.7 to -271.1 MPa
  bird-caging capacity 231.7 MPa, buckling 1.895, yield 1.349
  governing: armour (radial), utilisation 1.895, elastic False
```

The check is elastic. When the wires yield (`elastic False`) the stresses are
not capped, and the buckling check no longer applies. Lateral buckling of the
armour is reported (`sigma_lateral_Pa`) but not counted in the utilisation.

## Fatigue

### Fatigue utilities on a load record

**Goal:** armour damage at fixed fibres, strain-life damage of a copper fibre,
a {term}`Gumbel fit` of block maxima and the slip of the wires along their helix,
all from one record.

```{literalinclude} howto/scripts/howto_06_fatigue_utilities.py
:language: python
:start-after: "# [recipe]"
```

<!-- howto-output: howto_06_fatigue_utilities.py -->
```text
armour: critical layer inner_armour, wire at 0 deg, fibre at 0 deg
  inner_armour  damage in 600 s 5.859e-06
  outer_armour  damage in 600 s 2.245e-06
copper fibre: 122 cycles, max amplitude 1568 ue, damage 3.647e-05
curvature: mean 60 s maximum 0.0479 1/m, 1 h most probable 0.0761, 90 % fractile 0.0941 1/m
inner_armour  slip range 0.0 um, slipping share 0.00
outer_armour  slip range 284.7 um, slipping share 0.26
```

`jonswap_series` makes a synthetic screening record. `fatigue_from_series`
refuses effective or undeclared tension. The `copper_poon2022` constants are a
fit to copper conductor wire, and the fibre strain here is bending alone.

### SWT critical plane of a multiaxial cycle

**Goal:** the Smith–Watson–Topper {term}`critical plane` and life of one closed
stress and strain cycle at a material point.

```{literalinclude} howto/scripts/howto_07_swt_critical_plane.py
:language: python
:start-after: "# [recipe]"
```

<!-- howto-output: howto_07_swt_critical_plane.py -->
```text
critical plane normal at 16 deg to the wire axis
sampled SWT 75.3 kPa
life 1.189e+06 cycles
plane normal along the axis only: life 1.932e+06 cycles
```

The critical plane is the largest of the supplied normals, so refine the
angles and the time samples until it settles. The record must be one closed,
repeated cycle. Variable-amplitude histories are not counted.

## Thermal and electrical

### Frictional heating in cyclic bending

**Goal:** the temperatures of an energised cable bent cyclically, with and
without the heat of armour friction.

```{literalinclude} howto/scripts/howto_08_frictional_heating.py
:language: python
:start-after: "# [recipe]"
```

<!-- howto-output: howto_08_frictional_heating.py -->
```text
coupling none
  friction heat 0.00 W/m at start, 0.00 W/m at 6 h
  after 6 h: conductor 30.3 C, outer armour 24.7 C
coupling two_way
  friction heat 7.89 W/m at start, 11.62 W/m at 6 h
  after 6 h: conductor 32.1 C, outer armour 26.6 C
```

This cable declares no manufacturing clamping, so the friction rests on the
tension alone and the run warns with `MissingClampingStateWarning`. The heat is
the sectional value, away from clamps and terminations. `anchoring_factor` in
`CyclicBending` raises it near them.

### Relative insulation life

**Goal:** the {term}`electro-thermal ageing` factor of the insulation at operating
temperatures.

```{literalinclude} howto/scripts/howto_09_electrical_life.py
:language: python
:start-after: "# [recipe]"
```

<!-- howto-output: howto_09_electrical_life.py -->
```text
T = 50.0 C: field 4.23 kV/mm, L/L0 = 1.11e+03
T = 70.0 C: field 4.23 kV/mm, L/L0 = 103
T = 90.0 C: field 4.23 kV/mm, L/L0 = 12.4
```

The result is a screening factor of the Mazzanti (2007) model against the
reference at the threshold field and `T0_C`. It is not a calibrated lifetime.
The field of this cable is below the threshold of 5 kV/mm, so the factor
exceeds one even at the reference temperature.

## Global analysis results

### Reading CableDyn and RIFLEX results

**Goal:** load series at named sections of a CableDyn run and from a RIFLEX
ASCII time history.

```{literalinclude} howto/scripts/howto_10_read_global_results.py
:language: python
:start-after: "# [recipe]"
```

<!-- howto-output: howto_10_read_global_results.py -->
```text
curvature from tangent channels
hangoff  201 samples, dt 0.1 s, Te mean 9.28 kN, true mean 6.40 kN, |kappa| max 0.0084 1/m
sag      201 samples, dt 0.1 s, Te mean 1.75 kN, true mean -11.20 kN, |kappa| max 0.0984 1/m
riflex   240 samples, 59.8 s, effective tension 65.25 kN mean, kappa_x -0.0155 to 0.0156 1/m
```

The true wall force is the {term}`effective tension` less the hydrostatic pressure
times the outer area. At the sag bend it is compressive while the effective
tension is positive. RIFLEX columns are positional, so declare the tension
kind and the curvature convention, and convert effective tension with
`with_true_tension` before fatigue. The
[global–local guide](global_local_guide.md) covers both readers.
