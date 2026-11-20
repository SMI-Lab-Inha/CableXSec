<!-- SPDX-License-Identifier: Apache-2.0 -->

# Bending hysteresis

**Goal:** compute the {term}`moment–curvature loop` of the Ménard and Cartraud
(2023) cable, compare it with the loop they measured, split the work into
stored and dissipated parts, and see how the {term}`friction coefficient` and the
{term}`clamping pressure` change the loop. It follows on from
[Tutorial 1](01_first_cable.md).

**Script:**
[`doc/tutorials/scripts/tutorial_02_bending_hysteresis.py`](https://github.com/SMI-Lab-Inha/CableXSec/blob/main/doc/tutorials/scripts/tutorial_02_bending_hysteresis.py)
· **Run time:** a few seconds

The script reads the digitised measurement from `validation/paper_data`, so
run it from a source checkout:

```text
python doc/tutorials/scripts/tutorial_02_bending_hysteresis.py
```

The code blocks below are taken from the script and the output blocks are what
it prints; `tests/test_tutorials_01_02.py` keeps the two in step.

## Imports

```{literalinclude} scripts/tutorial_02_bending_hysteresis.py
:language: python
:start-at: "import sys"
:end-before: "[step-measured]"
```

## 1. The measured loop

Ménard and Cartraud bent a cable specimen in a four-point bench through six
curvature ranges. Range (b) cycles between about −0.1 and +0.1 1/m. The
repository holds its moment–curvature path, extracted from the vector paths of
the published figure, in the order it was recorded
(`validation/paper_data/menard_fig16b_ordered.csv`).

```{literalinclude} scripts/tutorial_02_bending_hysteresis.py
:language: python
:start-after: "[step-measured]"
:end-before: "[step-loop]"
```

<!-- tutorial-output -->
```text
measured points: 589
curvature -0.0974 to 0.0969 1/m
moment    -502 to 526 N m
```

## 2. The section law follows the measured path

`compute_mkappa` drives the {term}`section law` along any curvature history. The
script passes the measured curvature path twice. The first pass builds up the
slip state of the wires from the stress-free section; only the second pass is
compared.

`insert_cycle_crossings` adds a calculation point wherever the path crosses
zero curvature, so the work can be read exactly at the start and end of each
cycle. `original` indexes the measured points within the augmented path.

The friction coefficient 0.2 and the 0.3 MPa pressure of the cable's
`initial_state` are the values the source fixed on range (a) with its own FE
model; the pressure is a tuning parameter of that model. Range (b) played no
part in setting them.

```{literalinclude} scripts/tutorial_02_bending_hysteresis.py
:language: python
:start-after: "[step-loop]"
:end-before: "[step-work]"
```

<!-- tutorial-output -->
```text
model moment -532 to 530 N m
ordered RMSE / measured range: 2.86 %
```

Point by point in the recorded order, the moment error is 2.86 % of the
measured moment range.

## 3. The loop area is the friction work

`loop.work` holds the cumulative stored elastic energy and the cumulative
{term}`friction work` per metre of cable along the history. Between any two
points, the work done on the section is the change of stored energy plus the
friction work in between. Over a closed cycle the stored energy returns to its
value, so the work of the cycle equals the friction work, which is the area of
the loop.

```{literalinclude} scripts/tutorial_02_bending_hysteresis.py
:language: python
:start-after: "[step-work]"
:end-before: "[step-sensitivity]"
```

<!-- tutorial-output -->
```text
stored energy at the cycle crossings: [0.095 0.095 0.095] J/m
largest stored energy: 17.03 J/m
friction work per cycle, model:    [69.8 69.6] J/m
loop area per cycle, measured:     [62.4 62.2] J/m
difference of the means: +11.8 %
```

The stored energy is the same at every zero crossing, so each cycle closes. It
peaks at the curvature extremes.

With the source's 0.3 MPa the calculated friction work per cycle is 11.8 %
above the measured loop area (8.8 % before the armour shear stack was cut at
its support). The gated row BM1-loop of the [validation record](../validation.md)
does not take that pressure: it identifies the clamping on the measured work of
range (a) and predicts range (b), whose work is then 28.8 % below the
measurement, outside the registered 10 % target (FAIL; discussed in the
[benchmark register](../benchmarks.md)).

`loop.W_diss_J_per_m` is the friction work over the whole supplied history,
conditioning pass included. For the loss per cycle, take differences of
`loop.work.friction_J_per_m` between cycle boundaries.

```{figure} figures/tutorial_02_measured_loop.png
:alt: Measured and calculated moment-curvature loop of the Ménard cable, range (b).
:width: 60%

Range (b) of Ménard and Cartraud (2023): measured points and the CableXSec
loop on the same curvature path.
```

## 4. Friction and clamping set the loop width

The width of the loop is set by the {term}`friction stress` the wires can carry,
and that grows with the friction coefficient and with the contact pressure.
The script cycles the cable twice between ±0.1 1/m and evaluates the second,
closed cycle. It varies the friction coefficient through the `mu` argument,
and the clamping through a copy of the cable with another initial pressure
`p0`, made with `model_copy`. The bundled file is not changed.

```{literalinclude} scripts/tutorial_02_bending_hysteresis.py
:language: python
:start-after: "[step-sensitivity]"
:end-before: "[step-figures]"
```

<!-- tutorial-output -->
```text
mu = 0.1        peak   449 N m, friction work  36.5 J/m per cycle, inner slip onset 0.0005 1/m
mu = 0.2        peak   541 N m, friction work  72.7 J/m per cycle, inner slip onset 0.0009 1/m
mu = 0.3        peak   633 N m, friction work 108.7 J/m per cycle, inner slip onset 0.0014 1/m
p0 = 0.15 MPa   peak   449 N m, friction work  36.6 J/m per cycle, inner slip onset 0.0005 1/m
p0 = 0.30 MPa   peak   541 N m, friction work  72.7 J/m per cycle, inner slip onset 0.0009 1/m
p0 = 0.60 MPa   peak   725 N m, friction work 144.0 J/m per cycle, inner slip onset 0.0018 1/m
```

The friction work per cycle is close to proportional to the friction
coefficient and to the pressure, and so is the slip-onset curvature.

Halving the friction coefficient and halving the pressure give nearly the
same loop: the same peak, 449 N m, and 36.5 against 36.6 J/m. The friction stress scales with
`mu sigma_0` and departs from that product only at second order in
`mu sin(alpha)`, so a measured loop constrains the product of friction and
clamping, not the two separately. The
[benchmark register](../benchmarks.md) reports the same for the six Ménard
ranges.

The 0.60 MPa case doubles the pressure of the bounded chain at zero tension,
as a tension of about 40 kN would for the inner armour (Section 3 of
Tutorial 1).

```{literalinclude} scripts/tutorial_02_bending_hysteresis.py
:language: python
:start-after: "[step-figures]"
```

<!-- tutorial-output -->
```text
wrote tutorial_02_measured_loop.png
wrote tutorial_02_sensitivity.png
```

```{figure} figures/tutorial_02_sensitivity.png
:alt: Moment-curvature loops of the Ménard cable for three friction coefficients and three initial pressures.
:width: 100%

Second cycle between ±0.1 1/m. Left: friction coefficient 0.1, 0.2 and 0.3 at
the published 0.3 MPa. Right: initial pressure 0.15, 0.3 and 0.6 MPa at a
friction coefficient of 0.2.
```

## Where next

A bending test rarely imposes a uniform curvature. To compare with a test
through its own arrangement, use `simulate_bending_test` (Section 6.2 of the
[local analysis guide](../local_analysis_guide.md)).

The same section law can be exported as a hysteretic {term}`bending law` for a
global analysis with `export_bending_law` (Section 8 of the
[global–local guide](../global_local_guide.md)).

What the friction does to the wire stresses and to fatigue is the subject of
[Tutorial 3](03_global_to_fatigue.md).
