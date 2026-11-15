<!-- SPDX-License-Identifier: Apache-2.0 -->

# Contact loading and fretting

Armour wires press on each other and on the polymer layers beside them, and
they slip along their helices when the cable bends.

**Goal:** form the contact pressures of the armour over a tension history,
solve the Hertz contact where the wires of two armour layers cross, follow the
slip of every wire round the section, classify every slip cycle into a
{term}`fretting regime`, and recover a Smith–Watson–Topper (SWT) critical-plane
indicator next to a {term}`crossing`.

**Script:**
[`doc/tutorials/scripts/tutorial_04_contact_and_fretting.py`](https://github.com/SMI-Lab-Inha/CableXSec/blob/main/doc/tutorials/scripts/tutorial_04_contact_and_fretting.py)
· **Run time:** about ten seconds

Run it from a source checkout:

```text
python doc/tutorials/scripts/tutorial_04_contact_and_fretting.py
```

The code blocks below are taken from the script and the output blocks are what
it prints. `tests/test_tutorials_03_04.py` runs the script and checks that the
two still agree.

## Imports

```{literalinclude} scripts/tutorial_04_contact_and_fretting.py
:language: python
:start-at: "import math"
:end-before: "[step-cable]"
```

## 1. The two armour layers cross each other

```{literalinclude} scripts/tutorial_04_contact_and_fretting.py
:language: python
:start-after: "[step-cable]"
:end-before: "[step-history]"
```

<!-- tutorial-output -->
```text
Three-core export 220 kV (public representative)
inner_armour: 110 wires, d = 6.0 mm, R = 118.0 mm, lay 18 deg right hand, P0 = 3.0 MPa (representative)
outer_armour: 118 wires, d = 6.0 mm, R = 124.0 mm, lay 18 deg left hand, P0 = 3.0 MPa (representative)
```

The bundled 220 kV three-core export cable is a public representative
template, not a qualified design. Its two armour layers are laid in opposite
directions and touch: the inner wires reach 121 mm, where the outer wires
start. Each layer declares a {term}`manufacturing clamping` `P0` of 3 MPa, the
radial pressure jump across the layer. Tutorial 3 used a cable whose armour
layers are separated by a bedding. Here the wires cross each other directly.

## 2. The load history

```{literalinclude} scripts/tutorial_04_contact_and_fretting.py
:language: python
:start-after: "[step-history]"
:end-before: "[step-pressure]"
```

<!-- tutorial-output -->
```text
record: 1500 samples, curvature 0.0117 to 0.0295 1/m, tension 120 to 180 kN
```

The record is synthetic, as in Tutorial 3: 300 s of JONSWAP-shaped biaxial
curvature about a static curvature of 0.02 1/m, with a tension that follows
the in-plane curvature. The template does not declare the state of its armour
annulus, so the record is generated directly as true wall tension.

## 3. Clamping accumulates from the outside in

```{literalinclude} scripts/tutorial_04_contact_and_fretting.py
:language: python
:start-after: "[step-pressure]"
:end-before: "[step-hertz]"
```

<!-- tutorial-output -->
```text
inner_armour: P1+P2 9.151 to 9.226 MPa, crossing line load 21.08 to 21.25 kN/m
outer_armour: P1+P2 3.050 to 3.075 MPa, crossing line load 19.65 to 19.81 kN/m
```

`armour_contact_histories` gives, for every armour layer and sample, the
friction-setting pressure `P1+P2` and the line load on the crossed-wire
interface. `P1+P2` is the sum of the pressures on the inner and outer faces of
the layer, referred to the helix radius. These are the histories
`compute_fatigue` uses when it is given a tension history.

The clamping jumps accumulate from the outside in. The interface under the
outer armour carries `P0` and the interface under the inner armour `2 P0`, so
`P1+P2` is about `3 P0` for the inner layer and `P0` for the outer layer, plus
the increment from the tension. Over this record the pressures change by less
than 0.1 MPa.

## 4. Crossing wires meet in a Hertz contact

```{literalinclude} scripts/tutorial_04_contact_and_fretting.py
:language: python
:start-after: "[step-hertz]"
:end-before: "[step-fretting]"
```

<!-- tutorial-output -->
```text
crossing with outer_armour: angle 36 deg, 100.9 crossings per metre, 209.7 N per crossing
Hertz ellipse a = 374.0 um, b = 86.0 um, p0 = 3112 MPa, K_t = 34.1 N/um
peak von Mises 1835 MPa at depth 60.2 um; wire yield stress 690 MPa
```

Wires of opposite lay cross at the sum of their lay angles. Each inner wire
meets the outer wires about 101 times per metre, so the interface line load at
the mean tension gives 210 N per crossing.

`crossed_cylinder_curvatures` turns the wire diameters and the crossing angle
into the two principal curvatures of the gap. `solve_contact` gives the
elliptical {term}`Hertz contact`: the semi-axes, the peak pressure `p0` and the
tangential stiffness `K_t` used by the Mindlin slip model.
`compute_hertz_axis_stress` resolves the elastic stress on the axis of the
contact.

The elastic peak pressure and the peak von Mises stress lie well above the
yield stress of the wire material in the library. A crossing loaded this way
yields locally and flattens. The elastic Hertz values indicate the contact
loading; they are not the stresses of the deformed contact.

## 5. Every slip cycle has a fretting regime

```{literalinclude} scripts/tutorial_04_contact_and_fretting.py
:language: python
:start-after: "[step-fretting]"
:end-before: "[step-slip]"
```

<!-- tutorial-output -->
```text
contact loads at 150 kN; extreme fibre of the dynamic bending at -4.0 deg
inner_armour/outer_face on outer_armour: mu 0.12, load 209.7 N, p0 3112 MPa, transition amplitude 1.11 um
  largest slip amplitude 364 um; cycles in stick 0%, partial slip 16%, gross slip 84%
  damage per year at the contact fibre 0.090 at 1 deg from the extreme fibre; largest wear at 88 deg
outer_armour/inner_face on inner_armour: mu 0.12, load 209.7 N, p0 3112 MPa, transition amplitude 1.11 um
  largest slip amplitude 402 um; cycles in stick 0%, partial slip 15%, gross slip 85%
  damage per year at the contact fibre 0.076 at 2 deg from the extreme fibre; largest wear at 88 deg
outer_armour/outer_face: no normal load on this face
```

`compute_fretting` assesses both radial faces of every armour layer, keyed
`"<layer>/<face>"`. A face against an opposite-laid armour layer is a Mindlin
point contact at each crossing, with the Hertz contact of step 4. A face
against a polymer layer is a {term}`line contact` on the support stiffness of the
section. The normal loads are those at one reference tension, by default the
mean of the record.

Each wire's slip history is rainflow-counted, and every cycle is classified by
its amplitude against the {term}`transition amplitude`
`δ_t = 3 μ P / (2 K_t)`, at which the tangential load reaches `μ P`. Up to it
the cycle is partial slip, above it {term}`gross slip`, and without slip or
without load it is stick (Fouvry et al. 1995). The dissipated energy of each
cycle gives the wear. The stress cycles at the contact fibre are counted again
with the regime knock-down of the wire material, to give the damage per year
of this loading.

The outer face of the outer armour carries no load. The clamping jumps
accumulate from the outside in, so nothing presses on the outermost face.

The damage and the wear peak in different places. The damage at the contact
fibre peaks at the extreme fibre of the dynamic bending, where the stress
range is largest. The wear peaks near the neutral axis, 90° away, where the
slip is largest.

`FrettingFace` keeps every per-wire array: cycles and energy per regime,
largest amplitude, wear depth rate, and damage with and without the
knock-down. Its wear depth is computed on the unworn contact width, so it is
the rate at the start of wear. `fretting.provenance` lists the assumed
coefficients and the other limitations of the assessment.

## 6. Friction holds the wires back

`compute_fretting` takes the kinematic slip of each wire, the slip of a wire
free to slide along its helix: `r² cos²α / sin α` times the curvature change
at the neutral axis. Friction holds the wires back. `kinematic_history`
follows the friction law of the stress route instead, with the contact
pressure history of step 3:

```{literalinclude} scripts/tutorial_04_contact_and_fretting.py
:language: python
:start-after: "[step-slip]"
:end-before: "[step-index]"
```

<!-- tutorial-output -->
```text
inner_armour: full-slip amplitude up to 364 um, friction law up to 5.8 um, stations slipping 0.2% of the time
outer_armour: full-slip amplitude up to 402 um, friction law up to 136.1 um, stations slipping 6.9% of the time
```

```{figure} figures/tutorial_04_slip_amplitude.png
:alt: Slip amplitude of the inner and outer armour wires against azimuth, for full slip and for the friction law.
:width: 100%

Half the range of the axial slip of every wire over the record. Dashed:
kinematic slip of free wires, as used by `compute_fretting`. Solid: the
friction law with the contact pressure history.
```

The inner armour, with `P1+P2` of about 9 MPa, barely slips at these
curvatures. The outer armour, with about 3 MPa, slips over most of its
circumference, most near the neutral axis. The kinematic slip is therefore an
upper estimate of the {term}`slip amplitude`, and the regime shares and wear of
step 5 are upper estimates wherever friction holds the wires.

## 7. The shear–slip product ranks wires

```{literalinclude} scripts/tutorial_04_contact_and_fretting.py
:language: python
:start-after: "[step-index]"
:end-before: "[step-swt]"
```

<!-- tutorial-output -->
```text
shear-slip product mu p0 delta*: largest 136 kN/m at 88 deg from the extreme fibre
```

The {term}`shear–slip product` `Φ = μ p0 δ*` combines the nominal frictional shear
and the slip amplitude. It ranks wires and compares designs. It is not a
fretting life.

Nor is it the {term}`Ruiz parameter` (Ruiz et al. 1984), which is formed from the
local surface shear traction and slip and, in its second form, the local
tangential stress. CableXSec 0.1.0 does not compute Ruiz parameters.

## 8. The SWT critical plane next to a crossing

The SWT indicator `σ_max ε_a` is evaluated on many planes through one material
point over one closed load cycle. The plane with the largest value is the
sampled {term}`critical plane`. CableXSec gives the two parts of such a
calculation. `compute_hertz_subsurface_stress` integrates prescribed
Hertz-shaped normal and shear tractions at points below the surface, and
`compute_swt_cycle` finds the sampled critical plane of coincident stress and
strain histories.

```{literalinclude} scripts/tutorial_04_contact_and_fretting.py
:language: python
:start-after: "[step-swt]"
:end-before: "[step-figures]"
```

<!-- tutorial-output -->
```text
wire 109 (gross slip): largest bulk cycle 119.2 MPa about 147.9 MPa
SWT of the bulk cycle alone 0.0587 MPa
x = +0.0 a, z = 0.5 b: SWT 0.0000 MPa
x = -1.0 a, z = 0.5 b: SWT 0.1045 MPa
x = -1.1 a, z = 0.5 b: SWT 0.1361 MPa
x = -1.1 a, z = 0.05 b: SWT 0.4116 MPa, plane normal 8 deg from the wire axis
x = -1.1 a, z = 0.10 b: SWT 0.3710 MPa, plane normal 8 deg from the wire axis
x = -1.1 a, z = 0.20 b: SWT 0.2922 MPa, plane normal 8 deg from the wire axis
x = -1.1 a, z = 0.30 b: SWT 0.2255 MPa, plane normal 8 deg from the wire axis
x = -1.1 a, z = 0.50 b: SWT 0.1361 MPa, plane normal 8 deg from the wire axis
x = -1.1 a, z = 0.75 b: SWT 0.0868 MPa, plane normal 13 deg from the wire axis
x = -1.1 a, z = 1.00 b: SWT 0.0668 MPa, plane normal 10 deg from the wire axis
```

The script builds one cycle at the crossing of the inner-armour wire with the
largest contact-fibre damage, which is in gross slip. The cycle has three
parts.

The first is the bulk stress of the wire at its outer fibre, cycling over the
largest rainflow range of the record, along the wire axis. For two equal
crossed wires the major axis of the contact ellipse, `x`, bisects the acute
angle between the wire axes, so the wire axis lies at half the crossing angle,
18°, from it.

The second is the Hertz pressure of step 4, held fixed. The third is a
Hertz-shaped shear traction `μ p` along the wire axis that reverses over the
cycle in phase with the bulk stress: the traction of gross sliding.

These are stated simplifications. The footprint does not change with the
load, the shear traction is the sliding field and not a
{term}`Cattaneo–Mindlin` partial-slip distribution, and the material is elastic.

Under the middle of the contact the compressive Hertz field leaves no plane in
tension, and the indicator is zero. At the edge and just outside it, behind
the traction, the shear traction adds tension to the wire stress. At
`x = −1.1 a` the indicator near the surface is about seven times that of the
bulk cycle alone, and it falls towards the bulk value at a depth of one
semi-axis `b`. The critical plane lies nearly across the wire.

Points nearer the surface at the edge itself need a higher quadrature order.
`compute_hertz_subsurface_stress` raises `CableSolverError` when it cannot
converge; it never returns an unconverged stress.

`compute_swt_cycle` also returns a formal life from a strain-life law
(`SWTLifeConstants`). The SWT constants registered in CableXSec 0.1.0
(`sn_curves.SWT_LIFE`) are those of copper, and none are for armour steel, so
the script passes a placeholder law and prints no life. With constants from a
source, `result.life_cycles` gives it.

```{figure} figures/tutorial_04_swt_depth.png
:alt: SWT indicator against depth just outside the edge of the crossing contact, with the value of the bulk cycle alone.
:width: 100%

SWT indicator at `x = −1.1 a` against depth, and the indicator of the bulk
stress cycle alone.
```

## 9. Figures

```{literalinclude} scripts/tutorial_04_contact_and_fretting.py
:language: python
:start-after: "[step-figures]"
```

<!-- tutorial-output -->
```text
wrote tutorial_04_slip_amplitude.png
wrote tutorial_04_swt_depth.png
```

## Where next

The fretting options (wear coefficients, knock-down laws, section loss and
occurrence fractions) and the trellis fretting of stranded conductors are in
Sections 10 and 11 of the [local analysis guide](../local_analysis_guide.md).
Step 4 in its Section 5 writes out the crossing calculation of step 4 with the
Mindlin slip history of each wire.

`examples/hertz_axis_stress.py`, `examples/prescribed_contact_swt.py` and
`examples/critical_plane_swt.py` check the contact stress and critical-plane
routines on their own.
