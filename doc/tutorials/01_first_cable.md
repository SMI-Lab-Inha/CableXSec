<!-- SPDX-License-Identifier: Apache-2.0 -->

# Your first cable

**Goal:** load two bundled published cables, list their layers, compute their
axial, torsional and bending stiffness with the full-slip and no-slip bounds,
and see how tension changes the armour {term}`contact pressure` and the curvature
at which the armour starts to slip.

**Script:**
[`doc/tutorials/scripts/tutorial_01_first_cable.py`](https://github.com/SMI-Lab-Inha/CableXSec/blob/main/doc/tutorials/scripts/tutorial_01_first_cable.py)
· **Run time:** a few seconds

Run it from a source checkout:

```text
python doc/tutorials/scripts/tutorial_01_first_cable.py
```

The code blocks below are taken from the script and the output blocks are what
it prints. `tests/test_tutorials_01_02.py` runs the script and checks that the
two still agree.

## Imports

```{literalinclude} scripts/tutorial_01_first_cable.py
:language: python
:start-at: "import sys"
:end-before: "[step-load]"
```

The slip-curvature functions and the solved radial state come from the
mechanics modules. Everything else is on the top-level `cablexsec` namespace.

## 1. A cable is a list of layers

The cable of Ménard and Cartraud (2023) is a three-core dynamic cable with two
contra-wound layers of round armour wires. Every value in its definition is
taken from that source, and the YAML file cites it layer by layer
(`cablexsec/cables/menard2023.yaml`).

```{literalinclude} scripts/tutorial_01_first_cable.py
:language: python
:start-after: "[step-load]"
:end-before: "[step-stiffness]"
```

<!-- tutorial-output -->
```text
Ménard 2023 reference cable
source: Ménard & Cartraud, Marine Structures 91 (2023) 103465
layer              type                material                 geometry
cu_conductor       cylindrical_offset  copper_menard2023        3 cores at 17.7 mm, r = 0.0-4.0 mm
xlpe_insulation    cylindrical_offset  xlpe_menard2023          3 cores at 17.7 mm, r = 4.0-11.3 mm
power_core_shield  cylindrical_offset  hdpe_menard2023          3 cores at 17.7 mm, r = 11.3-15.3 mm
inner_sheath       cylindrical         hdpe_menard2023          r = 32.9-37.4 mm
inner_armour       helical             galvanised_steel_armour  55 wires, d = 4.0 mm, lay 20.1 deg, R = 39.4 mm
bedding            cylindrical         pfr_menard2023           r = 41.4-42.0 mm
outer_armour       helical             galvanised_steel_armour  63 wires, d = 4.0 mm, lay 19.6 deg, R = 44.0 mm
outer_sheath       cylindrical         hdpe_menard2023          r = 46.0-50.5 mm
```

The three power cores are `cylindrical_offset` layers: one definition places
three identical cores on the lay radius. The armour layers are `helical`, with
the wire count, wire diameter, {term}`lay angle` and {term}`helix radius` `R`.
Lengths in a cable file are in metres; the script prints millimetres. Every
layer type and key is described on the [file formats](../file_formats.md)
page.

## 2. The bending stiffness has two bounds

```{literalinclude} scripts/tutorial_01_first_cable.py
:language: python
:start-after: "[step-stiffness]"
:end-before: "[step-contact]"
```

<!-- tutorial-output -->
```text
EA, twist fixed         280.2 MN
EA, twist free          276.1 MN
GJ (armour lay)          59.2 kN m^2
tension-twist K_et      494.1 kN m
EI full slip             3.57 kN m^2
EI no slip (xi)         59.14 kN m^2
EI plane section       234.18 kN m^2
xi inner_armour         0.208
xi outer_armour         0.248
```

`ea_fixed_N` holds the ends against twist. `ea_free_N` lets them turn, which
removes the tension–twist coupling `K_et` of the helical layers. `gj_Nm2` is
the torsional stiffness that the helical lay gives under extension; the
independent torsion of the components is not included.

`ei_min_Nm2` is the full-slip bound: every wire and core bends about its own
axis. It is within 0.6 % of the analytical 3.593 kN m² of the source.
`ei_max_Nm2` is the no-slip bound: the wires stick and follow the bending of
the section, and their contribution is reduced by the {term}`shear coefficient` ξ
of the polymer layers that carry the wire forces (Yun et al. 2020).
`ei_plane_section_Nm2` is the same assembly with ξ = 1.

The bending stiffness of a slipping cable spans a factor of about 17 between
the two bounds. Where it sits depends on the curvature, the history and the
contact pressure. That is why CableXSec computes the bending response as a
hysteretic law and not as one number.

## 3. Tension raises the contact pressure

Friction between the armour wires and their neighbours needs a normal
pressure. The Ménard cable declares no {term}`manufacturing clamping` per layer.
It carries instead the uniform external pressure of 0.3 MPa that the source
calibrated against its bending test, labelled as a published calibration.
Tension adds to this pressure, because a helical layer under tension presses
inwards.

```{literalinclude} scripts/tutorial_01_first_cable.py
:language: python
:start-after: "[step-contact]"
:end-before: "[step-template]"
```

<!-- tutorial-output -->
```text
inner_armour: not declared; outer_armour: not declared; initial external pressure 300 kPa (published_calibration)
T [kN] layer         P bounded  P solved  k onset   k full
     0 inner_armour      0.600     0.448   0.0030   0.0051
     0 outer_armour      0.600     0.428   0.0025   0.0042
    50 inner_armour      1.412     0.882   0.0072   0.0119
    50 outer_armour      0.859     0.337   0.0036   0.0060
   100 inner_armour      2.223     1.625   0.0113   0.0187
   100 outer_armour      1.117     0.557   0.0047   0.0078
```

The two pressure columns are the armour contact pressure `P1 + P2` on both
wire faces, in MPa, from two contact models. The last two columns are the
curvatures, in 1/m, at which the first and the last wire of the layer start to
slip in the bounded model.

The {term}`bounded chain` (`contact_model="bounded"`) is the default of the
bending, stress and fatigue functions. It is a fixed-radius chain that passes
the pressure from layer to layer, so it carries the 0.3 MPa to both faces of
each armour layer: 0.600 MPa at zero tension.

The solved radial state (`solve_cable_radial_states`, or
`contact_model="solved"` in the analysis functions) is the elastic contact
problem of the whole section, the three cores included. It gives lower
pressures on this cable, and the outer sheath separates from the outer armour
between 50 and 60 kN. The [solved radial contact state](../radial_model.md)
notes compare the two models.

`k onset` is the curvature at which the first wire, on the {term}`neutral axis` of
bending, starts to slip. `k full` is the curvature at which the last wire, on
the extreme fibre, slips. Both scale with the contact pressure, so a higher
tension keeps the armour in stick to a larger curvature.

## 4. Without published clamping, only tension clamps

The {term}`IEA Wind Task 49 66 kV template` is a three-core inter-array cable
built from published sources ([66 kV template](../template_66kv.md)). Its
source does not state the manufacturing clamping, so the template declares it
as zero with the basis `not_published`.

```{literalinclude} scripts/tutorial_01_first_cable.py
:language: python
:start-after: "[step-template]"
:end-before: "[step-figure]"
```

<!-- tutorial-output -->
```text
IEA Task 49 / Janocha 2024 66 kV three-core (public template)
EA, twist fixed         636.8 MN
EI full slip            20.98 kN m^2
EI no slip (xi)       1434.40 kN m^2
inner_armour: winding stress 0 MPa (not_published); outer_armour: winding stress 0 MPa (not_published)
100 kN inner_armour  P = 0.086 MPa, slip onset 0.00023 1/m
100 kN outer_armour  P = 0.023 MPa, slip onset 0.00007 1/m
```

With zero clamping the only contact pressure is the one tension creates, and
the armour slips from a curvature below 0.0003 1/m even at 100 kN. The run
also writes a `MissingClampingStateWarning` to the standard error stream. The
friction results of such a cable rest on tension-induced contact alone; they
are sensitivities, not cable properties. Section 4.4 of the
[local analysis guide](../local_analysis_guide.md) shows how to declare a
clamping state with its basis and source.

## 5. Figure

```{literalinclude} scripts/tutorial_01_first_cable.py
:language: python
:start-after: "[step-figure]"
```

<!-- tutorial-output -->
```text
wrote tutorial_01_first_cable.png
```

```{figure} figures/tutorial_01_first_cable.png
:alt: Armour contact pressure against tension, and first-loading moment against curvature of the Ménard cable at 0 and 100 kN.
:width: 100%

Ménard and Cartraud (2023) cable. Left: armour contact pressure against
tension from the bounded chain (solid) and the solved radial state (dashed).
Right: moment on first loading at 0 and 100 kN with the full-slip and no-slip
slopes; the dots mark the slip onset of the two armour layers.
```

On first loading the moment follows the no-slip slope up to the slip onset of
the armour layers. It bends over while the {term}`slip front` spreads from the
neutral axis to the extreme fibres, and then rises with the full-slip slope.
At 100 kN the higher contact pressure delays slip and raises the moment that
friction carries. [Tutorial 2](02_bending_hysteresis.md) follows the same law
through load reversals.
