<!-- SPDX-License-Identifier: Apache-2.0 -->

# CableXSec user guide

CableXSec analyses the cross-section of an unbonded, helically armoured cable:
dynamic power cables, umbilicals and flexible pipes with tensile armour. It
gives the section properties for a global analysis, the stresses and capacity
of every component, installation checks, and the fatigue of the armour, the
conductors, screens, tapes and sheaths under load histories from a global
analysis, and the temperatures and frictional heating of an energised cable.
This guide follows a project from a cable definition to its results through
the Python interface; the full function reference is the
[local analysis guide](local_analysis_guide.md).

## 1. Installing

From a source checkout:

```text
pip install -e .
pip install -e ".[fast]"      # optional compiled kernels (numba)
pip install -e ".[contact]"   # optional concentric contact solver (Clarabel)
```

Without the compiled kernels the results agree to round-off, only slower.

## 2. A project, step by step

### 2.1 The cable

Start with one of these:

- **A bundled cable**: published constructions such as the IEA Wind Task 49
  66 kV three-core cable (`cx.iea_task49_66kv_yaml_path()`), CIGRE TB 880
  220 kV (`cx.cigre_tb880_yaml_path()`), the Ménard and Cartraud (2023) cable
  (`cx.menard2023_yaml_path()`) or the Sævik (2011) riser
  (`cx.saevik2011_yaml_path()`). Each states its source. Load one with
  `cx.load_cable(path, geometry_override=None)`.
- **A cable YAML file** of your own ([local analysis guide](local_analysis_guide.md),
  Section 4.1). A definition may write withheld geometry as `null` and supply
  it from a separate YAML passed as `geometry_override`, merged by layer name.

The material and S–N libraries hold cited defaults; `cx.register_material`
and `cx.register_sn_curve` add your own data for the session
([local analysis guide](local_analysis_guide.md), Section 4.2).

Run the preflight check before running a project
(`cablexsec.preflight.preflight(project)`, [project guide](project_guide.md)):
an error stops the analyses that depend on it; a warning is something to
confirm.

### 2.2 Section properties

`cx.compute_stiffness` gives the axial, bending and torsional stiffness for a
global analysis, with the stick and slip bending stiffness of the armour.
`cx.compute_mkappa` gives moment–curvature loops at each tension;
`cx.torque_balance` the torque per tension and the balance of the armour
layers; `cx.design_torque_balance` proposes the lay angle, number or diameter
of the wires of one armour layer that meets a target torque per tension,
beside the present design; `cx.export_bending_law` writes the
moment–curvature law in the formats of global analysis programs
([global–local guide](global_local_guide.md), Section 8).

The initial contact pressure from manufacture sets the friction of the armour.
When the cable declares none, the friction results rest on the tension alone;
the run then warns (`MissingClampingStateWarning`).

### 2.3 Capacity and installation

- **Wire stresses** and **component stresses** (`cx.compute_stress`,
  `cx.compute_component_response`): the stresses of every wire and component
  under tension, bending and pressure.
- **Sheath wall stress** (`cx.compute_shell_stress`): the contributions to the
  wall stress of a metallic sheath (bending at the smooth wall and at the
  corrugation root, axial, thermal and hoop stress) on both sides of the bend,
  against its yield stress. Metallic sheaths are made to yield in bending, so
  the result gives the comparison, not a verdict.
- **Capacity** (`cx.build_capacity_envelope`, `cx.compute_capacity`): the
  envelope of tension and curvature within the limits of every component, with
  the governing component.
- **Compression** (`cx.compression_screen`): the axial compression capacity
  of the armour ([ULS guide](uls_guide.md)).
- **Coiling** and **sheave** (`cx.compute_coiling`, `cx.compute_sheave`):
  installation over a coil, a reel or a sheave, with the minimum bend radius.

### 2.4 Load histories

A `LoadCaseSeries` holds the record of a global analysis at one cable
section. The readers are:

- `cx.read_time_history_csv` for a time-history CSV export, columns
  selected by header;
- `cx.read_riflex_ascii` for a RIFLEX ASCII time history;
- `cx.read_cabledyn_sections` for a CableDyn run
  ([global–local guide](global_local_guide.md));
- `cx.read_series_csv` for a CableXSec series CSV.

The series records which force it holds (true wall tension or effective
tension) and its source. `cx.jonswap_series` makes a synthetic JONSWAP record
for screening.

### 2.5 Fatigue

- **Fatigue of a load history** (`cx.compute_fatigue`,
  `cx.fatigue_from_series`): the damage of every armour wire at fixed fibres
  round the wire, and of the conductors and metallic sheaths with fatigue data
  (`cx.compute_conductor_fatigue`, `cx.compute_shell_fatigue`), with the
  critical wire and fibre.
- **Long-term fatigue** (`cx.compute_long_term_fatigue`): the damage over a
  scatter diagram, one `FatigueLoadCase` per sea state with its probability.
- **Armour fretting**, **core screen fatigue** and **tape fatigue** of a load
  history (`cx.compute_fretting`, `cx.compute_core_screen_fatigue`,
  `cx.compute_tape_fatigue`): slip regime and wear of the armour contacts, the
  wire screen along the core lay, and the strain-life of tapes.
- **Neutral axis under a load history** (`cx.compute_neutral_axes`): where the
  armour sticks and slips as the bending direction turns.

### 2.6 Thermal and electrical

- **Steady temperatures** (`cx.compute_thermal`): the IEC 60287 temperatures
  of the conductor, the armour and the surface at an entered current, with the
  environment of the cable (or the ambient and external resistance entered as
  arguments). It gives the temperatures of that current; it does not search
  for the rating.
- **Frictional heating in bending**
  (`cablexsec.analysis.frictional_heating.simulate_frictional_heating`): an
  energised cable bent cyclically, as in a heated bending test or at a bend
  stiffener. A current (or a conductor temperature held by a heated rig) heats
  the cable, the temperatures set the armour contact pressure, and the
  friction work of bending returns as heat.
- **Thermal fatigue of a load history** (`cx.compute_thermal_fatigue`): the
  armour fatigue at the steady temperatures of a current (or one
  temperature), against the unenergised cable.
- **Relative electrical life** (`cx.compute_electrical_life`): the insulation
  life at an operating temperature against a reference temperature and life.

### 2.7 The project file and the viewer

`cx.Project` keeps the cable, histories, settings and results in one
`.cxproj` file ([project guide](project_guide.md)).
`cablexsec.plotting.build_dashboard` writes a standalone HTML viewer of
stress, moment–curvature, capacity and fatigue results.

## 3. Reading results

- **Provenance.** Every result keeps its `provenance`: method, citations,
  gates and limitations.
- **Warnings.** A run that succeeds but rests on an assumption or leaves the
  validated range says so with a warning (for example `ExtrapolationWarning`,
  `MissingClampingStateWarning`).
- **Stale results.** In a project, a record made before the cable or the
  settings changed is stale (`project.is_stale(record)`); run it again for the
  present model.
- **Files for other programs.** `cx.export_bending_law` writes bending laws
  for global analysis programs (JSON, CSV and a moment–curvature table in kN
  and m).

## 4. Scope

CableXSec is a cross-section program for unbonded, helically armoured
products (the class of DNV Helica). The global response comes from a global
analysis program; CableXSec reads its records. Pressure armour, carcass
collapse and the tubes of umbilicals are outside its present scope.
