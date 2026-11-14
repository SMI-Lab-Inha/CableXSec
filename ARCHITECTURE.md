<!-- SPDX-License-Identifier: Apache-2.0 -->

# CableXSec architecture

This document describes how the package is organised.

## Purpose and boundary

CableXSec is a component-resolved local (cross-section) mechanics library for
dynamic power cables. It takes a validated cable definition and a section load
history from a global analysis (curvature components, tension, temperature,
optionally axial strain and pressure) and returns stiffness, moment–curvature
response, wire- and component-resolved stresses, contact quantities and
fatigue damage at fixed material points.

It is not a global dynamic solver or a finite-element code. Global
histories arrive through readers or user arrays; the only outbound coupling is
a cross-section tangent stiffness for a rod solver (`coupling.py`).

Python 3.11/3.12 (`requires-python = ">=3.11,<3.13"`). Runtime dependencies:
NumPy, SciPy, Matplotlib, pandas, Pydantic 2, PyYAML, `rainflow`.
Optional: `numba` (extra `fast`, compiled kernels; see the kernel section) and
`clarabel` (extra `contact`, used only by a validation cross-check script).

## Package layout

```
src/cablexsec/
  __init__.py          public re-exports, __version__
  api.py               stiffness, M-kappa, shell stress/fatigue, fatigue front door
  project.py           CableModel, settings, AnalysisRunner, AnalysisRecord, Project
  _project_codec.py    explicit JSON encoding of histories and result objects
  provenance.py        Provenance, ValidatedRange, finiteness/real-array guards
  exceptions.py        CableXSecError hierarchy
  cable_adapters.py    CableData -> section and physical-wire model inputs
  coupling.py          xsec_stiffness: tangent 6x6 for a global rod solver
  constants.py         numerical/physical constants with sources
  cli.py               console command cablexsec-batch (CableDyn sea-state batch)
  cables/              YAML schema, loader, packaged reference cables
  materials/           material property library, S-N/SWT curve registry
  mechanics/           section, contact, helix and thermal kernels
  analysis/            stress, fatigue, long-term, capacity, components, life
  io/                  load histories, CSV/RIFLEX/CableDyn readers, synthetic seas
  plotting/            publication figures and an HTML result viewer
```

### `cables` — cable definition

- `schema.py`: frozen Pydantic models for the cable and its layer variants
  (conductor, insulation system, screens, sheaths, helical armour, fillers,
  cores), field validators that reject non-finite values, and `load_cable`.
- `__init__.py`: paths to the packaged definitions
  `inter_array_single_core_33kv.yaml`, `inter_array_three_core_33kv.yaml`,
  `export_three_core_220kv.yaml`, `cigre_tb880_220kv.yaml`, `menard2023.yaml`
  and `menard2023_filled.yaml`, `iea_task49_66kv.yaml` (public 66 kV
  wet-design template) and `iea_task49_66kv_300.yaml` (derived 300 mm²
  section), `chae2023_coiling.yaml` (Chae, Nam and Shim 2023 coiling case),
  `witz1996.yaml` and `witz1996_rect.yaml` (Witz 1996 pipe, round or
  rectangular wires), `saevik2011.yaml` and `saevik2011_rect.yaml` (Sævik
  2011 FBG riser) and `thies2016_30kv.yaml` (Thies et al. 2016 three-core
  cable), with the two legacy copies below.
  `inter_array_single_core_33kv_legacy_r2.yaml` and
  `inter_array_three_core_33kv_legacy_r2.yaml` are byte-pinned copies of the
  33 kV examples before their geometry correction; they reproduce earlier
  results, and `validation/scripts/run_validation.py` uses them.
  `load_cable(path, geometry_override=None)` can merge geometry withheld as
  `null` in a definition from a separate YAML, by layer name.

`CableData` is the single input object for geometry and material choices.
It is immutable; edits produce a new validated object.

### `materials` — properties and fatigue curves

- `library.py`: the single source of material properties (moduli, Poisson
  ratios, expansion coefficients, conductivities) with citations.
- `sn_curves.py`: the registry of S-N curves (`SNcurve`, single or bilinear,
  optional knee and cut-off, test R ratio, environment, source, thickness
  exponent, explicit stress or slip-amplitude units; `SNcurve.from_knee`) and
  SWT strain-life constants. `register_sn_curve` / `unregister_sn_curve` add
  user curves (source and environment required; built-in names protected).
  Definitions are validated before any history is processed. Curves whose
  source standard is not yet available are blocked (`BlockedSNCurveError`)
  rather than silently used.
- `mean_stress.py`: `MeanStressSpec` and `equivalent_range`: Goodman,
  Gerber, Soderberg, SWT and Walker corrections to the curve's test R ratio
  or constant mean, with an explicit UTS or
  yield source, and the DNV-RP-C203 Eq. 2.15/2.16 range factors.
- `strain_life.py`: Coffin–Manson–Basquin constants of metallic sheaths,
  foils and tapes with their basis (published, derived, assumed), Masing
  stresses, SWT and Morrow corrections, `register_strain_life` (gap P6);
  also used for copper screen and conductor wires above yield and for tapes
  (gaps P7, P2; `copper_annealed_mus_derived` beside `copper_poon2022`).
  `library.register_material` adds user materials with a citation per
  property (gap P4).
- `wear.py`: energy wear coefficients per material pair (`WearCoefficient`,
  value, scatter band, source, assumption flag, worn body) and the
  fretting-fatigue knock-down laws (`FrettingKnockdown`). Steel-on-polymer
  coefficients are labelled assumptions.

### `mechanics` — section and contact kernels

Functional kernels operating on arrays; most are not called directly by users.

| Module | Responsibility |
| --- | --- |
| `geometry.py` | Helical and nested-helical kinematics (lay angle, curvature/torsion of a helix) |
| `section_geometry.py` | Material areas and volumes for the fixed-axis section |
| `section_conventions.py` | Transformations between rod resultants and local strain inputs: right-handed rotation curvature to strain gradient `(−k_y, k_x)`, bend moment to `(−M_y, M_x)` (checked against a global line model); the legacy `curvature_map="paper_r2"` only for reproducing earlier results; pressure-thrust force conversion |
| `stiffness.py` | Axial–torsional matrix and bending-stiffness bounds (EI full-stick, EI full-slip) |
| `element_stiffness.py` | Axial–torsion stiffness assembled from per-element stiffnesses (Sævik–Bruaseth population) |
| `axisymmetric.py` | Bounded inter-layer contact-pressure chain, winding and capstan pressure (default path for bending and fatigue) |
| `concentric_section.py` | Coupled extension, twist and radial contact of concentric elastic layers |
| `contact_qp.py` | `solve_contact_qp`: radial active-set contact with Lamé compliance; KKT-verified |
| `contact_pressure.py` | Three-core (trefoil) radial foundation-reaction approximation |
| `section_state.py` | `SectionState`: a solved normal state shared by bending and stress recovery |
| `papailiou.py` | Papailiou (1997) axial slip-front bending model; per-wire slip history (reference implementation) |
| `fast_slip.py` | Compiled per-wire friction-stress recursion (see below) |
| `distributed_friction.py` | Optional distributed elastic–Coulomb friction along the helix (`friction_model="distributed"`, see below) |
| `progressive_friction.py` | Optional progressive (Iwan) split of the direction-tied friction cap into parallel sub-elements (`friction_model="progressive"`, gap G8) |
| `clamp_layer.py` | Anchored-wire friction boundary layer at a clamp under a curvature profile (exact chain projection per increment) and the clamp closed forms (Kraincanic–Kebadze far field, `A = 1 + 0.54√(κ/κ_c)`, `ℓ_x = ½√(L_s P)`); used by `simulate_rig_test` and by the fatigue termination correction (`analysis/termination.py`); `anchored_wire_field_response` takes a non-proportional field `κ(x, t)` segment by segment (gap P5) |
| `along_wire.py` | Slip redistribution along helical wires through a non-uniform, time-varying (biaxial) curvature field between anchored or free ends: elastic (`spring`) or rigid interface, capstan bound, exact projection or semismooth Newton per increment, energy identity; `WireChainLayer`, `along_wire_history`, layer step and station Jacobian for the beam coupling (gap T2) |
| `mkappa.py` | Moment–curvature engine as a per-wire population sum |
| `critical_curvature.py` | Per-wire slip-onset curvature |
| `yun_coefficient.py` | Shear coefficient ξ for the stick-branch stiffness |
| `return_mapping.py` | Elastic–perfectly plastic Coulomb return-mapping element |
| `mindlin_contact.py` | Tangential Coulomb return map with Mindlin initial stiffness (screening) |
| `hertz_contact.py` | Elliptical Hertz footprint for crossed wires (Carlson-integral solve) |
| `hertz_stress.py` | `compute_hertz_axis_stress`: stress tensor on the contact symmetry axis |
| `hertz_field.py` | `compute_hertz_subsurface_stress`: off-axis tensors from prescribed normal and shear tractions |
| `wire_stress.py` | Per-wire stress components (axial, local bending, friction) |
| `shell_bending.py` | Metallic-sheath stiffness and stress |
| `core_lay.py`, `core_strands.py` | Helical transfer of assembled power cores; resolved internal wire rings on fixed core axes (no strand slip) |
| `core_friction.py` | Optional stick–slip law of laid power cores against neighbours, filler and ring, and of resolved internal rings (`compute_mkappa(core_friction=...)`, gap R1b): core stations on the armour return map, internal rings on the Karlsen-Husøy operator |
| `core_kinematics.py` | Core slip state (`"stuck"`, `"free"`, `"frictional"`) and the strain field of any layer inside a laid power core: core-axis strain per station plus the core's own bending (gap P8) |
| `bend_restrictor.py` | Non-proportional curvature fields `κ(x, t)` (gap P5): articulated restrictor (rigid-link chain, bore clearance, element-by-element locking on a tensioned beam) and conical polyurethane bend stiffener (large-deflection beam); `CurvatureField` feeds the rig, bending-test and termination solvers |
| `tendon_section.py` | Wire cross-sections (gap G1): round, rectangular with corner radius, profiled with declared properties; area, principal inertias, torsion constant, radial thickness, surface width, fill, perimeter stress points |
| `tendon_bending.py` | Local tendon bending (gap G1): normal, lateral and torsion curvature from the loxodrome to the geodesic, transverse Jenkin model, vector-Coulomb relief of the axial friction bound, Dong et al. (2018, 2025) anchored-tendon solutions |
| `strand_slip.py` | Karlsen-Husøy multilinear stick–slip operator and stranded-conductor ring parameters |
| `thermal.py` | `compute_thermal`: IEC 60287 radial ladder and 1-D Crank–Nicolson solver |
| `thermal_network.py` | `thermal_network`, `ThermalNetwork`: the `compute_thermal` ladder as a transient radial RC chain (same annuli and `T1` split, layer heat capacities, `R_ac(T_c)` solved implicitly, TR-BDF2 or Crank–Nicolson); layer temperatures on the slow time scale (gap 4.1) |
| `radial_solution.py` | Coupled radial–axial–torsional section solution with unilateral contact (one elastic energy minimised with common extension and twist), behind the solved contact route |
| `trefoil_section.py` | Plane generalised-plane-strain model of a three-core trefoil bundle with contact (six-node triangles on a 60° sector) |
| `radial_cable.py` | `solve_cable_radial_state(s)`: solved radial contact state of a cable definition (temperatures, moduli, flooded pressure); the plural form assembles the section once for a tension grid; `loose_layers="neighbour"` meshes loose core screens with the material of their softer structural neighbour (`loose_layer_sources`) |

### `analysis` — engineering results

| Module | Responsibility |
| --- | --- |
| `stress.py` | `compute_stress`: wire-resolved stress decomposition and slip state for a target load |
| `fatigue.py` | `compute_fatigue` engine: fixed-material-fibre stress histories under biaxial bending and normal load (round, rectangular or profiled wires), rainflow, Miner bulk damage, contact indicators, termination SCFs, optional clamp correction of the friction stress (`termination=`), user S-N options per layer (`mean_stress`, `component_scf`, thickness) |
| `sn_assessment.py` | `ComponentSCF` (axial, friction, normal and lateral bending factors, mean and dynamic) and the corrected Miner sum (`corrected_damage`, `corrected_damage_rows`): stress factor, mean-stress correction and thickness factor per rainflow cycle; used by the armour, conductor, sheath and tape routes |
| `termination.py` | `TerminationSpec` and `termination_friction_increment`: anchored-wire clamp increment of the friction stress for a section near an end fitting or clamped stiffener root (`clamp_layer` solver along the principal, optionally also the minor, curvature direction, or `solver="along_wire"` with `stick`, `capstan` and `lateral_relief`; `curvature_field=` for a non-proportional field; far-field reference from the same discrete solver, so the sectional result is recovered away from the anchor) |
| `fast_rainflow.py` | Compiled rainflow counting and Miner damage for many histories (see below) |
| `long_term.py` | `compute_long_term_fatigue`: probability- and duration-weighted damage at fixed fibres before hotspot selection |
| `sea_state_batch.py` | `run_sea_state_batch`: CableDyn runs with probabilities to weighted annual damage and life per section (reader, `compute_fatigue` with optional clamp, thermal, fretting and user S-N options, `compute_long_term_fatigue`), optional capacity check of every section and run (`CapacityBatchCheck`); spawned workers with one thread each, JSON specification, manifest with input digests; console command `cablexsec-batch` ([doc/global_local_guide.md](doc/global_local_guide.md), stage 1) |
| `critical_plane.py` | `compute_swt_cycle`: sampled SWT critical planes for one closed cycle at one point |
| `fretting_swt.py` | SWT strain-life solve and a simplified contact-driver estimate (screening) |
| `fretting_regime.py` | Per-contact fretting regime (stick, partial, gross slip) per rainflow slip cycle from the Mindlin transition amplitude and Fouvry ratios; Mindlin and Cattaneo-strip dissipated energy; energy wear depth; regime-dependent stress-range knock-down at the contact fibre (gap 3.1) |
| `fretting_assessment.py` | Applies `fretting_regime` to the faces of helical wire layers (polymer line contacts, armour crossings) and to conductor trellis crossings; annualised regime maps, wear and damage per wire with the angular distance from the extreme fibre; `compute_fretting` in `api.py` assembles the faces from a cable |
| `core_screen.py` | `compute_core_screen_fatigue`: wire screens wound about each power core; helical-layer stress and slip under the core curvature plus the core-axis strain of the declared core state, bulk S–N, face fretting with the declared tape/sheath pressure (gap P1) |
| `sheath_strain.py` | `compute_core_layer_strain` (gap P8) and the strain-life route of `compute_shell_fatigue` (gap P6): fibre strain of a concentric sheath or a core layer, bonded-laminate kinematics, damage map |
| `tape.py` | `compute_tape_fatigue` (gap P2): helical tape geometry (pitch, gap, overlap, edge angle), tape strain in the stuck, free and frictional states from the substrate or core-layer strain field, slip along the tape and between overlapped turns, strain-life damage |
| `components.py` | `compute_component_response`: stress/strain bounds and screening for every component |
| `section_stress.py` | `compute_section_stress`, `armour_stress_fields`: the longitudinal stress and utilisation at points over every component (one point per armour wire, angles and radii through the wall of continuous layers and cores, angles on each wire ring of a resolved conductor), from the functions of `components.py` at the same inputs, so each layer's extremes are its component extremes; the fields of the section stress maps |
| `capacity.py` | `compute_capacity`: tension–curvature envelope from independent screening mechanisms; `wire_yield_curvature_levels`: curvatures at several wire-usage levels |
| `capacity_check.py` | Capacity check of global time series: `build_capacity_envelope` (wire-usage level curves at the design factor, lateral and radial buckling, conductor and polymer limits), `CapacityEnvelope.from_points` (imported curve), `check_capacity` / `check_capacity_series` (utilisation history, maximum and time, exceedances, minimum bend radius, block-maxima Gumbel fit), `capacity_extremes` and `gumbel_fit` over seeds |
| `conductor_fatigue.py` | `assess_conductor_fatigue` behind `compute_conductor_fatigue`: ring-by-ring strand-slip strain, bulk Miner sums (optional user S-N options), opt-in trellis SWT fretting (not validated) and strain life above yield |
| `compression.py` | `compression_screen`, `compute_compression_capacity`: armour in compression; lateral buckling lower bound (Sævik–Thorsen Eq. 19) and round-wire limit with cyclic-curvature knock-down (Li–Vaz), friction first-slip gate, radial (bird-cage) buckling of the armour stack on the outboard sheath and tape foundation with hoop-capacity and yield interaction (Eq. 35), DNV-RP-F401 utilisation factors, combined tension/compression–curvature curve ([doc/uls_guide.md](doc/uls_guide.md)) |
| `coiling.py` | `compute_coiling`: a cable coiled into a static tank (curvature 2/D with one twist per winding, Chae, Nam and Shim 2023); the twisted section solved first (radial, axial and torsional), then the bending turned through each winding on fixed fibres; per armour layer the extreme wire stresses, the friction stress, the contact pressure, the radial-buckling (bird-caging) capacity in wire stress and the first-yield check (beyond first yield the result says the elastic stresses are not valid); the friction-free lateral bound is reported, not used; unbonded flexible pipes are refused and counter-helical (non-coilable) armour is flagged |
| `bending_law.py` | `export_bending_law`, `IwanBendingLaw`, `BendingLawExport`: the section moment–curvature backbone at several tensions decomposed into an exact Iwan law (parallel elastic–perfectly plastic elements) for global solvers; tension interpolation; JSON, CSV and kN–m table writers ([doc/global_local_guide.md](doc/global_local_guide.md) Section 8) |
| `sheave.py` | `compute_sheave`: a cable pulled over a sheave; groove line and global pressure, cosine groove distribution and run-on contact (Feyrer 2015), wire stresses, capacity utilisation and per-passage damage from the section analyses |
| `electrical_life.py` | `compute_electrical_life`: coaxial field and relative water-tree ageing factor |
| `curve_comparison.py` | Sequence-preserving comparison of measured cyclic curves (benchmark support) |
| `beam_test.py` | `simulate_bending_test`: three-point, four-point and cantilever bending tests of a hysteretic section; stations along the span, each inverting its own history-dependent law (`SectionLaw`: `compute_mkappa` options, distributed armour law, conductor strand slip) under load, displacement, moment or curvature control; reports the test's moment formula and curvature definition and the energies (gap T1; [doc/local_analysis_guide.md](doc/local_analysis_guide.md) Section 6.2) |
| `along_wire_test.py` | `AlongWireSpec`, `AlongWireSection`: the bending-test option that solves every helical wire along the specimen together with the beam (Newton on station curvatures and load with the exact station Jacobian; overhangs; anchored or free wire ends) |
| `rig_test.py` | `simulate_rig_test`: tension–bending rig with clamped ends; gauge strain histories (axial, local bending, friction with clamp increment), M–κ loops at stations, μ and clamp sensitivity, gauge recommendations; optional layer-temperature history (`thermal=`) ([doc/rig_test_guide.md](doc/rig_test_guide.md)) |
| `torque_balance.py` | `torque_balance`, `design_torque_balance`: end torque at fixed twist and twist of a free end under tension from the solved radial section (`solve_cable_radial_states`, both end conditions, declared clamping carried), layer torques `s n A sigma r sin(alpha)`, armour balance ratio, free-twist stress redistribution, fixed-radius values of `xsec_stiffness`; outer-armour lay angle, wire count or wire diameter that zeroes the torque/tension coefficient at a tension or minimises it over a range, within a lateral fill limit (gap 3.5; gates TQ1–TQ7) |
| `thermal_coupling.py` | `ThermalHistory`, `ThermalState`, `armour_thermal_state`, `compute_thermal_fatigue`: slow-time layer temperatures coupled one way into the solved radial contact state (free thermal strain, registered `E(T)`), the Yun stick stiffness and the wire stress; per-block fatigue with the fast record, dimensionless groups Π1–Π5 (gap 4.1) |

### `io` — load histories

- `series.py`: `LoadCaseSeries` with explicit force kind (true/effective),
  strain kind and signed curvature components; histories are copied into
  read-only storage on construction. CSV read/write; `fatigue_from_series`.
- `time_history_csv.py`: time-history CSV exports, columns selected by header.
- `cabledyn.py`: CableDyn deck and output files (standalone driver and
  OpenFAST module). CableDyn writes curvature magnitudes only; the reader
  reconstructs the curvature vector from the node tangent channels or
  positions, projects it on a twist-free section frame (material twist added
  on lines restrained in torsion), interpolates tension, forms hydrostatic
  pressure, and checks every file, channel and unit against the deck
  ([doc/global_local_guide.md](doc/global_local_guide.md)).
- `riflex_reader.py`: RIFLEX ASCII results.
- `synthetic.py`: self-contained JONSWAP-style load-series generator.

### `plotting`

Static publication figures (`cross_section`, `mkappa`, `stress`, `fatigue`,
`thermal`, `capacity`) sharing `style.py`, and `dashboard.py`, a
self-contained HTML viewer of result objects. Plotting is never imported by
the numerical modules.

### `api`, `project`, `provenance` — the public surface

- `api.py` holds `compute_stiffness`, `compute_mkappa`, `compute_shell_stress`,
  `compute_shell_fatigue` and the `compute_fatigue` front door, plus their
  result dataclasses. `__init__.py` re-exports these together with the
  analysis functions, readers, Hertz functions and error classes.
- `project.py` adds an object layer over the same functions: `CableModel`
  (owns a `CableData`), typed settings for stiffness, planar bending,
  stress, history fatigue, thermal rating, capacity, coiling, conductor
  fatigue and the bending-law export, `AnalysisRequest`, a stateless
  synchronous `AnalysisRunner`, immutable `AnalysisRecord` snapshots and a
  versioned `Project` (format version 2, version 1 files load; atomic save,
  checksums, stale-result detection). `_project_codec.py` performs the
  explicit, schema-checked encoding ([doc/project_guide.md](doc/project_guide.md)).
- `preflight.py`: `preflight` and `preflight_request` list, without running,
  the combinations a run would refuse (errors) and the runs that rest on an
  assumption the user should see first (warnings).
- `provenance.py`: every public result carries a `Provenance` (method,
  citations, verification gates, limitations). `ValidatedRange` and
  `check_range` emit `ExtrapolationWarning` outside a declared range;
  `require_finite` and `real_array` reject non-finite or complex input.
- `clamping.py`: `clamping_state`, `ClampingState`, `LayerClamping`: the
  explicit manufacturing-clamping state with its basis label (gap 4.1b).

## Data flow

```
YAML / CableData ──► cable_adapters ──► section inputs (layers, helices, wires)
                                              │
global history ──► io readers ──► LoadCaseSeries (read-only, explicit conventions)
(CableDyn runs ──► io.cabledyn section histories ──► sea_state_batch)
                                              │
                    ┌─────────────────────────┴──────────────────────┐
                    ▼                                                ▼
     normal state: axisymmetric chain              stiffness.py / coupling.py
     (default) or contact_qp / concentric          (EA, GJ, EI bounds, tangent)
     ──► SectionState (pressures per face)
                    │
                    ▼
     per-wire kinematics (geometry, wire_stress)
     + Papailiou slip front / fast_slip  ──► axial, bending, friction stress
       (or distributed_friction)
                    │                         at fixed fibres, every time step
                    ├──► compute_stress / compute_component_response
                    ├──► compute_mkappa (moment, dissipated work)
                    ▼
     fatigue.py: turning-point reduction ─► rainflow ─► S-N (registry) ─► Miner
                    │                       (fast_rainflow for batch use)
                    ├──► contact indicators (Hertz, Mindlin, SWT screening)
                    ▼
     FatigueResult (per fibre, per wire, termination-adjusted field)
                    ▼
     long_term.py: weighted accumulation at the same fibre ─► governing point
```

Key invariants:

1. Damage is accumulated at fixed material fibres and only then maximised;
   maxima from different physical locations are never summed.
2. Contact pressure comes from the declared winding and assembly state through
   the pressure chain; the default friction cap uses the current (held)
   bending direction and pressure, while the optional distributed model uses
   the pressure only and needs no bending direction. Changing-normal-load
   contact-life calculations are rejected rather than reusing frozen Hertz
   properties.
3. The thermal solution feeds the mechanics one way. On the default paths it
   gives restrained thermoelastic stress only. With `thermal=` (a
   `ThermalHistory` or `ThermalState`, solved contact model) the layer
   temperatures of each slow-time block also set the clamping, friction caps,
   polymer moduli and stick stiffness; frictional heating is not fed back
   (Π2). All layers at `T_ref` reproduce the unenergised result bit for
   bit.
4. Imported true-force histories drive both axial stress and contact by
   default, matching the direct API.
5. Every global reader (time-history CSV, CableDyn, RIFLEX rotation input) converts
   rotation curvature through the one function
   `section_conventions.rotation_to_strain_curvature`, so a fibre angle `φ`
   (from local x towards y) names the same physical fibre whichever code
   produced the history: fibre strain `k_x·y − k_y·x`, as a global line
   model reports it (`tests/test_curvature_sign_probe.py`).

## Numerical kernels and fallback

Two `numba`-compiled kernels carry the default fatigue route; the distributed, progressive and along-wire friction
models, the conductor operator and the fretting cycle sums have compiled
kernels of their own, each with a reference path:

- `mechanics/fast_slip.py` (`wire_friction_stress`) reproduces the
  `stress_Pa` output of `papailiou.wire_slip_history` for a constant stick
  stiffness: slip-front cap, scalar elastic–Coulomb return map per wire and
  retained bending direction at zero curvature. It does not form work terms.
  `analysis/fatigue.py` calls it when `numba` is importable and no
  stick-modulus override is supplied; otherwise the reference routine runs.
- `analysis/fast_rainflow.py` (`fatigue_rows`) reproduces turning-point
  reduction, the `rainflow` package reversal scan (ASTM E1049-85), three-point
  counting and Miner/equivalent-range sums cycle for cycle and in the same
  summation order. `fibre_fatigue` counts many fibres of a global-campaign
  history at once, with optional `prange` parallelism.
  `miner_rows` applies the same counter to the bulk Miner sums of
  `compute_fatigue` (including a termination stress-range factor) when
  `numba` is active; rows where the reference `miner_damage` would raise are
  rerun on the reference path. `rainflow_counter="reference"` keeps the
  reference counter available.

`fast_rainflow.rainflow_table` returns the same cycles with their sample
indices (range, mean, count, start, end) for the fretting module, which also
compiles the window slip amplitude of each stress cycle and the per-cycle
knocked-down Miner sum; `tests/test_fretting_regime.py` compares the table
with `rainflow_cycles` cycle for cycle, including plateaus and short records.

Both modules take `njit`/`prange` from `src/cablexsec/_numba.py`, a typed wrapper
that is the identity when `numba` is absent or `CABLEXSEC_DISABLE_NUMBA=1`, so
the NumPy/Python path remains the reference implementation and the fallback.
`numba` is the optional extra `fast` (`numba>=0.65`), and project engine
records store its version.
`tests/test_fast_slip.py`, `tests/test_fast_rainflow.py` and
`tests/test_compiled_counter.py` compare compiled and reference outputs;
recorded damage is identical. Measured cost for local
recovery is 0.76 s per section-hour on one thread and 0.57 s on eight threads;
the small parallel gain indicates the workload is memory-bound. These are
host-specific measurements, not comparator claims.

### Distributed friction model (optional)

`mechanics/distributed_friction.py` adds a second armour friction law,
selected with `friction_model="distributed"` in `wire_stress_field`,
`analysis.fatigue.compute_fatigue` and the public `compute_fatigue`; the
default `"direction_tied"` path is unchanged bit for bit. For each layer it
solves one wire over one pitch of its helix (uniform curvature, axial slip
only) as the elastic–Coulomb problem `n = e + u′`, `n′ = t`,
`t = k̂ (u − p)`, `|t| ≤ ρ(t) g(φ) + λ|n|`, with `k̂ = ξ/(1 − ξ)` from the
layer's Yun coefficient, the capstan bound `λ = μ |sin α|` and the clamping
ratio `ρ = P(t)/P_ref`. The normalisation is that of the reference solver:
below slip the field is the ξ-reduced stick stress of the default law, slip starts at the same
curvature `κ_on`, and planar monotonic loading to full slip gives the
package cap `σ0 expm1(μ |sin α| ψ)`. Each wire samples the field by phase.
There is no held bending direction.

The kernel is `numba`-compiled. Each increment is backward Euler: a
semismooth Newton iteration with a cyclic tridiagonal Sherman–Morrison solve
(O(N) per iteration) on the tridiagonal generalised Jacobian, including the
dependence of the capstan bound on `n`; on failure it falls back to the
reference scheme (Newton with Armijo backtracking on the convex increment
potential for a fixed bound, fixed point on the bound), and an increment that
still fails is repeated with twice as many substeps. Increments larger than
`distributed_max_increment` (default 0.1 `κ_on`) are subdivided. The
energy identity `W − D − U − R = 0` (external work, friction dissipation,
stored energy including the interface spring, backward-Euler remainder)
holds to round-off and is returned by `distributed_friction_history`.
`tests/test_distributed_friction.py` reproduces recorded reference values to
1e-10, compares the kernel with an independent NumPy port of the reference
solver, checks the rigid-stick limits, and records the planar differences
from the default law. Cost on one thread for the reference section-hour
(two layers, 36 001 samples): friction kernel 0.87 s at the default four
nodes per wire (0.44 s at two) against 0.094 s for `fast_slip`; stress
recovery and 32-fibre damage 1.46 s (1.04 s) against 0.70 s.

## Input schema and validation boundaries

| Boundary | Mechanism | Failure type |
| --- | --- | --- |
| Cable YAML | Pydantic models in `cables/schema.py`; finite-value validators; duplicate layer names rejected | `CableSchemaError`, Pydantic `ValidationError` |
| Load histories | `LoadCaseSeries.__post_init__`: shape, ordering, finiteness, complex rejection, read-only copies | `CableInputError` (also `ValueError`) |
| CSV histories | filename and line number reported, original exception chained | `CableSchemaError` |
| Material / curve / layer lookup | registry access | `CableLookupError` (also `KeyError`); `BlockedSNCurveError` |
| Analysis inputs | explicit guards in `api.py` and analysis modules; unit checks on S-N curves | `CableInputError` |
| Solvers | convergence and residual checks (SWT bracketed solve, contact QP) | `CableSolverError` (also `RuntimeError`) |
| Physical range | `ValidatedRange` | `ExtrapolationWarning` or `PhysicalRangeError` |
| Project files | strict decoding, checksums, engine-identity and reference checks | `CableSchemaError`; filesystem errors keep their native types |

All library failures derive from `CableXSecError` while remaining catchable by
the matching builtin exception.

## Provenance and reproducibility

- Each result carries `Provenance`; `validation/scripts/run_validation.py` regenerates
  `validation/validation_report.md` from live code, and the validation policy is in
  [doc/validation_policy.md](doc/validation_policy.md).
- `AnalysisRecord` stores the exact input snapshot, settings, results,
  source/dependency identity and registered material/curve data; a change in
  any relevant input or the environment marks the record stale.

## Testing strategy

`pytest` collects the tests of the `tests/test_*.py` files (some skip without
the optional Clarabel solver).

Layers:

- **Markers** (`--strict-markers`): `analytical_sanity` (closed-form or
  machine-precision gates), `independent_method` (comparison with an
  independent computational method), `full_scale` (published or full-scale
  data; scope varies by row).
- **Regression reproduction**: each defect fix adds tests that fail against
  the previous version and pass after correction.
- **Equivalence**: object/function equivalence for `project.py`; compiled
  versus reference kernels; matched-case array comparisons against earlier
  versions when performance work changes code paths.
- **Benchmarks**: `validation/validation_report.md` rows (Tier 1 analytical,
  Tier 2 public benchmarks). Failing scientific
  criteria are kept as strict expected failures, never relaxed.

## Extension points

- **New cable construction**: add a YAML file validated by `schema.py`; new
  layer kinds require a schema variant and an adapter in `cable_adapters.py`.
  Round, rectangular and profiled wire sections are represented
  (`mechanics/tendon_section.py`); interlocked carcass and pressure-armour
  profiles are not.
- **Materials and curves**: `register_material`, `register_sn_curve` and
  `register_strain_life` add user entries for the session with a citation
  (built-in entries cannot be replaced); mean-stress corrections
  (`MeanStressSpec`) and component SCFs (`ComponentSCF`) are per-call
  options. Projects store their own copies without mutating the global
  registry.
- **Contact normal state**: the bounded chain (`axisymmetric.py`) and the QP
  solver (`contact_qp.py`) share the `SectionState` output, so another radial
  model can be added behind the same state object.
- **Fatigue laws**: contact-life laws are selected by name in `fatigue.py`;
  unknown names fail. Critical-plane assessments enter via `critical_plane.py`.
- **Readers**: a new global-software reader should return a `LoadCaseSeries`
  with explicit force and strain conventions; it then also feeds the
  capacity check (`check_capacity_series`).
- **Analysis types in projects**: add a settings model, a runner branch and
  codec support in `_project_codec.py`, with save/open round-trip tests.
- **Compiled kernels**: any new kernel must keep a reference implementation,
  a no-`numba` fallback and an equality regression test.

Job scheduling, progress reporting and cancellation are not part of the
package; the project layer is designed so they can be added above it.
