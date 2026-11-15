<!-- SPDX-License-Identifier: Apache-2.0 -->

# Changelog

All notable changes to CableXSec are recorded in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `compute_mkappa` can let the armour contact pressure follow the bending history:
  `contact_tension_history_N` sets the per-sample `P1+P2` from a tension history through the
  contact model (the histories `compute_fatigue` forms from `tension_N`), and
  `contact_pressure_Pa` takes the per-sample `P1+P2` of every structural armour layer. Before,
  one bending calculation held one contact pressure. The pressure moves only the friction bound
  of each wire's return map, with the slip state carried across, so no material-work term
  arises: a falling bound releases the elastic energy `A cos α (c0² − c1²)/(2k)` of a slipping
  wire as friction work, and a rising bound leaves a sticking wire unchanged. The stick and
  full-slip stiffnesses, the laid-core and ring contact loads and the wire-yield range check
  stay at the reference state (the first sample of a tension history). A history equal to the
  fixed pressure reproduces the fixed-pressure loop bit for bit, and the result returns the
  pressures used (`MkappaResult.contact_pressure_Pa`). New validation row MK-Tcyc: a tension
  cycle inside the bending cycles gives the fatigue path's friction stress (3.4e-16 of the
  moment) and closes the work balance (1.1e-15 of the friction work). `SectionLaw` and the bending-test simulator keep a
  fixed pressure (their leg solver evaluates trial curvatures without a sample time), and so
  does the exported bending law.
- The exported bending law can carry a slip scaling variable: `export_bending_law(...,
  slip_scale=True)` adds `BendingLawExport.slip_scale`, a `SlipScaleTable` with the law at a
  reference tension split into friction groups (one per armour layer, `laid_cores` for the core
  lay stage) and each group's scale `ρ_g = R_g/R_g,ref` of its full-slip friction moment over a
  table of tensions and, optionally, external pressures (`slip_scale_tensions_N`,
  `slip_scale_external_pressure_Pa`), with the single cap-weighted scale `s` and the derivatives
  `∂/∂T` and `∂/∂p_e`. The friction bounds of the direction-tied law are proportional to `P1+P2`
  at fixed stick stiffness, so a group's law at another state is the reference law with fixed
  element stiffnesses and yield curvatures times `ρ_g`, and under a changing state every element
  follows the return map with the new cap (`SlipScaleTable.moment`, `.law_at`,
  `.dissipated_energy`). New validation row MX-scale: the section law is homogeneous in the
  bound (4e-16), the group yield curvatures scale exactly (8e-14), and a tension cycle inside the
  bending cycles adds nothing to the fixed-tension residual of the Iwan law (ratio 1.02; with
  symmetric wire bounds the scaled law is the contact-tension history of `compute_mkappa` to
  1e-14). One scale for all groups is exact when they scale alike (66 kV template, tension) and
  misses the per-group law by up to 0.3 % (33 kV three-core), 12 % (Ménard) and 21 % (66 kV with
  external pressure). The JSON law with a slip scale is format `cablexsec.bending_law/2` (a new
  `slip_scale` object; version-1 fields unchanged); `SlipScaleTable.to_csv` writes the table.
  Torque is not a table variable, and the internal-ring slip of resolved strands is refused.
  Project files written before `BendingLawExport.slip_scale` existed still load: the project
  codec accepts a missing field that has a default. `export_bending_law` refuses
  `contact_pressure_Pa` and `contact_tension_history_N` as options.

### Fixed

- The exported bending law no longer claims to follow `compute_mkappa` exactly. With the signed
  capstan bound the slip interval of a wire is asymmetric (the falling bound is the rising one
  times `exp(−λψ)`), which no Masing element carries; the Iwan law misses the section law's loops
  by about 0.1 % of the moment range on bending cycles near full slip (0.08–0.26 % on the 66 kV,
  Ménard and 33 kV three-core templates), and by 0.02 % at μ = 0.02. The provenance of
  `export_bending_law` and `cabledyn_bending_law`, the module notes and the global–local guide
  say so; the law itself is unchanged.
- The frictional heat of `compute_thermal_fatigue` (`frictional_heat_W_m`, `pi2` and
  `friction_temperature_rise_K` of every block) is now the dissipation of the distributed
  helix friction law, whatever friction law the fatigue uses. It was the work of the
  direction-tied cap, which turns with the bending direction and under rotating curvature
  dissipates too little: on a circular orbit 0.16 of the distributed work at twice the slip
  onset and 0.57 at large amplitude (orbit over planar work `π (sinh x₀ − λ)/(4 λ cosh x₀)`,
  `x₀ = λπ/2`, against `π/4` for the distributed law, now tested). Planar records change by
  less than 3 % from twice the slip onset; on the synthetic 66 kV record of the manual the
  largest block heat rises from 0.57 to 0.77 W/m with the signed capstan bound below
  (doc/template_66kv.md Section 4.2). Stresses and damage are unchanged. A
  layer with `ξ = 1` keeps the direction-tied work, and the provenance records it
  (`thermal_coupling.record_friction_work`).
- The capstan term of the armour friction bound is signed, as in Papailiou (1997),
  Eqs. 11, 13-15 and 21, which use the signed helix angle: a rising wire stress is bounded by
  `σ0 (exp(λψ) − 1)` and a falling one by `−σ0 (1 − exp(−λψ))`, `λ = μ |sin α|`
  (`wire_slip_stress_bounds`). Before, the tension bound was applied on both sides, as if
  a wire whose force falls pressed harder on its bed. A full reversal of a slipping wire now
  spans `2 σ0 sinh(λψ)`; the peak tensile friction stress, and so the yield and capacity
  checks on the tension side, are unchanged. The fix applies to the direction-tied,
  progressive, distributed (`b = ρ g + λ n`) and along-wire laws, the tape and laid-core
  laws, the station model of `compute_mkappa_papailiou` and the monotonic stress
  snapshots. The direction-hold threshold is now `f m(λ) κ_on`, `m = 1 − 3λ²/8 + O(λ³)`,
  the compression-side first slip, so that no wire slips below it.
- Size of the correction: the former bound overstated the full-slip extreme-fibre friction
  stress range by `2/(1 + exp(−λπ/2))`, +2.8 to +6.7 % for the bundled power-cable
  templates (`λ` 0.035 to 0.085), and the full-slip loop work by about `(π − 2)λ/2`
  (+6.5 % at 35°, μ 0.2). The bulk damage of the bundled templates falls by 0 to 16 % (the
  former values were 0 to 19 % high): no change while a layer sticks, the largest change
  once the friction range governs the bulk range. Below about three times the onset
  curvature the compression side now slips earlier and a loop can dissipate more.
- Validation rows (VALIDATION.md): BM1-loop and BM1-loop-uniform pass (ordered RMSE
  3.16 % to 2.71 %, cycle work +12.9 % to +8.8 % against the unchanged 10 % target);
  SV11-PLANE and SV11-GJ move closer to the measurements and still fail; WITZ-ID moves
  slightly further (300-bar energy −12.6 % to −14.2 %, width still failing), with the
  initial pressure of `witz1996.yaml` re-identified (1.037 to 1.071 MPa); the
  laid-core closed form of row CF is the signed one.
- `full_slip_curvature` takes the bound of each wire's side; the station slip fraction of
  the Papailiou diagnostic uses the same signed intervals as the moment.

## [0.1.0] - 2026-11-14

First public release. The Python interfaces are versioned but may change before `1.0.0`.
The evidence for every item below is in
[VALIDATION.md](https://github.com/SMI-Lab-Inha/CableXSec/blob/main/VALIDATION.md); its
generated table is reproduced by `python validation/scripts/run_validation.py`.

### Added

#### Cable definitions

- Schema-checked YAML cable definitions: helical wire layers (round wires and rectangular
  tendons), stranded conductors, solid and annular polymer layers, offset cores of
  three-core cables, metallic shells and tapes, with a source recorded for every material
  property. A value that cannot be shared may be written as `null` and supplied from a
  separate file (`load_cable(..., geometry_override=...)`); an incompletely specified
  cable is refused by every calculation with the missing fields named.
- Geometry that cannot exist is refused when a cable is built, with the item named:
  layers out of radial order or overlapping, cores that overlap or form more than one
  family, wires larger than their radial room, lay angles outside [0, 90) degrees
  (handedness is given by `lay_direction` only), non-integer counts, temperatures below
  absolute zero and termination factors below one. Unknown keys are refused at every
  level of a definition and of a geometry override. Round wires that do not fit side by
  side at their helix radius give a `WireFillWarning`.
- Bundled public cables: the three-core cable of Ménard and Cartraud (2023), with and
  without fillers; the IEA Wind Task 49 66 kV reference cable (Janocha et al., 2024) and a
  300 mm² variant; the CIGRE TB 880 220 kV export cable; the flexible pipes of Witz (1996)
  and Sævik (2011), with round wires and rectangular tendons; the 30 kV cable of Thies et al.
  (2016); the coiling cross-section of Chae et al. (2023); and representative 33 kV
  single-core, 33 kV three-core and 220 kV three-core templates.
- A material library with a cited source for every constant, S–N, strain–life and SWT
  curves with explicit unit assignment, and refusal of curves whose source is not yet
  available.

#### Section mechanics

- Axial, torsional and bending stiffness: fixed-radius axial–torsional Schur complement,
  full-slip and no-slip bending bounds, plane-section and shear-compliant no-slip bending,
  load sharing between layers, the neutral axes of the section under a biaxial curvature
  history (bending direction, strain neutral axis and the stress neutral axis of each
  armour layer), and the slip onset and full-slip curvature of each armour layer against
  tension (`armour_slip_curvatures`).
- Solved radial contact state: thick-cylinder compliances of every layer, a contact
  quadratic programme for concentric sections, winding (manufacturing) clamping on both
  armour faces, tension- and pressure-induced contact, effective and true tension, sealed
  and flooded annuli, and thermal strains.
- Hysteretic moment–curvature response of helical armour with frictional slip: the
  neutral-axis slip front (Papailiou), Coulomb saturation, return mapping, shear-compliant
  stick (Yun coefficient), progressive and distributed friction, and laid-core friction of
  three-core cables, with the stored and dissipated work of every history.
- Wire and component stresses at fixed material points: axial, local bending and frictional
  stresses of every wire; conductor strand slip; screen, sheath and tape strains; the
  section stress field; transverse slip and its work on the geodesic.
- Contact mechanics: elliptical Hertz contact with subsurface stress tensors, Mindlin
  partial slip, fretting regime maps (Fouvry), energy wear and SWT critical-plane recovery.
- Thermal analysis: an IEC 60287 thermal ladder for the steady conductor temperature with
  the cited loss factors (for three-core cables without the geometric factor and the sheath
  eddy loss, which makes it conservative), a transient thermal network, frictional heating
  and the thermal change of the clamping.
- Capacity and ultimate limit states: armour yield and sheath strain envelopes, armour
  compression (lateral and radial bird-caging), coiling into a static tank, cable over a
  sheave, critical curvature and the capacity check of time series with Gumbel extremes.
- Termination and bend-stiffener fields: termination-adjusted fatigue, bend restrictors and
  stiffener roots.

#### Fatigue and global-to-local post-processing

- Rainflow counting with residual half cycles and Miner damage at fixed material fibres of
  every wire, for bulk and conductor fatigue; contact (fretting) fatigue where a contact
  curve is available (copper through SWT, steel with a user-registered slip-amplitude curve;
  the bundled cables use bulk fatigue only); metallic sheath (no lead curve), tape,
  core-screen and thermal fatigue; electrical life screening.
- Probability-weighted long-term fatigue over a scatter of sea states.
- Readers for CableDyn section tables and curvature components, time-history CSV exports
  of global analyses, RIFLEX ASCII results and generic CSV series, with recorded sign conventions; a synthetic JONSWAP-type load series.
- `cablexsec-batch`: weighted annual armour damage from a set of CableDyn sea-state runs,
  with a manifest of input digests, checks, options and results.
- Hysteretic bending-law export for global analysis.
- Rig-test and bending-test simulation (three-point, four-point and cantilever
  arrangements) and strain-gauge planning.
- Versioned project files that keep the inputs, results, environment and provenance of
  every calculation, and a preflight report of the inputs.
- Optional compiled kernels (`numba`, the `fast` extra) for per-wire slip stress and rainflow
  counting; results agree with the NumPy reference to floating-point round-off and rainflow
  counts are bitwise equal.

#### Documentation

- A concepts page, four tutorials (a first cable; bending hysteresis against measurements;
  from a global time history to armour fatigue; contact loading and fretting), how-to
  recipes for the remaining capabilities, a file-formats reference and a glossary. Every
  tutorial and recipe is a script whose printed output is checked against its page in the
  test suite.

### Known limitations

- The supported model does not solve the cyclic redistribution of three-core contact.
  Fretting is excluded from the long-term fatigue projections, and contact stress functions
  use prescribed tractions.
- Several measured comparisons do not meet their registered targets; they are reported as
  failed in [VALIDATION.md](https://github.com/SMI-Lab-Inha/CableXSec/blob/main/VALIDATION.md)
  and their tolerances are not relaxed.
- A passing test does not establish that the model predicts every physical cable
  construction or a validated service life.

[0.1.0]: https://github.com/SMI-Lab-Inha/CableXSec/releases/tag/v0.1.0
