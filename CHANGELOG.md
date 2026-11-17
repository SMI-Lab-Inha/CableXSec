<!-- SPDX-License-Identifier: Apache-2.0 -->

# Changelog

All notable changes to CableXSec are recorded in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Validation row BM1-ID (gap R1b), targets registered before the run: on the Ménard and
  Cartraud (2023) cable with the laid cores, the trefoil wedge and the filler profiles (μ 0.2 on
  every contact), the initial pressure identified on the cycle work of range (b) is 220 kPa
  (the source's calibration: 300 kPa). Range (b) ordered RMSE 2.99 % (target < 5 %); predicted
  work of ranges (a), (c), (d), (e), (f) +26.0, +23.7, +12.9, −6.5, −11.7 % against a target of
  ±20 %: FAIL on the two smallest ranges. The armour law identified the same way predicts +35.8,
  +33.5, +21.7, −6.1 and −11.4 %. The laid-core law stays off by default.

- Frozen extrusion shrinkage of a concentric layer: `extrusion_shrinkage` (`hoop_strain`,
  `basis`, `source`; `cx.ExtrusionShrinkage`) declares the part of an extruded sheath's or
  bedding's cooling shrinkage that stays as a shrink fit on the layers inside it. The radial
  solutions carry it as an in-plane initial strain of the layer: the solved radial state (the
  ring around a trefoil bundle included, so the laid-core law reads the clamped trefoil) and the
  concentric `qp` route; `include_manufacturing=False` leaves it out. The bounded pressure chain,
  which has no displacement compatibility, refuses a non-zero value. A shrinkage outside an
  armour layer counts as that layer's clamping declaration. Checked against the Lamé shrink fit
  of a sleeve with an in-plane eigenstrain on a stiff core (both radial solvers, 1e-4).
- `calibration_scope` (`armour_only` or `all_contacts`) on a calibrated `initial_state` and on a
  `published_calibration` armour clamping. The laid-core stick-slip law warns
  (`cx.ClampingCalibrationWarning`) when it is combined with a value fitted with the armour
  friction alone, which already holds the friction of the other contacts. The Ménard and
  Cartraud (2023) pressure is tagged `all_contacts` (its source FE has friction on every
  contact) and the identified Witz (1996) pressures `armour_only`.
- Laid filler profiles in the core stick-slip law: `CoreFrictionSpec(filler_profiles=True)` (and
  `mu_filler_profile`) turns the interstitial filler annulus into three profiles laid with the
  cores that stick and slip with the same direction-tied cap. Their radius `√(J_f/A_f)` and lay
  angle come from the annulus less the core circles, so the stuck profiles add `ξ_f E J_f cos³β_f
  / 2`; each carries the ring force on both faces and has its own Yun `ξ_f`. They form their own
  slip-scale group (`filler_profiles`) in the exported bending law and are reported by the core
  lay stage of the Section properties analysis. Off by default. Checked by validation row CFW
  (profile full-slip moment against the closed form, μ = 0 bitwise, stick tangent against
  `EI_max + E J_f cos³β_f / 2`).
- `CoreFrictionSpec(central_filler=...)` selects the support of the inward core loads in the
  bounded core contact estimate: the neighbouring cores of a touching trefoil (default) or a
  central filler.
- A cylindrical layer defined by its section stiffness: `stiffness: {EA_N, EI_Nm2, GJ_Nm2,
  mass_kg_per_m, source}` (optional `poisson_ratio`) replaces `material` for a component whose
  properties come from a datasheet. It adds its `EA` to the axial stiffness and the load
  sharing, its `EI` to the stick and slip bending stiffness, its `GJ` to the torsional stiffness
  and its mass to the section; the radii place it in the bounded contact chain. A stand-in
  material, an isotropic solid filling the annulus (`E = EA/A`, `EI/I`, `m/A`, Poisson ratio
  0.3 unless given, no thermal expansion), serves the calculations that read a material. The
  coupled radial models refuse the layer. A stiffness block that repeats a material layer's
  values reproduces its stiffness, mass and fatigue damage to round-off.
- A serving layer: `construction: serving` on a `cylindrical` layer declares a served yarn or a
  wound tape. It keeps its mass, its radial and hoop stiffness (an annulus of its material in
  plane stress along the cable axis, in the concentric and the meshed three-core radial
  solutions) and its shear compliance under the armour, and carries no axial force, bending
  moment or torque: no contribution to the axial, bending (stick and slip) or torsional
  stiffness or to the load sharing; in the component response it is outside the load path, as a
  loose screen (kinematic upper-bound stress). A serving is non-structural (`structural: true` is refused) and never an interstitial filler. Layers
  without the key are unchanged. Checked in closed form: at zero Poisson ratio a serving gives
  the solid annulus's radial response with exactly its `E A` and `G J` removed, and an outer
  serving under external pressure has the plane-stress thick-cylinder displacements.
- Selected-position fatigue: `compute_fatigue(..., wire_positions=...)` evaluates chosen wires
  instead of every wire: `N` equally spaced positions per armour layer, explicit azimuths in
  degrees (the nearest wire to each) or a mapping per layer. Each selected wire's damage, fibre
  damage and critical fibre equal those of the evaluation of every wire bit for bit (each wire's
  friction recursion, stress field and rainflow count are independent of the others), in every
  friction model, contact-life branch, termination increment, friction law and S-N option.
  `FatigueResult.wire_index` and `wire_azimuth_deg` name the rows; the critical wire and the
  sums are over the selected wires. The direction-tied kernel then solves only the selected
  wires: on the 66 kV template, one hour at 10 Hz (36 000 samples), both armour layers, one
  thread, 12 positions per layer take 0.11 s instead of 1.24 s (4 fibres) and 0.31 s instead of
  3.72 s (32 fibres). The distributed model solves the helix of the whole layer, so there only
  the counting is saved (24.0 s to 23.3 s): its time is all in the compiled increment solver
  (23.1 of 24.0 s; 2.96 million substeps of 0.1 `κ_on`, 1.1–1.3 Newton solves each, about
  14 ns per helix node and substep), set by the substep rule at the template's small slip-onset
  curvature (`κ_on` 5.1e-5 and 1.6e-4 1/m with tension-only contact); no avoidable overhead was
  found outside it. `SeaStateBatchOptions(wire_positions=...)` passes the selection to every
  run, and `compute_long_term_fatigue` requires the same wires in every case and reports the
  critical wire by its number.
- The armour friction coefficient can depend on the contact pressure:
  `FrictionLaw(pressure_factor=((P, f), ...))`, a piecewise-linear factor `f(P)` on the
  coefficient, clamped at the table ends, evaluated at the local friction-setting pressure
  `(P1 + P2) g` of each wire (direction-tied kernel, NumPy reference and compiled) or helix node
  (distributed kernel) at every sample, so it follows a tension or pressure history in
  `compute_mkappa` and `compute_fatigue`. Measured coefficients of armour wire on anti-wear tape
  fall with pressure (μ_d 0.32, 0.20, 0.18 at 1, 4 and 8 MPa; Dai, Sævik and Ye 2018, p. 428).
  It combines with the static/dynamic and rate factors (`m f(P) μ`), needs no time base and
  is refused where a friction law is (progressive model, laid-core friction, termination clamp
  increment). The exported bending law accepts a law with only a pressure factor at the fixed
  contact state of each tension, and refuses it with `slip_scale=True`: `f(P)` breaks the
  homogeneity `M(κ; ρ) = ρ M(κ/ρ; 1)` the scaling rests on. `slider_history` takes
  `contact_pressure_Pa`. No table (or a table of unit factors) reproduces every result bit for
  bit within each code path. Checked by a spring-slider under pressure steps in closed form, the
  full-slip bound of every wire at its local pressure, the energy identity, a constant table
  against the scaled coefficient and the compiled kernel against the NumPy reference (new
  validation row FL-P).
- Measured friction loops of armour wire on anti-wear tape (Dai, Sævik and Ye, Marine Structures
  59 (2018) 423–443, Fig. 4: Test 1 at 8, 4 and 1 MPa; Test 3 at 0.01–0.2 Hz), digitised with
  provenance and a reproducible procedure (`validation/paper_data/dai2018_fig4a_test1.csv`,
  `dai2018_fig4c_test3.csv`; `validation/scripts/digitise_dai2018_fig4.py`), are compared with
  the static/dynamic `FrictionLaw` on one spring-slider (`check_dai2018_friction.py`, rows
  DAI18-DIG, DAI18-RS, DAI18-T1, DAI18-T3). Gated and passing: the digitised loop maxima against
  the printed static coefficients, and a re-stick peak at every reversal of all seven loops,
  which the law reproduces with `stick_time_s = 0` and not with the default `inf`. Stated, not
  gated: with the printed coefficients the peaks are within 1.8 % and the +x-stroke levels
  within 3.3 % (the measured −x-stroke levels run 4–14 % lower); the law drops to the dynamic
  level in one step where the measured force decays over 0.6–2.3 mm, so the work is 1–5 % low;
  the static peak rises 10.6 % from 0.01 to 0.2 Hz, which the law cannot represent, while the
  dynamic level does not move with frequency (no rate term supported at 0.13–2.5 mm/s).
- `friction_model="auto"` in `compute_fatigue` (public and `cablexsec.analysis.fatigue`) and
  `wire_stress_field`: the direction-tied cap for a planar curvature path and the distributed
  helix model for any other. A path is planar when its out-of-plane ratio
  (`curvature_path_out_of_plane_ratio`, below) is at most `PLANAR_PATH_TOLERANCE = 1e-3`. A
  layer that the distributed model does not cover (`ξ ≥ 1`, a stick modulus that changes over
  the record, a rate-dependent friction law) keeps the direction-tied cap. The provenance
  records the ratio, the model of every layer and the reason for a fallback. The default stays
  `"direction_tied"` in this release, bit for bit (see Deprecated).
- `DirectionTiedFrictionWarning` (a `FutureWarning`): `compute_fatigue` warns once per call when
  `friction_model` is left at its default and the curvature path is not planar. The
  direction-tied cap turns with the bending direction. Under a steadily rotating curvature at
  twice the slip onset its largest per-wire friction-stress range on the 66 kV inner armour was
  0.94 of the distributed model's with the template interface (`ξ = 0.81`) and 0.86 with a
  near-rigid one (`ξ = 0.99`): bulk damage 0.79 and 0.55 of it at an S–N slope of 4 where the
  friction range governs. Its friction work per revolution was 0.38–0.69 of the planar cycle,
  against 1.8–2.6 in the distributed model. On planar and flat paths (minor over major axis up
  to about 0.3) its ranges are at or above the distributed ones. The result is unchanged; an
  explicit `friction_model` keeps the chosen law without the warning. Cost of the distributed
  model on one thread: 1.0–1.3 times the direction-tied time on the shipped fatigue examples,
  about 4 times on the 66 kV template (substeps of 0.1 `κ_on`; see Changed for the faster
  kernel); with `"auto"` the largest damage
  of the shipped examples changes by at most 0.02 %, on an hour of the 66 kV template at 10 Hz
  it is 0.92 of the direction-tied value. Tests: tests/test_biaxial_friction_default.py.
- A bound option of the distributed friction model, `distributed_bound` in `compute_fatigue`
  (public and `cablexsec.analysis.fatigue`) and `wire_stress_field`, `bound` in
  `distributed_friction_history` and `distributed_wire_friction_stress`: `"signed"` (default,
  `b = ρ g + λ n` with the capstan term on the signed bending force of the wire, unchanged bit
  for bit) or `"uniform"` (`b = ρ g`, no capstan term, the same stress unit and onset). In
  planar full slip the uniform bound gives the linear Coulomb field `σ0 λ x` and the signed one
  `σ0 (C exp(λx) − 1)`, both spanning `σ0 λπ`, and the cycle work of the two forms differs by
  less than 0.04 % on the 66 kV template armour (tested). A capstan term on the total wire force
  (bending plus tension share) is not offered: the clamping `P1+P2` that scales `g` already
  holds the radial load of the tension share, which it would count twice.
- `cablexsec.analysis.fatigue.curvature_path_out_of_plane_ratio`: the largest distance of a
  curvature sample from the principal line through the origin over the largest curvature (zero
  for a planar path in any plane, `b/a` for an ellipse, one for a circle), and
  `PLANAR_PATH_TOLERANCE = 1e-3`, the planarity of `friction_model="auto"`.
- An anchored armour wire can be held by an elastic anchor instead of a rigid one, in every
  route that anchors a wire: the termination correction (`TerminationSpec(k_anchor_N_m=K)`,
  clamp and along-wire solvers, with or without a curvature field), the along-wire chain
  (`along_wire_history(..., anchor_stiffness_N_m=(K_start, K_end))`, `WireChain.build`) and
  the along-wire bending test that solves the beam with the wires (`AlongWireSpec(
  k_anchor_N_m=...)`); `anchored_wire_response` and `anchored_wire_field_response` take the
  normalised stiffness `anchor_stiffness = K R/(EA sin α)`. `K` [N/m] is the axial force on
  the wire end per unit axial displacement of the wire end at the anchor, per wire (the
  potting of an end fitting). The anchor node carries the spring in parallel with the
  friction of its half segment; `0` is a free wire end and `None` (default) or `inf` the rigid
  anchor on the unchanged code path, so existing results are unchanged bit for bit. The local
  analysis guide (Section 5, Step 6) derives a shear-lag estimate for a potted fitting, about
  `1e8` to `2e9` N/m per tendon (an order-of-magnitude estimate, not a sourced value). Checked
  against the closed form of a wire with a spring anchor next to a slip zone (anchor force, slip
  length and dissipation, rigid and elastic stick), the free-end and rigid limits, two
  independent kernels to round-off, the energy identity, the station Jacobian and the
  closed-cycle work of a bending test (doc/benchmarks.md, TA-V1 to TA-V6). The Sævik (2011)
  termination rows are reported over the stiffness range as a sensitivity study
  (`validation/scripts/check_saevik2011_anchor.py`), not fitted and not gated: from the rigid anchor to a free wire end the termination-corrected
  SV11-PLANE median stays 1.62 and SV11-GJ 1.80 (changes at round-off), the along-wire variant
  3.11 and 4.19 (changes below 5e-4). The end fitting sits 2.3 m before the bellmouth where the
  curvature is about 1% of its peak, so the wire next to it stays in stick and the anchor
  condition does not reach the stations; the over-prediction is a curvature-gradient effect.
  Section records of the sea-state batch carry the new key; records without it load as rigid.
- A friction law with a static/dynamic and a rate-dependent coefficient (`FrictionLaw`,
  `compute_mkappa(..., friction_law=, time_s=)`, `compute_fatigue(..., friction_law=, time_s=)`).
  Before, every contact pair had one rate-independent Coulomb coefficient. The law gives each
  wire (direction-tied model) or helix node (distributed model) a factor on its coefficient:
  `static_factor` until it slips, `dynamic_factor` while it slips and until it has stuck for
  `stick_time_s`, either order (`mu_s > mu_d` gives a breakaway peak, `mu_s < mu_d` velocity
  strengthening), times `R(v) = 1 + c ln(1 + v/v0) + v/v_eta` at slip rate `v`. The last term
  is a regularised Coulomb law `t = b sign(v) + eta v` with `eta = b/v_eta`, solved as a
  backward-Euler (Perzyna) return that is exact for each step (a bracketed Newton iteration
  with bisection that converges however steep the rate term is, and a return that does not
  converge is refused); `BendingWork.viscous_J_per_m`
  reports its dissipation and the work balance closes with it to round-off. The direction-tied
  kernel takes every option (NumPy reference and compiled kernel), the distributed kernel the
  static/dynamic option; the progressive model, the laid-core friction and the termination
  clamp increment refuse a non-neutral law, and so does the exported bending law (a Masing/Iwan
  law is rate independent). A law that depends on time needs the sample times and is refused
  without them. Neutral parameters reproduce every result bit for bit. `slider_history` runs
  the law on one spring-slider: breakaway peak, healing, steady logarithmic rate and the
  viscous loop area `pi eta omega x0^2` are tested in closed form. No rate law of armour
  friction is validated (the measured wire-on-tape loops above show no rate effect on the dynamic level).
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
- Reported comparison of the anchored-wire (clamp) solver with two published finite-element
  studies of one armour wire fixed at an end fitting under uniform curvature
  (`validation/scripts/check_anchored_wire_fe.py`, `tests/test_anchored_wire_fe.py`, digitised
  curves in `validation/paper_data/`; doc/benchmarks.md, AW-DONG25 and AW-ZT14; not gated). Dong
  et al. (2025) are reproduced at their verification case (156.4 against 156.9 MPa at a
  neutral-axis anchor, peak within 1.6 %); Zhu and Tan (2014) are not (1.70 against about 2.2
  times the free-field peak): their FE far field does not slip, and their analytical friction
  field breaks the Coulomb flow rule. The friction regime parameter of Zhu and Tan, not printed,
  follows from their Table 2. No solver change.
- An umbilical utilisation basis of the capacity envelope:
  `build_capacity_envelope(..., basis="iso_13628_5")` applies one utilisation factor `k_u` to
  every element of the section (allowable stress `k_u σ_y` of the armour wires, the conductor's
  0.2 % proof stress, screens and sheaths; allowable strain `k_u ε_allow` of polymers; the
  buckling screen), the form of the umbilical capacity curve in Skeie et al. (2012, Sec. 3.1).
  `ISO_13628_5_UTILISATION_FACTORS` holds the factors MecLim II (Sec. 3.5, p. 19) reports for
  the practice of API 17E / ISO 13628-5: 0.80 normal operation, 1.00 load-controlled and 0.75
  displacement-controlled installation, 1.00 abnormal operation; 0.80 is the default. The
  standard's own text is not reproduced: its element stress definitions and load categories are
  not in a public source and are not implemented. The envelope records its `basis`; the default
  `"power_cable"` (DNV-RP-F401 armour factor, MecLim II component factor) is unchanged.

### Deprecated

- Planned change: the default `friction_model` of `compute_fatigue` (public and
  `cablexsec.analysis.fatigue`) changes from `"direction_tied"` to `"auto"` in the next release.
  On a planar curvature path the two are the same, bit for bit; on a non-planar path the
  default will solve the distributed helix model: higher friction-stress ranges than the
  direction-tied cap under near-circular rotation, equal or lower ones on flat paths (see Added).
  Until then the default is unchanged and warns (`DirectionTiedFrictionWarning`) on non-planar
  paths; pass `friction_model` explicitly to keep the current behaviour.

### Changed

- `read_cabledyn_sections` places the `Dec`/`Azi` tangents of a finite-EI line (any section
  with `EI > 0`) at their nodes, as CableDyn 0.4.2 and later write them: the node's own
  cubic-Hermite tangent. Every value was placed at the mid-arc of the chord between the
  neighbouring nodes, so an end node's tangent (the hang-off) sat half an element inside the
  line, and on a non-uniform mesh the interior tangents near a change of element length were
  shifted too. On the 128-element lazy-wave test run of CableDyn 0.4.2 (0.2 m elements at the
  hang-off) the curvature at node 2 departed from CableDyn's `Curv` by 52 % and at node 3 by
  15 %, so the read failed its curvature check where `Curv` channels were written near the
  hang-off; without them it returned a largest curvature 0.25 m from the hang-off 14.5 % high
  (9.59e-3 against 8.37e-3 1/m). Now all checked nodes agree to 2.5e-4. Chords of an `EI = 0`
  line stay at their mid-arcs. `run_sea_state_batch` takes the option as
  `SeaStateBatchOptions.tangent_placement`. The output files carry no version marker; `tangent_placement="chord"` reads
  finite-EI output of CableDyn 0.4.1 and earlier, the placement is recorded in
  `checks["tangent_placement"]` and the provenance, and a failed curvature check names the
  option. The test run `tests/data/cabledyn/lw128` is now CableDyn 0.4.2 output (its nodal
  tension recovery moves the mean sag-bend `Te` of how-to 10 from 1.75 to 1.33 kN); the v0.1.0
  run is kept as `lw128_v010`.
- Templates whose source describes a layer as a yarn serving now declare it as one. The Chae,
  Nam and Shim (2023) coiling cable's two yarn layers: full-slip EI 47.3 to 30.6 kN·m², stick EI
  876 to 860 kN·m², fixed-end EA 631 to 626 MN, solved torsional secant under the source's
  torque at pitch 1.8 m 106.5 to 89.5 kN·m², least armour surface stress under that torque
  −204.8 to −216.9 MPa. Thies et al. (2016) 30 kV outer PP yarn serving: fixed-end EA 328.3 to
  325.8 MN, free-end EA 125.7 to 123.2 MN, bending stiffness unchanged (it was already
  non-structural). Layers the source does not call a serving are unchanged, among them the 66 kV
  template's PPY beddings (the source's footnote names the material polypyrrole).
- The validation report's test-taxonomy lines record the passed count only: the deselected
  count is the rest of the suite, so a test added anywhere else marked the committed report
  stale without changing a gate. CI checks the committed report before the coverage run, so a
  stale report fails within about 15 minutes instead of after the 45-minute coverage run.

- Janocha et al. (2024) comparisons (66 kV template, gap R1b) are code to code: the source is a
  finite-element model that states no initial contact state. The Fig. 12b knee moments and stick
  slope are retired as targets: Fig. 12b gives about twice the Fig. 6 moment of the same model
  at 0.2 1/m and 50 kN, and its inset rises at a secant of 2,500–3,000 kN·m², above the
  1653 kN·m² plane-section bound of the published section. With the cores and filler profiles
  the Fig. 6 half-height is 0.38 / 0.37 kN·m (bounded / solved) against about 1.0 kN·m read
  from the figure. `doc/template_66kv.md` Section 4.1 and the benchmarks are rewritten
  accordingly.
- `MissingClampingStateWarning` now states that at low tension the friction moment and the
  bending hysteresis tend to zero without a declared clamping state, so a low-tension
  hysteresis is unreliable, and names the extrusion shrinkage among the declarations.
- Bounded core contact loads of the laid-core law: the inward resultant of a core (capstan load
  plus its share of the ring line force) is now reacted by its two neighbours, whose contact
  forces sum to the resultant over `sin(π/n)`, besides the ring share on the outer face. The
  earlier estimate put both parts on one contact and left out the reaction. On the 66 kV
  template at 50 kN the core load rises from 9.6 to 16.4 kN/m and the loop half-height with core
  friction from 0.19 to 0.25 kN·m (bounded; the solved route is unchanged). Results without
  `core_friction` are unchanged.
- The distributed helix friction model (`friction_model="distributed"`, and `"auto"` on
  non-planar paths) runs about three times faster on the 66 kV template. `compute_fatigue` on
  one thread, warm, an hour of the 66 kV template at 10 Hz (36 001 samples, both armour
  layers): 43.0 s before, 13.1 s after, against 3.2 s for the direction-tied cap (13.6 times
  before, 4.1 times after). Shipped fatigue examples: 1.14–1.59 times the direction-tied time
  before, 1.01–1.28 times after. All of the time was in the compiled increment solver: about
  1.28 million substeps of 0.1 `κ_on`, each with 3.8 Newton solves on average and several passes
  over the 556–580 helix nodes per solve. The substep rule, the equations, the convergence
  tests and the defaults are unchanged; a new compiled kernel solves each increment faster:
  Newton starts from the increment of the last substep (the substeps of a sample are equal
  steps, so this usually has the converged stick/slip pattern: 1.2–1.6 solves per substep), the
  factorisation of its Jacobian is kept between solves and eliminated from both ends towards
  the middle row, so a row that changes refactors only the rows between it and the middle, and
  the residual, bound and pattern of an iterate are formed in one pass. An increment is solved
  again from zero, as before, if that start fails. The previous kernel is
  kept as the reference: it runs without numba (`CABLEXSEC_DISABLE_NUMBA=1`) and the tests
  compare the two (tests/test_distributed_fast.py: planar, rotating, one-sided and full-slip
  paths, both bound forms, contact shape, pressure history, pressure factor and static/dynamic
  law, kept kinematics; the recorded values of the independent stick_check solver hold for
  both). Differences are round-off: wire stress at most 3.4e-15 of the largest, damage
  1.2e-15 (66 kV template) and at most 1.4e-12 relative (shipped examples, damages of order
  1e-12), energies 1e-15 of the largest, kept displacement and slip fields to round-off (both
  kernels fix the slide of a full-slip increment by the same rule, see Fixed). With a finite
  healing time of a friction law a node's heal and break-away events can move by a substep
  under any round-off change (on a 32-node helix with the uniform bound a relative change of
  1e-15 in the curvature moves the reference stress by 0.24 %; the kernels differ by 0.24 %
  there and by 4e-15 with the default nodes).

- The distributed friction model of the fatigue path accepts a constant stick modulus other than
  the layer's (one block of a thermal history, which recomputes `ξ` from its moduli) as the Yun
  coefficient `ξ = k/(E cos²α)` of that state. Before, any stick modulus other than the layer's
  was refused; a stick modulus that changes over the record still is.

### Fixed

- The capacity documentation states the copper plastic-limit mechanism: the conductor's 0.2 %
  proof stress at its extreme wire fibre, on the strand basis of its bending stiffness, is a
  component limit of the envelope (the `yielding` mode of the conductor layer), separate from
  conductor fracture. The 66 kV template page no longer says the envelope lacks it (it governs
  the template's capacity, Section 4.6).
- The interstitial filler of a trefoil bundle twists as separate lobes with their own warping,
  not about the cable axis. The solved section added the filler's polar moment about the cable
  axis to the twist rigidity, as if the filler were bonded to the ring with no warping, although
  it touches the cores and the ring through frictionless contacts that carry no axial shear and
  the three outer interstices are disjoint. `saint_venant_rigidity` (in
  `cablexsec.mechanics.trefoil_section`) solves the warping of the filler lobes on the section
  mesh; cores (about their centres) and bonded ring layers (about the cable axis) are unchanged.
  The filler term falls from 7.9 to 0.32 kN m² on the Chae, Nam and Shim (2023) cable, whose free
  torsional stiffness at the source's twist of 0.125 rad/m (unwinding) falls from 89.5 to
  81.9 kN m² at an armour pitch of 1.8 m and from 58.1 to 50.5 kN m² at 3.0 m; and from 3.6 to
  0.14 kN m² on the Su et al. (2026) 35 kV cable, whose torsional stiffness with the ends fixed
  axially becomes 199.5 / 177.0 kN m² (clockwise / anticlockwise; FE 209.5 / 201.2, test
  209.8 / 201.3; previously 203.0 / 180.5). The band of that anticlockwise check is widened
  from 12 to 13 % and stated. New validation row TQ-SV checks the warping solver against the
  series solution of a rectangle and the polar moment of a mirrored annulus sector.
- `export_bending_law` keeps the stick stiffness at every tension. Its geometric curvature grid
  started at `1e-4 κ_max` whatever the tension, so a slip onset below that floor (66 kV template,
  `κ_max` 0.4 1/m: outer armour 7.3e-6 1/m and inner 2.3e-5 1/m at 10 kN) was inside the first
  chord, and the law started partly or fully slipped: stick stiffness 5.80e5 instead of the
  section's 1.26e6 N m² at 10 kN (1.255e6 at 50 kN). With `slip_scale=True` the default reference
  is the smallest tension, so every scaled tension inherited the softened group laws (secant
  `M/κ` of the scaled law 0.46–0.65 of `compute_mkappa` below full slip at 50–200 kN). The grid
  now extends below its floor at the same ratio until its first point lies under half the
  smallest first-slip curvature (the neutral-axis onset times the compression-side factor
  `m(λ)`, and the smallest sub-element cap of the uniform progressive density), and carries the
  onset and full-slip curvature of every armour layer and of the laid cores at every exported
  tension and at the slip-scale reference. On the 66 kV template at 10–200 kN the exported and
  the scaled laws start at the section's stick stiffness to round-off and match `compute_mkappa`
  between grid points within 5.2e-4 of the moment; the core, progressive, pressure-factor and
  solved-contact exports start at `EI_max` too. `kappa_1_m` now holds more than `n_points`
  points (the base grid is unchanged); the linear spacing gets one stick point and the slip
  points, and warns when a transition is narrower than its step; a law that still misses the
  stick stiffness warns. The exported laws of the validation and benchmark rows (MX-1, MX-2,
  MX-scale) do not change at their printed precision.
- The sea-state batch takes the default armour friction model of `compute_fatigue`:
  `SeaStateBatchOptions.friction_model` is `None` by default (was an explicit
  `"direction_tied"`), so the batch and `cablexsec-batch` follow the planned change of the
  default and warn with `DirectionTiedFrictionWarning` on non-planar records, which they never
  did. The default is still `"direction_tied"`: results are unchanged, bit for bit, and the
  manifest and the assessment digests record the model the default resolves to, so they are
  unchanged too (new `cablexsec.analysis.fatigue.DEFAULT_FRICTION_MODEL`). Warnings of the
  worker processes (`workers > 1`) were shown on the workers' standard error only; they are now
  issued again in the calling process.
- `friction_model="auto"` warns when it keeps the direction-tied law on a non-planar path
  (`FrictionModelFallbackWarning`, once per layer, from `compute_fatigue` and
  `wire_stress_field`): a rate-dependent friction law, a rigid interface (`ξ ≥ 1`) or a stick
  modulus that changes over the record. Before, only the provenance limitations recorded it.
- `curvature_path_out_of_plane_ratio` (the planarity test of `friction_model="auto"` and of the
  default-change notice) judged the out-of-plane distance against the largest curvature, so a
  static bend hid cycles normal to it: a static 0.05 1/m with ±4e-5 1/m normal cycles was 8e-4
  out of plane (planar) and `"auto"` took the direction-tied law, whose cap follows the static
  bending direction. The distance is now judged against the smaller of the largest curvature and
  the largest distance from the mean curvature, the latter floored at 1e-2 of the largest
  curvature (so a nearly constant curvature written with six significant digits, each component
  within 5e-6 of the largest, stays planar at 7.1e-4 or less): that path is 0.08 out of plane,
  with ±4e-4 1/m cycles 0.8. Paths without a static offset (every bundled check: planar
  paths, centred ellipses and circles) keep their ratio; a planar path with a static offset in
  its plane stays planar. Over the 1 203 tests that reach the planarity test and the shipped
  fatigue examples and tutorials, no path other than the new checks changes its verdict.

- `read_cabledyn_sections` reads runs whose static solve refined the mesh of a finite-EI line
  (`adaptive_mesh`, `tensile_safety` or CableDyn's automatic refinement). It rejected them with
  "mesh of line … differs from the deck sections": `.static.out`, `.elements.out` and the
  per-line files list every node of the refined mesh, while the node channels name the deck
  nodes. The current CableDyn refines the 24 deck elements of the test deck `lw24.dat` to 43, so
  that run could not be read. The reader now locates the deck nodes among the mesh nodes by the
  `DeckNode` column of `.static.out` or the `# DeckNode` line of `.Line<L>.p.out` (failing both,
  by their arc lengths in `.elements.out`), reads the node channels at the deck nodes and the
  positions and segment tensions on the refined mesh, and records the mesh in
  `checks["mesh"]`. A deck that does not match the output is still rejected: the files must
  list the same number of nodes, every `DeckNode` numbering must name the deck nodes in order
  and agree with the others, and the deck nodes must lie at the arc lengths of the deck sections
  in `.elements.out`. Without `.elements.out` (no placement of the added nodes) the positions are
  read at the deck nodes, `tension_source="auto"` takes the node channels and
  `tension_source="segments"` or chord tangents are refused by name. `adaptive_mesh` without
  `.elements.out` is no longer refused: the files show whether the mesh was refined. Real
  CableDyn 0.4.2 output of the refined run is in `tests/data/cabledyn/lw24r.*`; from the refined
  positions the curvature meets every `Curv` channel within the default 5 %.
- `read_cabledyn_deck` reads the LINE TYPES tokens of CableDyn 0.4.x: `GJ|K` and `GJ|K|Bt`
  (torque–tension coupling and torsional damping, CableDyn 0.4.0) in the `GJ` column, and
  `EI|Bb`, `EI|-zeta|f` and `EI|-zeta|f1|f2` (bending and Rayleigh damping) in the `EI` column.
  It rejected them as "not numeric", so no deck using them could be read. The line type keeps
  the numeric `EI` and `GJ` and the extra parts as CableDyn resolves them
  (`bending_damping_Nm2s`, `mass_damping_1_s`, `torque_tension_coupling_Nm`,
  `torsional_damping`, `polar_inertia_kg_m`, and the columns as written in `ei_token` and
  `gj_token`), with CableDyn's checks (`0 < ζ < 1`, positive frequencies, damping and coupling
  only on a finite-EI type, `K² < EA·GJ`). The tables are now read by position, as CableDyn
  reads them, instead of by header name: a row without a number token is a header or units
  row, and the token count selects the row form. An 11-column LINE TYPES row is the 10 columns
  and the VIV `Cl`, so a header naming its last column `GJ` no longer turns `Cl` into a `GJ`
  (CableDyn reads `GJ` only from the 14-column row), and stock-MoorDyn headers (`Mass/m`,
  `Cd Ca CdAx CaAx`) read. END CONNECTIONS `TorsStiffness` `Free`, `Zero` or `0` now leaves the
  end free in torsion (any token but `none`, `-` and `0` restrained it, so a deck with `Free`
  was refused for want of `Twist` channels), `EndA`/`End_A` name End A, and the restrained ends
  follow the anchor-first swap (`CableDynLine.torsion_restrained_ends`). A finite-EI line from
  a `Fixed` point to a rod end is swapped as CableDyn swaps it, an `END` row closes the OUTPUTS
  list, a quoted name may hold spaces, exponents may be written with `D`, and a duplicate line
  type or line id is refused. A line solved in torsion whose type couples torque and tension
  twists as it stretches, so `read_cabledyn_sections` needs its `Twist` channels even when only
  one end is restrained, and `section_channels` lists `Twist`/`Torq` for a line restrained at
  either end. `tests/data/cabledyn/lw24g.*` is a CableDyn 0.4.2 run of a deck with these
  tokens.
- `read_cabledyn_sections` closes every output file it reads. The per-line `.p.out` and
  `.t.out` readers were left open after a read (and after a failed one) until garbage
  collection, which raised `ResourceWarning`. The files of one run must now also end on the same
  row: a per-line file with more or fewer rows than the main table is rejected ("different row
  counts"), where before the extra rows were dropped when the chunk boundaries hid them.
- `read_cabledyn_deck` reads the OPTIONS record forms of CableDyn: `value keyword [commentary]`,
  positional records with the model first and the keyword after the numbers
  (`uniform 0.0 0.6 0.0 current` was read as a keyword `0.0` with the value `uniform`), the
  keyword-first `dynamic_solver` row, and a ` - ` description. The last record of a keyword
  wins, as in CableDyn (the first won), and the aliases `gravity`, `WtrDnsty`/`water_density`,
  `water_depth` and `dt` set `g`, `rhoW`, `WtrDpth` and `dtM` (they were ignored, so a deck
  that set the water density as `WtrDnsty` got the default 1025 kg/m³ in the hydrostatic
  pressure).
- `run_sea_state_batch` checks and digests `cable_source` before it reads any run; a missing
  file failed only after every section of every run had been computed.
- The global–local guide (Sections 1, 2, 2.1 and 7), `ARCHITECTURE.md` and the CableDyn reader
  describe CableDyn 0.4.x: node tension, curvature and bend moment recovered from nodal
  equilibrium (since CableDyn 0.2.0), the signed section-axis components `CurvX`/`CurvY` and
  torsion of lines restrained at one end (since 0.4.0), and the node tangent of finite-EI lines
  in the `Dec`/`Azi` channels (since 0.4.2; see Changed). The `compute_mkappa` docstring states that
  `tension_N` is the true wall tension, as in `compute_fatigue`.
- The slip of the distributed friction model over a full-slip increment (every helix node on its
  bound) is now unique. The equations fix the forces, the tractions and `u − p` there, but not a
  slide common to the displacement `u` and the slip `p` of all nodes; the solver left one node
  exactly on its bound, chosen between candidate nodes at round-off, so the kept slip field
  (`DistributedFrictionHistory.kinematic_field`) and the armour slip of
  `kinematic_history(..., friction_model="distributed")` (`slip_m`, built from `p`; `slipping`,
  from the step slip) could move with any round-off change. On a 32-node helix in full slip a
  relative change of 1e-15 in the curvature moved the slip by 12 % of the largest step slip
  (uniform bound), and the pitch-mean slip jumped by up to 2.7 % of the step slip from one
  full-slip step to the next. The increment now takes the slide that gives the step slip a zero
  pitch mean, or the admissible slide nearest to it (every node keeps slipping in the direction
  of its traction): a uniform curvature puts no net axial force on a wire over a pitch and slides
  no wire as a whole, the convention of the pointwise laws in `kinematic_history`, which keep the
  pitch mean of the slip over a step in which no station sticks. Partial slip is unchanged (the
  stick nodes fix the slide). Measured on the 66 kV template with `kinematic_history`
  (`friction_model="distributed"`, planar and rotating bending at 0.01 and 0.05 1/m, 121 and 601
  samples, 0–100 % of the samples in full slip): the armour slip changes by at most 4.9 µm, 0.3 %
  of the station slip amplitudes of 0.3–2.0 mm; no `slipping` flag changes; the pitch-mean slip
  no longer moves over full-slip steps in planar bending (it moved by up to 0.024 µm per step);
  wire stresses and friction work change by at most 1.7e-15 relative (round-off). The Menard
  2023 cable does not reach full slip in these histories and is unchanged. No validation row
  and no documentation example output changes (the examples use the direction-tied law). Tests:
  tests/test_distributed_slide.py.

- Provenance of the default armour coefficient `mu_on_hdpe = 0.20` (and `mu_on_steel = 0.20` of
  `steel_saevik2011`); the values are unchanged. The source is Dai, Sævik and Ye, Marine
  Structures 55 (2017) 137–161, Sec. 4.3.1, p. 156 (cited as Sec. 4.3 before): an estimate
  between anti-wear tape and inner armour and between two steel armour layers "at 20 MPa
  internal pressure", quoted from Sævik (1992). The interface is an anti-wear tape, not HDPE,
  and the 20 MPa is an internal pressure; the contact pressure is 2–4 MPa (Dai, Sævik and Ye
  2018, p. 428), where the measured dynamic coefficient of armour wire on in-service tape is
  0.32, 0.20 and 0.18 at 1, 4 and 8 MPa.
- The documentation said that the direction-tied and distributed friction laws give the same
  stress ranges under biaxial bending ("within 10 %", "fatigue is not affected"). That holds for
  the template interface from about four times the slip onset; on a near-rigid interface the
  direction-tied ranges on a circular orbit are 0.85–0.97 of the distributed ones from 1.5 to 8
  times the onset. The local analysis guide (Section 9, with the measured table), the concepts
  page and the notes of `wire_slip_history` and `thermal_coupling` now give the measurements.
- The pitch mean of the direction-tied friction stress is documented: its bounds integrate the
  capstan growth from the neutral axis (zero friction stress there), so in planar full slip the
  stress `σ0 (exp(λx) − 1)` has the pitch mean `σ0 (sinh(x₀)/x₀ − 1)`, `x₀ = λπ/2`, about `λπ/12`
  of the peak tensile friction stress (1.1 % on the 66 kV inner armour, 0.054 MPa), while the
  periodic field of a uniformly bent pitch, as in the distributed model, has zero mean (tested).
  The offset is kept: a zero-mean condition couples the wires, which the independent
  spring–sliders of the law do not, and a uniform shift would move wire stresses out of their
  own bounds. Removing it would leave the moment unchanged and change the largest per-wire
  stress range by less than `1e-4` (planar, one-sided and rotating cycles); it raises the mean
  stress of every wire by the offset.
- `anchored_heat_profile` documents that its anchoring ratio is solved with a constant friction
  bound while the sectional heat it multiplies carries the signed capstan bound. On a uniformly
  bent section the two bounds give cycle works within 0.25 % (66 kV template armour, 1.2 to 32
  times the slip onset); near an anchor, where the mean wire force reaches about `0.85 √a` bounds
  per radian, a capstan term would move the local bound by up to `λ 0.85 √a`, which the
  constant-bound solver does not resolve. A capstan bound breaks the translation invariance of
  its exact chain projection, so the solver is unchanged.
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
