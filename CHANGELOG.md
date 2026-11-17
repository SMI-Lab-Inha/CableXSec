<!-- SPDX-License-Identifier: Apache-2.0 -->

# Changelog

All notable changes to CableXSec are recorded in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

### Deprecated

- Planned change: the default `friction_model` of `compute_fatigue` (public and
  `cablexsec.analysis.fatigue`) changes from `"direction_tied"` to `"auto"` in the next release.
  On a planar curvature path the two are the same, bit for bit; on a non-planar path the
  default will solve the distributed helix model: higher friction-stress ranges than the
  direction-tied cap under near-circular rotation, equal or lower ones on flat paths (see Added).
  Until then the default is unchanged and warns (`DirectionTiedFrictionWarning`) on non-planar
  paths; pass `friction_model` explicitly to keep the current behaviour.

### Changed

- The validation report's test-taxonomy lines record the passed count only: the deselected
  count is the rest of the suite, so a test added anywhere else marked the committed report
  stale without changing a gate. CI checks the committed report before the coverage run, so a
  stale report fails within about 15 minutes instead of after the 45-minute coverage run.

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
