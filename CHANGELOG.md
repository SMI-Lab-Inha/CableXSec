<!-- SPDX-License-Identifier: Apache-2.0 -->

# Changelog

All notable changes to CableXSec are recorded in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- CableDyn range graphs and VIV lift files are read: `read_cabledyn_range_graph`
  (`<root>.Line<L>.range.out`, OpenFAST `<root>.CD.Line<L>.range.out`) and
  `read_cabledyn_lift` (`<root>.Line<L>.V.out`), each checked column by column
  against CableDyn's layout. On the new `lw24v` run the range extremes equal the
  extremes of the node channels exactly, as CableDyn documents.
  `read_cabledyn_sections` checks either file it finds against the run (mesh,
  range window) and records its digest.
- The hang-off's turn about its own tangent reaches the material frame of a line
  CableDyn solves in torsion with End A restrained: a `Body<N>` holder turns it
  with its `Body<N>Rx/Ry/Rz` channels, a `Fixed` one holds it, and a vessel or
  platform holder takes the new `hang_off_rotation` (attitude matrices, a roll
  history, or `"none"`; also a `SeaStateRun` field). A line restrained at End B
  only turns its free End A by `-Twist<L>` against End B's frame. Before, the End
  A frame was transported in time without twist: a line turned rigidly with its
  body by 40 deg at a 60 deg hang-off read its components turned by 35 deg.
- `CableDynLine.end_connections` and `end_kinds`: the END CONNECTIONS rows and
  the holders of the ends, as CableDyn reads them.
- `read_cabledyn_sections(curvature_source="components")` takes CableDyn's
  `CurvX`/`CurvY` at the deck nodes around each section, on CableDyn's section
  axes: exact, with no node positions or angles.
- `SeaStateBatchOptions` takes the reader's curvature check tolerances
  (`curvature_check_rtol`, `curvature_check_end_rtol`, `curvature_check_atol_1_m`):
  a coarse run that fails the check at the default could not be read by the batch.
- Distributed form of the laid-core law: `core_friction_history(...,
  friction_model="distributed" | "auto")` solves the cores and the filler profiles with the
  distributed helix model of the armour (the axial force of each core as a field along its
  pitch, slip of every helix node in the direction of its own traction, no bending direction);
  `"auto"` keeps the direction-tied law on planar paths, as in the fatigue route, and falls back
  with `FrictionModelFallbackWarning` on a rigid interface. `compute_neutral_axes` takes
  `friction_model="auto"` for the armour and gives the cores the armour's model unless
  `core_friction_model` says otherwise; `SectionLaw.from_cable(friction_model="distributed")`
  solves the cores with it too. Checks CF-D (targets registered before the run): planar
  reduction of `"auto"` 0, lattice rotation 5.7e-15, dissipation and balance, μ = 0 bit for bit,
  stick tangent 2.0e-6, compiled kernel 3.4e-16; the full-slip moment misses the continuous
  closed form by 1.02e-4 against the registered 1e-4 at the default stations (FAIL, documented:
  station quadrature; the model reproduces the station sum to 1e-10). Under non-proportional
  curvature the distributed core law takes 1.3 (random walk) to 2.3 (circular orbit) times the
  friction work of the direction-tied one on the 66 kV template.
- `core_load_history`: the core and filler-profile contact loads of the contact model over a
  tension history (bounded estimate exact, solved model exact up to 33 tensions).

- One exported bending law per friction group, each with its own scale table
  (`BendingLawExport.slip_group_laws()`, `SlipGroupLaw`, `to_group_csv`), for a global model
  that sums the group laws `s_g(T) B_g(κ / s_g(T))` instead of carrying one scale. Rows
  MX-group-B and MX-group-E (targets registered before the run): on the 66 kV template with
  laid cores, filler profiles and 0 / 20 / 40 MPa armour winding stress, 10–200 kN, the summed
  group laws reproduce the exported backbone within 4.6e-4 (target 0.5 %) and the cycle work
  within 0.06 % (target ±1 %); one scale for all groups misses by up to 5.5 % and 7 % with
  clamping (MX-group-1, reported).

- Reported rows SV11-SPLIT, SV11-REACH and SV11-EQ88: where the SV11 termination increment
  comes from. With the bellmouth curvature imposed uniformly from the end-fitting face the
  end-fitting anchor alone moves the SV11-PLANE median from 1.00 to 1.01 and SV11-GJ from 1.24
  to 1.27 (registered correction 1.62 and 1.80); with the armour anchored 20 m upstream instead
  the corrected ranges do not change. The increment is the anchored chain's response to the
  curvature gradients at the bellmouth contact, with the stuck straight inlet as the anchor, on
  a curvature field that does not carry the anchoring. At 0.5° and 1° the pipe sticks and the
  correction is zero (R 0.45, 0.54), so SV11-PLANE cannot pass through any termination
  correction. Eq. 88 confirms Sævik's bellmouth end (1.72 m: 3.90 m and 15.9°; Gjøsteen et al.'s
  1.79 m gives 3.79 m and 17.0°). The benchmark page adds the rig, stick and shear-stack
  sensitivities (`check_saevik2011.py OUTPUT --sensitivity`): the corrected median moves from
  1.62 to 1.25 when the rocker turns about the end-fitting face, the sectional one by 9 %; with
  the Yun shear stack cut at the nearest helical layer below (ξ 0.52 → 0.88) the small-angle
  sectional ratios reach the band. No registered row, input or default is changed; the SV11
  harness beam takes its rig as a `Rig` value (the registered rig by default).

- Validation rows DAI20-KAPPA and DAI20-COULOMB (Dai, Sævik and Ye 2020, steel-tube umbilical
  in a bellmouth; targets registered before the run). The SV11 rig beam in the measured
  bellmouth (0.1 1/m, 3 m, 200 kN), bending stiffness identified on the S2 curvature at 4°
  (59.1 kN m²), predicts the angles at which the contact front reaches S2 and S3 0.87° and
  0.89° early (band ±1.0°, PASS; mesh-converged; the rig pivot, inlet and rigid lengths are
  declared assumptions with their sensitivities on the benchmark page). The sectional
  full-slip range of the helical tubes with the source's contact line forces reproduces the
  source's 2D values within 1.1 % and misses the measured Coulomb ranges by +89 / −55 %
  (radial) and +114 / +163 % (radial plus hoop): the documented radial-only limit.

- Validation rows Z-EI and Z-ID (targets registered before the runs): the four-point loops of
  Zeidan (2025, PhD thesis, Ch. 5) and Zeidan et al. (2026, IJSS 338, 114089) on a 112 mm
  three-core cable (`cables/zeidan2025.yaml`, 75 + 82 armour wires of 3.5 mm, zero tension),
  digitised in plotted order from the native PDF polylines
  (`validation/scripts/digitise_zeidan2025.py`) and simulated in the test's arrangement with
  its own reduction (`validation/scripts/check_zeidan2025.py`). The plotted curvature is the
  loading-point formula applied to the mid-span deflection, 26–29 % above the central
  curvature of the source's Fig. 18. Both rows FAIL: the measured stick secant (141–240 kN m²)
  is 12–20 times the packaged no-slip EI_max (12.0 kN m², armour ξ 0.009/0.016, because the
  series shear stack counts the 8.75 mm EPR of the cores), and no initial pressure reaches the
  measured 40 mm work (largest 41.9 against 445.1 J/m). In a diagnostic with the armour ξ from
  the bedding alone the work of all four amplitudes is within ±11 %, the moment range 33–48 %
  short (slipping slope near EI_min), and the laid cores change the work by less than 0.2 %.
  The laid-core law stays off by default (registered rule not met).
- Scorecard rows AW-DONG25 and AW-ZT14 (code-to-code, stated, not gated): the anchored-wire
  clamp solver against the published FE of Dong et al. (2025) (anchor 156.4 against 156.9 MPa,
  peak 207.8 against 204.5 MPa) and Zhu and Tan (2014) (1.19 / 1.70 against 2.22, an FE whose
  far field is not in slip), so far reported only on the benchmark page.

- `compute_neutral_axes(..., core_friction=...)`: the biaxial section moment adds the
  stick-slip law of the laid power cores, their internal wire rings and the filler profiles
  (`core_friction_history` over the same curvature history), so that a global model's
  `(kx, ky)` history gives the section moment of the full accounting, not only of the armour.
  The core contact loads are those of the bounded model at the reference tension (the first
  sample of the tension history), as in `compute_mkappa`; a planar history gives the moment
  of `compute_mkappa` with the same options to round-off (tests at 0 and 90 degrees).
  `NeutralAxisResult.core_moment_Nm` holds the core share and `ei_max_Nm2` adds the core stick
  stiffness. Without `core_friction` the result is unchanged bit for bit.

- Su et al. (2026) 35 kV torsion checked term by term against the source's analytical model
  (targets registered before the run): armour torque 193.3 / 170.8 kN m² clockwise /
  anticlockwise against its Fig. 19 shares 193.0 / 172.3 (band 2 %), and at 1000 N m the
  separations and pressures of its Table 11 (armour 1–2 and 3–4 open anticlockwise; pressures
  −7 to +11 %, band 20 %). The benchmark register decomposes the −12.0 % anticlockwise gap to
  the FE and the test: the source's model ties the ring to the cores (Eq. 36), so it compares
  with the filled section, not the filler-free one; its polymer share is twice the thin-wall
  terms of its own Eq. 11 (12.0 against the exact 6.1 kN m²); the rest is the armour's direction
  dependence (11.3 % here, 10.1 % in the source's model, 4.0 % FE, 4.1 % test), whose FE split
  and test conditions are not published. No input is changed and the 13 % band stays.

- Validation row CH-GJ (stated, model-to-model, registered before the run): free torsional
  stiffness of the Chae, Nam and Shim (2023) coiling cable at the source's twist of 0.125 rad/m
  (unwinding, zero tension), against the range of the source's three cross-section tools
  (Table 5; no measurement). Armour pitch 1.8 / 3.0 m: packaged straight cores 81.9 / 50.5 kN m²;
  cores laid S as in the source at an assumed pitch of 2.5 m 61.1 / 35.0 and of 3.0 m
  64.4 / 37.3 kN m² (tools 59.6–83.2 / 34.3–44.7). The packaged file and the benchmark page state
  that its straight cores raise the free torsional stiffness by 1.3–1.4 times, and why the
  source's closed form (Eq. 14 as printed) is not a reference.

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

- The laid-core and filler-profile contact loads follow the tension history by default:
  `compute_neutral_axes(..., core_loads="follow")` with `tension_N` or `contact_tension_N`, and
  `compute_mkappa(..., core_loads="follow")` with `contact_tension_history_N`; before, they were
  held at the first sample, which stays available as `core_loads="reference"`. Check CF-T
  (registered before the run): a tension cycle of 10–70 kN inside bending cycles now meets the
  slip-scaled exported law within 0.49–1.19 times its fixed-tension residual (target 1.25);
  with the loads held it missed by up to 20 % of the moment range.

- The default slip-scale table of `export_bending_law(..., slip_scale=True)` starts at zero
  tension, with the caps of the contact model there (zero without clamping, the clamping caps
  otherwise; row MX-group-0), so the scale runs to zero instead of ending at the lowest
  exported tension. An explicit `slip_scale_tensions_N` is used as given.

- CI runs the tests in parallel: the test matrix, the NumPy reference path and both coverage
  passes run on one pytest-xdist worker per core (`pytest-xdist` joins the `dev` extra).
  Every concurrent process writes its own numba cache: numba numbers a new cache entry from
  the index it last read, so two processes compiling one kernel for different signatures at
  once can overwrite each other's entries. `release/check_coverage.py --numprocesses N` and
  `run_validation.py --jobs=N` take a worker count; `--jobs` evaluates the independent row
  groups in worker processes and runs the three taxonomy groups concurrently. The coverage
  run and the validation report check run side by side in two jobs, and the required check
  "Coverage and validation report" passes only when both pass. The tests, the 95 % threshold,
  the kernel coverage pass, the rows, the report and its staleness check are unchanged.

- The rows that compare with the measured loops of Ménard and Cartraud (2023), BM1-loop,
  BM1-loop-uniform and BM1-ID, are stated, not gated (status INFO, the verdict against the
  registered target kept): the test is an unreliable reference. The benchmark register states the
  evidence: the source's own limits (one specimen, end effects not investigated, achieved
  curvature ranges off target), its table and figure disagreeing on equal-span ranges (−15 %
  against +4 and +11 %), and an independent FE reconstruction (Kwon et al. 2026) that does not
  match. Targets and results are unchanged.

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

- CableDyn deck reader mirrors CableDyn's record checks: a section header CableDyn
  does not know is refused with the deck line (was skipped silently), as are a
  record over 512 bytes without a comment, a NUL or non-ASCII whitespace
  character outside a comment, a subnormal number, and END CONNECTIONS rows
  CableDyn refuses (`none` or `-` torsion stiffness, zero `Ez`, a zero or
  parallel `Nx` at a restrained end, two rows for one end, an unknown line, an
  `EI = 0` line). Bytes that are not UTF-8 no longer stop the reader.
- On a line CableDyn solves in torsion the reconstructed section axes start at
  End A's torsion normal `Nx` (End B's on a line restrained at End B only), as
  CableDyn's `CurvX`/`CurvY` axes do; they started at the reference vector and
  stood 90 deg away on the coupled VolturnUS-S fixture `vu_torsion`, whose
  components now agree within 0.3 to 1.2 %. `frame="section"` is refused on such
  a line. An OpenFAST root holding the glue-code `<root>.out` beside
  `<root>.CD.out` was refused as two runs; the CableDyn file is read.
- `read_cabledyn_sections` refuses a standalone deck without `TMax` or with
  `TMax 0` (a static run, one row) and a standalone record that ends before
  `TMax` (a run that stopped early), naming the cause; an OpenFAST `.CD.out` is
  no longer held against the deck's `dtM` (OpenFAST rounds it to whole glue
  steps) or `TMax`. Malformed output rows, non-numeric profile cells and
  non-text files raise `CableSchemaError` naming the file and line instead of
  `ValueError`, `IndexError` or `UnicodeDecodeError`; NaN or negative curvature
  check tolerances, a non-integer `line_id` and non-numeric section arcs are
  refused.
- Fallbacks are an error or a warning, never silent (`tests/test_silent_fallbacks.py`).
  `fretting_regime.range_at_cycles` bisected on [1e-6, 1e6] MPa and returned the bound when the
  life lay outside it (1e12 Pa for a life shorter than the curve's at 1e6 MPa, 1 Pa for a NaN
  life); it refuses a life that is not finite and positive or outside its bracket. The
  wire-yield guard of `compute_mkappa` swallowed every `ValueError`, so a bound that could not
  be evaluated left no trace; it now warns (`ExtrapolationWarning`), and a cable without a
  yield-stressed armour layer, which has no bound, is skipped as before. The bending-test
  inversion returned its closest cached point as converged when it had no new point to try; it
  bisects the bracket first (a target inside a step a few ulps wide, where the
  interpolants round onto cached points, is resolved to the step) and raises
  `CableSolverError` with the miss only when no point is left. A kinematic history whose
  laid cores have no friction law held them to the cable axis silently; it warns and records
  it in the provenance.
- `compute_sheave` and the rig-test M-kappa loops ran their internal calls under
  `warnings.simplefilter("ignore")`, dropping a wire-yield or clamping warning about the
  caller's own curvature; each distinct warning is now relayed once
  (`provenance.relayed_warnings`).
- The series shear stack of the armour stick coefficient ξ (`xi_for_layer`, Yun et al. 2020)
  stops at the layer's support. Walking inward, it adds each concentric cylindrical layer with
  a modulus (interstitial fillers still skipped) and stops at the first helical layer inside and
  at the laid power cores; no layer of an offset core is in it, and a single core on the axis
  keeps its concentric insulation (`yun_shear_stack` returns the layers). Before, it counted
  every layer inside the armour: the insulation of the offset cores of three-core cables and,
  in flexible pipes, the pressure sheath under the steel pressure armour. ξ before → after:
  Zeidan 2025 0.009 / 0.016 → 0.832 / 1.000 (EI_max 12.0 → 311.1 kN m²); Sævik 2011 tensile
  armours 0.52 / 0.51 → 0.88 / 0.89; Witz 1996 0.141 / 0.138 → 0.422 / 0.466; Ménard 2023
  0.208 / 0.248 → 0.686 / 0.888 (EI_max 59.1 → 189.8 kN m²); IEA 66 kV 0.775 / 0.857 →
  0.934 / 0.992; the 33 kV and 220 kV three-core cables, Thies 2016 and Chae 2023 likewise; the
  inner armour of the single-core 33 kV cable and of CIGRE TB 880 is unchanged, their outer
  armour becomes 1 or 0.985. GJ, EA and the full-slip EI_min do not change. The rule, the
  re-evaluated rows and held-out predictions were registered before the run
  (doc/benchmarks.md, "Yun shear stack cut at the support"); no band or target changed. Z-EI
  FAIL → PASS and SV11 (both diagnosed on, not independent): SV11-PLANE 1.62 → 2.17 (sectional
  1.00 → 1.19, 12 of 14), SV11-GJ 1.80 → 2.07 (1.24 → 1.27), both still FAIL; Z-ID still FAIL
  (P0 now identifiable, moment range −33 to −46 %). Held out: WITZ-ID 300-bar energy −14.2 →
  −12.8 % (still FAIL on the width), BM1-loop work +8.8 → +11.8 % (stated, not gated; its
  registered 10 % target now fails), BM1-ID (a) +26.0 → +28.5 %; each as predicted. The ring
  stack of a core's wire screen (`compute_core_screen_fatigue`) stops at another resolved wire
  screen inside, the same rule (no packaged cable changes). An armour on the armour inside it
  now has ξ = 1: the distributed helix model (`friction_model="distributed"`) takes that rigid
  interface with the along-wire chain's `RIGID_K_HAT` instead of refusing it, while
  `friction_model="auto"` and the frictional heat keep the direction-tied cap for it as before.
  `SectionLaw.from_cable(..., distributed_max_increment=0.02)` substeps the distributed law
  more finely than the fatigue default (0.1), because the test solver bisects along legs: on
  the Ménard cable in three-point bending the stiffer stick left a sampling residual of 8.5e-8
  at 0.1 (2e-9 at 0.02). The README and quickstart outputs, which had drifted before this
  change (peak 548 N m and 219.8 J/m printed against 541 N m and 212.0 J/m computed), are
  regenerated.

- `run_sea_state_batch` with `workers > 1`: each worker process writes a numba cache of its own
  instead of sharing the caller's. numba numbers a new on-disk cache entry from the index it
  last read and saves the index before the data, so two workers compiling one kernel for
  different signatures at the same time could leave the index pointing at the other's machine
  code, which a later process would load. Worker caches are slot folders `slot-<i>` under
  `cablexsec/numba-batch` in the user cache directory (`CABLEXSEC_BATCH_CACHE_DIR` overrides
  it), kept for later batches. Each worker locks its slot itself for as long as it runs (so
  the slot stays taken even if the calling process ends first), a batch beside another takes
  the next free slots, and a worker without a free slot or a writable folder gets a temporary
  cache. Results are unchanged, and runs go to whichever worker is free, as before.

- An interface of the radial section is open when its contact carries no line force, as the
  contact solver decides it (`RadialSectionResult.interface_open`, `open_interfaces`); the
  how-to recipe and the coiling check counted gaps above 1e-9 m. A closed contact reports
  the round-off of the linear solve as its gap: about 1.5e-10 m on two interfaces carrying
  1.1 MN/m of the 66 kV template at 100 kN and a sealed 2 MPa, within the solver's own
  complementarity tolerance in metres (2e-10 to 4e-10 m there). macOS runs crossed 1e-9 m and
  printed "open gaps 1" against the page's 0.

- `SectionLaw.from_cable` reads the cable's own materials (its `materials:` block) when the law
  is built and every time it is evaluated. Outside `simulate_bending_test` it read the library
  instead, and a layer whose material the library lacks silently dropped out of the stiffness
  (the Zeidan cable: a small-curvature secant of 4.48 instead of 6.99 kN m²).
- The public 66 kV template (`iea_task49_66kv.yaml`) and its derived 300 mm² section model the
  core screens as the source's section model does (Janocha et al. 2024, Table 3), not as solid
  copper annuli. Table 1 labels the 2.00 mm conductor screen copper tape and the 3.15 mm
  insulation screen copper; read as solid copper, the insulation screen alone held 602 mm² of
  copper per core and the cable weighed 76.33 kg/m against the published 55.76 kg/m
  ([IEATask49] Table 13; 547 N/m in air, the source's Table 5), +36.9 %. Table 3 merges the
  conductor screen into the 630 mm² conductor body and the insulation screen into the XLPE
  insulation. The layers `cu_conductor_screen` and `cu_wire_screen` become `conductor_screen`
  (loose, semiconducting, XLPE properties), `insulation_screen` (semiconducting, XLPE
  properties, carries load with the core) and `cu_tape_screen` (a loose 0.10 mm copper tape of
  assumed thickness, the metallic screen of the electrical and thermal analyses). Against the
  published values: mass 55.41 kg/m (−0.6 %; 300 mm²: 40.45 against 36.66 kg/m, earlier +55 %),
  EA 638.5 MN (−3.0 %, was 636.8), full-slip EI 21.74 kN·m² (−49 %, was 20.98; the source's own
  figures disagree on this value and no component bending of its section reaches 42 kN·m²),
  armour no-slip EI 1215 kN·m² (was 1260; the insulation screens add shear compliance to the
  armour support). The semiconducting screens are thermal resistance: the conductor reaches
  95.3 °C at 1093 A (88.9 °C with the copper annuli), and the solved armour contact at 50 kN
  rises (Fig. 6 half-height 0.115 against 0.081 kN·m armour only). The energised tables of
  doc/template_66kv.md Section 4.2 keep the earlier values as a record, with the regenerated
  reported state; benchmark TQ6 now differs by 1.5 % (band 1 %, FAIL), because the
  fixed-radius model now carries the insulation screens that the solved section already did.

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
- CableDyn reader, remaining differences from CableDyn's deck reader:
  - `EQUIVALENT BUOYANCY` (also `EQUIVALENT SECTIONS`, `BUOYANCY SECTIONS`) was ignored, so a
    line type showed the `Diam` and `MassDenInAir` of its LINE TYPES row, not the values CableDyn
    ran with. As in CableDyn, a row now replaces the diameter and sets the mass to
    `rhoW π D²/4 + w/g` (deck `rhoW` and `g`); `line_types_diameter_m`, `line_types_mass_kg_m`
    and `equivalent_submerged_weight_N_m` keep the row's values. CableDyn 0.4.2 gives
    byte-identical output for the `lw24` deck with the row `buoy 0.30 -100.0` and for the deck
    with `buoy` written with this diameter and mass. Its checks are mirrored (three columns, a
    known type, one row per type, `D > 0`, a non-negative mass).
  - Records are cut and split as CableDyn cuts and splits them: `--` starts a comment at the
    start of a record or after a separator (a LINES row ending in `-- comment` was refused for
    its column count), a record holding `---` is a section header named by its other
    characters, a byte-order mark is dropped, a section header met again continues the section
    (it was refused as a duplicate), and a token opening with a quote runs past its closing
    quote to the next separator. `#` and `!` still cut a record anywhere, inside quotes too, as
    CableDyn does (its `doc/options.rst` says so for file paths). OUTPUTS rows split on spaces,
    tabs and commas with the quotes dropped, and only an unquoted `END` closes the list (CableDyn
    reads a quoted one as a channel name).
  - LINE TYPES rows with a negative `Cd_n`, `Cd_t`, `Ca_n` or `Ca_t`, or a 14-column row whose
    `GAs`, `GJ`, `Irt` and `Irn` are neither all zero nor all positive, are refused, as by
    CableDyn.
  - A line restrained in torsion at one end only is solved in torsion by CableDyn only when
    every section gives `GJ`, End B is `Fixed`, a dynamic run is force-blended and a restrained
    End A is on a `Fixed` rod's end or a Rigid6 body if on either; otherwise it runs as without
    the torsion columns. `CableDynLine.torsion_solved` and `torsion_note` follow these rules (the
    deck's BODIES and RODS rows are read for them), `section_channels` lists `Twist`/`Torq` only
    on a solved line (CableDyn refuses them on any other), and `checks["torsion"]` records the
    result. A solved line restrained at one end without torque–tension coupling twists in the
    dynamics (torsional waves of its polar inertia, the roll of the restrained end's vessel or
    body), so a dynamic run of it is now refused without its `Twist` channels, where it was
    read twist-free: `tests/data/cabledyn/lw24k.*`, the `lw24g` deck with `K = 0` run by
    CableDyn 0.4.2, twists by up to −1.7e-3° in 1 s (0.18° in a 30 s run). Its statics stay
    twist-free and are read without them.
  - The anchor-first swap of a line from a `Fixed` NodeA to a `Free` or `Connect` NodeB applies
    only when NodeA is not above NodeB, as in CableDyn; a weight hung from an overhead point
    keeps its written order. A line end on a rod fixed to a body is a point of that body, and a
    zero-length rod is the point it collapses to.
- `run_sea_state_batch` compares the deck's `GJ` and torque–tension coupling `K` with the
  section's lay torsional stiffness `K_tt` and coupling `K_et` (on a 14-column LINE TYPES row;
  the check compared `EA` and `EI` only). `GJ` must lie within `torsion_rtol` (new option,
  default 25 %) of `K_tt`, and the coupling ratio `K/sqrt(EA GJ)` within `torsion_rtol` of
  `K_et/sqrt(EA_fixed K_tt)`. A mismatch stops the batch, as one of `EA` or `EI` does, unless
  `allow_property_mismatch=True`, and the manifest records both values and the ratios. The
  tolerance is wider than `property_rtol` because `K_tt` leaves out the components' own torsion
  and the radial relaxation of the layers.
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
