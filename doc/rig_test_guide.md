<!-- SPDX-License-Identifier: Apache-2.0 -->

# Rig-test simulation and strain-gauge planning

Modules:

- `cablexsec/analysis/rig_test.py`: `simulate_rig_test`, `RigProtocol`,
  `RigSetup`, `GaugeLocation` and the result objects.
- `cablexsec/mechanics/clamp_layer.py`: the anchored-wire friction solver and
  the clamp closed forms.

Tests: `tests/test_clamp_layer.py` (closed forms) and `tests/test_rig_test.py`
(limits and decomposition). Example: `examples/rig_test_gauge_plan.py`.

The function is a planning tool for a combined-loading rig: tension plus
single-plane cyclic bending, optionally at a raised temperature, on a specimen
clamped at both ends. It predicts what bonded strain gauges on the armour
wires would read, which gauges carry the largest and most informative signals,
and how those signals depend on the friction coefficient and the clamp. It is
not a validated prediction of a particular rig; it is the model the
measurements are meant to test.

## 1. Quick start

```python
import cablexsec as cx

cable = cx.load_cable(cx.inter_array_three_core_yaml_path(), geometry_override=None)
protocol = cx.RigProtocol.cyclic(tension_N=100e3, curvature_amplitude_1_m=0.2,
                                 n_cycles=3)               # R = 5 m
setup = cx.RigSetup(free_length_m=4.0, profile="uniform", clamp="anchored")
res = cx.simulate_rig_test(cable, protocol, setup)
print(res.summary())
g = res.gauge("outer_armour", 90.0, 0.05)    # nearest candidate gauge
print(g.ranges, g.range_sectional, g.range_by_mu)
```

Run time is about 5 s per case on one thread for this cable (two layers,
160 wires, about 100 candidate gauges, three friction coefficients) with
the compiled kernels (`pip install -e .[fast]`). Without `numba` the
anchored-wire solver runs on a NumPy path with the same results but is one to
two orders of magnitude slower.

## 2. Inputs

**Protocol** (`RigProtocol`): sample times, true wall tension, and either the
signed curvature at the reference station or the rotation of the clamp at
`x = 0`; optionally a uniform specimen temperature. `retain_from_s` selects
the samples over which ranges are taken. `RigProtocol.cyclic(...)` builds the
usual test: a tension ramp from zero over one period with the specimen
straight, then `n_cycles` sinusoidal bending cycles, ranges over the last
cycle. The tension may cycle with the bending period (`tension_amplitude_N`),
lagging it by `tension_phase_lag_deg` (0 in phase, 90 in quadrature, 180 in
antiphase); the ramp ends at the tension of the first cycle.

`RigProtocol.blocks([...])` builds a programme of constant-tension blocks, as
in the block programme of a full-scale fatigue test (CIGRE TB 862, 2022,
6.3.4.1): each `RigBlock(tension_N, amplitude, n_cycles, mean)` holds its
tension while the bending cycles about its mean, and the tension and mean
move linearly between blocks (`transition_s`, default one period).
`n_cycles` is the number of cycles simulated for gauge planning, not the
count of the test. Ranges are taken over the last cycle of the last block;
`protocol.block_times_s` gives each block's interval, so the ranges of one
block follow from `dataclasses.replace(protocol, retain_from_s=start)` on the
protocol cut at the block's end.

```python
programme = cx.RigProtocol.blocks(
    [cx.RigBlock(80e3, 0.029, 2), cx.RigBlock(80e3, 0.085, 2),
     cx.RigBlock(110e3, 0.207, 2)],
    samples_per_cycle=40,
)
```

**Setup** (`RigSetup`):

| Field | Meaning |
|---|---|
| `free_length_m` | Length between the clamp faces |
| `profile` | `"uniform"`: constant curvature over the free length (bending over a template, four-point rig). `"end_rotation"`: tensioned beam, clamp at `x = 0` rotated, clamp at `x = L` fixed; decay length `sqrt(EI/T)`. `"table"`: tabulated shape, e.g. from a finite-element model of the rig. `"field"`: a non-proportional `κ(x, t)` (`curvature_field=`, gap P5), e.g. a bend restrictor locking element by element (`cx.restrictor_curvature_field`) or a bend stiffener; build the protocol with `RigProtocol.from_curvature_field` ([local_analysis_guide.md](local_analysis_guide.md) Section 13.5) |
| `clamp` | `"anchored"`: the armour is anchored at both clamps. `"none"`: sectional result only |
| `clamp_setback_m` | Straight, frictional length between the armour anchor and the clamp face, at each end |
| `reference_x_m` | Station of the input curvature (defaults: mid-span, rotated clamp face, largest tabulated value) |
| `bending_stiffness_Nm2` | Secant `EI` for `"end_rotation"`; by default the secant stiffness of the package bending loop at the clamp amplitude is iterated to convergence |

The end-rotation profile is the solution of `EI w'''' − T w'' = 0` with
`w(0) = 0`, `w'(0) = θ`, `w(L) = w'(L) = 0`. For `sqrt(EI/T) ≪ L` the clamp
curvature is `θ / sqrt(EI/T)`, raised by a chord rotation of order
`sqrt(EI/T) / L` (`end_rotation_profile`).

**Gauges** (`GaugeLocation(layer, angle_deg, x_m)`): the angle is measured
round the section from the tensile extreme fibre (positive curvature) towards
`+y`, with the cable axis pointing from the clamp face into the specimen; the
gauge sits on the outward face of the wire nearest to that angle at `x_m`.
The default grid (`default_gauge_grid`) has eight angles at 0.05 m, a quarter,
half and one pitch from the clamp face and at mid-span, plus, at each station,
the two angles at which the wires that lie on the neutral axis at the anchor
pass. These are the worst wires of the clamp law.

**Options of `simulate_rig_test`:** the nominal `mu` (default: the material
`mu_on_hdpe`) and `mu_factors` (default 0.75 and 1.25) for the sensitivity
runs. `setback_variants_m` adds clamp set-back variants. `contact_model`
(`"auto"`, `"solved"` or `"bounded"`) selects the normal state.
`gauge_zero_tension_N` sets the tension at gauge zeroing.
`gauge_resolution_microstrain` is the assumed gauge resolution.
`segments_per_pitch` (64) and `max_increment` (0.5 `κ_c`) control the clamp
solver discretisation.

**Energised specimen** (`thermal=`): instead of a uniform protocol
temperature, a layer-temperature profile from `cx.ThermalHistory` (for
example `ThermalHistory.steady(cable, current_A)` for a specimen held at a
steady current, or `from_current` for a current programme placed on
`time_s`). Each slow-time block is solved with its free thermal strains and
registered polymer `E(T)`; samples take the clamping and wire stress of their
block, and the stick coefficient uses the polymer moduli of the block that
holds most retained samples. The axial gauge strain then includes the thermal
wire stress (mechanical strain; a temperature-compensated gauge is assumed).
All layers at 20 °C reproduce the default result bit for bit. Theory and
limits: [local_analysis_guide.md](local_analysis_guide.md) Section 7.3.

## 3. What is computed

**Normal state.** `"auto"` and `"solved"` use the solved radial state
(`solve_cable_radial_state`, [radial_model.md](radial_model.md)) at `tension_levels` (5)
tensions and, with a temperature history, three temperatures. Results are
interpolated over the protocol. Twist is held at zero, as at a rig clamp. The
state gives, per armour layer, the friction-setting pressure `P1+P2(t)` and
the uniform wire stress. The axial gauge strain is the change of that wire
stress from the zeroing state (`gauge_zero_tension_N`, 20 °C, straight),
divided by `E`. Cables the solved model rejects (stranded conductor whose
metal area differs from its envelope, corrugated sheath: the 220 kV template
and CIGRE TB 880) fall back to the bounded chain under `"auto"`. The
fallback is recorded in `notes`. A temperature history then raises an error.

**Sectional parts.** At the local curvature `κ(x, t) = κ_ref(t) k̂(x)`:

- **local bending**: the outward-fibre wire bending strain
  (`local_bending_fibre_stress`, fibre angle 0). It vanishes on the
  neutral-axis wire.
- **sectional friction**: the package direction-tied stick–slip stress
  (`wire_friction_stress`) with the pressure history. This is the ξ-reduced
  stick stress below slip and the signed Papailiou bounds in slip,
  `σ0 expm1(μ|sin α|ψ)` for a rising and `−σ0 (1 − exp(−μ|sin α|ψ))` for a
  falling stress. It vanishes on the neutral-axis wire.

**Clamp increment.** Every wire of the layer is followed along the whole
specimen, between anchors at both clamps (`anchored_wire_response`). The
model is rigid stick with a Coulomb bound per unit length scaled by
`P(t)/P_ref` and axial slip only. Each increment is the exact Euclidean
projection of the trial force onto the set `|n_j − n_{j−1}| ≤ ρ h`. The same
solver also integrates the pointwise (sectional) field `n_ref`. The friction
strain at a gauge is

    ε_f = ε_f,sectional + σ_u (n − n_ref) / E,    σ_u = σ0 μ |sin α|,

so the increment vanishes far from the clamps. The constant-curvature
sectional result is therefore the far-field limit (tested to 5 % of the
sectional cap about four boundary layers from the clamps).
The normalisation uses the package slip-onset curvature
`κ_c = slip_onset_curvature(spec)`, which includes the Yun coefficient.

**Moment–curvature loops** (`StationLoop`): `compute_mkappa` at the local
curvature and the mean protocol tension, plus the moment of the clamp
increment of all wire forces, `Σ σ_u A r cos α cos θ_k (n − n_ref)_k`.

**Ranges and sensitivity** (`GaugeResult`):

- `ranges`: max − min over the retained samples, per part and for the total.
- `range_sectional`: the total range without the clamp increment.
- `clamp_contrast`: `ranges["total"] − range_sectional`.
- `range_by_mu`: the total range for each friction coefficient. The friction
  laws depend on μ and `P1+P2` almost only through their product (exactly
  for the linearised clamp increment), so this is also the sensitivity to the
  clamping pressure.
- `range_by_setback`: the total range for each set-back variant.

**Scales** (`LayerScales`):

- pitch `P` and slip-onset curvature `κ_c`;
- `a = κ/κ_c`;
- slip length `L_s = a P/2π`;
- clamp boundary layer `ℓ_x = ½√(L_s P)`;
- set-back `d* = 0.4√(L_s P)` (a map fit);
- clamp law `A = 1 + 0.54√a`;
- averaged-law peak friction strain of the neutral-axis wire at the clamp,
  `(min(a, π/2) + 0.85√a) σ_u/E` (monotonic);
- sectional full-slip cap and axial strain at the mean tension.

**Recommendations** (`GaugeRecommendation`), per layer:

1. **Clamp boundary layer.** Among gauges within `max(P/2, ℓ_x)` of the
   clamp face and within 30° of the neutral axis, the one with the largest
   clamp contrast. Its outward-fibre local bending and sectional friction are
   small, so it reads the friction-fed mean force of the anchored wire (the
   clamp law). This is the most diagnostic signal: the sectional model
   predicts almost nothing there.
2. **Far field, extreme fibre.** The extreme-fibre gauge at mid-span with the
   largest range. It carries the largest local-bending and sectional friction
   ranges, for comparison with the package far-field result. Its clamp
   contrast confirms whether mid-span is outside the clamp layer.
3. **Friction calibration.** Outside the clamp layer, the gauge whose range
   changes most with μ.

Each recommendation gives the amplitude (half the total range) and its ratio
to the assumed resolution. Signals below three resolutions are flagged.

**Assumed resolution.** 2 µε (`DEFAULT_GAUGE_RESOLUTION_MICROSTRAIN`): a
bonded foil gauge on a quarter or half bridge with a 24-bit bridge amplifier
in a laboratory. Long-term drift, temperature and installation scatter are
larger (typically 5–20 µε over a test), so mean levels are less reliable than
cyclic ranges. Set the value for the actual system.

## 4. Verification

| Check | Reference | Result |
|---|---|---|
| Chain projection vs generic QP (SLSQP) and the NumPy reference | Exact optimum | 1e-6 (QP), 1e-12 (reference); `Σn` conserved |
| Far field, uniform curvature, a = 0.8, 1.3, 3 | Kraincanic–Kebadze slip front | Within 2h |
| Far field in cycles | Pointwise sectional field of the same solver | Within 2h |
| Below slip (a < 1) | Stuck force, no clamp effect | 1e-12; zero dissipation |
| Energy, cyclic path with varying bound | `W − D − U − R = 0` | 1e-9 of D |
| Mesh (32/64/128) and step (50/400) convergence of the clamp peak | Convergence | < 1 %, < 0.2 % |
| Clamp peak at a = 100 | Averaged law `π/2 + 0.8525√a` | Within 3 % |
| Cyclic A* at a = 20, uniform from the anchor | Reference regime map | Within 3 % |
| One-shot projection | First increment | 1e-12 |
| Rig, long specimen, mid-span | Package sectional friction | Within 5 % of the cap |
| Rig, `clamp="none"` | Sectional result | Identical |
| Rig parts | Package local bending and solved wire stress | 1e-9, 1e-6 |
| End-rotation profile | Untensioned beam; boundary conditions by integration; `θ/sqrt(EI/T)` | 2 %, 1e-6, 3 % |

The clamp solver is a port of a separately verified reference
implementation. Nothing here has been compared
with a helix finite-element model or a measurement; the gauge data are meant
to supply that comparison.

## 5. Example: public three-core 33 kV template

`examples/rig_test_gauge_plan.py`: the public 33 kV three-core template has a
4 m free length, a uniform profile, armour anchored at both clamps, 100 kN,
three cycles and the solved radial state. Strains are in microstrain (µε);
ranges are over the last cycle. The values are for the corrected geometry
(61 + 67 armour wires, declared semiconducting screens); the
earlier 76 + 84 wires overlapped.

| | Inner armour | Outer armour |
|---|---|---|
| Pitch | 0.82 m | 0.90 m |
| `P1+P2` | 7.36 MPa | 2.30 MPa |
| `κ_c` | 0.074 1/m | 0.020 1/m |
| Sectional full-slip cap | 426 µε | 145 µε |
| Axial strain at 100 kN | 148 µε | 152 µε |

Outer armour, by bending radius:

| R | a | ℓ_x | Neutral-axis gauge, x = 0.05 m: range with clamp | Same gauge, sectional | Mid-span extreme fibre: range | Mid-span: clamp contrast |
|---|---|---|---|---|---|---|
| 10 m | 5.1 | 0.41 m | 576 µε | 176 µε | 596 µε | −2 µε |
| 5 m | 10.2 | 0.57 m | 890 µε | 251 µε | 896 µε | −11 µε |
| 3 m | 17.0 | 0.74 m | 1163 µε | 381 µε | 1326 µε | 6 µε |

Notes on the table:

- The neutral-axis gauge at x = 0.05 m is the outer wire on the neutral
  axis at the anchor.
- The friction amplitude at that gauge is 233, 348 and 419 µε, against the
  averaged clamp-law peak of 305, 374 and 443 µε.
- At mid-span the extreme-fibre range is about 50–80 % local bending.
- With the 2 µε resolution every recommended signal is about 100–470 resolutions.
  The clamp contrast at the near-clamp neutral-axis gauge (400–780 µε) and
  the μ sensitivity of the mid-span gauge (±25 % μ: 110–160 µε, outer) are both
  far above it.

Practical consequences for a test plan:

1. **Neutral-axis gauges near both clamps.** Place them within about 0.05–0.1
   m of each clamp face, at the angle where the anchor's neutral-axis wires
   pass, on both sides (±90° at the anchor). This is the direct test of the
   anchored-wire friction (clamp) model. Add a second station at half a pitch,
   where the opposite neutral-axis wire passes.
2. **Extreme-fibre gauges at mid-span** (0° and 180°), at least `3ℓ_x` from
   either clamp. These test the sectional model and, with the bending loop,
   calibrate the friction product μ·P.
3. **A free length of at least 6–8 `ℓ_x`.** For the outer armour at R = 3 m
   that is 4.4–5.9 m. Otherwise the clamp layer reaches mid-span; at 4 m and
   R = 3 m the mid-span contrast is 6 µε here, but it was 42 µε with the
   longer `ℓ_x` (0.88 m) of the pre-correction geometry.
4. **Measure the anchorage.** Record where the armour is anchored inside the
   fittings (set-back and epoxy length). `setback_variants_m` shows how
   strongly the near-clamp readings depend on it.

### Specimen length (CIGRE TB 862)

`cx.tb862_specimen_checks(cable, test, ...)` compares the lengths of a test
specimen with the requirements of CIGRE TB 862 (2022), using the pitches of
the cable's helical components (`cx.component_pitches(cable)`: armour
layers, the core lay, resolved strand rings). `P_a` is the pitch of the outer
armour wires and `P_max` the longest pitch of any component.

| `test` | Requirement | Clause | Kind |
|---|---|---|---|
| `"fatigue_template"` | fatigue-loaded section `>= P_max`; `>= P_a` from it to each end fitting (with a uniform-profile `setup`: its free length and clamp set-back) | 6.3.2 | recommended |
| `"fatigue_bend_stiffener"` | top end fitting to stiffener flange `>= P_a` | 6.3.2 | shall |
| | stiffener tip to bottom end fitting `>= P_a` | 6.3.2 | recommended |
| `"bending_stiffness"` | sample `>= P_a` | 6.4.2.2 | recommended |
| `"compression_bending"` | sample `>= 2 P_max` | 6.6.1.2 | recommended |
| `"compression"` | sample between end fittings `>= P_max` | 6.6.1.2 | must |

Each `SpecimenCheck` holds the requirement, the clause, the required and the
actual length, `mandatory` and `passed`.

## 6. Limits

- Clamp model:
  - axial slip only, with no transverse slip or wire flattening;
  - rigid stick, with the ξ-reduced stick stiffness entering through `κ_c`
    only;
  - a constant bound along the wire;
  - one wire per layer, with no interaction between layers;
  - planar curvature, proportional `κ(x, t) = κ_ref(t) k̂(x)` with the
    default profiles; a non-proportional restrictor or stiffener field
    (`profile="field"`, gap P5) is solved segment by segment.

  The clamp increment uses the linearised Papailiou cap; the sectional part
  uses the package's signed Papailiou bounds.
- The averaged clamp law over-predicts for `a ≲ 10`. The solver is exact for
  the stated model at any `a`.
- Bending-induced redistribution of contact pressure (three-core ovalisation)
  is not modelled. Pressure is the mean solved value per layer.
- Moment loops use the mean protocol tension. The end-rotation profile is
  frozen at the mean tension and the secant stiffness.
- Gauge strain is mechanical wire strain on the outward fibre, with a
  temperature-compensated gauge assumed.
- An elastic-range note is added to `notes` when a gauge's stress change
  exceeds the wire yield strength.
- The clamp idealisation controls the near-clamp prediction. A bonded
  (epoxy) length inside a fitting is not frictional; put the anchor at its
  inner end with `clamp_setback_m = 0` measured from there.
- Use public cable definitions for shared results. Manufacturer data stay in
  local overrides.
