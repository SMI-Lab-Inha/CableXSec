<!-- SPDX-License-Identifier: Apache-2.0 -->

# Benchmark register

Current register of every comparison made with CableXSec. Each entry gives
the source, the comparison class, the quantity, the target, the result and the
status. Tiers, classes, status labels and the rules behind them are defined in
[validation_policy.md](validation_policy.md).

Numbers are taken from the generated scorecard
(`validation/validation_report.md`, written by
`validation/scripts/run_validation.py`) and from the validation and
application sections of the companion article (Section "Model validation" and
its tables). Where the two differ, the article value is used and the
difference is noted. Gate identifiers (for example BM1, WITZ-EI) refer to rows
of the scorecard.

Comparison classes: **A** analytical, **M** measured, **MM** model-to-model,
**I** internal (two CableXSec formulations on the same inputs).

**Signed capstan bound (unreleased correction).** The armour friction bound now
follows Papailiou (1997, Eqs. 11 and 13–15) with the signed helix angle: a
rising wire stress is bounded by `σ0 (exp(λψ) − 1)`, a falling one by
`−σ0 (1 − exp(−λψ))`, `λ = μ |sin α|`; before, the tension bound was used on
both sides. The rows of the scorecard and the tables produced by
`check_core_friction.py`, `check_g8_gradual_reversal.py`,
`check_t1_test_arrangement.py` and `check_saevik2011.py` are regenerated with
it; the former values are given where a status or a conclusion moved. Tables of
one-off studies that have no reproduction script in the repository (the Witz
mechanism and identification studies, the along-wire, two-way and
rectangular-tendon variants of the Sævik comparison, the identified-state
Ménard columns taken from the companion article) keep the values computed with
the former bound and say so. Their slipping-case values move by a few per cent
(the sectional Sævik ratios by 3–5 %); no conclusion drawn from them changes.
For orientation, the contact-level helical rod cell of the companion article
(`transverse_friction_cell.py`, uniform normal load, no capstan term) now gives
33.7 and 70.4 J/m on the Ménard cable at ±0.05 and ±0.10 1/m against 34.1 and
70.9 J/m from the section law (35.3 and 73.5 J/m with the former bound), the
`O(λ²)` agreement the signed bound implies.

**Sheave check.** `compute_sheave` now also reports the
design utilisation on the capacity envelope (normal factor 0.67). SH-3 checks
the unchanged characteristic ratio, now `characteristic_utilisation`, against
the same target; PASS. SH-4 (reported), 66 kV template at 50 kN: design
utilisation 0.701 at 4 m and 0.487 at 6 m (installation factor 0.78: 0.602
and 0.419); characteristic ratio 0.448 and 0.298 unchanged. Correction to the
recorded SH-4 outcome: the run-on force at 4 m is 15.4 to 46.4 kN (the code
before this change gives the same), not 15.4 to 45.0 kN. COMP-T, COMP-N and
COMP-O unchanged. The capacity check now never reports a utilisation below
the wire usage at zero curvature.

**Coiling capacity repair.** The Chae, Nam and Shim (2023)
row CH-CR was rerun after the coiling capacity repair and is unchanged (the
repair changes only the capacities and the utilisation, not the stresses). The coiling request now states its capacities
as wire stresses, P/(n A cos a): the source's Eqs. 17 to 19 divide by n A and
give the axial-direction component, cos a times the wire stress (535.4 MPa in
Table 6 is 555.1 MPa as a wire stress at pitch 1.8 m). CH-CR still checks the
source's own form.

**Correction to the finite-support no-slip EI (after this register's runs).**
The Yun shear coefficient of the finite-support no-slip EI counted an
interstitial filler as a concentric shear layer in series with the armour,
although it lies beside the cores. It is now left out. This raises the no-slip
EI of every cable with an interstitial filler (IEA Task 49 66 kV, the bundled
33 kV and 220 kV three-core cables, Chae 2023, CIGRE TB 880 and Thies 2016 by
14 to 58%) and moves the loops that use the coefficient slightly. Full-slip
EI, the plane-section full-stick EI and the cables without a filler (Ménard
BM1, Witz, Sævik) are unchanged. The benchmark input `menard2023.yaml` is
unchanged; `menard2023_filled.yaml` (with the PP fillers of the source FE
model) is the full construction.

## Summary

| Area | Class | Status |
| --- | --- | --- |
| Tier-1 analytical and engine-consistency gates (25 rows) | A, I | All PASS |
| Coiling capacity formulas, Chae, Nam and Shim (2023) (CH-CR) | A | PASS |
| User S–N data and capacity check of time series (13 rows) | A, I | All PASS (verification only) |
| Out and von Morgen (1997) frictionless slip bounds; Out (1989) effective curvature change | A | PASS; geodesic transverse map reproduces the frictionless slip vector; legacy transverse map in quadrature with it (stated) |
| Armour compression: Sævik and Thorsen (2017), Østergaard et al. (2012), Li and Vaz (2021a, b), Yang et al. (2015) | MM | PASS (published closed-form values and model plateaus reproduced) |
| Armour compression: de Sousa bird-caging tests (via Sævik and Thorsen 2017; Gjøsteen et al. 2027) | M | Straight-beam P2* within 1.4% of the 256 kN test mean; published per-term P2 not reproduced (stated) |
| Armour compression: Malta et al. (2022), sheath-only FE | MM | Reported only: screen 49% below the FE limit load (conservative) |
| Ménard and Cartraud (2023) full-slip stiffness | MM | PASS |
| Ménard and Cartraud (2023) six measured bending ranges, moment | M | Moment errors 2.71–10.55% (transfer) |
| Ménard and Cartraud (2023) six measured bending ranges, work | M | PASS at the retained 10% target (range b, +8.8 %; +12.9 % with the former capstan bound); ranges (a), (c), (d) +32 to +47 % |
| Armour longitudinal displacement, Ménard and Cartraud FE | MM | Phase and scale reproduced; peaks over-predicted |
| Tjahjanto et al. (2017), Leroy et al. (2017) FE armour slip | MM | Qualitative agreement only |
| Sævik and Bruaseth (2005) umbilical axial stiffness | M | NOT VALIDATED |
| CIGRE TB880 conductor temperature | MM | PARTIAL |
| Global–local consistency, floating-wind application | I | Full-slip results are upper estimates |
| Laid-core friction (gap R1b), Ménard and Cartraud (2023) six ranges | M | With the transferred 0.3 MPa the cores raise the work (range (b) +27.2 %); identified on (b) with the full accounting, predictions (a) and (c) FAIL (+26.0, +23.7 %), (d)–(f) within 13 % |
| Fretting regime and wear (gap 3.1): Mindlin/Cattaneo closed forms, Fouvry ratios, limits, rotation | A | PASS |
| Fretting transition, Fouvry et al. (1995) Table 4 | MM | PASS with the printed shear moduli; 1.6–1.7 × the measurements with G = E/2(1 + ν) (stated) |
| Fretting wear volume, crossed steel wires (Poon et al. 2024; Cruzado et al. 2013) | M | PASS (consistency: coefficient identified on the same tests) |
| Conductor fatigue version 2 (gap CF2), Nasution et al. (2013, 2014a, 2014b): bulk count with the SWT mean-stress correction against each conductor's own wire curve (default), critical-distance trellis rule (opt-in) | M | Calibration by proxy (the outer-wire curve carries the compaction SCF into the inner rings; see the section): dry tension–bending lives ×1.45–2.39 (×1.11–1.56 with the trellis rule at the lower threshold), radius trends 4.65 and 11.1 against 3.6 and 10.3, failure layer PASS; tension–tension lives ×0.86–1.16 but failure predicted in the centre wire, not the outer layer (FAIL, no compaction SCF); lubricated group ×12.5 (reported) |
| Conductor fatigue version 3 (gap CF3, opt-in): wire curve corrected for the compaction indentations, compaction factor per ring (Jiang et al. 2022), trellis rule at the full-hard threshold of Liaw et al. (1982) | M | FAIL on the registered gate: 300 mm² bending lives ×2.05–2.24 and radius trend 3.7 in band; 95 mm² bending ×4.1–9.5, trend 20–23 (10.3); tension–tension failure in the outer layer as observed, lives ×0.97–3.2 |
| Conductor trellis fretting, surface SWT (gap 1.3c, G1; `trellis_model="surface_swt"`) | M | FAIL: dry lives ×0.0075–0.12, radius trends 1.02 and 1.06 against 3.6 and 10.3; legacy option, not validated |
| Witz (1996) 2.5-inch flexible pipe, full-slip EI at 0 bar (WITZ-EI) | M | PASS (1007 against 1014 and 1190 N·m², parameter-free) |
| Witz (1996) pipe, 300-bar loop predicted after identification on the 0-bar loop (WITZ-ID) | M | FAIL: width −32 to −37%; energy −14.2% (inside its target); one loop identifies only μ·p₀ |
| Gradual stick-to-slip transition (gap G8): progressive law, zero-spread, dissipation and rotation limits | A | PASS |
| Gap G8 against Witz (300 bar), Hu et al. (2022) and Ménard range (b) | M | Hu transition share reproduced by the three-point test arrangement (35.6% against 33.8%); BM1-loop with the progressive law PASS (+8.2 %); Witz width and ratio and Hu loop area (+37%) FAIL |
| Sævik (2011) 8-inch riser, measured FBG inner-armour stress ranges, 14 load cases (SV11-PLANE); Gjøsteen et al. (2027) absolute LC 14 ranges (SV11-GJ) | M | FAIL: termination-corrected median ratio 1.62 (3 of 14 in band), absolute 1.80; sectional median 1.00 (11 of 14), absolute 1.24; small angles under-predicted about 2× |
| Static/dynamic friction law against measured wire-on-anti-wear-tape loops, Dai, Sævik and Ye (2018), Fig. 4, one spring-slider (DAI18-DIG, DAI18-RS, DAI18-T1, DAI18-T3) | M | Digitisation and re-sticking at every reversal (`stick_time_s = 0`) PASS; reported: Test 1 peak within 1.8 % and +x level within 3.3 % with the printed coefficients, the measured −x level 4–14 % lower (model +4.7 to +12.3 %); work 1–5 % low (no gradual breakaway decay); static peak +10.6 % at 0.2 Hz not representable; dynamic level rate independent |
| Test-arrangement simulator (gap T1): elastic closed forms, uniform-moment zone against `compute_mkappa`, energy, equilibrium | A, I | PASS |
| Along-wire redistribution (gap T2): periodic and clamp-layer limits, termination round-off, energy, refinement | A, I | PASS |
| Gap T2 against Sævik (2011), Gjøsteen et al. (2027), Hu, Ménard and Witz | M | Not explained (former capstan bound throughout): Sævik near-bellmouth median ratio 3.36 (2.12 in the clamp convention) against 1.65 for the clamp correction and 1.03 sectional; Hu area +65%, share 11%; Ménard work −3 to +3 points by range; end conditions immaterial |
| Two-way anchored termination: Sævik rig beam solved with the anchored along-wire chains of both armours | M | FAIL as registered (former capstan bound): median ratio 1.18 (9 of 14 in band), absolute 1.46; anchored over-prediction falls from 3.09 one-way (same stick setting) and 1.65 (clamp correction); curvature redistributes along the bellmouth instead of falling (TC-3 fails); hang-off rule stays sectional |
| Anchored wire at uniform curvature against published FE: Dong et al. (2025), Zhu and Tan (2014) (AW-DONG25, AW-ZT14) | MM | Reported: Dong et al. verification case reproduced (anchor 156.4 against 156.9 MPa, peak +1.6 %); Zhu and Tan not reproduced (1.70 against 2.2 normalised at the neutral-axis anchor): their FE far field is in stick and their analytical field breaks the Coulomb flow rule |
| Hysteretic bending law for global analysis (gap MX): Iwan law from the exported backbone against `compute_mkappa` | I | MX-1 PASS (≤ 0.05 % of the moment range, work ≤ 0.02 %), MX-3 PASS, MX-scale PASS (slip scaling variable); MX-2 FAIL (tension interpolation 3.9 % at 100 kN spacing; 0.27 % at 20 kN); MX-5 (global program) open |
| Cable over a sheave (gap SH): Feyrer (2015) Examples 3.2 and 3.4, groove factors, consistency with the section analyses | A, I | SH-1, SH-2, SH-3 PASS; no measured cable-over-sheave data held |
| Gap T1, benchmarks in their published arrangements | M | Hu three-point: transition share in its band (32.8–39.4% against 33.8%), loop area FAIL (+37% Papailiou cap, +64% Karlsen-Husøy; +4.9%/+28% at uniform curvature compared a different quantity); Ménard four-point unchanged; Witz arrangement not documented, basis unchanged |
| Torque balance (gap 3.5): Feyrer (2015) and Knapp (1981) closed forms, mirror lay, `xsec_stiffness` consistency | A, I | PASS |
| Power-core screens, core slip state, sheath strain life (gaps P1, P6, P8): bounds, single-core reduction, strain-life identities, Thies (2016) hand calculations | A, I | PASS (no published sheath or screen fatigue points held) |
| Restrictor and stiffener fields, tapes, copper strain life (gaps P5, P2, P7): rigid-link chain, tensioned beam, chain limit, proportional-field equivalence, tape states, default invariance | A, I | PASS (no restrictor test with curvature data beyond Thies 2016 held) |
| Torque balance: Kim and Lee (2017) balance angles; Utting and Jones (1987) fixed-end torque | MM, M | PASS (−0.06 to −0.26°; ×1.07–1.14 test, ×1.05–1.06 theory from omitted wire moments, stated) |
| Rectangular tendons and local tendon bending (gap G1): section closed forms, round limit bit for bit, Sævik (1992) Eqs. 2.30–2.32, 4.1, 4.2, numerical geodesic, Dong et al. (2018, 2025) | A, I, MM | PASS |
| Gap G1 against Sævik (2011) / Gjøsteen et al. (2027) with the true 5 × 12.5 mm tendons, with and without lateral relief | M | FAIL: sectional 1.06 (11 of 14), GJ 1.25; clamp correction 1.72 (GJ 1.85); along-wire 3.40 (GJ 4.68, former capstan bound); lateral relief changes these by 1–3 % (hypothesis not supported); end-restraint layer 0.1 m, 18 decay lengths short of the nearest gauge |
| Gap G1, Witz (1996) WITZ-ID on the true strips, solved radial route | M | FAIL: width −25.3 / −31.3 %, energy −5.9 % |

## Tier 1: analytical and consistency checks

All rows are verification, not validation. Source: the scorecard
(`validation/validation_report.md`).

| ID | Class | Quantity | Target | Result | Status |
| --- | --- | --- | --- | --- | --- |
| STATE-NQ | A | Annulus extension and Saint-Venant twist | < 1e-10 rel. | 6.7e-16; 0 | PASS |
| STATE-P | A | Closed-end thick tube, von Mises stress vs Lamé | < 1e-10 rel. | 7.3e-15 | PASS |
| STATE-T | A | Uniform free thermal expansion and stress | < 1e-10 strain; < 1e-5 Pa | 3.1e-15; 1.3e-6 Pa | PASS |
| A10 | A | Vector-slip rotation equivariance | < 1e-12 rel. | 3.8e-16 | PASS |
| A10b | A | Fixed-plane vector/loxodromic contrast | < 1e-12 rel. | 1.2e-16 | PASS |
| OvM | A | Contact slip at μ = 0 vs Out–von Morgen geodesic slip; closed forms vs exact geodesic | < 1e-12 rel.; ratio ≤ 1; < 1e-3 rel. at k = 1e-4 | 1.1e-15; 0.333; 3.8e-4 | PASS |
| OvM-g | A | Geodesic transverse map (η = 1) at μ = 0 vs Out–von Morgen slip vector; vs exact geodesic | < 1e-12 rel.; < 1e-3 rel. at k = 1e-4 | 1.1e-15 (δ*/Eq. 28: 2.7e-15); 2.6e-4 | PASS |
| Out89 | A | Contact plastic slip vs Out (1989) Δκ − 2κ_s and damage ratios A8, A11 | < 1e-12 rel.; < 1e-4 abs. | 9.6e-16; 5.0e-7 | PASS |
| FR | A | Fretting regime: Mindlin energy vs numerical Masing loop; Fouvry A/D/B; strip I(0); μ = ∞ stick energy; planar-path rotation | < 1e-9; < 1e-12; < 1e-12; 0 J; < 1e-12 rel. | 7.8e-12; 4.4e-16; 1.6e-15; 0 J; 4.4e-16 | PASS |
| CF | A | Laid-core friction: μ = 0 gives the armour-only loop; large μ tangent vs EI_max; full-slip core moment vs closed form; dissipation ≥ 0; rotation equivariance | 0 N·m; < 1e-9; < 5e-3; ≥ 0; < 1e-9 rel. | 0; 1.0e-15; 1.0e-4; 5.0e-6; 5.0e-14 | PASS |
| CFW | A | Laid-core contact statics (wedged trefoil and central filler, with and without a filler): radial equilibrium of one core; filler profiles: μ = 0 gives the armour-only loop; full-slip profile moment vs closed form; stick tangent vs `EI_max + E J_f cos³β_f / 2` | < 1e-12 rel.; 0 N·m; < 5e-3; < 1e-9 | 9.5e-17; 0; 1.0e-4; 7.8e-16 | PASS |
| A11 | A | Moment–curvature energy closure (loop area vs closed form) | < 0.1% | 0.082% | PASS |
| σ_B | A | Radial-fibre bending stress vs Ménard Eq. 14 | < 1e-6% | 0 | PASS |
| M-rec | I | Assembled per-wire moment vs `compute_mkappa`, single core | < 2% | 0.000% | PASS |
| M-rec-3core | I | As M-rec, offset-core geometry | < 2% | 0.000% | PASS |
| MK-Tcyc | I | `compute_mkappa` with a contact-tension history cycling inside the bending cycles vs the moment of the fatigue-path friction stress on the same history; work-balance residual over the friction work | < 1e-12 rel.; < 1e-12 rel. | 3.4e-16; 1.1e-15 | PASS |
| MX-scale | I | Slip-scaled bending law: `compute_mkappa` at 70 kN vs the 20 kN law with the bound scaled (66 kV, cores); group elements at 60 kN vs the 10 kN elements with yield curvatures scaled (Ménard); tension cycle inside the bending cycles, scaled law vs the contact-tension history over the fixed-tension Iwan residual | < 1e-12 rel.; < 1e-9 rel.; ratio ≤ 1.25 | 4.1e-16; 8.3e-14; 1.02 | PASS |
| FL-P | A, I | Pressure-dependent friction coefficient (`FrictionLaw.pressure_factor`): spring-slider under pressure steps vs closed form; compiled vs NumPy direction-tied kernel under a pressure history (static/dynamic law, local pressures below, inside and above the table); work balance | < 1e-9 rel. (three checks) | round-off | PASS |
| slip-onset | A | First wire to slip lies at the neutral axis (with the signed bound on the compression side, ψ ≈ 3λ/2) | abs(cos) < 0.2 | 0.156 (0.000 with the former bound) | PASS |
| xsec-K | A | Tangent bounds EI_max (κ→0) and EI_min (κ→∞) | < 0.01% | 0; 7.4e-12% | PASS |
| QP-KKT | A | Concentric contact complementarity | < 1e-8 N | 1.9e-13 N | PASS |
| QP-Lamé | A | Cylinder compliances vs compound-cylinder shrink fit | < 1e-6 | 1.3e-16 | PASS |
| COMP-T | A | Free thermal expansion, all components | < 1e-6 Pa | 0 | PASS |
| COMP-N | A | All-component axial-force equilibrium | < 1e-12 rel. | 0 | PASS |
| COMP-O | A | Zero-offset local and strain-compatible bounds | < 1e-6 Pa | 0 | PASS |
| SH-AX | A | Metallic sheath equal-strain axial stress | < 1e-12 rel. | 0 | PASS |
| io-known | A | Time-history CSV reader to fatigue, closed-form Miner damage | < 1e-10 | 0 | PASS |
| io-global | A | Global time-domain history vs its recorded run summary | < 1e-10 | 0 (201 samples) | PASS |
| σ_c1 | A | Bird-cage onset vs Chae Eq. 17 | < 1e-9% | 0 | PASS |

Further internal comparisons reported in the companion article (not scorecard
gates):

| Comparison | Class | Result |
| --- | --- | --- |
| Spatial armour model vs linearised section law, same normal load | I | Cycle work within 1.2% (0.05/m) and 0.6% (0.10/m) |
| Spatial model vs exponential slip limit | I | 4.9% and 4.5% higher in the section law (normal-load assumption) |
| Contact-level work with equilibrium transverse slip vs longitudinal only | I | Within 0.15%; transverse slip 1.1–6.7% of longitudinal |
| Spatial model mesh refinement | I | Peak moment < 0.04%, cycle work < 0.001% |
| Contact energy balance | I | Closes within 4e-9 J/m |
| Frozen vs updated clamping, concentric section | I | Up to 0.72% over-prediction of first-yield curvature |
| Global direct-strain force identity | I | Within 7.3e-12 N |

## Ménard and Cartraud (2023), published three-core cable

Source: Ménard and Cartraud, Marine Structures (2023), doi
10.1016/j.marstruc.2023.103465. Public construction in `menard2023.yaml`
(101 mm outer diameter, 55 + 63 armour wires of 4 mm). Measured loops are
recovered from the ordered vector paths of their Figure 16; reading resolution
for range (b) is 0.00069 1/m in curvature and 4.15 N·m in moment.

### Full-slip bending stiffness (BM1)

| Class | Quantity | Reference | Result | Target | Status |
| --- | --- | --- | --- | --- | --- |
| MM | EI_min, no fitted parameter | Published analytical value, 3593 N·m² | 3571 N·m², −0.61% | < 15% | PASS |

The reference is the source's own calculation, not a measurement, so this is a
model-to-model comparison. The scorecard lists BM1 under Tier 2 because the
case is public.

### Six measured bending ranges

Inputs transferred from the source without refitting: friction 0.2 and external
pressure 0.3 MPa, which the source calibrated on range (a). Ranges (b)–(f) lie
outside that calibration. Moment error is the ordered-history RMSE normalised by
the measured moment range. Work is integrated over the same plotted curvature
path for measurement and model, in J/m.

| Range | Curvature (1/m) | Moment error, transfer (%) | Moment error, identified (%) | Measured work (J/m) | Transfer work (J/m) | Identified work (J/m) |
| --- | --- | --- | --- | --- | --- | --- |
| (a) | −0.05 to +0.05 | 10.55 | 4.56 (fitted) | 20.2 ± 0.9 | 29.7 | 18.7 (fitted) |
| (b) | −0.10 to +0.10 | 2.71 | 4.50 | 62.3 ± 3.0 | 67.8 | 45.1 |
| (c) | −0.10 to 0 | 10.14 | 5.49 | 20.4 ± 1.0 | 29.6 | 18.6 |
| (d) | 0 to +0.10 | 8.34 | 5.55 | 22.5 ± 1.0 | 29.8 | 18.8 |
| (e) | −0.20 to 0 | 3.83 | 6.40 | 65.0 ± 3.0 | 66.5 | 44.1 |
| (f) | 0 to +0.20 | 3.85 | 7.04 | 69.2 ± 3.1 | 66.8 | 44.4 |

The transfer columns are those of the signed capstan bound
(`check_g8_gradual_reversal.py`); with the former bound they were 11.48, 3.16,
11.02, 9.08, 3.98 and 3.81 % (digitisation spread ±2.03, ±0.92, ±1.16, ±1.06,
±1.04 and ±2.84 points) and 30.8, 70.4, 30.6, 30.8, 68.9 and 69.2 J/m. The
identified columns are those of the companion article (former bound).

- **Transferred inputs.** Moment errors 2.71–10.55%. Work is 32.0–47.2% above
  the measurement in ranges (a), (c) and (d), beyond the digitisation bound of
  4.4–4.9%; range (b) is 8.8% above, inside the 10 % target; range (e) is
  2.2% above (bound 4.7 %) and range (f) 3.6% below, both within their
  digitisation bounds.
- **Identified state.** Pressure and support-rigidity multipliers (0.666 and
  0.288) fitted to range (a) only. Withheld moment errors 4.50–7.04%; withheld
  work 8.9–35.9% below the measurement (former bound).
- **Identifiability.** Re-identifying at friction 0.1–0.3 gives fit errors of
  4.555–4.566% with the product of friction and clamping constant within 4%.
  The loops constrain the frictional resistance, not friction and clamping
  separately.
- **Source table.** The source's tabulated energies (printed unit J) differ
  from the trace integrals by +8.5, −5.5, −20 and −23% in ranges (a), (d), (e)
  and (f). They are kept separately with their printed unit and not converted.

Scorecard gate:

| ID | Class | Target | Result | Status |
| --- | --- | --- | --- | --- |
| BM1-loop | M | Range (b), transferred inputs: ordered RMSE < 5%; matched work within 10% | RMSE 2.71%; work 67.8 vs 62.3 J/m, +8.8% (former bound: 3.16 %, 70.4 J/m, +12.9 %, FAIL) | PASS |

Since gap T1 the row is computed through the test's four-point arrangement
(moment `Fa L1`, curvature from a parabola through three LVDTs between the
actuators); the result is identical, because the measured curvature lies in
the uniform-moment zone. The uniform-curvature calculation is kept as
`BM1-loop-uniform`. The 10% work target is retained and pinned by
`tests/test_validation_report.py`.
With the former symmetric capstan bound the row failed (+12.9 %); the signed
bound lowers the full-slip loop work by 3.7 % at this cable's `λ` (0.069),
which brings it inside the target. Gap G8 had tested whether a gradual
stick-to-slip transition explains the excess: the progressive (continuity)
law lowers it further, to +8.2% (section on gap G8 below). The 8.8% refers to
the transferred-input calculation, in which the model over-predicts the work;
it is not the identified-state result.

### Armour longitudinal displacement

| Class | Quantity | Reference | Result | Status |
| --- | --- | --- | --- | --- |
| MM | Inner-armour wire displacement, 55 wires at 0.05 and 0.20 1/m | Source periodic 3D FE cell | RMSE 0.028 and 0.055 mm; peaks over-predicted by 23% (0.199 vs 0.162 mm) and 10% (0.797 vs 0.727 mm) | Phase and scale reproduced; no target declared |

The geometric helix relation is used with unchanged geometry and no fitted
displacement scale.

## Other published finite-element studies

| Source | Class | Quantity | Result | Status |
| --- | --- | --- | --- | --- |
| Tjahjanto et al. (2017) | MM | Armour slip decomposition at ±0.05 1/m in a 3D FE model of another cable | Source finds slip almost exclusively loxodromic; consistent with the small equilibrium transverse slip computed here | Qualitative; not recalculated |
| Leroy et al. (2017) | MM | Transverse armour curvature over ten one-sided cycles to 0.172 1/m | Source curve follows the helix without transverse displacement (the Δκ_g term) | Qualitative; not recalculated |

Both FE models include contact with neighbouring wires, which the equilibrium
calculation omits, so the computed transverse slip is an upper estimate. It is
also below the frictionless geodesic bound of Out and von Morgen (1997) at
every station (next section).

## Laid-core friction (gap R1b)

`compute_mkappa(..., core_friction=...)` adds the stick–slip law of the laid
power cores and of resolved internal wire rings
(`cablexsec/mechanics/core_friction.py`). Verification gate CF (Tier 1 table
above; `tests/test_core_friction.py`): the full-slip core moment equals
`(2/π) n q r² cos²β/sin²β · a cosh(aπ/2)/(1 + a²)`, `a = μ sin β`, within
1.0e-4 (after a reversal the cores of one side sit at the signed lower bound
`−σ0 (1 − e^(−aψ))` and the others at `σ0 (e^(aψ) − 1)`, so the integrand is
`sinh(aψ) sin ψ`; the former symmetric bound gave
`[(a e^(aπ/2) + 1)/(1 + a²) − 1]`); μ = 0 returns the armour-only loop
bitwise; large friction with a rigid contact gives
`compute_stiffness().ei_max_Nm2` (4.4e-16).

Published comparisons (`validation/scripts/check_core_friction.py`). Targets were
registered before the runs; nothing was refitted.

| Case | Class | Target | Armour only | With core friction | Status |
| --- | --- | --- | --- | --- | --- |
| Janocha (2024) 66 kV, 50 kN, Fig. 6 loop half-height at κ = 0 | C2C (FE, no initial contact state) | reported (about 1.0 kN·m) | 0.081 kN·m (solved) | 0.181 kN·m (μ 0.46, Table 2); 0.374 kN·m with the filler profiles | reported |
| Same, slipping slope; IEA Wind Task 49 Table 13 (its row cites the same source) | C2C | reported (42.47 kN·m²; Fig. 6 about 29) | 21.0 kN·m² | 21.0 kN·m² | reported |
| Same, Fig. 12b moment at 0.01 1/m and stick slope | — | retired (below) | 0.23 / 0.37 / 0.53 kN·m | 0.25 / 0.57 / 0.93 kN·m | retired |
| Ménard (2023) range (b), transferred inputs (0.3 MPa of the source FE), core μ 0.2 as in the source FE | M | RMSE < 5 %; work within 10 % (BM1-loop) | 2.71 %; +8.8 % | 5.15 %; +27.2 % | FAIL with the cores |
| Ménard (2023) ranges (a), (c)–(f), work | M | — | +47.2, +44.7, +32.0, +2.2, −3.6 % | +63.1, +59.8, +46.2, +19.3, +12.6 % | (worse) |

- **Janocha.** The source is a finite-element model, not a test, and states
  no initial contact state; its figures are compared code to code and not
  gated. Fig. 12b is retired as a target (below). With the full contact
  accounting (wedged trefoil and filler profiles) the Fig. 6 half-height at
  50 kN is 0.380 / 0.374 kN·m (bounded / solved) against about 1.0 kN·m read
  from the figure. Friction does not raise the full-slip slope; the slope
  difference lies in component bending ([template_66kv.md](template_66kv.md)
  Section 4.1).
- **Ménard.** The published pressure (0.3 MPa) was calibrated with friction
  on every contact of the source FE, so the armour-only transfer already
  carries the cores' share of the work; adding the cores without
  recalibration raises the work on every range. The bounded ring load on a
  core's outer face is 24.8 kN/m against the 16.5 kN/m per core that the
  source FE reports (about a third of its 46 kN/m inner-sheath load); with
  the neighbour reactions of the wedged trefoil a core carries 53.3 kN/m on
  all its contacts.
- **Core lay stage, whole report.** `validation/scripts/run_validation.py
  --core-lay-stage` evaluates every bending law of a cable with laid cores
  with the stage of the Section properties analysis (core friction 0.2, core
  insulation shear, no internal ring slip, wedged trefoil) and writes
  nothing. Rows that move: BM1-loop and BM1-loop-uniform ordered RMSE 2.71 %
  to 5.15 %, work +8.8 % to +27.2 % (PASS to FAIL; before the trefoil wedge
  3.88 % and +18.9 %); CF, CFW and MK-Tcyc read FAIL because the stage also
  switches their armour-only references; MX-scale's tension-cycle check on
  the Ménard cable reads 2.36 against at most 1.25 (the Ménard core group
  then joins the scaled law). No other row moves. The stage stays off by
  default.
- **Su et al. (2026).** No bending loop is published for the 35 kV cable
  (axial and torsional results only); not compared.

### Registered targets: full contact accounting (registered before the runs)

Registered on 2026-11-17, before any run of the comparisons below, in the
first commit of the change that adds them (`validation/scripts/check_core_friction.py`,
`identify_then_predict`). Nothing below is refitted after the run.

**Ménard and Cartraud (2023), identify on one range, predict five.**

- Model: `menard2023_filled.yaml` (the fillers of the source FE), bounded
  contact chain, armour μ 0.2, the laid-core law with the trefoil wedge and
  the filler profiles, μ 0.2 on every core and profile contact (the source's
  single coefficient, Section 5).
- Identification: the initial external pressure `P0` is the one value for
  which the matched cycle work of range (b) equals the measured 62.3 J/m
  (bisection to 0.1 %). It replaces the source's 0.3 MPa, which belongs to
  the source FE's contact accounting.
- Targets: range (b) ordered RMSE < 5 % of the moment range (a shape check;
  only the work is fitted). Ranges (a), (c), (d), (e), (f), predicted with the
  same `P0`: matched cycle work within ±20 % and ordered RMSE < 10 %.
- Reported beside it, not gated: the same identification with the armour law
  alone.

**Janocha et al. (2024) 66 kV, code-to-code only.**

- Fig. 6 (50 kN, ±0.2 1/m): loop half-height at κ = 0 with full accounting
  (wedge and filler profiles, MDPE μ 0.46 of Table 2), bounded and solved.
  Reported, not gated: the source is a finite-element model, not a test, and
  it states no initial contact state.
- Fig. 12b knee moments (M at 0.01 1/m) and its stick slope: retired as
  targets, for the reasons in the next subsection.

### Results against the registered targets

`python validation/scripts/check_core_friction.py` (one thread).

**Ménard and Cartraud (2023), identify on (b), predict the rest** (validation
row BM1-ID). The
identified initial pressure is 220 kPa with the full contact accounting and
275 kPa with the armour law alone (the source FE's calibration: 300 kPa).

| Range | Measured work [J/m] | Full accounting: work, ordered RMSE | Armour only: work, ordered RMSE | Target |
| --- | --- | --- | --- | --- |
| (b), identified | 62.3 | −0.0 %, 2.99 % | +0.0 %, 2.20 % | RMSE < 5 %: PASS |
| (a) | 20.2 | +26.0 %, 6.22 % | +35.8 %, 8.60 % | ±20 %, < 10 %: FAIL |
| (c) | 20.4 | +23.7 %, 7.05 % | +33.5 %, 8.32 % | FAIL |
| (d) | 22.6 | +12.9 %, 5.14 % | +21.7 %, 6.93 % | PASS |
| (e) | 65.0 | −6.5 %, 4.78 % | −6.1 %, 3.93 % | PASS |
| (f) | 69.2 | −11.7 %, 4.45 % | −11.4 %, 4.29 % | PASS |

The registered target fails on ranges (a) and (c), the two small-amplitude
ranges. The full accounting brings every small-amplitude prediction closer
than the armour law identified the same way ((a) +35.8 % → +26.0 %, (c)
+33.5 % → +23.7 %, (d) +21.7 % → +12.9 %) and leaves the large ranges (e)
and (f) unchanged within 0.6 %. What remains is an over-prediction of the work
of small cycles at the pressure that fits the larger ones.

**Janocha et al. (2024), code to code (50 kN, ±0.2 1/m, μ 0.46).**

| Model | Half-height at κ = 0, bounded / solved |
| --- | --- |
| Armour only | 0.112 / 0.081 kN·m |
| Laid cores, wedged trefoil | 0.246 / 0.181 kN·m |
| Laid cores and filler profiles | 0.380 / 0.374 kN·m |
| Source FE, Fig. 6 (read from the figure) | about 1.0 kN·m |

### Retired: Janocha et al. (2024) Fig. 12b knee moments and stick slope

The knee moments at 0.01 1/m (about 5.8 / 9.5 / 12.5 kN·m at 10 / 100 /
200 kN) and the initial slope of Fig. 12b were registered as targets in
gap R1b. They are retired, for three reasons read from the source's own
figures (about ±5 % of full scale):

1. **The two figures of the same model disagree.** At 0.2 1/m Fig. 6 (50 kN)
   gives about 6.8 kN·m. Fig. 12b gives about 13.8 kN·m at 10 kN and 17.0 kN·m
   at 100 kN, so about 15 kN·m at 50 kN, twice Fig. 6 at the same tension and
   curvature. The Fig. 12b knee at 10 kN (5.8 kN·m) is also several times the
   Fig. 6 friction moment at 50 kN (half-height about 1 kN·m), although a
   higher tension can only raise tension-induced friction.
2. **The stick branch is stiffer than any stuck section.** The Fig. 12b
   inset reaches about 3.0 kN·m at 1×10⁻³ 1/m and 5.0 kN·m at 2×10⁻³ 1/m at
   10 kN: a secant of 2,500–3,000 kN·m², above the 1653 kN·m² plane-section
   bound of the published section with every component stuck. No stick–slip
   law of the published geometry can follow that branch, and the knee
   moments at 0.01 1/m sit on it.
3. **The friction level of Fig. 6 depends on the mesh.** Its very coarse
   mesh gives a loop about three times as high as the normal and dense
   meshes, so the Fig. 6 level is kept only as a code-to-code value.

The IEA Wind Task 49 slipping stiffness (Table 13), whose row cites the same
source, is likewise reported code to code.

### Default of the laid-core law

The law stays off by default. The registered Ménard prediction fails on
the small ranges, and a cable without a declared clamping state has almost
no core friction at low tension, so switching the law on would change
results without a validated gain.

## Out and von Morgen (1997), frictionless slip bounds (OvM, Out89)

Sources: Out and von Morgen, Eng. Struct. 19(6) (1997) 507–515, Eqs. 20–30
and 41–42; Sævik (1992) thesis, Eqs. 2.29–2.30 and 4.2; Out (1989), OTC 6165,
Eq. 1 and Appendix Eqs. A8 and A11. Script `validation/scripts/check_out_von_morgen.py`, tests
`tests/test_out_von_morgen.py`, gates OvM and Out89 in the scorecard. A wire
with no friction follows the geodesic of the bent cylinder, so its slip
relative to the marked (deformed) helix bounds the slip of a wire held by
friction. Two boundary conditions are used: wire fixed at the intrados (a wire
remote from end fittings, Eqs. 26–28) and wire fixed at the neutral axis (the
extreme influence of an end fitting, Eqs. 20–22; twice the remote slip at the
far side). CableXSec is a sectional model without end fittings, so it is
compared with the intrados case and with the envelope of both.

| Check | Result | Status |
| --- | --- | --- |
| Closed forms against an exact numerical geodesic (Euler first integral, Eq. 7), 25–45° | Relative error 3.8k (intrados) and 3.0k (neutral axis), first order in k = κr as the source states; 3.8e-4 at k = 1e-4. The period length shortens by O(k²) | PASS |
| Wire-frame components of Eqs. 26–27 | Along the wire −s_δκ cos θ, the loxodromic slip of CableXSec (`geometric_slip_factor`, s_δ = r²cos²α/sin α); across it r²κ cos α(1 + sin²α) cos θ / sin²α, Sævik Eq. 2.30 (Sævik Fig. 4.2: 4.9 and 15.0 mm at 25°, R = 0.1 m, κ = 0.25 1/m) | PASS |
| CableXSec contact slip at μ = 0 (`MindlinContact` with `slip_increment`, η = 0), 15–55°, 24 azimuths | Longitudinal slip equals the geodesic's wire-direction component (1.1e-15); slip magnitude is sin α/(1 + 4 tan²α)^½ of the geodesic slip, at most 0.333; transverse slip zero | PASS |
| Approach as μ → 0, neutral-axis wire, 35° | Longitudinal slip / geodesic component 0.990, 0.997, 0.999, 1.000 at μ = 0.3, 0.1, 0.03, 0.001; never above 1 | PASS |
| Legacy transverse map (`transverse_map="legacy"`), η = 0.5–2 | Within the end-fitting envelope (largest ratio 0.71 at η = 2). It is in quadrature with the geodesic transverse slip: η s_δ κ at the extreme fibres, where the geodesic gives none, and zero at the neutral axis, where the geodesic gives (1 + sin²α)/(sin α cos α) = 2.8–4.3 times the longitudinal slip (15–55°). Kept for reproduction; selecting it implicitly warns | Disagreement in phase, stated |
| Geodesic transverse map (`transverse_map="geodesic"`, η = 1), 15–55°, 24 azimuths, μ = 0 | Slip vector equals Eqs. 26–27 with CableXSec's curvature sign (residual 1.1e-15); δ* over a ±κ cycle equals \|Δ\| of Eq. 28 (2.7e-15); transverse slip zero at the extreme fibres and s_g κ at the neutral axis, s_g = r²cos α(1 + sin²α)/sin²α; against the exact geodesic the error is first order in k (2.6e-3 and 2.6e-4 at k = 1e-3 and 1e-4, 35°); approached monotonically as μ → 0 | PASS |
| Equilibrium transverse-slip cell of the companion article (`validation/scripts/transverse_friction_cell.py`), published three-core cable, ramp to 0.10 1/m, geodesic feedback | Transverse slip in phase with the geodesic slip and below it at every station: 0.9–1.0% of it at μ = 0.2, rising to 9%, 13–14% and 20–25% at μ = 0.02, 0.015 and 0.012; longitudinal slip 0.97–0.999 of the loxodromic value. The demand vanishes on the geodesic (residual 4e-16). Below μ ≈ 0.01 the cell stops at gross transverse slip, so μ → 0 is not reached by the cell | PASS within its range |
| Local-bending curvature coefficients (`local_bending_fibre_stress`) against the exact curvature of the deformed helix (numerical, 20–50°) | Normal cos²α cos 2α equals Eq. 41; binormal cos α(1 + sin²α) agrees with the numerical value to 4e-5. Eq. 42 as printed in the library copy, cos α (3 sin²α + 1)^½, is 4–6% above the exact value | PASS; source Eq. 42 not used |
| Out (1989): plastic slip of one contact under constant-amplitude cycling, in curvature units | Δκ − 2κ_s with κ_s = μF_n/(K_t s_δ \|sin φ\|), residual 9.6e-16 | PASS |
| Out (1989): starting point uniform on the no-slip line | Mean damage ratio through the contact element equals A8 (a ≤ 1) and A11 (a > 1) for c = 1–3 within 5e-7 (midpoint rule) | PASS |

Contra-wound layers: Eq. 30 gives the relative slip 4κr²/tan α. CableXSec
computes the slip of a wire relative to its supporting surface and not at a
crossing of two layers, so Eq. 30 has no CableXSec counterpart; the loxodromic
components of two contra-wound wires differ by cos²α/2 of it.

The frictionless bound removes the qualification that transverse slip had been
compared with finite-element results only: the computed transverse slip is
below the geodesic bound everywhere, which is an analytical upper limit and
does not depend on neighbouring-wire contact. It does not show that the
frictional transverse slip itself is correct; Sævik (1992, Figs. 4.9–4.11)
reports more transverse slip near the limiting friction than his
Eq. 4.1, and no measured transverse slip of an armour wire is held.

## Sævik and Bruaseth (2005), umbilical axial stiffness (BM4)

| Class | Quantity | Target | Result | Status |
| --- | --- | --- | --- | --- |
| M | Fixed-twist EA of the published umbilical population, model/measured | Mean ratio < 1.25 | Mean ratio about 1.45; umbilical U1 above 2.0 | NOT VALIDATED |

The CableXSec calculation has no radial interaction between the umbilical
elements and behaves as the source's no-radial-interaction analytical bound;
the source FE with radial interaction gives a mean ratio of 0.946. Retained as
a negative finding.

## CIGRE TB880 thermal benchmark (BM5)

| Class | Quantity | Reference | Result | Status |
| --- | --- | --- | --- | --- |
| MM | Conductor temperature, three-core 220 kV cable, tabulated thermal network of Poradowski (2025) with circulating-current sheath loss only | FEM 89.62 °C; IEC 90 °C | 84.4 °C, about 5 K low | PARTIAL |

The eddy-current loss term is not included, and the IEC 60287-2-1 three-core
geometric factor is not yet applied, so `compute_thermal` on the TB880 cable
input reads 136.1 °C with the BM5 inputs (908.45 A, three cores, λ1′, T4 = 0.3211 K·m/W, 15 °C; after the filler is represented as an equal-area annulus, which overestimates the filler resistance: 0.166 against 0.112 K·m/W). The comparison uses the cited tabulated network, not
the cable definition.

## Temperature history coupled into clamping and fatigue (gap 4.1)

Gates of `tests/test_thermal_history.py` for the slow-time driver
(`mechanics/thermal_network.py`, `analysis/thermal_coupling.py`).

| Class | Quantity | Reference | Result | Status |
| --- | --- | --- | --- | --- |
| A | Steady state of the transient RC ladder at the BM5 inputs (908.45 A, three cores, λ1′, `T4` 0.3211 K·m/W, 15 °C, `R_ac` at 90 °C) | `compute_thermal` 136.06 °C (IEC-consistent profile) | Equal to 1e-12 at 1, 3 and 8 sub-annuli per layer; every annulus edge to 1e-11; heat through `T4` equals the heat generated to 1e-10 | PASS |
| A | Same, 66 kV three-core template at 1093 A | `compute_thermal` 88.85 °C | Equal to 1e-9 (conditioning of the thin copper screens) | PASS |
| A | Crank–Nicolson energy balance of the transient network | Discrete identity `ΣC ΔT = h/2 (Q_n + Q_n+1)` | Round-off (1e-9) | PASS |
| A | TR-BDF2 order; steady hold; step response | Second order; steady state | Error ratio 3–5 on halving the step; drift < 1e-8 K; step reaches the steady state to 1e-3 K | PASS |
| A | Coupled `R_ac(T_c)` | Fixed point | Re-evaluating `R_ac` at `T_c` changes no node by more than 1e-10 | PASS |
| A | Thermal mismatch contact pressure, solid polymer core in an aluminium tube | Lamé closed forms, plane strain and generalised plane strain (zero axial force) | 1e-9 at ΔT = 40 and 10 K; contact open at −20 K; same through `ThermalState` → solved radial state of a two-layer cable definition | PASS |
| A | Isothermal limit (all layers at `T_ref`) | Unenergised solved results | Fatigue damage per wire and per fibre bit for bit (with and without tension, single state and two-node history); rig gauge strains bit for bit | PASS |
| A | Monotonic trends, 66 kV template at 25/40/55 °C | Expansion raises clamping | P1+P2, slip-onset curvature and thermal wire stress rise in both armour layers; transferred softening lowers `k_f` at 30/50/70 °C | PASS |

Application to the public 66 kV template (0, 50 and 100 % of 1093 A, daily
cycle, energised rig): [template_66kv.md](template_66kv.md) Section 4.2.
Not yet compared with measurements.

## Global–local consistency, floating-wind application

Internal comparison in the turbine-attached lazy-wave cable (IEA 15 MW turbine
on the VolturnUS-S platform, published Ménard and Cartraud construction).
Operating hours use the full-slip global bending stiffness (3.571 kN·m²);
the local stick stiffness is 33–89 kN·m². The reference hour was repeated with
a global hysteretic bending law built from the section backbone with Masing
unloading (transferred contact state, combined clamping 0.6 MPa).

| Quantity | Result |
| --- | --- |
| Axial rigidity, both models | 280.204 MN; material force recovered within 7.3e-12 N |
| Torsion added with linear bending | Largest damage changes ≤ 6.3%, except the touchdown region (curvature frame) |
| Curvature range at the full-slip governing section, s = 263.478 m | 0.039 to 0.015 1/m |
| Damage at s = 263.478 m | 0.20 (outer) and 0.15 (inner) of the full-slip value |
| Governing section | Moves to the sag bend, s = 158.0 m |
| Governing damage | 0.33 (outer) and 0.26 (inner) of the full-slip value |
| Conservatism of full-slip law | About 3–4 for the governing damage; 5–7 at s = 263.478 m, hang-off and stiffener mid-length |
| Mesh and time step halved / fibres doubled | Governing damage changes 6.38% and 6.52% / 0.145% and 0.157% |

The full-slip damages of the 24 operating hours are therefore upper estimates.
The consistent run covers one hour, and the Masing rule approximates the biaxial
slip-front law.

Sealed against flooded armour annulus, damage ratio sealed/flooded
(loop-identified state, μ = 0.2, seawater S–N curve):

| Section | Fixed curvature (inner / outer) | Consistent bending (inner / outer) |
| --- | --- | --- |
| Hang-off, stiffener, buoyancy ends | 1.0 / 1.0 | 0.89–1.08 / 0.88–1.08 |
| Sag bend, s = 158.0 m | 1.4 / 1.9 | 0.93 / 1.01 |
| Hog bend, s = 191.5 m | 1.5 / 2.1 | 1.10 / 1.10 |
| Touchdown region, s = 260.478 m | 2.3 / 3.9 | 0.26 / 0.25 |
| Governing section, s = 263.478 m | 2.6 / 4.5 | 0.14 / 0.16 |

The increase found with fixed curvature is removed once each annulus condition
has its own global bending law.

## Fretting regime and wear (gap 3.1: FR, FOU95, WEAR-CW)

Code `cablexsec/analysis/fretting_regime.py`, `fretting_assessment.py` and
`materials/wear.py`; script `python validation/scripts/check_fretting_regime.py`
(targets in its docstring, set before the runs); tests
`tests/test_fretting_regime.py`; guide
[local_analysis_guide.md](local_analysis_guide.md) Section 11.

| ID | Class | Quantity | Target | Result | Status |
| --- | --- | --- | --- | --- | --- |
| FR | A | Mindlin energy per cycle (closed form) vs the numerical Masing loop of `mindlin_monotonic`, six amplitudes 0.05–4 δ_t | < 1e-9 rel. | 7.8e-12 | PASS |
| FR | A | Fouvry ratios at the transition, A = 0.2, D = 2(1/2)^{2/3} − 1, B = A/D | < 1e-12 | 4.4e-16 | PASS |
| FR | A | Small-amplitude energy vs Johnson (1985) Eq. 7.63 | O(x) rel. | (tests) | PASS |
| FR | A | Cattaneo strip integral I(0) = π/16; I(γ) vs adaptive quadrature; slip profile vs its gradient integral | < 1e-12; < 1e-9; < 1e-10 | 1.6e-15; (tests) | PASS |
| FR | A | μ = ∞: stick, zero energy and wear; μ → ∞: energy and wear → 0 | 0 J | 0 J | PASS |
| FR | A | Rotation of a planar slip path: amplitudes, regimes and wear unchanged | < 1e-12 rel. | 4.4e-16 | PASS |
| FOU95 | MM | Mindlin transition amplitude, Fouvry et al. (1995) Table 4 (HSS flat, 25.4 mm steel ball, 200–500 N), printed inputs | 3.2, 4.1, 4.6, 5.2 µm ± 0.1 | 3.25, 4.17, 4.63, 5.25 µm | PASS |
| WEAR-CW | M | Energy wear volume of crossed 0.45 mm steel wires at 45°, 1 N, μ 0.7, 50 000 cycles of ±65 µm, k = 2.93e-8 MPa⁻¹ (Poon et al. 2024 Tables 4–5, data of Cruzado et al. 2013) vs the measured scar (154 × 381 µm, depth 18 µm, elliptic paraboloid) | ratio 0.8–1.25 | 0.913 | PASS |

Stated rather than changed:

- **FOU95 shear moduli.** Fouvry et al. (1995) Table 3 prints G = E/(2(1 − ν))
  (159.7 and 146.5 GPa). With G = E/(2(1 + ν)) the same Mindlin formula gives
  5.8, 7.4, 8.2 and 9.3 µm, 1.6–1.7 times the measured transitions (3.6–5.4 µm
  after removal of the rig compliance). The gate reproduces the printed
  calculation; the published agreement with the measurements rests on the
  printed moduli.
- **WEAR-CW is a consistency check.** Poon et al. (2024) identified k on the
  same tests, so the gate checks the energy-to-volume chain and the stroke
  reading (130 µm peak to peak); with ±130 µm the ratio is 1.83. The mean
  depth over the unworn swept area is 105 µm against the measured 18 µm
  maximum: the reported depth rate is an upper estimate.
- **No regime test on cable contacts.** No published fretting test of steel
  on polyethylene, polypropylene or polyamide is in the library; those wear
  coefficients are assumptions (10⁻¹³ m³/J, a decade either side). The
  knock-down slip constant k = 2.7 µm is an assumption fitted to the 3–8 µm
  minimum-life range of Nakazawa et al. (1994, Fig. 3).
- **Locations.** Under plane bending the slip, and therefore wear, is largest
  at the neutral axis and zero at the extreme fibre, where the stress range
  and the knocked-down damage are largest (test
  `test_compute_fretting_66kv_locations`). This is consistent with the only
  armour-fretting failure locations in the library, which are qualitative
  (Sævik and Berge 1995, Figs 8, 28); it is not a validation.

## Conductor trellis contact (gap 1.3c: G1, G2)

Code `cablexsec/analysis/conductor_fatigue.py`
(`trellis_contact`, `trellis_force_ratio`); gates registered before the run;
tests `tests/test_conductor_fatigue.py`; guide
[local_analysis_guide.md](local_analysis_guide.md) Section 10. Nothing fitted.

Contact model as corrected: crossing force from the interface line load and
the crossing density (Raoof and Hobbs 1988, Eqs. 23–25; unchanged, 177.6 N
and 41.9 N for the 300 mm² conductor at 30 kN); tangential force from the
local stick shear flow, zero at the extreme fibre (Papailiou 1997); peak
pressure limited to 3 σ_y (Johnson 1985 §6.3), σ_y 300 MPa (300 mm²,
Nasution 2014b Fig. 9) and 225 MPa (95 mm², Nasution 2013 p. 249). Lives are
1/(per-cycle damage) of the most damaged ring, bulk + fretting, against the
geometric mean of the tests.

| ID | Class | Case | Test | Prediction | Ratio | Status |
| --- | --- | --- | --- | --- | --- | --- |
| G1 | M | A1 300 mm², ρ 3 m, dry | 2.33 × 10⁵ | 6.08 × 10³ | 0.026 | FAIL |
| G1 | M | A2 300 mm², ρ 6.5 m, dry | 8.32 × 10⁵ | 6.20 × 10³ | 0.0075 | FAIL |
| G1 | M | B1 95 mm², ρ 1.5 m, dry | 1.28 × 10⁵ | 1.56 × 10⁴ | 0.12 | FAIL |
| G1 | M | B2 95 mm², ρ 3 m, dry | 1.32 × 10⁶ | 1.66 × 10⁴ | 0.013 | FAIL |
| G1 | M | Radius trend T1 = A2/A1 (band 1.8–7.2) | 3.6 | 1.02 | — | FAIL |
| G1 | M | Radius trend T3 = B2/B1 (band 5.15–20.6) | 10.3 | 1.06 | — | FAIL |
| G2 | M | A3 300 mm², ρ 3 m, lubricated (μ 0.02) | 3.97 × 10⁵ | 7.6 × 10⁶ | 19 | reported |
| G2 | M | Lubrication effect A3/A1 | 1.7 | 1250 | — | reported |

Outcome and decision (rule registered with the gates): G1 fails, so the
default of `compute_conductor_fatigue` is the bulk S–N count; trellis SWT is
available only with `trellis_fretting=True` and emits
`UnvalidatedModelWarning`. With the same set-up the bulk count gives A1 ×1.2,
A2 ×2.6, B1 ×2.0 (×0.89 with the 95 mm² wire curve of Nasution 2013 Eq. 1),
B2 ×5.0 (×2.3), A3 ×20; radius trends 8.0 and 27 (95 mm² curve) against 3.6
and 10.3, lubrication effect 29 against 1.7.

Stated rather than changed:

- **Why the contact rule fails.** Every dry test is in gross slip at the
  neutral-axis crossings (curvature range 1.7–15 times the onset range
  2κ_c), and the pressure sits at its limit
  (900 and 675 MPa against elastic 1989 and 1385 MPa). The driver 2μp0 is
  then fixed (360 and 270 MPa), independent of curvature, whereas the dry
  lives need 95–160 MPa. The limit raised the lives about tenfold (elastic:
  551 and 1600 cycles); the shear-flow distribution changes no dry life,
  because the neutral-axis crossings slip in both readings.
- **Crossing force.** The Nasution FE point forces (43.6 and 21.8 N, 2014b
  Table 4; 12 N for the 95 mm² conductor, 2014a Table 2) are 1.9–4.1 times
  lower than the line-load reading. With the FE wire forces (2014b Fig. 30,
  795 N in the outer layer) the outer-layer clench is 59.6 kN/m, while
  21.8 N × 1490 crossings/m carries 32.5 kN/m and the in-layer forces of
  that layer are 0.03 N/mm; the FE values are therefore not forces per
  physical crossing and are not adopted. For the 95 mm² FE the in-layer hoop
  force of the outer layer (1.8 N/mm, the Raoof–Hobbs Eq. 19 term) closes
  80 % of the balance. Diagnosis with the FE forces: the elastic pressure
  falls to 1245 and 951 MPa, but the limit still governs (900 and 675 MPa),
  so the lives are unchanged (6.2 × 10³ and 1.7 × 10⁴).
- **Lubricated test.** A3 at μ = 0.1 (Nasution's own suggestion, diagnosis
  only) gives 7.3 × 10⁴ (×0.18). No μ gives both A1 and A3 with a
  friction-driven driver; the lubricant and its friction are not reported,
  and the mismatch is not fitted.
- **Transcription.** The registered targets list the A2 test mean as 838 000
  cycles; the geometric mean of the three lives is 832 000 and is used. The
  outcome does not depend on it.
- **Not modelled.** In-layer hoop share of the crossing force; spreading of
  slip from the neutral axis by accumulated wire displacement (Papailiou
  1997, Eq. 15); bulk stress at the contact; compaction-formed contact
  patches; critical-plane SWT; worn geometry. An elastic–plastic crossed-wire
  FE (3.21 mm, about 12°, 40–180 N) is the next step if contact fretting is
  to be re-gated.

## Conductor fatigue version 2 (gap CF2: CF2-L, CF2-T, CF2-R, CF2-F, CF2-X)

Code `cablexsec/analysis/conductor_fatigue.py`
(`CriticalDistanceSpec`, `critical_distance=` of `assess_conductor_fatigue`;
`compute_conductor_fatigue(mean_stress="swt")` default); targets registered
before the first run; tests `tests/test_conductor_fatigue.py`. Nothing is fitted to the test lives.

Model. Bulk: the stick–slip outer-fibre stress of every ring and azimuth,
rainflow counted with the Smith–Watson–Topper correction to the stress ratio of
the conductor's own wire curve (300 mm²: Nasution 2014b Table 1 fit; 95 mm²:
Nasution 2013 Eq. 1, registered as `nasution_2013_copper_95`). Trellis
(opt-in): at every crossing of oppositely laid rings the contact after plastic
indentation (mean pressure 3σ_y, Hertz-shaped reloading on the enlarged
patch) and the sliding traction field (Johnson §3.6 kernels) are added to the
bulk stress at depth L/2 below the contact, L = (1/π)(ΔK_th/Δσ₀)² with Δσ₀ the
wire-curve range at 10⁷ cycles and ΔK_th 3.8 MPa√m (transgranular; band 2.1,
intergranular; Tromans and Sun 1996, annealed copper, R = 0.33), and every plane
in the wire–depth plane is counted with the same SWT correction and curve.

| ID | Class | Case | Test | Bulk + SWT | Trellis, ΔK_th 3.8 (L 287/366 µm) | Trellis, ΔK_th 2.1 (L 88/112 µm) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CF2-L | M | A1 300 mm², ρ 3 m | 2.32 × 10⁵ | 4.27 × 10⁵ (1.84) | 1.84 | 1.51 | PASS |
| CF2-L | M | A2 300 mm², ρ 6.5 m | 8.32 × 10⁵ | 1.99 × 10⁶ (2.39) | 2.39 | 1.11 | PASS |
| CF2-L | M | B1 95 mm², ρ 1.5 m | 1.28 × 10⁵ | 1.85 × 10⁵ (1.45) | 1.45 | 1.45 | PASS |
| CF2-L | M | B2 95 mm², ρ 3 m | 1.32 × 10⁶ | 2.06 × 10⁶ (1.56) | 1.56 | 1.56 | PASS |
| CF2-T | M | A2/A1 (band 1.8–7.2), B2/B1 (5.15–20.6) | 3.6, 10.3 | 4.65, 11.1 | 4.65, 11.1 | 2.62, 11.1 | PASS |
| CF2-R | M | Ring of first failure, dry bending | 2nd/3rd layer; inner layer | ring 1 in all groups | ring 1 | ring 1 | PASS |
| CF2-F | M | Face of initiation | towards the outer layer | (bulk governs) | outer face; bulk governs | outer face governs in A1, A2 | reported |
| CF2-X | M | Tension–tension 95 mm², Δσ 160 / 190 MPa | 8.22 × 10⁵ / 3.76 × 10⁵, outer layer | 9.5 × 10⁵ (1.16) / 3.25 × 10⁵ (0.86), centre wire | same | same | lives PASS, location FAIL |
| CF2-G | M | A3 lubricated (μ 0.02 assumed) | 3.97 × 10⁵ | 4.97 × 10⁶ (12.5) | 12.5 | 11.3 | reported |

Stated rather than changed:

- **What the bending agreement rests on (found after the gate).** Both wire curves are fitted to outer wires, whose compaction
  indentations carry a stress concentration of 1.13–1.20 (Jiang et al. 2022,
  Table 1); inner wires carry 1.02–1.03 and the centre wire none (Nasution et
  al. 2013, p. 255). Applied to ring 1, the outer-wire curve therefore acts as
  an implicit knock-down. With the curve corrected for the indentations
  (Nasution 2013: m = 8.424, C = 2.88 × 10²⁵) and each ring's own factor, the
  bulk count moves the tension–tension failure to the outer layer (lives
  ×1.0–2.0) but gives the dry bending lives ×2.6–11 too long (radius trends 8.2
  and 28). The bending tests therefore contain a damage term at the inner
  crossings that the bulk stress does not carry: the agreement above is a
  calibration by proxy, not a validation of the inner-layer mechanics. Gap CF3
  (fretting damage at the crossings on the corrected curve) is the follow-up.
- **Choices made after an exploratory run.** A prototype run, after the
  targets were registered and before this gate run, selected
  the SWT correction to the curve's stress ratio and each conductor's own wire
  curve. Without the correction the lives are ×0.89–2.58 and the radius trends
  8.0 and 27 (FAIL). No other mean-stress correction could be checked: the
  library holds only annealed-copper strengths (220–235 MPa), below the 300 MPa
  proof stress of these wires.
- **The trellis term changes little.** The crossings are long, narrow patches
  (semi-axes about 0.4–1.0 mm by 30–60 µm after indentation), so the contact
  field has decayed at L/2 for the transgranular threshold and the bulk stress
  governs; only at the lower threshold does it govern A1 and A2. The surface
  evaluation of gap 1.3c, without a material length, is what made the contact
  rule one to two orders too severe.
- **Tension-only location.** The lives agree, but the three layers' lives
  differ by 14 % and the centre wire comes first; the tests failed in the
  outer layer at the indentations left by compaction (Nasution 2014b), whose
  stress concentration (1.02–1.20, Jiang et al. 2022) is not modelled.
- **Lubricated group.** The source does not report the lubricant's friction.
  The bulk + SWT life is ×7.6, ×3.7 and ×1.9 at μ = 0.05, 0.1 and 0.15.
- **Published copper fretting methods** (Poon et al. 2022–2024; Wang et al.
  2025, 2026) were reviewed and not adopted: their averaging size comes from an
  Al–Cu alloy, their SWT constants are to-failure wire data, and none predicts a
  conductor test.
- **Wear** of the crossings is not modelled.

## Conductor fatigue version 3 (gap CF3: CF3-L, CF3-T, CF3-R, CF3-X)

Targets registered before the gate run, with the exploratory runs of the
same model disclosed; nothing fitted. Opt-in inputs:
`CoreStrandRing.compaction_scf` (`ConductorRing.compaction_scf`), the curve `nasution_2013_copper_corrected`
(Nasution et al. 2013, p. 255: m = 8.424, C = 2.88 × 10²⁵ against the actual longitudinal range), the SWT
correction, and `CriticalDistanceSpec()` whose default threshold is now 2.55 MPa√m (full-hard ETP copper at
R = 0.1, Liaw et al. 1982, Table III). The designed 66 kV conductor takes `compaction="light"` or `"heavy"`.
The default conductor assessment is unchanged (gap CF2).

| Case | Test | Heavy compaction | Light compaction |
| --- | --- | --- | --- |
| A1 300 mm², ρ 3 m | 2.32 × 10⁵ | ×2.05, ring 1 | ×2.14, ring 1 |
| A2 300 mm², ρ 6.5 m | 8.32 × 10⁵ | ×2.15, ring 1 | ×2.24, ring 1 |
| B1 95 mm², ρ 1.5 m | 1.28 × 10⁵ | ×4.14, ring 2 | ×4.75, ring 1 |
| B2 95 mm², ρ 3 m | 1.32 × 10⁶ | ×9.02, ring 1 | ×9.45, ring 1 |
| Trends A2/A1 (3.6), B2/B1 (10.3) | | 3.74, 22.5 | 3.73, 20.5 |
| Tension–tension 160 / 190 MPa (outer layer) | 8.22 × 10⁵ / 3.76 × 10⁵ | ×1.90 / ×0.97, outer ring | ×3.20 / ×1.64, outer ring |
| A3 lubricated (reported) | 3.97 × 10⁵ | ×36 | ×41 |

Gate at the nominal threshold and both compaction levels: CF3-L, CF3-T, CF3-R and CF3-X FAIL. At 2.35 MPa√m
(R = 0.3) the 300 mm² lives are ×1.67–1.76 and the 95 mm² ×4.0–7.3.

What it shows:

- With each ring's own compaction factor the tension–tension failure moves to the outer layer, as observed,
  and the 300 mm² tension–bending lives and radius trend fall in their bands with no fitted input.
- The 95 mm² tension–bending tests remain four to nine times longer in the model than in the tests, and their
  radius trend twice too steep. The crossing force is the first suspect: 37 N in this model against 12 N in the
  published FE of the same conductor (Nasution et al. 2014a, Table 2), which in the model raises the contact
  pressure but at the material length (86 µm, larger than the 30–60 µm semi-width of the indented crossings)
  barely reaches the crack plane.
- The lubricated group is far off in every version; the lubricant's friction is not published.

## Hysteretic bending law for global analysis (gap MX)

Code `cablexsec/analysis/bending_law.py` (`export_bending_law`, `IwanBendingLaw`,
`BendingLawExport`); targets registered before the run; tests `tests/test_bending_law.py`;
guide [global_local_guide.md](global_local_guide.md), Section 8.

| ID | Class | Case | Result | Status |
| --- | --- | --- | --- | --- |
| MX-1 | I | Ménard cable at 0 and 50 kN, 66 kV template at 50 kN; cycles at 0.1, 0.4 and 1.0 of 0.2 1/m and an irregular history; 200 backbone points | max moment difference 0.0008–0.05 % of the range; work within 0.00003–0.02 % | PASS |
| MX-2 | I | Backbone at 50 kN interpolated between 0 and 100 kN against the direct backbone (Ménard) | 3.9 % of the range (spacing 50, 20, 10 kN: 1.5, 0.27, 0.07 %; 66 kV template 0.81 % at 100 kN) | FAIL |
| MX-3 | I | Stick tangent, full-slip slope and virgin backbone of the decomposition | 1e-16 to 2e-14 | PASS |
| MX-4 | I | Progressive law, Ménard at 50 kN | ≤ 0.06 % of the range, work ≤ 0.02 % | reported |
| MX-5 | MM | Hysteretic bending of a global program with the exported table | not run | open |
| MX-scale | I | Slip scaling variable (`slip_scale=True`): bound homogeneity of `compute_mkappa` (66 kV, 20 → 70 kN, cores); group elements at 60 kN against the 10 kN elements with yield curvatures times `ρ_g` (Ménard); tension cycle 10–70 kN inside ±0.03 1/m bending, scaled law against the contact-tension history over the fixed-tension residual | 4.1e-16; 8.3e-14; 1.02 (symmetric wire bounds on a breakpoint grid: 8e-15, work 7e-15) | PASS |
| MX-scale-1 | I | One scale for all groups against the per-group law, 10–150 kN (reference 10 kN), backbone and cycles to 1.5 times full slip | 66 kV 0; 33 kV three-core 0.10 % (0.29 % with cores); Ménard 12 %; with p_e 0–2 MPa: 66 kV 21 %, 33 kV 12 %, Ménard 19 % | reported |

The armour laws are parallel spring–sliders at fixed contact pressure, so the Iwan law follows Masing's rules; it is
exact when every wire's slip interval is symmetric. The signed capstan bound makes the interval asymmetric (the
falling bound is the rising one times `exp(−λψ)`), and the MX-1 residual comes from that besides the backbone
discretisation (bending at ±0.03 1/m on the Ménard cable: 0.14 % of the range at 10 kN with 300 points, 0.02 % with
symmetric bounds). The interpolation in tension is the other approximation; the slip scale replaces it with the
exact scaling of the friction caps (MX-scale; guide Section 8.1). Biaxial bending is outside it (per-wire,
direction-tied response in CableXSec).

## Cable over a sheave (gap SH)

Code `cablexsec/analysis/sheave.py` (`compute_sheave`, `sheave_contact`, `groove_pressure`);
targets registered before the run; tests `tests/test_sheave.py`; guide
[local_analysis_guide.md](local_analysis_guide.md), Section 8.2.

| ID | Class | Case | Result | Status |
| --- | --- | --- | --- | --- |
| SH-1 | A | Feyrer (2015) Example 3.2: tape 30 × 2 mm, S = 10 kN, D = 800 mm | 2.862°, 499.4 N, 9 988 N, 25 N/mm, 1.0 mm (printed 2.86°, 500 N, 9 988 N, 25 N/mm, 1 mm) | PASS |
| SH-2 | A | Groove factors, groove not undercut and Example 3.4 (47.5°, 80°) | 1.3512, 1.1701; 5.6265, 2.0615 (printed 1.351, 1.170; 5.627, 2.062) | PASS |
| SH-3 | I | Curvature on both diameter bases; stresses, utilisation and per-passage damage against the section analyses | round-off | PASS |
| SH-4 | I | 66 kV template, 50 kN, D = 4 and 6 m | κ 0.50 and 0.33 1/m; p₀ 136 kPa at 4 m; utilisation 0.448 and 0.298; largest wire stress 160 and 112 MPa; damage per passage 6.6e-7 and 1.3e-7; run-on angle outside the small-angle range | reported |

SH-1 and SH-2 check the transcription of Feyrer's closed forms, not their applicability to a polymer-sheathed cable.
The groove pressure is a global value for the sheath; the run-on peaks measured on steel wire ropes are not used.

## Armour compression (ST17, OST12, LV21, YANG15, MALTA22)

Gap R5 / 3.2. Code `cablexsec/analysis/compression.py`;
script `python validation/scripts/check_compression_buckling.py` (sources,
targets and tolerances in its docstring, recorded before the implementation
was run); tests `tests/test_compression_buckling.py`; guide
[uls_guide.md](uls_guide.md). Scorecard rows (Tier 2): ST17-T3, LV21, ST17-P34,
YANG15, ST17-SOUSA and OST12. Reproducing a published closed-form value is a
model-to-model comparison under the validation policy; it checks the
implementation against the source, not the physics. Flat wires use Roark's
torsion constant; the sources do not state theirs.

| ID | Class | Quantity | Target | Result | Status |
| --- | --- | --- | --- | --- | --- |
| ST17-T3 | MM | Eq. 19, both layers, Østergaard 6, 8, 14 in pipes (Sævik–Thorsen Table 3, analytical column) | 100, 243, 143 kN ± 3% | 98.0, 241.9, 141.1 kN | PASS |
| OST12 | MM | Eq. 19 per layer against the torsionally fixed–free plateaus with transverse slip in both layers (Østergaard et al. 2012, Fig. 14, read from the figure) | inner 49.5, outer 48, sum 98 kN ± 5% | 50.0, 48.0, 98.0 kN | PASS |
| LV21-Pcr | MM | Li–Vaz Eq. 11, umbilical wire (D 4.1 mm, r 35.75 mm, 20°); × 106 wires | 1563 N ± 0.1%; 166 kN ± 0.5% | 1563.1 N; 165.7 kN | PASS |
| LV21-slip | MM | Eq. 15 first-slip curvature at Pcr, μ = 0.05, 0.1, 0.15 | 3.2, 6.5, 9.7e-4 1/m ± 2% | 3.25, 6.50, 9.75e-4 1/m | PASS |
| LV21b | I | Cyclic-curvature knock-down table (Li–Vaz 2021b Table 3) at its nodes | 0.91, 0.88, 0.83 | exact | PASS |
| LV21-LB | MM | Sævik Eq. 19 below the Li–Vaz FE lateral limits (lower-bound property) | P1/Pcr below the smallest FE ratio, 0.82 (2021b Table 3) | 0.613 | PASS |
| ST17-P4 | MM | Sousa wire yield load (Eq. 34; 5.36 × 256 kN) | 1372 kN ± 0.5% | 1373 kN | PASS |
| ST17-P3 | MM | Sousa hoop-capacity load (Eq. 33): tape only; tape and sheath | 1020, 1110 kN ± 6% | 1065, 1152 kN | PASS |
| ST17-SOUSA | M | Straight-beam P2* (tape and sheath) against the de Sousa test mean | 256 kN ± 10% | 252.5 kN (−1.4%) | PASS |
| ST17-SOUSA-cb | M | Curved-beam P2* (tape and sheath) against the test mean | reported | 306.5 kN (+19.7%) | Reported |
| YANG15 | MM | Tape failure load, Yang et al. Eq. 9, 9.5 in pipe | 1.84 MN ± 3% | 1.817 MN | PASS |
| MALTA22 | MM | Sheath-only P2* (straight beam, sheath yield 20.74 MPa) against FE with the tape cut over 2000 mm (radial mode, 241.69 kN) | reported | 123.2 kN (−49%) | Reported (conservative) |

Friction trend (tests): above first slip the lateral capacity is the same for
μ = 0.05, 0.1 and 0.15, as Li and Vaz (2021a, Table 4: 0.83–0.85 Pcr) found;
the first-slip curvature is linear in μ and rises with hydrostatic clamping of
a sealed annulus; below it the lateral mode is closed and radial buckling
governs. Units: the closed forms give the same forces in N–m–Pa and N–mm–MPa.

Disagreements stated rather than changed:

- **Sævik and Thorsen (2017) Table 5, per-term P2.** From the printed Table 4
  inputs the straight- and curved-beam P2 are 19–33% above the tabulated values
  with all 95 wires in Eq. 22, and 6–22% below them with one layer's wires;
  neither convention reproduces all four rows (P2*, normalised: CB tape and
  sheath 1.20 here against 1.08; SB 0.99 against 0.82; CB tape 1.07 against
  0.91; SB tape 0.83 against 0.72). The tape modulus is printed as 1480 MPa in
  Sævik and Thorsen Table 4 and as 14 900 MPa (axial stiffness 1341 kN, failure
  strain 2.1%) in Gjøsteen et al. (2027) Table 6 for the same specimens; the
  gates use the printed 1480 MPa. CableXSec uses all wires of the stack, which
  is the self-consistent reading of Eqs. 22–23 for two layers that buckle
  together. The straight-beam model, which the source ranks conservative, is
  the default.
- **Hoop-capacity P3** is 3.8–4.4% above the quoted 1110 and 1020 kN; the
  source does not state the radii or the tape angle convention used.
- **Malta et al. (2022).** Their FE has friction 0.2, an elastic–plastic HDPE
  sheath with hardening and part of the tape intact outside the cut; the
  interaction formula with P3 at sheath yield is about a factor of two below
  it. The case is the only published analogue of a sheath-restrained power
  cable and is reported, not gated.
- **Small lay angles.** The reproduced cases span 20–38°. Power-cable armour
  is often laid at 10–18°; both lateral closed forms fall with sin²α and omit
  the buckle-length term, so the screen warns and the result is a lower bound
  there. No published lateral-buckling test or FE of a power cable is in the
  library.

## Witz (1996) 2.5-inch flexible pipe (WITZ-EI, WITZ-ID)

Gap 2.1 and the loop-energy mechanisms of gap 2.3. The
eight-layer ISSC case-study pipe of Witz (1996, Table 2), with the outer-armour
lay direction, steel moduli and carcass lay angle of Ramos and Pesce (2004,
Table 3 and Section 4), is packaged as `witz1996.yaml`
(`cx.witz1996_yaml_path()`; mapping in its header: area-equivalent round wires,
near-hoop helices for the interlocked carcass and the Zeta pressure armour,
isotropic tapes). The measured loops of Figs 5 (0 bar) and 6 (300 bar) were
digitised from a 600 dpi render (`validation/paper_data/witz1996_fig5_0bar.csv`
and `witz1996_fig6_300bar.csv`; reading resolution ±0.008 1/m and ±10 or
±15 N·m; source SHA-256 in the file headers). Targets were registered before
any model output existed. Witz reports no test procedure, specimen length, end
condition, temperature, friction coefficient or initial contact pressure. The
300-bar case is taken as closed-ended: wall tension 30 MPa × π × (35.1 mm)² =
116.1 kN on the pressure-sheath bore. Scorecard rows WITZ-EI and WITZ-ID;
tests `tests/test_witz1996_benchmark.py`.

| ID | Class | Quantity | Target | Result | Status |
| --- | --- | --- | --- | --- | --- |
| WITZ-EI | M | Full-slip EI_min, 0 bar (parameter-free) | [913, 1309] N·m² (gradients 1014 and 1190 ± 10%) | 1007 N·m² (−0.7% / −15.4%) | PASS |
| — | M | Full-slip EI_min, 300 bar | [653, 986] N·m² (725 and 896 ± 10%) | 1007 N·m² | FAIL (EI_min has no pressure or temperature dependence; Witz attributes the measured fall to heating during pressurisation, which was not recorded) |
| WITZ-ID, identification | M | Loop width at κ = 0, 0 bar | identified | p₀ = 1.071 MPa at μ = 0.10 (bounded chain; 1.037 MPa with the former capstan bound); 552.7 N·m | IDENTIFIED |
| — | M | 0-bar loop energy (withheld, correlated with the width) | ±15% of 1118 J/m | 1114 J/m (−0.3%) | PASS |
| — | M | 0-bar mean peak moment | ±10% of 1482 N·m | 1301 N·m (−12.2%) | FAIL |
| WITZ-ID | M | 300-bar loop width at κ = 0, predicted | [1692, 2489] N·m (measured [1990, 2164] ± 15%) | 1356 N·m (−31.8% / −37.3%; former bound 1383) | FAIL |
| WITZ-ID | M | 300-bar loop energy, predicted | ±15% of 3125 J/m | 2682 J/m (−14.2%; former bound 2732, −12.6%) | PASS |
| — | M | 300-bar mean peak moment, predicted | ±10% of 2043 N·m | 1712 N·m (−16.2%) | FAIL |
| — | M | 300-bar reversal secants; branch-shape RMS | [11538, 17566] N·m²; ≤ 10% of peak-to-peak | 14865 / 14571 N·m²; 6.6% | PASS |

The WITZ-ID gate requires both the width and the energy; it fails on the
width. The peak, secant and branch-shape rows above and the studies below were
computed with the former capstan bound and its identified 1.037 MPa.

The ISSC model-to-model rows (Witz Tables 3–7) are not tabulated here:
the fixed-radius section matrix reproduces the simple-model
rows within 0.5% and lies above the participants' axial and torsional means,
which allowed radial compliance.

**Mechanisms for the 300-bar width, one at a time.** Each was registered with
its setting before it was run, re-identifies its own clamping value on the
0-bar width (μ = 0.10) and predicts the 300-bar loop without refitting.

| Mechanism | 300-bar width [N·m] | Change | 300-bar energy [J/m] |
| --- | --- | --- | --- |
| Baseline: bounded chain, end-cap capstan only | 1383 | — | 2732 |
| (a) Solved radial route, bore pressure on the pressure sheath (`pressure_barrier`), wall tension and 30 MPa | 1485 | +7.4% | 2926 |
| (a) of which internal pressure through the Zeta (hoop route) | +77 | +5.6% | — |
| (a) of which solved instead of bounded capstan | +25 | +1.8% | — |
| (a) Zeta hoop stiffness × 0.5 or × 2 | 1478 / 1489 | ±0.5% | — |
| (b) Declared clamping: pressure jump or winding stress on both armours (bounded); pressure jump as residual wire stress (solved) | 1383 / 1383 / 1486 | 0.0% against its route | 2721 / 2721 / 2917 |
| (c) Distributed elastic–Coulomb law instead of the direction-tied cap | 1380 | −0.2% | 2713 |
| (d) Interface shear compliance ξ × 0.25 to ξ = 1; k_f ∝ (P1+P2)^(1/3) | 1383 | 0.0% | 2530–2790 |
| (e) Carcass and Zeta friction switched off | 1383 | 0.0% | 2733 |
| (e) Sheath sliding under axially locked Zeta turns (hand upper estimate; zero with interlock slack) | ≤ +93 | ≤ +6.7% | ≤ +185 |
| (f) Polymer loss factor 0.02–0.10 (assumed, pressure independent) | 1382–1383 | 0.0% | 2684–2723 |
| (f) Pressure-dependent μ: ratio μ(300 bar)/μ(0 bar) needed for the measured width | 1.48 (bounded), 1.38 (solved) | — | — |
| (g) Inputs: every YAML dimension, the lay directions (they enter only through \|sin α\|) and the Fig. 6 marker order rechecked | no error found | — | — |

No single mechanism closes the width; the best-founded one, (a), moves it by
+7.4%, to −25% / −31%. The loop law is the larger gap. The model loop has a
flat width (1485 N·m in route a), whereas the measured 300-bar width is
1419–1837 N·m at |κ| = 0.3–0.6 1/m and 1990–2164 N·m at κ = 0, where two
unloading markers at the same curvature differ by 174 N·m. The loop-mean width
(energy over curvature span) is 1544 N·m measured, against 1350 N·m (bounded,
−12.6%) and 1446 N·m (solved, −6.4%). The measured 300-bar loop has an
energy-to-width ratio of 1.57 1/m and the model 1.97: the measured reversal is
more gradual than the slip-front law gives (gap G8, symmetric Masing loop).
Kraincanic and Kebadze (2001, Figs 9–10) plot a model of this pipe with
μ = 0.1 and a 10-bar initial pressure differential close to both loops
(model-to-model; their values are not tabulated). Gap G8 (next section but
one) tested the gradual-transition mechanisms: none changes the width or the
ratio by more than 0.5%.

**Identification on one loop, prediction of the other.**

- At 0 bar, once the width is matched, the loop is the same for every μ from
  0.03 to 0.40: the branch-shape RMS (87.67 N·m) changes by 0.008 N·m against
  a 10 N·m reading resolution. One loop identifies μ·p₀ (0.094–0.106 MPa), not
  μ and p₀ separately. Over that μ range the predicted 300-bar width is
  796–4231 N·m: the prediction is set by the assumed friction coefficient, not
  by the data.
- Identified on the 300-bar width (μ = 0.10), the 0-bar width is predicted
  +126% (bounded) and +107% (solved): FAIL.
- Identified on both widths, μ = 0.179 (bounded; 0.169–0.188 over the 300-bar
  reading interval) or 0.160 (solved; 0.152–0.169). The withheld 300-bar energy
  is then +29% and the reversal secants fail; the peaks pass. No (μ, p₀) pair
  reproduces both the 300-bar width and the energy with this law.
- At μ = 0.10 the measured 300-bar width needs a further 2.2 MPa (solved) to
  2.6 MPa (bounded) of P1+P2 on both armours beyond the modelled transfer.

**Remaining gaps.** A pressure- and temperature-dependent EI_min and the
pressurisation-heating route (polymer expansion raising the armour contact
pressure, the mechanism of Chu et al. 2026) need PA11/PA12 E(T), expansion
coefficients and the test temperature, none of which is reported (gap G7). The
solved route needs a thickness-wire variant (EI_min 794 N·m²) when the
area-equivalent round wires are used, because they overlap the tapes; on the
true strips (gap G1) it gives the same width (−25 / −31 %). Interlocked
carcass and Zeta layer types (G2) and orthotropic tapes (G3) are open; the
loop-law question (G8) is answered in the next section.

## Gradual stick-to-slip transition (gap G8)

Targets registered before any G8 run; nothing fitted. Code
`cablexsec/mechanics/progressive_friction.py` (`friction_model="progressive"`
in `compute_mkappa` and `compute_fatigue`) and the option
`cap_shape="papailiou"` of `strand_slip_strain` and
`conductor_bending_moment`; tests `tests/test_progressive_friction.py`;
script `python validation/scripts/check_g8_gradual_reversal.py`; guide
[local_analysis_guide.md](local_analysis_guide.md) Sections 9 and 10.

**Finding.** In every armour case the default law finishes its
stick-to-slip transition within a few per cent of the stroke: the armour
slip-onset curvature is 0.0025–0.003 1/m for the Ménard cable (amplitudes
0.05–0.2 1/m) and 0.012–0.022 1/m for the Witz pipe at 300 bar (amplitude
1.0 1/m), and the default law reaches full slip at about π/2 times it. The
measured transitions last much longer: the Ménard reversal stiffness matches
the model over the first 0.0025 1/m (50 000–68 000 against 56 000 N·m²), but
the measured moment change at 0.01 1/m from reversal is 224 and 273 N·m in
ranges (a) and (b) (318 and 353 N·m in (e) and (f)) against 377–417 N·m,
and approaches the model only beyond 0.02–0.05 1/m; the Witz 300-bar loop keeps a second, slow hysteretic part
over about 1 1/m. A transition about ten times wider than the armour onset
is not produced by any friction-law mechanism at its physical magnitude.

Mechanisms, each at its own computed magnitude:

| Mechanism | How the magnitude is fixed | Witz 300 bar: width, energy, E/W | Ménard (b) work; (a) work | Hu loop area (three-point) |
| --- | --- | --- | --- | --- |
| Current law | — | 1356 N·m, 2682 J/m, 1.977 (former bound 1383, 2732, 1.976) | +8.8%; +47.2% (former +12.9%; +52.3%) | +64.6% |
| (i) Slip-onset spread over azimuth and layer (Papailiou cap; two armour layers) | in the current law for armour | — | — | — |
| (i) Same for the conductor: Papailiou cap instead of the Karlsen-Husøy amplitude | same quarter-pitch capacity | n/a | n/a | +36.9% (uniform curvature +4.9%) |
| (i) Friction continuity across the slip boundary (progressive law, continuity density) | parameter-free | 1357, 2671, 1.969 (former 1383, 2720, 1.967) | +8.2%; +45.2% (former +12.1%; +49.9%) | n/a |
| (ii) Shear lag along the helix (distributed law) | package interface stiffness | 1380, 2713 (phase 2.3; former bound) | +12.0%; +49.9% (former bound) | n/a |
| (iii) Iwan density of contact pressure or friction (uniform, spread h) | no published density; h = 0.5 and 0.9 as sensitivity | 1383, 2726 / 2714, 1.972 / 1.963 (former bound) | +12.5% (h 0.5; former bound) | ±0.1 point |
| (iii) Three-core contact shape (package helper) | ring on Winkler foundation | n/a | not available: the Ménard section has an inner sheath and no filler | n/a |
| (iv) Interface compliance: ξ × 0.25 to ξ = 1 (Witz phase 2.3); Mindlin trellis compliance (conductor, ξ 0.54–0.56 for rings 2–4) | Yun ξ; Hertz–Mindlin crossings | 0.0% width | — | −2.4 points |
| (v) Test arrangement: curvature varying along the specimen (added) | Hu: three-point bend, span 440 mm, κ = 12 w/L²; Ménard: four-point, uniform moment; Witz: not reported | n/a | none | transition share 1.7% → 35.6% (measured 33.8%) |

Gates (registered):

| ID | Quantity | Target | Before | After (progressive law; conductor with Papailiou cap) | Status |
| --- | --- | --- | --- | --- | --- |
| G8-W1 | Witz 300-bar width, predicted | [1692, 2489] N·m | 1356 | 1357 | FAIL |
| G8-W2 | Witz 300-bar energy, predicted | 3125 J/m ± 15% | 2682 (−14.2%) | 2671 (−14.5%) | PASS |
| G8-W3 | Witz 300-bar energy/width | [1.41, 1.73] 1/m | 1.977 | 1.969 | FAIL |
| G8-W4 | Witz secants (W10), 0-bar energy (W4) | unchanged bands | PASS | PASS (14570/14299 N·m²; 1113 J/m; former bound) | PASS |
| G8-H1 | Hu loop area, three-point reduction | 3.185 N·m/m ± 15% | 5.241 (+64.6%) | 4.359 (+36.9%) | FAIL |
| G8-H2 | Hu transition share, three-point reduction | 34% ± 10 points | 35.6% | 32.1% | PASS (from the test arrangement) |
| G8-M | Ménard BM1-loop, range (b) | RMSE < 5%, work within 10%; not worse | 2.71%, +8.8% | 2.52%, +8.2% | PASS (former bound: 3.16%, +12.9% and 2.94%, +12.1%, FAIL) |
| G8-L1–L3 | zero spread bitwise; dissipation ≥ 0; rotation equivariance | exact; ≥ 0; < 1e-12 | — | bitwise; ≥ 0; 9e-15 | PASS |

**BM1-loop.** With the source's calibration transferred (μ 0.2, 0.3 MPa),
the model loop is a near-parallelogram whose friction moment is mobilised
within about 0.012 1/m of each reversal, while the measured moment is
mobilised over 0.02–0.05 1/m: the work excess falls with amplitude (+47.2% at
±0.05 1/m, +8.8% at ±0.10, +2.2% and −3.6% at 0–0.2 1/m). With the former
symmetric capstan bound the row failed (+12.9 %) and no mechanism tested
closed it; the signed bound brings it inside the 10 % target, and the
progressive law removes a further 0.6 points. The small-amplitude ranges (a),
(c) and (d) keep their large excess.

**Witz.** Diagnosis only (two numbers solved from two targets, not adopted):
one added Jenkins population with stick stiffness 623 N·m² and capacity
316 N·m, onset 0.51 1/m (23 times the armour onset), on top of the current
loop gives width 2012 N·m, energy 3371 J/m (+7.8%), ratio 1.68, secants
15 488/15 195 N·m², shape RMS 2.6% of peak-to-peak and the lens-shaped width
profile (1638–2012 N·m against 1419–2164 measured). Its stiffness is close to
the bending stiffness of the polymer layers (654 N·m²). Candidate sources
are interfaces with polymer-controlled stick stiffness that are not modelled
(the interlocked carcass and Zeta, G2; orthotropic tape shear, G3; pressure
sheath against the Zeta); none is quantified from published data.

**Hu.** The measured gradual transition (33.8% of the stroke) is the
three-point test arrangement: a uniform-curvature section of the same law
shows 1.7%. With the arrangement reproduced, the remaining discrepancy is
the loop area (+37% with the Papailiou cap) at the right maximum moment
(8.11 against 8.18 N·m): the section's friction work, not its transition,
is too large. The earlier +28% (uniform curvature) compared different
quantities.

**Asymmetric and offset loops.** The progressive law is a parallel-Jenkins
population, so loops of offset or one-sided cycles (Ménard ranges (c)–(f))
and nested irregular loops follow from the same state without extra rules;
the measured 0-bar Witz asymmetry (peaks 1427 and −1537 N·m) is not
reproduced by a rate-independent symmetric law and is left to the
non-symmetric mechanisms of Lukassen (2019; transverse migration towards the
geodesic) and of tension-coupled re-sticking (Kim 2020).
Interlocked carcass and Zeta layer types (G2), orthotropic tapes (G3)
and an asymmetric, non-Masing loop law (G8) are open.

## Sævik (2011) FBG armour stresses (SV11-PLANE, SV11-GJ)

Gap 2.5. The 8-inch non-bonded riser of Sævik (2011,
Table 1; the same table in Gjøsteen et al. 2027, Table 7) is packaged as
`saevik2011.yaml` (`cx.saevik2011_yaml_path()`; assumptions A1-A7 in its
header). Fibre Bragg gratings sat in grooves on both sides of four tendons of
the inner tensile armour, five stations per tendon at 45° intervals over half
a pitch, 20 stations from 573 mm before to about 2.9 m after the start of the
curved bellmouth. Sævik plots, for 14 load cases (250-750 kN, 17-34 MPa,
end-angle ranges 0.5-32°), the measured stress range at every station
normalised by the static stress (Figs 17-23); Gjøsteen et al. (2027, Figs
27-35) plot the absolute LC 14 loops (±8°) at nine stations. Both are
digitised from 300 dpi renders (`validation/paper_data/saevik2011_figs17_23.csv`,
`gjosteen2027_lc14_ranges.csv`; reading resolution 0.009-0.073 normalised
and ±2 MPa; source SHA-256 in the headers). The digitised model curves
reproduce Sævik's Table 3 C/M (MM column) within 0.05 for 12 of 14 load
cases. Harness `validation/scripts/check_saevik2011.py`; tests
`tests/test_saevik2011_benchmark.py`. Targets were registered before any
model output.

**Mapping.** Tendons are area-equivalent round wires at the true mean radius
(gap G1), so the solved radial route rejects the overlapping round wires and
the bounded chain is used; the pressure armour is a near-hoop helix with the
tabulated 10.5 mm²/mm (gap G2); the carcass is omitted, so the internal
pressure acts on the seal-layer bore, which is the barrier a
`pressure_barrier` declaration would name. Not reported and assumed: outer
armour radius and angle (same angle, opposite hand), lay hands, tape and
sheath thicknesses, steel modulus 207 GPa. One polymer modulus (455.8 MPa) is
identified so that EI_min equals the tabulated 72.8 kN m², a section property
independent of the compared stresses. μ = 0.20 is declared from Dai, Sævik
and Ye (Marine Structures 55 (2017) 137–161, Sec. 4.3.1, p. 156), who quote it
from Sævik (1992) as an estimate between anti-wear tape and inner armour and
between two steel armour layers "at 20 MPa internal pressure"; the contact
pressure at that internal pressure is 2–4 MPa, and the measured dynamic
coefficient of armour wire on in-service tape is 0.32, 0.20 and 0.18 at 1, 4
and 8 MPa (Dai, Sævik and Ye 2018, Sec. 4.1, p. 428). It is not fitted.

**Prediction chain.** Section state at the wall tension T + p π r_b²
(closed ends) with the bore pressure; static stress = inner-armour wire
stress of that state (271 MPa at 500 kN and 34 MPa). The curvature along the
pipe is not published: a tensioned beam (effective tension T) with the
CableXSec first-loading M-κ backbone, clamped at the end-fitting face
2.305 m before the curved section (rig lengths of Gjøsteen et al.; the
rocker rotation point is assumed at the start of the curved section), inside
the 5 mm bellmouth gap and the curved profile of Sævik's Eq. 88 to 1.72 m,
with the far pin 15.5 m away. The symmetric cycle ±A/2 gives each station
the range 2κ_peak. The bound-constrained Newton solve converges to a
relative projected gradient below 1e-9; halving the 20 mm mesh changes the
station curvatures by at most 5% (11% at one station of the 32° case).
Station stress: the fatigue-path wire stress field (direction-tied friction)
over two cycles, range of the last, (a) sectional and (b) with
`TerminationSpec` (anchor at the end-fitting face, the peak curvature profile
as shape).

| ID | Class | Quantity | Target | Result | Status |
| --- | --- | --- | --- | --- | --- |
| SV11-PLANE | M | Per load case, R = mean predicted / mean measured normalised axial range at the 8 bending-plane stations (x.1 and x.5), termination-corrected | median R in [0.85, 1.18] and ≥ 12 of 14 R in [0.70, 1.43] | median 1.62 (0.45-3.59), 3 of 14 (former capstan bound 1.65) | FAIL |
| — | M | Same, sectional (reported beside the gate) | as above | median 1.00 (0.45-2.17), 11 of 14 (former 1.03) | reported |
| SV11-GJ | M | LC 14, ±8°: mean predicted / mean measured absolute range at 2.1, 2.5, 3.1, 3.5, termination-corrected | [0.72, 1.38] | 1.80 (455, 249, 282, 88 against 186, 146, 187, 78 MPa; former 1.90) | FAIL |
| — | M | Same, sectional | as above | 1.24 (former 1.32) | reported |
| — | M | Static stress at LC 14 against the value implied by Gjøsteen absolute / Sævik normalised ranges (281, 238, 341, 278 MPa) | none | 271 against 285 MPa (−5%) | INFO |
| — | M | μ identified on LC 14 (termination R = 1), other cases predicted | as SV11-PLANE | μ = 0.091; median 1.16, 8 of 14 (former capstan bound; not re-run) | reported (identified) |

The band of the gate follows from the data: the coincident bending-plane
pairs 1.5/2.1, 2.5/3.1 and 3.5/4.1 should read the same under a symmetric
cycle, and they differ by an RMS of 0.45 in ln (one sensor ≈ 0.32), so an
8-station mean scatters by about 0.11 in ln.

Per load case (R termination / sectional): 0.5° 0.45/0.45; 1° 0.54/0.54;
1.5° 0.73/0.73; 2° 0.98/0.98; 4° (250 kN, 17 MPa) 3.59/2.17; 4° 1.57/1.26;
7.5° 2.15/1.09; 8° 2.25/1.11; 15° 1.70/1.03 (500 kN), 1.77/1.11 (250 kN),
1.38/0.87 (750 kN), 2.38/1.32 (17 MPa); 24° 1.66/0.84; 32° 1.56/0.74.
Station-wise RMS of ln(pred/meas): 0.72 (termination), 0.49 (sectional),
against 0.32 for one sensor. The signed capstan bound lowers the slipping
cases by 3–5 % (sectional LC 4 2.28 to 2.17, LC 14 1.09 to 1.03, LC 19 1.39
to 1.32) and leaves the stick cases (0.5–2°) unchanged.

**What matches and what does not.**

- *Large angles, sectional.* From 4° up the sectional prediction at the
  bending-plane stations is 0.74-1.32 of the measurement (LC 4 aside), and
  the mean absolute LC 14 range is 1.24 times the measured one; at 15° and
  250, 500 and 750 kN, R is 1.11, 1.03 and 0.87. The section state also reproduces the
  normalising static stress within 5%. This holds at a declared μ; with the
  full-slip friction stress proportional to μ (P1+P2), it tests μ (P1+P2),
  not μ alone.
- *Small angles.* At 0.5-1.5° both variants under-predict by a factor of
  1.4-2.2, independent of μ: the beam model leaves the pipe in stick
  (EI_max 5.2 MN m²) with a broad curvature and at most a point contact near
  the start of the curved section, and the measured ranges at the smallest
  angle need about twice the predicted station curvature (2.2 at 0.5°, 1.9
  at 1.0°). The gap (5 mm) and the
  bellmouth profile (Eq. 88) are published and used; the pipe curvature at
  the stations, the rotation point and the stick stiffness are not. Sævik's
  own moment model, with an FE curvature in the bellmouth, reaches 0.76-1.09
  here, so the deficit lies in the beam curvature.
- *Termination correction.* With the bellmouth curvature profile the
  anchored-wire chain accumulates friction over a boundary layer of 1.9 m
  and roughly doubles the ranges in and next to the bellmouth (LC 14: 2.0
  against 0.66 measured at 2.1). The correction's stated limits (rigid stick,
  proportional loading, linearised cap, axial slip only) are violated by a
  contact profile that grows with the angle; the measurements do not support
  it in this form, and the source's own anchored model (SBM) stays near the
  data. The correction is unchanged; the result is reported.
- *Lateral and normal tendon bending.* The clamp correction is axial only
  and cannot predict lateral (strong-axis) or normal tendon bending. The
  sectional loxodromic lateral bending at the 6.25 mm tendon edge (round-wire
  tangential component × 12.5/8.92) over-predicts the one-side LC 14 range at
  the neutral-axis station 2.3 by 2.7 (402 against 149 MPa); averaged over
  the normalised neutral-axis stations at 4° and above it is close (0.36
  against 0.37), with large station scatter. The measured side is not
  stated, and at 45° stations the prediction depends on the unknown helix
  sense by up to a factor of 2.
- *Inconsistencies in the source.* LC 4 is 34 MPa in Table 2 but 17 MPa in
  Table 3 and the Fig. 19(b) caption (17 MPa used; R 2.28 sectional); LC 14
  is a 15° range in Table 2 and ±8° in Gjøsteen et al.; the bellmouth ends at
  1.72 m (Sævik) or 1.79 m (Gjøsteen), with a 5 mm or 10 mm gap.

**Missing.** Friction coefficient; outer-armour geometry, lay hands, tape,
sheath and pressure-armour dimensions; material properties; the normalising
static stress and whether the plotted value is axial or one side; station
coordinates (figures only); the curvature distribution along the pipe; the
end-fitting position relative to the bellmouth (taken from the rig lengths);
Sævik and Igland (2002) and Andersen et al. (2001) are not held.

## Measured wire-on-tape friction loops (DAI18-DIG, DAI18-RS, DAI18-T1, DAI18-T3)

Source: Dai, Sævik and Ye, "An anisotropic friction model in non-bonded
flexible risers", Marine Structures 59 (2018) 423–443, doi
10.1016/j.marstruc.2018.02.012, with the corrigendum in Marine Structures 63
(2019) 404, doi 10.1016/j.marstruc.2018.11.001. Small-scale rig (Sec. 2.2,
pp. 424–425, Fig. 1): one tensile armour wire between two anti-wear tapes cut from
a flexible pipe, normal load by weights, displacement-controlled cyclic motion
of 2 mm amplitude at 0.01–0.2 Hz (Sec. 2.3, p. 425). The y axis of Fig. 4 is
the friction force over twice the contact force, the two interfaces of the
wire (p. 428). Test No. 1 (Sec. 4.1, p. 428): in-service tape outer surface,
8, 4 and 1 MPa, 0.01 Hz; printed coefficients μ_s 0.22, 0.24, 0.38 and μ_d
0.18, 0.20, 0.32, selected as "the maximum (in the stick domain) and minimum
values in Fig. 4a". Test No. 3 (Sec. 4.3, p. 430): groove track of the tape
inner surface, 4 MPa, 0.01, 0.05, 0.1 and 0.2 Hz; no coefficients printed.
Test No. 2 (clean tape on a tailor-made steel bar) is not used: it is not a
wire on in-service tape, and its contact area and normal force are not
printed.

**Scope.** One interface on one spring-slider (`slider_history`), not a cable
section. The comparison tests the form of the static/dynamic `FrictionLaw`
(re-sticking, breakaway peak, dynamic level, rate independence); it does not
test the armour bound of a section, its capstan term or its contact pressure.

**Digitisation.** `validation/scripts/digitise_dai2018_fig4.py` reads the
raster embedded in PDF page 7 at its native resolution (JPEG, 1887 × 2063 px,
300 ppi; extracted without resampling). Axes are calibrated by least squares
on the tick-mark centroids (largest residual 0.0024 mm and 0.0010 in panel
(a), 0.0063 mm and 0.0005 in panel (c)); curve pixels are classified by hue
with the legend handles masked; the branches take the median of each pixel
column, the elastic segments the median x of each pixel row in the middle half
between the branch levels. Reading resolution (half-width; not experimental
uncertainty): ±0.008 mm and ±0.0021 in (a), ±0.012 mm and ±0.0012 in (c).
Overlapping curves of panel (c) hide one another (the 0.05 Hz loop most);
hidden parts are missing, not interpolated. The data are
`validation/paper_data/dai2018_fig4a_test1.csv` and `dai2018_fig4c_test3.csv`.
The rises of the friction force in the last part of each stroke, which the
paper attributes to the wire's out-of-straightness (p. 428) and to surface
irregularities (p. 430), are kept in the data and excluded from every
measure: the last 0.5 mm before each reversal.

**Model.** `slider_history` with the normalised bound `b = μ_d`,
`FrictionLaw(static_factor = μ_s/μ_d, stick_time_s = 0)`, no rate terms, the
stick stiffness of each loop's elastic segments (mean of its two reversals) and
a sinusoidal drive between the loop's measured reversal displacements; third
cycle from a stress-free start. Test 1 takes the printed μ_s and μ_d. Test 3
takes the digitised peak and dynamic level of the 0.01 Hz loop (0.2319 and
0.1797, mean of both strokes) and holds them at 0.05–0.2 Hz. Measures, the
same for model and measurement: the peak (largest |y| within 0.6 mm of travel
after a reversal), the dynamic level (smallest 0.1 mm running mean after the
peak, the paper's "minimum"), the decay length (travel from the peak to within
10 % of the peak-to-level drop) and the stroke work (∫ y dx from the start of
the digitised elastic segment to the excluded zone). Script
`validation/scripts/check_dai2018_friction.py`; tests
`tests/test_dai2018_friction.py`. Gated rows use criteria fixed by the printed
precision and by the presence of a peak, not by the size of the model errors;
the model errors are stated, not gated (INFO).

| ID | Class | Quantity | Target | Result | Status |
| --- | --- | --- | --- | --- | --- |
| DAI18-DIG | M | Digitised loop maximum, Test 1, vs printed μ_s 0.22/0.24/0.38 | within 0.005 (printed precision) + 0.0021 (reading) | 0.2237/0.2405/0.3786 (largest deviation 0.0037) | PASS |
| DAI18-RS | M | Peak above the dynamic level at every reversal, 7 loops; slider with `stick_time_s = 0` vs the default `inf` | > 5 reading resolutions on all 14 measured strokes; `0` on all 14; `inf` on none | measured 14/14 (excess 0.039–0.094); `0` 14/14; `inf` 0/14 | PASS |
| DAI18-T1 | M | Test 1 vs slider with the printed coefficients: peak, level (+x/−x stroke), work | stated, not gated | see the table below | INFO |
| DAI18-T3 | M | Test 3 vs the rate-independent slider with the 0.01 Hz parameters | stated, not gated | level −3.5/−2.4/+0.3 % at 0.05/0.1/0.2 Hz (model 0); peak +10.6 % at 0.2 Hz (model 0) | INFO |

| Loop | Model μ_s / μ_d | k (F/2N per mm) | Peak, measured +x/−x | Peak error | Level, measured +x/−x | Level error | Work error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Test 1, 8 MPa | 0.22 / 0.18 | 3.28 | 0.2237 / 0.2213 | −1.8 / −0.8 % | 0.1797 / 0.1718 | +0.1 / +4.7 % | −2.6 % |
| Test 1, 4 MPa | 0.24 / 0.20 | 5.50 | 0.2405 / 0.2381 | −0.4 / +0.6 % | 0.2020 / 0.1877 | −1.0 / +6.6 % | −1.9 % |
| Test 1, 1 MPa | 0.38 / 0.32 | 15.70 | 0.3756 / 0.3786 | +1.0 / +0.2 % | 0.3308 / 0.2848 | −3.3 / +12.3 % | −2.1 % |
| Test 3, 0.01 Hz | 0.2319 / 0.1797 | 3.08 | 0.2290 / 0.2348 | +1.2 / −1.3 % | 0.1807 / 0.1787 | −0.6 / +0.6 % | −5.2 % |
| Test 3, 0.05 Hz | as 0.01 Hz | 3.51 | 0.2323 / 0.2427 | −0.2 / −4.5 % | 0.1743 / 0.1725 | +3.1 / +4.2 % | −0.9 % |
| Test 3, 0.1 Hz | as 0.01 Hz | 3.46 | 0.2402 / 0.2529 | −3.5 / −8.3 % | 0.1767 / 0.1742 | +1.7 / +3.2 % | −3.6 % |
| Test 3, 0.2 Hz | as 0.01 Hz | 3.50 | 0.2515 / 0.2615 | −7.9 / −11.4 % | 0.1819 / 0.1785 | −1.2 / +0.7 % | −5.4 % |

Errors are model over measurement minus one; the work is that of both strokes
over the compared window.

**Findings.**

- *Re-sticking.* Every reversal of all seven loops shows a breakaway peak
  0.04–0.09 above the dynamic level, at 0.01 Hz (a 50 s cycle) as at 0.2 Hz
  (5 s). The law reproduces it with `stick_time_s = 0` (the static factor
  returns after one stick step); the default `inf` removes every peak after
  the first breakaway. For armour wire on anti-wear tape at these frequencies
  `stick_time_s = 0` is the evidence-based setting. The measured static to
  dynamic ratio is 1.19–1.22 (printed coefficients).
- *Dynamic level.* The +x stroke reaches the printed μ_d at 8 and 4 MPa
  (0.180 and 0.202); at 1 MPa its minimum is 0.331 against 0.32. The −x stroke
  runs 4–14 % lower (0.172, 0.188, 0.285), an asymmetry the paper does not
  discuss; one μ_d per loop cannot follow it, so the level errors of the −x
  stroke are +4.7 to +12.3 %.
- *Breakaway (model-form limit).* The measured peak lies 1.4–2.5 times as far
  from the reversal as the slider's, `(μ_s + μ_d) / k` after the reversal: the
  elastic rise rounds off before the peak. After the peak the measured force
  falls to within 10 % of the drop over 0.6–1.2 mm of travel on the +x strokes
  and 1.2–2.3 mm on the −x strokes, where the force keeps falling over most of
  the stroke; the slider drops in one step. The law has no transition
  distance. Dai et al. derived a transition distance of 1.96 times the elastic
  displacement from the clean tape and steel bar of Test 2 (p. 430); the
  in-service surfaces of Tests 1 and 3 decay over a longer distance, which is
  not separable here from the variation of the force along the stroke. The
  missing decay leaves the model's work 1.9–2.6 % (Test 1) and 0.9–5.4 %
  (Test 3) below the measured work over the compared window.
- *Frequency (model-form limit).* The static peak rises by 10.6 % from 0.01 to
  0.2 Hz (mean of both strokes; +9.8 % on +x, +11.4 % on −x). The paper states
  8 % and attributes it to the misalignment of the irregular wire and tape
  surfaces, amplified at the higher frequency (p. 430). The law's rate factor
  multiplies the
  dynamic factor only and cannot raise the static peak: the rate-independent
  model under-predicts the 0.2 Hz peak by 8–11 %. The dynamic level does not
  move with frequency (−3.5, −2.4 and +0.3 % at 0.05, 0.1 and 0.2 Hz, not
  monotonic) over peak sliding speeds `2 π f A` of 0.13–2.5 mm/s: no rate term
  on the dynamic coefficient (`rate_coefficient = 0`, no viscous term) is
  supported in this range.
- *Contact pressure.* The printed coefficients fall with pressure: μ_d 0.32,
  0.20, 0.18 and μ_s 0.38, 0.24, 0.22 at 1, 4 and 8 MPa, a factor 1.8 on μ_d
  from 1 to 8 MPa, larger than the static to dynamic ratio. The
  `pressure_factor` of `FrictionLaw` represents it (concepts, "The friction
  coefficient can depend on slip state and rate"). A factor table built from
  these three points reproduces their levels by construction; that is a
  parameter fit, not a validation.
- *Stick stiffness.* The slopes of the elastic segments (3.1–3.5 per mm in
  Test 3; 3.3, 5.5 and 15.7 per mm at 8, 4 and 1 MPa in Test 1, in F/(2N) per
  mm) are of the rig and both tapes in series and are inputs of the model. The
  1 MPa slope rests on a near-vertical segment a few pixel columns wide. The
  tape shear modulus of Test 2 (G = 352 MPa, p. 430 as corrected by the
  corrigendum) is not used: the contact area and normal force of the
  specimens are not printed.

**Missing.** Wire width and contact length, normal force, tape material,
temperature, the plotted cycle, repeat tests (one specimen per condition).

## Test arrangement (gap T1)

Code `cablexsec/analysis/beam_test.py`
(`simulate_bending_test`, `ThreePoint`, `FourPoint`, `Cantilever`,
`BendingProtocol`, `SectionLaw`); tests `tests/test_beam_test.py`; script
`python validation/scripts/check_t1_test_arrangement.py`; guide
[local_analysis_guide.md](local_analysis_guide.md) Section 6.2. Targets
registered before the benchmark runs (the simulator had been run once on Hu
as a code check, reproducing the G8 result already on record). Nothing
fitted.

Gap G8 found that the gradual transition of the Hu et al. (2022) loop comes
from the three-point test, in which the curvature varies along the span.
T1 makes the arrangement a general comparison tool: the specimen is
discretised into stations, each with the section's history-dependent law;
the set-ups are statically determinate, so the moment is the load times a
known influence line and every station's law is inverted from its own state
at every load step; the outputs are what the test reports (load,
displacement, the test's moment formula and its curvature definition).

**Verification.**

| ID | Class | Quantity | Target | Result | Status |
| --- | --- | --- | --- | --- | --- |
| T1-V1 | A | Elastic closed forms (three-point `FL³/48EI`, four-point load point `(F/2)a²(3L − 4a)/6EI`, cantilever `FL³/3EI`) under load, displacement, moment and curvature control | < 1e-12 rel. | ≤ 3e-16 | PASS |
| T1-V2 | I | Four-point, curvature control: every station of the uniform-moment zone against `compute_mkappa` over the same history | bitwise | bitwise (moment and curvature); LVDT parabola reads the curvature to 4e-16 1/m | PASS |
| T1-V3 | I | External work against stored plus friction work over the span; closed-cycle `∮F dδ` against span friction work | falls with refinement; < 1% at the default mesh | 5.0–5.2% (6 stations per segment), 0.53–0.60% (24), for all friction models | PASS |
| T1-V4 | I | Equilibrium residual of the final full-history run (leg property): direction-tied, progressive, laid-core, distributed, conductor caps | < 1e-8 rel. | 2e-14 to 1e-11 | PASS |

**Benchmarks in their published arrangements.**

| Benchmark | Arrangement (source) | Quantity | Measured | Uniform curvature (before) | Arrangement (after) | Gate |
| --- | --- | --- | --- | --- | --- | --- |
| Hu et al. (2022), Karlsen-Husøy amplitude | three-point, hinged, 440 mm, ±6 mm, third cycle; M = FL/4, κ = 12w/L² (Sec. 2.1, Eqs. 1–2) | loop area [N·m/m] | 3.185 | 4.084 (+28.2%) | 5.236 (+64.4%) | FAIL (±15%) |
| same | same | transition share of the TC–BC stroke | 33.8% | 1.7% | 39.4% (24 stations per half span), 35.7% (48), 37.0% (96) | PASS (34 ± 10 points) |
| same | same | maximum moment [N·m] | 8.18 (Nam et al. 2024) | 7.58 (−7.3%) | 8.83 (+7.9%) | reported |
| Hu, Papailiou cap | same | loop area | 3.185 | 3.340 (+4.9%) | 4.354 (+36.7%) | FAIL |
| same | same | transition share | 33.8% | 1.7% | 32.8% (24, 48 and 96 stations) | PASS |
| same | same | maximum moment | 8.18 | 7.07 (−13.6%) | 8.11 (−0.9%) | reported |
| Ménard and Cartraud (2023), ranges (a)–(f), transferred inputs | four-point, 2.5 m bench, actuators at the quarter points, ends pivot and slide; M = Fa L1; curvature from a parabola through three LVDTs between the actuators (Sec. 3) | ordered RMSE; work error | — | 10.55/2.71/10.14/8.34/3.83/3.85%; +47.2/+8.8/+44.7/+32.0/+2.2/−3.6% | identical (largest moment difference 3e-8 N·m) | BM1-loop PASS (unchanged by the arrangement; FAIL with the former capstan bound) |
| Witz (1996), WITZ-ID | not documented (p. 888: "No detailed information was supplied on the test equipment or procedures") | 300-bar width; energy; energy/width | [1990, 2164] N·m; 3125 J/m; 1.57 1/m | 1356 N·m; 2682 J/m; 1.977 1/m | basis unchanged; three-point reading of both loops as a sensitivity (p₀ re-identified on the 0-bar width: 0.739 MPa): width 1589 N·m (−20.2% / −26.6%), energy 2998 J/m (−4.1%), energy/width 1.886 1/m; 0-bar energy 1072 J/m (−4.1%) (former capstan bound: 0.715 MPa, 1621 N·m, 3057 J/m) | FAIL (unchanged) |
| Zeidan (2025/2026), Komperød (2017) | — | — | — | not in the package (no published construction or not yet registered) | — | — |

The Ménard positions of the three LVDTs are not reported; quarter points of
the uniform-moment zone were used, and any positions inside the zone give
the same curvature. Nam et al. (2024) enters only through the measured
maximum moment of the Hu test. The three-point
reduction is independent of the span for an Euler–Bernoulli beam, so the
Witz sensitivity needs no assumed dimension.

**What the arrangement explains.**

- **Hu transition.** The share of the reversal stroke in the slip transition
  comes from the curvature varying along the span: 1.7% at uniform curvature,
  32.8–39.4% in the arrangement against 33.8% measured, for both cap shapes.
  The Karlsen-Husøy value moves between 35.7% and 39.4% with the station
  count (each station's rings slip at one curvature), the Papailiou value is
  converged.
- **Hu loop area: not explained, and worse than it looked.** Reduced as the
  test reduces it, the same section gives a larger loop: reaching the
  reported curvature `12 w/L²`, a weighted mean of the curvature along the
  span, needs a larger mid-span moment (8.11 against 7.07 N·m with the
  Papailiou cap). The uniform-curvature agreement of the Papailiou cap
  (+4.9%) compared a different quantity; in the test's terms the section's
  friction work is 37% too large (64% with the Karlsen-Husøy amplitude).
- **Ménard.** The measured curvature lies in the uniform-moment zone, so the
  arrangement changes nothing; BM1-loop (+8.8%) and the excess work of ranges
  (a)–(d) remain, with the transition mobilised within about 0.012 1/m of a
  reversal against 0.02–0.05 1/m measured (gap G8).
- **Witz.** The source does not document its arrangement, so the
  uniform comparison stays the basis. Read as three-point tests, both Witz loops would give a 300-bar width of 1621 N·m, still below the band from 1692 N·m, and an energy-to-width ratio of 1.886 against [1.41, 1.73]; the energy would agree (−2.2%). An undocumented arrangement could therefore account for part of the width deficit, not for all of it, and not for the ratio.

**Not represented.** Euler–Bernoulli small deflection on frictionless
supports: no shear deformation, P-delta effect of an axial load or
self-weight (Hu notes the gravity of the overhanging ends). The stations are
independent sections, so axial redistribution of wire slip between stations
of different curvature is not represented; in both documented tests the span
or uniform zone is only one to two lay lengths long (Ménard: 1.25 m against
pitches of 0.64–0.73 m, which the source flags as a possible end effect).
This curvature-gradient effect was tested in gap T2 (next section): it
does not explain the work gaps.

## Along-wire redistribution (gap T2)

Module `cablexsec/mechanics/along_wire.py`, beam coupling
`cablexsec/analysis/along_wire_test.py` (`simulate_bending_test(...,
along_wire=AlongWireSpec(...))`), termination option
`TerminationSpec(solver="along_wire", stick=..., capstan=...)`; tests
`tests/test_along_wire.py`. Targets were registered before the benchmark
runs. Nothing is fitted and no input is recalibrated.

Every wire of a helical layer is followed along its own helix through the
curvature field of the specimen, rig or termination: axial force
`n = e + du/dφ`, friction line load bounded by the package's contact
pressure (with its capstan term), imposed strain from the local curvature,
ends anchored (`u = 0`) or free (`n = 0`). It generalises the sectional law
(one station, infinite length: the distributed model is its periodic
limit) and the clamp-layer solver (one anchor, a proportionally scaled
curvature shape). The stick stiffness is explicit: `"spring"` (the
distributed model's elastic interface `k̂ = ξ/(1 − ξ)`, full imposed strain;
the basis), `"rigid"` (ξ = 1) and `"scaled"` (rigid interface with the
ξ-scaled strain, the termination correction's default). In a bending test the
beam equilibrium `M(x) = F m(x)` is solved together with all wires (Newton on
the station curvatures and the load with the exact station Jacobian of the
wires on their stick/slip set). Biaxial imposed strains are solved directly
by the chain (termination option); the beam coupling is planar.

**Verification.**

| ID | Class | Quantity | Target | Result | Status |
| --- | --- | --- | --- | --- | --- |
| T2-V1a | I | Middle pitch of a 16-pitch anchored chain under uniform cyclic curvature against the periodic distributed kernel (same mesh): spring without and with capstan; rigid (the periodic kernel at `k̂ = 1e4`) | < 1e-9 rel. (rigid: the stand-in's 1e-4 stick defect) | 6e-15 and 9e-13; rigid 6e-4 | PASS |
| T2-V1b | I | Long four-point specimen with anchored ends, reported moment against the T1 simulator with the distributed law | < 0.5%, falling with length | 0.47% (6 pitches), 0.08% (12), 0.08% (24) | PASS |
| T2-V2a | A | Rigid chain from one anchor at constant curvature against `clamp_layer.anchored_wire_response` | < 1e-12 rel. | 4e-15 | PASS |
| T2-V2b | I | Termination increment, `solver="along_wire"` against the clamp solver in its convention (`scaled`, no capstan) and with `stick="rigid"` | < 1e-12 rel. | 1e-15 to 2e-14 | PASS |
| T2-V3a | I | Chain energy identity `W = ΔU + D + R` (spring, rigid, capstan; free and anchored ends) | round-off | below 1e-12 of the work | PASS |
| T2-V3b | I | Closed-cycle external work against the wire friction work in a beam test | < 1% | 0.3–0.6% | PASS |
| T2-V4 | I | Refinement: Sævik LC 14 (substep 0.5 → 0.25 κ_c; 64 → 128 segments per pitch); Hu (32 → 64 segments per pitch, 24 → 48 stations per segment) | < 1% | below 1e-5; +0.15% | PASS |
| T2-V5 | A, I | Free end: force within the half-node bound; free-free middle pitch tends to the periodic field; Lutchansky (1969) closed form near an anchor (spring, stick) | reported | yes; 1.0e-3 at 16 pitches; within 0.2% of the amplitude | PASS |

**Sævik (2011) and Gjøsteen et al. (2027): near-bellmouth gauges.** The rig
and section chain of SV11 unchanged up to the curvature field, which is
taken from the backbone beam at every angle sample of the symmetric cycle
(the bellmouth wrap grows with the angle: non-proportional). All 55
inner-armour wires are solved between the two end fittings (anchored);
station range at the station azimuth, normalised by the static stress.

Values in this subsection and in the two-way study below were computed with
the former symmetric capstan bound (no reproduction script for them is in the
repository); the scorecard SV11 rows now read 1.62 (termination) and 1.00
(sectional).

| ID | Class | Quantity | Target | Along-wire (spring, capstan) | Before | Status |
| --- | --- | --- | --- | --- | --- | --- |
| T2-SV-PLANE | M | median R over 14 load cases; R in [0.70, 1.43] | median in [0.85, 1.18], ≥ 12 of 14 | 3.36 (0.44–6.28), 2 of 14 | termination 1.65 (3 of 14); sectional 1.03 (11 of 14) | FAIL |
| T2-SV-GJ | M | LC 14 ±8°, mean absolute range at 2.1, 2.5, 3.1, 3.5 | [0.72, 1.38] | 4.62 (1000, 760, 773, 223 against 186, 146, 187, 78 MPa) | termination 1.90; sectional 1.32 | FAIL |
| T2-SV-SMALL | M | R at 0.5, 1, 1.5° | reported | 0.44, 0.54, 0.74 | 0.45, 0.54, 0.73 | unchanged |
| T2-SV-REF | I | the same chain's sectional limit (distributed law at each station's history) | reported | median 1.03 (11 of 14); GJ 1.31 | sectional 1.03; 1.32 | reproduces SV11-SEC |

Stick stiffness and capstan (T2-SV-STICK): median R / GJ ratio
`spring` 3.36 / 4.62; `spring` without capstan 3.09 / 4.18; `rigid` 3.46 /
4.78; `scaled` 2.12 / 2.90. With `rigid` the small-angle cases read 0.85,
1.03 and 1.40: that is the ξ = 1 stick stress of the section on the package's
(ξ-reduced) beam curvature, not an along-wire effect, and a consistent ξ = 1
beam would lower the station curvature. Diagnosis (not registered): with the
peak profile scaled proportionally, as the termination correction assumes,
LC 14 gives R 1.71 / GJ 1.82 in the `scaled` convention without capstan (the
clamp-solver correction: 1.76 / 1.90) and 2.38 / 2.41 with `spring`; the
non-proportional wrap raises them to 2.03 / 2.67 and 3.55 / 4.62.

**Two-way anchored termination.** Every one-way route above
takes the curvature from a beam with the sectional backbone and adds the
anchored-wire friction afterwards, so the anchoring enters the wire stress but
not the curvature that drives it. Sævik's own anchored model (SBM, Sævik 2011
Sec. 6) condenses the tendon slip into the beam element instead. Hypothesis
H-C: solving the rig beam with the moment of the anchored wire field (two-way)
brings the anchored stresses to the measurements. Targets were registered and
hashed before any run. The two-way calculation solves the rig beam of SV11 (unchanged geometry, bounds and rig) with the along-wire chains of
both tensile armours inside every load step: moment `EI_min κ + Σ` chain moment,
projected Newton with the chain Jacobian as tangent, chain state committed at
convergence, step halving on failure, beam mesh 0.04 m. Both chains use the
`spring` stick convention without the capstan bound (the outer-armour chain
does not converge with it). A relaxed fixed point between the one-way beam and
the along-wire solver oscillates on this rig (the bellmouth contact switches
between iterates) and is not used.

| ID | Quantity | Target | Result | Status |
| --- | --- | --- | --- | --- |
| TC-1 | SV11-PLANE, two-way | median R in [0.85, 1.18], ≥ 12 of 14 in [0.70, 1.43] | median 1.18 (1.175), 9 of 14 | FAIL |
| TC-2 | SV11-GJ, two-way | [0.72, 1.38] | 1.46 (260, 222, 215, 175 against 186, 146, 187, 78 MPa) | FAIL |
| TC-3 | LC 14 curvature range two-way / one-way at 1.5, 2.1, 2.5, 3.1 | ≤ 0.8 at three of four | 0.61, 0.62, 1.47, 1.69 | FAIL |
| TC-4a | μ → 0 limit equals the one-way sectional result | < 1e-6 | anchored wires 12 % at μ = 1e-6 and 1e-4 (a frictionless wire anchored at both ends carries its net strain between the anchors); free ends 7.5e-5 | FAIL as registered |
| TC-4b | far field (nodes beyond three boundary layers) | < 1 % | 7.2, 5.6, 3.8 % at 3.8, 4.0, 4.2 m | FAIL |
| — | uncoupled mode against the SV11 beam (LC 14, 7.5°) | — | largest curvature difference 8.5e-6 1/m (peak 0.148) | check |
| — | mesh 0.04 against 0.02 m (LC 14) | — | R 1.261 against 1.259; station stresses within 0.6 % | check |

Per load case (R two-way): 0.5° 0.47; 1° 0.53; 1.5° 0.73; 2° 0.98;
4° (250 kN, 17 MPa) 2.55; 4° 1.53; 7.5° 1.35; 8° 1.38; 15° 1.26 (500 kN), 1.28
(250 kN), 1.09 (750 kN), 1.70 (17 MPa); 24° 1.06; 32° 0.93.

By the registered rule H-C is rejected (TC-3 fails), and the termination route
is unchanged: the sectional stress stays the hang-off value and the one-way
anchored correction an upper bound (the `UnvalidatedModelWarning` of
`compute_fatigue` says so). Two results are reported beside the decision. The
two-way solution does not lower the curvature near the end fitting uniformly; it
spreads the bend along the bellmouth (the peak range at 1.5 and 2.1 falls to 0.6
of the one-way value, at 2.5 and 3.1 it rises 1.5–1.7 times). And most of the
anchored over-prediction of the one-way routes is this missing consistency:
the median falls from 3.09 to 1.18 and the absolute LC 14 ratio from 4.2 to 1.46.
The remaining excess sits at the lowest contact pressures (LC 4, LC 19), and the
small-angle deficit is unchanged. A clamped end therefore cannot be corrected
locally after a global run that does not carry the anchoring; the consistent
route is the two-way coupling of the global model
([global_local_guide.md](global_local_guide.md) Section 7).

**Short specimens** (arrangements, protocols and reductions of T1; the wire
end conditions are not reported, so each row is a sensitivity over the end
conditions, none is a basis). The Ménard and Witz rows were computed with the
former capstan bound; the Hu rows (conductor operator, no capstan term) are
unchanged.

| ID | Class | Benchmark and quantity | Target | Sectional reference | Along-wire (end-condition set) | Status |
| --- | --- | --- | --- | --- | --- | --- |
| T2-HU | M | Hu et al. (2022) three-point, ±6 mm, third cycle: loop area [N·m/m] | 3.185 ± 15% | 4.354 (+36.7%, T1 Papailiou cap) | 5.25–5.26 (+64.9–65.0%) for free ends with 0, 0.11, 0.22, 0.44 m overhang and for anchored ends | FAIL (moves away) |
| T2-HU | M | same, transition share of the TC–BC stroke | 34 ± 10 points | 32.8% | 10.7–11.1% | FAIL (moves away) |
| T2-HU | M | same, maximum moment [N·m] | 8.18 (reported) | 8.11 | 8.46–8.47 | reported |
| T2-MEN | M | Ménard and Cartraud (2023) four-point, ranges (a)–(f), work error (trapezoid of M dκ between crossings) | BM1-loop (b): RMSE < 5%, work within 10% | direction-tied +51.5/+12.2/+49.3/+36.4/+5.5/−0.7%; distributed +49.0/+11.3/+46.7/+34.2/+4.7/−1.5% | +54.2/+9.1/+52.0/+38.9/+2.9/−3.1% (free ends, 0 or 0.7 m overhang, or anchored: within 0.1 point of each other) | reported; BM1-loop basis unchanged (FAIL) |
| T2-MEN | M | same, ordered RMSE | — | 11.48/3.16/11.02/9.08/3.98/3.81% | 11.98/3.13/11.46/9.52/4.33/4.47% | reported |
| T2-WITZ | M | Witz (1996) WITZ-ID, four-point sensitivity, wires anchored at the supports, p₀ = 1.037 MPa not re-identified: 300-bar width; energy; energy/width; 0-bar width | [1692, 2489] N·m; 3125 ± 15% J/m; [1.41, 1.73] 1/m; 552.7 N·m (identified) | 1383; 2732; 1.976; 552.6 | span 2P: 1657; 3212; 1.939; 668 (+21%). Span 4P: 1248; 2572; 2.060; 552.5 | FAIL (unchanged) |

The scorecard BM1-loop work (+12.9% with the former capstan bound, +8.8%
now) integrates the section's energy fields; the along-wire run has no exact
station energy field, so all rows above use the trapezoid of the reported
M dκ between inserted crossings, in which the direction-tied law read +12.2%.
In that quadrature the along-wire range (b) was +9.1–9.2% (RMSE 3.13%), 0.8
points inside the 10% target; the scorecard row was not changed by it
(decision rule: no change of a basis from a sensitivity, and the same
mechanism moves ranges (a), (c) and (d) further from the measurement). The
scorecard row now passes on its own basis, through the signed capstan bound.

**Reading.**

- **Sævik bellmouth: the along-wire model is contradicted.** Between the end
  fitting and the curved section the pipe is straight in the bellmouth tube,
  and the curvature rises to 0.146 1/m within 0.7 m of the curved-section
  start, against a slip length of about 8 pitches (a ≈ 50 at LC 14). The
  axial Coulomb chain feeds a mean wire force across this gradient and from
  the anchor, so the predicted ranges at the gauges are 2–6 times the
  sectional ones; the measured ranges stay near the sectional ones (median
  1.03). Along-wire redistribution therefore does not remove the 1.65–1.90
  over-prediction of the clamp correction: it raises it (3.36 / 4.62 with the
  package's stick representation, 2.12 / 2.90 in the clamp correction's
  convention). The small-angle under-prediction (0.45–0.73) is a stick-regime
  result and is unchanged.
- **Stick stiffness.** In slip the spring and rigid conventions carry the
  full geometric slip demand and agree within 5%; the `scaled` convention
  reduces the demand by ξ = 0.52 and the over-prediction to about 2. The
  capstan term added 9–11% in the former symmetric form, which raised the
  bound of a falling force as well as of a rising one; with the signed term
  its share of a full reversal range is of third order in `λ`
  (`2 σ0 sinh(λψ)` against `2 σ0 λψ`), so that figure overstates it. It has
  not been re-measured. The proportional-loading assumption of the clamp
  correction under-states the along-wire effect at this rig by 20–50%.
- **Short specimens: the end conditions do not matter, the gradient does.**
  Hu and Ménard are insensitive to the wire end condition and to overhangs
  (all within 0.1–0.3%); the change comes from slip redistribution between
  stations of different curvature inside the span. For the Hu conductor
  (slip lengths 17 to several hundred pitches against a two-pitch span) it
  raises the loop area from +37% to +65% and shortens the transition share
  from 33% to 11%: both move away from the test. For Ménard it lowers the
  work of ranges (b) and (e) by about 2–3 points and raises (a), (c), (d) and
  the magnitude of (f) by 1.5–3 points; it is not a consistent explanation
  of the work errors. On range (b) the stick convention moves the work by
  about one point (rigid +8.3%, spring +9.1%, scaled +9.4%).
- **Witz (arrangement not reported).** A 2P Witz specimen with anchored
  ends widens the 300-bar
  loop to 1657 N·m (band from 1692) but widens the identification loop by
  21% and leaves the energy-to-width ratio at 1.94 against [1.41, 1.73].
- **Overall.** Slip redistribution along the wires, as an axial Coulomb
  process with the package's contact pressures, explains none of the
  registered gaps and moves the two cases where it acts strongly (steep
  gradients and long slip lengths: the Sævik bellmouth, the Hu conductor)
  away from the measurements. The option is kept for bounding end effects;
  no default and no calibration is changed. Candidate missing physics where
  the measurements stay sectional: contact-pressure redistribution under
  bending and the curvature distribution in the rig, which is not published.
  Transverse slip and lateral tendon bending (Sævik 1992; Dong et al. 2017,
  2025) were tested with gap G1 and change the ratios by 1–3 % only.

## Anchor compliance of the end fitting (TA)

Option `k_anchor_N_m` of `TerminationSpec` and `AlongWireSpec`,
`anchor_stiffness_N_m` of `along_wire_history` and `WireChain.build`, and the
normalised `anchor_stiffness` of the clamp-layer solver; tests
`tests/test_anchor_compliance.py`; sensitivity script
`validation/scripts/check_saevik2011_anchor.py`. The anchor of every route
above is rigid (`u = 0`). A potted end fitting holds the wire end with a finite
stiffness `K` per wire (axial force per unit axial displacement of the wire
end), here a spring at the anchor node in parallel with the friction of the
node's half segment. Normalised, `k_a = K R/(EA sin α)` (`ξ EA` in the
`scaled` convention): the anchor is as compliant as `EA/K` of free wire. The
rigid anchor stays the default and runs the unchanged code path; nothing is
fitted. A shear-lag estimate of `K` for a potted fitting is derived in the
[local analysis guide](local_analysis_guide.md) (Section 5, Step 6): about
`1e8`–`2e9` N/m for a 5 × 12.5 mm tendon, `EA/K ≈ 0.007`–`0.16` m. It is an
order-of-magnitude estimate, not a value from a source.

**Verification.** Closed form: a wire under a uniform imposed strain `A`
(from the virgin state, uniform bound `ρ`, interface stiffness `k̂ = β²`,
`β = ∞` for rigid stick) slips over `L` next to the anchor, with
`k_a ρ L²/2 + ρ(1 + k_a/β) L + k_a ρ/β² + ρ/β = A`; the anchor force is
`A − ρ/β − ρL`, the dissipation `ρ²(L²/(2β) + L³/6)`, and without slip the
anchor force is `k_a A/(k_a + β)`. `k_a → 0` is the free end, `k_a → ∞` the
rigid anchor.

| ID | Class | Quantity | Target | Result | Status |
| --- | --- | --- | --- | --- | --- |
| TA-V1 | A | Closed form above, `A = 3`, `k_a` = 10, 1, 0.1, 0: clamp solver (rigid stick, 256 segments per pitch) and chain (rigid stick and `β = 2`, 512 per pitch): anchor force, slip length, dissipation | force within 1e-4 `A`; slip length within 1.5 `h`; dissipation within 2e-3 (clamp) and 5e-4 (chain) | force 1e-5 `A` or better; slip length within 1.2 `h`; dissipation 5e-4 (clamp), 4e-5 (chain) | PASS |
| TA-V2 | I | Clamp solver against the along-wire chain with the same spring anchor (cyclic curvature, `k_a` = 0, 0.3, 3) | < 1e-12 rel. | force 5e-15, dissipation 4e-16 | PASS |
| TA-V3 | A, I | Limits: `None`/`inf` the unchanged code path (clamp, field, chain, termination, bending test); `0` the free end (chain bitwise; clamp against the chain's free end); termination increment (Ménard, `k_a` = 0 to 1e6) between the free and the rigid result, monotone, both solvers agreeing | bitwise; < 1e-12; monotone, < 1e-4 at `k_a = 1e6` | bitwise; 4e-15; range 28.2, 31.1, 55.3, 63.0, 63.73, 63.74 MPa against rigid 63.74 (deviation 3.7e-7 at 1e6, falling as `1/k_a`); solvers 3e-15 | PASS |
| TA-V4 | I | Energy identity `W = ΔU + D + R` with elastic anchors (rigid and spring stick, with and without capstan, one and two springs) | round-off | 2e-13 of the work | PASS |
| TA-V5 | I | Station Jacobian with two elastic anchors against the finite difference in stick | < 1e-7 | yes | PASS |
| TA-V6 | I | Two-way bending test (two-pitch three-point specimen, anchored ends): `k_a` = 0 equals free ends; closed-cycle external work against wire friction work | bitwise; < 1% | bitwise; 0.89% (rigid 0.89%, free 0.91%); peak load 1551 (free) ≤ 1553–1555 ≤ 1555 N (rigid) | PASS |
| TA-V7 | I | Projections with spring ends (first, last, both) against a generic quadratic programme | < 1e-5 | yes | PASS |

**Sævik (2011) sensitivity (reported, not gated).** The rows SV11-PLANE and
SV11-GJ (termination-corrected: clamp solver, `scaled`, anchor at the
end-fitting face) and the along-wire variant of gap T2 (`spring`, capstan
bound, all wires between the two end fittings, the beam curvature field at
every angle sample) re-run with an elastic anchor `K` per tendon at both end
fittings (`EA = 1.29e7 N`; `python validation/scripts/check_saevik2011_anchor.py
OUTPUT_JSON`, about 20 min on one thread). `K` is not fitted; the registered
rows keep the rigid anchor.

| `K` per tendon [N/m] | `EA/K` [m] | SV11-PLANE median R (in band) | SV11-GJ | T2-SV-PLANE median R (in band) | T2-SV-GJ |
| --- | --- | --- | --- | --- | --- |
| rigid (registered) | 0 | 1.62 (3 of 14) | 1.80 | 3.11 (2 of 14) | 4.19 |
| 2e9 | 0.0065 | 1.62 (3) | 1.80 | 3.11 (2) | 4.19 |
| 5e8 | 0.026 | 1.62 (3) | 1.80 | 3.11 (2) | 4.19 |
| 1e8 | 0.13 | 1.62 (3) | 1.80 | 3.11 (2) | 4.19 |
| 1e7 | 1.3 | 1.62 (3) | 1.80 | 3.11 (2) | 4.19 |
| 1e6 | 13 | 1.62 (3) | 1.80 | 3.11 (2) | 4.19 |
| 0 (free wire end) | ∞ | 1.62 (3) | 1.80 | 3.11 (2) | 4.19 |

The first four non-rigid rows span the potting estimate; the last three
extend it to a free end. The largest change of any per-case ratio is 3e-15
(round-off) for the termination correction and 4.5e-4 for the along-wire
chain (at `K = 0`; 1.4e-4 at `1e8`). The along-wire values are a re-run with
the signed capstan bound (the T2 table above, 3.36 and 4.62, used the former
symmetric bound and has no script on record); per load case 0.44, 0.54,
0.74, 0.99 (0.5–2°), 4.96 and 1.91 (4°, LC 4 and LC 5), 3.16–3.51 (7.5°, 8° and
15° at 500 and 250 kN),
3.06–3.19 (24–32°), 2.67 (15°, 750 kN), 5.67 (15°, 17 MPa, LC 19).

Why nothing moves: the end-fitting face lies 2.305 m before the curved
section, inside the straight bellmouth tube, where the beam curvature is about
1% of the bellmouth peak (LC 14, 15°: 2e-4 against 0.025 1/m). The wire next to
the anchor therefore stays in stick, apart from the first half segment, and the
anchor condition does not reach the stations: the rigid anchor of the
along-wire chain carries a range of 79 MPa (LC 14, ±8°, all 55 wires) against
875 MPa at station 2.1. The slip zones that make the corrected ranges start in
the bellmouth and are held there by a stuck stretch of wire, which acts as the
anchor whatever the fitting does. At this rig the termination correction is a
curvature-gradient effect, not an anchor effect; anchor compliance cannot
explain the over-prediction, and the SV11 and T2 decisions stand. The option
matters where the curvature at the anchor is high: on a Ménard layer with
uniform curvature from the anchor (TA-V3, 20 `κ_c`) the corrected range falls
by 1% at `k_a = 10` and by 13% at `k_a = 1`; for a potted fitting
(`K ≈ 1e8` N/m, `k_a ≈ 20` there) it is below 1%.

## Anchored wire at uniform curvature against published FE (AW-DONG25, AW-ZT14)

The anchored solver in its own regime: one armour wire fixed at an end
fitting, the pipe bent from straight to a uniform curvature (monotonic,
proportional), rigid anchor. Two published finite-element studies of exactly
this problem disagree: Zhu and Tan (2014, OMAE2014-23928, Figs 4–8) find about
2.2 times the free-field axial stress at a neutral-axis anchor, Dong et al.
(2025, Marine Structures 102, 103761, Figs 14 and 25) almost no rise of the
peak. Script `validation/scripts/check_anchored_wire_fe.py` (`--spring` adds
the interface sensitivity below; about two minutes), tests
`tests/test_anchored_wire_fe.py`, digitised curves
`validation/paper_data/dong2025_figs9_14_25.csv` and
`zhu_tan2014_figs4_8.csv` (procedure, calibration, reading uncertainty and
source SHA-256 in the file headers). **Reported, not gated**: the comparison
was run before an agreement criterion existed, and Dong et al.'s two FE
results for the same wire differ by 8 MPa (below), so no criterion
independent of the result was available. Nothing is fitted.

**Solver.** `clamp_layer.anchored_wire_response`: rigid stick, Coulomb bound
per unit length, axial slip only, `u = 0` at both wire ends, curvature raised
from zero in steps of 0.01 `κ_c`, 720 segments per pitch (halving both moves
the values below by less than 0.1 MPa or 0.002 normalised; the anchor value
does not depend on the wire length or on where the far end lies). The
normalised problem has one parameter, `a = κ/κ_c`: the stick stress gradient
at the neutral axis over the friction gradient (line load over wire area).
For `a ≥ π/2` the far field is the full-slip triangle of peak `σ_u π/2`.

**The normalised anchored stress depends on μ.** The far field divided by its
peak is independent of the friction coefficient once the wire is in full
slip; the anchored boundary layer is not. The anchor must cancel the far-field
slip at its azimuth, whose ratio to the slip capacity grows with
`a ∝ 1/(μ P)`. Neutral-axis wire, anchor / peak stress over the free-field
peak: `a` = 1.5: 0.13 / 1.00; 2: 0.31 / 1.16; 3: 0.65 / 1.36; 4.92:
1.19 / 1.70; 10: 2.11 / 2.38; 20: 3.07 / 3.21; 50: 4.69 / 4.77 (test
`test_normalised_anchor_stress_depends_on_a`). Zhu and Tan's Eq. 13 carries the
same dependence through `u_stk/u_max`. A normalised comparison therefore
needs `a` from each source.

**Inputs and assumptions.**

- Dong et al. (2025), Table 1: 6 × 3 mm wire, helix radius 0.10 m, helical
  angle 55° measured from the hoop direction (their Fig. 2; Eqs. 3, 12, 46),
  i.e. lay angle 35° from the axis; E = 210 GPa; pipe strain 0.10 % (uniform
  wire stress `E ε sin²α0` = 140.9 MPa, compatible with both anchors, so no
  slip); curvature 0.005 1/m (stick amplitude 70.46 MPa); μi = μo = 0.10,
  po = 2.55 MPa. Line load from their Eq. 38,
  `f = μi Ta κ03 + (μi + μo) po w` = 3894.5 N/m, `σ_u` = 37.72 MPa per radian
  of helix angle, `a` = 1.868: just past full slip, free-field amplitude
  59.25 MPa. Assumed: (D1) the FE carries the Eq. 38 line load (both faces);
  the FE's free-field slopes support it (235, 244 and 235 MPa per pitch,
  straight-line fits to Figs 14, 9 and 25, against 237 with Eq. 38 and 144
  with the inner face only);
  (D2) a constant bound at the mean tension (the tension term is 21 % of `f`
  and bending changes the tension by ±42 %, so the true bound varies by about
  ±9 % along the wire); (D3) one pitch with both wire ends fixed to the core,
  as in the source's FE (its Sec. 3.1); (D4) rigid stick (the FE's tangential
  penalty stiffness is not printed) and axial slip only (the FE's lateral end
  layer stays within 8 % of a pitch).
- Zhu and Tan (2014), Table 1: lay angle 38°, mean radius 0.254 m,
  4 × 12 mm wire, E = 210 GPa (printed as MPa), bending radius 180 m (stick
  amplitude 184.0 MPa), contact pressure 2.55 MPa; μ not printed. `a` from
  their Table 2: with a full-slip triangle far field, `U = u_stk/u_max` of
  Eqs. 13 and 16 equals `8a/π²`; the three non-trivial rows give
  U = 3.988, 3.986 and 3.986 (one value within the rounding of the table), so
  `a` = 4.919 and the free-field amplitude is `4A/(πU)` = 58.8 MPa; their
  Fig. 14 (stress without the end fitting) reads 58 ± 2.5 MPa (`a` 4.8–5.2).
  Assumed: (Z1) the hoop angle is measured from the compressive extreme fibre
  (their Eq. 1 and Fig. 4); (Z2) the far end of the four-pitch FE wire is not
  stated (immaterial to the anchor value); (Z3) the plotted stress is
  normalised by the free-field peak (not stated; the FE curves read 0.97–1.06
  far from the anchor).

**AW-DONG25: reproduced at the documented verification case.** Axial stress
[MPa]; FE and theory digitised (±1 MPa on the FE lines).

| Wire, figure | Quantity | FE | CableXSec | Source theory | Status |
| --- | --- | --- | --- | --- | --- |
| θ0 = π/2, Fig. 14 | stress at the anchor | 156.9 | 156.4 | 147.8 | reported |
| same | peak (position X1/P) | 204.5 (0.22) | 207.8 (0.217) | 204.4 (0.25) | reported |
| same | X1/P = 0.745 | 86.6 | 82.8 | 88.6 | reported |
| θ0 = π, Fig. 9 | stress at the anchor | 194.5 | 200.0 (free field) | 200.0 | reported |
| Fig. 25, θ0 = 0, 45, 90, 135, 180° | stress at the anchor | 89.3, 121.2, 164.8, 199.1, 195.0 | 81.8, 117.6, 156.4, 191.9, 200.0 | — | reported |
| same | anchor stress minus free field (45, 90, 135°) | 11.1, 25.9, 29.6 | 6.3, 15.5, 21.4 | — | reported |
| same | free-field extremes | 195.2, 89–91 | 200.0, 81.8 | — | reported |

At the case the source documents (its Sec. 3: one pitch, both ends fixed,
Figs 9 and 14) CableXSec reproduces the FE stress at the neutral-axis anchor
within 0.5 MPa, where the source's own theory (unshifted friction signs) is
9 MPa low, and the position of the shifted peak (0.217 against 0.22 of a
pitch; +3.3 MPa, 1.6 %). The anchor raises the stress at the anchor by
0.26 of the free-field amplitude and the peak by 4 %; at the intrados and the
extrados it changes nothing (CableXSec: the free field to 8e-14, test
`test_extreme_fibre_anchor_leaves_the_free_field`; the source states the
same). The FE's free field is 5–9 MPa inside the rigid-stick triangle at its
extremes (rounded by the tangential penalty), the scale of differences that
are not anchor effects. Fig. 25 does not agree with Fig. 14 for the same
nominal wire (θ0 = 90°: 164.8 against 156.9 MPa at the anchor, 198.4
against 204.5 at the peak; all its curves end at 540°, a model length the
source does not state); against Fig. 25 the CableXSec anchor rise is 0.57–0.72
of the FE's, a gap of the size of the source's own Fig. 14/Fig. 25
difference.

**AW-ZT14: not reproduced; the published FE is not in the regime of its model.**
Normalised stress; FE digitised (±0.07), source analytical model rebuilt from
its Eq. 9 with the Table 2 shifts (its plotted curves read 0.31 / 1.39,
2.27 and 1.84), CableXSec at `a` = 4.919.

| θi | FE anchor / peak | CableXSec anchor / peak | Source model anchor / peak |
| --- | --- | --- | --- |
| 0°, 180° | no anchor effect | no anchor effect (exact) | no anchor effect |
| 45° | 0.97 / 1.59 (165°) | 0.05 / 1.33 (161°) | 0.30 / 1.40 (145°) |
| 90° | 2.22 / 2.22 | 1.19 / 1.70 (137°) | 2.16 / 2.16 |
| 135° | 2.19 / 2.19 | 1.76 / 1.76 | 1.80 / 1.80 |

- *The FE far field is not in slip.* Far from the anchor the FE curve of the
  θi = 0 wire is a cosine (RMS distance 0.028, inside the reading
  uncertainty), not the full-slip triangle (0.154) that a rigid-stick Coulomb
  wire must carry at `a ≥ π/2` and that the source's analytical model
  assumes. The FE wire behaves as an elastically coupled wire, normalised by
  its own stick amplitude.
- *The source's analytical field breaks Coulomb's law.* With the Table 2
  shifts, the friction of Eq. 9 acts in the direction of the wire's slip
  relative to the core over 90–164° on the θi = 90° wire (74° from the
  anchor) and over 144–180° on the 45° wire (test
  `test_zt_shifted_field_breaks_the_coulomb_flow_rule`). Its agreement with the
  FE is therefore not evidence for a Coulomb boundary layer. The Coulomb
  solution at the same `a`, incremental (1.19 / 1.70) or as one projection
  that obeys the flow rule in total form (1.05 / 1.63), stays well below 2.2.
- *An elastic interface gives the order of the FE values* (sensitivity, not a
  fit: the source does not print its tangential stiffness). With the along-wire
  chain's elastic interface (`"spring"`, normalised shear stiffness `k̂`) at the
  same `a`, the far field stays a cosine for `k̂ ≤ 0.25`, and the normalised
  anchor stress is 1.86–2.62 on the 90° wire, 1.94–2.40 on the 135° wire and
  the 45° peak 1.38–1.83 (`k̂` = 0.25 to 0.05); the FE's 2.22, 2.19 and 1.59
  lie inside these bands. For `k̂ ≥ 1` the far field slips and the values
  return towards the rigid ones (90° wire: anchor 1.06–1.36, peak 1.55–1.56).

**Why the two papers conflict.** Both anchor the wire in the same way: its end
is fixed to the core, zero relative axial displacement (Dong et al. also fix
the lateral displacement and rotation, which act within 8 % of a pitch). The
difference is the slip state. Dong et al.'s wire is just past full slip
(`a` = 1.87): the far-field slip that the anchor must cancel is small, and the
anchor adds 0.26 of the amplitude at a neutral-axis anchor and 13 % at the
peak (normalised). At the `a` implied by Zhu and Tan's own Table 2 the same
Coulomb wire gives 1.19 and 1.70 in CableXSec, a real and larger effect that
grows roughly as `√a` (clamp law `A = 1 + 0.54 √a` at large `a`). Their 2.2
is neither: their FE far field is stuck, so its anchor value is an elastic
shear-lag concentration normalised by a stick amplitude, and the analytical
model that matches it violates the flow rule. CableXSec agrees with the Dong
et al. FE; with Zhu and Tan it agrees in pattern (no effect at the extreme
fibres, the largest rise for anchors from the neutral axis towards the
tensile side), not in magnitude.

**For the termination route.** Where its assumptions hold (uniform curvature
from a rigid anchor, proportional monotonic loading) the anchored solver
reproduces an independent FE that prints the inputs to define it. This does
not validate the correction near a bellmouth, where the SV11 over-prediction
is a curvature-gradient effect (TA, above); it removes the anchored solver
itself from the suspects. Not tested: cyclic loading (both studies are
monotonic), an interface stiffness from a source that prints it, transverse
slip.

## Torque balance and twist under tension (gap 3.5: TQ, TQ-KL, TQ-UJ)

Script `validation/scripts/check_torque_balance.py`, tests `tests/test_torque_balance.py`, rows TQ,
TQ-KL and TQ-UJ of the scorecard. `torque_balance` solves the coupled radial
section at fixed and free twist; the layer torque is `s n A σ r sin α`
(Knapp 1981, Eq. 10). The strands are built as a solid steel core with
touching helical layers (`galvanised_steel_armour`, ν = 0.3); the moduli
cancel in every ratio below. Tolerances were registered before the runs
from the reading precision of the source values and the omitted wire
moments (Costello 1997, Example 3.1: −5% torque for a seven-wire strand).

| ID | Class | Source and quantity | Target | Result | Status |
| --- | --- | --- | --- | --- | --- |
| TQ1 | A | Feyrer (2015) Example 2.8 with Fig. 2.4: torque constant `c_1S` of the 1×37 spiral rope (layers +14°, −14°, +14°), Eq. 2.74 | 0.0345 ± 0.2% (fixed radius); ± 1% (solved) | 0.03446 (−0.11%); 0.03470 (+0.6%, Poisson contraction of the centre wire, which Eq. 2.70 neglects) | PASS |
| TQ2 | A | Knapp (1981) Eqs. 18 and 20: fixed-twist torque per unit strain of the solved section with a radially stiff core (modulus ×1000) against the core factor `F_c = 1 − θ R_c tan²α / (2R)`, θ = 2ν | < 1e-4 relative | −1.3e-6 (the rigid-core value differs by 3.3e-3) | PASS |
| TQ3 (TQ-KL) | MM | Kim and Lee (2017) Section 5.1, 1+6+12 strand (Table 4): outer helix angle that balances the torque for six inner angles (Fig. 19 legend) and the final design (78.85°, 87.41°), beam FE with friction 0.115 | abs. difference ≤ 0.3° | −0.06 to −0.26° (final design 87.28° against 87.41°, an outer lay angle from the axis of 2.72° against 2.59°) | PASS |
| TQ4 (TQ-UJ) | M, MM | Utting and Jones (1987), Part I Table 1 (geometry, measured lay) and Part II Table 1: fixed-end torque per kN of six seven-wire strands, Machida–Durelli theory (column ii) and test (column vii) | model/theory 1.04–1.07; model/test 1.00–1.15 | 1.051–1.059; 1.067–1.136 | PASS |
| TQ5 | A | Mirror lay: every lay direction reversed (armour and laid cores) on the 1×37 rope and the 66 kV template | < 1e-9 relative | torque and twist reverse, wire stresses and balance ratio unchanged (0 and below 1e-9) | PASS |
| TQ6 | I | `fixed_radius` against `xsec_stiffness` (`EA_N`, `EA_torsion_coupling`, `GJ_Nm2`); solved 66 kV template against the fixed-radius torque coefficient | identical; < 1% | identical; 1.5026 against 1.5102 mm (−0.5%); EA 607 against 637 MN | PASS |
| TQ7 | I | Design round trip on the 66 kV template: zero torque coefficient at the returned lay angle; free twist and stress redistribution then vanish | < 1e-9 m; < 1e-12; < 1e-3 Pa | 1.3e-13 m; below 1e-12; below 1e-3 Pa | PASS |

The excess over the Machida–Durelli theory is the omitted bending and
twisting moments of the wires, which scale with `(d / 2r)²` (0.23 for these
strands, 3.4e-4 for the 66 kV armour). The tests lie a further 2–7% below the
theory; Utting and Jones attribute part of this to end rotation in the grips.
Kim and Lee's angles are figure legends given to 0.1° from a beam FE model,
not measurements. The as-built balanced Kevlar cable of Knapp (1981, Table 1;
1.07 N·m at 42.3 kN) is not used: the core modulus is not given.

Solved and fixed-radius values differ where the radial support of the armour
is compliant. On the packaged cables (`examples/torque_balance_templates.py`,
100 kN) the 66 kV template agrees within 0.5%; the Ménard cable gives 3.45
against 1.76 mm because its void interstices let the armour contract (solved
EA 111 against 280 MN); the 33 kV single-core template gives −2.08 against
−1.56 mm and the three-core template −1.94 against −1.64 mm. Before the
geometry correction the single-core template gave −3.60
against −1.49 mm, because 1.0 mm undeclared radial gaps between conductor,
insulation and screen left the armour unsupported, and both 33 kV templates
had a lateral armour fill of 1.13–1.24 (overlapping wires); their fill is now
0.98–0.99. The 220 kV and TB 880 cables need a strand radial law for the
solved section (fixed-radius values only).

## Power-core screens, core slip state and sheath strain life (gaps P1, P6, P8)

Tests `tests/test_core_screens_sheaths.py`; example
`examples/thies2016_screens_sheaths.py`. Verification only: no published
screen or sheath fatigue test with locations is held.

| ID | Class | Check | Target | Result | Status |
| --- | --- | --- | --- | --- | --- |
| P8-A | A | Stuck core: strain at every core station and section angle equals the plane-section strain `κ x` (aligned cores); laid core axis strain `r_c cos²β κ cos φ` | 1e-15 absolute | below 1e-15 | PASS |
| P8-B | A | Free core: mean strain over the core section (the axis strain) is zero; amplitude bounded by `κ ρ` times the larger Lutchansky factor | 1e-17 | exact zero | PASS |
| P8-C | A | Frictional state: zero friction gives the free state; rigid contact with `μ = 50` gives `ξ = 1`, no slip and the stuck strain; nominal state between the bounds | 1e-12 | PASS | PASS |
| P1-A | A | One core at the cable axis: per-core screen equals the existing helical-layer result (stress ranges at four fibres, bulk damage, `ξ`, `P1 + P2`), stuck and free | 1e-9 relative | PASS | PASS |
| P1-B | A, I | Stuck core adds `2 κ r_c cos²β` to the axis-strain range and no slip; declared outer-face pressure reaches the outer face; missing pressure warns | exact | PASS | PASS |
| P6-A | A | Modified universal slopes rewritten in reversals reproduce the range form | 1e-12 | PASS | PASS |
| P6-B | A | Fully reversed SWT with compatible `K'`, `n'` equals the uncorrected curve; Morrow with zero mean likewise; tensile means shorten life | 1e-9 | PASS | PASS |
| P6-C | A | Masing branches: doubled curve, closed-loop memory, return to the cyclic curve | 1e-12 | PASS | PASS |
| P6-D | A | Concentric sheath strain-life route: strain amplitude `SCF f_corr κ r`, damage `n / N_f`; elastic route unchanged and strict | 1e-6 | PASS | PASS |
| P6-E | I | Thies (2016) laminate, free core, nominal assumed foil inputs: damage per cycle against the hand `κ r` calculation | 0.2% aligned, 1% laid | +0.05% aligned (laminate hoop-strain factor), +0.7% laid (core-lay bending factor) | PASS |
| P1-T | I | Thies (2016) screen, free core, aligned: range, slip, bulk damage and fretting count of the hand core-level calculation (former capstan bound), corrected for the signed bound in full slip: range less `2σ0 (cosh(λπ/2) − 1)` (6.07 MPa), damage by the range ratio to the curve slope | 1% | 248 MPa, 822 µm, 6.2e-6 per cycle, fretting 37 on 4 wires | PASS |

Published strain-life points (`ε_a`, `N_f`) of a sheath, foil or screen
material are not in the reference library, so the registered constants are
checked against their own source equations only. The copper set is that of
Poon et al. (2022, Table 4); the aluminium foil set is an assumed bound.

## Restrictor and stiffener fields, tapes, copper strain life (gaps P5, P2, P7)

Tests `tests/test_restrictor_tapes_copper.py`; example
`examples/thies2016_screens_sheaths.py` (Part B). Verification only.

| ID | Class | Check | Target | Result | Status |
| --- | --- | --- | --- | --- | --- |
| P5-A | A | Rigid-link chain: `R_L = p / (2 sin(θ_L/2))`, arc value within `θ_L²/24`, `from_mbr` inverse; Thies (2016) pitch 0.36 m and 4.75° give 4.344 m against the stated MBR 2.54 m (8.13° per joint) | 1e-9 | PASS; the discrepancy is reported, both readings run | PASS |
| P5-B | A | No lock: field equals the tensioned-beam end-rotation profile of the rig solver (tensioned and untensioned) | 1e-12 | PASS | PASS |
| P5-C | A | Chain limit (`λ ≪ p`, long span): one element locks per `κ_L p` of root rotation | exact count | PASS | PASS |
| P5-D | A | Smeared rule, long span: front `a = θ/κ_L − λ`, cable moment continuous at the front | 5e-4 (finite span) | PASS | PASS |
| P5-E | A | Bend stiffener: bare cable at small angle `κ = θ/λ exp(−s/λ)`; a cone lowers the peak; odd in the angle | 2e-4 relative | PASS | PASS |
| P5-F | I | Proportional fields through the new paths: field clamp kernel against the proportional kernel; rig `"field"` against `"table"`; termination (clamp and along-wire) with a uniform field against the default; cantilever with a restrictor against `min(F(L−x)/EI, κ_L)` and the restrictor moment | 1e-10, 1e-6, 1e-9, 1e-9 | PASS | PASS |
| P2-A | A | Tape geometry: pitch, axial gap or overlap, coverage, edge angle | exact | PASS | PASS |
| P2-B | A | Tape states: stuck = surface strain `ε_z (cos²β + h sin²β)` at radius `ρ + t/2`; free = uniform part plus flat-wise Lutchansky bending, slip amplitude `ρ² κ abs(cos²β + h sin²β) / sin β`; frictional at zero pressure = free, at large pressure = stuck | 1e-12 (slip 3%, discrete integral) | PASS | PASS |
| P2-C | I | Core tapes follow the core slip state (stuck > free), hoop ratio from the polymer below | ordering | PASS | PASS |
| P7-A | I | Strain-life route leaves the elastic screen and conductor results bit for bit unchanged; damage returned per station and wire, per ring | exact | PASS | PASS |
| P7-B | I | The two copper sets (`copper_poon2022`, `copper_annealed_mus_derived`) agree within a decade at 0.2% amplitude | factor 10 | PASS | PASS |
| P7-C | I | `copper_stranded` `mu_on_cu` 0.25 is a labelled assumption with its bound; value unchanged | exact | PASS | PASS |

*Thies (2016) with the restrictor kinematics (P5-T, information).* Free
core, Z-ram tension at the largest angle, secant `EI` 3.7–4.9 kN·m²;
damage summed over the four blocks:

| Quantity | Lock from MBR 2.54 m | 4.75° per joint (4.34 m) | Constant-radius models (Part A) |
| --- | --- | --- | --- |
| Smallest radius 28/14/14/7° [m] (video 2.0/3.5/3.5/7.0) | 1.79/2.21/1.82/4.05 | 3.32/3.43/3.37/4.05 (reverse 2.00 m at the tailstock, 28°) | 2.54 or 2.0/3.5/7.0 |
| Locked elements at 28° (length) | 3 (1.08 m) | 7 (2.52 m) | – |
| Al laminate D at x = 0.05/0.6/1.2/2.0 m | 49/6.1/0.56/0.002 | 20/11/1.3/0.24 | 49/33/0.74/0 (lock 2.54) |
| Cu tape D, stuck, 50°, 20 mm (assumed) | 0.11 at 0.05 m | 0.013 at 0.05 m | not run |
| Screen wires D at 0.05 m: elastic S–N / strain life | 0.47 / 0.10 | 0.38 / 0.092 | 0.47 / – |
| Armour at the clamp (OC2017 in-air design curve): peak range, D at 0.02 m | 635 MPa, 7.5 | 416 MPa, 0.75 | 318–642 MPa, 11–13 (lock 2.54, anchored) |

The restrictor kinematics reduce the armour damage at the clamp of the
lock-2.54 model from 11–13 to 7.5 and keep the 4.34 m reading below one;
the laminate stays cracked within 0.6 m under both readings. The bent copper
tape does not reach D = 1 at a 50° lay with the assumed width (the
observed edge cracks need the edge contact stress, not modelled). The
strain-life route gives about one fifth of the elastic S–N damage of the
screen wires. None of these is a validation: the restrictor data are
inconsistent and the construction inputs are assumed.

## Rectangular tendons and lateral relief (gap G1)

Modules `cablexsec/mechanics/tendon_section.py` and
`cablexsec/mechanics/tendon_bending.py`; cable files `saevik2011_rect.yaml`
(5 x 12.5 mm tendons; polymer modulus identified again on the 72.8 kN m²,
410.6 MPa) and `witz1996_rect.yaml` (6 x 3 mm strips; carcass and Zeta as
profiled sections with the Witz Table 2 area and inertias); tests
`tests/test_tendon_section.py`, `tests/test_tendon_bending.py`,
`tests/test_rect_tendon_paths.py`, `validation/scripts/check_out_von_morgen.py`
(`tendon_bending_checks`). Targets were registered before the benchmark
runs (the strip full-slip EI of the Witz pipe was seen before registration
and is reported, not gated). Nothing is fitted.

Verification:

| ID | Class | Check | Target | Result | Status |
| --- | --- | --- | --- | --- | --- |
| G1-V1 | A | Rectangle area and inertias; rounded corners against a fine grid; torsion series (square 0.140577 a⁴, thin strip, Roark); corner maximum of a linear field | exact; 5e-4; 0.5 % | PASS | PASS |
| G1-V2 | I | Round wires bit for bit: SV11 LC 7 and LC 14 (880 recorded station values and the GJ predictions), SV11 and T2 summary values, WITZ-ID round | exact | PASS | PASS |
| G1-V3 | A | Curvature coefficients: loxodrome = fatigue path; geodesic lateral 0, normal `3 cos² a` (Sævik 1992 Eq. 2.32; Out and von Morgen Eq. 39), torsion Sævik Eq. 2.31; numerical geodesic on the torus; `w_g` = Sævik Eq. 2.30; Sævik Eq. 4.2 slip threshold | 1e-12; O(k) | 4e-16; 2e-5 at k = 1e-3 | PASS |
| G1-V4 | A | Transverse Jenkin amplitude against Sævik (1992) Eq. 4.1 (single-sine `4/π` on the friction term) | 1e-9 | PASS | PASS |
| G1-V5 | MM | Dong et al. (2018) Table 1 anchor stress (μ 0.025, 0.1); Dong et al. (2025) Table 1 (μ 0.025, 0.1) | 0.1 % | 21.52/33.58 against 21.509/33.573 MPa; 35.46/55.62 against 35.463/55.624 MPa | PASS |
| G1-V6 | I | Relief at zero demand is the identity in the sectional, termination and along-wire paths | exact | PASS | PASS |

Sævik (2011) and Gjøsteen et al. (2027), inner armour, gates as SV11
(median R in [0.85, 1.18] and at least 12 of 14 in [0.70, 1.43]; GJ ratio
in [0.72, 1.38]); entries are median R, count in band, GJ ratio:

| Model | Round (record) | Rectangle | Rectangle + lateral relief |
| --- | --- | --- | --- |
| Sectional, direction-tied | 1.00, 11, 1.24 | 1.06, 11, 1.25 | - |
| Sectional, distributed | 1.03, 11, 1.31 | 1.08, 11, 1.31 | 1.08, 11, 1.30 |
| Clamp correction (clamp solver) | 1.62, 3, 1.80 | 1.72, 2, 1.85 | - |
| Clamp convention, along-wire termination | 1.63, 3, 1.74 | 1.73, 2, 1.80 | 1.72, 3, 1.77 |
| Along-wire, spring stick and capstan (T2 basis) | 3.36, 2, 4.62 | 3.40, 2, 4.68 | 3.37, 2, 4.68 |
| 0.5/1/1.5° ranges | 0.45/0.54/0.73 | 0.50/0.59/0.80 | 0.50/0.59/0.81 |

The direction-tied sectional and clamp-correction rows are regenerated with the
signed capstan bound (`check_saevik2011.py`, `--tendons rectangular`; former
values 1.03, 11, 1.32 / 1.09, 11, 1.32 and 1.65, 3, 1.90 / 1.76, 2, 1.94); the
other rows were computed with the former bound.

All rectangle variants FAIL SV11-PLANE; SV11-GJ passes for the sectional
variants only. The registered hypothesis that lateral/transverse relief near
the anchor brings the anchored-wire models to the measured near-sectional
ranges is **not supported**: the relief (vector Coulomb, transverse demand
of tension, strong-axis stiffness and the bellmouth gradient, taken at its
largest value for the whole history, an upper estimate) changes the
termination-corrected ratios by 1-3 %. The true section changes the chain by
1-10 % through the stick stiffness only (ξ 0.52 to 0.58 with the 12.5 mm
support width). The end-restraint layer of the tendon is 0.1 m (Dong 2018;
stick boundary 0.05 m), 18 decay lengths short of the nearest gauge. At the
neutral-axis gauge 2.3 (LC 14) the stuck (loxodromic) lateral bending at the
assumed grooves (`y_l = ±5.25 mm`) gives a single side of 338 MPa against
149 MPa measured (round-wire value 223 MPa): the tendon moves about half way
to the geodesic, which the Coulomb transverse model at μ = 0.2 does not
produce (demand 0.22 of capacity). Even a motion of that size everywhere
would act on the axial friction like a halved friction coefficient, the
SV11-ID case (termination median 1.16, 8 of 14; sectional 0.58). At the
bending-plane stations the model gives no lateral bending, where
Gjøsteen's loops at 2.1 and 3.1 are inclined.

Witz (1996) on the true strips, solved radial route (pressure barrier =
pressure sheath, closed-end wall tension 116.1 kN, μ = 0.10), gate as
WITZ-ID: p₀ 0.981 MPa identified on the 0-bar width; 300-bar width 1486 N·m
(−25.3 / −31.3 %, band 1692–2489) **FAIL**, energy −5.9 % (inside), peaks
+15 / +6 %. The bounded route on the strips gives the round-wire width
(−30.5 / −36.1 %); the geodesic full-slip state (η = 1) changes EI_min
(1400 to 1235 N·m², reported; loxodromic 1400 N·m² is outside the round-wire
WITZ-EI band, geodesic inside) and the peaks (+3 / −2 %), not the width.
The solved-route width on the true geometry equals that of the earlier
thickness-wire variant (−25.4 / −31.4 %): the variant was not the cause of
the 300-bar width deficit.

## User S–N data and capacity check of time series

Modules `cablexsec/materials/mean_stress.py`,
`cablexsec/analysis/sn_assessment.py`, `cablexsec/analysis/capacity_check.py`
and the registry functions of `cablexsec/materials/sn_curves.py`; tests
`tests/test_sn_user_options.py` and `tests/test_capacity_check.py`.
Verification only: these are user options and post-processing, not new
mechanics, and none is compared with a measurement.

| ID | Class | Check | Target | Result | Status |
| --- | --- | --- | --- | --- | --- |
| SN-1 | A | `SNcurve.from_knee` with the DNV-RP-C203 (2024) Table 2-1 D curve (m 3, log a 12.164, m 5, knee 10⁷): fatigue limit and second intercept | table rounding (0.1 %) | 52.64 MPa against 52.63; log a₂ 15.6067 against 15.606 | PASS |
| SN-2 | A | Miner sum of constant-amplitude blocks on both branches against `n / N(Δσ)`; cut-off gives zero; equal to `miner_damage` (compiled and reference counters) | 1e-12 | PASS | PASS |
| SN-3 | A | Thickness exponent: damage ratio `(t/t_ref)^(k m)` above `t_ref`, factor 1 below (DNV-RP-C203 Eq. 2.10) | 1e-12 | PASS | PASS |
| MS-1 | A | DNV-RP-C203 Sec. 2.5: `f_m` 1.0, 0.8, 0.0 (Eq. 2.15) and 1.0, 0.9, 0.8 (Eq. 2.16) for tension, zero mean and compression, as stated in the standard | exact | PASS | PASS |
| MS-2 | A | Mean-stress identities (Dowling 2013, Ch. 9 forms): Goodman, Gerber, Soderberg at σ_a 100, σ_m 250 MPa; SWT; Walker γ 0.5 equals SWT, γ 1 none; zero mean is the identity | 1e-14 | PASS | PASS |
| MS-3 | A | Round trip: a cycle at the curve's own R (0.1, 0.5, −0.5) or constant mean maps to itself for every correction | 1e-12 (1e-9 Walker bisection) | PASS | PASS |
| SCF-1 | A | Uniform component SCF `k` multiplies damage by `k^m`: armour fibres, conductor rings, sheath (m 7), tape (m 6.16) | 1e-9 | PASS | PASS |
| SCF-2 | I | Defaults and identity options bit for bit; mean-only factors leave uncorrected damage unchanged; the separated fibre history equals `at_fibre` | exact; 1e-9 | PASS | PASS |
| CAP-1 | A | Synthetic history with known maximum (1.5 at 71.5 s), exceedance samples, up-crossings, time above, MBR ratio | 1e-12 | PASS | PASS |
| CAP-2 | A | Gumbel method of moments: formulas; 40,000 samples of Gumbel(2, 0.3) recovered; shift `β ln n` for block maxima | 0.01; 3 %; exact | PASS | PASS |
| CAP-3 | I | Envelope level at η equals `wire_yield_curvature(usage_limit=η)` bit for bit; utilisation 1 on the capacity curve; utilisation monotone in curvature | exact; 1e-12 | PASS | PASS |
| CAP-4 | I | Floating-wind `local_response` history (clamped hang-off, DLC 1.2, 10.74 m/s, one hour; Ménard section, η 0.67, assumed MBR 1.2 m): interpolated maximum against a direct evaluation at the same sample | 3 % | 0.0813 at 303.5 s against 0.0829 (below the lowest level, 0.1); no exceedance; MBR ratio 0.071 | PASS |
| CAP-5 | I | Sea-state batch: per-run maxima equal `check_capacity` on the same histories; fatigue results unchanged; Gumbel fit over the seeds | exact | PASS | PASS |

CAP-4 is a plumbing check on a real global history, not a capacity
statement for that cable: the Ménard YAML gives no MBR and the hang-off
utilisation is far from one. Interpolation between level curves is the
resolution limit; finer `levels` reduce it.

## Coiling capacity formulas (CH-CR)

Source: Chae, Nam and Shim (2023), Ocean Eng. 288, 116014. The coiling request
(`cablexsec.compute_coiling`) states the radial-buckling capacity of the armour
in the source's closed form. The check below evaluates that form from the
source's own inputs. No measured coiling stress exists for the source's cable,
so the coiling stresses themselves are not benchmarked.

| ID | Class | Check | Target (registered before running) | Result | Status |
| --- | --- | --- | --- | --- | --- |
| CH-CR | A | Critical radial-buckling stress σ_cr = (1/σ_c1 + 1/σ_c2)⁻¹ (Eqs. 17–19) from the source's own n R and α (Table 6) | Table 6 rounding (0.5 MPa) | largest difference 0.3 MPa (σ_c1 at 2.2 m) | PASS |

Request `cablexsec.compute_coiling`, tests `tests/test_coiling.py`. Any coiling
result depends on the core lay, which must be stated with it.

### Torsional stiffness of the coiling cable (CH-GJ, model-to-model)

The source's Table 5 gives the torsional stiffness of its cable from three
cross-section tools; none is measured. CH-GJ reports the free torsional
stiffness of the packaged cable at the source's twist (0.125 rad/m, Table 4;
the sense that unwinds the armour; zero tension), as a stated comparison
without a gate:

| Core lay | Armour pitch 1.8 m (kN m²) | Armour pitch 3.0 m (kN m²) |
| --- | --- | --- |
| Range of the three tools (Table 5) | 59.6–83.2 | 34.3–44.7 |
| Straight (packaged) | CH_S18 | CH_S30 |
| S, assumed pitch 2.5 m | CH_A18 | CH_A30 |
| S, assumed pitch 3.0 m | CH_B18 | CH_B30 |

- **Core lay.** The source lays the cores S, like the armour (Sec. 2.1.2), but
  gives no core pitch; the packaged file keeps straight cores. Under an
  unwinding twist the armour shortens; straight cores resist that shortening
  at the free end, whereas S-lay cores shorten with it, so straight cores give
  the larger stiffness. The core pitch is not identified on the tools.
- **Terms.** With straight cores at 1.8 m: the fixed-radius armour term
  `K_tt − K_et²/K_ee` is 81.9 kN m², the Saint-Venant terms of the cores about
  their centres 21.0 kN m² and of the filler lobes 0.3 kN m² (the yarns are
  servings and carry no torque); the radial response of the unwinding armour
  lowers the armour part of the solved section to about 60 kN m².
- **The closed form of the source (Eq. 14).** As printed, the coupling term
  sits inside the sum over helical layers; for one layer it then cancels the
  armour term exactly and leaves `Σ G_c J_c`, independent of the pitch.
  Komperød (2014), which the source cites, assembles the cable matrix first and
  then takes `k22 − k12²/k11` with the cylinders in `k11`. The published column
  (69.6–25.1 kN m²) is proportional to `1/P²` within 0.2 % and therefore holds
  no pitch-independent `G_c J_c` term; it is not used as a reference here.

## Planned benchmarks

None of these is a completed comparison. Each will be registered under the
procedure in [validation_policy.md](validation_policy.md) before any result is
quoted.

| Benchmark | Class | Quantity | Prerequisite |
| --- | --- | --- | --- |
| Skeie, Sødahl and Steinkjer (2012), Fig. 10 | MM | Armour stress hysteresis at several helix locations; rectangular 5 × 10 mm armour, 38 + 42 elements, axial force 1000 kN, curvature to ±0.3 1/m | Rectangular wire sections are in place (gap G1); remaining: core support and loading conventions from the source and registered targets; no equivalent circular diameter, because it would not reproduce the rectangular section's bending stress or inertia |
| Zeidan (2025) PhD thesis; Zeidan et al. (2026) | M | Four-point bending of a 112 mm three-core 12/20 kV cable: stick and slip stiffness and slip threshold (reported K1 106 kN·m², K2 9.00 kN·m², M_y 1.83 kN·m) and loop shape | Construction and contact inputs from the thesis; target declared before running; identified and transferred states separated |
| Tan, Quiggin and Sheldrake (2009), closed curvature path | MM | Moment trajectory on the path κx to 0.09 1/m, then κy to 0.09, then κx removed, then κy removed (Figs. 16–17), against a resultant-curvature vector model and a per-wire model | Implement the vector model as a reference; same construction inputs for all models; no measured 3D data exist |
| Karlsen-Husøy et al. (2026), Adaptive Hysteresis Model | MM | Conductor strain under identical tension and biaxial curvature histories; projection onto eight fixed orientations with a multilinear operator (Eqs. 6–10, Tables 3–4) | Rebuild the model from the published parameters; comparison of reversal laws only, since the calibration is to a proprietary cross-section model |
| Cen et al. (2026), cable recovery | MM | Slip stiffness against tension and armour and lead-sheath stress laws (Table 4, Figs. 12–16) | Construction from Tables 1–2; quasi-static, no fatigue |
| Kwon et al. (2026), 66 kV stress factors | MM | Trend of armour and conductor stress factors under tension only (Fig. 12) | Qualitative only: the 66 kV geometry appears only as a drawing |
| Strain-gauged armour test | M | Measured armour wire stress under combined tension and cyclic bending | Test with instrumented armour wires, measured clamping state and sensor definitions; required before any wire-stress or fatigue-life claim |

Further candidates from the literature review. Each is listed only because
the published tables or figures that hold its data are identified; none is a
power-cable armour-stress measurement, and the flexible-pipe and umbilical
cases test the armour stick–slip stress mechanism, not power-cable specifics.

| Benchmark | Class | Quantity | Prerequisite |
| --- | --- | --- | --- |
| Gjøsteen et al. (2027), flexible-pipe FBG and bird-caging test | M | Remaining: the LC 14 loop shapes (Figs 27–35) beyond the ranges used in SV11-GJ; bird-caging failure at 256 kN and 1.2% strain (Figs 24–25) | The FBG ranges are compared in the Sævik (2011) section (SV11-GJ); loop shapes need a curvature-gradient model that the SV11 result does not yet support. The bird-caging case (de Sousa tests) is compared in "Armour compression" (ST17-SOUSA) |
| Le Corre and Probyn (2009), steel-tube umbilical | M | Membrane and bending strains of three steel tubes under tension and cyclic bending, test against FEA and analytical (Tables 1–5) | Umbilical bundle construction; test details are partly proprietary |
| Dai, Sævik and Ye (2020), steel-tube umbilical with FBG | M | Tube strain ranges at a bellmouth (Tables 1–5) | Expected failure without hoop contacts: the source shows radial contacts alone mispredict by +87% and −55% (Table 4); run as a documented limit of a radial-only section model |
| Feld et al. (1995), three-core submarine power cable | M | Conductor strains under tension to 15 t and bend radii 6–2 m (Tables 3–4, Figs 11–15); interfacial shear constant against radial compression (Fig. 8) | Cable construction from the paper; conductor strains require the conductor treatment; no armour strains were measured |
| Witz and Tan (1995), ROV umbilical rotary bending | M, MM | Armour geometry (Table 1), test configurations with minimum bend radii 2.80, 0.64 and 0.49 m (Table 2), source-computed alternating armour stresses (Table 3), outer-armour failure at 145,000 cycles | Arbitrary curvature distribution along the specimen; Table 3 column pairing to be re-read from the PDF; the stresses are the authors' calculation (MM), only the life is measured |
| Knapp (1979), electromechanical and logging cables | M | Tension–torsion stiffness and fixed-end torque and free-end rotation against test (properties Tables IV–V, results Tables VI–VII; source differences 5.7% and 16.0%) | Compressible core as in the source |
| Sævik and Gjøsteen (2012), umbilicals 1–8 | M | Axisymmetric stiffness of eight umbilicals with full parameters (Tables 4–12) against tests (Tables 2–3) | Coupled radial–axial–torsional section equilibrium (BM4 shows the radial-free assembly fails) |
| Su, Zhao and Zhu (2026), 35 kV and 36 kV cable sections | MM | Tensile, compressive and bidirectional torsional stiffness against FE (sections Tables 1–2, FE Tables 6–8) | Compression branch and interlayer separation |
| Poon et al. (2023), floating-wind conductor fretting chain | MM | Contact slip, pressure and fretting lives for DLC 1.1 and 1.6 (loads Table 3, 7-wire geometry Table 4, lives Table 10) | Conductor contact output; CableDyn global loads for the VolturnUS-S case |
| Li et al. (2026, 2027), thermo-mechanical helical cable | MM | Closed-form thermal response (materials Table 1) and thermal-bending stick–slip loops at 30, 60 and 90 °C against FE (Table 1, Figs 13–20) | Uniform-temperature state as in the sources; heating mode declared |
| Fang et al. (2026), thermo-electro-mechanical RUC | MM | Bending moment against heating with separated expansion and softening (Tables 1–3, Figs 3–4, 16; public code) | Equivalent residual-stress calibration as in the source; heating mode declared |
| Poradowski and Anders (2025), corrugated-sheath thermal resistance | MM | Equivalent thermal resistivity of the corrugated sheath and air layer for 72 geometries and the CIGRE TB 880 400 kV cable | Corrugated-sheath air-gap resistance implemented |
| Komperød, Konradsen and Slora (2015), helical end effects | A | Closed-form axial strain of helical elements between two fixed points under bending (Tables 1–2) | The end-restraint solvers are in place (clamp correction; along-wire solver, gap T2); remaining: registered targets |

Electro-thermo-mechanical benchmarks for the energised-cable chain.

| Benchmark | Class | Quantity | Prerequisite |
| --- | --- | --- | --- |
| CIGRE TB880 (BM5) completion | MM | Conductor temperature of the three-core 220 kV cable from the cable definition, not only the tabulated network | λ1″ eddy-current loss and the IEC 60287-2-1 three-core factor; full IEC 60287-1-1 and 2-1 texts (previews only in the library) |
| Transient conductor temperature under a current step and a daily cycle | A | Crank–Nicolson solution against the closed-form exponential response of a lumped ladder | The current-history driver is in place (`ThermalHistory.from_current`, gap 4.1); remaining: registered targets |
| Fixed temperature against temperature history | I | Armour damage at the governing and hang-off sections of the floating-wind reference case with a synthetic power-correlated current history | Clamping history into `fast_slip`; joint current–sea-state binning |
| Komperød (2017), bending at two temperatures | M | Bending-stiffness change between +5 and +20 °C | Published construction and heating condition |

## What the register does not cover

One comparison tests recovered armour wire stresses against measurement: the
flexible-pipe FBG ranges of Sævik (2011) and Gjøsteen et al. (2027) (SV11-PLANE,
SV11-GJ, both FAIL). No comparison tests fretting quantities or fatigue lives
against measurement, and no measured armour stress of a power cable exists in
the data held. The bending comparisons concern section moment, loop work and
armour displacement. Run times reported in the companion article
describe the cost of CableXSec's own calculations and are not a speed
comparison with any other program.
