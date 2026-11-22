<!-- SPDX-License-Identifier: Apache-2.0 -->

# CableXSec verification and validation

This is the validation record of CableXSec. It is for users and reviewers who need the
evidence behind a CableXSec result: each case lists its reference, metric, measured result and
acceptance value, and names the script or test that reproduces it. The table is generated from
the live code by `python validation/scripts/run_validation.py`, which writes
[`validation/validation_report.md`](https://github.com/SMI-Lab-Inha/CableXSec/blob/main/validation/validation_report.md);
the benchmark definitions, registered targets and discussion are in the benchmark register of
the manual ([Benchmarks](https://cablexsec.readthedocs.io/en/latest/benchmarks.html)), and the
rules that govern targets and statuses in the
[validation policy](https://cablexsec.readthedocs.io/en/latest/validation_policy.html).

Case identifiers group the evidence into two tiers:

| Tier | Evidence |
|---|---|
| 1 | Analytical verification, internal consistency and comparisons with independent numerical methods |
| 2 | Comparisons with published measurements and published tabulated results |

Every input is a published value, a file in this repository or a declared assumption. The digitised measurements are
in [`validation/paper_data/`](https://github.com/SMI-Lab-Inha/CableXSec/blob/main/validation/paper_data/README.md)
with their sources. Targets were registered before the comparisons were run and are not relaxed
afterwards: a comparison that misses its target is reported as failed.

## Summary

| ID | Case | Reference | Result | Acceptance | Status |
|---|---|---|---|---|---|
| **STATE-NQ** | Annulus extension and Saint-Venant twist | closed form | 1.0e-15; 0 | <1e-10 | Pass |
| **STATE-P** | Closed-end thick tube, von Mises stress | Lamé solution | 1.9e-15 | <1e-10 | Pass |
| **STATE-T** | Uniform free thermal expansion | closed form | strain 2.0e-15; stress 3.1e-15 of E α ΔT | <1e-10; <1e-10 of E α ΔT | Pass |
| **A10, A10b** | Vector-slip rotation equivariance; vector and loxodromic contrast | identity | 3.8e-16; 1.2e-16 | <1e-12 | Pass |
| **OvM, OvM-g** | Frictionless contact slip and geodesic transverse map | Out and von Morgen (1997), Eqs. 26–28 | ≤2.9e-15; exact geodesic 2.6e-4 to 3.8e-4 | <1e-12; <1e-3 at k = 1e-4 | Pass |
| **Out89** | Contact plastic slip and damage ratios | Out (1989), Eqs. 1, A8, A11 | 9.6e-16; 5.0e-7 | <1e-12; <1e-4 | Pass |
| **CF** | Laid-core friction limits | closed forms, rotation equivariance | ≤1.0e-4 | per check | Pass |
| **CFW** | Laid-core contact statics (wedged trefoil, central filler) and laid filler profiles | closed forms, armour-only limit | ≤1.0e-4 | per check | Pass |
| **FR** | Fretting regime: Mindlin energy, Fouvry transition, Cattaneo strip | closed forms, numerical Masing loop | ≤7.8e-12 | <1e-9 to <1e-12 | Pass |
| **TQ** | Torque constant of a strand; coupled section; mirror lay | Feyrer, Example 2.8; Knapp (1981) | −0.11 %; −1.3e-6; 0 | ±0.2 %; <1e-4; <1e-9 | Pass |
| **TQ-SV** | Saint-Venant twist rigidity with free warping: rectangle; bonded annulus as a mirrored 60° sector | series solution; polar moment | 1.0e-4; −1.5e-8 | <3e-4; <1e-6 | Pass |
| **TQ-FT** | Twist-rate wire stress of the fatigue path: stress and damage, torque and tension carried by the wires, mirror lay, zero twist | Knapp (1981), Eqs. 2, 11; `K_et`, `K_tt` | 4.5e-15; 2.2e-16; 0; 0 | <1e-12; <1e-12; <1e-12; 0 | Pass |
| **A11** | Moment–curvature energy closure | loop area against closed form | 0.082 % | <0.1 % | Pass |
| **σ_B** | Radial-fibre bending stress: no friction stress (μ = 0), stuck wire, slipping wire; the normal curvature follows the slip state | Ménard and Cartraud (2023), Eq. 14 (μ = 0); `cos²α cos2α + ξ sin²α cos²α` (stuck); Eq. 14 + `(sin²α / r) σ_f / E` (slipping) | 0; 0; 0 | <1e-6 % each | Pass |
| **M-rec, M-rec-3core** | Assembled per-wire moment against the section engine | cross-engine | 0.000 % | <2 % | Pass |
| **MK-Tcyc** | Moment–curvature law with tension cycling inside the bending cycles; work balance under the changing friction bound | fatigue-path friction stress, same kernels | 3.4e-16; 8.9e-16 | <1e-12 | Pass |
| **MX-scale** | Exported bending law with a slip scaling variable: bound homogeneity, scaled group yield curvatures, tension cycle inside the bending cycles | `compute_mkappa` at another tension; contact-tension history | 5.5e-16; 1.2e-13; ratio 1.21 | <1e-12; <1e-9; ≤1.25 | Pass |
| **FL-P** | Pressure-dependent friction coefficient: spring-slider under pressure steps; compiled against NumPy kernel under a pressure history; work balance | closed form; NumPy reference | 3.0e-16; 8.7e-16; 2.5e-15 | <1e-9 | Pass |
| **slip-onset** | First wire to slip lies at the neutral axis (compression side, ψ ≈ 3λ/2) | Raoof; Papailiou | \|cos\| = 0.156 | \|cos\| < 0.2 | Pass |
| **xsec-K** | Coupling tangent bounds | EI_max, EI_min limits | ≤6.0e-13 % | <0.01 % | Pass |
| **QP-KKT, QP-Lamé** | Concentric contact complementarity; shrink-fit compliances | KKT conditions; compound cylinder | 1.1e-13 N; 1.3e-16 | <1e-8 N; <1e-6 | Pass |
| **COMP-T, COMP-N, COMP-O, SH-AX** | Component thermal expansion, axial equilibrium, zero-offset bounds, sheath axial stress | closed forms | 0 | <1e-6 Pa; <1e-12 | Pass |
| **io-known, io-global** | Reader to fatigue known answer; global time-domain history against its recorded run summary | closed-form Miner damage; recorded summary | 0 | <1e-10 | Pass |
| **σ_c1** | Bird-caging stress cross-check | Chae et al., Eq. 17 | 0 | <1e-9 % | Pass |
| **ST17-T3** | Lateral buckling lower bound, 6, 8 and 14 inch pipes | Sævik and Thorsen (2017), Table 3 | −2.03 / −0.45 / −1.34 % | ±3 % | Pass |
| **FOU95** | Mindlin transition amplitude, 200–500 N | Fouvry et al. (1995), Table 4 | 3.25 / 4.17 / 4.63 / 5.25 µm vs 3.2 / 4.1 / 4.6 / 5.2 | ±0.1 µm | Pass |
| **WEAR-CW** | Energy wear volume of crossed wires | Poon (2024), Cruzado (2013) | ratio 0.913 | 0.8–1.25 | Pass |
| **LV21** | Single-wire limit; first slip at three friction coefficients | Li and Vaz, Eqs. 11, 15 | ≤1.57 % | ±0.1 %; ±2 % | Pass |
| **ST17-P34** | Wire yield and hoop failure of the de Sousa pipe | Sævik and Thorsen (2017), Eqs. 33–34 | +0.04 / +4.43 / +3.76 % | ±0.5 %; ±6 % | Pass |
| **YANG15** | Tape failure load | Yang et al., Eq. 9 | −1.27 % | ±3 % | Pass |
| **ST17-SOUSA** | Radial bird-caging load against the de Sousa test | Sævik and Thorsen (2017), Eq. 35 | −1.35 % | ±10 % | Pass |
| **OST12** | Fixed-free plateaus per armour layer | Østergaard, Fig. 14 | +1.01 / −0.06 / −0.03 % | ±5 % | Pass |
| **TQ-KL** | Torque-balance helix angle, seven strand pairs | Kim and Lee (2017), beam FE | −0.26 to −0.06 deg | ≤0.3 deg | Pass |
| **TQ-UJ** | Fixed-end torque per tension, six strands | Utting and Jones (1987), Table 1 | theory 1.051–1.059; test 1.067–1.136 | 1.04–1.07; 1.00–1.15 | Pass |
| **BM1** | Full-slip bending stiffness EI_min, no calibration | Ménard and Cartraud (2023), Table 3 | 0.6 % (3571 N·m²) | <15 % | Pass |
| **BM1-loop** | ±0.10 m⁻¹ moment–curvature loop predicted with the clamping identified on the measured ±0.05 cycle work (190 kPa), four-point arrangement | Ménard and Cartraud (2023), Fig. 16 | ordered RMSE 5.45 %; cycle work 44.4 vs 62.3 J/m (−28.8 %) | RMSE <5 %; work within 10 % | Fail |
| **BM1-ID** | Laid cores, wedged trefoil and filler profiles: initial pressure identified on range (b), five ranges predicted | Ménard and Cartraud (2023), Fig. 16 | (a) +28.5 %, (c) +26.2 %, (d)–(f) within 16 % | work ±20 %, RMSE <10 % | Fail |
| **Z-EI** | Measured stick secant against the no-slip EI_max; slip slope against EI_min, 112 mm three-core cable, zero tension | Zeidan (2025), thesis Figs 5.5–5.6 | stick 141–240 kN·m² vs EI_max 311.1 kN·m² (plane-section 335); slip 8.65–9.70 (plotted curvature) vs EI_min 10.09 kN·m² with the bonded outer face (2.72 without) | stick ≤ EI_max; slip ≥ EI_min | **Fail** |
| **Z-ID** | Four-point loops at 30–60 mm in the test arrangement: pressure identified on 40 mm, three amplitudes predicted; outer armour face bonded (identified input, gap G8) | Zeidan (2025), Fig. 5.6 | P0 1085.9 kPa; work +9.9 / −4.7 / −9.0 %, RMSE 6.4 / 4.3 / 4.4 %, moment range −4.4 / −8.5 / −8.0 % (30 / 50 / 60 mm; 40 mm RMSE 4.5 %). Both faces sliding: P0 760 kPa, range −33 to −46 % | work ±20 %, RMSE <10 %, range ±10 % | Pass |
| **HU-3PB** | Conductor loop, third cycle, three-point arrangement | Hu et al. (2022) | loop area +36.7 % (Papailiou), +64.4 % (KH); transition share 32.8 % / 39.4 % | area ±15 %; share 34 ± 10 points | **Fail** |
| **HU-uniform** | Same loop at uniform curvature | Hu et al. (2022) | area +4.9 % (Papailiou) | — | Not comparable |
| **WITZ-EI** | Full-slip EI_min of a 2.5-inch flexible pipe, 0 bar, parameter-free | Witz (1996), measured gradients 1190 / 1014 N·m² | 1007 N·m² | 913–1309 N·m² | Pass |
| **WITZ-ID** | 300-bar loop predicted after identifying the initial pressure at 0 bar | Witz (1996), Figs 5–6 | width 1356 N·m vs 1990–2164; energy −12.8 % | width 1692–2489 N·m; energy ±15 % | **Fail** |
| **SV11-PLANE** | Inner-armour stress ranges at the bending-plane FBG stations, 14 load cases | Sævik (2011), Figs 17–23 | median ratio 2.17 (3 of 14 in band); sectional 1.19 (12 of 14) | median 0.85–1.18 and ≥12 of 14 in 0.70–1.43 | **Fail** |
| **SV11-GJ** | Absolute stress ranges near the termination, load case 14 | Gjøsteen et al. (2027), Figs 27–35 | mean ratio 2.07 (sectional 1.27) | 0.72–1.38 | **Fail** |
| **SV11-SPLIT** | Origin of the termination increment: end-fitting anchor alone (uniform curvature from the face); end fitting removed (anchor 20 m upstream) | Sævik (2011); Gjøsteen et al. (2027) | end fitting alone 1.23 / 1.36 (sectional 1.19 / 1.27); without end fitting unchanged (2.09, 2.07) | stated, not gated | Info |
| **SV11-REACH** | Small-angle slip demand and the cases outside the band whatever the correction does | Sævik (2011) | a_max below 1 at 0.5° and 1°; sectional outside: LC 4, 6 | stated, not gated | Info |
| **SV11-EQ88** | Bellmouth curvature law at the two published bellmouth ends (1.72 m; 1.79 m): radius and slope | Sævik (2011), Eq. 88 and text (3.9 m, 16°) | 3.90 m, 15.9°; 3.79 m, 17.0° | stated, not gated | Info |
| **AW-DONG25** | Armour wire anchored at both ends of one pitch at uniform curvature: stress at the anchor and peak, code-to-code | Dong et al. (2025), FE, Figs 14 and 25 | anchor 156.4 vs 156.9 MPa (source theory 147.8); peak 207.8 vs 204.5 MPa | stated, not gated | Info |
| **AW-ZT14** | Same at a = 4.92: normalised anchor and peak stress, code-to-code | Zhu and Tan (2014), FE, Figs 4–8 | 1.19 / 1.70 vs 2.22 (the FE far field is not in slip) | stated, not gated | Info |
| **DAI20-KAPPA** | Curvature at two stations inside a bellmouth against the rocking angle, steel-tube umbilical at 200 kN: contact angles predicted after identifying the bending stiffness on one curvature value | Dai, Sævik and Ye (2020), Fig. 9 | −0.87° (S2), −0.89° (S3); 59.1 kN m² identified | both within ±1.0° | Pass |
| **DAI20-COULOMB** | Sectional full-slip friction strain range of two helical steel tubes with the source's contact line forces | Dai, Sævik and Ye (2020), Tables 3–5 | radial +89 / −55 %; radial plus hoop +114 / +163 % | stated, not gated | Info |
| **DAI18-DIG** | Digitised friction loops of armour wire on anti-wear tape, Test 1: loop maximum against the printed static coefficient | Dai, Sævik and Ye (2018), Fig. 4a, p. 428 | 0.2237 / 0.2405 / 0.3786 vs 0.22 / 0.24 / 0.38 | within 0.005 + 0.0021 | Pass |
| **DAI18-RS** | Re-stick peak at every reversal, seven loops; spring-slider with `stick_time_s = 0` against the default `inf` | Dai, Sævik and Ye (2018), Figs 4a, 4c | measured 14 of 14 strokes; `0`: 14 of 14; `inf`: 0 of 14 | all; all; none | Pass |
| **DAI18-T1** | Test 1 loops (8, 4, 1 MPa) against the spring-slider with the printed coefficients | Dai, Sævik and Ye (2018), Fig. 4a | peak −1.8 to +1.0 %; level −3.3 to +12.3 %; work −1.9 to −2.6 %; no gradual decay after the peak | stated, not gated | Info |
| **DAI18-T3** | Test 3 loops (0.01–0.2 Hz) against the rate-independent spring-slider | Dai, Sævik and Ye (2018), Fig. 4c | dynamic level −3.5 to +0.3 %; static peak +10.6 % at 0.2 Hz not represented | stated, not gated | Info |
| **BM5** | Conductor temperature of the CIGRE TB 880 export cable, from the source's tabulated network inputs (`compute_thermal` on the cable itself is not yet comparable) | Poradowski tabulated thermal network; FEM 89.62 °C | 84.4 °C | ≤90 °C, about 5 K low | Partial |
| **BM4** | Axial stiffness of a population of umbilicals | Sævik and Bruaseth (2005) | mean ratio about 1.45 | <1.25 | Not validated |

**Pass** meets the registered target. **Fail** does not; the target is unchanged and the row
names the discrepancy. **Partial** reproduces a stated part of the comparison. **Not comparable**
records a comparison whose reference does not support the definition used. **Not validated**
records a negative finding that limits the scope of the model. **Info** is a comparison stated
without a target. The values above are rounded from the generated report, which holds
every digit.

## Scope

CableXSec is a local model: it takes a section load history (curvature components, tension,
pressure, temperature) from a global analysis and resolves the cross-section. The validated
quantities are those of the passing rows. In particular:

- The full-slip bending stiffness and the analytical limits of the slip model are confirmed
  against published data without calibration (BM1, WITZ-EI).
- On the Ménard and Cartraud (2023) loops one identified clamping does not fit the work of both
  the small and the large cycles: identified on the ±0.05 cycle, the ±0.10 cycle is 29 % low
  (BM1-loop); identified on it, the small cycles are up to 29 % high (BM1-ID). The source's
  0.3 MPa is a tuning parameter of its own FE model and is not used as an input of the gated
  rows. The 300-bar Witz loop is too narrow after
  identification at 0 bar. On the Zeidan (2025) cable the identified loops now meet the
  registered bands (Z-ID) with the outer armour face bonded; absolute dissipation predictions
  carry the uncertainty shown by WITZ-ID and Z-ID.
- **Gap G8 (slip-branch stiffness of compact sections): a bonded outer armour face.**
  - With both armour faces sliding, the Zeidan (2025) slip branch is about 3.5 times softer
    than measured (2.5–2.6 against 8.65–9.70 kN·m²) and the identified loops miss the moment
    range by a third or more.
  - The measured loops themselves show what is missing: their dissipation grows linearly with
    the amplitude (a friction moment mobilised early and then constant), while their range
    grows at 8–11 kN·m², an elastic, non-dissipative stiffness after slip.
  - Its source is an armour face that does not slip. Zeidan's sheath sits on the outer armour
    with no tape (thesis Table 5.2, Fig. 5.2); with `bonded_face: outer` (an identified input)
    the full-slip stiffness is 10.09 kN·m² and Z-ID passes. The rope-filler and taped
    constructions (Ménard and Cartraud 2023; Witz 1996) keep both faces sliding and need no
    change (BM1, WITZ-EI). Full-scale tests of dynamic cables with bitumen-coated armour
    report bending stiffness beyond the stick–slip law and attribute it to the bitumen and to
    bonded layers (doc/benchmarks.md); keying by a sheath extruded onto the wires is the
    construction identified here for Zeidan's cable.
  - Remaining: the bond is about 4–17 % stiffer than the measured slip slope in the plotted
    curvature, so the bound row Z-EI fails. The source's own optical deflection profiles
    (paper Fig. 18) put the central curvature at 0.78 of the plotted one, which would place
    the measured slope above EI_min; the registered row keeps the plotted reading.
  - The bonded face is used by the bending law and `compute_stiffness`. The wire-stress,
    fatigue and capacity routes refuse it until the bond's share of the wire force is in them.
  - The eight mechanisms ruled out earlier are recorded in doc/benchmarks.md ("Known
    limitation: slip-branch stiffness of compact sections").
- The termination-corrected (anchored-wire) armour stress is an upper estimate that the
  published near-termination measurements do not support (SV11-PLANE, SV11-GJ); the sectional
  result is the one closest to them (median 1.19, 12 of 14 in band, against the 1.18 bound). At
  this rig the end fitting alone moves the ratios by 3–7 %; the excess comes from the
  curvature gradients at the bellmouth contact on a one-way curvature field (SV11-SPLIT). At
  0.5° and 1° the pipe sticks and the correction is zero (SV11-REACH).
- The rig-beam curvature that drives a bellmouth calculation is checked against the one
  published curvature history inside a bellmouth (DAI20-KAPPA: contact angles within 1° with
  one identified stiffness, under declared rig assumptions); the sectional Coulomb range of
  helical umbilical tubes with radial contact forces is off by +89 / −55 % (DAI20-COULOMB),
  the source's own two-dimensional result, because hoop contacts are not modelled.
- The static/dynamic friction law is compared with measured friction loops of armour wire on
  anti-wear tape on one spring-slider, not on a cable section (DAI18 rows): the loops re-stick
  at every reversal (`stick_time_s = 0`), the dynamic level is rate independent at 0.01–0.2 Hz,
  and the law does not represent the gradual decay after the breakaway peak or the rise of the
  static peak with frequency.
- Cyclic redistribution of three-core contact is not solved; contact stress functions use
  prescribed tractions; fretting is excluded from the long-term fatigue projections.
- BM4 shows that the axial stiffness of umbilicals with radial interaction is not reproduced
  without that interaction.

## Tier 1: analytical and consistency checks

The rows are produced by `validation/scripts/run_validation.py`, and the tests marked
`analytical_sanity` and `independent_method` check the same quantities in the test suite:

```bash
python -m pytest -q -m analytical_sanity
python -m pytest -q -m independent_method
```

The independent references are the periodic friction and shear cells
(`validation/scripts/periodic_friction_cell.py`, `periodic_shear_cell.py`), the finite-difference
radial ring (`radial_foundation_fd.py`), the local Hertz reference
(`helical_hertz_reference.py`), the exact path-work bounds of an elastic–Coulomb slider
(`slider_initial_work.py`) and the NumPy reference of the compiled kernels.

## Tier 2: published measurements

```bash
python -m pytest -q -m full_scale
python validation/scripts/check_saevik2011.py
python validation/scripts/check_t1_test_arrangement.py
python validation/scripts/check_g8_gradual_reversal.py
python validation/scripts/check_dai2018_friction.py
```

The cases, their sources, the digitisation and the registered targets are described in the
benchmark register. The Ménard and Cartraud (2023) ranges were extracted from the vector paths of
the published figure (`validation/paper_data/menard_vector_extraction.json`); the other measured
curves were digitised from the published figures.

## Reproducing the report

```bash
python validation/scripts/run_validation.py          # rewrite validation/validation_report.md
python validation/scripts/run_validation.py --check  # compare with the committed report
```

The script exits with status 1 while any row is **Fail** or **Review**. The failed rows are the
registered comparisons listed above. BM1-loop compared the source's transferred 0.3 MPa in 0.1.0
(Fail) and until 2026-11-20; it now identifies the clamping on the ±0.05 cycle. BM1-loop-uniform
gave the same numbers at uniform curvature and is removed (see the changelog).
