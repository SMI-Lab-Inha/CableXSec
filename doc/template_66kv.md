<!-- SPDX-License-Identifier: Apache-2.0 -->

# Public 66 kV three-core wet-design template

`cablexsec/cables/iea_task49_66kv.yaml` (`cx.iea_task49_66kv_yaml_path()`) is a
public 66 kV three-core dynamic cable for rehearsing every analysis before
project data are available. It contains published values and marked
assumptions only. No project or industrial-partner data are used.

## 1. Source chosen

The template is the 66 kV variant of Janocha et al. (2024) [Janocha2024]:
three 630 mm² copper cores, XLPE insulation, copper screens, MDPE core
sheaths and fillers, two steel armour layers of 3.15 mm wires and an HDPE outer
sheath, 184 mm outer diameter. The IEA Wind Task 49 design basis (Hall et al.
2024, NREL/TP-5000-89709) [IEATask49] adopts this cable as its 630 mm²
reference (Table 13, row 630 mm², reference [32]) and scales it to other
conductor sizes.

| Source | 66 kV data | Use |
| --- | --- | --- |
| [Janocha2024] Sustainability 16, 2899 | Layer outer diameters and lay angles (Table 1), materials (Table 2), wire diameter and copper area (Section 2.1), model section (Fig. 4a), EA, GJ (Section 2.4), M–κ (Figs. 6, 12b), stress factors (Fig. 19), capacity (Section 3.3, Fig. 21) | Geometry, materials, comparison |
| [IEATask49] NREL/TP-5000-89709, Table 13 | D, m, EA, slipping EI at 50 kN, MBL, MBR, power and resistance of the 630 mm² cable; 25-year design life (Section 3); DFF 10 (Section 3.3.2) | Comparison, metadata |
| CIGRE TB 610 (2015), Section 5.4.3 | Wet design: water migration into the insulation is tolerated and a radial water barrier usually does not exist; lead sheaths are not used in dynamic cables | Wet-design declaration |
| ORE Catapult (2022) PN000486-RPT-013, Section 3 | Generic 66 kV dynamic cable case study selects a wet design with an oversheath and no metallic barrier | Wet-design declaration |
| Lozon et al. (2025) | Uses the 300 mm² row of [IEATask49] Table 13; no layer data | Not used |
| Beier et al. (2023, 2024) | Aggregate properties of a different 66 kV cable (Beier 2023 Table 2: D 0.116 m, EA 362 MN); no layer radii | Not used |
| Kwon et al. (2026) | 66 kV section shown only as a drawing (Fig. 8) with element counts and moduli; no dimension table | Not used |
| Rentschler et al. (2019), Thies et al. (2019) | Aggregate global-analysis properties (Thies 2019 Table 1: axial strength, MBR); no layer data | Not used |

Janocha 2024 is the only public 66 kV source with a complete layer table, a
material table and published section properties of the same cable.

## 2. Provenance of every input

Status: **P** published value; **D** derived from published values;
**A** assumption (not published).

| Item | Template value | Source | Status |
| --- | --- | --- | --- |
| Conductor | 3 × 630 mm² Cu, envelope 29.90 mm | Section 2.1; Table 1 | P |
| Conductor model | area-equivalent solid, r = 14.161 mm (exactly 630 mm²); stranding voids merged into the screen | Required by the radial solution (no strand radial law) | D |
| Conductor stranding (variant `iea_task49_66kv_stranded`) | 61 compacted wires 1 + 6 + 12 + 18 + 24, ring thickness 3.322 mm, 10.33 mm² each; lay ratio 12, equal-ratio or equal-pitch pattern, alternating, outer left hand | Wire count Fig. 4a; envelope Table 1; pitches designed (Section 4.5) | P / D / A |
| Conductor screen | copper tape, OD 33.90 mm, loose | Table 1; merged into the conductor body in Table 3 | P, loose: A |
| Insulation | XLPE, OD 57.70 mm (11.9 mm wall) | Table 1 | P |
| Insulation screen | copper, OD 64.00 mm, loose | Table 1; merged into the insulation in Table 3 | P, loose: A |
| Core sheath | MDPE, OD 70.40 mm | Table 1 | P |
| Core pitch-circle radius | 40.81 mm | Touching cores: 40.65 mm in the Table 1 section, 40.55 mm from Fig. 2b (111 mm); laid touching tubes at 10° need ≥ 40.80 mm | D |
| Core lay | 10° at the core-centre radius, pitch 1.4542 m, right hand | Table 1 (angle); radius of definition and hand | P / A |
| Filler | MDPE, OD 151.80 mm, three bodies as one interstitial annulus outside the cores | Table 1 | P, annulus: A |
| Bedding (inner) | PPY, OD 152.60 mm | Table 1 | P |
| Inner sheath | MDPE, OD 160.70 mm | Table 1 | P |
| Inner armour | 139 × 3.15 mm steel, r_helix 81.925 mm, 13°, left hand | Section 2.1 (d), Table 1 (OD 167.00 mm, angle), Fig. 4a (count), "opposite" lays (Section 2.1) | P / D (count) / A (hand) |
| Bedding (armour) | PPY, OD 167.40 mm | Table 1 | P |
| Outer armour | 145 × 3.15 mm steel, r_helix 85.275 mm, 10°, right hand | Section 2.1, Table 1 (OD 173.70 mm), Fig. 4a | P / D / A |
| Outer sheath | HDPE, OD 184.00 mm | Table 1 | P |
| Copper | E 112.2 GPa, ν 0.34, ρ 8890 kg/m³, μ 0.30 | Table 2 | P |
| Steel | E 200 GPa, ν 0.26, ρ 7800 kg/m³, μ 0.20, σ_y 375 MPa | Table 2; Section 2.1 | P |
| XLPE, MDPE, HDPE | E 1000 MPa, ν 0.40; ρ 925 / 956 / 980 kg/m³ | Table 2 | P |
| PPY | E 150 MPa, ν 0.40, ρ 895 kg/m³ | Table 2 (footnote says polypyrrole; the values are those of polypropylene yarn) | P |
| Steel UTS | 470 MPa (1.25 σ_y) | Enters only the thermal scaling of the yield check | A |
| Copper UTS | 220 MPa | [IEC60228] annealed Cu, as `copper_stranded` | P (other source) |
| Copper 0.2 % proof stress | 120 MPa | Section 2.1, p. 5: plastic limit of ETP copper (0.2 % plastic strain), after Robinson; the 250 MPa yield strength of the same sentence is not used | P |
| PE allowable strain | 7.7 % (MDPE, HDPE) | API 17J Table 6, p. 19 (PE sheaths of unbonded flexible pipe), transferred; no power-cable value is published (DNVGL-ST-0119 16.2.2, p. 149) | P (other source) |
| XLPE, PPY, copper tape screen limits | none | No public value; the results say "no limit given" | — |
| Copper bending modulus | 5 GPa effective | Same model assumption as `copper_stranded` | A |
| Thermal properties | λ, ρc_p, α of the generic library grades | [IEC60287], [Qiao2025] as cited in the library | P (other source) |
| Manufacturing clamping | `manufacturing_clamping` of both armour layers: zero, basis `not_published`; friction-dependent analyses warn (`MissingClampingStateWarning`) | Not stated by the source (no winding stress or extrusion pressure) | A |
| Wet design, flooded annulus | `water_ingress: {design: wet, armour_annulus: flooded}` | No barrier in Table 1; CIGRE TB 610 5.4.3; ORE Catapult 2022 | D |
| Armour S–N | `saevik_2011_flat_steel` (seawater, CP) | Flooded annulus; the source uses a DNV-RP-C203 curve (Fig. 27) of unstated class | A |
| Environment | 15 °C; T4 = 0.343 K·m/W | IEC 60287-2-1 4.2.2 with assumed 1.0 m burial and 0.7 K·m/W seabed | A |
| Electrical | ε_r 2.5, tan δ 0.001; Mazzanti life constants | [IEC60287-1-1] Table 3; [Mazzanti2007] Table I, as other templates | P (other source) |
| Design life, DFF, MBR | 25 years; 10; 2.76 m | [IEATask49] Section 3, Section 3.3.2, Table 13 | P |
| Minimum breaking load | `aggregate_properties.MBL_kN` 659 kN with `MBL_basis` approximate; shown beside the tension capacity of the Capacity envelope, never a limit | [IEATask49] Table 13, p. 35 (the source gives MBL and MBR as approximate values) | P (comparison) |

The armour wire counts were obtained by counting the 284 wire sections of the
66 kV model in Janocha Fig. 4a (139 inner, 145 outer; every section has the
same area, so no sections were merged). The image scale reproduces the 3.15 mm
wire diameter.

## 3. Wet design in the schema

The optional `water_ingress` block declares `design` (`wet` or `dry`) and
`armour_annulus` (`flooded` or `sealed`) with a `source`. Loading checks it
against the layers: a wet design may not contain a radial metallic water
barrier (a `cylindrical_shell`, or a `lead_sheath` or `aluminium_corrugated`
layer) and a dry design must contain one. The declaration does not change any
analysis default. For a flooded annulus, use `p_ext_Pa=0` (no hydrostatic
clamping) and a seawater S–N curve, as in [local_analysis_guide.md](local_analysis_guide.md)
Section 7.2; `contact_model="solved"` accepts `flooded_pressure_Pa` through
`solve_cable_radial_state`.

The MDPE filler made one further change necessary: an interstitial filler is
now recognised by geometry (non-structural annulus from the axis of a
multi-core cable) as well as by the `pp_filler` material, so its axial
stiffness and volume exclude the core envelopes for any material.

## 4. Analyses and comparison with the published properties

Run with `PYTHONPATH` at the repository root, one thread. Figure values of the
source are read from the plots (about ±5 % of full scale).

```python
import numpy as np
import cablexsec as cx

cable = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
s = cx.compute_stiffness(cable)
loop = cx.compute_mkappa(cable, 0.2 * np.sin(np.linspace(0, 4 * np.pi, 801)),
                         tension_N=50e3, contact_model="solved")
cap = cx.compute_capacity(cable, np.linspace(0.0, 1.0e6, 11))
th = cx.compute_thermal(cable, current_A=1093.0, conductor_area_m2=630e-6,
                        conductor_material="copper_janocha2024")
```

| Quantity | CableXSec | Published | Comment |
| --- | --- | --- | --- |
| EA (fixed ends) | 636.8 MN (free ends 629.8) | 658 MN, Janocha 2.4 / IEA Table 13 | −3.2 %; armour 416, copper 203, polymers 18 MN |
| GJ | 131.3 kN·m² | 152 kN·m² at 50 kN, Janocha 2.4 | −14 %; CableXSec GJ is the extension-only lay term, without polymer and core torsion |
| Full-slip EI | 21.0 kN·m², with or without core friction | 42.47 kN·m² slipping stiffness at 50 kN, IEA Table 13; branch slope about 29 kN·m² in Fig. 6 and about 41 kN·m² in Fig. 12b | −51 % (−28 % against Fig. 6); outer and inner sheath give 11.6 and 6.1 kN·m². Friction cannot raise the full-slip value (Section 4.1) |
| No-slip EI (plane section) | 1653 kN·m²; bending-law slope 853 kN·m² armour only, 981 kN·m² with core friction (core ξ = 0.73) | of order 1500–2000 kN·m², initial slope in the Fig. 12b inset (10 kN) | Read from the inset |
| M at 0.2 1/m, 50 kN loop | 4.31 / 4.28 kN·m (bounded / solved); 4.39 / 4.38 kN·m with core friction | about 6.8 kN·m, Fig. 6 | −35 % with core friction |
| Loop half-height at κ = 0 | 0.11 / 0.08 kN·m (bounded / solved); 0.20 / 0.19 kN·m with core friction | about 1 kN·m, Fig. 6 | Core friction (μ 0.46) adds 0.08–0.10 kN·m; still about 5 times low |
| Moment at slip onset, 10/100/200 kN | 0.23 / 0.44 / 0.67 kN·m at 0.01 1/m; 0.25 / 0.60 / 0.99 kN·m with core friction (bounded) | about 5.5 / 9.5 / 12 kN·m, Fig. 12b | Same cause; 12 to 22 times low |
| Armour axial stress at 1000 kN | 298 / 305 MPa (inner / outer) | about 355 MPa von Mises, Fig. 19b | Axial part only |
| Conductor axial stress at 1000 kN | 176 MPa | about 190 MPa, Fig. 19b | −7 % |
| Armour yield, pure bending | 1.16 1/m | about 1 1/m, Section 3.3 | |
| Armour yield, pure tension | about 1.1 MN | about 1000 kN, Section 3.3 | |
| Copper 120 MPa plastic limit, pure tension | 702 kN (characteristic envelope, factors 1.0) | about 600 kN (Fig. 21); MBL 659 kN (IEA Table 13) | +6.6 % against the MBL; governs the envelope (Section 4.6) |
| Copper 120 MPa plastic limit, pure bending | 0.362 1/m, bend radius 2.76 m, bent towards a core; 0.353 1/m, 2.84 m in the worst bending direction, 30° from a core (strand basis of the 5 GPa bending modulus) | 0.5 1/m (Section 3.3); MBR 2.76 m (IEA Table 13) | Set by the assumed bending modulus (Section 4.6) |
| Conductor temperature at 1093 A (125 MW) | 88.9 °C (armour 61.0 °C) | 125 MW rating, IEA Table 13 (90 °C XLPE basis) | Assumed T4; no three-core factor or λ2 |
| Mass | 76.3 kg/m; 52.5 kg/m without the screens | 55.76 kg/m, IEA Table 13 | Screens are full copper annuli in Table 1 |

Fatigue on a synthetic 2000 s history (κ_x = 0.004 sin(2πt/10) + noise,
κ_y = 0.002 sin(2πt/13 + 0.4), T = 50 ± 5 kN), maximum wire damage (inner
armour governs):

| Friction model | `contact_model` default (bounded) | `contact_model="solved"` |
| --- | --- | --- |
| `direction_tied` (default) | 1.31e-9 | 7.88e-10 |
| `distributed` | 1.30e-9 | 7.77e-10 |

The two friction laws agree within 1 % on this smooth history. Core
friction (Section 4.1) does not change these damages: the fatigue path is
curvature driven, and the armour wire stresses do not depend on the cores. At 50 kN the
solved radial contact gives lower friction-setting armour pressures than the
bounded chain (P1+P2 = 0.031 and 0.008 MPa against 0.043 and 0.012 MPa, inner
and outer) and about 40 % less damage. Run times on one thread: 2.2 and 2.7 s (direction-tied),
36 and 46 s (distributed).

### 4.1 Laid-core friction

`compute_mkappa(..., core_friction=...)` adds the stick–slip law of the laid
cores against their neighbours, the filler and the inner sheath, and of
resolved conductor wire rings (`cablexsec/mechanics/core_friction.py`;
[local_analysis_guide.md](local_analysis_guide.md) Section 6.1). Targets were
registered before the runs (`validation/scripts/check_core_friction.py`): Fig. 6 loop
half-height at κ = 0 (about 1.0 kN·m) and Fig. 12b knee moments within 30 %
(figure reading), slipping slopes (IEA Table 13, 42.47 kN·m²; Fig. 6, about
29 kN·m²; Fig. 12b, about 41 kN·m²) within 15 %. Inputs: Table 2 friction
coefficient of MDPE (0.46) on every core contact, rigid contact penalty, shear
of the core's XLPE and MDPE layers in the stick coefficient. The source
conductor is 61 resolved wires (Fig. 4a); their pitches are not published, so
the template keeps the area-equivalent solid and no internal ring slip. A
designed stranding under stated assumptions is available for conductor
fatigue (`iea_task49_66kv_stranded`, Section 4.5).

| Quantity at 50 kN | Armour only (bounded / solved) | With core friction (bounded / solved) | Published | Status |
| --- | --- | --- | --- | --- |
| Core normal load per core | — | 9.4 / 12.2 kN/m | not published | — |
| Loop half-height at κ = 0 | 0.114 / 0.083 kN·m | 0.195 / 0.187 kN·m | about 1.0 kN·m (Fig. 6) | FAIL |
| M at 0.2 1/m | 4.31 / 4.28 kN·m | 4.39 / 4.38 kN·m | about 6.8 kN·m (Fig. 6) | FAIL |
| Branch slope, 0.05–0.15 1/m | 21.0 kN·m² | 21.0 kN·m² | about 29 (Fig. 6); 42.47 (IEA Table 13) kN·m² | FAIL |
| M at 0.01 1/m, 10 / 100 / 200 kN (solved) | 0.23 / 0.37 / 0.54 kN·m | 0.25 / 0.58 / 0.96 kN·m | about 5.8 / 9.5 / 12.5 kN·m (Fig. 12b) | FAIL |
| Slope 0.1–0.4 1/m, 10 / 100 / 200 kN | 21.0 kN·m² | 21.0 kN·m² | about 41 / 41 / 44 kN·m² (Fig. 12b) | FAIL |

What the comparison shows:

- The core friction moment is linear in the core contact load. A 1.0 kN·m
  half-height needs about 107 kN/m per core, against 12.2 kN/m from the
  solved tension-induced contact at 50 kN. The source does not state an
  initial contact state; its Fig. 12b knee at 10 kN (5.8 kN·m) exceeds its
  own Fig. 6 half-height at 50 kN (about 1 kN·m), so the two figures cannot
  both follow from tension-induced Coulomb contact alone.
- Friction cannot raise the full-slip stiffness. The slope gap (21.0 against
  29–42 kN·m²) lies in the component bending stiffness: with the conductor
  bending as solid copper (112.2 GPa instead of the 5 GPa assumed), the cores
  would add 9.6 kN·m² (30.6 kN·m² in total), close to the Fig. 6 slope; the
  IEA value would still be 39 % above.
- A synthetic 61-wire conductor (lay ratio 12, Table 2 copper friction 0.30)
  raises the half-height by a further 0.03 kN·m.

### 4.2 Energised cable: temperature history into clamping and fatigue

The energised analysis (about 400 s on one thread) couples the
layer temperatures of the transient IEC ladder into the solved radial
contact state, the stick stiffness and the fatigue path
([local_analysis_guide.md](local_analysis_guide.md) Section 7.3). Inputs: the
template's environment (15 °C, `T4` 0.343 K·m/W, both assumed), rated
current 1093 A ([IEATask49] Table 13), `R_ac(T_c)` solved with the
temperatures; the fast record is the synthetic one above (2000 s, 200
bending cycles of 10 s, 50 ± 5 kN; noise seed 2024), contact model
`"solved"`, stress-free reference `T_ref` = 20 °C, set explicitly (5 K above
the ambient; Section 4.2.1). The Janocha grades have no registered
`E(T)` law, so the default variant is **expansion only** (template moduli
constant). The variant **with softening** transfers the [Qiao2025] relative
laws (XLPE law to the XLPE, HDPE law to the MDPE and HDPE; PPY unchanged): an
assumption, not a property of the source cable.

Steady load levels (inner armour; damage is the largest wire damage of the
record, the inner armour governs; bulk S–N slope m = 4). The template has no
manufacturing clamping, so every factor in this section is an upper-bound
sensitivity, not a cable property (Section 4.2.1). The thermal network takes
an armour layer's temperature at its own node (the collapsed radial gap); an
earlier version interpolated it into the next annulus. At
rated current the inner-armour clamping is 293 kPa instead of 294 kPa and the
thermal wire stress +84.3 MPa instead of +84.2 MPa (about 0.4 %); the damage
factors below were computed before the correction.

| Load | `T_c` / armour [°C] | P1+P2 inner / outer [kPa] | `k_f` / reference | σ_T inner [MPa] | Damage per bending cycle | vs 0 % |
| --- | --- | --- | --- | --- | --- | --- |
| reference, 20 °C (unenergised) | 20 / 20 | 31.1 / 8.2 | 1 | 0 | 3.67e-12 | 0.38 |
| 0 % (0 A) | 15.0 / 15.0 | 51.8 / 38.9 | 1 | −6.1 | 9.74e-12 | 1 |
| 50 % (546.5 A) | 30.1 / 24.4 | 71.3 / 18.2 | 1 | +14.8 | 1.72e-11 | 1.77 |
| 100 % (1093 A) | 88.5 / 60.7 | 294 / 108 | 1 | +84.2 | 1.12e-9 | 115 |
| 100 %, with softening | 88.5 / 60.7 | 222 / 82 | 0.49 / 0.58 | +63.4 | 4.74e-10 | 43.6 |
| 100 %, softening only | 88.5 / 60.7 | 32.6 / 8.7 | 0.49 / 0.58 | +0.5 | 4.11e-12 | 1.13 |

Daily cycle `I = 0.5 I_r (1 − cos 2πτ/24 h)` in its periodic regime (network
time constant 7.3 h): conductor 24.6–56.1 °C, inner armour 21.3–39.8 °C, inner
P1+P2 52–172 kPa (expansion only). Block-weighted damage per bending cycle
7.07e-11 (expansion) and 5.63e-11 (with softening); against the fixed
time-mean temperature the ratio is 1.59 and 1.36, against the 0 %
state 7.3 and 5.2 (upper-bound sensitivities, as above). The daily range of the thermal wire stress is 40 MPa
(inner) and 31 MPa (outer); that one slow cycle per day adds 2.4e-9, against
6.1e-7 per day from bending at this sea state. Diagnostics: Π1 = 3.8e-4;
Π3 = 14–48 in every block (the armour slips in every block, so the damage
follows the friction cap); frictional heating of the record at most
0.79 W/m, Π2 = 7.1e-3 at the peak-load block, and an armour temperature rise
of at most 0.29 K through the network (dissipation of the distributed friction
law; the direction-tied cap, which under-counts the work of the rotating
curvature of this record, gives 0.58 W/m, 5.2e-3 and 0.21 K;
[local_analysis_guide.md](local_analysis_guide.md) Section 9), so the
one-way coupling holds for this sea state (Π2 grows without bound as the current tends to zero, which
is why the temperature rise is the quantity to compare with the
measurement resolution).

Energised rig (`simulate_rig_test(..., thermal=...)`, 50 kN, uniform
bending of a 6 m specimen, gauges at mid-span, cold = 0 A at 15 °C, hot =
steady 1093 A): the armour stress range at the same bending rises with
temperature because the cap rises. These ranges are upper bounds of the
zero-clamping template, or hypotheses for a heated rig test, not predictions.

| Gauge (wire on the bending plane) | κ amplitude [1/m] | Range cold → hot, sectional [MPa] | Range cold → hot, anchored clamps [MPa] |
| --- | --- | --- | --- |
| inner armour, 0° | 0.005 | 8.4 → 34.9 (+26.6) | 12.9 → 34.9 (+22.0) |
| inner armour, 0° | 0.02 | 16.4 → 43.0 (+26.6) | 43.5 → 56.8 (+13.3) |
| outer armour, 0° | 0.005 | 8.6 → 18.8 (+10.2) | 7.5 → 18.0 (+10.5) |
| outer armour, 0° | 0.02 | 17.2 → 27.4 (+10.2) | 12.0 → 22.5 (+10.5) |

The mean wire stress shifts by +90 MPa (inner) and +67 MPa (outer) between
cold and hot at fixed tension; a temperature-compensated gauge does not
remove this mechanical strain. Wires on the neutral axis change by less than
0.6 MPa.

#### 4.2.1 Mechanism, reference temperature and sensitivity

An independent check of the factor of 115 found the number
correct within the model but not a property of the cable. The template has no
manufacturing clamping (none is published; it is declared `not_published`,
so friction-dependent analyses warn), and at 50 kN the armour clamping is only
31 and 8 kPa (inner and outer).

**Mechanism: axial thrust, then wire tension, then clamping.** The heated core
and polymers expand axially (the 1 GPa template polymers at 60–81 °C, and the
copper); at fixed tension the steel armour takes the thrust as extra wire
tension (+84 MPa in the inner armour), and the helix turns that tension into
radial clamping. Helix equilibrium of the solved wire stresses gives 238 of
the 294 kPa; the outer-sheath term is 26 kPa, and a Lamé estimate of the
radial polymer growth about 58 kPa. The rise is therefore axial, not a radial
polymer–steel expansion mismatch. A hand rule of mixtures at fixed tension
gives +66 MPa with uniaxial and +169 MPa with confined polymers; the computed
+84 MPa lies between, and about 31–42 MPa of it is copper–steel mismatch. The
direction agrees with constant-modulus models (Li et al. 2027; Fang et al.
2026). Π4 = 8.5 and 12.1 at rated current.

**Decomposition of the factor of 115** (inner armour, m = 4):

- clamping pressure alone: 108 (a uniform ×5.68 pressure scaling gives 117);
  the friction stress range rises ×4.6, the bending range does not change;
- stick stiffness: ×1.00;
- thermal wire stress: ×1.5, because it varies with tension within a block;
- mean shift (+90 MPa): no effect, as the reported run applies no
  mean-stress correction (a Goodman correction, available as
  `mean_stress="goodman"`, would add ×1.4–2.4; not
  rerun here).

**Reference temperature (the 15 °C / 20 °C offset).** The reported run sets
the stress-free reference to 20 °C, 5 K above the 15 °C ambient. The 0 A
state therefore sits 5 K below the stress-free state: the outer sheath
shrinks onto the armour and lifts the cold inner-armour clamping from 31 to
52 kPa, which is why the 0 % row differs from the 20 °C reference row.
`ThermalHistory.steady` and `from_current` now default `T_ref` to the
ambient (the cable stress-free as installed), so 0 A is the unenergised
state; the tables above pin `T_ref` at 20 °C.

**Sensitivity** (same run, one input changed; rated-current damage per
bending cycle over the cold state):

| Variant | Factor |
| --- | --- |
| As reported: `T_ref` 20 °C, zero clamping, constant moduli, m = 4 | 115 |
| Transferred [Qiao2025] softening laws (assumption) | 44 |
| Stress-free at the 15 °C ambient (the current default) | 310 |
| Seawater section, `T4` = 0.02 K·m/W (`T_c` 44 °C, armour 20 °C) | 213 |
| Armour winding stress 40 / 100 MPa | 20 / 1.9 |
| Polymer moduli × 0.2 / 0.05 / 0.01 (relaxation proxy, `T_ref` 15 °C) | 79 / 24 / 7 |
| Bulk S–N slope m = 3 / 4 / 5 / 6 | 35 / 115 / 384 / 1302 |

The cold baseline sets the ratio: manufacturing clamping raises the cold
damage 45–1190 times. Hot polyethylene and XLPE relax within hours to days,
which removes most of the polymer share; about 7 (the copper–steel part)
remains. Qiao et al. (2025) find that expansion and `E(T)` nearly cancel in
the bending hysteresis, and Guo et al. (2026) report bending stiffness
falling with current. The code now reports these dependencies: an explicit
`T_ref` with its basis, `polymer_relaxation_factor` with the ratio at each
bound (`relaxation_damage_ratio`), the S–N slope with every printed ratio
(`ThermalFatigueResult.summary()`) and warnings for zero clamping and for
polymers without `E(T)` above 40 °C.

Summary:

> In the public 66 kV template, which has no manufacturing clamping and
> time-independent polymer moduli, rated current raises the inner-armour
> clamping about six-fold at fixed tension. The core's restrained axial
> thermal expansion raises the armour wire tension. The computed bending
> damage per cycle rises by two orders of magnitude (115x; 44x with
> transferred softening laws). Because the friction stress range follows the
> clamping, the factor depends on the cold baseline. It falls to about 20x
> with 40 MPa armour winding stress, to about 2x with 100 MPa, and to 7–24x if
> polymer stresses relax. It is an upper-bound sensitivity. Constant-modulus
> models (Li et al. 2027; Fang et al. 2026) give the same direction;
> heated-cable tests (Guo et al. 2026) show softening.

The factors 115 and 44, the daily 7.3 and the rig ranges (8.4 → 34.9 MPa) are
upper-bound sensitivities of this zero-clamping template, or hypotheses for a
heated rig test; they are not cable properties.

### 4.3 Torque balance and twist under tension

```python
r = cx.torque_balance(cable, (0.0, 200e3), n_tensions=5)
d = cx.design_torque_balance(cable, "outer_lay_angle", tension_N=100e3)
```

With the assumed lay directions (cores and outer armour right-hand, inner
armour left-hand), the end torque with the twist restrained is 1.503 mm times
the tension (150 N·m at 100 kN; fixed-radius matrix 1.510 mm), and a free end
twists by −0.0533 °/m per 100 kN (5.3° over 100 m). The layer torques at
100 kN are −604 N·m (inner armour), +525 N·m (outer armour) and +229 N·m
(laid cores); the armour balance ratio is 0.869. A free end raises the inner
armour stress by 3.6 MPa and lowers the outer by 2.4 MPa at 100 kN (fixed
twist 30.2 and 31.4 MPa). The solved torsional rigidity at fixed extension is
162.8 kN·m² against the published 152 kN·m² at 50 kN (Janocha Section 2.4;
fixed-radius lay term 131.3 kN·m²), because the solved section includes the
polymer torsion.

The outer lay angle that zeroes the torque at 100 kN is 6.93° (template
10°; fill 0.859), or 103 outer wires (145), or 2.66 mm outer wires
(3.15 mm). With a left-hand core lay the cable torque changes sign
(−3.08 mm) and the balancing angle is 18.1°; with aligned cores (the armour
alone) it is 11.76°. The core lay direction is not published, so these
results show the size of the core contribution, not a property of the
reference design. `examples/torque_balance_templates.py` prints them with
the other packaged cables.

### 4.4 Screens and sheath: core slip state and strain life

The copper screens of Table 1 are full annuli, not wires, so the per-core
wire-screen assessment (`compute_core_screen_fatigue`, gap P1) does not apply;
the wet design has no metallic sheath, so the strain-life sheath route
(`compute_shell_fatigue(method="strain_life")`, gap P6) does not apply either.
The core slip state (gap P8) applies to the screen annulus: at the dynamic MBR
of 2.76 m its strain amplitude is 1.12% for free cores, 1.12% for the
frictional law (the cores slip: no manufacturing clamping is declared, at 0
and 50 kN) and 2.44% for stuck cores, largest at the tangential fibre of a
core on the neutral axis (free) or at the outer fibre of a core on the
extreme fibre (stuck); `cx.compute_core_layer_strain(cable, "cu_wire_screen",
kx, core_state=...)`. A wire or tape representation of the screens is a
geometry input that Table 1 does not give.

### 4.5 Designed conductor stranding (`iea_task49_66kv_stranded`)

The source gives the conductor's wire count (61, Fig. 4a), envelope (29.90 mm,
Table 1) and area (630 mm², Section 2.1), not its pitches. The variant
`cx.iea_task49_66kv_stranded(lay_pattern=..., outer_lay_ratio=..., outer_lay=...)`
returns the template with a designed, compacted conductor for
`compute_conductor_fatigue`; the default template keeps the area-equivalent
solid, which the radial solution needs.

- **Rings.** 1 + 6 + 12 + 18 + 24 wires compacted to the envelope: radial
  thickness `t = 29.90/9 = 3.322` mm, helix radii `i t`, and `630/61 = 10.33`
  mm² of copper per wire (`wire_area_m2`, the compacted area; fill 0.90 of the
  envelope). The envelope lies within the 28.7–32.5 mm of a compacted 630 mm²
  Class 2 conductor (IEC 60228:2004, Annex C, as reproduced by a manufacturer's
  catalogue; the standard itself is not held).
- **Lay ratios** (pitch over the ring's outer diameter) of the public copper
  conductors:

  | Source | Conductor | Lay ratio, inner to outer ring | Pattern |
  | --- | --- | --- | --- |
  | Nasution et al. (2014a) | 300 mm², 37 wires | 27.5, 17.8, 13.1 | pitches 265–295 mm |
  | Nasution et al. (2013) | 95 mm², 19 wires | 29.3, 17.6 | equal pitch 220 mm |
  | Nam et al. (2024) | 265 mm², 61 wires | about 32, 19, 14, 10 | about equal pitch 225 mm |
  | Jiang et al. (2022) | 120 mm² compacted, 19 wires | 18.2, 13.0 | "lay ratio 16" stated |
  | Smith et al. (2026) | 150 mm² compacted, dynamic cable | 12.3, 10.5, 12.2 | about equal ratio |

  Two practices appear: about equal pitch in every ring (long inner lays) and
  about equal lay ratio. The variant offers both (`"equal_ratio"`, default, and
  `"equal_pitch"`), with the outer lay ratio 12 (the published outer values are
  10–16). The lay pattern is a first-order input: for the same outer lay, the
  equal-pitch conductor gives about four times the bulk damage of ring 1 under
  the same bending (lay angles 3.3–13.1° against 9.9–13.1°), because the long
  inner lays carry more friction strain.
- **Directions** alternate; the outer ring is left hand (assumed; IEC 60228
  does not prescribe it).
- **Wire curve:** `nasution_2014_copper`, the wire curve of a compacted 300 mm²
  conductor, the nearest public compacted conductor wire; the SWT mean-stress
  correction is the default (gap CF2).
- **Not represented:** the compacted wire's own bending inertia (taken as that
  of a round wire of the ring thickness), the stress concentration of the
  compaction indentations (1.02–1.20, Jiang et al. 2022), and the conductor
  screen's contact with the outer ring.

### 4.6 Component limits and the capacity

Every component is checked against the limits its material gives
(`cablexsec/analysis/component_limits.py`): the conductor against its 0.2 %
proof stress (120 MPa, Janocha Section 2.1, p. 5) at the extreme wire fibre
and against its tensile strength on the net section; polymers against an
allowable strain (7.7 % for the PE layers, API 17J Table 6, p. 19); metallic
screens and sheaths against their yield stress as a screening check (their
first yield does not bound the capacity; an allowable strain does where the
material gives one). The XLPE insulation, the PPY beddings and the copper tape
screen have no public limit: the results say "no limit given".

The conductor is a solid with the 5 GPa effective bending modulus (an
assumption, Section 2). Its bending stress about its own axis is taken on the
basis of that stiffness: the slipping wires of radius R sqrt(E_b/E) that
give it, 0.211 times the plane-section stress of solid copper. At 80 kN and
0.1 1/m the conductor then takes 46 MPa in the frictional core slip state
(utilisation 0.39 against 120 MPa) instead of the 170 MPa of solid copper;
held to the cable axis (stuck) its core axis takes 457 MPa on the net section
(utilisation 4.07). The net-section stress at 80 kN is 13.67 MPa at zero
curvature (the tension share, E cos^2 eps at the 636.8 MN fixed-end EA) and
15.43 MPa at 0.1 1/m towards the first core, where friction holds the core
axis (the stick coefficient 0.733 and the core contact load set it).

The capacity envelope includes these limits with the component factor of the
lifecycle phase (OWA MecLim II guideline (2025) Table 5, p. 19: 0.80 normal,
1.00 installation and abnormal) beside the armour factor (DNV-RP-F401 Section
3.8.1, Table 3-1, p. 13: 0.67, 0.78, 1.00). The conductor's proof stress
governs at every tension:

| Quantity | Armour only | With the component limits | Published |
| --- | --- | --- | --- |
| Tension capacity at zero curvature, factors 1.0 | about 1.23 MN | 702 kN | MBL 659 kN (IEA Table 13) |
| Tension capacity, normal operation | 825 kN | 562 kN | — |
| Bend radius at zero tension, factors 1.0 | 0.86 m | 2.84 m (2.76 m bent towards a core) | MBR 2.76 m (IEA Table 13) |
| Bend radius at zero tension, normal operation | 1.28 m | 3.55 m | — |

The envelope holds for every bending direction: the component limits are
taken at their largest over bending towards a core, midway between two
cores and halfway between these (`cablexsec.bending_directions`), and on
this cable the largest lies 30° from a core at low tension. Bent towards a
core the conductor reaches its proof stress at 2.76 m, the published
minimum bend radius. That agreement rests on the assumed
5 GPa bending modulus; Janocha (Section 3.3) reaches the copper plastic limit
at 0.5 1/m in pure bending with resolved wires. The loose copper screens lie
outside the load path: their kinematic stress (384 MPa in the wire screen at
80 kN and 0.1 1/m) is flagged against their yield stress in Component
stresses and left out of the envelope.

### 4.7 Derived 300 mm² section (`iea_task49_66kv_300`)

Derived for analyses of the 300 mm² row of [IEATask49] Table 13
(D 0.161 m, m 36.66 kg/m, EA 469.6 MN, slipping EI 19.92 kN·m² at 50 kN, MBL
470 kN, MBR 2.41 m), the cable of Lozon et al. (2025) and of the CableDyn
lazy-wave reference. Table 13 gives no layer data for this row, so the section
is derived from this template (`cx.iea_task49_66kv_300_yaml_path()`; every
step is an assumption, listed in the file header):

- conductor: 300 mm² area-equivalent solid; stranded envelope 20.63 mm with
  the 630 mm² fill ratio;
- radial thicknesses of screens, XLPE (11.90 mm, set by the voltage class),
  core sheath, beddings, inner and outer sheath and the 3.15 mm armour wires
  kept from Table 1;
- core pitch-circle radius scaled with the core radius, filler in the 630 mm²
  ratio to the core envelope;
- armour wire counts from the 630 mm² coverage: 122 and 128 wires at 13° and
  10°;
- materials, lay directions, wet design, zero clamping, thermal boundary and
  S–N curves as this template.

| Property | Derived section | [IEATask49] Table 13, 300 mm² | Difference | 630 mm² template against its row |
| --- | --- | --- | --- | --- |
| Outer diameter | 164.0 mm | 161 mm | +1.9 % | 0 % |
| EA (fixed ends) | 477.5 MN | 469.6 MN | +1.7 % | −3.2 % |
| Full-slip EI at 50 kN | 14.0 kN·m² | 19.92 kN·m² | −30 % | −51 % |
| Dry mass | 56.9 kg/m | 36.66 kg/m | +55 % | +37 % |

The diameter and axial stiffness follow the table closely; the full-slip EI
and the mass carry the template's known differences (Section 5, items 1 and
8). Tests: `tests/test_template_66kv_300.py`.

## 5. Gaps found

1. **Friction outside the armour.** Closed in part by gap R1b (Section 4.1):
   the cores and resolved conductor rings now have an optional stick–slip
   law. With the tension-induced contact it adds 0.08–0.10 kN·m to the
   half-height at 50 kN; slip-onset moments remain 12 to 22 times lower than
   Fig. 12b, and the full-slip EI is unchanged. The published loops need
   either an initial core contact state about nine times the tension-induced
   one or a stiffer component bending than the template assumes; neither is
   published.
2. **Stranded conductor in the radial solution.** `contact_model="solved"`
   rejects a conductor whose metal area is below its envelope. The template
   uses the area-equivalent solid, so conductor component stresses are
   solid outer-fibre values (166 MPa at 0.1 1/m, 50 kN) rather than wire
   stresses (about 22 MPa, Fig. 19a). Wire count and pitches of the conductor
   are not published; the designed stranding of Section 4.5 resolves them
   under stated assumptions for conductor fatigue, while the radial solvers
   still need the area-equivalent solid.
3. **Capacity mechanisms.** The envelope has no copper plastic-limit or
   conductor bending mechanism; the source capacity (0.5 1/m, about 600 kN at
   100 % utilisation, Fig. 21) is governed by copper. In compression the
   armour screen ([uls_guide.md](uls_guide.md), Section 6) gives a cable
   capacity of 6.9 kN, set by lateral buckling of the 10° outer armour once
   the wires slip (4.6 kN at the DNV-RP-F401 normal-operation factor 0.67);
   the lay angles lie below the 20–38° range of the reproduced cases, so this
   is a lower bound.
4. **Torsion.** The fixed-radius GJ of `compute_stiffness` omits
   polymer and core torsion; the solved section of `torque_balance`
   includes them (162.8 against 152 kN·m², Section 4.3). The core lay
   direction, which decides the sign of the cable torque, is
   not published.
5. **Thermal.** Three-core geometric factor and armour loss λ2 are not
   implemented; T4 and
   ambient are assumptions; the filler is the equal-filler-area annulus. The
   template's polymer grades have no temperature-dependent modulus, so the
   energised results of Section 4.2 either keep the moduli or transfer the
   [Qiao2025] laws as an assumption; the manufacturing clamping, which sets
   how strongly heating changes the clamping, is not published (declared
   `not_published`, with a warning; Section 4.2.1).
6. **Unpublished items.** Armour wire counts (counted from Fig. 4a), lay
   directions, the radius at which the 10° core lay is defined, wet or dry
   design, manufacturing contact pressure, steel UTS and the S–N class are not
   stated by the source.
7. **Laid-core geometry.** Table 1 gives touching circular cores in the
   section; touching laid cores need a 0.16 mm larger pitch-circle radius,
   which puts the core envelope 0.11 mm into the bedding.
8. **Mass.** The copper screens of Table 1 are full annuli (2.0 and 3.15 mm);
   real tape and wire screens are lighter, so the template mass is high.
