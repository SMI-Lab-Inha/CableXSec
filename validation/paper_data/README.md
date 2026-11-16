<!-- SPDX-License-Identifier: Apache-2.0 -->

# Data of the companion article

These files are the numerical data behind the CableXSec calculations of the companion
journal article (under review in *Ocean Engineering*), restricted to what can be
published: digitised data of published experiments, inputs on published cables, and the
results CableXSec computed from them. File names keep the prefixes of the article's
figure scripts (`figN_`, `revision_`, `revision2_`, `revision3_`) so that every figure
value can be traced; the prefixes describe the order in which the studies were added,
not different methods. All files are UTF-8 CSV or JSON in SI units unless the column
name states otherwise.

Not included: the manuscript text and figure sources, global simulation files, the
global floating-wind model (a third-party model of the IEA 15 MW turbine on the
VolturnUS-S platform), the hull panel drawing of that model, and any cable data
that may not be published. The cable used in all floating-wind calculations is the
published cable of Ménard and Cartraud (2023), bundled as
`cablexsec/cables/menard2023.yaml`, with the mass assumptions of
[`examples/fowt/menard2023_global_mass.json`](../../examples/fowt/menard2023_global_mass.json).

## Published measurements (digitised)

| Files | Source | Used by |
|---|---|---|
| `menard_fig16{a-f}_ordered.csv`, `_nominal.csv`, `_comparison.csv`, `menard_vector_extraction.json`, `revision_menard_comparison.json`, `revision_menard_metrics.csv` | Ménard and Cartraud, Marine Structures 91 (2023) 103465, Fig. 16: six measured moment–curvature ranges, extracted from the vector paths of the published PDF (the JSON records the page, axes and point counts) | BM1, BM1-loop, G8, T1 rows of `validation/validation_report.md`; `tests/test_menard2023_bm1.py` |
| `menard_fig18b_005.csv`, `menard_fig18b_020.csv`, `menard_local_displacement.json` | Same article, Fig. 18b: armour longitudinal displacement of the published FE model | armour displacement comparison (doc/benchmarks.md) |
| `fig8_menard_measured.csv` | Fig. 16 range used for the article's bending-loop figure | `fig8_mkappa_model.csv` comparison |
| `witz1996_fig5_0bar.csv`, `witz1996_fig6_300bar.csv` | Witz, Marine Structures 9 (1996) 885–904, Figs 5 and 6 | WITZ-EI, WITZ-ID rows; `tests/test_witz1996_benchmark.py` |
| `saevik2011_figs17_23.csv` | Sævik, Computers and Structures 89 (2011) 2273–2291, Figs 17–23 (FBG armour stresses) | SV11-PLANE row; `tests/test_saevik2011_benchmark.py` |
| `gjosteen2027_lc14_ranges.csv` | Gjøsteen et al., Marine Structures 111 (2027) 104189, Figs 27–35 (load case 14) | SV11-GJ row |
| `dong2025_figs9_14_25.csv` | Dong et al., Marine Structures 102 (2025) 103761, Figs 9, 14 and 25: axial stress of an armour wire fixed at the end fitting (FE and theory) | AW-DONG25 (doc/benchmarks.md); `tests/test_anchored_wire_fe.py` |
| `zhu_tan2014_figs4_8.csv` | Zhu and Tan, OMAE2014-23928, Figs 4–8: normalised axial stress of an anchored armour wire (FE and analytical model) | AW-ZT14 (doc/benchmarks.md); `tests/test_anchored_wire_fe.py` |
| `revision2_digitisation_bounds.json` | Half-widths of the digitisation of the Ménard ranges | sensitivity of the bending comparison to digitisation |

## Calculations on the published cable

| Files | Content |
|---|---|
| `fig8_mkappa_model.csv`, `fig8_mkappa_metrics.csv` | Calculated bending loop and its metrics against the measured range |
| `menard_identified_{a-f}.csv`, `revision_identification*.{json,csv}`, `revision2_identifiability.json`, `revision2_eta_*.{csv,json}` | Identification of the clamping and shear-support multipliers on range (a) and the withheld validation on ranges (b)–(f) |
| `revision_spatial_slip.json`, `fig23_spatial_slip_{small,large}.csv`, `revision_slip_onset.csv`, `revision_section_work.csv` | Spatial frictional-support cell against the section law; slip onset around the circumference; work balance |
| `revision2_transverse/`, `revision2_transverse_{work,markers}.csv`, `revision2_loop_equivalent.csv`, `revision2_contact_map_{inner,outer}.csv`, `revision2_operating_transverse_bound.json`, `fig5_matched_kinematics.csv` | Transverse (geodesic) slip, its work and the contact map of both armour layers |
| `revision_contact_loop_eta*.csv`, `revision_contact_*.csv`, `revision_fretting_contact.{csv,json}`, `fig_fretting_{map,ratio,sections}.csv` | Contact loops, contact stiffness and state sensitivities, fretting contact loading and regime map |
| `fig9_*.csv`, `revision_component_field.json` | Component stress field of the section under a stated load state |
| `revision_concentric_thermal.{csv,json}`, `revision_thermal_effect_*.csv`, `revision_interaction*.{csv,json}` | Concentric thermal subproblem (material data of Qiao et al. 2025) and the tension–twist–pressure interaction |
| `fig1_history.csv`, `fig11_global_excerpt.csv`, `fig12_*.csv`, `fig14_*.csv`, `fig15_*.csv` | Illustrative histories, capacity curve, biaxial sheath paths and damage, shell fatigue and electrical life (several use the bundled 33 kV examples) |
| `ga_sections.csv`, `ga_del.csv` | Section positions and DEL values of the graphical abstract |
| `revision_timings.json` | Wall-clock times of the calculations on one processor thread, with the host and package versions |

## Floating-wind application (global-to-local)

The global responses were computed by global time-domain simulation of the IEA 15 MW /
VolturnUS-S system in 200 m water depth with the Ménard cable in a lazy-wave
configuration, and the local recovery used the CableXSec fixed-material-point fatigue
of the package. Only results are included.

| Files | Content |
|---|---|
| `reference_fowt_20260913/` | Static equilibrium geometry of the cable, buoyancy section, moorings, tower and rotor (coordinates for the system illustration; `*_detail.csv` and `*_clipped.csv` are cropped views) |
| `fowt_static_mooring{1,2,3}.csv` | Mooring-line geometry in the static equilibrium |
| `reference_hour_20260913/` | One operating hour (DLC 1.2, 10.74 m/s): curvature ranges along the cable and fixed-material-point DEL and damage at the material sections; `source.json` records the digests of the global and local results |
| `reference_hour_spectra_20260913/` | Curvature spectra at the sag bend, hog bend, touchdown region and the selected arc position |
| `revision_fowt_seed_section_{0-8}_{inner,outer}_armour.csv` | Equivalent stress ranges per wind speed over six turbulence seeds at nine material sections |
| `revision3_consistent_{inner,outer}_armour_U*.csv` | Median ratio, over the seeds, of the damage with the consistent hysteretic bending law to the damage with the linear global bending stiffness, along the cable at four wind speeds (`n` seeds, with the spread as error bars) |
| `revision_irregular_*.{csv,json}` | Prescribed-motion irregular-sea cases (three sea states, six seeds) and their statistics |
| `global_demo/` | A short global time-domain demonstration history (`global_history_si.csv`) and its recorded summary; the io-global row of the validation report checks the reader against this summary |
