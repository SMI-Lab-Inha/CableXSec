<!-- SPDX-License-Identifier: Apache-2.0 -->

# Validation data

This directory is for reviewers and users who want to check or reproduce the published
CableXSec assessments. It holds the compact inputs, the reproduction scripts and the
numerical results; the case definitions and pass criteria are in
[`VALIDATION.md`](../VALIDATION.md) and the benchmark register of the manual
(`doc/benchmarks.md`). Source publications, licensed standards, global models and
global simulation files are not redistributed.

| Path | Contents |
| --- | --- |
| [`validation_report.md`](validation_report.md) | The generated table of every analytical gate and benchmark, with targets, values and status |
| [`RELEASE_0_2_1.md`](RELEASE_0_2_1.md) | Environment, test results, gate results and distribution hashes for v0.2.1 |
| [`RELEASE_0_2_0.md`](RELEASE_0_2_0.md) | The same record for the 0.2.0 release |
| [`RELEASE_0_1_0.md`](RELEASE_0_1_0.md) | The same record for the 0.1.0 release |
| [`paper_data/`](paper_data/README.md) | Digitised published measurements and the results of the companion article's calculations |
| `scripts/` | The report generator, benchmark harnesses and reproduction scripts listed below |

## Reproduce the validation report

From the repository root, with CableXSec installed:

```powershell
python validation/scripts/run_validation.py           # rewrite validation_report.md
python validation/scripts/run_validation.py --check   # compare with the committed report
```

The script recomputes every row from the live code and the data in `paper_data/`, and
runs the three test-marker groups for the taxonomy counts (about an hour on one core). It
exits with status 1 while any row is `FAIL` or `REVIEW`; the failed rows are the
registered measured comparisons listed in [`VALIDATION.md`](../VALIDATION.md#summary).

## Scripts

| Script | Purpose |
| --- | --- |
| `run_validation.py` | Generates `validation_report.md` |
| `check_saevik2011.py` | Sævik (2011) and Gjøsteen et al. (2027) FBG armour-stress comparison (SV11 rows) |
| `check_core_friction.py` | Laid-core friction checks and published comparisons |
| `check_g8_gradual_reversal.py` | Gradual stick-to-slip transition against Witz, Hu et al. and Ménard |
| `check_t1_test_arrangement.py` | Benchmarks in their published test arrangements (three-point, four-point) |
| `check_out_von_morgen.py` | Frictionless slip bounds of Out and von Morgen (1997) and Out (1989) |
| `check_conductor_strand_slip.py` | Conductor strand slip and the Hu et al. (2022) conductor loop |
| `check_compression_buckling.py`, `compression_application.py` | Armour compression (lateral and radial bird-caging) checks and application |
| `check_torque_balance.py` | Torque balance against Feyrer, Knapp, Kim and Lee, and Utting and Jones |
| `check_fretting_regime.py`, `run_fretting_regime.py` | Fretting regime, Mindlin energy and wear checks |
| `periodic_friction_cell.py`, `periodic_shear_cell.py` | Independent spatial references: a helical rod on frictional support and a finite-element check of the elastic shear support |
| `check_dai2018_friction.py`, `digitise_dai2018_fig4.py` | Static/dynamic friction law against the measured wire-on-tape loops of Dai, Sævik and Ye (2018), and the digitisation of those loops (DAI18 rows) |
| `slider_initial_work.py` | Exact path-work bounds for an elastic–Coulomb slider with an unknown initial force |
| `transverse_friction_cell.py`, `transverse_friction_runs.py` | Helical rod on elastic support with two-component Coulomb friction, and the same-cable comparison of contact-level and section-level bending work |
| `radial_foundation_fd.py` | Independent periodic finite-difference solution of a unilateral radial ring |
| `radial_effect_report.py` | Effect of the solved radial contact state on armour friction and damage |
| `concentric_conductor_bend.py` | Bending reference with coupled axial slip in concentric wire rings |
| `helical_hertz_reference.py` | Independent local Hertz reference for helically swept circular sections |
| `benchmark_fatigue_counter.py`, `profile_release.py` | Timing of the fatigue counter and call profiles on public inputs |
| `benchmark_throughput.py` | One-core throughput of sea-state fatigue on the 66 kV template (direction-tied and distributed models, all wires or 12 positions, 4 or 32 fibres), damage differences from an earlier build, and the sea-state batch against the worker count |
| `fowt_construction.py`, `fowt_section_checks.py` | Construction and mass inputs of the floating-wind application, and the bending-range check of the metallic sheath |

The tests in `tests/` import these scripts by name; `tests/conftest.py` puts this folder
on the import path.

## Global analysis references

Global time-domain results are committed as the summary values and reduced histories
that the tests and the report check. The global models and simulation files are not
redistributed.
