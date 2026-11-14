<!-- SPDX-License-Identifier: Apache-2.0 -->

# Changelog

All notable changes to CableXSec are recorded in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
  load sharing between layers, and the neutral axes of an offset-core section.
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
- Thermal analysis: IEC 60287 steady conductor temperature with the cited loss factors, a
  thermal network, frictional heating and the thermal change of the clamping.
- Capacity and ultimate limit states: armour yield and sheath strain envelopes, armour
  compression (lateral and radial bird-caging), coiling into a static tank, cable over a
  sheave, critical curvature and the capacity check of time series with Gumbel extremes.
- Termination and bend-stiffener fields: termination-adjusted fatigue, bend restrictors and
  stiffener roots.

#### Fatigue and global-to-local post-processing

- Rainflow counting with residual half cycles and Miner damage at fixed material fibres of
  every wire, for bulk, contact (fretting) and conductor fatigue; sheath, tape, core-screen
  and thermal fatigue; electrical life screening.
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
