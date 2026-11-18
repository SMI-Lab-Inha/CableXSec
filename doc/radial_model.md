<!-- SPDX-License-Identifier: Apache-2.0 -->

# Solved radial contact state

Status: implemented as an opt-in contact model, `contact_model="solved"`.
The default contact model remains `"bounded"`, so no existing result
changes.

Modules:

- `cablexsec/mechanics/radial_solution.py`: coupled section solver and contact
  complementarity.
- `cablexsec/mechanics/trefoil_section.py`: plane three-core model.
- `cablexsec/mechanics/radial_cable.py`: `CableData` adapter and load semantics.

The option is exposed through the adapters in `cablexsec/cable_adapters.py`
and through `compute_mkappa(radial_state=...)` and
`compute_fatigue(contact_model="solved")`.

Tests:

- `tests/test_radial_solution.py`: closed form and consistency.
- `tests/test_radial_benchmarks.py`: published comparisons.

Effect report: `validation/scripts/radial_effect_report.py`.

---

## 1. Audit: how the radial and contact state is obtained before this change

### 1.1 Components

**A1. Bounded chain** (`mechanics/axisymmetric.py`, `compute_contact_forces`)
- Each helical layer adds a capstan pressure jump `T tan²α/(2π r²)` (Féret and Bournazel 1987, Eq. 5) plus a prescribed `pretension_Pa` jump `P0`.
- Jumps accumulate from the outside in. The result is projected to be non-negative.
- Not solved: radial displacement, gap compatibility, polymer hoop stiffness, temperature, twist.
- Layer tension comes from EA-weighted equal-strain load sharing, i.e. a rigid core.
- The chain sums pressures, whereas exact transmission across a thin layer with no hoop stiffness conserves the line force `p r` (Féret and Bournazel Eq. 2a). The two differ by `r_out/r_in`, about 9 % for the public templates.

**A2. Winding pressure** (`compute_winding_contact_pressure`)
- A prescribed residual wire stress `winding_stress_Pa` is converted into the same pressure jumps.
- Both manufacturing inputs are prescribed constants. They are added to every load case and cannot be relieved.

**A3. Concentric coupled section** (`mechanics/concentric_section.py`, `solve_concentric_section`)
- Solved:
  - exact Lamé generalised plane strain annuli and helical rings;
  - common extension and twist;
  - frictionless unilateral contact through the dual quadratic programme;
  - per-layer free thermal strain and initial wire strain;
  - internal and external pressure, tension, and torque or twist.
- Rejected: offset (trefoil) cores, nested strands, area-mismatched stranded conductors and corrugated shells.

**A4. Contact adapter** (`mechanics/contact_qp.py`, `solve_contact_qp`)
- The adapter to A3 for concentric cables.
- Manufacturing pressure is *not* converted to a residual strain.
- In `cable_adapters.armour_contact_pressure` (`"qp"`), `P1+P2` is the bounded-chain manufacturing pressure plus the solved face pressures. This is an explicit additive approximation.

**A5. Section state** (`mechanics/section_state.py`)
- The `qp` path solves A3 with temperatures and E(T).
- The `bounded` path uses the fixed-radius load share and A1.
- Three-core cables can only use `bounded`.

**A6. Three-core shape** (`mechanics/contact_pressure.py`, `cable_adapters.three_core_shape_for_layer`)
- A ring on a prescribed binary Winkler foundation with material-based constants.
- Exploratory only. It is never transferred to armour friction.

**A7. Hertz crossing loads** (`armour_crossing_line_loads`)
- The shared interface reaction divided by the wire count, then divided by the crossing density (`_fretting_contact`).
- In the bounded path the reaction is `2π r p_interface`, plus a superposed winding jump.
- In the `qp` path it is the solved reaction plus the superposed bounded pretension.

### 1.2 Where each load enters before this change

| Load | Bounded (A1/A2) | Concentric `qp` (A3/A4) | Three-core |
|---|---|---|---|
| Tension | EA-share capstan jumps | solved | bounded only |
| Torque or twist | no (twist 0) | solved | no |
| External pressure | added to every interface (dry) | outer surface | bounded |
| Internal pressure | subtracted at the innermost interface | bore | bounded |
| Temperature | no | free strain and E(T) via section state | no |
| Winding or pretension | prescribed jumps | superposed on the solution | prescribed jumps |
| Wet design (flooded interstices) | not represented | not represented | not represented |

### 1.3 What is solved and what is prescribed, by cable type

**(i) Single-core concentric cables**
- `qp` solves the full radial, axial and torsional equilibrium with unilateral contact.
- The manufacturing state is prescribed and superposed, so it can never open.
- The default `bounded` path solves no compatibility.

**(ii) Three-core cables**
- Nothing radial is solved.
- `P1+P2` is the prescribed manufacturing jump plus the rigid-core capstan increment of tension.
- Core-to-core and core-to-bedding forces are not computed.
- The polymer hoop stiffness, filler and bundle compliance do not enter.

**(iii) Umbilicals (BM4)**
- The element-stiffness assembler has no radial interaction.
- It over-predicts the fixed-twist axial stiffness of the Sævik and Bruaseth (2005) population by a mean factor of about 1.45 (U1 above 2).

## 2. What the literature solves

- **Sævik and Gjøsteen (2012), Sævik and Bruaseth (2005).**
  - 3D beam FE with penalty contact; radial interaction comes only from the surface-stiffness penalty. 2012 values: 7 MPa/mm for non-metal surfaces (3.5 for the soft core of umbilical 1) and 100 MPa/mm metal to metal. The 2005 paper used 100 MPa/mm at all interfaces.
  - Tables 4 to 12 are the same element-level EA, GJ, EI, radius and pitch as the 2005 appendix. They give no wire size, count, polymer modulus or contact topology.
  - Their stiffness results therefore cannot be reproduced from the tables alone.
- **Féret and Bournazel (1987).**
  - Kinematics, Eq. 4.
  - Pressure jump, Eq. 5; recursion, Eq. 6.
  - Unilateral gap rule (p < 0 gives p = 0 and an unknown gap).
- **Witz and Tan (1992a).**
  - Thin cylinders with thickness change.
  - Love-rod helices.
  - Substructuring at separated interfaces.
- **Custódio and Vaz (2002).** Generalised plane strain Lamé layers, Clebsch–Kirchhoff wires, lateral wire contact, and a core as an elastic foundation (1e8 to 2e9 Pa/m).
- **Knapp (1979).** A compressible core as a thick cylinder with a rigid insert in generalised plane strain; closed-form stiffness, Tables IV to VII.
- **Le and Knapp (1994).** Non-symmetric cross-sections as Fourier ring macro-elements, bonded contacts, no separation.
- **Dai, Sævik and Ye (2020).**
  - Hoop (side) contacts between same-layer elements.
  - Their 90° phase shift relative to radial contact friction reduces the friction amplitude. This is a bending effect, outside this axisymmetric solve.
- **Ménard and Cartraud (2023), Ménard (2020).**
  - Periodic 3D FE with all contacts resolved.
  - The initial state is *prescribed* as a uniform 0.3 MPa (thesis 0.45 MPa) pressure on the outer sheath, calibrated on the bending test.
  - The printed line loads at that state are 46 kN/m on the inner sheath and 16.5 kN/m per core.
- **Fang et al. (2023–2026).**
  - RUC FE of three-core cables with hard contact.
  - A prescribed calibrated pressure (0.4 MPa) for bending.
  - Normalised stiffness only; no geometry is published.
- **Su, Zhao and Zhu (2026).**
  - An analytical model with unilateral layer contact and an equivalent-ring kinematic link of the three cores, with the filler neglected. The link (Eq. 36) ties the binder ring to the cores' outer radius, so the ring is supported radially as in a filled bundle.
  - 3D FE with filler and cores; test EA for the 35 kV cable.
- **Gjøsteen et al. (2027).** A three-phase core contact element; trefoil core-to-core force under tension.

No source gives the radial state of a three-core cable as a function of tension, torque, pressure and temperature with a solved manufacturing state.

## 3. Design

### 3.1 One energy, one complementarity problem

The section state minimises the small-strain elastic energy per axial metre with common extension e and twist τ. It is subject to linearised unilateral gap constraints `g = g0 + C u ≥ 0`, reactions `λ ≥ 0` and `λ g = 0`.

**Concentric layers** (as in A3):
- Annuli use the exact basis `u = A r + B/r` in generalised plane strain.
- A helical ring stores `½ K (c² e + r c s τ + s² u/r − e_free + e_0)²`, with `K = n E A / cos α`.
- Its two faces sit at `u ∓ (t/2) e_free`, with `t` the radial thickness of the wire (`d` for a round wire, the declared thickness of a rectangular or profiled one, `radial_thickness_m`), so a free strain changes it.
- Faces are frictionless, unilateral and optionally compliant: `interface_compliance` is a series surface stiffness of the Sævik penalty type.

**Three-core bundle** (`trefoil_section.py`):
- Three identical multilayer cores, the interstitial filler and the bonded polymer layers enclosing them (bedding, binder, inner sheath) are a plane generalised-plane-strain FE model. It uses isoparametric 6-node triangles with a degree-4 rule.
- Uniform loads keep the three mirror planes of the trefoil. A 60° sector holding half a core is solved, and every energy is multiplied by six.
- Contacts:
  - Core to core: unilateral contact of the core with the 60° mirror plane.
  - Core to ring: node-to-segment, with the ring bore nodes against the core boundary.
  - Filler to core and filler to ring: node-to-node on coincident reference nodes.
- Each core is a helix of lay angle β (from `core_lay`): `e_core = cos²β e + r_c cosβ sinβ τ + sin²β u_c/r_c`. This produces the inward capstan load of the cores and the trefoil core-to-core force under tension.
- Layers outside the ring are coupled through the mean radial displacement of the ring outer surface. They apply uniform pressure (exact for the zeroth mode; armour has no hoop stiffness for other modes).

**Solver.**
- The dual linear complementarity problem in the reactions is solved exactly: NNLS on the scaled Cholesky factor of the contact flexibility, then polished on the active set, with complementarity checked.
- Sparse LU and contact responses are cached per end condition, so repeated load states cost one back-substitution and one complementarity solve.

### 3.2 Loads (`solve_cable_radial_state`)

| Input | Treatment |
|---|---|
| Tension | True wall tension, or prescribed axial strain (ends fixed axially, G9). |
| End cap | `end_cap=True` adds the closed-end thrust `π (p_i r_i² − p_e r_o²)` (G4). |
| Torque or twist | Twist fixed by default; torque on request. |
| External pressure | On the outermost surface (dry, watertight sheath). Defaults to the documented initial state, as before. |
| Internal pressure | On the bore of a concentric cable. Opt-in `pressure_barrier="<layer>"` names the innermost fluid-tight layer (a flexible pipe's pressure sheath; the interlocked carcass is not fluid-tight): the pressure acts on that layer's bore, the layers inside carry it on both faces (radially; their axial pressure term is not applied) and their face pressures exclude the fluid, and `end_cap` uses the barrier bore. Default unchanged. On the Witz (1996) pipe the default (pressure on the carcass bore) gives 6.04 MPa P1+P2 on the inner tensile armour at 300 bar and the barrier 7.55 MPa, equal to the section without the carcass within 0.03% (`tests/test_pressure_barrier.py`). |
| Flooded (wet design) | Every surface at water pressure p. The exact decomposition is a uniform stress −p plus the correction with free strain `−p(1−2ν)/E` and no surface load. Reactions are mechanical (net of water). The mechanical axial force is `T + p A_solid`, the effective-tension relation. |
| Temperature | Per-layer free strain `α (T − T_ref)`; E(T) for registered materials. |
| Manufacturing | The layer's declared `manufacturing_clamping` ([local_analysis_guide.md](local_analysis_guide.md) Section 4.4): a pressure jump (`pretension_Pa`) is converted by the inverse of Féret–Bournazel Eq. 5, plus a winding stress (`winding_stress_Pa`), giving a residual wire stress. Initial wire strains are *identified* so that the reference state (zero tension and twist, no pressure, T_ref) carries exactly these stresses. They are then carried through every load case, so contact opens or closes; nothing is superposed. A concentric layer's `extrusion_shrinkage` is an in-plane free strain `−hoop_strain` of that annulus (or of its elements, for the ring around a trefoil bundle), with `include_manufacturing` as the switch. |

### 3.3 Outputs

- Per helical layer:
  - inner and outer face line forces and pressures (`P1`, `P2`), referred to the helix radius so that `P·2πr/n` is the per-wire line load used by the Papailiou layers;
  - wire stress.
- Per interface:
  - line force, pressure at the contact radius and gap;
  - crossing reaction per wire for Hertz contact.
- Section quantities:
  - axial force and torque;
  - extension and twist tangent stiffness (fixed and free twist) on the active set.
- Three-core bundle:
  - core-to-core and core-to-ring forces;
  - filler forces;
  - mean ring pressure and displacement;
  - axial force of the cores;
  - representation notes.
- The bundle forces set the normal loads of the optional laid-core
  stick–slip law (`compute_mkappa(..., core_friction=...)`, gap R1b,
  `cablexsec/mechanics/core_friction.py`): per core two core-to-core
  forces, the filler-to-core force and the direct core-to-ring force, held
  fixed over the bending history. On the 66 kV template at 50 kN they give
  12.2 kN/m per core, almost all through the filler.

### 3.4 Stated representation choices

- Unlisted gaps inside a core (semiconducting screens) are filled with the softer neighbouring material.
- Loose (non-structural) core layers, such as copper tape and wire screens, are meshed with their declared material by default (`loose_layers="declared"`), so they carry axial load and torque in the solved section although the fixed-radius stiffness gives them none. `loose_layers="neighbour"` gives each the material of its softer structural neighbour; `torque_balance` uses it by default. On the 66 kV template the declared copper screens raise the solved torque/tension coefficient from 1.50 to 3.24 mm (fixed radius 1.51 mm).
- A serving (`construction: serving`: a served yarn or a wound tape) is an annulus, or ring of the three-core mesh, in plane stress along the cable axis: it has the radial and hoop stiffness of its material, so it transmits pressure, but no axial stress and no Saint-Venant torsion. It therefore adds nothing to the solved EA or GJ.
- Geometric overlaps below 0.5 % of the contact radius are treated as a snug fit (Ménard: 0.067 mm core-to-inner-sheath).
- A three-core cylindrical layer overlapping the cores is the interstitial filler, with three options:
  - `filler="solid"`: meshed in the outer interstices with unilateral contact.
  - `"void"`: flooded or loose yarn.
  - `"auto"`: as declared. The central interstice is void.
- Saint-Venant twist rigidity: the bonded ring layers twist about the cable axis and each core about its own centre (both exact for annuli). The filler lobes touch cores and ring through frictionless contacts that carry no axial shear, so they twist as separate bodies with their own warping (`saint_venant_rigidity`, validation row TQ-SV), not with their polar moment about the cable axis. The filler then adds 0.32 kN m² to the Chae, Nam and Shim (2023) cable instead of 7.9 kN m², and 0.14 kN m² instead of 3.6 kN m² to the Su et al. (2026) 35 kV cable.
- Stranded conductors whose metal area differs from the envelope, nested strands and corrugated shells are rejected, as in A3. The 220 kV template and CIGRE TB 880 are therefore unsupported.

### 3.5 Limits

- Linear elasticity and small displacement (linearised gaps).
- Frictionless radial solve, with no stick-slip feedback.
- No wire flattening or lateral (hoop) wire contact.
- No ovalisation of layers outside the ring.
- The core cross-section is taken in the cable plane, an error of order `1 − cos β`.
- Core layers are bonded.
- Ring layers are bonded to each other.
- Axisymmetric loads only. Bending redistribution is not solved.
- **The three-core state depends strongly on the filler representation** (sections 4 and 5). It is a modelling input, not a result.

## 4. Verification

### 4.1 Closed-form and consistency checks

| Check | Source of truth | Result |
|---|---|---|
| Concentric path vs existing `solve_concentric_section` (5 load cases) | Previous solver | Agreement to 1e-9 (reactions, strain, twist, tangent) |
| Prescribed strain returns the force that produced it | Inverse problem | 1e-9 |
| Single helix reaction | Féret–Bournazel Eq. 5 | 1e-10 |
| Rigid-bore stack | Σ EA cos³α (no-interaction bound) | 1e-9 |
| Foundation condensation | Closed form | 1e-9 |
| Ring far from cores | Plane-strain Lamé | 1e-5 (displacement and axial force) |
| Uniform free strain of a touching trefoil, with and without filler | Stress-free dilation | Contact forces below 0.01 N/m |
| Matched-material flooded section | Self-equilibrated water pressure | 1e-9 |
| Trefoil under core tension | Static equilibrium `F = F_zz sin²β/(√3 r_c)` | 1e-6 |
| Mesh 24 vs 48 cells; stabilisation 1e-10 vs 1e-12 | Convergence | Mean compliance within 1 %; stabilisation effect 1e-4 |
| Identified residual stress | Declared `pretension_Pa` (capstan) | Within 1 Pa; outer face jump `2πrP0` exact |

### 4.2 Published comparisons

| Case | Published | Model | Difference |
|---|---|---|---|
| Gjøsteen et al. (2027) three-phase tubes, 6.92 kN | 4.4 kN/m core-to-core | 4.41 kN/m | 0.3 % |
| Knapp (1979) EM cable k11 / k22 | 3.293e6 lb / 4.049e4 lb in² | 3.152e6 / 3.966e4 | −4.3 % / −2.0 % (rigid core: +19 %) |
| Knapp (1979) logging cable k11 / k22 | 1.260e6 / 8.961e3 | 1.190e6 / 8.544e3 | −5.6 % / −4.7 % |
| Knapp (1979) coupling | k12 / k21 (non-symmetric) | Lies between k12 and k21 | — |
| Su et al. (2026) 35 kV, EA tension (twist fixed, 500 kN secant) | FE 615.4 MN; test 631.7 MN | 608.7 MN | −1.1 % FE; −3.6 % test (no-interaction bound about +30 %) |
| Su et al. (2026) 35 kV, EA compression | FE 329.7 MN | 298.4 MN | −9.5 % |
| Su et al. (2026) 35 kV, GJ (axial strain fixed) | FE 209.5 / 201.2 kN m²; test 209.8 / 201.3 kN m² | 199.5 / 177.0 | −4.8 % / −12.0 % FE (the filler bonded to the ring without warping gave 203.0 / 180.5) |
| Su et al. (2026) 35 kV, armour torque (axial strain fixed) | Source analytical model, Fig. 19 shares: 193.0 / 172.3 kN m² | 193.3 / 170.8 | +0.2 % / −0.9 % |
| Su et al. (2026) 35 kV, interfaces at 1000 N m | Source analytical model, Table 11: anticlockwise separation at armour 1–2 and 3–4 (3–4 gap 0.031 mm), 58.3, 51.5, 55.2, 34.6, 32.0 kPa; clockwise armour 1–2 49.8, 3–4 49.3 kPa | Same separations (0.029 mm); 54.4, 57.0, 57.0, 38.4, 35.5 kPa; 55.2, 51.9 kPa | −7 % to +11 % |
| Su et al. (2026) 35 kV, interfaces at 100 kN | Source analytical model, Table 10 (kPa): 432.8, 283.9, 173.3, 176.4, 74.5; separation at armour 4 | 365.0, 255.6, 158.7, 161.9, 76.7; same separation | −16 % to +3 % |
| Ménard and Cartraud (2023), P = 0.3 MPa | 46 kN/m inner sheath | 26.6 (void interstices) to 72.0 (fully filled) kN/m | Bracket only |

**Inconclusive and negative cases**

- **Su et al. (2026) 35 kV, anticlockwise torsional stiffness.** −12.0 % against the FE and the test. The armour torque and the contact state agree with the source's analytical model (rows above). The polymer layers carry half of that model's share: their exact `G J` is 6.1 kN m², the model's share 12.0 kN m², twice the thin-wall terms of its Eq. 11. The rest is the armour's direction dependence: 11.3 % here, 10.1 % in the source's analytical model, 4.0 % FE, 4.1 % test; the FE's anticlockwise split and the test conditions are not published. No input is changed ([benchmarks.md](benchmarks.md), "Torsional stiffness of the Su et al. (2026) 35 kV cable").
- **Su et al. (2026), 36 kV.** The FE content is unclear: the tables are labelled "helical armour layers".
  - Compression with the armour package alone agrees (231 vs 234 MN).
  - Tension falls between the armour-only (220 MN) and full-section (389 MN) models, against FE 308 MN.
  - No gate is set.
- **BM4 (Sævik and Bruaseth 2005; Sævik and Gjøsteen 2012).** A unilateral ring stack on a radial foundation brackets every measurement between its unsupported and rigid limits. The rigid limit equals the existing assembler. The foundation stiffness needed per umbilical is:

  | Umbilical | U1 | U5 | U6 | U7 | U8 | U9 | U10 |
  |---|---|---|---|---|---|---|---|
  | Foundation (Pa/m) | 4.0e10 | 4.3e12 | 9.3e10 | 1.3e9 | 3.2e8 | 6.5e10 | 3.5e10 |

  U4 needs none. The values span four orders of magnitude, so **BM4 is not closed**. The tables carry no layer-wise radial data, and the source's own results rely on tuned penalty stiffnesses. The finding is recorded, not fitted.

## 5. Effect on the public templates

`validation/scripts/radial_effect_report.py` gives the friction-setting pressure `P1+P2`, the full-slip friction-stress cap, and the bulk damage of a fixed synthetic history (±0.05 1/m, 10 cycles). The synthetic history is a comparison device only.

The 33 kV values below are for the corrected example geometry (armour fill
at most 1, declared semiconducting screens). Against the pre-correction
geometry (76 + 84 overlapping wires, 1 mm core gaps filled with the
insulation material, kept as the `_legacy_r2` copies) the pressures differ by
about 1 % or less and the friction caps rise by 15–25 % with the smaller wire
count.

**Three-core 33 kV template** (PP filler declared; pre-correction copy `inter_array_three_core_33kv_legacy_r2`). Pressures in MPa, inner / outer armour.

| T (kN) | bounded | solved, filler | solved, void |
|---|---|---|---|
| 0 | 6.00 / 2.00 | 6.38 / 2.00 | 6.38 / 2.00 |
| 100 | 7.00 / 2.32 | 7.36 / 2.30 | 6.67 / 2.12 |

- Friction cap at 100 kN: bounded 85.1 / 30.8 MPa; solved 89.5 / 30.5 MPa.
- Bulk damage: unchanged for the inner armour; −2 % for the outer armour.
- The +6 % at zero tension is the line-force (`p r`) transmission across the armour (A1 note), not a compliance effect.

**Ménard and Cartraud (2023)** (P = 0.3 MPa calibration, fillers not in the input). Pressures in MPa.

| T (kN) | bounded | solved |
|---|---|---|
| 0 | 0.60 / 0.60 | 0.45 / 0.43 |
| 100 | 2.22 / 1.12 | 1.63 / 0.56 |

- At 100 kN the friction caps fall by 27 % (inner) and 50 % (outer).
- Synthetic bulk damage falls by 55 % (inner) and 68 % (outer).
- The published calibration (P with μ = 0.2) was made against the bending test with the source FE. A solved-state recalibration would be needed before this option is used for that benchmark.

**Single-core 33 kV template.** +11 % inner `P1+P2` at 0 kN (line-force transmission), −3 % outer at 100 kN. Damage is unchanged.

**220 kV three-core template and CIGRE TB 880.** Rejected: the stranded conductor area differs from its envelope, and the sheath is corrugated.

**Loads only the solved model represents** (33 kV three-core, T = 50 kN, inner / outer `P1+P2` in MPa):

| Case | Inner / outer | Change |
|---|---|---|
| Reference | 6.87 / 2.15 | — |
| All layers 50 °C | 7.66 / 2.38 | +12 % |
| Cores 70 °C | 8.05 / 2.51 | +17 % |
| Dry external pressure 1 MPa | 8.85 / 3.95 | +29 % |
| Flooded (wet design) 1 MPa | 6.94 / 2.18 | +1.1 % |

For a wet-design cable, external hydrostatic pressure therefore adds almost no armour clamping. The bounded chain adds the full pressure to every interface.

## 6. Interface items

| Item | Status |
|---|---|
| **G4** | `solve_cable_radial_state` accepts internal, external and flooded pressure and `end_cap=True` (closed-end thrust). `compute_mkappa(..., radial_state=state)` consumes the solved state. `solve_cable_radial_state` and `CableRadialState` are public. |
| **G9** | `RadialSection.solve(prescribed_axial_strain=...)` and `solve_cable_radial_state(prescribed_axial_strain=...)` support axially fixed ends; tested against the force-driven solution. |
| **G5** | Cable-declared materials in YAML: **not done** (schema and material-library change outside this gap); listed below. |

## 7. Remaining work

1. **Filler representation.** A measured or manufacturer filler geometry and modulus, profiled fillers and yarn compaction law. This is the dominant three-core uncertainty (Su 35 kV: removing the filler halves EA).
2. **Stranded conductors** as a radial bundle law, allowing the 220 kV and CIGRE cables, and **corrugated sheaths** as a radial constitutive model.
3. **Bending.** Non-uniform armour pressure over the cores, ovalisation of the outer layers, and the Dai et al. (2020) hoop contacts. The present solve is axisymmetric; the core friction law (gap R1b) uses its contact forces held fixed during bending.
4. **Interface compliance data.** Surface stiffness (Sævik penalty) for umbilical layers; BM4 remains open.
5. **Wire lateral contact and flattening** (Custódio and Vaz 2002), and plasticity.
6. **Recalibration.** The Ménard published-calibration pressure under the solved model, before any default change.
7. **Section state.** `SectionState` (stress recovery, capacity) does not yet consume the trefoil solution, so `wire_yield_curvature` re-solves at tension only.
8. **G5:** cable-declared materials.
9. **Su 36 kV:** clarify the FE content with the authors before using it as a gate.
