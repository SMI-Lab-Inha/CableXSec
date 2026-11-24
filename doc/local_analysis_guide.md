<!-- SPDX-License-Identifier: Apache-2.0 -->

# Local analysis guide

This guide is for engineers who want to run a local (cross-section) analysis of
a dynamic power cable with CableXSec. It covers what the package computes, the
inputs it needs, a complete workflow from a global time history to fixed-fibre
fatigue and contact indices, and the limits within which results can be used.

All quantities are SI (m, N, Pa, s, 1/m, °C unless a name says otherwise).
Stress is tensile-positive; pressure is positive in compression. The code
blocks run as written, except those that read your own files. The blocks of
one section run in sequence in one Python session; each section starts afresh.

A companion journal article describing the method is under review; its
reference will be added when it is published.

## 1. What CableXSec computes

CableXSec is a semi-analytical, component-resolved section model. For a cable
defined layer by layer it computes:

- **Section stiffness** (`compute_stiffness`): axial with fixed and free twist,
  axial–torsion coupling, and bending bounds: full slip (`ei_min_Nm2`),
  shear-compliant no slip with the Yun coefficient ξ (`ei_max_Nm2`) and the
  plane-section reference (`ei_plane_section_Nm2`). The shear stack of ξ is
  the concentric layers between an armour layer and its support, the next
  helical layer inside or the laid cores (`yun_shear_stack`).
- **Helical armour stick–slip bending** (`compute_mkappa`): Papailiou
  slip-front law, slip spreading from the neutral axis; friction cap from
  Féret–Bournazel winding clamping plus the tension increment; Yun shear support.
- **Axisymmetric normal response** (`solve_section_state`): extension, twist,
  layer stresses and interface pressures under tension, torque, pressure and
  temperature.
- **Stress of every component** (`compute_stress`,
  `compute_component_response`): conductor, insulation, screens, sheaths and
  each armour wire, with circumference extrema.
- **Fixed-fibre fatigue** (`compute_fatigue`, `fatigue_from_series`) under
  irregular biaxial curvature and tension or direct-strain histories from
  a global analysis; every material fibre keeps its angle and is counted alone.
- **Contact at armour crossings**: Hertz footprint and pressure, Mindlin
  tangential stiffness, slip amplitude `delta*` and the shear-slip product
  `mu * p0 * delta*` (an indicator, not the Ruiz parameter).
- **Fretting regime and wear** (`compute_fretting`, Section 11): stick,
  partial or gross slip per slip cycle on both faces of every armour wire,
  dissipated energy, energy-wear depth per year and a regime-dependent
  knock-down of the contact-fibre S–N damage.
- **Thermal clamping**: IEC 60287 steady temperatures, or temperature
  histories from the transient thermal network (Section 7.3), and the
  resulting differential expansion in the section state.
- **Corrugated metallic sheath** stress and fatigue (`compute_shell_stress`,
  `compute_shell_fatigue`, elastic S–N or strain life, Section 13.3).
- **User fatigue data** (Section 4.2): registered S–N curves
  (`register_sn_curve`), mean-stress corrections (`MeanStressSpec`) and
  component stress concentration factors (`ComponentSCF`) in the armour,
  conductor, sheath and tape routes and the sea-state batch.
- **First-yield capacity screening** (`compute_capacity`): wire yield,
  bird-caging, conductor fracture and, with a sourced allowable, polymer crushing.
- **Capacity check of global time series** (`build_capacity_envelope`,
  `check_capacity`, Section 8.1): utilisation history, maximum and its time,
  exceedances, minimum bend radius and Gumbel extremes over seeds.
- **Bending tests and finite specimens** (Sections 6.2–6.4): test
  arrangements, slip redistribution along the wires, rectangular and profiled
  wires with transverse slip.
- **Power-core screens, tapes, restrictor fields** (Section 13).
- **Armour compression screens** (`compression_screen`,
  `compute_compression_capacity`): lateral and radial buckling, design
  factors and the compression–curvature curve ([uls_guide.md](uls_guide.md)).
- **Torque balance and twist under tension** (`torque_balance`,
  `design_torque_balance`, Section 12): end torque with the twist restrained,
  twist of a free end, layer torques and the stress redistribution between
  them, and the outer-armour lay angle, wire count or wire diameter that
  balances the torque.

Every result object carries a `provenance` record with the method, citations,
verification gates and active limitations. Read it.

## 2. What CableXSec does not compute

- Global cable response; histories come from a global analysis program.
- Finite-element contact. Pressures come from the bounded pressure chain or the
  coupled concentric equilibrium (`"qp"`); neither resolves three-core cyclic
  contact redistribution or bending-induced ovalisation.
- Wire plasticity. `compute_mkappa` warns (`ExtrapolationWarning`) above the
  wire-yield curvature.
- A qualified steel fretting life. No steel slip-amplitude S–N curve is
  registered (`perier_2009_steel` is blocked); contact outputs are indicators.
  The fretting knock-down of Section 11 is a sourced factor with an assumed
  slip constant, and the steel-on-polymer wear coefficients are assumptions.
- Lead-sheath fatigue (`cigre_tb623_lead` raises `BlockedSNCurveError`).
- A qualified conductor fatigue or fretting life. `compute_conductor_fatigue`
  (Section 10) gives strand-slip wire strains and bulk Miner sums from public
  parameters; its checks against tests are model-to-test comparisons with the
  differences stated there (gap CF2: tension–bending lives within a factor of
  2.4, tension–tension location not reproduced). The trellis term is off by
  default; the legacy surface rule (`trellis_model="surface_swt"`) is not
  validated.
- The IEC three-core mutual-heating factor. (Transient heat transfer through
  the IEC 60287 ladder is available, Section 7.3.)
- The slip-branch bending stiffness of a compact, solid-filler three-core
  cable (known limitation, gap G8; [benchmarks.md](benchmarks.md), "Known
  limitation: slip-branch stiffness of compact sections").
  - On the one such measured cable, the model's slip tangent is about 3.5
    times softer than measured.
  - On a cable with rope fillers, EI_min matches the measurement.
  - A slip branch that is too soft over-predicts the curvature for a given
    moment, which is conservative for curvature-driven quantities. It
    under-predicts the section moment and the loads it puts on interfaces (bend
    stiffener, hang-off, the contacts inside the section), which is not
    conservative.

## 3. Installation

```powershell
pip install -e .
pip install -e .[fast]   # optional numba; enables the compiled kernels (Section 9)
```

numba is the optional extra `fast`. Without it (or with
`CABLEXSEC_DISABLE_NUMBA=1`) the same functions run on a NumPy/pure-Python
path with identical results, only slower.

## 4. Inputs

### 4.1 Cable definition (YAML)

A cable is a YAML file validated by the Pydantic schema in
`cablexsec/cables/schema.py`. Packaged definitions:

| Function | File | Content |
| --- | --- | --- |
| `inter_array_single_core_yaml_path()` | `inter_array_single_core_33kv.yaml` | Public representative 33 kV single core, double steel armour (34 + 40 wires) |
| `inter_array_three_core_yaml_path()` | `inter_array_three_core_33kv.yaml` | Public representative 33 kV three core (61 + 67 wires); cores laid at 10° (pitch 0.926 m, the IEA 66 kV bundle lay angle at this core-centre radius; direction assumed) |
| `cablexsec.cables.inter_array_single_core_legacy_r2_yaml_path()`, `..._three_core_legacy_r2_yaml_path()` | `*_33kv_legacy_r2.yaml` | Pinned pre-correction copies (armour fill 1.13–1.24, undeclared screens): reproduction of the figures of an earlier revision of the companion article and of the validation report only |
| `export_three_core_yaml_path()` | `export_three_core_220kv.yaml` | Public 220 kV topology example with corrugated Al sheath; cores laid at 10° (pitch 2.138 m, as the 33 kV template); not a qualified design |
| `cigre_tb880_yaml_path()` | `cigre_tb880_220kv.yaml` | Public CIGRE TB 880 / Poradowski 220 kV export cable |
| `menard2023_yaml_path()` | `menard2023.yaml` | Public Ménard and Cartraud (2023) three-core cable |
| `witz1996_yaml_path()` | `witz1996.yaml` | Public Witz (1996) ISSC 2.5-inch flexible pipe (benchmarks WITZ-EI, WITZ-ID); area-equivalent round wires, initial pressure identified on the 0-bar loop |
| `witz1996_yaml_path(wires="rectangular")` | `witz1996_rect.yaml` | The same pipe with its 6 × 3 mm strips, profiled carcass and Zeta (gap G1) |
| `saevik2011_yaml_path()`, `saevik2011_yaml_path(tendons="rectangular")` | `saevik2011.yaml`, `saevik2011_rect.yaml` | Public Sævik (2011) 8-inch riser for the FBG armour-stress benchmark (SV11); assumed outer layers listed in the header |
| `thies2016_yaml_path()` | `thies2016_30kv.yaml` | Public Thies et al. (2016) 30 kV three-core cable with wire screens and copper tape |
| `iea_task49_66kv_yaml_path()` | `iea_task49_66kv.yaml` | Public 66 kV three-core wet-design cable of Janocha et al. (2024) and IEA Wind Task 49; provenance in [template_66kv.md](template_66kv.md) |

Top-level blocks are `cable` (metadata), `environment` (`ambient_C`,
`T4_external_Km_W`, `frequency_Hz`), `initial_state`, `layers` (centre
outwards), optional `core_lay` (assembly pitch of offset cores) and optional
`aggregate_properties` (reference metadata, never used in place of a section
calculation). `initial_state.external_pressure_Pa` is applied to the assembled
cable before loading; a non-zero value needs an `identification` class
(`measured`, `manufacturing_specification`, `published_calibration`, or
`identified` when CableXSec fitted it to a named measured response, which
is then not a prediction) and a `source`.
Optional `water_ingress` declares `design` (`wet` or `dry`, CIGRE TB 610
Section 5.4.3) and `armour_annulus` (`flooded` or `sealed`). A wet design
may not contain a radial metallic water barrier (`cylindrical_shell`, lead or
aluminium sheath) and a dry design must contain one. The declaration does not
change any analysis default; clamping pressure and S–N curve stay explicit
(Section 7.2).

Layer types (`layer_type`):

| Type | Use | Main fields |
| --- | --- | --- |
| `helical` | Armour wires | `n_wires`, `d_wire_m`, `r_helix_m`, `lay_angle_deg`, `lay_direction`, `manufacturing_clamping` (Section 4.4; legacy `pretension_Pa`, `winding_stress_Pa`), `bulk_sn`, `fretting_sn`; flat or profiled wires: `wire_shape`, `wire_width_m`, `wire_thickness_m`, `wire_corner_radius_m`, `wire_area_m2`, `wire_I_normal_m4`, `wire_I_lateral_m4`, `wire_J_m4`, `transverse_slip_fraction` (below) |
| `helical_stranded` | Nested strands | `n_conductors`, `r_cable_m`, `lay_angle_cable_deg`, strand fields |
| `cylindrical` | Concentric polymer annulus | `r_inner_m`, `r_outer_m` |
| `cylindrical_offset` | Core component (conductor, insulation, screen) | `n_cores`, `r_offset_m`, `r_inner_m`, `r_outer_m`, optional `strand_layers`, `electrical` |
| `cylindrical_shell` | Metallic sheath | `r_mid_m`, `t_m`, `f_corrugation`, `scf`, `bulk_sn` |

Common fields: `name`, `material` (a key of the material registry), `source`,
`notes`, `structural` (false: no bending contribution and not a fatigue site,
though the layer still transmits pressure) and `termination_scf` (stress-range
factor at an end fitting; 1.0 means none). Component stress concentration
factors (axial, friction and local bending, mean and dynamic) are a fatigue
option, not a YAML field (Section 4.2).

```yaml
  - name: inner_armour
    layer_type: helical
    material: galvanised_steel_armour
    n_wires: 40
    d_wire_m: 0.004
    r_helix_m: 0.02447
    lay_angle_deg: 25.0
    lay_direction: right_hand
    manufacturing_clamping:         # Section 4.4
      initial_contact_pressure_Pa: 2.0e6
      basis: representative
      source: "Representative pressure jump P0; not a manufacturer value"
    fretting_sn: ""                 # empty: explicitly bulk-only
    bulk_sn: saevik_2011_flat_steel
```

Load and check a definition:

```python
import cablexsec as cx

cable = cx.load_cable(cx.inter_array_single_core_yaml_path(), geometry_override=None)
assert not cable.missing_fields()      # analyses require complete geometry
stiffness = cx.compute_stiffness(cable)
print(stiffness.ea_fixed_N, stiffness.ei_min_Nm2, stiffness.ei_max_Nm2)
print(stiffness.xi_per_layer)          # Yun shear coefficient per armour layer
```

`geometry_override` defaults to `None` (no override). A cable definition may
write withheld geometry as `null` and supply it from a separate YAML file,
merged by layer name, with `geometry_override=path`. An override layer that
matches no layer, an unknown key or a changed `layer_type` is refused. A file
that is not UTF-8 YAML with a mapping at its root and a list of layers raises
`CableSchemaError`; schema violations
in the fields raise `pydantic.ValidationError`. The manufacturing clamping
(Section 4.4) sets the contact pressure that drives friction at zero tension;
without it there is no
friction at zero tension. A pressure jump `P0` (`initial_contact_pressure_Pa`,
legacy `pretension_Pa`) is the radial pressure jump across its own layer, not
a face pressure: the jumps accumulate from the outside in, and the
friction-setting pressure is the sum of both face pressures. Two armour layers
each with `P0` = 2.0 MPa therefore have interface pressures of 4 and 2 MPa and
`P1+P2` of 6 MPa (inner) and 2 MPa (outer).

**Rectangular and profiled wires (gap G1).** Flat armour strips and tendons
are declared with `wire_shape`; `d_wire_m` is then filled in as the diameter
of the round wire of the same area (a given value must agree with it):

```yaml
  - name: inner_tensile_armour
    layer_type: helical
    material: steel_saevik2011
    n_wires: 55
    wire_shape: rectangular        # "round" (default), "rectangular", "profiled"
    wire_width_m: 0.0125           # in the cylinder surface, normal to the wire
    wire_thickness_m: 0.005        # radial
    wire_corner_radius_m: 0.0      # optional
    transverse_slip_fraction: 0.0  # full-slip transverse state, 0 loxodrome .. 1 geodesic
    r_helix_m: 0.1320
    lay_angle_deg: 24.80
```

A `profiled` wire (interlocked carcass, Z or C profile) gives its envelope
`wire_width_m` x `wire_thickness_m` and the source's `wire_area_m2`,
`wire_I_normal_m4` (tangential inertia), `wire_I_lateral_m4` (radial
inertia) and `wire_J_m4`. The section (`layer.wire_section`,
`cx.WireSection`) carries the area, both principal inertias, the torsion
constant (exact series for a rectangle), the radial thickness and surface
width, and the stress points (perimeter of the rounded rectangle). It is
used for the radial extent in every contact route (bounded, QP and solved:
the strips no longer overlap their tapes), the circumferential fill (a
layer that overfills is rejected), the Yun support width, the individual
bending in the full-slip `EI_min` (lateral and normal curvature with both
inertias), the local bending stress at the outer face `t/2` and the edge
`b/2` (`WireStressField.at_fibre` maps a fibre angle to the perimeter,
`at_point(y_n, y_l)` gives any section point, for example a gauge groove),
the capacity corner stress and the lateral buckling bound. Crossed flat
strips use the mean face pressure over the rhombus `b1 b2 / sin θ` (no
Hertz peak, no Hertz tangential stiffness: the slip-amplitude fretting
branch is off for them); `compute_fretting` and the torque-balance design
reject flat wires. Round wires are unchanged bit for bit. Packaged:
`cx.saevik2011_yaml_path("rectangular")` and
`cx.witz1996_yaml_path("rectangular")`.

### 4.2 Materials and S–N registry

Material constants and fatigue curves are module-level data with a citation for
each value:

```python
from cablexsec.materials import library, sn_curves

print(library.list_materials())
print(library.get_property("galvanised_steel_armour", "mu_on_hdpe"))   # 0.20
print(sn_curves.list_sn_curves())
curve = sn_curves.get_sn_curve("saevik_2011_flat_steel")
print(curve.environment, curve.cycles_to_failure(100.0))   # stress range in MPa
```

| Curve | Environment | Intended use |
| --- | --- | --- |
| `saevik_2011_flat_steel` | seawater, cathodic protection | Armour bulk fatigue, no cut-off |
| `dnv_rpc203_t_curve` | seawater, cathodic protection | Bilinear DNV T curve, DNV-RP-C203 (2024) Table 2-3; thickness exponent 0.25 above 16 mm |
| `oc2017_armour_in_air` | air | Armour in a dry, sealed annulus |
| `nasution_2014_copper` | air | Copper individual-wire bulk fatigue, R = 0.1 |
| `en1999_1_3_aluminium` | air | Corrugated Al sheath (category 71-7) |

The copper SWT constants `copper_poon2022` are in `sn_curves.SWT_LIFE`; a
`fretting_sn` that names them selects the per-cycle SWT fretting assessment
(Section 10).
Blocked entries are listed in `sn_curves.BLOCKED_SN_CURVES`. Bulk curves use
stress range in MPa; by default no mean-stress correction is applied (the
registered armour curves are stress-range curves, and the spiral-strand data
reviewed by Sævik (2011) do not collapse better under Goodman). A layer's
`bulk_sn` and `fretting_sn` select the curves; an unknown or blocked name is an
input error, never silent zero damage.

**User S–N curves, mean-stress corrections and component SCFs.** Partner or
project curves are registered for the session and then used by name, in a
YAML `bulk_sn` or as an override:

```python
import numpy as np
import cablexsec as cx

curve = cx.SNcurve.from_knee(
    "partner_d_seawater", stress_unit="MPa", m1=3.0, log_C1=11.764, m2=5.0,
    knee_cycles=1.0e6, environment="seawater, cathodic protection",
    reference="[DNV-RP-C203] Table 2-2, D curve",
)
cx.register_sn_curve(curve)          # reference and environment are required

cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
t = np.linspace(0, 12 * np.pi, 240)
kx, ky, tension = 0.05 * np.sin(t), 0.02 * np.cos(t), 4.0e4 + 1.0e4 * np.sin(0.3 * t)
res = cx.compute_fatigue(
    cable, kx, ky, tension_N=tension,
    sn_curves={"inner_armour": "partner_d_seawater"},
    mean_stress="goodman",            # UTS and its citation from the material
    scf=cx.ComponentSCF(friction_dynamic=1.2, bending_normal_dynamic=1.1,
                        source="[ModelAssumption] example factors"),
)
print(res.provenance.method.split("; user S-N options")[1][:120])
cx.unregister_sn_curve("partner_d_seawater")
```

- `SNcurve` holds one or two slopes, the knee (explicit or by continuity;
  `SNcurve.from_knee` from the knee cycles), a cut-off or endurance limit
  (`threshold`), the test stress ratio `R_ratio`, an `environment` label, the
  source (`reference`) and an optional thickness exponent `k` with
  `reference_thickness_m` (ranges are multiplied by `(t/t_ref)^k` for
  `t > t_ref`, DNV-RP-C203 Eq. 2.10; the wire's radial thickness is used for
  armour).
- `MeanStressSpec(method=...)`: `"goodman"`, `"gerber"`, `"soderberg"` (with
  `uts_Pa` or `yield_Pa` and `strength_source`, or
  `MeanStressSpec.from_material`), `"swt"`, `"walker"` (`walker_gamma` and
  `walker_source`), and the DNV-RP-C203 range factors `"dnv_base"` (Eq. 2.15)
  and `"dnv_pwht"` (Eq. 2.16). Every rainflow cycle is converted to the curve's
  test condition: its `R_ratio`, or a constant mean
  (`reference_mean_stress_Pa`). A compressive
  mean is taken as zero in Goodman, Gerber and Soderberg unless
  `compressive_mean="signed"`. A method name in `compute_fatigue` reads the
  strength of the layer's library material with its citation.
- `ComponentSCF`: separate factors on the axial part (tension or solved normal
  stress and thermal stress), the friction stress (with any termination
  increment), and the local bending at the outer face and at the edge, each
  for the mean over the record and for the fluctuation about it. A `source` is
  required when a factor differs from one.
- The same options exist in `compute_conductor_fatigue` (axial factors on the
  axial wire stress, friction factors on the stick–slip strain of the
  operator), `compute_shell_fatigue` (elastic route: normal-bending factors and
  an optional `static_mean_stress_Pa`), `compute_tape_fatigue` (`sn_curve=` adds
  a stress-range check of `E ε` beside the strain life, axial factors) and per
  section in `SectionRequest` of the sea-state batch.
- Without these options every result is unchanged bit for bit. The strain-life
  routes keep their own SWT or Morrow corrections.

### 4.3 Global histories

All readers return a `LoadCaseSeries`: uniform-step `time_s`,
`curvature_x_1_m`, `curvature_y_1_m`, `tension_N`, optional `axial_strain` and
`temperature_C`, plus `tension_kind` (`true`, `effective`, `unspecified`) and
`axial_strain_kind` (`direct`, `total`, `unspecified`). Arrays are copied into
read-only snapshots; edit with `dataclasses.replace`.

- **Curvature** is the axial-strain gradient: a fibre at (x, y) =
  r (cos φ, sin φ) has bending strain `kx*x + ky*y`, so the gradient points to
  the most tensile fibre. Global codes report the right-handed rotation
  curvature (k_x, k_y), fibre strain `k_x*y - k_y*x` (checked against a
  global line model: a hogging cantilever root, (0, +k), is in tension at
  φ = 180°, a sag bend, (0, −k), at φ = 0). The readers (`read_time_history_csv`,
  CableDyn `strain_gradient()`, RIFLEX
  `curvature_convention="rotation"`) convert it to the gradient
  `(-y curvature, x curvature)`; bend moments map to `(-M_y, M_x)`
  (`cablexsec.mechanics.section_conventions`). `curvature_map="paper_r2"`
  gives the negative mapping (every fibre and wire azimuth turned by 180°)
  and exists only to reproduce the results of the companion article's first
  submission. The curvature magnitude loses reversals
  and cannot be used.
- **Force**: fatigue needs true (wall) force. Effective tension is rejected
  until converted with `series.with_true_tension(external_pressure_Pa,
  external_area_m2, internal_pressure_Pa=..., internal_area_m2=...)`; an
  unqualified force column is not assumed to be true force.
- **Strain**: only `Direct tensile strain` (kind `direct`) enters the
  one-dimensional stress law; `Total mean axial strain` contains Poisson
  deformation and is rejected. When direct strain is present it drives the
  wire stress instead of tension. The armour contact then stays at the
  declared clamping unless a force history sets it:
  `compute_fatigue(..., axial_strain=strain, contact_tension_history_N=force)`
  with, for example, `force = strain * EA_fixed`, gives the damage of
  `tension_N=force`.

CableDyn runs are read section by section with `read_cabledyn_sections` and
evaluated over sea states with `run_sea_state_batch`; see
[global_local_guide.md](global_local_guide.md).

Readers:

| Function | Source | Notes |
| --- | --- | --- |
| `read_time_history_csv(path, ...)` | Time-history CSV export | Whole words of the heading, case-insensitive, exact heading first; must match one column; magnitude headings refused for curvature; row length checked. Defaults: `x curvature`, `y curvature`, `Wall tension` (kN, scale 1e3), strain in percent (scale 0.01) |
| `read_riflex_ascii(path, ...)` | RIFLEX ASCII | Positional columns; declare `tension_kind` and `curvature_convention` |
| `read_cabledyn_sections(out_root, deck, sections, ...)` | CableDyn driver or OpenFAST module files | Returns `CableDynSectionHistory` per section; curvature vector reconstructed from tangent channels or positions; `.to_load_case_series(A_o)` gives the true-force series |
| `read_series_csv` / `write_series_csv` | Canonical CSV | Round trip preserves kinds |
| `jonswap_series(...)` | Synthetic | For demonstrations and tests only |

### 4.4 Manufacturing clamping state

The clamping that the armour carries at zero applied tension (retained
winding stress, extrusion or sheath pressure) is an input, not a prediction.
It governs every friction-dependent result: the bending hysteresis, the
armour friction stress and its fatigue, the fretting contact loads and how
all of these change with tension and temperature. Two public cases show its
weight: the measured Ménard and Cartraud (2023) loops need an initial
pressure of 0.2–0.3 MPa (0.22 MPa identified with the full core and filler
accounting, 0.3 MPa calibrated by the source), and the cold-to-rated damage
factor of the 66 kV template falls from 115 to about 20 and 2 with 40 and
100 MPa armour winding stress ([template_66kv.md](template_66kv.md)
Section 4.2.1).

Declare it once per armour layer:

```yaml
    manufacturing_clamping:
      armour_winding_stress_Pa: 62.1e6      # or initial_contact_pressure_Pa (P0)
      basis: assumed                        # see below
      source: "Assumed ~9% of the 690 MPa proof stress (range 7-11%)"
```

- Exactly one value: `armour_winding_stress_Pa` (a retained wire stress,
  converted to a pressure jump by Féret and Bournazel 1987, Eq. 5, in the
  bounded chain and carried as a residual wire stress in the solved radial
  state) or `initial_contact_pressure_Pa` (the jump `P0` across the layer).
- `basis`: `measured`, `manufacturing_specification`,
  `published_calibration`, `assumed`, `representative` or `not_published`
  (which requires zero). `source` is required.
- A uniform pressure applied to the assembled cable before loading is the
  cable-level `initial_state` (Section 4.1, with its `identification`); it
  covers every layer (the Ménard cable: 0.3 MPa, published calibration).
- The legacy keys `winding_stress_Pa` and `pretension_Pa` still load with the
  same value; without a declaration their basis is reported as
  `undocumented`. Changing a legacy value with `model_copy` (a sensitivity
  study) removes the declaration from the copy.
- Every contact route reads the same value: the bounded chain, the
  concentric `qp` route and the solved radial state, and through them
  `compute_mkappa`, `compute_fatigue`, `compute_fretting`,
  `simulate_rig_test` and the thermal coupling.

`cx.clamping_state(cable)` resolves the declaration (`describe()` prints
it). When a friction-dependent analysis forms the armour contact pressure of
a cable with no declaration (and no identified `initial_state`), or whose
clamping is zero because it is declared `not_published`, it issues
`cx.MissingClampingStateWarning`. The results are then those of
tension-induced contact alone: friction stresses and damages sit at their
low end and ratios that change the clamping (temperature, tension) at their
high end. At low tension the friction moment and the bending hysteresis then
tend to zero, which a manufactured cable does not show: a low-tension
hysteresis is unreliable without a declared clamping state.

**Extrusion shrinkage.** An extruded sheath or bedding cools and shrinks onto
the layers inside it; the part that is not relaxed stays as a shrink fit.
Declare it on the `cylindrical` layer:

```yaml
    extrusion_shrinkage:
      hoop_strain: 0.004          # frozen in-plane shrinkage
      basis: assumed
      source: "Assumed frozen hoop shrinkage of the extruded sheath"
```

The radial solutions carry it as an initial in-plane strain of the layer
(hoop and radial, no axial part): the solved radial state
(`contact_model="solved"`, including the ring around a trefoil bundle, so the
laid-core law reads the clamped trefoil) and the concentric `"qp"` route.
Contacts then open or close under later loads, and
`include_manufacturing=False` leaves it out as it does the winding stress.
The bounded chain has no displacement compatibility to carry an interference
and refuses a non-zero value. The frozen strain depends on the extrusion and
cooling history and on the polymer's relaxation, which CableXSec does not
model: it is an input like the winding stress. A shrinkage of a layer
outside an armour layer counts as that layer's clamping declaration.

**Calibration scope.** A pressure calibrated on a measured loop holds
whatever friction the calibrating model had. `initial_state` (for
`published_calibration` and `identified` values) and `manufacturing_clamping`
(for `published_calibration`) take `calibration_scope`: `armour_only` or
`all_contacts`. A value fitted with the armour friction alone was raised
until the armour matched the measured dissipation, so it already holds the
friction of the other contacts; the core stick–slip law (Section 6.1) then
warns (`cx.ClampingCalibrationWarning`), because adding the cores counts that
friction twice: re-identify the value with the friction model in use. A value
fitted with friction on every contact is consistent with the core law in
principle, but it belongs to the calibrating model's contact accounting, so a
transfer to CableXSec is still a transfer, not a prediction.

Packaged cables:

| Cable | Clamping | Basis |
| --- | --- | --- |
| 33 kV single and three core | `P0` 2.0 MPa, both armours | representative |
| 220 kV three-core export | `P0` 3.0 MPa, both armours | representative |
| Ménard and Cartraud (2023) | `initial_state` 0.3 MPa | published calibration, all contacts |
| Witz (1996) 2.5-inch pipe | `initial_state` 1.037 MPa | identified (0-bar loop width, μ = 0.10), armour only |
| 66 kV IEA Task 49 template | zero | not published (warns) |
| CIGRE TB 880 export | none declared | missing (warns) |

## 5. Worked workflow: global history to fatigue and contact indices

### Step 1: cable and history

With a time-history CSV export at one physical section (signed x/y curvature, wall
tension and direct tensile strain):

```python
import cablexsec as cx

series = cx.read_time_history_csv(       # needs your own file
    "section_history.csv",                  # your export
    axial_strain_col="Direct tensile strain",
)
```

For a curvature export in 1/mm pass `curvature_scale_to_1_m=1000`. The rest of
this section uses a synthetic history so that it runs as written:

```python
import math
import numpy as np
import cablexsec as cx

cable = cx.load_cable(cx.inter_array_single_core_yaml_path(), geometry_override=None)
series = cx.jonswap_series(
    hs_m=3.0, tp_s=9.0, duration_s=600.0, dt_s=0.2,
    rms_curvature_1_m=0.01, mean_tension_N=50e3, tension_kind="true", seed=1,
)
kx, ky = series.curvature_x_1_m, series.curvature_y_1_m
```

### Step 2: fixed-fibre fatigue of the whole record

```python
result = cx.fatigue_from_series(cable, series)          # n_fibres=32 by default
print(result.critical_layer, result.phi_critical_deg, result.fibre_critical_deg)
print({name: d.max() for name, d in result.D_bulk_per_wire.items()})
print(result.D_bulk_per_fibre["inner_armour"].shape)    # (wires, fibres)
```

`D_bulk_per_fibre` keeps damage at every (wire, fibre) material point;
`D_bulk_per_wire` is its maximum over fibres. `phi_critical_deg` is the wire
azimuth and `fibre_critical_deg` the fibre angle from the outward radial
direction. With a true-force history, contact pressure follows the tension;
`contact_tension_N=` freezes it. Increase `n_fibres` to check angular
convergence. Never rainflow-count the instantaneous circumference maximum: in
rotating bending it can be constant while material points cycle. For direct
control use `cx.compute_fatigue(cable, kx, ky, ...)` with one of `tension_N`,
`axial_strain` or `section_states`.

**Local bending follows the slip state.** The normal (radial, weak-axis)
local bending of each wire is `E (t/2) [cos²α cos2α p_n + (sin²α / r) σ_f / E]`
with `p_n = κx cos φ + κy sin φ` and `σ_f` the friction stress the wire carries
at that sample (with any termination increment). The first term is the curve
curvature of the deformed helix (Out and von Morgen 1997, Eq. 41; Leroy and
Estrier 2001, Eq. 13); the second converts it to the rotation of the wire's
cross-section per unit material length, which is what bends the wire. A stuck
wire with the plane-section strain therefore takes `cos⁴α`, with the stick
coefficient `ξ` it takes `cos²α cos2α + ξ sin²α cos²α`, and a wire without
friction stress keeps `cos²α cos2α`. In Coulomb slip the slope stays
`cos²α cos2α`. The lateral term `cos α (1 + sin²α) p_l` does not depend on the
slip state. `radial_Pa` of `WireStressField` holds the sum, so every fibre
recovery and component stress concentration factor uses it. The stick and slip
limits were checked against an independent finite-element model of a wire on a
bent support (not published).

**Selected wire positions.** For long records at many cable locations,
`cx.compute_fatigue(..., wire_positions=12)` evaluates 12 equally spaced wire
positions per armour layer instead of every wire. A list of azimuths in degrees
(`wire_positions=[0, 90, 180, 270]`) takes the wire nearest to each, and a
mapping gives each layer its own selection. Wire `k` lies at the azimuth
`360 k / n_wires`, the convention of `phi_critical_deg`. Each selected wire's
damage equals its value in the evaluation of every wire, bit for bit, because
the friction recursion, stress field and rainflow count of one wire do not
depend on the others. The per-wire arrays then hold the selected wires in
ascending order; `result.wire_index` and `result.wire_azimuth_deg` name them.
The critical wire and the damage sums are over the selected wires only, so
choose enough positions to resolve the damage around the circumference: the
largest damage of a selection is a lower bound of the largest damage of the
layer. The direction-tied model solves only the selected wires. The
distributed model solves the helix of the whole layer and keeps the selected
wires, so there only the rainflow counting is saved. `SeaStateBatchOptions`
takes the same `wire_positions`, and `compute_long_term_fatigue` requires every
case to use the same wires.

### Step 3: discard the slip initialisation, count the retained hour

Friction state must be built up before the counted record. Solve the stress
field over the complete history once, then count only the retained samples.
This is the compiled path used for production runs:

```python
from cablexsec.analysis.fast_rainflow import fibre_fatigue
from cablexsec.analysis.fatigue import wire_stress_field
from cablexsec.cable_adapters import cable_to_fatigue_layers

strain = series.tension_N / cx.compute_stiffness(cable).ea_fixed_N  # use exported direct strain if available
n_init = 500                                     # initialisation samples
retained = slice(n_init, None)
duration_s = series.time_s[-1] - series.time_s[n_init]
angles = np.arange(32) * 2 * np.pi / 32

for layer in cable_to_fatigue_layers(cable, contact_tension_N=0.0):
    field = wire_stress_field(layer, kx, ky, axial_strain=strain)
    damage, del_1hz_MPa, s_min_MPa, s_max_MPa = fibre_fatigue(
        field.mean_Pa[:, retained], field.radial_Pa[:, retained],
        field.tangential_Pa[:, retained], angles, layer.sn_bulk, duration_s,
    )
    wire, fibre = np.unravel_index(np.argmax(damage), damage.shape)
    print(layer.name, damage.max(), wire, np.degrees(angles[fibre]))
```

The same lead-in is one argument of `compute_fatigue`:
`cx.compute_fatigue(cable, kx, ky, tension_N=T, count_from_sample=n_init)`
runs the friction state through the whole record from the stress-free section
and counts the bulk cycles from sample `n_init` on; `0` (default) counts every
sample, bit for bit as before. It is not available with a contact-life branch
or with `section_states`, and the result's provenance records it
(`tests/test_fatigue_lead_in.py`).

`WireStressField` stores three (wire, time) arrays: mean, radial and
tangential bending terms. `field.at_fibre(angle_rad, samples=mask)` recovers
any fibre without resetting slip. `fibre_fatigue` also returns the 1 Hz
equivalent stress range and the stress extrema per fibre. In a campaign, count
per retained hour and combine seeds only at identical (wire, fibre) points.

The loop above holds each layer's contact pressure at its zero-tension
(declared) value for the whole record. To let `P1+P2` follow the tension,
pass a history: `P, _ = cx.armour_contact_histories(cable, tension)` and
`wire_stress_field(..., contact_pressure_Pa=P[layer.name])`. In a flooded
annulus the armour is loaded by the effective tension (clamped at zero); a
sealed annulus also needs the hydrostatic clamping. The postprocessor's
`--contact tension_history` (default) does this with the effective tension;
`--contact declared` holds the declared clamping at zero tension, the
condition of the companion article's results, which it reproduces bit
for bit. For a cable without manufacturing clamping the declared state
carries no armour friction stress at all.

### Step 4: contact indices at armour crossings

Layers with an empty `fretting_sn` carry no Hertz data, so compute the crossing
directly. The interface line load comes from the same pressure chain as the
fatigue calculation:

```python
from cablexsec.cable_adapters import armour_crossing_line_loads
from cablexsec.materials import library
from cablexsec.mechanics.hertz_contact import (
    crossed_cylinder_curvatures, layer_crossing_angle_rad,
    layer_crossing_density_per_m, reduced_modulus, reduced_shear_modulus,
    solve_contact,
)
from cablexsec.mechanics.mindlin_contact import (
    MindlinContact, geometric_slip_factor, slip_increment,
)

layer = cable.get_layer("inner_armour")
partner, line_load = armour_crossing_line_loads(cable, 50e3)["inner_armour"]
hand = lambda l: 1.0 if l.lay_direction == "right_hand" else -1.0
a, b = math.radians(layer.lay_angle_deg), math.radians(partner.lay_angle_deg)
density = layer_crossing_density_per_m(
    partner.n_wires, layer.r_helix_m, a, hand(layer), partner.r_helix_m, b, hand(partner))
force = line_load / density                       # N per crossing
A, B = crossed_cylinder_curvatures(layer.d_wire_m, layer_crossing_angle_rad(a, b),
                                   partner_diameter_m=partner.d_wire_m)
E, nu = (library.get_property(layer.material, p) for p in ("E_Pa", "nu"))
hertz = solve_contact(force, A, B, reduced_modulus(E, nu, E, nu),
                      reduced_shear_modulus(E, nu, E, nu))

mu = library.get_property(layer.material, "mu_on_steel")
c_geo = geometric_slip_factor(layer.r_helix_m, a)
dkx, dky = np.diff(kx, prepend=0.0), np.diff(ky, prepend=0.0)
phi_wires = 2 * np.pi * np.arange(layer.n_wires) / layer.n_wires
delta_star, phi_ss = np.zeros(layer.n_wires), np.zeros(layer.n_wires)
for k, phi in enumerate(phi_wires):
    contact = MindlinContact(K_t=hertz.K_t_N_per_m, mu=mu)
    for x, y in zip(dkx, dky):
        contact.update(slip_increment(x, y, phi, c_geo, transverse_ratio=0.0),
                       force, hertz.p0_Pa)
    delta_star[k] = contact.fretting_amplitude
    phi_ss[k] = contact.shear_slip_product()
print(hertz.p0_Pa / 1e6, delta_star.max() * 1e6, phi_ss.max())
```

The slip amplitude `delta*` is half the largest chord of the plastic slip
trajectory; the shear-slip product is `mu * p0 * delta*` with the peak Hertz
pressure `p0`. It is not the Ruiz parameter, which uses the local surface shear
traction (and, in its second form, the local tangential stress), and no Ruiz
parameter is computed. Both quantities are indicators for
ranking wires and comparing designs, computed with constant normal load and a
frictionless Hertz footprint; they are not a fretting life.
`compute_hertz_axis_stress(hertz, depths_m)` and
`compute_hertz_subsurface_stress` give the elastic subsurface tensor for that
footprint (`examples/hertz_axis_stress.py`, `examples/prescribed_contact_swt.py`).

### Step 5: long-term aggregation

```python
case = cx.FatigueLoadCase(
    "Hs3_Tp9_seed1", result,
    duration_s=float(series.time_s[-1] - series.time_s[0]),
    probability=1.0, assessment_id="single-core-v1",
)
long_term = cx.compute_long_term_fatigue([case], exposure_years=25.0,
                                         design_fatigue_factor=10.0)
print(long_term.design_utilization, long_term.critical_layer, long_term.critical_fibre_deg)
```

Probabilities must cover the whole exposure and sum to one; divide a
condition's probability among its seeds. Damage is combined at each fixed
fibre before the maximum is taken. Do not split one continuous record into
chunks: cycles crossing chunk boundaries would be lost. `damage_basis=
"bulk_with_termination"` accumulates the SCF-amplified field (see
`examples/long_term_termination_fatigue.py`).

### Step 6 (optional): sections near an anchor

The section laws assume armour that is free to slip axially. Within a few
boundary layers of an end fitting that anchors the armour, or of a
bend-stiffener root where the armour is clamped, friction feeds a mean wire
force that the sectional law cannot carry. `termination=` replaces the
sectional friction stress of every wire by the anchored-wire solution of
`cablexsec/mechanics/clamp_layer.py` (sectional value plus the clamp
increment) along the actual curvature history:

```python
s_m = np.linspace(0.0, 2.0, 21)          # arc length from the clamp face [m]
kappa_std = np.exp(-s_m / 0.5)           # curvature shape, e.g. its standard deviation
spec = cx.TerminationSpec(
    distance_m=0.25,                     # section from the clamp face
    anchor="clamped",                    # or "none" for no correction
    setback_m=0.0,                       # straight frictional length to the anchor
    curvature_profile=(s_m, kappa_std),  # shape along the cable; None = uniform
    biaxial="principal",                 # or "components"
)
result = cx.compute_fatigue(cable, kx, ky, tension_N=series.tension_N, termination=spec)
print(result.clamp_correction)          # a_max, l_x, reach, principal fraction
```

A dictionary `{layer_name: TerminationSpec}` corrects selected layers only;
`wire_stress_field(..., termination=spec)` gives the corrected stress field
with the increment in `field.termination`.

- **Model.** Each wire is followed from the anchor (``x = -setback``) to the
  solved reach beyond the section: rigid stick, Coulomb bound per unit length
  `σ_u = σ0 μ|sin α|` scaled by the pressure history, axial slip only, exact
  chain projection per increment. The increment is `g_k σ_u (n − n_far)`,
  where `n_far` comes from the same discrete solver for a wire far from any
  anchor at the section curvature. The far-field limit is therefore the
  sectional result to round-off, and a section beyond the reach
  (`reach_lengths` boundary layers `ℓ_x = ½√(L_s P)` at the largest
  curvature, default 8) is returned unchanged, bit for bit. The sectional
  friction (direction-tied or distributed) keeps its full biaxial law.
- **Curvature profile.** `curvature_profile` is normalised to one at
  `distance_m`, so `kx, ky` are the histories at the analysed section; the
  loading is taken as proportional, `κ(x, t) = κ_s(t) k̂(x)`. The profile
  changes the result through both the anchor and the gradient.
- **Biaxial histories.** The clamp solver is planar. `"principal"` solves it
  along the principal direction of the centred history; the minor component
  (variance fraction `1 − principal_fraction`) gets no clamp increment.
  `"components"` adds a second planar solution along the minor direction;
  for non-proportional histories this superposition of two Coulomb responses
  is not exact (it tends to over-estimate when both components slip) and is
  not a bound. Both reduce to the exact planar result for planar histories
  in any direction. Report `principal_fraction` with the result.
- **Orientation.** `axis_sign=+1` when distance from the anchor increases
  along the positive axial direction of the curvature frame; with the lay hand
  of the layer (taken from the cable) it sets the sense in which wires turn
  along the cable. The sign changes individual wires, not the size of the
  effect.
- **Cost.** One solver run per wire plus one far-field run per distinct wire
  phase offset and direction. For a one-hour record at 10 Hz: about 1.5–2.5 s
  per wire at `a_max ≈ 25` and about 7 s at `a_max ≈ 110` (one thread). The
  cost grows with the solved reach (`∝ √a_max`) and with the number of load
  substeps (`∝ κ/κ_c`).

- **Governing input.** The slip-onset curvature `κ_c` sets the slip length
  and hence the size of the effect (`A − 1 ∝ √(κ/κ_c)`). The package uses
  its ξ-reduced stick stiffness; a rigid-interface `κ_c` (ξ = 1) is about
  4–5 times smaller and gives a larger correction. Report the convention
  with any corrected damage. The
  convention is an explicit option: `stick="scaled"` (default, unchanged),
  `stick="rigid"` (ξ = 1), and with `solver="along_wire"` also
  `stick="spring"` (the distributed model's elastic interface,
  `k̂ = ξ/(1 − ξ)`, full imposed strain) and `capstan=True`. The along-wire
  solver follows each wire's azimuth-dependent contact shape and takes the
  biaxial strain directly; with `stick="scaled"` and no capstan it equals the
  clamp solver to round-off.
- **Anchor compliance.** The anchor is rigid by default (`u = 0` at the
  anchor). `k_anchor_N_m=K` holds each wire end with the force `K u` instead
  (`K` in N/m per wire: the axial force on the wire end per unit axial
  displacement of the wire end at the anchor), in parallel with the friction
  of the first half segment; `0` is a free wire end, `None` or `inf` the
  rigid anchor on the unchanged code path. All solvers (clamp, along-wire,
  either under a curvature field) take it; the far-field reference is
  unchanged. Normalised, `k_a = K R/(EA sin α)` (`ξ EA` for `"scaled"`):
  the anchor is as compliant as a length `EA/K` of free wire, so it matters
  only when `EA/K` is comparable with the boundary layer `ℓ_x`.
  A closed form checks it: a wire under a uniform imposed strain `A` with
  a spring anchor slips over `L` next to the anchor, with
  `k_a ρ L²/2 + ρ(1 + k_a/β)L + k_a ρ/β² + ρ/β = A` (`β = √k̂`, `β = ∞`
  for rigid stick), anchor force `A − ρ/β − ρL` and dissipation
  `ρ²(L²/(2β) + L³/6)` ([benchmarks.md](benchmarks.md), TA-V1).
- **Estimating `K` for a potted end fitting.** This is an
  order-of-magnitude estimate from shear lag, not a value from a source. A
  wire of axial stiffness `EA` bonded over a length `L_e` in a potting of
  shear modulus `G_p`, with an effective shear-layer thickness `t_p` over
  the wire perimeter `p`, has the bond stiffness per unit length
  `k_s = G_p p/t_p` and `u'' = λ² u` with `λ = √(k_s/EA)`; its loaded end has
  `K_b = EA λ tanh(λ L_e)`, within 4 % of `√(EA k_s)` once `λ L_e > 2`. An
  unbonded wire length `l_f` between the anchor plane and the start of the
  bond adds `l_f/EA` in series: `1/K = 1/K_b + l_f/EA`. For a 5 × 12.5 mm
  tendon (`EA = 1.3e7 N`, `p = 35 mm`), `G_p = 0.5`–`4 GPa` (unfilled to
  filled epoxy, typical values) and `t_p = 0.5`–`5 mm` give `λ = 16`–`150
  1/m` and `K_b ≈ 2e8`–`2e9 N/m` for `L_e ≥ 0.15 m`; 0.1 m of unbonded wire
  lowers it to about `1e8 N/m`. A 5 mm round wire (`EA = 4e6 N`) gives
  `K_b ≈ 8e7`–`7e8 N/m`. The anchor then behaves like `EA/K ≈ 0.007`–`0.16
  m` of extra free wire. A mechanical wedge or a hooked wire end changes
  these numbers; measure or model the fitting when the result matters.
- **Measured check.** Near the bellmouth of the Sævik (2011) riser the
  corrected ranges are 1.65 times the FBG measurements (median, clamp
  solver), 2.1 with the along-wire solver in the `"scaled"` convention and
  3.4 with `"spring"` or `"rigid"`; the sectional ranges are 1.03
  ([benchmarks.md](benchmarks.md), SV11 and T2). The anchored-wire Coulomb model is not
  supported by these measurements. An elastic anchor in the estimated range,
  or a free wire end, does not change this (benchmarks.md, "Anchor compliance of the end
  fitting", a reported sensitivity).

What it does not represent: transverse slip (except the vector-Coulomb
relief of `lateral_relief=` with the along-wire solver, Section 6.4), wire
flattening, interaction between layers, a bonded length inside a fitting as
a distributed bond (put the anchor at its inner end, or lump the bond into
`k_anchor_N_m`), bending-induced pressure
redistribution and the change of wire and stick moduli over section states.
In its own regime (uniform curvature from a rigid anchor, monotonic loading)
the clamp solver reproduces the anchored-wire stress of a published
beam-on-shell finite-element model (Dong et al. 2025: 156.4 against 156.9 MPa
at a neutral-axis anchor, peak within 2 %); it does not reproduce the larger
rise of Zhu and Tan (2014), whose finite-element wire does not slip far from
the anchor ([benchmarks.md](benchmarks.md), "Anchored wire at uniform
curvature against published FE", reported). The anchor effect grows with
`κ/κ_c`: it vanishes below full slip and is about 1.2 (anchor) and 1.7
(peak) times the free-field peak at a neutral-axis anchor at `κ/κ_c ≈ 5`.
The only measured comparison is the Sævik (2011) check above, which it does
not meet; the near-anchor
damage depends strongly on the anchorage position and on `μ P`
(`A − 1 ∝ √(κ/κ_c)`). Treat corrected damage as a model estimate of the
termination effect, not a validated life.

## 6. Bending loop and single-curvature stress

```python
import numpy as np
import cablexsec as cx

cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
loop = cx.compute_mkappa(cable, 0.05 * np.sin(np.linspace(0, 6 * np.pi, 601)))
print(loop.EI_min_Nm2, loop.EI_max_Nm2, loop.W_diss_J_per_m)

stress = cx.compute_stress(cable, 0.02, 0.0, tension_N=1e4)
response = cx.compute_component_response(cable, 0.02, tension_N=1e4)
for state in response.states:
    print(state.name, state.core_state, state.sigma_local_max_Pa,
          state.sigma_frictional_max_Pa, state.sigma_stick_max_Pa)
```

- `W_diss_J_per_m` is friction work over the whole supplied history, including
  conditioning cycles. For closed-cycle loss use `loop.work` and
  `cablexsec.analysis.curve_comparison.insert_cycle_crossings`.
- `stick_support` selects the model of the armour stick coefficient ξ. It is
  accepted by `compute_mkappa`, `compute_stiffness`, `SectionLaw.from_cable`
  and `xi_for_layer`.
  - The default, `"yun_stack"`, is the series thin-layer stack of Yun et al.
    (2020).
  - `"annulus"` (opt-in) solves the shear of the layered polymer annulus
    under the armour with its actual inner boundary: rigid at a helical layer
    or laid cores, free at an empty bore, regular at the axis. It adds the
    concentration of the load on each wire's width.
  - On thin layers over a stiff support the two agree within a few per cent.
    On a soft or hollow core, `"annulus"` is much softer, because the core
    shears as a beam.
  - It is the exact elastic solution of that annulus: a thin layer on a rigid
    support gives t/G, a solid core R/G, and a hollow core with a free bore
    R(1+ρ²)/(G(1−ρ²)), several times the thin-strip value.
  - The option changes the stick branch, `EI_max` and the slip onset, not
    `EI_min`. It does not close gap G8.
  - Other entry points, such as the fatigue path and the thermal section,
    use the default.
- The contact pressure is that of one contact state unless a history is
  given. `contact_tension_history_N` (one tension per curvature sample) sets
  the armour `P1+P2` of every sample through the contact model, the same
  histories `compute_fatigue` forms from `tension_N`
  (`armour_contact_histories`); `contact_pressure_Pa` takes the per-sample
  `P1+P2` of every structural armour layer directly, for example from a
  sequence of solved section states. The friction bound of each wire follows
  the pressure with the slip state carried across, so a tension cycle inside
  the bending cycles gives the same friction stress as the fatigue path
  (registered check MK-Tcyc). Only that bound follows: the stick and full-slip
  stiffnesses, the laid-core and ring contact loads and the wire-yield range
  check use the reference state (the first sample of a tension history).
  `loop.contact_pressure_Pa` returns the pressures used. `SectionLaw` and the
  bending-test simulator keep a fixed pressure, because their leg solver
  evaluates trial curvatures that carry no sample time.

  ```python
  t = np.arange(1200)
  kappa = 0.03 * np.sin(2 * np.pi * t / 200)
  tension = 1e4 + 6e4 * np.sin(np.pi * t / 370) ** 2   # N, within the cycles
  loop = cx.compute_mkappa(cable, kappa, contact_tension_history_N=tension)
  ```
- `compute_stress` is a monotonic result at one curvature, not a cyclic history.
- Offset components (laid power cores) are reported in the three core slip
  states of Section 13.1: `sigma_stick_*` (core held to the cable axis,
  `"stuck"`), `sigma_frictional_*` (the core friction law of Section 6.1 in
  closed form for a monotonic ramp, default `CoreFrictionSpec()`, μ = 0.2 on
  every core contact; with `core_core_coupling="relative"` the end of a
  400-increment proportional ramp, since the coupled stations redistribute
  their stress) and `sigma_local_*` (free core slip: each core bends
  about its own axis). The stuck state holds only below the slip curvature of
  the cores; on the IEA 66 kV template the cores slip at about 3e-4 1/m, so at
  any practical curvature the frictional state is close to free slip.
  `strain_*`, `sigma_equivalent_max_Pa`, `sigma_net_section_max_Pa` and the
  utilisation use the state named by `state.core_state` (argument
  `core_state`, default `"frictional"`; `core_friction` sets the friction
  inputs). Cores without a declared core lay cannot slip over a uniform bend
  in this law (the slip cap grows as 1/sin β), so they are assessed stuck and
  `state.basis` says so. None of the states is a resolved contact solution.
- Every component is checked against all the limits its material gives
  (`cablexsec/analysis/component_limits.py`); `state.utilisations` lists each
  limit checked, `state.failure_mode` the governing one and `state.limits` the
  values with their sources. The conductor is checked against its 0.2 % proof
  stress (`sigma_y_Pa`; 120 MPa for the IEA 66 kV copper, Janocha et al. 2024,
  Sec. 2.1, p. 5) at the extreme wire fibre and against its tensile strength on
  the net section (`sigma_net_section_max_Pa`, the axial force of the most
  loaded core over its area, the basis of the conductor-fracture mechanism of
  `compute_capacity`); polymers against an allowable strain (`epsilon_allow`;
  7.7 % for PE, API 17J Table 6, p. 19); metallic screens and sheaths against
  their yield stress as a screening check. A component whose material gives
  no limit reports `"no limit given"`; none is assumed.
- A stranded conductor modelled as a solid with an effective bending modulus
  `E_b` below its axial modulus `E` bends on the basis of that stiffness: its
  bending stress about its own axis is that of the slipping wires of radius
  `R sqrt(E_b/E)` that give it, `sqrt(E_b/E)` times the plane-section stress
  of a solid (`cx.own_bending_ratio`); the core-axis stress (stick, friction)
  is added unscaled. On the IEA 66 kV template at 80 kN and 0.1 1/m the
  conductor takes 46 MPa in the frictional state (utilisation 0.39) and
  488 MPa held to the cable axis (4.07).
- A loose screen (no axial stiffness) carries the stress of the strain it
  follows, its modulus times that strain, marked `state.kinematic`; it is
  assessed against its limits but is never the critical component: a screen
  that slips over its core takes less.
- `mu` sets the armour friction coefficient as in `compute_stress`; a cable
  whose armour material has no published coefficient needs it.
- The utilisation of a corrugated metallic sheath is first yield of the von
  Mises stress at the corrugation root, `SCF f_corr E κ r` plus the axial,
  thermal and hoop terms: a local, strain-controlled first-yield check, not a
  failure of the sheath. The product `SCF f_corr` is the strain transfer of
  the corrugation; with the representative pair of the 220 kV export template
  (0.5 and 2.0) it is one, so that sheath behaves as a smooth tube (utilisation
  4.2 at 0.05 1/m) and `state.basis` says so. A metallic screen or sheath is
  made to deform plastically in bending, so its first yield does not bound
  the capacity of the cable (`cx.capacity_utilisation`); its allowable strain
  does, where its material gives one.
- `sigma_vm_Pa`, `sigma_vm_envelope_Pa` and `critical_sigma_vm_Pa` are a legacy
  axial/contact screening combination, `sqrt(sigma_axial**2 + 3*(0.31*p0)**2)`,
  dominated by GPa-level crossing pressures. It can exceed 1 at zero curvature
  and is not a first-yield check. `compute_component_response` reports it for
  armour as `contact_screen_utilization` (NaN for other layers). The armour
  `mechanical_utilization` is first yield, `max |sigma_A + sigma_B + sigma_f +
  sigma_T| / sigma_y`, on the same basis as `compute_capacity` (Section 8), and
  reaches 1 at its first-yield curvature.

### 6.1 Friction of the laid cores (three-core cables)

By default the bending law has armour friction only; the laid power cores
enter as individual bending (full slip, in `EI_min`). `core_friction` adds a
stick–slip law for the cores against their neighbours, the filler and the
surrounding ring, and for resolved internal wire rings (`strand_layers`):

```python
cable = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
spec = cx.CoreFrictionSpec(mu_core_filler=0.46, mu_core_core=0.46,
                           mu_core_sheath=0.46)   # Janocha 2024 Table 2, MDPE
loop = cx.compute_mkappa(cable, 0.2 * np.sin(np.linspace(0, 4 * np.pi, 801)),
                         tension_N=50e3, contact_model="solved",
                         core_friction=spec)
core = loop.core_friction.model
print(core.loads, core.xi, core.mu_effective, core.stick_EI_Nm2)
```

- Each core is a helix of the declared core lay; the stations of the three
  cores use the same direction-tied Papailiou cap and return map as the armour
  wires, so biaxial histories work
  (`cablexsec.mechanics.core_friction.core_friction_history`).
- Core–core coupling (`CoreFrictionSpec(core_core_coupling=...)`). By default
  (`"absolute"`) every contact of a core, its neighbours included, resists the
  slip of that core against the plane section, as for a helix on a fixed
  cylinder: full slip at `π/2` times the onset curvature. `"relative"` is a
  coupled intralayer slip model of laid cores after Skeie, Sødahl, Pettersen
  and Verde (2023, OMAE2023-104817), who formulated the relative slip of
  three cores, its onset and the coalescence of the slip zones at `π/3`:
  neighbouring cores touch at the same axial position, so the friction
  between them resists their relative slip `u_i − u_j`, while the filler and
  ring contacts keep the absolute slip (the interlayer share and the
  direction-tied cap are this library's). The stations of the cores at one axial position form a
  group whose intralayer friction shares sum to zero; the intralayer share of
  a station is bounded by `σ0 μ_cc (q_cc/q) |sin β| min(ψ, π/n)` (`ψ` from the
  neutral axis), the interlayer share by the Papailiou cap with
  `μ_a = (μ_cf q_cf + μ_cs q_cs)/q`, and the return map is the closest point
  of the group (`cablexsec.mechanics.station_coupling`). With core–core
  contact only (three cores) slip starts at the same onset, the largest
  frozen friction stress is 2/3 and the full-slip moment `√3/2` of the
  absolute values, and full slip comes at `2π/(3√3) = 1.209` times the onset.
  The largest friction stress stops growing earlier, at `π/3 = 1.047` times
  the onset, when the slip front passes `π/3` from the neutral axis; beyond
  it only the moment still grows. Skeie et al.'s Table 1 "full slip"
  curvature (1.049 times the onset in their model, 1.053 in their FE) is this
  saturation, not the coalescence at 1.209 (validation row SK23-T1).
  An independent field solver of the coupled axial equilibrium of the three
  cores (`validation/scripts/coupled_core_slip_reference.py`) gives the same
  onset, full-slip curvature, moment–curvature, friction-force distribution
  and reversal loop, with and without interlayer friction
  (`tests/test_core_coupled_slip.py`). It is a direction-tied law:
  `friction_model="distributed"` raises and `"auto"` keeps it on a non-planar
  path with a `FrictionModelFallbackWarning`. The stick state, the contact
  loads and the onset are those of the default. The default stays
  `"absolute"`; the effect on the shipped cables is in
  [template_66kv.md](template_66kv.md) Section 4.1.
- Friction model of the cores and profiles, as for the armour:
  `core_friction_history(..., friction_model="distributed")` solves each
  population with the distributed helix model of the armour (the axial force
  of a core as a field along its pitch, with the shear-lag interface of `ξ`
  and the slip of every helix node in the direction of its own traction, no
  bending direction), and `"auto"` keeps the direction-tied law on planar
  paths. Under rotating curvature the direction-tied cap turns with the
  bending direction and takes too little friction work (66 kV template, cores
  and profiles at μ 0.46: the distributed law takes 1.3 times the work on a
  random walk and 2.3 times on a circular orbit). `compute_neutral_axes`
  gives the cores the armour's `friction_model` (or `core_friction_model`).
  The distributed form needs a compliant interface (`ξ < 1`, the default with
  the core insulation shear).
- Over a tension history the core and profile contact loads follow the
  tension by default (`core_loads="follow"` of `compute_neutral_axes` and of
  `compute_mkappa` with `contact_tension_history_N`; the loads per sample from
  `core_load_history`), as the armour contact pressure does; `"reference"`
  holds them at the first sample. With the loads held, the section missed the
  slip-scaled exported law by up to 20 % of the moment range under a 10–70 kN
  tension cycle; with them following, by the law's own fixed-tension residual
  (rows CF-D and CF-T of the [benchmark register](benchmarks.md)).
- Normal loads: the solved trefoil contact forces (`contact_model="solved"`)
  or a bounded estimate. In the estimate each core carries its capstan load
  `C` and one third of the line force `R` inside the innermost armour (winding
  stress included) on its outer face; the inward resultant `C + R/3` is
  reacted by the two neighbouring cores of a touching trefoil, whose contact
  forces sum to `(C + R/3)/sin 60°` (default), or radially by a central
  filler (`CoreFrictionSpec(central_filler=True)`). The solved model reads the
  core support of the solved radial state. The loads are not changed by
  bending.
- Filler profiles (`CoreFrictionSpec(filler_profiles=True)`,
  `mu_filler_profile`): the non-structural interstitial filler annulus (a
  structural one is refused: it already bends in `EI_min`) becomes three laid
  profiles (the cores' pitch) that stick and slip like the cores. Their
  effective radius `r_f = √(J_f/A_f)` comes from the area and polar moment of
  the annulus less the core circles, so that the stuck profiles add
  `ξ_f E J_f cos³β_f / 2`; each carries the ring force `R/3` on both faces
  (`2R/3`; solved: the filler–ring and filler–core forces shared among the
  profiles) and has its own Yun `ξ_f` (contact penalty and shear of half the
  mean filler depth). They carry no share of the tension. The exported
  bending law gives them their own slip-scale group, `filler_profiles`.
- Stick coefficient ξ (Yun form): contact penalty `interface_stiffness_Pa_per_m`
  (default rigid) in series with the shear of the polymer layers inside the
  core (`core_shear=True`). With μ → ∞ and a rigid contact the tangent is
  `compute_stiffness(cable).ei_max_Nm2`; with μ = 0 the armour-only loop is
  returned bitwise.
- Full slip is unchanged: friction cannot raise `EI_min`. The cores' own
  bending is already in `EI_min`, at the declared bending modulus.
- Requires one family of laid cores (`core_lay` with a pitch). Cables without
  a core lay, single-core cables and unsupported solved states raise an error.
- Core lay stage of the section properties and the exported bending law:
  `compute_section_properties(..., core_friction=...)` and
  `export_bending_law(..., core_friction=...)`. The cores stick and carry
  their parallel-axis term until slip onset `μ q / (ξ EA_core sin β cos β)`
  (Skeie and Sødahl 2012, Eq. 2.31) and all slip from
  `q expm1(μ sin β π/2) / (ξ EA_core sin² β cos β)` (Papailiou 1997, Eq. 15;
  with `core_core_coupling="relative"` the full-slip curvature of the coupled
  stations, `core_full_slip_curvature`, Skeie et al. 2023);
  both curvatures are reported per case and are nodes of the backbone grid.
  With `CoreFrictionSpec(core_shear=False, strand_slip=False)` the backbone
  starts on `compute_stiffness(cable).ei_max_Nm2`; with the core insulation
  shear it starts below it by `(1 − ξ)` times the core parallel-axis term.
- Also accepted by `xsec_stiffness` and `simulate_rig_test` (station loops
  and secant stiffness). `compute_fatigue` is curvature driven; armour wire
  stresses do not depend on core friction, so it has no such option.
- Comparisons: `validation/scripts/check_core_friction.py` and
  [template_66kv.md](template_66kv.md) Section 4.1. On the published 66 kV
  and Ménard cables the law does not close the gaps (Section 14).

### 6.2 Comparing with a bending test: simulate the test arrangement

`compute_mkappa` is the loop of one section at uniform curvature. A bending
test rarely imposes that: in three-point bending the moment rises linearly
from the supports to the load, in four-point bending it is uniform only
between the loads, and the test reduces a force and a displacement to one
moment and one curvature. Compare a model with a test through the test's own
arrangement and reduction (gap T1):

```python
law = cx.SectionLaw.from_cable(cable, mu=0.2)            # any compute_mkappa options
test = cx.simulate_bending_test(
    law,
    cx.ThreePoint(span_m=0.44),                           # or FourPoint, Cantilever
    cx.BendingProtocol.cyclic("displacement", 0.006, n_cycles=3),
)
print(test.loop_work_J_per_m(), test.cycle_external_work_J())
```

- Stations along the span each carry the section law with their own history;
  the set-ups are statically determinate, so `M(x) = F m(x)` and every
  station is inverted from its own state at every load step. The final
  full-history evaluation gives `equilibrium_residual` (about 1e-11 for all
  friction models).
- Control: `"load"`, `"displacement"` (work-conjugate load-point
  displacement), `"moment"` or `"curvature"` as the test defines it.
  `curvature_definition`: `"station"` (true curvature at the measuring
  station), `"elastic"` (the set-up's elastic formula, `12 w/L²` for
  three-point, Hu et al. 2022 Eq. 2) or `"fit"` (parabola through deflection
  gauges `gauge_x_m`, as Ménard and Cartraud 2023 do with three LVDTs).
- `moment_Nm` is the moment the test reports (`F L/4`, `F a/2`, `F L`);
  `station_moment_Nm` is the law moment at the measuring station.
- Section laws: `SectionLaw.from_cable` (`friction_model` `"direction_tied"`,
  `"progressive"` or `"distributed"`, `core_friction`, any contact model;
  the first two bitwise equal to `compute_mkappa`),
  `SectionLaw.from_conductor` (strand-slip operator, either cap shape),
  `SectionLaw.elastic` or any rate-independent user law.
- Four-point bending under curvature control in the uniform-moment zone
  returns `compute_mkappa` bit for bit there; the arrangement matters for
  three-point and cantilever tests and for any test whose curvature is reduced
  from a deflection.
- Limits: Euler–Bernoulli small deflection on frictionless supports, no shear
  deformation, axial-load (P-delta) effect or self-weight; stations are
  independent sections, so axial redistribution of wire slip between stations
  of different curvature (within a pitch) is not represented unless
  `along_wire=` is given (Section 6.3). Clamped,
  tensioned rig specimens with a prescribed curvature profile are covered by
  `simulate_rig_test` ([rig_test_guide.md](rig_test_guide.md)).
- Benchmarks re-run in their arrangements: `python
  validation/scripts/check_t1_test_arrangement.py`
  ([benchmarks.md](benchmarks.md), gap T1).

### 6.3 Finite specimens: slip redistribution along the wires

A test specimen is one to a few lay lengths long, so a wire passes through
stations of different curvature and ends at a grip, an end fitting or a cut
end. `along_wire=` replaces the independent stations by every helical wire
solved along its own helix through the specimen
(`cablexsec/mechanics/along_wire.py`, gap T2):

```python
# The along-wire solver needs armour friction: a cable with a declared clamping state.
cable = cx.load_cable(cx.inter_array_single_core_yaml_path(), geometry_override=None)
path = 0.2 * np.sin(np.linspace(0.0, 4.0 * np.pi, 41))   # curvature [1/m]
spec = cx.AlongWireSpec(
    ends=("free", "free"),        # or "anchored" (end fitting, grip, clamp)
    overhang_m=(0.3, 0.3),        # straight specimen beyond the supports
    stick="spring",               # "spring" (default), "rigid" or "scaled"
    capstan=True,                 # the package cap's capstan term
    segments_per_pitch=32,
)
test = cx.simulate_bending_test(
    cable, cx.FourPoint(2.5, 1.25), cx.BendingProtocol("curvature", path),
    mu=0.2, curvature_definition="fit", gauge_x_m=(0.9375, 1.25, 1.5625),
    along_wire=spec,
)
print(test.loop_work_J_per_m(), test.along_wire.iterations.max())
```

- **Model.** Each wire: axial force `n = e + du/dφ`, friction line load
  bounded by the package's contact pressure (and capstan term), imposed strain
  `e = κ(x, t) R cos²α cos θ(x)` from the local curvature, ends anchored
  (`u = 0`), elastically anchored (`AlongWireSpec(k_anchor_N_m=(K_left,
  K_right))`, the end node held by `K u` in parallel with its friction;
  Section 5, Step 6) or free (`n = 0`). The station moment is the elastic (full-slip)
  part plus `Σ A R cos α σ cos θ` of the wires at the station; equilibrium
  `M(x) = F m(x)` at every station (zero on overhangs) is solved by Newton
  on the station curvatures and the load with the exact station Jacobian of
  the wires (quadratic convergence; about 3-5 iterations per step).
- **Stick stiffness** is an explicit option, because it sets the slip demand
  near anchors and gradients: `"spring"` is the distributed model's elastic
  interface (`k̂ = ξ/(1 − ξ)`, full imposed strain), `"rigid"` is `ξ = 1`,
  `"scaled"` is a rigid interface with the ξ-scaled strain (the termination
  correction's default). For ξ ≈ 0.2 the slip demand differs by about 1/ξ.
- **Limits.** The infinite-length limit is the distributed sectional law
  (`friction_model="distributed"`); the direction-tied and progressive laws,
  laid-core friction and the Karlsen-Husøy conductor amplitude have no
  along-wire form and are rejected. Conductor rings use
  `AlongWireSection.from_conductor` (Papailiou cap shape, rigid stick, one
  slipping order). Planar bending, axial slip only, contact pressure held at
  the section state.
- **Cost.** One chain per wire per layer per Newton iteration plus the
  Jacobian (one tridiagonal factorisation per wire, one solve per station):
  20-60 s for a 2.5 m Ménard specimen over two loops on one thread.
- **What the benchmarks show** ([benchmarks.md](benchmarks.md), gap T2): the along-wire model
  predicts gradient- and anchor-fed friction forces that the published
  near-bellmouth gauges (Sævik 2011) and short-specimen loops do not show.
  Use it to bound end effects of a test arrangement, not as a validated
  replacement of the sectional law.

The same solver is available for a section near a termination:
`TerminationSpec(..., solver="along_wire", stick="spring", capstan=True)`
(Section 5, Step 6), and on its own for a prescribed, non-proportional
curvature field: `cx.along_wire_history(cx.WireChainLayer.from_layer_spec(spec),
x_m, kappa_x, ends=("anchored", "anchored"))`, with
`anchor_stiffness_N_m=(K_start, K_end)` for elastic anchors.

### 6.4 Local tendon bending, transverse slip and lateral relief (gap G1)

`cablexsec/mechanics/tendon_bending.py` gives the local curvature of a
tendon on the bent pipe and its transverse equilibrium:

- `cx.curvature_coefficients(a, eta)`: normal, lateral and torsion
  coefficients per unit pipe curvature for a transverse slip fraction `eta`
  of the geodesic offset `w_g = r² cos a (1 + sin² a) / sin² a` (Sævik 1992
  Eq. 2.30). `eta = 0` is the loxodrome of the fatigue path (lateral
  `cos a (1 + sin² a)`, normal `cos² a cos 2a` for a tendon without friction
  stress; the stress recovery adds the term of the tendon's slip state,
  Section 5, Step 2), `eta = 1` the geodesic
  (lateral 0, normal `3 cos² a`, Sævik 1992 Eq. 2.32, Out and von Morgen
  1997 Eq. 39; checked against a numerical geodesic on the torus in
  `validation/scripts/check_out_von_morgen.py`).
- `cx.TransverseSlipModel(section, E, G, r, a, tension_N, capacity_N_per_m)`:
  the line load that holds a wire at the loxodrome,
  `q = (N + E I_l k²) k² w_g κ` with `k = sin a / r` (plus a gradient term
  `(4 E I_l k³ + 2 N k) cos a w_g dκ/dx` in the other phase), resisted by
  torsion and normal bending (`η₀ < 1` without friction, Dong et al. 2018
  Eq. 32) and by friction `c = μ (P1+P2)(2πr/n) cos a`. The transverse slip
  is a Coulomb (Jenkin) element per wire; its monotonic amplitude is Sævik
  (1992) Eq. 4.1 without the 4/π of his single-sine estimate.
- `wire_stress_field(..., transverse=model, curvature_gradient=(gx, gy))`:
  lateral curvature `cos a (1 + sin² a) p_l − k² v` and the normal
  curvature with the tangent-rotation term, so `tangential_Pa` (stress at
  the edge) includes the partial slip; `transverse_slip_m` and
  `transverse_load_N_per_m` are returned.
- `cx.LateralRelief(model, anchor_layer_m=...)`: while a wire slides
  axially, the friction vector also carries `q`, so the axial bound is at
  most `sqrt(c² − q²)` and at least `c / sqrt(1 + R_g²)` with
  `R_g = (1 + sin² a)/(sin a cos a)` (vector Coulomb). The factor of the
  largest demand of the history is applied for the whole history, an upper
  estimate of the relief. It is accepted by
  `wire_stress_field(..., friction_model="distributed", relief=...)`, by
  `TerminationSpec(solver="along_wire", lateral_relief=...)` (node-wise
  from the curvature profile, the anchor layer at its floor) and by
  `along_wire_history(..., relief=...)`.
- End restraint: `cx.dong_anchor_tension` (Dong et al. 2018 closed form)
  and `cx.dong_anchor_bending` (Dong et al. 2025 boundary-value problem)
  give the lateral bending at an anchored tendon; both reproduce the
  published examples within 0.1 %. The layer decays over
  `sqrt(E I_l / N*)`, about 0.1 m for a 5 x 12.5 mm tendon at 17 kN.
- **What the benchmarks show** ([benchmarks.md](benchmarks.md), gap G1): on the Sævik (2011)
  riser the transverse demand is 0.2-0.4 of the friction capacity, so the
  relief changes the near-bellmouth ranges by 1-3 %; it does not explain
  why the measured ranges stay near the sectional values. The stuck
  lateral bending of the 12.5 mm tendon over-predicts the neutral-axis
  gauge by about 2.3. Report the lateral component and the relief as
  sensitivities, not as validated quantities.

## 7. Thermal and annulus inputs

### 7.1 Temperature

```python
import numpy as np
import cablexsec as cx

cable = cx.load_cable(cx.inter_array_single_core_yaml_path(), geometry_override=None)
thermal = cx.compute_thermal(cable, current_A=600.0, conductor_area_m2=300e-6)
print(thermal.T_conductor_C, thermal.T_armour_C)

state = cx.solve_section_state(cable, tension_N=50e3, thermal=thermal, T_ref_C=20.0)
wire = cx.compute_stress(cable, 0.02, section_state=state)
components = cx.compute_component_response(cable, 0.02, section_state=state)
```

`compute_thermal` takes `ambient_C` and `T4_external_Km_W` from the
`environment` block unless passed. The section state applies each layer's free
thermal strain and the registered polymer E(T) laws within their source
ranges, with one temperature per annulus; `temperature_C=` (a value or a
mapping covering every layer) replaces the profile. On this path `sigma_A_Pa`
already contains the thermal stress and `sigma_T_Pa` is zero. Do not combine it
with the simpler fixed-strain routes: `compute_stress(..., thermal=thermal)`
and a `temperature_C` column in a `LoadCaseSeries` (accepted only with direct
strain) both add `-E α (T - T_ref)` at fixed global strain.

### 7.2 Flooded or sealed annulus

The annulus condition enters through two choices you make explicitly:

1. **Clamping pressure.** In a flooded annulus seawater reaches the armour, so
   hydrostatic pressure adds no net clamping: use `p_ext_Pa=0.0`. In a sealed
   (dry) annulus the external pressure acts on the outer sheath and clamps the
   armour: use the hydrostatic pressure at the section. `p_ext_Pa=None` falls
   back to `initial_state.external_pressure_Pa` of the YAML file.
2. **S–N curve.** Flooded armour: a seawater curve such as
   `saevik_2011_flat_steel`. Sealed and intact: an in-air curve such as
   `oc2017_armour_in_air`.

Continuing from the block above:

```python
import dataclasses
from cablexsec.analysis.fatigue import compute_fatigue as fatigue_from_layers
from cablexsec.cable_adapters import cable_to_fatigue_layers
from cablexsec.materials import sn_curves

p_hydro = 1025.0 * 9.81 * 100.0                  # 100 m water depth
flooded = cx.solve_section_state(cable, tension_N=50e3, p_ext_Pa=0.0, thermal=thermal)
sealed = cx.solve_section_state(cable, tension_N=50e3, p_ext_Pa=p_hydro, thermal=thermal)

t = np.linspace(0.0, 120.0, 1201)
kx, ky = 0.01 * np.sin(t), 0.004 * np.cos(t)

# Flooded: seawater curve from the YAML file, no hydrostatic clamping.
wet = cx.compute_fatigue(cable, kx, ky, section_states=[flooded] * t.size)

# Sealed: hydrostatic clamping and the in-air curve.
dry = sn_curves.get_sn_curve("oc2017_armour_in_air")
layers = [
    dataclasses.replace(layer, sn_bulk=dry)
    for layer in cable_to_fatigue_layers(
        cable, contact_tension_N=sealed.tension_N, contact_model="qp", section_state=sealed)
]
normal = {l.name: np.full(t.size, sealed.layers[l.name].longitudinal_stress_Pa) for l in layers}
sealed_result = fatigue_from_layers(layers, kx, ky, normal_stress_Pa=normal)
print(max(d.max() for d in wet.D_bulk_per_wire.values()),
      max(d.max() for d in sealed_result.D_bulk_per_wire.values()))
```

`section_states` takes one state per sample (repeat the same object for
constant conditions) and keeps the solved normal stress, moduli and pressures.
Alternatively, set `bulk_sn: oc2017_armour_in_air` in a copy of the YAML file
and pass `section_states=[sealed] * t.size` to `cx.compute_fatigue`.
Higher clamping raises the friction cap: wires stay in stick to larger
curvature and friction stress ranges change. Whether damage rises or falls
depends on the history; compute both cases.

`contact_model="bounded"` (the default of the stress, bending and fatigue
functions) is the fixed-radius pressure-transfer chain and supports offset
cores. `"qp"` (the default of `solve_section_state`) is the coupled concentric
equilibrium with Poisson effects and contact opening; it rejects offset,
nested-strand and corrugated geometry.

### 7.3 Temperature histories (energised analyses)

Sections 7.1 and 7.2 use one temperature per run. For a cable that carries a
varying current, `cx.ThermalHistory` holds the layer temperatures on the slow
time scale and couples them one way into clamping, friction and fatigue
(`cablexsec/analysis/thermal_coupling.py`). The option is opt-in: without `thermal=` every result is unchanged.

```python
import numpy as np
import cablexsec as cx

cable = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)

# Slow time: a daily current cycle through the transient IEC 60287 ladder,
# in its periodic regime (24 one-hour blocks).
tau = np.linspace(0.0, 86400.0, 25)
current = 0.5 * 1093.0 * (1.0 - np.cos(2 * np.pi * tau / 86400.0))
history = cx.ThermalHistory.from_current(cable, tau, current, periodic=True)

# Fast time: one sea-state record evaluated in every block.
t = np.arange(0.0, 2000.0)
kx = 0.004 * np.sin(2 * np.pi * t / 10)
ky = 0.002 * np.sin(2 * np.pi * t / 13 + 0.4)
tension = 50e3 + 5e3 * np.sin(2 * np.pi * t / 10)
result = cx.compute_thermal_fatigue(
    cable, kx, ky, history, tension_N=tension, record_duration_s=2000.0
)
print(result.summary())        # every ratio with its S-N slope
print(result.damage_max, result.damage_ratio_fixed_mean, result.pi1)

# One frozen state (steady rated current) through the ordinary entry point.
hot = cx.ThermalHistory.steady(cable, 1093.0)
single = cx.compute_fatigue(cable, kx, ky, tension_N=tension, thermal=hot)
```

What happens, per slow-time block:

1. **Temperatures.** `from_current` builds the IEC ladder of
   `compute_thermal` as a radial RC network (`cx.thermal_network`): the same
   annuli, `T1` split and filler annulus, sub-annuli of equal logarithmic
   width, heat capacities of every layer including armour wires and shells,
   the external `T4` as a series resistance. `R_ac(T_c)` is solved with the
   temperatures unless `resistance_temperature_C` fixes it (the
   `compute_thermal` convention). Integration is TR-BDF2 (L-stable, second
   order). `steady`, `uniform` (a heated rig) and user layer temperatures are
   the alternatives.
2. **Clamping.** Free thermal strains and polymer `E(T)` enter the solved
   radial contact state (`contact_model="solved"`, required with
   `thermal=`); its `P1+P2` sets the friction caps and its crossing reaction
   the crossed-wire load.
3. **Stiffness.** Registered `E(T)` laws (`xlpe_qiao2025`, `hdpe`; [Qiao2025]
   Eq. 9 after [Yan2021], 10–90 °C, no extrapolation) change the polymer
   moduli and, through the Yun coefficient, the stick stiffness `k_f`.
   Materials without a law keep their modulus; `modulus_transfer=
   {"mdpe_janocha2024": "hdpe", ...}` scales one by a registered law's ratio
   `E(T)/E(T_ref)` (an assumption that the result records). The friction
   coefficient and the steel modulus do not change.
4. **Fatigue.** Within a block the pressure, caps, stick stiffness and the
   thermal wire stress `sigma_T` (solved wire stress at the same tension minus
   the reference) are constant; the fast record is rainflow counted with them.
   `compute_thermal_fatigue` repeats the record in every block and weights the
   damage by block duration. In one long record, `compute_fatigue(...,
   thermal=history, time_s=t)` assigns each sample to its block and carries
   the slip state across block boundaries.

Diagnostics returned: `Pi1` (wave period over the network time constant),
`Pi2` per block (frictional work rate over the heat generated; the work is
that of the distributed friction law, Section 9), `Pi3` per
layer (curvature range over slip-onset curvature), `Pi4` and `Pi5` per armour
layer (`cx.armour_thermal_state`), the damage at the time-mean temperature
(`fixed_mean`, the fixed-temperature assumption), the unenergised
reference, and the daily range of `sigma_T` with the damage of that one slow
cycle (`D_thermal_cycle`, not combined with the bending extremes).

Stress-free reference temperature. Free thermal strain is
`alpha (T - T_ref)`, and every layer at `T_ref` is the unenergised analysis
with the library moduli. `ThermalHistory.steady` and `from_current` default
`T_ref` to the ambient of their network (the cable stress-free as installed),
so 0 A is the unenergised state bit for bit; `uniform` and `isothermal` (a
heated rig) default to 20 °C, the laboratory ambient and the temperature of
the library moduli. Pass `T_ref_C=` to choose; `history.T_ref_basis` records
the choice and the provenance reports it. The choice is not minor: the 66 kV
template run at `T_ref` = 20 °C, 5 K above its 15 °C ambient, puts the 0 A
state 5 K below the stress-free state, so the shrinking outer sheath lifts the
cold inner-armour clamping from 31 to 52 kPa and the cold-to-rated damage
factor is 115 rather than 310 ([template_66kv.md](template_66kv.md)
Section 4.2.1). Registered `E(T)` laws scale the library modulus by
`E(T)/E(T_ref)` when `T_ref` is not 20 °C, so the modulus is continuous at
the reference.

Polymer relaxation. The polymer moduli are time independent; hot
polyethylene and XLPE relax within hours to days. `polymer_relaxation_factor`
`f` (0 < f ≤ 1, on every builder) multiplies the polymer moduli of every
radial solve, the reference included, as a relaxed-modulus proxy for the
sustained clamping and thermal stresses; the stick stiffness of the fast
bending keeps the instantaneous modulus. `compute_thermal_fatigue` reports
the reference damage ratio at the history's factor and, when that is below
one, unrelaxed; `relaxation_factors=(0.05, ...)` adds further bounds
(`result.relaxation_damage_ratio`). A polymer layer without an `E(T)` law or
transfer above 40 °C raises `cx.UnmodelledPolymerSofteningWarning`.

Ratios. Within a block the thermal wire stress is the solved wire stress at
each sample's tension minus the reference at that tension, so it varies with
tension through the block. Damage ratios between thermal states scale with a
power of the friction stress range: they depend steeply on the S–N slope (for
the 66 kV template, m = 3 to 6 gives 35 to 1300 for the reported 115) and on
the cold clamping (Section 4.4). `result.summary()`, `result.sn_slope` and the
provenance give the slope of the governing layer with every ratio.

Rules and limits:

- All layers at `T_ref` (`ThermalHistory.isothermal`) reproduce the
  unenergised solved results bit for bit (without relaxation).
- The thermal network has no armour loss `λ2`, no sheath eddy loss `λ1″` and
  no IEC 60287-2-1 three-core factor (the full IEC texts are not held); a
  three-core filler is the equal-filler-area annulus; `T4` has no heat
  capacity, so a buried cable responds faster than in reality.
- Cooling below `T_ref` contracts the outer sheath onto the armour while the
  inner polymers shrink away from it; heating does the opposite. Both can
  raise `P1+P2`; the net sign depends on the construction and on whether
  softening is included. Compute the cases you need rather than assume a
  sign.
- The distributed friction model needs a constant stick modulus: use one
  `ThermalState` per call when `E(T)` changes `k_f`.
- `simulate_rig_test(..., thermal=...)` accepts the same histories for an
  energised rig ([rig_test_guide.md](rig_test_guide.md)).

## 8. Capacity screening

```python
import numpy as np
import cablexsec as cx
from cablexsec.materials import library

cable = cx.load_cable(cx.inter_array_single_core_yaml_path(), geometry_override=None)
tensions = np.linspace(0.0, 400e3, 5)
capacity = cx.compute_capacity(cable, tensions)
print(capacity.kappa_max_1_m, capacity.governing, capacity.unavailable_mechanisms)
E = library.get_property("galvanised_steel_armour", "E_Pa")
alpha = library.get_property("galvanised_steel_armour", "alpha_T_K")
thermal = cx.compute_thermal(cable, current_A=600.0, conductor_area_m2=300e-6)
sigma_T = -E * alpha * (thermal.T_armour_C - 20.0)      # restrained, compressive when hot
hot = cx.compute_capacity(cable, tensions, sigma_T_armour_Pa=sigma_T)
detail = cx.birdcage_buckling_detail(cable, cable.get_layer("outer_armour"))

# Corrugated aluminium sheath of the 220 kV topology example
export = cx.load_cable(cx.export_three_core_yaml_path(), geometry_override=None)
wall = cx.compute_shell_stress(export, "al_metallic_sheath", 0.01, 1e-4,
                               p_ext_Pa=1e6, delta_T_K=30.0)
sheath = cx.compute_shell_fatigue(export, "al_metallic_sheath",
                                  0.002 * np.sin(np.linspace(0, 40 * np.pi, 2001)))
print(wall.sigma_total_Pa, sheath.D_miner)
```

The envelope is the inner boundary of usage factor 1 over the mechanisms that
can be evaluated: wire first yield (same slip-front stress as bending and
fatigue), armour compression instability (armour in net compression only;
lateral slip assumed, see [uls_guide.md](uls_guide.md) for the full screen
with friction gate, sheath strength and design factor), conductor fracture,
and polymer crushing. Polymer crushing stays unavailable
(`NaN`, never governing) unless you pass `polymer_crush_allowable_Pa` together
with `polymer_crush_source`.

Sheath damage scales with SCF⁷ on the m = 7 curve, so with the template's
assumed corrugation factor and SCF the result demonstrates the calculation
only. The elastic route of `compute_shell_fatigue` refuses a history whose
root stress exceeds yield, including from a large mean curvature; the
strain-life route (`method="strain_life"`, Section 13.3) accepts it.

### 8.1 Capacity check of global time series

A global history (time-history CSV, RIFLEX or CableDyn) is placed sample by sample on
the tension–curvature envelope: a utilisation time series against the capacity
curve and its statistical extreme (utilisation levels as in Skeie et al. 2012,
Fig. 11).

```python
import numpy as np
import cablexsec as cx

cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
envelope = cx.build_capacity_envelope(
    cable, np.linspace(-2.0e4, 1.5e5, 18),        # must span the history
    utilisation_factor=cx.ARMOUR_UTILISATION_FACTORS["normal"],
    mbr_m=1.2, mbr_source="[ModelAssumption] example MBR",
)
t = np.arange(0.0, 600.0, 0.1)
kx, ky = 0.3 * np.sin(0.4 * t), 0.1 * np.cos(0.3 * t)
check = cx.check_capacity(envelope, t, 6.0e4 + 2.0e4 * np.sin(0.05 * t), kx, ky,
                          thresholds=(0.8, 1.0), blocks=10)
print(check.max_utilisation, check.time_of_max_s, check.governing_at_max)
print(check.exceedances[1], check.mbr.utilisation_max, check.block_extremes)
```

- The envelope holds, per tension, the curvatures at which the largest wire
  usage `max|σ|/σ_y` (the slip-front stress of `wire_yield_curvature`)
  reaches a set of design utilisation levels (default 0.1–1.0 and `1/η`), the
  usage at zero curvature, the lateral-buckling utilisation above first slip
  and the radial-buckling, conductor and (sourced) polymer-crush
  utilisations. The level at `η` is the design capacity curve
  (`kappa_capacity_1_m`, governing mechanism per tension). An imported curve
  (`CapacityEnvelope.from_points`, with its source) uses the curvature ratio.
- The utilisation of a sample is interpolated linearly in curvature between
  levels and in tension between grid points; the governing part (wire stress,
  lateral buckling, tension mechanisms) is kept per sample. Samples beyond
  the highest level are extrapolated and counted; tensions outside the grid
  raise unless `out_of_range="clamp"` (then counted).
- Reported: the utilisation history, its maximum and time, the tension and
  capacity curvature there, exceedance samples, up-crossings and time above
  each threshold, the minimum bend radius check (`κ·MBR`, default the cable's
  `MBR_dynamic_m`), the largest torque when given (no torsion capacity), and
  with `blocks` a Gumbel fit of block maxima moved to the record duration.
- `capacity_extremes(results)` fits a Gumbel distribution (method of
  moments) to the maxima of seeds of one condition: most probable maximum,
  mean and a fractile. `check_capacity_series` takes a `LoadCaseSeries`
  (declared tension kind) or a CableDyn section history.
- In the sea-state batch, `run_sea_state_batch(..., capacity=
  cx.CapacityBatchCheck(envelope, groups={...}))` checks every section and
  run, reports the overall maximum and a Gumbel fit per group of seeds, and
  writes `<section>_<run>_utilisation.npz`; fatigue results are unchanged.
- The envelope treats the section as axisymmetric (curvature magnitude) and
  each sample quasi-statically; the slip history is not carried. Its tension
  grid refers to one tension kind (`tension_kind`), checked against the
  history.

### 8.2 Cable over a sheave

`cx.compute_sheave(cable, sheave_diameter_m=6.0, tension_N=50e3)` treats a
cable pulled over a sheave or chute during installation and handling. The
diameter is at the cable centre (`diameter_basis="pitch"`, as in Feyrer 2015)
or at the groove root (`"groove_root"`, the cable diameter is added). The
curvature on the contact bow is `2/D`.

- Groove loads (Feyrer 2015, Sections 3.1.6 and 3.1.7): line pressure
  `q0 = 2S/D`, global pressure `p0 = 2S/(D d)`, peak pressure of the cosine
  groove distribution and the groove friction factor for the groove angles
  (`groove_angles_deg`, default 0 and 60 deg, a groove that is not undercut:
  1.351 and 1.170). `cx.groove_pressure` returns the two factors.
- Run-on contact: angle `theta_0 = arctan((2/D) sqrt(EI/S))` and force
  `Q = S sin theta_0`, bracketed by the full-slip and no-slip bending
  stiffness of the section (`run_on_angle_rad`, `run_on_force_N`).
  `cx.sheave_contact` is the tape solution alone. The solution assumes a small
  run-on angle; `run_on_small_angle` is `False` when `tan theta_0 > 0.1`, which
  a stiff cable on a small sheave at low tension does not satisfy (the 66 kV
  template at 50 kN on 4 and 6 m sheaves is flagged).
- Armour: wire stresses at the sheave curvature (`compute_stress`) and the
  damage of one passage, straight to the sheave and back at constant tension
  (`compute_fatigue`); `passes_allowed` is its inverse for the most damaged
  layer, without a design fatigue factor. `mbr_ratio` is the sheave radius
  over `MBR_installation_m` when the cable gives one.
- Capacity: `design_utilisation` is the utilisation of the sheave curvature on
  the capacity envelope of Section 8.1 (`build_capacity_envelope`: the same
  wire-stress levels, buckling and tension mechanisms, bounded contact chain
  and design utilisation factor, default `ARMOUR_UTILISATION_FACTORS["normal"]`,
  argument `utilisation_factor`), with `design_curvature_capacity_1_m`,
  `utilisation_factor` and `governing_mechanism`; pass `capacity_envelope=`
  to reuse the envelope of a capacity check, so the two agree exactly. The
  characteristic ratio `kappa / kappa_max(S)` of `compute_capacity` (design
  factor one) stays as `characteristic_utilisation` and
  `characteristic_curvature_capacity_1_m` (the former `utilisation` and
  `curvature_capacity_1_m`, kept as deprecated aliases). The two differ by
  more than the design factor because wire stress is not proportional to
  curvature: the 66 kV template at 50 kN on a 4 m sheave has a characteristic
  ratio of 0.448 and a design utilisation of 0.701 (stress usage 0.470 with
  the design factor 0.67).
- Not included: the groove pressure is reported for the sheath and is not
  applied to the armour contact; the measured run-on pressure peaks of steel
  wire ropes (Häberle 1995) are not transferred to cables; no tension change
  over the wrap from sheave friction and no crushing of the section. No
  measured cable-over-sheave data are held; gate SH in
  [benchmarks.md](benchmarks.md) checks Feyrer's worked examples and the
  internal consistency.

## 9. Performance and compiled kernels

Two numba kernels carry the cost of the default fatigue route (the
distributed, progressive, along-wire and conductor operators have compiled
kernels of their own; the user mean-stress and SCF options count cycles with
the compiled `rainflow_table` and sum them in NumPy, which is slower than the
default route). `cablexsec/mechanics/fast_slip.py`
runs the per-wire friction-stress recursion (Papailiou cap and elastic–Coulomb
return map) inside `wire_stress_field` when the stick stiffness is constant.
`cablexsec/analysis/fast_rainflow.py` counts and sums Miner damage at every
fixed fibre in one pass (`fibre_fatigue`, `fatigue_rows`) without storing fibre
histories.

On one thread the Step 3 path takes 0.76 s per section-hour for 118 wires × 32
fibres and 42 001 samples (600 s initialisation plus a 3600 s retained hour);
`parallel=True` spreads wires over threads.
`compute_fatigue` uses the compiled slip kernel, assembles the stress field in
one compiled pass (`fast_field.assemble_wire_field`) and counts every fixed
fibre with `fast_rainflow.fibre_miner_damage`, the compiled counter applied to
each fibre history as it is formed; its damage is bitwise equal to the
reference counter, still available with `rainflow_counter="reference"`.
`validation/scripts/benchmark_throughput.py` records the one-core throughput
of this route on a one-hour sea state of the 66 kV template.

The pure-Python routines remain the reference implementation.
`tests/test_fast_slip.py` and `tests/test_fast_rainflow.py` require the
compiled kernels to reproduce them cycle for cycle; damage is identical. The
first call compiles and caches the kernels; without numba the same functions
run uncompiled.

### Choosing the armour friction model

`compute_fatigue` (and `wire_stress_field`) take `friction_model`:

- `"direction_tied"` (default): each wire is an independent spring–slider with
  the ξ-reduced stick stiffness and the Papailiou bounds of the current bending
  direction (held below `0.1 m κ_on`, `direction_hold_fraction`,
  `m = 1 − 3λ²/8`, `λ = μ |sin α|`).

  **Planned change of the default.** The default of `compute_fatigue` becomes
  `"auto"` (below) in a later release. The two are the same, bit for bit, on
  a planar curvature path, so only non-planar paths change. Until then, left
  at the default on a non-planar path (out-of-plane ratio above
  `PLANAR_PATH_TOLERANCE`), `compute_fatigue` warns once per call
  (`DirectionTiedFrictionWarning`, a `FutureWarning`, with the measured bias
  below); the result of this release is unchanged. Pass `friction_model`
  explicitly to keep the current behaviour (`"direction_tied"`) or to take the
  coming one (`"auto"`) without the warning. The sea-state batch takes the
  same default (`SeaStateBatchOptions.friction_model=None`) and warns the same
  way, from its worker processes too; its manifest records the model the
  default resolves to.
- `"auto"` (opt-in): the direction-tied law for a planar curvature path and
  the distributed law for any other. The path is planar when its out-of-plane ratio
  (`curvature_path_out_of_plane_ratio`: the largest distance of a sample
  `(κ_x, κ_y)` from the principal line through the origin, over the smaller of
  the largest curvature and the largest distance of a sample from the mean
  curvature, the latter not below `1e-2` of the largest curvature) is at most
  `PLANAR_PATH_TOLERANCE = 1e-3`. A planar path in any plane gives round-off,
  a static offset in that plane included, the same path written with six
  significant digits less than `1e-5`, an ellipse its axis ratio. The second
  scale keeps a static bend from hiding cycles normal to it: a static
  curvature of 0.05 1/m with cycles of 4e-5 1/m normal to it is 8e-4 out of
  plane against the largest curvature, but 0.08 against the floor of its range
  (cycles of 4e-4 1/m: 0.8). The floor keeps a nearly constant curvature
  written with six significant digits planar (at most 7.1e-4).
  A layer the distributed law does not cover (rigid interface `ξ ≥ 1`, a
  stick modulus that changes over the record, a rate-dependent friction law)
  keeps the direction-tied law and warns (`FrictionModelFallbackWarning`, once
  per layer). The provenance records the ratio, the model of every layer and
  the reason for a fallback. `wire_stress_field` takes `"auto"` too.
- `"distributed"`: the friction force is solved along one pitch of the helix
  with interface shear stiffness `k̂ = ξ/(1 − ξ)` and the capstan bound
  `λ n` on the signed wire force, for any biaxial curvature history and the
  same clamping history. Stick zones then anchor the force distribution, so
  the result does not depend on a bending direction. `distributed_bound=
  "uniform"` drops the capstan term (`b = ρ g`, the linear Coulomb bound of the
  clamping, for sensitivity studies); the default `"signed"` is the bound
  above. A capstan term on the total wire force (bending plus tension share)
  is not offered: the clamping `P1+P2` that scales `g` already contains the
  radial load of the tension share.

In every law the capstan term acts on the wire's own force (Papailiou 1997,
Eq. 14 with the signed helix angle): a rising wire stress is bounded by
`σ0 expm1(λψ)`, a falling one by `−σ0 (1 − exp(−λψ))`. A full reversal of a
slipping wire therefore spans `2 σ0 sinh(λψ)`, which differs from the linear
Coulomb range `2 σ0 λψ` only at third order in `λ`; the peak tensile friction
stress is `σ0 expm1(λπ/2)`. Releases before the correction used the tension
bound on both sides, which overstated the extreme-fibre friction range by the
factor `2/(1 + exp(−λπ/2))` (+2.8 to +6.7 % for the bundled power-cable
templates, `λ` 0.035 to 0.085) and the full-slip loop work by about
`(π − 2)λ/2`.
- `"progressive"` (gap G8): each wire of the direction-tied law becomes a set
  of parallel spring–sliders (an Iwan population) with the same stick
  stiffness, whose bounds average to the same Papailiou bounds. The default
  density, `ProgressiveFriction(density="continuity")`, has no parameter: it
  is the friction-continuity slip law of Sævik (1992), Kim (2020) and
  Lukassen (2019), in which a stuck wire carries the stress of the slip
  boundary plus its elastic increment, so each wire leaves stick gradually
  instead of at one curvature. `density="uniform", spread=h` spreads the caps
  uniformly by ±h around the cap, for sensitivity studies only. Stick
  stiffness, onset curvature and full-slip moment are those of the default
  law; one sub-element (`n_elements=1`) or `spread=0` reproduces it bit for
  bit. The same option exists in `compute_mkappa`.

```python
import numpy as np
import cablexsec as cx
from cablexsec import compute_fatigue

cable = cx.load_cable(cx.menard2023_yaml_path())
t = np.linspace(0.0, 20.0, 401)
kx, ky, eps = 0.05 * np.sin(t), 0.02 * np.cos(t), np.zeros_like(t)
kappa = np.linspace(0.0, 0.1, 41)
res = compute_fatigue(cable, kx, ky, axial_strain=eps,
                      friction_model="distributed",
                      distributed_nodes_per_pitch=None,   # default 4 per wire
                      distributed_max_increment=0.1)      # in units of κ_on
print(res.provenance)  # records the model, N, k̂ and λ of each layer
```

`distributed_nodes_per_pitch` may be one integer or a dictionary per layer;
multiples of the wire count avoid interpolation. On the companion article's 30
section-hours (three sea states, five sections, two layers) the governing
damage changed by at most 0.15% between two and eight nodes per wire and by at
most 0.03% between four and eight; at the reference arc, substeps of 0.03
`κ_on` instead of 0.1 changed it by less than 1e-6 (relative). Check both
settings for a new history. The model requires
a constant stick modulus (a section-state history that changes ξ or the wire
modulus is rejected; a constant one other than the layer's, as in one block
of a thermal history, enters as the Yun coefficient `ξ = k/(E cos²α)` of that
state) and `0 < ξ < 1`.

The two laws coincide below slip onset and in planar full slip. In planar
partial slip the distributed law gives smaller stress ranges because the
elastic interface spreads the slip front: for the Ménard inner armour the
largest per-wire range is 0.95, 0.84, 0.89, 0.97 and 0.99 of the default
law at `κ/κ_on` = 1.2, 1.5, 3, 10 and 30. On the companion article's reference hours the
distributed total armour damage is 0.44–1.15 of the default law, depending on
the section; at the 263.478 m arc it is 0.62 (U = 10.74 m/s). These are
model-to-model differences, not a validation; no measured armour strain under
biaxial bending is available. Cost grows with the curvature increment per
sample in units of `κ_on` (the substeps are at most 0.1 `κ_on`). On one
thread, the distributed law took 1.0 to 1.5 times the direction-tied stress
and damage time on the shipped fatigue examples (inter-array single-core
cable, `examples/demo_05_fatigue_damage.py`, the long-term examples and the
release profile) and 13 to 16 times on the 66 kV template (an hour sampled at
10 Hz at 0.004 1/m: 43.6 s against 3.2 s; the 2000 s record of
[template_66kv.md](template_66kv.md): 36 s against 2.2 s). Planar records keep
the direction-tied cost under `"auto"`.

**Rotating curvature.** The direction-tied cap is the friction stress
accumulated from the current neutral axis, so it turns with the bending
direction. On a circular orbit a
wire passes the extreme fibre, where its cap is largest, while its drive is
stationary, and it slips mostly near the neutral axis, where its cap is close
to zero. In the distributed law every helix node slips at its own traction
bound. For an orbit of radius `R` much larger than the slip onset, the work of
one revolution over that of the two planar cycles of amplitude `R` (about the
two axes) is `π/4` in the distributed law for any `λ = μ |sin α|` (per
revolution every node slips through `4R`, against `4R |sin φ|` in a planar
cycle) and `π (sinh x₀ − λ)/(4 λ cosh x₀)`, `x₀ = λπ/2`, in the
direction-tied law with its signed bounds, which is `π(π/2 − 1)/4 = 0.448`
up to `O(λ²)`; `tests/test_orbit_friction_heat.py` checks both limits. On the 66 kV template (both armour layers,
100 kN):

| Circular orbit radius / `κ_on` | 2 | 4 | 8 | 16 | 32 | large (closed form) |
| --- | --- | --- | --- | --- | --- | --- |
| direction-tied work / distributed work | 0.16 | 0.29 | 0.39 | 0.47 | 0.51 | 0.57 |
| largest per-wire stress range, direction-tied / distributed | 0.92–0.94 | 0.97–0.98 | 0.99–1.00 | 1.00–1.01 | 1.01 | |

A planar cycle of the same amplitudes dissipates the same work in both laws
within 3 % (at large amplitude the direction-tied work is `sinh(x₀)/x₀`
times the distributed one, 1.0005–1.0008 here), and a figure-eight orbit
(`κ_y` at twice the frequency of `κ_x`) 0.66–0.78 of the distributed work.

The stress ranges also fall short on near-circular paths, and more so on a
stiffer interface, because the distributed law's stick zones then hold more
of the force. Largest per-wire friction-stress range of the inner armour of
the 66 kV template (100 kN), direction-tied over distributed, in the last of
four revolutions of an ellipse centred on the straight cable (the circle row
of the template interface is the table above; a stiffer interface widens the
gap):

| Minor / major axis | ξ | 1.5 `κ_on` | 2 `κ_on` | 4 `κ_on` | 8 `κ_on` |
| --- | --- | --- | --- | --- | --- |
| 1 (circle) | 0.81 (template) | 0.937 | 0.943 | 0.981 | 1.001 |
| 1 (circle) | 0.99 | 0.854 | 0.863 | 0.933 | 0.970 |
| 0.8 | 0.81 | 1.021 | 1.056 | 1.057 | 1.039 |
| 0.8 | 0.99 | 0.933 | 0.976 | 1.011 | 1.009 |
| 0.5 | 0.81 | 1.093 | 1.080 | 1.050 | 1.031 |
| 0.5 | 0.99 | 0.984 | 1.023 | 1.011 | 1.007 |
| 0.3 | 0.81 | 1.121 | 1.068 | 1.041 | 1.025 |
| 0.3 | 0.99 | 1.009 | 1.016 | 1.010 | 1.006 |
| 0 (planar) | 0.81 | 1.117 | 1.080 | 1.023 | 1.010 |
| 0 (planar) | 0.99 | 1.019 | 1.008 | 1.001 | 1.001 |

At an S–N slope of 4 the circle at twice the onset gives 0.79 (ξ = 0.81) and
0.55 (ξ = 0.99) of the distributed bulk damage where friction governs the
range. The friction work of one revolution is 0.38 to 0.69 of the planar cycle
of the same amplitude in the direction-tied law and 1.8 to 2.6 in the
distributed law (ξ = 0.99, two to eight times the onset). Flat paths
(minor over major axis up to about 0.3, and planar ones) give direction-tied
ranges at or above the distributed ones, so the switch of `"auto"` at the
planarity tolerance moves a result from the direction-tied value towards the
lower distributed one (at most the planar rows above). On the shipped
examples `"auto"` changes the largest damage by at most 0.02 %; on the 66 kV
hour above it gives 0.92 of the direction-tied value, and on the
[template_66kv.md](template_66kv.md) record 0.99
(`tests/test_biaxial_friction_default.py`).

**Pitch mean of the direction-tied law.** Its bounds integrate the capstan
growth from the neutral axis, where the friction stress is zero (Papailiou
1997, Eq. 14). In planar full slip the wire at the azimuth `x` from the
neutral axis toward tension carries `σ0 (exp(λx) − 1)`, whose pitch mean is
`σ0 (sinh(x₀)/x₀ − 1)`, `x₀ = λπ/2`: `λπ/12` of the peak tensile friction
stress to leading order (1.1 % on the 66 kV inner armour, 0.054 MPa; 0.9 % on
its outer armour). A periodic friction field of a uniformly bent pitch has zero
mean, as in the distributed law (`σ0 (C exp(λx) − 1)`). The offset is kept,
because a zero-mean condition couples the wires of a layer, which the
independent spring–sliders of this law do not. Removing it by a uniform shift
would leave the moment unchanged (round-off), change the largest per-wire
stress range by less than `1e-4` (planar, one-sided and rotating cycles from
two to 32 times the onset) and lower the mean stress of every wire by the
offset.

The frictional heat is affected the most: `compute_thermal_fatigue` takes the frictional heat of its `Pi2`
diagnostic from the distributed law whatever `friction_model` the fatigue
uses (64 nodes per pitch and substeps of `κ_on`, within 0.25 % of the
fatigue-path settings at about a hundredth of their cost;
`cablexsec.analysis.thermal_coupling.record_friction_work`). A layer with a
rigid interface (`ξ = 1`), which the distributed law does not cover, keeps the
direction-tied work, and the provenance says so. `compute_mkappa` and the
cyclic bending of `simulate_frictional_heating` are planar, where the two
works agree.

```python
from cablexsec import ProgressiveFriction, compute_fatigue, compute_mkappa
res = compute_fatigue(cable, kx, ky, axial_strain=eps,
                      friction_model="progressive",
                      progressive=ProgressiveFriction(n_elements=16))
loop = compute_mkappa(cable, kappa, friction_model="progressive")
```

The progressive (continuity) law is the rigid-interface limit of the
distributed law in planar bending. For the Ménard inner armour its largest
per-wire stress range is 0.83, 0.82, 0.88, 0.95 and 1.00 of the default law
at `κ/κ_on` = 1.5, 2, 3, 5 and 10 (distributed: 0.84, 0.83, 0.89, 0.94,
0.97); it keeps the held bending direction of the default law. Cost on one
thread was 1.5 times the default stress and damage time at 16 sub-elements
(1.2 times at 8) on a 36 000-sample Ménard history, against 3.8 times for the
distributed law on the same (shared) host. On the published loops it changes
little, because their amplitudes are 30–45 times the onset curvature: the
gradual transitions measured by Witz (1996) and Ménard and Cartraud (2023)
are not reproduced by any friction-law mechanism at its physical magnitude
([benchmarks.md](benchmarks.md), gap G8). Use it to bound the partial-slip
stress range, not as a validated loop model.

## 10. Stranded conductor: strand slip and trellis fretting

`compute_conductor_fatigue` assesses a `cylindrical_offset` conductor with
resolved `strand_layers`. Each wire ring receives the bilinear stick–slip
operator of Karlsen-Husøy et al. (2026, Eqs. 2–10), evaluated at eight fixed
azimuths:

- stick slope `R cos²α + (d/2) cos²α cos 2α` and slip slope
  `(d/2) cos²α cos 2α` (outer fibre of the wire);
- friction strain amplitude `π R τ / (2 sin α A E)` (Nasution et al. 2014,
  Eq. 7), with the friction force per unit wire length `τ` from the inward
  helix loads of the rings outside each interface (their Eq. 9 as read here)
  and an optional external pressure on the conductor surface;
- scaling with `|T|/T0` and with `|cos φ|`, the angle to the curvature
  resultant (Eq. 4); the direction is held while the curvature passes zero.

The general operator is `mechanics.strand_slip.strand_slip_strain`; it takes
any number of orders, so the trilinear parameters of a local analysis
(Karlsen-Husøy Table 4: `StrandSlipParameters.from_karlsen_husoy`) can be used
directly.

```python
import numpy as np
import cablexsec as cx

cable = cx.iea_task49_66kv_stranded()   # 61-wire conductor, designed stranding
t = np.linspace(0.0, 20.0 * np.pi, 2001)
kx, ky = 0.05 * np.sin(t), 0.01 * np.cos(t)
tension = 2.0e4 + 5.0e3 * np.sin(t)
result = cx.compute_conductor_fatigue(cable, kx, ky, tension_N=tension, mu_wire=0.2)
print(result.D_bulk_per_ring, result.critical_ring)
```

The damage is the bulk count: the elastic outer-fibre stress `E ε` against
the layer's `bulk_sn` (fixed azimuths, most damaged azimuth per ring). Every
cycle is mapped to the stress ratio of that curve by the
Smith–Watson–Topper correction (gap CF2; `mean_stress="swt"`, the
default; `mean_stress=None` restores the uncorrected count). With each
conductor's own wire curve (`nasution_2014_copper` for the 300 mm² wires,
`nasution_2013_copper_95` for the 95 mm² wires) this reproduces the four dry
tension–bending groups of Nasution et al. within ×1.45–2.39 and both radius
trends, and places the first failure in the observed layer
([benchmarks.md](benchmarks.md), "Conductor fatigue version 2").

A `fretting_sn` declared on the conductor (for example `copper_poon2022`) is not
evaluated by default and raises `UnevaluatedFatigueCurveWarning`. The trellis
term is added on request:

```python
result = cx.compute_conductor_fatigue(
    cable, kx, ky, tension_N=tension, trellis_fretting=True,
    critical_distance=cx.CriticalDistanceSpec(threshold_MPa_sqrt_m=3.8),  # default
    contact_yield_Pa=300e6,          # default; pass the conductor's own value
)
print(result.D_fretting_per_ring, result.critical_distance_m, result.trellis_face)
```

The default trellis rule is the point method of critical distances: at every
crossing of oppositely laid rings the contact after plastic indentation (mean
pressure 3σ_y) and the sliding traction are added to the bulk stress at depth
L/2, `L = (1/π)(ΔK_th/Δσ₀)²` with Δσ₀ the curve's range at 10⁷ cycles and
ΔK_th = 3.8 MPa√m (band 2.1–3.8, Tromans and Sun 1996), and every plane is
counted with the same SWT correction and curve. The ring damage is the larger
of the bulk and the trellis value. On the Nasution tests it changes the lives
by at most a factor of two, because the indented crossings are narrower than
the material length.

The legacy rule of gap 1.3c, `trellis_model="surface_swt"`, emits
`UnvalidatedModelWarning`. Where neighbouring rings are laid in
opposite directions:

- the crossing force is the interface line load divided by the crossings per
  unit length (Raoof and Hobbs 1988, Eqs. 23–25), without the in-layer hoop
  share of their Eq. 19;
- the tangential force at a crossing at azimuth θ follows the local stick
  shear flow `E A cos²α sin α κ_n`, `κ_n = −κ_x sin θ + κ_y cos θ` (zero at
  the extreme fibre, largest at the neutral axis, where slip starts; Papailiou
  1997), as a stop operator of `κ_n/κ_c` clamped to ±μP, with the
  neutral-axis slip curvature `κ_c = 2 ε_f /(π R cos²α)`;
- the peak pressure is limited to `3 σ_y` (fully plastic indentation, Johnson
  1985 §6.3) and the contact area grows once the Hertz mean pressure exceeds
  it;
- cycles of the tangential force are counted, cycles without slip are skipped
  (`partial_slip_cycles="cattaneo_mindlin"` gives them the partial-slip
  driver instead), and each slipping cycle receives its own SWT driver
  `σ = 2 μ p0 √(Q_a/(μ P))`.

The armour copper branch of `compute_fatigue` uses the per-cycle rule too;
the former estimate, which charged every bulk-stress cycle one gross-sliding
contact cycle, remains available as
`copper_contact_model="legacy_gross_sliding"` for comparison only.

Why the legacy surface rule is not used:

| Nasution test group | Test life | Bulk S–N (default) | Bulk + trellis SWT, corrected | Bulk + trellis SWT, gap 1.3 |
| --- | --- | --- | --- | --- |
| A1 300 mm², ρ 3 m, dry | 2.33 × 10⁵ | ×1.2 | ×0.026 | ×0.0024 |
| A2 300 mm², ρ 6.5 m, dry | 8.32 × 10⁵ | ×2.6 | ×0.0075 | ×0.0007 |
| A3 300 mm², ρ 3 m, lubricated | 3.97 × 10⁵ | ×20 | ×19 | ×5.8 |
| B1 95 mm², ρ 1.5 m, dry | 1.28 × 10⁵ | ×2.0 (×0.89 with the 95 mm² wire curve) | ×0.12 | ×0.013 |
| B2 95 mm², ρ 3 m, dry | 1.32 × 10⁶ | ×5.0 (×2.3 with the 95 mm² wire curve) | ×0.013 | ×0.0012 |

The corrected rule still puts every dry test in gross slip at the
neutral-axis crossings, with the pressure at its limit (900 and 675 MPa), so
its life does not depend on curvature: the radius ratios are 1.02 and 1.06
against the measured 3.6 and 10.3. Taking Nasution's FE crossing forces
(43.6 and 12 N) instead changes nothing once the contact saturates.

Checks against public data (`python validation/scripts/check_conductor_strand_slip.py`,
tested in `tests/test_strand_slip.py`); nothing below is fitted:

| Source | Quantity | Source value | CableXSec |
| --- | --- | --- | --- |
| Karlsen-Husøy et al. (2026) Table 3 | slope breakpoints after reversal | 0.0342, 0.0582 1/m (closed form) | 0.0342, 0.0582 1/m |
| Karlsen-Husøy et al. (2026) Tables 4–5 | ranking of the six methods | FS > SOE > FFS > MOE > AO > FSS (damage) | same order of strain range; MOE = AO in full slip |
| Hu et al. (2022) | full-slip bending stiffness | 14.4 N·m² (test) | 12.9 N·m² (−10%) |
| Hu / Nam et al. (2024) Table 3 | maximum moment at 0.37 1/m | 8.18 N·m (test) | 7.58 N·m (−7%) |
| Hu et al. (2022) | loop area | 3.185 N·m/m (test) | 4.08 N·m/m (+28%) |
| Utting and Jones (1987) Part II Table 2 | strand V wire strain per kN | 62.9 µε/kN (test) | 65.6 µε/kN (+4%) |
| Utting and Jones (1987) Part II Table 1 | fixed-end torque per kN, six strands | 0.47–0.86 N·m/kN | 6–12% higher |
| Nasution et al. (2014) Table 5 | 2nd-layer friction stress range | 117 MPa (analytical) | 107 MPa (−9%) |
| Nasution et al. (2014) Table 2 | life, r = 3 m, μ = 0.2 | 2.1–2.6 × 10⁵ (tests) | 3.1 × 10⁵ |
| Nasution et al. (2014) Table 2 | life, r = 6.5 m, μ = 0.2 | 6.2–9.8 × 10⁵ (tests) | 2.6 × 10⁶ (×3) |
| Nasution et al. (2014) Table 2 | life, r = 3 m, μ = 0.02 (lubricated) | 3.7–4.3 × 10⁵ (tests) | 9.3 × 10⁶ (×23) |

The Hu comparison carries the sheath pressure on the outer interface only
(the default `pressure_transmission="outer_interface"`); passing it to every
interface doubles the moment and multiplies the loop area by 3.5. The rows
above compare a uniform-curvature section with the test, which is a
three-point bend (span 440 mm; the reported curvature is 12 w/L², Hu Eq. 2).
Gap G8 (`python validation/scripts/check_g8_gradual_reversal.py`) separates the two
effects; gap T1 makes the test reduction a package function
(`simulate_bending_test`, Section 6.2, scorecard row HU-3PB). Reduced as the test reduces it, the same section law gives the
measured share of the stroke in the slip transition (35.6% against 33.8%):
the gradual transition of the test comes from the curvature varying along
the span, not from the section. The loop area is then +65% and the maximum
moment +8%. The option `cap_shape="papailiou"` of `strand_slip_strain` and
`conductor_bending_moment` replaces the amplitude `ε_f |cos φ|` of
Karlsen-Husøy Eq. 4, with which every azimuth of a ring slips at one
curvature, by Papailiou's neutral-axis cap `ε_f (2/π) asin|cos φ|` of the
same quarter-pitch capacity. With it the three-point loop has area +37%,
maximum moment 8.11 N·m (−0.9%) and transition share 32.1%; at uniform
curvature the area is +4.9%, but that is not the test's reduction. The
default stays Eq. 4. The plane-section stick
stiffness is 793 N·m², below the bending stiffness of a solid copper rod of the
conductor's outer diameter (1159 N·m² at 118 GPa), as a stick state must be;
Papailiou's (1997) stick term gives 790 N·m², as here. The only
measured tangent is 196.5 N·m² at reversal (Hu et al. 2022, Fig. 9), which
Hu et al. place in the slip-transition zone, not in stick. The Mindlin
compliance of the trellis crossings (gap G8) lowers the stick
coefficient of the outer rings to about 0.55 and the three-point area by
about two points only. The Nasution lives use the
`nasution_2014_copper` curve without the compaction stress concentration; the
source itself finds its lubricated tests and long lives at 6.5 m shorter than
its models predict. The bulk count's lubricated life (×20) is not resolvable
from public data: the lubricant and its friction coefficient are not
reported, and for lubricated friction between 0.02 and 0.1 every
friction-driven model here, bulk or contact, predicts a lubrication effect of
at least 5 against the measured 1.7.

Limits: the power core is taken in full slip inside the cable (no core-level
stick order); in-layer contact, copper plasticity, compaction and wear are not
modelled; the crossing force is held at the reference tension; the SWT driver
is a closed-form line-contact surface stress, not a critical-plane search.

## 11. Fretting regime, wear and fatigue knock-down

`compute_fretting` is an opt-in companion of `compute_fatigue`; it does not
change any fatigue result. For every structural armour layer it assesses both
radial faces of every wire:

- a polymer neighbour (bedding, sheath, tape) is a Hertz line contact of the
  wire on a flat, `b = (4 q R/(π E*))^½`, `p0 = 2q/(π b)`, with the line load
  `q` from the friction-setting pressure `P1 + P2` of the section law, split
  between the faces in the ratio of the bounded radial chain;
- an opposite-laid armour layer in direct contact is the Mindlin crossing of
  the existing contact model (Hertz `K_t`, crossing force from the shared
  interface line load).

Each wire's kinematic slip history (`slip_increment`; loxodromic by default,
geodesic or legacy transverse component on request) is rainflow counted, and
every counted cycle of amplitude `δ_a` is classified against the transition
amplitude `δ_t` with the contact's own μ:

| Contact law | `δ_t` | Partial-slip energy per cycle | Gross-slip energy per cycle |
| --- | --- | --- | --- |
| Point, Mindlin (crossing, trellis) | `3μP/(2K_t)` (Fouvry et al. 1995, Eq. 19) | `(24/5) μP δ_t f(x)`, `x = 1 − (1 − δ_a/δ_t)^{3/2}` (Johnson 1985, Eq. 7.60) | `4μP δ_a − 3.2 μP δ_t` |
| Line, linear support (wire on polymer) | `Σ μq / k_G`, `k_G = EA (sin α/r)² ξ/(1 − ξ)` | Cattaneo strip micro-slip `(32/π²)(μq)²/E* I(√(1 − x))`, `x = δ_a/δ_t` (derived here) | strip energy at `x = 1` plus `4μq(δ_a − δ_t)` |

`δ_a ≤ δ_t` is partial slip and `δ_a > δ_t` gross slip; for the Mindlin law
this is the Fouvry energy-ratio criterion `A > 0.2` (`fouvry_ratios` also
returns the sliding ratio `D` and `B = A/D`, transitions 0.26 and 0.77).
Stick means that no slip is possible: infinite friction, a zero-amplitude
cycle or an unloaded face. The two-dimensional tangential displacement of a
line contact depends on a reference point (Johnson 1985, p. 214), so the line
regime boundary is set by the support compliance of the section law and the
Hertz width enters only the energy and the wear area. Planar slip paths
(legacy transverse map) are counted on their principal projection with the
amplitude of each cycle taken as half its largest chord, `δ*`; the energy at
`δ*` is the uniaxial value, not the path-dependent dissipation of rotating
loads (Ciavarella 2014; memory rules of Aleshin and Bou Matar 2016).

Wear is `V = α Σ W` over gross-slip cycles (`partial_slip_wear=True` adds
partial-slip energy), with the material-pair coefficient of
`cablexsec.materials.wear`. The reported depth is the mean over the swept area
of each cycle on the unworn contact, which overstates the depth rate once the
scar has grown (crossed steel wires, Section "WEAR-CW" of [benchmarks.md](benchmarks.md): about
six times the measured maximum depth, while the volume is reproduced).

| Pair | α [m³/J] | Source |
| --- | --- | --- |
| `steel_steel` | 2.9 × 10⁻¹⁴ (6.3 × 10⁻¹⁵ – 5.5 × 10⁻¹⁴) | Cruzado et al. (2013) Table 3, `k/μ`; band O'Halloran (2017) |
| `copper_copper` | 9.4 × 10⁻¹³ (2.3 × 10⁻¹³ – 1.4 × 10⁻¹²) | Poon et al. (2023) Table 1; Poon et al. (2024) Table 3 |
| `steel_polyethylene`, `steel_polypropylene`, `steel_polyamide`, `steel_fluoroelastomer` | 1 × 10⁻¹³ (10⁻¹⁴ – 10⁻¹²) | **assumption**: no source in the library |

The steel-on-polymer depths are therefore order-of-magnitude indications that
scale linearly with α; replace the value with `wear={"steel_polyethylene":
get_wear_coefficient("steel_polyethylene").scaled(alpha)}` once a coefficient
for the actual bedding is measured.

The fatigue knock-down counts the contact-fibre stress (inner face: the
inward fibre, outer face: the outward fibre) and gives each stress cycle the
regime of the largest slip amplitude within its window. Partial-slip cycles
are multiplied by the Nishioka–Hirakawa notch factor
`K_f = 1 + 2μ p_m (1 − e^{−δ/k}) / σ_a,ref` (Nakazawa et al. 1994, Eq. 1),
with the mean contact pressure `p_m` clamped to the 40–160 MPa of those tests,
`σ_a,ref` half the S–N range at 10⁷ cycles and an assumed `k = 2.7 µm`; gross
slip has no crack knock-down because wear removes initiated cracks (Vingsbo
and Söderberg 1988, Fig. 10; Poon et al. 2023, 2024). `section_loss=True`
raises the stress by `1/(1 − f)`, `f` the wire section worn over the design
life on faces where the wire itself wears. `knockdown=None` skips the S–N
part.

```python
import numpy as np
import cablexsec as cx

cable = cx.load_cable(cx.iea_task49_66kv_yaml_path())
t = np.arange(0.0, 600.0, 0.1)
kx = 2e-3 * np.sin(2 * np.pi * t / 8.0)
ky = 1e-3 * np.sin(2 * np.pi * t / 11.0)
result = cx.compute_fretting(
    cable, kx, ky, duration_s=600.0, tension_N=np.full_like(t, 9.0e4)
)
for face, row in result.summary().items():
    print(face, row["share_gross_slip"], row["max_wear_per_year_m"],
          row.get("max_damage_offset_deg"))
```

Each `FrettingFace` holds per-wire cycles and energy per regime (per year),
the wear depth per year, `δ*`, `δ*/b`, the mixed-regime flag and the plain
and knocked-down damage at the contact fibre; `extreme_fibre_offset_deg` is
the angular distance of each wire from the extreme fibre of the principal
bending direction (0° extreme fibre, 90° neutral axis), and `location()`
returns the offsets of the largest wear and damage. Results are per year of
the supplied loading times `time_fraction`.

`cablexsec.analysis.fretting_assessment.conductor_trellis_fretting` gives the
same regime map for the trellis crossings of a stranded conductor: the
tangential-load ratio of each crossing is the friction mobilisation of the
Karlsen-Husøy operator, cycles without slip are partial slip with the Mindlin
energy, and slipping cycles are gross slip with the kinematic relative slip of
the two rings as an upper estimate of the sliding amplitude.
`helical_layer_fretting` with `FaceSpec` entries assesses any other helical
wire layer, for example screen wires on a tape.

`python validation/scripts/run_fretting_regime.py --output <file.json>` runs
the 66 kV template and the Ménard cable under a public synthetic history (or,
with `--history-dir`, under saved section histories). With that synthetic history the 66 kV
armour-on-polymer faces are 94–98% gross-slip cycles (`δ*/b` 2–8, `δ_t`
1.3–4.9 µm) with wear largest 71–72° from the extreme fibre and the
largest contact-fibre damage within 3° of it; the more strongly clamped Ménard
faces are in partial slip (`δ*/b` 0.07–0.6).

Limits: contact loads are held at one reference tension; the line contact
neglects the normal–tangential coupling of dissimilar materials; circular
Mindlin forms are applied to the area-equivalent elliptical crossing; the
steel-on-polymer wear coefficients and the knock-down slip constant are
assumptions; the regimes are not compared with fretting tests on cable
contacts.

## 12. Torque balance and twist under tension

A tensioned helical layer carries a torque about the cable axis. With the lay
sign `s` (+1 right-hand, −1 left-hand), `n` wires of area `A` at helix radius
`r` and lay angle `α` from the cable axis, a layer whose wires carry the
stress `σ` contributes

    M = s n A σ r sin α

(Knapp 1981, Eq. 10; Feyrer 2015, Eq. 2.11 with the axial wire force
`S = A σ cos α`). With both ends restrained against twist and fixed layer
radii, `σ = E ε cos²α`, so that the torque per unit tension is `K_et / K_ee`
of the Witz–Tan matrix that `compute_stiffness` and `xsec_stiffness` report;
this is Feyrer's torque constant `c_1S d_S` (Eq. 2.74). Two contra-helical
layers are torque balanced when their torques cancel (Knapp 1981,
Eqs. 22–24).

`torque_balance` takes the layer stresses from the solved radial section
(`solve_cable_radial_states`), the same state as the solved contact,
fatigue and thermal routes: radial contraction of the core, unilateral
contacts between layers (a layer that lifts off stops sharing load), laid
power cores with their own lay torque, polymer torsion and the declared
manufacturing clamping (Section 4.4), which is carried as residual wire
stress. The section is solved at every tension twice: with the twist
restrained (end torque) and with zero end torque (twist of a free end).

```python
import cablexsec as cx

cable = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
r = cx.torque_balance(cable, (0.0, 200e3), n_tensions=5)
print(r.summary())
r.torque_coefficient_m           # (M(T) - M(0)) / T at the largest tension [m]
r.twist_per_100kN_deg_per_m      # free end, secant [deg/m per 100 kN]
r.layer_torque_fixed_Nm          # torque increment of each helical layer
r.core_torque_fixed_Nm           # laid cores and other layers
r.balance_ratio                  # -dM_outer / sum(dM_inner) of the armour
r.stress_redistribution_Pa       # sigma_free - sigma_fixed per layer
r.fixed_radius                   # K_ee, K_et, K_tt and coefficients of xsec_stiffness
```

Signs: a positive torque or twist turns in the sense that tightens a
right-hand lay. A free end turns opposite to the fixed-end torque, which
unloads the layers that carry that torque and loads the others. Tangent
coefficients of each solved state are reported next to the secants; they are
equal while the contact set does not change. Loose core screens
(non-structural `cylindrical_offset` layers) are meshed with the material of
their softer structural neighbour (`loose_layers="neighbour"`, the default),
because the fixed-radius stiffness gives them no axial load; with
`loose_layers="declared"` loose screens carry axial load in the solved
section (on the 66 kV template, whose loose layers are a semiconducting screen
and a 0.10 mm copper tape, +4 % on the torque coefficient; full copper screen
annuli doubled it).

`design_torque_balance` varies one armour layer (the outermost structural
layer by default) to set the torque/tension coefficient at fixed twist to a
target (zero by default) at a tension, or to minimise its largest deviation
over a tension range:

```python
d = cx.design_torque_balance(cable, "outer_lay_angle", tension_N=100e3)
d.value, d.fill_ratio, d.result.twist_per_100kN_deg_per_m
cx.design_torque_balance(cable, "outer_wire_count", tension_N=100e3)
cx.design_torque_balance(cable, "outer_wire_diameter", tension_N=100e3,
                         stress_limit_Pa=0.6 * 375e6)
cx.design_torque_balance(cable, tension_range=(50e3, 300e3))   # minimax
```

The search interval is the intersection of `bounds` (defaults: 1° to
35.26°, the maximum of `sin α cos²α` below which the layer torque rises with
the lay angle (Knapp 1981, Fig. 3); a quarter to four times the wire count;
half to twice the wire diameter) and the lateral fill limits
`min_fill <= n d / (2 π r cos α) <= max_fill`; adjacent wires touch at a fill
of 1. A changed wire diameter keeps the layer's inner face and moves the
layers outside it. A whole wire count is returned with the continuous root.
`stress_limit_Pa` is checked at the largest tension with the twist fixed and
free, as the stress check of Kim and Lee (2017, Section 5); it is not
optimised, nor is the bending stiffness. The declared manufacturing clamping
of the varied layer is kept unchanged. When no admissible value meets the
target, `converged` is False and the closest bound is returned.

IEA Wind Task 49 66 kV template at 100 kN (core lay and armour lay
directions are assumptions of the template; `examples/torque_balance_templates.py`):

| Quantity | Solved section | Fixed-radius matrix (`xsec_stiffness`) |
| --- | --- | --- |
| Torque/tension, twist restrained | 1.503 mm (150 N·m at 100 kN) | 1.510 mm |
| Twist of a free end per 100 kN | −0.0533 °/m | −0.0666 °/m (no polymer or core torsion) |
| Layer torques at 100 kN | inner −604, outer +525, laid cores +229 N·m | |
| Armour balance ratio | 0.869 | |
| Free minus fixed wire stress at 100 kN | inner +3.6 MPa, outer −2.4 MPa (of 30.2 and 31.4 MPa) | |
| Outer armour that zeroes the torque | lay angle 6.93° (template 10°, fill 0.859); or 103 wires (145, fill 0.615); or 2.66 mm wires (3.15 mm, fill 0.734) | |
| Same, core lay left-hand | torque/tension −3.08 mm; lay angle 18.1° (fill 0.897) | |
| Same, aligned cores (armour alone) | torque/tension −0.77 mm; lay angle 11.76° (fill 0.871) | |

The laid cores add +229 N·m to a net armour torque of −79 N·m at 100 kN, so the
armour alone is not balanced at the zero-torque design (ratio 0.62). The
sign and size of the core contribution follow the core lay direction, which
the source does not state.

Checks (`python validation/scripts/check_torque_balance.py`,
`tests/test_torque_balance.py`, scorecard rows TQ, TQ-KL and TQ-UJ;
[benchmarks.md](benchmarks.md)):

| Source | Quantity | Source value | CableXSec |
| --- | --- | --- | --- |
| Feyrer (2015) Example 2.8, 1×37 spiral rope | torque constant `c_1S` | 0.0345 | 0.03446 (fixed radius); 0.03470 (solved, with Poisson contraction of the centre wire) |
| Knapp (1981) Eqs. 18, 20, core factor with θ = 2ν | fixed-twist torque per unit strain, radially stiff core | closed form | −1.3e-6 relative |
| Kim and Lee (2017) Section 5.1, 1+6+12 strand | balancing outer helix angle, seven pairs from the beam FE model | 86.3–88.8° | −0.06 to −0.26° |
| Utting and Jones (1987) Part II Table 1, six strands | fixed-end torque per kN | theory 0.50–0.92, test 0.47–0.86 N·m/kN | ×1.051–1.059 theory; ×1.067–1.136 test |

The model omits the bending and twisting moments of the individual wires
(Costello 1997, Eqs. 3.39–3.40), which lower the torque of a seven-wire
strand by about 5% (Costello 1997, Example 3.1): this is the systematic
excess over the Machida–Durelli theory. They scale with `(d / 2r)²`, about
0.23 for these strands and 3.4e-4 for the 66 kV armour wires. Friction, slip and wire
flattening are not modelled.

## 13. Power-core screens, core slip state and metallic sheaths

Metallic screens and sheaths of a
three-core cable sit inside a laid power core; their strain depends on how the
core moves in the cable section.

### 13.1 Core slip state (P8)

The strain of any layer inside a core, at radius `ρ` and angle `θ` in the core
section (`θ = 0` away from the cable axis), is the core-axis strain plus the
core's own bending: `ε = ε_c(φ, t) + ρ (g_r cos θ + g_t sin θ)`, with `φ` the
core station on the pitch circle and `g_r`, `g_t` the two Lutchansky
components of the helical core axis (Ménard and Cartraud 2023, Eq. 14). The
core slip state sets `ε_c` and is an explicit, required input:

| `core_state` | Core-axis strain `ε_c` | Meaning |
| --- | --- | --- |
| `"stuck"` | `r_c cos²β (κ_x cos φ + κ_y sin φ)` | plane-section upper bound (core bonded to the cable section) |
| `"free"` | `0` | full-slip lower bound (core slides along its helix) |
| `"frictional"` | station stress of the laid-core law / `E_core` | the core friction law of Section 6.1 (bounded or solved trefoil contact, shear-lag `ξ`, Papailiou cap) |

```python
import math
import numpy as np
import cablexsec as cx

cable = cx.load_cable(cx.thies2016_yaml_path())
kx = 0.05 * np.sin(np.linspace(0.0, 4.0 * np.pi, 161))
ky = np.zeros_like(kx)
field = cx.compute_core_layer_strain(cable, "al_laminate", kx, ky,
                                     core_state="frictional")
field.strain          # (station, theta, time)
field.maximum()       # largest amplitude, its station and angle
kin = cx.core_kinematics(cable, kx, ky, state="stuck")   # axis strain, gradients
```

A cable axial strain adds `ε cos²β` to the core axis in every state. The
layers of a core are taken as bonded (they share `ε_c`). The normal gradient
`g_r` follows the core slip state like the normal bending of an armour wire
(Section 5, Step 2): `g_r = cos²β cos2β κ_r + (sin²β / r_c) ε_b`, with `ε_b` the
bending part of `ε_c`, which is `cos⁴β κ_r` stuck and `cos²β cos2β κ_r` free.

### 13.2 Wire screens wound about each core (P1)

A copper wire screen of a power core is declared on the core
(`cylindrical_offset`, `construction: resolved_wire_screen`, one resolved ring
in `strand_layers`) with the pressure of the tape, foil or sheath over it:

```yaml
outer_face_pressure:
  pressure_Pa: 2.0e5
  identification: assumed_bound      # or measured, manufacturing_specification, ...
  bounds_Pa: [0.5e5, 5.0e5]          # required for assumed_bound
  source: "tape winding stress x thickness / radius, plus sheath shrink fit"
```

`compute_core_screen_fatigue(cable, kx, ky, core_state=...)` drives the
unchanged helical-layer kernel of the armour with the core curvature
(wire bending, Papailiou friction stress on `P1 + P2`, Yun `ξ` of the core
layers inside the ring) and adds the core-axis strain as `E cos²α ε_c`. Per
core station and wire it returns stress ranges at four fibres, bulk Miner
damage against the layer's `bulk_sn`, and on both faces (inner: the layer under
the ring; outer: the tape or foil over it) the fretting regime, slip, wear and
knock-down of Section 11. The outer face carries the declared pressure, the
inner face the same pressure; `core_contact=True` adds the core contact line
load spread over the ring. Ranges above yield are flagged (`exceeds_yield`):
the elastic S–N basis does not hold there; `strain_life=` adds the
strain-life route of Section 13.7 (gap P7). With
one core at the cable axis the result is the helical-layer result (gate P1-A).

The cable-level `initial_state` accepts the same `"assumed_bound"` class with
`bounds_Pa`, so a tape or sheath pressure over the outermost helical layer can
be declared for `compute_fretting` (gap P3, cable level).

### 13.3 Strain-life fatigue of sheaths, foils and tapes (P6)

`compute_shell_fatigue(..., method="strain_life")` treats a sheath, foil or
tape as strain controlled: fibre strain (`SCF f_corr κ r` for a concentric
sheath, the core-layer strain of Section 13.1 for a layer of a core), rainflow
strain cycles, Masing stresses on the cyclic curve, and the
Coffin–Manson–Basquin life with the SWT (default) or Morrow correction.
Strains above yield are accepted. A `bonded_laminate` foil follows the polymer
it is bonded to (Worzyk 2009, Sec. 2.3.2), including the polymer's hoop
strain (von Mises equivalent strain, a factor of 1.0003–1.002). The default
`method="elastic_sn"` (EN 1999-1-3 route) is unchanged.

Registered constants (`cablexsec.materials.strain_life`):

| Key | Basis | Source |
| --- | --- | --- |
| `copper_poon2022` | published | Poon et al. (2022) Table 4 (Karlsen 2010 conductor wire); applied to a copper tape or foil it is a transfer assumption |
| `aluminium_foil_mus_assumed` | assumed | modified universal slopes (Muralidharan and Manson 1988) with assumed foil inputs (`S_u` 70–110 MPa, ductility 0.3–1.0) |

No strain-life constants for lead alloys E or ½C, laminate foil alloys or
corrugated aluminium are in the reference library (the lead S–N points of
Kim and Choi 2026, Table 3, are quoted from a thesis not held; Fugløy 2017
tests stainless steel; EN 1999-1-3 excludes the foil alloys for lack of
fatigue data). Lead-sheath fatigue therefore stays blocked; user constants
are registered with `cx.register_strain_life(cx.StrainLifeConstants(...))`
and must carry their reference. `cx.register_material` adds a user material
for the process (a citation per property is required).

### 13.4 Applications

*Thies et al. (2016) restrictor test* (`cx.thies2016_yaml_path()`, public
reconstruction with stated bounds; `examples/thies2016_screens_sheaths.py`):
at `R = 2.54 m` the laminate strain amplitude is 0.80% (free), 0.82%
(frictional) and 1.81% (stuck); the summed laminate damage within 0.6 m of the
headstock is 49 (locking at the MBR) and 19 (video radii), cracking under
every bound. The screen wires carry a 247 MPa friction stress range at the
extreme fibre, 0.82 mm slip at the neutral axis and a bulk damage of 0.47;
the fretting knock-down raises the partial-slip wires next to the extreme
fibres to 19 (library Cu–Cu μ = 0.20) or 37 (μ = 0.25). The core lay (4.7°)
lowers the radial core curvature by 2%, which moves two of the four
partial-slip wires into gross slip (4 with aligned cores, 2 with the lay).

*IEA 66 kV template:* the metallic screen is a copper tape (its copper area
is not published; [template_66kv.md](template_66kv.md) Section 2.1), so P1
does not apply, and the wet design has no metallic sheath, so the sheath
route does not apply either. The core-layer strain of the tape at the MBR
(2.76 m) is 1.17% (free and frictional: the cores slip, no clamping is
declared) and 2.49% (stuck).

### 13.5 Bend restrictor and bend stiffener fields (P5)

Near a hang-off, a J-tube exit or a test-rig headstock the curvature is not
proportional, `κ(x, t) ≠ κ_ref(t) k(x)`: an articulated restrictor locks
element by element, so the bent length grows with the angle.
`cablexsec.mechanics.bend_restrictor` returns the field for a history of the
root angle:

```python
time = np.linspace(0.0, 10.0, 101)
rotation_rad = math.radians(10.0) * np.sin(2.0 * np.pi * time / 10.0)
ei, t = 5.0e3, 2.0e4                     # cable bending stiffness [N m^2], tension [N]
r = cx.ArticulatedRestrictor(pitch_m=0.36, lock_angle_rad=math.radians(4.75),
                             n_elements=15, bore_diameter_m=0.1375)
# or cx.ArticulatedRestrictor.from_mbr(2.54, pitch_m=0.36, ...)
field = cx.restrictor_curvature_field(r, rotation_rad, cable_od_m=0.123,
                                      bending_stiffness_Nm2=ei, tension_N=t,
                                      span_m=5.4, time_s=time)
field.kappa_1_m, field.n_locked, field.locked_length_m, field.notes
```

*Kinematics.* Rigid-link chain: the locked chain is a polygon of
circumradius `R_L = p / (2 sin(θ_L/2))` (arc value `p/θ_L` within
`θ_L²/24`). The cable lies on a circle concentric with the polygon inside the
bore clearance `c = (D_b − D_c)/2`, the one of least `EI/(2R) + T R`
(`sqrt(EI/2T)` clipped to `[R_L − c, R_L cos(θ_L/2) + c]`). Outside the
locked zone the cable is a small-deflection tensioned beam (`EI`, `T`, decay
length `λ = sqrt(EI/T)`) between the end of the locked zone and the far end
of the span (`w = w' = 0`, as the rig's `end_rotation` profile). The default
rule `"joints"` locks whole elements: element `k` turns by the cable rotation
over its pitch and locks at `κ_L p`; `"smeared"` uses a continuous front with
continuous cable moment. The lock limits element turning, not the local
curvature: the cable may curve more tightly inside the first free element, up
to the straight-bore limit `16 c/p²` (a larger peak is flagged). The notes
flag a stated MBR that differs from the rigid-link radius, reverse curvature
near the far end above the lock (not modelled: locking proceeds from the
root) and over-bending beyond a fully locked restrictor (`tip_overbend`).

*Bend stiffener.* `cx.BendStiffener(length_m, root_od_m, tip_od_m, E_Pa)`
with `cx.stiffener_curvature_field(...)`: a large-deflection Euler beam with
`EI(s) = EI_c + E_PU π (D(s)⁴ − D_c⁴)/64`, clamped root, tension at the far
end (Caire and Vaz 2007; linear polyurethane, solved by conservative finite
differences and Newton). Without stiffener and at small angles it is
`κ = θ/λ exp(−s/λ)`.

*Where the field is used.*

| Path | How |
| --- | --- |
| `simulate_rig_test` | `RigSetup(profile="field", curvature_field=field, reference_x_m=...)` with `RigProtocol.from_curvature_field(...)`; the clamp solver follows `κ(x, t)` segment by segment (`anchored_wire_field_response`) |
| `simulate_bending_test` | `restrictor=r, restrictor_root_x_m=..., cable_od_m=...` (load or moment control): the cable curvature is capped at the lock curvature and the restrictor carries the rest (`restrictor_moment_field_Nm`) |
| `compute_fatigue(termination=...)` | `TerminationSpec(curvature_field=field, distance_m=x)`, clamp or along-wire solver; the section history must equal the field at `x` |

With a proportional field each path reproduces the proportional result
(tests `tests/test_restrictor_tapes_copper.py`).

### 13.6 Helical tapes (P2)

A tape is a `cylindrical` or `cylindrical_offset` layer with a `tape` winding
(width, angle to the layer axis, number of tapes, source):

```yaml
tape:
  width_m: 0.020
  lay_angle_deg: 50.0
  source: "angle read from a figure (bound 40-60 deg); width assumed, bound 10-30 mm"
```

`cx.tape_geometry` gives the pitch `2πρ/tan β`, the axial gap
`P/n − w/sin β` (negative: overlap), the coverage and the edge angle
`w cos β/(2ρ)`. `cx.compute_tape_fatigue(cable, layer, kx, ky, state=...,
core_state=...)` takes the substrate strain field (the core slip state of
Section 13.1 for a tape of a power core) and returns the tape strain at four
fibres (both faces on the centre line, the two edges), the slip along the
tape, the slip between overlapped turns and the strain-life damage:

| `state` | Tape strain |
| --- | --- |
| `"stuck"` | surface strain along the tape, `ε_z (cos²β + h sin²β)`, at each fibre's azimuth (`h` the substrate hoop ratio, default `−ν` of the polymer below) |
| `"free"` | uniform part only plus the tape's own flat-wise bending; edge-wise bending only with `path="loxodromic"` |
| `"frictional"` | elastic–Coulomb along the tape with the Papailiou cap on the declared `contact_pressure_Pa` (`P1 + P2`) and `mu` (both required) |

Copper tape uses `copper_poon2022` (a transfer from conductor wire); other
tapes need their own constants. The contact stress of wires bearing on the
tape edges is not modelled.

### 13.7 Copper plasticity, curves and friction labels (P7)

`compute_core_screen_fatigue(..., strain_life="copper_poon2022")` and
`compute_conductor_fatigue(..., strain_life=...)` add a strain-life damage
(`D_strain_life`, `D_strain_life_per_ring`) of the wire strain on the Masing
cyclic curve, so stress ranges above yield are assessed rather than only
flagged; the elastic S–N results are unchanged. The wire strain is the
elastic-analysis strain `σ_el/E`, taken as imposed (strain controlled).

| Item | Outcome of the library search |
| --- | --- |
| Copper `K'`, `n'` | not published in the library; the compatibility estimate `n' = b/c` is used |
| Copper strain-life sets | `copper_poon2022` (published, Karlsen 2010 data) and `copper_annealed_mus_derived` (modified universal slopes on E 115 GPa, Karlsen 2010; `S_u` 235 MPa, Wan et al. 2022; reduction of area 0.725, Jiang et al. 2023) |
| Copper fretting knock-down | no plain-versus-fretting copper data; `copper_nishioka_hirakawa` stays the steel relation, labelled assumed |
| Mean in-air S–N of round galvanised armour wire | not published in the library (OC2017 gives a design curve for flat wire, m = 6, log a 20.5; DNV-RP-C203 HS mean curve is for forgings); none registered |
| `copper_stranded` μ 0.25 | Poon et al. (2022) contains no 0.25; relabelled as an assumption with bound 0.1–0.5 (Nasution et al. 2014b measured 0.2); value and results unchanged |

### 13.8 Thies (2016) with the restrictor kinematics

`examples/thies2016_screens_sheaths.py`, Part B: the headstock rotation and
the Z-ram tension of each block drive the restrictor field (pitch 0.36 m,
bore 137.5 mm, secant `EI` 3.7–4.9 kN·m² from the cable model). The two
readings of the published restrictor data are run: lock from the stated MBR
(8.13° per joint, `R_L` 2.54 m) and the stated 4.75° per joint (`R_L`
4.34 m). Results are in [benchmarks.md](benchmarks.md) (P5-T); damage values depend on the
assumed inputs and bounds of the reconstruction.

## 14. Limits of validity

- **User fatigue data.** Registered curves, mean-stress corrections and
  component SCFs are the user's data, recorded in the provenance; their
  sources are not checked. The component SCF mean is the record's time
  average. The capacity check (Section 8.1) uses the curvature magnitude and
  assesses each sample quasi-statically; between levels it interpolates, and
  beyond first yield it extrapolates (counted). Its Gumbel fits assume
  independent maxima of one condition.
- **Constant curvature.** The section law assumes curvature and tension are
  uniform over at least one lay length. Strong curvature gradients near bend
  stiffeners and end fittings are not resolved by the section law. A
  termination enters through the declared `termination_scf` and, for
  anchored armour, through the optional clamp correction
  (`termination=TerminationSpec(...)`, Section 5, Step 6): the clamp solver
  is planar and axial-slip-only, the along-wire solver biaxial with an
  optional lateral relief; neither is verified against helix FE, and against
  the Sævik (2011) FBG measurements both over-predict (SV11, T2). For a test
  specimen `simulate_rig_test` uses the same solver
  ([rig_test_guide.md](rig_test_guide.md)). A bend restrictor or stiffener
  field `κ(x, t)` (Section 13.5) is kinematic: small rotations, linear
  secant `EI`, no restrictor hysteresis, locking from the root only.
- **Bending loop calibration fixes the product, not the split.** Matching a
  measured moment–curvature loop identifies the friction cap, the product of
  friction coefficient and clamping pressure up to second order in
  `μ |sin α|`. Different (μ, pressure) pairs give the same loop. Outputs that depend on each factor
  separately, such as crossing pressure, slip amplitude and fretting indices,
  are not fixed by the loop. State which split you use.
- **Transverse slip.** The default loxodromic kinematics (`transverse_ratio=0`)
  contain no transverse slip; the longitudinal slip equals the wire-direction
  component of the frictionless geodesic slip of Out and von Morgen (1997)
  (gate OvM). For a transverse sensitivity use
  `FatigueLayer(transverse_ratio=eta, transverse_map="geodesic")` (or
  `slip_increment(..., transverse_map="geodesic", lay_angle_rad=alpha)`).
  This map is in phase with the frictionless slip: the transverse increment
  is `-eta (s_g/s_δ)` times the longitudinal one, with
  `s_g = r²cos α(1 + sin²α)/sin²α` (`transverse_geodesic_factor`), so both
  vanish at the extreme fibres and peak at the neutral axis, and every
  increment keeps one direction in the contact plane. `eta = 1` is the
  frictionless geodesic slip (gate OvM-g); `0 <= eta <= 1` scales the
  transverse part between the loxodromic and geodesic limits, so results are
  bounded by the frictionless slip, not solved frictional trajectories.
  At η = 1 the slip at the neutral axis is (1 + (s_g/s_δ)²)^½ times the
  loxodromic slip (3.0 at 35°, 4.9 at 13°, 6.1 at 10°). The equilibrium
  transverse slip with friction computed for the published cable is 1.1–6.7%
  of the longitudinal slip, which corresponds to η of about 0.003–0.02
  (geodesic ratio 3.5 at its 20° lay angles).
  The former map remains as `transverse_map="legacy"`. It is in quadrature
  with the geodesic slip: it puts transverse slip at the extreme fibres,
  where the geodesic slip has none, and none at the neutral axis, where the
  geodesic transverse slip is (1 + sin²α)/(sin α cos α) times the
  longitudinal slip (2.8 at 35°, 4.3 at 15°). Its results are not upper
  estimates of transverse motion. A non-zero `transverse_ratio` without
  `transverse_map` selects the legacy map and issues `TransverseMapWarning`.
- **Conductor fatigue.** `compute_fatigue` analyses only structural armour;
  the conductor is assessed by `compute_conductor_fatigue` (Section 10; bulk
  S–N by default, trellis fretting only on request and not validated), whose
  agreement with the public tests is stated there. Component stress of the
  conductor gives local and constrained-axis strain fields, not a fatigue
  life.
- **Layers inside a power core.** Their strain depends on the declared core
  slip state (Section 13); the stuck and free states are bounds, and the
  frictional state holds the axisymmetric contact loads fixed. The elastic
  per-core wire-screen damage flags ranges above yield; the optional
  strain-life route (`strain_life=`, Section 13.7) assesses them.
  Strain-life constants are published only for copper conductor wire; the
  aluminium foil constants are an assumed bound and lead is blocked.
- **Core friction.** `core_friction` adds the laid-core stick–slip law
  (Section 6.1). Its contact loads are the axisymmetric state held fixed in
  bending; the layers of one core move together. Its friction level follows
  the declared clamping state (Section 4.4): with tension-induced contact
  alone it is small at low tension. On the measured Ménard ranges, with the
  initial pressure identified on one range, it predicts the work of the two
  smallest ranges 24–26 % high (the registered ±20 % fails) and of the other
  three within 13 % ([benchmarks.md](benchmarks.md), gap R1b). It stays off
  by default.
- **Global full-slip bending.** If the global model uses the full-slip
  (minimum) bending stiffness, local damage computed from its curvature
  histories is a conservative upper estimate compared with a global model that
  carries a consistent hysteretic moment–curvature law. The difference can be
  several times in damage.
- **Contact.** Mean pressure is uniform over each wire population. Contact-life
  laws require constant normal contact; variable section states are accepted
  for bulk fatigue only.
- **Elastic range.** Check absolute mean-plus-cyclic curvature against the
  capacity envelope before interpreting S–N damage.

Numerical consistency is not experimental accuracy. See
[validation_policy.md](validation_policy.md) for the tiers and rules every
comparison follows, and [benchmarks.md](benchmarks.md) for the register of
comparisons with measurements, published results and other models.

To save analyses with their inputs and provenance, see
[project_guide.md](project_guide.md). Further runnable scripts are in
`examples/`; the docstrings in `cablexsec/api.py` give every parameter, unit and
reference.
