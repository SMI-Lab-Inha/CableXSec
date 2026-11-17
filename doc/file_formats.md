<!-- SPDX-License-Identifier: Apache-2.0 -->

# File formats

CableXSec reads a {term}`cable definition`, the load histories of a
{term}`global analysis` and its own project files. It writes projects and bending
laws for global analysis. Each file below is given with its layout, its units
and the rules it is checked against.

| File | Read by | Written by | Format |
| --- | --- | --- | --- |
| Cable definition | `load_cable` | (by hand) | YAML |
| Geometry override | `load_cable(..., geometry_override=)` | (by hand) | YAML |
| Time-history CSV export of a global analysis | `read_time_history_csv` | global analysis program | CSV, named columns |
| CableXSec exchange CSV | `read_series_csv` | `write_series_csv` | CSV, fixed columns |
| RIFLEX ASCII time history | `read_riflex_ascii` | RIFLEX | whitespace-separated columns |
| CableDyn run (driver or OpenFAST module) | `read_cabledyn_sections` | CableDyn | CableDyn output files and deck |
| CableDyn 0.4.0 station table | `cablexsec.io.read_cabledyn_station_table` | CableDyn | CSV, tab or whitespace |
| Sea-state batch specification | `cablexsec-batch`, `load_batch_spec` | (by hand) | JSON |
| Project | `Project.load` | `Project.save` | JSON |
| Hysteretic bending law | — | `BendingLawExport.to_csv`, `.to_json`, `SlipScaleTable.to_csv` | CSV, JSON |

## Every file follows three rules

**Units are SI.** A key or column name carries its unit as a suffix: `_m`,
`_Pa`, `_N`, `_s`, `_1_m` (1/m), `_deg`. Angles in a cable file are in
degrees; the loaded layers also give them in radians.

**Text is UTF-8.** YAML, JSON and project files are UTF-8. The CSV and RIFLEX
readers accept UTF-8, with or without a byte-order mark, and Windows-1252.

**A bad file stops with a named error.** A file that cannot be read as
specified never gives a partial or approximate result. A missing file raises
`FileNotFoundError`. A layout problem raises `CableSchemaError`, which names
the file, the column or the line. A value that breaks the cable {term}`schema`
raises `pydantic.ValidationError`, and an invalid project or batch file
raises `CableInputError`.

## Cable definition (YAML)

A cable is one YAML file whose root is a mapping. `cx.load_cable(path)` reads
it, validates it against the schema in `cablexsec/cables/schema.py` and returns
an immutable `CableData`. The bundled definitions, such as
`cx.menard2023_yaml_path()` and `cx.iea_task49_66kv_yaml_path()`, are complete
examples with a cited source for every value. Section 4.1 of the
[local analysis guide](local_analysis_guide.md) lists them all.

### Top-level blocks

| Block | Required | Content |
| --- | --- | --- |
| `cable` | yes | metadata: `name` (required), `description`, `manufacturer`, `withheld_note`, `source_document`, `voltage_kV`, `design_life_years`, `DFF`, `MBR_installation_m`, `MBR_dynamic_m` |
| `layers` | yes | list of layers from the centre outwards |
| `environment` | no | thermal boundary: `ambient_C`, `T4_external_Km_W` (external thermal resistance, K m/W), `frequency_Hz` (default 50), `source` |
| `initial_state` | no | uniform pressure on the assembled cable before loading: `external_pressure_Pa` (default 0), `identification`, `source`, `bounds_Pa`, `calibration_scope` |
| `water_ingress` | no | `design` (`wet` or `dry`), `armour_annulus` (`flooded` or `sealed`), `source` |
| `core_lay` | no | assembly lay of offset cores: `pitch_m`, `lay_direction`, `source`; omit for cores aligned with the axis |
| `aggregate_properties` | no | published values for comparison, never inputs: `EA_fixed_MN`, `EI_min_kNm2`, `EI_max_kNm2`, `GJ_kNm2`, `OD_mm`, `mass_kg_m`, `EI_min_tension_kN`, `GJ_tension_kN`, `MBL_kN`, `MBL_basis` |
| `materials` | no | materials of this cable (below) |

An unknown key is an error at every level: at the top, inside the blocks and
inside the layers. A mistyped key is reported, never ignored.

A non-zero `initial_state.external_pressure_Pa` needs an `identification`
(`measured`, `manufacturing_specification`, `published_calibration`,
`identified` or `assumed_bound`) and a `source`. `assumed_bound` also needs
`bounds_Pa`. A `published_calibration` or `identified` pressure may state
`calibration_scope`: `armour_only` (fitted with the armour friction alone; the
value then already holds the friction of the other contacts and the laid-core
friction law warns with `ClampingCalibrationWarning`) or `all_contacts`
(fitted with friction on every contact, cores included). A `wet` design may not contain a radial metallic water barrier,
that is a `cylindrical_shell` or a lead or aluminium sheath. A `dry` design
must contain one.

### Layers

Every layer has these keys:

| Key | Required | Meaning |
| --- | --- | --- |
| `name` | yes | unique within the cable |
| `layer_type` | yes | `helical`, `helical_stranded`, `cylindrical`, `cylindrical_offset` or `cylindrical_shell` |
| `material` | yes | a key of the cable's `materials` block or of the material library |
| `source`, `notes` | no | where the values come from; free text |
| `structural` | no | `false`: no bending stiffness and not a fatigue site; the layer still carries pressure (default `true`) |
| `termination_scf` | no | stress-range factor at an end fitting (default 1.0) |

The geometry keys below may be written as `null` when they are withheld, as
described under [Withheld geometry](#withheld-geometry). Every analysis needs
them filled.

| `layer_type` | Geometry an analysis needs | Optional keys |
| --- | --- | --- |
| `helical` (armour wires) | `n_wires`, `d_wire_m`, `r_helix_m`, `lay_angle_deg` | `lay_direction` (`right_hand` default, `left_hand`), `manufacturing_clamping`, `bulk_sn`, `fretting_sn`, `transverse_slip_fraction`; flat or profiled wires: `wire_shape` (`round` default, `rectangular`, `profiled`), `wire_width_m`, `wire_thickness_m`, `wire_corner_radius_m`, `wire_area_m2`, `wire_I_normal_m4`, `wire_I_lateral_m4`, `wire_J_m4` |
| `helical_stranded` (strand within conductor within cable) | `n_conductors`, `r_cable_m`, `lay_angle_cable_deg`, `n_strands`, `r_strand_m`, `d_strand_m`, `lay_angle_strand_deg` | `bulk_sn`, `fretting_sn` |
| `cylindrical` (polymer annulus, bedding, filler, sheath, serving) | `r_inner_m`, `r_outer_m` | `tape` (helical tape winding), `strain_life`, `extrusion_shrinkage`, `construction` (`serving`) |
| `cylindrical_offset` (core component: conductor, insulation, screen) | `r_offset_m`, `r_inner_m`, `r_outer_m` | `n_cores` (default 3), `conductor_area_mm2`, `strand_layers`, `n_strands`, `strand_diameter_mm`, `strand_lay_angle_deg`, `construction`, `electrical`, `bulk_sn`, `fretting_sn`, `strain_life`, `outer_face_pressure`, `tape` |
| `cylindrical_shell` (lead or corrugated aluminium sheath) | `r_mid_m`, `t_m` | `f_corrugation` (default 1), `scf` (default 1), `construction` (`bonded_laminate`), `bulk_sn`, `fretting_sn`, `strain_life` |

`construction: serving` on a `cylindrical` layer declares a served yarn or a
wound tape. The serving keeps its mass, its radial and hoop stiffness (an
annulus of its material in plane stress along the cable axis, so it transmits
contact pressure and its thermal expansion acts in the cross-section plane)
and its shear compliance in the finite-support no-slip coefficient. It carries
no axial force, no bending moment and no torque: it adds nothing to the axial,
bending (stick or slip) or torsional stiffness and nothing to the load
sharing. In the component response it lies outside the load path, like a
loose screen: it is given the stress of the strain it follows, an upper bound
marked `kinematic`. A serving is non-structural
(`structural` defaults to `false`; `true` is refused), needs `r_inner_m > 0`
and is never an interstitial filler. Without `construction` a `cylindrical`
layer is a solid annulus, as before.

`lay_angle_deg` lies between −90° and 90°. A rectangular or profiled wire
fills in `d_wire_m` as the diameter of the round wire of the same area. A
given value must agree with it, and a layer whose wires overfill the
circumference is rejected.

`bulk_sn` and `fretting_sn` name registered S–N curves. An empty `fretting_sn`
selects bulk fatigue only, and an unknown or blocked name stops the analysis.
Sections 4.1, 4.2 and 13 of the local analysis guide describe the optional
keys.

### Manufacturing clamping

The {term}`contact pressure` that an armour layer carries at zero tension is
declared on the layer:

```yaml
    manufacturing_clamping:
      initial_contact_pressure_Pa: 2.0e6   # or armour_winding_stress_Pa
      basis: representative
      source: "Representative pressure jump P0; not a manufacturer value"
```

| Key | Meaning |
| --- | --- |
| `initial_contact_pressure_Pa` | the radial pressure jump `P0` across this layer [Pa], not a face pressure; jumps accumulate from the outside in |
| `armour_winding_stress_Pa` | residual wire stress from stranding [Pa], converted to a pressure jump |
| `basis` | `measured`, `manufacturing_specification`, `published_calibration`, `assumed`, `representative` or `not_published` |
| `source` | required |
| `calibration_scope` | `armour_only` or `all_contacts`, for a `published_calibration` value (as for `initial_state`) |

Exactly one of the two values is given. `not_published` requires the value
zero, and analyses that depend on the friction then warn with
`MissingClampingStateWarning`. The legacy keys `pretension_Pa` and
`winding_stress_Pa` still load. Section 4.4 of the local analysis guide
covers the clamping state in full.

An extruded sheath or bedding declares its frozen shrinkage, the part of its
cooling shrinkage that stays as a shrink fit on the layers inside it:

```yaml
    extrusion_shrinkage:
      hoop_strain: 0.004
      basis: assumed
      source: "Assumed frozen hoop shrinkage of the extruded outer sheath"
```

| Key | Meaning |
| --- | --- |
| `hoop_strain` | frozen in-plane shrinkage of the layer (hoop and radial; no axial part), 0 to 0.1 |
| `basis` | `measured`, `manufacturing_specification`, `assumed`, `representative` or `not_published` (which requires zero) |
| `source` | required |

The radial solutions carry it as an initial strain of the layer
(`contact_model="solved"`, including the ring around a trefoil bundle, and
`"qp"`); the bounded pressure chain refuses a non-zero value. An interstitial
filler cannot declare one. A shrinkage covers the clamping declaration of the
armour layers inside the layer.

### Materials with sources

A layer's `material` names a material of the library
(`cablexsec.materials.library.list_materials()`) or of the cable's own
`materials` block. A cable material with the key of a library material
replaces it for that cable only. Every property is a value and its source:

```yaml
materials:
  supplier_hdpe:
    polymer: true
    E_Pa: {value: 9.0e8, source: "supplier data sheet"}
    nu: {value: 0.46, source: "model assumption"}
```

A property may also state its `unit`, which must then be the SI unit below.
`polymer: true` applies the polymer rules, temperature warnings and
relaxation. `temperature_law` names a registered law of the modulus against
temperature, scaled to `E_Pa`.

| Property | Unit | Property | Unit |
| --- | --- | --- | --- |
| `E_Pa` | Pa | `mu_on_hdpe` | – |
| `E_eff_axial_Pa` | Pa | `mu_on_steel` | – |
| `E_eff_bending_Pa` | Pa | `mu_on_cu` | – |
| `E_corrugated_Pa` | Pa | `alpha_T_K` | 1/K |
| `nu` | – | `lambda_W_mK` | W/(m K) |
| `rho_kg_m3` | kg/m³ | `rho_cp_J_m3K` | J/(m³ K) |
| `creep_coeff`, `creep_exponent` | – | `rho_ohm_m` | Ω m |
| `sigma_y_Pa`, `sigma_UTS_Pa` | Pa | `alpha_R_per_K` | 1/K |
| `epsilon_allow` | – (a ratio) | `epsilon_r` | – |

Values outside the admissible range of a property are rejected. Materials and
S–N curves can also be registered for a session with `cx.register_material`
and `cx.register_sn_curve`.

### Withheld geometry

A definition may write a geometry value that cannot be shared as `null` and
supply it at load time from a second YAML file:

```python
cable = cx.load_cable("cable.yaml", geometry_override="geometry.yaml")
```

The override mirrors the `layers` list. Its layers are matched by `name`, and
their keys replace those of the definition. It may also hold
`aggregate_properties`, `initial_state` and `core_lay`, and a `cable.name`
that must match the definition. Anything else is refused with the item
named: a layer name that is not in the definition, an unknown or misspelt key,
a layer given twice, a `layer_type` that differs from the definition, or an
override that supplies nothing. It never adds layers.

An override path that does not exist is ignored and the `null` values stay in
place. `cable.missing_fields()` lists them, and analyses refuse a cable that
is not fully specified.

### A minimal complete definition

The bundled 33 kV single-core template
(`cx.inter_array_single_core_yaml_path()`), without its comments and notes:

```yaml
cable:
  name: "Single-core inter-array 33 kV (public representative)"
  description: "33 kV single-core dynamic inter-array cable, polymer sheath"
  manufacturer: "Public representative template"
  source_document: "Representative public geometry (topology template)"
  voltage_kV: 33
  design_life_years: 25
  MBR_dynamic_m: 2.5

environment:
  ambient_C: 15.0
  T4_external_Km_W: 0.5
  frequency_Hz: 50.0
  source: "Representative seabed boundary (topology template)"

layers:
  - name: cu_conductor
    layer_type: cylindrical_offset
    material: copper_stranded
    n_cores: 1
    r_offset_m: 0.0
    r_inner_m: 0.0
    r_outer_m: 0.0097720502380584
    conductor_area_mm2: 300.0
    construction: aligned_solid_equivalent
    source: "Representative"

  - name: conductor_screen
    layer_type: cylindrical_offset
    material: xlpe_qiao2025
    n_cores: 1
    r_offset_m: 0.0
    r_inner_m: 0.0097720502380584
    r_outer_m: 0.01077
    source: "Representative; thermal treatment [IEC60287-2-1]"

  - name: xlpe_insulation
    layer_type: cylindrical_offset
    material: xlpe_qiao2025
    n_cores: 1
    r_offset_m: 0.0
    r_inner_m: 0.01077
    r_outer_m: 0.01877
    electrical:
      permittivity_relative: 2.3
      loss_factor_tan_delta: 0.0004
      vec_exponent_n: 15.0
      thermal_b_k: 12430
      vec_synergism_b: 4420
      field_threshold_kV_mm: 5.0
    source: "Representative; electrical [Mazzanti2007] Table I (XLPE)"

  - name: insulation_screen
    layer_type: cylindrical_offset
    material: xlpe_qiao2025
    n_cores: 1
    r_offset_m: 0.0
    r_inner_m: 0.01877
    r_outer_m: 0.01977
    source: "Representative; thermal treatment [IEC60287-2-1]"

  - name: cu_screen
    layer_type: cylindrical_offset
    material: copper_tape_screen
    n_cores: 1
    r_offset_m: 0.0
    r_inner_m: 0.01977
    r_outer_m: 0.02047
    structural: false
    source: "Representative"

  - name: hdpe_bedding
    layer_type: cylindrical
    material: hdpe
    r_inner_m: 0.02047
    r_outer_m: 0.02247
    source: "Representative"

  - name: inner_armour
    layer_type: helical
    material: galvanised_steel_armour
    n_wires: 34
    d_wire_m: 0.004
    r_helix_m: 0.02447
    lay_angle_deg: 25.0
    lay_direction: right_hand
    manufacturing_clamping:
      initial_contact_pressure_Pa: 2.0e6
      basis: representative
      source: "Representative pressure jump P0 of this demonstration template; not a manufacturer or published value"
    fretting_sn: ""
    bulk_sn: saevik_2011_flat_steel
    source: "Representative"

  - name: outer_armour
    layer_type: helical
    material: galvanised_steel_armour
    n_wires: 40
    d_wire_m: 0.004
    r_helix_m: 0.02847
    lay_angle_deg: 25.0
    lay_direction: left_hand
    manufacturing_clamping:
      initial_contact_pressure_Pa: 2.0e6
      basis: representative
      source: "Representative pressure jump P0 of this demonstration template; not a manufacturer or published value"
    fretting_sn: ""
    bulk_sn: saevik_2011_flat_steel
    source: "Representative"

  - name: hdpe_outer_sheath
    layer_type: cylindrical
    material: hdpe
    r_inner_m: 0.03047
    r_outer_m: 0.03447
    source: "Representative"

aggregate_properties:
  OD_mm: 68.9
```

The radii follow each other without gaps. The armour wires sit on the bedding,
so `r_helix_m` is the bedding outer radius plus the wire radius, and every
material is a key of the library. `tests/test_tutorials_03_04.py` loads this
block and checks that it gives the stiffness of the bundled file.

## Load histories

Every reader returns a `LoadCaseSeries`, the history of one cable section:

| Field | Unit | Meaning |
| --- | --- | --- |
| `time_s` | s | sample times, strictly increasing; a uniform step is expected |
| `curvature_x_1_m`, `curvature_y_1_m` | 1/m | strain-gradient components (below) |
| `tension_N` | N | axial force, of the kind `tension_kind` |
| `tension_kind` | – | `true`, `effective` or `unspecified` |
| `axial_strain`, `axial_strain_kind` | – | optional strain; `direct`, `total` or `unspecified` |
| `temperature_C` | °C | optional; `None` for an isothermal record |
| `source`, `provenance` | – | where the record came from |

The arrays are read-only copies. `dataclasses.replace` makes an edited series.

### Curvature is held as a strain gradient

CableXSec takes the {term}`strain gradient` `(g_x, g_y)`. A fibre at
`(x, y) = r (cos φ, sin φ)` of the section has the bending strain
`g_x x + g_y y`, and the gradient points to the most tensile fibre.

Global line programs write the {term}`rotation curvature` `(k_x, k_y)`, the
components of `t × dt/ds` on the section axes, with fibre strain
`k_x y − k_y x`. The readers convert it with

```text
g_x = −k_y,   g_y = k_x
```

This is `cablexsec.mechanics.section_conventions.rotation_to_strain_curvature`.
A sag bend of a line in a vertical plane, `k_y < 0`, is then in tension at
`φ = 0`. CableDyn 0.4.0 writes the curvature vector towards the centre of
curvature, `(CurvX, CurvY)`, and its reader takes `g = −(CurvX, CurvY)`.

A curvature magnitude cannot represent reversals of the bending direction and
is refused. Section 2 of the [global–local guide](global_local_guide.md)
describes the sign checks.

### The force kind must be declared

The wire stress needs the force that the armour carries. `fatigue_from_series`
and the project fatigue analysis accept `true` tension only, unless the record
holds a direct strain that drives the wire stress instead.

For a sealed armour annulus, {term}`effective tension` is converted with
`series.with_true_tension(external_pressure_Pa, external_area_m2)`. For a
flooded annulus the effective tension loads the armour and is passed to
`compute_fatigue` directly, as the sea-state batch does with
`tension_basis="effective"`. An `unspecified` force is refused until its kind
is declared.

Of the strains, only `direct`, the {term}`direct tensile strain`, drives the wire
stress. A `total` strain contains the Poisson contraction and is refused.

### Time-history CSV export

`cx.read_time_history_csv(path, ...)` reads a comma-separated file with one
header row of column names and one row per sample:

```text
Time (s),x curvature (rad/m),y curvature (rad/m),Effective tension (kN)
0,-0.000958504,-0.0483883,59.2063
0.2,-0.00121411,-0.0488292,59.5659
```

| Argument | Default | Selects or sets |
| --- | --- | --- |
| `time_col` | `"Time"` | time [s] |
| `curvature_x_col` | `"x curvature"` | first curvature component |
| `curvature_y_col` | `"y curvature"` | second component; `None` declares it zero |
| `tension_col` | `"Wall tension"` | axial force |
| `axial_strain_col` | `None` | optional strain column |
| `temperature_col` | `None` | optional temperature [°C] |
| `curvature_convention` | `"rotation"` | `"rotation"` or `"strain_gradient"` |
| `curvature_scale_to_1_m` | `1.0` | multiplier to 1/m |
| `tension_scale_to_N` | `1.0e3` | multiplier to N (the default reads kN) |
| `axial_strain_scale` | `0.01` | multiplier to a ratio (the default reads percent) |
| `tension_kind`, `axial_strain_kind` | inferred | override of the kind |

A selector matches a column whose heading contains it as whole words, without
regard to case or extra white space, and a heading equal to the selector is
taken first. It must match exactly one column: no match, two matches, or two
selectors on one column stop the read. A heading that denotes a magnitude or
an extreme (`max`, `min`, `magnitude`, `abs`, `resultant`) is refused for a
curvature component.

Units are not read from the headings. The scale arguments convert them.

The {term}`tension kind` is inferred from the heading. `wall tension` and
`true tension` give `true`, `effective tension` gives `effective`, and any
other heading gives `unspecified`. `direct tensile strain` gives the strain
kind `direct` and `total mean axial strain` the kind `total`. An explicit kind
may not contradict a recognised heading.

Blank lines are skipped and other columns are ignored. A header separated by
semicolons or tabs is refused. Every data row must have as many fields as the
header, and a non-numeric value is refused with an error that names the line
and column.

### CableXSec exchange CSV

`cx.write_series_csv(series, path)` writes, and `cx.read_series_csv(path)`
reads, a fixed layout that keeps every kind:

| Column | Content |
| --- | --- |
| `time_s` | time [s] |
| `curvature_x_1_m`, `curvature_y_1_m` | strain-gradient components [1/m] |
| `effective_tension_N`, `true_tension_N` or `axial_tension_N` | the force [N]; the column name gives its kind (`axial_tension_N` is `unspecified`) |
| `temperature_C` | temperature [°C]; empty cells for an isothermal record |
| `direct_axial_strain`, `total_axial_strain` or `axial_strain` | optional strain [–]; empty cells when absent |

The first three columns and exactly one tension column are required. Values
are written with full precision, so a series survives the round trip
unchanged. The bundled demonstration history
`validation/paper_data/global_demo/global_history_si.csv` is in this format.

### RIFLEX ASCII time history

`cx.read_riflex_ascii(path, ...)` reads whitespace-separated columns, one row
per sample. Lines starting with `#`, `%` or `!` and blank lines are skipped.
Columns are selected by their 1-based position, not by name:

| Argument | Default | Meaning |
| --- | --- | --- |
| `time_col`, `curvature_x_col`, `tension_col` | `1`, `2`, `3` | required columns |
| `curvature_y_col`, `temperature_col` | `None` | optional columns (`None`: planar, isothermal) |
| `curvature_scale_to_1_m`, `tension_scale_to_N` | `1.0`, `1.0` | unit multipliers |
| `tension_kind` | `"unspecified"` | declare `true` or `effective` from the selected result |
| `curvature_convention` | `"strain_gradient"` | `"rotation"` for rotation components |

The defaults differ from the CSV reader. Unless declared otherwise, the
curvature is taken as strain gradient and the tension in N.

### CableDyn outputs

Driver and OpenFAST module runs write CableDyn 0.1.x output.
`cx.read_cabledyn_sections(out_root, deck, {"name": arc_length_m, ...})` reads
the output files of one run together with the deck: `<root>.out`,
`<root>.Line<L>.p.out`, `<root>.Line<L>.t.out`, `<root>.static.out` and
`<root>.elements.out`, or `<root>.CD.out` from OpenFAST. It rebuilds the
signed curvature vector of each named section from the tangent channels or
node positions. It returns a `CableDynSectionHistory` per section with the
effective tension, rotation curvature components, hydrostatic pressure,
elevation, torque and twist. `history.to_load_case_series(A_o)` gives the
true-force series with strain-gradient curvature. Sections 1 and 2 of the
[global–local guide](global_local_guide.md) list the channels the deck must
request, the frame and the consistency checks.

CableDyn 0.4.0 writes station tables.
`cablexsec.io.read_cabledyn_station_table(path)` reads one station history:

| Column | Accepted names (any case) | Unit |
| --- | --- | --- |
| time | `time`, `time_s`, `Time(s)`, `t` | s |
| effective tension | `Ten`, `Ten_N`, `Ten(N)`, `effective_tension_N` | N |
| curvature components | `CurvX`, `CurvX_1_m`, `CurvX(1/m)`; `CurvY`, ... | 1/m |

The table is comma, tab or whitespace separated with one header row. Lines
starting with `#` are comments, and other columns are ignored. The series
holds effective tension and `g = −(CurvX, CurvY)`, with wire azimuths measured
from CableDyn's section `x` axis. `right_handed=False` reverses `y` for a
left-handed triad. `cablexsec.io.series_from_curvature_components` builds the
same series from arrays.

### Synthetic records

`cx.jonswap_series(...)` generates a record whose curvature has the shape of a
{term}`JONSWAP` wave spectrum and a chosen RMS, with optional dynamic tension and
out-of-plane curvature. It is for demonstrations and tests. It is not the
response of a global model.

## Sea-state batch specification

`cablexsec-batch batch.json --output-directory <new folder>` takes a JSON
object with four keys:

| Key | Content |
| --- | --- |
| `cable` | a YAML path, or `bundled:<file stem>` for a bundled cable such as `bundled:iea_task49_66kv` |
| `runs` | `name`, `out_root`, `deck`, `probability`; optional `line_id`, `start_time_s`, `end_time_s` |
| `sections` | `name`, `arc_length_m`; optional `termination` (the fields of `TerminationSpec`, for example `distance_m` and `k_anchor_N_m`; a record without `k_anchor_N_m` has a rigid anchor), `thermal`, `fretting`, `sn_curves`, `mean_stress`, `scf` |
| `options` | the fields of `SeaStateBatchOptions` |

Relative paths are resolved against the folder of the JSON file, and unknown
keys are rejected. The output folder receives `manifest.json`, with the
SHA-256 digests of every input, the options, checks and results, together
with `<section>_annual_damage.npz` and `<section>_<run>_history.npz`.
Section 3 of the [global–local guide](global_local_guide.md) gives an
example.

## Project files

`cx.Project.save(path)` writes, and `cx.Project.load(path)` reads, one
self-contained UTF-8 JSON file. This manual names them `.cxproj`, but the
extension is not checked.

```text
{"format": "CableXSec Project", "version": 2, "sha256": "<digest>", "payload": {...}}
```

| Payload key | Content |
| --- | --- |
| `name` | project name |
| `engine` | CableXSec version, source digest, Python, platform, dependency versions and the registered materials and curves at creation |
| `models` | cable definitions by name, in full |
| `load_cases` | load histories by name, with complete arrays |
| `analyses` | analysis requests (name, model, load case, settings) |
| `records` | completed runs: inputs, result, engine snapshot, fingerprint, run identifier and time |

The file does not refer to the original YAML or CSV files. Arrays are stored
as tagged lists with their type and shape, and non-finite numbers as tagged
values.

`sha256` is the digest of the canonical payload. `Project.load` rejects a
mismatch, an unknown version, duplicate keys and result types that do not
match their analysis with `CableInputError`, and it decodes only a fixed list
of data and result classes. Version 1 files load; saving writes version 2.
`Project.save` writes a temporary file in the target folder and replaces the
target in one step. The [project guide](project_guide.md) describes the
objects and the stale-result checks.

## Bending law for global analysis

`cx.export_bending_law(cable, tensions_N=..., kappa_max_1_m=...)` returns a
`BendingLawExport` with three outputs:

- `to_csv(path)`: the virgin moment–curvature {term}`backbone` at every exported
  tension, one row per curvature:

  ```text
  kappa_1_m,M_0N_Nm,M_20000N_Nm
  0.000000000e+00,0.000000000e+00,0.000000000e+00
  2.000000000e-05,1.127743287e+00,1.127743287e+00
  ```

  The first column is the curvature [1/m]; each further column is the moment
  [N m] at the tension named in its heading (`M_<tension>N_Nm`).
- `to_json(path)`: an object with `format` (`cablexsec.bending_law/1`; `/2`
  with a slip scale, below),
  `hysteresis_rule` (`masing_iwan`), `planar_only`, `tension_N`, `kappa_1_m`,
  `moment_Nm` (one backbone per tension), `ei_slip_Nm2` (the full-slip
  stiffness per tension), `elements` (the stiffness and yield curvature of the
  parallel elastic–perfectly plastic elements per tension), `settings`,
  `method` and `limitations`.
- `backbone_table_kNm(tension_N)`: the backbone at one tension as pairs of
  curvature [rad/m] and moment [kN m], the input of a moment–curvature table of
  a {term}`global line model`.

The law is planar and its contact state is fixed at each tension. Between
tensions it is interpolated linearly, as Section 8 of the
[global–local guide](global_local_guide.md) explains.

### Slip scale (format version 2)

`export_bending_law(..., slip_scale=True)` adds the slip scaling variable
(`BendingLawExport.slip_scale`, a `SlipScaleTable`; guide Section 8.1). The
JSON law then has `format` `cablexsec.bending_law/2` and one more object,
`slip_scale`; every version-1 field is unchanged, and a law without a slip
scale is still written as version 1, byte for byte as before. The object holds:

| Key | Content |
| --- | --- |
| `rule` | `yield_curvature_scaling` |
| `variables` | `["tension_N"]`, or `["tension_N", "external_pressure_Pa"]` with a pressure axis |
| `reference_tension_N`, `reference_external_pressure_Pa` | state of the reference law |
| `tension_N`, `external_pressure_Pa` | table axes (`external_pressure_Pa` is `null` without a pressure axis) |
| `ei_slip_Nm2` | full-slip stiffness [N m²] |
| `reference_elements` | `stiffness_Nm2`, `yield_curvature_1_m` of all groups merged at the reference |
| `scale` | single scale `s`, one row per tension, one column per pressure (one column without a pressure axis) |
| `dscale_dtension_1_per_N` | `∂s/∂T` [1/N], same shape |
| `dscale_dexternal_pressure_1_per_Pa` | `∂s/∂p_e` [1/Pa], same shape (pressure axis only) |
| `groups` | one object per friction group: `name` (the armour layer, `laid_cores` or `filler_profiles`), `kind` (`armour_layer`, `laid_cores` or `filler_profiles`), `reference_cap_Nm`, the group's `stiffness_Nm2` and reference `yield_curvature_1_m`, its `scale` `ρ_g` and the derivatives `dscale_dtension_1_per_N` and `dscale_dexternal_pressure_1_per_Pa` |

At a state with group scales `ρ_g` the law is `M = ei_slip_Nm2 κ + Σ k_i e_i`
over all group elements, with every element of group `g` updated by
`e_i ← clip(e_i + Δκ, −ρ_g κ_y,i, ρ_g κ_y,i)` (the return map with the cap of
the current state). A reader that carries one scale uses `reference_elements`
and `scale` in place of `ρ_g`; that is exact only when the groups scale alike
(guide Section 8.1). The table is interpolated linearly in tension and
pressure; the derivatives are those of the contact model at the table states.

`SlipScaleTable.to_csv(path)` writes the table as CSV, one row per state
(tension, then pressure):

```text
tension_N,external_pressure_Pa,s,ds_dT_1_per_N,s_inner_armour,ds_dT_inner_armour_1_per_N,s_outer_armour,ds_dT_outer_armour_1_per_N
2.000000000e+04,0.000000000e+00,1.000000000e+00,5.000000000e-05,...
```

With a pressure axis the columns `ds_dpe_1_per_Pa` and
`ds_dpe_<group>_1_per_Pa` follow each `ds_dT` column. `Project` files store the
table with the law; project files written before it existed still load.

`cablexsec.io.global_export` also writes the section properties of a set of
load cases as neutral CSV tables (`section_properties_csv`,
`bending_tables_csv`), each preceded by comment lines starting with `# ` that
state the cable, the load cases, the units and the source of every value.
