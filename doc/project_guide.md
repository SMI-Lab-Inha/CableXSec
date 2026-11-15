<!-- SPDX-License-Identifier: Apache-2.0 -->

# Project guide

A CableXSec project keeps cable designs, load histories, analysis requests and
completed runs together in one self-contained file. Each completed run stores
its inputs, result and calculation environment, so it can be reopened,
inspected and checked against the current inputs later. The implementation is
`cablexsec/project.py`. The function API in [local_analysis_guide.md](local_analysis_guide.md)
remains the full interface; a project wraps a subset of it.

## Objects

| Object | Role |
| --- | --- |
| `CableModel` | Owned, validated cable definition. `CableModel.from_yaml(path)` loads one (no geometry override unless passed). `.definition` returns an independent `CableData`. Methods `stiffness()`, `stress(settings)`, `bending(series, settings)`, `fatigue(series, settings)` call the function API. |
| `LoadCaseSeries` | Global history at one section (see the local analysis guide, Section 4.3). |
| `AnalysisRequest` | `name`, `model`, `settings` and, for history analyses, `load_case`. |
| `Project` | Named models, load cases and requests, plus the list of completed `AnalysisRecord`s. |
| `AnalysisRunner` | Synchronous, stateless executor used by `Project.run`. |
| `AnalysisRecord` | One completed run: `run_id`, `created_utc`, `fingerprint`, and accessors `model`, `series`, `request`, `inputs`, `engine`, `result`. |

Supported request types:

| Settings | Needs `load_case` | Calls | Scope |
| --- | --- | --- | --- |
| `StiffnessSettings()` | no | `compute_stiffness` | Section stiffness bounds |
| `StressSettings(kappa_x_1_m=..., kappa_y_1_m=..., tension_N=..., ...)` | no | `compute_stress` | One curvature; optional pressure, torque and contact model |
| `BendingSettings(mu=None, contact_model="bounded")` | yes | `compute_mkappa` | Planar x-curvature at constant true tension; y-curvature, temperature and strain histories are rejected |
| `FatigueSettings(contact_tension_N=None, mu=None, T_ref_C=0.0, n_fibres=32)` | yes | `fatigue_from_series` | The series fatigue bridge and its conventions |

Settings are frozen Pydantic models with `extra="forbid"`: an unknown or
mistyped option raises `pydantic.ValidationError` instead of being ignored.
Solved section states, thermal profiles, capacity, long-term aggregation and
contact indices are not project request types; use the functions directly.

## Example

```python
import dataclasses
import numpy as np
from cablexsec import (
    AnalysisRequest, CableModel, FatigueSettings, LoadCaseSeries, Project,
    StiffnessSettings, inter_array_single_core_yaml_path,
)

project = Project("Single-core inter-array screening")
project.set_model("design", CableModel.from_yaml(inter_array_single_core_yaml_path()))
t = np.linspace(0.0, 120.0, 1201)
project.set_load_case("operating", LoadCaseSeries(
    t, 0.008 * np.sin(t), 0.003 * np.cos(t), np.full_like(t, 5.0e4),
    tension_kind="true", source="synthetic demonstration history",
))
project.set_analysis(AnalysisRequest(
    name="stiffness", model="design", settings=StiffnessSettings()))
project.set_analysis(AnalysisRequest(
    name="bulk fatigue", model="design", load_case="operating",
    settings=FatigueSettings(n_fibres=16)))

project.run("stiffness")
record = project.run("bulk fatigue")
print(record.result.critical_layer, record.run_id, record.created_utc)

project.save("screening.cxproj")
reopened = Project.load("screening.cxproj")
print([reopened.is_stale(r) for r in reopened.records])          # [False, False]

series = reopened.load_cases["operating"]
reopened.set_load_case("operating",
                       dataclasses.replace(series, tension_N=series.tension_N * 1.2))
print([reopened.is_stale(r) for r in reopened.records])          # [False, True]
rerun = reopened.run("bulk fatigue")                             # appends a third record
```

A longer script is `examples/project_workflow.py --output design.cxproj`.

## Editing

`set_model`, `set_load_case` and `set_analysis` validate before replacing the
named entry, and `set_analysis` checks that the referenced model and load case
exist. The `models`, `load_cases` and `analyses` properties are read-only
mappings; edit by replacing an entry. Histories are read-only snapshots, so
use `dataclasses.replace` to change one. Names must be non-empty without
surrounding whitespace. Records cannot be edited or deleted, and there is no
method to remove a model, load case or request.

## Stale results

`project.run(name)` appends a new `AnalysisRecord` and never overwrites an
earlier one; a failed run leaves the existing records unchanged.

`record.fingerprint` is a SHA-256 over the run's inputs (cable definition,
request and history) and its engine snapshot. `project.is_stale(record)`
recomputes that digest from the current entry of the same analysis name, its
referenced model and load case, and today's engine:

- Editing the model, load case or settings used by that analysis makes the
  record stale. Edits to unrelated entries do not.
- Restoring identical inputs restores the match.
- Any change to the package source, the Python version, the platform, the
  numpy/scipy/pydantic/rainflow versions, the optional Clarabel version or the
  material, S–N or SWT registries makes every record stale. The source digest
  covers every `.py` file in `cablexsec/`, so a code change marks all records
  stale even if it does not affect the result.

A stale record is kept for comparison; run the analysis again to create a
current one.

## Provenance

Each record holds its own engine snapshot (`record.engine`): CableXSec
version, source digest, Python and platform, dependency versions and the
complete registered material and curve data at run time. `record.model`,
`record.series`, `record.request` and `record.result` rebuild independent
objects; editing them cannot change the stored record. The result keeps its
own `provenance` (method, citations, gates, limitations).

`project.engine` is the environment at project creation, and
`project.environment_changed` reports whether today's environment differs from
it. Archived registry data is a record only: opening a project never changes
the installed registries, and new runs use the installed library.

## Analysis families (format version 2)

| `kind` | Settings | Load case | Result |
| --- | --- | --- | --- |
| `stiffness` | `StiffnessSettings` | no | `StiffnessResult` |
| `stress` | `StressSettings` | no | `StressResult` |
| `bending` | `BendingSettings` (planar, constant true tension) | yes | `MkappaResult` |
| `fatigue` | `FatigueSettings` | yes | `FatigueResult` |
| `thermal` | `ThermalSettings` (steady IEC 60287 rating) | no | `ThermalResult` |
| `capacity` | `CapacitySettings` (tension sweep) | no | `CapacityResult` |
| `coiling` | `CoilingSettings` (static tank) | no | `CoilingResult` |
| `conductor_fatigue` | `ConductorFatigueSettings` (curvatures, true tension) | yes | `ConductorFatigueResult` |
| `bending_law` | `BendingLawSettings` (export for global analysis, optional slip scale) | no | `BendingLawExport` |

## Preflight report

`cablexsec.preflight.preflight(project)` checks every analysis against its cable and load case without running it
and returns a `PreflightReport` (`errors`, `warnings`, `ok`, `to_text()`); `preflight_request(cable, request, series)`
checks one request. Errors are combinations the package refuses at run time: an incomplete cable, the QP contact
solver on offset (three-core) cores, a resolved strand bundle in a radial solver or coiling, a blocked or unknown S–N
curve, conductor fatigue without a resolved conductor, the surface trellis rule without a fretting law, a load case
with the wrong tension kind or a non-planar history for planar bending. Warnings mark runs that succeed but rest on
an assumption: no declared manufacturing clamping, an armour layer without `bulk_sn`, the unvalidated surface trellis
rule, tension levels of an exported bending law more than 20 kN apart, and the planar limit of that law. The checks
mirror the run-time validations; a run can still fail on the solution itself (non-convergence, opened contacts).

## File format version 2

`PROJECT_FORMAT_VERSION` is 2. A `.cxproj` file is UTF-8 JSON with an envelope
`{"format": "CableXSec Project", "version": 2, "sha256": ..., "payload": ...}`.
The payload contains the project name, creation engine, cable definitions,
complete history arrays, requests and records. It does not refer to the
original YAML or CSV files. Non-finite numbers use tagged values rather than
non-standard JSON.

`Project.save(path)` writes and flushes a temporary file in the target
directory, then replaces the target atomically; the directory must exist.
`Project.load(path)` decodes only an allowlist of data and result classes (no
pickle, no dynamic imports) and rejects an unknown version, a checksum
mismatch, duplicate keys, inconsistent references, mismatched result types and
duplicate run identifiers with `CableInputError`. The checksum detects
accidental changes; it is not a signature. Version 1 files (stiffness, stress, bending and fatigue only) load
unchanged; a version 1 file that names a newer analysis kind is rejected. Saving always writes version 2.

## Limits

- Arrays are stored as JSON: easy to inspect, but large for long multi-section
  campaigns.
- Runs are synchronous. Independent records can be produced by
  `AnalysisRunner().run(model, request, series)` in worker processes; keep
  project edits on one thread.
- The engine snapshot records the installed numba version (`null` when not
  installed) but not whether `CABLEXSEC_DISABLE_NUMBA` was set. The compiled
  kernels reproduce the reference results, so this does not change damage.
- Project-local material or S–N catalogues are not supported: curves
  registered with `register_sn_curve` (and materials with
  `register_material`) live in the session and are recorded in the
  engine snapshot, but a project file does not carry them; mean-stress
  and component-SCF options are not yet fields of `FatigueSettings`.
