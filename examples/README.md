<!-- SPDX-License-Identifier: Apache-2.0 -->

# Examples

Every script runs as it is from any folder once CableXSec is installed, uses only the
cable definitions bundled with the package, and prints its results with the
provenance and limitations of each calculation. Inputs marked synthetic are
illustrations, not site or design data.

```powershell
python examples/demo_02_mkappa_validation.py
python examples/project_workflow.py --output my_project.cxproj
```

| Script | What it shows | Run time |
|---|---|---|
| `demo_02_mkappa_validation.py` | Hysteretic moment–curvature loop of the published Ménard and Cartraud (2023) cable against its full-slip and no-slip bounds | seconds |
| `demo_03_stress_distribution.py` | Per-wire stress components (axial, local bending, friction) over the circumference | seconds |
| `demo_05_fatigue_damage.py` | Rainflow counting and Miner damage at fixed material fibres of every armour wire | seconds |
| `long_term_bulk_fatigue.py` | Probability-weighted long-term bulk fatigue over a synthetic scatter of sea states | seconds |
| `long_term_termination_fatigue.py` | Lifetime projection with explicit termination range factors | seconds |
| `hertz_axis_stress.py` | Depth-resolved Hertz normal-contact stress between two obliquely crossed steel wires | seconds |
| `prescribed_contact_swt.py` | SWT critical-plane recovery at a subsurface point under prescribed contact tractions | seconds |
| `critical_plane_swt.py` | Single-point, one-cycle SWT calculation and its sampling refinement | seconds |
| `project_workflow.py` | Run a project (cable, load case, analysis), save it, reopen it and check its provenance | seconds |
| `rig_test_gauge_plan.py` | Strain-gauge plan for a combined tension–bending rig test of the public 33 kV three-core cable | under a minute |
| `torque_balance_templates.py` | Torque balance and twist under tension of the public 66 kV template and the example cables | one to two minutes |
| `thies2016_screens_sheaths.py` | Bend-restrictor kinematics, copper screens, tapes, laminate sheath and armour of the cable of Thies et al. (2016) | a few minutes |

`_console.py` is a small helper that the scripts share for printing on consoles that
cannot display every character.

`fowt/menard2023_global_mass.json` records the mass assumptions with which the
published Ménard cable was given global properties for the floating-wind application
of the companion article (see
[`validation/paper_data`](https://github.com/SMI-Lab-Inha/CableXSec/blob/main/validation/paper_data/README.md)).

The global-to-local command `cablexsec-batch` is described in the
[global–local guide](https://cablexsec.readthedocs.io/en/latest/global_local_guide.html);
its input is a set of CableDyn runs, of which `tests/data/cabledyn/` holds small
examples.
