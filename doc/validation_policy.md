<!-- SPDX-License-Identifier: Apache-2.0 -->

# Validation policy

CableXSec is an executable research model. What it may claim about a physical
cable is limited by the comparisons attached to each output quantity. A passing
test suite, a complete feature list or a fast run time does not substitute for
a comparison with measurement.

This file defines the validation tiers, the rules every comparison follows, how
`validation/scripts/run_validation.py` reports status, and how a new benchmark is
registered. The current register of comparisons is in
[benchmarks.md](benchmarks.md); the generated scorecard is
[validation/validation_report.md](https://github.com/SMI-Lab-Inha/CableXSec/blob/main/validation/validation_report.md).

## Verification and validation

**Verification** asks whether the equations are solved correctly. It uses
closed-form solutions, invariants, conservation laws, limiting cases,
convergence studies, solver optimality conditions and agreement between two
independent code paths of CableXSec. Verification says nothing about whether
the equations describe a real cable.

**Validation** asks whether the model describes a physical cable. It requires a
comparison with measured data for a stated construction, loading path and output
quantity. A comparison with another model (a published finite-element
calculation, an analytical result from a paper, or a commercial program) is a
**model-to-model comparison**. It can expose errors and confirm that two
formulations agree at matched inputs, but it is not validation, and the other
model is a reference, not ground truth.

## Validation tiers

Each gate in `run_validation.py` carries a tier number. The tier describes where
the reference comes from; the comparison class in the register states whether
that reference is analytical, measured or another model.

| Tier | Reference | Typical content | Establishes | Does not establish |
| --- | --- | --- | --- | --- |
| 1 | Closed-form result, invariant or second CableXSec code path | Lamé solutions, energy closure, KKT residuals, cross-engine moment recovery, I/O known answers | The implementation solves its stated equations | Fidelity to any physical cable |
| 2 | Public source: measurement or published model | Published bending loops, published analytical stiffness, published thermal network, published umbilical tests | Agreement, or disagreement, with a reproducible public case | General validity outside the tested construction and loading |

A tier is not a quality ranking. A Tier-2 row may be a model-to-model
comparison (for example the published full-slip stiffness of Ménard and
Cartraud), and a Tier-2 row may be a failure. The register records the class of
every comparison explicitly as one of:

- **Analytical**: closed form or exact limit.
- **Measured**: physical test data.
- **Model-to-model**: published FE, published analytical model or commercial
  program at matched inputs.
- **Internal**: two CableXSec formulations of different fidelity on the same
  inputs (for example section law against spatial armour model).

Measured and model-to-model results are kept in separate rows and are never
averaged or pooled into one accuracy figure.

## Rules

1. **State the quantity.** Every comparison names the exact output quantity,
   construction, loading path, metric and target.
2. **Declare the target before the result.** Targets are set from the source's
   own uncertainty, the digitisation resolution or a stated engineering
   tolerance, not from the value the model happens to produce.
3. **No tuning to the comparison target.** A parameter inferred from a response
   may not be used to validate that same response. If friction, clamping,
   support rigidity, a stress-concentration factor or any transfer coefficient
   is fitted, the case is labelled *identified*, the fitted data are named, and
   only data withheld from the fit count as a prediction. Identified and
   transferred-input results are reported side by side.
4. **Parameter independence.** Geometry, material properties, manufacturing
   state, boundary conditions and environment must come from a source
   independent of the compared output. When an initial contact state was not
   measured, the result is reported as a bound, an identifiability finding or
   not validated.
5. **Measured and model-to-model results stay separate.** A stiffness or
   displacement agreement with another model cannot stand in for agreement with
   measured moment, work or stress.
6. **Matched inputs for model-to-model comparisons.** Identical geometry,
   moduli, boundary conditions, load history and output definitions. Results at
   CableXSec's library moduli are reported separately, marked INFO, so that a
   modulus choice is not mistaken for a model difference.
7. **State digitisation uncertainty.** Curves read from published figures carry
   their reading resolution (line half-widths in curvature and moment), which is
   propagated to every reported error and work value. It is described as reading
   resolution, not as experimental uncertainty. The source file's SHA-256 is
   recorded at extraction.
8. **Units are compared as printed.** No undocumented conversion is inferred
   (for example a source table printed in J is not divided by an assumed
   specimen length to compare with an integral in J/m).
9. **Failing targets are reported, not loosened.** A target is never widened,
   and a failing row is never removed, after the result is known. A test
   (`tests/test_validation_report.py`) pins the Ménard ±0.10 work comparison to
   its unchanged 10% target and its FAIL status.
10. **Partial and negative results stay visible.** PARTIAL, NOT VALIDATED and
    NOT COMPARABLE rows remain in the scorecard and the register.
11. **No transfer of scope.** A component curve, a numerical kernel check or a
    section-property agreement is not a cable-life validation. Fatigue damage
    has not been compared with measured wire stress or measured life.
12. **No ranking claims.** No comparison is used to state that CableXSec is
    faster than, better than or equivalent to any other program.
    Run times describe the cost of CableXSec's own calculations only.

## How `run_validation.py` reports status

`python validation/scripts/run_validation.py` evaluates every gate from the live code and
writes [validation/validation_report.md](https://github.com/SMI-Lab-Inha/CableXSec/blob/main/validation/validation_report.md), which must not be edited
by hand. `--check` recomputes the rows without writing and stops if the
committed file is stale.

Each row has a tier, an identifier, a quantity, a target, a value and a status:

| Status | Meaning |
| --- | --- |
| PASS | The named gate met its declared target. It does not by itself mean experimental validation; read it with the tier and class. |
| FAIL | The target was not met. Reported, not loosened. |
| REVIEW | A historical target is unmet and the row awaits reassessment. |
| PARTIAL | Only part of the comparison could be performed; the missing part is named. |
| NOT VALIDATED | The comparison was made and the model is outside the target or lacks the physics; a negative finding. |
| NOT COMPARABLE | The reference uses a different assumption, so no target applies. |
| INFO | Stated, not gated (for example absolute values at library moduli). |

The script prints a summary with the number of FAIL, REVIEW and PARTIAL rows
and exits non-zero when any row is FAIL or REVIEW. NOT VALIDATED and PARTIAL
rows do not change the exit code; they remain listed limitations and are
excluded from validated claims. The generated file also lists the pytest marker
groups (`analytical_sanity`, `independent_method`, `full_scale`) with their pass
counts and a reproducibility audit stating, for each benchmark, whether the
inputs come from the cable and material definitions or from a harness.

## Registering a new benchmark

1. **Source.** Record the full reference, the source file and its SHA-256. Check the reported construction, loading, boundary
   conditions and units against the original document.
2. **Inputs.** Write the construction as a cable YAML or a documented harness
   input. Record which inputs are reported, which are assumed and which, if any,
   are identified, and from which data.
3. **Reference data.** Store digitised or tabulated reference data under
   `validation/paper_data/`, with the reading resolution stated in the file header or a companion note.
4. **Target.** Choose and write down the metric and target before running the
   comparison, with its justification (Rule 2).
5. **Gate.** Add a `Row` in `_rows()` of `validation/scripts/run_validation.py` within the
   block for its tier.
6. **Test.** Add a pytest with the appropriate marker. Where a comparison fails,
   pin the failure so that a later relaxation of the target is caught.
7. **Register.** Add the row to [benchmarks.md](benchmarks.md) with source,
   class, quantity, target, result and status, and regenerate
   `validation/validation_report.md`.
8. **Claims.** Any statement in the README, documentation or the companion article must
   cite the registered row and stay within its tier, class and population.
