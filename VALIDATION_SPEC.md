<!-- SPDX-License-Identifier: Apache-2.0 -->

# CableXSec validation specification

This file maps the evidence tiers used in [VALIDATION.md](VALIDATION.md) to the code that
produces them and states the release rule. The generator
`validation/scripts/run_validation.py` and the tests are the authority for the numerical
limits; the targets were registered before the comparisons were run and are not relaxed
afterwards (see the validation policy in the manual, `doc/validation_policy.md`).

## Tiers

| Tier | Evidence | Produced by | Acceptance |
| --- | --- | --- | --- |
| 1 | Closed-form solutions, conservation laws and identities (energy closure, equilibrium, Lamé, Hertz, Mindlin, Out–von Morgen, rotation equivariance) | Tier-1 rows of `validation/validation_report.md`; tests marked `analytical_sanity` | Each row's stated tolerance, typically below `1e-9` relative or at round-off |
| 1 | Comparisons with independent numerical methods (periodic friction and shear cells, finite-difference radial ring, independent Hertz reference, NumPy reference of the compiled kernels) | tests marked `independent_method`; scripts in `validation/scripts/` | Each test's stated tolerance |
| 2 | Comparisons with published measurements and published tabulated results (Ménard and Cartraud 2023, Witz 1996, Sævik 2011, Gjøsteen et al. 2027, Hu et al. 2022, Sævik and Thorsen 2017, Fouvry et al. 1995, Li and Vaz, Østergaard, Yang et al., Kim and Lee 2017, Utting and Jones 1987, CIGRE TB 880) | Tier-2 rows of `validation/validation_report.md`; tests marked `full_scale`; `doc/benchmarks.md` | The registered target of each row |

A row's status is one of `PASS` (meets the registered target), `FAIL` (does not; the target
stays), `PARTIAL` (a stated part of the comparison is not yet possible), `NOT COMPARABLE`
(the reference does not support the comparison as defined) and `NOT VALIDATED` (a negative
finding that limits the model's scope). Only `PASS` rows support a validation statement, and
only for the quantity the row names.

## Running the evidence

```bash
python -m pytest -q -m analytical_sanity
python -m pytest -q -m independent_method
python -m pytest -q -m full_scale
python validation/scripts/run_validation.py --check
```

## Release rule

A release candidate must pass the whole test suite on every platform of the CI matrix, and
`validation/validation_report.md` must be current for the release commit
(`run_validation.py --check` reports no stale artefact). The report's own exit status
reflects the registered `FAIL` rows; a row may change from `FAIL` to `PASS` only through a
model change that is documented in the changelog, never through a changed target.
