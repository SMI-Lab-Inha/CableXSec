<!-- SPDX-License-Identifier: Apache-2.0 -->

# CableXSec v0.2.1 release record

This record lists what was built and tested for the v0.2.1 release: the environments, the test
results and the release checks. Case definitions and pass criteria are in
[`VALIDATION.md`](../VALIDATION.md). The SHA-256 hashes of the published assets are those of
the `SHA256SUMS.txt` that the release workflow attaches to the GitHub release.

## Source

| Item | Value |
| --- | --- |
| CableXSec | `0.2.1` |
| Scope | the solver kernel (`cablexsec`) and its post-processing tools, the `cablexsec-batch` command, the tests, examples, validation data and scripts; no desktop application |
| Changes since 0.2.0 | a bonded outer armour face (gap G8), the relative core–core slip option of laid cores, the slip-state normal wire curvature (a correction that changes results), MATLAB-style figures; see [`CHANGELOG.md`](../CHANGELOG.md) |

## Environments of the local gates

Each environment is a new conda environment (conda-forge Python) into which the package and its
`dev`, `fast` and `contact` extras were installed with pip from the package index, as in CI.

| Item | Python 3.11 | Python 3.12 |
| --- | --- | --- |
| Operating system | Windows 11 (x86-64), one thread per process (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMBA_NUM_THREADS` = 1) | same |
| Python | 3.11.18 | 3.12.16 |
| Runtime | NumPy 2.4.6, SciPy 1.17.1, pandas 3.1.1, Pydantic 2.14.4, Matplotlib 3.11.2, PyYAML 6.0.4, rainflow 3.2.0 | NumPy 2.5.5, SciPy 1.18.1, pandas 3.1.1, Pydantic 2.14.4, Matplotlib 3.11.2, PyYAML 6.0.4, rainflow 3.2.0 |
| Optional | numba 0.68.1, Clarabel 0.11.1 | numba 0.68.1, Clarabel 0.11.1 |
| Tools | pytest 9.2.0, pytest-cov 7.1.0, pytest-xdist 3.8.0, Ruff 0.15.22, mypy 2.4.1, build 1.6.1, Twine 7.0.0, setuptools 84.2.1 | the same, and Sphinx 9.0.4, MyST-Parser 5.1.0, sphinx-rtd-theme 3.1.0 |

## Test results

| Gate | Result |
| --- | --- |
| Test suite, Python 3.12 (`python -m pytest -n 6`) | 5023 passed, 3 skipped (kernel-replacement tests that need the kernels as Python; they run in the `NUMBA_DISABLE_JIT=1` pass of the coverage gate) |
| Line and branch coverage of `cablexsec`, Python 3.11 (`release/check_coverage.py`: whole suite, then the kernel tests with `NUMBA_DISABLE_JIT=1`) | 96.60 %; gate 95 % |
| Ruff lint and format check (`src`, `tests`, `examples`, `validation`, `release`, `doc`) | clean on Python 3.11 and 3.12 |
| mypy, strict, package `cablexsec` (the CI configuration, Python 3.11 with NumPy 2.4.6) | no issues in 122 source files |
| Validation report (`validation/scripts/run_validation.py --check`, Python 3.11 and 3.12) | the committed report is current in both environments; 66 rows: 48 pass, 7 fail (BM1-loop, BM1-ID, Z-EI, HU-3PB, WITZ-ID, SV11-PLANE, SV11-GJ; registered targets unchanged), 8 stated without a target (info), 1 partial (BM5), 1 not comparable (HU-uniform), 1 not validated (BM4); taxonomy 1319 `analytical_sanity`, 92 `independent_method`, 59 `full_scale` tests |
| Sphinx documentation (`-W --keep-going`) | built with no warnings |
| Release metadata (`release/check_release_metadata.py`) | passed |
| Release-tree audit (`release/audit_release_tree.py --tree .` and `--archive-ref HEAD`) | passed |
| Python distributions (`python -m build`, `twine check`) | sdist and wheel pass; both carry LICENSE and NOTICE; the wheel holds only `cablexsec` and its metadata. With `SOURCE_DATE_EPOCH` set, the wheel is identical byte for byte between builds; the sdist holds identical files, but its PAX headers carry the build-time mtimes of the directories, so its hash differs between builds |
| Wheel in a new conda environment (Python 3.12), sdist in another (Python 3.11), each with the `fast` extra | in both: `cablexsec` imported from the environment, not the source tree; `release/smoke_test_wheel.py`; `cablexsec-batch --help`; the README quick start, with output identical to the README; all twelve examples, the four tutorial and the ten how-to scripts of the manual from a separate folder that holds their data but no source |

Against 0.2.0, Z-ID now passes with the bonded outer armour face of the Zeidan (2025) input and
Z-EI fails narrowly (`EI_min` 10.09 kN·m² against a measured slip slope of 8.65–9.70 kN·m² in
the plotted curvature); the bonded state is an identified input, not a published one. The other
rows keep their 0.2.0 status.

Under NumPy 2.5.5 (the Python 3.12 environment), mypy reports 40 errors in 14 files, all from
the NumPy 2.5 type stubs (argument-free `ndarray.min`/`max` and `ufunc.reduce` overloads), as
for 0.2.0; CI type-checks against the Python 3.11 wheels.

The review of the release pull request found that a bonded layer at a zero lay angle (no
slider stiffness) was refused by the slip integrators; it is now elastic-only (commit
`fix: a bonded layer at a zero lay angle is elastic-only`). The test suite above is that of the
final tree. The coverage and validation-report gates ran before this fix and before the
quick-start correction of the README; the documentation and distribution gates ran after the
correction and before this fix. Neither change moves a validation row, and CI repeats every gate
on the release pull request.

The CI matrix (`.github/workflows/python.yml`: Ubuntu, Windows and macOS with Python 3.11 and
3.12, a run without numba, coverage, the validation report, distributions and documentation)
runs on the release pull request.
