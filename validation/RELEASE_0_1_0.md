<!-- SPDX-License-Identifier: Apache-2.0 -->

# CableXSec v0.1.0 release record

This record lists what was built and tested for the v0.1.0 release: the environment, the test
results and the release checks. Case definitions and pass criteria are in
[`VALIDATION.md`](../VALIDATION.md). The SHA-256 hashes of the published assets are those of
the `SHA256SUMS.txt` that the release workflow attaches to the GitHub release.

## Source

| Item | Value |
| --- | --- |
| CableXSec | `0.1.0` |
| Scope | the solver kernel (`cablexsec`) and its post-processing tools, the `cablexsec-batch` command, the tests, examples, validation data and scripts; no desktop application |

## Environment of the local gates

| Item | Version |
| --- | --- |
| Operating system | Windows 11 (x86-64), one thread per process (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMBA_NUM_THREADS` = 1) |
| Python | 3.11.15 (conda-forge) |
| Runtime | NumPy 2.4.3, SciPy 1.17.1, pandas 3.0.2, Pydantic 2.13.3, Matplotlib 3.10.9, PyYAML 6.0.3, rainflow 3.2.0 |
| Optional | numba 0.65.1, Clarabel 0.11.1 |
| Tools | pytest 9.0.3, pytest-cov 7.1.0, Ruff 0.15.11, mypy 2.1.0, Sphinx 9.0.4, MyST-Parser 5.1.0, sphinx-rtd-theme 3.1.0, build 1.5.0, Twine 6.2.0, setuptools 82.0.1 |

## Test results

| Gate | Result |
| --- | --- |
| Test suite (`python -m pytest`, serial) | 3986 passed, 1 expected failure (xfail), 0 skipped |
| Line and branch coverage of `cablexsec` (`release/check_coverage.py`: whole suite, then the kernel tests with `NUMBA_DISABLE_JIT=1`) | 97.59 %; CI gate 95 % |
| Ruff lint and format check (`cablexsec`, `tests`, `examples`, `validation`, `release`) | clean |
| mypy, strict, package `cablexsec` | no issues in 117 source files |
| Validation report (`validation/scripts/run_validation.py`) | 47 rows: 38 pass, 6 fail (BM1-loop, BM1-loop-uniform, HU-3PB, WITZ-ID, SV11-PLANE, SV11-GJ; registered targets unchanged), 1 partial (BM5), 1 not comparable (HU-uniform), 1 not validated (BM4); taxonomy 939 `analytical_sanity`, 68 `independent_method`, 26 `full_scale` tests |
| Examples (`examples/*.py`, from a separate folder) | all twelve run to completion |
| Python distributions (`python -m build`, `twine check`) | sdist and wheel pass; both carry LICENSE and NOTICE; the wheel holds only `cablexsec` and its metadata |
| Wheel in a new virtual environment (`release/smoke_test_wheel.py`, quick start of the README, examples) | passed; `cablexsec` imported from the environment, not the source tree |
| Sphinx documentation (`-W --keep-going`) | built with no warnings |
| Release metadata (`release/check_release_metadata.py`) | passed |
| Release-tree audit (`release/audit_release_tree.py --tree .`) | passed on the staged tree |

The CI matrix (`.github/workflows/python.yml`: Ubuntu, Windows and macOS with Python 3.11 and
3.12, a run without numba, coverage, distributions and documentation) runs on the first push of
the public repository.
