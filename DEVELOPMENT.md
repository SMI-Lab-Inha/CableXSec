<!-- SPDX-License-Identifier: Apache-2.0 -->

# Developing CableXSec

This guide is for developers who work on CableXSec from source. It lists the
commands for the environment, tests, type and lint checks, the validation report,
the documentation, and the distributions. Contribution rules are in
[CONTRIBUTING.md](https://github.com/SMI-Lab-Inha/CableXSec/blob/main/CONTRIBUTING.md);
the release procedure is in
[RELEASE.md](https://github.com/SMI-Lab-Inha/CableXSec/blob/main/RELEASE.md).

## Environment

Requirements: Python 3.11 or 3.12 on Linux, macOS, or Windows, and git. The conda
environment in `environment.yml` provides Python and the developer tools
(`pre-commit`, `build`, `twine`); a plain virtual environment works as well.

```powershell
conda env create -f environment.yml
conda activate cablexsec
python -m pip install -e ".[dev,fast,contact]"
pre-commit install
```

The extras are optional at run time: `fast` adds `numba` for the compiled slip-stress
and rainflow kernels, `contact` adds `clarabel`, an independent solver used only by the
cross-check script `validation/scripts/concentric_conductor_bend.py` and its test. Setting `CABLEXSEC_DISABLE_NUMBA=1` selects the NumPy reference path even
when `numba` is installed.

## Tests

```powershell
python -m pytest -q                                   # whole suite
python -m pytest -q -m analytical_sanity              # closed-form and machine-precision gates
python -m pytest -q -m independent_method             # comparisons with independent methods
python -m pytest -q -m full_scale                     # published full-scale data
python -m pytest -q --cov --cov-branch --cov-report=term-missing
```

The whole suite takes about 20 minutes on one core. Do not run it with several
workers on a shared machine without limiting the threads of each worker; set
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `NUMBA_NUM_THREADS`
to `1`. The tests import the reproduction scripts of `validation/scripts/` by name
(`tests/conftest.py` puts that folder on the import path).

| Marker | Tests |
|---|---|
| `analytical_sanity` | Closed-form solutions, conservation laws and machine-precision identities |
| `independent_method` | Comparisons with an independent computational method |
| `full_scale` | Published full-scale or test-rig data |

Numerical tests compare against a closed form, an independent reference or a published
value. Do not replace such a test with a snapshot of CableXSec output.

## Validation report

```powershell
python validation/scripts/run_validation.py          # regenerate validation/validation_report.md
python validation/scripts/run_validation.py --check  # fail if the committed report is stale
```

The script recomputes every row from the live code, runs the three marker groups for
the test taxonomy, and exits non-zero while any row is `FAIL` or `REVIEW`; the failed
rows of the 0.1.0 report are registered, unresolved discrepancies (see
[VALIDATION.md](https://github.com/SMI-Lab-Inha/CableXSec/blob/main/VALIDATION.md)).

## Formatting, lint and types

```powershell
pre-commit run --all-files
python -m ruff check src tests examples validation release doc
python -m ruff format --check src tests examples validation release doc
python -m mypy
```

`mypy` is configured in `pyproject.toml` (`strict = true`, package `cablexsec` under
`src/`).

## Documentation

```powershell
python -m pip install -r doc/requirements.txt
python -m sphinx -W --keep-going -b html doc doc/_build/html
```

Markdown files outside `doc/` that the manual includes (`CHANGELOG.md`,
`CONTRIBUTING.md`, `DEVELOPMENT.md`, `VALIDATION.md`) must link to repository files
with absolute `https://github.com/SMI-Lab-Inha/CableXSec/blob/main/...` URLs.

## Distributions

```powershell
python -m build --outdir dist
python -m twine check dist/*
python -m venv wheel-test
wheel-test\Scripts\python -m pip install dist\cablexsec-0.1.0-py3-none-any.whl
wheel-test\Scripts\python release\smoke_test_wheel.py --version 0.1.0
```

The wheel contains only the `cablexsec` package (with its bundled cable definitions)
and its metadata; the sdist adds the tests, examples, validation data and documents.

## Coding standards

- Python 3.11 syntax, type annotations on every public function, NumPy arrays typed
  with `numpy.typing.NDArray`.
- SI units throughout, with the unit in the name (`_m`, `_Pa`, `_N`, `_1_m`, `_deg`).
- Every material constant, curve and benchmark value carries its source next to the
  value; a value that is assumed says so.
- Unsupported inputs raise a subclass of `CableXSecError` with a specific message.
- Results carry a `provenance` record (method, citations, gates, limitations); add to
  it when a calculation adds an assumption.

### SPDX headers

Every source file starts with:

```text
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
```

Use the comment syntax of the language (`<!-- ... -->` in Markdown, `..` in
reStructuredText).

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: cablexsec` in tests | package not installed | `python -m pip install -e ".[dev]"` |
| `import cablexsec` resolves to another checkout | an older editable install | `python -c "import cablexsec; print(cablexsec.__file__)"`, then reinstall |
| Tests slow or the machine unresponsive | BLAS or `numba` threads per process | set the thread variables above to `1` |
| Windows checkout differs from CI | CRLF line endings | `.gitattributes` enforces LF; run `git add --renormalize .` |
