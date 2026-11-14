<!-- SPDX-License-Identifier: Apache-2.0 -->

# CableXSec release procedure

This procedure is for maintainers. It turns one reviewed commit into the Python
assets of a CableXSec release. Run it from a clean checkout of the release commit.
Tags have the form `vMAJOR.MINOR.PATCH`; CableXSec follows semantic versioning and
stays below `1.0.0` while its interfaces may change. Development details are in
[DEVELOPMENT.md](DEVELOPMENT.md).

Work for the next release collects on `dev`. To release, merge `dev` into `main` and
run this procedure on the resulting `main` commit.

## 1. Set and check the release metadata

Update the version and date in `CHANGELOG.md`, `pyproject.toml`,
`cablexsec/__init__.py` and `CITATION.cff`, and add the release record
`validation/RELEASE_MAJOR_MINOR_PATCH.md`. Move every release note from
`[Unreleased]` to the new dated heading. Then run:

```powershell
python release/check_release_metadata.py
python release/audit_release_tree.py --tree .
pre-commit run --all-files
git diff --check
```

The audit rejects local-only directories, desktop-application files, generated
products, credentials, machine-specific paths, e-mail addresses other than the
maintainer's, and names that point at withheld material in the tracked files. After
creating the release commit, audit the Git archive as well:

```powershell
python release/audit_release_tree.py --archive-ref HEAD
```

## 2. Test the source tree

In a fresh environment with Python 3.11 and again with Python 3.12:

```powershell
python -m pip install -e ".[dev,fast,contact]"
python -m pytest -q --cov --cov-branch --cov-report=term-missing
python -m ruff check cablexsec tests examples validation release
python -m ruff format --check cablexsec tests examples validation release
python -m mypy
python validation/scripts/run_validation.py --check
```

Every test must pass. `run_validation.py --check` must report the committed
`validation/validation_report.md` as current; its exit status reflects the registered
failed rows and is recorded, not hidden. Confirm that
[VALIDATION.md](VALIDATION.md) matches the report at the release commit.

## 3. Build and check the distributions

```powershell
$env:SOURCE_DATE_EPOCH = git log -1 --format=%ct
python -m build --outdir dist
python -m twine check dist/*
python -m venv wheel-test
wheel-test\Scripts\python -m pip install dist\cablexsec-MAJOR.MINOR.PATCH-py3-none-any.whl
wheel-test\Scripts\python release\smoke_test_wheel.py --version MAJOR.MINOR.PATCH
```

Both distributions must contain the Apache-2.0 licence and the NOTICE; the wheel must
contain only the `cablexsec` package and its metadata. Run the examples of
[`examples/`](examples/README.md) against the installed wheel.

The release workflow (`.github/workflows/release.yml`) produces exactly these assets:

- `cablexsec-MAJOR.MINOR.PATCH-py3-none-any.whl`
- `cablexsec-MAJOR.MINOR.PATCH.tar.gz`
- `SHA256SUMS.txt`

Run the workflow manually (`workflow_dispatch`) as a preflight and keep its URL with
the release record.

## 4. Tag and publish

Tag only the commit that passed every check:

```powershell
$ReleaseVersion = "0.1.0"
git tag -s "v$ReleaseVersion" -m "CableXSec v$ReleaseVersion"
git push origin "v$ReleaseVersion"
```

If no signing key is available, use an annotated tag and state this in the release
notes. Create a draft GitHub release from the tag. Publishing the release triggers the
release workflow, which builds the distributions from the tag, checks that they carry
the tag's version, and attaches them with `SHA256SUMS.txt`. Compare the checksums with
the preflight artifact; if assets do not match, withdraw the release and investigate.

## 5. Close the release

Confirm that the release page lists the three assets, the documentation builds from
the tag, the GitHub source archives contain no credentials or run output, and a fresh
clone passes the test suite. Then add an empty `[Unreleased]` section to the
changelog.

## Publishing checklist for a new repository

Before the first release is published from a new GitHub repository:

- [ ] **About**: set the description and topics below, the website to
      <https://cablexsec.readthedocs.io>, and enable Releases.
- [ ] **Features**: enable Issues (the templates in `.github/ISSUE_TEMPLATE` disable
      blank issues); disable the Wiki and Projects unless they will be maintained.
- [ ] **Security**: enable private vulnerability reporting (used by
      [SECURITY.md](SECURITY.md)) and secret scanning.
- [ ] **Actions**: allow GitHub-hosted runners; the workflows need no secrets beyond
      the automatic `GITHUB_TOKEN`, and only the release job writes
      (`contents: write`).
- [ ] **Branch protection** on `main` and `dev`: require pull requests, the `python`
      workflow checks, and linear history.
- [ ] **Read the Docs**: import the repository as project `cablexsec`, build `latest`
      and the release tag, and check the badge in the README.
- [ ] **Citation**: confirm that **Cite this repository** renders from
      `CITATION.cff`.

Suggested description:

> Component-resolved local cross-section mechanics of dynamic power cables:
> stiffness, hysteretic bending with armour slip, wire and component stresses,
> contact, thermal response and fatigue at fixed material points, with readers for
> CableDyn, RIFLEX and CSV time-history results.

Suggested topics: `dynamic-power-cable`, `submarine-cable`, `cross-section-analysis`,
`helical-armour`, `fatigue`, `fretting`, `floating-offshore-wind`, `offshore-wind`,
`offshore-engineering`, `marine-engineering`, `structural-mechanics`, `python`,
`cabledyn`.
