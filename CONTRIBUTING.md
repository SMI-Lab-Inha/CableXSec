<!-- SPDX-License-Identifier: Apache-2.0 -->

# Contributing to CableXSec

Contributions are welcome: bug reports, documentation fixes, benchmark cases, and
code. This guide sets out how to contribute and the standards a change must meet.
Environment, test, and tooling commands are in
[DEVELOPMENT.md](https://github.com/SMI-Lab-Inha/CableXSec/blob/main/DEVELOPMENT.md);
the package design is in
[ARCHITECTURE.md](https://github.com/SMI-Lab-Inha/CableXSec/blob/main/ARCHITECTURE.md).
Everyone taking part follows the
[Code of Conduct](https://github.com/SMI-Lab-Inha/CableXSec/blob/main/CODE_OF_CONDUCT.md).

## Contributing upstream

If you have extended or corrected CableXSec in your own copy, please consider
offering the change here. A change merged upstream is tested on every platform in
CI, carried into the validation record, and maintained with the rest of the code,
and every user benefits from it. Open an issue first for a larger change so that
the design can be agreed before you invest in it.

## Issues

- **Bugs**: use the bug-report template. Include the CableXSec version, platform,
  Python version, a minimal cable YAML and script that reproduce the problem, and the
  full error output.
- **Enhancements**: use the feature-request template and describe the physical
  problem the feature solves.
- **Questions**: use the question template; see
  [SUPPORT.md](https://github.com/SMI-Lab-Inha/CableXSec/blob/main/SUPPORT.md).
- **Security issues**: follow
  [SECURITY.md](https://github.com/SMI-Lab-Inha/CableXSec/blob/main/SECURITY.md);
  do not open a public issue.

## Development setup

```powershell
git clone https://github.com/SMI-Lab-Inha/CableXSec.git
cd CableXSec
conda env create -f environment.yml
conda activate cablexsec
python -m pip install -e ".[dev,fast,contact]"
pre-commit install
python -m pytest -q
```

The conda environment provides Python and the developer tools on Linux, macOS, and
Windows. Coverage, type checking, the validation report, and the documentation build
are described in
[DEVELOPMENT.md](https://github.com/SMI-Lab-Inha/CableXSec/blob/main/DEVELOPMENT.md).

## Pull requests

`main` holds the released code; new work collects on `dev` until the next release.

1. Fork the repository and branch from `dev` with a descriptive name
   (`fix/<topic>`, `feat/<topic>`, `docs/<topic>`, `test/<topic>`, `perf/<topic>`,
   `ci/<topic>`).
2. Keep each pull request to one change. Add or update tests for it.
3. Run `pre-commit run --all-files` and the test suite (see
   [DEVELOPMENT.md](https://github.com/SMI-Lab-Inha/CableXSec/blob/main/DEVELOPMENT.md)).
4. Open the pull request against `dev` with a
   [Conventional Commits](https://www.conventionalcommits.org) title and complete
   the template.
5. CI must pass and a maintainer must approve before merge. Pull requests are
   squash-merged.

Commit messages follow Conventional Commits: a subject of at most 72 characters,
and a body that explains why the change is needed.

```text
feat: add rectangular-tendon lateral relief to the transverse cell
fix: correct radial-face indexing for a partial helical-layer selection
test: add closed-form check of the Coulomb saturation stress
```

Do not bypass hooks with `--no-verify`. Commits must be attributed to the people
who wrote them.

## Standards

- **Tests**: every behaviour change carries a pytest case. Numerical changes compare
  against a closed form, an independent reference, a published value, or a conserved
  quantity; do not replace such tests with snapshots of CableXSec output.
- **Validation**: a benchmark target is registered before the model is run against it
  and is not relaxed afterwards (see the validation policy in the manual). If a change
  alters a validation result, tolerance, or reference, regenerate
  `validation/validation_report.md` and update
  [VALIDATION.md](https://github.com/SMI-Lab-Inha/CableXSec/blob/main/VALIDATION.md)
  in the same pull request.
- **Sources**: every material constant, curve, and benchmark value states its source
  (publication, table, equation) next to the value.
- **Documentation**: document only implemented and tested behaviour. Update the manual
  in `doc/` and add a
  [CHANGELOG.md](https://github.com/SMI-Lab-Inha/CableXSec/blob/main/CHANGELOG.md)
  entry under `[Unreleased]` for user-visible changes (add that heading above the
  latest release if it is not there yet).
- **Errors**: unsupported inputs stop with a specific error message; never return an
  approximate result silently.
- **Python**: `ruff` for linting and formatting, `mypy --strict` for types, `pytest` for
  tests; line and branch coverage of the package stays at or above the threshold set in
  CI.
- **Other codes**: describe differences from Helica, UFLEX, BFLEX, or any other
  code neutrally and with the evidence, and cite them as listed in the manual.

## Repository contents

- Manual pages go in `doc/`.
- `examples/*.py` are runnable scripts; auxiliary inputs go in `examples/<topic>/`.
- Benchmark data and reproduction scripts go in `validation/`; release checks in
  `release/`.
- Do not commit build trees, run output, editor settings, copyrighted papers, licensed
  standards, manuscript sources, simulation files of commercial codes, or cable data
  that you are not permitted to publish.

## Licence and sign-off

CableXSec does not use a contributor licence agreement. As Section 5 of the
[Apache License 2.0](https://github.com/SMI-Lab-Inha/CableXSec/blob/main/LICENSE)
provides, a contribution you submit is licensed under the same licence. Sign off
each commit (`git commit -s`) to certify the
[Developer Certificate of Origin 1.1](https://developercertificate.org/): that you
wrote the change or otherwise have the right to submit it under this licence. New
source files carry the SPDX header given in
[DEVELOPMENT.md](https://github.com/SMI-Lab-Inha/CableXSec/blob/main/DEVELOPMENT.md#spdx-headers).
