<!-- SPDX-License-Identifier: Apache-2.0 -->

# Security policy

This policy tells you how to report a security vulnerability in CableXSec.

## Supported versions

Security fixes are released for the latest CableXSec version. Reproduce a report on
that version.

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a vulnerability

Do not disclose a vulnerability in a public issue. Use GitHub's private reporting
form under **Security > Advisories > Report a vulnerability**. If the form is
unavailable, email `jaehoon.seo@inha.ac.kr` with the subject
`CableXSec security report`.

Include the CableXSec version or commit, operating system, Python version, the
smallest reproducer you can share safely, and the observed impact. We acknowledge
reports within 7 days, agree a disclosure schedule with the reporter, and publish a
fixed release and advisory when warranted.

Cable and project files are YAML and JSON that CableXSec parses with safe loaders;
report any input that executes code or reads files outside those it was given.
Modelling limitations and suspect results belong in the public issue tracker unless
they also present a security risk.
