---
name: Bug report
about: Report incorrect results, a crash, or a rejected valid input
title: "Bug: <short description>"
labels: bug
---

<!-- SPDX-License-Identifier: Apache-2.0 -->

## What happened

<!-- The error message or the wrong result, and what you expected instead. -->

## How to reproduce

<!-- Attach the smallest input that shows the problem (cable YAML, load history, Python
     snippet) and every file it reads. Do not attach cable data that you are not
     permitted to publish. -->

```python
import cablexsec as cx
```

## Environment

- CableXSec version (`python -c "import cablexsec; print(cablexsec.__version__)"`):
- Installed from: release wheel, source checkout, or `pip install git+...`
- Optional extras installed (`fast`, `contact`):
- OS:
- Python version:

## Checks

- [ ] The problem occurs with the latest release.
- [ ] If results disagree with another code or a test, I have described both models and the
      quantity compared (see [VALIDATION.md](https://github.com/SMI-Lab-Inha/CableXSec/blob/main/VALIDATION.md)).
