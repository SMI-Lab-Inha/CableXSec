<!-- SPDX-License-Identifier: Apache-2.0 -->

# CableXSec

[![Release](https://img.shields.io/github/v/release/SMI-Lab-Inha/CableXSec?sort=semver)](https://github.com/SMI-Lab-Inha/CableXSec/releases/latest)
[![Python CI](https://github.com/SMI-Lab-Inha/CableXSec/actions/workflows/python.yml/badge.svg)](https://github.com/SMI-Lab-Inha/CableXSec/actions/workflows/python.yml)
[![Documentation](https://img.shields.io/readthedocs/cablexsec/latest?label=docs)](https://cablexsec.readthedocs.io/en/latest/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

CableXSec is an open-source Python library for the local (cross-section) mechanics of dynamic
power cables. It resolves what happens inside the cable at every point of a global analysis:
the helical armour wires sticking and slipping against their neighbours, the hysteretic
moment–curvature response that follows, the stress of every wire and component at fixed
material points, the contact pressures that set the friction, the temperatures, and the fatigue
damage of the armour, the conductor and the metallic sheaths.

It reads the results of global analyses (CableDyn, RIFLEX, and time-history CSV exports such as
those of OrcaFlex) and turns them into local stresses and damage, and it exports a hysteretic
bending law for use in global analysis. The
current release is `v0.1.0`; interfaces are versioned and may change before `1.0.0`.

## What CableXSec does

- **Cable definitions with sources.** YAML definitions of helical wire layers, rectangular
  tendons, stranded conductors, laid three-core sections, polymer layers, metallic sheaths and
  tapes, checked against a schema, with a cited source for every material property. Published
  cables are bundled: Ménard and Cartraud (2023), IEA Wind Task 49 66 kV, CIGRE TB 880, Witz
  (1996), Sævik (2011), Thies et al. (2016) and others.
- **Section properties and contact state.** Axial, torsional and bending stiffness with their
  full-slip and no-slip bounds; the radial contact chain through every layer under winding
  clamping, tension, pressure and temperature, for sealed and flooded annuli.
- **Hysteretic bending.** Moment–curvature loops of helical armour with the neutral-axis slip
  front, Coulomb saturation and shear-compliant stick, laid-core friction of three-core cables,
  and the stored and dissipated work of every history.
- **Stresses at fixed material points.** Axial, local bending and frictional stress of every
  wire around the circumference, transverse slip on the geodesic, conductor strand slip, and
  screen, sheath and tape strains.
- **Contact and fretting.** Hertz contact with subsurface stress tensors, Mindlin partial slip,
  fretting regime maps, wear and SWT critical-plane recovery.
- **Thermal response.** An IEC 60287 thermal ladder for the conductor temperature (for
  three-core cables without the geometric factor and the sheath eddy loss, which makes it
  conservative), a transient thermal network,
  frictional heating, and the thermal change of the armour clamping.
- **Fatigue and limit states.** Rainflow counting and Miner damage per wire and fibre; long-term
  damage over a scatter of sea states; termination-adjusted fatigue; armour compression,
  coiling, sheave and capacity checks of time series.
- **Traceable results.** Every result carries the method, citations, verification gates and
  limitations of its calculation; projects save inputs, results and environment together.
  Unsupported inputs stop with a named error rather than running approximately.

## Evidence

Every capability is tested (about 4,000 tests), and the comparisons with published data are
recorded with their targets, data and scripts in the
[validation record](VALIDATION.md). A selection:

| Case | Reference | Result |
|---|---|---|
| Full-slip bending stiffness, Ménard cable | Ménard and Cartraud (2023), Table 3 | 0.6 % |
| Full-slip bending stiffness, 2.5-inch flexible pipe | Witz (1996), measured gradients | within 0.7 % of the lower gradient |
| Lateral buckling capacity, three pipes | Sævik and Thorsen (2017), Table 3 | within 2.1 % |
| Mindlin transition amplitude | Fouvry et al. (1995), Table 4 | within 0.07 µm |
| Torque constant of a strand | Feyrer, Example 2.8 | 0.11 % |
| Moment–curvature loop energy closure | closed form | 0.08 % |

Comparisons that do not meet their registered targets — the Witz 300-bar loop, the Hu et
al. (2022) conductor loop and the Sævik (2011) near-termination stresses — are reported as
failed in the same record, with their tolerances unchanged.

## Quick start

Install the release wheel from the
[latest release](https://github.com/SMI-Lab-Inha/CableXSec/releases/latest) (Python 3.11 or
3.12), or install straight from the repository:

```powershell
python -m pip install "cablexsec[fast] @ git+https://github.com/SMI-Lab-Inha/CableXSec.git@v0.1.0"
```

The `fast` extra adds `numba` for the compiled slip-stress and rainflow kernels; without it the
results agree to round-off and are computed more slowly. Then:

```python
import numpy as np
import cablexsec as cx

cable = cx.load_cable(cx.menard2023_yaml_path())
stiffness = cx.compute_stiffness(cable)
print(f"EA = {stiffness.ea_fixed_N / 1e6:.0f} MN, "
      f"EI full slip = {stiffness.ei_min_Nm2:.0f} N m^2, "
      f"EI no slip = {stiffness.ei_max_Nm2 / 1e3:.1f} kN m^2")

curvature = 0.1 * np.sin(np.linspace(0.0, 6.0 * np.pi, 1801))  # three cycles, 1/m
loop = cx.compute_mkappa(cable, curvature, tension_N=0.0, contact_model="bounded")
print(f"peak moment {loop.moment_Nm.max():.0f} N m, "
      f"friction work {loop.W_diss_J_per_m:.1f} J/m over the history")

kx = 0.02 * np.sin(np.linspace(0.0, 200.0 * np.pi, 4001))
fatigue = cx.compute_fatigue(
    cable, kx, np.zeros_like(kx), tension_N=np.full_like(kx, 1.0e5)
)
for layer, damage in fatigue.D_total_per_wire.items():
    print(f"{layer}: largest Miner damage {damage.max():.3e} over the record")
```

```text
EA = 280 MN, EI full slip = 3571 N m^2, EI no slip = 189.8 kN m^2
peak moment 541 N m, friction work 218.0 J/m over the history
inner_armour: largest Miner damage 1.774e-06 over the record
outer_armour: largest Miner damage 3.005e-07 over the record
```

For a set of CableDyn sea-state runs, `cablexsec-batch batch.json --output-directory results`
computes the weighted annual armour damage at the material sections you name. Every shipped
script is listed in [`examples/`](examples/README.md).

## Documentation

The manual at <https://cablexsec.readthedocs.io> covers installation, the cable definition, the
local analysis and global-to-local guides, the ultimate-limit-state and rig-test guides, the
benchmark register, the validation policy and the Python API.

## Development

```powershell
git clone https://github.com/SMI-Lab-Inha/CableXSec.git
cd CableXSec
conda env create -f environment.yml
conda activate cablexsec
python -m pip install -e ".[dev,fast,contact]"
python -m pytest -q
```

See [DEVELOPMENT.md](DEVELOPMENT.md) for the test groups, type checks, the validation report and
the documentation build, and [ARCHITECTURE.md](ARCHITECTURE.md) for the package layout.

## Validation

CableXSec is verified against closed-form solutions and independent numerical methods, and
compared with published measurements on flexible pipes and power cables. Results, acceptance
criteria and reproduction commands are in [VALIDATION.md](VALIDATION.md); data and scripts are
in [`validation/`](validation/README.md), including the data of the companion journal article.

## Repository layout

| Directory | Contents |
|---|---|
| `src/cablexsec/` | Python package: cable definitions, mechanics, analyses, readers, plotting |
| `tests/` | pytest unit, integration and benchmark tests |
| `doc/` | Sphinx manual |
| `examples/` | Runnable example scripts |
| `release/` | Release checks |
| `validation/` | Validation report, reference data and reproduction scripts |

## Citation

If you use CableXSec in academic work, please cite the software release (CableXSec 0.1.0);
[CITATION.cff](CITATION.cff) contains the entry (GitHub **Cite this repository**) and the
manual's [How to cite](https://cablexsec.readthedocs.io/en/latest/citing.html) page gives
BibTeX. The journal article that describes the method is under review; its reference will be
added when it is published. Global-to-local studies with CableDyn should also cite CableDyn
([Ocean Engineering 368, 128332, 2026](https://doi.org/10.1016/j.oceaneng.2026.128332)).

## Contributing and support

Questions, bug reports, and feature requests are welcome in
[GitHub Issues](https://github.com/SMI-Lab-Inha/CableXSec/issues); see [SUPPORT.md](SUPPORT.md)
for where each belongs. Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.
Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md). Report
security issues privately as described in [SECURITY.md](SECURITY.md).

If you extend or fix CableXSec in your own copy, please consider contributing the change here,
so that every user benefits and the validation record stays in one place.

CableXSec is research software. It is not qualified for design decisions on its own; users are
responsible for model selection, input data, and compliance with the design codes that govern
their work.

Maintainer: Prof. Jae Hoon Seo, Department of Naval Architecture and Ocean Engineering, Inha
University, Republic of Korea, <jaehoon.seo@inha.ac.kr>

## Acknowledgements

CableXSec builds on published work on helical structures, flexible pipes and power cables, cited
in the code, the cable definitions and the manual. We are grateful to the authors whose
published data make the comparisons in [VALIDATION.md](VALIDATION.md) possible.

## Licence and name

CableXSec is distributed under the Apache License 2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).

"CableXSec" identifies the official project in this repository. Modified versions are welcome
under the licence, but, as Section 6 of the Apache License 2.0 provides, they may not use the
name in a way that implies they are the official CableXSec or endorsed by its maintainers;
describing a work as "based on CableXSec" is fine.
