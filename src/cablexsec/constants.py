# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Numerical and physical constants with literature provenance.

No physics module may write a bare numeric constant in a function body; every
such constant lives here with a ``# Source:`` citation (one place to audit).
Material- and geometry-specific values belong in the material library and the
cable YAML, not here.
"""

from __future__ import annotations

# Ratio of the maximum sub-surface shear stress to the peak Hertz pressure for
# a circular (or spherical) elastic contact at Poisson ratio nu = 0.3.
# Source: [Johnson1985] §4.2, eqs (4.43)-(4.44) and Table 4.1 (b/a = 1.0 row).
# The maximum over all ellipse aspect ratios is ~0.325 (at b/a ~ 0.4).
TAU_MAX_OVER_P0_NU03: float = 0.31

# Standard gravitational acceleration (hydrostatic pressure helpers).
# Source: CODATA / ISO 80000-3 standard value.
G_ACCEL_M_S2: float = 9.80665
