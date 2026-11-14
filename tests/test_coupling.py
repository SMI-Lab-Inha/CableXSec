# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""L6.3 — CableDyn coupling surface (xsec_stiffness).

This is infrastructure a global solver depends on, so:

* **contract** — the public signature and result fields are pinned (a downstream
  solver must not break on a silent change), with a version string;
* **bounds gate** — the curvature-dependent tangent equals EI_max at κ→0
  (full stick), tends to EI_min at large κ (full slip), is monotone non-
  increasing between, and always lies within the bounds. This ties the coupling
  output to the validated stiffness bounds;
* the result carries the slip-front-tangent / local-only / temperature
  limitations (R-HONEST).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.coupling import DOF_ORDER, XsecStiffnessResult, xsec_stiffness

pytestmark = pytest.mark.analytical_sanity


def _cable() -> cx.CableData:
    return cx.load_cable(cx.inter_array_single_core_yaml_path(), geometry_override=None)


def test_coupling_contract_is_versioned_and_stable() -> None:
    """Pin the public signature, version, and result fields a solver depends on."""
    assert cx.COUPLING_API_VERSION == "1.0"

    sig = inspect.signature(xsec_stiffness)
    assert list(sig.parameters)[:2] == ["cable", "kappa_1_m"]
    for kw in ("tension_N", "temperature_C", "mu"):
        assert kw in sig.parameters
        assert sig.parameters[kw].kind == inspect.Parameter.KEYWORD_ONLY

    fields = set(XsecStiffnessResult.__dataclass_fields__)
    assert {
        "kappa_1_m",
        "tension_N",
        "temperature_C",
        "EI_tangent_Nm2",
        "EI_max_Nm2",
        "EI_min_Nm2",
        "EA_N",
        "GJ_Nm2",
        "EA_torsion_coupling",
        "K_6x6",
        "dof_order",
        "api_version",
        "provenance",
    } <= fields

    r = xsec_stiffness(_cable(), 0.01)
    assert r.api_version == "1.0"
    assert r.dof_order == DOF_ORDER
    assert r.K_6x6.shape == (6, 6)


def test_tangent_bounds_gate() -> None:
    """κ→0 → EI_max; large κ → EI_min; monotone non-increasing; within bounds."""
    cable = _cable()
    r0 = xsec_stiffness(cable, 0.0)
    assert r0.EI_tangent_Nm2 == pytest.approx(r0.EI_max_Nm2, rel=1e-6)

    r_hi = xsec_stiffness(cable, 0.5)
    assert r_hi.EI_tangent_Nm2 == pytest.approx(r_hi.EI_min_Nm2, rel=1e-6)

    kappas = np.r_[0.0, np.geomspace(1.0e-4, 0.5, 80)]
    tangents = [xsec_stiffness(cable, k).EI_tangent_Nm2 for k in kappas]
    # monotone non-increasing (stiffness softens as more stations slip).
    assert all(a >= b - 1e-6 for a, b in zip(tangents, tangents[1:], strict=False))
    # always within [EI_min, EI_max].
    assert all(r0.EI_min_Nm2 - 1e-6 <= t <= r0.EI_max_Nm2 + 1e-6 for t in tangents)
    # a strictly-intermediate tangent exists in the knee (real transition).
    assert any(r0.EI_min_Nm2 * 1.01 < t < r0.EI_max_Nm2 * 0.99 for t in tangents)


def test_6x6_models_what_it_can_and_nans_the_rest() -> None:
    """Modelled DOFs are finite; shear + bending couplings are NaN, not faked."""
    r = xsec_stiffness(_cable(), 0.01)
    k = r.K_6x6
    # modelled: axial, torsion, axial-torsion coupling, both bending DOFs.
    assert np.isfinite(k[0, 0]) and k[0, 0] == pytest.approx(r.EA_N)
    assert np.isfinite(k[3, 3]) and k[3, 3] == pytest.approx(r.GJ_Nm2)
    assert k[0, 3] == pytest.approx(r.EA_torsion_coupling)
    assert k[4, 4] == pytest.approx(r.EI_tangent_Nm2)
    assert k[5, 5] == pytest.approx(r.EI_tangent_Nm2)
    # not modelled (Euler-Bernoulli): shear DOFs are NaN, not a fabricated 0.
    assert np.isnan(k[1, 1]) and np.isnan(k[2, 2])
    assert np.isnan(k[4, 0])  # bending<->axial coupling not modelled


def test_provenance_states_tangent_and_local_scope() -> None:
    r = xsec_stiffness(_cable(), 0.01, temperature_C=50.0)
    p = r.provenance
    assert p is not None
    blob = " ".join(p.limitations).lower()
    assert "local" in blob and "global solver owns" in blob
    assert "temperature" in blob  # temperature recorded-but-not-wired caveat
    assert r.temperature_C == 50.0
