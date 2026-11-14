# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""CableDyn coupling surface: the cross-section tangent stiffness.

The interface a **global Cosserat-rod / dynamic solver** (CableDyn, RIFLEX,
…) calls to get the cross-section's *curvature-dependent* tangent
bending stiffness at an operating point. This is the surface that makes
CableXSec infrastructure other solvers build on, so it is **versioned**
(:data:`COUPLING_API_VERSION`) and **contract-tested** — a downstream solver can
depend on the signature not changing silently.

What it returns — the **operating-point tangent**, not the bounds
---------------------------------------------------------------
A global solver integrating the rod needs ``dM/dκ`` at the *current* curvature,
which is the **local slope of the Papailiou slip-front M-κ backbone** — not the
``EI_min``/``EI_max`` bounds. :func:`xsec_stiffness` differentiates the implemented
slip-front moment along a monotonic loading path:

- at ``κ → 0`` the tangent → ``EI_max`` (all helix stations stick),
- at large ``κ`` the tangent → ``EI_min`` (all stations slip),
- monotonically softening between (more stations slip as ``κ`` grows).

That bounds behaviour ties the coupling output to the verified stiffness bounds
and is enforced by the L6.3 bounds gate.

Honest scope (R-HONEST): the result states it is the slip-front local tangent;
the global solver owns the dynamics. Shear DOFs and the bending↔axial/torsion
off-diagonal couplings are **not modelled** (Euler-Bernoulli section) and are
returned as ``NaN`` in the 6×6, never as a fabricated zero-stiffness.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cablexsec.cables.schema import CableData
from cablexsec.exceptions import PhysicalRangeError
from cablexsec.mechanics.core_friction import CoreFrictionSpec
from cablexsec.provenance import Provenance

FloatArray = NDArray[np.float64]

# Bump on any breaking change to xsec_stiffness's signature or XsecStiffnessResult
# fields. Pinned by tests/test_coupling.py::test_coupling_contract_is_versioned.
COUPLING_API_VERSION = "1.0"

# 6×6 generalised-strain DOF order (Cosserat rod section frame).
DOF_ORDER = ("axial", "shear_y", "shear_z", "torsion", "bending_y", "bending_z")

__all__ = [
    "COUPLING_API_VERSION",
    "DOF_ORDER",
    "XsecStiffnessResult",
    "xsec_stiffness",
]


@dataclass(frozen=True)
class XsecStiffnessResult:
    """Cross-section tangent stiffness at one operating point.

    Attributes
    ----------
    kappa_1_m, tension_N : float
        The operating point the tangent was evaluated at.
    temperature_C : float or None
        Operating temperature, if supplied (see the temperature limitation).
    EI_tangent_Nm2 : float
        **The headline value**: ``dM/dκ`` of the Papailiou slip-front backbone at
        ``kappa_1_m`` — the stiffness a global solver should use locally.
    EI_max_Nm2, EI_min_Nm2 : float
        Full-stick (κ→0) and full-slip (large κ) bounds the tangent lies between.
    EA_N, GJ_Nm2 : float
        Axial and (armour) torsional stiffness from the Witz-Tan axial-torsional
        matrix.
    EA_torsion_coupling : float
        The axial↔torsion off-diagonal ``K_et`` [N·m].
    K_6x6 : numpy.ndarray
        6×6 tangent stiffness in :data:`DOF_ORDER`. Modelled entries: axial,
        torsion, axial-torsion coupling, and both bending DOFs (= ``EI_tangent``).
        **Shear DOFs and bending↔(axial/torsion) couplings are NaN** (not
        modelled).
    dof_order : tuple[str, ...]
        The DOF labels for ``K_6x6`` (== :data:`DOF_ORDER`).
    api_version : str
        :data:`COUPLING_API_VERSION` at the time of the call.
    provenance : Provenance or None
    """

    kappa_1_m: float
    tension_N: float
    temperature_C: float | None
    EI_tangent_Nm2: float
    EI_max_Nm2: float
    EI_min_Nm2: float
    EA_N: float
    GJ_Nm2: float
    EA_torsion_coupling: float
    K_6x6: FloatArray
    dof_order: tuple[str, ...]
    api_version: str
    provenance: Provenance | None = None


def xsec_stiffness(
    cable: CableData,
    kappa_1_m: float,
    *,
    tension_N: float = 0.0,
    temperature_C: float | None = None,
    mu: float | None = None,
    n_ramp: int = 400,
    delta_kappa_1_m: float | None = None,
    core_friction: bool | CoreFrictionSpec | None = None,
) -> XsecStiffnessResult:
    r"""Cross-section tangent stiffness at ``(κ, T)`` for a global solver.

    Differentiates the implemented Papailiou slip-front moment
    (:func:`cablexsec.compute_mkappa`) along a monotonic loading path
    ``0 → κ`` and returns the local slope ``dM/dκ`` plus the axial / torsional
    stiffnesses and a 6×6 tangent.

    Parameters
    ----------
    cable : CableData
        A fully-specified cable.
    kappa_1_m : float
        Operating curvature magnitude [1/m] (``≥ 0``).
    tension_N : float, optional
        Operating effective tension [N] (sets the contact ``F_n``).
    temperature_C : float or None, optional
        Operating temperature [°C]. **Recorded but not yet wired into the
        tangent** (see limitations) — supplied here so the contract is stable
        when thermal coupling is added.
    mu : float, optional
        Armour friction coefficient (default material ``mu_on_hdpe``).
    n_ramp : int, optional
        Steps on the monotonic loading path used for the finite difference.
    delta_kappa_1_m : float or None, optional
        Central-difference half-step; default ``max(1e-4, 0.01·κ)``.
    core_friction : bool or CoreFrictionSpec, optional
        Include the laid-core stick-slip law in the moment
        (:func:`cablexsec.compute_mkappa`); default armour only.

    Returns
    -------
    XsecStiffnessResult
    """
    from cablexsec.api import compute_mkappa  # noqa: PLC0415 (avoid import cycle)
    from cablexsec.mechanics.stiffness import (  # noqa: PLC0415
        axial_torsional_matrix,
    )

    if not math.isfinite(kappa_1_m) or kappa_1_m < 0.0:
        raise PhysicalRangeError(
            f"kappa_1_m = {kappa_1_m} is physically inadmissible; a curvature "
            "magnitude must be finite and >= 0."
        )
    k0 = float(kappa_1_m)
    dk = delta_kappa_1_m if delta_kappa_1_m is not None else max(1.0e-4, 0.01 * k0)

    # Monotonic loading path 0 -> k0 + dk; tangent = central difference at k0.
    k_top = k0 + dk
    ramp = np.linspace(0.0, k_top, max(n_ramp, 8))
    mk = compute_mkappa(
        cable, ramp, tension_N=tension_N, mu=mu, core_friction=core_friction
    )
    if temperature_C is not None:
        warnings.warn(
            "temperature_C is recorded but no temperature-dependent section "
            "constitutive law is available; the returned tangent is isothermal.",
            RuntimeWarning,
            stacklevel=2,
        )
    moment = mk.moment_Nm
    i = int(np.argmin(np.abs(ramp - k0)))
    lo = max(i - 1, 0)
    hi = min(i + 1, ramp.size - 1)
    span = ramp[hi] - ramp[lo]
    ei_tan = (
        float((moment[hi] - moment[lo]) / span) if span > 0 else float(mk.EI_max_Nm2)
    )

    at = axial_torsional_matrix(cable)
    ea_n, k_et, gj = float(at.K_ee), float(at.K_et), float(at.K_tt)
    ei_max, ei_min = float(mk.EI_max_Nm2), float(mk.EI_min_Nm2)

    # 6×6 tangent: fill the modelled entries; NaN for un-modelled shear DOFs and
    # bending<->axial/torsion couplings (NOT a fabricated zero).
    k6 = np.full((6, 6), np.nan, dtype=np.float64)
    k6[0, 0] = ea_n  # axial
    k6[3, 3] = gj  # torsion (armour, Witz-Tan)
    k6[0, 3] = k6[3, 0] = k_et  # axial<->torsion coupling
    k6[4, 4] = ei_tan  # bending y (tangent)
    k6[5, 5] = ei_tan  # bending z (tangent)

    return XsecStiffnessResult(
        kappa_1_m=k0,
        tension_N=float(tension_N),
        temperature_C=temperature_C,
        EI_tangent_Nm2=ei_tan,
        EI_max_Nm2=ei_max,
        EI_min_Nm2=ei_min,
        EA_N=ea_n,
        GJ_Nm2=gj,
        EA_torsion_coupling=k_et,
        K_6x6=k6,
        dof_order=DOF_ORDER,
        api_version=COUPLING_API_VERSION,
        provenance=Provenance(
            method="cross-section tangent dM/dkappa from the Papailiou slip-front "
            "backbone (operating-point local slope), with the Witz-Tan "
            "axial-torsional stiffnesses",
            citations=("[Papailiou1997]", "[WitzTan1992a/b]", "[Kim2020] Eq.3.24"),
            gates=(
                "kappa->0 tangent == EI_max; large kappa -> EI_min; monotone "
                "softening between (L6.3 bounds gate)",
                f"versioned + contract-tested (COUPLING_API_VERSION "
                f"{COUPLING_API_VERSION})",
            ),
            limitations=(
                "returns the LOCAL cross-section tangent; the global solver owns "
                "the dynamics (this is not a global stiffness)",
                "tangent is the monotonic-loading (virgin-curve) slope; on a "
                "reversal the instantaneous unloading tangent is stiffer "
                "(Masing) — supply load history if reversal stiffness is needed",
                "temperature_C is recorded but NOT yet wired into the tangent "
                "(no E(T)/contact-pressure temperature dependence applied)",
                "6x6 shear DOFs and bending<->axial/torsion couplings are NaN "
                "(Euler-Bernoulli section; not modelled — not zero)",
                "GJ is the Witz-Tan armour torsional term (K_tt), not a "
                "full-section torsion with polymer/core contributions",
            ),
        ),
    )
