# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Metallic-sheath (cylindrical-shell) stiffness and stress.

Export cables carry a thin metallic sheath — extruded lead or corrugated
aluminium — modelled as a ``cylindrical_shell`` layer (mean radius ``r_mid``,
wall ``t``). This module provides its thin-wall axial and bending stiffness and
the membrane and bending stress components in the wall.

Thin-wall stiffness (mean radius ``r``, wall ``t``, effective modulus ``E_eff``):

.. math::
    EA = E_\text{eff}\, 2π r t \quad(\text{exact}), \qquad
    EI = E_\text{eff}\, π r^3 t,

where the thin-wall second moment ``π r^3 t`` equals the exact annulus value
``\tfrac{π}{4}(r_o^4 - r_i^4)`` to within ``(t/2r)^2`` — under 0.1 % for
``t/r ≲ 0.06`` (the closed-form gate). For a corrugated sheath the effective
modulus is reduced by the corrugation factor ``E_eff = f_\text{corr} E``
as an explicitly specified model input. The representative example value
``f_corr = 0.5`` is not a CIGRE recommendation or a measured property.

Stress components in the wall ([DNVGL-RP-C203] / thin-shell theory):

.. math::
    σ_\text{bend} = E_\text{eff} κ r,\quad
    σ_\text{peak} = \mathrm{SCF}\, σ_\text{bend},\quad
    σ_\text{hoop} = \frac{(p_i - p_o) r}{t},

.. math::
    σ_T = -E_\text{eff} α_T ΔT,\qquad
    σ_\text{axial} = E_\text{eff} ε_s,

The longitudinal and hoop components are orthogonal and are not added as
scalars. At the corrugation root,

.. math::
    σ_z = σ_\text{peak} + σ_T + σ_\text{axial},\qquad
    σ_\text{vm} = \sqrt{σ_z^2 + σ_\text{hoop}^2 - σ_z σ_\text{hoop}}.

The radial
Lamé ring stresses enter the axisymmetric chain via
:func:`cablexsec.mechanics.axisymmetric.lame_thick_wall`.

Sheath fatigue is assembled by :func:`cablexsec.compute_shell_fatigue`.
Corrugated aluminium can use the registered EN 1999-1-3 screening curve when
an independently specified corrugation SCF is present. Lead-sheath fatigue
remains blocked because no traceable life curve has been registered.

References
----------
[CIGRE TB 623] (mechanical testing context); [DNVGL-RP-C203] (wall stress);
[Johnson1985] / standard thin-shell theory; Lamé via :mod:`axisymmetric`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cablexsec.cables.schema import CylindricalShellLayer
from cablexsec.exceptions import CableInputError
from cablexsec.materials import library
from cablexsec.provenance import Provenance

__all__ = [
    "ShellStressResult",
    "shell_axial_stiffness",
    "shell_bending_stiffness",
    "shell_wall_stress",
]

_SHELL_STRESS_PROVENANCE = Provenance(
    method="thin-wall metallic-sheath stress: longitudinal root stress and "
    "plane-stress von Mises combination with hoop stress",
    citations=("[DNVGL-RP-C203]",),
    gates=("EI thin-wall matches exact annulus to (t/2r)^2 (<0.1% for t/r<=0.06)",),
    limitations=(
        "corrugated-Al fatigue can be screened with the registered EN 1999-1-3 "
        "curve; lead-sheath fatigue remains BLOCKED pending a traceable curve",
        "corrugation factor f_corr and root SCF are layer inputs; for CIGRE "
        "TB880 they are representative (not from the source table)",
        "thin-wall pressure stress uses the shell's mean radius and wall thickness; "
        "the effective modulus and root SCF do not establish corrugated-shell "
        "radial compliance",
    ),
)


@dataclass(frozen=True)
class ShellStressResult:
    """Five-contribution metallic-sheath wall stress [Pa].

    Attributes
    ----------
    sigma_bending_Pa, sigma_peak_Pa, sigma_hoop_Pa, sigma_thermal_Pa,
    sigma_axial_Pa, sigma_longitudinal_Pa, sigma_von_mises_Pa : float
        Bending, SCF-amplified peak, hoop, thermal, axial, longitudinal and
        plane-stress von Mises values [Pa]. ``sigma_total_Pa`` is retained as
        a compatibility alias for the governing von Mises value.
    provenance : Provenance or None
        Backing citations and the blocked-fatigue limitation.
    """

    sigma_bending_Pa: float
    sigma_peak_Pa: float
    sigma_hoop_Pa: float
    sigma_thermal_Pa: float
    sigma_axial_Pa: float
    sigma_longitudinal_Pa: float
    sigma_von_mises_Pa: float
    sigma_total_Pa: float
    provenance: Provenance | None = None


def _effective_modulus(layer: CylindricalShellLayer) -> float:
    """Corrugation-reduced modulus ``E_eff = f_corr · E`` [Pa]."""
    e = library.get_property(layer.material, "E_Pa")
    return e * layer.f_corrugation


def shell_axial_stiffness(layer: CylindricalShellLayer) -> float:
    """Thin-wall axial stiffness ``EA = E_eff · 2π r t`` [N] (area exact)."""
    assert layer.r_mid_m is not None and layer.t_m is not None
    return _effective_modulus(layer) * 2.0 * math.pi * layer.r_mid_m * layer.t_m


def shell_bending_stiffness(layer: CylindricalShellLayer) -> float:
    """Thin-wall bending stiffness ``EI = E_eff · π r³ t`` [N·m²].

    The thin-wall second moment ``π r³ t`` matches the exact annulus
    ``π/4 (r_o⁴ - r_i⁴)`` to ``(t/2r)²`` (< 0.1 % for ``t/r ≲ 0.06``).
    """
    assert layer.r_mid_m is not None and layer.t_m is not None
    return _effective_modulus(layer) * math.pi * layer.r_mid_m**3 * layer.t_m


def shell_wall_stress(
    layer: CylindricalShellLayer,
    kappa_1_m: float,
    eps_axial: float,
    EA_total_N: float,
    *,
    p_int_Pa: float = 0.0,
    p_ext_Pa: float = 0.0,
    delta_T_K: float = 0.0,
    youngs_modulus_Pa: float | None = None,
) -> ShellStressResult:
    """Five-contribution metallic-sheath wall stress.

    Parameters
    ----------
    layer : CylindricalShellLayer
        The metallic-sheath layer (mean radius, wall, ``scf``, ``f_corrugation``).
    kappa_1_m : float
        Cable curvature [1/m].
    eps_axial : float
        Global cable axial strain. Equal strain already performs the load
        sharing, so multiplying stress by another stiffness share would count
        load sharing twice.
    EA_total_N : float
        Total cable axial stiffness [N] for the axial-stress share.
    p_int_Pa, p_ext_Pa : float, optional
        Internal and external pressures [Pa] (hoop stress).
    delta_T_K : float, optional
        Sheath temperature rise above the stress-free reference [K].
    youngs_modulus_Pa : float, optional
        Material modulus before the declared corrugation factor is applied.
    """
    assert layer.r_mid_m is not None and layer.t_m is not None
    if youngs_modulus_Pa is not None and (
        not math.isfinite(youngs_modulus_Pa) or youngs_modulus_Pa <= 0
    ):
        raise CableInputError("shell material modulus must be finite and positive")
    e_eff = (
        _effective_modulus(layer)
        if youngs_modulus_Pa is None
        else youngs_modulus_Pa * layer.f_corrugation
    )
    r, t = layer.r_mid_m, layer.t_m
    alpha_t = library.get_property(layer.material, "alpha_T_K")

    sigma_bending = e_eff * kappa_1_m * r
    sigma_peak = layer.scf * sigma_bending  # corrugation-root SCF
    sigma_hoop = (p_int_Pa - p_ext_Pa) * r / t
    # Fully restrained thermal increment. Heating is compressive in the
    # tensile-positive convention. A common free strain must instead be solved
    # by compute_component_response.
    sigma_thermal = -e_eff * alpha_t * delta_T_K
    _ = EA_total_N  # retained for API compatibility
    sigma_axial = e_eff * eps_axial
    sigma_longitudinal = sigma_peak + sigma_thermal + sigma_axial
    sigma_von_mises = math.sqrt(
        sigma_longitudinal**2 + sigma_hoop**2 - sigma_longitudinal * sigma_hoop
    )
    return ShellStressResult(
        sigma_bending_Pa=sigma_bending,
        sigma_peak_Pa=sigma_peak,
        sigma_hoop_Pa=sigma_hoop,
        sigma_thermal_Pa=sigma_thermal,
        sigma_axial_Pa=sigma_axial,
        sigma_longitudinal_Pa=sigma_longitudinal,
        sigma_von_mises_Pa=sigma_von_mises,
        sigma_total_Pa=sigma_von_mises,
        provenance=_SHELL_STRESS_PROVENANCE,
    )
