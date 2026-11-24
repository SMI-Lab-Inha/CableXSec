# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Electrical insulation life — water-tree screening (Mazzanti 2007 model).

A **screening** post-processor: it ranks the electro-thermal ageing
severity of the XLPE insulation; it is *not* a calibrated life prediction.

Electro-thermal life model ([Mazzanti2007] Eq. 17, IEEE TPWRD 22(4):2000-2009):

.. math::
    L(E, T) = L_0 \left(\frac{E_0}{E}\right)^{\,n_0 - b\,c_T}\,\exp(-B\,c_T),
    \qquad c_T = \frac{1}{T_0} - \frac{1}{T}\ \ [\mathrm{K}^{-1}],

with ``E`` the maximum (Laplacian) field, ``E_0`` the threshold field below
which electrical ageing is negligible, ``n_0`` the voltage-endurance coefficient
(VEC) at ``T = T_0``, ``B = ΔW/k`` the Arrhenius coefficient [K], and ``b`` the
**electrical-thermal synergism** coefficient that makes the effective VEC
temperature-dependent, ``n(T) = n_0 - b\,c_T``. At the reference (``E = E_0``,
``T = T_0``) the factor is exactly 1, recovering ``L_0``.

Mazzanti Table I (XLPE): ``n_0 = 15``, ``B = 12430`` K, ``b = 4420``,
``E_0 = 5`` kV/mm. (Note ``n(90 °C) ≈ 12.1`` — a constant ``n = 12`` is ``n(T)``
at operating temperature, not the VEC ``n_0``; the synergism ``b`` must not be
dropped.)

The coaxial Laplacian field at the conductor screen is
``E = U_0 / (r_i ln(r_o/r_i))`` with ``U_0 = V_rated/√3``.

**Mechanical screening (NOT Mazzanti 2007).** Mazzanti's model is electro-
thermal only — it has no mechanical-strain term. The optional
:func:`mechanical_derating` factor ``exp(-β ε_mech)`` is a CableXSec screening
**extension** for water-tree growth under bending strain; it is *not* part of
[Mazzanti2007] and its ``β`` must be cited to a mechanical-strain / electrical-
tree source (cite-on-read; e.g. a study of mechanical strain and electrical-tree
growth in XLPE) before any quantitative use.

References
----------
[Mazzanti2007] G. Mazzanti, "Analysis of the Combined Effects of Load Cycling,
Thermal Transients, and Electrothermal Stress on Life Expectancy of High-Voltage
AC Cables," IEEE Trans. Power Delivery 22(4):2000-2009, Oct. 2007 — Eq. 17,
Table I (parameters). Screening use only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from cablexsec.cables.schema import CylindricalOffsetLayer
from cablexsec.cables.schema import scope_entry_points as _scope_entry_points
from cablexsec.exceptions import CableInputError
from cablexsec.provenance import Provenance, ValidatedRange, check_range

if TYPE_CHECKING:
    from cablexsec.cables.schema import CableData

__all__ = [
    "LifeResult",
    "coaxial_field",
    "compute_electrical_life",
    "mazzanti_life_factor",
    "mechanical_derating",
    "mechanical_strain",
]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LifeResult:
    """Water-tree screening result (relative, not a calibrated lifetime).

    Attributes
    ----------
    E_field_Vm : float
        Laplacian electric field at the conductor screen [V/m].
    eps_mech : float
        Mechanical (bending) strain amplitude at the insulation outer radius.
    life_factor : float
        ``L / L_0`` — dimensionless ageing factor relative to the reference
        condition (``1`` at reference; ``< 1`` indicates faster ageing).
    life_years : float
        ``L_0 * life_factor`` if a reference life ``L0_years`` is supplied.
    provenance : Provenance or None
        Backing citation and the screening-only limitation.
    """

    E_field_Vm: float
    eps_mech: float
    life_factor: float
    life_years: float
    provenance: Provenance | None = None


def coaxial_field(voltage_kV: float, r_inner_m: float, r_outer_m: float) -> float:
    r"""Peak coaxial Laplacian field at the inner conductor screen [V/m].

    ``E = U_0 / (r_i ln(r_o/r_i))`` with phase voltage ``U_0 = V/√3``.
    """
    u0 = voltage_kV * 1.0e3 / math.sqrt(3.0)
    return u0 / (r_inner_m * math.log(r_outer_m / r_inner_m))


def mechanical_strain(kappa_x_series: FloatArray, r_xlpe_outer_m: float) -> float:
    """Bending strain amplitude ``std(κ_x) · r_xlpe_outer`` at the insulation OD."""
    return float(np.std(np.asarray(kappa_x_series, dtype=np.float64))) * r_xlpe_outer_m


def mazzanti_life_factor(
    E_field_Vm: float,
    T_K: float,
    *,
    E0_Vm: float,
    T0_K: float,
    n0: float,
    B_K: float,
    b_synergism: float = 0.0,
) -> float:
    r"""Electro-thermal life factor ``L / L_0`` ([Mazzanti2007] Eq. 17).

    ``(E_0/E)^{n_0 - b c_T} · exp(-B c_T)`` with ``c_T = 1/T_0 - 1/T``. The VEC
    exponent is temperature-dependent through the synergism ``b``,
    ``n(T) = n_0 - b c_T``. Returns exactly ``1`` at the reference (``E = E_0``,
    ``T = T_0``). With ``b = 0`` it collapses to the simple inverse-power-law ×
    Arrhenius form. **Electro-thermal only — no mechanical term** (see
    :func:`mechanical_derating`).
    """
    c_t = 1.0 / T0_K - 1.0 / T_K
    exponent = n0 - b_synergism * c_t
    electrical = (E0_Vm / E_field_Vm) ** exponent
    thermal = math.exp(-B_K * c_t)
    return float(electrical * thermal)


def mechanical_derating(eps_mech: float, beta: float) -> float:
    r"""Mechanical screening multiplier ``exp(-β ε_mech)`` — NOT Mazzanti 2007.

    A CableXSec screening **extension** for water-tree growth under bending
    strain. [Mazzanti2007] is electro-thermal only (no mechanical term); this
    factor models the experimentally-observed acceleration of electrical/water
    tree growth under tensile mechanical strain in XLPE — see
    "The Effect of Mechanical Strain on Electrical Tree Development in XLPE".
    The coefficient ``β`` must be calibrated
    to / cited from that source (cite-on-read) before any quantitative use; it
    is **not** a Mazzanti parameter. ``1`` at zero strain; monotone-decreasing
    in ``ε_mech``. This electro-thermal (Mazzanti) ⊕ mechanical (separate
    source) decomposition is a Paper-2 contribution.
    """
    return math.exp(-beta * eps_mech)


def _insulation_layer(cable: CableData) -> CylindricalOffsetLayer:
    """The XLPE insulation layer carrying the Mazzanti ``electrical`` block."""
    for lyr in cable.layers:
        if (
            isinstance(lyr, CylindricalOffsetLayer)
            and lyr.electrical is not None
            and lyr.electrical.vec_exponent_n is not None
            and lyr.r_inner_m is not None
            and lyr.r_outer_m is not None
        ):
            return lyr
    raise CableInputError(
        "cable has no insulation layer with a Mazzanti 'electrical' block "
        "(vec_exponent_n / thermal_b_k / field_threshold_kV_mm)."
    )


def compute_electrical_life(
    cable: CableData,
    kappa_x_series: FloatArray,
    *,
    T_conductor_C: float,
    T0_C: float,
    L0_years: float = 1.0,
) -> LifeResult:
    """Water-tree screening for a cable's XLPE insulation ([Mazzanti2007] Eq. 17).

    Reads the VEC ``n0`` (``vec_exponent_n``), Arrhenius ``B`` (``thermal_b_k``),
    synergism ``b`` (``vec_synergism_b``) and threshold field ``E0``
    (``field_threshold_kV_mm``) from the insulation ``electrical`` block, plus the
    optional mechanical-screening ``β`` (``mechanical_beta``, a CableXSec
    extension — not Mazzanti). The reference field ``E0`` and temperature ``T0_C``
    define ``L_0``; the relative ageing factor ``L/L_0`` is returned.
    **Screening only.**

    Parameters
    ----------
    cable : CableData
        Cable with a voltage rating and an insulation ``electrical`` block.
    kappa_x_series : numpy.ndarray
        Curvature history [1/m]; its standard deviation sets ``ε_mech`` for the
        (flagged, non-Mazzanti) mechanical screening multiplier.
    T_conductor_C, T0_C : float
        Operating and reference temperatures [°C].
    L0_years : float, optional
        Reference life; ``life_years = L0_years · life_factor``.
    """
    if cable.cable.voltage_kV is None:
        raise CableInputError("cable has no voltage rating for the Laplacian field.")
    lyr = _insulation_layer(cable)
    elec = lyr.electrical
    assert (
        elec is not None
        and elec.vec_exponent_n is not None
        and elec.thermal_b_k is not None
        and elec.field_threshold_kV_mm is not None
        and lyr.r_inner_m is not None
        and lyr.r_outer_m is not None
    )
    e_field = coaxial_field(cable.cable.voltage_kV, lyr.r_inner_m, lyr.r_outer_m)
    e0_Vm = elec.field_threshold_kV_mm * 1.0e6  # kV/mm -> V/m
    factor = mazzanti_life_factor(
        e_field,
        T_conductor_C + 273.15,
        E0_Vm=e0_Vm,
        T0_K=T0_C + 273.15,
        n0=elec.vec_exponent_n,
        B_K=elec.thermal_b_k,
        b_synergism=elec.vec_synergism_b or 0.0,
    )
    eps_mech = mechanical_strain(kappa_x_series, lyr.r_outer_m)
    beta = elec.mechanical_beta
    mech_applied = bool(beta)
    if beta:  # flagged CableXSec extension (not Mazzanti)
        factor *= mechanical_derating(eps_mech, beta)

    # Soft guard: Mazzanti Table I (XLPE) was fitted up to the 90 degC limit.
    check_range(
        T_conductor_C,
        ValidatedRange(
            "conductor temperature [degC]",
            -20.0,
            90.0,
            "[Mazzanti2007] Table I (XLPE) fitted to the 90 degC operating limit",
        ),
    )
    limitations = [
        "SCREENING ONLY — relative ageing factor, not a calibrated lifetime",
        "Laplacian (capacitive) field; space-charge / WTR conductivity ignored",
    ]
    if mech_applied:
        limitations.append(
            "mechanical_derating exp(-beta*eps) is a CableXSec extension, NOT "
            "[Mazzanti2007]; its beta requires a cited mechanical-strain source"
        )
    return LifeResult(
        E_field_Vm=e_field,
        eps_mech=eps_mech,
        life_factor=factor,
        life_years=L0_years * factor,
        provenance=Provenance(
            method="Mazzanti (2007) Eq.17 electro-thermal life factor "
            "(temperature-dependent VEC via synergism b)",
            citations=("[Mazzanti2007] Eq.17, Table I",),
            gates=("reference E=E0,T=T0 -> factor=1 (exact)",),
            limitations=tuple(limitations),
        ),
    )


# Public entry points read the cable's own materials when called directly.
_scope_entry_points(globals(), __all__)
