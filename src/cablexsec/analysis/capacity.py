# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Cable capacity envelope from independently evaluated screening mechanisms.

The tension–curvature ``(N, κ)`` screen is the inner envelope of the mechanisms
that can be evaluated from the available cable-specific inputs. Each available
mechanism has its own usage factor ``UF`` (= demand / capacity). The governing
mechanism at a given ``(N, κ)`` is the one with the largest ``UF``, and the
boundary is ``max(UF) = 1``. An unavailable mechanism is represented by ``NaN``
and is never allowed to govern.

============== ============================================ ==================
Mechanism      Usage factor                                 Reference
============== ============================================ ==================
Wire yielding  ``σ_max / σ_y``                               frozen stress model
Bird-caging    ``N_armour_comp / P_crit``                    compression screen
Polymer crush  ``p_contact_max / σ_allow``                  supplied project limit
Conductor      ``σ_conductor / σ_UTS``                       conductor stress
Components     largest utilisation of the other components  component limits
============== ============================================ ==================

Component limits: the 0.2 % proof stress of the conductor on the strand basis
of its bending stiffness and the allowable strain of polymers, metallic
screens and sheaths, wherever the material gives them (the first yield of a
metallic screen or sheath does not bound the capacity,
:func:`~cablexsec.analysis.components.capacity_utilisation`)
(:mod:`cablexsec.analysis.component_limits`), from
:func:`~cablexsec.analysis.components.component_usage_curve` in the
frictional core slip state; their curvature capacity is where the largest of
them reaches one. A component without a limit is not checked; layers outside
the load path (loose screens) are left out.

Wire yielding and conductor fracture grow with curvature; their boundary defines
a curvature capacity ``κ_max(N)`` that decreases with tension. Bird-caging and
polymer crushing are tension-driven (curvature-independent): if an evaluated
mechanism reaches ``UF = 1`` at a tension the curvature capacity collapses to
zero there. Polymer crushing is deliberately unavailable unless the caller
supplies a cable- and layer-specific allowable with a source. A generic HDPE
strength is not a defensible substitute for grade, temperature, strain-rate,
confinement, and failure-mode dependent test data.

Wire yielding is evaluated at every physical wire using the same slip-front
law as the stress and moment--curvature engines. Friction stress grows from
the stick response and saturates only after local slip. The yield curvature is
the first positive root of the maximum absolute wire stress. In the
zero-contact, zero-tension limit it reduces to
``κ_y = σ_y / [E_w \cos α (1 + \sin^2 α)\,d/2]`` as an analytical gate.

Bird-caging is **active only when the armour is in net compression**
(``T_layer < 0``); under tension the armour wires carry tension and cannot
bird-cage. Its usage factor is the armour compression screen of
:mod:`cablexsec.analysis.compression` (lateral lower bound per layer, radial
buckling of the armour stack with wire-yield interaction), with demand and
capacity both expressed as layer axial forces. :func:`birdcage_buckling_detail`
keeps the per-layer breakdown of the friction-free **lateral** and **radial**
elastic capacities of [SaevikThorsen2017]:

.. math::
    P_1 &= n\,\frac{\cos α}{R^2}\bigl[GJ\sin^4 α
        + (4EI_2 + EI_3 - GJ)\sin^2 α\cos^2 α\bigr]      &&\text{(Eq. 19)}\\
    P_2 &= 2 n \cos α \sqrt{EI_2\,c}                       &&\text{(Eq. 22)}\\
    P_\text{crit} &= \min(P_1, P_2)                        &&\text{(Eq. 36)}

as layer axial forces (all ``n_w`` wires of the layer). For a round armour wire
``I_2 = I_3 = π d^4 / 64`` and ``GJ = EI/(1+ν)`` (``J = 2I``,
``G = E/[2(1+ν)]``); all from the YAML/material library, no calibration.

The radial foundation stiffness ``c`` is **layer-specific**: it is the hoop
stiffness of the continuous annular material lying *radially outboard* of that
armour layer (the inner armour is restrained by the bedding/cover stack above
it, the outer armour by the outer sheath), summed over those layers
([SaevikThorsen2017] Eq. 23, one wire-width contribution):

.. math::  c = \sum_\text{outboard annuli} \frac{2π E_s t_s \cos α}{n_w R}.

Helical (armour) layers are not continuous annuli and contribute no Eq.-23 hoop
term; a wire with no outboard annulus (a bare strand) has no radial foundation,
so only the lateral bound ``P_1`` applies and a runtime note is emitted. The
per-wire critical *stress* ``σ_c1 = P_2 / (n_w A) = (2 cos α / A)√(EI_2 c)``
reproduces the cross-check form of Chae, Nam & Shim (2023) Eq. 17.

Thermal stress
--------------
At rated current the signed armour thermal stress is added to the mechanical
wire stress before the yield comparison. A Goodman correction belongs to the
fatigue calculation and is not used as an extreme-load yield law. The
isothermal envelope is recovered exactly when ``σ_T = 0``.

References
----------
[SaevikThorsen2017] J. Offshore Mech. Arct. Eng. 139(4):041701, Eqs. 19/22/23/36
(armour lateral + radial bird-caging); [Saevik2011] §5.2 (F_n, friction
saturation); [MenardCartraud2023] Eq. 14 (local bending); [Costello1997] (helix
axial stress).

Notes
-----
The bird-caging model replaces the earlier Hobbs
column-buckling form ``4π²EA cos⁴α sinα/(n L_p²)``; it is replaced by
    the foundation-restrained [SaevikThorsen2017] ``min(P_1, P_2)`` model.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from cablexsec.analysis.compression import _armour_gaps, missing_armour_message
from cablexsec.cable_adapters import cable_to_mkappa_specs
from cablexsec.cables.schema import (
    CableData,
    CylindricalLayer,
    CylindricalOffsetLayer,
    CylindricalShellLayer,
    HelicalLayer,
)
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.materials import library
from cablexsec.mechanics.axisymmetric import (
    AxisymmetricResult,
    compute_contact_forces,
)
from cablexsec.mechanics.papailiou import wire_slip_stress_bounds
from cablexsec.mechanics.section_state import SectionState, solve_section_state
from cablexsec.mechanics.stiffness import ea_fixed
from cablexsec.mechanics.wire_stress import local_bending_amplitude
from cablexsec.provenance import Provenance


def _contact_forces(cable: CableData, T_N: float) -> AxisymmetricResult:
    """Bounded contact forces retaining the declared shell axial rigidity."""
    return compute_contact_forces(cable, T_N)


__all__ = [
    "BirdcageDetail",
    "CapacityResult",
    "MECHANISMS",
    "birdcage_buckling_detail",
    "birdcage_critical_force",
    "compute_capacity",
    "wire_yield_curvature",
    "wire_yield_curvature_levels",
]

FloatArray = NDArray[np.float64]

MECHANISMS = (
    "wire_yield",
    "bird_caging",
    "polymer_crush",
    "conductor_fracture",
    "component_limit",
)


@dataclass(frozen=True)
class CapacityResult:
    """Capacity envelope over a tension sweep.

    Attributes
    ----------
    tension_N : numpy.ndarray
        Input cable tensions (signed; negative = compression) [N].
    kappa_max_1_m : numpy.ndarray
        Governing curvature capacity at each tension [1/m] (0 where a
        tension-driven mechanism already reaches UF = 1).
    governing : list[str]
        Governing mechanism name at each tension (one of :data:`MECHANISMS`).
    uf_bird_caging, uf_polymer_crush, uf_conductor : numpy.ndarray
        Tension-driven usage factors at each tension (curvature-independent).
        Polymer-crush values are NaN when no sourced project allowable was
        supplied.
    unavailable_mechanisms : tuple[str, ...]
        Mechanisms excluded from governing selection because the required
        evidence was not supplied.
    provenance : Provenance or None
        Backing citations/gates and active limitations for the envelope.
    kappa_component_1_m : numpy.ndarray
        Curvature at which the largest utilisation of the components other
        than the armour reaches one (``inf`` where never; empty when the
        components were not checked) [1/m].
    governing_component : tuple[tuple[str, str], ...]
        The component and its limit at that curvature (empty strings where no
        component has a limit).
    kinematic_layers : tuple[str, ...]
        Layers outside the load path, not checked.
    """

    tension_N: FloatArray
    kappa_max_1_m: FloatArray
    governing: list[str]
    uf_bird_caging: FloatArray
    uf_polymer_crush: FloatArray
    uf_conductor: FloatArray
    unavailable_mechanisms: tuple[str, ...] = ()
    provenance: Provenance | None = None
    kappa_component_1_m: FloatArray = field(default_factory=lambda: np.zeros(0))
    governing_component: tuple[tuple[str, str], ...] = ()
    kinematic_layers: tuple[str, ...] = ()


@dataclass(frozen=True)
class BirdcageDetail:
    """Per-layer [SaevikThorsen2017] bird-caging buckling breakdown.

    Attributes
    ----------
    layer : str
        Armour layer name.
    P1_lateral_N, P2_radial_N : float
        Lateral (Eq. 19) and radial (Eq. 22) elastic buckling capacities of
        the whole layer, as axial forces of its ``n_w`` wires [N].
        ``P2_radial_N`` is ``inf`` when the layer has no outboard radial
        restraint (a bare strand): only ``P1`` then applies.
    P_crit_N : float
        ``min(P1, P2)`` — the governing layer critical load [N] (Eq. 36,
        without the Eq. 35 interaction; see
        :func:`cablexsec.analysis.compression.compression_screen`).
    foundation_c_Pa : float
        Radial foundation stiffness ``c`` [Pa] (Eq. 23, summed over the
        outboard annuli in ``restraint_layers``).
    restraint_layers : tuple[str, ...]
        Names of the continuous annular layers providing the radial restraint
        (empty for a bare strand).
    governing_mode : str
        ``"lateral"`` or ``"radial"`` — which of P1/P2 set ``P_crit``.
    provenance : Provenance or None
        Backing citation/gate and the active idealization limitation.
    """

    layer: str
    P1_lateral_N: float
    P2_radial_N: float
    P_crit_N: float
    foundation_c_Pa: float
    restraint_layers: tuple[str, ...]
    governing_mode: str
    provenance: Provenance | None = None


# Bird-caging provenance — shared by birdcage_buckling_detail and the envelope.
_BIRDCAGE_PROVENANCE = Provenance(
    method="Saevik-Thorsen min(P1 lateral Eq.19, P2 radial Eq.22) bird-caging, "
    "foundation-restrained with layer-specific c (Eq.23)",
    citations=("[SaevikThorsen2017]", "[ChaeNamShim2023] Eq.17 cross-check"),
    gates=("Chae sigma_c1 = P2/(nA) to 1.6e-16",),
    limitations=(
        "radial foundation c sums the contribution of each continuous annulus outboard "
        "of the armour; shell radial restraint uses the smooth-wall geometry "
        "and modulus, not a separately identified corrugation law",
        "friction-free lateral lower bound (no inter-wire friction in P1)",
    ),
)


def _armour_layers(cable: CableData) -> list[HelicalLayer]:
    """Helical (armour) layers with full geometry and a yield strength."""
    return [lyr for lyr in cable.helical_layers if not _armour_gaps(lyr)]


def _conductor_layer(cable: CableData) -> CylindricalOffsetLayer | None:
    """The (offset) conductor core layer, if present and fully specified."""
    for lyr in cable.layers:
        if isinstance(lyr, CylindricalOffsetLayer):
            try:
                library.get_property(lyr.material, "sigma_UTS_Pa")
            except KeyError:
                continue
            if lyr.r_outer_m is not None:
                return lyr
    return None


def _wire_area(layer: HelicalLayer) -> float:
    assert layer.d_wire_m is not None
    return math.pi * layer.d_wire_m**2 / 4.0


def _local_bending_slope(layer: HelicalLayer) -> float:
    """``∂σ_B/∂κ`` at the extreme-fibre wire [Pa·m] ([MenardCartraud2023] Eq. 14)."""
    assert layer.lay_angle_rad is not None and layer.d_wire_m is not None
    e_w = library.get_property(layer.material, "E_Pa")
    # At phi=phi_bend the perimeter amplitude gives the extreme-fibre slope.
    return float(
        _local_bending_max(layer, 1.0, np.array(0.0), e_w, layer.lay_angle_rad, 0.0)
    )


def _local_bending_max(
    layer: HelicalLayer,
    kappa: float,
    phi: NDArray[np.float64],
    e_w: float,
    lay_angle_rad: float,
    phi_bend: float,
) -> NDArray[np.float64]:
    """Largest local bending stress over the wire section [Pa].

    Round wires: :func:`local_bending_amplitude`. Rectangular and profiled
    wires (gap G1): the normal curvature at the outer face ``t/2`` and the
    lateral curvature at the edge ``b/2`` combined at the corner
    (:meth:`~cablexsec.mechanics.tendon_section.WireSection.max_bending_stress`).
    """
    assert layer.d_wire_m is not None
    if layer.wire_shape == "round":
        return np.asarray(
            local_bending_amplitude(
                kappa, phi, e_w, layer.d_wire_m, lay_angle_rad, phi_bend
            )
        )
    section = layer.wire_section
    angle = np.asarray(phi, dtype=np.float64) - phi_bend
    c, s = math.cos(lay_angle_rad), math.sin(lay_angle_rad)
    face = e_w * abs(kappa) * c * c * math.cos(2 * lay_angle_rad) * np.cos(angle)
    edge = e_w * abs(kappa) * c * (1 + s * s) * np.sin(angle)
    return section.max_bending_stress(
        face * section.half_thickness_m, edge * section.half_width_m
    )


def _round_wire_EI_GJ(layer: HelicalLayer) -> tuple[float, float]:
    """Round-wire weak-axis ``EI`` and torsion ``GJ`` [N·m²] for an armour wire.

    ``I = π d⁴/64`` (``EI_2 = EI_3`` for a circular section); ``J = 2I`` and
    ``G = E/[2(1+ν)]`` give ``GJ = EI/(1+ν)``.
    """
    assert layer.d_wire_m is not None
    e_w = library.get_property(layer.material, "E_Pa")
    nu = library.get_property(layer.material, "nu")
    second_moment = math.pi * layer.d_wire_m**4 / 64.0
    ei = e_w * second_moment
    return ei, ei / (1.0 + nu)


def _annulus_radii(layer: object) -> tuple[float, float] | None:
    """Inner/outer radius [m] of a continuous annular layer, or ``None``.

    Continuous annuli are ``cylindrical`` (``r_inner_m``/``r_outer_m``) and
    ``cylindrical_shell`` (``r_mid_m ± t_m/2``) layers. Helical and offset
    layers are not continuous annuli and return ``None``.
    """
    if isinstance(layer, CylindricalLayer):
        if layer.r_inner_m is None or layer.r_outer_m is None:
            return None
        return layer.r_inner_m, layer.r_outer_m
    if isinstance(layer, CylindricalShellLayer):
        if layer.r_mid_m is None or layer.t_m is None:
            return None
        return layer.r_mid_m - layer.t_m / 2.0, layer.r_mid_m + layer.t_m / 2.0
    return None


def _radial_foundation_stiffness(
    cable: CableData, layer: HelicalLayer
) -> tuple[float, tuple[str, ...]]:
    r"""Radial foundation stiffness ``c`` [Pa] and the restraining layer names.

    Sums the [SaevikThorsen2017] Eq. 23 one-wire-width hoop contribution
    ``2π E_s t_s cos α / (n_w R)`` over every continuous annular layer lying
    radially outboard of ``layer`` (inner radius ≥ this layer's helix radius).
    Returns ``(0.0, ())`` for a wire with no outboard annulus (a bare strand).
    """
    assert (
        layer.n_wires is not None
        and layer.r_helix_m is not None
        and layer.lay_angle_rad is not None
    )
    cos_a = math.cos(layer.lay_angle_rad)
    r_helix = layer.r_helix_m
    c = 0.0
    names: list[str] = []
    for other in cable.layers:
        radii = _annulus_radii(other)
        if radii is None or radii[0] < r_helix:
            continue  # not a continuous annulus, or not outboard of this layer
        t_s = radii[1] - radii[0]
        e_s = library.get_property(other.material, "E_Pa")
        c += 2.0 * math.pi * e_s * t_s * cos_a / (layer.n_wires * r_helix)
        names.append(other.name)
    return c, tuple(names)


def birdcage_buckling_detail(cable: CableData, layer: HelicalLayer) -> BirdcageDetail:
    r"""Per-layer [SaevikThorsen2017] bird-caging breakdown for an armour layer.

    Returns the lateral (Eq. 19) and radial (Eq. 22) elastic buckling
    capacities, ``P_crit = min(P1, P2)`` (Eq. 36), the radial foundation
    stiffness ``c`` (Eq. 23), and the names of the outboard annuli that set it.
    A zero lay angle (a straight wire cannot bird-cage) yields ``P_crit = inf``.
    """
    assert (
        layer.n_wires is not None
        and layer.r_helix_m is not None
        and layer.lay_angle_rad is not None
    )
    a = layer.lay_angle_rad
    if math.isclose(a, 0.0):
        return BirdcageDetail(
            layer=layer.name,
            P1_lateral_N=math.inf,
            P2_radial_N=math.inf,
            P_crit_N=math.inf,
            foundation_c_Pa=0.0,
            restraint_layers=(),
            governing_mode="lateral",
            provenance=_BIRDCAGE_PROVENANCE,
        )
    ei, gj = _round_wire_EI_GJ(layer)
    n_w, r = layer.n_wires, layer.r_helix_m
    sin_a, cos_a = math.sin(a), math.cos(a)
    if layer.wire_shape == "round":
        # Lateral elastic lower bound, round wire EI_2 = EI_3 = EI ([Eq. 19]).
        p1 = (
            n_w
            * (cos_a / r**2)
            * (gj * sin_a**4 + (5.0 * ei - gj) * sin_a**2 * cos_a**2)
        )
    else:
        # Rectangular/profiled wire (gap G1): EI_2 radial, EI_3 lateral.
        section = layer.wire_section
        e_w = library.get_property(layer.material, "E_Pa")
        nu = library.get_property(layer.material, "nu")
        ei = e_w * section.I_normal
        gj = e_w / (2.0 * (1.0 + nu)) * section.J
        ei3 = e_w * section.I_lateral
        p1 = (
            n_w
            * (cos_a / r**2)
            * (gj * sin_a**4 + (4.0 * ei + ei3 - gj) * sin_a**2 * cos_a**2)
        )
    # Radial buckling on the elastic foundation c ([Eq. 22/23]); no outboard
    # restraint (c = 0) means the radial mode does not apply.
    c, names = _radial_foundation_stiffness(cable, layer)
    p2 = 2.0 * n_w * cos_a * math.sqrt(ei * c) if c > 0.0 else math.inf
    p_crit = min(p1, p2)
    return BirdcageDetail(
        layer=layer.name,
        P1_lateral_N=p1,
        P2_radial_N=p2,
        P_crit_N=p_crit,
        foundation_c_Pa=c,
        restraint_layers=names,
        governing_mode="radial" if p2 < p1 else "lateral",
        provenance=_BIRDCAGE_PROVENANCE,
    )


def birdcage_critical_force(cable: CableData, layer: HelicalLayer) -> float:
    r"""Layer bird-caging critical load ``P_crit = min(P1, P2)`` [N].

    Thin wrapper over :func:`birdcage_buckling_detail` returning only
    ``P_crit`` ([SaevikThorsen2017] Eqs. 19/22/36). Returns ``+inf`` for a zero
    lay angle. When the layer has no outboard radial restraint (a bare strand),
    a runtime warning is emitted and only the lateral bound ``P1`` applies.
    """
    detail = birdcage_buckling_detail(cable, layer)
    if not detail.restraint_layers and math.isfinite(detail.P_crit_N):
        warnings.warn(
            f"armour layer {layer.name!r} has no outboard annular restraint; "
            "radial bird-caging (Eq. 22) does not apply — using the lateral "
            "bound P1 (Eq. 19) only.",
            stacklevel=2,
        )
    return detail.P_crit_N


def _wire_yield_curvature_closed_form(
    cable: CableData,
    T_N: float,
    *,
    mu: float | None = None,
    sigma_T_Pa: float = 0.0,
) -> float:
    r"""Legacy algebraic scale used only to bracket the common-stress root.

    This expression assumes fully mobilised friction and is not returned as a
    capacity result. The public solver below evaluates the incremental
    slip-front stress. At zero tension, contact and thermal stress this scale
    reduces to the analytical local-bending yield curvature.
    """
    layers = _armour_layers(cable)
    if not layers:
        raise CableInputError(missing_armour_message(cable))
    res = _contact_forces(cable, T_N)
    kappas: list[float] = []
    for lyr in layers:
        assert lyr.n_wires is not None and lyr.lay_angle_rad is not None
        sigma_y = library.get_property(lyr.material, "sigma_y_Pa")
        a_w = _wire_area(lyr)
        cos_a = math.cos(lyr.lay_angle_rad)
        # Tension axial stress in the wire (signed): T_layer / (n_w A_w cos α).
        n_aw_cos = lyr.n_wires * a_w * cos_a
        sigma_axial = res.layer_tension_N[lyr.name] / n_aw_cos
        # Saturated Coulomb friction stress μ F_n / A_w ([Saevik2011] §5.2).
        mu_eff = (
            mu if mu is not None else library.get_property(lyr.material, "mu_on_hdpe")
        )
        sigma_f = mu_eff * res.Fn_per_wire_N[lyr.name] / a_w
        # Retained only as a positive root-bracketing scale. The public solver
        # adds signed thermal stress directly and does not apply Goodman here.
        sigma_uts = library.get_property(lyr.material, "sigma_UTS_Pa")
        sigma_y_eff = sigma_y * (1.0 - sigma_T_Pa / sigma_uts)
        sigma_B_allow = sigma_y_eff - abs(sigma_axial) - sigma_f
        slope = _local_bending_slope(lyr)  # ∂σ_B/∂κ
        kappas.append(max(sigma_B_allow, 0.0) / slope if slope > 0 else math.inf)
    return min(kappas)


def wire_yield_curvature(
    cable: CableData,
    T_N: float,
    *,
    mu: float | None = None,
    sigma_T_Pa: float = 0.0,
    contact_model: str | None = None,
    section_state: SectionState | None = None,
    phi_bend: float = 0.0,
    usage_limit: float = 1.0,
) -> float:
    r"""Curvature at first armour-wire yield using the common stress path.

    Every physical wire retains its axial, local-bending and slip-front
    friction stress. The signed thermal stress is added directly. The smallest
    positive root of ``max(abs(sigma_total) / sigma_y) = usage_limit`` is
    returned (``usage_limit = 1`` is first yield; a design utilisation factor
    η < 1 gives the allowable curvature). The zero-contact, zero-tension limit
    reproduces the local-bending closed form.
    """
    if not math.isfinite(usage_limit) or not 0.0 < usage_limit <= 1.0:
        raise CableInputError("usage_limit must lie in (0, 1]")
    maximum_usage, bracket_scale = _wire_usage(
        cable,
        T_N,
        mu=mu,
        sigma_T_Pa=sigma_T_Pa,
        contact_model=contact_model,
        section_state=section_state,
        phi_bend=phi_bend,
    )
    return _usage_root(maximum_usage, bracket_scale, usage_limit)


def wire_yield_curvature_levels(
    cable: CableData,
    T_N: float,
    levels: NDArray[np.float64] | list[float] | tuple[float, ...],
    *,
    mu: float | None = None,
    sigma_T_Pa: float = 0.0,
    contact_model: str | None = None,
) -> NDArray[np.float64]:
    """Curvatures at which the largest wire usage reaches each level.

    One solution of the section per tension, then the root of
    :func:`wire_yield_curvature` for every usage level in (0, 1] (the
    utilisation-level curves of [Skeie2012] Fig. 11). Returns zeros where
    the usage at zero curvature already reaches the level and ``inf`` where
    it is never reached.
    """
    values = np.asarray(levels, dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise CableInputError("usage levels must be finite and positive")
    if np.any(values > 1.0):
        raise CableInputError("usage levels must lie in (0, 1]")
    maximum_usage, bracket_scale = _wire_usage(
        cable,
        T_N,
        mu=mu,
        sigma_T_Pa=sigma_T_Pa,
        contact_model=contact_model,
        section_state=None,
        phi_bend=0.0,
    )
    cache: list[float] = []

    def bracket() -> float:
        if not cache:
            cache.append(bracket_scale())
        return cache[0]

    return np.array(
        [_usage_root(maximum_usage, bracket, float(level)) for level in values]
    )


def _usage_root(
    maximum_usage: Callable[[float], float],
    bracket: Callable[[], float],
    usage_limit: float,
) -> float:
    """Smallest positive curvature with ``maximum_usage >= usage_limit``."""
    if maximum_usage(0.0) >= usage_limit:
        return 0.0
    bracket_scale = bracket()
    upper = max(2.0 * bracket_scale, 1.0e-9) if math.isfinite(bracket_scale) else 1.0
    while maximum_usage(upper) < usage_limit and upper < 1.0e3:
        upper *= 2.0
    if maximum_usage(upper) < usage_limit:
        return math.inf
    lower = 0.0
    for _ in range(60):
        middle = 0.5 * (lower + upper)
        if maximum_usage(middle) >= usage_limit:
            upper = middle
        else:
            lower = middle
    return upper


def _wire_usage(
    cable: CableData,
    T_N: float,
    *,
    mu: float | None,
    sigma_T_Pa: float,
    contact_model: str | None,
    section_state: SectionState | None,
    phi_bend: float,
) -> tuple[Callable[[float], float], Callable[[], float]]:
    """Largest wire usage as a function of curvature, and the bracket scale."""
    if section_state is not None:
        section_state.require_cable(cable)
        if (
            T_N != 0
            or sigma_T_Pa != 0
            or (
                contact_model is not None
                and contact_model != section_state.contact_model
            )
        ):
            raise CableSchemaError("supply section state or separate loads, not both")
        contact_model = section_state.contact_model
    else:
        contact_model = contact_model or "bounded"
        if contact_model == "qp":
            if sigma_T_Pa != 0:
                raise CableSchemaError("use a solved section state for thermal loading")
            section_state = solve_section_state(cable, tension_N=T_N)
    if section_state is not None:
        T_N = section_state.tension_N
    if not math.isfinite(phi_bend):
        raise CableSchemaError("bending-plane azimuth must be finite")
    layers = _armour_layers(cable)
    if not layers:
        raise CableInputError(missing_armour_message(cable))
    specs = cable_to_mkappa_specs(
        cable,
        T_N,
        mu=mu,
        contact_model=contact_model,
        section_state=section_state,
    )
    ea_full = ea_fixed(cable)

    def maximum_usage(kappa: float) -> float:
        usage = 0.0
        for layer, spec in zip(layers, specs, strict=True):
            assert layer.d_wire_m is not None
            phi = spec.phi_wires
            cos_phi = np.cos(phi - phi_bend)
            sigma_axial = (
                spec.E_wire_Pa * math.cos(spec.lay_angle_rad) ** 2 * T_N / ea_full
                if ea_full > 0.0
                else 0.0
            )
            if section_state is not None:
                sigma_axial = section_state.layers[layer.name].longitudinal_stress_Pa
            sigma_stick = spec.k_f * spec.r_helix_m * np.abs(cos_phi) * kappa
            cap_hi, cap_lo = wire_slip_stress_bounds(spec, phi_bend)
            sigma_slip = np.where(cos_phi >= 0.0, cap_hi, cap_lo)
            sigma_friction = np.sign(cos_phi) * np.minimum(sigma_stick, sigma_slip)
            sigma_bending = _local_bending_max(
                layer, kappa, phi, spec.E_wire_Pa, spec.lay_angle_rad, phi_bend
            )
            sigma_total = (
                np.abs(sigma_axial + sigma_friction + sigma_T_Pa) + sigma_bending
            )
            sigma_y = library.get_property(layer.material, "sigma_y_Pa")
            usage = max(usage, float(np.max(np.abs(sigma_total))) / sigma_y)
        return usage

    def bracket_scale() -> float:
        return _wire_yield_curvature_closed_form(cable, T_N, mu=mu, sigma_T_Pa=0.0)

    return maximum_usage, bracket_scale


def _conductor_uf(cable: CableData, T_N: float) -> float:
    """Conductor-fracture UF = conductor axial stress / σ_UTS (tension-driven).

    The conductor's axial stress is its net-section stress in the section
    state every stress result uses (:func:`~cablexsec.mechanics.section_state.
    solve_section_state`, the bounded fixed-radius section: equal axial strain
    of every layer, the core lay included), so this check and the net-section
    check of :func:`~cablexsec.analysis.components.compute_component_response`
    at zero curvature are one number. This is a tension limit-state
    (curvature-independent): at ``N = 0`` it is zero, so it never governs the
    pure-bending capacity.
    """
    cond = _conductor_layer(cable)
    if cond is None or T_N == 0.0:
        return 0.0
    sigma_uts = library.get_property(cond.material, "sigma_UTS_Pa")
    state = solve_section_state(cable, tension_N=T_N, contact_model="bounded")
    sigma_axial = abs(state.layers[cond.name].longitudinal_stress_Pa)
    return sigma_axial / sigma_uts


def _birdcage_uf(cable: CableData, T_N: float) -> float:
    """Armour compression-instability UF; zero unless the armour is compressed.

    Delegates to :func:`cablexsec.analysis.compression.compression_screen`
    with lateral slip assumed active (no curvature given) and no sheath hoop
    strength: the larger of the per-layer lateral utilisation and the stack
    radial utilisation (Eq. 35 with ``P_2`` and ``P_4``). Demand and capacity
    are both layer axial forces.
    """
    if not _armour_layers(cable) or T_N >= 0.0:
        return 0.0
    from cablexsec.analysis.compression import compression_screen  # noqa: PLC0415

    return compression_screen(cable, T_N).utilisation


def _crush_uf(cable: CableData, T_N: float, allowable_Pa: float | None) -> float:
    """Polymer-crush screen, or NaN when no project allowable is available."""
    if allowable_Pa is None:
        return math.nan
    res = _contact_forces(cable, abs(T_N))
    p = res.interface_pressure_Pa
    p_max = float(np.max(p)) if p.size else 0.0
    return p_max / allowable_Pa


def compute_capacity(
    cable: CableData,
    tension_N: FloatArray,
    *,
    mu: float | None = None,
    sigma_T_armour_Pa: float = 0.0,
    contact_model: str = "bounded",
    polymer_crush_allowable_Pa: float | None = None,
    polymer_crush_source: str | None = None,
    component_limits: bool = True,
) -> CapacityResult:
    """Capacity envelope ``κ_max(N)`` and governing mechanism over a tension sweep.

    Parameters
    ----------
    cable : CableData
        Fully-specified cable.
    tension_N : numpy.ndarray
        Cable tensions to sweep (signed; negative = axial compression) [N].
    mu : float, optional
        Armour friction coefficient (defaults to the material ``mu_on_hdpe``).
    sigma_T_armour_Pa : float, optional
        Signed armour thermal stress at the operating current (0 =
        isothermal). It is added directly to the mechanical wire stress.
    contact_model : str, optional
        Radial contact engine used by the common wire-stress path.
    polymer_crush_allowable_Pa : float, optional
        Cable- and layer-specific compressive allowable used for the polymer
        contact-pressure screen [Pa]. Omit it when no applicable test- or
        standard-backed value is available; the returned usage factor is then
        NaN and polymer crushing cannot govern.
    polymer_crush_source : str, optional
        Citation or project test identifier for ``polymer_crush_allowable_Pa``.
        It must be supplied together with the allowable.
    component_limits : bool
        Check the limits of the components other than the armour (module
        notes; default). ``False`` gives the armour-only envelope.

    Returns
    -------
    CapacityResult
    """
    from cablexsec.analysis.components import (  # noqa: PLC0415
        component_usage_curve,
    )

    if (polymer_crush_allowable_Pa is None) != (polymer_crush_source is None):
        raise CableInputError(
            "polymer_crush_allowable_Pa and polymer_crush_source must be "
            "supplied together"
        )
    if polymer_crush_allowable_Pa is not None and polymer_crush_allowable_Pa <= 0.0:
        raise CableInputError("polymer_crush_allowable_Pa must be > 0")
    if polymer_crush_source is not None and not polymer_crush_source.strip():
        raise CableInputError("polymer_crush_source must be non-empty")

    tensions = np.asarray(tension_N, dtype=np.float64)
    kappa_max = np.zeros(tensions.size, dtype=np.float64)
    governing: list[str] = []
    uf_bird = np.zeros(tensions.size, dtype=np.float64)
    uf_crush = np.zeros(tensions.size, dtype=np.float64)
    uf_cond = np.zeros(tensions.size, dtype=np.float64)
    k_comp = np.full(tensions.size, math.inf)
    comp_governing: list[tuple[str, str]] = []
    kinematic: set[str] = set()

    for i, n_t in enumerate(tensions):
        uf_bird[i] = _birdcage_uf(cable, float(n_t))
        uf_crush[i] = _crush_uf(cable, float(n_t), polymer_crush_allowable_Pa)
        uf_cond[i] = _conductor_uf(cable, float(n_t))
        # Tension-driven mechanisms gate the whole curvature capacity to zero.
        tension_uf = {
            "bird_caging": uf_bird[i],
            "polymer_crush": uf_crush[i],
            "conductor_fracture": uf_cond[i],
        }
        available_uf = {
            name: value for name, value in tension_uf.items() if math.isfinite(value)
        }
        worst = max(available_uf, key=lambda k: available_uf[k])
        if tension_uf[worst] >= 1.0:
            kappa_max[i] = 0.0
            governing.append(worst)
            comp_governing.append(("", ""))
            continue
        kappa_max[i] = wire_yield_curvature(
            cable,
            float(n_t),
            mu=mu,
            sigma_T_Pa=sigma_T_armour_Pa,
            contact_model=contact_model,
        )
        governing.append("wire_yield")
        if not component_limits:
            comp_governing.append(("", ""))
            continue
        reach = kappa_max[i] if math.isfinite(kappa_max[i]) else 0.0
        curve = component_usage_curve(
            cable,
            float(n_t),
            1.5 * max(reach, 0.2),
            mu=mu,
            contact_model=contact_model,
        )
        kinematic.update(curve.kinematic)
        k_comp[i] = curve.root(1.0)
        comp_governing.append(
            curve.governing(k_comp[i]) if curve.assessed else ("", "")
        )
        if k_comp[i] < kappa_max[i]:
            kappa_max[i] = k_comp[i]
            governing[-1] = "component_limit"

    return CapacityResult(
        tension_N=tensions,
        kappa_max_1_m=kappa_max,
        governing=governing,
        uf_bird_caging=uf_bird,
        uf_polymer_crush=uf_crush,
        uf_conductor=uf_cond,
        unavailable_mechanisms=(
            ("polymer_crush",) if polymer_crush_allowable_Pa is None else ()
        ),
        kappa_component_1_m=k_comp if component_limits else np.zeros(0),
        governing_component=tuple(comp_governing),
        kinematic_layers=tuple(sorted(kinematic)),
        provenance=Provenance(
            method="available-mechanism capacity envelope (max-UF): common "
            "slip-front wire stress to yield, bird-caging, conductor fracture, "
            "the limits of the other components where their materials give "
            "them, and polymer crush "
            "only with a sourced project allowable",
            citations=(
                "[SaevikThorsen2017] bird-caging",
                "[Saevik2011] F_n",
                "[MenardCartraud2023] Eq.14 local bending",
            )
            + ((polymer_crush_source,) if polymer_crush_source is not None else ()),
            gates=("N=0 yield-curvature closed form",),
            limitations=(
                "bird-caging = armour compression screen with lateral slip "
                "assumed active: per-layer lateral lower bound and radial stack "
                "buckling with wire-yield interaction; hoop-capacity failure P3 "
                "excluded (see compression_screen for P3, friction gate and design "
                "factor)",
                (
                    "polymer crush is excluded because no cable- and layer-specific "
                    "compressive allowable was supplied"
                    if polymer_crush_allowable_Pa is None
                    else "polymer crush compares the maximum reduced-chain interface "
                    "pressure with the supplied project allowable; it does not resolve "
                    "the local contact stress field"
                ),
                "the bounded contact chain retains declared shell axial stiffness; "
                "its pressure transfer does not resolve corrugation deformation",
                "thermal stress is added directly to the extreme wire stress; "
                "no fatigue Goodman correction is used for yield",
            ),
        ),
    )
