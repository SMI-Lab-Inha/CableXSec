# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Cross-section stiffness assembly: axial-torsional matrix and EI bounds.

This module assembles the global cross-section stiffness from the per-layer
contributions and supplies the load-sharing used by the contact and M-κ
engines.

Axial-torsional 2x2 (Witz-Tan / Knapp)
--------------------------------------
Each helical layer of ``n_w`` wires (area ``A_w``, modulus ``E_w``, helix
radius ``r_h``, lay angle ``alpha``, lay sign ``s = +/-1``) contributes to the
symmetric axial-twist matrix ``[[K_ee, K_et],[K_et, K_tt]]``:

    K_ee += n_w E_w A_w cos^3(alpha)                       (axial)
    K_et += s n_w E_w A_w r_h cos^2(alpha) sin(alpha)      (axial-twist coupling)
    K_tt += n_w E_w A_w r_h^2 cos(alpha) sin^2(alpha)      (twist)

(cylindrical layers add ``E A`` to ``K_ee`` only; a stiffness-defined layer
adds its datasheet ``E A`` to ``K_ee`` and its ``G J`` to ``K_tt``, the
measured torsion of that component). With both ends rotationally
fixed the twist is constrained, so ``EA_fixed = K_ee``; with one end free the
twist relaxes and the Schur complement applies,
``EA_free = K_ee - K_et^2 / K_tt`` ([WitzTan1992a]).

EI bounds (stuck / slip)
------------------------
* The component-axis plane-section limit adds the parallel-axis terms
  ``1/2 n_w E_w A_w r_h^2 cos^3(alpha)`` per helical layer and the
  offset-core terms to individual bending. A finite-support no-slip value
  applies ``xi`` to the helical parallel-axis terms only. Those stiffnesses
  have different kinematic definitions when ``xi < 1``.
* ``EI_min`` (full slip) = individual bending ONLY: ``n E I`` for wires, cores
  and centred annuli, with NO ``A d^2`` parallel-axis term (those belong to the
  stuck bound). This is the EI_max/EI_min boundary definition.

Yun shear coefficient
---------------------
``xi`` uses a per-wire supporting width ``b = d_w`` with the per-wire axial
rigidity, consistent with the sandwich-beam balance in [Yun2020].
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from cablexsec.cables.schema import (
    CableData,
    CylindricalLayer,
    CylindricalOffsetLayer,
    CylindricalShellLayer,
    HelicalLayer,
    _radial_tol,
    is_serving,
    layer_stiffness,
    uses_cable_materials,
)
from cablexsec.cables.schema import scope_entry_points as _scope_entry_points
from cablexsec.exceptions import CableInputError
from cablexsec.materials import library
from cablexsec.mechanics.core_lay import bending_factor, core_angle
from cablexsec.mechanics.core_strands import (
    axial_area,
    internal_parallel_inertia,
    wire_bending_inertia,
)
from cablexsec.mechanics.section_geometry import (
    filler_area,
    is_interstitial_filler,
    offset_metal_area,
)
from cablexsec.mechanics.shell_bending import (
    shell_axial_stiffness,
    shell_bending_stiffness,
)
from cablexsec.mechanics.tendon_bending import curvature_coefficients
from cablexsec.mechanics.yun_coefficient import (
    series_shear_stiffness,
    shear_modulus,
    yun_xi,
)

__all__ = [
    "AxialTorsional",
    "ei_max_stuck",
    "ei_min_slip",
    "axial_torsional_matrix",
    "axial_stiffness_by_layer",
    "ea_fixed",
    "ea_free",
    "load_sharing",
    "STICK_SUPPORTS",
    "require_stick_support",
    "xi_for_layer",
    "yun_shear_stack",
    "BondedFaceTerms",
    "bonded_face_stack",
    "bonded_face_stiffness",
    "bonded_face_terms",
    "bonded_face_ei",
    "BendingStiffnessContribution",
    "bending_stiffness_by_layer",
]


def _E(material: str) -> float:
    return library.get_property(material, "E_Pa")


def _has_modulus(material: str) -> bool:
    try:
        library.get_property(material, "E_Pa")
        return True
    except KeyError:
        return False


def _wire_area(layer: HelicalLayer) -> float:
    assert layer.d_wire_m is not None
    return math.pi * layer.d_wire_m**2 / 4.0


# --------------------------------------------------------------------------- #
# Axial-torsional matrix (Witz-Tan / Knapp)                                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AxialTorsional:
    """Symmetric 2x2 axial-twist stiffness ``[[K_ee, K_et],[K_et, K_tt]]``."""

    K_ee: float  # axial [N]
    K_et: float  # axial-twist coupling [N·m]
    K_tt: float  # twist [N·m^2]


def _layer_axial_modulus(layer: object) -> float:
    """Axial modulus used for a layer (effective axial for stranded copper)."""
    if isinstance(layer, CylindricalOffsetLayer | CylindricalLayer):
        mat = layer.material
        for key in ("E_eff_axial_Pa", "E_Pa"):
            try:
                return library.get_property(mat, key)
            except KeyError:
                continue
    return _E(layer.material)  # type: ignore[attr-defined]


def _layer_bending_modulus(layer: object) -> float:
    """Bending modulus used for individual layer bending (EI_min).

    Prefers a separately declared effective bending modulus; this is an input
    assumption unless independently characterised for that construction.
    Falls back to axial modulus then ``E_Pa``.
    """
    mat = layer.material  # type: ignore[attr-defined]
    for key in ("E_eff_bending_Pa", "E_eff_axial_Pa", "E_Pa"):
        try:
            return library.get_property(mat, key)
        except KeyError:
            continue
    return 0.0


# Modulus boundary separating structural metals from polymers [Pa].
_METAL_MODULUS_FLOOR = 10.0e9


def axial_torsional_matrix(
    cable: CableData, *, metals_only: bool = False
) -> AxialTorsional:
    """Assemble the fixed-radius extension-only lay matrix ([WitzTan1992a]).

    The coupling and torsional entries here arise from extension of laid
    wires and power cores. They exclude independent Saint-Venant torsion,
    core warping and radial relaxation. In particular, a free-twist result
    from this matrix is not the complete torsional constitutive response.

    With ``metals_only=True`` the polymer cylindrical annuli (XLPE, sheaths,
    bedding, filler) are excluded — a reporting convention for comparing with
    tools that count only the structural metals. In a fixed-end test the
    polymers *do* legitimately carry ``E A epsilon`` if strain-compatible; this
    flag only changes which layers are summed, never applies a knockdown.
    """
    axial = axial_stiffness_by_layer(cable, metals_only=metals_only)
    k_ee = sum(axial.values())
    k_et = k_tt = 0.0
    for lyr in cable.layers:
        if isinstance(lyr, CylindricalOffsetLayer) and lyr.name in axial:
            angle = core_angle(cable, lyr)
            arm = (lyr.r_offset_m or 0.0) * math.tan(angle)
            k_et += axial[lyr.name] * arm
            k_tt += axial[lyr.name] * arm * arm
        if isinstance(lyr, HelicalLayer):
            assert (
                lyr.n_wires is not None
                and lyr.r_helix_m is not None
                and lyr.lay_angle_rad is not None
            )
            e_w = _E(lyr.material)
            a_w = _wire_area(lyr)
            a = lyr.lay_angle_rad
            r = lyr.r_helix_m
            s = 1.0 if lyr.lay_direction == "right_hand" else -1.0
            base = lyr.n_wires * e_w * a_w
            k_et += s * base * r * math.cos(a) ** 2 * math.sin(a)
            k_tt += base * r**2 * math.cos(a) * math.sin(a) ** 2
        block = layer_stiffness(lyr)
        if block is not None and lyr.name in axial:
            k_tt += block.GJ_Nm2
    return AxialTorsional(K_ee=k_ee, K_et=k_et, K_tt=k_tt)


def _annulus_area(layer: CylindricalLayer | CylindricalOffsetLayer) -> float:
    ri = layer.r_inner_m or 0.0
    ro = layer.r_outer_m or 0.0
    area = math.pi * (ro**2 - ri**2)
    n_cores = getattr(layer, "n_cores", 1)
    return area * n_cores


def _core_assembly_radius(cable: CableData) -> float:
    """Outer radius of a single power-core assembly (max offset r_outer) [m]."""
    radii = [
        lyr.r_outer_m
        for lyr in cable.layers
        if isinstance(lyr, CylindricalOffsetLayer) and lyr.r_outer_m
    ]
    return max(radii) if radii else 0.0


def _filler_true_area(layer: CylindricalLayer, cable: CableData) -> float:
    """Area of the declared filler outside the actual core envelopes."""
    return filler_area(layer, cable)


def _axial_EA_contribution(
    layer: CylindricalLayer | CylindricalOffsetLayer, cable: CableData
) -> float:
    """Axial stiffness contribution ``E A`` of a (offset) cylindrical layer [N].

    Loose-layer treatment: a non-structural *offset* layer (loose helical tape
    screen) and a serving carry no axial load; a non-structural *cylindrical*
    layer (interstitial filler) carries axial load over its true area (cores
    excluded). Structural layers use the full annulus area.
    """
    if is_serving(layer):
        return 0.0
    block = layer_stiffness(layer)
    if block is not None:
        return block.EA_N
    if not layer.structural:
        if isinstance(layer, CylindricalLayer):  # interstitial filler
            area = (
                _filler_true_area(layer, cable)
                if is_interstitial_filler(layer, cable)
                else _annulus_area(layer)
            )
            return _layer_axial_modulus(layer) * area
        return 0.0  # loose tape screen
    if isinstance(layer, CylindricalOffsetLayer) and layer.strand_layers:
        return (
            _layer_axial_modulus(layer)
            * layer.n_cores
            * axial_area(layer)
            * math.cos(core_angle(cable, layer)) ** 3
        )
    # Stranded conductor: the Smith2026 modulus is the copper MATERIAL modulus,
    # so apply the strand-scale lay reduction cos^3(alpha_strand) to the real Cu
    # area ([WitzTan1992b] at strand scale; nested-helix path).
    if (
        isinstance(layer, CylindricalOffsetLayer)
        and layer.conductor_area_mm2 is not None
    ):
        a_cu = offset_metal_area(layer)
        # Missing lay means the declared aligned-axis reduction, not permission
        # to replace actual metal area by the outer conductor envelope.
        cos3 = math.cos(math.radians(layer.strand_lay_angle_deg or 0.0)) ** 3
        return (
            _layer_axial_modulus(layer)
            * a_cu
            * cos3
            * layer.n_cores
            * math.cos(core_angle(cable, layer)) ** 3
        )
    if not _has_modulus(layer.material) and not isinstance(
        layer, CylindricalOffsetLayer
    ):
        return 0.0
    projection = (
        math.cos(core_angle(cable, layer)) ** 3
        if isinstance(layer, CylindricalOffsetLayer)
        else 1.0
    )
    return _layer_axial_modulus(layer) * _annulus_area(layer) * projection


def ea_fixed(cable: CableData, *, metals_only: bool = False) -> float:
    """Fixed-end axial stiffness ``EA_fixed = K_ee`` [N] (twist constrained).

    Pass ``metals_only=True`` to exclude the polymer annuli (reporting
    convention; see :func:`axial_torsional_matrix`).
    """
    return axial_torsional_matrix(cable, metals_only=metals_only).K_ee


def ea_free(cable: CableData) -> float:
    """Free-end axial stiffness via the Schur complement [N] ([WitzTan1992a]).

    ``EA_free = K_ee - K_et^2 / K_tt`` (twist free to relax).
    """
    m = axial_torsional_matrix(cable)
    return m.K_ee - (m.K_et**2 / m.K_tt if m.K_tt > 0 else 0.0)


# --------------------------------------------------------------------------- #
# EI bounds                                                                   #
# --------------------------------------------------------------------------- #
def _individual_bending_by_layer(
    cable: CableData, modulus_scale: Mapping[str, float] | None = None
) -> dict[str, float]:
    """Individual bending of each declared component, without parallel axes."""
    contributions = {}
    for lyr in cable.layers:
        ei = 0.0
        if not lyr.structural:
            contributions[lyr.name] = 0.0
            continue
        if isinstance(lyr, HelicalLayer):
            assert lyr.d_wire_m is not None and lyr.n_wires is not None
            if lyr.wire_shape == "round" and lyr.transverse_slip_fraction == 0.0:
                i_wire = math.pi * lyr.d_wire_m**4 / 64.0
                ei += (
                    lyr.n_wires
                    * _E(lyr.material)
                    * i_wire
                    * bending_factor(lyr.lay_angle_rad or 0.0)
                )
            else:
                # Two principal inertias and the declared transverse state
                # (gap G1): lateral and normal curvature coefficients of
                # tendon_bending, averaged over a lay as bending_factor.
                section = lyr.wire_section
                angle = lyr.lay_angle_rad or 0.0
                coeff = curvature_coefficients(angle, lyr.transverse_slip_fraction)
                ei += (
                    lyr.n_wires
                    * _E(lyr.material)
                    * (
                        section.I_lateral * coeff.lateral**2
                        + section.I_normal * coeff.normal**2
                    )
                    / (2 * math.cos(angle))
                )
        elif isinstance(lyr, CylindricalOffsetLayer):
            if lyr.strand_layers:
                ei += (
                    lyr.n_cores
                    * _layer_axial_modulus(lyr)
                    * sum(wire_bending_inertia(ring) for ring in lyr.strand_layers)
                )
            else:
                ro = lyr.r_outer_m or 0.0
                ri = lyr.r_inner_m or 0.0
                i_core = math.pi * (ro**4 - ri**4) / 4.0
                ei += lyr.n_cores * _layer_bending_modulus(lyr) * i_core
            ei *= bending_factor(core_angle(cable, lyr))
        elif isinstance(lyr, CylindricalLayer):
            block = layer_stiffness(lyr)
            if block is not None:
                ei = block.EI_Nm2
            elif not _has_modulus(lyr.material):
                continue
            else:
                ro = lyr.r_outer_m or 0.0
                ri = lyr.r_inner_m or 0.0
                i_ann = math.pi * (ro**4 - ri**4) / 4.0
                ei += _layer_bending_modulus(lyr) * i_ann
        elif isinstance(lyr, CylindricalShellLayer):
            # Use the same effective thin-wall law as shell stress recovery.
            # The corrugation SCF amplifies local stress, not section rigidity.
            ei += shell_bending_stiffness(lyr)
        if modulus_scale is not None:
            ei *= modulus_scale[lyr.name]
        contributions[lyr.name] = ei
    return contributions


def ei_min_slip(
    cable: CableData, *, modulus_scale: Mapping[str, float] | None = None
) -> float:
    """Full-slip bending stiffness: individual bending only, NO parallel axis.

    This does not contain any ``A r_offset^2`` or ``A r_helix^2``
    term — those belong to the stuck bound.
    """
    return sum(_individual_bending_by_layer(cable, modulus_scale).values())


@dataclass(frozen=True)
class BendingStiffnessContribution:
    """Layer contribution [N m²] with shear compliance kept distinct.

    The plane-section value constrains all declared component and internal
    ring axes. Where strand geometry is absent, an effective individual-core
    bending modulus remains an input. Finite-support and armour-law values
    do not include unresolved internal-ring friction.
    """

    individual_Nm2: float
    core_parallel_axis_Nm2: float
    armour_parallel_axis_Nm2: float
    armour_shear_compliant_Nm2: float
    internal_strand_parallel_axis_Nm2: float = 0.0

    @property
    def plane_section_Nm2(self) -> float:
        return (
            self.individual_Nm2
            + self.core_parallel_axis_Nm2
            + self.armour_parallel_axis_Nm2
            + self.internal_strand_parallel_axis_Nm2
        )

    @property
    def shear_compliant_Nm2(self) -> float:
        # This historical reference constrains core axes, but does not impose
        # internal strand stick in a model that only evolves armour friction.
        return (
            self.individual_Nm2
            + self.core_parallel_axis_Nm2
            + self.armour_shear_compliant_Nm2
        )


def bending_stiffness_by_layer(
    cable: CableData, xi_map: Mapping[str, float] | None = None
) -> dict[str, BendingStiffnessContribution]:
    """Assemble individual, offset-core and helical parallel-axis stiffness.

    ``xi_map`` acts only on the helical parallel-axis term. Without it,
    the component-axis plane-section limit is returned. A finite-support
    no-slip stiffness must not be compared with that limit as if they had
    the same kinematic definition.
    """
    individual = _individual_bending_by_layer(cable)
    result = {}
    xi_map = xi_map or {}
    helical_names = {layer.name for layer in cable.helical_layers}
    if set(xi_map) - helical_names:
        raise CableInputError("shear coefficient names an unknown helical layer")
    for layer in cable.layers:
        core = armour = reduced = internal = 0.0
        if isinstance(layer, HelicalLayer):
            assert layer.n_wires and layer.r_helix_m
            assert layer.lay_angle_rad is not None
            xi = xi_map.get(layer.name, 1.0)
            if not math.isfinite(xi) or not 0 <= xi <= 1:
                raise CableInputError(
                    "shear coefficient must be finite and between zero and one"
                )
            armour = (
                (
                    0.5
                    * layer.n_wires
                    * _E(layer.material)
                    * _wire_area(layer)
                    * layer.r_helix_m**2
                    * math.cos(layer.lay_angle_rad) ** 3
                )
                if layer.structural
                else 0.0
            )
            reduced = xi * armour
        elif isinstance(layer, CylindricalOffsetLayer) and layer.structural:
            core = (
                0.5
                * _axial_EA_contribution(layer, cable)
                * (layer.r_offset_m or 0) ** 2
            )
            internal = (
                layer.n_cores
                * _layer_axial_modulus(layer)
                * internal_parallel_inertia(layer)
                * bending_factor(core_angle(cable, layer))
            )
        result[layer.name] = BendingStiffnessContribution(
            individual.get(layer.name, 0.0), core, armour, reduced, internal
        )
    return result


def ei_max_stuck(cable: CableData, xi_map: dict[str, float] | None = None) -> float:
    """Specified constrained-component reference stiffness [N m²].

    With ``xi_map=None``, constrain component and resolved internal-ring
    axes to plane sections. With an explicit map (including an empty map),
    constrain core axes and use the specified armour support, without
    adding internal-ring stick. Neither reference solves an internal-slip
    history. Homogeneous-core definitions retain their effective modulus.
    """
    contributions = bending_stiffness_by_layer(cable, xi_map).values()
    return sum(
        item.plane_section_Nm2 if xi_map is None else item.shear_compliant_Nm2
        for item in contributions
    )


# --------------------------------------------------------------------------- #
# Yun xi with the wire-contact shear width                                    #
# --------------------------------------------------------------------------- #
def yun_shear_stack(
    cable: CableData, layer_name: str
) -> tuple[CylindricalLayer | CylindricalOffsetLayer, ...]:
    """Layers that shear in series under a helical layer ([Yun2020]).

    Walks inward from ``layer_name`` (the YAML order is the radial order,
    checked by the schema) and returns, from the outside in, the concentric
    cylindrical layers with an isotropic modulus and a positive thickness.
    An interstitial filler lies beside the cores and is skipped; a shell
    layer is not counted. The walk stops at the first helical layer inside,
    which is the support, and at the first layer of laid power cores (any
    offset layer other than a single core at zero offset), whose bundle is
    the support. See :func:`xi_for_layer` for the physical basis.
    """
    names = [lyr.name for lyr in cable.layers]
    if layer_name not in names or not isinstance(
        cable.layers[names.index(layer_name)], HelicalLayer
    ):
        raise CableInputError(f"{layer_name!r} is not a helical layer of the cable")
    stack: list[CylindricalLayer | CylindricalOffsetLayer] = []
    for lyr in reversed(cable.layers[: names.index(layer_name)]):
        if isinstance(lyr, HelicalLayer):
            break
        if isinstance(lyr, CylindricalOffsetLayer) and not (
            lyr.n_cores == 1 and lyr.r_offset_m == 0.0
        ):
            break
        if not isinstance(lyr, CylindricalLayer | CylindricalOffsetLayer):
            continue
        if not _has_modulus(lyr.material) or is_interstitial_filler(lyr, cable):
            # A filler lies beside the cores, not between them and the armour:
            # counted as a concentric layer its radial depth would add a shear
            # compliance the section does not have.
            continue
        if (lyr.r_outer_m or 0.0) - (lyr.r_inner_m or 0.0) <= 0.0:
            continue
        stack.append(lyr)
    return tuple(stack)


STICK_SUPPORTS: tuple[str, ...] = ("yun_stack", "annulus")
"""Stick-support models of :func:`xi_for_layer`; ``"yun_stack"`` is the default."""


def require_stick_support(stick_support: object) -> str:
    """Return ``stick_support`` if it names a model of :data:`STICK_SUPPORTS`."""
    if not isinstance(stick_support, str) or stick_support not in STICK_SUPPORTS:
        raise CableInputError(
            f"stick_support must be one of {STICK_SUPPORTS}, got {stick_support!r}"
        )
    return stick_support


def _stack_support(
    cable: CableData,
    layer_name: str,
    stack: tuple[CylindricalLayer | CylindricalOffsetLayer, ...],
) -> str:
    """Inner boundary of the stack of :func:`yun_shear_stack`.

    ``"rigid"`` where the walk stops at a helical layer or at laid cores, or
    where a layer the stack skips (no isotropic modulus, a shell, a filler)
    lies inside it; ``"axis"`` where its innermost layer reaches the cable
    axis; ``"bore"`` only where nothing lies inside the innermost layer (an
    empty inner surface).
    """
    names = [lyr.name for lyr in cable.layers]
    inside = cable.layers[: names.index(layer_name)]
    for lyr in reversed(inside):
        if isinstance(lyr, HelicalLayer):
            return "rigid"
        if isinstance(lyr, CylindricalOffsetLayer) and not (
            lyr.n_cores == 1 and lyr.r_offset_m == 0.0
        ):
            return "rigid"
    innermost = stack[-1]
    r_bore = innermost.r_inner_m or 0.0
    if r_bore <= 0.0:
        return "axis"
    in_stack = {lyr.name for lyr in stack}
    if any(
        lyr.name not in in_stack
        and (getattr(lyr, "r_outer_m", None) or 0.0) <= r_bore + _radial_tol(r_bore)
        for lyr in inside
    ):
        # A skipped body inside the stack: keep the plane-section support of
        # the default model rather than assume a free bore.
        return "rigid"
    return "bore"


def _annulus_support_stiffness(
    cable: CableData,
    layer: HelicalLayer,
    stack: tuple[CylindricalLayer | CylindricalOffsetLayer, ...],
    youngs_modulus_Pa: Mapping[str, float] | None,
) -> float:
    """Per-wire stick-support stiffness of the ``"annulus"`` model [N/m²]."""
    from cablexsec.mechanics.stick_support import (  # noqa: PLC0415
        annulus_mode1_compliance,
        strip_concentration_compliance,
    )

    assert layer.lay_angle_rad is not None and layer.r_helix_m is not None
    assert layer.n_wires is not None
    rows: list[tuple[float, float, float]] = []
    local = 0.0
    for lyr in stack:
        e = (
            youngs_modulus_Pa[lyr.name]
            if youngs_modulus_Pa is not None
            else _E(lyr.material)
        )
        g = shear_modulus(e, library.get_property(lyr.material, "nu"))
        r_in, r_out = lyr.r_inner_m or 0.0, lyr.r_outer_m or 0.0
        rows.append((r_in, r_out, g))
        local += (r_out - r_in) / g
    support = _stack_support(cable, layer.name, stack)
    alpha = abs(layer.lay_angle_rad)
    r_contact = rows[0][1]
    alpha_c = math.atan(math.tan(alpha) * r_contact / layer.r_helix_m)
    spacing = 2.0 * math.pi * r_contact * math.cos(alpha_c) / layer.n_wires
    try:
        c_axial = annulus_mode1_compliance(rows, support)  # type: ignore[arg-type]
        compliance = (
            math.cos(alpha) ** 2 * c_axial + math.sin(alpha) ** 2 * local
        ) / spacing + strip_concentration_compliance(
            spacing,
            min(0.5 * layer.wire_surface_width_m, 0.5 * spacing),
            rows[0][1] - rows[0][0],
            rows[0][2],
        )
    except ValueError as error:
        raise CableInputError(
            f"{layer.name}: the annulus stick support cannot be evaluated: {error}"
        ) from error
    return math.inf if compliance <= 0.0 else 1.0 / compliance


def xi_for_layer(
    cable: CableData,
    layer_name: str,
    *,
    youngs_modulus_Pa: Mapping[str, float] | None = None,
    stick_support: str = "yun_stack",
) -> float:
    """Yun shear coefficient for a helical layer ([Yun2020]).

    ``xi = K_G / (K_G + EA_w sin^2(a) / r_h^2)`` with ``K_G`` the series shear
    stiffness ``1 / sum(t_i / (G_i b))`` of the layers between the wire and
    the surface that supports it (:func:`yun_shear_stack`). Walking inward
    from the helical layer, every concentric cylindrical layer with an
    isotropic modulus and a positive thickness is in the stack; an
    interstitial filler, which lies beside the cores, is not. The walk stops
    at the first helical layer inside, which is the support, and at the laid
    power cores, whose bundle is the support: the layers of an offset core
    are never in the stack. The concentric layers of a single core on the
    cable axis are concentric layers and stay in it.

    Physical basis: a stuck wire follows the plane-section strain of the
    surface it bears on, less the shear lag of the layers between them. A
    steel helical layer inside is such a surface, orders of magnitude stiffer
    than the tapes, and the compliance under it enters its own ``xi``. Laid
    cores bear on the bedding in line contact and bend about their own axes;
    their insulation is loaded by the core's own bending, not by a shear in
    series between the armour and the cable axis (the cores' own stick is the
    laid-core law, :mod:`cablexsec.mechanics.core_friction`).

    The spring and axial rigidity both refer to one wire, as in Yun et al.
    (2020), Eqs. (9)--(12). The supporting width is approximated by the wire
    surface width, not the combined width of every wire. The series stack is
    a support-compliance approximation, not a resolved shear solution of the
    section. ``youngs_modulus_Pa`` (keyed by layer name) replaces the library
    moduli of the wire and of the stack layers, for example at a temperature.

    ``stick_support="annulus"`` (opt-in) keeps the same stack and support but
    replaces the series strips with the exact mode-1 antiplane shear of the
    layered annulus (:mod:`cablexsec.mechanics.stick_support`). The support is
    rigid where the walk stops at a helical layer or at laid cores, free at an
    empty bore and regular at the axis. The traction is spread over the wire
    spacing, plus the concentration of the load on the wire's own width. For a
    thin layer on a rigid support it agrees with the default to within the
    ratio of the wire width to the spacing. For a soft or hollow core it is
    much softer, because the core shears as a beam: a free bore of radius
    ratio rho gives `R (1 + rho^2) / (G (1 - rho^2))` against the strip's
    `t / G`.
    """
    require_stick_support(stick_support)
    layer = cable.get_layer(layer_name)
    assert isinstance(layer, HelicalLayer)
    assert (
        layer.n_wires is not None
        and layer.d_wire_m is not None
        and layer.r_helix_m is not None
        and layer.lay_angle_rad is not None
    )
    if stick_support == "annulus":
        annulus_stack = yun_shear_stack(cable, layer_name)
        if not annulus_stack:
            return 1.0
        e_wire = (
            youngs_modulus_Pa[layer.name]
            if youngs_modulus_Pa is not None
            else _E(layer.material)
        )
        k_support = _annulus_support_stiffness(
            cable, layer, annulus_stack, youngs_modulus_Pa
        )
        return yun_xi(
            k_support,
            e_wire * _wire_area(layer),
            layer.lay_angle_rad,
            layer.r_helix_m,
        )
    b = layer.wire_surface_width_m
    stack: list[tuple[float, float, float]] = []
    for lyr in yun_shear_stack(cable, layer_name):
        g = shear_modulus(
            youngs_modulus_Pa[lyr.name]
            if youngs_modulus_Pa is not None
            else _E(lyr.material),
            library.get_property(lyr.material, "nu"),
        )
        stack.append(((lyr.r_outer_m or 0.0) - (lyr.r_inner_m or 0.0), g, b))
    k_g = series_shear_stiffness(stack)
    e_wire = (
        youngs_modulus_Pa[layer.name]
        if youngs_modulus_Pa is not None
        else _E(layer.material)
    )
    ea_w = e_wire * _wire_area(layer)
    return yun_xi(k_g, ea_w, layer.lay_angle_rad, layer.r_helix_m)


# --------------------------------------------------------------------------- #
# Bonded outer armour face (gap G8)                                           #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BondedFaceTerms:
    """Terms of a helical layer whose outer face is bonded (``bonded_face="outer"``).

    Attributes
    ----------
    stiffness_N_m2 : float
        Per-wire, per-length stiffness ``K_b`` of the bond through the polymer
        outside the wires [N/m²].
    xi_bond : float
        ``K_b / (K_b + EA_w sin^2(a) / r_h^2)``: the elastic share the bond keeps
        in full slip.
    xi_slider : float
        Stick coefficient left to the friction slider: ``xi_tot - xi_bond``,
        with ``xi_tot`` the support and the bond in parallel. The stuck layer
        is then unchanged, and full slip leaves exactly ``xi_bond``.
    ei_bond_Nm2 : float
        ``xi_bond * 1/2 n_w E_w A_w r_h^2 cos^3(a)``: the bond's bending
        stiffness, elastic and history-free, in parallel with the law.
    """

    stiffness_N_m2: float
    xi_bond: float
    xi_slider: float
    ei_bond_Nm2: float


@uses_cable_materials
def bonded_face_stack(
    cable: CableData, layer_name: str
) -> tuple[CylindricalLayer, ...]:
    """Polymer layers outside a helical layer with a bonded outer face (inside out).

    The layer must be the outermost helical layer, and a cylindrical layer
    with an isotropic modulus must touch its wires; every cylindrical layer
    outside it with a modulus and a positive thickness is in the stack, which
    is free of shear stress at the cable surface.
    """
    layer = cable.get_layer(layer_name)
    if not isinstance(layer, HelicalLayer) or layer.r_helix_m is None:
        raise CableInputError(f"{layer_name}: bonded_face needs a helical layer")
    if any(
        (h.r_helix_m or 0.0) > layer.r_helix_m
        for h in cable.helical_layers
        if h.name != layer.name
    ):
        raise CableInputError(
            f"{layer_name}: bonded_face='outer' is implemented for the outermost "
            "helical layer only"
        )
    r_face = layer.r_helix_m + 0.5 * layer.wire_radial_thickness_m
    stack = sorted(
        (
            lyr
            for lyr in cable.layers
            if isinstance(lyr, CylindricalLayer)
            and lyr.r_inner_m is not None
            and lyr.r_outer_m is not None
            and lyr.r_outer_m > lyr.r_inner_m
            and lyr.r_inner_m >= r_face - _radial_tol(r_face)
            and _has_modulus(lyr.material)
        ),
        key=lambda lyr: lyr.r_inner_m or 0.0,
    )
    if not stack or (stack[0].r_inner_m or 0.0) - r_face > _radial_tol(r_face):
        raise CableInputError(
            f"{layer_name}: bonded_face='outer' needs a cylindrical polymer layer "
            "on the wires (no gap)"
        )
    return tuple(stack)


@uses_cable_materials
def bonded_face_stiffness(
    cable: CableData,
    layer_name: str,
    *,
    youngs_modulus_Pa: Mapping[str, float] | None = None,
) -> float:
    """Per-wire bond stiffness ``K_b`` [N/m²] of a bonded outer face.

    The polymer outside the wires as a layered annulus loaded on its inner
    surface and free at the cable surface
    (:func:`~cablexsec.mechanics.stick_support.annulus_mode1_compliance_inner`),
    with the axial and circumferential shares of the wire direction as in the
    ``"annulus"`` stick support, spread over the wire spacing, plus the
    concentration of the load on the wire's own width.
    """
    from cablexsec.mechanics.stick_support import (  # noqa: PLC0415
        annulus_mode1_compliance_inner,
        strip_concentration_compliance,
    )

    layer = cable.get_layer(layer_name)
    assert isinstance(layer, HelicalLayer)
    assert layer.lay_angle_rad is not None and layer.r_helix_m is not None
    assert layer.n_wires is not None
    rows: list[tuple[float, float, float]] = []
    local = 0.0
    for lyr in bonded_face_stack(cable, layer_name):
        e = (
            youngs_modulus_Pa[lyr.name]
            if youngs_modulus_Pa is not None
            else _E(lyr.material)
        )
        g = shear_modulus(e, library.get_property(lyr.material, "nu"))
        r_in, r_out = lyr.r_inner_m or 0.0, lyr.r_outer_m or 0.0
        rows.append((r_in, r_out, g))
        local += (r_out - r_in) / g
    alpha = abs(layer.lay_angle_rad)
    r_contact = rows[0][0]
    alpha_c = math.atan(math.tan(alpha) * r_contact / layer.r_helix_m)
    spacing = 2.0 * math.pi * r_contact * math.cos(alpha_c) / layer.n_wires
    try:
        compliance = (
            math.cos(alpha) ** 2 * annulus_mode1_compliance_inner(rows)
            + math.sin(alpha) ** 2 * local
        ) / spacing + strip_concentration_compliance(
            spacing,
            min(0.5 * layer.wire_surface_width_m, 0.5 * spacing),
            rows[0][1] - rows[0][0],
            rows[0][2],
        )
    except ValueError as error:
        raise CableInputError(
            f"{layer_name}: the bonded-face stiffness cannot be evaluated: {error}"
        ) from error
    return math.inf if compliance <= 0.0 else 1.0 / compliance


@uses_cable_materials
def bonded_face_terms(
    cable: CableData,
    layer_name: str,
    xi_support: float,
    *,
    youngs_modulus_Pa: Mapping[str, float] | None = None,
) -> BondedFaceTerms:
    """Bond terms of a helical layer with a bonded outer face (gap G8).

    The wires slide only on their inner face. The bond is a spring ``K_b``
    (:func:`bonded_face_stiffness`) in parallel with the inner support
    ``K_s = xi_support k2 / (1 - xi_support)``, ``k2 = EA_w sin^2(a)/r_h^2``.
    Stuck, the layer has ``xi_tot = (K_s + K_b) / (K_s + K_b + k2)``
    (``1`` for a rigid support); in full slip the bond keeps
    ``xi_bond = K_b / (K_b + k2)``. The friction slider is given
    ``xi_tot - xi_bond`` and the bond's ``xi_bond`` share of the
    plane-section stiffness is added to the parallel EI of the law.
    """
    layer = cable.get_layer(layer_name)
    assert isinstance(layer, HelicalLayer)
    assert layer.lay_angle_rad is not None and layer.r_helix_m is not None
    assert layer.n_wires is not None
    if not (0.0 < xi_support <= 1.0):
        raise CableInputError(f"{layer_name}: xi_support must lie in (0, 1]")
    k_b = bonded_face_stiffness(cable, layer_name, youngs_modulus_Pa=youngs_modulus_Pa)
    e_wire = (
        youngs_modulus_Pa[layer.name]
        if youngs_modulus_Pa is not None
        else _E(layer.material)
    )
    ea_w = e_wire * _wire_area(layer)
    alpha = layer.lay_angle_rad
    k2 = ea_w * math.sin(alpha) ** 2 / layer.r_helix_m**2
    xi_bond = yun_xi(k_b, ea_w, alpha, layer.r_helix_m)
    if xi_support >= 1.0 or math.isinf(k_b):
        xi_tot = 1.0
    else:
        k_s = xi_support * k2 / (1.0 - xi_support)
        xi_tot = (k_s + k_b) / (k_s + k_b + k2)
    ei_plane = 0.5 * layer.n_wires * ea_w * layer.r_helix_m**2 * math.cos(alpha) ** 3
    return BondedFaceTerms(k_b, xi_bond, xi_tot - xi_bond, xi_bond * ei_plane)


@uses_cable_materials
def bonded_face_ei(
    cable: CableData, *, youngs_modulus_Pa: Mapping[str, float] | None = None
) -> float:
    """Sum of the bonds' bending stiffness ``ei_bond_Nm2`` over the cable [N·m²].

    Zero when no structural helical layer declares ``bonded_face="outer"``.
    The bond's share does not depend on the inner support, so
    ``xi_support = 1`` is used.
    """
    return sum(
        bonded_face_terms(
            cable, lyr.name, 1.0, youngs_modulus_Pa=youngs_modulus_Pa
        ).ei_bond_Nm2
        for lyr in cable.helical_layers
        if lyr.bonded_face == "outer" and lyr.structural
    )


# --------------------------------------------------------------------------- #
# Load sharing                                                                #
# --------------------------------------------------------------------------- #
def axial_stiffness_by_layer(
    cable: CableData, *, metals_only: bool = False
) -> dict[str, float]:
    """Layer axial rigidities [N] for the fixed-radius, fixed-twist model.

    Section stiffness, contact load sharing and component stresses consume
    this same assembly. Shell area and its declared corrugation reduction
    are retained without replacing the shell by an interstitial filler.
    This is not the coupled radial-deformation constitutive calculation.
    """
    ea: dict[str, float] = {}
    for lyr in cable.layers:
        if isinstance(lyr, HelicalLayer):
            assert lyr.lay_angle_rad is not None and lyr.n_wires is not None
            ea[lyr.name] = (
                lyr.n_wires
                * _E(lyr.material)
                * _wire_area(lyr)
                * math.cos(lyr.lay_angle_rad) ** 3
            )
        elif isinstance(lyr, CylindricalLayer | CylindricalOffsetLayer):
            if metals_only and _layer_axial_modulus(lyr) < _METAL_MODULUS_FLOOR:
                continue
            contrib = _axial_EA_contribution(lyr, cable)
            if contrib > 0.0:
                ea[lyr.name] = contrib
        elif isinstance(lyr, CylindricalShellLayer):
            if metals_only and _E(lyr.material) < _METAL_MODULUS_FLOOR:
                continue
            ea[lyr.name] = shell_axial_stiffness(lyr)
    return ea


def load_sharing(cable: CableData, T_N: float) -> dict[str, float]:
    """Equal-strain (EA-weighted) tension distribution across layers [N]."""
    ea = axial_stiffness_by_layer(cable)
    total = sum(ea.values())
    if total <= 0.0:
        return dict.fromkeys(ea, 0.0)
    eps = T_N / total
    return {name: eps * v for name, v in ea.items()}


# Public entry points read the cable's own materials when called directly.
_scope_entry_points(globals(), __all__)
