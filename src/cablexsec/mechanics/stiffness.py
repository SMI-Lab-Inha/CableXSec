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

(cylindrical layers add ``E A`` to ``K_ee`` only). With both ends rotationally
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
    is_serving,
)
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
    "xi_for_layer",
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
            if not _has_modulus(lyr.material):
                continue
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
def xi_for_layer(
    cable: CableData,
    layer_name: str,
    *,
    youngs_modulus_Pa: Mapping[str, float] | None = None,
) -> float:
    """Yun shear coefficient for a helical layer ([Yun2020]).

    The intervening polymer cylindrical layers (radially inside the helical
    layer) shear in series; an interstitial filler, which lies beside the
    cores, is not one of them. The spring and axial rigidity both refer to one
    wire, as in Yun et al. (2020), Eqs. (9)--(12). The supporting width is
    approximated by the wire diameter, not the combined width of every wire.
    The series stack is a support-compliance approximation, not a resolved
    shear solution of an offset three-core bundle.
    """
    layer = cable.get_layer(layer_name)
    assert isinstance(layer, HelicalLayer)
    assert (
        layer.n_wires is not None
        and layer.d_wire_m is not None
        and layer.r_helix_m is not None
        and layer.lay_angle_rad is not None
    )
    b = layer.wire_surface_width_m
    stack: list[tuple[float, float, float]] = []
    for lyr in cable.layers:
        if lyr.name == layer_name:
            break
        if isinstance(lyr, CylindricalLayer | CylindricalOffsetLayer):
            if not _has_modulus(lyr.material):
                continue
            if is_interstitial_filler(lyr, cable):
                # It lies beside the cores, not between them and the armour:
                # counted as a concentric layer its radial depth would add a
                # shear compliance the section does not have.
                continue
            ri = lyr.r_inner_m or 0.0
            ro = lyr.r_outer_m or 0.0
            t = ro - ri
            if t <= 0.0:
                continue
            g = shear_modulus(
                youngs_modulus_Pa[lyr.name]
                if youngs_modulus_Pa is not None
                else _E(lyr.material),
                library.get_property(lyr.material, "nu"),
            )
            stack.append((t, g, b))
    k_g = series_shear_stiffness(stack)
    e_wire = (
        youngs_modulus_Pa[layer.name]
        if youngs_modulus_Pa is not None
        else _E(layer.material)
    )
    ea_w = e_wire * _wire_area(layer)
    return yun_xi(k_g, ea_w, layer.lay_angle_rad, layer.r_helix_m)


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
