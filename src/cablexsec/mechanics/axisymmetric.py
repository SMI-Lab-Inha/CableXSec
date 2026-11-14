# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Bounded equilibrium inter-layer contact-pressure chain.

An open axisymmetric contact-force model: it computes the inter-layer contact
pressures and the per-wire normal force ``F_n`` under axial tension, external
hydrostatic pressure and internal pressure.

The chain accumulates capstan pressure jumps from the outer boundary, with a
non-negative least-squares projection. It does not solve radial displacements
or gap compatibility. Non-negative pressure alone is not an impenetrability
condition; opening and radial compliance require the separate concentric
section solver. The Lamé function in this module is a standalone analytical
building block and is not called by the bounded pressure chain.

Building blocks
---------------
* **Capstan transfer** — a helical layer of ``n_w`` wires at lay angle
  ``alpha`` and helix radius ``r_h`` under layer tension ``T`` presses radially
  inward; the per-wire normal force on the supporting interface is
  ``q_n = T tan²(alpha) / (n_w r_h) + P0 s`` per cable-axis length; the pretension
  ``P0`` is a manufacturing constant, never back-calculated).
* **Lamé ring** — standalone thick-wall stress and radial displacement under
  prescribed surface pressures. Hoop stress can carry part of a pressure
  difference; the bounded chain does not calculate this redistribution.

Temperature is not an input to this solver. Thermal expansion is handled only
by the separate one-dimensional thermoelastic response and is not coupled back
into contact pressure here.

References
----------
[Saevik2011] §5.2; standard Lamé thick-wall theory (e.g. Timoshenko,
*Theory of Elasticity*).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cablexsec.cables.schema import CableData, CylindricalLayer, HelicalLayer
from cablexsec.exceptions import CableInputError
from cablexsec.materials import library
from cablexsec.mechanics.stiffness import load_sharing

__all__ = [
    "AxisymmetricResult",
    "LameStresses",
    "capstan_force_per_wire",
    "compute_contact_forces",
    "compute_winding_contact_pressure",
    "lame_thick_wall",
]

FloatArray = NDArray[np.float64]


# --------------------------------------------------------------------------- #
# Building blocks                                                             #
# --------------------------------------------------------------------------- #
def capstan_force_per_wire(
    T_layer_N: float,
    lay_angle_rad: float,
    n_wires: int,
    r_helix_m: float,
    d_wire_m: float = 0.0,
    pretension_Pa: float = 0.0,
) -> float:
    """Per-wire normal line load from the capstan relation [N/m].

    ``q_n = T_layer tan²(alpha) / (n_wires r_helix) + P0 * s``,
    where T_layer is the cable-axis force, not the sum of wire tensions.
    A wire carries N_w = T_layer/(n cos(alpha)); its normal curvature is
    sin²(alpha)/r and ds_wire/dz = 1/cos(alpha). Thus q_n = N_w kappa_w/cos(alpha).
    This also gives the pressure jump in Féret--Bournazel (1987), Eq. (5).
    The pretension ``P0`` is a *pressure jump* [Pa] across this layer acting
    over the per-wire tributary width ``s = 2 pi r_helix / n_wires``; summed
    over the layer it adds exactly ``P0`` to the radial chain at this layer
    and to every interface inside it. Both terms are line loads. A discrete
    crossed-wire Hertz force must be obtained by dividing ``q_n`` by the
    crossing density; it must not be passed directly to a point-contact solve.
    """
    _ = d_wire_m  # retained for API compatibility
    if n_wires < 1 or r_helix_m <= 0.0:
        raise CableInputError("positive wire count and helix radius are required")
    if not all(
        math.isfinite(x) for x in (T_layer_N, lay_angle_rad, r_helix_m, pretension_Pa)
    ):
        raise CableInputError("capstan inputs must be finite")
    if abs(lay_angle_rad) >= math.pi / 2:
        raise CableInputError("lay angle must be strictly between -pi/2 and pi/2")
    capstan = T_layer_N * math.tan(lay_angle_rad) ** 2 / (n_wires * r_helix_m)
    spacing = 2.0 * math.pi * r_helix_m / n_wires
    return capstan + pretension_Pa * spacing


@dataclass(frozen=True)
class LameStresses:
    """Lamé thick-wall ring stresses/displacement at a radius."""

    sigma_r_Pa: float
    sigma_theta_Pa: float
    u_r_m: float


def lame_thick_wall(
    r_m: float,
    r_inner_m: float,
    r_outer_m: float,
    p_inner_Pa: float,
    p_outer_Pa: float,
    E_Pa: float,
    nu: float,
) -> LameStresses:
    """Lamé solution for a thick-wall cylinder under internal/external pressure.

    ``sigma_r = A + B/r^2``, ``sigma_theta = A - B/r^2`` with

        A = (p_i r_i^2 - p_o r_o^2) / (r_o^2 - r_i^2)
        B = (p_i - p_o) r_i^2 r_o^2 / (r_o^2 - r_i^2)

    (pressures taken positive in compression). Plane-stress radial displacement
    ``u_r = (r/E)(sigma_theta - nu sigma_r)``.

    Parameters
    ----------
    r_m : float
        Radius at which to evaluate, ``r_inner <= r <= r_outer`` [m].
    r_inner_m, r_outer_m : float
        Ring bore and outer radius [m].
    p_inner_Pa, p_outer_Pa : float
        Internal and external pressures (positive = compressive) [Pa].
    E_Pa, nu : float
        Young's modulus [Pa] and Poisson's ratio.
    """
    ri2, ro2, r2 = r_inner_m**2, r_outer_m**2, r_m**2
    denom = ro2 - ri2
    # Tension-positive stresses; applied pressures p_i, p_o are compression-positive.
    a = (p_inner_Pa * ri2 - p_outer_Pa * ro2) / denom
    b = (p_inner_Pa - p_outer_Pa) * ri2 * ro2 / denom
    sigma_r = a - b / r2
    sigma_theta = a + b / r2
    u_r = (r_m / E_Pa) * (sigma_theta - nu * sigma_r)  # plane stress
    return LameStresses(sigma_r_Pa=sigma_r, sigma_theta_Pa=sigma_theta, u_r_m=u_r)


# --------------------------------------------------------------------------- #
# Chain solver                                                                #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AxisymmetricResult:
    """Output of the axisymmetric contact-pressure chain.

    Attributes
    ----------
    interface_radii_m : numpy.ndarray
        Radii of the contact interfaces (outer face of each load-bearing
        layer), inner→outer [m].
    interface_pressure_Pa : numpy.ndarray
        Contact pressure at each interface (>= 0) [Pa].
    Fn_per_wire_N : dict[str, float]
        Per-wire normal line load per cable-axis length [N/m]. The historical
        field name is retained for compatibility.
    layer_tension_N : dict[str, float]
        Axial tension carried by each layer (EA-weighted share) [N].
    """

    interface_radii_m: FloatArray
    interface_pressure_Pa: FloatArray
    Fn_per_wire_N: dict[str, float]
    layer_tension_N: dict[str, float]


def _axial_EA(layer: HelicalLayer | CylindricalLayer) -> float:
    """Axial stiffness EA of a layer [N] (metallic area * E)."""
    E = library.get_property(layer.material, "E_Pa") if _has_E(layer.material) else 0.0
    if isinstance(layer, HelicalLayer):
        assert layer.n_wires is not None and layer.d_wire_m is not None
        area = layer.n_wires * math.pi * layer.d_wire_m**2 / 4.0
        a = layer.lay_angle_rad or 0.0
        return E * area * math.cos(a) ** 3  # helical axial projection
    assert layer.r_inner_m is not None and layer.r_outer_m is not None
    area = math.pi * (layer.r_outer_m**2 - layer.r_inner_m**2)
    return E * area


def _has_E(material: str) -> bool:
    try:
        library.get_property(material, "E_Pa")
        return True
    except KeyError:
        return False


def compute_contact_forces(
    cable: CableData,
    T_N: float,
    *,
    p_ext_Pa: float = 0.0,
    p_int_Pa: float = 0.0,
    layer_tension_N: dict[str, float] | None = None,
) -> AxisymmetricResult:
    """Compute inter-layer contact pressures and per-wire normal line loads.

    Parameters
    ----------
    cable : CableData
        A fully-specified cable (geometry not ``None``).
    T_N : float
        Total cable axial tension [N]. Distributed to layers by EA-weighted
        equal-strain load sharing unless ``layer_tension_N`` is given.
    p_ext_Pa, p_int_Pa : float, optional
        External hydrostatic and internal pressures [Pa].
    layer_tension_N : dict[str, float] or None, optional
        Explicit per-layer tensions (overrides the EA-weighted share); useful
        for the single-layer capstan limit (gate A2).

    Returns
    -------
    AxisymmetricResult

    Notes
    -----
    The radial chain accumulates, from the outside in, the inward line load of
    each helical layer plus the external pressure, and solves for non-negative
    interface pressures by their exact non-negative projection. The per-wire
    normal line load of a helical layer is its own capstan transfer plus the share of
    pressure bearing on it from the layers outside. In the single-helical-layer
    / rigid-foundation limit this reduces exactly to the capstan formula (A2).
    """
    helical = list(cable.helical_layers)
    if not helical:
        raise CableInputError("cable has no helical layers")

    # --- axial load share (equal strain, EA-weighted) ---
    if layer_tension_N is None:
        tension = load_sharing(cable, T_N)
    else:
        tension = dict(layer_tension_N)

    # --- helical radial line loads (inward), per unit cable length ---
    # W_i = n_w * F_n_capstan (total inward radial force / cable length at r_h).
    layers_sorted = sorted(helical, key=lambda x: x.r_helix_m or 0.0)  # inner -> outer
    radii = np.array([x.r_helix_m for x in layers_sorted], dtype=np.float64)
    W = np.zeros(len(layers_sorted), dtype=np.float64)
    for i, lyr in enumerate(layers_sorted):
        assert lyr.n_wires is not None and lyr.r_helix_m is not None
        fn_cap = capstan_force_per_wire(
            tension[lyr.name],
            lyr.lay_angle_rad or 0.0,
            lyr.n_wires,
            lyr.r_helix_m,
            d_wire_m=lyr.d_wire_m or 0.0,
            pretension_Pa=lyr.pretension_Pa or 0.0,
        )
        W[i] = lyr.n_wires * fn_cap  # total inward radial force per cable length

    # Each thin helical layer contributes its own pressure jump W_i/(2 pi r_i).
    # Accumulate those pressure jumps, not radial-force magnitudes evaluated
    # at different radii (Féret--Bournazel 1987, Eqs. 5--6).
    n = len(layers_sorted)
    rhs = np.zeros(n, dtype=np.float64)
    for i in range(n):
        rhs[i] = p_ext_Pa + np.sum(W[i:] / (2.0 * math.pi * radii[i:]))
    # Internal pressure relieves the innermost interface.
    rhs[0] = max(rhs[0] - p_int_Pa, 0.0) if p_int_Pa else rhs[0]

    # Non-negative projection only; no displacement/gap compatibility is solved.
    # The projection matrix is the identity: its exact constrained minimiser
    # is componentwise max(rhs, 0). An iterative bounded least-squares solve
    # adds avoidable tolerance-dependent positive pressures at inactive rows.
    p_iface = np.maximum(rhs, 0.0)

    # --- per-wire normal line load: own capstan + pressure from outside ---
    fn: dict[str, float] = {}
    for i, lyr in enumerate(layers_sorted):
        assert lyr.n_wires is not None and lyr.r_helix_m is not None
        fn_cap = capstan_force_per_wire(
            tension[lyr.name],
            lyr.lay_angle_rad or 0.0,
            lyr.n_wires,
            lyr.r_helix_m,
            d_wire_m=lyr.d_wire_m or 0.0,
            pretension_Pa=lyr.pretension_Pa or 0.0,
        )
        # Outward pressure pushing inward on this layer (from layers above).
        p_above = p_iface[i + 1] if i + 1 < n else p_ext_Pa
        area_per_wire = 2.0 * math.pi * lyr.r_helix_m / lyr.n_wires
        fn[lyr.name] = fn_cap + p_above * area_per_wire

    return AxisymmetricResult(
        interface_radii_m=radii,
        interface_pressure_Pa=p_iface,
        Fn_per_wire_N=fn,
        layer_tension_N=tension,
    )


def compute_winding_contact_pressure(
    cable: CableData,
    sigma_install_Pa: float | Mapping[str, float],
    *,
    p_ext_Pa: float = 0.0,
) -> dict[str, float]:
    r"""Inter-layer contact pressure ``P1 + P2`` from armour winding stress.

    Helical armour wires are applied under a controlled back-tension during
    stranding. This helper assumes a retained residual axial stress
    ``sigma_install`` and calculates its radial-pressure contribution. The
    applied manufacturing back-tension need not equal the retained stress
    after assembly. Zero applied cable tension does not imply zero clamping.
    Sheath extrusion and other assembly constraints can also produce normal
    contact forces; they are not inferred by this winding-only helper.
    A supplied residual stress is a measured, assumed or identified input,
    according to its source, not an independently predicted assembly state.

    A residual wire stress maps to an inter-layer contact-pressure increment by
    the capstan / Feret-Bournazel relation ([FeretBournazel1987] Eq. 5):

        Delta_P_i = n_i sigma_i A_i sin(alpha_i) tan(alpha_i) / (2 pi r_i^2)

    which is exactly the per-wire capstan transfer ``W_i = n_i F_n`` divided by
    the interface circumference. The increments accumulate outer→inner through
    the radial chain ([FeretBournazel1987] Eq. 6; :func:`compute_contact_forces`
    with cable-axis forces ``T_i = n_i A_i sigma_install cos(alpha_i)`` and
    zero applied tension). [FeretBournazel1987] Figs. 5-6 show that bending
    friction moment and apparent stiffness depend on this manufacturing contact
    pressure, matching the model's behaviour.

    Each wire is clamped on *both* radial faces, so the friction-setting
    pressure is the sum of the contact pressures below and above the layer
    (``P1`` from the layer inside / the bedding, ``P2`` from the layer outside /
    the external pressure) — this is the ``P1 + P2`` of [Kim2020] Eq. 3.24.

    Any layer ``pretension_Pa`` declared on the cable enters the same chain as
    an additional per-layer pressure jump ``P0``
    (:func:`capstan_force_per_wire`), so the result is the complete
    manufacturing ``P1 + P2`` from both inputs. With all winding stresses zero
    it is the pretension-only ``P1 + P2``.

    Parameters
    ----------
    cable : CableData
        A fully-specified cable (helical-layer geometry not ``None``).
    sigma_install_Pa : float or mapping of layer name to float
        Residual armour winding/installation stress [Pa]. No published value
        exists for a generic cable. The present elastic wire model requires
        a stress within the stated material range; a selected fraction of
        proof stress is not a measured manufacturing value.
        A scalar applies to all helical layers. A mapping declares each layer's
        retained stress separately; omitted layers have zero winding stress.
        Nonstructural helices remain in the physical pressure chain.
    p_ext_Pa : float, optional
        External pressure on the outermost helical layer [Pa] (default 0 for an
        in-air bending test).

    Returns
    -------
    dict[str, float]
        ``P1 + P2`` contact pressure [Pa] for each helical layer, keyed by name.

    References
    ----------
    [FeretBournazel1987] Eqs. (5), (6), Figs. 5-6; [Saevik2011] §5.2 (capstan
    transfer); [Kim2020] Eq. 3.24 (``P1 + P2`` friction-continuity).
    """
    helical = sorted(cable.helical_layers, key=lambda x: x.r_helix_m or 0.0)
    if not helical:
        raise CableInputError("cable has no helical layers")
    if isinstance(sigma_install_Pa, Mapping):
        if set(sigma_install_Pa) - {layer.name for layer in helical}:
            raise CableInputError("winding stress names an unknown helical layer")
        winding = {
            layer.name: sigma_install_Pa.get(layer.name, 0.0) for layer in helical
        }
    else:
        winding = dict.fromkeys((layer.name for layer in helical), sigma_install_Pa)
    if any(not math.isfinite(value) or value < 0 for value in winding.values()):
        raise CableInputError("winding stresses must be finite and non-negative")
    if not math.isfinite(p_ext_Pa) or p_ext_Pa < 0:
        raise CableInputError("external pressure must be finite and non-negative")

    tension: dict[str, float] = {}
    for lyr in helical:
        assert lyr.n_wires is not None and lyr.d_wire_m is not None
        area_w = math.pi * lyr.d_wire_m**2 / 4.0
        # Residual layer tension from the locked-in winding stress.
        tension[lyr.name] = (
            lyr.n_wires
            * area_w
            * winding[lyr.name]
            * math.cos(lyr.lay_angle_rad or 0.0)
        )

    res = compute_contact_forces(cable, 0.0, p_ext_Pa=p_ext_Pa, layer_tension_N=tension)
    p_iface = res.interface_pressure_Pa  # inner-face pressure per layer, inner→outer

    out: dict[str, float] = {}
    n = len(helical)
    for k, lyr in enumerate(helical):
        p_below = float(p_iface[k])
        p_above = float(p_iface[k + 1]) if k + 1 < n else p_ext_Pa
        out[lyr.name] = p_below + p_above  # P1 + P2 (both clamping faces)
    return out
