# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Translate cable definitions into section and physical-wire model inputs.

Geometry comes from CableData, material properties and fatigue parameters
from their registries, and pressure from the selected bounded, concentric
("qp") or solved radial ("solved") calculation. The solved option takes
face pressures, normal line loads and crossing reactions from one coupled
radial-axial-torsional contact solution that includes the manufacturing
residual wire stress; it is opt-in and the default remains "bounded".
Shared-interface reactions determine crossing forces without counting the
wire's other face. These adapters are shared by the public API,
wire-stress recovery and fatigue calculations.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cablexsec.analysis.fatigue import FatigueLayer
from cablexsec.cables.schema import (
    CableData,
    CylindricalLayer,
    CylindricalOffsetLayer,
    CylindricalShellLayer,
    HelicalLayer,
    Layer,
)
from cablexsec.clamping import require_radial_shrinkage_route, warn_if_unclamped
from cablexsec.exceptions import CableSchemaError
from cablexsec.materials import library
from cablexsec.materials.sn_curves import (
    SWT_LIFE,
    BlockedSNCurveError,
    get_sn_curve,
    get_swt_life,
)
from cablexsec.mechanics.axisymmetric import (
    compute_contact_forces,
    compute_winding_contact_pressure,
)
from cablexsec.mechanics.contact_pressure import solve_three_core
from cablexsec.mechanics.contact_qp import solve_contact_qp
from cablexsec.mechanics.hertz_contact import (
    crossed_cylinder_curvatures,
    layer_crossing_angle_rad,
    layer_crossing_density_per_m,
    reduced_modulus,
    reduced_shear_modulus,
    solve_contact,
)
from cablexsec.mechanics.mkappa import HelicalLayerSpec
from cablexsec.mechanics.papailiou import PapailiouLayer
from cablexsec.mechanics.radial_cable import (
    CableRadialState,
    solve_cable_radial_state,
    solve_cable_radial_states,
)
from cablexsec.mechanics.section_state import SectionState

FloatArray = NDArray[np.float64]

__all__ = [
    "armour_contact_pressure",
    "armour_normal_force",
    "winding_interface_pressure",
    "cable_to_fatigue_layers",
    "cable_to_mkappa_specs",
    "cable_to_papailiou_layers",
    "solved_contact_histories",
    "SolvedContactHistories",
    "three_core_shape_for_layer",
    "mechanical_equivalent",
    "require_fully_specified",
    "structural_armour_layers",
]


_CONTACT_MODELS = {"bounded", "qp", "solved"}


def _solved_state(
    cable: CableData,
    tension_N: float,
    contact_model: str,
    section_state: SectionState | None,
    radial_state: CableRadialState | None,
) -> CableRadialState | None:
    """Return the solved radial state when that model is selected."""
    if radial_state is not None:
        radial_state.require_cable(cable)
        if section_state is not None:
            raise CableSchemaError("supply a section state or a radial state, not both")
        if contact_model != "solved":
            raise CableSchemaError("a radial state requires contact_model='solved'")
        return radial_state
    if contact_model != "solved":
        return None
    if section_state is not None:
        raise CableSchemaError("the solved radial model does not use a section state")
    return solve_cable_radial_state(cable, tension_N)


def _normal_force_map(
    cable: CableData,
    tension_N: float,
    contact_model: str,
    section_state: SectionState | None = None,
    radial_state: CableRadialState | None = None,
) -> dict[str, float]:
    """Per-wire normal line-load map [N/m] from the contact model."""
    solved = _solved_state(cable, tension_N, contact_model, section_state, radial_state)
    if solved is not None:
        return solved.normal_line_load_per_wire()
    if section_state is not None:
        section_state.require_cable(cable)
        if contact_model != section_state.contact_model:
            raise CableSchemaError(
                "contact model conflicts with supplied section state"
            )
        result = (
            section_state.radial
            if section_state.radial is not None
            else section_state.bounded
        )
        return {} if result is None else result.Fn_per_wire_N
    mech = cable
    p_ext = cable.initial_state.external_pressure_Pa
    if contact_model == "bounded":
        require_radial_shrinkage_route(cable, "bounded pressure chain")
        return compute_contact_forces(mech, tension_N, p_ext_Pa=p_ext).Fn_per_wire_N
    if contact_model == "qp":
        return solve_contact_qp(cable, tension_N, p_ext_Pa=p_ext).Fn_per_wire_N
    raise CableSchemaError(
        "contact_model must be 'bounded', 'qp' (concentric coupled equilibrium) "
        "or 'solved' (radial contact solution)."
    )


def armour_normal_force(
    cable: CableData,
    tension_N: float = 0.0,
    *,
    contact_model: str = "bounded",
    radial_state: CableRadialState | None = None,
) -> dict[str, float]:
    """Return the selected model's per-armour-wire normal line load [N/m].

    The historical function name is retained for API compatibility.
    """
    return _normal_force_map(cable, tension_N, contact_model, radial_state=radial_state)


def _bounded_face_pressures(
    cable: CableData, inner_pressures: FloatArray, external_pressure: float
) -> dict[str, tuple[float, float]]:
    """Map both radial faces using the solver's complete helical-layer order.

    The structural flag selects bending/fatigue sites; it must not remove a
    physical pressure-chain row or move another layer onto that row.
    """
    ordered = sorted(cable.helical_layers, key=lambda layer: layer.r_helix_m or 0)
    outer_pressures = np.r_[inner_pressures[1:], external_pressure]
    return {
        layer.name: (float(below), float(above))
        for layer, below, above in zip(
            ordered, inner_pressures, outer_pressures, strict=True
        )
    }


def winding_interface_pressure(cable: CableData) -> FloatArray:
    """Interface pressure [Pa] from the armour winding stresses alone.

    One value under each helical layer, ordered inner to outer as in
    :func:`~cablexsec.mechanics.axisymmetric.compute_contact_forces`: the
    accumulated pressure jumps ``n sigma A sin(a) tan(a) / (2 pi r^2)``
    ([FeretBournazel1987] Eq. 5) of that layer and every helical layer
    outside it. The same jumps enter :func:`armour_crossing_line_loads`.
    """
    ordered = sorted(cable.helical_layers, key=lambda layer: layer.r_helix_m or 0)
    jumps = np.zeros(len(ordered))
    for i, layer in enumerate(ordered):
        if not layer.winding_stress_Pa:
            continue
        assert layer.n_wires and layer.d_wire_m and layer.r_helix_m
        angle = abs(layer.lay_angle_rad or 0)
        jumps[i] = (
            layer.n_wires
            * math.pi
            * layer.d_wire_m**2
            / 4
            * layer.winding_stress_Pa
            * math.sin(angle)
            * math.tan(angle)
            / (2 * math.pi * layer.r_helix_m**2)
        )
    return np.asarray(np.cumsum(jumps[::-1])[::-1], dtype=np.float64)


def armour_contact_pressure(
    cable: CableData,
    tension_N: float = 0.0,
    *,
    contact_model: str = "bounded",
    section_state: SectionState | None = None,
    radial_state: CableRadialState | None = None,
) -> dict[str, float]:
    r"""Friction-setting contact pressure ``P1+P2`` [Pa] per structural armour.

    From the cable's manufacturing contact state, with NO harness-injected value
    — the single source for both the Papailiou dissipation engine and the
    per-wire slip-front friction stress:

    * every layer's ``winding_stress_Pa`` and ``pretension_Pa`` enter ONE
      bounded radial chain (:func:`compute_winding_contact_pressure`, built on
      :func:`compute_contact_forces` at zero applied tension and zero external
      pressure). ``pretension_Pa`` is the per-layer manufacturing pressure
      jump ``P0`` across a helical layer (its own inward line load divided by
      ``2 pi r_helix``); the jumps accumulate from the outside in. The
      friction-setting pressure is the sum of the two face pressures,
      ``P1+P2 = p_inner_face + p_outer_face``. For two equal layers of ``P0``
      this gives ``3 P0`` (inner) and ``P0`` (outer), not ``2 P0`` on each;
    * a documented initial external pressure is propagated through the radial
      equilibrium chain and added to the two-face clamping pressure;
    * the applied-tension increment from the selected contact solver is then
      added on both faces;
    * a layer with neither input, and no pressure from layers outside it,
      has ``0`` (no friction-setting contact).

    The same chain supplies the bounded crossed-wire (Hertz) line load in
    :func:`armour_crossing_line_loads`, so friction and contact use one
    definition of ``pretension_Pa``.

    With ``contact_model="solved"`` (or a supplied ``radial_state``) both
    face pressures come from the coupled radial contact solution
    (:func:`~cablexsec.mechanics.radial_cable.solve_cable_radial_state`),
    in which pretension and winding stress are residual wire stresses and
    nothing is superposed. Face pressures are referred to the helix radius.

    The manufacturing inputs are the layers' ``manufacturing_clamping``
    declarations (or their legacy fields) and the cable's ``initial_state``
    (:mod:`cablexsec.clamping`). Every friction-dependent analysis forms its
    pressure here, so this is where
    :class:`~cablexsec.provenance.MissingClampingStateWarning` is issued when
    no clamping state is declared or a zero is declared as not published.
    """
    mech = cable
    armour = structural_armour_layers(cable)
    if contact_model not in _CONTACT_MODELS:
        raise CableSchemaError("contact_model must be 'bounded', 'qp' or 'solved'.")
    if armour:
        warn_if_unclamped(cable, stacklevel=2)
    solved = _solved_state(cable, tension_N, contact_model, section_state, radial_state)
    if solved is not None:
        values = solved.armour_contact_pressure()
        return {layer.name: max(values[layer.name], 0.0) for layer in armour}
    # Validate and solve even at zero tension. Unsupported offset/corrugated
    # geometry must not appear to work until the first non-zero load arrives.
    if section_state is not None:
        section_state.require_cable(cable)
        if contact_model != section_state.contact_model:
            raise CableSchemaError(
                "contact model conflicts with supplied section state"
            )
    radial = (
        section_state.radial
        if section_state is not None
        else (
            solve_contact_qp(
                cable, tension_N, p_ext_Pa=cable.initial_state.external_pressure_Pa
            )
            if contact_model == "qp"
            else None
        )
    )
    if radial is None:
        require_radial_shrinkage_route(cable, "bounded pressure chain")
    winding = {
        lyr.name: lyr.winding_stress_Pa
        for lyr in cable.helical_layers
        if lyr.winding_stress_Pa
    }
    p1p2: dict[str, float] = {}
    if winding or any(lyr.pretension_Pa for lyr in cable.helical_layers):
        # One convention for both manufacturing inputs: winding stress and
        # pretension are pressure jumps in the same accumulating radial chain
        # (compute_contact_forces), and P1+P2 sums the two face pressures.
        p1p2 = compute_winding_contact_pressure(mech, winding)
    pressure = {lyr.name: p1p2.get(lyr.name, 0.0) for lyr in armour}
    if radial is not None:
        # Explicit additive approximation: declared manufacturing clamping
        # plus mechanically calculated face pressures. The manufactured
        # pressure has not been converted into an assembly strain.
        names = list(radial.section.outer_displacement_m)
        for layer in armour:
            i = names.index(layer.name)
            below = float(radial.contact_pressure_Pa[i - 1]) if i else 0.0
            above = (
                float(radial.contact_pressure_Pa[i])
                if i < len(names) - 1
                else (
                    section_state.external_pressure_Pa
                    if section_state is not None
                    else cable.initial_state.external_pressure_Pa
                )
            )
            pressure[layer.name] += below + above
        return pressure
    if section_state is not None:
        bounded = section_state.bounded
        if bounded is None:
            return pressure
        # The saved state already includes the actual layer tensions and
        # external pressure. Subtract only the prescribed pretension already
        # present in the pressure base above, not a second mechanical solve.
        baseline = compute_contact_forces(cable, 0.0)
        delta = bounded.interface_pressure_Pa - baseline.interface_pressure_Pa
        increments = _bounded_face_pressures(
            cable, delta, section_state.external_pressure_Pa
        )
        for layer in armour:
            below, above = increments[layer.name]
            pressure[layer.name] = max(pressure[layer.name] + below + above, 0.0)
        return pressure
    p_external = cable.initial_state.external_pressure_Pa
    if p_external > 0.0:
        # Isolate the external-pressure increment so layer pretension, if any,
        # is not counted twice. The bounded radial equilibrium gives one
        # pressure immediately inside each helical layer, ordered inner to
        # outer. P1+P2 is the pressure on both radial faces.
        loaded = compute_contact_forces(mech, 0.0, p_ext_Pa=p_external)
        baseline = compute_contact_forces(mech, 0.0)
        delta = loaded.interface_pressure_Pa - baseline.interface_pressure_Pa
        increments = _bounded_face_pressures(cable, delta, p_external)
        for lyr in armour:
            below, above = increments[lyr.name]
            pressure[lyr.name] += below + above
    # Add only the applied-tension increment, on both radial faces. Work with
    # the solver's interface pressures directly rather than converting its
    # historical normal-line-load field through a fictitious contact area.
    if tension_N != 0.0:
        now = compute_contact_forces(mech, tension_N, p_ext_Pa=p_external)
        before = compute_contact_forces(mech, 0.0, p_ext_Pa=p_external)
        delta = now.interface_pressure_Pa - before.interface_pressure_Pa
        increments = _bounded_face_pressures(cable, delta, 0.0)
        for layer in armour:
            below, above = increments[layer.name]
            pressure[layer.name] = max(pressure[layer.name] + below + above, 0.0)
    return pressure


def three_core_shape_for_layer(
    cable: CableData, layer: HelicalLayer
) -> FloatArray | None:
    """Return an exploratory inner-support reaction shape, not armour pressure.

    This historical helper is retained for sensitivity comparisons. Its ring
    is loaded uniformly on the outer face and reacts on a prescribed inner
    foundation. Those two face pressures are different. The returned inner
    shape must not be transferred automatically to armour friction or Hertz
    contact. Foundation stiffnesses are material-based estimates, not a solved
    three-core contact state. Single-core/underspecified layouts return None.
    """
    offsets = [
        x
        for x in cable.layers
        if isinstance(x, CylindricalOffsetLayer)
        and x.n_cores == 3
        and x.r_offset_m is not None
        and x.r_outer_m is not None
    ]
    if not offsets or layer.n_wires is None or layer.r_helix_m is None:
        return None

    assert offsets[0].r_offset_m is not None
    r_offset = float(offsets[0].r_offset_m)
    r_core_outer = max(float(x.r_outer_m or 0.0) for x in offsets)
    if r_offset <= 0.0 or r_core_outer <= 0.0:
        return None
    theta = 2.0 * math.asin(min(r_core_outer / r_offset, 0.999))

    annuli = [
        x
        for x in cable.layers
        if isinstance(x, CylindricalLayer)
        and x.r_inner_m is not None
        and x.r_outer_m is not None
    ]
    filler = next((x for x in annuli if "filler" in x.name.lower()), None)
    bedding = next((x for x in annuli if "bedding" in x.name.lower()), None)
    if filler is None or bedding is None:
        return None
    assert filler.r_inner_m is not None and filler.r_outer_m is not None
    assert bedding.r_inner_m is not None and bedding.r_outer_m is not None

    try:
        e_core_vals: list[float] = []
        for x in offsets:
            try:
                e_core_vals.append(library.get_property(x.material, "E_Pa"))
            except KeyError:
                e_core_vals.append(library.get_property(x.material, "E_eff_axial_Pa"))
        e_core = max(e_core_vals)
        e_filler = library.get_property(filler.material, "E_Pa")
        e_bedding = library.get_property(bedding.material, "E_Pa")
    except KeyError:
        return None

    filler_t = max(float(filler.r_outer_m - filler.r_inner_m), 1.0e-6)
    bedding_t = max(float(bedding.r_outer_m - bedding.r_inner_m), 1.0e-6)
    r_bedding = 0.5 * float(bedding.r_inner_m + bedding.r_outer_m)
    # Winkler-like radial foundation estimates from the actual cable materials.
    k_core = max(e_core / max(r_core_outer, 1.0e-6), 1.0)
    k_filler = max(e_filler / filler_t, 1.0)
    # Ring bending stiffness per unit axial length for the bedding annulus.
    EI_ring = max(e_bedding * bedding_t**3 / 12.0, 1.0e-12)
    res = solve_three_core(
        1.0,
        k_core=k_core,
        k_filler=k_filler,
        theta_core_rad=theta,
        EI_ring=EI_ring,
        r_bedding_m=r_bedding,
        n_wires=layer.n_wires,
        phi_cores=tuple(2.0 * math.pi * k / 3.0 for k in range(3)),
    )
    return np.asarray(res.shape_g, dtype=np.float64)


def _above_core_mask(cable: CableData, phi: FloatArray) -> NDArray[np.bool_] | None:
    """Classify radial rays using core geometry, independently of pressure."""
    cores = [
        x
        for x in cable.layers
        if isinstance(x, CylindricalOffsetLayer)
        and x.n_cores == 3
        and x.r_offset_m is not None
        and x.r_outer_m is not None
    ]
    if not cores:
        return None
    mask = np.zeros(phi.shape, dtype=bool)
    for core in cores:
        assert core.r_outer_m is not None and core.r_offset_m is not None
        if core.r_offset_m <= core.r_outer_m:
            mask[:] = True
            continue
        half_width = math.asin(core.r_outer_m / core.r_offset_m)
        for centre in (0.0, 2 * math.pi / 3, 4 * math.pi / 3):
            distance = abs((phi - centre + math.pi) % (2 * math.pi) - math.pi)
            mask |= distance <= half_width
    return mask


def cable_to_papailiou_layers(
    cable: CableData,
    *,
    tension_N: float = 0.0,
    mu: float | None = None,
    contact_model: str = "bounded",
    radial_state: CableRadialState | None = None,
) -> list[PapailiouLayer]:
    r"""Build Papailiou slip-front layers for a cable's structural armour.

    The neutral-axis slip-front calculation uses the pressure state of each
    structural armour layer. Experimental accuracy is quantified separately
    in VALIDATION.md; selecting this law does not establish agreement
    with measured loop work. Per-layer ``contact_pressure_Pa`` (``P1+P2``) comes from
    :func:`armour_contact_pressure`. Returns an empty list if no structural
    armour carries a contact input.
    """
    require_fully_specified(cable, "cable_to_papailiou_layers")
    armour = structural_armour_layers(cable)
    p1p2 = armour_contact_pressure(
        cable, tension_N, contact_model=contact_model, radial_state=radial_state
    )

    layers: list[PapailiouLayer] = []
    for lyr in armour:
        assert (
            lyr.n_wires is not None
            and lyr.d_wire_m is not None
            and lyr.r_helix_m is not None
            and lyr.lay_angle_rad is not None
        )
        p = p1p2[lyr.name]
        if p <= 0.0:
            continue
        mu_eff = (
            mu if mu is not None else library.get_property(lyr.material, "mu_on_hdpe")
        )
        layers.append(
            PapailiouLayer(
                n_wires=lyr.n_wires,
                r_layer_m=lyr.r_helix_m,
                lay_angle_rad=lyr.lay_angle_rad,
                E_wire_Pa=library.get_property(lyr.material, "E_Pa"),
                A_wire_m2=math.pi * lyr.d_wire_m**2 / 4.0
                if lyr.wire_shape == "round"
                else lyr.wire_section.area,
                d_wire_m=lyr.d_wire_m,
                mu=mu_eff,
                contact_pressure_Pa=p,
            )
        )
    return layers


def mechanical_equivalent(cable: CableData) -> CableData:
    """Swap each ``cylindrical_shell`` for its equivalent annulus, else unchanged.

    The bounded section approximation uses inner and outer radii. A smooth
    shell of mean radius ``r`` and wall ``t`` has the annular area between
    ``r - t/2`` and ``r + t/2``. This conversion is not a constitutive model
    for corrugation or radial compaction; the coupled contact adapter rejects
    unsupported corrugated radial geometry. Returns the cable unchanged when
    it has no metallic sheath.
    """
    if not any(isinstance(x, CylindricalShellLayer) for x in cable.layers):
        return cable
    layers: list[Layer] = []
    for lyr in cable.layers:
        if isinstance(lyr, CylindricalShellLayer):
            assert lyr.r_mid_m is not None and lyr.t_m is not None
            layers.append(
                CylindricalLayer(
                    name=lyr.name,
                    material=lyr.material,
                    r_inner_m=lyr.r_mid_m - lyr.t_m / 2.0,
                    r_outer_m=lyr.r_mid_m + lyr.t_m / 2.0,
                    structural=False,
                    source=lyr.source,
                )
            )
        else:
            layers.append(lyr)
    # Replace the layers only: every other block of the definition (materials,
    # water ingress, environment, initial state, core lay) is kept.
    return cable.model_copy(update={"layers": tuple(layers)})


def require_fully_specified(cable: CableData, fn_name: str) -> None:
    """Raise :class:`CableSchemaError` if withheld/required geometry is null."""
    missing = cable.missing_fields()
    if missing:
        raise CableSchemaError(
            f"{fn_name}: cable {cable.cable.name!r} is not fully specified — "
            f"missing {', '.join(missing[:6])}"
            f"{' …' if len(missing) > 6 else ''}. Supply the geometry override."
        )


def structural_armour_layers(cable: CableData) -> list[HelicalLayer]:
    """Fully-specified, structural helical layers with a wire modulus, in order."""
    out: list[HelicalLayer] = []
    for lyr in cable.helical_layers:
        if not lyr.structural:
            continue
        if None in (lyr.n_wires, lyr.d_wire_m, lyr.r_helix_m, lyr.lay_angle_rad):
            continue
        try:
            library.get_property(lyr.material, "E_Pa")
        except KeyError:
            continue
        out.append(lyr)
    return out


def cable_to_mkappa_specs(
    cable: CableData,
    tension_N: float = 0.0,
    *,
    mu: float | None = None,
    contact_model: str = "bounded",
    section_state: SectionState | None = None,
    radial_state: CableRadialState | None = None,
) -> list[HelicalLayerSpec]:
    r"""Build frozen M-κ specs for a cable's structural armour layers.

    Each :class:`HelicalLayerSpec` field comes from a documented source: geometry
    from the YAML, ``E`` from the material library, ξ from
    :func:`~cablexsec.mechanics.stiffness.xi_for_layer`, and the per-wire normal
    contact force ``F_n`` from :func:`compute_contact_forces` at ``tension_N``.
    Mean face pressure is uniform over each population. Offset-core geometry
    alone does not determine its angular distribution. In particular, the
    exploratory inner-foundation reaction is not used as an outer-face load.
    Independently prescribed distributions can be supplied to the lower-level
    HelicalLayerSpec, with their force balance and physical face specified.

    Raises
    ------
    CableSchemaError
        If the cable has no fully-specified structural armour layer.
    """
    from cablexsec.mechanics.stiffness import xi_for_layer  # noqa: PLC0415

    require_fully_specified(cable, "cable_to_mkappa_specs")
    if contact_model == "solved" and radial_state is None:
        radial_state = solve_cable_radial_state(cable, tension_N)
    fn_map = _normal_force_map(
        cable, tension_N, contact_model, section_state, radial_state
    )
    pressure_map = armour_contact_pressure(
        cable,
        tension_N,
        contact_model=contact_model,
        section_state=section_state,
        radial_state=radial_state,
    )
    specs: list[HelicalLayerSpec] = []
    for lyr in structural_armour_layers(cable):
        assert (
            lyr.n_wires is not None
            and lyr.d_wire_m is not None
            and lyr.r_helix_m is not None
            and lyr.lay_angle_rad is not None
        )
        mu_eff = (
            mu if mu is not None else library.get_property(lyr.material, "mu_on_hdpe")
        )
        specs.append(
            HelicalLayerSpec(
                n_wires=lyr.n_wires,
                r_helix_m=lyr.r_helix_m,
                lay_angle_rad=lyr.lay_angle_rad,
                E_wire_Pa=section_state.youngs_modulus_Pa[lyr.name]
                if section_state is not None
                else library.get_property(lyr.material, "E_Pa"),
                A_wire_m2=math.pi * lyr.d_wire_m**2 / 4.0
                if lyr.wire_shape == "round"
                else lyr.wire_section.area,
                xi=xi_for_layer(
                    cable,
                    lyr.name,
                    youngs_modulus_Pa=section_state.youngs_modulus_Pa
                    if section_state is not None
                    else None,
                ),
                mu=mu_eff,
                F_n_axi_N=fn_map[lyr.name],
                shape_g=None,
                contact_pressure_Pa=pressure_map[lyr.name],
                normal_load_is_line=True,
                section=None if lyr.wire_shape == "round" else lyr.wire_section,
            )
        )
    if not specs:
        raise CableSchemaError(
            "cable has no fully-specified structural armour layer with a wire "
            "modulus (check helical layers' geometry, material E_Pa, structural)."
        )
    return specs


def armour_crossing_line_loads(
    cable: CableData,
    tension_N: float = 0.0,
    *,
    contact_model: str = "bounded",
    section_state: SectionState | None = None,
    radial_state: CableRadialState | None = None,
) -> dict[str, tuple[HelicalLayer, float]]:
    """Direct crossing partner and shared-interface line load per wire [N/m].

    A single interface reaction is divided by each population's wire count.
    Dividing once more by that population's crossing density gives the SAME
    point force from either side. Reactions on another radial face are not
    included. Bedding-separated, non-touching and same-lay pairs have no
    direct crossed-cylinder contact. A structural fatigue site touching two
    opposed partners requires separate interface histories and is rejected by
    this adapter. Nonstructural helical layers can still be physical partners;
    only structural sites receive entries in the returned mapping. With the
    solved radial model the shared reaction is that solution's interface
    line force, which already contains the manufacturing residual state.
    """
    if contact_model not in _CONTACT_MODELS:
        raise CableSchemaError("contact_model must be 'bounded', 'qp' or 'solved'")
    if section_state is not None:
        section_state.require_cable(cable)
        if contact_model != section_state.contact_model:
            raise CableSchemaError(
                "contact model conflicts with supplied section state"
            )
    if not math.isfinite(tension_N):
        raise CableSchemaError("contact tension must be finite")
    require_fully_specified(cable, "armour_crossing_line_loads")
    active_names = {layer.name for layer in structural_armour_layers(cable)}
    helices = sorted(cable.helical_layers, key=lambda x: x.r_helix_m or 0)
    pairs: list[tuple[HelicalLayer, HelicalLayer, float]] = []
    for inner, outer in zip(helices[:-1], helices[1:], strict=True):
        assert inner.r_helix_m and outer.r_helix_m
        assert inner.d_wire_m and outer.d_wire_m
        radius = inner.r_helix_m + inner.wire_radial_thickness_m / 2
        gap = outer.r_helix_m - outer.wire_radial_thickness_m / 2 - radius
        if (
            abs(gap) <= 1e-9
            and inner.lay_direction != outer.lay_direction
            and (inner.name in active_names or outer.name in active_names)
        ):
            pairs.append((inner, outer, radius))
    participants = [
        layer.name for pair in pairs for layer in pair[:2] if layer.name in active_names
    ]
    if len(set(participants)) != len(participants):
        raise CableSchemaError(
            "multiple direct crossing interfaces require separate contact histories"
        )
    if not pairs:
        return {}
    solved = _solved_state(cable, tension_N, contact_model, section_state, radial_state)
    if solved is not None:
        solved_result: dict[str, tuple[HelicalLayer, float]] = {}
        for inner, outer, _radius in pairs:
            assert inner.n_wires and outer.n_wires
            reaction = solved.interface_line_force(inner.name, outer.name)
            if inner.name in active_names:
                solved_result[inner.name] = (outer, reaction / inner.n_wires)
            if outer.name in active_names:
                solved_result[outer.name] = (inner, reaction / outer.n_wires)
        return solved_result
    mechanical = cable
    if section_state is not None:
        section_state.require_cable(cable)
    p_ext = (
        section_state.external_pressure_Pa
        if section_state is not None
        else cable.initial_state.external_pressure_Pa
    )
    result: dict[str, tuple[HelicalLayer, float]] = {}
    radial = (
        section_state.radial
        if section_state is not None
        else (
            solve_contact_qp(cable, tension_N, p_ext_Pa=p_ext)
            if contact_model == "qp"
            else None
        )
    )
    if contact_model not in {"bounded", "qp"}:
        raise CableSchemaError("contact_model must be 'bounded' or 'qp'")
    if radial is None:
        require_radial_shrinkage_route(cable, "bounded pressure chain")
    bounded = (
        section_state.bounded
        if section_state is not None
        else (
            compute_contact_forces(mechanical, tension_N, p_ext_Pa=p_ext)
            if radial is None
            else None
        )
    )
    all_helices = sorted(mechanical.helical_layers, key=lambda x: x.r_helix_m or 0)
    # Winding pressure is an explicitly superposed manufacturing assumption.
    winding_jumps = {}
    for layer in all_helices:
        assert layer.n_wires and layer.d_wire_m and layer.r_helix_m
        angle = abs(layer.lay_angle_rad or 0)
        winding_jumps[layer.name] = (
            layer.n_wires
            * math.pi
            * layer.d_wire_m**2
            / 4
            * (layer.winding_stress_Pa or 0)
            * math.sin(angle)
            * math.tan(angle)
            / (2 * math.pi * layer.r_helix_m**2)
        )
    for inner, outer, radius in pairs:
        assert inner.n_wires and outer.n_wires
        if radial is not None:
            names = list(radial.section.outer_displacement_m)
            index = names.index(inner.name)
            if names[index + 1] != outer.name:
                raise CableSchemaError(
                    "an intervening layer excludes direct wire contact"
                )
            reaction = float(radial.section.contact_line_force_N_per_m[index])
            # The coupled solve excludes pretension. Superpose the bounded
            # chain's pretension pressure at this interface: the accumulated
            # jumps of every helix at or outside the outer partner.
            reaction += (
                2
                * math.pi
                * radius
                * sum(
                    x.pretension_Pa or 0
                    for x in all_helices
                    if (x.r_helix_m or 0) >= (outer.r_helix_m or 0)
                )
            )
        else:
            assert bounded is not None
            index = [x.name for x in all_helices].index(outer.name)
            reaction = (
                2 * math.pi * radius * float(bounded.interface_pressure_Pa[index])
            )
        winding_pressure = sum(
            winding_jumps[x.name]
            for x in all_helices
            if (x.r_helix_m or 0) >= (outer.r_helix_m or 0)
        )
        reaction += 2 * math.pi * radius * winding_pressure
        if inner.name in active_names:
            result[inner.name] = (outer, reaction / inner.n_wires)
        if outer.name in active_names:
            result[outer.name] = (inner, reaction / outer.n_wires)
    return result


def _fretting_contact(
    layer: HelicalLayer,
    partner: HelicalLayer | None,
    q_n_N_per_m: float,
    *,
    youngs_modulus_Pa: Mapping[str, float] | None = None,
) -> tuple[float, float, float]:
    """Hertz tangential stiffness ``K_t`` [N/m] and peak pressure ``p_0`` [Pa].

    Uses the oblique-cylinder crossing with the opposite-laid partner armour
    layer. The axisymmetric solver supplies a line load ``q_n`` [N/m], which is
    divided by the helical crossing density before the point-contact Hertz
    solve. Returns ``(K_t, p_0, lambda)``; all are zero when no crossing exists.
    """
    if partner is None or q_n_N_per_m <= 0.0:
        return 0.0, 0.0, 0.0
    assert (
        layer.lay_angle_rad is not None
        and partner.lay_angle_rad is not None
        and layer.d_wire_m is not None
        and layer.r_helix_m is not None
        and partner.r_helix_m is not None
        and partner.n_wires is not None
    )
    assert partner.d_wire_m is not None
    # Opposed lay alone does not imply metal-to-metal contact. A bedding
    # layer or geometrical clearance between the wire surfaces excludes the
    # direct crossed-cylinder model.
    surface_gap = (
        abs(layer.r_helix_m - partner.r_helix_m)
        - (layer.wire_radial_thickness_m + partner.wire_radial_thickness_m) / 2
    )
    if abs(surface_gap) > 1e-9 or layer.lay_direction == partner.lay_direction:
        return 0.0, 0.0, 0.0
    sign_layer = 1.0 if layer.lay_direction == "right_hand" else -1.0
    sign_partner = 1.0 if partner.lay_direction == "right_hand" else -1.0
    density = layer_crossing_density_per_m(
        partner.n_wires,
        layer.r_helix_m,
        layer.lay_angle_rad,
        sign_layer,
        partner.r_helix_m,
        partner.lay_angle_rad,
        sign_partner,
    )
    if density <= 0.0:
        return 0.0, 0.0, 0.0
    force_per_crossing_N = q_n_N_per_m / density
    theta = layer_crossing_angle_rad(layer.lay_angle_rad, partner.lay_angle_rad)
    if layer.wire_shape != "round" or partner.wire_shape != "round":
        # Flat strips cross face on face over a rhombus of area
        # b1 b2 / sin(theta): mean face pressure, no Hertz peak and no Hertz
        # tangential stiffness (the slip-amplitude fretting branch is off).
        sine = abs(math.sin(theta))
        if sine <= 1e-12:
            return 0.0, 0.0, density
        patch = layer.wire_surface_width_m * partner.wire_surface_width_m / sine
        return 0.0, force_per_crossing_N / patch, density
    a_curv, b_curv = crossed_cylinder_curvatures(
        layer.d_wire_m, theta, partner_diameter_m=partner.d_wire_m
    )
    e_w = (
        youngs_modulus_Pa[layer.name]
        if youngs_modulus_Pa is not None
        else library.get_property(layer.material, "E_Pa")
    )
    nu = library.get_property(layer.material, "nu")
    e_partner = (
        youngs_modulus_Pa[partner.name]
        if youngs_modulus_Pa is not None
        else library.get_property(partner.material, "E_Pa")
    )
    nu_partner = library.get_property(partner.material, "nu")
    e_star = reduced_modulus(e_w, nu, e_partner, nu_partner)
    g_star = reduced_shear_modulus(e_w, nu, e_partner, nu_partner)
    hz = solve_contact(force_per_crossing_N, a_curv, b_curv, e_star, g_star)
    return hz.K_t_N_per_m, hz.p0_Pa, density


#: Exact solved states over a tension history; longer histories interpolate.
SOLVED_GRID_POINTS = 33


@dataclass(frozen=True)
class SolvedContactHistories:
    """Solved radial contact quantities over a tension history (per sample).

    ``pressure_Pa`` is ``P1+P2`` and ``crossing_line_load_N_per_m`` the
    crossed-wire line load per structural armour layer; ``wire_stress_Pa`` is
    the solved uniform wire stress of every helical layer.
    """

    pressure_Pa: dict[str, FloatArray]
    crossing_line_load_N_per_m: dict[str, FloatArray]
    wire_stress_Pa: dict[str, FloatArray]


def solved_contact_histories(
    cable: CableData,
    tension_N: FloatArray,
    *,
    crossing_loads: bool = True,
    **radial_inputs: Any,
) -> SolvedContactHistories:
    """Solved radial contact over a tension history.

    The solution is piecewise affine in tension. Up to 33 distinct tensions
    are solved exactly; longer histories are interpolated linearly between
    33 equally spaced exact solutions, which is exact except within one grid
    interval of an active-set change. ``radial_inputs`` (temperatures,
    moduli, reference temperature) are passed to
    :func:`~cablexsec.mechanics.radial_cable.solve_cable_radial_states`; with
    none, every state is :func:`solve_cable_radial_state` at its tension.
    ``crossing_loads=False`` skips the crossed-wire line loads (an empty
    ``force`` map) for callers that need only the pressure; the crossing
    representation refuses some geometries the pressure does not.
    """
    tension = np.asarray(tension_N, dtype=np.float64)
    unique, inverse = np.unique(tension, return_inverse=True)
    grid = (
        unique
        if unique.size <= SOLVED_GRID_POINTS
        else np.linspace(unique[0], unique[-1], SOLVED_GRID_POINTS)
    )
    if radial_inputs:
        states = solve_cable_radial_states(cable, grid, **radial_inputs)
    else:
        states = [solve_cable_radial_state(cable, float(value)) for value in grid]
    p_values = [
        armour_contact_pressure(cable, contact_model="solved", radial_state=state)
        for state in states
    ]
    f_values = (
        [
            armour_crossing_line_loads(
                cable, contact_model="solved", radial_state=state
            )
            for state in states
        ]
        if crossing_loads
        else []
    )

    def history(samples: list[float]) -> FloatArray:
        values = np.asarray(samples, dtype=np.float64)
        if grid is unique:
            return np.asarray(values[inverse])
        return np.asarray(np.interp(tension, grid, values))

    pressure = {
        name: history([state[name] for state in p_values]) for name in p_values[0]
    }
    force = {
        name: history([state[name][1] for state in f_values])
        for name in (f_values[0] if f_values else ())
    }
    stress = {
        name: history([state.result.wire_stress_Pa[name] for state in states])
        for name in states[0].result.wire_stress_Pa
    }
    return SolvedContactHistories(pressure, force, stress)


def cable_to_fatigue_layers(
    cable: CableData,
    *,
    contact_tension_N: float = 0.0,
    mu: float | None = None,
    contact_model: str = "bounded",
    section_state: SectionState | None = None,
    radial_state: CableRadialState | None = None,
) -> list[FatigueLayer]:
    r"""Assemble :class:`FatigueLayer` specs for a cable's structural armour.

    For each structural armour layer: the frozen M-κ spec (geometry, ξ, ``F_n``
    at ``contact_tension_N``), the bulk S-N curve from ``layer.bulk_sn``, the
    ultimate strength from the material library, and — routed by ``fretting_sn``
    — either a steel slip-amplitude S-N branch (Hertz ``K_t``/``p_0`` from the
    opposite-laid partner layer, μ = ``mu_on_steel``) or a copper SWT strain-life
    branch (μ = ``mu_on_cu``). An empty fretting name explicitly selects
    bulk-only assessment. A named but unavailable law is an input error;
    a defined law with no physical crossing has no crossing contribution.

    Raises
    ------
    CableSchemaError
        If a layer names an unknown/unavailable fatigue law or assigns a
        slip-amplitude curve to bulk stress (or a stress curve to slip).
    """
    require_fully_specified(cable, "cable_to_fatigue_layers")
    if contact_model == "solved" and radial_state is None:
        radial_state = solve_cable_radial_state(cable, contact_tension_N)
    specs = cable_to_mkappa_specs(
        cable,
        contact_tension_N,
        mu=mu,
        contact_model=contact_model,
        section_state=section_state,
        radial_state=radial_state,
    )
    armours = structural_armour_layers(cable)
    crossings = armour_crossing_line_loads(
        cable,
        contact_tension_N,
        contact_model=contact_model,
        section_state=section_state,
        radial_state=radial_state,
    )
    fatigue_layers: list[FatigueLayer] = []

    for spec, lyr in zip(specs, armours, strict=True):
        try:
            sn_bulk = get_sn_curve(lyr.bulk_sn)
        except BlockedSNCurveError as exc:
            raise CableSchemaError(
                f"layer {lyr.name!r} bulk S-N {lyr.bulk_sn!r} is BLOCKED: {exc}"
            ) from None
        except KeyError as exc:
            raise CableSchemaError(
                f"layer {lyr.name!r} names an unknown bulk S-N {lyr.bulk_sn!r}: {exc}"
            ) from None
        if sn_bulk.stress_unit != "MPa":
            raise CableSchemaError(
                f"layer {lyr.name!r} bulk fatigue requires a stress-range curve "
                f"in MPa, not {sn_bulk.stress_unit!r}"
            )
        sigma_uts = library.get_property(lyr.material, "sigma_UTS_Pa")

        # Opposite-laid partner armour (for the Hertz crossing contact).
        partner, f_n = crossings.get(lyr.name, (None, 0.0))

        sn_fret = None
        swt_const = None
        k_t = p0 = 0.0
        fret_mu = 0.0
        name = lyr.fretting_sn
        if name in SWT_LIFE:  # copper SWT strain-life branch
            swt_const = get_swt_life(name)
            fret_mu = library.get_property(lyr.material, "mu_on_cu")
            # K_t drives the crossing's slip history for the per-cycle driver.
            k_t, p0, crossing_density = _fretting_contact(
                lyr,
                partner,
                f_n,
                youngs_modulus_Pa=(
                    section_state.youngs_modulus_Pa
                    if section_state is not None
                    else None
                ),
            )
        elif name:  # explicitly requested steel slip-amplitude curve
            try:
                sn_fret = get_sn_curve(name)
            except (BlockedSNCurveError, KeyError) as exc:
                raise CableSchemaError(
                    f"layer {lyr.name!r} requests unavailable contact-fatigue "
                    f"law {name!r}: {exc}. Omit fretting_sn only when an "
                    "explicitly bulk-only assessment is intended."
                ) from None
            if sn_fret.stress_unit != "um":
                raise CableSchemaError(
                    f"layer {lyr.name!r} slip-amplitude fatigue requires a curve "
                    f"in um, not {sn_fret.stress_unit!r}"
                )
            fret_mu = library.get_property(lyr.material, "mu_on_steel")
            k_t, p0, crossing_density = _fretting_contact(
                lyr,
                partner,
                f_n,
                youngs_modulus_Pa=(
                    section_state.youngs_modulus_Pa
                    if section_state is not None
                    else None
                ),
            )
        else:
            crossing_density = 0.0
        # An empty fretting name explicitly requests bulk-only assessment.

        fatigue_layers.append(
            FatigueLayer(
                spec=spec,
                sn_bulk=sn_bulk,
                sigma_UTS_Pa=sigma_uts,
                name=lyr.name,
                sn_fretting=sn_fret,
                K_t_contact_N_per_m=k_t,
                fretting_mu=fret_mu,
                swt_const=swt_const,
                contact_p0_Pa=p0,
                crossing_density_per_m=crossing_density,
                contact_line_load_N_per_m=f_n,
                above_core_mask=_above_core_mask(cable, spec.phi_wires),
                contact_nu=library.get_property(lyr.material, "nu"),
                SCF=lyr.termination_scf,
                termination=lyr.termination_scf != 1.0,
                lay_hand=1 if lyr.lay_direction == "right_hand" else -1,
            )
        )
    return fatigue_layers
