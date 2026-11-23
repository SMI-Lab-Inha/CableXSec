# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Longitudinal stress over the whole cross-section, for section stress maps.

:func:`compute_section_stress` gives the longitudinal (axial) stress at
points over every component of the section at one tension, curvature and
bending direction, in one slip state of the laid cores, with the utilisation
of each component at those points. It adds no physics: every value comes from
the functions :func:`~cablexsec.analysis.components.compute_component_response`
uses, at the same inputs, so the extremes of each layer's field are that
layer's :class:`~cablexsec.analysis.components.ComponentState` extremes and the
largest utilisation of a layer is its ``mechanical_utilization``.

The fields, by the stress model of the layer:

Armour wires (``"wires"``)
    One point per wire, at its centre line: the tension, friction and thermal
    stress of :func:`~cablexsec.analysis.stress.compute_stress` (the wire's
    axial force over its area). The fibres round the wire add and subtract
    the local bending amplitude: ``sigma_min_Pa`` and ``sigma_max_Pa`` are
    the perimeter extremes of each wire, the first-yield basis of the wire
    utilisation.
Continuous layers (``"annulus"``)
    Concentric layers, unresolved cores (stuck, frictional or free) and
    metallic sheaths: points at ``n_points`` angles round the layer's own
    centre (the cable axis or each core axis) and at ``n_radii`` radii
    through the wall, where the plane-section field varies through it. The
    extremes lie at the outer radius. A metallic sheath is sampled at its
    mean radius, as the thin-shell equations are.
Resolved conductor rings (``"rings"``)
    Points at ``n_points`` angles on each wire ring of each core: the stress
    at the wire centre line, with the local bending amplitude of the wire as
    the fibre range.
Layers that follow the section (``kinematic``)
    A continuous layer the model leaves out of the load path (a loose tape or
    wire screen of a core: it adds no axial stiffness) is given the stress of
    the strain it follows, ``E`` times that strain: the axial strain of the
    section, the bending strain about the section's neutral axis (or about
    its own core axis in the free core state, as its core), less its free
    thermal strain, from the same plane-section formulas as the load-bearing
    layer it would be. ``kinematic`` is true and ``basis`` begins with
    :data:`KINEMATIC`. It carries no share of the tension in the model; its
    component state reports the same stress, marked ``kinematic``, and its
    utilisation against its limits.
No stress model (``"none"``)
    Nested stranded conductors, non-structural helical layers (pressure
    armour, carcass) and layers without a Young's modulus: no value;
    ``basis`` says why.

Positions are in the frame of the physics turned by ``rotation_rad`` about
the cable axis: the first core and the first armour wire of the physics lie
on its x axis, so ``rotation_rad = pi/2`` puts them at the top, as the
section drawing of the application does. The curvature components are given
in the physics frame, as for ``compute_component_response``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from cablexsec.analysis.component_limits import (
    assess,
    component_limits,
    point_usage,
)
from cablexsec.analysis.components import (
    KINEMATIC,
    ComponentResponse,
    _axial_stiffnesses,
    _component_type,
    _Continuum,
    _continuum_chunks,
    _CoreAxis,
    _following,
    _property,
    _yield_metric,
    compute_component_response,
    given_core_axes,
    own_bending_ratio,
)
from cablexsec.analysis.stress import FrictionField, StressResult, compute_stress
from cablexsec.cables.schema import (
    CableData,
    CylindricalLayer,
    CylindricalOffsetLayer,
    CylindricalShellLayer,
    HelicalLayer,
    HelicalStrandedLayer,
    is_serving,
)
from cablexsec.cables.schema import scope_entry_points as _scope_entry_points
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.core_friction import CoreFrictionSpec
from cablexsec.mechanics.core_kinematics import CoreSlipState
from cablexsec.mechanics.section_state import SectionState
from cablexsec.mechanics.shell_bending import shell_wall_stress
from cablexsec.mechanics.thermal import ThermalResult
from cablexsec.provenance import Provenance

__all__ = [
    "KINEMATIC",
    "LayerStressField",
    "SectionStressField",
    "armour_stress_fields",
    "compute_section_stress",
]

FloatArray = NDArray[np.float64]

FIELD_KINDS = ("wires", "annulus", "rings", "none")
"""Layouts of a layer's points (module notes)."""


@dataclass(frozen=True)
class LayerStressField:
    """The longitudinal stress of one layer at points over its section.

    Points are indexed ``[instance, radius, angle]``. ``centres_m`` holds the
    centre of each instance (the cable axis for a concentric layer, the axis
    of each core, the centre line of each wire); a point lies at
    ``radii_m[j]`` from its instance centre in the direction
    ``angles_rad[k] + turns_rad[i]`` (both ``[0.0]`` for a wire). A laid core
    is sampled in its own radial and tangential frame, as the core-lay
    bending of :func:`~cablexsec.mechanics.core_lay.core_bending_components`
    is written: its angles start from the direction away from the cable axis,
    so ``turns_rad`` is the azimuth of each core; it is zero for every other
    layout. ``sigma_Pa`` is the stress at the point (the
    centre line of a wire), ``sigma_min_Pa`` and ``sigma_max_Pa`` the range
    over the fibres of the wire there (equal to ``sigma_Pa`` in a continuous
    layer). ``utilisation`` is the utilisation of the layer against all the
    limits its material gives at the point (the largest over its modes), NaN
    where the layer has no limit. ``core_state`` names
    the core slip state the values are in (empty for concentric layers,
    armour and sheaths). ``basis`` says in words how the values were
    obtained, or why a layer has none. ``kinematic`` marks a layer outside
    the load path whose stress is its modulus times the strain it follows
    (module notes).
    """

    name: str
    kind: str
    centres_m: FloatArray
    radii_m: FloatArray
    angles_rad: FloatArray
    sigma_Pa: FloatArray
    sigma_min_Pa: FloatArray
    sigma_max_Pa: FloatArray
    utilisation: FloatArray
    failure_mode: str
    basis: str
    core_state: str = ""
    turns_rad: FloatArray | None = None
    kinematic: bool = False

    @property
    def has_values(self) -> bool:
        """Whether the layer has a stress model (a field to draw)."""
        return self.kind != "none"

    def turns(self) -> FloatArray:
        """The turn of each instance's own frame (zeros when not given)."""
        if self.turns_rad is None:
            return np.zeros(self.centres_m.shape[0])
        return np.asarray(self.turns_rad, dtype=np.float64)

    def points_m(self) -> FloatArray:
        """Positions of every point, shape ``(instances, radii, angles, 2)``."""
        radius = self.radii_m[None, :, None]
        angle = self.angles_rad[None, None, :] + self.turns()[:, None, None]
        x = self.centres_m[:, 0, None, None] + radius * np.cos(angle)
        y = self.centres_m[:, 1, None, None] + radius * np.sin(angle)
        return np.stack([x, y], axis=-1)


@dataclass(frozen=True)
class SectionStressField:
    """Longitudinal stress over the section at one tension and curvature.

    ``bending_direction_rad`` is the direction of the curvature vector in
    the frame of the positions: the fibres on that side of the cable axis
    are stretched by bending. ``response`` is the component response at the
    same inputs, whose extremes and utilisations the fields reproduce.
    """

    layers: tuple[LayerStressField, ...]
    response: ComponentResponse
    kappa_1_m: float
    bending_direction_rad: float
    rotation_rad: float
    tension_N: float
    core_state: str
    provenance: Provenance | None = field(default=None, compare=False)

    def layer(self, name: str) -> LayerStressField:
        """The field of the layer ``name``."""
        for item in self.layers:
            if item.name == name:
                return item
        raise CableInputError(f"no layer named {name!r}")


def _none(name: str, reason: str) -> LayerStressField:
    empty = np.zeros((0, 1, 1))
    return LayerStressField(
        name=name,
        kind="none",
        centres_m=np.zeros((0, 2)),
        radii_m=np.zeros(1),
        angles_rad=np.zeros(1),
        sigma_Pa=empty,
        sigma_min_Pa=empty.copy(),
        sigma_max_Pa=empty.copy(),
        utilisation=empty.copy(),
        failure_mode="not assessed",
        basis=reason,
    )


def _turn(points: FloatArray, angle: float) -> FloatArray:
    """``points`` (n, 2) turned by ``angle`` about the origin."""
    c, s = math.cos(angle), math.sin(angle)
    x, y = points[:, 0], points[:, 1]
    return np.stack([c * x - s * y, s * x + c * y], axis=-1)


def _core_phases(layer: CylindricalOffsetLayer) -> FloatArray:
    """Azimuth of each core axis in the physics frame: the first on the x axis."""
    return np.array(
        [2 * math.pi * core / layer.n_cores for core in range(layer.n_cores)],
        dtype=np.float64,
    )


def _core_centres(layer: CylindricalOffsetLayer) -> FloatArray:
    """Core axes in the physics frame: the first on the x axis."""
    offset = float(layer.r_offset_m or 0.0)
    phases = _core_phases(layer)
    return np.stack([offset * np.cos(phases), offset * np.sin(phases)], axis=-1)


def _through_wall(inner: float, outer: float, n_radii: int) -> FloatArray:
    """Sample radii from ``inner`` to ``outer`` (the outer radius alone for one)."""
    if n_radii == 1 or outer <= inner:
        return np.array([outer], dtype=np.float64)
    return np.linspace(inner, outer, n_radii)


def _bend_angles(n_points: int) -> FloatArray:
    """Angles from the bending direction that include both extreme fibres."""
    count = n_points + n_points % 2
    return np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)


def armour_stress_fields(
    cable: CableData, result: StressResult, *, rotation_rad: float = 0.0
) -> tuple[LayerStressField, ...]:
    """The wire fields of the structural armour layers of a wire-stress result.

    ``result`` is a :func:`~cablexsec.analysis.stress.compute_stress` result
    of ``cable``. One point per wire, at its centre line: the tension,
    friction and thermal stress (the wire's axial force over its area); the
    fibre range adds and subtracts the local bending amplitude round the wire
    (the perimeter extremes, the first-yield basis of the wire utilisation).
    Positions are turned by ``rotation_rad`` (module notes).
    """
    fields: list[LayerStressField] = []
    for layer in cable.helical_layers:
        name = layer.name
        if name not in result.sigma_vm_envelope_Pa:
            continue
        phi = np.radians(result.phi_deg[name]) + rotation_rad
        r_h = float(layer.r_helix_m or 0.0)
        centres = np.stack([r_h * np.cos(phi), r_h * np.sin(phi)], axis=-1)
        low = np.asarray(result.sigma_perimeter_min_Pa[name], dtype=np.float64)
        high = np.asarray(result.sigma_perimeter_max_Pa[name], dtype=np.float64)
        centre_line = (
            result.sigma_A_Pa[name] + result.sigma_f_Pa[name] + result.sigma_T_Pa[name]
        )
        metrics = [
            _yield_metric(layer, float(a), float(b))
            for a, b in zip(low, high, strict=True)
        ]
        usage = np.array([value for value, _ in metrics], dtype=np.float64)
        mode = metrics[0][1] if metrics else "not assessed"
        shape = (centres.shape[0], 1, 1)
        fields.append(
            LayerStressField(
                name=name,
                kind="wires",
                centres_m=centres,
                radii_m=np.zeros(1),
                angles_rad=np.zeros(1),
                sigma_Pa=np.asarray(centre_line, dtype=np.float64).reshape(shape),
                sigma_min_Pa=low.reshape(shape),
                sigma_max_Pa=high.reshape(shape),
                utilisation=usage.reshape(shape),
                failure_mode=mode,
                basis="wire centre line: tension, friction and thermal stress; "
                "the fibre range adds the local bending of the wire",
            )
        )
    return tuple(fields)


def compute_section_stress(
    cable: CableData,
    kappa_x: float,
    kappa_y: float = 0.0,
    *,
    tension_N: float = 0.0,
    mu: float | None = None,
    core_state: CoreSlipState = "frictional",
    core_friction: CoreFrictionSpec | None = None,
    thermal: ThermalResult | None = None,
    T_ref_C: float = 20.0,
    p_int_Pa: float = 0.0,
    p_ext_Pa: float | None = None,
    contact_model: str | None = None,
    n_points: int = 180,
    n_radii: int = 3,
    rotation_rad: float = 0.0,
    friction: FrictionField | None = None,
    armour: bool = True,
) -> SectionStressField:
    """Longitudinal stress at points over every component of the section.

    The inputs are those of
    :func:`~cablexsec.analysis.components.compute_component_response`, which
    is called with them; ``n_radii`` sets the radii sampled through the wall
    of a continuous layer and ``rotation_rad`` turns the returned positions
    (module notes). The per-layer extremes of the fields equal the
    component extremes in the core slip state each layer is assessed in.
    ``friction`` carries the friction a load history left on the armour
    wires and the laid cores (:class:`~cablexsec.analysis.stress.FrictionField`),
    in place of the slip fronts at the curvature: the stress of a frame of a
    history. ``armour=False`` leaves the armour wires without a field, for a
    caller that needs the other components only.
    """
    if n_radii < 1:
        raise CableInputError("n_radii must be at least 1")
    if not math.isfinite(rotation_rad):
        raise CableInputError("rotation_rad must be finite")
    response = compute_component_response(
        cable,
        kappa_x,
        kappa_y,
        tension_N=tension_N,
        mu=mu,
        core_state=core_state,
        core_friction=core_friction,
        thermal=thermal,
        T_ref_C=T_ref_C,
        p_int_Pa=p_int_Pa,
        p_ext_Pa=p_ext_Pa,
        contact_model=contact_model,
        n_points=n_points,
        friction=friction,
        armour=armour,
    )
    section_state = response.section_state
    assert section_state is not None
    spec = core_friction if core_friction is not None else CoreFrictionSpec()
    stiffness = {
        name: value * section_state.modulus_scale[name]
        for name, value in _axial_stiffnesses(cable).items()
    }
    ea_total = sum(stiffness.values())
    kappa = math.hypot(kappa_x, kappa_y)
    phi_bend = math.atan2(kappa_y, kappa_x)
    wire_result = (
        compute_stress(
            cable,
            kappa_x,
            kappa_y,
            mu=mu,
            section_state=section_state,
            friction=friction,
        )
        if cable.helical_layers and armour
        else None
    )
    states = {state.name: state for state in response.states}
    wires = {
        item.name: item
        for item in (
            armour_stress_fields(cable, wire_result, rotation_rad=rotation_rad)
            if wire_result is not None
            else ()
        )
    }
    axes: dict[float, _CoreAxis] = given_core_axes(
        cable, section_state, kappa_x, kappa_y, spec, friction
    )
    fields: list[LayerStressField] = []
    for layer in cable.layers:
        name = layer.name
        if isinstance(layer, HelicalLayer):
            if wire_result is None or name not in wire_result.sigma_vm_envelope_Pa:
                fields.append(_none(name, states[name].basis))
                continue
            fields.append(wires[name])
            continue
        if isinstance(layer, HelicalStrandedLayer):
            fields.append(_none(name, states[name].basis))
            continue
        if section_state.radial is not None and not is_serving(layer):
            fields.append(
                _radial_field(
                    layer,
                    section_state,
                    kappa,
                    phi_bend,
                    n_points,
                    n_radii,
                    rotation_rad,
                    states[name].mechanical_utilization,
                    states[name].failure_mode,
                )
            )
            continue
        if isinstance(layer, CylindricalShellLayer):
            fields.append(
                _shell_field(
                    layer,
                    section_state,
                    kappa,
                    phi_bend,
                    ea_total,
                    n_points,
                    rotation_rad,
                    states[name].failure_mode,
                    states[name].basis,
                )
            )
            continue
        assert isinstance(layer, CylindricalLayer | CylindricalOffsetLayer)
        fields.append(
            _continuum_field(
                cable,
                layer,
                section_state,
                stiffness,
                kappa_x,
                kappa_y,
                n_points,
                n_radii,
                axes,
                spec,
                core_state,
                rotation_rad,
                states[name].failure_mode,
            )
        )
    return SectionStressField(
        layers=tuple(fields),
        response=response,
        kappa_1_m=kappa,
        bending_direction_rad=phi_bend + rotation_rad,
        rotation_rad=rotation_rad,
        tension_N=response.tension_N,
        core_state=core_state,
        provenance=Provenance(
            method="longitudinal stress at points over every component of the "
            "section, from the component response at the same inputs",
            citations=(response.provenance.citations if response.provenance else ()),
            gates=(
                "the extremes of each layer's field equal its component extremes",
                "the largest utilisation of each layer equals its component "
                "utilisation",
            ),
            limitations=(
                *(response.provenance.limitations if response.provenance else ()),
                "values between the sampled points follow the plane-section "
                "field; a map draws each point's value over the cell round it",
                "a layer outside the load path (a loose screen) is given its "
                "modulus times the strain it follows; it carries no share of the "
                "tension in the model",
            ),
        ),
    )


def _continuum_field(
    cable: CableData,
    layer: CylindricalLayer | CylindricalOffsetLayer,
    section_state: SectionState,
    stiffness: dict[str, float],
    kappa_x: float,
    kappa_y: float,
    n_points: int,
    n_radii: int,
    axes: dict[float, _CoreAxis],
    spec: CoreFrictionSpec,
    core_state: str,
    rotation: float,
    failure_mode: str,
) -> LayerStressField:
    """A concentric layer, an unresolved core or a resolved core.

    A layer outside the load path is given the stress of the strain it
    follows (:func:`_following`), marked ``kinematic``.
    """
    name = layer.name
    alpha = _property(layer.material, "alpha_T_K")
    delta_T = section_state.free_thermal_strain[name] / alpha if alpha else 0.0
    thermal = alpha * delta_T

    def chunks_at(
        radius: float | None, state: SectionState, rigidity: dict[str, float]
    ) -> _Continuum:
        return _continuum_chunks(
            cable,
            layer,
            state,
            rigidity,
            kappa_x,
            kappa_y,
            thermal,
            n_points,
            axes,
            spec,
            core_state,
            radius,
        )

    outer = chunks_at(None, section_state, stiffness)
    kinematic = not outer.loaded
    if kinematic:
        if not section_state.youngs_modulus_Pa.get(name, 0.0) > 0.0:
            return _none(
                name,
                "has no Young's modulus in the model, so no stress is assigned",
            )
        # The same formulas as a load-bearing layer, at the strain it follows.
        section_state = _following(cable, layer, section_state)
        stiffness = {**stiffness, name: 1.0}
        outer = chunks_at(None, section_state, stiffness)
    basis = f"{KINEMATIC}; {outer.basis}" if kinematic else outer.basis
    state_key = outer.assessed or "stuck"
    limits = component_limits(layer.material, _component_type(layer))
    modulus = section_state.youngs_modulus_Pa[name]
    angles = np.linspace(0.0, 2.0 * math.pi, n_points, endpoint=False)
    if isinstance(layer, CylindricalLayer):
        centres = np.zeros((1, 2))
    else:
        centres = _core_centres(layer)
    if outer.kind == "resolved":
        assert isinstance(layer, CylindricalOffsetLayer)
        n_cores = layer.n_cores
        rings = layer.strand_layers
        parts = outer.chunks[state_key]
        shape = (n_cores, len(rings), n_points)
        low = np.zeros(shape)
        high = np.zeros(shape)
        usage = np.full(shape, math.nan)
        for index, part in enumerate(parts):
            ring, core = divmod(index, n_cores)
            low[core, ring] = part[:n_points]
            high[core, ring] = part[n_points:]
            net = float(np.mean(part))
            usage[core, ring] = np.maximum(
                point_usage(limits, part[:n_points], modulus, net),
                point_usage(limits, part[n_points:], modulus, net),
            )
        radii = np.array([float(ring.r_helix_m) for ring in rings])
        return LayerStressField(
            name=name,
            kind="rings",
            centres_m=_turn(centres, rotation),
            radii_m=radii,
            angles_rad=angles + rotation,
            sigma_Pa=0.5 * (low + high),
            sigma_min_Pa=low,
            sigma_max_Pa=high,
            utilisation=usage,
            failure_mode=failure_mode,
            basis="wire centre line of each ring; the fibre range adds the local "
            "bending of the wire; " + basis,
            core_state=outer.assessed,
            turns_rad=_core_phases(layer),
            kinematic=kinematic,
        )
    inner = float(layer.r_inner_m or 0.0)
    radii = _through_wall(inner, float(layer.r_outer_m or 0.0), n_radii)
    n_inst = centres.shape[0]
    sigma = np.zeros((n_inst, radii.size, n_points))
    for j, radius in enumerate(radii):
        at = (
            outer
            if j == radii.size - 1
            else chunks_at(float(radius), section_state, stiffness)
        )
        for i, part in enumerate(at.chunks[state_key]):
            sigma[i, j] = part
    usage = np.full(sigma.shape, math.nan)
    for i, part in enumerate(outer.chunks[state_key]):
        usage[i] = point_usage(limits, sigma[i], modulus, float(np.mean(part)))
    return LayerStressField(
        name=name,
        kind="annulus",
        centres_m=_turn(centres, rotation),
        radii_m=radii,
        angles_rad=angles + rotation,
        sigma_Pa=sigma,
        sigma_min_Pa=sigma.copy(),
        sigma_max_Pa=sigma.copy(),
        utilisation=usage,
        failure_mode=failure_mode,
        basis=basis,
        core_state=outer.assessed,
        # The core-lay bending is written in each core's radial frame.
        turns_rad=(
            _core_phases(layer)
            if outer.kind == "laid" and isinstance(layer, CylindricalOffsetLayer)
            else None
        ),
        kinematic=kinematic,
    )


def _shell_field(
    layer: CylindricalShellLayer,
    section_state: SectionState,
    kappa: float,
    phi_bend: float,
    ea_total: float,
    n_points: int,
    rotation: float,
    failure_mode: str,
    basis: str,
) -> LayerStressField:
    """A metallic sheath: the thin-shell stress at its mean radius, by angle.

    The wall stress of :func:`~cablexsec.mechanics.shell_bending.shell_wall_stress`
    is linear in the curvature, so the stress at an angle ``theta`` from the
    bending direction is its value at the curvature ``kappa cos(theta)``.
    """
    assert layer.r_mid_m is not None
    name = layer.name
    alpha = _property(layer.material, "alpha_T_K")
    delta_T = section_state.free_thermal_strain[name] / alpha if alpha else 0.0
    offsets = _bend_angles(n_points)
    sigma = np.zeros(offsets.size)
    usage = np.zeros(offsets.size)
    limits = component_limits(layer.material, _component_type(layer))
    for k, cosine in enumerate(np.cos(offsets)):
        state = shell_wall_stress(
            layer,
            kappa * float(cosine),
            section_state.axial_strain,
            ea_total,
            p_int_Pa=section_state.internal_pressure_Pa,
            p_ext_Pa=section_state.external_pressure_Pa,
            delta_T_K=delta_T,
            youngs_modulus_Pa=section_state.youngs_modulus_Pa[name],
        )
        sigma[k] = state.sigma_longitudinal_Pa
        usage[k] = assess(
            limits,
            peak_stress_Pa=state.sigma_von_mises_Pa,
            peak_strain=abs(section_state.axial_strain)
            + abs(kappa * float(cosine)) * float(layer.r_mid_m),
        ).utilisation
    shape = (1, 1, offsets.size)
    return LayerStressField(
        name=name,
        kind="annulus",
        centres_m=np.zeros((1, 2)),
        radii_m=np.array([float(layer.r_mid_m)]),
        angles_rad=offsets + phi_bend + rotation,
        sigma_Pa=sigma.reshape(shape),
        sigma_min_Pa=sigma.reshape(shape).copy(),
        sigma_max_Pa=sigma.reshape(shape).copy(),
        utilisation=usage.reshape(shape),
        failure_mode=failure_mode,
        basis=basis,
    )


def _radial_field(
    layer: CylindricalLayer | CylindricalOffsetLayer | CylindricalShellLayer,
    section_state: SectionState,
    kappa: float,
    phi_bend: float,
    n_points: int,
    n_radii: int,
    rotation: float,
    utilisation: float,
    failure_mode: str,
) -> LayerStressField:
    """A layer of the coupled radial section: the coupled normal stress with
    elastic bending about the cable axis superposed, through the wall."""
    normal = section_state.layers[layer.name]
    if isinstance(layer, CylindricalShellLayer):
        assert layer.r_mid_m is not None and layer.t_m is not None
        inner = layer.r_mid_m - layer.t_m / 2
        outer = layer.r_mid_m + layer.t_m / 2
        scf = layer.scf * layer.f_corrugation
    else:
        inner = float(layer.r_inner_m or 0.0)
        outer = float(layer.r_outer_m or 0.0)
        scf = own_bending_ratio(layer)
    modulus = section_state.youngs_modulus_Pa[layer.name]
    if isinstance(layer, CylindricalOffsetLayer):
        centres = _core_centres(layer)
    else:
        centres = np.zeros((1, 2))
    radii = _through_wall(inner, outer, n_radii)
    offsets = _bend_angles(n_points)
    bend = scf * modulus * kappa * radii[:, None] * np.cos(offsets)[None, :]
    sigma = np.broadcast_to(
        normal.longitudinal_stress_Pa + bend, (centres.shape[0], *bend.shape)
    ).copy()
    return LayerStressField(
        name=layer.name,
        kind="annulus",
        centres_m=_turn(centres, rotation),
        radii_m=radii,
        angles_rad=offsets + phi_bend + rotation,
        sigma_Pa=sigma,
        sigma_min_Pa=sigma.copy(),
        sigma_max_Pa=sigma.copy(),
        utilisation=np.full(sigma.shape, utilisation),
        failure_mode=failure_mode,
        basis="coupled normal section stress with superposed elastic bending "
        "about the cable axis",
    )


# Public entry points read the cable's own materials when called directly.
_scope_entry_points(globals(), __all__)
