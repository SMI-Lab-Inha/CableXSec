# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Cross-section response and failure screening for every cable component.

This module assembles existing CableXSec kernels without fitting a response
parameter. Helical armour is recovered wire by wire, metallic sheaths use the
thin-shell equations, and continuous or offset cylindrical components use
prescribed Euler--Bernoulli longitudinal-strain fields.

Laid power cores (offset components) are reported in three slip states of the
core axis, the states of :mod:`cablexsec.mechanics.core_kinematics`:

``"stuck"``
    The core is held to the cable section: its axis takes the plane-section
    strain ``r_c cos^2(beta) (kappa . e_phi)``.
``"free"``
    The core slides freely along its helix: its axis carries no bending
    strain and every fibre bends about the core's own axis only.
``"frictional"``
    The core stick-slip law of :mod:`cablexsec.mechanics.core_friction`
    (Papailiou cap set by the core contact loads, Yun stick coefficient),
    evaluated in closed form for a monotonic ramp to the target curvature, as
    :func:`~cablexsec.analysis.stress.compute_stress` does for the armour
    wires. Cores that are not laid (no ``core_lay``) cannot slip over a
    uniform bend in this law (the slip cap grows as ``1/sin(beta)``), so their
    frictional state is the stuck state.

The stuck state holds only below the slip curvature of the cores; above it
the frictional state is the consistent one. Extremes, strains and the
utilisation of an offset component use the frictional state by default
(``core_state``); all three stress ranges are returned.

Limits. Every component is checked against all the limits its material
gives (:mod:`cablexsec.analysis.component_limits`): the yield stress (for a
conductor its 0.2 % proof stress) on the largest stress magnitude, the
allowable strain on the largest mechanical strain magnitude, and for a
conductor the tensile strength on the net-section stress (the axial force of
the most loaded core over its area, as the conductor-fracture mechanism of
:func:`~cablexsec.analysis.capacity.compute_capacity`). The utilisation is the
largest of them and ``failure_mode`` names it; a component whose material
gives none has none, ``"no limit given"``. Armour wires keep their first-yield
check.

Stress basis of a stranded conductor in bending. A conductor modelled as a
solid whose material declares an effective bending modulus ``E_b`` below its
axial modulus ``E`` (``E_eff_bending_Pa``) bends with the stiffness
``E_b I`` (:func:`~cablexsec.mechanics.stiffness.ei_min_slip`), that of wires
that slip and bend about their own axes. Its bending stress about its own
axis is taken on that basis: wires of radius ``r_eq = R sqrt(E_b / E)``
filling the conductor area and bending about their own axes have the
stiffness ``E_b pi R^4 / 4``, so the extreme wire fibre takes
``E kappa r_eq``, and the plane-section bending about the conductor's own axis
is scaled by ``sqrt(E_b / E)`` (:func:`own_bending_ratio`). The axial part of
the stress (tension, core-axis friction or stick) is not scaled. A resolved
conductor (wire rings) is on the wire basis already: each wire bends about its
own axis with its own diameter.

Layers outside the load path (a loose tape or wire screen of a core: no axial
stiffness, :data:`KINEMATIC`) are given the stress of the strain they follow,
their modulus times that strain, and are assessed against their limits as
the others; ``kinematic`` marks them. That stress is an upper bound: a loose
screen that slips over its core takes less.

Thermal stress is formed from axial-force equilibrium. A common cable strain
is solved from the imposed tension and the free thermal forces of every
load-carrying component. This avoids treating ``E alpha DeltaT`` as an
independent tensile stress and gives zero stress for a freely expanding,
uniform straight bar.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cablexsec.analysis.component_limits import (
    Assessment,
    assess,
    component_limits,
)
from cablexsec.analysis.stress import FrictionField, compute_stress
from cablexsec.cable_adapters import require_fully_specified
from cablexsec.cables.schema import (
    CableData,
    CylindricalLayer,
    CylindricalOffsetLayer,
    CylindricalShellLayer,
    HelicalLayer,
    HelicalStrandedLayer,
    Layer,
)
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.materials import library
from cablexsec.mechanics.core_friction import CoreFrictionSpec, build_core_friction
from cablexsec.mechanics.core_kinematics import CORE_SLIP_STATES, CoreSlipState
from cablexsec.mechanics.core_lay import core_angle, core_bending_components
from cablexsec.mechanics.papailiou import wire_slip_stress_cap
from cablexsec.mechanics.section_geometry import axial_cosine
from cablexsec.mechanics.section_state import (
    LayerNormalState,
    SectionState,
    solve_section_state,
)
from cablexsec.mechanics.shell_bending import shell_wall_stress
from cablexsec.mechanics.stiffness import axial_stiffness_by_layer
from cablexsec.mechanics.thermal import ThermalResult
from cablexsec.mechanics.wire_stress import local_bending_amplitude
from cablexsec.provenance import Provenance, require_finite

__all__ = [
    "KINEMATIC",
    "ComponentResponse",
    "ComponentState",
    "ComponentUsageCurve",
    "bending_directions",
    "capacity_utilisation",
    "component_usage_curve",
    "compute_component_response",
    "core_component_utilisation",
    "own_bending_ratio",
    "usage_curvature_grid",
]

KINEMATIC = (
    "kinematic: the layer follows the strain of the section; it is not part of "
    "the load path in the model"
)
"""How the stress of a layer outside the load path is obtained, in words."""

_STATE_WORDS = {
    "stuck": "stuck (cores held to the cable axis)",
    "frictional": "frictional core slip",
    "free": "free core slip (each core bends about its own axis)",
}


@dataclass(frozen=True)
class ComponentState:
    """Stress, strain, temperature and available failure metric for one layer.

    ``sigma_local_*`` is the free core state: an offset component bends about
    its own axis only. ``sigma_stick_*`` is the stuck state: it adds the
    parallel-axis strain of the core held to the cable axis.
    ``sigma_frictional_*`` is the frictional state of the core friction law.
    The three coincide for concentric components. ``core_state`` names the
    state of ``strain_*``, ``sigma_equivalent_max_Pa``,
    ``sigma_net_section_max_Pa`` and ``mechanical_utilization`` (empty for
    concentric components, armour and sheaths). Each minimum and maximum is a
    spatial extremum within its stated kinematic assumption, not a resolved
    inter-component contact solution. A non-finite utilization means that the
    material database has no defensible limit for that failure mode. It is
    never silently replaced by a default.

    For every layer but the armour ``mechanical_utilization`` is the largest
    utilisation against the limits its material gives (module notes):
    the largest stress magnitude over the yield stress (for a conductor the
    0.2 % proof stress, on the strand basis of its bending stiffness), the
    largest mechanical strain magnitude over the allowable strain, and for a
    conductor the largest tensile net-section stress
    ``sigma_net_section_max_Pa`` over the tensile strength (the mean
    longitudinal stress over the section of the most loaded core, or of the
    most loaded wire ring of a conductor resolved into rings).
    ``failure_mode`` names the governing limit (``"yielding"``, ``"strain
    limit"``, ``"tensile fracture"``), ``utilisations`` gives every limit
    checked with its utilisation and ``limits`` the limits with their values
    and sources; without a limit the utilisation is NaN and the mode
    ``"no limit given"``. For armour wires it is the first-yield usage
    ``max |sigma_A + sigma_B + sigma_f + sigma_T| / sigma_y`` over every wire
    and wire-perimeter fibre, the same stress basis as
    :func:`~cablexsec.analysis.capacity.wire_yield_curvature` and
    :func:`~cablexsec.api.compute_capacity`; it reaches 1 at their first-yield
    curvature. ``contact_screen_utilization`` reports the legacy axial/contact
    screening indicator ``sqrt(sigma**2 + 3 (0.31 p0)**2) / sigma_y`` at the
    armour crossings, which equals ``sigma_equivalent_max_Pa / sigma_y``. It is
    dominated by the crossing pressure ``p0`` and can exceed 1 at zero
    curvature and tension, so it is not a yield check. It is NaN for every
    non-armour layer. For a metallic sheath it is first yield of the von Mises
    stress at the corrugation root (the corrugation stress concentration
    included).

    ``basis`` says, in words, how the values were obtained. ``kinematic``
    marks a layer outside the load path whose stresses are its modulus times
    the strain it follows (module notes).
    """

    name: str
    component_type: str
    material: str
    strain_min: float
    strain_max: float
    sigma_local_min_Pa: float
    sigma_local_max_Pa: float
    sigma_stick_min_Pa: float
    sigma_stick_max_Pa: float
    sigma_equivalent_max_Pa: float
    temperature_C: float
    mechanical_utilization: float
    failure_mode: str
    basis: str
    contact_screen_utilization: float = math.nan
    sigma_frictional_min_Pa: float = math.nan
    sigma_frictional_max_Pa: float = math.nan
    sigma_net_section_max_Pa: float = math.nan
    core_state: str = ""
    utilisations: tuple[tuple[str, float], ...] = ()
    limits: str = ""
    kinematic: bool = False

    @property
    def evidence(self) -> str:
        """Deprecated name of :attr:`basis`."""
        warnings.warn(
            "ComponentState.evidence is deprecated; use ComponentState.basis",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.basis


@dataclass(frozen=True)
class ComponentResponse:
    """All-component state at one tension, curvature and thermal condition.

    ``core_state`` is the requested slip state of the laid cores; each
    :class:`ComponentState` names the state it was assessed in.
    ``critical_component`` is the component of the load path with the largest
    utilisation; a layer outside it (``kinematic``) is assessed but not
    critical, its stress being an upper bound.
    """

    states: tuple[ComponentState, ...]
    axial_strain: float
    mechanical_axial_strain: float
    free_thermal_axial_strain: float
    tension_N: float
    kappa_x_1_m: float
    kappa_y_1_m: float
    critical_component: str
    critical_utilization: float
    provenance: Provenance | None = None
    section_state: SectionState | None = None
    core_state: str = "frictional"


def _property(material: str, name: str, default: float = 0.0) -> float:
    try:
        return library.get_property(material, name)
    except KeyError:
        return default


def _component_type(layer: Layer) -> str:
    """Component type of a layer.

    The roles that set limits come from the schema: the layer type, the core
    centre position (``cylindrical_offset`` with ``r_inner_m = 0`` is the
    conductor, the only type checked on its net section) and the material
    class (metal or polymer). Layer names never make a conductor; they only
    choose the descriptive label of a non-conductor layer where the schema
    cannot tell (screen or insulation, filler or protection), which does not
    change its limits.
    """
    if isinstance(layer, HelicalLayer):
        return "armour wire"
    if isinstance(layer, HelicalStrandedLayer):
        return "conductor strand"
    if isinstance(layer, CylindricalShellLayer):
        return "metallic sheath"
    core = isinstance(layer, CylindricalOffsetLayer)
    if isinstance(layer, CylindricalOffsetLayer):
        if (layer.r_inner_m or 0.0) <= 0.0:
            return "conductor"
        if layer.electrical is not None and layer.electrical.permittivity_relative:
            return "insulation"
    name = layer.name.lower()
    if "screen" in name or "shield" in name:
        return "screen"
    if "insulation" in name or "xlpe" in name:
        return "insulation"
    if "filler" in name:
        return "filler"
    if _is_metal(layer.material):
        return "metallic sheath" if "sheath" in name or not core else "screen"
    if "sheath" in name or "bedding" in name or "cover" in name:
        return "polymer protection"
    return "continuous layer"


def _is_metal(material: str) -> bool:
    """A metal by its modulus: every polymer of the library is below 10 GPa."""
    return not library.is_polymer(material) and _property(material, "E_Pa") >= 1.0e10


def _yield_metric(
    layer: Layer, sigma_min: float, sigma_max: float
) -> tuple[float, str]:
    limit = _property(layer.material, "sigma_y_Pa")
    if limit > 0.0:
        return max(abs(sigma_min), abs(sigma_max)) / limit, "yielding"
    return math.nan, "not assessed"


def _fracture_metric(layer: Layer, net_section_max: float) -> tuple[float, str]:
    limit = _property(layer.material, "sigma_UTS_Pa")
    if limit > 0.0 and math.isfinite(net_section_max):
        return max(net_section_max, 0.0) / limit, "tensile fracture"
    return math.nan, "not assessed"


def _assessment(
    layer: Layer,
    component_type: str,
    *,
    peak_stress_Pa: float,
    net_section_Pa: float = math.nan,
    peak_strain: float = math.nan,
) -> Assessment:
    """The utilisation of a non-armour component against all its limits."""
    return assess(
        component_limits(layer.material, component_type),
        peak_stress_Pa=peak_stress_Pa,
        net_section_Pa=net_section_Pa if component_type == "conductor" else math.nan,
        peak_strain=peak_strain,
    )


def own_bending_ratio(layer: Layer) -> float:
    """Bending stress about a component's own axis over its plane-section value.

    ``sqrt(E_b / E)`` for a continuous or offset layer modelled as a solid
    (no resolved wire rings) whose material declares an effective bending
    modulus ``E_b`` (``E_eff_bending_Pa``) below its axial modulus ``E``: the
    stress basis of the bending stiffness ``E_b I`` (module notes). One for
    every other layer.
    """
    if not isinstance(layer, CylindricalLayer | CylindricalOffsetLayer):
        return 1.0
    if isinstance(layer, CylindricalOffsetLayer) and layer.strand_layers:
        return 1.0
    bending = _property(layer.material, "E_eff_bending_Pa")
    try:
        axial = library.get_property(
            layer.material, library.axial_modulus_key(layer.material)
        )
    except KeyError:
        return 1.0
    if not (0.0 < bending < axial):
        return 1.0
    return math.sqrt(bending / axial)


def _own_basis(layer: Layer, ratio: float) -> str:
    """The bending-stress basis of a layer, in words ('' when plane section)."""
    if ratio >= 1.0:
        return ""
    bending = _property(layer.material, "E_eff_bending_Pa")
    return (
        "strand basis: the bending about its own axis is that of the wires its "
        f"effective bending modulus ({bending / 1e9:.3g} GPa) stands for, wires "
        "of a radius the square root of the ratio of the bending to the axial "
        "modulus times the conductor radius that slip and bend "
        f"about their own axes, {ratio:.3g} times the plane-section bending "
        "stress of a solid"
    )


def _cylindrical_points(
    layer: CylindricalLayer | CylindricalOffsetLayer,
    n_points: int,
    radius: float | None = None,
    own: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return fully compatible and local component coordinates.

    The points lie on the circle of ``radius`` about the layer's own centre
    (default the outer radius, where the extremes of a plane-section field
    lie), at ``n_points`` angles from the x axis. ``own`` scales the distance
    from the layer's own axis, the bending lever about it
    (:func:`own_bending_ratio`).
    """
    theta = np.linspace(0.0, 2.0 * math.pi, n_points, endpoint=False)
    radius = float(layer.r_outer_m or 0.0) if radius is None else float(radius)
    radius *= own
    x_local = radius * np.cos(theta)
    y_local = radius * np.sin(theta)
    if isinstance(layer, CylindricalLayer):
        return x_local, y_local, x_local, y_local

    x_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    offset = float(layer.r_offset_m or 0.0)
    for phi in np.linspace(0.0, 2.0 * math.pi, layer.n_cores, endpoint=False):
        x_all.append(offset * math.cos(phi) + x_local)
        y_all.append(offset * math.sin(phi) + y_local)
    return (
        np.concatenate(x_all),
        np.concatenate(y_all),
        np.tile(x_local, layer.n_cores),
        np.tile(y_local, layer.n_cores),
    )


def _axial_stiffnesses(cable: CableData) -> dict[str, float]:
    return axial_stiffness_by_layer(cable)


@dataclass(frozen=True)
class _CoreAxis:
    """Bending part of the core-axis strain of each core, per slip state.

    ``frictional`` is ``None`` when the core friction law cannot be built;
    ``note`` says how the frictional state was obtained, in words.
    """

    stuck: np.ndarray
    frictional: np.ndarray | None
    note: str

    def strain(self, state: str) -> np.ndarray | None:
        if state == "stuck":
            return self.stuck
        if state == "free":
            return np.zeros_like(self.stuck)
        return self.frictional


def _core_axis(
    cable: CableData,
    layer: CylindricalOffsetLayer,
    section_state: SectionState,
    kappa_x: float,
    kappa_y: float,
    spec: CoreFrictionSpec,
) -> _CoreAxis:
    """Core-axis bending strain of each core in the stuck and frictional states.

    The frictional strain is the closed-form monotonic slip front of the core
    population of :func:`~cablexsec.mechanics.core_friction.build_core_friction`:
    ``sign(c) min(k_f r_c |c| kappa, sigma_cap(c)) / E_core`` with
    ``c = cos(phi - phi_bend)``, the per-wire law of
    :func:`~cablexsec.analysis.stress.compute_stress` applied to the cores.
    """
    beta = core_angle(cable, layer)
    r_c = float(layer.r_offset_m or 0.0)
    phases = 2.0 * np.pi * np.arange(layer.n_cores) / layer.n_cores
    projection = kappa_x * np.cos(phases) + kappa_y * np.sin(phases)
    stuck = r_c * math.cos(beta) ** 2 * projection
    if cable.core_lay is None:
        return _CoreAxis(
            stuck,
            stuck.copy(),
            "the cores are not laid, so friction holds them to the cable axis "
            "over a uniform bend and the frictional state is the stuck state "
            "(declare the core lay of the cable for a frictional state)",
        )
    try:
        model = build_core_friction(
            cable,
            dataclasses.replace(spec, strand_slip=False),
            max(section_state.tension_N, 0.0),
            section_state=section_state,
        )
    except (CableSchemaError, CableInputError) as exc:
        return _CoreAxis(
            stuck,
            None,
            f"the core friction law does not apply ({exc}); the stuck state is "
            "assessed instead",
        )
    population = model.population
    kappa = math.hypot(kappa_x, kappa_y)
    if kappa == 0.0:
        frictional = np.zeros(layer.n_cores)
        slipping = 0
    else:
        phi_bend = math.atan2(kappa_y, kappa_x)
        stations = np.arange(layer.n_cores) * model.spec.n_stations_per_core
        cap = wire_slip_stress_cap(population, phi_bend)[stations]
        cosine = np.cos(phases - phi_bend)
        stick = population.k_f * population.r_helix_m * np.abs(cosine) * kappa
        frictional = np.sign(cosine) * np.minimum(stick, cap) / population.E_wire_Pa
        slipping = int(np.count_nonzero(stick >= cap))
    return _CoreAxis(
        stuck,
        frictional,
        f"core friction coefficient {model.mu_effective:.3g}, stick coefficient "
        f"{model.xi:.3g}, contact line load {model.loads.total_N_per_m:.4g} N/m per "
        f"core; {slipping} of {layer.n_cores} cores slipping in this section",
    )


def given_core_axes(
    cable: CableData,
    section_state: SectionState,
    kappa_x: float,
    kappa_y: float,
    spec: CoreFrictionSpec,
    friction: FrictionField | None,
) -> dict[float, _CoreAxis]:
    """The core axes whose frictional state a load history gives, by offset.

    Empty without a core field in ``friction`` or for cores that are not
    laid (their frictional state is the stuck state). The stuck state is
    that of :func:`_core_axis`; the frictional strain of each core is read
    from the field at its axis.
    """
    axes: dict[float, _CoreAxis] = {}
    if friction is None or friction.cores is None or cable.core_lay is None:
        return axes
    for layer in cable.layers:
        if not isinstance(layer, CylindricalOffsetLayer) or not layer.r_offset_m:
            continue
        offset = float(layer.r_offset_m)
        if offset in axes or layer.n_cores < 2:
            continue
        axis = _core_axis(cable, layer, section_state, kappa_x, kappa_y, spec)
        phases = 2.0 * np.pi * np.arange(layer.n_cores) / layer.n_cores
        strain = friction.core_strain(phases)
        assert strain is not None
        axes[offset] = _CoreAxis(
            axis.stuck,
            np.asarray(strain, dtype=np.float64),
            "the frictional state carried over from a load history (the core "
            "friction law followed over it)",
        )
    return axes


def _resolved_core_states(
    layer: CylindricalOffsetLayer,
    section_state: SectionState,
    kappa_x: float,
    kappa_y: float,
    n_points: int,
    axis: _CoreAxis,
) -> dict[str, list[np.ndarray]]:
    """Wire-fibre stresses over a complete internal wire lay, per core state.

    Chunks run over (ring, core); the mean of a chunk is the net-section
    stress of that ring in that core. The free state excludes the core and
    ring parallel axes; the stuck state includes both; the frictional state
    takes the core axis from the friction law and holds the internal rings
    to the core axis. These are kinematic assumptions, not a prediction of
    internal strand slip. Assembly lay uses a fixed-radius helical-rod bending
    transfer.
    """
    phi = np.linspace(0, 2 * math.pi, n_points, endpoint=False)
    beta = core_angle(section_state.cable, layer)
    e = section_state.youngs_modulus_Pa[layer.name]
    out: dict[str, list[np.ndarray]] = {}
    for state in CORE_SLIP_STATES:
        axis_strain = axis.strain(state)
        if axis_strain is None:
            continue
        chunks: list[np.ndarray] = []
        for ring, mean in zip(
            layer.strand_layers,
            section_state.layers[layer.name].strand_stress_Pa,
            strict=True,
        ):
            c2 = math.cos(ring.lay_angle_rad) ** 2
            for core in range(layer.n_cores):
                phase = 2 * math.pi * core / layer.n_cores
                local_x, local_y = core_bending_components(
                    beta, phase, kappa_x, kappa_y
                )
                amp = local_bending_amplitude(
                    math.hypot(local_x, local_y),
                    phi,
                    e,
                    ring.d_wire_m,
                    ring.lay_angle_rad,
                    math.atan2(local_y, local_x),
                )
                ring_strain = (
                    c2
                    * ring.r_helix_m
                    * (local_x * np.cos(phi) + local_y * np.sin(phi))
                )
                if state == "free":
                    centre = np.full(n_points, mean)
                else:
                    centre = mean + e * (c2 * axis_strain[core] + ring_strain)
                chunks.append(np.concatenate([centre - amp, centre + amp]))
        out[state] = chunks
    return out


def _laid_annulus_states(
    layer: CylindricalOffsetLayer,
    section_state: SectionState,
    kx: float,
    ky: float,
    n_points: int,
    axis: _CoreAxis,
    radius: float | None = None,
    own: float = 1.0,
) -> dict[str, list[np.ndarray]]:
    """Stress of an unresolved laid annulus per core state; one chunk per core.

    The points lie on the circle of ``radius`` about each core axis (default
    the outer radius), at angles in the core's radial and tangential frame
    (from the direction away from the cable axis), the frame of
    :func:`~cablexsec.mechanics.core_lay.core_bending_components`. ``own``
    scales the bending about the core's own axis (:func:`own_bending_ratio`).
    """
    beta = core_angle(section_state.cable, layer)
    theta = np.linspace(0, 2 * math.pi, n_points, endpoint=False)
    if radius is None:
        radius = layer.r_outer_m or 0.0
    lever = own * radius
    local_x, local_y = lever * np.cos(theta), lever * np.sin(theta)
    c2 = axial_cosine(layer) ** 2
    e = section_state.youngs_modulus_Pa[layer.name]
    mean = section_state.layers[layer.name].longitudinal_stress_Pa
    out: dict[str, list[np.ndarray]] = {}
    for state in CORE_SLIP_STATES:
        axis_strain = axis.strain(state)
        if axis_strain is None:
            continue
        chunks: list[np.ndarray] = []
        for i in range(layer.n_cores):
            phase = 2 * math.pi * i / layer.n_cores
            bx, by = core_bending_components(beta, phase, kx, ky)
            bending = e * c2 * (bx * local_x + by * local_y)
            chunks.append(mean + bending + e * c2 * axis_strain[i])
        out[state] = chunks
    return out


def _straight_offset_states(
    layer: CylindricalOffsetLayer,
    e_layer: float,
    axial_strain: float,
    kappa_x: float,
    kappa_y: float,
    thermal_strain: float,
    n_points: int,
    radius: float | None = None,
    own: float = 1.0,
) -> dict[str, list[np.ndarray]]:
    """Aligned offset cores (no core lay): one chunk per core.

    Same strain-gradient convention as the physical-wire equations:
    ``epsilon_s(x, y) = epsilon_0 + b_x x + b_y y``; ``own`` scales the lever
    about each core's own axis (:func:`own_bending_ratio`).
    """
    x_stick, y_stick, x_local, y_local = _cylindrical_points(
        layer, n_points, radius, own
    )
    projection = axial_cosine(layer) ** 2
    stick = e_layer * (
        projection * (axial_strain + kappa_x * x_stick + kappa_y * y_stick)
        - thermal_strain
    )
    local = e_layer * (
        projection * (axial_strain + kappa_x * x_local + kappa_y * y_local)
        - thermal_strain
    )
    stick_chunks = list(np.split(stick, layer.n_cores))
    return {
        "stuck": stick_chunks,
        "frictional": [chunk.copy() for chunk in stick_chunks],
        "free": list(np.split(local, layer.n_cores)),
    }


def _concentric_states(
    layer: CylindricalLayer,
    e_layer: float,
    axial_strain: float,
    kappa_x: float,
    kappa_y: float,
    thermal_strain: float,
    n_points: int,
    radius: float | None = None,
    own: float = 1.0,
) -> dict[str, list[np.ndarray]]:
    """A concentric layer: every core state is the same plane-section field.

    ``own`` scales the lever about the cable axis, the layer's own axis
    (:func:`own_bending_ratio`).
    """
    x, y, _, _ = _cylindrical_points(layer, n_points, radius, own)
    projection = axial_cosine(layer) ** 2
    stress = e_layer * (
        projection * (axial_strain + kappa_x * x + kappa_y * y) - thermal_strain
    )
    return {state: [stress.copy()] for state in CORE_SLIP_STATES}


@dataclass(frozen=True)
class _Continuum:
    """Stress chunks of a concentric or offset layer, per core slip state.

    ``kind`` says how the chunks are laid out: ``unloaded`` (one zero),
    ``concentric`` (one chunk round the cable axis), ``straight`` and
    ``laid`` (one chunk per core, round the core axis), ``resolved`` (one
    chunk per wire ring and core: the wire-centre stress less, then plus, the
    local bending amplitude at the ring's sampled angles).
    """

    chunks: dict[str, list[np.ndarray]]
    basis: str
    assessed: str
    loaded: bool
    kind: str


def _continuum_chunks(
    cable: CableData,
    layer: CylindricalLayer | CylindricalOffsetLayer,
    section_state: SectionState,
    stiffness: dict[str, float],
    kappa_x: float,
    kappa_y: float,
    thermal_strain: float,
    n_points: int,
    axes: dict[float, _CoreAxis],
    spec: CoreFrictionSpec,
    core_state: str,
    radius: float | None = None,
) -> _Continuum:
    """The stress of a concentric or offset layer on the circle of ``radius``.

    ``thermal_strain`` is the free thermal strain the layer's stress
    excludes; ``axes`` caches the core axes by offset. ``radius`` (default the
    outer radius) has no effect on a resolved core, whose points lie on its
    wire rings. The bending about the layer's own axis is on the basis of its
    bending stiffness (:func:`own_bending_ratio`).
    """
    e_layer = section_state.youngs_modulus_Pa[layer.name]
    own = own_bending_ratio(layer)
    own_words = _own_basis(layer, own)
    assessed = ""
    loaded = stiffness.get(layer.name, 0.0) > 0.0 and e_layer > 0.0
    if not loaded:
        return _Continuum(
            {state: [np.zeros(1)] for state in CORE_SLIP_STATES},
            "not assigned structural load",
            "",
            False,
            "unloaded",
        )
    if isinstance(layer, CylindricalLayer):
        return _Continuum(
            _concentric_states(
                layer,
                e_layer,
                section_state.axial_strain,
                kappa_x,
                kappa_y,
                thermal_strain,
                n_points,
                radius,
                own,
            ),
            "concentric layer bending about the cable axis; the core slip "
            "states coincide (plane-section kinematic assumptions)"
            + (f"; {own_words}" if own_words else ""),
            "",
            True,
            "concentric",
        )
    if not layer.r_offset_m and not layer.strand_layers:
        return _Continuum(
            _straight_offset_states(
                layer,
                e_layer,
                section_state.axial_strain,
                kappa_x,
                kappa_y,
                thermal_strain,
                n_points,
                radius,
                own,
            ),
            "core on the cable axis: the core slip states coincide "
            "(plane-section kinematic assumptions)"
            + (f"; {own_words}" if own_words else ""),
            "",
            True,
            "straight",
        )
    offset = float(layer.r_offset_m or 0.0)
    if offset not in axes:
        axes[offset] = _core_axis(cable, layer, section_state, kappa_x, kappa_y, spec)
    axis = axes[offset]
    # Without the friction law, or for cores that are not laid (whose
    # frictional state is the stuck state), the stuck state is named.
    assessed = (
        "stuck"
        if core_state == "frictional"
        and (axis.frictional is None or cable.core_lay is None)
        else core_state
    )
    if layer.strand_layers:
        chunks = _resolved_core_states(
            layer, section_state, kappa_x, kappa_y, n_points, axis
        )
        basis = (
            "resolved internal wires of a laid core: stuck, frictional "
            "and free kinematic assumptions of the core axis, the "
            "internal rings held to the core axis except in the free "
            "state, without an internal-slip solution"
        )
        kind = "resolved"
    elif cable.core_lay is not None:
        chunks = _laid_annulus_states(
            layer, section_state, kappa_x, kappa_y, n_points, axis, radius, own
        )
        basis = (
            "laid core: stuck, frictional and free kinematic assumptions "
            "of the core axis"
        )
        kind = "laid"
    else:
        chunks = _straight_offset_states(
            layer,
            e_layer,
            section_state.axial_strain,
            kappa_x,
            kappa_y,
            thermal_strain,
            n_points,
            radius,
            own,
        )
        basis = "aligned cores: stuck and free kinematic assumptions of the core axis"
        kind = "straight"
    if own_words:
        basis += f"; {own_words}"
    basis += f"; {axis.note}; extremes and utilisation in the " + (
        _STATE_WORDS[assessed] + " state"
    )
    return _Continuum(chunks, basis, assessed, True, kind)


def _following(
    cable: CableData,
    layer: CylindricalLayer | CylindricalOffsetLayer,
    section_state: SectionState,
) -> SectionState:
    """The section state with ``layer`` stressed at the strain it follows.

    A layer outside the load path has no longitudinal stress in the section
    state. It is given the stress the bounded section model gives a
    load-bearing layer at the same place: ``E ((cos a cos b)^2 eps_0 -
    eps_T)``, with ``a`` its own lay, ``b`` the core lay and ``eps_T`` its
    free thermal strain (per wire ring in a resolved core). Its axial force
    stays zero, so the core friction loads and the equilibrium of the
    section are those of the model.
    """
    name = layer.name
    modulus = section_state.youngs_modulus_Pa[name]
    free = section_state.free_thermal_strain[name]
    strain = section_state.axial_strain
    assembly = (
        math.cos(core_angle(cable, layer))
        if isinstance(layer, CylindricalOffsetLayer)
        else 1.0
    )
    normal = section_state.layers[name]
    if isinstance(layer, CylindricalOffsetLayer) and layer.strand_layers:
        rings = tuple(
            modulus * ((math.cos(ring.lay_angle_rad) * assembly) ** 2 * strain - free)
            for ring in layer.strand_layers
        )
        areas = [ring.area_m2 for ring in layer.strand_layers]
        mean = sum(s * a for s, a in zip(rings, areas, strict=True)) / sum(areas)
        followed = dataclasses.replace(
            normal, longitudinal_stress_Pa=mean, strand_stress_Pa=rings
        )
    else:
        mean = modulus * ((axial_cosine(layer) * assembly) ** 2 * strain - free)
        followed = dataclasses.replace(normal, longitudinal_stress_Pa=mean)
    return dataclasses.replace(
        section_state, layers={**section_state.layers, name: followed}
    )


def kinematic_chunks(
    cable: CableData,
    layer: CylindricalLayer | CylindricalOffsetLayer,
    section_state: SectionState,
    stiffness: dict[str, float],
    kappa_x: float,
    kappa_y: float,
    thermal_strain: float,
    n_points: int,
    axes: dict[float, _CoreAxis],
    spec: CoreFrictionSpec,
    core_state: str,
    radius: float | None = None,
) -> tuple[_Continuum, SectionState, dict[str, float]] | None:
    """The chunks of a layer outside the load path, at the strain it follows.

    The same formulas as a load-bearing layer at the same place
    (:func:`_following`), marked :data:`KINEMATIC` in the basis; also the
    section state and stiffnesses that give them (for other radii). ``None``
    for a layer that has no Young's modulus in the model.
    """
    if not section_state.youngs_modulus_Pa.get(layer.name, 0.0) > 0.0:
        return None
    followed = _following(cable, layer, section_state)
    rigidity = {**stiffness, layer.name: 1.0}
    outer = _continuum_chunks(
        cable,
        layer,
        followed,
        rigidity,
        kappa_x,
        kappa_y,
        thermal_strain,
        n_points,
        axes,
        spec,
        core_state,
        radius,
    )
    return (
        dataclasses.replace(outer, basis=f"{KINEMATIC}; {outer.basis}"),
        followed,
        rigidity,
    )


def _radial_equivalent(
    normal: LayerNormalState,
    section_state: SectionState,
    modulus: float,
    kappa: float,
    scf: float,
    inner: float,
    outer: float,
) -> float:
    """Largest von Mises stress of a layer of the coupled radial section.

    The coupled normal stress with elastic bending about the cable axis
    superposed, at the inner and outer radius on both sides of the bend.
    """
    equivalent = 0.0
    for radius in (inner, outer):
        inverse = normal.radial_stress_inverse_Pa_m2 / radius**2 if radius else 0.0
        radial = normal.radial_stress_constant_Pa - inverse
        hoop = normal.radial_stress_constant_Pa + inverse
        shear = normal.shear_modulus_Pa * radius * section_state.twist_per_m
        for sign in (-1, 1):
            axial = (
                normal.longitudinal_stress_Pa + sign * scf * modulus * kappa * radius
            )
            vm = math.sqrt(
                0.5
                * ((axial - radial) ** 2 + (radial - hoop) ** 2 + (hoop - axial) ** 2)
                + 3 * shear**2
            )
            equivalent = max(equivalent, vm)
    return equivalent


def compute_component_response(
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
    section_state: SectionState | None = None,
    torque_Nm: float = 0.0,
    prescribed_twist_per_m: float | None = 0.0,
    armour: bool = True,
    friction: FrictionField | None = None,
) -> ComponentResponse:
    """Compute the longitudinal response of every cross-section component.

    Manufacturing clamping enters through the declared cable inputs. The
    default bounded contact approximation also accepts offset cores. The
    optional coupled radial calculation requires a genuinely concentric
    section; it must be selected explicitly with contact_model="qp".

    Parameters
    ----------
    mu : float, optional
        Armour friction coefficient, as in
        :func:`~cablexsec.analysis.stress.compute_stress` (default: the armour
        material's ``mu_on_hdpe``).
    core_state : {"frictional", "stuck", "free"}
        Slip state of the laid cores in which the extremes, strains and
        utilisation of offset components are assessed (module notes). The
        stress ranges of all three states are always returned. When the core
        friction law cannot be built the frictional state falls back to the
        stuck state and the component's ``basis`` says why.
    core_friction : CoreFrictionSpec, optional
        Friction inputs of the laid cores (default :class:`CoreFrictionSpec`,
        friction coefficient 0.2 for every core contact,
        [MenardCartraud2023] Section 5). Internal strand slip of the core law
        is not used here.
    armour : bool
        Recover the armour wire stresses (default). ``False`` leaves the
        armour layers not assessed, for a caller that needs the other
        components only (:func:`component_usage_curve`).
    friction : FrictionField, optional
        The friction a load history left on the armour wires and the laid
        cores, in place of the slip fronts at the curvature: the armour wire
        stresses (:func:`~cablexsec.analysis.stress.compute_stress`) and the
        frictional state of the laid cores take it.
    """
    require_fully_specified(cable, "compute_component_response")
    require_finite(
        [kappa_x, kappa_y, tension_N, T_ref_C, p_int_Pa, p_ext_Pa or 0.0],
        "component-response inputs",
    )
    if n_points < 8:
        raise CableInputError("n_points must be at least 8")
    if core_state not in CORE_SLIP_STATES:
        raise CableInputError(
            f"core_state must be one of {', '.join(CORE_SLIP_STATES)}"
        )
    if mu is not None and (not math.isfinite(mu) or mu < 0.0):
        raise CableInputError("mu must be finite and non-negative")
    spec = core_friction if core_friction is not None else CoreFrictionSpec()

    if section_state is None:
        section_state = solve_section_state(
            cable,
            tension_N=tension_N,
            torque_Nm=torque_Nm,
            prescribed_twist_per_m=prescribed_twist_per_m,
            p_int_Pa=p_int_Pa,
            p_ext_Pa=p_ext_Pa,
            thermal=thermal,
            T_ref_C=T_ref_C,
            contact_model=contact_model or "bounded",
        )
    else:
        section_state.require_cable(cable)
        if (
            tension_N != 0
            or thermal is not None
            or p_int_Pa != 0
            or p_ext_Pa is not None
            or torque_Nm != 0
            or prescribed_twist_per_m != 0
            or (
                contact_model is not None
                and contact_model != section_state.contact_model
            )
        ):
            raise CableInputError(
                "supply a section state or separate section loads, not both"
            )
    tension_N = section_state.tension_N
    p_int_Pa, p_ext_Pa = (
        section_state.internal_pressure_Pa,
        section_state.external_pressure_Pa,
    )
    stiffness = {
        name: value * section_state.modulus_scale[name]
        for name, value in _axial_stiffnesses(cable).items()
    }
    ea_total = sum(stiffness.values())
    if ea_total <= 0.0:
        raise CableInputError("cable has no load-carrying axial component")

    axial_strain = section_state.axial_strain
    mechanical_strain = axial_strain
    if any(section_state.free_thermal_strain.values()):
        # Define a thermal increment by removing free expansion at the SAME
        # moduli, mechanical loads and twist condition. Contact can change,
        # so this is not an assumed linear superposition of independent loads.
        reference = solve_section_state(
            cable,
            tension_N=tension_N,
            torque_Nm=section_state.torque_Nm,
            prescribed_twist_per_m=section_state.prescribed_twist_per_m,
            p_int_Pa=p_int_Pa,
            p_ext_Pa=p_ext_Pa,
            youngs_modulus_Pa=section_state.youngs_modulus_Pa,
            initial_wire_strain=section_state.initial_wire_strain,
            contact_model=section_state.contact_model,
        )
        mechanical_strain = reference.axial_strain
    free_thermal_strain = axial_strain - mechanical_strain

    armour_result = None
    if cable.helical_layers and armour:
        armour_result = compute_stress(
            cable,
            kappa_x,
            kappa_y,
            mu=mu,
            section_state=section_state,
            friction=friction,
        )

    kappa = math.hypot(kappa_x, kappa_y)
    axes = given_core_axes(cable, section_state, kappa_x, kappa_y, spec, friction)
    states: list[ComponentState] = []
    for layer in cable.layers:
        temperature = section_state.temperature_C[layer.name]
        alpha = _property(layer.material, "alpha_T_K")
        delta_T = (
            section_state.free_thermal_strain[layer.name] / alpha if alpha else 0.0
        )
        component_type = _component_type(layer)

        if isinstance(layer, HelicalLayer) and (
            armour_result is None
            or layer.name not in armour_result.sigma_vm_envelope_Pa
        ):
            # Non-structural helical layer (e.g. an interlocked pressure
            # armour): compute_stress assigns it no longitudinal load.
            states.append(
                ComponentState(
                    name=layer.name,
                    component_type=component_type,
                    material=layer.material,
                    strain_min=0.0,
                    strain_max=0.0,
                    sigma_local_min_Pa=0.0,
                    sigma_local_max_Pa=0.0,
                    sigma_stick_min_Pa=0.0,
                    sigma_stick_max_Pa=0.0,
                    sigma_equivalent_max_Pa=0.0,
                    temperature_C=temperature,
                    mechanical_utilization=math.nan,
                    failure_mode="not assessed",
                    basis="non-structural helical layer; no longitudinal wire "
                    "stress is assigned"
                    if armour
                    else "armour wire stresses not recovered in this call",
                    sigma_frictional_min_Pa=0.0,
                    sigma_frictional_max_Pa=0.0,
                )
            )
            continue

        if isinstance(layer, HelicalLayer) and armour_result is not None:
            e_wire = section_state.youngs_modulus_Pa[layer.name]
            new_vm = armour_result.sigma_vm_envelope_Pa[layer.name]
            sigma_min = float(np.min(armour_result.sigma_perimeter_min_Pa[layer.name]))
            sigma_max = float(np.max(armour_result.sigma_perimeter_max_Pa[layer.name]))
            equivalent = float(np.max(new_vm))
            strain_min = sigma_min / e_wire if e_wire > 0.0 else 0.0
            strain_max = sigma_max / e_wire if e_wire > 0.0 else 0.0
            # First yield on the compute_capacity basis: the extreme
            # perimeter longitudinal stress |mean| + local-bending amplitude.
            utilization, mode = _yield_metric(layer, sigma_min, sigma_max)
            contact_screen, _ = _yield_metric(layer, -equivalent, equivalent)
            states.append(
                ComponentState(
                    name=layer.name,
                    component_type=component_type,
                    material=layer.material,
                    strain_min=strain_min,
                    strain_max=strain_max,
                    sigma_local_min_Pa=sigma_min,
                    sigma_local_max_Pa=sigma_max,
                    sigma_stick_min_Pa=sigma_min,
                    sigma_stick_max_Pa=sigma_max,
                    sigma_equivalent_max_Pa=equivalent,
                    temperature_C=temperature,
                    mechanical_utilization=utilization,
                    failure_mode=mode,
                    basis="armour wire slip front; the utilisation is first yield "
                    "of the largest longitudinal stress round a wire, the basis "
                    "of the capacity envelope; the axial and crossing-contact "
                    "screening indicator is reported separately and is neither a "
                    "coincident contact stress nor a first-yield check",
                    contact_screen_utilization=contact_screen,
                    sigma_frictional_min_Pa=sigma_min,
                    sigma_frictional_max_Pa=sigma_max,
                )
            )
            continue

        if section_state.radial is not None:
            normal = section_state.layers[layer.name]
            if isinstance(layer, CylindricalShellLayer):
                assert layer.r_mid_m is not None and layer.t_m is not None
                inner, outer = (
                    layer.r_mid_m - layer.t_m / 2,
                    layer.r_mid_m + layer.t_m / 2,
                )
                # Root stress SCF * f_corr * E * kappa * r, as shell_wall_stress.
                scf = layer.scf * layer.f_corrugation
            else:
                assert isinstance(layer, (CylindricalLayer, CylindricalOffsetLayer))
                inner, outer = (
                    float(layer.r_inner_m or 0.0),
                    float(layer.r_outer_m or 0.0),
                )
                # Bending about the cable axis, the layer's own axis, on the
                # basis of its bending stiffness (a stranded conductor).
                scf = own_bending_ratio(layer)
            modulus = section_state.youngs_modulus_Pa[layer.name]
            bend = scf * modulus * kappa * outer
            sigma_min, sigma_max = (
                normal.longitudinal_stress_Pa - bend,
                normal.longitudinal_stress_Pa + bend,
            )
            equivalent = _radial_equivalent(
                normal, section_state, modulus, kappa, scf, inner, outer
            )
            net_section = normal.longitudinal_stress_Pa
            if isinstance(layer, CylindricalShellLayer):
                net_section = math.nan
                judged = _assessment(
                    layer,
                    component_type,
                    peak_stress_Pa=equivalent,
                    peak_strain=abs(axial_strain) + kappa * outer,
                )
            else:
                judged = _assessment(
                    layer,
                    component_type,
                    peak_stress_Pa=max(abs(sigma_min), abs(sigma_max)),
                    net_section_Pa=net_section,
                    peak_strain=(
                        max(abs(sigma_min), abs(sigma_max)) / modulus
                        if modulus > 0.0
                        else math.nan
                    ),
                )
            utilization, mode = judged.utilisation, judged.mode
            states.append(
                ComponentState(
                    name=layer.name,
                    component_type=component_type,
                    material=layer.material,
                    strain_min=axial_strain - kappa * outer,
                    strain_max=axial_strain + kappa * outer,
                    sigma_local_min_Pa=sigma_min,
                    sigma_local_max_Pa=sigma_max,
                    sigma_stick_min_Pa=sigma_min,
                    sigma_stick_max_Pa=sigma_max,
                    sigma_equivalent_max_Pa=equivalent,
                    temperature_C=temperature,
                    mechanical_utilization=utilization,
                    failure_mode=mode,
                    basis="coupled normal section stress with superposed elastic "
                    "bending about the cable axis"
                    + (
                        f"; {_own_basis(layer, scf)}"
                        if not isinstance(layer, CylindricalShellLayer) and scf < 1.0
                        else ""
                    ),
                    sigma_frictional_min_Pa=sigma_min,
                    sigma_frictional_max_Pa=sigma_max,
                    sigma_net_section_max_Pa=net_section,
                    utilisations=judged.utilisations,
                    limits=judged.limits,
                )
            )
            continue

        if isinstance(layer, CylindricalShellLayer):
            assert layer.r_mid_m is not None and layer.t_m is not None
            extremes = [
                shell_wall_stress(
                    layer,
                    sign * kappa,
                    axial_strain,
                    ea_total,
                    p_int_Pa=p_int_Pa,
                    p_ext_Pa=p_ext_Pa,
                    delta_T_K=delta_T,
                    youngs_modulus_Pa=section_state.youngs_modulus_Pa[layer.name],
                )
                for sign in (-1.0, 1.0)
            ]
            sigma_min = min(state.sigma_longitudinal_Pa for state in extremes)
            sigma_max = max(state.sigma_longitudinal_Pa for state in extremes)
            equivalent = max(state.sigma_von_mises_Pa for state in extremes)
            judged = _assessment(
                layer,
                component_type,
                peak_stress_Pa=equivalent,
                peak_strain=abs(axial_strain) + kappa * layer.r_mid_m,
            )
            utilization, mode = judged.utilisation, judged.mode
            transfer = layer.scf * layer.f_corrugation
            states.append(
                ComponentState(
                    name=layer.name,
                    component_type=component_type,
                    material=layer.material,
                    strain_min=axial_strain - kappa * layer.r_mid_m,
                    strain_max=axial_strain + kappa * layer.r_mid_m,
                    sigma_local_min_Pa=sigma_min,
                    sigma_local_max_Pa=sigma_max,
                    sigma_stick_min_Pa=sigma_min,
                    sigma_stick_max_Pa=sigma_max,
                    sigma_equivalent_max_Pa=equivalent,
                    temperature_C=temperature,
                    mechanical_utilization=utilization,
                    failure_mode=mode,
                    basis="thin-shell sheath bending about the cable axis; the "
                    "stress at the corrugation root is the smooth-tube stress "
                    "times the corrugation stress concentration factor and the "
                    "corrugation stiffness factor; the corrugation stress "
                    "concentration factor times the corrugation stiffness factor "
                    f"is {transfer:.3g} for this sheath"
                    + (
                        ", so the corrugation does not relieve the smooth-tube "
                        "bending stress"
                        if transfer >= 1.0
                        else ""
                    )
                    + "; the stress utilisation is first yield of the von Mises "
                    "stress at the root",
                    sigma_frictional_min_Pa=sigma_min,
                    sigma_frictional_max_Pa=sigma_max,
                    utilisations=judged.utilisations,
                    limits=judged.limits,
                )
            )
            continue

        if isinstance(layer, HelicalStrandedLayer):
            states.append(
                ComponentState(
                    name=layer.name,
                    component_type=component_type,
                    material=layer.material,
                    strain_min=0.0,
                    strain_max=0.0,
                    sigma_local_min_Pa=0.0,
                    sigma_local_max_Pa=0.0,
                    sigma_stick_min_Pa=0.0,
                    sigma_stick_max_Pa=0.0,
                    sigma_equivalent_max_Pa=0.0,
                    temperature_C=temperature,
                    mechanical_utilization=math.nan,
                    failure_mode="not assessed",
                    basis="nested-strand stress path not yet integrated",
                )
            )
            continue

        assert isinstance(layer, CylindricalLayer | CylindricalOffsetLayer)
        e_layer = section_state.youngs_modulus_Pa[layer.name]
        thermal_strain = section_state.free_thermal_strain[layer.name]
        continuum = _continuum_chunks(
            cable,
            layer,
            section_state,
            stiffness,
            kappa_x,
            kappa_y,
            alpha * delta_T,
            n_points,
            axes,
            spec,
            core_state,
        )
        kinematic = False
        if not continuum.loaded:
            followed = kinematic_chunks(
                cable,
                layer,
                section_state,
                stiffness,
                kappa_x,
                kappa_y,
                alpha * delta_T,
                n_points,
                axes,
                spec,
                core_state,
            )
            if followed is not None:
                continuum, kinematic = followed[0], True
        chunks, basis = continuum.chunks, continuum.basis
        assessed = continuum.assessed
        loaded = continuum.loaded or kinematic
        fields = {state: np.concatenate(parts) for state, parts in chunks.items()}
        net = {
            state: max(float(part.sum()) / part.size for part in parts)
            for state, parts in chunks.items()
        }
        state_key = assessed or "stuck"
        field = fields[state_key]
        sigma_min = float(np.min(field))
        sigma_max = float(np.max(field))
        peak = max(abs(sigma_min), abs(sigma_max))
        if loaded:
            judged = _assessment(
                layer,
                component_type,
                peak_stress_Pa=peak,
                net_section_Pa=net[state_key],
                peak_strain=peak / e_layer if e_layer > 0.0 else math.nan,
            )
        else:
            judged = Assessment(math.nan, "not assessed", (), "")
        utilization, mode = judged.utilisation, judged.mode
        if component_type == "conductor" and loaded:
            basis += (
                "; the net-section stress is the conductor's axial force over "
                "its area, checked against the tensile strength; the 0.2 % "
                "proof stress is checked at the extreme wire fibre"
            )
        frictional = fields.get("frictional")
        states.append(
            ComponentState(
                name=layer.name,
                component_type=component_type,
                material=layer.material,
                strain_min=sigma_min / e_layer + thermal_strain if loaded else 0.0,
                strain_max=sigma_max / e_layer + thermal_strain if loaded else 0.0,
                sigma_local_min_Pa=float(np.min(fields["free"])),
                sigma_local_max_Pa=float(np.max(fields["free"])),
                sigma_stick_min_Pa=float(np.min(fields["stuck"])),
                sigma_stick_max_Pa=float(np.max(fields["stuck"])),
                sigma_equivalent_max_Pa=max(abs(sigma_min), abs(sigma_max)),
                temperature_C=temperature,
                mechanical_utilization=utilization,
                failure_mode=mode,
                basis=basis,
                sigma_frictional_min_Pa=math.nan
                if frictional is None
                else float(np.min(frictional)),
                sigma_frictional_max_Pa=math.nan
                if frictional is None
                else float(np.max(frictional)),
                sigma_net_section_max_Pa=net[state_key],
                core_state=assessed,
                utilisations=judged.utilisations,
                limits=judged.limits,
                kinematic=kinematic,
            )
        )

    # Layers outside the load path carry a kinematic upper bound; the
    # critical component is a component of the load path.
    assessed_states = [
        state
        for state in states
        if math.isfinite(state.mechanical_utilization) and not state.kinematic
    ]
    if assessed_states:
        critical = max(assessed_states, key=lambda state: state.mechanical_utilization)
        critical_name = critical.name
        critical_utilization = critical.mechanical_utilization
    else:
        critical_name = "not assessed"
        critical_utilization = math.nan

    return ComponentResponse(
        states=tuple(states),
        axial_strain=axial_strain,
        mechanical_axial_strain=mechanical_strain,
        free_thermal_axial_strain=free_thermal_strain,
        tension_N=tension_N,
        kappa_x_1_m=kappa_x,
        kappa_y_1_m=kappa_y,
        critical_component=critical_name,
        critical_utilization=critical_utilization,
        section_state=section_state,
        core_state=core_state,
        provenance=Provenance(
            method="calibration-free longitudinal stress of every cable component "
            "with thermoelastic axial-force equilibrium; laid cores in the stuck, "
            "frictional and free slip states",
            citations=(
                "[WitzTan1992a] strain compatibility and helical load sharing",
                "[Saevik2011] armour stress and contact",
                "[Papailiou1997] and [Yun2020] slip cap and stick coefficient of "
                "the laid cores",
                "[MenardCartraud2023] Eq. 14 bending of a laid core",
                "[CIGRE TB 623] metallic-sheath shell mechanics",
            ),
            gates=(
                "zero stress for uniform free thermal expansion",
                "axial force recovered at zero curvature",
                "offset-component local and fully compatible fields coincide "
                "at zero offset",
            ),
            limitations=(
                "the stuck and free core states are two prescribed kinematic "
                "assumptions; the frictional state holds the core contact loads "
                "of the straight section and is not a resolved inter-component "
                "contact solution",
                "the conductor is checked against its 0.2 % proof stress at the "
                "extreme wire fibre (strand basis of its bending stiffness) and "
                "against its tensile strength on the net section; stresses are "
                "elastic and are not limited by yielding",
                "polymers are checked against an allowable longitudinal strain "
                "where their material gives one; multiaxial failure is not "
                "assessed",
                "a layer outside the load path (a loose screen) takes the stress "
                "of the strain it follows, an upper bound for a screen that slips",
                "a utilisation that is not reported is an explicit missing limit "
                "(no limit given), not zero",
                "armour utilisation is first yield of the longitudinal wire "
                "stress without contact stresses; the separate contact screening "
                "indicator is neither a yield nor a fretting assessment",
                "sheath utilisation includes the declared corrugation stress "
                "concentration: it is first local yield at the corrugation root, "
                "not a failure of the sheath",
                "contact and shell fidelity depend on measured manufacturing "
                "geometry and independently specified initial state",
            ),
        ),
    )


# ----------------------------------------------------------------------------- #
# Utilisation of the components against curvature (capacity envelopes)         #
# ----------------------------------------------------------------------------- #


def capacity_utilisation(
    state: ComponentState, factor: float = 1.0
) -> tuple[float, str]:
    """The utilisation of a component that bounds the capacity of the cable.

    Every limit of the component counts, except the yield stress of a
    metallic screen or sheath (a metal that is not the conductor): a water
    barrier or screen is made to deform plastically in bending (a lead sheath
    creeps; an aluminium laminate foil withstands bending only by its bonding
    to the polymer, [Worzyk2009] Sec. 2.3.2), so its first yield, reported by
    :func:`compute_component_response` as a screening check, does not bound
    the capacity; its allowable strain does, where its material gives one.
    The armour is checked elsewhere. ``factor`` is a design utilisation
    factor: the yield and strain utilisations are divided by it, the
    net-section fracture of a conductor is not (characteristic, as the
    conductor-fracture mechanism of the capacity envelope). Returns the
    utilisation and the governing limit, NaN and ``"no limit given"`` when
    none counts.
    """
    if not (math.isfinite(factor) and 0.0 < factor <= 1.0):
        raise CableInputError("factor must lie in (0, 1]")
    values = {
        mode: value if mode == "tensile fracture" else value / factor
        for mode, value in state.utilisations
    }
    if state.component_type != "conductor" and _is_metal(state.material):
        values = {mode: v for mode, v in values.items() if mode == "strain limit"}
    if not values:
        return math.nan, "no limit given"
    mode = max(values, key=lambda key: values[key])
    return values[mode], mode


@dataclass(frozen=True)
class ComponentUsageCurve:
    """Utilisation of the non-armour components against curvature at one tension.

    ``usage[c, k]`` is the utilisation of component ``names[c]`` against all
    its limits at the curvature ``kappa_1_m[k]`` (:func:`capacity_utilisation`
    with the design factor ``factor``; characteristic at one),
    ``modes[c][k]`` its governing limit, the largest over the bending
    directions of :func:`bending_directions`. Between the tabulated
    curvatures (:func:`usage_curvature_grid`) the utilisation is linear;
    beyond the last it is extrapolated with the slope of the last interval
    (every stress of the components is linear in curvature once the core
    friction has capped). ``kinematic``
    names the layers outside the load path, whose kinematic utilisation is
    not part of the curve.
    """

    tension_N: float
    kappa_1_m: np.ndarray
    names: tuple[str, ...]
    usage: np.ndarray
    modes: tuple[tuple[str, ...], ...]
    kinematic: tuple[str, ...] = ()
    factor: float = 1.0

    @property
    def assessed(self) -> bool:
        """Whether any component has a limit (an empty curve has none)."""
        return len(self.names) > 0

    def largest(self) -> np.ndarray:
        """The largest utilisation of the components at each curvature."""
        if not self.names:
            return np.zeros(self.kappa_1_m.size)
        return np.asarray(np.max(self.usage, axis=0), dtype=np.float64)

    def at(self, kappa: float) -> float:
        """The largest component utilisation at ``kappa`` (0 without limits)."""
        if not self.names:
            return 0.0
        largest = self.largest()
        grid = self.kappa_1_m
        kappa = abs(float(kappa))
        if kappa <= grid[-1] or grid.size == 1:
            return float(np.interp(kappa, grid, largest))
        slope = (largest[-1] - largest[-2]) / (grid[-1] - grid[-2])
        return float(largest[-1] + max(slope, 0.0) * (kappa - grid[-1]))

    def root(self, level: float) -> float:
        """The smallest curvature at which the largest utilisation reaches ``level``.

        Zero when it is reached at zero curvature, ``inf`` when never.
        """
        if not self.names:
            return math.inf
        largest = self.largest()
        grid = self.kappa_1_m
        if largest[0] >= level:
            return 0.0
        above = np.flatnonzero(largest >= level)
        if above.size:
            k = int(above[0])
            k0, k1, u0, u1 = grid[k - 1], grid[k], largest[k - 1], largest[k]
            return float(k0 + (level - u0) * (k1 - k0) / (u1 - u0))
        if grid.size == 1:
            return math.inf
        slope = (largest[-1] - largest[-2]) / (grid[-1] - grid[-2])
        if slope <= 0.0:
            return math.inf
        return float(grid[-1] + (level - largest[-1]) / slope)

    def governing(self, kappa: float) -> tuple[str, str]:
        """The component and the limit that govern at ``kappa`` (empty without)."""
        if not self.names:
            return "", ""
        grid = self.kappa_1_m
        reach = abs(float(kappa))
        k = (
            min(int(np.searchsorted(grid, reach)), grid.size - 1)
            if math.isfinite(reach)
            else grid.size - 1
        )
        c = int(np.argmax(self.usage[:, k]))
        return self.names[c], self.modes[c][k]


def bending_directions(cable: CableData) -> tuple[float, ...]:
    """Bending directions (physics frame, radians) that bound the utilisation.

    A section of ``n`` laid cores, evenly spaced, repeats every ``2 pi / n``
    and is mirror-symmetric about each core axis and each axis midway between
    two cores, so every bending direction has its like between zero (towards
    a core) and ``pi / n`` (midway between two). The utilisation of the
    components varies smoothly over that sector and its largest value moves
    with the tension and the curvature (towards a core at high tension,
    between the axes at low tension), so the sector is sampled at zero,
    ``pi / (2 n)`` and ``pi / n``. A concentric section bends alike in every
    direction.
    """
    cores = [
        layer.n_cores
        for layer in cable.layers
        if isinstance(layer, CylindricalOffsetLayer) and layer.n_cores > 1
    ]
    if not cores:
        return (0.0,)
    n = max(cores)
    return (0.0, math.pi / (2 * n), math.pi / n)


def usage_curvature_grid(
    kappa_top_1_m: float, n_kappa: int = 14, nodes: Sequence[float] = ()
) -> np.ndarray:
    """Curvatures at which the component utilisation is tabulated.

    Zero, ``n_kappa`` curvatures spaced geometrically from a thousandth of
    ``kappa_top_1_m`` to it (the friction of the cores and the armour slips
    at small curvature, where the utilisation bends most), ``n_kappa`` evenly
    spaced to it, and the ``nodes`` within the range (a design curvature, the
    curvature of the minimum bend radius), so that the utilisation at a node
    is computed, not interpolated.
    """
    top = float(kappa_top_1_m)
    geometric = np.geomspace(top * 1e-3, top, n_kappa)
    even = np.linspace(0.0, top, n_kappa)
    extra = [float(k) for k in nodes if math.isfinite(k) and 0.0 < k <= top]
    return np.unique(np.concatenate(([0.0], geometric, even, extra)))


def component_usage_curve(
    cable: CableData,
    tension_N: float,
    kappa_top_1_m: float,
    *,
    n_kappa: int = 14,
    mu: float | None = None,
    core_state: CoreSlipState = "frictional",
    directions: Sequence[float] | None = None,
    contact_model: str = "bounded",
    include_kinematic: bool = False,
    factor: float = 1.0,
    nodes: Sequence[float] = (),
) -> ComponentUsageCurve:
    """Utilisation of every non-armour component with a limit, against curvature.

    The section is solved once at ``tension_N``; the component response
    (:func:`compute_component_response`) is evaluated at the curvatures of
    :func:`usage_curvature_grid` up to ``kappa_top_1_m`` (with the
    ``nodes``) in each bending direction of ``directions`` (default
    :func:`bending_directions`: towards a core and midway between two), and
    each component takes its largest utilisation over the directions. Layers
    outside the load path are left out unless ``include_kinematic``: their
    stress is the kinematic upper bound of a screen that does not slip. The
    utilisation is :func:`capacity_utilisation`: the yield of metallic
    screens and sheaths does not count, their allowable strain does; with a
    design utilisation ``factor`` below one it is the design utilisation.
    """
    if not (math.isfinite(kappa_top_1_m) and kappa_top_1_m > 0.0):
        raise CableInputError("kappa_top_1_m must be finite and positive")
    if n_kappa < 3:
        raise CableInputError("n_kappa must be at least 3")
    state = solve_section_state(cable, tension_N=tension_N, contact_model=contact_model)
    grid = usage_curvature_grid(kappa_top_1_m, n_kappa, nodes)
    angles = bending_directions(cable) if directions is None else tuple(directions)
    if not angles or not all(math.isfinite(a) for a in angles):
        raise CableInputError("give one finite bending direction or more")
    values: dict[str, list[float]] = {}
    modes: dict[str, list[str]] = {}
    kinematic: set[str] = set()
    for kappa in grid:
        found: dict[str, tuple[float, str]] = {}
        for angle in angles:
            response = compute_component_response(
                cable,
                float(kappa) * math.cos(angle),
                float(kappa) * math.sin(angle),
                mu=mu,
                core_state=core_state,
                section_state=state,
                n_points=72,
                armour=False,
            )
            for item in response.states:
                if item.component_type == "armour wire":
                    continue
                if item.kinematic:
                    kinematic.add(item.name)
                    if not include_kinematic:
                        continue
                value, mode = capacity_utilisation(item, factor)
                if not math.isfinite(value):
                    continue
                if item.name not in found or value > found[item.name][0]:
                    found[item.name] = (value, mode)
        for name, (value, mode) in found.items():
            values.setdefault(name, []).append(value)
            modes.setdefault(name, []).append(mode)
    names = tuple(name for name in values if len(values[name]) == grid.size)
    usage = (
        np.array([values[name] for name in names], dtype=np.float64)
        if names
        else np.zeros((0, grid.size))
    )
    return ComponentUsageCurve(
        tension_N=float(tension_N),
        kappa_1_m=grid,
        names=names,
        usage=usage,
        modes=tuple(tuple(modes[name]) for name in names),
        kinematic=tuple(sorted(kinematic)),
        factor=float(factor),
    )


def core_component_utilisation(
    cable: CableData,
    phi: np.ndarray,
    kappa_x: np.ndarray,
    kappa_y: np.ndarray,
    axial_strain: np.ndarray,
    axis_strain: np.ndarray,
) -> np.ndarray | None:
    """Largest utilisation of the components of a laid core, along a history.

    ``phi`` (stations) are the azimuths of the core axis round the cable,
    ``kappa_x``, ``kappa_y`` and ``axial_strain`` (samples) the section
    history and ``axis_strain`` (stations, samples) the strain of the core
    axis from the core friction law (its friction stress over the core
    modulus). Each load-bearing, unresolved layer of the core with a limit
    takes, as in :func:`compute_component_response`, the centre stress
    ``E c^2 (cos^2(b) eps_0 + eps_axis)`` and the bending about the core's own
    axis ``E c^2 r |kappa_local|`` on the basis of its bending stiffness
    (:func:`own_bending_ratio`), with ``c`` its own lay cosine and ``b`` the
    core lay angle; the utilisation is that of :mod:`component_limits` at the
    extreme fibre. Returns the largest over the layers (stations, samples), or
    ``None`` when no such layer has a limit. Layers outside the load path and
    conductors resolved into wire rings are not included.
    """
    phi = np.asarray(phi, dtype=np.float64)[:, None]
    kx = np.asarray(kappa_x, dtype=np.float64)[None, :]
    ky = np.asarray(kappa_y, dtype=np.float64)[None, :]
    eps0 = np.asarray(axial_strain, dtype=np.float64)[None, :]
    axis = np.asarray(axis_strain, dtype=np.float64)
    best: np.ndarray | None = None
    for layer in cable.layers:
        if not isinstance(layer, CylindricalOffsetLayer) or layer.strand_layers:
            continue
        if not layer.structural:
            continue
        kind = _component_type(layer)
        limits = component_limits(layer.material, kind)
        if not limits:
            continue
        try:
            modulus = library.get_property(
                layer.material, library.axial_modulus_key(layer.material)
            )
        except KeyError:
            continue
        beta = core_angle(cable, layer)
        c, s = math.cos(beta), math.sin(beta)
        radial = kx * np.cos(phi) + ky * np.sin(phi)
        tangential = -kx * np.sin(phi) + ky * np.cos(phi)
        local = np.hypot(
            c * c * math.cos(2 * beta) * radial, c * (1 + s * s) * tangential
        )
        c2 = axial_cosine(layer) ** 2
        lever = own_bending_ratio(layer) * float(layer.r_outer_m or 0.0)
        centre = modulus * c2 * (c * c * eps0 + axis)
        peak = np.abs(centre) + modulus * c2 * lever * local
        usage = np.full(peak.shape, -math.inf)
        for limit in limits:
            if limit.mode == "yielding":
                usage = np.maximum(usage, peak / limit.value)
            elif limit.mode == "tensile fracture":
                usage = np.maximum(usage, np.maximum(centre, 0.0) / limit.value)
            elif limit.mode == "strain limit" and modulus > 0.0:
                usage = np.maximum(usage, peak / modulus / limit.value)
        best = usage if best is None else np.maximum(best, usage)
    return best
