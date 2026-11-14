# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Stick-slip bending law for the laid power cores of a multi-core cable.

Gap R1b. Before this module only the armour wires carried friction in the
bending law; the laid power cores entered as individual bending (full slip)
or as a fully constrained parallel-axis term. Here each laid core is a
helical body of lay angle ``beta`` at pitch-circle radius ``r_c`` that sticks
to or slips against its neighbours, the interstitial filler and the
surrounding ring (bedding or inner sheath), with the same direction-tied
Papailiou cap and elastic--Coulomb return map as the armour wires
(:func:`cablexsec.mechanics.papailiou.wire_slip_history`). Resolved internal
wire rings of a core (stranded conductor, wire screen) slip with the
[KarlsenHusoy2026] operator of :mod:`cablexsec.mechanics.strand_slip`.

Core population
---------------
A core passes through every azimuth along one pitch. The section moment of
the pitch-averaged cores is the integral over helix stations with density
``n_cores / (2 pi)`` (the station integration of [Papailiou1997], as in
:class:`~cablexsec.mechanics.papailiou.PapailiouLayer`). ``n_cores *
n_stations_per_core`` equally spaced stations stand for the three cores; the
stations are mapped onto a :class:`~cablexsec.mechanics.mkappa.HelicalLayerSpec`
with ``E A n_stations = n_cores EA_core`` and the clamping pressure

.. math::

    P = \frac{n_{cores} \, q_{core}}{2 \pi r_c},

where ``q_core`` [N/m of cable] is the sum of the normal contact line loads
on one core. The Papailiou friction-setting stress of a core is then
``sigma_0 A_core = q_core cos(beta) r_c / sin^2 beta`` (``q_core cos(beta)`` is
the line load per core length, as ``q_wire`` of
:class:`~cablexsec.mechanics.papailiou.PapailiouLayer`) and the slip cap
``sigma_0 expm1(mu_eff |sin beta| psi)`` is that of a wire whose two faces
carry ``q_core`` (friction accumulates from the neutral-axis crossing over the
core length ``r_c psi / sin beta``). The effective friction coefficient is the
load-weighted mean ``mu_eff = sum(mu_i q_i) / sum(q_i)`` of the core--core,
core--filler and core--sheath contacts.

With full stick the stations give ``xi sum_l 0.5 EA_l r_c^2 cos^3 beta``,
the core parallel-axis term of
:func:`cablexsec.mechanics.stiffness.bending_stiffness_by_layer` times the
stick coefficient ``xi``. With full slip they carry the frozen friction
stress only; the individual bending of each core is already part of
``EI_min`` (:func:`cablexsec.mechanics.stiffness.ei_min_slip`), so the cores
add their own bending stiffness to both the no-slip and the full-slip value.

Stick coefficient
-----------------
The imposed plane-section displacement varies along the core with wave
number ``sin beta / r_c``. A shear-lag core of axial rigidity ``EA_core`` on
an interface of stiffness ``K_G`` per unit length takes the fraction

.. math::

    \xi = \frac{K_G}{K_G + EA_{core} \sin^2\beta / r_c^2},

which is the [Yun2020] coefficient used for the armour
(:func:`cablexsec.mechanics.yun_coefficient.yun_xi`). ``K_G`` combines in
series the contact (penalty) stiffness ``k_s`` [Pa/m] over the core
perimeter and, optionally, the through-thickness shear of the polymer layers
inside the core (the conductor carries most of ``EA_core`` and is loaded
through the insulation).

Normal loads
------------
``contact_model="solved"``: the trefoil contact forces of the solved radial
state (:class:`cablexsec.mechanics.radial_solution.BundleState`): two
core--core contacts, the filler--core force and the direct core--ring force
of one core. Otherwise the bounded estimate: the inward capstan load of a
core ``N_core tan^2 beta / r_c`` per metre of cable, with ``N_core`` its
cable-axis tension share (as ``capstan_force_per_wire`` for an armour wire),
plus one third of the line force ``2 pi r P`` that the bounded pressure chain
applies inside the innermost armour, with the armour winding-stress increment
superposed as for the armour clamping. With an interstitial filler both parts
bear on the filler; without one the capstan part bears on the two neighbouring
cores, whose contact forces sum to the capstan load over ``sin(pi/n)``
(1.155 times for a trefoil), and the ring part on the sheath.

Assumptions (stated, not solved)
--------------------------------
* Normal contact loads are those of the axisymmetric state and are held
  fixed during bending (as for the armour); the bending redistribution of
  core contact is not solved.
* The layers of one core are bonded and move together axially; the core is
  a rod of rigidity ``EA_core`` whose interior shear enters only through
  ``xi``.
* Core--core friction is applied to the slip of each core relative to the
  plane-section field, as are the filler and sheath contacts.
* The contact width for ``k_s`` is the core perimeter.
* Internal rings use the cable curvature times the helical-rod bending factor
  of the core lay; their external contact pressure is the core contact line
  load carried by line-force conservation, ``q_core / (2 pi r)``, an upper
  estimate for a thick polymer layer.

References
----------
[Papailiou1997] Eqs. 14, 15, 20; [Yun2020]; [KarlsenHusoy2026] Eqs. 2-10;
[Nasution2014b] Eqs. 4-9; [MenardCartraud2023] Section 5 (mu = 0.2 for all
contacts); [Janocha2024] Table 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from cablexsec.cables.schema import (
    CableData,
    CylindricalLayer,
    CylindricalOffsetLayer,
)
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.materials import library
from cablexsec.mechanics.axisymmetric import AxisymmetricResult, compute_contact_forces
from cablexsec.mechanics.core_lay import bending_factor, core_angle
from cablexsec.mechanics.mkappa import HelicalLayerSpec
from cablexsec.mechanics.papailiou import (
    DIRECTION_HOLD_FRACTION,
    BendingWork,
    wire_slip_history,
)
from cablexsec.mechanics.radial_cable import CableRadialState
from cablexsec.mechanics.section_geometry import is_interstitial_filler
from cablexsec.mechanics.section_state import SectionState
from cablexsec.mechanics.stiffness import axial_stiffness_by_layer, ea_fixed
from cablexsec.mechanics.strand_slip import (
    ConductorRing,
    StrandSlipParameters,
    conductor_ring_parameters,
    strand_slip_strain,
)
from cablexsec.mechanics.yun_coefficient import (
    series_shear_stiffness,
    shear_modulus,
    yun_xi,
)

__all__ = [
    "CoreContactLoads",
    "CoreFrictionHistory",
    "CoreFrictionModel",
    "CoreFrictionSpec",
    "StrandRingFriction",
    "build_core_friction",
    "core_contact_loads",
    "core_friction_history",
    "resolve_core_friction",
]

FloatArray = NDArray[np.float64]

#: Default friction coefficient of every core contact: the value Ménard and
#: Cartraud (2023, Section 5) adopt for all contacts of a three-core cable.
DEFAULT_CORE_MU = 0.2


@dataclass(frozen=True)
class CoreFrictionSpec:
    """Inputs of the core stick-slip law.

    Parameters
    ----------
    mu_core_filler, mu_core_core, mu_core_sheath : float
        Friction coefficients of the core--filler, core--core and core--ring
        (bedding or inner sheath) contacts. Default 0.2 for all
        ([MenardCartraud2023] Section 5); [Janocha2024] Table 2 gives 0.46
        for MDPE.
    interface_stiffness_Pa_per_m : float
        Elastic stick (penalty) stiffness of the core contact per unit area
        and slip [Pa/m]; ``inf`` (default) is a rigid contact. [Janocha2024]
        Table 2 gives 1200 MPa/mm = 1.2e12 Pa/m for MDPE.
    core_shear : bool
        Add the through-thickness shear of the polymer layers inside the core
        (outside its innermost structural layer) in series with the contact
        stiffness (default True).
    n_stations_per_core : int
        Helix stations per core (default 60).
    strand_slip : bool
        Apply the [KarlsenHusoy2026] slip of resolved internal wire rings of
        a core (``strand_layers``), where declared (default True).
    mu_strand : float or None
        Wire-wire friction of internal rings (default: the layer material's
        ``mu_on_cu``, else that of another layer of the core).
    mu_strand_outer : float or None
        Friction on the outer face of the outermost internal ring (default:
        ``mu_strand``).
    direction_hold_fraction : float
        Direction hold of the slip cap near zero curvature, as for the armour.
    """

    mu_core_filler: float = DEFAULT_CORE_MU
    mu_core_core: float = DEFAULT_CORE_MU
    mu_core_sheath: float = DEFAULT_CORE_MU
    interface_stiffness_Pa_per_m: float = math.inf
    core_shear: bool = True
    n_stations_per_core: int = 60
    strand_slip: bool = True
    mu_strand: float | None = None
    mu_strand_outer: float | None = None
    direction_hold_fraction: float = DIRECTION_HOLD_FRACTION

    def __post_init__(self) -> None:
        for name in ("mu_core_filler", "mu_core_core", "mu_core_sheath"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise CableInputError(f"{name} must be finite and non-negative")
        for name in ("mu_strand", "mu_strand_outer"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise CableInputError(f"{name} must be finite and non-negative")
        k_s = self.interface_stiffness_Pa_per_m
        if math.isnan(k_s) or k_s <= 0.0:
            raise CableInputError("interface stiffness must be positive (inf: rigid)")
        n = self.n_stations_per_core
        if isinstance(n, bool) or not isinstance(n, int) or n < 2:
            raise CableInputError("n_stations_per_core must be an integer >= 2")
        f = self.direction_hold_fraction
        if not math.isfinite(f) or not 0.0 <= f <= 1.0:
            raise CableInputError("direction hold fraction must lie in [0, 1]")


def resolve_core_friction(
    value: bool | CoreFrictionSpec | None,
) -> CoreFrictionSpec | None:
    """``True`` gives the default specification; ``None``/``False`` none."""
    if value is None or value is False:
        return None
    if value is True:
        return CoreFrictionSpec()
    if isinstance(value, CoreFrictionSpec):
        return value
    raise CableInputError("core_friction must be a bool or a CoreFrictionSpec")


@dataclass(frozen=True)
class CoreContactLoads:
    """Normal contact line loads on one core [N per metre of cable].

    ``core_core`` sums both neighbouring cores. ``source`` names the contact
    model that produced them.
    """

    core_core_N_per_m: float
    core_filler_N_per_m: float
    core_sheath_N_per_m: float
    source: str

    @property
    def total_N_per_m(self) -> float:
        return (
            self.core_core_N_per_m + self.core_filler_N_per_m + self.core_sheath_N_per_m
        )

    def mu_effective(self, spec: CoreFrictionSpec) -> float:
        """Load-weighted friction coefficient (``mu_core_filler`` if unloaded)."""
        total = self.total_N_per_m
        if total <= 0.0:
            return spec.mu_core_filler
        return (
            spec.mu_core_core * self.core_core_N_per_m
            + spec.mu_core_filler * self.core_filler_N_per_m
            + spec.mu_core_sheath * self.core_sheath_N_per_m
        ) / total


def _core_layers(cable: CableData) -> list[CylindricalOffsetLayer]:
    return [x for x in cable.layers if isinstance(x, CylindricalOffsetLayer)]


def _has_filler(cable: CableData) -> bool:
    return any(
        isinstance(x, CylindricalLayer) and is_interstitial_filler(x, cable)
        for x in cable.layers
    )


def _core_geometry(cable: CableData) -> tuple[int, float, float, float]:
    """``(n_cores, r_c, beta, R_core)`` of the single laid-core family."""
    cores = _core_layers(cable)
    offsets = {x.r_offset_m for x in cores}
    counts = {x.n_cores for x in cores}
    if len(offsets) != 1 or len(counts) != 1 or None in offsets:
        raise CableSchemaError(
            "core friction needs one family of laid cores with a common pitch "
            "radius and core count"
        )
    r_c = float(next(iter(offsets)))  # type: ignore[arg-type]
    n_cores = int(next(iter(counts)))
    if n_cores < 2 or r_c <= 0.0:
        raise CableSchemaError("core friction needs at least two offset cores")
    beta = abs(core_angle(cable, cores[0]))
    if not 0.0 < beta < math.pi / 2:
        raise CableSchemaError(
            "core friction needs a laid core (core_lay pitch giving a nonzero "
            "lay angle)"
        )
    r_core = max(float(x.r_outer_m or 0.0) for x in cores)
    return n_cores, r_c, beta, r_core


def _bounded_loads(
    cable: CableData, axi: AxisymmetricResult, n_cores: int, r_c: float, beta: float
) -> CoreContactLoads:
    core_names = {x.name for x in _core_layers(cable)}
    n_core = sum(v for k, v in axi.layer_tension_N.items() if k in core_names)
    # Inward capstan load of one core per metre of cable: the core force
    # along its helix is T/cos(beta), its curvature sin^2(beta)/r_c and
    # ds/dz = 1/cos(beta), so T tan^2(beta)/r_c, as for an armour wire
    # (axisymmetric.capstan_force_per_wire).
    capstan = max(n_core, 0.0) / n_cores * math.tan(beta) ** 2 / r_c
    # Pressure inside the innermost armour (the chain needs a helical layer),
    # plus the armour winding-stress increment superposed as in
    # cable_adapters.armour_contact_pressure.
    pressure = float(axi.interface_pressure_Pa[0])
    helical = cable.helical_layers
    if any(x.winding_stress_Pa for x in helical):
        tension = {
            x.name: (x.n_wires or 0)
            * math.pi
            * (x.d_wire_m or 0.0) ** 2
            / 4.0
            * (x.winding_stress_Pa or 0.0)
            * math.cos(x.lay_angle_rad or 0.0)
            for x in helical
        }
        wound = compute_contact_forces(cable, 0.0, layer_tension_N=tension)
        base = compute_contact_forces(cable, 0.0)
        pressure += float(
            wound.interface_pressure_Pa[0] - base.interface_pressure_Pa[0]
        )
    ring = 2.0 * math.pi * float(axi.interface_radii_m[0]) * pressure / n_cores
    if _has_filler(cable):
        return CoreContactLoads(0.0, capstan + ring, 0.0, "bounded")
    # Without a filler the inward load rests on the neighbouring cores. With
    # three or more cores each core touches two neighbours whose contact
    # normals lie pi/2 - pi/n off the radial direction, so the two contact
    # forces sum to capstan / sin(pi/n) (1.155 times for a trefoil); two cores
    # touch along the radial line.
    core_core = capstan / math.sin(math.pi / n_cores) if n_cores >= 3 else capstan
    return CoreContactLoads(core_core, 0.0, ring, "bounded")


def core_contact_loads(
    cable: CableData,
    tension_N: float = 0.0,
    *,
    contact_model: str = "bounded",
    radial_state: CableRadialState | None = None,
    section_state: SectionState | None = None,
) -> CoreContactLoads:
    """Normal contact line loads of one laid core (see the module notes)."""
    n_cores, r_c, beta, _ = _core_geometry(cable)
    if radial_state is not None or contact_model == "solved":
        if section_state is not None:
            raise CableSchemaError("supply a section state or a radial state, not both")
        if radial_state is None:
            from cablexsec.mechanics.radial_cable import (  # noqa: PLC0415
                solve_cable_radial_state,
            )

            radial_state = solve_cable_radial_state(cable, tension_N)
        radial_state.require_cable(cable)
        bundle = radial_state.result.bundle
        if bundle is None:
            raise CableSchemaError("the solved radial state has no core bundle")
        return CoreContactLoads(
            2.0 * max(bundle.core_core_force_N_per_m, 0.0),
            max(bundle.filler_core_force_N_per_m, 0.0),
            max(bundle.core_ring_force_N_per_m, 0.0),
            "solved",
        )
    if section_state is not None:
        section_state.require_cable(cable)
        axi = section_state.bounded
        if axi is None:
            axi = compute_contact_forces(
                cable,
                section_state.tension_N,
                p_ext_Pa=section_state.external_pressure_Pa,
            )
        return _bounded_loads(cable, axi, n_cores, r_c, beta)
    if contact_model not in {"bounded", "qp"}:
        raise CableSchemaError("contact_model must be 'bounded', 'qp' or 'solved'.")
    axi = compute_contact_forces(
        cable, tension_N, p_ext_Pa=cable.initial_state.external_pressure_Pa
    )
    return _bounded_loads(cable, axi, n_cores, r_c, beta)


@dataclass(frozen=True)
class StrandRingFriction:
    """One resolved internal ring with its centroid-strain parameters."""

    layer_name: str
    ring: ConductorRing
    parameters: StrandSlipParameters
    E_Pa: float
    #: ``n_cores * bending_factor(beta)``: the cable moment per ring moment.
    scale: float

    @property
    def stick_EI_Nm2(self) -> float:
        """Stuck ring parallel-axis stiffness ``0.5 n E A R^2 cos^3 a`` times scale."""
        c = math.cos(self.ring.lay_angle_rad)
        return (
            self.scale
            * 0.5
            * self.ring.n_wires
            * self.E_Pa
            * self.ring.area_m2
            * self.ring.r_helix_m**2
            * c**3
        )


@dataclass(frozen=True)
class CoreFrictionModel:
    """Assembled core population, internal rings and their inputs."""

    spec: CoreFrictionSpec
    population: HelicalLayerSpec
    loads: CoreContactLoads
    mu_effective: float
    xi: float
    n_cores: int
    r_offset_m: float
    lay_angle_rad: float
    EA_core_N: float
    plane_section_EI_Nm2: float
    strand_rings: tuple[StrandRingFriction, ...] = field(default=())
    tension_N: float = 0.0

    @property
    def core_stick_EI_Nm2(self) -> float:
        """Stick stiffness of the core population, ``xi`` times plane section."""
        return float(self.population.EI_stick)

    @property
    def strand_stick_EI_Nm2(self) -> float:
        return float(sum(r.stick_EI_Nm2 for r in self.strand_rings))

    @property
    def stick_EI_Nm2(self) -> float:
        return self.core_stick_EI_Nm2 + self.strand_stick_EI_Nm2


def _stick_coefficient(
    cable: CableData,
    spec: CoreFrictionSpec,
    ea_core: float,
    r_c: float,
    beta: float,
    r_core: float,
) -> float:
    stack: list[tuple[float, float, float]] = []
    if math.isfinite(spec.interface_stiffness_Pa_per_m):
        stack.append((1.0, spec.interface_stiffness_Pa_per_m, 2.0 * math.pi * r_core))
    if spec.core_shear:
        cores = sorted(_core_layers(cable), key=lambda x: x.r_inner_m or 0.0)
        # Layers outside the innermost structural one (the conductor) shear.
        start = next(x for x in cores if x.structural).r_outer_m or 0.0
        for lyr in cores:
            ri, ro = lyr.r_inner_m or 0.0, lyr.r_outer_m or 0.0
            if ri < start - 1e-12 or ro <= ri:
                continue
            try:
                e = library.get_property(lyr.material, "E_Pa")
                nu = library.get_property(lyr.material, "nu")
            except KeyError:  # no isotropic modulus: no shear compliance
                continue
            stack.append((ro - ri, shear_modulus(e, nu), math.pi * (ro + ri)))
    return yun_xi(series_shear_stiffness(stack), ea_core, beta, r_c)


def _strand_rings(
    cable: CableData,
    spec: CoreFrictionSpec,
    loads: CoreContactLoads,
    tension_N: float,
    n_cores: int,
    beta: float,
) -> tuple[StrandRingFriction, ...]:
    if not spec.strand_slip:
        return ()
    out: list[StrandRingFriction] = []
    compliance = 1.0 / ea_fixed(cable)
    core_mu: list[float] = []
    for lyr in _core_layers(cable):
        try:
            core_mu.append(library.get_property(lyr.material, "mu_on_cu"))
        except KeyError:
            continue
    for lyr in _core_layers(cable):
        if not lyr.structural or not lyr.strand_layers:
            continue
        try:
            e = library.get_property(lyr.material, "E_eff_axial_Pa")
        except KeyError:
            e = library.get_property(lyr.material, "E_Pa")
        mu = spec.mu_strand
        if mu is None:
            # The layer's own copper friction, else that of another core layer
            # (a wire screen takes the conductor's value).
            try:
                mu = library.get_property(lyr.material, "mu_on_cu")
            except KeyError:
                if not core_mu:
                    raise CableInputError(
                        f"layer {lyr.name!r}: no mu_on_cu in the core materials; "
                        "supply CoreFrictionSpec(mu_strand=...)"
                    ) from None
                mu = core_mu[0]
        rings = tuple(
            ConductorRing(
                ring.n_wires,
                ring.d_wire_m,
                ring.r_helix_m,
                ring.lay_angle_rad if ring.r_helix_m > 0 else 0.0,
                wire_area_m2=ring.wire_area_m2,
            )
            for ring in lyr.strand_layers
        )
        r_out = float(lyr.r_outer_m or 0.0) or max(
            r.r_helix_m + r.d_wire_m / 2 for r in rings
        )
        pressure = loads.total_N_per_m / (2.0 * math.pi * r_out)
        params = conductor_ring_parameters(
            rings,
            e,
            mu_wire=float(mu),
            mu_outer=spec.mu_strand_outer,
            reference_tension_N=tension_N if tension_N > 0.0 else 1.0,
            strain_per_tension=compliance,
            external_pressure_Pa=pressure,
            r_outer_m=r_out,
            fibre="centroid",
        )
        scale = n_cores * bending_factor(beta)
        for ring, parameters in zip(rings, params, strict=True):
            if ring.r_helix_m > 0.0:
                out.append(StrandRingFriction(lyr.name, ring, parameters, e, scale))
    return tuple(out)


def build_core_friction(
    cable: CableData,
    spec: CoreFrictionSpec,
    tension_N: float = 0.0,
    *,
    contact_model: str = "bounded",
    radial_state: CableRadialState | None = None,
    section_state: SectionState | None = None,
) -> CoreFrictionModel:
    """Assemble the core population and internal rings of a cable."""
    if not math.isfinite(tension_N) or tension_N < 0.0:
        raise CableInputError("tension must be finite and non-negative")
    n_cores, r_c, beta, r_core = _core_geometry(cable)
    loads = core_contact_loads(
        cable,
        tension_N,
        contact_model=contact_model,
        radial_state=radial_state,
        section_state=section_state,
    )
    axial = axial_stiffness_by_layer(cable)
    cos3 = math.cos(beta) ** 3
    contribution = sum(
        axial.get(x.name, 0.0) for x in _core_layers(cable) if x.structural
    )
    if contribution <= 0.0:
        raise CableSchemaError("the laid cores have no structural axial stiffness")
    ea_core = contribution / (n_cores * cos3)
    xi = _stick_coefficient(cable, spec, ea_core, r_c, beta, r_core)
    mu = loads.mu_effective(spec)
    n_st = n_cores * spec.n_stations_per_core
    area_core = math.pi * r_core**2
    population = HelicalLayerSpec(
        n_wires=n_st,
        r_helix_m=r_c,
        lay_angle_rad=beta,
        E_wire_Pa=ea_core / area_core,
        A_wire_m2=area_core * n_cores / n_st,
        xi=xi,
        mu=mu,
        F_n_axi_N=loads.total_N_per_m,
        contact_pressure_Pa=n_cores * loads.total_N_per_m / (2.0 * math.pi * r_c),
        normal_load_is_line=True,
    )
    return CoreFrictionModel(
        spec=spec,
        population=population,
        loads=loads,
        mu_effective=mu,
        xi=xi,
        n_cores=n_cores,
        r_offset_m=r_c,
        lay_angle_rad=beta,
        EA_core_N=ea_core,
        plane_section_EI_Nm2=0.5 * contribution * r_c**2,
        strand_rings=_strand_rings(cable, spec, loads, tension_N, n_cores, beta),
        tension_N=float(tension_N),
    )


@dataclass(frozen=True)
class CoreFrictionHistory:
    """Moments ``(samples, 2)`` and work of the cores and internal rings."""

    core_moment_Nm: FloatArray
    strand_moment_Nm: FloatArray
    core_slip_fraction: FloatArray
    work: BendingWork
    model: CoreFrictionModel

    @property
    def moment_Nm(self) -> FloatArray:
        return np.asarray(self.core_moment_Nm + self.strand_moment_Nm)


def _strand_history(
    model: CoreFrictionModel, kx: FloatArray, ky: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    count = kx.size
    moment = np.zeros((count, 2))
    stored, friction, endpoint, remainder = (np.zeros(count) for _ in range(4))
    tension = np.full(count, model.tension_N)
    for item in model.strand_rings:
        ring = item.ring
        theta = 2.0 * math.pi * np.arange(ring.n_wires) / ring.n_wires
        result = strand_slip_strain(item.parameters, kx, ky, tension, theta_rad=theta)
        eps = result.bending_strain  # (wires, samples): centroid wire strain
        drive = item.parameters.beta_m[0] * result.curvature_projection
        c = math.cos(ring.lay_angle_rad)
        force = item.E_Pa * ring.area_m2 * c * eps
        moment[:, 0] += item.scale * ring.r_helix_m * (np.cos(theta) @ force)
        moment[:, 1] += item.scale * ring.r_helix_m * (np.sin(theta) @ force)
        weight = item.scale * item.E_Pa * ring.area_m2 / c
        d_eps = np.diff(eps, axis=1, prepend=0.0)
        d_drive = np.diff(drive, axis=1, prepend=0.0)
        stored += 0.5 * weight * np.sum(eps**2, axis=0)
        friction += np.cumsum(weight * np.sum(eps * (d_drive - d_eps), axis=0))
        endpoint += np.cumsum(weight * np.sum(eps * d_drive, axis=0))
        remainder += np.cumsum(0.5 * weight * np.sum(d_eps**2, axis=0))
    return moment, stored, friction, endpoint, remainder


def core_friction_history(
    model: CoreFrictionModel,
    kappa_x: FloatArray,
    kappa_y: FloatArray,
    *,
    initial_bending_angle: float = 0.0,
) -> CoreFrictionHistory:
    """Biaxial history of the core population and the internal rings.

    The core stations use :func:`wire_slip_history` unchanged (direction-tied
    Papailiou cap); the rings use :func:`strand_slip_strain`. Both return
    moments in the material frame of the curvature components.
    """
    kx = np.asarray(kappa_x, dtype=np.float64)
    ky = np.asarray(kappa_y, dtype=np.float64)
    history = wire_slip_history(
        model.population,
        kx,
        ky,
        initial_bending_angle=initial_bending_angle,
        direction_hold_fraction=model.spec.direction_hold_fraction,
    )
    s_moment, s_stored, s_friction, s_endpoint, s_remainder = _strand_history(
        model, kx, ky
    )
    w = history.work
    stored = w.stored_J_per_m + s_stored
    friction = w.friction_J_per_m + s_friction
    endpoint = w.endpoint_work_J_per_m + s_endpoint
    remainder = w.integration_remainder_J_per_m + s_remainder
    return CoreFrictionHistory(
        core_moment_Nm=np.asarray(history.moment_Nm),
        strand_moment_Nm=s_moment,
        core_slip_fraction=history.slipping_count / float(model.population.n_wires),
        work=BendingWork(
            stored,
            friction,
            endpoint,
            remainder,
            endpoint - stored - friction - remainder,
        ),
        model=model,
    )
