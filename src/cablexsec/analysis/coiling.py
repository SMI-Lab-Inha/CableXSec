# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Armour stress of a cable coiled into a static tank.

Coiling is the simultaneous bending and twisting of a cable with one full
twist per turn ([DNVGL-RP-0360] Table 1-6; [ChaeNamShim2023] Sec. 2.1.3), so a
tank of diameter ``D`` imposes the curvature ``kappa = 2/D`` and the twist
``tau = 2 pi/(pi D) = 2/D`` together. Coiling is performed so that the armour
opens up ([DNVGL-RP-0360] Sec. 5.3.1.2, guidance note 2; [ChaeNamShim2023]
Sec. 1): the twist unwinds the outer armour, the unwound wires are compressed,
and the risk is radial buckling of the armour (bird-caging).

The request is solved in two steps that share one section state:

1. **Twisted section.** The coupled radial-axial-torsional section
   (:func:`~cablexsec.mechanics.radial_cable.solve_cable_radial_state`) is
   solved at the twist (or the torque) with the tension given, the ends free
   to move axially. The wire strain is ``eps cos^2(a) + r tau sin(a) cos(a) +
   (u/r) sin^2(a)`` (axial strain, twist and radial displacement). The step
   carries the radial expansion of the compressed armour against the layers
   outside it, the opening of the contacts inside it and the axial
   equilibrium between armour and cores. It gives the wire stress at the wire
   centre and the armour contact pressures.
2. **Bending through one winding.** The curvature is raised to ``kappa`` and
   its vector is turned through ``2 pi`` per winding in the cable frame, as
   the cross-section rotates once per winding relative to the plane of the
   coil. The friction stress (slip front on the solved contact pressures) and
   the local bending of each wire are recovered on fixed fibres with
   :func:`~cablexsec.analysis.fatigue.wire_stress_field`, on top of the wire
   stress of step 1.

Checks per structural armour layer, all in wire stress (along the wire):

* **Radial buckling (bird-caging).** The [SaevikThorsen2017] capacities are
  axial forces of the cable, ``P = n cos(a) Q`` with ``Q`` the force in one
  wire (Eqs. 32 and 34), so the wire stress at a capacity is ``P/(n A
  cos a)``: ``sigma_c1 = P_2/(n A cos a) = 2 sqrt(EI_2 c)/A`` (Eq. 22),
  ``sigma_c2 = P_4/(n A cos a) = sigma_y`` (Eq. 34) and their interaction
  ``sigma_cr = (1/sigma_c1 + 1/sigma_c2)^-1`` (Eq. 35 without the hoop term).
  [ChaeNamShim2023] Eqs. 17-19 divide by ``n A`` only, which gives the
  axial-direction component, ``cos a`` times the wire stress, and compare it
  with the wire stress (Table 8); the forms here are their Eqs. 17-19
  divided by ``cos a`` (3.5% at their pitch of 1.8 m). The demand is the
  most compressive wire-centre stress (twist and friction): buckling is set
  by the wire force, and the local bending of the wire is self-equilibrated
  over its section. [ChaeNamShim2023] compare the surface stress, which adds
  that local bending.
* **First yield.** The largest surface stress magnitude (twist, friction and
  local bending) over ``sigma_y``. Above one the wire yields, the elastic
  stresses and the elastic buckling check are not valid, and the result says
  so (:attr:`CoilingLayerResult.elastic_valid`); stresses are not capped.
* **Lateral buckling, reported only.** The friction-free lower bound
  ``P_1/(n A cos a)`` (Eq. 19). It does not enter the utilisation: lateral
  buckling of the armour is triggered by cyclic bending that removes the
  friction restraint and accumulates over many cycles ([SaevikThorsen2017]
  Sec. 4.2.1: tests of 800 to 6000 cycles; [LiVaz2021b] Eq. 16, Table 2),
  whereas a coiled section is bent once on loading and once on unloading;
  the bound tends to zero with the lay angle and the published comparisons
  cover lay angles of 20 to 38 degrees, above the 10 to 20 degrees of power
  cable armour; and [ChaeNamShim2023], the coiling source, check radial
  buckling only. Taken as a capacity, it would require a tank of about
  250 m for the cable of [ChaeNamShim2023], which that study coils in tanks
  of 10 to 18 m.

The utilisation of a layer is the larger of the radial buckling and the
first-yield utilisations.

Components other than the armour. The conductor, screens, insulation and
sheaths are checked against the limits their materials give
(:mod:`cablexsec.analysis.component_limits`) with
:func:`~cablexsec.analysis.components.compute_component_response` at the coil
curvature and tension, bent towards a core (the curvature turns through every
direction over a winding), in the frictional core slip state; the twist of the
coil is not applied to the cores; as in the capacity envelope, the yield of a
metallic screen or sheath does not count, its allowable strain does
(:func:`~cablexsec.analysis.components.capacity_utilisation`). Layers outside
the load path (loose screens)
are reported apart (``kinematic``): their stress is the kinematic upper bound
of a screen that does not slip. The result's utilisation is the largest of
the armour and component utilisations.

Scope: cables. An unbonded flexible pipe (a near-hoop carcass or pressure
armour) is refused: flexible pipes are spooled onto reels or carousels
([OC2017] Sec. A1.7.1), not coiled with one twist per turn. Counter-helical
double armour is torsion balanced and non-coilable ([DNVGL-RP-0360]
Sec. 4.4.1.6, guidance note 3); it is solved, and the result carries that
caveat with the stresses that one twist per turn would impose.

Excluded: the transition from the straight cable into the coil (curvature and
twist are uniform), contact with the tank floor and the coils below, the
weight of the coils above, wire-to-wire contact within a layer, and wire
plasticity (stresses are elastic; first yield is detected, not followed). The
capacities are elastic screening values.

References
----------
[ChaeNamShim2023] Ocean Eng. 288, 116014, Secs. 1, 2.1.3, Eqs. 17-19,
Table 8; [SaevikThorsen2017] J. Offshore Mech. Arct. Eng. 139, 041701,
Eqs. 19, 22, 23, 32, 34, 35, Sec. 4.2.1; [LiVaz2021b] Mar. Struct. 76,
102895, Eq. 16, Tables 2 and 3; [DNVGL-RP-0360] Subsea power cables in
shallow water (March 2016), Table 1-6, Secs. 4.4.1.6 and 5.3.1.2; [OC2017]
Handbook on design and operation of flexible pipes (2017), Sec. A1.7.1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from cablexsec.analysis.capacity import birdcage_buckling_detail
from cablexsec.analysis.fatigue import wire_stress_field
from cablexsec.cable_adapters import cable_to_fatigue_layers, structural_armour_layers
from cablexsec.cables.schema import CableData, HelicalLayer
from cablexsec.exceptions import CableInputError
from cablexsec.materials import library
from cablexsec.mechanics.radial_cable import solve_cable_radial_state
from cablexsec.provenance import Provenance

TwistSense = Literal["unwind", "tighten", "signed"]

#: Lay angle [deg] from which a helical layer is a near-hoop carcass or
#: pressure armour (85-89.5 deg in the packaged pipes); power-cable armour
#: and screens lie at 10-50 deg.
NEAR_HOOP_LAY_ANGLE_DEG = 75.0

_PROVENANCE = Provenance(
    method="coiling: twisted radial-axial-torsional section (free axial) with "
    "rotating bending, slip-front friction and local wire bending on fixed fibres; "
    "Saevik-Thorsen radial buckling with yield interaction in wire stress, first "
    "yield on the wire surface",
    citations=(
        "[ChaeNamShim2023]",
        "[SaevikThorsen2017]",
        "[LiVaz2021b]",
        "[DNVGL-RP-0360]",
    ),
    gates=("CH-CR",),
    limitations=(
        "uniform curvature and twist; no transition into the coil, no tank-floor "
        "or coil-on-coil contact, no weight of the coils above",
        "elastic wire stresses and elastic buckling capacities (screening values); "
        "the friction-free lateral bound is reported, not used",
        "no measured coiling stress is available; only the closed-form capacity "
        "formulas of the source are checked",
        "the components other than the armour are checked at the coil "
        "curvature without the twist of the coil; a component without a limit "
        "in its material is not checked",
    ),
)

_BEYOND_YIELD = (
    "beyond first yield: the armour surface stress exceeds the yield stress, so "
    "the elastic stresses and the elastic buckling check are not valid"
)
_COUNTER_HELICAL = (
    "counter-helical armour: a torsion-balanced design, non-coilable according "
    "to DNVGL-RP-0360 Sec. 4.4.1.6; the stresses are those that one twist per "
    "turn would impose"
)


@dataclass(frozen=True)
class CoilingLayerResult:
    """Coiling stresses and compressive capacity of one armour layer [Pa].

    ``axisymmetric_Pa`` is the wire stress of the twisted section;
    ``centre_min_Pa`` / ``centre_max_Pa`` add the friction stress over the
    winding; ``fibre_min_Pa`` / ``fibre_max_Pa`` add the local bending of the
    wire on ``n_fibres`` surface fibres. The capacities are wire stresses
    (along the wire): ``sigma_c1_Pa`` (radial buckling, ``P_2/(n A cos a)``),
    ``sigma_c2_Pa`` (the yield term ``P_4/(n A cos a) = sigma_y``, ``nan``
    without a yield stress), ``sigma_cr_Pa`` (their interaction) and
    ``sigma_lateral_Pa`` (friction-free lateral lower bound, reported only).
    ``capacity_Pa`` is ``sigma_cr_Pa``. ``utilisation`` is the larger of
    :attr:`buckling_utilisation` and :attr:`yield_utilisation`;
    ``governing_mode`` says which (``"radial"`` or ``"yield"``).

    Changed on 23 October 2026: the capacities were ``P/(n A)`` (Chae et al.
    Eqs. 17-19, ``cos a`` times the wire stress), the capacity was
    ``min(sigma_cr, sigma_lateral)``, the demand the surface stress, and
    ``governing_mode`` was ``"lateral"`` or ``"radial"``.
    """

    layer: str
    axisymmetric_Pa: float
    centre_min_Pa: float
    centre_max_Pa: float
    fibre_min_Pa: float
    fibre_max_Pa: float
    friction_max_Pa: float
    contact_pressure_Pa: float
    sigma_c1_Pa: float
    sigma_c2_Pa: float
    sigma_cr_Pa: float
    sigma_lateral_Pa: float
    capacity_Pa: float
    governing_mode: str
    utilisation: float

    @property
    def buckling_utilisation(self) -> float:
        """Most compressive wire-centre stress over ``sigma_cr`` (radial)."""
        return _ratio(max(0.0, -self.centre_min_Pa), self.sigma_cr_Pa)

    @property
    def yield_utilisation(self) -> float:
        """Largest surface stress magnitude over ``sigma_y`` (``nan`` without)."""
        peak = max(abs(self.fibre_min_Pa), abs(self.fibre_max_Pa))
        if not math.isfinite(self.sigma_c2_Pa):
            return math.nan
        return _ratio(peak, self.sigma_c2_Pa)

    @property
    def lateral_utilisation(self) -> float:
        """Most compressive wire-centre stress over the lateral bound (reported)."""
        return _ratio(max(0.0, -self.centre_min_Pa), self.sigma_lateral_Pa)

    @property
    def elastic_valid(self) -> bool:
        """Whether the wire stays below first yield (``True`` without ``sigma_y``)."""
        value = self.yield_utilisation
        return math.isnan(value) or value <= 1.0


@dataclass(frozen=True)
class CoilingResult:
    """Coiling state of a cable section.

    ``curvature_per_m`` and ``twist_per_m`` are the applied curvature and the
    solved twist (signed, package convention: positive tightens a right-hand
    lay); ``torque_Nm`` and ``axial_strain`` are those of the twisted section
    with free ends. ``open_interfaces`` names the contacts that opened. The
    provenance limitations add, when they apply, that the result is beyond
    first yield and that counter-helical armour is a non-coilable design.
    ``components`` gives, per component other than the armour with a limit,
    its utilisation and governing limit at the coil curvature (module notes);
    ``kinematic`` the same for the layers outside the load path, which do not
    enter :attr:`utilisation`.
    """

    tank_diameter_m: float | None
    curvature_per_m: float
    twist_per_m: float
    torque_Nm: float
    axial_strain: float
    tension_N: float
    layers: dict[str, CoilingLayerResult]
    open_interfaces: tuple[str, ...]
    provenance: Provenance = _PROVENANCE
    components: dict[str, tuple[float, str]] = field(default_factory=dict)
    kinematic: dict[str, tuple[float, str]] = field(default_factory=dict)

    @property
    def utilisation(self) -> float:
        """Largest utilisation of the armour layers and the other components."""
        armour = max((x.utilisation for x in self.layers.values()), default=0.0)
        others = max((value for value, _ in self.components.values()), default=0.0)
        return max(armour, others)

    @property
    def governing(self) -> str:
        """The armour layer or component of :attr:`utilisation`."""
        candidates = {name: x.utilisation for name, x in self.layers.items()}
        candidates.update({name: v for name, (v, _) in self.components.items()})
        if not candidates:
            return ""
        return max(candidates, key=lambda name: candidates[name])

    @property
    def elastic_valid(self) -> bool:
        """Whether every armour layer stays below first yield."""
        return all(x.elastic_valid for x in self.layers.values())


def _ratio(demand: float, capacity: float) -> float:
    if demand == 0.0:
        return 0.0
    return demand / capacity if math.isfinite(capacity) else 0.0


def flexible_pipe_layer(cable: CableData) -> HelicalLayer | None:
    """The first near-hoop helical layer (carcass or pressure armour), if any.

    A helical layer with a lay angle of at least
    :data:`NEAR_HOOP_LAY_ANGLE_DEG` marks an unbonded flexible pipe.
    """
    for layer in cable.helical_layers:
        angle = layer.lay_angle_deg
        if angle is not None and abs(angle) >= NEAR_HOOP_LAY_ANGLE_DEG:
            return layer
    return None


def counter_helical_armour(cable: CableData) -> bool:
    """Whether the structural armour has layers of both lay directions."""
    return len({x.lay_direction for x in structural_armour_layers(cable)}) > 1


def _unwind_sign(armour: list[HelicalLayer]) -> float:
    # Positive twist tightens a right-hand lay, so it unwinds a left-hand one.
    return 1.0 if armour[-1].lay_direction == "left_hand" else -1.0


def _capacities(
    cable: CableData, layer: HelicalLayer
) -> tuple[float, float, float, float]:
    """``sigma_c1, sigma_c2, sigma_cr, sigma_lateral`` in wire stress [Pa]."""
    detail = birdcage_buckling_detail(cable, layer)
    assert (
        layer.n_wires is not None
        and layer.lay_angle_rad is not None
        and layer.d_wire_m is not None
    )
    # d_wire_m is the area-equivalent diameter of a rectangular or profiled
    # wire. The capacities are cable axial forces, P = n cos(a) Q
    # ([SaevikThorsen2017] Eqs. 32, 34): the wire stress is P/(n A cos a).
    force_per_stress = (
        layer.n_wires
        * math.pi
        * layer.d_wire_m**2
        / 4.0
        * math.cos(layer.lay_angle_rad)
    )
    sigma_c1 = detail.P2_radial_N / force_per_stress
    sigma_lat = detail.P1_lateral_N / force_per_stress
    try:
        sigma_c2 = library.get_property(layer.material, "sigma_y_Pa")
    except KeyError:
        sigma_c2 = math.nan
    terms = [s for s in (sigma_c1, sigma_c2) if math.isfinite(s)]
    sigma_cr = 1.0 / sum(1.0 / s for s in terms) if terms else math.inf
    return sigma_c1, sigma_c2, sigma_cr, sigma_lat


def compute_coiling(
    cable: CableData,
    *,
    tank_diameter_m: float | None = None,
    curvature_per_m: float | None = None,
    twist_per_m: float | None = None,
    torque_Nm: float | None = None,
    twist_sense: TwistSense = "unwind",
    tension_N: float = 0.0,
    mu: float | None = None,
    turns: float = 1.0,
    n_ramp: int = 40,
    n_per_turn: int = 180,
    n_fibres: int = 32,
) -> CoilingResult:
    """Armour stresses, bird-caging and first-yield checks of a coiled cable.

    Parameters
    ----------
    cable : CableData
        A fully specified cable with at least one structural armour layer.
        An unbonded flexible pipe (a helical layer at
        :data:`NEAR_HOOP_LAY_ANGLE_DEG` or more) is refused.
    tank_diameter_m : float, optional
        Coil diameter ``D`` [m]: curvature ``2/D`` and twist ``2/D`` (one turn
        of the cable per winding). ``curvature_per_m``, ``twist_per_m`` or
        ``torque_Nm`` replace the corresponding value.
    curvature_per_m : float, optional
        Coil curvature [1/m], >= 0.
    twist_per_m : float, optional
        Twist magnitude [rad/m] (or the signed twist with
        ``twist_sense="signed"``).
    torque_Nm : float, optional
        Torque instead of twist (torque control, [ChaeNamShim2023] Table 3);
        magnitude, or signed with
        ``twist_sense="signed"``.
    twist_sense : {"unwind", "tighten", "signed"}
        ``"unwind"`` (default) turns the cable so that the outermost
        structural armour unwinds, the coiling practice; ``"tighten"`` the
        opposite; ``"signed"`` takes the value as given (positive tightens a
        right-hand lay).
    tension_N : float
        Tension held during coiling [N] (true wall tension), default 0.
    mu : float, optional
        Armour friction coefficient; default from the material library.
    turns : float
        Number of windings through which the curvature vector is turned.
    n_ramp, n_per_turn, n_fibres : int
        Samples of the curvature ramp, samples per winding and surface
        fibres per wire.

    Raises
    ------
    CableInputError
        Invalid inputs, no structural armour, or an unbonded flexible pipe.
    """
    hoop = flexible_pipe_layer(cable)
    if hoop is not None:
        assert hoop.lay_angle_deg is not None
        raise CableInputError(
            "coiling with one twist per turn does not apply to an unbonded "
            f"flexible pipe: {hoop.name} is a near-hoop helix (lay angle "
            f"{abs(hoop.lay_angle_deg):.1f} degrees), a carcass or pressure "
            "armour. Flexible pipes are spooled onto reels or carousels without "
            "imposed twist; check them in bending instead"
        )
    armour = structural_armour_layers(cable)
    if not armour:
        raise CableInputError("coiling needs at least one structural armour layer")
    if tank_diameter_m is not None and not (
        math.isfinite(tank_diameter_m) and tank_diameter_m > 0.0
    ):
        raise CableInputError("tank_diameter_m must be finite and > 0")
    base = 2.0 / tank_diameter_m if tank_diameter_m is not None else None
    kappa = curvature_per_m if curvature_per_m is not None else base
    if kappa is None:
        raise CableInputError("give tank_diameter_m or curvature_per_m")
    if not (math.isfinite(kappa) and kappa >= 0.0):
        raise CableInputError("curvature must be finite and >= 0")
    if torque_Nm is not None and twist_per_m is not None:
        raise CableInputError("give twist_per_m or torque_Nm, not both")
    twist = (
        twist_per_m
        if twist_per_m is not None
        else (None if torque_Nm is not None else base)
    )
    if twist is None and torque_Nm is None:
        raise CableInputError("give tank_diameter_m, twist_per_m or torque_Nm")
    if twist_sense not in ("unwind", "tighten", "signed"):
        raise CableInputError("twist_sense must be 'unwind', 'tighten' or 'signed'")
    for value, name in ((twist, "twist_per_m"), (torque_Nm, "torque_Nm")):
        if value is None:
            continue
        if not math.isfinite(value):
            raise CableInputError(f"{name} must be finite")
        if twist_sense != "signed" and value < 0.0:
            raise CableInputError(
                f"{name} is a magnitude with twist_sense={twist_sense!r}"
            )
    if not (math.isfinite(tension_N) and tension_N >= 0.0):
        raise CableInputError("tension_N must be finite and >= 0")
    if not (math.isfinite(turns) and turns > 0.0):
        raise CableInputError("turns must be finite and > 0")
    if min(n_ramp, n_per_turn, n_fibres) < 1:
        raise CableInputError("n_ramp, n_per_turn and n_fibres must be >= 1")

    sign = {"unwind": 1.0, "tighten": -1.0, "signed": 1.0}[twist_sense]
    if twist_sense != "signed":
        sign *= _unwind_sign(armour)
    state = solve_cable_radial_state(
        cable,
        tension_N,
        prescribed_twist_per_m=None if twist is None else sign * twist,
        torque_Nm=0.0 if torque_Nm is None else sign * torque_Nm,
        loose_layers="neighbour",
    )
    result = state.result
    solved_twist = float(result.twist_per_m)

    # The cross-section turns with the twist; in the cable frame the
    # curvature vector turns the other way, once per winding.
    n_turn = max(2, int(math.ceil(n_per_turn * turns)) + 1)
    rotation = -math.copysign(1.0, solved_twist) if solved_twist else 1.0
    phi = np.r_[
        np.zeros(n_ramp), rotation * np.linspace(0.0, 2.0 * math.pi * turns, n_turn)
    ]
    amp = np.r_[np.linspace(0.0, kappa, n_ramp), np.full(n_turn, kappa)]
    kx, ky = amp * np.cos(phi), amp * np.sin(phi)

    layers_out: dict[str, CoilingLayerResult] = {}
    fatigue_layers = cable_to_fatigue_layers(
        cable, mu=mu, contact_model="solved", radial_state=state
    )
    angles = np.linspace(0.0, 2.0 * math.pi, n_fibres, endpoint=False)
    for spec_layer, layer in zip(fatigue_layers, armour, strict=True):
        sigma0 = float(result.wire_stress_Pa[layer.name])
        field = wire_stress_field(
            spec_layer,
            kx,
            ky,
            normal_stress_Pa=np.full(kx.shape, sigma0),
            keep_friction=True,
        )
        fibres = np.stack([field.at_fibre(float(a)) for a in angles])
        assert field.friction_Pa is not None
        c1, c2, cr, lat = _capacities(cable, layer)
        partial = CoilingLayerResult(
            layer=layer.name,
            axisymmetric_Pa=sigma0,
            centre_min_Pa=float(field.mean_Pa.min()),
            centre_max_Pa=float(field.mean_Pa.max()),
            fibre_min_Pa=float(fibres.min()),
            fibre_max_Pa=float(fibres.max()),
            friction_max_Pa=float(np.abs(field.friction_Pa).max()),
            contact_pressure_Pa=float(spec_layer.spec.contact_pressure_Pa),
            sigma_c1_Pa=c1,
            sigma_c2_Pa=c2,
            sigma_cr_Pa=cr,
            sigma_lateral_Pa=lat,
            capacity_Pa=cr,
            governing_mode="radial",
            utilisation=0.0,
        )
        buckling = partial.buckling_utilisation
        yielding = partial.yield_utilisation
        if math.isfinite(yielding) and yielding > buckling:
            layers_out[layer.name] = replace(
                partial, governing_mode="yield", utilisation=yielding
            )
        else:
            layers_out[layer.name] = replace(partial, utilisation=buckling)
    # Open or closed as the contact solver decides it (no line force), not by
    # the reported gap, whose round-off follows the linear algebra.
    open_ifaces = tuple("/".join(name) for name in result.open_interfaces)
    from cablexsec.analysis.components import (  # noqa: PLC0415
        capacity_utilisation,
        compute_component_response,
    )

    response = compute_component_response(
        cable, float(kappa), 0.0, tension_N=float(tension_N), mu=mu
    )
    components: dict[str, tuple[float, str]] = {}
    kinematic: dict[str, tuple[float, str]] = {}
    for item in response.states:
        if item.component_type == "armour wire":
            continue
        usage, mode = capacity_utilisation(item)
        if not math.isfinite(usage):
            continue
        target = kinematic if item.kinematic else components
        target[item.name] = (float(usage), mode)
    caveats: list[str] = []
    if not all(x.elastic_valid for x in layers_out.values()):
        caveats.append(_BEYOND_YIELD)
    if counter_helical_armour(cable):
        caveats.append(_COUNTER_HELICAL)
    provenance = (
        replace(_PROVENANCE, limitations=(*_PROVENANCE.limitations, *caveats))
        if caveats
        else _PROVENANCE
    )
    return CoilingResult(
        tank_diameter_m=tank_diameter_m,
        curvature_per_m=float(kappa),
        twist_per_m=solved_twist,
        torque_Nm=float(result.torque_Nm),
        axial_strain=float(result.axial_strain),
        tension_N=float(tension_N),
        layers=layers_out,
        open_interfaces=open_ifaces,
        provenance=provenance,
        components=components,
        kinematic=kinematic,
    )
