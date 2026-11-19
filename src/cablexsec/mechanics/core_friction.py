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
:class:`~cablexsec.mechanics.papailiou.PapailiouLayer`) and the slip bounds
``sigma_0 expm1(mu_eff |sin beta| psi)`` (rising stress) and
``sigma_0 (1 - exp(-mu_eff |sin beta| psi))`` (falling stress) are those of a
wire whose two faces carry ``q_core`` (friction accumulates from the
neutral-axis crossing over the core length ``r_c psi / sin beta``). The
effective friction coefficient is the load-weighted mean
``mu_eff = sum(mu_i q_i) / sum(q_i)`` of the core--core, core--filler and
core--sheath contacts.

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
of one core. Otherwise the bounded estimate. Each core carries two inward
loads per metre of cable: its capstan load ``C = N_core tan^2 beta / r_c``
(``N_core`` its cable-axis tension share, as ``capstan_force_per_wire`` for an
armour wire) and its share ``R / n`` of the line force ``R = 2 pi r P`` that
the bounded pressure chain applies inside the innermost armour (the armour
winding-stress increment superposed as for the armour clamping). ``R / n``
reaches the core on its outer face, through the filler or directly from the
ring. The inward resultant ``F = C + R / n`` is reacted on the inner side of
the core:

* without a central filler (``CoreFrictionSpec.central_filler`` ``None`` or
  ``False``, the default) by the two neighbouring cores of a touching
  trefoil, whose contact normals lie ``pi/2 - pi/n`` off the radial
  direction, so that the two contact forces sum to ``F / sin(pi/n)``
  (1.155 ``F`` for three cores; two cores touch along the radial line);
* with a central filler (``central_filler=True``) by that filler, radially.

The normal load of a core is then ``R/n + F/sin(pi/n)`` (wedged trefoil) or
``R/n + F`` (central filler). The earlier estimate put ``C + R/n`` on one
contact and left out the reaction on the inner side.

Filler profiles
---------------
With ``CoreFrictionSpec(filler_profiles=True)`` the interstitial filler (the
non-structural filler annulus around the cores) is a population of ``n``
shaped profiles laid with the cores (same pitch), each sticking to or
slipping against the cores and the ring with the same direction-tied cap and
return map. The filler region is the annulus minus the core envelopes
(circles of the core radius in the cable cross-section); its area ``A_f`` and
polar moment ``J_f`` about the cable axis give the effective radius
``r_f = sqrt(J_f / A_f)``, so that ``1/2 E A_f r_f^2`` is the stuck bending
stiffness ``E J_f / 2`` of the region (times ``xi cos^3 beta_f``), and the lay
angle ``tan beta_f = 2 pi r_f / pitch``. A profile is loaded on both faces:
the ring line force ``R / n`` on its outer face and the same force, passed
on, against the cores (bounded estimate), or the solved filler--ring and
filler--core forces divided among the profiles (solved), so ``q_f = 2 R / n``
in the bounded estimate. Its stick coefficient is the Yun form with the
contact penalty and the shear of half the mean filler depth
``A_f / (2 pi r_b)`` over the profile perimeter ``4 pi r_b / n`` (``r_b`` the
filler outer radius). The profiles carry no share of the cable tension (the
filler stays non-structural axially), so they have no capstan load.

Friction models
---------------
:func:`core_friction_history` takes the friction model of the cores and the
filler profiles as the armour does (``friction_model``):

* ``"direction_tied"`` (default): every station a spring--slider with the
  Papailiou cap of :func:`wire_slip_history`, tied to the current bending
  direction. Under rotating curvature the cap turns with the bending
  direction and the friction work is too low (see :func:`wire_slip_history`).
* ``"distributed"``: each population (cores, profiles) is the distributed
  helix model of the armour
  (:func:`~cablexsec.mechanics.distributed_friction.distributed_friction_history`):
  the axial force of a core is a field along one pitch of its helix with the
  shear-lag interface of the stick coefficient ``xi`` (``k_hat = xi / (1 -
  xi)``), the Coulomb bound of its contact pressure and the signed capstan
  term ``lam = mu_eff |sin beta|``, and the slip of every helix node in the
  direction of its own traction, with no bending direction. The stations
  sample that field. The stress unit is the same ``sigma_0`` (the pressure
  times the station width is the line load ``q_core`` of a core), so the
  planar full-slip field is ``sigma_0 (C exp(lam x) - 1)``,
  ``C = lam pi / (2 sinh(lam pi / 2))`` (``x`` the azimuth from the
  geometric neutral axis toward tension), and the full-slip moment is that of
  the direction-tied law times ``C``. It needs a compliant interface
  (``xi < 1``).
* ``"auto"``: the direction-tied law on a planar curvature path (out-of-plane
  ratio at most :data:`~cablexsec.analysis.fatigue.PLANAR_PATH_TOLERANCE`, as
  in the fatigue route), the distributed law otherwise; a rigid population
  keeps the direction-tied law with a
  :class:`~cablexsec.provenance.FrictionModelFallbackWarning`.

The internal rings keep the [KarlsenHusoy2026] operator in every model.

Contact loads over a tension history
------------------------------------
:func:`core_load_history` evaluates the core and profile contact loads of the
contact model at every sample of a tension history (the bounded chain is
affine in tension, so two evaluations give it; the solved model is solved at
up to 33 tensions and interpolated beyond, as the armour histories). The
friction bound of each population then follows its own load. The profile
pressure is proportional to its face load. The core pressure is the pressure
of the sample's line load times the ratio of the population's full-slip
friction moment at the sample's load-weighted coefficient to that at the
reference coefficient, so that loading the contact types in another
proportion changes the bound level as it changes the group cap of an
exported law's slip scale
(:class:`~cablexsec.analysis.bending_law.SlipScaleTable`); the coefficient in
the capstan term stays at the reference value. The internal rings take the
tension history; their pressure-held part stays at the reference core load.

Assumptions (stated, not solved)
--------------------------------
* Normal contact loads are those of the axisymmetric state, at the
  reference tension or following a tension history
  (:func:`core_load_history`), and are not changed by bending (as for the
  armour); the bending redistribution of core contact is not solved.
* The wedged trefoil assumes touching cores. The outer-face load ``R / n`` of
  a core and the face loads ``2 R / n`` of a profile are estimates: a profile
  face is not radial, and part of ``R`` reaches the cores directly.
* A central filler reacts the inward core loads; it is not part of the
  slipping filler-profile population.
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

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Literal

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
    wire_slip_stress_bounds,
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
    "CORE_FRICTION_MODELS",
    "CORE_LOAD_MODES",
    "CoreContactLoads",
    "CoreFrictionHistory",
    "CoreFrictionModel",
    "CoreFrictionModelName",
    "CoreFrictionSpec",
    "CoreLoadHistory",
    "CoreLoadMode",
    "FillerProfiles",
    "StrandRingFriction",
    "build_core_friction",
    "core_contact_loads",
    "core_friction_history",
    "core_load_history",
    "resolve_core_friction",
]

FloatArray = NDArray[np.float64]

#: Friction model of the core law (:func:`core_friction_history`).
CoreFrictionModelName = Literal["direction_tied", "distributed", "auto"]
CORE_FRICTION_MODELS: tuple[str, ...] = ("direction_tied", "distributed", "auto")
#: Core contact loads of a route with a tension history: ``"follow"`` the
#: history (:func:`core_load_history`) or hold the ``"reference"`` state.
CoreLoadMode = Literal["follow", "reference"]
CORE_LOAD_MODES: tuple[str, ...] = ("follow", "reference")
#: Solved tensions of a solved-model load history before it is interpolated.
_SOLVED_LOAD_NODES = 33

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
    central_filler : bool or None
        Load path of the inward core loads in the bounded estimate (module
        notes): ``None`` or ``False`` (default) for a touching trefoil wedged
        on its neighbours, ``True`` for a central filler that carries them
        radially. The solved contact model reads the core support of the
        solved radial state and accepts only ``None``.
    filler_profiles : bool
        Treat the interstitial filler as laid frictional profiles, one per
        core interstice (module notes; default False). Needs a non-structural
        filler annulus around the cores and a core lay.
    mu_filler_profile : float or None
        Friction coefficient of the profile contacts (default:
        ``mu_core_filler``).
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
    central_filler: bool | None = None
    filler_profiles: bool = False
    mu_filler_profile: float | None = None

    @property
    def filler_mu(self) -> float:
        """Friction coefficient of the filler-profile contacts."""
        if self.mu_filler_profile is None:
            return self.mu_core_filler
        return self.mu_filler_profile

    def __post_init__(self) -> None:
        for name in ("mu_core_filler", "mu_core_core", "mu_core_sheath"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise CableInputError(f"{name} must be finite and non-negative")
        if self.central_filler is not None and not isinstance(
            self.central_filler, bool
        ):
            raise CableInputError("central_filler must be None, True or False")
        if not isinstance(self.filler_profiles, bool):
            raise CableInputError("filler_profiles must be True or False")
        for name in ("mu_strand", "mu_strand_outer", "mu_filler_profile"):
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
    model that produced them. ``filler_profile_N_per_m`` is the sum of the
    normal loads on both faces of one filler profile (module notes), the
    load of the filler-profile population.
    """

    core_core_N_per_m: float
    core_filler_N_per_m: float
    core_sheath_N_per_m: float
    source: str
    filler_profile_N_per_m: float = 0.0

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
    cable: CableData,
    axi: AxisymmetricResult,
    n_cores: int,
    r_c: float,
    beta: float,
    central_filler: bool,
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
    # Share of the ring line force on one core's outer face (through the
    # filler where there is one, else directly from the ring).
    ring = 2.0 * math.pi * float(axi.interface_radii_m[0]) * pressure / n_cores
    has_filler = _has_filler(cable)
    inward = capstan + ring
    filler_face = ring if has_filler else 0.0
    sheath_face = 0.0 if has_filler else ring
    # A profile carries the ring force of its interstice on its outer face
    # and passes it on to the cores: both faces, 2 R / n.
    profile = 2.0 * ring if has_filler else 0.0
    if central_filler:
        # The central filler reacts the inward resultant radially.
        return CoreContactLoads(
            0.0, filler_face + inward, sheath_face, "bounded", profile
        )
    # Wedged trefoil: each core rests on its two neighbours, whose contact
    # normals lie pi/2 - pi/n off the radial direction, so the two contact
    # forces sum to F / sin(pi/n) (1.155 F for three cores); two cores touch
    # along the radial line (sin(pi/2) = 1).
    core_core = inward / math.sin(math.pi / n_cores)
    return CoreContactLoads(core_core, filler_face, sheath_face, "bounded", profile)


def core_contact_loads(
    cable: CableData,
    tension_N: float = 0.0,
    *,
    contact_model: str = "bounded",
    radial_state: CableRadialState | None = None,
    section_state: SectionState | None = None,
    central_filler: bool | None = None,
) -> CoreContactLoads:
    """Normal contact line loads of one laid core (see the module notes).

    ``central_filler`` selects the support of the inward core loads in the
    bounded estimate (``None``/``False``: wedged trefoil; ``True``: central
    filler); the solved model reads its own and accepts only ``None``.
    """
    n_cores, r_c, beta, _ = _core_geometry(cable)
    if radial_state is not None or contact_model == "solved":
        if section_state is not None:
            raise CableSchemaError("supply a section state or a radial state, not both")
        if central_filler is not None:
            raise CableInputError(
                "central_filler applies to the bounded core contact estimate; the "
                "solved radial state sets the core support itself"
            )
        if radial_state is None:
            from cablexsec.mechanics.radial_cable import (  # noqa: PLC0415
                solve_cable_radial_state,
            )

            radial_state = solve_cable_radial_state(cable, tension_N)
        radial_state.require_cable(cable)
        bundle = radial_state.result.bundle
        if bundle is None:
            raise CableSchemaError("the solved radial state has no core bundle")
        filler_core = max(bundle.filler_core_force_N_per_m, 0.0)
        # One profile per interstice: the filler--ring force of the whole
        # ring and the filler--core forces of all cores, shared equally.
        profile = (
            max(bundle.filler_ring_force_N_per_m, 0.0) + n_cores * filler_core
        ) / n_cores
        return CoreContactLoads(
            2.0 * max(bundle.core_core_force_N_per_m, 0.0),
            filler_core,
            max(bundle.core_ring_force_N_per_m, 0.0),
            "solved",
            profile,
        )
    from cablexsec.clamping import require_radial_shrinkage_route  # noqa: PLC0415

    require_radial_shrinkage_route(
        cable, "bounded core contact estimate", "contact_model='solved'"
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
        return _bounded_loads(cable, axi, n_cores, r_c, beta, bool(central_filler))
    if contact_model not in {"bounded", "qp"}:
        raise CableSchemaError("contact_model must be 'bounded', 'qp' or 'solved'.")
    axi = compute_contact_forces(
        cable, tension_N, p_ext_Pa=cable.initial_state.external_pressure_Pa
    )
    return _bounded_loads(cable, axi, n_cores, r_c, beta, bool(central_filler))


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
class FillerProfiles:
    """Laid filler profiles as a frictional helical population (module notes).

    ``line_load_N_per_m`` is the sum of the normal loads on both faces of one
    profile; ``EA_profile_N`` the axial rigidity of one profile;
    ``plane_section_EI_Nm2`` the stuck term ``1/2 E A_f r_f^2`` without
    ``xi cos^3 beta_f``.
    """

    population: HelicalLayerSpec
    n_profiles: int
    layer_name: str
    area_m2: float
    r_effective_m: float
    lay_angle_rad: float
    EA_profile_N: float
    line_load_N_per_m: float
    mu: float
    xi: float
    plane_section_EI_Nm2: float

    @property
    def stick_EI_Nm2(self) -> float:
        """Stick stiffness of the profiles, ``xi`` times plane section ``cos^3``."""
        return float(self.population.EI_stick)


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
    fillers: FillerProfiles | None = None

    @property
    def core_stick_EI_Nm2(self) -> float:
        """Stick stiffness of the core population, ``xi`` times plane section."""
        return float(self.population.EI_stick)

    @property
    def strand_stick_EI_Nm2(self) -> float:
        return float(sum(r.stick_EI_Nm2 for r in self.strand_rings))

    @property
    def filler_stick_EI_Nm2(self) -> float:
        return 0.0 if self.fillers is None else self.fillers.stick_EI_Nm2

    @property
    def stick_EI_Nm2(self) -> float:
        return (
            self.core_stick_EI_Nm2 + self.strand_stick_EI_Nm2 + self.filler_stick_EI_Nm2
        )

    @property
    def populations(self) -> tuple[HelicalLayerSpec, ...]:
        """The helical populations of the law: the cores, then the profiles."""
        if self.fillers is None:
            return (self.population,)
        return (self.population, self.fillers.population)


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


def _filler_region(
    r_in: float, r_out: float, n_cores: int, r_c: float, r_core: float
) -> tuple[float, float]:
    """Area and polar moment about the cable axis of a filler annulus less the
    core envelopes (circles of radius ``r_core`` at ``r_c``) [m², m⁴].

    At radius ``r`` a core circle covers the angle
    ``2 acos((r² + r_c² - r_core²) / (2 r r_c))``; the integrals over ``r`` are
    Gauss--Legendre sums on the pieces between the core-circle radii, where
    the integrand has square-root ends (relative error below 1e-6).
    """
    nodes, weights = np.polynomial.legendre.leggauss(64)
    edges = sorted(
        {r_in, r_out, *(x for x in (r_c - r_core, r_c + r_core) if r_in < x < r_out)}
    )
    area = polar = 0.0
    for a, b in zip(edges[:-1], edges[1:], strict=False):
        # Substitute r = a + (b - a) (1 - cos t) / 2 to smooth the ends.
        t = 0.5 * math.pi * (nodes + 1.0)
        r = a + 0.5 * (b - a) * (1.0 - np.cos(t))
        dr = 0.5 * (b - a) * np.sin(t) * 0.5 * math.pi
        cosine = np.clip((r**2 + r_c**2 - r_core**2) / (2.0 * r * r_c), -1.0, 1.0)
        covered = np.where(np.abs(r - r_c) < r_core, 2.0 * np.arccos(cosine), 0.0)
        width = np.maximum(2.0 * math.pi - n_cores * covered, 0.0)
        area += float(np.sum(weights * width * r * dr))
        polar += float(np.sum(weights * width * r**3 * dr))
    return area, polar


def _filler_profiles(
    cable: CableData,
    spec: CoreFrictionSpec,
    loads: CoreContactLoads,
    n_cores: int,
    r_c: float,
    beta: float,
    r_core: float,
) -> FillerProfiles:
    fillers = [
        x
        for x in cable.layers
        if isinstance(x, CylindricalLayer) and is_interstitial_filler(x, cable)
    ]
    if len(fillers) != 1:
        raise CableSchemaError(
            "filler profiles need exactly one interstitial filler layer around "
            f"the cores (found {len(fillers)})"
        )
    layer = fillers[0]
    if layer.structural:
        # A structural filler annulus already bends in EI_min; the profiles
        # would add the same material a second time.
        raise CableSchemaError(
            f"filler profiles need a non-structural filler annulus; "
            f"{layer.name!r} is structural (declare structural: false)"
        )
    r_in, r_out = float(layer.r_inner_m or 0.0), float(layer.r_outer_m or 0.0)
    area, polar = _filler_region(r_in, r_out, n_cores, r_c, r_core)
    if not (area > 0.0 and polar > 0.0):
        raise CableSchemaError(
            f"filler layer {layer.name!r} leaves no area around the cores"
        )
    try:
        e_f = library.get_property(layer.material, "E_Pa")
        nu_f = library.get_property(layer.material, "nu")
    except KeyError:
        raise CableInputError(
            f"filler layer {layer.name!r}: material {layer.material!r} has no "
            "isotropic modulus (E_Pa, nu) for the filler profiles"
        ) from None
    r_f = math.sqrt(polar / area)
    beta_f = math.atan(math.tan(beta) * r_f / r_c)
    ea_profile = e_f * area / n_cores
    stack: list[tuple[float, float, float]] = []
    perimeter = 4.0 * math.pi * r_out / n_cores
    if math.isfinite(spec.interface_stiffness_Pa_per_m):
        stack.append((1.0, spec.interface_stiffness_Pa_per_m, perimeter))
    if spec.core_shear:
        depth = area / (2.0 * math.pi * r_out)
        stack.append((0.5 * depth, shear_modulus(e_f, nu_f), perimeter))
    xi_f = yun_xi(series_shear_stiffness(stack), ea_profile, beta_f, r_f)
    n_st = n_cores * spec.n_stations_per_core
    q_f = loads.filler_profile_N_per_m
    population = HelicalLayerSpec(
        n_wires=n_st,
        r_helix_m=r_f,
        lay_angle_rad=beta_f,
        E_wire_Pa=e_f,
        A_wire_m2=area / n_st,
        xi=xi_f,
        mu=spec.filler_mu,
        F_n_axi_N=q_f,
        contact_pressure_Pa=n_cores * q_f / (2.0 * math.pi * r_f),
        normal_load_is_line=True,
    )
    return FillerProfiles(
        population=population,
        n_profiles=n_cores,
        layer_name=layer.name,
        area_m2=area,
        r_effective_m=r_f,
        lay_angle_rad=beta_f,
        EA_profile_N=ea_profile,
        line_load_N_per_m=q_f,
        mu=spec.filler_mu,
        xi=xi_f,
        plane_section_EI_Nm2=0.5 * e_f * area * r_f**2,
    )


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
    from cablexsec.clamping import (  # noqa: PLC0415
        warn_if_calibrated_with_armour_only,
    )

    warn_if_calibrated_with_armour_only(cable, stacklevel=2)
    loads = core_contact_loads(
        cable,
        tension_N,
        contact_model=contact_model,
        radial_state=radial_state,
        section_state=section_state,
        central_filler=spec.central_filler,
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
        fillers=(
            _filler_profiles(cable, spec, loads, n_cores, r_c, beta, r_core)
            if spec.filler_profiles
            else None
        ),
    )


@dataclass(frozen=True)
class CoreLoadHistory:
    """Core and filler-profile contact loads over a tension history.

    ``core_N_per_m`` is the total normal line load of one core and
    ``core_mu_effective`` its load-weighted friction coefficient at every
    sample; ``core_pressure_Pa`` the friction-setting pressure of the core
    population (module notes). ``filler_profile_N_per_m`` and
    ``filler_pressure_Pa`` are those of the filler profiles (``None``
    without them). ``contact_model`` names the model the loads come from.
    """

    tension_N: FloatArray
    core_N_per_m: FloatArray
    core_mu_effective: FloatArray
    core_pressure_Pa: FloatArray
    contact_model: str
    filler_profile_N_per_m: FloatArray | None = None
    filler_pressure_Pa: FloatArray | None = None


def _population_cap_per_pressure(spec: HelicalLayerSpec, mu: float) -> float:
    """Planar full-slip friction moment of a population per unit pressure,
    without the common factor ``A cos(beta) r``."""
    unit = dataclasses.replace(spec, mu=mu, contact_pressure_Pa=1.0)
    upper, lower = wire_slip_stress_bounds(unit, 0.0)
    c = np.cos(unit.phi_wires)
    return float(np.sum(np.where(c > 0.0, upper, lower) * np.abs(c)))


def _load_nodes(tension: FloatArray, contact_model: str) -> FloatArray:
    """Tensions at which the contact model is evaluated (module notes)."""
    if contact_model != "solved":
        # Bounded estimate (also the core loads of the qp model): affine.
        return np.array([0.0, max(float(np.max(tension)), 1.0)])
    unique = np.unique(tension)
    if unique.size <= _SOLVED_LOAD_NODES:
        return np.asarray(unique, dtype=np.float64)
    return np.linspace(float(unique[0]), float(unique[-1]), _SOLVED_LOAD_NODES)


def core_load_history(
    cable: CableData,
    model: CoreFrictionModel,
    tension_N: FloatArray,
    *,
    contact_model: str = "bounded",
) -> CoreLoadHistory:
    """Core and filler-profile contact loads of ``model`` over a tension history.

    The loads of :func:`core_contact_loads` with the model's core support at
    every sample of ``tension_N`` [N] (finite, non-negative, 1-D): exact for
    the bounded estimate (affine in tension, two evaluations; the ``"qp"``
    model takes the bounded estimate for the cores too), and for the solved
    model exact up to 33 distinct tensions and interpolated linearly between
    33 equally spaced solutions beyond. Pass the result to
    :func:`core_friction_history` (``loads``) so that the friction bounds of
    the cores and profiles follow the history (module notes).
    """
    tension = np.asarray(tension_N, dtype=np.float64)
    if tension.ndim != 1 or not tension.size or not np.isfinite(tension).all():
        raise CableInputError(
            "the tension history must be a finite, non-empty 1-D array"
        )
    if np.any(tension < 0.0):
        raise CableInputError("the tension history must not be negative")
    if contact_model not in {"bounded", "qp", "solved"}:
        raise CableSchemaError("contact_model must be 'bounded', 'qp' or 'solved'.")
    nodes = _load_nodes(tension, contact_model)
    support = None if contact_model == "solved" else model.spec.central_filler
    at_nodes = [
        core_contact_loads(
            cable, float(value), contact_model=contact_model, central_filler=support
        )
        for value in nodes
    ]

    def follow(name: str) -> FloatArray:
        values = np.array([getattr(item, name) for item in at_nodes])
        if nodes.size == 1:
            return np.full(tension.shape, float(values[0]))
        return np.asarray(np.interp(tension, nodes, values), dtype=np.float64)

    core_core = follow("core_core_N_per_m")
    filler = follow("core_filler_N_per_m")
    sheath = follow("core_sheath_N_per_m")
    total = core_core + filler + sheath
    spec = model.spec
    weighted = (
        spec.mu_core_core * core_core
        + spec.mu_core_filler * filler
        + spec.mu_core_sheath * sheath
    )
    positive = total > 0.0
    mu = np.full(total.shape, spec.mu_core_filler)
    mu[positive] = weighted[positive] / total[positive]
    population = model.population
    pressure = model.n_cores * total / (2.0 * math.pi * model.r_offset_m)
    reference_mu = float(population.mu)
    if np.any(np.abs(mu - reference_mu) > 1e-12 * max(reference_mu, 1e-300)):
        # The full-slip friction moment of the population at each sample's
        # coefficient over that at the reference coefficient.
        reference = _population_cap_per_pressure(population, reference_mu)
        values, inverse = np.unique(mu, return_inverse=True)
        ratio = np.array(
            [
                _population_cap_per_pressure(population, float(value)) / reference
                if reference > 0.0
                else 1.0
                for value in values
            ]
        )
        pressure = pressure * ratio[inverse]
    profile = profile_pressure = None
    if model.fillers is not None:
        profile = follow("filler_profile_N_per_m")
        r_f = model.fillers.population.r_helix_m
        profile_pressure = model.fillers.n_profiles * profile / (2.0 * math.pi * r_f)
    return CoreLoadHistory(
        tension_N=tension,
        core_N_per_m=np.asarray(total, dtype=np.float64),
        core_mu_effective=np.asarray(mu, dtype=np.float64),
        core_pressure_Pa=np.asarray(pressure, dtype=np.float64),
        contact_model=contact_model,
        filler_profile_N_per_m=profile,
        filler_pressure_Pa=profile_pressure,
    )


@dataclass(frozen=True)
class CoreFrictionHistory:
    """Moments ``(samples, 2)`` and work of the cores, internal rings and
    filler profiles (``filler_moment_Nm`` zero without profiles).

    ``friction_model`` is the law the cores and profiles were solved with
    (``"direction_tied"`` or ``"distributed"``; ``"auto"`` resolved).
    ``core_slip_fraction`` is the fraction of core stations in slip
    (direction-tied) or of helix nodes on their bound (distributed).
    ``loads`` is the load history the bounds followed (``None``: the
    model's reference loads).
    """

    core_moment_Nm: FloatArray
    strand_moment_Nm: FloatArray
    core_slip_fraction: FloatArray
    work: BendingWork
    model: CoreFrictionModel
    filler_moment_Nm: FloatArray | None = None
    friction_model: str = "direction_tied"
    loads: CoreLoadHistory | None = None

    @property
    def moment_Nm(self) -> FloatArray:
        total = self.core_moment_Nm + self.strand_moment_Nm
        if self.filler_moment_Nm is not None:
            total = total + self.filler_moment_Nm
        return np.asarray(total)


def _strand_history(
    model: CoreFrictionModel,
    kx: FloatArray,
    ky: FloatArray,
    tension_N: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    count = kx.size
    moment = np.zeros((count, 2))
    stored, friction, endpoint, remainder = (np.zeros(count) for _ in range(4))
    tension = np.full(count, model.tension_N) if tension_N is None else tension_N
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


def _resolve_core_model(
    model: CoreFrictionModel,
    friction_model: str,
    kx: FloatArray,
    ky: FloatArray,
    stacklevel: int,
) -> str:
    """Friction model of the core law over this path (``"auto"`` resolved)."""
    if friction_model not in CORE_FRICTION_MODELS:
        raise CableInputError(
            f"the core friction model must be one of {CORE_FRICTION_MODELS}, "
            f"got {friction_model!r}"
        )
    if friction_model == "direction_tied":
        return friction_model
    from cablexsec.analysis.fatigue import (  # noqa: PLC0415
        PLANAR_PATH_TOLERANCE,
        _warn_auto_fallback,
        curvature_path_out_of_plane_ratio,
    )

    if (
        friction_model == "auto"
        and curvature_path_out_of_plane_ratio(kx, ky) <= PLANAR_PATH_TOLERANCE
    ):
        return "direction_tied"
    rigid = [
        f"the {name} (xi = {pop.xi:.6g})"
        for name, pop in zip(
            ("cores", "filler profiles"), model.populations, strict=False
        )
        if not (math.isfinite(pop.xi) and 0.0 <= pop.xi < 1.0)
    ]
    if not rigid:
        return "distributed"
    reason = f"the interface is rigid for {' and '.join(rigid)}"
    if friction_model == "distributed":
        raise CableInputError(
            f"the distributed core law needs a compliant interface (xi < 1): "
            f"{reason}; give a finite interface stiffness or the core shear "
            "(CoreFrictionSpec), or use friction_model='direction_tied'"
        )
    _warn_auto_fallback("the laid-core law", reason, stacklevel + 1)
    return "direction_tied"


def _distributed_population(
    spec: HelicalLayerSpec,
    kx: FloatArray,
    ky: FloatArray,
    pressure: FloatArray | None,
    nodes_per_pitch: int | None,
) -> tuple[FloatArray, BendingWork, FloatArray]:
    """Moment ``(samples, 2)``, work and node slip fraction of one population
    from the distributed helix model."""
    from cablexsec.mechanics.distributed_friction import (  # noqa: PLC0415
        distributed_friction_history,
    )

    history = distributed_friction_history(
        spec, kx, ky, contact_pressure_Pa=pressure, nodes_per_pitch=nodes_per_pitch
    )
    phi = np.asarray(spec.phi_wires, dtype=np.float64)
    weight = spec.A_wire_m2 * math.cos(spec.lay_angle_rad) * spec.r_helix_m
    directions = np.column_stack((np.cos(phi), np.sin(phi)))
    moment = weight * (history.stress_Pa.T @ directions)
    work = BendingWork(
        history.stored_J_per_m,
        history.dissipation_J_per_m,
        history.endpoint_work_J_per_m,
        history.integration_remainder_J_per_m,
        history.balance_residual_J_per_m,
    )
    return np.asarray(moment, dtype=np.float64), work, history.slip_fraction


def core_friction_history(
    model: CoreFrictionModel,
    kappa_x: FloatArray,
    kappa_y: FloatArray,
    *,
    initial_bending_angle: float = 0.0,
    friction_model: CoreFrictionModelName = "direction_tied",
    loads: CoreLoadHistory | None = None,
    nodes_per_pitch: int | None = None,
) -> CoreFrictionHistory:
    """Biaxial history of the core population, the internal rings and the
    filler profiles.

    ``friction_model`` selects the law of the cores and the profiles (module
    notes): ``"direction_tied"`` (default) uses :func:`wire_slip_history`
    unchanged (direction-tied Papailiou cap; ``initial_bending_angle`` is its
    held direction at the start), ``"distributed"`` the distributed helix
    model with ``nodes_per_pitch`` helix nodes per pitch (default four per
    station), ``"auto"`` the direction-tied law on a planar path and the
    distributed law otherwise. The rings use :func:`strand_slip_strain`.
    ``loads`` (:func:`core_load_history`, one sample per curvature sample)
    makes the contact loads follow a tension history; ``None`` holds the
    model's reference loads. All return moments in the material frame of the
    curvature components.
    """
    kx = np.asarray(kappa_x, dtype=np.float64)
    ky = np.asarray(kappa_y, dtype=np.float64)
    core_pressure: FloatArray | None = None
    filler_pressure: FloatArray | None = None
    tension: FloatArray | None = None
    if loads is not None:
        if loads.tension_N.shape != kx.shape:
            raise CableInputError(
                "the core load history must have one sample per curvature sample"
            )
        if (loads.filler_pressure_Pa is None) != (model.fillers is None):
            raise CableInputError(
                "the core load history belongs to another core model "
                "(its filler profiles differ)"
            )
        core_pressure = loads.core_pressure_Pa
        filler_pressure = loads.filler_pressure_Pa
        tension = loads.tension_N
    resolved = _resolve_core_model(model, friction_model, kx, ky, stacklevel=2)
    s_moment, s_stored, s_friction, s_endpoint, s_remainder = _strand_history(
        model, kx, ky, tension
    )
    if resolved == "distributed":
        core_moment, w, slip = _distributed_population(
            model.population, kx, ky, core_pressure, nodes_per_pitch
        )
    else:
        history = wire_slip_history(
            model.population,
            kx,
            ky,
            contact_pressure_Pa=core_pressure,
            initial_bending_angle=initial_bending_angle,
            direction_hold_fraction=model.spec.direction_hold_fraction,
        )
        core_moment, w = np.asarray(history.moment_Nm), history.work
        slip = history.slipping_count / float(model.population.n_wires)
    stored = w.stored_J_per_m + s_stored
    friction = w.friction_J_per_m + s_friction
    endpoint = w.endpoint_work_J_per_m + s_endpoint
    remainder = w.integration_remainder_J_per_m + s_remainder
    filler_moment = None
    if model.fillers is not None:
        if resolved == "distributed":
            filler_moment, fw, _ = _distributed_population(
                model.fillers.population, kx, ky, filler_pressure, nodes_per_pitch
            )
        else:
            filler = wire_slip_history(
                model.fillers.population,
                kx,
                ky,
                contact_pressure_Pa=filler_pressure,
                initial_bending_angle=initial_bending_angle,
                direction_hold_fraction=model.spec.direction_hold_fraction,
            )
            filler_moment, fw = np.asarray(filler.moment_Nm), filler.work
        stored = stored + fw.stored_J_per_m
        friction = friction + fw.friction_J_per_m
        endpoint = endpoint + fw.endpoint_work_J_per_m
        remainder = remainder + fw.integration_remainder_J_per_m
    return CoreFrictionHistory(
        core_moment_Nm=core_moment,
        strand_moment_Nm=s_moment,
        core_slip_fraction=np.asarray(slip, dtype=np.float64),
        work=BendingWork(
            stored,
            friction,
            endpoint,
            remainder,
            endpoint - stored - friction - remainder,
        ),
        model=model,
        filler_moment_Nm=filler_moment,
        friction_model=resolved,
        loads=loads,
    )
