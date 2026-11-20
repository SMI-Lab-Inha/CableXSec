# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Bending-test simulator: what a three-point, four-point or cantilever test measures.

A bending test does not impose a uniform curvature. In a three-point test the
moment rises linearly from the supports to the load point; in a four-point
test it is uniform only between the loading points; a cantilever carries the
largest moment at the root. The test then reduces a force and a displacement
(or a few deflection readings) to one "moment" and one "curvature". A
hysteretic section compared with such a test at uniform curvature is compared
with a different quantity (gap G8: the gradual transition of the Hu et al.
(2022) conductor loop is the three-point arrangement).

:func:`simulate_bending_test` reproduces the test:

* the specimen is discretised along its length into stations (nodes);
* every station carries the section's history-dependent moment--curvature law
  (:class:`SectionLaw`: the :func:`~cablexsec.api.compute_mkappa` law of a
  cable with any of its friction models, the conductor strand-slip operator or
  any user law);
* the set-ups are statically determinate, so the moment at every station is
  the applied load times a known influence line, ``M(x, t) = F(t) m(x)``; at
  each load step the curvature of every station is found by inverting its own
  law from its own state;
* load, displacement, moment or curvature control (the curvature as the test
  defines it, which needs an outer solve for the load);
* outputs are what the test reports: load and load-point displacement, the
  moment at the measuring station from the test's own formula, the curvature
  as the test defines it (true station curvature, elastic deflection formula
  or a least-squares fit through deflection gauges), the curvature field and
  the external, stored and dissipated energies.

Mechanics and numerics
----------------------
Deflections follow from the curvature field by virtual work,
``w(xi) = int kappa(x) G(x, xi) dx`` with ``G`` the moment of a unit load at
``xi`` (simply supported span or cantilever), integrated exactly for a
curvature that is linear between stations. The load-point displacement is the
work conjugate of the total load, ``delta = int kappa(x) m(x) dx``.

The section laws are rate-independent and, along a monotone leg, the state at
a curvature depends only on the state at the start of the leg and that
curvature (play/Iwan operators with an exact return map). Each station is
therefore advanced leg by leg: the moment along a leg is evaluated in one call
over a sorted set of trial curvatures appended to the station's reduced
history, and every sample of the leg is solved by a bracketed secant and
inverse-quadratic search. A final call over the full curvature history of
every station gives the reported moments and energies; its departure from
``F m(x)`` is returned as the equilibrium residual and checks the leg
property for the law used.

Limitations
-----------
Euler--Bernoulli small-deflection beam on frictionless supports; no shear
deformation, second-order (P-delta) effect of an axial load or self-weight.
By default each station is an independent section at its own curvature: the
axial redistribution of wire slip between stations of different curvature (a
curvature-gradient effect within a pitch) is not represented. The option
``along_wire=AlongWireSpec(...)`` (gap T2,
:mod:`cablexsec.analysis.along_wire_test`) solves every helical wire along
its helix through the specimen instead, with anchored or free wire ends and
overhangs beyond the supports.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from cablexsec.analysis.along_wire_test import (
    AlongWireOutput,
    AlongWireSection,
    AlongWireSpec,
    run_along_wire_test,
)
from cablexsec.api import _prepare_mkappa_law
from cablexsec.cable_adapters import require_fully_specified
from cablexsec.cables.schema import CableData, uses_cable_materials
from cablexsec.exceptions import CableInputError, CableSolverError
from cablexsec.mechanics.bend_restrictor import ArticulatedRestrictor
from cablexsec.mechanics.core_friction import (
    CoreFrictionSpec,
    core_friction_history,
    resolve_core_friction,
)
from cablexsec.mechanics.core_lay import bending_factor
from cablexsec.mechanics.distributed_friction import distributed_friction_history
from cablexsec.mechanics.progressive_friction import ProgressiveFriction
from cablexsec.mechanics.radial_cable import CableRadialState
from cablexsec.mechanics.section_state import SectionState
from cablexsec.mechanics.strand_slip import (
    CAP_SHAPES,
    CapShape,
    ConductorRing,
    StrandSlipParameters,
    conductor_bending_moment,
)
from cablexsec.provenance import Provenance, real_array, require_finite

__all__ = [
    "AlongWireOutput",
    "AlongWireSection",
    "AlongWireSpec",
    "BendingProtocol",
    "BendingTestResult",
    "Cantilever",
    "FourPoint",
    "SectionLaw",
    "SectionResponse",
    "ThreePoint",
    "simulate_bending_test",
]

FloatArray = NDArray[np.float64]
Control = Literal["load", "displacement", "moment", "curvature"]
CurvatureDefinition = Literal["station", "elastic", "fit"]
_CONTROLS = ("load", "displacement", "moment", "curvature")
_DEFINITIONS = ("station", "elastic", "fit")
_MAX_ITERATIONS = 200
#: Substep of the distributed helix law in a section law [units of the slip
#: onset curvature]. The test solver bisects along monotone legs, so the law's
#: dependence on the sampling of a leg must stay near the equilibrium
#: tolerance. The fatigue default (0.1) leaves 8.5e-8 on the Menard cable in
#: three-point bending once the armour sits on a stiff support (xi 0.69 and
#: 0.89); 0.02 leaves 2e-9.
SECTION_LAW_MAX_INCREMENT = 0.02


# --------------------------------------------------------------------------- #
# Section laws                                                                 #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SectionResponse:
    """Moment and cumulative energies of a section over a curvature history.

    ``friction_J_per_m`` and ``stored_J_per_m`` are ``None`` for laws that do
    not account their work (the loop area then gives the dissipation).
    """

    moment_Nm: FloatArray
    friction_J_per_m: FloatArray | None = None
    stored_J_per_m: FloatArray | None = None


class SectionLaw:
    """History-dependent moment--curvature law of one section (planar bending).

    Parameters
    ----------
    evaluate : callable
        Maps a curvature history (1-D, from the unloaded virgin state) to a
        :class:`SectionResponse`.
    ei_stick_Nm2, ei_slip_Nm2 : float
        Largest and smallest tangent stiffness of the law (no-slip and
        full-slip slopes). They only seed the search; a wrong bound costs
        iterations, not accuracy. ``ei_slip_Nm2`` must be positive.
    description : str
        Text for the provenance of the test result.

    Notes
    -----
    The law must be rate independent and the response along a monotone leg
    must depend only on the state at the start of the leg and the curvature
    reached (true of the package's play and Iwan operators). The equilibrium
    residual of :class:`BendingTestResult` checks this for every run.
    """

    def __init__(
        self,
        evaluate: Callable[[FloatArray], SectionResponse],
        *,
        ei_stick_Nm2: float,
        ei_slip_Nm2: float,
        description: str,
    ) -> None:
        for name, value in (
            ("ei_stick_Nm2", ei_stick_Nm2),
            ("ei_slip_Nm2", ei_slip_Nm2),
        ):
            require_finite(value, name)
        if not 0.0 < ei_slip_Nm2 <= ei_stick_Nm2:
            raise CableInputError("need 0 < ei_slip_Nm2 <= ei_stick_Nm2")
        self._evaluate = evaluate
        self.ei_stick_Nm2 = float(ei_stick_Nm2)
        self.ei_slip_Nm2 = float(ei_slip_Nm2)
        self.description = description

    def evaluate(self, kappa: FloatArray) -> SectionResponse:
        """Response over a curvature history [1/m] from the virgin state."""
        return self._evaluate(np.ascontiguousarray(kappa, dtype=np.float64))

    def moment(self, kappa: FloatArray) -> FloatArray:
        """Moment history [N m] over a curvature history [1/m]."""
        return self.evaluate(kappa).moment_Nm

    # ----------------------------------------------------------------- builders
    @classmethod
    def elastic(cls, ei_Nm2: float) -> SectionLaw:
        """Linear-elastic section ``M = EI kappa`` (verification)."""
        require_finite(ei_Nm2, "ei_Nm2")
        if ei_Nm2 <= 0:
            raise CableInputError("ei_Nm2 must be positive")
        ei = float(ei_Nm2)

        def run(kappa: FloatArray) -> SectionResponse:
            return SectionResponse(
                ei * kappa, np.zeros_like(kappa), 0.5 * ei * kappa**2
            )

        return cls(
            run, ei_stick_Nm2=ei, ei_slip_Nm2=ei, description=f"elastic EI={ei:g} N m^2"
        )

    @classmethod
    @uses_cable_materials
    def from_cable(
        cls,
        cable: CableData,
        *,
        tension_N: float = 0.0,
        mu: float | None = None,
        phi_bend: float = 0.0,
        contact_model: str | None = None,
        section_state: SectionState | None = None,
        radial_state: CableRadialState | None = None,
        core_friction: bool | CoreFrictionSpec | None = None,
        friction_model: Literal[
            "direction_tied", "progressive", "distributed"
        ] = "direction_tied",
        progressive: ProgressiveFriction | None = None,
        distributed_nodes_per_pitch: int | None = None,
        distributed_max_increment: float = SECTION_LAW_MAX_INCREMENT,
    ) -> SectionLaw:
        """The :func:`~cablexsec.api.compute_mkappa` law of a cable.

        Arguments as in :func:`~cablexsec.api.compute_mkappa`. With
        ``friction_model`` ``"direction_tied"`` or ``"progressive"`` the moment
        is bitwise that of :func:`~cablexsec.api.compute_mkappa` for the same
        history. ``"distributed"`` replaces each armour layer's direction-tied
        stress by the distributed helix model of
        :mod:`cablexsec.mechanics.distributed_friction` (the fatigue-path
        option) and assembles the moment from the wire stresses in the same
        way, and solves the laid cores and filler profiles of
        ``core_friction`` with the same model
        (:func:`~cablexsec.mechanics.core_friction.core_friction_history`,
        ``friction_model="distributed"``); it substeps internally
        (``distributed_max_increment``, in units of the slip-onset curvature,
        :data:`SECTION_LAW_MAX_INCREMENT` by default), so its leg property
        holds to the substep accuracy.

        The contact pressure is that of the one contact state given here.
        The pressure and tension histories of
        :func:`~cablexsec.api.compute_mkappa` (``contact_pressure_Pa``,
        ``contact_tension_history_N``) are not offered: a pressure belongs
        to a sample time, and the test solver evaluates every station over
        trial curvature histories (a reduced history plus sorted trial
        curvatures) that carry no time, so the leg property would not hold.

        The law reads the cable's own materials (its ``materials:`` block)
        when it is built and every time it is evaluated, also outside a call
        of :func:`simulate_bending_test`.
        """
        require_fully_specified(cable, "SectionLaw.from_cable")
        require_finite(tension_N, "tension_N")
        require_finite(phi_bend, "phi_bend")
        if friction_model not in ("direction_tied", "progressive", "distributed"):
            raise CableInputError(
                "friction_model must be 'direction_tied', 'progressive' or "
                "'distributed'"
            )
        if progressive is not None and friction_model != "progressive":
            raise CableInputError(
                "progressive settings require friction_model='progressive'"
            )
        if progressive is not None and not isinstance(progressive, ProgressiveFriction):
            raise CableInputError("progressive must be a ProgressiveFriction")
        if core_friction is not None and not isinstance(
            core_friction, (bool, CoreFrictionSpec)
        ):
            raise CableInputError("core_friction must be a bool or CoreFrictionSpec")
        tension_N, contact_model = _contact_state(
            cable, tension_N, contact_model, section_state, radial_state
        )
        settings = (
            (progressive or ProgressiveFriction())
            if friction_model == "progressive"
            else None
        )
        law = _prepare_mkappa_law(
            cable,
            tension_N,
            mu=mu,
            phi_bend=phi_bend,
            contact_model=contact_model,
            section_state=section_state,
            radial_state=radial_state,
            core_spec=resolve_core_friction(core_friction),
            progressive=settings,
        )
        ei_stick = sum(layer.EI_stick for layer in law.layers) + law.ei_min
        if law.core_model is not None:
            ei_stick += law.core_model.stick_EI_Nm2
        ei_slip = law.ei_min
        if ei_slip <= 0:
            raise CableInputError(
                "the cable has no positive full-slip bending stiffness"
            )
        text = (
            f"compute_mkappa law of the cable ({contact_model} contact, "
            f"tension {tension_N:g} N, {friction_model} armour friction"
            + (", laid-core friction" if law.core_model is not None else "")
            + ")"
        )
        if friction_model != "distributed":

            def run(kappa: FloatArray) -> SectionResponse:
                with cable.material_scope():
                    out = law.run(kappa)
                return SectionResponse(
                    out.moment_Nm, out.work.friction_J_per_m, out.work.stored_J_per_m
                )

            return cls(
                run, ei_stick_Nm2=ei_stick, ei_slip_Nm2=ei_slip, description=text
            )

        cos_b, sin_b = math.cos(phi_bend), math.sin(phi_bend)

        def run_distributed(kappa: FloatArray) -> SectionResponse:
            with cable.material_scope():
                return _distributed(kappa)

        def _distributed(kappa: FloatArray) -> SectionResponse:
            kx, ky = kappa * cos_b, kappa * sin_b
            moment = law.ei_min * kappa
            friction = np.zeros_like(kappa)
            stored = 0.5 * law.ei_min * kappa**2
            for spec in law.layers:
                hist = distributed_friction_history(
                    spec,
                    kx,
                    ky,
                    nodes_per_pitch=distributed_nodes_per_pitch,
                    max_increment=distributed_max_increment,
                )
                weight = spec.A_wire_m2 * math.cos(spec.lay_angle_rad) * spec.r_helix_m
                proj = np.cos(spec.phi_wires) * cos_b + np.sin(spec.phi_wires) * sin_b
                moment = moment + weight * (proj @ hist.stress_Pa)
                friction = friction + hist.dissipation_J_per_m
                stored = stored + hist.stored_J_per_m
            if law.core_model is not None:
                core = core_friction_history(
                    law.core_model,
                    kx,
                    ky,
                    friction_model="distributed",
                    nodes_per_pitch=distributed_nodes_per_pitch,
                    max_increment=distributed_max_increment,
                )
                moment = moment + core.moment_Nm @ np.array([cos_b, sin_b])
                friction = friction + core.work.friction_J_per_m
                stored = stored + core.work.stored_J_per_m
            return SectionResponse(moment, friction, stored)

        return cls(
            run_distributed,
            ei_stick_Nm2=ei_stick,
            ei_slip_Nm2=ei_slip,
            description=text,
        )

    @classmethod
    def from_conductor(
        cls,
        rings: Sequence[ConductorRing],
        E_Pa: float,
        parameters: Sequence[StrandSlipParameters],
        *,
        tension_N: float = 0.0,
        additional_EI_Nm2: float = 0.0,
        cap_shape: CapShape = "karlsen_husoy",
    ) -> SectionLaw:
        """Conductor strand-slip law (:func:`conductor_bending_moment`).

        ``parameters`` are the centroid-fibre ring parameters
        (:func:`~cablexsec.mechanics.strand_slip.conductor_ring_parameters`);
        ``additional_EI_Nm2`` adds an elastic part such as a sheath. The
        operator does not account its friction work, so the dissipation of a
        test follows from the loop area.
        """
        if cap_shape not in CAP_SHAPES:
            raise CableInputError(f"cap_shape must be one of {CAP_SHAPES}")
        require_finite(tension_N, "tension_N")
        require_finite(additional_EI_Nm2, "additional_EI_Nm2")
        rings_t = tuple(rings)
        params_t = tuple(parameters)
        if len(rings_t) != len(params_t):
            raise CableInputError("one parameter set per ring is required")
        elastic = float(additional_EI_Nm2)
        stick = slip = 0.0
        for ring, p in zip(rings_t, params_t, strict=True):
            elastic += (
                E_Pa
                * ring.n_wires
                * math.pi
                * ring.d_wire_m**4
                / 64.0
                * bending_factor(ring.lay_angle_rad)
            )
            if ring.r_helix_m == 0:
                continue
            lever = (
                0.5
                * ring.n_wires
                * E_Pa
                * ring.area_m2
                * math.cos(ring.lay_angle_rad)
                * ring.r_helix_m
            )
            stick += lever * float(p.beta_m[0])
            slip += lever * float(p.beta_m[-1])
        tension = float(tension_N)

        def run(kappa: FloatArray) -> SectionResponse:
            moment, _ = conductor_bending_moment(
                rings_t,
                E_Pa,
                params_t,
                kappa,
                np.zeros_like(kappa),
                np.full_like(kappa, tension),
                additional_EI_Nm2=additional_EI_Nm2,
                cap_shape=cap_shape,
            )
            return SectionResponse(moment)

        return cls(
            run,
            ei_stick_Nm2=elastic + stick,
            ei_slip_Nm2=elastic + slip,
            description=f"conductor strand-slip operator ({cap_shape} cap, "
            f"{len(rings_t)} rings, tension {tension:g} N)",
        )


def _contact_state(
    cable: CableData,
    tension_N: float,
    contact_model: str | None,
    section_state: object | None,
    radial_state: object | None,
) -> tuple[float, str]:
    """Tension and contact model from a radial state, a section state or a
    contact-model tension (one source only), as in ``compute_mkappa``."""
    if tension_N < 0:
        raise CableInputError("tension_N must be >= 0")
    if section_state is not None and not isinstance(section_state, SectionState):
        raise CableInputError("section_state must be a SectionState")
    if radial_state is not None and not isinstance(radial_state, CableRadialState):
        raise CableInputError("radial_state must be a CableRadialState")
    if radial_state is not None:
        radial_state.require_cable(cable)
        if (
            section_state is not None
            or tension_N != 0
            or contact_model
            not in {
                None,
                "solved",
            }
        ):
            raise CableInputError(
                "supply a radial state alone, without tension, section state "
                "or another contact model"
            )
        return radial_state.tension_N, "solved"
    if section_state is not None:
        section_state.require_cable(cable)
        if tension_N != 0 or (
            contact_model is not None and contact_model != section_state.contact_model
        ):
            raise CableInputError(
                "supply a section state or a contact-model tension, not both"
            )
        return section_state.tension_N, section_state.contact_model
    model = contact_model or "bounded"
    if model not in {"bounded", "qp", "solved"}:
        raise CableInputError("contact_model must be 'bounded', 'qp' or 'solved'")
    return tension_N, model


def _mkappa_law_for_along_wire(
    cable: object,
    *,
    tension_N: float = 0.0,
    mu: float | None = None,
    contact_model: str | None = None,
    section_state: object | None = None,
    radial_state: object | None = None,
) -> Any:
    """Layer specifications of :meth:`SectionLaw.from_cable` (no core friction)
    for the along-wire section (same contact-state rules)."""
    if not isinstance(cable, CableData):
        raise CableInputError("the along-wire section needs a CableData")
    require_fully_specified(cable, "AlongWireSection.from_cable")
    require_finite(tension_N, "tension_N")
    tension_N, contact_model = _contact_state(
        cable, tension_N, contact_model, section_state, radial_state
    )
    # Types checked by _contact_state.
    assert section_state is None or isinstance(section_state, SectionState)
    assert radial_state is None or isinstance(radial_state, CableRadialState)
    return _prepare_mkappa_law(
        cable,
        tension_N,
        mu=mu,
        phi_bend=0.0,
        contact_model=contact_model,
        section_state=section_state,
        radial_state=radial_state,
        core_spec=None,
        progressive=None,
    )


# --------------------------------------------------------------------------- #
# Arrangements                                                                 #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Beam:
    """Statically determinate beam: key points, influence line and kernels."""

    name: str
    length: float
    kind: Literal["simply_supported", "cantilever"]
    key_points: tuple[float, ...]
    load_points: tuple[float, ...]  # equal share of the total load each
    measuring_x: float
    elastic_factor: float  # kappa = factor * w(elastic_x) for an elastic beam
    elastic_x: float
    formula: str

    def kernel(self, x: FloatArray, xi: float) -> FloatArray:
        """Moment at ``x`` of a unit load at ``xi`` (deflection kernel)."""
        if self.kind == "cantilever":
            return np.maximum(xi - x, 0.0)
        span = self.length
        return np.where(x <= xi, x * (span - xi) / span, xi * (span - x) / span)

    def influence(self, x: FloatArray) -> FloatArray:
        """Moment per unit total load, ``m(x)`` [m]."""
        share = 1.0 / len(self.load_points)
        return np.asarray(sum(share * self.kernel(x, p) for p in self.load_points))


@dataclass(frozen=True)
class ThreePoint:
    """Three-point bending: simple supports ``span_m`` apart, load at mid-span.

    The test reports ``M = F L / 4`` and, by default, ``kappa = 12 w / L^2``
    from the mid-span deflection (Hu et al. 2022, Eqs. 1--2).
    """

    span_m: float

    def _beam(self) -> _Beam:
        span = _positive(self.span_m, "span_m")
        mid = 0.5 * span
        return _Beam(
            f"three-point bending, span {span:g} m",
            span,
            "simply_supported",
            (0.0, mid, span),
            (mid,),
            mid,
            12.0 / span**2,
            mid,
            "M = F L/4; kappa = 12 w/L^2",
        )


@dataclass(frozen=True)
class FourPoint:
    """Four-point bending: supports ``span_m`` apart, two equal loads
    ``loading_span_m`` apart, symmetric; uniform moment between the loads.

    The test reports ``M = (F/2) a`` with ``a`` the shear span; by default the
    curvature is the station curvature in the uniform-moment zone. The
    elastic definition uses the mid-span deflection,
    ``kappa = 24 w / (3 L^2 - 4 a^2)``.
    """

    span_m: float
    loading_span_m: float

    def _beam(self) -> _Beam:
        span = _positive(self.span_m, "span_m")
        inner = _positive(self.loading_span_m, "loading_span_m")
        if inner >= span:
            raise CableInputError("loading_span_m must be shorter than span_m")
        a = 0.5 * (span - inner)
        mid = 0.5 * span
        return _Beam(
            f"four-point bending, span {span:g} m, loading span {inner:g} m",
            span,
            "simply_supported",
            (0.0, a, mid, span - a, span),
            (a, span - a),
            mid,
            24.0 / (3.0 * span**2 - 4.0 * a**2),
            mid,
            "M = F a/2 (a = shear span); kappa uniform between the loads",
        )


@dataclass(frozen=True)
class Cantilever:
    """Cantilever of ``length_m`` built in at ``x = 0``, load at the free end.

    The test reports the root moment ``M = F L``; the elastic curvature
    definition is ``kappa = 3 w_tip / L^2``.
    """

    length_m: float

    def _beam(self) -> _Beam:
        length = _positive(self.length_m, "length_m")
        return _Beam(
            f"cantilever, length {length:g} m",
            length,
            "cantilever",
            (0.0, length),
            (length,),
            0.0,
            3.0 / length**2,
            length,
            "M = F L at the root; kappa = 3 w_tip/L^2",
        )


Arrangement = ThreePoint | FourPoint | Cantilever


def _positive(value: float, name: str) -> float:
    require_finite(value, name)
    if value <= 0:
        raise CableInputError(f"{name} must be positive")
    return float(value)


# --------------------------------------------------------------------------- #
# Protocol                                                                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BendingProtocol:
    """Controlled quantity of the test, sample by sample, from the unloaded state.

    ``control``: ``"load"`` (total applied force [N]), ``"displacement"``
    (work-conjugate load-point displacement [m], the mid-span deflection of a
    three-point test), ``"moment"`` (the moment the test reports at its
    measuring station [N m]) or ``"curvature"`` (the curvature as the test
    defines it [1/m]). ``retain_from`` is the first sample of the retained
    window (for example the last cycle).
    """

    control: Control
    values: FloatArray
    retain_from: int = 0

    def __post_init__(self) -> None:
        if self.control not in _CONTROLS:
            raise CableInputError(f"control must be one of {_CONTROLS}")
        v = real_array(self.values, "values")
        if v.ndim != 1 or v.size < 2 or not np.isfinite(v).all():
            raise CableInputError("values must be a finite 1-D history of >= 2 samples")
        out = np.array(v, dtype=np.float64, copy=True)
        out.flags.writeable = False
        object.__setattr__(self, "values", out)
        if not 0 <= int(self.retain_from) < v.size:
            raise CableInputError("retain_from must index a sample")

    @classmethod
    def cyclic(
        cls,
        control: Control,
        amplitude: float,
        *,
        n_cycles: int = 3,
        samples_per_cycle: int = 400,
        mean: float = 0.0,
    ) -> BendingProtocol:
        """``mean + amplitude sin(2 pi t)`` over ``n_cycles`` (a ramp to the mean
        first if it is not zero); the last cycle is retained."""
        require_finite(amplitude, "amplitude")
        require_finite(mean, "mean")
        if n_cycles < 1 or samples_per_cycle < 8:
            raise CableInputError("need n_cycles >= 1 and samples_per_cycle >= 8")
        t = np.linspace(0.0, n_cycles, n_cycles * samples_per_cycle + 1)[1:]
        cyc = mean + amplitude * np.sin(2.0 * math.pi * t)
        ramp = (
            np.linspace(0.0, mean, samples_per_cycle // 4 + 1)[1:]
            if mean != 0.0
            else np.zeros(0)
        )
        values = np.concatenate([ramp, cyc])
        start = ramp.size + (n_cycles - 1) * samples_per_cycle - 1
        return cls(control, values, max(start, 0))


# --------------------------------------------------------------------------- #
# Result                                                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BendingTestResult:
    """Output of :func:`simulate_bending_test` (sample ``k`` is ``values[k]``).

    Attributes
    ----------
    load_N, displacement_m : numpy.ndarray
        Total applied force and its work-conjugate load-point displacement.
    moment_Nm : numpy.ndarray
        Moment the test reports at its measuring station (``F m(x_meas)``).
    curvature_1_m : numpy.ndarray
        Curvature as the test defines it (``curvature_definition``).
    station_curvature_1_m, station_moment_Nm : numpy.ndarray
        True curvature and law moment at the measuring station; their loop is
        bitwise the section law over that station's curvature history.
    x_m, moment_per_load_m : numpy.ndarray
        Stations and influence line ``m(x)``.
    curvature_field_1_m, moment_field_Nm : numpy.ndarray
        Station curvature and law moment, shape (stations, samples).
    external_work_J : numpy.ndarray
        Cumulative ``int F d(delta)`` (trapezoidal in time).
    friction_work_J, stored_energy_J : numpy.ndarray or None
        Cumulative frictional work and stored energy over the span (trapezoid
        in x) when the law accounts them.
    friction_field_J_per_m, stored_field_J_per_m : numpy.ndarray or None
        The same per station [J/m], shape (stations, samples).
    equilibrium_residual : float
        ``max |M_law - F m(x)| / max |M|`` over stations and samples.
    along_wire : AlongWireOutput or None
        With the along-wire option: all stations (overhangs included), the
        armour moment per layer, wire energies and solver statistics.
    restrictor_moment_field_Nm : numpy.ndarray or None
        With a restrictor: the moment carried by the locked restrictor,
        ``F m(x) - M_cable`` where the cable is on its lock curvature, zero
        elsewhere, shape (stations, samples).
    """

    arrangement: str
    control: str
    curvature_definition: str
    values: FloatArray
    retained: NDArray[np.bool_]
    load_N: FloatArray
    displacement_m: FloatArray
    moment_Nm: FloatArray
    curvature_1_m: FloatArray
    station_curvature_1_m: FloatArray
    station_moment_Nm: FloatArray
    x_m: FloatArray
    moment_per_load_m: FloatArray
    measuring_x_m: float
    curvature_field_1_m: FloatArray
    moment_field_Nm: FloatArray
    gauge_x_m: FloatArray
    gauge_deflection_m: FloatArray
    external_work_J: FloatArray
    friction_work_J: FloatArray | None
    stored_energy_J: FloatArray | None
    friction_field_J_per_m: FloatArray | None
    stored_field_J_per_m: FloatArray | None
    equilibrium_residual: float
    law_evaluations: int
    provenance: Provenance
    notes: tuple[str, ...] = field(default=())
    along_wire: AlongWireOutput | None = None
    restrictor_moment_field_Nm: FloatArray | None = None

    def loop_work_J_per_m(self) -> float:
        """Area of the reported moment--curvature loop over the retained window
        [J/m] (closed by the segment back to its first sample)."""
        k = self.curvature_1_m[self.retained]
        m = self.moment_Nm[self.retained]
        return float(abs(_closed_integral(m, k)))

    def cycle_external_work_J(self) -> float:
        """``oint F d(delta)`` over the retained window [J] (closed path)."""
        f = self.load_N[self.retained]
        d = self.displacement_m[self.retained]
        return float(_closed_integral(f, d))

    def cycle_friction_work_J(self) -> float | None:
        """Frictional work over the retained window [J], if accounted."""
        if self.friction_work_J is None:
            return None
        w = self.friction_work_J[self.retained]
        return float(w[-1] - w[0])

    def deflection_m(self, x_m: float) -> FloatArray:
        """Deflection history at ``x_m`` [m] (exact at stations and gauges)."""
        beam_kernel = self._kernel
        g = beam_kernel(self.x_m, float(x_m))
        return np.asarray(_product_weights(self.x_m, g) @ self.curvature_field_1_m)

    @property
    def _kernel(self) -> Callable[[FloatArray, float], FloatArray]:
        span = float(self.x_m[-1])
        if self.arrangement.startswith("cantilever"):
            return lambda x, xi: np.maximum(xi - x, 0.0)
        return lambda x, xi: np.where(
            x <= xi, x * (span - xi) / span, xi * (span - x) / span
        )


def _closed_integral(y: FloatArray, x: FloatArray) -> float:
    yc = np.r_[y, y[:1]]
    xc = np.r_[x, x[:1]]
    return float(np.sum(0.5 * (yc[1:] + yc[:-1]) * np.diff(xc)))


def _product_weights(x: FloatArray, g: FloatArray) -> FloatArray:
    """Weights ``c`` with ``int kappa g dx = c . kappa`` for both linear per segment."""
    h = np.diff(x)
    w = np.zeros_like(x)
    w[:-1] += h / 6.0 * (2.0 * g[:-1] + g[1:])
    w[1:] += h / 6.0 * (g[:-1] + 2.0 * g[1:])
    return w


def _trapezoid_weights(x: FloatArray) -> FloatArray:
    h = np.diff(x)
    w = np.zeros_like(x)
    w[:-1] += 0.5 * h
    w[1:] += 0.5 * h
    return w


# --------------------------------------------------------------------------- #
# Monotone inverse                                                             #
# --------------------------------------------------------------------------- #
class _MonotoneInverse:
    """Solve ``u(v) = T`` for a non-decreasing ``u`` with ``u(0) = 0``, v >= 0.

    ``fn`` evaluates ``u`` at a sorted array of new points (one call per
    batch). All evaluations are cached; each target is bracketed by cached
    points and refined by inverse quadratic interpolation inside the bracket,
    with regula falsi and periodic bisection as safeguards. Points beyond the
    cache are reached by extrapolation with the secant of the last two points
    and with the smallest slope ``slope_min`` (which over-reaches when it is a
    true lower bound of the slope).
    """

    def __init__(
        self,
        fn: Callable[[FloatArray], FloatArray],
        *,
        slope_min: float,
        slope_max: float,
        tol: float,
        convex: bool = False,
    ) -> None:
        self.convex = convex
        self.fn = fn
        self.v = np.zeros(1)
        self.u = np.zeros(1)
        self.slope_min = slope_min
        self.slope_max = slope_max
        self.tol = tol
        self.calls = 0

    def _add(self, points: FloatArray) -> None:
        pts = np.unique(points[np.isfinite(points) & (points > 0.0)])
        pts = pts[~np.isin(pts, self.v)]
        if pts.size == 0:
            return
        values = np.asarray(self.fn(pts), dtype=np.float64)
        self.calls += 1
        self.v = np.r_[self.v, pts]
        self.u = np.r_[self.u, values]

    def solve(self, targets: FloatArray) -> tuple[FloatArray, FloatArray]:
        T = np.asarray(targets, dtype=np.float64)
        n = T.size
        sol_v = np.full(n, np.nan)
        sol_u = np.full(n, np.nan)
        if n == 0:
            return sol_v, sol_u
        for it in range(_MAX_ITERATIONS):
            order = np.argsort(self.v, kind="stable")
            V = self.v[order]
            U = np.maximum.accumulate(self.u[order])
            Uraw = self.u[order]
            idx = np.searchsorted(U, T, side="left")
            m = V.size
            hi = np.minimum(idx, m - 1)
            lo = np.maximum(idx - 1, 0)
            r_hi = np.abs(Uraw[hi] - T)
            r_lo = np.abs(Uraw[lo] - T)
            hit_hi = (idx < m) & (r_hi <= self.tol)
            hit_lo = (idx > 0) & (r_lo <= self.tol)
            width = V[hi] - V[lo]
            tiny = (
                (idx < m)
                & (idx > 0)
                & (width <= 4e-16 * np.maximum(np.abs(V[hi]), 1e-300))
            )
            done = hit_hi | hit_lo | tiny | (T <= 0.0)
            pick_hi = hit_hi | (tiny & (r_hi <= r_lo))
            sol_v = np.where(pick_hi, V[hi], np.where(T <= 0.0, 0.0, V[lo]))
            sol_u = np.where(pick_hi, Uraw[hi], np.where(T <= 0.0, 0.0, Uraw[lo]))
            if done.all():
                return sol_v, sol_u
            open_ = ~done
            beyond = open_ & (idx >= m)
            inside = open_ & (idx < m) & (idx > 0)
            cands: list[FloatArray] = []
            if beyond.any():
                dT = T[beyond] - U[-1]
                secant = m >= 2 and U[-1] > U[-2]
                if secant:
                    sec = (U[-1] - U[-2]) / (V[-1] - V[-2])
                    cands.append(V[-1] + dT / sec)
                # concave u: the secant under-reaches, the smallest slope
                # over-reaches; convex u: the secant over-reaches
                if not (self.convex and secant):
                    cands.append(V[-1] + dT / self.slope_min)
                    cands.append(V[-1] + dT / self.slope_max)
            if inside.any():
                i_lo, i_hi = lo[inside], hi[inside]
                vl, vh, ul, uh = V[i_lo], V[i_hi], U[i_lo], U[i_hi]
                t = T[inside]
                denom = np.where(uh > ul, uh - ul, 1.0)
                rf = vl + (t - ul) * (vh - vl) / denom
                # inverse quadratic through a third neighbour
                i3 = np.where(i_lo > 0, i_lo - 1, np.minimum(i_hi + 1, m - 1))
                v3, u3 = V[i3], U[i3]
                with np.errstate(divide="ignore", invalid="ignore"):
                    iq = (
                        vl * (t - uh) * (t - u3) / ((ul - uh) * (ul - u3))
                        + vh * (t - ul) * (t - u3) / ((uh - ul) * (uh - u3))
                        + v3 * (t - ul) * (t - uh) / ((u3 - ul) * (u3 - uh))
                    )
                margin = 1e-3 * (vh - vl)
                ok = np.isfinite(iq) & (iq > vl + margin) & (iq < vh - margin)
                cands.append(np.where(ok, iq, rf))
                cands.append(rf)
                if it % 3 == 2:
                    cands.append(0.5 * (vl + vh))
            before = self.v.size
            self._add(np.concatenate(cands))
            if self.v.size == before and inside.any():
                # The interpolants rounded onto cached points (a target inside
                # a step of u a few ulps of v wide, as the distributed friction
                # model gives): bisect. A bracket with no float inside it is
                # closed above (tiny).
                self._add(0.5 * (V[lo[inside]] + V[hi[inside]]))
            if self.v.size == before:
                # No new point to try: the open targets are not reached within
                # the tolerance, so the closest cached point would be a wrong
                # answer returned as a converged one.
                worst = float(np.max(np.minimum(r_hi, r_lo)[open_]))
                raise CableSolverError(
                    "bending-test inversion stalled: no new curvature point to try, "
                    f"{int(np.count_nonzero(open_))} target(s) missed by up to "
                    f"{worst:.3g} (tolerance {self.tol:.3g})"
                )
        raise CableSolverError("bending-test inversion did not converge")


# --------------------------------------------------------------------------- #
# Station groups                                                               #
# --------------------------------------------------------------------------- #
class _Group:
    """Stations sharing one influence value: one curvature history."""

    def __init__(
        self, law: SectionLaw, mbar: float, tol_m: float, kappa_lock: float = math.inf
    ) -> None:
        self.law = law
        self.mbar = mbar
        self.tol_m = tol_m
        self.kappa_lock = kappa_lock
        self.history: list[float] = []
        self.kappa_a = 0.0
        self.moment_a = 0.0
        self.calls = 0
        self._leg: _MonotoneInverse | None = None
        self._direction = 0.0

    def start_leg(self) -> None:
        self._leg = None
        self._direction = 0.0

    def _inverse(self, direction: float) -> _MonotoneInverse:
        if self._leg is None or direction != self._direction:
            history = np.asarray(self.history, dtype=np.float64)
            k_a, m_a = self.kappa_a, self.moment_a

            def fn(points: FloatArray) -> FloatArray:
                path = np.r_[history, k_a + direction * points]
                m = self.law.moment(path)[history.size :]
                return np.asarray(direction * (m - m_a))

            self._leg = _MonotoneInverse(
                fn,
                slope_min=self.law.ei_slip_Nm2,
                slope_max=self.law.ei_stick_Nm2,
                tol=self.tol_m,
            )
            self._direction = direction
        return self._leg

    def solve_moments(self, targets: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Curvature and achieved moment for moment targets along the leg."""
        if math.isfinite(self.kappa_lock) and abs(self.kappa_a) >= self.kappa_lock * (
            1.0 - 1e-12
        ):
            # Locked at the start of the leg: while the applied moment stays
            # beyond the cable's locked moment the restrictor carries it.
            side = math.copysign(1.0, self.kappa_a)
            targets = np.where(
                side * (targets - self.moment_a) > 0.0, self.moment_a, targets
            )
        dm = targets - self.moment_a
        nz = dm[np.abs(dm) > 0]
        if nz.size == 0:
            return np.full(targets.shape, self.kappa_a), np.full(
                targets.shape, self.moment_a
            )
        direction = float(np.sign(nz[0]))
        inv = self._inverse(direction)
        before = inv.calls
        v, u = inv.solve(np.maximum(direction * dm, 0.0))
        self.calls += inv.calls - before
        kappa = self.kappa_a + direction * v
        moment = self.moment_a + direction * u
        if math.isfinite(self.kappa_lock):
            # Restrictor (gap P5): the cable curvature is capped at the lock
            # curvature; the restrictor carries the rest of the moment.
            over = np.abs(kappa) > self.kappa_lock
            if np.any(over):
                k_cap = math.copysign(self.kappa_lock, direction)
                path = np.r_[np.asarray(self.history, dtype=np.float64), k_cap]
                self.calls += 1
                m_cap = float(self.law.moment(path)[-1])
                kappa = np.where(over, k_cap, kappa)
                moment = np.where(over, m_cap, moment)
        return kappa, moment

    def evaluate_curvatures(self, kappa: FloatArray) -> FloatArray:
        """Law moment for prescribed (monotone) curvatures along the leg."""
        path = np.r_[np.asarray(self.history, dtype=np.float64), kappa]
        self.calls += 1
        return np.asarray(self.law.moment(path)[len(self.history) :])

    def commit(self, kappa_end: float, moment_end: float) -> None:
        if kappa_end != self.kappa_a or not self.history:
            self.history.append(kappa_end)
        self.kappa_a = kappa_end
        self.moment_a = moment_end
        self._leg = None


def _legs(values: FloatArray) -> list[tuple[int, int]]:
    """Index ranges ``[start, stop)`` of monotone runs of ``values`` (from 0)."""
    steps = np.sign(np.diff(np.r_[0.0, values]))
    legs: list[tuple[int, int]] = []
    start, current = 0, 0.0
    for k, s in enumerate(steps):
        if s == 0 or s == current:
            continue
        if current == 0:
            current = s
            continue
        legs.append((start, k))
        start, current = k, s
    legs.append((start, values.size))
    return legs


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #
def _test_setup(
    arrangement: object,
    protocol: object,
    curvature_definition: str | None,
    gauge_x_m: Sequence[float],
    stations_per_segment: int,
    rtol: float,
) -> tuple[_Beam, str, FloatArray]:
    """Checked beam, curvature definition and gauge positions of a test."""
    if not isinstance(arrangement, (ThreePoint, FourPoint, Cantilever)):
        raise CableInputError("arrangement must be ThreePoint, FourPoint or Cantilever")
    if not isinstance(protocol, BendingProtocol):
        raise CableInputError("protocol must be a BendingProtocol")
    if isinstance(stations_per_segment, bool) or int(stations_per_segment) < 2:
        raise CableInputError("stations_per_segment must be at least 2")
    require_finite(rtol, "rtol")
    if not 0.0 < rtol < 1e-3:
        raise CableInputError("rtol must lie in (0, 1e-3)")
    beam = arrangement._beam()
    definition = curvature_definition or (
        "elastic" if isinstance(arrangement, ThreePoint) else "station"
    )
    if definition not in _DEFINITIONS:
        raise CableInputError(f"curvature_definition must be one of {_DEFINITIONS}")
    gauges = np.asarray([float(g) for g in gauge_x_m], dtype=np.float64)
    if gauges.size and (
        not np.isfinite(gauges).all() or gauges.min() < 0 or gauges.max() > beam.length
    ):
        raise CableInputError("gauge positions must lie on the specimen")
    if definition == "fit" and np.unique(gauges).size < 3:
        raise CableInputError("the fit definition needs at least three gauges")
    return beam, definition, gauges


def simulate_bending_test(
    section: SectionLaw | CableData,
    arrangement: Arrangement,
    protocol: BendingProtocol,
    *,
    curvature_definition: CurvatureDefinition | None = None,
    gauge_x_m: Sequence[float] = (),
    stations_per_segment: int = 24,
    rtol: float = 1e-11,
    along_wire: AlongWireSpec | None = None,
    restrictor: ArticulatedRestrictor | None = None,
    restrictor_root_x_m: float = 0.0,
    cable_od_m: float | None = None,
    **law_options: object,
) -> BendingTestResult:
    r"""Simulate a statically determinate bending test of a hysteretic section.

    Parameters
    ----------
    section : SectionLaw, CableData or AlongWireSection
        The section law, or a cable, for which ``law_options`` (``tension_N``,
        ``mu``, ``contact_model``, ``section_state``, ``radial_state``,
        ``core_friction``, ``friction_model``, ``progressive``) build
        :meth:`SectionLaw.from_cable`. With ``along_wire`` a cable (options
        ``tension_N``, ``mu``, ``contact_model``, ``section_state``,
        ``radial_state``) or an :class:`AlongWireSection`.
    arrangement : ThreePoint, FourPoint or Cantilever
        Test set-up.
    protocol : BendingProtocol
        Controlled quantity, sample by sample.
    curvature_definition : {"station", "elastic", "fit"}, optional
        How the test defines its curvature: the true curvature at the
        measuring station (default for four-point and cantilever), the
        elastic deflection formula of the set-up (default for three-point,
        ``12 w / L^2``), or ``"fit"``, a least-squares parabola through the
        deflections at ``gauge_x_m`` (at least three; ``kappa = -w''``).
    gauge_x_m : sequence of float
        Deflection-gauge positions (stations are added there).
    stations_per_segment : int
        Stations between consecutive key points (supports, loads, mid-span).
    rtol : float
        Moment and control tolerance relative to their largest magnitude
        (with ``along_wire`` the default is ``1e-8``).
    along_wire : AlongWireSpec, optional
        Solve the helical layers along their wires through the specimen
        (finite length, overhangs, anchored or free wire ends) instead of
        independent sections (:mod:`cablexsec.analysis.along_wire_test`).
    restrictor : ArticulatedRestrictor, optional
        Articulated bend restrictor over ``[restrictor_root_x_m,
        restrictor_root_x_m + n p]`` (gap P5): the cable curvature there is
        capped at the cable lock curvature
        (:meth:`~cablexsec.mechanics.bend_restrictor.ArticulatedRestrictor.cable_locking_radius_m`,
        no tension), and the restrictor carries the rest of ``F m(x)``
        (``restrictor_moment_field_Nm``). Load or moment control only;
        not with ``along_wire``.
    restrictor_root_x_m : float
        Position of the restrictor root on the specimen [m].
    cable_od_m : float, optional
        Cable outer diameter for the bore clearance (default: the cable's
        ``aggregate_properties.OD_mm``).

    Returns
    -------
    BendingTestResult

    Notes
    -----
    Statically determinate set-ups only (``M = F m(x)``); see the module
    docstring for the beam assumptions. With ``"curvature"`` control and the
    station definition the measuring station follows the prescribed history
    exactly, so its loop is bitwise the section law over that history.
    """
    lock_curvature = math.inf
    if restrictor is not None:
        if along_wire is not None:
            raise CableInputError("a restrictor is not available with along_wire")
        if protocol.control not in ("load", "moment"):
            raise CableInputError("a restrictor needs load or moment control")
        if cable_od_m is None:
            od_mm = (
                section.aggregate_properties.OD_mm
                if isinstance(section, CableData)
                else None
            )
            if od_mm is None:
                raise CableInputError("give cable_od_m for the restrictor clearance")
            cable_od_m = float(od_mm) / 1e3
        require_finite(restrictor_root_x_m, "restrictor_root_x_m")
        lock_curvature = 1.0 / restrictor.cable_locking_radius_m(cable_od_m)
    if along_wire is not None:
        return _simulate_along_wire(
            section,
            arrangement,
            protocol,
            along_wire,
            curvature_definition=curvature_definition,
            gauge_x_m=gauge_x_m,
            stations_per_segment=stations_per_segment,
            rtol=rtol if rtol != 1e-11 else 1e-8,
            law_options=law_options,
        )
    if isinstance(section, CableData):
        law = SectionLaw.from_cable(section, **law_options)  # type: ignore[arg-type]
    elif isinstance(section, SectionLaw):
        if law_options:
            raise CableInputError("law options need a cable, not a SectionLaw")
        law = section
    else:
        raise CableInputError("section must be a SectionLaw or a CableData")
    beam, definition, gauges = _test_setup(
        arrangement,
        protocol,
        curvature_definition,
        gauge_x_m,
        stations_per_segment,
        rtol,
    )

    # ------------------------------------------------------------- stations
    n_seg = int(stations_per_segment)
    keys = sorted(set(beam.key_points))
    x = np.unique(
        np.concatenate(
            [
                np.linspace(a, b, n_seg + 1)
                for a, b in zip(keys[:-1], keys[1:], strict=True)
            ]
            + [gauges]
        )
    )
    mbar = beam.influence(x)
    i_meas = int(np.argmin(np.abs(x - beam.measuring_x)))
    mbar_meas = float(mbar[i_meas])
    disp_w = _product_weights(x, beam.influence(x))
    gauge_w = np.array(
        [_product_weights(x, beam.kernel(x, g)) for g in gauges]
    ).reshape(gauges.size, x.size)
    if definition == "station":
        curv_w = np.zeros(x.size)
        curv_w[i_meas] = 1.0
    elif definition == "elastic":
        curv_w = beam.elastic_factor * _product_weights(
            x, beam.kernel(x, beam.elastic_x)
        )
    else:
        vand = np.column_stack((gauges**2, gauges, np.ones_like(gauges)))
        curv_w = -2.0 * (np.linalg.pinv(vand)[0] @ gauge_w)

    # groups of equal influence (symmetry, uniform-moment zone)
    scale = float(np.max(np.abs(mbar)))
    key = np.round(mbar / scale, 12)
    station_lock = np.full(x.size, math.inf)
    if restrictor is not None:
        in_restrictor = (x >= restrictor_root_x_m - 1e-12) & (
            x <= restrictor_root_x_m + restrictor.length_m + 1e-12
        )
        station_lock[in_restrictor] = lock_curvature
        # stations inside and outside the restrictor never share a history
        key = key + np.where(in_restrictor, 10.0, 0.0)
    uniq, inverse = np.unique(key, return_inverse=True)
    group_lock = np.array(
        [float(station_lock[inverse == j][0]) for j in range(uniq.size)]
    )
    group_mbar = np.array([float(mbar[inverse == j].mean()) for j in range(uniq.size)])
    for j in range(uniq.size):  # identical values within a group
        mbar[inverse == j] = group_mbar[j]
    active = np.abs(group_mbar) > 1e-14 * scale
    g_meas = int(inverse[i_meas])

    values = protocol.values
    n = values.size
    control = protocol.control
    # moment scale for tolerances
    if control == "load":
        f_scale = float(np.max(np.abs(values)))
    elif control == "moment":
        f_scale = float(np.max(np.abs(values))) / abs(mbar_meas)
    else:
        f_scale = math.nan  # set after the first leg from the solution
    tol_m_guess = rtol * law.ei_stick_Nm2 * 1e-3  # refined below

    groups = [
        _Group(law, float(group_mbar[j]), tol_m_guess, float(group_lock[j]))
        if active[j]
        else None
        for j in range(uniq.size)
    ]
    w_group = np.array([float(curv_w[inverse == j].sum()) for j in range(uniq.size)])
    d_group = np.array([float(disp_w[inverse == j].sum()) for j in range(uniq.size)])

    load = np.zeros(n)
    kappa_g = np.zeros((uniq.size, n))
    station_direct = control == "curvature" and definition == "station"
    if station_direct and not active[g_meas]:
        raise CableInputError("the measuring station carries no moment")

    def set_tolerance(f_max: float) -> None:
        for j, grp in enumerate(groups):
            if grp is not None:
                grp.tol_m = rtol * max(abs(group_mbar[j]) * f_max, 1e-300)

    if control in ("load", "moment"):
        set_tolerance(f_scale)
    elif not station_direct:
        # tolerance from an elastic estimate of the load scale
        lin = float(
            np.sum(
                (w_group if control == "curvature" else d_group)[active]
                * group_mbar[active]
            )
            / law.ei_slip_Nm2
        )
        if lin <= 0:
            raise CableInputError(
                "the controlled quantity does not increase with the load "
                "for this arrangement and curvature definition"
            )
        # inner tolerance well below the control tolerance
        set_tolerance(1e-2 * float(np.max(np.abs(values))) / lin)

    weights_c = w_group if control == "curvature" else d_group
    for start, stop in _legs(values):
        idx = np.arange(start, stop)
        for grp in groups:
            if grp is not None:
                grp.start_leg()
        if control in ("load", "moment"):
            f_leg = values[idx] if control == "load" else values[idx] / mbar_meas
            achieved: dict[int, FloatArray] = {}
            for j, grp in enumerate(groups):
                if grp is None:
                    continue
                k_sol, m_sol = grp.solve_moments(grp.mbar * f_leg)
                kappa_g[j, idx] = k_sol
                achieved[j] = m_sol
            load[idx] = f_leg
        elif station_direct:
            gm = groups[g_meas]
            assert gm is not None
            k_meas = values[idx]
            m_meas = gm.evaluate_curvatures(k_meas)
            f_leg = m_meas / gm.mbar
            achieved = {g_meas: m_meas}
            kappa_g[g_meas, idx] = k_meas
            if math.isnan(f_scale):
                f_scale = float(np.max(np.abs(f_leg))) or 1.0
                set_tolerance(f_scale)
            for j, grp in enumerate(groups):
                if grp is None or j == g_meas:
                    continue
                k_sol, m_sol = grp.solve_moments(grp.mbar * f_leg)
                kappa_g[j, idx] = k_sol
                achieved[j] = m_sol
            load[idx] = f_leg
        else:
            f_leg, achieved, k_leg = _solve_controlled(
                groups,
                weights_c,
                values[idx],
                load[start - 1] if start > 0 else 0.0,
                (kappa_g[:, start - 1] if start > 0 else np.zeros(uniq.size)),
                rtol,
            )
            kappa_g[:, idx] = k_leg
            load[idx] = f_leg
        for j, grp in enumerate(groups):
            if grp is not None:
                grp.commit(float(kappa_g[j, stop - 1]), float(achieved[j][-1]))

    # ------------------------------------------------------------- final run
    m_g = np.zeros((uniq.size, n))
    fr_g: FloatArray | None = np.zeros((uniq.size, n))
    st_g: FloatArray | None = np.zeros((uniq.size, n))
    calls = sum(grp.calls for grp in groups if grp is not None)
    for j, grp in enumerate(groups):
        if grp is None:
            continue
        resp = law.evaluate(kappa_g[j])
        calls += 1
        m_g[j] = resp.moment_Nm
        if resp.friction_J_per_m is None or fr_g is None:
            fr_g = None
        else:
            fr_g[j] = resp.friction_J_per_m
        if resp.stored_J_per_m is None or st_g is None:
            st_g = None
        else:
            st_g[j] = resp.stored_J_per_m
    if station_direct:
        load = m_g[g_meas] / group_mbar[g_meas]
    kappa_field = kappa_g[inverse]
    moment_field = m_g[inverse]
    m_ref = float(np.max(np.abs(moment_field))) or 1.0
    applied = mbar[:, None] * load[None, :]
    restrictor_moment: FloatArray | None = None
    if restrictor is not None:
        locked = np.abs(kappa_field) >= station_lock[:, None] * (1.0 - 1e-12)
        restrictor_moment = np.where(locked, applied - moment_field, 0.0)
        residual = float(
            np.max(np.where(locked, 0.0, np.abs(moment_field - applied))) / m_ref
        )
    else:
        residual = float(np.max(np.abs(moment_field - applied)) / m_ref)
    displacement = disp_w @ kappa_field
    curvature = curv_w @ kappa_field
    gauge_defl = gauge_w @ kappa_field
    increments = np.diff(np.r_[0.0, displacement])
    f_mid = 0.5 * (load + np.r_[0.0, load[:-1]])
    external = np.cumsum(f_mid * increments)
    q = _trapezoid_weights(x)
    friction = None if fr_g is None else q @ fr_g[inverse]
    stored = None if st_g is None else q @ st_g[inverse]
    retained = np.zeros(n, dtype=bool)
    retained[int(protocol.retain_from) :] = True
    notes = []
    if residual > 1e3 * rtol:
        notes.append(
            f"equilibrium residual {residual:.2e} exceeds 1000 rtol: the law's "
            "response along a monotone leg depends on its sampling"
        )
    if restrictor is not None and restrictor_moment is not None:
        notes.append(
            f"articulated restrictor over x = {restrictor_root_x_m:.4g}-"
            f"{restrictor_root_x_m + restrictor.length_m:.4g} m: cable curvature "
            f"capped at {lock_curvature:.6g} 1/m; peak restrictor moment "
            f"{float(np.max(np.abs(restrictor_moment))):.6g} N m; residual over "
            "unlocked stations"
        )
    return BendingTestResult(
        arrangement=beam.name,
        control=control,
        curvature_definition=definition,
        values=values,
        retained=retained,
        load_N=load,
        displacement_m=displacement,
        moment_Nm=load * mbar_meas,
        curvature_1_m=curvature,
        station_curvature_1_m=kappa_field[i_meas],
        station_moment_Nm=moment_field[i_meas],
        x_m=x,
        moment_per_load_m=mbar,
        measuring_x_m=float(x[i_meas]),
        curvature_field_1_m=kappa_field,
        moment_field_Nm=moment_field,
        gauge_x_m=gauges,
        gauge_deflection_m=gauge_defl,
        external_work_J=external,
        friction_work_J=friction,
        stored_energy_J=stored,
        friction_field_J_per_m=None if fr_g is None else fr_g[inverse],
        stored_field_J_per_m=None if st_g is None else st_g[inverse],
        equilibrium_residual=residual,
        law_evaluations=calls,
        restrictor_moment_field_Nm=restrictor_moment,
        provenance=Provenance(
            method=(
                f"statically determinate {beam.name}: M = F m(x) at {x.size} "
                f"stations, per-station history-dependent inversion of the "
                f"section law ({law.description}); test reduction {beam.formula}; "
                f"curvature definition '{definition}'"
            ),
            citations=("[HuEtAl2022] Eqs. 1-2", "[MenardCartraud2023] Eq. 1"),
            gates=(
                "elastic closed forms (three-point, four-point, cantilever)",
                "uniform-moment zone reproduces the section loop bitwise",
                "external work equals stored energy plus friction work",
            ),
            limitations=(
                "Euler-Bernoulli small deflection on frictionless supports; no "
                "shear deformation, P-delta or self-weight",
                "stations are independent sections: no axial redistribution of "
                "wire slip between stations of different curvature",
                "planar bending in one plane; statically determinate set-ups",
            ),
        ),
        notes=tuple(notes),
    )


def _solve_controlled(
    groups: list[_Group | None],
    weights: FloatArray,
    targets: FloatArray,
    f_start: float,
    kappa_start: FloatArray,
    rtol: float,
) -> tuple[FloatArray, dict[int, FloatArray], FloatArray]:
    """Loads of one leg for a controlled linear functional ``c = sum w_g kappa_g``."""
    c_start = float(weights @ kappa_start)
    dc = targets - c_start
    nz = dc[np.abs(dc) > 0]
    n_g = len(groups)
    if nz.size == 0:
        k = np.repeat(kappa_start[:, None], targets.size, axis=1)
        moments = {
            j: np.full(targets.size, grp.moment_a)
            for j, grp in enumerate(groups)
            if grp is not None
        }
        return np.full(targets.size, f_start), moments, k
    direction = float(np.sign(nz[0]))
    slope_lo = slope_hi = 0.0
    for j, grp in enumerate(groups):
        if grp is None:
            continue
        a = weights[j] * grp.mbar
        slope_lo += a / grp.law.ei_stick_Nm2
        slope_hi += a / grp.law.ei_slip_Nm2
    if slope_lo <= 0:
        raise CableInputError(
            "the controlled quantity does not increase with the load "
            "for this arrangement and curvature definition"
        )
    cache: dict[float, tuple[FloatArray, dict[int, FloatArray]]] = {}

    def fn(points: FloatArray) -> FloatArray:
        loads = f_start + direction * points
        k = np.repeat(kappa_start[:, None], points.size, axis=1).astype(np.float64)
        moments: dict[int, FloatArray] = {}
        for j, grp in enumerate(groups):
            if grp is None:
                continue
            k[j], moments[j] = grp.solve_moments(grp.mbar * loads)
        c = weights @ k
        for i, p in enumerate(points):
            cache[float(p)] = (
                k[:, i].copy(),
                {j: m[i : i + 1] for j, m in moments.items()},
            )
        return np.asarray(direction * (c - c_start))

    c_tol = rtol * max(float(np.max(np.abs(targets))), abs(c_start), 1e-300)
    inv = _MonotoneInverse(
        fn, slope_min=slope_lo, slope_max=slope_hi, tol=c_tol, convex=True
    )
    v, _ = inv.solve(np.maximum(direction * dc, 0.0))
    k_out = np.zeros((n_g, targets.size))
    m_out: dict[int, FloatArray] = {
        j: np.zeros(targets.size) for j, grp in enumerate(groups) if grp is not None
    }
    for i, vi in enumerate(v):
        if vi == 0.0 or float(vi) not in cache:
            k_out[:, i] = kappa_start
            for j, grp in enumerate(groups):
                if grp is not None:
                    m_out[j][i] = grp.moment_a
            continue
        k_i, m_i = cache[float(vi)]
        k_out[:, i] = k_i
        for j, mm in m_i.items():
            m_out[j][i] = mm[0]
    return f_start + direction * v, m_out, k_out


# --------------------------------------------------------------------------- #
# Along-wire coupling                                                          #
# --------------------------------------------------------------------------- #
_ALONG_WIRE_OPTIONS = (
    "tension_N",
    "mu",
    "contact_model",
    "section_state",
    "radial_state",
)


def _simulate_along_wire(
    section: object,
    arrangement: object,
    protocol: object,
    spec: AlongWireSpec,
    *,
    curvature_definition: CurvatureDefinition | None,
    gauge_x_m: Sequence[float],
    stations_per_segment: int,
    rtol: float,
    law_options: dict[str, object],
) -> BendingTestResult:
    """:func:`simulate_bending_test` with the helical layers along their wires."""
    if not isinstance(spec, AlongWireSpec):
        raise CableInputError("along_wire must be an AlongWireSpec")
    if isinstance(section, CableData):
        unknown = sorted(set(law_options) - set(_ALONG_WIRE_OPTIONS))
        if unknown:
            raise CableInputError(
                "the along-wire model replaces the sectional armour law; options "
                f"{unknown} do not apply (accepted: {list(_ALONG_WIRE_OPTIONS)})"
            )
        aw_section = AlongWireSection.from_cable(
            section,
            stick=spec.stick,
            capstan=spec.capstan,
            sense=spec.sense,
            **law_options,  # type: ignore[arg-type]
        )
    elif isinstance(section, AlongWireSection):
        if law_options:
            raise CableInputError("law options need a cable, not an AlongWireSection")
        aw_section = section
    else:
        raise CableInputError(
            "with along_wire the section must be a CableData or an AlongWireSection"
        )
    beam, definition, gauges = _test_setup(
        arrangement,
        protocol,
        curvature_definition,
        gauge_x_m,
        stations_per_segment,
        rtol,
    )
    assert isinstance(protocol, BendingProtocol)  # checked by _test_setup
    out = run_along_wire_test(
        aw_section,
        beam,
        protocol,
        spec,
        definition=definition,
        gauges=gauges,
        stations_per_segment=int(stations_per_segment),
        rtol=float(rtol),
    )
    x_all: FloatArray = out["x_all"]  # type: ignore[assignment]
    span: slice = out["span"]  # type: ignore[assignment]
    kappa: FloatArray = out["kappa"]  # type: ignore[assignment]
    moment: FloatArray = out["moment"]  # type: ignore[assignment]
    load: FloatArray = out["load"]  # type: ignore[assignment]
    mbar: FloatArray = out["mbar"]  # type: ignore[assignment]
    energy: FloatArray = out["energy"]  # type: ignore[assignment]
    i_meas = int(out["i_meas"])  # type: ignore[call-overload]
    mbar_meas = float(out["mbar_meas"])  # type: ignore[arg-type]
    disp_w: FloatArray = out["disp_w"]  # type: ignore[assignment]
    curv_w: FloatArray = out["curv_w"]  # type: ignore[assignment]
    gauge_w: FloatArray = out["gauge_w"]  # type: ignore[assignment]
    m_ref = float(np.max(np.abs(moment))) or 1.0
    residual = float(np.max(np.abs(moment - mbar[:, None] * load[None, :])) / m_ref)
    displacement = disp_w @ kappa
    increments = np.diff(np.r_[0.0, displacement])
    f_mid = 0.5 * (load + np.r_[0.0, load[:-1]])
    external = np.cumsum(f_mid * increments)
    retained = np.zeros(load.size, dtype=bool)
    retained[int(protocol.retain_from) :] = True
    notes = []
    if residual > 1e3 * rtol:
        notes.append(f"equilibrium residual {residual:.2e} exceeds 1000 rtol")
    aw_out = AlongWireOutput(
        x_m=x_all,
        span=span,
        curvature_1_m=kappa,
        moment_Nm=moment,
        armour_moment_Nm=out["armour"],  # type: ignore[arg-type]
        energy_J=energy,
        iterations=out["iterations"],  # type: ignore[arg-type]
        residual=out["residual"],  # type: ignore[arg-type]
        segments=out["segments"],  # type: ignore[arg-type]
        spec=spec,
    )
    fr: FloatArray = out["friction_field"]  # type: ignore[assignment]
    st: FloatArray = out["stored_field"]  # type: ignore[assignment]
    return BendingTestResult(
        arrangement=beam.name,
        control=protocol.control,
        curvature_definition=definition,
        values=protocol.values,
        retained=retained,
        load_N=load,
        displacement_m=displacement,
        moment_Nm=load * mbar_meas,
        curvature_1_m=curv_w @ kappa,
        station_curvature_1_m=kappa[i_meas],
        station_moment_Nm=moment[i_meas],
        x_m=x_all[span],
        moment_per_load_m=mbar[span],
        measuring_x_m=float(x_all[i_meas]),
        curvature_field_1_m=kappa[span],
        moment_field_Nm=moment[span],
        gauge_x_m=gauges,
        gauge_deflection_m=gauge_w @ kappa,
        external_work_J=external,
        friction_work_J=energy[:, 1].copy(),
        stored_energy_J=energy[:, 2].copy(),
        friction_field_J_per_m=fr[span],
        stored_field_J_per_m=st[span],
        equilibrium_residual=residual,
        law_evaluations=int(out["evaluations"]),  # type: ignore[call-overload]
        provenance=Provenance(
            method=(
                f"statically determinate {beam.name}: M = F m(x) at {x_all.size} "
                f"stations (overhangs {spec.overhang_m[0]:g}/"
                f"{spec.overhang_m[1]:g} m), "
                f"helical layers solved along their wires ({aw_section.description}; "
                f"wire ends {spec.ends[0]}/{spec.ends[1]}, "
                f"{spec.segments_per_pitch} segments per pitch); test reduction "
                f"{beam.formula}; curvature definition '{definition}'"
            ),
            citations=("[HuEtAl2022] Eqs. 1-2", "[MenardCartraud2023] Eq. 1"),
            gates=(
                "long specimen reproduces the distributed sectional law",
                "external work equals stored energy plus friction work",
                "station and wire-segment refinement",
            ),
            limitations=(
                "Euler-Bernoulli small deflection on frictionless supports; no "
                "shear deformation, P-delta or self-weight",
                "axial wire slip only; contact pressure held at the section state; "
                "no laid-core friction",
                "planar bending in one plane; statically determinate set-ups",
            ),
        ),
        notes=tuple(notes),
        along_wire=aw_out,
    )
