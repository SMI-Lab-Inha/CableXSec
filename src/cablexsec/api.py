# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""CableXSec section, component and wire-response calculations.

The public functions are re-exported from the top-level package. Results
identify the selected mechanics and their source references. The solved
normal section state can be reused in bending, stress and fatigue recovery.
Model choices and supported geometries are explicit; a result from the
fixed-radius approximation is not a coupled radial-equilibrium prediction.

Entry points
------------
============================ ===================== ==========================
function                     returns               accepts a cable directly?
============================ ===================== ==========================
:func:`load_cable`            ``CableData``         (loader)
:func:`compute_stiffness`     ``StiffnessResult``   yes
:func:`compute_mkappa`        ``MkappaResult``      yes (cable→spec adapter)
:func:`compute_stress`        ``StressResult``      yes (per-wire, all layers)
:func:`compute_thermal`       ``ThermalResult``     yes
:func:`compute_capacity`      ``CapacityResult``    yes
:func:`compression_screen`    ``CompressionScreenResult`` yes (armour in compression)
:func:`compute_compression_capacity` ``CompressionCapacityResult`` yes
:func:`compute_fatigue`       ``FatigueResult``     yes (cable→spec adapter)
:func:`compute_electrical_life` ``ElectricalLifeResult`` yes
:func:`compute_shell_stress`  ``ShellStressResult`` yes (metallic sheath)
:func:`compute_conductor_fatigue` ``ConductorFatigueResult`` yes (stranded conductor)
:func:`torque_balance`        ``TorqueBalanceResult`` yes (fixed and free twist)
:func:`design_torque_balance` ``TorqueBalanceDesign`` yes (armour layer parameter)
============================ ===================== ==========================

:func:`compute_stress` returns the stress decomposition for every structural
helical layer. :func:`compute_component_response` reports every declared cable
component, including offset-core bounds and metallic sheaths.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

# Evidence-tracked engines (frozen core + non-frozen assemblers).
from cablexsec.analysis.capacity import (
    BirdcageDetail,
    CapacityResult,
    birdcage_buckling_detail,
    compute_capacity,
    wire_yield_curvature,
)
from cablexsec.analysis.components import (
    ComponentResponse,
    ComponentState,
    compute_component_response,
)
from cablexsec.analysis.compression import (
    ARMOUR_UTILISATION_FACTORS,
    CompressionCapacityResult,
    CompressionScreenResult,
    TapeLayer,
    compression_screen,
    compute_compression_capacity,
)
from cablexsec.analysis.conductor_fatigue import (
    DEFAULT_CONTACT_YIELD_PA,
    FULLY_PLASTIC_PRESSURE_FACTOR,
    ConductorFatigueResult,
    CriticalDistanceSpec,
    assess_conductor_fatigue,
)
from cablexsec.analysis.electrical_life import LifeResult, compute_electrical_life
from cablexsec.analysis.fatigue import (
    CopperContactModel,
    DistributedBound,
    FatigueLayer,
    FatigueResult,
    FrictionModel,
    RainflowCounter,
    _warn_default_direction_tied,
    curvature_path_out_of_plane_ratio,
    rainflow_cycles,
)
from cablexsec.analysis.fatigue import (
    compute_fatigue as _compute_fatigue_specs,
)
from cablexsec.analysis.fretting_assessment import CableFrettingResult, FrettingFace
from cablexsec.analysis.fretting_swt import PartialSlipCycles
from cablexsec.analysis.sheath_strain import SheathStrainLifeResult
from cablexsec.analysis.sn_assessment import (
    ComponentSCF,
    corrected_damage,
    needs_correction,
    scaled_component,
)
from cablexsec.analysis.stress import StressResult, compute_stress
from cablexsec.analysis.termination import TerminationSpec
from cablexsec.analysis.thermal_coupling import (
    ThermalHistory,
    ThermalState,
    thermal_fatigue_inputs,
    thermal_provenance_notes,
)
from cablexsec.analysis.torque_balance import (
    TorqueBalanceDesign,
    TorqueBalanceResult,
    design_torque_balance,
    torque_balance,
)

# Cable loading / schema.
from cablexsec.cable_adapters import (
    armour_contact_pressure,
    armour_crossing_line_loads,
    cable_to_fatigue_layers,
    cable_to_mkappa_specs,
    require_fully_specified,
    solved_contact_histories,
    structural_armour_layers,
)
from cablexsec.cables import (
    cigre_tb880_yaml_path,
    export_three_core_yaml_path,
    iea_task49_66kv_300_yaml_path,
    iea_task49_66kv_stranded,
    iea_task49_66kv_yaml_path,
    inter_array_single_core_yaml_path,
    inter_array_three_core_yaml_path,
    load_cable,
    menard2023_yaml_path,
    saevik2011_yaml_path,
    witz1996_yaml_path,
)
from cablexsec.cables.schema import (
    CableData,
    CylindricalLayer,
    CylindricalShellLayer,
    HelicalLayer,
    scope_entry_points,
)
from cablexsec.exceptions import (
    BlockedSNCurveError,
    CableInputError,
    CableLookupError,
    CableSchemaError,
    CableSolverError,
    CableXSecError,
    PhysicalRangeError,
)
from cablexsec.materials.mean_stress import MeanStressMethod, MeanStressSpec
from cablexsec.materials.sn_curves import SNcurve, get_sn_curve
from cablexsec.materials.strain_life import (
    MeanStressCorrection,
    StrainLifeConstants,
    get_strain_life,
)
from cablexsec.materials.wear import FrettingKnockdown, WearCoefficient
from cablexsec.mechanics.concentric_section import (
    ConcentricSectionResult,
    ElasticAnnulus,
    ElasticHelicalRing,
    solve_concentric_section,
)
from cablexsec.mechanics.contact_qp import ContactQPResult, solve_contact_qp
from cablexsec.mechanics.core_friction import (
    CoreFrictionHistory,
    CoreFrictionModel,
    CoreFrictionSpec,
    build_core_friction,
    core_friction_history,
    resolve_core_friction,
)
from cablexsec.mechanics.core_kinematics import CoreSlipState
from cablexsec.mechanics.distributed_friction import DEFAULT_MAX_INCREMENT
from cablexsec.mechanics.friction_law import (
    FrictionLaw,
    active_friction_law,
    friction_law_time_steps,
)
from cablexsec.mechanics.mindlin_contact import TransverseMap
from cablexsec.mechanics.mkappa import HelicalLayerSpec
from cablexsec.mechanics.papailiou import (
    BendingWork,
    PapailiouLayer,
    PapailiouWireResult,
    _station_return,
    compute_mkappa_wire_slip_front,
)
from cablexsec.mechanics.progressive_friction import ProgressiveFriction
from cablexsec.mechanics.radial_cable import (
    CableRadialState,
    solve_cable_radial_state,
)
from cablexsec.mechanics.section_state import SectionState, solve_section_state
from cablexsec.mechanics.shell_bending import (
    ShellStressResult,
    shell_wall_stress,
)
from cablexsec.mechanics.stiffness import (
    axial_torsional_matrix,
    bending_stiffness_by_layer,
    ea_fixed,
    ea_free,
    ei_max_stuck,
    ei_min_slip,
    xi_for_layer,
)
from cablexsec.mechanics.strand_slip import ConductorRing, PressureTransmission
from cablexsec.mechanics.thermal import ThermalResult, compute_thermal
from cablexsec.provenance import (
    ExtrapolationWarning,
    Provenance,
    UnevaluatedFatigueCurveWarning,
    UnvalidatedModelWarning,
    ValidatedRange,
    check_range,
    real_array,
    require_finite,
)

# Public alias: the project-standard name for the water-tree life result.
ElectricalLifeResult = LifeResult

FloatArray = NDArray[np.float64]
#: A mean-stress correction: a full specification or a method name whose
#: strength is read from the layer's library material.
MeanStressOption = MeanStressSpec | MeanStressMethod

__all__ = [
    # loading
    "load_cable",
    "cigre_tb880_yaml_path",
    "menard2023_yaml_path",
    "inter_array_single_core_yaml_path",
    "inter_array_three_core_yaml_path",
    "export_three_core_yaml_path",
    "iea_task49_66kv_300_yaml_path",
    "iea_task49_66kv_stranded",
    "iea_task49_66kv_yaml_path",
    "saevik2011_yaml_path",
    "witz1996_yaml_path",
    "CableData",
    # compute entry points
    "compute_stiffness",
    "compute_mkappa",
    "compute_thermal",
    "compute_capacity",
    "compression_screen",
    "compute_compression_capacity",
    "ARMOUR_UTILISATION_FACTORS",
    "TapeLayer",
    "compute_fatigue",
    "compute_electrical_life",
    "compute_stress",
    "compute_component_response",
    "compute_shell_stress",
    "compute_shell_fatigue",
    "compute_conductor_fatigue",
    "compute_fretting",
    "armour_contact_histories",
    "CableFrettingResult",
    "torque_balance",
    "design_torque_balance",
    "TorqueBalanceResult",
    "TorqueBalanceDesign",
    "solve_contact_qp",
    "solve_concentric_section",
    "solve_section_state",
    "SectionState",
    "solve_cable_radial_state",
    "CableRadialState",
    "ConcentricSectionResult",
    "ElasticAnnulus",
    "ElasticHelicalRing",
    "ContactQPResult",
    "CoreFrictionSpec",
    "CriticalDistanceSpec",
    "FrictionLaw",
    "ProgressiveFriction",
    "CoreFrictionHistory",
    # cable -> frozen-spec adapters
    "cable_to_mkappa_specs",
    "cable_to_fatigue_layers",
    # result objects
    "StiffnessResult",
    "MkappaResult",
    "StressResult",
    "ComponentResponse",
    "ComponentState",
    "ThermalResult",
    "CapacityResult",
    "CompressionScreenResult",
    "CompressionCapacityResult",
    "BirdcageDetail",
    "birdcage_buckling_detail",
    "FatigueResult",
    "FatigueLayer",
    "TerminationSpec",
    "ElectricalLifeResult",
    "ShellStressResult",
    "ShellFatigueResult",
    "ConductorFatigueResult",
    # provenance + exceptions
    "Provenance",
    "ValidatedRange",
    "ExtrapolationWarning",
    "UnevaluatedFatigueCurveWarning",
    "UnvalidatedModelWarning",
    "CableXSecError",
    "CableInputError",
    "CableLookupError",
    "CableSolverError",
    "CableSchemaError",
    "BlockedSNCurveError",
    "PhysicalRangeError",
]


# --------------------------------------------------------------------------- #
# Stiffness                                                                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StiffnessResult:
    """Cross-section stiffness bounds for a cable.

    Attributes
    ----------
    cable_name : str
        Name of the analysed cable.
    ea_fixed_N : float
        Axial stiffness with the ends constrained against twist ``K_ee`` [N].
    ea_free_N : float
        Free-end (Schur-reduced) axial stiffness ``K_ee − K_et²/K_tt`` [N].
    ei_min_Nm2 : float
        Full-slip bending stiffness (individual-wire bending) [N·m²].
    ei_max_Nm2 : float
        No-slip bending stiffness with the selected shear compliance [N·m²].
        This is not the component-axis plane-section limit when xi < 1.
    ei_plane_section_Nm2 : float
        Plane-section reference without armour shear reduction, including
        parallel axes for internal wire rings where their geometry is supplied.
    ei_armour_no_slip_Nm2 : float
        Nominal no-slip stiffness of the current armour bending law, with
        individual core bending but without constrained-core parallel axes.
    gj_Nm2 : float
        Extension-induced lay contribution ``K_tt`` [N·m²]. Independent
        component torsion and radial relaxation are not included here.
    axial_torsion_coupling_Nm : float
        Axial-twist coupling ``K_et`` [N·m].
    xi_per_layer : dict[str, float]
        Yun shear coefficient ξ for each helical layer.
    provenance : Provenance or None
        Backing citations/gates.
    """

    cable_name: str
    ea_fixed_N: float
    ea_free_N: float
    ei_min_Nm2: float
    ei_max_Nm2: float
    gj_Nm2: float
    axial_torsion_coupling_Nm: float
    xi_per_layer: dict[str, float]
    provenance: Provenance | None = None
    ei_plane_section_Nm2: float = math.nan
    ei_armour_no_slip_Nm2: float = math.nan

    @property
    def ei_shear_compliant_Nm2(self) -> float:
        """Explicit name for the historical ``ei_max_Nm2`` output."""
        return self.ei_max_Nm2


def compute_stiffness(cable: CableData) -> StiffnessResult:
    r"""Cross-section axial/torsional/bending stiffness bounds for a cable.

    Assembles the axial-twist 2x2 reduction, individual bending, and
    parallel-axis contributions of the declared components. The no-slip
    stiffness with finite polymer shear compliance and the component-axis
    plane-section limit are returned separately. A comparison must use the
    same kinematic definition and material moduli on both sides. An effective
    core modulus does not resolve contact and slip among its internal strands.

    Parameters
    ----------
    cable : CableData
        A fully-specified cable.

    Returns
    -------
    StiffnessResult

    Raises
    ------
    CableSchemaError
        If the cable geometry is not fully specified.

    References
    ----------
    [WitzTan1992a] (axial Schur); [WitzTan1992b] (full-slip EI); [Yun2020] (ξ).

    Examples
    --------
    >>> from cablexsec import load_cable, compute_stiffness, menard2023_yaml_path
    >>> s = compute_stiffness(load_cable(menard2023_yaml_path(),
    ...                                  geometry_override=None))
    >>> s.ei_min_Nm2 > 0.0
    True
    """
    require_fully_specified(cable, "compute_stiffness")
    at = axial_torsional_matrix(cable)
    xi_map = {lyr.name: xi_for_layer(cable, lyr.name) for lyr in cable.helical_layers}
    bending = bending_stiffness_by_layer(cable, xi_map)
    return StiffnessResult(
        cable_name=cable.cable.name,
        ea_fixed_N=ea_fixed(cable),
        ea_free_N=ea_free(cable),
        ei_min_Nm2=ei_min_slip(cable),
        ei_max_Nm2=ei_max_stuck(cable, xi_map),
        gj_Nm2=at.K_tt,
        axial_torsion_coupling_Nm=at.K_et,
        xi_per_layer=xi_map,
        ei_plane_section_Nm2=ei_max_stuck(cable),
        ei_armour_no_slip_Nm2=sum(
            layer.individual_Nm2 + layer.armour_shear_compliant_Nm2
            for layer in bending.values()
        ),
        provenance=Provenance(
            method="Fixed-radius axial Schur; individual, plane-section and "
            "shear-compliant bending",
            citations=("[WitzTan1992a]", "[WitzTan1992b]", "[Yun2020]"),
            gates=(
                "A7 EA-Schur",
                "A8/A8c EI_min",
                "A8b EI_max",
                "separate component-axis plane-section and shear-compliant "
                "stiffness assemblies; comparisons in VALIDATION.md",
            ),
            limitations=(
                "These are reference component assemblies, not a solved "
                "pressure-dependent section tangent or a history-dependent "
                "operating bending stiffness.",
            ),
        ),
    )


# --------------------------------------------------------------------------- #
# M-kappa (cable -> frozen HelicalLayerSpec adapter)                           #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MkappaResult:
    """Moment-curvature hysteresis loop for a cable.

    Attributes
    ----------
    kappa_1_m, moment_Nm : numpy.ndarray
        Curvature [1/m] and bending moment [N·m] along the driven history.
    W_diss_J_per_m : float
        Frictional work per cable-axis metre accumulated over the complete
        supplied history [J/m], including initialisation if supplied. Not
        mechanical work on an open path and not automatically a per-cycle loss.
        Under a rate-dependent friction law it includes the rate-dependent
        (viscous) part, ``work.viscous_J_per_m``.
    EI_apparent_Nm2 : float
        Secant bending stiffness over the loop [N·m²].
    EI_max_Nm2, EI_min_Nm2 : float
        Armour-law no-slip slope and full-slip backbone [N·m²]. The former
        does not constrain relative axial motion of the offset power cores.
    slip_fraction : numpy.ndarray
        Fraction of wires slipping at each step.
    tension_N : float
        Axial tension the contact forces were evaluated at [N]; with a
        contact-tension history, its first sample (the reference state).
    provenance : Provenance or None
        Backing citations/gates and active limitations.
    work : BendingWork or None
        Cumulative stored energy, frictional work and incremental work balance.
        Returned by compute_mkappa; optional only for compatibility with manually
        constructed result objects. For this scalar bending law, mechanical
        work between two stations is the stored-energy change plus the
        frictional-work increment, including within-step yield transitions.
        This holds with a contact-pressure history too: the pressure moves
        only the friction bound of the return map, and a bound that falls
        below a wire's stress releases stored energy as friction work.
    core_friction : CoreFrictionHistory or None
        Core and internal-ring contribution when ``core_friction`` was
        requested (moments in the material frame, contact loads, ``xi``,
        effective friction); ``None`` for the armour-only law. When present,
        ``moment_Nm``, ``work``, ``W_diss_J_per_m`` and ``EI_max_Nm2`` include
        it; ``slip_fraction`` remains the armour-wire fraction.
    contact_pressure_Pa : dict[str, numpy.ndarray] or None
        Per-sample friction-setting pressure ``P1+P2`` [Pa] of every
        structural armour layer when the pressure followed a history
        (``contact_pressure_Pa`` or ``contact_tension_history_N`` of
        :func:`compute_mkappa`); ``None`` when it was held fixed.
    """

    kappa_1_m: FloatArray
    moment_Nm: FloatArray
    W_diss_J_per_m: float
    EI_apparent_Nm2: float
    EI_max_Nm2: float
    EI_min_Nm2: float
    slip_fraction: FloatArray
    tension_N: float
    provenance: Provenance | None = None
    work: BendingWork | None = None
    core_friction: CoreFrictionHistory | None = None
    contact_pressure_Pa: dict[str, FloatArray] | None = None


@dataclass(frozen=True)
class _MkappaRun:
    """One evaluation of a prepared :class:`_MkappaLaw` over a history."""

    armour: PapailiouWireResult
    moment_Nm: FloatArray
    work: BendingWork
    EI_max_Nm2: float
    W_diss_J_per_m: float
    core_history: CoreFrictionHistory | None


@dataclass(frozen=True)
class _MkappaLaw:
    """The moment--curvature law of :func:`compute_mkappa`, prepared once.

    Holds the layer specifications, the full-slip stiffness and the optional
    core-friction model, so that the same law can be evaluated over many
    curvature histories (beam-test stations) with the arithmetic of
    :func:`compute_mkappa`.
    """

    layers: list[HelicalLayerSpec]
    ei_min: float
    phi_bend: float
    progressive: ProgressiveFriction | None
    core_model: CoreFrictionModel | None

    def run(
        self,
        kappa: FloatArray,
        contact_pressure_Pa: Sequence[FloatArray] | None = None,
        friction_law: FrictionLaw | None = None,
        time_s: FloatArray | None = None,
    ) -> _MkappaRun:
        """Evaluate the law; ``contact_pressure_Pa`` is one armour ``P1+P2``
        history per layer (order of ``layers``), ``None`` for fixed pressure.
        The laid-core model keeps its own fixed contact loads. A non-neutral
        ``friction_law`` (armour wires, direction-tied law) is refused with
        the progressive law and the laid-core model."""
        phi_bend = self.phi_bend
        law = active_friction_law(friction_law)
        if law is not None and (
            self.progressive is not None or self.core_model is not None
        ):
            raise CableSchemaError(
                "a friction law is implemented for the direction-tied armour law "
                "only; not with the progressive law or core friction"
            )
        if self.progressive is None:
            res = compute_mkappa_wire_slip_front(
                self.layers,
                kappa,
                phi_bend=phi_bend,
                EI_min_Nm2=self.ei_min,
                contact_pressure_Pa=contact_pressure_Pa,
                friction_law=law,
                time_s=time_s,
            )
        else:
            res = compute_mkappa_wire_slip_front(
                self.layers,
                kappa,
                phi_bend=phi_bend,
                EI_min_Nm2=self.ei_min,
                progressive=self.progressive,
                contact_pressure_Pa=contact_pressure_Pa,
            )
        model = self.core_model
        if model is None:
            return _MkappaRun(
                res, res.moment_Nm, res.work, res.EI_max_Nm2, res.W_diss_J_per_m, None
            )
        direction = np.array([math.cos(phi_bend), math.sin(phi_bend)])
        core_history = core_friction_history(
            model,
            kappa * direction[0],
            kappa * direction[1],
            initial_bending_angle=phi_bend,
        )
        moment = res.moment_Nm + core_history.moment_Nm @ direction
        aw, cw = res.work, core_history.work
        work = BendingWork(
            aw.stored_J_per_m + cw.stored_J_per_m,
            aw.friction_J_per_m + cw.friction_J_per_m,
            aw.endpoint_work_J_per_m + cw.endpoint_work_J_per_m,
            aw.integration_remainder_J_per_m + cw.integration_remainder_J_per_m,
            aw.balance_residual_J_per_m + cw.balance_residual_J_per_m,
        )
        return _MkappaRun(
            res,
            moment,
            work,
            res.EI_max_Nm2 + model.stick_EI_Nm2,
            float(work.friction_J_per_m[-1]),
            core_history,
        )


def _prepare_mkappa_law(
    cable: CableData,
    tension_N: float,
    *,
    mu: float | None,
    phi_bend: float,
    contact_model: str,
    section_state: SectionState | None,
    radial_state: CableRadialState | None,
    core_spec: CoreFrictionSpec | None,
    progressive: ProgressiveFriction | None,
) -> _MkappaLaw:
    """Layer specifications, full-slip stiffness and core model of the law."""
    if core_spec is not None and contact_model == "solved" and radial_state is None:
        # Solve once; the armour and the cores read the same radial state.
        radial_state = solve_cable_radial_state(cable, tension_N)
    layers = cable_to_mkappa_specs(
        cable,
        tension_N,
        mu=mu,
        contact_model=contact_model,
        section_state=section_state,
        radial_state=radial_state,
    )
    ei_min = ei_min_slip(
        cable,
        modulus_scale=section_state.modulus_scale
        if section_state is not None
        else None,
    )
    model = (
        None
        if core_spec is None
        else build_core_friction(
            cable,
            core_spec,
            tension_N,
            contact_model=contact_model,
            radial_state=radial_state,
            section_state=section_state,
        )
    )
    return _MkappaLaw(layers, ei_min, phi_bend, progressive, model)


def compute_mkappa(
    cable: CableData,
    kappa_history: FloatArray,
    *,
    tension_N: float = 0.0,
    mu: float | None = None,
    phi_bend: float = 0.0,
    contact_model: str | None = None,
    section_state: SectionState | None = None,
    radial_state: CableRadialState | None = None,
    core_friction: bool | CoreFrictionSpec | None = None,
    friction_model: Literal["direction_tied", "progressive"] = "direction_tied",
    progressive: ProgressiveFriction | None = None,
    contact_pressure_Pa: Mapping[str, FloatArray] | None = None,
    contact_tension_history_N: FloatArray | None = None,
    friction_law: FrictionLaw | None = None,
    time_s: FloatArray | None = None,
) -> MkappaResult:
    r"""Moment-curvature hysteresis of a cable over a curvature history.

    Uses the physical-wire Papailiou slip-front law: slip starts at the
    neutral axis and propagates towards the extreme fibres. The per-layer
    friction-setting contact pressure ``P1+P2`` combines the cable's cited
    manufacturing state with the applied-tension increment from the selected
    contact solver. The physical wire return maps retain ``phi_bend`` and use
    uniform mean pressure on each armour population. An inner bedding-support
    reaction is not substituted for armour pressure. ``EI_min`` is the
    individual-component full-slip stiffness.

    By default the pressure is that of one contact state, held over the
    history. ``contact_tension_history_N`` lets it follow a tension history
    (the contact histories of :func:`compute_fatigue` and
    :func:`armour_contact_histories`), and ``contact_pressure_Pa`` takes the
    per-sample pressure of each armour layer directly. The pressure then
    sets the friction bound of every wire at every sample, in the same
    return map as the fatigue path: the slip state is carried across the
    change, a falling bound clips a wire's friction stress (friction work),
    and a rising bound leaves a sticking wire unchanged. What follows the
    history is the armour friction bound only. The stick stiffness,
    ``EI_min``, ``EI_max``, the laid-core and internal-ring contact loads
    (``core_friction``) and the wire-yield range check stay at the reference
    state (``tension_N``, ``section_state`` or ``radial_state``; the first
    sample of a tension history).

    Parameters
    ----------
    cable : CableData
        A fully-specified cable.
    kappa_history : numpy.ndarray
        Driven curvature time series [1/m].
    tension_N : float, optional
        Axial tension for the contact ``F_n`` [N].
    mu : float, optional
        Armour friction coefficient (default: material ``mu_on_hdpe``).
    phi_bend : float, optional
        Bending-plane azimuth [rad].
    contact_model : {"bounded", "qp", "solved"}, optional
        Contact model for per-wire normal forces. ``"bounded"`` is the
        default chain, used by the published bending comparisons; ``"qp"``
        uses the KKT radial-compliance solver for controlled cross-model
        studies; ``"solved"`` uses the
        coupled radial contact solution (concentric and three-core,
        manufacturing residual wire stress included; doc/radial_model.md).
    section_state : SectionState, optional
        Previously solved normal equilibrium. Its layer moduli and shear
        support are held fixed over the supplied bending history, and so are
        its contact pressures unless ``contact_pressure_Pa`` is given. Do not
        also supply tension or a conflicting contact model.
    radial_state : CableRadialState, optional
        A solved radial state from :func:`solve_cable_radial_state`, which
        accepts internal/external or flooded pressure, an end-cap thrust,
        torque or twist, prescribed axial strain and layer temperatures. Its
        face pressures and line loads are held fixed over the bending history
        unless ``contact_pressure_Pa`` is given. Do not also supply tension, a
        section state or another contact model.
    core_friction : bool or CoreFrictionSpec, optional
        Add the stick-slip law of the laid power cores against their
        neighbours, the filler and the surrounding ring, and of resolved
        internal wire rings (:mod:`cablexsec.mechanics.core_friction`, gap
        R1b). ``True`` uses the default :class:`CoreFrictionSpec`. Core
        contact loads come from the same contact model (solved trefoil forces
        or the bounded estimate). ``None`` (default) keeps the armour-only law
        bitwise unchanged. Requires one family of laid cores.
    friction_model : {"direction_tied", "progressive"}, optional
        Armour friction law. ``"direction_tied"`` (default) is one
        spring-slider per wire with the Papailiou cap. ``"progressive"``
        splits every wire into parallel sub-elements whose caps average to
        the same cap (:mod:`cablexsec.mechanics.progressive_friction`, gap
        G8): by default the parameter-free friction-continuity density, which
        spreads the stick-to-slip transition of each wire; stick stiffness
        and full-slip moment are unchanged.
    progressive : ProgressiveFriction, optional
        Settings of the progressive law (density, sub-elements, spread);
        ``None`` uses :class:`ProgressiveFriction` defaults. Only with
        ``friction_model="progressive"``.
    contact_pressure_Pa : dict[str, numpy.ndarray], optional
        Per-sample friction-setting pressure ``P1+P2`` [Pa] of every
        structural armour layer, keyed by layer name (all of them), each the
        shape of ``kappa_history``, finite and non-negative; for example the
        histories of :func:`armour_contact_histories` or of a sequence of
        solved section states. The reference state (``tension_N``,
        ``section_state`` or ``radial_state``) supplies everything else. A
        history equal to the fixed pressure at every sample reproduces the
        fixed-pressure result bit for bit.
    contact_tension_history_N : numpy.ndarray, optional
        Tension history [N] that sets the armour contact pressure at every
        sample through the selected contact model, as ``tension_N`` does in
        :func:`compute_fatigue`: the bounded chain is affine in tension, the
        qp model is solved at every distinct tension, and the solved model is
        exact up to 33 distinct tensions and interpolated between 33 exact
        solutions beyond. Non-negative, the shape of ``kappa_history``. The
        reference state is the first sample; the wire-yield range check uses
        the largest tension. Not with ``tension_N``, ``section_state``,
        ``radial_state`` or ``contact_pressure_Pa``.
    friction_law : FrictionLaw, optional
        Static/dynamic and rate-dependent armour friction coefficient
        (:class:`~cablexsec.mechanics.friction_law.FrictionLaw`) of every
        armour wire: a static factor until a wire slips, a dynamic factor
        while it slips and after it until it has stuck for the healing time,
        and a logarithmic and viscous rate term, and a factor on the
        coefficient as a function of the local contact pressure
        (``pressure_factor``, evaluated at each sample's ``P1+P2``). Only
        with the direction-tied law and without ``core_friction``. ``work``
        then carries the rate-dependent dissipation in ``viscous_J_per_m``.
        ``None`` (default) or a neutral law is the rate-independent law, bit
        for bit.
    time_s : numpy.ndarray, optional
        Sample times [s] of ``kappa_history``, required by a law that depends
        on time (a slip rate or a finite positive healing time); refused
        otherwise.

    Returns
    -------
    MkappaResult

    Notes
    -----
    Energy under a pressure history: the pressure enters only the friction
    bound of each spring-slider, not its stiffness, so no material-work term
    arises (``wire_slip_history`` keeps that term for a changing stiffness).
    A bound that drops from ``c0`` to ``c1`` below a slipping wire's stress
    at fixed curvature releases the stored energy ``A cos a (c0^2 - c1^2) /
    (2 k)`` as friction work (``k`` the stick modulus); the reported friction
    increment of one such step is ``A cos a c1 (c0 - c1) / k`` and the rest
    is the integration remainder of ``work``, which vanishes as the change is
    resolved in smaller steps. A rising bound does no work.

    References
    ----------
    [Kim2020] (friction stress); [Yun2020] (ξ); [MenardCartraud2023] (Eq. 14);
    [WitzTan1992b] (EI bounds); [KarlsenHusoy2026] (internal rings).

    Examples
    --------
    >>> import numpy as np
    >>> from cablexsec import load_cable, compute_mkappa, menard2023_yaml_path
    >>> cab = load_cable(menard2023_yaml_path(), geometry_override=None)
    >>> r = compute_mkappa(cab, 0.05 * np.sin(np.linspace(0, 6 * np.pi, 600)))
    >>> r.EI_min_Nm2 <= r.EI_max_Nm2
    True
    """
    require_fully_specified(cable, "compute_mkappa")
    require_finite(tension_N, "tension_N")
    require_finite(phi_bend, "phi_bend")
    if tension_N < 0.0:
        raise CableSchemaError("tension_N must be >= 0 for the contact model.")
    if mu is not None:
        require_finite(mu, "mu")
        if mu < 0.0:
            raise CableSchemaError("mu must be >= 0.")
    kappa = real_array(kappa_history, "kappa_history")
    if kappa.ndim != 1 or kappa.size < 2:
        raise CableSchemaError(
            "kappa_history must be a one-dimensional history with at least 2 points."
        )
    require_finite(kappa, "kappa_history")
    if friction_model not in ("direction_tied", "progressive"):
        raise CableSchemaError(
            "friction_model must be 'direction_tied' or 'progressive' "
            "for compute_mkappa."
        )
    if progressive is not None and friction_model != "progressive":
        raise CableSchemaError(
            "progressive settings require friction_model='progressive'."
        )
    progressive_settings = (
        (progressive or ProgressiveFriction())
        if friction_model == "progressive"
        else None
    )
    if friction_law is not None and not isinstance(friction_law, FrictionLaw):
        raise CableSchemaError("friction_law must be a FrictionLaw or None.")
    law = active_friction_law(friction_law)
    if law is not None and (
        friction_model != "direction_tied" or core_friction not in (None, False)
    ):
        raise CableSchemaError(
            "a friction law is implemented for the direction-tied armour law "
            "only; not with the progressive law or core friction"
        )
    try:
        friction_law_time_steps(law, time_s, kappa.size)
    except CableInputError as error:
        raise CableSchemaError(str(error)) from error
    tension_history = None
    if contact_tension_history_N is not None:
        if (
            tension_N != 0
            or section_state is not None
            or radial_state is not None
            or contact_pressure_Pa is not None
        ):
            raise CableSchemaError(
                "contact_tension_history_N sets the contact state: give it "
                "without tension_N, a section or radial state or "
                "contact_pressure_Pa"
            )
        tension_history = real_array(
            contact_tension_history_N, "contact_tension_history_N"
        ).astype(np.float64)
        require_finite(tension_history, "contact_tension_history_N")
        if tension_history.shape != kappa.shape:
            raise CableSchemaError(
                "contact_tension_history_N must match the kappa_history shape"
            )
        if np.any(tension_history < 0.0):
            raise CableSchemaError("contact_tension_history_N must be >= 0.")
        tension_N = float(tension_history[0])
    if radial_state is not None:
        radial_state.require_cable(cable)
        if (
            section_state is not None
            or tension_N != 0
            or contact_model not in {None, "solved"}
        ):
            raise CableSchemaError(
                "supply a radial state alone, without tension, section state "
                "or another contact model"
            )
        tension_N = radial_state.tension_N
        contact_model = "solved"
    elif section_state is not None:
        section_state.require_cable(cable)
        if tension_N != 0 or (
            contact_model is not None and contact_model != section_state.contact_model
        ):
            raise CableSchemaError(
                "supply a section state or a contact-model tension, not both"
            )
        tension_N = section_state.tension_N
        contact_model = section_state.contact_model
    else:
        contact_model = contact_model or "bounded"
    if contact_model not in {"bounded", "qp", "solved"}:
        raise CableSchemaError("contact_model must be 'bounded', 'qp' or 'solved'.")
    # Armour layers in the order of the law's specifications.
    names = [layer.name for layer in structural_armour_layers(cable)]
    pressure_history: dict[str, FloatArray] | None = None
    if contact_pressure_Pa is not None:
        pressure_history = _mkappa_pressure_history(
            contact_pressure_Pa, names, kappa.shape
        )
    mkappa_law = _prepare_mkappa_law(
        cable,
        tension_N,
        mu=mu,
        phi_bend=phi_bend,
        contact_model=contact_model,
        section_state=section_state,
        radial_state=radial_state,
        core_spec=resolve_core_friction(core_friction),
        progressive=progressive_settings,
    )
    if tension_history is not None:
        # pressure only: the crossing loads (unused here) refuse some layer stacks
        histories, _ = _contact_histories(
            cable, tension_history, contact_model, crossing_loads=False
        )
        pressure_history = {name: histories[name] for name in names}

    # Soft guard: the linear-elastic wire-stress assumption holds below the
    # armour yield curvature (where the capacity envelope's wire-yield mechanism
    # is reached). Beyond it the elastic M-κ is unconservative.
    try:
        kappa_yield = wire_yield_curvature(
            cable,
            float(np.max(tension_history))
            if tension_history is not None
            else tension_N
            if section_state is None
            else 0.0,
            mu=mu,
            section_state=section_state,
            contact_model=contact_model,
            phi_bend=phi_bend,
        )
        if math.isfinite(kappa_yield) and kappa_yield > 0.0:
            check_range(
                float(np.max(np.abs(kappa))),
                ValidatedRange(
                    "peak |curvature| [1/m]",
                    0.0,
                    kappa_yield,
                    "armour wire-yield curvature (capacity wire-yield mechanism)",
                ),
            )
    except (ValueError, KeyError):
        pass  # no yield-strength armour: no curvature bound to check

    run = mkappa_law.run(
        kappa,
        None
        if pressure_history is None
        else [pressure_history[name] for name in names],
        friction_law=law,
        time_s=time_s,
    )
    res, moment, work, ei_max = run.armour, run.moment_Nm, run.work, run.EI_max_Nm2
    w_diss, core_history = run.W_diss_J_per_m, run.core_history
    law_extra = (
        ""
        if progressive_settings is None
        else "; progressive (Iwan) armour friction law, "
        + progressive_settings.describe()
    )
    if law is not None:
        law_extra += "; armour friction law: " + law.describe()
    history_extra = (
        ""
        if pressure_history is None
        else "; the armour contact pressure follows the "
        + (
            f"tension history through the {contact_model} contact model"
            if tension_history is not None
            else "supplied per-sample history"
        )
    )
    history_limitations: tuple[str, ...] = (
        ()
        if pressure_history is None
        else (
            "only the armour friction bound follows the contact-pressure "
            "history; the stick stiffness, EI_min, any laid-core and ring "
            "contact loads and the wire-yield check use the reference state",
        )
    )
    model = mkappa_law.core_model
    if model is None:
        # Armour-only law: the historical path, bitwise unchanged.
        method_extra = ""
        citations_extra: tuple[str, ...] = ()
        limitations_extra: tuple[str, ...] = ()
    else:
        method_extra = (
            "; laid-core stick-slip (direction-tied Papailiou cap, "
            f"{model.loads.source} core contact loads, stick coefficient "
            f"{model.xi:.3f}, effective friction coefficient "
            f"{model.mu_effective:.3f})"
            + (
                f" and {len(model.strand_rings)} internal wire rings "
                "(Karlsen and Husøy operator)"
                if model.strand_rings
                else ""
            )
            + (
                f" and {model.fillers.n_profiles} laid filler profiles (stick "
                f"coefficient {model.fillers.xi:.3f}, friction coefficient "
                f"{model.fillers.mu:.3f})"
                if model.fillers is not None
                else ""
            )
        )
        citations_extra = ("[Yun2020] (core xi)", "[KarlsenHusoy2026]")
        limitations_extra = (
            "core contact loads are the axisymmetric state held fixed in "
            "bending; the layers of a core move together; core-core friction "
            "acts on the slip of each core relative to the plane-section field",
            "internal rings take the core contact line load by line-force "
            "conservation (upper estimate) and the cable curvature",
        )
    # Secant apparent stiffness from the curvature extremes on the loop.
    k_hi, k_lo = int(np.argmax(kappa)), int(np.argmin(kappa))
    dk = kappa[k_hi] - kappa[k_lo]
    ei_app = float((moment[k_hi] - moment[k_lo]) / dk) if dk != 0 else float(ei_max)
    return MkappaResult(
        kappa_1_m=res.kappa,
        moment_Nm=moment,
        W_diss_J_per_m=w_diss,
        EI_apparent_Nm2=ei_app,
        EI_max_Nm2=ei_max,
        EI_min_Nm2=res.EI_min_Nm2,
        slip_fraction=res.slip_fraction,
        tension_N=tension_N,
        work=work,
        core_friction=core_history,
        contact_pressure_Pa=pressure_history,
        provenance=Provenance(
            method="Papailiou slip-front moment-curvature law of the physical "
            "wires (slip front from the neutral axis); the armour contact "
            "pressure combines the manufacturing contact with the applied-tension "
            f"increment of the {contact_model} contact model"
            + history_extra
            + law_extra
            + method_extra,
            citations=(
                "[Papailiou1997]",
                "[Kim2020] Eq.3.24 (sigma0)",
                "[FeretBournazel1987]",
                "[WitzTan1992b] (EI_min)",
                *citations_extra,
            ),
            gates=(
                "Papailiou helical normal-force and pressure-conversion checks",
                "incremental stored-energy and friction-work balance",
            ),
            limitations=(
                "Papailiou neutral-axis slip-front caps drive the same physical "
                "wire return maps as the wire stress and fatigue analyses",
                "mean armour face pressure is uniform over each wire population; "
                "an inner bedding-support reaction is not used as an outer "
                "armour-pressure distribution",
                "applied tension changes the armour contact pressure through a "
                "tributary-area mapping; "
                "the full-scale comparison has an assumed, not measured, "
                "manufacturing contact state",
                "the dissipated work is zero if the cable carries no winding "
                "stress or pretension (no friction-setting contact)",
                "linear-elastic wire stress (no wire yield); a curvature beyond "
                "first wire yield is flagged",
                "the coupled radial contact model is verified against its "
                "optimality conditions but not against measured wire populations; "
                "the full-scale comparison does not establish its accuracy",
                "the solved contact model is verified against closed forms and "
                "published section results (the radial model notes); the "
                "three-core state depends on the declared filler representation",
                *limitations_extra,
                *history_limitations,
                *(
                    ()
                    if law is None
                    else (
                        "friction law: no rate-dependent or static/dynamic "
                        "friction data of armour interfaces is in the validation "
                        "set; the wire slip rate is the representative value "
                        "(r / |sin a|) times the plastic strain rate",
                    )
                ),
            ),
        ),
    )


def _mkappa_pressure_history(
    contact_pressure_Pa: Mapping[str, FloatArray],
    names: Sequence[str],
    shape: tuple[int, ...],
) -> dict[str, FloatArray]:
    """Check a per-layer ``P1+P2`` history for :func:`compute_mkappa`."""
    if not isinstance(contact_pressure_Pa, Mapping):
        raise CableSchemaError(
            "contact_pressure_Pa must map armour layer names to pressure histories"
        )
    given, expected = set(contact_pressure_Pa), set(names)
    if given != expected:
        raise CableSchemaError(
            "contact_pressure_Pa must give every structural armour layer "
            f"{sorted(expected)}; missing {sorted(expected - given)}, "
            f"unknown {sorted(given - expected)}"
        )
    out: dict[str, FloatArray] = {}
    for name in names:
        label = f"contact_pressure_Pa[{name!r}]"
        values = real_array(contact_pressure_Pa[name], label).astype(np.float64)
        require_finite(values, label)
        if values.shape != shape:
            raise CableSchemaError(f"{label} must match the kappa_history shape")
        if np.any(values < 0.0):
            raise CableSchemaError(f"{label} must be >= 0.")
        out[name] = values
    return out


def _papailiou_slip_fraction(
    layers: list[PapailiouLayer], kappa: FloatArray
) -> FloatArray:
    """Fraction of Papailiou helix-angle stations in slip per step (diagnostic).

    Re-runs the station spring-sliders (signed capstan interval of each
    layer's :meth:`~cablexsec.mechanics.papailiou.PapailiouLayer.station_bounds`)
    tracking slip — consistent with the moment that
    :func:`compute_mkappa_papailiou` returns.
    """
    if not layers:
        return np.zeros(np.asarray(kappa).size, dtype=np.float64)
    station = [lyr.station_bounds() for lyr in layers]
    forces = [np.zeros_like(s[0]) for s in station]
    total = sum(len(s[0]) for s in station) or 1
    kap = np.asarray(kappa, dtype=np.float64)
    sf = np.zeros(kap.size, dtype=np.float64)
    kappa_prev = 0.0
    for t in range(kap.size):
        d_kappa = kap[t] - kappa_prev
        kappa_prev = kap[t]
        n_slip = 0
        for i, (k_phi, upper, lower, _w) in enumerate(station):
            active = k_phi > 0.0
            f_new, _slip, slipping = _station_return(
                forces[i][active],
                d_kappa,
                k_phi[active],
                upper[active],
                lower[active],
            )
            forces[i][active] = f_new
            n_slip += int(np.count_nonzero(slipping))
        sf[t] = n_slip / total
    return sf


# --------------------------------------------------------------------------- #
# Metallic-sheath wall stress                                                  #
# --------------------------------------------------------------------------- #
def compute_shell_stress(
    cable: CableData,
    layer_name: str,
    kappa_1_m: float,
    eps_axial: float,
    *,
    p_int_Pa: float = 0.0,
    p_ext_Pa: float = 0.0,
    delta_T_K: float = 0.0,
) -> ShellStressResult:
    r"""Five-contribution wall stress of a cable's metallic sheath.

    Thin wrapper over :func:`cablexsec.mechanics.shell_bending.shell_wall_stress`
    that resolves the named ``cylindrical_shell`` layer and the total axial
    stiffness from the cable. The sheath carries its EA-weighted axial share; the
    SCF-amplified bending acts at the corrugation root. For *damage*, see
    :func:`compute_shell_fatigue` (corrugated-Al via EN 1999-1-3; a lead sheath
    stays BLOCKED — a stated Phase-2 limitation).

    Parameters
    ----------
    cable : CableData
        Cable containing the metallic sheath.
    layer_name : str
        Name of the ``cylindrical_shell`` layer.
    kappa_1_m : float
        Cable curvature [1/m].
    eps_axial : float
        Global cable axial strain.
    p_int_Pa, p_ext_Pa, delta_T_K : float, optional
        Internal/external pressure [Pa] and sheath temperature rise [K].
        The temperature term is the restrained increment at the supplied
        global strain. Use compute_component_response to solve common free
        strain and differential thermal stress in axial equilibrium.

    Returns
    -------
    ShellStressResult

    Raises
    ------
    CableSchemaError
        If ``layer_name`` is not a ``cylindrical_shell`` layer of the cable.

    Examples
    --------
    >>> from cablexsec import load_cable, compute_shell_stress, cigre_tb880_yaml_path
    >>> cab = load_cable(cigre_tb880_yaml_path(), geometry_override=None)
    >>> s = compute_shell_stress(cab, "cu_metallic_sheath", 0.03, 1e-3,
    ...                          p_ext_Pa=2e6, delta_T_K=40.0)
    >>> s.sigma_total_Pa > 0.0
    True
    """
    try:
        layer = cable.get_layer(layer_name)
    except KeyError as exc:
        raise CableSchemaError(str(exc)) from None
    if not isinstance(layer, CylindricalShellLayer):
        raise CableSchemaError(
            f"layer {layer_name!r} is a {type(layer).__name__}, not a "
            "cylindrical_shell; compute_shell_stress needs a metallic-sheath layer."
        )
    require_finite(
        (kappa_1_m, eps_axial, p_int_Pa, p_ext_Pa, delta_T_K), "shell-stress inputs"
    )
    ea_total = ea_fixed(cable, metals_only=False)
    return shell_wall_stress(
        layer,
        kappa_1_m,
        eps_axial,
        ea_total,
        p_int_Pa=p_int_Pa,
        p_ext_Pa=p_ext_Pa,
        delta_T_K=delta_T_K,
    )


@dataclass(frozen=True)
class ShellFatigueResult:
    """Metallic-sheath wall fatigue over a curvature history.

    Attributes
    ----------
    D_miner : float
        Miner cumulative damage at the corrugation root.
    n_cycles : float
        Total rainflow cycle count in the history.
    sigma_peak_amplitude_max_Pa : float
        Largest SCF-amplified corrugation-root bending stress reached [Pa].
    sn_curve : str
        Name of the S-N curve used (e.g. ``en1999_1_3_aluminium``).
    layer_name : str
    provenance : Provenance or None
    """

    D_miner: float
    n_cycles: float
    sigma_peak_amplitude_max_Pa: float
    sn_curve: str
    layer_name: str
    provenance: Provenance | None = None
    method: str = "elastic_sn"
    strain_life: SheathStrainLifeResult | None = None


def compute_shell_fatigue(
    cable: CableData,
    layer_name: str,
    kappa_history: FloatArray,
    *,
    method: Literal["elastic_sn", "strain_life"] = "elastic_sn",
    kappa_y: FloatArray | None = None,
    strain_life: str | StrainLifeConstants | None = None,
    mean_stress_correction: MeanStressCorrection = "swt",
    axial_strain: FloatArray | None = None,
    core_state: CoreSlipState | None = None,
    n_theta: int = 36,
    n_stations: int | None = None,
    core_friction: CoreFrictionSpec | None = None,
    contact_tension_N: float = 0.0,
    contact_model: str = "bounded",
    hoop_ratio: float | None = None,
    sn_curve: str | SNcurve | None = None,
    mean_stress: MeanStressOption | None = None,
    scf: ComponentSCF | None = None,
    static_mean_stress_Pa: float = 0.0,
) -> ShellFatigueResult:
    r"""Metallic-sheath fatigue over a curvature history.

    Two routes. ``method="elastic_sn"`` (default, unchanged) is the elastic
    corrugated-Al route described below. ``method="strain_life"`` (gap P6)
    is the strain-controlled route of
    :func:`cablexsec.analysis.sheath_strain.strain_life_sheath_fatigue`: the
    fibre strain of the sheath (a concentric ``cylindrical_shell``, or a
    foil, sheath or tape of a power core under the declared ``core_state``),
    rainflow strain cycles, Masing stresses on the cyclic stress--strain
    curve and the Coffin--Manson--Basquin life with the SWT (default) or
    Morrow mean-stress correction, using the layer's ``strain_life``
    constants (or ``strain_life=``). It accepts strains above yield. A
    ``bonded_laminate`` foil follows the bonded polymer, including its hoop
    strain (``hoop_ratio`` overrides ``-nu`` of that polymer). The
    strain-life options are rejected by the elastic route.

    Elastic route.

    The cyclic driver is the SCF-amplified corrugation-root **bending** stress
    ``σ_peak(t) = SCF · E_eff · κ(t) · r_mid`` (the frozen shell-stress kernel;
    static hoop/thermal/axial terms shift the mean, not the rainflow *range*).
    Cycles are rainflow-counted ([ASTM E1049]) and Miner-summed against the
    layer's ``bulk_sn`` curve — for a corrugated-Al sheath, EN 1999-1-3
    ``en1999_1_3_aluminium`` (detail 1.4, 71-7).

    **Modeling choice (not a calibration):** the corrugated-Al sheath fatigue
    uses the EN 1999-1-3 **plain-member** detail category applied to the
    SCF-amplified corrugation-root peak stress. The corrugation geometry enters
    via the modeled SCF; the curve is plain-material to avoid double-counting.
    71-7 is the conservative general-alloy choice; 90-7 (detail 1.2) is the
    less-conservative alternative.

    Parameters
    ----------
    cable : CableData
        Cable containing the metallic sheath.
    layer_name : str
        Name of the ``cylindrical_shell`` layer.
    kappa_history : numpy.ndarray
        Cable curvature time series [1/m].
    sn_curve, mean_stress, scf, static_mean_stress_Pa
        User S-N options of the elastic route. ``sn_curve`` replaces the
        layer's ``bulk_sn`` (a registered name or an :class:`SNcurve`);
        ``mean_stress`` applies a mean-stress correction
        (:mod:`cablexsec.materials.mean_stress`; a method name takes the
        strength of the sheath material) to the cycle mean of the root
        bending stress plus ``static_mean_stress_Pa`` (hoop, thermal or
        axial stress that this route does not compute); ``scf`` scales the
        root bending stress by its normal-bending factors, mean and dynamic.
        Without them the result is unchanged; the strain-life route rejects
        them.

    Returns
    -------
    ShellFatigueResult

    Raises
    ------
    CableSchemaError
        If ``layer_name`` is not a ``cylindrical_shell`` layer, or has no
        ``bulk_sn`` curve assigned.
    BlockedSNCurveError
        If the sheath's ``bulk_sn`` is a deferred curve (e.g. a lead sheath:
        ``cigre_tb623_lead`` is BLOCKED — a stated Phase-2 limitation).
    ValueError
        If the declared elastic root bending stress exceeds the material yield
        stress, or that limit is unavailable. Static mean curvature is included
        in this check; a small cyclic range does not make a yielded state elastic.
    """
    from cablexsec.materials.library import get_property  # noqa: PLC0415
    from cablexsec.materials.sn_curves import get_sn_curve  # noqa: PLC0415

    kappa = real_array(kappa_history, "kappa_history")
    if kappa.ndim != 1:
        raise CableSchemaError("kappa_history must be one-dimensional")
    require_finite(kappa, "kappa_history")
    user_sn = (
        sn_curve is not None
        or mean_stress is not None
        or scf is not None
        or static_mean_stress_Pa != 0.0
    )
    if method == "strain_life":
        if user_sn:
            raise CableInputError(
                "sn_curve, mean_stress, scf and static_mean_stress_Pa apply to "
                "method='elastic_sn' (the strain-life route has its own SWT or "
                "Morrow correction)"
            )
        return _strain_life_shell_fatigue(
            cable,
            layer_name,
            kappa,
            kappa_y=kappa_y,
            strain_life=strain_life,
            correction=mean_stress_correction,
            axial_strain=axial_strain,
            core_state=core_state,
            n_theta=n_theta,
            n_stations=n_stations,
            core_friction=core_friction,
            contact_tension_N=contact_tension_N,
            contact_model=contact_model,
            hoop_ratio=hoop_ratio,
        )
    if method != "elastic_sn":
        raise CableInputError("method must be 'elastic_sn' or 'strain_life'")
    if any(
        v is not None
        for v in (
            kappa_y,
            strain_life,
            axial_strain,
            core_state,
            core_friction,
            hoop_ratio,
        )
    ):
        raise CableInputError(
            "kappa_y, strain_life, axial_strain, core_state, core_friction and "
            "hoop_ratio apply to method='strain_life' only"
        )
    try:
        layer = cable.get_layer(layer_name)
    except KeyError as exc:
        raise CableSchemaError(str(exc)) from None
    if not isinstance(layer, CylindricalShellLayer):
        raise CableSchemaError(
            f"layer {layer_name!r} is a {type(layer).__name__}, not a "
            "cylindrical_shell; compute_shell_fatigue needs a metallic-sheath layer."
        )
    if not layer.bulk_sn and sn_curve is None:
        raise CableSchemaError(
            f"shell layer {layer_name!r} has no bulk_sn S-N curve assigned; "
            "cannot compute sheath fatigue."
        )
    if sn_curve is None:
        sn = get_sn_curve(layer.bulk_sn)  # BlockedSNCurveError for a lead sheath
    else:
        sn = get_sn_curve(sn_curve) if isinstance(sn_curve, str) else sn_curve
        if not isinstance(sn, SNcurve) or sn.stress_unit != "MPa":
            raise CableInputError("sn_curve must be a stress-range S-N curve in MPa")
    if not math.isfinite(static_mean_stress_Pa):
        raise CableInputError("static_mean_stress_Pa must be finite")
    spec = (
        MeanStressSpec.from_material(mean_stress, layer.material)
        if isinstance(mean_stress, str)
        else mean_stress
    )

    ea_total = ea_fixed(cable, metals_only=False)
    # σ_peak is linear in κ; one frozen call gives the per-unit-curvature slope.
    k_per_kappa = shell_wall_stress(layer, 1.0, 0.0, ea_total).sigma_peak_Pa
    sigma_peak = k_per_kappa * kappa  # [Pa], the cyclic corrugation-root stress
    try:
        yield_stress = get_property(layer.material, "sigma_y_Pa")
    except KeyError:
        raise CableInputError(
            f"No yield stress is available for shell {layer_name!r}; "
            "the elastic sheath-fatigue range cannot be assessed."
        ) from None
    maximum_stress = float(np.max(np.abs(sigma_peak))) if sigma_peak.size else 0.0
    if not math.isfinite(yield_stress) or yield_stress <= 0:
        raise CableInputError("sheath fatigue requires a finite positive yield stress")
    if maximum_stress > yield_stress:
        raise CableInputError(
            f"Shell {layer_name!r}: elastic root bending stress "
            f"{maximum_stress / 1e6:.6g} MPa exceeds yield stress "
            f"{yield_stress / 1e6:.6g} MPa. Reassess the construction and "
            "constitutive range before interpreting an elastic S-N result."
        )

    damage = 0.0
    n_cycles = 0.0
    if user_sn and (needs_correction(sn, spec, scf) or static_mean_stress_Pa != 0.0):
        cyclic = (
            sigma_peak
            if scf is None
            else scaled_component(
                sigma_peak, scf.bending_normal_mean, scf.bending_normal_dynamic
            )
        )
        damage = corrected_damage(
            cyclic + static_mean_stress_Pa,
            sn,
            mean_stress=spec,
            thickness_m=layer.t_m,
        )
        n_cycles = math.fsum(c for _r, _m, c, _i, _j in rainflow_cycles(cyclic))
    else:
        for rng_Pa, _mean, count, _i0, _i1 in rainflow_cycles(sigma_peak):
            # EN 1999-1-3 Δσ is a stress RANGE; feed the rainflow range directly.
            damage += count / sn.cycles_to_failure(rng_Pa / 1.0e6)
            n_cycles += count
    user_notes: tuple[str, ...] = ()
    if user_sn:
        user_notes = (
            f"user S-N options: curve {sn.name} ({sn.reference}); "
            + (spec.describe() if spec is not None else "no mean-stress correction")
            + f"; static mean {static_mean_stress_Pa / 1e6:.6g} MPa; "
            + (scf.describe() if scf is not None else "no component SCF"),
        )

    return ShellFatigueResult(
        D_miner=float(damage),
        n_cycles=float(n_cycles),
        sigma_peak_amplitude_max_Pa=maximum_stress,
        sn_curve=sn.name,
        layer_name=layer_name,
        provenance=Provenance(
            method="metallic-sheath wall fatigue: rainflow + Miner on the "
            "SCF-amplified corrugation-root bending stress range",
            citations=(
                "[EN1999-1-3] Table J.1/J.2 detail 1.4 (71-7)",
                "[ASTM E1049]",
                "[Miner1945]",
            ),
            gates=(
                "N_f = C·Δσ^-7 closed form (curve registers correctly)",
                "below the 40.6 MPa range cut-off -> infinite life / zero damage",
            ),
            limitations=(
                "MODELING CHOICE (not calibration): EN 1999-1-3 plain-member "
                "category 71-7 (type 1.4) on the SCF-amplified root stress; "
                "geometry via the modeled SCF, plain-material curve to avoid "
                "double-counting; 90-7 (detail 1.2) is the less-conservative "
                "alternative",
                "single slope m=7, flat constant-amplitude cut-off at N_L=1e8 "
                "(Δσ_L=40.6 MPa)",
                "Δσ is a stress RANGE (rainflow range); no Goodman mean "
                "correction. EN 1999-1-3 derives the 71-7 category for stress "
                "ratio R>=0.5 (J.1(2)); FULLY-REVERSED curvature bending gives "
                "R≈-1 (outside that validity) — applying the category there is a "
                "stated modeling choice, not a validated mean-stress treatment",
                "RESULT FIDELITY IS BOUNDED BY THE LAYER SCF AND THE SUPPLIED "
                "HISTORY: damage ∝ SCF^7 on the m=7 slope (halving SCF cuts "
                "damage ~128x), so a representative f_corr/SCF yields a "
                "capability demonstration, NOT a design fatigue life — a real "
                "assessment needs the cable's actual corrugation SCF and design "
                "sea state",
                "cyclic stress is bending-driven; static hoop/thermal/axial "
                "terms shift the mean, not the range",
                "lead sheath (cigre_tb623_lead) stays BLOCKED — a stated "
                "Phase-2 limitation",
                *user_notes,
            ),
        ),
    )


def _strain_life_shell_fatigue(
    cable: CableData,
    layer_name: str,
    kappa: FloatArray,
    *,
    kappa_y: FloatArray | None,
    strain_life: str | StrainLifeConstants | None,
    correction: MeanStressCorrection,
    axial_strain: FloatArray | None,
    core_state: CoreSlipState | None,
    n_theta: int,
    n_stations: int | None,
    core_friction: CoreFrictionSpec | None,
    contact_tension_N: float,
    contact_model: str,
    hoop_ratio: float | None,
) -> ShellFatigueResult:
    """Strain-life route of :func:`compute_shell_fatigue` (gap P6)."""
    from cablexsec.analysis.sheath_strain import (  # noqa: PLC0415
        strain_life_sheath_fatigue,
    )

    ky = None
    if kappa_y is not None:
        ky = real_array(kappa_y, "kappa_y")
        if ky.shape != kappa.shape:
            raise CableSchemaError("kappa_y must match kappa_history")
        require_finite(ky, "kappa_y")
    eps = None
    if axial_strain is not None:
        eps = real_array(axial_strain, "axial_strain")
        if eps.shape != kappa.shape:
            raise CableSchemaError("axial_strain must match kappa_history")
        require_finite(eps, "axial_strain")
    result = strain_life_sheath_fatigue(
        cable,
        layer_name,
        kappa,
        ky,
        strain_life=strain_life,
        correction=correction,
        axial_strain=eps,
        core_state=core_state,
        n_theta=n_theta,
        n_stations=n_stations,
        core_friction=core_friction,
        contact_tension_N=contact_tension_N,
        contact_model=contact_model,
        hoop_ratio=hoop_ratio,
    )
    const = result.constants
    limitations = [
        f"strain-life constants {const.name!r} ({const.basis}): {const.reference}",
        "Masing cyclic stress-strain curve with closed-loop memory; cyclic "
        "mean-stress relaxation not followed",
        "uniaxial strain-life curve; the foil or sheath is strain controlled by "
        "the body it follows",
    ]
    if const.cyclic_curve_is_estimate:
        limitations.append("cyclic curve K', n' by the compatibility estimate n' = b/c")
    if const.basis != "published":
        limitations.append(
            f"constants are {const.basis}: a bound estimate, not a material "
            "property; run the stated bounds"
        )
    if result.core_state is not None:
        limitations.append(
            f"core slip state {result.core_state!r} (stuck: plane-section upper "
            "bound; free: full-slip lower bound of the core-axis strain)"
        )
    layer = cable.get_layer(layer_name)
    if (
        isinstance(layer, CylindricalShellLayer)
        and layer.construction != "bonded_laminate"
        and (layer.scf != 1.0 or layer.f_corrugation != 1.0)
    ):
        limitations.append(
            "corrugation strain concentration SCF f_corr applied to strain "
            "(elastic ratio; its growth after root yielding is not included, "
            "a lower estimate)"
        )
    return ShellFatigueResult(
        D_miner=result.D_miner,
        n_cycles=result.n_cycles,
        sigma_peak_amplitude_max_Pa=result.stress_max_Pa,
        sn_curve=const.name,
        layer_name=layer_name,
        method="strain_life",
        strain_life=result,
        provenance=Provenance(
            method=(
                "metallic-sheath strain-life fatigue: rainflow strain cycles, "
                "Masing stresses, Coffin-Manson-Basquin life with the "
                f"{result.correction!r} mean-stress correction, Miner sum"
            ),
            citations=(
                "[Basquin1910]",
                "[Coffin1954]",
                "[Manson1954]",
                "[Morrow1968]",
                "[SmithWatsonTopper1970]",
                "[Dowling2013] Ch. 14",
                "[ASTM E1049]",
                "[Worzyk2009] Sec. 2.3.2",
            ),
            gates=(
                "fully reversed SWT with compatible K', n' equals the "
                "uncorrected curve",
                "Masing loop closure and memory",
            ),
            limitations=tuple(limitations),
        ),
    )


def compute_conductor_fatigue(
    cable: CableData,
    kappa_x: FloatArray,
    kappa_y: FloatArray,
    *,
    tension_N: FloatArray | None = None,
    layer_name: str | None = None,
    mu_wire: float | None = None,
    mu_outer: float | None = None,
    reference_tension_N: float | None = None,
    external_pressure_Pa: float = 0.0,
    pressure_transmission: PressureTransmission = "outer_interface",
    n_orientations: int = 8,
    partial_slip_cycles: PartialSlipCycles = "skip",
    rainflow_counter: RainflowCounter = "auto",
    trellis_fretting: bool = False,
    trellis_model: Literal["critical_distance", "surface_swt"] = "critical_distance",
    critical_distance: CriticalDistanceSpec | None = None,
    contact_yield_Pa: float | None = DEFAULT_CONTACT_YIELD_PA,
    contact_pressure_factor: float = FULLY_PLASTIC_PRESSURE_FACTOR,
    strain_life: str | StrainLifeConstants | None = None,
    strain_life_correction: MeanStressCorrection = "swt",
    sn_curve: str | SNcurve | None = None,
    mean_stress: MeanStressOption | None = "swt",
    scf: ComponentSCF | None = None,
) -> ConductorFatigueResult:
    r"""Strand-slip fatigue of a resolved stranded conductor.

    The conductor layer (a ``cylindrical_offset`` layer with resolved
    ``strand_layers``) is assessed ring by ring with the multilinear
    stick-slip operator of [KarlsenHusoy2026]; the default damage is the bulk
    wire-stress S-N count (``bulk_sn``), which reproduces four of the five
    published Nasution conductor test groups within a factor of 3.
    Per-slip-cycle SWT fretting at the trellis contacts of oppositely laid
    rings (:func:`cablexsec.analysis.conductor_fatigue.assess_conductor_fatigue`)
    is evaluated only with ``trellis_fretting=True`` and the layer's
    ``fretting_sn``. It is **not validated**: it predicts the dry tests one to
    two orders of magnitude short and does not resolve the bending radius
    (gap 1.3c), and requesting it emits
    :class:`~cablexsec.provenance.UnvalidatedModelWarning`. A ``fretting_sn``
    declared but not requested emits
    :class:`~cablexsec.provenance.UnevaluatedFatigueCurveWarning`.
    The conductor axis follows the cable curvature; the axial strain is the
    tension divided by the cable's fixed-end axial stiffness. The power core
    is taken in full slip inside the cable (no core-level stick order).

    Parameters
    ----------
    cable : CableData
        A fully specified cable with a resolved stranded conductor.
    kappa_x, kappa_y : numpy.ndarray
        Curvature history [1/m].
    tension_N : numpy.ndarray or None, optional
        True material force history [N].
    layer_name : str or None, optional
        Conductor layer; defaults to the only layer with resolved strands that
        declares a ``bulk_sn`` curve.
    mu_wire : float or None, optional
        Wire-wire friction; defaults to the library ``mu_on_cu`` of the
        conductor material. [Nasution2014b] and Hu et al. (2022) use 0.2.
    mu_outer : float or None, optional
        Friction on the conductor surface; defaults to ``mu_wire``.
    reference_tension_N : float or None, optional
        Tension for the friction parameters and crossing forces (default: the
        largest absolute tension of the history).
    external_pressure_Pa : float, optional
        Contact pressure of the conductor screen on the conductor surface.
    pressure_transmission : {"all_interfaces", "outer_interface"}, optional
        Transmission of that pressure to inner interfaces.
    n_orientations : int, optional
        Azimuths per ring (default 8).
    partial_slip_cycles : {"skip", "cattaneo_mindlin"}, optional
        Treatment of fretting cycles without slip.
    rainflow_counter : {"auto", "compiled", "reference"}, optional
        Counter for the bulk Miner sums.
    trellis_fretting : bool, optional
        Add the trellis SWT fretting damage (default ``False``; not
        validated).
    contact_yield_Pa : float or None, optional
        Wire yield stress for the trellis pressure limit (default 300 MPa,
        [Nasution2014b] Fig. 9); ``None`` keeps the elastic Hertz pressure.
        Used only with ``trellis_fretting=True``.
    contact_pressure_factor : float, optional
        Pressure limit over the yield stress (default 3, [Johnson1985] §6.3).
    strain_life : str, StrainLifeConstants or None, optional
        Gap P7: strain-life constants (for example ``"copper_poon2022"`` or
        ``"copper_annealed_mus_derived"``) for an additional strain-life
        damage of the outer-fibre strain (``D_strain_life_per_ring``;
        Masing cyclic curve), which assesses ranges above the wire yield.
        ``None`` (default) leaves the result unchanged.
    strain_life_correction : {"swt", "morrow", "none"}, optional
        Mean-stress correction of the strain-life route.
    sn_curve : str or SNcurve, optional
        Bulk curve replacing the layer's ``bulk_sn`` (a registered name or a
        curve).
    mean_stress : MeanStressSpec or str or None, optional
        Mean-stress correction of the bulk cycles
        (:mod:`cablexsec.materials.mean_stress`; a method name takes the
        strength of the conductor material). Default ``"swt"``: every cycle
        is mapped to the stress ratio of the bulk curve by the Smith-Watson-
        Topper correction, the configuration that reproduces the Nasution
        conductor tests (gap CF2, doc/benchmarks.md). ``None`` counts the
        unmodified ranges (the behaviour before gap CF2).
    trellis_model : {"critical_distance", "surface_swt"}, optional
        Trellis rule used with ``trellis_fretting=True``. The default is the
        point method of critical distances
        (:class:`~cablexsec.analysis.conductor_fatigue.CriticalDistanceSpec`),
        which needs no ``fretting_sn``; ``"surface_swt"`` is the per-slip-cycle
        surface rule of gap 1.3c (not validated; needs ``fretting_sn``).
    critical_distance : CriticalDistanceSpec, optional
        Threshold, material length and sampling of the critical-distance rule
        (default: the class defaults).
    scf : ComponentSCF, optional
        Factors on the axial part (axial factors) and on the stick-slip
        bending strain of the operator (friction factors), mean and dynamic.

    Returns
    -------
    ConductorFatigueResult

    Raises
    ------
    CableSchemaError
        If no resolved conductor exists, ``bulk_sn`` is missing or not a
        stress-range curve, ``fretting_sn`` names no registered SWT
        constants, or ``trellis_fretting=True`` without a ``fretting_sn``.
    """
    from cablexsec.cables.schema import CylindricalOffsetLayer  # noqa: PLC0415
    from cablexsec.materials.library import get_property  # noqa: PLC0415
    from cablexsec.materials.sn_curves import (  # noqa: PLC0415
        SWT_LIFE,
        get_sn_curve,
        get_swt_life,
    )

    require_fully_specified(cable, "compute_conductor_fatigue")
    kx = real_array(kappa_x, "kappa_x")
    ky = real_array(kappa_y, "kappa_y")
    if kx.ndim != 1 or kx.shape != ky.shape or kx.size < 2:
        raise CableSchemaError("kappa_x and kappa_y must be equal 1-D histories")
    require_finite(kx, "kappa_x")
    require_finite(ky, "kappa_y")
    tension = None
    if tension_N is not None:
        tension = real_array(tension_N, "tension_N")
        if tension.shape != kx.shape:
            raise CableSchemaError("tension_N must match the curvature histories")
        require_finite(tension, "tension_N")
    candidates = [
        layer
        for layer in cable.layers
        if isinstance(layer, CylindricalOffsetLayer)
        and layer.strand_layers
        and (layer.name == layer_name if layer_name is not None else layer.bulk_sn)
    ]
    if len(candidates) != 1:
        raise CableSchemaError(
            "compute_conductor_fatigue needs exactly one cylindrical_offset layer "
            "with resolved strand_layers"
            + (
                " that declares bulk_sn; pass layer_name"
                if layer_name is None
                else f" named {layer_name!r}"
            )
        )
    layer = candidates[0]
    if not layer.bulk_sn and sn_curve is None:
        raise CableSchemaError(f"conductor {layer.name!r} has no bulk_sn curve")
    try:
        if sn_curve is None:
            sn_bulk = get_sn_curve(layer.bulk_sn)
        else:
            sn_bulk = get_sn_curve(sn_curve) if isinstance(sn_curve, str) else sn_curve
    except (BlockedSNCurveError, KeyError) as exc:
        raise CableSchemaError(
            f"conductor {layer.name!r} bulk S-N unavailable: {exc}"
        ) from None
    conductor_spec = (
        MeanStressSpec.from_material(mean_stress, layer.material)
        if isinstance(mean_stress, str)
        else mean_stress
    )
    if sn_bulk.stress_unit != "MPa":
        raise CableSchemaError("conductor bulk fatigue needs a stress-range curve")
    swt = None
    if layer.fretting_sn and layer.fretting_sn not in SWT_LIFE:
        raise CableSchemaError(
            f"conductor {layer.name!r} fretting_sn={layer.fretting_sn!r} is not "
            f"a registered SWT strain-life key ({', '.join(sorted(SWT_LIFE))})"
        )
    unevaluated = ""
    if trellis_model not in ("critical_distance", "surface_swt"):
        raise CableSchemaError(
            "trellis_model must be 'critical_distance' or 'surface_swt'"
        )
    cd_spec = None
    if trellis_fretting and trellis_model == "critical_distance":
        cd_spec = (
            critical_distance
            if critical_distance is not None
            else CriticalDistanceSpec()
        )
    elif trellis_fretting:
        if not layer.fretting_sn:
            raise CableSchemaError(
                f"trellis_fretting=True needs a fretting_sn on conductor {layer.name!r}"
            )
        swt = get_swt_life(layer.fretting_sn)
    elif layer.fretting_sn:
        unevaluated = (
            f"conductor {layer.name!r} declares fretting_sn={layer.fretting_sn!r}, "
            "but the trellis term is evaluated only with trellis_fretting=True; "
            "the damage is the bulk S-N count (gaps 1.3c, CF2). Pass "
            "trellis_fretting=True to add it."
        )
        warnings.warn(unevaluated, UnevaluatedFatigueCurveWarning, stacklevel=2)
    rings = tuple(
        ConductorRing.from_pitch(
            ring.n_wires,
            ring.d_wire_m,
            ring.r_helix_m,
            ring.pitch_m,
            1.0 if ring.lay_direction == "right_hand" else -1.0,
            ring.wire_area_m2,
            ring.compaction_scf,
        )
        for ring in layer.strand_layers
    )
    modulus = get_property(layer.material, "E_eff_axial_Pa")
    nu = get_property(layer.material, "nu")
    friction = get_property(layer.material, "mu_on_cu") if mu_wire is None else mu_wire
    result = assess_conductor_fatigue(
        rings,
        modulus,
        nu,
        kx,
        ky,
        tension,
        sn_bulk=sn_bulk,
        swt_const=swt,
        mu_wire=friction,
        mu_outer=mu_outer,
        strain_per_tension=1.0 / ea_fixed(cable),
        reference_tension_N=reference_tension_N,
        external_pressure_Pa=external_pressure_Pa,
        r_outer_m=layer.r_outer_m,
        pressure_transmission=pressure_transmission,
        n_orientations=n_orientations,
        partial_slip_cycles=partial_slip_cycles,
        rainflow_counter=rainflow_counter,
        contact_yield_Pa=contact_yield_Pa,
        contact_pressure_factor=contact_pressure_factor,
        strain_life=(
            None
            if strain_life is None
            else (
                get_strain_life(strain_life)
                if isinstance(strain_life, str)
                else strain_life
            )
        ),
        correction=strain_life_correction,
        mean_stress=conductor_spec,
        scf=scf,
        critical_distance=cd_spec,
    )
    if unevaluated:
        result = replace(
            result,
            provenance=replace(
                result.provenance,
                limitations=(*result.provenance.limitations, unevaluated),
            ),
        )
    return result


# --------------------------------------------------------------------------- #
# Fatigue (cable-level adapter over the screening spec engine)                 #
# --------------------------------------------------------------------------- #
def compute_fatigue(
    cable: CableData,
    kappa_x: FloatArray,
    kappa_y: FloatArray,
    *,
    tension_N: FloatArray | None = None,
    axial_strain: FloatArray | None = None,
    contact_tension_N: float | None = None,
    contact_tension_history_N: FloatArray | None = None,
    mu: float | None = None,
    contact_model: str | None = None,
    thermal_stress_Pa: dict[str, FloatArray] | None = None,
    section_states: Sequence[SectionState] | None = None,
    n_fibres: int = 32,
    rainflow_counter: RainflowCounter = "auto",
    friction_model: FrictionModel | None = None,
    distributed_nodes_per_pitch: int | Mapping[str, int] | None = None,
    distributed_max_increment: float = DEFAULT_MAX_INCREMENT,
    distributed_bound: DistributedBound = "signed",
    copper_contact_model: CopperContactModel = "slip_cycle",
    partial_slip_cycles: PartialSlipCycles = "skip",
    termination: TerminationSpec | Mapping[str, TerminationSpec] | None = None,
    thermal: ThermalHistory | ThermalState | None = None,
    time_s: FloatArray | None = None,
    progressive: ProgressiveFriction | None = None,
    sn_curves: Mapping[str, str | SNcurve] | None = None,
    mean_stress: MeanStressOption | Mapping[str, MeanStressOption] | None = None,
    scf: ComponentSCF | Mapping[str, ComponentSCF] | None = None,
    count_from_sample: int = 0,
    friction_law: FrictionLaw | None = None,
) -> FatigueResult:
    r"""Per-wire fatigue of a cable over a biaxial curvature (and tension) history.

    Assembles the :class:`FatigueLayer` specs from the cable YAML
    (:func:`cablexsec.cable_adapters.cable_to_fatigue_layers`: bulk S-N from
    ``bulk_sn``, steel slip-amplitude or copper SWT fretting routed by
    ``fretting_sn``, Hertz ``K_t``/``p_0`` from the opposite-laid partner layer)
    and runs rainflow counting and Miner summation. The selected material
    curve determines any mean-stress treatment; no generic Goodman correction
    is applied to the adopted armour stress-range curve.
    An omitted/empty ``fretting_sn`` explicitly selects bulk-only assessment.
    A requested but unknown or unavailable contact-life law raises an input
    error rather than silently contributing zero contact damage.
    Only structural helical armour is assessed. A ``bulk_sn`` or
    ``fretting_sn`` declared on any other layer (for example the stranded
    copper conductor of a three-core cable, or a metallic sheath) is not
    evaluated, and an armour ``fretting_sn`` without a direct crossed-wire
    contact has no contact damage. Each such case emits
    :class:`~cablexsec.provenance.UnevaluatedFatigueCurveWarning` and is
    appended to ``provenance.limitations``; the damage values are unchanged.
    The copper SWT branch counts cycles of each crossing's tangential force,
    skips cycles without slip and evaluates one SWT driver per slipping cycle
    (see :func:`cablexsec.analysis.fatigue.compute_fatigue`); the former
    one-driver-per-bulk-cycle estimate is available only through
    ``copper_contact_model="legacy_gross_sliding"``. The stranded conductor is
    assessed by :func:`compute_conductor_fatigue`.

    Parameters
    ----------
    cable : CableData
        A fully-specified cable.
    kappa_x, kappa_y : numpy.ndarray
        Axial-strain-gradient components [1/m], giving bending strain
        kappa_x*x+kappa_y*y in the material frame.
    tension_N : numpy.ndarray or None, optional
        True material force [N] for the fixed-radius ``σ_A`` approximation
        (uses ``EA_fixed``). Effective tension must be pressure-converted.
    axial_strain : numpy.ndarray or None, optional
        Stress-driving axial strain for the fixed-radius uniaxial law,
        instead of true material force divided by the matched axial stiffness.
        For a linear global line model this is the direct tensile strain, not
        the total mean axial strain including Poisson deformation. This input
        does not add radial deformation or pressure redistribution. Use
        ``section_states`` to retain solved coupled-section stresses instead.
        Tension and prescribed strain are mutually exclusive inputs.
    contact_tension_N : float or None, optional
        Explicit frozen reference tension for contact ``F_n`` [N]. When omitted
        and ``tension_N`` is supplied, contact pressure and normal force are
        refreshed over the tension history. With no tension history, zero is used.
        Registered contact-life laws require constant normal contact; changing
        force with frozen Hertz stiffness/pressure is rejected. Bulk-only
        assessment retains the changing normal load. A supplied frozen reference
        is an explicit modelling assumption, not a variable-contact solution.
    contact_tension_history_N : numpy.ndarray or None, optional
        Force history [N] that sets the armour contact (declared manufacturing
        clamping plus the tension increment) while ``axial_strain`` drives
        the wire stress: the contact of the tension route over a
        prescribed-strain record, for example the strain times the axial
        stiffness ``EA_fixed`` it was formed with. Non-negative, the shape
        of the curvature histories; only with ``axial_strain`` and without
        ``contact_tension_N``. With the same force on both routes the damage
        equals that of ``tension_N``.
    mu : float, optional
        Armour friction coefficient for the M-κ stress (default ``mu_on_hdpe``).
    section_states : sequence of SectionState, optional
        One normal equilibrium per curvature sample, obtained with
        :func:`solve_section_state`. Repeated states may reuse the same object.
        Wire mean stress, material moduli, shear support and face pressures
        come from these states. Bending slip is retained between samples.
        Tension, prescribed strain, frozen contact tension and a separate
        thermal stress must then be omitted. Variable normal states currently
        support bulk fatigue; a registered contact-life branch requires
        constant Hertz contact properties. The concentric ``qp`` tension
        interface constructs consistent normal states automatically.
    n_fibres : int, optional
        Equally spaced material fibres around each circular wire (default 32).
        Each fibre is counted separately; per-wire bulk damage is the largest
        accumulated fibre damage. Increase this count to check angular
        convergence. An instantaneous stress envelope is not a fibre history.
    rainflow_counter : {"auto", "compiled", "reference"}, optional
        Rainflow counter for the bulk Miner sums. ``"auto"`` (default) uses the
        compiled counter when numba is active and the reference counter
        otherwise; both give the same damage. See
        :func:`cablexsec.analysis.fatigue.compute_fatigue`.
    friction_model : {"direction_tied", "auto", "distributed", "progressive"}, optional
        Armour friction-stress model. ``"direction_tied"`` (the default,
        ``None``) keeps the pointwise Papailiou cap at the held bending
        direction. ``"progressive"``
        splits each wire of that law into parallel sub-elements whose caps
        average to the same cap (gap G8; settings ``progressive``, default the
        friction-continuity density). ``"distributed"``
        solves the elastic--Coulomb friction field along the helix of each
        layer with the capstan bound ``mu |sin a|`` and the Yun interface
        stiffness ``xi/(1 - xi)``; the model and its parameters are recorded in
        ``provenance``. It requires a constant stick modulus over
        ``section_states`` and a thermal history (one ``ThermalState`` per
        call). ``"auto"`` (opt-in) takes the direction-tied model for a
        planar curvature path and the distributed model for any other, where
        it applies: the direction-tied cap turns with the bending direction
        and under rotating curvature gives friction-stress ranges below those
        of the distributed model. The model of every layer is recorded in
        ``provenance``. The default changes to ``"auto"`` in the next
        release: left at its default on a non-planar path (the only case in
        which the two differ), the direction-tied model warns once per call
        (:class:`~cablexsec.provenance.DirectionTiedFrictionWarning`, a
        ``FutureWarning``, with the measured bias); the result of this
        release is unchanged. See
        :func:`cablexsec.analysis.fatigue.compute_fatigue`.
    distributed_nodes_per_pitch : int, dict[str, int] or None, optional
        Helix nodes per pitch for the distributed model (all layers or per
        layer; default four per wire).
    distributed_max_increment : float, optional
        Largest curvature increment per substep of the distributed model, in
        units of the slip-onset curvature (default 0.1).
    distributed_bound : {"signed", "uniform"}, optional
        Friction bound of the distributed model: ``"signed"`` (default, the
        capstan term on the signed bending force of the wire) or
        ``"uniform"`` (no capstan term).
    progressive : ProgressiveFriction, optional
        Settings of the progressive law; only with
        ``friction_model="progressive"``.
    copper_contact_model : {"slip_cycle", "legacy_gross_sliding"}, optional
        Copper SWT contact branch; ``"slip_cycle"`` (default) is the per-cycle
        slip-history assessment, ``"legacy_gross_sliding"`` the former estimate.
    partial_slip_cycles : {"skip", "cattaneo_mindlin"}, optional
        Treatment of non-slipping cycles in the ``"slip_cycle"`` branch.
    termination : TerminationSpec, dict[str, TerminationSpec] or None, optional
        Clamp correction for a section near an end fitting or a clamped
        bend-stiffener root (all armour layers, or per layer name): the
        friction stress becomes the sectional value plus the anchored-wire
        clamp increment along the curvature history. The lay hand of each
        layer comes from the cable. The anchor is rigid unless
        ``TerminationSpec.k_anchor_N_m`` gives its stiffness per wire. See
        :func:`cablexsec.analysis.fatigue.compute_fatigue` and
        doc/local_analysis_guide.md.
    thermal : ThermalHistory or ThermalState, optional
        Energised analysis (opt-in). Layer
        temperatures on the slow time scale are coupled one way into the
        solved radial contact state (free thermal strain and registered
        polymer ``E(T)``: clamping ``P1+P2``, friction caps and crossing
        loads), into the stick stiffness through the Yun coefficient and into
        the wire stress (``sigma_T`` at fixed tension). The state is frozen
        within each slow-time block; the slip state is carried across blocks.
        Requires ``contact_model="solved"`` (the default when ``thermal`` is
        given). All layers at ``T_ref`` reproduce the unenergised solved
        result bit for bit. See :mod:`cablexsec.analysis.thermal_coupling`
        and :func:`~cablexsec.analysis.thermal_coupling.compute_thermal_fatigue`
        for block-weighted damage.
    time_s : numpy.ndarray, optional
        Sample times [s] of the curvature record, placing it on the slow time
        axis of a multi-node ``ThermalHistory``, and the time base of a
        time-dependent ``friction_law``. Refused when neither uses it.
    sn_curves : dict[str, str or SNcurve], optional
        Bulk S-N curve per armour layer name, replacing the layer's
        ``bulk_sn`` (a registered name, see
        :func:`cablexsec.materials.sn_curves.register_sn_curve`, or a curve).
    mean_stress : MeanStressSpec, str or dict, optional
        Mean-stress correction of the bulk cycles, for every layer or per
        layer name (:mod:`cablexsec.materials.mean_stress`). A method name
        (``"goodman"``, ``"gerber"``, ``"soderberg"``, ``"swt"``,
        ``"dnv_base"``, ``"dnv_pwht"``) takes the strength of the layer's
        library material with its citation. ``None`` (default) counts the
        unmodified ranges, bit for bit as before.
    scf : ComponentSCF or dict, optional
        Stress concentration factors of the axial, friction and local
        bending parts, mean and dynamic, for every layer or per layer name
        (:class:`~cablexsec.analysis.sn_assessment.ComponentSCF`).
    count_from_sample : int, optional
        Lead-in: the friction state runs through the whole history from the
        stress-free section, and the bulk cycles are counted from this sample
        on (default 0, every sample, bit for bit as before). Not available
        with a contact-life branch or with ``section_states``.
    friction_law : FrictionLaw, optional
        Static/dynamic and rate-dependent armour friction coefficient
        (:class:`~cablexsec.mechanics.friction_law.FrictionLaw`), with the
        sample times ``time_s`` when it depends on time: the direction-tied
        model takes every option, the distributed model the static/dynamic
        option; refused with the progressive model and a termination
        correction. A pressure factor of the law follows the contact pressure
        of every sample (both models). The crossing contact-life branches keep
        their rate-independent coefficient. ``None`` (default) or a neutral
        law gives the earlier result, bit for bit.

    Returns
    -------
    FatigueResult

    References
    ----------
    [ASTM E1049]; [Miner1945]; [DNVGL-RP-C203]; [Perier2009]; [Poon2022];
    [Poon2023].

    Examples
    --------
    >>> import numpy as np
    >>> from cablexsec import load_cable, compute_fatigue, menard2023_yaml_path
    >>> cab = load_cable(menard2023_yaml_path(), geometry_override=None)
    >>> k = 0.02 * np.sin(np.linspace(0, 20 * np.pi, 400))
    >>> res = compute_fatigue(cab, k, np.zeros_like(k))
    >>> res.critical_layer in {"inner_armour", "outer_armour"}
    True
    """
    require_fully_specified(cable, "compute_fatigue")
    kx = real_array(kappa_x, "kappa_x")
    ky = real_array(kappa_y, "kappa_y")
    if kx.ndim != 1 or ky.ndim != 1 or kx.size < 2 or kx.shape != ky.shape:
        raise CableSchemaError(
            "kappa_x and kappa_y must be same-shape 1-D histories with "
            "at least 2 points."
        )
    require_finite(kx, "kappa_x")
    require_finite(ky, "kappa_y")
    if friction_model is None:
        # The default law, with a warning on a non-planar path (once per call).
        friction_model = "direction_tied"
        # stacklevel 3: this function, its material-scope wrapper, the caller.
        _warn_default_direction_tied(
            curvature_path_out_of_plane_ratio(kx, ky), stacklevel=3
        )
    if thermal is not None:
        if contact_model is None:
            contact_model = "solved"
        if contact_model != "solved":
            raise CableSchemaError(
                "thermal coupling needs the solved radial contact state "
                "(contact_model='solved'); the bounded and qp paths have no "
                "radial thermal mismatch"
            )
        if any(
            value is not None
            for value in (
                section_states,
                axial_strain,
                contact_tension_N,
                contact_tension_history_N,
                thermal_stress_Pa,
            )
        ):
            raise CableSchemaError(
                "thermal coupling takes tension_N only; section states, prescribed "
                "strain, a frozen contact tension or a separate thermal stress "
                "conflict with it"
            )
    if friction_law is not None and not isinstance(friction_law, FrictionLaw):
        raise CableSchemaError("friction_law must be a FrictionLaw or None.")
    law = active_friction_law(friction_law)
    law_time = law is not None and law.needs_time
    if thermal is None and time_s is not None and not law_time:
        raise CableSchemaError(
            "time_s is used only with a thermal history or a time-dependent "
            "friction law"
        )
    if law_time:
        try:
            friction_law_time_steps(law, time_s, kx.size)
        except CableInputError as error:
            raise CableSchemaError(str(error)) from error
    law_time_s = time_s if law_time else None
    if contact_model is None:
        contact_model = (
            section_states[0].contact_model
            if section_states is not None and len(section_states) > 0
            else "bounded"
        )
    if contact_model not in {"bounded", "qp", "solved"}:
        raise CableSchemaError("contact_model must be 'bounded', 'qp' or 'solved'.")
    if contact_model == "solved" and section_states is not None:
        raise CableSchemaError("section states belong to the 'bounded'/'qp' models")
    if mu is not None:
        require_finite(mu, "mu")
        if mu < 0.0:
            raise CableSchemaError("mu must be >= 0.")
    tension = None
    strain = None
    if axial_strain is not None:
        if tension_N is not None:
            raise CableSchemaError("prescribe axial strain or tension, not both")
        require_finite(axial_strain, "axial_strain")
        strain = np.asarray(axial_strain, dtype=np.float64)
        if strain.shape != kx.shape:
            raise CableSchemaError(
                "axial_strain must match the curvature-history shape"
            )
    if tension_N is not None:
        require_finite(tension_N, "tension_N")
        tension = np.asarray(tension_N, dtype=np.float64)
        if tension.shape != kx.shape:
            raise CableSchemaError(
                "tension_N must have the same shape as the curvature histories."
            )
        if np.any(tension < 0.0):
            raise CableSchemaError("tension_N must be >= 0.")
    if contact_tension_N is not None:
        require_finite(contact_tension_N, "contact_tension_N")
        if contact_tension_N < 0.0:
            raise CableSchemaError("contact_tension_N must be >= 0.")
    contact_history = None
    if contact_tension_history_N is not None:
        if strain is None or contact_tension_N is not None:
            raise CableSchemaError(
                "contact_tension_history_N sets the contact of a prescribed-strain "
                "record: give it with axial_strain and without contact_tension_N"
            )
        require_finite(contact_tension_history_N, "contact_tension_history_N")
        contact_history = np.asarray(contact_tension_history_N, dtype=np.float64)
        if contact_history.shape != kx.shape:
            raise CableSchemaError(
                "contact_tension_history_N must match the curvature-history shape"
            )
        if np.any(contact_history < 0.0):
            raise CableSchemaError("contact_tension_history_N must be >= 0.")
    for name, history in (thermal_stress_Pa or {}).items():
        require_finite(history, f"thermal_stress_Pa[{name!r}]")

    if section_states is not None:
        if any(
            value is not None
            for value in (
                tension_N,
                axial_strain,
                contact_tension_N,
                contact_tension_history_N,
                thermal_stress_Pa,
            )
        ):
            raise CableSchemaError(
                "supply section states or separate axial/thermal/contact loads, "
                "not both"
            )
        if len(section_states) != kx.size:
            raise CableSchemaError("section states must match the curvature history")
    elif contact_model == "qp" and strain is None:
        if contact_tension_N is not None or thermal_stress_Pa is not None:
            raise CableSchemaError(
                "the coupled normal calculation requires consistent section states; "
                "use solve_section_state for thermal or prescribed contact conditions"
            )
        values = tension if tension is not None else np.zeros_like(kx)
        unique, inverse = np.unique(values, return_inverse=True)
        solved = [
            solve_section_state(cable, tension_N=float(value)) for value in unique
        ]
        section_states = [solved[index] for index in inverse]
    elif contact_model == "qp":
        raise CableSchemaError(
            "prescribed strain alone does not define the coupled radial state; "
            "supply solved section states"
        )

    if section_states is not None:
        if count_from_sample:
            raise CableSchemaError(
                "count_from_sample is not available with section_states"
            )
        return _fatigue_from_section_states(
            cable,
            kx,
            ky,
            section_states,
            contact_model,
            mu,
            n_fibres,
            rainflow_counter,
            friction_model=friction_model,
            distributed_nodes_per_pitch=distributed_nodes_per_pitch,
            distributed_max_increment=distributed_max_increment,
            distributed_bound=distributed_bound,
            copper_contact_model=copper_contact_model,
            partial_slip_cycles=partial_slip_cycles,
            termination=termination,
            progressive=progressive,
            sn_options=(sn_curves, mean_stress, scf),
            friction_law=law,
            time_s=law_time_s,
        )

    reference_tension = (
        contact_tension_N
        if contact_tension_N is not None
        else float(tension[0])
        if tension is not None
        else float(contact_history[0])
        if contact_history is not None
        else 0.0
    )
    layers = _apply_sn_options(
        cable,
        cable_to_fatigue_layers(
            cable,
            contact_tension_N=reference_tension,
            mu=mu,
            contact_model=contact_model,
        ),
        sn_curves,
        mean_stress,
        scf,
    )
    contact_pressure_history = None
    contact_force_history = None
    stick_history = None
    thermal_notes: tuple[str, ...] = ()
    if thermal is not None:
        # time_s serves two consumers: the slow time axis of a ThermalHistory
        # and the friction law. A frozen ThermalState has no time axis, so
        # times given for the law are not passed to it (alone they are still
        # refused by the thermal helper).
        thermal_time_s = (
            None if law_time and isinstance(thermal, ThermalState) else time_s
        )
        inputs = thermal_fatigue_inputs(
            cable, layers, kx.size, tension, thermal, time_s=thermal_time_s
        )
        if (
            inputs.stick_modulus_Pa is not None
            and friction_model == "distributed"
            and any(
                not np.allclose(v, v[0], rtol=1e-12, atol=0.0)
                for v in inputs.stick_modulus_Pa.values()
            )
        ):
            raise CableSchemaError(
                "the distributed friction model needs a constant stick modulus; "
                "the thermal history changes it between blocks (use one "
                "ThermalState per call)"
            )
        contact_pressure_history = inputs.contact_pressure_Pa
        contact_force_history = inputs.contact_force_N
        stick_history = inputs.stick_modulus_Pa
        thermal_stress_Pa = inputs.thermal_stress_Pa
        thermal_notes = thermal_provenance_notes(inputs)
    elif tension is not None and contact_tension_N is None:
        contact_pressure_history, contact_force_history = _contact_histories(
            cable, tension, contact_model
        )
    elif contact_history is not None:
        contact_pressure_history, contact_force_history = _contact_histories(
            cable, contact_history, contact_model
        )
    result = _compute_fatigue_specs(
        layers,
        kx,
        ky,
        tension_N=tension,
        axial_strain=strain,
        EA_full_N=ea_fixed(cable),
        thermal_stress_Pa=thermal_stress_Pa,
        contact_pressure_Pa=contact_pressure_history,
        contact_force_N=contact_force_history,
        stick_modulus_Pa=stick_history,
        n_fibres=n_fibres,
        rainflow_counter=rainflow_counter,
        friction_model=friction_model,
        distributed_nodes_per_pitch=distributed_nodes_per_pitch,
        distributed_max_increment=distributed_max_increment,
        distributed_bound=distributed_bound,
        copper_contact_model=copper_contact_model,
        partial_slip_cycles=partial_slip_cycles,
        termination=termination,
        progressive=progressive,
        count_from_sample=count_from_sample,
        friction_law=law,
        time_s=law_time_s,
    )
    if thermal_notes and result.provenance is not None:
        result = replace(
            result,
            provenance=replace(
                result.provenance,
                limitations=(*result.provenance.limitations, *thermal_notes),
            ),
        )
    return _flag_unevaluated_curves(cable, layers, result)


def _per_layer(value: Any, names: Sequence[str], label: str) -> dict[str, Any]:
    """Expand one option for every layer, or check a per-layer mapping."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        unknown = set(value) - set(names)
        if unknown:
            raise CableSchemaError(
                f"{label} names layers that are not assessed: {sorted(unknown)}"
            )
        return dict(value)
    return dict.fromkeys(names, value)


def _apply_sn_options(
    cable: CableData,
    layers: Sequence[FatigueLayer],
    sn_curves: Mapping[str, str | SNcurve] | None,
    mean_stress: MeanStressOption | Mapping[str, MeanStressOption] | None,
    scf: ComponentSCF | Mapping[str, ComponentSCF] | None,
) -> list[FatigueLayer]:
    """User S-N curve, mean-stress correction and component SCFs per layer.

    With no option the layers are returned unchanged (results bit for bit).
    """
    out = list(layers)
    if sn_curves is None and mean_stress is None and scf is None:
        return out
    names = [layer.name for layer in out]
    if sn_curves is not None and not isinstance(sn_curves, Mapping):
        raise CableSchemaError("sn_curves must map layer names to curves")
    curves = _per_layer(sn_curves, names, "sn_curves")
    corrections = _per_layer(mean_stress, names, "mean_stress")
    factors = _per_layer(scf, names, "scf")
    for index, layer in enumerate(out):
        changes: dict[str, Any] = {}
        if layer.name in curves:
            curve = curves[layer.name]
            changes["sn_bulk"] = (
                get_sn_curve(curve) if isinstance(curve, str) else curve
            )
            if not isinstance(changes["sn_bulk"], SNcurve):
                raise CableSchemaError("sn_curves values must be names or SNcurve")
        if layer.name in corrections:
            option = corrections[layer.name]
            if isinstance(option, str):
                option = MeanStressSpec.from_material(
                    option,  # type: ignore[arg-type]
                    cable.get_layer(layer.name).material,
                )
            if not isinstance(option, MeanStressSpec):
                raise CableSchemaError("mean_stress must be a MeanStressSpec or a name")
            changes["mean_stress"] = option
        if layer.name in factors:
            if not isinstance(factors[layer.name], ComponentSCF):
                raise CableSchemaError("scf must be a ComponentSCF")
            changes["component_scf"] = factors[layer.name]
        if changes:
            out[index] = replace(layer, **changes)
    return out


def _unevaluated_curve_notes(
    cable: CableData, layers: Sequence[FatigueLayer]
) -> tuple[str, ...]:
    """Describe declared ``bulk_sn``/``fretting_sn`` curves that are not used.

    :func:`compute_fatigue` assesses only the structural helical armour
    (:func:`~cablexsec.cable_adapters.structural_armour_layers`). Curves named
    on any other layer (a conductor, a sheath or a non-structural helix) are
    not evaluated. An armour ``fretting_sn`` is evaluated only when the layer
    has a direct crossed-wire contact with a positive Hertz state.
    """
    assessed = {layer.name: layer for layer in layers}
    armour = {layer.name for layer in structural_armour_layers(cable)}
    notes: list[str] = []
    for layer in cable.layers:
        declared = [
            (field_name, value)
            for field_name in ("bulk_sn", "fretting_sn")
            if (value := getattr(layer, field_name, ""))
        ]
        if not declared:
            continue
        if layer.name not in armour or layer.name not in assessed:
            curves = ", ".join(f"{key}={value!r}" for key, value in declared)
            hint = (
                "; sheath fatigue is assessed by compute_shell_fatigue"
                if isinstance(layer, CylindricalShellLayer)
                else "; the stranded conductor is assessed by compute_conductor_fatigue"
                if getattr(layer, "strand_layers", ())
                else ""
            )
            notes.append(
                f"layer {layer.name!r} declares {curves}, but compute_fatigue "
                "assesses only structural helical armour; these curves are not "
                f"evaluated and contribute no damage{hint}"
            )
            continue
        spec = assessed[layer.name]
        contact_evaluated = (
            spec.swt_const is not None and spec.contact_p0_Pa > 0.0
        ) or (spec.sn_fretting is not None and spec.K_t_contact_N_per_m > 0.0)
        fretting_sn = getattr(layer, "fretting_sn", "")
        if fretting_sn and not contact_evaluated:
            notes.append(
                f"layer {layer.name!r} declares fretting_sn={fretting_sn!r}, "
                "but has no direct crossed-wire contact with a positive Hertz "
                "state; the contact-life curve is not evaluated and its damage "
                "is zero"
            )
    return tuple(notes)


def _flag_unevaluated_curves(
    cable: CableData,
    layers: Sequence[FatigueLayer],
    result: FatigueResult,
    *,
    stacklevel: int = 3,
) -> FatigueResult:
    """Warn about, and record in provenance, declared curves left unused."""
    notes = _unevaluated_curve_notes(cable, layers)
    if not notes:
        return result
    for note in notes:
        warnings.warn(note, UnevaluatedFatigueCurveWarning, stacklevel=stacklevel)
    provenance = result.provenance
    if provenance is not None:
        provenance = replace(provenance, limitations=(*provenance.limitations, *notes))
    return replace(result, provenance=provenance)


def _fatigue_from_section_states(
    cable: CableData,
    kx: FloatArray,
    ky: FloatArray,
    states: Sequence[SectionState],
    contact_model: str,
    mu: float | None,
    n_fibres: int,
    rainflow_counter: RainflowCounter = "auto",
    *,
    friction_model: FrictionModel = "direction_tied",
    distributed_nodes_per_pitch: int | Mapping[str, int] | None = None,
    distributed_max_increment: float = DEFAULT_MAX_INCREMENT,
    distributed_bound: DistributedBound = "signed",
    copper_contact_model: CopperContactModel = "slip_cycle",
    partial_slip_cycles: PartialSlipCycles = "skip",
    termination: TerminationSpec | Mapping[str, TerminationSpec] | None = None,
    progressive: ProgressiveFriction | None = None,
    sn_options: tuple[Any, Any, Any] = (None, None, None),
    friction_law: FrictionLaw | None = None,
    time_s: FloatArray | None = None,
) -> FatigueResult:
    """Use each normal equilibrium for wire stress, stiffness and contact.

    The bulk calculation retains the bending-slip history as normal state
    changes. Existing contact-life laws use a constant Hertz contact; varying
    that state requires a variable-normal-load contact formulation, not an
    unchanged stiffness or peak pressure borrowed from the first sample.
    """
    layers_by_state: dict[int, list[FatigueLayer]] = {}
    for state in states:
        state.require_cable(cable)
        if state.contact_model != contact_model:
            raise CableSchemaError("all section states must use the same contact model")
        if id(state) not in layers_by_state:
            layers_by_state[id(state)] = cable_to_fatigue_layers(
                cable,
                contact_tension_N=state.tension_N,
                mu=mu,
                contact_model=contact_model,
                section_state=state,
            )
    initial = _apply_sn_options(cable, layers_by_state[id(states[0])], *sn_options)
    pressure: dict[str, FloatArray] = {}
    force: dict[str, FloatArray] = {}
    normal: dict[str, FloatArray] = {}
    modulus: dict[str, FloatArray] = {}
    stick: dict[str, FloatArray] = {}
    for index, layer in enumerate(initial):
        history = [layers_by_state[id(state)][index] for state in states]
        name = layer.name
        pressure[name] = np.array([item.spec.contact_pressure_Pa for item in history])
        force[name] = np.array(
            [item.contact_line_load_N_per_m or 0 for item in history]
        )
        normal[name] = np.array(
            [state.layers[name].longitudinal_stress_Pa for state in states]
        )
        modulus[name] = np.array([item.spec.E_wire_Pa for item in history])
        stick[name] = np.array([item.spec.k_f for item in history])
        has_contact_life = layer.sn_fretting is not None or layer.swt_const is not None
        if has_contact_life and any(
            item.K_t_contact_N_per_m != layer.K_t_contact_N_per_m
            or item.contact_p0_Pa != layer.contact_p0_Pa
            or item.spec.E_wire_Pa != layer.spec.E_wire_Pa
            for item in history
        ):
            raise CableSchemaError(
                f"{name}: the registered contact-life model requires constant normal "
                "load and contact moduli; use a bulk-only layer for this variable "
                "section-state history"
            )
    result = _compute_fatigue_specs(
        initial,
        kx,
        ky,
        normal_stress_Pa=normal,
        wire_modulus_Pa=modulus,
        stick_modulus_Pa=stick,
        contact_pressure_Pa=pressure,
        contact_force_N=force,
        n_fibres=n_fibres,
        rainflow_counter=rainflow_counter,
        friction_model=friction_model,
        distributed_nodes_per_pitch=distributed_nodes_per_pitch,
        distributed_max_increment=distributed_max_increment,
        distributed_bound=distributed_bound,
        copper_contact_model=copper_contact_model,
        partial_slip_cycles=partial_slip_cycles,
        termination=termination,
        progressive=progressive,
        friction_law=friction_law,
        time_s=time_s,
    )
    # Called from compute_fatigue: point the warning at that caller.
    return _flag_unevaluated_curves(cable, initial, result, stacklevel=4)


def _solved_contact_histories(
    cable: CableData, tension: FloatArray, *, crossing_loads: bool = True
) -> tuple[dict[str, FloatArray], dict[str, FloatArray]]:
    """Solved radial contact over a tension history.

    The solution is piecewise affine in tension. Up to 33 distinct tensions
    are solved exactly; longer histories are interpolated linearly between
    33 equally spaced exact solutions, which is exact except within one grid
    interval of an active-set change.
    """
    histories = solved_contact_histories(cable, tension, crossing_loads=crossing_loads)
    return histories.pressure_Pa, histories.crossing_line_load_N_per_m


def armour_contact_histories(
    cable: CableData,
    tension_N: FloatArray,
    *,
    contact_model: str = "bounded",
) -> tuple[dict[str, FloatArray], dict[str, FloatArray]]:
    """Armour ``P1+P2`` and crossing line-load histories over a tension history.

    The same histories that :func:`compute_fatigue` forms from ``tension_N``
    (no frozen ``contact_tension_N``): per structural armour layer, the
    friction-setting pressure ``P1+P2`` [Pa] and the crossed-wire line load
    [N/m] at every sample, on top of the cable's declared manufacturing
    clamping state (:mod:`cablexsec.clamping`). ``tension_N`` must be finite
    and non-negative; which tension measure loads the armour (true wall
    tension, or effective tension in a flooded annulus) is the caller's
    choice.
    """
    tension = real_array(tension_N, "tension_N").astype(np.float64)
    require_finite(tension, "tension_N")
    if tension.ndim != 1 or np.any(tension < 0.0):
        raise CableSchemaError("tension_N must be a non-negative 1-D history")
    if contact_model not in {"bounded", "qp", "solved"}:
        raise CableSchemaError("contact_model must be 'bounded', 'qp' or 'solved'.")
    return _contact_histories(cable, tension, contact_model)


def _contact_histories(
    cable: CableData,
    tension_N: FloatArray,
    contact_model: str,
    *,
    crossing_loads: bool = True,
) -> tuple[dict[str, FloatArray], dict[str, FloatArray]]:
    """Evaluate contact state over a non-negative tension history.

    ``crossing_loads=False`` returns the pressures only (an empty force map):
    the crossed-wire representation refuses a structural layer with two
    crossing interfaces, which the pressure does not need.

    The bounded equilibrium chain is affine in tension while its non-negative
    contacts remain active, so two endpoint solves give the exact history. The
    QP path is piecewise affine; unique tension states are solved and cached to
    retain exact active-set transitions.
    """
    tension = np.asarray(tension_N, dtype=np.float64)
    if contact_model == "solved":
        return _solved_contact_histories(cable, tension, crossing_loads=crossing_loads)
    if contact_model == "bounded":
        scale = max(float(np.max(tension)), 1.0)
        p0 = armour_contact_pressure(cable, 0.0, contact_model=contact_model)
        p1 = armour_contact_pressure(cable, scale, contact_model=contact_model)
        f0 = (
            armour_crossing_line_loads(cable, 0.0, contact_model=contact_model)
            if crossing_loads
            else {}
        )
        f1 = (
            armour_crossing_line_loads(cable, scale, contact_model=contact_model)
            if crossing_loads
            else {}
        )
        pressure = {
            name: np.asarray(p0[name] + (p1[name] - p0[name]) * tension / scale)
            for name in p0
        }
        force = {
            name: np.asarray(
                f0[name][1] + (f1[name][1] - f0[name][1]) * tension / scale
            )
            for name in f0
        }
        return pressure, force

    unique, inverse = np.unique(tension, return_inverse=True)
    p_values = [
        armour_contact_pressure(cable, float(value), contact_model=contact_model)
        for value in unique
    ]
    f_values = (
        [
            armour_crossing_line_loads(cable, float(value), contact_model=contact_model)
            for value in unique
        ]
        if crossing_loads
        else []
    )
    names = p_values[0].keys() if p_values else ()
    pressure = {
        name: np.asarray([state[name] for state in p_values], dtype=np.float64)[inverse]
        for name in names
    }
    force = {
        name: np.asarray([state[name][1] for state in f_values], dtype=np.float64)[
            inverse
        ]
        for name in (f_values[0] if f_values else ())
    }
    return pressure, force


# --------------------------------------------------------------------------- #
# Fretting regime, wear and knock-down (gap 3.1)                              #
# --------------------------------------------------------------------------- #
def _face_partners(
    cable: CableData, layer_name: str
) -> tuple[
    CylindricalLayer | HelicalLayer | None, CylindricalLayer | HelicalLayer | None
]:
    """Layers touching the inner and outer faces of a helical layer."""
    helix = next(x for x in cable.helical_layers if x.name == layer_name)
    assert helix.r_helix_m is not None and helix.d_wire_m is not None
    r_in = helix.r_helix_m - helix.wire_radial_thickness_m / 2
    r_out = helix.r_helix_m + helix.wire_radial_thickness_m / 2
    inner: CylindricalLayer | HelicalLayer | None = None
    outer: CylindricalLayer | HelicalLayer | None = None
    for other in cable.layers:
        if other.name == layer_name:
            continue
        if isinstance(other, CylindricalLayer):
            if other.r_outer_m is not None and abs(other.r_outer_m - r_in) <= 1e-6:
                inner = other
            if other.r_inner_m is not None and abs(other.r_inner_m - r_out) <= 1e-6:
                outer = other
        elif isinstance(other, HelicalLayer) and other.r_helix_m and other.d_wire_m:
            half = other.wire_radial_thickness_m / 2
            if abs(other.r_helix_m + half - r_in) <= 1e-6:
                inner = other
            if abs(other.r_helix_m - half - r_out) <= 1e-6:
                outer = other
    return inner, outer


def compute_fretting(
    cable: CableData,
    kappa_x: FloatArray,
    kappa_y: FloatArray,
    *,
    duration_s: float,
    tension_N: FloatArray | None = None,
    axial_strain: FloatArray | None = None,
    contact_tension_N: float | None = None,
    mu: float | None = None,
    contact_model: str = "bounded",
    time_fraction: float = 1.0,
    knockdown: FrettingKnockdown | str | None = "auto",
    section_loss: bool = False,
    wear: Mapping[str, WearCoefficient] | None = None,
    partial_slip_wear: bool = False,
    transverse_ratio: float = 0.0,
    transverse_map: TransverseMap | None = None,
) -> CableFrettingResult:
    r"""Fretting regime, energy, wear and fatigue knock-down of the armour.

    Opt-in companion of :func:`compute_fatigue`, which it leaves unchanged.
    Every structural armour layer is assessed on both radial faces: a polymer
    neighbour gives a Hertz line contact on the support stiffness of the
    section law, an opposite-laid armour layer in contact gives the Mindlin
    crossing (:mod:`cablexsec.analysis.fretting_assessment`). Each wire's
    kinematic slip history is rainflow counted and every cycle classified as
    stick, partial slip or gross slip from the transition amplitude with the
    contact's own friction coefficient ([Fouvry1995]); the dissipated energy
    gives the mean wear depth with the material-pair coefficient of
    :mod:`cablexsec.materials.wear` ([Fouvry1996]). With a knock-down the
    contact-fibre stress cycles are counted again with the regime factors.

    Parameters
    ----------
    cable : CableData
        A fully specified cable.
    kappa_x, kappa_y : numpy.ndarray
        Curvature history [1/m].
    duration_s : float
        Duration of the record [s] (results are per year of this loading).
    tension_N, axial_strain : numpy.ndarray or None, optional
        Axial loading for the contact-fibre stress (one of them).
    contact_tension_N : float or None, optional
        Tension of the contact loads; default the mean of ``tension_N`` (or 0).
    mu : float or None, optional
        Friction coefficient of the polymer faces (default: the section law's).
    contact_model : {"bounded", "qp", "solved"}, optional
        Radial contact model of the normal loads.
    time_fraction : float, optional
        Fraction of the year spent in this loading.
    knockdown : FrettingKnockdown, str or None, optional
        ``"auto"`` selects the registered law of the wire material
        (:data:`cablexsec.materials.wear.FRETTING_KNOCKDOWNS`); a key or an
        instance selects one; ``None`` skips the S-N assessment.
    section_loss : bool, optional
        Raise the contact-fibre stress for the wire section worn over the
        cable's design life (faces whose worn body is the wire).
    wear : mapping or None, optional
        Wear coefficients replacing registered ones, keyed by pair
        (for example ``"steel_polyethylene"``).
    partial_slip_wear : bool, optional
        Also wear on partial-slip energy.
    transverse_ratio, transverse_map : optional
        Transverse slip component (as :class:`FatigueLayer`).

    Returns
    -------
    CableFrettingResult
    """
    from cablexsec.analysis.fretting_assessment import (  # noqa: PLC0415
        FaceSpec,
        helical_layer_fretting,
        principal_bending_angle,
    )
    from cablexsec.cable_adapters import (  # noqa: PLC0415
        _bounded_face_pressures,
        winding_interface_pressure,
    )
    from cablexsec.materials import library  # noqa: PLC0415
    from cablexsec.materials.wear import (  # noqa: PLC0415
        get_knockdown,
        get_wear_coefficient,
        wear_pair_for,
    )
    from cablexsec.mechanics.axisymmetric import (  # noqa: PLC0415
        compute_contact_forces,
    )
    from cablexsec.mechanics.hertz_contact import (  # noqa: PLC0415
        layer_crossing_angle_rad,
        layer_crossing_density_per_m,
    )

    require_fully_specified(cable, "compute_fretting")
    flat = [x.name for x in cable.helical_layers if x.wire_shape != "round"]
    if flat:
        raise CableSchemaError(
            "the fretting-regime contact model is for round wires; "
            f"{', '.join(flat)} are rectangular or profiled (gap G1)"
        )
    kx = real_array(kappa_x, "kappa_x")
    ky = real_array(kappa_y, "kappa_y")
    if kx.ndim != 1 or kx.shape != ky.shape or kx.size < 2:
        raise CableSchemaError("kappa_x and kappa_y must be equal 1-D histories")
    require_finite(kx, "kappa_x")
    require_finite(ky, "kappa_y")
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise CableSchemaError("duration_s must be finite and positive")
    if tension_N is not None and axial_strain is not None:
        raise CableSchemaError("prescribe axial strain or tension, not both")
    tension = None if tension_N is None else real_array(tension_N, "tension_N")
    strain = None if axial_strain is None else real_array(axial_strain, "axial_strain")
    for label, history in (("tension_N", tension), ("axial_strain", strain)):
        if history is not None:
            require_finite(history, label)
            if history.shape != kx.shape:
                raise CableSchemaError(f"{label} must match the curvature histories")
    reference = (
        float(contact_tension_N)
        if contact_tension_N is not None
        else (float(np.mean(tension)) if tension is not None else 0.0)
    )
    layers = cable_to_fatigue_layers(
        cable, contact_tension_N=reference, mu=mu, contact_model=contact_model
    )
    p_ext = cable.initial_state.external_pressure_Pa
    chain = compute_contact_forces(cable, reference, p_ext_Pa=p_ext)
    interface = chain.interface_pressure_Pa
    winding = winding_interface_pressure(cable)
    if np.any(winding):
        # P1+P2 contains the winding-stress clamping (armour_contact_pressure);
        # split it between the faces with the same chain, not without it.
        interface = interface + winding
    split = _bounded_face_pressures(cable, interface, p_ext)
    crossings = armour_crossing_line_loads(
        cable, reference, contact_model=contact_model
    )
    overrides = dict(wear or {})
    limitations: list[str] = []
    faces_out: dict[str, FrettingFace] = {}
    for layer in layers:
        helix = next(x for x in cable.helical_layers if x.name == layer.name)
        below, above = split.get(layer.name, (0.0, 0.0))
        total = below + above
        share_in = below / total if total > 0 else 0.5
        p1p2 = layer.spec.contact_pressure_Pa
        nu_wire = library.get_property(helix.material, "nu")
        inner, outer = _face_partners(cable, layer.name)
        specs: list[FaceSpec] = []
        for face_name, angle, partner, pressure in (
            ("inner_face", math.pi, inner, p1p2 * share_in),
            ("outer_face", 0.0, outer, p1p2 * (1.0 - share_in)),
        ):
            if partner is None:
                limitations.append(f"{layer.name}/{face_name}: no touching layer")
                continue
            coefficient: WearCoefficient | None
            try:
                key = wear_pair_for(helix.material, partner.material)
                coefficient = overrides.get(key, get_wear_coefficient(key))
            except CableLookupError:
                coefficient = None
                limitations.append(f"{layer.name}/{face_name}: no wear coefficient")
            e_partner = library.get_property(partner.material, "E_Pa")
            nu_partner = library.get_property(partner.material, "nu")
            if isinstance(partner, HelicalLayer):
                pair = crossings.get(layer.name)
                if pair is None:
                    limitations.append(f"{layer.name}/{face_name}: no crossing load")
                    continue
                assert partner.d_wire_m is not None
                assert partner.lay_angle_rad is not None
                assert helix.lay_angle_rad is not None
                assert partner.n_wires is not None
                assert helix.r_helix_m is not None and partner.r_helix_m is not None
                # Geometric, as in the adapter: the fatigue layer carries a
                # density only when it declares a contact-fatigue law.
                density = layer_crossing_density_per_m(
                    partner.n_wires,
                    helix.r_helix_m,
                    helix.lay_angle_rad,
                    1.0 if helix.lay_direction == "right_hand" else -1.0,
                    partner.r_helix_m,
                    partner.lay_angle_rad,
                    1.0 if partner.lay_direction == "right_hand" else -1.0,
                )
                specs.append(
                    FaceSpec(
                        name=face_name,
                        angle_rad=angle,
                        kind="point",
                        partner=partner.name,
                        partner_E_Pa=e_partner,
                        partner_nu=nu_partner,
                        mu=library.get_property(helix.material, "mu_on_steel"),
                        wear=coefficient,
                        partner_d_wire_m=partner.d_wire_m,
                        crossing_angle_rad=layer_crossing_angle_rad(
                            helix.lay_angle_rad, partner.lay_angle_rad
                        ),
                        crossing_force_N=pair[1] / density,
                        crossing_density_per_m=density,
                    )
                )
            else:
                specs.append(
                    FaceSpec(
                        name=face_name,
                        angle_rad=angle,
                        kind="line",
                        partner=partner.name,
                        partner_E_Pa=e_partner,
                        partner_nu=nu_partner,
                        mu=layer.spec.mu,
                        wear=coefficient,
                        pressure_Pa=pressure,
                    )
                )
        if transverse_ratio > 0 or transverse_map is not None:
            layer = replace(
                layer, transverse_ratio=transverse_ratio, transverse_map=transverse_map
            )
        law: FrettingKnockdown | None
        if knockdown == "auto":
            family = "copper" if "copper" in helix.material.lower() else "steel"
            law = get_knockdown(f"{family}_nishioka_hirakawa")
        elif isinstance(knockdown, str):
            law = get_knockdown(knockdown)
        else:
            law = knockdown
        result = helical_layer_fretting(
            layer,
            specs,
            kx,
            ky,
            duration_s=duration_s,
            nu_wire=nu_wire,
            tension_N=tension,
            EA_full_N=ea_fixed(cable) if tension is not None else None,
            axial_strain=strain,
            time_fraction=time_fraction,
            design_life_years=cable.cable.design_life_years,
            knockdown=law,
            section_loss=section_loss,
            partial_slip_wear=partial_slip_wear,
        )
        for face_name, face in result.items():
            faces_out[f"{layer.name}/{face_name}"] = face
            used = face.contact.wear
            if used is not None and used.assumed:
                limitations.append(
                    f"{layer.name}/{face_name}: wear coefficient {used.pair!r} "
                    "is an assumption"
                )
    provenance = Provenance(
        method=(
            "fretting regime per slip cycle (Mindlin transition amplitude, "
            "Fouvry criteria), Cattaneo/Mindlin dissipated energy, energy wear, "
            "regime-dependent stress-range knock-down at the contact fibre"
        ),
        citations=(
            "[Johnson1985] eqs. 7.29, 7.42, 7.60",
            "[Fouvry1995]",
            "[Fouvry1996]",
            "[VingsboSoderberg1988]",
            "[Nakazawa1994] Eq. 1",
            "[Cruzado2013]",
            "[Poon2023]",
        ),
        limitations=tuple(limitations)
        + (
            f"contact loads at the reference tension {reference:.6g} N; their "
            "variation within the record is not followed",
            "two-dimensional slip paths use the uniaxial energy at delta*",
            "wear depth uses the unworn contact width (no geometry update)",
        ),
    )
    return CableFrettingResult(
        faces=faces_out,
        extreme_fibre_deg=math.degrees(principal_bending_angle(kx, ky)),
        reference_tension_N=reference,
        duration_s=duration_s,
        time_fraction=time_fraction,
        design_life_years=cable.cable.design_life_years,
        provenance=provenance,
    )


# Every public function that takes a cable reads the cable's own materials
# (``materials:`` block) through its material scope, on any thread.
scope_entry_points(globals(), __all__)
