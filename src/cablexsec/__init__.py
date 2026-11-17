# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""CableXSec local mechanics for floating-wind dynamic power cables.

Developed by Jae Hoon Seo, SMI Lab, Department of Naval Architecture and Ocean
Engineering, Inha University.

Public API
----------
Load a cable, then call a ``compute_*`` entry point. Calculation results
identify the selected mechanics, references and applicable scope. A solved
normal section state can be reused in bending, component stress and bulk
fatigue recovery:

>>> import numpy as np
>>> from cablexsec import load_cable, compute_stiffness, menard2023_yaml_path
>>> cable = load_cable(menard2023_yaml_path(), geometry_override=None)
>>> stiff = compute_stiffness(cable)
>>> stiff.ei_min_Nm2 > 0.0
True
>>> print(stiff.provenance.method)
Fixed-radius axial Schur; individual, plane-section and shear-compliant bending

Entry points: :func:`load_cable`, :func:`compute_stiffness`,
:func:`solve_section_state`,
:func:`compute_mkappa`, :func:`compute_thermal`, :func:`compute_capacity`,
:func:`compression_screen`, :func:`compute_compression_capacity`,
:func:`build_capacity_envelope`, :func:`check_capacity`, :func:`compute_coiling`,
:func:`compute_fatigue`, :func:`compute_electrical_life`,
:func:`compute_component_response`, :func:`compute_section_stress`,
:func:`compute_shell_stress`,
:func:`simulate_rig_test`, :func:`simulate_bending_test`,
:func:`torque_balance`,
:func:`design_torque_balance`. See
:mod:`cablexsec.api` for the full surface.
"""

from __future__ import annotations

from cablexsec.analysis.beam_test import (
    AlongWireSection,
    AlongWireSpec,
    BendingProtocol,
    BendingTestResult,
    Cantilever,
    FourPoint,
    SectionLaw,
    SectionResponse,
    ThreePoint,
    simulate_bending_test,
)
from cablexsec.analysis.bending_law import (
    BendingLawExport,
    IwanBendingLaw,
    SlipScaleTable,
    export_bending_law,
)
from cablexsec.analysis.capacity_check import (
    CapacityCheckResult,
    CapacityEnvelope,
    GumbelFit,
    build_capacity_envelope,
    capacity_extremes,
    check_capacity,
    check_capacity_series,
    gumbel_fit,
)
from cablexsec.analysis.coiling import (
    CoilingLayerResult,
    CoilingResult,
    compute_coiling,
)
from cablexsec.analysis.component_limits import (
    COMPONENT_UTILISATION_FACTORS,
    component_factor_for,
    component_limits,
)
from cablexsec.analysis.components import (
    ComponentUsageCurve,
    bending_directions,
    capacity_utilisation,
    component_usage_curve,
    own_bending_ratio,
)
from cablexsec.analysis.core_screen import (
    CoreScreenResult,
    compute_core_screen_fatigue,
)
from cablexsec.analysis.critical_plane import SWTCycleResult, compute_swt_cycle
from cablexsec.analysis.kinematics import (
    HelixField,
    KinematicHistory,
    kinematic_history,
    kinematic_slip_amplitude,
)
from cablexsec.analysis.long_term import (
    FatigueLoadCase,
    LongTermFatigueResult,
    compute_long_term_fatigue,
)
from cablexsec.analysis.neutral_axis import (
    ArmourAxes,
    NeutralAxisResult,
    compute_neutral_axes,
)
from cablexsec.analysis.rig_test import (
    GaugeLocation,
    RigProtocol,
    RigSetup,
    RigTestResult,
    simulate_rig_test,
)
from cablexsec.analysis.sea_state_batch import (
    CapacityBatchCheck,
    SeaStateBatchOptions,
    SeaStateRun,
    SectionRequest,
    run_sea_state_batch,
)
from cablexsec.analysis.section_properties import (
    LoadSharingResult,
    MassProperties,
    SectionLoadCase,
    SectionPropertiesResult,
    compute_load_sharing,
    compute_mass_properties,
    compute_section_properties,
)
from cablexsec.analysis.section_stress import (
    LayerStressField,
    SectionStressField,
    armour_stress_fields,
    compute_section_stress,
)
from cablexsec.analysis.sheath_strain import (
    CoreLayerStrainResult,
    SheathStrainLifeResult,
    compute_core_layer_strain,
)
from cablexsec.analysis.sheave import (
    SheaveContact,
    SheaveResult,
    compute_sheave,
    groove_pressure,
    sheave_contact,
)
from cablexsec.analysis.slip_curvatures import (
    LayerSlipCurvatures,
    armour_slip_curvatures,
)
from cablexsec.analysis.sn_assessment import ComponentSCF
from cablexsec.analysis.tape import TapeResult, compute_tape_fatigue, tape_geometry
from cablexsec.analysis.thermal_coupling import (
    ArmourThermalState,
    ThermalFatigueResult,
    ThermalHistory,
    ThermalState,
    armour_thermal_state,
    compute_thermal_fatigue,
)
from cablexsec.api import (
    ARMOUR_UTILISATION_FACTORS,
    BirdcageDetail,
    BlockedSNCurveError,
    CableData,
    CableFrettingResult,
    CableRadialState,
    CableSchemaError,
    CableXSecError,
    CapacityResult,
    ComponentResponse,
    ComponentState,
    CompressionCapacityResult,
    CompressionScreenResult,
    ConcentricSectionResult,
    ConductorFatigueResult,
    ContactQPResult,
    CoreFrictionHistory,
    CoreFrictionSpec,
    CriticalDistanceSpec,
    ElasticAnnulus,
    ElasticHelicalRing,
    ElectricalLifeResult,
    ExtrapolationWarning,
    FatigueLayer,
    FatigueResult,
    FrictionLaw,
    MkappaResult,
    PhysicalRangeError,
    ProgressiveFriction,
    Provenance,
    SectionState,
    ShellFatigueResult,
    ShellStressResult,
    StiffnessResult,
    StressResult,
    TapeLayer,
    TerminationSpec,
    ThermalResult,
    TorqueBalanceDesign,
    TorqueBalanceResult,
    UnevaluatedFatigueCurveWarning,
    UnvalidatedModelWarning,
    ValidatedRange,
    armour_contact_histories,
    birdcage_buckling_detail,
    cable_to_fatigue_layers,
    cable_to_mkappa_specs,
    cigre_tb880_yaml_path,
    compression_screen,
    compute_capacity,
    compute_component_response,
    compute_compression_capacity,
    compute_conductor_fatigue,
    compute_electrical_life,
    compute_fatigue,
    compute_fretting,
    compute_mkappa,
    compute_shell_fatigue,
    compute_shell_stress,
    compute_stiffness,
    compute_stress,
    compute_thermal,
    design_torque_balance,
    export_three_core_yaml_path,
    iea_task49_66kv_300_yaml_path,
    iea_task49_66kv_stranded,
    iea_task49_66kv_yaml_path,
    inter_array_single_core_yaml_path,
    inter_array_three_core_yaml_path,
    load_cable,
    menard2023_yaml_path,
    saevik2011_yaml_path,
    solve_cable_radial_state,
    solve_concentric_section,
    solve_contact_qp,
    solve_section_state,
    torque_balance,
    witz1996_yaml_path,
)
from cablexsec.cables import chae2023_yaml_path, thies2016_yaml_path
from cablexsec.cables.schema import (
    FacePressure,
    ManufacturingClamping,
    TapeWinding,
    scope_entry_points,
)
from cablexsec.clamping import ClampingState, LayerClamping, clamping_state
from cablexsec.coupling import (
    COUPLING_API_VERSION,
    XsecStiffnessResult,
    xsec_stiffness,
)
from cablexsec.exceptions import CableInputError, CableLookupError, CableSolverError
from cablexsec.io import (
    LoadCaseSeries,
    fatigue_from_series,
    jonswap_series,
    read_cabledyn_sections,
    read_riflex_ascii,
    read_series_csv,
    read_time_history_csv,
    write_series_csv,
)
from cablexsec.materials.library import register_material, unregister_material
from cablexsec.materials.mean_stress import MEAN_STRESS_METHODS, MeanStressSpec
from cablexsec.materials.sn_curves import (
    SNcurve,
    get_sn_curve,
    list_sn_curves,
    register_sn_curve,
    unregister_sn_curve,
)
from cablexsec.materials.strain_life import (
    StrainLifeConstants,
    register_strain_life,
    strain_life_damage,
)
from cablexsec.mechanics.along_wire import WireChainLayer, along_wire_history
from cablexsec.mechanics.bend_restrictor import (
    ArticulatedRestrictor,
    BendStiffener,
    CurvatureField,
    restrictor_curvature_field,
    stiffener_curvature_field,
)
from cablexsec.mechanics.core_kinematics import (
    CORE_SLIP_STATES,
    CoreKinematics,
    core_kinematics,
)
from cablexsec.mechanics.hertz_field import (
    HertzFieldStress,
    compute_hertz_subsurface_stress,
)
from cablexsec.mechanics.hertz_stress import (
    HertzAxisStress,
    compute_hertz_axis_stress,
)
from cablexsec.mechanics.tendon_bending import (
    LateralRelief,
    TransverseSlipModel,
    curvature_coefficients,
    dong_anchor_bending,
    dong_anchor_tension,
)
from cablexsec.mechanics.tendon_section import WireSection
from cablexsec.mechanics.thermal_network import ThermalNetwork, thermal_network
from cablexsec.provenance import (
    DirectionTiedFrictionWarning,
    FrictionModelFallbackWarning,
    MissingClampingStateWarning,
    UnmodelledPolymerSofteningWarning,
)

__version__ = "0.1.0"

from cablexsec.project import (  # noqa: E402
    PROJECT_FORMAT_VERSION,
    AnalysisRecord,
    AnalysisRequest,
    AnalysisRunner,
    BendingSettings,
    CableModel,
    FatigueSettings,
    Project,
    StiffnessSettings,
    StressSettings,
)

__all__ = [
    "BendingLawExport",
    "IwanBendingLaw",
    "SlipScaleTable",
    "export_bending_law",
    "LayerSlipCurvatures",
    "armour_slip_curvatures",
    "LoadSharingResult",
    "MassProperties",
    "SectionLoadCase",
    "SectionPropertiesResult",
    "compute_load_sharing",
    "compute_mass_properties",
    "compute_section_properties",
    "SheaveContact",
    "SheaveResult",
    "compute_sheave",
    "groove_pressure",
    "sheave_contact",
    "ArmourAxes",
    "NeutralAxisResult",
    "compute_neutral_axes",
    "HelixField",
    "KinematicHistory",
    "kinematic_history",
    "kinematic_slip_amplitude",
    "PROJECT_FORMAT_VERSION",
    "AnalysisRecord",
    "AnalysisRequest",
    "AnalysisRunner",
    "BendingSettings",
    "CableModel",
    "FatigueSettings",
    "Project",
    "StiffnessSettings",
    "StressSettings",
    "SWTCycleResult",
    "compute_swt_cycle",
    "HertzAxisStress",
    "compute_hertz_axis_stress",
    "HertzFieldStress",
    "compute_hertz_subsurface_stress",
    "__version__",
    # loading
    "load_cable",
    "CableData",
    # manufacturing clamping state (gap 4.1b)
    "ManufacturingClamping",
    "ClampingState",
    "LayerClamping",
    "clamping_state",
    "cigre_tb880_yaml_path",
    "menard2023_yaml_path",
    "inter_array_single_core_yaml_path",
    "inter_array_three_core_yaml_path",
    "export_three_core_yaml_path",
    "iea_task49_66kv_300_yaml_path",
    "iea_task49_66kv_stranded",
    "iea_task49_66kv_yaml_path",
    "saevik2011_yaml_path",
    "thies2016_yaml_path",
    "witz1996_yaml_path",
    # power-core screens, core slip state and sheath strain life (P1, P6, P8)
    "CORE_SLIP_STATES",
    "CoreKinematics",
    "core_kinematics",
    "compute_core_layer_strain",
    "CoreLayerStrainResult",
    "compute_core_screen_fatigue",
    "CoreScreenResult",
    "FacePressure",
    "StrainLifeConstants",
    "register_strain_life",
    "strain_life_damage",
    "SheathStrainLifeResult",
    "register_material",
    "unregister_material",
    # user S-N curves, mean-stress corrections and component SCFs
    "SNcurve",
    "get_sn_curve",
    "list_sn_curves",
    "register_sn_curve",
    "unregister_sn_curve",
    "MeanStressSpec",
    "MEAN_STRESS_METHODS",
    "ComponentSCF",
    # capacity check of global time series
    "CapacityEnvelope",
    "CapacityCheckResult",
    "CapacityBatchCheck",
    "GumbelFit",
    "build_capacity_envelope",
    "check_capacity",
    "check_capacity_series",
    "capacity_extremes",
    "gumbel_fit",
    # coiling into a static tank
    "CoilingLayerResult",
    "CoilingResult",
    "compute_coiling",
    "chae2023_yaml_path",
    # bend restrictor / stiffener fields, tapes, copper plasticity (P5, P2, P7)
    "ArticulatedRestrictor",
    "BendStiffener",
    "CurvatureField",
    "restrictor_curvature_field",
    "stiffener_curvature_field",
    "TapeWinding",
    "TapeResult",
    "compute_tape_fatigue",
    "tape_geometry",
    # compute entry points
    "compute_stiffness",
    "compute_mkappa",
    "compute_thermal",
    "compute_capacity",
    "compression_screen",
    "compute_compression_capacity",
    "ARMOUR_UTILISATION_FACTORS",
    "TapeLayer",
    "compute_component_response",
    # component limits (0.2 % proof stress, allowable strain) and capacity
    "COMPONENT_UTILISATION_FACTORS",
    "ComponentUsageCurve",
    "bending_directions",
    "capacity_utilisation",
    "component_factor_for",
    "component_limits",
    "component_usage_curve",
    "own_bending_ratio",
    "compute_section_stress",
    "armour_stress_fields",
    "SectionStressField",
    "LayerStressField",
    "compute_fatigue",
    "compute_long_term_fatigue",
    "FatigueLoadCase",
    "LongTermFatigueResult",
    # rig-test simulation and gauge planning
    "simulate_rig_test",
    "RigProtocol",
    "RigSetup",
    "GaugeLocation",
    "RigTestResult",
    # bending-test arrangements (gap T1)
    "simulate_bending_test",
    "BendingProtocol",
    "BendingTestResult",
    "SectionLaw",
    "SectionResponse",
    "ThreePoint",
    "FourPoint",
    "Cantilever",
    # slip redistribution along the wires (gap T2)
    "AlongWireSpec",
    "AlongWireSection",
    "WireChainLayer",
    "along_wire_history",
    "LateralRelief",
    "TransverseSlipModel",
    "WireSection",
    "curvature_coefficients",
    "dong_anchor_bending",
    "dong_anchor_tension",
    # energised analyses: temperature histories
    "ThermalHistory",
    "ThermalState",
    "ThermalNetwork",
    "thermal_network",
    "compute_thermal_fatigue",
    "ThermalFatigueResult",
    "armour_thermal_state",
    "ArmourThermalState",
    "compute_electrical_life",
    "compute_stress",
    "compute_shell_stress",
    "compute_shell_fatigue",
    "compute_conductor_fatigue",
    "compute_fretting",
    "armour_contact_histories",
    "CableFrettingResult",
    # torque balance and twist under tension (gap 3.5)
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
    "CoreFrictionHistory",
    "CoreFrictionSpec",
    "FrictionLaw",
    "ProgressiveFriction",
    "birdcage_buckling_detail",
    "cable_to_mkappa_specs",
    "cable_to_fatigue_layers",
    # coupling (CableDyn / global-solver surface, L6.3)
    "xsec_stiffness",
    "XsecStiffnessResult",
    "COUPLING_API_VERSION",
    # global-analysis I/O (L6.1–L6.2)
    "LoadCaseSeries",
    "jonswap_series",
    "read_time_history_csv",
    "read_riflex_ascii",
    "read_series_csv",
    "write_series_csv",
    "fatigue_from_series",
    # global-local coupling, stage 1 (CableDyn runs to annual damage)
    "read_cabledyn_sections",
    "run_sea_state_batch",
    "SeaStateRun",
    "SectionRequest",
    "SeaStateBatchOptions",
    # result objects
    "StiffnessResult",
    "MkappaResult",
    "StressResult",
    "ThermalResult",
    "CapacityResult",
    "CompressionScreenResult",
    "CompressionCapacityResult",
    "ComponentResponse",
    "ComponentState",
    "BirdcageDetail",
    "FatigueResult",
    "FatigueLayer",
    "TerminationSpec",
    "ElectricalLifeResult",
    "ShellStressResult",
    "ShellFatigueResult",
    "ConductorFatigueResult",
    "CriticalDistanceSpec",
    # provenance + exceptions
    "Provenance",
    "ValidatedRange",
    "ExtrapolationWarning",
    "DirectionTiedFrictionWarning",
    "FrictionModelFallbackWarning",
    "UnevaluatedFatigueCurveWarning",
    "UnvalidatedModelWarning",
    "MissingClampingStateWarning",
    "UnmodelledPolymerSofteningWarning",
    "CableXSecError",
    "CableInputError",
    "CableLookupError",
    "CableSolverError",
    "CableSchemaError",
    "BlockedSNCurveError",
    "PhysicalRangeError",
]

# Public functions that take a cable read its own materials in its material
# scope; the entry points of cablexsec.api are wrapped there already.
scope_entry_points(globals(), __all__)
