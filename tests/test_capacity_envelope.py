# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Capacity envelopes and published section properties beside their definitions.

An envelope of the components at zero curvature only scans the tension and
compression capacities: its level curves are passed at zero curvature or
never, and at zero curvature it gives the utilisation of the full envelope.
An envelope with component level curves interpolates between them as the
wire-stress levels do. The published properties of a cable are set beside
the computed value of the same definition at the stated tension, with a
reference case where no load case has that tension.
"""

from __future__ import annotations

import dataclasses
import math
import warnings

import numpy as np
import pytest
from _bundled import bundled
from test_cabledyn_inputs import _case

import cablexsec as cx
from cablexsec.analysis.capacity_check import (
    CapacityEnvelope,
    build_capacity_envelope,
    check_capacity_series,
)
from cablexsec.analysis.section_properties import (
    PublishedComparison,
    SectionLoadCase,
    compare_published,
    compute_mass_properties,
    compute_section_properties,
    tangent_bending_stiffness,
)
from cablexsec.cables.schema import (
    AggregateProperties,
    CableData,
    CableMeta,
    CylindricalLayer,
)
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.io.cabledyn import read_cabledyn_sections
from cablexsec.mechanics.core_friction import CoreFrictionSpec, build_core_friction

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def menard() -> CableData:
    return bundled("menard2023")


@pytest.fixture(scope="module")
def cigre() -> CableData:
    return bundled("poradowski2025_220kv")


# -- capacity envelopes ------------------------------------------------------------


def test_zero_curvature_envelope_scans_the_tension_capacity(menard) -> None:
    grid = [0.0, 1.0e5]
    zero = build_capacity_envelope(menard, grid, curvatures=False)
    full = build_capacity_envelope(menard, grid)
    # Every level is passed at zero curvature or not at all.
    levels = zero.kappa_levels_1_m
    assert np.all((levels == 0.0) | np.isinf(levels))
    np.testing.assert_array_equal(zero.component_kappa_1_m, [0.0])
    tension = np.array([0.0, 2.5e4, 5.0e4, 1.0e5])
    straight = np.zeros(tension.size)
    u_zero, _, _ = zero.utilisation(tension, straight)
    u_full, _, _ = full.utilisation(tension, straight)
    np.testing.assert_allclose(u_zero, u_full, rtol=1e-9, atol=1e-12)
    # A bent sample lies beyond the tabulated curvatures and says so.
    _, _, counts = zero.utilisation(tension, np.full(tension.size, 0.1))
    assert counts["extrapolated_beyond_levels"] == tension.size


def test_zero_curvature_envelope_leaves_out_the_loose_screens() -> None:
    iea = bundled("iea_task49_66kv")
    zero = build_capacity_envelope(iea, [0.0, 3.0e5, 6.0e5], curvatures=False)
    assert zero.kinematic_layers == ("conductor_screen", "cu_tape_screen")
    # At zero curvature the utilisation only rises with the tension.
    tension = np.linspace(0.0, 6.0e5, 7)
    usage, _, _ = zero.utilisation(tension, np.zeros(tension.size))
    assert np.all(np.diff(usage) > 0.0)


def test_component_factor_must_be_a_utilisation_factor(menard) -> None:
    for factor in (0.0, 1.2, math.nan):
        with pytest.raises(CableInputError, match="component_utilisation_factor"):
            build_capacity_envelope(menard, [0.0], component_utilisation_factor=factor)


def test_component_level_curves_interpolate_like_the_wire_levels() -> None:
    envelope = CapacityEnvelope(
        tension_N=np.array([0.0, 1.0e5]),
        kappa_capacity_1_m=np.array([0.3, 0.1]),
        tension_utilisation=np.zeros(2),
        utilisation_factor=1.0,
        tension_kind="effective",
        levels=np.array([0.5, 1.0]),
        kappa_levels_1_m=np.array([[0.2, 0.1], [0.4, 0.2]]),
        zero_curvature_utilisation=np.zeros(2),
        lateral_utilisation=np.zeros(2),
        first_slip_curvature_1_m=np.full(2, np.inf),
        component_levels_1_m=np.array([[0.1, 0.4], [0.3, 0.8]]),
        component_zero_utilisation=np.zeros(2),
    )
    assert envelope.has_levels and envelope.has_components
    kappa = np.array([0.05, 0.1, 0.3, 0.5])
    usage, governing, counts = envelope.utilisation(np.zeros(4), kappa)
    # Components: linear from zero to the first level, between the levels, and
    # beyond the last with the slope of the last two (2.5 per 1/m).
    np.testing.assert_allclose(usage, [0.25, 0.5, 1.0, 1.5])
    np.testing.assert_array_equal(governing, [3, 3, 3, 3])
    assert counts["extrapolated_beyond_levels"] == 1
    # At the second tension the wire levels govern below the component levels.
    usage, governing, _ = envelope.utilisation(np.full(1, 1.0e5), np.array([0.15]))
    np.testing.assert_allclose(usage, [0.75])
    np.testing.assert_array_equal(governing, [0])
    # Halfway in tension the level curvatures are interpolated.
    usage, governing, _ = envelope.utilisation(np.full(1, 0.5e5), np.array([0.15]))
    np.testing.assert_allclose(usage, [0.5])
    np.testing.assert_array_equal(governing, [0])


def test_series_torque_is_carried_to_the_check(tmp_path) -> None:
    root, deck = _case(tmp_path, nt=6)
    history = read_cabledyn_sections(root, deck, {"mid": 5.0}).histories["mid"]
    envelope = CapacityEnvelope.from_points([0.0, 1e5], [0.5, 0.3], source="test")
    plain = check_capacity_series(envelope, history)
    torque = 40.0 * np.cos(np.asarray(history.time_s))
    twisted = check_capacity_series(
        envelope, dataclasses.replace(history, torque_Nm=torque)
    )
    assert twisted.torque_max_Nm == pytest.approx(float(np.max(np.abs(torque))))
    # Torque is reported, not part of the curvature utilisation.
    assert twisted.max_utilisation == plain.max_utilisation


# -- published section properties ------------------------------------------------


def _published(cable: CableData, **values: float | None) -> CableData:
    return cable.model_copy(
        update={"aggregate_properties": AggregateProperties(**values)}
    )


def test_published_value_at_a_tension_no_case_has(cigre) -> None:
    published = _published(
        cigre,
        EI_min_kNm2=50.0,
        EI_min_tension_kN=60.0,
        GJ_kNm2=500.0,
        GJ_tension_kN=40.0,
    )
    case = SectionLoadCase("A", 40e3)
    result = compute_section_properties(
        published, [case], published_tolerance=0.1, n_points=8
    )
    rows = {row.key: row for row in result.published}
    # The bending stiffness is stated at 60 kN: a reference case at 60 kN.
    reference = compute_section_properties(
        cigre, [SectionLoadCase("B", 60e3)], n_points=8, published_tolerance=None
    ).cases[0]
    slip = rows["bending_full_slip"]
    assert slip.computed == pytest.approx(reference.ei_slip_Nm2, rel=1e-12)
    assert slip.condition == "at 60 kN" and "at 60 kN" in slip.counterpart
    assert slip.difference == pytest.approx(slip.computed / 50e3 - 1.0)
    # The torsion is stated at the case tension, where the solved section does
    # not accept the cable: the lay contribution at fixed radius.
    torsion = rows["torsional_stiffness"]
    assert result.cases[0].route == "fixed_radius"
    assert torsion.computed == result.gj_lay_Nm2
    assert "does not accept the cable at 40 kN" in torsion.counterpart
    assert torsion.flagged == (abs(torsion.difference) > 0.1)


def test_published_value_without_a_case(cigre) -> None:
    result = compute_section_properties(
        cigre, [SectionLoadCase("A", 40e3)], n_points=8, published_tolerance=None
    )
    assert result.published == ()
    stated = _published(
        cigre,
        EI_min_kNm2=50.0,
        EI_min_tension_kN=70.0,
        GJ_kNm2=500.0,
        GJ_tension_kN=70.0,
    )
    rows = {row.key: row for row in compare_published(stated, result)}
    # No case at 70 kN and no way to compute one: the bounds, said so.
    assert rows["bending_full_slip"].computed == result.ei_full_slip_Nm2
    assert "(no case at 70 kN)" in rows["bending_full_slip"].counterpart
    assert rows["torsional_stiffness"].computed == result.gj_lay_Nm2
    assert "(no case at 70 kN)" in rows["torsional_stiffness"].counterpart
    unstated = _published(cigre, EI_min_kNm2=50.0, GJ_kNm2=500.0)
    rows = {
        row.key: row
        for row in compare_published(unstated, result, case_at=lambda tension: None)
    }
    assert rows["bending_full_slip"].computed == result.ei_full_slip_Nm2
    assert rows["bending_full_slip"].condition == ""
    assert rows["torsional_stiffness"].counterpart == (
        "lay contribution only, fixed radius"
    )
    for tolerance in (0.0, -0.1, math.inf):
        with pytest.raises(CableInputError, match="tolerance"):
            compare_published(stated, result, tolerance=tolerance)
        if tolerance != math.inf:
            with pytest.raises(CableInputError, match="tolerance"):
                compute_section_properties(
                    cigre, [SectionLoadCase("A", 40e3)], published_tolerance=tolerance
                )


def test_a_missing_computed_value_is_not_flagged() -> None:
    row = PublishedComparison(
        "mass", "Mass per length", 50.0, math.nan, "kg/m", "", "dry mass", 0.1
    )
    assert math.isnan(row.difference) and not row.flagged
    hit = PublishedComparison(
        "mass", "Mass per length", 50.0, 60.0, "kg/m", "at 0 kN", "dry mass", 0.1
    )
    assert hit.difference == pytest.approx(0.2) and hit.flagged
    assert "published 50 kg/m at 0 kN; computed 60 kg/m" in hit.describe()


def test_progress_and_pressure_basis_of_the_fixed_radius_route(cigre) -> None:
    seen: list[tuple[float, str]] = []
    result = compute_section_properties(
        cigre,
        [
            SectionLoadCase("dry", 40e3),
            SectionLoadCase("deep", 40e3, external_pressure_Pa=1.0e6),
        ],
        n_points=8,
        progress=lambda fraction, message: seen.append((fraction, message)),
    )
    fractions = [fraction for fraction, _ in seen]
    assert fractions[-1] == 1.0 and fractions == sorted(fractions)
    assert any("deep" in message for _, message in seen)
    dry, deep = result.cases
    assert dry.wall_tension_basis == "the effective tension (no external pressure)"
    # Flooded and fixed radius: the pressure gives no clamping, the assembly
    # takes the effective tension.
    assert deep.route == "fixed_radius"
    assert "flooded annulus" in deep.wall_tension_basis
    assert any("no clamping of the armour" in note for note in deep.notes)
    assert deep.axial_stiffness_N == dry.axial_stiffness_N


def test_section_properties_need_a_structural_armour() -> None:
    polymer = CableData(
        cable=CableMeta(name="polymer rod"),
        layers=(
            CylindricalLayer(
                name="rod", material="hdpe", r_inner_m=0.0, r_outer_m=0.02
            ),
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(CableSchemaError, match="no structural armour layer"):
            compute_section_properties(polymer, [SectionLoadCase("A", 1e3)])


def test_weight_in_air_and_backbone_shape(cigre) -> None:
    mass = compute_mass_properties(cigre, gravity_m_s2=9.81)
    assert mass.weight_in_air_N_per_m == pytest.approx(9.81 * mass.mass_kg_per_m)
    with pytest.raises(CableInputError, match="matching curvature and moment"):
        tangent_bending_stiffness(np.array([0.0, 1.0]), np.array([0.0]))
    with pytest.raises(CableInputError, match="matching curvature and moment"):
        tangent_bending_stiffness(np.array([0.0]), np.array([0.0]))


def test_core_slip_takes_the_contact_of_the_case_state() -> None:
    cable = bundled("export_three_core_220kv")
    result = compute_section_properties(
        cable,
        [
            SectionLoadCase("hot", 20e3, temperature_C=40.0),
            SectionLoadCase("reference", 20e3),
        ],
        core_friction=True,
        n_points=8,
        published_tolerance=None,
    )
    hot, reference = result.cases
    assert hot.route == reference.route == "fixed_radius"
    assert hot.cores is not None and reference.cores is not None
    spec = CoreFrictionSpec()
    state = cx.solve_section_state(
        cable, tension_N=20e3, temperature_C=40.0, contact_model="bounded"
    )
    # The core contact line load of the bounded chain at the case temperature,
    # the same contact model as the armour.
    warm = build_core_friction(
        cable, spec, 20e3, contact_model="bounded", section_state=state
    )
    cold = build_core_friction(cable, spec, 20e3, contact_model="bounded")
    assert hot.cores.contact_line_load_N_per_m == pytest.approx(
        warm.loads.total_N_per_m, rel=1e-12
    )
    assert reference.cores.contact_line_load_N_per_m == pytest.approx(
        cold.loads.total_N_per_m, rel=1e-12
    )
    assert hot.cores.onset_curvature_1_m < hot.cores.full_slip_curvature_1_m
