# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Frozen extrusion shrinkage, calibration scope and the clamping warning.

Verification only: the shrink fit of a sleeve on a stiff core against the
Lamé solution with an in-plane eigenstrain, the two radial solvers against
each other, the identity of a zero shrinkage, the refusal of the bounded
chain and the clamping-state and calibration warnings.
"""

import math
import warnings

import numpy as np
import pytest
from pydantic import ValidationError

import cablexsec as cx
from cablexsec.clamping import clamping_state
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.mechanics.concentric_section import (
    ElasticAnnulus,
    solve_concentric_section,
)
from cablexsec.mechanics.core_friction import build_core_friction, core_contact_loads
from cablexsec.mechanics.radial_solution import RadialSection

pytestmark = [
    pytest.mark.analytical_sanity,
    pytest.mark.filterwarnings(
        "ignore::cablexsec.provenance.MissingClampingStateWarning"
    ),
]

SHRINK = cx.ExtrusionShrinkage(hoop_strain=0.005, basis="assumed", source="test")


def _with_shrinkage(cable, name, shrink=SHRINK):
    layers = tuple(
        x.model_copy(update={"extrusion_shrinkage": shrink}) if x.name == name else x
        for x in cable.layers
    )
    return cable.model_copy(update={"layers": layers})


def _lame_shrink_fit(a, b, e, nu, strain):
    """Pressure of a sleeve (a..b) with in-plane eigenstrain -strain on a rigid
    core in plane strain: u(a) = 0, sigma_r(b) = 0."""
    lam = e * nu / ((1 + nu) * (1 - 2 * nu))
    mu = e / (2 * (1 + nu))
    eig = -strain
    amp = (lam + mu) * eig / (lam + mu + mu * a * a / (b * b))
    return -(2 * (lam + mu) * (amp - eig) + 2 * mu * amp)


@pytest.mark.parametrize("solver", ["concentric", "radial"])
def test_shrink_fit_on_a_stiff_core_matches_lame(solver):
    a, b, e, nu, strain = 0.05, 0.056, 1.0e9, 0.4, 0.004
    layers = [
        ElasticAnnulus("core", 0.0, a, 1.0e6 * e, 0.3),
        ElasticAnnulus("sleeve", a, b, e, nu, free_transverse_strain=-strain),
    ]
    if solver == "concentric":
        pressure = float(solve_concentric_section(layers).contact_pressure_Pa[0])
    else:
        result = RadialSection(layers).solve()
        pressure = float(result.interface_line_force_N_per_m[0] / (2 * math.pi * a))
    expected = _lame_shrink_fit(a, b, e, nu, strain)
    assert expected > 0.0
    assert pressure == pytest.approx(expected, rel=1e-4)
    # Thin-ring limit E eps t / ((1 - nu^2) r): within the thickness ratio.
    thin = e * strain * (b - a) / ((1 - nu * nu) * 0.5 * (a + b))
    assert pressure == pytest.approx(thin, rel=0.15)


def test_zero_or_omitted_shrinkage_is_the_same_section():
    cable = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
    zero = cx.ExtrusionShrinkage(hoop_strain=0.0, basis="not_published", source="x")
    other = _with_shrinkage(cable, "hdpe_outer_sheath", zero)
    a = cx.solve_cable_radial_state(cable, 5.0e4)
    b = cx.solve_cable_radial_state(other, 5.0e4)
    assert a.armour_contact_pressure() == b.armour_contact_pressure()
    assert a.result.bundle == b.result.bundle


def test_solved_state_carries_the_shrinkage():
    cable = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
    base = cx.solve_cable_radial_state(cable, 0.0)
    outer = cx.solve_cable_radial_state(
        _with_shrinkage(cable, "hdpe_outer_sheath"), 0.0
    )
    before, after = base.armour_contact_pressure(), outer.armour_contact_pressure()
    for name in ("inner_armour", "outer_armour"):
        assert after[name] > before[name] + 1.0e4
    # The ring around the trefoil (the inner sheath) clamps the cores.
    ring = cx.solve_cable_radial_state(_with_shrinkage(cable, "mdpe_inner_sheath"), 0.0)
    assert ring.result.bundle is not None and base.result.bundle is not None
    assert (
        ring.result.bundle.filler_core_force_N_per_m
        > base.result.bundle.filler_core_force_N_per_m + 1.0e3
    )
    # include_manufacturing=False leaves the shrinkage out.
    off = cx.solve_cable_radial_state(
        _with_shrinkage(cable, "hdpe_outer_sheath"), 0.0, include_manufacturing=False
    )
    np.testing.assert_allclose(
        list(off.armour_contact_pressure().values()),
        list(before.values()),
        rtol=1e-12,
        atol=1e-9,
    )
    # The laid-core law reads the solved, clamped trefoil.
    clamped = _with_shrinkage(cable, "mdpe_inner_sheath")
    loads = core_contact_loads(clamped, 0.0, contact_model="solved")
    assert loads.total_N_per_m > 1.0e3
    # An interstitial filler cannot declare one.
    with pytest.raises(CableSchemaError, match="interstitial filler"):
        cx.solve_cable_radial_state(_with_shrinkage(cable, "mdpe_filler"), 0.0)


def test_qp_route_carries_the_shrinkage():
    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    shrunk = _with_shrinkage(cable, "hdpe_outer_sheath")
    from cablexsec.cable_adapters import armour_contact_pressure

    base = armour_contact_pressure(cable, 1.0e4, contact_model="qp")
    clamped = armour_contact_pressure(shrunk, 1.0e4, contact_model="qp")
    for name in base:
        assert clamped[name] > base[name]
    solved = armour_contact_pressure(shrunk, 1.0e4, contact_model="solved")
    for name in base:
        assert (
            solved[name]
            > armour_contact_pressure(cable, 1.0e4, contact_model="solved")[name]
        )


def test_bounded_chain_refuses_a_shrinkage():
    cable = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
    shrunk = _with_shrinkage(cable, "hdpe_outer_sheath")
    k = 0.1 * np.sin(np.linspace(0.0, 2 * math.pi, 41))
    with pytest.raises(CableInputError, match="extrusion shrinkage"):
        cx.compute_mkappa(shrunk, k, tension_N=5.0e4)
    with pytest.raises(CableInputError, match="contact_model='solved'"):
        core_contact_loads(shrunk, 5.0e4)
    with pytest.raises(CableInputError, match="extrusion shrinkage"):
        cx.solve_section_state(shrunk, tension_N=5.0e4, contact_model="bounded")
    # The solved route runs.
    res = cx.compute_mkappa(shrunk, k, tension_N=5.0e4, contact_model="solved")
    assert np.all(np.isfinite(res.moment_Nm))


def test_schema_of_the_shrinkage_and_scope():
    with pytest.raises(ValidationError, match="source"):
        cx.ExtrusionShrinkage(hoop_strain=0.01, basis="assumed", source=" ")
    with pytest.raises(ValidationError, match="not_published"):
        cx.ExtrusionShrinkage(hoop_strain=0.01, basis="not_published", source="x")
    with pytest.raises(ValidationError, match="physical basis"):
        cx.ExtrusionShrinkage(
            hoop_strain=0.01, basis="published_calibration", source="x"
        )
    for bad in (-0.001, 0.1, math.nan):
        with pytest.raises(ValidationError):
            cx.ExtrusionShrinkage(hoop_strain=bad, basis="assumed", source="x")
    from cablexsec.cables.schema import InitialStateProperties

    with pytest.raises(ValidationError, match="calibration_scope"):
        InitialStateProperties(
            external_pressure_Pa=1e5,
            identification="assumed_bound",
            bounds_Pa=(0.0, 2e5),
            source="x",
            calibration_scope="all_contacts",
        )
    with pytest.raises(ValidationError, match="calibration_scope"):
        cx.ManufacturingClamping(
            armour_winding_stress_Pa=5e7,
            basis="assumed",
            source="x",
            calibration_scope="armour_only",
        )
    cx.ManufacturingClamping(
        armour_winding_stress_Pa=5e7,
        basis="published_calibration",
        source="x",
        calibration_scope="armour_only",
    )


def test_clamping_state_reads_the_shrinkage():
    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    layers = tuple(
        x.model_copy(
            update={
                "manufacturing_clamping": None,
                "winding_stress_Pa": None,
                "pretension_Pa": None,
            }
        )
        if x.layer_type == "helical"
        else x
        for x in cable.layers
    )
    bare = cable.model_copy(update={"layers": layers})
    state = clamping_state(bare)
    assert state.missing_layers == ("inner_armour", "outer_armour")
    message = state.warning_message()
    assert message is not None
    assert "low-tension hysteresis is unreliable" in message
    # A shrinking outer sheath covers both armour layers inside it.
    covered = clamping_state(_with_shrinkage(bare, "hdpe_outer_sheath"))
    assert covered.missing_layers == ()
    assert covered.is_declared and not covered.is_zero
    assert covered.warning_message() is None
    assert "extrusion shrinkage 0.5 %" in covered.describe()
    # A bedding inside both covers neither.
    inner = clamping_state(_with_shrinkage(bare, "hdpe_bedding"))
    assert inner.missing_layers == ("inner_armour", "outer_armour")
    # A not-published zero shrinkage is reported as such.
    zero = cx.ExtrusionShrinkage(hoop_strain=0.0, basis="not_published", source="x")
    unpublished = clamping_state(_with_shrinkage(bare, "hdpe_outer_sheath", zero))
    text = unpublished.warning_message()
    assert text is not None and "hdpe_outer_sheath" in text


def test_calibration_scope_warns_on_double_counted_core_friction():
    from cablexsec.cables.schema import InitialStateProperties

    menard = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    assert menard.initial_state.calibration_scope == "all_contacts"
    witz = cx.load_cable(cx.witz1996_yaml_path(), geometry_override=None)
    assert witz.initial_state.calibration_scope == "armour_only"
    # Fitted with friction on every contact: consistent with the core law.
    with warnings.catch_warnings():
        warnings.simplefilter("error", cx.ClampingCalibrationWarning)
        build_core_friction(menard, cx.CoreFrictionSpec(), 0.0)
    # Fitted with the armour alone: the core law double counts.
    armour_only = menard.model_copy(
        update={
            "initial_state": InitialStateProperties(
                external_pressure_Pa=3.0e5,
                identification="identified",
                source="test",
                calibration_scope="armour_only",
            )
        }
    )
    with pytest.warns(cx.ClampingCalibrationWarning, match="armour friction alone"):
        build_core_friction(armour_only, cx.CoreFrictionSpec(), 0.0)
    # Without the core law nothing is added, so no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error", cx.ClampingCalibrationWarning)
        cx.compute_mkappa(armour_only, 0.05 * np.sin(np.linspace(0, 6.3, 41)), mu=0.2)
    # The same tag on an armour clamping.
    clamp = cx.ManufacturingClamping(
        initial_contact_pressure_Pa=1.0e5,
        basis="published_calibration",
        source="test",
        calibration_scope="armour_only",
    )
    layers = tuple(
        x.model_copy(update={"manufacturing_clamping": clamp})
        if x.layer_type == "helical"
        else x
        for x in menard.layers
    )
    tagged = menard.model_copy(update={"layers": layers})
    with pytest.warns(cx.ClampingCalibrationWarning, match="armour"):
        build_core_friction(tagged, cx.CoreFrictionSpec(), 0.0)


def test_fretting_refuses_a_shrinkage():
    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    shrunk = _with_shrinkage(cable, "hdpe_outer_sheath")
    k = 0.002 * np.sin(np.linspace(0.0, 2 * math.pi, 41))
    with pytest.raises(CableInputError, match="fretting"):
        cx.compute_fretting(
            shrunk, k, np.zeros_like(k), duration_s=10.0, contact_model="solved"
        )


def test_other_bounded_consumers_refuse_a_shrinkage():
    from cablexsec.analysis.capacity import _crush_uf
    from cablexsec.cable_adapters import _normal_force_map

    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    shrunk = _with_shrinkage(cable, "hdpe_outer_sheath")
    with pytest.raises(CableInputError, match="extrusion shrinkage"):
        _normal_force_map(shrunk, 1.0e4, contact_model="bounded")
    assert _normal_force_map(shrunk, 1.0e4, contact_model="qp")
    with pytest.raises(CableInputError, match="crush"):
        _crush_uf(shrunk, 1.0e4, 5.0e6)
    assert math.isnan(_crush_uf(shrunk, 1.0e4, None))


def test_qp_section_state_recovers_the_shrink_fit_stresses():
    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    shrunk = _with_shrinkage(cable, "hdpe_outer_sheath")
    base = cx.solve_section_state(cable, tension_N=1.0e4, contact_model="qp")
    state = cx.solve_section_state(shrunk, tension_N=1.0e4, contact_model="qp")
    # The recovered layer forces balance the section force (checked inside),
    # and the shrunk sheath is in hoop tension that the base sheath is not.
    sheath = state.layers["hdpe_outer_sheath"]
    a = (
        sheath.radial_stress_constant_Pa
        + sheath.radial_stress_inverse_Pa_m2 / 0.0335**2
    )
    plain = base.layers["hdpe_outer_sheath"]
    b = plain.radial_stress_constant_Pa + plain.radial_stress_inverse_Pa_m2 / 0.0335**2
    assert a > b + 1.0e6
