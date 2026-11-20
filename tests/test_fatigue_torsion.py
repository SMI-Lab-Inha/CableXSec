# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Twist-rate wire stress of the fatigue path (scorecard row TQ-FT).

The wire strain of a helical layer under the cable strain ``eps`` and twist
rate ``tau`` with plane sections and fixed radii is
``eps cos^2(a) + s R tau sin(a) cos(a)`` (Knapp 1981, Eqs. 2 and 11 with a
radially rigid core). Targets were registered before the runs: closed form
and torque/force consistency to 1e-12, mirror lay and zero twist exact.
"""

from __future__ import annotations

import math
import warnings

import check_torque_balance as tq
import numpy as np
import pytest
from test_cabledyn_reader import _bending_case, _write_case

import cablexsec as cx
from cablexsec.analysis.conductor_fatigue import (
    assess_conductor_fatigue,
    ring_twist_strain_per_rate,
)
from cablexsec.analysis.fatigue import (
    compute_fatigue as compute_fatigue_specs,
)
from cablexsec.analysis.fatigue import (
    twist_stress_coefficient,
    wire_stress_field,
)
from cablexsec.analysis.sea_state_batch import (
    SeaStateBatchOptions,
    SeaStateRun,
    SectionRequest,
    run_sea_state_batch,
)
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.io.cabledyn import read_cabledyn_sections, section_channels
from cablexsec.mechanics.stiffness import axial_torsional_matrix
from cablexsec.mechanics.strand_slip import ConductorRing
from cablexsec.provenance import TorsionalStressWarning


@pytest.fixture(scope="module")
def cable():
    return cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)


@pytest.fixture(scope="module")
def checks():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tq.fatigue_torsion_checks()


def _history(n: int = 301) -> tuple[np.ndarray, ...]:
    t = np.linspace(0.0, 30.0, n)
    kx = 0.03 * np.sin(2 * math.pi * t / 7.0)
    ky = 0.01 * np.cos(2 * math.pi * t / 5.0)
    tension = 50.0e3 + 8.0e3 * np.sin(2 * math.pi * t / 4.0)
    tau = 8.0e-4 + 3.0e-4 * np.sin(2 * math.pi * t / 6.0)
    return t, kx, ky, tension, tau


# --------------------------------------------------------------------------- #
# Registered verification (row TQ-FT)                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.analytical_sanity
def test_pure_twist_stress_and_damage_follow_the_closed_form(checks) -> None:
    assert checks["closed_form"] < 1e-12
    assert checks["damage"] < 1e-12


@pytest.mark.analytical_sanity
def test_wire_stresses_carry_the_section_torque_and_tension(checks) -> None:
    assert checks["torque"] < 1e-12
    assert checks["force"] < 1e-12


@pytest.mark.analytical_sanity
def test_mirror_lay_with_reversed_twist_and_zero_twist(checks) -> None:
    assert checks["mirror"] < 1e-12
    assert checks["zero"] == 0.0


# --------------------------------------------------------------------------- #
# Signs, routes and the default                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.analytical_sanity
def test_right_hand_layer_tightens_under_positive_twist(cable) -> None:
    """S (left-hand) and Z (right-hand) layers take opposite twist stresses."""
    layers = {fl.name: fl for fl in cx.cable_to_fatigue_layers(cable)}
    inner, outer = layers["inner_armour"], layers["outer_armour"]
    assert outer.lay_hand == 1 and inner.lay_hand == -1
    assert twist_stress_coefficient(outer) > 0 > twist_stress_coefficient(inner)
    a = outer.spec.lay_angle_rad
    expected = outer.spec.E_wire_Pa * outer.spec.r_helix_m * math.sin(a) * math.cos(a)
    assert twist_stress_coefficient(outer) == pytest.approx(expected, rel=1e-15)
    m = axial_torsional_matrix(cable)
    held = twist_stress_coefficient(
        outer, axial_twist_coupling_Nm=m.K_et, EA_full_N=m.K_ee
    )
    assert held == pytest.approx(
        expected - outer.spec.E_wire_Pa * math.cos(a) ** 2 * m.K_et / m.K_ee,
        rel=1e-15,
    )


def test_axial_strain_route_holds_the_strain(cable) -> None:
    t, kx, ky, _tension, tau = _history(61)
    layer = cx.cable_to_fatigue_layers(cable)[1]
    strain = np.full_like(t, 1.0e-4)
    base = wire_stress_field(layer, kx, ky, axial_strain=strain)
    twisted = wire_stress_field(
        layer, kx, ky, axial_strain=strain, twist_rate_per_m=tau
    )
    added = twisted.mean_Pa - base.mean_Pa
    assert np.allclose(
        added, twist_stress_coefficient(layer) * tau[None, :], rtol=1e-12, atol=1e-6
    )
    assert np.array_equal(twisted.radial_Pa, base.radial_Pa)


def test_twist_raises_armour_damage_of_the_tightened_layer(cable) -> None:
    t, kx, ky, tension, tau = _history()
    options = {"tension_N": tension, "friction_model": "direction_tied"}
    plain = cx.compute_fatigue(cable, kx, ky, **options)
    # A twist history in phase with the tension adds range to the layer that
    # it tightens, which also carries the tension stress.
    swing = 2.0e-3 * (tension - tension.mean()) / 8.0e3
    twisted = cx.compute_fatigue(cable, kx, ky, twist_rate_per_m=swing, **options)
    assert twisted.D_bulk_per_wire["outer_armour"].max() > (
        plain.D_bulk_per_wire["outer_armour"].max()
    )
    assert twisted.provenance is not None
    assert "[Knapp1981] Eqs. 2, 11" in twisted.provenance.citations
    assert any("twist-rate stress" in x for x in twisted.provenance.limitations)
    assert plain.provenance is not None
    assert not any("twist-rate" in x for x in plain.provenance.limitations)


def test_default_is_unchanged_bit_for_bit(cable) -> None:
    t, kx, ky, tension, _tau = _history()
    options = {"tension_N": tension, "friction_model": "direction_tied"}
    a = cx.compute_fatigue(cable, kx, ky, **options)
    b = cx.compute_fatigue(cable, kx, ky, twist_rate_per_m=None, **options)
    c = cx.compute_fatigue(cable, kx, ky, twist_rate_per_m=np.zeros_like(t), **options)
    for name, values in a.D_bulk_per_fibre.items():
        assert np.array_equal(values, b.D_bulk_per_fibre[name])
        assert np.array_equal(values, c.D_bulk_per_fibre[name])


@pytest.mark.parametrize("model", ["distributed", "progressive"])
def test_twist_enters_every_friction_model_as_axial_stress(cable, model) -> None:
    """The twist stress is uniform over a layer: the slip state is unchanged,
    so the twisted field differs from the untwisted one by the axial term."""
    t, kx, ky, tension, tau = _history(41)
    layer = cx.cable_to_fatigue_layers(cable)[0]
    m = axial_torsional_matrix(cable)
    options = {"friction_model": model}
    base = wire_stress_field(layer, kx, ky, tension, m.K_ee, **options)
    twisted = wire_stress_field(
        layer,
        kx,
        ky,
        tension,
        m.K_ee,
        twist_rate_per_m=tau,
        axial_twist_coupling_Nm=m.K_et,
        **options,
    )
    coefficient = twist_stress_coefficient(
        layer, axial_twist_coupling_Nm=m.K_et, EA_full_N=m.K_ee
    )
    assert np.allclose(
        twisted.mean_Pa - base.mean_Pa, coefficient * tau[None, :], atol=1e-5
    )


def test_twist_inputs_are_checked(cable) -> None:
    t, kx, ky, tension, tau = _history(41)
    layers = cx.cable_to_fatigue_layers(cable)
    m = axial_torsional_matrix(cable)
    with pytest.raises(CableInputError, match="axial_twist_coupling_Nm"):
        compute_fatigue_specs(
            layers, kx, ky, tension_N=tension, EA_full_N=m.K_ee, twist_rate_per_m=tau
        )
    with pytest.raises(CableInputError, match="omit axial_twist_coupling_Nm"):
        compute_fatigue_specs(
            layers,
            kx,
            ky,
            axial_strain=np.full_like(t, 1e-4),
            twist_rate_per_m=tau,
            axial_twist_coupling_Nm=m.K_et,
        )
    with pytest.raises(CableInputError, match="only with twist_rate_per_m"):
        compute_fatigue_specs(layers, kx, ky, axial_twist_coupling_Nm=m.K_et)
    with pytest.raises(CableInputError, match="match the curvature"):
        compute_fatigue_specs(layers, kx, ky, twist_rate_per_m=tau[:-1])
    normal = {layer.name: np.full_like(t, 1e8) for layer in layers}
    with pytest.raises(CableInputError, match="solved normal stress"):
        compute_fatigue_specs(
            layers, kx, ky, normal_stress_Pa=normal, twist_rate_per_m=tau
        )
    with pytest.raises(CableSchemaError, match="qp contact model"):
        cx.compute_fatigue(
            cable, kx, ky, tension_N=tension, contact_model="qp", twist_rate_per_m=tau
        )
    with pytest.raises(CableSchemaError, match="match the curvature"):
        cx.compute_fatigue(cable, kx, ky, tension_N=tension, twist_rate_per_m=tau[:3])
    with pytest.raises(CableInputError, match="must be finite"):
        twist_stress_coefficient(
            layers[0], axial_twist_coupling_Nm=math.inf, EA_full_N=m.K_ee
        )
    with pytest.raises(CableInputError, match="requires finite positive EA_full_N"):
        twist_stress_coefficient(layers[0], axial_twist_coupling_Nm=m.K_et)
    rings = (ConductorRing.from_pitch(6, 2.7e-3, 3.3e-3, 0.1196, 1.0),)
    conductor = {
        "sn_bulk": cx.get_sn_curve("nasution_2014_copper"),
        "swt_const": None,
        "mu_wire": 0.2,
    }
    with pytest.raises(CableInputError, match="finite axis_strain_per_twist_m"):
        assess_conductor_fatigue(
            rings, 1.0e11, 0.34, kx, ky, twist_rate_per_m=tau, **conductor
        )
    with pytest.raises(CableInputError, match="only with twist_rate_per_m"):
        assess_conductor_fatigue(
            rings, 1.0e11, 0.34, kx, ky, axis_strain_per_twist_m=1e-3, **conductor
        )
    with pytest.raises(CableInputError, match="match the curvature"):
        assess_conductor_fatigue(
            rings,
            1.0e11,
            0.34,
            kx,
            ky,
            twist_rate_per_m=tau[:4],
            axis_strain_per_twist_m=1e-3,
            **conductor,
        )


# --------------------------------------------------------------------------- #
# Conductor                                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.analytical_sanity
def test_conductor_ring_twist_strain() -> None:
    ring = ConductorRing.from_pitch(24, 2.7e-3, 0.0133, 0.3588, -1.0)
    a = ring.lay_angle_rad
    axis = 7.2e-3
    expected = math.cos(a) ** 2 * axis - 0.0133 * math.sin(a) * math.cos(a)
    assert ring_twist_strain_per_rate(ring, axis) == pytest.approx(expected, rel=1e-15)
    centre = ConductorRing.from_pitch(1, 2.7e-3, 0.0, None, 1.0)
    assert ring_twist_strain_per_rate(centre, axis) == axis


def test_conductor_takes_the_twist_strain(cable) -> None:
    t, kx, ky, tension, _tau = _history(201)
    conductor = cx.iea_task49_66kv_stranded()
    plain = cx.compute_conductor_fatigue(conductor, kx, ky, tension_N=tension)
    zero = cx.compute_conductor_fatigue(
        conductor, kx, ky, tension_N=tension, twist_rate_per_m=np.zeros_like(t)
    )
    assert np.array_equal(plain.D_bulk_per_orientation, zero.D_bulk_per_orientation)
    # A pure twist cycle without bending or tension change: every ring's
    # stress range is E * compaction * |strain per rate| * range of tau.
    still = np.zeros_like(t)
    tau = 2.0e-3 * np.sin(2 * math.pi * t / 6.0)
    twisted = cx.compute_conductor_fatigue(
        conductor,
        still,
        still,
        tension_N=np.full_like(t, 50e3),
        twist_rate_per_m=tau,
    )
    layer = next(x for x in conductor.layers if getattr(x, "strand_layers", ()))
    m = axial_torsional_matrix(conductor)
    from cablexsec.mechanics.core_lay import core_angle  # noqa: PLC0415

    axis = (layer.r_offset_m or 0.0) * math.tan(core_angle(conductor, layer)) - (
        m.K_et / m.K_ee
    )
    modulus = cx.materials.library.get_property(layer.material, "E_eff_axial_Pa")
    for i, ring in enumerate(layer.strand_layers):
        rings = ConductorRing.from_pitch(
            ring.n_wires,
            ring.d_wire_m,
            ring.r_helix_m,
            ring.pitch_m,
            1.0 if ring.lay_direction == "right_hand" else -1.0,
            ring.wire_area_m2,
            ring.compaction_scf,
        )
        expected = (
            modulus
            * ring.compaction_scf
            * abs(ring_twist_strain_per_rate(rings, axis))
            * (tau.max() - tau.min())
        )
        assert twisted.stress_range_max_Pa[i] == pytest.approx(expected, rel=1e-9)
    with pytest.raises(CableSchemaError, match="match the curvature"):
        cx.compute_conductor_fatigue(
            conductor, kx, ky, tension_N=tension, twist_rate_per_m=tau[:5]
        )


# --------------------------------------------------------------------------- #
# CableDyn reader and the sea-state batch                                     #
# --------------------------------------------------------------------------- #
def _twisted_case(tmp_path, nodes, *, stiffness, rates=(1.0e-3, 3.0e-3)):
    """Twist channels of two element rates around node 21 (s = 2.0 m)."""
    nt = 31
    t = np.arange(nt) * 0.1
    positions = _bending_case(0.03 + 0.01 * np.sin(t), 0.2 * np.sin(t), nt)
    arcs = np.arange(101) * 0.1
    swing = 1.0 + 0.5 * np.sin(2 * math.pi * t / 1.5)
    channels = {}
    for j in nodes:
        s = arcs[j - 1]
        # Twist [deg] from End A: rate rates[0] up to node 21, rates[1] after.
        twist = rates[0] * min(s, 2.0) + rates[1] * max(s - 2.0, 0.0)
        channels[f"Twist1N{j}"] = np.degrees(twist * swing)
    return _write_case(
        tmp_path,
        positions,
        tension=np.full((nt, 100), 30000.0),
        channels=channels,
        deck_kwargs={
            "ea": stiffness.ea_fixed_N,
            "ei": stiffness.ei_min_Nm2,
            "torsion": True,
        },
    ), swing


@pytest.mark.analytical_sanity
def test_reader_forms_the_twist_rate(tmp_path, cable) -> None:
    stiffness = cx.compute_stiffness(cable)
    (root, deck), swing = _twisted_case(tmp_path, (20, 21, 22), stiffness=stiffness)
    with warnings.catch_warnings():
        warnings.simplefilter("error", TorsionalStressWarning)
        read = read_cabledyn_sections(
            root,
            deck,
            {"on_node": 2.0, "inside": 2.05, "before": 1.95},
            pressure="none",
            hang_off_rotation="none",
        )
    h = read.histories
    # On a node: the mean of the two element rates; inside an element: its own.
    assert np.allclose(h["on_node"].twist_rate_per_m, 2.0e-3 * swing, rtol=1e-12)
    assert np.allclose(h["inside"].twist_rate_per_m, 3.0e-3 * swing, rtol=1e-12)
    assert np.allclose(h["before"].twist_rate_per_m, 1.0e-3 * swing, rtol=1e-12)
    assert read.checks["twist_rate"]["on_node"] == ["N20-N21", "N21-N22"]
    window = h["on_node"].window(1.0)
    assert window.twist_rate_per_m is not None
    assert window.twist_rate_per_m.size == window.time_s.size
    assert "twist_rate_per_m" in h["on_node"].to_columns(0.03)
    assert {"Twist1N20", "Twist1N21", "Twist1N22"} <= set(
        section_channels(deck, {"on_node": 2.0}, positions=False)
    )


def test_reader_names_missing_twist_channels(tmp_path, cable) -> None:
    stiffness = cx.compute_stiffness(cable)
    (root, deck), _ = _twisted_case(tmp_path, (21, 22), stiffness=stiffness)
    with pytest.warns(TorsionalStressWarning, match="needs Twist1N20"):
        read = read_cabledyn_sections(
            root, deck, {"a": 2.0}, pressure="none", hang_off_rotation="none"
        )
    assert read.histories["a"].twist_rate_per_m is None
    assert read.histories["a"].twist_rad is not None


def test_batch_applies_or_announces_the_twist_stress(tmp_path, cable) -> None:
    stiffness = cx.compute_stiffness(cable)
    (root, deck), _ = _twisted_case(tmp_path, (20, 21, 22), stiffness=stiffness)
    run = SeaStateRun("r", root, deck, 1.0, hang_off_rotation="none")
    sections = [SectionRequest("a", 2.0)]
    base = {"tension_basis": "true", "allow_property_mismatch": True}
    with pytest.warns(TorsionalStressWarning, match="twist_stress is False"):
        off = run_sea_state_batch(
            cable, [run], sections, options=SeaStateBatchOptions(**base)
        )
    on = run_sea_state_batch(
        cable,
        [run],
        sections,
        options=SeaStateBatchOptions(twist_stress=True, **base),
    )
    assert off.manifest["inputs_sha256"] != on.manifest["inputs_sha256"]
    d_off = off.sections["a"].long_term.annual_damage_per_fibre
    d_on = on.sections["a"].long_term.annual_damage_per_fibre
    assert any(not np.array_equal(d_off[k], d_on[k]) for k in d_off)
    (tmp_path / "b").mkdir()
    (root2, deck2), _ = _twisted_case(tmp_path / "b", (21, 22), stiffness=stiffness)
    with (
        pytest.raises(CableSchemaError, match="twist_stress needs"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        run_sea_state_batch(
            cable,
            [SeaStateRun("r", root2, deck2, 1.0, hang_off_rotation="none")],
            sections,
            options=SeaStateBatchOptions(twist_stress=True, **base),
        )
