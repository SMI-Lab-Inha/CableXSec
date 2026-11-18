# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Neutral axes under a curvature history: closed forms and independent checks."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.neutral_axis import angle_from, neutral_axes, sign_changes
from cablexsec.cable_adapters import cable_to_mkappa_specs, structural_armour_layers
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.papailiou import (
    slip_onset_curvature,
    wire_slip_history,
    wire_slip_stress_bounds,
)
from cablexsec.mechanics.section_conventions import rotation_to_strain_curvature
from cablexsec.mechanics.stiffness import ea_fixed

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def cable() -> cx.CableData:
    """The public 33 kV three core: onset 0.044 and 0.012 1/m (clamped)."""
    path = Path(cx.__file__).parent / "cables" / "inter_array_three_core_33kv.yaml"
    return cx.load_cable(path, geometry_override=None)


def _layers(cable: cx.CableData, mu: float | None = None) -> list[tuple[str, object]]:
    specs = cable_to_mkappa_specs(cable, 0.0, mu=mu)
    names = [layer.name for layer in structural_armour_layers(cable)]
    return list(zip(names, specs, strict=True))


def _rotating(
    kappa: float, revolutions: int = 3, per_revolution: int = 2000, sense: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    turn = np.linspace(0.0, 2 * math.pi * revolutions, revolutions * per_revolution + 1)
    return turn, kappa * np.cos(sense * turn), kappa * np.sin(sense * turn)


def _last_revolution(turn: np.ndarray, revolutions: int = 3) -> np.ndarray:
    return turn > 2 * math.pi * (revolutions - 1)


# -- helpers -----------------------------------------------------------------------


def test_angle_from_wraps_to_a_half_open_turn() -> None:
    reference = np.array([0.0, 0.0, 0.0, 3.0, np.nan])
    angle = np.array([math.pi, -math.pi, 3 * math.pi / 2, -3.0, 1.0])
    out = angle_from(reference, angle)
    np.testing.assert_allclose(
        out[:4], [math.pi, math.pi, -math.pi / 2, -6.0 + 2 * math.pi]
    )
    assert math.isnan(out[4])


def test_sign_changes_round_a_ring() -> None:
    stress = np.array(
        [
            [1.0, 1.0, 0.0, 1.0],
            [1.0, -1.0, 0.0, -1.0],
            [-1.0, 1.0, 0.0, 0.0],
            [-1.0, -1.0, 0.0, 1.0],
        ]
    )
    # Column 3 passes over the zero of wire 2: + - (0) + is two changes.
    np.testing.assert_array_equal(sign_changes(stress), [2, 4, 0, 2])


def test_the_cap_is_kept_on_request(cable: cx.CableData) -> None:
    spec = cable_to_mkappa_specs(cable, 0.0)[1]
    turn, kx, ky = _rotating(3 * slip_onset_curvature(spec), revolutions=1)
    plain = wire_slip_history(spec, kx, ky)
    kept = wire_slip_history(spec, kx, ky, keep_cap=True)
    assert plain.cap_Pa is None and kept.cap_Pa is not None
    np.testing.assert_array_equal(plain.stress_Pa, kept.stress_Pa)
    np.testing.assert_array_equal(plain.moment_Nm, kept.moment_Nm)
    assert np.all(np.abs(kept.stress_Pa) <= kept.cap_Pa * (1 + 1e-12))
    # Above the hold threshold the cap is the bound of the current direction
    # on the side of the wire's stress: upper for a nonnegative stress, lower
    # (signed capstan) for a negative one.
    index = 777
    upper, lower = wire_slip_stress_bounds(spec, math.atan2(ky[index], kx[index]))
    np.testing.assert_allclose(
        kept.cap_Pa[:, index],
        np.where(kept.stress_Pa[:, index] < 0.0, lower, upper),
        rtol=1e-12,
    )
    assert np.any(kept.stress_Pa[:, index] < 0.0)
    assert np.any(kept.stress_Pa[:, index] > 0.0)
    # The slipping count of the return map is the wires at their cap that
    # moved; a wire below its cap is never counted.
    below = np.abs(kept.stress_Pa) < kept.cap_Pa * (1 - 1e-12)
    assert np.all(np.count_nonzero(~below, axis=0) >= plain.slipping_count)


# -- uniaxial bending --------------------------------------------------------------


def test_planar_loading_keeps_every_axis_in_the_plane(cable: cx.CableData) -> None:
    kappa = np.linspace(0.0, 0.1, 401)
    result = cx.compute_neutral_axes(
        cable, kappa, np.zeros_like(kappa), direction_share=0.0
    )
    theta = result.bending_direction_rad
    assert math.isnan(theta[0])  # no direction at zero curvature
    np.testing.assert_array_equal(theta[1:], 0.0)
    np.testing.assert_allclose(result.moment_direction_rad[1:], 0.0, atol=1e-12)
    for layer, (_, spec) in zip(result.layers, _layers(cable), strict=True):
        np.testing.assert_allclose(result.stress_angle_rad(layer)[1:], 0.0, atol=1e-9)
        stuck = layer.stuck
        assert stuck is not None
        # Symmetric about the plane of bending: wire k and its mirror image.
        phi = np.mod(layer.phi_wires_rad, 2 * math.pi)
        mirror = [int(np.argmin(np.abs(np.mod(-p, 2 * math.pi) - phi))) for p in phi]
        np.testing.assert_array_equal(stuck, stuck[mirror])
        arc = layer.stick_arc_rad
        assert arc is not None
        assert arc[1] == pytest.approx(2 * math.pi)  # all in stick at first
        assert np.all(np.diff(arc) <= 1e-12)  # the stick zone only shrinks
        # Slip starts at the neutral axis: at 1.2 times the onset curvature
        # the wires near +-90 deg slip and those at the extreme fibres stick.
        onset = slip_onset_curvature(spec)  # type: ignore[arg-type]
        assert arc[int(np.searchsorted(kappa, 0.99 * onset))] == pytest.approx(
            2 * math.pi
        )
        partial = stuck[:, int(np.searchsorted(kappa, 1.2 * onset))]
        near_axis = np.abs(np.cos(phi)) < 0.2
        near_fibre = np.abs(np.cos(phi)) > 0.98
        assert not np.any(partial[near_axis]) and np.all(partial[near_fibre])
        assert arc[-1] == 0.0  # beyond full slip
        np.testing.assert_array_equal(layer.sign_changes[1:], 2)


def test_planar_cycles_flip_the_direction(cable: cx.CableData) -> None:
    kappa = 0.08 * np.sin(np.linspace(0.0, 4 * math.pi, 801))
    result = cx.compute_neutral_axes(cable, kappa, np.zeros_like(kappa))
    theta = result.bending_direction_rad
    defined = np.isfinite(theta)
    np.testing.assert_allclose(
        np.abs(theta[defined]), np.where(kappa[defined] > 0, 0.0, math.pi)
    )
    angle = np.abs(result.moment_angle_rad)
    angle = angle[np.isfinite(angle)]
    # The moment is along the curvature or opposite it (after a reversal).
    assert np.all(np.isclose(angle, 0.0, atol=1e-9) | np.isclose(angle, math.pi))
    assert np.any(np.isclose(angle, math.pi))


# -- rotating curvature of constant magnitude --------------------------------------


def _independent_lead(spec: object, kappa: float, per_revolution: int = 20000) -> float:
    """Lead of the stress resultant over the bending direction of one wire
    population under steady anticlockwise rotation, from a scalar
    spring-slider written out here: the signed Papailiou bounds
    -sigma0 (1 - exp(-m psi)) <= sigma <= sigma0 (exp(m psi) - 1),
    psi = asin|cos(beta)|, m = mu |sin a|, and the elastic-Coulomb update.
    Every wire sees the same history shifted in time, so the steady stress
    S(beta) of one wire gives the distribution round a continuous layer."""
    a = spec.lay_angle_rad  # type: ignore[attr-defined]
    r = spec.r_helix_m  # type: ignore[attr-defined]
    n = spec.n_wires  # type: ignore[attr-defined]
    area = spec.A_wire_m2  # type: ignore[attr-defined]
    pressure = spec.contact_pressure_Pa  # type: ignore[attr-defined]
    sigma0 = (
        r * pressure * (2 * math.pi * r / n) * math.cos(a) / (area * math.sin(a) ** 2)
    )
    m = spec.mu * abs(math.sin(a))  # type: ignore[attr-defined]
    k = spec.k_f  # type: ignore[attr-defined]
    turn = np.linspace(0.0, 6 * math.pi, 3 * per_revolution + 1)
    drive = r * kappa * np.cos(turn)  # the wire at azimuth zero
    psi = np.arcsin(np.abs(np.cos(turn)))
    upper = sigma0 * (np.exp(m * psi) - 1)
    lower = sigma0 * (1 - np.exp(-m * psi))
    stress = np.empty_like(turn)
    force = 0.0
    for i in range(turn.size):
        trial = force + k * (drive[i] - (drive[i - 1] if i else 0.0))
        force = max(-lower[i], min(upper[i], trial))
        stress[i] = force
    last = slice(2 * per_revolution, 3 * per_revolution)
    beta = -turn[last]  # the wire's azimuth from the bending direction
    s = stress[last]
    return math.atan2(float(np.sum(s * np.sin(beta))), float(np.sum(s * np.cos(beta))))


def test_rotation_in_partial_slip_against_an_independent_calculation(
    cable: cx.CableData,
) -> None:
    layers = _layers(cable)
    name, spec = layers[1]
    kappa = 3 * slip_onset_curvature(spec)  # type: ignore[arg-type]
    turn, kx, ky = _rotating(kappa)
    result = neutral_axes(layers, kx, ky)  # type: ignore[arg-type]
    last = _last_revolution(turn)
    outer = result.layers[1]
    lead = float(np.mean(result.stress_angle_rad(outer)[last]))
    expected = _independent_lead(spec, kappa)
    assert math.degrees(lead) == pytest.approx(math.degrees(expected), abs=0.1)
    # The stress axis leads (11.06 deg). The signed bound lets the compression
    # side slip earlier than the former symmetric cap (10 deg), so at three
    # times the onset more of the ring slips and the lead is larger.
    assert 10.5 < math.degrees(lead) < 11.5
    # Energy: the work per revolution, 2 pi |b| |m| sin(lead), is the friction
    # dissipation of the same history (plastic work of the slip increments).
    history = wire_slip_history(spec, kx, ky)  # type: ignore[arg-type]
    start = int(np.searchsorted(turn, 4 * math.pi))
    friction = history.work.friction_J_per_m
    dissipated = friction[-1] - friction[start]
    moment = float(np.mean(np.hypot(*outer.moment_Nm[last].T)))
    assert math.asin(dissipated / (2 * math.pi * kappa * moment)) == pytest.approx(
        lead, abs=math.radians(0.01)
    )
    # The inner layer is still in stick at this curvature: no lead.
    np.testing.assert_allclose(
        result.stress_angle_rad(result.layers[0]), 0.0, atol=1e-9
    )
    assert name == "outer_armour"


def test_rotation_reversed_reverses_the_lead(cable: cx.CableData) -> None:
    layers = _layers(cable)
    kappa = 3 * slip_onset_curvature(layers[1][1])  # type: ignore[arg-type]
    turn, kx, ky = _rotating(kappa, sense=-1.0)
    result = neutral_axes(layers, kx, ky)  # type: ignore[arg-type]
    last = _last_revolution(turn)
    lead = np.mean(result.stress_angle_rad(result.layers[1])[last])
    assert math.degrees(lead) == pytest.approx(
        -math.degrees(_independent_lead(layers[1][1], kappa)), abs=0.1
    )
    assert np.all(result.moment_angle_rad[last] < 0.0)


def test_full_slip_moment_is_normal_to_the_curvature(cable: cx.CableData) -> None:
    """Closed form: every wire at the bound of the sign of its strain rate,
    sigma_k = sigma0 (e^{m psi} - 1) for sin beta_k > 0 and
    -sigma0 (1 - e^{-m psi}) for sin beta_k < 0 (signed capstan), whose
    resultant is normal to the curvature (Kimball 1925): the in-plane part
    cancels between beta and pi - beta. Opposite wires (beta, beta + pi) add
    up to the reversal span 2 sigma0 sinh(m psi), and with psi = pi/2 - beta
    on [0, pi/2] the continuous-layer magnitude is
    M = A cos(a) r sigma0 (n/2pi) 4 int_0^{pi/2} sinh(m psi) cos(psi) dpsi
      = A cos(a) r sigma0 (n/2pi) 4 (sinh(m pi/2) - m)/(1 + m^2)."""
    layers = _layers(cable)
    name, spec = layers[1]
    kappa = 1e4 * slip_onset_curvature(spec)  # type: ignore[arg-type]
    turn, kx, ky = _rotating(kappa, per_revolution=4000)
    result = neutral_axes(layers, kx, ky)  # type: ignore[arg-type]
    last = _last_revolution(turn)
    layer = result.layers[1]
    lead = np.degrees(result.stress_angle_rad(layer)[last])
    assert np.mean(lead) == pytest.approx(90.0, abs=4.0)
    assert np.all((lead > 80.0) & (lead <= 90.0 + 1e-9))
    a, r, n = spec.lay_angle_rad, spec.r_helix_m, spec.n_wires  # type: ignore[attr-defined]
    sigma0 = (
        r
        * spec.contact_pressure_Pa
        * (2 * math.pi * r / n)
        * math.cos(a)
        / (  # type: ignore[attr-defined]
            spec.A_wire_m2 * math.sin(a) ** 2  # type: ignore[attr-defined]
        )
    )
    m = spec.mu * abs(math.sin(a))  # type: ignore[attr-defined]
    closed = (
        spec.A_wire_m2  # type: ignore[attr-defined]
        * math.cos(a)
        * r
        * sigma0
        * (n / (2 * math.pi))
        * 4
        * (math.sinh(m * math.pi / 2) - m)
        / (1 + m * m)
    )
    direction = np.column_stack((kx, ky))[last] / kappa
    normal = np.column_stack((-direction[:, 1], direction[:, 0]))
    across = np.sum(layer.moment_Nm[last] * normal, axis=1)
    assert np.mean(across) == pytest.approx(closed, rel=0.01)
    assert name == "outer_armour"


def test_no_lag_in_stick_or_without_friction(cable: cx.CableData) -> None:
    layers = _layers(cable)
    onset = slip_onset_curvature(layers[0][1])  # type: ignore[arg-type]
    turn, kx, ky = _rotating(0.5 * min(onset, slip_onset_curvature(layers[1][1])))  # type: ignore[arg-type]
    stick = neutral_axes(layers, kx, ky, ei_min_Nm2=1e3)  # type: ignore[arg-type]
    np.testing.assert_allclose(stick.moment_angle_rad[1:], 0.0, atol=1e-9)
    for layer in stick.layers:
        np.testing.assert_allclose(stick.stress_angle_rad(layer)[1:], 0.0, atol=1e-9)
        assert layer.stick_arc_rad is not None
        np.testing.assert_allclose(layer.stick_arc_rad[1:], 2 * math.pi)
    # Zero friction: no friction stress, so no stress axis and no stick zone;
    # the moment is EI_min times the curvature, along it.
    frictionless = _layers(cable, mu=0.0)
    turn, kx, ky = _rotating(0.05)
    free = neutral_axes(frictionless, kx, ky, ei_min_Nm2=4.4e3)  # type: ignore[arg-type]
    np.testing.assert_allclose(free.moment_angle_rad, 0.0, atol=1e-12)
    np.testing.assert_allclose(free.moment_Nm, 4.4e3 * np.column_stack((kx, ky)))
    for layer in free.layers:
        assert np.all(np.isnan(layer.stress_direction_rad))
        assert not np.any(layer.stuck)


# -- strain neutral axis and conventions -------------------------------------------


def test_strain_axis_offset(cable: cx.CableData) -> None:
    kappa = np.array([0.0, 1e-4, 0.01, 0.05, -0.05])
    tension = np.full(kappa.size, 60e3)
    result = cx.compute_neutral_axes(
        cable, kappa, np.zeros_like(kappa), tension_N=tension
    )
    strain = 60e3 / ea_fixed(cable)
    np.testing.assert_allclose(result.axial_strain, strain)
    offset = result.strain_axis_offset_m
    assert math.isnan(offset[0])  # no curvature
    assert math.isnan(offset[1])  # the line misses the section
    np.testing.assert_allclose(offset[2:], -strain / np.abs(kappa[2:]))
    assert np.all(offset[2:] < 0.0)  # towards the compressed side
    direct = cx.compute_neutral_axes(
        cable, kappa, np.zeros_like(kappa), axial_strain=np.full(kappa.size, -2e-4)
    )
    np.testing.assert_allclose(direct.strain_axis_offset_m[3:], 2e-4 / 0.05)
    with pytest.raises(CableInputError, match="not both"):
        cx.compute_neutral_axes(
            cable, kappa, kappa, tension_N=tension, axial_strain=tension
        )


def test_global_rotation_curvature_maps_to_the_tensile_fibre(
    cable: cx.CableData,
) -> None:
    # Global-model hogging, (x, y) curvature (0, +k): tensile fibre at 180 deg.
    rotation_x = np.array([0.0, 0.0, 0.02])
    rotation_y = np.array([0.02, -0.02, 0.0])
    bx, by = rotation_to_strain_curvature(rotation_x, rotation_y)
    result = cx.compute_neutral_axes(cable, bx, by)
    np.testing.assert_allclose(
        np.degrees(result.bending_direction_rad), [180.0, 0.0, 90.0]
    )
    # The rotation-curvature vector lies along the strain neutral axis, 90 deg
    # clockwise of the bending direction.
    np.testing.assert_allclose(
        np.degrees(
            angle_from(result.bending_direction_rad, np.arctan2(rotation_y, rotation_x))
        ),
        -90.0,
    )


def test_other_friction_laws_give_axes_but_no_stick_zone(cable: cx.CableData) -> None:
    turn, kx, ky = _rotating(0.04, revolutions=1, per_revolution=200)
    for law in ("distributed", "progressive"):
        result = cx.compute_neutral_axes(cable, kx, ky, friction_model=law)  # type: ignore[arg-type]
        assert all(
            layer.stuck is None and layer.stick_arc_rad is None
            for layer in result.layers
        )
        assert np.isfinite(result.layers[1].stress_direction_rad[-1])
        assert "direction-tied law only" in " ".join(result.provenance.limitations)
    with pytest.raises(CableInputError, match="friction_model"):
        cx.compute_neutral_axes(cable, kx, ky, friction_model="other")  # type: ignore[arg-type]


def test_inputs_are_checked(cable: cx.CableData) -> None:
    layers = _layers(cable)
    with pytest.raises(CableInputError, match="matching"):
        neutral_axes(layers, np.zeros(3), np.zeros(4))  # type: ignore[arg-type]
    with pytest.raises(CableInputError, match="finite"):
        neutral_axes(layers, np.array([np.nan]), np.zeros(1))  # type: ignore[arg-type]
    with pytest.raises(CableInputError, match="share"):
        neutral_axes(layers, np.zeros(2), np.zeros(2), direction_share=1.0)  # type: ignore[arg-type]
    with pytest.raises(CableInputError, match="negative"):
        cx.compute_neutral_axes(
            cable, np.zeros(2), np.zeros(2), tension_N=np.array([1.0, -1.0])
        )


# -- core friction in the section moment ---------------------------------------


_CORES = cx.CoreFrictionSpec(
    mu_core_filler=0.46, mu_core_core=0.46, mu_core_sheath=0.46, filler_profiles=True
)


@pytest.fixture(scope="module")
def cored() -> cx.CableData:
    """The public 66 kV template: laid cores, trefoil wedge and filler profiles."""
    return cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)


@pytest.mark.parametrize("phi", [0.0, 0.5 * math.pi])
def test_core_friction_moment_is_that_of_compute_mkappa_in_planar_bending(
    cored: cx.CableData, phi: float
) -> None:
    kappa = 0.02 * np.sin(np.linspace(0.0, 4.0 * math.pi, 401))
    tension = np.full(kappa.size, 50e3)
    ref = cx.compute_mkappa(
        cored, kappa, tension_N=50e3, core_friction=_CORES, phi_bend=phi
    )
    result = cx.compute_neutral_axes(
        cored,
        kappa * math.cos(phi),
        kappa * math.sin(phi),
        tension_N=tension,
        core_friction=_CORES,
    )
    along = result.moment_Nm @ np.array([math.cos(phi), math.sin(phi)])
    scale = float(np.ptp(ref.moment_Nm))
    np.testing.assert_allclose(along, ref.moment_Nm, rtol=0.0, atol=1e-12 * scale)
    assert result.ei_max_Nm2 == pytest.approx(ref.EI_max_Nm2, rel=1e-12)
    assert result.core_moment_Nm is not None
    # The cores add a hysteresis of their own (a moment at zero curvature).
    assert float(np.max(np.abs(result.core_moment_Nm))) > 0.05 * scale
    assert "laid-core law" in " ".join(result.provenance.limitations)


def test_core_friction_off_keeps_the_armour_moment_bit_for_bit(
    cored: cx.CableData,
) -> None:
    turn, kx, ky = _rotating(0.01, revolutions=1, per_revolution=120)
    tension = np.full(kx.size, 50e3)  # no clamping: contact from tension
    base = cx.compute_neutral_axes(cored, kx, ky, tension_N=tension)
    off = cx.compute_neutral_axes(cored, kx, ky, tension_N=tension, core_friction=False)
    np.testing.assert_array_equal(base.moment_Nm, off.moment_Nm)
    assert base.core_moment_Nm is None and off.core_moment_Nm is None
    on = cx.compute_neutral_axes(cored, kx, ky, tension_N=tension, core_friction=_CORES)
    assert on.core_moment_Nm is not None
    np.testing.assert_allclose(
        on.moment_Nm - on.core_moment_Nm, base.moment_Nm, rtol=0.0, atol=1e-9
    )
    # Friction dissipates under the turning curvature: the core moment leads
    # it (positive work per unit of anticlockwise rotation).
    late = slice(60, None)
    work = kx[late] * on.core_moment_Nm[late, 1] - ky[late] * on.core_moment_Nm[late, 0]
    assert float(np.mean(work)) > 0.0


def test_core_moment_inputs_are_checked(cable: cx.CableData) -> None:
    layers = _layers(cable)
    with pytest.raises(CableInputError, match=r"one \(x, y\) pair"):
        neutral_axes(layers, np.zeros(3), np.zeros(3), core_moment_Nm=np.zeros((2, 2)))  # type: ignore[arg-type]
    with pytest.raises(CableInputError, match="needs the core moment"):
        neutral_axes(layers, np.zeros(3), np.zeros(3), core_stick_EI_Nm2=1.0)  # type: ignore[arg-type]
    with pytest.raises(CableInputError, match="core stick stiffness"):
        neutral_axes(  # type: ignore[arg-type]
            layers,
            np.zeros(3),
            np.zeros(3),
            core_moment_Nm=np.zeros((3, 2)),
            core_stick_EI_Nm2=-1.0,
        )
