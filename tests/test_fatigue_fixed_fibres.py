# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Material-point histories, known bending limits and angular refinement."""

import math
from dataclasses import replace

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.fatigue import (
    FatigueLayer,
    compute_fatigue,
    miner_damage,
)
from cablexsec.materials.sn_curves import SNcurve
from cablexsec.mechanics.mkappa import HelicalLayerSpec
from cablexsec.mechanics.wire_stress import local_bending_fibre_stress


def _layer(alpha=0.0, n_wires=1):
    return FatigueLayer(
        spec=HelicalLayerSpec(
            n_wires=n_wires,
            r_helix_m=0.05,
            lay_angle_rad=alpha,
            E_wire_Pa=200e9,
            A_wire_m2=math.pi * 0.002**2,
            xi=1,
            mu=0,
            F_n_axi_N=0,
        ),
        sn_bulk=SNcurve(
            name="analytical test",
            stress_unit="MPa",
            m1=4,
            log_C1=12,
            threshold=0,
        ),
        sigma_UTS_Pa=1e9,
        name="wire",
    )


def test_rotating_bending_counts_material_history_not_constant_envelope():
    layer = _layer()
    phase = np.linspace(0, 8 * np.pi, 801)
    kx, ky = 0.03 * np.cos(phase), 0.03 * np.sin(phase)
    result = compute_fatigue([layer], kx, ky, n_fibres=4)
    amplitude = 200e9 * 0.002 * 0.03
    expected = 4 / layer.sn_bulk.cycles_to_failure(2 * amplitude / 1e6)
    assert result.D_bulk_per_fibre["wire"][0, 0] == pytest.approx(expected, rel=1e-12)
    assert np.all(result.D_bulk_per_fibre["wire"] > 0)
    # Instantaneous max stress on the circumference is constant in this case.
    moving_envelope = np.full_like(phase, amplitude)
    assert miner_damage(moving_envelope, layer.sn_bulk, layer.sigma_UTS_Pa) == 0


def test_transverse_bending_is_not_lost_at_a_radial_neutral_fibre():
    layer = _layer()
    curvature = np.array([0.0, 0.03])
    result = compute_fatigue([layer], curvature * 0, curvature, n_fibres=4)
    expected = 0.5 / layer.sn_bulk.cycles_to_failure(200e9 * 0.002 * 0.03 / 1e6)
    assert result.D_bulk_per_fibre["wire"][0, 0] == 0
    assert result.D_bulk_per_wire["wire"][0] == pytest.approx(expected, rel=1e-12)
    assert result.fibre_critical_deg == pytest.approx(90)


def test_half_cycle_matches_every_fixed_fibre_and_continuous_maximum():
    layer = _layer(alpha=0.4)
    kx, ky = np.array([0, 0.02]), np.array([0, 0.07])
    normal = np.array([0, -2e6])
    c, s = math.cos(0.4), math.sin(0.4)
    A = 200e9 * 0.002 * c**2 * math.cos(0.8) * kx[1]
    B = 200e9 * 0.002 * c * (1 + s**2) * ky[1]
    continuous = 0.5 / layer.sn_bulk.cycles_to_failure(
        (abs(normal[1]) + math.hypot(A, B)) / 1e6
    )
    maxima = []
    for count in (4, 16, 64):
        result = compute_fatigue(
            [layer],
            kx,
            ky,
            normal_stress_Pa={"wire": normal},
            n_fibres=count,
        )
        theta = np.radians(result.fibre_angles_deg)
        ranges = abs(normal[1] + A * np.cos(theta) + B * np.sin(theta))
        expected = np.array(
            [0.5 / layer.sn_bulk.cycles_to_failure(value / 1e6) for value in ranges]
        )
        np.testing.assert_allclose(
            result.D_bulk_per_fibre["wire"][0], expected, rtol=1e-12
        )
        maxima.append(result.D_bulk_per_wire["wire"][0])
        assert 180 < result.fibre_critical_deg < 360
    assert np.all(np.diff(maxima) >= 0)
    assert maxima[-1] > maxima[0]
    assert maxima[-1] <= continuous * (1 + 1e-13)
    assert maxima[-1] == pytest.approx(continuous, rel=0.005)


def test_time_dependent_modulus_is_used_at_each_material_fibre():
    layer = _layer(alpha=0.3)
    kx, ky = np.array([0.04, 0.04]), np.array([0.01, 0.01])
    moduli = np.array([200e9, 100e9])
    result = compute_fatigue(
        [layer],
        kx,
        ky,
        wire_modulus_Pa={"wire": moduli},
        n_fibres=8,
    )
    for j, theta in enumerate(np.radians(result.fibre_angles_deg)):
        stress = local_bending_fibre_stress(kx, ky, 0, moduli, 0.004, 0.3, theta)
        expected = 0.5 / layer.sn_bulk.cycles_to_failure(abs(np.diff(stress)[0]) / 1e6)
        assert result.D_bulk_per_fibre["wire"][0, j] == pytest.approx(
            expected, rel=1e-12
        )


def test_loading_rotation_permutes_wire_damage_without_rotating_fibres():
    layer = _layer(alpha=0.3, n_wires=4)
    phase = np.linspace(0, 3 * np.pi, 121)
    kx = 0.04 * np.sin(phase) + 0.01 * np.sin(3 * phase)
    ky = 0.02 * np.cos(2 * phase)
    first = compute_fatigue([layer], kx, ky, n_fibres=16)
    rotated = compute_fatigue([layer], -ky, kx, n_fibres=16)
    np.testing.assert_allclose(
        rotated.D_bulk_per_fibre["wire"],
        np.roll(first.D_bulk_per_fibre["wire"], 1, axis=0),
        rtol=1e-12,
        atol=1e-25,
    )


def test_fibre_resolution_does_not_change_wire_level_contact_indicator():
    from cablexsec.materials.sn_curves import get_swt_life

    layer = replace(
        _layer(alpha=0.3),
        swt_const=get_swt_life("copper_poon2022"),
        contact_p0_Pa=26e6,
        fretting_mu=0.8,
        K_t_contact_N_per_m=1e6,
        crossing_density_per_m=50.0,
        contact_line_load_N_per_m=100.0,
    )
    phase = np.linspace(0, 4 * np.pi, 81)
    results = [
        compute_fatigue(
            [layer],
            0.03 * np.sin(phase),
            0.01 * np.cos(phase),
            n_fibres=count,
        )
        for count in (4, 16)
    ]
    np.testing.assert_array_equal(
        results[0].D_fretting_per_wire["wire"],
        results[1].D_fretting_per_wire["wire"],
    )
    assert results[0].D_fretting_per_wire["wire"][0] > 0


@pytest.mark.parametrize("value", [True, np.bool_(False), 3, 0, -4, 4.5, np.nan])
def test_fibre_count_is_checked(value):
    with pytest.raises(ValueError, match="n_fibres"):
        compute_fatigue([_layer()], np.zeros(2), np.zeros(2), n_fibres=value)


def test_public_api_retains_fibre_locations():
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    k = np.array([0, 0.03, -0.03, 0.01])
    result = cx.compute_fatigue(cable, k, k * 0.3, n_fibres=12)
    np.testing.assert_allclose(result.fibre_angles_deg, np.arange(12) * 30, atol=1e-12)
    for name, values in result.D_bulk_per_fibre.items():
        assert values.shape[1] == 12
        np.testing.assert_array_equal(values.max(axis=1), result.D_bulk_per_wire[name])
        assert result.fibre_critical_deg_per_wire[name].shape == values.shape[:1]


def test_history_adapter_preserves_fibre_damage_and_resolution():
    from cablexsec.io.series import LoadCaseSeries, fatigue_from_series

    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    k = np.array([0, 0.03, -0.03, 0.01])
    series = LoadCaseSeries(
        time_s=np.arange(4, dtype=float),
        curvature_x_1_m=k,
        curvature_y_1_m=k * 0.3,
        tension_N=np.full(4, 1000.0),
        tension_kind="true",
    )
    wrapped = fatigue_from_series(cable, series, n_fibres=8)
    direct = cx.compute_fatigue(
        cable,
        k,
        k * 0.3,
        tension_N=series.tension_N,
        n_fibres=8,
    )
    np.testing.assert_array_equal(wrapped.fibre_angles_deg, direct.fibre_angles_deg)
    assert wrapped.fibre_critical_deg == direct.fibre_critical_deg
    for name in direct.D_bulk_per_fibre:
        np.testing.assert_array_equal(
            wrapped.D_bulk_per_fibre[name],
            direct.D_bulk_per_fibre[name],
        )
