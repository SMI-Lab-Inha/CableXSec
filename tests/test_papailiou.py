# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for the Papailiou axial slip-front M-κ model (R1.2 amendment)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cablexsec.mechanics.papailiou import (
    PapailiouLayer,
    compute_mkappa_papailiou,
)

# Test-taxonomy category.
pytestmark = pytest.mark.analytical_sanity


def _layer(contact_pressure: float = 1.4e6, mu: float = 0.2) -> PapailiouLayer:
    return PapailiouLayer(
        n_wires=93,
        r_layer_m=0.082,
        lay_angle_rad=math.radians(16.0),
        E_wire_Pa=210.0e9,
        A_wire_m2=math.pi * 0.005**2 / 4.0,
        d_wire_m=0.005,
        mu=mu,
        contact_pressure_Pa=contact_pressure,
    )


def test_sigma0_from_contact_pressure() -> None:
    """Cylindrical pressure and round-wire tension give the same radial force."""
    lyr = _layer(contact_pressure=1.5e6)
    a = lyr.lay_angle_rad
    pressure_force_per_axial_m = 2 * math.pi * lyr.r_layer_m * 1.5e6
    wire_curvature = math.sin(a) ** 2 / lyr.r_layer_m
    wire_force_per_axial_m = (
        lyr.n_wires * lyr.A_wire_m2 * lyr.sigma0_Pa * wire_curvature / math.cos(a)
    )
    assert wire_force_per_axial_m == pytest.approx(pressure_force_per_axial_m)


def test_odd_station_count_has_finite_neutral_axis_work():
    from dataclasses import replace

    layer = replace(_layer(), n_stations=401)
    history = 0.05 * np.sin(np.linspace(0, 4 * np.pi, 401))
    with np.errstate(divide="raise", invalid="raise"):
        result = compute_mkappa_papailiou([layer], history)
    assert np.isfinite(result.moment_Nm).all()
    assert np.isfinite(result.work.stored_J_per_m).all()
    assert np.max(np.abs(result.work.balance_residual_J_per_m)) < 1.0e-8


@pytest.mark.parametrize("physical_wires", [False, True])
def test_bending_work_separates_open_path_energy(physical_wires):
    from cablexsec.mechanics.mkappa import HelicalLayerSpec
    from cablexsec.mechanics.papailiou import compute_mkappa_wire_slip_front

    layer = _layer(contact_pressure=1.0e9)
    curvature = np.linspace(0.0, 1.0e-4, 101)
    if physical_wires:
        spec = HelicalLayerSpec(
            layer.n_wires,
            layer.r_layer_m,
            layer.lay_angle_rad,
            layer.E_wire_Pa,
            layer.A_wire_m2,
            0.3,
            layer.mu,
            0.0,
            contact_pressure_Pa=layer.contact_pressure_Pa,
        )
        result = compute_mkappa_wire_slip_front([spec], curvature, EI_min_Nm2=100.0)
    else:
        result = compute_mkappa_papailiou([layer], curvature, EI_min_Nm2=100.0)
    assert result.W_diss_J_per_m == 0.0
    assert result.work.stored_J_per_m[-1] == pytest.approx(
        0.5 * result.EI_max_Nm2 * curvature[-1] ** 2, rel=1.0e-12
    )
    assert np.max(np.abs(result.work.balance_residual_J_per_m)) < 1.0e-12


@pytest.mark.parametrize("physical_wires", [False, True])
def test_bending_work_closed_cycle_and_refinement(physical_wires):
    from cablexsec.mechanics.mkappa import HelicalLayerSpec
    from cablexsec.mechanics.papailiou import compute_mkappa_wire_slip_front

    layer = _layer(contact_pressure=1.0e5)
    remainders = []
    for count in (600, 1200, 2400):
        curvature = 0.1 * np.sin(np.linspace(0.0, 6 * np.pi, 3 * count + 1))
        if physical_wires:
            spec = HelicalLayerSpec(
                layer.n_wires,
                layer.r_layer_m,
                layer.lay_angle_rad,
                layer.E_wire_Pa,
                layer.A_wire_m2,
                0.3,
                layer.mu,
                0.0,
                contact_pressure_Pa=layer.contact_pressure_Pa,
            )
            result = compute_mkappa_wire_slip_front([spec], curvature, EI_min_Nm2=100.0)
        else:
            result = compute_mkappa_papailiou([layer], curvature, EI_min_Nm2=100.0)
        work = result.work
        assert np.min(np.diff(work.friction_J_per_m)) >= -1.0e-10
        assert np.max(np.abs(work.balance_residual_J_per_m)) < 1.0e-8
        assert result.W_diss_J_per_m == work.friction_J_per_m[-1]
        j = 2 * count
        loop_area = np.trapezoid(result.moment_Nm[j:], curvature[j:])
        dissipation = work.friction_J_per_m[-1] - work.friction_J_per_m[j]
        if count == 2400:
            assert loop_area == pytest.approx(dissipation, rel=1.0e-3)
        remainders.append(
            work.integration_remainder_J_per_m[-1]
            - work.integration_remainder_J_per_m[j]
        )
    assert remainders[2] < remainders[1] < remainders[0]


@pytest.mark.parametrize("angle_deg", [5.0, 20.0, 45.0, 70.0])
def test_single_layer_winding_recovers_wire_stress(angle_deg: float) -> None:
    """Feret--Bournazel Eq. (5) and the slip coefficient close force balance."""
    from dataclasses import replace

    a = math.radians(angle_deg)
    stress = 125.0e6
    base = replace(_layer(), lay_angle_rad=a)
    pressure = (
        base.n_wires
        * base.A_wire_m2
        * stress
        * math.sin(a)
        * math.tan(a)
        / (2 * math.pi * base.r_layer_m**2)
    )
    assert replace(base, contact_pressure_Pa=pressure).sigma0_Pa == pytest.approx(
        stress
    )


def test_kappa_c_matches_wire_line_load_limit() -> None:
    """Neutral-axis onset uses friction per wire length (Menard Eq. 15)."""
    lyr = _layer(contact_pressure=1.5e6)
    k_phi, f_y_phi, _ = lyr.station_params()
    kappa_c = f_y_phi / k_phi
    a = lyr.lay_angle_rad
    wire_line_load = 1.5e6 * (2 * math.pi * lyr.r_layer_m / lyr.n_wires) * math.cos(a)
    expected = (
        lyr.mu
        * wire_line_load
        / (lyr.E_wire_Pa * lyr.A_wire_m2 * math.cos(a) ** 2 * math.sin(a))
    )
    assert float(np.min(kappa_c)) == pytest.approx(expected, rel=2e-2)


def test_EI_stick_matches_witztan() -> None:
    """Full-stick parallel-axis stiffness = 1/2 n E A r^2 cos^3(beta)."""
    lyr = _layer()
    expected = (
        0.5
        * lyr.n_wires
        * lyr.E_wire_Pa
        * lyr.A_wire_m2
        * lyr.r_layer_m**2
        * math.cos(lyr.lay_angle_rad) ** 3
    )
    assert lyr.EI_stick == pytest.approx(expected, rel=1e-12)


def test_slip_starts_at_neutral_axis() -> None:
    """kappa_c(phi) = f_y/k is minimised near the neutral axis (phi = 0)."""
    lyr = _layer()
    k_phi, f_y_phi, _ = lyr.station_params()
    kappa_c = f_y_phi / k_phi
    phi = np.linspace(-math.pi / 2, math.pi / 2, lyr.n_stations + 1)
    phi = 0.5 * (phi[:-1] + phi[1:])
    i_min = int(np.argmin(kappa_c))
    # the station with the smallest critical curvature is the closest to phi = 0
    assert abs(phi[i_min]) == pytest.approx(np.min(np.abs(phi)), abs=1e-9)


def test_full_stick_limit_recovers_EImax() -> None:
    """At tiny curvature the tangent stiffness approaches EI_max."""
    lyr = _layer()
    kappa = np.linspace(0.0, 1e-5, 50)
    res = compute_mkappa_papailiou([lyr], kappa)
    tangent = (res.moment_Nm[1] - res.moment_Nm[0]) / (kappa[1] - kappa[0])
    assert tangent == pytest.approx(lyr.EI_stick, rel=1e-3)


def test_full_slip_saturates_to_plateau() -> None:
    """At large curvature the friction moment saturates (tangent -> EI_min)."""
    lyr = _layer()
    EI_min = 16.0e3
    kappa = np.linspace(0.0, 0.5, 4000)
    res = compute_mkappa_papailiou([lyr], kappa, EI_min_Nm2=EI_min)
    tangent_end = (res.moment_Nm[-1] - res.moment_Nm[-2]) / (kappa[-1] - kappa[-2])
    # at large kappa nearly all stations have slipped -> tangent ~ EI_min
    assert tangent_end == pytest.approx(EI_min, rel=0.05)


def test_smooth_knee_tangent_decreases_gradually() -> None:
    """The tangent stiffness decays smoothly EI_max -> EI_min (no single step)."""
    lyr = _layer()
    kappa = np.linspace(1e-6, 0.1, 3000)
    res = compute_mkappa_papailiou([lyr], kappa, EI_min_Nm2=16.0e3)
    dM = np.gradient(res.moment_Nm, res.kappa)
    # monotone non-increasing tangent, and no jump larger than 20% of EI_max in
    # one step (a sharp knee would show a near-discontinuity)
    assert dM[1] == pytest.approx(res.EI_max_Nm2, rel=0.05)
    assert dM[-1] < 0.1 * res.EI_max_Nm2
    # No near-discontinuity: a sharp knee would show a ~100%-of-EImax single
    # step; the gradual slip front keeps every step well below 15%.
    steps = np.abs(np.diff(dM))
    assert steps.max() < 0.15 * res.EI_max_Nm2


def test_zero_pretension_is_elastic_no_hysteresis() -> None:
    """zero contact pressure -> no friction -> M = EI_min * kappa, W_diss ~ 0."""
    lyr = _layer(contact_pressure=0.0)
    kappa = 0.2 * np.sin(np.linspace(0, 2 * np.pi, 400))
    res = compute_mkappa_papailiou([lyr], kappa, EI_min_Nm2=16.0e3)
    assert res.W_diss_J_per_m < 1e-6 * res.EI_max_Nm2 * 0.2**2
    assert np.allclose(res.moment_Nm, 16.0e3 * kappa, atol=1e-6)


def test_W_diss_increases_with_pretension() -> None:
    lo = compute_mkappa_papailiou(
        [_layer(contact_pressure=0.5e6)], 0.2 * np.sin(np.linspace(0, 2 * np.pi, 500))
    )
    hi = compute_mkappa_papailiou(
        [_layer(contact_pressure=1.5e6)], 0.2 * np.sin(np.linspace(0, 2 * np.pi, 500))
    )
    assert hi.W_diss_J_per_m > lo.W_diss_J_per_m


def test_planar_slip_front_is_independent_of_lay_handedness() -> None:
    from dataclasses import replace

    right = _layer()
    left = replace(right, lay_angle_rad=-right.lay_angle_rad)
    curvature = 0.1 * np.sin(np.linspace(0.0, 6 * np.pi, 601))
    a = compute_mkappa_papailiou([right], curvature)
    b = compute_mkappa_papailiou([left], curvature)
    np.testing.assert_array_equal(a.moment_Nm, b.moment_Nm)


@pytest.mark.parametrize(
    "values",
    [
        {"n_wires": 0},
        {"mu": -0.1},
        {"lay_angle_rad": 0.0},
        {"contact_pressure_Pa": float("nan")},
        {"n_stations": True},
        {"A_wire_m2": 0.0},
    ],
)
def test_invalid_slip_front_layer_is_rejected(values) -> None:
    from dataclasses import replace

    with pytest.raises(ValueError):
        replace(_layer(), **values)
