# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Independent spatial and quadrature checks of separated radial support."""

import math

import numpy as np
import pytest
from radial_foundation_fd import solve_radial_foundation_fd
from scipy.optimize import brentq

from cablexsec.mechanics.contact_pressure import solve_three_core

pytestmark = pytest.mark.independent_method

PARAMETERS = {
    "q0_Pa": 3e5,
    "k_core": 2.4e12,
    "k_filler": 1.4e10,
    "theta_core_rad": 1.72,
    "EI_ring": 2.0,
    "r_bedding_m": 0.109,
    "n_wires": 110,
    "phi_cores": (0.0, 2 * math.pi / 3, 4 * math.pi / 3),
}


def test_separated_support_energy_by_independent_root_bracketing_and_quadrature():
    result = solve_three_core(**PARAMETERS, max_harmonic=144)
    modes = np.asarray(result.harmonic_orders)
    coefficients = result.displacement_coefficients_m
    assert coefficients is not None

    def value(phi):
        return float(np.cos(modes * phi) @ coefficients)

    scan = np.linspace(0, math.pi / 3, 4097)
    displacement = result.radial_displacement(scan)
    changes = np.flatnonzero(displacement[:-1] * displacement[1:] < 0)
    roots = [brentq(value, scan[i], scan[i + 1], xtol=2e-14) for i in changes]
    assert len(roots) > 0  # This check must actually exercise separation.
    edges = sorted([0.0, PARAMETERS["theta_core_rad"] / 2, math.pi / 3, *roots])
    gauss, weights = np.polynomial.legendre.leggauss(160)
    energy, work, mean_foundation, closed_length = 0.0, 0.0, 0.0, 0.0
    radius = PARAMETERS["r_bedding_m"]
    for lo, hi in zip(edges, edges[1:], strict=False):
        phi = (lo + hi) / 2 + (hi - lo) / 2 * gauss
        measure = weights * (hi - lo) / 2
        basis = np.cos(phi[:, None] * modes)
        u = basis @ coefficients
        curvature = basis @ ((1 - modes**2) * coefficients) / radius**2
        stiffness = (
            PARAMETERS["k_core"]
            if (lo + hi) / 2 < PARAMETERS["theta_core_rad"] / 2
            else PARAMETERS["k_filler"]
        )
        reaction = stiffness * np.maximum(u, 0.0)
        energy += (
            0.5
            * radius
            * np.sum(measure * (PARAMETERS["EI_ring"] * curvature**2 + reaction * u))
        )
        work += radius * PARAMETERS["q0_Pa"] * np.sum(measure * u)
        mean_foundation += np.sum(measure * reaction) * 3 / math.pi
        if value((lo + hi) / 2) > 0:
            closed_length += hi - lo
    assert energy == pytest.approx(work / 2, rel=3e-10)
    assert mean_foundation == pytest.approx(
        result.foundation_mean_pressure_Pa, rel=3e-10
    )
    assert closed_length * 3 / math.pi == pytest.approx(
        result.contact_fraction, abs=2e-10
    )
    assert np.min(result.shape_g) == 0.0


def test_separated_support_agrees_with_independent_spatial_refinement():
    result = solve_three_core(**PARAMETERS, max_harmonic=576)
    errors = []
    for nodes in (768, 1536, 3072):
        spatial = solve_radial_foundation_fd(
            pressure_Pa=PARAMETERS["q0_Pa"],
            core_stiffness_N_per_m3=PARAMETERS["k_core"],
            filler_stiffness_N_per_m3=PARAMETERS["k_filler"],
            support_width_rad=PARAMETERS["theta_core_rad"],
            rigidity_Nm=PARAMETERS["EI_ring"],
            radius_m=PARAMETERS["r_bedding_m"],
            nodes=nodes,
        )
        exact = result.radial_displacement(spatial.phi_rad)
        errors.append(
            np.linalg.norm(spatial.displacement_m - exact) / np.linalg.norm(exact)
        )
        assert spatial.stored_energy_J_per_m == pytest.approx(
            spatial.pressure_work_J_per_m / 2, rel=2e-7
        )
        assert spatial.relative_equilibrium_residual < 2e-7
    assert errors[-1] < 0.001
    assert errors[2] < errors[1] < errors[0]


def test_independent_spatial_uniform_support_has_analytical_solution():
    pressure, support, rigidity, radius = 3e5, 2e10, 100.0, 0.0775
    result = solve_radial_foundation_fd(
        pressure_Pa=pressure,
        core_stiffness_N_per_m3=support,
        filler_stiffness_N_per_m3=support,
        support_width_rad=1.0,
        rigidity_Nm=rigidity,
        radius_m=radius,
        nodes=192,
    )
    np.testing.assert_allclose(
        result.displacement_m, pressure / (support + rigidity / radius**4), rtol=1e-10
    )
