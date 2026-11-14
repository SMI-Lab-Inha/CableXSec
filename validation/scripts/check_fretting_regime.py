# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Verification and published checks of the fretting regime module (gap 3.1).

Gates (BENCHMARKS.md, "Fretting regime and wear"):

* FR (Tier 1): Mindlin energy closed form against the numerical Masing loop of
  :func:`cablexsec.mechanics.mindlin_contact.mindlin_monotonic`; Fouvry
  transition ratios A, D, B; Cattaneo strip limit ``I(0) = pi/16``; infinite
  friction gives stick and no energy; rotation of a planar path changes no
  amplitude.
* FOU95 (Tier 2, MM): [Fouvry1995] Tables 3-4, Mindlin transition amplitudes
  of the HSS/steel-ball case from the printed inputs (G as printed).
* WEAR-CW (Tier 2, M): energy wear volume of crossed 0.45 mm steel wires
  ([Poon2024] Tables 4-5, data of [Cruzado2013]) against the volume of the
  measured scar (elliptic paraboloid cap). The coefficient was identified by
  [Poon2024] on the same tests, so the check covers the energy-to-volume chain
  and the stroke convention, not the prediction of the coefficient.

Targets were set before the runs. Run from the repository root:
``python validation/scripts/check_fretting_regime.py``.
"""

from __future__ import annotations

import math

import numpy as np

from cablexsec.analysis import fretting_regime as fr
from cablexsec.materials.wear import WearCoefficient
from cablexsec.mechanics.hertz_contact import crossed_cylinder_curvatures, solve_contact
from cablexsec.mechanics.mindlin_contact import mindlin_monotonic

#: [Fouvry1995] Table 4: P [N], mu at transition, Mindlin delta_t [um].
FOUVRY_TABLE4 = (
    (200.0, 0.97, 3.2),
    (300.0, 0.95, 4.1),
    (400.0, 0.87, 4.6),
    (500.0, 0.85, 5.2),
)
FOUVRY_TOLERANCE_UM = 0.1  # one printed decimal
WEAR_BAND = (0.8, 1.25)


def verification_checks() -> dict[str, float]:
    """Tier-1 residuals (see the module docstring)."""
    mu, force, stiffness = 0.6, 25.0, 3.0e7
    dt = fr.mindlin_transition_amplitude(mu, force, stiffness)
    worst = 0.0
    for ratio in (0.05, 0.3, 0.7, 1.0, 1.5, 4.0):
        da = ratio * dt
        q_star = float(mindlin_monotonic(min(da, dt), mu, force, stiffness))
        x = np.linspace(-da, da, 400_001)
        upper = np.minimum(
            -q_star + 2.0 * mindlin_monotonic(0.5 * (x + da), mu, force, stiffness),
            q_star,
        )
        numeric = 2.0 * float(np.trapezoid(upper, x))
        closed = float(fr.mindlin_cycle_energy(np.array([da]), mu, force, dt)[0])
        worst = max(worst, abs(closed / numeric - 1.0))
    a, d, b = fr.fouvry_ratios(np.array([1.0]), 1.0)
    fouvry = max(
        abs(a[0] - 0.2),
        abs(d[0] - (2 * 0.5 ** (2 / 3) - 1)),
        abs(b[0] - 0.2 / (2 * 0.5 ** (2 / 3) - 1)),
    )
    strip = abs(fr._strip_integral(np.array([0.0]))[0] / (math.pi / 16) - 1.0)
    stick = fr.point_contact("p", 10.0, 50.0, 80.0, 210e9, 0.3, 210e9, 0.3, math.inf)
    t = np.linspace(0, 12 * math.pi, 3001)
    path = np.c_[30e-6 * np.sin(t), 12e-6 * np.cos(1.3 * t)]
    stick_energy = fr.assess_contact(path, stick, partial_slip_wear=True).total_energy
    contact = fr.point_contact("p", 20.0, 40.0, 70.0, 210e9, 0.3, 210e9, 0.3, 0.4)
    base = fr.assess_contact(path, contact)
    rotation = 0.0
    for angle in (0.3, 1.1, 2.5):
        c, s = math.cos(angle), math.sin(angle)
        turned = fr.assess_contact(path @ np.array([[c, s], [-s, c]]), contact)
        rotation = max(
            rotation,
            float(np.max(np.abs(turned.amplitude_m / base.amplitude_m - 1.0))),
        )
    return {
        "mindlin": worst,
        "fouvry": float(fouvry),
        "strip": float(strip),
        "stick_energy": float(stick_energy),
        "rotation": rotation,
    }


def fouvry_case() -> list[tuple[float, float, float, float]]:
    """``(P, a [um], delta_t printed G [um], delta_t consistent G [um])``."""
    radius, e_star, nu = 12.7e-3, 119.4e9, 0.28
    rows = []
    for force, mu, _printed in FOUVRY_TABLE4:
        hz = solve_contact(force, 1 / (2 * radius), 1 / (2 * radius), e_star)
        g_printed = 1.0 / ((2 - nu) / 159.7e9 + (2 - nu) / 146.5e9)
        g_true = 1.0 / (
            (2 - nu) / (230e9 / (2 * (1 + nu))) + (2 - nu) / (211e9 / (2 * (1 + nu)))
        )
        rows.append(
            (
                force,
                1e6 * hz.a_m,
                1e6
                * fr.mindlin_transition_amplitude(mu, force, 8 * g_printed * hz.a_m),
                1e6 * fr.mindlin_transition_amplitude(mu, force, 8 * g_true * hz.a_m),
            )
        )
    return rows


def wear_case() -> float:
    """Module volume over measured scar volume, crossed steel wires at 45 deg."""
    d, theta, force, mu, cycles = 0.45e-3, math.radians(45), 1.0, 0.7, 50_000
    a_c, b_c = crossed_cylinder_curvatures(d, theta)
    k = 2.93e-8 * 1e-6
    coefficient = WearCoefficient(
        "test", k / mu, k / mu, k / mu, "[Poon2024]", False, "wire"
    )
    contact = fr.point_contact(
        "wire", force, a_c, b_c, 210e9, 0.33, 210e9, 0.33, mu, wear=coefficient
    )
    u = 65e-6 * np.sin(np.linspace(0, 2 * math.pi * 200, 200 * 16 + 1))
    volume = fr.assess_contact(u, contact).wear_volume * cycles / 200
    measured = math.pi / 8 * 154e-6 * 381e-6 * 18e-6
    return volume / measured


def main() -> None:
    v = verification_checks()
    print("FR:", {k: f"{x:.2e}" for k, x in v.items()})
    for force, a, printed, true in fouvry_case():
        print(
            f"FOU95 P={force:.0f} N a={a:.1f} um dt={printed:.2f} um (G as printed), {true:.2f} um (G = E/2(1+nu))"
        )
    print(f"WEAR-CW volume ratio {wear_case():.3f}")


if __name__ == "__main__":
    main()
