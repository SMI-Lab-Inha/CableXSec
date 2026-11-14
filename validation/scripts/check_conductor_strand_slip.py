# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Public-data checks of the conductor strand-slip operator.

Run as ``python validation/scripts/check_conductor_strand_slip.py``. Every input below is
read from the cited public source; nothing is fitted to the compared value.

1. Karlsen-Husoy et al. (2026) Tables 3-4: the multilinear operator's slope
   breakpoints and the strain-range ordering of the six transfer methods.
2. Hu et al. (2022) reciprocating three-point bending of a 61-wire conductor
   (Table 1 geometry, Table 2 materials, friction and sheath pressure from
   Section 3): full-slip bending stiffness 14.4 N m^2, loop area 3.185 N m/m,
   maximum moment 8.18 N m (Nam et al. 2024, Table 3).
3. Nam et al. (2024) Table 4: UFLEX 2D stick stiffness 1705 N m^2 of the same
   conductor.
4. Utting and Jones (1987) seven-wire strands: wire strain per unit tension
   (Part II Table 2, Strand V) and fixed-end torque per unit tension
   (Part II Table 1).
5. Nasution et al. (2014) 300 mm^2 conductor under 30 kN and reversed
   bending: analytical friction stress range of the 2nd layer (117 MPa,
   Table 5) and measured lives (Table 2).
"""

from __future__ import annotations

import math

import numpy as np

from cablexsec.analysis.conductor_fatigue import assess_conductor_fatigue
from cablexsec.materials.sn_curves import get_sn_curve
from cablexsec.mechanics.strand_slip import (
    ConductorRing,
    StrandSlipParameters,
    conductor_axial_stiffness,
    conductor_bending_moment,
    conductor_ring_parameters,
    strand_slip_strain,
)

# --------------------------------------------------------------------------- #
# Source data                                                                 #
# --------------------------------------------------------------------------- #
KH_TABLE3 = StrandSlipParameters.from_karlsen_husoy(
    1.4e-6, (4.2e-2, 7.5e-3, 2.0e-3), (5.9e-4, 7.5e-4), 1.0
)
#: Karlsen-Husoy et al. (2026) Table 4, T0 = 187 kN, alpha = 6.84e-7 /kN.
KH_TABLE4 = {
    "AO": ((0.135, 1.98e-2, 2.54e-3), (4.30e-5, 6.34e-5)),
    "SOE": ((0.135, 1.98e-2), (4.30e-5,)),
    "MOE": ((0.135, 2.54e-3), (6.34e-5,)),
    "FS": ((0.135, 0.135), (0.0,)),
    "FFS": ((1.98e-2, 1.98e-2), (0.0,)),
    "FSS": ((2.54e-3, 2.54e-3), (0.0,)),
}
#: Karlsen-Husoy et al. (2026) Table 5 annual damage (ordering only).
KH_TABLE5_DAMAGE = {
    "AO": 9.66e-6,
    "SOE": 1.24e-4,
    "MOE": 1.37e-5,
    "FS": 2.29,
    "FFS": 5.26e-5,
    "FSS": 6.46e-9,
}

HU_D = 2.35e-3
HU_LAYERS = ((6, 3.7), (12, -7.4), (18, 11.2), (24, -15.0))  # Table 1
HU_E, HU_SHEATH_E, HU_SHEATH_T = 118e9, 0.42e9, 1.5e-3  # Table 2, Table 1
HU_MU_WIRE, HU_MU_SHEATH, HU_PRESSURE = 0.2, 0.3, 0.542e6  # Section 3
HU_KAPPA = 12 * 6e-3 / 0.44**2  # Eq. 2, w = 6 mm, L = 440 mm
HU_EI_SLIP, HU_AREA, NAM_M_MAX, NAM_EI_STICK = 14.4, 3.185, 8.18, 1705.0

UJ_DC, UJ_DH, UJ_E = 3.94e-3, 3.73e-3, 197.9e9  # Part I Table 1
#: Part II Table 1: helix angle [deg], measured fixed-end torque [N m/kN].
UJ_TORQUE = {
    "I": (73.0, 0.86),
    "II": (76.0, 0.72),
    "III": (77.1, 0.69),
    "IV": (77.8, 0.65),
    "V": (79.0, 0.58),
    "VI": (80.8, 0.47),
}
#: Part II Table 2, Strand V: strand load [kN] -> gauges a, b, c [microstrain].
UJ_GAUGES = {20.0: (311.4, 339.2, 693.2), 70.0: (3535.0, 3371.0, 3865.0)}

NAS_D, NAS_E, NAS_T = 3.21e-3, 115e9, 30e3  # Section 2.1, Eq. 2
NAS_LAYERS = ((6, 265e-3, 4.33), (12, 285e-3, -8.03), (18, 295e-3, 11.56))
NAS_FRICTION_RANGE_L2 = 117e6  # Table 5, analytical, both radii
#: Table 2: radius [m], friction coefficient -> measured lives.
NAS_TESTS = {
    (3.0, 0.2): (212644, 225244, 262394),
    (6.5, 0.2): (617474, 951059, 980782),
    (3.0, 0.02): (387178, 372636, 434643),
}


def hu_rings() -> tuple[ConductorRing, ...]:
    rings = [ConductorRing(1, HU_D, 0.0, 0.0)]
    for index, (n, angle) in enumerate(HU_LAYERS, start=1):
        rings.append(
            ConductorRing(
                n, HU_D, index * HU_D, math.radians(abs(angle)), math.copysign(1, angle)
            )
        )
    return tuple(rings)


def nasution_rings() -> tuple[ConductorRing, ...]:
    rings = [ConductorRing(1, NAS_D, 0.0, 0.0)]
    for index, (n, pitch, angle) in enumerate(NAS_LAYERS, start=1):
        radius = index * NAS_D
        rings.append(
            ConductorRing.from_pitch(n, NAS_D, radius, pitch, math.copysign(1, angle))
        )
    return tuple(rings)


# --------------------------------------------------------------------------- #
# Checks                                                                      #
# --------------------------------------------------------------------------- #
def karlsen_husoy_breakpoints() -> dict[str, float]:
    """Slope changes after a reversal against the closed-form curvatures."""
    t = np.linspace(0, 4 * math.pi, 8001)
    kappa = 0.05 * np.sin(t)
    tension = np.full_like(kappa, KH_TABLE3.T0_N)  # T = T0 as in the source
    result = strand_slip_strain(
        KH_TABLE3, kappa, np.zeros_like(kappa), tension, theta_rad=[0]
    )
    strain = result.bending_strain[0]
    start = 5000  # reversal at kappa = +0.05 in the second loop
    slope = np.diff(strain[start:]) / np.diff(kappa[start:])
    beta = KH_TABLE3.beta_m
    eps = KH_TABLE3.eps_f_ref
    first = 2 * eps[0] / (beta[0] - beta[1])
    rise = (beta[0] - beta[2]) * first
    second = first + (2 * eps[1] - rise) / (beta[1] - beta[2])
    # First sample whose secant slope falls below the mid-slope of each pair.
    observed = [
        0.05 - kappa[start + int(np.argmax(slope < 0.5 * (upper + lower)))]
        for upper, lower in ((beta[0], beta[1]), (beta[1], beta[2]))
    ]
    return {
        "first_closed_form": first,
        "first_operator": observed[0],
        "second_closed_form": second,
        "second_operator": observed[1],
        "full_slip_range": float(strain[4000:].max() - strain[4000:].min()),
        "full_slip_closed_form": beta[2] * 0.1 + 2 * eps[1],
    }


def karlsen_husoy_methods(amplitude: float) -> dict[str, float]:
    """Strain range of each Table 4 method for a sinusoidal curvature."""
    t = np.linspace(0, 6 * math.pi, 6001)
    kappa = amplitude * np.sin(t)
    ranges = {}
    for name, (beta, eps) in KH_TABLE4.items():
        params = StrandSlipParameters.from_karlsen_husoy(6.84e-7, beta, eps, 187.0)
        result = strand_slip_strain(
            params,
            kappa,
            np.zeros_like(kappa),
            np.full_like(kappa, 187e3),
            theta_rad=[0],
        )
        steady = result.strain[0, 2000:]
        ranges[name] = float(steady.max() - steady.min())
    return ranges


def hu_loop(transmission: str) -> dict[str, float]:
    rings = hu_rings()
    r_outer = 4.5 * HU_D
    params = conductor_ring_parameters(
        rings,
        HU_E,
        mu_wire=HU_MU_WIRE,
        mu_outer=HU_MU_SHEATH,
        reference_tension_N=100.0,  # Nam et al. (2024) Section 2.1
        external_pressure_Pa=HU_PRESSURE,
        r_outer_m=r_outer,
        pressure_transmission=transmission,  # type: ignore[arg-type]
        fibre="centroid",
    )
    sheath_ei = HU_SHEATH_E * math.pi / 4 * ((r_outer + HU_SHEATH_T) ** 4 - r_outer**4)
    t = np.linspace(0, 6 * math.pi, 6001)
    kappa = HU_KAPPA * np.sin(t)
    tension = np.full_like(kappa, 100.0)
    moment, _ = conductor_bending_moment(
        rings,
        HU_E,
        params,
        kappa,
        np.zeros_like(kappa),
        tension,
        additional_EI_Nm2=sheath_ei,
    )
    loop = slice(4000, 6001)  # third cycle, as in the test
    area = float(abs(np.trapezoid(moment[loop], kappa[loop])))
    stick_moment, _ = conductor_bending_moment(
        rings,
        HU_E,
        [StrandSlipParameters(p.alpha_per_N, (p.beta_m[0],), (), 1.0) for p in params],
        np.array([1e-6]),
        np.array([0.0]),
        additional_EI_Nm2=sheath_ei,
    )
    # Full-slip slope between 70% and 95% of the loading branch.
    branch = slice(5200, 5450)
    slope = float(np.polyfit(kappa[branch], moment[branch], 1)[0])
    return {
        "EI_full_slip": slope,
        "EI_stick": float(stick_moment[0] / 1e-6),
        "M_max": float(moment[loop].max()),
        "loop_area": area,
    }


def utting_jones() -> dict[str, float]:
    out: dict[str, float] = {}
    r_h = (UJ_DC + UJ_DH) / 2
    for name, (helix, torque) in UJ_TORQUE.items():
        lay = math.radians(90 - helix)
        rings = (
            ConductorRing(1, UJ_DC, 0.0, 0.0),
            ConductorRing(6, UJ_DH, r_h, lay),
        )
        ea = conductor_axial_stiffness(rings, UJ_E)
        wire_tension = UJ_E * rings[1].area_m2 * math.cos(lay) ** 2 / ea  # per N
        predicted = 6 * wire_tension * r_h * math.sin(lay) * 1e3  # N m per kN
        out[f"torque_{name}"] = predicted / torque
        if name == "V":
            params = conductor_ring_parameters(
                rings, UJ_E, mu_wire=0.0, reference_tension_N=1.0, fibre="centroid"
            )
            out["strain_per_kN_V_model"] = params[1].alpha_per_N * 1e3 * 1e6
            measured = [
                (b - a) / 50.0
                for a, b in zip(UJ_GAUGES[20.0], UJ_GAUGES[70.0], strict=True)
            ]
            out["strain_per_kN_V_measured"] = float(np.mean(measured))
    return out


def nasution() -> dict[str, float]:
    rings = nasution_rings()
    out: dict[str, float] = {}
    params = conductor_ring_parameters(
        rings, NAS_E, mu_wire=0.2, reference_tension_N=NAS_T
    )
    out["friction_range_L2_MPa"] = 2 * params[1].eps_f_ref[0] * NAS_E / 1e6
    sn = get_sn_curve("nasution_2014_copper")
    for (radius, mu), lives in NAS_TESTS.items():
        t = np.linspace(0, 6 * math.pi, 1801)
        kappa = np.sin(t) / radius
        result = assess_conductor_fatigue(
            rings,
            NAS_E,
            0.3,
            kappa,
            np.zeros_like(kappa),
            np.full_like(kappa, NAS_T),
            sn_bulk=sn,
            swt_const=None,
            mu_wire=mu,
            reference_tension_N=NAS_T,
        )
        per_cycle = result.D_bulk_per_ring / 3.0
        ring = int(np.argmax(per_cycle))
        key = f"r{radius:g}_mu{mu:g}"
        out[f"{key}_critical_layer"] = ring + 1  # source counts the centre as 1st
        out[f"{key}_range_L2_MPa"] = result.stress_range_max_Pa[1] / 1e6
        out[f"{key}_life_model"] = 1.0 / per_cycle[ring]
        out[f"{key}_life_tests_mean"] = float(np.mean(lives))
    return out


def main() -> None:
    print("Karlsen-Husoy (2026) Table 3 breakpoints")
    for key, value in karlsen_husoy_breakpoints().items():
        print(f"  {key:24s} {value:.6g}")
    for amplitude in (0.002, 0.05):
        ranges = karlsen_husoy_methods(amplitude)
        order = sorted(ranges, key=ranges.get, reverse=True)
        print(
            f"Table 4 strain ranges at kappa_a = {amplitude}: "
            + ", ".join(f"{k} {ranges[k]:.3g}" for k in order)
        )
    print(
        "  Table 5 damage order: "
        + ", ".join(sorted(KH_TABLE5_DAMAGE, key=KH_TABLE5_DAMAGE.get, reverse=True))
    )
    for transmission in ("outer_interface", "all_interfaces"):
        hu = hu_loop(transmission)
        print(f"Hu (2022) loop, sheath pressure to {transmission}")
        print(f"  EI full slip {hu['EI_full_slip']:.3g} (test {HU_EI_SLIP})")
        print(f"  EI stick     {hu['EI_stick']:.4g} (Nam UFLEX {NAM_EI_STICK})")
        print(f"  M max        {hu['M_max']:.3g} (test {NAM_M_MAX})")
        print(f"  loop area    {hu['loop_area']:.3g} (test {HU_AREA})")
    print("Utting and Jones (1987)")
    for key, value in utting_jones().items():
        print(f"  {key:26s} {value:.4g}")
    print("Nasution et al. (2014)")
    for key, value in nasution().items():
        print(f"  {key:30s} {value:.4g}")


if __name__ == "__main__":
    main()
