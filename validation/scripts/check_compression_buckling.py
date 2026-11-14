# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Published-case check of the armour compression screens.

Run as ``python validation/scripts/check_compression_buckling.py``. Verification only:
nothing here is used by an analysis path. Every target below was recorded
from the source before the implementation was run.

Sources (library copies)
------------------------
* Sævik and Thorsen (2017), J. Offshore Mech. Arct. Eng. 139(4) 041701:
  Table 1 (three Østergaard pipes), Table 3 (Eq. 19 lateral capacity: 100,
  243 and 143 kN in the "Analytical model" column), Table 4 and Section 4.3.2
  (de Sousa 4-inch pipe: P3 = 1020 kN tape only, 1110 kN tape and sheath;
  P4 = 5.36 x 256 kN; test mean 256 kN; Table 5 normalised model values).
* Østergaard et al. (2012), Mar. Struct. 27 64-81, Fig. 14: torsionally
  fixed-free 6-inch pipe with transverse slip in both layers; force plateaus
  read from the figure as about 49.5 kN (inner), 48 kN (outer) and 98 kN
  (sum).
* Li and Vaz (2021a), Ocean Eng. 220 108306: Eq. 11 gives 1563 N for the
  umbilical wire of Table 1 (D 4.1 mm, r 35.75 mm, 20 deg, 210 GPa,
  nu 0.3); Eq. 15 gives first-slip curvatures 3.2e-4, 6.5e-4 and 9.7e-4 1/m
  for mu = 0.05, 0.1 and 0.15 at that load; Table 4 shows P*/Pcr = 0.83-0.85
  for mu = 0.05-0.15 (friction-independent limit).
* Li and Vaz (2021b), Mar. Struct. 76 102895: 106 wires give 166 kN (Section
  3); Table 3 knock-down 0.91, 0.88, 0.83 at kappa_max 0.1, 0.2, 0.3 1/m.
* Yang et al. (2015), Ocean Eng. 108 594-605, Eq. 9: tape failure load
  1.84 MN for the 9.5-inch pipe (tape 2.4 mm, 261 MPa, 30 deg armour).
* Malta et al. (2022), Ocean Eng. 245 110470, Tables 2-4 and 6: 6-inch pipe
  FE limit load 241.69 kN with the high-strength tape cut over 2000 mm
  (radial mode), the closest published analogue of a power cable restrained
  by its outer sheath only. FE with friction 0.2 and plasticity; a
  cross-check, not a reproduction.

Tolerances
----------
Eq. 19 and Eq. 11 reproductions 3% and 0.1% (torsion constant of a flat wire
is not stated in the source; Roark's form is used); figure readings 5%;
first-slip curvatures 2% (two-figure rounding); P4 0.5%; P3 6% (radius and
tape conventions not stated); Yang Eq. 9 3%; straight-beam P2* against the
de Sousa test mean 10%; the Malta cross-check is reported, not gated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cablexsec.analysis.compression import (
    TapeLayer,
    combined_radial_load,
    cyclic_curvature_knockdown,
    first_slip_curvature,
    lateral_buckling_load,
    lateral_buckling_load_round,
    radial_buckling_load_curved,
    radial_buckling_load_straight,
    radial_capacity_failure_load,
    rectangular_wire_section,
    round_wire_section,
    sheath_foundation_stiffness,
    tape_foundation_stiffness,
    wire_yield_load,
)


@dataclass(frozen=True)
class Check:
    """One published target against the computed value."""

    ident: str
    quantity: str
    published: float
    computed: float
    tolerance: float  # relative; nan = reported only

    @property
    def error(self) -> float:
        return self.computed / self.published - 1.0

    @property
    def passed(self) -> bool:
        return math.isnan(self.tolerance) or abs(self.error) <= self.tolerance


# --------------------------------------------------------------------------- #
# Sævik and Thorsen (2017) Table 3 / Østergaard (2012) Fig. 14                #
# --------------------------------------------------------------------------- #
#: Table 1: (outer diameter m, lay angle deg, thickness m, width m, wires).
OSTERGAARD_PIPES: dict[str, tuple[tuple[float, float, float, float, int], ...]] = {
    "6in": ((0.201, 26.2, 3e-3, 10e-3, 52), (0.209, 26.2, 3e-3, 10e-3, 54)),
    "8in": ((0.276, 30.0, 5e-3, 12.5e-3, 54), (0.289, 30.3, 5e-3, 12.5e-3, 56)),
    "14in": ((0.442, 31.5, 4e-3, 15e-3, 70), (0.452, 31.0, 4e-3, 15e-3, 72)),
}
TABLE3_ANALYTICAL_KN = {"6in": 100.0, "8in": 243.0, "14in": 143.0}
FIG14_PLATEAU_KN = {"inner": 49.5, "outer": 48.0, "sum": 98.0}


def lateral_pipe_layers(pipe: str) -> list[float]:
    """Eq. 19 capacity of each layer [N] (R = outer radius - t/2)."""
    out = []
    for od, a_deg, t, w, n in OSTERGAARD_PIPES[pipe]:
        sec = rectangular_wire_section(210e9, 0.3, t, w)
        out.append(
            lateral_buckling_load(n, od / 2.0 - t / 2.0, math.radians(a_deg), sec)
        )
    return out


def saevik_table3_checks() -> list[Check]:
    checks = [
        Check(
            f"ST17-T3-{pipe}",
            f"Eq. 19 lateral capacity, {pipe} pipe, both layers [kN]",
            TABLE3_ANALYTICAL_KN[pipe],
            sum(lateral_pipe_layers(pipe)) / 1e3,
            0.03,
        )
        for pipe in OSTERGAARD_PIPES
    ]
    inner, outer = lateral_pipe_layers("6in")
    checks += [
        Check("OST12-in", "Fig. 14 inner-layer plateau [kN]", 49.5, inner / 1e3, 0.05),
        Check("OST12-out", "Fig. 14 outer-layer plateau [kN]", 48.0, outer / 1e3, 0.05),
        Check(
            "OST12-sum", "Fig. 14 sum of layers [kN]", 98.0, (inner + outer) / 1e3, 0.05
        ),
    ]
    return checks


# --------------------------------------------------------------------------- #
# Li and Vaz (2021a, b)                                                       #
# --------------------------------------------------------------------------- #
LV_WIRE = {"d": 4.1e-3, "r": 35.75e-3, "phi_deg": 20.0, "E": 210e9, "nu": 0.3}
LV_PCR_N = 1563.0
LV_UMBILICAL_KN = 166.0
LV_SLIP = {0.05: 3.2e-4, 0.1: 6.5e-4, 0.15: 9.7e-4}
LV_FE_RATIO = {0.1: 0.91, 0.2: 0.88, 0.3: 0.83}


def li_vaz_checks() -> list[Check]:
    sec = round_wire_section(LV_WIRE["E"], LV_WIRE["nu"], LV_WIRE["d"])
    phi = math.radians(LV_WIRE["phi_deg"])
    pcr = lateral_buckling_load_round(1, LV_WIRE["r"], phi, sec)
    checks = [
        Check("LV21-Pcr", "Eq. 11 single-wire limit [N]", LV_PCR_N, pcr, 1e-3),
        Check(
            "LV21-106",
            "Eq. 11 x 106 wires [kN]",
            LV_UMBILICAL_KN,
            106 * pcr / 1e3,
            5e-3,
        ),
    ]
    q_wire = pcr / math.cos(phi)  # helix-direction compression at Pcr
    for mu, target in LV_SLIP.items():
        k = first_slip_curvature(mu, q_wire, phi, sec, LV_WIRE["r"])
        checks.append(
            Check(
                f"LV21-slip-{mu}", f"Eq. 15 first slip, mu {mu} [1/m]", target, k, 0.02
            )
        )
    for kappa, ratio in LV_FE_RATIO.items():
        eta, _ = cyclic_curvature_knockdown(kappa, LV_WIRE["r"])
        checks.append(
            Check(
                f"LV21b-eta-{kappa}",
                f"Table 3 knock-down at {kappa} 1/m",
                ratio,
                eta,
                1e-12,
            )
        )
    return checks


def li_vaz_lower_bound() -> dict[str, float]:
    """Sævik Eq. 19 for the Li-Vaz wire against the FE limits (lower bound)."""
    sec = round_wire_section(LV_WIRE["E"], LV_WIRE["nu"], LV_WIRE["d"])
    phi = math.radians(LV_WIRE["phi_deg"])
    p1 = lateral_buckling_load(1, LV_WIRE["r"], phi, sec)
    pcr = lateral_buckling_load_round(1, LV_WIRE["r"], phi, sec)
    return {
        "P1_N": p1,
        "Pcr_N": pcr,
        "P1_over_Pcr": p1 / pcr,
        "min_FE_ratio": min(LV_FE_RATIO.values()),
    }


# --------------------------------------------------------------------------- #
# de Sousa 4-inch pipe (Sævik and Thorsen 2017 Section 4.3)                   #
# --------------------------------------------------------------------------- #
SOUSA_TEST_N = 256e3
SOUSA_P3_TAPE_N = 1020e3
SOUSA_P3_TAPE_SHEATH_N = 1110e3
SOUSA_P4_N = 5.36 * SOUSA_TEST_N
#: Table 5 (normalised by 256 kN): (P2, P2*) for each published assumption.
SOUSA_TABLE5 = {
    ("curved", True): (1.96, 1.08),
    ("straight", True): (1.24, 0.82),
    ("curved", False): (1.51, 0.91),
    ("straight", False): (1.04, 0.72),
}


@dataclass(frozen=True)
class SousaResult:
    P2_N: float
    P3_N: float
    P4_N: float
    P2_star_N: float
    half_wavelength_m: float


def sousa_case(model: str, with_sheath: bool) -> SousaResult:
    """Radial capacity of the de Sousa pipe (Sævik-Thorsen Table 4 inputs)."""
    alpha = math.radians(35.0)
    n_tot = 47 + 48
    sec = rectangular_wire_section(205e9, 0.3, 2e-3, 7e-3)
    r_mean = 0.5 * ((0.136 / 2 - 1e-3) + (0.138 / 2 - 1e-3))
    tape = TapeLayer(
        name="aramid",
        E_Pa=1480e6,  # Table 4 as printed
        area_m2=0.6e-3 * 150e-3,
        lay_angle_rad=math.radians(84.0),
        n_plies=2,
        r_m=0.1384 / 2 - 0.3e-3,
        sigma_ult_Pa=313e6,
        source="[SaevikThorsen2017] Table 4",
    )
    e_s, t_s, sy_s = 750e6, 5e-3, 20e6
    c = tape_foundation_stiffness(
        n_tot, r_mean, alpha, tape.n_plies, tape.E_Pa * tape.area_m2, tape.lay_angle_rad
    )
    hoop = tape.hoop_strength_N_m
    if with_sheath:
        c += sheath_foundation_stiffness(n_tot, r_mean, alpha, e_s, t_s)
        hoop += min(e_s * tape.ultimate_strain, sy_s) * t_s
    if model == "curved":
        p2, hw = radial_buckling_load_curved(n_tot, r_mean, alpha, sec, c)
    else:
        p2 = radial_buckling_load_straight(n_tot, alpha, sec, c)
        hw = math.pi * (sec.EI2_Nm2 / c) ** 0.25
    p3 = radial_capacity_failure_load(hoop, math.tan(alpha) ** 2 / r_mean)
    p4 = wire_yield_load(n_tot, alpha, 1260e6, sec.area_m2)
    return SousaResult(p2, p3, p4, combined_radial_load(p2, p3, p4), hw)


def sousa_checks() -> list[Check]:
    straight = sousa_case("straight", True)
    tape_only = sousa_case("straight", False)
    return [
        Check(
            "ST17-P4",
            "Sousa P4 (Eq. 34) [kN]",
            SOUSA_P4_N / 1e3,
            straight.P4_N / 1e3,
            5e-3,
        ),
        Check(
            "ST17-P3t",
            "Sousa P3 tape only (Eq. 33) [kN]",
            SOUSA_P3_TAPE_N / 1e3,
            tape_only.P3_N / 1e3,
            0.06,
        ),
        Check(
            "ST17-P3ts",
            "Sousa P3 tape and sheath (Eq. 33) [kN]",
            SOUSA_P3_TAPE_SHEATH_N / 1e3,
            straight.P3_N / 1e3,
            0.06,
        ),
        Check(
            "ST17-SOUSA",
            "Sousa P2* straight beam, tape and sheath, vs test mean [kN]",
            SOUSA_TEST_N / 1e3,
            straight.P2_star_N / 1e3,
            0.10,
        ),
        Check(
            "ST17-SOUSA-cb",
            "Sousa P2* curved beam, tape and sheath, vs test mean [kN]",
            SOUSA_TEST_N / 1e3,
            sousa_case("curved", True).P2_star_N / 1e3,
            math.nan,
        ),
    ]


def sousa_table5() -> list[tuple[str, float, float, float, float]]:
    """``(assumption, P2 published, P2 here, P2* published, P2* here)``, normalised."""
    rows = []
    for (model, sheath), (p2_pub, star_pub) in SOUSA_TABLE5.items():
        res = sousa_case(model, sheath)
        label = (
            f"{'CB' if model == 'curved' else 'SB'} tape{' + sheath' if sheath else ''}"
        )
        rows.append(
            (
                label,
                p2_pub,
                res.P2_N / SOUSA_TEST_N,
                star_pub,
                res.P2_star_N / SOUSA_TEST_N,
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# Yang et al. (2015) Eq. 9 and Malta et al. (2022)                            #
# --------------------------------------------------------------------------- #
YANG_TAPE_FAIL_N = 1.84e6
MALTA_RK8_N = 241.69e3


def yang_check() -> Check:
    """Tape failure, 9.5-inch pipe: tape 2.4 mm, 261 MPa, outer armour OD 305.3 mm."""
    r_tape = 0.3053 / 2 + 1.2e-3
    hoop = 261e6 * 2.4e-3
    p3 = radial_capacity_failure_load(hoop, math.tan(math.radians(30.0)) ** 2 / r_tape)
    return Check(
        "YANG15", "Eq. 9 tape failure load [MN]", YANG_TAPE_FAIL_N / 1e6, p3 / 1e6, 0.03
    )


def malta_sheath_only() -> SousaResult:
    """6-inch pipe of Malta et al. restrained by the 7 mm HDPE sheath only."""
    layers = ((0.2065, 36.0, 50), (0.2145, 38.0, 52))  # OD, lay angle, wires
    t, w = 4e-3, 8e-3
    sec = rectangular_wire_section(207e9, 0.3, t, w)
    e_s, t_s, sy_s = 0.571e9, 7e-3, 20.74e6
    n_tot = sum(n for _, _, n in layers)
    radii = [od / 2 - t / 2 for od, _, _ in layers]
    r_mean = sum(n * r for (_, _, n), r in zip(layers, radii, strict=True)) / n_tot
    shares = []
    for _, a_deg, n in layers:
        shares.append(n * math.cos(math.radians(a_deg)) ** 3)
    s_tot = sum(shares)
    p2 = p4 = 0.0
    tan2_over_r = 0.0
    for (_, a_deg, n), r, s in zip(layers, radii, shares, strict=True):
        a = math.radians(a_deg)
        c = sheath_foundation_stiffness(n_tot, r_mean, a, e_s, t_s)
        p2 += radial_buckling_load_straight(n, a, sec, c)
        p4 += wire_yield_load(n, a, 650e6, sec.area_m2)
        tan2_over_r += (s / s_tot) * math.tan(a) ** 2 / r
    p3 = radial_capacity_failure_load(sy_s * t_s, tan2_over_r)
    c0 = sheath_foundation_stiffness(n_tot, r_mean, math.radians(36.0), e_s, t_s)
    return SousaResult(
        p2,
        p3,
        p4,
        combined_radial_load(p2, p3, p4),
        math.pi * (sec.EI2_Nm2 / c0) ** 0.25,
    )


def all_checks() -> list[Check]:
    malta = malta_sheath_only()
    return (
        saevik_table3_checks()
        + li_vaz_checks()
        + sousa_checks()
        + [
            yang_check(),
            Check(
                "MALTA22",
                "Sheath-only P2* (SB) vs FE RK8, tape cut 2000 mm [kN]",
                MALTA_RK8_N / 1e3,
                malta.P2_star_N / 1e3,
                math.nan,
            ),
        ]
    )


def main() -> None:
    print(
        f"{'ID':16s} {'quantity':62s} {'published':>10s} {'here':>10s} {'error':>8s}  status"
    )
    for c in all_checks():
        status = (
            "report" if math.isnan(c.tolerance) else ("PASS" if c.passed else "FAIL")
        )
        print(
            f"{c.ident:16s} {c.quantity:62s} {c.published:10.4g} {c.computed:10.4g} "
            f"{100 * c.error:7.2f}%  {status}"
        )
    print("\nSaevik-Thorsen Table 5 (normalised by the 256 kN test mean):")
    print(
        f"{'assumption':18s} {'P2 pub':>7s} {'P2 here':>8s} {'P2* pub':>8s} {'P2* here':>9s}"
    )
    for label, p2p, p2h, sp, sh in sousa_table5():
        print(f"{label:18s} {p2p:7.2f} {p2h:8.2f} {sp:8.2f} {sh:9.2f}")
    lb = li_vaz_lower_bound()
    print(
        "\nLi-Vaz wire: Saevik P1 = {P1_N:.0f} N = {P1_over_Pcr:.3f} Pcr; "
        "FE limit >= {min_FE_ratio:.2f} Pcr".format(**lb)
    )
    m = malta_sheath_only()
    print(
        f"Malta sheath only: P2 {m.P2_N / 1e3:.0f} kN, P3 {m.P3_N / 1e3:.0f} kN, "
        f"P4 {m.P4_N / 1e3:.0f} kN, P2* {m.P2_star_N / 1e3:.0f} kN, "
        f"half-wave {m.half_wavelength_m:.3f} m"
    )


if __name__ == "__main__":
    main()
