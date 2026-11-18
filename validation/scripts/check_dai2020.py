# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Dai, Sævik and Ye (2020) steel-tube umbilical in a bellmouth (rows DAI20-*).

The only measured curvature history inside a bellmouth that is published
with the rig geometry (Dai et al. 2020, Marine Structures 72, 102724): a 9 m
steel-tube umbilical at 200 kN, rocked against a bellmouth of constant
curvature 0.1 1/m (length 3 m), with fibre Bragg gratings inside the centre
tube and two helical tubes at station S2 (0.675 m from the bellmouth
entrance) and S3 (0.352 m from the bellmouth end). Targets and every assumed
input were registered before any model output (doc/benchmarks.md, Dai
section).

DAI20-KAPPA (the rig beam of the SV11 harness in this bellmouth). The
umbilical's bending stiffness is not published, so it is identified on one
number, the S2 curvature at 4 deg; the angles at which S2 reaches 90 % and S3
50 % of the bellmouth curvature are then predicted. Linear elastic beam
(effective tension 200 kN, no gravity, small slopes), clamped at the end
fitting 2.1 m before the entrance, free between the fitting and the entrance
(no inlet tube), 3 mm entrance gap, 9 m of umbilical and 2.2 m of rigid
arrangement to a pin; the tower turns the fitting and the bellmouth about the
entrance (assumed).

DAI20-COULOMB (stated, not gated). The sectional full-slip range of a helical
tube (direction-tied law, signed capstan bound) with the source's own contact
line forces (its Table 3, radial and radial-plus-hoop models), mu = 0.2 as
assumed by the source, tube steel 207 GPa (assumed), tube sections from its
Table 1, against the measured "Coulomb friction strain range" (Tables 4-5).

Usage::

    python validation/scripts/check_dai2020.py [--json OUT]

Thread counts are set to one.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from functools import lru_cache
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "validation" / "paper_data"
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_saevik2011 as sv  # noqa: E402

# Rig, [Dai2020] Sec. 2.1, 2.3, Figs 1, 8 and 14 ------------------------------
TENSION_N = 200e3  # results reported at 200 kN only (p. 3)
KAPPA_BELL = 0.1  # constant bellmouth curvature (p. 3; Fig. 1 R = 10 m)
BELL_LENGTH_M = 3.0  # Fig. 1, L = 3 m
GAP_M = 0.003  # entrance gap of the source's model (p. 7: 1-8 mm immaterial)
INLET_M = 2.1  # end fitting to entrance, read from the Fig. 14 axis (assumed)
UMBILICAL_M = 9.0  # umbilical between the end fittings (Fig. 14 axis 0-9 m)
RIGID_RIGHT_M = 2.2  # adapter to the tensioner pin, Fig. 1 sketch (assumed)
X_S2 = 0.675  # from the entrance (Sec. 2.3)
X_S3 = BELL_LENGTH_M - 0.352  # 0.352 m from the bellmouth end (Sec. 2.3)

RIG = sv.Rig(
    x_anchor_m=-INLET_M,
    free_length_m=UMBILICAL_M,
    rigid_right_m=RIGID_RIGHT_M,
    gap_m=GAP_M,
    bell_end_m=BELL_LENGTH_M,
    kappa_b0=KAPPA_BELL,
    kappa_b1=0.0,
    inlet_bounded=False,
    x_rotation_m=0.0,
)

# Registered targets (doc/benchmarks.md) --------------------------------------
IDENTIFY_ANGLE_DEG = 4.0
S2_LEVEL, S3_LEVEL = 0.9, 0.5  # fractions of the bellmouth curvature
ANGLE_BAND_DEG = 1.0

# Tubes, [Dai2020] Table 1 (outer diameter, thickness, radial location), 7 deg --
LAY_DEG = 7.0
MU = 0.2  # assumed by the source (p. 10)
E_STEEL_PA = 207e9  # not printed; assumed
TUBES = {
    # name: (count, radial location [m], outer diameter [m], thickness [m])
    "large": (3, 0.03925, 0.0284, 0.0015),
    "small": (9, 0.0445, 0.0179, 0.0026),
}
# Contact line forces [N/m], [Dai2020] Table 3: radial model (tube-inner core,
# tube-mid sheath) and radial plus constrained hoop model (the two hoop rows
# counted, as the source's "+13 %" for the large tube requires).
LINE_FORCE_N_M = {
    "radial": {"large": 27000.0 + 22000.0, "small": 6000.0 + 2000.0},
    "hoop": {
        "large": 4500.0 + 11000.0 + 20000.0 + 20000.0,
        "small": 550.0 + 6350.0 + 20000.0 + 20000.0,
    },
}
MEASURED_MICROSTRAIN = {"large": 200.0, "small": 158.0}  # Tables 4 and 5
SOURCE_2D_MICROSTRAIN = {  # Table 4 (the source's Eq. 14 on Table 3)
    "radial": {"large": 374.0, "small": 70.4},
    "hoop": {"large": 424.0, "small": 414.0},
}


def measured_fig9() -> dict[str, np.ndarray]:
    """Digitised Fig. 9: per station, rows (angle, kappa_mid, lo, hi)."""
    lines = [
        ln
        for ln in (DATA / "dai2020_fig9_curvature.csv")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln and not ln.startswith("#")
    ]
    out: dict[str, list[list[float]]] = {"S2": [], "S3": []}
    for r in csv.DictReader(lines):
        out[r["station"]].append(
            [
                float(r["angle_deg"]),
                float(r["kappa_mid"]),
                float(r["kappa_lo"]),
                float(r["kappa_hi"]),
            ]
        )
    return {k: np.array(v) for k, v in out.items()}


def _crossing(angle: np.ndarray, value: np.ndarray, level: float) -> float:
    """First angle (increasing) at which ``value`` reaches ``level``."""
    above = np.nonzero(value >= level)[0]
    if above.size == 0 or above[0] == 0:
        return math.nan
    i = int(above[0])
    return float(
        angle[i - 1]
        + (level - value[i - 1]) * (angle[i] - angle[i - 1]) / (value[i] - value[i - 1])
    )


def measured_targets() -> dict[str, float]:
    """Measured observables, each the mean of the two rocking directions."""
    fig = measured_fig9()
    s2, s3 = fig["S2"], fig["S3"]

    def sides(rows: np.ndarray, sign: int) -> tuple[np.ndarray, np.ndarray]:
        # S2 and S3 curvature is positive at negative angles (as plotted)
        keep = rows[:, 0] * sign > 0
        a = np.abs(rows[keep, 0])
        k = -sign * rows[keep, 1]
        order = np.argsort(a)
        return a[order], k[order]

    k4 = [float(np.interp(IDENTIFY_ANGLE_DEG, *sides(s2, s))) for s in (-1, 1)]
    a2 = [_crossing(*sides(s2, s), S2_LEVEL * KAPPA_BELL) for s in (-1, 1)]
    a3 = [_crossing(*sides(s3, s), S3_LEVEL * KAPPA_BELL) for s in (-1, 1)]
    return {
        "s2_kappa_at_4deg": float(np.mean(k4)),
        "s2_kappa_at_4deg_sides": k4,  # type: ignore[dict-item]
        "s2_angle_90": float(np.mean(a2)),
        "s2_angle_90_sides": a2,  # type: ignore[dict-item]
        "s3_angle_50": float(np.mean(a3)),
        "s3_angle_50_sides": a3,  # type: ignore[dict-item]
    }


def _elastic_beam(ei: float, h: float = 0.02) -> sv.Beam:
    k = np.linspace(0.0, 1.0, 5)
    return sv.Beam(PchipInterpolator(k, ei * k), ei, TENSION_N, h=h, rig=RIG)


def station_curvature(
    ei: float, angles_deg: np.ndarray, h: float = 0.02
) -> tuple[np.ndarray, np.ndarray, float]:
    """|kappa| at S2 and S3 along a monotonic rocking sweep (elastic: path free)."""
    beam = _elastic_beam(ei, h)
    w = None
    s2, s3 = [], []
    residual = 0.0
    previous = 0.0
    for a in angles_deg:
        steps = max(1, int(math.ceil(abs(a - previous) / 0.25)))
        for j in range(1, steps + 1):
            w, k = beam.solve(math.radians(previous + (a - previous) * j / steps), w)
        previous = float(a)
        residual = max(residual, beam.residual)
        s2.append(abs(float(np.interp(X_S2, beam.x, k))))
        s3.append(abs(float(np.interp(X_S3, beam.x, k))))
    return np.array(s2), np.array(s3), residual


@lru_cache(maxsize=1)
def kappa_check(h: float = 0.02) -> dict:
    """DAI20-KAPPA: identify EI on S2 at 4 deg, predict the two contact angles."""
    meas = measured_targets()

    def miss(log_ei: float) -> float:
        s2, _s3, _r = station_curvature(10.0**log_ei, np.array([IDENTIFY_ANGLE_DEG]), h)
        return float(s2[0]) - meas["s2_kappa_at_4deg"]

    log_ei = brentq(miss, 3.0, 7.0, xtol=1e-4)
    ei = 10.0**log_ei
    angles = np.arange(0.25, 20.0 + 1e-9, 0.25)
    s2, s3, residual = station_curvature(ei, angles, h)
    a2 = _crossing(angles, s2, S2_LEVEL * KAPPA_BELL)
    a3 = _crossing(angles, s3, S3_LEVEL * KAPPA_BELL)
    err2 = a2 - meas["s2_angle_90"]
    err3 = a3 - meas["s3_angle_50"]
    return {
        "EI_Nm2": ei,
        "measured": meas,
        "s2_angle_90": a2,
        "s3_angle_50": a3,
        "error_s2_deg": err2,
        "error_s3_deg": err3,
        "beam_residual": residual,
        "gate": bool(abs(err2) <= ANGLE_BAND_DEG and abs(err3) <= ANGLE_BAND_DEG),
        "curve": {
            "angle_deg": angles.tolist(),
            "S2": s2.tolist(),
            "S3": s3.tolist(),
        },
    }


def coulomb_check() -> dict:
    """DAI20-COULOMB: sectional full-slip range of each tube [microstrain]."""
    from cablexsec.analysis.fatigue import (  # noqa: PLC0415
        FatigueLayer,
        wire_stress_field,
    )
    from cablexsec.materials.sn_curves import SN_CURVES  # noqa: PLC0415
    from cablexsec.mechanics.mkappa import HelicalLayerSpec  # noqa: PLC0415

    alpha = math.radians(LAY_DEG)
    kappa = np.concatenate(
        [
            np.linspace(0.0, KAPPA_BELL, 61),
            np.linspace(KAPPA_BELL, -KAPPA_BELL, 121)[1:],
            np.linspace(-KAPPA_BELL, KAPPA_BELL, 121)[1:],
            np.linspace(KAPPA_BELL, -KAPPA_BELL, 121)[1:],
        ]
    )
    last = slice(kappa.size - 241, kappa.size)
    out: dict = {}
    for model, forces in LINE_FORCE_N_M.items():
        out[model] = {}
        for tube, (count, radius, od, t) in TUBES.items():
            area = math.pi * (od - t) * t
            width = 2.0 * math.pi * radius / count
            # line load per unit tube length q = P (2 pi r / n) cos(alpha)
            pressure = forces[tube] / (width * math.cos(alpha))
            spec = HelicalLayerSpec(
                n_wires=count,
                r_helix_m=radius,
                lay_angle_rad=alpha,
                E_wire_Pa=E_STEEL_PA,
                A_wire_m2=area,
                xi=1.0,
                mu=MU,
                F_n_axi_N=0.0,
                contact_pressure_Pa=pressure,
            )
            layer = FatigueLayer(
                spec=spec,
                sn_bulk=SN_CURVES["saevik_2011_flat_steel"],
                sigma_UTS_Pa=1.0e9,
                name=f"{tube}_tubes",
            )
            field = wire_stress_field(layer, kappa, np.zeros_like(kappa))
            stress_range = float(np.max(np.ptp(field.mean_Pa[:, last], axis=1)))
            strain = stress_range / E_STEEL_PA * 1e6
            out[model][tube] = {
                "microstrain": strain,
                "vs_measured": strain / MEASURED_MICROSTRAIN[tube] - 1.0,
                "vs_source_2d": strain / SOURCE_2D_MICROSTRAIN[model][tube] - 1.0,
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    kap = kappa_check()
    cou = coulomb_check()
    print(
        f"DAI20-KAPPA EI {kap['EI_Nm2'] / 1e3:.1f} kN m2; S2 90 %: "
        f"{kap['s2_angle_90']:.2f} vs {kap['measured']['s2_angle_90']:.2f} deg; "
        f"S3 50 %: {kap['s3_angle_50']:.2f} vs {kap['measured']['s3_angle_50']:.2f} "
        f"deg; gate {kap['gate']}; residual {kap['beam_residual']:.1e}"
    )
    for model, tubes in cou.items():
        for tube, v in tubes.items():
            print(
                f"DAI20-COULOMB {model} {tube}: {v['microstrain']:.1f} microstrain, "
                f"{100 * v['vs_measured']:+.1f} % vs measured, "
                f"{100 * v['vs_source_2d']:+.1f} % vs source 2D"
            )
    if args.json is not None:
        args.json.write_text(
            json.dumps({"kappa": kap, "coulomb": cou}, indent=1), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
