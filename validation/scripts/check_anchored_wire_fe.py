# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Anchored wire at uniform curvature against two published FE studies.

Reported, not gated (doc/benchmarks.md, "Anchored wire at uniform curvature
against published FE"). One armour wire, fixed at an end fitting, in a pipe
bent to a uniform curvature from the straight state; the clamp-layer solver
(:func:`cablexsec.mechanics.clamp_layer.anchored_wire_response`: rigid stick,
Coulomb bound, axial slip only, ``u = 0`` at the anchors) against

* Zhu and Tan (2014), OMAE2014-23928, Table 1, Table 2 and Figs 4-8
  (normalised stress, 3D FE and the source's analytical shifting model);
* Dong et al. (2025), Marine Structures 102, 103761, Table 1, Figs 9, 14
  and 25 (beam-on-shell FE).

The single parameter of the normalised problem is ``a = kappa / kappa_c``,
the stick stress gradient at the neutral axis over the friction gradient.
Zhu and Tan do not print the friction coefficient; ``a`` follows from their
own Table 2 (Eqs. 13 and 16 contain the stick-to-slip displacement ratio
``U = u_stk / u_max = 8 a / pi^2``; the three non-trivial rows give the same
``U``) and is cross-checked by their Fig. 14 (free-field amplitude). Dong et
al. print every input; the friction line load is their Eq. 38.

Usage::

    python validation/scripts/check_anchored_wire_fe.py [--spring] [--json OUT]

``--spring`` adds the elastic-interface sensitivity of the Zhu-Tan case
(along-wire chain, ``k_hat`` grid; about a minute). Thread counts are set to
one.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from cablexsec.mechanics.clamp_layer import (  # noqa: E402
    anchored_wire_response,
    uniform_curvature_force,
)

FloatArray = NDArray[np.float64]
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "validation" / "paper_data"
ZT_CSV = DATA / "zhu_tan2014_figs4_8.csv"
DONG_CSV = DATA / "dong2025_figs9_14_25.csv"

# --------------------------------------------------------------------------- #
# Zhu and Tan (2014)                                                          #
# --------------------------------------------------------------------------- #
#: Table 1 (Young's modulus printed as 210 MPa; 210 GPa is meant).
ZT_E_PA = 210e9
ZT_LAY_RAD = math.radians(38.0)
ZT_R_HELIX_M = 0.254
ZT_BEND_RADIUS_M = 180.0
#: Table 2: initial hoop position, first and second shifting angle [deg].
ZT_TABLE2 = ((0.0, 0.0, 0.0), (45.0, 35.8, 0.0), (90.0, 90.0, 7.3),
             (135.0, 45.0, 13.7), (180.0, 0.0, 0.0))  # fmt: skip


def zt_stick_amplitude_pa() -> float:
    """Stick axial stress amplitude ``E cos^2 a r / R`` (Eq. 1) [Pa]."""
    return ZT_E_PA * math.cos(ZT_LAY_RAD) ** 2 * ZT_R_HELIX_M / ZT_BEND_RADIUS_M


def zt_shift_angles(theta_i: float, ratio: float) -> tuple[float, float]:
    """Shifting angles ``(theta_m, theta_n)`` of Eqs. 13-18 [rad].

    ``theta_i`` from the compressive extreme fibre; ``ratio`` the amplitude
    ratio ``U`` of the stick displacement (Eq. 2) to the full-slip one, so
    that ``u_stk(theta_i) / u_max = -U sin theta_i``.
    """
    pi = math.pi
    u = -ratio * math.sin(theta_i)
    disc = 8 * theta_i**2 - 24 * pi * theta_i + 16 * pi**2 + 2 * pi**2 * u
    if disc >= 0.0:
        return (pi - theta_i) - 0.25 * math.sqrt(disc), 0.0
    disc_n = 8 * theta_i**2 - 40 * pi * theta_i + 48 * pi**2 - 2 * pi**2 * u
    return pi - theta_i, (theta_i - 2 * pi) + 0.25 * math.sqrt(disc_n)


def zt_ratio_from_table2() -> list[float]:
    """``U`` that reproduces each non-trivial Table 2 row (bisection)."""
    out = []
    for ti, tm, tn in ZT_TABLE2[1:4]:
        target = math.radians(tm if tn == 0.0 else tn)
        index = 0 if tn == 0.0 else 1

        def resid(
            ratio: float, ti: float = ti, index: int = index, target: float = target
        ) -> float:
            return zt_shift_angles(math.radians(ti), ratio)[index] - target

        lo, hi = 3.0, 4.5
        f_lo = resid(lo)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            f_mid = resid(mid)
            if (f_mid > 0.0) == (f_lo > 0.0):
                lo, f_lo = mid, f_mid
            else:
                hi = mid
        out.append(0.5 * (lo + hi))
    return out


def a_from_ratio(ratio: float) -> float:
    """``a = pi^2 U / 8`` (rigid stick, full-slip triangle far field)."""
    return math.pi**2 * ratio / 8.0


def zt_analytical_field(
    theta_i: float, theta_m: float, theta_n: float, samples: int = 60001
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Shifted friction field of Eq. 9 and its stress (Eq. 10), normalised.

    Returns the hoop angle (from the compressive fibre) over three turns, the
    friction sign and the stress over the free-field peak; the constant is
    set by the unshifted far field (stress ``+1`` at ``5 pi``).
    """
    pi = math.pi
    th = np.linspace(theta_i, theta_i + 6 * pi, samples)
    f = np.where(
        th <= pi - theta_m,
        1.0,
        np.where(
            th <= 2 * pi + theta_n,
            -1.0,
            np.where(th <= 3 * pi, 1.0, (-1.0) ** np.floor(th / pi)),
        ),
    )
    dth = th[1] - th[0]
    s = np.concatenate(([0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * dth))) * 2 / pi
    s += 1.0 - np.interp(5 * pi, th, s)
    return th, f, s


def coulomb_violation(
    theta: FloatArray, friction_sign: FloatArray, stress: FloatArray, a: float
) -> float:
    """Arc [rad] over which friction does not oppose the relative slip.

    ``stress`` normalised by the full-slip peak, so the stick stress is
    ``-(2 a / pi) cos theta`` (``theta`` from the compressive fibre) and the
    slip of the wire relative to the core, from the anchor, is the integral
    of the difference. In slip the stress gradient (``friction_sign``) must
    have the sign of the slip.
    """
    stick = -(2 * a / math.pi) * np.cos(theta)
    d = stress - stick
    slip = np.concatenate(([0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(theta))))
    moving = np.abs(slip) > 1e-6 * max(float(np.abs(slip).max()), 1e-300)
    bad = moving & (np.sign(friction_sign) != np.sign(slip))
    return float(np.sum(bad[1:] * np.diff(theta)))


# --------------------------------------------------------------------------- #
# Dong et al. (2025)                                                          #
# --------------------------------------------------------------------------- #
#: Table 1. The helical angle 55 deg is measured from the hoop direction
#: (Fig. 2; Eqs. 3, 12, 46): the lay angle from the pipe axis is 35 deg.
DONG = {
    "E_Pa": 210e9,
    "width_m": 6e-3,
    "thickness_m": 3e-3,
    "r_helix_m": 0.10,
    "alpha0_rad": math.radians(55.0),
    "pipe_strain": 1e-3,
    "curvature_1_m": 0.005,
    "mu_inner": 0.10,
    "mu_outer": 0.10,
    "p_outer_Pa": 2.55e6,
}


def dong_parameters() -> dict[str, float]:
    """Normalisation of the Dong case (Eq. 38 friction, mean tension)."""
    d = DONG
    area = d["width_m"] * d["thickness_m"]
    s2 = math.sin(d["alpha0_rad"]) ** 2
    c2 = math.cos(d["alpha0_rad"]) ** 2
    sigma_t = d["E_Pa"] * d["pipe_strain"] * s2
    tension = sigma_t * area
    # Eq. 38: f = mu_i T_a kappa_03 + (mu_i + mu_o) p_o w, kappa_03 = cos^2 a0 / R
    f = (
        d["mu_inner"] * tension * c2 / d["r_helix_m"]
        + (d["mu_inner"] + d["mu_outer"]) * d["p_outer_Pa"] * d["width_m"]
    )
    k = math.cos(d["alpha0_rad"]) / d["r_helix_m"]  # helix angle per arc length
    stick = d["E_Pa"] * s2 * d["curvature_1_m"] * d["r_helix_m"]
    sigma_u = f / area / k  # stress per radian of helix angle at the bound
    return {
        "friction_N_m": f,
        "tension_stress_Pa": sigma_t,
        "stick_amplitude_Pa": stick,
        "sigma_u_Pa": sigma_u,
        "a": stick / sigma_u,
    }


# --------------------------------------------------------------------------- #
# CableXSec                                                                   #
# --------------------------------------------------------------------------- #
def anchored_profile(
    theta0_tensile: float,
    a: float,
    *,
    pitches: float = 4.0,
    segments_per_pitch: int = 720,
    step: float = 0.01,
) -> tuple[FloatArray, FloatArray]:
    """Wire force ``n`` of one wire anchored at both ends, monotonic ``0 -> a``.

    ``theta0_tensile`` is the anchor azimuth from the tensile extreme fibre;
    returns the helix angle from the anchor [rad] and ``n`` (full-slip peak
    ``pi / 2``).
    """
    m = round(pitches * segments_per_pitch)
    h = 2 * math.pi / segments_per_pitch
    phi = (np.arange(m) + 0.5) * h
    e_hat = np.cos(theta0_tensile + phi)
    load = np.linspace(0.0, a, max(1, math.ceil(a / step)) + 1)[1:]
    record = np.zeros(load.size, dtype=bool)
    record[-1] = True
    n, _, _ = anchored_wire_response(e_hat, load, h, record=record)
    return phi, n[-1]


def zt_cablexsec(a: float) -> dict[float, dict[str, float]]:
    """Normalised anchored stress for the five Zhu-Tan wires."""
    out = {}
    for ti, _, _ in ZT_TABLE2:
        phi, n = anchored_profile(math.radians(ti) - math.pi, a)
        s = n / (math.pi / 2)
        first = phi < 2 * math.pi
        k = int(np.argmax(s[first]))
        j = int(np.argmin(s[first]))
        out[ti] = {
            "anchor": float(s[0]),
            "peak": float(s[first][k]),
            "peak_hoop_deg": ti + math.degrees(float(phi[k])),
            "min": float(s[first][j]),
            "min_hoop_deg": ti + math.degrees(float(phi[j])),
        }
    return out


def dong_cablexsec(theta0_deg: float, pitches: float = 1.0) -> dict[str, float]:
    """Axial stress [MPa] of a Dong wire (one pitch, both ends fixed)."""
    p = dong_parameters()
    phi, n = anchored_profile(
        math.radians(theta0_deg) - math.pi, p["a"], pitches=pitches
    )
    stress = (p["tension_stress_Pa"] + p["sigma_u_Pa"] * n) / 1e6
    x = phi / (2 * math.pi)
    k = int(np.argmax(stress))
    return {
        "anchor": float(stress[0]),
        "peak": float(stress[k]),
        "peak_X": float(x[k]),
        "peak_hoop_deg": theta0_deg + 360.0 * float(x[k]),
        "at_X_0p75": float(np.interp(0.75, x, stress)),
        "free_field_at_anchor": float(
            (
                p["tension_stress_Pa"]
                + p["sigma_u_Pa"]
                * uniform_curvature_force(
                    np.array([math.radians(theta0_deg) - math.pi]), p["a"]
                )[0]
            )
            / 1e6
        ),
    }


def peak_versus_a(
    grid: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 4.92, 10.0, 20.0, 50.0),
    segments_per_pitch: int = 720,
    step: float = 0.01,
) -> list[tuple[float, float, float]]:
    """Neutral-axis wire: normalised anchor and peak stress against ``a``."""
    rows = []
    for a in grid:
        phi, n = anchored_profile(
            -math.pi / 2,
            a,
            pitches=max(2.0, math.ceil(0.25 * a)),
            segments_per_pitch=segments_per_pitch,
            step=step,
        )
        far = float(np.abs(uniform_curvature_force(np.array([0.0]), a))[0])
        first = phi < 2 * math.pi
        rows.append((a, float(n[0] / far), float(n[first].max() / far)))
    return rows


# --------------------------------------------------------------------------- #
# Digitised data                                                              #
# --------------------------------------------------------------------------- #
def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    return list(csv.DictReader(lines))


def zt_digitised() -> dict[tuple[float, str], tuple[FloatArray, FloatArray]]:
    out: dict[tuple[float, str], list[tuple[float, float]]] = {}
    for row in read_csv(ZT_CSV):
        key = (float(row["theta_i_deg"]), row["series"])
        out.setdefault(key, []).append(
            (float(row["hoop_deg"]), float(row["normalised_stress"]))
        )
    return {
        k: (np.array([p[0] for p in v]), np.array([p[1] for p in v]))
        for k, v in out.items()
    }


def dong_digitised() -> dict[tuple[str, str], tuple[FloatArray, FloatArray]]:
    out: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for row in read_csv(DONG_CSV):
        key = (row["figure"], row["series"])
        out.setdefault(key, []).append((float(row["x"]), float(row["stress_MPa"])))
    return {
        k: (np.array([p[0] for p in v]), np.array([p[1] for p in v]))
        for k, v in out.items()
    }


def curve_features(
    x: FloatArray, y: FloatArray, window: float, start_span: float
) -> dict[str, float]:
    """Start value (median over ``start_span``) and the peak within ``window``."""
    i = np.argsort(x)
    x, y = x[i], y[i]
    start = float(np.median(y[x <= x[0] + start_span]))
    w = x <= x[0] + window
    k = int(np.argmax(y[w]))
    return {"start": start, "peak": float(y[w][k]), "peak_x": float(x[w][k])}


# --------------------------------------------------------------------------- #
# Elastic-interface sensitivity (Zhu-Tan)                                     #
# --------------------------------------------------------------------------- #
def zt_spring_sensitivity(
    a: float, k_hats: tuple[float, ...] = (0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 10.0)
) -> list[dict[str, float]]:
    """Along-wire chain with an elastic interface ``k_hat`` (``"spring"``).

    Reports the far-field peak force, the RMS distance of the normalised far
    field from a cosine and the normalised anchor/peak stress of the 45, 90
    and 135 deg wires. A sensitivity: the source's tangential stiffness is
    not printed.
    """
    from cablexsec.mechanics.along_wire import WireChainLayer, along_wire_history

    pitch = 2 * math.pi * ZT_R_HELIX_M / math.tan(ZT_LAY_RAD)
    kappa = 1.0 / ZT_BEND_RADIUS_M
    rows = []
    for k_hat in k_hats:
        row: dict[str, float] = {"k_hat": k_hat}
        for ti in (0.0, 45.0, 90.0, 135.0):
            layer = WireChainLayer(
                name="zt",
                n_wires=1,
                r_helix_m=ZT_R_HELIX_M,
                lay_angle_rad=ZT_LAY_RAD,
                kappa_c_1_m=kappa / a,
                k_hat=k_hat,
                lam=0.0,
                force_unit_N=1.0,
                wire_area_m2=48e-6,
                ea_N=ZT_E_PA * 48e-6,
                phase0_rad=math.radians(ti) - math.pi,
            )
            x = np.array([0.0, 4 * pitch])
            curv = np.outer(np.ones(2), np.linspace(0.0, kappa, 201)[1:])
            hist = along_wire_history(
                layer, x, curv, segments_per_pitch=256, max_increment=0.05
            )
            n = hist.force[0, :, -1]
            th = hist.record_x_m / pitch * 2 * math.pi
            far = (th > 2 * math.pi) & (th < 6 * math.pi)
            peak = float(np.abs(n[far]).max())
            s = n / peak
            if ti == 0.0:
                row["far_peak"] = peak
                row["rms_vs_cosine"] = float(
                    np.sqrt(np.mean((s[far] + np.cos(th[far])) ** 2))
                )
            else:
                first = th < 2 * math.pi
                row[f"anchor_{ti:.0f}"] = float(s[0])
                row[f"peak_{ti:.0f}"] = float(s[first].max())
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Report                                                                      #
# --------------------------------------------------------------------------- #
def report(spring: bool = False) -> dict:
    ratios = zt_ratio_from_table2()
    ratio = float(np.mean(ratios))
    a_zt = a_from_ratio(ratio)
    stick = zt_stick_amplitude_pa()
    out: dict = {
        "zt": {
            "U_rows": ratios,
            "a": a_zt,
            "stick_MPa": stick / 1e6,
            "free_field_MPa": 4 * stick / (math.pi * ratio) / 1e6,
        }
    }
    viol = {}
    for ti, _, _ in ZT_TABLE2[1:4]:
        tm, tn = zt_shift_angles(math.radians(ti), ratio)
        th, f, s = zt_analytical_field(math.radians(ti), tm, tn)
        viol[ti] = math.degrees(coulomb_violation(th, f, s, a_zt))
    out["zt"]["analytical_violation_deg"] = viol
    out["zt"]["cablexsec"] = zt_cablexsec(a_zt)
    dig = zt_digitised()
    out["zt"]["fea"] = {
        ti: curve_features(*dig[(ti, "fea")], window=355.0, start_span=4.0)
        for ti, _, _ in ZT_TABLE2
    }
    x0, y0 = dig[(0.0, "fea")]
    th = np.radians(x0)
    tri = 1 - 2 * np.abs(np.mod(th, 2 * math.pi) - math.pi) / math.pi
    w = (x0 > 30) & (x0 < 1050)
    out["zt"]["fea_far_field_rms"] = {
        "cosine": float(np.sqrt(np.mean((y0[w] + np.cos(th[w])) ** 2))),
        "triangle": float(np.sqrt(np.mean((y0[w] - tri[w]) ** 2))),
    }
    out["peak_vs_a"] = peak_versus_a()
    p = dong_parameters()
    out["dong"] = {"parameters": p}
    out["dong"]["cablexsec"] = {t: dong_cablexsec(t) for t in (0, 45, 90, 135, 180)}
    dd = dong_digitised()
    out["dong"]["fig9_fe"] = curve_features(*dd[("9", "numerical")], 0.25, 0.004)
    out["dong"]["fig14_fe"] = curve_features(*dd[("14", "numerical")], 0.75, 0.004)
    out["dong"]["fig14_theory"] = curve_features(
        *dd[("14", "theoretical")], 0.75, 0.006
    )
    x14, y14 = dd[("14", "numerical")]
    out["dong"]["fig14_fe_at_0p75"] = float(np.interp(0.745, x14, y14))
    fig25 = {}
    for t in ("0", "45", "90", "135", "180"):
        fig25[t] = curve_features(*dd[("25", t)], 180.0, 4.0)
    xa, ya = dd[("25", "0")]
    fig25["free_max"] = float(ya[(xa > 150) & (xa < 210)].max())
    fig25["free_min"] = float(ya[(xa > 330) & (xa < 400)].min())
    out["dong"]["fig25_fe"] = fig25
    if spring:
        out["zt"]["spring"] = zt_spring_sensitivity(a_zt)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spring", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    out = report(spring=args.spring)
    text = json.dumps(out, indent=1, default=float)
    if args.json is not None:
        args.json.write_text(text, encoding="utf-8")
    sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
