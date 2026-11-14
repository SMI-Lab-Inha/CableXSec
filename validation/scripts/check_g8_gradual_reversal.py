# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Gap G8: gradual stick-to-slip transition, public-data checks.

Run as ``python validation/scripts/check_g8_gradual_reversal.py``. Targets were
registered before the runs (doc/benchmarks.md, section "Gradual
stick-to-slip transition (gap G8)"); nothing is fitted.

1. Hu et al. (2022) copper conductor, third reversed cycle at +-0.372 1/m:
   loop area (3.185 N m/m), maximum moment (8.18 N m, Nam et al. 2024) and
   the share of the reversal stroke in the slip transition (33.78 %, Hu
   Section 2.2). Section law: the package's ring parameters
   (``conductor_ring_parameters``) with the Karlsen-Husoy amplitude or the
   Papailiou neutral-axis cap (``cap_shape``). Two reductions: uniform
   curvature, and the test's three-point bending (span 440 mm, moment
   linear from the supports, curvature reported as 12 w / L^2, Hu Eqs. 1-2;
   by virtual work the reported curvature is 3 int_0^1 s kappa(s M) ds).
2. Menard and Cartraud (2023), six measured ranges, transferred inputs
   (mu 0.2, YAML clamping): ordered moment RMSE and cycle work with the
   default and the progressive armour law (BM1-loop is range b).
3. Witz (1996) 300-bar loop after identification on the 0-bar width
   (WITZ-ID set-up) with the default and the progressive law.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
from check_conductor_strand_slip import (
    HU_AREA,
    HU_D,
    HU_E,
    HU_KAPPA,
    HU_MU_SHEATH,
    HU_MU_WIRE,
    HU_PRESSURE,
    HU_SHEATH_E,
    HU_SHEATH_T,
    NAM_M_MAX,
    hu_rings,
)
from scipy.optimize import brentq

import cablexsec as cx
from cablexsec.analysis.curve_comparison import cycle_work, insert_cycle_crossings
from cablexsec.mechanics.core_lay import bending_factor
from cablexsec.mechanics.strand_slip import (
    conductor_bending_moment,
    conductor_ring_parameters,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "validation" / "paper_data"
HU_SHARE = 0.3378  # Hu Section 2.2: transition share of the TC-BC stroke
HU_TENSION = 100.0  # Nam et al. (2024) Section 2.1
CYCLE = np.linspace(0, 6 * math.pi, 1501)
LAST = slice(1000, 1501)


# --------------------------------------------------------------------------- #
# Hu conductor                                                                 #
# --------------------------------------------------------------------------- #
def _hu_setup() -> tuple[tuple, tuple, float]:
    rings = hu_rings()
    r_outer = 4.5 * HU_D
    params = conductor_ring_parameters(
        rings,
        HU_E,
        mu_wire=HU_MU_WIRE,
        mu_outer=HU_MU_SHEATH,
        reference_tension_N=HU_TENSION,
        external_pressure_Pa=HU_PRESSURE,
        r_outer_m=r_outer,
        fibre="centroid",
    )
    sheath = HU_SHEATH_E * math.pi / 4 * ((r_outer + HU_SHEATH_T) ** 4 - r_outer**4)
    wire = sum(
        HU_E
        * r.n_wires
        * math.pi
        * r.d_wire_m**4
        / 64
        * bending_factor(r.lay_angle_rad)
        for r in rings
    )
    return rings, params, sheath + wire


class _Stations:
    """The conductor operator (one slipping order) at many planar stations."""

    def __init__(self, n_stations: int, cap_shape: str) -> None:
        rings, params, self.ei = _hu_setup()
        self.rings = []
        for ring, p in zip(rings, params, strict=True):
            if ring.r_helix_m == 0:
                continue
            theta = 2 * math.pi * np.arange(ring.n_wires) / ring.n_wires
            c = np.abs(np.cos(theta))
            shape = c if cap_shape == "karlsen_husoy" else (2 / math.pi) * np.arcsin(c)
            eps_f = float(p.friction_amplitude(np.array([HU_TENSION]))[0, 0])
            self.rings.append(
                {
                    "b0": p.beta_m[0],
                    "b1": p.beta_m[1],
                    "cos": np.cos(theta),
                    "cap": eps_f * shape,
                    "w": HU_E
                    * ring.area_m2
                    * math.cos(ring.lay_angle_rad)
                    * ring.r_helix_m,
                    "eps": np.zeros((n_stations, ring.n_wires)),
                }
            )
        self.kappa = np.zeros(n_stations)

    def trial(self, kappa: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        moment = self.ei * kappa
        states = []
        for r in self.rings:
            ke = kappa[:, None] * r["cos"]
            pred = r["eps"] + r["b0"] * (ke - self.kappa[:, None] * r["cos"])
            centre = r["b1"] * ke
            value = np.clip(pred, centre - r["cap"], centre + r["cap"])
            states.append(value)
            moment = moment + r["w"] * (value @ r["cos"])
        return moment, states

    def commit(self, kappa: np.ndarray, states: list[np.ndarray]) -> None:
        for r, value in zip(self.rings, states, strict=True):
            r["eps"] = value
        self.kappa = kappa.copy()


def _three_point(moment_mid: np.ndarray, cap_shape: str, n_st: int = 24) -> np.ndarray:
    """Reported curvature 3 int s kappa(s M) ds for a mid-span moment history."""
    s = (np.arange(n_st) + 0.5) / n_st
    st = _Stations(n_st, cap_shape)
    out = np.empty_like(moment_mid)
    for i, m in enumerate(moment_mid):
        target = s * m
        m_old, _ = st.trial(st.kappa)
        dm = target - m_old
        lo = st.kappa + np.minimum(dm / 5.0, dm / 1e5) - 1e-12
        hi = st.kappa + np.maximum(dm / 5.0, dm / 1e5) + 1e-12
        for _ in range(42):  # bisection; the section law is monotone in kappa
            mid = 0.5 * (lo + hi)
            up = st.trial(mid)[0] > target
            hi, lo = np.where(up, mid, hi), np.where(up, lo, mid)
        k_new = 0.5 * (lo + hi)
        st.commit(k_new, st.trial(k_new)[1])
        out[i] = 3.0 * np.mean(s * k_new)
    return out


def _hu_metrics(kappa: np.ndarray, moment: np.ndarray) -> dict[str, float]:
    k, m = kappa[LAST], moment[LAST]
    i0 = int(np.argmax(k))
    i1 = i0 + int(np.argmin(k[i0:]))
    lk, lm = k[i0 : i1 + 1], m[i0 : i1 + 1]
    sel = (lk < -0.15) & (lk > -0.33)
    slope = float(np.polyfit(lk[sel], lm[sel], 1)[0])
    ok = np.abs(np.gradient(lm, lk) - slope) <= 0.10 * abs(slope)
    win = max(3, lk.size // 10)
    share = next(
        (
            float((lk[0] - lk[j]) / (lk[0] - lk[-1]))
            for j in range(lk.size - win)
            if ok[j : j + win].all()
        ),
        1.0,
    )
    return {
        "area": float(abs(np.trapezoid(m, k))),
        "M_max": float(m.max()),
        "slip_slope": slope,
        "share": share,
    }


def hu_cases() -> dict[str, dict[str, float]]:
    rings, params, ei = _hu_setup()
    kappa = HU_KAPPA * np.sin(CYCLE)
    out = {}
    for shape in ("karlsen_husoy", "papailiou"):
        moment, _ = conductor_bending_moment(
            rings,
            HU_E,
            params,
            kappa,
            np.zeros_like(kappa),
            np.full_like(kappa, HU_TENSION),
            additional_EI_Nm2=ei
            - sum(
                HU_E
                * r.n_wires
                * math.pi
                * r.d_wire_m**4
                / 64
                * bending_factor(r.lay_angle_rad)
                for r in rings
            ),
            cap_shape=shape,  # type: ignore[arg-type]
        )
        out[f"uniform, {shape}"] = _hu_metrics(kappa, moment)

        def peak(amplitude: float, shape: str = shape) -> tuple[np.ndarray, np.ndarray]:
            m_mid = amplitude * np.sin(CYCLE)
            return m_mid, _three_point(m_mid, shape)

        a0, a1 = 7.0, 8.0
        f0 = peak(a0)[1][LAST].max() - HU_KAPPA
        for _ in range(12):  # secant on the moment amplitude: kappa_rep peak
            f1 = peak(a1)[1][LAST].max() - HU_KAPPA
            if abs(f1) < 1e-5 * HU_KAPPA:
                break
            a0, a1, f0 = a1, a1 - f1 * (a1 - a0) / (f1 - f0), f1
        m_mid, k_rep = peak(a1)
        out[f"three-point, {shape}"] = _hu_metrics(k_rep, m_mid)
    return out


# --------------------------------------------------------------------------- #
# Menard and Witz                                                              #
# --------------------------------------------------------------------------- #
LAWS: dict[str, dict] = {
    "default": {},
    "progressive": {"friction_model": "progressive"},
}


def menard_cases() -> dict[str, dict[str, tuple[float, float]]]:
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for rng in "abcdef":
        d = np.genfromtxt(
            DATA / f"menard_fig16{rng}_ordered.csv", delimiter=",", names=True
        )
        k = np.asarray(d["curvature_1_per_m"], float)
        m = np.asarray(d["moment_Nm"], float)
        crossing = 0.0 if rng in "ab" else None
        measured = float(np.mean(cycle_work(k, m, crossing=crossing)))
        for tag, kw in LAWS.items():
            seq = cx.compute_mkappa(cable, np.tile(k, 2), mu=0.2, **kw).moment_Nm[
                -k.size :
            ]
            rmse = float(np.sqrt(np.mean((seq - m) ** 2)) / np.ptp(m) * 100)
            aug, _, cross = insert_cycle_crossings(k, crossing=crossing)
            work = cx.compute_mkappa(cable, np.tile(aug, 2), mu=0.2, **kw).work
            mech = (work.stored_J_per_m + work.friction_J_per_m)[-len(aug) :]
            w = float(np.mean(np.diff(mech[cross])))
            out.setdefault(tag, {})[rng] = (rmse, (w - measured) / measured * 100)
    return out


def witz_cases() -> dict[str, dict[str, float]]:
    import run_validation as rv  # noqa: PLC0415

    cable = cx.load_cable(cx.witz1996_yaml_path(), geometry_override=None)
    m0 = rv._witz_measured(DATA / "witz1996_fig5_0bar.csv")
    m3 = rv._witz_measured(DATA / "witz1996_fig6_300bar.csv")

    def loop(
        meas: dict[str, float], state: cx.SectionState, kw: dict
    ) -> tuple[float, float]:
        step = 0.0025
        hi, lo = meas["kappa_max"], meas["kappa_min"]
        kappa = np.concatenate(
            [
                np.linspace(0.0, hi, round(hi / step) + 1),
                np.linspace(hi, lo, round((hi - lo) / step) + 1)[1:],
                np.linspace(lo, hi, round((hi - lo) / step) + 1)[1:],
            ]
        )
        r = cx.compute_mkappa(cable, kappa, mu=rv._WITZ_MU, section_state=state, **kw)
        i_max, i_min = int(np.argmax(kappa)), int(np.argmin(kappa))
        kd, md = kappa[i_max : i_min + 1], r.moment_Nm[i_max : i_min + 1]
        ku, mu = kappa[i_min:], r.moment_Nm[i_min:]
        width = float(np.interp(0.0, ku, mu) - np.interp(0.0, kd[::-1], md[::-1]))
        poly = np.column_stack((np.r_[kd, ku[1:]], np.r_[md, mu[1:]]))
        return width, rv._shoelace(poly)

    def state(p0: float, loaded: bool) -> cx.SectionState:
        return cx.solve_section_state(
            cable,
            tension_N=rv._WITZ_P_INT * math.pi * rv._WITZ_R_SEAL**2 if loaded else 0.0,
            p_int_Pa=rv._WITZ_P_INT if loaded else 0.0,
            p_ext_Pa=p0,
            contact_model="bounded",
        )

    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tag, kw in LAWS.items():
            p0 = brentq(
                lambda x, kw=kw: loop(m0, state(x, False), kw)[0] - m0["width_lo"],
                1.0e4,
                1.0e7,
                xtol=1.0,
            )
            width, energy = loop(m3, state(p0, True), kw)
            out[tag] = {
                "p0_MPa": p0 / 1e6,
                "width": width,
                "energy": energy,
                "energy_per_width": energy / width,
                "width_lo": m3["width_lo"],
                "width_hi": m3["width_hi"],
                "energy_measured": m3["energy"],
            }
    return out


def main() -> None:
    print("Hu et al. (2022), third cycle at +-0.372 1/m")
    print(
        f"  measured: area {HU_AREA} N m/m, M_max {NAM_M_MAX} N m, share {HU_SHARE:.1%}"
    )
    for name, r in hu_cases().items():
        print(
            f"  {name:26s} area {r['area']:.3f} ({(r['area'] / HU_AREA - 1) * 100:+.1f}%), "
            f"M_max {r['M_max']:.2f}, slip slope {r['slip_slope']:.2f} N m^2, "
            f"share {r['share']:.1%}"
        )
    print("Menard (2023), transferred inputs: ordered RMSE [%], work error [%]")
    for tag, rows in menard_cases().items():
        print(
            f"  {tag:12s} "
            + "  ".join(f"({k}) {a:.2f} / {b:+.1f}" for k, (a, b) in rows.items())
        )
    print("Witz (1996), 300-bar loop predicted after identification at 0 bar")
    for tag, r in witz_cases().items():
        print(
            f"  {tag:12s} p0 {r['p0_MPa']:.4f} MPa, width {r['width']:.0f} N m "
            f"(measured {r['width_lo']:.0f}-{r['width_hi']:.0f}), energy "
            f"{r['energy']:.0f} J/m (measured {r['energy_measured']:.0f}), "
            f"energy/width {r['energy_per_width']:.3f} 1/m (measured "
            f"{r['energy_measured'] / r['width_lo']:.2f})"
        )


if __name__ == "__main__":
    main()
