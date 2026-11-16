# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Sævik (2011) FBG armour stresses against CableXSec (gap 2.5, rows SV11-*).

The 8-inch riser of Sævik (2011, Table 1) is packaged as ``saevik2011.yaml``.
Its measured inner-armour stress ranges at 20 Bragg stations near a bellmouth
(Figs 17-23, normalised) and the absolute LC 14 loops of Gjøsteen et al.
(2027, Figs 27-35) are digitised in ``validation/paper_data``. Targets were
registered before any model output (doc/benchmarks.md, Sævik section).

Prediction chain, per load case (tension T, internal pressure p, rocker angle
range A):

1. Section state: bounded contact chain at the wall tension
   ``T + p pi r_b^2`` (closed ends, bore of the seal layer) with ``p`` on the
   seal-layer bore; friction coefficient 0.20 declared (Sævik 1992 via Dai et
   al. 2017). Static stress = inner-armour wire stress of this state.
2. Curvature along the pipe: a tensioned beam (effective tension ``T``) with
   the CableXSec first-loading M-kappa backbone of that state, clamped at the
   end-fitting face 2.305 m before the curved bellmouth section, inside a
   straight bellmouth tube (radial gap 5 mm) followed by the curved profile
   ``kappa_b = 0.067 + 0.11 s`` to ``s = 1.72 m`` (Sævik Eq. 88), with a far
   pin 12.18 m + 3.31 m away (Gjøsteen et al. 2027, Sec. 4.6.1). The rocker
   rotates the clamp and bellmouth by ``A / 2`` about the start of the curved
   section (assumed). Small-slope theory, no gravity. The symmetric cycle
   ``+-A/2`` gives the station curvature range ``2 kappa_peak``.
3. Station stress: fatigue-path wire stress field of the inner armour under
   ``0 -> +k -> -k -> +k -> -k``; range over the last cycle at the station
   azimuth (interpolated between the bracketing wires). (a) sectional,
   (b) with :class:`cablexsec.TerminationSpec` (anchor at the end-fitting
   face, the first-loading peak profile as curvature shape). Axial
   (centroid) range; single-side ranges add the loxodromic lateral bending at
   the 6.25 mm tendon edge (round-wire tangential component x 12.5 / 8.921).

Usage::

    python validation/scripts/check_saevik2011.py OUTPUT_JSON [--mu 0.2] [--identify]
        [--tendons round|rectangular]

``--tendons rectangular`` runs the same chain on ``saevik2011_rect.yaml``
(5 x 12.5 mm tendons, gap G1); the default is the round area-equivalent
definition of the rows on record.

Thread counts are set to one.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
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
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "validation" / "paper_data"

MU_DECLARED = 0.20  # [Saevik1992] via [DaiSaevikYe2017] Sec. 4.3
R_BORE_M = 0.1076  # seal-layer bore, [Saevik2011] Table 1 (215.2 mm)
INNER = "inner_tensile_armour"
TENDON_WIDTH_M = 0.0125  # [Saevik2011] Table 1
# Rig (Gjosteen et al. 2027 Sec. 4.6.1; Saevik 2011 Sec. 7.1, Eq. 88)
X_ANCHOR_M = -(3.366 - 1.061)  # end-fitting face; rotation point at x = 0 (assumed)
FREE_LENGTH_M = 12.18  # flexible pipe between the end fittings
RIGID_RIGHT_M = 3.31  # rigid arrangement to the right-hand pin
GAP_M = 0.005  # minimum radial gap pipe-bellmouth [Saevik2011] Sec. 7.1
BELL_END_M = 1.72  # end of the curved profile [Saevik2011] Sec. 7.1
KAPPA_B0, KAPPA_B1 = 0.067, 0.11  # [Saevik2011] Eq. 88
GJ_HALF_ANGLE_DEG = 8.0  # [Gjosteen2027] LC 14, +-8 deg
TENDONS = "round"  # "round" (rows on record) or "rectangular" (gap G1)
K_ANCHOR_N_M: float | None = None  # end-fitting anchor stiffness per wire (rigid)
_POLYMER = {"round": "polymer_saevik2011", "rectangular": "polymer_saevik2011_rect"}


def use_tendons(tendons: str) -> None:
    """Select the tendon definition and clear the cached section results."""
    global TENDONS
    if tendons not in _POLYMER:
        raise ValueError("tendons must be 'round' or 'rectangular'")
    TENDONS = tendons
    cable.cache_clear()
    gate_values.cache_clear()


def use_anchor(k_anchor_N_m: float | None) -> None:
    """Set the anchor stiffness of the termination correction (None: rigid).

    Sensitivity studies only (``check_saevik2011_anchor.py``); the registered
    rows use the rigid anchor.
    """
    global K_ANCHOR_N_M
    K_ANCHOR_N_M = k_anchor_N_m
    gate_values.cache_clear()


@dataclass(frozen=True)
class LoadCase:
    lc: int
    angle_range_deg: float
    tension_N: float  # noqa: N815
    pressure_Pa: float  # noqa: N815


def _rows(path: Path) -> list[dict[str, str]]:
    lines = [
        ln
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln and not ln.startswith("#")
    ]
    return list(csv.DictReader(lines))


def measured_saevik() -> list[dict[str, str]]:
    """Digitised Figs 17-23 (one row per load case and station)."""
    return _rows(DATA / "saevik2011_figs17_23.csv")


def measured_gjosteen() -> dict[str, float]:
    """Measured LC 14 ranges [MPa] by station (Gjosteen et al. Figs 27-35)."""
    return {
        r["station"]: float(r["range_MPa"])
        for r in _rows(DATA / "gjosteen2027_lc14_ranges.csv")
        if r["series"] == "measured"
    }


def load_cases() -> list[LoadCase]:
    seen: dict[int, LoadCase] = {}
    for r in measured_saevik():
        lc = int(r["load_case"])
        seen.setdefault(
            lc,
            LoadCase(
                lc,
                float(r["angle_range_deg"]),
                float(r["tension_kN"]) * 1e3,
                float(r["pressure_MPa"]) * 1e6,
            ),
        )
    return sorted(seen.values(), key=lambda c: (c.angle_range_deg, c.lc))


def stations() -> list[tuple[str, float, float]]:
    """(station, x [m] from the start of the curved section, azimuth [deg])."""
    out = []
    for r in measured_saevik():
        if r["load_case"] == "14":
            out.append((r["station"], float(r["x_mm"]) / 1e3, float(r["azimuth_deg"])))
    return out


# --------------------------------------------------------------------------- #
# Section                                                                      #
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def cable():  # type: ignore[no-untyped-def]
    import cablexsec as cx  # noqa: PLC0415

    return cx.load_cable(cx.saevik2011_yaml_path(TENDONS), geometry_override=None)


def section(case: LoadCase, mu: float):  # type: ignore[no-untyped-def]
    """Section state, inner-armour fatigue layer and static stress."""
    import cablexsec as cx  # noqa: PLC0415
    from cablexsec.cable_adapters import cable_to_fatigue_layers  # noqa: PLC0415

    cab = cable()
    wall = case.tension_N + case.pressure_Pa * math.pi * R_BORE_M**2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cx.MissingClampingStateWarning)
        state = cx.solve_section_state(
            cab, tension_N=wall, p_int_Pa=case.pressure_Pa, contact_model="bounded"
        )
        layers = cable_to_fatigue_layers(
            cab, mu=mu, contact_model="bounded", section_state=state
        )
    layer = next(x for x in layers if x.name == INNER)
    static = state.layers[INNER].longitudinal_stress_Pa
    return state, layer, static


def backbone(case: LoadCase, mu: float):  # type: ignore[no-untyped-def]
    """First-loading M-kappa of the section state (PCHIP), and its EI bounds."""
    import cablexsec as cx  # noqa: PLC0415

    state, _layer, _static = section(case, mu)
    kappa = np.concatenate([[0.0], np.geomspace(1e-6, 0.6, 360)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cx.MissingClampingStateWarning)
        res = cx.compute_mkappa(cable(), kappa, mu=mu, section_state=state)
    m = np.maximum.accumulate(np.asarray(res.moment_Nm))
    return (
        PchipInterpolator(kappa, m, extrapolate=False),
        float(res.EI_max_Nm2),
        float(res.EI_min_Nm2),
    )


# --------------------------------------------------------------------------- #
# Global tensioned beam in the bellmouth                                       #
# --------------------------------------------------------------------------- #
class Beam:
    """Finite-difference tensioned beam with a hysteresis backbone and bounds."""

    def __init__(
        self,
        moment: PchipInterpolator,
        ei_min: float,
        tension_N: float,
        h: float = 0.02,
    ) -> None:
        self.h = h
        x_pin = X_ANCHOR_M + FREE_LENGTH_M + RIGID_RIGHT_M
        self.x = np.arange(X_ANCHOR_M, x_pin + 0.5 * h, h)
        self.n = self.x.size
        self.T = tension_N
        self.kmax = float(moment.x[-1])
        self.m_end = float(moment(self.kmax))
        self.ei_end = ei_min
        k = np.asarray(moment.x)
        self.moment = moment
        self.dm = moment.derivative()
        self.energy = moment.antiderivative()
        self.rigid = self.x > X_ANCHOR_M + FREE_LENGTH_M
        self.lower = np.full(self.n, -np.inf)
        self.upper = np.full(self.n, np.inf)
        tube = self.x <= 0.0
        self.lower[tube], self.upper[tube] = -GAP_M, GAP_M
        bell = (self.x > 0.0) & (self.x <= BELL_END_M)
        s = self.x[bell]
        wb = GAP_M + KAPPA_B0 * s**2 / 2 + KAPPA_B1 * s**3 / 6
        self.lower[bell], self.upper[bell] = -wb, wb
        self.fixed = np.zeros(self.n, dtype=bool)
        self.fixed[[0, 1, self.n - 1]] = True
        self.residual = math.nan
        del k

    def _law(self, kappa: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.abs(kappa)
        inside = a <= self.kmax
        w = np.where(
            inside,
            self.energy(np.minimum(a, self.kmax)),
            self.energy(self.kmax)
            + self.m_end * (a - self.kmax)
            + 0.5 * self.ei_end * (a - self.kmax) ** 2,
        )
        m = np.where(
            inside,
            self.moment(np.minimum(a, self.kmax)),
            self.m_end + self.ei_end * (a - self.kmax),
        ) * np.sign(kappa)
        t = np.where(inside, self.dm(np.minimum(a, self.kmax)), self.ei_end)
        t = np.maximum(t, self.ei_end)
        return w, m, t

    def _parts(self, w: np.ndarray):  # type: ignore[no-untyped-def]
        h = self.h
        kappa = (w[:-2] - 2 * w[1:-1] + w[2:]) / h**2
        we, m, t = self._law(kappa)
        rigid = self.rigid[1:-1]
        ei_r = 1.0e9
        we = np.where(rigid, 0.5 * ei_r * kappa**2, we)
        m = np.where(rigid, ei_r * kappa, m)
        t = np.where(rigid, ei_r, t)
        slope = np.diff(w) / h
        energy = float(np.sum(we) * h + 0.5 * self.T * np.sum(slope**2) * h)
        grad = np.zeros(self.n)
        c = m * h / h**2
        grad[:-2] += c
        grad[1:-1] -= 2 * c
        grad[2:] += c
        g_t = self.T * slope
        grad[:-1] -= g_t
        grad[1:] += g_t
        d = t * h / h**4
        rows, cols, vals = [], [], []
        idx = np.arange(self.n - 2)
        stencil = ((0, 1.0), (1, -2.0), (2, 1.0))
        for a, ca in stencil:
            for b, cb in stencil:
                rows.append(idx + a)
                cols.append(idx + b)
                vals.append(d * ca * cb)
        e = np.arange(self.n - 1)
        kt = self.T / h
        for a, b, s in ((0, 0, 1), (1, 1, 1), (0, 1, -1), (1, 0, -1)):
            rows.append(e + a)
            cols.append(e + b)
            vals.append(np.full(e.size, kt * s))
        hess = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n, self.n),
        )
        return energy, grad, hess, kappa

    def solve(
        self,
        half_angle_rad: float,
        w0: np.ndarray | None = None,
        tol: float = 1e-9,
        max_iter: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Projected-Newton minimisation of the beam energy at one rocker angle."""
        w = np.zeros(self.n) if w0 is None else w0.copy()
        x_pin = self.x[-1]
        w[0] = w[1] = 0.0
        w[-1] = -x_pin * math.sin(half_angle_rad)
        lo, hi = self.lower.copy(), self.upper.copy()
        lo[self.fixed], hi[self.fixed] = w[self.fixed], w[self.fixed]
        w = np.clip(w, lo, hi)
        energy, grad, hess, kappa = self._parts(w)
        scale = max(1.0, float(np.max(np.abs(grad))))
        for _ in range(max_iter):
            at_lo = (w <= lo + 1e-12) & (grad > 0)
            at_hi = (w >= hi - 1e-12) & (grad < 0)
            active = at_lo | at_hi | self.fixed
            free = ~active
            pg = np.where(free, grad, 0.0)
            if np.max(np.abs(pg)) < tol * scale:
                break
            step = np.zeros(self.n)
            hff = hess[free][:, free].tocsc()
            step[free] = spla.spsolve(hff, -grad[free])
            alpha = 1.0
            for _ls in range(40):
                trial = np.clip(w + alpha * step, lo, hi)
                e_t, g_t, h_t, k_t = self._parts(trial)
                if e_t <= energy + 1e-4 * float(grad @ (trial - w)) or alpha < 1e-8:
                    break
                alpha *= 0.5
            w, energy, grad, hess, kappa = trial, e_t, g_t, h_t, k_t
        at_lo = (w <= lo + 1e-12) & (grad > 0)
        at_hi = (w >= hi - 1e-12) & (grad < 0)
        free = ~(at_lo | at_hi | self.fixed)
        self.residual = float(np.max(np.abs(np.where(free, grad, 0.0))) / scale)
        k_nodes = np.zeros(self.n)
        k_nodes[1:-1] = kappa
        k_nodes[0], k_nodes[-1] = k_nodes[1], k_nodes[-2]
        return w, k_nodes

    def peak_profile(
        self, half_angle_deg: float, steps: int = 12
    ) -> tuple[np.ndarray, np.ndarray]:
        w = None
        k = np.zeros(self.n)
        for j in range(1, steps + 1):
            w, k = self.solve(math.radians(half_angle_deg) * j / steps, w)
        assert w is not None
        return w, k


# --------------------------------------------------------------------------- #
# Station stresses                                                             #
# --------------------------------------------------------------------------- #
def _history(peak: float, per_leg: int = 60) -> np.ndarray:
    legs = [np.linspace(0, peak, per_leg + 1)]
    for a, b in ((peak, -peak), (-peak, peak), (peak, -peak)):
        legs.append(np.linspace(a, b, 2 * per_leg + 1)[1:])
    return np.concatenate(legs)


def _at_azimuth(values: np.ndarray, azimuth_deg: float) -> np.ndarray:
    """Linear interpolation of a (wire, time) field to an azimuth."""
    n = values.shape[0]
    pos = (azimuth_deg % 360.0) / 360.0 * n
    i0 = int(math.floor(pos)) % n
    i1 = (i0 + 1) % n
    f = pos - math.floor(pos)
    return np.asarray((1 - f) * values[i0] + f * values[i1])


def station_ranges(
    layer,
    peak: float,
    x_s: float,
    azimuth_deg: float,  # type: ignore[no-untyped-def]
    profile: tuple[np.ndarray, np.ndarray] | None,
) -> dict[str, float]:
    """Axial and single-side ranges [Pa] over the last cycle at one station."""
    from cablexsec.analysis.fatigue import wire_stress_field  # noqa: PLC0415
    from cablexsec.analysis.termination import TerminationSpec  # noqa: PLC0415

    kx = _history(peak)
    ky = np.zeros_like(kx)
    term = None
    if profile is not None:
        term = TerminationSpec(
            distance_m=x_s - X_ANCHOR_M,
            curvature_profile=profile,
            k_anchor_N_m=K_ANCHOR_N_M,
        )
    field = wire_stress_field(layer, kx, ky, termination=term)
    last = slice(kx.size - 4 * 60 - 1, kx.size)
    ratio = (
        TENDON_WIDTH_M / (2 * math.sqrt(layer.spec.A_wire_m2 / math.pi))
        if layer.spec.section is None
        else 1.0  # rectangular section: the edge component is at b/2 already
    )
    out: dict[str, float] = {}
    for sign_label, az in (("p", azimuth_deg), ("m", -azimuth_deg)):
        mean = _at_azimuth(field.mean_Pa, az)[last]
        lat = _at_azimuth(field.tangential_Pa, az)[last] * ratio
        out[f"axial_{sign_label}"] = float(np.ptp(mean))
        out[f"lateral_{sign_label}"] = float(np.ptp(lat))
        out[f"side_lo_{sign_label}"] = float(
            min(np.ptp(mean + lat), np.ptp(mean - lat))
        )
        out[f"side_hi_{sign_label}"] = float(
            max(np.ptp(mean + lat), np.ptp(mean - lat))
        )
    inc = field.termination
    out["termination_applied"] = float(bool(inc is not None and inc.applied))
    out["boundary_layer_m"] = float(inc.boundary_layer_m) if inc is not None else 0.0
    out["reach_m"] = float(inc.reach_m) if inc is not None else 0.0
    return out


def predict_case(
    case: LoadCase,
    mu: float,
    half_angle_deg: float | None = None,
    only: tuple[str, ...] | None = None,
) -> dict:
    """Station predictions of one load case (all stations, or ``only``)."""
    _state, layer, static = section(case, mu)
    moment, ei_max, ei_min = backbone(case, mu)
    beam = Beam(moment, ei_min, case.tension_N)
    half = case.angle_range_deg / 2 if half_angle_deg is None else half_angle_deg
    w, k = beam.peak_profile(half)
    xp = beam.x - X_ANCHOR_M
    profile = (xp, k)
    out = {
        "lc": case.lc,
        "angle_range_deg": 2 * half,
        "tension_N": case.tension_N,
        "pressure_Pa": case.pressure_Pa,
        "mu": mu,
        "static_Pa": static,
        "EI_max_Nm2": ei_max,
        "EI_min_Nm2": ei_min,
        "contact_pressure_Pa": float(layer.spec.contact_pressure_Pa),
        "beam_residual": beam.residual,
        "kappa_profile": {"x_m": beam.x[::5].tolist(), "kappa_1_m": k[::5].tolist()},
        "stations": {},
    }
    for name, x_s, az in stations():
        if only is not None and name not in only:
            continue
        peak = float(np.interp(x_s, beam.x, k))
        sec = station_ranges(layer, peak, x_s, az, None)
        term = station_ranges(layer, peak, x_s, az, profile)
        out["stations"][name] = {
            "x_m": x_s,
            "azimuth_deg": az,
            "kappa_peak": peak,
            "sectional": sec,
            "termination": term,
        }
    return out


def metrics(results: list[dict]) -> dict:
    """Registered metrics (SV11-PLANE/SEC/MAX/STATION) from the predictions."""
    meas = measured_saevik()
    by = {(int(r["load_case"]), r["station"]): float(r["measured"]) for r in meas}
    out: dict = {"cases": {}}
    logs_t, logs_s = [], []
    for res in results:
        lc = res["lc"]
        plane = [
            s for s, v in res["stations"].items() if v["azimuth_deg"] in (0.0, 180.0)
        ]
        m = np.array([by[(lc, s)] for s in plane])
        pt = (
            np.array([res["stations"][s]["termination"]["axial_p"] for s in plane])
            / res["static_Pa"]
        )
        ps = (
            np.array([res["stations"][s]["sectional"]["axial_p"] for s in plane])
            / res["static_Pa"]
        )
        out["cases"][lc] = {
            "R_termination": float(pt.mean() / m.mean()),
            "R_sectional": float(ps.mean() / m.mean()),
            "max_ratio_termination": float(pt.max() / m.max()),
            "max_ratio_sectional": float(ps.max() / m.max()),
            "stations": {
                s: {
                    "measured": float(a),
                    "termination": float(b),
                    "sectional": float(c),
                }
                for s, a, b, c in zip(plane, m, pt, ps, strict=True)
            },
        }
        logs_t.extend(np.log(pt / m))
        logs_s.extend(np.log(ps / m))
    rt = np.array([v["R_termination"] for v in out["cases"].values()])
    rs = np.array([v["R_sectional"] for v in out["cases"].values()])
    out["median_R_termination"] = float(np.median(rt))
    out["median_R_sectional"] = float(np.median(rs))
    out["count_in_band_termination"] = int(np.sum((rt >= 0.70) & (rt <= 1.43)))
    out["count_in_band_sectional"] = int(np.sum((rs >= 0.70) & (rs <= 1.43)))
    out["rms_log_station_termination"] = float(np.sqrt(np.mean(np.square(logs_t))))
    out["rms_log_station_sectional"] = float(np.sqrt(np.mean(np.square(logs_s))))
    out["gate_plane"] = bool(
        0.85 <= out["median_R_termination"] <= 1.18
        and out["count_in_band_termination"] >= 12
    )
    return out


def plane_stations() -> tuple[str, ...]:
    """The bending-plane stations (*.1 and *.5) used by SV11-PLANE."""
    return tuple(s for s, _x, az in stations() if az in (0.0, 180.0))


@lru_cache(maxsize=2)
def gate_values(mu: float = MU_DECLARED) -> dict:
    """SV11-PLANE and SV11-GJ metrics, bending-plane stations only (cached)."""
    plane = plane_stations()
    results = [predict_case(c, mu, only=plane) for c in load_cases()]
    return {
        "plane": metrics(results),
        "gjosteen": gjosteen(mu, plane_only=True),
        "static_MPa": {r["lc"]: r["static_Pa"] / 1e6 for r in results},
        "beam_residual": max(r["beam_residual"] for r in results),
    }


def gjosteen(mu: float, plane_only: bool = False) -> dict:
    """SV11-GJ: LC 14 at +-8 deg, absolute ranges [MPa]."""
    case = next(c for c in load_cases() if c.lc == 14)
    names = ("2.1", "2.5", "3.1", "3.5") + (
        () if plane_only else ("2.2", "2.3", "3.2", "3.3", "3.4")
    )
    res = predict_case(case, mu, half_angle_deg=GJ_HALF_ANGLE_DEG, only=names)
    meas = measured_gjosteen()
    plane = ["2.1", "2.5", "3.1", "3.5"]
    pred = {s: res["stations"][s]["termination"]["axial_p"] / 1e6 for s in plane}
    sec = {s: res["stations"][s]["sectional"]["axial_p"] / 1e6 for s in plane}
    ratio = float(np.mean([pred[s] for s in plane]) / np.mean([meas[s] for s in plane]))
    ratio_s = float(
        np.mean([sec[s] for s in plane]) / np.mean([meas[s] for s in plane])
    )
    others = {}
    for s in names[4:]:
        st = res["stations"][s]["termination"]
        others[s] = {
            "measured_MPa": meas[s],
            "axial_MPa": [st["axial_p"] / 1e6, st["axial_m"] / 1e6],
            "side_MPa": [
                min(st["side_lo_p"], st["side_lo_m"]) / 1e6,
                max(st["side_hi_p"], st["side_hi_m"]) / 1e6,
            ],
            "lateral_MPa": [st["lateral_p"] / 1e6, st["lateral_m"] / 1e6],
        }
    return {
        "plane_pred_MPa": pred,
        "plane_sectional_MPa": sec,
        "plane_measured_MPa": {s: meas[s] for s in plane},
        "ratio_termination": ratio,
        "ratio_sectional": ratio_s,
        "gate": bool(0.72 <= ratio <= 1.38),
        "others": others,
        "static_Pa": res["static_Pa"],
        "kappa_peak": {s: res["stations"][s]["kappa_peak"] for s in res["stations"]},
    }


def identify_polymer_modulus() -> float:
    """Polymer modulus that makes EI_min equal 72.8 kN m2 ([Saevik2011] Table 1)."""
    from scipy.optimize import brentq  # noqa: PLC0415

    import cablexsec as cx  # noqa: PLC0415
    from cablexsec.materials import library  # noqa: PLC0415

    name = _POLYMER[TENDONS]
    original = library.MATERIALS[name]["E_Pa"]

    def ei(e: float) -> float:
        library.MATERIALS[name]["E_Pa"] = (e, "identification probe")
        return float(
            cx.compute_stiffness(
                cx.load_cable(cx.saevik2011_yaml_path(TENDONS), geometry_override=None)
            ).ei_min_Nm2
        )

    try:
        return float(brentq(lambda e: ei(e) - 72.8e3, 1e7, 1e10, xtol=1e3))
    finally:
        library.MATERIALS[name]["E_Pa"] = original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("--mu", type=float, default=MU_DECLARED)
    parser.add_argument("--cases", type=int, nargs="*", default=None)
    parser.add_argument("--tendons", choices=("round", "rectangular"), default="round")
    parser.add_argument(
        "--identify",
        action="store_true",
        help="also identify mu on LC 14 (SV11-ID) and predict the others",
    )
    args = parser.parse_args()
    use_tendons(args.tendons)
    t0 = time.time()
    cases = [c for c in load_cases() if args.cases is None or c.lc in args.cases]
    results = []
    for case in cases:
        t1 = time.time()
        res = predict_case(case, args.mu)
        results.append(res)
        print(
            f"LC{case.lc} A={case.angle_range_deg} done {time.time() - t1:.1f}s",
            flush=True,
        )
    out = {
        "mu": args.mu,
        "results": results,
        "metrics": metrics(results),
        "gjosteen": gjosteen(args.mu)
        if args.cases is None or 14 in args.cases
        else None,
    }
    if args.identify:
        from scipy.optimize import brentq  # noqa: PLC0415

        lc14 = next(c for c in load_cases() if c.lc == 14)

        def r14(mu: float) -> float:
            return metrics([predict_case(lc14, mu)])["cases"][14]["R_termination"] - 1.0

        mu_id = float(brentq(r14, 0.03, 0.6, xtol=1e-3))
        res_id = [predict_case(c, mu_id) for c in cases]
        out["identified"] = {"mu": mu_id, "metrics": metrics(res_id)}
    out["seconds"] = time.time() - t0
    args.output.write_text(json.dumps(out, indent=1), encoding="utf-8")
    m = out["metrics"]
    print(json.dumps({k: v for k, v in m.items() if k != "cases"}, indent=1))
    for lc, v in m["cases"].items():
        print(
            lc,
            f"R_t {v['R_termination']:.2f} R_s {v['R_sectional']:.2f} "
            f"max_t {v['max_ratio_termination']:.2f}",
        )
    if out["gjosteen"]:
        g = out["gjosteen"]
        print(
            "GJ",
            f"{g['ratio_termination']:.2f} {g['ratio_sectional']:.2f}",
            g["plane_pred_MPa"],
            g["plane_measured_MPa"],
        )


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
