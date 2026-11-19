# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Z-EI and Z-ID: the Zeidan (2025) four-point loops of a 112 mm cable.

Targets were registered (doc/benchmarks.md, "Zeidan (2025) four-point loops")
in the commit that added the cable and the data, before any analysis of the
cable was run; this script reports, it does not fit beyond the registered
identification.

The test is simulated in its arrangement (supports 2.0 m, loads 1.05 m
apart, shear span 0.475 m, zero tension) and reduced as the test reduces it:
``M = F a`` per loading point and ``kappa = 6 y / (a (3 L - 4 a))`` (thesis
Eq. 5.1). Reading R2 (primary): ``y`` is the mid-span deflection (paper
Fig. 18); reading R1: the loading-point deflection.

Run: ``PYTHONPATH=src python validation/scripts/check_zeidan2025.py`` (one
thread; a few minutes).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import cablexsec as cx
from cablexsec.analysis.beam_test import (
    BendingProtocol,
    FourPoint,
    SectionLaw,
    SectionResponse,
)
from cablexsec.api import CoreFrictionSpec
from cablexsec.cables.schema import (
    CableData,
    CylindricalLayer,
    InitialStateProperties,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "validation" / "paper_data"
CABLE = ROOT / "src" / "cablexsec" / "cables" / "zeidan2025.yaml"
FloatArray = NDArray[np.float64]

SPAN_M = 2.0  # thesis Sec. 5.3, supports
LOADING_SPAN_M = 1.05  # loading points
SHEAR_SPAN_M = 0.5 * (SPAN_M - LOADING_SPAN_M)  # a = 0.475 m
#: Thesis Eq. 5.1: kappa = EQ51 * y.
EQ51 = 6.0 / (SHEAR_SPAN_M * (3.0 * SPAN_M - 4.0 * SHEAR_SPAN_M))
#: Elastic mid-span formula: kappa = ELASTIC_MID * w_mid.
ELASTIC_MID = 24.0 / (3.0 * SPAN_M**2 - 4.0 * SHEAR_SPAN_M**2)
#: Plotted curvature per true elastic curvature under reading R2 (1.425).
R2_FACTOR = EQ51 / ELASTIC_MID
AMPLITUDES_MM = (30, 40, 50, 60)
IDENTIFY_MM = 40
MU = 0.2
FULL = CoreFrictionSpec(filler_profiles=True)
ARRANGEMENT = FourPoint(SPAN_M, LOADING_SPAN_M)
#: Initial pressures scanned for the bracket of the identification [Pa].
P0_GRID = (1e4, 2e4, 5e4, 1e5, 1.5e5, 2e5, 3e5, 5e5, 1e6, 2e6, 3.5e6, 5e6, 1e7)
#: Source FE calibration of the outer-sheath pressure (paper Sec. 5.3) [Pa].
SOURCE_PRESSURE_PA = 3.5e6
#: Published bilinear fit of the stabilised loops (thesis Table 5.4).
BILINEAR = {"K1_Nm2": 106e3, "K2_Nm2": 9.0e3, "My_Nm": 1.83e3}
#: Fig. 18 central curvature at 60 mm (parabola, x 0.5-1.0 m) [1/m] and the
#: loading-point to mid-span deflection ratio (40.6 / 60.0 mm).
FIG18 = {"kappa_60": 0.143, "ratio_60": 0.677}


# --------------------------------------------------------------------------- #
# Measured records                                                             #
# --------------------------------------------------------------------------- #
def load_record(figure: str, amplitude_mm: int) -> tuple[FloatArray, FloatArray]:
    """Plotted curvature [1/m] and moment [N m] of thesis Fig. 5.5 or 5.6."""
    path = DATA / f"zeidan2025_fig{figure}_{amplitude_mm}mm.csv"
    data = np.loadtxt(path, delimiter=",", comments="#", skiprows=7)
    return data[:, 1], data[:, 2]


def load_deflection(amplitude_mm: int) -> tuple[FloatArray, FloatArray]:
    """Paper Fig. 18 deflection along the half span (x [m], w [m])."""
    data = np.loadtxt(
        DATA / "zeidan2026_fig18_deflection.csv",
        delimiter=",",
        comments="#",
        skiprows=5,
    )
    sel = data[:, 0] == amplitude_mm
    return data[sel, 1], data[sel, 2]


def central_curvature(x: FloatArray, w: FloatArray, start: float = 0.5) -> float:
    """Curvature of the parabola ``w_m + kappa/2 (x - 1)^2`` fitted to the
    central zone (x from ``start`` to mid-span, 1.0 m)."""
    sel = (x >= start) & (x <= 1.0)
    design = np.column_stack([np.ones(int(sel.sum())), 0.5 * (x[sel] - 1.0) ** 2])
    coef, *_ = np.linalg.lstsq(design, w[sel], rcond=None)
    return float(coef[1])  # deflection negative downwards: sagging > 0


# --------------------------------------------------------------------------- #
# Loop metrics (measured and model alike)                                      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Loop:
    """One closed cycle from the curvature minimum, with its metrics."""

    kappa: FloatArray
    moment: FloatArray
    work: float  # J/m (mean over complete cycles for a record)
    works: tuple[float, ...]
    moment_range: float
    centre: float
    half_height: float
    stick_secant: tuple[float, float]  # after the minimum, after the maximum
    slip_slope: float

    @property
    def peak(self) -> int:
        return int(np.argmax(self.kappa))


def _cycle_work(k: FloatArray, m: FloatArray) -> float:
    return float(abs(np.sum(0.5 * (m[1:] + m[:-1]) * np.diff(k))))


def _minima(k: FloatArray) -> list[int]:
    """Index of the curvature minimum of every visit below 0.9 of the minimum."""
    low = k < 0.9 * k.min()
    idx, i = [], 0
    while i < k.size:
        if low[i]:
            j = i
            while j < k.size and low[j]:
                j += 1
            idx.append(i + int(np.argmin(k[i:j])))
            i = j
        else:
            i += 1
    return idx


def loop_metrics(k: FloatArray, m: FloatArray, *, record: bool) -> Loop:
    """Metrics of the last complete cycle (``record``) or of one closed cycle.

    Work: mean over the complete cycles between curvature minima (record) or
    the loop area. Centre: mean of the moments at the two reversals.
    Half-height: half the branch difference at kappa = 0. Stick secant: from
    each reversal to the first point where the moment has moved by 30 % of
    the moment range. Slip slope: least squares on the ascending branch for
    kappa from 0 to 0.8 of the amplitude.
    """
    if record:
        lows = _minima(k)
        works = tuple(
            _cycle_work(k[a : b + 1], m[a : b + 1])
            for a, b in zip(lows[:-1], lows[1:], strict=True)
        )
        a, b = lows[-2], lows[-1]
        kc, mc = k[a : b + 1], m[a : b + 1]
        work = float(np.mean(works))
    else:
        i = int(np.argmin(k))
        kc = np.concatenate([k[i:], k[1 : i + 1]])
        mc = np.concatenate([m[i:], m[1 : i + 1]])
        work = _cycle_work(kc, mc)
        works = (work,)
    top = int(np.argmax(kc))
    rng = float(mc[top] - mc[0])
    centre = 0.5 * float(mc[top] + mc[0])
    ka, ma = kc[: top + 1], mc[: top + 1]
    kd, md = kc[top:], mc[top:]

    def at(kb: FloatArray, mb: FloatArray, x: float) -> float:
        order = np.argsort(kb)
        return float(np.interp(x, kb[order], mb[order]))

    half = 0.5 * (at(ka, ma, 0.0) - at(kd, md, 0.0))

    def secant(kb: FloatArray, mb: FloatArray, sign: float) -> float:
        moved = sign * (mb - mb[0])
        j = int(np.flatnonzero(moved >= 0.3 * rng)[0])
        return float((mb[j] - mb[0]) / (kb[j] - kb[0]))

    amp = float(kc.max())
    sel = (ka >= 0.0) & (ka <= 0.8 * amp)
    slope = float(np.polyfit(ka[sel], ma[sel], 1)[0])
    return Loop(
        kc,
        mc,
        work,
        works,
        rng,
        centre,
        half,
        (secant(ka, ma, 1.0), secant(kd, md, -1.0)),
        slope,
    )


def branch_rmse(measured: Loop, model: Loop) -> float:
    """RMSE of the model branches at the measured curvature samples, measured
    centre removed, as a fraction of the measured moment range."""
    errors = []
    for meas_k, meas_m, mod_k, mod_m in (
        (
            measured.kappa[: measured.peak + 1],
            measured.moment[: measured.peak + 1],
            model.kappa[: model.peak + 1],
            model.moment[: model.peak + 1],
        ),
        (
            measured.kappa[measured.peak :],
            measured.moment[measured.peak :],
            model.kappa[model.peak :],
            model.moment[model.peak :],
        ),
    ):
        order = np.argsort(mod_k)
        pred = np.interp(meas_k, mod_k[order], mod_m[order])
        errors.append(pred - (meas_m - measured.centre))
    err = np.concatenate(errors)
    return float(np.sqrt(np.mean(err**2)) / measured.moment_range)


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #
def load_base() -> CableData:
    return cx.load_cable(CABLE, geometry_override=None)


def with_pressure(
    cable: CableData,
    pressure_Pa: float,
    *,
    scope: str = "all_contacts",
    identification: str = "identified",
    source: str = "identified on Zeidan (2025) Fig. 5.6, 40 mm cycle work",
) -> CableData:
    """Copy of the cable with the initial external pressure replaced."""
    state = (
        InitialStateProperties(
            external_pressure_Pa=pressure_Pa,
            identification=identification,  # type: ignore[arg-type]
            source=source,
            calibration_scope=scope,  # type: ignore[arg-type]
        )
        if pressure_Pa > 0.0
        else InitialStateProperties()
    )
    return cable.model_copy(update={"initial_state": state})


Section = Callable[[CableData, CoreFrictionSpec | None, float], SectionLaw]


def packaged_section(
    cable: CableData, spec: CoreFrictionSpec | None, mu: float
) -> SectionLaw:
    """The packaged section law (``SectionLaw.from_cable``), bounded chain."""
    return SectionLaw.from_cable(
        cable, tension_N=0.0, mu=mu, contact_model="bounded", core_friction=spec
    )


def concentric_xi(cable: CableData) -> list[float]:
    """Armour stick coefficients from the concentric support (the bedding)
    alone, in the order of the helical layers ([Yun2020] with one shear
    layer). The diagnostic predates the support-cut stack (2026-11-19): the
    packaged stack, which then counted the layers of the offset cores, now
    gives the same inner coefficient and 1 for the outer armour, which bears
    on the inner one without a layer between them."""
    from cablexsec.materials import library  # noqa: PLC0415
    from cablexsec.mechanics.yun_coefficient import (  # noqa: PLC0415
        series_shear_stiffness,
        shear_modulus,
        yun_xi,
    )

    out = []
    with cable.material_scope():
        bed = cable.get_layer("bedding")
        assert isinstance(bed, CylindricalLayer)
        assert bed.r_inner_m is not None and bed.r_outer_m is not None
        g = shear_modulus(
            library.get_property(bed.material, "E_Pa"),
            library.get_property(bed.material, "nu"),
        )
        for layer in cable.helical_layers:
            assert layer.d_wire_m is not None and layer.r_helix_m is not None
            assert layer.lay_angle_rad is not None
            ea = library.get_property(layer.material, "E_Pa")
            ea *= math.pi * layer.d_wire_m**2 / 4.0
            k_g = series_shear_stiffness(
                [(bed.r_outer_m - bed.r_inner_m, g, layer.d_wire_m)]
            )
            out.append(yun_xi(k_g, ea, layer.lay_angle_rad, layer.r_helix_m))
    return out


def concentric_support_section(
    cable: CableData, spec: CoreFrictionSpec | None, mu: float
) -> SectionLaw:
    """Diagnostic law, not a registered model: the packaged law with the
    armour stick coefficients of :func:`concentric_xi`."""
    from cablexsec.api import _prepare_mkappa_law  # noqa: PLC0415
    from cablexsec.mechanics.core_friction import (  # noqa: PLC0415
        resolve_core_friction,
    )

    xis = concentric_xi(cable)
    with cable.material_scope():
        law = _prepare_mkappa_law(
            cable,
            0.0,
            mu=mu,
            phi_bend=0.0,
            contact_model="bounded",
            section_state=None,
            radial_state=None,
            core_spec=resolve_core_friction(spec),
            progressive=None,
        )
    if len(xis) != len(law.layers):
        raise RuntimeError("every helical layer must be a structural armour")
    layers = [replace(x, xi=xi) for x, xi in zip(law.layers, xis, strict=True)]
    law = replace(law, layers=layers)

    def run(kappa: FloatArray) -> SectionResponse:
        with cable.material_scope():
            out = law.run(kappa)
        return SectionResponse(
            out.moment_Nm, out.work.friction_J_per_m, out.work.stored_J_per_m
        )

    stick = sum(x.EI_stick for x in layers) + law.ei_min
    if law.core_model is not None:
        stick += law.core_model.stick_EI_Nm2
    return SectionLaw(
        run,
        ei_stick_Nm2=stick,
        ei_slip_Nm2=law.ei_min,
        description="diagnostic: armour xi from the bedding alone",
    )


@dataclass(frozen=True)
class ModelLoop:
    loop: Loop
    kappa_centre_peak: float  # true station curvature at mid-span, at the peak
    deflection_ratio: float  # loading point / mid-span at the peak


def model_loop(
    section: SectionLaw,
    amplitude_mm: float,
    *,
    reading: str = "R2",
    samples_per_cycle: int = 160,
) -> ModelLoop:
    """The test simulated in its arrangement and reduced by the test's formula."""
    y = amplitude_mm * 1e-3
    if reading == "R2":
        protocol = BendingProtocol.cyclic(
            "curvature",
            ELASTIC_MID * y,
            n_cycles=3,
            samples_per_cycle=samples_per_cycle,
        )
        res = cx.simulate_bending_test(
            section, ARRANGEMENT, protocol, curvature_definition="elastic"
        )
        mid = res.curvature_1_m / ELASTIC_MID
        kappa = EQ51 * mid
    else:
        protocol = BendingProtocol.cyclic(
            "displacement", y, n_cycles=3, samples_per_cycle=samples_per_cycle
        )
        res = cx.simulate_bending_test(
            section, ARRANGEMENT, protocol, curvature_definition="elastic"
        )
        mid = res.curvature_1_m / ELASTIC_MID
        kappa = EQ51 * res.displacement_m
    keep = res.retained
    loop = loop_metrics(kappa[keep], res.moment_Nm[keep], record=False)
    peak = int(np.argmax(np.where(keep, kappa, -np.inf)))
    return ModelLoop(
        loop,
        float(res.station_curvature_1_m[peak]),
        float(res.displacement_m[peak] / mid[peak]),
    )


@dataclass(frozen=True)
class Identification:
    pressure_pa: float | None  # None: the measured work is not reached
    largest_work: float
    largest_at_pa: float
    scan: tuple[tuple[float, float], ...]


def identify(
    make: Section,
    spec: CoreFrictionSpec | None,
    target_work: float,
    *,
    reading: str = "R2",
    mu: float = MU,
) -> Identification:
    """Smallest initial pressure whose 40 mm cycle work equals the target.

    The work is not monotone in the pressure (a high pressure keeps the
    armour stuck), so the grid ``P0_GRID`` is scanned for the first bracket
    and the bracket is bisected to 0.1 % of the work.
    """
    base = load_base()
    scope = "all_contacts" if spec is not None else "armour_only"

    def work(p: float) -> float:
        section = make(with_pressure(base, p, scope=scope), spec, mu)
        return model_loop(section, IDENTIFY_MM, reading=reading).loop.work

    scan = []
    previous = (0.0, 0.0)
    bracket = None
    for p in P0_GRID:
        w = work(p)
        scan.append((p, w))
        if previous[1] < target_work <= w:
            bracket = (previous[0], p)
            break
        previous = (p, w)
    largest = max(scan, key=lambda s: s[1])
    if bracket is None:
        return Identification(None, largest[1], largest[0], tuple(scan))
    low, high = bracket
    for _ in range(40):
        mid = 0.5 * (low + high)
        w = work(mid)
        if abs(w / target_work - 1.0) < 1e-3:
            low = high = mid
            break
        if w < target_work:
            low = mid
        else:
            high = mid
    return Identification(0.5 * (low + high), largest[1], largest[0], tuple(scan))


# --------------------------------------------------------------------------- #
# Registered rows                                                              #
# --------------------------------------------------------------------------- #
def measured_loops(figure: str = "5_6") -> dict[int, Loop]:
    return {
        amp: loop_metrics(*load_record(figure, amp), record=True)
        for amp in AMPLITUDES_MM
    }


def z_ei() -> dict[str, object]:
    """Registered bound row Z-EI (parameter-free)."""
    stiff = cx.compute_stiffness(load_base())
    loops = measured_loops()
    stick = {a: tuple(R2_FACTOR * s for s in loops[a].stick_secant) for a in loops}
    slip = {a: loops[a].slip_slope for a in loops}
    largest = max(max(v) for v in stick.values())
    passed = largest <= stiff.ei_max_Nm2 and min(slip.values()) >= stiff.ei_min_Nm2
    return {
        "ei_max_Nm2": stiff.ei_max_Nm2,
        "ei_plane_section_Nm2": stiff.ei_plane_section_Nm2,
        "ei_min_Nm2": stiff.ei_min_Nm2,
        "xi": dict(stiff.xi_per_layer),
        "stick_true_Nm2": stick,
        "slip_Nm2": slip,
        "pass": passed,
        "pass_plane_section": largest <= stiff.ei_plane_section_Nm2,
    }


def predict(
    make: Section,
    spec: CoreFrictionSpec | None,
    pressure_Pa: float,
    *,
    reading: str = "R2",
    mu: float = MU,
    figure: str = "5_6",
) -> dict[int, dict[str, float]]:
    """Work, branch RMSE and moment range against the measured loops."""
    base = load_base()
    scope = "all_contacts" if spec is not None else "armour_only"
    section = make(with_pressure(base, pressure_Pa, scope=scope), spec, mu)
    loops = measured_loops(figure)
    out = {}
    for amp in AMPLITUDES_MM:
        mod = model_loop(section, amp, reading=reading)
        meas = loops[amp]
        out[amp] = {
            "work": mod.loop.work,
            "work_error": mod.loop.work / meas.work - 1.0,
            "rmse": branch_rmse(meas, mod.loop),
            "range_error": mod.loop.moment_range / meas.moment_range - 1.0,
            "half_height": mod.loop.half_height,
            "stick_secant": float(np.mean(mod.loop.stick_secant)),
            "slip_slope": mod.loop.slip_slope,
            "kappa_centre_peak": mod.kappa_centre_peak,
            "deflection_ratio": mod.deflection_ratio,
        }
    return out


def z_id_pass(rows: dict[int, dict[str, float]]) -> bool:
    return rows[IDENTIFY_MM]["rmse"] < 0.05 and all(
        abs(rows[a]["work_error"]) <= 0.20
        and rows[a]["rmse"] < 0.10
        and abs(rows[a]["range_error"]) <= 0.10
        for a in AMPLITUDES_MM
        if a != IDENTIFY_MM
    )


def z_id(
    make: Section = packaged_section,
    spec: CoreFrictionSpec | None = FULL,
    *,
    reading: str = "R2",
    mu: float = MU,
) -> dict[str, object]:
    """Registered row Z-ID (packaged law) or one of its reported variants."""
    target = measured_loops()[IDENTIFY_MM].work
    ident = identify(make, spec, target, reading=reading, mu=mu)
    if ident.pressure_pa is None:
        return {"identification": ident, "rows": None, "pass": False}
    rows = predict(make, spec, ident.pressure_pa, reading=reading, mu=mu)
    return {"identification": ident, "rows": rows, "pass": z_id_pass(rows)}


def bilinear_half_height() -> float:
    """Half-height of a kinematic bilinear loop with the published fit."""
    k1, k2, my = BILINEAR["K1_Nm2"], BILINEAR["K2_Nm2"], BILINEAR["My_Nm"]
    return (1.0 - k2 / k1) * my


def bilinear_work(amplitude: float) -> float:
    """Work of a kinematic bilinear loop of curvature amplitude ``amplitude``."""
    k1, my = BILINEAR["K1_Nm2"], BILINEAR["My_Nm"]
    return 4.0 * bilinear_half_height() * (amplitude - my / k1)


def _fmt_rows(rows: dict[int, dict[str, float]]) -> str:
    return "; ".join(
        f"{a} mm W {r['work']:.0f} J/m ({r['work_error'] * 100:+.1f} %), "
        f"RMSE {r['rmse'] * 100:.1f} %, range {r['range_error'] * 100:+.1f} %"
        for a, r in rows.items()
    )


def main() -> None:
    warnings.simplefilter("ignore")
    loops = measured_loops()
    first = measured_loops("5_5")
    print("Measured (Fig. 5.6 stabilised; Fig. 5.5 first series):")
    for a in AMPLITUDES_MM:
        m, f = loops[a], first[a]
        print(
            f"  {a} mm: W {m.work:.1f} J/m ({min(m.works):.1f}-{max(m.works):.1f}), "
            f"range {m.moment_range:.0f} N m, centre {m.centre:.0f}, half-height "
            f"{m.half_height:.0f}, stick {m.stick_secant[0] / 1e3:.0f}/"
            f"{m.stick_secant[1] / 1e3:.0f} kN m2 (plotted), slip "
            f"{m.slip_slope / 1e3:.2f} kN m2 | first series W {f.work:.1f}"
        )
    for a in AMPLITUDES_MM:
        x, w = load_deflection(a)
        print(
            f"  Fig. 18 {a} mm: central curvature {central_curvature(x, w):.4f} 1/m "
            f"(plotted {EQ51 * a * 1e-3:.4f}); w(0.475) "
            f"{np.interp(SHEAR_SPAN_M, x, w) * 1e3:.1f} mm, w(1.0) "
            f"{np.interp(0.999, x, w) * 1e3:.1f} mm"
        )
    print(
        f"  bilinear fit (K1 106, K2 9, My 1.83): half-height "
        f"{bilinear_half_height():.0f} N m; work "
        + ", ".join(
            f"{a} mm {bilinear_work(EQ51 * a * 1e-3):.0f}" for a in AMPLITUDES_MM
        )
        + " J/m"
    )
    ei = z_ei()
    print("Z-EI:", ei)
    base = load_base()
    for label, p in (("S0", 0.0), ("ST", SOURCE_PRESSURE_PA)):
        for name, spec in (("full", FULL), ("armour", None)):
            cable = with_pressure(
                base,
                p,
                identification="published_calibration",
                source="[Zeidan2026] Sec. 5.3",
            )
            section = packaged_section(cable, spec, MU)
            cells = []
            for a in AMPLITUDES_MM:
                mod = model_loop(section, a).loop
                cells.append(f"{a} mm W {mod.work:.1f}, range {mod.moment_range:.0f}")
            print(f"  {label} {name}: " + "; ".join(cells))
    for label, make, spec, reading, mu in (
        ("Z-ID full accounting", packaged_section, FULL, "R2", MU),
        ("armour only", packaged_section, None, "R2", MU),
        ("full, reading R1", packaged_section, FULL, "R1", MU),
        ("full, mu 0.24", packaged_section, FULL, "R2", 0.24),
        ("diagnostic (bedding xi), full", concentric_support_section, FULL, "R2", MU),
        ("diagnostic (bedding xi), armour", concentric_support_section, None, "R2", MU),
    ):
        out = z_id(make, spec, reading=reading, mu=mu)
        ident = out["identification"]
        assert isinstance(ident, Identification)
        if out["rows"] is None:
            print(
                f"{label}: P0 not identifiable, largest 40 mm work "
                f"{ident.largest_work:.1f} J/m at {ident.largest_at_pa / 1e3:.0f} kPa"
            )
            continue
        rows = out["rows"]
        assert isinstance(rows, dict)
        print(
            f"{label}: P0 {ident.pressure_pa / 1e3:.1f} kPa; {_fmt_rows(rows)}; "
            f"{'PASS' if out['pass'] else 'FAIL'}"
        )
        r60 = rows[60]
        print(
            f"   60 mm: half-height {r60['half_height']:.0f} N m, stick "
            f"{r60['stick_secant'] / 1e3:.0f} kN m2, slip {r60['slip_slope'] / 1e3:.2f}"
            f" kN m2, central curvature {r60['kappa_centre_peak']:.4f} 1/m, "
            f"deflection ratio {r60['deflection_ratio']:.3f}"
        )
    print("diagnostic xi (bedding only):", concentric_xi(base))


if __name__ == "__main__":
    main()
