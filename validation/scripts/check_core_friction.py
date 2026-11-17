# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Gap R1b: published comparisons of the laid-core stick-slip law.

Targets were registered (doc/template_66kv.md Section 4.1, doc/benchmarks.md
"Laid-core friction") before this script was run; it reports, it does not fit.

1. IEA Wind Task 49 / Janocha et al. (2024) 66 kV template, 50 kN:
   Fig. 6 (normal mesh, +/-0.2 1/m): loop half-height at kappa = 0 about
   1.0 kN m, moment at 0.2 1/m about 6.8 kN m, branch slope about 29 kN m^2;
   IEA Table 13 slipping EI 42.47 kN m^2; Fig. 12b monotonic curves: moment at
   the knee (0.01 1/m) about 5.8 / 9.5 / 12.5 kN m at 10 / 100 / 200 kN and
   slope 0.1-0.4 1/m about 41 / 41 / 44 kN m^2. Friction coefficients are the
   source's Table 2 (MDPE 0.46; copper 0.30 has no resolved conductor here).
   Pass criteria: half-height and knee moments within 30 % (figure reading),
   slopes within 15 %.
2. Menard and Cartraud (2023), six measured ranges (Fig. 16), transferred
   inputs (mu 0.2, P 0.3 MPa, bounded chain), core friction mu 0.2 for every
   core contact as in the source FE (Section 5), no recalibration. Gate
   BM1-loop: range (b) ordered RMSE < 5 % and matched cycle work within 10 %.
3. Su et al. (2026) 35 kV: no bending loop is published (axial and torsional
   results only; the analytical model omits friction), so no comparison.

Run: ``PYTHONPATH=. python validation/scripts/check_core_friction.py`` (one thread).
"""

from __future__ import annotations

import math
import time
import warnings
from pathlib import Path

import numpy as np

import cablexsec as cx
from cablexsec.analysis.curve_comparison import cycle_work, insert_cycle_crossings
from cablexsec.api import CoreFrictionSpec

ROOT = Path(__file__).resolve().parents[2]
JANOCHA = CoreFrictionSpec(mu_core_filler=0.46, mu_core_core=0.46, mu_core_sheath=0.46)


def _descending_zero(k: np.ndarray, m: np.ndarray) -> float:
    i = int(np.flatnonzero((k[:-1] > 0) & (k[1:] <= 0))[-1])
    w = k[i] / (k[i] - k[i + 1])
    return float(m[i] + w * (m[i + 1] - m[i]))


def verification_checks() -> dict[str, float]:
    """Scorecard gate CF (66 kV template, 50 kN, solved contact).

    ``mu0``: largest moment difference to the armour-only loop at zero core
    friction [N m] (bitwise: 0). ``inf``: relative difference of the tangent
    and EI_max to compute_stiffness().ei_max_Nm2 at large friction and rigid
    contact. ``closed``: relative error of the full-slip core moment against
    ``(2/pi) n q r^2 cos^2 b / sin^2 b [(a e^(a pi/2) + 1)/(1 + a^2) - 1]``,
    ``a = mu sin b``. ``dissipation``: smallest friction-work increment on a
    random biaxial walk relative to the total (>= 0). ``rotation``: moment
    equivariance residual for a rotation by whole station spacings.
    """
    from cablexsec.mechanics.core_friction import (  # noqa: PLC0415
        build_core_friction,
        core_friction_history,
    )

    cable = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
    k = 0.2 * np.sin(np.linspace(0.0, 4.0 * math.pi, 801))
    base = cx.compute_mkappa(cable, k, tension_N=50e3, contact_model="solved")
    free = CoreFrictionSpec(mu_core_filler=0.0, mu_core_core=0.0, mu_core_sheath=0.0)
    zero = cx.compute_mkappa(
        cable, k, tension_N=50e3, contact_model="solved", core_friction=free
    )
    mu0 = float(np.max(np.abs(zero.moment_Nm - base.moment_Nm)))

    rigid = CoreFrictionSpec(
        mu_core_filler=1e3, mu_core_core=1e3, mu_core_sheath=1e3, core_shear=False
    )
    tiny = 1e-6 * np.sin(np.linspace(0.0, 2.0 * math.pi, 201))
    stuck = cx.compute_mkappa(cable, tiny, tension_N=50e3, mu=1e3, core_friction=rigid)
    reference = cx.compute_stiffness(cable).ei_max_Nm2
    mask = np.abs(tiny) > 1e-9
    inf = max(
        abs(stuck.EI_max_Nm2 / reference - 1.0),
        float(np.max(np.abs(stuck.moment_Nm[mask] / tiny[mask] / reference - 1.0))),
    )

    model = build_core_friction(cable, JANOCHA, 50e3, contact_model="solved")
    q, b, r = model.loads.total_N_per_m, model.lay_angle_rad, model.r_offset_m
    a = model.mu_effective * math.sin(b)
    # Full slip after a reversal, signed capstan bound: the cores of one side
    # at -sigma0 (1 - exp(-a psi)), the others at sigma0 expm1(a psi); the
    # moment integrand is their half sum, sinh(a psi) sin(psi), and
    # int_0^{pi/2} sinh(a psi) sin(psi) dpsi = a cosh(a pi/2) / (1 + a^2).
    integral = a * math.cosh(a * math.pi / 2) / (1.0 + a * a)
    expected = (
        2 / math.pi * model.n_cores * q * r * r * math.cos(b) ** 2 / math.sin(b) ** 2
    )
    expected *= integral
    long = 0.2 * np.sin(np.linspace(0.0, 4.0 * math.pi, 4001))
    core = core_friction_history(model, long, np.zeros_like(long)).core_moment_Nm[:, 0]
    closed = abs(-_descending_zero(long, core) / expected - 1.0)

    rng = np.random.default_rng(3)
    kx = np.cumsum(rng.normal(scale=0.01, size=600))
    ky = np.cumsum(rng.normal(scale=0.01, size=600))
    walk = core_friction_history(model, kx, ky)
    friction = walk.work.friction_J_per_m
    dissipation = float(np.min(np.diff(friction)) / friction[-1])
    angle = 2 * math.pi * 7 / model.population.n_wires
    c, s = math.cos(angle), math.sin(angle)
    turned = core_friction_history(model, c * kx - s * ky, s * kx + c * ky)
    expected_m = walk.moment_Nm @ np.array([[c, s], [-s, c]])
    rotation = float(
        np.max(np.abs(turned.moment_Nm - expected_m)) / np.max(np.abs(walk.moment_Nm))
    )
    return {
        "mu0": mu0,
        "inf": inf,
        "closed": closed,
        "dissipation": dissipation,
        "rotation": rotation,
    }


def wedge_filler_checks() -> dict[str, float]:
    """Scorecard gate CFW (66 kV template, 50 kN): core load statics and the
    laid filler profiles.

    ``statics``: largest relative residual of the radial equilibrium of one
    core in the bounded estimate, wedged trefoil and central filler, with and
    without a filler (66 kV template and Menard cable): the two neighbour
    contacts times ``sin(pi/n)`` (or the central reaction) against the
    capstan load plus the ring share, and the outer face against the ring
    share. ``mu0``: largest moment difference to the armour-only loop with
    the profiles on and every coefficient zero (bitwise: 0). ``closed``:
    relative error of the full-slip profile moment against the cores' closed
    form with the profile radius, lay angle, both-face load and coefficient
    (solved and bounded, the larger). ``stick``: relative error of the
    small-curvature tangent against ``EI_max`` plus ``E J_f cos^3 beta_f / 2``
    at large friction, rigid contact and no shear.
    """
    from cablexsec.materials import library  # noqa: PLC0415
    from cablexsec.mechanics.axisymmetric import (  # noqa: PLC0415
        compute_contact_forces,
    )
    from cablexsec.mechanics.core_friction import (  # noqa: PLC0415
        _core_geometry,
        _core_layers,
        build_core_friction,
        core_contact_loads,
        core_friction_history,
    )

    cable = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
    menard_cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    tension = 50e3
    statics = 0.0
    for target, filled in ((cable, True), (menard_cable, False)):
        n, r_c, beta, _ = _core_geometry(target)
        axi = compute_contact_forces(
            target, tension, p_ext_Pa=target.initial_state.external_pressure_Pa
        )
        names = {x.name for x in _core_layers(target)}
        share = sum(v for k, v in axi.layer_tension_N.items() if k in names) / n
        capstan = share * math.tan(beta) ** 2 / r_c
        ring = 2 * math.pi * axi.interface_radii_m[0] * axi.interface_pressure_Pa[0] / n
        for central in (False, True):
            loads = core_contact_loads(target, tension, central_filler=central)
            outer = loads.core_filler_N_per_m if filled else loads.core_sheath_N_per_m
            if central:
                reaction = loads.core_filler_N_per_m - (ring if filled else 0.0)
                residuals = (reaction - capstan - ring, loads.core_core_N_per_m)
                if not filled:
                    residuals += (loads.core_sheath_N_per_m - ring,)
            else:
                residuals = (
                    loads.core_core_N_per_m * math.sin(math.pi / n) - capstan - ring,
                    outer - ring,
                )
            statics = max(statics, max(abs(r) for r in residuals) / (capstan + ring))

    k = 0.2 * np.sin(np.linspace(0.0, 4.0 * math.pi, 801))
    base = cx.compute_mkappa(cable, k, tension_N=tension)
    zero = CoreFrictionSpec(
        mu_core_filler=0.0, mu_core_core=0.0, mu_core_sheath=0.0, filler_profiles=True
    )
    res = cx.compute_mkappa(cable, k, tension_N=tension, core_friction=zero)
    mu0 = float(np.max(np.abs(res.moment_Nm - base.moment_Nm)))

    spec = CoreFrictionSpec(
        mu_core_filler=0.46,
        mu_core_core=0.46,
        mu_core_sheath=0.46,
        filler_profiles=True,
    )
    long = 0.2 * np.sin(np.linspace(0.0, 4.0 * math.pi, 4001))
    closed = 0.0
    for route in ("bounded", "solved"):
        model = build_core_friction(cable, spec, tension, contact_model=route)
        f = model.fillers
        assert f is not None
        b, r, q = f.lay_angle_rad, f.r_effective_m, f.line_load_N_per_m
        a = f.mu * math.sin(b)
        expected = 2 / math.pi * f.n_profiles * q * r * r * math.cos(b) ** 2
        expected *= a * math.cosh(a * math.pi / 2) / (1 + a * a) / math.sin(b) ** 2
        hist = core_friction_history(model, long, np.zeros_like(long))
        assert hist.filler_moment_Nm is not None
        closed = max(
            closed,
            abs(-_descending_zero(long, hist.filler_moment_Nm[:, 0]) / expected - 1),
        )

    rigid = CoreFrictionSpec(
        mu_core_filler=1e3,
        mu_core_core=1e3,
        mu_core_sheath=1e3,
        core_shear=False,
        filler_profiles=True,
    )
    tiny = 1e-6 * np.sin(np.linspace(0.0, 2.0 * math.pi, 201))
    stuck = cx.compute_mkappa(
        cable, tiny, tension_N=tension, mu=1e3, core_friction=rigid
    )
    assert stuck.core_friction is not None
    fillers = stuck.core_friction.model.fillers
    assert fillers is not None
    layer = next(x for x in cable.layers if x.name == fillers.layer_name)
    filler_ei = (
        0.5
        * library.get_property(layer.material, "E_Pa")
        * fillers.area_m2
        * fillers.r_effective_m**2
        * math.cos(fillers.lay_angle_rad) ** 3
    )
    reference = cx.compute_stiffness(cable).ei_max_Nm2 + filler_ei
    mask = np.abs(tiny) > 1e-9
    stick = float(np.max(np.abs(stuck.moment_Nm[mask] / tiny[mask] / reference - 1)))
    return {"statics": statics, "mu0": mu0, "closed": closed, "stick": stick}


def _branch_slope(k: np.ndarray, m: np.ndarray, lo: float, hi: float) -> float:
    """Slope of the last ascending branch between two curvatures."""
    start = int(np.flatnonzero((k[:-1] < 0) & (k[1:] >= 0))[-1])
    kb, mb = k[start:], m[start:]
    stop = int(np.argmax(kb))
    kb, mb = kb[: stop + 1], mb[: stop + 1]
    return float((np.interp(hi, kb, mb) - np.interp(lo, kb, mb)) / (hi - lo))


def janocha() -> None:
    cable = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
    k = 0.2 * np.sin(np.linspace(0.0, 4.0 * math.pi, 1601))
    print("Janocha 66 kV, 50 kN, +/-0.2 1/m (Fig. 6; IEA Table 13)")
    print("  targets: half-height ~1.0 kN m, M(0.2) ~6.8 kN m, slope ~29 kN m2,")
    print("           slipping EI 42.47 kN m2 (IEA Table 13)")
    for model in ("bounded", "solved"):
        for label, spec in (("armour only", None), ("core friction", JANOCHA)):
            t0 = time.perf_counter()
            res = cx.compute_mkappa(
                cable, k, tension_N=50e3, contact_model=model, core_friction=spec
            )
            dt = time.perf_counter() - t0
            m = res.moment_Nm
            extra = ""
            if res.core_friction is not None:
                mod = res.core_friction.model
                extra = (
                    f" | q_core {mod.loads.total_N_per_m / 1e3:.1f} kN/m, "
                    f"xi {mod.xi:.3f}, core stick {mod.core_stick_EI_Nm2 / 1e3:.0f} kN m2"
                )
            print(
                f"  {model:7s} {label:13s}: half-height {-_descending_zero(k, m) / 1e3:.3f} "
                f"kN m, M(0.2) {m.max() / 1e3:.2f} kN m, slope(0.05-0.15) "
                f"{_branch_slope(k, m, 0.05, 0.15) / 1e3:.1f} kN m2, EI_min "
                f"{res.EI_min_Nm2 / 1e3:.1f}, EI_max {res.EI_max_Nm2 / 1e3:.0f} kN m2, "
                f"W {res.W_diss_J_per_m:.0f} J/m (2 cycles), {dt:.3f} s{extra}"
            )
    print(
        "Janocha Fig. 12b monotonic, targets M(0.01) ~5.8/9.5/12.5 kN m, slope ~41/41/44"
    )
    ramp = np.linspace(0.0, 0.4, 801)
    for tension in (10e3, 100e3, 200e3):
        row = []
        for spec in (None, JANOCHA):
            res = cx.compute_mkappa(
                cable,
                ramp,
                tension_N=tension,
                contact_model="solved",
                core_friction=spec,
            )
            m01 = float(np.interp(0.01, ramp, res.moment_Nm))
            slope = float(
                (
                    np.interp(0.4, ramp, res.moment_Nm)
                    - np.interp(0.1, ramp, res.moment_Nm)
                )
                / 0.3
            )
            row.append(f"M(0.01) {m01 / 1e3:.2f} kN m, slope {slope / 1e3:.1f} kN m2")
        print(
            f"  {tension / 1e3:5.0f} kN: armour only {row[0]}; core friction {row[1]}"
        )
    # Normal load that would close the Fig. 6 half-height (linear in q).
    base = cx.compute_mkappa(cable, k, tension_N=50e3, contact_model="solved")
    cored = cx.compute_mkappa(
        cable, k, tension_N=50e3, contact_model="solved", core_friction=JANOCHA
    )
    q = cored.core_friction.model.loads.total_N_per_m
    per_q = (
        _descending_zero(k, base.moment_Nm) - _descending_zero(k, cored.moment_Nm)
    ) / q
    need = (1.0e3 + _descending_zero(k, base.moment_Nm)) / per_q
    print(
        f"  core line load needed for a 1.0 kN m half-height: {need / 1e3:.0f} kN/m per core "
        f"(solved {q / 1e3:.1f} kN/m)"
    )


def menard() -> None:
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    print("Menard and Cartraud (2023) Fig. 16, transferred inputs (mu 0.2, P 0.3 MPa)")
    for tag, crossing in zip("abcdef", (0.0, 0.0, None, None, None, None), strict=True):
        data = np.genfromtxt(
            ROOT / "validation" / "paper_data" / f"menard_fig16{tag}_ordered.csv",
            delimiter=",",
            names=True,
        )
        kappa = np.asarray(data["curvature_1_per_m"], dtype=float)
        moment = np.asarray(data["moment_Nm"], dtype=float)
        measured = float(np.mean(cycle_work(kappa, moment, crossing=crossing)))
        cells = []
        for spec in (None, True):
            res = cx.compute_mkappa(
                cable, np.tile(kappa, 2), mu=0.2, core_friction=spec
            )
            seq = res.moment_Nm[-kappa.size :]
            rmse = float(np.sqrt(np.mean((seq - moment) ** 2)) / np.ptp(moment) * 100)
            aug, _, cross = insert_cycle_crossings(kappa, crossing=crossing)
            run = cx.compute_mkappa(cable, np.tile(aug, 2), mu=0.2, core_friction=spec)
            mech = (run.work.stored_J_per_m + run.work.friction_J_per_m)[-aug.size :]
            work = float(np.mean(np.diff(mech[cross])))
            cells.append(
                f"RMSE {rmse:5.2f} %, W {work:5.1f} J/m ({(work / measured - 1) * 100:+5.1f} %)"
            )
        print(
            f"  ({tag}) measured W {measured:5.1f} J/m | armour only {cells[0]} | core friction {cells[1]}"
        )
    mod = cx.compute_mkappa(
        cable,
        0.1 * np.sin(np.linspace(0, 4 * math.pi, 401)),
        mu=0.2,
        core_friction=True,
    ).core_friction.model
    print(
        f"  bounded ring load on a core's outer face "
        f"{mod.loads.core_sheath_N_per_m / 1e3:.1f} kN/m (source FE: 16.5 kN/m "
        f"on a core, about a third of its 46 kN/m inner-sheath load); all "
        f"contacts of a core, wedged trefoil "
        f"{mod.loads.total_N_per_m / 1e3:.1f} kN/m; xi {mod.xi:.3f}"
    )


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    print("Verification (gate CF):", verification_checks())
    print("Verification (gate CFW):", wedge_filler_checks())
    janocha()
    menard()
