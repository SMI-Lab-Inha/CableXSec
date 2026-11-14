# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Restrictor kinematics, screens, tape, laminate and armour: Thies (2016).

Example of gaps P1, P2, P5, P6, P7 and P8 on the public reconstruction of the
18/30 kV three-core cable tested in an articulated bend restrictor by Thies
et al. (2016) (``cx.thies2016_yaml_path()``; every unpublished input is an
assumed nominal with a stated bound). Load blocks from [Thies2016] Table 6:
headstock angle, Z-ram tension at the largest angle (Fig. 5: tension is
smallest there) and cycles.

Part A uses constant-radius models: the cable bends at ``1/R`` over the
length ``theta R`` from the headstock, ``R`` 2.54 m (locking at the
published MBR) or the radii read from the test video (2.0/3.5/7.0 m).

Part B (gap P5) replaces them with the restrictor kinematics: the
non-proportional field ``kappa(x, t)`` of the articulated restrictor
(15 shells of pitch 0.36 m, bore 137.5 mm) under the headstock rotation and
the Z-ram tension, for the two readings of the published restrictor data
(lock angle from the stated MBR 2.54 m, or the stated 4.75 deg per joint,
which gives 4.34 m), with the secant bending stiffness of the cable model.
The field drives the Al laminate (strain life), the Cu tape (gap P2), the
screen wires (gap P7 strain-life route beside the elastic S-N) and the
armour at the bolted headstock clamp (rig solver with the field).

Run: ``python examples/thies2016_screens_sheaths.py`` (a few minutes).
"""

from __future__ import annotations

import math
import os
import warnings

for _key in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np  # noqa: E402

import cablexsec as cx  # noqa: E402
from cablexsec.analysis.fretting_assessment import SECONDS_PER_YEAR  # noqa: E402
from cablexsec.materials.sn_curves import get_sn_curve  # noqa: E402

BLOCKS = ((28.0, 1167), (14.0, 25014), (14.0, 24990), (7.0, 24990))
#: Z-ram tension at the largest angle per block [N] ([Thies2016] Table 6).
BLOCK_TENSION_N = (20e3, 10e3, 15e3, 15e3)
VIDEO_RADIUS_M = {28.0: 2.0, 14.0: 3.5, 7.0: 7.0}
CURVATURE_MODELS = ("lock_2.54", "spread_video")
SPECIMEN_M = 5.4
CABLE_OD_M = 0.123
STATIONS_M = (0.05, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0)
RESTRICTORS = {
    "mbr_2.54": cx.ArticulatedRestrictor.from_mbr(
        2.54,
        pitch_m=0.36,
        n_elements=15,
        bore_diameter_m=0.1375,
        source="[Thies2016] Table 1 MBR 2.54 m; pitch 5.4 m / 15 shells",
    ),
    "angle_4.75": cx.ArticulatedRestrictor(
        pitch_m=0.36,
        lock_angle_rad=math.radians(4.75),
        n_elements=15,
        bore_diameter_m=0.1375,
        source="[Thies2016] Table 1 bending angle 4.75 deg per joint",
        stated_mbr_m=2.54,
    ),
}


def radius(model: str, angle_deg: float) -> float:
    return 2.54 if model == "lock_2.54" else VIDEO_RADIUS_M[angle_deg]


def blocks_at(model: str, x_m: float) -> list[tuple[int, float]]:
    """(cycles, curvature) of the blocks that bend station ``x``."""
    out = []
    for angle, cycles in BLOCKS:
        r = radius(model, angle)
        if x_m <= math.radians(angle) * r + 1e-12:
            out.append((cycles, 1.0 / r))
    return out


def record(kappa: float, cycles: int = 4) -> np.ndarray:
    t = np.linspace(0.0, float(cycles), cycles * 96 + 1)
    return np.asarray(kappa * np.cos(2.0 * np.pi * t))


def secant_ei(cable: cx.CableData, kappa: float) -> float:
    history = kappa * np.sin(np.linspace(0.0, 4.0 * math.pi, 161))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loop = cx.compute_mkappa(cable, history, tension_N=0.0, contact_model="bounded")
    return float(loop.EI_apparent_Nm2)


def block_field(
    cable: cx.CableData,
    restrictor: cx.ArticulatedRestrictor,
    angle: float,
    tension: float,
    cycles: int = 4,
) -> tuple[cx.CurvatureField, float, np.ndarray]:
    """Field of ``cycles`` headstock cycles, EI iterated to the secant value."""
    t = np.linspace(0.0, float(cycles), cycles * 48 + 1)
    rotation = math.radians(angle) * np.cos(2.0 * np.pi * t)
    ei = cx.compute_stiffness(cable).ei_min_Nm2
    for _ in range(4):
        probe = cx.restrictor_curvature_field(
            restrictor,
            [math.radians(angle)],
            cable_od_m=CABLE_OD_M,
            bending_stiffness_Nm2=ei,
            tension_N=tension,
            span_m=SPECIMEN_M,
        )
        new = secant_ei(cable, probe.forward_peak())
        if abs(new - ei) <= 1e-3 * ei:
            break
        ei = new
    field = cx.restrictor_curvature_field(
        restrictor,
        rotation,
        cable_od_m=CABLE_OD_M,
        bending_stiffness_Nm2=ei,
        tension_N=tension,
        span_m=SPECIMEN_M,
        x_m=np.linspace(0.0, SPECIMEN_M, 541),
        time_s=t,
    )
    return field, ei, t


def part_a(cable: cx.CableData) -> None:
    print("A. Constant-radius models")
    print("Laminate strain amplitude at 28 deg, R 2.54 m, by core slip state")
    k = record(1.0 / 2.54)
    for state in ("free", "frictional", "stuck"):
        field = cx.compute_core_layer_strain(cable, "al_laminate", k, core_state=state)
        print(f"  {state:10s} {100 * field.maximum()['strain_amplitude']:.3f} %")
    print("Al laminate strain-life damage (free core, assumed MUS constants)")
    per_cycle: dict[float, float] = {}
    for model in CURVATURE_MODELS:
        row = []
        for x in (0.05, 0.6, 0.9, 1.2):
            damage = 0.0
            for cycles, kappa in blocks_at(model, x):
                if kappa not in per_cycle:
                    res = cx.compute_shell_fatigue(
                        cable,
                        "al_laminate",
                        record(kappa),
                        method="strain_life",
                        core_state="free",
                    )
                    per_cycle[kappa] = res.D_miner / res.strain_life.n_cycles
                damage += cycles * per_cycle[kappa]
            row.append(f"x {x:.2f} m: D {damage:.3g}")
        print(f"  {model:12s} " + ", ".join(row))
    print("Copper wire screen, free core, R 2.54 m (per cycle)")
    for label, mu_outer in (("library mu_on_cu 0.20", None), ("mu Cu-Cu 0.25", 0.25)):
        kappa = record(1.0 / 2.54, 10)
        res = cx.compute_core_screen_fatigue(
            cable,
            kappa,
            core_state="free",
            duration_s=10.0,
            mu=0.2,
            mu_outer=mu_outer,
            n_stations=4,
        )
        crit = res.critical()
        per_year = SECONDS_PER_YEAR / 10.0
        cycles = sum(n for n, _ in blocks_at("lock_2.54", 0.05))
        face = res.faces[0]["outer_face"]  # core on the extension side
        fretting = face.D_fretting_per_year / per_year / 10.0 * cycles
        print(
            f"  {label}: stress range {float(crit['stress_range_max_Pa']) / 1e6:.0f} MPa, "
            f"slip {1e6 * float(crit['slip_amplitude_max_m']):.0f} um, "
            f"bulk D/cycle {float(crit['D_bulk_max']) / 10:.2g}; x 0.05 m, lock 2.54: "
            f"bulk D {float(crit['D_bulk_max']) / 10 * cycles:.2g}, fretting D max "
            f"{np.max(fretting):.3g}, wires with fretting D >= 1: "
            f"{int(np.sum(fretting >= 1.0))}"
        )


def part_b(cable: cx.CableData) -> None:
    print("\nB. Restrictor kinematics (gap P5)")
    sn = get_sn_curve("oc2017_armour_in_air")
    for name, restrictor in RESTRICTORS.items():
        print(
            f"\nRestrictor {name}: R_L {restrictor.locking_radius_m:.3f} m, lock "
            f"{math.degrees(restrictor.lock_angle_rad):.2f} deg, cable lock radius "
            f"{restrictor.cable_locking_radius_m(CABLE_OD_M):.3f} m"
        )
        fields = []
        for (angle, cycles), tension in zip(BLOCKS, BLOCK_TENSION_N, strict=True):
            field, ei, t = block_field(cable, restrictor, angle, tension)
            fields.append((angle, cycles, tension, field, t))
            line = (
                f"  {angle:4.0f} deg, T {tension / 1e3:.0f} kN: EI {ei / 1e3:.2f} kN m2, "
                f"lambda {math.sqrt(ei / tension):.2f} m, smallest radius "
                f"{1 / field.forward_peak():.2f} m (video {VIDEO_RADIUS_M[angle]:.1f} m), "
                f"locked elements {int(field.n_locked.max())} "
                f"({float(field.locked_length_m.max()):.2f} m), bent length (half "
                f"peak) {float(field.bent_length_m(0.5).max()):.2f} m"
            )
            if field.peak() > field.forward_peak():
                line += f", reverse radius at the tailstock {1 / field.peak():.2f} m"
            print(line)
            for note in field.notes:
                if "stated MBR" not in note:
                    print(f"      note: {note}")
        al: list[float] = []
        tape_stuck: list[float] = []
        tape_fric: list[float] = []
        screen_sn: list[float] = []
        screen_life: list[float] = []
        cache: dict[tuple[float, float], tuple[float, float, float, float, float]] = {}
        for x in STATIONS_M:
            sums = np.zeros(5)
            for angle, cycles, tension, field, _ in fields:
                k = field.at(x)
                key = (angle, tension)
                if float(np.max(np.abs(k))) <= 0.0:
                    continue
                n_rec = 4.0
                shell = cx.compute_shell_fatigue(
                    cable, "al_laminate", k, method="strain_life", core_state="free"
                )
                stuck = cx.compute_tape_fatigue(
                    cable, "cu_tape", k, state="stuck", core_state="free", n_stations=12
                )
                fric = cx.compute_tape_fatigue(
                    cable,
                    "cu_tape",
                    k,
                    state="frictional",
                    core_state="free",
                    n_stations=12,
                    contact_pressure_Pa=4.0e5,
                    mu=0.2,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scr = cx.compute_core_screen_fatigue(
                        cable,
                        k,
                        core_state="free",
                        duration_s=n_rec,
                        mu=0.2,
                        n_stations=4,
                        strain_life="copper_poon2022",
                    )
                assert scr.D_strain_life is not None
                values = (
                    shell.D_miner / n_rec,
                    stuck.D_miner / n_rec,
                    fric.D_miner / n_rec,
                    float(np.max(scr.D_bulk)) / n_rec,
                    float(np.max(scr.D_strain_life)) / n_rec,
                )
                cache[key] = values
                sums += cycles * np.asarray(values)
            al.append(sums[0])
            tape_stuck.append(sums[1])
            tape_fric.append(sums[2])
            screen_sn.append(sums[3])
            screen_life.append(sums[4])

        def fmt(values: list[float]) -> str:
            return ", ".join(
                f"{x:.2f}: {v:.3g}" for x, v in zip(STATIONS_M, values, strict=True)
            )

        print("  Damage over the blocks by station x [m] (free core):")
        print(f"    Al laminate (strain life, MUS assumed): {fmt(al)}")
        print(f"    Cu tape stuck (Poon 2022):              {fmt(tape_stuck)}")
        print(f"    Cu tape frictional (P1+P2 0.4 MPa):     {fmt(tape_fric)}")
        print(f"    Screen wires, elastic S-N:              {fmt(screen_sn)}")
        print(f"    Screen wires, strain life (Poon 2022):  {fmt(screen_life)}")
        # Armour at the bolted headstock clamp, field-driven rig solver
        damage = {0.02: 0.0, 0.1: 0.0, 0.25: 0.0, 0.5: 0.0}
        peak_range = 0.0
        for _angle, cycles, _tension, field, t in fields:
            prot = cx.RigProtocol.from_curvature_field(
                field, time_s=t, tension_N=0.0, reference_x_m=0.0, retain_from_s=3.0
            )
            gauges = [
                cx.GaugeLocation("armour", a, xx)
                for a in range(0, 360, 30)
                for xx in damage
            ]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = cx.simulate_rig_test(
                    cable,
                    prot,
                    cx.RigSetup(
                        free_length_m=SPECIMEN_M,
                        profile="field",
                        curvature_field=field,
                        reference_x_m=0.0,
                    ),
                    gauges=gauges,
                    contact_model="bounded",
                    mu_factors=(),
                )
            for xx in damage:
                worst = (
                    max(g.ranges["total"] for g in res.gauges if g.location.x_m == xx)
                    * 1e-6
                    * 210e9
                    / 1e6
                )
                peak_range = max(peak_range, worst)
                damage[xx] += cycles / float(sn.cycles_to_failure(worst))
        print(
            f"    Armour (anchored at headstock, OC2017 in-air design curve): peak range "
            f"{peak_range:.0f} MPa; D "
            + ", ".join(f"x {xx:.2f}: {d:.3g}" for xx, d in damage.items())
        )


def main() -> None:
    cable = cx.load_cable(cx.thies2016_yaml_path(), geometry_override=None)
    part_a(cable)
    part_b(cable)


if __name__ == "__main__":
    main()
