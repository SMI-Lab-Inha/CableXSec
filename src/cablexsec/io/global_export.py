# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Files of section properties for global analysis programs.

Every file starts with a header of comment lines: the program and version, the
cable, the fingerprint of the cable and settings it came from, each load case
(effective tension, end condition for twist, temperature, pressure and the
tension the section models received), the units, the source of every
value and every computed value that differs from the cable's published one
by more than the tolerance (:func:`published_lines`). Writers return text;
nothing here computes physics.

* :func:`section_properties_csv`: one row per load case with every value and
  its unit, a neutral table for any global analysis program.
* :func:`bending_tables_csv`: the moment-curvature backbone of every case.
* :func:`cabledyn_bending_law`: the CableDyn bending law
  (``cablexsec.bending_law/1``, the format of
  :meth:`~cablexsec.analysis.bending_law.BendingLawExport.to_json`), for
  cases that differ in tension only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from cablexsec.analysis.bending_law import BendingLawExport, IwanBendingLaw
from cablexsec.analysis.section_properties import (
    SectionCaseResult,
    SectionPropertiesResult,
)
from cablexsec.exceptions import CableInputError
from cablexsec.provenance import Provenance

__all__ = [
    "bending_tables_csv",
    "cabledyn_bending_law",
    "header_lines",
    "published_lines",
    "section_properties_csv",
]


def _version() -> str:
    from cablexsec import __version__  # noqa: PLC0415 - the package imports io

    return __version__


def _axial_source(case: SectionCaseResult) -> str:
    if case.route == "solved":
        return (
            f"tangent at the case state, twist {case.case.twist}, solved radial "
            "section with unilateral contact"
        )
    return (
        f"fixed-radius assembly, twist {case.case.twist}, no contact, independent "
        "of the tension (solved section not used, see above)"
    )


def _torsion_source(case: SectionCaseResult) -> str:
    if case.route == "solved":
        return (
            "tangent at constant tension at the case state, solved radial "
            f"section; secant over a {case.torque_step_Nm:g} N m step: "
            f"{case.torsional_stiffness_positive_Nm2:.6g} N m2 positive twist, "
            f"{case.torsional_stiffness_negative_Nm2:.6g} N m2 negative twist "
            "(positive twist stretches right-hand laid wires)"
        )
    return (
        "lay contribution only (extension of laid wires and cores), the same "
        "in both twist directions; component torsion and radial relaxation "
        "are not included"
    )


def _words(text: str) -> str:
    """Identifiers in a sentence as words (``cu_conductor`` as ``cu conductor``)."""
    return text.replace("_", " ")


def _bending_source(result: SectionPropertiesResult, case: SectionCaseResult) -> str:
    mu = result.settings.get("mu")
    friction = "the armour materials' friction" if mu is None else f"mu {mu:g}"
    cores = case.cores
    stage = (
        ""
        if cores is None
        else "; core lay stage: the laid cores stick and slip (core friction "
        f"{cores.mu:.3g}, stick coefficient {cores.xi:.3g}, contact line load "
        f"{cores.contact_line_load_N_per_m:.6g} N/m per core, slip from "
        f"{cores.onset_curvature_1_m:.6g} to {cores.full_slip_curvature_1_m:.6g} "
        "1/m)"
    )
    return (
        "monotonic backbone of the Papailiou slip-front law of the armour "
        f"wires ({case.bending_contact} contact model, {friction}){stage}; stick "
        f"tangent {case.ei_stick_Nm2:.6g} N m2, full-slip tangent "
        f"{case.ei_slip_Nm2:.6g} N m2"
    )


def header_lines(
    result: SectionPropertiesResult,
    fingerprint: str,
    *,
    title: str,
    units: str,
    cases: Sequence[SectionCaseResult] | None = None,
) -> list[str]:
    """Header comment lines (without the comment mark) shared by every file."""
    lines = [
        f"{title}",
        f"Written by CableXSec {_version()}",
        f"Cable: {result.cable_name}",
        f"Fingerprint (cable and settings): {fingerprint or 'not given'}",
        f"Units: {units}",
        "Tension: effective tension; the tension given to the section models "
        "follows the annulus setting "
        f"({result.settings.get('annulus', 'flooded')})",
    ]
    chosen = list(cases if cases is not None else result.cases)
    for case in chosen:
        lines.append(
            f"Load case {case.case.name}: {case.case.describe()}; tension given "
            f"to the section {case.wall_tension_N:.6g} N; axial and torsional "
            + (
                "values from the solved radial section"
                if case.route == "solved"
                else "values from the fixed-radius assembly"
            )
        )
    reasons = {case.route_reason for case in chosen if case.route != "solved"}
    for reason in sorted(reasons):
        lines.append(
            "Solved radial section not used: "
            + (_words(reason) if reason else "not requested")
        )
    return lines + published_lines(result)


def published_lines(result: SectionPropertiesResult) -> list[str]:
    """Header lines of the computed values that differ from the published ones.

    Every flagged difference (above the tolerance) is stated with both
    values and the computed counterpart; the published values within the
    tolerance are named on one line. No lines when the cable publishes none.
    """
    if not result.published:
        return []
    tolerance = result.published[0].tolerance
    flagged = result.published_flagged
    lines: list[str] = []
    if flagged:
        lines.append(
            "WARNING: computed values differ from the cable's published values "
            f"by more than {100.0 * tolerance:g} %; the cable data are used as "
            "given:"
        )
        lines += [f"  {item.describe()}" for item in flagged]
    within = [item.quantity.lower() for item in result.published if not item.flagged]
    if within:
        lines.append(
            f"Within {100.0 * tolerance:g} % of the published values: "
            + "; ".join(within)
        )
    return lines


def _comment(lines: Sequence[str], mark: str = "# ") -> str:
    return "\n".join(f"{mark}{line}".rstrip() for line in lines)


_COLUMNS: tuple[tuple[str, str], ...] = (
    ("effective tension", "N"),
    ("tension given to the section", "N"),
    ("axial stiffness tangent", "N"),
    ("axial stiffness secant from zero tension", "N"),
    ("torque from the tension twist held", "N m"),
    ("twist from the tension twist free", "rad/m"),
    ("torsional stiffness tangent at constant tension", "N m2"),
    ("torsional stiffness positive twist", "N m2"),
    ("torsional stiffness negative twist", "N m2"),
    ("bending stiffness all armour wires stick", "N m2"),
    ("bending stiffness all armour wires slip", "N m2"),
)


def _case_values(case: SectionCaseResult) -> list[float]:
    return [
        case.case.tension_N,
        case.wall_tension_N,
        case.axial_stiffness_N,
        case.axial_stiffness_secant_N,
        case.torque_change_Nm,
        case.twist_change_rad_per_m,
        case.torsional_stiffness_tangent_Nm2,
        case.torsional_stiffness_positive_Nm2,
        case.torsional_stiffness_negative_Nm2,
        case.ei_stick_Nm2,
        case.ei_slip_Nm2,
    ]


def _number(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.9g}"


def section_properties_csv(
    result: SectionPropertiesResult, fingerprint: str = ""
) -> str:
    """One row per load case: every value with its unit in the column name."""
    mass = result.mass
    lines = header_lines(
        result,
        fingerprint,
        title="Section properties per load case (neutral table)",
        units="SI (N, m, kg); the unit of each column is in its name",
    )
    lines += [
        "Case-independent values:",
        f"  outer diameter {mass.outer_diameter_m:.9g} m",
        f"  dry mass {_number(mass.mass_kg_per_m)} kg/m (solids only)",
        "  weight in water, voids dry "
        f"{_number(mass.submerged_weight_dry_voids_N_per_m)} N/m; voids flooded "
        f"{_number(mass.submerged_weight_flooded_voids_N_per_m)} N/m (water "
        f"density {mass.water_density_kg_m3:g} kg/m3)",
        "  bending stiffness full stick (plane sections) "
        f"{result.ei_full_stick_Nm2:.9g} N m2",
        "  bending stiffness no slip, shear-compliant support "
        f"{result.ei_no_slip_shear_compliant_Nm2:.9g} N m2",
        "  bending stiffness full slip (individual wires) "
        f"{result.ei_full_slip_Nm2:.9g} N m2",
        "Empty cells: the value is not given for that case (a torque change "
        "needs the twist held, a twist change the twist free; fixed-radius "
        "cases have no secant, torque or twist change)",
    ]
    armour = [_words(slip.layer) for slip in result.cases[0].armour]
    header = ["load case", "twist end", "axial and torsional values"]
    header += [f"{name} [{unit}]" for name, unit in _COLUMNS]
    for layer in armour:
        header += [
            f"{layer} contact pressure P1+P2 [Pa]",
            f"{layer} slip onset curvature [1/m]",
            f"{layer} full slip curvature [1/m]",
        ]
    with_cores = any(case.cores is not None for case in result.cases)
    if with_cores:
        header += [
            "laid cores contact line load per core [N/m]",
            "laid cores slip onset curvature [1/m]",
            "laid cores full slip curvature [1/m]",
        ]
    rows = [",".join(_csv_text(name) for name in header)]
    for case in result.cases:
        route = "solved section" if case.route == "solved" else "fixed radius"
        cells = [_csv_text(case.case.name), case.case.twist, route]
        cells += [_number(v) for v in _case_values(case)]
        for slip in case.armour:
            cells += [
                _number(slip.contact_pressure_Pa),
                _number(slip.onset_curvature_1_m),
                _number(slip.full_slip_curvature_1_m),
            ]
        if with_cores:
            core = case.cores
            cells += (
                ["", "", ""]
                if core is None
                else [
                    _number(core.contact_line_load_N_per_m),
                    _number(core.onset_curvature_1_m),
                    _number(core.full_slip_curvature_1_m),
                ]
            )
        rows.append(",".join(cells))
    return _comment(lines) + "\n" + "\n".join(rows) + "\n"


def _csv_text(value: str) -> str:
    return (
        '"' + value.replace('"', '""') + '"' if any(c in value for c in ',"') else value
    )


def bending_tables_csv(result: SectionPropertiesResult, fingerprint: str = "") -> str:
    """The backbone of every case: curvature, then one moment column per case."""
    lines = header_lines(
        result,
        fingerprint,
        title="Moment-curvature backbones per load case",
        units="curvature 1/m, moment N m",
    )
    for case in result.cases:
        lines.append(f"Case {case.case.name}: {_bending_source(result, case)}")
    kappa = result.cases[0].kappa_1_m
    for case in result.cases[1:]:
        if not np.array_equal(case.kappa_1_m, kappa):
            raise CableInputError("the backbones do not share one curvature grid")
    header = ["curvature [1/m]"] + [
        _csv_text(f"moment {case.case.name} [N m]") for case in result.cases
    ]
    rows = [",".join(header)]
    for index, k in enumerate(kappa):
        rows.append(
            ",".join(
                [f"{k:.9g}"] + [f"{case.moment_Nm[index]:.9g}" for case in result.cases]
            )
        )
    return _comment(lines) + "\n" + "\n".join(rows) + "\n"


def cabledyn_bending_law(
    result: SectionPropertiesResult, fingerprint: str = ""
) -> BendingLawExport:
    """The CableDyn bending law of the cases, interpolated in tension.

    The law interpolates between tensions only, so the cases must differ in
    tension alone (one end condition, temperature and pressure) and have
    different tensions.
    """
    cases = sorted(result.cases, key=lambda c: c.case.tension_N)
    keys = {
        (c.case.twist, c.case.temperature_C, c.case.external_pressure_Pa) for c in cases
    }
    if len(keys) != 1:
        raise CableInputError(
            "the CableDyn bending law interpolates in tension only; the load "
            "cases differ in end condition, temperature or pressure"
        )
    tensions = np.array([c.case.tension_N for c in cases], dtype=np.float64)
    if np.any(np.diff(tensions) <= 0.0):
        raise CableInputError(
            "the CableDyn bending law needs a different tension in every case"
        )
    kappa = cases[0].kappa_1_m
    moments = np.array([c.moment_Nm for c in cases], dtype=np.float64)
    ei_slip = np.array([c.ei_slip_Nm2 for c in cases], dtype=np.float64)
    laws = tuple(
        IwanBendingLaw.from_backbone(kappa, moments[i], float(ei_slip[i]))
        for i in range(len(cases))
    )
    header = header_lines(
        result,
        fingerprint,
        title="CableDyn bending law from CableXSec section properties",
        units="tension N, curvature 1/m, moment N m, stiffness N m2",
        cases=cases,
    )
    return BendingLawExport(
        tension_N=tensions,
        kappa_1_m=kappa.copy(),
        moment_Nm=moments,
        ei_slip_Nm2=ei_slip,
        laws=laws,
        provenance=Provenance(
            method=(
                "virgin moment-curvature backbone of each load case "
                "(compute_section_properties), decomposed into parallel elastic, "
                "perfectly plastic elements (Iwan model; Masing's rules, exact for "
                "symmetric wire slip intervals)"
            ),
            citations=("[Iwan1966]", "[Papailiou1997]"),
            limitations=(
                "planar bending; the per-wire direction-tied biaxial response is "
                "not an isotropic hysteretic law",
                "contact state fixed within one law; between tensions the "
                "backbone is interpolated linearly",
                "beyond the largest exported curvature the full-slip slope continues",
            ),
        ),
        settings={
            "header": header,
            "fingerprint": fingerprint,
            "cable": result.cable_name,
            "load_cases": [c.case.name for c in cases],
            "twist_end": cases[0].case.twist,
            **{k: v for k, v in result.settings.items() if v is not None},
        },
    )
