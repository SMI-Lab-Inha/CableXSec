# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Preflight capability report: unsupported combinations found before a run.

:func:`preflight_request` checks one analysis request against its cable and
load case without running it; :func:`preflight` checks every analysis of a
:class:`~cablexsec.project.Project`. Errors are combinations the package
refuses at run time (so the run would fail); warnings are runs that succeed
but rest on an assumption the user should see first (no declared clamping, a
coarse tension grid for the exported bending law, missing fatigue curves).
The checks mirror the run-time validations; a run can still fail for reasons
that depend on the solution itself (non-convergence, opened contacts).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from cablexsec.cables.schema import CableData, CylindricalOffsetLayer
from cablexsec.cables.schema import scope_entry_points as _scope_entry_points
from cablexsec.clamping import clamping_state
from cablexsec.exceptions import CableSchemaError
from cablexsec.io.series import LoadCaseSeries
from cablexsec.materials.sn_curves import BLOCKED_SN_CURVES, SN_CURVES, SWT_LIFE
from cablexsec.project import _HISTORY_KINDS, AnalysisRequest, Project

__all__ = ["PreflightIssue", "PreflightReport", "preflight", "preflight_request"]

Severity = Literal["error", "warning"]

#: Largest tension spacing of an exported bending law for an interpolation
#: error below about 1 % of the moment range (gap MX, BENCHMARKS).
BENDING_LAW_TENSION_SPACING_N = 20e3


@dataclass(frozen=True)
class PreflightIssue:
    """One finding: ``code`` is stable for programs, ``message`` for people."""

    severity: Severity
    analysis: str
    code: str
    message: str


@dataclass(frozen=True)
class PreflightReport:
    """All findings of a preflight check."""

    issues: tuple[PreflightIssue, ...]

    @property
    def errors(self) -> tuple[PreflightIssue, ...]:
        return tuple(x for x in self.issues if x.severity == "error")

    @property
    def warnings(self) -> tuple[PreflightIssue, ...]:
        return tuple(x for x in self.issues if x.severity == "warning")

    @property
    def ok(self) -> bool:
        """True when no finding would stop a run."""
        return not self.errors

    def to_text(self) -> str:
        if not self.issues:
            return "preflight: no findings"
        return "\n".join(
            f"{x.severity.upper():7s} {x.analysis}: [{x.code}] {x.message}"
            for x in self.issues
        )


def _offset_cores(cable: CableData) -> bool:
    return any(
        isinstance(layer, CylindricalOffsetLayer)
        and (layer.n_cores != 1 or (layer.r_offset_m or 0.0) != 0.0)
        for layer in cable.layers
    )


def _strand_bundle(cable: CableData) -> bool:
    return any(
        isinstance(layer, CylindricalOffsetLayer)
        and (bool(layer.strand_layers) or layer.strand_lay_angle_deg not in (None, 0.0))
        for layer in cable.layers
    )


def preflight_request(
    cable: CableData,
    request: AnalysisRequest,
    series: LoadCaseSeries | None = None,
) -> tuple[PreflightIssue, ...]:
    """Findings for one analysis request on one cable and load case."""
    from cablexsec.cable_adapters import require_fully_specified  # noqa: PLC0415

    name, settings = request.name, request.settings
    kind = settings.kind
    issues: list[PreflightIssue] = []

    def add(severity: Severity, code: str, message: str) -> None:
        issues.append(PreflightIssue(severity, name, code, message))

    try:
        require_fully_specified(cable, kind)
    except CableSchemaError as error:
        add("error", "incomplete-cable", str(error))
        return tuple(issues)

    contact = getattr(settings, "contact_model", None)
    radial_solver = kind == "coiling" or contact in ("qp", "solved")
    if contact == "qp" and _offset_cores(cable):
        add(
            "error",
            "qp-offset-cores",
            "the QP contact solver is concentric only; use 'bounded' or 'solved' "
            "for offset (three-core) cables",
        )
    if radial_solver and _strand_bundle(cable):
        add(
            "error",
            "strand-bundle-radial",
            "the radial solvers need a solid conductor; a resolved strand bundle "
            "has no strand radial law (use the area-equivalent template)",
        )

    friction_kinds = (
        "bending",
        "fatigue",
        "stress",
        "capacity",
        "bending_law",
        "coiling",
    )
    if kind in friction_kinds:
        message = clamping_state(cable).warning_message()
        if message is not None:
            add("warning", "no-clamping", message)

    if kind == "fatigue":
        for layer in cable.helical_layers:
            if not layer.structural:
                continue
            curve = layer.bulk_sn
            if curve in BLOCKED_SN_CURVES:
                add(
                    "error",
                    "blocked-sn-curve",
                    f"layer {layer.name!r}: S-N curve {curve!r} is blocked "
                    f"({BLOCKED_SN_CURVES[curve]})",
                )
            elif curve and curve not in SN_CURVES:
                add(
                    "error",
                    "unknown-sn-curve",
                    f"layer {layer.name!r}: unknown S-N curve {curve!r}",
                )
            elif not curve:
                add(
                    "warning",
                    "no-sn-curve",
                    f"layer {layer.name!r} declares no bulk_sn",
                )

    if kind == "conductor_fatigue":
        conductors = [
            layer
            for layer in cable.layers
            if isinstance(layer, CylindricalOffsetLayer) and layer.strand_layers
        ]
        if not conductors:
            add(
                "error",
                "no-resolved-conductor",
                "conductor fatigue needs a cylindrical_offset layer with resolved "
                "strand_layers (for the 66 kV template: iea_task49_66kv_stranded)",
            )
        else:
            curve = getattr(settings, "sn_curve", None) or conductors[0].bulk_sn
            if not curve:
                add(
                    "error",
                    "no-sn-curve",
                    "the conductor declares no bulk_sn and none is given",
                )
            elif curve not in SN_CURVES:
                add(
                    "error",
                    "unknown-sn-curve",
                    f"unknown conductor S-N curve {curve!r}",
                )
            if (
                getattr(settings, "trellis_fretting", False)
                and getattr(settings, "trellis_model", "") == "surface_swt"
                and conductors[0].fretting_sn not in SWT_LIFE
            ):
                add(
                    "error",
                    "no-fretting-law",
                    "trellis_model='surface_swt' needs a registered fretting_sn "
                    "on the conductor",
                )
            if (
                getattr(settings, "trellis_fretting", False)
                and getattr(settings, "trellis_model", "") == "surface_swt"
            ):
                add(
                    "warning",
                    "unvalidated-model",
                    "the surface SWT trellis rule fails its gate (gap 1.3c); prefer "
                    "trellis_model='critical_distance'",
                )

    if kind == "bending_law":
        tensions = np.asarray(getattr(settings, "tensions_N", ()), dtype=float)
        if (
            tensions.size > 1
            and np.max(np.diff(tensions)) > BENDING_LAW_TENSION_SPACING_N
        ):
            add(
                "warning",
                "coarse-tension-grid",
                f"tension levels up to {np.max(np.diff(tensions)) / 1e3:.0f} kN apart; "
                "interpolation between them can miss the backbone by more than 1 % "
                "of the moment range (keep them within "
                f"{BENDING_LAW_TENSION_SPACING_N / 1e3:.0f} kN)",
            )
        if np.any(np.diff(tensions) <= 0):
            add("error", "tension-order", "tensions_N must rise strictly")
        add(
            "warning",
            "planar-law",
            "the exported law is planar; a global solver applying it to the "
            "resultant curvature assumes isotropic hysteresis, which the "
            "section model does not",
        )

    if kind in _HISTORY_KINDS:
        if series is None:
            add("error", "missing-load-case", "this analysis needs a load case")
        else:
            if series.n_steps < 2:
                add(
                    "error", "short-history", "the load case needs at least two samples"
                )
            if (
                kind in ("bending", "conductor_fatigue")
                and series.tension_kind != "true"
            ):
                add(
                    "error",
                    "tension-kind",
                    f"the load case tension is {series.tension_kind!r}; this analysis "
                    "needs true (wall) tension",
                )
            if kind == "bending" and (
                np.any(series.curvature_y_1_m != 0)
                or np.any(series.tension_N != series.tension_N[0])
            ):
                add(
                    "error",
                    "not-planar",
                    "planar bending needs zero y-curvature and constant tension",
                )
            if kind in ("bending", "conductor_fatigue") and (
                series.temperature_C is not None or series.axial_strain is not None
            ):
                add(
                    "error",
                    "unused-history",
                    "this analysis does not consume temperature or axial strain "
                    "histories; remove them from the load case",
                )
            if kind == "fatigue":
                issues.extend(
                    PreflightIssue("error", name, code, message)
                    for code, message in _fatigue_history_errors(series)
                )
    return tuple(issues)


def _fatigue_history_errors(series: LoadCaseSeries) -> list[tuple[str, str]]:
    """The load-case combinations the history fatigue refuses at run time.

    Mirrors :func:`cablexsec.io.series.fatigue_from_series`: without a direct
    axial strain the force column must be true tension and no temperature
    history may be given.
    """
    errors: list[tuple[str, str]] = []
    strain = series.axial_strain is not None
    if strain and series.axial_strain_kind != "direct":
        errors.append(
            (
                "strain-kind",
                f"the axial strain is {series.axial_strain_kind!r}; fatigue needs "
                "direct axial strain",
            )
        )
    if not strain and series.tension_kind != "true":
        errors.append(
            (
                "tension-kind",
                f"the load case tension is {series.tension_kind!r}; fatigue "
                "without a direct axial strain needs true (wall) tension",
            )
        )
    if not strain and series.temperature_C is not None:
        errors.append(
            (
                "temperature-force",
                "a temperature history with a force history needs solved "
                "section states; give a direct axial strain history or use "
                "compute_fatigue",
            )
        )
    return errors


def preflight(project: Project) -> PreflightReport:
    """Findings for every analysis of a project, without running any.

    A project only holds analyses whose model and load case exist
    (:meth:`~cablexsec.project.Project.set_analysis` and loading check it).
    """
    issues: list[PreflightIssue] = []
    for request in project.analyses.values():
        model = project.models[request.model]
        series = (
            None if request.load_case is None else project.load_cases[request.load_case]
        )
        issues.extend(preflight_request(model.definition, request, series))
    return PreflightReport(tuple(issues))


# Public entry points read the cable's own materials when called directly.
_scope_entry_points(globals(), __all__)
