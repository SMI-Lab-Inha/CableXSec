# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Long-term fatigue at fixed material fibres for stationary load cases."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from cablexsec.analysis.fatigue import FatigueResult
from cablexsec.exceptions import CableInputError
from cablexsec.provenance import Provenance, real_array

FloatArray = NDArray[np.float64]

__all__ = ["FatigueLoadCase", "LongTermFatigueResult", "compute_long_term_fatigue"]


@dataclass(frozen=True)
class FatigueLoadCase:
    """One stationary record and its fraction of total exposure time.

    ``duration_s`` is the retained record duration, excluding removed transients.
    For equally weighted independent seeds of one condition, divide that
    condition's probability by its number of seeds. Do not repeat the full
    condition probability for every seed. Unequal seed weights must be explicit.

    ``assessment_id`` is a caller-controlled identifier for the SAME construction,
    indexed physical wires, material frame, S-N assignments and model settings.
    Prefer a digest of those inputs. The aggregator checks identity, layer names,
    array shapes and fibre angles; it cannot infer geometry/curve identity from
    FatigueResult alone. Conditions may vary loading and temperature consistently
    with the declared assessment. Different spatial cable stations require
    separate assessments. Probabilities must cover the whole exposure population.
    """

    name: str
    result: FatigueResult
    duration_s: float
    probability: float
    assessment_id: str


@dataclass(frozen=True)
class LongTermFatigueResult:
    """Stationary Miner projection on an explicit longitudinal damage basis.

    Arrays retain (physical wire, material fibre) indexing. Design utilisation
    equals maximum annual selected damage times exposure years times the supplied
    design fatigue factor. ``allowable_exposure_years`` is its reciprocal rate;
    Infinity denotes zero computed damage or a reciprocal beyond floating-point
    range, not a physical infinite-life claim. ``damage_basis`` records whether
    the input stresses included the declared termination range factors. Neither
    basis includes the wire-level fretting/contact-life indicators.
    """

    annual_damage_per_fibre: dict[str, FloatArray]
    annual_damage_per_wire: dict[str, FloatArray]
    fibre_angles_deg: FloatArray
    critical_layer: str
    critical_wire_index: int
    critical_fibre_index: int
    critical_fibre_deg: float
    annual_damage: float
    exposure_years: float
    design_fatigue_factor: float
    seconds_per_year: float
    design_utilization: float
    allowable_exposure_years: float
    case_contributions_at_critical_per_year: dict[str, float]
    assessment_id: str
    provenance: Provenance
    damage_basis: Literal["bulk", "bulk_with_termination"] = "bulk"


def _positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise CableInputError(f"{name} must be finite and positive")


def _readonly(values: FloatArray) -> FloatArray:
    result = values.copy()
    result.flags.writeable = False
    return result


def compute_long_term_fatigue(
    cases: Sequence[FatigueLoadCase],
    *,
    exposure_years: float,
    design_fatigue_factor: float = 1.0,
    seconds_per_year: float = 365.25 * 24 * 3600,
    damage_basis: Literal["bulk", "bulk_with_termination"] = "bulk",
) -> LongTermFatigueResult:
    """Accumulate stationary fixed-fibre damage rates before choosing the maximum.

    At each material point, annual damage is
    ``seconds_per_year * sum(probability * record_damage / duration_s)``.
    Probabilities must sum to one within 1e-12; missing conditions are neither
    silently normalised nor assigned zero damage. No rainflow cycles are joined
    across records. These must represent independent stationary exposures, not
    chunks of one continuous history with uncounted boundary cycles.

    ``damage_basis='bulk'`` retains the unamplified bulk assessment. Explicitly
    select ``'bulk_with_termination'`` to use each record's fixed-fibre field
    counted at SCF-amplified stress ranges. Every record must contain that field
    with the same layer/wire/fibre layout as its bulk field. No factor is applied
    again, and neither wire maxima nor wire-level termination adjustments are
    accumulated. The caller's assessment identity must include the SCF model.
    Fretting remains excluded because its wire-level indicator does not establish
    a coincident material point or a validated combined damage law.
    The chosen design factor is caller policy, not a code-compliance assertion.
    """
    if damage_basis not in {"bulk", "bulk_with_termination"}:
        raise CableInputError("damage_basis must be 'bulk' or 'bulk_with_termination'")
    if not cases:
        raise CableInputError("at least one fatigue load case is required")
    for name, value in (
        ("exposure_years", exposure_years),
        ("design_fatigue_factor", design_fatigue_factor),
        ("seconds_per_year", seconds_per_year),
    ):
        _positive(value, name)
    if design_fatigue_factor < 1:
        raise CableInputError("design_fatigue_factor must be at least one")
    reference = cases[0]
    if not reference.assessment_id.strip():
        raise CableInputError(
            "assessment_id must identify the indexed model and S-N basis"
        )
    angles = real_array(reference.result.fibre_angles_deg, "fibre angles")
    if (
        angles.ndim != 1
        or angles.size == 0
        or not np.isfinite(angles).all()
        or np.any((angles < 0) | (angles >= 360))
        or np.unique(angles).size != angles.size
    ):
        raise CableInputError("fibre angles must be distinct finite values in [0, 360)")
    layers = sorted(reference.result.D_bulk_per_fibre)
    if not layers:
        raise CableInputError(
            "fixed-fibre bulk damage is required; wire maxima are insufficient"
        )
    names: set[str] = set()
    selected_fields = [
        case.result.D_bulk_per_fibre
        if damage_basis == "bulk"
        else case.result.D_bulk_with_termination_per_fibre
        for case in cases
    ]
    for case, selected in zip(cases, selected_fields, strict=True):
        if not case.name.strip() or case.name in names:
            raise CableInputError("load case names must be nonempty and unique")
        names.add(case.name)
        _positive(case.duration_s, "duration_s")
        if not math.isfinite(case.probability) or not 0 <= case.probability <= 1:
            raise CableInputError("probability must be finite and between zero and one")
        if case.assessment_id != reference.assessment_id:
            raise CableInputError("all cases must share the same assessment_id")
        if not np.array_equal(case.result.fibre_angles_deg, angles):
            raise CableInputError(
                "all cases must share the same ordered material fibre angles"
            )
        if sorted(case.result.D_bulk_per_fibre) != layers:
            raise CableInputError("all cases must contain the same layers")
        if sorted(selected) != layers:
            raise CableInputError(
                f"{case.name}: {damage_basis} requires fixed-fibre damage "
                "for every analysed layer; wire maxima are insufficient"
            )
        for layer in layers:
            values = real_array(selected[layer], f"{case.name}: {layer} damage")
            if (
                values.ndim != 2
                or values.shape[0] == 0
                or values.shape[1] != angles.size
                or values.shape != reference.result.D_bulk_per_fibre[layer].shape
                or values.shape != case.result.D_bulk_per_fibre[layer].shape
                or not np.isfinite(values).all()
                or np.any(values < 0)
            ):
                raise CableInputError(
                    f"{case.name}/{layer}: incompatible or invalid fibre damage"
                )
    if not math.isclose(
        math.fsum(c.probability for c in cases), 1, rel_tol=0, abs_tol=1e-12
    ):
        raise CableInputError(
            "probabilities must sum to one; specify the complete exposure population"
        )

    annual = {}
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            for layer in layers:
                terms = [
                    np.asarray(selected[layer], dtype=np.float64)
                    * (c.probability * seconds_per_year / c.duration_s)
                    for c, selected in zip(cases, selected_fields, strict=True)
                ]
                annual[layer] = _readonly(np.sum(terms, axis=0))
        except FloatingPointError as error:
            raise CableInputError(
                "damage accumulation exceeds floating-point range"
            ) from error
    critical_layer = max(layers, key=lambda name: float(np.max(annual[name])))
    wire, fibre = np.unravel_index(
        np.argmax(annual[critical_layer]), annual[critical_layer].shape
    )
    damage = float(annual[critical_layer][wire, fibre])
    design_rate = damage * design_fatigue_factor
    utilization = design_rate * exposure_years
    if not math.isfinite(utilization):
        raise CableInputError("design utilisation exceeds floating-point range")
    contributions = {
        c.name: float(selected[critical_layer][wire, fibre])
        * (c.probability * seconds_per_year / c.duration_s)
        for c, selected in zip(cases, selected_fields, strict=True)
    }
    limitations = [
        (
            "Stationary bulk-only Miner projection; excludes fretting, termination "
            "increments, ageing and sequence effects."
            if damage_basis == "bulk"
            else "Stationary Miner projection of bulk damage counted at termination-"
            "adjusted stress ranges at fixed fibres; excludes fretting, ageing "
            "and sequence effects. SCFs require independent qualification."
        ),
        "Assessment identity and physical wire ordering are declared by the caller, "
        "not inferred from result arrays.",
        "Independent records retain their own residual cycles; continuous-history "
        "chunks cannot be accumulated this way.",
        "Probabilities, representative duration, material curves and design factor "
        "require an engineering assessment.",
    ]
    citations = ["[Miner1945]"]
    for case in cases:
        if case.result.provenance is not None:
            limitations.extend(case.result.provenance.limitations)
            citations.extend(case.result.provenance.citations)
    return LongTermFatigueResult(
        annual_damage_per_fibre=annual,
        annual_damage_per_wire={
            name: _readonly(np.max(value, axis=1)) for name, value in annual.items()
        },
        fibre_angles_deg=_readonly(angles),
        critical_layer=critical_layer,
        critical_wire_index=int(wire),
        critical_fibre_index=int(fibre),
        critical_fibre_deg=float(angles[fibre]),
        annual_damage=damage,
        exposure_years=exposure_years,
        design_fatigue_factor=design_fatigue_factor,
        seconds_per_year=seconds_per_year,
        design_utilization=utilization,
        allowable_exposure_years=math.inf if design_rate == 0 else 1 / design_rate,
        case_contributions_at_critical_per_year=contributions,
        assessment_id=reference.assessment_id,
        damage_basis=damage_basis,
        provenance=Provenance(
            method=f"Probability-weighted stationary {damage_basis} Miner damage "
            "at fixed material fibres",
            citations=tuple(dict.fromkeys(citations)),
            gates=("LT1: material-point accumulation and exposure scaling",),
            limitations=tuple(dict.fromkeys(limitations)),
        ),
    )
