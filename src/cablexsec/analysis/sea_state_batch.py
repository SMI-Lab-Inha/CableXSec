# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Sea-state batch: CableDyn runs to weighted annual armour damage.

One CableDyn run per sea state (or seed), each with its probability of
occurrence. For every run the requested sections are read with
:func:`cablexsec.io.cabledyn.read_cabledyn_sections`, the transient is removed,
and :func:`cablexsec.compute_fatigue` is evaluated per section, with the
optional clamp correction (``termination``), energised state (``thermal``) and
fretting assessment (:func:`cablexsec.compute_fretting`). The fixed-fibre
damage of the runs is accumulated with
:func:`cablexsec.analysis.long_term.compute_long_term_fatigue`:

.. math::

   D_\mathrm{year} = s_\mathrm{year}\sum_i p_i\,\frac{D_i}{T_i},

with :math:`p_i` the probability, :math:`D_i` the record damage and
:math:`T_i` the retained duration of run :math:`i`, at every material fibre
before the maximum is taken; the life is
:math:`1/(\mathrm{DFF}\,D_\mathrm{year})`. Probabilities must sum to one.

Axial force
-----------
``compute_fatigue`` takes the material force; the armour contact pressure
follows its history (no frozen contact state). ``tension_basis`` declares
which force that is:

* ``"true"``: :math:`T_e - p_oA_o` with the hydrostatic pressure of the run
  and the cable's outer area (sealed, dry annulus);
* ``"effective"``: :math:`T_e` (flooded annulus: seawater reaches the
  armour, so the hydrostatic pressure acts on every face of the layers);
* ``"auto"`` (default): from ``cable.water_ingress.armour_annulus``; a cable
  that does not declare it is rejected.

A negative material force is rejected by ``compute_fatigue`` with the time
and section named.

Parallel execution
------------------
Runs are processed in worker processes (``workers``, at most
:data:`MAX_WORKERS`), each limited to one numerical thread. Results do not
depend on the worker count. The output directory, if given, must not exist:
results are written once, with a manifest of SHA-256 digests of every input
file, the cable definition and the options.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import platform
import sys
import warnings
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from cablexsec.analysis.capacity_check import (
    CapacityCheckResult,
    CapacityEnvelope,
    check_capacity,
    gumbel_fit,
)
from cablexsec.analysis.fatigue import (
    DEFAULT_FRICTION_MODEL,
    FatigueResult,
    FrictionModel,
    WirePositions,
)
from cablexsec.analysis.long_term import (
    FatigueLoadCase,
    LongTermFatigueResult,
    compute_long_term_fatigue,
)
from cablexsec.analysis.termination import TerminationSpec
from cablexsec.analysis.thermal_coupling import ThermalHistory, ThermalState
from cablexsec.cables.schema import CableData
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.io.cabledyn import (
    CableDynLineType,
    CableDynRead,
    CableDynSectionHistory,
    CurvatureSource,
    Frame,
    TangentPlacement,
    _hang_off_input,
    read_cabledyn_deck,
    read_cabledyn_sections,
)

FloatArray = NDArray[np.float64]
TensionBasis = Literal["auto", "true", "effective"]

#: Hard cap on worker processes of one batch.
MAX_WORKERS = 8
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

__all__ = [
    "MAX_WORKERS",
    "CapacityBatchCheck",
    "SeaStateBatchOptions",
    "SeaStateBatchResult",
    "SeaStateRun",
    "SectionBatchResult",
    "SectionRequest",
    "load_batch_spec",
    "run_sea_state_batch",
]


@dataclass(frozen=True)
class SeaStateRun:
    """One CableDyn run and its probability of occurrence.

    ``start_time_s``/``end_time_s`` bound the retained record (transient
    removal); the frame is still transported from ``t = 0``.
    ``hang_off_rotation`` is the run's ``hang_off_rotation`` of
    :func:`~cablexsec.io.cabledyn.read_cabledyn_sections` (the turn of the
    holder of a line restrained in torsion at End A): ``None``, ``"none"``,
    or an array with one entry per output row.
    """

    name: str
    out_root: Path
    deck: Path
    probability: float
    line_id: int = 1
    start_time_s: float = 0.0
    end_time_s: float | None = None
    hang_off_rotation: Any = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not self.name.strip() or "/" in self.name or "\\" in self.name:
            raise CableInputError("run names must be nonempty and contain no slash")
        object.__setattr__(self, "out_root", Path(self.out_root))
        object.__setattr__(self, "deck", Path(self.deck))
        if not math.isfinite(self.probability) or not 0.0 <= self.probability <= 1.0:
            raise CableInputError(f"{self.name}: probability must be in [0, 1]")
        if not math.isfinite(self.start_time_s) or self.start_time_s < 0:
            raise CableInputError(f"{self.name}: start_time_s must be >= 0")
        if self.end_time_s is not None and not self.end_time_s > self.start_time_s:
            raise CableInputError(f"{self.name}: end_time_s must exceed start_time_s")
        rotation = _hang_off_input(self.hang_off_rotation)
        object.__setattr__(
            self,
            "hang_off_rotation",
            rotation.values if rotation.values is not None else self.hang_off_rotation,
        )


def _hang_off_record(value: Any) -> Any:
    """The manifest entry of a run's ``hang_off_rotation``."""
    if value is None or isinstance(value, str):
        return value
    array = np.ascontiguousarray(value, dtype=np.float64)
    return {"shape": list(array.shape), "sha256": _sha256_bytes(array.tobytes())}


@dataclass(frozen=True)
class SectionRequest:
    """A material section and its local options.

    ``termination`` applies the clamp correction (sections near an end
    fitting or a clamped stiffener root); ``thermal`` the energised state
    (one-way, solved contact model); ``fretting`` adds
    :func:`cablexsec.compute_fretting` with the run probability as time
    fraction. ``sn_curves``, ``mean_stress`` and ``scf`` are the user S-N
    options of :func:`cablexsec.compute_fatigue` (registered curve names per
    layer, a mean-stress correction and component stress concentration
    factors); ``None`` leaves the section's result and record unchanged.
    """

    name: str
    arc_length_m: float
    termination: TerminationSpec | None = None
    thermal: ThermalState | ThermalHistory | None = None
    fretting: bool = False
    sn_curves: Mapping[str, str] | None = None
    mean_stress: Any = None
    scf: Any = None

    def __post_init__(self) -> None:
        if not self.name.strip() or "/" in self.name or "\\" in self.name:
            raise CableInputError("section names must be nonempty and contain no slash")
        if not math.isfinite(self.arc_length_m) or self.arc_length_m < 0:
            raise CableInputError(f"{self.name}: arc_length_m must be finite and >= 0")


@dataclass(frozen=True)
class SeaStateBatchOptions:
    """Reader and fatigue options shared by every run and section.

    ``friction_model`` ``None`` (default) takes the default armour friction
    model of :func:`cablexsec.compute_fatigue`, with its notice on non-planar
    paths (:class:`~cablexsec.provenance.DirectionTiedFrictionWarning`); the
    manifest and the assessment digests record the model it resolves to.

    The property check compares the deck's ``EA`` with ``EA_fixed`` and its
    ``EI`` with the full-slip to no-slip range within ``property_rtol``, and,
    on a 14-column ``LINE TYPES`` row, its ``GJ`` with the lay torsional
    stiffness ``K_tt`` and its torque-tension coupling ``K`` with ``K_et``
    within ``torsion_rtol``: ``GJ`` relative to ``K_tt``, ``K`` as the
    coupling ratio ``c = K/sqrt(EA GJ)`` (``|c| < 1``; the free-twist axial
    stiffness is ``EA (1 - c^2)``), against ``K_et/sqrt(EA_fixed K_tt)``.
    ``K_tt`` leaves out the components' own torsion and the radial relaxation
    of the layers, which move a section's torsional stiffness by tens of
    percent either way, hence the wider default.
    """

    n_fibres: int = 32
    mu: float | None = None
    contact_model: str | None = None
    friction_model: FrictionModel | None = None
    #: Wires to evaluate (``compute_fatigue(wire_positions=...)``); ``None``
    #: evaluates every wire.
    wire_positions: WirePositions | None = None
    frame: Frame = "arc"
    curvature_source: CurvatureSource = "auto"
    #: ``"chord"`` reads finite-EI output of CableDyn 0.4.1 and earlier
    #: (:func:`~cablexsec.io.cabledyn.read_cabledyn_sections`).
    tangent_placement: TangentPlacement = "auto"
    reference_vector: tuple[float, float, float] = (0.0, 0.0, 1.0)
    #: Tolerances of the reader's check against CableDyn's ``Curv`` channels
    #: (:func:`~cablexsec.io.cabledyn.read_cabledyn_sections`).
    curvature_check_rtol: float = 0.05
    curvature_check_end_rtol: float = 0.25
    curvature_check_atol_1_m: float | None = None
    tension_basis: TensionBasis = "auto"
    external_diameter_m: float | None = None
    exposure_years: float = 1.0
    design_fatigue_factor: float = 1.0
    property_rtol: float = 0.05
    torsion_rtol: float = 0.25
    allow_property_mismatch: bool = False
    workers: int = 1
    chunk_rows: int = 4096

    def __post_init__(self) -> None:
        if (
            isinstance(self.workers, bool)
            or not isinstance(self.workers, int)
            or not 1 <= self.workers <= MAX_WORKERS
        ):
            raise CableInputError(f"workers must be an integer in [1, {MAX_WORKERS}]")
        for name in (
            "curvature_check_rtol",
            "curvature_check_end_rtol",
            "curvature_check_atol_1_m",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise CableInputError(f"{name} must be a finite number >= 0")
        if self.tension_basis not in ("auto", "true", "effective"):
            raise CableInputError("tension_basis must be auto, true or effective")
        if self.external_diameter_m is not None and not (
            math.isfinite(self.external_diameter_m) and self.external_diameter_m > 0
        ):
            raise CableInputError("external_diameter_m must be positive")
        if not math.isfinite(self.property_rtol) or self.property_rtol <= 0:
            raise CableInputError("property_rtol must be positive")
        if not math.isfinite(self.torsion_rtol) or self.torsion_rtol <= 0:
            raise CableInputError("torsion_rtol must be positive")
        object.__setattr__(
            self, "reference_vector", tuple(float(v) for v in self.reference_vector)
        )


@dataclass(frozen=True)
class CapacityBatchCheck:
    """Capacity check of every section and run (gap: capacity of time series).

    ``envelope`` is a
    :class:`~cablexsec.analysis.capacity_check.CapacityEnvelope` whose
    tension grid spans the runs; its ``tension_kind`` selects the
    effective tension of the run or the material force of the tension basis.
    ``groups`` names the runs that are seeds of one condition (a Gumbel fit
    of their maxima is reported per group with two runs or more); ``None``
    treats all runs as one group. ``blocks`` adds a within-record
    block-maxima
    fit per run.
    """

    envelope: CapacityEnvelope
    thresholds: tuple[float, ...] = (1.0,)
    fractile: float = 0.9
    groups: Mapping[str, Sequence[str]] | None = None
    blocks: int | None = None
    out_of_range: Literal["raise", "clamp"] = "raise"

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, CapacityEnvelope):
            raise CableInputError("envelope must be a CapacityEnvelope")
        object.__setattr__(self, "thresholds", tuple(float(v) for v in self.thresholds))
        if not 0.0 < self.fractile < 1.0:
            raise CableInputError("fractile must lie in (0, 1)")


@dataclass(frozen=True)
class SectionBatchResult:
    """Weighted result of one section over all runs.

    ``capacity`` (with a :class:`CapacityBatchCheck`) summarises the
    utilisation of every run, the overall maximum and the Gumbel extremes
    per group; ``capacity_runs`` holds the utilisation histories.
    """

    name: str
    arc_length_m: float
    long_term: LongTermFatigueResult
    life_years: float
    runs: dict[str, dict[str, Any]]
    fretting: dict[str, dict[str, float]] | None = None
    capacity: dict[str, Any] | None = None
    capacity_runs: dict[str, CapacityCheckResult] = field(
        default_factory=dict, repr=False
    )


@dataclass(frozen=True)
class SeaStateBatchResult:
    """All sections, and the manifest written with them."""

    sections: dict[str, SectionBatchResult]
    manifest: dict[str, Any]
    output_directory: Path | None = None
    histories: dict[str, dict[str, CableDynSectionHistory]] = field(
        default_factory=dict, repr=False
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def cable_digest(cable: CableData) -> str:
    """SHA-256 of the validated cable definition (canonical JSON)."""
    return _sha256_bytes(_canonical(cable.model_dump(mode="json")).encode())


def _termination_record(spec: TerminationSpec | None) -> Any:
    if spec is None:
        return None
    record = asdict(spec)
    if spec.curvature_profile is not None:
        record["curvature_profile"] = [a.tolist() for a in spec.curvature_profile]
    return record


def _thermal_record(state: ThermalState | ThermalHistory | None) -> Any:
    if state is None:
        return None
    if isinstance(state, ThermalState):
        return {
            "kind": "ThermalState",
            "layer_temperature_C": dict(state.layer_temperature_C),
            "T_ref_C": state.T_ref_C,
            "temperature_dependent_modulus": state.temperature_dependent_modulus,
            "thermal_expansion": state.thermal_expansion,
            "modulus_transfer": dict(state.modulus_transfer),
            "label": state.label,
        }
    return {
        "kind": "ThermalHistory",
        "time_s": state.time_s.tolist(),
        "layer_temperature_C": {
            k: v.tolist() for k, v in state.layer_temperature_C.items()
        },
        "T_ref_C": state.T_ref_C,
        "source": state.source,
    }


def _option_record(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return {key: _option_record(item) for key, item in value.items()}
    return {"kind": type(value).__name__, **asdict(value)}


def _section_record(section: SectionRequest) -> dict[str, Any]:
    record = {
        "name": section.name,
        "arc_length_m": section.arc_length_m,
        "termination": _termination_record(section.termination),
        "thermal": _thermal_record(section.thermal),
        "fretting": section.fretting,
    }
    # User S-N options enter the record only when set, so that the digests
    # of existing assessments are unchanged.
    for key in ("sn_curves", "mean_stress", "scf"):
        value = getattr(section, key)
        if value is not None:
            record[key] = _option_record(value)
    return record


def _outer_diameter(cable: CableData) -> float:
    from cablexsec.analysis.compression import _outer_radius  # noqa: PLC0415

    return 2.0 * _outer_radius(cable)


def _resolve_basis(
    cable: CableData, basis: TensionBasis
) -> Literal["true", "effective"]:
    if basis != "auto":
        return basis
    water = cable.water_ingress
    if water is None:
        raise CableInputError(
            "the cable does not declare water_ingress.armour_annulus; choose "
            "tension_basis='true' (sealed) or 'effective' (flooded) explicitly"
        )
    return "effective" if water.armour_annulus == "flooded" else "true"


def _limit_threads() -> None:
    for name in _THREAD_VARIABLES:
        os.environ[name] = "1"
    from cablexsec._numba import HAVE_NUMBA  # noqa: PLC0415

    if HAVE_NUMBA:  # numba may already be imported in a reused worker
        import importlib  # noqa: PLC0415

        importlib.import_module("numba").set_num_threads(1)


def _portable(section: SectionRequest) -> SectionRequest | tuple[Any, ...]:
    """A picklable form: thermal states hold read-only mapping proxies."""
    state = section.thermal
    if state is None:
        return section
    values = {}
    for item in fields(state):
        value = getattr(state, item.name)
        values[item.name] = (
            dict(value) if isinstance(value, MappingProxyType) else value
        )
    return (
        section.name,
        section.arc_length_m,
        section.termination,
        section.fretting,
        type(state),
        values,
        None if section.sn_curves is None else dict(section.sn_curves),
        section.mean_stress,
        section.scf,
    )


def _restore(item: SectionRequest | tuple[Any, ...]) -> SectionRequest:
    if isinstance(item, SectionRequest):
        return item
    name, arc, termination, fretting, kind, values, curves, mean, scf = item
    return SectionRequest(
        name, arc, termination, kind(**values), fretting, curves, mean, scf
    )


@dataclass(frozen=True)
class _Task:
    cable: CableData
    run: SeaStateRun
    sections: tuple[SectionRequest | tuple[Any, ...], ...]
    options: SeaStateBatchOptions
    basis: Literal["true", "effective"]
    external_area_m2: float
    capacity: CapacityBatchCheck | None = None


def _fretting_summary(result: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, face in result.faces.items():
        out[key] = {
            "wear_depth_per_year_m": np.asarray(face.wear_depth_per_year_m).tolist(),
            "D_fretting_per_year": np.asarray(face.D_fretting_per_year).tolist(),
            "D_plain_per_year": np.asarray(face.D_plain_per_year).tolist(),
        }
    return out


def _evaluate_in_worker(task: _Task) -> dict[str, Any]:
    """:func:`_evaluate` in a worker process, with its warnings recorded.

    A spawned worker shows warnings on its own standard error, out of the
    caller's reach; they travel back with the outcome (``"warnings"``) and the
    caller issues them again.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        outcome = _evaluate(task)
    outcome["warnings"] = [
        _portable_warning(w.category, str(w.message)) for w in caught
    ]
    return outcome


def _portable_warning(
    category: type[Warning], message: str
) -> tuple[type[Warning], str]:
    """A warning that survives the return from a worker process."""
    try:
        pickle.loads(pickle.dumps(category))
    except Exception:  # noqa: BLE001 (a category that does not pickle)
        return UserWarning, f"{category.__name__}: {message}"
    return category, message


def _evaluate(task: _Task) -> dict[str, Any]:
    """Read one run and evaluate every section (worker entry point).

    Worker processes are limited to one thread by the pool initializer; the
    in-process path leaves the caller's thread settings alone.
    """
    from cablexsec.api import compute_fatigue, compute_fretting  # noqa: PLC0415

    run, opts = task.run, task.options
    sections = [_restore(item) for item in task.sections]
    read: CableDynRead = read_cabledyn_sections(
        run.out_root,
        run.deck,
        {section.name: section.arc_length_m for section in sections},
        line_id=run.line_id,
        frame=opts.frame,
        curvature_source=opts.curvature_source,
        tangent_placement=opts.tangent_placement,
        reference_vector=opts.reference_vector,
        pressure="hydrostatic" if task.basis == "true" else "none",
        chunk_rows=opts.chunk_rows,
        hang_off_rotation=run.hang_off_rotation,
        curvature_check_rtol=opts.curvature_check_rtol,
        curvature_check_end_rtol=opts.curvature_check_end_rtol,
        curvature_check_atol_1_m=opts.curvature_check_atol_1_m,
    )
    results: dict[str, Any] = {}
    histories: dict[str, CableDynSectionHistory] = {}
    for section in sections:
        history = read.histories[section.name].window(run.start_time_s, run.end_time_s)
        histories[section.name] = history
        kx, ky = history.strain_gradient()
        force = history.true_tension_N(task.external_area_m2)
        if np.any(force < 0):
            k = int(np.argmin(force))
            raise CableSchemaError(
                f"run {run.name!r}, section {section.name!r}: material force "
                f"{force[k]:.4g} N < 0 at t = {history.time_s[k]:g} s "
                f"(tension basis {task.basis}); compute_fatigue needs a tensile "
                "force history"
            )
        fatigue: FatigueResult = compute_fatigue(
            task.cable,
            kx,
            ky,
            tension_N=force,
            mu=opts.mu,
            contact_model=opts.contact_model,
            n_fibres=opts.n_fibres,
            friction_model=opts.friction_model,
            wire_positions=opts.wire_positions,
            termination=section.termination,
            thermal=section.thermal,
            time_s=history.time_s
            if isinstance(section.thermal, ThermalHistory)
            else None,
            sn_curves=section.sn_curves,
            mean_stress=section.mean_stress,
            scf=section.scf,
        )
        fretting = None
        if section.fretting:
            fretting = _fretting_summary(
                compute_fretting(
                    task.cable,
                    kx,
                    ky,
                    duration_s=history.duration_s,
                    tension_N=force,
                    mu=opts.mu,
                    contact_model=opts.contact_model or "bounded",
                    time_fraction=run.probability,
                )
            )
        capacity = None
        if task.capacity is not None:
            check = task.capacity
            capacity = check_capacity(
                check.envelope,
                history.time_s,
                history.effective_tension_N
                if check.envelope.tension_kind == "effective"
                else force,
                kx,
                ky,
                tension_kind=check.envelope.tension_kind,
                name=section.name,
                thresholds=check.thresholds,
                torque_Nm=history.torque_Nm,
                blocks=check.blocks,
                fractile=check.fractile,
                out_of_range=check.out_of_range,
            )
        results[section.name] = {
            "capacity": capacity,
            "fatigue": fatigue,
            "fretting": fretting,
            "duration_s": history.duration_s,
            "statistics": {
                "samples": int(history.time_s.size),
                "time_s": [float(history.time_s[0]), float(history.time_s[-1])],
                "effective_tension_N": [
                    float(history.effective_tension_N.min()),
                    float(history.effective_tension_N.max()),
                ],
                "material_force_N": [float(force.min()), float(force.max())],
                "rotation_curvature_x_1_m": [
                    float(history.rotation_curvature_x_1_m.min()),
                    float(history.rotation_curvature_x_1_m.max()),
                ],
                "rotation_curvature_y_1_m": [
                    float(history.rotation_curvature_y_1_m.min()),
                    float(history.rotation_curvature_y_1_m.max()),
                ],
                "curvature_norm_max_1_m": float(history.curvature_norm_1_m.max()),
            },
        }
    return {
        "run": run.name,
        "results": results,
        "histories": histories,
        "checks": dict(read.checks),
        "files": [
            {
                "role": f.role,
                "path": str(f.path.resolve()),
                "sha256": f.sha256,
                "size_bytes": f.size_bytes,
            }
            for f in read.files
        ],
    }


def _property_check(
    cable: CableData,
    runs: Sequence[SeaStateRun],
    sections: Sequence[SectionRequest],
    options: SeaStateBatchOptions,
) -> list[dict[str, Any]]:
    from cablexsec.api import compute_stiffness  # noqa: PLC0415

    stiffness = compute_stiffness(cable)
    records: list[dict[str, Any]] = []
    for run in runs:
        deck = read_cabledyn_deck(run.deck)
        line = deck.line(run.line_id)
        for section in sections:
            line_type = deck.line_type(line.line_type_at(section.arc_length_m))
            ea_ok = line_type.ea_N is not None and math.isclose(
                line_type.ea_N, stiffness.ea_fixed_N, rel_tol=options.property_rtol
            )
            ei_ok = (
                stiffness.ei_min_Nm2 * (1 - options.property_rtol)
                <= line_type.ei_Nm2
                <= stiffness.ei_max_Nm2 * (1 + options.property_rtol)
            )
            records.append(
                {
                    "run": run.name,
                    "section": section.name,
                    "line_type": line_type.name,
                    "global_EA_N": line_type.ea_N,
                    "local_EA_fixed_N": stiffness.ea_fixed_N,
                    "global_EI_Nm2": line_type.ei_Nm2,
                    "local_EI_range_Nm2": [stiffness.ei_min_Nm2, stiffness.ei_max_Nm2],
                    "EA_match": ea_ok,
                    "EI_within_bounds": ei_ok,
                    **_torsion_check(line_type, stiffness, options.torsion_rtol),
                }
            )
    bad = [r for r in records if _mismatches(r)]
    if bad and not options.allow_property_mismatch:
        first = bad[0]
        raise CableInputError(
            "the global line properties do not describe the local section "
            f"({first['run']}/{first['section']}: {'; '.join(_mismatches(first))}); "
            "set allow_property_mismatch only for a documented comparison"
        )
    return records


def _torsion_check(
    line_type: CableDynLineType, stiffness: Any, rtol: float
) -> dict[str, Any]:
    """``GJ`` and coupling ``K`` of a 14-column row against ``K_tt`` and ``K_et``.

    ``None`` matches where the deck gives no ``GJ`` (a 10-column row, or a
    14-column row without the finite-EI columns) or CableXSec computes no
    lay torsional stiffness.
    """
    gj, k = line_type.gj_Nm2, line_type.torque_tension_coupling_Nm
    local_gj = float(stiffness.gj_Nm2)
    local_k = float(stiffness.axial_torsion_coupling_Nm)
    record: dict[str, Any] = {
        "global_GJ_Nm2": gj,
        "local_GJ_Nm2": local_gj,
        "global_K_Nm": k if gj is not None else None,
        "local_K_Nm": local_k,
        "global_coupling_ratio": None,
        "local_coupling_ratio": None,
        "GJ_match": None,
        "K_match": None,
    }
    if gj is None or not gj > 0.0 or not local_gj > 0.0:
        return record
    record["GJ_match"] = math.isclose(gj, local_gj, rel_tol=rtol)
    ea = line_type.ea_N
    local_ratio = local_k / math.sqrt(float(stiffness.ea_fixed_N) * local_gj)
    # CableDyn takes K only with a linear-elastic EA, so K = 0 elsewhere.
    ratio = 0.0 if k == 0.0 or ea is None or not ea > 0.0 else k / math.sqrt(ea * gj)
    record["global_coupling_ratio"] = ratio
    record["local_coupling_ratio"] = local_ratio
    record["K_match"] = abs(ratio - local_ratio) <= rtol
    return record


def _mismatches(record: Mapping[str, Any]) -> list[str]:
    """The quantities of a property-check record that do not match."""
    out = []
    if not record["EA_match"]:
        out.append(
            f"EA {record['global_EA_N']} N against {record['local_EA_fixed_N']:.4g} N"
        )
    if not record["EI_within_bounds"]:
        low, high = record["local_EI_range_Nm2"]
        out.append(f"EI {record['global_EI_Nm2']} N·m² against [{low:.4g}, {high:.4g}]")
    if record["GJ_match"] is False:
        out.append(
            f"GJ {record['global_GJ_Nm2']} N·m² against {record['local_GJ_Nm2']:.4g}"
        )
    if record["K_match"] is False:
        out.append(
            f"K {record['global_K_Nm']} N·m (coupling ratio "
            f"{record['global_coupling_ratio']:.3g}) against "
            f"{record['local_K_Nm']:.4g} N·m ({record['local_coupling_ratio']:.3g})"
        )
    return out


_SPEC_KEYS = {"cable", "runs", "sections", "options"}
_RUN_KEYS = {
    "name",
    "out_root",
    "deck",
    "probability",
    "line_id",
    "start_time_s",
    "end_time_s",
    "hang_off_rotation",
}
_SECTION_KEYS = {
    "name",
    "arc_length_m",
    "termination",
    "thermal",
    "fretting",
    "sn_curves",
    "mean_stress",
    "scf",
}


def _mean_stress_entry(value: Any) -> Any:
    from cablexsec.materials.mean_stress import MeanStressSpec  # noqa: PLC0415

    if value is None or isinstance(value, str):
        return value
    if not isinstance(value, Mapping):
        raise CableInputError("mean_stress must be a method name or an object")
    if "method" in value:
        return MeanStressSpec(**value)
    return {key: _mean_stress_entry(item) for key, item in value.items()}


def _scf_entry(value: Any) -> Any:
    from cablexsec.analysis.sn_assessment import ComponentSCF  # noqa: PLC0415

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise CableInputError("scf must be an object")
    if "source" in value:
        return ComponentSCF(**value)
    return {key: _scf_entry(item) for key, item in value.items()}


def _strict_keys(record: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(record) - allowed
    if unknown:
        raise CableInputError(f"{where}: unknown keys {sorted(unknown)}")


def load_batch_spec(
    path: str | Path,
) -> tuple[
    CableData, list[SeaStateRun], list[SectionRequest], SeaStateBatchOptions, Path
]:
    """Read a JSON batch specification.

    ``{"cable": <yaml path or "bundled:<name>">, "runs": [...], "sections":
    [...], "options": {...}}``. Run keys: ``name``, ``out_root``, ``deck``,
    ``probability``, optional ``line_id``, ``start_time_s``, ``end_time_s``,
    ``hang_off_rotation`` (``"none"`` or nested lists, one entry per output
    row).
    Section keys: ``name``, ``arc_length_m``, optional ``termination`` (the
    :class:`TerminationSpec` fields), ``thermal`` (the :class:`ThermalState`
    fields), ``fretting``, ``sn_curves`` (layer to registered curve name),
    ``mean_stress`` (a method name, the ``MeanStressSpec`` fields, or
    either per layer) and ``scf`` (the ``ComponentSCF`` fields with
    ``source``, or those per layer). Options are the :class:`SeaStateBatchOptions`
    fields. Relative paths are resolved against the specification's folder;
    unknown keys are rejected.
    """
    import cablexsec as cx  # noqa: PLC0415

    spec_path = Path(path)
    data = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CableInputError(f"{spec_path.name}: the specification must be an object")
    _strict_keys(data, _SPEC_KEYS, spec_path.name)
    base = spec_path.resolve().parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else base / candidate

    cable_entry = data.get("cable")
    if not isinstance(cable_entry, str) or not cable_entry.strip():
        raise CableInputError(f"{spec_path.name}: 'cable' must name a YAML file")
    if cable_entry.startswith("bundled:"):
        cable_path = Path(cx.__file__).parent / "cables" / f"{cable_entry[8:]}.yaml"
    else:
        cable_path = resolve(cable_entry)
    cable = cx.load_cable(cable_path, geometry_override=None)
    runs = []
    for i, record in enumerate(data.get("runs") or []):
        _strict_keys(record, _RUN_KEYS, f"runs[{i}]")
        runs.append(
            SeaStateRun(
                name=record["name"],
                out_root=resolve(record["out_root"]),
                deck=resolve(record["deck"]),
                probability=float(record["probability"]),
                line_id=int(record.get("line_id", 1)),
                start_time_s=float(record.get("start_time_s", 0.0)),
                end_time_s=None
                if record.get("end_time_s") is None
                else float(record["end_time_s"]),
                hang_off_rotation=record.get("hang_off_rotation"),
            )
        )
    sections = []
    for i, record in enumerate(data.get("sections") or []):
        _strict_keys(record, _SECTION_KEYS, f"sections[{i}]")
        termination = record.get("termination")
        if termination is not None:
            termination = dict(termination)
            if termination.get("curvature_profile") is not None:
                x, shape = termination["curvature_profile"]
                termination["curvature_profile"] = (
                    np.asarray(x, dtype=np.float64),
                    np.asarray(shape, dtype=np.float64),
                )
            termination = TerminationSpec(**termination)
        thermal = record.get("thermal")
        sections.append(
            SectionRequest(
                name=record["name"],
                arc_length_m=float(record["arc_length_m"]),
                termination=termination,
                thermal=None if thermal is None else ThermalState(**thermal),
                fretting=bool(record.get("fretting", False)),
                sn_curves=record.get("sn_curves"),
                mean_stress=_mean_stress_entry(record.get("mean_stress")),
                scf=_scf_entry(record.get("scf")),
            )
        )
    option_values = dict(data.get("options") or {})
    if "reference_vector" in option_values:
        option_values["reference_vector"] = tuple(option_values["reference_vector"])
    try:
        options = SeaStateBatchOptions(**option_values)
    except TypeError as error:
        raise CableInputError(f"{spec_path.name}: options: {error}") from error
    return cable, runs, sections, options, cable_path


def run_sea_state_batch(
    cable: CableData,
    runs: Sequence[SeaStateRun],
    sections: Sequence[SectionRequest],
    *,
    options: SeaStateBatchOptions | None = None,
    output_directory: str | Path | None = None,
    cable_source: str | Path | None = None,
    capacity: CapacityBatchCheck | None = None,
) -> SeaStateBatchResult:
    """Weighted annual armour damage of named sections over CableDyn runs.

    Parameters
    ----------
    cable : CableData
        The local cross-section.
    runs : sequence of SeaStateRun
        CableDyn runs; probabilities must sum to one.
    sections : sequence of SectionRequest
        Material sections by arc length from End A.
    options : SeaStateBatchOptions, optional
        Reader, fatigue and aggregation options.
    output_directory : str or pathlib.Path, optional
        New directory for the manifest, summary, section histories and
        annual damage fields. It must not exist.
    cable_source : str or pathlib.Path, optional
        The cable YAML, digested into the manifest.
    capacity : CapacityBatchCheck, optional
        Capacity check of every section and run: utilisation histories,
        maxima and their times, exceedances, the minimum bend radius check
        and Gumbel extremes per group of seeds
        (:mod:`cablexsec.analysis.capacity_check`). ``None`` (default)
        leaves the results and the manifest unchanged.

    Returns
    -------
    SeaStateBatchResult
    """
    opts = SeaStateBatchOptions() if options is None else options
    if not runs or not sections:
        raise CableInputError("at least one run and one section are required")
    names = [run.name for run in runs]
    if len(set(names)) != len(names):
        raise CableInputError("run names must be unique")
    section_names = [section.name for section in sections]
    if len(set(section_names)) != len(section_names):
        raise CableInputError("section names must be unique")
    if not math.isclose(
        math.fsum(run.probability for run in runs), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise CableInputError(
            "run probabilities must sum to one; specify the complete exposure "
            "population (divide a condition's probability among its seeds)"
        )
    output = None if output_directory is None else Path(output_directory)
    if output is not None and output.exists():
        raise FileExistsError(f"{output} exists; choose a new output directory")
    # The cable file is digested before any run is read, so a wrong path
    # fails at once rather than after every section is computed.
    cable_file = None
    if cable_source is not None:
        source_path = Path(cable_source)
        if not source_path.is_file():
            raise FileNotFoundError(f"cable_source {source_path} is not a file")
        cable_file = {
            "path": str(source_path.resolve()),
            "sha256": _sha256_bytes(source_path.read_bytes()),
        }
    basis = _resolve_basis(cable, opts.tension_basis)
    diameter = (
        opts.external_diameter_m
        if opts.external_diameter_m is not None
        else _outer_diameter(cable)
    )
    area = math.pi * diameter**2 / 4.0
    properties = _property_check(cable, runs, sections, opts)
    groups: dict[str, list[str]] = {}
    if capacity is not None:
        groups = (
            {"all_runs": list(names)}
            if capacity.groups is None
            else {key: list(value) for key, value in capacity.groups.items()}
        )
        unknown = {name for value in groups.values() for name in value} - set(names)
        if unknown:
            raise CableInputError(
                f"capacity groups name unknown runs: {sorted(unknown)}"
            )

    tasks = [
        _Task(
            cable,
            run,
            tuple(_portable(s) for s in sections),
            opts,
            basis,
            area,
            capacity,
        )
        for run in runs
    ]
    if opts.workers == 1 or len(tasks) == 1:
        outcomes = [_evaluate(task) for task in tasks]
    else:
        import multiprocessing  # noqa: PLC0415

        context = multiprocessing.get_context("spawn")
        saved = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
        try:
            # Spawned workers inherit one-thread settings before any import.
            for name in _THREAD_VARIABLES:
                os.environ[name] = "1"
            with ProcessPoolExecutor(
                max_workers=min(opts.workers, len(tasks)),
                mp_context=context,
                initializer=_limit_threads,
            ) as pool:
                outcomes = list(pool.map(_evaluate_in_worker, tasks))
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    for outcome in outcomes:
        for category, message in outcome.pop("warnings", ()):
            warnings.warn(message, category, stacklevel=2)

    digest_cable = cable_digest(cable)
    option_record = asdict(opts)
    option_record.pop("workers")  # does not change results
    for name in (
        "curvature_check_rtol",
        "curvature_check_end_rtol",
        "curvature_check_atol_1_m",
    ):
        # A check of the read, not an input of the result: the digests of
        # existing assessments stay.
        option_record.pop(name)
    if option_record["friction_model"] is None:
        # The model the default resolves to: the digests of existing
        # assessments are unchanged, and follow a change of the default.
        option_record["friction_model"] = DEFAULT_FRICTION_MODEL
    if opts.wire_positions is None:
        # Absent when unset, so that the digests of existing assessments stay.
        option_record.pop("wire_positions")
    results: dict[str, SectionBatchResult] = {}
    histories: dict[str, dict[str, CableDynSectionHistory]] = {}
    for section in sections:
        assessment = _sha256_bytes(
            _canonical(
                {
                    "cable": digest_cable,
                    "section": _section_record(section),
                    "options": option_record,
                    "tension_basis": basis,
                    "external_diameter_m": diameter,
                }
            ).encode()
        )
        cases = []
        per_run: dict[str, dict[str, Any]] = {}
        fretting_total: dict[str, dict[str, FloatArray]] = {}
        for run, outcome in zip(runs, outcomes, strict=True):
            item = outcome["results"][section.name]
            fatigue: FatigueResult = item["fatigue"]
            cases.append(
                FatigueLoadCase(
                    name=run.name,
                    result=fatigue,
                    duration_s=item["duration_s"],
                    probability=run.probability,
                    assessment_id=assessment,
                )
            )
            histories.setdefault(section.name, {})[run.name] = outcome["histories"][
                section.name
            ]
            maximum = max(
                float(np.max(values)) for values in fatigue.D_bulk_per_fibre.values()
            )
            per_run[run.name] = {
                "probability": run.probability,
                "duration_s": item["duration_s"],
                "record_damage_max": maximum,
                "critical_layer": fatigue.critical_layer,
                "statistics": item["statistics"],
            }
            if item["fretting"] is not None:
                for face, values in item["fretting"].items():
                    total = fretting_total.setdefault(face, {})
                    for key, array in values.items():
                        increment = np.asarray(array, dtype=np.float64)
                        total[key] = (
                            total.get(key, np.zeros_like(increment)) + increment
                        )
        long_term = compute_long_term_fatigue(
            cases,
            exposure_years=opts.exposure_years,
            design_fatigue_factor=opts.design_fatigue_factor,
        )
        life = (
            math.inf
            if long_term.annual_damage == 0
            else 1.0 / (long_term.annual_damage * opts.design_fatigue_factor)
        )
        fretting_summary = None
        if fretting_total:
            fretting_summary = {
                face: {
                    "max_wear_depth_per_year_m": float(
                        np.max(values["wear_depth_per_year_m"])
                    ),
                    "max_D_fretting_per_year": float(
                        np.nanmax(values["D_fretting_per_year"])
                    )
                    if np.isfinite(values["D_fretting_per_year"]).any()
                    else math.nan,
                }
                for face, values in fretting_total.items()
            }
        capacity_runs: dict[str, CapacityCheckResult] = {}
        capacity_summary = None
        if capacity is not None:
            for run, outcome in zip(runs, outcomes, strict=True):
                capacity_runs[run.name] = outcome["results"][section.name]["capacity"]
            capacity_summary = _capacity_summary(capacity_runs, groups, capacity)
        results[section.name] = SectionBatchResult(
            name=section.name,
            arc_length_m=section.arc_length_m,
            long_term=long_term,
            life_years=life,
            runs=per_run,
            fretting=fretting_summary,
            capacity=capacity_summary,
            capacity_runs=capacity_runs,
        )

    import cablexsec  # noqa: PLC0415

    manifest: dict[str, Any] = {
        "format": "cablexsec-sea-state-batch",
        "version": 1,
        "cablexsec": cablexsec.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cable": {
            "name": cable.cable.name,
            "definition_sha256": digest_cable,
            "file": cable_file,
        },
        "options": option_record,
        "curvature_check": {
            "rtol": opts.curvature_check_rtol,
            "end_rtol": opts.curvature_check_end_rtol,
            "atol_1_m": opts.curvature_check_atol_1_m,
        },
        "workers": opts.workers,
        "tension_basis": basis,
        "external_diameter_m": diameter,
        "property_check": properties,
        "runs": [
            {
                "name": run.name,
                "probability": run.probability,
                "line_id": run.line_id,
                "retained_s": [run.start_time_s, run.end_time_s],
                "hang_off_rotation": _hang_off_record(run.hang_off_rotation),
                "files": outcome["files"],
                "checks": outcome["checks"],
            }
            for run, outcome in zip(runs, outcomes, strict=True)
        ],
        "sections": [_section_record(section) for section in sections],
        "results": {
            name: {
                "annual_damage": result.long_term.annual_damage,
                "life_years": None
                if math.isinf(result.life_years)
                else result.life_years,
                "design_utilization": result.long_term.design_utilization,
                "critical_layer": result.long_term.critical_layer,
                "critical_wire_index": result.long_term.critical_wire_index,
                "critical_fibre_deg": result.long_term.critical_fibre_deg,
                "run_contributions_per_year": (
                    result.long_term.case_contributions_at_critical_per_year
                ),
                "runs": result.runs,
                "fretting": result.fretting,
                # Present only when a capacity check was requested.
                **({} if result.capacity is None else {"capacity": result.capacity}),
            }
            for name, result in results.items()
        },
    }
    if capacity is not None:
        manifest["capacity_check"] = {
            "envelope_source": capacity.envelope.source,
            "tension_kind": capacity.envelope.tension_kind,
            "utilisation_factor": capacity.envelope.utilisation_factor,
            "tension_grid_N": capacity.envelope.tension_N.tolist(),
            "thresholds": list(capacity.thresholds),
            "fractile": capacity.fractile,
            "groups": groups,
            "blocks": capacity.blocks,
        }
    manifest["inputs_sha256"] = _sha256_bytes(
        _canonical(
            {
                "cable": digest_cable,
                "files": [f["sha256"] for r in manifest["runs"] for f in r["files"]],
                "options": option_record,
                "sections": manifest["sections"],
                "probabilities": [run.probability for run in runs],
            }
        ).encode()
    )
    if output is not None:
        _write(output, manifest, results, histories, area)
    return SeaStateBatchResult(
        sections=results,
        manifest=manifest,
        output_directory=output,
        histories=histories,
    )


def _write(
    output: Path,
    manifest: dict[str, Any],
    results: Mapping[str, SectionBatchResult],
    histories: Mapping[str, Mapping[str, CableDynSectionHistory]],
    area: float,
) -> None:
    output.mkdir(parents=True)
    for name, result in results.items():
        arrays: dict[str, Any] = {"fibre_angles_deg": result.long_term.fibre_angles_deg}
        for layer, values in result.long_term.annual_damage_per_fibre.items():
            arrays[f"annual_damage_{layer}"] = values
        np.savez_compressed(output / f"{name}_annual_damage.npz", **arrays)
        for run_name, check in result.capacity_runs.items():
            np.savez_compressed(
                output / f"{name}_{run_name}_utilisation.npz",
                time_s=check.time_s,
                utilisation=check.utilisation,
                governing=check.governing,
                tension_N=check.tension_N,
                curvature_1_m=check.curvature_1_m,
                mechanisms=np.array(check.mechanisms),
            )
        for run_name, history in histories[name].items():
            columns = history.to_columns(area)
            np.savez_compressed(
                output / f"{name}_{run_name}_history.npz",
                data=np.column_stack(list(columns.values())),
                columns=np.array(list(columns)),
            )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8"
    )


def _capacity_summary(
    checks: Mapping[str, CapacityCheckResult],
    groups: Mapping[str, Sequence[str]],
    spec: CapacityBatchCheck,
) -> dict[str, Any]:
    """Per-run maxima, the overall maximum and Gumbel extremes per group."""
    worst = max(checks, key=lambda name: checks[name].max_utilisation)
    extremes: dict[str, Any] = {}
    for group, members in groups.items():
        maxima = [checks[name].max_utilisation for name in members]
        if len(maxima) < 2 or not all(math.isfinite(v) for v in maxima):
            extremes[group] = {
                "runs": list(members),
                "fit": None,
                "note": "fewer than two finite maxima: no Gumbel fit",
            }
            continue
        extremes[group] = {
            "runs": list(members),
            "fit": vars(gumbel_fit(maxima, fractile=spec.fractile)),
            "method": "Gumbel method of moments on seed maxima [Gumbel1958]",
        }
    return {
        "runs": {name: check.summary() for name, check in checks.items()},
        "max_utilisation": checks[worst].max_utilisation,
        "max_run": worst,
        "time_of_max_s": checks[worst].time_of_max_s,
        "governing_at_max": checks[worst].governing_at_max,
        "extremes": extremes,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")
