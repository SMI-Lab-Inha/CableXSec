# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Object-oriented analyses and versioned, self-contained engineering projects.

Version 2 supports stiffness, scalar planar bending, local stress, history
fatigue, steady thermal rating, capacity envelopes, coiling, conductor fatigue
and the hysteretic bending law for global analysis; version 1 files (the first
four) still load. It delegates physics to the established functions. Jobs are
synchronous; GUI scheduling, cancellation and project-local material plugins are
separate work.
"""

from __future__ import annotations

import hashlib
import os
import platform
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal, cast
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from cablexsec import api
from cablexsec._project_codec import canonical, pack, parse, unpack
from cablexsec.analysis import bending_law, coiling
from cablexsec.analysis.bending_law import BendingLawExport
from cablexsec.analysis.capacity import CapacityResult
from cablexsec.analysis.coiling import CoilingResult
from cablexsec.analysis.conductor_fatigue import (
    ConductorFatigueResult,
    CriticalDistanceSpec,
)
from cablexsec.cables.schema import CableData, load_cable, uses_cable_materials
from cablexsec.exceptions import CableInputError, CableLookupError
from cablexsec.io import series as _series
from cablexsec.io.series import LoadCaseSeries
from cablexsec.materials import library, sn_curves
from cablexsec.mechanics import thermal as _thermal
from cablexsec.mechanics.thermal import ThermalResult

# The physics a model runs reads the cable's own ``materials`` block: every
# function called here runs in the cable's material scope. The decorator
# returns a function that is already scoped unchanged, so nothing is applied
# twice; the ``api`` functions are scoped by ``cablexsec.api``.
compute_thermal = uses_cable_materials(_thermal.compute_thermal)
fatigue_from_series = uses_cable_materials(_series.fatigue_from_series)
compute_coiling = uses_cable_materials(coiling.compute_coiling)
export_bending_law = uses_cable_materials(bending_law.export_bending_law)

PROJECT_FORMAT_VERSION = 2
_READABLE_VERSIONS = (1, 2)
AnalysisResult = (
    api.StiffnessResult
    | api.MkappaResult
    | api.StressResult
    | api.FatigueResult
    | ThermalResult
    | CapacityResult
    | CoilingResult
    | ConductorFatigueResult
    | BendingLawExport
)


class _Settings(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, strict=True
    )


class StiffnessSettings(_Settings):
    kind: Literal["stiffness"] = "stiffness"


class BendingSettings(_Settings):
    """Planar x-curvature with constant true tension from the load case."""

    kind: Literal["bending"] = "bending"
    mu: float | None = Field(default=None, ge=0)
    contact_model: Literal["bounded", "qp"] = "bounded"


class StressSettings(_Settings):
    """Isothermal local recovery; advanced solved states use the function API."""

    kind: Literal["stress"] = "stress"
    kappa_x_1_m: float = 0.0
    kappa_y_1_m: float = 0.0
    tension_N: float = 0.0
    mu: float | None = Field(default=None, ge=0)
    contact_model: Literal["bounded", "qp"] = "bounded"
    p_int_Pa: float = Field(default=0.0, ge=0)
    p_ext_Pa: float | None = Field(default=None, ge=0)
    torque_Nm: float = 0.0
    prescribed_twist_per_m: float | None = 0.0


class FatigueSettings(_Settings):
    """The existing LoadCaseSeries fatigue bridge and its thermal conventions."""

    kind: Literal["fatigue"] = "fatigue"
    contact_tension_N: float | None = None
    mu: float | None = Field(default=None, ge=0)
    T_ref_C: float = 0.0
    n_fibres: int = Field(default=32, ge=4)


class ThermalSettings(_Settings):
    """Steady IEC 60287 rating (:func:`cablexsec.mechanics.thermal.compute_thermal`)."""

    kind: Literal["thermal"] = "thermal"
    current_A: float = Field(ge=0)
    conductor_area_m2: float = Field(gt=0)
    n_cores: int | None = Field(default=None, ge=1)
    #: ``None``: the material of the cable's conductor layer.
    conductor_material: str | None = None
    conductor_temp_C: float = 90.0
    y_s: float = Field(default=0.0, ge=0)
    y_p: float = Field(default=0.0, ge=0)
    lambda1: float = Field(default=0.0, ge=0)
    T4_external_Km_W: float | None = Field(default=None, ge=0)
    ambient_C: float | None = None


class CapacitySettings(_Settings):
    """Capacity envelope over a tension sweep (:func:`cablexsec.compute_capacity`)."""

    kind: Literal["capacity"] = "capacity"
    tension_N: tuple[float, ...] = Field(min_length=1, strict=False)
    mu: float | None = Field(default=None, ge=0)
    sigma_T_armour_Pa: float = 0.0
    contact_model: Literal["bounded", "qp", "solved"] = "bounded"
    polymer_crush_allowable_Pa: float | None = Field(default=None, gt=0)
    polymer_crush_source: str | None = None


class CoilingSettings(_Settings):
    """Coiling into a static tank (:func:`cablexsec.compute_coiling`)."""

    kind: Literal["coiling"] = "coiling"
    tank_diameter_m: float | None = Field(default=None, gt=0)
    curvature_per_m: float | None = Field(default=None, gt=0)
    twist_per_m: float | None = None
    torque_Nm: float | None = None
    twist_sense: Literal["unwind", "tighten", "signed"] = "unwind"
    tension_N: float = 0.0
    mu: float | None = Field(default=None, ge=0)
    turns: float = Field(default=1.0, gt=0)
    n_ramp: int = Field(default=40, ge=2)
    n_per_turn: int = Field(default=180, ge=8)
    n_fibres: int = Field(default=32, ge=4)


class ConductorFatigueSettings(_Settings):
    """Stranded-conductor fatigue over the load case (curvatures, true tension)."""

    kind: Literal["conductor_fatigue"] = "conductor_fatigue"
    layer_name: str | None = None
    mu_wire: float | None = Field(default=None, ge=0)
    mu_outer: float | None = Field(default=None, ge=0)
    reference_tension_N: float | None = Field(default=None, gt=0)
    external_pressure_Pa: float = Field(default=0.0, ge=0)
    n_orientations: int = Field(default=8, ge=1)
    mean_stress: Literal["swt", "none"] = "swt"
    sn_curve: str | None = None
    trellis_fretting: bool = False
    trellis_model: Literal["critical_distance", "surface_swt"] = "critical_distance"
    trellis_threshold_MPa_sqrt_m: float | None = Field(default=None, gt=0)


class BendingLawSettings(_Settings):
    """Hysteretic bending law for global analysis (``export_bending_law``)."""

    kind: Literal["bending_law"] = "bending_law"
    tensions_N: tuple[float, ...] = Field(min_length=1, strict=False)
    kappa_max_1_m: float = Field(gt=0)
    n_points: int = Field(default=200, ge=3)
    spacing: Literal["geometric", "linear"] = "geometric"
    mu: float | None = Field(default=None, ge=0)
    contact_model: Literal["bounded", "qp", "solved"] = "bounded"
    friction_model: Literal["direction_tied", "progressive"] = "direction_tied"
    slip_scale: bool = False
    slip_scale_reference_N: float | None = Field(default=None, ge=0, strict=False)
    slip_scale_tensions_N: tuple[float, ...] | None = Field(
        default=None, min_length=1, strict=False
    )
    slip_scale_external_pressure_Pa: tuple[float, ...] | None = Field(
        default=None, min_length=1, strict=False
    )


AnalysisSettings = Annotated[
    StiffnessSettings
    | BendingSettings
    | StressSettings
    | FatigueSettings
    | ThermalSettings
    | CapacitySettings
    | CoilingSettings
    | ConductorFatigueSettings
    | BendingLawSettings,
    Field(discriminator="kind"),
]

#: Settings kinds that read a load case.
_HISTORY_KINDS = ("bending", "fatigue", "conductor_fatigue")


class AnalysisRequest(_Settings):
    name: str = Field(min_length=1)
    model: str = Field(min_length=1)
    settings: AnalysisSettings
    load_case: str | None = None

    @model_validator(mode="after")
    def _history_requirement(self) -> AnalysisRequest:
        _name(self.name)
        _name(self.model)
        requires_history = self.settings.kind in _HISTORY_KINDS
        if requires_history != (self.load_case is not None):
            raise ValueError(
                "bending, fatigue and conductor_fatigue require a load_case; "
                "the other analyses do not"
            )
        if self.load_case is not None:
            _name(self.load_case)
        return self


_VERSION_1_KINDS = ("stiffness", "stress", "bending", "fatigue")
_RESULT_TYPES: dict[str, type] = {
    "stiffness": api.StiffnessResult,
    "stress": api.StressResult,
    "bending": api.MkappaResult,
    "fatigue": api.FatigueResult,
    "thermal": ThermalResult,
    "capacity": CapacityResult,
    "coiling": CoilingResult,
    "conductor_fatigue": ConductorFatigueResult,
    "bending_law": BendingLawExport,
}


def _name(value: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise CableInputError(
            "names must be nonempty and have no surrounding whitespace"
        )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _engine() -> str:
    from cablexsec import __version__

    package = Path(__file__).parent
    source = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        source.update(path.relative_to(package).as_posix().encode())
        source.update(b"\0")
        source.update(path.read_bytes())
    try:
        contact_version = version("clarabel")
    except PackageNotFoundError:
        contact_version = None
    try:
        kernel_version: str | None = version("numba")
    except PackageNotFoundError:
        kernel_version = None
    return canonical(
        {
            "cablexsec": __version__,
            "source_sha256": source.hexdigest(),
            "python": platform.python_version(),
            "platform": platform.system(),
            "clarabel": contact_version,
            # Optional compiled kernels; None when numba is not installed.
            "numba": kernel_version,
            "dependencies": {
                name: version(name)
                for name in ("numpy", "scipy", "pydantic", "rainflow")
            },
            "materials": pack(library.MATERIALS),
            "temperature_models": pack(library.TEMPERATURE_MODULUS_MODELS),
            "sn_curves": pack(sn_curves.SN_CURVES),
            "swt_life": pack(sn_curves.SWT_LIFE),
            "blocked_curves": pack(sn_curves.BLOCKED_SN_CURVES),
        }
    )


def _validate_engine(value: Any) -> None:
    fields = {
        "cablexsec",
        "source_sha256",
        "python",
        "platform",
        "clarabel",
        "dependencies",
        "materials",
        "temperature_models",
        "sn_curves",
        "swt_life",
        "blocked_curves",
    }
    # Records written before the numba version was stored omit that field.
    if not isinstance(value, dict) or set(value) not in (fields, fields | {"numba"}):
        raise CableInputError("invalid engine metadata fields")
    if value.get("numba") is not None and (
        not isinstance(value["numba"], str) or not value["numba"]
    ):
        raise CableInputError("invalid engine optional numba version")
    if any(
        not isinstance(value[name], str) or not value[name]
        for name in ("cablexsec", "source_sha256", "python", "platform")
    ):
        raise CableInputError("invalid engine identity")
    digest = value["source_sha256"]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise CableInputError("invalid engine source digest")
    dependencies = value["dependencies"]
    if (
        not isinstance(dependencies, dict)
        or set(dependencies) != {"numpy", "scipy", "pydantic", "rainflow"}
        or any(not isinstance(v, str) or not v for v in dependencies.values())
    ):
        raise CableInputError("invalid engine dependencies")
    if value["clarabel"] is not None and not isinstance(value["clarabel"], str):
        raise CableInputError("invalid engine optional solver version")
    for name in (
        "materials",
        "temperature_models",
        "sn_curves",
        "swt_life",
        "blocked_curves",
    ):
        if not isinstance(unpack(value[name]), dict):
            raise CableInputError(f"invalid engine registry: {name}")


@dataclass(frozen=True, init=False)
class CableModel:
    """An owned, validated cable definition with convenient analysis methods."""

    _json: str

    def __init__(self, definition: CableData) -> None:
        # Revalidation also catches invalid Pydantic model_copy(update=...) edits.
        validated = CableData.model_validate(definition.model_dump(mode="json"))
        object.__setattr__(self, "_json", canonical(validated.model_dump(mode="json")))

    @classmethod
    def from_yaml(
        cls, path: str | Path, *, geometry_override: str | Path | None = None
    ) -> CableModel:
        return cls(load_cable(path, geometry_override=geometry_override))

    @property
    def definition(self) -> CableData:
        """Return an independent definition; edits cannot mutate the model."""
        return CableData.model_validate_json(self._json)

    def stiffness(self) -> api.StiffnessResult:
        return api.compute_stiffness(self.definition)

    def stress(self, settings: StressSettings) -> api.StressResult:
        options = StressSettings.model_validate(settings.model_dump())
        return api.compute_stress(
            self.definition,
            options.kappa_x_1_m,
            options.kappa_y_1_m,
            **options.model_dump(exclude={"kind", "kappa_x_1_m", "kappa_y_1_m"}),
        )

    def bending(
        self, series: LoadCaseSeries, settings: BendingSettings | None = None
    ) -> api.MkappaResult:
        options = BendingSettings.model_validate(
            (settings or BendingSettings()).model_dump()
        )
        if series.n_steps < 2 or series.tension_kind != "true":
            raise CableInputError(
                "planar bending needs at least two samples and true tension"
            )
        if np.any(series.curvature_y_1_m != 0) or np.any(
            series.tension_N != series.tension_N[0]
        ):
            raise CableInputError(
                "planar bending requires zero y-curvature and constant tension"
            )
        if series.temperature_C is not None or series.axial_strain is not None:
            raise CableInputError(
                "planar bending does not consume temperature or axial strain histories"
            )
        return api.compute_mkappa(
            self.definition,
            series.curvature_x_1_m,
            tension_N=float(series.tension_N[0]),
            **options.model_dump(exclude={"kind"}),
        )

    def fatigue(
        self, series: LoadCaseSeries, settings: FatigueSettings | None = None
    ) -> api.FatigueResult:
        options = FatigueSettings.model_validate(
            (settings or FatigueSettings()).model_dump()
        )
        return fatigue_from_series(
            self.definition, series, **options.model_dump(exclude={"kind"})
        )

    def thermal(self, settings: ThermalSettings) -> ThermalResult:
        options = ThermalSettings.model_validate(settings.model_dump())
        return compute_thermal(
            self.definition,
            options.current_A,
            options.conductor_area_m2,
            **options.model_dump(exclude={"kind", "current_A", "conductor_area_m2"}),
        )

    def capacity(self, settings: CapacitySettings) -> CapacityResult:
        options = CapacitySettings.model_validate(settings.model_dump())
        return api.compute_capacity(
            self.definition,
            np.asarray(options.tension_N, dtype=np.float64),
            **options.model_dump(exclude={"kind", "tension_N"}),
        )

    def coiling(self, settings: CoilingSettings) -> CoilingResult:
        options = CoilingSettings.model_validate(settings.model_dump())
        return compute_coiling(self.definition, **options.model_dump(exclude={"kind"}))

    def conductor_fatigue(
        self, series: LoadCaseSeries, settings: ConductorFatigueSettings | None = None
    ) -> ConductorFatigueResult:
        options = ConductorFatigueSettings.model_validate(
            (settings or ConductorFatigueSettings()).model_dump()
        )
        if series.n_steps < 2 or series.tension_kind != "true":
            raise CableInputError(
                "conductor fatigue needs at least two samples and true tension"
            )
        if series.temperature_C is not None or series.axial_strain is not None:
            raise CableInputError(
                "conductor fatigue does not consume temperature or axial strain "
                "histories (the strand model takes curvature and true tension)"
            )
        threshold = options.trellis_threshold_MPa_sqrt_m
        return api.compute_conductor_fatigue(
            self.definition,
            series.curvature_x_1_m,
            series.curvature_y_1_m,
            tension_N=series.tension_N,
            layer_name=options.layer_name,
            mu_wire=options.mu_wire,
            mu_outer=options.mu_outer,
            reference_tension_N=options.reference_tension_N,
            external_pressure_Pa=options.external_pressure_Pa,
            n_orientations=options.n_orientations,
            mean_stress=None if options.mean_stress == "none" else options.mean_stress,
            sn_curve=options.sn_curve,
            trellis_fretting=options.trellis_fretting,
            trellis_model=options.trellis_model,
            critical_distance=(
                None
                if threshold is None
                else CriticalDistanceSpec(threshold_MPa_sqrt_m=threshold)
            ),
        )

    def bending_law(self, settings: BendingLawSettings) -> BendingLawExport:
        options = BendingLawSettings.model_validate(settings.model_dump())
        return export_bending_law(
            self.definition,
            tensions_N=options.tensions_N,
            kappa_max_1_m=options.kappa_max_1_m,
            n_points=options.n_points,
            spacing=options.spacing,
            mu=options.mu,
            contact_model=options.contact_model,
            friction_model=options.friction_model,
            slip_scale=options.slip_scale,
            slip_scale_reference_N=options.slip_scale_reference_N,
            slip_scale_tensions_N=options.slip_scale_tensions_N,
            slip_scale_external_pressure_Pa=options.slip_scale_external_pressure_Pa,
        )


@dataclass(frozen=True)
class AnalysisRecord:
    """A completed run. Accessors reconstruct independent inputs/results."""

    run_id: str
    created_utc: str
    _inputs_json: str
    _result_json: str
    _engine_json: str

    @property
    def fingerprint(self) -> str:
        return _digest(canonical([self._inputs_json, self._engine_json]))

    @property
    def request(self) -> AnalysisRequest:
        return AnalysisRequest.model_validate(parse(self._inputs_json)["request"])

    @property
    def inputs(self) -> dict[str, Any]:
        return cast(dict[str, Any], parse(self._inputs_json))

    @property
    def model(self) -> CableModel:
        return CableModel(CableData.model_validate(self.inputs["model"]))

    @property
    def series(self) -> LoadCaseSeries | None:
        return cast(LoadCaseSeries | None, unpack(self.inputs["series"]))

    @property
    def engine(self) -> dict[str, Any]:
        return cast(dict[str, Any], parse(self._engine_json))

    @property
    def result(self) -> AnalysisResult:
        return cast(AnalysisResult, unpack(parse(self._result_json)))


def _inputs(
    model: CableModel, request: AnalysisRequest, series: LoadCaseSeries | None
) -> str:
    return canonical(
        {
            "model": parse(model._json),
            "request": request.model_dump(mode="json"),
            "series": pack(series),
        }
    )


class AnalysisRunner:
    """Synchronous, stateless executor; each call owns its inputs and record."""

    def run(
        self,
        model: CableModel,
        request: AnalysisRequest,
        series: LoadCaseSeries | None = None,
    ) -> AnalysisRecord:
        request = AnalysisRequest.model_validate(request.model_dump())
        if (request.load_case is not None) != (series is not None):
            raise CableInputError("request and supplied history must agree")
        inputs = _inputs(model, request, series)
        engine = _engine()
        options = request.settings
        result: AnalysisResult
        if isinstance(options, StiffnessSettings):
            result = model.stiffness()
        elif isinstance(options, StressSettings):
            result = model.stress(options)
        elif isinstance(options, BendingSettings):
            assert series is not None
            result = model.bending(series, options)
        elif isinstance(options, FatigueSettings):
            assert series is not None
            result = model.fatigue(series, options)
        elif isinstance(options, ThermalSettings):
            result = model.thermal(options)
        elif isinstance(options, CapacitySettings):
            result = model.capacity(options)
        elif isinstance(options, CoilingSettings):
            result = model.coiling(options)
        elif isinstance(options, ConductorFatigueSettings):
            assert series is not None
            result = model.conductor_fatigue(series, options)
        else:
            result = model.bending_law(options)
        return AnalysisRecord(
            str(uuid4()),
            datetime.now(UTC).isoformat(),
            inputs,
            canonical(pack(result)),
            engine,
        )


class Project:
    """Named models, histories, analysis requests and preserved completed runs.

    Mutating methods validate before replacing an entry. A completed record is
    never overwritten by an edit or rerun. Use one controlling application thread
    for project edits; AnalysisRunner can execute independent snapshots in workers.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._engine_json = _engine()
        self._models: dict[str, CableModel] = {}
        self._loads: dict[str, LoadCaseSeries] = {}
        self._analyses: dict[str, AnalysisRequest] = {}
        self._records: list[AnalysisRecord] = []

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        _name(value)
        self._name = value

    @property
    def engine(self) -> dict[str, Any]:
        """Saved creation environment, including registered material/curve data."""
        return cast(dict[str, Any], parse(self._engine_json))

    @property
    def environment_changed(self) -> bool:
        """Whether today's source, dependencies or registry differ from creation."""
        return self._engine_json != _engine()

    @property
    def models(self) -> Mapping[str, CableModel]:
        return MappingProxyType(self._models)

    @property
    def load_cases(self) -> Mapping[str, LoadCaseSeries]:
        return MappingProxyType(self._loads)

    @property
    def analyses(self) -> Mapping[str, AnalysisRequest]:
        return MappingProxyType(self._analyses)

    @property
    def records(self) -> tuple[AnalysisRecord, ...]:
        return tuple(self._records)

    def set_model(self, name: str, model: CableModel) -> None:
        _name(name)
        if not isinstance(model, CableModel):
            raise CableInputError("model must be a CableModel")
        self._models[name] = CableModel(model.definition)

    def set_load_case(self, name: str, series: LoadCaseSeries) -> None:
        _name(name)
        if not isinstance(series, LoadCaseSeries):
            raise CableInputError("load case must be a LoadCaseSeries")
        self._loads[name] = replace(series)

    def _resolve(
        self, request: AnalysisRequest
    ) -> tuple[CableModel, LoadCaseSeries | None]:
        try:
            return self._models[request.model], self._loads[
                request.load_case
            ] if request.load_case is not None else None
        except KeyError as error:
            raise CableLookupError(
                f"analysis references a missing model or load case: {error}"
            ) from error

    def set_analysis(self, request: AnalysisRequest) -> None:
        validated = AnalysisRequest.model_validate(request.model_dump())
        self._resolve(validated)
        self._analyses[validated.name] = validated

    def run(self, name: str) -> AnalysisRecord:
        try:
            request = self._analyses[name]
        except KeyError as error:
            raise CableLookupError(f"unknown analysis: {name}") from error
        model, series = self._resolve(request)
        record = AnalysisRunner().run(model, request, series)
        self._records.append(record)
        return record

    def is_stale(self, record: AnalysisRecord) -> bool:
        request = self._analyses.get(record.request.name)
        if request is None:
            return True
        model, series = self._resolve(request)
        return record.fingerprint != _digest(
            canonical([_inputs(model, request, series), _engine()])
        )

    def save(self, path: str | Path) -> Path:
        """Atomically replace a self-contained UTF-8 JSON project file."""
        _name(self.name)
        payload = {
            "name": self.name,
            "engine": self.engine,
            "models": {k: parse(v._json) for k, v in self._models.items()},
            "load_cases": {k: pack(v) for k, v in self._loads.items()},
            "analyses": [v.model_dump(mode="json") for v in self._analyses.values()],
            "records": [vars(v) for v in self._records],
        }
        document = canonical(
            {
                "format": "CableXSec Project",
                "version": PROJECT_FORMAT_VERSION,
                "sha256": _digest(canonical(payload)),
                "payload": payload,
            }
        )
        target = Path(path)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target.parent,
                prefix=".cablexsec-",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(document + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return target

    @classmethod
    def load(cls, path: str | Path) -> Project:
        """Load versions 1 and 2; reject unknown versions and inconsistent contents."""
        try:
            text = Path(path).read_bytes().decode("utf-8")
        except UnicodeDecodeError as error:
            raise CableInputError(
                f"invalid project file (not UTF-8): {error}"
            ) from None
        data = parse(text)
        try:
            if (
                set(data) != {"format", "version", "sha256", "payload"}
                or data["format"] != "CableXSec Project"
            ):
                raise CableInputError("invalid project envelope")
            if (
                type(data["version"]) is not int
                or data["version"] not in _READABLE_VERSIONS
            ):
                raise CableInputError("unsupported project format version")
            version_1 = data["version"] == 1
            payload = data["payload"]
            if data["sha256"] != _digest(canonical(payload)):
                raise CableInputError("project checksum mismatch")
            if set(payload) != {
                "name",
                "engine",
                "models",
                "load_cases",
                "analyses",
                "records",
            }:
                raise CableInputError("invalid project fields")
            project = cls(payload["name"])
            _validate_engine(payload["engine"])
            project._engine_json = canonical(payload["engine"])
            for name, definition in payload["models"].items():
                project.set_model(
                    name, CableModel(CableData.model_validate(definition))
                )
            for name, packed in payload["load_cases"].items():
                series = unpack(packed)
                if not isinstance(series, LoadCaseSeries):
                    raise CableInputError("load case must contain a LoadCaseSeries")
                project.set_load_case(name, series)
            for item in payload["analyses"]:
                request = AnalysisRequest.model_validate(item)
                if version_1 and request.settings.kind not in _VERSION_1_KINDS:
                    raise CableInputError(
                        f"analysis kind {request.settings.kind!r} needs "
                        "project version 2"
                    )
                if request.name in project.analyses:
                    raise CableInputError("duplicate analysis name")
                project.set_analysis(request)
            for item in payload["records"]:
                record = AnalysisRecord(**item)
                expected = _RESULT_TYPES[record.request.settings.kind]
                if not isinstance(record.result, expected):
                    raise CableInputError("result type does not match its analysis")
                inputs = record.inputs
                if set(inputs) != {"model", "request", "series"}:
                    raise CableInputError("invalid record input fields")
                CableModel(CableData.model_validate(inputs["model"]))
                history = unpack(inputs["series"])
                if (record.request.load_case is None and history is not None) or (
                    record.request.load_case is not None
                    and not isinstance(history, LoadCaseSeries)
                ):
                    raise CableInputError("record history does not match its request")
                _validate_engine(parse(record._engine_json))
                _name(record.run_id)
                datetime.fromisoformat(record.created_utc)
                if any(r.run_id == record.run_id for r in project.records):
                    raise CableInputError("duplicate run identifier")
                project._records.append(record)
            return project
        except (
            TypeError,
            ValueError,
            KeyError,
            AttributeError,
            ValidationError,
        ) as error:
            raise CableInputError(f"invalid CableXSec project: {error}") from error
