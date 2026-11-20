# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Section histories with explicit force, strain and curvature conventions.

Readers retain the declared material force or effective tension and signed
strain-gradient components. The fatigue interface requires true force or
direct stress-driving strain. Pressure-thrust conversion and normal-section
equilibrium are separate operations, not inferred from an unlabelled column.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from cablexsec.analysis.fatigue import FatigueResult
from cablexsec.cables.schema import CableData, uses_cable_materials
from cablexsec.exceptions import CableSchemaError
from cablexsec.provenance import Provenance

FloatArray = NDArray[np.float64]

# Canonical exchange columns written/read by write_series_csv / read_series_csv.
_CSV_COLUMNS = (
    "time_s",
    "curvature_x_1_m",
    "curvature_y_1_m",
    "effective_tension_N",
    "temperature_C",
)

__all__ = [
    "LoadCaseSeries",
    "fatigue_from_series",
    "read_series_csv",
    "write_series_csv",
]


@dataclass(frozen=True)
class LoadCaseSeries:
    """A uniform-step global-analysis load history at one cross-section.

    Histories are copied into read-only snapshots. Editing an input array later
    cannot change this series. Use ``dataclasses.replace`` to create and validate
    an edited series; its histories are independent snapshots too.

    Attributes
    ----------
    time_s : numpy.ndarray
        Sample times [s], strictly increasing, (approximately) uniform step.
    curvature_x_1_m, curvature_y_1_m : numpy.ndarray
        Axial-strain-gradient components [1/m] in the section frame, such
        that bending strain is curvature_x*x + curvature_y*y. Native rod
        rotation curvatures must be transformed before local recovery.
    tension_N : numpy.ndarray
        Axial resultant [N]; tension_kind distinguishes true from effective
        tension. Neither can silently be substituted for the other in water.
    axial_strain, axial_strain_kind : numpy.ndarray or None, str
        Optional direct stress-driving or total geometric axial strain. Only
        kind "direct" can enter the one-dimensional fatigue bridge. The
        canonical CSV preserves this distinction in its column heading.
    temperature_C : numpy.ndarray or None
        Optional conductor/section temperature history [°C]; ``None`` means the
        series is isothermal and the downstream ``σ_T`` term is zero.
    source : str
        Where the data came from (file name / generator), for provenance.
    provenance : Provenance or None
        Origin record; the *fatigue* provenance is attached by
        :func:`fatigue_from_series`.
    """

    time_s: FloatArray
    curvature_x_1_m: FloatArray
    curvature_y_1_m: FloatArray
    tension_N: FloatArray
    temperature_C: FloatArray | None = None
    source: str = ""
    provenance: Provenance | None = None
    tension_kind: Literal["effective", "true", "unspecified"] = "unspecified"
    axial_strain: FloatArray | None = None
    axial_strain_kind: Literal["direct", "total", "unspecified"] = "unspecified"

    def __post_init__(self) -> None:
        for name in (
            "time_s",
            "curvature_x_1_m",
            "curvature_y_1_m",
            "tension_N",
            "temperature_C",
            "axial_strain",
        ):
            values = getattr(self, name)
            if values is None and name in ("temperature_C", "axial_strain"):
                continue
            try:
                if np.iscomplexobj(values):
                    raise CableSchemaError(f"LoadCaseSeries: {name} must be real")
                values = np.asarray(values, dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as error:
                raise CableSchemaError(
                    f"LoadCaseSeries: {name} must be a real numeric history"
                ) from error
            if values.ndim != 1 or not np.isfinite(values).all():
                raise CableSchemaError(
                    f"LoadCaseSeries: {name} must be a finite one-dimensional history"
                )
            # Immutable backing storage prevents both source-array aliasing and
            # re-enabling writes through setflags on the published snapshot.
            snapshot = np.frombuffer(values.tobytes(), dtype=np.float64)
            object.__setattr__(self, name, snapshot)
        if self.time_s.size == 0:
            raise CableSchemaError("LoadCaseSeries: time_s must not be empty")
        if np.any(np.diff(self.time_s) <= 0.0):
            raise CableSchemaError("LoadCaseSeries: time_s must be strictly increasing")
        if self.axial_strain_kind not in ("direct", "total", "unspecified"):
            raise CableSchemaError(
                "axial_strain_kind must be direct, total or unspecified"
            )
        if self.tension_kind not in ("effective", "true", "unspecified"):
            raise CableSchemaError(
                "tension_kind must be effective, true or unspecified"
            )
        n = self.time_s.size
        if self.axial_strain is not None and (
            self.axial_strain.shape != self.time_s.shape
            or not np.isfinite(self.axial_strain).all()
        ):
            raise CableSchemaError(
                "axial_strain must be finite and match the time history"
            )
        for name in ("curvature_x_1_m", "curvature_y_1_m", "tension_N"):
            if getattr(self, name).size != n:
                raise CableSchemaError(
                    f"LoadCaseSeries: {name} length {getattr(self, name).size} "
                    f"!= time_s length {n}."
                )
        if self.temperature_C is not None and self.temperature_C.size != n:
            raise CableSchemaError(
                f"LoadCaseSeries: temperature_C length {self.temperature_C.size} "
                f"!= time_s length {n}."
            )

    @property
    def n_steps(self) -> int:
        """Number of time samples."""
        return int(self.time_s.size)

    def __reduce__(self) -> tuple[type[LoadCaseSeries], tuple[object, ...]]:
        """Revalidate snapshots when copied or transferred to a worker process."""
        return type(self), tuple(getattr(self, item.name) for item in fields(self))

    @property
    def dt_s(self) -> float:
        """Mean sample step [s] (``0`` for a single-sample series)."""
        if self.time_s.size < 2:
            return 0.0
        return float(np.mean(np.diff(self.time_s)))

    def with_true_tension(
        self,
        external_pressure_Pa: FloatArray | float,
        external_area_m2: float,
        *,
        internal_pressure_Pa: FloatArray | float = 0.0,
        internal_area_m2: float = 0.0,
    ) -> LoadCaseSeries:
        """Convert effective tension without changing the source history in place."""
        from cablexsec.mechanics.section_conventions import true_tension_from_effective

        if self.tension_kind != "effective":
            raise CableSchemaError(
                "pressure-thrust conversion requires effective tension"
            )
        true = true_tension_from_effective(
            self.tension_N,
            external_pressure_Pa,
            external_area_m2,
            internal_pressure_Pa=internal_pressure_Pa,
            internal_area_m2=internal_area_m2,
        )
        return replace(
            self,
            tension_N=true,
            tension_kind="true",
            source=self.source + "; pressure-thrust force conversion",
        )


def write_series_csv(series: LoadCaseSeries, path: str | Path) -> Path:
    """Write a :class:`LoadCaseSeries` to the canonical exchange CSV.

    The columns are :data:`_CSV_COLUMNS`; ``temperature_C`` is written empty when
    the series is isothermal. The format round-trips exactly through
    :func:`read_series_csv`, including force and strain kinds.
    """
    p = Path(path)
    temp = series.temperature_C  # local for mypy narrowing
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        tension_column = {
            "effective": "effective_tension_N",
            "true": "true_tension_N",
            "unspecified": "axial_tension_N",
        }[series.tension_kind]
        strain_column = {
            "direct": "direct_axial_strain",
            "total": "total_axial_strain",
            "unspecified": "axial_strain",
        }[series.axial_strain_kind]
        w.writerow((*_CSV_COLUMNS[:3], tension_column, _CSV_COLUMNS[4], strain_column))
        for i in range(series.n_steps):
            w.writerow(
                [
                    repr(float(series.time_s[i])),
                    repr(float(series.curvature_x_1_m[i])),
                    repr(float(series.curvature_y_1_m[i])),
                    repr(float(series.tension_N[i])),
                    repr(float(temp[i])) if temp is not None else "",
                    repr(float(series.axial_strain[i]))
                    if series.axial_strain is not None
                    else "",
                ]
            )
    return p


#: A decimal number as a CSV cell spells one (``float`` alone also takes
#: ``1_000``); ``nan`` and ``inf`` pass, for the finite check to name them.
_NUMBER = re.compile(
    r"\s*[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf|infinity)\s*",
    re.IGNORECASE,
)


def parse_number(cell: str) -> float:
    """One numeric CSV cell; ``ValueError`` for anything but a decimal number."""
    if not isinstance(cell, str) or not _NUMBER.fullmatch(cell):
        raise ValueError(f"not a decimal number: {cell!r}")
    return float(cell)


def read_series_csv(path: str | Path) -> LoadCaseSeries:
    """Read the canonical exchange CSV written by :func:`write_series_csv`."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"series CSV not found: {p}")
    times, kx, ky, ten, temp = [], [], [], [], []
    strains = []
    any_strain = False
    any_temp = False
    try:
        text = p.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError:
        raise CableSchemaError(
            f"series CSV {p.name} is not UTF-8 text (write_series_csv writes UTF-8)"
        ) from None
    # A newline-preserving stream: a quoted cell keeps an embedded newline
    # (and is refused), where split lines would join its parts.
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fields = set(reader.fieldnames or [])
    if len(fields) != len(reader.fieldnames or []):
        raise CableSchemaError(
            f"series CSV {p.name} repeats a column name: {reader.fieldnames}"
        )
    tension_columns = fields & {
        "effective_tension_N",
        "true_tension_N",
        "axial_tension_N",
    }
    if len(tension_columns) != 1:
        raise CableSchemaError(
            "series CSV must contain exactly one named tension column"
        )
    tension_column = tension_columns.pop()
    strain_columns = fields & {
        "direct_axial_strain",
        "total_axial_strain",
        "axial_strain",
    }
    if len(strain_columns) > 1:
        raise CableSchemaError(
            "series CSV must contain at most one named axial-strain column"
        )
    strain_column = next(iter(strain_columns), "axial_strain")
    strain_kind: Literal["direct", "total", "unspecified"] = (
        "direct"
        if strain_column == "direct_axial_strain"
        else "total"
        if strain_column == "total_axial_strain"
        else "unspecified"
    )
    kind: Literal["effective", "true", "unspecified"]
    kind = (
        "true"
        if tension_column == "true_tension_N"
        else ("effective" if tension_column == "effective_tension_N" else "unspecified")
    )
    missing = set(_CSV_COLUMNS[:3]) - fields
    if missing:
        raise CableSchemaError(
            f"series CSV {p.name} missing columns {sorted(missing)}; "
            f"found {reader.fieldnames}."
        )
    for row in reader:
        # DictReader keeps the cells beyond the header as a list under None.
        surplus = row.pop(None, None) or []
        cells = [*row.values(), *surplus]
        if not any((value or "").strip() for value in cells):
            continue  # a blank row (spaces or separators only)
        if any(value.strip() for value in surplus):
            raise CableSchemaError(
                f"series CSV {p.name}, line {reader.line_num}: more cells than the "
                f"{len(reader.fieldnames or [])} columns of the header"
            )
        try:
            times.append(parse_number(row["time_s"]))
            kx.append(parse_number(row["curvature_x_1_m"]))
            ky.append(parse_number(row["curvature_y_1_m"]))
            ten.append(parse_number(row[tension_column]))
            strain_cell = (row.get(strain_column) or "").strip()
            any_strain |= bool(strain_cell)
            strains.append(parse_number(strain_cell) if strain_cell else np.nan)
            t_cell = (row.get("temperature_C") or "").strip()
            if t_cell:
                any_temp = True
                temp.append(parse_number(t_cell))
            else:
                temp.append(np.nan)
        except (TypeError, ValueError) as error:
            raise CableSchemaError(
                f"series CSV {p.name}, line {reader.line_num}: "
                "expected numeric history values"
            ) from error
    return LoadCaseSeries(
        time_s=np.asarray(times, dtype=np.float64),
        curvature_x_1_m=np.asarray(kx, dtype=np.float64),
        curvature_y_1_m=np.asarray(ky, dtype=np.float64),
        tension_N=np.asarray(ten, dtype=np.float64),
        temperature_C=np.asarray(temp, dtype=np.float64) if any_temp else None,
        source=p.name,
        tension_kind=kind,
        axial_strain=np.asarray(strains, dtype=np.float64) if any_strain else None,
        axial_strain_kind=strain_kind,
        provenance=Provenance(
            method="canonical CableXSec exchange CSV",
            citations=(),
            gates=("write/read round-trip identity (L6 plumbing gate)",),
            limitations=("data origin is the upstream global analysis",),
        ),
    )


@uses_cable_materials
def fatigue_from_series(
    cable: CableData,
    series: LoadCaseSeries,
    *,
    contact_tension_N: float | None = None,
    contact_tension_history_N: FloatArray | None = None,
    mu: float | None = None,
    T_ref_C: float = 0.0,
    n_fibres: int = 32,
) -> FatigueResult:
    r"""Map a :class:`LoadCaseSeries` onto :func:`cablexsec.compute_fatigue`.

    Forwards ``curvature_x/y`` and ``tension`` to fixed-material-fibre rainflow
    counting and Miner summation with the scope of ``compute_fatigue``.
    No generic Goodman transformation is applied. With prescribed direct
    strain, a ``temperature_C`` history is mapped to a uniform
    per-helical-layer thermal stress ``σ_T = -E α_T (T − T_ref_C)`` and passed to
    the fatigue pipeline; otherwise the analysis is isothermal (``σ_T = 0``).
    This is a prescribed-strain thermal increment, not force-controlled free
    expansion. A force/temperature history requires solved ``section_states``
    through ``compute_fatigue``. The uniform-per-layer mapping and constant
    elastic properties are recorded in the result provenance. ``E_Pa`` and
    ``alpha_T_K`` are read from the cable's own ``materials:`` block first and
    from the library second, by every import path of this function.

    Parameters
    ----------
    cable : CableData
        A fully-specified cable.
    series : LoadCaseSeries
        The global-analysis history.
    contact_tension_N : float or None, optional
        Explicit frozen contact reference tension [N]. By default the true-force
        history also drives contact, matching ``compute_fatigue``. With prescribed
        strain there is no inferred normal-force history; an omitted reference
        uses zero applied contact tension plus declared manufacturing clamping.
    contact_tension_history_N : numpy.ndarray or None, optional
        With prescribed direct strain, a force history [N] that sets the
        armour contact (declared clamping plus the tension increment), as
        :func:`cablexsec.compute_fatigue` takes it; for example the strain
        times the axial stiffness ``EA_fixed``.
    mu : float, optional
        Armour friction coefficient (default material ``mu_on_hdpe``).
    T_ref_C : float, optional
        Stress-free reference temperature for the ``σ_T`` mapping [°C].
    n_fibres : int, optional
        Fixed circumference fibres per wire; forwarded for angular refinement.

    Returns
    -------
    FatigueResult
        With a provenance recording the series source and the temperature
        limitation in addition to the pipeline's own caveats.

    Raises
    ------
    CableSchemaError
        With a temperature history, when the material of a structural helical
        layer lacks ``E_Pa`` or ``alpha_T_K``.
    """
    from cablexsec.api import compute_fatigue  # noqa: PLC0415 (avoid import cycle)

    if series.axial_strain is not None and series.axial_strain_kind != "direct":
        raise CableSchemaError(
            "The one-dimensional fatigue law requires direct axial strain, "
            "not total or unqualified strain containing Poisson deformation. "
            "Use a direct-strain record or the true-force route."
        )
    if series.tension_kind == "effective" and series.axial_strain is None:
        raise CableSchemaError(
            "effective tension is not a material force; convert pressure thrust "
            "with with_true_tension, or use a prescribed-strain local calculation"
        )
    if series.tension_kind == "unspecified" and series.axial_strain is None:
        raise CableSchemaError(
            "declare true or effective tension before fatigue; an unqualified "
            "force column is not assumed to be material force"
        )
    if series.temperature_C is not None and series.axial_strain is None:
        raise CableSchemaError(
            "force and temperature histories require solved section_states; "
            "the thermal-strain increment requires prescribed direct strain"
        )

    thermal_stress = None
    if series.temperature_C is not None:
        from cablexsec.materials import library  # noqa: PLC0415

        thermal_stress = {}
        for lyr in cable.helical_layers:
            if not lyr.structural:
                continue
            props = {}
            for prop in ("E_Pa", "alpha_T_K"):
                try:
                    props[prop] = library.get_property(lyr.material, prop)
                except KeyError as error:
                    raise CableSchemaError(
                        f"layer {lyr.name!r}: material {lyr.material!r} has no "
                        f"{prop!r}, which the thermal stress of the temperature "
                        "history needs; add it to the material with its source "
                        "or omit temperature_C"
                    ) from error
            e_w, alpha_t = props["E_Pa"], props["alpha_T_K"]
            # Fixed-global-strain thermal increment. Heating is compressive in
            # the tensile-positive convention. Cross-section free/common
            # expansion belongs to compute_component_response.
            thermal_stress[lyr.name] = -e_w * alpha_t * (series.temperature_C - T_ref_C)

    res = compute_fatigue(
        cable,
        series.curvature_x_1_m,
        series.curvature_y_1_m,
        tension_N=series.tension_N if series.axial_strain is None else None,
        axial_strain=series.axial_strain,
        contact_tension_N=contact_tension_N,
        contact_tension_history_N=contact_tension_history_N,
        mu=mu,
        thermal_stress_Pa=thermal_stress,
        n_fibres=n_fibres,
    )
    base = res.provenance
    origin = series.source or "an upstream global analysis"
    extra_lims = (
        f"driven by a LoadCaseSeries from {origin} "
        f"({series.n_steps} steps, dt≈{series.dt_s:.4g} s)",
        "temperature_C mapped to the fixed-strain increment "
        "sigma_T=-E*alpha_T*(T-T_ref)"
        if series.temperature_C is not None
        else "isothermal fatigue (no temperature_C in LoadCaseSeries)",
        "contact follows the true material-force history"
        if contact_tension_N is None and series.axial_strain is None
        else "contact follows the supplied contact tension history with the "
        "declared manufacturing clamping; the prescribed strain drives the wire "
        "stress"
        if contact_tension_history_N is not None
        else "contact uses an explicit frozen reference tension "
        f"of {contact_tension_N if contact_tension_N is not None else 0:g} N "
        "plus declared manufacturing clamping; prescribed strain does not "
        "determine a coupled normal state",
    )
    prov = Provenance(
        method=(base.method if base else "fixed-fibre rainflow + Miner"),
        citations=(base.citations if base else ()),
        gates=(base.gates if base else ()),
        limitations=((base.limitations if base else ()) + extra_lims),
    )
    return _with_provenance(res, prov)


def _with_provenance(res: FatigueResult, prov: Provenance) -> FatigueResult:
    """Return a copy of ``res`` with ``provenance`` replaced (frozen dataclass)."""
    from dataclasses import replace  # noqa: PLC0415

    return replace(res, provenance=prov)
