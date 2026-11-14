# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Explicit JSON encoding for project histories and supported result objects."""

from __future__ import annotations

import json
import math
import types
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from functools import cache
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import numpy as np

from cablexsec.analysis.bending_law import BendingLawExport, IwanBendingLaw
from cablexsec.analysis.capacity import CapacityResult
from cablexsec.analysis.coiling import CoilingLayerResult, CoilingResult
from cablexsec.analysis.conductor_fatigue import ConductorFatigueResult
from cablexsec.analysis.fatigue import FatigueResult
from cablexsec.analysis.stress import StressResult
from cablexsec.api import MkappaResult, StiffnessResult
from cablexsec.cables.schema import CableData
from cablexsec.exceptions import CableInputError
from cablexsec.io.series import LoadCaseSeries
from cablexsec.materials.sn_curves import SNcurve, SWTLifeConstants
from cablexsec.mechanics.axisymmetric import AxisymmetricResult
from cablexsec.mechanics.concentric_section import (
    ConcentricSectionResult,
    ConcentricTangent,
)
from cablexsec.mechanics.contact_qp import ContactQPResult
from cablexsec.mechanics.papailiou import BendingWork
from cablexsec.mechanics.section_state import LayerNormalState, SectionState
from cablexsec.mechanics.strand_slip import StrandSlipParameters
from cablexsec.mechanics.thermal import ThermalResult
from cablexsec.provenance import Provenance

_TYPES: dict[str, Any] = {
    cls.__name__: cls
    for cls in (
        StiffnessResult,
        StressResult,
        MkappaResult,
        FatigueResult,
        LoadCaseSeries,
        BendingWork,
        Provenance,
        SNcurve,
        SWTLifeConstants,
        AxisymmetricResult,
        ConcentricSectionResult,
        ConcentricTangent,
        ContactQPResult,
        LayerNormalState,
        SectionState,
        # Project format version 2 (21 October 2026).
        ThermalResult,
        CapacityResult,
        CoilingResult,
        CoilingLayerResult,
        ConductorFatigueResult,
        StrandSlipParameters,
        BendingLawExport,
        IwanBendingLaw,
    )
}


@cache
def _annotations(cls: type[Any]) -> dict[str, Any]:
    return get_type_hints(cls)


def _matches(value: Any, annotation: Any) -> bool:
    if annotation is Any:
        return True
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, types.UnionType):
        return any(_matches(value, option) for option in args)
    if origin is Literal:
        return any(type(value) is type(option) and value == option for option in args)
    if origin in (dict, Mapping):
        return isinstance(value, dict) and all(
            _matches(k, args[0]) and _matches(v, args[1]) for k, v in value.items()
        )
    if origin is tuple:
        return isinstance(value, tuple) and all(_matches(v, args[0]) for v in value)
    if origin is list:
        return isinstance(value, list) and all(_matches(v, args[0]) for v in value)
    if origin is np.ndarray:
        return isinstance(value, np.ndarray) and value.dtype == np.dtype(
            get_args(args[1])[0]
        )
    if annotation is float:
        return type(value) in (int, float)
    if annotation in (str, int, bool):
        return type(value) is annotation
    return isinstance(value, annotation)


def pack(value: Any) -> Any:
    """Encode only explicit data types; never serialize executable objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else {"float": str(float(value))}
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "fib":
            raise CableInputError("unsupported project array dtype")
        return {
            "array": pack(value.ravel().tolist()),
            "dtype": value.dtype.str,
            "shape": list(value.shape),
        }
    if isinstance(value, CableData):
        return {"model": "CableData", "fields": value.model_dump(mode="json")}
    if (
        _TYPES.get(type(value).__name__) is type(value)
        and is_dataclass(value)
        and not isinstance(value, type)
    ):
        return {
            "type": type(value).__name__,
            "fields": {f.name: pack(getattr(value, f.name)) for f in fields(value)},
        }
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {"mapping": {k: pack(v) for k, v in value.items()}}
    if isinstance(value, (list, tuple)):
        return {
            "tuple" if isinstance(value, tuple) else "list": [pack(v) for v in value]
        }
    raise CableInputError(f"unsupported project value: {type(value).__name__}")


def unpack(value: Any) -> Any:
    """Decode a fixed type allowlist, with no dynamic imports or pickle."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if not isinstance(value, dict):
        raise CableInputError("invalid encoded project value")
    keys = set(value)
    if keys == {"model", "fields"} and value["model"] == "CableData":
        return CableData.model_validate(value["fields"])
    if keys == {"float"} and value["float"] in ("nan", "inf", "-inf"):
        return float(value["float"])
    if keys == {"mapping"} and isinstance(value["mapping"], dict):
        return {k: unpack(v) for k, v in value["mapping"].items()}
    if keys in ({"tuple"}, {"list"}):
        kind = next(iter(keys))
        if isinstance(value[kind], list):
            items = [unpack(v) for v in value[kind]]
            return tuple(items) if kind == "tuple" else items
    if keys == {"array", "dtype", "shape"}:
        try:
            dtype = np.dtype(value["dtype"])
            if dtype.kind not in "fib" or not isinstance(value["shape"], list):
                raise ValueError("unsupported dtype or shape")
            if any(type(n) is not int or n < 0 for n in value["shape"]):
                raise ValueError("invalid array shape")
            array = np.asarray(unpack(value["array"]), dtype=dtype)
            if array.ndim != 1 or array.size != math.prod(value["shape"]):
                raise ValueError("array shape does not match data")
            return array.reshape(value["shape"])
        except (TypeError, ValueError, OverflowError) as error:
            raise CableInputError("invalid project array") from error
    if keys == {"type", "fields"} and isinstance(value["type"], str):
        cls = _TYPES.get(value["type"])
        payload = value["fields"]
        if (
            cls is not None
            and isinstance(payload, dict)
            and set(payload) == {f.name for f in fields(cls)}
        ):
            decoded = {k: unpack(v) for k, v in payload.items()}
            for name, annotation in _annotations(cls).items():
                if not _matches(decoded[name], annotation):
                    raise CableInputError(
                        f"invalid project field {cls.__name__}.{name}"
                    )
            return cls(**decoded)
    raise CableInputError("unknown or malformed project value")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise CableInputError(f"duplicate project key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise CableInputError(f"nonstandard JSON constant: {value}")


def parse(text: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    except (ValueError, RecursionError) as error:
        raise CableInputError(f"invalid project JSON: {error}") from error
