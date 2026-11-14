# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Cable YAML schema, loader, and validator.

The cable YAML file is the single source of truth for cable geometry. No
geometry value is written anywhere else in the library.

This module defines frozen Pydantic v2 models for the five layer types

==================== ============================================
``layer_type``        meaning
==================== ============================================
``helical``           wires wound helically about the cable axis
``helical_stranded``  nested helix: strand within conductor within cable
``cylindrical``       solid annular layer (bedding, filler, sheath)
``cylindrical_offset``offset cylinders at trefoil positions (three cores)
``cylindrical_shell`` thin-wall structural shell (lead / corrugated Al)
==================== ============================================

Withheld geometry
-----------------
A cable definition may write a geometry value that cannot be shared as
``null``. The values are then supplied at load time from a separate override
file (``geometry_override``), which is deep-merged onto the definition by
layer ``name``. A cable whose required geometry is still ``null`` after the
merge is *not* fully specified; query this with
:meth:`CableData.missing_fields`.

All length values are SI metres with the ``_m`` suffix; angles are given in
degrees in the YAML (suffix ``_deg``) and exposed in radians via helper
properties.

Materials
---------
A layer's ``material`` names a material of the library
(:mod:`cablexsec.materials.library`) or of the cable's own optional
``materials:`` block, which maps a key to properties with a value and a
source each::

    materials:
      supplier_hdpe:
        polymer: true
        E_Pa: {value: 9.0e8, source: "supplier data sheet"}
        nu: {value: 0.46, source: "model assumption"}

A cable material with the key of a library material replaces it for that
cable only. The library itself never changes: the physics reads a cable's
materials inside :meth:`CableData.material_scope`, which every public entry
point that takes a cable enters (:func:`uses_cable_materials`).
"""

from __future__ import annotations

import functools
import inspect
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal, ParamSpec, Self, TypeVar

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)

from cablexsec.exceptions import CableLookupError, CableSchemaError
from cablexsec.materials import library
from cablexsec.materials.properties import MATERIAL_PROPERTIES
from cablexsec.mechanics.tendon_section import WireSection

__all__ = [
    "CableData",
    "CableMaterial",
    "CableMeta",
    "CoreStrandRing",
    "CoreLay",
    "CylindricalLayer",
    "CylindricalOffsetLayer",
    "CylindricalShellLayer",
    "ElectricalProperties",
    "EnvironmentProperties",
    "FacePressure",
    "HelicalLayer",
    "HelicalStrandedLayer",
    "InitialStateProperties",
    "Layer",
    "ManufacturingClamping",
    "MaterialValue",
    "PressureIdentification",
    "RADIAL_WATER_BARRIER_MATERIALS",
    "WaterIngressProperties",
    "WireFillWarning",
    "load_cable",
    "material_overlay",
    "scope_entry_points",
    "takes_cable",
    "uses_cable_materials",
]

LayDirection = Literal["left_hand", "right_hand"]

# Absolute zero on the Celsius scale (SI definition of the kelvin).
_ABSOLUTE_ZERO_C = -273.15


class _Frozen(BaseModel):
    """Base config: immutable, reject unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class ElectricalProperties(_Frozen):
    """Optional electrical block on an insulation layer.

    ``vec_exponent_n`` (n0), ``thermal_b_k`` (B), ``vec_synergism_b`` (b) and
    ``field_threshold_kV_mm`` (E0) parameterise the [Mazzanti2007] Eq. 17
    electro-thermal life model. ``mechanical_beta`` parameterises the CableXSec
    mechanical-strain screening extension (NOT part of Mazzanti 2007).
    """

    permittivity_relative: float | None = None
    loss_factor_tan_delta: float | None = None
    vec_exponent_n: float | None = None  # n0 — VEC at T0 [Mazzanti2007] Table I
    thermal_b_k: float | None = None  # B [K] [Mazzanti2007] Table I
    vec_synergism_b: float | None = None  # b — electrical-thermal synergism
    field_threshold_kV_mm: float | None = None  # E0 [kV/mm] [Mazzanti2007] Table I
    mechanical_beta: float | None = None  # CableXSec extension (not Mazzanti)

    @model_validator(mode="after")
    def _physical_values(self) -> ElectricalProperties:
        positive = (
            "permittivity_relative",
            "vec_exponent_n",
            "thermal_b_k",
            "field_threshold_kV_mm",
        )
        nonnegative = ("loss_factor_tan_delta", "vec_synergism_b", "mechanical_beta")
        for name in positive:
            value = getattr(self, name)
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be > 0")
        for name in nonnegative:
            value = getattr(self, name)
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be >= 0")
        return self


class EnvironmentProperties(_Frozen):
    """Thermal boundary of the surroundings for current-rating calculations.

    Holds the external (Robin-type) series thermal resistance ``T4`` applied at
    the cable surface in the IEC 60287 ladder ([IEC60287-1-1] Eq. 1) — soil for
    a buried cable, or the bend-stiffener + water gap for a dynamic cable section
    ([Poradowski2025]). Each value carries its source in the YAML ``source``
    string of the block. ``T4`` is taken verbatim as an input here; CableXSec
    does not yet derive it from the installation geometry.
    """

    ambient_C: float | None = None
    T4_external_Km_W: float | None = None
    frequency_Hz: float = 50.0
    source: str = ""

    @model_validator(mode="after")
    def _physical_values(self) -> EnvironmentProperties:
        if self.ambient_C is not None and self.ambient_C <= _ABSOLUTE_ZERO_C:
            raise ValueError(
                f"ambient_C={self.ambient_C!r} degC is not above absolute zero "
                f"({_ABSOLUTE_ZERO_C} degC)"
            )
        if self.T4_external_Km_W is not None and self.T4_external_Km_W < 0.0:
            raise ValueError("T4_external_Km_W must be >= 0")
        if self.frequency_Hz <= 0.0:
            raise ValueError("frequency_Hz must be > 0")
        return self


#: Identification class of a declared initial pressure. ``"identified"``
#: marks a value fitted by CableXSec to a named measured response;
#: ``"assumed_bound"`` an engineering estimate with a stated bound
#: (``bounds_Pa``), used to bracket a result, never to calibrate it.
PressureIdentification = Literal[
    "none",
    "measured",
    "manufacturing_specification",
    "published_calibration",
    "identified",
    "assumed_bound",
]


class InitialStateProperties(_Frozen):
    """Documented initial contact state used by nonlinear section analyses.

    ``external_pressure_Pa`` represents a uniform pressure applied to the
    assembled cable before the mechanical load history. ``identification``
    makes the class of the value explicit so a calibrated reproduction cannot
    be mistaken for an independent prediction: ``"identified"`` marks a value
    that CableXSec itself fitted to a named measured response (that response
    is then not a prediction; VALIDATION_POLICY Rule 3).
    """

    external_pressure_Pa: float = 0.0
    identification: PressureIdentification = "none"
    source: str = ""
    bounds_Pa: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _physical_values(self) -> InitialStateProperties:
        if self.external_pressure_Pa < 0.0:
            raise ValueError("external_pressure_Pa must be >= 0")
        if self.external_pressure_Pa > 0.0:
            if self.identification == "none":
                raise ValueError(
                    "a non-zero initial external pressure requires an "
                    "identification class"
                )
            if not self.source.strip():
                raise ValueError(
                    "a non-zero initial external pressure requires a source"
                )
        elif self.identification != "none":
            raise ValueError(
                "initial-state identification requires external_pressure_Pa > 0"
            )
        _check_bounds(self.external_pressure_Pa, self.identification, self.bounds_Pa)
        return self


def _check_bounds(
    value: float, identification: str, bounds: tuple[float, float] | None
) -> None:
    """An ``assumed_bound`` value carries its bound and lies inside it."""
    if identification == "assumed_bound" and bounds is None:
        raise ValueError(
            "an assumed_bound pressure requires bounds_Pa = (lower, upper)"
        )
    if bounds is not None:
        lower, upper = bounds
        if not 0.0 <= lower <= upper:
            raise ValueError("bounds_Pa must satisfy 0 <= lower <= upper")
        if not lower <= value <= upper:
            raise ValueError("the pressure must lie inside bounds_Pa")


class FacePressure(_Frozen):
    """Radial pressure on the outer face of a resolved wire screen (gap P1).

    The pressure of the tape, foil or sheath wound or shrunk over a wire
    screen of a power core, at zero cable load [Pa], referred to the outer
    face of the wires. It is an input, not a prediction: ``identification``
    states its class and ``source`` cites it. ``"assumed_bound"`` marks an
    engineering estimate with its bound ``bounds_Pa`` (for example a tape
    winding stress times the tape thickness over the radius); results that
    depend on it must be run at both ends of the bound.
    """

    pressure_Pa: float
    identification: PressureIdentification
    source: str
    bounds_Pa: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _physical_values(self) -> FacePressure:
        if not math.isfinite(self.pressure_Pa) or self.pressure_Pa < 0.0:
            raise ValueError("pressure_Pa must be finite and >= 0")
        if self.identification == "none":
            raise ValueError("a face pressure requires an identification class")
        if not self.source.strip():
            raise ValueError("a face pressure requires a source")
        _check_bounds(self.pressure_Pa, self.identification, self.bounds_Pa)
        return self


class TapeWinding(_Frozen):
    """Helical tape of a cylindrical layer (gap P2).

    A tape of width ``width_m`` (perpendicular to the tape) wound at
    ``lay_angle_deg`` to the layer axis, ``n_tapes`` tapes side by side; the
    tape thickness is the layer thickness and its mid radius the layer's.
    With ``n w / (P sin beta) < 1`` the tapes form an open helix (gaps),
    above one they overlap. ``source`` states where the values come from
    (an assumed value names its bound there or in ``notes``).
    """

    width_m: float
    lay_angle_deg: float
    n_tapes: int = Field(default=1, strict=True)
    lay_direction: LayDirection = "right_hand"
    source: str
    notes: str = ""

    @model_validator(mode="after")
    def _tape_values(self) -> TapeWinding:
        if not math.isfinite(self.width_m) or self.width_m <= 0.0:
            raise ValueError("tape width_m must be > 0")
        if not 0.0 < self.lay_angle_deg < 90.0:
            raise ValueError("tape lay_angle_deg must lie in (0, 90)")
        if self.n_tapes < 1:
            raise ValueError("n_tapes must be >= 1")
        if not self.source.strip():
            raise ValueError("a tape winding requires a source")
        return self


# Materials that form a radial (metallic) water barrier when present as a
# continuous sheath ([CIGRE TB 610] Section 5.4.3: extruded lead, welded or
# corrugated aluminium/copper; a bonded Al-PE laminate, [Worzyk2009] Sec.
# 2.3.2). A wire or tape screen is not a barrier.
RADIAL_WATER_BARRIER_MATERIALS = frozenset(
    {"lead_sheath", "aluminium_corrugated", "aluminium_laminate_foil"}
)


class WaterIngressProperties(_Frozen):
    """Wet or dry design and the resulting state of the armour annulus.

    ``design`` follows [CIGRE TB 610] Section 5.4.3: a *wet* design tolerates
    water migration into the insulation and usually has no radial water
    barrier; a *dry* design keeps the insulation dry with an extruded or welded
    metallic sheath. ``armour_annulus`` states whether seawater reaches the
    armour (``flooded``) or the outer sheath keeps it dry (``sealed``).

    The declaration is checked against the layers but does not change any
    analysis default: hydrostatic clamping (``p_ext_Pa``) and the S-N curve
    remain explicit choices of the caller (doc/local_analysis_guide.md 7.2).
    """

    design: Literal["wet", "dry"]
    armour_annulus: Literal["flooded", "sealed"]
    source: str = ""


ClampingBasis = Literal[
    "not_published",
    "assumed",
    "representative",
    "manufacturing_specification",
    "measured",
    "published_calibration",
]


class ManufacturingClamping(_Frozen):
    """Manufacturing clamping state of one helical (armour) layer.

    The clamping that the layer carries at zero applied tension is an input,
    not a prediction (:mod:`cablexsec.clamping`). Give exactly one of

    * ``armour_winding_stress_Pa``: residual armour wire stress retained from
      stranding (``sigma_install``). The bounded chain converts it into a
      pressure jump by [FeretBournazel1987] Eq. 5; the solved radial state
      carries it as a residual wire stress;
    * ``initial_contact_pressure_Pa``: the radial pressure jump ``P0`` across
      this layer (its own inward line load divided by ``2 pi r_helix``), not a
      face pressure and not ``P1+P2``. Jumps accumulate from the outside in.

    ``basis`` states where the value comes from and ``source`` cites it.
    ``"not_published"`` declares that no value is available and requires a
    zero; analyses that depend on it then warn
    (:class:`~cablexsec.provenance.MissingClampingStateWarning`).
    """

    armour_winding_stress_Pa: float | None = None
    initial_contact_pressure_Pa: float | None = None
    basis: ClampingBasis
    source: str

    @model_validator(mode="after")
    def _one_value(self) -> ManufacturingClamping:
        values = [
            v
            for v in (self.armour_winding_stress_Pa, self.initial_contact_pressure_Pa)
            if v is not None
        ]
        if len(values) != 1:
            raise ValueError(
                "manufacturing_clamping needs exactly one of "
                "armour_winding_stress_Pa and initial_contact_pressure_Pa"
            )
        if values[0] < 0.0:
            raise ValueError("manufacturing clamping must be >= 0")
        if not self.source.strip():
            raise ValueError("manufacturing_clamping requires a source")
        if self.basis == "not_published" and values[0] != 0.0:
            raise ValueError(
                "a not_published manufacturing clamping must be zero; give the "
                "value's basis (assumed, representative, ...) instead"
            )
        return self

    @property
    def kind(self) -> str:
        """``"winding_stress"`` or ``"contact_pressure"``."""
        return (
            "winding_stress"
            if self.armour_winding_stress_Pa is not None
            else "contact_pressure"
        )

    @property
    def value_Pa(self) -> float:
        """The declared winding stress or pressure jump [Pa]."""
        value = (
            self.armour_winding_stress_Pa
            if self.armour_winding_stress_Pa is not None
            else self.initial_contact_pressure_Pa
        )
        assert value is not None
        return float(value)

    @property
    def legacy_fields(self) -> dict[str, float | None]:
        """The equivalent ``winding_stress_Pa`` / ``pretension_Pa`` values."""
        if self.kind == "winding_stress":
            return {"winding_stress_Pa": self.value_Pa, "pretension_Pa": None}
        return {"winding_stress_Pa": None, "pretension_Pa": self.value_Pa}


_LEGACY_CLAMPING_KEYS = ("winding_stress_Pa", "pretension_Pa")


def _is_radial_water_barrier(layer: Any) -> bool:
    return (
        getattr(layer, "layer_type", "") == "cylindrical_shell"
        or getattr(layer, "material", "") in RADIAL_WATER_BARRIER_MATERIALS
    )


class _LayerBase(_Frozen):
    """Fields common to every layer type."""

    name: str
    material: str
    source: str = ""
    notes: str = ""
    structural: bool = True  # False = no bending stiffness, not a fatigue site
    # Stress-concentration factor at a cable termination (end fitting). Applied
    # to counted stress ranges before S-N evaluation at flagged terminations;
    # 1.0 = no termination amplification (field interior).
    termination_scf: float = 1.0

    @model_validator(mode="after")
    def _base_physical_values(self) -> _LayerBase:
        if not self.name.strip() or not self.material.strip():
            raise ValueError("layer name and material must be non-empty")
        if self.termination_scf < 1.0:
            raise ValueError(
                f"termination_scf={self.termination_scf!r} must be >= 1 (a stress "
                "concentration factor; 1 = no amplification at the termination)"
            )
        return self


#: Wire counts are whole numbers: a ``bool``, a string or a float is refused
#: rather than coerced (``True`` would otherwise count as one wire).
_Count = Annotated[int, Field(strict=True)]


def _check_lay_angle(name: str, value: float | None) -> None:
    """A lay angle lies in [0, 90) degrees.

    The handedness of a lay is given by ``lay_direction`` only. A signed angle
    would be read as a handedness by some analyses and ignored by others.
    0 degrees is a straight wire, a limit the analyses handle; 90 degrees
    (hoop) is not a helix.
    """
    if value is not None and not 0.0 <= value < 90.0:
        raise ValueError(
            f"{name}={value!r} must lie in [0, 90) degrees; give the handedness "
            "with lay_direction ('right_hand' or 'left_hand'), not with the sign "
            "of the angle"
        )


class WireFillWarning(UserWarning):
    """Round wires of a helical layer do not fit side by side at their radius.

    The lateral fill ``n d / (2 pi r cos(alpha))`` of round wires exceeds 1:
    neighbouring wires (or turns of a near-hoop wire) would occupy the same
    space. A warning, not yet an error, because two kinds of bundled
    definition exceed 1 on purpose: the pinned ``*_legacy_r2`` copies, which
    reproduce earlier results with their infeasible armour, and a near-hoop
    round wire standing for an interlocked pressure-armour profile by its
    thickness (``saevik2011``). A non-round section that overfills its layer
    is an error.
    """


def _warn_round_fill(
    name: str, n_wires: int, d_wire_m: float, r_helix_m: float, lay_angle_deg: float
) -> None:
    fill = (
        n_wires
        * d_wire_m
        / (2.0 * math.pi * r_helix_m * math.cos(math.radians(lay_angle_deg)))
    )
    if fill > 1.0 + 1e-9:
        warnings.warn(
            f"{name}: {n_wires} round wires of diameter {d_wire_m:g} m do not "
            f"fit side by side at radius {r_helix_m:g} m (lateral fill "
            f"{fill:.4f} > 1); neighbouring wires would overlap",
            WireFillWarning,
            stacklevel=2,
        )


class HelicalLayer(_LayerBase):
    """Wires wound helically about the cable axis (armour)."""

    layer_type: Literal["helical"] = "helical"
    n_wires: _Count | None = None
    # Round wire diameter. For a rectangular or profiled wire it is the
    # diameter of the round wire of the same area, filled in from the section
    # (a given value must agree with it); paths that need only the area keep
    # using it, the contact geometry and the bending of the wire use
    # ``wire_section`` (gap G1).
    d_wire_m: float | None = None
    r_helix_m: float | None = None
    lay_angle_deg: float | None = None
    lay_direction: LayDirection = "right_hand"
    # Wire cross-section (gap G1). "round" (default): ``d_wire_m``.
    # "rectangular": ``wire_width_m`` (in the cylinder surface, normal to the
    # wire) x ``wire_thickness_m`` (radial), optional ``wire_corner_radius_m``.
    # "profiled": the envelope width x thickness for the contact geometry and
    # stress points plus the area, second moments and torsion constant
    # declared from the source (``wire_area_m2``, ``wire_I_normal_m4``,
    # ``wire_I_lateral_m4``, ``wire_J_m4``). See
    # :mod:`cablexsec.mechanics.tendon_section`.
    wire_shape: Literal["round", "rectangular", "profiled"] = "round"
    wire_width_m: float | None = None
    wire_thickness_m: float | None = None
    wire_corner_radius_m: float = 0.0
    wire_area_m2: float | None = None
    wire_I_normal_m4: float | None = None  # noqa: N815
    wire_I_lateral_m4: float | None = None  # noqa: N815
    wire_J_m4: float | None = None  # noqa: N815
    # Declared transverse state of the wires in full slip (gap G1): the
    # fraction ``eta`` of the geodesic transverse slip (0 = loxodrome, the
    # earlier convention; 1 = geodesic). It sets the lateral and normal
    # curvature of the wire's individual bending in the full-slip bending
    # stiffness (cablexsec.mechanics.tendon_bending.curvature_coefficients).
    transverse_slip_fraction: float = 0.0
    # Declared manufacturing clamping state (value, basis, source); see
    # ManufacturingClamping and cablexsec.clamping. When given it sets the
    # equivalent legacy field below, which every contact route reads.
    manufacturing_clamping: ManufacturingClamping | None = None
    # Legacy form of initial_contact_pressure_Pa (still accepted; without a
    # manufacturing_clamping block its basis is reported as "undocumented").
    # Manufacturing contact pretension P0 [Pa]: the radial PRESSURE JUMP across
    # this helical layer (its own inward line load per cable length divided by
    # 2 pi r_helix), NOT a face pressure and NOT P1+P2. Jumps accumulate from
    # the outside in through the bounded radial chain (compute_contact_forces),
    # so the pressure under layer i is the sum of P0 of layers i..outermost;
    # the friction-setting P1+P2 sums the pressures on both faces. Two equal
    # layers of P0 give interface pressures (2 P0, P0) and P1+P2 = (3 P0, P0).
    # Used identically for friction (armour_contact_pressure), the per-wire
    # normal line load and the crossed-wire Hertz load.
    pretension_Pa: float | None = None
    # Legacy form of armour_winding_stress_Pa (still accepted, as above).
    # Residual armour winding/installation stress (sigma_install) locked in during
    # stranding. The single cited manufacturing input that drives the inter-layer
    # contact pressure (Feret-Bournazel chain) and hence the M-κ dissipation when
    # there is no applied tension. A process bound (typically ~5-15% of proof),
    # NOT a fit to a measured W_diss. See compute_winding_contact_pressure.
    winding_stress_Pa: float | None = None
    fretting_sn: str = ""
    bulk_sn: str = ""

    @model_validator(mode="before")
    @classmethod
    def _apply_clamping_declaration(cls, data: Any) -> Any:
        """Set the legacy field from ``manufacturing_clamping`` (one value)."""
        if not isinstance(data, Mapping) or data.get("manufacturing_clamping") is None:
            return data
        declared = data["manufacturing_clamping"]
        if not isinstance(declared, ManufacturingClamping):
            declared = ManufacturingClamping.model_validate(declared)
        out = dict(data)
        out["manufacturing_clamping"] = declared
        for key, value in declared.legacy_fields.items():
            given = data.get(key)
            if given is not None and float(given) != (0.0 if value is None else value):
                raise ValueError(
                    f"{key}={given!r} conflicts with manufacturing_clamping; "
                    "declare the clamping once"
                )
            out[key] = value
        return out

    @model_validator(mode="before")
    @classmethod
    def _fill_equivalent_diameter(cls, data: Any) -> Any:
        """Area-equivalent ``d_wire_m`` of a rectangular or profiled wire."""
        if not isinstance(data, Mapping):
            return data
        shape = data.get("wire_shape", "round")
        if shape == "round":
            extra = [k for k in _NON_ROUND_KEYS if data.get(k) not in (None, 0.0, 0)]
            if extra:
                raise ValueError(
                    f"{', '.join(extra)} need wire_shape 'rectangular' or 'profiled'"
                )
            return data
        section = _section_from_mapping(data)
        if section is None:
            return data
        equivalent = section.equivalent_diameter_m
        given = data.get("d_wire_m")
        out = dict(data)
        if given is None:
            out["d_wire_m"] = equivalent
        elif abs(float(given) - equivalent) > 1e-6 * equivalent:
            raise ValueError(
                f"d_wire_m={given!r} is not the area-equivalent diameter "
                f"{equivalent:.7g} m of the {shape} section; omit it"
            )
        return out

    @model_validator(mode="after")
    def _helical_physical_values(self) -> HelicalLayer:
        if self.wire_shape != "round":
            # Definitions with withheld geometry may leave the dimensions null.
            section = _section_from_mapping(self.__dict__)
            if (
                section is not None
                and self.n_wires
                and self.r_helix_m
                and self.lay_angle_deg is not None
                and 0.0 < self.lay_angle_deg < 90.0
            ):
                fill = section.fill_factor(
                    self.n_wires, self.r_helix_m, math.radians(self.lay_angle_deg)
                )
                if fill > 1.0 + 1e-9:
                    raise ValueError(
                        f"{self.name}: {self.n_wires} wires of width "
                        f"{section.surface_width_m} m overfill the layer "
                        f"(fill {fill:.4f} > 1)"
                    )
        for name in ("n_wires", "d_wire_m", "r_helix_m"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0")
        if not 0.0 <= self.transverse_slip_fraction <= 1.0:
            raise ValueError("transverse_slip_fraction must lie in [0, 1]")
        _check_lay_angle("lay_angle_deg", self.lay_angle_deg)
        if self.r_helix_m is not None and (
            self.wire_shape == "round" or _section_from_mapping(self.__dict__)
        ):
            thickness = (
                self.d_wire_m
                if self.wire_shape == "round"
                else self.wire_section.radial_thickness_m
            )
            if thickness is not None and thickness > 2.0 * self.r_helix_m:
                raise ValueError(
                    f"{self.name}: the wire (radial thickness {thickness:g} m) "
                    f"crosses the cable axis at r_helix_m={self.r_helix_m:g} m"
                )
        if (
            self.wire_shape == "round"
            and self.n_wires
            and self.d_wire_m
            and self.r_helix_m
            and self.lay_angle_deg is not None
        ):
            _warn_round_fill(
                self.name,
                self.n_wires,
                self.d_wire_m,
                self.r_helix_m,
                self.lay_angle_deg,
            )
        for name in ("pretension_Pa", "winding_stress_Pa"):
            value = getattr(self, name)
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be >= 0")
        declared = self.manufacturing_clamping
        if declared is not None and any(
            getattr(self, key) != value for key, value in declared.legacy_fields.items()
        ):
            raise ValueError("manufacturing_clamping and its legacy field disagree")
        return self

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        """Copy; keeps ``manufacturing_clamping`` and its legacy field in step.

        Updating ``manufacturing_clamping`` sets the legacy field (it is
        validated). Updating ``winding_stress_Pa`` or ``pretension_Pa``
        directly, as sensitivity studies do, removes the declaration: the
        copy's value has no documented basis and is reported as
        ``"undocumented"``.
        """
        if update:
            changed = dict(update)
            if changed.get("manufacturing_clamping") is not None:
                declared = changed["manufacturing_clamping"]
                if not isinstance(declared, ManufacturingClamping):
                    declared = ManufacturingClamping.model_validate(declared)
                if any(key in changed for key in _LEGACY_CLAMPING_KEYS):
                    raise ValueError(
                        "update manufacturing_clamping or its legacy fields, not both"
                    )
                changed["manufacturing_clamping"] = declared
                changed.update(declared.legacy_fields)
            elif any(key in changed for key in _LEGACY_CLAMPING_KEYS):
                changed.setdefault("manufacturing_clamping", None)
            update = changed
        return super().model_copy(update=update, deep=deep)

    @property
    def lay_angle_rad(self) -> float | None:
        """Lay angle in radians, or ``None`` if unspecified."""
        return None if self.lay_angle_deg is None else math.radians(self.lay_angle_deg)

    @property
    def wire_section(self) -> WireSection:
        """Cross-section of one wire (``tendon_section.WireSection``)."""
        if self.wire_shape == "round":
            if self.d_wire_m is None:
                raise CableLookupError(f"{self.name}: d_wire_m is not specified")
            return WireSection(shape="round", diameter_m=self.d_wire_m)
        section = _section_from_mapping(self.__dict__)
        if section is None:
            raise CableLookupError(f"{self.name}: wire section is not specified")
        return section

    @property
    def wire_radial_thickness_m(self) -> float:
        """Radial extent of a wire [m]: ``d_wire_m`` or the strip thickness."""
        if self.wire_shape == "round":
            if self.d_wire_m is None:
                raise CableLookupError(f"{self.name}: d_wire_m is not specified")
            return self.d_wire_m
        return self.wire_section.radial_thickness_m

    @property
    def wire_surface_width_m(self) -> float:
        """Width of a wire in the cylinder surface, normal to it [m]."""
        if self.wire_shape == "round":
            if self.d_wire_m is None:
                raise CableLookupError(f"{self.name}: d_wire_m is not specified")
            return self.d_wire_m
        return self.wire_section.surface_width_m


_NON_ROUND_KEYS = (
    "wire_width_m",
    "wire_thickness_m",
    "wire_corner_radius_m",
    "wire_area_m2",
    "wire_I_normal_m4",
    "wire_I_lateral_m4",
    "wire_J_m4",
)


def _section_from_mapping(data: Mapping[str, Any]) -> WireSection | None:
    """The non-round section of a layer mapping, or ``None`` when incomplete."""
    shape = data.get("wire_shape", "round")
    width, thickness = data.get("wire_width_m"), data.get("wire_thickness_m")
    if width is None or thickness is None:
        return None
    radius = float(data.get("wire_corner_radius_m") or 0.0)
    if shape == "rectangular":
        return WireSection(
            shape="rectangular",
            width_m=float(width),
            thickness_m=float(thickness),
            corner_radius_m=radius,
        )
    declared = [
        data.get(k)
        for k in ("wire_area_m2", "wire_I_normal_m4", "wire_I_lateral_m4", "wire_J_m4")
    ]
    values = [float(v) for v in declared if v is not None]
    if len(values) != 4:
        return None
    return WireSection(
        shape="profiled",
        width_m=float(width),
        thickness_m=float(thickness),
        corner_radius_m=radius,
        area_m2=values[0],
        I_normal_m4=values[1],
        I_lateral_m4=values[2],
        J_m4=values[3],
    )


class HelicalStrandedLayer(_LayerBase):
    """Nested helix: strand within conductor bundle within the cable."""

    layer_type: Literal["helical_stranded"] = "helical_stranded"
    # Level 1 — conductor bundle orbiting the cable axis.
    n_conductors: _Count | None = None
    r_cable_m: float | None = None
    lay_angle_cable_deg: float | None = None
    # Level 2 — strand orbiting the conductor axis.
    n_strands: _Count | None = None
    r_strand_m: float | None = None
    d_strand_m: float | None = None
    lay_angle_strand_deg: float | None = None
    fretting_sn: str = ""
    bulk_sn: str = ""

    @model_validator(mode="after")
    def _stranded_physical_values(self) -> HelicalStrandedLayer:
        for name in (
            "n_conductors",
            "r_cable_m",
            "n_strands",
            "r_strand_m",
            "d_strand_m",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0")
        for name in ("lay_angle_cable_deg", "lay_angle_strand_deg"):
            _check_lay_angle(name, getattr(self, name))
        d, r = self.d_strand_m, self.r_strand_m
        if d is not None and r is not None:
            if d > 2.0 * r:
                raise ValueError(
                    f"{self.name}: d_strand_m={d:g} m exceeds the diameter "
                    f"2 r_strand_m={2.0 * r:g} m of the strand ring; the strands "
                    "would cross the conductor axis"
                )
            if self.n_strands and self.lay_angle_strand_deg is not None:
                _warn_round_fill(
                    self.name, self.n_strands, d, r, self.lay_angle_strand_deg
                )
        return self

    @property
    def lay_angle_cable_rad(self) -> float | None:
        if self.lay_angle_cable_deg is None:
            return None
        return math.radians(self.lay_angle_cable_deg)

    @property
    def lay_angle_strand_rad(self) -> float | None:
        if self.lay_angle_strand_deg is None:
            return None
        return math.radians(self.lay_angle_strand_deg)


class CylindricalLayer(_LayerBase):
    """Solid annular layer (bedding, filler, polymer sheath)."""

    layer_type: Literal["cylindrical"] = "cylindrical"
    r_inner_m: float | None = None
    r_outer_m: float | None = None
    # Helical tape winding of the layer (gap P2) and its strain-life constants.
    tape: TapeWinding | None = None
    strain_life: str = ""

    @model_validator(mode="after")
    def _annulus_physical_values(self) -> CylindricalLayer:
        if self.r_inner_m is not None and self.r_inner_m < 0.0:
            raise ValueError("r_inner_m must be >= 0")
        if self.r_outer_m is not None and self.r_outer_m <= 0.0:
            raise ValueError("r_outer_m must be > 0")
        if (
            self.r_inner_m is not None
            and self.r_outer_m is not None
            and self.r_outer_m <= self.r_inner_m
        ):
            raise ValueError("r_outer_m must be greater than r_inner_m")
        return self


class CoreStrandRing(_Frozen):
    """One uniform wire ring about a power-core axis, or its straight centre.

    Different rings retain their own diameter and pitch. These are internal
    wire lays, not the lay of the complete power core about the cable axis.
    A centre wire has zero radius and no pitch; all other rings need a pitch.

    Compacted conductors: ``d_wire_m`` is then the radial thickness of the
    compacted wire (it sets the ring radii, the contact geometry and the
    fibre distance of the wire's own bending), and ``wire_area_m2`` its copper
    area normal to its axis (it sets the axial stiffness and the stress). The
    area may exceed ``pi d^2 / 4`` up to the cell the wire fills: ``d`` times
    the ring's circumferential spacing normal to the wire, ``2 pi r cos(alpha)
    / n`` (``d^2`` for the centre wire). Omitted, the wire is round.
    """

    n_wires: int = Field(gt=0, strict=True)
    d_wire_m: float = Field(gt=0, allow_inf_nan=False)
    r_helix_m: float = Field(ge=0, allow_inf_nan=False)
    pitch_m: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    lay_direction: LayDirection = "right_hand"
    wire_area_m2: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    # Stress concentration of the compaction indentations of this ring's
    # wires (Jiang et al. 2022, Table 1: 1.02-1.03 inner, 1.13-1.20 outer;
    # none on the centre wire, Nasution et al. 2013). Use with a curve
    # corrected for the indentations (nasution_2013_copper_corrected).
    compaction_scf: float = Field(default=1.0, ge=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _geometry(self) -> CoreStrandRing:
        if self.r_helix_m == 0:
            if self.n_wires != 1 or self.pitch_m is not None:
                raise ValueError("a centre wire needs n_wires=1 and no pitch")
            cell = self.d_wire_m**2
        else:
            if self.n_wires < 3 or self.pitch_m is None:
                raise ValueError("a wire ring needs at least three wires and a pitch")
            spacing = 2 * self.r_helix_m * math.sin(math.pi / self.n_wires)
            if self.d_wire_m > spacing * (1 + 1e-10):
                raise ValueError("wire diameter exceeds the ring centre spacing")
            cell = (
                self.d_wire_m
                * 2
                * math.pi
                * self.r_helix_m
                * math.cos(self.lay_angle_rad)
                / self.n_wires
            )
        if self.wire_area_m2 is not None and self.wire_area_m2 > cell * (1 + 1e-10):
            raise ValueError(
                "compacted wire area exceeds the cell it can fill "
                "(thickness times circumferential spacing)"
            )
        return self

    @property
    def lay_angle_rad(self) -> float:
        return math.atan2(2 * math.pi * self.r_helix_m, self.pitch_m or 1.0)

    @property
    def area_m2(self) -> float:
        """Metal area of all wires in this ring, normal to their axes."""
        if self.wire_area_m2 is not None:
            return self.n_wires * self.wire_area_m2
        return self.n_wires * math.pi * self.d_wire_m**2 / 4


class CylindricalOffsetLayer(_LayerBase):
    """Offset cylinders at trefoil positions (three power cores)."""

    layer_type: Literal["cylindrical_offset"] = "cylindrical_offset"
    n_cores: _Count = 3
    r_offset_m: float | None = None
    r_inner_m: float | None = None
    r_outer_m: float | None = None
    # Conductor-specific (concentric-stranded) descriptors.
    conductor_area_mm2: float | None = None
    strand_diameter_mm: float | None = None
    n_strands: _Count | None = None
    strand_lay_angle_deg: float | None = None
    strand_layers: tuple[CoreStrandRing, ...] = ()
    construction: str = ""
    electrical: ElectricalProperties | None = None
    fretting_sn: str = ""
    bulk_sn: str = ""
    # Strain-life constants (cablexsec.materials.strain_life) of a metallic
    # layer of the core (laminate foil, sheath, tape); gap P6.
    strain_life: str = ""
    # Pressure of the tape/foil/sheath over a resolved wire screen (gap P1).
    outer_face_pressure: FacePressure | None = None
    # Helical tape winding of the layer (gap P2).
    tape: TapeWinding | None = None

    @model_validator(mode="after")
    def _offset_physical_values(self) -> CylindricalOffsetLayer:
        if (
            self.outer_face_pressure is not None
            and self.construction != "resolved_wire_screen"
        ):
            raise ValueError(
                "outer_face_pressure applies to a resolved_wire_screen layer"
            )
        if self.n_cores <= 0:
            raise ValueError("n_cores must be > 0")
        if self.r_offset_m is not None and self.r_offset_m < 0.0:
            raise ValueError("r_offset_m must be >= 0")
        for name in (
            "r_outer_m",
            "conductor_area_mm2",
            "strand_diameter_mm",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be > 0")
        if self.r_inner_m is not None and self.r_inner_m < 0.0:
            raise ValueError("r_inner_m must be >= 0")
        if (
            self.r_inner_m is not None
            and self.r_outer_m is not None
            and self.r_outer_m <= self.r_inner_m
        ):
            raise ValueError("r_outer_m must be greater than r_inner_m")
        if self.n_strands is not None and self.n_strands <= 0:
            raise ValueError("n_strands must be > 0")
        if (
            self.strand_lay_angle_deg is not None
            and abs(self.strand_lay_angle_deg) >= 90.0
        ):
            # 0 is a straight (solid) conductor; 90 degrees would carry no
            # axial load and divide the material area by cos(90) = 0.
            raise ValueError("abs(strand_lay_angle_deg) must be < 90")
        if self.strand_layers:
            if (
                self.strand_diameter_mm is not None
                or self.strand_lay_angle_deg is not None
            ):
                raise ValueError(
                    "resolved strand_layers cannot use a single diameter or lay angle"
                )
            if self.n_strands is not None and self.n_strands != sum(
                ring.n_wires for ring in self.strand_layers
            ):
                raise ValueError("n_strands disagrees with resolved strand_layers")
            if self.conductor_area_mm2 is not None and not math.isclose(
                self.conductor_area_mm2,
                sum(ring.area_m2 for ring in self.strand_layers) * 1e6,
                rel_tol=5e-4,  # permit a rounded reported area, not a changed geometry
            ):
                raise ValueError("conductor area disagrees with resolved strand_layers")
            previous_outer = self.r_inner_m or 0.0
            for ring in self.strand_layers:
                inner = max(0.0, ring.r_helix_m - ring.d_wire_m / 2)
                outer = ring.r_helix_m + ring.d_wire_m / 2
                if inner < previous_outer - 1e-12:
                    raise ValueError(
                        "strand_layers overlap or are not radially ordered"
                    )
                if self.r_outer_m is not None and outer > self.r_outer_m + 1e-12:
                    raise ValueError("strand_layers extend outside the component")
                previous_outer = outer
        return self


class CylindricalShellLayer(_LayerBase):
    """Thin-wall structural shell: lead or corrugated-Al sheath."""

    layer_type: Literal["cylindrical_shell"] = "cylindrical_shell"
    r_mid_m: float | None = None
    t_m: float | None = None
    f_corrugation: float = 1.0
    scf: float = 1.0
    fretting_sn: str = ""
    bulk_sn: str = ""
    # Strain-life constants (cablexsec.materials.strain_life); gap P6.
    strain_life: str = ""
    # "" (extruded or corrugated sheath) or "bonded_laminate" (a smooth foil
    # bonded to a polymer sheath: its strain follows the polymer).
    construction: Literal["", "bonded_laminate"] = ""

    @model_validator(mode="after")
    def _shell_physical_values(self) -> CylindricalShellLayer:
        for name in ("r_mid_m", "t_m"):
            value = getattr(self, name)
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be > 0")
        if self.f_corrugation <= 0.0 or self.scf <= 0.0:
            raise ValueError("f_corrugation and scf must be > 0")
        if self.construction == "bonded_laminate" and self.f_corrugation != 1.0:
            raise ValueError("a bonded laminate is smooth: f_corrugation must be 1")
        return self


# Discriminated union over the ``layer_type`` tag.
Layer = Annotated[
    HelicalLayer
    | HelicalStrandedLayer
    | CylindricalLayer
    | CylindricalOffsetLayer
    | CylindricalShellLayer,
    Field(discriminator="layer_type"),
]

_LAYER_ADAPTER: TypeAdapter[Any] = TypeAdapter(Layer)


# --------------------------------------------------------------------------- #
# Cable materials                                                             #
# --------------------------------------------------------------------------- #


class MaterialValue(_Frozen):
    """One property of a cable material: its SI value and where it comes from.

    The source is required; a value without a literature reference says so
    (``"user input"``, ``"model assumption"``).
    """

    value: float
    source: str

    @model_validator(mode="after")
    def _has_source(self) -> MaterialValue:
        if not self.source.strip():
            raise ValueError(
                "every material value needs a source (a reference, "
                "'user input' or 'model assumption')"
            )
        return self


class CableMaterial(_Frozen):
    """A material defined in the cable file (the ``materials:`` block).

    Property names, SI units and admissible ranges are those of
    :data:`cablexsec.materials.properties.MATERIAL_PROPERTIES`, the names the
    material library uses. In YAML a property may state its ``unit``; it
    must then be the SI unit of the name. ``polymer`` applies the polymer
    rules (temperature warnings, relaxation) and ``temperature_law`` names a
    registered ``E(T)`` law, scaled to this material's ``E_Pa``.
    """

    name: str
    description: str = ""
    polymer: bool = False
    temperature_law: str = ""
    E_Pa: MaterialValue | None = None
    E_eff_axial_Pa: MaterialValue | None = None
    E_eff_bending_Pa: MaterialValue | None = None
    E_corrugated_Pa: MaterialValue | None = None
    nu: MaterialValue | None = None
    rho_kg_m3: MaterialValue | None = None
    creep_coeff: MaterialValue | None = None
    creep_exponent: MaterialValue | None = None
    sigma_y_Pa: MaterialValue | None = None
    sigma_UTS_Pa: MaterialValue | None = None
    epsilon_allow: MaterialValue | None = None
    mu_on_hdpe: MaterialValue | None = None
    mu_on_steel: MaterialValue | None = None
    mu_on_cu: MaterialValue | None = None
    alpha_T_K: MaterialValue | None = None
    lambda_W_mK: MaterialValue | None = None
    rho_cp_J_m3K: MaterialValue | None = None
    rho_ohm_m: MaterialValue | None = None
    alpha_R_per_K: MaterialValue | None = None
    epsilon_r: MaterialValue | None = None

    @model_validator(mode="before")
    @classmethod
    def _stated_units(cls, data: Any) -> Any:
        """Check and drop the optional ``unit`` of each property."""
        if not isinstance(data, Mapping):
            return data
        cleaned = dict(data)
        for name, definition in MATERIAL_PROPERTIES.items():
            entry = cleaned.get(name)
            if not isinstance(entry, Mapping) or "unit" not in entry:
                continue
            stated = entry["unit"]
            if stated != definition.unit:
                expected = definition.unit or "no unit (a ratio)"
                raise ValueError(
                    f"{name} is stored in {expected}, not {stated!r}; give the "
                    "value in SI units"
                )
            cleaned[name] = {k: v for k, v in entry.items() if k != "unit"}
        return cleaned

    @model_validator(mode="after")
    def _admissible(self) -> CableMaterial:
        if not self.name.strip() or self.name != self.name.strip():
            raise ValueError(
                "material name must be non-empty, without leading or trailing spaces"
            )
        if self.temperature_law and (
            self.temperature_law not in library.TEMPERATURE_MODULUS_MODELS
        ):
            raise ValueError(
                f"temperature_law {self.temperature_law!r} is not a registered "
                "temperature-modulus law; choose one of "
                + ", ".join(sorted(library.TEMPERATURE_MODULUS_MODELS))
            )
        if self.temperature_law and self.E_Pa is None:
            raise ValueError("a temperature_law scales E_Pa, which is not given")
        problems = [
            problem
            for name, (value, _) in self.entries().items()
            if (problem := MATERIAL_PROPERTIES[name].problem(value)) is not None
        ]
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def entries(self) -> dict[str, tuple[float, str]]:
        """The given properties as ``(value, source)`` pairs, in library form."""
        entries: dict[str, tuple[float, str]] = {}
        for name in MATERIAL_PROPERTIES:
            item: MaterialValue | None = getattr(self, name)
            if item is not None:
                entries[name] = (item.value, item.source)
        return entries

    def yaml_body(self) -> dict[str, Any]:
        """The mapping of this material under its key in the cable YAML."""
        return self.model_dump(mode="json", exclude={"name"}, exclude_defaults=True)


@lru_cache(maxsize=64)
def material_overlay(
    materials: tuple[CableMaterial, ...],
) -> library.MaterialOverlay | None:
    """The library overlay of a cable's materials (``None`` when it has none)."""
    if not materials:
        return None
    return library.MaterialOverlay(
        {material.name: material.entries() for material in materials},
        frozenset(material.name for material in materials if material.polymer),
        {
            material.name: material.temperature_law
            for material in materials
            if material.temperature_law
        },
    )


class CableMeta(_Frozen):
    """The ``cable:`` metadata block."""

    name: str
    description: str = ""
    manufacturer: str = ""
    withheld_note: str = ""
    source_document: str = ""
    voltage_kV: float | None = None
    design_life_years: float | None = None
    DFF: float | None = None
    MBR_installation_m: float | None = None
    MBR_dynamic_m: float | None = None

    @model_validator(mode="after")
    def _physical_values(self) -> CableMeta:
        if not self.name.strip():
            raise ValueError("cable name must be non-empty")
        for name in (
            "voltage_kV",
            "design_life_years",
            "DFF",
            "MBR_installation_m",
            "MBR_dynamic_m",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be > 0")
        return self


class AggregateProperties(_Frozen):
    """Published aggregate properties of the cable (safe to ship).

    Comparison targets, never inputs of an analysis. ``EI_min_tension_kN``
    and ``GJ_tension_kN`` are the tensions at which the source states the
    full-slip (slipping) bending stiffness and the torsional stiffness;
    ``None`` when the source states none. ``MBL_kN`` is a published minimum
    breaking load, with ``MBL_basis`` saying how and where the source states
    it (an approximate value is labelled so wherever it is shown).
    """

    EA_fixed_MN: float | None = None
    EI_min_kNm2: float | None = None
    EI_max_kNm2: float | None = None
    GJ_kNm2: float | None = None
    OD_mm: float | None = None
    mass_kg_m: float | None = None
    EI_min_tension_kN: float | None = None
    GJ_tension_kN: float | None = None
    MBL_kN: float | None = None
    """Published minimum breaking load [kN], a comparison target."""
    MBL_basis: str = ""
    """How the source states the minimum breaking load and where (for an
    approximate value: that it is approximate)."""

    @model_validator(mode="after")
    def _physical_values(self) -> AggregateProperties:
        for name in type(self).model_fields:
            value = getattr(self, name)
            if value is None or isinstance(value, str):
                continue
            if name.endswith("_tension_kN"):
                if value < 0.0:
                    raise ValueError(f"{name} must be >= 0")
            elif value <= 0.0:
                raise ValueError(f"{name} must be > 0")
        return self


# Geometry fields that must be present (non-null) for a layer to count as
# fully specified, keyed by layer_type.
_REQUIRED_GEOMETRY: dict[str, tuple[str, ...]] = {
    "helical": ("n_wires", "d_wire_m", "r_helix_m", "lay_angle_deg"),
    "helical_stranded": (
        "n_conductors",
        "r_cable_m",
        "lay_angle_cable_deg",
        "n_strands",
        "r_strand_m",
        "d_strand_m",
        "lay_angle_strand_deg",
    ),
    "cylindrical": ("r_inner_m", "r_outer_m"),
    "cylindrical_offset": ("r_offset_m", "r_inner_m", "r_outer_m"),
    "cylindrical_shell": ("r_mid_m", "t_m"),
}


# Radial order of the layers (CableData._radially_ordered). Touching layers
# may differ by rounding: a radius may fall short of its neighbour's by this
# much. The power cores use the snug-fit tolerance of the trefoil model
# (trefoil_section.TrefoilBundle.overlap_tolerance, 0.5 % of the radius).
_RADIAL_TOL_ABS_M = 1e-9
_RADIAL_TOL_REL = 1e-6
_CORE_SNUG_FIT = 5e-3


def _radial_tol(radius: float) -> float:
    return _RADIAL_TOL_ABS_M + _RADIAL_TOL_REL * abs(radius)


def _is_power_core(layer: Any) -> bool:
    """An offset (trefoil) power-core layer, radii about the core axis."""
    return (
        isinstance(layer, CylindricalOffsetLayer)
        and layer.r_offset_m is not None
        and not (layer.n_cores == 1 and layer.r_offset_m == 0.0)
    )


def _concentric_extent(layer: Any) -> tuple[float, float] | None:
    """Radial extent about the cable axis used for the layer order.

    A round wire counts by its helix radius alone: its diameter may be the
    area-equivalent diameter of a strip (gap G1, the round variants of the
    flexible-pipe definitions), which can overlap the neighbouring tapes. A
    non-round wire counts by its radial thickness. ``None`` when the layer
    has no concentric extent (power cores, nested helices) or its geometry is
    withheld.
    """
    if isinstance(layer, HelicalLayer):
        r = layer.r_helix_m
        if r is None:
            return None
        if layer.wire_shape == "round":
            return (r, r)
        if _section_from_mapping(layer.__dict__) is None:
            return None
        half = layer.wire_section.radial_thickness_m / 2.0
        return (r - half, r + half)
    if isinstance(layer, CylindricalLayer) or (
        isinstance(layer, CylindricalOffsetLayer)
        and layer.n_cores == 1
        and layer.r_offset_m == 0.0
    ):
        if layer.r_inner_m is None or layer.r_outer_m is None:
            return None
        return (layer.r_inner_m, layer.r_outer_m)
    if isinstance(layer, CylindricalShellLayer):
        if layer.r_mid_m is None or layer.t_m is None:
            return None
        return (layer.r_mid_m - layer.t_m / 2.0, layer.r_mid_m + layer.t_m / 2.0)
    return None


def _core_envelope(layers: Sequence[Any]) -> float | None:
    """Check the power cores; return their envelope radius about the cable axis.

    All power-core layers form one family (one ``n_cores`` and one
    ``r_offset_m``), the layers of a core follow each other from its axis
    outwards without overlapping, and neighbouring cores do not overlap.
    """
    cores = [x for x in layers if _is_power_core(x)]
    if not cores:
        return None
    families = sorted({(x.n_cores, x.r_offset_m) for x in cores})
    if len(families) > 1:
        raise ValueError(
            "the power-core layers must form one core family (one n_cores and "
            "one r_offset_m); found (n_cores, r_offset_m) = "
            + ", ".join(f"({n}, {r:g})" for n, r in families)
        )
    n_cores, r_offset = families[0]
    previous: Any = None
    for layer in cores:
        if layer.r_inner_m is None or layer.r_outer_m is None:
            continue
        if previous is not None and layer.r_inner_m < previous.r_outer_m - _radial_tol(
            previous.r_outer_m
        ):
            raise ValueError(
                f"core layer {layer.name!r} (r_inner_m={layer.r_inner_m:g} m) "
                f"overlaps {previous.name!r} (r_outer_m={previous.r_outer_m:g} m); "
                "list the layers of a core from its axis outwards"
            )
        previous = layer
    radii = [x.r_outer_m for x in cores if x.r_outer_m is not None]
    if not radii:
        return None
    r_core = max(radii)
    if n_cores > 1:
        spacing = 2.0 * r_offset * math.sin(math.pi / n_cores)
        if 2.0 * r_core > spacing * (1.0 + _CORE_SNUG_FIT):
            raise ValueError(
                f"neighbouring power cores overlap: {n_cores} cores of radius "
                f"{r_core:g} m at r_offset_m={r_offset:g} m are only "
                f"{spacing:g} m apart"
            )
    return float(r_offset + r_core)


def _check_radial_order(layers: Sequence[Any]) -> None:
    """Layers are listed from the axis outwards and do not overlap.

    Analyses read the YAML order as the radial order (the shear stack under an
    armour layer is the layers listed before it, [Yun2020] in
    :func:`cablexsec.mechanics.stiffness.xi_for_layer`), so a layer listed out
    of order or overlapping another is refused here rather than mis-assembled.
    """
    envelope = _core_envelope(layers)
    inside = (
        None if envelope is None else envelope * (1.0 - _CORE_SNUG_FIT)
    )  # beside the cores: the interstitial filler only
    previous: tuple[str, float] | None = None
    helices: list[HelicalLayer] = []
    fillers: list[str] = []
    seen_outer: str | None = None
    for layer in layers:
        if _is_power_core(layer) and seen_outer is not None:
            raise ValueError(
                f"power-core layer {layer.name!r} is listed after {seen_outer!r}, "
                "which lies outside the cores; list the layers from the axis "
                "outwards"
            )
        extent = _concentric_extent(layer)
        if extent is None:
            continue
        lo, hi = extent
        if inside is not None:
            if lo < inside:
                if not isinstance(layer, CylindricalLayer):
                    raise ValueError(
                        f"{layer.name!r} (from r = {lo:g} m) lies inside the "
                        f"power-core envelope of radius {envelope:g} m"
                    )
                fillers.append(layer.name)
            elif seen_outer is None:
                seen_outer = layer.name
        if lo < 0.0:
            raise ValueError(f"{layer.name!r} extends below the cable axis")
        if previous is not None and lo < previous[1] - _radial_tol(previous[1]):
            raise ValueError(
                f"{layer.name!r} (from r = {lo:g} m) overlaps or lies inside "
                f"{previous[0]!r} (to r = {previous[1]:g} m); list the layers "
                "from the axis outwards without overlapping"
            )
        previous = (layer.name, hi)
        if isinstance(layer, HelicalLayer):
            helices.append(layer)
    if len(fillers) > 1:
        raise ValueError(
            "only one interstitial filler may lie beside the power cores; "
            f"found {', '.join(map(repr, fillers))}"
        )
    for inner, outer in zip(helices, helices[1:], strict=False):
        r_in, r_out = inner.r_helix_m, outer.r_helix_m
        assert r_in is not None and r_out is not None
        reach_out = inner.d_wire_m if inner.wire_shape == "round" else None
        reach_in = outer.d_wire_m if outer.wire_shape == "round" else None
        if reach_out is not None and r_in + reach_out / 2.0 > r_out:
            raise ValueError(
                f"{inner.name!r}: wires of diameter {reach_out:g} m at r_helix_m="
                f"{r_in:g} m reach past the helix radius {r_out:g} m of "
                f"{outer.name!r}; the wire is larger than its radial room"
            )
        if reach_in is not None and r_out - reach_in / 2.0 < r_in:
            raise ValueError(
                f"{outer.name!r}: wires of diameter {reach_in:g} m at r_helix_m="
                f"{r_out:g} m reach inside the helix radius {r_in:g} m of "
                f"{inner.name!r}; the wire is larger than its radial room"
            )


class CoreLay(_Frozen):
    """Common assembly lay of power cores, distinct from internal wire lay.

    None pitch is a withheld-value placeholder, not a straight-core definition.
    Omit the whole block to select the aligned-axis model explicitly.
    """

    pitch_m: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    lay_direction: LayDirection = "right_hand"
    source: str = ""


class CableData(_Frozen):
    """A fully parsed cable definition."""

    cable: CableMeta
    layers: tuple[Layer, ...]
    aggregate_properties: AggregateProperties = AggregateProperties()
    environment: EnvironmentProperties = EnvironmentProperties()
    initial_state: InitialStateProperties = InitialStateProperties()
    core_lay: CoreLay | None = None
    water_ingress: WaterIngressProperties | None = None
    materials: tuple[CableMaterial, ...] = ()

    @field_validator("materials", mode="before")
    @classmethod
    def _materials_by_key(cls, value: Any) -> Any:
        """Accept the YAML form: a mapping from material key to its properties."""
        if value is None:
            return ()
        if isinstance(value, Mapping):
            items = []
            for key, body in value.items():
                if not isinstance(body, Mapping):
                    raise ValueError(
                        f"material {key!r} must be a mapping of its properties"
                    )
                if "name" in body:
                    raise ValueError(
                        f"material {key!r}: the key is the name; remove 'name'"
                    )
                items.append({"name": key, **body})
            return tuple(items)
        return value

    @field_serializer("materials")
    def _materials_as_mapping(
        self, materials: tuple[CableMaterial, ...]
    ) -> dict[str, Any]:
        return {material.name: material.yaml_body() for material in materials}

    @model_serializer(mode="wrap")
    def _without_empty_materials(
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ) -> Any:
        # A cable without materials dumps as before the block existed, so
        # digests of existing cable definitions do not change.
        data = handler(self)
        if not self.materials and isinstance(data, dict):
            data.pop("materials", None)
        return data

    @model_validator(mode="after")
    def _unique_layer_names(self) -> CableData:
        names = [layer.name for layer in self.layers]
        if len(names) != len(set(names)):
            raise ValueError("layer names must be unique within a cable")
        keys = [material.name for material in self.materials]
        if len(keys) != len(set(keys)):
            raise ValueError("material names must be unique within a cable")
        return self

    @model_validator(mode="after")
    def _radially_ordered(self) -> CableData:
        _check_radial_order(self.layers)
        return self

    @model_validator(mode="after")
    def _water_ingress_consistent(self) -> CableData:
        water = self.water_ingress
        if water is None:
            return self
        barriers = [x.name for x in self.layers if _is_radial_water_barrier(x)]
        if water.design == "wet" and barriers:
            raise ValueError(
                "a wet design has no radial metallic water barrier; found "
                + ", ".join(barriers)
            )
        if water.design == "dry" and not barriers:
            raise ValueError(
                "a dry design needs a radial metallic water barrier layer "
                "(cylindrical_shell or lead/aluminium sheath)"
            )
        return self

    def material_scope(self) -> AbstractContextManager[None]:
        """Context in which the physics reads this cable's materials.

        Inside it a material key resolves in the cable's ``materials`` block
        first and in the library second; a cable without materials makes
        the library alone answer, also inside the scope of another cable.
        The scope is local to the current thread (and task).
        """
        return library.material_scope(material_overlay(self.materials))

    def get_material(self, name: str) -> CableMaterial:
        """Return the cable material ``name``."""
        for material in self.materials:
            if material.name == name:
                return material
        raise CableLookupError(
            f"No cable material named {name!r}. Materials of the cable: "
            f"{', '.join(m.name for m in self.materials) or 'none'}."
        )

    def get_layer(self, name: str) -> Layer:
        """Return the layer with the given ``name``."""
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise CableLookupError(
            f"No layer named {name!r}. "
            f"Layers: {', '.join(layer.name for layer in self.layers)}."
        )

    @property
    def helical_layers(self) -> tuple[HelicalLayer, ...]:
        """All ``helical`` layers, in YAML order."""
        return tuple(x for x in self.layers if isinstance(x, HelicalLayer))

    @property
    def structural_layers(self) -> tuple[Layer, ...]:
        """Layers that bear bending and are fatigue sites (``structural`` True).

        Non-structural layers (loose tape screens, interstitial fillers) carry
        no bending stress and are never fatigue sites anywhere in the pipeline.
        """
        return tuple(x for x in self.layers if x.structural)

    def missing_fields(self) -> list[str]:
        """Return ``"layer_name.field"`` for every still-null required value.

        An empty list means the cable geometry is fully specified.
        """
        missing: list[str] = []
        if self.core_lay is not None and self.core_lay.pitch_m is None:
            missing.append("core_lay.pitch_m")
        for layer in self.layers:
            for field in _REQUIRED_GEOMETRY.get(layer.layer_type, ()):
                if getattr(layer, field) is None:
                    missing.append(f"{layer.name}.{field}")
            if (
                isinstance(layer, CylindricalOffsetLayer)
                and layer.construction
                in {"resolved_concentric_stranded", "resolved_wire_screen"}
                and not layer.strand_layers
            ):
                missing.append(f"{layer.name}.strand_layers")
        return missing

    def is_fully_specified(self) -> bool:
        """True if no required geometry value is still ``null`` (withheld)."""
        return not self.missing_fields()


_OVERRIDE_KEYS = frozenset(
    {"cable", "layers", "aggregate_properties", "initial_state", "core_lay"}
)
_OVERRIDE_MAPPINGS = ("aggregate_properties", "initial_state")


def _deep_merge_layers(
    raw: dict[str, Any], override: dict[str, Any], *, label: str = "geometry override"
) -> dict[str, Any]:
    """Overlay withheld values from ``override`` onto raw cable data.

    The override file mirrors the cable file's ``layers`` list (matched by
    ``name``); for each matching layer, override keys replace template values.
    Top-level ``aggregate_properties`` and ``initial_state`` mappings are
    merged too, ``core_lay`` replaces the definition's, and ``cable`` may hold
    only the ``name`` that must match the definition. The override never
    introduces new layers.

    Anything the merge would otherwise drop is refused with
    :class:`CableSchemaError`: an unknown top-level key (a misspelt ``layer:``
    included), an override layer that is not a mapping, has no name, names no
    layer of the definition, is given twice or changes the ``layer_type``, and
    an override that supplies nothing.
    """
    unknown = sorted(map(str, set(override) - _OVERRIDE_KEYS))
    if unknown:
        raise CableSchemaError(
            f"{label}: unknown top-level key(s) {', '.join(unknown)}; an override "
            f"may hold only {', '.join(sorted(_OVERRIDE_KEYS))}"
        )
    raw_cable = raw.get("cable")
    raw_identity = raw_cable.get("name") if isinstance(raw_cable, Mapping) else None
    ov_cable = override.get("cable")
    if ov_cable is None:
        ov_cable = {}
    if not isinstance(ov_cable, Mapping) or set(ov_cable) - {"name"}:
        raise CableSchemaError(
            f"{label}: the override's 'cable' block may hold only the 'name' of "
            "the target cable"
        )
    override_identity = ov_cable.get("name")
    if override_identity is not None and override_identity != raw_identity:
        raise CableSchemaError(
            f"{label}: geometry override cable identity does not match the target "
            f"cable: {override_identity!r} != {raw_identity!r}"
        )
    for key in _OVERRIDE_MAPPINGS:
        if key in override and not isinstance(override[key], Mapping):
            raise CableSchemaError(f"{label}: {key!r} must be a mapping")
    if override.get("core_lay") is not None and not isinstance(
        override["core_lay"], Mapping
    ):
        raise CableSchemaError(f"{label}: 'core_lay' must be a mapping")
    ov_layers = override.get("layers") or []
    if not ov_layers and not any(
        key in override for key in (*_OVERRIDE_MAPPINGS, "core_lay")
    ):
        raise CableSchemaError(
            f"{label}: the override supplies no values (no layers, "
            "aggregate_properties, initial_state or core_lay)"
        )
    by_name = {
        layer["name"]: layer
        for layer in raw.get("layers", [])
        if isinstance(layer, dict) and "name" in layer
    }
    merged: set[str] = set()
    for index, ov_layer in enumerate(ov_layers):
        if not isinstance(ov_layer, dict):
            raise CableSchemaError(
                f"{label}: layer entry {index} must be a mapping with a 'name', "
                f"not {type(ov_layer).__name__}"
            )
        name = ov_layer.get("name")
        if not isinstance(name, str):
            raise CableSchemaError(
                f"{label}: layer entry {index} has no 'name' naming the layer it "
                "overrides"
            )
        target = by_name.get(name)
        if target is None:
            raise CableSchemaError(
                f"{label}: layer {name!r} is not a layer of the cable "
                f"(layers: {', '.join(map(str, by_name))}); an override never "
                "adds layers"
            )
        if name in merged:
            raise CableSchemaError(f"{label}: layer {name!r} is given twice")
        merged.add(name)
        given_type = ov_layer.get("layer_type")
        if given_type is not None and given_type != target.get("layer_type"):
            raise CableSchemaError(
                f"{label}: layer {name!r} has layer_type "
                f"{target.get('layer_type')!r} in the definition, not {given_type!r}"
            )
        if (
            any(key in ov_layer for key in _LEGACY_CLAMPING_KEYS)
            and "manufacturing_clamping" not in ov_layer
        ):
            # A legacy clamping value in the override replaces the template's
            # declaration (it would otherwise conflict with it).
            target.pop("manufacturing_clamping", None)
        for key, value in ov_layer.items():
            if key in {"name", "layer_type"}:
                continue
            if (
                key == "electrical"
                and isinstance(value, dict)
                and isinstance(target.get("electrical"), dict)
            ):
                target["electrical"].update(value)
            else:
                target[key] = value
        if ov_layer.get("manufacturing_clamping") is not None:
            for key in _LEGACY_CLAMPING_KEYS:
                if key not in ov_layer:
                    target.pop(key, None)
    for key in _OVERRIDE_MAPPINGS:
        if key in override:
            base = raw.get(key)
            raw[key] = {**(base if isinstance(base, Mapping) else {}), **override[key]}
    if "core_lay" in override:
        raw["core_lay"] = override["core_lay"]
    return raw


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Parse a cable YAML file whose root is a mapping (empty file: ``{}``)."""
    try:
        data = yaml.safe_load(path.read_bytes().decode("utf-8"))
    except UnicodeDecodeError:
        raise CableSchemaError(f"{path.name}: not a UTF-8 text file") from None
    except yaml.YAMLError as exc:
        raise CableSchemaError(f"{path.name}: invalid YAML: {exc}") from None
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise CableSchemaError(
            f"{path.name}: the YAML root must be a mapping, not {type(data).__name__}"
        )
    if not isinstance(data.get("layers", []), list):
        raise CableSchemaError(f"{path.name}: 'layers' must be a list of layers")
    return data


_TOP_LEVEL_KEYS = frozenset(
    {
        "cable",
        "layers",
        "aggregate_properties",
        "environment",
        "initial_state",
        "core_lay",
        "water_ingress",
        "materials",
    }
)


def load_cable(
    yaml_path: str | Path,
    *,
    geometry_override: str | Path | None = None,
) -> CableData:
    """Load and validate a cable definition from YAML.

    Parameters
    ----------
    yaml_path : str or pathlib.Path
        Path to the cable YAML file (e.g. ``cablexsec/cables/menard2023.yaml``).
    geometry_override : str, pathlib.Path or None, optional
        Path to a separate YAML file supplying withheld geometry values,
        deep-merged by layer ``name``. If the path does not exist it is
        ignored (the definition's ``null`` placeholders remain). ``None``
        (default) applies no override. An existing override must be a
        readable file that supplies values; what it supplies must reach the
        cable (see :func:`_deep_merge_layers`).

    Returns
    -------
    CableData
        The schema-checked, immutable cable definition.

    Raises
    ------
    FileNotFoundError
        If ``yaml_path`` does not exist.
    CableSchemaError
        If a file is not UTF-8 YAML with a mapping at its root and a list of
        layers, or if it has a top-level key other than ``cable``, ``layers``,
        ``aggregate_properties``, ``environment``, ``initial_state``,
        ``core_lay``, ``water_ingress`` and ``materials``; if an existing
        geometry override is not a readable file, supplies nothing, holds a
        key other than ``cable`` (``name`` only), ``layers``,
        ``aggregate_properties``, ``initial_state`` and ``core_lay``, names a
        layer the cable does not have, or does not match the cable's name.
    pydantic.ValidationError
        If the merged data violates the schema (out-of-range or inconsistent
        values, layers not listed from the axis outwards, overlapping layers).
    """
    path = Path(yaml_path)
    if not path.is_file():
        raise FileNotFoundError(f"Cable YAML not found: {path}")
    raw = _read_yaml_mapping(path)

    if geometry_override is not None:
        ov_path = Path(geometry_override)
        # A missing override is the documented withheld-geometry case: the
        # null placeholders remain and missing_fields() lists them. A path
        # that exists but cannot be read, or supplies nothing, is an error.
        if ov_path.exists():
            if not ov_path.is_file():
                raise CableSchemaError(
                    f"geometry override {ov_path} exists but is not a file"
                )
            try:
                override = _read_yaml_mapping(ov_path)
            except OSError as exc:
                raise CableSchemaError(
                    f"geometry override {ov_path} cannot be read: {exc}"
                ) from None
            raw = _deep_merge_layers(
                raw, override, label=f"geometry override {ov_path.name}"
            )

    # A misspelt block would otherwise be dropped silently and its values left
    # at their defaults; unknown keys inside the blocks are already refused.
    unknown = sorted(set(raw) - _TOP_LEVEL_KEYS)
    if unknown:
        raise CableSchemaError(
            f"{path.name}: unknown top-level key(s) {', '.join(map(str, unknown))}; "
            f"allowed: {', '.join(sorted(_TOP_LEVEL_KEYS))}"
        )

    layers = tuple(
        _LAYER_ADAPTER.validate_python(item) for item in raw.get("layers", [])
    )
    return CableData(
        cable=CableMeta.model_validate(raw.get("cable", {})),
        layers=layers,
        aggregate_properties=AggregateProperties.model_validate(
            raw.get("aggregate_properties", {})
        ),
        environment=EnvironmentProperties.model_validate(raw.get("environment", {})),
        initial_state=InitialStateProperties.model_validate(
            raw.get("initial_state", {})
        ),
        core_lay=(
            CoreLay.model_validate(raw["core_lay"])
            if raw.get("core_lay") is not None
            else None
        ),
        water_ingress=(
            WaterIngressProperties.model_validate(raw["water_ingress"])
            if raw.get("water_ingress") is not None
            else None
        ),
        materials=raw.get("materials") or (),
    )


P = ParamSpec("P")
R = TypeVar("R")


def _cable_argument(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
    for value in (*args, *kwargs.values()):
        if isinstance(value, CableData):
            return value
    return None


def uses_cable_materials(func: Callable[P, R]) -> Callable[P, R]:
    """Run ``func`` in the material scope of its first cable argument.

    A public entry point that takes a :class:`CableData` is wrapped with this
    decorator, so its physics reads the cable's own materials whichever
    thread calls it. Without a cable argument ``func`` runs unchanged.
    """
    if getattr(func, "__uses_cable_materials__", False):
        return func

    @functools.wraps(func)
    def run(*args: P.args, **kwargs: P.kwargs) -> R:
        cable = _cable_argument(args, kwargs)
        if cable is None:
            return func(*args, **kwargs)
        with cable.material_scope():
            return func(*args, **kwargs)

    run.__uses_cable_materials__ = True  # type: ignore[attr-defined]
    return run


def takes_cable(func: Callable[..., Any]) -> bool:
    """Whether a parameter of ``func`` is annotated with :class:`CableData`."""
    try:
        parameters = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return False
    return any("CableData" in str(p.annotation) for p in parameters)


def scope_entry_points(namespace: dict[str, Any], names: Sequence[str]) -> None:
    """Wrap every function of ``names`` in ``namespace`` that takes a cable.

    Used by :mod:`cablexsec` and :mod:`cablexsec.api` on their public
    functions (see :func:`uses_cable_materials`).
    """
    for name in names:
        value = namespace.get(name)
        if inspect.isfunction(value) and takes_cable(value):
            namespace[name] = uses_cable_materials(value)
