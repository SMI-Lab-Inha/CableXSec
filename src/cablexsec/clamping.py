# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Manufacturing clamping state of a cable: resolution, report and warning.

The radial contact pressure that the armour carries at zero applied tension
(armour winding stress, extrusion or sheath pressure) is not predicted by
CableXSec: it is an input. It governs every friction-dependent result: the
bending hysteresis, the armour friction stress and its fatigue, the fretting
contact loads and the change of all of these with temperature. Two public
cases show its size: the measured Ménard and Cartraud (2023) loops need an
initial pressure of 0.2-0.3 MPa (0.22 MPa identified with the full core and
filler accounting, 0.3 MPa calibrated by the source), and the cold-to-rated damage
factor of the 66 kV template falls from 115 to about 20 and 2 with 40 and
100 MPa armour winding stress (doc/template_66kv.md Section 4.2).

Where the state is declared
---------------------------
* **Per helical layer**, ``manufacturing_clamping`` of
  :class:`~cablexsec.cables.schema.HelicalLayer`: either a retained armour
  winding stress (``armour_winding_stress_Pa``) or the layer's radial
  pressure jump ``P0`` (``initial_contact_pressure_Pa``), with ``basis`` and
  ``source``. The legacy keys ``winding_stress_Pa`` and ``pretension_Pa``
  still load and carry the same value; a layer that has only a legacy value
  is reported with basis ``"undocumented"``.
* **Per cable**, ``initial_state``: a uniform pressure applied to the
  assembled cable before the load history, with an identification class
  (for example the published calibration of [MenardCartraud2023]).

Every contact route reads the same values: the bounded chain, the
concentric ``qp`` route and the solved radial state (where both layer inputs
become residual wire stresses identified at the reference state), and
through them ``compute_mkappa``, the fatigue, fretting and rig-test paths and
the thermal coupling.

:func:`clamping_state` resolves the declaration; :func:`warn_if_unclamped`
issues :class:`~cablexsec.provenance.MissingClampingStateWarning` when a
friction-dependent result rests on an undeclared state or on a zero that is
declared as not published.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

from cablexsec.cables.schema import CableData, CylindricalLayer, HelicalLayer
from cablexsec.exceptions import CableInputError
from cablexsec.provenance import (
    ClampingCalibrationWarning,
    MissingClampingStateWarning,
)

__all__ = [
    "ClampingState",
    "LayerClamping",
    "LayerShrinkage",
    "clamping_state",
    "require_radial_shrinkage_route",
    "warn_if_calibrated_with_armour_only",
    "warn_if_unclamped",
]


@dataclass(frozen=True)
class LayerShrinkage:
    """Declared frozen extrusion shrinkage of one concentric layer."""

    layer: str
    r_inner_m: float
    hoop_strain: float
    basis: str
    source: str

    def describe(self) -> str:
        return (
            f"{self.layer}: extrusion shrinkage {self.hoop_strain * 100:g} % "
            f"({self.basis})"
        )


@dataclass(frozen=True)
class LayerClamping:
    """Resolved manufacturing clamping of one structural helical layer.

    ``kind`` is ``"winding_stress"`` (a wire stress), ``"contact_pressure"``
    (the radial pressure jump ``P0``), ``"both"`` (two legacy values, which
    the contact routes add) or ``"none"``. ``basis`` is the declared basis,
    ``"undocumented"`` for a legacy value without a declaration, or
    ``"missing"`` when nothing is declared.
    """

    layer: str
    kind: str
    winding_stress_Pa: float
    contact_pressure_Pa: float
    basis: str
    source: str

    @property
    def is_zero(self) -> bool:
        return self.winding_stress_Pa == 0.0 and self.contact_pressure_Pa == 0.0

    def describe(self) -> str:
        if self.basis == "missing":
            return f"{self.layer}: not declared"
        parts = []
        if self.kind in ("winding_stress", "both"):
            parts.append(f"winding stress {self.winding_stress_Pa / 1e6:g} MPa")
        if self.kind in ("contact_pressure", "both"):
            parts.append(f"pressure jump {self.contact_pressure_Pa / 1e3:g} kPa")
        return f"{self.layer}: {' + '.join(parts)} ({self.basis})"


@dataclass(frozen=True)
class ClampingState:
    """Manufacturing clamping declared by a cable definition."""

    cable: str
    layers: tuple[LayerClamping, ...]
    initial_external_pressure_Pa: float
    initial_identification: str
    initial_source: str
    shrinkage: tuple[LayerShrinkage, ...] = ()
    #: Structural helical layers inside a layer that declares a shrinkage.
    shrink_covered: frozenset[str] = field(default_factory=frozenset)

    @property
    def missing_layers(self) -> tuple[str, ...]:
        """Structural helical layers with no declaration and no legacy value,
        and no declared extrusion shrinkage of a layer outside them."""
        return tuple(
            x.layer
            for x in self.layers
            if x.basis == "missing" and x.layer not in self.shrink_covered
        )

    @property
    def has_initial_state(self) -> bool:
        """The cable declares an identified initial external pressure."""
        return self.initial_identification != "none"

    @property
    def is_declared(self) -> bool:
        """Every structural helical layer is covered by a declaration.

        An identified cable-level initial state covers all layers.
        """
        return self.has_initial_state or not self.missing_layers

    @property
    def is_zero(self) -> bool:
        """No clamping at zero tension from either declaration."""
        return (
            self.initial_external_pressure_Pa == 0.0
            and all(x.is_zero for x in self.layers)
            and all(x.hoop_strain == 0.0 for x in self.shrinkage)
        )

    def warning_message(self) -> str | None:
        """Text of the warning a friction-dependent analysis issues, or None."""
        if not self.layers:
            return None
        consequence = (
            "friction-dependent results (bending hysteresis, armour friction "
            "stress, fretting contact loads, thermal or tension changes of the "
            "clamping) then rest on tension-induced contact alone: friction "
            "stresses and damages sit at their low end and ratios that change "
            "the clamping at their high end. At low tension the friction "
            "moment, the bending hysteresis and its damping tend to zero, which "
            "a manufactured cable does not show, so a low-tension hysteresis is "
            "unreliable without a declared clamping state"
        )
        if not self.is_declared:
            return (
                f"cable {self.cable!r} declares no manufacturing clamping state "
                f"for {', '.join(self.missing_layers)}; {consequence}. Declare "
                "the manufacturing clamping of each armour layer, an armour "
                "winding stress or an initial contact pressure, or the frozen "
                "extrusion shrinkage of the sheath over it, with its basis and "
                "source (the local analysis guide, Section 4.4)."
            )
        unpublished = [x.layer for x in self.layers if x.basis == "not_published"]
        unpublished += [x.layer for x in self.shrinkage if x.basis == "not_published"]
        if self.is_zero and unpublished:
            return (
                f"cable {self.cable!r} has zero manufacturing clamping because "
                f"it is not published ({', '.join(unpublished)}); {consequence}. "
                "Treat them as upper-bound sensitivities, not cable properties "
                "(the local analysis guide, Section 4.4)."
            )
        return None

    def describe(self) -> str:
        """One-line report of the declared state."""
        parts = [x.describe() for x in self.layers]
        parts += [x.describe() for x in self.shrinkage]
        if self.has_initial_state:
            parts.append(
                f"initial external pressure "
                f"{self.initial_external_pressure_Pa / 1e3:g} kPa "
                f"({self.initial_identification})"
            )
        return "; ".join(parts) if parts else "no helical layers"


def _layer_clamping(layer: HelicalLayer) -> LayerClamping:
    declared = layer.manufacturing_clamping
    winding = float(layer.winding_stress_Pa or 0.0)
    pressure = float(layer.pretension_Pa or 0.0)
    if declared is not None:
        return LayerClamping(
            layer.name,
            declared.kind,
            winding,
            pressure,
            declared.basis,
            declared.source,
        )
    has_w = layer.winding_stress_Pa is not None and winding != 0.0
    has_p = layer.pretension_Pa is not None and pressure != 0.0
    if not (has_w or has_p):
        return LayerClamping(layer.name, "none", 0.0, 0.0, "missing", "")
    kind = (
        "both"
        if has_w and has_p
        else ("winding_stress" if has_w else "contact_pressure")
    )
    return LayerClamping(
        layer.name, kind, winding, pressure, "undocumented", layer.source
    )


def clamping_state(cable: CableData) -> ClampingState:
    """Resolve the manufacturing clamping declared by ``cable``.

    Only structural helical layers are reported: they are the friction and
    fatigue sites. A legacy ``winding_stress_Pa`` or ``pretension_Pa`` value
    without a ``manufacturing_clamping`` block counts as declared with basis
    ``"undocumented"``; a legacy zero counts as missing.
    """
    initial = cable.initial_state
    shrinkage = tuple(
        LayerShrinkage(
            layer.name,
            float(layer.r_inner_m or 0.0),
            float(layer.extrusion_shrinkage.hoop_strain),
            layer.extrusion_shrinkage.basis,
            layer.extrusion_shrinkage.source,
        )
        for layer in cable.layers
        if isinstance(layer, CylindricalLayer) and layer.extrusion_shrinkage is not None
    )
    helical = [layer for layer in cable.helical_layers if layer.structural]
    covered = frozenset(
        layer.name
        for layer in helical
        if any(s.r_inner_m > float(layer.r_helix_m or 0.0) for s in shrinkage)
    )
    return ClampingState(
        cable=cable.cable.name,
        layers=tuple(_layer_clamping(layer) for layer in helical),
        initial_external_pressure_Pa=float(initial.external_pressure_Pa),
        initial_identification=initial.identification,
        initial_source=initial.source,
        shrinkage=shrinkage,
        shrink_covered=covered,
    )


def require_radial_shrinkage_route(
    cable: CableData, route: str, remedy: str = "contact_model 'solved' or 'qp'"
) -> None:
    """Refuse a declared non-zero extrusion shrinkage on the bounded chain.

    The frozen shrinkage of a concentric layer is an interference that only
    a displacement-compatible solution carries: the solved radial state and
    the concentric ``qp`` route. The bounded pressure chain would silently
    drop it.
    """
    shrinking = [
        layer.name
        for layer in cable.layers
        if isinstance(layer, CylindricalLayer)
        and layer.extrusion_shrinkage is not None
        and layer.extrusion_shrinkage.hoop_strain > 0.0
    ]
    if shrinking:
        raise CableInputError(
            f"{', '.join(shrinking)}: a frozen extrusion shrinkage is carried by "
            f"a radial solution, not by the {route}; use {remedy}"
        )


def warn_if_calibrated_with_armour_only(
    cable: CableData, *, stacklevel: int = 2
) -> bool:
    """Warn when core friction is added to a value fitted with the armour alone.

    Issues :class:`~cablexsec.provenance.ClampingCalibrationWarning` when the
    cable's initial pressure or an armour clamping declares
    ``calibration_scope="armour_only"``; returns whether it did. Such a value
    was raised until the armour friction alone matched the measured
    dissipation, so it already holds the share of the other contacts; the
    laid-core stick-slip law, which calls this, adds that share again.
    """
    names = []
    if cable.initial_state.calibration_scope == "armour_only":
        names.append("initial_state")
    names += [
        layer.name
        for layer in cable.helical_layers
        if layer.manufacturing_clamping is not None
        and layer.manufacturing_clamping.calibration_scope == "armour_only"
    ]
    if not names:
        return False
    warnings.warn(
        f"cable {cable.cable.name!r}: {', '.join(names)} was calibrated with "
        "the armour friction alone, so it already holds the friction of the "
        "other contacts; the core stick-slip law adds the cores' share a "
        "second time. "
        "Re-identify the clamping with the friction model in use (the local "
        "analysis guide, Section 4.4).",
        ClampingCalibrationWarning,
        stacklevel=stacklevel + 1,
    )
    return True


def warn_if_unclamped(cable: CableData, *, stacklevel: int = 2) -> ClampingState:
    """Warn when a friction-dependent result rests on no declared clamping.

    Issues :class:`~cablexsec.provenance.MissingClampingStateWarning` when no
    clamping state is declared for a structural armour layer (and the cable
    has no identified initial state), or when the clamping is zero because it
    is declared as not published. Returns the resolved state.
    """
    state = clamping_state(cable)
    message = state.warning_message()
    if message is not None:
        warnings.warn(message, MissingClampingStateWarning, stacklevel=stacklevel + 1)
    return state
