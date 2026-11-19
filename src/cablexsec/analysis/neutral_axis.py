# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Neutral axes of the cable section under a biaxial curvature history.

Under biaxial bending with stick-slip of the helical armour, "neutral axis"
has more than one meaning. This module reports, at every sample of a history,
the quantities below. Conventions are those of the load histories
(:mod:`cablexsec.mechanics.section_conventions`): the curvature is the
axial-strain gradient ``b = (kx, ky)`` [1/m], so that a section point
``p = (x, y)`` has the axial strain ``eps0 + kx x + ky y``; angles are
measured from the section x axis towards y. Global-model, CableDyn and RIFLEX
rotation curvatures are converted to ``b`` by the readers.

1. **Bending direction** ``theta = atan2(ky, kx)``: the direction of the
   tensile extreme fibre. A global program's rotation-curvature vector
   ``k = (b_y, -b_x)`` lies along the strain neutral axis, 90 degrees
   clockwise of ``theta``.
2. **Strain neutral axis**: the line ``eps0 + b . p = 0`` of zero total axial
   strain, perpendicular to ``b`` at the signed distance
   ``e = -eps0 / |b|`` from the cable axis, measured along ``theta``
   (positive towards the tensile fibre; tension moves the line towards the
   compressed side). It is the plane-section strain of the beam: bonded
   components follow it, slipping wires do not (item 3).
3. **Stress neutral axis of an armour layer**: the diameter perpendicular to
   the first circular harmonic of the wire bending stress round the layer,

   .. math::  \mathbf R = \frac{2}{n}\sum_k \sigma_k (\cos\varphi_k,
              \sin\varphi_k), \qquad \psi_L = \operatorname{atan2}(R_y, R_x),

   with ``sigma_k`` the stick-plus-friction stress of wire ``k`` from the
   slip law of the fatigue route ([Papailiou1997] cap, [Kim2020] friction
   stress), the part of the wire stress that bending causes at the wire
   centroid. The layer's bending moment is ``A cos(alpha) r (n/2) R``, so
   ``psi_L`` is also the direction of the layer's moment. Where the stress
   is antisymmetric about a diameter (stick, first loading from a straight
   section, full slip under steady rotation) the stress changes sign on the
   axis ``psi_L +- 90 deg``; after a reversal it can change sign more than
   twice round the layer, and the axis is then the least-squares diameter.
   The number of sign changes is returned so that this can be seen.
4. **Moment direction**: the direction ``psi`` of the section moment
   ``m = sum_L m_L + EI_min b`` conjugate to ``b`` (the moment-curvature law
   of :func:`cablexsec.compute_mkappa`; with ``core_friction`` it adds the
   laid-core law of :func:`cablexsec.compute_mkappa`). The angle
   from the curvature to a moment, ``delta = psi - theta`` wrapped to
   (-180, 180] deg, is positive anticlockwise. Under a curvature that turns
   anticlockwise at constant magnitude the moment leads: the work per unit
   rotation, ``|b| |m| sin(delta)``, is the friction dissipation and cannot be
   negative. In full slip the friction moment is normal to the curvature
   ([Kimball1925]).
5. **Stick zone of an armour layer** (direction-tied law only): the wires
   whose bending stress is below the cap of the slip law, i.e. whose friction
   is not fully mobilised. Slip starts at the neutral axis, where the cap
   vanishes ([Papailiou1997]; [KraincanicKebadze2001]), and the stick zones
   shrink towards the extreme fibres; a reversal returns every wire to stick.

Directions are not defined at zero curvature, moment or stress: each is set
to NaN where its magnitude is below ``direction_share`` times its largest
value over the record, and the strain neutral axis also where it misses the
section (``|e|`` above the section radius).

The direction-tied law ties the cap to the current bending direction; under
non-proportional curvature the stress neutral axis depends on that
assumption. A rate-anchored law ([Dong2015]: friction integrated from the
point of zero bending-strain rate) or the distributed law (stick zones along
the helix) can give another axis; ``friction_model`` selects the laws of the
fatigue route (``"auto"``: the direction-tied law on a planar path, the
distributed law on every other path where it applies, as in
:func:`cablexsec.compute_fatigue`), and the stick zone is given for the
direction-tied law only. The laid-core law of ``compute_neutral_axes(...,
core_friction=...)`` takes the same model unless ``core_friction_model`` says
otherwise (:func:`~cablexsec.mechanics.core_friction.core_friction_history`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from cablexsec.exceptions import CableInputError
from cablexsec.provenance import Provenance

if TYPE_CHECKING:
    from cablexsec.cables.schema import CableData
    from cablexsec.mechanics.core_friction import (
        CoreFrictionModelName,
        CoreFrictionSpec,
        CoreLoadMode,
    )
    from cablexsec.mechanics.mkappa import HelicalLayerSpec

__all__ = [
    "ArmourAxes",
    "NeutralAxisResult",
    "angle_from",
    "compute_neutral_axes",
    "neutral_axes",
    "sign_changes",
]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
FrictionLaw = Literal["direction_tied", "distributed", "progressive", "auto"]

DIRECTION_SHARE = 0.01
"""Default share of the largest value below which a direction is not given."""

CITATIONS = (
    "[Papailiou1997]",
    "[Kim2020]",
    "[KraincanicKebadze2001]",
    "[Kimball1925]",
    "[Dong2015]",
)


@dataclass(frozen=True)
class ArmourAxes:
    """Stress neutral axis, moment and stick zone of one armour layer.

    Arrays over samples have shape (samples,), over wires and samples
    (wires, samples). Angles in radians from x towards y.
    """

    name: str
    r_helix_m: float
    phi_wires_rad: FloatArray
    """Wire azimuths at the section."""
    resultant_Pa: FloatArray
    """First circular harmonic ``R`` of the wire bending stress, (samples, 2)."""
    stress_direction_rad: FloatArray
    """Direction ``psi_L`` of ``R`` (NaN where not defined); the stress
    neutral axis is the diameter perpendicular to it."""
    moment_Nm: FloatArray
    """Moment of the layer's wire stresses, conjugate to ``b``, (samples, 2)."""
    sign_changes: NDArray[np.int64]
    """Times the wire bending stress changes sign round the layer."""
    stuck: BoolArray | None
    """Wires in stick (direction-tied law), (wires, samples); ``None`` for
    the other laws."""
    friction_model: str

    @property
    def stick_arc_rad(self) -> FloatArray | None:
        """Angular extent of the wires in stick, ``2 pi`` times their share."""
        if self.stuck is None:
            return None
        count = self.stuck.shape[0]
        return np.asarray(
            2.0 * math.pi * np.count_nonzero(self.stuck, axis=0) / count,
            dtype=np.float64,
        )


@dataclass(frozen=True)
class NeutralAxisResult:
    """The neutral axes of :func:`compute_neutral_axes` at every sample."""

    kappa_x_1_m: FloatArray
    kappa_y_1_m: FloatArray
    axial_strain: FloatArray
    """Axial strain of the cable axis ``eps0``."""
    bending_direction_rad: FloatArray
    """``theta``; NaN below the curvature threshold."""
    strain_axis_offset_m: FloatArray
    """``e = -eps0/|b|`` along ``theta``; NaN where not defined or off the
    section."""
    moment_Nm: FloatArray
    """Section moment conjugate to ``b``, (samples, 2)."""
    moment_direction_rad: FloatArray
    """``psi``; NaN below the moment threshold."""
    layers: tuple[ArmourAxes, ...]
    curvature_threshold_1_m: float
    section_radius_m: float
    ei_min_Nm2: float
    ei_max_Nm2: float
    """Full-stick bending stiffness: the stick stiffness of the armour layers
    plus ``EI_min``."""
    direction_share: float
    provenance: Provenance
    core_moment_Nm: FloatArray | None = None
    """Moment of the laid-core law (cores, internal rings, filler profiles)
    in ``moment_Nm``, (samples, 2); ``None`` without core friction."""

    @property
    def curvature_1_m(self) -> FloatArray:
        """Curvature magnitude ``|b|``."""
        return np.asarray(np.hypot(self.kappa_x_1_m, self.kappa_y_1_m))

    @property
    def moment_angle_rad(self) -> FloatArray:
        """``delta = psi - theta`` wrapped to (-pi, pi]."""
        return angle_from(self.bending_direction_rad, self.moment_direction_rad)

    def stress_angle_rad(self, layer: ArmourAxes) -> FloatArray:
        """``psi_L - theta`` of ``layer``: the angle from the strain neutral
        axis to the layer's stress neutral axis, wrapped to (-pi, pi]."""
        return angle_from(self.bending_direction_rad, layer.stress_direction_rad)


def angle_from(reference: FloatArray, angle: FloatArray) -> FloatArray:
    """``angle - reference`` wrapped to (-pi, pi] (NaN where either is NaN)."""
    difference = np.asarray(angle, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    return np.asarray(-(np.mod(-difference + math.pi, 2.0 * math.pi) - math.pi))


def sign_changes(stress: FloatArray, tolerance: float = 0.0) -> NDArray[np.int64]:
    """Sign changes round a closed ring of wires, per sample.

    ``stress`` has shape (wires, samples), the wires in order round the ring.
    Values with magnitude at or below ``tolerance`` take no sign and are
    passed over.
    """
    values = np.asarray(stress, dtype=np.float64)
    sign = np.where(np.abs(values) > tolerance, np.sign(values), 0.0)
    count = sign.shape[0]
    index = np.where(sign != 0.0, np.arange(count)[:, None], -1)
    index = np.maximum.accumulate(index, axis=0)
    last = index[-1]
    index = np.where(index < 0, last[None, :], index)
    filled = np.take_along_axis(sign, np.maximum(index, 0), axis=0)
    filled[:, last < 0] = 0.0
    changed = filled != np.roll(filled, 1, axis=0)
    return np.asarray(np.count_nonzero(changed, axis=0), dtype=np.int64)


def _direction(vector: FloatArray, share: float) -> tuple[FloatArray, float]:
    """Direction of (samples, 2) vectors, NaN below ``share`` of the largest."""
    magnitude = np.hypot(vector[:, 0], vector[:, 1])
    largest = float(np.max(magnitude)) if magnitude.size else 0.0
    threshold = share * largest
    defined = (magnitude > 0.0) & (magnitude >= threshold)
    angle = np.where(defined, np.arctan2(vector[:, 1], vector[:, 0]), np.nan)
    return np.asarray(angle, dtype=np.float64), threshold


def _stuck(
    spec: HelicalLayerSpec,
    kx: FloatArray,
    ky: FloatArray,
    stress: FloatArray,
    cap: FloatArray,
) -> BoolArray:
    """Wires in stick: friction not fully mobilised, ``|sigma| < cap``.

    A wire on the neutral axis of the cap has a zero cap and a zero stress;
    it is in stick when the elastic update alone carries it there (its trial
    stress ``sigma_prev + k_f r b . e_k`` increment is zero to rounding), as
    for every wire below the onset curvature, and in slip otherwise (no
    friction, or slip through the axis).
    """
    phi = np.asarray(spec.phi_wires, dtype=np.float64)
    drive = spec.r_helix_m * (
        np.cos(phi)[:, None] * kx[None, :] + np.sin(phi)[:, None] * ky[None, :]
    )
    previous = np.zeros_like(stress)
    previous[:, 1:] = stress[:, :-1]
    trial = previous + spec.k_f * np.diff(drive, axis=1, prepend=0.0)
    scale = spec.k_f * spec.r_helix_m * float(np.max(np.hypot(kx, ky), initial=0.0))
    tolerance = 1e-9 * scale
    below = np.abs(stress) < cap * (1.0 - 1e-12)
    on_axis = (cap <= tolerance) & (np.abs(trial) <= tolerance)
    # Without friction (every cap zero) no wire is held.
    on_axis &= np.any(cap > tolerance, axis=0)[None, :]
    return np.asarray(below | on_axis)


def _layer_axes(
    name: str,
    spec: HelicalLayerSpec,
    kx: FloatArray,
    ky: FloatArray,
    pressure: FloatArray | None,
    friction_model: FrictionLaw,
    share: float,
) -> ArmourAxes:
    """Wire bending stress, its resultant, the moment and the stick zone."""
    from cablexsec.mechanics.distributed_friction import (  # noqa: PLC0415
        distributed_wire_friction_stress,
    )
    from cablexsec.mechanics.papailiou import wire_slip_history  # noqa: PLC0415
    from cablexsec.mechanics.progressive_friction import (  # noqa: PLC0415
        progressive_wire_history,
    )

    phi = np.asarray(spec.phi_wires, dtype=np.float64)
    weight = spec.A_wire_m2 * math.cos(spec.lay_angle_rad)
    directions = np.column_stack((np.cos(phi), np.sin(phi)))
    stuck: BoolArray | None = None
    if friction_model == "direction_tied":
        history = wire_slip_history(
            spec, kx, ky, contact_pressure_Pa=pressure, keep_cap=True
        )
        stress = history.stress_Pa
        moment = np.asarray(history.moment_Nm, dtype=np.float64)
        assert history.cap_Pa is not None
        cap = history.cap_Pa
        stuck = _stuck(spec, kx, ky, stress, cap)
    elif friction_model == "progressive":
        progressive = progressive_wire_history(
            spec, kx, ky, contact_pressure_Pa=pressure
        )
        stress = progressive.stress_Pa
        moment = np.asarray(progressive.moment_Nm, dtype=np.float64)
    elif friction_model == "distributed":
        stress = distributed_wire_friction_stress(
            spec, kx, ky, contact_pressure_Pa=pressure
        )
        moment = np.asarray(
            weight * spec.r_helix_m * (stress.T @ directions), dtype=np.float64
        )
    else:
        raise CableInputError(
            "friction_model must be 'direction_tied', 'distributed', 'progressive' "
            "or 'auto'"
        )
    stress = np.asarray(stress, dtype=np.float64)
    resultant = np.asarray((2.0 / spec.n_wires) * (stress.T @ directions))
    direction, _ = _direction(resultant, share)
    scale = float(np.max(np.abs(stress))) if stress.size else 0.0
    return ArmourAxes(
        name=name,
        r_helix_m=float(spec.r_helix_m),
        phi_wires_rad=phi,
        resultant_Pa=resultant,
        stress_direction_rad=direction,
        moment_Nm=moment,
        sign_changes=sign_changes(stress, 1e-9 * scale),
        stuck=stuck,
        friction_model=friction_model,
    )


def _layer_models(
    layers: Sequence[tuple[str, HelicalLayerSpec]],
    kx: FloatArray,
    ky: FloatArray,
    friction_model: str,
) -> list[str]:
    """Friction model of every layer; ``"auto"`` resolved as in the fatigue route."""
    if friction_model != "auto":
        return [friction_model] * len(layers)
    from cablexsec.analysis.fatigue import (  # noqa: PLC0415
        _resolve_friction_model,
        _warn_auto_fallback,
        curvature_path_out_of_plane_ratio,
    )

    ratio = curvature_path_out_of_plane_ratio(kx, ky)
    models = []
    for name, spec in layers:
        model, reason = _resolve_friction_model("auto", spec, ratio, None, None)
        if reason:
            _warn_auto_fallback(name, reason, 4)
        models.append(model)
    return models


def neutral_axes(
    layers: Sequence[tuple[str, HelicalLayerSpec]],
    kappa_x: FloatArray,
    kappa_y: FloatArray,
    *,
    axial_strain: FloatArray | None = None,
    contact_pressure_Pa: Mapping[str, FloatArray] | None = None,
    ei_min_Nm2: float = 0.0,
    section_radius_m: float = math.inf,
    friction_model: FrictionLaw = "direction_tied",
    direction_share: float = DIRECTION_SHARE,
    core_moment_Nm: FloatArray | None = None,
    core_stick_EI_Nm2: float = 0.0,
    core_description: str | None = None,
) -> NeutralAxisResult:
    """Neutral axes of a section of armour ``layers`` (name, spec) over a history.

    ``kappa_x``, ``kappa_y`` are the axial-strain gradient [1/m];
    ``axial_strain`` the strain of the cable axis (zero when ``None``);
    ``contact_pressure_Pa`` the ``P1+P2`` history of each layer by name (the
    layer's own clamping when absent). ``friction_model`` ``"auto"`` takes
    the direction-tied law on a planar path and the distributed law on every
    layer it applies to otherwise (a rigid interface keeps the direction-tied
    law and warns, :class:`~cablexsec.provenance.FrictionModelFallbackWarning`);
    each layer reports its model. ``core_moment_Nm`` (samples, 2) is the
    moment of a laid-core law over the same history, added to the section
    moment, ``core_stick_EI_Nm2`` its stick stiffness, added to
    ``ei_max_Nm2``, and ``core_description`` names its friction model and
    loads in the provenance. See the module text for the quantities.
    """
    kx = np.asarray(kappa_x, dtype=np.float64)
    ky = np.asarray(kappa_y, dtype=np.float64)
    if kx.ndim != 1 or ky.shape != kx.shape or not kx.size:
        raise CableInputError("curvature histories must be matching, non-empty 1-D")
    if not (np.isfinite(kx).all() and np.isfinite(ky).all()):
        raise CableInputError("curvature histories must be finite")
    share = float(direction_share)
    if not (math.isfinite(share) and 0.0 <= share < 1.0):
        raise CableInputError("the direction share must lie in [0, 1)")
    if not (math.isfinite(ei_min_Nm2) and ei_min_Nm2 >= 0.0):
        raise CableInputError("EI_min must be finite and non-negative")
    if not section_radius_m > 0.0:
        raise CableInputError("the section radius must be positive")
    if not (math.isfinite(core_stick_EI_Nm2) and core_stick_EI_Nm2 >= 0.0):
        raise CableInputError(
            "the core stick stiffness must be finite and non-negative"
        )
    core: FloatArray | None = None
    if core_moment_Nm is not None:
        core = np.asarray(core_moment_Nm, dtype=np.float64)
        if core.shape != (kx.size, 2) or not np.isfinite(core).all():
            raise CableInputError(
                "the core moment must be finite, one (x, y) pair per sample"
            )
    elif core_stick_EI_Nm2 != 0.0:
        raise CableInputError("a core stick stiffness needs the core moment")
    strain = (
        np.zeros_like(kx)
        if axial_strain is None
        else np.asarray(axial_strain, dtype=np.float64)
    )
    if strain.shape != kx.shape or not np.isfinite(strain).all():
        raise CableInputError("axial strain must be finite and match the curvature")
    pressures = dict(contact_pressure_Pa or {})
    models = _layer_models(layers, kx, ky, friction_model)
    armour = tuple(
        _layer_axes(name, spec, kx, ky, pressures.get(name), model, share)  # type: ignore[arg-type]
        for (name, spec), model in zip(layers, models, strict=True)
    )
    curvature = np.column_stack((kx, ky))
    theta, threshold = _direction(curvature, share)
    magnitude = np.hypot(kx, ky)
    with np.errstate(divide="ignore", invalid="ignore"):
        offset = np.where(np.isfinite(theta), -strain / magnitude, np.nan)
    offset = np.where(np.abs(offset) <= section_radius_m, offset, np.nan)
    moment = ei_min_Nm2 * curvature
    for layer in armour:
        moment = moment + layer.moment_Nm
    if core is not None:
        moment = moment + core
    psi, _ = _direction(moment, share)
    limitations = [
        "the wire bending stress is that of the slip law of the fatigue route at "
        "the declared clamping (and its tension increment); bending-induced "
        "contact redistribution is not solved",
        "the moment adds EI_min times the curvature to the armour wire moments; "
        + (
            "core friction of laid power cores is not included"
            if core is None
            else "it adds the laid-core law over the same history"
            + ("" if not core_description else f" ({core_description})")
        ),
        "the stress neutral axis is the diameter of the first circular harmonic; "
        "after a reversal the stress can change sign more than twice round a "
        "layer",
    ]
    if any(model != "direction_tied" for model in models):
        limitations.append("the stick zone is given for the direction-tied law only")
    else:
        limitations.append(
            "the direction-tied cap follows the current bending direction; under "
            "non-proportional curvature the stress neutral axis depends on that "
            "assumption"
        )
    return NeutralAxisResult(
        kappa_x_1_m=kx,
        kappa_y_1_m=ky,
        axial_strain=strain,
        bending_direction_rad=theta,
        strain_axis_offset_m=np.asarray(offset, dtype=np.float64),
        moment_Nm=np.asarray(moment, dtype=np.float64),
        moment_direction_rad=psi,
        layers=armour,
        curvature_threshold_1_m=threshold,
        section_radius_m=float(section_radius_m),
        ei_min_Nm2=float(ei_min_Nm2),
        ei_max_Nm2=float(
            ei_min_Nm2 + sum(spec.EI_stick for _, spec in layers) + core_stick_EI_Nm2
        ),
        direction_share=share,
        provenance=Provenance(
            method=(
                "neutral axes of the section: bending direction, strain neutral "
                "axis, stress neutral axis of each armour layer (first circular "
                f"harmonic of the {_models_text(friction_model, models)} wire bending "
                "stress), moment direction and stick zone"
            ),
            citations=CITATIONS,
            limitations=tuple(limitations),
        ),
        core_moment_Nm=core,
    )


def _models_text(friction_model: str, models: Sequence[str]) -> str:
    """The armour friction model of the provenance (``"auto"`` resolved)."""
    if friction_model != "auto":
        return friction_model.replace("_", "-")
    chosen = sorted({model.replace("_", "-") for model in models})
    return f"{' and '.join(chosen) or 'direction-tied'} (auto)"


def section_radius(cable: CableData) -> float:
    """Outer radius of the cable's layers [m]."""
    from cablexsec.cables.schema import HelicalLayer  # noqa: PLC0415

    reach = 0.0
    for layer in cable.layers:
        outer = getattr(layer, "r_outer_m", None)
        if outer is not None:
            reach = max(reach, float(outer))
        if (
            isinstance(layer, HelicalLayer)
            and layer.r_helix_m is not None
            and layer.d_wire_m is not None
        ):
            reach = max(reach, layer.r_helix_m + layer.wire_radial_thickness_m / 2.0)
    if reach <= 0.0:
        raise CableInputError("the cable has no layer with an outer radius")
    return reach


def compute_neutral_axes(
    cable: CableData,
    kappa_x: FloatArray,
    kappa_y: FloatArray,
    *,
    tension_N: FloatArray | None = None,
    axial_strain: FloatArray | None = None,
    contact_tension_N: FloatArray | None = None,
    mu: float | None = None,
    friction_model: FrictionLaw = "direction_tied",
    direction_share: float = DIRECTION_SHARE,
    section_radius_m: float | None = None,
    core_friction: bool | CoreFrictionSpec | None = None,
    core_friction_model: CoreFrictionModelName | None = None,
    core_loads: CoreLoadMode = "follow",
) -> NeutralAxisResult:
    """Neutral axes of ``cable`` over a curvature history (module text).

    ``tension_N`` is the force on the armour [N]: the axis strain is
    ``tension / EA_fixed`` and the armour contact pressure follows it
    (:func:`cablexsec.armour_contact_histories`, as in
    :func:`cablexsec.compute_fatigue`). ``axial_strain`` is a direct axis
    strain instead; the contact pressure is then the declared clamping. With
    neither, the axis strain is zero at the declared clamping.
    ``contact_tension_N`` (a non-negative history [N]) sets the contact
    pressure instead, the declared clamping plus its tension increment: the
    force behind a direct strain, or zeros for the declared clamping alone.
    ``section_radius_m`` (default: the outer radius of the layers) bounds the
    strain neutral axis.

    ``core_friction`` (``True`` or a
    :class:`~cablexsec.mechanics.core_friction.CoreFrictionSpec`) adds the
    stick-slip law of the laid power cores, their internal wire rings and the
    filler profiles to the section moment, as ``core_friction`` of
    :func:`cablexsec.compute_mkappa` does: the biaxial history of
    :func:`~cablexsec.mechanics.core_friction.core_friction_history` with the
    contact loads of the bounded contact model. ``core_friction_model``
    selects the friction model of the cores and profiles (``"direction_tied"``,
    ``"distributed"`` or ``"auto"``, module notes of
    :mod:`cablexsec.mechanics.core_friction`); ``None`` (default) takes
    ``friction_model`` (the direction-tied law with the progressive armour
    law, which the cores do not have). ``core_loads`` sets the core and
    profile contact loads: ``"follow"`` (default) evaluates them at every
    sample of the tension history that sets the armour contact pressure
    (``contact_tension_N``, else ``tension_N``;
    :func:`~cablexsec.mechanics.core_friction.core_load_history`), as the
    armour pressure does, so that the section agrees with a law whose sliders
    all scale with the tension; ``"reference"`` holds them at the first
    sample of that history (the earlier behaviour, as in
    :func:`cablexsec.compute_mkappa` at one tension). Without a tension
    history both hold the loads at zero tension. A planar history at a
    constant tension gives, along its plane, the moment of
    :func:`cablexsec.compute_mkappa` with the same options (direction-tied
    law). ``None`` (default) keeps the armour-only moment bit for bit.
    """
    from cablexsec.api import armour_contact_histories  # noqa: PLC0415
    from cablexsec.cable_adapters import (  # noqa: PLC0415
        cable_to_mkappa_specs,
        structural_armour_layers,
    )
    from cablexsec.mechanics.core_friction import (  # noqa: PLC0415
        CORE_FRICTION_MODELS,
        CORE_LOAD_MODES,
        build_core_friction,
        core_friction_history,
        core_load_history,
        resolve_core_friction,
    )
    from cablexsec.mechanics.stiffness import ea_fixed, ei_min_slip  # noqa: PLC0415

    if tension_N is not None and axial_strain is not None:
        raise CableInputError("give tension or axial strain, not both")
    if core_loads not in CORE_LOAD_MODES:
        raise CableInputError(
            f"core_loads must be one of {CORE_LOAD_MODES}, got {core_loads!r}"
        )
    if core_friction_model is not None and core_friction_model not in (
        CORE_FRICTION_MODELS
    ):
        raise CableInputError(
            f"core_friction_model must be one of {CORE_FRICTION_MODELS} or None, "
            f"got {core_friction_model!r}"
        )
    kx = np.asarray(kappa_x, dtype=np.float64)
    strain: FloatArray | None = None
    pressure: dict[str, FloatArray] | None = None
    load_tension: FloatArray | None = None
    reference = 0.0
    if tension_N is not None:
        tension = np.asarray(tension_N, dtype=np.float64)
        if tension.shape != kx.shape or not np.isfinite(tension).all():
            raise CableInputError("tension must be finite and match the curvature")
        if np.any(tension < 0.0):
            raise CableInputError("the force on the armour must not be negative")
        reference = float(tension[0])
        strain = tension / ea_fixed(cable)
        pressure = armour_contact_histories(cable, tension)[0]
        load_tension = tension
    elif axial_strain is not None:
        strain = np.asarray(axial_strain, dtype=np.float64)
    if contact_tension_N is not None:
        contact = np.asarray(contact_tension_N, dtype=np.float64)
        if contact.shape != kx.shape or not np.isfinite(contact).all():
            raise CableInputError(
                "the contact tension must be finite and match the curvature"
            )
        if np.any(contact < 0.0):
            raise CableInputError("the contact tension must not be negative")
        reference = float(contact[0])
        pressure = armour_contact_histories(cable, contact)[0]
        load_tension = contact
    specs = cable_to_mkappa_specs(cable, reference, mu=mu, contact_model="bounded")
    names = [layer.name for layer in structural_armour_layers(cable)]
    if not specs:
        raise CableInputError("the cable has no structural armour layer")
    core_spec = resolve_core_friction(core_friction)
    if core_spec is None and core_friction_model is not None:
        raise CableInputError("core_friction_model needs core_friction")
    core_moment: FloatArray | None = None
    core_stick = 0.0
    core_description: str | None = None
    if core_spec is not None:
        model = build_core_friction(
            cable, core_spec, reference, contact_model="bounded"
        )
        loads = (
            core_load_history(cable, model, load_tension, contact_model="bounded")
            if core_loads == "follow" and load_tension is not None
            else None
        )
        if core_friction_model is not None:
            core_model: str = core_friction_model
        elif friction_model in CORE_FRICTION_MODELS:
            core_model = friction_model
        else:
            core_model = "direction_tied"
        history = core_friction_history(
            model,
            kx,
            np.asarray(kappa_y, dtype=np.float64),
            friction_model=core_model,  # type: ignore[arg-type]
            loads=loads,
        )
        core_moment = history.moment_Nm
        core_stick = model.stick_EI_Nm2
        core_description = (
            f"{history.friction_model.replace('_', '-')} core law"
            + (", " + core_model if core_model == "auto" else "")
            + ", core contact loads "
            + (
                "following the tension history"
                if loads is not None
                else f"at the reference tension {reference:g} N"
            )
        )
    return neutral_axes(
        list(zip(names, specs, strict=True)),
        kx,
        np.asarray(kappa_y, dtype=np.float64),
        axial_strain=strain,
        contact_pressure_Pa=pressure,
        ei_min_Nm2=ei_min_slip(cable),
        section_radius_m=(
            section_radius(cable) if section_radius_m is None else section_radius_m
        ),
        friction_model=friction_model,
        direction_share=direction_share,
        core_moment_Nm=core_moment,
        core_stick_EI_Nm2=core_stick,
        core_description=core_description,
    )
