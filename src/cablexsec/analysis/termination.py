# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Termination (clamp) correction of the armour friction stress for fatigue.

The sectional friction laws of the fatigue path assume curvature that is
uniform over at least one pitch and armour that is free to slip axially. Near
an end fitting that anchors the armour, or at a bend-stiffener root where the
armour is clamped, the wires are anchored: friction feeds a mean wire force
that decays over a boundary layer of length ``l_x = sqrt(L_s P) / 2``
(:mod:`cablexsec.mechanics.clamp_layer`). :class:`TerminationSpec` describes
the section position relative to the anchor, and
:func:`termination_friction_increment` returns the clamp increment of the
friction stress of every wire, which the fatigue path adds to the sectional
friction stress::

    sigma_f = sigma_f,sectional + g_k sigma_u (n_anchored(x_s, t) - n_far(t)).

``n_anchored`` is the anchored-wire solution at the section (rigid stick,
Coulomb bound per unit length scaled by ``P(t) / P_ref``, exact chain
projection), ``n_far`` the solution of the *same discrete solver* for a wire
far from any anchor under uniform curvature equal to the section curvature,
``sigma_u = sigma_0 mu |sin a|`` the slope of the linearised Papailiou cap and
``g_k`` the per-wire contact-shape factor. Because the reference is the far
field of the same discretisation, with bitwise-identical wire phases, load
path and bound history, the increment decays to round-off away from the anchor
and the sectional result is recovered in the far field. The mesh has an odd
number of segments per pitch: with an even number, segments come in opposite
pairs and the discrete far field keeps an offset that depends on the length of
the solved domain (about 1-2 % of the cap), so the two runs would not agree.
Sections farther from the anchor than the solved reach (``reach_lengths``
boundary layers at the largest curvature) are returned unchanged, bit for
bit. The discrete boundary layer has finite support of about 6-9 ``l_x``
(growing slowly with mesh refinement); the default reach of 8 ``l_x`` cuts a
tail below about 0.1 % of the cap.

Biaxial histories
-----------------
The clamp solver is planar: the stuck force per unit load is
``k_hat(x) cos theta(x)`` for one bending plane. The curvature history at the
section is resolved in the principal frame of its centred second moment
(direction ``beta`` unless ``direction_rad`` is given):

* ``biaxial="principal"`` (default): the clamp increment is computed along the
  principal component ``kappa_p = kappa_x cos beta + kappa_y sin beta`` only.
  It is exact, within the clamp model, for planar histories in any direction;
  for a biaxial history the minor component (variance fraction
  ``1 - principal_fraction``, reported) receives no clamp increment, so the
  correction is low by roughly that fraction of the clamp-induced range;
* ``biaxial="components"``: planar increments along the principal and the
  minor direction are computed separately and added. It also reduces to the
  planar result for planar histories. For biaxial histories the superposition
  of two Coulomb responses, each with the full friction bound, is not exact:
  it tends to over-estimate the increment when both components slip at the
  same time, and it is not a proven bound.

The sectional friction stress, to which the increment is added, keeps the
full biaxial direction-tied (or distributed) law of the package in both modes.

Stick stiffness and solver (gap T2)
-----------------------------------
The clamp increment depends strongly on how the package's ``xi``-reduced
stick stiffness is carried into the anchored wire (about 1/xi in the slip
demand; a clamped hang-off damage factor of 5 against 50 in the
curvature-gradient study). It is an explicit option:

* ``stick="scaled"`` (default, unchanged): rigid interface with the imposed
  strain scaled by ``xi`` (``kappa_c`` the package onset curvature);
* ``stick="rigid"``: ``xi = 1`` (``kappa_c`` the rigid-stick onset);
* ``stick="spring"`` (``solver="along_wire"`` only): elastic interface
  ``k_hat = xi / (1 - xi)`` with the full imposed strain, the distributed
  model's representation (Lutchansky shear lag at the anchor).

``solver="along_wire"`` uses :mod:`cablexsec.mechanics.along_wire`: every
wire is solved along its own helix with its azimuth-dependent contact shape,
the optional capstan bound (``capstan=True``, the package cap) and the
biaxial imposed strain ``k_hat(x)(kappa_x cos theta + kappa_y sin theta)``
directly (the ``biaxial`` setting is not used; the history is still scaled
by one shape). The reference is the same solver far from any anchor, so the
increment vanishes there as for the clamp solver.

Limits
------
Axial slip only, one wire per layer without layer interaction, proportional
curvature ``kappa(x, t) = kappa_s(t) k_hat(x)`` with ``k_hat(x_s) = 1``,
wire modulus and stick stiffness at the layer reference. The clamp solver
uses a rigid interface, a constant bound and a linearised cap for the
increment (the sectional part keeps the ``expm1`` cap). The anchorage
position controls the result near the anchor; a bonded (epoxy) length inside
a fitting is not frictional, so the anchor is its inner end. Measured
near-termination armour stresses (Sævik 2011; Gjøsteen et al. 2027) are not
reproduced by the clamp increment in its default convention (BENCHMARKS,
rows SV11 and T2).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.along_wire import (
    STICK_CONVENTIONS,
    WireChainLayer,
    _wire_history,
)
from cablexsec.mechanics.bend_restrictor import CurvatureField
from cablexsec.mechanics.clamp_layer import (
    anchored_wire_field_response,
    anchored_wire_response,
    clamp_boundary_layer_length,
)
from cablexsec.mechanics.mkappa import HelicalLayerSpec
from cablexsec.mechanics.tendon_bending import LateralRelief
from cablexsec.provenance import (
    UnvalidatedModelWarning,
    real_array,
    require_finite,
)

__all__ = [
    "TerminationIncrement",
    "TerminationSpec",
    "principal_direction",
    "termination_friction_increment",
]

FloatArray = NDArray[np.float64]

_ANCHORS = ("clamped", "none")

CLAMPED_TERMINATION_WARNING = (
    "the anchored-wire Coulomb correction (TerminationSpec anchor='clamped', "
    "clamp or along_wire solver) is an upper estimate that is not supported "
    "by the only published near-termination measurements (Saevik 2011 / "
    "Gjosteen 2027 bellmouth FBG ranges: corrected ranges 1.65-3.4 times "
    "measured, sectional 1.03 times; doc/benchmarks.md SV11 and T2). Report "
    "it as a sensitivity bound beside the sectional result."
)


def warn_if_clamped(
    termination: TerminationSpec | Mapping[str, TerminationSpec] | None,
    *,
    stacklevel: int = 3,
) -> None:
    """Issue :class:`UnvalidatedModelWarning` once for a clamped termination.

    ``anchor="none"`` specifications and ``None`` do not warn.
    """
    if termination is None:
        return
    specs = (
        [termination]
        if isinstance(termination, TerminationSpec)
        else list(termination.values())
    )
    if any(spec.anchor == "clamped" for spec in specs):
        warnings.warn(
            CLAMPED_TERMINATION_WARNING,
            UnvalidatedModelWarning,
            stacklevel=stacklevel,
        )


_BIAXIAL = ("principal", "components")
_SOLVERS = ("clamp", "along_wire")


@dataclass(frozen=True)
class TerminationSpec:
    """Position of a fatigue section relative to an armour anchor.

    Attributes
    ----------
    distance_m : float
        Distance of the analysed section from the clamp face (end-fitting
        face or bend-stiffener root) [m], measured along the cable away from
        the anchor.
    anchor : {"clamped", "none"}
        ``"clamped"``: the armour is anchored (end fitting, clamped stiffener
        root). ``"none"``: no correction; the sectional result is returned
        unchanged.
    setback_m : float
        Straight, frictional length between the armour anchor and the clamp
        face, with zero curvature [m]. Zero puts the anchor at the face.
    curvature_profile : tuple of numpy.ndarray or None
        ``(x_m, shape)``: curvature shape along the cable from the clamp face
        (``x = 0``), increasing ``x``; held at its end values outside the
        table. ``None`` is uniform curvature from the face. The shape is
        normalised to one at ``distance_m``, so the curvature history passed
        to the fatigue path is the history at the section. Proportional
        loading ``kappa(x, t) = kappa_s(t) k_hat(x)`` is assumed.
    biaxial : {"principal", "components"}
        Treatment of biaxial histories (module notes).
    direction_rad : float or None
        Bending direction of the planar clamp solution in the curvature frame;
        ``None`` uses the principal direction of the history.
    axis_sign : int
        ``+1`` when the distance from the anchor increases along the positive
        axial direction of the right-handed curvature frame, ``-1`` otherwise.
        With the layer lay hand it sets the sense in which a wire turns along
        the cable.
    segments_per_pitch : int
        Clamp-solver segments per pitch (a minimum). An odd count of up to
        1.25 times this value is used: with an even count the discrete far
        field keeps a domain-dependent offset of about 1-2 % of the cap.
    max_increment : float
        Largest load increment per substep, in units of the slip-onset
        curvature at the largest profile value.
    reach_lengths : float
        Solved reach beyond the section, in clamp boundary layers ``l_x`` at
        the largest curvature (at least one pitch). Sections farther than this
        from the anchor take the sectional result unchanged.
    stick : {"scaled", "rigid", "spring"}
        Stick-stiffness convention of the anchored wire (module notes);
        ``"spring"`` needs ``solver="along_wire"``.
    solver : {"clamp", "along_wire"}
        Anchored-wire solver (module notes).
    capstan : bool
        Capstan term of the bound (``solver="along_wire"`` only).
    lateral_relief : LateralRelief or None
        Vector-Coulomb relief of the axial friction bound by the transverse
        load of the tendon (gap G1,
        :class:`~cablexsec.mechanics.tendon_bending.LateralRelief`): the bound
        at every node of the anchored and far chains is multiplied by its
        factor at the local curvature amplitude and gradient (from the
        profile) and, within its ``anchor_layer_m`` of the anchor, by its
        floor. Needs ``solver="along_wire"`` and a proportional profile.
    curvature_field : CurvatureField or None
        Non-proportional planar field ``kappa(x, t)`` from the clamp face
        (gap P5: a bend restrictor locking joint by joint or a bend
        stiffener, :mod:`cablexsec.mechanics.bend_restrictor`), one column
        per sample of the history, in the bending plane of the clamp
        solution. Replaces ``curvature_profile``; the history at the section
        (projected on that plane) must equal the field at ``distance_m``.

    Notes
    -----
    With ``anchor="clamped"`` the anchored-wire Coulomb correction, with
    either solver, is an upper estimate. It is not supported by the only
    published near-termination measurements: on the bellmouth FBG ranges of
    Saevik (2011) and Gjosteen et al. (2027) the corrected ranges are 1.65-3.4
    times the measured ones, the sectional ranges 1.03 times
    (doc/benchmarks.md, SV11 and T2). :func:`compute_fatigue` therefore
    issues :class:`~cablexsec.provenance.UnvalidatedModelWarning` for it;
    ``anchor="none"`` does not warn.
    """

    distance_m: float
    anchor: str = "clamped"
    setback_m: float = 0.0
    curvature_profile: tuple[FloatArray, FloatArray] | None = None
    biaxial: str = "principal"
    direction_rad: float | None = None
    axis_sign: int = 1
    segments_per_pitch: int = 64
    max_increment: float = 0.5
    reach_lengths: float = 8.0
    stick: str = "scaled"
    solver: str = "clamp"
    capstan: bool = False
    curvature_field: CurvatureField | None = None
    lateral_relief: LateralRelief | None = None

    def __post_init__(self) -> None:
        if self.lateral_relief is not None:
            if not isinstance(self.lateral_relief, LateralRelief):
                raise CableInputError("lateral_relief must be a LateralRelief")
            if self.solver != "along_wire" or self.curvature_field is not None:
                raise CableInputError(
                    "lateral_relief needs solver='along_wire' and a "
                    "proportional curvature profile"
                )
        if self.curvature_field is not None:
            if self.curvature_profile is not None:
                raise CableInputError(
                    "give curvature_profile or curvature_field, not both"
                )
            if not isinstance(self.curvature_field, CurvatureField):
                raise CableInputError("curvature_field must be a CurvatureField")
        require_finite(self.distance_m, "distance_m")
        require_finite(self.setback_m, "setback_m")
        if self.distance_m < 0 or self.setback_m < 0:
            raise CableInputError("distance_m and setback_m must be non-negative")
        if self.anchor not in _ANCHORS:
            raise CableInputError(f"anchor must be one of {_ANCHORS}")
        if self.biaxial not in _BIAXIAL:
            raise CableInputError(f"biaxial must be one of {_BIAXIAL}")
        if self.direction_rad is not None:
            require_finite(self.direction_rad, "direction_rad")
        if self.axis_sign not in (1, -1):
            raise CableInputError("axis_sign must be +1 or -1")
        if self.solver not in _SOLVERS:
            raise CableInputError(f"solver must be one of {_SOLVERS}")
        if self.stick not in STICK_CONVENTIONS:
            raise CableInputError(f"stick must be one of {STICK_CONVENTIONS}")
        if self.solver == "clamp" and (self.stick == "spring" or self.capstan):
            raise CableInputError(
                "the spring stick convention and the capstan bound need "
                "solver='along_wire'"
            )
        if (
            isinstance(self.segments_per_pitch, (bool, np.bool_))
            or not isinstance(self.segments_per_pitch, (int, np.integer))
            or self.segments_per_pitch < 8
        ):
            raise CableInputError("segments_per_pitch must be an integer of at least 8")
        for name in ("max_increment", "reach_lengths"):
            value = getattr(self, name)
            require_finite(value, name)
            if value <= 0:
                raise CableInputError(f"{name} must be positive")
        if self.curvature_profile is not None:
            try:
                x_raw, s_raw = self.curvature_profile
            except (TypeError, ValueError) as exc:
                raise CableInputError(
                    "curvature_profile must be a pair (x_m, shape)"
                ) from exc
            x = real_array(x_raw, "curvature_profile x_m")
            s = real_array(s_raw, "curvature_profile shape")
            if (
                x.ndim != 1
                or x.shape != s.shape
                or x.size < 2
                or not (np.isfinite(x).all() and np.isfinite(s).all())
                or np.any(np.diff(x) <= 0)
                or x[0] < 0
            ):
                raise CableInputError(
                    "curvature_profile must be finite, start at x >= 0 and "
                    "increase strictly in x"
                )
            x = np.array(x, dtype=np.float64)
            s = np.array(s, dtype=np.float64)
            x.flags.writeable = False
            s.flags.writeable = False
            object.__setattr__(self, "curvature_profile", (x, s))
            if self._raw_shape(np.array([self.distance_m]))[0] == 0:
                raise CableInputError(
                    "the curvature profile vanishes at the section distance"
                )

    def _raw_shape(self, x: FloatArray) -> FloatArray:
        """Un-normalised shape; zero on the straight set-back (``x < 0``)."""
        x = np.asarray(x, dtype=np.float64)
        if self.curvature_profile is None:
            raw = np.ones_like(x)
        else:
            px, ps = self.curvature_profile
            raw = np.interp(np.maximum(x, px[0]), px, ps)
        return np.asarray(np.where(x >= 0.0, raw, 0.0))

    def shape(self, x_m: FloatArray) -> FloatArray:
        """Curvature shape normalised to one at the section."""
        norm = float(self._raw_shape(np.array([self.distance_m]))[0])
        return np.asarray(self._raw_shape(x_m) / norm)

    def shape_maximum(self) -> float:
        """Largest ``|k_hat|`` over the cable (at least one)."""
        if self.curvature_field is not None:
            here = float(np.max(np.abs(self.curvature_field.at(self.distance_m))))
            return max(1.0, self.curvature_field.peak() / here) if here > 0 else 1.0
        if self.curvature_profile is None:
            return 1.0
        norm = abs(float(self._raw_shape(np.array([self.distance_m]))[0]))
        return max(1.0, float(np.max(np.abs(self.curvature_profile[1]))) / norm)


@dataclass(frozen=True)
class TerminationIncrement:
    """Clamp increment of the friction stress of one layer.

    ``stress_Pa`` has shape (wire, time). ``applied`` is false when no
    correction was made (``anchor="none"``, no friction, or the section beyond
    the solved reach); ``stress_Pa`` is then exactly zero. ``directions_rad``
    are the bending directions solved (one for ``"principal"``, two for
    ``"components"``); ``principal_fraction`` is the variance fraction of the
    section history along the first of them. ``a_max`` is the largest
    ``|kappa| / kappa_c`` on the profile, ``boundary_layer_m`` the clamp layer
    ``l_x`` at that value and ``reach_m`` the solved length beyond the
    section.
    """

    stress_Pa: FloatArray
    applied: bool
    directions_rad: tuple[float, ...]
    principal_fraction: float
    kappa_c_1_m: float
    sigma_u_Pa: float
    pitch_m: float
    a_max: float
    boundary_layer_m: float
    reach_m: float
    segments_per_pitch: int
    note: str = ""


def principal_direction(
    kappa_x: FloatArray, kappa_y: FloatArray
) -> tuple[float, float]:
    """Principal direction of a curvature history and its variance fraction.

    The eigenvector of the largest eigenvalue of the centred second moment of
    ``(kappa_x, kappa_y)``. A history without variance uses the direction of
    its mean (or ``0``) with fraction one.
    """
    kx = np.asarray(kappa_x, dtype=np.float64)
    ky = np.asarray(kappa_y, dtype=np.float64)
    k = np.stack([kx - kx.mean(), ky - ky.mean()], axis=1)
    moment = k.T @ k
    values, vectors = np.linalg.eigh(moment)
    total = float(values.sum())
    if not total > 0:
        mx, my = float(kx.mean()), float(ky.mean())
        return (math.atan2(my, mx) if (mx or my) else 0.0), 1.0
    v = vectors[:, -1]
    return math.atan2(float(v[1]), float(v[0])), float(values[-1] / total)


def _odd_mesh(requested: int, n_wires: int) -> int:
    """Odd segments per pitch, at least ``requested``, sharing most phases.

    An even count puts every segment opposite another (``cos`` changes sign),
    and the discrete far field of the chain then keeps an offset that depends
    on the domain length (about 1-2 % of the cap). With an odd count the far
    field is unique, so the anchored and far-field runs agree to round-off.
    Among odd counts up to ``1.25 requested`` the one with the largest common
    divisor with the wire count (fewest distinct wire phase offsets, hence
    fewest far-field runs) is taken, the smallest on a tie.
    """
    start = requested + 1 - requested % 2
    candidates = range(start, max(start, int(math.ceil(1.25 * requested))) + 1, 2)
    return max(candidates, key=lambda s: (math.gcd(s, n_wires), -s))


def _fine_path(
    load: FloatArray, scale: float, max_increment: float
) -> tuple[FloatArray, NDArray[np.bool_], NDArray[np.int64], FloatArray]:
    """Substep a sample path (from the virgin state) so increments stay small.

    Returns the fine load, the record flags (last substep of each sample),
    the sample interval of each substep and its interpolation weight.
    """
    path = np.concatenate([[0.0], load])
    counts = np.maximum(
        1, np.ceil(np.abs(np.diff(path)) * scale / max_increment)
    ).astype(np.int64)
    interval = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    position = np.arange(interval.size) - starts[interval] + 1
    weight = position / counts[interval]
    fine = path[interval] * (1.0 - weight) + path[interval + 1] * weight
    record = position == counts[interval]
    return fine, record, interval, weight


def termination_friction_increment(
    spec: HelicalLayerSpec,
    kappa_x: FloatArray,
    kappa_y: FloatArray,
    termination: TerminationSpec,
    *,
    lay_hand: int = 1,
    contact_pressure_Pa: FloatArray | None = None,
) -> TerminationIncrement:
    """Clamp increment of the friction stress of every wire of one layer [Pa].

    Parameters
    ----------
    spec : HelicalLayerSpec
        The layer (reference contact pressure, friction, Yun coefficient).
    kappa_x, kappa_y : numpy.ndarray
        Curvature (strain-gradient) history at the section [1/m].
    termination : TerminationSpec
        Section position, anchor and solver settings.
    lay_hand : int
        ``+1`` right-hand lay, ``-1`` left-hand lay.
    contact_pressure_Pa : numpy.ndarray or None
        Friction-setting pressure history; the friction bound is scaled by
        its ratio to ``spec.contact_pressure_Pa``.

    Returns
    -------
    TerminationIncrement
    """
    kx = np.ascontiguousarray(real_array(kappa_x, "kappa_x"), dtype=np.float64)
    ky = np.ascontiguousarray(real_array(kappa_y, "kappa_y"), dtype=np.float64)
    if (
        kx.ndim != 1
        or kx.shape != ky.shape
        or not (np.isfinite(kx).all() and np.isfinite(ky).all())
    ):
        raise CableInputError("curvature histories must be finite matching vectors")
    if lay_hand not in (1, -1):
        raise CableInputError("lay_hand must be +1 or -1")
    n_w = spec.n_wires
    n_t = kx.size
    zero = np.zeros((n_w, n_t), dtype=np.float64)
    pitch = 2 * math.pi * spec.r_helix_m / math.tan(spec.lay_angle_rad)
    beta, fraction = principal_direction(kx, ky)
    if termination.direction_rad is not None:
        beta = float(termination.direction_rad)
        total = float(np.var(kx) + np.var(ky))
        along = kx * math.cos(beta) + ky * math.sin(beta)
        fraction = float(np.var(along) / total) if total > 0 else 1.0
    directions = (
        (beta,)
        if termination.biaxial == "principal"
        else (
            beta,
            beta + math.pi / 2,
        )
    )
    sin_a = abs(math.sin(spec.lay_angle_rad))
    width = 2 * math.pi * spec.r_helix_m / n_w
    p_ref = float(spec.contact_pressure_Pa)
    sigma0 = (
        spec.r_helix_m
        * p_ref
        * width
        * math.cos(spec.lay_angle_rad)
        / (spec.A_wire_m2 * sin_a**2)
        if sin_a > 1e-12
        else 0.0
    )
    sigma_u = sigma0 * spec.mu * sin_a
    stiffness = spec.k_f * spec.r_helix_m
    if termination.stick in ("rigid", "spring") and spec.xi > 0:
        stiffness /= spec.xi  # full strain: E cos^2(a) r
    kappa_c = sigma_u / stiffness if stiffness > 0 else 0.0
    spp = _odd_mesh(termination.segments_per_pitch, n_w)

    def result(
        stress: FloatArray,
        applied: bool,
        a_max: float,
        layer_m: float,
        reach: float,
        note: str,
    ) -> TerminationIncrement:
        return TerminationIncrement(
            stress_Pa=stress,
            applied=applied,
            directions_rad=tuple(float(d) for d in directions),
            principal_fraction=float(fraction),
            kappa_c_1_m=float(kappa_c),
            sigma_u_Pa=float(sigma_u),
            pitch_m=float(pitch),
            a_max=float(a_max),
            boundary_layer_m=float(layer_m),
            reach_m=float(reach),
            segments_per_pitch=spp,
            note=note,
        )

    if termination.anchor == "none":
        return result(zero, False, 0.0, 0.0, 0.0, "anchor='none': no correction")
    g = (
        np.ones(n_w)
        if spec.shape_g is None
        else np.asarray(spec.shape_g, dtype=np.float64)
    )
    if not (kappa_c > 0 and p_ref > 0 and np.any(g > 0)):
        return result(
            zero,
            False,
            0.0,
            0.0,
            0.0,
            "no friction-setting pressure or friction: no clamp effect",
        )
    if contact_pressure_Pa is None:
        rho_samples = np.ones(n_t)
    else:
        pressure = np.asarray(contact_pressure_Pa, dtype=np.float64)
        if pressure.shape != kx.shape or not np.isfinite(pressure).all():
            raise CableInputError("contact pressure must match the curvature history")
        rho_samples = pressure / p_ref

    if termination.solver == "along_wire":
        loads = [kx, ky]
    else:
        loads = [kx * math.cos(d) + ky * math.sin(d) for d in directions]
    shape_max = termination.shape_maximum()
    g_min = float(np.min(g[g > 0]))
    peak = (
        float(np.max(np.hypot(kx, ky)))
        if termination.solver == "along_wire"
        else max(float(np.max(np.abs(load))) for load in loads)
    )
    a_max = peak * shape_max / (kappa_c * g_min)
    layer_m = float(clamp_boundary_layer_length(max(a_max, math.pi / 2), pitch))
    reach = max(termination.reach_lengths * layer_m, pitch)
    span = termination.setback_m + termination.distance_m
    if span >= reach:
        return result(
            zero,
            False,
            a_max,
            layer_m,
            reach,
            f"section {span:.3g} m from the anchor, beyond the solved reach "
            f"{reach:.3g} m: sectional result unchanged",
        )

    h = 2 * math.pi / spp
    h_x = pitch / spp
    j_s = int(math.floor(span / h_x))
    n_far = int(math.ceil(reach / h_x))
    sense = lay_hand * termination.axis_sign
    x_mid = np.asarray(
        termination.distance_m + (np.arange(j_s + 1 + n_far) - j_s) * h_x,
        dtype=np.float64,
    )
    shape = termination.shape(x_mid)
    n_ref = 2 * n_far + spp
    ref_shape = np.ones(n_ref)
    ref_steps = (sense * (np.arange(n_ref) - n_far)) % spp
    ref_index = np.arange(n_far, n_far + spp, dtype=np.int64)
    anchored_steps = sense * (np.arange(x_mid.size) - j_s)
    stress = np.zeros((n_w, n_t), dtype=np.float64)
    # Wire k sits at phase 2 pi k / n_w = (whole_k + part_k / n_w) h at the
    # section; wires sharing part_k (and g_k) share one far-field run, and
    # every phase is formed as base + integer * h in both runs, so the
    # uniform-curvature parts of the two runs are bitwise identical.
    whole = (np.arange(n_w) * spp) // n_w
    part = (np.arange(n_w) * spp) % n_w

    field = termination.curvature_field
    if field is not None:
        if field.n_samples != n_t:
            raise CableInputError(
                "the curvature field needs one column per history sample"
            )
        if termination.solver == "clamp" and termination.biaxial != "principal":
            raise CableInputError(
                "a curvature field is planar: use biaxial='principal'"
            )
        plane = kx * math.cos(beta) + ky * math.sin(beta)
        here = field.at(termination.distance_m)
        tol = 1e-6 * max(
            float(np.max(np.abs(here))), float(np.max(np.abs(plane))), 1e-300
        )
        if (
            float(np.max(np.abs(np.hypot(kx, ky) - np.abs(here)))) > tol
            or float(np.max(np.abs(plane - here))) > tol
        ):
            raise CableInputError(
                "the section history must equal the curvature field at distance_m "
                "in the bending plane of the clamp solution"
            )
        stress = _field_increment(
            spec,
            termination,
            field,
            beta,
            plane,
            rho_samples,
            g,
            kappa_c,
            sigma_u,
            spp,
            x_mid,
            j_s,
            n_far,
            sense,
            whole,
            part,
            kx,
            ky,
        )
        return result(
            stress,
            True,
            a_max,
            layer_m,
            reach,
            f"anchored-wire clamp increment under the {field.kind} curvature "
            f"field ({termination.solver} solver)",
        )

    if termination.solver == "along_wire":
        stress = _along_wire_increment(
            spec,
            termination,
            kx,
            ky,
            rho_samples,
            g,
            kappa_c,
            shape_max,
            spp,
            shape,
            j_s,
            n_far,
            sense,
            whole,
            part,
            x_mid,
            h_x,
        )
        return result(
            stress,
            True,
            a_max,
            layer_m,
            reach,
            f"along-wire clamp increment ({termination.stick} stick, capstan "
            f"{'on' if termination.capstan else 'off'}; biaxial strain direct)",
        )

    for direction, load in zip(directions, loads, strict=True):
        if not np.any(load):
            continue
        far: dict[
            tuple[float, int],
            tuple[FloatArray, FloatArray, NDArray[np.bool_], FloatArray, float],
        ] = {}
        paths: dict[float, tuple[FloatArray, FloatArray, NDArray[np.bool_]]] = {}
        for k in range(n_w):
            gk = float(g[k])
            if gk <= 0:
                continue
            if gk not in paths:
                fine, record, interval, weight = _fine_path(
                    load / (kappa_c * gk), shape_max, termination.max_increment
                )
                rho_path = np.concatenate([[rho_samples[0]], rho_samples])
                rho = (
                    rho_path[interval] * (1 - weight) + rho_path[interval + 1] * weight
                )
                paths[gk] = (fine, rho, record)
            fine, rho, record = paths[gk]
            key = (gk, int(part[k]))
            if key not in far:
                # theta measured from the tensile extreme fibre of `direction`
                base = -direction + h * float(part[k]) / n_w
                e_ref = ref_shape * np.cos(base + ref_steps * h)
                rows, _r, _d = anchored_wire_response(
                    e_ref, fine, h, rho=rho, record=record, index=ref_index
                )
                far[key] = (fine, rho, record, rows, base)
            rows, base = far[key][3], far[key][4]
            steps = (int(whole[k]) + anchored_steps) % spp
            e_anchor = shape * np.cos(base + steps * h)
            n_rec, _r, _d = anchored_wire_response(
                e_anchor,
                fine,
                h,
                rho=rho,
                record=record,
                index=np.array([j_s], dtype=np.int64),
            )
            column = (sense * int(whole[k])) % spp
            stress[k] += sigma_u * gk * (n_rec[:, 0] - rows[:, column])
    return result(
        stress,
        True,
        a_max,
        layer_m,
        reach,
        f"anchored-wire clamp increment, {termination.biaxial} mode",
    )


def _field_increment(
    spec: HelicalLayerSpec,
    termination: TerminationSpec,
    field: CurvatureField,
    beta: float,
    plane: FloatArray,
    rho_samples: FloatArray,
    g: FloatArray,
    kappa_c: float,
    sigma_u: float,
    spp: int,
    x_mid: FloatArray,
    j_s: int,
    n_far: int,
    sense: int,
    whole: NDArray[np.int64],
    part: NDArray[np.int64],
    kx: FloatArray,
    ky: FloatArray,
) -> FloatArray:
    """Clamp increment under a non-proportional planar field [Pa] (gap P5).

    The anchored chain carries ``kappa(x, t)`` segment by segment; the far
    reference carries the section history uniformly (same solver, same
    substeps), so the increment is the anchorage effect at the section.
    """
    n_w = spec.n_wires
    n_t = plane.size
    h = 2 * math.pi / spp
    local = np.zeros((n_t, x_mid.size))
    inside = x_mid >= 0.0
    if np.any(inside):
        local[:, inside] = field.sample(x_mid[inside]).T
    n_ref = 2 * n_far + spp
    ref_steps = (sense * (np.arange(n_ref) - n_far)) % spp
    ref_index = np.arange(n_far, n_far + spp, dtype=np.int64)
    anchored_steps = sense * (np.arange(x_mid.size) - j_s)
    stress = np.zeros((n_w, n_t))
    if termination.solver == "along_wire":
        return _along_wire_field(
            spec,
            termination,
            local,
            beta,
            kx,
            ky,
            rho_samples,
            g,
            kappa_c,
            spp,
            j_s,
            n_far,
            sense,
            whole,
            part,
        )
    for k in range(n_w):
        gk = float(g[k])
        if gk <= 0:
            continue
        drive = local / (kappa_c * gk)
        far_drive = np.repeat((plane / (kappa_c * gk))[:, None], n_ref, axis=1)
        previous = np.vstack((np.zeros((1, drive.shape[1])), drive[:-1]))
        far_prev = np.vstack((np.zeros((1, n_ref)), far_drive[:-1]))
        step = np.maximum(
            np.max(np.abs(drive - previous), axis=1),
            np.max(np.abs(far_drive - far_prev), axis=1),
        )
        counts = np.maximum(1, np.ceil(step / termination.max_increment)).astype(
            np.int64
        )
        base = -beta + h * float(part[k]) / n_w
        rows, _r, _d = anchored_wire_field_response(
            far_drive,
            np.cos(base + ref_steps * h),
            h,
            max_increment=termination.max_increment,
            rho=rho_samples,
            index=ref_index,
            substeps=counts,
        )
        steps = (int(whole[k]) + anchored_steps) % spp
        n_rec, _r, _d = anchored_wire_field_response(
            drive,
            np.cos(base + steps * h),
            h,
            max_increment=termination.max_increment,
            rho=rho_samples,
            index=np.array([j_s], dtype=np.int64),
            substeps=counts,
        )
        column = (sense * int(whole[k])) % spp
        stress[k] = sigma_u * gk * (n_rec[:, 0] - rows[:, column])
    return stress


def _along_wire_field(
    spec: HelicalLayerSpec,
    termination: TerminationSpec,
    local: FloatArray,
    beta: float,
    kx: FloatArray,
    ky: FloatArray,
    rho_samples: FloatArray,
    g: FloatArray,
    kappa_c: float,
    spp: int,
    j_s: int,
    n_far: int,
    sense: int,
    whole: NDArray[np.int64],
    part: NDArray[np.int64],
) -> FloatArray:
    """Along-wire clamp increment under a planar field [Pa] (gap P5)."""
    layer = WireChainLayer.from_layer_spec(
        spec, stick=termination.stick, capstan=termination.capstan
    )
    n_w = spec.n_wires
    n_t = kx.size
    h = 2 * math.pi / spp
    peak = float(np.max(np.abs(local))) if local.size else 0.0
    if not peak > 0:
        return np.zeros((n_w, n_t))
    path = np.vstack((np.zeros((1, local.shape[1])), local / kappa_c))
    steps = np.max(np.abs(np.diff(path, axis=0)), axis=1)
    counts = np.maximum(1, np.ceil(steps / termination.max_increment)).astype(np.int64)
    interval = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    position = np.arange(interval.size) - starts[interval] + 1
    weight = position / counts[interval]
    record = position == counts[interval]
    fine_local = (
        path[interval] * (1.0 - weight)[:, None] + path[interval + 1] * weight[:, None]
    )
    sec_x = np.concatenate([[0.0], kx / kappa_c])
    sec_y = np.concatenate([[0.0], ky / kappa_c])
    far_x = np.ascontiguousarray(
        (sec_x[interval] * (1.0 - weight) + sec_x[interval + 1] * weight)[:, None]
    )
    far_y = np.ascontiguousarray(
        (sec_y[interval] * (1.0 - weight) + sec_y[interval + 1] * weight)[:, None]
    )
    anc_x = np.ascontiguousarray(fine_local * math.cos(beta))
    anc_y = np.ascontiguousarray(fine_local * math.sin(beta))
    rho_path = np.concatenate([[rho_samples[0]], rho_samples])
    rho = np.ascontiguousarray(
        rho_path[interval] * (1 - weight) + rho_path[interval + 1] * weight
    )
    k_hat = float(layer.k_hat)
    lam = float(layer.lam)
    rigid = layer.rigid

    def run(
        theta: FloatArray,
        fx: FloatArray,
        fy: FloatArray,
        rec: NDArray[np.int64],
    ) -> FloatArray:
        m = theta.size
        w = np.full(m + 1, h)
        w[0] = w[-1] = 0.5 * h
        fixed = np.ones(m + 1, dtype=np.bool_)
        fixed[1:-1] = False
        theta_node = np.r_[theta - 0.5 * sense * h, theta[-1] + 0.5 * sense * h]
        gnode = np.ascontiguousarray(layer.shape_at(theta_node))
        out_n = np.zeros((fx.shape[0], rec.size))
        out_e = np.zeros((fx.shape[0], 4))
        stats = np.zeros(4, dtype=np.int64)
        status = _wire_history(
            fx,
            fy,
            np.ascontiguousarray(np.cos(theta)),
            np.ascontiguousarray(np.sin(theta)),
            gnode,
            rho,
            k_hat,
            lam,
            h,
            w,
            fixed,
            True,
            math.inf,
            rigid,
            rec,
            np.zeros(rec.size),
            out_n,
            out_e,
            stats,
        )
        if status != 0:
            raise CableInputError(
                "along-wire termination increment did not converge at fine "
                f"substep {-status - 1} (statistics {stats.tolist()})"
            )
        return np.asarray(out_n[record])

    n_ref = 2 * n_far + spp
    ref_steps = (sense * (np.arange(n_ref) - n_far)) % spp
    ref_index = np.arange(n_far, n_far + spp, dtype=np.int64)
    anchored_steps = sense * (np.arange(local.shape[1]) - j_s)
    uniform_g = spec.shape_g is None or bool(np.all(g == g[0]))
    stress = np.zeros((n_w, n_t))
    far: dict[tuple[int, int], FloatArray] = {}
    for k in range(n_w):
        if g[k] <= 0 and uniform_g:
            continue
        base = h * float(part[k]) / n_w
        key = (int(part[k]), -1 if uniform_g else k)
        if key not in far:
            far[key] = run(
                np.asarray(base + ref_steps * h, dtype=np.float64),
                far_x,
                far_y,
                ref_index,
            )
        steps_k = (int(whole[k]) + anchored_steps) % spp
        n_anch = run(
            np.asarray(base + steps_k * h, dtype=np.float64),
            anc_x,
            anc_y,
            np.array([j_s], dtype=np.int64),
        )
        column = (sense * int(whole[k])) % spp
        stress[k] = layer.stress_unit_Pa * (n_anch[:, 0] - far[key][:, column])
    return stress


def _along_wire_increment(
    spec: HelicalLayerSpec,
    termination: TerminationSpec,
    kx: FloatArray,
    ky: FloatArray,
    rho_samples: FloatArray,
    g: FloatArray,
    kappa_c: float,
    shape_max: float,
    spp: int,
    shape: FloatArray,
    j_s: int,
    n_far: int,
    sense: int,
    whole: NDArray[np.int64],
    part: NDArray[np.int64],
    x_mid: FloatArray | None = None,
    h_x: float | None = None,
) -> FloatArray:
    """Clamp increment of every wire with the along-wire solver [Pa].

    Same meshes and wire phases as the clamp solver (anchored chain from the
    anchor to the reach, far chain of uniform curvature around the section);
    the imposed strain is biaxial and the bound follows the wire's azimuth
    (times the lateral relief factor when one is given).
    """
    layer = WireChainLayer.from_layer_spec(
        spec, stick=termination.stick, capstan=termination.capstan
    )
    n_w = spec.n_wires
    n_t = kx.size
    h = 2 * math.pi / spp
    scale = float(np.max(np.hypot(kx, ky))) if n_t else 0.0
    if not scale > 0:
        return np.zeros((n_w, n_t))
    # common fine path (the far and anchored runs share every substep)
    fine_x, record, interval, weight = _fine_path(
        kx / kappa_c, shape_max, termination.max_increment
    )
    path_y = np.concatenate([[0.0], ky / kappa_c])
    fine_y = path_y[interval] * (1.0 - weight) + path_y[interval + 1] * weight
    rho_path = np.concatenate([[rho_samples[0]], rho_samples])
    rho = np.ascontiguousarray(
        rho_path[interval] * (1 - weight) + rho_path[interval + 1] * weight
    )
    fx = np.ascontiguousarray(fine_x[:, None])
    fy = np.ascontiguousarray(fine_y[:, None])
    k_hat = float(layer.k_hat)
    lam = float(layer.lam)
    rigid = layer.rigid
    uniform_g = spec.shape_g is None or bool(np.all(g == g[0]))
    relief = termination.lateral_relief
    relief_nodes: tuple[FloatArray, FloatArray, FloatArray] | None = None
    if relief is not None:
        assert x_mid is not None and h_x is not None
        x_node = np.r_[x_mid - 0.5 * h_x, x_mid[-1] + 0.5 * h_x]
        shape_node = np.interp(x_node, x_mid, shape)
        relief_nodes = (
            scale * np.abs(shape_node),
            scale * np.abs(np.gradient(shape_node, h_x)),
            x_node + termination.setback_m,
        )

    def run(
        theta: FloatArray,
        shp: FloatArray,
        rec: NDArray[np.int64],
        anchored: bool = False,
    ) -> FloatArray:
        m = theta.size
        w = np.full(m + 1, h)
        w[0] = w[-1] = 0.5 * h
        fixed = np.ones(m + 1, dtype=np.bool_)
        fixed[1:-1] = False
        theta_node = np.r_[theta - 0.5 * sense * h, theta[-1] + 0.5 * sense * h]
        gnode = np.ascontiguousarray(layer.shape_at(theta_node))
        if relief is not None:
            if anchored:
                assert relief_nodes is not None
                amp, grad, dist = relief_nodes
                gnode = gnode * relief.factor(theta_node, amp, grad, dist)
            else:
                gnode = gnode * relief.factor(theta_node, scale, 0.0, None)
            gnode = np.ascontiguousarray(gnode)
        out_n = np.zeros((fx.shape[0], rec.size))
        out_e = np.zeros((fx.shape[0], 4))
        stats = np.zeros(4, dtype=np.int64)
        status = _wire_history(
            fx,
            fy,
            np.ascontiguousarray(shp * np.cos(theta)),
            np.ascontiguousarray(shp * np.sin(theta)),
            gnode,
            rho,
            k_hat,
            lam,
            h,
            w,
            fixed,
            True,
            math.inf,
            rigid,
            rec,
            np.zeros(rec.size),
            out_n,
            out_e,
            stats,
        )
        if status != 0:
            raise CableInputError(
                "along-wire termination increment did not converge at fine "
                f"substep {-status - 1} (statistics {stats.tolist()})"
            )
        return np.asarray(out_n[record])

    # Phases are base + (integer in [0, spp)) h in both runs, so the
    # uniform-curvature parts are bitwise identical (as for the clamp solver).
    n_ref = 2 * n_far + spp
    ref_steps = (sense * (np.arange(n_ref) - n_far)) % spp
    ref_index = np.arange(n_far, n_far + spp, dtype=np.int64)
    anchored_steps = sense * (np.arange(shape.size) - j_s)
    ref_shape = np.ones(n_ref)
    stress = np.zeros((n_w, n_t))
    far: dict[tuple[int, int], FloatArray] = {}
    for k in range(n_w):
        if g[k] <= 0 and uniform_g:
            continue
        base = h * float(part[k]) / n_w
        key = (int(part[k]), -1 if uniform_g else k)
        if key not in far:
            far[key] = run(
                np.asarray(base + ref_steps * h, dtype=np.float64), ref_shape, ref_index
            )
        steps = (int(whole[k]) + anchored_steps) % spp
        n_anch = run(
            np.asarray(base + steps * h, dtype=np.float64),
            shape,
            np.array([j_s], dtype=np.int64),
            anchored=True,
        )
        column = (sense * int(whole[k])) % spp
        stress[k] = layer.stress_unit_Pa * (n_anch[:, 0] - far[key][:, column])
    return stress
