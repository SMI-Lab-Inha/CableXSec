# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Fixed-material-fibre fatigue under biaxial bending and normal loading.

This module assembles the physical-wire Papailiou neutral-axis slip-front
return map shared with public M-κ and stress. Local wire bending retains both
components in :func:`cablexsec.mechanics.wire_stress.local_bending_fibre_stress`.
Each sampled circumference fibre keeps its material angle throughout the
history. Per-wire bulk damage is the maximum of the separately counted fibre
damages, not rainflow counting of instantaneous perimeter extrema. The contact
indicator is obtained from a two-component Coulomb return map with Mindlin
initial stiffness. The S-N parameters come from
:mod:`cablexsec.materials.sn_curves`.

For an arbitrary irregular curvature/tension history ``kappa_x(t)``,
``kappa_y(t)``, ``N(t)`` the per-wire longitudinal stress is built incrementally
(the rate-independent return map handles reversal without special cases), then:

1. rainflow counting ([ASTM E1049]) → ``(range, mean, count)`` cycles;
2. Miner summation on the unmodified stress range used by the selected S--N
   curve ([Miner1945]);
3. a contact branch selected only when supported for the material:

   * **steel armour** — no steel slip-amplitude life curve is registered
     because the former Périer attribution is not supported by the cited
     source. The slip amplitude ``delta*`` is computed only when a steel
     ``sn_fretting`` curve is supplied; it is then used for that curve's
     damage and discarded. :class:`FatigueResult` has no ``delta*``, ``Phi``
     (shear-slip product ``mu p0 delta*``), frictional-work or Ruiz-type
     field, and no Ruiz parameter is computed here. The internal contact
     update passes no peak pressure, so
     :meth:`~cablexsec.mechanics.mindlin_contact.MindlinContact.shear_slip_product`
     would return zero for those contacts. The ``delta*``, ``Phi``, frictional
     work and Ruiz-type indices ``F1``, ``F2`` of the companion article come
     from a separate steel-on-polymer line-contact model;
   * **copper conductor** — the SWT critical-plane *strain-life* method
     (:mod:`cablexsec.analysis.fretting_swt`, [Poon2022] Eq. 30), because
     [Poon2022] characterises copper fretting by strain-life constants, not a
     slip-amplitude S-N. The contact peak pressure ``p_0`` drives
     ``sigma_max = 2 mu p_0`` -> SWT -> ``N_f``;

4. a termination adjustment from counting SCF-amplified stress ranges at each
   fixed fibre, avoiding duplication of the unamplified bulk term. S-N knees
   and cutoffs are evaluated after amplification.

The decomposition ``sigma_total = sigma_A + sigma_B + sigma_f + sigma_T``
follows [Nasution2014b] Eq. 3 with an explicit thermal extension. ``sigma_T``
is included when the caller supplies a layer history; otherwise the run is
explicitly isothermal.

References
----------
[ASTM E1049]; [Miner1945]; [Nasution2014b] Eqs. 3, 5; [Poon2022];
[MenardCartraud2023] Eq. 14; [Saevik2011] Eq. 89.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import rainflow
from numpy.typing import NDArray

from cablexsec._numba import HAVE_NUMBA
from cablexsec.analysis.fast_rainflow import miner_rows
from cablexsec.analysis.fretting_swt import (
    PartialSlipCycles,
    swt_driver_from_contact,
    swt_life,
    swt_slip_cycle_damage,
)
from cablexsec.analysis.sn_assessment import (
    ComponentSCF,
    corrected_damage_rows,
    needs_correction,
    scaled_component,
)
from cablexsec.analysis.termination import (
    TerminationIncrement,
    TerminationSpec,
    termination_friction_increment,
    warn_if_clamped,
)
from cablexsec.exceptions import CableInputError
from cablexsec.materials.mean_stress import MeanStressSpec
from cablexsec.materials.sn_curves import SNcurve, SWTLifeConstants
from cablexsec.mechanics.distributed_friction import (
    BOUND_FORMS,
    DEFAULT_MAX_INCREMENT,
    DEFAULT_NODES_PER_WIRE,
    FRICTION_MODELS,
    distributed_wire_friction_stress,
)
from cablexsec.mechanics.fast_slip import wire_friction_stress
from cablexsec.mechanics.friction_law import (
    FrictionLaw,
    active_friction_law,
    friction_law_time_steps,
)
from cablexsec.mechanics.mindlin_contact import (
    MindlinContact,
    TransverseMap,
    geometric_slip_factor,
    slip_increment,
    transverse_map_or_warn,
)
from cablexsec.mechanics.mkappa import HelicalLayerSpec
from cablexsec.mechanics.papailiou import DIRECTION_HOLD_FRACTION, wire_slip_history
from cablexsec.mechanics.progressive_friction import (
    ProgressiveFriction,
    progressive_wire_friction_stress,
    progressive_wire_history,
)
from cablexsec.mechanics.tendon_bending import LateralRelief, TransverseSlipModel
from cablexsec.mechanics.tendon_section import WireSection
from cablexsec.mechanics.wire_stress import local_bending_fibre_stress
from cablexsec.provenance import (
    DirectionTiedFrictionWarning,
    FrictionModelFallbackWarning,
    Provenance,
    real_array,
)

# The compiled kernels are selected only when numba is active (module
# attributes, so that tests can force the reference paths).
_FAST_SLIP = HAVE_NUMBA
_FAST_RAINFLOW = HAVE_NUMBA

RainflowCounter = Literal["auto", "compiled", "reference"]
FrictionModel = Literal["auto", "direction_tied", "distributed", "progressive"]
#: Armour friction model of :func:`compute_fatigue` when ``friction_model`` is
#: left at ``None`` (the planned change to ``"auto"`` is announced by
#: :class:`~cablexsec.provenance.DirectionTiedFrictionWarning`).
DEFAULT_FRICTION_MODEL: FrictionModel = "direction_tied"
DistributedBound = Literal["signed", "uniform"]
#: Out-of-plane ratio (:func:`curvature_path_out_of_plane_ratio`) at or below
#: which ``friction_model="auto"`` treats a curvature path as planar (and above
#: which the default direction-tied law warns).
PLANAR_PATH_TOLERANCE = 1e-3
#: Smallest range about the mean, relative to the largest curvature, against
#: which :func:`curvature_path_out_of_plane_ratio` judges a path. Six
#: significant digits put each component within 5e-6 of the largest
#: curvature (half a unit in the last digit of a mantissa near 1), so a
#: rounded static curvature is at most 7.1e-6 off its line and 7.1e-4 out of
#: plane against this floor; cycles normal to a static bend count from 1e-5
#: of it.
_PATH_SPREAD_FLOOR = 1e-2
#: Measured bias of the direction-tied law under rotating curvature, quoted by
#: :class:`~cablexsec.provenance.DirectionTiedFrictionWarning`.
DIRECTION_TIED_ROTATION_BIAS = (
    "under a steadily rotating curvature at twice the slip onset its largest "
    "per-wire friction-stress range was 0.94 (template interface, xi = 0.81) "
    "and 0.86 (near-rigid interface, xi = 0.99) of the distributed helix "
    "model on the 66 kV template inner armour, i.e. bulk damage about 0.79 "
    "and 0.55 of it at an S-N slope of 4 where the friction range governs; "
    "on planar and flat paths (minor over major axis up to about 0.3) its "
    "ranges are at or above the distributed ones"
)
CopperContactModel = Literal["slip_cycle", "legacy_gross_sliding"]
COPPER_CONTACT_MODELS: tuple[str, ...] = ("slip_cycle", "legacy_gross_sliding")

__all__ = [
    "DEFAULT_FRICTION_MODEL",
    "PLANAR_PATH_TOLERANCE",
    "FatigueLayer",
    "FatigueResult",
    "WireStressField",
    "wire_stress_field",
    "compute_fatigue",
    "curvature_path_out_of_plane_ratio",
    "miner_damage",
    "rainflow_cycles",
    "select_wires",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

#: Wires to evaluate: ``N`` equally spaced positions per layer, explicit
#: azimuths [deg], or one of these per layer name.
WirePositions = int | Sequence[float] | Mapping[str, int | Sequence[float]]


def select_wires(n_wires: int, positions: int | Sequence[float]) -> IntArray:
    """Indices of the wires at the selected positions of a layer, ascending.

    Wire ``k`` lies at the azimuth ``360 k / n_wires`` degrees (the azimuth of
    ``phi_critical_deg``). An integer ``N`` (1 to ``n_wires``) takes the wires
    nearest to ``N`` equally spaced azimuths from zero; a sequence takes the
    wire nearest to each azimuth [deg]. A wire named twice is evaluated once.
    """
    n = int(n_wires)
    pitch = 360.0 / n
    if isinstance(positions, (bool, np.bool_)):
        raise CableInputError("wire positions must be a count or azimuths in degrees")
    if isinstance(positions, (int, np.integer)):
        if not 1 <= positions <= n:
            raise CableInputError(
                f"the number of wire positions must lie between 1 and {n}"
            )
        angles = np.arange(int(positions)) * (360.0 / int(positions))
    else:
        try:
            angles = np.asarray(positions, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise CableInputError(
                "wire positions must be a count or azimuths in degrees"
            ) from error
        if angles.ndim != 1 or angles.size == 0 or not np.isfinite(angles).all():
            raise CableInputError(
                "wire azimuths must be a non-empty sequence of finite degrees"
            )
    index = np.floor(np.mod(angles, 360.0) / pitch + 0.5).astype(np.int64) % n
    return np.asarray(np.unique(index), dtype=np.int64)


def _layer_wires(
    positions: WirePositions | None, layer: FatigueLayer
) -> IntArray | None:
    """The evaluated wires of ``layer`` (``None``: every wire)."""
    if positions is None:
        return None
    if isinstance(positions, Mapping):
        positions = positions[layer.name]
    wires = select_wires(layer.spec.n_wires, positions)
    return None if wires.size == layer.spec.n_wires else wires


def curvature_path_out_of_plane_ratio(
    kappa_x: FloatArray, kappa_y: FloatArray
) -> float:
    """Out-of-plane extent of a curvature path, relative to its curvature range.

    A planar history keeps one bending plane: every sample ``(kappa_x,
    kappa_y)`` lies on one line through the origin. The line is the principal
    axis of the second moment ``sum k k^T`` of the samples about the origin
    (not about their mean: the bending plane passes through the straight
    cable). The ratio is the largest distance of a sample from that line over
    the smaller of the largest curvature magnitude and the largest distance
    of a sample from the mean curvature: zero for a planar path in any plane,
    a static offset in the plane included, one for a circle, ``b/a`` for an
    ellipse of semi-axes ``a >= b`` centred on the origin. The second scale
    keeps a static bend from hiding the cycles about it: a static curvature
    with small cycles normal to its plane is as far from planar as the cycles
    are large, not as small as they are against the static bend. A range
    below ``1e-2`` of the largest curvature is judged against that floor, so
    that the rounding of a nearly constant curvature in output files (six
    significant digits: at most ``7.1e-4``) stays planar; cycles normal to a
    static bend then count from ``1e-5`` of it. A zero history returns zero.
    """
    kx = np.asarray(kappa_x, dtype=np.float64)
    ky = np.asarray(kappa_y, dtype=np.float64)
    if kx.shape != ky.shape:
        raise CableInputError("kappa_x and kappa_y must have the same shape")
    peak = float(np.max(np.hypot(kx, ky))) if kx.size else 0.0
    if peak == 0.0 or not math.isfinite(peak):
        return 0.0
    # Scale first: the second moment of tiny or huge curvatures stays finite.
    x, y = kx / peak, ky / peak
    sxx, syy, sxy = float(x @ x), float(y @ y), float(x @ y)
    angle = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    distance = float(np.max(np.abs(y * math.cos(angle) - x * math.sin(angle))))
    # The range about the mean (in units of the peak): a static bend with
    # cycles normal to its plane is judged against the cycles.
    spread = float(np.max(np.hypot(x - np.mean(x), y - np.mean(y))))
    return distance / min(1.0, max(spread, _PATH_SPREAD_FLOOR))


def _distributed_spec(
    spec: HelicalLayerSpec,
    stick_modulus_Pa: FloatArray | None,
    friction_law: FrictionLaw | None,
) -> tuple[HelicalLayerSpec | None, str]:
    """Layer for the distributed model, or ``None`` and why the model does not apply.

    A constant stick modulus is the Yun coefficient ``xi = k / (E cos^2 a)``
    of that state (a thermal block recomputes ``xi`` from its moduli), so the
    layer is returned with that coefficient. A stick modulus that changes
    over the record and a rate-dependent friction law are not covered by the
    distributed model; a rigid interface (``xi >= 1``) it represents only by
    the along-wire penalty ``RIGID_K_HAT``, so the exact direction-tied cap
    is kept for it.
    """
    law = active_friction_law(friction_law)
    if law is not None and law.rate_dependent:
        return None, "the friction law is rate dependent"
    if stick_modulus_Pa is not None:
        stick = np.asarray(stick_modulus_Pa, dtype=np.float64)
        if stick.size and not np.allclose(stick, stick[0], rtol=1e-12, atol=0.0):
            return None, "the stick modulus changes over the record"
        if stick.size and not np.allclose(stick, spec.k_f, rtol=1e-12, atol=0.0):
            spec = dataclasses.replace(
                spec,
                xi=float(stick[0])
                / (spec.E_wire_Pa * math.cos(spec.lay_angle_rad) ** 2),
            )
    if not (math.isfinite(spec.xi) and 0.0 <= spec.xi < 1.0):
        return None, f"the interface is rigid (xi = {spec.xi:.6g})"
    return spec, ""


def _resolve_friction_model(
    friction_model: str,
    spec: HelicalLayerSpec,
    out_of_plane: float,
    stick_modulus_Pa: FloatArray | None,
    friction_law: FrictionLaw | None,
) -> tuple[str, str]:
    """Friction model of one layer and, for ``"auto"``, why it fell back."""
    _check_friction_model(friction_model)
    if friction_model != "auto":
        return friction_model, ""
    if out_of_plane <= PLANAR_PATH_TOLERANCE:
        return "direction_tied", ""
    usable, reason = _distributed_spec(spec, stick_modulus_Pa, friction_law)
    if usable is None:
        return "direction_tied", reason
    return "distributed", ""


def _warn_auto_fallback(name: str, reason: str, stacklevel: int) -> None:
    """Warn that ``friction_model="auto"`` kept the direction-tied law."""
    warnings.warn(
        f"friction_model='auto' kept the direction-tied model for {name} on a "
        f"non-planar curvature path because {reason}; the distributed model does "
        "not cover that case, and the direction-tied cap turns with the bending "
        f"direction: {DIRECTION_TIED_ROTATION_BIAS}",
        FrictionModelFallbackWarning,
        stacklevel=stacklevel + 1,
    )


def _warn_default_direction_tied(out_of_plane: float, stacklevel: int) -> None:
    """Warn that the default direction-tied law runs on a non-planar path."""
    if out_of_plane <= PLANAR_PATH_TOLERANCE:
        return
    warnings.warn(
        "the default friction_model will change from 'direction_tied' to 'auto' "
        "in the next release; pass friction_model explicitly to keep the current "
        "behaviour. This curvature path is not planar (out-of-plane ratio "
        f"{out_of_plane:.3g} > {PLANAR_PATH_TOLERANCE:g}), so 'auto' would "
        "solve the distributed helix model on every layer it covers. The "
        "direction-tied cap turns "
        f"with the bending direction: {DIRECTION_TIED_ROTATION_BIAS}.",
        DirectionTiedFrictionWarning,
        stacklevel=stacklevel + 1,
    )


#: Values within this relative distance of the largest are a tie: the
#: critical fibre, wire and layer are the first of them, so a round-off
#: difference (compiled against reference kernels, or symmetric wires) does
#: not move the reported location.
_TIE_RTOL = 1e-12


def _first_maximum(values: FloatArray) -> NDArray[np.intp]:
    """Index of the first value of each row within ``_TIE_RTOL`` of its maximum."""
    peak = np.max(values, axis=1, keepdims=True)
    near = values >= peak - _TIE_RTOL * np.abs(peak)
    return np.asarray(np.argmax(near, axis=1))


def _check_distributed_bound(bound: str) -> None:
    if bound not in BOUND_FORMS:
        raise CableInputError(
            f"distributed_bound must be one of {BOUND_FORMS}, got {bound!r}"
        )


def _validate_tension(
    tension_N: FloatArray | None,
    EA_full_N: float | None,
    shape: tuple[int, ...],
) -> None:
    if tension_N is None:
        return
    tension = real_array(tension_N, "tension_N")
    if tension.shape != shape:
        raise CableInputError(
            "tension_N must have the same shape as curvature histories"
        )
    if not np.isfinite(tension).all():
        raise CableInputError("tension_N must be finite")
    if EA_full_N is None or not math.isfinite(EA_full_N) or EA_full_N <= 0:
        raise CableInputError("tension_N requires finite positive EA_full_N")


def rainflow_cycles(
    sigma_series_Pa: FloatArray,
) -> Iterator[tuple[float, float, float, int, int]]:
    """Stress-range cycles, retaining residual half cycles for short records.

    NumPy first removes monotonic interior samples, preserving the last index
    of each turning plateau. The upstream counter then handles the much shorter
    turning-point record. Cycle indices are mapped back to the original record.
    The upstream counter omits a two-sample open excursion, which is retained
    here explicitly. Zero-range entries are omitted from physical cycle counts.
    """
    s = real_array(sigma_series_Pa, "stress history")
    if s.ndim != 1 or not np.isfinite(s).all():
        raise CableInputError("stress history must be one-dimensional and finite")
    if s.size < 2:
        return
    if s.size == 2:
        span = float(abs(s[1] - s[0]))
        if span > 0:
            yield span, float(0.5 * s[0] + 0.5 * s[1]), 0.5, 0, 1
        return
    changes = np.flatnonzero(s[1:] != s[:-1])
    if changes.size == 0:
        return
    rising = s[changes + 1] > s[changes]
    turning = changes[1:][rising[1:] != rising[:-1]]
    indices = np.concatenate(([0], turning, [s.size - 1]))
    if indices.size == 2:
        # Monotonic histories reduce to a single residual half cycle. Use the
        # upstream arithmetic here so ordinary output remains bitwise equal.
        first, last = float(s[0]), float(s[-1])
        yield abs(last - first), 0.5 * (first + last), 0.5, 0, s.size - 1
        return
    for span, mean, count, start, end in rainflow.extract_cycles(s[indices].tolist()):
        if span > 0:
            yield span, mean, count, int(indices[start]), int(indices[end])


# --------------------------------------------------------------------------- #
# Bulk damage: rainflow + Miner (the testable kernel, gate A12)                #
# --------------------------------------------------------------------------- #
def miner_damage(
    sigma_series_Pa: FloatArray,
    sn_curve: SNcurve,
    sigma_UTS_Pa: float,
    *,
    stress_range_factor: float = 1.0,
) -> float:
    """Miner damage of a stress history against a stress-range S-N curve.

    Rainflow-counts ``sigma_series_Pa`` ([ASTM E1049]) and sums
    ``count/N_f(Delta sigma)`` ([Miner1945]). The registered cable-wire curves
    are defined directly in stress range. No generic Goodman transformation is
    imposed: the spiral-strand evidence reviewed by [Saevik2011] and
    [AlaniRaoof1997] does not show improved collapse under that correction.
    A correction chosen by the user (``FatigueLayer.mean_stress``) is applied
    by :func:`cablexsec.analysis.sn_assessment.corrected_damage` instead.
    ``sigma_UTS_Pa`` is retained for API compatibility. A positive constant
    ``stress_range_factor`` scales counted ranges before S-N evaluation (e.g.
    a termination SCF), including crossing a knee or fatigue cutoff. It does
    not scale already accumulated damage using a single S-N slope.
    """
    _ = sigma_UTS_Pa
    if sn_curve.stress_unit != "MPa":
        raise CableInputError("Miner stress damage requires an S-N curve in MPa")
    if not math.isfinite(stress_range_factor) or stress_range_factor <= 0:
        raise CableInputError("stress_range_factor must be finite and positive")
    damage = 0.0
    for rng_Pa, _mean_Pa, count, _i0, _i1 in rainflow_cycles(sigma_series_Pa):
        stress_range = rng_Pa / 1.0e6 * stress_range_factor
        if not math.isfinite(stress_range) or stress_range <= 0:
            raise CableInputError("amplified stress range exceeds floating-point range")
        n_f = sn_curve.cycles_to_failure(stress_range)
        damage += count / n_f
    if not math.isfinite(damage):
        raise CableInputError("accumulated damage exceeds floating-point range")
    return damage


def _use_compiled_counter(rainflow_counter: str) -> bool:
    if rainflow_counter == "auto":
        return _FAST_RAINFLOW
    if rainflow_counter == "compiled":
        return True
    if rainflow_counter == "reference":
        return False
    raise CableInputError("rainflow_counter must be 'auto', 'compiled' or 'reference'")


def _miner_damage_rows(
    histories_Pa: FloatArray,
    sn_curve: SNcurve,
    sigma_UTS_Pa: float,
    stress_range_factor: float,
    compiled: bool,
) -> FloatArray:
    """:func:`miner_damage` of every row of a (wire, time) stress array.

    The compiled counter reproduces the reference cycle for cycle and in the
    same summation order. Any row where the reference would raise is rerun on
    the reference path, so errors are identical as well.
    """
    if not compiled:
        return np.array(
            [
                miner_damage(
                    row,
                    sn_curve,
                    sigma_UTS_Pa,
                    stress_range_factor=stress_range_factor,
                )
                for row in histories_Pa
            ],
            dtype=np.float64,
        )
    if sn_curve.stress_unit != "MPa":
        raise CableInputError("Miner stress damage requires an S-N curve in MPa")
    if not math.isfinite(stress_range_factor) or stress_range_factor <= 0:
        raise CableInputError("stress_range_factor must be finite and positive")
    histories = np.asarray(histories_Pa, dtype=np.float64)
    if histories.ndim != 2 or not np.isfinite(histories).all():
        raise CableInputError("stress history must be one-dimensional and finite")
    damage, rerun = miner_rows(histories, sn_curve, stress_range_factor)
    for k in np.flatnonzero(rerun):
        damage[k] = miner_damage(
            histories[k],
            sn_curve,
            sigma_UTS_Pa,
            stress_range_factor=stress_range_factor,
        )
    return damage


# --------------------------------------------------------------------------- #
# Per-layer fatigue specification                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FatigueLayer:
    """One helical layer for the fatigue pipeline (wraps the M-κ frozen spec).

    Parameters
    ----------
    spec : HelicalLayerSpec
        The frozen M-kappa layer (geometry, ``k_f`` and combined face-pressure
        shape). Its friction-cap shape is not the shared crossing pressure.
    sn_bulk : SNcurve
        Bulk longitudinal-stress S-N curve (stress unit MPa).
    sigma_UTS_Pa : float
        Ultimate tensile strength [Pa], retained for API compatibility.
    sn_fretting : SNcurve or None, optional
        Steel slip-amplitude fretting S-N curve (stress unit ``um``); ``None``
        disables the δ*-S-N fretting branch.
    K_t_contact_N_per_m : float, optional
        Area-equivalent tangential contact stiffness [N/m]; ``0`` disables the
        δ*-S-N fretting branch.
    fretting_mu : float, optional
        Friction coefficient at the wire-wire contact (Coulomb element and the
        SWT driver ``2 mu p_0``).
    swt_const : SWTLifeConstants or None, optional
        Copper SWT strain-life constants; if set (with ``contact_p0_Pa``), the
        copper SWT fretting branch is used *instead of* the δ*-S-N branch.
    contact_p0_Pa : float, optional
        Reference peak Hertz contact pressure at the wire-wire contact [Pa]
        (mean crossing force); per-wire values use contact_shape_g, independently
        of the section friction-cap distribution.
    contact_shape_g : numpy.ndarray or None
        Prescribed shared-interface force ratios at the wire positions, with
        unit mean. None means uniform. For fixed geometry Hertz pressure and
        tangential stiffness both scale with its cube root. This is not a
        pressure redistribution calculated from the core geometry.
    above_core_mask : numpy.ndarray or None
        Geometrical radial-ray/core-intersection flags at the wire positions.
        Without these flags the critical-wire core location is unclassified;
        a pressure maximum is not used as a substitute for geometry.
    contact_nu : float, optional
        Poisson's ratio of the wire material for the SWT surface strain.
    E_eff_axial_Pa : float or None, optional
        Axial modulus for the tension stress ``sigma_A``; defaults to the M-κ
        spec wire modulus.
    SCF : float, optional
        Termination stress-concentration factor (default 1.0 = none).
    termination : bool, optional
        If ``True``, evaluate the S-N curve at SCF-amplified cycle ranges.
        The reported wire adjustment is amplified maximum minus bulk maximum.
        The friction boundary layer of anchored armour is a separate option
        (``compute_fatigue(..., termination=TerminationSpec(...))``).
    transverse_ratio : float, optional
        Ratio ``eta`` of the transverse contact slip. The established
        loxodromic model (``eta=0``) is the default; it is unaffected by
        ``transverse_map``.
    transverse_map : {"geodesic", "legacy"} or None, optional
        Transverse map used when ``eta > 0`` (see
        :func:`cablexsec.mechanics.mindlin_contact.slip_increment`).
        ``"geodesic"`` scales the frictionless geodesic transverse slip of
        Out and von Morgen (1997), in phase with the longitudinal slip,
        ``0 <= eta <= 1``. ``"legacy"`` is the former map, in quadrature with
        the geodesic slip. ``None`` with ``eta > 0`` selects the legacy map
        and issues ``TransverseMapWarning``.
    lay_hand : int, optional
        ``+1`` right-hand lay (default), ``-1`` left-hand lay; used only by the
        termination (clamp) correction to set the sense in which a wire turns
        along the cable.
    mean_stress : MeanStressSpec or None, optional
        Mean-stress correction of the bulk cycles
        (:mod:`cablexsec.materials.mean_stress`); ``None`` (default) counts
        the unmodified ranges.
    component_scf : ComponentSCF or None, optional
        Stress concentration factors of the axial, friction and both local
        bending parts, mean and dynamic
        (:class:`~cablexsec.analysis.sn_assessment.ComponentSCF`).
    thickness_m : float or None, optional
        Thickness for the curve's thickness exponent (default: the radial
        thickness of the wire). Used only by a curve with an exponent.
    """

    spec: HelicalLayerSpec
    sn_bulk: SNcurve
    sigma_UTS_Pa: float
    name: str = "layer"
    sn_fretting: SNcurve | None = None
    K_t_contact_N_per_m: float = 0.0
    fretting_mu: float = 0.0
    swt_const: SWTLifeConstants | None = None
    contact_p0_Pa: float = 0.0
    crossing_density_per_m: float = 0.0
    contact_nu: float = 0.34
    E_eff_axial_Pa: float | None = None
    SCF: float = 1.0
    termination: bool = False
    transverse_ratio: float = 0.0
    transverse_map: TransverseMap | None = None
    contact_line_load_N_per_m: float | None = None
    contact_shape_g: FloatArray | None = None
    above_core_mask: NDArray[np.bool_] | None = None
    lay_hand: int = 1
    mean_stress: MeanStressSpec | None = None
    component_scf: ComponentSCF | None = None
    thickness_m: float | None = None

    def __post_init__(self) -> None:
        if self.sn_bulk.stress_unit != "MPa":
            raise CableInputError("bulk fatigue requires an S-N curve in MPa")
        if self.mean_stress is not None and not isinstance(
            self.mean_stress, MeanStressSpec
        ):
            raise CableInputError("mean_stress must be a MeanStressSpec")
        if self.component_scf is not None and not isinstance(
            self.component_scf, ComponentSCF
        ):
            raise CableInputError("component_scf must be a ComponentSCF")
        if self.thickness_m is not None and not (
            math.isfinite(self.thickness_m) and self.thickness_m > 0
        ):
            raise CableInputError("thickness_m must be finite and positive")
        if self.lay_hand not in (1, -1):
            raise CableInputError("lay_hand must be +1 or -1")
        if self.sn_fretting is not None and self.sn_fretting.stress_unit != "um":
            raise CableInputError("slip-amplitude fatigue requires an S-N curve in um")
        if not math.isfinite(self.SCF) or self.SCF <= 0:
            raise CableInputError("SCF must be finite and positive")
        transverse_map_or_warn(self.transverse_ratio, self.transverse_map, stacklevel=4)
        if self.contact_shape_g is not None:
            shape = np.asarray(self.contact_shape_g, dtype=float)
            if (
                shape.shape != (self.spec.n_wires,)
                or not np.isfinite(shape).all()
                or np.any(shape < 0)
                or not np.isclose(np.mean(shape), 1.0, rtol=0, atol=1e-10)
            ):
                raise CableInputError(
                    "contact_shape_g must contain finite nonnegative per-wire "
                    "ratios with unit mean"
                )
            shape = shape.copy()
            shape.flags.writeable = False
            object.__setattr__(self, "contact_shape_g", shape)
        if self.above_core_mask is not None:
            mask = np.asarray(self.above_core_mask)
            if mask.shape != (self.spec.n_wires,) or mask.dtype != np.bool_:
                raise CableInputError(
                    "above_core_mask must contain one Boolean per wire"
                )
            mask = mask.copy()
            mask.flags.writeable = False
            object.__setattr__(self, "above_core_mask", mask)


@dataclass(frozen=True)
class FatigueResult:
    """Per-wire fatigue damage and fixed-fibre bulk damage.

    ``D_bulk_per_fibre`` arrays have shape (wire, circumference fibre). Their
    maximum over the second axis gives ``D_bulk_per_wire``. ``fibre_angles_deg``
    identifies the fixed material locations, starting at the outward radial
    fibre. The critical fibre angles concern bulk damage. Contact-life terms
    remain wire-level indicators and do not establish a common surface point
    for a combined bulk/contact life prediction.

    ``D_bulk_with_termination_per_fibre`` retains the separately evaluated
    SCF-adjusted bulk damage. Its governing fibre can differ from unamplified
    bulk for a bilinear curve or a fatigue cutoff. ``D_term_per_wire`` is the
    difference between the two wire maxima, not a common-point damage field.
    For an explicitly supplied factor below one this adjustment can be negative.

    ``clamp_correction`` summarises, per layer, the anchored-wire friction
    correction requested with ``termination`` (applied or not, ``a_max``,
    boundary layer, reach, bending directions and principal variance
    fraction). The corrected friction stress is already in every bulk damage
    field; it is empty when no termination was requested.

    ``wire_index`` names, per layer, the wires a selected-position evaluation
    (``wire_positions``) assessed: row ``i`` of that layer's per-wire arrays
    is wire ``wire_index[layer][i]``, at the azimuth ``wire_azimuth_deg``
    (``360 k / n_wires`` degrees). A layer absent from them was assessed at
    every wire.
    """

    D_bulk_per_wire: dict[str, FloatArray]
    D_fretting_per_wire: dict[str, FloatArray]
    D_term_per_wire: dict[str, FloatArray]
    D_total_per_wire: dict[str, FloatArray]
    phi_critical_deg: float
    critical_layer: str
    above_core: bool | None
    D_fretting_fraction: float
    governing: str = field(default="bulk")
    provenance: Provenance | None = None
    fibre_angles_deg: FloatArray = field(default_factory=lambda: np.empty(0))
    D_bulk_per_fibre: dict[str, FloatArray] = field(default_factory=dict)
    fibre_critical_deg_per_wire: dict[str, FloatArray] = field(default_factory=dict)
    fibre_critical_deg: float = math.nan
    D_bulk_with_termination_per_fibre: dict[str, FloatArray] = field(
        default_factory=dict
    )
    termination_fibre_critical_deg_per_wire: dict[str, FloatArray] = field(
        default_factory=dict
    )
    clamp_correction: dict[str, dict[str, float | bool | str]] = field(
        default_factory=dict
    )
    wire_index: dict[str, IntArray] = field(default_factory=dict)
    wire_azimuth_deg: dict[str, FloatArray] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Stress time series per wire (consumes the frozen core)                       #
# --------------------------------------------------------------------------- #
def _wire_stress_series(
    layer: FatigueLayer,
    kappa_x: FloatArray,
    kappa_y: FloatArray,
    tension_N: FloatArray | None,
    EA_full_N: float | None,
    thermal_stress_Pa: FloatArray | None,
    contact_pressure_Pa: FloatArray | None,
    *,
    axial_strain: FloatArray | None = None,
    normal_stress_Pa: FloatArray | None = None,
    wire_modulus_Pa: FloatArray | None = None,
    stick_modulus_Pa: FloatArray | None = None,
    direction_hold_fraction: float = DIRECTION_HOLD_FRACTION,
    friction_model: FrictionModel = "direction_tied",
    distributed_nodes_per_pitch: int | None = None,
    distributed_max_increment: float = DEFAULT_MAX_INCREMENT,
    progressive: ProgressiveFriction | None = None,
    friction_out: list[FloatArray] | None = None,
    friction_law: FrictionLaw | None = None,
    time_s: FloatArray | None = None,
    distributed_bound: DistributedBound = "signed",
    wires: IntArray | None = None,
) -> FloatArray:
    """Build ``sigma_total(phi_k, t)`` for every wire of one layer [Pa].

    ``wires`` (indices) builds those rows only, in that order, equal to the
    same rows of the full field: the direction-tied kernel evaluates only
    them; the other friction models solve the layer and keep their rows.

    ``friction_out``, when given, receives the friction stress ``sigma_f``
    (for the component stress concentration factors). ``friction_law`` and
    ``time_s`` (:class:`~cablexsec.mechanics.friction_law.FrictionLaw`) pass to
    the direction-tied and distributed kernels; the progressive law refuses a
    non-neutral law.

    Shape ``(n_wires, n_steps)``. ``sigma_total = sigma_A + sigma_B + sigma_f``.
    Curvatures are axial-strain-gradient components, not rotation-curvature
    components. A prescribed axial_strain is the stress-driving uniaxial
    strain, an alternative to true material force/EA, not geometric total
    strain including an unaccounted Poisson contribution.
    A solved normal_stress_Pa history instead retains the layer's extension,
    twist, pressure and thermal response, without separate axial/thermal terms.
    Optional moduli carry material and shear-support changes between steps.
    """
    spec = layer.spec
    if normal_stress_Pa is not None and any(
        value is not None for value in (tension_N, axial_strain, thermal_stress_Pa)
    ):
        raise CableInputError(
            "supply solved normal stress or separate axial/thermal loads"
        )
    for label, history in (
        ("normal stress", normal_stress_Pa),
        ("thermal stress", thermal_stress_Pa),
        ("wire modulus", wire_modulus_Pa),
        ("stick modulus", stick_modulus_Pa),
    ):
        if history is not None:
            values = np.asarray(history)
            if values.shape != kappa_x.shape or not np.isfinite(values).all():
                raise CableInputError(
                    f"{label} must be finite and match curvature histories"
                )
            if label in {"wire modulus", "stick modulus"} and np.any(values <= 0):
                raise CableInputError(f"{label} must be positive")
    if axial_strain is not None:
        if tension_N is not None:
            raise CableInputError("prescribe axial strain or tension, not both")
        axial_strain = np.asarray(axial_strain, dtype=np.float64)
        if axial_strain.shape != kappa_x.shape or not np.isfinite(axial_strain).all():
            raise CableInputError(
                "axial strain must be finite and match curvature histories"
            )
    _validate_tension(tension_N, EA_full_N, kappa_x.shape)
    phi = spec.phi_wires  # (n_wires,)
    if wires is not None:
        phi = phi[wires]
    n_steps = kappa_x.size
    a = spec.lay_angle_rad

    # Outward radial-fibre stress is retained by this helper. The fatigue
    # driver adds the other fixed circumference fibres after solving slip once.
    # A rectangular or profiled wire (gap G1) has its outer face at t/2.
    d_wire = (
        2.0 * math.sqrt(spec.A_wire_m2 / math.pi)
        if spec.section is None
        else 2.0 * spec.section.half_thickness_m
    )
    sigma_B_kt = local_bending_fibre_stress(
        kappa_x,
        kappa_y,
        phi[:, None],
        spec.E_wire_Pa if wire_modulus_Pa is None else wire_modulus_Pa,
        d_wire,
        a,
    )

    friction_model, _reason = _resolve_friction_model(  # type: ignore[assignment]
        friction_model,
        spec,
        curvature_path_out_of_plane_ratio(kappa_x, kappa_y)
        if friction_model == "auto"
        else 0.0,
        stick_modulus_Pa,
        friction_law,
    )
    if progressive is not None and friction_model != "progressive":
        raise CableInputError(
            "progressive settings require friction_model='progressive'"
        )
    law = active_friction_law(friction_law)
    if law is not None and friction_model == "progressive":
        raise CableInputError(
            "a friction law acts on the single spring-slider per wire; the "
            "progressive law is not implemented with it"
        )
    wires_done = False
    if friction_model == "progressive":
        # Progressive (Iwan) split of the direction-tied cap (gap G8).
        if stick_modulus_Pa is None and _FAST_SLIP:
            sigma_f_kt = progressive_wire_friction_stress(
                spec,
                kappa_x,
                kappa_y,
                progressive=progressive,
                contact_pressure_Pa=contact_pressure_Pa,
                direction_hold_fraction=direction_hold_fraction,
            )
        else:
            sigma_f_kt = progressive_wire_history(
                spec,
                kappa_x,
                kappa_y,
                progressive=progressive,
                contact_pressure_Pa=contact_pressure_Pa,
                stick_modulus_Pa=stick_modulus_Pa,
                direction_hold_fraction=direction_hold_fraction,
            ).stress_Pa
    elif friction_model == "distributed":
        # Distributed elastic-Coulomb helix model (see distributed_friction).
        # A constant stick modulus enters as the Yun coefficient of its state.
        usable, _reason = _distributed_spec(spec, stick_modulus_Pa, None)
        sigma_f_kt = distributed_wire_friction_stress(
            spec if usable is None else usable,
            kappa_x,
            kappa_y,
            contact_pressure_Pa=contact_pressure_Pa,
            stick_modulus_Pa=stick_modulus_Pa if usable is None else None,
            nodes_per_pitch=distributed_nodes_per_pitch,
            max_increment=distributed_max_increment,
            friction_law=law,
            time_s=time_s,
            bound=distributed_bound,
        )
    elif stick_modulus_Pa is None and _FAST_SLIP:
        # Compiled force recursion, identical to wire_slip_history().stress_Pa
        # for a constant stick stiffness (tests/test_fast_slip.py).
        sigma_f_kt = wire_friction_stress(
            spec,
            kappa_x,
            kappa_y,
            contact_pressure_Pa=contact_pressure_Pa,
            direction_hold_fraction=direction_hold_fraction,
            friction_law=law,
            time_s=time_s,
            wires=wires,
        )
        wires_done = True
    else:
        sigma_f_kt = wire_slip_history(
            spec,
            kappa_x,
            kappa_y,
            contact_pressure_Pa=contact_pressure_Pa,
            stick_modulus_Pa=stick_modulus_Pa,
            direction_hold_fraction=direction_hold_fraction,
            friction_law=law,
            time_s=time_s,
        ).stress_Pa

    if wires is not None and not wires_done:
        sigma_f_kt = np.ascontiguousarray(sigma_f_kt[wires])
    # sigma_A (tension) — [Nasution2014b] Eq. 5: sigma_A = E cos^2 a * N / EA_full.
    sigma_A_kt = np.zeros((phi.size, n_steps), dtype=np.float64)
    if normal_stress_Pa is not None:
        sigma_A_kt += np.asarray(normal_stress_Pa)[None, :]
    if axial_strain is not None:
        e_ax = layer.E_eff_axial_Pa or spec.E_wire_Pa
        sigma_A_kt += (e_ax * math.cos(a) ** 2 * axial_strain)[None, :]
    if tension_N is not None:
        assert EA_full_N is not None  # validated before stress recovery
        e_ax = layer.E_eff_axial_Pa or spec.E_wire_Pa
        sigma_A_t = e_ax * math.cos(a) ** 2 * np.asarray(tension_N) / EA_full_N
        sigma_A_kt += sigma_A_t[None, :]

    sigma_T_kt = np.zeros((phi.size, n_steps), dtype=np.float64)
    if thermal_stress_Pa is not None:
        sigma_T_kt += np.asarray(thermal_stress_Pa, dtype=np.float64)[None, :]

    if friction_out is not None:
        friction_out.append(np.asarray(sigma_f_kt, dtype=np.float64))
    return np.asarray(sigma_A_kt + sigma_B_kt + sigma_f_kt + sigma_T_kt)


@dataclass(frozen=True)
class WireStressField:
    """Longitudinal stress on fixed wire fibres, in Pa.

    Arrays have shape (wire, time). The normal/slip contribution and two
    local bending components recover any material fibre without another slip
    calculation or a three-dimensional wire-by-fibre-by-time allocation.
    ``radial_Pa`` is the bending stress at the outer face (``y_n = t/2``,
    the radial fibre of a round wire) and ``tangential_Pa`` at the edge
    (``y_l = b/2``). ``termination`` holds the clamp increment of the
    friction stress (already included in ``mean_Pa``) when a termination
    correction was requested. ``section`` is the cross-section of a
    rectangular or profiled wire (``None``: round). ``transverse_slip_m``
    and ``transverse_load_N_per_m`` are the transverse slip from the
    loxodrome and its contact line load when a transverse model was given.
    ``friction_Pa`` is the friction stress (with any termination increment),
    a part of ``mean_Pa``, kept only on request (``keep_friction=True``) for
    the component stress concentration factors.
    """

    mean_Pa: FloatArray
    radial_Pa: FloatArray
    tangential_Pa: FloatArray
    termination: TerminationIncrement | None = None
    section: WireSection | None = None
    transverse_slip_m: FloatArray | None = None
    transverse_load_N_per_m: FloatArray | None = None
    friction_Pa: FloatArray | None = None

    def scf_fibre(self, angle_rad: float, scf: ComponentSCF) -> FloatArray:
        """Fibre history with component stress concentration factors.

        The axial part (``mean_Pa - friction_Pa``), the friction part and
        both local bending parts are scaled separately, mean and dynamic
        (:func:`~cablexsec.analysis.sn_assessment.scaled_component`).
        """
        if self.friction_Pa is None:
            raise CableInputError(
                "component SCFs need the friction stress (keep_friction=True)"
            )
        n, lat = self.fibre_offsets(angle_rad)
        axial = self.mean_Pa - self.friction_Pa
        return np.asarray(
            scaled_component(axial, scf.axial_mean, scf.axial_dynamic)
            + scaled_component(
                self.friction_Pa, scf.friction_mean, scf.friction_dynamic
            )
            + scaled_component(
                self.radial_Pa * n, scf.bending_normal_mean, scf.bending_normal_dynamic
            )
            + scaled_component(
                self.tangential_Pa * lat,
                scf.bending_lateral_mean,
                scf.bending_lateral_dynamic,
            )
        )

    def fibre_offsets(self, angle_rad: float) -> tuple[float, float]:
        """Multipliers of ``radial_Pa`` and ``tangential_Pa`` at a fibre angle.

        ``(cos, sin)`` for a round wire; the normalised perimeter point of the
        ray at that angle for a rectangular or profiled wire.
        """
        if self.section is None or self.section.is_round:
            return math.cos(angle_rad), math.sin(angle_rad)
        n, lat = self.section.perimeter_offsets(angle_rad)
        return float(n), float(lat)

    def at_fibre(
        self,
        angle_rad: float,
        *,
        samples: slice | int | NDArray[np.bool_] | NDArray[np.int64] = slice(None),
    ) -> FloatArray:
        """Recover one fixed angle; select retained samples after solving history."""
        if not math.isfinite(angle_rad):
            raise CableInputError("material fibre angle must be finite")
        if self.section is None or self.section.is_round:
            return (
                self.mean_Pa[:, samples]
                + self.radial_Pa[:, samples] * math.cos(angle_rad)
                + self.tangential_Pa[:, samples] * math.sin(angle_rad)
            )
        n, lat = self.fibre_offsets(angle_rad)
        return (
            self.mean_Pa[:, samples]
            + self.radial_Pa[:, samples] * n
            + self.tangential_Pa[:, samples] * lat
        )

    def at_point(
        self,
        normal_m: float,
        lateral_m: float,
        *,
        samples: slice | int | NDArray[np.bool_] | NDArray[np.int64] = slice(None),
    ) -> FloatArray:
        """Stress at the section point ``(y_n, y_l)`` [m] (e.g. a gauge groove)."""
        if not (math.isfinite(normal_m) and math.isfinite(lateral_m)):
            raise CableInputError("section point must be finite")
        if self.section is None:
            raise CableInputError(
                "at_point needs the wire section; use at_fibre for round wires"
            )
        return (
            self.mean_Pa[:, samples]
            + self.radial_Pa[:, samples] * (normal_m / self.section.half_thickness_m)
            + self.tangential_Pa[:, samples] * (lateral_m / self.section.half_width_m)
        )


def wire_stress_field(
    layer: FatigueLayer,
    kappa_x: FloatArray,
    kappa_y: FloatArray,
    tension_N: FloatArray | None = None,
    EA_full_N: float | None = None,
    thermal_stress_Pa: FloatArray | None = None,
    contact_pressure_Pa: FloatArray | None = None,
    *,
    axial_strain: FloatArray | None = None,
    normal_stress_Pa: FloatArray | None = None,
    wire_modulus_Pa: FloatArray | None = None,
    stick_modulus_Pa: FloatArray | None = None,
    direction_hold_fraction: float = DIRECTION_HOLD_FRACTION,
    friction_model: FrictionModel = "direction_tied",
    distributed_nodes_per_pitch: int | None = None,
    distributed_max_increment: float = DEFAULT_MAX_INCREMENT,
    termination: TerminationSpec | None = None,
    progressive: ProgressiveFriction | None = None,
    transverse: TransverseSlipModel | None = None,
    curvature_gradient: tuple[FloatArray, FloatArray] | None = None,
    relief: LateralRelief | None = None,
    keep_friction: bool = False,
    friction_law: FrictionLaw | None = None,
    time_s: FloatArray | None = None,
    distributed_bound: DistributedBound = "signed",
    wires: IntArray | None = None,
) -> WireStressField:
    """Recover the shared stress field once, retaining all preceding slip history.

    ``wires`` (indices into the layer's wires, as from :func:`select_wires`)
    recovers the field of those wires only, rows in that order. Each row
    equals the same row of the full field: the direction-tied kernel solves
    only the selected wires, the other friction models and the termination
    increment solve the layer and keep the selected rows. Not combined with
    ``transverse``, whose slip couples neighbouring wires.

    ``keep_friction=True`` also stores the friction stress in
    ``friction_Pa`` (component stress concentration factors); the field is
    otherwise identical.

    ``friction_model`` selects the armour friction stress: ``"direction_tied"``
    (default, :func:`~cablexsec.mechanics.fast_slip.wire_friction_stress`) or
    ``"distributed"``
    (:func:`~cablexsec.mechanics.distributed_friction.distributed_friction_history`
    with ``distributed_nodes_per_pitch``, ``distributed_max_increment`` and
    the bound form ``distributed_bound``).
    ``"progressive"`` splits each wire into parallel sub-elements on the
    direction-tied cap
    (:mod:`cablexsec.mechanics.progressive_friction`, settings ``progressive``).
    ``"auto"`` (opt-in) takes the direction-tied model for a planar
    curvature path and the distributed model otherwise, where it applies
    (see :func:`compute_fatigue`).
    ``direction_hold_fraction`` applies to the direction-tied and progressive
    models.

    ``termination`` (:class:`~cablexsec.analysis.termination.TerminationSpec`)
    adds the anchored-wire clamp increment to the friction stress of a section
    near an end fitting or a clamped bend-stiffener root
    (:func:`~cablexsec.analysis.termination.termination_friction_increment`;
    planar clamp solver, biaxial treatment as documented there). ``None``, an
    ``anchor="none"`` spec or a section beyond the solved reach leaves the
    sectional field unchanged, bit for bit.

    Local tendon bending (gap G1, :mod:`cablexsec.mechanics.tendon_bending`).
    A rectangular or profiled wire (``layer.spec.section``) takes its local
    bending at the outer face ``t/2`` and the edge ``b/2``. ``transverse``
    (:class:`~cablexsec.mechanics.tendon_bending.TransverseSlipModel`) adds
    the transverse slip towards the geodesic per wire (Jenkin element): the
    lateral curvature becomes ``cos a (1 + sin^2 a) p_l - k^2 v`` and the
    normal curvature gains the tangent-rotation term, so that full slip gives
    the geodesic values. ``curvature_gradient`` is the history of
    ``(dkappa_x/dx, dkappa_y/dx)`` [1/m^2] at the section (bellmouth or
    stiffener gradient), used by ``transverse`` and ``relief``. ``relief``
    (:class:`~cablexsec.mechanics.tendon_bending.LateralRelief`) lowers the
    axial friction bound by the transverse load (vector Coulomb, upper
    estimate of the relief); it needs ``friction_model="distributed"``, whose
    bound follows the azimuth along each wire. ``None`` for all three leaves
    the field of a round wire unchanged, bit for bit.

    ``friction_law`` (:class:`~cablexsec.mechanics.friction_law.FrictionLaw`)
    sets a static/dynamic and rate-dependent coefficient of the armour
    friction (direction-tied model; static/dynamic only for the distributed
    model), with the sample times ``time_s`` when the law needs them. The
    termination clamp increment is rate independent and is refused with a
    non-neutral law. ``None`` or a neutral law leaves the field unchanged,
    bit for bit.
    """
    kx, ky = real_array(kappa_x, "kappa_x"), real_array(kappa_y, "kappa_y")
    _check_distributed_bound(distributed_bound)
    if friction_model == "auto":
        friction_model, reason = _resolve_friction_model(  # type: ignore[assignment]
            friction_model,
            layer.spec,
            curvature_path_out_of_plane_ratio(kx, ky),
            stick_modulus_Pa,
            friction_law,
        )
        if reason:
            _warn_auto_fallback(layer.name, reason, stacklevel=2)
    if (
        active_friction_law(friction_law) is not None
        and isinstance(termination, TerminationSpec)
        and termination.anchor != "none"
    ):
        raise CableInputError(
            "the termination clamp increment is rate independent; it is not "
            "combined with a non-neutral friction law"
        )
    friction_law_time_steps(friction_law, time_s, kx.size)  # checks the time base
    if wires is not None:
        wires = np.asarray(wires, dtype=np.int64)
        if (
            wires.ndim != 1
            or wires.size == 0
            or np.any(wires < 0)
            or np.any(wires >= layer.spec.n_wires)
            or np.any(np.diff(wires) <= 0)
        ):
            raise CableInputError(
                "wires must be ascending, distinct indices of the layer's wires"
            )
        if transverse is not None:
            raise CableInputError(
                "transverse slip couples neighbouring wires; evaluate every wire"
            )
    unrelieved_spec = layer.spec
    gradient = None
    if curvature_gradient is not None:
        gx = real_array(curvature_gradient[0], "curvature gradient x")
        gy = real_array(curvature_gradient[1], "curvature gradient y")
        if gx.shape != kx.shape or gy.shape != kx.shape:
            raise CableInputError("curvature gradient must match the curvature history")
        gradient = (gx, gy)
    if relief is not None:
        if friction_model != "distributed":
            raise CableInputError(
                "lateral relief needs friction_model='distributed' (its bound "
                "follows the azimuth along each wire)"
            )
        if not isinstance(relief, LateralRelief):
            raise CableInputError("relief must be a LateralRelief")
        amplitude = float(np.max(np.hypot(kx, ky))) if kx.size else 0.0
        grad_amp = float(np.max(np.hypot(*gradient))) if gradient is not None else 0.0
        base = (
            np.ones(layer.spec.n_wires)
            if layer.spec.shape_g is None
            else np.asarray(layer.spec.shape_g, dtype=np.float64)
        )
        relieved = base * relief.factor(layer.spec.phi_wires, amplitude, grad_amp)
        layer = dataclasses.replace(
            layer, spec=dataclasses.replace(layer.spec, shape_g=relieved)
        )
    friction_parts: list[FloatArray] | None = [] if keep_friction else None
    outward = _wire_stress_series(
        layer,
        kx,
        ky,
        tension_N,
        EA_full_N,
        thermal_stress_Pa,
        contact_pressure_Pa,
        axial_strain=axial_strain,
        normal_stress_Pa=normal_stress_Pa,
        wire_modulus_Pa=wire_modulus_Pa,
        stick_modulus_Pa=stick_modulus_Pa,
        direction_hold_fraction=direction_hold_fraction,
        friction_model=friction_model,
        distributed_nodes_per_pitch=distributed_nodes_per_pitch,
        distributed_max_increment=distributed_max_increment,
        progressive=progressive,
        friction_out=friction_parts,
        friction_law=friction_law,
        time_s=time_s,
        distributed_bound=distributed_bound,
        wires=wires,
    )
    spec = layer.spec
    increment = None
    if termination is not None:
        if not isinstance(termination, TerminationSpec):
            raise CableInputError("termination must be a TerminationSpec or None")
        increment = termination_friction_increment(
            unrelieved_spec,
            kx,
            ky,
            termination,
            lay_hand=layer.lay_hand,
            contact_pressure_Pa=contact_pressure_Pa,
        )
        if wires is not None:
            increment = dataclasses.replace(
                increment, stress_Pa=np.ascontiguousarray(increment.stress_Pa[wires])
            )
        if increment.applied:
            outward = outward + increment.stress_Pa
            if friction_parts:
                friction_parts[0] = friction_parts[0] + increment.stress_Pa
    friction = friction_parts[0] if friction_parts else None
    modulus = spec.E_wire_Pa if wire_modulus_Pa is None else wire_modulus_Pa
    if spec.section is None:
        diameter = 2 * math.sqrt(spec.A_wire_m2 / math.pi)
        depth, width = diameter, diameter
    else:
        depth = 2 * spec.section.half_thickness_m
        width = 2 * spec.section.half_width_m
    phi_eval = spec.phi_wires if wires is None else spec.phi_wires[wires]
    radial = local_bending_fibre_stress(
        kx,
        ky,
        phi_eval[:, None],
        modulus,
        depth,
        spec.lay_angle_rad,
    )
    tangential = local_bending_fibre_stress(
        kx,
        ky,
        phi_eval[:, None],
        modulus,
        width,
        spec.lay_angle_rad,
        math.pi / 2,
    )
    mean = outward - radial
    if transverse is None:
        return WireStressField(
            mean, radial, tangential, increment, spec.section, friction_Pa=friction
        )
    if not isinstance(transverse, TransverseSlipModel):
        raise CableInputError("transverse must be a TransverseSlipModel")
    phi = spec.phi_wires[:, None]
    p_l = -kx[None, :] * np.sin(phi) + ky[None, :] * np.cos(phi)
    g_n = (
        None
        if gradient is None
        else gradient[0][None, :] * np.cos(phi) + gradient[1][None, :] * np.sin(phi)
    )
    v, q, k_l = transverse.slip_history(p_l, g_n)
    tangential = np.asarray(modulus * k_l * (width / 2), dtype=np.float64)
    # Rotation of the wire tangent by dv/ds on the cylinder: normal curvature
    # change -(sin 2a / r) k dv/dphi (full slip: the geodesic 3 cos^2 a).
    step = 2 * math.pi / spec.n_wires
    dv = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * step)
    k_wave = abs(math.sin(spec.lay_angle_rad)) / spec.r_helix_m
    dk_n = -(math.sin(2 * spec.lay_angle_rad) / spec.r_helix_m) * k_wave * dv
    radial = np.asarray(radial + modulus * dk_n * (depth / 2), dtype=np.float64)
    return WireStressField(
        mean, radial, tangential, increment, spec.section, v, q, friction_Pa=friction
    )


def _fretting_amplitudes_um(
    layer: FatigueLayer,
    kappa_x: FloatArray,
    kappa_y: FloatArray,
    contact_force_N: FloatArray | None,
    wires: IntArray | None = None,
) -> FloatArray:
    """Per-wire fretting slip amplitude ``delta*`` [um] from the Mindlin model.

    Zero everywhere if the layer has no fretting stiffness/curve. ``wires``
    evaluates those wires only (one independent contact per wire).
    """
    spec = layer.spec
    rows = np.arange(spec.n_wires) if wires is None else wires
    n_w = rows.size
    if layer.K_t_contact_N_per_m <= 0.0 or layer.sn_fretting is None:
        return np.zeros(n_w, dtype=np.float64)
    c_geo = geometric_slip_factor(spec.r_helix_m, spec.lay_angle_rad)
    g = (
        np.ones(spec.n_wires)
        if layer.contact_shape_g is None
        else layer.contact_shape_g
    )[rows]
    phi = spec.phi_wires[rows]
    delta_star = np.zeros(n_w, dtype=np.float64)
    kx, ky = np.asarray(kappa_x), np.asarray(kappa_y)
    for k in range(n_w):
        if g[k] == 0:
            continue  # No normal force, hence no contact-fatigue contribution.
        contact = MindlinContact(
            K_t=layer.K_t_contact_N_per_m * float(g[k]) ** (1 / 3),
            mu=layer.fretting_mu,
        )
        kx_prev = ky_prev = 0.0
        for t in range(kx.size):
            d_delta = slip_increment(
                float(kx[t] - kx_prev),
                float(ky[t] - ky_prev),
                float(phi[k]),
                c_geo,
                transverse_ratio=layer.transverse_ratio,
                transverse_map=layer.transverse_map or "legacy",
                lay_angle_rad=spec.lay_angle_rad,
            )
            kx_prev, ky_prev = float(kx[t]), float(ky[t])
            if contact_force_N is None:
                if layer.contact_line_load_N_per_m is None:
                    raise CableInputError(
                        "fretting requires the shared-interface line load, "
                        "not a layer-face sum"
                    )
                mean_line_load = layer.contact_line_load_N_per_m
            else:
                mean_line_load = float(contact_force_N[t])
            if layer.crossing_density_per_m <= 0.0:
                continue
            force_per_crossing = (
                mean_line_load * float(g[k]) / layer.crossing_density_per_m
            )
            contact.update(d_delta, force_per_crossing)
        delta_star[k] = contact.fretting_amplitude * 1.0e6  # m -> um
    return delta_star


def _contact_force_histories(
    layer: FatigueLayer,
    kappa_x: FloatArray,
    kappa_y: FloatArray,
    contact_force_N: FloatArray | None,
    wires: IntArray | None = None,
) -> tuple[FloatArray, NDArray[np.bool_], FloatArray]:
    """Tangential force, slip flags and normal force at each wire's crossing.

    The same kinematic slip family and Coulomb return map with Mindlin initial
    stiffness as :func:`_fretting_amplitudes_um` are advanced for every wire
    (or for the ``wires`` given, rows in that order).
    Returns the signed tangential force along the principal direction of its
    trajectory ``(n_wires, n_steps)`` [N], the slip flag of each increment and
    the constant normal force per crossing ``(n_wires,)`` [N].
    """
    spec = layer.spec
    rows = np.arange(spec.n_wires) if wires is None else wires
    n_w = rows.size
    kx, ky = np.asarray(kappa_x), np.asarray(kappa_y)
    n_t = kx.size
    if layer.K_t_contact_N_per_m <= 0.0 or layer.crossing_density_per_m <= 0.0:
        raise CableInputError(
            f"{layer.name}: the slip-cycle SWT branch needs a positive tangential "
            "contact stiffness and crossing density; use "
            "copper_contact_model='legacy_gross_sliding' only to reproduce the "
            "former estimate"
        )
    if contact_force_N is None:
        if layer.contact_line_load_N_per_m is None:
            raise CableInputError(
                "fretting requires the shared-interface line load, not a layer-face sum"
            )
        line_load = float(layer.contact_line_load_N_per_m)
    else:
        line_load = float(contact_force_N[0])
    c_geo = geometric_slip_factor(spec.r_helix_m, spec.lay_angle_rad)
    g = (
        np.ones(spec.n_wires)
        if layer.contact_shape_g is None
        else layer.contact_shape_g
    )[rows]
    phi = spec.phi_wires[rows]
    force = np.zeros((n_w, n_t), dtype=np.float64)
    slipping = np.zeros((n_w, n_t), dtype=np.bool_)
    normal = line_load * np.asarray(g, dtype=np.float64) / layer.crossing_density_per_m
    for k in range(n_w):
        if g[k] == 0:
            continue
        contact = MindlinContact(
            K_t=layer.K_t_contact_N_per_m * float(g[k]) ** (1 / 3),
            mu=layer.fretting_mu,
        )
        history = np.zeros((n_t, 2), dtype=np.float64)
        kx_prev = ky_prev = 0.0
        for t in range(n_t):
            d_delta = slip_increment(
                float(kx[t] - kx_prev),
                float(ky[t] - ky_prev),
                float(phi[k]),
                c_geo,
                transverse_ratio=layer.transverse_ratio,
                transverse_map=layer.transverse_map or "legacy",
                lay_angle_rad=spec.lay_angle_rad,
            )
            kx_prev, ky_prev = float(kx[t]), float(ky[t])
            f_t, state, _c = contact.update(d_delta, float(normal[k]))
            history[t] = f_t
            slipping[k, t] = state == "slip"
        if layer.transverse_ratio == 0.0:
            force[k] = history[:, 0]
        else:
            # Count along the principal axis of the planar force trajectory.
            _u, _s, vt = np.linalg.svd(history, full_matrices=False)
            force[k] = history @ vt[0]
    return force, slipping, normal


# --------------------------------------------------------------------------- #
# Full pipeline                                                                #
# --------------------------------------------------------------------------- #
def compute_fatigue(
    layers: list[FatigueLayer],
    kappa_x: FloatArray,
    kappa_y: FloatArray,
    *,
    tension_N: FloatArray | None = None,
    axial_strain: FloatArray | None = None,
    EA_full_N: float | None = None,
    thermal_stress_Pa: dict[str, FloatArray] | None = None,
    contact_pressure_Pa: dict[str, FloatArray] | None = None,
    contact_force_N: dict[str, FloatArray] | None = None,
    normal_stress_Pa: dict[str, FloatArray] | None = None,
    wire_modulus_Pa: dict[str, FloatArray] | None = None,
    stick_modulus_Pa: dict[str, FloatArray] | None = None,
    n_fibres: int = 32,
    rainflow_counter: RainflowCounter = "auto",
    direction_hold_fraction: float = DIRECTION_HOLD_FRACTION,
    friction_model: FrictionModel | None = None,
    distributed_nodes_per_pitch: int | Mapping[str, int] | None = None,
    distributed_max_increment: float = DEFAULT_MAX_INCREMENT,
    distributed_bound: DistributedBound = "signed",
    copper_contact_model: CopperContactModel = "slip_cycle",
    partial_slip_cycles: PartialSlipCycles = "skip",
    termination: TerminationSpec | Mapping[str, TerminationSpec] | None = None,
    progressive: ProgressiveFriction | None = None,
    count_from_sample: int = 0,
    friction_law: FrictionLaw | None = None,
    time_s: FloatArray | None = None,
    wire_positions: WirePositions | None = None,
) -> FatigueResult:
    """Run the per-wire fatigue pipeline over a curvature/tension history.

    Parameters
    ----------
    layers : list of FatigueLayer
        Helical layers with their S-N curves and contact data.
    kappa_x, kappa_y : numpy.ndarray
        Biaxial curvature time series [1/m] (same length).
    tension_N : numpy.ndarray or None, optional
        Cable tension time series [N] for the ``sigma_A`` term; requires
        ``EA_full_N``.
    EA_full_N : float or None, optional
        Finite positive full cross-section axial stiffness [N], required whenever
        tension_N is supplied. Tension is never silently dropped.
    contact_force_N : dict[str, numpy.ndarray] or None, optional
        Legacy parameter name: values are shared-interface line loads [N/m],
        converted to force per crossing using crossing_density_per_m. Contact-life
        laws require a constant value consistent with their Hertz reference.
        Variable-normal-load bulk fatigue remains supported.
    n_fibres : int, optional
        Number of equally spaced, fixed material fibres around each circular
        wire (default 32). Increase this count to check angular convergence.
        Stress is counted separately at each fibre; the largest accumulated
        bulk damage is returned for the wire.
    wire_positions : int, sequence of float or mapping, optional
        Wires to evaluate instead of every wire (:func:`select_wires`): ``N``
        equally spaced positions per layer, explicit azimuths [deg] (the
        nearest wire to each), or a mapping from every analysed layer name to
        either. The per-wire arrays then hold the selected wires in ascending
        order, ``wire_index`` names them, and each value equals the value of
        the same wire in the full evaluation (bit for bit). Sums and the
        critical wire are over the selected wires.
    rainflow_counter : {"auto", "compiled", "reference"}, optional
        Counter for the bulk Miner sums. ``"compiled"`` uses
        :func:`cablexsec.analysis.fast_rainflow.miner_rows`, which reproduces
        :func:`miner_damage` cycle for cycle and in the same summation order
        (numba-compiled when available, otherwise interpreted and slow);
        ``"reference"`` uses :func:`miner_damage`. ``"auto"`` (default)
        selects the compiled counter when numba is active.
    direction_hold_fraction : float, optional
        Fraction ``f`` of the smallest slip-onset curvature below which the
        slip-cap bending direction is held (default
        ``DIRECTION_HOLD_FRACTION``); ``0`` reproduces the earlier update.
        Used by the direction-tied friction model only.
    friction_model : {"direction_tied", "auto", "distributed", "progressive"}, optional
        Armour friction-stress model. ``"direction_tied"`` (the default,
        ``None``) is the pointwise Papailiou cap at the held bending direction
        (:mod:`cablexsec.mechanics.fast_slip`). ``"distributed"`` solves the
        elastic--Coulomb friction field along one pitch of the helix with the
        capstan bound ``mu |sin a|`` and interface stiffness
        ``k_hat = xi/(1 - xi)``
        (:mod:`cablexsec.mechanics.distributed_friction`); it is slower and
        requires a constant stick modulus (a constant one other than the
        layer's enters as the Yun coefficient of that state). ``"progressive"``
        splits each wire of the direction-tied law into parallel sub-elements
        whose caps average to the same cap
        (:mod:`cablexsec.mechanics.progressive_friction`, gap G8; settings in
        ``progressive``). ``"auto"`` (opt-in) takes the direction-tied model
        when the curvature path is planar (out-of-plane ratio
        :func:`curvature_path_out_of_plane_ratio` at most
        ``PLANAR_PATH_TOLERANCE``) and the distributed model otherwise; a
        layer that the distributed model does not cover exactly (rigid
        interface ``xi >= 1``, which it represents only by the along-wire
        penalty, a stick modulus that changes over the record, a
        rate-dependent friction law) keeps the direction-tied model, and the
        provenance records the model of every layer and the reason for such a
        fallback. The direction-tied cap turns with the bending direction:
        under a steadily rotating curvature its stress ranges fall below those
        of the distributed model (to 0.86 of them at twice the slip onset on a
        near-rigid interface), while on planar and flat elliptical paths they
        are at or above them. The default changes to ``"auto"`` in the next
        release: left at its default on a non-planar path (the only case in
        which the two differ), the model warns once per call
        (:class:`~cablexsec.provenance.DirectionTiedFrictionWarning`, a
        ``FutureWarning``); the result of this release is unchanged.
    distributed_nodes_per_pitch : int, dict[str, int] or None, optional
        Helix nodes per pitch of the distributed model, for every layer or per
        layer name (default ``DEFAULT_NODES_PER_WIRE`` times the wire count of
        each layer). Multiples of the wire count avoid interpolation.
    distributed_max_increment : float, optional
        Largest curvature increment per substep of the distributed model, in
        units of the slip-onset curvature (default ``DEFAULT_MAX_INCREMENT``).
    distributed_bound : {"signed", "uniform"}, optional
        Friction bound of the distributed model: ``"signed"`` (default) with
        the capstan term on the signed bending force of the wire,
        ``"uniform"`` without it
        (:func:`~cablexsec.mechanics.distributed_friction.distributed_friction_history`).
    copper_contact_model : {"slip_cycle", "legacy_gross_sliding"}, optional
        Copper SWT contact branch. ``"slip_cycle"`` (default) counts cycles of
        the crossing's tangential force, skips cycles without slip and gives
        each remaining cycle its own SWT driver
        (:func:`~cablexsec.analysis.fretting_swt.swt_slip_cycle_damage`).
        ``"legacy_gross_sliding"`` reproduces the former estimate described in
        the notes, for comparison only.
    partial_slip_cycles : {"skip", "cattaneo_mindlin"}, optional
        Treatment of non-slipping cycles in the ``"slip_cycle"`` branch.
    termination : TerminationSpec, dict[str, TerminationSpec] or None, optional
        Termination (clamp) correction for a section near an end fitting or a
        clamped bend-stiffener root, for every layer or per layer name
        (layers not named are uncorrected). The friction stress of every wire
        becomes the sectional value plus the anchored-wire clamp increment
        along the curvature history
        (:func:`~cablexsec.analysis.termination.termination_friction_increment`).
        The clamp solver is planar: by default the increment follows the
        principal component of the history (``biaxial="principal"``), with
        ``"components"`` adding the minor component; the sectional friction
        keeps the full biaxial law. The anchor is rigid unless
        ``k_anchor_N_m`` gives its stiffness per wire (an end-fitting
        potting). ``None`` (default) gives the sectional
        result, as does a section beyond the solved reach, bit for bit.
    count_from_sample : int, optional
        Lead-in: the friction state runs through the whole history, and the
        bulk cycles are counted from this sample on (default 0, every sample,
        bit for bit as before). Lets a record start from the stress-free
        section and drop the start-up from the damage. Not available with a
        contact-life branch.
    friction_law : FrictionLaw, optional
        Static/dynamic and rate-dependent armour friction coefficient
        (:class:`~cablexsec.mechanics.friction_law.FrictionLaw`), for the
        direction-tied model and (static/dynamic and pressure factor only)
        the distributed model; refused with the progressive model and a
        termination correction. The contact-life branches keep their own
        rate-independent crossing contact. ``None`` (default) or a neutral
        law gives the earlier result, bit for bit.
    time_s : numpy.ndarray, optional
        Sample times [s] of the history, required by a law that depends on
        time (slip rate or a finite healing time).

    Returns
    -------
    FatigueResult

    Notes
    -----
    Copper SWT contact branch (a layer with ``swt_const`` and a positive
    ``contact_p0_Pa``). With ``copper_contact_model="slip_cycle"`` each wire
    contact is advanced by the same kinematic slip family and Coulomb return
    map with Mindlin initial stiffness as the steel slip-amplitude branch
    (``K_t_contact_N_per_m`` and the force per crossing are required). The
    tangential-force history is rainflow counted; cycles in which the
    crossing does not slip are skipped (or, on request, given the
    Cattaneo-Mindlin partial-slip driver); every slipping cycle receives
    ``SWT = sigma_max sigma_a (1 - nu^2)/E`` with
    ``sigma = 2 mu p0_k sqrt(Q/(mu P))`` from its own tangential-load
    amplitude and the wire's Hertz pressure ``p0_k = contact_p0_Pa g_k^(1/3)``.
    Damage is the Miner sum of ``count / N_f(SWT)``.

    With ``copper_contact_model="legacy_gross_sliding"`` the contact damage of
    wire ``k`` is ``N_k / N_f(p0_k)``, where ``N_k`` is the TOTAL rainflow
    cycle count of that wire's bulk longitudinal stress at the outward radial
    fibre and ``N_f`` the SWT life at ONE fixed driver ``2 mu p0_k``. Every
    counted cycle, however small, is then one gross-sliding contact cycle, so
    the estimate can overstate contact damage for histories dominated by small
    cycles in which the contact sticks. It is retained only to reproduce
    earlier results.

    Only the supplied ``layers`` are assessed. Curves declared on cable layers
    that are not converted to a :class:`FatigueLayer` (for example a stranded
    conductor or a sheath) are not evaluated here;
    :func:`cablexsec.api.compute_fatigue` warns about them with
    :class:`~cablexsec.provenance.UnevaluatedFatigueCurveWarning`.
    """
    compiled = _use_compiled_counter(rainflow_counter)
    default_model = friction_model is None
    if friction_model is None:
        friction_model = DEFAULT_FRICTION_MODEL
    _check_friction_model(friction_model)
    _check_distributed_bound(distributed_bound)
    if copper_contact_model not in COPPER_CONTACT_MODELS:
        raise CableInputError(
            f"copper_contact_model must be one of {COPPER_CONTACT_MODELS}, "
            f"got {copper_contact_model!r}"
        )
    if partial_slip_cycles not in ("skip", "cattaneo_mindlin"):
        raise CableInputError(
            "partial_slip_cycles must be 'skip' or 'cattaneo_mindlin'"
        )
    kx = real_array(kappa_x, "kappa_x")
    ky = real_array(kappa_y, "kappa_y")
    if (
        isinstance(n_fibres, (bool, np.bool_))
        or not isinstance(n_fibres, (int, np.integer))
        or n_fibres < 4
    ):
        raise CableInputError("n_fibres must be an integer of at least 4")
    if kx.shape != ky.shape:
        raise CableInputError("kappa_x and kappa_y must have the same shape")
    if kx.ndim != 1 or kx.size < 2:
        raise CableInputError("curvature histories must be 1-D with at least 2 points")
    if (
        isinstance(count_from_sample, (bool, np.bool_))
        or not isinstance(count_from_sample, (int, np.integer))
        or not 0 <= count_from_sample <= kx.size - 2
    ):
        raise CableInputError(
            "count_from_sample must be an integer from 0 to the history length minus 2"
        )
    count_from = int(count_from_sample)
    if not np.isfinite(kx).all() or not np.isfinite(ky).all():
        raise CableInputError("curvature histories must be finite")
    if axial_strain is not None:
        if tension_N is not None:
            raise CableInputError("prescribe axial strain or tension, not both")
        strain = real_array(axial_strain, "axial_strain")
        if strain.shape != kx.shape or not np.isfinite(strain).all():
            raise CableInputError(
                "axial_strain must be finite and match the curvature histories"
            )
    if normal_stress_Pa is not None and any(
        value is not None for value in (tension_N, axial_strain, thermal_stress_Pa)
    ):
        raise CableInputError(
            "supply solved normal stress or separate axial/thermal loads"
        )
    _validate_tension(tension_N, EA_full_N, kx.shape)
    layer_names = {layer.name for layer in layers}
    if len(layer_names) != len(layers):
        raise CableInputError("fatigue layer names must be unique")
    if isinstance(wire_positions, Mapping) and set(wire_positions) != layer_names:
        raise CableInputError("wire_positions must cover exactly the analysed layers")
    selected = {layer.name: _layer_wires(wire_positions, layer) for layer in layers}
    if (
        isinstance(distributed_nodes_per_pitch, Mapping)
        and set(distributed_nodes_per_pitch) != layer_names
    ):
        raise CableInputError(
            "distributed_nodes_per_pitch must cover exactly the analysed layers"
        )
    if termination is not None and not isinstance(termination, TerminationSpec):
        if not isinstance(termination, Mapping) or not all(
            isinstance(value, TerminationSpec) for value in termination.values()
        ):
            raise CableInputError(
                "termination must be a TerminationSpec or a mapping of layer "
                "names to TerminationSpec"
            )
        unknown_layers = set(termination) - layer_names
        if unknown_layers:
            raise CableInputError(
                f"termination contains unknown layers: {sorted(unknown_layers)}"
            )
    # The clamped correction exceeds the only published near-termination
    # measurements (SV11, T2); the result itself is unchanged.
    warn_if_clamped(termination)
    for label, histories in (
        ("thermal_stress_Pa", thermal_stress_Pa),
        ("contact_pressure_Pa", contact_pressure_Pa),
        ("contact_force_N", contact_force_N),
        ("normal_stress_Pa", normal_stress_Pa),
        ("wire_modulus_Pa", wire_modulus_Pa),
        ("stick_modulus_Pa", stick_modulus_Pa),
    ):
        if histories is not None:
            for name, history in histories.items():
                real_array(history, f"{label}[{name!r}]")
                if np.asarray(history).shape != kx.shape:
                    raise CableInputError(
                        f"{label}[{name!r}] must match the curvature-history shape"
                    )
                if not np.isfinite(history).all():
                    raise CableInputError(f"{label}[{name!r}] must be finite")
                if label in {"contact_pressure_Pa", "contact_force_N"} and np.any(
                    np.asarray(history) < 0
                ):
                    raise CableInputError(f"{label}[{name!r}] must be nonnegative")
            if label in {
                "thermal_stress_Pa",
                "contact_pressure_Pa",
                "contact_force_N",
            }:
                unknown = set(histories) - layer_names
                if unknown:
                    raise CableInputError(
                        f"{label} contains unknown layers: {sorted(unknown)}"
                    )
    for label, histories in (
        ("normal_stress_Pa", normal_stress_Pa),
        ("wire_modulus_Pa", wire_modulus_Pa),
        ("stick_modulus_Pa", stick_modulus_Pa),
    ):
        if histories is not None and set(histories) != {layer.name for layer in layers}:
            raise CableInputError(
                f"{label} must cover exactly the analysed helical layers"
            )

    for layer in layers:
        has_contact_life = layer.sn_fretting is not None or layer.swt_const is not None
        if has_contact_life and contact_force_N is not None:
            force = contact_force_N.get(layer.name)
            if force is not None:
                reference = layer.contact_line_load_N_per_m
                if reference is None:
                    reference = float(force[0])
                # Hertz stiffness and peak pressure were solved at one reference
                # force. Varying only the Coulomb cap does not update that contact.
                if not np.allclose(force, reference, rtol=1e-12, atol=0):
                    raise CableInputError(
                        f"{layer.name}: contact-life laws require constant normal "
                        "contact matching the supplied Hertz properties; variable "
                        "contact requires a variable-load contact-life formulation"
                    )

    d_bulk: dict[str, FloatArray] = {}
    d_fret: dict[str, FloatArray] = {}
    d_term: dict[str, FloatArray] = {}
    d_total: dict[str, FloatArray] = {}
    d_fibre: dict[str, FloatArray] = {}
    d_adjusted_fibre: dict[str, FloatArray] = {}
    critical_fibre: dict[str, FloatArray] = {}
    critical_adjusted_fibre: dict[str, FloatArray] = {}
    clamp_summary: dict[str, dict[str, float | bool | str]] = {}
    fibre_angles = np.arange(n_fibres) * (2 * math.pi / n_fibres)
    out_of_plane = curvature_path_out_of_plane_ratio(kx, ky)
    if default_model:
        _warn_default_direction_tied(out_of_plane, stacklevel=2)
    chosen_model: dict[str, str] = {}
    fallback_reason: dict[str, str] = {}
    if count_from:
        for layer in layers:
            if (layer.swt_const is not None and layer.contact_p0_Pa > 0.0) or (
                layer.sn_fretting is not None and layer.K_t_contact_N_per_m > 0.0
            ):
                raise CableInputError(
                    "count_from_sample is not available with the contact-life branch "
                    f"of layer {layer.name!r}"
                )

    for layer in layers:
        spec = layer.spec
        name = layer.name
        layer_stick = None if stick_modulus_Pa is None else stick_modulus_Pa[name]
        model, reason = _resolve_friction_model(
            friction_model, spec, out_of_plane, layer_stick, friction_law
        )
        chosen_model[name] = model
        if reason:
            fallback_reason[name] = reason
            _warn_auto_fallback(name, reason, stacklevel=2)
        layer_termination = (
            termination
            if termination is None or isinstance(termination, TerminationSpec)
            else termination.get(name)
        )
        field = wire_stress_field(
            layer,
            kx,
            ky,
            tension_N,
            EA_full_N,
            None if thermal_stress_Pa is None else thermal_stress_Pa.get(name),
            None if contact_pressure_Pa is None else contact_pressure_Pa.get(name),
            axial_strain=axial_strain,
            normal_stress_Pa=None
            if normal_stress_Pa is None
            else normal_stress_Pa[name],
            wire_modulus_Pa=None if wire_modulus_Pa is None else wire_modulus_Pa[name],
            stick_modulus_Pa=layer_stick,
            direction_hold_fraction=direction_hold_fraction,
            friction_model=model,  # type: ignore[arg-type]
            distributed_nodes_per_pitch=_layer_nodes(distributed_nodes_per_pitch, name),
            distributed_max_increment=distributed_max_increment,
            distributed_bound=distributed_bound,
            termination=layer_termination,
            progressive=progressive,
            keep_friction=layer.component_scf is not None
            and not layer.component_scf.is_identity,
            friction_law=friction_law,
            time_s=time_s,
            wires=selected[name],
        )
        if field.termination is not None and layer_termination is not None:
            clamp_summary[name] = _clamp_summary(layer_termination, field.termination)

        wires = selected[name]
        rows = np.arange(spec.n_wires, dtype=np.int64) if wires is None else wires
        n_eval = rows.size
        sigma_kt = field.at_fibre(0.0)
        fibre_damage = np.empty((n_eval, n_fibres))
        amplified_damage = np.empty_like(fibre_damage)
        corrected = needs_correction(
            layer.sn_bulk, layer.mean_stress, layer.component_scf
        )
        thickness = layer.thickness_m or (
            2.0 * math.sqrt(spec.A_wire_m2 / math.pi)
            if spec.section is None
            else 2.0 * spec.section.half_thickness_m
        )
        for j, theta in enumerate(fibre_angles):
            if not corrected:
                history = field.at_fibre(theta)
                if count_from:
                    history = history[:, count_from:]
                fibre_damage[:, j] = _miner_damage_rows(
                    history, layer.sn_bulk, layer.sigma_UTS_Pa, 1.0, compiled
                )
                amplified_damage[:, j] = (
                    _miner_damage_rows(
                        history, layer.sn_bulk, layer.sigma_UTS_Pa, layer.SCF, compiled
                    )
                    if layer.termination and layer.SCF != 1.0
                    else fibre_damage[:, j]
                )
                continue
            # User S-N options: component SCFs, mean-stress correction and
            # the curve's thickness factor (sn_assessment).
            history = (
                field.scf_fibre(float(theta), layer.component_scf)
                if layer.component_scf is not None
                and not layer.component_scf.is_identity
                else field.at_fibre(theta)
            )
            if count_from:
                history = history[:, count_from:]
            fibre_damage[:, j] = corrected_damage_rows(
                history,
                layer.sn_bulk,
                mean_stress=layer.mean_stress,
                thickness_m=thickness,
                compiled=compiled,
            )
            amplified_damage[:, j] = (
                corrected_damage_rows(
                    history,
                    layer.sn_bulk,
                    mean_stress=layer.mean_stress,
                    stress_factor=layer.SCF,
                    thickness_m=thickness,
                    compiled=compiled,
                )
                if layer.termination and layer.SCF != 1.0
                else fibre_damage[:, j]
            )
        critical_indices = _first_maximum(fibre_damage)
        db = fibre_damage[np.arange(n_eval), critical_indices]
        d_fibre[name] = fibre_damage
        critical_fibre[name] = np.degrees(fibre_angles[critical_indices])
        adjusted_indices = _first_maximum(amplified_damage)
        adjusted_db = amplified_damage[np.arange(n_eval), adjusted_indices]
        d_adjusted_fibre[name] = amplified_damage
        critical_adjusted_fibre[name] = np.degrees(fibre_angles[adjusted_indices])
        # Subtract the unamplified maximum only for decomposition. Form total
        # directly from adjusted damage to avoid cancellation for small factors.
        dt = adjusted_db - db

        # Fretting branch — two formulations, chosen per material.
        df = np.zeros_like(db)
        if (
            layer.swt_const is not None
            and layer.contact_p0_Pa > 0.0
            and copper_contact_model == "slip_cycle"
        ):
            # Copper: per-cycle SWT from the crossing's tangential-force
            # history, stick cycles skipped ([Poon2023] Eqs. 7-9).
            force, slipping, normal = _contact_force_histories(
                layer,
                kx,
                ky,
                None if contact_force_N is None else contact_force_N.get(name),
                wires,
            )
            g = (
                np.ones(spec.n_wires)
                if layer.contact_shape_g is None
                else layer.contact_shape_g
            )[rows]
            e_w = layer.E_eff_axial_Pa or spec.E_wire_Pa
            for k in range(n_eval):
                if g[k] == 0:
                    continue
                df[k] = swt_slip_cycle_damage(
                    force[k],
                    slipping[k],
                    float(normal[k]),
                    layer.contact_p0_Pa * float(g[k]) ** (1.0 / 3.0),
                    layer.fretting_mu,
                    e_w,
                    layer.contact_nu,
                    layer.swt_const,
                    partial_slip=partial_slip_cycles,
                ).damage
        elif layer.swt_const is not None and layer.contact_p0_Pa > 0.0:
            # Legacy copper estimate: every bulk-stress cycle at gross sliding.
            n_cycles = _cycle_count(sigma_kt)
            g = (
                np.ones(spec.n_wires)
                if layer.contact_shape_g is None
                else layer.contact_shape_g
            )[rows]
            e_w = layer.E_eff_axial_Pa or spec.E_wire_Pa
            for k in range(n_eval):
                if g[k] == 0:
                    continue
                p0_k = layer.contact_p0_Pa * float(g[k]) ** (1.0 / 3.0)  # Hertz scaling
                _sig, _eps, swt = swt_driver_from_contact(
                    p0_k, layer.fretting_mu, e_w, layer.contact_nu
                )
                n_f = swt_life(swt, layer.swt_const)
                if math.isfinite(n_f) and n_f > 0.0:
                    df[k] = float(n_cycles[k]) / n_f
        elif layer.sn_fretting is not None and layer.K_t_contact_N_per_m > 0.0:
            # Custom steel slip-amplitude curve, when explicitly supplied.
            n_cycles = _cycle_count(sigma_kt)
            delta_star_um = _fretting_amplitudes_um(
                layer,
                kx,
                ky,
                None if contact_force_N is None else contact_force_N.get(name),
                wires,
            )
            for k in range(n_eval):
                n_f = layer.sn_fretting.cycles_to_failure(float(delta_star_um[k]))
                if math.isfinite(n_f) and n_f > 0.0:
                    df[k] = float(n_cycles[k]) / n_f

        d_bulk[name] = db
        d_term[name] = dt
        d_fret[name] = df
        with np.errstate(over="ignore", invalid="ignore"):
            d_total[name] = adjusted_db + df
        if not np.isfinite(d_total[name]).all():
            raise CableInputError("fatigue damage exceeds floating-point range")

    # Critical wire across all layers.
    crit_layer, crit_k, crit_val = "", 0, -1.0
    for name, dvec in d_total.items():
        k = int(_first_maximum(dvec[None, :])[0])
        if dvec[k] > crit_val * (1.0 + _TIE_RTOL):
            crit_layer, crit_k, crit_val = name, k, float(dvec[k])

    if crit_layer:
        critical = next(lyr for lyr in layers if lyr.name == crit_layer)
        spec = critical.spec
        crit_wires = selected[crit_layer]
        crit_wire = crit_k if crit_wires is None else int(crit_wires[crit_k])
        phi_crit = math.degrees(2.0 * math.pi * crit_wire / spec.n_wires)
        above_core = (
            None
            if critical.above_core_mask is None
            else bool(critical.above_core_mask[crit_wire])
        )
        parts = {
            "bulk": float(d_bulk[crit_layer][crit_k]),
            "fretting": float(d_fret[crit_layer][crit_k]),
            "termination": float(d_term[crit_layer][crit_k]),
        }
        governing = max(parts, key=lambda kk: parts[kk])
    else:
        phi_crit, above_core, governing = math.nan, None, "bulk"

    with np.errstate(over="ignore"):
        tot = sum(float(np.sum(v)) for v in d_total.values())
        fret_tot = sum(float(np.sum(v)) for v in d_fret.values())
    if not math.isfinite(tot):
        # The ratio can remain representable when summing finite per-wire
        # damage overflows. Keep the original arithmetic in ordinary cases.
        scale = max(float(np.max(v)) for v in d_total.values())
        tot = sum(float(np.sum(v / scale)) for v in d_total.values())
        fret_tot = sum(float(np.sum(v / scale)) for v in d_fret.values())
    frac = fret_tot / tot if tot > 0.0 else 0.0

    # Provenance: which fretting formulations were actually exercised.
    copper_method = (
        "per-cycle SWT strain-life [Poon2022] on slipping cycles of the crossing "
        "tangential force ([Poon2023] Miner sum; Cattaneo-Mindlin line-contact "
        "surface stress; non-slipping cycles "
        + ("skipped" if partial_slip_cycles == "skip" else "given partial-slip drivers")
        + ")"
        if copper_contact_model == "slip_cycle"
        else "legacy SWT strain-life [Poon2022] with one gross-sliding driver per "
        "bulk-stress cycle"
    )
    fret_methods = sorted(
        {
            copper_method
            if lyr.swt_const is not None and lyr.contact_p0_Pa > 0.0
            else "custom steel slip-amplitude S-N"
            if lyr.sn_fretting is not None and lyr.K_t_contact_N_per_m > 0.0
            else "none"
            for lyr in layers
        }
    )
    prov = Provenance(
        method="fixed-material-fibre rainflow + Miner on registered "
        "stress-range curves; "
        "contact-life branch(es): " + ", ".join(fret_methods),
        citations=(
            "[ASTM E1049]",
            "[Miner1945]",
            "[Nasution2014b] Eq.3/5",
            "[Saevik2011] Eq.89",
            "[Poon2022]",
            "[Poon2023]",
        ),
        gates=("A12 rainflow+Miner kernel",),
        limitations=(
            (
                "normal wire stress is supplied from solved section states; "
                "no additional axial or thermal stress is superposed"
                if normal_stress_Pa is not None
                else "thermal stress enters through the supplied thermal_stress_Pa; "
                "prescribed strain alone does not infer radial deformation"
            ),
            (
                "normal contact pressure follows the supplied history; bending-induced "
                if contact_pressure_Pa is not None
                else "normal contact pressure uses the frozen layer reference; "
                "bending-induced "
            )
            + "redistribution of that normal contact is outside this superposition",
            f"bulk damage uses {n_fibres} fixed circumference fibres per wire; "
            "refine this angular discretisation for the analysed history",
            "contact-life terms remain wire-level indicators; their addition to "
            "maximum fibre bulk damage does not locate a common surface hotspot",
            (
                "the SWT contact driver is a closed-form line-contact surface "
                "stress with elastic surface strain, evaluated per slipping cycle; "
                "it does not perform a critical-plane search, include wear or "
                "qualify elliptical-contact fretting life"
                if copper_contact_model == "slip_cycle"
                else "the legacy SWT contact driver assumes gross sliding at every "
                "bulk-stress cycle and elastic surface strain; it does not perform "
                "a critical-plane search or qualify elliptical-contact fretting life"
            ),
        ),
    )
    prov = _sn_option_provenance(prov, layers)
    distributed_layers = [
        lyr for lyr in layers if chosen_model[lyr.name] == "distributed"
    ]
    if distributed_layers:
        # The layer as solved (a constant stick modulus sets its xi).
        solved_specs = {}
        for lyr in distributed_layers:
            stick = None if stick_modulus_Pa is None else stick_modulus_Pa[lyr.name]
            usable, _reason = _distributed_spec(lyr.spec, stick, None)
            solved_specs[lyr.name] = lyr.spec if usable is None else usable
        prov = _distributed_provenance(
            prov,
            distributed_layers,
            distributed_nodes_per_pitch,
            distributed_max_increment,
            solved_specs,
            distributed_bound,
        )
    prov = _friction_model_provenance(
        prov, friction_model, out_of_plane, chosen_model, fallback_reason
    )
    if friction_model == "progressive":
        settings = progressive or ProgressiveFriction()
        prov = Provenance(
            method=prov.method
            + "; armour friction stress: progressive (Iwan) split of the "
            f"direction-tied cap (friction_model='progressive', {settings.describe()})",
            citations=(*prov.citations, "[Iwan1967]", "[Saevik1992]", "[Lukassen2019]"),
            gates=(*prov.gates, "tests/test_progressive_friction.py"),
            limitations=(
                *prov.limitations,
                "progressive friction law: independent sub-elements per wire with "
                "the wire's stick stiffness; caps average to the Papailiou cap; "
                "prescribed mean clamping; the published loops it was compared "
                "with are not reproduced by it (doc/benchmarks.md, gap G8)",
            ),
        )
    if clamp_summary:
        prov = _termination_provenance(prov, clamp_summary)
    law = active_friction_law(friction_law)
    if law is not None:
        prov = Provenance(
            method=prov.method
            + f"; armour friction coefficient: friction law ({law.describe()})",
            citations=prov.citations,
            gates=(*prov.gates, "tests/test_friction_law.py"),
            limitations=(
                *prov.limitations,
                "friction law: the static/dynamic law is compared with measured "
                "wire-on-tape friction loops on one spring-slider only (DAI18 "
                "rows, doc/benchmarks.md), not on a cable section; no rate law "
                "of armour friction is in the validation set; the slip rate of "
                "the direction-tied kernel is a representative wire value "
                "(r / |sin a| times the plastic strain rate); the crossing "
                "contact branches keep the rate-independent coefficient",
            ),
        )
    if count_from:
        prov = dataclasses.replace(
            prov,
            limitations=(
                *prov.limitations,
                f"lead-in: the first {count_from} samples run the friction state "
                "and are not counted",
            ),
        )
    return FatigueResult(
        D_bulk_per_wire=d_bulk,
        D_fretting_per_wire=d_fret,
        D_term_per_wire=d_term,
        D_total_per_wire=d_total,
        phi_critical_deg=phi_crit,
        critical_layer=crit_layer,
        above_core=above_core,
        D_fretting_fraction=frac,
        governing=governing,
        provenance=prov,
        fibre_angles_deg=np.degrees(fibre_angles),
        D_bulk_per_fibre=d_fibre,
        fibre_critical_deg_per_wire=critical_fibre,
        fibre_critical_deg=(
            float(critical_fibre[crit_layer][crit_k]) if crit_layer else math.nan
        ),
        D_bulk_with_termination_per_fibre=d_adjusted_fibre,
        termination_fibre_critical_deg_per_wire=critical_adjusted_fibre,
        clamp_correction=clamp_summary,
        wire_index={
            name: wires.copy() for name, wires in selected.items() if wires is not None
        },
        wire_azimuth_deg={
            lyr.name: np.asarray(360.0 * wires / lyr.spec.n_wires, dtype=np.float64)
            for lyr in layers
            if (wires := selected[lyr.name]) is not None
        },
    )


def _sn_option_provenance(prov: Provenance, layers: list[FatigueLayer]) -> Provenance:
    """Record user S-N curves, mean-stress corrections and component SCFs."""
    parts: list[str] = []
    for layer in layers:
        if not needs_correction(layer.sn_bulk, layer.mean_stress, layer.component_scf):
            continue
        items = [f"curve {layer.sn_bulk.name} ({layer.sn_bulk.reference})"]
        if layer.mean_stress is not None and layer.mean_stress.active:
            items.append(
                f"mean stress: {layer.mean_stress.describe()} to the curve's "
                + (
                    "reference mean"
                    if layer.mean_stress.reference_mean_stress_Pa is not None
                    else f"R = {layer.sn_bulk.R_ratio:g}"
                )
            )
        if layer.component_scf is not None and not layer.component_scf.is_identity:
            items.append(layer.component_scf.describe())
        if layer.sn_bulk.thickness_exponent != 0.0:
            items.append(
                f"thickness exponent {layer.sn_bulk.thickness_exponent:g}, t_ref "
                f"{layer.sn_bulk.reference_thickness_m:g} m"
            )
        parts.append(f"{layer.name}: " + "; ".join(items))
    if not parts:
        return prov
    return Provenance(
        method=prov.method
        + "; user S-N options (corrected cycle by cycle): "
        + " | ".join(parts),
        citations=(
            *prov.citations,
            "[Dowling2013] Ch. 9 mean-stress equations",
            "[DNV-RP-C203] Sec. 2.4-2.5",
        ),
        gates=(*prov.gates, "tests/test_sn_user_options.py"),
        limitations=(
            *prov.limitations,
            "component SCF mean is the time average of the record; mean-stress "
            "corrections use the rainflow mean of each cycle",
        ),
    )


def _clamp_summary(
    spec: TerminationSpec, increment: TerminationIncrement
) -> dict[str, float | bool | str]:
    """Compact per-layer record of the termination correction."""
    return {
        "applied": increment.applied,
        "anchor": spec.anchor,
        "distance_m": spec.distance_m,
        "setback_m": spec.setback_m,
        "biaxial": spec.biaxial,
        "direction_deg": math.degrees(increment.directions_rad[0]),
        "principal_fraction": increment.principal_fraction,
        "kappa_c_1_m": increment.kappa_c_1_m,
        "pitch_m": increment.pitch_m,
        "a_max": increment.a_max,
        "boundary_layer_m": increment.boundary_layer_m,
        "reach_m": increment.reach_m,
        "segments_per_pitch": increment.segments_per_pitch,
        "note": increment.note,
    }


def _termination_provenance(
    prov: Provenance, summary: Mapping[str, Mapping[str, float | bool | str]]
) -> Provenance:
    """Record the anchored-wire clamp correction and its limits."""
    applied = [name for name, item in summary.items() if item["applied"]]
    parts = [
        f"{name}: {item['note']} (distance {item['distance_m']:g} m, set-back "
        f"{item['setback_m']:g} m, a_max {float(item['a_max']):.3g}, l_x "
        f"{float(item['boundary_layer_m']):.3g} m, principal fraction "
        f"{float(item['principal_fraction']):.3f})"
        for name, item in summary.items()
    ]
    method = prov.method
    limitations = list(prov.limitations)
    if applied:
        method += (
            "; armour friction stress near an anchor: sectional law plus the "
            "anchored-wire clamp increment (rigid stick, Coulomb, exact chain "
            "projection, far-field reference of the same solver)"
        )
        limitations.append(
            "termination correction: planar clamp solver along the principal "
            "(and, with biaxial='components', the minor) curvature direction; "
            "axial slip only, rigid stick, one wire per layer without layer "
            "interaction, proportional curvature profile, linearised cap for the "
            "increment; not verified against helix FE or a test; the anchorage "
            "position controls the near-anchor result"
        )
    limitations.append("termination: " + "; ".join(parts))
    return Provenance(
        method=method,
        citations=(
            *prov.citations,
            *(
                ("[KraincanicKebadze2001]", "[RaoofHobbs1984]", "[ZhuTan2014]")
                if applied
                else ()
            ),
        ),
        gates=(
            *prov.gates,
            *(
                ("tests/test_termination.py: far field, decay, clamp law",)
                if applied
                else ()
            ),
        ),
        limitations=tuple(limitations),
    )


def _layer_nodes(
    nodes_per_pitch: int | Mapping[str, int] | None, name: str
) -> int | None:
    if isinstance(nodes_per_pitch, Mapping):
        return nodes_per_pitch[name]
    return nodes_per_pitch


def _check_friction_model(friction_model: str) -> None:
    if friction_model not in FRICTION_MODELS:
        raise CableInputError(
            f"friction_model must be one of {FRICTION_MODELS}, got {friction_model!r}"
        )


def _friction_model_provenance(
    prov: Provenance,
    friction_model: str,
    out_of_plane: float,
    chosen: Mapping[str, str],
    fallback: Mapping[str, str],
) -> Provenance:
    """Record the friction model of every layer and the path planarity."""
    method = prov.method
    limitations = list(prov.limitations)
    planar = out_of_plane <= PLANAR_PATH_TOLERANCE
    if friction_model == "auto":
        method += (
            "; armour friction model 'auto' (curvature path out-of-plane ratio "
            f"{out_of_plane:.3g}, planar at or below {PLANAR_PATH_TOLERANCE:g}): "
            + ", ".join(f"{name}: {model}" for name, model in chosen.items())
        )
        limitations.extend(
            f"friction_model='auto' kept the direction-tied model for {name} on a "
            f"non-planar curvature path because {reason}; the distributed model "
            "does not cover that case"
            for name, reason in fallback.items()
        )
    tied = [name for name in fallback if chosen[name] == "direction_tied"]
    if tied and not planar:
        limitations.append(
            "direction-tied friction cap on a non-planar curvature path "
            f"(out-of-plane ratio {out_of_plane:.3g}; layers "
            + ", ".join(tied)
            + "): the cap turns with the bending direction; "
            f"{DIRECTION_TIED_ROTATION_BIAS} (doc/local_analysis_guide.md, "
            "Section 9)"
        )
    if method == prov.method and len(limitations) == len(prov.limitations):
        return prov
    return dataclasses.replace(prov, method=method, limitations=tuple(limitations))


def _distributed_provenance(
    prov: Provenance,
    layers: list[FatigueLayer],
    nodes_per_pitch: int | Mapping[str, int] | None,
    max_increment: float,
    specs: Mapping[str, HelicalLayerSpec] | None = None,
    bound: str = "signed",
) -> Provenance:
    """Record the distributed friction model and its parameters."""
    parts = []
    for layer in layers:
        spec = layer.spec if specs is None else specs[layer.name]
        chosen = _layer_nodes(nodes_per_pitch, layer.name)
        nodes = DEFAULT_NODES_PER_WIRE * spec.n_wires if chosen is None else int(chosen)
        k_hat = spec.xi / (1.0 - spec.xi) if spec.xi < 1.0 else math.inf
        lam = spec.mu * abs(math.sin(spec.lay_angle_rad))
        parts.append(
            f"{layer.name}: N = {nodes} nodes per pitch, k_hat = {k_hat:.6g}, "
            f"lam = {lam:.6g}"
        )
    bound_text = (
        "capstan bound lam = mu |sin a|"
        if bound == "signed"
        else "uniform bound without the capstan term (stress unit sigma_0 lam, "
        "lam = mu |sin a|)"
    )
    return Provenance(
        method=prov.method
        + "; armour friction stress: distributed elastic-Coulomb helix model "
        f"(friction_model='distributed', {bound_text}, "
        "interface k_hat = xi/(1 - xi), max increment "
        f"{max_increment:g} kappa_on; " + "; ".join(parts) + ")",
        citations=(*prov.citations, "[Papailiou1997]", "[Yun2020]"),
        gates=prov.gates,
        limitations=(
            *prov.limitations,
            "distributed friction model: one wire per layer solved over one pitch "
            "under uniform curvature (no end effects), axial slip only, rigid core "
            "apart from the Yun interface compliance, prescribed mean clamping and "
            "constant interface stiffness; the direction hold fraction does not "
            "apply",
        ),
    )


def _cycle_count(sigma_kt: FloatArray) -> FloatArray:
    """Total rainflow cycle count per wire (used for the fretting branch)."""
    n_w = sigma_kt.shape[0]
    counts = np.zeros(n_w, dtype=np.float64)
    for k in range(n_w):
        counts[k] = sum(c for _r, _m, c, _i0, _i1 in rainflow_cycles(sigma_kt[k]))
    return counts
