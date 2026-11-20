# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Result provenance and validated-range guards.

Two small primitives shared by every public result object and
``compute_*`` entry point:

* :class:`Provenance` — an immutable record attached to every result object
  naming the *method*, its literature *citations*, the validation *gates* that
  back it, and any active *limitations*. A user therefore cannot obtain a number
  without its caveat: the caveat travels with the value.

* :class:`ValidatedRange` + :func:`check_range` — a *soft* guard. When a
  ``compute_*`` input falls outside the envelope a model was validated against,
  the call still proceeds but emits an :class:`ExtrapolationWarning` naming the
  bound and its source. (A *hard*, physically-inadmissible input instead raises
  :class:`cablexsec.exceptions.PhysicalRangeError`.)

Examples
--------
>>> from cablexsec.provenance import Provenance, ValidatedRange, check_range
>>> p = Provenance(method="full-slip EI", citations=("[WitzTan1992b]",),
...                gates=("A8", "BM1"))
>>> "WitzTan" in str(p)
True
>>> import warnings
>>> vr = ValidatedRange("curvature", 0.0, 0.05, "[A11] tested |kappa| <= 0.05 1/m")
>>> with warnings.catch_warnings(record=True) as w:
...     warnings.simplefilter("always")
...     check_range(0.5, vr)
...     bool(w and issubclass(w[0].category, Warning))
True
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cablexsec.exceptions import PhysicalRangeError

__all__ = [
    "ClampingCalibrationWarning",
    "DirectionTiedFrictionWarning",
    "FrictionModelFallbackWarning",
    "ExtrapolationWarning",
    "MissingClampingStateWarning",
    "Provenance",
    "UnevaluatedFatigueCurveWarning",
    "UnmodelledPolymerSofteningWarning",
    "UnvalidatedModelWarning",
    "TorsionalStressWarning",
    "ValidatedRange",
    "check_range",
    "relayed_warnings",
    "require_finite",
]


class ExtrapolationWarning(UserWarning):
    """A ``compute_*`` input is outside the model's validated envelope.

    The result is still returned — the model *may* be sound here, but it has
    not been validated. The warning message names the bound and its source.
    """


class DirectionTiedFrictionWarning(FutureWarning):
    """The default armour friction law of the fatigue path will change.

    Emitted once per call by :func:`cablexsec.api.compute_fatigue` and
    :func:`cablexsec.analysis.fatigue.compute_fatigue` when ``friction_model``
    is left at its default and the curvature path is not planar
    (:func:`cablexsec.analysis.fatigue.curvature_path_out_of_plane_ratio`
    above :data:`cablexsec.analysis.fatigue.PLANAR_PATH_TOLERANCE`). The
    default changes from ``"direction_tied"`` to ``"auto"`` in a later
    release; on a planar path the two are the same, bit for bit, so only
    non-planar paths warn. The direction-tied cap turns with the bending
    direction, and under rotating curvature its friction-stress ranges fall
    below those of the distributed helix model (the message gives the
    measured bias). The result of this release is the direction-tied one,
    unchanged. Pass ``friction_model`` explicitly to keep the current
    behaviour (``"direction_tied"``) or to take the coming one (``"auto"``)
    without the warning.
    """


class FrictionModelFallbackWarning(UserWarning):
    """``friction_model="auto"`` kept the direction-tied law on a non-planar path.

    Emitted by :func:`cablexsec.analysis.fatigue.compute_fatigue` (and so by
    :func:`cablexsec.api.compute_fatigue`) and by
    :func:`cablexsec.analysis.fatigue.wire_stress_field` once for every layer
    that ``"auto"`` would solve with the distributed helix model on a
    non-planar curvature path but cannot: a rigid interface (``xi >= 1``), a
    stick modulus that changes over the record or a rate-dependent friction
    law. That layer keeps the direction-tied cap, whose friction-stress
    ranges fall below the distributed ones under rotating curvature; the
    provenance limitations record the same reason.
    """


class UnevaluatedFatigueCurveWarning(UserWarning):
    """A cable layer declares an S-N or contact-life curve that is not evaluated.

    Emitted by :func:`cablexsec.api.compute_fatigue` when a layer names a
    ``bulk_sn`` or ``fretting_sn`` curve that the wire-fatigue calculation does
    not use: for example a conductor or sheath (not a structural helical
    armour layer), or an armour ``fretting_sn`` with no direct crossed-wire
    contact. The returned damage then excludes that curve; the same note is
    recorded in the result's provenance limitations.
    """


class UnvalidatedModelWarning(UserWarning):
    """An opt-in model that failed its validation gate was requested.

    The result is returned with the model included, but the model is known not
    to reproduce the published tests it was checked against (the message names
    the gate). Emitted, for example, when conductor trellis fretting is
    requested (gap 1.3c).
    """


class MissingClampingStateWarning(UserWarning):
    """A friction-dependent result rests on no declared manufacturing clamping.

    Emitted when the armour contact pressure of a cable is formed and a
    structural armour layer has no ``manufacturing_clamping`` declaration
    (and the cable no identified ``initial_state``), or when the clamping is
    zero because it is declared as not published. Bending hysteresis, armour
    friction stress, fretting loads and their changes with temperature then
    rest on tension-induced contact alone (:mod:`cablexsec.clamping`); at low
    tension they tend to zero, so a low-tension hysteresis is unreliable.
    """


class ClampingCalibrationWarning(UserWarning):
    """A calibrated clamping value is combined with friction it already holds.

    Emitted by the laid-core stick-slip law when the cable's initial pressure
    or an armour clamping was calibrated with the armour friction alone
    (``calibration_scope="armour_only"``): that value was raised until the
    armour matched the measured dissipation, so it already carries the
    friction of the other contacts, which the law then adds a second time.
    Re-identify the value with the friction model in use
    (:mod:`cablexsec.clamping`).
    """


class TorsionalStressWarning(UserWarning):
    """The cable twists, but the torsional wire stress cannot be formed or is omitted.

    Emitted by :func:`cablexsec.io.cabledyn.read_cabledyn_sections` when
    CableDyn writes ``Twist<L>N<J>`` or ``Torq<L>N<J>`` at a section but not
    the ``Twist`` channels of the section's element(s) from which the twist
    rate is formed (the message names them), and by the sea-state batch when
    a section's history has a twist rate that is not applied. A fatigue result
    of such a history then has no torsional wire stress.
    """


class UnmodelledPolymerSofteningWarning(UserWarning):
    """A polymer layer runs hot with a temperature-independent modulus.

    Emitted by the thermal coupling when a polymer layer without a registered
    ``E(T)`` law (and without an explicit ``modulus_transfer``) exceeds
    :data:`cablexsec.analysis.thermal_coupling.POLYMER_SOFTENING_WARNING_C`.
    Its modulus, and so its share of the thermal clamping, is then that of
    the reference temperature.
    """


@dataclass(frozen=True)
class Provenance:
    """Immutable record of how a result was produced and how far to trust it.

    Attributes
    ----------
    method : str
        One-line description of the computation (e.g. ``"Saevik-Thorsen
        min(P1,P2) bird-caging"``).
    citations : tuple[str, ...]
        Literature sources in the ``[AuthorYear]`` convention.
    gates : tuple[str, ...]
        Validation gate identifiers backing the result (e.g. ``"A11"``,
        ``"BM5"``, ``"H1"``).
    limitations : tuple[str, ...]
        Active caveats the user must see — idealizations, screening-only
        status, blocked sub-models, calibration omissions.
    """

    method: str
    citations: tuple[str, ...] = ()
    gates: tuple[str, ...] = ()
    limitations: tuple[str, ...] = field(default=())

    def __str__(self) -> str:
        parts = [f"method: {self.method}"]
        if self.citations:
            parts.append("cites: " + ", ".join(self.citations))
        if self.gates:
            parts.append("gates: " + ", ".join(self.gates))
        if self.limitations:
            parts.append("LIMITATIONS:\n  - " + "\n  - ".join(self.limitations))
        return "\n".join(parts)


@dataclass(frozen=True)
class ValidatedRange:
    """A closed interval ``[low, high]`` a model input was validated against.

    Attributes
    ----------
    name : str
        Human name of the quantity (used in the warning message).
    low, high : float
        Inclusive validated bounds (same unit as the checked value).
    bound_cite : str
        Source of the bound (gate ID, benchmark, or material limit) — shown in
        the warning so the user knows *why* this is the edge of validation.
    """

    name: str
    low: float
    high: float
    bound_cite: str

    def contains(self, value: float) -> bool:
        """True if ``value`` is within ``[low, high]`` (inclusive)."""
        return self.low <= value <= self.high


def check_range(value: float, vr: ValidatedRange, *, stacklevel: int = 3) -> bool:
    """Warn (don't raise) if ``value`` is outside the validated range ``vr``.

    Returns ``True`` if within range, ``False`` (after warning) if outside.
    Emits :class:`ExtrapolationWarning` naming the offending quantity, the value,
    the breached bound, and its source.
    """
    if vr.contains(value):
        return True
    edge = "above" if value > vr.high else "below"
    bound = vr.high if value > vr.high else vr.low
    warnings.warn(
        f"{vr.name} = {value:.4g} is {edge} the validated range "
        f"[{vr.low:.4g}, {vr.high:.4g}] ({vr.bound_cite}); result is an "
        f"extrapolation beyond {bound:.4g} and is not validated there.",
        ExtrapolationWarning,
        stacklevel=stacklevel,
    )
    return False


@contextmanager
def relayed_warnings(*, stacklevel: int = 3) -> Iterator[None]:
    """Gather the warnings of internal calls and pass each distinct one on once.

    For a function that evaluates other entry points on its own sweeps: their
    warnings (a curvature beyond the wire-yield bound, an undeclared clamping
    state) concern the caller's input too, so they are relayed, not silenced.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    seen: set[tuple[type[Warning], str]] = set()
    for item in caught:
        key = (item.category, str(item.message))
        if key not in seen:
            seen.add(key)
            warnings.warn(str(item.message), item.category, stacklevel=stacklevel)


def require_finite(value: float | Iterable[float], name: str) -> None:
    """Raise :class:`PhysicalRangeError` if any element of ``value`` is non-finite.

    A NaN/inf curvature, tension, or temperature is physically inadmissible and
    would otherwise propagate to a silently-wrong (NaN) result — this turns it
    into a loud, actionable failure (a *hard* domain error, distinct from the
    *soft* validated-range :func:`check_range` warning).
    """
    if np.iscomplexobj(cast(ArrayLike, value)):
        raise PhysicalRangeError(f"{name} must contain real values")
    if isinstance(value, np.ndarray):
        ok = bool(np.isfinite(value).all())
    elif isinstance(value, (int, float, np.number)):
        ok = math.isfinite(float(value))
    else:
        ok = all(math.isfinite(float(v)) for v in value)
    if not ok:
        raise PhysicalRangeError(
            f"{name} contains a non-finite value (NaN or inf); a physically "
            f"admissible {name} is required. Check the input before calling."
        )


def real_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Convert an input without silently discarding complex components."""
    try:
        if np.iscomplexobj(value):
            raise PhysicalRangeError(f"{name} must contain real values")
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise PhysicalRangeError(f"{name} must contain real numeric values") from error
