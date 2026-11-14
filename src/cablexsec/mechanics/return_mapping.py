# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Elastic-perfectly plastic Coulomb return-mapping element.

The scalar element gives a factor-two reversal range at fixed stiffness and
yield limit. Vector radial return extends this discrete constitutive model
to nonproportional increments. It does not reconstruct a Cattaneo-Mindlin
partial-slip traction field or establish Masing similarity for arbitrary
biaxial paths, variable normal loads or variable stiffness.

The element is instantiated twice in the library:

* **1-D scalar** — per-wire friction stress in the M-κ engine. The driving
  variable is the signed curvature projection; the yield limit is the Coulomb
  saturation ``mu * F_n / A_w``.
* **2-D vector** — tangential force at a wire-wire contact in the fretting
  (Mindlin) engine. The driving variable is the 2-D slip-increment vector; the
  yield limit ``mu * F_n`` is refreshed every step (non-proportional loading).

Per increment ``du`` of the driving kinematic variable, with elastic stiffness
``k`` and (possibly step-varying) yield limit ``f_y``::

    f_trial = f_prev + k * du
    if |f_trial| <= f_y:        # stick (elastic)
        f = f_trial
        slip = 0
    else:                       # slip (return to the surface)
        f = f_y * f_trial / |f_trial|
        slip = du - (f - f_prev) / k

``|.|`` is the absolute value in 1-D and the Euclidean norm (over the last
axis) in 2-D. Both forms are vectorised over a population of elements via the
leading array dimensions.

References
----------
[Saevik2011] §5.2, Eqs. 54--55 (Coulomb saturation); [Kim2020] (stick branch);
Mindlin/Cattaneo partial-slip mechanics are distinct from this return map.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from cablexsec.exceptions import CableInputError

__all__ = ["CoulombElement", "coulomb_return_map"]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _finite_real(values: float | FloatArray, name: str) -> FloatArray:
    if np.iscomplexobj(values):
        raise CableInputError(f"{name} must be real")
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise CableInputError(f"{name} must be finite")
    return result


def _stiffness_value(value: float) -> float:
    message = "elastic stiffness k must be real, finite and positive"
    if np.iscomplexobj(value) or np.ndim(value) != 0:
        raise CableInputError(message)
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise CableInputError(message)
    return result


def _project_vector(trial: FloatArray, cap: FloatArray) -> tuple[FloatArray, BoolArray]:
    # An unrepresentable trial norm is allowed if its components and the
    # projected force are representable. Normalize before multiplying by cap;
    # cap/|trial| itself can underflow even when the projected force is finite.
    with np.errstate(over="ignore"):
        magnitude = np.hypot(trial[..., 0], trial[..., 1])
    slipping = magnitude > cap
    scale = np.max(np.abs(trial), axis=-1)
    unit = trial / np.where(scale > 0, scale, 1)[..., None]
    length = np.hypot(unit[..., 0], unit[..., 1])
    projected = unit * (cap / np.where(length > 0, length, 1))[..., None]
    result = np.where(slipping[..., None], projected, trial)
    # Keep the representable result inside the closed yield disk. Otherwise a
    # rounding overshoot causes fictitious plastic slip on an unchanged hold.
    for _ in range(2):
        outside = np.hypot(result[..., 0], result[..., 1]) > cap
        if not np.any(outside):
            break
        result = np.where(outside[..., None], np.nextafter(result, 0), result)
    if np.any(np.hypot(result[..., 0], result[..., 1]) > cap):
        raise CableInputError("vector projection cannot represent the yield constraint")
    return np.asarray(result), np.asarray(slipping)


def _scalar_return_map(
    previous: FloatArray,
    increment: FloatArray,
    stiffness: float | FloatArray,
    cap: FloatArray,
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """Shared arithmetic for validated scalar populations.

    Internal callers must establish finite real, broadcast-compatible inputs,
    positive stiffness and nonnegative caps, and trap floating-point overflow,
    invalid operations and division by zero. This is not an unchecked public API.
    """
    trial = previous + stiffness * increment
    slipping = np.abs(trial) > cap
    force = np.where(slipping, np.copysign(cap, trial), trial)
    slip = np.where(slipping, increment - (force - previous) / stiffness, 0.0)
    return np.asarray(force), np.asarray(slip), np.asarray(slipping)


def coulomb_return_map(
    f_prev: float | FloatArray,
    du: float | FloatArray,
    k: float | FloatArray,
    f_y: float | FloatArray,
    *,
    vector: bool = False,
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """One radial-return increment (pure function, no state).

    Parameters
    ----------
    f_prev : numpy.ndarray
        Internal force at the start of the increment. Shape ``(...,)`` for 1-D
        or ``(..., 2)`` for 2-D.
    du : numpy.ndarray
        Increment of the driving kinematic variable, same shape as ``f_prev``.
    k : float or numpy.ndarray
        Finite positive elastic stiffness per element. In vector mode its
        shape broadcasts over the population, never the component axis.
    f_y : numpy.ndarray
        Yield limit (magnitude), shape ``(...,)``. May vary per increment.
    vector : bool, optional
        If ``True``, treat the last axis as a 2-D vector and use the Euclidean
        norm; otherwise scalar absolute value.

    Returns
    -------
    f_new : numpy.ndarray
        Updated internal force, same shape as ``f_prev``.
    slip : numpy.ndarray
        Plastic slip increment (zero where the element sticks), same shape as
        ``f_prev``.
    slipping : numpy.ndarray
        Boolean mask (shape ``(...,)``) — ``True`` where the element slipped.

    Inputs must be real and finite; yield limits must be nonnegative. Vector
    inputs have a final component dimension of two (scalars broadcast to both
    components). Incompatible shapes and unrepresentable trial components or
    slip increments raise ValueError. Inputs are never modified.
    """
    previous = _finite_real(f_prev, "previous force")
    increment = _finite_real(du, "increment")
    stiffness = _finite_real(k, "stiffness")
    cap = _finite_real(f_y, "yield limit")
    if np.any(stiffness <= 0) or np.any(cap < 0):
        raise CableInputError("stiffness must be positive and yield limit nonnegative")
    if vector:
        previous = np.broadcast_to(previous, (2,)) if previous.ndim == 0 else previous
        increment = (
            np.broadcast_to(increment, (2,)) if increment.ndim == 0 else increment
        )
        if previous.shape[-1] != 2 or increment.shape[-1] != 2:
            raise CableInputError("vector force and increment must have two components")
        stiffness = stiffness[..., None]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            if vector:
                trial = previous + stiffness * increment
                f_new, slipping = _project_vector(trial, cap)
                slip = np.where(
                    slipping[..., None], increment - (f_new - previous) / stiffness, 0.0
                )
            else:
                f_new, slip, slipping = _scalar_return_map(
                    previous, increment, stiffness, cap
                )
        except FloatingPointError as error:
            raise CableInputError(
                "return-map intermediate exceeds floating-point range"
            ) from error
    return np.asarray(f_new), np.asarray(slip), np.asarray(slipping)


@dataclass
class CoulombElement:
    """Stateful Coulomb return-mapping element (1-D or 2-D, vectorised).

    Parameters
    ----------
    k : float
        Elastic stiffness relating the internal force to the driving variable.
    vector : bool, optional
        ``False`` (default) for a 1-D scalar element; ``True`` for a 2-D
        vector element (Euclidean norm over the last axis).
    shape : tuple of int, optional
        Leading shape of the element population (e.g. ``(n_wires,)``). The
        stored force has shape ``shape`` (1-D) or ``shape + (2,)`` (2-D).

    Notes
    -----
    The element accumulates the total slip trajectory in :attr:`slip_total`,
    which the fretting engine uses to form the fretting amplitude ``delta*``.
    """

    k: float
    vector: bool = False
    shape: tuple[int, ...] = ()
    f: FloatArray = field(init=False)
    slip_total: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        self.k = _stiffness_value(self.k)
        self.reset()

    def set_stiffness(self, k: float) -> None:
        """Change stiffness at fixed elastic displacement and plastic slip.

        A material change scales the stored force, not the accumulated slip.
        The following update enforces the current friction limit, including
        when displacement is held fixed while normal load decreases.
        """
        k = _stiffness_value(k)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                # Decompose the three factors so a finite result is retained
                # when k/self.k alone would overflow or underflow.
                force_mantissa, force_exponent = np.frexp(self.f)
                new_mantissa, new_exponent = np.frexp(k)
                old_mantissa, old_exponent = np.frexp(self.k)
                force = np.ldexp(
                    force_mantissa * (new_mantissa / old_mantissa),
                    force_exponent + new_exponent - old_exponent,
                )
            except FloatingPointError as error:
                raise CableInputError(
                    "rescaled force exceeds floating-point range"
                ) from error
        if not np.isfinite(force).all():
            raise CableInputError("rescaled force must be finite")
        if np.any((self.f != 0) & (force == 0)):
            raise CableInputError("rescaled force is outside floating-point range")
        self.f = force
        self.k = k

    def _state_shape(self) -> tuple[int, ...]:
        return (*self.shape, 2) if self.vector else self.shape

    def reset(self) -> None:
        """Zero the internal force and accumulated slip."""
        self.f = np.zeros(self._state_shape(), dtype=np.float64)
        self.slip_total = np.zeros(self._state_shape(), dtype=np.float64)

    def update(
        self, du: float | FloatArray, f_y: float | FloatArray
    ) -> tuple[FloatArray, BoolArray]:
        """Advance the element by one increment.

        Parameters
        ----------
        du : numpy.ndarray
            Increment of the driving variable (broadcastable to the state).
        f_y : numpy.ndarray
            Current yield limit (magnitude), broadcastable to :attr:`shape`.

        Returns
        -------
        f : numpy.ndarray
            The updated internal force (a view of :attr:`f`).
        slipping : numpy.ndarray
            Boolean mask of elements that slipped this increment.
        """
        state_shape = self._state_shape()
        increment_shape, cap_shape = np.shape(du), np.shape(f_y)
        if (
            increment_shape not in ((), state_shape)
            and np.broadcast_shapes(increment_shape, state_shape) != state_shape
        ) or (
            cap_shape not in ((), self.shape)
            and np.broadcast_shapes(cap_shape, self.shape) != self.shape
        ):
            raise CableInputError(
                "update must preserve the declared element population"
            )
        f_new, slip, slipping = coulomb_return_map(
            self.f, du, self.k, f_y, vector=self.vector
        )
        with np.errstate(over="raise", invalid="raise"):
            try:
                total = self.slip_total + slip
            except FloatingPointError as error:
                raise CableInputError(
                    "accumulated slip exceeds floating-point range"
                ) from error
        if not np.isfinite(total).all():
            raise CableInputError("accumulated slip must be finite")
        self.f = f_new
        self.slip_total = total
        return self.f, slipping
