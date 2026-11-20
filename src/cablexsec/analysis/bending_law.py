# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Hysteretic bending law of a cable section for global analysis (gap MX).

In planar bending at fixed contact pressure the default armour law gives every
wire an elastic, perfectly plastic spring--slider with a fixed slip interval,
in parallel with the elastic full-slip stiffness. With symmetric intervals the
section law is an Iwan parallel model, which obeys Masing's rules exactly
([Iwan1966]). The signed capstan bound of [Papailiou1997] makes the interval
of a wire asymmetric (its falling bound is the rising one times
``exp(-lam psi)``), so an Iwan law fitted to the virgin backbone misses the
section law's loops by a residual: about 0.1 % of the moment range on
bending cycles of one amplitude near full slip. On nested irregular
histories it is larger: it peaks on reloading branches in the doubled
transition zone and grows with the capstan factor, to 0.4-0.6 % of the
moment range at ``lam`` about 0.09 (global--local guide, Section 8). The
memory of earlier reversals is unaffected. A piecewise-linear
virgin backbone ``M(kappa)`` with non-increasing tangent decomposes uniquely
into parallel elastic, perfectly plastic elements:

.. math::

    M(\kappa) = EI_{slip}\,\kappa + \sum_i k_i\, e_i, \qquad
    e_i \leftarrow \operatorname{clip}(e_i + \Delta\kappa, -\kappa_{y,i},
    \kappa_{y,i}),

with ``k_i = t_i - t_{i+1}`` the drop of the backbone tangent at the
breakpoint ``kappa_{y,i}`` (``t`` after the last breakpoint is ``EI_slip``).
The set reproduces the backbone on virgin loading and every unloading and
reloading branch by Masing's rules, including the memory of earlier
reversals.

:func:`export_bending_law` computes the backbone with
:func:`cablexsec.api.compute_mkappa` at a set of tensions and returns a
:class:`BendingLawExport` with one :class:`IwanBendingLaw` per tension, linear
interpolation in tension (concave backbones stay concave) and writers for a
generic CSV/JSON table and a moment--curvature table in kN and m.

Limits: planar bending. The per-wire, direction-tied biaxial response of
CableXSec is not an isotropic hysteretic law; a global solver that applies the
exported law to the resultant curvature uses an assumption the section model
does not make. Tension and annulus state are held fixed within one law.

Slip scaling variable
---------------------
With ``slip_scale=True`` the export also carries a :class:`SlipScaleTable`:
the law at a reference state split into friction groups (one per structural
armour layer and one for the laid-core stage), and the scale of every group
over a table of tensions and, optionally, external pressures. In the
direction-tied law every wire of a group is a spring--slider of fixed
stiffness ``k_f = xi E cos^2 a`` whose friction bounds are proportional to
the group's friction-setting pressure ``P1+P2`` (``sigma_0`` of
:func:`~cablexsec.mechanics.papailiou.wire_slip_stress_bounds`; for the
laid cores, to their contact line load while its load-weighted friction
coefficient is unchanged). The return map
``f <- clip(f + k d, -rho l, rho u)`` is positively homogeneous, so a group
whose bounds scale by ``rho`` obeys

.. math::

    M_g(\kappa; \rho) = \rho\, M_g(\kappa/\rho; 1)

over any curvature history, and its Iwan elements keep their stiffnesses
while their yield curvatures scale: ``rho sum k_i clip(kappa/rho, ...,
kappa_y,i) = sum k_i clip(kappa, ..., rho kappa_y,i)``. The group scale is
``rho_g = R_g / R_g,ref`` with ``R_g`` the group's planar full-slip friction
moment (its cap). For a changing scale every element follows the return map
with the new cap, the rule the fatigue kernels apply to every wire:

.. math::

    e_i \leftarrow \operatorname{clip}(e_i + \Delta\kappa, -\rho_g \kappa_{y,i},
    \rho_g \kappa_{y,i}), \qquad M = EI_{slip}\,\kappa + \sum_i k_i e_i .

The single scale ``s = sum_g R_g / sum_g R_g,ref`` (cap-weighted) is exact
when every group scales alike; otherwise it keeps the stick stiffness and the
full-slip moment exact and misplaces the transition (quantified in the
global--local guide, Section 8).

References
----------
[Iwan1966] W.D. Iwan, A distributed-element model for hysteresis and its
steady-state dynamic response, J. Appl. Mech. 33 (1966) 893-900.
[Papailiou1997] Eqs. (9)-(15) (the signed capstan bound;
:mod:`cablexsec.mechanics.papailiou`).
"""

from __future__ import annotations

import dataclasses
import json
import math
import numbers
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.friction_law import FrictionLaw
from cablexsec.mechanics.papailiou import (
    _compression_onset_factor,
    full_slip_curvature,
    slip_onset_curvature,
    wire_slip_stress_bounds,
)
from cablexsec.provenance import Provenance

if TYPE_CHECKING:
    from cablexsec.api import _MkappaLaw
    from cablexsec.cables.schema import CableData
    from cablexsec.mechanics.mkappa import HelicalLayerSpec

__all__ = [
    "CORE_GROUP",
    "FILLER_GROUP",
    "BendingLawExport",
    "IwanBendingLaw",
    "SlipGroupLaw",
    "SlipScaleTable",
    "export_bending_law",
]

FloatArray = NDArray[np.float64]

#: Relative tolerance on a rising backbone tangent (numerical noise of the
#: section law); larger rises are rejected as a non-Masing backbone.
_TANGENT_RISE_TOLERANCE = 1e-6

#: Name of the friction group of the laid-core stage in a :class:`SlipScaleTable`.
CORE_GROUP = "laid_cores"

#: Name of the friction group of the laid filler profiles in a
#: :class:`SlipScaleTable` (``CoreFrictionSpec(filler_profiles=True)``).
FILLER_GROUP = "filler_profiles"

#: Relative step of the central differences of the group caps.
_DIFFERENCE_STEP = 1e-4

#: Floor of the base geometric curvature grid, relative to ``kappa_max``.
_GRID_FLOOR = 1e-4

#: The first curvature of the grid lies at most this fraction of the smallest
#: first-slip curvature, so that the first chord of every backbone is in stick.
_STICK_MARGIN = 0.5

#: Deepest extension of the geometric grid below its base floor [decades].
_MAX_EXTENSION_DECADES = 6.0

#: Grid curvatures closer than this relative gap are merged.
_MERGE_GAP = 1e-9

#: Relative tolerance of the exported stick stiffness against the section's.
_STICK_TOLERANCE = 1e-6


@dataclass(frozen=True)
class IwanBendingLaw:
    """Parallel elastic, perfectly plastic elements plus the full-slip stiffness.

    Attributes
    ----------
    ei_slip_Nm2 : float
        Stiffness of the always-elastic part (full-slip slope) [N m^2].
    stiffness_Nm2 : numpy.ndarray
        Stiffness of each element [N m^2], non-negative.
    yield_curvature_1_m : numpy.ndarray
        Curvature at which each element slips [1/m], positive, ascending.
    """

    ei_slip_Nm2: float
    stiffness_Nm2: FloatArray
    yield_curvature_1_m: FloatArray

    def __post_init__(self) -> None:
        k = np.asarray(self.stiffness_Nm2, dtype=np.float64)
        ky = np.asarray(self.yield_curvature_1_m, dtype=np.float64)
        if k.ndim != 1 or k.shape != ky.shape:
            raise CableInputError(
                "element stiffness and yield must be equal 1-D arrays"
            )
        if not (np.isfinite(k).all() and np.isfinite(ky).all()):
            raise CableInputError("element data must be finite")
        if np.any(k < 0) or np.any(ky <= 0) or np.any(np.diff(ky) <= 0):
            raise CableInputError(
                "elements need k >= 0 and positive, strictly ascending yield curvatures"
            )
        if not (math.isfinite(self.ei_slip_Nm2) and self.ei_slip_Nm2 >= 0):
            raise CableInputError("ei_slip_Nm2 must be finite and >= 0")
        object.__setattr__(self, "stiffness_Nm2", k)
        object.__setattr__(self, "yield_curvature_1_m", ky)

    @property
    def ei_stick_Nm2(self) -> float:
        """Initial (no-slip) tangent stiffness [N m^2]."""
        return float(self.ei_slip_Nm2 + self.stiffness_Nm2.sum())

    @classmethod
    def from_backbone(
        cls,
        kappa_1_m: ArrayLike,
        moment_Nm: ArrayLike,
        ei_slip_Nm2: float,
    ) -> IwanBendingLaw:
        """Decompose a piecewise-linear virgin backbone into Iwan elements.

        ``kappa_1_m`` starts at zero and rises strictly; ``moment_Nm`` starts
        at zero. The tangent must not rise (beyond numerical noise, which is
        removed by a running minimum) and must not fall below ``ei_slip_Nm2``
        after the last point, which continues the backbone.
        """
        k = np.asarray(kappa_1_m, dtype=np.float64)
        m = np.asarray(moment_Nm, dtype=np.float64)
        if k.ndim != 1 or k.shape != m.shape or k.size < 2:
            raise CableInputError(
                "backbone needs equal 1-D arrays of two or more points"
            )
        if not (np.isfinite(k).all() and np.isfinite(m).all()):
            raise CableInputError("backbone must be finite")
        if k[0] != 0.0 or m[0] != 0.0 or np.any(np.diff(k) <= 0):
            raise CableInputError(
                "backbone must start at (0, 0) and rise strictly in curvature"
            )
        tangent = np.diff(m) / np.diff(k)
        scale = max(float(np.max(np.abs(tangent))), 1e-30)
        rise = np.diff(tangent)
        if np.any(rise > _TANGENT_RISE_TOLERANCE * scale):
            raise CableInputError(
                "backbone tangent rises: the section law is not of Masing type here"
            )
        tangent = np.minimum.accumulate(tangent)
        # A full-slip plateau of a friction-only backbone (ei_slip_Nm2 = 0)
        # may fall below zero by rounding; such noise is zero.
        tangent = np.where(
            (tangent < 0.0) & (tangent > -_TANGENT_RISE_TOLERANCE * scale),
            0.0,
            tangent,
        )
        if not (math.isfinite(ei_slip_Nm2) and ei_slip_Nm2 >= 0):
            raise CableInputError("ei_slip_Nm2 must be finite and >= 0")
        tail = min(float(ei_slip_Nm2), float(tangent[-1]))
        following = np.append(tangent[1:], tail)
        stiffness = tangent - following
        return cls(tail, stiffness, k[1:].copy())

    def backbone(self, kappa_1_m: ArrayLike) -> FloatArray:
        """Virgin-loading moment at curvatures of either sign [N m]."""
        k = np.asarray(kappa_1_m, dtype=np.float64)
        e = np.clip(np.abs(k)[..., None], 0.0, self.yield_curvature_1_m)
        return np.sign(k) * (self.ei_slip_Nm2 * np.abs(k) + e @ self.stiffness_Nm2)

    def moment(self, kappa_history: ArrayLike) -> FloatArray:
        """Moment history [N m] over a curvature history from the virgin state."""
        kappa = np.asarray(kappa_history, dtype=np.float64)
        if kappa.ndim != 1 or not np.isfinite(kappa).all():
            raise CableInputError("curvature history must be a finite 1-D array")
        e = np.zeros_like(self.stiffness_Nm2)
        out = np.empty_like(kappa)
        previous = 0.0
        for n, value in enumerate(kappa):
            e = np.clip(
                e + (value - previous),
                -self.yield_curvature_1_m,
                self.yield_curvature_1_m,
            )
            previous = value
            out[n] = self.ei_slip_Nm2 * value + float(e @ self.stiffness_Nm2)
        return out

    def dissipated_energy(self, kappa_history: ArrayLike) -> float:
        """Energy dissipated over the history [J/m] (sum of slider work)."""
        kappa = np.asarray(kappa_history, dtype=np.float64)
        e = np.zeros_like(self.stiffness_Nm2)
        work = 0.0
        previous = 0.0
        for value in kappa:
            trial = e + (value - previous)
            clipped = np.clip(
                trial, -self.yield_curvature_1_m, self.yield_curvature_1_m
            )
            # plastic increment trial - clipped, at the element moment k * clipped
            work += float(
                np.sum(self.stiffness_Nm2 * np.abs(clipped) * np.abs(trial - clipped))
            )
            e = clipped
            previous = value
        return work


def _ratio(values: FloatArray, reference: float) -> FloatArray:
    return np.asarray(values / reference, dtype=np.float64)


def _bracket(
    grid: FloatArray, values: FloatArray, label: str
) -> tuple[NDArray[np.intp], NDArray[np.intp], FloatArray]:
    """Lower and upper grid index and weight of every value (linear)."""
    if not np.isfinite(values).all():
        raise CableInputError(f"{label} must be finite")
    if np.any(values < grid[0]) or np.any(values > grid[-1]):
        raise CableInputError(
            f"{label} outside the table range {grid[0]:g}-{grid[-1]:g}"
        )
    if grid.size == 1:
        zero = np.zeros(values.shape, dtype=np.intp)
        return zero, zero, np.zeros(values.shape)
    lower = np.clip(np.searchsorted(grid, values) - 1, 0, grid.size - 2)
    weight = (values - grid[lower]) / (grid[lower + 1] - grid[lower])
    return lower, lower + 1, weight


@dataclass(frozen=True)
class SlipScaleTable:
    r"""Slip scaling variable of an exported law over tension and pressure.

    The law at the reference state is split into friction groups: one per
    structural armour layer (named after it), :data:`CORE_GROUP` for the
    laid-core stage and :data:`FILLER_GROUP` for its laid filler profiles.
    ``group_laws[g]`` holds the group's elements (fixed
    stiffness ``k_i``, reference yield curvature ``kappa_y,i``; no elastic
    part), ``ei_slip_Nm2`` the elastic full-slip stiffness. At a state with
    group caps ``R_g`` the law is the reference law with the yield curvatures
    of group ``g`` multiplied by ``rho_g = R_g / R_g,ref``; with one scale for
    all groups it is ``s = sum_g R_g / sum_g R_g,ref``.

    A changing state moves the caps, not the stiffnesses. Each element then
    follows the return map with the new cap,
    ``e_i <- clip(e_i + d kappa, -rho kappa_y,i, rho kappa_y,i)``: its slip
    state is carried across, a falling cap clips the element at fixed
    curvature (friction work) and a rising cap leaves a sticking element
    unchanged (:meth:`moment`).

    The group laws come from backbones on the export's curvature grid, so a
    group of scale ``rho`` reproduces its law at the state up to
    ``rho kappa_max``; beyond, the full-slip slope continues. The state is
    interpolated linearly in the table (bilinearly with a pressure axis).

    Attributes
    ----------
    reference_tension_N, reference_external_pressure_Pa : float
        State of the reference law.
    tension_N : numpy.ndarray
        Table tensions [N], strictly ascending.
    external_pressure_Pa : numpy.ndarray or None
        Table external pressures [Pa], strictly ascending; ``None`` when the
        table runs over tension only (at the reference external pressure).
    group_names : tuple of str
        Armour layer names, then :data:`CORE_GROUP` when the cores slip and
        :data:`FILLER_GROUP` when the filler profiles do.
    group_laws : tuple of IwanBendingLaw
        Elements of every group at the reference state (``ei_slip_Nm2`` 0).
    ei_slip_Nm2 : float
        Full-slip (elastic) stiffness [N m^2], independent of the state.
    reference_cap_Nm : numpy.ndarray
        Full-slip friction moment ``R_g`` of every group at the reference [N m].
    group_cap_Nm : numpy.ndarray
        ``R_g`` at every table state, shape (groups, tensions, pressures) [N m];
        one pressure column without a pressure axis.
    group_cap_dT_Nm_per_N : numpy.ndarray
        ``dR_g/dT`` [N m/N], same shape (central differences of the contact
        model, one-sided at zero tension).
    group_cap_dpe_Nm_per_Pa : numpy.ndarray or None
        ``dR_g/dp_e`` [N m/Pa], same shape; ``None`` without a pressure axis.
    """

    reference_tension_N: float
    reference_external_pressure_Pa: float
    tension_N: FloatArray
    external_pressure_Pa: FloatArray | None
    group_names: tuple[str, ...]
    group_laws: tuple[IwanBendingLaw, ...]
    ei_slip_Nm2: float
    reference_cap_Nm: FloatArray
    group_cap_Nm: FloatArray
    group_cap_dT_Nm_per_N: FloatArray
    group_cap_dpe_Nm_per_Pa: FloatArray | None = None

    def __post_init__(self) -> None:
        t = np.asarray(self.tension_N, dtype=np.float64)
        p = (
            None
            if self.external_pressure_Pa is None
            else np.asarray(self.external_pressure_Pa, dtype=np.float64)
        )
        for grid, label in ((t, "tension_N"), (p, "external_pressure_Pa")):
            if grid is None:
                continue
            if (
                grid.ndim != 1
                or grid.size < 1
                or not np.isfinite(grid).all()
                or np.any(grid < 0.0)
                or np.any(np.diff(grid) <= 0.0)
            ):
                raise CableInputError(
                    f"{label} of a slip-scale table must be finite, non-negative "
                    "and strictly ascending"
                )
        n_groups = len(self.group_names)
        if n_groups < 1 or len(self.group_laws) != n_groups:
            raise CableInputError("one law per friction group is required")
        if len(set(self.group_names)) != n_groups:
            raise CableInputError("friction group names must be distinct")
        reference = np.asarray(self.reference_cap_Nm, dtype=np.float64)
        if (
            reference.shape != (n_groups,)
            or not np.isfinite(reference).all()
            or np.any(reference <= 0.0)
        ):
            raise CableInputError(
                "every friction group needs a finite, positive reference cap"
            )
        shape = (n_groups, t.size, 1 if p is None else p.size)
        cap = np.asarray(self.group_cap_Nm, dtype=np.float64)
        d_t = np.asarray(self.group_cap_dT_Nm_per_N, dtype=np.float64)
        if cap.shape != shape or d_t.shape != shape:
            raise CableInputError(f"group caps and derivatives need shape {shape}")
        if not (np.isfinite(cap).all() and np.isfinite(d_t).all()):
            raise CableInputError("group caps and derivatives must be finite")
        if np.any(cap < 0.0):
            raise CableInputError("group caps must be non-negative")
        d_p = None
        if p is None:
            if self.group_cap_dpe_Nm_per_Pa is not None:
                raise CableInputError(
                    "a pressure derivative needs an external-pressure axis"
                )
        else:
            if self.group_cap_dpe_Nm_per_Pa is None:
                raise CableInputError(
                    "an external-pressure axis needs the pressure derivative"
                )
            d_p = np.asarray(self.group_cap_dpe_Nm_per_Pa, dtype=np.float64)
            if d_p.shape != shape or not np.isfinite(d_p).all():
                raise CableInputError(
                    f"the pressure derivative must be finite with shape {shape}"
                )
        values = [self.reference_tension_N, self.reference_external_pressure_Pa]
        if not np.isfinite(values).all() or min(values) < 0.0:
            raise CableInputError("the reference state must be finite and >= 0")
        if not (math.isfinite(self.ei_slip_Nm2) and self.ei_slip_Nm2 >= 0.0):
            raise CableInputError("ei_slip_Nm2 must be finite and >= 0")
        object.__setattr__(self, "tension_N", t)
        object.__setattr__(self, "external_pressure_Pa", p)
        object.__setattr__(self, "group_names", tuple(self.group_names))
        object.__setattr__(self, "group_laws", tuple(self.group_laws))
        object.__setattr__(self, "reference_cap_Nm", reference)
        object.__setattr__(self, "group_cap_Nm", cap)
        object.__setattr__(self, "group_cap_dT_Nm_per_N", d_t)
        object.__setattr__(self, "group_cap_dpe_Nm_per_Pa", d_p)

    # Scales at the table states ------------------------------------------
    @property
    def group_scale(self) -> FloatArray:
        """``rho_g`` at every table state, shape (groups, tensions, pressures)."""
        return self.group_cap_Nm / self.reference_cap_Nm[:, None, None]

    @property
    def group_scale_dT_1_per_N(self) -> FloatArray:
        """``d rho_g / dT`` [1/N] at every table state."""
        return self.group_cap_dT_Nm_per_N / self.reference_cap_Nm[:, None, None]

    @property
    def group_scale_dpe_1_per_Pa(self) -> FloatArray | None:
        """``d rho_g / dp_e`` [1/Pa] at every table state (pressure axis only)."""
        if self.group_cap_dpe_Nm_per_Pa is None:
            return None
        return self.group_cap_dpe_Nm_per_Pa / self.reference_cap_Nm[:, None, None]

    @property
    def scale(self) -> FloatArray:
        """Single cap-weighted scale ``s``, shape (tensions, pressures)."""
        return _ratio(self.group_cap_Nm.sum(axis=0), self.reference_cap_Nm.sum())

    @property
    def scale_dT_1_per_N(self) -> FloatArray:
        """``ds/dT`` [1/N] at every table state."""
        return _ratio(
            self.group_cap_dT_Nm_per_N.sum(axis=0), self.reference_cap_Nm.sum()
        )

    @property
    def scale_dpe_1_per_Pa(self) -> FloatArray | None:
        """``ds/dp_e`` [1/Pa] at every table state (pressure axis only)."""
        if self.group_cap_dpe_Nm_per_Pa is None:
            return None
        return _ratio(
            self.group_cap_dpe_Nm_per_Pa.sum(axis=0), self.reference_cap_Nm.sum()
        )

    # Scales at arbitrary states ------------------------------------------
    def group_scales(
        self,
        tension_N: ArrayLike,
        external_pressure_Pa: ArrayLike | None = None,
        *,
        per_group: bool = True,
    ) -> FloatArray:
        """Group scales at states, shape (groups, samples).

        Bilinear interpolation of the group caps in the table (exact for the
        bounded contact chain, which is affine in tension and pressure). With
        ``per_group=False`` every group takes the single scale ``s``. Without
        a pressure axis ``external_pressure_Pa`` must be ``None`` (the
        reference external pressure).
        """
        t = np.atleast_1d(np.asarray(tension_N, dtype=np.float64))
        if t.ndim != 1:
            raise CableInputError("tension must be a scalar or a 1-D history")
        i0, i1, wt = _bracket(self.tension_N, t, "tension")
        cap = self.group_cap_Nm
        if self.external_pressure_Pa is None:
            if external_pressure_Pa is not None:
                raise CableInputError(
                    "the table has no external-pressure axis; it holds the "
                    "reference external pressure"
                )
            values = (1.0 - wt) * cap[:, i0, 0] + wt * cap[:, i1, 0]
        else:
            if external_pressure_Pa is None:
                p = np.full(t.shape, self.reference_external_pressure_Pa)
            else:
                p = np.broadcast_to(
                    np.asarray(external_pressure_Pa, dtype=np.float64), t.shape
                )
            j0, j1, wp = _bracket(self.external_pressure_Pa, p, "external pressure")
            values = (1.0 - wt) * (1.0 - wp) * cap[:, i0, j0] + wt * (1.0 - wp) * cap[
                :, i1, j0
            ]
            values = (
                values + (1.0 - wt) * wp * cap[:, i0, j1] + wt * wp * cap[:, i1, j1]
            )
        if per_group:
            return np.asarray(values / self.reference_cap_Nm[:, None])
        single = values.sum(axis=0) / self.reference_cap_Nm.sum()
        return np.asarray(np.broadcast_to(single, values.shape).copy())

    def _elements(self) -> tuple[FloatArray, FloatArray, NDArray[np.intp]]:
        k = np.concatenate([law.stiffness_Nm2 for law in self.group_laws])
        y = np.concatenate([law.yield_curvature_1_m for law in self.group_laws])
        group = np.concatenate(
            [
                np.full(law.stiffness_Nm2.size, g, dtype=np.intp)
                for g, law in enumerate(self.group_laws)
            ]
        )
        return k, y, group

    def law_at(
        self,
        tension_N: float,
        external_pressure_Pa: float | None = None,
        *,
        per_group: bool = True,
    ) -> IwanBendingLaw:
        """The scaled law at one state (elements of equal yield merged)."""
        rho = self.group_scales(
            [tension_N],
            None if external_pressure_Pa is None else [external_pressure_Pa],
            per_group=per_group,
        )[:, 0]
        return self._merged(rho)

    def reference_law(self) -> IwanBendingLaw:
        """The law at the reference state, where every group scale is 1.

        Built from the group laws alone, so it exists when the reference
        state lies outside the table's axes (the default reference is the
        first exported tension with contact).
        """
        return self._merged(np.ones(len(self.group_laws)))

    def _merged(self, rho: FloatArray) -> IwanBendingLaw:
        """The group elements with yield curvatures times ``rho[g]``, merged."""
        k, y, group = self._elements()
        y = y * rho[group]
        keep = y > 0.0
        yields, index = np.unique(y[keep], return_inverse=True)
        stiffness = np.zeros(yields.size)
        np.add.at(stiffness, index, k[keep])
        return IwanBendingLaw(self.ei_slip_Nm2, stiffness, yields)

    def _history(
        self,
        kappa_history: ArrayLike,
        tension_history_N: ArrayLike,
        external_pressure_history_Pa: ArrayLike | None,
        per_group: bool,
    ) -> tuple[FloatArray, float]:
        kappa = np.asarray(kappa_history, dtype=np.float64)
        if kappa.ndim != 1 or not np.isfinite(kappa).all():
            raise CableInputError("curvature history must be a finite 1-D array")
        tension = np.asarray(tension_history_N, dtype=np.float64)
        if tension.shape != kappa.shape:
            raise CableInputError("tension history must match the curvature history")
        if external_pressure_history_Pa is not None:
            pressure = np.asarray(external_pressure_history_Pa, dtype=np.float64)
            if pressure.shape != kappa.shape:
                raise CableInputError(
                    "external-pressure history must match the curvature history"
                )
        else:
            pressure = None
        rho = self.group_scales(tension, pressure, per_group=per_group)
        k, y, group = self._elements()
        bound = y[:, None] * rho[group]  # (elements, samples)
        e = np.zeros_like(k)
        out = np.empty_like(kappa)
        work = 0.0
        previous = 0.0
        for n, value in enumerate(kappa):
            trial = e + (value - previous)
            clipped = np.clip(trial, -bound[:, n], bound[:, n])
            work += float(np.sum(k * np.abs(clipped) * np.abs(trial - clipped)))
            e = clipped
            previous = value
            out[n] = self.ei_slip_Nm2 * value + float(e @ k)
        return out, work

    def moment(
        self,
        kappa_history: ArrayLike,
        tension_history_N: ArrayLike,
        external_pressure_history_Pa: ArrayLike | None = None,
        *,
        per_group: bool = True,
    ) -> FloatArray:
        """Moment history [N m] under curvature and state histories.

        From the virgin state; every element follows the return map with the
        cap of the current sample (see the class notes). ``per_group=False``
        applies the single scale ``s`` to all groups.
        """
        return self._history(
            kappa_history, tension_history_N, external_pressure_history_Pa, per_group
        )[0]

    def dissipated_energy(
        self,
        kappa_history: ArrayLike,
        tension_history_N: ArrayLike,
        external_pressure_history_Pa: ArrayLike | None = None,
        *,
        per_group: bool = True,
    ) -> float:
        """Slider work over the histories [J/m].

        Every slip increment counts at the element's new force, as in
        :meth:`IwanBendingLaw.dissipated_energy`. A cap that falls below an
        element's force at fixed curvature adds ``k |e_1| |e_0 - e_1|`` (the
        backward-Euler estimate the section law reports).
        """
        return self._history(
            kappa_history, tension_history_N, external_pressure_history_Pa, per_group
        )[1]

    def record(self) -> dict[str, Any]:
        """The ``slip_scale`` object of the JSON law (format version 2)."""
        rho = self.group_scale
        rho_t = self.group_scale_dT_1_per_N
        rho_p = self.group_scale_dpe_1_per_Pa
        reference = self.reference_law()
        groups = []
        for g, (name, law) in enumerate(
            zip(self.group_names, self.group_laws, strict=True)
        ):
            item: dict[str, Any] = {
                "name": name,
                "kind": name if name in (CORE_GROUP, FILLER_GROUP) else "armour_layer",
                "reference_cap_Nm": float(self.reference_cap_Nm[g]),
                "stiffness_Nm2": law.stiffness_Nm2.tolist(),
                "yield_curvature_1_m": law.yield_curvature_1_m.tolist(),
                "scale": rho[g].tolist(),
                "dscale_dtension_1_per_N": rho_t[g].tolist(),
            }
            if rho_p is not None:
                item["dscale_dexternal_pressure_1_per_Pa"] = rho_p[g].tolist()
            groups.append(item)
        out: dict[str, Any] = {
            "rule": "yield_curvature_scaling",
            "variables": ["tension_N"]
            + ([] if self.external_pressure_Pa is None else ["external_pressure_Pa"]),
            "reference_tension_N": self.reference_tension_N,
            "reference_external_pressure_Pa": self.reference_external_pressure_Pa,
            "tension_N": self.tension_N.tolist(),
            "external_pressure_Pa": None
            if self.external_pressure_Pa is None
            else self.external_pressure_Pa.tolist(),
            "ei_slip_Nm2": self.ei_slip_Nm2,
            "reference_elements": {
                "stiffness_Nm2": reference.stiffness_Nm2.tolist(),
                "yield_curvature_1_m": reference.yield_curvature_1_m.tolist(),
            },
            "scale": self.scale.tolist(),
            "dscale_dtension_1_per_N": self.scale_dT_1_per_N.tolist(),
            "groups": groups,
        }
        if rho_p is not None:
            dp = self.scale_dpe_1_per_Pa
            assert dp is not None
            out["dscale_dexternal_pressure_1_per_Pa"] = dp.tolist()
        return out

    def csv_text(self) -> str:
        """One row per table state: the scales and their derivatives."""
        pressures = (
            np.array([self.reference_external_pressure_Pa])
            if self.external_pressure_Pa is None
            else self.external_pressure_Pa
        )
        header = ["tension_N", "external_pressure_Pa", "s", "ds_dT_1_per_N"]
        if self.external_pressure_Pa is not None:
            header.append("ds_dpe_1_per_Pa")
        for name in self.group_names:
            header += [f"s_{name}", f"ds_dT_{name}_1_per_N"]
            if self.external_pressure_Pa is not None:
                header.append(f"ds_dpe_{name}_1_per_Pa")
        rho, rho_t = self.group_scale, self.group_scale_dT_1_per_N
        rho_p, s_p = self.group_scale_dpe_1_per_Pa, self.scale_dpe_1_per_Pa
        s, s_t = self.scale, self.scale_dT_1_per_N
        rows = [",".join(header)]
        for i, tension in enumerate(self.tension_N):
            for j, pressure in enumerate(pressures):
                row = [tension, pressure, s[i, j], s_t[i, j]]
                if s_p is not None:
                    row.append(s_p[i, j])
                for g in range(len(self.group_names)):
                    row += [rho[g, i, j], rho_t[g, i, j]]
                    if rho_p is not None:
                        row.append(rho_p[g, i, j])
                rows.append(",".join(f"{float(v):.9e}" for v in row))
        return "\n".join(rows) + "\n"

    def to_csv(self, path: str | Path) -> Path:
        """Write :meth:`csv_text`."""
        out = Path(path)
        out.write_text(self.csv_text(), encoding="utf-8")
        return out

    def group_law(self, index: int, kappa_1_m: ArrayLike) -> SlipGroupLaw:
        """Friction group ``index`` as a law of its own (:class:`SlipGroupLaw`).

        ``kappa_1_m`` is the curvature grid of the export (the grid the group
        laws were decomposed on). The first group carries the elastic
        full-slip part ``EI_slip kappa``.
        """
        if not 0 <= index < len(self.group_names):
            raise CableInputError(
                f"group index {index} outside 0-{len(self.group_names) - 1}"
            )
        kappa = np.asarray(kappa_1_m, dtype=np.float64)
        law = self.group_laws[index]
        if kappa.ndim != 1 or kappa.size != law.stiffness_Nm2.size + 1:
            raise CableInputError(
                "the curvature grid must be the export grid of the group laws"
            )
        elastic = self.ei_slip_Nm2 if index == 0 else 0.0
        name = self.group_names[index]
        return SlipGroupLaw(
            name=name,
            kind=name if name in (CORE_GROUP, FILLER_GROUP) else "armour_layer",
            kappa_1_m=kappa,
            law=IwanBendingLaw(elastic, law.stiffness_Nm2, law.yield_curvature_1_m),
            table=self,
            index=index,
        )


@dataclass(frozen=True)
class SlipGroupLaw:
    r"""One friction group of a slip-scaled law, as a law of its own.

    The global model sums the group laws. Group ``g`` is the backbone
    ``B_g(kappa)`` of the group at the reference state (``law``, on the
    export's curvature grid) with its own scale ``s_g`` over the table states
    (the ``rho_g`` of :class:`SlipScaleTable`):

    .. math::

        M_g(\kappa; T) = s_g(T)\, B_g(\kappa / s_g(T)), \qquad
        M = \sum_g M_g ,

    and under a changing state every element follows the return map with the
    cap of the current sample (:meth:`moment`). The first group carries the
    elastic full-slip part ``EI_slip kappa`` (``law.ei_slip_Nm2``), which the
    scaling leaves unchanged; every other group has no elastic part, so its
    backbone ends on the plateau of its full-slip friction moment. A group of
    zero scale (no contact) carries its elastic part only.

    Attributes
    ----------
    name, kind : str
        Group name (the armour layer, :data:`CORE_GROUP` or
        :data:`FILLER_GROUP`) and kind (``"armour_layer"``, ``"laid_cores"``
        or ``"filler_profiles"``).
    kappa_1_m : numpy.ndarray
        Curvature grid of the backbone [1/m], from zero.
    law : IwanBendingLaw
        Elements of the group at the reference state.
    table : SlipScaleTable
        The table the group belongs to.
    index : int
        Position of the group in ``table.group_names``.
    """

    name: str
    kind: str
    kappa_1_m: FloatArray
    law: IwanBendingLaw
    table: SlipScaleTable
    index: int

    @property
    def moment_Nm(self) -> FloatArray:
        """Backbone ``B_g`` at the reference state on ``kappa_1_m`` [N m]."""
        return self.law.backbone(self.kappa_1_m)

    @property
    def tension_N(self) -> FloatArray:
        """Table tensions [N]."""
        return self.table.tension_N

    @property
    def scale(self) -> FloatArray:
        """``s_g`` at the table states, shape (tensions, pressures)."""
        return np.asarray(self.table.group_scale[self.index])

    @property
    def scale_dT_1_per_N(self) -> FloatArray:
        """``d s_g / dT`` [1/N] at the table states."""
        return np.asarray(self.table.group_scale_dT_1_per_N[self.index])

    def scale_at(
        self, tension_N: ArrayLike, external_pressure_Pa: ArrayLike | None = None
    ) -> FloatArray:
        """``s_g`` at states (linear interpolation in the table), shape (samples,)."""
        return np.asarray(
            self.table.group_scales(tension_N, external_pressure_Pa)[self.index]
        )

    def law_at(
        self, tension_N: float, external_pressure_Pa: float | None = None
    ) -> IwanBendingLaw:
        """The group law at one state: yield curvatures times ``s_g``.

        Elements of a zero scale are dropped (the group has no friction).
        """
        rho = float(
            self.scale_at(
                [tension_N],
                None if external_pressure_Pa is None else [external_pressure_Pa],
            )[0]
        )
        keep = (self.law.yield_curvature_1_m * rho) > 0.0
        return IwanBendingLaw(
            self.law.ei_slip_Nm2,
            self.law.stiffness_Nm2[keep],
            self.law.yield_curvature_1_m[keep] * rho,
        )

    def backbone_at(
        self,
        kappa_1_m: ArrayLike,
        tension_N: float,
        external_pressure_Pa: float | None = None,
    ) -> FloatArray:
        """Virgin backbone at one state, ``s_g B_g(kappa / s_g)`` [N m].

        The table form a global model applies: the reference backbone
        (piecewise linear through the grid points, continued by its tail
        slope) at the curvature divided by the scale, times the scale.
        """
        k = np.asarray(kappa_1_m, dtype=np.float64)
        rho = float(
            self.scale_at(
                [tension_N],
                None if external_pressure_Pa is None else [external_pressure_Pa],
            )[0]
        )
        if rho <= 0.0:
            return np.asarray(self.law.ei_slip_Nm2 * k)
        return np.asarray(rho * self.law.backbone(k / rho))

    def moment(
        self,
        kappa_history: ArrayLike,
        tension_history_N: ArrayLike,
        external_pressure_history_Pa: ArrayLike | None = None,
    ) -> FloatArray:
        """Moment history of the group [N m] under curvature and state histories.

        From the virgin state; every element follows the return map with the
        cap ``s_g kappa_y,i`` of the current sample, as in
        :meth:`SlipScaleTable.moment`.
        """
        kappa = np.asarray(kappa_history, dtype=np.float64)
        if kappa.ndim != 1 or not np.isfinite(kappa).all():
            raise CableInputError("curvature history must be a finite 1-D array")
        tension = np.asarray(tension_history_N, dtype=np.float64)
        if tension.shape != kappa.shape:
            raise CableInputError("tension history must match the curvature history")
        pressure = None
        if external_pressure_history_Pa is not None:
            pressure = np.asarray(external_pressure_history_Pa, dtype=np.float64)
            if pressure.shape != kappa.shape:
                raise CableInputError(
                    "external-pressure history must match the curvature history"
                )
        rho = self.scale_at(tension, pressure)
        k, y = self.law.stiffness_Nm2, self.law.yield_curvature_1_m
        e = np.zeros_like(k)
        out = np.empty_like(kappa)
        previous = 0.0
        for n, value in enumerate(kappa):
            bound = y * rho[n]
            e = np.clip(e + (value - previous), -bound, bound)
            previous = value
            out[n] = self.law.ei_slip_Nm2 * value + float(e @ k)
        return out

    def backbone_csv_text(self) -> str:
        """``kappa_1_m,M_Nm``: the reference backbone ``B_g`` on the grid.

        Two comment lines (``# ``) name the group and the slope that continues
        the backbone beyond its last point.
        """
        rows = [
            f"# friction group {self.name} ({self.kind}); scale s_{self.name}(T) in "
            f"the group's scale table; M_g(kappa, T) = s B(kappa / s)",
            f"# beyond the last point the backbone continues with slope "
            f"{self.law.ei_slip_Nm2:.9e} N m2",
            "kappa_1_m,M_Nm",
        ]
        rows += [
            f"{float(k):.9e},{float(m):.9e}"
            for k, m in zip(self.kappa_1_m, self.moment_Nm, strict=True)
        ]
        return "\n".join(rows) + "\n"

    def scale_csv_text(self) -> str:
        """One row per table state: tension, pressure, ``s_g`` and derivatives."""
        table = self.table
        pressures = (
            np.array([table.reference_external_pressure_Pa])
            if table.external_pressure_Pa is None
            else table.external_pressure_Pa
        )
        header = ["tension_N", "external_pressure_Pa", "s", "ds_dT_1_per_N"]
        rho_p = table.group_scale_dpe_1_per_Pa
        if rho_p is not None:
            header.append("ds_dpe_1_per_Pa")
        rows = [",".join(header)]
        for i, tension in enumerate(table.tension_N):
            for j, pressure in enumerate(pressures):
                row = [tension, pressure, self.scale[i, j], self.scale_dT_1_per_N[i, j]]
                if rho_p is not None:
                    row.append(rho_p[self.index, i, j])
                rows.append(",".join(f"{float(v):.9e}" for v in row))
        return "\n".join(rows) + "\n"


@dataclass(frozen=True)
class BendingLawExport:
    """Backbones and Iwan laws of one cable at a set of tensions.

    ``slip_scale`` holds the slip scaling variable of the law
    (:class:`SlipScaleTable`) when the export was asked for it.
    """

    tension_N: FloatArray
    kappa_1_m: FloatArray
    moment_Nm: FloatArray  # (tension, kappa)
    ei_slip_Nm2: FloatArray  # (tension,)
    laws: tuple[IwanBendingLaw, ...]
    provenance: Provenance
    settings: dict[str, Any] = field(default_factory=dict)
    slip_scale: SlipScaleTable | None = None

    def law_at(self, tension_N: float) -> IwanBendingLaw:
        """Law at a tension within the exported range (linear interpolation)."""
        t = self.tension_N
        if not (t[0] <= tension_N <= t[-1]):
            raise CableInputError(
                f"tension {tension_N:g} N outside the exported range "
                f"{t[0]:g}-{t[-1]:g} N"
            )
        if t.size == 1:
            return self.laws[0]
        j = int(np.clip(np.searchsorted(t, tension_N) - 1, 0, t.size - 2))
        w = (tension_N - t[j]) / (t[j + 1] - t[j])
        moment = (1 - w) * self.moment_Nm[j] + w * self.moment_Nm[j + 1]
        ei = (1 - w) * self.ei_slip_Nm2[j] + w * self.ei_slip_Nm2[j + 1]
        return IwanBendingLaw.from_backbone(self.kappa_1_m, moment, float(ei))

    def slip_group_laws(self) -> tuple[SlipGroupLaw, ...]:
        """One law per friction group, each with its own scale table.

        The global model sums them (:class:`SlipGroupLaw`); needs the export's
        slip scale (``slip_scale=True``).
        """
        if self.slip_scale is None:
            raise CableInputError(
                "the group laws need the slip scale: export with slip_scale=True"
            )
        table = self.slip_scale
        return tuple(
            table.group_law(g, self.kappa_1_m) for g in range(len(table.group_names))
        )

    def to_group_csv(self, directory: str | Path) -> list[Path]:
        """Write ``<group>_backbone.csv`` and ``<group>_slip_scale.csv`` per group.

        The backbone files hold the reference backbone of every group
        (:meth:`SlipGroupLaw.backbone_csv_text`), the scale files its own
        ``s_g(T)`` (:meth:`SlipGroupLaw.scale_csv_text`). ``<group>`` is the
        group name made a safe file name (:func:`_file_stem`: letters,
        digits, ``.``, ``-`` and ``_`` kept, anything else ``_``, a numbered
        suffix where two names would meet); the first comment line of the
        backbone file keeps the name itself. Every file lies in
        ``directory``. Returns the paths in group order, backbone before
        scale.
        """
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        groups = self.slip_group_laws()
        paths: list[Path] = []
        for group, stem in zip(
            groups, _file_stems([g.name for g in groups]), strict=True
        ):
            backbone = folder / f"{stem}_backbone.csv"
            backbone.write_text(group.backbone_csv_text(), encoding="utf-8")
            scale = folder / f"{stem}_slip_scale.csv"
            scale.write_text(group.scale_csv_text(), encoding="utf-8")
            paths += [backbone, scale]
        return paths

    def to_csv(self, path: str | Path) -> Path:
        """Backbones as columns ``kappa_1_m, M_<T>N_Nm, ...``."""
        out = Path(path)
        header = "kappa_1_m," + ",".join(f"M_{t:.6g}N_Nm" for t in self.tension_N)
        data = np.column_stack([self.kappa_1_m, self.moment_Nm.T])
        np.savetxt(out, data, delimiter=",", header=header, comments="", fmt="%.9e")
        return out

    def to_json(self, path: str | Path) -> Path:
        """Backbones, Iwan elements, full-slip stiffness and provenance."""
        out = Path(path)
        out.write_text(self.json_text(), encoding="utf-8")
        return out

    def json_text(self) -> str:
        """The text :meth:`to_json` writes.

        Format version 1 without a slip scale (unchanged); version 2 adds the
        ``slip_scale`` object and leaves every version-1 field as it was.
        """
        record: dict[str, Any] = {
            "format": "cablexsec.bending_law/1"
            if self.slip_scale is None
            else "cablexsec.bending_law/2",
            "hysteresis_rule": "masing_iwan",
            "planar_only": True,
            "tension_N": self.tension_N.tolist(),
            "kappa_1_m": self.kappa_1_m.tolist(),
            "moment_Nm": self.moment_Nm.tolist(),
            "ei_slip_Nm2": self.ei_slip_Nm2.tolist(),
            "elements": [
                {
                    "stiffness_Nm2": law.stiffness_Nm2.tolist(),
                    "yield_curvature_1_m": law.yield_curvature_1_m.tolist(),
                }
                for law in self.laws
            ],
            "settings": self.settings,
            "method": self.provenance.method,
            "limitations": list(self.provenance.limitations),
        }
        if self.slip_scale is not None:
            record["slip_scale"] = self.slip_scale.record()
        return json.dumps(record, indent=1)

    def backbone_table_kNm(self, tension_N: float) -> list[tuple[float, float]]:
        """Curvature [rad/m] and moment [kN m] pairs of the virgin backbone.

        The pairs are the input of a bending-stiffness table of a global line
        model (curvature against bending moment in kN and m); whether a global
        program's hysteretic option applies Masing's rules to it is target
        MX-5 and is not assumed here.
        """
        law = self.law_at(tension_N)
        moment = law.backbone(self.kappa_1_m)
        return [
            (float(k), float(m) / 1e3)
            for k, m in zip(self.kappa_1_m, moment, strict=True)
        ]


def export_bending_law(
    cable: CableData,
    *,
    tensions_N: Sequence[float],
    kappa_max_1_m: float,
    n_points: int = 200,
    spacing: str = "geometric",
    slip_scale: bool = False,
    slip_scale_reference_N: float | None = None,
    slip_scale_tensions_N: Sequence[float] | None = None,
    slip_scale_external_pressure_Pa: Sequence[float] | None = None,
    **mkappa_options: Any,
) -> BendingLawExport:
    """Hysteretic bending law of a cable for global analysis.

    Parameters
    ----------
    cable : CableData
        A fully specified cable.
    tensions_N : sequence of float
        Ascending tensions at which the backbone is computed [N].
    kappa_max_1_m : float
        Largest curvature of the backbone [1/m]; the full-slip slope continues
        it beyond.
    n_points : int, optional
        Points of the base curvature grid including zero (default 200). The
        export adds the slip onset and full-slip curvature of every friction
        group (armour layer, laid cores) at every exported tension and at the
        slip-scale reference, so ``kappa_1_m`` holds more points.
    spacing : {"geometric", "linear"}, optional
        Point spacing. ``"geometric"`` (default) runs from ``1e-4
        kappa_max_1_m`` and is extended below at the same ratio until its
        first point lies under half the smallest first-slip curvature, so
        every stick-to-slip transition is resolved at the grid's density and
        the first chord of every backbone is the stick stiffness.
        ``"linear"`` gets one stick point below the smallest first-slip
        curvature and the onset and full-slip points; a transition narrower
        than its step is resolved by those points alone, which is warned.
    slip_scale : bool, optional
        Also export the slip scaling variable (:class:`SlipScaleTable`,
        ``BendingLawExport.slip_scale``): the law at a reference state split
        into friction groups and the scale of every group over a table of
        states. Default ``False`` (the export is unchanged).
    slip_scale_reference_N : float, optional
        Reference tension of the scaled law (default: the smallest of
        ``tensions_N`` at which every friction group has contact). The group
        laws use the export's curvature grid, and a group of scale ``rho``
        reproduces its law at the state up to ``rho kappa_max_1_m``; beyond,
        the full-slip slope continues as in every exported law. With caps
        that rise with tension the smallest tension gives ``rho >= 1``. The
        grid carries the slip curvatures of the reference, so the group laws
        start in stick at any reference.
    slip_scale_tensions_N : sequence of float, optional
        Ascending tensions of the table [N]. The default is ``tensions_N``
        with zero tension in front (when the lowest exported tension is
        positive): the table runs to zero with the caps of the contact model
        at zero tension, which are zero without clamping (every contact load
        then comes from the tension, and ``s -> 0``) and the clamping caps
        otherwise, so a global model need not hold the lowest scale below
        the lowest exported tension.
    slip_scale_external_pressure_Pa : sequence of float, optional
        Ascending external pressures of the table [Pa]. ``None`` (default)
        runs the table over tension only, at the cable's initial external
        pressure, with the contact states of :func:`compute_mkappa`. With a
        pressure axis every state, the reference included, is a solved
        section state (``solve_section_state`` for the bounded and qp contact
        models, ``solve_cable_radial_state`` for the solved model).
    **mkappa_options
        Passed to :func:`cablexsec.api.compute_mkappa` (``mu``,
        ``contact_model``, ``friction_model``, ``progressive``,
        ``core_friction``); planar bending only. A non-neutral
        ``friction_law`` (static/dynamic or rate dependent) is refused: the
        exported parallel elastic-plastic law with Masing's rules is rate
        independent and cannot represent it. A law with only a pressure
        factor is accepted without ``slip_scale``: at the fixed contact state
        of each exported tension it is a fixed coefficient per wire, and the
        law is exported as for any coefficient. With ``slip_scale`` it is
        refused, because ``f(P)`` breaks the homogeneity
        ``M(kappa; rho) = rho M(kappa/rho; 1)`` on which the scaling rests
        (the coefficient moves inside the capstan exponent with ``P``).

    Notes
    -----
    Torque is not a table variable: the section states that carry a torque
    (the coupled qp and solved models with a free twist) differ at zero torque
    from the fixed-twist states of the exported backbones, and a torque moves
    the ``P1+P2`` of contra-wound layers in opposite senses. The internal-ring
    slip of resolved strands (``strand_layers``) has no cap scaling and is
    refused with ``slip_scale``.
    """
    from cablexsec.api import compute_mkappa  # noqa: PLC0415

    t = np.asarray(tensions_N, dtype=np.float64)
    if t.ndim != 1 or t.size < 1 or not np.isfinite(t).all() or np.any(t < 0):
        raise CableInputError("tensions_N must be finite, non-negative values")
    if np.any(np.diff(t) <= 0):
        raise CableInputError("tensions_N must rise strictly")
    if not (math.isfinite(kappa_max_1_m) and kappa_max_1_m > 0):
        raise CableInputError("kappa_max_1_m must be finite and positive")
    if (
        isinstance(n_points, bool)
        or not isinstance(n_points, numbers.Integral)
        or n_points < 3
    ):
        raise CableInputError(
            f"n_points must be an integer of at least 3: {n_points!r}"
        )
    for name in (
        "tension_N",
        "phi_bend",
        "section_state",
        "radial_state",
        "contact_pressure_Pa",
        "contact_tension_history_N",
    ):
        if name in mkappa_options:
            raise CableInputError(f"{name} is set by the export, not an option")
    friction_law = mkappa_options.get("friction_law")
    if friction_law is not None and (
        not isinstance(friction_law, FrictionLaw)
        or not friction_law.coefficient_neutral
    ):
        # A static/dynamic or rate-dependent coefficient has no Masing/Iwan
        # (rate-independent, parallel elastic-plastic) representation.
        raise CableInputError(
            "a non-neutral friction law cannot be exported as a Masing/Iwan "
            "bending law (it is state or rate dependent); export the "
            "rate-independent law"
        )
    if slip_scale and friction_law is not None and friction_law.pressure_dependent:
        # The scaling rule M(kappa; rho) = rho M(kappa/rho; 1) needs bounds
        # proportional to P1+P2 at a fixed coefficient; f(P) moves the
        # coefficient inside the capstan exponent with the pressure.
        raise CableInputError(
            "a pressure-dependent friction coefficient is not exactly "
            "homogeneous in the contact pressure, so the slip scaling variable "
            "cannot represent it; export without slip_scale (one law per "
            "tension at its fixed contact state)"
        )
    if "time_s" in mkappa_options:
        raise CableInputError(
            "time_s is not an export option: the exported backbone is rate independent"
        )
    if not slip_scale and (
        slip_scale_reference_N is not None
        or slip_scale_tensions_N is not None
        or slip_scale_external_pressure_Pa is not None
    ):
        raise CableInputError("the slip_scale_* options need slip_scale=True")
    if spacing not in ("geometric", "linear"):
        raise CableInputError("spacing must be 'geometric' or 'linear'")
    # compute_mkappa checks the options with its own messages before the slip
    # curvatures are read from the same prepared law.
    compute_mkappa(cable, np.zeros(2), tension_N=float(t[0]), **mkappa_options)
    contact_model, common = _law_options(mkappa_options)
    transitions: list[_Transition] = []
    for tension in t:
        state = _state_law(cable, float(tension), None, contact_model, common)
        transitions += _slip_transitions(state, float(tension), friction_law)
    reference_N: float | None = None
    if slip_scale:
        pressures = (
            None
            if slip_scale_external_pressure_Pa is None
            else _ascending(
                slip_scale_external_pressure_Pa, "slip_scale_external_pressure_Pa"
            )
        )
        reference_pressure = (
            None
            if pressures is None
            else float(cable.initial_state.external_pressure_Pa)
        )
        reference_N = _reference_tension(
            cable,
            slip_scale_reference_N,
            t,
            reference_pressure,
            contact_model,
            common,
        )
        state = _state_law(
            cable, reference_N, reference_pressure, contact_model, common
        )
        transitions += _slip_transitions(state, reference_N, None)
    kappa = _export_grid(kappa_max_1_m, n_points, spacing, transitions)
    moments = np.empty((t.size, kappa.size))
    ei_slip = np.empty(t.size)
    laws = []
    for i, tension in enumerate(t):
        result = compute_mkappa(
            cable, kappa, tension_N=float(tension), **mkappa_options
        )
        moments[i] = np.asarray(result.moment_Nm, dtype=np.float64)
        moments[i, 0] = 0.0
        ei_slip[i] = float(result.EI_min_Nm2)
        laws.append(IwanBendingLaw.from_backbone(kappa, moments[i], ei_slip[i]))
        stick = float(result.EI_max_Nm2)
        if abs(laws[-1].ei_stick_Nm2 - stick) > _STICK_TOLERANCE * stick:
            warnings.warn(
                f"the exported law at {tension:g} N starts at "
                f"{laws[-1].ei_stick_Nm2:.6g} N m2, not at the stick stiffness "
                f"{stick:.6g} N m2 of the section: its first grid curvature "
                f"{kappa[1]:.3g} 1/m is already partly slipped",
                UserWarning,
                stacklevel=2,
            )
    table = (
        _slip_scale_table(
            cable,
            kappa,
            reference_N=reference_N,
            candidates_N=t,
            tensions_N=(
                _default_table_tensions(t)
                if slip_scale_tensions_N is None
                else slip_scale_tensions_N
            ),
            external_pressure_Pa=slip_scale_external_pressure_Pa,
            options=mkappa_options,
        )
        if slip_scale
        else None
    )
    cores = mkappa_options.get("core_friction") not in (None, False)
    scale_method = ""
    scale_limitations: tuple[str, ...] = ()
    if table is not None:
        scale_method = (
            "; slip scaling variable: the reference law split into one friction "
            "group per armour layer"
            + (" and the laid cores" if cores else "")
            + ", yield curvatures scaled by each group's full-slip friction moment "
            "relative to the reference at fixed element stiffnesses"
        )
        scale_limitations = (
            "slip scale: exact per group for the direction-tied armour law (caps "
            "proportional to P1+P2 at fixed stick stiffness), for the laid-core "
            "group while its load-weighted friction coefficient is unchanged; one "
            "scale for all groups is exact only when the groups scale alike; "
            "torque is not a table variable",
        )
    provenance = Provenance(
        method=(
            "virgin moment-curvature backbone of compute_mkappa at each tension"
            + (
                ", with the core lay stage (laid cores stick, then slip with the "
                "Papailiou cap of their contact loads)"
                if cores
                else ""
            )
            + ", decomposed into parallel elastic, perfectly plastic elements "
            "(Iwan model; Masing's rules, exact for symmetric wire slip intervals)"
            + scale_method
        ),
        citations=("[Iwan1966]",),
        limitations=(
            "planar bending; the per-wire direction-tied biaxial response is not "
            "an isotropic hysteretic law",
            "tension and annulus state fixed within one law; between exported "
            "tensions the backbone is interpolated linearly",
            "beyond the largest exported curvature the full-slip slope continues",
            *scale_limitations,
        ),
    )
    settings: dict[str, Any] = {
        "kappa_max_1_m": float(kappa_max_1_m),
        "n_points": int(n_points),
        "spacing": spacing,
        **{k: repr(v) for k, v in mkappa_options.items()},
    }
    if table is not None:
        settings["slip_scale_reference_N"] = table.reference_tension_N
    return BendingLawExport(
        tension_N=t,
        kappa_1_m=kappa,
        moment_Nm=moments,
        ei_slip_Nm2=ei_slip,
        laws=tuple(laws),
        provenance=provenance,
        settings=settings,
        slip_scale=table,
    )


def _friction_cap(spec: HelicalLayerSpec) -> float:
    """Planar full-slip friction moment of one wire population [N m].

    Every wire at the bound on the side of its drive under a large positive
    curvature about ``phi_bend = 0``: the rising (tension) bound where
    ``cos(phi) > 0``, the falling bound where ``cos(phi) < 0``, with the moment
    arm of :func:`~cablexsec.mechanics.papailiou.wire_slip_history`.
    """
    upper, lower = wire_slip_stress_bounds(spec, 0.0)
    c = np.cos(spec.phi_wires)
    bound = np.where(c > 0.0, upper, lower)
    return float(
        spec.A_wire_m2
        * math.cos(spec.lay_angle_rad)
        * spec.r_helix_m
        * np.sum(bound * np.abs(c))
    )


def _file_stem(name: str) -> str:
    """A group name as one safe file-name component.

    Letters, digits, ``.``, ``-`` and ``_`` are kept and every other
    character becomes ``_``; a stem of dots only (``.``, ``..``) or an empty
    one becomes ``group``, so the component never leaves its directory.
    """
    stem = "".join(
        c if (c.isascii() and c.isalnum()) or c in "._-" else "_" for c in name
    )
    if not stem.strip("."):
        return "group"
    return stem


def _file_stems(names: Sequence[str]) -> list[str]:
    """Distinct safe stems (case-insensitive) of the group names, in order."""
    stems: list[str] = []
    taken: set[str] = set()
    for name in names:
        base = stem = _file_stem(name)
        count = 1
        while stem.lower() in taken:
            count += 1
            stem = f"{base}_{count}"
        taken.add(stem.lower())
        stems.append(stem)
    return stems


def _default_table_tensions(tensions: FloatArray) -> FloatArray:
    """The exported tensions with zero tension in front.

    The scale table then runs to zero tension with the caps of the contact
    model there (zero without clamping, where every contact load comes from
    the tension), instead of ending at the lowest exported tension.
    """
    if tensions[0] > 0.0:
        return np.concatenate([[0.0], tensions])
    return tensions


def _ascending(values: ArrayLike, label: str) -> FloatArray:
    grid = np.asarray(values, dtype=np.float64)
    if (
        grid.ndim != 1
        or grid.size < 1
        or not np.isfinite(grid).all()
        or np.any(grid < 0.0)
        or np.any(np.diff(grid) <= 0.0)
    ):
        raise CableInputError(f"{label} must be finite, non-negative and rise strictly")
    return grid


@dataclass(frozen=True)
class _Transition:
    """Stick-to-slip transition of one friction group at one state [1/m]."""

    tension_N: float
    first_slip: float  # lower bound of the curvature of the first slip
    onset: float
    full_slip: float


def _law_options(options: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Contact model and the law settings of :func:`_prepare_mkappa_law`."""
    from cablexsec.mechanics.core_friction import (  # noqa: PLC0415
        resolve_core_friction,
    )
    from cablexsec.mechanics.progressive_friction import (  # noqa: PLC0415
        ProgressiveFriction,
    )

    friction_model = options.get("friction_model", "direction_tied")
    common: dict[str, Any] = {
        "mu": options.get("mu"),
        "phi_bend": 0.0,
        "core_spec": resolve_core_friction(options.get("core_friction")),
        "progressive": (options.get("progressive") or ProgressiveFriction())
        if friction_model == "progressive"
        else None,
        # The slip transitions and the slip-scale group laws take the stick
        # support of the exported backbones (compute_mkappa).
        "stick_support": options.get("stick_support", "yun_stack"),
    }
    return options.get("contact_model") or "bounded", common


def _state_law(
    cable: CableData,
    tension: float,
    pressure: float | None,
    contact_model: str,
    common: dict[str, Any],
) -> _MkappaLaw:
    """The prepared law at a tension and, optionally, a solved pressure state.

    Without a pressure, the fixed-tension contact state of
    :func:`compute_mkappa`; with one, a solved section state
    (``solve_section_state``, or ``solve_cable_radial_state`` for the solved
    model).
    """
    from cablexsec.api import _prepare_mkappa_law  # noqa: PLC0415
    from cablexsec.mechanics.radial_cable import (  # noqa: PLC0415
        solve_cable_radial_state,
    )
    from cablexsec.mechanics.section_state import (  # noqa: PLC0415
        solve_section_state,
    )

    if pressure is None:
        return _prepare_mkappa_law(
            cable,
            tension,
            contact_model=contact_model,
            section_state=None,
            radial_state=None,
            **common,
        )
    if contact_model == "solved":
        radial = solve_cable_radial_state(cable, tension, external_pressure_Pa=pressure)
        return _prepare_mkappa_law(
            cable,
            radial.tension_N,
            contact_model="solved",
            section_state=None,
            radial_state=radial,
            **common,
        )
    state = solve_section_state(
        cable, tension_N=tension, p_ext_Pa=pressure, contact_model=contact_model
    )
    return _prepare_mkappa_law(
        cable,
        state.tension_N,
        contact_model=contact_model,
        section_state=state,
        radial_state=None,
        **common,
    )


def _group_caps(law: _MkappaLaw) -> FloatArray:
    """Planar full-slip friction moment of every friction group [N m]."""
    values = [_friction_cap(spec) for spec in law.layers]
    if law.core_model is not None:
        values += [_friction_cap(spec) for spec in law.core_model.populations]
    return np.asarray(values, dtype=np.float64)


def _reference_tension(
    cable: CableData,
    reference_N: float | None,
    candidates_N: FloatArray,
    pressure: float | None,
    contact_model: str,
    common: dict[str, Any],
) -> float:
    """The given reference, or the first candidate at which every group has
    contact (the last candidate when none has)."""
    if reference_N is not None:
        value = float(reference_N)
        if not (math.isfinite(value) and value >= 0.0):
            raise CableInputError("slip_scale_reference_N must be finite and >= 0")
        return value
    # The smallest exported tension at which every group has contact: with
    # caps rising with tension the scales are then >= 1.
    return next(
        (
            float(value)
            for value in candidates_N
            if np.all(
                _group_caps(
                    _state_law(cable, float(value), pressure, contact_model, common)
                )
                > 0.0
            )
        ),
        float(candidates_N[-1]),
    )


def _slip_transitions(
    law: _MkappaLaw, tension: float, friction_law: FrictionLaw | None
) -> list[_Transition]:
    """Slip onset and full-slip curvature of every friction group with friction.

    ``first_slip`` bounds the first slip of a wire from below: the neutral-axis
    onset of :func:`~cablexsec.mechanics.papailiou.slip_onset_curvature` times
    the compression-side factor ``m(lam)`` and, for the uniform progressive
    density, the smallest sub-element cap fraction ``1 - h (1 - 1/N)``. A
    pressure-dependent coefficient enters at the state's ``P1+P2``.
    """
    spread = 1.0
    if law.progressive is not None and law.progressive.density == "uniform":
        n = law.progressive.n_elements
        spread = 1.0 - law.progressive.spread * (1.0 - 1.0 / n)
    specs = [(spec, spread) for spec in law.layers]
    if law.core_model is not None:
        specs += [(spec, 1.0) for spec in law.core_model.populations]
    out = []
    for spec, factor in specs:
        if friction_law is not None and friction_law.pressure_dependent:
            scale = float(friction_law.pressure_factor_at(spec.contact_pressure_Pa))
            spec = dataclasses.replace(spec, mu=spec.mu * scale)
        onset = slip_onset_curvature(spec)
        if not (math.isfinite(onset) and onset > 0.0):
            continue  # no friction: the group is in full slip from zero
        lam = spec.mu * abs(math.sin(spec.lay_angle_rad))
        out.append(
            _Transition(
                tension,
                onset * _compression_onset_factor(lam) * factor,
                onset,
                full_slip_curvature(spec),
            )
        )
    return out


def _export_grid(
    kappa_max: float,
    n_points: int,
    spacing: str,
    transitions: Sequence[_Transition],
) -> FloatArray:
    """Zero, the base grid, its extension into stick and the slip curvatures."""
    if spacing == "geometric":
        base = np.geomspace(kappa_max * _GRID_FLOOR, kappa_max, n_points - 1)
    else:
        base = np.linspace(0.0, kappa_max, n_points)[1:]
    first = min((tr.first_slip for tr in transitions), default=math.inf)
    target = _STICK_MARGIN * first
    extra: list[float] = []
    if target < base[0]:
        if spacing == "geometric":
            ratio = base[1] / base[0]
            floor = max(target, base[0] * 10.0**-_MAX_EXTENSION_DECADES)
            count = math.ceil(math.log(base[0] / floor) / math.log(ratio))
            extra += (base[0] * ratio ** -np.arange(count, 0, -1)).tolist()
            if target < floor:
                warnings.warn(
                    f"the first slip at {first:.3g} 1/m lies more than "
                    f"{_MAX_EXTENSION_DECADES:g} decades below the grid floor "
                    f"{base[0]:.3g} 1/m; the exported law starts at the stick "
                    f"stiffness only from {floor:.3g} 1/m (raise n_points, lower "
                    "kappa_max_1_m or drop the near-zero tension)",
                    UserWarning,
                    stacklevel=3,
                )
        else:
            extra.append(target)
    if spacing == "linear":
        narrow = [
            tr
            for tr in transitions
            if tr.onset < kappa_max and tr.full_slip - tr.onset < base[0]
        ]
        if narrow:
            worst = min(narrow, key=lambda tr: tr.full_slip - tr.onset)
            warnings.warn(
                f"the linear step {base[0]:.3g} 1/m is wider than the "
                f"stick-to-slip transition {worst.onset:.3g}-"
                f"{worst.full_slip:.3g} 1/m at {worst.tension_N:g} N, which "
                "only its onset and full-slip points resolve; use the "
                "geometric spacing",
                UserWarning,
                stacklevel=3,
            )
    for tr in transitions:
        extra += [tr.onset, tr.full_slip]
    nodes = np.array(
        [k for k in extra if math.isfinite(k) and 0.0 < k < kappa_max],
        dtype=np.float64,
    )
    grid = np.unique(np.concatenate([base, nodes]))
    keep = np.concatenate([[True], np.diff(grid) > _MERGE_GAP * grid[1:]])
    return np.concatenate([[0.0], grid[keep]])


def _slip_scale_table(
    cable: CableData,
    kappa: FloatArray,
    *,
    reference_N: float | None,
    candidates_N: FloatArray,
    tensions_N: ArrayLike,
    external_pressure_Pa: ArrayLike | None,
    options: dict[str, Any],
) -> SlipScaleTable:
    """Reference group laws and group caps over the table states."""
    from cablexsec.cable_adapters import structural_armour_layers  # noqa: PLC0415
    from cablexsec.mechanics.core_friction import (  # noqa: PLC0415
        core_friction_history,
    )
    from cablexsec.mechanics.papailiou import (  # noqa: PLC0415
        compute_mkappa_wire_slip_front,
    )

    tensions = _ascending(tensions_N, "slip_scale_tensions_N")
    pressures = (
        None
        if external_pressure_Pa is None
        else _ascending(external_pressure_Pa, "slip_scale_external_pressure_Pa")
    )
    contact_model, common = _law_options(options)
    reference_pe = float(cable.initial_state.external_pressure_Pa)

    def prepare(tension: float, pressure: float | None) -> _MkappaLaw:
        return _state_law(cable, tension, pressure, contact_model, common)

    def caps(tension: float, pressure: float | None) -> FloatArray:
        return _group_caps(prepare(tension, pressure))

    reference_pressure = None if pressures is None else reference_pe
    reference_N = _reference_tension(
        cable, reference_N, candidates_N, reference_pressure, contact_model, common
    )
    reference = prepare(reference_N, reference_pressure)
    names = [layer.name for layer in structural_armour_layers(cable)]
    backbones = [
        compute_mkappa_wire_slip_front(
            [spec],
            kappa,
            phi_bend=0.0,
            EI_min_Nm2=0.0,
            progressive=reference.progressive,
        ).moment_Nm
        for spec in reference.layers
    ]
    core = reference.core_model
    if core is not None:
        if core.strand_rings:
            raise CableInputError(
                "the internal-ring slip of resolved strands has no cap scaling; "
                "export the slip scale with CoreFrictionSpec(strand_slip=False)"
            )
        names.append(CORE_GROUP)
        history = core_friction_history(
            core,
            kappa,
            np.zeros_like(kappa),
        )
        backbones.append(history.core_moment_Nm[:, 0])
        if history.filler_moment_Nm is not None:
            names.append(FILLER_GROUP)
            backbones.append(history.filler_moment_Nm[:, 0])
    reference_cap = caps(reference_N, reference_pressure)
    empty = [
        name for name, value in zip(names, reference_cap, strict=True) if value <= 0
    ]
    if empty:
        raise CableInputError(
            f"no friction at the reference tension {reference_N:g} N in "
            f"{', '.join(empty)}; choose a reference with contact"
        )
    group_laws = []
    for backbone in backbones:
        values = np.asarray(backbone, dtype=np.float64).copy()
        values[0] = 0.0
        group_laws.append(IwanBendingLaw.from_backbone(kappa, values, 0.0))
    columns: list[float | None] = (
        [None] if pressures is None else [float(p) for p in pressures]
    )
    shape = (len(names), tensions.size, len(columns))
    cap, d_t = np.empty(shape), np.empty(shape)
    d_p = None if pressures is None else np.empty(shape)
    h_t = _DIFFERENCE_STEP * max(float(tensions[-1]), reference_N, 1e3)
    h_p = (
        0.0 if pressures is None else _DIFFERENCE_STEP * max(float(pressures[-1]), 1e5)
    )
    for i, value in enumerate(tensions):
        tension = float(value)
        low_t, high_t = max(tension - h_t, 0.0), tension + h_t
        for j, pressure in enumerate(columns):
            cap[:, i, j] = caps(tension, pressure)
            d_t[:, i, j] = (caps(high_t, pressure) - caps(low_t, pressure)) / (
                high_t - low_t
            )
            if d_p is not None and pressure is not None:
                low_p, high_p = max(pressure - h_p, 0.0), pressure + h_p
                d_p[:, i, j] = (caps(tension, high_p) - caps(tension, low_p)) / (
                    high_p - low_p
                )
    return SlipScaleTable(
        reference_tension_N=float(reference_N),
        reference_external_pressure_Pa=reference_pe,
        tension_N=tensions,
        external_pressure_Pa=pressures,
        group_names=tuple(names),
        group_laws=tuple(group_laws),
        ei_slip_Nm2=float(reference.ei_min),
        reference_cap_Nm=reference_cap,
        group_cap_Nm=cap,
        group_cap_dT_Nm_per_N=d_t,
        group_cap_dpe_Nm_per_Pa=d_p,
    )
