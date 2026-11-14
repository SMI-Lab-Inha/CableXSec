# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Hysteretic bending law of a cable section for global analysis (gap MX).

In planar bending at fixed contact pressure the default armour law gives every
wire an elastic, perfectly plastic spring--slider with a fixed cap, in parallel
with the elastic full-slip stiffness. The section law is then an Iwan parallel
model, which obeys Masing's rules exactly ([Iwan1966]). A piecewise-linear
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

References
----------
[Iwan1966] W.D. Iwan, A distributed-element model for hysteresis and its
steady-state dynamic response, J. Appl. Mech. 33 (1966) 893-900.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cablexsec.exceptions import CableInputError
from cablexsec.provenance import Provenance

if TYPE_CHECKING:
    from cablexsec.cables.schema import CableData

__all__ = [
    "BendingLawExport",
    "IwanBendingLaw",
    "export_bending_law",
]

FloatArray = NDArray[np.float64]

#: Relative tolerance on a rising backbone tangent (numerical noise of the
#: section law); larger rises are rejected as a non-Masing backbone.
_TANGENT_RISE_TOLERANCE = 1e-6


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


@dataclass(frozen=True)
class BendingLawExport:
    """Backbones and Iwan laws of one cable at a set of tensions."""

    tension_N: FloatArray
    kappa_1_m: FloatArray
    moment_Nm: FloatArray  # (tension, kappa)
    ei_slip_Nm2: FloatArray  # (tension,)
    laws: tuple[IwanBendingLaw, ...]
    provenance: Provenance
    settings: dict[str, Any] = field(default_factory=dict)

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
        """The text :meth:`to_json` writes."""
        record = {
            "format": "cablexsec.bending_law/1",
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
        Backbone points including zero (default 200).
    spacing : {"geometric", "linear"}, optional
        Point spacing. ``"geometric"`` resolves the stick-to-slip transition at
        small curvature (default).
    **mkappa_options
        Passed to :func:`cablexsec.api.compute_mkappa` (``mu``,
        ``contact_model``, ``friction_model``, ``progressive``,
        ``core_friction``); planar bending only.
    """
    from cablexsec.api import compute_mkappa  # noqa: PLC0415

    t = np.asarray(tensions_N, dtype=np.float64)
    if t.ndim != 1 or t.size < 1 or not np.isfinite(t).all() or np.any(t < 0):
        raise CableInputError("tensions_N must be finite, non-negative values")
    if np.any(np.diff(t) <= 0):
        raise CableInputError("tensions_N must rise strictly")
    if not (math.isfinite(kappa_max_1_m) and kappa_max_1_m > 0):
        raise CableInputError("kappa_max_1_m must be finite and positive")
    if n_points < 3:
        raise CableInputError("n_points must be at least 3")
    for name in ("tension_N", "phi_bend", "section_state", "radial_state"):
        if name in mkappa_options:
            raise CableInputError(f"{name} is set by the export, not an option")
    if spacing == "geometric":
        kappa = np.concatenate(
            [[0.0], np.geomspace(kappa_max_1_m * 1e-4, kappa_max_1_m, n_points - 1)]
        )
    elif spacing == "linear":
        kappa = np.linspace(0.0, kappa_max_1_m, n_points)
    else:
        raise CableInputError("spacing must be 'geometric' or 'linear'")
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
    cores = mkappa_options.get("core_friction") not in (None, False)
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
            "(Iwan model, Masing's rules exact for the planar direction-tied law)"
        ),
        citations=("[Iwan1966]",),
        limitations=(
            "planar bending; the per-wire direction-tied biaxial response is not "
            "an isotropic hysteretic law",
            "tension and annulus state fixed within one law; between exported "
            "tensions the backbone is interpolated linearly",
            "beyond the largest exported curvature the full-slip slope continues",
        ),
    )
    return BendingLawExport(
        tension_N=t,
        kappa_1_m=kappa,
        moment_Nm=moments,
        ei_slip_Nm2=ei_slip,
        laws=tuple(laws),
        provenance=provenance,
        settings={
            "kappa_max_1_m": float(kappa_max_1_m),
            "n_points": int(n_points),
            "spacing": spacing,
            **{k: repr(v) for k, v in mkappa_options.items()},
        },
    )
