# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Capacity check of global tension-curvature time series.

A global analysis (CableDyn, RIFLEX or another program) gives the tension and the
curvature at a cable section. Each sample is placed on the cross-section
capacity envelope and given a design utilisation; the history of that
utilisation, its maximum and the time it occurs, the exceedance counts and
statistical extremes over seeds or sea states are reported. This is the
post-processing scheme described for Helica: utilisation time series from
global results plotted against the capacity curve, with the statistical
extreme utilisation reported ([HelicaFlier2018] p. 2; capacity curves at
utilisation levels, [Skeie2012] Sec. 3.1, Fig. 11). The mapping of a global
history into a tension-curvature failure envelope is also that of Cen et al.
(2026, Fig. 19).

Envelope (:func:`build_capacity_envelope`). On a tension grid:

* armour wire stress: the curvatures at which the largest wire usage
  ``max |sigma| / sigma_y`` (the common slip-front stress of
  :func:`~cablexsec.analysis.capacity.wire_yield_curvature`) reaches a set
  of levels. Level ``s`` is the design utilisation ``s / eta``, with ``eta``
  the design utilisation factor (default
  ``ARMOUR_UTILISATION_FACTORS["normal"]``, [DNV-RP-F401] Table 3-1); the
  level ``s = eta`` is the design capacity curve;
* armour lateral buckling in compression (gated by first slip) and radial
  buckling of the armour stack (:func:`~cablexsec.analysis.compression.\
compression_screen`, the same design factor);
* conductor fracture and, with a sourced allowable, polymer crushing
  (characteristic, curvature independent);
* the limits of every other component of the load path
  (:mod:`~cablexsec.analysis.component_limits`: the 0.2 % proof stress of
  the conductor on the strand basis of its bending stiffness, the allowable
  strain of polymers and of metallic screens and sheaths, whose first yield
  does not bound the capacity:
  :func:`~cablexsec.analysis.components.capacity_utilisation`),
  from :func:`~cablexsec.analysis.components.component_usage_curve` in the
  frictional core slip state, bent towards a core. Their design utilisation
  is the utilisation over the component factor ``eta_c`` of the same
  lifecycle phase, the net-section fracture of a conductor staying
  characteristic as the conductor-fracture mechanism ([MecLim2025] Table 5,
  p. 19: 0.80 normal operation, 1.00 installation and abnormal operation;
  :func:`~cablexsec.analysis.component_limits.component_factor_for`), and
  their level curves are those of that design utilisation. Layers outside
  the load path (loose screens) are not part of the envelope: their stress
  is the kinematic upper bound of a screen that does not slip; they are
  named in ``kinematic_layers``.

Utilisation of a sample ``(T, kappa)``: the largest of (i) the wire-stress
utilisation, interpolated linearly in curvature between the level curves
(from the usage at zero curvature) and linearly in tension between grid
points, (ii) the lateral-buckling design utilisation with slip active,
where the curvature exceeds first slip (the largest tabulated knock-down,
conservative), (iii) the curvature-independent mechanisms and (iv) the
component limits, interpolated as the wire stress from their own level
curves. Beyond the
highest level the wire utilisation is extrapolated with the slope of the last
two levels, and such samples are counted. An imported envelope
(:meth:`CapacityEnvelope.from_points`) uses the curvature ratio
``kappa / kappa_capacity(T)``.

Statistics. Exceedances are samples above a threshold, their up-crossings
and the time above it. The minimum bend radius check compares the curvature
with ``1/MBR`` (the cable's ``MBR_dynamic_m`` unless another is given). The
extreme utilisation is a Gumbel (type I) distribution fitted by the method of
moments to the maxima of seeds of one condition (:func:`gumbel_fit`,
[Gumbel1958]): scale ``beta = sqrt(6) s / pi``, location
``mu = mean - 0.5772 beta``; the most probable maximum is ``mu`` and the
``p`` fractile ``mu - beta ln(-ln p)``. Within one record, the maxima of
equal blocks give the same fit, shifted by ``beta ln(T / T_block)`` to the
record (or a target) duration.

References
----------
[HelicaFlier2018] p. 2; [Skeie2012] Sec. 3.1; [DNV-RP-F401] (September
2019, amended September 2021) Sec. 3.8.1, Table 3-1, p. 13; [MecLim2025]
Table 5, p. 19; [Gumbel1958]; Cen et al. (2026) Fig. 19.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cablexsec.cables.schema import CableData
from cablexsec.exceptions import CableInputError
from cablexsec.provenance import Provenance

__all__ = [
    "CapacityCheckResult",
    "CapacityEnvelope",
    "ExceedanceStatistics",
    "GumbelFit",
    "MBRCheck",
    "build_capacity_envelope",
    "capacity_extremes",
    "check_capacity",
    "check_capacity_series",
    "gumbel_fit",
]

FloatArray = NDArray[np.float64]
TensionKind = Literal["effective", "true", "material"]
OutOfRange = Literal["raise", "clamp"]
EULER_GAMMA = 0.5772156649015329
_MECHANISMS = (
    "wire_stress",
    "lateral_buckling",
    "tension_mechanisms",
    "component_limits",
)


def _finite_1d(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise CableInputError(f"{name} must be a finite one-dimensional array")
    return array


def _interp_rows(
    x: FloatArray, grid: FloatArray, rows: FloatArray, out_of_range: OutOfRange
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Linear interpolation of each row of ``rows`` over ``grid`` at ``x``.

    Infinite entries stay infinite. Returns the values (rows, samples) and
    the samples outside the grid.
    """
    outside = (x < grid[0]) | (x > grid[-1])
    if np.any(outside) and out_of_range == "raise":
        k = int(np.flatnonzero(outside)[0])
        raise CableInputError(
            f"tension {x[k]:.6g} N lies outside the envelope grid "
            f"[{grid[0]:.6g}, {grid[-1]:.6g}] N; extend the grid or pass "
            "out_of_range='clamp'"
        )
    xc = np.clip(x, grid[0], grid[-1])
    if grid.size == 1:
        return np.repeat(rows[:, :1], x.size, axis=1), outside
    index = np.clip(np.searchsorted(grid, xc, side="right") - 1, 0, grid.size - 2)
    w = (xc - grid[index]) / (grid[index + 1] - grid[index])
    a = rows[:, index]
    b = rows[:, index + 1]
    with np.errstate(invalid="ignore"):
        value = a + w * (b - a)
    value = np.where(np.isinf(a) | np.isinf(b), np.inf, value)
    return np.asarray(value, dtype=np.float64), outside


@dataclass(frozen=True)
class CapacityEnvelope:
    """Tension-curvature capacity envelope with utilisation levels.

    Attributes
    ----------
    tension_N : numpy.ndarray
        Strictly increasing tension grid [N] (negative: compression).
    kappa_capacity_1_m : numpy.ndarray
        Curvature at design utilisation one at each tension [1/m].
    tension_utilisation : numpy.ndarray
        Largest curvature-independent design utilisation (radial buckling,
        conductor fracture, polymer crushing).
    utilisation_factor : float
        Design utilisation factor ``eta``.
    tension_kind : {"effective", "true", "material"}
        The tension the grid refers to; checked against the history.
    levels : numpy.ndarray
        Design utilisation of the wire-stress level curves (ascending; empty
        for an imported envelope).
    kappa_levels_1_m : numpy.ndarray
        Curvature of each level curve, shape (level, tension) [1/m].
    zero_curvature_utilisation : numpy.ndarray
        Wire-stress design utilisation at zero curvature.
    lateral_utilisation : numpy.ndarray
        Lateral-buckling design utilisation with slip active.
    first_slip_curvature_1_m : numpy.ndarray
        Curvature above which the lateral mode is active (``inf`` in
        tension) [1/m].
    governing : tuple[str, ...]
        Governing mechanism of the capacity curve at each tension
        (``"component_limit"`` where a component other than the armour
        governs).
    component_utilisation_factor : float
        Design utilisation factor ``eta_c`` of the components other than the
        armour.
    component_levels_1_m : numpy.ndarray
        Curvature at which the design utilisation of the components reaches
        each level, shape (level, tension) [1/m] (empty: not assessed).
    component_zero_utilisation : numpy.ndarray
        Design utilisation of the components at zero curvature.
    component_governing : tuple[tuple[str, str], ...]
        The component and its limit that govern the component utilisation at
        each tension, at its design capacity: (layer, mode), empty strings
        where no component has a limit.
    kinematic_layers : tuple[str, ...]
        Layers outside the load path, left out of the envelope.
    component_kappa_1_m : numpy.ndarray
        Curvatures at which the design utilisation of the components is
        tabulated [1/m] (empty: the level curves are used).
    component_usage : numpy.ndarray
        Largest design utilisation of the components, shape (tension,
        curvature), over the bending directions: the values the utilisation
        of a sample is interpolated from.
    wire_beyond_slope : numpy.ndarray
        Slope of the wire design utilisation beyond its highest level, from
        the exact usage, where the last two level curves give none [m].
    mbr_m : float or None
        Minimum bend radius for the curvature check [m].
    mbr_source : str
        Source of ``mbr_m``.
    source : str
        How the envelope was obtained.
    provenance : Provenance or None
    """

    tension_N: FloatArray
    kappa_capacity_1_m: FloatArray
    tension_utilisation: FloatArray
    utilisation_factor: float
    tension_kind: TensionKind
    levels: FloatArray = field(default_factory=lambda: np.zeros(0))
    kappa_levels_1_m: FloatArray = field(default_factory=lambda: np.zeros((0, 0)))
    zero_curvature_utilisation: FloatArray = field(default_factory=lambda: np.zeros(0))
    lateral_utilisation: FloatArray = field(default_factory=lambda: np.zeros(0))
    first_slip_curvature_1_m: FloatArray = field(default_factory=lambda: np.zeros(0))
    governing: tuple[str, ...] = ()
    mbr_m: float | None = None
    mbr_source: str = ""
    source: str = ""
    provenance: Provenance | None = None
    component_utilisation_factor: float = 1.0
    component_levels_1_m: FloatArray = field(default_factory=lambda: np.zeros((0, 0)))
    component_zero_utilisation: FloatArray = field(default_factory=lambda: np.zeros(0))
    component_governing: tuple[tuple[str, str], ...] = ()
    kinematic_layers: tuple[str, ...] = ()
    component_kappa_1_m: FloatArray = field(default_factory=lambda: np.zeros(0))
    component_usage: FloatArray = field(default_factory=lambda: np.zeros((0, 0)))
    wire_beyond_slope: FloatArray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        grid = _finite_1d(self.tension_N, "tension_N")
        if grid.size > 1 and np.any(np.diff(grid) <= 0):
            raise CableInputError("the envelope tension grid must increase strictly")
        kappa = np.asarray(self.kappa_capacity_1_m, dtype=np.float64)
        if kappa.shape != grid.shape or np.any(np.isnan(kappa)) or np.any(kappa < 0):
            raise CableInputError("kappa_capacity_1_m must be nonnegative per tension")
        if self.tension_kind not in ("effective", "true", "material"):
            raise CableInputError("tension_kind must be effective, true or material")
        if not math.isfinite(self.utilisation_factor) or not (
            0.0 < self.utilisation_factor <= 1.0
        ):
            raise CableInputError("utilisation_factor must lie in (0, 1]")
        if self.mbr_m is not None and not (
            math.isfinite(self.mbr_m) and self.mbr_m > 0
        ):
            raise CableInputError("mbr_m must be finite and positive")
        for name in ("tension_N", "kappa_capacity_1_m", "tension_utilisation"):
            values = np.array(getattr(self, name), dtype=np.float64)
            values.flags.writeable = False
            object.__setattr__(self, name, values)

    @property
    def has_levels(self) -> bool:
        """``True`` for a built envelope with wire-stress level curves."""
        return self.levels.size > 0

    @property
    def has_components(self) -> bool:
        """``True`` when the envelope checks components other than the armour."""
        return self.component_levels_1_m.size > 0

    @classmethod
    def from_points(
        cls,
        tension_N: ArrayLike,
        kappa_capacity_1_m: ArrayLike,
        *,
        source: str,
        tension_kind: TensionKind = "effective",
        utilisation_factor: float = 1.0,
        mbr_m: float | None = None,
        mbr_source: str = "",
    ) -> CapacityEnvelope:
        """An imported design capacity curve (curvature-ratio utilisation).

        ``kappa_capacity_1_m`` is the allowable curvature at each tension,
        already including the design factor; ``source`` names the curve (a
        report, another program's output, a standard).
        """
        if not source.strip():
            raise CableInputError("an imported capacity curve needs its source")
        tension = _finite_1d(tension_N, "tension_N")
        kappa = np.asarray(kappa_capacity_1_m, dtype=np.float64)
        return cls(
            tension_N=tension,
            kappa_capacity_1_m=kappa,
            tension_utilisation=np.zeros_like(tension),
            utilisation_factor=utilisation_factor,
            tension_kind=tension_kind,
            governing=("imported",) * tension.size,
            mbr_m=mbr_m,
            mbr_source=mbr_source,
            source=source,
            provenance=Provenance(
                method="imported tension-curvature capacity curve; utilisation "
                "kappa / kappa_capacity(T), linear in tension",
                citations=(source,),
                limitations=(
                    "an imported curve gives no utilisation levels: the "
                    "curvature ratio is used, which equals the stress "
                    "utilisation only where stress is proportional to curvature",
                ),
            ),
        )

    def utilisation(
        self,
        tension_N: ArrayLike,
        curvature_1_m: ArrayLike,
        *,
        out_of_range: OutOfRange = "raise",
    ) -> tuple[FloatArray, NDArray[np.int64], dict[str, int]]:
        """Design utilisation of each ``(T, kappa)`` sample.

        Returns the utilisation, the index of the governing mechanism
        (``0`` wire stress or curvature ratio, ``1`` lateral buckling, ``2``
        curvature-independent mechanisms, ``3`` the limits of the components
        other than the armour) and counts of samples outside the tension grid
        and extrapolated beyond the highest level.
        """
        tension = _finite_1d(tension_N, "tension_N")
        kappa = np.abs(_finite_1d(curvature_1_m, "curvature_1_m"))
        if kappa.shape != tension.shape:
            raise CableInputError("tension and curvature histories must match")
        grid = self.tension_N
        tension_u, outside = _interp_rows(
            tension, grid, self.tension_utilisation[None, :], out_of_range
        )
        u_tension = tension_u[0]
        extrapolated = np.zeros(kappa.shape, dtype=np.bool_)
        if not self.has_levels:
            cap, _ = _interp_rows(
                tension, grid, self.kappa_capacity_1_m[None, :], "clamp"
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                u_curv = np.where(
                    cap[0] > 0, kappa / cap[0], np.where(kappa > 0, np.inf, 0)
                )
            u_lat = np.zeros_like(kappa)
        else:
            k_levels, _ = _interp_rows(tension, grid, self.kappa_levels_1_m, "clamp")
            zero_u, _ = _interp_rows(
                tension, grid, self.zero_curvature_utilisation[None, :], "clamp"
            )
            fallback = None
            if self.wire_beyond_slope.size:
                fallback = _interp_rows(
                    tension, grid, self.wire_beyond_slope[None, :], "clamp"
                )[0][0]
            u_curv, extrapolated = _level_utilisation(
                kappa, k_levels, self.levels, zero_u[0], fallback
            )
            lateral, _ = _interp_rows(
                tension, grid, self.lateral_utilisation[None, :], "clamp"
            )
            slip, _ = _interp_rows(
                tension, grid, self.first_slip_curvature_1_m[None, :], "clamp"
            )
            u_lat = np.where(kappa >= slip[0], lateral[0], 0.0)
        u_comp = np.zeros_like(kappa)
        if self.component_usage.size:
            u_comp, beyond = _table_utilisation(
                tension, kappa, grid, self.component_kappa_1_m, self.component_usage
            )
            extrapolated = extrapolated | beyond
        elif self.has_levels and self.has_components:
            c_levels, _ = _interp_rows(
                tension, grid, self.component_levels_1_m, "clamp"
            )
            c_zero, _ = _interp_rows(
                tension, grid, self.component_zero_utilisation[None, :], "clamp"
            )
            u_comp, beyond = _level_utilisation(kappa, c_levels, self.levels, c_zero[0])
            extrapolated = extrapolated | beyond
        stack = np.vstack([u_curv, u_lat, u_tension, u_comp])
        governing = np.argmax(stack, axis=0).astype(np.int64)
        utilisation = np.max(stack, axis=0)
        counts = {
            "outside_tension_grid": int(np.count_nonzero(outside)),
            "extrapolated_beyond_levels": int(np.count_nonzero(extrapolated)),
        }
        return np.asarray(utilisation, dtype=np.float64), governing, counts

    def curvature_capacity(self, tension_N: ArrayLike) -> FloatArray:
        """Design curvature capacity at the given tensions (clamped) [1/m]."""
        tension = _finite_1d(tension_N, "tension_N")
        values, _ = _interp_rows(
            tension, self.tension_N, self.kappa_capacity_1_m[None, :], "clamp"
        )
        return np.asarray(values[0], dtype=np.float64)


def _table_utilisation(
    tension: FloatArray,
    kappa: FloatArray,
    grid: FloatArray,
    k_grid: FloatArray,
    table: FloatArray,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Utilisation interpolated from a (tension, curvature) table.

    Linear in curvature between the tabulated curvatures and linear in
    tension between the grid tensions (clamped at the ends of the grid);
    beyond the last curvature each row is extended with the slope of its last
    interval (never falling), and such samples are flagged.
    """
    xc = np.clip(tension, grid[0], grid[-1])
    if grid.size == 1:
        i = np.zeros(tension.size, dtype=np.int64)
        w = np.zeros(tension.size)
    else:
        i = np.clip(np.searchsorted(grid, xc, side="right") - 1, 0, grid.size - 2)
        w = (xc - grid[i]) / (grid[i + 1] - grid[i])
    j = np.minimum(i + 1, grid.size - 1)
    beyond = kappa > k_grid[-1]
    if k_grid.size == 1:
        low = table[i, 0]
        high = table[j, 0]
        return (1.0 - w) * low + w * high, beyond
    k = np.clip(np.searchsorted(k_grid, kappa, side="right") - 1, 0, k_grid.size - 2)
    span = k_grid[k + 1] - k_grid[k]
    v = np.clip((kappa - k_grid[k]) / span, 0.0, 1.0)

    def row(index: NDArray[np.int64]) -> FloatArray:
        inside = (1.0 - v) * table[index, k] + v * table[index, k + 1]
        last = table[index, -1]
        slope = np.maximum((last - table[index, -2]) / (k_grid[-1] - k_grid[-2]), 0.0)
        outside = last + slope * (kappa - k_grid[-1])
        return np.asarray(np.where(beyond, outside, inside), dtype=np.float64)

    out = (1.0 - w) * row(i) + w * row(j)
    return out, beyond


def _level_utilisation(
    kappa: FloatArray,
    k_levels: FloatArray,
    levels: FloatArray,
    zero_u: FloatArray,
    fallback_slope: FloatArray | None = None,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Utilisation by linear interpolation in curvature between level curves.

    Beyond the highest level the slope of the last two levels extends it;
    where those coincide (both passed at zero curvature) ``fallback_slope``
    does (the slope of the exact usage there), so the utilisation keeps
    rising with curvature instead of staying flat.
    """
    n_levels = levels.size
    count = np.sum(k_levels <= kappa[None, :], axis=0)
    out = np.empty_like(kappa)
    extrapolated = np.zeros(kappa.shape, dtype=np.bool_)
    columns = np.arange(kappa.size)
    # Below the first level curve: from the usage at zero curvature.
    below = count == 0
    if np.any(below):
        k1 = k_levels[0, below]
        u0 = np.minimum(zero_u[below], levels[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            out[below] = np.where(
                np.isfinite(k1) & (k1 > 0),
                u0 + (levels[0] - u0) * kappa[below] / k1,
                u0,
            )
    inside = (count > 0) & (count < n_levels)
    if np.any(inside):
        upper = count[inside]
        lower = upper - 1
        cols = columns[inside]
        ka = k_levels[lower, cols]
        kb = k_levels[upper, cols]
        ua = levels[lower]
        ub = levels[upper]
        # A level below the usage at zero curvature is passed already at
        # kappa = 0 (its curve collapses there): interpolate from the usage at
        # zero curvature, not from that level, or the usage is understated.
        za = zero_u[inside]
        passed = za > ua
        ka = np.where(passed, 0.0, ka)
        ua = np.where(passed, za, ua)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(
                np.isfinite(kb) & (kb > ka), (kappa[inside] - ka) / (kb - ka), 0.0
            )
        out[inside] = ua + frac * (ub - ua)
    top = count == n_levels
    if np.any(top):
        cols = columns[top]
        kn = k_levels[n_levels - 1, cols]
        un = levels[n_levels - 1]
        spare = np.zeros(cols.size) if fallback_slope is None else fallback_slope[cols]
        if n_levels > 1:
            km = k_levels[n_levels - 2, cols]
            um = levels[n_levels - 2]
            spread = kn > km
            with np.errstate(divide="ignore", invalid="ignore"):
                slope = np.where(spread, (un - um) / (kn - km), spare)
        else:
            spread = kn > 0
            with np.errstate(divide="ignore", invalid="ignore"):
                slope = np.where(spread, un / kn, spare)
        # Without a slope of the levels the usage at zero curvature (above
        # the highest level) is where the exact slope starts from.
        start = np.where(spread, un, np.maximum(zero_u[top], un))
        value = start + slope * (kappa[top] - kn)
        out[top] = np.maximum(value, np.maximum(zero_u[top], un))
        extrapolated[top] = kappa[top] > kn
    # Bending only adds to the wire usage at zero curvature.
    return np.maximum(out, zero_u), extrapolated


def _slope_beyond(usage: Any, k_top: float, eta: float) -> float:
    """Slope of the wire design utilisation beyond the highest level [m].

    From the exact usage: at the curvature of the highest level (zero when the
    usage passes it already at zero curvature) over a step of a hundredth of
    that curvature or of 1e-3 1/m. It extends the utilisation beyond the
    highest level where the last two level curves coincide (both at zero) and
    give no slope of their own.
    """
    k0 = k_top if math.isfinite(k_top) else 0.0
    step = max(0.01 * k0, 1.0e-3)
    return float(max((usage(k0 + step) - usage(k0)) / step / eta, 0.0))


def _zero_curve(
    cable: CableData, tension: float, mu: float | None, factor: float
) -> Any:
    """The utilisation of the components at zero curvature only, as a curve."""
    from cablexsec.analysis.components import (  # noqa: PLC0415
        ComponentUsageCurve,
        capacity_utilisation,
        compute_component_response,
    )

    response = compute_component_response(cable, 0.0, tension_N=tension, mu=mu)
    names: list[str] = []
    values: list[float] = []
    modes: list[str] = []
    kinematic = []
    for item in response.states:
        if item.component_type == "armour wire":
            continue
        if item.kinematic:
            kinematic.append(item.name)
            continue
        value, mode = capacity_utilisation(item, factor)
        if math.isfinite(value):
            names.append(item.name)
            values.append(value)
            modes.append(mode)
    return ComponentUsageCurve(
        tension_N=tension,
        kappa_1_m=np.zeros(1),
        names=tuple(names),
        usage=np.array(values, dtype=np.float64).reshape(len(names), 1),
        modes=tuple((m,) for m in modes),
        kinematic=tuple(sorted(kinematic)),
        factor=factor,
    )


def build_capacity_envelope(
    cable: CableData,
    tension_N: ArrayLike,
    *,
    utilisation_factor: float | None = None,
    levels: ArrayLike | None = None,
    tension_kind: TensionKind = "effective",
    mu: float | None = None,
    annulus: Literal["flooded", "sealed"] = "flooded",
    p_ext_Pa: float = 0.0,
    radial_model: Literal["straight", "curved"] = "straight",
    sheath_hoop_strength_Pa: float | Mapping[str, float] | None = None,
    sheath_strength_source: str | None = None,
    polymer_crush_allowable_Pa: float | None = None,
    polymer_crush_source: str | None = None,
    mbr_m: float | None = None,
    mbr_source: str = "",
    component_utilisation_factor: float | None = None,
    component_limits: bool = True,
    curvature_nodes: ArrayLike = (),
    curvatures: bool = True,
) -> CapacityEnvelope:
    """Capacity envelope with utilisation levels over a tension grid.

    Parameters
    ----------
    cable : CableData
        Fully specified cable with armour layers.
    tension_N : array_like
        Strictly increasing tension grid [N]; it must span the histories to
        be checked (negative values for compression).
    utilisation_factor : float, optional
        Design utilisation factor ``eta`` (default
        ``ARMOUR_UTILISATION_FACTORS["normal"]``, 0.67).
    levels : array_like, optional
        Design utilisation levels of the wire-stress curves, ascending
        (default 0.1, 0.2, ..., 1.0 and ``1/eta``); each must satisfy
        ``level * eta <= 1``.
    tension_kind : {"effective", "true", "material"}
        The tension the grid refers to (default effective tension, as global
        programs report it). In a flooded annulus it is the force on the
        armour; with ``annulus="sealed"`` the true wall tension
        ``T_e - p_ext A_o`` drives the armour (``p_ext_Pa``).
    mu, annulus, p_ext_Pa, radial_model, sheath_hoop_strength_Pa, \
sheath_strength_source
        Options of :func:`~cablexsec.analysis.compression.compression_screen`.
    polymer_crush_allowable_Pa, polymer_crush_source
        Sourced polymer contact-pressure allowable of
        :func:`~cablexsec.analysis.capacity.compute_capacity` (optional).
    mbr_m, mbr_source : float, str
        Minimum bend radius for the curvature check (default the cable's
        ``MBR_dynamic_m``).
    component_utilisation_factor : float, optional
        Design utilisation factor of the components other than the armour
        (default: the factor of the lifecycle phase of ``utilisation_factor``,
        :func:`~cablexsec.analysis.component_limits.component_factor_for`).
    component_limits : bool
        Check the limits of the components other than the armour (default).
        ``False`` gives the armour-only envelope.
    curvature_nodes : array_like, optional
        Curvatures at which the utilisation of the components is computed
        rather than interpolated (a design curvature); the curvature of the
        minimum bend radius is always one.
    curvatures : bool
        ``False`` computes the components at zero curvature only: an envelope
        to scan the utilisation at zero curvature over tension (the tension
        and compression capacities), not to check a bent section.
    """
    from cablexsec.analysis import capacity as cap  # noqa: PLC0415
    from cablexsec.analysis.component_limits import (  # noqa: PLC0415
        component_factor_for,
    )
    from cablexsec.analysis.components import (  # noqa: PLC0415
        component_usage_curve,
        usage_curvature_grid,
    )
    from cablexsec.analysis.compression import (  # noqa: PLC0415
        ARMOUR_UTILISATION_FACTORS,
        _check_options,
        _Options,
        _screen,
    )

    eta = (
        ARMOUR_UTILISATION_FACTORS["normal"]
        if utilisation_factor is None
        else float(utilisation_factor)
    )
    eta_c = (
        component_factor_for(eta)
        if component_utilisation_factor is None
        else float(component_utilisation_factor)
    )
    if not (math.isfinite(eta_c) and 0.0 < eta_c <= 1.0):
        raise CableInputError("component_utilisation_factor must lie in (0, 1]")
    grid = _finite_1d(tension_N, "tension_N")
    if grid.size > 1 and np.any(np.diff(grid) <= 0):
        raise CableInputError("the tension grid must increase strictly")
    if levels is None:
        values = [0.1 * k for k in range(1, 11)]
        if eta < 1.0:
            values.append(1.0 / eta)
        level_values = np.array(sorted(set(values)), dtype=np.float64)
    else:
        level_values = _finite_1d(levels, "levels")
        if np.any(np.diff(level_values) <= 0) or level_values[0] <= 0:
            raise CableInputError("levels must be positive and ascending")
    if np.any(level_values * eta > 1.0 + 1e-12):
        raise CableInputError(
            "a level above 1/eta is beyond first yield and is not computed"
        )
    if (polymer_crush_allowable_Pa is None) != (polymer_crush_source is None):
        raise CableInputError(
            "polymer_crush_allowable_Pa and polymer_crush_source go together"
        )
    _check_options(
        annulus,
        p_ext_Pa,
        radial_model,
        eta,
        sheath_hoop_strength_Pa,
        sheath_strength_source,
        None,
    )
    opts = _Options(
        annulus=annulus,
        p_ext_Pa=p_ext_Pa,
        mu=mu,
        radial_model=radial_model,
        sheath_hoop_strength_Pa=sheath_hoop_strength_Pa,
        sheath_strength_source=sheath_strength_source,
        tapes=(),
        utilisation_factor=eta,
    )
    characteristic = np.minimum(level_values * eta, 1.0)
    n_t = grid.size
    k_levels = np.zeros((level_values.size, n_t))
    zero_u = np.zeros(n_t)
    beyond = np.zeros(n_t)
    lateral_u = np.zeros(n_t)
    slip = np.full(n_t, np.inf)
    tension_u = np.zeros(n_t)
    k_design = np.zeros(n_t)
    k_lat = np.full(n_t, math.inf)
    drives = np.zeros(n_t)
    radials = np.zeros(n_t)
    conductors = np.zeros(n_t)
    crushes = np.zeros(n_t)
    for i, t in enumerate(grid):
        screen = _screen(cable, float(t), None, opts)
        drive = screen.driving_tension_N
        drives[i] = drive
        usage, bracket = cap._wire_usage(
            cable,
            drive,
            mu=mu,
            sigma_T_Pa=0.0,
            contact_model=None,
            section_state=None,
            phi_bend=0.0,
        )
        cache: list[float] = []

        def cached(bracket: Any = bracket, cache: list[float] = cache) -> float:
            if not cache:
                cache.append(bracket())
            return cache[0]

        zero_u[i] = usage(0.0) / eta
        if curvatures:
            k_levels[:, i] = [
                cap._usage_root(usage, cached, float(level)) for level in characteristic
            ]
            k_design[i] = cap._usage_root(usage, cached, eta)
            beyond[i] = _slope_beyond(usage, float(k_levels[-1, i]), eta)
        else:
            # Zero curvature only: a level is passed there or not at all.
            k_levels[:, i] = np.where(characteristic <= usage(0.0), 0.0, math.inf)
            k_design[i] = 0.0 if usage(0.0) >= eta else math.inf
        radials[i] = screen.radial.utilisation / eta
        conductors[i] = cap._conductor_uf(cable, drive)
        crush = cap._crush_uf(cable, drive, polymer_crush_allowable_Pa)
        crushes[i] = crush
        tension_u[i] = max(
            radials[i], conductors[i], 0.0 if not math.isfinite(crush) else crush
        )
        if drive < 0.0:
            active = [lat for lat in screen.lateral if math.isfinite(lat.capacity_N)]
            lateral_u[i] = max((lat.utilisation for lat in active), default=0.0) / eta
            slip[i] = min(
                (lat.first_slip_curvature_1_m for lat in screen.lateral),
                default=math.inf,
            )
        k_lat[i] = slip[i] if lateral_u[i] >= 1.0 else math.inf
    mbr = mbr_m if mbr_m is not None else cable.cable.MBR_dynamic_m
    # The components on one curvature grid over every tension, so that their
    # utilisation is interpolated between computed values, not rebuilt from
    # the curvatures of a few levels.
    nodes = [abs(float(k)) for k in np.atleast_1d(curvature_nodes)]
    if mbr is not None and mbr > 0.0:
        nodes.append(1.0 / float(mbr))
    reach = k_levels[np.isfinite(k_levels)]
    top = 1.5 * max(
        float(reach.max()) if reach.size else 0.0,
        max(nodes, default=0.0),
        0.2,
    )
    c_grid = usage_curvature_grid(top, nodes=nodes) if curvatures else np.zeros(1)
    c_table = np.zeros((n_t, c_grid.size))
    c_levels = np.full((level_values.size, n_t), math.inf)
    c_zero = np.zeros(n_t)
    c_governing: list[tuple[str, str]] = []
    kinematic: set[str] = set()
    assessed = False
    governing: list[str] = []
    k_cap = np.zeros(n_t)
    for i in range(n_t):
        k_component = math.inf
        if component_limits:
            curve = (
                component_usage_curve(
                    cable,
                    float(drives[i]),
                    top,
                    mu=mu,
                    factor=eta_c,
                    nodes=nodes,
                    directions=None,
                )
                if curvatures
                else _zero_curve(cable, float(drives[i]), mu, eta_c)
            )
            kinematic.update(curve.kinematic)
            if curve.assessed:
                assessed = True
                largest = curve.largest()
                c_table[i] = largest
                c_zero[i] = float(largest[0])
                c_levels[:, i] = [curve.root(float(level)) for level in level_values]
                k_component = curve.root(1.0)
                c_governing.append(
                    curve.governing(
                        k_component if math.isfinite(k_component) else math.inf
                    )
                )
            else:
                c_governing.append(("", ""))
        else:
            c_governing.append(("", ""))
        crush = crushes[i]
        # A tension mechanism at one (to round-off) closes the envelope.
        if tension_u[i] >= 1.0 - 1e-9 and tension_u[i] >= c_zero[i]:
            k_cap[i] = 0.0
            governing.append(
                "radial_buckling"
                if radials[i]
                >= max(conductors[i], crush if math.isfinite(crush) else 0.0)
                else "conductor_fracture"
                if conductors[i] >= (crush if math.isfinite(crush) else 0.0)
                else "polymer_crush"
            )
        elif k_component < min(k_lat[i], k_design[i]) or c_zero[i] >= 1.0:
            k_cap[i] = k_component
            governing.append("component_limit")
        elif k_lat[i] < k_design[i]:
            k_cap[i] = k_lat[i]
            governing.append("lateral_buckling")
        else:
            k_cap[i] = k_design[i]
            governing.append("wire_yield")
    mbr_note = (
        mbr_source
        if mbr_m is not None
        else ("cable YAML MBR_dynamic_m" if mbr is not None else "")
    )
    if mbr_m is not None and not mbr_source.strip():
        raise CableInputError("mbr_m needs its source in mbr_source")
    return CapacityEnvelope(
        tension_N=grid,
        kappa_capacity_1_m=k_cap,
        tension_utilisation=tension_u,
        utilisation_factor=eta,
        tension_kind=tension_kind,
        levels=level_values,
        kappa_levels_1_m=k_levels,
        zero_curvature_utilisation=zero_u,
        lateral_utilisation=lateral_u,
        first_slip_curvature_1_m=slip,
        governing=tuple(governing),
        mbr_m=mbr,
        mbr_source=mbr_note,
        source=f"CableXSec envelope of {cable.cable.name}",
        component_utilisation_factor=eta_c,
        component_levels_1_m=c_levels if assessed else np.zeros((0, 0)),
        component_zero_utilisation=c_zero if assessed else np.zeros(0),
        component_governing=tuple(c_governing) if assessed else (),
        kinematic_layers=tuple(sorted(kinematic)),
        component_kappa_1_m=c_grid if assessed else np.zeros(0),
        component_usage=c_table if assessed else np.zeros((0, 0)),
        wire_beyond_slope=beyond,
        provenance=Provenance(
            method=(
                "capacity envelope with utilisation levels: largest armour "
                "wire stress over the yield stress, from the common slip-front "
                "stress, at design utilisation levels "
                f"{', '.join(f'{v:g}' for v in level_values)} (design "
                f"utilisation factor {eta:g}); armour lateral and radial buckling "
                f"({annulus} annulus, {radial_model} radial model); conductor "
                "fracture"
                + ("; polymer crush" if polymer_crush_allowable_Pa is not None else "")
                + (
                    "; limits of the other components (0.2 % proof stress of the "
                    "conductor, allowable strain of polymers and of metallic "
                    "screens and sheaths) with the component design utilisation "
                    f"factor {eta_c:g}"
                    if assessed
                    else ""
                )
            ),
            citations=(
                "[Skeie2012] Sec. 3.1, Fig. 11",
                "[HelicaFlier2018] p. 2",
                "[DNV-RP-F401] Sec. 3.8.1, Table 3-1, p. 13",
                "[SaevikThorsen2017]",
                "[LiVaz2021a]",
            )
            + (
                ("[MecLim2025] Table 5, p. 19 (components other than the armour)",)
                if assessed
                else ()
            )
            + ((polymer_crush_source,) if polymer_crush_source else ()),
            gates=(
                "level eta equals wire_yield_curvature(usage_limit=eta)",
                "tests/test_capacity_check.py",
            ),
            limitations=(
                "the design utilisation factor is applied to the armour wire "
                "stress and "
                "buckling capacities, the component factor to the limits of the "
                "other components; conductor fracture and polymer crushing "
                "are characteristic",
                "a component without a limit in its material is not checked; "
                "layers outside the load path (loose screens) are left out, "
                "their kinematic stress being an upper bound",
                "lateral buckling utilisation uses the slip-active capacity with "
                "the largest tabulated knock-down once the curvature passes "
                "first slip (conservative)",
                "torsion is not part of the envelope",
                f"tension grid refers to {tension_kind} tension",
            ),
        ),
    )


@dataclass(frozen=True)
class ExceedanceStatistics:
    """Exceedance of one utilisation (or curvature-ratio) threshold."""

    threshold: float
    samples: int
    upcrossings: int
    time_above_s: float


@dataclass(frozen=True)
class MBRCheck:
    """Curvature against the minimum bend radius ``MBR``.

    ``utilisation_max = kappa_max * MBR``; exceedances of ``kappa > 1/MBR``.
    """

    mbr_m: float
    source: str
    curvature_limit_1_m: float
    curvature_max_1_m: float
    minimum_radius_m: float
    utilisation_max: float
    time_of_max_s: float
    exceedance: ExceedanceStatistics


@dataclass(frozen=True)
class GumbelFit:
    """Gumbel (type I) fit of maxima by the method of moments ([Gumbel1958]).

    ``most_probable`` is the mode ``mu``; ``fractile_value`` the
    ``fractile`` quantile; ``expected`` the mean ``mu + 0.5772 beta``.
    ``duration_shift`` is the ``beta ln(n)`` already added for a duration
    ``n`` times that of one maximum (block maxima), else zero.
    """

    n: int
    mean: float
    std: float
    location: float
    scale: float
    most_probable: float
    expected: float
    fractile: float
    fractile_value: float
    duration_shift: float = 0.0


def gumbel_fit(
    maxima: ArrayLike, *, fractile: float = 0.9, duration_ratio: float = 1.0
) -> GumbelFit:
    """Method-of-moments Gumbel fit of independent maxima.

    ``duration_ratio`` (>= 1) moves the distribution to the maximum over
    that many periods of one maximum: location ``mu + beta ln(ratio)``.
    """
    values = _finite_1d(maxima, "maxima")
    if values.size < 2:
        raise CableInputError("a Gumbel fit needs at least two maxima")
    if not 0.0 < fractile < 1.0:
        raise CableInputError("fractile must lie in (0, 1)")
    if not math.isfinite(duration_ratio) or duration_ratio < 1.0:
        raise CableInputError("duration_ratio must be at least one")
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    scale = math.sqrt(6.0) * std / math.pi
    shift = scale * math.log(duration_ratio)
    location = mean - EULER_GAMMA * scale + shift
    return GumbelFit(
        n=int(values.size),
        mean=mean,
        std=std,
        location=location,
        scale=scale,
        most_probable=location,
        expected=location + EULER_GAMMA * scale,
        fractile=fractile,
        fractile_value=location - scale * math.log(-math.log(fractile)),
        duration_shift=shift,
    )


@dataclass(frozen=True)
class CapacityCheckResult:
    """Utilisation history of one section and its statistics.

    ``utilisation`` is the design utilisation (one at the capacity curve);
    ``governing`` indexes :attr:`mechanisms`. ``block_extremes`` is the
    within-record Gumbel fit when blocks were requested.
    """

    name: str
    time_s: FloatArray
    tension_N: FloatArray
    curvature_1_m: FloatArray
    utilisation: FloatArray
    governing: NDArray[np.int64]
    max_utilisation: float
    time_of_max_s: float
    index_of_max: int
    tension_at_max_N: float
    curvature_at_max_1_m: float
    capacity_curvature_at_max_1_m: float
    governing_at_max: str
    exceedances: tuple[ExceedanceStatistics, ...]
    mbr: MBRCheck | None
    torque_max_Nm: float | None
    block_extremes: GumbelFit | None
    counts: dict[str, int]
    provenance: Provenance
    mechanisms: tuple[str, ...] = _MECHANISMS

    def summary(self) -> dict[str, Any]:
        """JSON-ready summary (no time series)."""
        return {
            "max_utilisation": self.max_utilisation,
            "time_of_max_s": self.time_of_max_s,
            "tension_at_max_N": self.tension_at_max_N,
            "curvature_at_max_1_m": self.curvature_at_max_1_m,
            "capacity_curvature_at_max_1_m": self.capacity_curvature_at_max_1_m,
            "governing_at_max": self.governing_at_max,
            "exceedances": [vars(e) for e in self.exceedances],
            "mbr": None
            if self.mbr is None
            else {**vars(self.mbr), "exceedance": vars(self.mbr.exceedance)},
            "torque_max_Nm": self.torque_max_Nm,
            "block_extremes": None
            if self.block_extremes is None
            else vars(self.block_extremes),
            "counts": dict(self.counts),
        }


def _exceedance(
    values: FloatArray, threshold: float, dt: FloatArray
) -> ExceedanceStatistics:
    above = values > threshold
    ups = int(np.count_nonzero(~above[:-1] & above[1:])) + int(bool(above[0]))
    return ExceedanceStatistics(
        threshold=float(threshold),
        samples=int(np.count_nonzero(above)),
        upcrossings=ups,
        time_above_s=float(np.sum(dt[above])),
    )


def _sample_durations(time: FloatArray) -> FloatArray:
    """Time attributed to each sample (half the adjacent steps; two or more)."""
    step = np.diff(time)
    dt = np.zeros_like(time)
    dt[:-1] += 0.5 * step
    dt[1:] += 0.5 * step
    return dt


def check_capacity(
    envelope: CapacityEnvelope,
    time_s: ArrayLike,
    tension_N: ArrayLike,
    kappa_x_1_m: ArrayLike,
    kappa_y_1_m: ArrayLike | None = None,
    *,
    tension_kind: TensionKind = "effective",
    name: str = "section",
    thresholds: Sequence[float] = (1.0,),
    mbr_m: float | None = None,
    mbr_source: str = "",
    torque_Nm: ArrayLike | None = None,
    blocks: int | None = None,
    fractile: float = 0.9,
    extreme_duration_s: float | None = None,
    out_of_range: OutOfRange = "raise",
) -> CapacityCheckResult:
    """Check a tension-curvature history against a capacity envelope.

    Parameters
    ----------
    envelope : CapacityEnvelope
        From :func:`build_capacity_envelope` or
        :meth:`CapacityEnvelope.from_points`.
    time_s, tension_N : array_like
        Sample times [s] and tension [N] of the kind ``tension_kind``.
    kappa_x_1_m, kappa_y_1_m : array_like
        Curvature components [1/m] (either convention: only the magnitude
        is used, the section being treated as axisymmetric).
    thresholds : sequence of float
        Utilisation thresholds for the exceedance counts (default 1).
    mbr_m, mbr_source : float, str
        Minimum bend radius (default the envelope's).
    torque_Nm : array_like, optional
        Torque history; its largest magnitude is reported (no torsion
        capacity is part of the envelope).
    blocks : int, optional
        Number of equal blocks for a within-record Gumbel fit of the block
        maxima, moved to ``extreme_duration_s`` (default the record).
    out_of_range : {"raise", "clamp"}
        Tensions outside the envelope grid raise, or use the nearest grid
        tension and are counted.
    """
    if tension_kind != envelope.tension_kind:
        raise CableInputError(
            f"the history gives {tension_kind} tension but the envelope grid "
            f"refers to {envelope.tension_kind} tension"
        )
    time = _finite_1d(time_s, "time_s")
    if time.size < 2 or np.any(np.diff(time) <= 0):
        raise CableInputError("time_s must increase strictly with two samples or more")
    tension = _finite_1d(tension_N, "tension_N")
    kx = _finite_1d(kappa_x_1_m, "kappa_x_1_m")
    ky = (
        np.zeros_like(kx) if kappa_y_1_m is None else _finite_1d(kappa_y_1_m, "kappa_y")
    )
    if not (tension.shape == kx.shape == ky.shape == time.shape):
        raise CableInputError("time, tension and curvature histories must match")
    kappa = np.hypot(kx, ky)
    utilisation, governing, counts = envelope.utilisation(
        tension, kappa, out_of_range=out_of_range
    )
    k = int(np.argmax(utilisation))
    dt = _sample_durations(time)
    exceed = tuple(_exceedance(utilisation, float(c), dt) for c in thresholds)
    mbr = envelope.mbr_m if mbr_m is None else mbr_m
    if mbr_m is not None and not mbr_source.strip():
        raise CableInputError("mbr_m needs its source in mbr_source")
    mbr_check = None
    if mbr is not None:
        km = int(np.argmax(kappa))
        limit = 1.0 / mbr
        mbr_check = MBRCheck(
            mbr_m=float(mbr),
            source=mbr_source if mbr_m is not None else envelope.mbr_source,
            curvature_limit_1_m=limit,
            curvature_max_1_m=float(kappa[km]),
            minimum_radius_m=math.inf if kappa[km] == 0 else 1.0 / float(kappa[km]),
            utilisation_max=float(kappa[km]) * float(mbr),
            time_of_max_s=float(time[km]),
            exceedance=_exceedance(kappa, limit, dt),
        )
    torque_max = None
    if torque_Nm is not None:
        torque = _finite_1d(torque_Nm, "torque_Nm")
        if torque.shape != time.shape:
            raise CableInputError("torque_Nm must match the time samples")
        torque_max = float(np.max(np.abs(torque)))
    block_fit = None
    if blocks is not None:
        if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 2:
            raise CableInputError("blocks must be an integer of at least 2")
        edges = np.linspace(time[0], time[-1], blocks + 1)
        index = np.clip(np.searchsorted(edges, time, side="right") - 1, 0, blocks - 1)
        maxima = np.array(
            [
                np.max(utilisation[index == b])
                for b in range(blocks)
                if np.any(index == b)
            ]
        )
        block_duration = (time[-1] - time[0]) / blocks
        target = (
            time[-1] - time[0] if extreme_duration_s is None else extreme_duration_s
        )
        block_fit = gumbel_fit(
            maxima, fractile=fractile, duration_ratio=max(1.0, target / block_duration)
        )
    limitations = [
        "curvature magnitude only: the envelope treats the section as "
        "axisymmetric (wire positions are not resolved against the bending "
        "direction)",
        "each sample is assessed quasi-statically on the envelope; the slip "
        "history of the time series is not carried",
    ]
    if counts["outside_tension_grid"]:
        limitations.append(
            f"{counts['outside_tension_grid']} samples outside the tension grid "
            "used the nearest grid tension"
        )
    if counts["extrapolated_beyond_levels"]:
        limitations.append(
            f"{counts['extrapolated_beyond_levels']} samples beyond the highest "
            "level curve (first yield) were extrapolated linearly in curvature"
        )
    if torque_max is not None:
        limitations.append("torque reported only; no torsion capacity is checked")
    base = envelope.provenance
    provenance = Provenance(
        method="capacity check of a global time series: design utilisation per "
        "sample on the tension-curvature envelope ("
        + envelope.source
        + "), maximum, exceedances"
        + (", minimum bend radius" if mbr_check is not None else "")
        + (", Gumbel block maxima" if block_fit is not None else ""),
        citations=(
            *(() if base is None else base.citations),
            "[Gumbel1958]",
        ),
        gates=("tests/test_capacity_check.py",),
        limitations=(*(() if base is None else base.limitations), *limitations),
    )
    capacity_at_max = float(envelope.curvature_capacity(tension[k : k + 1])[0])
    return CapacityCheckResult(
        name=name,
        time_s=time,
        tension_N=tension,
        curvature_1_m=kappa,
        utilisation=utilisation,
        governing=governing,
        max_utilisation=float(utilisation[k]),
        time_of_max_s=float(time[k]),
        index_of_max=k,
        tension_at_max_N=float(tension[k]),
        curvature_at_max_1_m=float(kappa[k]),
        capacity_curvature_at_max_1_m=capacity_at_max,
        governing_at_max=_MECHANISMS[int(governing[k])],
        exceedances=exceed,
        mbr=mbr_check,
        torque_max_Nm=torque_max,
        block_extremes=block_fit,
        counts=counts,
        provenance=provenance,
    )


def check_capacity_series(
    envelope: CapacityEnvelope,
    series: Any,
    *,
    external_area_m2: float | None = None,
    **options: Any,
) -> CapacityCheckResult:
    """:func:`check_capacity` of a reader output.

    Accepts a :class:`~cablexsec.io.series.LoadCaseSeries` (time-history CSV,
    RIFLEX, canonical CSV; its ``tension_kind`` must be declared)
    or a :class:`~cablexsec.io.cabledyn.CableDynSectionHistory` (effective
    tension, or the true tension with ``external_area_m2``; its torque is
    reported).
    """
    from cablexsec.io.cabledyn import CableDynSectionHistory  # noqa: PLC0415
    from cablexsec.io.series import LoadCaseSeries  # noqa: PLC0415

    if isinstance(series, LoadCaseSeries):
        if series.tension_kind == "unspecified":
            raise CableInputError(
                "the series does not declare its tension kind (effective or true)"
            )
        return check_capacity(
            envelope,
            series.time_s,
            series.tension_N,
            series.curvature_x_1_m,
            series.curvature_y_1_m,
            tension_kind=series.tension_kind,
            **options,
        )
    if isinstance(series, CableDynSectionHistory):
        kx, ky = series.strain_gradient()
        if envelope.tension_kind == "effective":
            tension = series.effective_tension_N
        else:
            if external_area_m2 is None:
                raise CableInputError("true tension needs external_area_m2")
            tension = series.true_tension_N(external_area_m2)
        options.setdefault("name", series.name)
        if series.torque_Nm is not None:
            options.setdefault("torque_Nm", series.torque_Nm)
        return check_capacity(
            envelope,
            series.time_s,
            tension,
            kx,
            ky,
            tension_kind=envelope.tension_kind,
            **options,
        )
    raise CableInputError("series must be a LoadCaseSeries or a CableDynSectionHistory")


def capacity_extremes(
    results: Sequence[CapacityCheckResult], *, fractile: float = 0.9
) -> GumbelFit:
    """Gumbel fit of the maximum utilisation of seeds of one condition."""
    if len(results) < 2:
        raise CableInputError("an extreme-value fit needs at least two seeds")
    return gumbel_fit([r.max_utilisation for r in results], fractile=fractile)
