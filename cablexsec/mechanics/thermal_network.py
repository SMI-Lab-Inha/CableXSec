# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Transient radial thermal network of a cable definition (slow time).

The network is the IEC 60287 conduction ladder of
:func:`cablexsec.mechanics.thermal.compute_thermal` written as a lumped
resistance--capacitance chain, so that a current history ``I(tau)`` gives the
temperature of every cable layer on the slow time scale.

Construction
------------
* The heat path, its ``T1`` split at the metallic sheath or screen and the
  equal-filler-area annulus of a three-core filler are those of
  ``compute_thermal`` (one private assembly is shared by both).
* Every annulus is divided into ``n_per_layer`` sub-annuli of equal
  logarithmic width. A sub-annulus between ``r_a`` and ``r_b`` has the
  conductance ``m 2 pi lambda / ln(r_b/r_a)`` and the heat capacity
  ``m rho c pi (r_b^2 - r_a^2)`` split equally between its two nodes. The
  multiplicity ``m`` is ``n`` (the loaded cores in parallel) for an annulus
  whose IEC heat flow is that of one core and ``1`` for an annulus that
  carries the losses of all ``n`` cores, so that the chain carries the total
  heat and its temperature drops are exactly the IEC drops
  ([IEC60287-1-1] 1.4.1.1 Eq. 1; [NeherMcGrath1957] Eqs. 1--2).
* The conductor node holds the conductor heat capacity
  (``n A_c rho c``); metallic shells and armour wires, which carry no
  conduction resistance in the ladder, add their heat capacity to the nearest
  node at their radius. Radial gaps between annuli have no resistance.
* Sources, per unit cable length: ``n (W_c + W_d/2)`` at the conductor node
  and ``n (lambda1 W_c + W_d/2)`` at the node where ``T1`` ends, so that the
  flow through ``T1`` is ``n (W_c + W_d/2)`` and outside it
  ``n (W_c (1 + lambda1) + W_d)``, as in Eq. 1. The surface node is joined
  to the ambient temperature by the external resistance ``T4`` (a series
  resistance without heat capacity).
* ``W_c = I^2 R_ac(T_c)`` with ``R_ac = R_20 (1 + alpha_20 (T_c - 20))(1 + y_s
  + y_p)`` ([IEC60287-1-1] 2.1). The loss is affine in ``T_c``, so the
  temperature-dependent resistance is solved together with the temperatures
  (implicitly, without iteration) unless a fixed evaluation temperature is
  given, which is the convention of ``compute_thermal``.

With a fixed resistance temperature the steady state of the network equals
``compute_thermal`` at every ladder radius to round-off, independently of
``n_per_layer`` (series resistances; tests/test_thermal_history.py).

Time integration
----------------
``C dT/dtau = -K(tau) T + s(tau)`` is integrated by the L-stable, second-order
TR-BDF2 scheme (default; thin metallic layers have time constants of seconds,
which the trapezoidal rule would leave oscillating at large steps) or by the
Crank--Nicolson rule, for which the discrete energy balance
``sum C dT = h/2 (Q_n + Q_{n+1})`` holds to round-off.

The armour loss ``lambda2 W_c`` per core enters at the nodes of the structural
armour layers in equal shares, as in the steady ladder; ``lambda1`` and
``lambda2`` are whatever the caller supplies.

Not represented (stated, not hidden): the sheath eddy-current factor
``lambda1''`` (the IEC 60287-1-1 texts are not held), the IEC 60287-2-1
three-core geometric factor, the heat capacity of the surroundings (``T4`` is
quasi-steady, which shortens the response of a buried cable) and axial heat
flow. Frictional heating enters through
:mod:`cablexsec.analysis.frictional_heating`, which adds its sources to
:meth:`ThermalNetwork.system_matrices` at the nodes of
:meth:`ThermalNetwork.source_weights`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from cablexsec.exceptions import CableInputError, CableLookupError
from cablexsec.materials.library import get_property
from cablexsec.mechanics.thermal import (
    _cable_dielectric_loss,
    _ladder_geometry,
    ac_resistance,
)
from cablexsec.provenance import Provenance

if TYPE_CHECKING:
    from cablexsec.cables.schema import CableData

__all__ = ["ThermalNetwork", "thermal_network"]

FloatArray = NDArray[np.float64]

_SCHEMES = ("tr-bdf2", "crank-nicolson")
# TR-BDF2 stage parameter (L-stable choice 2 - sqrt(2)).
_GAMMA = 2.0 - math.sqrt(2.0)


@dataclass(frozen=True)
class ThermalNetwork:
    """Lumped radial RC network of one cable (whole cable, per metre).

    Attributes
    ----------
    radius_m : numpy.ndarray
        Node radii [m]; per-core nodes are measured from the core axis.
    per_core : numpy.ndarray of bool
        Node lies in a per-core annulus (multi-core cables).
    conductance_W_mK : numpy.ndarray
        Conductance between node ``i`` and ``i + 1`` [W/(m K)].
    capacitance_J_mK : numpy.ndarray
        Heat capacity of each node [J/(m K)].
    external_resistance_Km_W : float
        ``T4`` between the surface node and the ambient [K m/W].
    sheath_node : int
        Node at which ``T1`` ends (sheath or screen losses enter there).
    n_conductors : int
        Number of loaded conductors ``n``.
    r_dc20_ohm_m, alpha_R_per_K : float
        ``R_20 = rho_20 / A`` and ``alpha_20``.
    y_s, y_p : float
        Skin and proximity factors as supplied ([IEC60287-1-1] 2.1).
    conductor_material : str
        Library key of the conductor (for ``rho_20`` and ``alpha_20``).
    conductor_area_m2 : float
        Conductor cross-section of one core [m^2].
    dielectric_loss_W_m : float
        Dielectric loss of one core [W/m] (zero below 100 kV).
    lambda1 : float
        Sheath loss factor supplied by the caller.
    ambient_C : float
        Default ambient temperature [deg C] (cable environment block).
    layer_weights : numpy.ndarray
        ``(n_nodes, n_layers)``: a layer temperature is ``T @ layer_weights``
        (area-weighted mean over an annulus; conductor node for the conductor;
        linear interpolation at the radius of shells and armour wires).
    layer_names : tuple of str
        Column names of ``layer_weights``.
    """

    radius_m: FloatArray
    per_core: NDArray[np.bool_]
    conductance_W_mK: FloatArray
    capacitance_J_mK: FloatArray
    external_resistance_Km_W: float
    sheath_node: int
    n_conductors: int
    r_dc20_ohm_m: float
    alpha_R_per_K: float
    y_s: float
    y_p: float
    conductor_material: str
    conductor_area_m2: float
    dielectric_loss_W_m: float
    lambda1: float
    ambient_C: float
    layer_weights: FloatArray
    layer_names: tuple[str, ...]
    provenance: Provenance | None = field(default=None, compare=False)
    lambda2: float = 0.0
    armour_source: FloatArray | None = field(default=None, compare=False)
    passive_cores: int = 0

    # ------------------------------------------------------------------ #
    @property
    def n_nodes(self) -> int:
        return int(self.radius_m.size)

    def conductor_loss_W_m(
        self,
        current_A: float,
        conductor_temperature_C: float,
        *,
        resistance_temperature_C: float | None = None,
    ) -> float:
        """Joule loss of one conductor ``I^2 R_ac`` [W/m] ([IEC60287-1-1] 2.1)."""
        t_r = (
            conductor_temperature_C
            if resistance_temperature_C is None
            else resistance_temperature_C
        )
        return float(current_A) ** 2 * self._r_ac(t_r)

    def _r_ac(self, temperature_C: float) -> float:
        return ac_resistance(
            get_property(self.conductor_material, "rho_ohm_m"),
            self.conductor_area_m2,
            temperature_C,
            self.alpha_R_per_K,
            y_s=self.y_s,
            y_p=self.y_p,
        )

    def total_heat_W_m(
        self,
        current_A: float,
        conductor_temperature_C: float,
        *,
        resistance_temperature_C: float | None = None,
    ) -> float:
        """Heat generated in the cable ``n (W_c (1 + λ1 + λ2) + W_d)`` [W/m]."""
        w_c = self.conductor_loss_W_m(
            current_A,
            conductor_temperature_C,
            resistance_temperature_C=resistance_temperature_C,
        )
        return self.n_conductors * (
            w_c * (1.0 + self.lambda1 + self.lambda2) + self.dielectric_loss_W_m
        )

    # ------------------------------------------------------------------ #
    def _system(
        self,
        current_A: float,
        ambient_C: float,
        resistance_temperature_C: float | None,
    ) -> tuple[FloatArray, FloatArray]:
        """``K`` and ``s`` of ``C dT/dt = -K T + s`` at one current."""
        n_nodes = self.n_nodes
        g = self.conductance_W_mK
        k = np.zeros((n_nodes, n_nodes))
        idx = np.arange(n_nodes - 1)
        k[idx, idx] += g
        k[idx + 1, idx + 1] += g
        k[idx, idx + 1] -= g
        k[idx + 1, idx] -= g
        g_ext = 1.0 / self.external_resistance_Km_W
        k[-1, -1] += g_ext
        s = np.zeros(n_nodes)
        s[-1] += g_ext * ambient_C
        n = float(self.n_conductors)
        i2 = float(current_A) ** 2
        if resistance_temperature_C is None:
            # R_ac(T) = r20 acf (1 + alpha (T - 20)) = a0 + a1 T.
            r20 = self._r_ac(20.0)
            a1 = r20 * self.alpha_R_per_K
            a0 = r20 - 20.0 * a1
        else:
            a0, a1 = self._r_ac(resistance_temperature_C), 0.0
        w0, w1 = i2 * a0, i2 * a1
        half_wd = 0.5 * self.dielectric_loss_W_m
        s[0] += n * (w0 + half_wd)
        s[self.sheath_node] += n * (self.lambda1 * w0 + half_wd)
        k[0, 0] -= n * w1
        k[self.sheath_node, 0] -= n * self.lambda1 * w1
        if self.lambda2 > 0.0 and self.armour_source is not None:
            s += n * self.lambda2 * w0 * self.armour_source
            k[:, 0] -= n * self.lambda2 * w1 * self.armour_source
        return k, s

    def steady_state(
        self,
        current_A: float,
        *,
        ambient_C: float | None = None,
        resistance_temperature_C: float | None = None,
    ) -> FloatArray:
        """Steady node temperatures [deg C] at a constant current.

        ``resistance_temperature_C=None`` solves ``R_ac(T_c)`` with the
        temperatures; a value evaluates ``R_ac`` there (``compute_thermal``
        convention, default 90 deg C there). Raises if no stable steady state
        exists (thermal runaway of the coupled resistance).
        """
        amb = self.ambient_C if ambient_C is None else float(ambient_C)
        _check_finite(current_A, "current_A")
        _check_finite(amb, "ambient_C")
        k, s = self._system(current_A, amb, resistance_temperature_C)
        if resistance_temperature_C is None:
            _check_stable(k, self.capacitance_J_mK)
        return np.asarray(np.linalg.solve(k, s), dtype=np.float64)

    def time_constants_s(self) -> FloatArray:
        """Time constants of the unloaded network [s], longest first."""
        k, _ = self._system(0.0, 0.0, 0.0)
        c = self.capacitance_J_mK
        scale = 1.0 / np.sqrt(c)
        sym = k * scale[:, None] * scale[None, :]
        eig = np.linalg.eigvalsh(0.5 * (sym + sym.T))
        return np.asarray(np.sort(1.0 / eig)[::-1], dtype=np.float64)

    def transient(
        self,
        time_s: Sequence[float] | FloatArray,
        current_A: Sequence[float] | FloatArray | float,
        *,
        ambient_C: Sequence[float] | FloatArray | float | None = None,
        initial: str | FloatArray = "steady",
        resistance_temperature_C: float | None = None,
        scheme: str = "tr-bdf2",
        max_step_s: float = 60.0,
        periodic: bool = False,
        periodic_tolerance_K: float = 1.0e-3,
        max_periods: int = 60,
    ) -> FloatArray:
        """Node temperatures [deg C] at ``time_s``, shape ``(times, nodes)``.

        ``current_A`` and ``ambient_C`` are linear between the given times.
        Each interval is divided into equal steps of at most ``max_step_s``.
        ``initial`` is ``"steady"`` (steady state at the first current),
        ``"ambient"`` or an array of node temperatures. ``periodic=True``
        repeats the history from its own end state until the start and end
        states agree within ``periodic_tolerance_K`` (a daily cycle in its
        periodic regime); the current and ambient must then return to their
        initial values.
        """
        t = np.asarray(time_s, dtype=np.float64)
        if t.ndim != 1 or t.size < 2 or not np.isfinite(t).all():
            raise CableInputError("time_s must be a finite vector of >= 2 samples")
        if np.any(np.diff(t) <= 0):
            raise CableInputError("time_s must be strictly increasing")
        cur = _as_series(current_A, t, "current_A")
        amb = _as_series(
            self.ambient_C if ambient_C is None else ambient_C, t, "ambient_C"
        )
        if scheme not in _SCHEMES:
            raise CableInputError(f"scheme must be one of {_SCHEMES}")
        _check_finite(max_step_s, "max_step_s")
        if max_step_s <= 0:
            raise CableInputError("max_step_s must be positive")
        if resistance_temperature_C is None:
            k_max, _ = self._system(float(np.max(np.abs(cur))), 0.0, None)
            _check_stable(k_max, self.capacitance_J_mK)
        if isinstance(initial, str):
            if initial == "steady":
                state = self.steady_state(
                    float(cur[0]),
                    ambient_C=float(amb[0]),
                    resistance_temperature_C=resistance_temperature_C,
                )
            elif initial == "ambient":
                state = np.full(self.n_nodes, float(amb[0]))
            else:
                raise CableInputError("initial must be 'steady', 'ambient' or an array")
        else:
            state = np.asarray(initial, dtype=np.float64).copy()
            if state.shape != (self.n_nodes,) or not np.isfinite(state).all():
                raise CableInputError(
                    "initial node temperatures do not match the network"
                )
        if periodic:
            span = max(1.0, float(np.max(np.abs(cur))))
            if abs(cur[-1] - cur[0]) > 1e-9 * span or abs(
                amb[-1] - amb[0]
            ) > 1e-9 * max(1.0, float(np.max(np.abs(amb)))):
                raise CableInputError(
                    "a periodic history must end at its initial current and ambient"
                )
            for _ in range(max_periods):
                out = self._integrate(
                    t, cur, amb, state, resistance_temperature_C, scheme, max_step_s
                )
                change = float(np.max(np.abs(out[-1] - out[0])))
                state = out[-1].copy()
                if change < periodic_tolerance_K:
                    return out
            raise CableInputError(
                "periodic thermal regime not reached; increase max_periods"
            )
        return self._integrate(
            t, cur, amb, state, resistance_temperature_C, scheme, max_step_s
        )

    def _integrate(
        self,
        t: FloatArray,
        cur: FloatArray,
        amb: FloatArray,
        state: FloatArray,
        resistance_temperature_C: float | None,
        scheme: str,
        max_step_s: float,
    ) -> FloatArray:
        c = self.capacitance_J_mK
        out = np.empty((t.size, self.n_nodes))
        out[0] = state
        temp = state.copy()

        def system(time: float) -> tuple[FloatArray, FloatArray]:
            return self._system(
                float(np.interp(time, t, cur)),
                float(np.interp(time, t, amb)),
                resistance_temperature_C,
            )

        k_old, s_old = system(float(t[0]))
        for i in range(t.size - 1):
            steps = max(1, math.ceil((t[i + 1] - t[i]) / max_step_s - 1e-12))
            h = (t[i + 1] - t[i]) / steps
            for j in range(steps):
                t0 = t[i] + j * h
                t1 = t[i + 1] if j == steps - 1 else t0 + h
                h = t1 - t0
                if scheme == "crank-nicolson":
                    k_new, s_new = system(float(t1))
                    lhs = np.diag(c / h) + 0.5 * k_new
                    rhs = (c / h) * temp - 0.5 * (k_old @ temp) + 0.5 * (s_old + s_new)
                    temp = np.linalg.solve(lhs, rhs)
                else:
                    tg = t0 + _GAMMA * h
                    k_g, s_g = system(float(tg))
                    hg = _GAMMA * h
                    lhs = np.diag(c / hg) + 0.5 * k_g
                    rhs = (c / hg) * temp - 0.5 * (k_old @ temp) + 0.5 * (s_old + s_g)
                    t_gamma = np.linalg.solve(lhs, rhs)
                    k_new, s_new = system(float(t1))
                    w = (1.0 - _GAMMA) / (2.0 - _GAMMA)
                    a1 = 1.0 / (_GAMMA * (2.0 - _GAMMA))
                    a0 = (1.0 - _GAMMA) ** 2 / (_GAMMA * (2.0 - _GAMMA))
                    lhs = np.diag(c) + w * h * k_new
                    rhs = c * (a1 * t_gamma - a0 * temp) + w * h * s_new
                    temp = np.linalg.solve(lhs, rhs)
                k_old, s_old = k_new, s_new
            out[i + 1] = temp
        return out

    def resistance_to_ambient_Km_W(self, layer_name: str) -> float:
        """Series thermal resistance from a layer's node to the ambient [K m/W].

        The temperature rise of that layer per watt per metre injected there
        (e.g. frictional heating at an armour layer, ETM Pi2 diagnostic).
        """
        try:
            column = self.layer_weights[:, self.layer_names.index(layer_name)]
        except ValueError as exc:
            raise CableInputError(f"unknown layer {layer_name!r}") from exc
        node = int(np.argmax(column))
        return float(
            np.sum(1.0 / self.conductance_W_mK[node:]) + self.external_resistance_Km_W
        )

    def system_matrices(
        self,
        current_A: float,
        *,
        ambient_C: float | None = None,
        resistance_temperature_C: float | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """``K`` and ``s`` of ``C dT/dt = -K T + s`` at one current.

        The assembly :meth:`transient` integrates: conduction, the external
        resistance to ``ambient_C`` and the conductor and sheath losses
        (``resistance_temperature_C=None`` linearises ``R_ac(T_c)`` into
        ``K``). Copies, so callers may add their own sources.
        """
        amb = self.ambient_C if ambient_C is None else float(ambient_C)
        _check_finite(current_A, "current_A")
        _check_finite(amb, "ambient_C")
        k, s = self._system(current_A, amb, resistance_temperature_C)
        return k.copy(), s.copy()

    def source_weights(self, layer_name: str) -> FloatArray:
        """Node shares of a heat source released in one layer (they sum to one).

        The shares are the layer's temperature weights normalised: an
        annulus spreads its heat over its nodes by area, a shell or an armour
        layer between the two nodes that bracket its radius. One watt per
        metre of cable released in the layer adds these node sources.
        """
        try:
            column = self.layer_weights[:, self.layer_names.index(layer_name)]
        except ValueError as exc:
            raise CableInputError(f"unknown layer {layer_name!r}") from exc
        total = float(column.sum())
        if total <= 0.0:
            raise CableInputError(f"layer {layer_name!r} has no node weights")
        return np.asarray(column / total, dtype=np.float64)

    # ------------------------------------------------------------------ #
    def layer_temperatures(
        self, node_temperature_C: FloatArray
    ) -> dict[str, FloatArray]:
        """Layer temperatures from node temperatures (last axis = nodes)."""
        values = np.asarray(node_temperature_C, dtype=np.float64)
        if values.shape[-1] != self.n_nodes:
            raise CableInputError("node temperatures do not match the network")
        mapped = values @ self.layer_weights
        return {
            name: np.asarray(mapped[..., j], dtype=np.float64)
            for j, name in enumerate(self.layer_names)
        }

    def core_mean_temperatures(
        self, node_temperature_C: FloatArray
    ) -> dict[str, FloatArray]:
        """Layer temperatures with each core layer averaged over all cores.

        With fewer loaded conductors than cores, the passive cores take the
        temperature of the first cable-level node, which holds their heat
        capacity. The cores of a cable definition share one temperature in
        the section model; for parallel cores at a common axial strain the
        mean gives their axial force exactly.
        """
        temps = self.layer_temperatures(node_temperature_C)
        if self.passive_cores == 0 or bool(self.per_core.all()):
            return temps
        values = np.asarray(node_temperature_C, dtype=np.float64)
        bundle = values[..., int(np.flatnonzero(~self.per_core)[0])]
        n, passive = float(self.n_conductors), float(self.passive_cores)
        for j, name in enumerate(self.layer_names):
            if np.any(self.layer_weights[~self.per_core, j] != 0.0):
                continue
            temps[name] = (n * temps[name] + passive * bundle) / (n + passive)
        return temps


def _check_finite(value: float, name: str) -> None:
    if not math.isfinite(float(value)):
        raise CableInputError(f"{name} must be finite")


def _as_series(
    values: Sequence[float] | FloatArray | float, t: FloatArray, name: str
) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(t.size, float(arr))
    if arr.shape != t.shape or not np.isfinite(arr).all():
        raise CableInputError(f"{name} must be finite and match time_s")
    return arr


def _check_stable(k: FloatArray, c: FloatArray) -> None:
    """All modes of ``C dT/dt = -K T`` must decay (no thermal runaway)."""
    scale = 1.0 / np.sqrt(c)
    eig = np.linalg.eigvals(k * scale[:, None] * scale[None, :])
    if np.min(eig.real) <= 0.0:
        raise CableInputError(
            "no stable thermal state: the temperature-dependent conductor "
            "resistance gives thermal runaway at this current"
        )


def _rho_cp(material: str, label: str) -> float:
    try:
        return get_property(material, "rho_cp_J_m3K")
    except CableLookupError as exc:
        raise CableInputError(
            f"{label}: material {material!r} has no volumetric heat capacity "
            "rho_cp_J_m3K for the transient thermal network"
        ) from exc


def thermal_network(
    cable: CableData,
    conductor_area_m2: float,
    *,
    n_cores: int | None = None,
    conductor_material: str = "copper_stranded",
    y_s: float = 0.0,
    y_p: float = 0.0,
    lambda1: float = 0.0,
    T4_external_Km_W: float | None = None,
    ambient_C: float | None = None,
    n_per_layer: int = 4,
    lambda2: float = 0.0,
) -> ThermalNetwork:
    """Build the transient thermal network of a cable definition.

    Arguments as :func:`cablexsec.mechanics.thermal.compute_thermal`;
    ``n_per_layer`` sub-annuli per conduction annulus (the steady state does
    not depend on it). See the module notes for the construction.
    """
    env = cable.environment
    t4 = T4_external_Km_W if T4_external_Km_W is not None else env.T4_external_Km_W
    t_amb = ambient_C if ambient_C is not None else env.ambient_C
    if t4 is None or t_amb is None:
        raise CableInputError(
            "thermal_network needs T4_external_Km_W and ambient_C "
            "(set them in the cable 'environment' block or pass overrides)."
        )
    if not (math.isfinite(t4) and t4 > 0):
        raise CableInputError("T4_external_Km_W must be positive and finite")
    if (
        isinstance(n_per_layer, bool)
        or int(n_per_layer) != n_per_layer
        or n_per_layer < 1
    ):
        raise CableInputError("n_per_layer must be a positive integer")
    if not (math.isfinite(conductor_area_m2) and conductor_area_m2 > 0):
        raise CableInputError("conductor_area_m2 must be positive")
    for name, value in (
        ("y_s", y_s),
        ("y_p", y_p),
        ("lambda1", lambda1),
        ("lambda2", lambda2),
    ):
        _check_finite(value, name)
        if value < 0:
            raise CableInputError(f"{name} must be non-negative")
    geo = _ladder_geometry(cable, n_cores)
    n = geo.n
    if n < 1:
        raise CableInputError("n_cores must be at least 1")
    t1_flags = geo.in_t1
    last_t1 = max((i for i, flag in enumerate(t1_flags) if flag), default=-1)
    if any(not flag for flag in t1_flags[: last_t1 + 1]):
        raise CableInputError("the T1 annuli of the heat path are not contiguous")

    radius: list[float] = [geo.path[0].layer.r_inner_m]
    # Inner radius of the annulus that starts at a node; it exceeds the node
    # radius where a radial gap (no resistance) is collapsed into that node.
    gap_end: dict[int, float] = {}
    per_core: list[bool] = [geo.path[0].per_core]
    conductance: list[float] = []
    capacitance: list[float] = [
        _rho_cp(conductor_material, "conductor") * conductor_area_m2 * n
    ]
    # Heat capacity of the loaded cores (conductors and per-core annuli).
    core_capacity = capacitance[0]
    # Per ladder layer: (node index, sub-area weight) pairs for the mean.
    annulus_weights: dict[str, list[tuple[int, float]]] = {}
    sheath_node = 0
    for index, (annulus, inner) in enumerate(zip(geo.path, t1_flags, strict=True)):
        if geo.multi_core:
            flow = 1.0 if annulus.per_core else float(n)
        else:
            flow = 1.0 if inner else float(n)
        mult = float(n) / flow
        lyr = annulus.layer
        rho_cp = _rho_cp(annulus.material, annulus.layer_name)
        edges = lyr.r_inner_m * (lyr.r_outer_m / lyr.r_inner_m) ** (
            np.arange(n_per_layer + 1) / n_per_layer
        )
        edges[-1] = lyr.r_outer_m
        weights: list[tuple[int, float]] = []
        gap_end[len(radius) - 1] = float(lyr.r_inner_m)
        for r_a, r_b in zip(edges[:-1], edges[1:], strict=True):
            start = len(radius) - 1
            area = math.pi * (r_b**2 - r_a**2)
            conductance.append(
                mult * 2.0 * math.pi * lyr.lambda_W_mK / math.log(r_b / r_a)
            )
            half = 0.5 * mult * rho_cp * area
            capacitance[start] += half
            capacitance.append(half)
            if annulus.per_core:
                core_capacity += 2.0 * half
            radius.append(float(r_b))
            per_core.append(annulus.per_core)
            weights += [(start, 0.5 * area), (start + 1, 0.5 * area)]
        annulus_weights[annulus.layer_name] = weights
        if index == last_t1:
            sheath_node = len(radius) - 1

    r_arr = np.asarray(radius, dtype=np.float64)
    pc_arr = np.asarray(per_core, dtype=bool)
    cap = np.asarray(capacitance, dtype=np.float64)
    frame = np.flatnonzero(~pc_arr) if np.any(~pc_arr) else np.arange(r_arr.size)
    # Cores without current (n < cores of the cable) carry no heat along the
    # ladder; their heat capacity sits in the bundle at the first cable-level
    # node.
    passive = geo.n_geo - n if geo.multi_core and np.any(~pc_arr) else 0
    if passive > 0:
        cap[frame[0]] += passive * core_capacity / n

    # A node spans [radius, gap end]: the collapsed gap has no resistance, so
    # a shell or armour layer inside it sits on that node alone; inside an
    # annulus the position is interpolated in ln r (the conduction profile).
    r_hi = r_arr.copy()
    for node, r_in in gap_end.items():
        r_hi[node] = max(r_hi[node], r_in)

    def interpolation(r: float) -> list[tuple[int, float]]:
        lo, hi = r_arr[frame], r_hi[frame]
        if r <= hi[0]:
            return [(int(frame[0]), 1.0)]
        if r >= lo[-1]:
            return [(int(frame[-1]), 1.0)]
        j = int(np.searchsorted(lo, r, side="right")) - 1
        if r <= hi[j]:
            return [(int(frame[j]), 1.0)]
        w = math.log(r / hi[j]) / math.log(lo[j + 1] / hi[j])
        return [(int(frame[j]), 1.0 - w), (int(frame[j + 1]), w)]

    names: list[str] = []
    columns: list[FloatArray] = []
    notes: list[str] = []
    for layer in cable.layers:
        column = np.zeros(r_arr.size)
        if layer.name in annulus_weights:
            pairs = annulus_weights[layer.name]
            total = sum(w for _, w in pairs)
            for node, w in pairs:
                column[node] += w / total
        elif layer.layer_type == "cylindrical_shell":
            r_mid = float(getattr(layer, "r_mid_m", 0.0) or 0.0)
            t_shell = float(getattr(layer, "t_m", 0.0) or 0.0)
            pairs = interpolation(r_mid)
            for node, w in pairs:
                column[node] += w
            if r_mid > 0 and t_shell > 0:
                node = max(pairs, key=lambda p: p[1])[0]
                cap[node] += _rho_cp(layer.material, layer.name) * (
                    2.0 * math.pi * r_mid * t_shell
                )
        elif layer.layer_type in ("helical", "helical_stranded"):
            r_h = float(getattr(layer, "r_helix_m", 0.0) or 0.0)
            pairs = interpolation(r_h)
            for node, w in pairs:
                column[node] += w
            d_w = getattr(layer, "d_wire_m", None)
            n_w = getattr(layer, "n_wires", None)
            alpha = getattr(layer, "lay_angle_rad", None)
            if d_w and n_w and alpha is not None:
                node = max(pairs, key=lambda p: p[1])[0]
                cap[node] += _rho_cp(layer.material, layer.name) * (
                    n_w * math.pi * d_w**2 / 4.0 / math.cos(alpha)
                )
        elif (getattr(layer, "r_inner_m", None) or 0.0) <= 0.0 and (
            layer.layer_type == "cylindrical_offset" or not geo.multi_core
        ):
            column[0] = 1.0  # the conductor (heat source)
        else:
            r_mid = 0.5 * (
                float(getattr(layer, "r_inner_m", 0.0) or 0.0)
                + float(getattr(layer, "r_outer_m", 0.0) or 0.0)
            )
            for node, w in interpolation(r_mid):
                column[node] += w
            notes.append(
                f"layer {layer.name!r} is not a conduction annulus; its "
                "temperature is interpolated at its mid radius"
            )
        names.append(layer.name)
        columns.append(column)

    rho = get_property(conductor_material, "rho_ohm_m")
    alpha_r = get_property(conductor_material, "alpha_R_per_K")
    armour_columns = [
        columns[names.index(layer.name)]
        for layer in cable.helical_layers
        if layer.structural and layer.r_helix_m and layer.name in names
    ]
    if lambda2 > 0.0 and not armour_columns:
        raise CableInputError("lambda2 > 0 needs a structural armour layer")
    armour_source = np.zeros(len(columns[0]))
    for column in armour_columns:
        armour_source += column / float(column.sum()) / len(armour_columns)
    limitations = [
        "IEC 60287 ladder as a lumped RC chain; external T4 has no heat capacity",
        "sheath eddy loss lambda1'' not included ([IEC60287-1-1] full text not "
        "held); lambda1 and the armour loss factor lambda2 as supplied",
        "three-core geometric factor [IEC60287-2-1] not applied "
        "(filler as equal-filler-area annulus)",
        *geo.geometry_notes,
        *notes,
    ]
    if passive > 0:
        limitations.append(
            f"{passive} of {geo.n_geo} cores without current: their heat capacity "
            "lumped at the first cable-level node"
        )
    prov = Provenance(
        method="transient IEC 60287 ladder (lumped radial RC network, TR-BDF2)",
        citations=(
            "[IEC60287-1-1] 1.4.1.1 Eq.1, 2.1",
            "[IEC60287-2-1]",
            "[NeherMcGrath1957] Eqs.1-2",
            "[Poradowski2025] Eq.2",
        ),
        gates=(
            "steady state = compute_thermal at fixed R_ac temperature (round-off)",
            "Crank-Nicolson discrete energy balance (round-off)",
        ),
        limitations=tuple(limitations),
    )
    return ThermalNetwork(
        radius_m=r_arr,
        per_core=pc_arr,
        conductance_W_mK=np.asarray(conductance, dtype=np.float64),
        capacitance_J_mK=cap,
        external_resistance_Km_W=float(t4),
        sheath_node=sheath_node,
        n_conductors=n,
        r_dc20_ohm_m=rho / conductor_area_m2,
        alpha_R_per_K=alpha_r,
        y_s=float(y_s),
        y_p=float(y_p),
        conductor_material=conductor_material,
        conductor_area_m2=float(conductor_area_m2),
        dielectric_loss_W_m=_cable_dielectric_loss(cable),
        lambda1=float(lambda1),
        ambient_C=float(t_amb),
        layer_weights=np.column_stack(columns),
        layer_names=tuple(names),
        provenance=prov,
        lambda2=float(lambda2),
        armour_source=armour_source,
        passive_cores=max(passive, 0),
    )
