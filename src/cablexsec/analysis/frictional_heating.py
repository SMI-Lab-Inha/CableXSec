# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Frictional heating of a cable under cyclic bending, coupled with its temperature.

Two time scales. Bending cycles take seconds; the cable heats over hours.
On the fast scale one bending cycle at the current thermal state gives the
friction work per cycle ``W_j`` of every component ``j`` that releases heat:
each structural armour layer (slip-front law of
:func:`~cablexsec.mechanics.papailiou.compute_mkappa_wire_slip_front`, one
layer at a time, which is exact because the wires of the law are independent
sliders) and the laid-core bundle (core and internal-ring friction of
:mod:`cablexsec.mechanics.core_friction`). The clamping, caps and stick
stiffness are those of the section solved at the layer temperatures
(:func:`~cablexsec.analysis.thermal_coupling.thermal_section`: free thermal
expansion and temperature-dependent moduli in the solved radial state). The
heat released per metre of cable is

.. math::

    q_j = a_j \, f \, W_j,

with the bending frequency ``f`` and an anchoring factor ``a_j`` (one for
the sectional value far from clamps and terminations; the curvature
gradients and anchored armour of a test specimen or a termination raise it,
:func:`anchored_heat_profile`).

On the slow scale the heat enters the radial network of
:mod:`cablexsec.mechanics.thermal_network` at the nodes of the layer where
it is released (``ThermalNetwork.source_weights``):
an armour layer's heat at that layer, the core-bundle heat at the outermost
layer of the cores. The conductor is either driven by a current (Joule loss
with ``R_ac(T_c)``, as in service) or held at a prescribed temperature (a
heated rig whose induction heating is controlled on the conductor
temperature); the heat the conductor then needs is reported.

Coupling (``coupling``):

* ``"two_way"``: staggered. At the start of every update interval the node
  temperatures give a thermal state, the section is solved at it and the heat
  ``q_j`` is held over the interval. Heating raises the clamping through
  thermal expansion (and lowers it through softening where a polymer has a
  registered ``E(T)`` law), the clamping sets the friction work, the friction
  work heats the cable.
* ``"one_way"``: the heat is that of the reference state (every layer at
  ``T_ref``) throughout: frictional heating without its feedback.
* ``"none"``: no frictional heat (the electro-thermal analysis alone).

Time integration: the L-stable second-order TR-BDF2 scheme of
:meth:`~cablexsec.mechanics.thermal_network.ThermalNetwork.transient`, with
the conductor node eliminated when its temperature is prescribed. The
staggered two-way result converges as the update interval shrinks
(tests/test_frictional_heating.py).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from cablexsec.analysis.thermal_coupling import ThermalState, thermal_section
from cablexsec.api import compute_mkappa
from cablexsec.cable_adapters import structural_armour_layers
from cablexsec.cables.schema import CableData, CylindricalOffsetLayer
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.clamp_layer import anchored_wire_node_dissipation
from cablexsec.mechanics.papailiou import compute_mkappa_wire_slip_front
from cablexsec.mechanics.thermal_network import ThermalNetwork

__all__ = [
    "CORE_BUNDLE",
    "AnchoredHeat",
    "COUPLINGS",
    "CyclicBending",
    "FrictionHeat",
    "FrictionHeatingResult",
    "anchored_heat_profile",
    "friction_heat",
    "simulate_frictional_heating",
]

FloatArray = NDArray[np.float64]

#: Name of the heat source of the laid-core bundle.
CORE_BUNDLE = "core bundle"
#: Coupling modes of :func:`simulate_frictional_heating`.
COUPLINGS = ("two_way", "one_way", "none")
_GAMMA = 2.0 - math.sqrt(2.0)


@dataclass(frozen=True)
class CyclicBending:
    """Cyclic bending on the fast time scale.

    The curvature is ``mean - amplitude cos(2 pi f t)``: one-sided bending
    from straight onto a former has ``mean = amplitude``, reversed bending
    about a straight cable ``mean = 0``. The history starts straight, ramps
    to the start of the cycle over a quarter cycle when that start is not
    zero, and runs ``cycles`` cycles; the work of the last cycle is the
    steady work per cycle.

    Attributes
    ----------
    tension_N : float
        True wall tension [N].
    amplitude_1_m, mean_1_m : float
        Curvature amplitude and mean [1/m].
    frequency_Hz : float
        Bending cycles per second.
    cycles, points_per_cycle : int
        Length and resolution of the fast history.
    mu : float or None
        Armour friction coefficient (``None``: the material value).
    core_friction : bool
        Include the laid-core bundle (cables with a core lay only).
    anchoring_factor : mapping
        ``{component: a_j}``, heat over its sectional value (default one).
    """

    tension_N: float
    amplitude_1_m: float
    frequency_Hz: float
    mean_1_m: float = 0.0
    cycles: int = 3
    points_per_cycle: int = 160
    mu: float | None = None
    core_friction: bool = True
    anchoring_factor: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("tension_N", "amplitude_1_m", "frequency_Hz", "mean_1_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise CableInputError(f"{name} must be finite")
        if self.tension_N < 0:
            raise CableInputError("tension_N must be non-negative")
        if self.amplitude_1_m <= 0 or self.frequency_Hz <= 0:
            raise CableInputError("amplitude and frequency must be positive")
        if self.cycles < 2 or self.points_per_cycle < 16:
            raise CableInputError("need at least 2 cycles of 16 points")
        factors = {str(k): float(v) for k, v in self.anchoring_factor.items()}
        if any(not math.isfinite(v) or v < 0 for v in factors.values()):
            raise CableInputError("anchoring factors must be finite and non-negative")
        object.__setattr__(self, "anchoring_factor", MappingProxyType(factors))

    def history(self) -> tuple[FloatArray, int]:
        """The fast curvature history and the index where its last cycle starts."""
        n = self.points_per_cycle
        start = self.mean_1_m - self.amplitude_1_m
        ramp = (
            np.linspace(0.0, start, n // 4 + 1)[:-1]
            if start != 0.0
            else np.empty(0, dtype=np.float64)
        )
        phase = np.linspace(0.0, self.cycles, self.cycles * n + 1)
        cycles = self.mean_1_m - self.amplitude_1_m * np.cos(2 * math.pi * phase)
        kappa = np.concatenate([ramp, cycles])
        return np.asarray(kappa, dtype=np.float64), int(kappa.size - 1 - n)


@dataclass(frozen=True)
class FrictionHeat:
    """Friction of one bending cycle at one thermal state.

    ``work_J_m`` is the sectional friction work per cycle and metre of every
    component, ``heat_W_m`` the heat released (anchoring factor times
    frequency times work) and ``contact_pressure_Pa`` the friction-setting
    pressure ``P1+P2`` of every armour layer.
    """

    work_J_m: Mapping[str, float]
    heat_W_m: Mapping[str, float]
    contact_pressure_Pa: Mapping[str, float]

    @property
    def total_heat_W_m(self) -> float:
        return float(sum(self.heat_W_m.values()))


def friction_heat(
    cable: CableData,
    bending: CyclicBending,
    state: ThermalState | None = None,
    *,
    T_ref_C: float = 20.0,
) -> FrictionHeat:
    """Friction work and heat of one bending cycle at a thermal state.

    ``state=None`` is the reference state: every layer at ``T_ref_C``.
    """
    if state is None:
        state = ThermalState(
            {layer.name: T_ref_C for layer in cable.layers}, T_ref_C=T_ref_C
        )
    section = thermal_section(cable, state, tension_N=bending.tension_N, mu=bending.mu)
    kappa, last = bending.history()
    work: dict[str, float] = {}
    pressure: dict[str, float] = {}
    for layer, spec in zip(structural_armour_layers(cable), section.specs, strict=True):
        cumulative = compute_mkappa_wire_slip_front([spec], kappa).work.friction_J_per_m
        work[layer.name] = float(cumulative[-1] - cumulative[last])
        pressure[layer.name] = float(spec.contact_pressure_Pa)
    if bending.core_friction and cable.core_lay is not None:
        result = compute_mkappa(
            cable,
            kappa,
            mu=bending.mu,
            radial_state=section.radial_state,
            core_friction=True,
        )
        if result.core_friction is None:  # pragma: no cover - guarded by core_lay
            raise CableInputError("core friction was requested but not computed")
        cumulative = result.core_friction.work.friction_J_per_m
        work[CORE_BUNDLE] = float(cumulative[-1] - cumulative[last])
    heat = {
        name: bending.anchoring_factor.get(name, 1.0) * bending.frequency_Hz * value
        for name, value in work.items()
    }
    return FrictionHeat(
        MappingProxyType(work), MappingProxyType(heat), MappingProxyType(pressure)
    )


@dataclass(frozen=True)
class AnchoredHeat:
    """Friction heat along a specimen anchored at both ends, over its sectional value.

    ``ratio[j]`` is the dissipation per cycle at ``x_m[j]`` (nodes between
    the clamps) over that of the same wires far from any anchor under the
    curvature history of the reference station.
    """

    x_m: FloatArray
    ratio: FloatArray
    pitch_m: float
    slip_onset_curvature_1_m: float

    def mean(self, lower_m: float, upper_m: float) -> float:
        """Mean ratio over ``lower_m <= x <= upper_m``."""
        inside = (self.x_m >= lower_m) & (self.x_m <= upper_m)
        if not inside.any():
            raise CableInputError("no node lies in the requested range")
        return float(self.ratio[inside].mean())

    def share_outside(self, lower_m: float, upper_m: float) -> float:
        """Share of the specimen's heat released outside ``[lower_m, upper_m]``."""
        inside = (self.x_m >= lower_m) & (self.x_m <= upper_m)
        total = float(self.ratio.sum())
        return float(self.ratio[~inside].sum() / total) if total > 0 else math.nan


def anchored_heat_profile(
    pitch_m: float,
    slip_onset_curvature_1_m: float,
    length_m: float,
    curvature: Callable[[FloatArray], FloatArray] | FloatArray,
    last_cycle_start: int,
    *,
    reference_x_m: float | None = None,
    phases: int = 16,
    segments_per_pitch: int = 96,
    far_field_pitches: int = 10,
) -> AnchoredHeat:
    """Friction heat along one armour layer of a specimen anchored at both clamps.

    Every wire of the layer is followed along the specimen with the anchored
    solver of :mod:`cablexsec.mechanics.clamp_layer` (rigid stick, Coulomb
    bound, both ends anchored), ``phases`` wires spread round the section;
    their node dissipation over the last cycle is averaged and divided by
    that of the same wires on a long, uniformly bent cable driven by the
    curvature history of ``reference_x_m`` (default: the station of largest
    curvature), whose middle pitches give the sectional value.

    ``curvature`` is either a function returning the curvature history at
    positions ``x`` (shape ``(samples, x.size)``) or such an array sampled at
    the segment midpoints of the specimen mesh (``segments_per_pitch``
    segments per pitch over ``length_m``). The last cycle starts at sample
    ``last_cycle_start``.
    """
    for name, value in (
        ("pitch_m", pitch_m),
        ("slip_onset_curvature_1_m", slip_onset_curvature_1_m),
        ("length_m", length_m),
    ):
        if not math.isfinite(value) or value <= 0:
            raise CableInputError(f"{name} must be positive")
    if phases < 1 or segments_per_pitch < 8 or far_field_pitches < 4:
        raise CableInputError("need phases >= 1, 8 segments per pitch, 4 far pitches")
    m = max(8, round(length_m / pitch_m * segments_per_pitch))
    x_mid = (np.arange(m) + 0.5) * length_m / m
    x_node = np.arange(1, m) * length_m / m
    field_values = (
        np.asarray(curvature(x_mid), dtype=np.float64)
        if callable(curvature)
        else np.asarray(curvature, dtype=np.float64)
    )
    if field_values.ndim != 2 or field_values.shape[1] != m:
        raise CableInputError(f"the curvature field must have {m} columns")
    if not 0 <= last_cycle_start < field_values.shape[0]:
        raise CableInputError("last_cycle_start must index a sample")
    column = (
        int(np.argmax(np.max(np.abs(field_values), axis=0)))
        if reference_x_m is None
        else int(np.argmin(np.abs(x_mid - reference_x_m)))
    )
    drive = field_values / slip_onset_curvature_1_m
    h = 2 * math.pi * (length_m / pitch_m) / m
    mf = far_field_pitches * segments_per_pitch
    hf = 2 * math.pi * far_field_pitches / mf
    xf = (np.arange(mf) + 0.5) * far_field_pitches * pitch_m / mf
    far_drive = np.repeat(drive[:, column : column + 1], mf, axis=1)
    near = np.zeros(m - 1)
    far = np.zeros(mf - 1)
    for p in range(phases):
        theta0 = 2 * math.pi * p / phases
        near += anchored_wire_node_dissipation(
            drive,
            np.cos(theta0 + 2 * math.pi * x_mid / pitch_m),
            h,
            from_sample=last_cycle_start,
        )
        far += anchored_wire_node_dissipation(
            far_drive,
            np.cos(theta0 + 2 * math.pi * xf / pitch_m),
            hf,
            from_sample=last_cycle_start,
        )
    near_density = near / (length_m / m)
    far_density = far / (far_field_pitches * pitch_m / mf)
    middle = slice(
        (far_field_pitches // 2 - 1) * segments_per_pitch,
        (far_field_pitches // 2 + 1) * segments_per_pitch,
    )
    reference = float(far_density[middle].mean())
    if reference <= 0.0:
        raise CableInputError("the reference station does not slip: no sectional heat")
    return AnchoredHeat(
        x_node,
        near_density / reference,
        float(pitch_m),
        float(slip_onset_curvature_1_m),
    )


@dataclass(frozen=True)
class FrictionHeatingResult:
    """Temperatures and friction heat of a coupled heating history.

    Arrays are given at ``time_s``. ``friction_heat_W_m[j][i]`` is the heat of
    component ``j`` over the interval that starts at ``time_s[i]`` (the last
    entry repeats the one before). ``conductor_heat_W_m[i]`` is the mean heat
    supplied to the conductors over the same interval: their Joule loss
    (current-driven) or the heater power the set point needs (heated rig).
    """

    time_s: FloatArray
    node_temperature_C: FloatArray
    layer_temperature_C: Mapping[str, FloatArray]
    friction_heat_W_m: Mapping[str, FloatArray]
    contact_pressure_Pa: Mapping[str, FloatArray]
    conductor_heat_W_m: FloatArray
    coupling: str
    updates: int
    network: ThermalNetwork

    @property
    def total_friction_heat_W_m(self) -> FloatArray:
        values = list(self.friction_heat_W_m.values())
        if not values:
            return np.zeros(self.time_s.size)
        return np.asarray(np.sum(values, axis=0), dtype=np.float64)


def _core_heat_layer(cable: CableData) -> str | None:
    cores = [
        layer for layer in cable.layers if isinstance(layer, CylindricalOffsetLayer)
    ]
    if not cores:
        return None
    return max(cores, key=lambda layer: float(layer.r_outer_m or 0.0)).name


def simulate_frictional_heating(
    cable: CableData,
    network: ThermalNetwork,
    bending: CyclicBending,
    time_s: Sequence[float] | FloatArray,
    *,
    current_A: Sequence[float] | FloatArray | float | None = None,
    conductor_temperature_C: Sequence[float] | FloatArray | float | None = None,
    bending_active: Sequence[bool] | NDArray[np.bool_] | None = None,
    ambient_C: float | None = None,
    initial: str | FloatArray = "ambient",
    coupling: str = "two_way",
    update_interval_s: float = 600.0,
    max_step_s: float = 60.0,
    T_ref_C: float | None = None,
    resistance_temperature_C: float | None = None,
    heat_layers: Mapping[str, str] | None = None,
    temperature_dependent_modulus: bool = True,
    thermal_expansion: bool = True,
    modulus_transfer: Mapping[str, str] | None = None,
    polymer_relaxation_factor: float = 1.0,
    heater_only: bool = True,
    progress: Callable[[float, str], None] | None = None,
) -> FrictionHeatingResult:
    """Heat a cable by its conductor and by the friction of cyclic bending.

    Give exactly one of ``current_A`` (linear between samples) and
    ``conductor_temperature_C`` (set point, linear between samples). With a
    set point and ``heater_only`` the conductor follows it while the heater
    power it needs is positive; when the friction heat alone keeps the
    conductor above it the heater is off and the conductor floats until it
    falls back to the set point (an induction-heated rig cannot cool).
    ``heater_only=False`` holds the set point exactly.
    ``bending_active[i]`` switches the bending on over the interval that
    starts at ``time_s[i]`` (default: always on). ``initial`` is
    ``"ambient"`` or an array of node temperatures. ``T_ref_C`` (stress-free
    reference) defaults to the ambient. ``heat_layers`` maps a heat source to
    the network layer that receives it (defaults in the module notes).
    ``progress`` is called with the share of the intervals done and a message
    after every interval of ``time_s`` (it may raise to stop the run).
    """
    if coupling not in COUPLINGS:
        raise CableInputError(f"coupling must be one of {COUPLINGS}")
    if (current_A is None) == (conductor_temperature_C is None):
        raise CableInputError(
            "give exactly one of current_A and conductor_temperature_C"
        )
    for name, value in (
        ("update_interval_s", update_interval_s),
        ("max_step_s", max_step_s),
    ):
        if not math.isfinite(value) or value <= 0:
            raise CableInputError(f"{name} must be positive")
    t = np.asarray(time_s, dtype=np.float64)
    if t.ndim != 1 or t.size < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise CableInputError("time_s must be strictly increasing, at least 2 samples")
    amb = network.ambient_C if ambient_C is None else float(ambient_C)
    t_ref = amb if T_ref_C is None else float(T_ref_C)
    active = (
        np.ones(t.size, dtype=bool)
        if bending_active is None
        else np.asarray(bending_active, dtype=bool)
    )
    if active.shape != t.shape:
        raise CableInputError("bending_active must match time_s")
    prescribed = conductor_temperature_C is not None
    driver = np.broadcast_to(
        np.asarray(
            conductor_temperature_C if prescribed else current_A, dtype=np.float64
        ),
        t.shape,
    ).copy()
    if not np.isfinite(driver).all():
        raise CableInputError("the conductor history must be finite")

    sources = {layer.name: layer.name for layer in structural_armour_layers(cable)}
    core_layer = _core_heat_layer(cable)
    if core_layer is not None:
        sources[CORE_BUNDLE] = core_layer
    sources.update(heat_layers or {})
    weights = {name: network.source_weights(layer) for name, layer in sources.items()}

    n = network.n_nodes
    c = network.capacitance_J_mK
    if isinstance(initial, str):
        if initial != "ambient":
            raise CableInputError('initial must be "ambient" or node temperatures')
        temp = np.full(n, amb)
    else:
        temp = np.asarray(initial, dtype=np.float64).copy()
        if temp.shape != (n,) or not np.isfinite(temp).all():
            raise CableInputError("initial node temperatures do not match the network")
    if prescribed:
        temp[0] = driver[0]
    free = np.arange(1, n) if prescribed else np.arange(n)

    def state_of(nodes: FloatArray) -> ThermalState:
        temps = network.core_mean_temperatures(nodes)
        return ThermalState(
            {name: float(value) for name, value in temps.items()},
            T_ref_C=t_ref,
            temperature_dependent_modulus=temperature_dependent_modulus,
            thermal_expansion=thermal_expansion,
            modulus_transfer=dict(modulus_transfer or {}),
            polymer_relaxation_factor=polymer_relaxation_factor,
            T_ref_basis="frictional heating: given or ambient",
        )

    reference: FrictionHeat | None = None
    if coupling != "none":
        reference = friction_heat(
            cable,
            bending,
            ThermalState(
                {layer.name: t_ref for layer in cable.layers},
                T_ref_C=t_ref,
                temperature_dependent_modulus=temperature_dependent_modulus,
                thermal_expansion=thermal_expansion,
                modulus_transfer=dict(modulus_transfer or {}),
                polymer_relaxation_factor=polymer_relaxation_factor,
            ),
        )
    names = list(weights)
    zero = FrictionHeat(
        MappingProxyType(dict.fromkeys(names, 0.0)),
        MappingProxyType(dict.fromkeys(names, 0.0)),
        MappingProxyType(
            dict.fromkeys(reference.contact_pressure_Pa if reference else {}, math.nan)
        ),
    )
    current_heat = zero
    last_update = -math.inf
    updates = 0

    def driver_at(time: float) -> float:
        return float(np.interp(time, t, driver))

    # Without current the system does not depend on time (heated rig).
    k_rig, s_rig = network.system_matrices(
        0.0, ambient_C=amb, resistance_temperature_C=20.0
    )
    every = np.arange(n)

    def full_system(time: float, q: FloatArray) -> tuple[FloatArray, FloatArray]:
        if prescribed:
            return k_rig, s_rig + q
        k, s = network.system_matrices(
            driver_at(time),
            ambient_C=amb,
            resistance_temperature_C=resistance_temperature_C,
        )
        return k, s + q

    def advance(
        nodes: FloatArray, t0: float, t1: float, q: FloatArray, held: bool
    ) -> FloatArray:
        """One TR-BDF2 step; ``held`` keeps the conductor on its set point."""
        idx = free if held else every

        def reduced(time: float) -> tuple[FloatArray, FloatArray]:
            k, s = full_system(time, q)
            if held:
                s = s - k[:, 0] * driver_at(time)
            return k[np.ix_(idx, idx)], s[idx]

        hh = t1 - t0
        hg = _GAMMA * hh
        cv = c[idx]
        x = nodes[idx]
        k_old, s_old = reduced(t0)
        k_g, s_g = reduced(t0 + hg)
        lhs = np.diag(cv / hg) + 0.5 * k_g
        rhs = (cv / hg) * x - 0.5 * (k_old @ x) + 0.5 * (s_old + s_g)
        x_g = np.linalg.solve(lhs, rhs)
        k_new, s_new = reduced(t1)
        w = (1.0 - _GAMMA) / (2.0 - _GAMMA)
        a1 = 1.0 / (_GAMMA * (2.0 - _GAMMA))
        a0 = (1.0 - _GAMMA) ** 2 / (_GAMMA * (2.0 - _GAMMA))
        lhs = np.diag(cv) + w * hh * k_new
        rhs = cv * (a1 * x_g - a0 * x) + w * hh * s_new
        out = nodes.copy()
        out[idx] = np.linalg.solve(lhs, rhs)
        if held:
            out[0] = driver_at(t1)
        return out

    def heater_power(
        old: FloatArray, new: FloatArray, q: FloatArray, hh: float
    ) -> float:
        """Heat the set point needs at the end of a held step [W/m]."""
        return float((k_rig @ new - s_rig - q)[0] + c[0] * (new[0] - old[0]) / hh)

    out_nodes = np.empty((t.size, n))
    out_nodes[0] = temp
    heat_out = {k: np.zeros(t.size) for k in names}
    pressure_out = {k: np.full(t.size, math.nan) for k in zero.contact_pressure_Pa}
    conductor_out = np.zeros(t.size)
    held = prescribed
    for i in range(t.size - 1):
        if active[i] and coupling != "none":
            if coupling == "one_way":
                current_heat = reference if reference is not None else zero
            elif t[i] - last_update >= update_interval_s - 1e-9 or current_heat is zero:
                current_heat = friction_heat(cable, bending, state_of(temp))
                last_update = float(t[i])
                updates += 1
        else:
            current_heat = zero
            last_update = -math.inf
        q = np.zeros(n)
        for name in names:
            q += current_heat.heat_W_m.get(name, 0.0) * weights[name]
        for name in names:
            heat_out[name][i] = current_heat.heat_W_m.get(name, 0.0)
        for name in pressure_out:
            pressure_out[name][i] = current_heat.contact_pressure_Pa.get(name, math.nan)
        steps = max(1, math.ceil((t[i + 1] - t[i]) / max_step_s - 1e-12))
        h = (t[i + 1] - t[i]) / steps
        energy = 0.0
        for j in range(steps):
            t0 = float(t[i] + j * h)
            t1 = float(t[i + 1]) if j == steps - 1 else t0 + h
            if not prescribed:
                new = advance(temp, t0, t1, q, False)
                energy += (
                    0.5
                    * (t1 - t0)
                    * network.n_conductors
                    * (
                        network.conductor_loss_W_m(
                            driver_at(t0),
                            float(temp[0]),
                            resistance_temperature_C=resistance_temperature_C,
                        )
                        + network.conductor_loss_W_m(
                            driver_at(t1),
                            float(new[0]),
                            resistance_temperature_C=resistance_temperature_C,
                        )
                    )
                )
            elif held:
                new = advance(temp, t0, t1, q, True)
                power = heater_power(temp, new, q, t1 - t0)
                if heater_only and power < 0.0:
                    new = advance(temp, t0, t1, q, False)
                    held = False
                else:
                    energy += power * (t1 - t0)
            else:
                new = advance(temp, t0, t1, q, False)
                if new[0] <= driver_at(t1):
                    held = True
            temp = new
        conductor_out[i] = energy / (t[i + 1] - t[i])
        out_nodes[i + 1] = temp
        if progress is not None:
            progress((i + 1) / (t.size - 1), f"Interval {i + 1} of {t.size - 1}")

    for name in names:
        heat_out[name][-1] = heat_out[name][-2]
    for name in pressure_out:
        pressure_out[name][-1] = pressure_out[name][-2]
    conductor_out[-1] = conductor_out[-2]
    layers = network.layer_temperatures(out_nodes)
    return FrictionHeatingResult(
        time_s=t,
        node_temperature_C=out_nodes,
        layer_temperature_C=MappingProxyType(dict(layers)),
        friction_heat_W_m=MappingProxyType(heat_out),
        contact_pressure_Pa=MappingProxyType(pressure_out),
        conductor_heat_W_m=conductor_out,
        coupling=coupling,
        updates=updates,
        network=network,
    )
