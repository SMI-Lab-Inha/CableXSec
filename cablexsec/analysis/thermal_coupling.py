# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Temperature histories coupled one way into clamping, friction and fatigue.

Implements the slow-time part of an electro-thermo-mechanical (ETM) model for
energised analyses:

``I(tau) -> T_i(tau) -> (eps_T,i, E_i(T)) -> q_j(tau), k_f(tau) -> sigma(t; tau)``.

Two time scales (ETM Pi1 = T_w / tau_th << 1). A :class:`ThermalHistory`
holds layer temperatures on the slow time ``tau`` (from a current history
through the transient IEC ladder of
:mod:`cablexsec.mechanics.thermal_network`, from a steady current, or as
supplied). It is cut into blocks; within a block the temperatures, moduli,
clamping and friction caps are frozen at the block-mean state
(:class:`ThermalState`) and the fast bending history ``kappa(t)`` is
evaluated with them. Between blocks the slow state changes; in a single
record the wire slip state is carried across block boundaries.

One-way coupling (ETM Pi2 = q_f / W << 1). Friction does not heat the cable;
the frictional work is reported as a diagnostic only.

What a temperature does to the mechanics (per block):

(a) **Clamping.** Free thermal strain ``alpha_i (T_i - T_ref)`` of every layer
    and the temperature-dependent polymer moduli enter the solved radial
    contact state (:func:`~cablexsec.mechanics.radial_cable.solve_cable_radial_state`,
    unilateral frictionless contact, residual manufacturing wire stress
    identified at ``T_ref``). Its ``P1+P2`` sets the friction cap
    ``sigma_0 g expm1(mu |sin a| psi)`` of every wire, and its crossing
    reaction the crossed-wire contact load.
(b) **Stiffness.** Polymer moduli follow the registered ``E(T)`` laws of the
    material library (``xlpe_qiao2025``, ``hdpe``: [Qiao2025] Eq. 9 after
    [Yan2021], 10--90 deg C, outside which the library refuses to
    extrapolate). The interface shear (stick) stiffness
    ``k_f = xi E_w cos^2 a`` follows through the Yun coefficient ``xi``
    ([Yun2020]) of the polymer layers inside each armour layer. Materials
    without a registered law keep their modulus unless a *relative transfer*
    of a registered law is requested explicitly (``modulus_transfer``; an
    assumption that is recorded). The steel wire modulus and the friction
    coefficient are constant (ETM Section 5: no defensible ``mu(T)``).
(c) **Fatigue.** Pressure, caps, stick stiffness and the thermal wire stress
    ``sigma_T = sigma_w(T, F) - sigma_w(T_ref, F)`` (solved wire stress at the
    same true wall tension) are constant within a block; the bending cycles
    of the block are rainflow counted with them.

Stress-free reference temperature ``T_ref``. Free thermal strains are
``alpha (T - T_ref)`` and the reference state (every layer at ``T_ref``) is
the unenergised analysis with the library moduli. :meth:`ThermalHistory.steady`
and :meth:`ThermalHistory.from_current` default ``T_ref`` to the ambient of
their thermal network (the cable installed and stress-free at its
surroundings), so the 0 A state is the unenergised reference.
:meth:`ThermalHistory.uniform` and :meth:`ThermalHistory.isothermal` (a
heated rig) default to 20 deg C, the laboratory ambient and the temperature
of the library moduli
(:data:`~cablexsec.materials.library.MODULUS_REFERENCE_TEMPERATURE_C`).
The choice matters: with the public 66 kV template at its 15 deg C ambient
and ``T_ref`` = 20 deg C, the 0 A state sits 5 K below the stress-free state,
the outer sheath shrinks onto the armour and the cold inner-armour clamping
rises from 31 to 52 kPa at 50 kN; the cold-to-rated damage factor is then
115 instead of 310 (doc/template_66kv.md Section 4.2). The basis is recorded
(``T_ref_basis``) and reported in the provenance.

Polymer relaxation. Moduli are time independent; hot polyethylene and XLPE
relax within hours to days under sustained stress. ``polymer_relaxation_factor``
``f`` (0 < f <= 1) multiplies the polymer moduli of every radial solve,
reference included (the sustained clamping and thermal stresses), but not the
stick stiffness of the fast bending. It is a relaxed-modulus proxy for a
bound, not a viscoelastic law; :func:`compute_thermal_fatigue` reports the
damage ratio at ``f`` and at ``f = 1`` (and at any further
``relaxation_factors``). Polymers without an ``E(T)`` law that run above
:data:`POLYMER_SOFTENING_WARNING_C` raise
:class:`~cablexsec.provenance.UnmodelledPolymerSofteningWarning`.

Within a block the thermal wire stress is not one number: ``sigma_T`` is the
solved wire stress at each sample's own tension minus the reference at that
tension, so it varies with tension through the block (by about a factor 1.5
in the damage of the 66 kV template at 50 +- 5 kN).

Damage ratios between thermal states scale with a power of the friction
stress range, so they depend steeply on the S-N slope (for the 66 kV
template, slopes 3 to 6 give 35 to 1300 for the reported 115). Every ratio is
reported with the S-N slope of the governing layer
(:attr:`ThermalFatigueResult.sn_slope`).

ETM hypotheses addressed: H1 (two time scales: structure and the ``Pi1``
diagnostic from the network time constant), H2 (one-way coupling: ``Pi2``
diagnostic from the frictional work), H3 (damage with ``T(tau)`` against a
fixed temperature: :func:`compute_thermal_fatigue` reports both). Not
implemented: H4 (hang-off: bend-stiffener and water-gap thermal resistance),
H5 (joint current and sea-state distribution), H6 (insulation ageing with
strain).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from cablexsec.cable_adapters import (
    cable_to_mkappa_specs,
    solved_contact_histories,
    structural_armour_layers,
)
from cablexsec.cables.schema import CableData, CylindricalOffsetLayer
from cablexsec.clamping import clamping_state
from cablexsec.exceptions import CableInputError
from cablexsec.materials import library
from cablexsec.materials.sn_curves import get_sn_curve
from cablexsec.mechanics.mkappa import HelicalLayerSpec
from cablexsec.mechanics.papailiou import slip_onset_curvature, wire_slip_history
from cablexsec.mechanics.radial_cable import (
    CableRadialState,
    solve_cable_radial_state,
    solve_cable_radial_states,
)
from cablexsec.mechanics.stiffness import xi_for_layer
from cablexsec.mechanics.thermal_network import ThermalNetwork, thermal_network
from cablexsec.provenance import (
    Provenance,
    UnmodelledPolymerSofteningWarning,
    real_array,
    require_finite,
)

if TYPE_CHECKING:
    from cablexsec.analysis.fatigue import FatigueLayer, FatigueResult

__all__ = [
    "POLYMER_SOFTENING_WARNING_C",
    "REFERENCE_TOLERANCE_K",
    "ArmourThermalState",
    "ThermalBlock",
    "ThermalBlockResult",
    "ThermalFatigueInputs",
    "ThermalFatigueResult",
    "ThermalHistory",
    "ThermalSection",
    "ThermalState",
    "armour_thermal_state",
    "compute_thermal_fatigue",
    "thermal_fatigue_inputs",
    "thermal_material_state",
    "thermal_provenance_notes",
    "thermal_section",
]

FloatArray = NDArray[np.float64]

#: Hypotheses of the ETM model and what this module does with them.
HYPOTHESIS_COVERAGE: Mapping[str, str] = MappingProxyType(
    {
        "H1": "implemented: slow-time blocks with frozen state; Pi1 diagnostic "
        "from the network time constant and the record period",
        "H2": "implemented as a diagnostic: one-way coupling; frictional heat "
        "of the record, Pi2 against the block heat and the armour temperature "
        "rise it would cause",
        "H3": "implemented: damage with T(tau) against the fixed time-mean "
        "temperature and the unenergised reference",
        "H4": "not implemented: bend-stiffener and water-gap thermal resistance "
        "at the hang-off are not in the network (T4 can only be overridden)",
        "H5": "not implemented: joint current and sea-state distribution in "
        "long-term damage (blocks can be weighted by the caller)",
        "H6": "not implemented: insulation ageing with mechanical strain",
    }
)


#: Layer temperature above which a polymer without an ``E(T)`` law warns
#: (deg C). PE and XLPE moduli fall by about half between 20 and 60 deg C
#: ([Qiao2025] Eq. 9), so a constant modulus overstates their thermal stress.
POLYMER_SOFTENING_WARNING_C = 40.0

#: Layer temperatures within this of ``T_ref`` are the reference state [K]
#: (round-off of the thermal network, not a physical temperature change).
REFERENCE_TOLERANCE_K = 1.0e-6

_LABORATORY_T_REF_C = library.MODULUS_REFERENCE_TEMPERATURE_C


def _relaxation_factor(value: float) -> float:
    require_finite(value, "polymer_relaxation_factor")
    if not 0.0 < value <= 1.0:
        raise CableInputError("polymer_relaxation_factor must lie in (0, 1]")
    return float(value)


def _frozen(values: FloatArray) -> FloatArray:
    out = np.array(values, dtype=np.float64, copy=True)
    out.flags.writeable = False
    return out


# --------------------------------------------------------------------------- #
# Thermal state and history                                                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ThermalState:
    """Frozen slow-time thermal state of one cable section.

    Attributes
    ----------
    layer_temperature_C : mapping
        Temperature of every cable layer [deg C].
    T_ref_C : float
        Stress-free reference temperature: free thermal strain is
        ``alpha (T - T_ref)`` and the state with every layer at ``T_ref`` is
        the unenergised analysis with the library moduli. Default 20 deg C
        here; the network builders of :class:`ThermalHistory` default it to
        the ambient.
    temperature_dependent_modulus : bool
        Apply registered polymer ``E(T)`` laws (and any transfer).
    thermal_expansion : bool
        Apply the free thermal strains (``False`` isolates softening).
    modulus_transfer : mapping
        ``{material: registered_law_material}``: scale the material's modulus
        by ``E_law(T) / E_law(T_ref)`` (explicit transfer assumption).
    conductor_temperature_C, total_heat_W_m : float or None
        Conductor temperature and heat generated in the cable, if known.
    label : str
        Free description.
    polymer_relaxation_factor : float
        Factor ``f`` in (0, 1] on the polymer moduli of every radial solve,
        the reference included (relaxed-modulus proxy for sustained
        clamping and thermal stress; the stick stiffness of the fast bending
        keeps the instantaneous modulus). 1 means no relaxation.
    T_ref_basis : str
        How ``T_ref_C`` was chosen (reported in the provenance).
    """

    layer_temperature_C: Mapping[str, float]
    T_ref_C: float = 20.0
    temperature_dependent_modulus: bool = True
    thermal_expansion: bool = True
    modulus_transfer: Mapping[str, str] = field(default_factory=dict)
    conductor_temperature_C: float | None = None
    total_heat_W_m: float | None = None
    label: str = ""
    polymer_relaxation_factor: float = 1.0
    T_ref_basis: str = ""

    def __post_init__(self) -> None:
        temps = {str(k): float(v) for k, v in self.layer_temperature_C.items()}
        if not temps or not all(math.isfinite(v) for v in temps.values()):
            raise CableInputError("layer temperatures must be finite and non-empty")
        require_finite(self.T_ref_C, "T_ref_C")
        object.__setattr__(
            self,
            "polymer_relaxation_factor",
            _relaxation_factor(self.polymer_relaxation_factor),
        )
        for material, law in self.modulus_transfer.items():
            if law not in library.TEMPERATURE_MODULUS_MODELS:
                raise CableInputError(
                    f"modulus_transfer[{material!r}] = {law!r} is not a registered "
                    "temperature-modulus law"
                )
        object.__setattr__(self, "layer_temperature_C", MappingProxyType(temps))
        object.__setattr__(
            self, "modulus_transfer", MappingProxyType(dict(self.modulus_transfer))
        )

    @property
    def is_reference(self) -> bool:
        """Every layer at ``T_ref``: the unenergised reference temperatures.

        Differences below :data:`REFERENCE_TOLERANCE_K` are network round-off
        (a steady 0 A state at the ambient ``T_ref`` solves to within about
        1e-9 K) and count as the reference. With a relaxation factor below
        one the reference temperatures still carry the relaxed polymer
        moduli (:func:`thermal_material_state`).
        """
        return all(
            abs(v - self.T_ref_C) <= REFERENCE_TOLERANCE_K
            for v in self.layer_temperature_C.values()
        )


@dataclass(frozen=True)
class ThermalBlock:
    """One slow-time block ``[start_s, end_s]`` and its block-mean state."""

    start_s: float
    end_s: float
    weight: float
    state: ThermalState


@dataclass(frozen=True)
class ThermalHistory:
    """Layer temperatures on the slow time scale (piecewise linear in time).

    Build it with :meth:`from_current` (transient IEC ladder),
    :meth:`steady` (one steady current), :meth:`uniform` (uniform specimen
    temperature), :meth:`isothermal` or directly from layer temperatures.
    A history with one time node is a single frozen state.

    ``T_ref_C`` is the stress-free reference temperature (module notes):
    the network builders default it to the network ambient, ``uniform`` and
    ``isothermal`` to 20 deg C; ``T_ref_basis`` records the choice.
    ``polymer_relaxation_factor`` is the relaxed-modulus proxy of
    :class:`ThermalState`.
    """

    time_s: FloatArray
    layer_temperature_C: Mapping[str, FloatArray]
    T_ref_C: float = 20.0
    temperature_dependent_modulus: bool = True
    thermal_expansion: bool = True
    modulus_transfer: Mapping[str, str] = field(default_factory=dict)
    current_A: FloatArray | None = None
    ambient_C: FloatArray | None = None
    conductor_temperature_C: FloatArray | None = None
    total_heat_W_m: FloatArray | None = None
    time_constant_s: float | None = None
    network: ThermalNetwork | None = field(default=None, compare=False, repr=False)
    source: str = "layer temperatures supplied"
    polymer_relaxation_factor: float = 1.0
    T_ref_basis: str = ""

    def __post_init__(self) -> None:
        t = real_array(self.time_s, "time_s").astype(np.float64)
        if t.ndim != 1 or t.size < 1 or not np.isfinite(t).all():
            raise CableInputError("time_s must be a finite 1-D vector")
        if np.any(np.diff(t) <= 0):
            raise CableInputError("time_s must be strictly increasing")
        object.__setattr__(self, "time_s", _frozen(t))
        temps: dict[str, FloatArray] = {}
        for name, values in self.layer_temperature_C.items():
            arr = real_array(values, f"layer_temperature_C[{name!r}]").astype(
                np.float64
            )
            if arr.ndim == 0:
                arr = np.full(t.size, float(arr))
            if arr.shape != t.shape or not np.isfinite(arr).all():
                raise CableInputError(
                    f"layer_temperature_C[{name!r}] must be finite and match time_s"
                )
            temps[str(name)] = _frozen(arr)
        if not temps:
            raise CableInputError("layer temperatures are required")
        object.__setattr__(self, "layer_temperature_C", MappingProxyType(temps))
        for name in (
            "current_A",
            "ambient_C",
            "conductor_temperature_C",
            "total_heat_W_m",
        ):
            value = getattr(self, name)
            if value is not None:
                arr = np.asarray(value, dtype=np.float64)
                if arr.shape != t.shape or not np.isfinite(arr).all():
                    raise CableInputError(f"{name} must be finite and match time_s")
                object.__setattr__(self, name, _frozen(arr))
        require_finite(self.T_ref_C, "T_ref_C")
        object.__setattr__(
            self,
            "polymer_relaxation_factor",
            _relaxation_factor(self.polymer_relaxation_factor),
        )
        object.__setattr__(
            self, "modulus_transfer", MappingProxyType(dict(self.modulus_transfer))
        )
        # Validate the options once through a state.
        self.state_at(float(t[0]))

    # ------------------------------------------------------------ builders
    @classmethod
    def from_current(
        cls,
        cable: CableData,
        time_s: Sequence[float] | FloatArray,
        current_A: Sequence[float] | FloatArray | float,
        *,
        conductor_area_m2: float | None = None,
        conductor_material: str | None = None,
        n_cores: int | None = None,
        y_s: float = 0.0,
        y_p: float = 0.0,
        lambda1: float = 0.0,
        ambient_C: Sequence[float] | FloatArray | float | None = None,
        T4_external_Km_W: float | None = None,
        resistance_temperature_C: float | None = None,
        initial: str | FloatArray = "steady",
        periodic: bool = False,
        max_step_s: float = 60.0,
        n_per_layer: int = 4,
        T_ref_C: float | None = None,
        temperature_dependent_modulus: bool = True,
        thermal_expansion: bool = True,
        modulus_transfer: Mapping[str, str] | None = None,
        polymer_relaxation_factor: float = 1.0,
    ) -> ThermalHistory:
        """Layer temperatures from a current history (transient IEC ladder).

        The conductor area and material default to the cable's conductor
        layer (``conductor_area_mm2``). Other arguments as
        :func:`~cablexsec.mechanics.thermal_network.thermal_network` and
        :meth:`~cablexsec.mechanics.thermal_network.ThermalNetwork.transient`;
        ``resistance_temperature_C=None`` solves ``R_ac(T_c)`` with the
        temperatures. ``T_ref_C=None`` takes the time-mean ambient as the
        stress-free reference (module notes).
        """
        area, material = _conductor_defaults(
            cable, conductor_area_m2, conductor_material
        )
        net = thermal_network(
            cable,
            area,
            n_cores=n_cores,
            conductor_material=material,
            y_s=y_s,
            y_p=y_p,
            lambda1=lambda1,
            T4_external_Km_W=T4_external_Km_W,
            n_per_layer=n_per_layer,
        )
        t = np.asarray(time_s, dtype=np.float64)
        cur = np.broadcast_to(np.asarray(current_A, dtype=np.float64), t.shape).copy()
        amb = np.broadcast_to(
            np.asarray(
                net.ambient_C if ambient_C is None else ambient_C, dtype=np.float64
            ),
            t.shape,
        ).copy()
        nodes = net.transient(
            t,
            cur,
            ambient_C=amb,
            initial=initial,
            resistance_temperature_C=resistance_temperature_C,
            max_step_s=max_step_s,
            periodic=periodic,
        )
        t_ref, t_ref_basis = _reference_temperature(
            T_ref_C, float(np.mean(amb)), _AMBIENT_BASIS
        )
        heat = np.array(
            [
                net.total_heat_W_m(
                    float(i),
                    float(n[0]),
                    resistance_temperature_C=resistance_temperature_C,
                )
                for i, n in zip(cur, nodes, strict=True)
            ]
        )
        return cls(
            time_s=t,
            layer_temperature_C=net.layer_temperatures(nodes),
            T_ref_C=t_ref,
            temperature_dependent_modulus=temperature_dependent_modulus,
            thermal_expansion=thermal_expansion,
            modulus_transfer=dict(modulus_transfer or {}),
            current_A=cur,
            ambient_C=amb,
            conductor_temperature_C=nodes[:, 0],
            total_heat_W_m=heat,
            time_constant_s=float(net.time_constants_s()[0]),
            network=net,
            source="transient IEC 60287 ladder from a current history",
            polymer_relaxation_factor=polymer_relaxation_factor,
            T_ref_basis=t_ref_basis,
        )

    @classmethod
    def steady(
        cls,
        cable: CableData,
        current_A: float,
        *,
        conductor_area_m2: float | None = None,
        conductor_material: str | None = None,
        n_cores: int | None = None,
        y_s: float = 0.0,
        y_p: float = 0.0,
        lambda1: float = 0.0,
        ambient_C: float | None = None,
        T4_external_Km_W: float | None = None,
        resistance_temperature_C: float | None = None,
        n_per_layer: int = 4,
        T_ref_C: float | None = None,
        temperature_dependent_modulus: bool = True,
        thermal_expansion: bool = True,
        modulus_transfer: Mapping[str, str] | None = None,
        polymer_relaxation_factor: float = 1.0,
    ) -> ThermalHistory:
        """Single frozen state: the steady layer temperatures at one current.

        ``T_ref_C=None`` takes the network ambient as the stress-free
        reference, so 0 A is the unenergised state (module notes).
        """
        area, material = _conductor_defaults(
            cable, conductor_area_m2, conductor_material
        )
        net = thermal_network(
            cable,
            area,
            n_cores=n_cores,
            conductor_material=material,
            y_s=y_s,
            y_p=y_p,
            lambda1=lambda1,
            T4_external_Km_W=T4_external_Km_W,
            ambient_C=ambient_C,
            n_per_layer=n_per_layer,
        )
        nodes = net.steady_state(
            current_A, resistance_temperature_C=resistance_temperature_C
        )
        temps = net.layer_temperatures(nodes[None, :])
        t_ref, t_ref_basis = _reference_temperature(
            T_ref_C, net.ambient_C, _AMBIENT_BASIS
        )
        return cls(
            time_s=np.array([0.0]),
            layer_temperature_C=temps,
            T_ref_C=t_ref,
            temperature_dependent_modulus=temperature_dependent_modulus,
            thermal_expansion=thermal_expansion,
            modulus_transfer=dict(modulus_transfer or {}),
            current_A=np.array([float(current_A)]),
            ambient_C=np.array([net.ambient_C]),
            conductor_temperature_C=np.array([nodes[0]]),
            total_heat_W_m=np.array(
                [
                    net.total_heat_W_m(
                        current_A,
                        float(nodes[0]),
                        resistance_temperature_C=resistance_temperature_C,
                    )
                ]
            ),
            time_constant_s=float(net.time_constants_s()[0]),
            network=net,
            source=f"steady IEC 60287 ladder at {float(current_A):g} A",
            polymer_relaxation_factor=polymer_relaxation_factor,
            T_ref_basis=t_ref_basis,
        )

    @classmethod
    def uniform(
        cls,
        cable: CableData,
        time_s: Sequence[float] | FloatArray,
        temperature_C: Sequence[float] | FloatArray | float,
        *,
        T_ref_C: float | None = None,
        temperature_dependent_modulus: bool = True,
        thermal_expansion: bool = True,
        modulus_transfer: Mapping[str, str] | None = None,
        polymer_relaxation_factor: float = 1.0,
    ) -> ThermalHistory:
        """Every layer at one (time-varying) temperature, e.g. a heated rig.

        ``T_ref_C=None`` takes 20 deg C, the laboratory ambient and the
        temperature of the library moduli.
        """
        t = np.atleast_1d(np.asarray(time_s, dtype=np.float64))
        temp = np.broadcast_to(np.asarray(temperature_C, dtype=np.float64), t.shape)
        t_ref, t_ref_basis = _reference_temperature(
            T_ref_C, _LABORATORY_T_REF_C, _LABORATORY_BASIS
        )
        return cls(
            time_s=t,
            layer_temperature_C={layer.name: temp.copy() for layer in cable.layers},
            T_ref_C=t_ref,
            temperature_dependent_modulus=temperature_dependent_modulus,
            thermal_expansion=thermal_expansion,
            modulus_transfer=dict(modulus_transfer or {}),
            source="uniform layer temperature",
            polymer_relaxation_factor=polymer_relaxation_factor,
            T_ref_basis=t_ref_basis,
        )

    @classmethod
    def isothermal(
        cls, cable: CableData, *, T_ref_C: float | None = None
    ) -> ThermalHistory:
        """Every layer at ``T_ref``: the unenergised reference (bitwise default)."""
        t_ref = _LABORATORY_T_REF_C if T_ref_C is None else float(T_ref_C)
        return cls.uniform(cable, [0.0], t_ref, T_ref_C=T_ref_C)

    # ------------------------------------------------------------ queries
    def _state(
        self,
        temps: Mapping[str, float],
        conductor: float | None,
        heat: float | None,
        label: str,
    ) -> ThermalState:
        return ThermalState(
            layer_temperature_C=temps,
            T_ref_C=self.T_ref_C,
            temperature_dependent_modulus=self.temperature_dependent_modulus,
            thermal_expansion=self.thermal_expansion,
            modulus_transfer=self.modulus_transfer,
            conductor_temperature_C=conductor,
            total_heat_W_m=heat,
            label=label,
            polymer_relaxation_factor=self.polymer_relaxation_factor,
            T_ref_basis=self.T_ref_basis,
        )

    def state_at(self, time_s: float) -> ThermalState:
        """Linearly interpolated state at one slow time."""
        t = self.time_s

        def at(values: FloatArray | None) -> float | None:
            return None if values is None else float(np.interp(time_s, t, values))

        temps = {
            k: float(np.interp(time_s, t, v))
            for k, v in self.layer_temperature_C.items()
        }
        return self._state(
            temps,
            at(self.conductor_temperature_C),
            at(self.total_heat_W_m),
            f"t = {time_s:g} s",
        )

    def mean_state(self) -> ThermalState:
        """Time-mean state over the whole history (the fixed-temperature case)."""
        if self.time_s.size == 1:
            return self.state_at(float(self.time_s[0]))
        a, b = float(self.time_s[0]), float(self.time_s[-1])
        return self._block_state(a, b, "time mean")

    def _block_state(self, a: float, b: float, label: str) -> ThermalState:
        t = self.time_s

        def mean(values: FloatArray | None) -> float | None:
            return None if values is None else _interval_mean(t, values, a, b)

        temps = {
            k: _interval_mean(t, v, a, b) for k, v in self.layer_temperature_C.items()
        }
        return self._state(
            temps, mean(self.conductor_temperature_C), mean(self.total_heat_W_m), label
        )

    def blocks(
        self, edges_s: Sequence[float] | FloatArray | None = None
    ) -> tuple[ThermalBlock, ...]:
        """Slow-time blocks with their block-mean states.

        ``edges_s`` default to the history's time nodes (one block per
        interval). The weight of a block is its share of the total duration.
        A single-node history is one block of weight one.
        """
        if self.time_s.size == 1 and edges_s is None:
            s0 = float(self.time_s[0])
            return (ThermalBlock(s0, s0, 1.0, self.state_at(s0)),)
        edges = (
            self.time_s if edges_s is None else np.asarray(edges_s, dtype=np.float64)
        )
        if (
            edges.ndim != 1
            or edges.size < 2
            or np.any(np.diff(edges) <= 0)
            or edges[0] < self.time_s[0]
            or edges[-1] > self.time_s[-1]
        ):
            raise CableInputError(
                "block edges must increase and lie within the history time span"
            )
        total = float(edges[-1] - edges[0])
        return tuple(
            ThermalBlock(
                float(a),
                float(b),
                float(b - a) / total,
                self._block_state(float(a), float(b), f"block {a:g}-{b:g} s"),
            )
            for a, b in zip(edges[:-1], edges[1:], strict=True)
        )

    def with_options(self, **changes: Any) -> ThermalHistory:
        """Copy with changed material options (e.g. ``thermal_expansion``)."""
        return replace(self, **changes)


_AMBIENT_BASIS = "ambient (default: stress-free at the surroundings)"
_LABORATORY_BASIS = "laboratory default (library modulus reference)"


def _reference_temperature(
    T_ref_C: float | None, default_C: float, default_basis: str
) -> tuple[float, str]:
    """Explicit stress-free reference temperature, or the builder's default."""
    if T_ref_C is None:
        return float(default_C), f"{default_basis}, {float(default_C):g} deg C"
    require_finite(T_ref_C, "T_ref_C")
    return float(T_ref_C), f"explicit, {float(T_ref_C):g} deg C"


def _interval_mean(t: FloatArray, y: FloatArray, a: float, b: float) -> float:
    """Mean of the piecewise-linear ``y(t)`` over ``[a, b]``."""
    if b <= a:
        return float(np.interp(a, t, y))
    inner = t[(t > a) & (t < b)]
    grid = np.concatenate([[a], inner, [b]])
    values = np.interp(grid, t, y)
    return float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(grid)) / (b - a))


def _conductor_defaults(
    cable: CableData, area_m2: float | None, material: str | None
) -> tuple[float, str]:
    conductor = next(
        (
            layer
            for layer in cable.layers
            if isinstance(layer, CylindricalOffsetLayer)
            and (layer.r_inner_m or 0.0) <= 0.0
        ),
        None,
    )
    if area_m2 is None:
        mm2 = getattr(conductor, "conductor_area_mm2", None)
        if mm2 is None:
            raise CableInputError(
                "conductor_area_m2 is required (the cable declares no "
                "conductor_area_mm2)"
            )
        area_m2 = float(mm2) * 1e-6
    if material is None:
        if conductor is None:
            raise CableInputError("conductor_material is required")
        material = conductor.material
    return float(area_m2), str(material)


# --------------------------------------------------------------------------- #
# Temperature -> material state                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ThermalMaterialState:
    """Inputs of the solved radial state and of ``xi`` for one thermal state.

    ``radial_inputs`` are the keyword inputs of the block's radial solve and
    ``reference_radial_inputs`` those of its reference (every layer at
    ``T_ref``): empty, the unenergised default, unless polymer relaxation is
    requested. ``xi_moduli_Pa`` (instantaneous moduli for the stick
    stiffness) never carry the relaxation.
    """

    is_reference: bool
    radial_inputs: Mapping[str, Any]
    xi_moduli_Pa: Mapping[str, float] | None
    modulus_sources: Mapping[str, str]
    reference_radial_inputs: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )


def _modulus_at(material: str, temperature_C: float, T_ref_C: float) -> float:
    """Registered ``E(T)`` of a material in the cable's reference frame.

    The library modulus is the modulus of the reference state. At the
    library's own 20 deg C the law is used directly (it reproduces ``E_Pa``
    there); at another ``T_ref`` it scales ``E_Pa`` by ``E(T) / E(T_ref)``,
    so the modulus is continuous at the reference state.
    """
    modulus = library.youngs_modulus_at_temperature(material, temperature_C)
    if T_ref_C == _LABORATORY_T_REF_C:
        return modulus
    return (
        library.get_property(material, "E_Pa")
        * modulus
        / library.youngs_modulus_at_temperature(material, T_ref_C)
    )


def _relaxed_reference_inputs(cable: CableData, state: ThermalState) -> dict[str, Any]:
    """Radial inputs of the relaxed reference state (empty without relaxation)."""
    f = state.polymer_relaxation_factor
    if f == 1.0:
        return {}
    moduli = {
        layer.name: f * library.get_property(layer.material, "E_Pa")
        for layer in cable.layers
        if library.is_polymer(layer.material)
    }
    if not moduli:
        return {}
    return {
        "T_ref_C": state.T_ref_C,
        "temperature_dependent_modulus": False,
        "youngs_modulus_Pa": moduli,
    }


def thermal_material_state(
    cable: CableData, state: ThermalState
) -> ThermalMaterialState:
    """Map a thermal state on the material inputs of the mechanics.

    The reference state (every layer at ``T_ref``) maps on the unchanged
    default inputs, so results are those of the unenergised analysis bit for
    bit; with ``polymer_relaxation_factor`` below one it maps on the relaxed
    reference instead. Otherwise every cable layer must have a temperature.
    A polymer layer without an ``E(T)`` law (or transfer) above
    :data:`POLYMER_SOFTENING_WARNING_C` raises
    :class:`~cablexsec.provenance.UnmodelledPolymerSofteningWarning`.
    """
    reference_inputs = _relaxed_reference_inputs(cable, state)
    if state.is_reference:
        return ThermalMaterialState(
            True,
            MappingProxyType(dict(reference_inputs)),
            None,
            MappingProxyType({}),
            MappingProxyType(reference_inputs),
        )
    names = {layer.name for layer in cable.layers}
    temps = state.layer_temperature_C
    missing = names - set(temps)
    if missing:
        raise CableInputError(
            f"thermal state has no temperature for layers {sorted(missing)}"
        )
    changed: dict[str, float] = {}
    sources: dict[str, str] = {}
    if state.temperature_dependent_modulus:
        for layer in cable.layers:
            t = temps[layer.name]
            material = layer.material
            if library.has_temperature_modulus_model(material):
                changed[layer.name] = _modulus_at(material, t, state.T_ref_C)
                sources[layer.name] = library.get_temperature_modulus_citation(material)
            elif material in state.modulus_transfer:
                law = state.modulus_transfer[material]
                ratio = library.law_modulus_at_temperature(
                    law, t
                ) / library.law_modulus_at_temperature(law, state.T_ref_C)
                changed[layer.name] = library.get_property(material, "E_Pa") * ratio
                sources[layer.name] = (
                    f"relative {law} law (ASSUMED transfer): "
                    + library.get_temperature_law_citation(law)
                )
    constant = sorted(
        layer.name
        for layer in cable.layers
        if library.is_polymer(layer.material)
        and layer.name not in changed
        and temps[layer.name] > POLYMER_SOFTENING_WARNING_C
    )
    if constant:
        warnings.warn(
            f"polymer layers {', '.join(constant)} run above "
            f"{POLYMER_SOFTENING_WARNING_C:g} deg C with a temperature-independent "
            "modulus (no registered E(T) law or modulus_transfer"
            + ("" if state.temperature_dependent_modulus else ", E(T) switched off")
            + "); softening and relaxation, which lower their share of the "
            "thermal clamping and wire stress, are not modelled (see "
            "modulus_transfer and polymer_relaxation_factor)",
            UnmodelledPolymerSofteningWarning,
            stacklevel=2,
        )
    radial: dict[str, Any] = {
        "T_ref_C": state.T_ref_C,
        "temperature_dependent_modulus": False,
    }
    if state.thermal_expansion:
        radial["temperature_C"] = dict(temps)
    relaxed = dict(changed)
    if reference_inputs:
        f = state.polymer_relaxation_factor
        for layer in cable.layers:
            if library.is_polymer(layer.material):
                relaxed[layer.name] = f * changed.get(
                    layer.name, library.get_property(layer.material, "E_Pa")
                )
    if relaxed:
        radial["youngs_modulus_Pa"] = relaxed
    if "temperature_C" not in radial and "youngs_modulus_Pa" not in radial:
        radial = {}
    xi_moduli: dict[str, float] | None = None
    if changed:
        xi_moduli = {}
        for layer in cable.layers:
            if layer.name in changed:
                xi_moduli[layer.name] = changed[layer.name]
                continue
            try:
                xi_moduli[layer.name] = library.get_property(layer.material, "E_Pa")
            except KeyError:
                continue
    return ThermalMaterialState(
        False,
        MappingProxyType(radial),
        None if xi_moduli is None else MappingProxyType(xi_moduli),
        MappingProxyType(sources),
        MappingProxyType(reference_inputs),
    )


def _stick_modulus(
    cable: CableData,
    name: str,
    reference_xi: float,
    e_wire: float,
    lay_angle: float,
    xi_moduli: Mapping[str, float] | None,
) -> float:
    xi = (
        reference_xi
        if xi_moduli is None
        else xi_for_layer(cable, name, youngs_modulus_Pa=xi_moduli)
    )
    return xi * e_wire * math.cos(lay_angle) ** 2


def _reference_state(
    cable: CableData, tension_N: float, material: ThermalMaterialState
) -> CableRadialState:
    """Solved reference state (every layer at ``T_ref``) at one tension."""
    if material.reference_radial_inputs:
        return solve_cable_radial_states(
            cable, (tension_N,), **material.reference_radial_inputs
        )[0]
    return solve_cable_radial_state(cable, tension_N)


# --------------------------------------------------------------------------- #
# Armour state of one block (diagnostics)                                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ArmourThermalState:
    """Clamping and friction scales of one armour layer at a thermal state.

    ``*_reference`` values are those of the unenergised reference state at
    the same tension. ``thermal_wire_stress_Pa`` is the solved wire-stress
    change at fixed tension, ``friction_cap_Pa`` the largest wire cap
    ``sigma_0 expm1(mu |sin a| pi/2)``.
    """

    layer: str
    temperature_C: float
    contact_pressure_Pa: float
    contact_pressure_reference_Pa: float
    stick_modulus_Pa: float
    stick_modulus_reference_Pa: float
    slip_onset_curvature_1_m: float
    slip_onset_curvature_reference_1_m: float
    friction_cap_Pa: float
    friction_cap_reference_Pa: float
    thermal_wire_stress_Pa: float

    @property
    def pi4(self) -> float:
        """ETM Pi4: relative thermal change of the clamping pressure."""
        ref = self.contact_pressure_reference_Pa
        return (self.contact_pressure_Pa - ref) / ref if ref > 0 else math.nan

    @property
    def pi5(self) -> float:
        """ETM Pi5: thermal wire stress over the friction cap."""
        cap = self.friction_cap_Pa
        return self.thermal_wire_stress_Pa / cap if cap > 0 else math.nan


def _cap_max(spec: Any) -> float:
    sin_a = abs(math.sin(spec.lay_angle_rad))
    if spec.contact_pressure_Pa <= 0 or sin_a < 1e-12:
        return 0.0
    width = 2 * math.pi * spec.r_helix_m / spec.n_wires
    sigma0 = (
        spec.r_helix_m
        * spec.contact_pressure_Pa
        * width
        * math.cos(spec.lay_angle_rad)
        / (spec.A_wire_m2 * sin_a**2)
    )
    return float(sigma0 * math.expm1(spec.mu * sin_a * math.pi / 2))


def armour_thermal_state(
    cable: CableData,
    state: ThermalState,
    *,
    tension_N: float = 0.0,
    mu: float | None = None,
) -> dict[str, ArmourThermalState]:
    """Armour clamping, stick stiffness and slip onset at one thermal state.

    The reference is the state with every layer at ``T_ref`` and the same
    tension (relaxed polymer moduli if ``polymer_relaxation_factor`` < 1).
    ``thermal_wire_stress_Pa`` is the wire-stress change at this tension
    only: within a block it varies with the tension of each sample.
    """
    section = thermal_section(cable, state, tension_N=tension_N, mu=mu)
    out: dict[str, ArmourThermalState] = {}
    for layer, ref_spec, spec_hot in zip(
        structural_armour_layers(cable),
        section.reference_specs,
        section.specs,
        strict=True,
    ):
        out[layer.name] = ArmourThermalState(
            layer=layer.name,
            temperature_C=float(
                state.layer_temperature_C.get(layer.name, state.T_ref_C)
            ),
            contact_pressure_Pa=float(spec_hot.contact_pressure_Pa),
            contact_pressure_reference_Pa=float(ref_spec.contact_pressure_Pa),
            stick_modulus_Pa=float(spec_hot.k_f),
            stick_modulus_reference_Pa=float(ref_spec.k_f),
            slip_onset_curvature_1_m=float(slip_onset_curvature(spec_hot)),
            slip_onset_curvature_reference_1_m=float(slip_onset_curvature(ref_spec)),
            friction_cap_Pa=_cap_max(spec_hot),
            friction_cap_reference_Pa=_cap_max(ref_spec),
            thermal_wire_stress_Pa=float(
                section.radial_state.result.wire_stress_Pa[layer.name]
                - section.reference_state.result.wire_stress_Pa[layer.name]
            ),
        )
    return out


@dataclass(frozen=True)
class ThermalSection:
    """The solved section of one thermal state at one tension.

    ``radial_state`` is solved with the layer temperatures (expansion and
    moduli of :func:`thermal_material_state`); ``reference_state`` with every
    layer at ``T_ref``. ``specs`` are the armour layers of ``radial_state``
    with the stick stiffness of the instantaneous moduli (no relaxation);
    ``reference_specs`` those of the reference.
    """

    radial_state: CableRadialState
    reference_state: CableRadialState
    specs: tuple[HelicalLayerSpec, ...]
    reference_specs: tuple[HelicalLayerSpec, ...]


def thermal_section(
    cable: CableData,
    state: ThermalState,
    *,
    tension_N: float = 0.0,
    mu: float | None = None,
) -> ThermalSection:
    """Solve the section of a thermal state: clamping and armour specifications.

    The single source of the clamping, stick stiffness and caps at a
    temperature for :func:`armour_thermal_state` and for the frictional
    heating of :mod:`cablexsec.analysis.frictional_heating`.
    """
    material = thermal_material_state(cable, state)
    reference = _reference_state(cable, tension_N, material)
    hot = (
        reference
        if material.is_reference or not material.radial_inputs
        else solve_cable_radial_states(cable, (tension_N,), **material.radial_inputs)[0]
    )
    ref_specs = cable_to_mkappa_specs(
        cable, tension_N, mu=mu, contact_model="solved", radial_state=reference
    )
    hot_specs = cable_to_mkappa_specs(
        cable, tension_N, mu=mu, contact_model="solved", radial_state=hot
    )
    specs: list[HelicalLayerSpec] = []
    for layer, hot_spec in zip(structural_armour_layers(cable), hot_specs, strict=True):
        k_hot = _stick_modulus(
            cable,
            layer.name,
            hot_spec.xi,
            hot_spec.E_wire_Pa,
            hot_spec.lay_angle_rad,
            material.xi_moduli_Pa,
        )
        specs.append(
            replace(
                hot_spec,
                xi=k_hot / (hot_spec.E_wire_Pa * math.cos(hot_spec.lay_angle_rad) ** 2),
            )
        )
    return ThermalSection(hot, reference, tuple(specs), tuple(ref_specs))


# --------------------------------------------------------------------------- #
# Fatigue inputs over one record                                              #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ThermalFatigueInputs:
    """Per-sample slow-time inputs of the fatigue pipeline for one record.

    ``None`` entries mean "unchanged default" (the reference state needs no
    history). ``block_index`` assigns every curvature sample to ``states``.
    """

    contact_pressure_Pa: dict[str, FloatArray] | None
    contact_force_N: dict[str, FloatArray] | None
    stick_modulus_Pa: dict[str, FloatArray] | None
    thermal_stress_Pa: dict[str, FloatArray] | None
    block_index: NDArray[np.int64]
    states: tuple[ThermalState, ...]
    notes: tuple[str, ...]


def _record_blocks(
    thermal: ThermalHistory | ThermalState,
    n_samples: int,
    time_s: FloatArray | None,
) -> tuple[tuple[ThermalState, ...], NDArray[np.int64]]:
    if isinstance(thermal, ThermalState):
        if time_s is not None:
            raise CableInputError("time_s is used only with a ThermalHistory")
        return (thermal,), np.zeros(n_samples, dtype=np.int64)
    if not isinstance(thermal, ThermalHistory):
        raise CableInputError("thermal must be a ThermalHistory or a ThermalState")
    if thermal.time_s.size == 1:
        return (thermal.state_at(float(thermal.time_s[0])),), np.zeros(
            n_samples, dtype=np.int64
        )
    if time_s is None:
        raise CableInputError(
            "a ThermalHistory with several time nodes needs the sample times "
            "time_s of the curvature record (or pass one ThermalState)"
        )
    t = real_array(time_s, "time_s").astype(np.float64)
    if t.shape != (n_samples,) or not np.isfinite(t).all() or np.any(np.diff(t) < 0):
        raise CableInputError("time_s must be finite, non-decreasing and match kappa")
    edges = thermal.time_s
    if t[0] < edges[0] or t[-1] > edges[-1]:
        raise CableInputError("the record lies outside the thermal history")
    index = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, edges.size - 2)
    used = np.unique(index)
    blocks = thermal.blocks()
    states = tuple(blocks[i].state for i in used)
    remap = np.zeros(edges.size - 1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    return states, remap[index].astype(np.int64)


def thermal_fatigue_inputs(
    cable: CableData,
    layers: Sequence[FatigueLayer],
    n_samples: int,
    tension_N: FloatArray | None,
    thermal: ThermalHistory | ThermalState,
    *,
    time_s: FloatArray | None = None,
    wire_thermal_stress: bool = True,
) -> ThermalFatigueInputs:
    """Clamping, stick stiffness and thermal stress per curvature sample.

    Each block state is solved over the same tension grid as the isothermal
    solved path (:func:`~cablexsec.cable_adapters.solved_contact_histories`);
    samples take the values of their block. A record entirely at the
    reference state returns the default inputs unchanged. The thermal wire
    stress of a sample is the solved wire stress at its own tension minus the
    reference at that tension, so it varies with tension within a block.
    """
    states, index = _record_blocks(thermal, n_samples, time_s)
    tension = (
        np.zeros(n_samples)
        if tension_N is None
        else np.asarray(tension_N, dtype=np.float64)
    )
    materials = [thermal_material_state(cable, s) for s in states]
    # A state whose temperatures change no mechanical input (for example no
    # expansion and no E(T) law) is mechanically the reference state.
    mechanically_reference = all(
        not m.radial_inputs and m.xi_moduli_Pa is None for m in materials
    )
    names = [layer.name for layer in layers]
    notes: list[str] = []
    if mechanically_reference:
        if tension_N is None:
            return ThermalFatigueInputs(None, None, None, None, index, states, ())
        hist = solved_contact_histories(cable, tension)
        return ThermalFatigueInputs(
            hist.pressure_Pa,
            hist.crossing_line_load_N_per_m,
            None,
            None,
            index,
            states,
            (),
        )
    reference_inputs = materials[0].reference_radial_inputs
    if any(m.reference_radial_inputs != reference_inputs for m in materials):
        raise CableInputError("the states of one record need one relaxation factor")
    reference_hist = solved_contact_histories(cable, tension, **reference_inputs)
    pressure = {n: np.empty(n_samples) for n in names}
    force = {
        n: np.empty(n_samples)
        for n in names
        if n in reference_hist.crossing_line_load_N_per_m
    }
    sigma_t = {n: np.zeros(n_samples) for n in names}
    stick = {n: np.empty(n_samples) for n in names}
    for b, (state, material) in enumerate(zip(states, materials, strict=True)):
        mask = index == b
        if not np.any(mask):
            continue
        hist = (
            reference_hist
            if material.is_reference or not material.radial_inputs
            else solved_contact_histories(cable, tension, **material.radial_inputs)
        )
        for layer in layers:
            n = layer.name
            pressure[n][mask] = hist.pressure_Pa[n][mask]
            if n in force:
                force[n][mask] = hist.crossing_line_load_N_per_m[n][mask]
            if wire_thermal_stress:
                sigma_t[n][mask] = (
                    hist.wire_stress_Pa[n][mask]
                    - reference_hist.wire_stress_Pa[n][mask]
                )
            spec = layer.spec
            stick[n][mask] = _stick_modulus(
                cable,
                n,
                spec.xi,
                spec.E_wire_Pa,
                spec.lay_angle_rad,
                material.xi_moduli_Pa,
            )
        if material.modulus_sources:
            notes.append(
                f"{state.label}: E(T) of " + ", ".join(sorted(material.modulus_sources))
            )
    stick_out: dict[str, FloatArray] | None = stick
    if all(np.all(stick[layer.name] == layer.spec.k_f) for layer in layers):
        stick_out = None
    thermal_out: dict[str, FloatArray] | None = sigma_t
    if not wire_thermal_stress or all(not np.any(v) for v in sigma_t.values()):
        thermal_out = None
    return ThermalFatigueInputs(
        pressure, force, stick_out, thermal_out, index, states, tuple(notes)
    )


def thermal_provenance_notes(inputs: ThermalFatigueInputs) -> tuple[str, ...]:
    """Provenance limitations describing the thermal coupling of a record."""
    temps = [
        (min(s.layer_temperature_C.values()), max(s.layer_temperature_C.values()))
        for s in inputs.states
    ]
    lo = min(t[0] for t in temps)
    hi = max(t[1] for t in temps)
    first = inputs.states[0]
    basis = f" ({first.T_ref_basis})" if first.T_ref_basis else ""
    f = first.polymer_relaxation_factor
    relaxation = (
        f"polymer moduli x {f:g} in every radial solve, reference included "
        "(relaxed-modulus proxy; the stick stiffness keeps the instantaneous "
        "modulus)"
        if f != 1.0
        else "polymer moduli time independent: no relaxation of the thermal "
        "polymer stresses (polymer_relaxation_factor gives a relaxed bound)"
    )
    return (
        f"thermal coupling (one way, ETM model): {len(inputs.states)} slow-time "
        f"block(s), layer temperatures {lo:.1f}-{hi:.1f} deg C, stress-free "
        f"reference T_ref {first.T_ref_C:g} deg C{basis}; clamping from the "
        "solved radial state with free thermal strain and E(T); friction "
        "coefficient constant",
        "within a block temperature, moduli, P1+P2, caps and stick stiffness "
        "are frozen; the thermal wire stress varies with the sample tension "
        "(solved at each tension minus the reference at that tension); slip "
        "state is carried across block boundaries",
        relaxation,
        *inputs.notes,
    )


# --------------------------------------------------------------------------- #
# Block-weighted fatigue                                                      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ThermalBlockResult:
    """Fatigue of the bending record evaluated at one slow-time block.

    ``frictional_heat_W_m`` is the armour frictional work rate of the record
    (all armour layers, at the block clamping and mean tension);
    ``pi2`` its ratio to the heat generated in the block (undefined, and
    large, as the current tends to zero); ``friction_temperature_rise_K`` the
    steady temperature rise it would cause at the armour through the network
    resistance to ambient, the quantity H2 compares with the measurement
    resolution of temperature.
    """

    block: ThermalBlock
    armour: Mapping[str, ArmourThermalState]
    fatigue: FatigueResult
    damage_max: float
    pi3: Mapping[str, float]
    pi2: float
    frictional_heat_W_m: float = math.nan
    friction_temperature_rise_K: float = math.nan


@dataclass(frozen=True)
class ThermalFatigueResult:
    """Block-weighted fatigue of one bending record under a thermal history.

    ``D_total_per_wire`` is ``sum_b w_b D_b`` per wire (the record repeated in
    every block with the block's thermal state, weighted by block duration):
    damage per record length of bending at the time-weighted thermal state.
    ``reference`` is the unenergised result (all layers at ``T_ref``) and
    ``fixed_mean`` the result at the time-mean layer temperatures (the
    fixed-temperature assumption tested by ETM H3). ``thermal_stress_range_Pa``
    is the range over the blocks of the thermal wire stress at the mean
    tension and ``D_thermal_cycle`` the damage of that one slow cycle per
    history period on the layer's bulk S-N curve (not combined with the
    bending extremes, so a lower bound of the slow-cycle contribution).

    Damage ratios depend steeply on the S-N slope and on the cold clamping;
    ``sn_slope`` holds the bulk S-N slope(s) of every armour layer and
    ``governing_layer`` the layer of ``damage_max``; :meth:`summary` prints
    every ratio with its slope. ``relaxation_damage_ratio`` maps polymer
    relaxation factors on ``damage_ratio_reference`` (the history's own factor,
    ``1`` when the history is relaxed, and any requested
    ``relaxation_factors``): the relaxation bounds of the ratio.
    ``clamping`` describes the declared manufacturing clamping.
    """

    blocks: tuple[ThermalBlockResult, ...]
    D_total_per_wire: dict[str, FloatArray]
    damage_max: float
    reference: FatigueResult
    fixed_mean: FatigueResult
    pi1: float
    hypotheses: Mapping[str, str]
    provenance: Provenance
    thermal_stress_range_Pa: Mapping[str, float] = field(default_factory=dict)
    D_thermal_cycle: Mapping[str, float] = field(default_factory=dict)
    sn_slope: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    governing_layer: str = ""
    T_ref_C: float = math.nan
    T_ref_basis: str = ""
    polymer_relaxation_factor: float = 1.0
    relaxation_damage_ratio: Mapping[float, float] = field(default_factory=dict)
    clamping: str = ""

    @property
    def damage_ratio_reference(self) -> float:
        ref = _max_damage(self.reference)
        return self.damage_max / ref if ref > 0 else math.nan

    @property
    def damage_ratio_fixed_mean(self) -> float:
        ref = _max_damage(self.fixed_mean)
        return self.damage_max / ref if ref > 0 else math.nan

    def slope_label(self) -> str:
        """S-N slope of the governing layer, e.g. ``"m = 4"``."""
        slopes = self.sn_slope.get(self.governing_layer, ())
        if not slopes:
            return "S-N slope not available"
        text = "/".join(f"{m:g}" for m in slopes)
        return f"bulk S-N slope m = {text} ({self.governing_layer})"

    def summary(self) -> str:
        """Damage ratios, each with the S-N slope it depends on."""
        slope = self.slope_label()
        lines = [
            f"thermal fatigue over {len(self.blocks)} block(s); stress-free "
            f"reference T_ref = {self.T_ref_C:g} deg C ({self.T_ref_basis})",
            f"manufacturing clamping: {self.clamping}",
            f"damage ratio to the unenergised reference: "
            f"{self.damage_ratio_reference:.4g} ({slope})",
            f"damage ratio to the fixed time-mean temperature: "
            f"{self.damage_ratio_fixed_mean:.4g} ({slope})",
        ]
        for factor, ratio in sorted(self.relaxation_damage_ratio.items()):
            lines.append(
                f"damage ratio to the reference, polymer relaxation factor "
                f"{factor:g}: {ratio:.4g} ({slope})"
            )
        lines.append(
            "ratios scale with a power of the friction stress range: a steeper "
            "S-N slope or a lower cold clamping raises them"
        )
        return "\n".join(lines)


def _max_damage(result: FatigueResult) -> float:
    return max(
        (float(np.max(v)) for v in result.D_total_per_wire.values()), default=0.0
    )


def _principal_range(kx: FloatArray, ky: FloatArray) -> tuple[float, FloatArray]:
    angles = np.linspace(0.0, math.pi, 73)[:-1]
    best, series = -1.0, kx
    for a in angles:
        s = kx * math.cos(a) + ky * math.sin(a)
        r = float(np.ptp(s))
        if r > best:
            best, series = r, s
    return best, series


def compute_thermal_fatigue(
    cable: CableData,
    kappa_x: FloatArray,
    kappa_y: FloatArray,
    thermal: ThermalHistory,
    *,
    tension_N: FloatArray | None = None,
    record_duration_s: float | None = None,
    edges_s: Sequence[float] | FloatArray | None = None,
    mu: float | None = None,
    relaxation_factors: Sequence[float] = (),
    progress: Callable[[float, str], None] | None = None,
    **fatigue_options: Any,
) -> ThermalFatigueResult:
    """Fatigue of a bending record repeated over the blocks of a thermal history.

    Every block of ``thermal`` (default: one per interval of its time nodes)
    evaluates the same fast record ``kappa_x, kappa_y`` (and tension) with the
    block's frozen thermal state through
    :func:`cablexsec.api.compute_fatigue` (``contact_model="solved"``). The
    block damages are weighted by block duration. The unenergised reference
    and the time-mean fixed-temperature results are returned for comparison
    (ETM H3). ``record_duration_s`` enables the ``Pi1`` and ``Pi2``
    diagnostics. ``fatigue_options`` are passed to ``compute_fatigue``.

    The reference is every layer at ``thermal.T_ref_C`` with the history's
    polymer relaxation factor. When that factor is below one the ratio is
    also evaluated unrelaxed, and ``relaxation_factors`` adds further factors
    (each a complete repeat of the analysis): together they bound the ratio
    in ``relaxation_damage_ratio``. Ratios are reported with the S-N slope
    (:meth:`ThermalFatigueResult.summary`). ``progress`` is called with the
    share of the fatigue runs done and a message after each block, the
    reference and the time-mean run (it may raise to stop the run).
    """
    from cablexsec.api import compute_fatigue  # noqa: PLC0415

    def report(done: int, total: int, message: str) -> None:
        if progress is not None:
            progress(done / total, message)

    if not isinstance(thermal, ThermalHistory):
        raise CableInputError("thermal must be a ThermalHistory")
    factors = [_relaxation_factor(float(f)) for f in relaxation_factors]
    for key in ("contact_model", "thermal", "time_s", "section_states"):
        if key in fatigue_options:
            raise CableInputError(f"{key!r} is set by compute_thermal_fatigue")
    kx = real_array(kappa_x, "kappa_x").astype(np.float64)
    ky = real_array(kappa_y, "kappa_y").astype(np.float64)
    if record_duration_s is not None:
        require_finite(record_duration_s, "record_duration_s")
        if record_duration_s <= 0:
            raise CableInputError("record_duration_s must be positive")
    tension = None if tension_N is None else np.asarray(tension_N, dtype=np.float64)
    t_mean = 0.0 if tension is None else float(np.mean(tension))

    def run(state: ThermalState) -> FatigueResult:
        return compute_fatigue(
            cable,
            kx,
            ky,
            tension_N=tension,
            mu=mu,
            contact_model="solved",
            thermal=state,
            **fatigue_options,
        )

    blocks = thermal.blocks(edges_s)
    runs = len(blocks) + 2
    d_range, principal = _principal_range(kx, ky)
    upcross = int(
        np.count_nonzero(
            (principal[:-1] - np.mean(principal) < 0)
            & (principal[1:] - np.mean(principal) >= 0)
        )
    )
    period = (
        record_duration_s / upcross
        if record_duration_s is not None and upcross > 0
        else math.nan
    )
    pi1 = (
        period / thermal.time_constant_s
        if thermal.time_constant_s and math.isfinite(period)
        else math.nan
    )
    reference_specs = cable_to_mkappa_specs(
        cable, t_mean, mu=mu, contact_model="solved"
    )
    results: list[ThermalBlockResult] = []
    weighted: dict[str, FloatArray] = {}
    for block in blocks:
        fat = run(block.state)
        armour = armour_thermal_state(cable, block.state, tension_N=t_mean, mu=mu)
        pi3 = {
            name: (d_range / a.slip_onset_curvature_1_m)
            if a.slip_onset_curvature_1_m > 0
            else math.inf
            for name, a in armour.items()
        }
        pi2 = q_f = rise = math.nan
        if record_duration_s is not None:
            q_f, rise = 0.0, 0.0
            for layer, spec in zip(
                structural_armour_layers(cable),
                reference_specs,
                strict=True,
            ):
                a = armour[layer.name]
                spec_b = replace(
                    spec,
                    contact_pressure_Pa=a.contact_pressure_Pa,
                    xi=a.stick_modulus_Pa
                    / (spec.E_wire_Pa * math.cos(spec.lay_angle_rad) ** 2),
                )
                hist = wire_slip_history(spec_b, kx, ky)
                q_layer = float(hist.work.friction_J_per_m[-1]) / record_duration_s
                q_f += q_layer
                if thermal.network is not None:
                    rise += q_layer * thermal.network.resistance_to_ambient_Km_W(
                        layer.name
                    )
                else:
                    rise = math.nan
            if block.state.total_heat_W_m:
                pi2 = q_f / float(block.state.total_heat_W_m)
        damage = _max_damage(fat)
        results.append(
            ThermalBlockResult(block, armour, fat, damage, pi3, pi2, q_f, rise)
        )
        report(len(results), runs, f"Block {len(results)} of {len(blocks)}")
        for name, values in fat.D_total_per_wire.items():
            weighted[name] = (
                weighted.get(name, np.zeros_like(values)) + block.weight * values
            )
    reference = run(
        ThermalState(
            {layer.name: thermal.T_ref_C for layer in cable.layers},
            T_ref_C=thermal.T_ref_C,
            polymer_relaxation_factor=thermal.polymer_relaxation_factor,
            T_ref_basis=thermal.T_ref_basis,
        )
    )
    report(runs - 1, runs, "Unenergised reference")
    fixed = run(thermal.mean_state())
    report(runs, runs, "Time-mean temperatures")
    damage_max = max((float(np.max(v)) for v in weighted.values()), default=0.0)
    stress_range: dict[str, float] = {}
    slow_damage: dict[str, float] = {}
    slopes: dict[str, tuple[float, ...]] = {}
    for layer in structural_armour_layers(cable):
        changes = [r.armour[layer.name].thermal_wire_stress_Pa for r in results]
        stress_range[layer.name] = float(max(changes) - min(changes))
        curve = get_sn_curve(layer.bulk_sn)
        slopes[layer.name] = (
            (curve.m1,) if curve.m2 is None else (curve.m1, float(curve.m2))
        )
        n_f = curve.cycles_to_failure(stress_range[layer.name] / 1e6)
        slow_damage[layer.name] = 1.0 / n_f if math.isfinite(n_f) and n_f > 0 else 0.0
    governing = max(
        weighted, key=lambda name: float(np.max(weighted[name])), default=""
    )
    reference_max = _max_damage(reference)
    own_ratio = damage_max / reference_max if reference_max > 0 else math.nan
    relaxation_ratio = {thermal.polymer_relaxation_factor: own_ratio}
    extra = set(factors) | (
        {1.0} if thermal.polymer_relaxation_factor != 1.0 else set()
    )
    for factor in sorted(extra - {thermal.polymer_relaxation_factor}):
        relaxation_ratio[factor] = compute_thermal_fatigue(
            cable,
            kx,
            ky,
            thermal.with_options(polymer_relaxation_factor=factor),
            tension_N=tension,
            edges_s=edges_s,
            mu=mu,
            **fatigue_options,
        ).damage_ratio_reference
    clamping = clamping_state(cable).describe()
    slope_text = "/".join(f"{m:g}" for m in slopes.get(governing, ()))
    prov = Provenance(
        method="slow-time thermal blocks (transient IEC ladder or supplied "
        "temperatures) coupled one way into the solved radial contact state, "
        "E(T), stick stiffness and per-wire fatigue; block damages weighted by "
        "duration",
        citations=(
            "[IEC60287-1-1]",
            "[Qiao2025] Eq. 9 / [Yan2021] (E(T))",
            "[Yun2020] (xi)",
            "[Papailiou1997]",
        ),
        gates=(
            "isothermal T_ref limit = unenergised result bit for bit",
            "two-layer thermal mismatch pressure against the Lame closed form",
            "network steady state = compute_thermal (BM5 inputs 136.1 degC)",
        ),
        limitations=(
            "one-way coupling; frictional heating only as the Pi2 diagnostic",
            "block-mean state frozen within a block; record repeated per block",
            "friction coefficient independent of temperature",
            "E(T) only for registered laws (10-90 degC) or explicit transfers",
            "lambda2, lambda1'' and the IEC 60287-2-1 three-core factor are not "
            "included in the thermal network (IEC full texts not held)",
            f"stress-free reference T_ref {thermal.T_ref_C:g} deg C"
            + (f" ({thermal.T_ref_basis})" if thermal.T_ref_basis else ""),
            f"damage ratios scale with a power of the friction stress range: "
            f"governing layer {governing}, bulk S-N slope m = {slope_text}; "
            "they also depend on the cold clamping",
            f"manufacturing clamping: {clamping}",
            "polymer moduli x "
            f"{thermal.polymer_relaxation_factor:g} in the radial solves "
            "(1 = no relaxation)",
            *(f"{k}: {v}" for k, v in HYPOTHESIS_COVERAGE.items()),
        ),
    )
    return ThermalFatigueResult(
        blocks=tuple(results),
        D_total_per_wire=weighted,
        damage_max=damage_max,
        reference=reference,
        fixed_mean=fixed,
        pi1=pi1,
        hypotheses=HYPOTHESIS_COVERAGE,
        provenance=prov,
        thermal_stress_range_Pa=stress_range,
        D_thermal_cycle=slow_damage,
        sn_slope=MappingProxyType(slopes),
        governing_layer=governing,
        T_ref_C=thermal.T_ref_C,
        T_ref_basis=thermal.T_ref_basis,
        polymer_relaxation_factor=thermal.polymer_relaxation_factor,
        relaxation_damage_ratio=MappingProxyType(relaxation_ratio),
        clamping=clamping,
    )
