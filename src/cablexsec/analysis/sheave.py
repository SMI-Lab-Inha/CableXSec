# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Cable bent over a sheave under tension (installation and handling).

A cable of diameter ``d`` pulled with tension ``S`` over a sheave of pitch
diameter ``D`` (centre of the sheave to the centre of the cable) takes the
curvature ``2/D`` on the contact bow. The groove carries the global line
pressure ([Feyrer2015] Eq. 3.27)

.. math::  q_0 = \frac{2S}{D}, \qquad p_0 = \frac{q_0}{d} = \frac{2S}{D\,d}

(Eq. 3.27a), distributed across a groove of angles ``gamma_1 .. gamma_2`` as
``k = k_0 cos gamma`` (Donandt; [Feyrer2015] Eqs. 3.30-3.31), with the peak
pressure and the groove friction factor

.. math::

    k_{max} = \frac{\cos\gamma_1}{\tfrac14\sin 2\gamma_2 + \tfrac12\gamma_2
    - \tfrac14\sin 2\gamma_1 - \tfrac12\gamma_1}\, p_0, \qquad
    \frac{f(\mu)}{\mu} = \frac{4(\sin\gamma_2 - \sin\gamma_1)}
    {\sin 2\gamma_2 - \sin 2\gamma_1 + 2\gamma_2 - 2\gamma_1}

(Eqs. 3.31, 3.32; a groove that is not undercut, ``gamma_1 = 0``,
``gamma_2 = 60 deg``, gives 1.351 and 1.170). Because the cable has bending
stiffness ``EI`` the contact begins at the run-on angle
``theta_0 = arctan((2/D) sqrt(EI/S))`` with a concentrated force
``Q = S sin theta_0`` (Eqs. 3.14-3.24; ``Q -> (2/D) sqrt(S EI)``). Measured
run-on peaks of wire ropes are higher than ``q_0`` (Häberle 1995 in
[Feyrer2015] Eq. 3.28 and Figs 3.18-3.23); those regressions are for steel
wire ropes and are not applied to a polymer-sheathed cable.

:func:`compute_sheave` adds the armour response with the package's section
models: wire stresses on the sheave (:func:`compute_stress`), the design
utilisation of the sheave curvature on the capacity envelope
(:func:`~cablexsec.analysis.capacity_check.build_capacity_envelope`, the same
design utilisation factor, bounded contact chain and mechanisms as the
capacity check of global histories), the characteristic curvature ratio
``kappa / kappa_max(S)`` of :func:`compute_capacity` (design factor one), and
the damage of one passage, the cable bending from straight onto the sheave and
back (:func:`compute_fatigue` on that planar history at constant tension). The
groove pressure is reported for the sheath and is not applied to the armour
contact (a one-sided pressure; see limitations).

References
----------
[Feyrer2015] K. Feyrer, Wire Ropes: Tension, Endurance, Reliability, 2nd edn,
Springer, 2015, Section 3.1.6-3.1.7.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from cablexsec.exceptions import CableInputError
from cablexsec.provenance import Provenance, relayed_warnings

if TYPE_CHECKING:
    from cablexsec.analysis.capacity_check import CapacityEnvelope
    from cablexsec.cables.schema import CableData

#: Largest tan(theta_0) for which the tape solution of [Feyrer2015] Eq. 3.24
#: (small run-on angle, tan ~ sin) is used without a flag.
SMALL_ANGLE_TAN = 0.1

__all__ = [
    "GOVERNING_MECHANISMS",
    "SMALL_ANGLE_TAN",
    "SheaveContact",
    "SheaveResult",
    "compute_sheave",
    "groove_pressure",
    "sheave_contact",
]


@dataclass(frozen=True)
class SheaveContact:
    """Run-on contact of a member of bending stiffness ``EI`` ([Feyrer2015] 3.1.6)."""

    run_on_angle_rad: float
    run_on_force_N: float
    tension_at_contact_N: float
    line_pressure_N_per_m: float
    lever_arm_m: float


def sheave_contact(
    tension_N: float, pitch_diameter_m: float, bending_stiffness_Nm2: float
) -> SheaveContact:
    """Run-on angle, contact force and line pressure over a sheave.

    Eqs. 3.14, 3.15, 3.23, 3.24 and 3.27 of [Feyrer2015] for a member that is
    rigid across its section, with ``tanh(omega x_0) = 1``.
    """
    for name, value in (
        ("tension_N", tension_N),
        ("pitch_diameter_m", pitch_diameter_m),
        ("bending_stiffness_Nm2", bending_stiffness_Nm2),
    ):
        if not (math.isfinite(value) and value > 0):
            raise CableInputError(f"{name} must be finite and positive")
    S, D, EJ = tension_N, pitch_diameter_m, bending_stiffness_Nm2
    theta0 = math.atan(2.0 / D * math.sqrt(EJ / S))
    return SheaveContact(
        run_on_angle_rad=theta0,
        run_on_force_N=S * math.sin(theta0),
        tension_at_contact_N=S * math.cos(theta0),
        line_pressure_N_per_m=2.0 * S / D,
        lever_arm_m=2.0 * EJ / (D * S),
    )


def groove_pressure(
    gamma1_rad: float = 0.0, gamma2_rad: float = math.radians(60.0)
) -> tuple[float, float]:
    """Peak pressure over ``p0`` and friction factor ``f(mu)/mu`` of a groove.

    Cosine pressure distribution between the groove angles (Eqs. 3.31, 3.32 of
    [Feyrer2015]); ``gamma1 > 0`` is an undercut groove.
    """
    g1, g2 = float(gamma1_rad), float(gamma2_rad)
    if not (0.0 <= g1 < g2 < math.pi / 2):
        raise CableInputError("groove angles need 0 <= gamma1 < gamma2 < 90 deg")
    denominator = (
        0.25 * math.sin(2 * g2) + 0.5 * g2 - 0.25 * math.sin(2 * g1) - 0.5 * g1
    )
    peak = math.cos(g1) / denominator
    friction = (
        4.0
        * (math.sin(g2) - math.sin(g1))
        / (math.sin(2 * g2) - math.sin(2 * g1) + 2 * g2 - 2 * g1)
    )
    return peak, friction


#: Governing mechanism of a design utilisation, by the index the capacity
#: envelope returns (:meth:`CapacityEnvelope.utilisation`).
GOVERNING_MECHANISMS = (
    "wire_stress",
    "lateral_buckling",
    "tension_mechanisms",
    "component_limits",
)


@dataclass(frozen=True)
class SheaveResult:
    """Cable over a sheave: curvature, groove loads and armour response.

    Two capacity measures are reported. ``design_utilisation`` is the
    utilisation of the sheave curvature on the capacity envelope, with the
    design utilisation factor ``utilisation_factor`` and the bounded contact
    chain, the basis of the capacity check of global histories; one is the
    design capacity ``design_curvature_capacity_1_m``.
    ``characteristic_utilisation`` is
    ``kappa / characteristic_curvature_capacity_1_m``, the curvature over the
    first-yield (characteristic, design factor one) curvature capacity of
    :func:`~cablexsec.analysis.capacity.compute_capacity`.
    ``governing_mechanism`` names the mechanism of the design utilisation
    (:data:`GOVERNING_MECHANISMS`); where it is a component limit,
    ``governing_component`` names the component and its limit (the
    conductor's 0.2 % proof stress, a polymer's allowable strain), as
    ``"layer: mode"``. Both capacity measures include the limits of the
    components other than the armour where their materials give them.
    """

    pitch_diameter_m: float
    cable_diameter_m: float
    tension_N: float
    curvature_1_m: float
    line_pressure_N_per_m: float
    global_pressure_Pa: float
    groove_peak_pressure_Pa: float
    groove_friction_factor: float
    run_on_angle_rad: tuple[float, float]
    run_on_force_N: tuple[float, float]
    run_on_small_angle: tuple[bool, bool]
    wire_stress_max_Pa: dict[str, float]
    wire_stress_min_Pa: dict[str, float]
    characteristic_curvature_capacity_1_m: float
    characteristic_utilisation: float
    damage_per_pass: dict[str, float]
    mbr_ratio: float | None
    design_utilisation: float = math.nan
    design_curvature_capacity_1_m: float = math.nan
    utilisation_factor: float = math.nan
    governing_mechanism: str = ""
    governing_component: str = ""
    contact_model: str = "bounded"
    provenance: Provenance = field(
        default_factory=lambda: Provenance(method="", citations=(), limitations=())
    )

    @property
    def passes_allowed(self) -> float:
        """Passages to a damage of one for the most damaged layer (no DFF)."""
        worst = max(self.damage_per_pass.values(), default=0.0)
        return math.inf if worst <= 0 else 1.0 / worst

    @property
    def utilisation(self) -> float:
        """Deprecated: :attr:`characteristic_utilisation` (design factor one)."""
        warnings.warn(
            "SheaveResult.utilisation is deprecated; use design_utilisation (the "
            "capacity envelope basis) or characteristic_utilisation",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.characteristic_utilisation

    @property
    def curvature_capacity_1_m(self) -> float:
        """Deprecated: :attr:`characteristic_curvature_capacity_1_m`."""
        warnings.warn(
            "SheaveResult.curvature_capacity_1_m is deprecated; use "
            "design_curvature_capacity_1_m or "
            "characteristic_curvature_capacity_1_m",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.characteristic_curvature_capacity_1_m


def _outer_diameter(cable: CableData) -> float:
    radius = 0.0
    for layer in cable.layers:
        outer = getattr(layer, "r_outer_m", None)
        offset = getattr(layer, "r_offset_m", None) or 0.0
        if outer is not None:
            radius = max(radius, float(outer) + float(offset))
        helix = getattr(layer, "r_helix_m", None)
        wire = getattr(layer, "d_wire_m", None)
        if helix is not None and wire is not None:
            radius = max(radius, float(helix) + 0.5 * float(wire))
    if radius <= 0:
        raise CableInputError("cable outer diameter could not be determined; pass it")
    return 2.0 * radius


def compute_sheave(
    cable: CableData,
    *,
    sheave_diameter_m: float,
    tension_N: float,
    diameter_basis: Literal["pitch", "groove_root"] = "pitch",
    cable_diameter_m: float | None = None,
    groove_angles_deg: tuple[float, float] = (0.0, 60.0),
    mu: float | None = None,
    contact_model: str = "bounded",
    n_ramp: int = 40,
    utilisation_factor: float | None = None,
    capacity_envelope: CapacityEnvelope | None = None,
) -> SheaveResult:
    """Cable pulled over a sheave: groove pressure, armour stress, capacity, damage.

    Parameters
    ----------
    cable : CableData
        A fully specified cable.
    sheave_diameter_m : float
        Sheave diameter [m], at the cable centre (``"pitch"``, as in
        [Feyrer2015]) or at the groove root (``"groove_root"``: the cable
        diameter is added).
    tension_N : float
        Cable tension over the sheave [N], positive.
    cable_diameter_m : float, optional
        Outer diameter of the cable; default: the outermost layer.
    groove_angles_deg : (float, float), optional
        Groove angles of the cosine pressure distribution (default 0 and 60
        deg, a groove that is not undercut).
    mu, contact_model
        Armour friction and contact model of the section analyses. The
        capacity envelope always uses the bounded contact chain.
    n_ramp : int, optional
        Steps of each ramp of the passage history (straight to the sheave and
        back).
    utilisation_factor : float, optional
        Design utilisation factor of the capacity envelope (default
        ``ARMOUR_UTILISATION_FACTORS["normal"]``, the default of
        :func:`~cablexsec.analysis.capacity_check.build_capacity_envelope`).
        Not used with ``capacity_envelope``, which carries its own.
    capacity_envelope : CapacityEnvelope, optional
        An envelope already built for this cable (for example the one of a
        capacity check); its tension grid must contain ``tension_N``.
        Default: an envelope built at ``tension_N`` with
        ``utilisation_factor`` and ``mu``.
    """
    from cablexsec.analysis.capacity import compute_capacity  # noqa: PLC0415
    from cablexsec.analysis.capacity_check import (  # noqa: PLC0415
        build_capacity_envelope,
    )
    from cablexsec.analysis.stress import compute_stress  # noqa: PLC0415
    from cablexsec.api import compute_fatigue, compute_mkappa  # noqa: PLC0415

    if not (math.isfinite(tension_N) and tension_N > 0):
        raise CableInputError("tension_N must be finite and positive")
    if not (math.isfinite(sheave_diameter_m) and sheave_diameter_m > 0):
        raise CableInputError("sheave_diameter_m must be finite and positive")
    if diameter_basis not in ("pitch", "groove_root"):
        raise CableInputError("diameter_basis must be 'pitch' or 'groove_root'")
    if n_ramp < 2:
        raise CableInputError("n_ramp must be at least 2")
    if capacity_envelope is not None and utilisation_factor is not None:
        raise CableInputError(
            "give a utilisation factor or a capacity envelope, not both"
        )
    d = _outer_diameter(cable) if cable_diameter_m is None else float(cable_diameter_m)
    if not (math.isfinite(d) and d > 0):
        raise CableInputError("cable_diameter_m must be finite and positive")
    D = sheave_diameter_m + (d if diameter_basis == "groove_root" else 0.0)
    if d >= D:
        raise CableInputError(
            "the sheave pitch diameter must exceed the cable diameter"
        )
    kappa = 2.0 / D
    peak_factor, friction_factor = groove_pressure(
        *(math.radians(a) for a in groove_angles_deg)
    )
    q0 = 2.0 * tension_N / D
    p0 = q0 / d

    with relayed_warnings():
        loop = compute_mkappa(
            cable,
            np.linspace(0.0, kappa, 50),
            tension_N=tension_N,
            mu=mu,
            contact_model=contact_model,
        )
        stress = compute_stress(
            cable, kappa, 0.0, tension_N=tension_N, mu=mu, contact_model=contact_model
        )
        capacity = compute_capacity(
            cable, np.array([tension_N]), mu=mu, contact_model=contact_model
        )
        envelope = (
            build_capacity_envelope(
                cable,
                np.array([tension_N]),
                utilisation_factor=utilisation_factor,
                mu=mu,
            )
            if capacity_envelope is None
            else capacity_envelope
        )
        design, governing, _counts = envelope.utilisation(
            np.array([tension_N]), np.array([kappa])
        )
        design_capacity = float(envelope.curvature_capacity(np.array([tension_N]))[0])
        ramp = np.linspace(0.0, kappa, n_ramp)
        history = np.concatenate([ramp, np.full(n_ramp // 2, kappa), ramp[::-1]])
        fatigue = compute_fatigue(
            cable,
            history,
            np.zeros_like(history),
            tension_N=np.full_like(history, tension_N),
            mu=mu,
            contact_model=contact_model,
        )
    contacts = [
        sheave_contact(tension_N, D, float(ei))
        for ei in (loop.EI_min_Nm2, loop.EI_max_Nm2)
    ]
    kappa_cap = float(capacity.kappa_max_1_m[0])
    mbr = cable.cable.MBR_installation_m
    component = ""
    if int(governing[0]) == 3 and envelope.component_governing:
        grid = np.asarray(envelope.tension_N, dtype=np.float64)
        layer, mode = envelope.component_governing[
            int(np.argmin(np.abs(grid - tension_N)))
        ]
        component = f"{layer}: {mode}" if layer else ""
    return SheaveResult(
        pitch_diameter_m=D,
        cable_diameter_m=d,
        tension_N=float(tension_N),
        curvature_1_m=kappa,
        line_pressure_N_per_m=q0,
        global_pressure_Pa=p0,
        groove_peak_pressure_Pa=peak_factor * p0,
        groove_friction_factor=friction_factor,
        run_on_angle_rad=(contacts[0].run_on_angle_rad, contacts[1].run_on_angle_rad),
        run_on_force_N=(contacts[0].run_on_force_N, contacts[1].run_on_force_N),
        run_on_small_angle=(
            math.tan(contacts[0].run_on_angle_rad) <= SMALL_ANGLE_TAN,
            math.tan(contacts[1].run_on_angle_rad) <= SMALL_ANGLE_TAN,
        ),
        wire_stress_max_Pa={
            k: float(np.max(v)) for k, v in stress.sigma_total_Pa.items()
        },
        wire_stress_min_Pa={
            k: float(np.min(v)) for k, v in stress.sigma_total_Pa.items()
        },
        characteristic_curvature_capacity_1_m=kappa_cap,
        characteristic_utilisation=kappa / kappa_cap if kappa_cap > 0 else math.inf,
        damage_per_pass={
            k: float(np.max(v)) for k, v in fatigue.D_total_per_wire.items()
        },
        mbr_ratio=None if mbr is None else 0.5 * D / mbr,
        design_utilisation=float(design[0]),
        design_curvature_capacity_1_m=design_capacity,
        utilisation_factor=float(envelope.utilisation_factor),
        governing_mechanism=GOVERNING_MECHANISMS[int(governing[0])],
        governing_component=component,
        contact_model=contact_model,
        provenance=Provenance(
            method=(
                "sheave curvature 2/D; groove line and mean pressure, cosine "
                "groove distribution and run-on contact ([Feyrer2015] 3.1.6-3.1.7); "
                "armour wire stress of the section analysis; design utilisation "
                "on the capacity envelope with the design utilisation factor "
                f"{envelope.utilisation_factor:.3g}; curvature over the "
                "first-yield curvature capacity; damage of one passage from the "
                "armour fatigue analysis"
            ),
            citations=(
                "[Feyrer2015] Eqs. 3.14-3.32",
                "[DNV-RP-F401] Sec. 3.8.1, Table 3-1, p. 13",
                "[MecLim2025] Table 5, p. 19",
            ),
            limitations=(
                "the groove pressure is reported for the sheath and is not "
                "applied to the armour contact (one-sided pressure, not resolved)",
                "the measured run-on pressure peaks of wire ropes (Häberle 1995) "
                "are not transferred to cables; the line load and the cosine "
                "groove peak are global values",
                "run-on force bracketed by the full-slip and no-slip bending "
                "stiffness of the section; the tape solution assumes a small run-on "
                "angle, which stiff cables on small sheaves at low tension do not "
                "satisfy (flagged)",
                "the capacity envelope uses the bounded contact chain whatever "
                "contact model the stresses and the damage use",
                "one passage: straight to the sheave and back at constant "
                "tension, in one plane; no sheave friction on the cable axis "
                "(tension change over the wrap) and no crushing of the section",
            ),
        ),
    )
