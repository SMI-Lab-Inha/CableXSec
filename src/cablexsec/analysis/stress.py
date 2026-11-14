# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Per-wire stress-component distribution — pure assembly.

Exposes the per-wire stress in every modelled structural helical layer at a
target curvature state ``(κ_x, κ_y)`` reached by a monotonic ramp from the
straight cable. The fixed-radius calculation uses the components below.
When a solved normal section state is supplied, sigma_A contains its complete
wire stress from extension, twist, radial motion and free/initial strain;
sigma_T is then zero to avoid counting the thermal stress twice.

============ ============================================== ====================
component    definition                                     source
============ ============================================== ====================
``σ_A``      ``E cos²α · N / EA_fixed`` (uniform per wire)   [Nasution2014b] Eq. 5
``σ_B``      outward radial-fibre bending stress             [MenardCartraud2023] 14
``σ_f``      Papailiou slip-front cap of the stick stress    [Papailiou1997]/[Kim2020]
``σ_T``      ``-E α_T (T(r_h) − T_ref)`` (0 if isothermal)   restrained increment
``σ_vm``     legacy axial/contact screening indicator       see limitations below
============ ====================================================== ==============

with ``σ_total = σ_A + σ_B + σ_f + σ_T`` and ``p_0`` the Hertz peak pressure of
the opposite-laid armour crossing.
The retained sigma_vm names do not denote a resolved contact tensor invariant:
the axial and contact terms need not be coincident, and the result is not a
proven upper bound. A local contact assessment must recover coincident tensors.

The friction stress ``σ_f`` is the **same Papailiou slip-front physics** that
:func:`~cablexsec.api.compute_mkappa` integrates, evaluated per physical wire.
A wire at azimuth ``θ = φ_k − φ_bend`` carries the stick axial stress
``E cos²β r |cosθ| κ`` capped by the frozen-friction slip stress
``σ_0 (exp(μ sinβ ψ) − 1)`` at the neutral-axis angle ``ψ = arcsin|cosθ|``
([Papailiou1997] Eq. 14, with ``σ_0 = (P1+P2) b_t r cosα / (A_w sin²α)``,
``b_t = 2π r / n``, per [Kim2020] Eq. 3.24; see
:func:`~cablexsec.mechanics.papailiou.wire_slip_stress_cap`). Because
``sinψ = |cosθ|`` maps the physical azimuth one-to-one onto the Papailiou
neutral-axis coordinate, the assembled cable moment
``Σ σ_f A_w r_h cos(φ−φ_bend) cosβ + EI_min κ`` recovers
:func:`~cablexsec.api.compute_mkappa` (the Papailiou engine) to within per-wire
discretisation. This is an internal consistency check, not independent
validation. Slip starts at the **neutral axis** (smallest ``κ_c``), per the
Raoof/Kim mechanism. The per-wire stick-slip boundary curvature ``κ_c,k`` is
reported.

References
----------
[Nasution2014b] Eq. 5; [MenardCartraud2023] Eq. 14; [Papailiou1997] Eqs. 14-15;
[Kim2020] Eq. 3.24; [Johnson1985]; [WitzTan1992b] (EI_min backbone).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from cablexsec.cable_adapters import (
    _fretting_contact,
    armour_crossing_line_loads,
    cable_to_mkappa_specs,
    require_fully_specified,
    structural_armour_layers,
)
from cablexsec.cables.schema import CableData
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.materials import library
from cablexsec.mechanics.papailiou import wire_slip_stress_cap
from cablexsec.mechanics.section_state import SectionState, solve_section_state
from cablexsec.mechanics.stiffness import ea_fixed, ei_min_slip
from cablexsec.mechanics.thermal import ThermalResult
from cablexsec.mechanics.wire_stress import (
    local_bending_amplitude,
    local_bending_stress,
    von_mises_stress,
)
from cablexsec.provenance import Provenance, require_finite

__all__ = ["FrictionField", "StressResult", "compute_stress"]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class FrictionField:
    """Friction carried over from a load history, round the section.

    The stress routes take the friction stress of a wire from the slip front
    at the curvature, which holds for a curvature reached from straight. A
    load history leaves another friction state (the stick zones and the slip
    the friction law followed); this field carries it. ``wires`` gives, per
    armour layer, stations ``(azimuth [rad], friction stress [Pa])`` round
    the section in the physics frame (the first wire on the x axis), read at
    each wire by periodic linear interpolation, as the kinematics reads them;
    ``cores`` gives stations ``(azimuth [rad], strain)`` of the bending part
    of the core-axis strain of the laid cores in the frictional state (the
    friction stress of the core law over its modulus), read at each core
    axis. Everything else (tension, local bending, the other components) is
    computed as without it.
    """

    wires: Mapping[str, tuple[FloatArray, FloatArray]] = field(default_factory=dict)
    cores: tuple[FloatArray, FloatArray] | None = None

    def wire_friction(self, name: str, phi: FloatArray) -> FloatArray | None:
        """The friction stress of the wires of layer ``name`` at azimuths
        ``phi`` [rad], or ``None`` when the field has none for the layer."""
        stations = self.wires.get(name)
        if stations is None:
            return None
        return _read(stations, phi)

    def core_strain(self, phases: FloatArray) -> FloatArray | None:
        """The frictional core-axis strain of the cores at ``phases`` [rad]."""
        if self.cores is None:
            return None
        return _read(self.cores, phases)

    def turned(self, angle: float) -> FrictionField:
        """The field turned by ``angle`` [rad] about the cable axis."""
        return FrictionField(
            {
                name: (azimuth + angle, values)
                for name, (azimuth, values) in self.wires.items()
            },
            None if self.cores is None else (self.cores[0] + angle, self.cores[1]),
        )


def _read(stations: tuple[FloatArray, FloatArray], phi: FloatArray) -> FloatArray:
    """Station values at ``phi``, periodic and linear between the stations."""
    from cablexsec.analysis.kinematics import periodic_interpolate  # noqa: PLC0415

    azimuth, values = (np.asarray(a, dtype=np.float64) for a in stations)
    if azimuth.ndim != 1 or values.shape != azimuth.shape or not azimuth.size:
        raise CableInputError("a friction field needs one value per station")
    if not (np.isfinite(azimuth).all() and np.isfinite(values).all()):
        raise CableInputError("a friction field must be finite")
    return np.asarray(periodic_interpolate(azimuth, values, np.asarray(phi)))


@dataclass(frozen=True)
class StressResult:
    """Per-wire stress decomposition for every structural helical layer.

    All per-layer fields are dicts keyed by layer name; each array runs over the
    layer's wires (azimuth ``phi_deg``).

    Attributes
    ----------
    phi_deg : dict[str, numpy.ndarray]
        Wire azimuths [deg].
    sigma_A_Pa, sigma_B_Pa, sigma_f_Pa, sigma_T_Pa, sigma_total_Pa,
    sigma_vm_Pa : dict[str, numpy.ndarray]
        Tension, local-bending, friction, thermal and total axial stress at the
        outward radial fibre [Pa]. sigma_vm_Pa is a legacy axial/contact
        screening indicator, not a resolved invariant at that physical fibre.
    sigma_perimeter_min_Pa, sigma_perimeter_max_Pa : dict[str, numpy.ndarray]
        Longitudinal stress extrema around each wire circumference [Pa].
    sigma_vm_envelope_Pa : dict[str, numpy.ndarray]
        Circumference stress envelope with the specified Hertz-shear
        approximation [Pa], not a spatially resolved contact stress tensor.
    slipping : dict[str, numpy.ndarray of bool]
        Whether each wire has reached gross slip at the target curvature.
    kappa_c_1_m : dict[str, numpy.ndarray]
        Per-wire stick-slip boundary curvature [1/m] (``inf`` at the neutral
        axis).
    kappa_mag_1_m, phi_bend_rad, tension_N : float
        The target curvature magnitude, its plane azimuth, and the axial tension.
    moment_recovered_Nm : float
        Cable moment assembled from ``σ_f`` + ``EI_min κ`` — recovers
        :func:`~cablexsec.api.compute_mkappa` (the Papailiou engine) at this
        curvature to within per-wire discretisation (A11 cross-engine gate).
    critical_layer : str
    critical_phi_deg, critical_sigma_vm_Pa : float
        Wire location and value of the peak legacy screening envelope. This
        does not locate a contact-volume hotspot or establish first yield.
    provenance : Provenance or None
    """

    phi_deg: dict[str, FloatArray]
    sigma_A_Pa: dict[str, FloatArray]
    sigma_B_Pa: dict[str, FloatArray]
    sigma_f_Pa: dict[str, FloatArray]
    sigma_T_Pa: dict[str, FloatArray]
    sigma_total_Pa: dict[str, FloatArray]
    sigma_vm_Pa: dict[str, FloatArray]
    slipping: dict[str, BoolArray]
    kappa_c_1_m: dict[str, FloatArray]
    kappa_mag_1_m: float
    phi_bend_rad: float
    tension_N: float
    moment_recovered_Nm: float
    critical_layer: str
    critical_phi_deg: float
    critical_sigma_vm_Pa: float
    provenance: Provenance | None = None
    section_state: SectionState | None = None
    sigma_perimeter_min_Pa: dict[str, FloatArray] = field(default_factory=dict)
    sigma_perimeter_max_Pa: dict[str, FloatArray] = field(default_factory=dict)
    sigma_vm_envelope_Pa: dict[str, FloatArray] = field(default_factory=dict)


def compute_stress(
    cable: CableData,
    kappa_x: float,
    kappa_y: float = 0.0,
    *,
    tension_N: float = 0.0,
    mu: float | None = None,
    thermal: ThermalResult | None = None,
    T_ref_C: float = 20.0,
    contact_model: str | None = None,
    section_state: SectionState | None = None,
    p_int_Pa: float = 0.0,
    p_ext_Pa: float | None = None,
    torque_Nm: float = 0.0,
    prescribed_twist_per_m: float | None = 0.0,
    friction: FrictionField | None = None,
) -> StressResult:
    r"""Per-wire all-component stress at a target curvature ``(κ_x, κ_y)``.

    The friction stress is the closed-form Papailiou slip-front (the same engine
    :func:`~cablexsec.api.compute_mkappa` integrates) evaluated per physical wire
    at ``κ_mag = √(κ_x² + κ_y²)`` along ``φ_bend = atan2(κ_y, κ_x)``, so the
    assembled cable moment recovers :func:`~cablexsec.api.compute_mkappa` to
    within per-wire discretisation (cross-engine consistency, gate A11).

    Parameters
    ----------
    cable : CableData
        A fully-specified cable.
    kappa_x, kappa_y : float
        Target biaxial curvature [1/m].
    tension_N : float, optional
        Axial tension for ``σ_A`` and the contact ``F_n`` [N].
    mu : float, optional
        Armour friction coefficient (default material ``mu_on_hdpe``).
    thermal : ThermalResult, optional
        If given, ``σ_T = -E α_T (T(r_h) - T_ref_C)`` is the
        fixed-global-strain thermal increment from the radial profile;
        otherwise ``σ_T = 0`` (isothermal). Use
        :func:`cablexsec.compute_component_response` when common axial strain
        and differential thermal expansion must be solved in equilibrium.
    T_ref_C : float, optional
        Stress-free reference temperature for ``σ_T`` [°C].
    section_state : SectionState, optional
        Reuse a normal equilibrium without prescribing separate section loads.
        The concentric calculation retains normal pressure, extension/twist
        and thermal expansion. Local bending is superposed on that normal
        state; bending-induced ovalisation and contact redistribution are
        not solved by this reduced model.
    friction : FrictionField, optional
        The friction stress a load history left on the wires, in place of the
        slip front at the curvature (:class:`FrictionField`); a wire is
        reported slipping where its friction stress has reached its slip
        stress.

    Returns
    -------
    StressResult

    Examples
    --------
    >>> from cablexsec import load_cable, compute_stress, menard2023_yaml_path
    >>> cab = load_cable(menard2023_yaml_path(), geometry_override=None)
    >>> s = compute_stress(cab, 0.05, tension_N=1.0e4)
    >>> s.critical_layer in {"inner_armour", "outer_armour"}
    True
    """
    require_fully_specified(cable, "compute_stress")
    require_finite([kappa_x, kappa_y, tension_N], "curvature/tension")
    if section_state is not None:
        section_state.require_cable(cable)
        if (
            tension_N != 0
            or thermal is not None
            or p_int_Pa != 0
            or p_ext_Pa is not None
            or torque_Nm != 0
            or prescribed_twist_per_m != 0
            or (
                contact_model is not None
                and contact_model != section_state.contact_model
            )
        ):
            raise CableSchemaError(
                "supply a section state or separate section loads, not both"
            )
        contact_model = section_state.contact_model
    else:
        contact_model = contact_model or "bounded"
        if (
            contact_model == "qp"
            or p_int_Pa
            or p_ext_Pa is not None
            or torque_Nm
            or prescribed_twist_per_m != 0
        ):
            section_state = solve_section_state(
                cable,
                tension_N=tension_N,
                torque_Nm=torque_Nm,
                prescribed_twist_per_m=prescribed_twist_per_m,
                p_int_Pa=p_int_Pa,
                p_ext_Pa=p_ext_Pa,
                thermal=thermal,
                T_ref_C=T_ref_C,
                contact_model=contact_model,
            )
    if section_state is not None:
        tension_N = section_state.tension_N
    specs = cable_to_mkappa_specs(
        cable,
        tension_N,
        mu=mu,
        contact_model=contact_model,
        section_state=section_state,
    )
    armours = structural_armour_layers(cable)
    crossings = armour_crossing_line_loads(
        cable, tension_N, contact_model=contact_model, section_state=section_state
    )
    ea_full = ea_fixed(cable)
    ei_min = ei_min_slip(
        cable,
        modulus_scale=section_state.modulus_scale
        if section_state is not None
        else None,
    )

    kappa_mag = math.hypot(kappa_x, kappa_y)
    phi_bend = math.atan2(kappa_y, kappa_x)

    phi_deg: dict[str, FloatArray] = {}
    s_A: dict[str, FloatArray] = {}
    s_B: dict[str, FloatArray] = {}
    s_f: dict[str, FloatArray] = {}
    s_T: dict[str, FloatArray] = {}
    s_tot: dict[str, FloatArray] = {}
    s_vm: dict[str, FloatArray] = {}
    s_min: dict[str, FloatArray] = {}
    s_max: dict[str, FloatArray] = {}
    s_envelope: dict[str, FloatArray] = {}
    slip: dict[str, BoolArray] = {}
    kc: dict[str, FloatArray] = {}
    moment_friction = 0.0
    crit_layer, crit_phi, crit_vm = "", 0.0, -1.0

    for spec, lyr in zip(specs, armours, strict=True):
        assert lyr.r_helix_m is not None
        phi = spec.phi_wires
        cphi = np.cos(phi - phi_bend)  # signed bending lever cos(θ) ∈ [−1, 1]
        a = spec.lay_angle_rad
        e_w = spec.E_wire_Pa
        a_w = spec.A_wire_m2
        r_h = lyr.r_helix_m
        d_wire = 2.0 * math.sqrt(a_w / math.pi)
        cos_b = math.cos(a)

        # sigma_f: the SAME Papailiou slip-front physics compute_mkappa
        # integrates, evaluated per physical wire. Measure transform — a wire
        # at azimuth θ = φ−φ_bend sits at neutral-axis angle ψ with
        # sin ψ = |cos θ| ([Papailiou1997] coords). sigma_0 from the cable's
        # P1+P2 ([Kim2020] Eq. 3.24). Stick stress is capped by the frozen
        # slip stress; slip onset is at the neutral axis (smallest κ_c).
        abs_c = np.abs(cphi)
        sigma_slip = wire_slip_stress_cap(spec, phi_bend)
        sigma_stuck = spec.k_f * r_h * abs_c * kappa_mag
        sigma_f = np.sign(cphi) * np.minimum(sigma_stuck, sigma_slip)
        slipping: BoolArray = np.asarray(sigma_stuck >= sigma_slip, dtype=bool)
        given = None if friction is None else friction.wire_friction(lyr.name, phi)
        if given is not None:
            # The friction a load history left, in place of the slip front.
            sigma_f = np.asarray(given, dtype=np.float64)
            slipping = np.asarray(
                np.abs(sigma_f) >= np.asarray(sigma_slip) * (1.0 - 1e-9), dtype=bool
            )
        # Moment about the bending axis: signed lever A r cosβ cos θ; sums to
        # EI_max κ at full stick (Σ cos²θ = n/2), recovering compute_mkappa.
        moment_friction += float(np.sum(sigma_f * a_w * r_h * cos_b * cphi))

        # sigma_A: uniform tension share ([Nasution2014b] Eq. 5).
        sigma_A_val = e_w * math.cos(a) ** 2 * tension_N / ea_full if ea_full else 0.0
        if section_state is not None:
            # The complete solved non-bending stress includes extension,
            # twist, radial motion and free/initial strain. Do not add a
            # second thermal increment or replace it with N/EA.
            sigma_A_val = section_state.layers[lyr.name].longitudinal_stress_Pa
        sigma_A = np.full(spec.n_wires, sigma_A_val)
        # sigma_B: local wire bending ([MenardCartraud2023] Eq. 14).
        section = spec.section
        depth = d_wire if section is None else 2.0 * section.half_thickness_m
        sigma_B = np.asarray(
            local_bending_stress(kappa_mag, phi, e_w, depth, a, phi_bend)
        )
        # sigma_T: fully restrained thermal increment (0 if isothermal).
        # Heating at fixed global strain is compressive in the tensile-positive
        # convention. Free/common-strain expansion is assembled by
        # compute_component_response through axial equilibrium.
        if thermal is not None and section_state is None:
            assert lyr.r_helix_m is not None
            t_layer = float(np.interp(lyr.r_helix_m, thermal.r_m, thermal.T_C))
            alpha_t = library.get_property(lyr.material, "alpha_T_K")
            sigma_T = np.full(spec.n_wires, -e_w * alpha_t * (t_layer - T_ref_C))
        else:
            sigma_T = np.zeros(spec.n_wires)

        sigma_total = sigma_A + sigma_B + sigma_f + sigma_T
        partner, line_load = crossings.get(lyr.name, (None, 0.0))
        _, p0, _ = _fretting_contact(
            lyr,
            partner,
            line_load,
            youngs_modulus_Pa=(
                section_state.youngs_modulus_Pa if section_state is not None else None
            ),
        )
        sigma_vm = np.asarray(von_mises_stress(sigma_total, p0))
        if section is None:
            bending_amplitude = local_bending_amplitude(
                kappa_mag, phi, e_w, d_wire, a, phi_bend
            )
        else:
            # Rectangular/profiled wire (gap G1): corner of the section.
            s_a = math.sin(a)
            face = e_w * kappa_mag * cos_b**2 * math.cos(2 * a) * cphi
            edge = e_w * kappa_mag * cos_b * (1 + s_a * s_a) * np.sin(phi - phi_bend)
            bending_amplitude = section.max_bending_stress(
                face * section.half_thickness_m, edge * section.half_width_m
            )
        mean = sigma_A + sigma_f + sigma_T
        s_min[lyr.name] = mean - bending_amplitude
        s_max[lyr.name] = mean + bending_amplitude
        envelope = von_mises_stress(np.abs(mean) + bending_amplitude, p0)
        s_envelope[lyr.name] = envelope

        # Per-wire stick-slip boundary curvature, consistent with the engine:
        # κ_c = σ_slip / (E cos²β r |cos θ|) — smallest at the neutral axis.
        denom = spec.k_f * r_h * abs_c
        safe = np.where(denom > 0.0, denom, 1.0)
        kappa_c = np.where(denom > 0.0, sigma_slip / safe, np.inf)

        phi_deg[lyr.name] = np.degrees(phi)
        s_A[lyr.name] = sigma_A
        s_B[lyr.name] = sigma_B
        s_f[lyr.name] = sigma_f
        s_T[lyr.name] = sigma_T
        s_tot[lyr.name] = sigma_total
        s_vm[lyr.name] = sigma_vm
        slip[lyr.name] = slipping
        kc[lyr.name] = kappa_c

        k_peak = int(np.argmax(envelope))
        if envelope[k_peak] > crit_vm:
            crit_layer = lyr.name
            crit_phi = float(np.degrees(phi[k_peak]))
            crit_vm = float(envelope[k_peak])

    moment_recovered = moment_friction + ei_min * kappa_mag
    return StressResult(
        phi_deg=phi_deg,
        sigma_A_Pa=s_A,
        sigma_B_Pa=s_B,
        sigma_f_Pa=s_f,
        sigma_T_Pa=s_T,
        sigma_total_Pa=s_tot,
        sigma_vm_Pa=s_vm,
        slipping=slip,
        kappa_c_1_m=kc,
        kappa_mag_1_m=kappa_mag,
        phi_bend_rad=phi_bend,
        tension_N=tension_N,
        moment_recovered_Nm=moment_recovered,
        critical_layer=crit_layer,
        critical_phi_deg=crit_phi,
        critical_sigma_vm_Pa=crit_vm,
        section_state=section_state,
        sigma_perimeter_min_Pa=s_min,
        sigma_perimeter_max_Pa=s_max,
        sigma_vm_envelope_Pa=s_envelope,
        provenance=Provenance(
            method="stress of every armour wire from tension, local wire "
            "bending, friction and temperature, with a legacy axial and crossing-"
            "contact screening indicator; friction from the Papailiou slip "
            "front of the bending law, evaluated wire by wire",
            citations=(
                "[Nasution2014b] Eq.5",
                "[MenardCartraud2023] Eq.14",
                "[Papailiou1997] Eqs.14-15",
                "[Kim2020] Eq.3.24",
                "[Johnson1985]",
            ),
            gates=(
                "sigma_B vs closed form (phi=0)",
                "sigma_f per-wire kappa_c boundary (neutral-axis slip onset)",
                "assembled moment recovers compute_mkappa Papailiou engine "
                "within per-wire discretisation (A11 cross-engine)",
            ),
            limitations=(
                "the screening indicator combines axial and circular-contact "
                "shear estimates that need not be coincident; it is not a "
                "resolved Hertz stress invariant, a proven upper bound or a "
                "qualified first-yield check",
                (
                    "friction stress of the wires carried over from a load "
                    "history (its friction law), not the slip front at the "
                    "curvature"
                    if friction is not None and friction.wires
                    else "monotonic load to the target curvature (no reversal history)"
                ),
                "the distribution over the wires agrees with the strain-gauge "
                "observation of slip onset at the neutral axis (Raoof 1989b, "
                "1992) and with the Papailiou closed form; a quantitative "
                "comparison with digitised wire-stress figures (Kim 2020, "
                "Ménard and Cartraud 2023) is still to be made",
                (
                    "the tension stress keeps the complete solved normal wire "
                    "stress; thermal strain is not added a second time"
                    if section_state is not None
                    else "a temperature profile on its own adds a fixed-strain "
                    "increment; the solved section state gives thermal axial-force "
                    "equilibrium"
                ),
                "the crossing contact pressure of the screening indicator comes "
                "from the opposite-laid armour crossing only (zero for a lone "
                "armour layer)",
                "uniform mean face pressure is used for armour friction; "
                "the shape of the inner foundation is not an independently "
                "determined armour-pressure distribution",
                "the concentric normal calculation and monotonic bending "
                "superposition are checked separately; agreement of their shared "
                "stress paths does not establish experimental accuracy",
            ),
        ),
    )
