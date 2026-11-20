# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Helical wire screens wound about each power core of a three-core cable.

Gap P1. A copper wire screen of a power core is a ring of ``n`` wires on a
helix of radius ``r_s`` and lay angle ``alpha`` about the **core** axis
(declared as a ``cylindrical_offset`` layer with
``construction: resolved_wire_screen`` and one resolved ring in
``strand_layers``). Its wire stress has two parts:

1. the response of a helical layer about the core axis to the core's own
   bending: the core curvature components ``(g_r, g_t)`` of
   :class:`~cablexsec.mechanics.core_kinematics.CoreKinematics` drive the
   unchanged helical-layer kernel of the armour
   (:func:`~cablexsec.analysis.fatigue.wire_stress_field`: wire bending, the
   direction-tied Papailiou friction stress on the clamping ``P1 + P2`` of the
   two faces, Yun stick coefficient of the core layers inside the ring), with
   the wire azimuth measured in the core section from the outward radial
   direction (``theta = 0`` away from the cable axis);
2. the core-axis strain ``epsilon_c`` of the core slip state (P8), carried
   by the wire as ``E cos^2(alpha) epsilon_c``. It is common to the core's
   layers, so it adds no slip between the screen and its neighbours.

Each station of the core along its lay (azimuth ``phi`` on the pitch circle)
is assessed: bulk stress ranges and Miner damage against the layer's
``bulk_sn`` curve at the four wire fibres, and on each radial face the
fretting regime, slip amplitude, wear and knock-down of
:func:`~cablexsec.analysis.fretting_assessment.helical_layer_fretting`.

Faces and clamping
------------------
The inner face bears on the layer of the same core whose outer radius is the
ring's inner radius (a semiconducting or swelling tape); the outer face on
the layer starting at the ring's outer radius (a copper tape, a swelling
tape or a foil). The outer face carries the declared ``outer_face_pressure``
(tape/foil winding or sheath shrink fit,
:class:`~cablexsec.cables.schema.FacePressure`); the inner face carries the
same pressure (the wires are laid without tension). With
``core_contact=True`` the core contact line load of the bundle
(:func:`~cablexsec.mechanics.core_friction.core_contact_loads`), spread over
the ring circumference, is added to both faces (an upper estimate of its
transfer through the core sheath, as in the core friction law). The
friction-setting pressure is ``P1 + P2 = p_inner + p_outer`` and the section
cap uses the pressure-weighted friction coefficient of the two faces.

Limitations (stated)
--------------------
* Elastic wire stress: positions above the yield stress are flagged
  (``exceeds_yield``). With ``strain_life=`` (gap P7) every fibre is also
  assessed by the strain-life route on the imposed wire strain
  ``sigma_el / E`` (Masing cyclic curve, Coffin--Manson--Basquin with SWT
  or Morrow). No copper-specific fretting knock-down is published in the
  reference library: ``"auto"`` applies the steel relation, labelled
  assumed.
* One resolved ring per screen layer; the counter-helix tape is assessed by
  :func:`~cablexsec.analysis.tape.compute_tape_fatigue`.
* Contact loads are fixed during the history (as for the armour).

References
----------
[Papailiou1997]; [Yun2020]; [Fouvry1995]; [Johnson1985]; gap P8 kinematics
(:mod:`cablexsec.mechanics.core_kinematics`).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cablexsec.analysis.fatigue import FatigueLayer, miner_damage, wire_stress_field
from cablexsec.analysis.fretting_assessment import (
    SECONDS_PER_YEAR,
    FaceSpec,
    FrettingFace,
    helical_layer_fretting,
)
from cablexsec.cables.schema import CableData, CylindricalOffsetLayer
from cablexsec.exceptions import CableInputError, CableLookupError, CableSchemaError
from cablexsec.materials import library
from cablexsec.materials.sn_curves import BlockedSNCurveError, get_sn_curve
from cablexsec.materials.strain_life import (
    MEAN_STRESS_CORRECTIONS,
    MeanStressCorrection,
    StrainLifeConstants,
    get_strain_life,
    strain_life_damage_map,
)
from cablexsec.materials.wear import (
    FrettingKnockdown,
    WearCoefficient,
    get_knockdown,
    get_wear_coefficient,
    wear_pair_for,
)
from cablexsec.mechanics.core_friction import CoreFrictionSpec, core_contact_loads
from cablexsec.mechanics.core_kinematics import (
    CoreKinematics,
    CoreSlipState,
    core_kinematics,
)
from cablexsec.mechanics.mkappa import HelicalLayerSpec
from cablexsec.mechanics.radial_cable import CableRadialState
from cablexsec.mechanics.stiffness import ea_fixed
from cablexsec.mechanics.yun_coefficient import (
    series_shear_stiffness,
    shear_modulus,
    yun_xi,
)
from cablexsec.provenance import MissingClampingStateWarning, Provenance

__all__ = [
    "CoreScreenResult",
    "compute_core_screen_fatigue",
    "screen_layers",
]

FloatArray = NDArray[np.float64]

#: Fibres of the screen wire at which bulk damage is evaluated: outer face,
#: tangential sides and inner face.
FIBRES_RAD = (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)
DEFAULT_SCREEN_STATIONS = 12
_TOL = 1e-6


@dataclass(frozen=True)
class CoreScreenResult:
    """Per-station, per-wire response of a screen ring about each core.

    Arrays are indexed ``(station, wire)``; ``phase_deg`` is the core
    station on the pitch circle (0 deg: extension side of ``kappa_x``),
    ``wire_azimuth_deg`` the wire position in the core section (0 deg: away
    from the cable axis). ``stress_range_Pa`` has a third axis over the
    fibres :data:`FIBRES_RAD`. ``D_bulk`` is the record's Miner damage
    (maximum over the fibres) and ``D_bulk_per_year`` its annualised value;
    ``faces[station][face]`` are the fretting results (annualised).
    ``D_strain_life`` (gap P7, with ``strain_life=``) is the strain-life
    damage of the record, maximum over the fibres, shape ``(station, wire)``.
    """

    layer_name: str
    core_state: str
    phase_deg: FloatArray
    wire_azimuth_deg: FloatArray
    stress_range_Pa: FloatArray
    stress_max_abs_Pa: FloatArray
    D_bulk: FloatArray
    D_bulk_per_year: FloatArray
    faces: tuple[dict[str, FrettingFace], ...]
    slip_amplitude_m: FloatArray
    axis_strain_range: FloatArray
    exceeds_yield: NDArray[np.bool_]
    p_inner_Pa: float
    p_outer_Pa: float
    mu_inner: float
    mu_outer: float
    mu_section: float
    xi: float
    sn_curve: str
    kinematics: CoreKinematics
    provenance: Provenance
    D_strain_life: FloatArray | None = None
    strain_life_constants: str = ""

    def critical(self) -> dict[str, float | str]:
        """Station, wire and value of the largest bulk and fretting damage."""
        s, w = np.unravel_index(int(np.argmax(self.D_bulk)), self.D_bulk.shape)
        out: dict[str, float | str] = {
            "D_bulk_max": float(self.D_bulk[s, w]),
            "bulk_phase_deg": float(self.phase_deg[s]),
            "bulk_wire_azimuth_deg": float(self.wire_azimuth_deg[w]),
            "stress_range_max_Pa": float(np.max(self.stress_range_Pa)),
            "slip_amplitude_max_m": float(np.max(self.slip_amplitude_m)),
        }
        best = -1.0
        for k, faces in enumerate(self.faces):
            for name, face in faces.items():
                values = face.D_fretting_per_year
                if np.isfinite(values).all() and values.size and np.max(values) > best:
                    best = float(np.max(values))
                    out["D_fretting_per_year_max"] = best
                    out["fretting_phase_deg"] = float(self.phase_deg[k])
                    out["fretting_face"] = name
                    out["fretting_wire_azimuth_deg"] = float(
                        self.wire_azimuth_deg[int(np.argmax(values))]
                    )
        return out


def screen_layers(cable: CableData) -> list[CylindricalOffsetLayer]:
    """Resolved wire screens of the power cores (``resolved_wire_screen``)."""
    return [
        x
        for x in cable.layers
        if isinstance(x, CylindricalOffsetLayer)
        and x.construction == "resolved_wire_screen"
        and x.strand_layers
    ]


def _core_family(
    cable: CableData, layer: CylindricalOffsetLayer
) -> list[CylindricalOffsetLayer]:
    return [
        x
        for x in cable.layers
        if isinstance(x, CylindricalOffsetLayer)
        and x.name != layer.name
        and x.r_offset_m == layer.r_offset_m
        and x.n_cores == layer.n_cores
    ]


def _partners(
    cable: CableData, layer: CylindricalOffsetLayer, r_in: float, r_out: float
) -> tuple[CylindricalOffsetLayer | None, CylindricalOffsetLayer | None]:
    inner = outer = None
    for other in _core_family(cable, layer):
        if other.r_outer_m is not None and abs(other.r_outer_m - r_in) <= _TOL:
            inner = other
        if other.r_inner_m is not None and abs(other.r_inner_m - r_out) <= _TOL:
            outer = other
    return inner, outer


def _pair_mu(wire_material: str, partner_material: str) -> float:
    """Friction of the screen wire on a partner (copper or polymer)."""
    key = "mu_on_cu" if "copper" in partner_material.lower() else "mu_on_hdpe"
    if "steel" in partner_material.lower():
        key = "mu_on_steel"
    return library.get_property(wire_material, key)


def _ring_xi(
    cable: CableData,
    layer: CylindricalOffsetLayer,
    r_helix: float,
    d_wire: float,
    ea_wire: float,
    alpha: float,
) -> float:
    """Yun coefficient of the ring on the shear of the core layers inside it.

    The core's layers are concentric about the ring's own axis, so they form
    the series stack of :func:`~cablexsec.mechanics.stiffness.xi_for_layer`:
    every layer between the ring and its support. As there, the stack stops
    at a helical layer inside, here another resolved wire screen of the core.
    """
    floor = r_helix - d_wire / 2 + _TOL
    support = max(
        (
            other.r_outer_m or 0.0
            for other in _core_family(cable, layer)
            if other.construction == "resolved_wire_screen"
            and (other.r_outer_m or 0.0) <= floor
        ),
        default=0.0,
    )
    stack: list[tuple[float, float, float]] = []
    for other in sorted(_core_family(cable, layer), key=lambda x: x.r_inner_m or 0.0):
        ri, ro = other.r_inner_m or 0.0, other.r_outer_m or 0.0
        if ro > floor or ro <= ri or ri < support - _TOL:
            continue
        try:
            e = library.get_property(other.material, "E_Pa")
            nu = library.get_property(other.material, "nu")
        except KeyError:
            continue
        stack.append((ro - ri, shear_modulus(e, nu), d_wire))
    return yun_xi(series_shear_stiffness(stack), ea_wire, alpha, r_helix)


def compute_core_screen_fatigue(
    cable: CableData,
    kappa_x: ArrayLike,
    kappa_y: ArrayLike | None = None,
    *,
    core_state: CoreSlipState,
    layer_name: str | None = None,
    duration_s: float = 1.0,
    time_fraction: float = 1.0,
    tension_N: ArrayLike | None = None,
    axial_strain: ArrayLike | None = None,
    n_stations: int | None = None,
    core_friction: CoreFrictionSpec | None = None,
    contact_tension_N: float = 0.0,
    contact_model: str = "bounded",
    radial_state: CableRadialState | None = None,
    core_contact: bool = False,
    mu_inner: float | None = None,
    mu_outer: float | None = None,
    mu: float | None = None,
    xi: float | None = None,
    knockdown: FrettingKnockdown | str | None = "auto",
    wear: Mapping[str, WearCoefficient] | None = None,
    partial_slip_wear: bool = False,
    strain_life: str | StrainLifeConstants | None = None,
    correction: MeanStressCorrection = "swt",
) -> CoreScreenResult:
    """Stress ranges, bulk fatigue, slip and fretting of a per-core wire screen.

    Parameters
    ----------
    cable : CableData
        Cable whose power cores carry a ``resolved_wire_screen`` layer.
    kappa_x, kappa_y : array_like
        Cable curvature history [1/m].
    core_state : {"stuck", "free", "frictional"}
        Core slip state (:mod:`cablexsec.mechanics.core_kinematics`);
        required.
    layer_name : str or None
        Screen layer (default: the only resolved wire screen).
    duration_s, time_fraction : float
        Record duration [s] and occurrence fraction for annualised values.
    tension_N, axial_strain : array_like or None
        Cable axial loading (tension over the fixed-end ``EA``, or strain).
    n_stations : int or None
        Core stations assessed along one lay (default 12; the frictional
        state assesses that many equally spaced stations of its population).
    core_friction, contact_tension_N, contact_model, radial_state
        Inputs of the frictional core law and of the core contact loads.
    core_contact : bool
        Add the core contact line load to both faces (default False).
    mu_inner, mu_outer : float or None
        Face friction coefficients (default: the wire material's ``mu_on_cu``
        on copper, ``mu_on_hdpe`` on a polymer or tape).
    mu : float or None
        Friction coefficient of the section cap (default: pressure-weighted
        mean of the faces).
    xi : float or None
        Stick coefficient of the ring (default: Yun coefficient on the core
        layers inside the ring).
    knockdown : FrettingKnockdown, str or None
        Fretting knock-down (``"auto"``: the registered law of the wire
        material family; ``None`` skips the face S-N assessment).
    wear : mapping or None
        Wear coefficients replacing registered ones, keyed by pair.
    partial_slip_wear : bool
        Also wear on partial-slip energy.
    strain_life : str, StrainLifeConstants or None
        Gap P7: also assess every fibre by the strain-life route
        (:mod:`cablexsec.materials.strain_life`: Masing cyclic curve,
        Coffin--Manson--Basquin) on the wire strain ``sigma_el / E`` of the
        elastic analysis, taken as imposed (strain controlled: the wire
        follows the core); ranges above yield are then assessed, not only
        flagged. ``None`` (default) leaves the result unchanged.
    correction : {"swt", "morrow", "none"}
        Mean-stress correction of the strain-life route.
    """
    candidates = screen_layers(cable)
    if layer_name is not None:
        candidates = [x for x in candidates if x.name == layer_name]
    if len(candidates) != 1:
        raise CableSchemaError(
            "compute_core_screen_fatigue needs exactly one resolved_wire_screen "
            "layer" + ("" if layer_name is None else f" named {layer_name!r}")
        )
    layer = candidates[0]
    rings = [r for r in layer.strand_layers if r.r_helix_m > 0.0]
    if len(rings) != 1:
        raise CableSchemaError(
            f"screen {layer.name!r} must have exactly one resolved wire ring"
        )
    ring = rings[0]
    if not layer.bulk_sn:
        raise CableSchemaError(f"screen {layer.name!r} has no bulk_sn curve")
    try:
        sn_bulk = get_sn_curve(layer.bulk_sn)
    except (BlockedSNCurveError, KeyError) as exc:
        raise CableSchemaError(
            f"screen {layer.name!r} bulk S-N {layer.bulk_sn!r} unavailable: {exc}"
        ) from None
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise CableInputError("duration_s must be finite and positive")
    if not math.isfinite(time_fraction) or time_fraction < 0.0:
        raise CableInputError("time_fraction must be finite and non-negative")
    if tension_N is not None and axial_strain is not None:
        raise CableInputError("prescribe axial strain or tension, not both")
    kx = np.asarray(kappa_x, dtype=np.float64)
    strain = None
    if tension_N is not None:
        strain = np.asarray(tension_N, dtype=np.float64) / ea_fixed(cable)
    elif axial_strain is not None:
        strain = np.asarray(axial_strain, dtype=np.float64)
    kin = core_kinematics(
        cable,
        kx,
        kappa_y,
        state=core_state,
        axial_strain=strain,
        n_stations=None
        if core_state == "frictional"
        else (DEFAULT_SCREEN_STATIONS if n_stations is None else n_stations),
        core_friction=core_friction,
        contact_tension_N=contact_tension_N,
        contact_model=contact_model,
        radial_state=radial_state,
    )
    if core_state == "frictional":
        count = DEFAULT_SCREEN_STATIONS if n_stations is None else n_stations
        if count < 1 or count > kin.n_stations:
            raise CableInputError("n_stations must lie in [1, population size]")
        stations = np.unique(
            np.round(np.arange(count) * kin.n_stations / count).astype(int)
        )
    else:
        stations = np.arange(kin.n_stations)

    material = layer.material
    try:
        e_wire = library.get_property(material, "E_eff_axial_Pa")
    except KeyError:
        e_wire = library.get_property(material, "E_Pa")
    nu_wire = library.get_property(material, "nu")
    sigma_uts = library.get_property(material, "sigma_UTS_Pa")
    try:
        sigma_y = library.get_property(material, "sigma_y_Pa")
    except KeyError:
        sigma_y = math.inf
    alpha = ring.lay_angle_rad
    area = math.pi * ring.d_wire_m**2 / 4.0
    r_in = ring.r_helix_m - ring.d_wire_m / 2.0
    r_out = ring.r_helix_m + ring.d_wire_m / 2.0
    inner, outer = _partners(cable, layer, r_in, r_out)

    limitations: list[str] = []
    declared = layer.outer_face_pressure
    if declared is None:
        warnings.warn(
            f"screen {layer.name!r} declares no outer_face_pressure: its faces "
            "carry only the core contact load (zero without core_contact)",
            MissingClampingStateWarning,
            stacklevel=2,
        )
        p_wind = 0.0
        limitations.append("no declared outer-face (tape/sheath) pressure")
    else:
        p_wind = declared.pressure_Pa
        limitations.append(
            f"outer-face pressure {p_wind / 1e3:.6g} kPa "
            f"({declared.identification}: {declared.source})"
            + (
                f"; bound {declared.bounds_Pa[0] / 1e3:.6g}-"
                f"{declared.bounds_Pa[1] / 1e3:.6g} kPa: run both ends"
                if declared.bounds_Pa is not None
                else ""
            )
        )
    p_core = 0.0
    if core_contact:
        loads = core_contact_loads(
            cable,
            contact_tension_N,
            contact_model=contact_model,
            radial_state=radial_state,
            central_filler=None
            if core_friction is None
            else core_friction.central_filler,
        )
        p_core = loads.total_N_per_m / (2.0 * math.pi * r_out)
        limitations.append(
            f"core contact load {loads.total_N_per_m:.6g} N/m ({loads.source}) "
            "added to both faces (upper estimate of its transfer)"
        )
    p_outer = p_wind + p_core
    p_inner = p_outer

    def face_mu(partner: CylindricalOffsetLayer | None, given: float | None) -> float:
        if given is not None:
            if not math.isfinite(given) or given < 0.0:
                raise CableInputError("friction coefficients must be finite, >= 0")
            return float(given)
        if partner is None:
            return 0.0
        try:
            return _pair_mu(material, partner.material)
        except CableLookupError:
            raise CableInputError(
                f"no friction coefficient of {material!r} on {partner.material!r}; "
                "pass mu_inner/mu_outer"
            ) from None

    m_in, m_out = face_mu(inner, mu_inner), face_mu(outer, mu_outer)
    if inner is None:
        p_inner = 0.0
        limitations.append(f"{layer.name}: no core layer touches the inner face")
    if outer is None:
        p_outer = 0.0
        limitations.append(f"{layer.name}: no core layer touches the outer face")
    total = p_inner + p_outer
    mu_sec = (
        float(mu)
        if mu is not None
        else ((m_in * p_inner + m_out * p_outer) / total if total > 0 else m_in)
    )
    ring_xi = (
        float(xi)
        if xi is not None
        else _ring_xi(cable, layer, ring.r_helix_m, ring.d_wire_m, e_wire * area, alpha)
    )
    spec = HelicalLayerSpec(
        n_wires=ring.n_wires,
        r_helix_m=ring.r_helix_m,
        lay_angle_rad=alpha,
        E_wire_Pa=e_wire,
        A_wire_m2=area,
        xi=ring_xi,
        mu=mu_sec,
        F_n_axi_N=0.0,
        contact_pressure_Pa=total,
        normal_load_is_line=True,
    )
    fatigue_layer = FatigueLayer(
        spec=spec,
        sn_bulk=sn_bulk,
        sigma_UTS_Pa=sigma_uts,
        name=layer.name,
        contact_nu=nu_wire,
        lay_hand=1 if ring.lay_direction == "right_hand" else -1,
    )
    overrides = dict(wear or {})
    faces: list[FaceSpec] = []
    for name, angle, partner, pressure, m in (
        ("inner_face", math.pi, inner, p_inner, m_in),
        ("outer_face", 0.0, outer, p_outer, m_out),
    ):
        if partner is None:
            continue
        coefficient: WearCoefficient | None
        try:
            key = wear_pair_for(material, partner.material)
            coefficient = overrides.get(key, get_wear_coefficient(key))
        except CableLookupError:
            coefficient = None
            limitations.append(f"{layer.name}/{name}: no wear coefficient")
        try:
            e_partner = library.get_property(partner.material, "E_Pa")
        except KeyError:
            e_partner = library.get_property(partner.material, "E_eff_axial_Pa")
        faces.append(
            FaceSpec(
                name=name,
                angle_rad=angle,
                kind="line",
                partner=partner.name,
                partner_E_Pa=e_partner,
                partner_nu=library.get_property(partner.material, "nu"),
                mu=m,
                wear=coefficient,
                pressure_Pa=pressure,
            )
        )
    law: FrettingKnockdown | None
    if knockdown == "auto":
        family = "copper" if "copper" in material.lower() else "steel"
        law = get_knockdown(f"{family}_nishioka_hirakawa")
    elif isinstance(knockdown, str):
        law = get_knockdown(knockdown)
    else:
        law = knockdown

    life_constants: StrainLifeConstants | None = None
    if strain_life is not None:
        if correction not in MEAN_STRESS_CORRECTIONS:
            raise CableInputError(
                f"mean-stress correction must be one of {MEAN_STRESS_CORRECTIONS}"
            )
        life_constants = (
            get_strain_life(strain_life)
            if isinstance(strain_life, str)
            else strain_life
        )
    per_year = SECONDS_PER_YEAR / duration_s * time_fraction
    n_st, n_w = stations.size, ring.n_wires
    strain_rows: list[FloatArray] = []
    ranges = np.zeros((n_st, n_w, len(FIBRES_RAD)))
    peak = np.zeros((n_st, n_w))
    damage = np.zeros((n_st, n_w))
    slip = np.zeros((n_st, n_w))
    results: list[dict[str, FrettingFace]] = []
    for row, index in enumerate(stations):
        g_r = kin.gradient_radial_per_m[index]
        g_t = kin.gradient_tangential_per_m[index]
        eps_c = kin.axis_strain[index]
        field = wire_stress_field(fatigue_layer, g_r, g_t, axial_strain=eps_c)
        for f, angle in enumerate(FIBRES_RAD):
            stress = field.at_fibre(angle)
            if life_constants is not None:
                strain_rows.append(np.asarray(stress / e_wire))
            ranges[row, :, f] = np.ptp(stress, axis=1)
            peak[row] = np.maximum(peak[row], np.max(np.abs(stress), axis=1))
            for w in range(n_w):
                damage[row, w] = max(
                    damage[row, w], miner_damage(stress[w], sn_bulk, sigma_uts)
                )
        faces_out = (
            helical_layer_fretting(
                fatigue_layer,
                faces,
                g_r,
                g_t,
                duration_s=duration_s,
                nu_wire=nu_wire,
                axial_strain=eps_c,
                time_fraction=time_fraction,
                design_life_years=cable.cable.design_life_years,
                knockdown=law,
                partial_slip_wear=partial_slip_wear,
            )
            if faces
            else {}
        )
        for face in faces_out.values():
            slip[row] = np.maximum(slip[row], face.max_amplitude_m)
        results.append(faces_out)
    exceeds = np.asarray(peak > sigma_y)
    d_life: FloatArray | None = None
    if life_constants is not None:
        n_t = strain_rows[0].shape[1] if strain_rows else 0
        # rows ordered (station, fibre, wire) -> damage (station, fibre, wire)
        life = strain_life_damage_map(
            np.concatenate(strain_rows, axis=0).reshape(-1, n_t),
            life_constants,
            correction=correction,
        )
        d_life = np.asarray(life.damage.reshape(n_st, len(FIBRES_RAD), n_w).max(axis=1))
        limitations.append(
            f"strain-life route ({life_constants.name}, {life_constants.basis}, "
            f"{correction}): wire strain sigma_el / E taken as imposed (strain "
            "controlled); Masing cyclic curve"
            + (
                " by the compatibility estimate"
                if life_constants.cyclic_curve_is_estimate
                else ""
            )
        )
    if np.any(exceeds):
        limitations.append(
            f"elastic wire stress exceeds the yield stress {sigma_y / 1e6:.6g} MPa "
            f"at {int(np.count_nonzero(exceeds))} station-wire positions: outside "
            "the elastic S-N basis"
            + (
                "; assessed by the strain-life route (D_strain_life)"
                if d_life is not None
                else " (pass strain_life= for the strain-life route, gap P7)"
            )
        )
    if core_state == "stuck":
        limitations.append(
            "stuck core: plane-section upper bound of the core-axis strain"
        )
    elif core_state == "free":
        limitations.append("free core: full-slip lower bound of the core-axis strain")
    phase = np.degrees(kin.phase_rad[stations])
    provenance = Provenance(
        method=(
            "per-core helical wire screen: helical-layer stress and slip under "
            "the core curvature plus the core-axis strain of the "
            f"{core_state!r} core state; bulk rainflow/Miner and face fretting"
        ),
        citations=(
            "[Papailiou1997]",
            "[Yun2020]",
            "[MenardCartraud2023] Eq. 14",
            "[Fouvry1995]",
            "[Johnson1985]",
            sn_bulk.reference,
        ),
        gates=(
            "single core (r_c = 0): equal to the helical-layer result",
            "stuck core: plane-section axis strain; free core: zero axis strain",
        ),
        limitations=tuple(limitations)
        + (
            "contact loads fixed during the history",
            "the counter-helix tape is assessed separately (compute_tape_fatigue)",
        ),
    )
    axis_range = np.ptp(kin.axis_strain[stations], axis=1)
    return CoreScreenResult(
        layer_name=layer.name,
        core_state=core_state,
        phase_deg=phase,
        wire_azimuth_deg=np.degrees(spec.phi_wires),
        stress_range_Pa=ranges,
        stress_max_abs_Pa=peak,
        D_bulk=damage,
        D_bulk_per_year=damage * per_year,
        faces=tuple(results),
        slip_amplitude_m=slip,
        axis_strain_range=np.asarray(axis_range),
        exceeds_yield=exceeds,
        p_inner_Pa=p_inner,
        p_outer_Pa=p_outer,
        mu_inner=m_in,
        mu_outer=m_out,
        mu_section=mu_sec,
        xi=ring_xi,
        sn_curve=layer.bulk_sn,
        kinematics=replace(kin),
        provenance=provenance,
        D_strain_life=d_life,
        strain_life_constants="" if life_constants is None else life_constants.name,
    )
