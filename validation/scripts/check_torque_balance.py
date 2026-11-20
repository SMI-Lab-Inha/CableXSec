# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Public checks of the torque-balance helper (gap 3.5).

Run as ``python validation/scripts/check_torque_balance.py``. This is a verification
module: nothing here is used by an analysis path. Every case is a public
source in the reference library; nothing is fitted.

Sources
-------
* Feyrer (2015), Wire Ropes, 2nd ed., Example 2.8 with Fig. 2.4: open spiral
  rope 1x37, centre wire 1.35 mm, three layers of 6, 12 and 18 wires of
  1.25 mm at +14, -14 and +14 deg; torque constant ``c_1S = 0.0345`` with
  ``d_S = 8.85 mm`` (Eq. 2.74, Poisson contraction neglected).
* Knapp (1981), J. Eng. Ind. 103, 61-66, Eqs. 3-4, 18 and 20: the wire
  strain of a cable whose core contracts as ``-theta/2 eps R_c`` reduces the
  layer torque by the core factor ``F_c = 1 - theta R_c tan^2(a) / (2 R)``.
  A core of Poisson ratio ``nu`` that is radially stiff compared with the
  wire pressure contracts with ``theta = 2 nu``.
* Kim and Lee (2017), Constr. Build. Mater. 151, 591-606, Section 5.1 and
  Table 4: two-layer strand 1+6+12, core 3.66 mm, wires 3.33 mm, layer 1
  right-hand, layer 2 left-hand; torque-balance pairs of helix angles
  (measured from the cross-section) from the beam FE model, Fig. 19 legend
  (72.5-86.3, 75.0-86.7, 77.5-87.1, 80.0-87.6, 82.5-88.2, 85.0-88.8) and the
  final design (78.85, 87.41), Section 5.1.
* Utting and Jones (1987), Int. J. Mech. Sci. 29, Part I Table 1 (core
  3.94 mm, wires 3.73 mm, measured lay angles) and Part II Table 1 (fixed-end
  torque per kN: Machida-Durelli theory, column ii, and test, column vii).
* Knapp (1981), Eqs. 2 and 11 with a radially rigid core (theta = 0): the
  wire strain ``eps cos^2(a) + gamma sin^2(a)`` with the twist strain
  ``gamma = R tau / tan(a)``, i.e. ``eps cos^2(a) + s R tau sin(a) cos(a)``,
  the twist-rate stress of the fatigue path (row TQ-FT).
"""

from __future__ import annotations

import math
import warnings

import numpy as np

import cablexsec as cx
from cablexsec.analysis.fatigue import (
    FatigueLayer,
    miner_damage,
    twist_stress_coefficient,
    wire_stress_field,
)
from cablexsec.analysis.torque_balance import design_torque_balance, torque_balance
from cablexsec.cables.schema import CableData, CableMeta, CylindricalLayer, HelicalLayer
from cablexsec.materials import library
from cablexsec.materials.sn_curves import get_sn_curve
from cablexsec.mechanics.mkappa import HelicalLayerSpec
from cablexsec.mechanics.stiffness import axial_torsional_matrix

STEEL = "galvanised_steel_armour"  # E and nu cancel or are stated per case

# --------------------------------------------------------------------------- #
# Source data                                                                 #
# --------------------------------------------------------------------------- #
FEYRER_D0, FEYRER_D, FEYRER_DS = 1.35e-3, 1.25e-3, 8.85e-3
FEYRER_LAYERS = ((6, 14.0), (12, -14.0), (18, 14.0))
FEYRER_C1S = 0.0345

KIM_DC, KIM_DW = 3.66e-3, 3.33e-3
KIM_PAIRS = (  # (alpha_1, alpha_2) helix angles from the cross-section [deg]
    (72.5, 86.3),
    (75.0, 86.7),
    (77.5, 87.1),
    (80.0, 87.6),
    (82.5, 88.2),
    (85.0, 88.8),
    (78.85, 87.41),
)
KIM_TENSION_N = 100.0e3

UJ_DC, UJ_DH = 3.94e-3, 3.73e-3
UJ_STRANDS = {  # name: (measured lay angle deg, Machida-Durelli, test) N m/kN
    "I": (17.03, 0.923, 0.86),
    "II": (13.99, 0.757, 0.72),
    "III": (12.92, 0.699, 0.69),
    "IV": (12.20, 0.659, 0.65),
    "V": (11.03, 0.596, 0.58),
    "VI": (9.23, 0.498, 0.47),
}
UJ_TENSION_N = 40.0e3  # Part II Table 1 load range
#: Any stress-range curve in MPa: TQ-FT compares stresses and Miner sums.
SN_CURVE = "oc2017_armour_in_air"


# --------------------------------------------------------------------------- #
# Cable builders                                                              #
# --------------------------------------------------------------------------- #
def strand(
    name: str,
    core_diameter_m: float,
    wire_diameter_m: float,
    layers: tuple[tuple[int, float], ...],
    *,
    core_material: str = STEEL,
) -> CableData:
    """Steel core with touching helical layers; signed lay angle in degrees.

    A positive angle is a right-hand lay, a negative one a left-hand lay.
    """
    items: list[CylindricalLayer | HelicalLayer] = [
        CylindricalLayer(
            name="core",
            material=core_material,
            r_inner_m=0.0,
            r_outer_m=core_diameter_m / 2,
        )
    ]
    radius = core_diameter_m / 2 + wire_diameter_m / 2
    for i, (count, angle) in enumerate(layers, start=1):
        items.append(
            HelicalLayer(
                name=f"layer_{i}",
                material=STEEL,
                n_wires=count,
                d_wire_m=wire_diameter_m,
                r_helix_m=radius,
                lay_angle_deg=abs(angle),
                lay_direction="right_hand" if angle > 0 else "left_hand",
            )
        )
        radius += wire_diameter_m
    return CableData(cable=CableMeta(name=name), layers=tuple(items))


def feyrer_rope() -> CableData:
    return strand("Feyrer 1x37", FEYRER_D0, FEYRER_D, FEYRER_LAYERS)


def kim_strand(alpha_1: float, alpha_2: float) -> CableData:
    """Kim and Lee two-layer strand; helix angles from the cross-section."""
    return strand(
        "Kim and Lee 1+6+12",
        KIM_DC,
        KIM_DW,
        ((6, 90.0 - alpha_1), (12, -(90.0 - alpha_2))),
    )


def uj_strand(lay_deg: float) -> CableData:
    return strand("Utting and Jones 1+6", UJ_DC, UJ_DH, ((6, lay_deg),))


# --------------------------------------------------------------------------- #
# Checks                                                                      #
# --------------------------------------------------------------------------- #
def feyrer() -> dict[str, float]:
    """Torque constant of Feyrer Example 2.8 (fixed radius and solved)."""
    rope = feyrer_rope()
    at = axial_torsional_matrix(rope)
    solved = torque_balance(rope, tensions_N=[10.0e3])
    return {
        "c1S_fixed_radius": at.K_et / at.K_ee / FEYRER_DS,
        "c1S_solved": solved.torque_coefficient_m / FEYRER_DS,
        "c1S_source": FEYRER_C1S,
    }


def knapp_core_factor(core_scale: float = 1.0e3) -> dict[str, float]:
    """Fixed-twist torque per unit strain against Knapp's F_c with theta = 2 nu.

    The core modulus is raised by ``core_scale`` so that the wire pressure
    does not compress it; it still contracts with its Poisson ratio.
    """
    rope = feyrer_rope()
    e = 210.0e9
    nu = 0.30
    state = torque_balance(
        rope, tensions_N=[1.0e6], youngs_modulus_Pa={"core": core_scale * e}
    )
    strain = state.axial_strain_fixed[-1] - state.axial_strain_fixed[0]
    model = (state.torque_fixed_Nm[-1] - state.torque_fixed_Nm[0]) / strain
    rc = FEYRER_D0 / 2
    knapp = 0.0
    rigid = 0.0
    for layer in rope.helical_layers:
        assert layer.n_wires and layer.d_wire_m and layer.r_helix_m
        a = math.radians(layer.lay_angle_deg or 0.0)
        s = 1.0 if layer.lay_direction == "right_hand" else -1.0
        base = s * layer.n_wires * e * math.pi * layer.d_wire_m**2 / 4
        term = base * layer.r_helix_m * math.sin(a) * math.cos(a) ** 2
        rigid += term
        knapp += term * (1 - 2 * nu * rc * math.tan(a) ** 2 / (2 * layer.r_helix_m))
    return {"model_Nm": model, "knapp_Nm": knapp, "rigid_Nm": rigid}


def kim_pairs() -> list[dict[str, float]]:
    """Outer helix angle that balances each inner angle, against Kim and Lee."""
    rows = []
    for alpha_1, alpha_2 in KIM_PAIRS:
        cable = kim_strand(alpha_1, alpha_2)
        design = design_torque_balance(
            cable, "outer_lay_angle", tension_N=KIM_TENSION_N, bounds=(0.2, 20.0)
        )
        rows.append(
            {
                "alpha_1": alpha_1,
                "alpha_2_source": alpha_2,
                "alpha_2_model": 90.0 - design.value,
                "difference_deg": (90.0 - design.value) - alpha_2,
                "coefficient_m": design.coefficient_m,
                "fill": design.fill_ratio,
            }
        )
    return rows


def utting_jones() -> list[dict[str, float]]:
    """Fixed-end torque per kN of the six seven-wire strands."""
    rows = []
    for name, (lay, md, test) in UJ_STRANDS.items():
        result = torque_balance(uj_strand(lay), tensions_N=[UJ_TENSION_N])
        model = result.torque_coefficient_m * 1e3  # N m per kN
        rows.append(
            {
                "strand": float(list(UJ_STRANDS).index(name) + 1),
                "model_Nm_per_kN": model,
                "md_Nm_per_kN": md,
                "test_Nm_per_kN": test,
                "model_over_md": model / md,
                "model_over_test": model / test,
            }
        )
    return rows


def mirror_lay(cable: CableData) -> CableData:
    """The same cable with every lay direction reversed (cores included)."""
    layers = []
    for layer in cable.layers:
        if isinstance(layer, HelicalLayer):
            flipped = (
                "left_hand" if layer.lay_direction == "right_hand" else "right_hand"
            )
            layers.append(layer.model_copy(update={"lay_direction": flipped}))
        else:
            layers.append(layer)
    update: dict[str, object] = {"layers": tuple(layers)}
    if cable.core_lay is not None:
        flipped = (
            "left_hand"
            if cable.core_lay.lay_direction == "right_hand"
            else "right_hand"
        )
        update["core_lay"] = cable.core_lay.model_copy(
            update={"lay_direction": flipped}
        )
    return cable.model_copy(update=update)


def verification_checks() -> dict[str, float]:
    """Scorecard row TQ: closed forms and the mirror symmetry."""
    fey = feyrer()
    knapp = knapp_core_factor()
    rope = feyrer_rope()
    a = torque_balance(rope, tensions_N=[10.0e3])
    b = torque_balance(mirror_lay(rope), tensions_N=[10.0e3])
    mirror = max(
        abs(a.torque_fixed_Nm[-1] + b.torque_fixed_Nm[-1]) / abs(a.torque_fixed_Nm[-1]),
        abs(a.twist_free_rad_per_m[-1] + b.twist_free_rad_per_m[-1])
        / abs(a.twist_free_rad_per_m[-1]),
    )
    return {
        "feyrer_c1S": fey["c1S_fixed_radius"],
        "feyrer_error": fey["c1S_fixed_radius"] / FEYRER_C1S - 1.0,
        "knapp_error": knapp["model_Nm"] / knapp["knapp_Nm"] - 1.0,
        "mirror": mirror,
    }


def published_checks() -> dict[str, float]:
    """Scorecard rows TQ-KL and TQ-UJ."""
    kim = [row["difference_deg"] for row in kim_pairs()]
    uj = utting_jones()
    return {
        "kim_min_deg": min(kim),
        "kim_max_deg": max(kim),
        "uj_md_min": min(row["model_over_md"] for row in uj),
        "uj_md_max": max(row["model_over_md"] for row in uj),
        "uj_test_min": min(row["model_over_test"] for row in uj),
        "uj_test_max": max(row["model_over_test"] for row in uj),
    }


# --------------------------------------------------------------------------- #
# Twist-rate stress of the fatigue path (row TQ-FT)                           #
# --------------------------------------------------------------------------- #
FT_TENSION_N = 60.0e3
FT_TWIST = 1.0e-3  # rad/m, amplitude of the twist history
FT_SAMPLES = 401


def _rope_fatigue_layers(cable: CableData) -> list[FatigueLayer]:
    """Fatigue layers of a strand straight from its schema data."""
    curve = get_sn_curve(SN_CURVE)
    out = []
    for layer in cable.helical_layers:
        assert layer.n_wires and layer.r_helix_m and layer.d_wire_m
        spec = HelicalLayerSpec(
            n_wires=layer.n_wires,
            r_helix_m=layer.r_helix_m,
            lay_angle_rad=math.radians(abs(layer.lay_angle_deg or 0.0)),
            E_wire_Pa=library.get_property(layer.material, "E_Pa"),
            A_wire_m2=math.pi * layer.d_wire_m**2 / 4,
            xi=1.0,
            mu=0.0,
            F_n_axi_N=0.0,
        )
        out.append(
            FatigueLayer(
                spec=spec,
                sn_bulk=curve,
                sigma_UTS_Pa=1.0e9,
                name=layer.name,
                lay_hand=1 if layer.lay_direction == "right_hand" else -1,
            )
        )
    return out


def _closed_form_stress(
    layer: HelicalLayer, k_et: float, k_ee: float, tension: float, tau: np.ndarray
) -> np.ndarray:
    """Knapp (1981) Eqs. 2, 11 (theta = 0) with eps = (T - K_et tau) / K_ee."""
    e = library.get_property(layer.material, "E_Pa")
    a = math.radians(abs(layer.lay_angle_deg or 0.0))
    s = 1.0 if layer.lay_direction == "right_hand" else -1.0
    assert layer.r_helix_m is not None
    eps = (tension - k_et * tau) / k_ee
    helix = s * layer.r_helix_m * math.sin(a) * math.cos(a)
    return np.asarray(e * (math.cos(a) ** 2 * eps + helix * tau))


def fatigue_torsion_checks() -> dict[str, float]:
    """Scorecard row TQ-FT: the twist-rate wire stress of the fatigue path.

    * ``closed_form``: largest relative error of the recovered wire stress of
      the 66 kV template armour (adapter-built layers) under tension and a
      twist history without bending, against the closed form from the schema
      data; ``damage`` the same for the per-wire damage of
      :func:`cablexsec.compute_fatigue` against the Miner sum of the
      closed-form history.
    * ``torque``, ``force``: the Feyrer 1x37 rope, whose helical layers alone
      carry ``K_et`` and ``K_tt``: torque ``sum s n A sigma R sin(a)`` and
      axial force ``sum n A sigma cos(a) + E A_core eps`` of the fatigue
      stresses at ``(T, tau)`` against ``K_et eps + K_tt tau`` and ``T``.
    * ``mirror``: every lay reversed and the twist negated, bending included:
      largest relative difference of the per-wire damage.
    * ``zero``: zero twist against no twist input: largest absolute damage
      difference, armour fibres and conductor azimuths (bit for bit: 0).
    """
    t = np.linspace(0.0, 40.0, FT_SAMPLES)
    tau = FT_TWIST * (0.3 + np.sin(2 * math.pi * t / 10.0))
    zero_k = np.zeros_like(t)
    tension = np.full_like(t, FT_TENSION_N)
    cable = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
    m = axial_torsional_matrix(cable)
    layers = cx.cable_to_fatigue_layers(cable, contact_tension_N=FT_TENSION_N)
    exact = {
        layer.name: _closed_form_stress(
            cable.get_layer(layer.name),  # type: ignore[arg-type]
            m.K_et,
            m.K_ee,
            FT_TENSION_N,
            tau,
        )
        for layer in layers
    }
    closed = 0.0
    for layer in layers:
        field = wire_stress_field(
            layer,
            zero_k,
            zero_k,
            tension,
            m.K_ee,
            twist_rate_per_m=tau,
            axial_twist_coupling_Nm=m.K_et,
        )
        ref = exact[layer.name]
        error = np.max(np.abs(field.mean_Pa - ref)) / np.max(np.abs(ref))
        closed = max(closed, float(error))
    result = cx.compute_fatigue(
        cable,
        zero_k,
        zero_k,
        tension_N=tension,
        twist_rate_per_m=tau,
        friction_model="direction_tied",
    )
    damage = 0.0
    for layer in layers:
        ref_damage = miner_damage(exact[layer.name], layer.sn_bulk, layer.sigma_UTS_Pa)
        got = result.D_bulk_per_wire[layer.name]
        damage = max(damage, float(np.max(np.abs(got - ref_damage))) / ref_damage)

    rope = feyrer_rope()
    mr = axial_torsional_matrix(rope)
    t0, tau0 = 10.0e3, 2.0e-3
    eps = (t0 - mr.K_et * tau0) / mr.K_ee
    torque = 0.0
    force = library.get_property(STEEL, "E_Pa") * math.pi * FEYRER_D0**2 / 4 * eps
    for fl in _rope_fatigue_layers(rope):
        spec = fl.spec
        a = spec.lay_angle_rad
        sigma = spec.E_wire_Pa * math.cos(a) ** 2 * t0 / mr.K_ee + tau0 * (
            twist_stress_coefficient(
                fl, axial_twist_coupling_Nm=mr.K_et, EA_full_N=mr.K_ee
            )
        )
        area = spec.n_wires * spec.A_wire_m2
        torque += fl.lay_hand * area * sigma * spec.r_helix_m * math.sin(a)
        force += area * sigma * math.cos(a)
    torque_ref = mr.K_et * eps + mr.K_tt * tau0

    kx = 0.03 * np.sin(2 * math.pi * t / 7.0)
    ky = 0.01 * np.cos(2 * math.pi * t / 5.0)
    options = {"tension_N": tension, "friction_model": "direction_tied"}
    base = cx.compute_fatigue(cable, kx, ky, twist_rate_per_m=tau, **options)
    flipped = cx.compute_fatigue(
        mirror_lay(cable), kx, ky, twist_rate_per_m=-tau, **options
    )
    mirror = max(
        float(np.max(np.abs(flipped.D_bulk_per_wire[k] - v)) / np.max(v))
        for k, v in base.D_bulk_per_wire.items()
    )
    none = cx.compute_fatigue(cable, kx, ky, **options)
    zeros = cx.compute_fatigue(cable, kx, ky, twist_rate_per_m=zero_k, **options)
    zero = max(
        float(np.max(np.abs(zeros.D_bulk_per_fibre[k] - v)))
        for k, v in none.D_bulk_per_fibre.items()
    )
    conductor = cx.iea_task49_66kv_stranded()
    c_none = cx.compute_conductor_fatigue(conductor, kx, ky, tension_N=tension)
    c_zero = cx.compute_conductor_fatigue(
        conductor, kx, ky, tension_N=tension, twist_rate_per_m=zero_k
    )
    difference = c_zero.D_bulk_per_orientation - c_none.D_bulk_per_orientation
    zero = max(zero, float(np.max(np.abs(difference))))
    return {
        "closed_form": closed,
        "damage": damage,
        "torque": abs(torque / torque_ref - 1.0),
        "force": abs(force / t0 - 1.0),
        "mirror": mirror,
        "zero": zero,
    }


def main() -> None:
    print("Feyrer (2015) Example 2.8")
    for key, value in feyrer().items():
        print(f"  {key:18s} {value:.5f}")
    print("Knapp (1981) core factor, theta = 2 nu")
    for key, value in knapp_core_factor().items():
        print(f"  {key:10s} {value:.6g}")
    print("Kim and Lee (2017) balance pairs [deg]")
    for row in kim_pairs():
        print(
            f"  a1 {row['alpha_1']:6.2f}  source a2 {row['alpha_2_source']:6.2f}  "
            f"model {row['alpha_2_model']:7.3f}  diff {row['difference_deg']:+.3f}  "
            f"fill {row['fill']:.3f}"
        )
    print("Utting and Jones (1987) fixed-end torque [N m/kN]")
    for row in utting_jones():
        print(
            f"  strand {int(row['strand'])}  model {row['model_Nm_per_kN']:.4f}  "
            f"M-D {row['md_Nm_per_kN']:.3f}  test {row['test_Nm_per_kN']:.2f}  "
            f"model/M-D {row['model_over_md']:.4f}  model/test "
            f"{row['model_over_test']:.4f}"
        )
    print("Mirror lay symmetry")
    rope = feyrer_rope()
    a = torque_balance(rope, tensions_N=[10.0e3])
    b = torque_balance(mirror_lay(rope), tensions_N=[10.0e3])
    print(
        f"  torque {a.torque_fixed_Nm[-1]:.6g} / {b.torque_fixed_Nm[-1]:.6g}; "
        f"twist {a.twist_free_rad_per_m[-1]:.6g} / {b.twist_free_rad_per_m[-1]:.6g}"
    )
    print("Twist-rate wire stress of the fatigue path (TQ-FT)")
    for key, value in fatigue_torsion_checks().items():
        print(f"  {key:12s} {value:.3g}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
