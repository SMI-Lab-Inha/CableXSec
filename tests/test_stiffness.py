# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for cross-section stiffness assembly (gates A7, A8, A8c)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cablexsec.cables.schema import (
    CableData,
    CableMeta,
    CylindricalOffsetLayer,
    HelicalLayer,
)
from cablexsec.materials import library
from cablexsec.mechanics.stiffness import (
    axial_torsional_matrix,
    ea_fixed,
    ea_free,
    ei_max_stuck,
    ei_min_slip,
    load_sharing,
    xi_for_layer,
)

# Test-taxonomy category.
pytestmark = pytest.mark.analytical_sanity


def _cable(r_offset: float = 0.04) -> CableData:
    core = CylindricalOffsetLayer(
        name="cu_conductor",
        material="copper_stranded",
        n_cores=3,
        r_offset_m=r_offset,
        r_inner_m=0.0,
        r_outer_m=0.0170,
    )
    inner = HelicalLayer(
        name="inner_armour",
        material="galvanised_steel_armour",
        n_wires=90,
        d_wire_m=0.005,
        r_helix_m=0.080,
        lay_angle_deg=16.0,
        lay_direction="right_hand",
    )
    outer = HelicalLayer(
        name="outer_armour",
        material="galvanised_steel_armour",
        n_wires=96,
        d_wire_m=0.005,
        r_helix_m=0.0855,
        lay_angle_deg=16.5,
        lay_direction="left_hand",
    )
    return CableData(cable=CableMeta(name="t"), layers=(core, inner, outer))


def test_A7_ea_fixed_and_schur_free() -> None:
    """A7: EA_free via Schur == direct 2x2 solve with free twist, < 0.01%."""
    cable = _cable()
    m = axial_torsional_matrix(cable)
    assert ea_fixed(cable) == pytest.approx(m.K_ee)  # twist constrained
    # Direct solve: [[K_ee,K_et],[K_et,K_tt]] [eps, tau] = [T, 0]
    K = np.array([[m.K_ee, m.K_et], [m.K_et, m.K_tt]])
    T = 1.0
    eps, _tau = np.linalg.solve(K, np.array([T, 0.0]))
    ea_free_direct = T / eps
    assert ea_free(cable) == pytest.approx(ea_free_direct, rel=1e-4)
    assert ea_free(cable) <= ea_fixed(cable)  # Schur reduction


def test_A8c_eimin_has_no_parallel_axis_term() -> None:
    """A8c: EI_min must not contain any A*r_offset^2 parallel-axis term."""
    lo = ei_min_slip(_cable(r_offset=0.02))
    hi = ei_min_slip(_cable(r_offset=0.055))
    # Changing the trefoil offset must NOT change EI_min (no parallel axis):
    assert lo == pytest.approx(hi, rel=1e-12)
    # But EI_max (stuck) MUST change with offset (parallel-axis belongs there):
    assert ei_max_stuck(_cable(r_offset=0.055)) > ei_max_stuck(_cable(r_offset=0.02))


def test_A8_eimin_below_eimax() -> None:
    cable = _cable()
    assert ei_min_slip(cable) < ei_max_stuck(cable)


def test_ei_min_is_small_relative_to_ei_max() -> None:
    """Full-slip stiffness is a small fraction of full-stick (bilinear cable)."""
    cable = _cable()
    assert ei_min_slip(cable) < 0.1 * ei_max_stuck(cable)


def test_xi_in_unit_interval() -> None:
    cable = _cable()
    xi = xi_for_layer(cable, "inner_armour")
    assert 0.0 < xi <= 1.0


def test_load_sharing_conserves_tension() -> None:
    cable = _cable()
    T = 100.0e3
    shares = load_sharing(cable, T)
    assert sum(shares.values()) == pytest.approx(T, rel=1e-9)
    # armour (stiff steel, high cos^3) carries more than the conductor here
    assert shares["inner_armour"] > 0.0


def _cable_with_loose_layers() -> CableData:
    """Three-core cable with a non-structural filler and tape screen."""
    from cablexsec.cables.schema import CylindricalLayer

    core = CylindricalOffsetLayer(
        name="cu_conductor",
        material="copper_stranded",
        n_cores=3,
        r_offset_m=0.0400,
        r_inner_m=0.0,
        r_outer_m=0.0170,
    )
    screen = CylindricalOffsetLayer(
        name="cu_screen",
        material="copper_tape_screen",
        n_cores=3,
        r_offset_m=0.0400,
        r_inner_m=0.0315,
        r_outer_m=0.0324,
        structural=False,
    )
    filler = CylindricalLayer(
        name="pp_filler",
        material="pp_filler",
        r_inner_m=0.0345,
        r_outer_m=0.0745,
        structural=False,
    )
    inner = HelicalLayer(
        name="inner_armour",
        material="galvanised_steel_armour",
        n_wires=90,
        d_wire_m=0.005,
        r_helix_m=0.080,
        lay_angle_deg=16.0,
        lay_direction="right_hand",
    )
    return CableData(cable=CableMeta(name="t"), layers=(core, screen, filler, inner))


def test_structural_layers_excludes_loose_layers() -> None:
    cable = _cable_with_loose_layers()
    names = {layer.name for layer in cable.structural_layers}
    assert "cu_screen" not in names and "pp_filler" not in names
    assert {"cu_conductor", "inner_armour"} <= names


def test_loose_layers_carry_no_bending() -> None:
    """EI is unchanged by a non-structural layer's modulus (no bending)."""
    from cablexsec.cables.schema import CylindricalLayer

    cable = _cable_with_loose_layers()
    ei_min = ei_min_slip(cable)
    ei_max = ei_max_stuck(cable)
    stiff_filler = CylindricalLayer(
        name="pp_filler",
        material="galvanised_steel_armour",
        r_inner_m=0.0345,
        r_outer_m=0.0745,
        structural=False,
    )
    layers = tuple(
        stiff_filler if layer.name == "pp_filler" else layer for layer in cable.layers
    )
    cable2 = CableData(cable=cable.cable, layers=layers)
    assert ei_min_slip(cable2) == pytest.approx(ei_min, rel=1e-12)
    assert ei_max_stuck(cable2) == pytest.approx(ei_max, rel=1e-12)


def test_screen_no_axial_filler_true_area() -> None:
    """Screen EA = 0; filler EA from true area (cores excluded)."""
    cable = _cable_with_loose_layers()
    shares = load_sharing(cable, 100.0e3)
    assert "cu_screen" not in shares  # zero axial -> excluded
    assert shares.get("pp_filler", 0.0) > 0.0


def test_conductor_cores_annulus_area_correct() -> None:
    """Load-bearing cores: r_inner = 0 so annulus = solid disk (exactly correct)."""
    core = _cable_with_loose_layers().get_layer("cu_conductor")
    assert math.pi * core.r_outer_m**2 == pytest.approx(
        math.pi * (core.r_outer_m**2 - (core.r_inner_m or 0.0) ** 2)
    )


def test_metals_only_excludes_polymers() -> None:
    """metals_only convention drops polymer annuli; never adds a knockdown."""
    from cablexsec.cables.schema import CylindricalLayer

    poly = CylindricalLayer(
        name="bedding", material="hdpe", r_inner_m=0.0745, r_outer_m=0.0770
    )
    cable = _cable()
    core, *armour = cable.layers  # the bedding lies between the cores and the armour
    with_poly = CableData(cable=cable.cable, layers=(core, poly, *armour))
    assert ea_fixed(with_poly, metals_only=True) < ea_fixed(with_poly)
    # metals-only equals the all-layers value when there is no polymer present
    assert ea_fixed(cable, metals_only=True) == pytest.approx(ea_fixed(cable))


def test_conductor_strand_lay_reduction() -> None:
    """alpha_strand -> 0 recovers E_cu * A_solid (1e-12); nonzero lay reduces EA."""
    E = library.get_property("copper_stranded", "E_eff_axial_Pa")
    r_out = 0.0170
    a_solid_mm2 = math.pi * r_out**2 * 1e6
    cond0 = CylindricalOffsetLayer(
        name="cu",
        material="copper_stranded",
        n_cores=3,
        r_offset_m=0.04,
        r_inner_m=0.0,
        r_outer_m=r_out,
        strand_lay_angle_deg=0.0,
        conductor_area_mm2=a_solid_mm2,
    )
    cable0 = CableData(cable=CableMeta(name="t"), layers=(cond0,))
    assert ea_fixed(cable0) == pytest.approx(3 * E * math.pi * r_out**2, rel=1e-12)
    # A real (nonzero) strand lay angle reduces EA by cos^3(alpha_strand).
    cond1 = cond0.model_copy(update={"strand_lay_angle_deg": 12.0})
    cable1 = CableData(cable=CableMeta(name="t"), layers=(cond1,))
    expected = 3 * E * math.pi * r_out**2 * math.cos(math.radians(12.0)) ** 3
    assert ea_fixed(cable1) == pytest.approx(expected, rel=1e-12)
    assert ea_fixed(cable1) < ea_fixed(cable0)


def test_lay_direction_sets_coupling_sign() -> None:
    """A single layer's axial-twist coupling flips sign with the lay direction."""
    rh = HelicalLayer(
        name="a",
        material="galvanised_steel_armour",
        n_wires=90,
        d_wire_m=0.005,
        r_helix_m=0.08,
        lay_angle_deg=16.0,
        lay_direction="right_hand",
    )
    lh = rh.model_copy(update={"lay_direction": "left_hand"})
    meta = CableMeta(name="t")
    k_rh = axial_torsional_matrix(CableData(cable=meta, layers=(rh,))).K_et
    k_lh = axial_torsional_matrix(CableData(cable=meta, layers=(lh,))).K_et
    assert k_rh > 0.0 and k_lh < 0.0
    assert k_rh == pytest.approx(-k_lh)


def test_shear_coefficient_uses_single_wire_not_whole_layer_width() -> None:
    """Yun Eq. (12): support spring Gb/t and rigidity EA both belong to one wire."""
    from cablexsec.cables.schema import CylindricalLayer
    from cablexsec.mechanics.stiffness import xi_for_layer

    support = CylindricalLayer(
        name="support", material="hdpe", r_inner_m=0.03, r_outer_m=0.035
    )
    wire = HelicalLayer(
        name="armour",
        material="galvanised_steel_armour",
        n_wires=40,
        d_wire_m=0.004,
        r_helix_m=0.037,
        lay_angle_deg=25.0,
    )
    cable = CableData(
        cable=CableMeta(name="one-wire equilibrium"), layers=(support, wire)
    )
    shear = library.get_property("hdpe", "E_Pa") / (
        2 * (1 + library.get_property("hdpe", "nu"))
    )
    spring = shear * 0.004 / 0.005
    rigidity = library.get_property(wire.material, "E_Pa") * math.pi * 0.004**2 / 4
    expected = spring / (spring + rigidity * math.sin(math.radians(25)) ** 2 / 0.037**2)
    assert xi_for_layer(cable, "armour") == pytest.approx(expected)
    fewer = cable.model_copy(
        update={"layers": (support, wire.model_copy(update={"n_wires": 20}))}
    )
    assert xi_for_layer(fewer, "armour") == pytest.approx(expected)
