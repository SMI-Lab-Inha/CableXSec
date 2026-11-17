# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Published comparisons of the solved radial section (doc/radial_model.md).

* Knapp (1979), Tables IV and VI: closed-form tension-torsion stiffness of two
  armoured cables with a compressible core (independent derivation).
* Su, Zhao and Zhu (2026), 35 kV three-core cable, Tables 1, 6, 7 and 10:
  three-dimensional FE and test axial stiffness, FE torsional stiffness and
  the analytical interface-pressure pattern.
* Ménard and Cartraud (2023), Sec. 6.4: printed inner-sheath line load at
  the calibrated 0.3 MPa state. The source filler geometry is not tabulated,
  so only a bracket by the two filler representations is checked.
* Sævik and Bruaseth (2005) / Sævik and Gjøsteen (2012) umbilical axial
  stiffness (BM4): the element tables carry no radial data; a radial
  foundation under a unilateral layer stack is identified, not predicted.
"""

from __future__ import annotations

import math
from dataclasses import replace
from functools import lru_cache

import pytest

import cablexsec as cx
from cablexsec.mechanics.concentric_section import ElasticAnnulus, ElasticHelicalRing
from cablexsec.mechanics.radial_cable import build_radial_section
from cablexsec.mechanics.radial_solution import AxialMember, RadialSection
from cablexsec.mechanics.trefoil_section import ElasticFiller, TrefoilBundle

IN = 0.0254
PSI = 6894.757
LB = 4.448222
MPA = 1e6


def _knapp(core, inner, outer, wire_e_psi):
    r_c, r_cc, a_c, e_c, nu_c = core
    e_c *= PSI
    rings = []
    for name, (n, area, radius, angle) in (("inner", inner), ("outer", outer)):
        d = math.sqrt(4 * area / math.pi) * IN
        rings.append(
            ElasticHelicalRing(
                name, radius * IN, d, n, math.radians(angle), wire_e_psi * PSI
            )
        )
    # Knapp (1979): no wire flattening, the layers rest on each other and on
    # the core (Eq. 40). Place the outer layer on the inner one.
    outer_ring = rings[1]
    rings[1] = replace(
        outer_ring,
        helix_radius_m=rings[0].outer_radius_m + outer_ring.wire_diameter_m / 2,
    )
    core_outer = rings[0].inner_radius_m
    annulus = ElasticAnnulus("core", r_cc * IN, core_outer, e_c, nu_c)
    core_axial = a_c * IN**2 * e_c - e_c * math.pi * (core_outer**2 - (r_cc * IN) ** 2)
    section = RadialSection(
        [annulus, *rings],
        members=[AxialMember("copper", core_axial)],
        rigid_bore=r_cc > 0,
    )
    k = section.solve(tension_N=1e3).tangent
    return k[0, 0] / LB, k[0, 1] / LB / IN, k[1, 1] / LB / IN**2


@pytest.mark.independent_method
@pytest.mark.parametrize(
    ("core", "inner", "outer", "wire_e", "table"),
    [
        (  # EM cable, Table IV; Table VI k11, k12, k21, k22
            (0.22, 0.058, 6.165, 23000, 0.49),
            (24, 2.552e-3, 0.2485, 20.6),
            (24, 3.960e-3, 0.3125, -19.0),
            29e6,
            (3.293e6, -1.096e5, -1.078e5, 4.049e4),
        ),
        (  # Logging cable, Table V
            (0.144, 0.0, 1.841, 23000, 0.49),
            (24, 1.195e-3, 0.1635, 21.6),
            (24, 1.886e-3, 0.2075, -21.0),
            26e6,
            (1.260e6, -3.385e4, -3.277e4, 8.961e3),
        ),
    ],
)
def test_knapp_1979_compressible_core_stiffness(core, inner, outer, wire_e, table):
    """Energy-consistent model against Knapp's non-symmetric closed form.

    Knapp's matrix is non-symmetric (k12 != k21); the symmetric coupling here
    lies between his two values. Axial and torsional terms agree within 6 %
    (the rigid-core k11 is 16-21 % higher, so the core compliance is
    captured).
    """
    k11, k12, k22 = _knapp(core, inner, outer, wire_e)
    assert k11 == pytest.approx(table[0], rel=0.06)
    assert k22 == pytest.approx(table[3], rel=0.05)
    assert min(table[1], table[2]) <= k12 <= max(table[1], table[2])


def _su35(filler: bool = True, cells: int = 48) -> RadialSection:
    """Su, Zhao and Zhu (2026) Table 1 (35 kV), cores straight (Sec. 2)."""
    core = (
        ElasticAnnulus("conductor", 0.0, 0.005, 120e3 * MPA, 0.33),
        ElasticAnnulus("insulation", 0.005, 0.0171, 100 * MPA, 0.2),
        ElasticAnnulus("tape", 0.0171, 0.01925, 350 * MPA, 0.4),
        ElasticAnnulus("core_sheath", 0.01925, 0.02125, 932 * MPA, 0.4),
    )
    a = 0.02125
    ring = (
        ElasticAnnulus("binder", 0.0458, 0.04665, 512 * MPA, 0.4),
        ElasticAnnulus("sheath", 0.04665, 0.05015, 932 * MPA, 0.4),
    )
    bundle = TrefoilBundle(
        core,
        2 * a / math.sqrt(3),
        ring,
        filler=ElasticFiller("filler", 3000 * MPA, 0.4) if filler else None,
        angular_cells=cells,
    )
    lay = math.radians(15)
    steel = 210e3 * MPA
    layers = [
        ElasticHelicalRing("armour1", 0.05215, 0.004, 72, lay, steel),
        ElasticHelicalRing("armour2", 0.05615, 0.004, 78, -lay, steel),
        ElasticAnnulus("binder2", 0.05815, 0.0589, 512 * MPA, 0.4),
        ElasticHelicalRing("armour3", 0.0609, 0.004, 84, lay, steel),
        ElasticHelicalRing("armour4", 0.0649, 0.004, 90, -lay, steel),
        ElasticAnnulus("binder3", 0.0669, 0.06765, 512 * MPA, 0.4),
        ElasticAnnulus("outer_sheath", 0.06765, 0.07365, 932 * MPA, 0.4),
    ]
    return RadialSection(layers, bundle=bundle)


@lru_cache(maxsize=2)
def _su35_cached(filler: bool) -> RadialSection:
    return _su35(filler)


@pytest.mark.full_scale
def test_su_2026_35kv_axial_and_torsional_stiffness():
    """Table 6/7: FE EA 615.4 MN (test 631.7), compression 329.7 MN,
    GJ 209.5 (clockwise) and 201.2 kN m² (anticlockwise).

    FE end conditions (Table 5): twist fixed in tension, axial displacement
    fixed in torsion. Secant values at 500 kN and 5000 N m as in Table 3.
    """
    section = _su35_cached(True)
    tension = section.solve(tension_N=500e3)
    compression = section.solve(tension_N=-500e3)
    ea = 500e3 / tension.axial_strain
    assert ea == pytest.approx(615.4e6, rel=0.02)
    assert ea == pytest.approx(631.7e6, rel=0.05)
    assert -500e3 / compression.axial_strain == pytest.approx(329.7e6, rel=0.12)
    gj = sorted(
        abs(
            5000
            / section.solve(
                torque_Nm=sign * 5000,
                prescribed_twist_per_m=None,
                prescribed_axial_strain=0.0,
            ).twist_per_m
        )
        for sign in (1, -1)
    )
    # Stiffer direction against the FE clockwise value, softer against the
    # anticlockwise value (test, Table 6: 209.8 and 201.3). The filler lobes
    # twist as separate bodies: -4.8 % and -12.0 %. The band of the softer
    # direction was 12 % when the filler added its polar moment about the
    # cable axis (-10.3 %); that bonded term is not supported by the
    # frictionless filler contacts, and Su's own model without the filler
    # gives -8.4 %, so the remaining gap is not the filler.
    assert gj[1] == pytest.approx(209.5e3, rel=0.05)
    assert gj[0] == pytest.approx(201.2e3, rel=0.13)
    # The no-radial-interaction bound is about 30 % stiffer than the FE.
    rigid = 0.0
    for layer in section.layers:
        if isinstance(layer, ElasticHelicalRing):
            rigid += (
                layer.wire_count
                * math.pi
                * layer.wire_diameter_m**2
                / 4
                * layer.youngs_modulus_Pa
                * math.cos(layer.signed_lay_angle_rad) ** 3
            )
    assert rigid > 1.2 * 615.4e6


@pytest.mark.full_scale
def test_su_2026_35kv_interface_pattern_at_100kN():
    """Table 10 (analytical model of the source, 100 kN): contact on every
    armour interface, separation between armour 4 and its binder; pressures
    of the armour interfaces within 20 %."""
    section = _su35_cached(True)
    result = section.solve(tension_N=100e3, prescribed_twist_per_m=0.0)
    pressure = {
        pair: force / (2 * math.pi * radius) / 1e3
        for pair, force, radius in zip(
            result.interface_names,
            result.interface_line_force_N_per_m,
            result.interface_radius_m,
            strict=True,
        )
    }
    source = {  # kPa, Table 10 rows L7-L8 ... L11-L12
        ("bundle", "armour1"): 432.8,
        ("armour1", "armour2"): 283.9,
        ("armour2", "binder2"): 173.3,
        ("binder2", "armour3"): 176.4,
        ("armour3", "armour4"): 74.5,
    }
    for pair, value in source.items():
        assert pressure[pair] == pytest.approx(value, rel=0.2), pair
    assert pressure[("armour4", "binder3")] == 0.0
    assert (
        result.interface_gap_m[
            list(result.interface_names).index(("armour4", "binder3"))
        ]
        > 0
    )


@pytest.mark.full_scale
def test_su_2026_filler_representation_is_first_order():
    """Removing the interstitial filler halves the axial stiffness."""
    void = _su35_cached(False).solve(tension_N=500e3)
    assert 500e3 / void.axial_strain < 0.6 * 615.4e6


@pytest.mark.full_scale
def test_menard_2023_inner_sheath_load_is_bracketed_by_filler_representations():
    """Sec. 6.4: mean inner-sheath line load close to 46 kN/m at P = 0.3 MPa.

    The source FE contains PP fillers of untabulated shape. Void interstices
    and fully filled outer interstices must bracket the printed value at
    fixed extension and twist (Ménard 2020, Sec. 3.2.6).
    """
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    section, _ = build_radial_section(cable)
    loads = []
    for filler in (None, ElasticFiller("pp", 1300 * MPA, 0.4)):
        bundle = replace(section.bundle, filler=filler)
        solved = RadialSection(section.layers, bundle=bundle).solve(
            external_pressure_Pa=0.3e6, prescribed_axial_strain=0.0
        )
        loads.append(solved.interface_line_force_N_per_m[0])
    assert loads[0] < 46e3 < loads[1]


@pytest.mark.independent_method
def test_bm4_common_stack_identification_is_not_a_prediction():
    """BM4: the rigid-foundation stack equals the no-interaction assembler;
    the foundation stiffness needed to match each measurement differs by
    orders of magnitude between umbilicals (the source tables carry no
    layer-wise radial data), so BM4 is not closed by this model."""
    from cablexsec.mechanics.element_stiffness import ea_fixed_from_elements
    from tests.test_bm4_saevik_bruaseth import _MEAS_FIXED_MN, _POP, _RAW

    def stack(rows, foundation):
        rings, members = [], []
        for i, (r_mm, pitch_mm, ea_mn, _gj, _ei) in enumerate(rows):
            if pitch_mm == 0:
                members.append(AxialMember(f"s{i}", ea_mn * 1e6))
                continue
            radius = r_mm / 1000
            angle = math.atan(2 * math.pi * radius / abs(pitch_mm / 1000))
            area = 1e-6
            rings.append(
                (radius, angle * math.copysign(1, pitch_mm), ea_mn * 1e6 / area)
            )
        rings.sort()
        layers = []
        outer = None
        for i, (radius, angle, modulus) in enumerate(rings):
            d = math.sqrt(4e-6 / math.pi)
            centre = radius if outer is None else outer + d / 2
            layers.append(ElasticHelicalRing(f"h{i}", centre, d, 1, angle, modulus))
            outer = centre + d / 2
        kwargs = (
            {"rigid_bore": True}
            if foundation is None
            else {"bore_foundation_Pa_per_m": foundation}
        )
        return (
            RadialSection(layers, members=members, **kwargs)
            .solve(tension_N=1e5)
            .ea_fixed_twist_N
        )

    for u in (4, 6, 8):
        assert stack(_RAW[u], None) == pytest.approx(
            ea_fixed_from_elements(_POP[u]), rel=0.02
        )
    # The unilateral stack can represent umbilical 1 (609 MN lies between
    # the unsupported and rigid limits), but no common foundation stiffness
    # fits the population: U1 needs more than 1e9 Pa/m, U8 less.
    assert stack(_RAW[1], 1e3) < _MEAS_FIXED_MN[1] * 1e6 < stack(_RAW[1], None)
    assert stack(_RAW[1], 1e9) < _MEAS_FIXED_MN[1] * 1e6
    assert stack(_RAW[8], 1e9) > _MEAS_FIXED_MN[8] * 1e6
