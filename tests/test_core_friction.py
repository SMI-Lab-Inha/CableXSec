# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Gap R1b: stick-slip bending law of the laid power cores and internal rings.

Verification only (limits, closed forms, energy, invariance). The published
comparisons are in ``scripts/check_core_friction.py`` and
``doc/template_66kv.md``.
"""

import math

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.api import CoreFrictionSpec
from cablexsec.cables.schema import CoreStrandRing
from cablexsec.coupling import xsec_stiffness
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.mechanics.core_friction import (
    build_core_friction,
    core_contact_loads,
    core_friction_history,
    resolve_core_friction,
)
from cablexsec.mechanics.stiffness import (
    bending_stiffness_by_layer,
    xi_for_layer,
)

pytestmark = pytest.mark.analytical_sanity

T = 50.0e3
SPEC = CoreFrictionSpec(mu_core_filler=0.46, mu_core_core=0.46, mu_core_sheath=0.46)


@pytest.fixture(scope="module")
def cable():
    return cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)


@pytest.fixture(scope="module")
def stranded(cable):
    """66 kV template with a synthetic 61-wire conductor (test geometry only)."""
    conductor = cable.layers[0]
    d = conductor.r_outer_m / 4.5
    rings = [CoreStrandRing(n_wires=1, d_wire_m=d, r_helix_m=0.0)]
    for i, n in enumerate((6, 12, 18, 24), start=1):
        r = i * d
        rings.append(
            CoreStrandRing(n_wires=n, d_wire_m=d, r_helix_m=r, pitch_m=12 * 2 * r)
        )
    layer = conductor.model_copy(
        update={
            "strand_layers": tuple(rings),
            "conductor_area_mm2": None,
            "construction": "resolved_concentric_stranded",
        }
    )
    return cable.model_copy(update={"layers": (layer, *cable.layers[1:])})


def _loop(amplitude, n=801, cycles=2):
    return amplitude * np.sin(np.linspace(0.0, 2 * cycles * math.pi, n))


def _descending_zero(k, m):
    """Moment interpolated at the last kappa = 0 crossing from above."""
    i = np.flatnonzero((k[:-1] > 0) & (k[1:] <= 0))[-1]
    w = k[i] / (k[i] - k[i + 1])
    return float(m[i] + w * (m[i + 1] - m[i]))


def test_default_path_is_unchanged(cable):
    k = _loop(0.2)
    base = cx.compute_mkappa(cable, k, tension_N=T)
    for value in (None, False):
        other = cx.compute_mkappa(cable, k, tension_N=T, core_friction=value)
        assert np.array_equal(base.moment_Nm, other.moment_Nm)
        assert base.W_diss_J_per_m == other.W_diss_J_per_m
        assert base.EI_max_Nm2 == other.EI_max_Nm2
        assert other.core_friction is None
    assert resolve_core_friction(True) == CoreFrictionSpec()
    with pytest.raises(CableInputError):
        resolve_core_friction("yes")  # type: ignore[arg-type]


@pytest.mark.parametrize("model", ["bounded", "solved"])
def test_zero_friction_recovers_armour_only_law(cable, model):
    k = _loop(0.2)
    base = cx.compute_mkappa(cable, k, tension_N=T, contact_model=model)
    zero = CoreFrictionSpec(mu_core_filler=0.0, mu_core_core=0.0, mu_core_sheath=0.0)
    res = cx.compute_mkappa(
        cable, k, tension_N=T, contact_model=model, core_friction=zero
    )
    assert np.array_equal(res.moment_Nm, base.moment_Nm)
    assert res.W_diss_J_per_m == base.W_diss_J_per_m
    stick = res.core_friction.model.core_stick_EI_Nm2
    assert res.EI_max_Nm2 == pytest.approx(base.EI_max_Nm2 + stick, rel=1e-12)


def test_infinite_friction_gives_the_no_slip_section(cable):
    """Large friction and rigid contact: the tangent is the no-slip reference."""
    rigid = CoreFrictionSpec(
        mu_core_filler=1.0e3,
        mu_core_core=1.0e3,
        mu_core_sheath=1.0e3,
        core_shear=False,
    )
    k = _loop(1.0e-6, n=201, cycles=1)
    res = cx.compute_mkappa(cable, k, tension_N=T, mu=1.0e3, core_friction=rigid)
    stiff = cx.compute_stiffness(cable)
    assert res.EI_max_Nm2 == pytest.approx(stiff.ei_max_Nm2, rel=1e-12)
    tangent = res.moment_Nm[1:] / k[1:]
    finite = np.abs(k[1:]) > 1e-9
    assert np.allclose(tangent[finite], stiff.ei_max_Nm2, rtol=1e-9)
    assert res.W_diss_J_per_m == pytest.approx(0.0, abs=1e-15)
    # Core part alone equals the plane-section core parallel axis.
    parts = bending_stiffness_by_layer(cable)
    plane = sum(p.core_parallel_axis_Nm2 for p in parts.values())
    assert res.core_friction.model.core_stick_EI_Nm2 == pytest.approx(plane, 1e-12)


def test_full_slip_core_moment_matches_closed_form(cable):
    """Half-height at kappa = 0 of the core population after full reversal.

    Signed capstan: after the reversal the cores that were on the tension
    side sit at their lower bound ``-sigma0 (1 - exp(-a psi))`` and the others
    at their upper bound ``sigma0 expm1(a psi)``, so the moment integrand is
    their half sum ``sinh(a psi) sin(psi)`` and
    ``int_0^{pi/2} sinh(a psi) sin(psi) dpsi = a cosh(a pi/2) / (1 + a^2)``
    (the same magnitude at the peak). The former symmetric cap gave
    ``(a exp(a pi/2) + 1) / (1 + a^2) - 1``.
    """
    model = build_core_friction(cable, SPEC, T, contact_model="solved")
    q = model.loads.total_N_per_m
    beta, r = model.lay_angle_rad, model.r_offset_m
    a = model.mu_effective * math.sin(beta)
    integral = a * math.cosh(a * math.pi / 2) / (1.0 + a * a)
    closed = (
        2.0
        / math.pi
        * model.n_cores
        * q
        * r
        * r
        * math.cos(beta) ** 2
        / math.sin(beta) ** 2
        * integral
    )
    k = _loop(0.2, n=4001)
    hist = core_friction_history(model, k, np.zeros_like(k))
    m = hist.core_moment_Nm[:, 0]
    assert -_descending_zero(k, m) == pytest.approx(closed, rel=5e-3)
    assert m.max() == pytest.approx(closed, rel=5e-3)
    # Small-angle form 2 n mu q r^2 cos(beta)^2 / (pi sin beta): the signed
    # bound leaves only O(a^2), cosh(a pi/2) / (1 + a^2) = 1 + (pi^2/8 - 1) a^2.
    small = 2 * model.n_cores * model.mu_effective * q * r * r
    small *= math.cos(beta) ** 2 / (math.pi * math.sin(beta))
    assert closed == pytest.approx(small, rel=a * a)


@pytest.mark.parametrize("model", ["bounded", "solved"])
def test_dissipation_is_nonnegative_and_work_balances(cable, model):
    rng = np.random.default_rng(3)
    steps = rng.normal(scale=0.01, size=600)
    kx = np.cumsum(steps)
    ky = np.cumsum(rng.normal(scale=0.01, size=600))
    core = build_core_friction(cable, SPEC, T, contact_model=model)
    hist = core_friction_history(core, kx, ky)
    friction = hist.work.friction_J_per_m
    assert friction[-1] > 0.0
    assert np.all(np.diff(friction) >= -1e-12 * friction[-1])
    assert np.max(np.abs(hist.work.balance_residual_J_per_m)) < 1e-9 * friction[-1]
    res = cx.compute_mkappa(
        cable, _loop(0.15), tension_N=T, contact_model=model, core_friction=SPEC
    )
    assert np.all(np.diff(res.work.friction_J_per_m) >= -1e-9)
    assert np.max(np.abs(res.work.balance_residual_J_per_m)) < 1e-6 * res.W_diss_J_per_m


def test_biaxial_rotation_invariance(cable, stranded):
    rng = np.random.default_rng(11)
    kx = np.cumsum(rng.normal(scale=0.01, size=300))
    ky = np.cumsum(rng.normal(scale=0.01, size=300))
    core = build_core_friction(cable, SPEC, T, contact_model="solved")
    base = core_friction_history(core, kx, ky)
    n_st = core.population.n_wires
    for angle, tol in ((2 * math.pi * 7 / n_st, 1e-9), (0.3, 1.5e-2)):
        c, s = math.cos(angle), math.sin(angle)
        rot = core_friction_history(core, c * kx - s * ky, s * kx + c * ky)
        expected = base.moment_Nm @ np.array([[c, s], [-s, c]])
        scale = np.max(np.abs(base.moment_Nm))
        assert np.max(np.abs(rot.moment_Nm - expected)) <= tol * scale
        assert rot.work.friction_J_per_m[-1] == pytest.approx(
            base.work.friction_J_per_m[-1], rel=max(tol, 1e-9) * 2
        )
    # Internal rings: rotation by one wire pitch of every ring (common 2 pi/6).
    model = build_core_friction(stranded, SPEC, T, contact_model="bounded")
    ring_base = core_friction_history(model, kx, ky).strand_moment_Nm
    angle = 2 * math.pi / 6
    c, s = math.cos(angle), math.sin(angle)
    ring_rot = core_friction_history(
        model, c * kx - s * ky, s * kx + c * ky
    ).strand_moment_Nm
    expected = ring_base @ np.array([[c, s], [-s, c]])
    assert np.max(np.abs(ring_rot - expected)) <= 1e-9 * np.max(np.abs(ring_base))


def test_contact_loads(cable):
    state = cx.solve_cable_radial_state(cable, T)
    loads = core_contact_loads(cable, T, radial_state=state)
    bundle = state.result.bundle
    assert loads.source == "solved"
    assert loads.core_filler_N_per_m == bundle.filler_core_force_N_per_m
    assert loads.core_sheath_N_per_m == bundle.core_ring_force_N_per_m
    assert loads.core_core_N_per_m == 2 * bundle.core_core_force_N_per_m
    mixed = CoreFrictionSpec(mu_core_filler=0.4, mu_core_sheath=0.1)
    weighted = (0.4 * loads.core_filler_N_per_m + 0.1 * loads.core_sheath_N_per_m) / (
        loads.core_filler_N_per_m + loads.core_sheath_N_per_m
    )
    assert loads.mu_effective(mixed) == pytest.approx(weighted)
    # Solved: one profile per interstice shares the filler--ring force of the
    # ring and the filler--core forces of all cores.
    assert loads.filler_profile_N_per_m == pytest.approx(
        (bundle.filler_ring_force_N_per_m + 3 * bundle.filler_core_force_N_per_m) / 3,
        rel=1e-12,
    )
    bounded = core_contact_loads(cable, T)
    assert bounded.source == "bounded" and bounded.core_core_N_per_m > 0.0
    assert core_contact_loads(cable, 0.0).total_N_per_m == 0.0
    assert core_contact_loads(cable, 0.0).mu_effective(mixed) == 0.4
    # Without a filler the ring share bears on the sheath.
    menard = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    split = core_contact_loads(menard, T)
    assert split.core_core_N_per_m > 0.0 and split.core_sheath_N_per_m > 0.0
    assert split.core_filler_N_per_m == 0.0 and split.filler_profile_N_per_m == 0.0
    section = cx.solve_section_state(cable, tension_N=T, contact_model="bounded")
    assert core_contact_loads(cable, section_state=section) == bounded
    with pytest.raises(CableSchemaError):
        core_contact_loads(cable, T, radial_state=state, section_state=section)
    with pytest.raises(CableSchemaError):
        core_contact_loads(cable, T, contact_model="other")
    # The solved state sets its own core support.
    with pytest.raises(CableInputError, match="central_filler"):
        core_contact_loads(cable, T, radial_state=state, central_filler=False)
    with pytest.raises(CableInputError, match="central_filler"):
        build_core_friction(
            cable, CoreFrictionSpec(central_filler=True), T, contact_model="solved"
        )


def _bounded_parts(cable, tension):
    """Capstan load C and ring share R/n of one core in the bounded chain."""
    from cablexsec.mechanics.axisymmetric import compute_contact_forces
    from cablexsec.mechanics.core_friction import _core_geometry, _core_layers

    n_cores, r_c, beta, _ = _core_geometry(cable)
    axi = compute_contact_forces(
        cable, tension, p_ext_Pa=cable.initial_state.external_pressure_Pa
    )
    names = {x.name for x in _core_layers(cable)}
    share = sum(v for k, v in axi.layer_tension_N.items() if k in names) / n_cores
    capstan = share * math.tan(beta) ** 2 / r_c
    ring = (
        2 * math.pi * axi.interface_radii_m[0] * axi.interface_pressure_Pa[0] / n_cores
    )
    return n_cores, capstan, ring


@pytest.mark.parametrize("path", ["iea", "menard"])
@pytest.mark.parametrize("central", [None, False, True])
def test_bounded_core_load_statics(cable, path, central):
    """Radial equilibrium of one core in the bounded estimate.

    Inward capstan C and ring share R/n on the outer face; the inward
    resultant is reacted by the two neighbour contacts at pi/2 - pi/n off the
    radial direction (wedged trefoil), or radially by a central filler.
    """
    target = (
        cable
        if path == "iea"
        else cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    )
    n, capstan, ring = _bounded_parts(target, T)
    assert capstan > 0.0 and ring > 0.0
    loads = core_contact_loads(target, T, central_filler=central)
    filled = path == "iea"
    outer = loads.core_filler_N_per_m if filled else loads.core_sheath_N_per_m
    if central:
        assert loads.core_core_N_per_m == 0.0
        if filled:
            assert outer == pytest.approx(2 * ring + capstan, rel=1e-12)
        else:
            assert loads.core_filler_N_per_m == pytest.approx(ring + capstan, rel=1e-12)
            assert loads.core_sheath_N_per_m == pytest.approx(ring, rel=1e-12)
    else:
        assert outer == pytest.approx(ring, rel=1e-12)
        # The radial components of the two neighbour contacts balance C + R/n.
        radial = loads.core_core_N_per_m * math.cos(math.pi / 2 - math.pi / n)
        assert radial == pytest.approx(capstan + ring, rel=1e-12)
    assert loads.filler_profile_N_per_m == (
        pytest.approx(2 * ring, rel=1e-12) if filled else 0.0
    )


def test_bounded_capstan_load_and_trefoil_contacts():
    """Capstan load T tan^2(beta)/r_c per core, carried by two contacts at 30 deg."""
    from cablexsec.mechanics.axisymmetric import compute_contact_forces
    from cablexsec.mechanics.core_friction import _core_geometry, _core_layers

    menard = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    n_cores, r_c, beta, _ = _core_geometry(menard)
    axi = compute_contact_forces(
        menard, T, p_ext_Pa=menard.initial_state.external_pressure_Pa
    )
    names = {x.name for x in _core_layers(menard)}
    share = sum(v for k, v in axi.layer_tension_N.items() if k in names) / n_cores
    capstan = share * math.tan(beta) ** 2 / r_c
    ring = (
        2 * math.pi * axi.interface_radii_m[0] * axi.interface_pressure_Pa[0] / n_cores
    )
    loads = core_contact_loads(menard, T)
    assert n_cores == 3
    # The two neighbour contacts lie 30 deg off the radial direction and
    # carry the capstan load and the ring share of the core.
    assert loads.core_core_N_per_m == pytest.approx(
        (capstan + ring) / math.sin(math.pi / 3), rel=1e-12
    )
    assert loads.core_core_N_per_m * math.cos(math.pi / 6) == pytest.approx(
        capstan + ring
    )


def test_bounded_loads_include_armour_winding_stress(cable):
    from cablexsec.mechanics.axisymmetric import compute_winding_contact_pressure

    sigma = 40.0e6
    layers = tuple(
        x.model_copy(update={"winding_stress_Pa": sigma})
        if x.name in {"inner_armour", "outer_armour"}
        else x
        for x in cable.layers
    )
    wound = cable.model_copy(update={"layers": layers})
    loads = core_contact_loads(wound, 0.0)
    p12 = compute_winding_contact_pressure(wound, sigma)
    inner = next(x for x in wound.layers if x.name == "inner_armour")
    expected = (
        2 * math.pi * inner.r_helix_m * (p12["inner_armour"] - p12["outer_armour"])
    )
    assert loads.core_filler_N_per_m == pytest.approx(expected / 3, rel=1e-12)
    assert loads.core_core_N_per_m == pytest.approx(
        expected / 3 / math.sin(math.pi / 3), rel=1e-12
    )


def test_section_state_path(cable):
    section = cx.solve_section_state(cable, tension_N=T, contact_model="bounded")
    k = _loop(0.1)
    a = cx.compute_mkappa(cable, k, section_state=section, core_friction=SPEC)
    b = cx.compute_mkappa(cable, k, tension_N=T, core_friction=SPEC)
    assert a.core_friction.model.loads == b.core_friction.model.loads


def test_stick_coefficient(cable):
    rigid = build_core_friction(
        cable, CoreFrictionSpec(core_shear=False), T, contact_model="bounded"
    )
    assert rigid.xi == 1.0
    penalty = build_core_friction(
        cable,
        CoreFrictionSpec(core_shear=False, interface_stiffness_Pa_per_m=1.2e12),
        T,
    )
    k_g = 1.2e12 * 2 * math.pi * 0.0352
    coupling = penalty.EA_core_N * math.sin(penalty.lay_angle_rad) ** 2
    coupling /= penalty.r_offset_m**2
    assert penalty.xi == pytest.approx(k_g / (k_g + coupling), rel=1e-12)
    full = build_core_friction(cable, CoreFrictionSpec(), T)
    assert 0.0 < full.xi < penalty.xi < 1.0
    assert full.core_stick_EI_Nm2 == pytest.approx(
        full.xi * full.plane_section_EI_Nm2, rel=1e-12
    )


def test_internal_ring_limits(stranded):
    k = _loop(0.1)
    none = CoreFrictionSpec(strand_slip=False)
    free = CoreFrictionSpec(mu_strand=0.0)
    a = cx.compute_mkappa(stranded, k, tension_N=T, core_friction=none)
    b = cx.compute_mkappa(stranded, k, tension_N=T, core_friction=free)
    assert not a.core_friction.model.strand_rings
    assert len(b.core_friction.model.strand_rings) == 4
    assert np.array_equal(b.core_friction.strand_moment_Nm, np.zeros((k.size, 2)))
    assert np.allclose(b.moment_Nm, a.moment_Nm, rtol=0, atol=1e-9)
    # Stuck rings: the internal plane-section term of the stiffness assembly.
    stuck = CoreFrictionSpec(
        mu_core_filler=1e3,
        mu_core_core=1e3,
        mu_core_sheath=1e3,
        core_shear=False,
        mu_strand=1e3,
    )
    tiny = _loop(1.0e-7, n=101, cycles=1)
    res = cx.compute_mkappa(stranded, tiny, tension_N=T, mu=1e3, core_friction=stuck)
    xi_map = {x.name: xi_for_layer(stranded, x.name) for x in stranded.helical_layers}
    parts = bending_stiffness_by_layer(stranded, xi_map).values()
    reference = sum(
        p.shear_compliant_Nm2 + p.internal_strand_parallel_axis_Nm2 for p in parts
    )
    assert res.EI_max_Nm2 == pytest.approx(reference, rel=1e-12)
    finite = np.abs(tiny) > 1e-12
    assert np.allclose(res.moment_Nm[finite] / tiny[finite], reference, rtol=1e-8)
    # Finite strand friction dissipates and balances; zero tension uses the
    # pressure-held amplitude.
    c = cx.compute_mkappa(stranded, k, tension_N=T, core_friction=CoreFrictionSpec())
    strand_work = c.W_diss_J_per_m - a.W_diss_J_per_m
    assert strand_work > 0.0
    assert np.max(np.abs(c.work.balance_residual_J_per_m)) < 1e-6 * c.W_diss_J_per_m
    zero_t = build_core_friction(stranded, CoreFrictionSpec(), 0.0)
    assert zero_t.strand_rings[0].parameters.T0_N == 1.0


def test_inputs_are_validated(cable):
    for bad in (
        {"mu_core_filler": -0.1},
        {"mu_core_core": math.nan},
        {"mu_strand": -1.0},
        {"interface_stiffness_Pa_per_m": 0.0},
        {"n_stations_per_core": 1},
        {"n_stations_per_core": True},
        {"direction_hold_fraction": 2.0},
    ):
        with pytest.raises(CableInputError):
            CoreFrictionSpec(**bad)
    with pytest.raises(CableInputError):
        build_core_friction(cable, SPEC, -1.0)
    single = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    with pytest.raises(CableSchemaError, match="two offset cores"):
        cx.compute_mkappa(single, _loop(0.05), core_friction=True)
    moved = cable.layers[1].model_copy(update={"r_offset_m": 0.05})
    split = cable.model_copy(
        update={"layers": (cable.layers[0], moved, *cable.layers[2:])}
    )
    with pytest.raises(CableSchemaError, match="one family"):
        build_core_friction(split, SPEC, T)
    straight = cx.load_cable(
        cx.inter_array_three_core_yaml_path(), geometry_override=None
    ).model_copy(update={"core_lay": None})
    with pytest.raises(CableSchemaError, match="laid core"):
        cx.compute_mkappa(straight, _loop(0.05), core_friction=True)


def test_strand_friction_needs_a_coefficient(stranded):
    from cablexsec.materials import library

    # Copper without any mu_on_cu in the core (copper_wire_screen now has one,
    # so the screen of the template takes the same stand-in).
    props = {
        k: v
        for k, v in library.get_material("copper_wire_screen").items()
        if k != "mu_on_cu"
    }
    cx.register_material("test_cu_without_mu", props)
    try:
        layers = tuple(
            x.model_copy(update={"material": "test_cu_without_mu"})
            if i == 0 or x.material == "copper_wire_screen"
            else x
            for i, x in enumerate(stranded.layers)
        )
        cable = stranded.model_copy(update={"layers": layers})
        with pytest.raises(CableInputError, match="mu_strand"):
            build_core_friction(cable, CoreFrictionSpec(), T)
        model = build_core_friction(cable, CoreFrictionSpec(mu_strand=0.3), T)
        assert model.strand_rings
    finally:
        cx.unregister_material("test_cu_without_mu")


def test_coupling_and_provenance(cable):
    plain = xsec_stiffness(cable, 0.002, tension_N=T)
    cored = xsec_stiffness(cable, 0.002, tension_N=T, core_friction=SPEC)
    assert cored.EI_max_Nm2 > plain.EI_max_Nm2
    assert cored.EI_tangent_Nm2 >= plain.EI_tangent_Nm2
    res = cx.compute_mkappa(cable, _loop(0.1), tension_N=T, core_friction=SPEC)
    assert "laid-core stick-slip" in res.provenance.method
    assert "[KarlsenHusoy2026]" in res.provenance.citations


def test_janocha_template_regression(cable):
    """Pinned values of the published comparison (doc/template_66kv.md).

    Janocha et al. (2024) Fig. 6 (50 kN): loop half-height at kappa = 0 about
    1 kN m. The model is far below it; these values pin the model, not the
    comparison.
    """
    k = _loop(0.2)
    res = cx.compute_mkappa(
        cable, k, tension_N=T, contact_model="solved", core_friction=SPEC
    )
    # 180.6 N m with the signed capstan bound (187.1 with the former cap).
    assert -_descending_zero(k, res.moment_Nm) == pytest.approx(180.6, rel=1e-2)
    assert res.EI_min_Nm2 == pytest.approx(21.0e3, rel=1e-2)


PROFILES = CoreFrictionSpec(
    mu_core_filler=0.46, mu_core_core=0.46, mu_core_sheath=0.46, filler_profiles=True
)


def test_filler_region_area_and_polar_moment():
    from cablexsec.mechanics.core_friction import _filler_region

    # Cores wholly inside the annulus: closed form.
    r_out, n, r_c, r_core = 0.08, 3, 0.04, 0.03
    area, polar = _filler_region(0.0, r_out, n, r_c, r_core)
    exact_a = math.pi * r_out**2 - n * math.pi * r_core**2
    exact_j = math.pi * r_out**4 / 2 - n * (
        math.pi * r_core**4 / 2 + math.pi * r_core**2 * r_c**2
    )
    assert area == pytest.approx(exact_a, rel=1e-9)
    assert polar == pytest.approx(exact_j, rel=1e-9)
    # Cores crossing the outer radius (the 66 kV template): a fine grid.
    r_out, r_c, r_core = 0.0759, 0.04081, 0.0352
    area, polar = _filler_region(0.0, r_out, 3, r_c, r_core)
    h = r_out / 1500
    x = np.arange(-r_out + h / 2, r_out, h)
    xx, yy = np.meshgrid(x, x)
    rr = np.hypot(xx, yy)
    inside = rr < r_out
    for k in range(3):
        cx_, cy_ = (
            r_c * math.cos(2 * math.pi * k / 3),
            r_c * math.sin(2 * math.pi * k / 3),
        )
        inside &= np.hypot(xx - cx_, yy - cy_) >= r_core
    assert area == pytest.approx(np.count_nonzero(inside) * h * h, rel=2e-3)
    assert polar == pytest.approx(np.sum(rr[inside] ** 2) * h * h, rel=2e-3)


@pytest.mark.parametrize("model", ["bounded", "solved"])
def test_filler_profile_full_slip_moment_matches_closed_form(cable, model):
    """Same signed-capstan closed form as the cores, with the profile radius,
    lay angle, both-face load and friction coefficient."""
    core = build_core_friction(cable, PROFILES, T, contact_model=model)
    f = core.fillers
    assert f is not None and f.n_profiles == 3
    assert f.line_load_N_per_m == core.loads.filler_profile_N_per_m > 0.0
    beta, r, q = f.lay_angle_rad, f.r_effective_m, f.line_load_N_per_m
    # Same pitch as the cores.
    assert math.tan(beta) / r == pytest.approx(
        math.tan(core.lay_angle_rad) / core.r_offset_m, rel=1e-12
    )
    a = f.mu * math.sin(beta)
    closed = (
        2.0 / math.pi * 3 * q * r * r * math.cos(beta) ** 2 / math.sin(beta) ** 2
    ) * (a * math.cosh(a * math.pi / 2) / (1.0 + a * a))
    k = _loop(0.2, n=4001)
    hist = core_friction_history(core, k, np.zeros_like(k))
    m = hist.filler_moment_Nm[:, 0]
    assert -_descending_zero(k, m) == pytest.approx(closed, rel=5e-3)
    assert m.max() == pytest.approx(closed, rel=5e-3)
    assert np.array_equal(
        hist.moment_Nm,
        hist.core_moment_Nm + hist.strand_moment_Nm + hist.filler_moment_Nm,
    )


@pytest.mark.parametrize("model", ["bounded", "solved"])
def test_zero_friction_with_filler_profiles_recovers_armour_only_law(cable, model):
    k = _loop(0.2)
    base = cx.compute_mkappa(cable, k, tension_N=T, contact_model=model)
    zero = CoreFrictionSpec(
        mu_core_filler=0.0,
        mu_core_core=0.0,
        mu_core_sheath=0.0,
        filler_profiles=True,
    )
    res = cx.compute_mkappa(
        cable, k, tension_N=T, contact_model=model, core_friction=zero
    )
    assert np.array_equal(res.moment_Nm, base.moment_Nm)
    assert res.W_diss_J_per_m == base.W_diss_J_per_m
    # A profile coefficient of its own: zero there too.
    other = CoreFrictionSpec(
        mu_core_filler=0.0,
        mu_core_core=0.0,
        mu_core_sheath=0.0,
        filler_profiles=True,
        mu_filler_profile=0.0,
    )
    res = cx.compute_mkappa(
        cable, k, tension_N=T, contact_model=model, core_friction=other
    )
    assert np.array_equal(res.moment_Nm, base.moment_Nm)


def test_filler_profiles_stick_limit(cable):
    """Large friction, rigid contact, no shear: the profiles add E J_f cos^3 / 2."""
    from cablexsec.materials import library

    rigid = CoreFrictionSpec(
        mu_core_filler=1.0e3,
        mu_core_core=1.0e3,
        mu_core_sheath=1.0e3,
        core_shear=False,
        filler_profiles=True,
    )
    k = _loop(1.0e-6, n=201, cycles=1)
    res = cx.compute_mkappa(cable, k, tension_N=T, mu=1.0e3, core_friction=rigid)
    f = res.core_friction.model.fillers
    assert f.xi == 1.0
    filler = next(x for x in cable.layers if x.name == f.layer_name)
    e = library.get_property(filler.material, "E_Pa")
    polar = f.area_m2 * f.r_effective_m**2
    assert f.stick_EI_Nm2 == pytest.approx(
        0.5 * e * polar * math.cos(f.lay_angle_rad) ** 3, rel=1e-12
    )
    expected = cx.compute_stiffness(cable).ei_max_Nm2 + f.stick_EI_Nm2
    assert res.EI_max_Nm2 == pytest.approx(expected, rel=1e-12)
    finite = np.abs(k[1:]) > 1e-9
    tangent = res.moment_Nm[1:] / k[1:]
    assert np.allclose(tangent[finite], expected, rtol=1e-9)
    # The default shear of half the filler depth lowers the profile xi.
    sheared = build_core_friction(cable, PROFILES, T)
    assert 0.0 < sheared.fillers.xi < 1.0


def test_filler_profiles_work_and_inputs(cable):
    rng = np.random.default_rng(5)
    kx = np.cumsum(rng.normal(scale=0.01, size=500))
    ky = np.cumsum(rng.normal(scale=0.01, size=500))
    core = build_core_friction(cable, PROFILES, T)
    hist = core_friction_history(core, kx, ky)
    plain = core_friction_history(build_core_friction(cable, SPEC, T), kx, ky)
    friction = hist.work.friction_J_per_m
    assert friction[-1] > plain.work.friction_J_per_m[-1] > 0.0
    assert np.all(np.diff(friction) >= -1e-12 * friction[-1])
    assert np.max(np.abs(hist.work.balance_residual_J_per_m)) < 1e-9 * friction[-1]
    assert plain.filler_moment_Nm is None
    assert core.stick_EI_Nm2 == pytest.approx(
        core.core_stick_EI_Nm2 + core.fillers.stick_EI_Nm2, rel=1e-15
    )
    assert core.populations == (core.population, core.fillers.population)
    # The profile coefficient defaults to the core--filler one.
    own = build_core_friction(
        cable, CoreFrictionSpec(filler_profiles=True, mu_filler_profile=0.3), T
    )
    assert own.fillers.mu == 0.3 and core.fillers.mu == 0.46
    menard = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    with pytest.raises(CableSchemaError, match="interstitial filler"):
        build_core_friction(menard, PROFILES, T)
    for bad in (
        {"central_filler": 1},
        {"filler_profiles": "yes"},
        {"mu_filler_profile": -0.1},
        {"mu_filler_profile": math.nan},
    ):
        with pytest.raises(CableInputError):
            CoreFrictionSpec(**bad)  # type: ignore[arg-type]
