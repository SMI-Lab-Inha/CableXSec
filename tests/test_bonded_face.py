# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Bonded outer armour face (``bonded_face="outer"``, gap G8).

Closed form of the inner-loaded annulus compliance; the default unchanged bit
for bit; the stuck section unchanged and the full-slip tangent raised by the
bond's share; the friction cap from the inner face only; energy balance; and
refusal on the routes that would miss the bond's share of the wire force.
"""

from __future__ import annotations

import math
from importlib.resources import files

import numpy as np
import pytest
from pydantic import ValidationError

import cablexsec as cx
from cablexsec.api import _prepare_mkappa_law
from cablexsec.cable_adapters import (
    _bounded_face_pressures,
    cable_to_mkappa_specs,
)
from cablexsec.cables.schema import CableData
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.mechanics.axisymmetric import compute_contact_forces
from cablexsec.mechanics.stick_support import annulus_mode1_compliance_inner
from cablexsec.mechanics.stiffness import (
    bonded_face_ei,
    bonded_face_terms,
    ei_min_slip,
    xi_for_layer,
)

G = 3.0e8


def _zeidan() -> CableData:
    return cx.load_cable(
        files("cablexsec.cables") / "zeidan2025.yaml", geometry_override=None
    )


def _bond(
    cable: CableData, name: str = "outer_armour", face: str = "outer"
) -> CableData:
    layers = [
        lyr.model_copy(update={"bonded_face": face}) if lyr.name == name else lyr
        for lyr in cable.layers
    ]
    return cable.model_copy(update={"layers": layers})


def _law(cable: CableData):  # type: ignore[no-untyped-def]
    with cable.material_scope():
        return _prepare_mkappa_law(
            cable,
            0.0,
            mu=0.2,
            phi_bend=0.0,
            contact_model="bounded",
            section_state=None,
            radial_state=None,
            core_spec=None,
            progressive=None,
        )


@pytest.mark.analytical_sanity
def test_inner_loaded_single_layer_is_the_closed_form() -> None:
    r_i, r_o = 0.05, 0.056
    expected = r_i * (r_i**2 + r_o**2) / (G * (r_o**2 - r_i**2))
    assert annulus_mode1_compliance_inner([(r_i, r_o, G)]) == pytest.approx(
        expected, rel=1e-12
    )


@pytest.mark.analytical_sanity
def test_inner_loaded_split_layer_equals_the_whole_layer() -> None:
    whole = annulus_mode1_compliance_inner([(0.05, 0.056, G)])
    split = annulus_mode1_compliance_inner([(0.05, 0.053, G), (0.053, 0.056, G)])
    assert split == pytest.approx(whole, rel=1e-12)


def test_inner_loaded_refuses_overlap_and_empty_is_zero() -> None:
    assert annulus_mode1_compliance_inner([]) == 0.0
    with pytest.raises(ValueError, match="overlap"):
        annulus_mode1_compliance_inner([(0.05, 0.056, G), (0.054, 0.06, G)])


def test_default_cable_has_no_bond_and_an_unchanged_law() -> None:
    cable = _bond(_zeidan(), face="none")
    with cable.material_scope():
        assert bonded_face_ei(cable) == 0.0
        ei_min = ei_min_slip(cable)
    menard = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    explicit = _bond(menard, name="outer_armour", face="none")
    np.testing.assert_array_equal(
        _law(menard).run(np.linspace(0.0, 0.1, 51)).moment_Nm,
        _law(explicit).run(np.linspace(0.0, 0.1, 51)).moment_Nm,
    )
    explicit = _bond(cable, face="none")
    kappa = np.r_[np.linspace(0.0, 0.1, 51), np.linspace(0.1, -0.1, 101)[1:]]
    a, b = _law(cable).run(kappa), _law(explicit).run(kappa)
    np.testing.assert_array_equal(a.moment_Nm, b.moment_Nm)
    assert cx.compute_stiffness(cable).ei_min_Nm2 == ei_min


def test_bond_terms_keep_the_stuck_layer_and_add_the_bond_share() -> None:
    cable = _bond(_zeidan())
    with cable.material_scope():
        xi_support = xi_for_layer(cable, "outer_armour")
        terms = bonded_face_terms(cable, "outer_armour", xi_support)
        # A compliant support: stuck, the support and the bond in parallel.
        soft = bonded_face_terms(cable, "outer_armour", 0.5)
        ei_bond = bonded_face_ei(cable)
    assert 0.0 < terms.xi_bond < 1.0
    if xi_support >= 1.0:
        assert terms.xi_slider + terms.xi_bond == pytest.approx(1.0, abs=1e-15)
    k2 = terms.stiffness_N_m2 * (1.0 - terms.xi_bond) / terms.xi_bond
    k_s = 0.5 * k2 / 0.5  # xi_support = 0.5
    expected = (k_s + terms.stiffness_N_m2) / (k_s + terms.stiffness_N_m2 + k2)
    assert soft.xi_slider + soft.xi_bond == pytest.approx(expected, rel=1e-12)
    assert ei_bond == pytest.approx(terms.ei_bond_Nm2, rel=1e-15)


def test_law_stuck_stiffness_unchanged_and_full_slip_tangent_raised() -> None:
    base, bonded = _bond(_zeidan(), face="none"), _zeidan()
    law0, law1 = _law(base), _law(bonded)
    with bonded.material_scope():
        ei_bond = bonded_face_ei(bonded)
    assert law1.ei_min == pytest.approx(law0.ei_min + ei_bond, rel=1e-15)
    stuck0 = law0.ei_min + sum(s.EI_stick for s in law0.layers)
    stuck1 = law1.ei_min + sum(s.EI_stick for s in law1.layers)
    assert stuck1 == pytest.approx(stuck0, rel=1e-12)
    kappa = np.linspace(0.0, 0.3, 601)
    m = law1.run(kappa).moment_Nm
    tangent = (m[-1] - m[-41]) / (kappa[-1] - kappa[-41])
    assert tangent == pytest.approx(law1.ei_min, rel=1e-6)


def test_friction_cap_is_the_inner_face_share() -> None:
    base, bonded = _bond(_zeidan(), face="none"), _zeidan()
    with bonded.material_scope():
        p_ext = bonded.initial_state.external_pressure_Pa
        cf = compute_contact_forces(bonded, 0.0, p_ext_Pa=p_ext)
        below, above = _bounded_face_pressures(bonded, cf.interface_pressure_Pa, p_ext)[
            "outer_armour"
        ]
        s0 = cable_to_mkappa_specs(base, mu=0.2)
        s1 = cable_to_mkappa_specs(bonded, mu=0.2, bonded_faces=True)
    share = below / (below + above)
    assert 0.0 < share < 1.0
    names = [lyr.name for lyr in bonded.helical_layers]
    i = names.index("outer_armour")
    assert s1[i].contact_pressure_Pa == pytest.approx(
        s0[i].contact_pressure_Pa * share, rel=1e-12
    )
    assert s1[i].F_n_axi_N == pytest.approx(s0[i].F_n_axi_N * share, rel=1e-12)
    j = names.index("inner_armour")
    assert s1[j] == s0[j]


def test_bonded_law_closes_its_energy_balance() -> None:
    kappa = np.r_[np.linspace(0.0, 0.15, 151), np.linspace(0.15, -0.15, 301)[1:]]
    work = _law(_bond(_zeidan())).run(kappa).work
    scale = float(np.max(np.abs(work.friction_J_per_m))) + 1.0
    assert np.max(np.abs(work.balance_residual_J_per_m)) < 1e-6 * scale


def test_routes_without_the_bond_share_refuse_a_bonded_cable() -> None:
    bonded = _bond(_zeidan())
    with bonded.material_scope(), pytest.raises(CableSchemaError, match="bonded_face"):
        cable_to_mkappa_specs(bonded, mu=0.2)
    with bonded.material_scope(), pytest.raises(CableSchemaError, match="bounded"):
        cable_to_mkappa_specs(bonded, mu=0.2, contact_model="qp", bonded_faces=True)


def test_bond_needs_the_outermost_helical_layer() -> None:
    cable = _bond(_zeidan(), name="inner_armour")
    with cable.material_scope(), pytest.raises(CableInputError, match="outermost"):
        bonded_face_terms(cable, "inner_armour", 1.0)


def test_schema_refuses_an_unknown_face() -> None:
    lyr = next(lyr for lyr in _zeidan().layers if lyr.name == "outer_armour")
    with pytest.raises(ValidationError):
        type(lyr).model_validate({**lyr.model_dump(), "bonded_face": "inner"})


def test_bond_share_is_small_for_a_thick_soft_free_sheath() -> None:
    cable = _bond(_zeidan())
    with cable.material_scope():
        terms = bonded_face_terms(cable, "outer_armour", 1.0)
    assert math.isfinite(terms.stiffness_N_m2) and terms.xi_bond < 0.2


def test_exported_law_carries_the_bond_and_runs_outside_a_scope() -> None:
    """The global-analysis export (raw module function, no outer material
    scope) reads the cable's own materials and exports the bonded full-slip
    stiffness."""
    from cablexsec.analysis.bending_law import export_bending_law  # noqa: PLC0415

    cable = _zeidan()
    out = export_bending_law(
        cable, tensions_N=[0.0, 50e3], kappa_max_1_m=0.2, n_points=40
    )
    sliding = cx.compute_stiffness(_bond(cable, face="none")).ei_min_Nm2
    bond = bonded_face_ei(cable)  # decorated: its own material scope
    np.testing.assert_allclose(out.ei_slip_Nm2, sliding + bond, rtol=1e-12)
    assert bond > 1e3


def test_tension_history_keeps_the_inner_face_share() -> None:
    """A constant tension history gives the fixed-tension law: the history's
    P1+P2 is split into the inner face's share at every sample, as the specs
    are at a fixed tension."""
    cable = _zeidan()
    kappa = np.r_[np.linspace(0.0, 0.1, 51), np.linspace(0.1, -0.1, 101)[1:]]
    fixed = cx.compute_mkappa(cable, kappa, tension_N=50e3, mu=0.2)
    history = cx.compute_mkappa(
        cable,
        kappa,
        mu=0.2,
        contact_tension_history_N=np.full(kappa.shape, 50e3),
    )
    np.testing.assert_allclose(history.moment_Nm, fixed.moment_Nm, rtol=1e-9, atol=1e-9)
    unsplit = cx.compute_mkappa(
        _bond(cable, face="none"),
        kappa,
        mu=0.2,
        contact_tension_history_N=np.full(kappa.shape, 50e3),
    )
    assert np.max(np.abs(history.moment_Nm - unsplit.moment_Nm)) > 1.0


def test_a_pressure_history_is_refused_for_a_bonded_face() -> None:
    cable = _zeidan()
    kappa = np.linspace(0.0, 0.1, 11)
    pressure = {lyr.name: np.full(kappa.shape, 1e6) for lyr in cable.helical_layers}
    with pytest.raises(CableSchemaError, match="contact_tension_history_N"):
        cx.compute_mkappa(cable, kappa, mu=0.2, contact_pressure_Pa=pressure)
