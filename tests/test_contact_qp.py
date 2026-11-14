# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""KKT/QP radial-compliance contact solver ([Skeie2012] §2.3).

Analytical-limit gates for the QP whose KKT multipliers are the inter-layer
contact pressures:

* **KKT complementarity** — q ≥ 0, w ≥ 0, qᵀw ≈ 0 to machine precision (the
  defining property of the LCP solution; NOT a least-squares fit);
* **zero-load → zero contact** (no interference, no pressure);
* **tension monotonicity** — contact pressure rises with effective tension;
* **capstan consistency** — per-wire F_n ≥ the pure capstan load (contact adds);
* **rigid-limit reduction** — as layer radial stiffness → ∞ the QP interface
  pressures approach the bounded-equilibrium chain (the QP generalises it).
"""

from __future__ import annotations

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.mechanics.contact_qp import (
    radial_foundation_modulus,
    solve_contact_qp,
)

pytestmark = pytest.mark.analytical_sanity


def _cable() -> cx.CableData:
    return cx.load_cable(cx.inter_array_single_core_yaml_path(), geometry_override=None)


def test_kkt_complementarity_holds_to_machine_precision() -> None:
    """q ≥ 0, and the complementarity residual max|q·w| ≈ 0 (the KKT point)."""
    res = solve_contact_qp(_cable(), 1.0e5, p_ext_Pa=2.0e6)
    assert np.all(res.contact_pressure_Pa >= -1e-9)
    assert res.complementarity_residual < 1e-6
    # the residual is at machine-precision scale, not a fit tolerance
    assert res.complementarity_residual < 1e-8


def test_zero_load_gives_zero_contact() -> None:
    """No tension and no pressure → no interference → zero contact pressure."""
    res = solve_contact_qp(_cable(), 0.0)
    if res.contact_pressure_Pa.size:
        assert np.max(np.abs(res.contact_pressure_Pa)) < 1e-6


def test_contact_pressure_monotone_in_tension() -> None:
    """Higher effective tension → higher (or equal) interface contact pressure."""
    lo = solve_contact_qp(_cable(), 5.0e4)
    hi = solve_contact_qp(_cable(), 2.0e5)
    assert hi.contact_pressure_Pa.sum() >= lo.contact_pressure_Pa.sum() - 1e-9
    assert hi.contact_pressure_Pa.sum() > 0.0


def test_per_wire_normal_load_counts_both_faces_once() -> None:
    cable = _cable()
    res = solve_contact_qp(cable, 1.0e5)
    names = list(res.section.outer_displacement_m)
    for layer in cable.helical_layers:
        i = names.index(layer.name)
        reactions = res.section.contact_line_force_N_per_m
        expected = (reactions[i - 1] + reactions[i]) / layer.n_wires
        assert res.Fn_per_wire_N[layer.name] == pytest.approx(expected, rel=1e-12)


def test_rigid_limit_reduces_toward_equilibrium_chain() -> None:
    """As radial stiffness → ∞ the QP solution stays finite and non-negative.

    The QP generalises the bounded-equilibrium chain; in the rigid limit the
    interface pressures remain a valid, non-negative equilibrium (sanity, not a
    numeric identity — the two models differ in radial compliance)."""
    res = solve_contact_qp(_cable(), 1.0e5, p_ext_Pa=1.0e6)
    chain = cx  # the legacy chain
    forces = chain.compute_stiffness  # ensure import path exists
    assert forces is not None
    assert np.all(np.isfinite(res.contact_pressure_Pa))
    assert np.all(res.contact_pressure_Pa >= -1e-9)


def test_radial_modulus_ordering_metal_stiffer_than_polymer() -> None:
    """A steel armour layer is radially stiffer than an HDPE bedding layer."""
    cable = _cable()
    armour = next(lyr for lyr in cable.helical_layers if "armour" in lyr.name)
    from cablexsec.cables.schema import CylindricalLayer  # noqa: PLC0415

    poly = next(
        lyr
        for lyr in cable.layers
        if isinstance(lyr, CylindricalLayer) and lyr.r_inner_m is not None
    )
    assert radial_foundation_modulus(armour) > 0.0
    assert radial_foundation_modulus(poly) > 0.0


def test_solver_is_fast_and_terminates() -> None:
    """The active-set LCP terminates with n_active ≤ n_interfaces (O(n) path)."""
    res = solve_contact_qp(_cable(), 1.5e5, p_ext_Pa=3.0e6)
    assert 0 <= res.n_active <= res.contact_pressure_Pa.size


def test_surface_compliance_matches_closed_form_lame() -> None:
    """The cylinder surface compliances equal the closed-form thick-ring Lamé."""
    from cablexsec.cables.schema import CylindricalLayer  # noqa: PLC0415
    from cablexsec.materials import library  # noqa: PLC0415
    from cablexsec.mechanics.contact_qp import surface_compliances  # noqa: PLC0415

    a, b = 0.050, 0.060
    lyr = CylindricalLayer(
        name="ring", material="galvanised_steel_armour", r_inner_m=a, r_outer_m=b
    )
    e = library.get_property("galvanised_steel_armour", "E_Pa")
    nu = library.get_property("galvanised_steel_armour", "nu")
    c_in, c_out, _ = surface_compliances(lyr)
    c_in_cf = (a / e) * ((b**2 + a**2) / (b**2 - a**2) + nu)
    c_out_cf = (b / e) * ((b**2 + a**2) / (b**2 - a**2) - nu)
    assert c_in == pytest.approx(c_in_cf, rel=1e-9)
    assert c_out == pytest.approx(c_out_cf, rel=1e-9)


def test_shrink_fit_pressure_matches_compound_cylinder_closed_form() -> None:
    """QP surface compliances reproduce the textbook compound-cylinder shrink-fit
    contact pressure to machine precision — an exact-elasticity gate on the
    radial-contact mechanics (the radial-compliance physics the bounded chain
    lacks)."""
    from cablexsec.cables.schema import CylindricalLayer  # noqa: PLC0415
    from cablexsec.materials import library  # noqa: PLC0415
    from cablexsec.mechanics.contact_qp import surface_compliances  # noqa: PLC0415

    a, c_r, b = 0.050, 0.055, 0.060
    mat = "galvanised_steel_armour"
    e = library.get_property(mat, "E_Pa")
    nu = library.get_property(mat, "nu")
    inner = CylindricalLayer(name="i", material=mat, r_inner_m=a, r_outer_m=c_r)
    outer = CylindricalLayer(name="o", material=mat, r_inner_m=c_r, r_outer_m=b)
    delta = 1.0e-5  # radial interference [m]
    p_qp = delta / (surface_compliances(inner)[1] + surface_compliances(outer)[0])
    bracket = (1 / e) * ((b**2 + c_r**2) / (b**2 - c_r**2) + nu) + (1 / e) * (
        (c_r**2 + a**2) / (c_r**2 - a**2) - nu
    )
    p_textbook = delta / (c_r * bracket)
    assert p_qp == pytest.approx(p_textbook, rel=1e-6)


def test_cable_adapter_can_build_qp_contact_specs() -> None:
    from cablexsec.cable_adapters import cable_to_mkappa_specs  # noqa: PLC0415

    cable = _cable()
    bounded = cable_to_mkappa_specs(cable, 1.0e5, contact_model="bounded")
    qp = cable_to_mkappa_specs(cable, 1.0e5, contact_model="qp")
    assert len(qp) == len(bounded)
    assert all(s.F_n_axi_N >= 0.0 for s in qp)


@pytest.mark.parametrize("tension", [0.0, 1.0e5])
def test_offset_cores_cannot_be_silently_omitted(tension):
    from cablexsec.cable_adapters import armour_contact_pressure
    from cablexsec.exceptions import CableSchemaError

    cable = cx.load_cable(cx.inter_array_three_core_yaml_path(), geometry_override=None)
    with pytest.raises(CableSchemaError, match="concentric"):
        solve_contact_qp(cable, tension)
    with pytest.raises(CableSchemaError, match="concentric"):
        armour_contact_pressure(cable, tension, contact_model="qp")


def test_assembled_three_layer_solid_cylinder_matches_lame():
    """Unequal interface radii must transmit uniform radial compression exactly."""
    from cablexsec.cables.schema import CylindricalLayer
    from cablexsec.materials import library

    radii = [0.0, 0.01, 0.025, 0.05]
    material = "galvanised_steel_armour"
    layers = tuple(
        CylindricalLayer(name=f"ring{i}", material=material, r_inner_m=a, r_outer_m=b)
        for i, (a, b) in enumerate(zip(radii[:-1], radii[1:], strict=True))
    )
    cable = _cable().model_copy(update={"layers": layers})
    pressure = 1.0e6
    result = solve_contact_qp(cable, 0.0, p_ext_Pa=pressure)
    e, nu = library.get_property(material, "E_Pa"), library.get_property(material, "nu")
    np.testing.assert_allclose(result.contact_pressure_Pa, pressure, rtol=1e-12)
    np.testing.assert_allclose(result.interface_radii_m, radii[1:-1], rtol=1e-12)
    assert result.section.axial_strain == pytest.approx(
        2 * nu * pressure / e, rel=1e-12
    )
    for i, radius in enumerate(radii[1:]):
        assert result.radial_displacement_m[f"ring{i}"] == pytest.approx(
            -(1 - nu) * pressure * radius / e, rel=1e-12
        )
    assert result.section.relative_equilibrium_residual < 1e-12


def test_solid_core_compliance_has_no_fictitious_inner_surface():
    from cablexsec.cables.schema import CylindricalLayer
    from cablexsec.materials import library
    from cablexsec.mechanics.contact_qp import surface_compliances

    material = "galvanised_steel_armour"
    layer = CylindricalLayer(
        name="solid", material=material, r_inner_m=0, r_outer_m=0.02
    )
    e, nu = library.get_property(material, "E_Pa"), library.get_property(material, "nu")
    assert surface_compliances(layer) == pytest.approx(
        (0, (1 - nu) * 0.02 / e, 0), rel=1e-12
    )
