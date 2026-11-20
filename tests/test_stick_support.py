# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Opt-in ``"annulus"`` stick support of the armour stick coefficient ξ.

Closed-form limits of the layered-annulus compliance and of the strip
concentration; an independent finite-difference solution of a two-layer
annulus; the default (``"yun_stack"``) unchanged bit for bit through every
entry point; refusal of an unknown model; and the model-to-model row XI-FE.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.beam_test import SectionLaw
from cablexsec.cables.schema import CableData
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.mechanics.stick_support import (
    annulus_mode1_compliance,
    strip_concentration_compliance,
)
from cablexsec.mechanics.stiffness import STICK_SUPPORTS, xi_for_layer

G = 3.0e8


@pytest.mark.analytical_sanity
def test_thin_layer_on_a_rigid_support_is_the_strip_compliance() -> None:
    radius, thickness = 0.05, 0.001
    c = annulus_mode1_compliance([(radius - thickness, radius, G)], "rigid")
    assert c * G / thickness == pytest.approx(1.0, abs=2.0 * thickness / radius)


@pytest.mark.analytical_sanity
def test_solid_and_hollow_cores_have_their_closed_forms() -> None:
    radius, rho = 0.04, 0.5
    assert annulus_mode1_compliance([(0.0, radius, G)], "axis") == pytest.approx(
        radius / G, rel=1e-14
    )
    hollow = annulus_mode1_compliance([(rho * radius, radius, G)], "bore")
    assert hollow == pytest.approx(
        radius * (1 + rho**2) / (G * (1 - rho**2)), rel=1e-14
    )
    # The free bore is several times softer than a thin strip of the same wall.
    assert hollow * G / (radius - rho * radius) > 3.0


def _finite_difference(
    layers: list[tuple[float, float, float]], support: str, n: int = 1201
) -> float:
    """Mode-1 antiplane annulus (r G w')' - G w / r = 0, unit outer traction."""
    r_in, r_out = layers[-1][0], layers[0][1]
    r = np.linspace(r_in, r_out, n)
    h = r[1] - r[0]
    mid = 0.5 * (r[1:] + r[:-1])

    def modulus(x: np.ndarray) -> np.ndarray:
        g = np.empty_like(x)
        for a, b, gi in layers:
            g[(x >= a) & (x <= b)] = gi
        return g

    g_mid, g_node = modulus(mid), modulus(r)
    a = np.zeros((n, n))
    rhs = np.zeros(n)
    for i in range(1, n - 1):
        a[i, i - 1] = mid[i - 1] * g_mid[i - 1] / h**2
        a[i, i + 1] = mid[i] * g_mid[i] / h**2
        a[i, i] = -(a[i, i - 1] + a[i, i + 1]) - g_node[i] / r[i]
    if support == "rigid":
        a[0, 0] = 1.0
    else:
        a[0, 0], a[0, 1] = -1.0, 1.0
    a[-1, -2], a[-1, -1] = -g_mid[-1] / h, g_mid[-1] / h
    rhs[-1] = 1.0
    return float(np.linalg.solve(a, rhs)[-1])


@pytest.mark.independent_method
@pytest.mark.parametrize("support", ["rigid", "bore"])
def test_two_layer_annulus_matches_finite_differences(support: str) -> None:
    layers = [(0.030, 0.040, 1.0e8), (0.020, 0.030, 2.0e9)]
    exact = annulus_mode1_compliance(layers, support)  # type: ignore[arg-type]
    assert _finite_difference(layers, support) == pytest.approx(exact, rel=1e-3)


@pytest.mark.analytical_sanity
def test_strip_concentration_limits() -> None:
    spacing = 0.006
    # Full coverage: no concentration.
    assert strip_concentration_compliance(spacing, spacing / 2, 0.01, G) == 0.0
    # Thin layer (t << a): mean plus concentration is the strip of the contact
    # width, t / (2 a G) (Parseval).
    half, t = 0.0015, 1.0e-5
    total = t / (G * spacing) + strip_concentration_compliance(spacing, half, t, G)
    assert total == pytest.approx(t / (2 * half * G), rel=5e-3)
    # Half-space line load: the compliance grows by ln(10) / (pi G) per decade
    # of contact width.
    wide = strip_concentration_compliance(spacing, 1e-4, 1.0, G)
    narrow = strip_concentration_compliance(spacing, 1e-5, 1.0, G)
    assert (narrow - wide) * math.pi * G == pytest.approx(math.log(10.0), rel=1e-3)


@pytest.mark.analytical_sanity
def test_support_inputs_are_refused() -> None:
    with pytest.raises(ValueError, match="axis"):
        annulus_mode1_compliance([(0.01, 0.02, G)], "axis")
    with pytest.raises(ValueError, match="r_inner"):
        annulus_mode1_compliance([(0.02, 0.01, G)], "rigid")
    with pytest.raises(ValueError, match="unknown"):
        annulus_mode1_compliance([(0.01, 0.02, G)], "clamped")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        strip_concentration_compliance(0.006, 0.0, 0.01, G)


def _menard() -> CableData:
    return cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)


@pytest.mark.analytical_sanity
def test_default_is_the_yun_stack_bit_for_bit() -> None:
    cable = _menard()
    kappa = 0.05 * np.sin(np.linspace(0.0, 4.0 * np.pi, 240))
    for layer in cable.helical_layers:
        assert xi_for_layer(cable, layer.name) == xi_for_layer(
            cable, layer.name, stick_support="yun_stack"
        )
    a = cx.compute_mkappa(cable, kappa)
    b = cx.compute_mkappa(cable, kappa, stick_support="yun_stack")
    assert np.array_equal(a.moment_Nm, b.moment_Nm)
    assert a.EI_max_Nm2 == b.EI_max_Nm2
    assert (
        cx.compute_stiffness(cable).ei_max_Nm2
        == cx.compute_stiffness(cable, stick_support="yun_stack").ei_max_Nm2
    )
    assert (
        SectionLaw.from_cable(cable).ei_stick_Nm2
        == SectionLaw.from_cable(cable, stick_support="yun_stack").ei_stick_Nm2
    )


@pytest.mark.analytical_sanity
def test_annulus_is_close_to_the_stack_on_thin_layers_over_a_stiff_support() -> None:
    cable = _menard()  # sheath and bedding over laid cores and the inner armour
    for layer in cable.helical_layers:
        yun = xi_for_layer(cable, layer.name)
        annulus = xi_for_layer(cable, layer.name, stick_support="annulus")
        assert annulus == pytest.approx(yun, rel=0.02)
    kappa = 0.05 * np.sin(np.linspace(0.0, 4.0 * np.pi, 240))
    a = cx.compute_mkappa(cable, kappa, stick_support="annulus")
    b = SectionLaw.from_cable(cable, stick_support="annulus")
    assert a.EI_max_Nm2 == pytest.approx(b.ei_stick_Nm2, rel=1e-12)
    assert a.EI_min_Nm2 == cx.compute_mkappa(cable, kappa).EI_min_Nm2


@pytest.mark.analytical_sanity
def test_unknown_stick_support_is_refused_everywhere() -> None:
    cable = _menard()
    kappa = np.array([0.0, 0.01])
    layer = cable.helical_layers[0].name
    assert STICK_SUPPORTS == ("yun_stack", "annulus")
    with pytest.raises(CableInputError, match="stick_support"):
        xi_for_layer(cable, layer, stick_support="series")
    with pytest.raises(CableSchemaError, match="stick_support"):
        cx.compute_mkappa(cable, kappa, stick_support="series")
    with pytest.raises(CableSchemaError, match="stick_support"):
        cx.compute_stiffness(cable, stick_support="series")
    with pytest.raises(CableInputError, match="stick_support"):
        SectionLaw.from_cable(cable, stick_support="series")


@pytest.mark.analytical_sanity
def test_free_bore_only_where_nothing_lies_inside(tmp_path: Path) -> None:
    import check_stick_support_fe as fe

    source = fe.DATA / "line_contact_tube_hollow.yaml"
    hollow = cx.load_cable(source, geometry_override=None)
    # A layer without an isotropic modulus fills the bore: the stack skips
    # it, and the support is then kept rigid (as in the default model), not
    # treated as a free bore.
    text = source.read_text(encoding="utf-8").replace(
        "layers:\n",
        "materials:\n  yarn_probe:\n    description: yarn without a modulus\n"
        "    rho_kg_m3: {value: 500.0, source: test}\n\nlayers:\n"
        "  - name: bore_yarn\n    layer_type: cylindrical\n    material: yarn_probe\n"
        "    r_inner_m: 0.0\n    r_outer_m: 0.020\n    source: test\n\n",
        1,
    )
    path = tmp_path / "filled_bore.yaml"
    path.write_text(text, encoding="utf-8")
    filled = cx.load_cable(path, geometry_override=None)
    free = xi_for_layer(hollow, "armour", stick_support="annulus")
    with filled.material_scope():
        supported = xi_for_layer(filled, "armour", stick_support="annulus")
        series = xi_for_layer(filled, "armour")
    assert free < 0.6 * supported
    assert supported == pytest.approx(series, rel=0.05)


@pytest.mark.independent_method
def test_line_contact_tube_row() -> None:
    import check_stick_support_fe as fe

    rows = fe.stick_comparison()
    hollow, copper = rows["hollow"], rows["copper"]
    # Hollow core: the annulus support follows the core's shear; the series
    # stack takes the wall as a thin strip on a rigid support.
    assert abs(hollow["annulus_error"]) < 0.06
    assert hollow["yun_stack_error"] > 0.9
    # Copper core: a thin layer on a stiff support, both within 10 %.
    assert abs(copper["annulus_error"]) < 0.10
    assert abs(copper["yun_stack_error"]) < 0.10
