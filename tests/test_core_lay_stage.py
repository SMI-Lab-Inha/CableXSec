# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Core lay stage of the bending law: the laid cores stick, then slip.

Without the stage the moment-curvature law lets the laid power cores bend
about their own axes from the first curvature, so it starts below the no-slip
bound by the core parallel-axis term. With it the cores stick until the
friction of their contacts lets them slip (the Papailiou slip front of
:mod:`cablexsec.mechanics.core_friction`). Checks: the start on the no-slip
bound, the slip stage, the end at full slip, closed-form slip curvatures
([Skeie2012] Eqs. 2.31 and 2.35, [Papailiou1997] Eq. 15) and the option off
bit for bit. Public cables only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.section_properties import (
    SectionLoadCase,
    compute_section_properties,
)
from cablexsec.api import CoreFrictionSpec
from cablexsec.cable_adapters import cable_to_mkappa_specs
from cablexsec.exceptions import CableSchemaError
from cablexsec.io import global_export
from cablexsec.mechanics.core_friction import build_core_friction
from cablexsec.mechanics.papailiou import (
    _compression_onset_factor,
    slip_onset_curvature,
)
from cablexsec.mechanics.stiffness import bending_stiffness_by_layer

pytestmark = [
    pytest.mark.analytical_sanity,
    pytest.mark.filterwarnings("ignore::UserWarning"),
]

T = 50.0e3
#: Cores held rigidly and no internal ring slip: the definitions of the
#: no-slip bound (compute_stiffness().ei_max_Nm2).
RIGID = CoreFrictionSpec(core_shear=False, strand_slip=False)


@pytest.fixture(scope="module")
def cable() -> cx.CableData:
    return cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)


@pytest.fixture(scope="module")
def bound(cable: cx.CableData) -> float:
    return float(cx.compute_stiffness(cable).ei_max_Nm2)


@pytest.fixture(scope="module")
def core_parallel_axis(cable: cx.CableData) -> float:
    parts = bending_stiffness_by_layer(cable).values()
    return float(sum(p.core_parallel_axis_Nm2 for p in parts))


@pytest.fixture(scope="module")
def stuck_curvature(cable: cx.CableData) -> float:
    """Half the smallest slip onset of the armour and the cores at ``T``."""
    armour = [slip_onset_curvature(s) for s in cable_to_mkappa_specs(cable, T)]
    cores = [
        slip_onset_curvature(build_core_friction(cable, spec, T).population)
        for spec in (RIGID, CoreFrictionSpec())
    ]
    smallest = min(armour + cores)
    assert smallest > 0.0
    return 0.5 * smallest


def _initial_tangent(
    cable: cx.CableData, kappa: float, spec: CoreFrictionSpec | None
) -> float:
    res = cx.compute_mkappa(
        cable, np.array([0.0, kappa]), tension_N=T, core_friction=spec
    )
    return float(res.moment_Nm[1] / kappa)


def test_stage_starts_on_the_no_slip_bound(
    cable: cx.CableData,
    bound: float,
    core_parallel_axis: float,
    stuck_curvature: float,
) -> None:
    # Cores held rigidly: exactly the no-slip bound (same definitions: the
    # armour terms with the shear coefficient xi of xi_for_layer, the cores'
    # parallel-axis terms with xi = 1, individual bending of every part).
    assert _initial_tangent(cable, stuck_curvature, RIGID) == pytest.approx(
        bound, rel=1e-12
    )
    # Core insulation shear (default): below the bound by (1 - xi_core) of
    # the core parallel-axis term, the only difference of definition.
    xi = build_core_friction(cable, CoreFrictionSpec(), T).xi
    assert 0.0 < xi < 1.0
    assert _initial_tangent(cable, stuck_curvature, CoreFrictionSpec()) == (
        pytest.approx(bound - (1.0 - xi) * core_parallel_axis, rel=1e-12)
    )
    # Stage off: the earlier law, below the bound by the whole core term.
    assert _initial_tangent(cable, stuck_curvature, None) == pytest.approx(
        bound - core_parallel_axis, rel=1e-12
    )


def _closed_form(model: object) -> tuple[float, float, float]:
    """Onset and full-slip curvature and full-slip core moment of the cores."""
    q = model.loads.total_N_per_m  # type: ignore[attr-defined]
    mu = model.mu_effective  # type: ignore[attr-defined]
    xi = model.xi  # type: ignore[attr-defined]
    ea = model.EA_core_N  # type: ignore[attr-defined]
    beta = model.lay_angle_rad  # type: ignore[attr-defined]
    r = model.r_offset_m  # type: ignore[attr-defined]
    n = model.n_cores  # type: ignore[attr-defined]
    s, c = math.sin(beta), math.cos(beta)
    a = mu * s
    # [Skeie2012] Eq. 2.31 with f = mu q cos(beta) per core length.
    onset = mu * q / (xi * ea * s * c)
    # [Papailiou1997] Eq. 15 at the extreme fibre (psi = pi/2).
    full = q * math.expm1(a * math.pi / 2) / (xi * ea * s * s * c)
    # Full-slip moment with the signed capstan bound: tension-side cores at
    # sigma0 expm1(a psi), compression-side cores at -sigma0 (1 - exp(-a psi)),
    # int_0^{pi/2} sinh(a psi) sin(psi) dpsi = a cosh(a pi/2) / (1 + a^2).
    integral = a * math.cosh(a * math.pi / 2) / (1.0 + a * a)
    moment = 2.0 / math.pi * n * q * r * r * c * c / (s * s) * integral
    return onset, full, moment


def test_core_slip_curvatures_have_closed_forms(cable: cx.CableData) -> None:
    result = compute_section_properties(
        cable, [SectionLoadCase("design", T)], core_friction=RIGID, n_points=40
    )
    cores = result.cases[0].cores
    assert cores is not None
    model = build_core_friction(cable, RIGID, T)
    onset, full, _ = _closed_form(model)
    assert cores.onset_curvature_1_m == pytest.approx(onset, rel=1e-12)
    assert cores.full_slip_curvature_1_m == pytest.approx(full, rel=1e-12)
    assert cores.contact_source == "bounded"
    assert cores.contact_line_load_N_per_m == model.loads.total_N_per_m
    # The ratio tends to the pi/2 of [Skeie2012] Eq. 2.35 at small friction.
    weak = CoreFrictionSpec(
        mu_core_core=1e-6,
        mu_core_filler=1e-6,
        mu_core_sheath=1e-6,
        core_shear=False,
        strand_slip=False,
    )
    on_w, full_w, _ = _closed_form(build_core_friction(cable, weak, T))
    assert full_w / on_w == pytest.approx(math.pi / 2, rel=1e-6)
    # Both curvatures are nodes of the curvature grid.
    kappa = result.cases[0].kappa_1_m
    for value in (onset, full):
        assert np.min(np.abs(kappa - value)) <= 1e-15 * value
    # The files for global analysis carry the stage.
    table = global_export.section_properties_csv(result)
    assert "laid cores full slip curvature [1/m]" in table
    assert f"{full:.9g}" in table
    assert "core lay stage" in global_export.bending_tables_csv(result)
    law = global_export.cabledyn_bending_law(result)
    assert law.settings["core_lay_stage"]["core_shear"] is False


def test_stage_passes_through_core_slip_and_ends_at_full_slip(
    cable: cx.CableData, bound: float
) -> None:
    case = [SectionLoadCase("design", T)]
    on = compute_section_properties(cable, case, core_friction=RIGID)
    with_cores = on.cases[0]
    assert with_cores.cores is not None
    without = cx.compute_mkappa(cable, with_cores.kappa_1_m, tension_N=T)
    low, high, tangent = with_cores.tangent
    first = int(np.flatnonzero(low > 0.0)[0])
    assert tangent[0] == pytest.approx(bound, rel=1e-9)
    assert tangent[first] == pytest.approx(bound, rel=1e-9)
    assert with_cores.ei_stick_Nm2 == pytest.approx(bound, rel=1e-12)
    assert tangent[-1] == pytest.approx(on.ei_full_slip_Nm2, rel=1e-9)
    assert np.all(np.diff(tangent) <= 1e-9 * bound)  # concave backbone
    # The armour law is unchanged, so the difference is the core stage.
    kappa = with_cores.kappa_1_m
    core = with_cores.moment_Nm - without.moment_Nm
    model = build_core_friction(cable, RIGID, T)
    onset, full, moment = _closed_form(model)
    # Compression-side cores slip first, at m(a) onset (m = 1 - 3a^2/8): the
    # grid node at the neutral-axis onset itself has already left the stick line.
    a = model.mu_effective * math.sin(model.lay_angle_rad)
    stick = kappa < _compression_onset_factor(a) * onset
    assert np.count_nonzero(stick) > 3
    np.testing.assert_allclose(
        core[stick], model.core_stick_EI_Nm2 * kappa[stick], rtol=1e-9, atol=1e-9
    )
    slipping = (kappa > onset) & (kappa < full)
    assert np.count_nonzero(slipping) > 3
    secant = core[slipping] / kappa[slipping]
    assert np.all(secant <= model.core_stick_EI_Nm2 * (1.0 + 1e-12))
    assert secant[-1] < 0.9 * model.core_stick_EI_Nm2
    assert np.all(np.diff(core[slipping]) > 0.0)
    # Every core slips beyond the full-slip curvature: the core moment is the
    # frozen friction moment (station integration of 60 stations per core).
    beyond = kappa >= full
    assert np.count_nonzero(beyond) > 3
    np.testing.assert_allclose(core[beyond], core[beyond][0], rtol=1e-9)
    assert core[beyond][0] == pytest.approx(moment, rel=5e-3)


def test_option_off_is_bit_for_bit(cable: cx.CableData) -> None:
    cases = [SectionLoadCase("low", 10e3), SectionLoadCase("design", T)]
    base = compute_section_properties(cable, cases, n_points=40)
    for value in (None, False):
        other = compute_section_properties(
            cable, cases, n_points=40, core_friction=value
        )
        for a, b in zip(base.cases, other.cases, strict=True):
            assert np.array_equal(a.kappa_1_m, b.kappa_1_m)
            assert np.array_equal(a.moment_Nm, b.moment_Nm)
            assert a.ei_stick_Nm2 == b.ei_stick_Nm2 and b.cores is None
    # The backbone is the armour-only compute_mkappa, unchanged.
    kappa = base.cases[1].kappa_1_m
    direct = cx.compute_mkappa(cable, kappa, tension_N=T, contact_model="bounded")
    assert np.array_equal(base.cases[1].moment_Nm[1:], direct.moment_Nm[1:])
    assert base.settings["core_lay_stage"] is None


def test_solved_contact_gives_the_core_loads_of_the_solved_bundle(
    cable: cx.CableData,
) -> None:
    result = compute_section_properties(
        cable,
        [SectionLoadCase("design", T)],
        bending_contact="solved",
        core_friction=RIGID,
        n_points=30,
    )
    cores = result.cases[0].cores
    assert cores is not None and cores.contact_source == "solved"
    assert cores.contact_line_load_N_per_m > 0.0
    settings = result.settings["core_lay_stage"]
    assert settings["core_shear"] is False
    assert settings["interface_stiffness_Pa_per_m"] is None


def test_cores_that_are_not_laid_are_refused() -> None:
    cable = cx.load_cable(
        cx.inter_array_three_core_yaml_path(), geometry_override=None
    ).model_copy(update={"core_lay": None})
    with pytest.raises(CableSchemaError, match="laid cores"):
        compute_section_properties(
            cable, [SectionLoadCase("design", T)], core_friction=True, n_points=20
        )


def test_bending_law_export_with_the_core_lay_stage(
    cable: cx.CableData, bound: float
) -> None:
    tensions = [10e3, T]
    law = cx.export_bending_law(
        cable, tensions_N=tensions, kappa_max_1_m=0.2, n_points=80, core_friction=RIGID
    )
    for index, tension in enumerate(tensions):
        direct = cx.compute_mkappa(
            cable, law.kappa_1_m, tension_N=tension, core_friction=RIGID
        )
        np.testing.assert_array_equal(law.moment_Nm[index, 1:], direct.moment_Nm[1:])
        np.testing.assert_allclose(
            law.laws[index].backbone(law.kappa_1_m),
            law.moment_Nm[index],
            rtol=1e-9,
            atol=1e-9,
        )
    first = law.moment_Nm[1, 1] / law.kappa_1_m[1]
    assert first == pytest.approx(bound, rel=1e-9)
    assert "core lay stage" in law.provenance.method
    assert "core_friction" in law.settings


def test_filler_profiles_in_the_core_lay_stage(cable: cx.CableData) -> None:
    spec = CoreFrictionSpec(strand_slip=False, filler_profiles=True)
    result = compute_section_properties(
        cable, [SectionLoadCase("design", T)], core_friction=spec, n_points=40
    )
    cores = result.cases[0].cores
    assert cores is not None
    model = build_core_friction(cable, spec, T)
    assert model.fillers is not None
    assert cores.filler_line_load_N_per_m == model.fillers.line_load_N_per_m
    onset = slip_onset_curvature(model.fillers.population)
    assert cores.filler_onset_curvature_1_m == pytest.approx(onset, rel=1e-12)
    assert cores.filler_full_slip_curvature_1_m is not None
    assert cores.filler_full_slip_curvature_1_m > onset
    assert cores.stick_bending_stiffness_Nm2 == pytest.approx(
        model.stick_EI_Nm2, rel=1e-12
    )
    kappa = result.cases[0].kappa_1_m
    for value in (onset, cores.filler_full_slip_curvature_1_m):
        assert np.min(np.abs(kappa - value)) <= 1e-15 * value
    assert result.settings["core_lay_stage"]["filler_profiles"] is True
    table = global_export.section_properties_csv(result)
    assert "filler profiles full slip curvature [1/m]" in table
    assert "filler profiles: line load" in global_export.bending_tables_csv(result)
    # Without the profiles the columns stay out.
    plain = compute_section_properties(
        cable, [SectionLoadCase("design", T)], core_friction=RIGID, n_points=40
    )
    assert plain.cases[0].cores is not None
    assert plain.cases[0].cores.filler_line_load_N_per_m is None
    assert "filler profiles" not in global_export.section_properties_csv(plain)
