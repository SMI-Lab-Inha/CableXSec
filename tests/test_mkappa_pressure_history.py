# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""``compute_mkappa`` with a contact-pressure or tension history (gap a1).

The armour friction bound of every wire follows a per-sample ``P1+P2``; the
slip state is carried across the change. Checks: a constant history is the
fixed-pressure law; a pressure step inside stick changes nothing; a pressure
cycle at fixed curvature in full slip dissipates the closed-form work of each
wire and then shakes down; a tension history is the fatigue path's friction
stress (same kernels); per-cycle work in full slip follows the closed form at
every tension and rises with tension.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.fatigue import wire_stress_field
from cablexsec.cable_adapters import (
    cable_to_fatigue_layers,
    cable_to_mkappa_specs,
    structural_armour_layers,
)
from cablexsec.exceptions import (
    CableInputError,
    CableSchemaError,
    PhysicalRangeError,
)
from cablexsec.mechanics.mkappa import HelicalLayerSpec
from cablexsec.mechanics.papailiou import (
    compute_mkappa_wire_slip_front,
    full_slip_curvature,
    slip_onset_curvature,
)

pytestmark = pytest.mark.analytical_sanity


@pytest.fixture(scope="module")
def cable() -> Any:
    return cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)


def _names(cable: Any) -> list[str]:
    return [layer.name for layer in structural_armour_layers(cable)]


def _bounds(spec: HelicalLayerSpec, pressure: float) -> tuple[Any, Any]:
    """Closed-form Papailiou bounds (upper, lower) at ``phi_bend = 0``.

    ``sigma0 = r P (2 pi r / n) cos a / (A sin^2 a)``, ``psi = asin|cos phi|``,
    upper ``sigma0 expm1(lam psi)``, lower ``-sigma0 expm1(-lam psi)``.
    """
    sin_a = abs(math.sin(spec.lay_angle_rad))
    width = 2.0 * math.pi * spec.r_helix_m / spec.n_wires
    sigma0 = (
        spec.r_helix_m
        * pressure
        * width
        * math.cos(spec.lay_angle_rad)
        / (spec.A_wire_m2 * sin_a**2)
    )
    psi = np.arcsin(np.clip(np.abs(np.cos(spec.phi_wires)), 0.0, 1.0))
    lam = spec.mu * sin_a
    return sigma0 * np.expm1(lam * psi), -sigma0 * np.expm1(-lam * psi)


def _weight(spec: HelicalLayerSpec) -> float:
    return spec.A_wire_m2 * math.cos(spec.lay_angle_rad)


# --------------------------------------------------------------------------- #
# Constant history                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("friction_model", ["direction_tied", "progressive"])
def test_constant_pressure_history_is_the_fixed_pressure_law(
    cable: Any, friction_model: Any
) -> None:
    """The same return map with the same pressure: identical arithmetic."""
    tension = 2.0e4
    kappa = 0.04 * np.sin(np.linspace(0.0, 4.0 * np.pi, 301))
    specs = cable_to_mkappa_specs(cable, tension)
    pressure = {
        name: np.full(kappa.size, spec.contact_pressure_Pa)
        for name, spec in zip(_names(cable), specs, strict=True)
    }
    ref = cx.compute_mkappa(
        cable, kappa, tension_N=tension, friction_model=friction_model
    )
    got = cx.compute_mkappa(
        cable,
        kappa,
        tension_N=tension,
        friction_model=friction_model,
        contact_pressure_Pa=pressure,
    )
    assert np.array_equal(got.moment_Nm, ref.moment_Nm)
    assert np.array_equal(got.slip_fraction, ref.slip_fraction)
    assert got.work is not None and ref.work is not None
    assert np.array_equal(got.work.friction_J_per_m, ref.work.friction_J_per_m)
    assert np.array_equal(got.work.stored_J_per_m, ref.work.stored_J_per_m)
    assert got.W_diss_J_per_m == ref.W_diss_J_per_m
    assert ref.contact_pressure_Pa is None
    assert got.contact_pressure_Pa is not None
    assert set(got.contact_pressure_Pa) == set(_names(cable))
    assert got.provenance is not None
    assert "supplied per-sample history" in got.provenance.method
    assert any("reference state" in text for text in got.provenance.limitations)


def test_constant_tension_history_is_the_fixed_tension_law(cable: Any) -> None:
    tension = 3.0e4
    kappa = 0.04 * np.sin(np.linspace(0.0, 4.0 * np.pi, 301))
    ref = cx.compute_mkappa(cable, kappa, tension_N=tension)
    got = cx.compute_mkappa(
        cable, kappa, contact_tension_history_N=np.full(kappa.size, tension)
    )
    scale = float(np.max(np.abs(ref.moment_Nm)))
    np.testing.assert_allclose(got.moment_Nm, ref.moment_Nm, rtol=0, atol=1e-12 * scale)
    assert got.tension_N == tension
    assert got.provenance is not None
    assert "tension history through the bounded" in got.provenance.method


def _touching_alternate_wires() -> Any:
    """Three touching structural armour layers laid alternately: the middle one
    has two crossing interfaces, which the crossed-wire line loads refuse."""
    from cablexsec.cables.schema import CableData, CableMeta, HelicalLayer

    layers = tuple(
        HelicalLayer(
            name=f"wire-{i}",
            material="galvanised_steel_armour",
            n_wires=12 * (i + 1),
            d_wire_m=0.002,
            r_helix_m=0.02 + 0.002 * i,
            lay_angle_deg=20 + 10 * i,
            lay_direction="left_hand" if i == 1 else "right_hand",
        )
        for i in range(3)
    )
    return CableData(cable=CableMeta(name="touching alternate wires"), layers=layers)


def test_tension_history_on_a_stack_the_crossing_loads_refuse() -> None:
    # Bounded contact: the qp and solved models do not solve this sheath-free
    # synthetic section reliably even at a fixed tension (singular on some
    # platforms); the next test covers them on a real cable.
    cable = _touching_alternate_wires()
    with pytest.raises(CableSchemaError):
        cx.armour_contact_histories(cable, np.array([1.0e4]))
    tension = 1.0e4
    kappa = 0.05 * np.sin(np.linspace(0.0, 4.0 * np.pi, 121))
    ref = cx.compute_mkappa(cable, kappa, tension_N=tension)
    got = cx.compute_mkappa(
        cable, kappa, contact_tension_history_N=np.full(kappa.size, tension)
    )
    scale = float(np.max(np.abs(ref.moment_Nm)))
    np.testing.assert_allclose(got.moment_Nm, ref.moment_Nm, rtol=0, atol=1e-12 * scale)


@pytest.mark.parametrize("contact_model", ["bounded", "qp", "solved"])
def test_tension_history_never_forms_crossing_loads(
    contact_model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cablexsec.api as api_module
    import cablexsec.cable_adapters as adapters

    # concentric, so the coupled radial contact of qp applies
    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    tension = 2.0e4
    kappa = 0.03 * np.sin(np.linspace(0.0, 4.0 * np.pi, 61))
    ref = cx.compute_mkappa(
        cable, kappa, tension_N=tension, contact_model=contact_model
    )

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("crossing loads formed on the bending route")

    monkeypatch.setattr(api_module, "armour_crossing_line_loads", refuse)
    monkeypatch.setattr(adapters, "armour_crossing_line_loads", refuse)
    got = cx.compute_mkappa(
        cable,
        kappa,
        contact_tension_history_N=np.full(kappa.size, tension),
        contact_model=contact_model,
    )
    scale = float(np.max(np.abs(ref.moment_Nm)))
    np.testing.assert_allclose(got.moment_Nm, ref.moment_Nm, rtol=0, atol=1e-12 * scale)


def test_core_friction_keeps_its_reference_loads(cable: Any) -> None:
    """With laid-core friction only the armour bound follows the history."""
    tension = 2.0e4
    kappa = 0.03 * np.sin(np.linspace(0.0, 4.0 * np.pi, 201))
    specs = cable_to_mkappa_specs(cable, tension)
    pressure = {
        name: np.full(kappa.size, spec.contact_pressure_Pa)
        for name, spec in zip(_names(cable), specs, strict=True)
    }
    ref = cx.compute_mkappa(cable, kappa, tension_N=tension, core_friction=True)
    got = cx.compute_mkappa(
        cable,
        kappa,
        tension_N=tension,
        core_friction=True,
        contact_pressure_Pa=pressure,
    )
    assert np.array_equal(got.moment_Nm, ref.moment_Nm)
    assert got.core_friction is not None and ref.core_friction is not None
    assert np.array_equal(got.core_friction.moment_Nm, ref.core_friction.moment_Nm)


# --------------------------------------------------------------------------- #
# Pressure changes                                                             #
# --------------------------------------------------------------------------- #
def test_pressure_steps_inside_stick_change_nothing(cable: Any) -> None:
    """Below every wire's slip onset at the lowest pressure the stress is the
    trial stress, whatever the bound: moment and work are unchanged."""
    tension = 2.0e4
    specs = cable_to_mkappa_specs(cable, tension)
    # Half the compression-side onset (factor <= 1) at the lowest pressure.
    onset = min(slip_onset_curvature(spec) for spec in specs)
    low, high = 0.6, 1.7
    amplitude = 0.4 * low * onset
    kappa = amplitude * np.sin(np.linspace(0.0, 6.0 * np.pi, 601))
    factor = np.ones(kappa.size)
    factor[97:233] = high
    factor[233:410] = low
    factor[410:] = 1.3
    pressure = {
        name: factor * spec.contact_pressure_Pa
        for name, spec in zip(_names(cable), specs, strict=True)
    }
    ref = cx.compute_mkappa(cable, kappa, tension_N=tension)
    got = cx.compute_mkappa(
        cable, kappa, tension_N=tension, contact_pressure_Pa=pressure
    )
    assert ref.work is not None and got.work is not None
    assert float(np.max(ref.slip_fraction)) == 0.0
    assert float(np.max(got.slip_fraction)) == 0.0
    scale = float(np.max(np.abs(ref.moment_Nm)))
    np.testing.assert_allclose(got.moment_Nm, ref.moment_Nm, rtol=0, atol=1e-12 * scale)
    assert float(np.max(np.abs(got.work.friction_J_per_m))) == 0.0


@pytest.mark.parametrize("steps", [1, 400])
def test_pressure_cycle_at_fixed_curvature_in_full_slip(cable: Any, steps: int) -> None:
    """Hand calculation per wire. At ``kappa_h = 4 max kappa_fs`` every wire
    sits at its bound ``c0``. Lowering the pressure to ``r P0`` in ``n`` equal
    steps clips it to ``c1 = r c0``; each step releases stored energy as
    friction work, ``w (c0^2 - c1^2) / (2 k)`` in total, of which the
    backward-Euler friction increment is all but ``w (c0 - c1)^2 / (2 k n)``
    (integration remainder). Raising it back, and a second cycle to a higher
    pressure than ``r P0``, leave every wire inside its bound: no more work
    (shakedown)."""
    tension = 2.0e4
    specs = cable_to_mkappa_specs(cable, tension)
    names = _names(cable)
    kappa_h = 4.0 * max(full_slip_curvature(spec) for spec in specs)
    ramp, hold = 200, steps + 300
    kappa = np.concatenate([np.linspace(0.0, kappa_h, ramp), np.full(hold, kappa_h)])
    ratio = 0.55
    factor = np.ones(kappa.size)
    start = ramp + 50
    factor[start : start + steps] = (
        1.0 - (1.0 - ratio) * np.arange(1, steps + 1) / steps
    )
    end = start + steps
    factor[end : end + 50] = ratio
    factor[end + 50 : end + 100] = 1.0
    factor[end + 100 : end + 150] = ratio + 0.15
    factor[end + 150 :] = 1.0
    pressure = {
        name: factor * spec.contact_pressure_Pa
        for name, spec in zip(names, specs, strict=True)
    }
    res = cx.compute_mkappa(
        cable, kappa, tension_N=tension, contact_pressure_Pa=pressure
    )
    assert res.work is not None
    friction = res.work.friction_J_per_m
    remainder = res.work.integration_remainder_J_per_m
    released = drop = 0.0
    for spec in specs:
        upper0, lower0 = _bounds(spec, spec.contact_pressure_Pa)
        c0 = np.where(np.cos(spec.phi_wires) >= 0.0, upper0, lower0)
        c1 = ratio * c0
        w, k = _weight(spec), spec.k_f
        released += w * float(np.sum(c0**2 - c1**2)) / (2.0 * k)
        drop += w * float(np.sum((c0 - c1) ** 2)) / (2.0 * k * steps)
    dw = friction[-1] - friction[start - 1]
    dr = remainder[-1] - remainder[start - 1]
    assert dw > 0.0
    assert dw == pytest.approx(released - drop, rel=1e-12)
    assert dw + dr == pytest.approx(released, rel=1e-12)
    # No work after the pressure drop: the rise and the second cycle stick.
    np.testing.assert_allclose(friction[end:], friction[end - 1], rtol=1e-14)
    # Stored energy follows the bound; the work balance closes.
    stored = res.work.stored_J_per_m
    assert stored[start - 1] - stored[end - 1] == pytest.approx(released, rel=1e-12)
    balance = res.work.balance_residual_J_per_m
    assert float(np.max(np.abs(balance))) <= 1e-12 * float(np.max(stored))


# --------------------------------------------------------------------------- #
# Tension history against the fatigue path                                     #
# --------------------------------------------------------------------------- #
def _fatigue_path_moment(
    cable: Any,
    kappa: Any,
    tension: Any,
    ei_min: float,
    friction_model: str = "direction_tied",
) -> Any:
    """Armour moment from the fatigue path's friction stress, same history."""
    pressure, _ = cx.armour_contact_histories(cable, tension)
    layers = cable_to_fatigue_layers(cable, contact_tension_N=float(tension[0]))
    moment = ei_min * kappa
    zero = np.zeros_like(kappa)
    for layer in layers:
        field = wire_stress_field(
            layer,
            kappa,
            zero,
            contact_pressure_Pa=pressure[layer.name],
            friction_model=friction_model,  # type: ignore[arg-type]
            keep_friction=True,
        )
        assert field.friction_Pa is not None
        spec = layer.spec
        lever = _weight(spec) * spec.r_helix_m * np.cos(spec.phi_wires)
        moment = moment + lever @ field.friction_Pa
    return moment


@pytest.mark.parametrize("friction_model", ["direction_tied", "progressive"])
def test_tension_cycle_within_bending_cycles_is_the_fatigue_path(
    cable: Any, friction_model: str
) -> None:
    """Tension cycling at an incommensurate period inside the bending cycles:
    the M-kappa law and the fatigue friction stress are the same kernels
    (an identity; the fatigue path may use the compiled recursion)."""
    t = np.arange(1200)
    kappa = 0.03 * np.sin(2.0 * np.pi * t / 200.0)
    tension = 1.0e4 + 6.0e4 * np.sin(np.pi * t / 370.0) ** 2
    res = cx.compute_mkappa(
        cable,
        kappa,
        contact_tension_history_N=tension,
        friction_model=friction_model,  # type: ignore[arg-type]
    )
    expected = _fatigue_path_moment(
        cable, kappa, tension, res.EI_min_Nm2, friction_model
    )
    scale = float(np.max(np.abs(expected)))
    np.testing.assert_allclose(res.moment_Nm, expected, rtol=0, atol=1e-12 * scale)
    assert res.work is not None
    # Friction work never decreases; the work balance closes.
    assert float(np.min(np.diff(res.work.friction_J_per_m))) >= 0.0
    balance = res.work.balance_residual_J_per_m
    stored = res.work.stored_J_per_m
    assert float(np.max(np.abs(balance))) <= 1e-12 * float(np.max(stored))
    # The pressure is the fatigue path's contact history.
    pressure, _ = cx.armour_contact_histories(cable, tension)
    assert res.contact_pressure_Pa is not None
    for name, values in res.contact_pressure_Pa.items():
        assert np.array_equal(values, pressure[name])


def test_full_slip_cycle_work_follows_tension(cable: Any) -> None:
    """Stepped tension, two bending cycles per level, each step one sample
    after the positive extreme. In full slip a cycle drives every wire between ``+upper`` and
    ``-lower`` of the current pressure, so the second cycle at a level does
    the closed-form work ``sum w s (2 r |cos phi| kappa_a - s / k)``,
    ``s = upper + lower``, whatever came before: equal on the rising and the
    falling leg, increasing with tension while ``s < k r |cos phi| kappa_a``.
    The history is also the fatigue path's."""
    levels = [0.0, 2.0e4, 4.0e4, 8.0e4, 4.0e4, 2.0e4, 0.0]
    specs_hi = cable_to_mkappa_specs(cable, max(levels))
    kappa_a = 3.0 * max(full_slip_curvature(spec) for spec in specs_hi)
    n = 400
    cycle = kappa_a * np.cos(2.0 * np.pi * np.arange(n) / n)
    # Load from zero to the first extreme, then whole cycles per level.
    lead = np.linspace(0.0, kappa_a, 101)[:-1]
    kappa = np.concatenate([lead, np.tile(cycle, 2 * len(levels)), [kappa_a]])
    # A level holds from one sample after its first extreme to its last.
    tension = np.concatenate(
        [np.full(lead.size + 1, levels[0])]
        + [np.full(2 * n, level) for level in levels]
    )
    res = cx.compute_mkappa(cable, kappa, contact_tension_history_N=tension)
    assert res.work is not None and res.contact_pressure_Pa is not None
    friction = res.work.friction_J_per_m
    names = _names(cable)
    works = []
    for index, level in enumerate(levels):
        first = lead.size + (2 * index + 1) * n  # start of the second cycle
        last = first + n  # back at the positive extreme
        work = friction[last] - friction[first]
        works.append(work)
        expected = 0.0
        for name, spec in zip(names, cable_to_mkappa_specs(cable, level), strict=True):
            p = float(res.contact_pressure_Pa[name][first])
            assert p == pytest.approx(spec.contact_pressure_Pa, rel=1e-12)
            upper, lower = _bounds(spec, p)
            s = upper + lower
            drive = spec.r_helix_m * np.abs(np.cos(spec.phi_wires)) * kappa_a
            assert np.all(s < spec.k_f * drive)  # rising part of s (2D - s/k)
            expected += _weight(spec) * float(np.sum(s * (2.0 * drive - s / spec.k_f)))
        assert work == pytest.approx(expected, rel=1e-10)
    assert works[0] < works[1] < works[2] < works[3]
    assert works[3] > works[4] > works[5] > works[6]
    for rising, falling in ((0, 6), (1, 5), (2, 4)):
        assert works[rising] == pytest.approx(works[falling], rel=1e-12)
    expected_moment = _fatigue_path_moment(cable, kappa, tension, res.EI_min_Nm2)
    scale = float(np.max(np.abs(expected_moment)))
    np.testing.assert_allclose(
        res.moment_Nm, expected_moment, rtol=0, atol=1e-12 * scale
    )


# --------------------------------------------------------------------------- #
# Inputs                                                                       #
# --------------------------------------------------------------------------- #
def test_pressure_history_inputs_are_checked(cable: Any) -> None:
    kappa = np.linspace(0.0, 0.01, 11)
    names = _names(cable)
    good = {name: np.full(kappa.size, 1.0e6) for name in names}
    with pytest.raises(CableSchemaError, match="every structural armour layer"):
        cx.compute_mkappa(cable, kappa, contact_pressure_Pa={names[0]: good[names[0]]})
    with pytest.raises(CableSchemaError, match="unknown"):
        cx.compute_mkappa(
            cable, kappa, contact_pressure_Pa={**good, "sheath": good[names[0]]}
        )
    with pytest.raises(CableSchemaError, match="must map"):
        cx.compute_mkappa(cable, kappa, contact_pressure_Pa=[1.0])  # type: ignore[arg-type]
    for bad, error, match in (
        (np.full(kappa.size - 1, 1.0e6), CableSchemaError, "shape"),
        (np.full(kappa.size, -1.0), CableSchemaError, ">= 0"),
        (np.full(kappa.size, np.nan), PhysicalRangeError, "finite"),
    ):
        with pytest.raises(error, match=match):
            cx.compute_mkappa(cable, kappa, contact_pressure_Pa={**good, names[0]: bad})


def test_tension_history_inputs_are_checked(cable: Any) -> None:
    kappa = np.linspace(0.0, 0.01, 11)
    tension = np.full(kappa.size, 1.0e4)
    names = _names(cable)
    pressure = {name: np.full(kappa.size, 1.0e6) for name in names}
    for extra in (
        {"tension_N": 1.0e4},
        {"contact_pressure_Pa": pressure},
        {"section_state": object()},
        {"radial_state": object()},
    ):
        with pytest.raises(CableSchemaError, match="sets the contact state"):
            cx.compute_mkappa(
                cable,
                kappa,
                contact_tension_history_N=tension,
                **extra,  # type: ignore[arg-type]
            )
    with pytest.raises(CableSchemaError, match="shape"):
        cx.compute_mkappa(cable, kappa, contact_tension_history_N=tension[:-1])
    with pytest.raises(CableSchemaError, match=">= 0"):
        cx.compute_mkappa(cable, kappa, contact_tension_history_N=-tension)
    with pytest.raises(PhysicalRangeError, match="finite"):
        cx.compute_mkappa(
            cable, kappa, contact_tension_history_N=np.full(kappa.size, np.inf)
        )


def test_kernel_needs_one_pressure_entry_per_layer(cable: Any) -> None:
    specs = cable_to_mkappa_specs(cable, 0.0)
    kappa = np.linspace(0.0, 0.01, 5)
    with pytest.raises(CableInputError, match="per layer"):
        compute_mkappa_wire_slip_front(specs, kappa, contact_pressure_Pa=[None])
    mixed = compute_mkappa_wire_slip_front(
        specs, kappa, contact_pressure_Pa=[None] * len(specs)
    )
    plain = compute_mkappa_wire_slip_front(specs, kappa)
    assert np.array_equal(mixed.moment_Nm, plain.moment_Nm)
