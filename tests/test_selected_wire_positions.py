# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Selected-position fatigue (``wire_positions``) equals the full evaluation.

Every wire's friction recursion, field and rainflow count are independent of
the other wires, so the damage of a selected wire must equal its damage in
the evaluation of every wire bit for bit, in every friction model and
fatigue branch that the selection reaches.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace

import numpy as np
import pytest

import cablexsec as cx
import cablexsec.analysis.fatigue as fatigue_module
from cablexsec.analysis.fatigue import (
    FatigueLayer,
    compute_fatigue,
    select_wires,
)
from cablexsec.analysis.long_term import FatigueLoadCase, compute_long_term_fatigue
from cablexsec.analysis.termination import TerminationSpec
from cablexsec.exceptions import CableInputError
from cablexsec.materials.sn_curves import SNcurve, get_sn_curve, get_swt_life
from cablexsec.mechanics.friction_law import FrictionLaw
from cablexsec.mechanics.mkappa import HelicalLayerSpec

PER_WIRE = (
    "D_bulk_per_wire",
    "D_fretting_per_wire",
    "D_term_per_wire",
    "D_total_per_wire",
    "D_bulk_per_fibre",
    "fibre_critical_deg_per_wire",
    "D_bulk_with_termination_per_fibre",
    "termination_fibre_critical_deg_per_wire",
)


@pytest.fixture(scope="module")
def cable() -> cx.CableData:
    return cx.load_cable(cx.iea_task49_66kv_yaml_path())


def _history(n: int = 300) -> tuple[np.ndarray, ...]:
    # Two incommensurate periods: a non-planar path with partial and full slip.
    t = np.arange(n) * 0.1
    kx = 0.004 * np.sin(2 * math.pi * t / 10) + 0.001 * np.sin(2 * math.pi * t / 2.3)
    ky = 0.002 * np.cos(2 * math.pi * t / 9)
    tension = 1.0e5 + 1.0e4 * np.sin(2 * math.pi * t / 11)
    return t, kx, ky, tension


def _assert_rows_equal(sub, full) -> None:
    assert sub.wire_index
    for name, rows in sub.wire_index.items():
        for key in PER_WIRE:
            np.testing.assert_array_equal(
                getattr(sub, key)[name], getattr(full, key)[name][rows]
            )
    np.testing.assert_array_equal(sub.fibre_angles_deg, full.fibre_angles_deg)


def test_select_wires_counts_and_azimuths() -> None:
    np.testing.assert_array_equal(select_wires(12, 4), [0, 3, 6, 9])
    np.testing.assert_array_equal(select_wires(139, 1), [0])
    assert select_wires(139, 12).size == 12
    np.testing.assert_array_equal(select_wires(10, 10), np.arange(10))
    # Nearest wire, wrapped into [0, 360), each wire once, ascending.
    np.testing.assert_array_equal(
        select_wires(12, [359.0, 44.0, 46.0, -30.0, 720.0]), [0, 1, 2, 11]
    )
    for bad in (0, 13, True, [], [math.nan], [[0.0]], "x"):
        with pytest.raises(CableInputError):
            select_wires(12, bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("model", ["direction_tied", "distributed", "progressive"])
def test_selected_wires_equal_the_full_evaluation(cable, model) -> None:
    _, kx, ky, tension = _history()
    kwargs = {"tension_N": tension, "friction_model": model, "n_fibres": 8}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = cx.compute_fatigue(cable, kx, ky, **kwargs)
        sub = cx.compute_fatigue(cable, kx, ky, wire_positions=12, **kwargs)
    _assert_rows_equal(sub, full)
    layers = [x.name for x in cable.helical_layers if x.structural]
    assert set(sub.wire_index) == set(layers)
    for name in layers:
        n = cable.get_layer(name).n_wires
        np.testing.assert_array_equal(sub.wire_index[name], select_wires(n, 12))
        np.testing.assert_allclose(
            sub.wire_azimuth_deg[name], 360.0 * sub.wire_index[name] / n, rtol=0
        )
    assert not full.wire_index and not full.wire_azimuth_deg


def test_reference_kernels_agree_row_for_row(cable, monkeypatch) -> None:
    # Numba off (interpreted direction-tied recursion and rainflow counter).
    monkeypatch.setattr(fatigue_module, "_FAST_SLIP", False)
    _, kx, ky, tension = _history(200)
    kwargs = {
        "tension_N": tension,
        "friction_model": "direction_tied",
        "n_fibres": 4,
        "rainflow_counter": "reference",
    }
    full = cx.compute_fatigue(cable, kx, ky, **kwargs)
    sub = cx.compute_fatigue(cable, kx, ky, wire_positions=[0.0, 90.0], **kwargs)
    _assert_rows_equal(sub, full)


def test_friction_law_and_sn_options_agree(cable) -> None:
    t, kx, ky, tension = _history(300)
    law = FrictionLaw(static_factor=1.2, stick_time_s=0.5, rate_coefficient=0.01)
    names = [x.name for x in cable.helical_layers if x.structural]
    positions = {names[0]: 5, names[1]: [10.0, 200.0, 300.0]}
    for kwargs in (
        {"friction_law": law, "time_s": t},
        {"mean_stress": "goodman"},
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full = cx.compute_fatigue(
                cable,
                kx,
                ky,
                tension_N=tension,
                friction_model="direction_tied",
                n_fibres=4,
                **kwargs,
            )
            sub = cx.compute_fatigue(
                cable,
                kx,
                ky,
                tension_N=tension,
                friction_model="direction_tied",
                n_fibres=4,
                wire_positions=positions,
                **kwargs,
            )
        _assert_rows_equal(sub, full)
        assert sub.wire_index[names[0]].size == 5
        assert sub.wire_index[names[1]].size == 3


def _copper_layer(n_wires: int = 12) -> FatigueLayer:
    shape = 1.0 + 0.5 * np.cos(2 * np.pi * np.arange(n_wires) / n_wires)
    return FatigueLayer(
        spec=HelicalLayerSpec(
            n_wires=n_wires,
            r_helix_m=0.05,
            lay_angle_rad=math.radians(12.0),
            E_wire_Pa=97.0e9,
            A_wire_m2=math.pi * 0.003**2 / 4,
            xi=0.98,
            mu=0.2,
            F_n_axi_N=50.0,
            shape_g=shape,
        ),
        sn_bulk=get_sn_curve("nasution_2014_copper"),
        sigma_UTS_Pa=220e6,
        name="cu",
        swt_const=get_swt_life("copper_poon2022"),
        contact_p0_Pa=26.0e6,
        fretting_mu=0.8,
        contact_nu=0.34,
        E_eff_axial_Pa=97.0e9,
        K_t_contact_N_per_m=1.0e6,
        crossing_density_per_m=50.0,
        contact_line_load_N_per_m=100.0,
        contact_shape_g=shape,
        above_core_mask=np.arange(n_wires) % 3 == 0,
    )


@pytest.mark.parametrize("copper", ["slip_cycle", "legacy_gross_sliding"])
def test_contact_life_branches_agree(copper) -> None:
    t = np.linspace(0, 10 * math.pi, 1501)
    kx, ky = 0.01 * np.sin(t), 0.004 * np.cos(0.7 * t)
    layer = _copper_layer()
    full = compute_fatigue([layer], kx, ky, copper_contact_model=copper)
    sub = compute_fatigue(
        [layer], kx, ky, copper_contact_model=copper, wire_positions=[30.0, 95.0]
    )
    _assert_rows_equal(sub, full)
    assert np.all(sub.D_fretting_per_wire["cu"] > 0)


def test_termination_clamp_and_scf_agree() -> None:
    t = np.linspace(0, 6 * math.pi, 300)
    kx, ky = 0.01 * np.sin(t), 0.004 * np.cos(0.7 * t)
    base = _copper_layer()
    layer = replace(
        base,
        spec=replace(base.spec, contact_pressure_Pa=2.0e6),
        swt_const=None,
        contact_p0_Pa=0.0,
        SCF=1.5,
        termination=True,
    )
    spec = TerminationSpec(anchor="clamped", distance_m=0.2)
    full = compute_fatigue([layer], kx, ky, termination=spec)
    sub = compute_fatigue([layer], kx, ky, termination=spec, wire_positions=4)
    assert full.clamp_correction["cu"]["applied"]
    assert sub.clamp_correction == full.clamp_correction
    _assert_rows_equal(sub, full)
    assert np.all(sub.D_term_per_wire["cu"] > 0)


def test_critical_wire_is_named_by_its_number() -> None:
    t = np.linspace(0, 10 * math.pi, 1501)
    kx, ky = 0.01 * np.sin(t), 0.004 * np.cos(0.7 * t)
    layer = _copper_layer()
    full = compute_fatigue([layer], kx, ky)
    k = int(np.argmax(full.D_total_per_wire["cu"]))
    # A selection holding the critical wire and one other.
    other = (k + 5) % 12
    sub = compute_fatigue([layer], kx, ky, wire_positions=[30.0 * other, 30.0 * k])
    assert sub.critical_layer == "cu"
    assert sub.phi_critical_deg == pytest.approx(full.phi_critical_deg, abs=0)
    assert sub.above_core == full.above_core


def test_steel_slip_amplitude_branch_agrees() -> None:
    custom = SNcurve(
        name="test_only_slip_curve",
        stress_unit="um",
        m1=2.0,
        log_C1=10.0,
        threshold=0.0,
    )
    layer = replace(_copper_layer(), swt_const=None, sn_fretting=custom)
    t = np.linspace(0, 6 * math.pi, 900)
    kx, ky = 0.01 * np.sin(t), 0.003 * np.cos(t)
    full = compute_fatigue([layer], kx, ky)
    sub = compute_fatigue([layer], kx, ky, wire_positions=3)
    _assert_rows_equal(sub, full)
    assert np.all(sub.D_fretting_per_wire["cu"] > 0)


def test_wire_positions_are_checked(cable) -> None:
    _, kx, ky, tension = _history(200)
    with pytest.raises(CableInputError, match="exactly the analysed layers"):
        cx.compute_fatigue(
            cable,
            kx,
            ky,
            tension_N=tension,
            friction_model="direction_tied",
            wire_positions={"inner_armour": 4},
        )
    with pytest.raises(CableInputError, match="between 1 and"):
        cx.compute_fatigue(
            cable,
            kx,
            ky,
            tension_N=tension,
            friction_model="direction_tied",
            wire_positions=1000,
        )


def test_long_term_needs_the_same_wires(cable) -> None:
    _, kx, ky, tension = _history(300)
    run = {"tension_N": tension, "friction_model": "direction_tied", "n_fibres": 4}
    a = cx.compute_fatigue(cable, kx, ky, wire_positions=6, **run)
    b = cx.compute_fatigue(cable, 0.5 * kx, ky, wire_positions=6, **run)
    c = cx.compute_fatigue(cable, kx, ky, wire_positions=7, **run)
    cases = [
        FatigueLoadCase("a", a, 30.0, 0.5, "id"),
        FatigueLoadCase("b", b, 30.0, 0.5, "id"),
    ]
    result = compute_long_term_fatigue(cases, exposure_years=1.0)
    layer = result.critical_layer
    rows = result.annual_damage_per_wire[layer]
    row = int(np.argmax(rows))
    assert result.critical_wire_index == int(a.wire_index[layer][row])
    np.testing.assert_array_equal(result.wire_index[layer], a.wire_index[layer])
    with pytest.raises(CableInputError, match="same wires"):
        compute_long_term_fatigue(
            [cases[0], FatigueLoadCase("c", c, 30.0, 0.5, "id")], exposure_years=1.0
        )


def test_selection_survives_a_project_round_trip() -> None:
    from cablexsec import _project_codec as codec

    t = np.linspace(0, 6 * math.pi, 200)
    result = compute_fatigue(
        [_copper_layer()], 0.01 * np.sin(t), 0.003 * np.cos(t), wire_positions=3
    )
    back = codec.unpack(codec.parse(codec.canonical(codec.pack(result))))
    np.testing.assert_array_equal(back.wire_index["cu"], [0, 4, 8])
    assert back.wire_index["cu"].dtype == np.int64
    np.testing.assert_array_equal(back.wire_azimuth_deg["cu"], [0.0, 120.0, 240.0])
