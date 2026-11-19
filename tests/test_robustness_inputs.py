# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Adversarial inputs of the public API: each case failed, or answered
differently with and without the compiled kernels, before 0.2.0.

Every invalid input is refused early with a message naming it; every valid
one gives a finite answer, the same (to round-off) on both kernel paths.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from _bundled import bundled
from pydantic import ValidationError

import cablexsec as cx
from cablexsec.analysis.fatigue import _first_maximum
from cablexsec.analysis.sea_state_batch import SeaStateBatchOptions, SeaStateRun
from cablexsec.exceptions import CableInputError, CableSchemaError
from cablexsec.mechanics.papailiou import wire_slip_stress_bounds

CABLE = "inter_array_single_core_33kv"


def _history(n=64):
    t = np.linspace(0.0, 6.3, n)
    return 0.02 * np.sin(t), 0.01 * np.cos(t), np.full(n, 2.0e5)


# --------------------------------------------------------------------------- #
# A very large friction coefficient                                           #
# --------------------------------------------------------------------------- #
def test_very_large_mu_sticks_on_both_kernel_paths(monkeypatch):
    """mu = 1e6 overflowed the capstan factor of the reference return map
    ("wire history exceeds floating-point range") while the compiled path
    held the wires stuck. Both now give the stuck answer, that of mu = 1."""
    cable = bundled(CABLE)
    kx, ky, ten = _history()
    reference = cx.compute_fatigue(
        cable, kx, ky, tension_N=ten, mu=1.0, friction_model="direction_tied"
    )
    import cablexsec.analysis.fatigue as fatigue

    # The compiled kernels where numba runs them, and the reference kernels.
    for compiled in sorted({fatigue._FAST_SLIP, False}):
        monkeypatch.setattr(fatigue, "_FAST_SLIP", compiled)
        result = cx.compute_fatigue(
            cable, kx, ky, tension_N=ten, mu=1e6, friction_model="direction_tied"
        )
        for name, damage in result.D_total_per_wire.items():
            scale = float(np.max(reference.D_total_per_wire[name]))
            np.testing.assert_allclose(
                damage, reference.D_total_per_wire[name], rtol=0.0, atol=1e-12 * scale
            )


def test_overflowing_capstan_bound_is_finite_and_unreachable():
    from cablexsec.cable_adapters import cable_to_mkappa_specs

    spec = cable_to_mkappa_specs(bundled(CABLE), 1e5, mu=1e6)[0]
    upper, lower = wire_slip_stress_bounds(spec, 0.0)
    assert np.isfinite(upper).all() and np.max(upper) >= 1e300
    assert np.isfinite(lower).all() and np.all(lower >= 0.0)


# --------------------------------------------------------------------------- #
# Ties of the critical location                                               #
# --------------------------------------------------------------------------- #
def test_critical_location_is_the_first_of_tied_values():
    """Damage at symmetric fibres differs by round-off between the kernel
    paths; argmax then reported 63 deg on one path and 243 deg on the other.
    Values within 1e-12 of the maximum are a tie, won by the first."""
    values = np.array([[1.0, 3.0, 3.0 * (1 + 1e-14), 2.0], [0.0, 0.0, 0.0, 0.0]])
    assert _first_maximum(values).tolist() == [1, 0]
    assert _first_maximum(np.array([[1.0, 3.0, 3.0 * (1 + 1e-9)]])).tolist() == [2]


# --------------------------------------------------------------------------- #
# Thermal and bending-law arguments                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"current_A": -1.0}, "current_A must not be negative"),
        ({"ambient_C": -300.0}, "ambient_C = -300 °C is below absolute zero"),
        ({"conductor_temp_C": -274.0}, "conductor_temp_C = -274 °C is below"),
        ({"n_cores": 0}, "n_cores must be a positive integer"),
        ({"n_cores": 1.5}, "n_cores must be a positive integer"),
        ({"n_cores": True}, "n_cores must be a positive integer"),
    ],
)
def test_thermal_arguments_are_checked(kwargs, match):
    """A zero core count divided by zero (ZeroDivisionError); a negative
    current and temperatures below absolute zero were accepted."""
    cable = bundled(CABLE)
    with pytest.raises(CableInputError, match=match):
        cx.compute_thermal(
            cable, **{"current_A": 600.0, "conductor_area_m2": 3e-4, **kwargs}
        )


@pytest.mark.parametrize("n_points", [20.5, "20", True, 2])
def test_bending_law_point_count_is_an_integer(n_points):
    """A fractional point count raised numpy's TypeError."""
    with pytest.raises(CableInputError, match="n_points must be an integer"):
        cx.export_bending_law(
            bundled(CABLE), tensions_N=[0.0], kappa_max_1_m=0.1, n_points=n_points
        )


# --------------------------------------------------------------------------- #
# Cable definitions                                                           #
# --------------------------------------------------------------------------- #
def _yaml(tmp_path, change):
    path = Path(cx.__file__).parent / "cables" / f"{CABLE}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for layer in data["layers"]:
        if layer["name"] == change[0]:
            layer[change[1]] = change[2]
    out = tmp_path / "cable.yaml"
    out.write_text(yaml.safe_dump(data), encoding="utf-8")
    return out


@pytest.mark.parametrize(
    ("change", "where", "detail"),
    [
        (
            ("xlpe_insulation", "r_outer_m", math.nan),
            "layer 2 'xlpe_insulation'",
            "finite",
        ),
        (
            ("inner_armour", "d_wire_m", -1.0),
            "layer 6 'inner_armour'",
            "d_wire_m must be > 0",
        ),
        (
            ("inner_armour", "lay_angle_deg", 90.0),
            "layer 6 'inner_armour'",
            r"\[0, 90\)",
        ),
        (("inner_armour", "n_wires", 2.5), "layer 6 'inner_armour'", "valid integer"),
        (
            ("inner_armour", "n_wires", 0),
            "layer 6 'inner_armour'",
            "n_wires must be > 0",
        ),
        (("outer_armour", "bogus_key", 1), "layer 7 'outer_armour'", "Extra inputs"),
    ],
)
def test_cable_errors_name_the_layer(tmp_path, change, where, detail):
    """The error of a layer was titled with the union of every layer kind and
    located within the kind only; it now names the layer and the file."""
    with pytest.raises(ValidationError) as caught:
        cx.load_cable(_yaml(tmp_path, change), geometry_override=None)
    text = str(caught.value)
    assert f"for {where} of cable.yaml" in text
    assert f"layers.{where.split()[1]}." in text
    assert caught.match(detail)


@pytest.mark.parametrize(
    "change",
    [
        ("inner_armour", "lay_angle_deg", 1e-3),
        ("inner_armour", "lay_angle_deg", 89.0),
        ("inner_armour", "n_wires", 1),
    ],
)
def test_extreme_but_valid_geometry_gives_a_finite_answer(tmp_path, change):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # an overfilled near-hoop layer warns
        cable = cx.load_cable(_yaml(tmp_path, change), geometry_override=None)
        kx, ky, ten = _history(32)
        result = cx.compute_fatigue(
            cable, kx, ky, tension_N=ten, friction_model="direction_tied"
        )
    assert all(np.isfinite(v).all() for v in result.D_total_per_wire.values())
    assert np.isfinite(cx.compute_stiffness(cable).ei_min_Nm2)


# --------------------------------------------------------------------------- #
# Exchange CSV                                                                #
# --------------------------------------------------------------------------- #
_HEAD = "time_s,curvature_x_1_m,curvature_y_1_m,true_tension_N\n"
_ROWS = "".join(f"{0.1 * k:.6e},{0.01 * k:.6e},0,1.0e5\n" for k in range(5))


@pytest.mark.parametrize(
    "data",
    [
        (_HEAD + _ROWS).encode(),
        (_HEAD + _ROWS).replace("\n", "\r\n").encode(),
        b"\xef\xbb\xbf" + (_HEAD + _ROWS).encode(),
        (_HEAD + _ROWS + "\n , ,\n\n").encode(),
        (_HEAD + _ROWS + ",,,,,,\n").encode(),  # more blank cells than columns
        (_HEAD + _ROWS.replace("e", "E")).encode(),
    ],
)
def test_series_csv_reads_line_ends_bom_and_blank_rows(tmp_path, data):
    path = tmp_path / "s.csv"
    path.write_bytes(data)
    series = cx.read_series_csv(path)
    assert series.n_steps == 5 and series.tension_kind == "true"


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ((_HEAD + _ROWS).encode("utf-16"), "not UTF-8 text"),
        ("timé,x\n1,2\n".encode("latin-1"), "not UTF-8 text"),
        ((_HEAD + _ROWS + "0.6,1_0,0,1e5\n").encode(), "line 7"),
        ((_HEAD + _ROWS + "0.6,1e-2,0\n").encode(), "line 7"),
        ((_HEAD + _ROWS + "0.6,1e-2,0,1e5,7\n").encode(), "line 7: more cells"),
    ],
)
def test_series_csv_errors_name_the_file(tmp_path, data, match):
    """A file that is not UTF-8 raised UnicodeDecodeError."""
    path = tmp_path / "s.csv"
    path.write_bytes(data)
    with pytest.raises(CableSchemaError, match=match):
        cx.read_series_csv(path)


# --------------------------------------------------------------------------- #
# Sea-state batch options and runs                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"exposure_years": 0.0}, "exposure_years must be finite and positive"),
        ({"exposure_years": math.nan}, "exposure_years"),
        ({"design_fatigue_factor": 0.5}, "design_fatigue_factor must be at least one"),
        ({"design_fatigue_factor": math.inf}, "design_fatigue_factor"),
        ({"n_fibres": 2}, "n_fibres must be an integer of at least 4"),
        ({"n_fibres": 8.0}, "n_fibres"),
        ({"chunk_rows": 4}, "chunk_rows"),
        ({"mu": -0.1}, "mu must be finite and >= 0"),
        ({"mu": math.nan}, "mu must be finite"),
    ],
)
def test_batch_options_are_checked_before_any_run(kwargs, match):
    """These reached compute_fatigue or compute_long_term_fatigue only after
    every run had been read and evaluated."""
    with pytest.raises(CableInputError, match=match):
        SeaStateBatchOptions(**kwargs)


@pytest.mark.parametrize("line_id", [0, -1, True, 1.0])
def test_batch_run_line_id_is_checked(tmp_path, line_id):
    with pytest.raises(CableInputError, match="line_id must be a positive integer"):
        SeaStateRun("r", tmp_path / "o", tmp_path / "d.dat", 1.0, line_id=line_id)


def test_very_large_mu_gives_the_stuck_moment_curvature_loop():
    """compute_mkappa with mu = 1e6 raised "wire history exceeds
    floating-point range" on both kernel paths; the wires simply stick."""
    cable = bundled(CABLE)
    kappa = np.concatenate([np.linspace(0, 0.05, 30), np.linspace(0.05, -0.05, 60)])
    stuck = cx.compute_mkappa(cable, kappa, tension_N=1e5, mu=1.0)
    huge = cx.compute_mkappa(cable, kappa, tension_N=1e5, mu=1e6)
    scale = float(np.max(np.abs(stuck.moment_Nm)))
    np.testing.assert_allclose(
        huge.moment_Nm, stuck.moment_Nm, rtol=0, atol=1e-12 * scale
    )
