# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Input ownership and catchable engineering failures for application clients."""

import copy
import pickle
from dataclasses import replace

import numpy as np
import pytest

import cablexsec as cx
from cablexsec import CableSchemaError, CableXSecError, LoadCaseSeries
from cablexsec.analysis.fretting_swt import swt_life
from cablexsec.materials.library import get_material, get_property
from cablexsec.materials.sn_curves import SNcurve, get_sn_curve, get_swt_life
from cablexsec.mechanics.hertz_contact import HertzContact
from cablexsec.mechanics.return_mapping import CoulombElement

FIELDS = (
    "time_s",
    "curvature_x_1_m",
    "curvature_y_1_m",
    "tension_N",
    "temperature_C",
    "axial_strain",
)


def history(**overrides):
    values = {name: np.arange(3.0) for name in FIELDS}
    values.update(tension_kind="true", axial_strain_kind="direct")
    values.update(overrides)
    return LoadCaseSeries(**values)


@pytest.mark.parametrize("name", FIELDS)
@pytest.mark.parametrize("layout", ["contiguous", "strided", "readonly"])
def test_history_owns_snapshot_independent_of_caller(name, layout):
    source = np.arange(6.0)[::2] if layout == "strided" else np.arange(3.0)
    expected = source.copy()
    if layout == "readonly":
        source.flags.writeable = False
    series = history(**{name: source})
    stored = getattr(series, name)
    np.testing.assert_array_equal(stored, expected)
    assert not np.shares_memory(stored, source)
    assert source.flags.writeable == (layout != "readonly")
    if layout != "readonly":
        source[1] = np.nan
        np.testing.assert_array_equal(stored, expected)
    with pytest.raises(ValueError):
        stored[1] = 99
    with pytest.raises(ValueError):
        stored.setflags(write=True)


@pytest.mark.parametrize("name", FIELDS)
def test_replace_validates_edits_and_keeps_original(name):
    original = history()
    edited = getattr(original, name).copy() + 10
    updated = replace(original, **{name: edited})
    edited[0] = -999
    np.testing.assert_array_equal(getattr(original, name), [0, 1, 2])
    np.testing.assert_array_equal(getattr(updated, name), [10, 11, 12])
    with pytest.raises(CableSchemaError, match=name):
        replace(original, **{name: [0, np.nan, 2]})


def test_replace_rechecks_time_order_and_shapes():
    original = history()
    with pytest.raises(CableSchemaError, match="strictly increasing"):
        replace(original, time_s=[0, 99, 2])
    with pytest.raises(CableSchemaError, match="length"):
        replace(original, tension_N=[1, 2])


@pytest.mark.parametrize("name", FIELDS)
@pytest.mark.parametrize("bad", [[0, 1j, 2], [0, "bad", 2], [0, 10**400, 2]])
def test_invalid_array_conversion_has_field_specific_error(name, bad):
    with pytest.raises(CableSchemaError, match=name):
        history(**{name: bad})


def test_force_conversion_and_csv_preserve_snapshot_contract(tmp_path):
    original = history(tension_kind="effective", axial_strain=None)
    converted = original.with_true_tension(10.0, 2.0)
    np.testing.assert_array_equal(original.tension_N, [0, 1, 2])
    np.testing.assert_array_equal(converted.tension_N, [-20, -19, -18])
    restored = cx.read_series_csv(
        cx.write_series_csv(converted, tmp_path / "loads.csv")
    )
    assert restored.axial_strain is None
    assert restored.tension_kind == "true"
    for name in FIELDS[:-1]:
        np.testing.assert_array_equal(getattr(restored, name), getattr(converted, name))
        assert not getattr(restored, name).flags.writeable


@pytest.mark.parametrize(
    "call, builtin, message",
    [
        (
            lambda: cx.compute_long_term_fatigue([], exposure_years=20),
            ValueError,
            "at least one",
        ),
        (lambda: SNcurve("bad", "MPa", -1, 12), ValueError, "slopes"),
        (
            lambda: SNcurve("ok", "MPa", 3, 12).cycles_to_failure(np.nan),
            ValueError,
            "finite",
        ),
        (lambda: CoulombElement(k=-1), ValueError, "positive"),
        (lambda: get_material("missing-material"), KeyError, "missing-material"),
        (
            lambda: get_property("galvanised_steel_armour", "missing-property"),
            KeyError,
            "missing-property",
        ),
        (lambda: get_sn_curve("missing-curve"), KeyError, "missing-curve"),
        (lambda: get_swt_life("missing-curve"), KeyError, "missing-curve"),
        (
            lambda: cx.load_cable(
                cx.menard2023_yaml_path(), geometry_override=None
            ).get_layer("missing-layer"),
            KeyError,
            "missing-layer",
        ),
        (
            lambda: cx.compute_hertz_axis_stress(HertzContact(1, 1, 1, 0, 0, 0), [-1]),
            ValueError,
            "depth",
        ),
        (
            lambda: cx.compute_hertz_subsurface_stress(
                HertzContact(1, 1, 1, 0, 0, 0),
                [0, 0, 0.2],
                initial_order=4,
                max_order=8,
            ),
            RuntimeError,
            "did not converge",
        ),
    ],
)
def test_engineering_failures_support_common_and_builtin_handlers(
    call, builtin, message
):
    with pytest.raises(builtin, match=message) as caught:
        call()
    assert isinstance(caught.value, CableXSecError)


@pytest.mark.parametrize(
    "name,builtin",
    [
        ("CableInputError", ValueError),
        ("CableLookupError", KeyError),
        ("CableSolverError", RuntimeError),
    ],
)
def test_error_types_are_public_and_builtin_compatible(name, builtin):
    from cablexsec import api

    error_type = getattr(cx, name)
    assert getattr(api, name) is error_type
    assert issubclass(error_type, builtin)
    assert issubclass(error_type, CableXSecError)


def test_filesystem_errors_are_not_disguised(tmp_path):
    with pytest.raises(FileNotFoundError) as caught:
        cx.load_cable(tmp_path / "absent.yaml", geometry_override=None)
    assert not isinstance(caught.value, CableXSecError)


@pytest.mark.parametrize(
    "clone", [copy.copy, copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))]
)
def test_worker_transfer_and_copy_reconstruct_readonly_snapshots(clone):
    original = history()
    restored = clone(original)
    for name in FIELDS:
        np.testing.assert_array_equal(getattr(restored, name), getattr(original, name))
        assert not np.shares_memory(getattr(restored, name), getattr(original, name))
        with pytest.raises(ValueError):
            getattr(restored, name).setflags(write=True)


@pytest.mark.parametrize("row", ["0,0,0,nope", "0,0", "0,0,0,1,bad"])
def test_csv_numeric_errors_are_catchable_and_keep_cause(tmp_path, row):
    path = tmp_path / "invalid.csv"
    path.write_text(
        "time_s,curvature_x_1_m,curvature_y_1_m,true_tension_N,temperature_C\n"
        + row
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(CableXSecError, match="line 2") as caught:
        cx.read_series_csv(path)
    assert isinstance(caught.value.__cause__, (ValueError, TypeError))


def test_swt_iteration_exhaustion_is_catchable_and_keeps_cause():
    with pytest.raises(RuntimeError, match="iteration limit") as caught:
        swt_life(1e6, get_swt_life("copper_poon2022"), max_iter=1)
    assert isinstance(caught.value, CableXSecError)
    assert isinstance(caught.value.__cause__, RuntimeError)
