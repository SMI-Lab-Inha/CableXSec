# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Invalid histories must fail before CSV export or physical recovery."""

import numpy as np
import pytest

from cablexsec.exceptions import CableSchemaError
from cablexsec.io.series import LoadCaseSeries, read_series_csv, write_series_csv


def history(**overrides):
    values = {
        "time_s": np.arange(3.0),
        "curvature_x_1_m": np.zeros(3),
        "curvature_y_1_m": np.zeros(3),
        "tension_N": np.ones(3),
    }
    values.update(overrides)
    return LoadCaseSeries(**values)


@pytest.mark.parametrize(
    "field",
    [
        "time_s",
        "curvature_x_1_m",
        "curvature_y_1_m",
        "tension_N",
        "temperature_C",
        "axial_strain",
    ],
)
@pytest.mark.parametrize("bad", [[0.0, np.nan, 2.0], [0.0, np.inf, 2.0], [[0, 1, 2]]])
def test_rejects_nonfinite_and_multidimensional_fields(field, bad):
    with pytest.raises(CableSchemaError, match=field):
        history(**{field: bad})


@pytest.mark.parametrize("times", [[], [0, 0, 1], [2, 1, 0]])
def test_rejects_empty_or_nonincreasing_times(times):
    with pytest.raises(CableSchemaError, match="time_s"):
        history(time_s=times)


def test_list_inputs_roundtrip(tmp_path):
    original = history(tension_N=[1, 2, 3])
    recovered = read_series_csv(write_series_csv(original, tmp_path / "history.csv"))
    np.testing.assert_array_equal(recovered.tension_N, [1, 2, 3])
    assert recovered.dt_s == 1.0


def test_partial_temperature_column_is_rejected(tmp_path):
    path = tmp_path / "history.csv"
    path.write_text(
        "time_s,curvature_x_1_m,curvature_y_1_m,true_tension_N,temperature_C\n"
        "0,0,0,1,20\n1,0,0,1,\n",
        encoding="utf-8",
    )
    with pytest.raises(CableSchemaError, match="temperature_C"):
        read_series_csv(path)
