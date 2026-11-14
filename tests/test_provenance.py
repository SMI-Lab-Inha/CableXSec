# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""L1.3/L1.4 — exception hierarchy, Provenance record, validated-range guards."""

from __future__ import annotations

import warnings

import pytest

from cablexsec.exceptions import (
    BlockedSNCurveError,
    CableSchemaError,
    CableXSecError,
    PhysicalRangeError,
)
from cablexsec.provenance import (
    ExtrapolationWarning,
    Provenance,
    ValidatedRange,
    check_range,
)

# --------------------------------------------------------------------------- #
# L1.3 — exception hierarchy                                                  #
# --------------------------------------------------------------------------- #
# Test-taxonomy category.
pytestmark = pytest.mark.analytical_sanity


def test_all_errors_subclass_cablexsec_error() -> None:
    for exc in (CableSchemaError, BlockedSNCurveError, PhysicalRangeError):
        assert issubclass(exc, CableXSecError)


def test_dual_inheritance_keeps_builtin_compat() -> None:
    """BlockedSNCurveError stays a KeyError; PhysicalRangeError stays a ValueError."""
    assert issubclass(BlockedSNCurveError, KeyError)
    assert issubclass(PhysicalRangeError, ValueError)


def test_blocked_sn_curve_error_is_the_same_object_as_sn_curves() -> None:
    """The historical import path resolves to the canonical exceptions class."""
    from cablexsec.materials.sn_curves import (
        BlockedSNCurveError as FromSNCurves,  # noqa: PLC0415
    )

    assert FromSNCurves is BlockedSNCurveError


def test_blocked_sn_curve_error_still_raised_by_registry() -> None:
    from cablexsec.materials import sn_curves  # noqa: PLC0415

    with pytest.raises(BlockedSNCurveError):
        sn_curves.get_sn_curve("cigre_tb623_lead")
    # still catchable as the historical KeyError (lead stays BLOCKED)
    with pytest.raises(KeyError):
        sn_curves.get_sn_curve("cigre_tb623_lead")


# --------------------------------------------------------------------------- #
# L1.2 — Provenance record                                                     #
# --------------------------------------------------------------------------- #
def test_provenance_str_includes_all_sections() -> None:
    p = Provenance(
        method="full-slip EI",
        citations=("[WitzTan1992b]",),
        gates=("A8", "BM1"),
        limitations=("calibration-free EI_min only",),
    )
    s = str(p)
    assert "full-slip EI" in s
    assert "WitzTan1992b" in s
    assert "A8" in s
    assert "LIMITATIONS" in s and "calibration-free" in s


def test_provenance_is_frozen() -> None:
    p = Provenance(method="x")
    with pytest.raises((AttributeError, TypeError)):
        p.method = "y"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# L1.4 — validated-range guard (warn, do not raise)                            #
# --------------------------------------------------------------------------- #
def test_check_range_silent_within_bounds() -> None:
    vr = ValidatedRange("kappa", 0.0, 0.05, "[A11]")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail
        assert check_range(0.03, vr) is True


def test_check_range_warns_above_and_below() -> None:
    vr = ValidatedRange("kappa [1/m]", 0.0, 0.05, "[A11] tested |kappa|<=0.05")
    for value in (0.5, -1.0):
        with pytest.warns(ExtrapolationWarning, match="validated range"):
            assert check_range(value, vr) is False


def test_check_range_message_names_bound_and_source() -> None:
    vr = ValidatedRange("temperature [degC]", -20.0, 90.0, "[Mazzanti2007] Table I")
    with pytest.warns(ExtrapolationWarning) as rec:
        check_range(120.0, vr)
    msg = str(rec[0].message)
    assert "Mazzanti2007" in msg and "90" in msg
