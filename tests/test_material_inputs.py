# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""S-N curves, corrected Miner sums, wear data and user materials: input rules.

Verification: invalid definitions are rejected with their own message, the
closed-form Miner sum of a single-slope curve under a stress factor
(``D ~ k^m``), the Nishioka--Hirakawa notch factor at its limits, and the
packaged-template selectors.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.analysis.sn_assessment import (
    ComponentSCF,
    corrected_damage,
    corrected_damage_rows,
    needs_correction,
)
from cablexsec.exceptions import CableInputError, CableLookupError
from cablexsec.materials import library
from cablexsec.materials.mean_stress import MeanStressSpec
from cablexsec.materials.sn_curves import SNcurve, register_sn_curve
from cablexsec.materials.wear import (
    FrettingKnockdown,
    WearCoefficient,
    get_knockdown,
    get_wear_coefficient,
    wear_pair_for,
)

MPA = 1.0e6


def _single(**kwargs) -> SNcurve:
    base = {"name": "unit_single", "stress_unit": "MPa", "m1": 3.0, "log_C1": 12.0}
    base.update(kwargs)
    return SNcurve(**base)


def _cycles(range_mpa: float, n: int) -> np.ndarray:
    return np.tile([-range_mpa / 2, range_mpa / 2], n) * MPA


# --------------------------------------------------------------------------- #
# Component SCFs and corrected Miner sums                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan, True, "2"])
def test_component_scf_rejects_non_positive_or_non_numeric(value):
    with pytest.raises(CableInputError, match="axial_mean must be finite and positive"):
        ComponentSCF(axial_mean=value)


def test_component_scf_needs_a_source_unless_identity():
    with pytest.raises(CableInputError, match="need their source"):
        ComponentSCF(friction_dynamic=1.5)
    identity = ComponentSCF()
    assert identity.is_identity and identity.describe() == "no component SCF"
    scf = ComponentSCF.uniform(bending=1.2, source="assumed")
    assert not scf.is_identity
    assert needs_correction(_single(), None, scf)
    assert not needs_correction(_single(), None, identity)


def test_stress_factor_scales_single_slope_damage_by_k_to_the_m():
    curve = _single()
    history = _cycles(80.0, 10)
    base = corrected_damage(history, curve)
    scaled = corrected_damage(history, curve, stress_factor=1.5)
    assert base > 0.0
    assert scaled / base == pytest.approx(1.5**3, rel=1e-12)
    # The row form reproduces the scalar form row by row.
    rows = corrected_damage_rows(np.vstack([history, 2 * history]), curve)
    assert rows[0] == pytest.approx(base, rel=1e-12)
    assert rows[1] / rows[0] == pytest.approx(8.0, rel=1e-12)


def test_constant_history_has_no_cycles_and_no_damage():
    curve = _single()
    flat = np.full(16, 50.0 * MPA)
    assert corrected_damage(flat, curve) == 0.0
    assert corrected_damage(flat, curve, mean_stress=MeanStressSpec()) == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"stress_factor": 0.0}, "stress_factor must be finite and positive"),
        ({"stress_factor": math.nan}, "stress_factor must be finite and positive"),
    ],
)
def test_corrected_damage_rejects_invalid_stress_factor(kwargs, message):
    with pytest.raises(CableInputError, match=message):
        corrected_damage(_cycles(80.0, 2), _single(), **kwargs)


def test_corrected_damage_rejects_invalid_inputs():
    fretting = SNcurve(name="slip", stress_unit="um", m1=2.0, log_C1=10.0)
    with pytest.raises(CableInputError, match="requires an S-N curve in MPa"):
        corrected_damage(_cycles(80.0, 2), fretting)
    with pytest.raises(CableInputError, match="one-dimensional and finite"):
        corrected_damage(np.array([0.0, math.nan, 1.0]), _single())
    with pytest.raises(CableInputError, match="stress histories must be two-dim"):
        corrected_damage_rows(_cycles(80.0, 2), _single())
    # A curve whose lives leave the floating-point range is refused rather
    # than summed as zero damage.
    absurd = _single(log_C1=400.0)
    with pytest.raises(CableInputError, match="exceeds floating-point range"):
        corrected_damage(_cycles(80.0, 2), absurd)


# --------------------------------------------------------------------------- #
# S-N curve definitions                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"R_ratio": 1.0}, "R_ratio must be below 1"),
        ({"thickness_exponent": -0.1}, "thickness_exponent must be nonnegative"),
        (
            {"thickness_exponent": 0.2, "reference_thickness_m": 0.0},
            "reference_thickness_m must be positive",
        ),
        ({"thickness_exponent": 0.2}, "needs reference_thickness_m"),
    ],
)
def test_sn_curve_definition_rules(kwargs, message):
    with pytest.raises(CableInputError, match=message):
        _single(**kwargs)


def test_knee_cycles_and_thickness_must_be_positive():
    with pytest.raises(CableInputError, match="knee_cycles must be positive"):
        SNcurve.from_knee(
            "k", stress_unit="MPa", m1=3.0, log_C1=12.0, m2=5.0, knee_cycles=0.0
        )
    curve = _single(thickness_exponent=0.25, reference_thickness_m=0.016)
    # DNV-RP-C203 Eq. 2.10: no correction at or below the reference thickness.
    assert curve.thickness_factor(0.010) == 1.0
    assert curve.thickness_factor(0.032) == pytest.approx(2.0**0.25)
    with pytest.raises(CableInputError, match="thickness_m must be positive"):
        curve.thickness_factor(0.0)


def test_register_requires_an_sn_curve_instance():
    with pytest.raises(CableInputError, match="register an SNcurve instance"):
        register_sn_curve({"name": "dict"})  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Wear coefficients and fretting knock-down                                    #
# --------------------------------------------------------------------------- #
def _wear(alpha: float, low: float, high: float) -> WearCoefficient:
    return WearCoefficient("x_y", alpha, low, high, "assumed", True, "x")


def test_wear_coefficient_band_rules():
    with pytest.raises(CableInputError, match="finite and >= 0"):
        _wear(-1.0, 0.0, 1.0)
    with pytest.raises(CableInputError, match="finite and >= 0"):
        _wear(math.inf, 0.0, math.inf)
    with pytest.raises(CableInputError, match="must lie in its band"):
        _wear(2.0, 0.0, 1.0)
    # A scaled copy widens the band to contain the new nominal value.
    scaled = _wear(1.0, 0.5, 2.0).scaled(4.0)
    assert scaled.low_m3_per_J == 0.5 and scaled.high_m3_per_J == 4.0
    assert scaled.assumed


def test_wear_lookups_name_unknown_keys():
    assert wear_pair_for("hdpe", "galvanised_steel_armour") == "steel_polyethylene"
    assert wear_pair_for("copper_stranded", "copper_wire_screen") == "copper_copper"
    with pytest.raises(CableLookupError, match="no wear family"):
        wear_pair_for("polyester_tape", "galvanised_steel_armour")
    with pytest.raises(CableLookupError, match="unknown wear pair"):
        get_wear_coefficient("glass_glass")
    with pytest.raises(CableLookupError, match="unknown knock-down"):
        get_knockdown("titanium")


def _knockdown(**kwargs) -> FrettingKnockdown:
    base = {
        "name": "k",
        "partial_slip_factor": 1.0,
        "gross_slip_factor": 1.0,
        "nishioka_hirakawa": True,
        "pressure_cap_Pa": 1.0e8,
        "slip_constant_m": 1.0e-6,
        "reference_cycles": 1.0e7,
        "source": "assumed",
        "assumed": True,
    }
    base.update(kwargs)
    return FrettingKnockdown(**base)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"partial_slip_factor": 0.0}, "finite and positive"),
        ({"pressure_cap_Pa": -1.0}, "pressure cap must be nonnegative"),
        ({"pressure_cap_Pa": math.nan}, "pressure cap must be nonnegative"),
        ({"slip_constant_m": 0.0}, "slip constant and reference cycles"),
        ({"reference_cycles": 1.0}, "slip constant and reference cycles"),
    ],
)
def test_knockdown_definition_rules(kwargs, message):
    with pytest.raises(CableInputError, match=message):
        _knockdown(**kwargs)


def test_nishioka_hirakawa_factor_limits():
    law = _knockdown()
    delta = np.array([0.0, 1.0e-6, 1.0e-3])
    factor = law.notch_factor(0.5, 2.0e8, delta, reference_range_Pa=2.0e8)
    # No slip: no knock-down; saturated slip: 1 + 2 mu p_cap / sigma_a.
    assert factor[0] == pytest.approx(1.0)
    assert factor[-1] == pytest.approx(1.0 + 2 * 0.5 * 1.0e8 / 1.0e8)
    assert np.all(np.diff(factor) > 0.0)
    with pytest.raises(CableInputError, match="reference stress range"):
        law.notch_factor(0.5, 2.0e8, delta, reference_range_Pa=0.0)


# --------------------------------------------------------------------------- #
# User materials and packaged templates                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("name", "properties", "message"),
    [
        ("  ", {"E_Pa": (1.0, "x")}, "non-empty string"),
        ("user_empty", {}, "at least one property"),
        ("user_key", {" ": (1.0, "x")}, "property names must be non-empty"),
        ("user_pair", {"E_Pa": 1.0}, "must be a \\(value, citation\\) pair"),
    ],
)
def test_register_material_rules(name, properties, message):
    with pytest.raises(CableInputError, match=message):
        cx.register_material(name, properties)


def test_unregister_material_rules():
    with pytest.raises(CableInputError, match="built-in material"):
        cx.unregister_material("hdpe")
    with pytest.raises(CableLookupError, match="no user material"):
        cx.unregister_material("never_registered_material")
    cx.register_material(
        "user_soft_tape", {"E_Pa": (1.0e9, "[ModelAssumption]")}, polymer=True
    )
    try:
        assert library.get_property("user_soft_tape", "E_Pa") == 1.0e9
        assert "user_soft_tape" in library.POLYMER_MATERIALS
    finally:
        cx.unregister_material("user_soft_tape")
    with pytest.raises(CableLookupError, match="no user material"):
        cx.unregister_material("user_soft_tape")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lay_pattern": "random"}, "lay_pattern must be"),
        ({"outer_lay_ratio": 0.0}, "outer_lay_ratio must be finite and positive"),
        ({"outer_lay_ratio": math.inf}, "outer_lay_ratio must be finite"),
        ({"compaction": "medium"}, "compaction must be"),
    ],
)
def test_stranded_template_options_are_checked(kwargs, message):
    with pytest.raises(ValueError, match=message):
        cx.iea_task49_66kv_stranded(**kwargs)


def test_template_selectors_reject_unknown_wire_shapes():
    assert cx.witz1996_yaml_path("rectangular").name == "witz1996_rect.yaml"
    assert cx.saevik2011_yaml_path("rectangular").name == "saevik2011_rect.yaml"
    with pytest.raises(ValueError, match="wires must be"):
        cx.witz1996_yaml_path("oval")
    with pytest.raises(ValueError, match="tendons must be"):
        cx.saevik2011_yaml_path("oval")
