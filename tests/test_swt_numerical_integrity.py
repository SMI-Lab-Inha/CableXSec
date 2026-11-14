# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Independent strain-life closure, solver failures and contact-driver domains."""

import math
from dataclasses import replace

import numpy as np
import pytest

from cablexsec.analysis.fretting_swt import swt_driver_from_contact, swt_life
from cablexsec.materials.sn_curves import SWTLifeConstants, get_swt_life


def material():
    return get_swt_life("copper_poon2022")


def forward(life, const):
    return (const.sigma_f_prime_Pa**2 / const.E_Pa) * (2 * life) ** (
        2 * const.b
    ) + const.sigma_f_prime_Pa * const.eps_f_prime * (2 * life) ** (const.b + const.c)


@pytest.mark.parametrize(
    "life", [1e-100, 0.001, 0.1, 0.5, 1, 100, 1e8, 1e30, 1e100, 1e250]
)
def test_recovers_life_from_independent_forward_equation(life):
    const = material()
    target = forward(life, const)
    result = swt_life(target, const)
    assert result == pytest.approx(life, rel=1e-10, abs=0)
    assert forward(result, const) == pytest.approx(target, rel=1e-12, abs=0)


@pytest.mark.parametrize("life", [1e-20, 1, 1e8, 1e120])
def test_elastic_only_closed_form_across_life_scales(life):
    const = replace(material(), eps_f_prime=0)
    assert swt_life(forward(life, const), const) == pytest.approx(
        life, rel=1e-11, abs=0
    )


def test_equal_exponents_have_independent_closed_form():
    const = SWTLifeConstants(3, -0.25, 2, -0.25, 1)
    # Both terms have exponent -1/2 and coefficients 9 and 6.
    assert swt_life(0.3, const) == pytest.approx(0.5 * ((9 + 6) / 0.3) ** 2, rel=1e-12)


@pytest.mark.parametrize("initial", [1e-200, 0.01, 1, 12000, 12345, 12500, 1e6, 1e200])
def test_answer_is_independent_of_initial_guess(initial):
    const = material()
    assert swt_life(forward(12345, const), const, n_init=initial) == pytest.approx(
        12345, rel=1e-11
    )


def test_iteration_exhaustion_is_not_returned_as_a_life():
    with pytest.raises(RuntimeError, match="converg"):
        swt_life(1e5, material(), max_iter=1)


@pytest.mark.parametrize("target", [math.nan, math.inf, -math.inf])
def test_nonfinite_indicators_are_rejected(target):
    with pytest.raises(ValueError):
        swt_life(target, material())


@pytest.mark.parametrize("target", [0, -1])
def test_nonpositive_finite_indicators_have_no_tensile_driver(target):
    assert swt_life(target, material()) == math.inf


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_init": 0},
        {"n_init": -1},
        {"n_init": math.nan},
        {"n_init": math.inf},
        {"tol": 0},
        {"tol": -1},
        {"tol": 1},
        {"tol": math.nan},
        {"tol": math.inf},
        {"max_iter": 0},
        {"max_iter": -1},
        {"max_iter": 1.5},
        {"max_iter": True},
    ],
)
def test_invalid_solver_controls_are_rejected(kwargs):
    with pytest.raises(ValueError):
        swt_life(1e5, material(), **kwargs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sigma_f_prime_Pa", 0),
        ("sigma_f_prime_Pa", -1),
        ("sigma_f_prime_Pa", math.nan),
        ("E_Pa", 0),
        ("E_Pa", math.inf),
        ("eps_f_prime", -1),
        ("eps_f_prime", math.nan),
        ("b", 0),
        ("b", 0.1),
        ("c", 0),
        ("c", 0.1),
    ],
)
def test_invalid_material_constants_are_rejected(field, value):
    const = replace(material(), **{field: value})
    with pytest.raises(ValueError):
        swt_life(1e5, const)


def test_large_strength_coefficient_need_not_overflow_its_square():
    const = SWTLifeConstants(1e200, -0.5, 0, -0.5, 1e200)
    assert swt_life(1e198, const) == pytest.approx(50, rel=1e-12)


def test_unrepresentable_positive_life_is_not_infinite_runout():
    with pytest.raises(ValueError, match="range"):
        swt_life(1e-300, material())
    const = SWTLifeConstants(1e-200, -0.1, 0, -0.5, 1e200)
    with pytest.raises(ValueError, match="range"):
        swt_life(1e200, const)


@pytest.mark.parametrize(
    "values",
    [
        (-1, 0.3, 1e9, 0.3),
        (math.nan, 0.3, 1e9, 0.3),
        (math.inf, 0.3, 1e9, 0.3),
        (1, -1, 1e9, 0.3),
        (1, math.nan, 1e9, 0.3),
        (1, 0.3, 0, 0.3),
        (1, 0.3, -1, 0.3),
        (1, 0.3, math.inf, 0.3),
        (1, 0.3, 1e9, -1),
        (1, 0.3, 1e9, 0.51),
        (1, 0.3, 1e9, math.nan),
    ],
)
def test_invalid_contact_driver_inputs_are_rejected(values):
    with pytest.raises(ValueError):
        swt_driver_from_contact(*values)


@pytest.mark.parametrize("values", [(0, 0.3, 1e9, 0.3), (1e6, 0, 1e9, 0.5)])
def test_zero_contact_or_friction_gives_zero_driver(values):
    assert swt_driver_from_contact(*values) == (0, 0, 0)


def test_contact_driver_preserves_representable_results_through_large_factors():
    np.testing.assert_allclose(
        swt_driver_from_contact(1e-308, 1e308, 1, 0.3), [2, 1.82, 3.64], rtol=2e-12
    )


def test_unrepresentable_positive_contact_driver_is_explicit():
    for values in [(1e308, 1, 1, 0.3), (1e-300, 1e-300, 1, 0.3)]:
        with pytest.raises(ValueError, match="range"):
            swt_driver_from_contact(*values)
