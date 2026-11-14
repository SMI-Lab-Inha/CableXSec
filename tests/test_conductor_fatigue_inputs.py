# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Stranded-conductor fatigue: input rules, the legacy contact form, strain life."""

from __future__ import annotations

import math

import check_conductor_strand_slip as check
import numpy as np
import pytest

from cablexsec.analysis.conductor_fatigue import (
    CriticalDistanceSpec,
    _trellis_contact,
    assess_conductor_fatigue,
    trellis_contact,
    trellis_force_ratio,
)
from cablexsec.exceptions import CableInputError
from cablexsec.materials.sn_curves import SNcurve, get_sn_curve, get_swt_life
from cablexsec.materials.strain_life import get_strain_life

SN = get_sn_curve("nasution_2014_copper")


def _assess(kappa: np.ndarray, **kwargs):
    return assess_conductor_fatigue(
        check.nasution_rings(),
        check.NAS_E,
        0.3,
        kappa,
        np.zeros_like(kappa),
        np.full_like(kappa, check.NAS_T),
        mu_wire=0.2,
        reference_tension_N=check.NAS_T,
        **{"sn_bulk": SN, "swt_const": None, **kwargs},
    )


def test_legacy_contact_form_is_the_elastic_hertz_contact():
    rings = check.nasution_rings()
    legacy = _trellis_contact(rings[1], rings[2], 9.3e4, check.NAS_E, 0.3)
    elastic = trellis_contact(
        rings[1], rings[2], 9.3e4, check.NAS_E, 0.3, yield_stress_Pa=None
    )
    assert legacy is not None and elastic is not None
    assert legacy == (elastic.force_N, elastic.p0_Pa)
    # The centre wire has no trellis contact in either form.
    assert _trellis_contact(rings[0], rings[1], 9.3e4, check.NAS_E, 0.3) is None
    with pytest.raises(CableInputError, match="pressure factor must be finite"):
        trellis_contact(rings[1], rings[2], 9.3e4, check.NAS_E, 0.3, pressure_factor=0)


def test_force_ratio_rejects_unequal_or_non_finite_histories():
    theta = np.radians([0.0, 90.0])
    kappa = np.linspace(0.0, 0.2, 5)
    with pytest.raises(CableInputError, match="equal 1-D histories"):
        trellis_force_ratio(kappa, kappa[:-1], theta, 0.1)
    with pytest.raises(CableInputError, match="must be finite"):
        trellis_force_ratio(kappa, np.full(5, math.nan), theta, 0.1)


def test_critical_distance_needs_points_and_planes():
    with pytest.raises(CableInputError, match="at least one point and one plane"):
        CriticalDistanceSpec(n_points=0)
    with pytest.raises(CableInputError, match="at least one point and one plane"):
        CriticalDistanceSpec(n_planes=0)


def test_assessment_rejects_conflicting_rules_and_short_histories():
    kappa = np.sin(np.linspace(0, 2 * math.pi, 41)) / 3.0
    with pytest.raises(CableInputError, match="surface SWT or the critical-distance"):
        _assess(
            kappa,
            swt_const=get_swt_life("copper_poon2022"),
            critical_distance=CriticalDistanceSpec(),
        )
    with pytest.raises(CableInputError, match="equal 1-D arrays"):
        _assess(np.zeros(1))
    with pytest.raises(CableInputError, match="requires an S-N curve in MPa"):
        _assess(kappa, sn_bulk=SNcurve(name="s", stress_unit="um", m1=2, log_C1=9))
    with pytest.raises(CableInputError, match="tension must match"):
        assess_conductor_fatigue(
            check.nasution_rings(),
            check.NAS_E,
            0.3,
            kappa,
            np.zeros_like(kappa),
            np.ones(3),
            sn_bulk=SN,
            swt_const=None,
            mu_wire=0.2,
        )


def test_strain_life_damage_grows_with_the_bending_amplitude():
    t = np.linspace(0, 6 * math.pi, 601)
    life = get_strain_life("copper_poon2022")
    small = _assess(np.sin(t) / 6.5, strain_life=life)
    large = _assess(np.sin(t) / 3.0, strain_life=life)
    assert small.D_strain_life_per_ring is not None
    assert large.D_strain_life_per_ring is not None
    assert small.D_strain_life_per_ring.shape == (4,)
    assert np.all(small.D_strain_life_per_ring >= 0.0)
    # Same history at a larger curvature amplitude: never less damage.
    assert np.all(large.D_strain_life_per_ring >= small.D_strain_life_per_ring)
    assert large.D_strain_life_per_ring.max() > small.D_strain_life_per_ring.max()
    # Without the option the result carries no strain-life damage.
    assert _assess(np.sin(t) / 3.0).D_strain_life_per_ring is None
