# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Invariants of the public fatigue and moment-curvature calls.

Tolerances were set before the first run: 5% for rotation and mirror of the
bending plane (wire azimuth discretisation), 10% for a repeated history
(start-up transient), machine precision otherwise.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import cablexsec as cx
from cablexsec.cable_adapters import cable_to_mkappa_specs
from cablexsec.mechanics.papailiou import wire_slip_stress_bounds

pytestmark = pytest.mark.analytical_sanity

_CABLES = ("menard2023", "inter_array_single_core_33kv", "inter_array_three_core_33kv")
_T = np.linspace(0.0, 20.0 * math.pi, 801)
_KX = 0.06 * np.sin(_T) + 0.02 * np.sin(2.3 * _T)
_KY = 0.03 * np.cos(0.7 * _T)
_TENSION = 6.0e4 + 1.5e4 * np.sin(0.9 * _T)


@pytest.fixture(scope="module", params=_CABLES)
def cable(request):
    path = Path(cx.__file__).parent / "cables" / f"{request.param}.yaml"
    return cx.load_cable(path, geometry_override=None)


def _damage(cable, kx, ky, tension=_TENSION):
    result = cx.compute_fatigue(cable, kx, ky, tension_N=tension, n_fibres=16)
    return {k: float(np.sum(v)) for k, v in result.D_bulk_per_wire.items()}, result


def _worst_relative(a, b):
    return max(abs(a[k] - b[k]) / max(abs(a[k]), abs(b[k]), 1e-300) for k in a)


def test_deterministic_non_negative_and_zero_for_zero_curvature(cable):
    base, first = _damage(cable, _KX, _KY)
    _, second = _damage(cable, _KX, _KY)
    for name, damage in first.D_bulk_per_wire.items():
        np.testing.assert_array_equal(damage, second.D_bulk_per_wire[name])
        assert np.all(damage >= 0.0)
    assert all(v > 0.0 for v in base.values())
    zero, _ = _damage(cable, 0 * _KX, 0 * _KY, np.full(_T.size, 6.0e4))
    assert all(v == 0.0 for v in zero.values())


def test_damage_rises_with_curvature_amplitude(cable):
    small, _ = _damage(cable, 0.5 * _KX, 0.5 * _KY)
    base, _ = _damage(cable, _KX, _KY)
    large, _ = _damage(cable, 2.0 * _KX, 2.0 * _KY)
    for name in base:
        assert small[name] <= base[name] <= large[name]


@pytest.mark.parametrize("angle_deg", [37.0, 90.0])
def test_rotation_of_the_bending_plane(cable, angle_deg):
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    base, _ = _damage(cable, _KX, _KY)
    rotated, _ = _damage(cable, c * _KX - s * _KY, s * _KX + c * _KY)
    assert _worst_relative(base, rotated) <= 0.05


@pytest.mark.parametrize("signs", [(-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)])
def test_mirror_of_the_bending_plane(cable, signs):
    base, _ = _damage(cable, _KX, _KY)
    mirrored, _ = _damage(cable, signs[0] * _KX, signs[1] * _KY)
    assert _worst_relative(base, mirrored) <= 0.05


def test_repeated_history_doubles_damage(cable):
    base, _ = _damage(cable, _KX, _KY)
    twice, _ = _damage(
        cable, np.r_[_KX, _KX[1:]], np.r_[_KY, _KY[1:]], np.r_[_TENSION, _TENSION[1:]]
    )
    for name in base:
        assert 1.9 <= twice[name] / base[name] <= 2.1


def _mirror_defect_terms(cable):
    """Full-slip value and envelope of ``M(kappa) + M(-kappa)``.

    The friction bounds belong to the wire (upper ``u`` for a rising, lower
    ``l`` for a falling stress), so the mirrored history is the original one
    with the bounds swapped: ``M(kappa) + M(-kappa) = sum_w W r c_w D_w``,
    ``c_w = cos(phi_w)``, ``W = A cos a``, with ``D_w`` the stress difference
    of the two bound orders under the same drive. A wire and its mirror image
    (``c -> -c``) have the same ``D``, so an even wire count cancels exactly;
    in full slip ``D_w = u_w - l_w = 2 sigma0 (cosh(lam psi) - 1)``, and the
    stop operator is monotone in its bounds, ``0 <= D_w <= u_w - l_w``.
    """
    full = positive = negative = 0.0
    for spec in cable_to_mkappa_specs(cable, 5.0e4):
        upper, lower = wire_slip_stress_bounds(spec, 0.0)
        c = np.cos(spec.phi_wires)
        term = spec.A_wire_m2 * math.cos(spec.lay_angle_rad) * spec.r_helix_m
        span = term * c * (upper - lower)
        full += float(np.sum(span))
        positive += float(np.sum(span[c > 0]))
        negative -= float(np.sum(span[c < 0]))
    return full, max(positive, negative)


def test_moment_curvature_symmetry_and_work_balance(cable):
    kappa = 0.08 * np.sin(np.linspace(0.0, 4.0 * math.pi, 161))
    loop = cx.compute_mkappa(cable, kappa, tension_N=5.0e4)
    mirror = cx.compute_mkappa(cable, -kappa, tension_N=5.0e4)
    scale = float(np.max(np.abs(loop.moment_Nm)))
    defect = np.asarray(loop.moment_Nm) + np.asarray(mirror.moment_Nm)
    full, envelope = _mirror_defect_terms(cable)
    if all(spec.n_wires % 2 == 0 for spec in cable_to_mkappa_specs(cable, 5.0e4)):
        assert np.max(np.abs(defect)) <= 1e-9 * scale
    else:
        # Odd wire counts: M(-kappa) = -M(kappa) up to the signed-capstan
        # asymmetry (8e-5 of the peak for menard2023 in full slip), even in
        # kappa: swapping the two histories leaves the sum unchanged.
        slipped = (np.asarray(loop.slip_fraction) == 1.0) & (
            np.asarray(mirror.slip_fraction) == 1.0
        )
        assert slipped.any() and abs(full) > 1e-6 * scale
        np.testing.assert_allclose(defect[slipped], full, rtol=0, atol=1e-9 * scale)
        assert np.max(np.abs(defect)) <= envelope + 1e-9 * scale
    assert loop.work is not None
    friction = np.asarray(loop.work.friction_J_per_m)
    assert np.all(np.diff(friction) >= -1e-12 * max(1.0, friction[-1]))
    assert np.all(np.asarray(loop.work.stored_J_per_m) >= 0.0)
    work = float(np.max(np.abs(loop.work.endpoint_work_J_per_m)))
    assert np.max(np.abs(loop.work.balance_residual_J_per_m)) <= 1e-6 * work
    assert loop.W_diss_J_per_m == pytest.approx(friction[-1], rel=1e-9)
    assert loop.EI_min_Nm2 <= loop.EI_apparent_Nm2 * (1 + 1e-9)
    assert loop.EI_apparent_Nm2 <= loop.EI_max_Nm2 * (1 + 1e-9)
    slip = np.asarray(loop.slip_fraction)
    assert np.all((slip >= 0.0) & (slip <= 1.0))
