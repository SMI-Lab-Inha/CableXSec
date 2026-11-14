# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Independent tensor projections, analytic extrema and engineering contracts."""

import math
from dataclasses import replace

import numpy as np
import pytest

from cablexsec import SWTCycleResult, compute_swt_cycle
from cablexsec.materials.sn_curves import SWTLifeConstants

E = 200e9
NU = 0.3
MATERIAL = SWTLifeConstants(800e6, -0.5, 0, -0.6, E)


def hooke(stress):
    return (
        (1 + NU) * stress
        - NU * np.trace(stress, axis1=1, axis2=2)[:, None, None] * np.eye(3)
    ) / E


def uniaxial(mean=30e6, amplitude=100e6):
    stress = np.zeros((5, 3, 3))
    stress[:, 0, 0] = mean + amplitude * np.array([0, 1, 0, -1, 0])
    return stress, hooke(stress)


def evaluate(stress, strain, normals=(1, 0, 0), **kwargs):
    return compute_swt_cycle(
        stress,
        strain,
        normals,
        MATERIAL,
        material_point_id="wire-1/contact-A",
        **kwargs,
    )


def test_uniaxial_elastic_swt_and_life_closed_form():
    result = evaluate(*uniaxial(), normals=np.eye(3))
    assert isinstance(result, SWTCycleResult)
    expected_stress = np.array([130e6, 0, 0])
    expected_amplitude = np.array([100e6, NU * 100e6, NU * 100e6]) / E
    np.testing.assert_allclose(result.maximum_normal_stress_Pa, expected_stress)
    np.testing.assert_allclose(result.normal_strain_amplitude, expected_amplitude)
    np.testing.assert_allclose(result.swt_Pa, expected_stress * expected_amplitude)
    # b=-1/2 and no plastic term: SWT = sf'^2 / (E * 2N).
    assert result.life_cycles == pytest.approx(800e6**2 / (2 * E * 130e6 * 100e6 / E))
    assert result.critical_plane_index == 0
    np.testing.assert_array_equal(result.critical_plane_normal, [1, 0, 0])
    assert result.n_samples == 5
    assert result.material_point_id == "wire-1/contact-A"
    assert result.material == MATERIAL


def test_pure_shear_uses_tensor_not_engineering_strain():
    stress = np.zeros((5, 3, 3))
    tau = 80e6
    stress[:, 0, 1] = stress[:, 1, 0] = tau * np.array([0, 1, 0, -1, 0])
    result = evaluate(stress, hooke(stress), [1, 1, 0])
    shear_modulus = E / (2 * (1 + NU))
    assert result.maximum_normal_stress_Pa[0] == pytest.approx(tau)
    assert result.normal_strain_amplitude[0] == pytest.approx(tau / (2 * shear_modulus))
    assert result.sampled_max_swt_Pa == pytest.approx(tau**2 / (2 * shear_modulus))


def harmonic(intervals):
    phase = np.linspace(0, 2 * np.pi, intervals + 1)
    mean = 1e6 * np.array([[35, 8, -6], [8, -15, 5], [-6, 5, 12]])
    sine = 1e6 * np.array([[100, 45, -30], [45, -20, 10], [-30, 10, 40]])
    cosine = 1e6 * np.array([[-20, 10, 25], [10, 60, 15], [25, 15, -30]])
    stress = (
        mean
        + np.sin(phase)[:, None, None] * sine
        + np.cos(phase)[:, None, None] * cosine
    )
    # Make closure exact independently of trigonometric roundoff.
    stress[-1] = stress[0]
    return stress, hooke(stress), mean, sine, cosine


@pytest.mark.parametrize("intervals", [32, 128, 512])
def test_nonproportional_extrema_obey_analytic_time_discretization_bound(intervals):
    stress, strain, mean, sine, cosine = harmonic(intervals)
    normals = np.array([[1, 2, 3], [-3, 1, 2], [2, -4, 1]], dtype=float)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    result = evaluate(stress, strain, normals)
    for i, normal in enumerate(normals):
        sigma_amplitude = np.hypot(normal @ sine @ normal, normal @ cosine @ normal)
        sigma_exact = normal @ mean @ normal + sigma_amplitude
        strain_modes = hooke(np.stack([sine, cosine]))
        eps_exact = np.hypot(*(normal @ mode @ normal for mode in strain_modes))
        # Every continuous peak has a sample within pi/intervals.
        loss = 1 - np.cos(np.pi / intervals)
        assert (
            -1e-7
            <= sigma_exact - result.maximum_normal_stress_Pa[i]
            <= sigma_amplitude * loss + 1e-7
        )
        assert (
            -1e-18
            <= eps_exact - result.normal_strain_amplitude[i]
            <= eps_exact * loss + 1e-18
        )
        assert result.swt_Pa[i] <= max(sigma_exact, 0) * eps_exact * (1 + 1e-14)


def test_common_rotation_preserves_plane_indicators_and_life():
    stress, strain, *_ = harmonic(64)
    normals = np.array([[1, 0, 0], [1, 2, 3], [-2, 1, 0.5]])
    rotation, _ = np.linalg.qr(
        np.array([[1, 2, 3], [-3, 2, 1], [4, -1, 2]], dtype=float)
    )
    baseline = evaluate(stress, strain, normals)
    rotated = evaluate(
        rotation @ stress @ rotation.T,
        rotation @ strain @ rotation.T,
        normals @ rotation.T,
    )
    np.testing.assert_allclose(rotated.swt_Pa, baseline.swt_Pa, rtol=2e-14)
    np.testing.assert_allclose(
        rotated.maximum_normal_stress_Pa, baseline.maximum_normal_stress_Pa, rtol=2e-14
    )
    assert rotated.life_cycles == pytest.approx(baseline.life_cycles, rel=2e-14)
    np.testing.assert_allclose(
        rotated.critical_plane_normal,
        rotation @ baseline.critical_plane_normal,
        atol=2e-15,
    )


@pytest.mark.parametrize(
    "plane_batch,time_batch", [(1, 1), (2, 7), (3, 19), (100, 1000)]
)
def test_chunking_preserves_indicators(plane_batch, time_batch):
    stress, strain, *_ = harmonic(64)
    normals = np.array([[1, 0, 0], [1, 2, 3], [-2, 1, 0.5], [0, 1, 0]])
    base = evaluate(stress, strain, normals)
    chunked = evaluate(
        stress,
        strain,
        normals,
        plane_batch_size=plane_batch,
        time_batch_size=time_batch,
    )
    np.testing.assert_allclose(chunked.swt_Pa, base.swt_Pa, rtol=2e-14)
    assert chunked.life_cycles == pytest.approx(base.life_cycles, rel=2e-14)


def test_normal_sign_scaling_duplicates_and_tie_order():
    result = evaluate(*uniaxial(), [[1e-300, 0, 0], [-1e300, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(result.swt_Pa, np.repeat(result.swt_Pa[0], 3))
    assert result.critical_plane_index == 0


@pytest.mark.parametrize("kind", ["zero", "compression", "constant"])
def test_no_tensile_cyclic_driver_has_no_critical_plane(kind):
    if kind == "zero":
        stress, strain = np.zeros((3, 3, 3)), np.zeros((3, 3, 3))
    elif kind == "compression":
        stress, strain = uniaxial(mean=-150e6)
    else:
        stress, strain = uniaxial(amplitude=0)
    result = evaluate(stress, strain)
    assert result.critical_plane_index is None
    assert result.critical_plane_normal is None
    assert result.sampled_max_swt_Pa == 0
    assert math.isinf(result.life_cycles)


def test_strain_mean_does_not_change_swt_but_tensile_stress_mean_does():
    stress, strain = uniaxial()
    baseline = evaluate(stress, strain)
    shifted_strain = evaluate(stress, strain + np.diag([0.01, -0.02, 0.03]))
    shifted_stress = evaluate(stress + np.diag([20e6, 0, 0]), strain)
    np.testing.assert_allclose(shifted_strain.swt_Pa, baseline.swt_Pa, rtol=1e-14)
    assert (
        shifted_stress.sampled_max_swt_Pa / baseline.sampled_max_swt_Pa
        == pytest.approx(150 / 130)
    )


def test_plane_refinement_can_reveal_missed_maximum():
    stress, strain = uniaxial(mean=0)
    coarse = evaluate(stress, strain, [[1, 1, 0], [1, -1, 0]])
    refined = evaluate(stress, strain, [[1, 1, 0], [1, -1, 0], [1, 0, 0]])
    assert refined.sampled_max_swt_Pa > coarse.sampled_max_swt_Pa
    assert refined.critical_plane_index == 2
    assert refined.sampled_max_swt_Pa == pytest.approx(100e6**2 / E)


def test_inputs_unchanged_results_detached_and_readonly():
    stress, strain = uniaxial()
    normals = np.eye(3)
    before = [array.copy() for array in (stress, strain, normals)]
    result = evaluate(stress, strain, normals)
    for value, original in zip((stress, strain, normals), before, strict=True):
        np.testing.assert_array_equal(value, original)
        value[:] = 0
    assert result.sampled_max_swt_Pa > 0
    np.testing.assert_array_equal(result.plane_normals, np.eye(3))
    for array in (
        result.plane_normals,
        result.maximum_normal_stress_Pa,
        result.normal_strain_amplitude,
        result.swt_Pa,
        result.critical_plane_normal,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 0


@pytest.mark.parametrize("field", ["stress", "strain"])
@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros((2, 3, 3)),
        np.zeros((3, 2, 3)),
        np.zeros((3, 3)),
        np.full((3, 3, 3), np.nan),
        np.full((3, 3, 3), np.inf),
        np.ones((3, 3, 3), dtype=complex),
    ],
)
def test_invalid_tensor_histories_rejected(field, invalid):
    stress, strain = uniaxial()
    with pytest.raises(ValueError, match=field):
        evaluate(
            invalid if field == "stress" else stress,
            invalid if field == "strain" else strain,
        )


def test_mismatched_sample_counts_rejected():
    stress, strain = uniaxial()
    with pytest.raises(ValueError, match="share sample"):
        evaluate(stress, strain[:3])


@pytest.mark.parametrize("field", ["stress", "strain"])
def test_nonsymmetric_tensor_rejected(field):
    stress, strain = uniaxial()
    (stress if field == "stress" else strain)[2, 0, 1] = 1
    with pytest.raises(ValueError, match="symmetric"):
        evaluate(stress, strain)


@pytest.mark.parametrize("field", ["stress", "strain"])
def test_open_cycle_rejected(field):
    stress, strain = uniaxial()
    (stress if field == "stress" else strain)[-1] *= 1.1
    with pytest.raises(ValueError, match=f"{field} must close"):
        evaluate(stress, strain)


def test_closure_tolerance_and_reported_frobenius_error():
    stress, strain = uniaxial()
    stress[-1, 0, 0] += 0.1
    result = evaluate(stress, strain, closure_rtol=1e-8)
    assert result.stress_closure_relative_error == pytest.approx(0.1 / 130e6, rel=1e-6)
    assert result.strain_closure_relative_error == 0
    with pytest.raises(ValueError, match="close"):
        evaluate(stress, strain, closure_rtol=0)


@pytest.mark.parametrize(
    "normals",
    [
        [],
        [1, 2],
        [[0, 0, 0]],
        [1, np.nan, 0],
        [1, np.inf, 0],
        [1j, 0, 0],
        np.ones((1, 1, 3)),
    ],
)
def test_invalid_normals_rejected(normals):
    with pytest.raises(ValueError, match="normals"):
        evaluate(*uniaxial(), normals)


@pytest.mark.parametrize("value", ["", " ", None, 1])
def test_material_point_identity_required(value):
    with pytest.raises(ValueError, match="material_point_id"):
        compute_swt_cycle(*uniaxial(), [1, 0, 0], MATERIAL, material_point_id=value)


@pytest.mark.parametrize("value", [-1, 1, np.nan, np.inf])
def test_invalid_closure_tolerance_rejected(value):
    with pytest.raises(ValueError, match="closure_rtol"):
        evaluate(*uniaxial(), closure_rtol=value)


@pytest.mark.parametrize("key", ["plane_batch_size", "time_batch_size"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_batch_sizes_rejected(key, value):
    with pytest.raises(ValueError, match="positive integers"):
        evaluate(*uniaxial(), **{key: value})


def test_material_validation_not_bypassed_for_zero_driver():
    zero = np.zeros((3, 3, 3))
    with pytest.raises(ValueError, match="modulus"):
        compute_swt_cycle(
            zero, zero, [1, 0, 0], replace(MATERIAL, E_Pa=0), material_point_id="point"
        )


@pytest.mark.parametrize(
    "stress_scale,strain_scale", [(1e300, 1e100), (1e-200, 1e-200)]
)
def test_unrepresentable_driver_raises(stress_scale, strain_scale):
    shape = np.zeros((3, 3, 3))
    shape[1, 0, 0] = 1
    with pytest.raises(ValueError, match="floating-point range"):
        evaluate(shape * stress_scale, shape * strain_scale)


def test_provenance_limits_are_attached_to_result():
    result = evaluate(*uniaxial())
    wording = " ".join(result.provenance.limitations)
    assert "supplied normals" in wording
    assert "One repeated cycle" in wording
    assert "calibrated service-life" in wording
