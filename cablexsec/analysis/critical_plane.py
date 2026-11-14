# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Sampled SWT critical planes for one repeated cycle at one material point.

Normal stress and strain follow n.T @ tensor @ n. The SWT indicator uses the
maximum tensile normal stress times the normal strain amplitude on that SAME
plane and material point (Poon et al. 2022, section 2.4, eqs 28-30 and 37-38).
https://doi.org/10.1111/ffe.13683
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cablexsec.analysis.fretting_swt import swt_life
from cablexsec.exceptions import CableInputError
from cablexsec.materials.sn_curves import SWTLifeConstants
from cablexsec.provenance import Provenance

FloatArray = NDArray[np.float64]

__all__ = ["SWTCycleResult", "compute_swt_cycle"]


def _readonly(value: FloatArray) -> FloatArray:
    result = value.copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class SWTCycleResult:
    """Plane-resolved indicators and formal life for one closed repeated cycle.

    The critical plane is the largest indicator among the SUPPLIED normals.
    No spatial maximum or unsampled angular maximum is inferred. A zero driver
    has no critical plane (index None) and returns infinite no-tensile-driver
    life, not proof of infinite physical life under other mechanisms.
    """

    material_point_id: str
    plane_normals: FloatArray
    maximum_normal_stress_Pa: FloatArray
    normal_strain_amplitude: FloatArray
    swt_Pa: FloatArray
    critical_plane_index: int | None
    life_cycles: float
    material: SWTLifeConstants
    n_samples: int
    stress_closure_relative_error: float
    strain_closure_relative_error: float
    provenance: Provenance

    @property
    def sampled_max_swt_Pa(self) -> float:
        return float(np.max(self.swt_Pa))

    @property
    def critical_plane_normal(self) -> FloatArray | None:
        if self.critical_plane_index is None:
            return None
        return _readonly(self.plane_normals[self.critical_plane_index])


def _history(values: ArrayLike, name: str) -> FloatArray:
    if np.iscomplexobj(values):
        raise CableInputError(f"{name} must be real")
    result = np.asarray(values, dtype=np.float64)
    if (
        result.ndim != 3
        or result.shape[1:] != (3, 3)
        or result.shape[0] < 3
        or not np.isfinite(result).all()
    ):
        raise CableInputError(
            f"{name} must be finite with shape (at least 3 samples, 3, 3)"
        )
    return result


def _history_scale_and_closure(
    history: FloatArray, name: str, closure_rtol: float, batch: int
) -> tuple[float, float]:
    scale = float(np.max(np.abs(history)))
    if scale == 0:
        return 0.0, 0.0
    peak_norm = 0.0
    for start in range(0, len(history), batch):
        part = history[start : start + batch] / scale
        if np.any(np.abs(part - np.swapaxes(part, -1, -2)) > 1e-12):
            raise CableInputError(f"{name} must be symmetric tensor data")
        peak_norm = max(peak_norm, float(np.max(np.linalg.norm(part, axis=(1, 2)))))
    error = float(np.linalg.norm(history[-1] / scale - history[0] / scale)) / peak_norm
    if error > closure_rtol:
        raise CableInputError(f"{name} must close over one repeated cycle")
    return scale, error


def _six_components(part: FloatArray, scale: float) -> FloatArray:
    normalized = part / scale if scale else np.zeros_like(part)
    return np.column_stack(
        (
            normalized[:, 0, 0],
            normalized[:, 1, 1],
            normalized[:, 2, 2],
            0.5 * normalized[:, 0, 1] + 0.5 * normalized[:, 1, 0],
            0.5 * normalized[:, 0, 2] + 0.5 * normalized[:, 2, 0],
            0.5 * normalized[:, 1, 2] + 0.5 * normalized[:, 2, 1],
        )
    )


def compute_swt_cycle(
    stress_Pa: ArrayLike,
    strain: ArrayLike,
    plane_normals: ArrayLike,
    material: SWTLifeConstants,
    *,
    material_point_id: str,
    closure_rtol: float = 1e-8,
    plane_batch_size: int = 64,
    time_batch_size: int = 4096,
) -> SWTCycleResult:
    """Recover sampled SWT planes from coincident mechanical tensor histories.

    Stress is tensile-positive in Pa; strain is dimensionless mechanical total
    strain (elastic plus plastic as appropriate to the fitted law). Both have
    shape (n_samples, 3, 3), at the same material point and times, in one fixed
    material coordinate frame. Off-diagonal strain entries are TENSOR shear
    epsilon_ij = engineering gamma_ij/2. No constitutive or thermal conversion
    and no contact stress reconstruction are inferred.

    Supply normals with shape (n_planes, 3), or one (3,) normal. Nonzero finite
    vectors are normalized. n and -n represent the same plane; duplicates are
    allowed, and exact ties retain the first supplied index. Refine both time
    and angular sampling: the reported critical plane is a sampled maximum.

    The record must describe ONE closed, repeatable cycle, including its last
    endpoint. Closure is checked by endpoint Frobenius difference relative to
    the largest history Frobenius norm, separately for stress and strain.
    This does not count irregular cycles or accumulate variable-amplitude life.
    material_point_id is a caller-declared identifier, not a location validator.

    Projection workspace is bounded by plane_batch_size * time_batch_size;
    histories are not expanded into a full time-by-plane tensor field.
    """
    if not isinstance(material_point_id, str) or not material_point_id.strip():
        raise CableInputError("a nonempty material_point_id is required")
    if not math.isfinite(closure_rtol) or not 0 <= closure_rtol < 1:
        raise CableInputError("closure_rtol must be finite and satisfy 0 <= value < 1")
    for size in (plane_batch_size, time_batch_size):
        if isinstance(size, bool) or not isinstance(size, Integral) or size < 1:
            raise CableInputError("batch sizes must be positive integers")
    stress, deformation = _history(stress_Pa, "stress"), _history(strain, "strain")
    if stress.shape != deformation.shape:
        raise CableInputError(
            "stress and strain must share sample times and array shape"
        )
    sigma_scale, sigma_closure = _history_scale_and_closure(
        stress, "stress", closure_rtol, time_batch_size
    )
    eps_scale, eps_closure = _history_scale_and_closure(
        deformation, "strain", closure_rtol, time_batch_size
    )
    if np.iscomplexobj(plane_normals):
        raise CableInputError("plane normals must be real")
    normals = np.atleast_2d(np.asarray(plane_normals, dtype=np.float64))
    if (
        normals.ndim != 2
        or normals.shape[1] != 3
        or not len(normals)
        or not np.isfinite(normals).all()
    ):
        raise CableInputError("plane normals must be finite with shape (n_planes, 3)")
    magnitudes = np.max(np.abs(normals), axis=1)
    if np.any(magnitudes == 0):
        raise CableInputError("plane normals must be nonzero")
    normals = normals / magnitudes[:, None]
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]
    maxima = np.full(len(normals), -np.inf)
    strain_maxima, strain_minima = (
        np.full(len(normals), -np.inf),
        np.full(len(normals), np.inf),
    )

    for first in range(0, len(normals), plane_batch_size):
        selected = slice(first, first + plane_batch_size)
        nx, ny, nz = normals[selected].T
        weights = np.column_stack(
            (nx * nx, ny * ny, nz * nz, 2 * nx * ny, 2 * nx * nz, 2 * ny * nz)
        )
        for start in range(0, len(stress), time_batch_size):
            stop = start + time_batch_size
            normal_stress = _six_components(stress[start:stop], sigma_scale) @ weights.T
            maxima[selected] = np.maximum(
                maxima[selected], np.max(normal_stress, axis=0)
            )
            normal_strain = (
                _six_components(deformation[start:stop], eps_scale) @ weights.T
            )
            strain_maxima[selected] = np.maximum(
                strain_maxima[selected], np.max(normal_strain, axis=0)
            )
            strain_minima[selected] = np.minimum(
                strain_minima[selected], np.min(normal_strain, axis=0)
            )
    with np.errstate(over="raise", invalid="raise", under="raise"):
        try:
            maximum_stress = maxima * sigma_scale
            amplitude = (0.5 * strain_maxima - 0.5 * strain_minima) * eps_scale
            drivers = np.maximum(maximum_stress, 0) * amplitude
        except FloatingPointError as error:
            raise CableInputError(
                "SWT projection exceeds floating-point range"
            ) from error
    critical = int(np.argmax(drivers)) if np.max(drivers) > 0 else None
    life = swt_life(float(np.max(drivers)), material)
    return SWTCycleResult(
        material_point_id=material_point_id,
        plane_normals=_readonly(normals),
        maximum_normal_stress_Pa=_readonly(maximum_stress),
        normal_strain_amplitude=_readonly(amplitude),
        swt_Pa=_readonly(drivers),
        critical_plane_index=critical,
        life_cycles=life,
        material=material,
        n_samples=len(stress),
        stress_closure_relative_error=sigma_closure,
        strain_closure_relative_error=eps_closure,
        provenance=Provenance(
            method="Sampled SWT planes at one material point over one repeated cycle",
            citations=("Poon et al. (2022), section 2.4; doi:10.1111/ffe.13683",),
            gates=(
                "Uniaxial/shear closed forms, tensor-frame invariance "
                "and sampled harmonic extrema",
            ),
            limitations=(
                "Caller-supplied coincident mechanical tensors in a fixed material "
                "frame; no contact or constitutive reconstruction.",
                "Critical plane is the maximum of supplied normals and samples; "
                "time, angular and spatial convergence are not inferred.",
                "One repeated cycle only; no irregular-history cycle counting, "
                "block damage summation or calibrated service-life qualification.",
                "Material calibration range and applicability of SWT to the "
                "failure mechanism must be assessed separately.",
            ),
        ),
    )
