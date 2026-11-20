# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Compiled assembly of the fixed-fibre wire stress field.

:func:`assemble_wire_field` forms the three components of
:class:`cablexsec.analysis.fatigue.WireStressField` (``mean_Pa``,
``radial_Pa``, ``tangential_Pa``) from the friction stress, the per-sample
axial and thermal stress and the curvature history in one pass, with the
arithmetic of the NumPy expressions it replaces, operation for operation:
the local bending stress of
:func:`cablexsec.mechanics.wire_stress.local_bending_fibre_stress` at the
outward radial fibre and at the edge (fibre angle ``pi/2``), the outward
total ``((sigma_A + sigma_B) + sigma_f) + sigma_T`` plus any termination
increment, and ``mean = outward - radial``. The trigonometric factors are
evaluated by NumPy, as in the reference expressions, and passed in. The
results are equal to the NumPy path bit for bit
(``tests/test_fast_fatigue_path.py``).

Without numba (or with ``CABLEXSEC_DISABLE_NUMBA=1``) the NumPy expressions
are used instead (see :mod:`cablexsec._numba`).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from cablexsec._numba import HAVE_NUMBA, njit

__all__ = ["HAVE_NUMBA", "assemble_wire_field"]

FloatArray = NDArray[np.float64]


@njit(cache=True)
def _assemble_kernel(
    kx: FloatArray,
    ky: FloatArray,
    cos_phi: FloatArray,
    sin_phi: FloatArray,
    coef_radial: FloatArray,
    coef_edge: FloatArray,
    c_radial: float,
    c_tangential: float,
    cos_outward: float,
    sin_outward: float,
    cos_edge: float,
    sin_edge: float,
    axial: FloatArray,
    friction: FloatArray,
    thermal: FloatArray,
    extra: FloatArray,
    has_extra: bool,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    n_w, n_t = friction.shape
    mean = np.empty((n_w, n_t))
    radial = np.empty((n_w, n_t))
    edge = np.empty((n_w, n_t))
    for w in range(n_w):
        cp = cos_phi[w]
        sp = sin_phi[w]
        for i in range(n_t):
            x = kx[i]
            y = ky[i]
            rb = c_radial * (x * cp + y * sp)
            tb = c_tangential * ((-x) * sp + y * cp)
            sb = coef_radial[i] * (rb * cos_outward + tb * sin_outward)
            out = ((axial[i] + sb) + friction[w, i]) + thermal[i]
            if has_extra:
                out = out + extra[w, i]
            radial[w, i] = sb
            edge[w, i] = coef_edge[i] * (rb * cos_edge + tb * sin_edge)
            mean[w, i] = out - sb
    return mean, radial, edge


def assemble_wire_field(
    kappa_x: FloatArray,
    kappa_y: FloatArray,
    phi: FloatArray,
    modulus_Pa: float | FloatArray,
    depth_m: float,
    width_m: float,
    lay_angle_rad: float,
    axial_Pa: FloatArray,
    friction_Pa: FloatArray,
    thermal_Pa: FloatArray,
    extra_Pa: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``(mean, radial, tangential)`` of the wire stress field [Pa].

    Equal bit for bit to the NumPy expressions ::

        radial = local_bending_fibre_stress(kx, ky, phi[:, None], E, depth, a)
        tangential = local_bending_fibre_stress(
            kx, ky, phi[:, None], E, width, a, pi / 2
        )
        outward = axial[None, :] + radial + friction + thermal[None, :]
        outward = outward + extra  # when given
        mean = outward - radial

    ``modulus_Pa`` is a constant or one value per sample; ``axial_Pa`` and
    ``thermal_Pa`` hold one value per sample, ``friction_Pa`` and
    ``extra_Pa`` one row per wire.
    """
    kx = np.ascontiguousarray(kappa_x, dtype=np.float64)
    ky = np.ascontiguousarray(kappa_y, dtype=np.float64)
    friction = np.ascontiguousarray(friction_Pa, dtype=np.float64)
    n_t = kx.size
    angle = np.asarray(phi, dtype=np.float64)[:, None]
    cos_phi = np.ascontiguousarray(np.cos(angle)[:, 0])
    sin_phi = np.ascontiguousarray(np.sin(angle)[:, 0])
    # Factors exactly as local_bending_fibre_stress forms them.
    c, s = math.cos(lay_angle_rad), math.sin(lay_angle_rad)
    c_radial = c * c * math.cos(2 * lay_angle_rad)
    c_tangential = c * (1 + s * s)
    modulus = np.asarray(modulus_Pa)
    coef_radial = np.ascontiguousarray(
        np.broadcast_to(modulus * depth_m / 2, (n_t,)), dtype=np.float64
    )
    coef_edge = np.ascontiguousarray(
        np.broadcast_to(modulus * width_m / 2, (n_t,)), dtype=np.float64
    )
    outward_theta = np.asarray(0.0)
    edge_theta = np.asarray(math.pi / 2)
    extra = (
        np.zeros((0, 0))
        if extra_Pa is None
        else np.ascontiguousarray(extra_Pa, dtype=np.float64)
    )
    return _assemble_kernel(
        kx,
        ky,
        cos_phi,
        sin_phi,
        coef_radial,
        coef_edge,
        c_radial,
        c_tangential,
        float(np.cos(outward_theta)),
        float(np.sin(outward_theta)),
        float(np.cos(edge_theta)),
        float(np.sin(edge_theta)),
        np.ascontiguousarray(axial_Pa, dtype=np.float64),
        friction,
        np.ascontiguousarray(thermal_Pa, dtype=np.float64),
        extra,
        extra_Pa is not None,
    )
