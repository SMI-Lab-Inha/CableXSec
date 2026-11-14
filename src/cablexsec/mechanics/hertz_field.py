# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Subsurface tensors from prescribed normal and tangential Hertz tractions.

The Boussinesq and Cerruti kernels are Johnson (1985), Contact Mechanics,
sections 3.2 and 3.6, equations 3.19/3.20 and 3.76. Normal pressure and the two
signed shear tractions are integrated at the SAME supplied spatial points.
This is a traction-to-stress calculation, not a partial-slip contact solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import roots_legendre

from cablexsec.exceptions import CableInputError, CableSolverError
from cablexsec.mechanics.hertz_contact import HertzContact
from cablexsec.provenance import Provenance

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

__all__ = ["HertzFieldStress", "compute_hertz_subsurface_stress"]


def _readonly(values: FloatArray) -> FloatArray:
    result = values.copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class HertzFieldStress:
    """Coincident tensors at supplied points; peaks are only sampled maxima.

    x/y follow the contact a/b semi-axes, z points into the assessed body, and
    stress is tensile-positive. Traction ratios are the applied surface forces
    in +x/+y divided by positive pressure. Thus the surface boundary condition
    approached from within the body is sigma_iz = -(qx, qy, p)_i.
    Refinement differences are numerical diagnostics, not certified error bounds.
    """

    points_m: FloatArray
    normal_stress_Pa: FloatArray
    tangential_stress_Pa: FloatArray
    total_stress_Pa: FloatArray
    principal_stress_Pa: FloatArray
    von_mises_Pa: FloatArray
    maximum_shear_Pa: FloatArray
    tangential_traction_ratio: FloatArray
    quadrature_order: IntArray
    refinement_difference_Pa: FloatArray
    semi_axes_m: tuple[float, float]
    peak_pressure_Pa: float
    poisson_ratio: float
    provenance: Provenance

    @property
    def sampled_peak_von_mises_Pa(self) -> float:
        return float(np.max(self.von_mises_Pa))

    @property
    def sampled_peak_point_m(self) -> FloatArray:
        return _readonly(self.points_m[np.argmax(self.von_mises_Pa)])


def _point_stress(relative: FloatArray, nu: float) -> FloatArray:
    """Unit +z/+x/+y point-force kernels, shape (3, n, 6), tensile-positive.

    Component order is xx, yy, zz, xy, xz, yz. Radius normalization avoids
    high physical-coordinate powers and the removable radial-axis singularity
    in Johnson's Cartesian Boussinesq expression.
    """
    radius = np.hypot.reduce(relative, axis=1)
    x, y, z = (relative / radius[:, None]).T
    h = 1 + z
    v = 1 - 2 * nu
    c = (2 + z) / h**2
    normal = np.column_stack(
        (
            v * (z - 1 / h + x * x * c) - 3 * z * x * x,
            v * (z - 1 / h + y * y * c) - 3 * z * y * y,
            -3 * z**3,
            v * x * y * c - 3 * x * y * z,
            -3 * x * z * z,
            -3 * y * z * z,
        )
    )

    def tangential(xx: FloatArray, yy: FloatArray) -> FloatArray:
        tail = 1 / h**2 + 2 / h**3
        return np.column_stack(
            (
                -3 * xx**3 + v * (xx - 3 * xx / h**2 + xx**3 * tail),
                -3 * xx * yy**2 + v * (xx - xx / h**2 + xx * yy**2 * tail),
                -3 * xx * z * z,
                -3 * xx * xx * yy + v * (-yy / h**2 + xx * xx * yy * tail),
                -3 * xx * xx * z,
                -3 * xx * yy * z,
            )
        )

    along_x = tangential(x, y)
    along_y = tangential(y, x)[:, [1, 0, 2, 3, 5, 4]]
    return np.asarray(
        np.stack((normal, along_x, along_y)) / (2 * math.pi * radius**2)[None, :, None],
        dtype=np.float64,
    )


@lru_cache(maxsize=5)
def _quadrature(order: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Unit-disk Hertz quadrature; sine substitution removes the edge root."""
    nodes, weights = roots_legendre(order)
    t = (nodes + 1) * (math.pi / 4)
    angle = (np.arange(4 * order) + 0.5) * (math.pi / (2 * order))
    u = np.sin(t)[:, None] * np.cos(angle)
    v = np.sin(t)[:, None] * np.sin(angle)
    radial = weights * (math.pi / 4) * np.sin(t) * np.cos(t) ** 2
    area = np.broadcast_to(radial[:, None] * (math.pi / (2 * order)), u.shape)
    return _readonly(u.ravel()), _readonly(v.ravel()), _readonly(area.ravel())


def _integrate(
    a: float, b: float, point: FloatArray, nu: float, order: int
) -> FloatArray:
    u, v, weights = _quadrature(order)
    result = np.zeros((3, 6))
    for start in range(0, len(u), 8192):
        stop = start + 8192
        relative = np.column_stack(
            (
                point[0] - a * u[start:stop],
                point[1] - b * v[start:stop],
                np.full_like(u[start:stop], point[2]),
            )
        )
        result += np.einsum(
            "j,ijk->ik", weights[start:stop], _point_stress(relative, nu)
        )
    return np.asarray(result * (a * b), dtype=np.float64)


def _tensor(values: FloatArray) -> FloatArray:
    result = np.zeros((len(values), 3, 3))
    result[:, 0, 0], result[:, 1, 1], result[:, 2, 2] = values[:, :3].T
    result[:, 0, 1] = result[:, 1, 0] = values[:, 3]
    result[:, 0, 2] = result[:, 2, 0] = values[:, 4]
    result[:, 1, 2] = result[:, 2, 1] = values[:, 5]
    return result


def compute_hertz_subsurface_stress(
    contact: HertzContact,
    points_m: ArrayLike,
    *,
    poisson_ratio: float = 0.3,
    tangential_traction_ratio: ArrayLike = (0.0, 0.0),
    background_stress_Pa: ArrayLike | None = None,
    rtol: float = 1e-6,
    atol_Pa: float = 1e-3,
    initial_order: int = 16,
    max_order: int = 512,
) -> HertzFieldStress:
    """Integrate prescribed Hertz-shaped tractions at positive-depth points.

    Pressure is p0*sqrt(1-(x/a)^2-(y/b)^2) on the ellipse. The signed applied
    shear tractions are qx=ratio[0]*p and qy=ratio[1]*p. A ratio vector of length
    mu represents a prescribed gross-sliding traction field; this function
    neither infers slip nor solves a Cattaneo-Mindlin history or dissimilar-body
    normal/tangential coupling. Intermediate ratios do NOT imply partial slip.

    Points have shape (n,3) or (3,), in metres and in the contact frame. z must
    be strictly positive. No surface limit or finite-wire boundary correction
    is substituted. Optional background tensors are coincident in the SAME
    frame, shape (3,3) or (n,3,3); addition does not change the footprint.

    Quadrature doubles radial order (4 times as many azimuthal nodes). Two
    successive changes must meet atol_Pa + rtol*stress_scale. The scale and
    changes use absolute contributions of each force basis, avoiding false
    convergence from normal/shear cancellation. A geometric spacing check also
    guards unresolved shallow peaks. Exhaustion raises RuntimeError; refine
    max_order or use a separately verified near-surface formulation. Returned
    differences are convergence diagnostics, not rigorous integration bounds.

    Orders are bounded at 1024 to limit memory. Only one point and 8192 source
    nodes are projected at a time; cached quadrature grids are bounded separately.
    """
    a, b, pressure = contact.a_m, contact.b_m, contact.p0_Pa
    if not all(
        math.isfinite(x) for x in (a, b, pressure, poisson_ratio, rtol, atol_Pa)
    ):
        raise CableInputError("contact, elastic and tolerance inputs must be finite")
    if min(a, b) <= 0 or pressure < 0 or not -1 < poisson_ratio <= 0.5:
        raise CableInputError(
            "positive axes, nonnegative pressure and -1 < nu <= 0.5 required"
        )
    if not 0 <= rtol < 1 or atol_Pa < 0 or rtol == atol_Pa == 0:
        raise CableInputError(
            "nonnegative tolerances, at least one positive, and rtol < 1 required"
        )
    for order in (initial_order, max_order):
        if (
            isinstance(order, bool)
            or not isinstance(order, Integral)
            or not 4 <= order <= 1024
        ):
            raise CableInputError("quadrature orders must be integers from 4 to 1024")
    if max_order <= initial_order:
        raise CableInputError("max_order must exceed initial_order")
    if np.iscomplexobj(points_m) or np.iscomplexobj(tangential_traction_ratio):
        raise CableInputError("points and traction ratios must be real")
    points = np.atleast_2d(np.asarray(points_m, dtype=np.float64))
    ratios = np.asarray(tangential_traction_ratio, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not len(points)
        or not np.isfinite(points).all()
        or np.any(points[:, 2] <= 0)
    ):
        raise CableInputError(
            "points must have shape (n,3), be finite and have strictly positive depth"
        )
    if ratios.shape != (2,) or not np.isfinite(ratios).all():
        raise CableInputError(
            "tangential_traction_ratio must contain two finite components"
        )
    background = np.zeros((len(points), 3, 3))
    if background_stress_Pa is not None:
        if np.iscomplexobj(background_stress_Pa):
            raise CableInputError("background stress must be real")
        supplied = np.asarray(background_stress_Pa, dtype=np.float64)
        if (
            supplied.shape not in {(3, 3), background.shape}
            or not np.isfinite(supplied).all()
        ):
            raise CableInputError(
                "background stress must be finite with shape (3,3) or (n,3,3)"
            )
        transpose = np.swapaxes(supplied, -1, -2)
        if not np.allclose(supplied, transpose, rtol=1e-12, atol=0):
            raise CableInputError(
                "background stress must be symmetric in the contact frame"
            )
        background[:] = 0.5 * supplied + 0.5 * transpose
    normal, tangential = np.zeros((len(points), 6)), np.zeros((len(points), 6))
    orders = np.zeros(len(points), dtype=np.int64)
    differences = np.zeros(len(points))
    coefficients = np.r_[1.0, ratios]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            if pressure:
                for index, point in enumerate(points):
                    scale = max(a, b, float(np.max(np.abs(point))))
                    aa, bb, position = a / scale, b / scale, point / scale
                    separation = max(
                        position[2], math.hypot(position[0], position[1]) - max(aa, bb)
                    )
                    previous = _integrate(
                        aa, bb, position, poisson_ratio, initial_order
                    )
                    order, consecutive = initial_order, 0
                    while order < max_order:
                        order = min(2 * order, max_order)
                        current = _integrate(aa, bb, position, poisson_ratio, order)
                        error = (
                            float(
                                np.max(
                                    np.sum(
                                        np.abs(current - previous)
                                        * np.abs(coefficients[:, None]),
                                        axis=0,
                                    )
                                )
                            )
                            * pressure
                        )
                        magnitude = (
                            float(
                                np.max(
                                    np.sum(
                                        np.abs(current) * np.abs(coefficients[:, None]),
                                        axis=0,
                                    )
                                )
                            )
                            * pressure
                        )
                        consecutive = (
                            consecutive + 1
                            if error <= atol_Pa + rtol * magnitude
                            else 0
                        )
                        resolved = math.pi * max(aa, bb) / (2 * order) <= separation
                        if consecutive >= 2 and resolved:
                            break
                        previous = current
                    else:
                        raise CableSolverError(
                            f"contact quadrature did not converge at point {index}; "
                            f"max_order={max_order}"
                        )
                    normal[index] = pressure * current[0]
                    tangential[index] = pressure * (ratios @ current[1:])
                    if normal[index, 2] == 0:
                        raise CableInputError(
                            "normal contact stress is outside floating-point range"
                        )
                    orders[index], differences[index] = order, error
            normal_tensor, tangential_tensor = _tensor(normal), _tensor(tangential)
            total = normal_tensor + tangential_tensor + background
            principal = np.asarray(np.linalg.eigvalsh(total), dtype=np.float64)
            half = 0.5 * principal
            pairwise = np.column_stack(
                (
                    half[:, 2] - half[:, 1],
                    half[:, 1] - half[:, 0],
                    half[:, 2] - half[:, 0],
                )
            )
            vm = math.sqrt(2) * np.hypot.reduce(pairwise, axis=1)
        except FloatingPointError as error:
            raise CableInputError(
                "contact field exceeds floating-point range"
            ) from error
    if not all(
        np.isfinite(value).all() for value in (total, principal, vm, differences)
    ):
        raise CableInputError("contact field produced nonfinite values")
    orders.flags.writeable = False
    return HertzFieldStress(
        points_m=_readonly(points),
        normal_stress_Pa=_readonly(normal_tensor),
        tangential_stress_Pa=_readonly(tangential_tensor),
        total_stress_Pa=_readonly(total),
        principal_stress_Pa=_readonly(principal),
        von_mises_Pa=_readonly(vm),
        maximum_shear_Pa=_readonly(pairwise[:, 2]),
        tangential_traction_ratio=_readonly(ratios),
        quadrature_order=orders,
        refinement_difference_Pa=_readonly(differences),
        semi_axes_m=(a, b),
        peak_pressure_Pa=pressure,
        poisson_ratio=poisson_ratio,
        provenance=Provenance(
            method="Boussinesq-Cerruti integration of prescribed Hertz tractions",
            citations=("Johnson (1985), Contact Mechanics, sections 3.2 and 3.6",),
            gates=(
                "Point-force displacement derivatives, equilibrium "
                "and analytical contact limits",
            ),
            limitations=(
                "Homogeneous isotropic elastic half-space at positive depth; "
                "no surface, finite-body, plasticity or wear solution.",
                "Shear traction is prescribed proportional to Hertz pressure; "
                "no partial-slip history or coupled contact solution is inferred.",
                "Refinement differences are diagnostics, not certified error bounds; "
                "peaks refer only to supplied spatial samples.",
                "Background tensors must be coincident and in the contact frame. "
                "No cable fatigue-life qualification is established.",
            ),
        ),
    )
