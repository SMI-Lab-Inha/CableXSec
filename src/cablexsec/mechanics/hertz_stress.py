# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Resolved normal-contact stresses on the symmetry axis of a Hertz ellipse.

The implementation uses Greenwood (2022), Table 1, with scipy's Carlson RD
divided by three (the paper uses RD-dot). The table's x/y permutations are
retained; the displayed equation (11c) and the later prose surface-axis labels
contain inconsistent permutations. Independent Boussinesq integration and the
circular and line-contact limits verify the implemented axis convention.

Reference: https://doi.org/10.1177/03093247211060288
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import elliprd

from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.hertz_contact import HertzContact
from cablexsec.provenance import Provenance

FloatArray = NDArray[np.float64]

__all__ = ["HertzAxisStress", "compute_hertz_axis_stress"]


def _readonly(values: FloatArray) -> FloatArray:
    result = values.copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class HertzAxisStress:
    """Tensile-positive Cauchy stresses at the supplied contact-axis depths.

    The contact x/y directions follow a_m/b_m; z points into the body. Arrays
    have shapes (depth,), (depth, 3, 3) for tensors, and (depth, 3) for ascending
    principal stresses. Invariants refer to the total tensor, including an
    explicitly supplied, coincident background stress. Peak properties select
    only among the supplied samples, not the entire contact volume.
    """

    depth_m: FloatArray
    contact_stress_Pa: FloatArray
    total_stress_Pa: FloatArray
    principal_stress_Pa: FloatArray
    von_mises_Pa: FloatArray
    maximum_shear_Pa: FloatArray
    a_m: float
    b_m: float
    p0_Pa: float
    poisson_ratio: float
    provenance: Provenance

    @property
    def sampled_peak_von_mises_Pa(self) -> float:
        return float(np.max(self.von_mises_Pa))

    @property
    def sampled_peak_depth_m(self) -> float:
        return float(self.depth_m[np.argmax(self.von_mises_Pa)])


def compute_hertz_axis_stress(
    contact: HertzContact,
    depth_m: ArrayLike,
    *,
    poisson_ratio: float = 0.3,
    background_stress_Pa: ArrayLike | None = None,
) -> HertzAxisStress:
    """Resolve frictionless elliptical Hertz stress on its normal symmetry axis.

    ``poisson_ratio`` is for the body being assessed, not the reduced contact
    pair. The interval -1 < nu <= 0.5 includes the incompressible elastic limit.
    Depth is nonnegative, in metres; a scalar or nonempty 1-D array is accepted.
    Geometry and peak pressure come from ``solve_contact``. Either axis order
    is supported, with x following a_m and y following b_m.

    An optional background tensor must be symmetric, tensile-positive, expressed
    in THIS contact frame and evaluated at these SAME physical points. Supply a
    (3, 3) tensor for a uniform background or (n_depths, 3, 3) tensors. Small
    symmetry differences (relative 1e-12, zero absolute tolerance) are averaged.
    This is elastic stress superposition; it does not re-solve the footprint.

    Normal-contact stresses are diagonal on the axis. Shear traction, frictional
    partial slip, finite-wire boundaries, plasticity and stresses off this axis
    are absent. The contact patch must be small relative to body dimensions for
    the half-space approximation. A sampled peak is not a 3-D failure envelope
    or a fretting-life prediction.
    """
    a, b, pressure = contact.a_m, contact.b_m, contact.p0_Pa
    if not all(math.isfinite(x) for x in (a, b, pressure, poisson_ratio)):
        raise CableInputError(
            "contact dimensions, pressure and Poisson ratio must be finite"
        )
    if min(a, b) <= 0 or pressure < 0:
        raise CableInputError(
            "positive contact semi-axes and nonnegative pressure are required"
        )
    if not -1 < poisson_ratio <= 0.5:
        raise CableInputError("Poisson ratio must satisfy -1 < nu <= 0.5")
    depth = np.atleast_1d(np.asarray(depth_m, dtype=np.float64))
    if (
        depth.ndim != 1
        or not depth.size
        or not np.isfinite(depth).all()
        or np.any(depth < 0)
    ):
        raise CableInputError(
            "depth_m must be a nonempty finite nonnegative 1-D history or scalar"
        )

    background = np.zeros((depth.size, 3, 3))
    if background_stress_Pa is not None:
        supplied = np.asarray(background_stress_Pa, dtype=np.float64)
        if (
            supplied.shape not in {(3, 3), (depth.size, 3, 3)}
            or not np.isfinite(supplied).all()
        ):
            raise CableInputError(
                "background stress must be finite with shape (3, 3) or (depth, 3, 3)"
            )
        transpose = np.swapaxes(supplied, -1, -2)
        if not np.allclose(supplied, transpose, rtol=1e-12, atol=0):
            raise CableInputError(
                "background stress must be symmetric in the contact frame"
            )
        background[:] = 0.5 * supplied + 0.5 * transpose

    # Pointwise length normalization removes physical-unit powers from Carlson
    # arguments and avoids cancellation in the circular limit (no a^2-b^2).
    with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
        try:
            scale = np.maximum(max(a, b), depth)
            aa, bb, zz = a / scale, b / scale, depth / scale
            gamma = zz * zz
            p, q = aa * aa + gamma, bb * bb + gamma
            root_p, root_q = np.sqrt(p), np.sqrt(q)
            ab = aa * bb
            common = root_p + root_q
            # Greenwood's RD-dot is one third of scipy.special.elliprd.
            term_x = (2.0 / 3.0) * zz * elliprd(q, gamma, p)
            term_y = (2.0 / 3.0) * zz * elliprd(p, gamma, q)
            tail_x, tail_y = 1 / (root_p * common), 1 / (root_q * common)
            sigma_x = ab * (term_x - tail_x + poisson_ratio * (term_y - 2 * tail_y))
            sigma_y = ab * (term_y - tail_y + poisson_ratio * (term_x - 2 * tail_x))
            sigma_z = -ab / (root_p * root_q)
            tensor = np.zeros_like(background)
            tensor[:, 0, 0] = pressure * sigma_x
            tensor[:, 1, 1] = pressure * sigma_y
            tensor[:, 2, 2] = pressure * sigma_z
            total = tensor + background
            principal = np.asarray(np.linalg.eigvalsh(total), dtype=np.float64)
            # Half-differences avoid unnecessary overflow when opposite signed
            # principal stresses still have a representable maximum shear.
            half = 0.5 * principal
            differences = np.stack(
                (
                    half[:, 2] - half[:, 1],
                    half[:, 1] - half[:, 0],
                    half[:, 2] - half[:, 0],
                ),
                axis=1,
            )
            vm = math.sqrt(2) * np.hypot.reduce(differences, axis=1)
            shear = differences[:, 2]
        except FloatingPointError as error:
            raise CableInputError(
                "contact stress calculation exceeds floating-point range"
            ) from error
    if (
        not np.isfinite(total).all()
        or not np.isfinite(principal).all()
        or not np.isfinite(vm).all()
    ):
        raise CableInputError("contact stress calculation produced nonfinite values")

    return HertzAxisStress(
        depth_m=_readonly(depth),
        contact_stress_Pa=_readonly(tensor),
        total_stress_Pa=_readonly(total),
        principal_stress_Pa=_readonly(principal),
        von_mises_Pa=_readonly(vm),
        maximum_shear_Pa=_readonly(shear),
        a_m=a,
        b_m=b,
        p0_Pa=pressure,
        poisson_ratio=poisson_ratio,
        provenance=Provenance(
            method=(
                "Elliptical Hertz normal-contact axis stress "
                "via Carlson elliptic integrals"
            ),
            citations=("Greenwood (2022), Table 1; doi:10.1177/03093247211060288",),
            gates=(
                "Circular closed form and independent Boussinesq pressure integration",
            ),
            limitations=(
                "Frictionless isotropic elastic half-space; contact size must "
                "be small relative to body dimensions.",
                "Only the contact symmetry axis is sampled; off-axis stresses, "
                "shear traction, plasticity and wear are absent.",
                "Sampled peaks are not global contact-volume maxima or "
                "validated cable failure/fatigue predictions.",
                "Any background stress is caller-supplied at the same physical "
                "points in the contact coordinate frame; the footprint "
                "is not re-solved.",
            ),
        ),
    )
