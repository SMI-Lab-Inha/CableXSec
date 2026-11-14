# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Explicit transformations between rod resultants and local strain inputs.

Convention (right-handed; checked against a global line model, 17 October 2026).
The material frame (x, y, s) is right-handed, with s along the tangent t
from End A to End B. Global codes (``x/y curvature`` and ``x/y bend
moment`` exports, CableDyn, a RIFLEX rotation export) report the
rotation-curvature vector ``kappa = t x dt/ds`` on (x, y). A fibre at
(x, y) = r (cos phi, sin phi) then has axial strain::

    eps = k_x * y - k_y * x

and the bend moment from fibre forces f, conjugate to (k_x, k_y), is
``M = (sum f*y, -sum f*x)``. CableXSec wire routines take the axial-strain
gradient (b_x, b_y), with eps = b_x * x + b_y * y, and its conjugate moment
(m_x, m_y) = sum f*(x, y), so::

    (b_x, b_y) = (-k_y, k_x)        and        (m_x, m_y) = (-M_y, M_x).

This is a change of representation, not a rotation of the cross-section or
its wire azimuths: phi is measured from x towards y. In a global model a hogging
cantilever root, (k_x, k_y) = (0, +k), has its tensile fibre at phi = 180 deg;
a sag bend, (0, -k), at phi = 0.

``curvature_map="paper_r2"`` keeps the earlier mapping (b_x, b_y) =
(k_y, -k_x) and (m_x, m_y) = (M_y, -M_x), the negative of the right-handed
one. It places the tensile fibre 180 deg from its physical position. Damage
of equally spaced identical wires is unchanged for even wire counts and
differs by a few tenths of a per cent at most for odd counts. It is retained
only so that earlier results computed with it reproduce bit for bit; do
not use it for new work or for comparison with measured strains.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from cablexsec.exceptions import CableInputError

CurvatureMap = Literal["right_handed", "paper_r2"]
CURVATURE_MAPS: tuple[CurvatureMap, ...] = ("right_handed", "paper_r2")


def _require_map(curvature_map: str) -> None:
    if curvature_map not in CURVATURE_MAPS:
        raise CableInputError(
            f"curvature_map must be one of {CURVATURE_MAPS}, not {curvature_map!r}"
        )


def _to_gradient_frame(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    curvature_map: CurvatureMap,
    quantity: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    _require_map(curvature_map)
    x, y = np.broadcast_arrays(first, second)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise CableInputError(f"{quantity} components must be finite")
    if curvature_map == "paper_r2":
        return y.copy(), -x
    return -y, x.copy()


def rotation_to_strain_curvature(
    curvature_x: NDArray[np.float64],
    curvature_y: NDArray[np.float64],
    *,
    curvature_map: CurvatureMap = "right_handed",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Axial-strain gradient (b_x, b_y) [1/m] from rotation curvature (k_x, k_y).

    ``right_handed`` (default) returns (-k_y, k_x): the fibre strain
    b_x*x + b_y*y equals k_x*y - k_y*x. ``paper_r2`` returns the earlier
    (k_y, -k_x), kept only to reproduce earlier results.
    """
    return _to_gradient_frame(curvature_x, curvature_y, curvature_map, "curvature")


def rotation_to_strain_moment(
    moment_x: NDArray[np.float64],
    moment_y: NDArray[np.float64],
    *,
    curvature_map: CurvatureMap = "right_handed",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Moment (m_x, m_y) = sum f*(x, y) [N m] conjugate to the strain gradient.

    ``right_handed`` (default) maps the rotation-vector bend moment (M_x, M_y)
    to (-M_y, M_x), so that m . b = M . k with b from
    :func:`rotation_to_strain_curvature`. ``paper_r2`` returns the earlier
    (M_y, -M_x).
    """
    return _to_gradient_frame(moment_x, moment_y, curvature_map, "moment")


def true_tension_from_effective(
    effective_tension_N: NDArray[np.float64],
    external_pressure_Pa: NDArray[np.float64] | float,
    external_area_m2: float,
    *,
    internal_pressure_Pa: NDArray[np.float64] | float = 0.0,
    internal_area_m2: float = 0.0,
) -> NDArray[np.float64]:
    """Material resultant T=T_eff+p_i*A_i-p_o*A_o, in N.

    Areas must describe the same load-carrying section as the global tension.
    This force conversion does not itself account for pressure-induced radial
    strain or Poisson coupling in a local constitutive model.
    """
    areas = [external_area_m2, internal_area_m2]
    if (
        not np.isfinite(areas).all()
        or min(areas) < 0
        or internal_area_m2 > external_area_m2
    ):
        raise CableInputError(
            "finite non-negative areas with inner <= outer are required"
        )
    effective, outside, inside = np.broadcast_arrays(
        effective_tension_N, external_pressure_Pa, internal_pressure_Pa
    )
    if not all(np.isfinite(a).all() for a in (effective, outside, inside)):
        raise CableInputError("force and pressure histories must be finite")
    if np.any(outside < 0) or np.any(inside < 0):
        raise CableInputError("pressure histories must be non-negative")
    return np.asarray(
        effective + inside * internal_area_m2 - outside * external_area_m2
    )
