# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Slip onset and full-slip curvature of each armour layer against tension.

The bending law of a cable (:func:`~cablexsec.analysis.bending_law.
export_bending_law`) sticks below the slip onset curvature of its armour and
reaches its full-slip slope beyond the full-slip curvature. Both curvatures
rise with the tension, which clamps the armour against its neighbours; this
module evaluates them on a tension grid for the same contact chain as the
bending law (the bounded radial chain, planar bending):

* the slip onset is the neutral-axis limit of
  :func:`~cablexsec.mechanics.papailiou.slip_onset_curvature`,
  ``mu P (2 pi r / n) cos(a) g_min / (A |sin a| k)`` with the stick stiffness
  ``k = xi E cos^2(a)``, where the first wire starts to slip;
* the full-slip curvature is
  :func:`~cablexsec.mechanics.papailiou.full_slip_curvature`, where the last
  wire starts to slip.

Zero clamping (no tension and no other contact load) gives zero for both.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cablexsec.cable_adapters import cable_to_mkappa_specs, structural_armour_layers
from cablexsec.cables.schema import CableData
from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.papailiou import full_slip_curvature, slip_onset_curvature

__all__ = ["LayerSlipCurvatures", "armour_slip_curvatures"]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LayerSlipCurvatures:
    """Slip curvatures of one armour layer at each tension of the grid."""

    layer: str
    onset_1_m: FloatArray
    """Curvature at which the first wire slips (neutral-axis limit) [1/m]."""
    full_slip_1_m: FloatArray
    """Curvature from which every wire of the layer slips [1/m]."""


def armour_slip_curvatures(
    cable: CableData,
    tensions_N: Sequence[float] | FloatArray,
    *,
    mu: float | None = None,
) -> tuple[LayerSlipCurvatures, ...]:
    """Slip onset and full-slip curvature of every structural armour layer.

    Parameters
    ----------
    cable : CableData
        A cable with fully specified structural armour.
    tensions_N : sequence of float
        Tensions at which the curvatures are evaluated [N]; finite.
    mu : float, optional
        Friction coefficient of the armour; ``None`` takes the armour
        material's own, as the bending law does.

    Returns
    -------
    tuple of LayerSlipCurvatures
        One entry per structural armour layer, inside out.
    """
    tensions = np.asarray(tensions_N, dtype=np.float64).ravel()
    if tensions.size < 1 or not np.isfinite(tensions).all():
        raise CableInputError("the tensions must be finite, one or more")
    names = [layer.name for layer in structural_armour_layers(cable)]
    onset = np.zeros((len(names), tensions.size))
    full = np.zeros((len(names), tensions.size))
    for j, tension in enumerate(tensions):
        specs = cable_to_mkappa_specs(
            cable, float(tension), mu=mu, contact_model="bounded"
        )
        for i, spec in enumerate(specs):
            onset[i, j] = slip_onset_curvature(spec)
            full[i, j] = full_slip_curvature(spec)
    return tuple(
        LayerSlipCurvatures(name, onset[i], full[i]) for i, name in enumerate(names)
    )
