# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Bending-range checks for the declared elastic metallic-sheath reduction.

This is a necessary material-range check, not a whole-cable design assessment.
It neither identifies corrugation properties nor adds an elastoplastic law.
"""

from __future__ import annotations

import math

from cablexsec.cables.schema import CylindricalShellLayer
from cablexsec.materials import library
from cablexsec.mechanics.shell_bending import (
    shell_bending_stiffness,
    shell_wall_stress,
)


def assess_shell_bending(cable, max_curvature_1_m):
    """Compare the model's peak pure-bending stress with its material limit.

    Curvature is the maximum magnitude, including the static mean rather than
    only the cyclic range. Axial, pressure and thermal stresses still require
    their separate combined assessment. A representative corrugation factor
    is not qualified by satisfying this check.
    """
    kappa = float(max_curvature_1_m)
    if not math.isfinite(kappa) or kappa < 0:
        raise ValueError("maximum curvature must be finite and nonnegative")
    layers = []
    for layer in cable.layers:
        if not isinstance(layer, CylindricalShellLayer) or not layer.structural:
            continue
        stress = shell_wall_stress(layer, kappa, 0.0, 1.0)
        try:
            limit = library.get_property(layer.material, "sigma_y_Pa")
        except KeyError:
            limit = None
        if limit is not None and (not math.isfinite(limit) or limit <= 0):
            limit = None
        peak = abs(stress.sigma_peak_Pa)
        layers.append(
            {
                "layer": layer.name,
                "material": layer.material,
                "corrugation_factor": layer.f_corrugation,
                "SCF": layer.scf,
                "EI_Nm2": shell_bending_stiffness(layer),
                "nominal_bending_stress_Pa": abs(stress.sigma_bending_Pa),
                "root_bending_stress_Pa": peak,
                "yield_stress_Pa": limit,
                "root_bending_utilization": None if limit is None else peak / limit,
                "status": (
                    "missing_material_limit"
                    if limit is None
                    else "exceeds_elastic_bending_range"
                    if peak > limit
                    else "within_elastic_bending_range"
                ),
            }
        )
    statuses = {layer["status"] for layer in layers}
    status = (
        "exceeds_elastic_bending_range"
        if "exceeds_elastic_bending_range" in statuses
        else "missing_material_limit"
        if "missing_material_limit" in statuses
        else "within_elastic_bending_range"
        if layers
        else "no_structural_shells"
    )
    return {
        "scope": "declared structural metallic-sheath pure bending only; "
        "not combined loading, corrugation qualification or whole-cable acceptance",
        "max_curvature_1_m": kappa,
        "status": status,
        "layers": layers,
    }


def require_shell_bending_range(assessment):
    """Stop an operating workflow whose elastic sheath has not been assessed."""
    if not isinstance(assessment, dict) or assessment.get("status") not in {
        "within_elastic_bending_range",
        "no_structural_shells",
    }:
        raise ValueError(
            "The declared metallic-sheath elastic bending range is exceeded "
            "or not assessed. Preserve this case and reassess the construction "
            "before an operating campaign or fatigue interpretation."
        )
