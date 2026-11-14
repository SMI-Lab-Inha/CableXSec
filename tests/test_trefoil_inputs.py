# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Trefoil bundle model: input rules, snug-fit overlaps and filled core gaps."""

from __future__ import annotations

import math

import pytest

from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.concentric_section import ElasticAnnulus
from cablexsec.mechanics.trefoil_section import (
    ElasticFiller,
    TrefoilBundle,
    build_trefoil_model,
)

A = 0.02
RC = 2 * A / math.sqrt(3)  # touching cores: rc sin(60 deg) = a


def _bundle(**kwargs) -> TrefoilBundle:
    base = {
        "core_layers": (
            ElasticAnnulus("cond", 0.0, 0.01, 117e9, 0.34),
            ElasticAnnulus("ins", 0.01, A, 0.15e9, 0.45),
        ),
        "core_centre_radius_m": RC,
        "ring_layers": (ElasticAnnulus("ring", RC + A, RC + A + 0.003, 0.9e9, 0.46),),
        "angular_cells": 12,
    }
    base.update(kwargs)
    return TrefoilBundle(**base)


def test_envelope_is_the_core_centre_plus_the_core_radius():
    bundle = _bundle()
    assert bundle.core_radius_m == A
    assert bundle.envelope_radius_m == pytest.approx(RC + A)
    # Touching cores in a snug ring: no overlap to adjust, no gap to fill.
    model = build_trefoil_model(bundle)
    assert model.adjusted_overlaps_m == {}
    assert model.filled_core_gaps == ()


def _ring(ri: float, ro: float, **kwargs) -> ElasticAnnulus:
    return ElasticAnnulus(
        "ring", ri, ro, kwargs.get("e", 0.9e9), kwargs.get("nu", 0.46)
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"core_layers": ()}, "at least one layer"),
        ({"ring_layers": ()}, "needs a surrounding ring"),
        (
            {
                "ring_layers": (
                    _ring(RC + A, RC + A + 0.002),
                    _ring(RC + A + 0.003, RC + A + 0.005),
                )
            },
            "ring layers must be contiguous",
        ),
        ({"ring_layers": (_ring(RC + A, RC + A + 0.003, e=math.nan),)}, "finite"),
        ({"ring_layers": (_ring(RC + A, RC + A),)}, "positive thickness"),
        (
            {"ring_layers": (_ring(RC + A, RC + A + 0.003, nu=0.5),)},
            "nu in \\(-1, 0.5\\)",
        ),
        ({"filler": ElasticFiller("pp", 1.5e9, 0.5)}, "filler modulus"),
        ({"filler": ElasticFiller("pp", -1.0, 0.4)}, "filler modulus"),
        ({"angular_cells": 20}, "multiple of six"),
        ({"angular_cells": 6}, "multiple of six"),
        ({"angular_cells": True}, "multiple of six"),
        ({"core_lay_angle_rad": math.pi / 4}, "below 45 degrees"),
        ({"core_centre_radius_m": 0.05}, "ring bore must enclose"),
        ({"radial_cell_size_m": -1e-3}, "radial cell size must be positive"),
        (
            {
                "core_layers": (
                    ElasticAnnulus("cond", 0.0, 0.012, 117e9, 0.34),
                    ElasticAnnulus("ins", 0.01, A, 0.15e9, 0.45),
                )
            },
            "core layers overlap",
        ),
        (
            {"ring_layers": (_ring(RC + A - 1e-3, RC + A + 0.003),)},
            "core_ring geometric overlap",
        ),
        ({"core_centre_radius_m": 0.95 * RC}, "core_core geometric overlap"),
    ],
)
def test_invalid_bundles_are_rejected(kwargs, message):
    with pytest.raises(CableInputError, match=message):
        build_trefoil_model(_bundle(**kwargs))


def test_small_overlaps_are_taken_up_as_a_snug_fit():
    bore_overlap = 1e-5
    rc = RC * (1 - 1e-4)
    bundle = _bundle(
        core_centre_radius_m=rc,
        ring_layers=(_ring(rc + A - bore_overlap, rc + A + 0.003),),
    )
    model = build_trefoil_model(bundle)
    assert model.adjusted_overlaps_m["core_ring"] == pytest.approx(bore_overlap)
    assert model.adjusted_overlaps_m["core_core"] == pytest.approx(
        A - rc * math.sin(math.pi / 3)
    )


@pytest.mark.parametrize(("soft_inner", "donor"), [(True, "cond"), (False, "ins")])
def test_a_radial_core_gap_is_filled_with_the_softer_neighbour(soft_inner, donor):
    e_in, e_out = (1e9, 2e9) if soft_inner else (2e9, 1e9)
    core = (
        ElasticAnnulus("cond", 0.0, 0.009, e_in, 0.34),
        ElasticAnnulus("ins", 0.01, A, e_out, 0.45),
    )
    model = build_trefoil_model(_bundle(core_layers=core))
    (note,) = model.filled_core_gaps
    assert note.startswith("from 9 to 10 mm radius filled")
    assert note.endswith(f"material of the {donor}")
