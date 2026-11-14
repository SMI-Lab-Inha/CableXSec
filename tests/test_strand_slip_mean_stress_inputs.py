# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Strand-slip operator and mean-stress corrections: input checks and edges."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cablexsec.exceptions import CableInputError
from cablexsec.materials.mean_stress import MeanStressSpec, equivalent_range
from cablexsec.mechanics.strand_slip import (
    ConductorRing,
    StrandSlipParameters,
    conductor_bending_moment,
    conductor_ring_parameters,
    interface_line_loads,
    strand_slip_strain,
)

PARAMS = StrandSlipParameters(1.4e-9, (0.04, 0.01), (2e-4,), 1e3)
K = 0.05 * np.sin(np.linspace(0.0, 4.0 * math.pi, 41))


def _rings():
    d = 3.21e-3
    return [
        ConductorRing(1, d, 0.0, 0.0),
        ConductorRing.from_pitch(6, d, d, 0.265, 1),
        ConductorRing.from_pitch(12, d, 2 * d, 0.285, -1),
    ]


# --------------------------------------------------------------------------- #
# Strand slip                                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("args", "kwargs", "match"),
    [
        ((K.astype(complex), K), {}, "real"),
        ((K, K[:-1]), {}, "equal 1-D"),
        ((K * np.inf, K), {}, "finite"),
        ((K, K), {"tension_N": np.ones(3)}, "tension must be finite"),
        ((K, K), {"n_orientations": 0}, "n_orientations"),
        ((K, K), {"theta_rad": np.array([[0.0]])}, "theta_rad"),
    ],
)
def test_strand_slip_checks(args, kwargs, match):
    with pytest.raises(CableInputError, match=match):
        strand_slip_strain(PARAMS, *args, **kwargs)


def test_zero_reference_tension_keeps_the_zero_tension_amplitude():
    # Tension-independent friction strain: no reference tension needed.
    params = StrandSlipParameters(0.0, (0.04, 0.01), (1e-4,), 0.0, (1e-4,))
    np.testing.assert_allclose(params.friction_amplitude(np.array([0.0, 5e3])), 1e-4)


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((6, 0.003, 0.01, math.inf), "finite"),
        ((6, 0.0, 0.01, 0.1), "positive diameter"),
        ((6, 0.003, 0.01, math.pi / 2), "lay angle"),
    ],
)
def test_ring_checks(args, match):
    with pytest.raises(CableInputError, match=match):
        ConductorRing(*args)
    with pytest.raises(CableInputError, match="wire_area_m2"):
        ConductorRing(6, 0.003, 0.01, 0.1, wire_area_m2=-1.0)


def test_interface_and_parameter_checks():
    rings = _rings()
    with pytest.raises(CableInputError, match="pressure_transmission"):
        interface_line_loads(rings, 115e9, 1e-3, pressure_transmission="inner")  # type: ignore[arg-type]
    loads = interface_line_loads(rings, 115e9, 1e-3, external_pressure_Pa=1e5)
    assert len(loads) == len(rings) - 1 and np.all(np.isfinite(loads))
    for kwargs, match in (
        ({"rings": []}, "at least one ring"),
        ({"E_Pa": -1.0}, "positive|finite"),
        ({"reference_tension_N": 0.0}, "reference tension"),
        ({"fibre": "core"}, "fibre"),
        ({"mu_outer": -0.1}, "mu_outer"),
        ({"strain_per_tension": 0.0}, "strain_per_tension"),
    ):
        args = {
            "rings": rings,
            "E_Pa": 115e9,
            "mu_wire": 0.3,
            "reference_tension_N": 3e4,
        }
        args.update(kwargs)
        with pytest.raises(CableInputError, match=match):
            conductor_ring_parameters(**args)
    params = conductor_ring_parameters(
        rings, 115e9, mu_wire=0.3, reference_tension_N=3e4
    )
    with pytest.raises(CableInputError, match="one parameter set per ring"):
        conductor_bending_moment(
            rings, 115e9, params[:-1], K, 0 * K, np.full(K.size, 3e4)
        )


# --------------------------------------------------------------------------- #
# Mean stress                                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"method": "morrow"}, "method"),
        ({"method": "goodman", "compressive_mean": "absolute"}, "compressive_mean"),
        ({"method": "goodman", "uts_Pa": -1.0, "strength_source": "x"}, "positive"),
        (
            {"method": "walker", "walker_gamma": 1.5, "walker_source": "x"},
            "walker_gamma",
        ),
        ({"method": "dnv_base", "reference_mean_stress_Pa": 1e8}, "reference mean"),
        (
            {
                "method": "goodman",
                "uts_Pa": 6e8,
                "strength_source": "x",
                "reference_mean_stress_Pa": math.inf,
            },
            "reference_mean_stress_Pa must be finite",
        ),
    ],
)
def test_mean_stress_checks(kwargs, match):
    with pytest.raises(CableInputError, match=match):
        MeanStressSpec(**kwargs)


def test_descriptions():
    soderberg = MeanStressSpec("soderberg", yield_Pa=4e8, strength_source="handbook")
    assert "yield 400 MPa (handbook)" in soderberg.describe()
    walker = MeanStressSpec("walker", walker_gamma=0.6, walker_source="fit")
    assert "gamma 0.6 (fit)" in walker.describe()
    at_mean = MeanStressSpec(
        "goodman", uts_Pa=6e8, strength_source="x", reference_mean_stress_Pa=1e8
    )
    assert "curve at mean 100 MPa" in at_mean.describe()
    assert MeanStressSpec("dnv_base").describe().startswith("DNV-RP-C203 Eq. (2.15)")


def test_equivalent_range_checks_and_edges():
    spec = MeanStressSpec("goodman", uts_Pa=6e8, strength_source="x")
    with pytest.raises(CableInputError, match="same shape"):
        equivalent_range(np.ones(2), np.ones(3), spec)
    with pytest.raises(CableInputError, match="nonnegative"):
        equivalent_range(np.array([-1.0]), np.array([0.0]), spec)
    with pytest.raises(CableInputError, match="curve_R"):
        equivalent_range(np.array([1e8]), np.array([0.0]), spec, curve_R=1.0)
    # A curve measured at a mean stress at or above the strength.
    tested_at_strength = MeanStressSpec(
        "goodman", uts_Pa=6e8, strength_source="x", reference_mean_stress_Pa=7e8
    )
    with pytest.raises(CableInputError, match="reaches the strength"):
        equivalent_range(np.array([1e8]), np.array([0.0]), tested_at_strength)
    walker = MeanStressSpec("walker", walker_gamma=0.5, walker_source="SWT")
    zero = equivalent_range(
        np.array([0.0, 0.0]), np.array([1e8, 0.0]), walker, curve_R=0.1
    )
    np.testing.assert_array_equal(zero, 0.0)
