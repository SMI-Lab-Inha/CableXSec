# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""A14 — independent-method stiffness cross-check (Witz-Tan vs Raoof-Huang).

Validates the frozen stiffness engine against the INDEPENDENT orthotropic-sheet /
Hruska method of [RaoofHuang1992] (J. Eng. Mech. 118(12):2335-2351), distinct
from Witz-Tan. Cable: the fully-specified single-layer 16 mm steel spiral strand
from their Table 1/2 (n=12, D=3.25 mm, α=11.9°; Eq. 19 gives r≈6.4 mm → d≈16 mm).
The multi-layer 127 mm strand is NOT used — Table 1/2 give only its outer layer,
so building it would require gap-filling the inner layers.

Comparison (dimensionless, E_s cancels):
* AXIAL: our plane-section factor EA/(E_s A_T) = cos³α agrees with the
  Raoof-Huang **no-slip** axial factor (Eq. 11/12) to a few %.
* BENDING (plane-section): our EI_max agrees with the Raoof-Huang **full-slip**
  bending bound (Eqs. 14-20) to ~6-12%. NOTE both Raoof-Huang bounds assume
  plane sections (within-layer line-contact slip only), so they map to OUR
  EI_max regime; our EI_min (individual-wire bending, full inter-layer slip) is
  a LOWER bound Raoof-Huang does not model — a genuine slip-definition
  difference, not an engine error.

Gate: few-% (axial) / ~10% (bending) agreement — two different approximations
bracket the same plane-section stiffness, i.e. method-independence. Eq. 11 cubic
coefficient is OCR-ambiguous (6.51 vs 6.57); the result is robust to both.
"""

from __future__ import annotations

import math

import pytest

from cablexsec.cables.schema import CableData, CableMeta
from cablexsec.materials.library import get_property
from cablexsec.mechanics.stiffness import ea_fixed, ei_max_stuck, ei_min_slip

_N, _D, _ALPHA_DEG = 12, 3.25e-3, 11.9


# Test-taxonomy category.
pytestmark = pytest.mark.independent_method


def _strand() -> CableData:
    from cablexsec.cables.schema import _LAYER_ADAPTER  # noqa: PLC0415

    a = math.radians(_ALPHA_DEG)
    # Eq. 19 Phillips-Costello helix radius (single layer).
    r = (_D / 2.0) * math.sqrt(
        1.0 + math.tan(math.pi / 2 - math.pi / _N) ** 2 / math.cos(a) ** 2
    )
    layer = _LAYER_ADAPTER.validate_python(
        {
            "name": "strand_layer",
            "layer_type": "helical",
            "material": "galvanised_steel_armour",
            "n_wires": _N,
            "d_wire_m": _D,
            "r_helix_m": r,
            "lay_angle_deg": _ALPHA_DEG,
        }
    )
    return CableData(cable=CableMeta(name="16 mm spiral strand"), layers=(layer,))


def _raoof_huang_bounds(c2: float) -> dict[str, float]:
    """Raoof-Huang (1992) full-slip/no-slip axial & bending factors (/E_s)."""
    a = math.radians(_ALPHA_DEG)
    ca = math.cos(a)
    r = (_D / 2.0) * math.sqrt(1.0 + math.tan(math.pi / 2 - math.pi / _N) ** 2 / ca**2)
    d = 2 * r + _D
    h = ca**4  # Eq. 9 / Eq. 20 (single layer)
    e_fs = -0.264 - 2.00 * h + c2 * h**2 - 3.31 * h**3  # Eq. 11
    gamma = 4.00 - 7.92 * e_fs + 7.24 * e_fs**2 - 2.32 * e_fs**3  # Eq. 12a
    e_ns = gamma * e_fs
    i_gross = (math.pi / 4) * (math.pi * d**4 / 64)  # Eq. 14
    return {
        "axial_fs": e_fs,
        "axial_ns": e_ns,
        "ei_fs": e_fs * i_gross,  # /E_s [m^4]
        "ei_ns": e_ns * i_gross,
    }


def test_A14_axial_matches_raoof_huang_no_slip_within_5pct() -> None:
    cable = _strand()
    es = get_property("galvanised_steel_armour", "E_Pa")
    a_t = _N * math.pi * _D**2 / 4.0
    our_axial = ea_fixed(cable) / (es * a_t)  # = cos^3 a
    for c2 in (6.51, 6.57):  # OCR-ambiguous cubic coefficient — robust to both
        rh = _raoof_huang_bounds(c2)
        rel = abs(our_axial - rh["axial_ns"]) / rh["axial_ns"]
        assert rel < 0.05, f"axial {our_axial:.4f} vs RH no-slip {rh['axial_ns']:.4f}"


def test_A14_EI_max_matches_raoof_huang_full_slip_plane_section() -> None:
    cable = _strand()
    es = get_property("galvanised_steel_armour", "E_Pa")
    our_ei_max = ei_max_stuck(cable, {"strand_layer": 1.0}) / es  # /E_s [m^4]
    for c2 in (6.51, 6.57):
        rh = _raoof_huang_bounds(c2)
        rel = abs(our_ei_max - rh["ei_fs"]) / rh["ei_fs"]
        assert rel < 0.13, f"EI_max {our_ei_max:.3e} vs RH fs {rh['ei_fs']:.3e}"


def test_A14_EI_min_is_a_lower_regime_not_modelled_by_raoof_huang() -> None:
    """Witz-Tan EI_min (full inter-layer slip) far below RH plane-section bounds."""
    cable = _strand()
    es = get_property("galvanised_steel_armour", "E_Pa")
    our_ei_min = ei_min_slip(cable) / es
    rh = _raoof_huang_bounds(6.51)
    assert our_ei_min < 0.1 * rh["ei_fs"]  # individual-wire bending << plane-section


def test_A14_mu_on_steel_cited_to_raoof_huang() -> None:
    """μ = 0.12 for galvanised steel is grounded to [RaoofHuang1992]."""
    from cablexsec.materials.library import get_citation, get_property  # noqa: PLC0415

    assert get_property("galvanised_steel_armour", "mu_on_steel") == pytest.approx(0.12)
    assert "RaoofHuang1992" in get_citation("galvanised_steel_armour", "mu_on_steel")
