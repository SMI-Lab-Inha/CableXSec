# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for the material property library."""

from __future__ import annotations

import pytest

from cablexsec.materials import library

# Phase 1 materials that must always be present.
PHASE1_MATERIALS = [
    "galvanised_steel_armour",
    "copper_stranded",
    "xlpe_qiao2025",
    "hdpe",
    "pp_filler",
]


# Test-taxonomy category.
pytestmark = pytest.mark.analytical_sanity


def test_phase1_materials_present() -> None:
    available = library.list_materials()
    for name in PHASE1_MATERIALS:
        assert name in available


def test_every_entry_is_value_citation_pair() -> None:
    """Each property must be a (float, non-empty citation) pair."""
    for name in library.list_materials():
        for prop, entry in library.get_material(name).items():
            assert isinstance(entry, tuple) and len(entry) == 2, (name, prop)
            value, citation = entry
            assert isinstance(value, float), (name, prop, value)
            assert isinstance(citation, str) and citation.strip(), (name, prop)


def test_get_property_returns_value_only() -> None:
    assert library.get_property("galvanised_steel_armour", "E_Pa") == 210.0e9
    assert library.get_property("copper_stranded", "E_eff_axial_Pa") == 111.0e9
    assert library.get_property("xlpe_qiao2025", "nu") == 0.45


def test_get_citation_returns_source() -> None:
    assert library.get_citation("galvanised_steel_armour", "E_Pa") == (
        "[DNVGL-ST-0359] Table 4-1"
    )


def test_temperature_dependent_xlpe_modulus_decreases() -> None:
    """Qiao Eq. (9) values are recovered without extrapolation."""
    e20 = library.youngs_modulus_at_temperature("xlpe_qiao2025", 20.0)
    e50 = library.youngs_modulus_at_temperature("xlpe_qiao2025", 50.0)
    e80 = library.youngs_modulus_at_temperature("xlpe_qiao2025", 80.0)
    assert e20 == pytest.approx(150.0e6)
    assert e50 == pytest.approx(67.5e6)
    assert e80 == pytest.approx(33.6e6)
    assert e20 > e50 > e80


def test_temperature_dependent_hdpe_modulus_recovers_qiao_equation() -> None:
    assert library.youngs_modulus_at_temperature("hdpe", 20.0) == pytest.approx(885.6e6)
    assert library.youngs_modulus_at_temperature("hdpe", 90.0) == pytest.approx(150.6e6)


def test_temperature_modulus_provenance_names_equation_and_range() -> None:
    citation = library.get_temperature_modulus_citation("xlpe_qiao2025")
    assert "[Qiao2025] Eq. 9" in citation
    assert "10--90 degC" in citation


@pytest.mark.parametrize("temperature_c", [9.999, 90.001, float("nan")])
def test_temperature_modulus_rejects_unsupported_extrapolation(
    temperature_c: float,
) -> None:
    with pytest.raises(ValueError):
        library.youngs_modulus_at_temperature("xlpe_qiao2025", temperature_c)


def test_temperature_modulus_requires_a_sourced_model() -> None:
    with pytest.raises(KeyError, match="no sourced temperature-dependent"):
        library.youngs_modulus_at_temperature("galvanised_steel_armour", 20.0)


def test_unknown_material_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="Unknown material"):
        library.get_property("unobtanium", "E_Pa")


def test_unknown_property_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="has no property"):
        library.get_property("hdpe", "sigma_y_Pa")


def test_phase2_materials_present() -> None:
    for name in ("lead_sheath", "aluminium_corrugated"):
        assert name in library.list_materials()


def test_missing_friction_property_names_the_explicit_argument() -> None:
    with pytest.raises(KeyError, match=r"pass the coefficient explicitly \(mu="):
        library.get_property("aisi304_witz1996", "mu_on_hdpe")
    with pytest.raises(KeyError) as info:
        library.get_property("hdpe", "sigma_y_Pa")
    assert "friction" not in str(info.value)
