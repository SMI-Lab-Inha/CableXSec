# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Reader, series and coupling inputs that must not be read silently wrong.

- every import path of ``fatigue_from_series`` reads the cable's own
  ``materials:`` block for the thermal stress,
- a thermal history on a layer whose material lacks ``E_Pa`` or ``alpha_T_K``
  stops instead of dropping that layer's thermal stress,
- a data row whose field count differs from the header stops the read,
- column selectors match whole words, so ``x curvature`` never selects
  ``Max curvature``, and a magnitude heading is refused for a component,
- the tangent stiffness is the central difference at ``kappa +- dk``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import cablexsec as cx
import cablexsec.io as cio
from cablexsec.cables.schema import CableData
from cablexsec.coupling import xsec_stiffness
from cablexsec.exceptions import CableSchemaError
from cablexsec.io.cabledyn_components import read_cabledyn_station_table
from cablexsec.materials import library
from cablexsec.project import CableModel

ARMOUR = "galvanised_steel_armour"


# --------------------------------------------------------------------------- #
# Thermal stress of fatigue_from_series                                       #
# --------------------------------------------------------------------------- #
def _cable() -> CableData:
    return cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)


def _armour_block(
    key: str, *, alpha_factor: float = 1.0, drop: str | None = None
) -> dict[str, Any]:
    block: dict[str, Any] = {}
    for prop, (value, _source) in library.get_material(ARMOUR).items():
        if prop == drop:
            continue
        scale = alpha_factor if prop == "alpha_T_K" else 1.0
        block[prop] = {"value": float(value) * scale, "source": "test input"}
    return {key: block}


def _with_armour(key: str, **kwargs: Any) -> CableData:
    raw = _cable().model_dump(mode="json")
    raw["materials"] = _armour_block(key, **kwargs)
    for layer in raw["layers"]:
        if layer.get("material") == ARMOUR:
            layer["material"] = key
    return CableData.model_validate(raw)


def _thermal_series(*, temperature: bool = True) -> cx.LoadCaseSeries:
    t = np.arange(200) * 0.5
    k = 0.02 * np.sin(2 * np.pi * t / 8)
    return cx.LoadCaseSeries(
        time_s=t,
        curvature_x_1_m=k,
        curvature_y_1_m=0 * k,
        tension_N=0 * k + 1e4,
        tension_kind="true",
        axial_strain=1e-4 + 0 * k,
        axial_strain_kind="direct",
        temperature_C=20 + 30 * np.sin(2 * np.pi * t / 50) if temperature else None,
    )


def _peak(result: Any) -> float:
    return max(float(np.max(v)) for v in result.D_bulk_per_fibre.values())


def test_every_import_path_reads_the_cable_materials_for_thermal_stress() -> None:
    series = _thermal_series()
    library_damage = _peak(cx.fatigue_from_series(_cable(), series))
    override = _with_armour(ARMOUR, alpha_factor=3.0)
    top = _peak(cx.fatigue_from_series(override, series))
    sub = _peak(cio.fatigue_from_series(override, series))
    project = _peak(CableModel(override).fatigue(series))
    assert top > 10 * library_damage
    assert sub == top
    assert project == top


@pytest.mark.parametrize("missing", ["alpha_T_K", "E_Pa"])
def test_missing_thermal_property_is_named_not_dropped(missing: str) -> None:
    cable = _with_armour("trial_steel", drop=missing)
    with pytest.raises(CableSchemaError, match=rf"trial_steel.*{missing}"):
        cx.fatigue_from_series(cable, _thermal_series())


def test_missing_thermal_property_is_harmless_without_temperature() -> None:
    cable = _with_armour("trial_steel", drop="alpha_T_K")
    result = cx.fatigue_from_series(cable, _thermal_series(temperature=False))
    assert _peak(result) > 0.0


# --------------------------------------------------------------------------- #
# Row length against the header                                               #
# --------------------------------------------------------------------------- #
def test_csv_row_with_an_extra_field_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "a.csv"
    path.write_text(
        "Time,x curvature,y curvature,Wall tension\n"
        "0,0.1,0.2,1,234.5\n"
        "0.1,0.1,0.2,1234.5\n"
    )
    with pytest.raises(CableSchemaError, match=r"line 2.*5 columns.*4"):
        cx.read_time_history_csv(path)


def test_csv_row_with_a_missing_field_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "a.csv"
    path.write_text(
        "Time,x curvature,y curvature,Wall tension,Other\n"
        "\n"
        "0,0.1,0.2,1,7\n"
        "0.1,0.1,0.2,1\n"
    )
    with pytest.raises(CableSchemaError, match=r"line 4"):
        cx.read_time_history_csv(path)


def test_station_table_header_split_by_a_unit_is_refused(tmp_path: Path) -> None:
    rows = "\n".join(
        f"{0.1 * i} {1000 + i} {0.01 * (i + 1)} {-0.02 * (i + 1)} {99.0 + i}"
        for i in range(4)
    )
    path = tmp_path / "s.txt"
    path.write_text("# station 3\ntime Ten (N) CurvX CurvY Torq\n" + rows + "\n")
    with pytest.raises(CableSchemaError, match=r"line 3.*5 columns.*6"):
        read_cabledyn_station_table(path)


def test_station_table_short_header_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "s.txt"
    path.write_text(
        "time Ten CurvX CurvY\n"
        + "\n".join(f"{0.1 * i} 5 {1000 + i} 0.01 0.02" for i in range(4))
        + "\n"
    )
    with pytest.raises(CableSchemaError, match=r"line 2"):
        read_cabledyn_station_table(path)


def test_station_table_with_units_and_comments_still_reads(tmp_path: Path) -> None:
    path = tmp_path / "s.txt"
    path.write_text(
        "# header follows\n"
        "time(s) Ten(N) CurvX(1/m) CurvY(1/m) Torq\n"
        "\n"
        "0.0 1000 0.01 -0.02 5\n"
        "# mid comment\n"
        "0.1 1001 0.02 -0.04 5\n"
        "0.2 1002 0.03 -0.06 5\n"
    )
    series = read_cabledyn_station_table(path)
    np.testing.assert_allclose(series.curvature_x_1_m, [-0.01, -0.02, -0.03])
    np.testing.assert_allclose(series.curvature_y_1_m, [0.02, 0.04, 0.06])


# --------------------------------------------------------------------------- #
# Whole-word column selection                                                 #
# --------------------------------------------------------------------------- #
def test_x_curvature_selector_does_not_match_max_curvature(tmp_path: Path) -> None:
    path = tmp_path / "a.csv"
    path.write_text(
        "Time (s),Max curvature (rad/m),y curvature (rad/m),Wall tension (kN)\n"
        "0,0.05,0.01,10\n0.1,0.06,-0.02,11\n0.2,0.04,0.03,12\n"
    )
    with pytest.raises(CableSchemaError, match="not found"):
        cx.read_time_history_csv(path)


def test_x_curvature_is_selected_beside_a_magnitude_column(tmp_path: Path) -> None:
    path = tmp_path / "b.csv"
    path.write_text(
        "Time (s),x curvature (rad/m),y curvature (rad/m),Max curvature (rad/m),"
        "Wall tension (kN)\n0,0.05,0.01,0.05,10\n0.1,0.06,-0.02,0.07,11\n"
    )
    series = cx.read_time_history_csv(path)
    np.testing.assert_allclose(series.curvature_x_1_m, [-0.01, 0.02])
    np.testing.assert_allclose(series.curvature_y_1_m, [0.05, 0.06])


@pytest.mark.parametrize(
    "heading", ["Max curvature", "Resultant curvature", "Curvature magnitude"]
)
def test_magnitude_heading_is_refused_for_a_component(
    tmp_path: Path, heading: str
) -> None:
    path = tmp_path / "c.csv"
    path.write_text(
        f"Time,{heading},y curvature,Wall tension\n0,0.05,0.01,10\n0.1,0.06,0.02,11\n"
    )
    for convention in ("rotation", "strain_gradient"):
        with pytest.raises(CableSchemaError, match="magnitude"):
            cx.read_time_history_csv(
                path, curvature_x_col=heading, curvature_convention=convention
            )


def test_exact_heading_is_preferred_and_whitespace_is_normalised(
    tmp_path: Path,
) -> None:
    path = tmp_path / "d.csv"
    path.write_text(
        " Time ,  X   Curvature ,Max x curvature, Y Curvature , WALL TENSION \n"
        "0,0.1,0.5,0.2,1\n0.1,0.1,0.5,0.2,1\n"
    )
    series = cx.read_time_history_csv(path, curvature_x_col="x  curvature")
    np.testing.assert_allclose(series.curvature_x_1_m, [-0.2, -0.2])
    np.testing.assert_allclose(series.curvature_y_1_m, [0.1, 0.1])


def test_documented_defaults_read_the_demo_history() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "validation" / "paper_data" / "global_demo" / "global_history_si.csv"
    series = cx.read_time_history_csv(
        path,
        time_col="time_s",
        curvature_x_col="curvature_x_1_m",
        curvature_y_col="curvature_y_1_m",
        tension_col="effective_tension_N",
        tension_kind="effective",
        curvature_convention="strain_gradient",
        tension_scale_to_N=1.0,
    )
    assert series.n_steps > 10


# --------------------------------------------------------------------------- #
# Tangent stiffness at kappa +- dk                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kappa", "dk", "n_ramp"),
    [(0.005, 1.0e-4, 400), (0.005, 1.0e-4, 8), (0.007, 1.0e-4, 400), (0.007, 1e-5, 8)],
)
def test_tangent_is_the_central_difference_at_kappa_plus_minus_dk(
    kappa: float, dk: float, n_ramp: int
) -> None:
    cable = _cable()
    tension = 20.0e3
    step = dk / 20.0
    n = round((kappa + dk) / step)
    fine = np.linspace(0.0, kappa + dk, n + 1)
    moment = cx.compute_mkappa(cable, fine, tension_N=tension).moment_Nm
    j = round((kappa - dk) / step)
    assert fine[j] == pytest.approx(kappa - dk, abs=1e-12)
    reference = (moment[-1] - moment[j]) / (fine[-1] - fine[j])
    result = xsec_stiffness(
        cable, kappa, tension_N=tension, delta_kappa_1_m=dk, n_ramp=n_ramp
    )
    assert result.EI_tangent_Nm2 == pytest.approx(reference, rel=0.01)


def test_mechanical_equivalent_keeps_every_block_of_the_definition():
    from cablexsec.cable_adapters import mechanical_equivalent
    from cablexsec.cables.schema import CylindricalShellLayer

    path = cx.export_three_core_yaml_path()
    cable = cx.load_cable(path, geometry_override=None)
    if not any(isinstance(x, CylindricalShellLayer) for x in cable.layers):
        pytest.skip("template without a metallic shell")
    equivalent = mechanical_equivalent(cable)
    assert equivalent is not cable
    for field in (
        "materials",
        "water_ingress",
        "environment",
        "initial_state",
        "core_lay",
    ):
        assert getattr(equivalent, field) == getattr(cable, field)
    assert not any(isinstance(x, CylindricalShellLayer) for x in equivalent.layers)
