# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Physical bounds apply to direct model construction as well as YAML loading."""

import math

import pytest
from pydantic import ValidationError

from cablexsec.cables import schema as s

LAYERS = [
    s.HelicalLayer,
    s.HelicalStrandedLayer,
    s.CylindricalLayer,
    s.CylindricalOffsetLayer,
    s.CylindricalShellLayer,
]
MODELS = LAYERS + [
    s.ElectricalProperties,
    s.EnvironmentProperties,
    s.InitialStateProperties,
    s.CableMeta,
    s.AggregateProperties,
]


def defaults(model):
    if model in LAYERS:
        return {"name": "fixture", "material": "galvanised_steel_armour"}
    return {"name": "fixture"} if model is s.CableMeta else {}


FLOAT_FIELDS = [
    (model, name)
    for model in MODELS
    for name, field in model.model_fields.items()
    if "float" in str(field.annotation)
]


@pytest.mark.parametrize("model,field", FLOAT_FIELDS)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_all_physical_float_fields_reject_nonfinite(model, field, value):
    with pytest.raises(ValidationError, match=field):
        model(**(defaults(model) | {field: value}))


POSITIVE_FIELDS = [
    (s.ElectricalProperties, "permittivity_relative"),
    (s.ElectricalProperties, "vec_exponent_n"),
    (s.ElectricalProperties, "thermal_b_k"),
    (s.ElectricalProperties, "field_threshold_kV_mm"),
    (s.EnvironmentProperties, "frequency_Hz"),
    (s.HelicalLayer, "n_wires"),
    (s.HelicalLayer, "d_wire_m"),
    (s.HelicalLayer, "r_helix_m"),
    *[
        (s.HelicalStrandedLayer, n)
        for n in ("n_conductors", "r_cable_m", "n_strands", "r_strand_m", "d_strand_m")
    ],
    (s.CylindricalLayer, "r_outer_m"),
    *[
        (s.CylindricalOffsetLayer, n)
        for n in (
            "n_cores",
            "r_outer_m",
            "conductor_area_mm2",
            "strand_diameter_mm",
            "n_strands",
        )
    ],
    *[(s.CylindricalShellLayer, n) for n in ("r_mid_m", "t_m", "f_corrugation", "scf")],
    *[(model, "termination_scf") for model in LAYERS],
    *[
        (s.CableMeta, n)
        for n in (
            "voltage_kV",
            "design_life_years",
            "DFF",
            "MBR_installation_m",
            "MBR_dynamic_m",
        )
    ],
    *[
        (s.AggregateProperties, n)
        for n in s.AggregateProperties.model_fields
        if not n.endswith("_tension_kN")
    ],
]


@pytest.mark.parametrize("model,field", POSITIVE_FIELDS)
def test_positive_physical_fields_reject_zero(model, field):
    with pytest.raises(ValidationError):
        model(**(defaults(model) | {field: 0}))


@pytest.mark.parametrize("field", ["EI_min_tension_kN", "GJ_tension_kN"])
def test_published_tensions_may_be_zero_but_not_negative(field):
    # A published stiffness may be stated at zero tension.
    assert getattr(s.AggregateProperties(**{field: 0.0}), field) == 0.0
    with pytest.raises(ValidationError, match=field):
        s.AggregateProperties(**{field: -1.0})


@pytest.mark.parametrize(
    "model,field",
    [
        *[
            (s.ElectricalProperties, n)
            for n in ("loss_factor_tan_delta", "vec_synergism_b", "mechanical_beta")
        ],
        (s.EnvironmentProperties, "T4_external_Km_W"),
        (s.InitialStateProperties, "external_pressure_Pa"),
        (s.HelicalLayer, "pretension_Pa"),
        (s.HelicalLayer, "winding_stress_Pa"),
        (s.CylindricalOffsetLayer, "r_offset_m"),
        (s.CylindricalOffsetLayer, "r_inner_m"),
        (s.CylindricalLayer, "r_inner_m"),
    ],
)
def test_nonnegative_fields_reject_negative(model, field):
    with pytest.raises(ValidationError, match=field):
        model(**(defaults(model) | {field: -1}))


@pytest.mark.parametrize(
    "model,field",
    [
        (s.HelicalLayer, "lay_angle_deg"),
        (s.HelicalStrandedLayer, "lay_angle_cable_deg"),
        (s.HelicalStrandedLayer, "lay_angle_strand_deg"),
        (s.CylindricalOffsetLayer, "strand_lay_angle_deg"),
    ],
)
def test_out_of_range_lay_angle(model, field):
    with pytest.raises(ValidationError, match=field):
        model(**(defaults(model) | {field: 91}))


@pytest.mark.parametrize("model", LAYERS + [s.CableMeta])
def test_blank_name_rejected(model):
    with pytest.raises(ValidationError, match="name"):
        model(**(defaults(model) | {"name": " "}))


def test_nested_helix_angle_conversion_preserves_missing_and_given_values():
    empty = s.HelicalStrandedLayer(**defaults(s.HelicalStrandedLayer))
    assert empty.lay_angle_cable_rad is None
    assert empty.lay_angle_strand_rad is None
    specified = empty.model_copy(
        update={"lay_angle_cable_deg": 30, "lay_angle_strand_deg": 45}
    )
    assert specified.lay_angle_cable_rad == pytest.approx(math.pi / 6)
    assert specified.lay_angle_strand_rad == pytest.approx(math.pi / 4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"external_pressure_Pa": 1},
        {"external_pressure_Pa": 1, "identification": "measured"},
        {"identification": "measured", "source": "test"},
    ],
)
def test_initial_pressure_needs_consistent_provenance(kwargs):
    with pytest.raises(ValidationError):
        s.InitialStateProperties(**kwargs)
