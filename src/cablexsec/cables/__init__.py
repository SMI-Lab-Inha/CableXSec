# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Cable definitions: YAML schema, loader, and packaged reference cables."""

from __future__ import annotations

import math
from importlib import resources
from pathlib import Path
from typing import Literal

from cablexsec.cables.schema import CableData, CoreStrandRing, load_cable

__all__ = [
    "CableData",
    "chae2023_yaml_path",
    "poradowski2025_220kv_yaml_path",
    "export_three_core_yaml_path",
    "iea_task49_66kv_300_yaml_path",
    "iea_task49_66kv_stranded",
    "iea_task49_66kv_yaml_path",
    "inter_array_single_core_legacy_r2_yaml_path",
    "inter_array_single_core_yaml_path",
    "inter_array_three_core_legacy_r2_yaml_path",
    "inter_array_three_core_yaml_path",
    "load_cable",
    "menard2023_yaml_path",
    "saevik2011_yaml_path",
    "thies2016_yaml_path",
    "witz1996_yaml_path",
]


def _packaged(name: str) -> Path:
    return Path(str(resources.files("cablexsec.cables") / name))


def inter_array_single_core_yaml_path() -> Path:
    """Path to the public single-core inter-array 33 kV template (solid core)."""
    return _packaged("inter_array_single_core_33kv.yaml")


def inter_array_three_core_yaml_path() -> Path:
    """Path to the public three-core inter-array 33 kV template."""
    return _packaged("inter_array_three_core_33kv.yaml")


# Pinned copies of the 33 kV examples as they stood before the geometry
# correction of 18 October 2026 (armour fill above 1, undeclared screens).
# They are kept to reproduce earlier results; tests pin their SHA-256. Do not
# edit; new work uses the corrected files.
def inter_array_single_core_legacy_r2_yaml_path() -> Path:
    """Pinned pre-correction 33 kV single-core example (earlier results)."""
    return _packaged("inter_array_single_core_33kv_legacy_r2.yaml")


def inter_array_three_core_legacy_r2_yaml_path() -> Path:
    """Pinned pre-correction 33 kV three-core example (earlier results)."""
    return _packaged("inter_array_three_core_33kv_legacy_r2.yaml")


def chae2023_yaml_path() -> Path:
    """Path to the artificial coiling cable of Chae, Nam and Shim (2023).

    Single-armour three-core cable with equivalent cores (Ocean Eng. 288,
    116014), armour pitch 1.8 m; the public model-to-model case of the
    coiling request (:func:`cablexsec.compute_coiling`).
    """
    return _packaged("chae2023_coiling.yaml")


def export_three_core_yaml_path() -> Path:
    """Path to the public three-core export 220 kV template (corrugated-Al sheath)."""
    return _packaged("export_three_core_220kv.yaml")


def iea_task49_66kv_yaml_path() -> Path:
    """Path to the public 66 kV three-core wet-design template.

    Janocha et al. (2024) 66 kV cable, the 630 mm2 reference cable of the
    IEA Wind Task 49 design basis; provenance in ``doc/template_66kv.md``.
    """
    return _packaged("iea_task49_66kv.yaml")


def iea_task49_66kv_300_yaml_path() -> Path:
    """Path to the 66 kV three-core 300 mm2 section derived from the 630 mm2 template.

    The IEA Wind Task 49 design basis (Hall et al. 2024, Table 13) gives only
    aggregate properties for its 300 mm2 row; this section keeps the layer
    thicknesses, lay angles, materials and armour coverage of the Janocha et al.
    (2024) template and scales the conductor. Every derived dimension is an
    assumption, listed in the file header.
    """
    return _packaged("iea_task49_66kv_300.yaml")


#: Envelope of the 66 kV conductor, 29.90 mm ([Janocha2024] Table 1), within
#: the 28.7-32.5 mm of a compacted 630 mm^2 Class 2 conductor (IEC 60228:2004).
_IEA66_CONDUCTOR_OD_M = 29.90e-3
#: 61 wires, 1 + 6 + 12 + 18 + 24 ([Janocha2024] Fig. 4a).
_IEA66_RINGS = (1, 6, 12, 18, 24)


def iea_task49_66kv_stranded(
    *,
    lay_pattern: Literal["equal_ratio", "equal_pitch"] = "equal_ratio",
    outer_lay_ratio: float = 12.0,
    outer_lay: Literal["left_hand", "right_hand"] = "left_hand",
    compaction: Literal["none", "light", "heavy"] = "none",
) -> CableData:
    """The 66 kV template with a designed, compacted 61-wire conductor.

    The source publishes the wire count (61, Fig. 4a), the envelope (29.90 mm,
    Table 1) and the area (630 mm^2), not the pitches. The design here:

    * five rings 1 + 6 + 12 + 18 + 24, compacted to the envelope: each ring
      has the radial thickness ``t = 29.90 / 9`` mm and the helix radius
      ``i t``; every wire carries ``630 / 61`` mm^2 of copper (fill 0.90 of
      the envelope; compaction coefficients 0.84-0.91 in Zhang et al. 2025);
    * lay ratio (pitch over the ring's outer diameter ``(2i + 1) t``):
      ``"equal_ratio"`` gives every ring ``outer_lay_ratio`` (Smith et al.
      2026: 10.5-12.3 on a compacted dynamic-cable conductor);
      ``"equal_pitch"`` gives every ring the pitch of the outer ring
      (Nasution et al. 2013, 2014; Nam et al. 2024: nearly equal pitches,
      outer lay ratios 10-13). The published outer lay ratios span about
      10-16 (Jiang et al. 2022 state 16), the range to sweep;
    * lay directions alternate, the outer ring ``outer_lay`` (assumed);
    * ``compaction="light"`` or ``"heavy"`` gives each ring the stress
      concentration of its compaction indentations (Jiang et al. 2022,
      Table 1: centre 1.0, inner rings 1.019/1.026, outer ring 1.129/1.202)
      and the wire curve corrected for them (``nasution_2013_copper_corrected``,
      gap CF3); ``"none"`` keeps the outer-wire curve without factors.

    The conductor is then a resolved strand bundle, which the coupled radial
    solvers reject ("a resolved strand bundle needs a strand radial law");
    use it for :func:`cablexsec.compute_conductor_fatigue` and the bounded
    contact paths. The conductor screen starts at the envelope. Bulk curve:
    the wire curve of a compacted 300 mm^2 conductor
    (``nasution_2014_copper``), the nearest public compacted conductor wire.
    """
    if lay_pattern not in ("equal_ratio", "equal_pitch"):
        raise ValueError("lay_pattern must be 'equal_ratio' or 'equal_pitch'")
    if not (math.isfinite(outer_lay_ratio) and outer_lay_ratio > 0):
        raise ValueError("outer_lay_ratio must be finite and positive")
    factors = {"none": (1.0, 1.0), "light": (1.019, 1.129), "heavy": (1.026, 1.202)}
    if compaction not in factors:
        raise ValueError("compaction must be 'none', 'light' or 'heavy'")
    inner_scf, outer_scf = factors[compaction]
    cable = load_cable(iea_task49_66kv_yaml_path())
    t = _IEA66_CONDUCTOR_OD_M / 9.0
    wire_area = 630.0e-6 / sum(_IEA66_RINGS)
    outer_pitch = outer_lay_ratio * _IEA66_CONDUCTOR_OD_M
    other: Literal["left_hand", "right_hand"] = (
        "right_hand" if outer_lay == "left_hand" else "left_hand"
    )
    n_rings = len(_IEA66_RINGS)
    rings = [
        CoreStrandRing(n_wires=1, d_wire_m=t, r_helix_m=0.0, wire_area_m2=wire_area)
    ]
    for i, n in enumerate(_IEA66_RINGS[1:], start=1):
        pitch = (
            outer_lay_ratio * (2 * i + 1) * t
            if lay_pattern == "equal_ratio"
            else outer_pitch
        )
        rings.append(
            CoreStrandRing(
                n_wires=n,
                d_wire_m=t,
                r_helix_m=i * t,
                pitch_m=pitch,
                lay_direction=outer_lay if (n_rings - 1 - i) % 2 == 0 else other,
                wire_area_m2=wire_area,
                compaction_scf=outer_scf if i == n_rings - 1 else inner_scf,
            )
        )
    r_env = _IEA66_CONDUCTOR_OD_M / 2.0
    layers = []
    for layer in cable.layers:
        if layer.name == "cu_conductor":
            layer = layer.model_copy(
                update={
                    "r_outer_m": r_env,
                    "strand_layers": tuple(rings),
                    "construction": "resolved_concentric_stranded",
                    "bulk_sn": (
                        "nasution_2014_copper"
                        if compaction == "none"
                        else "nasution_2013_copper_corrected"
                    ),
                    "notes": (
                        "Designed compacted 61-wire conductor "
                        f"({lay_pattern}, outer lay ratio {outer_lay_ratio:g}); "
                        "see iea_task49_66kv_stranded"
                    ),
                }
            )
        elif layer.name == "conductor_screen":
            layer = layer.model_copy(update={"r_inner_m": r_env})
        layers.append(layer)
    return CableData.model_validate(
        cable.model_copy(update={"layers": tuple(layers)}).model_dump()
    )


def poradowski2025_220kv_yaml_path() -> Path:
    """Return the path to the packaged Poradowski & Anders (2025) 220 kV cable.

    A fully public export-cable definition (Ocean Eng. 328, 121027, Table 1)
    as a single-core-equivalent stack, used for the export demonstration and
    benchmark BM5.
    """
    return Path(str(resources.files("cablexsec.cables") / "poradowski2025_220kv.yaml"))


def menard2023_yaml_path() -> Path:
    """Return the path to the packaged Ménard & Cartraud (2023) cable (BM1).

    A fully public three-core dynamic-cable definition (Marine Structures 91,
    Table 1) used for benchmark BM1 (calibration-free EI_min validation).
    """
    return Path(str(resources.files("cablexsec.cables") / "menard2023.yaml"))


def witz1996_yaml_path(wires: str = "round") -> Path:
    """Return the path to the packaged Witz (1996) 2.5-inch flexible pipe.

    The public ISSC case-study pipe (Marine Structures 9, Table 2; Ramos and
    Pesce 2004, Table 3) used for benchmarks WITZ-EI and WITZ-ID. Its initial
    external pressure is identified on the 0-bar loop (``identification:
    identified``). ``wires="round"``: area-equivalent round wires (the rows
    on record); ``"rectangular"``: the 6 x 3 mm armour strips and the
    profiled carcass and Zeta sections of Witz Table 2 (gap G1).
    """
    if wires == "round":
        return _packaged("witz1996.yaml")
    if wires == "rectangular":
        return _packaged("witz1996_rect.yaml")
    raise ValueError("wires must be 'round' or 'rectangular'")


def saevik2011_yaml_path(tendons: str = "round") -> Path:
    """Return the path to the packaged Saevik (2011) 8-inch FBG flexible riser.

    The public riser of Computers and Structures 89, Table 1 (and Gjosteen et
    al. 2027, Table 7) used for the SV11 rows: measured inner-armour stress
    ranges at fibre Bragg grating stations near a bellmouth.
    ``tendons="round"``: area-equivalent round wires (the rows on record);
    ``"rectangular"``: the 5 x 12.5 mm tendons (gap G1). The polymer modulus
    is identified on the published bending stiffness for each.
    """
    if tendons == "round":
        return _packaged("saevik2011.yaml")
    if tendons == "rectangular":
        return _packaged("saevik2011_rect.yaml")
    raise ValueError("tendons must be 'round' or 'rectangular'")


def thies2016_yaml_path() -> Path:
    """Return the path to the packaged Thies et al. (2016) example cable.

    The 18/30 kV three-core static cable of the bend-restrictor test
    (Int. J. Marine Energy 16), reconstructed from public data with stated
    bounds: per-core copper wire screen (gap P1), core slip state (P8) and
    bonded Al-PE laminate (P6). An example, not a manufacturer's design.
    """
    return _packaged("thies2016_30kv.yaml")
