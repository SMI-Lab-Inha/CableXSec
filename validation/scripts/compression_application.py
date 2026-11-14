# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Apply the armour compression screens to the bundled cables.

Run as ``python validation/scripts/compression_application.py``. Prints, for every
bundled cable, the per-layer lateral capacity, the radial stack capacity and
the cable compression capacity with the DNV-RP-F401 utilisation factors.

With ``--local-response DIR`` (a CableXSec ``local_response`` directory of
a global-to-local recovery, read only) and ``--cable KEY``, it also screens the effective
tension minima of that run: the arc-wise minimum from ``arc_response.csv``
and, at each recovered site, the minimum effective and true tension together
with the largest curvature of the record. The global model must use the
named cable's section; the script checks the outer diameter.

Assumptions stated in the output: the outer-sheath hoop strength for the
hoop-capacity failure ``P_3`` is the HDPE yield stress 20.74 MPa of Malta et
al. (2022) Table 4 (a public flexible-pipe sheath value, not the cables' own
grade); the annulus is flooded unless ``--annulus sealed``.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np

import cablexsec as cx
from cablexsec.analysis.compression import CompressionScreenResult

SHEATH_STRENGTH_PA = 20.74e6
SHEATH_SOURCE = (
    "ASSUMED: HDPE sheath yield 20.74 MPa, Malta et al. (2022) Table 4 "
    "(flexible-pipe outer sheath), not the cable's own grade"
)
CABLES = {
    "iea_task49_66kv": cx.iea_task49_66kv_yaml_path,
    "menard2023": cx.menard2023_yaml_path,
    "inter_array_single_core_33kv": cx.inter_array_single_core_yaml_path,
    "inter_array_three_core_33kv": cx.inter_array_three_core_yaml_path,
    "export_three_core_220kv": cx.export_three_core_yaml_path,
    "cigre_tb880_220kv": cx.cigre_tb880_yaml_path,
}


def screen(
    cable: cx.CableData,
    tension_N: float,
    curvature_1_m: float | None,
    *,
    annulus: str = "flooded",
    p_ext_Pa: float = 0.0,
    eta: float = 1.0,
) -> CompressionScreenResult:
    return cx.compression_screen(
        cable,
        tension_N,
        curvature_1_m,
        annulus=annulus,  # type: ignore[arg-type]
        p_ext_Pa=p_ext_Pa,
        sheath_hoop_strength_Pa=SHEATH_STRENGTH_PA,
        sheath_strength_source=SHEATH_SOURCE,
        utilisation_factor=eta,
    )


def capacity_table() -> list[dict[str, object]]:
    """Capacities of every bundled cable with lateral slip active (no curvature)."""
    rows: list[dict[str, object]] = []
    for key, path in CABLES.items():
        cable = cx.load_cable(path(), geometry_override=None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", cx.ExtrapolationWarning)
            res = screen(cable, 0.0, None)
        rows.append(
            {
                "cable": key,
                "lay_angles_deg": [
                    round(math.degrees(lyr.lay_angle_rad or 0.0), 1)
                    for lyr in cable.helical_layers
                ],
                "P1_kN": [round(lat.P1_saevik_N / 1e3, 2) for lat in res.lateral],
                "lateral_capacity_kN": [
                    round(lat.capacity_N / 1e3, 2) for lat in res.lateral
                ],
                "P2_kN": res.radial.P2_N / 1e3,
                "P3_kN": res.radial.P3_N / 1e3,
                "P4_kN": res.radial.P4_N / 1e3,
                "P2_star_kN": res.radial.P2_star_N / 1e3,
                "armour_share": res.armour_share,
                "capacity_kN": res.compression_capacity_N / 1e3,
                "governing_at_capacity": min(
                    [
                        (lat.capacity_N / lat.load_share, f"lateral:{lat.layer}")
                        for lat in res.lateral
                    ]
                    + [(res.radial.P2_star_N / res.armour_share, "radial")]
                )[1],
                "allowable_normal_kN": 0.67 * res.compression_capacity_N / 1e3,
                "allowable_installation_kN": 0.78 * res.compression_capacity_N / 1e3,
            }
        )
    return rows


def _history(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    cols = [str(c) for c in data["columns"]]
    values = data["data"]
    return {c: values[:, i] for i, c in enumerate(cols)}


def local_response_margins(
    directory: Path, cable_key: str, annulus: str, eta: float
) -> dict[str, object]:
    """Screen the tension minima of one local-response record."""
    cable = cx.load_cable(CABLES[cable_key](), geometry_override=None)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    od = float(summary["matched_section_properties_SI"]["outer_diameter_m"])
    r_out = max(getattr(lyr, "r_outer_m", 0.0) or 0.0 for lyr in cable.layers)
    if not math.isclose(od, 2.0 * r_out, rel_tol=0.02):
        raise SystemExit(
            f"global section OD {od:.4f} m does not match {cable_key} ({2 * r_out:.4f} m)"
        )
    arc = np.genfromtxt(directory / "arc_response.csv", delimiter=",", names=True)
    i_min = int(np.argmin(arc["effective_tension_kN_min"]))
    out: dict[str, object] = {
        "cable": cable_key,
        "annulus": annulus,
        "utilisation_factor": eta,
        "arc_min_effective_tension_kN": float(arc["effective_tension_kN_min"][i_min]),
        "arc_min_at_m": float(arc["arc_length_m"][i_min]),
        "arc_points_in_compression": int(np.sum(arc["effective_tension_kN_min"] < 0.0)),
    }
    sites = []
    for path in sorted(directory.glob("*_history.npz")):
        h = _history(path)
        kappa = np.hypot(h["strain_gradient_x_1_m"], h["strain_gradient_y_1_m"])
        te = float(np.min(h["effective_tension_N"]))
        tw = float(np.min(h["true_tension_N"]))
        p_ext = float(np.max(h["external_pressure_Pa"]))
        k_max = float(np.max(kappa))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", cx.ExtrapolationWarning)
            res = screen(cable, te, k_max, annulus=annulus, p_ext_Pa=p_ext, eta=eta)
        sites.append(
            {
                "site": path.name.replace("_history.npz", ""),
                "min_effective_tension_kN": te / 1e3,
                "min_true_tension_kN": tw / 1e3,
                "max_external_pressure_MPa": p_ext / 1e6,
                "max_curvature_1_m": k_max,
                "driving_tension_kN": res.driving_tension_N / 1e3,
                "utilisation": res.utilisation,
                "design_utilisation": res.design_utilisation,
                "governing": res.governing,
                "capacity_kN": res.compression_capacity_N / 1e3,
                "allowable_kN": res.allowable_compression_N / 1e3,
                "margin_kN": res.margin_N / 1e3,
            }
        )
    out["sites"] = sites
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--local-response", type=Path, default=None)
    parser.add_argument("--cable", choices=sorted(CABLES), default="menard2023")
    parser.add_argument("--annulus", choices=("flooded", "sealed"), default="flooded")
    parser.add_argument("--eta", type=float, default=0.67)
    args = parser.parse_args()
    print(f"P3 assumption: {SHEATH_SOURCE}")
    for row in capacity_table():
        print(json.dumps(row, default=float))
    if args.local_response is not None:
        res = local_response_margins(
            args.local_response, args.cable, args.annulus, args.eta
        )
        print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
