# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Demo 02 — M-κ hysteresis and the BM1 (Ménard 2023) EI_min comparison.

The full-slip EI_min of the public Ménard cable is compared without calibration
against the analytical value (3593 N·m²). W_diss / EI_max are not benchmarked for
this cable — the paper calibrates the initial contact pressure (P = 0.3 MPa),
which we do not import — so this demo states that limitation rather than
printing a W_diss "validation" it cannot back.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _console import configure_console

import cablexsec as cx
from cablexsec.plotting.style import COLORS, apply_style

configure_console()

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

_ANALYTICAL_EI_MIN = 3593.0  # [MenardCartraud2023] Table 3 (Witz-Tan full slip)


def main() -> None:
    cable = cx.load_cable(cx.menard2023_yaml_path(), geometry_override=None)
    stiff = cx.compute_stiffness(cable)
    err = abs(stiff.ei_min_Nm2 - _ANALYTICAL_EI_MIN) / _ANALYTICAL_EI_MIN

    print("=== BM1 - Menard & Cartraud (2023), calibration-free EI_min ===")
    print(f"  EI_min (CableXSec) = {stiff.ei_min_Nm2:.0f} N.m^2")
    print(f"  EI_min (analytic)  = {_ANALYTICAL_EI_MIN:.0f} N.m^2")
    print(f"  error              = {err * 100:.1f}%  (gate < 15%)")
    print("  LIMITATION: W_diss / EI_max are BLOCKED for this cable - the source")
    print("  calibrates the initial contact pressure (P=0.3 MPa); not imported.")

    # An illustrative M-κ loop (the engine is gated by A11 energy closure).
    kappa = 0.03 * np.sin(np.linspace(0.0, 6.0 * np.pi, 1200))
    mk = cx.compute_mkappa(cable, kappa, tension_N=0.0)
    print(
        f"\n  illustrative loop: EI_apparent {mk.EI_apparent_Nm2:.0f} N.m^2, "
        f"W_diss {mk.W_diss_J_per_m:.3e} J/m (engine gate: A11)"
    )

    apply_style()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(mk.kappa_1_m * 1e3, mk.moment_Nm, color=COLORS["BLACK"], label="CableXSec")
    ax.set_xlabel(r"curvature $\kappa$ [$10^{-3}$ m$^{-1}$]")
    ax.set_ylabel("moment [N·m]")
    ax.set_title("Ménard cable — M–κ loop (EI_min comparison, A11)")
    ax.legend()
    fig.savefig(OUT / "demo_02_mkappa.png")
    print(f"saved {OUT / 'demo_02_mkappa.png'}")


if __name__ == "__main__":
    main()
