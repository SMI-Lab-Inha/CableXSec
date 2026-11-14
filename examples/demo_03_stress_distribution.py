# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Demo 03 — per-wire all-component stress distribution.

Per-wire σ_A + σ_B + σ_f + σ_T + σ_vm for every structural helical layer at a
fixed tension and curvature. The implementation is analytically verified (σ_B closed
form, σ_f stick-slip boundary, moment recovery vs M-κ) — NOT experimentally; the
demo prints that provenance verbatim so no number is read more confidently than
its result object allows.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from _console import configure_console

import cablexsec as cx
from cablexsec.plotting.style import COLORS, apply_style

configure_console()

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)


def main() -> None:
    cable = cx.load_cable(
        cx.inter_array_single_core_yaml_path(), geometry_override=None
    )
    sr = cx.compute_stress(cable, kappa_x=0.02, tension_N=1.0e5)

    mpa = 1e6
    print("=== per-wire stress @ kappa=0.02 1/m, T=100 kN ===")
    for name in sr.phi_deg:
        print(
            f"  {name}: peak σ_total {max(abs(sr.sigma_total_Pa[name])) / mpa:6.1f} MPa, "
            f"peak σ_vm {max(sr.sigma_vm_Pa[name]) / mpa:6.1f} MPa, "
            f"slipping {int(sr.slipping[name].sum())}/{len(sr.slipping[name])}"
        )
    print(
        f"  critical wire: {sr.critical_layer} @ {sr.critical_phi_deg:.0f} deg, "
        f"σ_vm {sr.critical_sigma_vm_Pa / mpa:.1f} MPa"
    )
    print(f"  moment recovered (= M-kappa engine): {sr.moment_recovered_Nm:.1f} N*m")
    print("\n  PROVENANCE (validation is analytical, not experimental):")
    for lim in sr.provenance.limitations:
        print(f"    - {lim}")

    apply_style()
    name = sr.critical_layer
    phi = sr.phi_deg[name]
    fig, ax = plt.subplots(figsize=(6, 4))
    for comp, col, lab in [
        (sr.sigma_A_Pa, COLORS["GREEN"], r"$\sigma_A$"),
        (sr.sigma_B_Pa, COLORS["BLUE"], r"$\sigma_B$"),
        (sr.sigma_f_Pa, COLORS["ORANGE"], r"$\sigma_f$"),
        (sr.sigma_total_Pa, COLORS["BLACK"], r"$\sigma_{total}$"),
    ]:
        ax.plot(phi, comp[name] / mpa, color=col, label=lab)
    ax.set_xlabel(r"wire azimuth $\varphi$ [deg]")
    ax.set_ylabel("stress [MPa]")
    ax.set_title(f"Per-wire stress — {name} (analytically verified)")
    ax.legend(ncol=2)
    fig.savefig(OUT / "demo_03_stress_distribution.png")
    print(f"\nsaved {OUT / 'demo_03_stress_distribution.png'}")


if __name__ == "__main__":
    main()
