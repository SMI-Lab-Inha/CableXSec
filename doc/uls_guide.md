<!-- SPDX-License-Identifier: Apache-2.0 -->

# Ultimate-limit-state guide: armour in compression

This guide covers the screens for tensile armour in axial compression:
lateral buckling and radial (bird-cage) buckling, their interaction with wire
yield, the design factor and the combined capacity curve. The tension–bending
first-yield envelope is described in
[local_analysis_guide.md](local_analysis_guide.md), Section 8.

Code: `cablexsec/analysis/compression.py`. Published-case check:
`python validation/scripts/check_compression_buckling.py`. Application to the bundled
cables and to a global record: `python validation/scripts/compression_application.py`.
Tests: `tests/test_compression_buckling.py`. Gates: [benchmarks.md](benchmarks.md),
section "Armour compression".

## 1. Where a floating-wind cable sees compression

Compression in the armour needs a negative driving tension. For a flooded
(wet-design) annulus that is the effective tension; for a sealed (dry) annulus
it is the true wall tension `T_w = T_e - p_e A_e` (Sævik and Thorsen 2017,
Eq. 1, with no bore). A power cable has no bore, so the reverse end-cap
compression of an empty flexible pipe does not arise.

Effective-tension minima of a dynamic cable lie where the line weight is
supported from below or the cable meets the seabed:

- the touchdown region, where vertical motion of the floater moves the
  touchdown point and the cable is cycled in bending at low tension (Li and
  Vaz 2021b, introduction; Reda 2020, Section 2.1, for installation);
- the sag bend and the ends of the buoyancy section of a lazy wave, at low
  mean tension and the largest dynamic curvature;
- installation and storage: lay-vessel heave at the touchdown point and the
  constant-compression and compression-with-bending tests of CIGRE TB 862
  (MecLim II, Section 5.7).

The hang-off and the bend-stiffener region carry the top tension and do not
normally go into compression. Lateral buckling also needs cyclic bending
(Sævik and Thorsen 2017, Section 4.2.1; Li and Vaz 2021a), which the
touchdown region and the sag bend supply.

## 2. Formulations

All forces are axial (cable-direction) forces of a whole layer or of the
armour stack, positive in compression. A layer of `n` wires at helix radius
`R` and lay angle `α` has wire bending rigidities `EI_2` (radial) and `EI_3`
(lateral) and torsion rigidity `GJ`; for round wires `EI_2 = EI_3 = EπD⁴/64`
and `GJ = EI/(1 + ν)`.

**Lateral buckling, lower bound** (Sævik and Thorsen 2017, Eq. 19; friction
free, buckle-length term omitted):

    P1 = n cos α / R² [GJ sin⁴α + (4 EI_2 + EI_3 − GJ) sin²α cos²α]

**Lateral buckling, round wires** (Li and Vaz 2021a, Eq. 11, after Li and Vaz
2019):

    P_cr = n EπD⁴ sin²(2α) cos α / (64 R²) · [1 + 2G cos²α / (E sin²α + 2G)]

reduced by the cyclic-curvature knock-down `η(κ_max R)` of the finite-element
study of Li and Vaz (2021b, Table 3): 0.91, 0.88 and 0.83 at
`κ_max R` = 0.0036, 0.0072 and 0.0107, linear in between, 0.91 held below and
0.83 held above (flagged). The layer capacity is `min(P1, η P_cr)`. For the
public cables P1 always governs, because P1 is 0.55–0.66 of `P_cr`.

**Friction gate** (Li and Vaz 2021a, Eqs. 12–15): the wires migrate laterally
only after they slip at the neutral axis. At the lateral capacity the
first-slip curvature is

    κ_slip = μ (Q sin²α / R + q_ext) / (E A sin α cos²α),  Q = P_lat / (n cos α)

with `q_ext = 2π r_o p_e cos α / n` for a sealed annulus and zero for a
flooded one. Below `κ_slip` the lateral mode is not active. Above it the
capacity does not depend on μ: Li and Vaz (2021a, Table 4) find 0.83–0.85
`P_cr` for μ = 0.05–0.15. The gate ignores elastic slip, which delays slip in
the source finite-element model (their Table 2), so the gate opens early and
the screen stays on the safe side.

**Radial buckling** of the armour stack on the elastic foundation of the
outboard hoop layers, straight beam (Eq. 22, default) or curved beam
(Eqs. 29–32):

    P2 = Σ_i 2 n_i cos α_i √(EI_2 c_i)

with the per-wire foundation stiffness of every continuous annulus outboard of
the outermost armour (Eq. 23) and of any tape (Eq. 24), taken over all wires
of the stack at the wire-count weighted mean radius:

    c_sheath = 2π E_s t_s cos α / (n_tot R),
    c_tape   = n_t EA_t sin⁴α_t cos α / (n_tot R² cos α_t).

**Hoop-capacity failure** (Eq. 33, generalised to layers of different lay
angle): `P3 = 2πH / Σ_i s_i tan²α_i / R_i`, with `s_i` the armour load share of
layer `i` and `H` the hoop force capacity per length,
`H = Σ_tapes n_t σ_ut A_t sin²α_t / (2π cos α_t R_t) + Σ_sheaths σ_s t_s`,
where a sheath behind a tape is taken at `min(E_s ε_ut, σ_s)`. For two equal
and opposite layers this is Eq. 33.

**Wire yield** (Eq. 34): `P4 = Σ_i n_i cos α_i σ_y A`.

**Interaction** (Eq. 35): `1/P2* = 1/P2 + 1/P3 + 1/P4`. Unavailable terms
(`inf`) drop out.

**Load sharing.** The cable force is shared among all layers by equal axial
strain (the EA-weighted, fixed-twist sharing of the stress and capacity
paths). Layer `i` carries `s_i C`; the stack carries `Σ s_i C`. The cable
compression capacity is `min(P_lat,i / s_i, P2* / Σ s_i)` over the active
modes.

## 3. Inputs: cable data and assumptions

| Input | Source | Note |
| --- | --- | --- |
| Wire count, diameter, helix radius, lay angle | Cable YAML | Round, rectangular and profiled wires (their own inertias in the lateral and radial screens) |
| E, ν, σ_y of the armour | Material library | |
| Outboard annuli (sheath, serving) and their E, thickness | Cable YAML, library | Bedding between armour layers is inside the stack and is not credited |
| Friction μ for the gate | `mu`, default the armour `mu_on_hdpe` | Capacity above first slip does not depend on it |
| Maximum cyclic curvature | Caller (`curvature_1_m`) | `None`: lateral slip assumed and η = 0.83 |
| Annulus state and external pressure | Caller (`annulus`, `p_ext_Pa`) | Not inferred from the YAML; a mismatch with the declaration is listed |
| Sheath hoop strength for P3 | Caller, with a source | Without it P3 is unavailable and listed |
| High-strength tape | Caller (`TapeLayer`) | Usually absent in power cables |
| Design utilisation factor η | Caller | See Section 4 |

Power cables usually have no high-strength tape. The radial restraint is then
the outer sheath alone, modelled through its own hoop stiffness and, if a
sourced strength is supplied, its hoop capacity. Every result lists the
assumptions it used in `assumptions` and the method in `provenance`.

## 4. Design factor

No power-cable code gives an armour compression capacity. DNV-ST-0359 lists
no compression limit among the mechanical specifications, DNV-RP-0360 keeps
compression within the manufacturer's recommendation, and MecLim II
(Section 4.3) notes that a single compression limit is usually given. The
screen therefore uses the armour utilisation factors of DNV-RP-F401
(September 2019, amended September 2021), Section 3.8.1, Table 3-1, p. 13, as
`cx.ARMOUR_UTILISATION_FACTORS`; the utilisation is the load over the lesser
of the yield strength and 90 % of the tensile strength. API 17J (3rd edition,
2008, Table 6, p. 19) gives the same 0.67 for the tensile armour of flexible
pipe in normal operation; the factor is cited to DNV-RP-F401, the power-cable
document.

| Condition | η |
| --- | --- |
| Normal operation | 0.67 |
| Installation | 0.78 |
| Abnormal operation | 1.00 |

Applying these yield-based factors to a buckling capacity is an assumption of
this screen. MecLim II (Table 5) proposes 0.75 for displacement-controlled
installation of other components because of lateral buckling. The design
utilisation is `utilisation / η`; the allowable compression is `η` times the
capacity; the margin is `allowable + driving tension` (positive within the
allowable).

The components other than the steel armour take the factors of MecLim II
(OWA guideline, July 2025, Table 5, p. 19) as
`cx.COMPONENT_UTILISATION_FACTORS`: 0.80 in normal operation, 1.00 in
installation under load control and 1.00 in abnormal operation, on the limit
that marks the onset of damage of each component (the yield or 0.2 % proof
stress of a metal, the allowable strain of a non-metal; MecLim II
Section 3.4, p. 15). The capacity envelope
(`cx.build_capacity_envelope`) checks every component with a limit in its
material (`cablexsec/analysis/component_limits.py`) and names the governing
component; a component without a limit says "no limit given". The Compression
analysis gives a verdict at its own compression: the design utilisation
against the design compression capacity with the armour slipping, the
mechanism named, and the radial (bird-caging) utilisation with the armour
stuck beside it.

## 5. Use

```python
import numpy as np
import cablexsec as cx

cable = cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)
res = cx.compression_screen(
    cable,
    tension_N=-3.0e3,            # effective tension, negative = compression
    curvature_1_m=0.15,          # maximum cyclic curvature from straight
    annulus="flooded",
    sheath_hoop_strength_Pa=20.74e6,
    sheath_strength_source="ASSUMED: HDPE yield, Malta et al. (2022) Table 4",
    utilisation_factor=cx.ARMOUR_UTILISATION_FACTORS["normal"],
)
print(res.governing, res.utilisation, res.design_utilisation, res.margin_N)
for lat in res.lateral:
    print(lat.layer, lat.P1_saevik_N, lat.capacity_N, lat.first_slip_curvature_1_m)
print(res.radial.P2_N, res.radial.P3_N, res.radial.P4_N, res.radial.P2_star_N)
print(res.assumptions)

# Combined curve: curvature capacity over signed tension (tension and compression)
curve = cx.compute_compression_capacity(
    cable, np.linspace(-1.0e4, 1.0e5, 12), utilisation_factor=0.67
)
print(curve.kappa_max_1_m, curve.governing)
```

`compute_compression_capacity` returns, at each tension, the smaller of the
first-yield curvature at usage η (the common slip-front wire stress of
`compute_capacity`) and the cyclic curvature at which a layer reaches η times
its lateral capacity; a radial design utilisation of one sets the capacity to
zero. At η = 1 and positive tension it equals the wire-yield branch of
`compute_capacity`. The bird-caging usage of `compute_capacity` now comes from
the same screen with lateral slip assumed and without P3.

A lay angle outside 20–38° (the range of the reproduced cases) raises an
`ExtrapolationWarning`: both lateral closed forms fall with sin²α and omit the
buckle-length term, so at small lay angles they become increasingly
conservative.

## 6. Bundled cables

`python validation/scripts/compression_application.py` with lateral slip active, η = 1,
a flooded annulus and an assumed sheath hoop strength of 20.74 MPa (HDPE
sheath yield of Malta et al. 2022, Table 4; not the cables' own grades):

| Cable | Lay angles (°) | P1 per layer (kN) | P2* stack (kN) | Armour share | Capacity (kN) | Governing | Allowable, η 0.67 / 0.78 (kN) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IEA Task 49 66 kV | 13, 10 | 3.98, 2.35 | 281 | 0.654 | 6.93 | lateral, outer | 4.64 / 5.40 |
| Ménard 2023 | 20.1, 19.6 | 39.6, 35.0 | 136 | 0.925 | 70.5 | lateral, outer | 47.2 / 55.0 |
| Inter-array 33 kV, single core | 25, 25 | 87.6, 76.2 | 52.9 | 0.809 | 65.4 | radial | 43.8 / 51.0 |
| Inter-array 33 kV, three core | 22, 22 | 68.2, 62.5 | 138 | 0.797 | 150 | lateral, outer | 100 / 117 |
| Export 220 kV, three core | 18, 18 | 37.4, 36.3 | 515 | 0.726 | 96.7 | lateral, outer | 64.8 / 75.4 |
| CIGRE TB880 220 kV | 12, 12 | 24.8, 23.2 | 1076 | 0.892 | 50.6 | lateral, outer | 33.9 / 39.5 |

For the 66 kV template the outer armour (145 wires at 10°) governs: a cable
compression of 6.9 kN loads it to its friction-free lower bound once the wires
slip, and the normal-operation allowable is 4.6 kN. Its lay angles are below
the validated range, so the value is a lower bound; the Li–Vaz limit with the
smallest tabulated knock-down is 1.5 times higher (outer layer 3.6 kN against
P1 2.35 kN). Radial buckling and yield (P2* 281 kN; 356 kN without P3) are
far from governing. The single-core 33 kV cable is governed by the hoop
capacity of its thin sheath (P3 63 kN), which depends entirely on the assumed
strength. The 33 kV rows are for the corrected example geometry; with the earlier, overlapping wire counts the three-core
cable was governed radially at 172 kN.

To screen a CableXSec local-response record of a global analysis, pass its
directory:

```bash
python validation/scripts/compression_application.py --local-response <dir> --cable menard2023 --annulus flooded
```

The script checks that the global section diameter matches the named cable,
then screens the arc-wise effective-tension minimum and, at every recovered
site, the minimum effective (or, sealed, true wall) tension with the largest
curvature of the record. Pairing the two extremes is conservative.

## 7. Limits

- Analytical screen: no imperfections, no plasticity within the buckling
  modes, no contact between neighbouring wires, no global torsional
  instability after an inner-layer failure (Østergaard et al. 2012).
- The lateral capacity is the post-buckling, friction-free lower bound once
  the wires slip. Tests exceed it by a factor of 1.6–1.9 (Sævik and Thorsen
  2017, Table 3), partly because the tests ran a limited number of cycles.
- The knock-down covers round wires at κ_max R ≤ 0.0107; beyond it the last
  value is held, while the source shows the limit still falling.
- The radial foundation uses the thin-ring hoop stiffness of the outboard
  annuli at the mean armour radius; the published per-term P2 values of the de
  Sousa case are not reproduced from the printed inputs ([benchmarks.md](benchmarks.md)).
- Against the only published finite-element case without a tape (Malta et
  al. 2022, tape cut over 2000 mm), the sheath-only interaction is
  conservative by a factor of about two, set by P3 at sheath yield.
- The design factor is a transfer from DNV-RP-F401 armour yield utilisation.
- Annulus state, external pressure, sheath strength and cyclic curvature are
  caller inputs; the screen does not infer them.

## References

- Sævik, S., Thorsen, M.J. (2017). An analytical treatment of buckling and
  instability of tensile armors in flexible pipes. J. Offshore Mech. Arct.
  Eng. 139(4), 041701.
- Li, X., Vaz, M.A. (2021a). Frictional effect on circular armour wire
  lateral buckling in umbilicals. Ocean Eng. 220, 108306.
- Li, X., Vaz, M.A. (2021b). Finite element study on circular armour wire
  lateral buckling in umbilicals. Mar. Struct. 76, 102895.
- Østergaard, N.H., Lyckegaard, A., Andreasen, J.H. (2012). On modelling of
  lateral buckling failure in flexible pipe tensile armour layers. Mar.
  Struct. 27, 64–81.
- Yang, X., Sævik, S., Sun, L. (2015). Numerical analysis of buckling failure
  in flexible pipe tensile armor wires. Ocean Eng. 108, 594–605.
- Malta, E.R., Shiri, H., Martins, C.A. (2022). Finite element analysis of
  flexible pipes under compression: a study on the damaged high strength tape
  and its effects on tensile armor instability. Ocean Eng. 245, 110470.
- Gjøsteen, J., Sævik, S., Yu, Z., Ye, N., Chen, K. (2027). Finite elements
  for the structural analysis of power cables. Mar. Struct. 111, 104189,
  Section 4.5 and Table 6 (de Sousa test data).
- Reda, A. (2020). Numerical modelling of subsea cables and pipelines. PhD
  thesis, Curtin University, Section 2.1.
- DNV-RP-F401 (2019), Table 3-1; DNV-ST-0359 (2016, amended 2021);
  DNV-RP-0360; MecLim II guidelines (Offshore Wind Accelerator).
