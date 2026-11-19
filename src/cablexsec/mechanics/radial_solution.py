# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Coupled radial-axial-torsional section solution with unilateral contact.

One small-strain elastic energy is minimised for the whole cross-section with
common cable extension and twist:

* concentric annuli use the exact axisymmetric basis u = A r + B/r in
  generalised plane strain (as :mod:`cablexsec.mechanics.concentric_section`);
* helical wire layers keep their axial strain energy with
  extension/twist/radial coupling (wire strain c² e + r c s tau + s² u / r);
* an optional three-core trefoil bundle (cores, filler, surrounding ring) is
  a plane finite-element model (:mod:`cablexsec.mechanics.trefoil_section`)
  coupled to the first concentric layer by its mean outer radial
  displacement;
* axial members add extension and twist rigidity only (for elements whose
  radial behaviour is carried by another layer).

All normal contacts are frictionless and unilateral. The contact problem is
solved exactly as a linear complementarity problem in the contact reactions
(forces per axial metre, full section). Free strains of every layer include
temperature and, for a flooded (wet) section, the strain of the uniform
water pressure; see :func:`flooded_free_strain`.

Manufacturing residual pressure is represented by residual wire strains.
:meth:`RadialSection.identify_initial_wire_strain` finds the initial wire
strains for which, at the reference state, every helical layer carries the
requested residual wire stress. A pretension_Pa pressure jump P0 maps to
that stress by the capstan relation (Féret and Bournazel 1987, Eq. 5).
Contacts can then open or close under later loads; nothing is superposed.

Excluded: friction and stick-slip in the radial solve, wire flattening and
lateral (hoop) wire contact, plasticity, ovalisation of layers outside the
trefoil ring, and the change of contact geometry under large displacement.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray
from scipy.optimize import nnls

from cablexsec.exceptions import CableInputError, CableSolverError
from cablexsec.mechanics.concentric_section import (
    ElasticAnnulus,
    ElasticHelicalRing,
    annulus_constitutive,
    annulus_torsional_rigidity,
)
from cablexsec.mechanics.trefoil_section import (
    SYMMETRY_FACTOR,
    TrefoilBundle,
    TrefoilModel,
    build_trefoil_model,
)

__all__ = [
    "AxialMember",
    "BundleState",
    "RadialSection",
    "RadialSectionResult",
    "capstan_residual_stress",
    "flooded_free_strain",
    "solve_lcp",
]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AxialMember:
    """Extension and twist rigidity only; no radial freedom or contact."""

    name: str
    axial_rigidity_N: float
    torsional_rigidity_Nm2: float = 0.0
    free_axial_strain: float = 0.0


Layer = ElasticAnnulus | ElasticHelicalRing


def flooded_free_strain(
    youngs_modulus_Pa: float, poisson_ratio: float, pressure_Pa: float
) -> float:
    """Free strain of an isotropic solid under uniform water pressure.

    A flooded section has the pressure on every surface. Its solution is a
    uniform hydrostatic stress -p in every solid plus the incompatibility
    correction obtained with free strain -p (1 - 2 nu)/E and no surface
    load. Contact reactions of the corrected problem are mechanical (net of
    water pressure). The true wall tension is the mechanical tension minus
    p times the solid area.
    """
    if not (
        math.isfinite(pressure_Pa)
        and pressure_Pa >= 0
        and youngs_modulus_Pa > 0
        and -1 < poisson_ratio < 0.5
    ):
        raise CableInputError("flooded strain needs finite p >= 0, E > 0, valid nu")
    return -pressure_Pa * (1 - 2 * poisson_ratio) / youngs_modulus_Pa


def capstan_residual_stress(
    pressure_jump_Pa: float,
    *,
    wire_count: int,
    wire_diameter_m: float,
    helix_radius_m: float,
    lay_angle_rad: float,
    wire_area_m2: float | None = None,
) -> float:
    """Wire stress whose capstan pressure jump is P0 (Féret-Bournazel Eq. 5).

    P0 = n sigma A sin(alpha) tan(alpha) / (2 pi r²); the inverse is returned.
    ``wire_area_m2`` replaces the round-wire area (rectangular wires).
    """
    angle = abs(lay_angle_rad)
    if not 0 < angle < math.pi / 2:
        raise CableInputError("capstan stress needs a lay angle in (0, pi/2)")
    area = math.pi * wire_diameter_m**2 / 4 if wire_area_m2 is None else wire_area_m2
    return (
        pressure_jump_Pa
        * 2
        * math.pi
        * helix_radius_m**2
        / (wire_count * area * math.sin(angle) * math.tan(angle))
    )


def solve_lcp(
    flexibility: FloatArray, free_gap: FloatArray, *, tolerance: float = 1e-10
) -> FloatArray:
    """Reactions r >= 0 with gap g + F r >= 0 and r (g + F r) = 0.

    F must be symmetric positive definite. The problem is the dual of the
    elastic contact problem and is solved as a non-negative least-squares
    problem on the scaled Cholesky factor, then polished on its active set.
    """
    n = len(free_gap)
    if n == 0:
        return np.zeros(0)
    scale = 1 / np.sqrt(np.diag(flexibility))
    f = flexibility * scale[:, None] * scale[None, :]
    g = free_gap * scale
    # Unit-diagonal scaling; a jitter of at most 1e-10 (a compliance of that
    # fraction of each row's own flexibility) regularises nearly dependent
    # contact rows without changing the solution measurably.
    jitter = 0.0
    for _ in range(8):
        try:
            low = np.linalg.cholesky(f + jitter * np.eye(n))
            break
        except np.linalg.LinAlgError:
            jitter = 1e-14 if jitter == 0.0 else 10 * jitter
    else:
        raise CableSolverError("contact flexibility is not positive definite")
    f = f + jitter * np.eye(n)
    rhs = -np.linalg.solve(low, g)
    reaction, _ = nnls(low.T, rhs, maxiter=50 * n + 100)
    # Polish on the identified active set; keep the NNLS solution if the
    # exact active-set iteration does not improve it.
    active = reaction > 0
    for _ in range(4 * n + 10):
        trial = np.zeros(n)
        if np.any(active):
            trial[active] = np.linalg.solve(f[np.ix_(active, active)], -g[active])
        negative = np.flatnonzero(active & (trial < 0))
        if negative.size:
            active[negative[np.argmin(trial[negative])]] = False
            continue
        gaps = g + f @ trial
        penetrating = np.flatnonzero(~active & (gaps < -tolerance))
        if penetrating.size:
            active[penetrating[np.argmin(gaps[penetrating])]] = True
            continue
        reaction = trial
        break
    gaps = g + f @ reaction
    scale_ref = max(1.0, float(np.max(np.abs(g))))
    if np.min(gaps, initial=0.0) < -1e-8 * scale_ref or np.max(
        np.abs(reaction * gaps), initial=0.0
    ) > 1e-8 * scale_ref * max(1.0, float(np.max(reaction, initial=0.0))):
        raise CableSolverError("contact complementarity was not achieved")
    return np.asarray(reaction * scale, dtype=np.float64)


@dataclass(frozen=True)
class BundleState:
    """Contact state of the trefoil bundle (forces per axial metre).

    core_core_force_N_per_m is the normal force between two neighbouring
    cores; core_ring_force_N_per_m is the direct normal force between one
    core and the ring; filler forces are totals per core interstice side
    (filler-core, per core) and per ring (filler-ring, whole ring).
    ring_outer_pressure_Pa is the uniform pressure from the layer outside.
    core_axial_force_N is the total of the three cores along the cable axis.
    """

    core_core_force_N_per_m: float
    core_ring_force_N_per_m: float
    filler_core_force_N_per_m: float
    filler_ring_force_N_per_m: float
    ring_outer_pressure_Pa: float
    ring_outer_mean_displacement_m: float
    core_centre_displacement_m: float
    core_axial_force_N: float
    adjusted_overlaps_m: Mapping[str, float]
    filled_core_gaps: tuple[str, ...]


@dataclass(frozen=True)
class RadialSectionResult:
    """Solved section state; reactions are line forces per axial metre.

    interface_line_force_N_per_m[i] acts between concentric_names[i] and
    concentric_names[i+1] (with a bundle, index 0 is bundle ring to first
    concentric layer, named "bundle"). helical_face_pressure_Pa gives
    (inner face, outer face) per helical layer as face line force divided by
    2 pi r_helix, so that P * 2 pi r_helix / n is the per-wire line load (the
    convention of the bounded chain and the Papailiou layers). A surface
    pressure is included where the layer is innermost or outermost.
    Interface pressure at the contact radius is line force / (2 pi r_i).
    """

    axial_strain: float
    twist_per_m: float
    axial_force_N: float
    torque_Nm: float
    interface_names: tuple[tuple[str, str], ...]
    interface_radius_m: FloatArray
    interface_line_force_N_per_m: FloatArray
    interface_gap_m: FloatArray
    inner_displacement_m: Mapping[str, float]
    outer_displacement_m: Mapping[str, float]
    helical_face_line_force_N_per_m: Mapping[str, tuple[float, float]]
    helical_face_pressure_Pa: Mapping[str, tuple[float, float]]
    wire_stress_Pa: Mapping[str, float]
    tangent: FloatArray  # [[EA, coupling], [coupling, GJ]] at fixed pressure
    active_contacts: int
    complementarity_residual: float
    relative_equilibrium_residual: float
    bundle: BundleState | None = None
    notes: tuple[str, ...] = field(default=())
    layer_axial_force_N: Mapping[str, float] = field(default_factory=dict)
    """Axial force of each layer and member [N]: the derivative of its strain
    energy with the cable extension, that is the axial projection of its
    stress resultant over its cross-section (a helical layer's wire forces
    times the cosine of the lay angle; each trefoil element's axial stress
    times its axial strain coefficient, the three cores of a layer
    together). The values sum to ``axial_force_N``. In a flooded section they
    are mechanical forces, net of the uniform water pressure."""

    @property
    def ea_fixed_twist_N(self) -> float:
        return float(self.tangent[0, 0])

    @property
    def ea_free_twist_N(self) -> float:
        return float(self.tangent[0, 0] - self.tangent[0, 1] ** 2 / self.tangent[1, 1])

    @property
    def gj_fixed_tension_Nm2(self) -> float:
        return float(self.tangent[1, 1] - self.tangent[0, 1] ** 2 / self.tangent[0, 0])

    def p1_plus_p2(self, name: str) -> float:
        inner, outer = self.helical_face_pressure_Pa[name]
        return inner + outer

    @property
    def interface_open(self) -> NDArray[np.bool_]:
        """Whether each interface is open: its contact carries no line force.

        The contact solver decides it (the complementarity of
        :func:`solve_lcp`): an open contact has exactly zero reaction, a
        closed one a positive line force. The reported gap of a closed
        contact is the round-off of the linear solve, of the order of the
        solver's complementarity tolerance in metres, ``1e-8`` of the scaled
        problem times the square root of the contact flexibility (1e-10 to
        4e-10 m on the 66 kV template), and it moves with the platform's
        linear algebra; so a gap threshold would classify a closed contact by
        its noise. ``interface_gap_m`` keeps the reported gaps.
        """
        return np.asarray(self.interface_line_force_N_per_m <= 0.0)

    @property
    def open_interfaces(self) -> tuple[tuple[str, str], ...]:
        """Names of the open interfaces (:attr:`interface_open`)."""
        return tuple(
            name
            for name, is_open in zip(
                self.interface_names, self.interface_open, strict=True
            )
            if is_open
        )


class RadialSection:
    """Assembled section; solve repeatedly for different load states.

    layers are concentric (sorted by inner radius) and lie outside the
    bundle when a bundle is given. members add rigidity only. With
    rigid_bore the innermost concentric surface is fixed radially;
    bore_foundation_Pa_per_m instead supports it on a radial foundation
    (pressure per inward displacement). interface_compliance maps an
    interface name pair to a normal compliance [m per N/m of line force],
    a penalty-type surface stiffness in series with the contact.
    """

    def __init__(
        self,
        layers: Sequence[Layer],
        *,
        bundle: TrefoilBundle | None = None,
        members: Sequence[AxialMember] = (),
        rigid_bore: bool = False,
        bore_foundation_Pa_per_m: float | None = None,
        interface_compliance: Mapping[tuple[str, str], float] | None = None,
        quadrature_order: int = 24,
    ) -> None:
        self.layers = sorted(layers, key=lambda x: x.inner_radius_m)
        self.members = tuple(members)
        self.bundle = bundle
        names = [x.name for x in self.layers] + [m.name for m in self.members]
        if len(set(names)) != len(names) or "bundle" in names:
            raise CableInputError("layer names must be unique and not 'bundle'")
        if not self.layers and bundle is None:
            raise CableInputError("at least one layer is required")
        self.model: TrefoilModel | None = (
            build_trefoil_model(bundle) if bundle is not None else None
        )
        if (
            self.model is not None
            and self.layers
            and self.layers[0].inner_radius_m < self.model.ring_outer_radius_m - 1e-12
        ):
            raise CableInputError("concentric layers overlap the bundle ring")
        if rigid_bore and (bundle is not None or not self.layers):
            raise CableInputError("a rigid bore needs a concentric innermost layer")
        if bore_foundation_Pa_per_m is not None and (
            rigid_bore
            or bundle is not None
            or not self.layers
            or not (
                math.isfinite(bore_foundation_Pa_per_m) and bore_foundation_Pa_per_m > 0
            )
        ):
            raise CableInputError("bore foundation needs a positive finite stiffness")
        self.rigid_bore = rigid_bore
        self.bore_foundation = bore_foundation_Pa_per_m
        self._assemble(quadrature_order)
        compliance = dict(interface_compliance or {})
        if set(compliance) - set(self.interface_names) or any(
            not (math.isfinite(v) and v >= 0) for v in compliance.values()
        ):
            raise CableInputError("interface compliance names an unknown interface")
        self.compliance = np.zeros(len(self.gap0))
        for pair, value in compliance.items():
            self.compliance[self.bundle_rows + self.interface_names.index(pair)] = value
        self._factors: dict[
            tuple[bool, bool], tuple[tuple[Any, sp.dia_matrix], NDArray[np.int64]]
        ] = {}

    # ------------------------------------------------------------------ #
    def _assemble(self, quadrature_order: int) -> None:
        model = self.model
        nb = model.size if model is not None else 2
        dofs = nb
        inner_ids: list[int] = []
        outer_ids: list[int] = []
        for i, layer in enumerate(self.layers):
            a, b = layer.inner_radius_m, layer.outer_radius_m
            if (
                not np.isfinite([a, b, layer.youngs_modulus_Pa]).all()
                or b <= a
                or a < 0
            ):
                raise CableInputError("layer geometry must be finite and ordered")
            if layer.youngs_modulus_Pa <= 0:
                raise CableInputError("layer modulus must be positive")
            if i and a < self.layers[i - 1].outer_radius_m - 1e-12:
                raise CableInputError("concentric layers overlap")
            if isinstance(layer, ElasticHelicalRing):
                if not 1e-8 < abs(layer.signed_lay_angle_rad) < math.pi / 2:
                    raise CableInputError("helical lay angle must be in (0, pi/2)")
                inner_ids.append(dofs)
                outer_ids.append(dofs)
                dofs += 1
            else:
                if not -1 < layer.poisson_ratio < 0.5:
                    raise CableInputError("Poisson ratio must lie in (-1, 0.5)")
                if a > 0:
                    inner_ids.append(dofs)
                    outer_ids.append(dofs + 1)
                    dofs += 2
                else:
                    # A solid layer elsewhere overlaps the layer or bundle ring
                    # inside it, which is refused above and in __init__.
                    assert i == 0 and model is None
                    inner_ids.append(-1)
                    outer_ids.append(dofs)
                    dofs += 1
        self.n = dofs
        self.inner_ids, self.outer_ids = inner_ids, outer_ids
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        load = np.zeros(dofs)
        unit_initial: dict[str, FloatArray] = {}

        def add_dense(ids: list[int], block: FloatArray) -> None:
            for p, ip in enumerate(ids):
                for q, iq in enumerate(ids):
                    if block[p, q] != 0.0:
                        rows.append(ip)
                        cols.append(iq)
                        vals.append(float(block[p, q]))

        nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
        torsion = 0.0
        axial = 0.0
        # Row of the extension freedom of each layer's own stiffness and load:
        # (name, freedoms, coefficients, load, coefficient of an initial wire
        # strain), so that its axial force is coefficients . u - load.
        axial_rows: list[tuple[str, list[int], FloatArray, float, float]] = []
        for i, layer in enumerate(self.layers):
            a, b, e = (
                layer.inner_radius_m,
                layer.outer_radius_m,
                layer.youngs_modulus_Pa,
            )
            if isinstance(layer, ElasticHelicalRing):
                angle, r = layer.signed_lay_angle_rad, layer.helix_radius_m
                c, s = math.cos(angle), math.sin(angle)
                ids = [0, 1, inner_ids[i]]
                vec = np.array([c * c, r * c * s, s * s / r])
                rigidity = layer.total_area_m2 * e / c
                add_dense(ids, rigidity * np.outer(vec, vec))
                load[ids] += (
                    rigidity
                    * (layer.free_thermal_strain - layer.initial_wire_strain)
                    * vec
                )
                unit = np.zeros(dofs)
                unit[ids] = -rigidity * vec
                unit_initial[layer.name] = unit
                axial_rows.append(
                    (
                        layer.name,
                        ids,
                        rigidity * vec[0] * vec,
                        float(
                            rigidity
                            * vec[0]
                            * (layer.free_thermal_strain - layer.initial_wire_strain)
                        ),
                        float(rigidity * vec[0]),
                    )
                )
                continue
            constitutive = annulus_constitutive(
                e, layer.poisson_ratio, layer.carries_axial_load
            )
            if a > 0:
                ids = [0, inner_ids[i], outer_ids[i]]
                coeff = np.linalg.inv(np.array([[a, 1 / a], [b, 1 / b]]))
            else:
                ids = [0, outer_ids[i]]
            block = np.zeros((len(ids), len(ids)))
            vec_load = np.zeros(len(ids))
            for node, weight in zip(nodes, weights, strict=True):
                rr = 0.5 * (a + b) + 0.5 * (b - a) * node
                bmat = np.zeros((3, len(ids)))
                bmat[2, 0] = 1.0
                if a > 0:
                    bmat[0, 1:] = np.array([1.0, -1 / rr**2]) @ coeff
                    bmat[1, 1:] = np.array([1.0, 1 / rr**2]) @ coeff
                else:
                    bmat[0, 1] = 1 / b
                    bmat[1, 1] = 1 / b
                vw = 2 * math.pi * rr * 0.5 * (b - a) * weight
                block += vw * bmat.T @ constitutive @ bmat
                vec_load += vw * bmat.T @ constitutive @ layer.free_strain_vector
            add_dense(ids, block)
            load[ids] += vec_load
            axial_rows.append(
                (layer.name, ids, block[0].copy(), float(vec_load[0]), 0.0)
            )
            torsion += annulus_torsional_rigidity(layer)
        for member in self.members:
            if member.axial_rigidity_N < 0 or member.torsional_rigidity_Nm2 < 0:
                raise CableInputError("member rigidities must be non-negative")
            axial += member.axial_rigidity_N
            torsion += member.torsional_rigidity_Nm2
            load[0] += member.axial_rigidity_N * member.free_axial_strain
            axial_rows.append(
                (
                    member.name,
                    [0],
                    np.array([member.axial_rigidity_N]),
                    member.axial_rigidity_N * member.free_axial_strain,
                    0.0,
                )
            )
        self.axial_rows = tuple(axial_rows)
        rows += [0, 1]
        cols += [0, 1]
        vals += [axial, torsion]
        if self.bore_foundation is not None:
            r0 = self.layers[0].inner_radius_m
            if inner_ids[0] < 0:
                raise CableInputError("a solid innermost layer has no bore")
            rows.append(inner_ids[0])
            cols.append(inner_ids[0])
            vals.append(2 * math.pi * r0 * self.bore_foundation)
        k = sp.coo_matrix((vals, (rows, cols)), shape=(dofs, dofs)).tocsr()
        transform_blocks = []
        if model is not None:
            kb = model.stiffness.tocoo()
            k = (
                k
                + sp.coo_matrix((kb.data, (kb.row, kb.col)), shape=(dofs, dofs)).tocsr()
            )
            load[:nb] += model.free_strain_load
            transform_blocks.append(model.transform)
            transform_blocks.append(sp.identity(dofs - nb, format="csr"))
            self.transform = sp.block_diag(transform_blocks, format="csr")
        else:
            self.transform = sp.identity(dofs, format="csr")
        if self.rigid_bore:
            if inner_ids[0] < 0:
                raise CableInputError("a solid innermost layer has no bore")
            keep = [j for j in range(dofs) if j != inner_ids[0]]
            self.transform = sp.identity(dofs, format="csr")[:, keep]
        self.stiffness = k
        self.free_load = load
        self.unit_initial = unit_initial

        # A helical ring has one radial freedom (its wire centreline). Its
        # faces lie d/2 inside and outside; a free strain (temperature,
        # flooding) also changes the wire diameter, which shifts both faces.
        self.face_shift = [
            0.5 * layer.thickness_m * layer.free_thermal_strain
            if isinstance(layer, ElasticHelicalRing)
            else 0.0
            for layer in self.layers
        ]
        # Contact rows and initial gaps.
        crow: list[dict[int, float]] = []
        gaps: list[float] = []
        names: list[tuple[str, str]] = []
        radii: list[float] = []
        self.bundle_rows = 0
        if model is not None:
            cb = model.contact.tocoo()
            crow.extend({} for _ in range(model.contact.shape[0]))
            for r, c, v in zip(cb.row, cb.col, cb.data, strict=True):
                crow[r][int(c)] = crow[r].get(int(c), 0.0) + float(v)
            gaps.extend(model.gap.tolist())
            self.bundle_rows = len(crow)
            if self.layers:
                row = {
                    int(c): -float(v)
                    for c, v in zip(
                        model.outer_weights.indices,
                        model.outer_weights.data,
                        strict=True,
                    )
                }
                row[inner_ids[0]] = row.get(inner_ids[0], 0.0) + 1.0
                crow.append(row)
                gaps.append(
                    self.layers[0].inner_radius_m
                    - model.ring_outer_radius_m
                    - self.face_shift[0]
                )
                names.append(("bundle", self.layers[0].name))
                radii.append(
                    0.5 * (self.layers[0].inner_radius_m + model.ring_outer_radius_m)
                )
        for i in range(len(self.layers) - 1):
            crow.append({outer_ids[i]: -1.0, inner_ids[i + 1]: 1.0})
            gaps.append(
                self.layers[i + 1].inner_radius_m
                - self.layers[i].outer_radius_m
                - self.face_shift[i]
                - self.face_shift[i + 1]
            )
            names.append((self.layers[i].name, self.layers[i + 1].name))
            radii.append(
                0.5
                * (self.layers[i + 1].inner_radius_m + self.layers[i].outer_radius_m)
            )
        contact = sp.lil_matrix((len(crow), dofs))
        for i, row in enumerate(crow):
            for c, v in row.items():
                contact[i, c] = v
        self.contact = contact.tocsr()
        self.gap0 = np.array(gaps, dtype=np.float64)
        self.interface_names = tuple(names)
        self.interface_radius = np.array(radii, dtype=np.float64)

    # ------------------------------------------------------------------ #
    def _factor(
        self, fix_strain: bool, fix_twist: bool
    ) -> tuple[tuple[Any, sp.dia_matrix], NDArray[np.int64]]:
        key = (fix_strain, fix_twist)
        if key not in self._factors:
            kr = (self.transform.T @ self.stiffness @ self.transform).tocsc()
            kr = 0.5 * (kr + kr.T)
            free = np.array(
                [
                    j
                    for j in range(kr.shape[0])
                    if not ((j == 0 and fix_strain) or (j == 1 and fix_twist))
                ]
            )
            kff = kr[free][:, free].tocsc()
            diag = kff.diagonal()
            if np.any(diag <= 0):
                raise CableInputError("section has an unsupported elastic mechanism")
            scale = sp.diags(1 / np.sqrt(diag))
            lu = spla.splu((scale @ kff @ scale).tocsc())
            self._factors[key] = ((lu, scale), free)
        return self._factors[key]

    def _solve_k(
        self, fix_strain: bool, fix_twist: bool, rhs: FloatArray
    ) -> FloatArray:
        (lu, scale), _ = self._factor(fix_strain, fix_twist)
        d = scale.diagonal()
        if rhs.ndim == 1:
            return np.asarray(d * lu.solve(d * rhs), dtype=np.float64)
        return np.asarray(d[:, None] * lu.solve(d[:, None] * rhs), dtype=np.float64)

    def surface_loads(
        self,
        internal_pressure_Pa: float,
        external_pressure_Pa: float,
        internal_pressure_layer: int = 0,
    ) -> FloatArray:
        """Radial surface loads [N/m per dof] of the bore and outer pressure.

        ``internal_pressure_layer`` is the index of the innermost fluid-tight
        layer (default 0, the bore). The bore fluid then also fills every
        layer inside it, which carries the pressure on both faces (a radial
        load only; the axial pressure on those layers' cross-sections is not
        applied). Contact reactions inside the barrier are mechanical, net of
        the fluid.
        """
        load = np.zeros(self.n)
        barrier = int(internal_pressure_layer)
        if not 0 <= barrier < max(len(self.layers), 1):
            raise CableInputError("internal pressure layer index out of range")
        if internal_pressure_Pa:
            if self.model is not None or self.inner_ids[0] < 0:
                raise CableInputError("this section has no internal pressure surface")
            load[self.inner_ids[barrier]] += (
                2 * math.pi * self.layers[barrier].inner_radius_m * internal_pressure_Pa
            )
            for j in range(barrier):
                layer = self.layers[j]
                load[self.inner_ids[j]] += (
                    2 * math.pi * layer.inner_radius_m * internal_pressure_Pa
                )
                load[self.outer_ids[j]] -= (
                    2 * math.pi * layer.outer_radius_m * internal_pressure_Pa
                )
        if external_pressure_Pa:
            if self.layers:
                load[self.outer_ids[-1]] -= (
                    2 * math.pi * self.layers[-1].outer_radius_m * external_pressure_Pa
                )
            else:
                assert self.model is not None
                w = self.model.outer_weights
                load[w.indices] -= (
                    2 * math.pi * self.model.ring_outer_radius_m * external_pressure_Pa
                ) * w.data
        return load

    def solve(
        self,
        *,
        tension_N: float = 0.0,
        prescribed_axial_strain: float | None = None,
        torque_Nm: float = 0.0,
        prescribed_twist_per_m: float | None = 0.0,
        internal_pressure_Pa: float = 0.0,
        external_pressure_Pa: float = 0.0,
        initial_wire_strain: Mapping[str, float] | None = None,
        internal_pressure_layer: int = 0,
    ) -> RadialSectionResult:
        """Solve one load state (SI units, true axial force, see module).

        ``internal_pressure_layer`` selects the innermost fluid-tight layer
        (:meth:`surface_loads`); 0 is the bore.
        """
        values = [tension_N, torque_Nm, internal_pressure_Pa, external_pressure_Pa]
        values += [
            x
            for x in (prescribed_axial_strain, prescribed_twist_per_m)
            if x is not None
        ]
        if (
            not np.isfinite(values).all()
            or min(internal_pressure_Pa, external_pressure_Pa) < 0
        ):
            raise CableInputError("loads must be finite and pressures non-negative")
        if prescribed_axial_strain is not None and tension_N:
            raise CableInputError("prescribe tension or axial strain, not both")
        if prescribed_twist_per_m is not None and torque_Nm:
            raise CableInputError("prescribe torque or twist, not both")
        initial = dict(initial_wire_strain or {})
        if set(initial) - set(self.unit_initial):
            raise CableInputError("initial wire strain names an unknown helical layer")
        fix_e = prescribed_axial_strain is not None
        fix_t = prescribed_twist_per_m is not None
        load = self.free_load + self.surface_loads(
            internal_pressure_Pa, external_pressure_Pa, internal_pressure_layer
        )
        for name, value in initial.items():
            load = load + value * self.unit_initial[name]
        base_load = load.copy()
        load[0] += tension_N
        load[1] += torque_Nm
        t = self.transform
        f_red = t.T @ load
        if not hasattr(self, "_c_red"):
            self._c_red = (
                (self.contact @ t).toarray()
                if self.contact.shape[0]
                else np.zeros((0, t.shape[1]))
            )
        c_red = self._c_red
        _, free = self._factor(fix_e, fix_t)
        u_red = np.zeros(t.shape[1])
        if fix_e:
            u_red[0] = prescribed_axial_strain
        if fix_t:
            u_red[1] = prescribed_twist_per_m
        if not hasattr(self, "_kr"):
            self._kr = (t.T @ self.stiffness @ t).tocsr()
        rhs = f_red - self._kr @ u_red
        base = self._solve_k(fix_e, fix_t, rhs[free])
        cf = c_red[:, free]
        if len(self.gap0):
            response, flexibility = self._contact_response(fix_e, fix_t, cf)
            flexibility = flexibility + np.diag(self.compliance)
            free_gap = self.gap0 + c_red @ u_red + cf @ base
            reaction = solve_lcp(flexibility, free_gap)
            u_red[free] += base + response @ reaction
        else:
            reaction = np.zeros(0)
            u_red[free] += base
        u = t @ u_red
        gaps = self.gap0 + self.contact @ u + self.compliance * reaction
        residual = t.T @ (self.stiffness @ u - load - self.contact.T @ reaction)
        rel = float(np.linalg.norm(residual[free])) / max(
            float(np.linalg.norm(f_red[free])),
            float(np.max(np.abs(reaction), initial=0.0)),
            1.0,
        )
        internal = self.stiffness @ u - base_load
        active = reaction > 0
        tangent = self._tangent(c_red, active)
        return self._result(
            u,
            reaction,
            gaps,
            internal,
            tangent,
            rel,
            initial,
            # Bore fluid inside a barrier is not a contact (friction) pressure.
            internal_pressure_Pa if internal_pressure_layer == 0 else 0.0,
            external_pressure_Pa,
        )

    def _contact_response(
        self, fix_e: bool, fix_t: bool, cf: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Cached K^-1 C^T and contact flexibility for one end condition."""
        cache: dict[tuple[bool, bool], tuple[FloatArray, FloatArray]] = (
            self.__dict__.setdefault("_responses", {})
        )
        key = (fix_e, fix_t)
        if key not in cache:
            response = self._solve_k(fix_e, fix_t, cf.T)
            flexibility = cf @ response
            cache[key] = (response, 0.5 * (flexibility + flexibility.T))
        response, flexibility = cache[key]
        return response, flexibility

    def _tangent(self, c_red: FloatArray, active: NDArray[np.bool_]) -> FloatArray:
        """Extension/twist stiffness on the current active contact set.

        Schur complement on the internal freedoms (extension and twist
        fixed), so a section without independent torsional rigidity still
        has a defined fixed-twist stiffness. Active contacts are kept closed
        through their compliance: S = S0 + G^T (F_A + c_A)^-1 G.
        """
        _, free = self._factor(True, True)
        kr = self._kr
        k_ig = kr[free][:, [0, 1]].toarray()
        k_gg = kr[[0, 1]][:, [0, 1]].toarray()
        x = self._solve_k(True, True, k_ig)
        stiffness = k_gg - k_ig.T @ x
        if np.any(active):
            response, flexibility = self._contact_response(True, True, c_red[:, free])
            g = c_red[active][:, [0, 1]] - c_red[active][:, free] @ x
            f_a = flexibility[np.ix_(active, active)] + np.diag(self.compliance[active])
            stiffness = stiffness + g.T @ np.linalg.solve(f_a, g)
        return np.asarray(0.5 * (stiffness + stiffness.T), dtype=np.float64)

    def _result(
        self,
        u: FloatArray,
        reaction: FloatArray,
        gaps: FloatArray,
        internal: FloatArray,
        tangent: FloatArray,
        rel: float,
        initial: Mapping[str, float],
        p_int: float,
        p_ext: float,
    ) -> RadialSectionResult:
        nbr = self.bundle_rows
        iface = reaction[nbr:]
        igap = gaps[nbr:]
        inner_disp, outer_disp, face_force, face_pressure, stress = {}, {}, {}, {}, {}
        offset = 1 if (self.model is not None and self.layers) else 0
        for i, layer in enumerate(self.layers):
            shift = self.face_shift[i]
            inner_disp[layer.name] = (
                float(u[self.inner_ids[i]]) - shift if self.inner_ids[i] >= 0 else 0.0
            )
            outer_disp[layer.name] = float(u[self.outer_ids[i]]) + shift
            if not isinstance(layer, ElasticHelicalRing):
                continue
            k_below = i - 1 + offset
            if k_below >= 0:
                below = float(iface[k_below])
                r_below = float(self.interface_radius[k_below])
            else:
                r_below = layer.inner_radius_m
                below = 2 * math.pi * r_below * p_int
            if i < len(self.layers) - 1:
                above = float(iface[i + offset])
                r_above = float(self.interface_radius[i + offset])
            else:
                r_above = layer.outer_radius_m
                above = 2 * math.pi * r_above * p_ext
            face_force[layer.name] = (below, above)
            # Referred to the helix radius: P * 2 pi r_helix / n is then the
            # per-wire line load, the convention of the Papailiou layers and
            # of the bounded pressure chain.
            face_pressure[layer.name] = (
                below / (2 * math.pi * layer.helix_radius_m),
                above / (2 * math.pi * layer.helix_radius_m),
            )
            angle, r = layer.signed_lay_angle_rad, layer.helix_radius_m
            c, s = math.cos(angle), math.sin(angle)
            strain = (
                c * c * u[0]
                + r * c * s * u[1]
                + s * s * u[self.inner_ids[i]] / r
                - layer.free_thermal_strain
                + layer.initial_wire_strain
                + initial.get(layer.name, 0.0)
            )
            stress[layer.name] = float(layer.youngs_modulus_Pa * strain)
        bundle_state = None
        if self.model is not None:
            bundle_state = self._bundle_state(u, reaction, p_ext)
        comp = float(np.max(np.abs(reaction * gaps), initial=0.0))
        layer_forces = self.layer_axial_forces(u, initial)
        return RadialSectionResult(
            float(u[0]),
            float(u[1]),
            float(internal[0]),
            float(internal[1]),
            self.interface_names,
            self.interface_radius.copy(),
            iface.copy(),
            igap.copy(),
            inner_disp,
            outer_disp,
            face_force,
            face_pressure,
            stress,
            tangent,
            int(np.count_nonzero(reaction > 0)),
            comp,
            rel,
            bundle_state,
            layer_axial_force_N=layer_forces,
        )

    def layer_axial_forces(
        self, u: FloatArray, initial: Mapping[str, float] | None = None
    ) -> dict[str, float]:
        """Axial force of each layer and member [N] for the displacements ``u``.

        Each is the layer's own row of the extension freedom, ``k_l . u -
        f_l``: the derivative of its strain energy with the extension, which
        for an annulus is the integral of its axial stress and for a helical
        layer its wire forces times the cosine of the lay angle. The trefoil
        elements give theirs from the axial stress at the quadrature points,
        grouped by layer (three cores together). Because the global extension
        row is the sum of these rows, the forces sum to the section's axial
        force; no force is assigned by difference.
        """
        initial = dict(initial or {})
        forces: dict[str, float] = {}
        if self.model is not None:
            names = self.model.element_layers
            for name, value in zip(names, self._element_axial_forces(u), strict=True):
                forces[name] = forces.get(name, 0.0) + float(value)
        for name, ids, coefficients, load, initial_coefficient in self.axial_rows:
            value = float(coefficients @ u[ids]) - load
            value += initial_coefficient * initial.get(name, 0.0)
            forces[name] = forces.get(name, 0.0) + value
        return forces

    def layer_solid_areas(self) -> dict[str, float]:
        """Area of each layer's material in the cable cross-section [m²].

        A helical layer's wires are cut obliquely, so their area in the cable
        plane is the wire area over the cosine of the lay angle; a trefoil
        layer's is that of its elements (the three cores of a layer
        together). The values sum to the solid area that a flooded section's
        mechanical axial force uses.
        """
        areas: dict[str, float] = {}
        for layer in self.layers:
            if isinstance(layer, ElasticHelicalRing):
                area = layer.total_area_m2 / math.cos(layer.signed_lay_angle_rad)
            else:
                area = math.pi * (layer.outer_radius_m**2 - layer.inner_radius_m**2)
            areas[layer.name] = areas.get(layer.name, 0.0) + area
        model = self.model
        if model is not None:
            p = model.nodes[model.element_nodes[:, :3]]
            tri = 0.5 * np.abs(
                (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
            )
            for name, value in zip(model.element_layers, tri, strict=True):
                areas[name] = areas.get(name, 0.0) + SYMMETRY_FACTOR * float(value)
        return areas

    def _element_axial_forces(self, u: FloatArray) -> FloatArray:
        """Extension row of every trefoil element (full section) [N].

        The derivative of the element's energy with the extension: its axial
        stress integrated over the element, times the axial strain
        coefficient of the extension (cos² of the core lay angle for a core
        element, one for the ring and the filler), times the symmetry factor.
        """
        from cablexsec.mechanics.trefoil_section import _QW, _SHAPES  # noqa: PLC0415

        model = self.model
        assert model is not None
        conn = model.element_nodes
        coords = model.nodes[conn]
        e = model.element_moduli
        nu = model.element_poisson
        e0 = model.element_free_strain
        transverse = (
            np.zeros_like(e0)
            if model.element_transverse_strain is None
            else model.element_transverse_strain
        )
        lam = e * nu / ((1 + nu) * (1 - 2 * nu))
        mu = e / (2 * (1 + nu))
        ux = u[2 + 2 * conn]
        uy = u[3 + 2 * conn]
        amap = model.element_axial_map
        ezz = (
            amap[:, 0] * u[0]
            + amap[:, 1] * u[1]
            + amap[:, 2] * u[model.core_centre_dofs[0]]
            + amap[:, 3] * u[model.core_centre_dofs[1]]
        )
        force = np.zeros(len(conn))
        for (_, dshape), w in zip(_SHAPES, _QW, strict=True):
            jac = np.einsum("ka,eki->eai", dshape, coords)
            det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
            inv = np.empty_like(jac)
            inv[:, 0, 0] = jac[:, 1, 1] / det
            inv[:, 1, 1] = jac[:, 0, 0] / det
            inv[:, 0, 1] = -jac[:, 0, 1] / det
            inv[:, 1, 0] = -jac[:, 1, 0] / det
            dn = np.einsum("ka,eia->eki", dshape, inv)
            exx = np.einsum("ek,ek->e", dn[:, :, 0], ux)
            eyy = np.einsum("ek,ek->e", dn[:, :, 1], uy)
            szz = lam * (exx + eyy + ezz - 3 * e0 - 2 * transverse) + 2 * mu * (
                ezz - e0
            )
            force += w * det * szz * model.element_axial_weight
        return np.asarray(SYMMETRY_FACTOR * force * amap[:, 0], dtype=np.float64)

    def _bundle_state(
        self, u: FloatArray, reaction: FloatArray, p_ext: float
    ) -> BundleState:
        model = self.model
        assert model is not None
        totals = {}
        for fam in model.families:
            totals[fam.name] = fam.force_factor * float(np.sum(reaction[fam.rows]))
        mean_out = float((model.outer_weights @ u[: model.size])[0])
        if self.layers:
            line = float(reaction[self.bundle_rows])
        else:
            line = 2 * math.pi * model.ring_outer_radius_m * p_ext
        return BundleState(
            totals.get("core_core", 0.0),
            totals.get("core_ring", 0.0),
            totals.get("filler_core", 0.0),
            totals.get("filler_ring", 0.0),
            line / (2 * math.pi * model.ring_outer_radius_m),
            mean_out,
            float(np.mean(u[list(model.core_centre_dofs)])),
            self._core_axial_force(u),
            dict(model.adjusted_overlaps_m),
            model.filled_core_gaps,
        )

    def _core_axial_force(self, u: FloatArray) -> float:
        """Cable-axis force of the three cores from element stresses."""
        from cablexsec.mechanics.trefoil_section import _QW, _SHAPES  # noqa: PLC0415

        model = self.model
        assert model is not None
        idx = model.element_groups.get("core")
        if idx is None or len(idx) == 0:
            return 0.0
        conn = model.element_nodes[idx]
        coords = model.nodes[conn]
        e = model.element_moduli[idx]
        nu = model.element_poisson[idx]
        e0 = model.element_free_strain[idx]
        transverse = (
            np.zeros_like(e0)
            if model.element_transverse_strain is None
            else model.element_transverse_strain[idx]
        )
        lam = e * nu / ((1 + nu) * (1 - 2 * nu))
        mu = e / (2 * (1 + nu))
        ux = u[2 + 2 * conn]
        uy = u[3 + 2 * conn]
        amap = model.element_axial_map[idx]
        ezz = (
            amap[:, 0] * u[0]
            + amap[:, 1] * u[1]
            + amap[:, 2] * u[model.core_centre_dofs[0]]
            + amap[:, 3] * u[model.core_centre_dofs[1]]
        )
        force = 0.0
        for (_, dshape), w in zip(_SHAPES, _QW, strict=True):
            jac = np.einsum("ka,eki->eai", dshape, coords)
            det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
            inv = np.empty_like(jac)
            inv[:, 0, 0] = jac[:, 1, 1] / det
            inv[:, 1, 1] = jac[:, 0, 0] / det
            inv[:, 0, 1] = -jac[:, 0, 1] / det
            inv[:, 1, 0] = -jac[:, 1, 0] / det
            dn = np.einsum("ka,eia->eki", dshape, inv)
            exx = np.einsum("ek,ek->e", dn[:, :, 0], ux)
            eyy = np.einsum("ek,ek->e", dn[:, :, 1], uy)
            szz = lam * (exx + eyy + ezz - 3 * e0 - 2 * transverse) + 2 * mu * (
                ezz - e0
            )
            force += float(np.sum(w * det * szz))
        # The sector holds half a core. The cable-plane integral F_zz equals
        # the rod force divided by cos(beta); its axial projection is
        # F_zz cos²(beta), the derivative of the energy with extension.
        cb = math.cos(model.core_lay_angle_rad)
        return SYMMETRY_FACTOR * force * cb * cb

    # ------------------------------------------------------------------ #
    def identify_initial_wire_strain(
        self,
        residual_stress_Pa: Mapping[str, float],
        *,
        tolerance_Pa: float = 1.0,
        max_iterations: int = 30,
    ) -> dict[str, float]:
        """Initial wire strains giving the residual wire stresses at reference.

        The reference state has zero tension, zero twist, no surface
        pressure and the layers' stated free strains. Layers not named keep
        zero initial strain. A Newton iteration on the piecewise-linear map
        is used; it stops when every stress error is below tolerance_Pa.
        """
        names = list(residual_stress_Pa)
        if set(names) - set(self.unit_initial):
            raise CableInputError("residual stress names an unknown helical layer")
        if not names:
            return {}
        target = np.array([residual_stress_Pa[n] for n in names], dtype=np.float64)
        moduli = {x.name: x.youngs_modulus_Pa for x in self.layers}
        guess = np.array([target[i] / moduli[n] for i, n in enumerate(names)])

        def stresses(x: FloatArray) -> FloatArray:
            res = self.solve(initial_wire_strain=dict(zip(names, x, strict=True)))
            return np.array([res.wire_stress_Pa[n] for n in names])

        for _ in range(max_iterations):
            current = stresses(guess)
            error = current - target
            if np.max(np.abs(error)) <= tolerance_Pa:
                return dict(zip(names, (float(v) for v in guess), strict=True))
            jac = np.zeros((len(names), len(names)))
            for j in range(len(names)):
                step = 1e-6 * max(abs(guess[j]), 1e-4)
                trial = guess.copy()
                trial[j] += step
                jac[:, j] = (stresses(trial) - current) / step
            guess = guess - np.linalg.solve(jac, error)
        raise CableSolverError("initial wire strain identification did not converge")
