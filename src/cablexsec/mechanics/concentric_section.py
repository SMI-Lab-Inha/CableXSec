# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Coupled extension, twist and radial contact of concentric elastic layers.

Annuli use the exact axisymmetric displacement basis u(r)=A r+B/r and
three-dimensional isotropic elasticity. A solid core uses u(r)=A r. Helical
wire layers retain their axial strain energy per cable-axis length, including
extension/radial/twist coupling. Contact reactions are forces per axial metre;
this weighting preserves reciprocal compliance between unequal-radius
surfaces. No trefoil homogenisation, ovalisation, plasticity, wire bending or
temperature-dependent friction is implied by this concentric subproblem.

Each layer is axially compatible with the cable. Radial interfaces are
frictionless unilateral constraints. Normal contact is solved through the
dual of the elastic energy minimisation. A prescribed initial wire strain
represents prestrain, not a measured residual winding stress after assembly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cablexsec.exceptions import CableInputError, CableSolverError

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ElasticAnnulus:
    name: str
    inner_radius_m: float
    outer_radius_m: float
    youngs_modulus_Pa: float
    poisson_ratio: float
    free_thermal_strain: float = 0.0
    # In-plane (radial and hoop) free strain added to the isotropic one: a
    # frozen extrusion shrinkage is -hoop_strain (no axial part).
    free_transverse_strain: float = 0.0
    # False for a yarn or tape serving: plane stress along the cable axis
    # (no axial stress) and no Saint-Venant torsion; the radial and hoop
    # stiffness are those of the isotropic material.
    carries_axial_load: bool = True

    @property
    def free_strain_vector(self) -> NDArray[np.float64]:
        """Free strain (radial, hoop, axial)."""
        e = self.free_thermal_strain
        t = self.free_transverse_strain
        return np.array([e + t, e + t, e], dtype=np.float64)


def annulus_constitutive(
    youngs_modulus_Pa: float, poisson_ratio: float, carries_axial_load: bool = True
) -> FloatArray:
    """Isotropic stiffness in (radial, hoop, axial) strains [Pa], 3 x 3.

    Three-dimensional elasticity for a solid annulus; plane stress along the
    axis (zero axial row and column) for a serving that carries no axial load.
    """
    e, nu = youngs_modulus_Pa, poisson_ratio
    if not carries_axial_load:
        plane = e / (1 - nu * nu)
        out = np.zeros((3, 3))
        out[:2, :2] = plane * np.array([[1.0, nu], [nu, 1.0]])
        return out
    shear = e / (2 * (1 + nu))
    lame = e * nu / ((1 + nu) * (1 - 2 * nu))
    return np.asarray(lame * np.ones((3, 3)) + 2 * shear * np.eye(3), dtype=np.float64)


def annulus_torsional_rigidity(layer: ElasticAnnulus) -> float:
    """Saint-Venant torsional rigidity ``G J`` of an annulus [N m^2].

    Zero for a serving, which carries no torque.
    """
    if not layer.carries_axial_load:
        return 0.0
    a, b = layer.inner_radius_m, layer.outer_radius_m
    shear = layer.youngs_modulus_Pa / (2 * (1 + layer.poisson_ratio))
    return shear * math.pi * (b**4 - a**4) / 2


@dataclass(frozen=True)
class ElasticHelicalRing:
    name: str
    helix_radius_m: float
    wire_diameter_m: float
    wire_count: int
    signed_lay_angle_rad: float
    youngs_modulus_Pa: float
    free_thermal_strain: float = 0.0
    initial_wire_strain: float = 0.0
    # Rectangular or profiled wires (gap G1): radial thickness and area of
    # one wire. ``None`` uses the round wire of diameter ``wire_diameter_m``.
    radial_thickness_m: float | None = None
    wire_area_m2: float | None = None

    @property
    def thickness_m(self) -> float:
        """Radial extent of a wire [m]."""
        if self.radial_thickness_m is None:
            return self.wire_diameter_m
        return self.radial_thickness_m

    @property
    def total_area_m2(self) -> float:
        """Steel area of the ring normal to the wires [m^2]."""
        if self.wire_area_m2 is None:
            return self.wire_count * math.pi * self.wire_diameter_m**2 / 4
        return self.wire_count * self.wire_area_m2

    @property
    def inner_radius_m(self) -> float:
        if self.radial_thickness_m is None:
            return self.helix_radius_m - self.wire_diameter_m / 2
        return self.helix_radius_m - self.radial_thickness_m / 2

    @property
    def outer_radius_m(self) -> float:
        if self.radial_thickness_m is None:
            return self.helix_radius_m + self.wire_diameter_m / 2
        return self.helix_radius_m + self.radial_thickness_m / 2


@dataclass(frozen=True)
class ConcentricTangent:
    """Incremental axial/twist law at fixed surface pressure and temperature.

    ``[dN, dQ] = matrix @ [d(strain), d(twist)]`` with radial displacements
    eliminated on the current contact branch. The diagonal entries have
    units N and N m²; the off-diagonal entries have units N m. An interface
    with zero gap and zero reaction is a branch junction, not a unique
    two-sided derivative. The returned matrix retains the solver's selected
    active set; use a new equilibrium solve when that set changes.
    """

    matrix: FloatArray
    active_interfaces: tuple[bool, ...]
    branch_junction_interfaces: tuple[int, ...]

    @property
    def ea_fixed_N(self) -> float:
        return float(self.matrix[0, 0])

    @property
    def ea_free_N(self) -> float:
        """Axial rigidity with torque held, allowing twist to change."""
        if self.matrix[1, 1] <= 0:
            raise CableInputError(
                "free-twist stiffness requires a torsionally supported section"
            )
        return float(self.matrix[0, 0] - self.matrix[0, 1] ** 2 / self.matrix[1, 1])

    @property
    def gj_at_fixed_tension_Nm2(self) -> float:
        return float(self.matrix[1, 1] - self.matrix[0, 1] ** 2 / self.matrix[0, 0])


def _condense_radial_tangent(
    matrix: FloatArray, contact: FloatArray, active: NDArray[np.bool_]
) -> FloatArray:
    """Eliminate radial freedom subject to active radial compatibility."""
    radial = matrix[2:, 2:]
    scale = 1 / np.sqrt(np.diag(radial))
    scaled = radial * scale[:, None] * scale[None, :]
    coupling = matrix[2:, :2]
    constraints = contact[active, 2:]
    rhs = np.column_stack((coupling, constraints.T))
    response = np.linalg.solve(scaled, rhs * scale[:, None]) * scale[:, None]
    radial_response = response[:, :2]
    if np.any(active):
        constraint_response = response[:, 2:]
        radial_response -= constraint_response @ np.linalg.solve(
            constraints @ constraint_response, constraints @ radial_response
        )
    return matrix[:2, :2] - coupling.T @ radial_response


@dataclass(frozen=True)
class ConcentricSectionResult:
    axial_strain: float
    twist_per_m: float
    inner_displacement_m: dict[str, float]
    outer_displacement_m: dict[str, float]
    contact_pressure_Pa: FloatArray
    contact_line_force_N_per_m: FloatArray
    gap_m: FloatArray
    axial_resultant_N: float
    torque_resultant_Nm: float
    relative_equilibrium_residual: float
    complementarity_residual_N: float
    tangent: ConcentricTangent


def solve_concentric_section(
    layers: list[ElasticAnnulus | ElasticHelicalRing],
    *,
    tension_N: float = 0.0,
    torque_Nm: float = 0.0,
    prescribed_twist_per_m: float | None = 0.0,
    internal_pressure_Pa: float = 0.0,
    external_pressure_Pa: float = 0.0,
    quadrature_order: int = 24,
) -> ConcentricSectionResult:
    """Solve one small-strain concentric section in SI units.

    Tension is the true cable-axis resultant. Internal/external pressures act
    on radial surfaces only; any closed-end pressure thrust must be included
    explicitly in tension_N. Twist is fixed by default. Pass None for
    prescribed_twist_per_m to prescribe torque instead. Layer temperatures
    enter through the supplied modulus and free thermal strain; no global
    material database is mutated.
    """
    if not layers:
        raise CableInputError("at least one layer is required")
    if len({layer.name for layer in layers}) != len(layers):
        raise CableInputError("layer names must be unique")
    numeric = [tension_N, torque_Nm, internal_pressure_Pa, external_pressure_Pa]
    if prescribed_twist_per_m is not None:
        numeric.append(prescribed_twist_per_m)
    if not np.isfinite(numeric).all():
        raise CableInputError("loads and prescribed twist must be finite")
    if prescribed_twist_per_m is not None and torque_Nm != 0.0:
        raise CableInputError("prescribe either torque or twist, not both")
    if min(internal_pressure_Pa, external_pressure_Pa) < 0:
        raise CableInputError("surface pressures must be non-negative")
    if (
        isinstance(quadrature_order, bool)
        or not isinstance(quadrature_order, int)
        or quadrature_order < 4
    ):
        raise CableInputError("at least four radial quadrature points are required")
    dofs = 2
    inner_ids, outer_ids = [], []
    for i, layer in enumerate(layers):
        a, b = layer.inner_radius_m, layer.outer_radius_m
        if not np.isfinite(
            [
                a,
                b,
                layer.youngs_modulus_Pa,
                layer.free_thermal_strain,
                getattr(layer, "free_transverse_strain", 0.0),
            ]
        ).all():
            raise CableInputError(
                "layer geometry, modulus and free strain must be finite"
            )
        if a < 0 or b <= a or layer.youngs_modulus_Pa <= 0:
            raise CableInputError(
                "positive thickness/modulus and non-negative bore required"
            )
        if i and a < layers[i - 1].outer_radius_m - 1.0e-12:
            raise CableInputError(
                "initial overlap is not supported; specify assembly prestrain"
            )
        if isinstance(layer, ElasticHelicalRing):
            angle = layer.signed_lay_angle_rad
            if not math.isfinite(angle) or not 1.0e-8 < abs(angle) < math.pi / 2:
                raise CableInputError(
                    "helical ring requires non-zero lay angle below pi/2"
                )
            if (
                isinstance(layer.wire_count, bool)
                or not isinstance(layer.wire_count, int)
                or layer.wire_count < 1
                or not math.isfinite(layer.initial_wire_strain)
            ):
                raise CableInputError(
                    "positive wire count and finite initial strain required"
                )
            inner_ids.append(dofs)
            outer_ids.append(dofs)
            dofs += 1
        else:
            if not -1.0 < layer.poisson_ratio < 0.5:
                raise CableInputError("Poisson ratio must lie between -1 and 0.5")
            inner_ids.append(dofs if a > 0 else -1)
            outer_ids.append(dofs + 1 if a > 0 else dofs)
            dofs += 2 if a > 0 else 1
    matrix, load = np.zeros((dofs, dofs)), np.zeros(dofs)
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    for i, layer in enumerate(layers):
        a, b, e = layer.inner_radius_m, layer.outer_radius_m, layer.youngs_modulus_Pa
        if isinstance(layer, ElasticHelicalRing):
            angle, r = layer.signed_lay_angle_rad, layer.helix_radius_m
            cosine, sine = math.cos(angle), math.sin(angle)
            strain_map = np.zeros(dofs)
            strain_map[0] = cosine**2
            strain_map[1] = r * cosine * sine
            strain_map[inner_ids[i]] = sine**2 / r
            rigidity = layer.total_area_m2 * e / cosine
            matrix += rigidity * np.outer(strain_map, strain_map)
            load += (
                rigidity
                * (layer.free_thermal_strain - layer.initial_wire_strain)
                * strain_map
            )
        else:
            constitutive = annulus_constitutive(
                e, layer.poisson_ratio, layer.carries_axial_load
            )
            # Rows of the strain map are radial, circumferential and axial.
            if a > 0:
                coeff = np.linalg.inv(np.array([[a, 1 / a], [b, 1 / b]]))
            for node, weight in zip(nodes, weights, strict=True):
                r = 0.5 * (a + b) + 0.5 * (b - a) * node
                strain_map = np.zeros((3, dofs))
                strain_map[2, 0] = 1.0
                if a > 0:
                    strain_map[0, [inner_ids[i], outer_ids[i]]] = (
                        np.array([1.0, -1 / r**2]) @ coeff
                    )
                    strain_map[1, [inner_ids[i], outer_ids[i]]] = (
                        np.array([1.0, 1 / r**2]) @ coeff
                    )
                else:
                    strain_map[0, outer_ids[i]] = 1 / b
                    strain_map[1, outer_ids[i]] = 1 / b
                volume_weight = 2 * math.pi * r * 0.5 * (b - a) * weight
                matrix += volume_weight * strain_map.T @ constitutive @ strain_map
                load += (
                    volume_weight
                    * strain_map.T
                    @ constitutive
                    @ layer.free_strain_vector
                )
            matrix[1, 1] += annulus_torsional_rigidity(layer)
    initial_load = load.copy()
    load[0] += tension_N
    load[1] += torque_Nm
    if inner_ids[0] >= 0:
        load[inner_ids[0]] += (
            2 * math.pi * layers[0].inner_radius_m * internal_pressure_Pa
        )
    elif internal_pressure_Pa:
        raise CableInputError("a solid core has no internal pressure surface")
    load[outer_ids[-1]] -= (
        2 * math.pi * layers[-1].outer_radius_m * external_pressure_Pa
    )
    contact_count = len(layers) - 1
    contact = np.zeros((contact_count, dofs))
    initial_gap = np.zeros(contact_count)
    interface_radius = np.zeros(contact_count)
    for i in range(contact_count):
        contact[i, outer_ids[i]] = -1.0
        contact[i, inner_ids[i + 1]] = 1.0
        initial_gap[i] = layers[i + 1].inner_radius_m - layers[i].outer_radius_m
        interface_radius[i] = 0.5 * (
            layers[i + 1].inner_radius_m + layers[i].outer_radius_m
        )
    free = (
        np.arange(dofs)
        if prescribed_twist_per_m is None
        else np.r_[0, np.arange(2, dofs)]
    )
    displacement = np.zeros(dofs)
    if prescribed_twist_per_m is not None:
        displacement[1] = prescribed_twist_per_m
    reduced = matrix[np.ix_(free, free)]
    if np.any(np.diag(reduced) <= 0):
        raise CableInputError("section has an unsupported elastic mechanism")
    scale = 1 / np.sqrt(np.diag(reduced))
    reduced_scaled = reduced * scale[:, None] * scale[None, :]
    try:
        np.linalg.cholesky(reduced_scaled)
    except np.linalg.LinAlgError as exc:
        raise CableInputError("section has an unsupported elastic mechanism") from exc
    rhs = (load - matrix @ displacement)[free]
    inverse_rhs = np.linalg.solve(reduced_scaled, rhs * scale) * scale
    response = (
        np.linalg.solve(reduced_scaled, contact[:, free].T * scale[:, None])
        * scale[:, None]
    )
    flexibility = contact[:, free] @ response
    free_gap = initial_gap + contact @ displacement + contact[:, free] @ inverse_rhs
    active = free_gap < 0.0
    reactions = np.zeros(contact_count)
    for _ in range(4 * contact_count + 10):
        reactions[:] = 0.0
        if np.any(active):
            reactions[active] = np.linalg.solve(
                flexibility[np.ix_(active, active)], -free_gap[active]
            )
        # An inactive interface carries exactly zero reaction. Do not export
        # small negative contact pressures as nominally admissible forces.
        negative = np.flatnonzero(active & (reactions < 0.0))
        if negative.size:
            active[negative[np.argmin(reactions[negative])]] = False
            continue
        gaps = free_gap + flexibility @ reactions
        penetrating = np.flatnonzero(~active & (gaps < -1.0e-12))
        if penetrating.size:
            active[penetrating[np.argmin(gaps[penetrating])]] = True
            continue
        break
    else:
        raise CableSolverError("radial contact active set did not converge")
    displacement[free] += inverse_rhs + response @ reactions
    residual = matrix @ displacement - load - contact.T @ reactions
    relative_residual = float(np.linalg.norm(residual[free] * scale)) / max(
        float(np.linalg.norm(rhs * scale)), 1.0
    )
    internal = matrix @ displacement - initial_load
    tangent_matrix = _condense_radial_tangent(matrix, contact, active)
    # These absolute tolerances identify a possible branch junction only;
    # they do not close a gap or change the contact solution or its tangent.
    junctions = tuple(
        int(i) for i in np.flatnonzero((np.abs(gaps) <= 1e-12) & (reactions <= 1e-8))
    )
    tangent_matrix.setflags(write=False)
    return ConcentricSectionResult(
        float(displacement[0]),
        float(displacement[1]),
        {
            layer.name: float(displacement[inner_ids[i]]) if inner_ids[i] >= 0 else 0.0
            for i, layer in enumerate(layers)
        },
        {
            layer.name: float(displacement[outer_ids[i]])
            for i, layer in enumerate(layers)
        },
        reactions / (2 * math.pi * interface_radius),
        reactions,
        initial_gap + contact @ displacement,
        float(internal[0]),
        float(internal[1]),
        float(relative_residual),
        float(np.max(np.abs(reactions * gaps))) if contact_count else 0.0,
        ConcentricTangent(tangent_matrix, tuple(bool(a) for a in active), junctions),
    )
