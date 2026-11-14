# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Small bending reference with coupled axial slip in concentric wire rings.

At physical azimuth phi, a ring's axial warping is a(z) cos(phi)+b(z) sin(phi).
Its wire strain divided by cos(alpha)^2 has harmonic coefficients
[r*kappa + a' + q*b, b' - q*a], q=tan(alpha)/r. Integrating wire axial energy
around a ring gives n*E*A*cos(alpha)^3/4 times the squared coefficient norm.
The cable-axis beam retains individual-wire and sheath bending separately.

Each cylindrical interface has ONE relative-warping history. With uniform
prescribed pressure p, integrating Coulomb work in azimuth gives
4*r*mu*p*norm(delta[a,b]) per axial metre. Action and reaction therefore
follow from the same potential. This first-harmonic, fixed-radius reduction
does not resolve discrete wire contacts or pressure redistribution. Optional
sheath warping uses a declared annular beam displacement mode. It imports no
CableXSec mechanics and is not a production core law. The optional axial-shear
mode is a diagnostic, not the radial shell freedom of Hu et al.'s HSHEAR363
element; its material constants alone do not make the models equivalent.

A Hermite beam matches three-point displacement loading; ring warping uses
linear interpolation. Ends have zero deflection and free rotation/warping.
The midpoint displacement is prescribed. A second-order-cone formulation
solves the convex incremental friction potential without contact penalties.
"""

# ruff: noqa: N815 -- SI symbols in result and material names

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import block_diag, csc_matrix, triu


@dataclass(frozen=True)
class ConductorRing:
    count: int
    radius_m: float
    angle_rad: float
    wire_diameter_m: float
    modulus_Pa: float

    def __post_init__(self):
        if (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or self.count < 1
        ):
            raise ValueError("positive integer wire count required")
        values = [self.radius_m, self.wire_diameter_m, self.modulus_Pa]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("positive finite ring geometry and modulus required")
        if not math.isfinite(self.angle_rad) or abs(self.angle_rad) >= math.pi / 2:
            raise ValueError("finite lay angle below pi/2 required")

    @property
    def harmonic_rigidity_N(self):
        area = math.pi * self.wire_diameter_m**2 / 4
        return self.count * self.modulus_Pa * area * math.cos(self.angle_rad) ** 3 / 2


@dataclass(frozen=True)
class AnnularSheath:
    """Annular axial/shear energy with warping measured at its inner radius.

    Extra axial displacement is (x*a+y*b)/ri. Axial bending energy is
    E*I/(2*ri²)*[(ri*kappa+a')²+b'²]; uniform transverse-shear energy is
    G*A/(2*ri²)*(a²+b²). No fitted shear factor or contact spring is used.
    This is an annular beam mode, not a resolved Saint-Venant warping field.
    """

    inner_radius_m: float
    outer_radius_m: float
    modulus_Pa: float
    poisson_ratio: float

    def __post_init__(self):
        values = [
            self.inner_radius_m,
            self.outer_radius_m,
            self.modulus_Pa,
            self.poisson_ratio,
        ]
        if (
            not np.isfinite(values).all()
            or not 0 < self.inner_radius_m < self.outer_radius_m
            or self.modulus_Pa <= 0
        ):
            raise ValueError(
                "finite ordered sheath radii and positive modulus required"
            )
        if not -1 < self.poisson_ratio < 0.5:
            raise ValueError("sheath Poisson ratio must lie between -1 and 0.5")

    @property
    def bending_Nm2(self):
        return (
            self.modulus_Pa
            * math.pi
            * (self.outer_radius_m**4 - self.inner_radius_m**4)
            / 4
        )

    @property
    def shear_N_m2(self):
        area = math.pi * (self.outer_radius_m**2 - self.inner_radius_m**2)
        shear_modulus = self.modulus_Pa / (2 * (1 + self.poisson_ratio))
        return shear_modulus * area / self.inner_radius_m**2


@dataclass(frozen=True)
class BendHistory:
    displacement_m: np.ndarray
    force_N: np.ndarray
    apparent_curvature_1_m: np.ndarray
    apparent_moment_Nm: np.ndarray
    elastic_energy_J: np.ndarray
    friction_work_J: np.ndarray
    endpoint_work_J: np.ndarray
    integration_remainder_J: np.ndarray
    work_residual_J: np.ndarray
    iterations: np.ndarray
    equilibrium_relative_residual: np.ndarray
    contact_work_gap_J: np.ndarray
    final_displacements: np.ndarray
    numerical_status: tuple[str, ...]


class ConcentricConductorBend:
    def __init__(
        self,
        *,
        rings,
        interface_radii_m,
        interface_friction,
        interface_pressure_Pa,
        individual_bending_Nm2,
        span_m,
        elements=24,
        quadrature=2,
        sheath=None,
    ):
        self.rings = tuple(rings)
        nr = len(self.rings)
        if not nr:
            raise ValueError("at least one wire ring required")
        for value, minimum in ((elements, 4), (quadrature, 2)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError("integer mesh and quadrature counts required")
        if elements % 2:
            raise ValueError("an even element count places a node at the loading point")
        parameters = [span_m, individual_bending_Nm2]
        if not np.isfinite(parameters).all() or min(parameters) <= 0:
            raise ValueError("positive finite span and bending rigidity required")
        radii, friction, pressure = (
            np.asarray(value, dtype=float)
            for value in (interface_radii_m, interface_friction, interface_pressure_Pa)
        )
        for value in (radii, friction, pressure):
            if value.shape != (nr + 1,) or not np.isfinite(value).all():
                raise ValueError("one finite input per shared interface required")
        if np.any(radii <= 0) or np.any(np.diff(radii) <= 0):
            raise ValueError("ordered positive interface radii required")
        if np.any(friction < 0) or np.any(pressure < 0):
            raise ValueError("nonnegative contact inputs required")
        ring_radii = np.array([ring.radius_m for ring in self.rings])
        if not np.all((radii[:-1] < ring_radii) & (ring_radii < radii[1:])):
            raise ValueError("each ring must lie between its bounding interfaces")
        if sheath is not None and (
            not isinstance(sheath, AnnularSheath)
            or not math.isclose(radii[-1], sheath.inner_radius_m, rel_tol=1e-12)
        ):
            raise ValueError("the elastic sheath must meet the outer contact interface")
        self.span = span_m
        self.sheath = sheath
        # The explicit sheath must not also be in the supplied backbone.
        self.individual_bending = individual_bending_Nm2 + (
            sheath.bending_Nm2 if sheath else 0.0
        )
        self.elements = elements
        nodes = elements + 1
        ndof = 2 * nodes * (nr + 1 + int(sheath is not None))
        beam_dofs = 2 * nodes
        stiffness = np.zeros((ndof, ndof))
        gauss, weights = np.polynomial.legendre.leggauss(quadrature)
        h = span_m / elements
        contacts, limits = [], []
        for element in range(elements):
            beam = np.array(
                [2 * element, 2 * element + 1, 2 * element + 2, 2 * element + 3]
            )
            for point, weight in zip((gauss + 1) / 2, weights * h / 2, strict=True):
                shape = np.array([1 - point, point])
                gradient = np.array([-1, 1]) / h
                curvature = (
                    np.array(
                        [
                            -6 + 12 * point,
                            h * (-4 + 6 * point),
                            6 - 12 * point,
                            h * (-2 + 6 * point),
                        ]
                    )
                    / h**2
                )
                stiffness[np.ix_(beam, beam)] += (
                    individual_bending_Nm2 * weight * np.outer(curvature, curvature)
                )
                warp_maps = []
                for j, ring in enumerate(self.rings):
                    offset = beam_dofs + 2 * nodes * j
                    cosine = offset + 2 * np.array([element, element + 1])
                    sine = cosine + 1
                    transport = math.tan(ring.angle_rad) / ring.radius_m
                    harmonic = np.zeros((2, ndof))
                    harmonic[0, beam] = ring.radius_m * curvature
                    harmonic[0, cosine] = gradient
                    harmonic[0, sine] = transport * shape
                    harmonic[1, sine] = gradient
                    harmonic[1, cosine] = -transport * shape
                    stiffness += (
                        ring.harmonic_rigidity_N * weight * harmonic.T @ harmonic
                    )
                    warp = np.zeros((2, ndof))
                    warp[0, cosine] = shape
                    warp[1, sine] = shape
                    warp_maps.append(warp)
                # Central-wire bending is in the backbone. The optional
                # sheath has its own axial/shear energy and contact motion.
                fixed = np.zeros((2, ndof))
                outer = fixed
                if sheath is not None:
                    offset = beam_dofs + 2 * nodes * nr
                    cosine = offset + 2 * np.array([element, element + 1])
                    sine = cosine + 1
                    radius = sheath.inner_radius_m
                    axial = np.zeros((2, ndof))
                    axial[0, beam] = radius * curvature
                    axial[0, cosine] = gradient
                    axial[1, sine] = gradient
                    outer = np.zeros((2, ndof))
                    outer[0, cosine] = shape
                    outer[1, sine] = shape
                    stiffness += weight * (
                        sheath.bending_Nm2 / radius**2 * axial.T @ axial
                        + sheath.shear_N_m2 * outer.T @ outer
                    )
                bounded = [fixed, *warp_maps, outer]
                for j in range(nr + 1):
                    contacts.append(bounded[j + 1] - bounded[j])
                    limits.append(4 * radii[j] * friction[j] * pressure[j] * weight)
        self.stiffness = stiffness
        self.contact = np.stack(contacts)
        self.limits = np.asarray(limits)
        self.loading_dof = elements
        fixed_dofs = [0, 2 * elements, self.loading_dof]
        self.free = np.setdiff1d(np.arange(ndof), fixed_dofs)
        self.ndof = ndof
        kff = stiffness[np.ix_(self.free, self.free)]
        # Diagonal change of coordinates only; no stiffness regularisation.
        self.scale = 1 / np.sqrt(np.diag(kff))
        self.scaled_stiffness = self.scale[:, None] * kff * self.scale[None, :]

    def run(self, displacement_m, *, tolerance=1e-7, maximum_iterations=200):
        imposed = np.asarray(displacement_m, dtype=float)
        if imposed.ndim != 1 or not len(imposed) or not np.isfinite(imposed).all():
            raise ValueError("finite nonempty displacement history required")
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("positive finite tolerance required")
        if (
            isinstance(maximum_iterations, bool)
            or not isinstance(maximum_iterations, int)
            or maximum_iterations < 1
        ):
            raise ValueError("positive integer iteration limit required")
        import clarabel

        active = np.flatnonzero(self.limits > 0)
        cone_scale = 1e-4
        nf, nc = len(self.free), len(active)
        contact_rows = self.contact[active][:, :, self.free]
        matrix = np.zeros((3 * nc, nf + nc))
        for i in range(nc):
            matrix[3 * i, nf + i] = -1
            matrix[3 * i + 1 : 3 * i + 3, :nf] = (
                -contact_rows[i] * self.scale / cone_scale
            )
        quadratic = triu(
            block_diag(
                (csc_matrix(self.scaled_stiffness), csc_matrix((nc, nc))), format="csc"
            ),
            format="csc",
        )
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_threads = 1
        settings.max_iter = maximum_iterations
        settings.tol_gap_abs = settings.tol_gap_rel = min(1e-12, tolerance * 1e-6)
        settings.tol_feas = min(1e-12, tolerance * 1e-6)
        cones = [clarabel.SecondOrderConeT(3) for _ in active]
        linear = np.r_[np.zeros(nf), self.limits[active] * cone_scale]
        solver = (
            clarabel.DefaultSolver(
                quadratic, linear, csc_matrix(matrix), np.zeros(3 * nc), cones, settings
            )
            if nc
            else None
        )
        d = np.zeros(self.ndof)
        previous = d.copy()
        old_relative = np.zeros((len(self.limits), 2))
        results = np.zeros((len(imposed), 9))
        counts = np.zeros(len(imposed), dtype=int)
        statuses = []
        cumulative_work = cumulative_loss = cumulative_remainder = 0.0
        previous_w = 0.0
        for step, w in enumerate(imposed):
            # Solve an increment about the preceding accepted displacement.
            # This removes the large cancelling quadratic constant associated
            # with prescribing the entire midpoint displacement in one node.
            prescribed = previous.copy()
            prescribed[self.loading_dof] = w
            offset = self.contact @ prescribed - old_relative
            force_offset = (self.stiffness @ prescribed)[self.free]
            d = prescribed.copy()
            traction = np.zeros_like(old_relative)
            if solver is None:
                d[self.free] += (
                    self.scale
                    * np.linalg.lstsq(
                        self.scaled_stiffness, -self.scale * force_offset, rcond=1e-13
                    )[0]
                )
                iteration = 1
                status = "LinearSolution"
            else:
                rhs = np.column_stack(
                    (np.zeros(nc), offset[active] / cone_scale)
                ).ravel()
                solver.update(
                    q=np.r_[
                        self.scale * force_offset, self.limits[active] * cone_scale
                    ],
                    b=rhs,
                )
                solution = solver.solve()
                # Solver labels alone do not establish a mechanical solution.
                # A precision-limited candidate still has to satisfy every
                # original equilibrium, yield and maximum-dissipation check.
                if solution.status not in (
                    clarabel.SolverStatus.Solved,
                    clarabel.SolverStatus.AlmostSolved,
                ):
                    raise RuntimeError(f"conductor step {step}: {solution.status}")
                status = str(solution.status)
                d[self.free] += self.scale * np.asarray(solution.x[:nf])
                traction[active] = (
                    -np.asarray(solution.z).reshape(nc, 3)[:, 1:] / cone_scale
                )
                iteration = solution.iterations
            actual = self.contact @ d - old_relative
            residual = self.stiffness @ d + np.einsum(
                "qid,qi->d", self.contact, traction
            )
            force_scale = max(
                abs(residual[self.loading_dof]),
                48 * self.individual_bending * abs(w) / self.span**3,
                1.0,
            )
            # Beam rotational residuals are moments; divide by span
            # before comparing with displacement-force residuals.
            force_residual = residual.copy()
            force_residual[1 : 2 * (self.elements + 1) : 2] /= self.span
            eq = float(np.linalg.norm(force_residual[self.free]) / force_scale)
            gap = float(
                np.sum(
                    self.limits * np.linalg.norm(actual, axis=1)
                    - np.sum(traction * actual, axis=1)
                )
            )
            excess = max(
                float(np.max(np.linalg.norm(traction, axis=1) - self.limits)), 0.0
            )
            work_scale = max(
                abs(residual[self.loading_dof] * (w - previous_w)),
                abs(0.5 * d @ self.stiffness @ d),
                1e-6,
            )
            if (
                eq > tolerance
                or abs(gap) > tolerance * work_scale
                or excess > tolerance * max(float(self.limits.max()), 1.0)
            ):
                raise RuntimeError(
                    f"conductor step {step} failed mechanical checks: equilibrium={eq:.3g}, "
                    f"contact work gap={gap:.3g} J, yield excess={excess:.3g} N"
                )
            force = residual[self.loading_dof]
            relative = self.contact @ d
            cumulative_loss += float(
                np.sum(self.limits * np.linalg.norm(relative - old_relative, axis=1))
            )
            cumulative_work += force * (w - previous_w)
            change = d - previous
            cumulative_remainder += 0.5 * change @ self.stiffness @ change
            energy = 0.5 * d @ self.stiffness @ d
            balance = cumulative_work - energy - cumulative_loss - cumulative_remainder
            results[step] = [
                force,
                energy,
                cumulative_loss,
                cumulative_work,
                cumulative_remainder,
                balance,
                eq,
                gap,
                excess,
            ]
            counts[step] = iteration
            statuses.append(status)
            old_relative, previous, previous_w = relative.copy(), d.copy(), float(w)
        return BendHistory(
            imposed.copy(),
            results[:, 0],
            12 * imposed / self.span**2,
            results[:, 0] * self.span / 4,
            results[:, 1],
            results[:, 2],
            results[:, 3],
            results[:, 4],
            results[:, 5],
            counts,
            results[:, 6],
            results[:, 7],
            d.copy(),
            tuple(statuses),
        )
