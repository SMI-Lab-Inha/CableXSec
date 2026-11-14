# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Three-core (trefoil) radial foundation-reaction approximation.

This is a ring on prescribed support, not a resolved cable contact model.
Its foundation reaction is on the supported face; transferring that shape
to armour pressure on the loaded face is a separate approximation requiring
independent assessment. Circumferential membrane and core-bundle slip are
not solved here.

The radial equation is EI_ring/R^4 (d²/dphi²+1)²u + k(phi)max(u,0) = q0.
The prescribed binary foundation has three equally spaced support sectors.
Its full angular width is theta_core, not theta_core/2. A cosine basis in
the core-aligned coordinate contains modes 0,3,...,max_harmonic. Reflection
symmetry removes sine terms in that coordinate only. Product coefficients
through twice the largest retained harmonic are integrated analytically.

The returned shape is k(phi)max(u(phi),0) divided by its mean over the requested
wire positions. It is not an absolute normal force. Mean equilibrium
includes the ring's constant-mode reaction as well as foundation pressure;
both are reported. Uniform support gives a uniform pressure shape.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from cablexsec.exceptions import CableInputError, CableSolverError

__all__ = ["ThreeCoreResult", "foundation_stiffness", "solve_three_core"]

FloatArray = NDArray[np.float64]

# Galerkin harmonics retained (trefoil symmetry → multiples of three).
_DEFAULT_MAX_HARMONIC = 36


def foundation_stiffness(
    phi: float,
    k_core: float,
    k_filler: float,
    theta_core: float,
    phi_cores: tuple[float, ...],
) -> float:
    """Binary radial foundation stiffness ``k(phi)`` [N/m^3].

    ``k_core`` if ``phi`` is within ``theta_core/2`` of any core centre,
    otherwise ``k_filler``.
    """
    for pc in phi_cores:
        d = abs(((phi - pc + math.pi) % (2.0 * math.pi)) - math.pi)
        if d <= 0.5 * theta_core:
            return k_core
    return k_filler


@dataclass(frozen=True)
class ThreeCoreResult:
    """Output of the inner radial-foundation calculation.

    Attributes
    ----------
    phi_wires_rad : numpy.ndarray
        Wire azimuths [rad].
    shape_g : numpy.ndarray
        Dimensionless INNER support-reaction shape, normalised so that
        ``mean_k(g) == 1``. This is not the prescribed uniform outer pressure
        and is not automatically transferred to the M-kappa engine. Uniform
        foundation gives a uniform shape.
    k0, k3, k6 : float
        Core-aligned Fourier coefficients of foundation stiffness [N/m^3].
    u0, u3, u6 : float
        First three core-aligned radial-deflection amplitudes [m]. Higher
        retained modes are in displacement_coefficients_m.
    balance_error : float
        Relative mean equilibrium residual including ring and foundation.
    """

    phi_wires_rad: FloatArray
    shape_g: FloatArray
    k0: float
    k3: float
    k6: float
    u0: float
    u3: float
    u6: float
    balance_error: float
    phase_rad: float = 0.0
    harmonic_orders: tuple[int, ...] = (0, 3, 6)
    displacement_coefficients_m: FloatArray | None = None
    foundation_mean_pressure_Pa: float = 0.0
    ring_mean_reaction_Pa: float = 0.0
    contact_fraction: float = 1.0
    galerkin_residual: float = 0.0
    harmonic_change: float | None = None

    def radial_displacement(self, phi_rad: FloatArray) -> FloatArray:
        """Evaluate the retained harmonics in the core-aligned coordinate."""
        coefficients = self.displacement_coefficients_m
        if coefficients is None:
            coefficients = np.array([self.u0, self.u3, self.u6])
        return np.asarray(
            np.cos(
                (np.asarray(phi_rad)[..., None] - self.phase_rad)
                * np.asarray(self.harmonic_orders)
            )
            @ coefficients,
            dtype=np.float64,
        )


def _cos_projection(
    k_core: float,
    k_filler: float,
    theta_core: float,
    phi_cores: tuple[float, ...],
    n: int,
) -> float:
    """``(1/pi) integral_0^{2pi} k(phi) cos(n phi) dphi`` (``2pi`` for n=0)."""
    norm = 2.0 * math.pi if n == 0 else math.pi

    # Split at every discontinuity and integrate the constant-foundation times
    # cosine analytically. This avoids quadrature warnings at binary steps and
    # makes the Galerkin coefficients deterministic to floating-point roundoff.
    pts = sorted(
        {
            (pc + s * 0.5 * theta_core) % (2.0 * math.pi)
            for pc in phi_cores
            for s in (-1.0, 1.0)
        }
    )
    val = 0.0
    edges = [0.0, *pts, 2.0 * math.pi]
    for lo, hi in zip(edges, edges[1:], strict=False):
        if hi > lo:
            stiffness = foundation_stiffness(
                0.5 * (lo + hi), k_core, k_filler, theta_core, phi_cores
            )
            if n == 0:
                val += stiffness * (hi - lo)
            else:
                val += stiffness * (math.sin(n * hi) - math.sin(n * lo)) / n
    return val / norm


def _active_foundation_matrix(
    displacement: FloatArray,
    k_core: float,
    k_filler: float,
    theta_core: float,
) -> tuple[FloatArray, float]:
    """Integrate the compression-only support on its actual contact arcs.

    Set theta=3*phi on one reflection half-period, 0 <= theta <= pi.
    The displacement is a Chebyshev polynomial in cos(theta). Its real roots
    and the support boundary split the integral; no clipping of a previously
    solved pressure field or wire-grid contact quadrature is used.
    """
    roots = np.polynomial.chebyshev.chebroots(displacement)
    real = roots.real[(abs(roots.imag) < 1e-10) & (abs(roots.real) < 1)]
    edge = min(1.5 * theta_core, math.pi)
    boundaries = sorted({0.0, edge, math.pi, *np.arccos(real)})
    size = len(displacement)
    order = np.arange(2 * size - 1)
    integrals = np.zeros(len(order))
    contact_length = 0.0
    for lo, hi in zip(boundaries, boundaries[1:], strict=False):
        midpoint = 0.5 * (lo + hi)
        if np.polynomial.chebyshev.chebval(math.cos(midpoint), displacement) <= 0:
            continue
        contact_length += hi - lo
        stiffness = k_core if midpoint < edge else k_filler
        integrals[0] += stiffness * (hi - lo)
        integrals[1:] += (
            stiffness * (np.sin(order[1:] * hi) - np.sin(order[1:] * lo)) / order[1:]
        )
    indices = np.arange(size)
    matrix = 0.5 * (
        integrals[indices[:, None] + indices]
        + integrals[abs(indices[:, None] - indices)]
    )
    norms = np.full(size, math.pi / 2)
    norms[0] = math.pi
    return matrix / norms[:, None], contact_length / math.pi


def _compression_only_solution(
    bilateral: FloatArray,
    ring_diagonal: FloatArray,
    k_core: float,
    k_filler: float,
    theta_core: float,
    scale: float,
) -> tuple[FloatArray, FloatArray, float, float]:
    """Minimise the strictly convex ring plus unilateral-foundation energy."""
    displacement = bilateral * scale
    diagonal = ring_diagonal / scale
    norms = np.ones(len(displacement))
    norms[0] = 2.0
    load = np.zeros(len(displacement))
    load[0] = 1.0

    def state(values: FloatArray) -> tuple[FloatArray, float, float, FloatArray]:
        foundation, fraction = _active_foundation_matrix(
            values, k_core / scale, k_filler / scale, theta_core
        )
        tangent = foundation + np.diag(diagonal)
        residual = tangent @ values - load
        potential = float(0.5 * values @ (norms * (tangent @ values)) - 2 * values[0])
        return tangent, fraction, potential, residual

    for _ in range(50):
        tangent, fraction, potential, residual = state(displacement)
        error = float(np.max(abs(residual)))
        if error < 1e-10:
            foundation = (tangent - np.diag(diagonal)) * scale
            return displacement / scale, foundation, fraction, error
        direction = np.linalg.solve(tangent, -residual)
        slope = float(direction @ (norms * residual))
        step = 1.0
        for _ in range(30):
            trial = displacement + step * direction
            _, _, next_potential, next_residual = state(trial)
            # Residual descent handles the final roundoff-sized energy change.
            if next_potential <= potential + 1e-4 * step * slope or (
                np.max(abs(next_residual)) < 0.5 * error
            ):
                displacement = trial
                break
            step *= 0.5
        else:
            raise CableSolverError(
                "compression-only ring equilibrium line search failed"
            )
    raise CableSolverError("compression-only ring equilibrium did not converge")


def solve_three_core(
    q0_Pa: float,
    *,
    k_core: float,
    k_filler: float,
    theta_core_rad: float,
    EI_ring: float,
    r_bedding_m: float,
    n_wires: int,
    phi_cores: tuple[float, ...] = (
        math.pi / 6.0,
        math.pi / 6.0 + 2.0 * math.pi / 3.0,
        math.pi / 6.0 + 4.0 * math.pi / 3.0,
    ),
    max_harmonic: int | None = None,
) -> ThreeCoreResult:
    """Solve a convergent core-aligned Galerkin family for the support shape.

    Parameters
    ----------
    q0_Pa : float
        Uniform applied armour contact pressure (from the axisymmetric step) [Pa].
        Only its sign/magnitude scaling matters; the returned ``g`` is
        normalised to mean 1 and is independent of ``q0``.
    k_core, k_filler : float
        Foundation stiffness over a core / over a filler void [N/m^3].
    theta_core_rad : float
        Angular full-width subtended by a core, ``2 arcsin(r_core_outer/r_trefoil)``.
    EI_ring : float
        Bending stiffness of the bedding ring per axial metre [N m].
    r_bedding_m : float
        Mean bedding-ring radius [m].
    n_wires : int
        Number of armour wires (defines the azimuthal sampling).
    phi_cores : tuple of float, optional
        Three equally spaced core-centre azimuths [rad].
    max_harmonic : int or None
        Largest retained harmonic, a multiple of three of at least six.
        With None, double the order until wire reaction, mean displacement
        and contact fraction change by less than 0.5%, up to harmonic 1152.

    Returns
    -------
    ThreeCoreResult
    """
    if not np.isfinite(
        [q0_Pa, k_core, k_filler, theta_core_rad, EI_ring, r_bedding_m]
    ).all():
        raise CableInputError("ring/foundation inputs must be finite")
    if min(k_core, k_filler, r_bedding_m) <= 0 or q0_Pa < 0 or EI_ring <= 0:
        raise CableInputError(
            "positive support, ring rigidity and radius; nonnegative pressure required"
        )
    if not 0 < theta_core_rad <= 2 * math.pi:
        raise CableInputError("core angular full width must lie between zero and 2pi")
    if isinstance(n_wires, bool) or not isinstance(n_wires, int) or n_wires < 1:
        raise CableInputError("n_wires must be a positive integer")
    if max_harmonic is None:

        def at_order(order: int) -> ThreeCoreResult:
            return solve_three_core(
                q0_Pa,
                k_core=k_core,
                k_filler=k_filler,
                theta_core_rad=theta_core_rad,
                EI_ring=EI_ring,
                r_bedding_m=r_bedding_m,
                n_wires=n_wires,
                phi_cores=phi_cores,
                max_harmonic=order,
            )

        previous = at_order(_DEFAULT_MAX_HARMONIC)
        for order in (72, 144, 288, 576, 1152):
            current = at_order(order)
            change = max(
                float(np.max(abs(current.shape_g - previous.shape_g)))
                / max(1.0, float(np.max(current.shape_g))),
                abs(current.u0 - previous.u0) / max(abs(current.u0), 1e-300),
                abs(current.contact_fraction - previous.contact_fraction),
            )
            if change < 0.005:
                return replace(current, harmonic_change=change)
            previous = current
        raise CableSolverError("ring/contact harmonic refinement did not reach 0.5%")
    if (
        isinstance(max_harmonic, bool)
        or not isinstance(max_harmonic, int)
        or max_harmonic < 6
        or max_harmonic % 3
    ):
        raise CableInputError("max_harmonic must be a multiple of three, at least six")
    if len(phi_cores) != 3 or not np.isfinite(phi_cores).all():
        raise CableInputError("three finite core-centre angles required")
    centres = np.sort(np.mod(phi_cores, 2 * math.pi))
    spacing = np.diff(np.r_[centres, centres[0] + 2 * math.pi])
    if not np.allclose(spacing, 2 * math.pi / 3, rtol=0, atol=1e-10):
        raise CableInputError("the trefoil basis requires equally spaced cores")
    phase = float(centres[0])
    aligned_centres = tuple(2 * math.pi * k / 3 for k in range(3))
    k0 = _cos_projection(k_core, k_filler, theta_core_rad, aligned_centres, 0)
    k3 = _cos_projection(k_core, k_filler, theta_core_rad, aligned_centres, 3)
    k6 = _cos_projection(k_core, k_filler, theta_core_rad, aligned_centres, 6)

    # Foundation-reaction matrix in the core-aligned cosine basis:
    #   Kf[i,j] = (1/N_i) ∫ k(φ) cos(n_j φ) cos(n_i φ) dφ.
    # Products cos(n_j)cos(n_i) = ½[cos((i+j))+cos((i-j))], so each entry is a
    # combination of foundation coefficients through twice the largest mode.
    n = tuple(range(0, max_harmonic + 1, 3))
    kf = np.zeros((len(n), len(n)), dtype=np.float64)
    kcoef = {
        0: k0,
        3: k3,
        6: k6,
    }
    kcoef.update(
        {
            order: _cos_projection(
                k_core, k_filler, theta_core_rad, aligned_centres, order
            )
            for order in range(9, 2 * max_harmonic + 1, 3)
        }
    )
    norm = {order: 2.0 * math.pi if order == 0 else math.pi for order in n}
    for i, ni in enumerate(n):
        for j, nj in enumerate(n):
            # cos(nj)cos(ni) = 1/2[cos(ni+nj) + cos(|ni-nj|)]; project each term.
            s, d = ni + nj, abs(ni - nj)

            def term(p: int) -> float:
                # ∫_0^{2π} k(φ) cos(pφ) dφ = (π or 2π) * kcoef[p]
                base = 2.0 * math.pi if p == 0 else math.pi
                return base * kcoef.get(p, 0.0)

            integral = 0.5 * (term(s) + term(d))
            kf[i, j] = integral / norm[ni]

    d_diag = np.diag([EI_ring / r_bedding_m**4 * (nn**2 - 1) ** 2 for nn in n]).astype(
        np.float64
    )
    q_vec = np.zeros(len(n))
    # Solve unit pressure so the zero-pressure limit has a defined shape.
    q_vec[0] = 1.0
    u_vec = np.asarray(np.linalg.solve(d_diag + kf, q_vec), dtype=np.float64)
    u_vec, active_kf, contact_fraction, residual = _compression_only_solution(
        u_vec, np.diag(d_diag), k_core, k_filler, theta_core_rad, k0
    )
    u0, u3, u6 = (float(x * q0_Pa) for x in u_vec[:3])

    phi_wires = 2.0 * np.pi * np.arange(n_wires) / n_wires

    def k_of(phi: FloatArray) -> FloatArray:
        return np.array(
            [
                foundation_stiffness(p, k_core, k_filler, theta_core_rad, phi_cores)
                for p in phi
            ],
            dtype=np.float64,
        )

    def u_of(phi: FloatArray) -> FloatArray:
        return np.asarray(
            np.cos((phi[:, None] - phase) * np.asarray(n)) @ u_vec, dtype=np.float64
        )

    q_contact = k_of(phi_wires) * np.maximum(u_of(phi_wires), 0.0)
    # Dimensionless azimuthal shape, normalised to mean 1 over the wire set.
    q_mean_wires = float(np.mean(q_contact))
    if q_mean_wires != 0.0:
        shape_g = q_contact / q_mean_wires
    else:
        shape_g = np.ones_like(q_contact)

    # Exact mean equilibrium includes both foundation and ring reactions.
    mean_contact = float(active_kf[0] @ u_vec)
    ring_reaction = d_diag[0, 0] * u_vec[0]
    balance_error = abs(mean_contact + ring_reaction - 1.0)

    return ThreeCoreResult(
        phi_wires_rad=phi_wires,
        shape_g=shape_g,
        k0=k0,
        k3=k3,
        k6=k6,
        u0=u0,
        u3=u3,
        u6=u6,
        balance_error=balance_error,
        phase_rad=phase,
        harmonic_orders=n,
        displacement_coefficients_m=u_vec * q0_Pa,
        foundation_mean_pressure_Pa=float(mean_contact * q0_Pa),
        ring_mean_reaction_Pa=float(ring_reaction * q0_Pa),
        contact_fraction=contact_fraction,
        galerkin_residual=residual,
    )
