# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Helical rod on elastic support with two-component Coulomb friction.

Extends :mod:`periodic_friction_cell` (longitudinal friction only) to
a frictional support that also resists motion transverse to the wire, in the
cylinder surface. The longitudinal warping u(s) is solved from rod
equilibrium as before. The transverse direction is treated in one of three
ways:

``"none"``
    No transverse load or motion; reproduces the one-component cell.
``"prescribed"``
    A transverse displacement history is imposed at every station, as in the
    assumed contact map of the companion article (Eq. 2.15). The 2D slider is
    displacement controlled in both directions and returns radially onto the
    Coulomb circle.
``"equilibrium"``
    No transverse displacement is imposed. The support must instead carry the
    transverse line load that keeps the wire on its loxodromic path,

        q_t = (N + E I_w q^2) * dk_g,   dk_g = kappa cos(a) (1 + sin^2 a) sin(phi),

    where N is the current wire axial force, E I_w the wire bending stiffness
    about its radial axis, q = sin(a)/r the helix wavenumber and dk_g the
    in-surface (geodesic) curvature change of the loxodrome, the X3 term of
    the local-bending stress in Eq. (2.8). The transverse traction is fixed at
    q_t (load control) and the longitudinal friction limit at each station is
    reduced to sqrt(c^2 - q_t^2). Transverse plastic slip follows the
    associated flow rule, dp_t = dp_l q_t / t_l while sliding. The coupling
    with N is resolved by fixed-point iteration within each increment.

Assumptions: planar bending, uniform normal load and support stiffness,
isotropic tangential support stiffness, small strain, reflection symmetry over
a half-turn (as in the one-component cell). The transverse equilibrium
neglects any restraint from neighbouring wires in the same layer. This is a
reference calculation for the same construction as the section law, not a
resolved three-dimensional contact solution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_banded

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TransverseCellResult:
    curvature_1_m: FloatArray
    moment_Nm: FloatArray  # noqa: N815
    friction_J_m: FloatArray  # noqa: N815 -- total, both components
    friction_longitudinal_J_m: FloatArray  # noqa: N815
    friction_transverse_J_m: FloatArray  # noqa: N815
    stored_J_m: FloatArray  # noqa: N815 -- rod plus support elastic energy
    endpoint_work_J_m: FloatArray  # noqa: N815
    incremental_remainder_J_m: FloatArray  # noqa: N815
    demand_ratio_max: FloatArray  # max |q_t| / c over stations, per increment
    longitudinal_plastic_m: FloatArray  # (increments, elements, 4)
    transverse_plastic_m: FloatArray  # (increments, elements, 4)
    phi_rad: FloatArray


def solve_transverse_cell(
    *,
    axial_rigidity_N: float,
    bending_rigidity_Nm2: float,
    helix_radius_m: float,
    lay_angle_rad: float,
    wire_count: int,
    support_N_m2: float,
    friction_limit_N_m: float,
    curvature_1_m: FloatArray,
    mode: str = "equilibrium",
    eta: float = 0.0,
    tension_per_wire_N: float = 0.0,
    elements: int = 64,
    tolerance: float = 1e-10,
    maximum_iterations: int = 80,
    maximum_fixed_point: int = 50,
    geodesic_feedback: bool = False,
    feedback_modes: int = 3,
) -> TransverseCellResult:
    """Solve the two-component cell.

    With ``geodesic_feedback`` the accumulated transverse plastic slip w(s)
    of the wire relative to its support reduces its in-surface curvature
    change: dk_g,total = kappa c_g sin(phi) + w''(s). The transverse demand
    then is q_t = N dk_g,total - E I_w (dk_g,total)'', evaluated with a sine
    series of w over the half-turn (w = 0 at the reflection-symmetry ends).
    The wire is therefore attracted to, and cannot ratchet beyond, the
    geodesic of the bent cylinder. Without it the demand is evaluated on the
    original loxodromic path at every increment.
    """
    if mode not in ("none", "prescribed", "equilibrium"):
        raise ValueError("mode must be 'none', 'prescribed' or 'equilibrium'")
    scalars = [
        axial_rigidity_N,
        bending_rigidity_Nm2,
        helix_radius_m,
        lay_angle_rad,
        support_N_m2,
        friction_limit_N_m,
        eta,
        tension_per_wire_N,
    ]
    if not np.isfinite(scalars).all():
        raise ValueError("finite inputs are required")
    if axial_rigidity_N <= 0 or helix_radius_m <= 0 or support_N_m2 <= 0:
        raise ValueError("positive rigidity, radius and support stiffness required")
    if friction_limit_N_m <= 0 or bending_rigidity_Nm2 < 0 or eta < 0:
        raise ValueError("positive friction, nonnegative EI and eta required")
    curvature = np.asarray(curvature_1_m, dtype=float)
    ea, stiffness, cap0 = axial_rigidity_N, support_N_m2, friction_limit_N_m
    alpha = abs(lay_angle_rad)
    cosine, sine = math.cos(alpha), math.sin(alpha)
    q = sine / helix_radius_m
    half_length = math.pi * helix_radius_m / sine
    spacing = half_length / elements
    gauss, gauss_weights = np.polynomial.legendre.leggauss(4)
    shape = np.column_stack(((1 - gauss) / 2, (1 + gauss) / 2))
    weights = gauss_weights * spacing / 2
    gradient = np.array([-1.0, 1.0]) / spacing
    phi = (np.arange(elements)[:, None] + (gauss[None, :] + 1) / 2) * math.pi / elements
    reference = helix_radius_m * cosine**2 * np.cos(phi)
    geodesic = cosine * (1 + sine**2) * np.sin(phi)  # dk_g per unit kappa
    s_delta = helix_radius_m**2 * cosine**2 / sine
    transverse_map = s_delta * np.cos(phi)  # Eq. 2.15, kappa_y = 0
    factor = wire_count / (half_length * cosine)

    u = np.zeros(elements + 1)
    plastic = np.zeros((2, elements, 4))
    old_strain = np.zeros((elements, 4))
    old_elastic = np.zeros((2, elements, 4))
    n = len(curvature)
    moment = np.zeros(n)
    friction = np.zeros(n)
    friction_l = np.zeros(n)
    friction_t = np.zeros(n)
    stored = np.zeros(n)
    endpoint = np.zeros(n)
    remainder = np.zeros(n)
    demand = np.zeros(n)
    history_pl = np.zeros((n, elements, 4))
    history_pt = np.zeros((n, elements, 4))
    cum_f = cum_fl = cum_ft = cum_w = cum_r = previous_kappa = 0.0

    def assemble(displacement, imposed, transverse_slip, q_t, cap_l, previous):
        nodal = np.column_stack((displacement[:-1], displacement[1:]))
        strain = imposed + (nodal @ gradient)[:, None]
        slip_l = nodal @ shape.T
        if mode == "prescribed":
            trial_l = stiffness * (slip_l - previous[0])
            trial_t = stiffness * (transverse_slip - previous[1])
            norm = np.hypot(trial_l, trial_t)
            scale = np.where(norm > cap0, cap0 / np.maximum(norm, 1e-300), 1.0)
            t_l, t_t = trial_l * scale, trial_t * scale
            plastic_state = norm > cap0
            n_l = np.where(plastic_state, trial_l / np.maximum(norm, 1e-300), 0.0)
            tangent = np.where(
                plastic_state, stiffness * scale * (1 - n_l**2), stiffness
            )
            elastic = np.stack((t_l, t_t)) / stiffness
            slip = np.stack((slip_l, transverse_slip))
            dp = slip - previous - elastic
            friction_potential = 0.5 * (t_l**2 + t_t**2) / stiffness + cap0 * np.hypot(
                dp[0], dp[1]
            )
        else:
            trial_l = stiffness * (slip_l - previous[0])
            t_l = np.clip(trial_l, -cap_l, cap_l)
            tangent = stiffness * (np.abs(trial_l) <= cap_l)
            t_t = q_t
            elastic = np.stack((t_l / stiffness, t_t / stiffness))
            dp_l = slip_l - previous[0] - elastic[0]
            friction_potential = 0.5 * t_l**2 / stiffness + cap_l * np.abs(dp_l)
        local_force = ea * (strain @ weights)[:, None] * gradient[None, :] + np.einsum(
            "eq,q,qi->ei", t_l, weights, shape
        )
        residual = np.zeros(elements + 1)
        residual[:-1] += local_force[:, 0]
        residual[1:] += local_force[:, 1]
        local_stiffness = np.einsum("eq,q,qi,qj->eij", tangent, weights, shape, shape)
        local_stiffness += ea * spacing * np.outer(gradient, gradient)[None, :, :]
        diagonal = local_stiffness[:-1, 1, 1] + local_stiffness[1:, 0, 0]
        off = local_stiffness[1:-1, 0, 1]
        banded = np.zeros((3, elements - 1))
        banded[1] = diagonal
        banded[0, 1:] = off
        banded[2, :-1] = off
        potential = float(np.sum((0.5 * ea * strain**2 + friction_potential) @ weights))
        return residual, banded, potential, strain, slip_l, t_l, t_t, elastic

    def solve_increment(imposed, transverse_slip, q_t, cap_l, start):
        disp = start.copy()
        initial_force = ea * (imposed @ weights)[:, None] * gradient[None, :]
        load = initial_force[:-1, 1] + initial_force[1:, 0]
        scale = max(float(np.linalg.norm(load)), 1.0)
        for _ in range(maximum_iterations):
            fields = assemble(disp, imposed, transverse_slip, q_t, cap_l, plastic)
            residual, banded, potential = fields[:3]
            if float(np.linalg.norm(residual[1:-1])) / scale <= tolerance:
                return disp, fields
            increment = solve_banded(
                (1, 1), banded, -residual[1:-1], check_finite=False
            )
            directional = float(residual[1:-1] @ increment)
            step_length = 1.0
            for _ in range(50):
                candidate = disp.copy()
                candidate[1:-1] += step_length * increment
                trial = assemble(
                    candidate, imposed, transverse_slip, q_t, cap_l, plastic
                )
                roundoff = (
                    32 * np.finfo(float).eps * max(abs(potential), abs(trial[2]), 1.0)
                )
                if trial[2] <= potential + 1e-4 * step_length * directional or (
                    abs(trial[2] - potential) <= roundoff
                    and np.linalg.norm(trial[0][1:-1])
                    < 0.5 * np.linalg.norm(residual[1:-1])
                ):
                    disp = candidate
                    break
                step_length *= 0.5
            else:
                raise RuntimeError("line search failed")
        raise RuntimeError("equilibrium did not converge")

    zeros = np.zeros((elements, 4))
    # Sine basis over the half-turn for the drift feedback, evaluated at the
    # Gauss stations: w = sum b_n sin(n phi); d/ds = q d/dphi.
    modes = np.arange(1, feedback_modes + 1)
    basis = np.sin(modes[None, :] * phi.reshape(-1, 1))  # (stations, modes)
    station_weights = np.broadcast_to(weights, phi.shape).reshape(-1)
    gram = basis.T @ (basis * station_weights[:, None])

    def drift_terms(transverse_plastic):
        """Return w'' and (w'')'' at the stations from the plastic drift."""
        if not geodesic_feedback:
            return zeros, zeros
        rhs = basis.T @ (transverse_plastic.reshape(-1) * station_weights)
        coefficients = np.linalg.solve(gram, rhs)
        second = -(basis @ (coefficients * (modes * q) ** 2)).reshape(phi.shape)
        fourth = (basis @ (coefficients * (modes * q) ** 4)).reshape(phi.shape)
        return second, fourth

    for step, kappa in enumerate(curvature):
        imposed = kappa * reference
        transverse_slip = (
            eta * kappa * transverse_map if mode == "prescribed" else zeros
        )
        if mode == "equilibrium":
            strain_guess = old_strain if step else imposed
            w2, w4 = drift_terms(plastic[1])
            # Total in-surface curvature change and its second derivative.
            curvature_g = kappa * geodesic + w2
            curvature_g_ss = -q * q * kappa * geodesic + w4
            for _ in range(maximum_fixed_point):
                axial = ea * strain_guess + tension_per_wire_N
                q_t = axial * curvature_g - bending_rigidity_Nm2 * curvature_g_ss
                ratio = np.abs(q_t) / cap0
                if np.any(ratio >= 1.0):
                    raise RuntimeError(
                        f"gross transverse slip: demand ratio {ratio.max():.3f} at step {step}"
                    )
                cap_l = cap0 * np.sqrt(1.0 - ratio**2)
                u_new, fields = solve_increment(imposed, transverse_slip, q_t, cap_l, u)
                strain_new = fields[3]
                change = float(np.max(np.abs(ea * (strain_new - strain_guess)))) / cap0
                strain_guess = strain_new
                if change <= 1e-9:
                    break
            else:
                raise RuntimeError("transverse fixed point did not converge")
            demand[step] = float(ratio.max())
        else:
            q_t = zeros
            cap_l = np.full((elements, 4), cap0)
            u_new, fields = solve_increment(imposed, transverse_slip, q_t, cap_l, u)
            if mode == "prescribed":
                demand[step] = float(np.max(np.abs(fields[6]))) / cap0
        u = u_new
        _, _, _, strain, slip_l, t_l, t_t, elastic = fields
        new_pl = slip_l - elastic[0]
        dpl = new_pl - plastic[0]
        if mode == "prescribed":
            new_pt = transverse_slip - elastic[1]
            dpt = new_pt - plastic[1]
            d_total = cap0 * np.hypot(dpl, dpt)
            d_l = t_l * dpl
            d_t = t_t * dpt
        elif mode == "equilibrium":
            sliding = np.abs(dpl) > 0
            dpt = np.where(sliding, dpl * q_t / np.where(sliding, t_l, 1.0), 0.0)
            new_pt = plastic[1] + dpt
            d_l = t_l * dpl
            d_t = q_t * dpt
            d_total = d_l + d_t
        else:
            new_pt = plastic[1]
            dpt = zeros
            d_l = t_l * dpl
            d_t = zeros
            d_total = d_l
        cum_f += factor * float(np.sum(d_total @ weights))
        cum_fl += factor * float(np.sum(d_l @ weights))
        cum_ft += factor * float(np.sum(d_t @ weights))
        cum_r += factor * float(
            np.sum(
                (
                    0.5 * ea * (strain - old_strain) ** 2
                    + 0.5 * stiffness * ((elastic - old_elastic) ** 2).sum(0)
                )
                @ weights
            )
        )
        m = factor * ea * float(np.sum((strain * reference) @ weights))
        if mode == "prescribed":
            # The imposed transverse map is kinematically tied to curvature,
            # so its traction contributes the conjugate moment (Eq. 2.29).
            m += factor * float(np.sum((t_t * eta * transverse_map) @ weights))
        cum_w += m * (kappa - previous_kappa)
        moment[step] = m
        friction[step], friction_l[step], friction_t[step] = cum_f, cum_fl, cum_ft
        stored[step] = factor * float(
            np.sum(
                (0.5 * ea * strain**2 + 0.5 * stiffness * (elastic**2).sum(0)) @ weights
            )
        )
        endpoint[step], remainder[step] = cum_w, cum_r
        plastic = np.stack((new_pl, new_pt))
        history_pl[step], history_pt[step] = new_pl, new_pt
        old_strain, old_elastic = strain.copy(), elastic.copy()
        previous_kappa = float(kappa)
    return TransverseCellResult(
        curvature.copy(),
        moment,
        friction,
        friction_l,
        friction_t,
        stored,
        endpoint,
        remainder,
        demand,
        history_pl,
        history_pt,
        phi.copy(),
    )
