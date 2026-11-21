# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Independent field solver of the coupled axial slip of laid cores.

Reference for ``CoreFrictionSpec(core_core_coupling="relative")``
(:mod:`cablexsec.mechanics.core_friction`). It imports only NumPy and SciPy and
shares no code with the package.

Model (normalised, one pitch, periodic). ``n`` equal cores; ``x`` is the
azimuth of core 0 along the pitch and core ``i`` sits at the azimuth
``theta_i = x + 2 pi i / n``. Each core has its own field (no symmetry
reduction): the slip ``u_i(x)`` relative to the plane section and the axial
force ``n_i = a cos(theta_i - phi_b) + u_i'`` (``a`` the curvature in units in
which the axial rigidity and the helix geometry are one). Equilibrium
``n_i' = -f_i``, with the contact forces per unit ``x``:

* intralayer contacts between neighbours ``(i, j)`` at the same ``x``: an
  elastic--Coulomb spring on the relative slip ``u_i - u_j`` (penalty ``k_c``,
  bound ``b_c``); the force on ``j`` is minus the force on ``i``;
* an interlayer contact of every core (filler, bedding, sheath): an
  elastic--Coulomb spring on the absolute slip ``u_i`` (``k_a``, ``b_a``).

Bounds are linear Coulomb bounds (no capstan term). The rigid contact is the
limit of a large penalty.

Discretisation: ``N`` nodes per core and pitch, ``h = 2 pi / N``; nodes carry
``u`` and the contact springs, segments ``j`` (between nodes ``j`` and
``j + 1``, at ``x = (j + 1/2) h``) carry ``n``. Each load increment minimises
the incremental energy

    sum_i sum_j h n_ij^2 / 2 + sum_springs h Huber_{k, b}(r - p_old)

over the nodal slips (``r`` the spring's relative or absolute slip, ``p_old``
its slip at the start of the increment; the Huber envelope is the exact
backward-Euler potential of a spring--slider), by semismooth Newton with Armijo
backtracking and sparse direct solves. The pitch mean of every core's slip is
held at zero by Lagrange multipliers: this fixes the slides of whole cores that
change no force (a common slide without interlayer friction, and per-core
slides in full slip). On a planar path from a stress-free state the slip of
every core has zero pitch mean, so the constraint changes nothing else.

Outputs per sample: the section moment of the cores (pitch mean of
``n cos(theta - phi_b)`` summed over the cores; the stuck moment is
``n a / 2``), the dissipated work, the fraction of springs in stick, and on
request the force fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.linalg import splu

FloatArray = NDArray[np.float64]

__all__ = ["CoupledSlipReference", "ReferencePath", "run_path"]


@dataclass(frozen=True)
class CoupledSlipReference:
    """Cores, contacts and grid of the reference model.

    ``b_c`` is the bound of one intralayer contact (``n >= 3``: two contacts
    per core; ``n = 2``: the single contact takes ``2 b_c``, so the per-core
    capacity is ``b_a + 2 b_c`` in every case). ``a_cr = b_a + 2 b_c`` is the
    uncoupled slip-onset curvature.
    """

    n_cores: int = 3
    nodes: int = 720
    b_c: float = 1.0
    b_a: float = 0.0
    k_c: float = 1.0e5
    k_a: float = 1.0e5

    @property
    def a_cr(self) -> float:
        return self.b_a + 2.0 * self.b_c

    def contacts(self) -> list[tuple[int, int, float]]:
        n = self.n_cores
        if n == 2:
            return [(0, 1, 2.0 * self.b_c)] if self.b_c > 0 else []
        if self.b_c <= 0:
            return []
        return [(i, (i + 1) % n, self.b_c) for i in range(n)]


@dataclass
class ReferencePath:
    """Results of :func:`run_path` (one entry per curvature sample)."""

    a: FloatArray
    moment: FloatArray
    dissipation: FloatArray
    stick_fraction: FloatArray
    newton_iterations: NDArray[np.int64]
    forces: dict[int, FloatArray] = field(default_factory=dict)
    contact_forces: dict[int, FloatArray] = field(default_factory=dict)
    interlayer_forces: dict[int, FloatArray] = field(default_factory=dict)
    segment_azimuth: FloatArray | None = None


def _huber(r: FloatArray, k: float, b: float) -> tuple[FloatArray, ...]:
    """Value, slope and curvature of the Huber envelope at ``r``."""
    limit = b / k
    inside = np.abs(r) <= limit
    value = np.where(inside, 0.5 * k * r * r, b * np.abs(r) - 0.5 * b * limit)
    slope = np.clip(k * r, -b, b)
    # A spring on its bound at the start of an increment (|k r| = b after the
    # last update) is taken as sticking; the Newton step then decides.
    curvature = np.where(np.abs(k * r) <= b * (1.0 + 1e-12), k, 0.0)
    return value, slope, curvature


class _Solver:
    def __init__(self, ref: CoupledSlipReference, phi_b: float) -> None:
        self.ref = ref
        n, big_n = ref.n_cores, ref.nodes
        self.n, self.N = n, big_n
        self.h = 2.0 * np.pi / big_n
        x_seg = (np.arange(big_n) + 0.5) * self.h
        self.theta = x_seg[None, :] + 2.0 * np.pi * np.arange(n)[:, None] / n
        self.shape = np.cos(self.theta - phi_b)  # (n, N)
        self.contacts = ref.contacts()
        self.u = np.zeros((n, big_n))
        self.p_c = np.zeros((len(self.contacts), big_n))
        self.p_a = np.zeros((n, big_n))
        # Difference operator D u = (u_{j+1} - u_j) / h, periodic, per core.
        eye = sp.identity(big_n, format="csr")
        shift = sp.csr_matrix(
            (np.ones(big_n), (np.arange(big_n), (np.arange(big_n) + 1) % big_n)),
            shape=(big_n, big_n),
        )
        d1 = (shift - eye) / self.h
        self.D = sp.block_diag([d1] * n, format="csr")
        self.K_axial = (self.h * (self.D.T @ self.D)).tocsr()
        self.C = sp.block_diag(
            [sp.csr_matrix(np.full((1, big_n), self.h))] * n, format="csr"
        )

    # Energy, gradient and generalised Hessian at the slip field ``u``.
    def _parts(self, u: FloatArray, a: float, hess: bool):
        ref, h = self.ref, self.h
        force = a * self.shape + (self.D @ u.ravel()).reshape(u.shape)
        energy = 0.5 * h * float(np.sum(force * force))
        grad = h * (self.D.T @ force.ravel())
        rows: list[FloatArray] = []
        cols: list[FloatArray] = []
        vals: list[FloatArray] = []
        nodes = np.arange(self.N)
        for c, (i, j, b) in enumerate(self.contacts):
            r = u[i] - u[j] - self.p_c[c]
            value, slope, curv = _huber(r, ref.k_c, b)
            energy += h * float(np.sum(value))
            grad_c = np.zeros_like(u)
            grad_c[i] += h * slope
            grad_c[j] -= h * slope
            grad = grad + grad_c.ravel()
            if hess:
                wi, wj = i * self.N + nodes, j * self.N + nodes
                for rr, cc, sgn in (
                    (wi, wi, 1),
                    (wj, wj, 1),
                    (wi, wj, -1),
                    (wj, wi, -1),
                ):
                    rows.append(rr)
                    cols.append(cc)
                    vals.append(sgn * h * curv)
        if ref.b_a > 0.0:
            r = u - self.p_a
            value, slope, curv = _huber(r, ref.k_a, ref.b_a)
            energy += h * float(np.sum(value))
            grad = grad + h * slope.ravel()
            if hess:
                idx = np.arange(self.n * self.N)
                rows.append(idx)
                cols.append(idx)
                vals.append(h * curv.ravel())
        if not hess:
            return energy, grad, None
        size = self.n * self.N
        extra = (
            sp.csr_matrix(
                (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                shape=(size, size),
            )
            if vals
            else sp.csr_matrix((size, size))
        )
        return energy, grad, (self.K_axial + extra).tocsc()

    def increment(self, a: float) -> int:
        """Solve the increment to curvature ``a``; returns Newton iterations."""
        u = self.u.copy()
        scale = max(abs(a), self.ref.a_cr, 1.0) * self.h
        for iteration in range(1, 200):
            energy, grad, hess = self._parts(u, a, True)
            m = self.C.shape[0]
            kkt = sp.bmat([[hess, self.C.T], [self.C, None]], format="csc")
            rhs = np.concatenate([-grad, -(self.C @ u.ravel())])
            # Converged: projected gradient (gradient minus its multiplier part).
            lam = np.linalg.lstsq(self.C.T.toarray(), -grad, rcond=None)[0]
            residual = grad + self.C.T @ lam
            if np.max(np.abs(residual)) <= 1e-10 * scale:
                self.u = u
                return iteration - 1
            try:
                step = splu(kkt).solve(rhs)[: self.n * self.N]
            except RuntimeError:  # singular generalised Hessian: regularise
                reg = kkt + sp.block_diag(
                    [sp.identity(self.n * self.N) * 1e-12, sp.csc_matrix((m, m))]
                )
                step = splu(reg.tocsc()).solve(rhs)[: self.n * self.N]
            slope0 = float(grad @ step)
            t = 1.0
            while True:
                trial = u + t * step.reshape(u.shape)
                e_trial = self._parts(trial, a, False)[0]
                if e_trial <= energy + 1e-4 * t * slope0 + 1e-15 * abs(energy):
                    break
                t *= 0.5
                if t < 1e-12:
                    break
            u = trial
        raise RuntimeError("reference Newton did not converge")

    def commit(
        self, a: float
    ) -> tuple[float, float, FloatArray, FloatArray, FloatArray]:
        """Update the slips; return work, stick fraction and the fields."""
        ref, h, u = self.ref, self.h, self.u
        dissipation = 0.0
        stick = 0
        total = 0
        contact_f = np.zeros((len(self.contacts), self.N))
        for c, (i, j, b) in enumerate(self.contacts):
            r = u[i] - u[j] - self.p_c[c]
            slope = np.clip(ref.k_c * r, -b, b)
            slipping = np.abs(ref.k_c * r) > b
            new_p = np.where(slipping, u[i] - u[j] - slope / ref.k_c, self.p_c[c])
            dissipation += h * float(
                np.sum(np.abs(slope) * np.abs(new_p - self.p_c[c]))
            )
            self.p_c[c] = new_p
            contact_f[c] = -slope  # force on core i from core j
            stick += int(np.count_nonzero(~slipping))
            total += self.N
        inter_f = np.zeros((self.n, self.N))
        if ref.b_a > 0.0:
            r = u - self.p_a
            slope = np.clip(ref.k_a * r, -ref.b_a, ref.b_a)
            slipping = np.abs(ref.k_a * r) > ref.b_a
            new_p = np.where(slipping, u - slope / ref.k_a, self.p_a)
            dissipation += h * float(np.sum(np.abs(slope) * np.abs(new_p - self.p_a)))
            self.p_a = new_p
            inter_f = -slope
            stick += int(np.count_nonzero(~slipping))
            total += self.n * self.N
        force = a * self.shape + (self.D @ u.ravel()).reshape(u.shape)
        return dissipation, stick / max(total, 1), force, contact_f, inter_f


def run_path(
    ref: CoupledSlipReference,
    a_path: FloatArray,
    *,
    phi_b: float = 0.0,
    keep: tuple[int, ...] = (),
) -> ReferencePath:
    """Planar curvature path ``a_path`` (bending direction ``phi_b``)."""
    a_path = np.asarray(a_path, dtype=np.float64)
    solver = _Solver(ref, phi_b)
    count = a_path.size
    moment = np.zeros(count)
    work = np.zeros(count)
    stick = np.zeros(count)
    iters = np.zeros(count, dtype=np.int64)
    out = ReferencePath(a_path, moment, work, stick, iters)
    out.segment_azimuth = solver.theta.copy()
    total = 0.0
    previous = 0.0

    def advance(a_from: float, a_to: float, depth: int) -> tuple[int, float, tuple]:
        # Solve to ``a_to``; an increment that does not converge is split.
        try:
            used = solver.increment(a_to)
        except RuntimeError:
            if depth >= 12:
                raise
            mid = 0.5 * (a_from + a_to)
            n1, d1, _ = advance(a_from, mid, depth + 1)
            n2, d2, state = advance(mid, a_to, depth + 1)
            return n1 + n2, d1 + d2, state
        d, s, force, contact_f, inter_f = solver.commit(a_to)
        return used, d, (s, force, contact_f, inter_f)

    for index, a in enumerate(a_path):
        iters[index], d, (s, force, contact_f, inter_f) = advance(previous, float(a), 0)
        previous = float(a)
        total += d
        work[index] = total
        stick[index] = s
        moment[index] = float(np.sum(force * solver.shape)) / ref.nodes
        if index in keep:
            out.forces[index] = force.copy()
            out.contact_forces[index] = contact_f
            out.interlayer_forces[index] = inter_f
    return out
