# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Moment-curvature (M-κ) engine: per-wire population sum.

The cable bending moment is the sum, over every wire of every helical layer,
of the friction (stick-slip) stress times its moment lever, plus the full-slip
residual stiffness of the cross-section:

    M(kappa) = sum_layers sum_k sigma_f,k * A_w r_h cos(a) cos(phi_k - phi_bend)
             + EI_min * kappa.

The friction stress ``sigma_f,k`` is the universal Coulomb element (R1
Instantiation A) driven by the signed curvature projection
``u = kappa r_h cos(phi_k - phi_bend)`` with stress stiffness
``k_f = xi E_w cos^2(a)`` and yield ``f_y,k = mu F_n,k / A_w``, where the
physical per-wire normal force is ``F_n,k = F_n_axi * g(phi_k)`` — the
axisymmetric capstan value modulated by the dimensionless three-core
shape.

Why this auxiliary population law has a smooth knee
----------------------------------------------------
Wire ``k`` slips when ``kappa >= kappa_slip,k = kappa_c,k / |cos(phi_k -
phi_bend)|`` with ``kappa_c,k = f_y,k / k_f``. The projection factor
``1/|cos|`` spans ``[kappa_c, inf)`` across the population: bending-plane wires
saturate first, near-neutral-axis wires never saturate. Summing over the
population gives a smooth knee from EI_max to EI_min. This auxiliary Coulomb
population law is distinct from the Papailiou neutral-axis slip-front used by
the public :func:`cablexsec.api.compute_mkappa` path. The three-core shape widens
the threshold distribution; contact pressure and friction remain physical
inputs whose identification class must be reported.

References
----------
[Kim2020] Eq. 3 (friction stress); [Yun2020] (xi); [Saevik2011] §5.2 / Eq. 55
(kappa_c); [WitzTan1992b] (EI bounds); [MenardCartraud2023].
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.return_mapping import CoulombElement
from cablexsec.mechanics.tendon_section import WireSection

__all__ = ["HelicalLayerSpec", "MkappaResult", "compute_mkappa_loop", "energy_closure"]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class HelicalLayerSpec:
    """Reduced description of one helical layer for the M-κ sum.

    Parameters
    ----------
    n_wires : int
    r_helix_m, lay_angle_rad : float
    E_wire_Pa, A_wire_m2 : float
    xi : float
        Yun shear coefficient (1.0 = plane sections / Witz-Tan).
    mu : float
        Friction coefficient.
    F_n_axi_N : float
        Normal point force [N] for directly constructed auxiliary specimens.
        Cable adapters instead set normal_load_is_line=True and supply N/m.
        That case requires an explicit transfer length for this auxiliary law.
    shape_g : numpy.ndarray or None
        Dimensionless three-core shape ``g(phi_k)`` (mean 1), one value per
        wire; ``None`` => uniform (g = 1).
    section : WireSection or None
        Cross-section of a rectangular or profiled wire (gap G1): local
        bending and stress points of the fatigue path. ``None`` is the round
        wire of area ``A_wire_m2`` (the earlier behaviour, bit for bit).
    """

    n_wires: int
    r_helix_m: float
    lay_angle_rad: float
    E_wire_Pa: float
    A_wire_m2: float
    xi: float
    mu: float
    F_n_axi_N: float
    shape_g: FloatArray | None = None
    contact_pressure_Pa: float = 0.0
    normal_load_is_line: bool = False
    friction_transfer_length_m: float | None = None
    section: WireSection | None = None

    @property
    def phi_wires(self) -> FloatArray:
        return 2.0 * np.pi * np.arange(self.n_wires) / self.n_wires

    @property
    def k_f(self) -> float:
        """Stick stiffness ``k_f = xi E_w cos^2(a)`` [Pa] (relates sigma_f to the
        wire axial strain). The moment arm carries ``r_h cos(a)`` separately, so
        the net EI_max scaling lands on cos^3(a) per [WitzTan1992b]."""
        return self.xi * self.E_wire_Pa * math.cos(self.lay_angle_rad) ** 2

    @property
    def strain_scale(self) -> float:
        """``c_strain = r_h`` used with ``k_f = xi E cos^2(a)``."""
        return self.r_helix_m

    @property
    def f_y(self) -> FloatArray:
        """Auxiliary lumped Coulomb stress, with explicit force dimensions.

        A normal line load cannot be divided by area to obtain stress.
        Multiplication by a declared axial transfer length is required.
        The public Papailiou model uses its separate pressure/helix relation.
        """
        g = np.ones(self.n_wires) if self.shape_g is None else np.asarray(self.shape_g)
        force = self.F_n_axi_N
        if self.normal_load_is_line:
            length = self.friction_transfer_length_m
            if length is None or not math.isfinite(length) or length <= 0:
                raise CableInputError(
                    "auxiliary Coulomb law requires a positive friction transfer length"
                )
            force *= length
        return self.mu * force * g / self.A_wire_m2

    @property
    def moment_lever(self) -> float:
        """Projected wire-force lever ``A_w r_h cos(a)``."""
        return self.A_wire_m2 * self.r_helix_m * math.cos(self.lay_angle_rad)

    @property
    def EI_stick(self) -> float:
        """Friction stick EI per layer = ``1/2 xi n_w E_w A_w r_h^2 cos^3(a)``.

        Equals the Witz-Tan plane-section bound for ``xi = 1`` ([WitzTan1992b]).
        """
        return self.k_f * self.strain_scale * self.moment_lever * self.n_wires / 2.0


@dataclass(frozen=True)
class MkappaResult:
    """Moment-curvature loop and derived quantities."""

    kappa: FloatArray
    moment_Nm: FloatArray
    W_diss_J_per_m: float
    EI_apparent_Nm2: float
    EI_max_Nm2: float
    EI_min_Nm2: float
    slip_fraction: FloatArray


def compute_mkappa_loop(
    layers: list[HelicalLayerSpec],
    kappa_history: FloatArray,
    *,
    phi_bend: float = 0.0,
    EI_min_Nm2: float = 0.0,
) -> MkappaResult:
    """Integrate the per-wire population M-κ loop over a curvature history.

    Parameters
    ----------
    layers : list of HelicalLayerSpec
        Helical layers contributing friction stress.
    kappa_history : numpy.ndarray
        Curvature time series [1/m] (e.g. one or more sinusoidal cycles).
    phi_bend : float, optional
        Bending-plane azimuth [rad].
    EI_min_Nm2 : float, optional
        Full-slip residual stiffness (individual wire bending + cylindrical
        layers + conductor) [N·m^2]. Added as ``EI_min * kappa``.

    Returns
    -------
    MkappaResult
    """
    kappa = np.asarray(kappa_history, dtype=np.float64)
    n_steps = kappa.size
    moment = np.zeros(n_steps, dtype=np.float64)
    slip_frac = np.zeros(n_steps, dtype=np.float64)

    elements = [CoulombElement(k=lyr.k_f, shape=(lyr.n_wires,)) for lyr in layers]
    cos_phi = [np.cos(lyr.phi_wires - phi_bend) for lyr in layers]
    f_y = [lyr.f_y for lyr in layers]

    total_wires = sum(lyr.n_wires for lyr in layers) or 1
    kappa_prev = 0.0
    w_diss = 0.0
    for t in range(n_steps):
        d_kappa = kappa[t] - kappa_prev
        kappa_prev = kappa[t]
        m_t = 0.0
        n_slip = 0
        for lyr, el, cphi, fy in zip(layers, elements, cos_phi, f_y, strict=True):
            # drive = wire axial strain increment = d_kappa * r_h * cos(phi);
            # k_f already contains the cos^2(a) projection.
            du = d_kappa * lyr.strain_scale * cphi
            previous_slip = el.slip_total.copy()
            sigma_f, slipping = el.update(du, fy)
            w_diss += float(
                lyr.A_wire_m2
                * math.cos(lyr.lay_angle_rad)
                * np.sum(sigma_f * (el.slip_total - previous_slip))
            )
            m_t += float(np.sum(sigma_f * lyr.moment_lever * cphi))
            n_slip += int(np.count_nonzero(slipping))
        moment[t] = m_t + EI_min_Nm2 * kappa[t]
        slip_frac[t] = n_slip / total_wires

    ei_max = sum(lyr.EI_stick for lyr in layers) + EI_min_Nm2
    # Apparent secant stiffness from the curvature extremes on the loop.
    i_hi, i_lo = int(np.argmax(kappa)), int(np.argmin(kappa))
    dk = kappa[i_hi] - kappa[i_lo]
    ei_app = (moment[i_hi] - moment[i_lo]) / dk if dk != 0 else math.nan
    return MkappaResult(
        kappa=kappa,
        moment_Nm=moment,
        W_diss_J_per_m=w_diss,
        EI_apparent_Nm2=ei_app,
        EI_max_Nm2=ei_max,
        EI_min_Nm2=EI_min_Nm2,
        slip_fraction=slip_frac,
    )


def energy_closure(layers: list[HelicalLayerSpec], kappa_amp: float) -> float:
    """Closed-form dissipated energy per symmetric cycle (gate A11).

    For a symmetric cycle ``kappa(t) = kappa_a sin(wt)`` each slipped wire
    dissipates ``4 f_y,k (u_a,k - f_y,k/k_f)`` in the (sigma_f, strain) plane,
    with drive amplitude ``u_a,k = kappa_a c_strain |cos(phi_k)|``
    (``c_strain = r_h``). Mapping to the cable energy multiplies by
    ``moment_lever / c_strain = A_w cos(a)`` (the cos(phi) factors cancel).
    Wires with ``u_a,k <= f_y,k/k_f`` stay stuck and dissipate nothing.

    Returns
    -------
    float
        Total dissipated energy per cycle [J/m].
    """
    total = 0.0
    for lyr in layers:
        u_a = kappa_amp * lyr.strain_scale * np.abs(np.cos(lyr.phi_wires))
        u_slip = lyr.f_y / lyr.k_f
        slipped = u_a > u_slip
        contrib = 4.0 * lyr.f_y[slipped] * (u_a[slipped] - u_slip[slipped])
        total += float(np.sum(contrib)) * lyr.moment_lever / lyr.strain_scale
    return total
