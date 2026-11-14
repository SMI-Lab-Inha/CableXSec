# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Self-contained JONSWAP-style load-series generator.

Produces a :class:`~cablexsec.io.series.LoadCaseSeries` of curvature (and
optional tension) from a JONSWAP wave spectrum, so the demos and tests run with
**no proprietary global-analysis files**. This is a screening/illustration
generator — it maps a sea state to a *representative* curvature amplitude
spectrum, not a calibrated hydrodynamic response. That scope is stated in the
returned provenance (R-HONEST).

The JONSWAP one-sided spectrum ([Hasselmann1973], DNV-RP-C205 form):

    S(ω) = α g² ω⁻⁵ exp(−5/4 (ω_p/ω)⁴) γ^r,
    r = exp(−(ω − ω_p)² / (2 σ² ω_p²)),   σ = 0.07 (ω≤ω_p) else 0.09

with ``ω_p = 2π/Tp`` and ``α`` scaled so the zeroth moment ``m₀`` matches the
target ``Hs`` (``Hs = 4√m₀``). Curvature is synthesised as a random-phase sum of
harmonic components with amplitudes ``√(2 S(ω) dω)`` scaled to a target RMS
curvature — i.e. the wave spectrum sets the *shape*, the user sets the
*amplitude*.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from cablexsec.exceptions import CableInputError
from cablexsec.io.series import LoadCaseSeries
from cablexsec.provenance import Provenance

FloatArray = NDArray[np.float64]

__all__ = ["check_jonswap", "jonswap_series"]


def _jonswap_psd(
    omega: FloatArray, hs_m: float, tp_s: float, gamma: float
) -> FloatArray:
    """One-sided JONSWAP spectral density ``S(ω)`` [m²·s], shape-normalised to Hs."""
    g = 9.80665
    omega_p = 2.0 * math.pi / tp_s
    sigma = np.where(omega <= omega_p, 0.07, 0.09)
    r = np.exp(-((omega - omega_p) ** 2) / (2.0 * sigma**2 * omega_p**2))
    with np.errstate(divide="ignore", over="ignore"):
        base = g**2 * omega ** (-5.0) * np.exp(-1.25 * (omega_p / omega) ** 4)
    base = np.where(omega > 0.0, base, 0.0) * gamma**r
    # Scale alpha so that m0 = ∫S dω matches Hs = 4 sqrt(m0).
    d_omega = float(omega[1] - omega[0])
    m0_raw = float(np.sum(base) * d_omega)
    if m0_raw <= 0.0:
        return np.zeros_like(omega)
    alpha = (hs_m / 4.0) ** 2 / m0_raw
    return alpha * base


def _superpose(samples: int, amp: FloatArray, phases: FloatArray) -> FloatArray:
    """``Σ amp_i cos(ω_i t_k + φ_i)`` at ``samples`` times, by an inverse FFT.

    The components lie on the harmonic grid of :func:`jonswap_series`:
    ``ω_i = i Δω`` for ``i = 1..M`` with ``Δω = π / (M Δt)``, at the times
    ``t_k = k Δt``, so ``ω_i t_k = 2π i k / (2M)`` and the sum is the real
    part of the discrete Fourier synthesis of length ``2M`` of the complex
    amplitudes ``amp_i exp(j φ_i)``, periodic in ``k`` with that length. It
    equals the term-by-term sum to round-off in ``O(M log M)`` operations
    instead of ``O(M × samples)``.
    """
    count = amp.size
    length = 2 * count
    spectrum = np.zeros(length, dtype=np.complex128)
    spectrum[1 : count + 1] = amp * np.exp(1j * phases)
    cycle = np.fft.ifft(spectrum).real * length
    index = np.arange(samples) % length
    return np.asarray(cycle[index], dtype=np.float64)


def check_jonswap(
    *,
    hs_m: float = 3.0,
    tp_s: float = 9.0,
    gamma: float = 3.3,
    duration_s: float = 3600.0,
    dt_s: float = 0.5,
    rms_curvature_1_m: float = 0.01,
    mean_tension_N: float = 0.0,
    tension_curvature_coupling_N_per_1_m: float = 0.0,
    tension_kind: str = "effective",
    dynamic_tension_rms_N: float = 0.0,
    tension_phase_deg: float = 0.0,
    out_of_plane_fraction: float = 0.0,
) -> int:
    """Check the parameters of :func:`jonswap_series` without generating.

    Returns the number of samples; raises :class:`CableInputError` naming the
    first parameter that is refused, as :func:`jonswap_series` does.
    """
    for name, value in (
        ("dt_s", dt_s),
        ("duration_s", duration_s),
        ("tp_s", tp_s),
        ("gamma", gamma),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise CableInputError(f"{name} must be finite and positive")
    for name, value in (("hs_m", hs_m), ("rms_curvature_1_m", rms_curvature_1_m)):
        if not math.isfinite(value) or value < 0.0:
            raise CableInputError(f"{name} must be finite and non-negative")
    if not all(
        math.isfinite(value)
        for value in (
            mean_tension_N,
            tension_curvature_coupling_N_per_1_m,
            tension_phase_deg,
        )
    ):
        raise CableInputError("tension parameters must be finite")
    for name, value in (
        ("dynamic_tension_rms_N", dynamic_tension_rms_N),
        ("out_of_plane_fraction", out_of_plane_fraction),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise CableInputError(f"{name} must be finite and non-negative")
    if tension_kind not in ("effective", "true"):
        raise CableInputError("synthetic force must be declared effective or true")
    n = int(round(duration_s / dt_s))
    if n < 2:
        raise CableInputError("duration_s / dt_s must provide at least two samples")
    return n


def jonswap_series(
    *,
    hs_m: float = 3.0,
    tp_s: float = 9.0,
    gamma: float = 3.3,
    duration_s: float = 3600.0,
    dt_s: float = 0.5,
    rms_curvature_1_m: float = 0.01,
    mean_tension_N: float = 0.0,
    tension_curvature_coupling_N_per_1_m: float = 0.0,
    seed: int = 0,
    tension_kind: Literal["effective", "true"] = "effective",
    dynamic_tension_rms_N: float = 0.0,
    tension_phase_deg: float = 0.0,
    out_of_plane_fraction: float = 0.0,
) -> LoadCaseSeries:
    r"""Generate a JONSWAP-driven :class:`LoadCaseSeries` (curvature + tension).

    Parameters
    ----------
    hs_m, tp_s, gamma : float
        Significant wave height [m], peak period [s], JONSWAP peakedness.
    duration_s, dt_s : float
        Record length and uniform time step [s].
    rms_curvature_1_m : float
        Target RMS of the synthesised curvature_x history [1/m]; the spectrum
        sets the *shape* and this sets the *amplitude*.
    mean_tension_N : float
        Constant mean force [N], with meaning declared by ``tension_kind``.
    tension_kind : str, optional
        Effective tension by default. Select ``true`` for an explicitly
        prescribed material-force example. Effective tension must be converted
        with the relevant pressure thrust before material stress recovery.
    tension_curvature_coupling_N_per_1_m : float
        Optional linear tension fluctuation ``mean + k·κ_x`` [N per 1/m], a crude
        proxy for dynamic-tension/curvature correlation (0 = constant tension).
    seed : int
        RNG seed for the random component phases (deterministic given a seed).
    dynamic_tension_rms_N : float, optional
        RMS [N] of a dynamic tension from the same sea state (0, the default:
        none). See the notes for the model.
    tension_phase_deg : float, optional
        Phase [deg] by which the dynamic tension lags the in-plane curvature
        ``curvature_x`` component by component (0: in phase, 180: in
        antiphase).
    out_of_plane_fraction : float, optional
        RMS of ``curvature_y`` as a fraction of ``rms_curvature_1_m`` (0, the
        default: planar bending).

    Returns
    -------
    LoadCaseSeries
        ``temperature_C`` is ``None``.

    Notes
    -----
    Screening generator: a representative curvature spectrum from a sea state,
    **not** a calibrated vessel/cable hydrodynamic response. Use a real global
    analysis (time-history CSV, RIFLEX or CableDyn reader) for design fatigue.

    Dynamic tension. Every wave component that bends the cable also changes
    its tension (vessel heave and surge at a hang-off), with one transfer
    phase for all components: ``ΔT(t) = c Σ a_i cos(ω_i t + φ_i − θ)``, where
    ``a_i`` and ``φ_i`` are the amplitudes and phases of ``curvature_x``,
    ``θ`` the phase ``tension_phase_deg`` and ``c`` scales the RMS to
    ``dynamic_tension_rms_N``. The tension then cycles with the curvature,
    correlated with it by ``cos θ``. Out-of-plane curvature. ``curvature_y``
    is a second random-phase realisation of the same spectrum (independent
    phases from the same seed), scaled to ``out_of_plane_fraction`` times the
    in-plane RMS, so that the curvature path is biaxial and the bending
    direction turns. Both are stated simple models, not responses of a global
    analysis.
    """
    n = check_jonswap(
        hs_m=hs_m,
        tp_s=tp_s,
        gamma=gamma,
        duration_s=duration_s,
        dt_s=dt_s,
        rms_curvature_1_m=rms_curvature_1_m,
        mean_tension_N=mean_tension_N,
        tension_curvature_coupling_N_per_1_m=tension_curvature_coupling_N_per_1_m,
        tension_kind=tension_kind,
        dynamic_tension_rms_N=dynamic_tension_rms_N,
        tension_phase_deg=tension_phase_deg,
        out_of_plane_fraction=out_of_plane_fraction,
    )
    t = np.arange(n, dtype=np.float64) * dt_s

    # Frequency grid up to the Nyquist of the sampling; build the JONSWAP PSD.
    # The components are the harmonics i π / (n_comp dt_s), which
    # :func:`_superpose` sums by an inverse FFT.
    n_comp = max(n // 2, 16)
    omega_max = math.pi / dt_s
    omega = np.linspace(omega_max / n_comp, omega_max, n_comp)
    d_omega = float(omega[1] - omega[0])
    psd = _jonswap_psd(omega, hs_m, tp_s, gamma)
    amp = np.sqrt(2.0 * psd * d_omega)  # component amplitudes

    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * math.pi, size=n_comp)
    # Random-phase superposition: kappa(t) = Σ amp_i cos(ω_i t + φ_i).
    kappa = _superpose(n, amp, phases)

    # Scale to the requested RMS curvature (spectrum sets shape, user sets level).
    rms = float(np.sqrt(np.mean(kappa**2)))
    if rms > 0.0:
        kappa = kappa * (rms_curvature_1_m / rms)

    tension = mean_tension_N + tension_curvature_coupling_N_per_1_m * kappa
    if dynamic_tension_rms_N > 0.0:
        lag = math.radians(tension_phase_deg)
        dynamic = _superpose(n, amp, phases - lag)
        size = float(np.sqrt(np.mean(dynamic**2)))
        if size > 0.0:
            tension = tension + dynamic * (dynamic_tension_rms_N / size)

    kappa_y = np.zeros_like(kappa)
    if out_of_plane_fraction > 0.0:
        phases_y = rng.uniform(0.0, 2.0 * math.pi, size=n_comp)
        kappa_y = _superpose(n, amp, phases_y)
        size = float(np.sqrt(np.mean(kappa_y**2)))
        if size > 0.0:
            kappa_y = kappa_y * (out_of_plane_fraction * rms_curvature_1_m / size)

    planar = out_of_plane_fraction == 0.0
    constant = dynamic_tension_rms_N == 0.0
    return LoadCaseSeries(
        time_s=t,
        curvature_x_1_m=kappa,
        curvature_y_1_m=kappa_y,
        tension_N=tension,
        tension_kind=tension_kind,
        temperature_C=None,
        source=f"jonswap_series(Hs={hs_m}m, Tp={tp_s}s, gamma={gamma})",
        provenance=Provenance(
            method="JONSWAP random-phase curvature synthesis "
            "(shape from the wave spectrum, amplitude scaled to a target RMS)",
            citations=("[Hasselmann1973] JONSWAP", "DNV-RP-C205 spectral form"),
            gates=("deterministic given seed; round-trips through the exchange CSV",),
            limitations=(
                "SCREENING generator: a representative curvature spectrum, NOT a "
                "calibrated hydrodynamic cable response — use a real global "
                "analysis for design fatigue",
                (
                    "planar bending (curvature_y = 0)"
                    if planar
                    else "out-of-plane curvature: an independent realisation of "
                    f"the same spectrum, RMS {out_of_plane_fraction:g} of the "
                    "in-plane RMS"
                )
                + "; isothermal (no temperature)",
                "tension is constant or a crude linear curvature proxy"
                if constant
                else "dynamic tension: the in-plane curvature components with "
                f"one transfer phase of {tension_phase_deg:g} deg, RMS "
                f"{dynamic_tension_rms_N:g} N",
            ),
        ),
    )
