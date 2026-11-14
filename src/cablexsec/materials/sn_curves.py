# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Single source of truth for fatigue S-N curve parameters.

Every S-N parameter used anywhere in CableXSec lives in this module. The
:class:`SNcurve` dataclass holds the parameters of a (possibly bilinear)
basquin curve

.. math::

    \\log_{10} N = \\log_{10} C - m \\, \\log_{10} S

where ``S`` is the stress range (unit ``stress_unit``) and ``N`` is the
number of cycles to failure. A non-``None`` ``threshold`` defines a constant-
amplitude fatigue limit below which life is treated as infinite (run-out).

Curve registry status
----------------------
Fully parameterised here (parameters with literature provenance):

* ``dnv_rpc203_t_curve``  — bilinear tubular T-curve in seawater with cathodic
  protection, [DNV-RP-C203] (2024) Table 2-3.
* ``dnv_rpc203_t_curve_free_corrosion`` — the T curve in seawater with free
  corrosion (one slope, m = 3), [DNV-RP-C203] (2024) Table 2-3.
* ``dnv_rpc203_b1_seawater_cp`` and ``dnv_rpc203_b1_free_corrosion`` — the B1
  curve of non-welded base material in seawater with cathodic protection
  (Table 2-2) and with free corrosion (Table 2-4), [DNV-RP-C203] (2024).
* ``saevik_2011_flat_steel`` — flat steel in seawater with cathodic
  protection, [Saevik2011] Eq. (89), used for armour stress-range screening.
* ``nasution_2014_copper``  — copper conductor individual-wire S-N (bulk
  axial), [Nasution2014b] Table 1.
* ``nasution_2013_copper_95`` — the same for the wires of a 95 mm^2
  conductor, [Nasution2013] Eq. 1.
* ``nasution_2013_copper_corrected`` — conductor wire against the actual
  longitudinal stress range, corrected for the compaction indentations,
  [Nasution2013] p. 255; used with ring compaction factors (gap CF3).
* ``en1999_1_3_aluminium``  — corrugated-Al sheath plain-member S-N, EN 1999-1-3
  Table J.1/J.2 detail type 1.4 (category 71-7), single slope ``m = 7``,
  constant-amplitude cut-off ``Δσ_L = 40.6 MPa`` at ``N_L = 10⁸``. Applied to the
  SCF-amplified corrugation-root peak stress (PI-confirmed detail-category
  selection; the corrugation geometry enters via the modelled SCF, so the curve
  is plain-material to avoid double-counting).

Strain-life (Coffin-Manson-Basquin) constants of metallic sheaths, foils and
tapes (gap P6) live in :mod:`cablexsec.materials.strain_life`.

Copper fretting fatigue is **not** a slip-amplitude S-N power law. [Poon2022]
predicts it with an SWT critical-plane *strain-life* method (their Eq. 30),
using the constants of :data:`SWT_LIFE` (Karlsen 2010 SPC-copper data,
[Poon2022] Table 4). It is implemented in
:mod:`cablexsec.analysis.fretting_swt`, not as an ``SNcurve``. (An earlier
``poon_2022_copper`` power-law curve with ``m=4.1, log_C=12.20, 3 μm``
threshold was a mis-citation — those numbers appear in no [Poon2022] table —
and has been removed.)

Deferred (parameters not yet read from source) — needed for the
metallic-sheath fatigue benchmarks:

* ``cigre_tb623_lead``         — [CIGRE TB 623] (lead sheath; source parameters
  not yet available — a stated limitation).

This is intentionally absent from :data:`SN_CURVES` rather than guessed; it is
registered in :data:`BLOCKED_SN_CURVES`, and :func:`get_sn_curve` raises
:class:`BlockedSNCurveError` for it so callers report lead-sheath *stress*
without *damage* until the cited source is read. (The aluminium sheath curve was
previously blocked under the mis-cited name ``bs7910_class_d_aluminium``; it is
now ``en1999_1_3_aluminium``, registered above from EN 1999-1-3 Annex J.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cablexsec.exceptions import CableInputError, CableLookupError

__all__ = [
    "BLOCKED_SN_CURVES",
    "BUILT_IN_SN_CURVES",
    "BlockedSNCurveError",
    "SNcurve",
    "SN_CURVES",
    "SWTLifeConstants",
    "SWT_LIFE",
    "get_sn_curve",
    "get_swt_life",
    "list_sn_curves",
    "register_sn_curve",
    "unregister_sn_curve",
]


# Canonical definition lives in cablexsec.exceptions (re-exported here for the
# historical import path and to keep ``except KeyError`` callers working).
from cablexsec.exceptions import BlockedSNCurveError  # noqa: E402,F401


def _finite_real(value: float, name: str) -> float:
    # The fast path covers ordinary Python floats used by the cycle counter.
    if type(value) is not float and (
        isinstance(value, bool) or not isinstance(value, Real)
    ):
        raise CableInputError(f"{name} must be a finite real scalar")
    try:
        number = float(value)
    except OverflowError as error:
        raise CableInputError(f"{name} must be a finite real scalar") from error
    if not math.isfinite(number):
        raise CableInputError(f"{name} must be a finite real scalar")
    return number


def _positive_power10(exponent: float, name: str) -> float:
    try:
        result = 10.0**exponent
    except OverflowError as error:
        raise CableInputError(f"{name} exceeds floating-point range") from error
    if not math.isfinite(result) or result <= 0:
        raise CableInputError(f"{name} exceeds floating-point range")
    return float(result)


@dataclass(frozen=True)
class SNcurve:
    """Parameters of a single (optionally bilinear) S-N fatigue curve.

    Attributes
    ----------
    name : str
        Registry key.
    stress_unit : str
        Unit of the independent variable: ``"MPa"`` for stress-range curves
        or ``"um"`` for fretting slip-amplitude curves.
    m1, log_C1 : float
        Slope and intercept of the high-stress (low-cycle) branch.
    m2, log_C2 : float or None
        Slope and intercept of the low-stress (high-cycle) branch for a
        bilinear curve; ``None`` for a single-slope curve.
    knee_stress : float or None
        Stress at the branch transition. If ``None`` and the curve is
        bilinear, it is derived from branch continuity.
    threshold : float or None
        Constant-amplitude fatigue limit (same unit as ``stress_unit``).
        Below it, life is infinite (run-out). ``None`` for no cut-off.
    R_ratio : float
        Stress ratio of the underlying tests (default fully reversed, -1). A
        mean-stress correction (:mod:`cablexsec.materials.mean_stress`) maps
        every counted cycle to this test condition.
    environment : str
        Test environment, e.g. ``"seawater"``.
    reference : str
        ``[AuthorYear]`` literature citation (the source; required by
        :func:`register_sn_curve`).
    applicability : str
        Free-text note on the intended application.
    thickness_exponent : float
        Thickness (size) exponent ``k`` of [DNV-RP-C203] Eq. (2.10): stress
        ranges are multiplied by ``(t / t_ref)^k`` for ``t > t_ref`` before
        the curve is entered. ``0`` (default) disables the correction.
    reference_thickness_m : float or None
        Reference thickness ``t_ref`` [m]; required when
        ``thickness_exponent > 0``.
    """

    name: str
    stress_unit: str
    m1: float
    log_C1: float
    m2: float | None = None
    log_C2: float | None = None
    knee_stress: float | None = None
    threshold: float | None = None
    R_ratio: float = -1.0
    environment: str = "seawater"
    reference: str = ""
    applicability: str = ""
    thickness_exponent: float = 0.0
    reference_thickness_m: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise CableInputError("S-N curve name must be nonempty")
        if not isinstance(self.stress_unit, str) or self.stress_unit not in (
            "MPa",
            "um",
        ):
            raise CableInputError("S-N curve stress_unit must be 'MPa' or 'um'")
        if (self.m2 is None) != (self.log_C2 is None):
            raise CableInputError("a second S-N branch requires both m2 and log_C2")
        for name in (
            "m1",
            "log_C1",
            "m2",
            "log_C2",
            "knee_stress",
            "threshold",
            "R_ratio",
        ):
            value = getattr(self, name)
            if value is not None or name in ("m1", "log_C1", "R_ratio"):
                object.__setattr__(self, name, _finite_real(value, name))
        if self.m1 <= 0 or (self.m2 is not None and self.m2 <= 0):
            raise CableInputError("S-N slopes must be positive")
        if self.threshold is not None and self.threshold < 0:
            raise CableInputError("S-N threshold must be nonnegative")
        if self.knee_stress is not None and (
            not self.is_bilinear or self.knee_stress <= 0
        ):
            raise CableInputError("a positive knee_stress requires both S-N branches")
        if self.is_bilinear and self.knee_stress is None and self.m1 == self.m2:
            raise CableInputError("equal S-N slopes require an explicit knee_stress")
        if not self.R_ratio < 1.0:
            raise CableInputError("S-N R_ratio must be below 1")
        exponent = _finite_real(self.thickness_exponent, "thickness_exponent")
        object.__setattr__(self, "thickness_exponent", exponent)
        if exponent < 0:
            raise CableInputError("thickness_exponent must be nonnegative")
        if self.reference_thickness_m is not None:
            t_ref = _finite_real(self.reference_thickness_m, "reference_thickness_m")
            if t_ref <= 0:
                raise CableInputError("reference_thickness_m must be positive")
            object.__setattr__(self, "reference_thickness_m", t_ref)
        if exponent > 0 and self.reference_thickness_m is None:
            raise CableInputError("a thickness exponent needs reference_thickness_m")
        # Fail at definition time if an inferred branch switch cannot be stored.
        self.transition_stress()

    @classmethod
    def from_knee(
        cls,
        name: str,
        *,
        m1: float,
        log_C1: float,
        m2: float,
        knee_cycles: float,
        **kwargs: Any,
    ) -> SNcurve:
        """Bi-linear curve from the first branch, the second slope and the knee.

        The second intercept follows from continuity at ``N = knee_cycles``,
        the transition used by [DNV-RP-C203] Tables 2-1 and 2-2 (``10^6`` or
        ``10^7`` cycles): ``log_C2 = log N_t + m2 log S_t`` with
        ``log S_t = (log_C1 - log N_t) / m1``. Other fields as keywords.
        """
        n_t = _finite_real(knee_cycles, "knee_cycles")
        if n_t <= 0:
            raise CableInputError("knee_cycles must be positive")
        log_n = math.log10(n_t)
        log_s = (float(log_C1) - log_n) / float(m1)
        return cls(
            name=name,
            m1=m1,
            log_C1=log_C1,
            m2=m2,
            log_C2=log_n + float(m2) * log_s,
            knee_stress=_positive_power10(log_s, "knee stress"),
            **kwargs,
        )

    def thickness_factor(self, thickness_m: float | None) -> float:
        """Stress-range factor ``(t / t_ref)^k`` for ``t > t_ref``, else 1.

        [DNV-RP-C203] Eq. (2.10): ``t = t_ref`` is used below the reference
        thickness. Returns exactly ``1.0`` when the curve has no exponent.
        """
        if self.thickness_exponent == 0.0:
            return 1.0
        if thickness_m is None:
            raise CableInputError(
                f"S-N curve {self.name!r} has a thickness exponent; give the "
                "component thickness"
            )
        t = _finite_real(thickness_m, "thickness_m")
        if t <= 0:
            raise CableInputError("thickness_m must be positive")
        assert self.reference_thickness_m is not None
        return float(
            max(1.0, t / self.reference_thickness_m) ** self.thickness_exponent
        )

    def log10_life(self, stress: NDArray[np.float64]) -> NDArray[np.float64]:
        """``log10 N`` for an array of positive stresses (``inf`` at run-out).

        Vectorised form of :meth:`cycles_to_failure` (same branch and
        threshold rules) for the corrected Miner sums; non-positive stresses
        and stresses at or below the threshold give ``+inf``.
        """
        s = np.asarray(stress, dtype=np.float64)
        out = np.full(s.shape, np.inf)
        live = s > 0.0
        if self.threshold is not None:
            live &= s > self.threshold
        knee = self.transition_stress()
        m = np.full(s.shape, self.m1)
        log_c = np.full(s.shape, self.log_C1)
        if knee is not None:
            assert self.m2 is not None and self.log_C2 is not None
            low = s < knee
            m[low] = self.m2
            log_c[low] = self.log_C2
        with np.errstate(divide="ignore", invalid="ignore"):
            out[live] = log_c[live] - m[live] * np.log10(s[live])
        return out

    @property
    def is_bilinear(self) -> bool:
        """True if a second (high-cycle) branch is defined."""
        return self.m2 is not None and self.log_C2 is not None

    def transition_stress(self) -> float | None:
        """Stress at the bilinear branch transition.

        Returns the explicit ``knee_stress`` if given, otherwise the stress
        at which the two branches intersect (derived from continuity, not
        an independent fitted parameter). Returns ``None`` for single-slope
        curves.
        """
        if not self.is_bilinear:
            return None
        if self.knee_stress is not None:
            return self.knee_stress
        assert self.m2 is not None and self.log_C2 is not None  # for type
        # log_C1 - m1 log S = log_C2 - m2 log S  =>  log S = (C2-C1)/(m2-m1)
        log_s = (self.log_C2 - self.log_C1) / (self.m2 - self.m1)
        return _positive_power10(log_s, "S-N transition stress")

    def cycles_to_failure(self, stress: float) -> float:
        """Number of cycles to failure at a given stress (or slip amplitude).

        Parameters
        ----------
        stress : float
            Stress range in ``stress_unit`` (MPa), or fretting slip amplitude
            in micrometres for fretting curves.

        Returns
        -------
        float
            Cycles to failure ``N``. Returns ``inf`` at or below the fatigue
            threshold, and for finite non-positive ``stress``. Nonfinite or
            complex inputs are rejected. Positive lives outside floating-point
            range raise ValueError; zero/Infinity must not represent numerical
            underflow/overflow. Representable fractional lives are formal
            extrapolations, not proof of physical sub-cycle failure.
        """
        stress = _finite_real(stress, "stress")
        if stress <= 0.0:
            return math.inf
        if self.threshold is not None and stress <= self.threshold:
            return math.inf
        m, log_c = self.m1, self.log_C1
        knee = self.transition_stress()
        if knee is not None and stress < knee:
            assert self.m2 is not None and self.log_C2 is not None  # for type
            m, log_c = self.m2, self.log_C2
        log_n = log_c - m * math.log10(stress)
        return _positive_power10(log_n, "S-N life")


# ---------------------------------------------------------------------------
# Curve registry. Parameter values are data,
# with provenance carried in each ``reference`` field.
# ---------------------------------------------------------------------------
SN_CURVES: dict[str, SNcurve] = {
    "dnv_rpc203_t_curve": SNcurve(
        # [DNV-RP-C203] (October 2024) Table 2-3, seawater with cathodic
        # protection: m1 = 3, log a1 = 12.18 for N <= 1.8e6; m2 = 5,
        # log a2 = 16.13 above; the branches meet at 94.4 MPa. Reference
        # thickness 16 mm, k = 0.25. (Until 31 Oct 2026 this entry held
        # 12.164 / 15.606, the in-air D curve of Table 2-1, under this name.)
        name="dnv_rpc203_t_curve",
        stress_unit="MPa",
        m1=3.0,
        log_C1=12.18,
        m2=5.0,
        log_C2=16.13,
        environment="seawater",
        reference="[DNV-RP-C203] (2024) Table 2-3, seawater with cathodic protection",
        applicability=(
            "Tubular-joint T curve in seawater with cathodic protection; used "
            "for armour bulk fatigue as a general offshore design curve."
        ),
        thickness_exponent=0.25,
        reference_thickness_m=0.016,
    ),
    "dnv_rpc203_t_curve_free_corrosion": SNcurve(
        # [DNV-RP-C203] (October 2024) Table 2-3, seawater with free corrosion:
        # one slope m = 3, log a = 12.03, no fatigue limit. Reference
        # thickness 16 mm, k = 0.25.
        name="dnv_rpc203_t_curve_free_corrosion",
        stress_unit="MPa",
        m1=3.0,
        log_C1=12.03,
        environment="seawater, free corrosion",
        reference="[DNV-RP-C203] (2024) Table 2-3, seawater with free corrosion",
        applicability=(
            "Tubular-joint T curve in seawater without cathodic protection; "
            "armour in a flooded annulus that the protection does not reach."
        ),
        thickness_exponent=0.25,
        reference_thickness_m=0.016,
    ),
    "dnv_rpc203_b1_seawater_cp": SNcurve(
        # [DNV-RP-C203] (October 2024) Table 2-2, seawater with cathodic
        # protection, curve B1 (non-welded base material, Tables A-1 and A-9):
        # m1 = 4, log a1 = 14.977 for N <= 1e6; m2 = 5, log a2 = 17.222; k = 0.
        name="dnv_rpc203_b1_seawater_cp",
        stress_unit="MPa",
        m1=4.0,
        log_C1=14.977,
        m2=5.0,
        log_C2=17.222,
        environment="seawater",
        reference=(
            "[DNV-RP-C203] (2024) Table 2-2, B1, seawater with cathodic protection"
        ),
        applicability=(
            "Non-welded base material in seawater with cathodic protection; "
            "the DNV class nearest to drawn armour wire."
        ),
    ),
    "dnv_rpc203_b1_free_corrosion": SNcurve(
        # [DNV-RP-C203] (October 2024) Table 2-4, seawater with free corrosion,
        # curve B1: one slope m = 3, log a = 12.436, k = 0.
        name="dnv_rpc203_b1_free_corrosion",
        stress_unit="MPa",
        m1=3.0,
        log_C1=12.436,
        environment="seawater, free corrosion",
        reference="[DNV-RP-C203] (2024) Table 2-4, B1, seawater with free corrosion",
        applicability=(
            "Non-welded base material in seawater without cathodic protection; "
            "armour in a flooded annulus that the protection does not reach."
        ),
    ),
    "saevik_2011_flat_steel": SNcurve(
        name="saevik_2011_flat_steel",
        stress_unit="MPa",
        m1=4.0,
        log_C1=15.01,
        environment="seawater",
        reference="[Saevik2011] Eq. (89)",
        applicability=(
            "Flat steel in seawater with cathodic protection; armour-wire "
            "stress-range screening with no endurance cut-off."
        ),
    ),
    "oc2017_armour_in_air": SNcurve(
        name="oc2017_armour_in_air",
        stress_unit="MPa",
        m1=6.0,
        log_C1=20.5,
        R_ratio=0.1,  # [OC2017] D3.3.2: axial tension-tension tests at R = 0.1
        environment="air",
        reference=(
            "[OC2017] Handbook on Design and Operation of Flexible Pipes, "
            "Section D2.6, Eq. (D2.1): in-air design curve of Balmoral riser "
            "armour wires (Wang and Berge 2012), log10 a about 20.5, 1/m = 1/6"
        ),
        applicability=(
            "Armour wires in a dry (sealed, intact) annulus; one slope with "
            "the design fatigue limit removed, as applied in OC2017 D2.6."
        ),
    ),
    "nasution_2014_copper": SNcurve(
        name="nasution_2014_copper",
        stress_unit="MPa",
        # Least-squares fit log N = log_C - m log(Δσ) to the nine individual
        # outer-wire tension-tension points of [Nasution2014b] Table 1
        # (Δσ = 130/160/190 MPa). Single slope; no run-out reached in the data.
        m1=6.160,
        log_C1=19.949,
        R_ratio=0.1,
        environment="air",
        reference="[Nasution2014b] Table 1 (9-wire fit, nominal stress range)",
        applicability=(
            "Copper power-conductor individual-wire bulk axial fatigue "
            "(ETP/UNS C11000), R=0.1. Nominal stress range basis; combine "
            "with a stress-concentration factor for local hot-spots."
        ),
    ),
    "nasution_2013_copper_95": SNcurve(
        name="nasution_2013_copper_95",
        stress_unit="MPa",
        # N (dsigma)^m = C with m = 6.238, C = 6.098e19 ([Nasution2013] Eq. 1,
        # p. 249): outer wires of a 95 mm^2 compacted conductor, tension-tension.
        m1=6.238,
        log_C1=math.log10(6.098e19),
        R_ratio=0.1,
        environment="air",
        reference="[Nasution2013] Eq. 1 (95 mm^2, outer wires, nominal range)",
        applicability=(
            "Copper power-conductor individual-wire bulk axial fatigue, 95 mm^2 "
            "compacted conductor, R=0.1, nominal stress range."
        ),
    ),
    "nasution_2013_copper_corrected": SNcurve(
        name="nasution_2013_copper_corrected",
        stress_unit="MPa",
        # N (dsigma)^m = C, m = 8.424, C = 2.88e25 ([Nasution2013] p. 255):
        # outer-wire data lifted by the bending stresses of their compaction
        # eccentricities, i.e. against the actual longitudinal stress range;
        # 95 and 300 mm^2 wires share one corrected band ([Nasution2014a]).
        m1=8.424,
        log_C1=math.log10(2.88e25),
        R_ratio=0.1,
        environment="air",
        reference="[Nasution2013] p. 255 (corrected for the indentations)",
        applicability=(
            "Copper conductor wire against the actual longitudinal stress "
            "range, R=0.1; combine with the ring's compaction_scf (gap CF3)."
        ),
    ),
    "en1999_1_3_aluminium": SNcurve(
        name="en1999_1_3_aluminium",
        stress_unit="MPa",
        # EN 1999-1-3 plain-member detail type 1.4, category 71-7 (general-alloy
        # tube/extrusion, surface-irregularity initiation). Single slope m=7.
        # Δσ_c = 71 MPa (range) at N_C = 2e6: log_C = log10(2e6) + 7 log10(71).
        m1=7.0,
        log_C1=19.2598,
        # Constant-amplitude cut-off Δσ_L = 40.6 MPa at N_L = 1e8 (Table J.2):
        # the m=7 line through (2e6, 71) reaches (1e8, 40.6); below it, run-out.
        threshold=40.6,
        R_ratio=0.5,  # J.1(2): values valid for stress ratio R >= 0.5
        environment="air",
        reference="[EN1999-1-3] Table J.1/J.2 detail 1.4 (cat 71-7), p.79-80 "
        "(BS EN 1999-1-3:2007+A1:2011)",
        applicability=(
            "Corrugated-aluminium metallic-sheath fatigue. Plain-member curve "
            "applied to the SCF-amplified corrugation-root peak stress (range "
            "basis); the corrugation geometry enters via the modelled SCF, so "
            "the curve is plain-material to avoid double-counting. 71-7 is the "
            "conservative general-alloy choice; 90-7 (detail 1.2) is the "
            "less-conservative alternative. Single slope m=7, cut-off at 1e8."
        ),
    ),
}


@dataclass(frozen=True)
class SWTLifeConstants:
    """Smith-Watson-Topper strain-life constants for a material.

    The SWT fatigue-indicator life equation ([Poon2022] Eq. 30) is

    .. math::

        \\sigma_{max}\\,\\frac{\\Delta\\varepsilon_a}{2}
        = \\frac{(\\sigma_f')^2}{E}(2N_f)^{2b}
        + \\sigma_f' \\varepsilon_f' (2N_f)^{b+c},

    combining the Basquin (HCF) elastic line ``Δσ/2 = σ_f' (2N)^b`` and the
    Coffin-Manson (LCF) plastic line ``Δε_pl/2 = ε_f' (2N)^c``.

    Attributes
    ----------
    sigma_f_prime_Pa : float
        Fatigue strength coefficient ``σ_f'`` [Pa].
    b : float
        Fatigue strength exponent (Basquin), negative.
    eps_f_prime : float
        Fatigue ductility coefficient ``ε_f'`` [-].
    c : float
        Fatigue ductility exponent (Coffin-Manson), negative.
    E_Pa : float
        Young's modulus used to fit the elastic line [Pa].
    reference : str
        ``[AuthorYear]`` literature citation.
    """

    sigma_f_prime_Pa: float
    b: float
    eps_f_prime: float
    c: float
    E_Pa: float
    reference: str = ""


# SWT strain-life constants. Values are data, provenance in ``reference``.
SWT_LIFE: dict[str, SWTLifeConstants] = {
    "copper_poon2022": SWTLifeConstants(
        sigma_f_prime_Pa=1035.0e6,
        b=-0.175,
        eps_f_prime=0.8989,
        c=-0.601,
        E_Pa=97.0e9,
        reference="[Poon2022] Table 4 (Karlsen 2010 SPC Class-2 copper data)",
    ),
}


# Sheath-fatigue S-N curves named in the design spec whose source parameters
# are NOT yet available. Registered as explicit BLOCKED placeholders (name ->
# reason) rather than guessed: the cited documents are qualification/assessment
# standards that do not provide an extractable S-N table.
# Adding real parameters requires reading the source first (cite-on-read).
BLOCKED_SN_CURVES: dict[str, str] = {
    "perier_2009_steel": (
        "Périer et al. (2009) reports test conditions, individual lives and "
        "fretting endurance observations, but not the m=3.7/logC=11.94 "
        "slip-amplitude curve formerly attributed to its Table 2. A validated "
        "wire-contact damage law is required before quantitative steel "
        "fretting life is enabled."
    ),
    "cigre_tb623_lead": (
        "[CIGRE TB 623] is a mechanical-qualification "
        "standard, no extractable lead-sheath S-N parameters; source needed. "
        "No strain-life constants for lead alloy E or 1/2C are in the reference "
        "library either (gap P6): [KimChoi2026] Table 3 lists stress-amplitude "
        "points of an unnamed lead alloy quoted from Moreno (2020), which is "
        "not in the library, and [CIGRE TB 610] Sec. 5.4.3 states that lead "
        "sheaths cannot be used in dynamic cables. Lead-sheath fatigue is a "
        "stated limitation."
    ),
}


def get_swt_life(name: str) -> SWTLifeConstants:
    """Return the :class:`SWTLifeConstants` registered under ``name``."""
    try:
        return SWT_LIFE[name]
    except KeyError:
        raise CableLookupError(
            f"Unknown SWT life constants {name!r}. "
            f"Registered: {', '.join(sorted(SWT_LIFE))}."
        ) from None


def list_sn_curves() -> list[str]:
    """Return the names of all registered S-N curves, sorted."""
    return sorted(SN_CURVES)


def get_sn_curve(name: str) -> SNcurve:
    """Return the :class:`SNcurve` registered under ``name``.

    Raises
    ------
    BlockedSNCurveError
        If ``name`` is a deferred curve whose source is not yet available
        (see :data:`BLOCKED_SN_CURVES`) — it must not be invented.
    KeyError
        If ``name`` is not a registered curve.
    """
    if name in BLOCKED_SN_CURVES:
        raise BlockedSNCurveError(
            f"S-N curve {name!r} is BLOCKED: {BLOCKED_SN_CURVES[name]} "
            f"Report sheath stress only until the source is added."
        )
    try:
        return SN_CURVES[name]
    except KeyError:
        raise CableLookupError(
            f"Unknown S-N curve {name!r}. "
            f"Registered curves: {', '.join(list_sn_curves())}."
        ) from None


#: Names of the built-in curves; user registrations cannot replace them.
BUILT_IN_SN_CURVES: frozenset[str] = frozenset(SN_CURVES)


def register_sn_curve(curve: SNcurve, *, replace_existing: bool = False) -> None:
    """Register a user S-N curve for this process.

    The curve is then available by name wherever a registered curve is
    (``bulk_sn`` in a cable YAML, :func:`get_sn_curve`, the ``sn_curves``
    overrides of the fatigue routes). Single-slope and bi-linear curves with
    a knee, a cut-off or endurance limit (``threshold``), a thickness
    exponent, the test ``R_ratio`` and an ``environment`` label are
    accepted; the source (``reference``) and the environment are required.
    Built-in and blocked names cannot be replaced; a user curve only with
    ``replace_existing=True``.

    Examples
    --------
    >>> curve = SNcurve.from_knee(
    ...     "partner_d_air", stress_unit="MPa", m1=3.0, log_C1=12.164, m2=5.0,
    ...     knee_cycles=1e7, environment="air",
    ...     reference="[DNV-RP-C203] Table 2-1, D curve (example)")
    >>> register_sn_curve(curve)
    >>> round(get_sn_curve("partner_d_air").transition_stress(), 1)
    52.6
    >>> unregister_sn_curve("partner_d_air")
    """
    if not isinstance(curve, SNcurve):
        raise CableInputError("register an SNcurve instance")
    if curve.name in BUILT_IN_SN_CURVES or curve.name in BLOCKED_SN_CURVES:
        raise CableInputError(f"{curve.name!r} is a built-in or blocked curve name")
    if curve.name in SN_CURVES and not replace_existing:
        raise CableInputError(
            f"S-N curve {curve.name!r} is already registered; pass "
            "replace_existing=True"
        )
    if not curve.reference.strip():
        raise CableInputError(
            "a registered S-N curve needs its source in 'reference' (an "
            "assumption must say so, e.g. '[ModelAssumption] ...')"
        )
    if not curve.environment.strip():
        raise CableInputError("a registered S-N curve needs an environment label")
    SN_CURVES[curve.name] = curve


def unregister_sn_curve(name: str) -> None:
    """Remove a user curve registered with :func:`register_sn_curve`."""
    if name in BUILT_IN_SN_CURVES:
        raise CableInputError(f"{name!r} is a built-in S-N curve")
    if name not in SN_CURVES:
        raise CableLookupError(f"no user S-N curve {name!r}")
    del SN_CURVES[name]
