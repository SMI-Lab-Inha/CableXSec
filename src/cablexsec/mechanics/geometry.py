# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Helical and nested-helical cable kinematics.

This module provides the geometric strain/curvature kinematics that every
downstream stress and fatigue calculation rests on. It is pure kinematics: no
material property, no stress, no calibration. Stresses are formed by
multiplying these strains by a modulus in the constitutive modules.

Two scales
----------
``HelixGeometry``
    One helical layer of ``n_wires`` wires at helix radius ``r_helix_m`` and
    lay angle ``alpha`` about the cable axis. Carries the single-helix axial
    strain (helix kinematics [Costello1997] Ch. 2-3; bending/curvature term
    [Knapp1979], [FeretBournazel1987]) and the bending stress kernel
    [MenardCartraud2023] Eq. 14.

``NestedHelixGeometry``
    A two-scale recursion: a conductor bundle wound about the cable axis
    (Level 1), and strands wound about the conductor axis (Level 2). The cable
    kinematic state is mapped to a conductor state, then the *same* single-helix
    formulas are applied at the strand scale.

Conventions
-----------
* All lengths in metres, angles in radians, curvatures in 1/m.
* ``phi`` is the wire azimuth measured from the +x axis; ``phi_bend`` is the
  azimuth of the bending plane. With biaxial curvature ``(kappa_x, kappa_y)``
  the relation ``kappa_x cos(phi) + kappa_y sin(phi) = |kappa| cos(phi - phi_bend)``
  holds, so the biaxial and scalar forms agree.
* ``lay_direction`` (``"right_hand"`` / ``"left_hand"``) sets the sign of the
  torsion-axial coupling term via :attr:`HelixGeometry.lay_sign`.

Conductor-curvature convention
------------------------------
The local curvature of the helical conductor centreline is taken as
``kappa_cond = kappa cos(alpha_c) (1 + sin^2 alpha_c) cos(phi_c - phi_bend)``,
the local-bending form consistent with [MenardCartraud2023] Eq. 14. With
``r_strand = 0`` the conductor-curvature term vanishes, so the nested model
reduces exactly to the single helix (gate G1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cablexsec.exceptions import CableInputError

__all__ = ["HelixGeometry", "NestedHelixGeometry"]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class HelixGeometry:
    """Kinematics of a single helical wire layer about the cable axis.

    Parameters
    ----------
    n_wires : int
        Number of wires in the layer.
    r_helix_m : float
        Helix (pitch-circle) radius, m.
    lay_angle_rad : float
        Lay angle ``alpha`` measured from the cable axis, rad. Positive.
    d_wire_m : float
        Wire diameter, m.
    lay_direction : str, optional
        ``"right_hand"`` (default) or ``"left_hand"``.
    """

    n_wires: int
    r_helix_m: float
    lay_angle_rad: float
    d_wire_m: float
    lay_direction: str = "right_hand"

    def __post_init__(self) -> None:
        if self.n_wires < 1:
            raise CableInputError("n_wires must be >= 1")
        if self.lay_direction not in ("right_hand", "left_hand"):
            raise CableInputError(
                f"lay_direction must be 'right_hand' or 'left_hand', "
                f"got {self.lay_direction!r}"
            )

    @property
    def lay_sign(self) -> float:
        """+1 for a right-hand lay, -1 for a left-hand lay."""
        return 1.0 if self.lay_direction == "right_hand" else -1.0

    @property
    def wire_area_m2(self) -> float:
        """Cross-sectional area of one wire, m^2."""
        return math.pi * (self.d_wire_m**2) / 4.0

    @property
    def pitch_length_m(self) -> float:
        """Axial length of one full helix turn (lay length), m.

        ``L_pitch = 2 pi r_helix / tan(alpha)``. Returns ``inf`` for a
        straight (``alpha = 0``) wire.
        """
        t = math.tan(self.lay_angle_rad)
        return math.inf if t == 0.0 else 2.0 * math.pi * self.r_helix_m / t

    def wire_azimuths(self) -> FloatArray:
        """Azimuth ``phi_k = 2 pi k / n_wires`` of each wire, rad, shape (n,)."""
        k = np.arange(self.n_wires, dtype=np.float64)
        return 2.0 * np.pi * k / self.n_wires

    def axial_strain(
        self,
        eps_s: float,
        kappa_x: float,
        kappa_y: float,
        tau_s: float,
        phi: float | FloatArray,
    ) -> FloatArray:
        """Wire axial strain from the cable strain state.

        ``eps_w = eps_s cos^2(a) + kx r cos^2(a) cos(phi) + ky r cos^2(a) sin(phi)
        + tau r cos(a) sin(a)``.

        The extension/twist terms follow the helix kinematics of
        [Costello1997] Ch. 2-3; the bending (curvature) term is the
        bent-cable wire strain of [Knapp1979] and [FeretBournazel1987].

        Parameters
        ----------
        eps_s : float
            Cable axial strain (dimensionless).
        kappa_x, kappa_y : float
            Cable curvature components, 1/m.
        tau_s : float
            Cable twist rate, 1/m.
        phi : float or numpy.ndarray
            Wire azimuth(s), rad.

        Returns
        -------
        numpy.ndarray
            Wire axial strain at each ``phi`` (0-d array for scalar input).
        """
        a = self.lay_angle_rad
        r = self.r_helix_m
        cos_a = math.cos(a)
        phi_arr = np.asarray(phi, dtype=np.float64)
        eps = (
            eps_s * cos_a**2
            + kappa_x * r * cos_a**2 * np.cos(phi_arr)
            + kappa_y * r * cos_a**2 * np.sin(phi_arr)
            + self.lay_sign * tau_s * r * cos_a * math.sin(a)
        )
        return np.asarray(eps, dtype=np.float64)

    def bending_stress(
        self,
        kappa: float,
        phi: float | FloatArray,
        E_Pa: float,
        phi_bend: float = 0.0,
    ) -> FloatArray:
        """Local wire bending stress [MenardCartraud2023] Eq. 14.

        ``sigma_B = E kappa cos(a) (1 + sin^2 a) (d/2) cos(phi - phi_bend)``.

        Parameters
        ----------
        kappa : float
            Cable curvature magnitude, 1/m.
        phi : float or numpy.ndarray
            Wire azimuth(s), rad.
        E_Pa : float
            Wire Young's modulus, Pa.
        phi_bend : float, optional
            Bending-plane azimuth, rad.

        Returns
        -------
        numpy.ndarray
            Bending stress at each ``phi``, Pa.
        """
        a = self.lay_angle_rad
        cos_a = math.cos(a)
        phi_arr = np.asarray(phi, dtype=np.float64)
        sigma = (
            E_Pa
            * kappa
            * cos_a
            * (1.0 + math.sin(a) ** 2)
            * (self.d_wire_m / 2.0)
            * np.cos(phi_arr - phi_bend)
        )
        return np.asarray(sigma, dtype=np.float64)


@dataclass(frozen=True)
class NestedHelixGeometry:
    """Two-scale nested-helix kinematics.

    Level 1 is the conductor bundle wound about the cable axis; Level 2 is the
    strand wound about the conductor axis. The cable state ``(eps_s, kappa,
    tau_s)`` is first mapped to the conductor state ``(eps_cond, kappa_cond)``
    at the conductor azimuth ``phi_c``; the single-helix formulas are then
    re-applied at the strand scale.

    Parameters
    ----------
    r_cable_m : float
        Conductor-bundle helix radius about the cable axis (Level 1), m.
    lay_angle_cable_rad : float
        Conductor-bundle lay angle ``alpha_c``, rad.
    n_strands : int
        Number of strands in the conductor (Level 2).
    r_strand_m : float
        Strand helix radius about the conductor axis, m.
    lay_angle_strand_rad : float
        Strand lay angle ``alpha_s``, rad.
    d_strand_m : float
        Strand diameter, m.
    lay_direction : str, optional
        Lay direction of the conductor bundle; ``"right_hand"`` (default).
    """

    r_cable_m: float
    lay_angle_cable_rad: float
    n_strands: int
    r_strand_m: float
    lay_angle_strand_rad: float
    d_strand_m: float
    lay_direction: str = "right_hand"

    def __post_init__(self) -> None:
        if self.n_strands < 1:
            raise CableInputError("n_strands must be >= 1")

    @property
    def _level1(self) -> HelixGeometry:
        """The conductor bundle as a single-helix layer (Level 1)."""
        # n_wires here labels the bundle helix; d is unused at this scale.
        return HelixGeometry(
            n_wires=1,
            r_helix_m=self.r_cable_m,
            lay_angle_rad=self.lay_angle_cable_rad,
            d_wire_m=0.0,
            lay_direction=self.lay_direction,
        )

    @property
    def strand_area_m2(self) -> float:
        """Cross-sectional area of one strand, m^2."""
        return math.pi * (self.d_strand_m**2) / 4.0

    def strand_azimuths(self) -> FloatArray:
        """Azimuth ``phi_s,j = 2 pi j / n_strands`` of each strand, rad."""
        j = np.arange(self.n_strands, dtype=np.float64)
        return 2.0 * np.pi * j / self.n_strands

    def conductor_axial_strain(
        self,
        eps_s: float,
        kappa_x: float,
        kappa_y: float,
        tau_s: float,
        phi_c: float,
    ) -> float:
        """Conductor-bundle axial strain at azimuth ``phi_c`` (Step 1).

        This is the single-helix axial strain (see :meth:`HelixGeometry.axial_strain`)
        applied at Level 1 with ``(alpha_c, r_cable)``.
        """
        return float(self._level1.axial_strain(eps_s, kappa_x, kappa_y, tau_s, phi_c))

    def conductor_curvature(
        self,
        kappa: float,
        phi_c: float,
        phi_bend: float = 0.0,
    ) -> float:
        """Local curvature of the conductor centreline (Step 1).

        ``kappa_cond = kappa cos(a_c) (1 + sin^2 a_c) cos(phi_c - phi_bend)``.
        """
        a_c = self.lay_angle_cable_rad
        return float(
            kappa
            * math.cos(a_c)
            * (1.0 + math.sin(a_c) ** 2)
            * math.cos(phi_c - phi_bend)
        )

    def strand_axial_strain(
        self,
        eps_s: float,
        kappa_x: float,
        kappa_y: float,
        tau_s: float,
        phi_c: float,
        phi_strand: float | FloatArray,
        kappa: float | None = None,
        phi_bend: float = 0.0,
    ) -> FloatArray:
        """Strand axial strain (Step 2, ``sigma_A,strand / E``).

        ``eps_strand = eps_cond cos^2(a_s) + kappa_cond r_s cos^2(a_s) cos(phi_s)``.

        Parameters
        ----------
        eps_s, kappa_x, kappa_y, tau_s : float
            Cable kinematic state.
        phi_c : float
            Conductor azimuth about the cable axis, rad.
        phi_strand : float or numpy.ndarray
            Strand azimuth(s) about the conductor axis, rad.
        kappa : float or None, optional
            Cable curvature magnitude used for the conductor-curvature term.
            Defaults to ``hypot(kappa_x, kappa_y)``.
        phi_bend : float, optional
            Bending-plane azimuth, rad. Defaults consistently to
            ``atan2(kappa_y, kappa_x)`` when ``kappa`` is not given.

        Returns
        -------
        numpy.ndarray
            Strand axial strain at each ``phi_strand``.

        Notes
        -----
        With ``r_strand_m = 0`` (and ``alpha_s = 0``) this reduces exactly to
        :meth:`conductor_axial_strain` — the G1 consistency gate.
        """
        if kappa is None:
            kappa = math.hypot(kappa_x, kappa_y)
            if kappa_x != 0.0 or kappa_y != 0.0:
                phi_bend = math.atan2(kappa_y, kappa_x)
        eps_cond = self.conductor_axial_strain(eps_s, kappa_x, kappa_y, tau_s, phi_c)
        kappa_cond = self.conductor_curvature(kappa, phi_c, phi_bend)
        a_s = self.lay_angle_strand_rad
        cos_as = math.cos(a_s)
        phi_arr = np.asarray(phi_strand, dtype=np.float64)
        eps = eps_cond * cos_as**2 + kappa_cond * self.r_strand_m * cos_as**2 * np.cos(
            phi_arr
        )
        return np.asarray(eps, dtype=np.float64)
