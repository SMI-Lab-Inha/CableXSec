# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Single source of truth for material properties.

Every physical material constant used anywhere in CableXSec lives in this
module. No physical constant may be written inside a function body elsewhere
in the library.

Each property is stored as a ``(value, citation)`` pair, where ``value`` is
a float in strict SI units (the key carries the unit suffix, e.g. ``E_Pa``,
``alpha_T_K``) and ``citation`` identifies the source or explicitly states a
model assumption. An assumption label is not a literature validation.

Access values through :func:`get_property` (returns the float) and provenance
through :func:`get_citation` (returns the source string). Both raise a
:class:`KeyError` with an actionable message for unknown materials or keys.
Temperature-dependent polymer modulus is evaluated through
:func:`youngs_modulus_at_temperature`. That function is deliberately bounded
to the temperature interval exercised by its source model. It does not
silently extrapolate a polymer fit.

Cable materials
---------------
A cable file may define its own materials (``materials:`` in the cable
YAML). They never enter :data:`MATERIALS`: a :class:`MaterialOverlay` holds
them, and :func:`material_scope` makes it the active overlay of the current
context (a :class:`contextvars.ContextVar`, so each thread and each task has
its own). Every lookup of this module resolves a key in the active overlay
first and in the library second, so a cable material with the key of a
library material replaces it for that cable only. Outside a scope the
library alone answers.

Examples
--------
>>> get_property("galvanised_steel_armour", "E_Pa")
210000000000.0
>>> get_citation("galvanised_steel_armour", "E_Pa")
'[DNVGL-ST-0359] Table 4-1'
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType

from cablexsec.exceptions import CableInputError, CableLookupError

# A material property entry: the SI value and its literature citation.
Property = tuple[float, str]
# A material: a mapping from unit-suffixed property name to its entry.
Material = dict[str, Property]

# Polynomial coefficients are in MPa with temperature in degrees Celsius.
# The interval is the range exercised in Qiao et al. (2025), Sections 3.1--3.3.
# Qiao et al. reproduce the fits from Yan et al. (2021) as their Eq. (9).
TemperatureModulusModel = tuple[tuple[float, ...], float, float, str]
TEMPERATURE_MODULUS_MODELS: dict[str, TemperatureModulusModel] = {
    "xlpe_qiao2025": (
        (0.027, -4.64, 232.0),
        10.0,
        90.0,
        "[Qiao2025] Eq. 9, after [Yan2021]; exercised over 10--90 degC",
    ),
    "hdpe": (
        (0.0001, 0.047, -16.7, 1200.0),
        10.0,
        90.0,
        "[Qiao2025] Eq. 9, after [Yan2021]; exercised over 10--90 degC",
    ),
}

#: Temperature at which the library moduli ``E_Pa`` are stated [deg C]. The
#: registered ``E(T)`` laws reproduce ``E_Pa`` here ([Qiao2025] Eq. 9 at
#: 20 degC), and it is the laboratory ambient of a rig test.
MODULUS_REFERENCE_TEMPERATURE_C = 20.0

#: Polymer (thermoplastic and elastomer) materials of the library. Their
#: moduli depend on temperature and relax under sustained stress; only the
#: materials in :data:`TEMPERATURE_MODULUS_MODELS` carry an ``E(T)`` law.
POLYMER_MATERIALS: frozenset[str] = frozenset(
    {
        "xlpe_qiao2025",
        "hdpe",
        "pp_filler",
        "polyester_tape",
        "xlpe_menard2023",
        "hdpe_menard2023",
        "pp_menard2023",
        "pfr_menard2023",
        "xlpe_janocha2024",
        "mdpe_janocha2024",
        "hdpe_janocha2024",
        "ppy_janocha2024",
        "core_tube_chae2023",
        "filler_chae2023",
        "yarn_chae2023",
    }
)


def is_polymer(material: str) -> bool:
    """True for a polymer material (:data:`POLYMER_MATERIALS`, or a cable material
    declared ``polymer`` in the active overlay)."""
    overlay = _ACTIVE_OVERLAY.get()
    if overlay is not None and material in overlay:
        return material in overlay.polymers
    return material in POLYMER_MATERIALS


# ---------------------------------------------------------------------------
# Cable materials: an overlay resolved before the library, per context.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MaterialOverlay:
    """Materials of one cable, resolved before the library while it is active.

    ``materials`` maps a key to its ``(value, citation)`` properties,
    ``polymers`` names the polymer materials among them and
    ``temperature_laws`` gives the ``E(T)`` law (a key of
    :data:`TEMPERATURE_MODULUS_MODELS`) of a material that has one. The
    mappings are read-only copies, so an overlay never changes once made.
    """

    materials: Mapping[str, Mapping[str, Property]]
    polymers: frozenset[str] = frozenset()
    temperature_laws: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frozen = {
            name: MappingProxyType(dict(entry))
            for name, entry in self.materials.items()
        }
        object.__setattr__(self, "materials", MappingProxyType(frozen))
        object.__setattr__(
            self, "temperature_laws", MappingProxyType(dict(self.temperature_laws))
        )
        unknown = (self.polymers | set(self.temperature_laws)) - set(frozen)
        if unknown:
            raise CableInputError(
                f"overlay names materials it does not define: {sorted(unknown)}"
            )
        for name, law in self.temperature_laws.items():
            if law not in TEMPERATURE_MODULUS_MODELS:
                raise CableInputError(
                    f"material {name!r}: {law!r} is not a registered "
                    "temperature-modulus law"
                )

    def __contains__(self, material: object) -> bool:
        return material in self.materials


_ACTIVE_OVERLAY: ContextVar[MaterialOverlay | None] = ContextVar(
    "cablexsec_material_overlay", default=None
)


@contextmanager
def material_scope(overlay: MaterialOverlay | None) -> Iterator[None]:
    """Resolve material keys in ``overlay`` first while the block runs.

    ``None`` makes the library alone answer, also inside an outer scope.
    The overlay belongs to the current context only: another thread, or
    another task, keeps its own.
    """
    token = _ACTIVE_OVERLAY.set(overlay)
    try:
        yield
    finally:
        _ACTIVE_OVERLAY.reset(token)


def active_overlay() -> MaterialOverlay | None:
    """The overlay of the current context (``None`` outside a scope)."""
    return _ACTIVE_OVERLAY.get()


# ---------------------------------------------------------------------------
# Material property database.
#
# This module-level mapping IS the designated data store for material
# constants; the literal values here are data, not magic numbers in code.
# ---------------------------------------------------------------------------
#: Provenance of the 0.20 armour-on-anti-wear-tape coefficient. Dai, Saevik
#: and Ye, Marine Structures 55 (2017) 137-161, doi
#: 10.1016/j.marstruc.2017.05.010, Sec. 4.3.1, p. 156: "The friction
#: coefficient between anti-wear tape and inner armor, and between two layers
#: of steel armors was estimated to be 0.20 at 20 MPa internal pressure, see
#: Saevik [21]" ([21] the 1992 thesis). Dai, Saevik and Ye, Marine Structures
#: 59 (2018) 423-443, Sec. 4.1, p. 428: at that internal pressure the contact
#: pressure is 4 MPa (inner wire on tape) and 2 MPa (outer wire on tape), and
#: the measured dynamic (static) coefficient of armour wire on in-service tape
#: is 0.32 (0.38) at 1 MPa, 0.20 (0.24) at 4 MPa and 0.18 (0.22) at 8 MPa.
_DAI2017_ANTIWEAR_TAPE = (
    "[DaiSaevikYe2017] Sec. 4.3.1, p. 156: 0.20 estimated between anti-wear tape "
    "and inner armour and between two steel armour layers 'at 20 MPa internal "
    "pressure', quoted from [Saevik1992]; an anti-wear tape interface, not HDPE. "
    "The contact pressure at that internal pressure is 2-4 MPa; the measured "
    "mu_d of armour wire on in-service tape is 0.32, 0.20 and 0.18 at 1, 4 and "
    "8 MPa ([DaiSaevikYe2018] Sec. 4.1, p. 428)"
)
_DAI2017_STEEL = (
    "[DaiSaevikYe2017] Sec. 4.3.1, p. 156: 0.20 estimated between two steel "
    "armour layers (and between anti-wear tape and inner armour) 'at 20 MPa "
    "internal pressure', quoted from [Saevik1992]"
)

MATERIALS: dict[str, Material] = {
    "galvanised_steel_armour": {
        "E_Pa": (210.0e9, "[DNVGL-ST-0359] Table 4-1"),
        "nu": (0.30, "[DNVGL-ST-0359] Table 4-1"),
        "rho_kg_m3": (7850.0, "[DNVGL-ST-0359]"),
        "sigma_y_Pa": (690.0e6, "[DNVGL-ST-0359] Table 4-1, 0.2% proof"),
        "sigma_UTS_Pa": (1400.0e6, "[DNVGL-ST-0359] Table 4-1"),
        "alpha_T_K": (12.0e-6, "[DNVGL-ST-0359]"),
        "lambda_W_mK": (50.0, "Standard steel"),
        "rho_cp_J_m3K": (3.76e6, "Standard steel"),
        # Interface: armour wire on a polymer anti-wear tape (not measured on
        # HDPE). The 20 MPa is the internal pressure of the full-scale test,
        # not a contact pressure.
        "mu_on_hdpe": (
            0.20,
            _DAI2017_ANTIWEAR_TAPE,
        ),
        "mu_on_steel": (
            0.12,
            "[RaoofHuang1992] galvanised-wire large-scale tests; cf. [Perier2009]",
        ),
    },
    "copper_stranded": {
        # NB: this is the copper MATERIAL Young's modulus (lay reduction applied
        # separately via cos^3(alpha_strand)), not a homogenized conductor modulus.
        "E_eff_axial_Pa": (111.0e9, "[Smith2026] Cu material modulus, 110-112 GPa"),
        "E_eff_bending_Pa": (
            5.0e9,
            "[ModelAssumption] effective individual-conductor bending modulus; "
            "not a measured copper modulus or a Knapp1979 result",
        ),
        "nu": (0.34, "Standard Cu"),
        "rho_kg_m3": (8960.0, "Standard Cu"),
        # The 0.2 % proof stress of ETP copper, as for copper_janocha2024: a
        # published value for the annealed ETP copper of power-cable
        # conductors.
        "sigma_y_Pa": (
            120.0e6,
            "[Janocha2024] Sec. 2.1, p. 5: 0.2 % plastic-strain limit (proof "
            "stress) of the ETP copper conductor, 120 MPa, after Robinson; a "
            "published value for ETP copper",
        ),
        "sigma_UTS_Pa": (
            220.0e6,
            "Standard annealed Cu (IEC 60228 sets no tensile limit for copper; "
            "its Tables 1-2 cover aluminium only)",
        ),
        "alpha_T_K": (17.0e-6, "Standard Cu"),
        "lambda_W_mK": (400.0, "Standard Cu"),
        "rho_cp_J_m3K": (3.45e6, "Standard Cu"),
        "rho_ohm_m": (1.7241e-8, "[IEC60287-1-1] Table 1, copper at 20 degC"),
        "alpha_R_per_K": (3.93e-3, "[IEC60287-1-1] Table 1, copper"),
        # F6 citation fix (18 Oct 2026): [Poon2022] contains no 0.25 (it uses
        # 0.8, range 0.08-1.4). No library source gives 0.25 for Cu-Cu
        # stranded wire, so the value is kept (results unchanged) as an
        # assumption inside the band of the measured and used values.
        "mu_on_cu": (
            0.25,
            "[ModelAssumption] Cu wire on Cu wire, bound 0.1-0.5: [Nasution2014b] "
            "Sec. 2.5 measured 0.2 unlubricated (0.02 lubricated); [HuEtAl2022] "
            "use 0.2 with a 0.1-0.5 sensitivity; dry fretting contacts reach "
            "0.6-1.5 at steady state ([Poon2023] Table 1); [Poon2022] Table 1 "
            "uses 0.8 and contains no 0.25",
        ),
    },
    "copper_wire_screen": {
        "E_Pa": (117.0e9, "Standard annealed Cu; not a measured screen-bundle modulus"),
        "nu": (0.34, "Standard Cu"),
        "rho_kg_m3": (8960.0, "Standard Cu"),
        "alpha_T_K": (17.0e-6, "Standard Cu"),
        "lambda_W_mK": (400.0, "Standard Cu"),
        "rho_cp_J_m3K": (3.45e6, "Standard Cu"),
        "rho_ohm_m": (1.7241e-8, "[IEC60287-1-1] Table 1, copper at 20 degC"),
        "alpha_R_per_K": (3.93e-3, "[IEC60287-1-1] Table 1, copper"),
        # Gap P1/P4. Cold-drawn and annealed ETP Cu wire, lowest strain rate
        # (1e-4 1/s) of [Wan2022] Table 2 (range 145-155 / 235-262 MPa).
        "sigma_y_Pa": (145.0e6, "[Wan2022] Table 2, annealed ETP Cu wire, 1e-4 1/s"),
        "sigma_UTS_Pa": (235.0e6, "[Wan2022] Table 2, annealed ETP Cu wire, 1e-4 1/s"),
        "mu_on_cu": (
            0.20,
            "[Nasution2014b] Sec. 2.5, measured Cu wire on Cu wire (Int. J. "
            "Fatigue 59); [Poon2022] Table 1 uses 0.8 (range 0.08 greased to "
            "1.4 dry)",
        ),
        "mu_on_hdpe": (
            0.20,
            "[ModelAssumption] Cu wire on polymer or semiconducting tape, bound "
            "0.1-0.3; no measured Cu-polymer coefficient in the reference library",
        ),
    },
    "copper_tape_screen": {
        "E_Pa": (117.0e9, "Standard annealed Cu (tape screen)"),
        "nu": (0.34, "Standard Cu"),
        "rho_kg_m3": (8960.0, "Standard Cu"),
        "alpha_T_K": (17.0e-6, "Standard Cu"),
        "lambda_W_mK": (400.0, "Standard Cu"),
        "rho_cp_J_m3K": (3.45e6, "Standard Cu"),
        "rho_ohm_m": (1.7241e-8, "[IEC60287-1-1] Table 1, copper at 20 degC"),
        "alpha_R_per_K": (3.93e-3, "[IEC60287-1-1] Table 1, copper"),
    },
    "xlpe_qiao2025": {
        "E_Pa": (150.0e6, "[Qiao2025] Eq. 9 evaluated at 20 degC"),
        "nu": (0.45, "[Qiao2025] Table 2"),
        "rho_kg_m3": (920.0, "Standard XLPE"),
        "alpha_T_K": (100.0e-6, "[Qiao2025] Table 2"),
        "lambda_W_mK": (0.286, "[IEC60287]"),
        "rho_cp_J_m3K": (2.4e6, "Standard XLPE"),
        "epsilon_r": (2.5, "[IEC60287-1-1] Table 3, XLPE (2.3 is the PE row)"),
    },
    "hdpe": {
        "E_Pa": (885.6e6, "[Qiao2025] Eq. 9 evaluated at 20 degC"),
        "nu": (0.48, "[Qiao2025] Table 2"),
        "rho_kg_m3": (955.0, "Standard HDPE"),
        "alpha_T_K": (100.0e-6, "[Qiao2025] Table 2"),
        "lambda_W_mK": (
            1.0 / 3.5,
            "[IEC60287-2-1] Table 1: polyethylene 3.5 K m/W (insulation, "
            "protective covering)",
        ),
        "rho_cp_J_m3K": (1.9e6, "Standard HDPE"),
        "epsilon_allow": (
            0.077,
            "[API17J] (2008) Table 6, p. 19: the 7.7 % maximum allowable strain "
            "of the PE sheaths of unbonded flexible pipe, applied to the PE "
            "sheaths of a power cable; no power-cable criterion is published "
            "([DNVGL-ST-0119] 16.2.2, p. 149)",
        ),
    },
    "pp_filler": {
        "E_Pa": (1.5e9, "Standard PP"),
        "nu": (0.40, "Standard PP"),
        "rho_kg_m3": (905.0, "Standard PP"),
        "alpha_T_K": (1.5e-4, "Standard PP"),
        "lambda_W_mK": (
            1.0 / 6.0,
            "[IEC60287-2-1] 4.1.2.2.4: fillers of extruded-insulation cables "
            "6 to 13 K m/W, the lower value; 6.0 for polypropylene yarn in "
            "CIGRE TB 880 (2022) Table 2-3",
        ),
        "rho_cp_J_m3K": (1.8e6, "Standard PP"),
    },
    "polyester_tape": {
        # Binder and armour-bedding tapes whose grade is not reported:
        # standard PET film values, a model
        # assumption. A thin tape adds little axial stiffness; it enters the
        # radial chain between the layers it separates.
        "E_Pa": (3.0e9, "[ModelAssumption] standard PET film; tape grade not reported"),
        "nu": (0.38, "[ModelAssumption] standard PET"),
        "rho_kg_m3": (1380.0, "[ModelAssumption] standard PET"),
        "alpha_T_K": (7.0e-5, "[ModelAssumption] standard PET"),
        "lambda_W_mK": (0.20, "[ModelAssumption] standard PET"),
        "rho_cp_J_m3K": (1.7e6, "[ModelAssumption] standard PET"),
    },
    # ----- BM1 Ménard & Cartraud (2023) cable, Table 2 moduli -----
    # This cable's polymers and (solid-cylinder) conductor use the moduli
    # tabulated by the source; they differ from the generic Phase-1 grades
    # above, so they are stored as cable-specific materials (cited).
    "copper_menard2023": {
        "E_Pa": (110.0e9, "[MenardCartraud2023] Table 2 (solid-cylinder conductor)"),
        "nu": (0.30, "[MenardCartraud2023] Table 2"),
    },
    "xlpe_menard2023": {
        "E_Pa": (350.0e6, "[MenardCartraud2023] Table 2"),
        "nu": (0.40, "[MenardCartraud2023] Table 2"),
        # Thermal properties are not reported by this mechanical benchmark.
        # They are intentionally absent rather than borrowed from another grade.
    },
    "hdpe_menard2023": {
        "E_Pa": (1380.0e6, "[MenardCartraud2023] Table 2"),
        "nu": (0.40, "[MenardCartraud2023] Table 2"),
        "epsilon_allow": (
            0.077,
            "[API17J] (2008) Table 6, p. 19: the 7.7 % maximum allowable strain "
            "of the PE sheaths of unbonded flexible pipe, applied to the PE "
            "sheaths of a power cable; no power-cable criterion is published "
            "([DNVGL-ST-0119] 16.2.2, p. 149)",
        ),
    },
    "pp_menard2023": {
        "E_Pa": (1300.0e6, "[MenardCartraud2023] Table 2"),
        "nu": (0.40, "[MenardCartraud2023] Table 2"),
    },
    "pfr_menard2023": {
        "E_Pa": (512.0e6, "[MenardCartraud2023] Table 2 (perfluoro-elastomer bedding)"),
        "nu": (0.40, "[MenardCartraud2023] Table 2"),
    },
    # ----- Public 66 kV wet-design template: Janocha et al. (2024) -----
    # Density, modulus, Poisson ratio and friction coefficient are Table 2 of
    # [Janocha2024] (the IEA Wind Task 49 reference cable). Properties that the
    # source does not report (thermal, electrical, ultimate strength) repeat
    # the cited values of the generic grades above or are marked as model
    # assumptions; none is fitted to a published cable result.
    "copper_janocha2024": {
        "E_Pa": (112.2e9, "[Janocha2024] Table 2 (copper)"),
        "E_eff_axial_Pa": (
            112.2e9,
            "[Janocha2024] Table 2 copper MATERIAL modulus (lay reduction separate)",
        ),
        "E_eff_bending_Pa": (
            5.0e9,
            "[ModelAssumption] effective individual-conductor bending modulus, "
            "as copper_stranded; not reported by Janocha2024",
        ),
        "nu": (0.34, "[Janocha2024] Table 2 (copper)"),
        "rho_kg_m3": (8890.0, "[Janocha2024] Table 2 (copper)"),
        # The 0.2 % plastic limit is the conductor's stress limit; the source's
        # 250 MPa yield strength is not used (Janocha takes the 120 MPa plastic
        # limit as the conductor capacity, Sec. 3.3, Fig. 21).
        "sigma_y_Pa": (
            120.0e6,
            "[Janocha2024] Sec. 2.1, p. 5: 0.2 % plastic-strain limit (proof "
            "stress) of the ETP copper conductor, 120 MPa, after Robinson",
        ),
        "sigma_UTS_Pa": (220.0e6, "Standard annealed Cu, as copper_stranded"),
        "alpha_T_K": (17.0e-6, "Standard Cu, as copper_stranded"),
        "lambda_W_mK": (400.0, "Standard Cu, as copper_stranded"),
        "rho_cp_J_m3K": (3.45e6, "Standard Cu, as copper_stranded"),
        "rho_ohm_m": (1.7241e-8, "[IEC60287-1-1] Table 1, copper at 20 degC"),
        "alpha_R_per_K": (3.93e-3, "[IEC60287-1-1] Table 1, copper"),
        "mu_on_cu": (0.30, "[Janocha2024] Table 2 friction coefficient (copper)"),
    },
    "steel_janocha2024": {
        "E_Pa": (200.0e9, "[Janocha2024] Table 2 (steel)"),
        "nu": (0.26, "[Janocha2024] Table 2 (steel)"),
        "rho_kg_m3": (7800.0, "[Janocha2024] Table 2 (steel)"),
        "sigma_y_Pa": (375.0e6, "[Janocha2024] Section 2.1, armour wire yield"),
        "sigma_UTS_Pa": (
            470.0e6,
            "[ModelAssumption] 1.25 x the 375 MPa yield; not reported by "
            "Janocha2024; enters only the thermal yield scaling of the capacity "
            "check",
        ),
        "alpha_T_K": (12.0e-6, "[DNVGL-ST-0359], as galvanised_steel_armour"),
        "lambda_W_mK": (50.0, "Standard steel, as galvanised_steel_armour"),
        "rho_cp_J_m3K": (3.76e6, "Standard steel, as galvanised_steel_armour"),
        "mu_on_hdpe": (0.20, "[Janocha2024] Table 2 friction coefficient (steel)"),
        "mu_on_steel": (0.20, "[Janocha2024] Table 2 friction coefficient (steel)"),
    },
    "xlpe_janocha2024": {
        "E_Pa": (1000.0e6, "[Janocha2024] Table 2 (XLPE)"),
        "nu": (0.40, "[Janocha2024] Table 2 (XLPE)"),
        "rho_kg_m3": (925.0, "[Janocha2024] Table 2 (XLPE)"),
        "alpha_T_K": (100.0e-6, "[Qiao2025] Table 2, as xlpe_qiao2025"),
        "lambda_W_mK": (0.286, "[IEC60287], as xlpe_qiao2025"),
        "rho_cp_J_m3K": (2.4e6, "Standard XLPE, as xlpe_qiao2025"),
        "epsilon_r": (2.5, "[IEC60287-1-1] Table 3, XLPE above 18/30 kV"),
    },
    "mdpe_janocha2024": {
        "E_Pa": (1000.0e6, "[Janocha2024] Table 2 (MDPE)"),
        "nu": (0.40, "[Janocha2024] Table 2 (MDPE)"),
        "rho_kg_m3": (956.0, "[Janocha2024] Table 2 (MDPE)"),
        "alpha_T_K": (100.0e-6, "[Qiao2025] Table 2 (HDPE), as hdpe"),
        "lambda_W_mK": (
            1.0 / 3.5,
            "[IEC60287-2-1] Table 1: polyethylene 3.5 K m/W (insulation, "
            "protective covering), as hdpe",
        ),
        "rho_cp_J_m3K": (1.9e6, "Standard HDPE, as hdpe"),
        "epsilon_allow": (
            0.077,
            "[API17J] (2008) Table 6, p. 19: the 7.7 % maximum allowable strain "
            "of the PE sheaths of unbonded flexible pipe, applied to the PE "
            "sheaths of a power cable; no power-cable criterion is published "
            "([DNVGL-ST-0119] 16.2.2, p. 149)",
        ),
    },
    "hdpe_janocha2024": {
        "E_Pa": (1000.0e6, "[Janocha2024] Table 2 (HDPE)"),
        "nu": (0.40, "[Janocha2024] Table 2 (HDPE)"),
        "rho_kg_m3": (980.0, "[Janocha2024] Table 2 (HDPE)"),
        "alpha_T_K": (100.0e-6, "[Qiao2025] Table 2, as hdpe"),
        "lambda_W_mK": (
            1.0 / 3.5,
            "[IEC60287-2-1] Table 1: polyethylene 3.5 K m/W (insulation, "
            "protective covering), as hdpe",
        ),
        "rho_cp_J_m3K": (1.9e6, "Standard HDPE, as hdpe"),
        "epsilon_allow": (
            0.077,
            "[API17J] (2008) Table 6, p. 19: the 7.7 % maximum allowable strain "
            "of the PE sheaths of unbonded flexible pipe, applied to the PE "
            "sheaths of a power cable; no power-cable criterion is published "
            "([DNVGL-ST-0119] 16.2.2, p. 149)",
        ),
    },
    "ppy_janocha2024": {
        # Labelled polypyrrole in the source footnote; the 895 kg/m3 density
        # and 150 MPa modulus are those of a polypropylene-yarn bedding.
        "E_Pa": (150.0e6, "[Janocha2024] Table 2 (PPY)"),
        "nu": (0.40, "[Janocha2024] Table 2 (PPY)"),
        "rho_kg_m3": (895.0, "[Janocha2024] Table 2 (PPY)"),
        "alpha_T_K": (1.5e-4, "Standard PP, as pp_filler"),
        "lambda_W_mK": (
            1.0 / 6.0,
            "CIGRE TB 880 (2022) Table 2-3: polypropylene yarn 6.0 K m/W, the "
            "fibrous materials of [IEC60287-2-1] Table 1",
        ),
        "rho_cp_J_m3K": (1.8e6, "Standard PP, as pp_filler"),
    },
    # ----- Public flexible-pipe benchmark: Witz (1996) ISSC 2.5-inch pipe -----
    # Polymer moduli are Witz Table 2. Steel moduli are not in Witz; Ramos
    # and Pesce (2004, Section 4) assumed them and the ISSC simple-model rows
    # (Witz Tables 3-6) are consistent with 207 GPa. Poisson ratio 0.3 is the
    # nominal value of Witz. Mechanical benchmark only: no thermal data.
    "aisi304_witz1996": {
        "E_Pa": (190.0e9, "[RamosPesce2004] Sec. 4 (assumed in source)"),
        "nu": (0.30, "[Witz1996] nominal 0.3"),
    },
    "fi15_steel_witz1996": {
        "E_Pa": (207.0e9, "[RamosPesce2004] Sec. 4 (assumed in source)"),
        "nu": (0.30, "[Witz1996] nominal 0.3"),
    },
    "fi41_steel_witz1996": {
        "E_Pa": (207.0e9, "[RamosPesce2004] Sec. 4 (assumed in source)"),
        "nu": (0.30, "[Witz1996] nominal 0.3"),
    },
    "pa12_grilamid_witz1996": {
        "E_Pa": (284.0e6, "[Witz1996] Table 2 (Grilamid nylon 12)"),
        "nu": (0.30, "[Witz1996] nominal polymer Poisson ratio"),
    },
    "pa11_rilsan_witz1996": {
        "E_Pa": (301.0e6, "[Witz1996] Table 2 (Rilsan nylon 11)"),
        "nu": (0.30, "[Witz1996] nominal polymer Poisson ratio"),
    },
    "fabric_tape_witz1996": {
        "E_Pa": (
            300.0e6,
            "[Witz1996] Table 2 transverse stiffness (pipe-axial direction "
            "of an 84 deg tape), as [RamosPesce2004]",
        ),
        "nu": (0.30, "[Witz1996] nominal 0.3"),
    },
    # ----- Public flexible-pipe benchmark: Saevik (2011) 8-inch FBG riser -----
    # The source gives no material data. Steel modulus as the Witz (1996)
    # armour pair. The single polymer modulus (pressure sheath, tapes, outer
    # sheath) is identified so that the full-slip EI_min of saevik2011.yaml
    # equals the tabulated "elastic bending stiffness (20 degC)" of 72.8 kN m2
    # ([Saevik2011] Table 1), a section property independent of the compared
    # armour stresses. Friction 0.20: the estimate between anti-wear tape and
    # inner armour and between two steel armour layers "at 20 MPa internal
    # pressure" of [DaiSaevikYe2017] Sec. 4.3.1, p. 156, quoted from
    # [Saevik1992] (_DAI2017_ANTIWEAR_TAPE); not identified on this test.
    "steel_saevik2011": {
        "E_Pa": (
            207.0e9,
            "[ModelAssumption] as fi41_steel_witz1996; not in [Saevik2011]",
        ),
        "nu": (0.30, "[ModelAssumption] nominal steel"),
        "alpha_T_K": (12.0e-6, "[DNVGL-ST-0359], as galvanised_steel_armour"),
        "sigma_UTS_Pa": (
            1400.0e6,
            "[ModelAssumption] as galvanised_steel_armour; not in [Saevik2011]",
        ),
        "mu_on_hdpe": (0.20, _DAI2017_ANTIWEAR_TAPE),
        "mu_on_steel": (0.20, _DAI2017_STEEL),
    },
    "polymer_saevik2011": {
        "E_Pa": (
            455.8e6,
            "[Identified] on the full-slip EI 72.8 kN m2 of [Saevik2011] Table 1 "
            "(scripts/check_saevik2011.py identify_polymer_modulus)",
        ),
        "nu": (0.40, "[ModelAssumption] nominal polymer"),
        "alpha_T_K": (100.0e-6, "[ModelAssumption] nominal polymer"),
    },
    # Same identification for the rectangular-tendon definition
    # (saevik2011_rect.yaml, gap G1): loxodromic full-slip state of the strips.
    "polymer_saevik2011_rect": {
        "E_Pa": (
            410.6e6,
            "[Identified] on the full-slip EI 72.8 kN m2 of [Saevik2011] Table 1 "
            "with rectangular tendons (scripts/check_saevik2011.py "
            "identify_polymer_modulus --geometry rect)",
        ),
        "nu": (0.40, "[ModelAssumption] nominal polymer"),
        "alpha_T_K": (100.0e-6, "[ModelAssumption] nominal polymer"),
    },
    # ----- Public model-to-model case: Chae, Nam and Shim (2023) coiling -----
    # Artificial single-armour three-core cable (coiling case). Moduli,
    # Poisson ratios and yield stress from [ChaeNamShim2023] Table 2 and the
    # equivalent power core of Sec. 2.1.2; friction 0.2 between helical layers
    # (Sec. 2.1.3). Properties the source does not give are assumptions that
    # enter no coiling result (strength for the fatigue adapter, expansion).
    "steel_chae2023": {
        "E_Pa": (200.0e9, "[ChaeNamShim2023] Table 2"),
        "nu": (0.30, "[ChaeNamShim2023] Table 2"),
        "rho_kg_m3": (7800.0, "[ChaeNamShim2023] Table 2, specific gravity 7.80"),
        "sigma_y_Pa": (400.0e6, "[ChaeNamShim2023] Table 2"),
        "sigma_UTS_Pa": (
            550.0e6,
            "[ModelAssumption] not given by [ChaeNamShim2023]; "
            "used by no coiling result",
        ),
        "alpha_T_K": (12.0e-6, "[DNVGL-ST-0359], as galvanised_steel_armour"),
        "mu_on_hdpe": (0.20, "[ChaeNamShim2023] Sec. 2.1.3, all helical layers"),
        "mu_on_steel": (0.20, "[ChaeNamShim2023] Sec. 2.1.3, all helical layers"),
    },
    "core_rod_chae2023": {
        "E_Pa": (77.0e9, "[ChaeNamShim2023] Sec. 2.1.2, equivalent conductor rod"),
        "nu": (0.43, "[ChaeNamShim2023] Sec. 2.1.2"),
        "alpha_T_K": (17.0e-6, "[ModelAssumption] as copper; isothermal case"),
    },
    "core_tube_chae2023": {
        "E_Pa": (506.0e6, "[ChaeNamShim2023] Sec. 2.1.2, equivalent core tube"),
        "nu": (0.47, "[ChaeNamShim2023] Sec. 2.1.2"),
        "alpha_T_K": (100.0e-6, "[ModelAssumption] nominal polymer; isothermal case"),
    },
    "filler_chae2023": {
        "E_Pa": (882.0e6, "[ChaeNamShim2023] Table 2, sheath and filler"),
        "nu": (0.45, "[ChaeNamShim2023] Table 2"),
        "alpha_T_K": (100.0e-6, "[ModelAssumption] nominal polymer; isothermal case"),
    },
    "yarn_chae2023": {
        "E_Pa": (1500.0e6, "[ChaeNamShim2023] Table 2, 1st and 2nd yarn"),
        "nu": (0.45, "[ChaeNamShim2023] Table 2"),
        "alpha_T_K": (100.0e-6, "[ModelAssumption] nominal polymer; isothermal case"),
    },
    # ----- Phase 2 additions (metallic-sheath export cables) -----
    "lead_sheath": {
        "E_Pa": (16.0e9, "Standard Pb"),
        "nu": (0.44, "Standard Pb"),
        "rho_kg_m3": (11340.0, "Standard Pb"),
        "sigma_y_Pa": (12.0e6, "Standard Pb, very low yield"),
        "alpha_T_K": (29.0e-6, "Standard Pb, high expansion"),
        "lambda_W_mK": (35.0, "Standard Pb (IEC 60287-2-1 lists no metal resistivity)"),
        "rho_cp_J_m3K": (1.47e6, "Standard Pb"),
        "rho_ohm_m": (21.4e-8, "[IEC60287-1-1] Table 1, lead or lead alloy"),
        "alpha_R_per_K": (4.0e-3, "[IEC60287-1-1] Table 1, lead or lead alloy"),
        "creep_exponent": (4.5, "[CIGRE TB 623] Norton creep"),
        "creep_coeff": (1.2e-14, "[CIGRE TB 623] at 50 degC"),
    },
    # Gap P6. Smooth aluminium foil of an Al-PE laminate sheath, bonded to
    # the polymer: "only by this bonding, the aluminium sheath is able to
    # withstand the cable bending without wrinkles or creases" ([Worzyk2009]
    # Sec. 2.3.2, foil 0.1-0.3 mm). Alloy and temper are not published for
    # the cables in the library; strength values are assumed (O temper).
    "aluminium_laminate_foil": {
        "E_Pa": (70.0e9, "Standard Al, as aluminium_corrugated"),
        "nu": (0.33, "Standard Al"),
        "rho_kg_m3": (2700.0, "Standard Al"),
        "sigma_y_Pa": (
            40.0e6,
            "[ModelAssumption] O-temper foil 0.2% proof, bound 30-95 MPa; no "
            "foil property in the reference library",
        ),
        "sigma_UTS_Pa": (
            90.0e6,
            "[ModelAssumption] O-temper foil, bound 70-110 MPa; no foil property "
            "in the reference library",
        ),
        "alpha_T_K": (23.0e-6, "Standard Al"),
        "lambda_W_mK": (
            230.0,
            "Standard Al (IEC 60287-2-1 lists no metal resistivity)",
        ),
        "rho_cp_J_m3K": (2.43e6, "Standard Al"),
        "rho_ohm_m": (2.84e-8, "[IEC60287-1-1] Table 1, aluminium sheath"),
        "alpha_R_per_K": (4.03e-3, "[IEC60287-1-1] Table 1, aluminium sheath"),
    },
    "aluminium_corrugated": {
        "E_Pa": (70.0e9, "Standard Al 1050"),
        "E_corrugated_Pa": (
            35.0e9,
            "Representative modulus (factor 0.5), not a measured sheath property",
        ),
        "nu": (0.33, "Standard Al"),
        "rho_kg_m3": (2700.0, "Standard Al"),
        "sigma_y_Pa": (95.0e6, "1050-H14 temper"),
        "alpha_T_K": (23.0e-6, "Standard Al"),
        "lambda_W_mK": (
            230.0,
            "Standard Al (IEC 60287-2-1 lists no metal resistivity)",
        ),
        "rho_cp_J_m3K": (2.43e6, "Standard Al"),
        "rho_ohm_m": (2.84e-8, "[IEC60287-1-1] Table 1, aluminium sheath"),
        "alpha_R_per_K": (4.03e-3, "[IEC60287-1-1] Table 1, aluminium sheath"),
    },
}


#: Names of the built-in materials; user registrations cannot replace them.
BUILT_IN_MATERIALS: frozenset[str] = frozenset(MATERIALS)


def register_material(
    name: str,
    properties: dict[str, Property],
    *,
    replace_existing: bool = False,
    polymer: bool = False,
) -> None:
    """Register a user material for this process (gap P4).

    Every property is a ``(value, citation)`` pair in SI units with a
    unit-suffixed key, as the built-in entries; a citation is required (an
    assumption must say so, e.g. ``"[ModelAssumption] ..."``). Built-in
    materials cannot be replaced; a user material only with
    ``replace_existing=True``. ``polymer=True`` adds the name to
    :data:`POLYMER_MATERIALS` (temperature warnings apply).

    Examples
    --------
    >>> register_material("my_tape", {"E_Pa": (2.0e9, "[ModelAssumption] tape")})
    >>> get_property("my_tape", "E_Pa")
    2000000000.0
    >>> unregister_material("my_tape")
    """
    global POLYMER_MATERIALS
    if not isinstance(name, str) or not name.strip():
        raise CableInputError("material name must be a non-empty string")
    if name in BUILT_IN_MATERIALS:
        raise CableInputError(f"{name!r} is a built-in material and cannot be replaced")
    if name in MATERIALS and not replace_existing:
        raise CableInputError(
            f"material {name!r} is already registered; pass replace_existing=True"
        )
    if not properties:
        raise CableInputError("a material needs at least one property")
    entry: Material = {}
    for key, item in properties.items():
        if not isinstance(key, str) or not key.strip():
            raise CableInputError("property names must be non-empty strings")
        try:
            value, citation = item
        except (TypeError, ValueError):
            raise CableInputError(
                f"property {key!r} must be a (value, citation) pair"
            ) from None
        value = float(value)
        if not math.isfinite(value):
            raise CableInputError(f"property {key!r} must be finite")
        if not isinstance(citation, str) or not citation.strip():
            raise CableInputError(f"property {key!r} needs a citation")
        entry[key] = (value, citation)
    MATERIALS[name] = entry
    if polymer:
        POLYMER_MATERIALS = POLYMER_MATERIALS | {name}


def unregister_material(name: str) -> None:
    """Remove a user material registered with :func:`register_material`."""
    global POLYMER_MATERIALS
    if name in BUILT_IN_MATERIALS:
        raise CableInputError(f"{name!r} is a built-in material")
    if name not in MATERIALS:
        raise CableLookupError(f"no user material {name!r}")
    del MATERIALS[name]
    POLYMER_MATERIALS = POLYMER_MATERIALS - {name}


def list_materials() -> list[str]:
    """Return the names of all materials in the library, sorted."""
    return sorted(MATERIALS)


def get_material(name: str) -> Mapping[str, Property]:
    """Return the full ``(value, citation)`` property mapping for a material.

    A material of the active overlay (:func:`material_scope`) comes first.

    Parameters
    ----------
    name : str
        Material key, e.g. ``"galvanised_steel_armour"``.

    Returns
    -------
    dict[str, tuple[float, str]]
        Mapping from unit-suffixed property name to ``(value, citation)``.

    Raises
    ------
    KeyError
        If ``name`` is not a known material.
    """
    overlay = _ACTIVE_OVERLAY.get()
    if overlay is not None and name in overlay:
        return overlay.materials[name]
    try:
        return MATERIALS[name]
    except KeyError:
        cable = (
            f" Materials of the cable: {', '.join(sorted(overlay.materials))}."
            if overlay is not None and overlay.materials
            else ""
        )
        raise CableLookupError(
            f"Unknown material {name!r}. "
            f"Known materials: {', '.join(list_materials())}.{cable}"
        ) from None


def get_property(material: str, prop: str) -> float:
    """Return the SI value of a single material property.

    Parameters
    ----------
    material : str
        Material key, e.g. ``"copper_stranded"``.
    prop : str
        Unit-suffixed property name, e.g. ``"E_eff_axial_Pa"``.

    Returns
    -------
    float
        The property value in strict SI units.

    Raises
    ------
    KeyError
        If the material or the property is unknown.
    """
    entry = get_material(material)
    try:
        return entry[prop][0]
    except KeyError:
        hint = (
            " No friction coefficient is published for this material; pass the "
            "coefficient explicitly (mu=..., mu_wire=...) or add it to the "
            "material with its source."
            if prop.startswith("mu_on_")
            else ""
        )
        raise CableLookupError(
            f"Material {material!r} has no property {prop!r}. "
            f"Available: {', '.join(sorted(entry))}.{hint}"
        ) from None


def get_citation(material: str, prop: str) -> str:
    """Return the literature citation for a single material property.

    Parameters
    ----------
    material : str
        Material key.
    prop : str
        Unit-suffixed property name.

    Returns
    -------
    str
        The ``[AuthorYear]`` citation string.

    Raises
    ------
    KeyError
        If the material or the property is unknown.
    """
    entry = get_material(material)
    try:
        return entry[prop][1]
    except KeyError:
        raise CableLookupError(
            f"Material {material!r} has no property {prop!r}. "
            f"Available: {', '.join(sorted(entry))}."
        ) from None


def axial_modulus_key(material: str) -> str:
    """The modulus key of a material modelled as a solid isotropic annulus.

    ``E_eff_axial_Pa`` for a material that gives it without ``E_Pa`` (a
    stranded conductor, whose axial key holds the copper wire modulus), and
    ``E_Pa`` otherwise.
    """
    entry = get_material(material)
    if "E_eff_axial_Pa" in entry and "E_Pa" not in entry:
        return "E_eff_axial_Pa"
    return "E_Pa"


def _no_law(material: str) -> CableLookupError:
    return CableLookupError(
        f"Material {material!r} has no sourced temperature-dependent "
        "Young's-modulus model."
    )


def _in_overlay(material: str) -> bool:
    overlay = _ACTIVE_OVERLAY.get()
    return overlay is not None and material in overlay


def _law_of(material: str) -> str | None:
    """The ``E(T)`` law of a material: its overlay law, or its own library law."""
    overlay = _ACTIVE_OVERLAY.get()
    if overlay is not None and material in overlay:
        return overlay.temperature_laws.get(material)
    return material if material in TEMPERATURE_MODULUS_MODELS else None


def has_temperature_modulus_model(material: str) -> bool:
    """Whether ``material`` has a sourced ``E(T)`` law (the active overlay first)."""
    return _law_of(material) is not None


def law_modulus_at_temperature(law: str, temperature_c: float) -> float:
    """Evaluate the registered ``E(T)`` law ``law`` in Pa.

    The polynomial is evaluated with Horner's method. Extrapolation beyond the
    range exercised in the source study is rejected because polymer modulus is
    grade, rate, ageing and temperature dependent.

    Raises
    ------
    KeyError
        If ``law`` is not a key of :data:`TEMPERATURE_MODULUS_MODELS`.
    ValueError
        If ``temperature_c`` is not finite or lies outside the sourced range.
    """
    try:
        coefficients, lower_c, upper_c, _ = TEMPERATURE_MODULUS_MODELS[law]
    except KeyError:
        raise _no_law(law) from None
    if not math.isfinite(temperature_c):
        raise CableInputError("temperature_c must be finite")
    if not lower_c <= temperature_c <= upper_c:
        raise CableInputError(
            f"temperature_c={temperature_c:g} is outside the sourced "
            f"{lower_c:g}--{upper_c:g} degC range for {law!r}"
        )

    modulus_mpa = 0.0
    for coefficient in coefficients:
        modulus_mpa = modulus_mpa * temperature_c + coefficient
    return modulus_mpa * 1.0e6


def youngs_modulus_at_temperature(material: str, temperature_c: float) -> float:
    """Return a sourced temperature-dependent Young's modulus in Pa.

    A library material with a model in :data:`TEMPERATURE_MODULUS_MODELS`
    takes its law directly (:func:`law_modulus_at_temperature`). A cable
    material of the active overlay that names a law takes the shape of the
    law scaled to its own ``E_Pa`` at :data:`MODULUS_REFERENCE_TEMPERATURE_C`,
    so an edited modulus moves the whole curve.

    Parameters
    ----------
    material : str
        Material key with a registered temperature model.
    temperature_c : float
        Temperature in degrees Celsius.

    Raises
    ------
    KeyError
        If no sourced temperature model is registered for ``material``.
    ValueError
        If ``temperature_c`` is not finite or lies outside the sourced range.
    """
    law = _law_of(material)
    if law is None:
        raise _no_law(material)
    if not _in_overlay(material):
        return law_modulus_at_temperature(law, temperature_c)
    return (
        get_property(material, "E_Pa")
        * law_modulus_at_temperature(law, temperature_c)
        / law_modulus_at_temperature(law, MODULUS_REFERENCE_TEMPERATURE_C)
    )


def get_temperature_law_citation(law: str) -> str:
    """Return the source and applicability statement of the ``E(T)`` law ``law``."""
    try:
        return TEMPERATURE_MODULUS_MODELS[law][3]
    except KeyError:
        raise _no_law(law) from None


def get_temperature_modulus_citation(material: str) -> str:
    """Return the source and applicability statement for a material's ``E(T)``."""
    law = _law_of(material)
    if law is None:
        raise _no_law(material)
    citation = get_temperature_law_citation(law)
    if _in_overlay(material):
        return (
            f"{law} law scaled to the cable material's Young's modulus "
            f"({get_citation(material, 'E_Pa')}): {citation}"
        )
    return citation
