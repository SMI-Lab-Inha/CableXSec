# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""CableXSec exception hierarchy.

A single, importable hierarchy for CableXSec's explicit input, lookup and solver
failures. Existing handlers for builtin ValueError, KeyError and RuntimeError
also catch the corresponding subclasses. Native filesystem errors, Pydantic
schema validation and unexpected programming/dependency errors retain their
original types; they are not indiscriminately converted into engineering errors.

==========================  ==================================================
Exception                   Raised when
==========================  ==================================================
:class:`CableXSecError`      base of every CableXSec error (catch-all)
:class:`CableSchemaError`    a cable cannot be loaded/adapted — missing or
                             physically inconsistent geometry/material
:class:`BlockedSNCurveError` a named S-N curve exists in the registry but its
                             source standard is not yet available (deferred,
                             not a typo)
:class:`PhysicalRangeError`  an input is outside the physically admissible
                             domain (e.g. negative diameter, T below 0 K) —
                             a hard error, distinct from the *soft*
                             extrapolation warning of
                             :mod:`cablexsec.provenance`
==========================  ==================================================

``BlockedSNCurveError`` and ``PhysicalRangeError`` also subclass the builtin
that best matches their semantics (``KeyError`` / ``ValueError``) so existing
``except KeyError`` / ``except ValueError`` callers keep working.

Examples
--------
>>> from cablexsec.exceptions import CableXSecError, PhysicalRangeError
>>> issubclass(PhysicalRangeError, CableXSecError)
True
>>> issubclass(PhysicalRangeError, ValueError)
True
"""

from __future__ import annotations

__all__ = [
    "BlockedSNCurveError",
    "CableSchemaError",
    "CableXSecError",
    "CableInputError",
    "CableLookupError",
    "CableSolverError",
    "PhysicalRangeError",
]


class CableXSecError(Exception):
    """Base for explicit CableXSec input, lookup and solver failures."""


class CableInputError(CableXSecError, ValueError):
    """An analysis input or derived numerical value violates its contract."""


class CableLookupError(CableXSecError, KeyError):
    """A requested layer, material, property or life law is unavailable."""


class CableSolverError(CableXSecError, RuntimeError):
    """A calculation fails its convergence or equilibrium checks."""


class CableSchemaError(CableXSecError):
    """A cable definition is missing required data or is inconsistent.

    Raised by the public API when a cable cannot be adapted to an analysis —
    e.g. an armour layer with no lay angle, or a cable that is not fully
    specified (withheld geometry still ``null``). The message names the
    offending layer/field so the caller can fix the YAML or supply the
    geometry override.
    """


class BlockedSNCurveError(CableXSecError, KeyError):
    """A registered S-N curve's source standard is not yet available.

    Distinguishes a *deliberately deferred* curve (the parameters cannot be
    cited until the standard is obtained) from a misspelled or
    unknown curve name. Catchable as either ``CableXSecError`` or ``KeyError``.
    """


class PhysicalRangeError(CableXSecError, ValueError):
    """An input lies outside the physically admissible domain.

    A hard validation failure (e.g. negative geometry, non-positive absolute
    temperature) — the computation cannot proceed and would otherwise produce a
    meaningless number. This is distinct from the *soft* validated-range
    extrapolation warning (:class:`cablexsec.provenance.ExtrapolationWarning`),
    which fires when an input is physically valid but outside the range against
    which a model was validated. Catchable as ``CableXSecError`` or
    ``ValueError``.
    """
