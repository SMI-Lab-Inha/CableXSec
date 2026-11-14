# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Explicit construction and mass inputs for the global/local application."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np

import cablexsec as cx
from cablexsec.cable_adapters import armour_contact_pressure, cable_to_fatigue_layers
from cablexsec.cables.schema import (
    CylindricalLayer,
    CylindricalOffsetLayer,
    CylindricalShellLayer,
    HelicalLayer,
    HelicalStrandedLayer,
)
from cablexsec.materials import library
from cablexsec.mechanics.section_geometry import material_volume_per_length
from cablexsec.mechanics.stiffness import axial_torsional_matrix, ea_fixed, ei_min_slip
from cablexsec.provenance import MissingClampingStateWarning


def _area_and_radius(layer: object) -> tuple[float, float]:
    """Return material area per cable length and outer radial extent."""
    if isinstance(layer, HelicalLayer):
        assert layer.n_wires and layer.d_wire_m and layer.r_helix_m
        assert layer.lay_angle_rad is not None
        area = layer.n_wires * math.pi * layer.d_wire_m**2 / 4.0
        area /= math.cos(layer.lay_angle_rad)
        return area, layer.r_helix_m + layer.d_wire_m / 2.0
    if isinstance(layer, HelicalStrandedLayer):
        assert layer.n_conductors and layer.n_strands and layer.d_strand_m
        assert layer.r_cable_m and layer.r_strand_m
        assert layer.lay_angle_cable_rad is not None
        assert layer.lay_angle_strand_rad is not None
        area = (
            layer.n_conductors * layer.n_strands * math.pi * layer.d_strand_m**2 / 4.0
        )
        area /= math.cos(layer.lay_angle_cable_rad)
        area /= math.cos(layer.lay_angle_strand_rad)
        radius = layer.r_cable_m + layer.r_strand_m + layer.d_strand_m / 2.0
        return area, radius
    if isinstance(layer, CylindricalOffsetLayer):
        assert layer.r_inner_m is not None and layer.r_outer_m is not None
        assert layer.r_offset_m is not None
        area = layer.n_cores * math.pi * (layer.r_outer_m**2 - layer.r_inner_m**2)
        return area, layer.r_offset_m + layer.r_outer_m
    if isinstance(layer, CylindricalLayer):
        assert layer.r_inner_m is not None and layer.r_outer_m is not None
        area = math.pi * (layer.r_outer_m**2 - layer.r_inner_m**2)
        return area, layer.r_outer_m
    if isinstance(layer, CylindricalShellLayer):
        assert layer.r_mid_m is not None and layer.t_m is not None
        area = 2.0 * math.pi * layer.r_mid_m * layer.t_m
        return area, layer.r_mid_m + layer.t_m / 2.0
    raise TypeError(f"unsupported cable layer {type(layer).__name__}")


def _global_properties(
    cable: cx.CableData, *, density_overrides=None, additional_mass_kg_m=0.0
) -> dict[str, float]:
    """Derive the homogeneous-line inputs from one public cable definition."""
    densities = {} if density_overrides is None else dict(density_overrides)
    if set(densities) - {layer.material for layer in cable.layers}:
        raise ValueError("density override refers to an absent cable material")
    if any(
        isinstance(value, bool) or not math.isfinite(value) or value <= 0
        for value in densities.values()
    ):
        raise ValueError("material densities must be finite and positive")
    if not math.isfinite(additional_mass_kg_m) or additional_mass_kg_m < 0:
        raise ValueError("additional distributed mass must be finite and nonnegative")
    dry_mass_kg_m = additional_mass_kg_m
    outer_radius_m = 0.0
    for layer in cable.layers:
        _, radius_m = _area_and_radius(layer)
        area_m2 = material_volume_per_length(layer, cable)
        density = (
            densities[layer.material]
            if layer.material in densities
            else library.get_property(layer.material, "rho_kg_m3")
        )
        dry_mass_kg_m += area_m2 * density
        outer_radius_m = max(outer_radius_m, radius_m)
    return {
        "outer_diameter_m": 2.0 * outer_radius_m,
        "dry_mass_kg_m": dry_mass_kg_m,
        "EA_N": ea_fixed(cable),
        "EI_full_slip_Nm2": ei_min_slip(cable),
        "GJ_Nm2": axial_torsional_matrix(cable).K_tt,
    }


# Version of the mechanical record below. A record of another version is not
# compared; the run then falls back to its file hash.
MECHANICAL_DIGEST_VERSION = 1
# Bounded-chain armour contact pressures P1+P2 are recorded at these tensions.
DIGEST_TENSIONS_N = (0.0, 5.0e4, 1.0e5)
_SIGNIFICANT = 12


def _plain(value):
    """JSON form of the mechanics, floats to 12 significant digits."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _plain(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return repr(number)
        return float(f"{number:.{_SIGNIFICANT}g}")
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"no mechanical record for {type(value).__name__}")


def section_mechanics(cable):
    """Record the mechanics a section file defines for the local recovery.

    The record holds the construction stiffness passed to the global model
    (outer diameter, EA, full-slip EI, GJ), the bounded-chain armour contact
    pressures P1+P2 at :data:`DIGEST_TENSIONS_N`, and the frozen fatigue
    layers (wire geometry, contact, S-N curves). Comments, sources and other
    text that change none of these do not change the record. Mass is not
    included: densities may come from a separate mass model, and
    ``require_matching_section_properties`` compares the global mass.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingClampingStateWarning)
        pressures = {
            f"{tension:.0f}": armour_contact_pressure(cable, tension)
            for tension in DIGEST_TENSIONS_N
        }
        layers = cable_to_fatigue_layers(cable, contact_tension_N=0.0)
    return {
        "version": MECHANICAL_DIGEST_VERSION,
        "properties_SI": _plain(
            {
                "outer_diameter_m": 2.0
                * max(_area_and_radius(layer)[1] for layer in cable.layers),
                "EA_N": ea_fixed(cable),
                "EI_full_slip_Nm2": ei_min_slip(cable),
                "GJ_Nm2": axial_torsional_matrix(cable).K_tt,
            }
        ),
        "contact_pressure_Pa": _plain(pressures),
        "fatigue_layers": _plain(layers),
    }


def mechanical_digest(cable):
    """SHA-256 of :func:`section_mechanics`, for run manifests."""
    text = json.dumps(section_mechanics(cable), sort_keys=True, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def section_record(section_file):
    """Section identity for a new run manifest: file hash and mechanical digest."""
    path = Path(section_file).resolve(strict=True)
    return {
        "local_section_file": str(path),
        "local_section_sha256": _file_sha256(path),
        "local_section_mechanical_sha256": mechanical_digest(
            cx.load_cable(path, geometry_override=None)
        ),
        "local_section_mechanical_version": MECHANICAL_DIGEST_VERSION,
    }


def _file_sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _git(directory, *arguments):
    return subprocess.run(
        ["git", "-C", str(directory), *arguments],
        capture_output=True,
        check=True,
    ).stdout


def recorded_section_blob(path, sha256):
    """Recover a section file's earlier content by its recorded SHA-256.

    Searches every commit of the repository holding ``path`` that touched
    it. Returns ``(content, blob id, commit)`` or ``None``.
    """
    path = Path(path).resolve()
    try:
        top = Path(_git(path.parent, "rev-parse", "--show-toplevel").decode().strip())
        relative = path.relative_to(top.resolve()).as_posix()
        commits = _git(top, "log", "--all", "--format=%H", "--", relative).split()
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None
    seen = set()
    for commit in (c.decode() for c in commits):
        try:
            blob = _git(top, "rev-parse", f"{commit}:{relative}").decode().strip()
        except subprocess.CalledProcessError:
            continue  # the file is absent in this commit (deleted or renamed)
        if blob in seen:
            continue
        seen.add(blob)
        content = _git(top, "cat-file", "blob", blob)
        variants = (content, content.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        if any(hashlib.sha256(v).hexdigest() == sha256 for v in variants):
            return content, blob, commit
    return None


def case_section(inputs, *, accept_legacy_section=False):
    """Load a run's section and state how its identity was established.

    Returns ``(cable, identity)``. The section is accepted when

    1. the file hash equals the recorded hash (``basis: file``);
    2. the file changed but its mechanical digest equals the one recorded in
       the run manifest (``basis: mechanical_digest``), for example after a
       declaration that changes no property, pressure or fatigue input; or
    3. the run records only a file hash, ``accept_legacy_section`` is set,
       and the recorded file, recovered from git history by that hash, has
       the same mechanics as the current file (``basis: git_recomputed``).

    Otherwise a ``ValueError`` explains why the run is refused.
    """
    path = Path(inputs.get("local_section_file", cx.export_three_core_yaml_path()))
    if not path.is_absolute():
        raise ValueError("the saved local section path must be absolute")
    recorded = inputs["local_section_sha256"]
    actual = _file_sha256(path)
    cable = cx.load_cable(path, geometry_override=None)
    identity = {
        "basis": "file",
        "recorded_file_sha256": recorded,
        "current_file_sha256": actual,
        "mechanical_sha256": inputs.get("local_section_mechanical_sha256"),
        "note": "section file unchanged since the global calculation",
    }
    if actual == recorded:
        return cable, identity
    changed = (
        f"local section changed after the global calculation ({path.name}: "
        f"recorded sha256 {recorded[:12]}, current {actual[:12]})"
    )
    recorded_mechanics = inputs.get("local_section_mechanical_sha256")
    version = inputs.get("local_section_mechanical_version")
    if recorded_mechanics is not None and version == MECHANICAL_DIGEST_VERSION:
        current = mechanical_digest(cable)
        if current != recorded_mechanics:
            raise ValueError(
                changed + "; its construction properties or contact pressures "
                "also differ from the recorded mechanical digest"
            )
        identity |= {
            "basis": "mechanical_digest",
            "mechanical_sha256": current,
            "note": "section file changed after the global calculation; its "
            "construction properties and contact pressures equal the recorded "
            "mechanical digest",
        }
        return cable, identity
    if not accept_legacy_section:
        raise ValueError(
            changed + "; the run records no mechanical digest. Accept it "
            "explicitly (accept_legacy_section, --accept-legacy-section) to "
            "compare the mechanics of the recorded file, recovered from git "
            "history, with the current file"
        )
    found = recorded_section_blob(path, recorded)
    if found is None:
        raise ValueError(
            changed + "; the recorded file is not in the git history of this "
            "path, so its mechanics cannot be recomputed. The run is refused"
        )
    content, blob, commit = found
    with tempfile.TemporaryDirectory() as folder:
        # Same file name as the recorded section file.
        old_path = Path(folder) / path.name
        old_path.write_bytes(content)
        old = mechanical_digest(cx.load_cable(old_path, geometry_override=None))
    current = mechanical_digest(cable)
    if old != current:
        raise ValueError(
            changed + f"; the recorded file (git blob {blob[:12]}) and the "
            "current file differ in construction properties or contact pressures"
        )
    identity |= {
        "basis": "git_recomputed",
        "mechanical_sha256": current,
        "recorded_git_blob": blob,
        "recorded_git_commit": commit,
        "note": "section file changed after the global calculation; the run "
        "records only a file hash and was accepted explicitly: the recorded "
        "file, recovered from git history, and the current file have equal "
        "construction properties and contact pressures",
    }
    return cable, identity


def load_case_section(inputs, *, accept_legacy_section=False):
    """Use the saved section, with the export path only for legacy records.

    See :func:`case_section` for when a changed section file is accepted.
    """
    return case_section(inputs, accept_legacy_section=accept_legacy_section)[0]


def construction_properties(cable, mass_model=None):
    """Retain the section mechanics and apply separately declared mass inputs.

    The optional filler contributes mass only. Its occupied volume is the
    enclosing sheath interior less the helical core material volume. This
    is a bulk-volume assumption, not a reconstructed contact cross-section.
    """
    if mass_model is None:
        return _global_properties(cable)
    allowed = {"description", "densities", "unmodelled_filler"}
    if set(mass_model) - allowed:
        raise ValueError("unrecognised global mass-model field")
    if (
        not isinstance(mass_model.get("description"), str)
        or not mass_model["description"].strip()
    ):
        raise ValueError("the mass-model assumptions must be described")
    entries = mass_model.get("densities", {})
    densities = {}
    for material, entry in entries.items():
        if set(entry) != {"rho_kg_m3", "source"} or not str(entry["source"]).strip():
            raise ValueError("each density needs its value and source")
        densities[material] = entry["rho_kg_m3"]
    extra = 0.0
    if mass_model.get("unmodelled_filler") is not None:
        filler = mass_model["unmodelled_filler"]
        if set(filler) != {"enclosing_layer", "rho_kg_m3", "filled_fraction", "source"}:
            raise ValueError("unrecognised filler mass-model field")
        if not str(filler["source"]).strip():
            raise ValueError("filler mass needs a documented basis")
        fraction, density = filler["filled_fraction"], filler["rho_kg_m3"]
        if (
            not math.isfinite(fraction)
            or not 0 <= fraction <= 1
            or not math.isfinite(density)
            or density <= 0
        ):
            raise ValueError("invalid filler density or occupied fraction")
        enclosing = cable.get_layer(filler["enclosing_layer"])
        if not isinstance(enclosing, CylindricalLayer) or enclosing.r_inner_m is None:
            raise ValueError("filler must be inside a specified cylindrical sheath")
        interior = math.pi * enclosing.r_inner_m**2
        components = [
            layer for layer in cable.layers if isinstance(layer, CylindricalOffsetLayer)
        ]
        if (
            not components
            or len({(layer.n_cores, layer.r_offset_m) for layer in components}) != 1
            or any(layer.r_offset_m >= enclosing.r_inner_m for layer in components)
        ):
            raise ValueError("core family is not within the declared filler enclosure")
        # This uses bulk-volume conservation, not overlap areas of the initial
        # contact geometry. It does not assert that those surfaces are solved.
        if any(
            isinstance(layer, CylindricalLayer)
            and layer is not enclosing
            and layer.r_outer_m <= enclosing.r_inner_m
            for layer in cable.layers
        ):
            raise ValueError(
                "interior cylindrical material may double-count the added filler"
            )
        occupied = sum(material_volume_per_length(layer, cable) for layer in components)
        if occupied > interior:
            raise ValueError("core material exceeds the enclosing interior volume")
        extra = fraction * density * (interior - occupied)
    return _global_properties(
        cable, density_overrides=densities, additional_mass_kg_m=extra
    )
