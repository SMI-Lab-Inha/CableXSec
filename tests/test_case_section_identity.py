# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Run-manifest section identity: file hash, mechanical digest, git recovery.

A run built from a section file must not be post-processed with different
mechanics, but a declaration that changes no construction property, contact
pressure or fatigue input (for example the explicit zero
``manufacturing_clamping`` of the 66 kV template, 94d5295) must not make the
run unusable.
"""

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest
from fowt_construction import (
    case_section,
    load_case_section,
    mechanical_digest,
    recorded_section_blob,
    section_mechanics,
    section_record,
)

import cablexsec as cx

pytestmark = pytest.mark.analytical_sanity

PRESSURE = "  external_pressure_Pa: 300000.0\n"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def section(tmp_path):
    path = tmp_path / "menard2023.yaml"
    shutil.copyfile(cx.menard2023_yaml_path(), path)
    assert PRESSURE in path.read_text(encoding="utf-8")
    return path


def _append_comment(path):
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("# a later comment: no mechanical change\n")


def _edit(path, old, new):
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1
    path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")


def test_new_manifest_records_file_hash_and_mechanical_digest(section):
    record = section_record(section)
    assert record["local_section_file"] == str(section.resolve())
    assert record["local_section_sha256"] == _sha(section)
    cable = cx.load_cable(section, geometry_override=None)
    assert record["local_section_mechanical_sha256"] == mechanical_digest(cable)
    mechanics = section_mechanics(cable)
    assert set(mechanics) == {
        "version",
        "properties_SI",
        "contact_pressure_Pa",
        "fatigue_layers",
    }
    # Ménard: 0.3 MPa external pressure gives 0.6 MPa P1+P2 at zero tension.
    zero = mechanics["contact_pressure_Pa"]["0"]
    assert zero == pytest.approx({"inner_armour": 6.0e5, "outer_armour": 6.0e5})
    loaded, identity = case_section(record)
    assert identity["basis"] == "file"
    assert "unchanged" in identity["note"]
    assert loaded.cable.name == cable.cable.name


def test_comment_change_is_accepted_by_mechanical_digest(section):
    record = section_record(section)
    _append_comment(section)
    assert _sha(section) != record["local_section_sha256"]
    cable, identity = case_section(record)
    assert identity["basis"] == "mechanical_digest"
    assert identity["current_file_sha256"] == _sha(section)
    assert identity["mechanical_sha256"] == record["local_section_mechanical_sha256"]
    assert "changed" in identity["note"] and "equal" in identity["note"]
    assert load_case_section(record).cable.name == cable.cable.name


@pytest.mark.parametrize(
    "old,new",
    [
        (PRESSURE, "  external_pressure_Pa: 310000.0\n"),  # contact pressure only
        ("n_wires: 55 ", "n_wires: 54 "),  # construction
    ],
)
def test_mechanical_change_is_refused_with_a_digest(section, old, new):
    record = section_record(section)
    _edit(section, old, new)
    with pytest.raises(ValueError, match="changed.*also differ"):
        case_section(record)


def test_digest_of_another_version_is_not_compared(section):
    record = section_record(section) | {"local_section_mechanical_version": 0}
    _append_comment(section)
    with pytest.raises(ValueError, match="no mechanical digest"):
        case_section(record)


def test_legacy_run_needs_the_explicit_flag_and_its_git_file(section):
    legacy = {
        "local_section_file": str(section),
        "local_section_sha256": _sha(section),
    }
    _append_comment(section)
    with pytest.raises(ValueError, match="changed.*no mechanical digest"):
        case_section(legacy)
    # tmp_path is outside every repository: the old file cannot be recovered.
    with pytest.raises(ValueError, match="not in the git history"):
        case_section(legacy, accept_legacy_section=True)


def _git(folder, *args):
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-C",
            str(folder),
            *args,
        ],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repository(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    folder = tmp_path / "repo"
    folder.mkdir()
    _git(folder, "init", "-q")
    path = folder / "menard2023.yaml"
    shutil.copyfile(cx.menard2023_yaml_path(), path)
    _git(folder, "add", "menard2023.yaml")
    _git(folder, "commit", "-q", "-m", "section")
    return folder, path


def test_legacy_run_is_accepted_when_git_file_has_equal_mechanics(repository):
    folder, path = repository
    legacy = {"local_section_file": str(path), "local_section_sha256": _sha(path)}
    _append_comment(path)
    _git(folder, "commit", "-q", "-am", "comment")
    found = recorded_section_blob(path, legacy["local_section_sha256"])
    assert found is not None
    content, blob, commit = found
    assert hashlib.sha256(content).hexdigest() == legacy["local_section_sha256"]
    cable, identity = case_section(legacy, accept_legacy_section=True)
    assert identity["basis"] == "git_recomputed"
    assert identity["recorded_git_blob"] == blob
    assert identity["recorded_git_commit"] == commit
    assert "accepted explicitly" in identity["note"]
    assert identity["mechanical_sha256"] == mechanical_digest(cable)


def test_legacy_run_is_refused_when_git_file_mechanics_differ(repository):
    folder, path = repository
    legacy = {"local_section_file": str(path), "local_section_sha256": _sha(path)}
    _edit(path, PRESSURE, "  external_pressure_Pa: 310000.0\n")
    with pytest.raises(ValueError, match="differ in construction properties"):
        case_section(legacy, accept_legacy_section=True)
