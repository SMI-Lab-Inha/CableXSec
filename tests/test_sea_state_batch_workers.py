# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""The worker processes of a sea-state batch each write their own numba cache.

numba numbers a new on-disk cache entry from the index it last read, so two
processes compiling one kernel for different signatures at the same time can
leave the index pointing at the other's machine code. Each worker therefore
writes a slot directory whose lock it holds itself for as long as it runs, and
the slot is kept for later batches.
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import pytest
from test_sea_state_batch import _runs

import cablexsec as cx
import cablexsec.analysis.sea_state_batch as batch
from cablexsec._numba import kernels_compiled
from cablexsec.analysis.sea_state_batch import (
    SeaStateBatchOptions,
    SectionRequest,
    run_sea_state_batch,
)

_ONE_THREAD = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


@pytest.fixture(scope="module")
def cable():
    return cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)


@pytest.fixture(scope="module")
def stiffness(cable):
    return cx.compute_stiffness(cable)


@contextmanager
def _held(path: Path) -> Iterator[None]:
    """Hold the slot lock ``path`` in this process, as another batch would."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as handle:
        assert batch._lock(handle)
        try:
            yield
        finally:
            batch._unlock(handle)


def _caches(pools) -> list[Path]:
    """The numba cache directory each worker runs with."""
    return [Path(pool.submit(os.getenv, "NUMBA_CACHE_DIR").result()) for pool in pools]


def _indexes(folder: Path) -> dict[str, int]:
    """Modification times of the numba index files below ``folder``."""
    return {
        str(p.relative_to(folder)): p.stat().st_mtime_ns for p in folder.rglob("*.nbi")
    }


def test_each_worker_starts_with_its_own_numba_cache(tmp_path, monkeypatch):
    caller = str(tmp_path / "caller")
    monkeypatch.setenv("NUMBA_CACHE_DIR", caller)
    caches = [tmp_path / f"cache-{i}" for i in range(3)]
    pools = [batch._start_worker(cache) for cache in caches]
    try:
        assert _caches(pools) == caches
    finally:
        for pool in pools:
            pool.shutdown()
    assert os.environ["NUMBA_CACHE_DIR"] == caller


def test_the_worker_holds_its_slot_for_as_long_as_it_runs(tmp_path):
    lock = tmp_path / "slot-0.lock"
    pool = batch._start_worker(tmp_path / "slot-0")
    try:
        assert pool.submit(batch._claim_slot, str(lock)).result()
        # The slot is taken by the worker process itself, so it stays taken
        # whatever happens to the process that started it.
        assert not batch._slot_is_free(lock)
        assert not pool.submit(batch._claim_slot, str(lock)).result()
    finally:
        pool.shutdown()
    assert batch._slot_is_free(lock)


def test_slots_are_kept_and_a_second_batch_reuses_them(
    tmp_path, monkeypatch, cable, stiffness
):
    slots = tmp_path / "slots"
    monkeypatch.setenv("CABLEXSEC_BATCH_CACHE_DIR", str(slots))
    runs = _runs(
        tmp_path,
        stiffness,
        amplitudes=(0.01, 0.02, 0.015),
        probabilities=(0.5, 0.3, 0.2),
    )
    sections = [SectionRequest("a", 2.0)]
    options = SeaStateBatchOptions(workers=2)
    # First batch: two workers, each starting from an empty slot of its own.
    first = run_sea_state_batch(cable, runs, sections, options=options)
    written = {name: _indexes(slots / name) for name in ("slot-0", "slot-1")}
    if kernels_compiled():
        assert all(written.values()), written
    assert not (slots / "slot-2").exists()
    # Second batch: the slots are reused, and no kernel is compiled again.
    second = run_sea_state_batch(cable, runs, sections, options=options)
    assert {name: _indexes(slots / name) for name in written} == written
    one = run_sea_state_batch(
        cable, runs, sections, options=SeaStateBatchOptions(workers=1)
    )
    a = one.sections["a"].long_term.annual_damage_per_fibre
    for result in (first, second):
        b = result.sections["a"].long_term.annual_damage_per_fibre
        for layer in a:
            if all(os.environ.get(name) == "1" for name in _ONE_THREAD):
                # The same kernels on one BLAS thread in every process.
                np.testing.assert_array_equal(b[layer], a[layer])
            else:
                # The workers run BLAS on one thread, this process on its own
                # setting; threaded BLAS reductions differ at round-off.
                np.testing.assert_allclose(b[layer], a[layer], rtol=1e-10, atol=0.0)
        assert result.manifest["inputs_sha256"] == one.manifest["inputs_sha256"]


def test_a_batch_beside_another_takes_the_next_free_slots(tmp_path, monkeypatch):
    slots = tmp_path / "slots"
    monkeypatch.setenv("CABLEXSEC_BATCH_CACHE_DIR", str(slots))
    with _held(slots / "slot-0.lock"):
        assert not batch._slot_is_free(slots / "slot-0.lock")
        with batch._workers(2) as pools:
            assert _caches(pools) == [slots / "slot-1", slots / "slot-2"]
            assert not batch._slot_is_free(slots / "slot-1.lock")
        assert batch._slot_is_free(slots / "slot-1.lock")
    with batch._workers(2) as pools:
        assert _caches(pools) == [slots / "slot-0", slots / "slot-1"]


def test_a_slot_won_by_another_process_meanwhile_is_skipped(tmp_path, monkeypatch):
    slots = tmp_path / "slots"
    monkeypatch.setenv("CABLEXSEC_BATCH_CACHE_DIR", str(slots))
    stack = ExitStack()
    is_free = batch._slot_is_free

    def taken_after_the_check(path):
        # Another batch takes slot 0 between the check and the worker's claim.
        free = is_free(path)
        if path.name == "slot-0.lock":
            stack.enter_context(_held(path))
        return free

    started = []
    start = batch._start_worker

    def recording(cache):
        started.append(cache)
        return start(cache)

    monkeypatch.setattr(batch, "_slot_is_free", taken_after_the_check)
    monkeypatch.setattr(batch, "_start_worker", recording)
    with stack, batch._workers(1) as pools:
        assert started == [slots / "slot-0", slots / "slot-1"]
        assert _caches(pools) == [slots / "slot-1"]


def test_without_a_free_slot_a_worker_gets_a_temporary_cache(tmp_path, monkeypatch):
    slots = tmp_path / "slots"
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setenv("CABLEXSEC_BATCH_CACHE_DIR", str(slots))
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    with ExitStack() as stack:
        for slot in range(batch._CACHE_SLOTS - 1):
            stack.enter_context(_held(slots / f"slot-{slot}.lock"))
        with batch._workers(3) as pools:
            caches = _caches(pools)
            assert caches[0] == slots / f"slot-{batch._CACHE_SLOTS - 1}"
            assert [c.parent.parent for c in caches[1:]] == [scratch, scratch]
            assert len(set(caches)) == 3
    assert list(scratch.iterdir()) == []


def test_an_unusable_cache_folder_falls_back_to_temporary_caches(tmp_path, monkeypatch):
    blocked = tmp_path / "file"
    blocked.write_text("not a folder", encoding="utf-8")
    monkeypatch.setenv("CABLEXSEC_BATCH_CACHE_DIR", str(blocked / "slots"))
    with batch._workers(2) as pools:
        caches = _caches(pools)
    assert len(set(caches)) == 2
    assert all(blocked not in c.parents for c in caches)


def test_cache_folder_follows_the_override_and_the_platform(tmp_path, monkeypatch):
    monkeypatch.setenv("CABLEXSEC_BATCH_CACHE_DIR", str(tmp_path / "mine"))
    assert batch._batch_cache_directory() == tmp_path / "mine"
    monkeypatch.delenv("CABLEXSEC_BATCH_CACHE_DIR")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    expected = {
        "win32": tmp_path / "local",
        "darwin": Path.home() / "Library" / "Caches",
    }.get(sys.platform, tmp_path / "xdg")
    assert batch._batch_cache_directory() == expected / "cablexsec" / "numba-batch"


def test_a_worker_that_fails_to_start_leaves_no_pool(tmp_path, monkeypatch):
    started = []
    real = batch.ProcessPoolExecutor

    class Failing(real):
        def submit(self, *args, **kwargs):
            if self in started[1:2]:
                raise RuntimeError("worker failed to start")
            return super().submit(*args, **kwargs)

    def pool(*args, **kwargs):
        started.append(Failing(*args, **kwargs))
        return started[-1]

    monkeypatch.setattr(batch, "ProcessPoolExecutor", pool)
    monkeypatch.setenv("CABLEXSEC_BATCH_CACHE_DIR", str(tmp_path / "slots"))
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    with pytest.raises(RuntimeError, match="failed to start"), batch._workers(3):
        pass
    assert len(started) == 2
    # The started pool is shut down (the second one never ran), the first
    # slot is free again, and the caller's environment is restored.
    with pytest.raises(RuntimeError, match="after shutdown"):
        started[0].submit(os.getpid)
    assert batch._slot_is_free(tmp_path / "slots" / "slot-0.lock")
    assert "NUMBA_CACHE_DIR" not in os.environ
