# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""The worker processes of a sea-state batch each write their own numba cache.

numba numbers a new on-disk cache entry from the index it last read, so two
processes compiling one kernel for different signatures at the same time can
leave the index pointing at the other's machine code.
"""

from __future__ import annotations

import os
import tempfile
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


@pytest.fixture(scope="module")
def cable():
    return cx.load_cable(cx.iea_task49_66kv_yaml_path(), geometry_override=None)


@pytest.fixture(scope="module")
def stiffness(cable):
    return cx.compute_stiffness(cable)


def test_each_worker_starts_with_its_own_numba_cache(tmp_path, monkeypatch):
    caller = str(tmp_path / "caller")
    monkeypatch.setenv("NUMBA_CACHE_DIR", caller)
    pools = batch._worker_pools(3, tmp_path / "workers")
    try:
        seen = [pool.submit(os.getenv, "NUMBA_CACHE_DIR").result() for pool in pools]
    finally:
        for pool in pools:
            pool.shutdown()
    assert seen == [str(tmp_path / "workers" / f"worker-{i}") for i in range(3)]
    assert os.environ["NUMBA_CACHE_DIR"] == caller


def test_cold_cache_workers_match_the_in_process_batch(
    tmp_path, monkeypatch, cable, stiffness
):
    # Three runs on two workers, each starting from an empty cache directory of
    # its own below a fresh temporary directory, removed after the batch.
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    written: dict[str, list[str]] = {}
    evaluate = batch._evaluate_in_pools

    def recording(tasks, pools):
        outcomes = evaluate(tasks, pools)
        for root in scratch.iterdir():
            for worker in sorted(root.iterdir()):
                written[worker.name] = sorted(p.name for p in worker.rglob("*.nbi"))
        return outcomes

    monkeypatch.setattr(batch, "_evaluate_in_pools", recording)
    runs = _runs(
        tmp_path,
        stiffness,
        amplitudes=(0.01, 0.02, 0.015),
        probabilities=(0.5, 0.3, 0.2),
    )
    sections = [SectionRequest("a", 2.0)]
    two = run_sea_state_batch(
        cable, runs, sections, options=SeaStateBatchOptions(workers=2)
    )
    if kernels_compiled():
        # Each worker compiled its kernels into its own directory.
        assert sorted(written) == ["worker-0", "worker-1"]
        assert all(written.values())
    assert list(scratch.iterdir()) == []
    one = run_sea_state_batch(
        cable, runs, sections, options=SeaStateBatchOptions(workers=1)
    )
    a, b = one.sections["a"].long_term, two.sections["a"].long_term
    # Equal to round-off: spawned workers run BLAS on one thread, the calling
    # process on its own setting (as in test_worker_count_does_not_change_results).
    for layer in a.annual_damage_per_fibre:
        np.testing.assert_allclose(
            a.annual_damage_per_fibre[layer],
            b.annual_damage_per_fibre[layer],
            rtol=1e-10,
            atol=0.0,
        )
    assert one.manifest["inputs_sha256"] == two.manifest["inputs_sha256"]


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
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    with pytest.raises(RuntimeError, match="failed to start"):
        batch._worker_pools(3, Path(tmp_path))
    assert len(started) == 2
    # Both pools are shut down, and the caller's environment is restored.
    with pytest.raises(RuntimeError, match="after shutdown"):
        started[0].submit(os.getpid)
    assert "NUMBA_CACHE_DIR" not in os.environ
