"""Tests for the index tiers and the scale-adaptive selector.

Covers the two things the spec most wants correct: (1) the selector picks the
right tier from count + RAM and honours overrides, and (2) IVFPQ's exact re-rank
recovers the true nearest neighbour that raw PQ would only approximate.
"""

import dataclasses

import numpy as np
import pytest

from dedup.config import Stage3Config
from dedup.indexing.base import normalize
from dedup.indexing.flat import FlatIndex
from dedup.indexing.ivfpq import IvfPqIndex
from dedup.indexing.selector import select_index


def _cfg(**kw) -> Stage3Config:
    return dataclasses.replace(Stage3Config(), **kw)


# ----------------------------- selector logic ----------------------------- #
def test_selector_picks_flat_for_small_n():
    idx = select_index(50_000, 768, _cfg(index_type="auto"))
    assert isinstance(idx, FlatIndex)


def test_selector_picks_ivfflat_for_medium_n():
    idx = select_index(2_000_000, 768, _cfg(index_type="auto", ram_budget_gb=64))
    assert type(idx).__name__ == "IvfFlatIndex"


def test_selector_picks_ivfpq_for_large_n():
    idx = select_index(20_000_000, 768, _cfg(index_type="auto", ram_budget_gb=64))
    assert type(idx).__name__ == "IvfPqIndex"


def test_ram_budget_forces_pq_when_full_vectors_dont_fit():
    # 500k x 768 fp32 ~= 1.4GB. Both Flat and IVFFlat keep FULL vectors, so with
    # a 0.1GB budget neither fits — only the compressed IVFPQ does. The selector
    # must skip past IVFFlat (count alone would have allowed it) to IVFPQ.
    idx = select_index(500_000, 768, _cfg(index_type="auto", ram_budget_gb=0.1))
    assert type(idx).__name__ == "IvfPqIndex"


def test_explicit_override_beats_auto():
    # Huge N would auto-select IVFPQ, but an explicit request wins.
    idx = select_index(20_000_000, 768, _cfg(index_type="flat"))
    assert isinstance(idx, FlatIndex)


# ----------------------------- flat exactness ----------------------------- #
def test_flat_finds_exact_nearest_neighbour():
    rng = np.random.default_rng(0)
    vecs = normalize(rng.standard_normal((500, 64)).astype(np.float32))
    idx = FlatIndex(64)
    idx.add(vecs)
    sims, ids = idx.search(vecs, k=1)
    # Each vector's nearest neighbour is itself at cosine ~1.0 (exact search).
    assert np.all(ids[:, 0] == np.arange(500))
    assert np.allclose(sims[:, 0], 1.0, atol=1e-4)


# --------------------------- ivfpq re-ranking ----------------------------- #
def test_ivfpq_rerank_recovers_self_as_top1():
    rng = np.random.default_rng(1)
    n, dim = 3000, 64
    vecs = normalize(rng.standard_normal((n, dim)).astype(np.float32))

    idx = IvfPqIndex(dim, nlist=32, nprobe=8, m=8, nbits=8, rerank=True, rerank_k=50)
    idx.train(vecs)
    idx.add(vecs)
    idx.attach_originals(vecs)

    sims, ids = idx.search(vecs, k=1)
    # With exact re-ranking against the originals, a vector's own copy scores a
    # true cosine of 1.0 and must win — PQ alone would often mis-rank it.
    self_hits = np.mean(ids[:, 0] == np.arange(n))
    assert self_hits > 0.98, f"only {self_hits:.3f} self-hits after re-rank"


def test_ivfpq_rejects_indivisible_m():
    with pytest.raises(ValueError):
        IvfPqIndex(dim=768, nlist=16, nprobe=4, m=7, nbits=8)  # 768 % 7 != 0
