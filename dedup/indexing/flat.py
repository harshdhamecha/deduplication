"""Tier 1 — IndexFlatIP: exact, brute-force inner-product search.

WHEN: small datasets (< ~1M vectors) that fit in RAM. This is the default the
selector reaches for whenever it can, because it has ZERO approximation error —
perfect recall, no training, no parameters to tune. For a portfolio dataset
(hundreds to a few hundred-thousand images) there is simply no reason to give up
recall for a speed we don't need.

COST: memory and time are linear in N*dim. At 1M x 768 fp32 that's ~3GB and a
full scan per query — fine on a workstation, not fine at 50M, which is exactly
where the selector switches tiers.
"""

from __future__ import annotations

import faiss
import numpy as np

from dedup.indexing.base import VectorIndex


class FlatIndex(VectorIndex):
    def __init__(self, dim: int):
        super().__init__(dim)
        self._index = faiss.IndexFlatIP(dim)  # inner product == cosine on normed vecs

    def add(self, vectors: np.ndarray) -> None:
        self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        return self._index.search(np.ascontiguousarray(queries, dtype=np.float32), k)

    def save(self, path: str) -> None:
        faiss.write_index(self._index, path)

    @property
    def ntotal(self) -> int:
        return self._index.ntotal
