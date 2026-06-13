"""Tier 2 — IndexIVFFlat: partitioned but uncompressed.

WHEN: medium datasets (~1M-10M) where a full Flat scan per query gets slow but
the vectors still fit in RAM uncompressed. IVF clusters the space into ``nlist``
Voronoi cells (via a coarse quantizer) and, at query time, only scans the
``nprobe`` nearest cells instead of everything — a big speedup.

WHAT IT TRADES: a small, *tunable* recall hit. A true neighbour sitting just
across a cell boundary can be missed if its cell isn't among the nprobe probed.
Raising nprobe recovers recall at the cost of speed (nprobe == nlist degenerates
to exact). Crucially, vectors are stored in FULL (no quantization), so the only
error source is cell pruning — distances within a probed cell are exact.

nlist/nprobe tuning: a common rule of thumb is nlist ~ sqrt(N); the selector
sets it from N and you tune nprobe to taste (recall vs latency).
"""

from __future__ import annotations

import faiss
import numpy as np

from dedup.indexing.base import VectorIndex


class IvfFlatIndex(VectorIndex):
    def __init__(self, dim: int, nlist: int, nprobe: int):
        super().__init__(dim)
        self.nlist = nlist
        self.nprobe = nprobe
        quantizer = faiss.IndexFlatIP(dim)  # coarse quantizer also uses cosine
        self._index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        self._index.nprobe = nprobe

    @property
    def requires_training(self) -> bool:
        return True

    def train(self, sample: np.ndarray) -> None:
        # The coarse quantizer learns the nlist cell centroids from this sample.
        # FAISS wants >= ~39*nlist points to train well; the selector samples
        # accordingly and we let FAISS warn if it's under-fed.
        self._index.train(np.ascontiguousarray(sample, dtype=np.float32))

    def add(self, vectors: np.ndarray) -> None:
        self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        return self._index.search(np.ascontiguousarray(queries, dtype=np.float32), k)

    def save(self, path: str) -> None:
        faiss.write_index(self._index, path)

    @property
    def ntotal(self) -> int:
        return self._index.ntotal
