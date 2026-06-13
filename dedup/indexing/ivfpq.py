"""Tier 3 — IndexIVFPQ: partitioned AND product-quantized (compressed).

WHEN: large datasets (10M+) where even full uncompressed vectors no longer fit
in RAM. Product Quantization splits each vector into ``m`` sub-vectors and
replaces each with an ``nbits``-bit codebook id. A 768-d fp32 vector (3072 bytes)
becomes m=64 codes at nbits=8 -> 64 bytes: a ~48x compression that lets a
billion vectors live on disk/RAM that otherwise couldn't.

WHAT IT TRADES: real approximation error, on TWO axes now — IVF cell pruning
(as in IVFFlat) *and* lossy PQ codes (the stored vector is a codebook
reconstruction, not the original). PQ distances are therefore biased.

HOW WE CLAW PRECISION BACK — exact re-ranking:
We ask the PQ index for a generous ``rerank_k`` candidates (cheap, approximate),
then recompute EXACT cosine for just those candidates against the ORIGINAL
vectors held on a memory-mapped array, and keep the true top-k. This is the
standard "approximate shortlist, exact rescoring" pattern: we pay PQ's tiny cost
over the whole set and exact cost over only a handful per query, recovering most
of the precision PQ gave up. It requires the originals to remain available
(memmapped) — which is why Stage 3 stores them rather than discarding after build.
"""

from __future__ import annotations

import faiss
import numpy as np

from dedup.indexing.base import VectorIndex


class IvfPqIndex(VectorIndex):
    def __init__(self, dim: int, nlist: int, nprobe: int, m: int, nbits: int,
                 rerank: bool = True, rerank_k: int = 100):
        super().__init__(dim)
        if dim % m != 0:
            # PQ partitions the dimension into m equal sub-vectors; m must divide
            # dim. Failing loudly here beats a cryptic FAISS assertion later.
            raise ValueError(f"pq_m={m} must divide embedding dim={dim}")
        self.nlist, self.nprobe, self.m, self.nbits = nlist, nprobe, m, nbits
        self.rerank = rerank
        self.rerank_k = rerank_k
        quantizer = faiss.IndexFlatIP(dim)
        self._index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits,
                                       faiss.METRIC_INNER_PRODUCT)
        self._index.nprobe = nprobe
        # Reference to the original (memmapped) vectors for exact re-ranking.
        self._originals: np.ndarray | None = None

    @property
    def requires_training(self) -> bool:
        return True

    def attach_originals(self, vectors: np.ndarray) -> None:
        """Provide the original (full-precision, possibly memmapped) vectors used
        for exact re-ranking. Insertion order must match what was add()ed."""
        self._originals = vectors

    def train(self, sample: np.ndarray) -> None:
        # Training learns BOTH the nlist coarse centroids and the PQ codebooks,
        # so it needs a larger sample than IVFFlat (the selector sizes it).
        self._index.train(np.ascontiguousarray(sample, dtype=np.float32))

    def add(self, vectors: np.ndarray) -> None:
        self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(queries, dtype=np.float32)
        if not self.rerank:
            return self._index.search(q, k)
        if self._originals is None:
            raise RuntimeError(
                "rerank=True but originals not attached; call attach_originals() "
                "with the memmapped vectors before searching."
            )
        # 1) Approximate shortlist from PQ (wider than k).
        shortlist = max(self.rerank_k, k)
        _, cand_idx = self._index.search(q, shortlist)
        # 2) Exact cosine rescoring of the shortlist against the originals.
        return self._rerank(q, cand_idx, k)

    def _rerank(self, queries: np.ndarray, cand_idx: np.ndarray, k: int
                ) -> tuple[np.ndarray, np.ndarray]:
        n = len(queries)
        out_sim = np.full((n, k), -1.0, dtype=np.float32)
        out_idx = np.full((n, k), -1, dtype=np.int64)
        for i in range(n):
            cands = cand_idx[i][cand_idx[i] >= 0]  # drop -1 padding
            if cands.size == 0:
                continue
            # Exact inner product against original vectors == true cosine.
            exact = self._originals[cands].astype(np.float32) @ queries[i]
            order = np.argsort(-exact)[:k]
            out_sim[i, :len(order)] = exact[order]
            out_idx[i, :len(order)] = cands[order]
        return out_sim, out_idx

    def save(self, path: str) -> None:
        faiss.write_index(self._index, path)

    @property
    def ntotal(self) -> int:
        return self._index.ntotal
