"""Common interface for the three FAISS index tiers.

All three indexes operate on **L2-normalized** vectors and use FAISS's
inner-product metric, so a returned similarity is exactly cosine similarity in
[-1, 1]. Normalizing once up front (in the embedding store) lets every tier —
exact or approximate — speak the same "cosine >= threshold" language, which is
what Stage 3's dedup decision needs.

WHY one interface over three very different indexes: the whole point of the
project is that the *same* pipeline swaps its index by scale. Hiding Flat /
IVFFlat / IVFPQ behind `train / add / search / save` means the selector can
return any of them and nothing downstream changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows so inner product == cosine similarity.

    Done in float32 regardless of storage dtype: FAISS wants float32, and
    normalizing in fp16 would lose precision right where it matters (near-1.0
    cosines are exactly the duplicates we're trying to separate from non-dups).
    """
    v = np.ascontiguousarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid div-by-zero on a pathological all-zero vector
    return v / norms


class VectorIndex(ABC):
    """A cosine-similarity nearest-neighbour index over normalized vectors."""

    def __init__(self, dim: int):
        self.dim = dim

    @property
    def requires_training(self) -> bool:
        """IVF indexes must see a sample to learn partitions; Flat does not."""
        return False

    def train(self, sample: np.ndarray) -> None:  # noqa: B027 - no-op default
        """Learn structure from a representative sample (no-op for exact Flat)."""

    @abstractmethod
    def add(self, vectors: np.ndarray) -> None:
        """Add normalized vectors to the index (call in batches at scale)."""

    @abstractmethod
    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (similarities, indices), each shape (len(queries), k).

        Similarities are cosine (inner product of normalized vectors). Indices
        are row positions in insertion order; -1 pads when fewer than k exist.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @property
    @abstractmethod
    def ntotal(self) -> int:
        ...
