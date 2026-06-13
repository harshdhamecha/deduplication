"""Memory-mapped embedding store.

Embeddings are written to a flat numpy memmap on disk, not held in a Python list
or a giant in-RAM array. WHY:
  * Scale: 10M x 768 fp16 is ~15GB — it must live on disk and be paged in.
  * Resumability: a memmap is durable; a crash mid-extraction loses only the
    un-written tail, and the row count we've checkpointed tells us where to resume.
  * Re-ranking: IVFPQ's exact rescoring needs the ORIGINAL vectors after the
    compressed index is built; a memmap gives random access without loading all.

We store **L2-normalized** vectors, so inner product == cosine everywhere (index
search and re-rank alike). fp16 is the default (halves footprint); the precision
loss is negligible for cosine near the dedup threshold and still vastly more
accurate than the PQ codes it re-ranks against. Set fp16_embeddings=false for
fp32 if you want bit-exact re-ranking.

Layout under output_dir/stage3/:
  embeddings.dat   the (N, dim) memmap
  embeddings.meta  json: {n, dim, dtype} so it can be reopened
  ids.txt          row i -> image path (alignment with the survivor list)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dedup.indexing.base import normalize


class EmbeddingStore:
    def __init__(self, root: str | Path, dim: int, fp16: bool = True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.dtype = np.float16 if fp16 else np.float32
        self._mm: np.memmap | None = None
        self._n = 0

    @property
    def data_path(self) -> Path:
        return self.root / "embeddings.dat"

    @property
    def meta_path(self) -> Path:
        return self.root / "embeddings.meta"

    def create(self, n: int) -> None:
        """Allocate the (n, dim) memmap on disk (sparse until written)."""
        self._n = n
        self._mm = np.memmap(self.data_path, dtype=self.dtype, mode="w+", shape=(n, self.dim))
        self.meta_path.write_text(json.dumps(
            {"n": n, "dim": self.dim, "dtype": np.dtype(self.dtype).name}))

    def reopen_write(self) -> None:
        """Reopen an existing memmap for writing (mode r+), for resumed runs.

        Unlike create() (mode w+, which zeroes the file), this preserves rows
        already written so a crash mid-extraction resumes instead of restarting.
        """
        meta = json.loads(self.meta_path.read_text())
        self._n = meta["n"]
        self._mm = np.memmap(self.data_path, dtype=np.dtype(meta["dtype"]),
                             mode="r+", shape=(meta["n"], meta["dim"]))

    def write_rows(self, indices: list[int], vectors: np.ndarray) -> None:
        """Write normalized vectors to arbitrary (possibly non-contiguous) rows.

        Multi-worker decode yields batches in non-contiguous order, so we write
        by explicit row index rather than a contiguous slice.
        """
        assert self._mm is not None, "call create()/reopen_write() first"
        normed = normalize(vectors).astype(self.dtype)
        for row, vec in zip(indices, normed):
            self._mm[row] = vec

    def flush(self) -> None:
        if self._mm is not None:
            self._mm.flush()

    def open_read(self) -> np.memmap:
        """Reopen the store read-only (e.g. for re-ranking / a fresh process)."""
        meta = json.loads(self.meta_path.read_text())
        return np.memmap(self.data_path, dtype=np.dtype(meta["dtype"]),
                         mode="r", shape=(meta["n"], meta["dim"]))

    def save_ids(self, ids: list[str]) -> None:
        (self.root / "ids.txt").write_text("\n".join(ids) + ("\n" if ids else ""))

    def load_ids(self) -> list[str]:
        return [x for x in (self.root / "ids.txt").read_text().splitlines() if x]
