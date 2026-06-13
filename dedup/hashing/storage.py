"""Storage backends for Stage 1's hash -> file-list bookkeeping.

One interface, two implementations, picked by scale:

  * InMemoryBackend  — a plain dict. Fastest, simplest, default for small data.
  * LmdbBackend      — a memory-mapped on-disk key-value store. Used when the set
                       of hashes would be too large to hold in RAM.

WHY this abstraction exists at all: Stage 1 must hold, for every distinct file
hash seen so far, the list of paths that produced it. At a few hundred thousand
images that's a trivial dict. At 50M images the dict of hashes alone is many GB
and we'd rather page it to disk than OOM. The interface lets the rest of Stage 1
stay identical regardless of which side of that line we're on.

WHY LMDB (not RocksDB, not sqlite): LMDB is a single-file, memory-mapped,
read-optimised KV store with trivial Python bindings and no server process —
ideal for a write-once/read-many dedup pass on a single machine. RocksDB wins on
write-heavy workloads we don't have here; sqlite would work but carries SQL
overhead we don't need for pure key->value.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from dedup import get_logger

logger = get_logger()


class StorageBackend(ABC):
    """Maps a hash key to the list of item ids (file paths) that share it."""

    @abstractmethod
    def add(self, key: str, value: str) -> None:
        """Append ``value`` to the list stored under ``key``."""

    @abstractmethod
    def groups(self) -> Iterator[tuple[str, list[str]]]:
        """Yield (key, values) for every key. Duplicate groups have len>1."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of distinct keys."""

    def close(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Release resources (LMDB env). No-op for in-memory."""


class InMemoryBackend(StorageBackend):
    def __init__(self) -> None:
        self._store: dict[str, list[str]] = {}

    def add(self, key: str, value: str) -> None:
        self._store.setdefault(key, []).append(value)

    def groups(self) -> Iterator[tuple[str, list[str]]]:
        yield from self._store.items()

    def __len__(self) -> int:
        return len(self._store)


class LmdbBackend(StorageBackend):
    """Disk-backed backend. Values are stored newline-joined under each key.

    Appending is read-modify-write per key, which is fine for dedup: duplicate
    groups are almost always tiny (1-3 paths), so the value blobs stay small and
    rewrites are cheap. The cost we're buying down is *RAM*, not write latency.
    """

    def __init__(self, path: str | Path, map_size: int = 32 * 1024**3):
        # Lazy import: lmdb is only needed on the large-scale path, so the rest
        # of the pipeline (and its tests) runs without it installed.
        import lmdb  # noqa: PLC0415

        self.path = str(path)
        Path(self.path).mkdir(parents=True, exist_ok=True)
        # map_size is the max DB size (sparse file); 32GB is generous headroom —
        # it costs nothing until actually written.
        self.env = lmdb.open(self.path, map_size=map_size, subdir=True)

    def add(self, key: str, value: str) -> None:
        kb = key.encode()
        with self.env.begin(write=True) as txn:
            existing = txn.get(kb)
            blob = (existing + b"\n" + value.encode()) if existing else value.encode()
            txn.put(kb, blob)

    def groups(self) -> Iterator[tuple[str, list[str]]]:
        with self.env.begin() as txn:
            for kb, vb in txn.cursor():
                yield kb.decode(), vb.decode().split("\n")

    def __len__(self) -> int:
        with self.env.begin() as txn:
            return txn.stat()["entries"]

    def close(self) -> None:
        self.env.close()


def select_backend(
    mode: str,
    estimated_count: int,
    threshold: int,
    lmdb_path: str | Path | None = None,
) -> StorageBackend:
    """Choose a backend, logging the decision (the project's visibility rule).

    ``mode`` is "auto" | "memory" | "lmdb". In auto mode we switch to LMDB once
    the estimated item count exceeds ``threshold`` — the point past which holding
    every hash in a dict stops being comfortable in RAM.
    """
    use_lmdb = mode == "lmdb" or (mode == "auto" and estimated_count > threshold)

    if use_lmdb:
        if lmdb_path is None:
            raise ValueError("LMDB backend selected but no lmdb_path provided")
        logger.info(
            "Stage1 storage: LMDB (disk-backed) — estimated %d items > threshold %d; "
            "holding hashes in RAM would be wasteful at this scale.",
            estimated_count, threshold,
        )
        return LmdbBackend(lmdb_path)

    logger.info(
        "Stage1 storage: in-memory dict — estimated %d items <= threshold %d; "
        "fits comfortably in RAM, no reason to pay disk overhead.",
        estimated_count, threshold,
    )
    return InMemoryBackend()
