"""Stage 1 — exact duplicate detection via SHA-256 over raw file bytes.

This is the cheapest, most certain stage: two files with the same SHA-256 are
byte-identical (collision probability is cryptographically negligible), so we can
drop one with zero risk — including zero annotation risk, since identical bytes
imply identical pixels and therefore identical "correct" boxes.

Because exact dups are content-identical, *which* survivor we keep is arbitrary;
we keep the lexicographically-first path (deterministic). The interesting keep
decisions (most-annotated, highest-res) belong to Stage 4, where duplicates are
only *near*-identical and the choice actually matters.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dedup import get_logger
from dedup.config import Config
from dedup.hashing.storage import select_backend
from dedup.io.state import DuplicateGroup, load_stage_input, write_groups, write_survivors
from dedup.profiling.metrics import StageMetrics, track_stage

logger = get_logger()

STAGE = "stage1"
_CHUNK = 1 << 20  # 1 MiB read chunks — hash huge files without loading them whole


def sha256_file(path: str | Path) -> str:
    """Streaming SHA-256 of a file's bytes (constant memory, any file size)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _checkpoint_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / STAGE / "checkpoint.json"


def run_stage1(cfg: Config, force: bool = False) -> StageMetrics:
    """Run Stage 1 and persist survivors + duplicate groups + metrics.

    Resumability: progress (count of processed files) is checkpointed every
    batch. On the disk-backed (LMDB) path the hash map is itself durable, so a
    resumed run skips the already-processed prefix of the deterministic input
    order. The in-memory path restarts from scratch — acceptable because it is
    only chosen for small datasets where a full re-run is cheap.
    """
    output_dir = cfg.io.output_dir
    inputs = load_stage_input(cfg.io.image_root, output_dir, STAGE)

    if not cfg.stage1.enabled:
        logger.info("stage1 disabled — passing all %d items through", len(inputs))
        write_survivors(output_dir, STAGE, inputs)
        m = StageMetrics(stage=STAGE, items_in=len(inputs))
        m.save(output_dir)
        return m

    lmdb_path = Path(output_dir) / STAGE / "hashes.lmdb"
    backend = select_backend(
        cfg.stage1.storage_backend, len(inputs), cfg.stage1.lmdb_threshold, lmdb_path
    )
    is_lmdb = type(backend).__name__ == "LmdbBackend"

    # Resume only on the durable (LMDB) path.
    start_idx = 0
    ckpt = _checkpoint_path(output_dir)
    if is_lmdb and not force and ckpt.exists():
        start_idx = json.loads(ckpt.read_text()).get("processed", 0)
        logger.info("stage1 resume: skipping %d already-hashed files", start_idx)

    with track_stage(STAGE, len(inputs)) as m:
        for i in range(start_idx, len(inputs)):
            path = inputs[i]
            backend.add(sha256_file(path), path)
            if (i + 1) % 1000 == 0:
                if is_lmdb:
                    ckpt.write_text(json.dumps({"processed": i + 1}))
                logger.info("stage1 hashed %d/%d", i + 1, len(inputs))

        # Resolve groups: keep the first path, drop the rest as exact dups.
        survivors: list[str] = []
        groups: list[DuplicateGroup] = []
        for key, paths in backend.groups():
            paths = sorted(paths)
            survivors.append(paths[0])
            if len(paths) > 1:
                groups.append(DuplicateGroup(
                    kept=paths[0], removed=paths[1:], reason="sha256", key=key,
                ))

        m.items_removed = len(inputs) - len(survivors)
        m.extra["backend"] = "lmdb" if is_lmdb else "memory"
        m.extra["n_duplicate_groups"] = len(groups)

    backend.close()
    write_survivors(output_dir, STAGE, sorted(survivors))
    write_groups(output_dir, STAGE, groups)
    m.save(output_dir)
    return m
