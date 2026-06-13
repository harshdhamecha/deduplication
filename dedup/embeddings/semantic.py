"""Stage 3 — embedding-based semantic deduplication runner.

Flow:
  1. Extract a normalized embedding per surviving image (resumable: a boolean
     `done` mask tracks which rows are written, so a crash loses <= one batch).
  2. Build the scale-adaptive index via the selector (Flat / IVFFlat / IVFPQ).
  3. Query every embedding for neighbours with cosine >= threshold (excluding
     self) to flag semantic-duplicate pairs.
  4. Group transitively (Union-Find), keep a deterministic representative, and
     persist survivors + pairs (for Stage 4) + groups + metrics.

This stage catches what pixel hashing structurally cannot: same scene under
different crops/lighting/encoding where the *content* matches even though the
pixels (and thus perceptual hashes) diverge.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from dedup import get_logger
from dedup.clustering.union_find import connected_components
from dedup.config import Config
from dedup.embeddings.loader import log_throughput, make_loader, maybe_pre_resize
from dedup.embeddings.store import EmbeddingStore
from dedup.hashing.search import Pair
from dedup.indexing.selector import select_index
from dedup.io.state import DuplicateGroup, load_stage_input, write_groups, write_survivors
from dedup.profiling.metrics import StageMetrics, track_stage

logger = get_logger()

STAGE = "stage3"
# Neighbours fetched per query. Must comfortably exceed the largest expected
# duplicate cluster; pairs beyond this k for one image are recovered via its
# other members' queries anyway (the graph is undirected).
SEARCH_K = 100


def run_stage3(cfg: Config, force: bool = False) -> StageMetrics:
    output_dir = cfg.io.output_dir
    inputs = load_stage_input(cfg.io.image_root, output_dir, STAGE)

    if not cfg.stage3.enabled:
        logger.info("stage3 disabled — passing all %d items through", len(inputs))
        write_survivors(output_dir, STAGE, inputs)
        m = StageMetrics(stage=STAGE, items_in=len(inputs))
        m.save(output_dir)
        return m

    from dedup.embeddings.extractor import build_extractor  # lazy: needs torch

    extractor = build_extractor(cfg.stage3)
    store = EmbeddingStore(Path(output_dir) / STAGE, extractor.dim, cfg.stage3.fp16_embeddings)

    with track_stage(STAGE, len(inputs)) as m:
        _extract(cfg, extractor, store, inputs, force)

        vectors = store.open_read()
        index = select_index(len(inputs), extractor.dim, cfg.stage3)
        _train_and_add(index, vectors, cfg)
        if hasattr(index, "attach_originals"):
            index.attach_originals(vectors)  # exact re-rank source for IVFPQ

        pairs = _flag_pairs(index, vectors, inputs, cfg.stage3.cosine_threshold)

        survivors: list[str] = []
        groups: list[DuplicateGroup] = []
        reason = f"cosine>={cfg.stage3.cosine_threshold}({cfg.stage3.backbone})"
        for comp in connected_components(inputs, pairs):
            members = sorted(comp)
            survivors.append(members[0])
            if len(members) > 1:
                groups.append(DuplicateGroup(kept=members[0], removed=members[1:],
                                             reason=reason))

        m.items_removed = len(inputs) - len(survivors)
        m.extra.update(backbone=cfg.stage3.backbone, dim=extractor.dim,
                       index_type=type(index).__name__, n_flagged_pairs=len(pairs),
                       n_semantic_groups=len(groups))

    write_survivors(output_dir, STAGE, sorted(survivors))
    write_groups(output_dir, STAGE, groups)
    _write_pairs(output_dir, pairs)
    m.save(output_dir)
    return m


def _extract(cfg: Config, extractor, store: EmbeddingStore, inputs: list[str],
             force: bool) -> None:
    """Extract embeddings into the memmap, resumable via a boolean `done` mask."""
    n = len(inputs)
    done_path = Path(cfg.io.output_dir) / STAGE / "done.npy"

    paths = inputs
    if cfg.stage3.pre_resize:
        paths = maybe_pre_resize(inputs, Path(cfg.io.output_dir) / STAGE / "resized")

    if not force and done_path.exists() and store.meta_path.exists():
        done = np.load(done_path)
        # A checkpoint is only valid for the exact input set it was built from. If
        # the inputs changed between runs (re-fetched/re-planted data, or different
        # upstream survivors), the saved mask/memmap rows no longer align with these
        # inputs — blindly resuming either crashes (mask shorter than n) or silently
        # writes embeddings to the wrong rows. Detect the mismatch and recompute.
        meta_n = json.loads(store.meta_path.read_text())["n"]
        ids_path = store.root / "ids.txt"
        stale = (done.shape[0] != n or meta_n != n
                 or (ids_path.exists() and store.load_ids() != inputs))
        if stale:
            logger.warning(
                "stage3 resume: checkpoint is for a different input set "
                "(checkpoint=%d rows, current input=%d) — recomputing embeddings "
                "from scratch. Pass --force to skip this check.",
                done.shape[0], n)
            done = np.zeros(n, dtype=bool)
            store.create(n)  # w+ reallocates the memmap to the new shape
        else:
            store.reopen_write()
            logger.info("stage3 resume: %d/%d embeddings already done", int(done.sum()), n)
    else:
        done = np.zeros(n, dtype=bool)
        store.create(n)

    remaining_orig = [i for i in range(n) if not done[i]]
    if not remaining_orig:
        store.save_ids(inputs)
        return
    remaining_paths = [paths[i] for i in remaining_orig]

    loader = make_loader(remaining_paths, extractor, cfg.stage3.batch_size,
                         cfg.stage3.num_workers)
    start = time.perf_counter()
    n_done = 0
    for local_idxs, batch in loader:
        if batch is None:
            continue
        emb = extractor.embed(batch)
        orig_rows = [remaining_orig[li] for li in local_idxs]
        store.write_rows(orig_rows, emb)
        for r in orig_rows:
            done[r] = True
        n_done += len(orig_rows)
        # Checkpoint every batch so a crash loses at most this batch.
        store.flush()
        np.save(done_path, done)
        if n_done % (cfg.stage3.batch_size * 10) < cfg.stage3.batch_size:
            log_throughput("stage3", n_done, time.perf_counter() - start)

    log_throughput("stage3 (total)", n_done, time.perf_counter() - start)
    store.save_ids(inputs)


def _train_and_add(index, vectors: np.ndarray, cfg: Config) -> None:
    if index.requires_training:
        # Train on a random sample (FAISS wants enough points to fit centroids/
        # codebooks). Cap the sample so training stays fast on huge sets.
        n = len(vectors)
        sample_size = min(n, max(50_000, 40 * cfg.stage3.nlist))
        rng = np.random.default_rng(cfg.seed)
        sample = vectors[np.sort(rng.choice(n, size=min(sample_size, n), replace=False))]
        index.train(np.asarray(sample, dtype=np.float32))

    # Add in batches so we never materialise the whole set in RAM at once.
    batch = 100_000
    for s in range(0, len(vectors), batch):
        index.add(np.asarray(vectors[s:s + batch], dtype=np.float32))


def _flag_pairs(index, vectors: np.ndarray, inputs: list[str], threshold: float
                ) -> set[Pair]:
    """Query each embedding for >= threshold neighbours; return id-pairs."""
    pairs: set[Pair] = set()
    k = min(len(inputs), SEARCH_K)
    batch = 4096
    for s in range(0, len(vectors), batch):
        q = np.asarray(vectors[s:s + batch], dtype=np.float32)
        sims, idxs = index.search(q, k)
        for row in range(len(q)):
            i = s + row
            for sim, j in zip(sims[row], idxs[row]):
                if j < 0 or j == i or sim < threshold:
                    continue
                a, b = inputs[i], inputs[j]
                pairs.add((a, b) if a <= b else (b, a))
    return pairs


def _write_pairs(output_dir: str | Path, pairs: set[Pair]) -> Path:
    path = Path(output_dir) / STAGE / "pairs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump([list(p) for p in sorted(pairs, key=lambda x: (str(x[0]), str(x[1])))], fh)
    return path
