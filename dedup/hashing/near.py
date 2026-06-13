"""Stage 2 — perceptual / near-duplicate detection runner.

Pipeline for the stage:
  1. Hash every surviving image with the configured algorithms (phash+dhash default).
  2. For each algorithm, find all id-pairs within the Hamming threshold using the
     selected search strategy (brute force / multi-index / BK-tree).
  3. Combine the per-algorithm pair sets ("any" = union, "all" = intersection).
  4. Persist the flagged pairs (for Stage 4's cross-stage clustering) and also
     reduce to survivors here (group via Union-Find, keep one per group) so the
     stage is independently runnable and the cascade can shrink before Stage 3.

Note the division of labour: Stage 2 only *flags* near-duplicate pairs. The
real, annotation-aware decision of WHICH image to keep belongs to Stage 4 — here
we keep a deterministic representative (lexicographically first) purely so the
survivor set is well-defined when running this stage alone.
"""

from __future__ import annotations

import json
from collections.abc import Hashable
from pathlib import Path

from dedup import get_logger
from dedup.clustering.union_find import connected_components
from dedup.config import Config
from dedup.hashing.perceptual import compute_hashes
from dedup.hashing.search import Pair, get_search_strategy
from dedup.io.state import DuplicateGroup, load_stage_input, write_groups, write_survivors
from dedup.profiling.metrics import StageMetrics, track_stage

logger = get_logger()

STAGE = "stage2"


def _combine(pair_sets: list[set[Pair]], mode: str) -> set[Pair]:
    if not pair_sets:
        return set()
    if mode == "any":
        return set().union(*pair_sets)       # union: flagged by ANY hash
    if mode == "all":
        return set.intersection(*pair_sets)  # intersection: flagged by ALL hashes
    raise ValueError(f"Unknown combine mode '{mode}' (use 'any' or 'all')")


def run_stage2(cfg: Config, force: bool = False) -> StageMetrics:
    output_dir = cfg.io.output_dir
    inputs = load_stage_input(cfg.io.image_root, output_dir, STAGE)

    if not cfg.stage2.enabled:
        logger.info("stage2 disabled — passing all %d items through", len(inputs))
        write_survivors(output_dir, STAGE, inputs)
        m = StageMetrics(stage=STAGE, items_in=len(inputs))
        m.save(output_dir)
        return m

    with track_stage(STAGE, len(inputs)) as m:
        hashes_by_algo = compute_hashes(inputs, cfg.stage2.hashes, cfg.stage2.hash_size)

        # One search per algorithm; the strategy is chosen once by scale.
        strategy = get_search_strategy(cfg.stage2.search_strategy, len(inputs))
        per_algo: list[set[Pair]] = []
        for algo, hashes in hashes_by_algo.items():
            pairs = strategy.find_pairs(hashes, cfg.stage2.hamming_threshold)
            logger.info("stage2 %s: %d near pairs (<=%d)", algo, len(pairs),
                        cfg.stage2.hamming_threshold)
            per_algo.append(pairs)

        pairs = _combine(per_algo, cfg.stage2.combine)
        reason = f"{'|'.join(cfg.stage2.hashes)}<={cfg.stage2.hamming_threshold}" \
                 f"[{cfg.stage2.combine}]"

        # Group transitively and keep a deterministic representative per group.
        survivors: list[str] = []
        groups: list[DuplicateGroup] = []
        for comp in connected_components(inputs, pairs):
            members = sorted(comp)
            survivors.append(members[0])
            if len(members) > 1:
                groups.append(DuplicateGroup(kept=members[0], removed=members[1:],
                                             reason=reason))

        m.items_removed = len(inputs) - len(survivors)
        m.extra["search_strategy"] = type(strategy).__name__
        m.extra["n_flagged_pairs"] = len(pairs)
        m.extra["n_near_dup_groups"] = len(groups)

    write_survivors(output_dir, STAGE, sorted(survivors))
    write_groups(output_dir, STAGE, groups)
    _write_pairs(output_dir, pairs)  # for Stage 4 cross-stage clustering
    m.save(output_dir)
    return m


def _write_pairs(output_dir: str | Path, pairs: set[Pair]) -> Path:
    path = Path(output_dir) / STAGE / "pairs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump([list(p) for p in sorted(pairs, key=lambda x: (str(x[0]), str(x[1])))], fh)
    return path
