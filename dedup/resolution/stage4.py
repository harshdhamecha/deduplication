"""Stage 4 runner — cluster, resolve (annotation-aware), and check leakage.

Pulls together the three Stage-4 concerns:
  clustering  (dedup.clustering.cluster.build_clusters)
  resolution  (dedup.resolution.strategies.resolve_cluster)
  leakage     (dedup.leakage.check.leakage_report)

Writes the authoritative survivor set plus a full audit trail (per-cluster
decisions, review flags, sampling weights, leakage report) under stage4/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from dedup import get_logger
from dedup.clustering.cluster import build_clusters
from dedup.config import Config
from dedup.io.annotations import ImageAnnotations, get_parser
from dedup.io.state import write_survivors
from dedup.leakage.check import infer_split_from_path, leakage_report
from dedup.profiling.metrics import StageMetrics, track_stage
from dedup.resolution.strategies import resolve_cluster

logger = get_logger()
STAGE = "stage4"


def run_stage4(cfg: Config, force: bool = False) -> StageMetrics:
    output_dir = cfg.io.output_dir
    clusters = build_clusters(cfg.io.image_root, output_dir)
    items = [m for c in clusters for m in c]

    anns = _load_annotations(cfg, items)
    embeddings = _load_embeddings_for(clusters, output_dir) \
        if cfg.stage4.keep_strategy == "keep_central" else None

    with track_stage(STAGE, len(items)) as m:
        survivors: list[str] = []
        removed: list[str] = []
        weights: dict[str, float] = {}
        cluster_records = []
        review = []

        for comp in clusters:
            res = resolve_cluster(comp, anns, cfg.stage4.keep_strategy, embeddings)
            survivors.extend(res.kept)
            removed.extend(res.removed)
            weights.update(res.weights)
            rec = {"members": sorted(comp), "kept": res.kept, "removed": res.removed,
                   "strategy": res.strategy, "review": res.review, "note": res.note}
            if len(comp) > 1:
                cluster_records.append(rec)
            if res.review:
                review.append(rec)

        m.items_removed = len(removed)
        m.extra.update(
            keep_strategy=cfg.stage4.keep_strategy,
            n_clusters=len(clusters),
            n_nontrivial_clusters=len(cluster_records),
            n_review_flagged=len(review),
        )

    # Persist survivors + full audit trail.
    write_survivors(output_dir, STAGE, sorted(survivors))
    out = Path(output_dir) / STAGE
    (out / "clusters.json").write_text(json.dumps(cluster_records, indent=2))
    (out / "removed.txt").write_text("\n".join(sorted(removed)) + ("\n" if removed else ""))
    if weights:
        (out / "weights.json").write_text(json.dumps(weights, indent=2))
    if review:
        (out / "review.json").write_text(json.dumps(review, indent=2))
        logger.warning("stage4: %d cluster(s) flagged for review (conflicting "
                       "annotations) — see %s", len(review), out / "review.json")

    # Leakage check (uses the same flagged pairs as the clustering).
    if cfg.leakage.enabled:
        _run_leakage(cfg, items, anns, m)

    m.save(output_dir)
    return m


def _load_annotations(cfg: Config, items: list[str]) -> dict[str, ImageAnnotations]:
    """Map clustered image PATH -> ImageAnnotations, matching on file basename.

    Annotations key images by COCO file_name; our ids are filesystem paths, so we
    bridge them by basename. If no annotations are configured we return empty (the
    resolver degrades to path-tiebreak / falls back from keep_central)."""
    if not cfg.io.annotations or not Path(cfg.io.annotations).exists():
        logger.warning("stage4: no annotations at %s — resolution will be "
                       "annotation-blind (boxes/classes treated as 0).",
                       cfg.io.annotations)
        return {}
    by_id = get_parser(cfg.io.annotation_format, cfg.io.annotations).parse()
    by_basename = {os.path.basename(a.file_name): a for a in by_id.values()}
    # Re-key onto the actual paths we clustered (Stage 1 survivors).
    anns = {p: by_basename[os.path.basename(p)] for p in items
            if os.path.basename(p) in by_basename}
    logger.info("stage4: matched annotations for %d/%d clustered images",
                len(anns), len(items))
    return anns


def _load_embeddings_for(clusters, output_dir) -> dict[str, np.ndarray]:
    """Load embeddings (path -> vector) for clustered members, for keep_central."""
    store_dir = Path(output_dir) / "stage3"
    ids_file = store_dir / "ids.txt"
    meta = store_dir / "embeddings.meta"
    if not ids_file.exists() or not meta.exists():
        logger.warning("stage4: keep_central requested but no Stage 3 embeddings "
                       "found — will fall back to keep_most_annotated per cluster.")
        return {}
    from dedup.embeddings.store import EmbeddingStore

    info = json.loads(meta.read_text())
    store = EmbeddingStore(store_dir, info["dim"], info["dtype"] == "float16")
    ids = store.load_ids()
    mm = store.open_read()
    row_of = {p: i for i, p in enumerate(ids)}
    wanted = {m for c in clusters if len(c) > 1 for m in c}
    return {p: np.asarray(mm[row_of[p]], dtype=np.float32) for p in wanted if p in row_of}


def _run_leakage(cfg: Config, items, anns: dict[str, ImageAnnotations], m: StageMetrics):
    known = [cfg.leakage.train_split] + cfg.leakage.eval_splits
    split_of = {i: infer_split_from_path(i, known) for i in items}
    split_of = {k: v for k, v in split_of.items() if v is not None}
    partition_of = {p: a.partition_key for p, a in anns.items() if a.partition_key}

    pairs = _all_pairs(cfg.io.output_dir)
    report = leakage_report(items, pairs, split_of, cfg.leakage.train_split,
                            cfg.leakage.eval_splits, partition_of or None)
    (Path(cfg.io.output_dir) / STAGE / "leakage.json").write_text(json.dumps(report, indent=2))
    m.extra["leakage"] = {es: r["leaked_fraction"] for es, r in report["by_split"].items()}

    if not split_of:
        logger.warning("stage4 leakage: could not infer any train/eval split from "
                       "paths (looked for %s as path components) — leakage report is "
                       "empty. Provide split-labelled paths or metadata.", known)


def _all_pairs(output_dir: str | Path) -> list[tuple[str, str]]:
    pairs = []
    for stage in ("stage2", "stage3"):
        path = Path(output_dir) / stage / "pairs.json"
        if path.exists():
            pairs.extend((a, b) for a, b in json.loads(path.read_text()))
    return pairs
