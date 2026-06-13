"""Build duplicate clusters from the pairs flagged across Stages 2 and 3.

WHY Stage 4 re-clusters instead of trusting the intermediate survivors: Stages 2
and 3 each removed duplicates *greedily* (keep the lexicographically-first member)
purely so they could run standalone and shrink the set for the next stage. That
greedy keeper is almost never the RIGHT one for detection. Stage 4 throws those
provisional decisions away, rebuilds the full equivalence graph from every
flagged pair (near + semantic, transitively closed via Union-Find), and lets the
annotation-aware resolver pick the real survivor per cluster.

Items = the set that entered Stage 2 (i.e. Stage 1's survivors, or the raw
enumeration if Stage 1 didn't run). Pairs = Stage 2's pairs UNION Stage 3's pairs.
"""

from __future__ import annotations

import json
from pathlib import Path

from dedup import get_logger
from dedup.clustering.union_find import connected_components
from dedup.io.state import read_survivors
from dedup.io.images import enumerate_images

logger = get_logger()


def _load_pairs(output_dir: str | Path, stage: str) -> list[tuple[str, str]]:
    path = Path(output_dir) / stage / "pairs.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    return [(a, b) for a, b in raw]


def build_clusters(image_root: str | Path, output_dir: str | Path
                   ) -> list[list[str]]:
    """Return all components (including singletons) over Stage1 survivors using
    the union of Stage 2 + Stage 3 flagged pairs."""
    items = read_survivors(output_dir, "stage1")
    if items is None:
        items = [str(p) for p in enumerate_images(image_root)]
        logger.info("stage4: no stage1 survivors found; clustering %d enumerated images",
                    len(items))

    pairs = _load_pairs(output_dir, "stage2") + _load_pairs(output_dir, "stage3")
    logger.info("stage4: clustering %d items over %d flagged pairs (stage2+stage3)",
                len(items), len(pairs))

    comps = connected_components(items, pairs)
    n_multi = sum(1 for c in comps if len(c) > 1)
    logger.info("stage4: %d clusters (%d non-trivial, %d singletons)",
                len(comps), n_multi, len(comps) - n_multi)
    return comps
