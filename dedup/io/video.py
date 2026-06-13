"""Optional video-frame subsampling pre-filter.

Consecutive frames from one video are near-identical, so embedding every frame
burns GPU time to rediscover that they're duplicates. When sequence metadata is
present (a partition key per image + an orderable frame name), we can cheaply
keep only every k-th frame BEFORE the expensive Stage 3 embedding, trading a
little recall for a large compute saving.

This is a deliberate, opt-in shortcut (stage4.video_subsample_k > 1): it assumes
within-shot redundancy is acceptable to drop. With k<=1 it is a no-op. It is NOT
a substitute for the dedup stages — it's a way to not pay full price for footage
we already know is redundant.
"""

from __future__ import annotations

import os
from collections import defaultdict

from dedup import get_logger

logger = get_logger()


def subsample_by_partition(
    items: list[str],
    partition_of: dict[str, str],
    k: int,
    order_key=os.path.basename,
) -> list[str]:
    """Keep every k-th frame within each partition; pass through items with no
    partition. Frame order within a partition is by ``order_key`` (default: file
    name, which sorts sequential frame numbers correctly when zero-padded)."""
    if k <= 1:
        return items

    by_part: dict[str, list[str]] = defaultdict(list)
    no_part: list[str] = []
    for it in items:
        part = partition_of.get(it)
        (by_part[part] if part is not None else no_part).append(it)

    kept = list(no_part)
    for part, frames in by_part.items():
        frames_sorted = sorted(frames, key=order_key)
        kept.extend(frames_sorted[::k])  # frame 0, k, 2k, ...

    dropped = len(items) - len(kept)
    logger.info("video subsample (k=%d): %d -> %d frames (dropped %d redundant)",
                k, len(items), len(kept), dropped)
    return sorted(kept)
