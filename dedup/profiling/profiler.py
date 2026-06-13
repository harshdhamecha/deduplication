"""Duplicate-distribution profiler — measure BEFORE removing anything.

Profiling-first discipline: the project's value is a *defensible number*, so we
characterise the dataset's duplication before any stage deletes a file. That way
the impact claim ("we removed X% redundant data, Y% of it exact") is grounded in
a measurement taken on the untouched input.

This module currently reports the EXACT-duplicate distribution (the part we can
compute with Stage 1's machinery alone). The near-dup and semantic-dup fractions
are layered on in Step 6 once Stages 2 and 3 exist, behind this same entry point.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from dedup import get_logger
from dedup.config import Config
from dedup.hashing.exact import sha256_file
from dedup.io.state import load_stage_input

logger = get_logger()


def profile_exact(image_paths: list[str]) -> dict:
    """Compute exact-duplicate statistics over the given files (no removal)."""
    by_hash: dict[str, int] = Counter()
    for p in image_paths:
        by_hash[sha256_file(p)] += 1

    total = len(image_paths)
    distinct = len(by_hash)
    # "Redundant" files = every copy beyond the first in each group.
    redundant = total - distinct
    # Histogram of group sizes: how many hashes appear once, twice, ...
    size_hist = dict(sorted(Counter(by_hash.values()).items()))

    return {
        "total_files": total,
        "distinct_files": distinct,
        "exact_duplicate_files": redundant,
        "exact_duplicate_fraction": round(redundant / total, 4) if total else 0.0,
        "group_size_histogram": {str(k): v for k, v in size_hist.items()},
    }


def profile(cfg: Config) -> dict:
    """Run the profiler over the dataset and write/print a report."""
    inputs = load_stage_input(cfg.io.image_root, cfg.io.output_dir, "stage1")
    logger.info("profiling %d files (exact-duplicate distribution)", len(inputs))

    report = {"exact": profile_exact(inputs)}
    # near/semantic added in Step 6 — recorded explicitly so the report shows
    # what's measured vs not, rather than silently omitting it.
    report["near"] = {"status": "pending (Stage 2 — Step 6)"}
    report["semantic"] = {"status": "pending (Stage 3 — Step 6)"}

    out = Path(cfg.io.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "profile.json").write_text(json.dumps(report, indent=2))

    _print_summary(report["exact"])
    return report


def _print_summary(ex: dict) -> None:
    print("\n=== Duplicate Distribution (exact) ===")
    print(f"  total files     : {ex['total_files']}")
    print(f"  distinct files  : {ex['distinct_files']}")
    print(f"  exact duplicates: {ex['exact_duplicate_files']} "
          f"({ex['exact_duplicate_fraction'] * 100:.2f}%)")
    print("  group-size histogram (size: #groups):")
    for size, n in ex["group_size_histogram"].items():
        print(f"    {size:>3} : {n}")
