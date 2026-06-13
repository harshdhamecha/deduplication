"""Cross-split train/test leakage check — the project's headline metric.

A near-duplicate of a test image sitting in the training set silently inflates
reported accuracy: the model effectively memorised the test sample. So after
dedup, before splitting (or to audit an existing split), we report:

    "X% of {val/test} images have a near-duplicate in train."

We reuse the SAME pair signal Stages 2 and 3 produced — a test image is
"leaked" if it shares a flagged pair (perceptual OR semantic, above threshold)
with any training image. No new threshold, no new model: the contamination
number is a direct consequence of the duplicate graph we already built.

Hard partition key: if images carry source metadata (video id, capture session,
URL), frames from one source must never straddle the split — they're the same
scene. We report any partition that spans train and an eval split, because that
is leakage by construction even when pixel/feature similarity didn't trip.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from dedup import get_logger

logger = get_logger()


def leakage_report(
    items: Iterable[str],
    pairs: Iterable[tuple[str, str]],
    split_of: dict[str, str],
    train_split: str,
    eval_splits: list[str],
    partition_of: dict[str, str] | None = None,
) -> dict:
    """Compute the leakage report. ``split_of`` maps id -> split name; ids with no
    known split are ignored (e.g. unlabelled extras)."""
    neighbours: dict[str, set[str]] = defaultdict(set)
    for a, b in pairs:
        neighbours[a].add(b)
        neighbours[b].add(a)

    report: dict = {"by_split": {}, "partition_violations": []}
    for es in eval_splits:
        eval_imgs = [i for i in items if split_of.get(i) == es]
        leaked = [
            i for i in eval_imgs
            if any(split_of.get(n) == train_split for n in neighbours.get(i, ()))
        ]
        frac = round(len(leaked) / len(eval_imgs), 4) if eval_imgs else 0.0
        report["by_split"][es] = {
            "eval_images": len(eval_imgs),
            "leaked_images": len(leaked),
            "leaked_fraction": frac,
            "examples": leaked[:10],   # a few for eyeballing; full set lives in pairs
        }

    # Partition spanning: a source whose frames appear in train AND an eval split.
    if partition_of:
        by_part: dict[str, set[str]] = defaultdict(set)
        for img, part in partition_of.items():
            s = split_of.get(img)
            if s is not None:
                by_part[part].add(s)
        for part, splits in by_part.items():
            if train_split in splits and any(es in splits for es in eval_splits):
                report["partition_violations"].append(
                    {"partition": part, "splits": sorted(splits)})

    _print_summary(report, train_split)
    return report


def _print_summary(report: dict, train_split: str) -> None:
    print("\n=== Cross-Split Leakage ===")
    for es, r in report["by_split"].items():
        print(f"  {es}: {r['leaked_images']}/{r['eval_images']} images "
              f"({r['leaked_fraction'] * 100:.2f}%) have a near-duplicate in {train_split}")
    if report["partition_violations"]:
        print(f"  ! {len(report['partition_violations'])} partition(s) span "
              f"train and eval (same-source frames split across the boundary)")


def infer_split_from_path(path: str, known_splits: list[str]) -> str | None:
    """Best-effort split inference: a split name appearing as a path component.

    e.g. '.../images/train2017/x.jpg' -> 'train' if 'train' is a known split and
    a component contains it. Used when no explicit split metadata is provided.
    """
    parts = path.replace("\\", "/").lower().split("/")
    for split in known_splits:
        if any(split in p for p in parts):
            return split
    return None
