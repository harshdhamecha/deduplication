"""Inter-stage state: how survivors and duplicate groups are persisted to disk.

Every stage writes two artifacts under ``output_dir/<stage>/``:
  survivors.txt          one surviving image path per line (input to next stage)
  duplicate_groups.json  the groups it found: which image was kept, which removed

WHY persist between stages (not just pass in memory): the spec requires each
stage to be independently runnable and the pipeline to be resumable. With state
on disk, ``dedup run --stage 3`` can pick up Stage 2's survivors after a crash or
on a fresh process, and the duplicate_groups records give the final report an
audit trail of *why* each image was dropped.

A DuplicateGroup is intentionally generic across stages 1/2/3 — exact, near, and
semantic duplicates all reduce to "these ids are the same thing; keep one".
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dedup import get_logger
from dedup.io.images import enumerate_images

logger = get_logger()


@dataclass
class DuplicateGroup:
    kept: str               # the survivor (resolution decided per stage/strategy)
    removed: list[str]      # ids dropped as duplicates of `kept`
    reason: str             # e.g. "sha256", "phash<=8", "cosine>=0.92"
    key: str | None = None  # the shared hash, when meaningful

    def to_dict(self) -> dict:
        return asdict(self)


def stage_dir(output_dir: str | Path, stage: str) -> Path:
    d = Path(output_dir) / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_survivors(output_dir: str | Path, stage: str, survivors: list[str]) -> Path:
    path = stage_dir(output_dir, stage) / "survivors.txt"
    # Write to a temp file then rename: a crash mid-write must not leave a
    # truncated survivors list that a resumed run would silently trust.
    tmp = path.with_suffix(".txt.part")
    tmp.write_text("\n".join(survivors) + ("\n" if survivors else ""))
    tmp.rename(path)
    return path


def read_survivors(output_dir: str | Path, stage: str) -> list[str] | None:
    path = Path(output_dir) / stage / "survivors.txt"
    if not path.exists():
        return None
    return [line for line in path.read_text().splitlines() if line]


def write_groups(output_dir: str | Path, stage: str, groups: list[DuplicateGroup]) -> Path:
    path = stage_dir(output_dir, stage) / "duplicate_groups.json"
    with open(path, "w") as fh:
        json.dump([g.to_dict() for g in groups], fh, indent=2)
    return path


# Stage ordering, so "the input to stage N" is well defined.
_STAGE_ORDER = ["stage1", "stage2", "stage3", "stage4"]


def load_stage_input(image_root: str | Path, output_dir: str | Path, stage: str) -> list[str]:
    """Return the input id list for ``stage``.

    For stage1 (or when no prior survivors exist) this is the full image
    enumeration. Otherwise it's the survivors of the most recent prior stage that
    has run — so the cascade composes, and a single stage can be re-run in place.
    """
    idx = _STAGE_ORDER.index(stage)
    for prev in reversed(_STAGE_ORDER[:idx]):
        survivors = read_survivors(output_dir, prev)
        if survivors is not None:
            logger.info("%s input: %d survivors from %s", stage, len(survivors), prev)
            return survivors

    paths = [str(p) for p in enumerate_images(image_root)]
    logger.info("%s input: %d images enumerated from %s", stage, len(paths), image_root)
    return paths
