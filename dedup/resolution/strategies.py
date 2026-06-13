"""Annotation-aware cluster resolution: given a duplicate cluster, decide which
image to KEEP. For object detection the right survivor is the one that preserves
the most supervision, not an arbitrary one — deleting the wrong copy silently
throws away labelled boxes.

Strategies (selectable via stage4.keep_strategy):
  keep_most_annotated (default) — max boxes, tie-break on class diversity. Keeps
      the richest-labelled view of the scene.
  keep_highest_res             — max pixel area. Best when labels are comparable
      but you want the sharpest image for training.
  keep_central                 — embedding closest to the cluster centroid. The
      most "representative" frame; needs Stage 3 embeddings.
  weighted_sample              — keep ALL, emit per-image weight 1/cluster_size.
      Doesn't delete anything — downweights redundancy at train time instead, so
      no information is lost. Choose this when you'd rather not hard-delete.

CONFLICT GUARD: if cluster members carry *disagreeing* class sets (the same
scene labelled with different objects present), we do NOT auto-resolve — we keep
everyone and flag the cluster for human review. Auto-picking one would bake in a
labelling error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dedup.io.annotations import ImageAnnotations


@dataclass
class ResolutionResult:
    kept: list[str]
    removed: list[str]
    strategy: str
    review: bool = False                       # flagged for human review (conflict)
    weights: dict[str, float] = field(default_factory=dict)  # weighted_sample only
    note: str = ""


def _annotated_members(members, anns):
    return [(m, anns[m]) for m in members if m in anns and anns[m].num_boxes > 0]


def _has_class_conflict(members, anns: dict[str, ImageAnnotations]) -> bool:
    """True if annotated members disagree on which classes are present.

    We treat distinct, non-nested class sets as a conflict: if one frame is
    labelled {person, car} and its near-duplicate {person}, that's an annotation
    gap worth a human's eyes, not an automatic keep. Identical or subset sets are
    not a conflict (one is just a superset — that's normal and resolvable).
    """
    class_sets = [a.class_ids for _, a in _annotated_members(members, anns) if a.class_ids]
    for i in range(len(class_sets)):
        for j in range(i + 1, len(class_sets)):
            a, b = class_sets[i], class_sets[j]
            if not (a <= b or b <= a):          # neither is a subset of the other
                return True
    return False


def resolve_cluster(
    members: list[str],
    anns: dict[str, ImageAnnotations],
    strategy: str,
    embeddings: dict[str, np.ndarray] | None = None,
) -> ResolutionResult:
    members = sorted(members)
    if len(members) == 1:
        return ResolutionResult(kept=members, removed=[], strategy=strategy)

    if strategy == "weighted_sample":
        # Keep everything; redundancy is corrected by sampling weight, not deletion.
        w = round(1.0 / len(members), 6)
        return ResolutionResult(kept=members, removed=[], strategy=strategy,
                                weights={m: w for m in members},
                                note="kept all; per-image weight 1/cluster_size")

    if _has_class_conflict(members, anns):
        return ResolutionResult(kept=members, removed=[], strategy=strategy,
                                review=True,
                                note="conflicting class annotations — kept all, flagged for review")

    keeper = _pick_keeper(members, anns, strategy, embeddings)
    return ResolutionResult(kept=[keeper], removed=[m for m in members if m != keeper],
                            strategy=strategy)


def _pick_keeper(members, anns, strategy, embeddings) -> str:
    if strategy == "keep_most_annotated":
        # Sort by (num_boxes, num_classes); stable tiebreak on path for determinism.
        def score(m):
            a = anns.get(m)
            return (a.num_boxes, a.num_classes) if a else (0, 0)
        return max(members, key=lambda m: (score(m), _neg_path(m)))

    if strategy == "keep_highest_res":
        def area(m):
            a = anns.get(m)
            return a.area if a else 0
        return max(members, key=lambda m: (area(m), _neg_path(m)))

    if strategy == "keep_central":
        if not embeddings or any(m not in embeddings for m in members):
            # No embeddings (Stage 3 skipped): fall back rather than fail the run.
            return _pick_keeper(members, anns, "keep_most_annotated", None)
        mat = np.stack([embeddings[m] for m in members]).astype(np.float32)
        centroid = mat.mean(axis=0)
        sims = mat @ centroid
        return members[int(np.argmax(sims))]

    raise ValueError(f"Unknown keep_strategy '{strategy}'")


def _neg_path(m: str):
    """Make lexicographically-smaller paths win ties deterministically when used
    inside a max()."""
    return tuple(-ord(c) for c in m)
