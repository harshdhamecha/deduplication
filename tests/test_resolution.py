"""Tests for annotation-aware cluster resolution and the conflict guard."""

import numpy as np

from dedup.io.annotations import ImageAnnotations
from dedup.resolution.strategies import resolve_cluster


def _ann(path, boxes=0, classes=(), w=0, h=0):
    return ImageAnnotations(image_id=path, file_name=path, width=w, height=h,
                            num_boxes=boxes, class_ids=set(classes))


def test_keep_most_annotated_picks_richest_supervision():
    members = ["a.jpg", "b.jpg", "c.jpg"]
    anns = {
        "a.jpg": _ann("a.jpg", boxes=1, classes=[1]),
        "b.jpg": _ann("b.jpg", boxes=5, classes=[1, 2, 3]),   # most boxes -> kept
        "c.jpg": _ann("c.jpg", boxes=2, classes=[1]),
    }
    res = resolve_cluster(members, anns, "keep_most_annotated")
    assert res.kept == ["b.jpg"]
    assert set(res.removed) == {"a.jpg", "c.jpg"}


def test_keep_highest_res_picks_largest_area():
    members = ["s.jpg", "big.jpg"]
    anns = {"s.jpg": _ann("s.jpg", boxes=1, classes=[1], w=100, h=100),
            "big.jpg": _ann("big.jpg", boxes=1, classes=[1], w=1000, h=1000)}
    res = resolve_cluster(members, anns, "keep_highest_res")
    assert res.kept == ["big.jpg"]


def test_weighted_sample_keeps_all_with_normalized_weights():
    members = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    res = resolve_cluster(members, {}, "weighted_sample")
    assert set(res.kept) == set(members)
    assert res.removed == []
    assert all(abs(w - 0.25) < 1e-6 for w in res.weights.values())


def test_conflicting_class_sets_flagged_for_review_not_resolved():
    # Same scene, disagreeing labels ({1,2} vs {3}) -> neither subset -> review.
    members = ["a.jpg", "b.jpg"]
    anns = {"a.jpg": _ann("a.jpg", boxes=2, classes=[1, 2]),
            "b.jpg": _ann("b.jpg", boxes=1, classes=[3])}
    res = resolve_cluster(members, anns, "keep_most_annotated")
    assert res.review is True
    assert set(res.kept) == set(members)      # kept all, deleted nothing
    assert res.removed == []


def test_subset_class_sets_are_not_a_conflict():
    # {1} is a subset of {1,2}: a normal superset relation, resolvable.
    members = ["a.jpg", "b.jpg"]
    anns = {"a.jpg": _ann("a.jpg", boxes=1, classes=[1]),
            "b.jpg": _ann("b.jpg", boxes=3, classes=[1, 2])}
    res = resolve_cluster(members, anns, "keep_most_annotated")
    assert res.review is False
    assert res.kept == ["b.jpg"]


def test_keep_central_uses_embedding_centroid():
    members = ["a.jpg", "b.jpg", "c.jpg"]
    # a,b near each other; c is an outlier. Centroid sits near a/b, so the kept
    # one should be a or b, not the outlier c.
    emb = {"a.jpg": np.array([1.0, 0.0]), "b.jpg": np.array([0.9, 0.1]),
           "c.jpg": np.array([-1.0, 0.0])}
    res = resolve_cluster(members, {}, "keep_central", embeddings=emb)
    assert res.kept[0] in {"a.jpg", "b.jpg"}
    assert "c.jpg" in res.removed
