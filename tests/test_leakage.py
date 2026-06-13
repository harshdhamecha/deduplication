"""Tests for the cross-split leakage check and split inference."""

from dedup.leakage.check import infer_split_from_path, leakage_report


def test_leaked_test_image_detected_via_pair_with_train():
    items = ["train/a.jpg", "train/b.jpg", "val/x.jpg", "val/y.jpg"]
    split_of = {"train/a.jpg": "train", "train/b.jpg": "train",
                "val/x.jpg": "val", "val/y.jpg": "val"}
    # val/x is a near-duplicate of train/a -> leaked; val/y is clean.
    pairs = [("train/a.jpg", "val/x.jpg")]
    rep = leakage_report(items, pairs, split_of, "train", ["val"])
    v = rep["by_split"]["val"]
    assert v["eval_images"] == 2
    assert v["leaked_images"] == 1
    assert v["leaked_fraction"] == 0.5
    assert "val/x.jpg" in v["examples"]


def test_no_leakage_when_pairs_within_same_split():
    items = ["train/a.jpg", "train/b.jpg", "val/x.jpg"]
    split_of = {"train/a.jpg": "train", "train/b.jpg": "train", "val/x.jpg": "val"}
    pairs = [("train/a.jpg", "train/b.jpg")]   # both train -> not leakage
    rep = leakage_report(items, pairs, split_of, "train", ["val"])
    assert rep["by_split"]["val"]["leaked_images"] == 0


def test_partition_spanning_violation_reported():
    items = ["train/a.jpg", "val/b.jpg"]
    split_of = {"train/a.jpg": "train", "val/b.jpg": "val"}
    # Same source video on both sides of the split -> spanning violation.
    partition_of = {"train/a.jpg": "vid7", "val/b.jpg": "vid7"}
    rep = leakage_report(items, [], split_of, "train", ["val"], partition_of)
    assert len(rep["partition_violations"]) == 1
    assert rep["partition_violations"][0]["partition"] == "vid7"


def test_infer_split_from_path():
    assert infer_split_from_path("/data/images/train2017/x.jpg", ["train", "val"]) == "train"
    assert infer_split_from_path("/data/images/val2017/x.jpg", ["train", "val"]) == "val"
    assert infer_split_from_path("/data/images/misc/x.jpg", ["train", "val"]) is None
