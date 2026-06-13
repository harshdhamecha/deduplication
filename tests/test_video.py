"""Tests for the optional video-frame subsampling pre-filter."""

from dedup.io.video import subsample_by_partition


def test_keeps_every_kth_frame_within_partition():
    items = [f"vid1/frame_{i:03d}.jpg" for i in range(6)]
    partition_of = {it: "vid1" for it in items}
    kept = subsample_by_partition(items, partition_of, k=3)
    assert kept == ["vid1/frame_000.jpg", "vid1/frame_003.jpg"]


def test_k_le_1_is_noop():
    items = ["a.jpg", "b.jpg"]
    assert subsample_by_partition(items, {}, k=1) == items


def test_unpartitioned_items_pass_through():
    items = ["vid1/f0.jpg", "vid1/f1.jpg", "vid1/f2.jpg", "loose.jpg"]
    partition_of = {"vid1/f0.jpg": "vid1", "vid1/f1.jpg": "vid1", "vid1/f2.jpg": "vid1"}
    kept = subsample_by_partition(items, partition_of, k=2)
    assert "loose.jpg" in kept                       # no partition -> always kept
    assert "vid1/f0.jpg" in kept and "vid1/f2.jpg" in kept
    assert "vid1/f1.jpg" not in kept


def test_multiple_partitions_independent():
    items = ["a/0.jpg", "a/1.jpg", "b/0.jpg", "b/1.jpg"]
    partition_of = {"a/0.jpg": "a", "a/1.jpg": "a", "b/0.jpg": "b", "b/1.jpg": "b"}
    kept = subsample_by_partition(items, partition_of, k=2)
    assert set(kept) == {"a/0.jpg", "b/0.jpg"}       # frame 0 of each video
