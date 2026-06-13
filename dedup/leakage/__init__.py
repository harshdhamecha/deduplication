"""Cross-split leakage check (filled in Step 5) — the headline resume metric.

After dedup but before splitting, detect any val/test image that has a
training-set neighbour above the similarity threshold and report it:
"X% of test images have a near-duplicate in train." Train/test contamination
silently inflates reported mAP; quantifying it is the project's payload.

If source metadata exists (video id, capture session, URL), a hard partition
key guarantees same-scene frames never land on opposite sides of the split.
"""
