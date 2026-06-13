"""Stage 4 cluster resolution (filled in Step 5).

Given a duplicate cluster, decide which image(s) to keep — annotation-aware,
because for detection the "best" duplicate is the one with the richest
supervision, not an arbitrary survivor.

Planned keep-strategies (selectable):
  keep_most_annotated  (default) most boxes / most class diversity.
  keep_highest_res     largest pixel area.
  keep_central         embedding closest to the cluster centroid.
  weighted_sample      keep all, emit per-image sampling weights (1/cluster_size).

When duplicates carry CONFLICTING annotations for the same scene, flag for
review rather than auto-resolving — silently picking one would corrupt labels.
"""
